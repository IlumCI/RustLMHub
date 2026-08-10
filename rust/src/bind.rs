// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::cfg::Cfg;
use crate::ops::W;
use crate::st::{bf16_to_f32, Dtype, St};

const PRE: &str = "language_model.model.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wdt {
    F32,
    Bf16,
    I8,
}

struct Req {
    name: String,
    want: i64,
    take: i64,
    narrow: bool,
}

struct Plan {
    r: Vec<Req>,
    narrow_ok: bool,
}

impl Plan {
    fn new(narrow_ok: bool) -> Plan {
        Plan { r: Vec::new(), narrow_ok }
    }
    fn w(&mut self, want: i64, take: i64, name: String) {
        self.r.push(Req { name, want, take: if take < 0 { want } else { take }, narrow: false });
    }
    fn n(&mut self, want: i64, name: String) {
        let narrow = self.narrow_ok;
        self.r.push(Req { name, want, take: want, narrow });
    }
}

fn align8(x: usize) -> usize {
    (x + 7) & !7
}

fn plan_layer(p: &mut Plan, c: &Cfg, l: i32) {
    let (h, hd) = (c.hidden as i64, c.kda_head_dim as i64);
    let pw = c.kda_heads as i64 * hd;
    let q = |s: &str| format!("{PRE}layers.{l}.{s}");

    for (nm, d) in [
        ("input_layernorm.weight", 0),
        ("post_attention_layernorm.weight", 0),
        ("self_attention_res_norm.weight", 0),
        ("self_attention_res_proj.weight", 0),
        ("mlp_res_norm.weight", 0),
        ("mlp_res_proj.weight", 0),
    ] {
        let _ = d;
        p.w(h, -1, q(nm));
    }

    if c.is_mla(l) {
        let qh = (c.qk_nope + c.qk_rope) as i64;
        p.n(c.q_lora as i64 * h, q("self_attn.q_a_proj.weight"));
        p.w(c.q_lora as i64, -1, q("self_attn.q_a_layernorm.weight"));
        p.n(c.n_heads as i64 * qh * c.q_lora as i64, q("self_attn.q_b_proj.weight"));
        p.n((c.kv_lora + c.qk_rope) as i64 * h, q("self_attn.kv_a_proj_with_mqa.weight"));
        p.w(c.kv_lora as i64, -1, q("self_attn.kv_a_layernorm.weight"));
        p.n(
            c.n_heads as i64 * (c.qk_nope + c.v_head) as i64 * c.kv_lora as i64,
            q("self_attn.kv_b_proj.weight"),
        );
        p.n(h * c.n_heads as i64 * c.v_head as i64, q("self_attn.o_proj.weight"));
        if c.mla_out_gate {
            p.n(c.n_heads as i64 * c.v_head as i64 * h, q("self_attn.g_proj.weight"));
        }
    } else {
        for nm in ["q_proj", "k_proj", "v_proj", "g_proj"] {
            p.n(pw * h, q(&format!("self_attn.{nm}.weight")));
        }
        p.n(h * pw, q("self_attn.o_proj.weight"));
        for nm in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            p.w(pw * c.conv_k as i64, -1, q(&format!("self_attn.{nm}.weight")));
        }
        p.n(hd * h, q("self_attn.f_a_proj.weight"));
        p.n(pw * hd, q("self_attn.f_b_proj.weight"));
        p.n(c.kda_heads as i64 * h, q("self_attn.b_proj.weight"));
        p.w(hd, c.kda_heads as i64, q("self_attn.A_log"));
        p.w(pw, -1, q("self_attn.dt_bias"));
        p.w(hd, -1, q("self_attn.o_norm.weight"));
    }

    if c.is_dense(l) {
        p.n(c.dense_inter as i64 * h, q("mlp.gate_proj.weight"));
        p.n(c.dense_inter as i64 * h, q("mlp.up_proj.weight"));
        p.n(h * c.dense_inter as i64, q("mlp.down_proj.weight"));
    } else {
        let si = c.moe_inter as i64 * c.n_shared as i64;
        p.w(c.n_experts as i64 * h, -1, q("block_sparse_moe.gate.weight"));
        p.w(c.n_experts as i64, -1, q("block_sparse_moe.gate.e_score_correction_bias"));
        p.n(c.latent as i64 * h, q("block_sparse_moe.routed_expert_down_proj.weight"));
        p.n(h * c.latent as i64, q("block_sparse_moe.routed_expert_up_proj.weight"));
        p.w(c.latent as i64, -1, q("block_sparse_moe.routed_expert_norm.weight"));
        p.n(si * h, q("block_sparse_moe.shared_experts.gate_proj.weight"));
        p.n(si * h, q("block_sparse_moe.shared_experts.up_proj.weight"));
        p.n(h * si, q("block_sparse_moe.shared_experts.down_proj.weight"));
    }
}

fn plan_model(p: &mut Plan, c: &Cfg, want_lm_head: bool) {
    let h = c.hidden as i64;
    p.n(c.vocab as i64 * h, format!("{PRE}embed_tokens.weight"));
    p.w(h, -1, format!("{PRE}norm.weight"));
    p.w(h, -1, format!("{PRE}output_attn_res_norm.weight"));
    p.w(h, -1, format!("{PRE}output_attn_res_proj.weight"));
    if want_lm_head {
        p.n(c.vocab as i64 * h, "language_model.lm_head.weight".into());
    }
}

#[derive(Clone, Copy)]
pub struct Slot {
    pub off: usize,
    pub take: i64,
    pub narrow: bool,
}

pub struct Bind {
    pub blob: Vec<u8>,
    pub at: HashMap<String, Slot>,
    pub wdt: Wdt,
    pub layer: i32,
}

fn resolve(p: &Plan, st: &St) -> Result<(usize, Vec<Slot>, usize), String> {
    let mut off = 0usize;
    let mut out = Vec::with_capacity(p.r.len());
    let mut demoted = 0usize;
    let mut bad = Vec::new();
    for q in &p.r {
        let Some(t) = st.find(&q.name) else {
            bad.push(format!("missing tensor {}", q.name));
            out.push(Slot { off: 0, take: 0, narrow: false });
            continue;
        };
        let have = t.numel();
        if q.want >= 0 && have != q.want {
            bad.push(format!("{} has {have} elements, engine expects {}", q.name, q.want));
        }
        if q.take > have {
            bad.push(format!("{}: asked for {} of {have} elements", q.name, q.take));
        }
        let mut narrow = q.narrow;
        if narrow && t.dtype != Dtype::Bf16 {
            narrow = false;
            demoted += 1;
        }
        if narrow && q.take != have {
            bad.push(format!("{}: a partial take of a narrow tensor is not implemented", q.name));
        }
        off = align8(off);
        out.push(Slot { off, take: q.take, narrow });
        off += q.take as usize * if narrow { 2 } else { 4 };
    }
    if !bad.is_empty() {
        return Err(format!("k3_bind:\n    {}", bad.join("\n    ")));
    }
    Ok((off, out, demoted))
}

fn load(p: &Plan, st: &St, slots: &[Slot], blob: &mut [u8]) -> Result<(), String> {
    for (q, s) in p.r.iter().zip(slots) {
        let t = st.find(&q.name).ok_or_else(|| format!("k3_bind: vanished {}", q.name))?;
        let have = t.numel();
        let dst = &mut blob[s.off..];
        if s.narrow {
            if st.read(t, &mut dst[..t.nbytes as usize]) != t.nbytes {
                return Err(format!("k3_bind: short read of {}", q.name));
            }
        } else {
            let n = s.take as usize;
            let mut tmp = vec![0f32; have as usize];
            if st.read_f32(t, &mut tmp) != have {
                return Err(format!("k3_bind: short read of {}", q.name));
            }
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tmp.as_ptr().cast::<u8>(), n * 4) };
            dst[..n * 4].copy_from_slice(bytes);
        }
    }
    Ok(())
}

fn build(p: &mut Plan, st: &St, fill: impl Fn(&mut Plan)) -> Result<Bind, String> {
    fill(p);
    let (mut need, mut slots, demoted) = resolve(p, st)?;
    let mut narrow_ok = p.narrow_ok;
    if demoted > 0 && narrow_ok {
        eprintln!("k3_bind: {demoted} large tensor(s) are not BF16; binding at fp32 instead");
        narrow_ok = false;
        *p = Plan::new(false);
        fill(p);
        let r = resolve(p, st)?;
        need = r.0;
        slots = r.1;
    }
    let mut blob = vec![0u8; need];
    load(p, st, &slots, &mut blob)?;
    let at = p.r.iter().zip(&slots).map(|(q, s)| (q.name.clone(), *s)).collect();
    Ok(Bind {
        blob,
        at,
        wdt: if narrow_ok { Wdt::Bf16 } else { Wdt::F32 },
        layer: -1,
    })
}

impl Bind {
    pub fn layer(st: &St, c: &Cfg, l: i32) -> Result<Bind, String> {
        let mut p = Plan::new(true);
        let mut b = build(&mut p, st, |p| plan_layer(p, c, l))?;
        b.layer = l;
        Ok(b)
    }

    pub fn model(st: &St, c: &Cfg, want_lm_head: bool) -> Result<Bind, String> {
        let mut p = Plan::new(true);
        build(&mut p, st, |p| plan_model(p, c, want_lm_head))
    }

    pub fn bytes(st: &St, c: &Cfg, l: i32) -> Result<usize, String> {
        let mut p = Plan::new(true);
        plan_layer(&mut p, c, l);
        resolve(&p, st).map(|r| r.0)
    }

    pub fn f32s(&self, name: &str) -> &[f32] {
        let s = self.at.get(name).unwrap_or_else(|| panic!("k3_bind: no slot for {name}"));
        assert!(!s.narrow, "k3_bind: {name} is stored narrow");
        unsafe {
            std::slice::from_raw_parts(
                self.blob[s.off..].as_ptr().cast::<f32>(),
                s.take as usize,
            )
        }
    }

    pub fn opt_f32s(&self, name: &str) -> Option<&[f32]> {
        self.at.contains_key(name).then(|| self.f32s(name))
    }

    pub fn mat(&self, name: &str) -> W<'_> {
        let s = self.at.get(name).unwrap_or_else(|| panic!("k3_bind: no slot for {name}"));
        let n = s.take as usize;
        match self.wdt {
            Wdt::Bf16 if s.narrow => W::Bf16(unsafe {
                std::slice::from_raw_parts(self.blob[s.off..].as_ptr().cast::<u16>(), n)
            }),
            _ => W::F32(unsafe {
                std::slice::from_raw_parts(self.blob[s.off..].as_ptr().cast::<f32>(), n)
            }),
        }
    }

    pub fn embed_row(&self, name: &str, row: i64, hidden: usize, dst: &mut [f32]) {
        let s = self.at[name];
        if s.narrow {
            let p = unsafe {
                std::slice::from_raw_parts(self.blob[s.off..].as_ptr().cast::<u16>(), s.take as usize)
            };
            let base = row as usize * hidden;
            for i in 0..hidden {
                dst[i] = bf16_to_f32(p[base + i]);
            }
        } else {
            let p = unsafe {
                std::slice::from_raw_parts(self.blob[s.off..].as_ptr().cast::<f32>(), s.take as usize)
            };
            dst[..hidden].copy_from_slice(&p[row as usize * hidden..][..hidden]);
        }
    }
}

pub fn widen_bytes(c: &Cfg) -> usize {
    let h = c.hidden as usize;
    let n = 6 * h
        + c.q_lora as usize
        + c.kv_lora as usize
        + c.latent as usize
        + c.n_experts as usize * h;
    n * 4 + 4096
}

#[derive(Clone, Copy)]
pub enum Loc {
    Run(usize, usize),
    Widen(usize, usize),
}

pub struct MemBind {
    pub at: HashMap<String, (Loc, i64)>,
    pub wdt: Wdt,
    pub widen_used: usize,
    pub layer: i32,
}

pub struct MemSrc<'a> {
    pub run: &'a [u8],
}

/// `find` reports where a tensor sits inside the packed run: (offset, nbytes, dtype).
pub fn layer_mem<F>(
    c: &Cfg,
    l: i32,
    find: F,
    run: &[u8],
    widen: &mut [u8],
) -> Result<MemBind, String>
where
    F: Fn(&str) -> Option<(i64, i64, Dtype)>,
{
    let mut p = Plan::new(true);
    plan_layer(&mut p, c, l);

    let mut at = HashMap::with_capacity(p.r.len());
    let mut w = 0usize;
    let mut narrowed_all = true;
    let mut i8_seen = false;

    for q in &p.r {
        let (off, nb, dt) = find(&q.name)
            .ok_or_else(|| format!("k3_bind_mem: {} not present in the packed run", q.name))?;

        if dt == Dtype::I8R {
            if q.narrow {
                at.insert(q.name.clone(), (Loc::Run(off as usize, nb as usize), q.take));
                i8_seen = true;
                continue;
            }
            let take = q.take;
            if (nb - take) % 4 != 0 {
                return Err(format!("k3_bind_mem: {} bad int8 layout", q.name));
            }
            let rows = (nb - take) / 4;
            if rows <= 0 || take % rows != 0 {
                return Err(format!("k3_bind_mem: {} bad int8 shape", q.name));
            }
            let cols = take / rows;
            w = align8(w);
            if w + take as usize * 4 > widen.len() {
                return Err(format!("k3_bind_mem: widen area too small at {}", q.name));
            }
            let rowb = 4 + cols as usize;
            for r in 0..rows as usize {
                let base = off as usize + r * rowb;
                let scale = f32::from_le_bytes(run[base..base + 4].try_into().unwrap());
                for k in 0..cols as usize {
                    let v = run[base + 4 + k] as i8 as f32 * scale;
                    let d = w + (r * cols as usize + k) * 4;
                    widen[d..d + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            at.insert(q.name.clone(), (Loc::Widen(w, take as usize * 4), take));
            w += take as usize * 4;
            continue;
        }

        let esz = match dt {
            Dtype::F32 => 4,
            Dtype::U8 => 1,
            _ => 2,
        };
        let have = nb / esz;
        if q.want >= 0 && have != q.want {
            return Err(format!(
                "k3_bind_mem: {} has {have} elements, engine expects {}",
                q.name, q.want
            ));
        }
        if q.take > have {
            return Err(format!("k3_bind_mem: {}: asked for {} of {have}", q.name, q.take));
        }

        if q.narrow {
            if dt == Dtype::Bf16 {
                at.insert(q.name.clone(), (Loc::Run(off as usize, nb as usize), q.take));
                continue;
            }
            narrowed_all = false;
        }
        if dt == Dtype::F32 {
            at.insert(q.name.clone(), (Loc::Run(off as usize, nb as usize), q.take));
            continue;
        }
        if dt != Dtype::Bf16 {
            return Err(format!("k3_bind_mem: {} has dtype {:?}, cannot widen", q.name, dt));
        }
        w = align8(w);
        if w + q.take as usize * 4 > widen.len() {
            return Err(format!("k3_bind_mem: widen area too small at {}", q.name));
        }
        for k in 0..q.take as usize {
            let h = u16::from_le_bytes(run[off as usize + k * 2..][..2].try_into().unwrap());
            let d = w + k * 4;
            widen[d..d + 4].copy_from_slice(&bf16_to_f32(h).to_le_bytes());
        }
        at.insert(q.name.clone(), (Loc::Widen(w, q.take as usize * 4), q.take));
        w += q.take as usize * 4;
    }

    if !narrowed_all && !i8_seen {
        return Err(format!("k3_bind_mem: layer {l} has a non-BF16 large tensor"));
    }
    Ok(MemBind {
        at,
        wdt: if i8_seen { Wdt::I8 } else { Wdt::Bf16 },
        widen_used: w,
        layer: l,
    })
}

impl MemBind {
    pub fn bytes<'a>(&self, name: &str, run: &'a [u8], widen: &'a [u8]) -> &'a [u8] {
        match self.at.get(name).unwrap_or_else(|| panic!("k3_bind_mem: no slot for {name}")).0 {
            Loc::Run(o, n) => &run[o..o + n],
            Loc::Widen(o, n) => &widen[o..o + n],
        }
    }

    pub fn f32s<'a>(&self, name: &str, run: &'a [u8], widen: &'a [u8]) -> &'a [f32] {
        let (_, take) = self.at[name];
        let b = self.bytes(name, run, widen);
        unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), take as usize) }
    }

    pub fn mat<'a>(&self, name: &str, run: &'a [u8], widen: &'a [u8]) -> W<'a> {
        let (_, take) = self.at[name];
        let b = self.bytes(name, run, widen);
        match self.wdt {
            Wdt::I8 => W::I8(b),
            _ => W::Bf16(unsafe {
                std::slice::from_raw_parts(b.as_ptr().cast::<u16>(), take as usize)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Cfg {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/ref_k3.json");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(p).expect("ref_k3.json")).expect("json");
        crate::cfg::load(&v["config"], p).expect("config")
    }

    #[test]
    fn a_log_is_taken_per_head_not_per_channel() {
        let c = cfg();
        let mut p = Plan::new(true);
        plan_layer(&mut p, &c, 0);
        let r = p.r.iter().find(|r| r.name.ends_with("A_log")).expect("A_log requested");
        assert_eq!(r.want, c.kda_head_dim as i64, "the checkpoint ships head_dim values");
        assert_eq!(r.take, c.kda_heads as i64, "but only the first n_heads are per-head gains");
        assert!(r.take < r.want, "taking all of them is the documented silent bug");
    }

    #[test]
    fn conv1d_is_requested_by_element_count_not_rank() {
        let c = cfg();
        let mut p = Plan::new(true);
        plan_layer(&mut p, &c, 0);
        let want = c.kda_heads as i64 * c.kda_head_dim as i64 * c.conv_k as i64;
        for nm in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            let r = p.r.iter().find(|r| r.name.contains(nm)).expect(nm);
            assert_eq!(r.want, want, "{nm} ships rank 3 [H*D][1][conv_k]; only numel matters");
            assert!(!r.narrow, "shortconv reads it elementwise, so it must be fp32");
        }
    }

    #[test]
    fn elementwise_tensors_are_never_left_narrow() {
        let c = cfg();
        let mut p = Plan::new(true);
        plan_layer(&mut p, &c, 0);
        for n in ["input_layernorm", "o_norm", "dt_bias", "A_log", "gate.weight", "res_norm"] {
            for r in p.r.iter().filter(|r| r.name.contains(n)) {
                assert!(!r.narrow, "{} is read elementwise and must widen", r.name);
            }
        }
    }

    #[test]
    fn a_dense_layer_asks_for_the_mlp_and_an_moe_layer_does_not() {
        let c = cfg();
        let mut d = Plan::new(true);
        plan_layer(&mut d, &c, 0);
        assert!(c.is_dense(0));
        assert!(d.r.iter().any(|r| r.name.contains("mlp.gate_proj")));
        assert!(!d.r.iter().any(|r| r.name.contains("block_sparse_moe")));

        let mut m = Plan::new(true);
        plan_layer(&mut m, &c, c.n_layers - 1);
        assert!(m.r.iter().any(|r| r.name.contains("block_sparse_moe.gate.weight")));
        assert!(!m.r.iter().any(|r| r.name.contains("mlp.gate_proj")));
    }

    #[test]
    fn narrow_off_forces_every_matrix_to_fp32() {
        let c = cfg();
        let mut p = Plan::new(false);
        plan_layer(&mut p, &c, 5);
        assert!(p.r.iter().all(|r| !r.narrow));
    }
}
