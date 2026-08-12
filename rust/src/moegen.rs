// SPDX-License-Identifier: Apache-2.0
//
// The block that eleven MoE families share.
//
// WHAT THIS COVERS
//     llama/mixtral, qwen2moe, qwen3moe, glm4moe, olmoe, granitemoe, phimoe, dbrx,
//     ernie4_5-moe, bailingmoe, hunyuan-moe -- everything in `moearch::ARCHES` whose
//     `uses_common_block()` is true.
//
//     They are one architecture with flags:
//
//         rms_norm -> GQA (+ optional per-head q/k norm, rope) -> residual
//         rms_norm -> MoE FFN (+ optional shared expert)       -> residual
//
//     The exotic ones are excluded on purpose. `qwen35moe` alternates linear and full
//     attention and fuses its output gate into the query projection; `deepseek_v4`
//     projects through a latent bottleneck. Those are different blocks, not flags, and
//     pretending otherwise is how a model runs and is quietly wrong.
//
// WHAT IS REUSED RATHER THAN REWRITTEN
//     Almost everything. `gqa::gqa_last` scores one query against the prefix,
//     `qwen35run::route` picks and renormalises the experts, `qwen35run::expert_fwd` runs
//     one expert straight off its cache slot, and the streaming cache addresses stacked
//     experts identically for every family. What is new here is the assembly, not the
//     arithmetic -- which is also why the arithmetic keeps whatever verification it
//     already had.

use crate::gguf::{Meta, Value};
use crate::moearch::Arch;
use crate::ops::{self, Acc, W};
use crate::qwen35::weight_of;
use crate::st::{Dtype, St};

/// Dimensions common to every family in the table, read under that family's own prefix.
#[derive(Clone, Debug)]
pub struct Cfg {
    pub arch: &'static Arch,
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
}

impl Cfg {
    pub fn from_meta(m: &Meta) -> Result<Cfg, String> {
        let name = m
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or("no general.architecture")?;
        let arch = crate::moearch::find(name)
            .ok_or_else(|| format!("{name:?} is not a known MoE family"))?;
        if !arch.uses_common_block() {
            return Err(format!(
                "{name:?} needs its own block, not the common one -- see moearch.rs"
            ));
        }
        // Every key is required. A defaulted head count or rope base yields a model that
        // loads, runs, and is wrong, which is the failure this codebase keeps meeting.
        let u = |k: &str| -> Result<usize, String> {
            m.get(&format!("{name}.{k}"))
                .and_then(Value::as_u)
                .map(|v| v as usize)
                .ok_or_else(|| format!("{name}.{k} is missing from the gguf metadata"))
        };
        let f = |k: &str| -> Result<f32, String> {
            m.get(&format!("{name}.{k}"))
                .and_then(Value::as_f)
                .map(|v| v as f32)
                .ok_or_else(|| format!("{name}.{k} is missing from the gguf metadata"))
        };
        let hidden = u("embedding_length")?;
        let n_heads = u("attention.head_count")?;
        // Most of these families do not store a head dim; it is hidden / n_heads unless
        // the checkpoint says otherwise (Qwen3 and GLM-4 both do say otherwise).
        let head_dim = u("attention.key_length").unwrap_or(hidden / n_heads.max(1));
        Ok(Cfg {
            arch,
            n_layers: u("block_count")?,
            hidden,
            vocab: 0,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_heads,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim,
            // Partial rope is the exception; absent the key, the whole head rotates.
            n_rot: u("rope.dimension_count").unwrap_or(head_dim),
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            n_experts: u("expert_count")?,
            topk: u("expert_used_count")?,
            moe_inter: u("expert_feed_forward_length")?,
            shared_inter: u("expert_shared_feed_forward_length").unwrap_or(0),
        })
    }
}

struct QW {
    b: Vec<u8>,
    d: Dtype,
}

impl QW {
    fn load(st: &St, name: &str) -> Result<QW, String> {
        let t = st.find(name).ok_or_else(|| format!("missing {name}"))?;
        let mut b = vec![0u8; t.nbytes as usize];
        st.read(t, &mut b);
        if weight_of(&b, t.dtype).is_none() {
            return Err(format!("{name} is {}, which has no matmul kernel", t.dtype.name()));
        }
        Ok(QW { b, d: t.dtype })
    }
    fn w(&self) -> W<'_> {
        weight_of(&self.b, self.d).expect("validated on load")
    }
    fn bytes(&self) -> usize {
        self.b.len()
    }
}

fn f32s(st: &St, name: &str) -> Result<Vec<f32>, String> {
    let t = st.find(name).ok_or_else(|| format!("missing {name}"))?;
    let mut v = vec![0f32; t.numel() as usize];
    st.read_f32(t, &mut v);
    Ok(v)
}

/// One block's non-expert weights.
pub struct Block {
    attn_norm: Vec<f32>,
    wq: QW,
    wk: QW,
    wv: QW,
    wo: QW,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    gate_inp: Vec<f32>,
    shared: Option<(QW, QW, QW, Option<Vec<f32>>)>,
}

/// Everything resident: the blocks plus the IO path. Routed experts stream.
pub struct Trunk {
    pub cfg: Cfg,
    blocks: Vec<Block>,
    pub io: crate::qwen35::Io,
    pub rope: crate::gqa::Rope,
    pub bytes: usize,
    pub expert_dt: Vec<[Dtype; 3]>,
}

impl Trunk {
    pub fn load(st: &St, cfg: &Cfg, max_ctx: usize) -> Result<Trunk, String> {
        let vt = st.find("token_embd.weight").ok_or("missing token_embd.weight")?;
        let vocab = (vt.numel() as usize) / cfg.hidden;
        let mut c = cfg.clone();
        c.vocab = vocab;
        let io = crate::qwen35::Io::load(st, c.hidden, vocab, c.eps)?;
        let mut bytes = io.bytes();
        let mut blocks = Vec::with_capacity(c.n_layers);
        let mut expert_dt = Vec::with_capacity(c.n_layers);

        for l in 0..c.n_layers {
            let p = |n: &str| format!("blk.{l}.{n}");
            let (q_norm, k_norm) = if c.arch.qk_norm {
                (Some(f32s(st, &p("attn_q_norm.weight"))?), Some(f32s(st, &p("attn_k_norm.weight"))?))
            } else {
                (None, None)
            };
            // A shared expert is optional per family, and its gate is optional even then:
            // Qwen2 gates it with a learned scalar, others simply add it.
            let shared = if c.arch.shared_expert {
                Some((
                    QW::load(st, &p("ffn_gate_shexp.weight"))?,
                    QW::load(st, &p("ffn_up_shexp.weight"))?,
                    QW::load(st, &p("ffn_down_shexp.weight"))?,
                    f32s(st, &p("ffn_gate_inp_shexp.weight")).ok(),
                ))
            } else {
                None
            };
            let b = Block {
                attn_norm: f32s(st, &p("attn_norm.weight"))?,
                wq: QW::load(st, &p("attn_q.weight"))?,
                wk: QW::load(st, &p("attn_k.weight"))?,
                wv: QW::load(st, &p("attn_v.weight"))?,
                wo: QW::load(st, &p("attn_output.weight"))?,
                q_norm,
                k_norm,
                ffn_norm: f32s(st, &p(&format!("{}.weight", c.arch.ffn_norm)))?,
                gate_inp: f32s(st, &p("ffn_gate_inp.weight"))?,
                shared,
            };
            bytes += b.wq.bytes() + b.wk.bytes() + b.wv.bytes() + b.wo.bytes()
                + b.gate_inp.len() * 4
                + b.shared.as_ref().map_or(0, |(g, u, d, _)| g.bytes() + u.bytes() + d.bytes());
            blocks.push(b);

            let dt = |n: &str| -> Result<Dtype, String> {
                st.find(&format!("blk.{l}.{n}")).map(|t| t.dtype)
                    .ok_or_else(|| format!("missing blk.{l}.{n}"))
            };
            // Per LAYER, never read once from block 0: k-quant builds mix formats across
            // layers by importance, and a Q4_K buffer handed to the Q6_K kernel decodes
            // noise where the two happen to be the same width.
            expert_dt.push([
                dt("ffn_gate_exps.weight")?,
                dt("ffn_up_exps.weight")?,
                dt("ffn_down_exps.weight")?,
            ]);
        }
        let rope = crate::gqa::rope(c.head_dim, c.n_rot, max_ctx.max(1), c.rope_base,
                                    crate::gqa::Pairing::Halves);
        Ok(Trunk { cfg: c, blocks, io, rope, bytes, expert_dt })
    }
}

/// Per-layer KV. Every block has one -- unlike the hybrid families, nothing here is
/// recurrent, so state grows with context in all layers.
#[derive(Clone, Default)]
pub struct LayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

pub struct Session {
    pub kv: Vec<LayerKv>,
    pub pos: usize,
}

impl Session {
    pub fn new(t: &Trunk) -> Session {
        Session { kv: (0..t.cfg.n_layers).map(|_| LayerKv::default()).collect(), pos: 0 }
    }
}

/// One token through the whole stack.
pub fn step(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    id: u32,
    logits: &mut [f32],
) -> Result<(), String> {
    let c = &t.cfg;
    let (hid, hd) = (c.hidden, c.head_dim);
    let (nh, nkv) = (c.n_heads, c.n_kv_heads);
    let mut x = vec![0f32; hid];
    t.io.embed_row(id, &mut x)?;
    let mut h = vec![0f32; hid];
    let mut tmp = vec![0f32; hd];

    for l in 0..c.n_layers {
        // Tell the cache where it is in the layer sweep, so eviction can use Belady's rule
        // on the layer axis instead of LRU -- which is the WORST policy on a cyclic scan.
        cache.at_layer(l, c.n_layers);
        let b = &t.blocks[l];

        ops::rmsnorm(&mut h, &x, &b.attn_norm, hid, c.eps);
        let mut q = vec![0f32; nh * hd];
        let mut k = vec![0f32; nkv * hd];
        let mut v = vec![0f32; nkv * hd];
        ops::mmw(&mut q, &h, b.wq.w(), hid, nh * hd);
        ops::mmw(&mut k, &h, b.wk.w(), hid, nkv * hd);
        ops::mmw(&mut v, &h, b.wv.w(), hid, nkv * hd);

        // Per-head norm BEFORE rope. After it, the rotation would be applied to
        // unnormalised vectors and then partly normalised away.
        if let (Some(qn), Some(kn)) = (&b.q_norm, &b.k_norm) {
            for i in 0..nh {
                ops::rmsnorm(&mut tmp, &q[i * hd..][..hd], qn, hd, c.eps);
                q[i * hd..][..hd].copy_from_slice(&tmp);
            }
            for i in 0..nkv {
                ops::rmsnorm(&mut tmp, &k[i * hd..][..hd], kn, hd, c.eps);
                k[i * hd..][..hd].copy_from_slice(&tmp);
            }
        }
        for i in 0..nh {
            crate::gqa::apply_rope(&mut q[i * hd..][..hd], &t.rope, s.pos, false);
        }
        for i in 0..nkv {
            crate::gqa::apply_rope(&mut k[i * hd..][..hd], &t.rope, s.pos, false);
        }
        s.kv[l].k.extend_from_slice(&k);
        s.kv[l].v.extend_from_slice(&v);
        s.kv[l].len += 1;

        let d = crate::gqa::GqaDims {
            n_heads: nh, n_kv_heads: nkv, head_dim: hd, eps: c.eps, acc: Acc::F64, qk_norm: false,
        };
        let mut ctx = vec![0f32; nh * hd];
        crate::gqa::gqa_last(&mut ctx, &q, &s.kv[l].k, &s.kv[l].v, &d, s.kv[l].len, None, None);
        let mut attn = vec![0f32; hid];
        ops::mmw(&mut attn, &ctx, b.wo.w(), nh * hd, hid);
        for i in 0..hid {
            x[i] += attn[i];
        }

        // ---- MoE ----
        ops::rmsnorm(&mut h, &x, &b.ffn_norm, hid, c.eps);
        let mut rl = vec![0f32; c.n_experts];
        ops::mmw(&mut rl, &h, W::F32(&b.gate_inp), hid, c.n_experts);
        let sel = crate::qwen35run::route(&rl, c.topk);
        let ids: Vec<usize> = sel.iter().map(|(e, _)| *e).collect();
        cache.prefetch_many(st, l, &ids, crate::cache::gguf_expert_src);

        let mut out = vec![0f32; hid];
        let inter = c.moe_inter;
        let mut gu = vec![0f32; 2 * inter];
        let mut act = vec![0f32; inter];
        for (e, wt) in &sel {
            let slot = cache
                .get(st, l, *e, &crate::cache::gguf_expert_src(l, *e))
                .ok_or_else(|| format!("layer {l} expert {e} could not be cached"))?;
            crate::qwen35run::expert_fwd(
                &mut out, &h, &cache.expert(slot), &t.expert_dt[l], hid, inter, *wt,
                &mut gu, &mut act,
            )?;
        }
        if let Some((sg, su, sd, sgate)) = &b.shared {
            let si = c.shared_inter;
            let mut sgu = vec![0f32; 2 * si];
            let mut sact = vec![0f32; si];
            ops::mmw(&mut sgu[..si], &h, sg.w(), hid, si);
            ops::mmw(&mut sgu[si..], &h, su.w(), hid, si);
            ops::glu(&mut sact, &sgu, si, crate::qwen35run::SWIGLU);
            // The gate is optional: Qwen2 scales the shared expert by a learned sigmoid,
            // others add it unscaled. Applying a gate that is not there would silently
            // halve the branch.
            if let Some(g) = sgate {
                let mut dot = 0.0f64;
                for (a, b) in g.iter().zip(h.iter()) {
                    dot += *a as f64 * *b as f64;
                }
                let g = crate::libm::sigmoidf(dot as f32);
                for vv in sact.iter_mut() {
                    *vv *= g;
                }
            }
            let mut sh = vec![0f32; hid];
            ops::mmw(&mut sh, &sact, sd.w(), si, hid);
            for i in 0..hid {
                out[i] += sh[i];
            }
        }
        for i in 0..hid {
            x[i] += out[i];
        }
    }
    s.pos += 1;
    if std::env::var_os("MOE_SUMS").is_some() {
        // Compared against llama-eval-callback's `result_norm`, which is the last tensor
        // its graph emits when the vocabulary head is tied to the embedding.
        let mut n = vec![0f32; hid];
        ops::rmsnorm(&mut n, &x, t.io.norm(), hid, c.eps);
        eprintln!("  result_norm {:.6}", n.iter().map(|v| *v as f64).sum::<f64>());
    }
    t.io.logits(&x, logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(arch: &str, extra: &[(&str, u64)]) -> Meta {
        let mut m = Meta::new();
        m.insert("general.architecture".into(), Value::Str(arch.into()));
        for (k, v) in [
            ("block_count", 4u64), ("embedding_length", 256),
            ("attention.head_count", 4), ("attention.head_count_kv", 2),
            ("expert_count", 4), ("expert_used_count", 2),
            ("expert_feed_forward_length", 32),
        ] {
            m.insert(format!("{arch}.{k}"), Value::U(v));
        }
        for (k, v) in extra {
            m.insert(format!("{arch}.{k}"), Value::U(*v));
        }
        m.insert(format!("{arch}.attention.layer_norm_rms_epsilon"), Value::F(1e-6));
        m
    }

    /// The families the table says share this block must all parse through it.
    #[test]
    fn every_common_family_reads_its_own_hparams() {
        for a in crate::moearch::ARCHES.iter().filter(|a| a.uses_common_block()) {
            let c = Cfg::from_meta(&meta(a.name, &[])).unwrap_or_else(|e| {
                panic!("{} should parse through the common block: {e}", a.name)
            });
            assert_eq!(c.arch.name, a.name);
            assert_eq!((c.n_layers, c.hidden, c.n_heads), (4, 256, 4));
        }
    }

    /// The exotic stacks must be REFUSED here, not silently run through a block that does
    /// not describe them. Both would produce fluent, wrong output.
    #[test]
    fn the_exotic_stacks_are_refused_by_the_common_block() {
        for name in ["qwen35moe", "deepseek_v4"] {
            let e = Cfg::from_meta(&meta(name, &[])).unwrap_err();
            assert!(e.contains("its own block"), "{name}: {e}");
        }
        assert!(Cfg::from_meta(&meta("mamba", &[])).unwrap_err().contains("not a known"));
    }

    /// Head dim is hidden/n_heads unless the checkpoint says otherwise -- Qwen3 and GLM-4
    /// both say otherwise, and assuming the division there gives a wrong-shaped read.
    #[test]
    fn head_dim_defaults_to_the_division_but_the_checkpoint_wins() {
        let c = Cfg::from_meta(&meta("llama", &[])).unwrap();
        assert_eq!(c.head_dim, 64, "256 / 4");
        let c = Cfg::from_meta(&meta("qwen3moe", &[("attention.key_length", 128)])).unwrap();
        assert_eq!(c.head_dim, 128, "the checkpoint's value must win");
        // And rope covers the whole head unless a partial width is declared.
        assert_eq!(c.n_rot, 128);
        let c = Cfg::from_meta(&meta("qwen3moe", &[("rope.dimension_count", 32)])).unwrap();
        assert_eq!(c.n_rot, 32);
    }

    /// A missing REQUIRED key must be an error. Defaulting a head count produces a model
    /// that loads and is wrong.
    #[test]
    fn a_missing_required_key_is_refused() {
        let mut m = meta("llama", &[]);
        m.remove("llama.attention.head_count_kv");
        assert!(Cfg::from_meta(&m).unwrap_err().contains("head_count_kv"));
    }

    /// Families without a shared expert must not be given one, and vice versa -- the
    /// loader asks for `_shexp` tensors only when the table says they exist.
    #[test]
    fn the_shared_expert_follows_the_family_table() {
        assert!(!Cfg::from_meta(&meta("qwen3moe", &[])).unwrap().arch.shared_expert);
        assert!(Cfg::from_meta(&meta("qwen2moe", &[])).unwrap().arch.shared_expert);
        assert!(!Cfg::from_meta(&meta("llama", &[])).unwrap().arch.shared_expert);
    }
}
