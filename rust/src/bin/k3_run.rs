// SPDX-License-Identifier: Apache-2.0
//
// usage: k3_run MODEL_DIR [--trunk DIR] [--tok DIR] [--prompt S | --ids a,b,c]
//               [--gen N] [--incremental] [--trunk-gb G] [--cache-gb G]
//               [--layers N] [--dump-logits FILE] [--out FILE]

use std::path::PathBuf;
use std::process::ExitCode;

use k3::bind::{self, Bind};
use k3::cache::{k3_expert_names, Cache};
use k3::cfg::{self, Cfg};
use k3::io::Trunk;
use k3::ops::{self, AttnResidual, Attn, KdaDims, LayerW, MlaDims, Mlp, MoeDims, Residual, W};
use k3::st::St;
use k3::tok_k3::TokK3;

struct Args {
    model: PathBuf,
    trunk: Option<PathBuf>,
    tok: Option<PathBuf>,
    prompt: Option<String>,
    ids: Option<Vec<u32>>,
    gen: usize,
    incremental: bool,
    trunk_gb: f64,
    cache_gb: f64,
    layers: Option<usize>,
    dump_logits: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: PathBuf::new(),
        trunk: None,
        tok: None,
        prompt: None,
        ids: None,
        gen: 16,
        incremental: false,
        trunk_gb: 4.0,
        cache_gb: 2.0,
        layers: None,
        dump_logits: None,
        out: None,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let need = |i: usize| -> Result<String, String> {
            v.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", v[i]))
        };
        let num = |s: String, w: &str| -> Result<f64, String> {
            s.parse().map_err(|_| format!("{w} wants a number"))
        };
        match v[i].as_str() {
            "--trunk" => { a.trunk = Some(need(i)?.into()); i += 2 }
            "--tok" => { a.tok = Some(need(i)?.into()); i += 2 }
            "--prompt" => { a.prompt = Some(need(i)?); i += 2 }
            "--ids" => {
                a.ids = Some(
                    need(i)?
                        .split(',')
                        .map(|s| s.trim().parse::<u32>().map_err(|_| "--ids wants integers".to_string()))
                        .collect::<Result<_, _>>()?,
                );
                i += 2
            }
            "--gen" => { a.gen = num(need(i)?, "--gen")? as usize; i += 2 }
            "--incremental" => { a.incremental = true; i += 1 }
            "--trunk-gb" => { a.trunk_gb = num(need(i)?, "--trunk-gb")?; i += 2 }
            "--cache-gb" => { a.cache_gb = num(need(i)?, "--cache-gb")?; i += 2 }
            "--layers" => { a.layers = Some(num(need(i)?, "--layers")? as usize); i += 2 }
            "--dump-logits" => { a.dump_logits = Some(need(i)?.into()); i += 2 }
            "--out" => { a.out = Some(need(i)?.into()); i += 2 }
            s if s.starts_with("--") => return Err(format!("unknown flag {s}")),
            s => { a.model = s.into(); i += 1 }
        }
    }
    if a.model.as_os_str().is_empty() {
        return Err("a model directory is required".into());
    }
    if a.prompt.is_none() && a.ids.is_none() {
        return Err("one of --prompt or --ids is required".into());
    }
    Ok(a)
}

/// Weights for one layer, either bound from the shards or pointed into a trunk slot.
enum Layer<'a> {
    Shard(&'a Bind),
    Mem(bind::MemBind),
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("k3_run: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let a = parse().map_err(|e| {
        format!(
            "{e}\nusage: k3_run MODEL_DIR [--trunk DIR] [--tok DIR] \
             [--prompt S | --ids a,b,c] [--gen N] [--incremental]"
        )
    })?;

    let c: Cfg = cfg::load_file(&a.model.join("config.json")).map_err(|e| e.to_string())?;
    println!("{}", c.summary(&a.model.join("config.json").display().to_string()));
    let n_layers = a.layers.unwrap_or(c.n_layers as usize).min(c.n_layers as usize);

    let st = St::open(&a.model).map_err(|e| e.to_string())?;
    println!("checkpoint: {} shards, {} tensors", st.nshard(), st.tensors.len());

    let tok = match a.tok.as_ref().or(Some(&a.model)) {
        Some(d) if a.prompt.is_some() => Some(TokK3::from_dir(d)?),
        _ => None,
    };
    let mut ids: Vec<u32> = match (&a.ids, &a.prompt) {
        (Some(v), _) => v.clone(),
        (None, Some(p)) => tok.as_ref().expect("tokenizer").encode(p),
        _ => unreachable!("parse() requires one of them"),
    };
    println!("prompt: {} tokens {:?}", ids.len(), &ids[..ids.len().min(12)]);

    let model = Bind::model(&st, &c, true)?;
    let mut trunk = match &a.trunk {
        Some(d) => Some(
            Trunk::open(d, bind::widen_bytes(&c), (a.trunk_gb * 1e9) as i64)
                .map_err(|e| e.to_string())?,
        ),
        None => None,
    };
    if let Some(t) = &trunk {
        println!("trunk: {} layers packed, {} pinned", t.n_layers, t.npin());
    }

    let slot = {
        let names = k3_expert_names(c.first_dense as usize, 0);
        k3::cache::check_expert(&st, &names, ops::MXFP4_GROUP as i64)?;
        k3::cache::slot_need(&k3::cache::locate(&st, &names).ok_or("cannot locate expert 0")?)
    };
    let mut cache = Cache::new((a.cache_gb * 1e9) as i64, slot, c.n_experts as usize, c.topk as usize)?;
    println!("expert cache: {} slots of {:.1} MB", cache.nslot(), slot as f64 / 1e6);

    let (e, vocab) = (c.hidden as usize, c.vocab as usize);
    let maxb = (c.n_layers / c.attn_res_block + 2) as usize;
    let kper = kda_state_len(&c);
    let mut kstate = vec![0f32; kper * n_layers];
    // The MLA KV cache, one pair per layer, sized for prompt + generation. Only the MLA
    // layers touch theirs; a KDA layer's stays empty.
    let cap = ids.len() + a.gen.max(1) + 1;
    let kvd = (c.qk_nope + c.v_head) as usize;
    let mut kvc: Vec<Vec<f32>> = (0..n_layers)
        .map(|l| if c.is_mla(l as i32) { vec![0f32; cap * c.n_heads as usize * kvd] } else { Vec::new() })
        .collect();
    let mut ropec: Vec<Vec<f32>> = (0..n_layers)
        .map(|l| if c.is_mla(l as i32) { vec![0f32; cap * c.qk_rope as usize] } else { Vec::new() })
        .collect();

    // Shard-bound layers when there is no packed trunk. Bound once, not per token: the
    // whole point of the trunk path is that this does not fit at K3's size.
    let resident: Vec<Option<Bind>> = if trunk.is_none() {
        (0..n_layers).map(|l| Bind::layer(&st, &c, l as i32).ok()).collect()
    } else {
        Vec::new()
    };

    let mut text = String::new();
    let mut logits = vec![0f32; vocab];
    let mut dumped: Vec<f32> = Vec::new();

    for step in 0..a.gen.max(1) {
        let t_len = if a.incremental && step > 0 { 1 } else { ids.len() };
        let base = ids.len() - t_len;
        let cached = if a.incremental { base } else { 0 };

        if !a.incremental {
            kstate.fill(0.0);
        }
        let mut h = vec![0f32; t_len * e];
        for t in 0..t_len {
            model.embed_row("language_model.model.embed_tokens.weight", ids[base + t] as i64, e, &mut h[t * e..]);
        }
        let mut res = AttnResidual::new(&h, t_len, e, maxb, c.rms_eps);

        for l in 0..n_layers {
            if let Some(tr) = trunk.as_mut() {
                tr.prefetch(l + 1);
                let run_len = tr.run_len(l);
                // Snapshot the layer's ~28 tensor descriptors before the slot is borrowed
                // mutably; it is nothing next to a 1.27 GB read.
                let map: std::collections::HashMap<String, (i64, i64, k3::st::Dtype)> = tr.lay[l]
                    .tensors
                    .iter()
                    .map(|(k, t)| (k.clone(), (t.off, t.nbytes, t.dtype)))
                    .collect();
                let slot = tr.layer(l).map_err(|x| x.to_string())?;
                let (run, widen) = slot.split_at_mut(run_len);
                let mb = bind::layer_mem(&c, l as i32, |n| map.get(n).copied(), run, widen)?;
                let (run, widen) = (&*run, &*widen);
                step_layer(&mut res, &Layer::Mem(mb), Some((run, widen)), &c, l, t_len,
                           &mut kstate[l * kper..][..kper], &mut kvc[l], &mut ropec[l],
                           &st, &mut cache, cached)?;
            } else {
                let b = resident[l].as_ref().ok_or_else(|| format!("layer {l} did not bind"))?;
                step_layer(&mut res, &Layer::Shard(b), None, &c, l, t_len,
                           &mut kstate[l * kper..][..kper], &mut kvc[l], &mut ropec[l],
                           &st, &mut cache, cached)?;
            }
        }

        let hfin = res
            .finish(
                model.f32s("language_model.model.output_attn_res_norm.weight"),
                model.f32s("language_model.model.output_attn_res_proj.weight"),
            )
            .to_vec();

        let last = (t_len - 1) * e;
        let mut nrm = vec![0f32; e];
        ops::rmsnorm(&mut nrm, &hfin[last..][..e], model.f32s("language_model.model.norm.weight"), e, c.rms_eps);
        ops::mmw(&mut logits, &nrm, model.mat("language_model.lm_head.weight"), e, vocab);
        if a.dump_logits.is_some() && step == 0 {
            dumped = logits.clone();
        }

        let next = logits
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or("empty logits")?;
        ids.push(next);
        if let Some(t) = &tok {
            let p = t.decode(&[next]);
            print!("{p}");
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
            text.push_str(&p);
            if next == t.eos {
                break;
            }
        }
    }
    println!();

    if let Some(p) = &a.dump_logits {
        let mut b = Vec::with_capacity(dumped.len() * 4);
        for v in &dumped {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(p, b).map_err(|e| e.to_string())?;
        println!("logits -> {}", p.display());
    }
    if let Some(p) = &a.out {
        let s = if text.is_empty() {
            ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
        } else {
            text.clone()
        };
        std::fs::write(p, s).map_err(|e| e.to_string())?;
    }
    cache.report("final");
    if let Some(t) = &trunk {
        t.report("final");
    }
    Ok(())
}

fn kda_state_len(c: &Cfg) -> usize {
    let (h, d) = (c.kda_heads as usize, c.kda_head_dim as usize);
    let p = h * d;
    h * d * d + 3 * p * (c.conv_k as usize - 1)
}

#[allow(clippy::too_many_arguments)]
fn step_layer(
    res: &mut AttnResidual,
    lay: &Layer<'_>,
    mem: Option<(&[u8], &[u8])>,
    c: &Cfg,
    l: usize,
    t_len: usize,
    state: &mut [f32],
    kvc: &mut [f32],
    ropec: &mut [f32],
    st: &St,
    cache: &mut Cache,
    cached: usize,
) -> Result<(), String> {
    let (e, hd) = (c.hidden as usize, c.kda_head_dim as usize);
    let pw = c.kda_heads as usize * hd;
    let li = l as i32;
    let q = |s: &str| format!("language_model.model.layers.{l}.{s}");

    let f = |n: &str| -> &[f32] {
        match (lay, mem) {
            (Layer::Shard(b), _) => b.f32s(n),
            (Layer::Mem(m), Some((r, w))) => m.f32s(n, r, w),
            _ => unreachable!("a mem bind needs its buffers"),
        }
    };
    let m = |n: &str| -> W<'_> {
        match (lay, mem) {
            (Layer::Shard(b), _) => b.mat(n),
            (Layer::Mem(mb), Some((r, w))) => mb.mat(n, r, w),
            _ => unreachable!("a mem bind needs its buffers"),
        }
    };

    res.begin_layer(
        l,
        c.attn_res_block as usize,
        &ops::fold_residual(f(&q("self_attention_res_norm.weight")), f(&q("self_attention_res_proj.weight"))),
        &ops::fold_residual(f(&q("mlp_res_norm.weight")), f(&q("mlp_res_proj.weight"))),
    );

    let (kw, mw);
    let (kd, md);
    let attn = if c.is_mla(li) {
        md = MlaDims {
            hidden: e,
            n_heads: c.n_heads as usize,
            q_lora: c.q_lora as usize,
            kv_lora: c.kv_lora as usize,
            qk_nope: c.qk_nope as usize,
            qk_rope: c.qk_rope as usize,
            v_head: c.v_head as usize,
            rms_eps: c.rms_eps,
        };
        mw = ops::MlaW {
            q_a: m(&q("self_attn.q_a_proj.weight")),
            q_b: m(&q("self_attn.q_b_proj.weight")),
            kv_a: m(&q("self_attn.kv_a_proj_with_mqa.weight")),
            kv_b: m(&q("self_attn.kv_b_proj.weight")),
            o: m(&q("self_attn.o_proj.weight")),
            g: c.mla_out_gate.then(|| m(&q("self_attn.g_proj.weight"))),
            q_a_norm: f(&q("self_attn.q_a_layernorm.weight")),
            kv_a_norm: f(&q("self_attn.kv_a_layernorm.weight")),
        };
        Attn::Mla(&mw, &md)
    } else {
        kd = KdaDims {
            hidden: e,
            heads: c.kda_heads as usize,
            head_dim: hd,
            conv_k: c.conv_k as usize,
            gate_lb: c.gate_lb,
            rms_eps: c.rms_eps,
        };
        kw = ops::KdaW {
            q: m(&q("self_attn.q_proj.weight")),
            k: m(&q("self_attn.k_proj.weight")),
            v: m(&q("self_attn.v_proj.weight")),
            q_conv: f(&q("self_attn.q_conv1d.weight")),
            k_conv: f(&q("self_attn.k_conv1d.weight")),
            v_conv: f(&q("self_attn.v_conv1d.weight")),
            f_a: m(&q("self_attn.f_a_proj.weight")),
            f_b: m(&q("self_attn.f_b_proj.weight")),
            a_log: f(&q("self_attn.A_log")),
            dt_bias: f(&q("self_attn.dt_bias")),
            b: m(&q("self_attn.b_proj.weight")),
            g: m(&q("self_attn.g_proj.weight")),
            o_norm: f(&q("self_attn.o_norm.weight")),
            o: m(&q("self_attn.o_proj.weight")),
        };
        let _ = pw;
        Attn::Kda(&kw, &kd)
    };

    // Attention first, so the MoE below sees the post-attention stream the router must
    // score. decoder_layer_inc drives both halves through the Residual trait.
    let dims = MoeDims {
        hidden: e,
        latent: c.latent as usize,
        moe_inter: c.moe_inter as usize,
        n_experts: c.n_experts as usize,
        topk: c.topk as usize,
        n_shared: c.n_shared as usize,
        routed_scale: c.routed_scale,
        renorm: c.moe_renorm,
        latent_norm: c.latent_norm,
        rms_eps: c.rms_eps,
        situ_b1: c.situ_b1,
        situ_b2: c.situ_b2,
    };

    if c.is_dense(li) {
        let w = LayerW {
            in_norm: f(&q("input_layernorm.weight")),
            post_norm: f(&q("post_attention_layernorm.weight")),
            attn_res_norm: f(&q("self_attention_res_norm.weight")),
            attn_res_proj: f(&q("self_attention_res_proj.weight")),
            mlp_res_norm: f(&q("mlp_res_norm.weight")),
            mlp_res_proj: f(&q("mlp_res_proj.weight")),
            attn,
            mlp: Mlp::Dense {
                gate: m(&q("mlp.gate_proj.weight")),
                up: m(&q("mlp.up_proj.weight")),
                down: m(&q("mlp.down_proj.weight")),
                inter: c.dense_inter as usize,
                b1: c.situ_b1,
                b2: c.situ_b2,
            },
        };
        let kv = (!kvc.is_empty()).then_some((&mut kvc[..], &mut ropec[..]));
        ops::decoder_layer_inc(res, &w, e, t_len, state, c.rms_eps, kv, cached);
        return Ok(());
    }

    // Streamed MoE. Attention runs through decoder_layer_inc with a dense-free MLP is not
    // expressible, so the two halves are driven here: pre(Attn)/post(Attn) then the MoE.
    res.pre(ops::Sub::Attn, &mut vec![0f32; t_len * e]);
    let modin = res.hidden().to_vec();
    let mut hin = vec![0f32; t_len * e];
    for t in 0..t_len {
        ops::rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], f(&q("input_layernorm.weight")), e, c.rms_eps);
    }
    let mut tmp = vec![0f32; t_len * e];
    match &attn {
        Attn::Kda(kw, kd) => ops::kda_layer(&mut tmp, &hin, kw, kd, t_len, state),
        Attn::Mla(mw, md) => {
            let kv = (!kvc.is_empty()).then_some((&mut kvc[..], &mut ropec[..]));
            ops::mla_cached(&mut tmp, &hin, mw, md, t_len, kv, cached)
        }
    }
    res.post(ops::Sub::Attn, &tmp, ());

    res.pre(ops::Sub::Mlp, &mut vec![0f32; t_len * e]);
    let modin = res.hidden().to_vec();
    for t in 0..t_len {
        ops::rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], f(&q("post_attention_layernorm.weight")), e, c.rms_eps);
    }

    let gate = f(&q("block_sparse_moe.gate.weight"));
    let bias = Some(f(&q("block_sparse_moe.gate.e_score_correction_bias")));
    let (idx, wt, uniq) = ops::route_chunk(&hin, gate, bias, &dims, 0, t_len);
    cache.prefetch_many(st, l, &uniq, k3_expert_names);
    let mut slots = Vec::with_capacity(idx.len());
    for &x in &idx {
        let ex = x as usize;
        let s = cache
            .get(st, l, ex, &k3_expert_names(l, ex))
            .ok_or_else(|| format!("k3_run: L{l} expert {ex} would not load"))?;
        slots.push(s);
    }
    let eqs: Vec<ops::Eq> = slots
        .iter()
        .map(|&s| {
            let q = cache.expert(s);
            ops::Eq { p1: q.p1, s1: q.s1, p3: q.p3, s3: q.s3, p2: q.p2, s2: q.s2 }
        })
        .collect();
    let moew = ops::MoeW {
        gate,
        bias,
        w1: &[],
        w3: &[],
        w2: &[],
        latent_norm: f(&q("block_sparse_moe.routed_expert_norm.weight")),
        down: m(&q("block_sparse_moe.routed_expert_down_proj.weight")),
        up: m(&q("block_sparse_moe.routed_expert_up_proj.weight")),
        sh1: m(&q("block_sparse_moe.shared_experts.gate_proj.weight")),
        sh3: m(&q("block_sparse_moe.shared_experts.up_proj.weight")),
        sh2: m(&q("block_sparse_moe.shared_experts.down_proj.weight")),
    };
    ops::moe_packed(&mut tmp, &hin, &moew, &dims, t_len, &idx, &wt, &eqs);
    res.post(ops::Sub::Mlp, &tmp, ());
    Ok(())
}
