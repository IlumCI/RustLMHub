// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4-Flash generation. Prefill then greedy decode.
//
// usage: v4_run --model DIR --tokenizer tokenizer.json --prompt "..." [--n 16]
//
// The model directory must hold every shard. A partial download is REFUSED rather than
// run: a missing layer would otherwise read as zeros and the model would still emit
// fluent, wrong text, which is the failure mode this codebase exists to avoid.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use k3::st::St;
use k3::tok::Tok;

struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    n: usize,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: PathBuf::new(),
        tokenizer: PathBuf::new(),
        prompt: String::new(),
        n: 16,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let need = |i: usize| -> Result<String, String> {
            v.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", v[i]))
        };
        match v[i].as_str() {
            "--model" => a.model = need(i)?.into(),
            "--tokenizer" => a.tokenizer = need(i)?.into(),
            "--prompt" => a.prompt = need(i)?,
            "--n" => a.n = need(i)?.parse().map_err(|_| "--n wants a number".to_string())?,
            other => return Err(format!("unknown flag {other}")),
        }
        i += 2;
    }
    if a.model.as_os_str().is_empty() || a.tokenizer.as_os_str().is_empty() {
        return Err("--model and --tokenizer are required".into());
    }
    Ok(a)
}

/// Which tensors a complete DeepSeek-V4-Flash checkpoint must expose. Reported all at
/// once rather than one per run, the way k3_cfg reports missing config keys.
fn missing(s: &St, n_layers: usize) -> Vec<String> {
    let mut out = Vec::new();
    for t in ["embed.weight", "norm.weight", "head.weight"] {
        if s.find(t).is_none() {
            out.push(t.to_string());
        }
    }
    for l in 0..n_layers {
        for t in [
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn.attn_sink",
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn.gate.weight",
            "hc_attn_fn",
            "hc_ffn_fn",
        ] {
            let name = format!("layers.{l}.{t}");
            if s.find(&name).is_none() {
                out.push(name);
            }
        }
    }
    out
}

fn main() -> ExitCode {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("v4_run: {e}");
            eprintln!("usage: v4_run --model DIR --tokenizer tokenizer.json --prompt \"...\" [--n 16]");
            return ExitCode::from(2);
        }
    };

    let tok = match Tok::from_file(&a.tokenizer, 0, 1) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("v4_run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ids = match tok.encode(&a.prompt, false) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("v4_run: encode: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("tokenizer : {} entries", tok.vocab_size());
    println!("prompt    : {:?} -> {} tokens {:?}", a.prompt, ids.len(),
             &ids[..ids.len().min(12)]);

    let s = match St::open(Path::new(&a.model)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("v4_run: {e}");
            return ExitCode::FAILURE;
        }
    };
    let layers: BTreeSet<usize> = s
        .tensors
        .iter()
        .filter_map(|t| t.name.strip_prefix("layers."))
        .filter_map(|r| r.split('.').next())
        .filter_map(|n| n.parse().ok())
        .collect();
    println!(
        "checkpoint: {} shards, {} tensors, {} distinct layers present",
        s.nshard(),
        s.tensors.len(),
        layers.len()
    );

    const N_LAYERS: usize = 43;
    let miss = missing(&s, N_LAYERS);
    if !miss.is_empty() {
        eprintln!(
            "\nv4_run: this checkpoint is incomplete: {} of the required tensors are absent.",
            miss.len()
        );
        for m in miss.iter().take(8) {
            eprintln!("    {m}");
        }
        if miss.len() > 8 {
            eprintln!("    ... and {} more", miss.len() - 8);
        }
        let have: Vec<usize> = layers.iter().copied().collect();
        eprintln!("  layers present: {have:?}");
        eprintln!(
            "\n  Refusing to generate. A missing layer reads as zeros and the model still\n  \
             emits fluent, wrong text -- there is no error to notice at run time.\n  \
             The full release is 166.9 GB across 48 shards:\n    \
             hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir <DIR>"
        );
        return ExitCode::FAILURE;
    }

    match generate(&s, &tok, &ids, a.n) {
        Ok(text) => {
            println!("\n{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\nv4_run: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Prefill the prompt, then greedy-decode `n` tokens.
///
/// The trunk stays in its stored FP8/BF16 form throughout: dequantising it to f32 would
/// turn 8.29 GB into roughly 33 GB and put it out of reach of a 16 GB machine, which is
/// the arrangement this whole engine exists to make work. Only the routed experts move,
/// and they move through the cache in packed MXFP4.
fn generate(st: &St, tok: &Tok, prompt: &[u32], n: usize) -> Result<String, String> {
    use k3::cache::{v4_expert_names, Cache};
    use k3::ops::{HcLayer, HyperConnResidual, W};
    use k3::v4::*;

    const N_LAYERS: usize = 43;
    const HIDDEN: usize = 4096;
    const HC: usize = 4;
    const EPS: f32 = 1e-6;
    const HASH_LAYERS: usize = 3;

    // compress_ratios is 46 entries long for 43 decoder layers; see v4::COMPRESS_RATIOS
    // for why the three trailing zeros are not layers 40..42.
    let ratio_of = compress_ratio;

    let raw = |n: &str| -> Result<Vec<u8>, String> {
        let t = st.find(n).ok_or_else(|| format!("missing {n}"))?;
        let mut b = vec![0u8; t.nbytes as usize];
        st.read(t, &mut b);
        Ok(b)
    };
    let f32s = |n: &str| -> Result<Vec<f32>, String> {
        let t = st.find(n).ok_or_else(|| format!("missing {n}"))?;
        let mut v = vec![0f32; t.numel() as usize];
        st.read_f32(t, &mut v);
        Ok(v)
    };
    let i32s = |n: &str| -> Result<Vec<i32>, String> {
        let t = st.find(n).ok_or_else(|| format!("missing {n}"))?;
        let mut v = vec![0f32; t.numel() as usize];
        st.read_f32(t, &mut v);
        Ok(v.into_iter().map(|x| x as i32).collect())
    };

    let slot = {
        let r = k3::cache::locate(st, &v4_expert_names(2, 0)).ok_or("layer 2 expert 0")?;
        k3::cache::slot_need(&r)
    };
    let mut cache = Cache::new(5_000_000_000, slot, 256, 6)?;
    eprintln!("expert cache: {} slots of {:.1} MB", cache.nslot(), slot as f64 / 1e6);

    let embed = f32s("embed.weight")?;
    let final_norm = f32s("norm.weight")?;
    let (head, head_s) = (raw("head.weight")?, raw("head.scale").unwrap_or_default());
    let vocab = tok.vocab_size();

    let md = MoeDimsV4 {
        hidden: HIDDEN,
        moe_inter: 2048,
        n_experts: 256,
        topk: 6,
        route_scale: 1.5,
        swiglu_limit: 10.0,
        n_hash_layers: HASH_LAYERS,
    };

    let mut ids: Vec<u32> = prompt.to_vec();
    let mut text = String::new();

    for step in 0..n {
        let t_len = ids.len();
        // Hyper-Connections carry hc_mult copies of the stream; they start identical.
        let mut x0 = vec![0f32; t_len * HC * HIDDEN];
        for (t, &id) in ids.iter().enumerate() {
            let row = &embed[id as usize * HIDDEN..][..HIDDEN];
            for k in 0..HC {
                x0[(t * HC + k) * HIDDEN..][..HIDDEN].copy_from_slice(row);
            }
        }
        let mut res = HyperConnResidual::new(&x0, t_len, HIDDEN, HC, EPS, EPS, 20);

        for l in 0..N_LAYERS {
            let p = |n: &str| format!("layers.{l}.{n}");
            let ratio = ratio_of(l);
            let rope = if ratio == 0 {
                precompute_rope(64, t_len.max(2), 0, 10000.0, 16.0, 32.0, 1.0)
            } else {
                precompute_rope(64, t_len.max(2), 65536, 160000.0, 16.0, 32.0, 1.0)
            };

            let (wqa, wqas) = (raw(&p("attn.wq_a.weight"))?, raw(&p("attn.wq_a.scale"))?);
            let (wqb, wqbs) = (raw(&p("attn.wq_b.weight"))?, raw(&p("attn.wq_b.scale"))?);
            let (wkv, wkvs) = (raw(&p("attn.wkv.weight"))?, raw(&p("attn.wkv.scale"))?);
            let (woa, woas) = (raw(&p("attn.wo_a.weight"))?, raw(&p("attn.wo_a.scale"))?);
            let (wob, wobs) = (raw(&p("attn.wo_b.weight"))?, raw(&p("attn.wo_b.scale"))?);
            let (qn, kvn, sink) = (
                f32s(&p("attn.q_norm.weight"))?,
                f32s(&p("attn.kv_norm.weight"))?,
                f32s(&p("attn.attn_sink"))?,
            );
            let ad = AttnDims {
                hidden: HIDDEN, n_heads: 64, head_dim: 512, rope_head_dim: 64,
                q_lora_rank: 1024, o_lora_rank: 1024, o_groups: 8, window: 128,
                compress_ratio: ratio, eps: EPS,
            };
            let attn = AttnW {
                wq_a: W::F8Block { w: &wqa, scale: &wqas, block: 128 },
                q_norm: &qn,
                wq_b: W::F8Block { w: &wqb, scale: &wqbs, block: 128 },
                wkv: W::F8Block { w: &wkv, scale: &wkvs, block: 128 },
                kv_norm: &kvn,
                wo_a: W::F8Block { w: &woa, scale: &woas, block: 128 },
                wo_b: W::F8Block { w: &wob, scale: &wobs, block: 128 },
                attn_sink: &sink,
            };

            let (ckv, cg, ape, cn, ikv, ig, iape, inn, iwq, iwqs, iwp);
            let (cw, cd, icw, icd, iw, idm);
            let compressed = if ratio > 0 {
                ckv = f32s(&p("attn.compressor.wkv.weight"))?;
                cg = f32s(&p("attn.compressor.wgate.weight"))?;
                ape = f32s(&p("attn.compressor.ape"))?;
                cn = f32s(&p("attn.compressor.norm.weight"))?;
                cw = CompressorW { ape: &ape, wkv: W::F32(&ckv), wgate: W::F32(&cg), norm: &cn };
                cd = CompressorDims { hidden: HIDDEN, head_dim: 512, rope_head_dim: 64,
                                      ratio, rotate: false, eps: EPS };
                let indexer = if ratio == 4 {
                    ikv = f32s(&p("attn.indexer.compressor.wkv.weight"))?;
                    ig = f32s(&p("attn.indexer.compressor.wgate.weight"))?;
                    iape = f32s(&p("attn.indexer.compressor.ape"))?;
                    inn = f32s(&p("attn.indexer.compressor.norm.weight"))?;
                    iwq = raw(&p("attn.indexer.wq_b.weight"))?;
                    iwqs = raw(&p("attn.indexer.wq_b.scale"))?;
                    iwp = f32s(&p("attn.indexer.weights_proj.weight"))?;
                    icw = CompressorW { ape: &iape, wkv: W::F32(&ikv), wgate: W::F32(&ig), norm: &inn };
                    icd = CompressorDims { hidden: HIDDEN, head_dim: 128, rope_head_dim: 64,
                                           ratio: 4, rotate: true, eps: EPS };
                    iw = IndexerW {
                        wq_b: W::F8Block { w: &iwq, scale: &iwqs, block: 128 },
                        weights_proj: W::F32(&iwp),
                    };
                    idm = IndexerDims { hidden: HIDDEN, n_heads: 64, head_dim: 128,
                                        rope_head_dim: 64, q_lora_rank: 1024,
                                        index_topk: 512, ratio: 4 };
                    Some((&iw, &idm, &icw, &icd))
                } else { None };
                Some(Compressed { w: &cw, d: &cd, indexer })
            } else { None };

            let (hcaf, hcab, hcas) = (f32s(&p("hc_attn_fn"))?, f32s(&p("hc_attn_base"))?,
                                      f32s(&p("hc_attn_scale"))?);
            let (hcff, hcfb, hcfs) = (f32s(&p("hc_ffn_fn"))?, f32s(&p("hc_ffn_base"))?,
                                      f32s(&p("hc_ffn_scale"))?);
            let hc = HcLayer {
                attn_fn: &hcaf, attn_base: &hcab, attn_scale: [hcas[0], hcas[1], hcas[2]],
                ffn_fn: &hcff, ffn_base: &hcfb, ffn_scale: [hcfs[0], hcfs[1], hcfs[2]],
            };

            let gate = f32s(&p("ffn.gate.weight"))?;
            let bias = f32s(&p("ffn.gate.e_score_correction_bias")).ok();
            let t2e = if l < HASH_LAYERS { Some(i32s(&p("ffn.gate.tid2eid"))?) } else { None };
            let (s1, s1s) = (raw(&p("ffn.shared_experts.w1.weight"))?,
                             raw(&p("ffn.shared_experts.w1.scale"))?);
            let (s3, s3s) = (raw(&p("ffn.shared_experts.w3.weight"))?,
                             raw(&p("ffn.shared_experts.w3.scale"))?);
            let (s2, s2s) = (raw(&p("ffn.shared_experts.w2.weight"))?,
                             raw(&p("ffn.shared_experts.w2.scale"))?);
            let moe = MoeWV4 {
                gate: &gate,
                bias: bias.as_deref(),
                tid2eid: t2e.as_deref(),
                sh1: W::F8Block { w: &s1, scale: &s1s, block: 128 },
                sh3: W::F8Block { w: &s3, scale: &s3s, block: 128 },
                sh2: W::F8Block { w: &s2, scale: &s2s, block: 128 },
            };

            let an = f32s(&p("attn_norm.weight"))?;
            let fnorm = f32s(&p("ffn_norm.weight"))?;
            let layer = LayerV4 {
                attn, attn_dims: &ad, compressed, hc,
                attn_norm: &an, ffn_norm: &fnorm, moe,
            };
            res.begin_layer(&layer.hc);
            layer_forward(&mut res, &layer, &md, l, &ids, t_len, &rope, st, &mut cache,
                          v4_expert_names, EPS)?;
        }

        // hc_head reduces the copies one last time, then norm and the vocabulary head.
        let mut last = vec![0f32; HIDDEN];
        let state = res.state();
        for k in 0..HC {
            let src = &state[((t_len - 1) * HC + k) * HIDDEN..][..HIDDEN];
            for i in 0..HIDDEN {
                last[i] += src[i] / HC as f32;
            }
        }
        let mut normed = vec![0f32; HIDDEN];
        k3::ops::rmsnorm(&mut normed, &last, &final_norm, HIDDEN, EPS);
        let mut logits = vec![0f32; vocab];
        let hw = if head_s.is_empty() {
            W::Bf16(unsafe {
                std::slice::from_raw_parts(head.as_ptr().cast::<u16>(), head.len() / 2)
            })
        } else {
            W::F8Block { w: &head, scale: &head_s, block: 128 }
        };
        k3::ops::mmw(&mut logits, &normed, hw, HIDDEN, vocab);

        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or("empty logits")?;
        if next == tok.eos {
            break;
        }
        let piece = tok.piece(next)?;
        print!("{piece}");
        use std::io::Write as _;
        let _ = std::io::stdout().flush();
        text.push_str(&piece);
        ids.push(next);
        cache.report(&format!("  [token {}]", step + 1));
    }
    Ok(text)
}
