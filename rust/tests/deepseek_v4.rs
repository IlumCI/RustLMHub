// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4-Flash reuses the MXFP4 expert format K3 uses: E2M1 nibbles in packed
// bytes, one E8M0 scale per group of 32. inference/kernel.py pins the three constants
// that matter -- block_size 32, scale dtype float8_e8m0fnu, fp4_max 6.0 -- so
// ops::mxfp4_dequant applies to a 304B model it was never written for.
//
// The fixture is real checkpoint bytes from
// deepseek-ai/DeepSeek-V4-Flash-0731 model-00002-of-00048.safetensors, decoded
// independently with numpy + ml_dtypes rather than by restating this implementation.

use k3::ops;
use serde_json::Value;

fn fixture() -> Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/deepseek_v4_mxfp4.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}")))
        .expect("valid JSON")
}

fn bytes(v: &Value, k: &str) -> Vec<u8> {
    v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect()
}

#[test]
fn dequantises_a_real_deepseek_v4_expert_bit_exactly() {
    let v = fixture();
    let rows = v["rows"].as_u64().unwrap() as usize;
    let pcols = v["pcols"].as_u64().unwrap() as usize;
    let group = v["group"].as_u64().unwrap() as usize;
    assert_eq!(group, ops::MXFP4_GROUP, "V4 must use the same group as K3");

    let packed = bytes(&v, "packed");
    let scales = bytes(&v, "scales");
    let want: Vec<u32> =
        v["expected_bits"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();

    let mut got = vec![0f32; rows * pcols * 2];
    ops::mxfp4_dequant(&mut got, &packed, &scales, rows, pcols, group);

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), *w, "element {i}: {g} vs {}", f32::from_bits(*w));
    }
}

// The quantiser picks a power-of-two scale so amax/scale lands at fp4_max = 6.0
// (inference/kernel.py:134). Every group must therefore stay within it -- a group
// exceeding 6.0 means the scale was paired with the wrong group, which is the
// row-stride error this checks for. It says nothing about nibble order: reversing that
// permutes elements within a pair and leaves every group statistic identical.
#[test]
fn every_group_respects_fp4_max() {
    let v = fixture();
    let rows = v["rows"].as_u64().unwrap() as usize;
    let pcols = v["pcols"].as_u64().unwrap() as usize;
    let group = v["group"].as_u64().unwrap() as usize;
    let width = pcols * 2;
    let ngrp = width / group;

    let packed = bytes(&v, "packed");
    let scales = bytes(&v, "scales");
    let mut out = vec![0f32; rows * width];
    ops::mxfp4_dequant(&mut out, &packed, &scales, rows, pcols, group);

    for r in 0..rows {
        for g in 0..ngrp {
            let sb = scales[r * ngrp + g];
            if sb == 255 {
                continue;
            }
            let sc = k3::st::e8m0_to_f32(sb);
            let amax = out[r * width + g * group..][..group]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
            assert!(
                amax / sc <= 6.0 + 1e-3,
                "row {r} group {g}: amax/scale = {} exceeds fp4_max",
                amax / sc
            );
        }
    }
}

fn load(name: &str) -> Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}")))
        .expect("valid JSON")
}

#[test]
fn fp8_block_matmul_matches_numpy_on_real_attention_weights() {
    let v = load("deepseek_v4_fp8.json");
    let rows = v["rows"].as_u64().unwrap() as usize;
    let k_in = v["k_in"].as_u64().unwrap() as usize;
    let block = v["block"].as_u64().unwrap() as usize;
    let w = bytes(&v, "weight");
    let sc = bytes(&v, "scale");
    let want: Vec<f32> = v["expected_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect();

    let x: Vec<f32> = (0..k_in).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut got = vec![0f32; rows];
    ops::matmul_fp8_block(&mut got, &x, &w, &sc, k_in, rows, block);

    // f64 accumulation on both sides, but numpy sums in a different order, so this is
    // an agreement check rather than a bitwise one.
    for (i, (g, t)) in got.iter().zip(&want).enumerate() {
        let d = (g - t).abs();
        assert!(d <= 1e-6 + 1e-5 * t.abs(), "row {i}: got {g:e} want {t:e} (d {d:e})");
    }
}

// inference/model.py clamps `up` on both sides but `gate` only from above. Symmetric
// clamping still yields a bounded, plausible activation and is wrong.
#[test]
fn swiglu_clamp_is_asymmetric() {
    let limit = 10.0f32;
    let x = [-50.0f32, 1.0]; // gate well below -limit, up in range
    let mut y = [0f32; 1];
    ops::glu(&mut y, &x, 1, ops::Glu::SwigluClamped { limit });

    let g = -50.0f32; // NOT clamped
    let want = (g * (1.0 / (1.0 + (-g).exp()))) * 1.0f32;
    assert_eq!(y[0], want);

    let sym = -10.0f32; // what a symmetric clamp would have used
    let wrong = (sym * (1.0 / (1.0 + (-sym).exp()))) * 1.0f32;
    assert_ne!(y[0], wrong, "clamping gate's minimum is plausible and wrong");
}

#[test]
fn swiglu_clamps_up_on_both_sides() {
    let limit = 10.0f32;
    for up in [-50.0f32, 50.0] {
        let x = [1.0f32, up];
        let mut y = [0f32; 1];
        ops::glu(&mut y, &x, 1, ops::Glu::SwigluClamped { limit });
        let g = 1.0f32;
        let want = (g * (1.0 / (1.0 + (-g).exp()))) * up.clamp(-limit, limit);
        assert_eq!(y[0], want, "up={up}");
    }
}

// sqrt(softplus) is the V4 scoring function; torch's softplus goes linear above 20.
#[test]
fn sqrt_softplus_scoring_matches_the_reference_definition() {
    let hidden = 1;
    let n_experts = 4;
    let logits = [-3.0f32, 0.0, 2.5, 25.0];
    let w: Vec<f32> = logits.to_vec(); // x = [1.0] makes the dot product the logit
    let x = [1.0f32];
    let mut idx = vec![0i32; n_experts];
    let mut wt = vec![0f32; n_experts];
    ops::router_scored(
        &mut idx, &mut wt, &x, &w, None, hidden, n_experts, n_experts, false, 1.0,
        ops::Scoring::SqrtSoftplus,
    );
    let sp = |v: f32| if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
    for (j, &e) in idx.iter().enumerate() {
        let want = sp(logits[e as usize]).sqrt();
        assert!((wt[j] - want).abs() < 1e-6, "expert {e}: {} vs {want}", wt[j]);
    }
    assert_eq!(idx[0], 3, "largest logit ranks first");
}

// The generalised router must reproduce the K3 one exactly on K3's settings, or the
// refactor has silently changed a model that already passes its fixtures.
#[test]
fn scored_router_reproduces_the_k3_router_on_sigmoid() {
    let (hidden, n_experts, topk) = (16, 8, 3);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.31).sin()).collect();
    let w: Vec<f32> = (0..n_experts * hidden).map(|i| (i as f32 * 0.17).cos()).collect();
    let bias: Vec<f32> = (0..n_experts).map(|i| (i as f32 * 0.05) - 0.2).collect();

    let (mut i1, mut w1) = (vec![0i32; topk], vec![0f32; topk]);
    let (mut i2, mut w2) = (vec![0i32; topk], vec![0f32; topk]);
    ops::router(&mut i1, &mut w1, &x, &w, Some(&bias), hidden, n_experts, topk, true, 1.5);
    ops::router_scored(
        &mut i2, &mut w2, &x, &w, Some(&bias), hidden, n_experts, topk, true, 1.5,
        ops::Scoring::Sigmoid,
    );
    assert_eq!(i1, i2);
    assert_eq!(
        w1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        w2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "generalising the score function must not change K3's arithmetic"
    );
}

// Hash routing ignores the scores entirely when choosing experts.
#[test]
fn hash_routing_selects_by_token_id_not_by_score() {
    let (hidden, n_experts, topk) = (4, 8, 2);
    let x = vec![1.0f32; hidden];
    let w: Vec<f32> = (0..n_experts * hidden).map(|i| i as f32).collect(); // expert 7 scores highest
    let tid2eid: Vec<i32> = vec![5, 1, /* token 1 */ 0, 3];
    let (mut idx, mut wt) = (vec![0i32; topk], vec![0f32; topk]);
    ops::router_hashed(
        &mut idx, &mut wt, &x, &w, &tid2eid, 0, hidden, n_experts, topk, 1.0,
        ops::Scoring::SqrtSoftplus,
    );
    assert_eq!(idx, vec![5, 1], "token 0 must get its table entry, not the top-scoring experts");
    assert!((wt.iter().sum::<f32>() - 1.0).abs() < 1e-6, "weights renormalise to route_scale");
}

fn f32s(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[track_caller]
fn near(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol + tol * w.abs(), "{what}[{i}]: {g:e} vs {w:e}");
    }
}

#[test]
fn hc_sinkhorn_matches_the_reference() {
    let v = load("deepseek_v4_hc.json");
    let hc = v["hc"].as_u64().unwrap() as usize;
    let iters = v["iters"].as_u64().unwrap() as usize;
    let eps = v["hc_eps"].as_f64().unwrap() as f32;
    let base = f32s(&v, "hc_base");
    let sc = f32s(&v, "hc_scale");
    let mixes: Vec<f32> = v["mixes_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect();

    let (pre, post, comb) =
        k3::ops::hc_split_sinkhorn(&mixes, &[sc[0], sc[1], sc[2]], &base, hc, iters, eps);
    near("pre", &pre, &f32s(&v, "pre"), 1e-6);
    near("post", &post, &f32s(&v, "post"), 1e-6);
    near("comb", &comb, &f32s(&v, "comb"), 1e-5);
}

// Sinkhorn drives the combination matrix toward doubly stochastic. The reference ends
// on a COLUMN normalisation, so columns land on 1 exactly and rows only approach it --
// asserting both at 1 would be wrong.
#[test]
fn sinkhorn_leaves_columns_normalised() {
    let v = load("deepseek_v4_hc.json");
    let hc = v["hc"].as_u64().unwrap() as usize;
    let comb = f32s(&v, "comb");
    for k in 0..hc {
        let col: f32 = (0..hc).map(|j| comb[j * hc + k]).sum();
        assert!((col - 1.0).abs() < 1e-4, "column {k} sums to {col}");
    }
}

// post is 2*sigmoid(...) while pre is sigmoid(...) + eps. Dropping the factor of two
// halves every module contribution and still produces a stable, plausible stream. It
// cannot be caught by a value range -- with the real hc_attn_scale[1] of 0.019 every
// sigmoid lands below 0.5, so post stays under 1 either way. Compare the formulae.
#[test]
fn hc_post_weights_carry_the_factor_of_two() {
    let v = load("deepseek_v4_hc.json");
    let hc = v["hc"].as_u64().unwrap() as usize;
    let iters = v["iters"].as_u64().unwrap() as usize;
    let eps = v["hc_eps"].as_f64().unwrap() as f32;
    let base = f32s(&v, "hc_base");
    let sc = f32s(&v, "hc_scale");
    let mixes: Vec<f32> = v["mixes_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| f32::from_bits(x.as_u64().unwrap() as u32))
        .collect();

    let (pre, post, _) =
        k3::ops::hc_split_sinkhorn(&mixes, &[sc[0], sc[1], sc[2]], &base, hc, iters, eps);
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    for j in 0..hc {
        let arg = mixes[j + hc] * sc[1] + base[j + hc];
        assert!((post[j] - 2.0 * sig(arg)).abs() < 1e-6, "post[{j}] must be 2*sigmoid");
        // Relative, not absolute: with these weights sigmoid(arg) is small enough that
        // 2*sig and sig differ by less than 1e-3 in absolute terms while still being a
        // factor of two apart.
        assert!(
            (post[j] / sig(arg) - 2.0).abs() < 1e-5,
            "post[{j}] must be twice the sigmoid, got ratio {}",
            post[j] / sig(arg)
        );
        // pre takes hc_scale[0] and its own base slice, and adds eps rather than doubling.
        let parg = mixes[j] * sc[0] + base[j];
        assert!((pre[j] - (sig(parg) + eps)).abs() < 1e-6, "pre[{j}]");
    }
}

#[test]
fn hyper_connections_pre_and_post_match_the_reference() {
    let v = load("deepseek_v4_hc.json");
    let hc = v["hc"].as_u64().unwrap() as usize;
    let d = v["d"].as_u64().unwrap() as usize;
    let iters = v["iters"].as_u64().unwrap() as usize;
    let hc_eps = v["hc_eps"].as_f64().unwrap() as f32;
    let norm_eps = v["norm_eps"].as_f64().unwrap() as f32;
    let fnw = f32s(&v, "fn");
    let base = f32s(&v, "hc_base");
    let sc = f32s(&v, "hc_scale");
    let x = f32s(&v, "x");
    let module_out = f32s(&v, "module_out");

    let layer = k3::ops::HcLayer {
        attn_fn: &fnw,
        attn_base: &base,
        attn_scale: [sc[0], sc[1], sc[2]],
        ffn_fn: &fnw,
        ffn_base: &base,
        ffn_scale: [sc[0], sc[1], sc[2]],
    };
    let mut r = k3::ops::HyperConnResidual::new(&x, 1, d, hc, norm_eps, hc_eps, iters);
    r.begin_layer(&layer);

    use k3::ops::Residual;
    let mut input = vec![0f32; d];
    let carry = r.pre(k3::ops::Sub::Attn, &mut input);
    near("hc_pre", &input, &f32s(&v, "pre_out"), 1e-5);

    r.post(k3::ops::Sub::Attn, &module_out, carry);
    near("hc_post", r.state(), &f32s(&v, "post_out"), 1e-5);
}

// y[k] = post[k]*out + sum_j comb[j][k]*residual[j]: the sum runs over comb's FIRST
// index. Transposing it is a bounded, plausible mix of the same copies.
#[test]
fn hc_post_sums_over_combs_first_index() {
    let v = load("deepseek_v4_hc.json");
    let hc = v["hc"].as_u64().unwrap() as usize;
    let d = v["d"].as_u64().unwrap() as usize;
    let comb = f32s(&v, "comb");
    let x = f32s(&v, "x");
    let out = f32s(&v, "post_out");
    let post = f32s(&v, "post");
    let mo = f32s(&v, "module_out");

    let asym = (0..hc).any(|j| (0..hc).any(|k| (comb[j * hc + k] - comb[k * hc + j]).abs() > 1e-3));
    assert!(asym, "fixture must have an asymmetric comb or this proves nothing");

    for k in 0..hc {
        for i in 0..d {
            let want = post[k] * mo[i] + (0..hc).map(|j| comb[j * hc + k] * x[j * d + i]).sum::<f32>();
            assert!((out[k * d + i] - want).abs() < 1e-5, "[{k}][{i}]");
        }
    }
}

#[test]
fn attention_matches_the_reference() {
    let v = load("deepseek_v4_attn.json");
    let u = |k: &str| v[k].as_u64().unwrap() as usize;
    let d = k3::v4::AttnDims {
        hidden: u("hidden"),
        n_heads: u("n_heads"),
        head_dim: u("head_dim"),
        rope_head_dim: u("rope_head_dim"),
        q_lora_rank: u("q_lora_rank"),
        o_lora_rank: u("o_lora_rank"),
        o_groups: u("o_groups"),
        window: u("window"),
        compress_ratio: 0,
        eps: v["eps"].as_f64().unwrap() as f32,
    };
    let t_len = u("T");
    let (wq_a, wq_b, wkv) = (f32s(&v, "wq_a"), f32s(&v, "wq_b"), f32s(&v, "wkv"));
    let (wo_a, wo_b) = (f32s(&v, "wo_a"), f32s(&v, "wo_b"));
    let (qn, kvn, sink) = (f32s(&v, "q_norm"), f32s(&v, "kv_norm"), f32s(&v, "attn_sink"));
    let w = k3::v4::AttnW {
        wq_a: k3::ops::W::F32(&wq_a),
        q_norm: &qn,
        wq_b: k3::ops::W::F32(&wq_b),
        wkv: k3::ops::W::F32(&wkv),
        kv_norm: &kvn,
        wo_a: k3::ops::W::F32(&wo_a),
        wo_b: k3::ops::W::F32(&wo_b),
        attn_sink: &sink,
    };
    let rope = k3::v4::precompute_rope(
        d.rope_head_dim,
        t_len.max(2),
        0,
        v["rope_theta"].as_f64().unwrap() as f32,
        16.0,
        32.0,
        1.0,
    );
    let x = f32s(&v, "x");
    let mut got = vec![0f32; t_len * d.hidden];
    k3::v4::attention_window(&mut got, &x, &w, &d, t_len, &rope);
    near("v4 attention", &got, &f32s(&v, "out"), 1e-5);
}

// Guard the entry rather than silently running a windowed approximation of a layer
// that should have been compressed. compress_ratios is [0,0,4,128,4,128,...], so most
// of the 43 layers take a path that does not exist yet.
#[test]
#[should_panic(expected = "must agree")]
fn compressed_layers_are_refused_not_approximated() {
    let v = load("deepseek_v4_attn.json");
    let u = |k: &str| v[k].as_u64().unwrap() as usize;
    let d = k3::v4::AttnDims {
        hidden: u("hidden"),
        n_heads: u("n_heads"),
        head_dim: u("head_dim"),
        rope_head_dim: u("rope_head_dim"),
        q_lora_rank: u("q_lora_rank"),
        o_lora_rank: u("o_lora_rank"),
        o_groups: u("o_groups"),
        window: u("window"),
        compress_ratio: 4,
        eps: 1e-6,
    };
    let (wq_a, wq_b, wkv) = (f32s(&v, "wq_a"), f32s(&v, "wq_b"), f32s(&v, "wkv"));
    let (wo_a, wo_b) = (f32s(&v, "wo_a"), f32s(&v, "wo_b"));
    let (qn, kvn, sink) = (f32s(&v, "q_norm"), f32s(&v, "kv_norm"), f32s(&v, "attn_sink"));
    let w = k3::v4::AttnW {
        wq_a: k3::ops::W::F32(&wq_a),
        q_norm: &qn,
        wq_b: k3::ops::W::F32(&wq_b),
        wkv: k3::ops::W::F32(&wkv),
        kv_norm: &kvn,
        wo_a: k3::ops::W::F32(&wo_a),
        wo_b: k3::ops::W::F32(&wo_b),
        attn_sink: &sink,
    };
    let rope = k3::v4::precompute_rope(d.rope_head_dim, 4, 0, 10000.0, 16.0, 32.0, 1.0);
    let mut out = vec![0f32; u("T") * d.hidden];
    k3::v4::attention_prefill(&mut out, &f32s(&v, "x"), &w, &d, u("T"), &rope, None);
}

fn comp_dims(c: &Value, rotate: bool) -> k3::v4::CompressorDims {
    let u = |k: &str| c[k].as_u64().unwrap() as usize;
    k3::v4::CompressorDims {
        hidden: u("hidden"),
        head_dim: u("head_dim"),
        rope_head_dim: u("rope_head_dim"),
        ratio: u("ratio"),
        rotate,
        eps: c["eps"].as_f64().unwrap() as f32,
    }
}

#[test]
fn hadamard_matches_the_reference_transform() {
    let v = load("deepseek_v4_compress.json");
    let mut x = f32s(&v["hadamard"], "x");
    let n = x.len();
    k3::v4::hadamard(&mut x, n);
    near("hadamard", &x, &f32s(&v["hadamard"], "out"), 1e-5);
}

#[test]
fn hadamard_is_its_own_inverse_at_this_scaling() {
    let n = 64;
    let orig: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin()).collect();
    let mut x = orig.clone();
    k3::v4::hadamard(&mut x, n);
    k3::v4::hadamard(&mut x, n);
    for (a, b) in x.iter().zip(&orig) {
        assert!((a - b).abs() < 1e-5, "H*H must be the identity at n^-0.5 scaling");
    }
}

#[test]
fn compressor_matches_the_reference_on_both_window_shapes() {
    let v = load("deepseek_v4_compress.json");
    for c in v["compress"].as_array().unwrap() {
        let d = comp_dims(c, false);
        let (wkv, wg, ape, norm) =
            (f32s(c, "wkv"), f32s(c, "wgate"), f32s(c, "ape"), f32s(c, "norm"));
        let w = k3::v4::CompressorW {
            ape: &ape,
            wkv: k3::ops::W::F32(&wkv),
            wgate: k3::ops::W::F32(&wg),
            norm: &norm,
        };
        let t_len = c["T"].as_u64().unwrap() as usize;
        let rope = k3::v4::precompute_rope(d.rope_head_dim, t_len.max(2), 0, 10000.0, 16.0, 32.0, 1.0);
        let got = k3::v4::compress_prefill(&f32s(c, "x"), &w, &d, t_len, &rope);
        near(&format!("compress ratio {}", d.ratio), &got, &f32s(c, "out"), 1e-5);
        // ratio 4 overlaps; anything else does not.
        assert_eq!(d.overlap(), d.ratio == 4);
        assert_eq!(d.coff(), if d.ratio == 4 { 2 } else { 1 });
    }
}

#[test]
fn indexer_matches_the_reference() {
    let v = load("deepseek_v4_compress.json");
    let c = &v["indexer"];
    let u = |k: &str| c[k].as_u64().unwrap() as usize;
    let d = comp_dims(c, true);
    let (wkv, wg, ape, norm) = (f32s(c, "wkv"), f32s(c, "wgate"), f32s(c, "ape"), f32s(c, "norm"));
    let w = k3::v4::CompressorW {
        ape: &ape,
        wkv: k3::ops::W::F32(&wkv),
        wgate: k3::ops::W::F32(&wg),
        norm: &norm,
    };
    let t_len = u("T");
    let rope = k3::v4::precompute_rope(d.rope_head_dim, t_len.max(2), 0, 10000.0, 16.0, 32.0, 1.0);

    let ikvc = k3::v4::compress_prefill(&f32s(c, "x"), &w, &d, t_len, &rope);
    near("indexer compressed kv", &ikvc, &f32s(c, "ikvc"), 1e-5);

    let (wq_b, wp, qr) = (f32s(c, "wq_b"), f32s(c, "weights_proj"), f32s(c, "qr"));
    let iw = k3::v4::IndexerW {
        wq_b: k3::ops::W::F32(&wq_b),
        weights_proj: k3::ops::W::F32(&wp),
    };
    let id = k3::v4::IndexerDims {
        hidden: u("hidden"),
        n_heads: u("n_heads"),
        head_dim: u("head_dim"),
        rope_head_dim: u("rope_head_dim"),
        q_lora_rank: u("q_lora_rank"),
        index_topk: u("index_topk"),
        ratio: u("ratio"),
    };
    let got = k3::v4::indexer_prefill(&qr, &f32s(c, "x"), &ikvc, &iw, &id, t_len, &rope, 0);
    let want = c["topk"].as_array().unwrap();
    for t in 0..t_len {
        let w: Vec<i64> = want[t].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        assert_eq!(got[t], w, "token {t} selected a different block order");
    }
}

// A block is causal for token t only when b < (t+1)/ratio. Off by one either way and
// a token either sees a block built from its own future or misses its last complete one.
#[test]
fn compressed_blocks_become_visible_at_multiples_of_the_ratio() {
    let rows = k3::v4::compress_topk_prefill(12, 3, 4, 0);
    let live = |t: usize| rows[t].iter().filter(|&&b| b >= 0).count();
    assert_eq!((0..12).map(live).collect::<Vec<_>>(), vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]);
}

#[test]
fn compress_topk_offsets_into_the_concatenated_cache() {
    // Indices are appended after the sliding window, so they carry an offset.
    let rows = k3::v4::compress_topk_prefill(8, 2, 4, 128);
    assert_eq!(rows[7], vec![128, 129], "block b must land at b + offset");
    assert_eq!(rows[0], vec![-1, -1], "masked slots stay -1, never offset");
}

// ---------------------------------------------------------------- compress_ratios ----
//
// The array is 46 long for 43 decoder layers, and the mismatch is the whole trap: the
// three trailing zeros belong to the layer-sized blocks the release ships beyond the
// decoder stack, not to layers 40..42. Transcribed from the released config.json of
// deepseek-ai/DeepSeek-V4-Flash-0731.

#[test]
fn compress_ratios_is_longer_than_the_decoder_stack() {
    const NUM_HIDDEN_LAYERS: usize = 43;
    assert_eq!(
        k3::v4::COMPRESS_RATIOS.len(),
        46,
        "the released array has 46 entries; a 43-entry transcription has dropped the \
         extra blocks and shifted every trailing zero onto a real decoder layer"
    );
    assert!(k3::v4::COMPRESS_RATIOS.len() > NUM_HIDDEN_LAYERS);
}

// The regression this file exists for. Treating the trailing zeros as the last three
// decoder layers gives layers 40, 41 and 42 a ratio of 0, which is a VALID ratio meaning
// "pure sliding window" -- so the model runs, stays fluent, and is wrong in three layers.
#[test]
fn the_last_three_decoder_layers_are_compressed_not_windowed() {
    assert_eq!(k3::v4::compress_ratio(40), 4);
    assert_eq!(k3::v4::compress_ratio(41), 128);
    assert_eq!(k3::v4::compress_ratio(42), 4);
}

#[test]
fn only_the_first_two_decoder_layers_are_pure_sliding_window() {
    const NUM_HIDDEN_LAYERS: usize = 43;
    let zeros: Vec<usize> = (0..NUM_HIDDEN_LAYERS)
        .filter(|&l| k3::v4::compress_ratio(l) == 0)
        .collect();
    assert_eq!(zeros, vec![0, 1], "layers 0 and 1 alone have ratio 0");
}

#[test]
fn compressed_layers_alternate_four_and_one_twenty_eight() {
    const NUM_HIDDEN_LAYERS: usize = 43;
    for l in 2..NUM_HIDDEN_LAYERS {
        let want = if l % 2 == 0 { 4 } else { 128 };
        assert_eq!(k3::v4::compress_ratio(l), want, "layer {l}");
    }
}

// The three blocks past the decoder stack. They are the reason the array is 46 long, and
// keeping them in the table -- rather than truncating to 43 -- is what makes the indices
// of the real layers line up with the released file.
#[test]
fn the_blocks_past_the_decoder_stack_carry_the_trailing_zeros() {
    assert_eq!(&k3::v4::COMPRESS_RATIOS[43..], &[0, 0, 0]);
}
