// SPDX-License-Identifier: Apache-2.0
//
// The descriptor against the config.json files the models actually ship. A hand-written
// approximation would agree with itself and with nothing else.

use std::path::PathBuf;

fn fx(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/arch")).join(name)
}

fn spec(name: &str) -> k3::arch::Spec {
    k3::arch::spec_file(&fx(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn every_published_config_parses() {
    for f in ["deepseek_v4_flash.json", "glm_5_1.json", "minimax_m3.json"] {
        let s = spec(f);
        assert!(s.hidden > 0 && s.n_layers > 0 && s.vocab > 0, "{f}: {s:?}");
        assert!(s.n_experts > 0 && s.topk > 0 && s.topk <= s.n_experts, "{f}");
        assert!(s.n_heads > 0 && s.head_dim > 0, "{f}");
    }
}

#[test]
fn deepseek_v4_flash_matches_its_release() {
    let s = spec("deepseek_v4_flash.json");
    assert_eq!(s.family, k3::arch::Family::V4);
    assert_eq!((s.hidden, s.n_layers, s.vocab), (4096, 43, 129280));
    assert_eq!((s.n_experts, s.topk, s.n_shared), (256, 6, 1));
    assert_eq!((s.n_heads, s.head_dim), (64, 512));
    assert_eq!(s.moe_inter, 2048);
    assert_eq!(s.n_hash_layers, 3);
    assert_eq!(s.scoring, k3::ops::Scoring::SqrtSoftplus);
    assert!(matches!(s.glu, k3::ops::Glu::SwigluClamped { limit } if limit == 10.0));
    assert_eq!(s.extra_layers, 1);
    assert!(!s.is_dense(0), "V4 has no dense layer");
}

#[test]
fn glm_5_1_matches_its_release() {
    let s = spec("glm_5_1.json");
    assert_eq!(s.family, k3::arch::Family::Glm);
    assert_eq!((s.hidden, s.n_layers, s.vocab), (6144, 78, 154880));
    assert_eq!((s.n_experts, s.topk, s.n_shared), (256, 8, 1));
    assert_eq!(s.scoring, k3::ops::Scoring::Sigmoid);
    assert_eq!(s.rope_theta, 1e6, "rope lives under rope_parameters, not at top level");
    // Both rope flags are true in the release. Pairing adjacent elements versus splitting
    // the axis in half are different rotations that both produce working attention.
    assert!(s.rope_interleave);
    assert!(s.is_dense(0) && s.is_dense(2) && !s.is_dense(3), "first_k_dense_replace = 3");
}

#[test]
fn minimax_m3_matches_its_release() {
    let s = spec("minimax_m3.json");
    assert_eq!(s.family, k3::arch::Family::M3);
    assert_eq!((s.hidden, s.n_layers, s.vocab), (6144, 60, 200064));
    assert_eq!((s.n_experts, s.topk), (128, 4));
    assert_eq!((s.n_heads, s.n_kv_heads), (64, 4), "GQA, not MHA");
    assert_eq!(s.head_dim, 128);
    assert!(s.gemma_norm, "use_gemma_norm: the gain is an offset, so the factor is 1 + w");
    assert_eq!(s.rotary_dim, 64, "partial rotary: only 64 of 128 dims rotate");
    assert_eq!(s.dense_inter, 12288);
    assert!(s.is_dense(0) && s.is_dense(2) && !s.is_dense(3), "moe_layer_freq [0,0,0,1,..]");
}

// Every one of the three ships weights for layer blocks past the decoder stack. Counting
// them as decoder layers is what made compress_ratios look three entries too long.
#[test]
fn none_of_the_extra_blocks_are_counted_as_decoder_layers() {
    for f in ["deepseek_v4_flash.json", "glm_5_1.json", "minimax_m3.json"] {
        let s = spec(f);
        assert!(s.extra_layers <= 8, "{f}: {} extra blocks looks wrong", s.extra_layers);
    }
}

// The one seam docs/MULTI_MODEL.md called "still wrong". K3's reference upcasts to f64;
// none of these three do.
#[test]
fn only_k3_accumulates_rmsnorm_in_f64() {
    for f in ["deepseek_v4_flash.json", "glm_5_1.json", "minimax_m3.json"] {
        assert_eq!(spec(f).rms_acc, k3::ops::Acc::F32, "{f}");
    }
}

// The descriptor must reproduce docs/MULTI_MODEL.md's measured byte counts from
// config.json alone. If it cannot, some field is being read wrong in a way that no shape
// check would catch.
#[test]
fn deepseek_v4_byte_counts_match_the_measured_figures() {
    let s = spec("deepseek_v4_flash.json");
    let one = s.expert_bytes(4, 32) as f64 / 1e6;
    let per_tok = s.expert_bytes_per_token(4, 32) as f64 / 1e9;
    assert!((one - 13.37).abs() < 0.01, "one expert: {one:.2} MB, doc says 13.37");
    assert!((per_tok - 3.45).abs() < 0.02, "per token: {per_tok:.2} GB, doc says 3.45");
}
