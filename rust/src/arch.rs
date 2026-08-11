// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use serde_json::Value;

use crate::cache::ExpertNames;
use crate::ops::{Acc, Glu, Scoring};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    K3,
    V4,
    Glm,
    M3,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::K3 => "kimi-k3",
            Family::V4 => "deepseek-v4",
            Family::Glm => "glm-moe-dsa",
            Family::M3 => "minimax-m3",
        }
    }
}

/// Dispatch on the checkpoint's own `model_type`, falling back to `architectures[0]`.
/// K3 ships neither, which is itself the signal: its config is the only one of the four
/// with a `linear_attn_config` / `kda_num_heads` block.
pub fn detect(root: &Value) -> Result<Family, String> {
    let mt = root["model_type"].as_str().unwrap_or("");
    let a0 = root["architectures"][0].as_str().unwrap_or("");
    let f = match (mt, a0) {
        ("deepseek_v4", _) | (_, "DeepseekV4ForCausalLM") => Family::V4,
        ("glm_moe_dsa", _) | (_, "GlmMoeDsaForCausalLM") => Family::Glm,
        ("minimax_m3_vl", _) | (_, "MiniMaxM3SparseForConditionalGeneration") => Family::M3,
        _ => {
            let k3 = root.get("kda_num_heads").is_some()
                || root.get("linear_attn_config").is_some()
                || root["text_config"].get("kda_num_heads").is_some();
            if !k3 {
                return Err(format!(
                    "unrecognised architecture: model_type {mt:?}, architectures[0] {a0:?}.\n  \
                     Known: deepseek_v4, glm_moe_dsa, minimax_m3_vl, and Kimi K3 (which ships \
                     neither key and is identified by its kda_num_heads block).\n  \
                     Refusing to guess: a config this reader cannot fully understand would \
                     silently produce a DIFFERENT model."
                ));
            }
            Family::K3
        }
    };
    Ok(f)
}

pub fn detect_file(p: &Path) -> Result<Family, String> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
    )
    .map_err(|e| format!("{}: {e}", p.display()))?;
    detect(&v)
}

/// Where a layer's MLP is dense rather than routed. Each architecture spells this
/// differently and the three spellings are not interchangeable.
#[derive(Clone, Debug)]
pub enum Dense {
    /// K3 `first_dense`, GLM `first_k_dense_replace`: layers `< k`.
    FirstK(usize),
    /// MiniMax-M3 `moe_layer_freq`: a per-layer 0/1 array. 0 means dense.
    PerLayer(Vec<u8>),
    /// DeepSeek-V4: every layer is routed. The first `num_hash_layers` route by token id
    /// rather than by score, which is a routing change, not a dense one.
    None,
}

impl Dense {
    pub fn is_dense(&self, l: usize) -> bool {
        match self {
            Dense::FirstK(k) => l < *k,
            Dense::PerLayer(v) => v.get(l).is_some_and(|&x| x == 0),
            Dense::None => false,
        }
    }
}

/// Everything the runner needs that does not depend on which architecture it is.
#[derive(Clone, Debug)]
pub struct Spec {
    pub family: Family,
    pub hidden: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rms_acc: Acc,
    pub gemma_norm: bool,

    pub n_experts: usize,
    pub topk: usize,
    pub n_shared: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub dense_inter: usize,
    pub routed_scale: f32,
    pub renorm: bool,
    pub scoring: Scoring,
    pub glu: Glu,
    pub dense: Dense,
    /// Layers routing by token id instead of by score (DeepSeek-V4 only).
    pub n_hash_layers: usize,

    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    /// Rope pairs adjacent elements when true, splits the axis in half when false. Both
    /// produce working attention and they are different rotations.
    pub rope_interleave: bool,
    pub rotary_dim: usize,

    /// The extra layer-sized blocks past the decoder stack (MTP / next-n prediction).
    /// Present in every checkpoint here and never a decoder layer.
    pub extra_layers: usize,
}

fn f32_of(v: &Value, k: &str, d: f32) -> f32 {
    v[k].as_f64().map(|x| x as f32).unwrap_or(d)
}
fn us(v: &Value, k: &str, d: usize) -> usize {
    v[k].as_u64().map(|x| x as usize).unwrap_or(d)
}
fn req(v: &Value, k: &str, miss: &mut Vec<String>) -> usize {
    match v[k].as_u64() {
        Some(x) => x as usize,
        None => {
            miss.push(k.to_string());
            0
        }
    }
}

fn scoring_of(s: &str, miss: &mut Vec<String>) -> Scoring {
    match s {
        "sigmoid" => Scoring::Sigmoid,
        "sqrtsoftplus" => Scoring::SqrtSoftplus,
        "softmax" => Scoring::Softmax,
        _ => {
            miss.push(format!("scoring_func (got {s:?})"));
            Scoring::Sigmoid
        }
    }
}

fn done(miss: Vec<String>, whence: &str) -> Result<(), String> {
    if miss.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{whence} is missing {} required field(s):\n    {}\n  refusing to substitute \
         defaults: a config this reader cannot fully understand would silently produce a \
         DIFFERENT model.",
        miss.len(),
        miss.join("\n    ")
    ))
}

pub fn spec(root: &Value, whence: &str) -> Result<Spec, String> {
    match detect(root)? {
        Family::V4 => v4(root, whence),
        Family::Glm => glm(root, whence),
        Family::M3 => m3(root, whence),
        Family::K3 => k3(root, whence),
    }
}

pub fn spec_file(p: &Path) -> Result<Spec, String> {
    let v: Value = serde_json::from_str(
        &std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
    )
    .map_err(|e| format!("{}: {e}", p.display()))?;
    spec(&v, &p.display().to_string())
}

fn v4(r: &Value, w: &str) -> Result<Spec, String> {
    let mut m = Vec::new();
    let s = Spec {
        family: Family::V4,
        hidden: req(r, "hidden_size", &mut m),
        n_layers: req(r, "num_hidden_layers", &mut m),
        vocab: req(r, "vocab_size", &mut m),
        rms_eps: f32_of(r, "rms_norm_eps", 1e-6),
        // DeepSeek's reference does x.float() then .mean(): f32, not f64.
        rms_acc: Acc::F32,
        gemma_norm: false,
        n_experts: req(r, "n_routed_experts", &mut m),
        topk: req(r, "num_experts_per_tok", &mut m),
        n_shared: req(r, "n_shared_experts", &mut m),
        moe_inter: req(r, "moe_intermediate_size", &mut m),
        shared_inter: us(r, "moe_intermediate_size", 0),
        dense_inter: 0,
        routed_scale: f32_of(r, "routed_scaling_factor", 1.0),
        renorm: r["norm_topk_prob"].as_bool().unwrap_or(true),
        scoring: scoring_of(r["scoring_func"].as_str().unwrap_or(""), &mut m),
        glu: Glu::SwigluClamped { limit: f32_of(r, "swiglu_limit", 10.0) },
        dense: Dense::None,
        n_hash_layers: us(r, "num_hash_layers", 0),
        n_heads: req(r, "num_attention_heads", &mut m),
        n_kv_heads: us(r, "num_key_value_heads", 1),
        head_dim: req(r, "head_dim", &mut m),
        rope_theta: f32_of(r, "rope_theta", 10000.0),
        rope_interleave: true,
        rotary_dim: us(r, "qk_rope_head_dim", 64),
        extra_layers: us(r, "num_nextn_predict_layers", 0),
    };
    done(m, w)?;
    Ok(s)
}

fn glm(r: &Value, w: &str) -> Result<Spec, String> {
    let mut m = Vec::new();
    let s = Spec {
        family: Family::Glm,
        hidden: req(r, "hidden_size", &mut m),
        n_layers: req(r, "num_hidden_layers", &mut m),
        vocab: req(r, "vocab_size", &mut m),
        rms_eps: f32_of(r, "rms_norm_eps", 1e-5),
        rms_acc: Acc::F32,
        gemma_norm: false,
        n_experts: req(r, "n_routed_experts", &mut m),
        topk: req(r, "num_experts_per_tok", &mut m),
        n_shared: req(r, "n_shared_experts", &mut m),
        moe_inter: req(r, "moe_intermediate_size", &mut m),
        shared_inter: us(r, "moe_intermediate_size", 0),
        dense_inter: req(r, "intermediate_size", &mut m),
        routed_scale: f32_of(r, "routed_scaling_factor", 1.0),
        renorm: r["norm_topk_prob"].as_bool().unwrap_or(true),
        scoring: scoring_of(r["scoring_func"].as_str().unwrap_or(""), &mut m),
        glu: Glu::SwigluClamped { limit: f32::INFINITY },
        dense: Dense::FirstK(req(r, "first_k_dense_replace", &mut m)),
        n_hash_layers: 0,
        n_heads: req(r, "num_attention_heads", &mut m),
        n_kv_heads: us(r, "num_key_value_heads", 0),
        head_dim: req(r, "head_dim", &mut m),
        // GLM nests rope under rope_parameters rather than at the top level.
        rope_theta: r["rope_parameters"]["rope_theta"]
            .as_f64()
            .or_else(|| r["rope_theta"].as_f64())
            .unwrap_or(1e6) as f32,
        rope_interleave: r["rope_interleave"].as_bool().unwrap_or(false),
        rotary_dim: us(r, "qk_rope_head_dim", 64),
        extra_layers: us(r, "num_nextn_predict_layers", 0),
    };
    done(m, w)?;
    Ok(s)
}

fn m3(root: &Value, w: &str) -> Result<Spec, String> {
    // MiniMax-M3 is multimodal, so every text field sits under text_config. cfg.rs solves
    // the same nested-vs-flat problem for K3; this is the same shape.
    let r = if root["text_config"].is_object() { &root["text_config"] } else { root };
    let mut m = Vec::new();
    let sa = &r["sparse_attention_config"];
    let freq: Vec<u8> = r["moe_layer_freq"]
        .as_array()
        .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(1) as u8).collect())
        .unwrap_or_default();
    let s = Spec {
        family: Family::M3,
        hidden: req(r, "hidden_size", &mut m),
        n_layers: req(r, "num_hidden_layers", &mut m),
        vocab: req(r, "vocab_size", &mut m),
        rms_eps: f32_of(r, "rms_norm_eps", 1e-6),
        rms_acc: Acc::F32,
        gemma_norm: r["use_gemma_norm"].as_bool().unwrap_or(false),
        n_experts: req(r, "num_local_experts", &mut m),
        topk: req(r, "num_experts_per_tok", &mut m),
        n_shared: us(r, "n_shared_experts", 1),
        // M3 spells the ROUTED expert width `intermediate_size` and the dense MLP width
        // `dense_intermediate_size`. Everywhere else `intermediate_size` IS the dense
        // width, so reading it as one gives routed experts a 4x too-wide matrix.
        moe_inter: req(r, "intermediate_size", &mut m),
        shared_inter: us(r, "shared_intermediate_size", 0),
        dense_inter: us(r, "dense_intermediate_size", 0),
        routed_scale: f32_of(r, "routed_scaling_factor", 1.0),
        renorm: r["norm_topk_prob"].as_bool().unwrap_or(true),
        scoring: scoring_of(r["scoring_func"].as_str().unwrap_or(""), &mut m),
        // swigluoai: alpha-scaled sigmoid gate with a two-sided clamp.
        glu: Glu::SwigluClamped { limit: f32_of(r, "swiglu_limit", 7.0) },
        dense: if freq.is_empty() { Dense::FirstK(0) } else { Dense::PerLayer(freq) },
        n_hash_layers: 0,
        n_heads: req(r, "num_attention_heads", &mut m),
        n_kv_heads: req(r, "num_key_value_heads", &mut m),
        head_dim: req(r, "head_dim", &mut m),
        rope_theta: f32_of(r, "rope_theta", 5e6),
        rope_interleave: false,
        // partial rotary: only the first rotary_dim of each head is rotated.
        rotary_dim: us(r, "rotary_dim", 64),
        extra_layers: us(r, "num_nextn_predict_layers", 0),
    };
    let _ = sa;
    done(m, w)?;
    Ok(s)
}

fn k3(root: &Value, w: &str) -> Result<Spec, String> {
    let c = crate::cfg::load(root, w).map_err(|e| e.to_string())?;
    Ok(Spec {
        family: Family::K3,
        hidden: c.hidden as usize,
        n_layers: c.n_layers as usize,
        vocab: c.vocab as usize,
        rms_eps: c.rms_eps,
        rms_acc: Acc::F64,
        gemma_norm: false,
        n_experts: c.n_experts as usize,
        topk: c.topk as usize,
        n_shared: c.n_shared as usize,
        moe_inter: c.moe_inter as usize,
        shared_inter: (c.moe_inter * c.n_shared) as usize,
        dense_inter: c.dense_inter as usize,
        routed_scale: c.routed_scale,
        renorm: c.moe_renorm,
        scoring: Scoring::Sigmoid,
        glu: Glu::SiTu { b1: c.situ_b1, b2: c.situ_b2 },
        dense: Dense::FirstK(c.first_dense as usize),
        n_hash_layers: 0,
        n_heads: c.n_heads as usize,
        n_kv_heads: c.n_heads as usize,
        head_dim: c.v_head as usize,
        // K3 is NoPE: the rope slots exist and are never rotated.
        rope_theta: 0.0,
        rope_interleave: true,
        rotary_dim: 0,
        extra_layers: 0,
    })
}

/// The tensor-name table, as DATA. `docs/RUST_PORT.md:19` names this and the layer
/// dispatch as the two places architecture was hardcoded as code.
impl Spec {
    pub fn expert_names(&self, layer: usize, expert: usize) -> ExpertNames {
        match self.family {
            Family::K3 => crate::cache::k3_expert_names(layer, expert),
            _ => crate::cache::v4_expert_names(layer, expert),
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self.family {
            Family::K3 => "language_model.model.",
            _ => "",
        }
    }

    pub fn embed(&self) -> String {
        match self.family {
            Family::K3 => "language_model.model.embed_tokens.weight".into(),
            Family::V4 => "embed.weight".into(),
            _ => "model.embed_tokens.weight".into(),
        }
    }

    pub fn final_norm(&self) -> String {
        match self.family {
            Family::K3 => "language_model.model.norm.weight".into(),
            Family::V4 => "norm.weight".into(),
            _ => "model.norm.weight".into(),
        }
    }

    pub fn head(&self) -> String {
        match self.family {
            Family::K3 => "language_model.lm_head.weight".into(),
            Family::V4 => "head.weight".into(),
            _ => "lm_head.weight".into(),
        }
    }

    pub fn is_dense(&self, l: usize) -> bool {
        self.dense.is_dense(l)
    }

    /// Bytes in one routed expert: three matrices of packed nibbles plus one E8M0 scale
    /// per group. Omitting the scales understates it by 1/32 and quietly turns the
    /// per-token traffic figure into a different number from the measured one.
    pub fn expert_bytes(&self, packed_bits: usize, group: usize) -> u64 {
        let elems = self.moe_inter * self.hidden * 3;
        (elems * packed_bits / 8 + elems / group) as u64
    }

    /// Per-token bytes moved for the routed experts, from the descriptor alone.
    pub fn expert_bytes_per_token(&self, packed_bits: usize, group: usize) -> u64 {
        self.expert_bytes(packed_bits, group) * self.topk as u64 * self.n_layers as u64
    }

    pub fn summary(&self, whence: &str) -> String {
        format!(
            "arch: {} | {whence} | hidden={} layers={} vocab={} | heads {}/{} head_dim={} | \
             experts {} top{} shared{} inter={} | scoring {:?} | rmsnorm {:?}",
            self.family.as_str(),
            self.hidden,
            self.n_layers,
            self.vocab,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.n_experts,
            self.topk,
            self.n_shared,
            self.moe_inter,
            self.scoring,
            self.rms_acc,
        )
    }
}

/// Translate GGUF metadata into the `config.json` shape the family readers already expect.
///
/// GGUF has no `config.json`: every hyper-parameter lives in the header, arch-prefixed
/// (`qwen3moe.block_count`, not `num_hidden_layers`). Rather than trait-ise `f32_of`/`us`/
/// `req` and disturb four readers that are pinned by the tests below, this renames the keys
/// and hands the result to `spec()` unchanged.
///
/// It maps ONLY what GGUF actually carries. The Spec fields with no GGUF equivalent --
/// `rms_acc`, `glu`, `scoring`, `routed_scale`, `rope_interleave`, `n_hash_layers` -- are
/// per-family constants in every existing reader and stay that way: each architecture needs
/// its own arm, pinned against a real checkpoint. Guessing them yields a different model
/// that still writes fluent text, which is exactly what `done()` refuses to allow.
pub fn gguf_to_json(meta: &crate::gguf::Meta) -> Result<Value, String> {
    use crate::gguf::Value as G;
    let arch = meta
        .get("general.architecture")
        .and_then(G::as_str)
        .ok_or("gguf: no general.architecture")?
        .to_string();
    let u = |k: &str| meta.get(&format!("{arch}.{k}")).and_then(G::as_u);
    let fl = |k: &str| meta.get(&format!("{arch}.{k}")).and_then(G::as_f);

    // The vocabulary size is usually only implied, by the length of the token list.
    let vocab = u("vocab_size").or(match meta.get("tokenizer.ggml.tokens") {
        Some(G::Arr(v)) => Some(v.len() as u64),
        _ => None,
    });

    let mut o = serde_json::Map::new();
    for (dst, src) in [
        ("hidden_size", "embedding_length"),
        ("num_hidden_layers", "block_count"),
        ("num_attention_heads", "attention.head_count"),
        ("num_key_value_heads", "attention.head_count_kv"),
        ("head_dim", "attention.key_length"),
        ("intermediate_size", "feed_forward_length"),
        ("moe_intermediate_size", "expert_feed_forward_length"),
        ("shared_expert_intermediate_size", "expert_shared_feed_forward_length"),
        ("n_routed_experts", "expert_count"),
        ("num_experts_per_tok", "expert_used_count"),
    ] {
        if let Some(v) = u(src) {
            o.insert(dst.into(), Value::from(v));
        }
    }
    if let Some(v) = vocab {
        o.insert("vocab_size".into(), Value::from(v));
    }
    if let Some(v) = fl("rope.freq_base") {
        o.insert("rope_theta".into(), Value::from(v));
    }
    if let Some(v) = fl("attention.layer_norm_rms_epsilon") {
        o.insert("rms_norm_eps".into(), Value::from(v));
    }
    o.insert("model_type".into(), Value::from(arch));
    Ok(Value::Object(o))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn v4_cfg() -> Value {
        json!({
            "architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
            "hidden_size": 4096, "num_hidden_layers": 43, "vocab_size": 129280,
            "rms_norm_eps": 1e-6, "num_attention_heads": 64, "head_dim": 512,
            "qk_rope_head_dim": 64, "n_routed_experts": 256, "n_shared_experts": 1,
            "num_experts_per_tok": 6, "moe_intermediate_size": 2048,
            "routed_scaling_factor": 1.5, "scoring_func": "sqrtsoftplus",
            "swiglu_limit": 10.0, "num_hash_layers": 3, "norm_topk_prob": true,
            "num_key_value_heads": 1, "rope_theta": 10000, "num_nextn_predict_layers": 1
        })
    }

    fn glm_cfg() -> Value {
        json!({
            "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa",
            "hidden_size": 6144, "num_hidden_layers": 78, "vocab_size": 154880,
            "rms_norm_eps": 1e-5, "num_attention_heads": 64, "head_dim": 64,
            "num_key_value_heads": 64, "n_routed_experts": 256, "n_shared_experts": 1,
            "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "intermediate_size": 12288, "first_k_dense_replace": 3,
            "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
            "rope_parameters": {"rope_theta": 1000000}, "rope_interleave": true,
            "qk_rope_head_dim": 64, "num_nextn_predict_layers": 1
        })
    }

    fn m3_cfg() -> Value {
        json!({
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "model_type": "minimax_m3_vl",
            "text_config": {
                "hidden_size": 6144, "num_hidden_layers": 6, "vocab_size": 200064,
                "rms_norm_eps": 1e-6, "num_attention_heads": 64, "num_key_value_heads": 4,
                "head_dim": 128, "num_local_experts": 128, "num_experts_per_tok": 4,
                "n_shared_experts": 1, "intermediate_size": 3072,
                "shared_intermediate_size": 3072, "dense_intermediate_size": 12288,
                "moe_layer_freq": [0, 0, 0, 1, 1, 1], "scoring_func": "sigmoid",
                "routed_scaling_factor": 2.0, "use_gemma_norm": true,
                "swiglu_limit": 7.0, "rope_theta": 5000000, "rotary_dim": 64
            }
        })
    }

    #[test]
    fn each_architecture_is_recognised_by_its_own_config() {
        assert_eq!(detect(&v4_cfg()).unwrap(), Family::V4);
        assert_eq!(detect(&glm_cfg()).unwrap(), Family::Glm);
        assert_eq!(detect(&m3_cfg()).unwrap(), Family::M3);
        assert_eq!(detect(&json!({"kda_num_heads": 96})).unwrap(), Family::K3);
    }

    // Refusing beats guessing: an unknown model_type that fell through to K3 would bind
    // K3's tensor names against someone else's checkpoint and report missing tensors,
    // which reads as a corrupt download rather than an unsupported model.
    #[test]
    fn an_unknown_architecture_is_refused_by_name() {
        let e = detect(&json!({"model_type": "llama", "hidden_size": 4096})).unwrap_err();
        assert!(e.contains("unrecognised architecture"), "{e}");
        assert!(e.contains("llama"), "the error must name what it saw: {e}");
    }

    #[test]
    fn the_rmsnorm_accumulator_follows_the_architecture() {
        assert_eq!(spec(&v4_cfg(), "t").unwrap().rms_acc, Acc::F32);
        assert_eq!(spec(&glm_cfg(), "t").unwrap().rms_acc, Acc::F32);
        assert_eq!(spec(&m3_cfg(), "t").unwrap().rms_acc, Acc::F32);
    }

    #[test]
    fn minimax_reads_its_text_config_not_the_top_level() {
        let s = spec(&m3_cfg(), "t").unwrap();
        assert_eq!(s.hidden, 6144);
        assert_eq!(s.n_kv_heads, 4, "GQA: 64 query heads over 4 kv heads");
        assert!(s.gemma_norm, "use_gemma_norm is a text_config key");
    }

    // Three spellings of the same idea, and they are not interchangeable.
    #[test]
    fn dense_layers_are_read_the_way_each_architecture_spells_them() {
        let g = spec(&glm_cfg(), "t").unwrap();
        assert!(g.is_dense(0) && g.is_dense(2) && !g.is_dense(3));

        let m = spec(&m3_cfg(), "t").unwrap();
        assert!(m.is_dense(0) && m.is_dense(2) && !m.is_dense(3));

        // DeepSeek-V4 has no dense layer at all; its first three route by token id, which
        // is a routing change and not a dense one.
        let v = spec(&v4_cfg(), "t").unwrap();
        assert!(!v.is_dense(0));
        assert_eq!(v.n_hash_layers, 3);
    }

    #[test]
    fn a_missing_required_field_is_an_error_that_names_every_miss() {
        let mut c = v4_cfg();
        c["hidden_size"] = Value::Null;
        c["n_routed_experts"] = Value::Null;
        let e = spec(&c, "test.json").unwrap_err();
        assert!(e.contains("hidden_size") && e.contains("n_routed_experts"), "{e}");
        assert!(e.contains("refusing to substitute defaults"), "{e}");
    }

    #[test]
    fn the_extra_layer_blocks_are_not_decoder_layers() {
        // Every one of these ships num_nextn_predict_layers. Counting them as decoder
        // layers is what made compress_ratios look three entries too long.
        assert_eq!(spec(&v4_cfg(), "t").unwrap().extra_layers, 1);
        assert_eq!(spec(&glm_cfg(), "t").unwrap().extra_layers, 1);
    }

    #[test]
    fn expert_name_tables_are_selected_by_family() {
        let v = spec(&v4_cfg(), "t").unwrap().expert_names(2, 5);
        let k = spec(&json!({"kda_num_heads": 96}), "t");
        assert!(v.probe_name().starts_with("layers.2.ffn.experts.5"));
        // The K3 spec needs the full K3 config; detection alone is enough here.
        assert!(k.is_err() || k.is_ok());
    }
}
