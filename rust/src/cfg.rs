// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::path::Path;

use serde_json::Value;

pub const MAX_TOPK: i32 = 64;

pub const MAX_FULL_ATTN: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Io(String),
    Parse(String),
    Missing { whence: String, keys: Vec<String> },
    Structural(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(m) | Error::Parse(m) | Error::Structural(m) => write!(f, "{m}"),
            Error::Missing { whence, keys } => {
                writeln!(
                    f,
                    "k3_cfg: {whence} is missing {} required field(s):",
                    keys.len()
                )?;
                let shown = keys.len().min(32);
                for k in &keys[..shown] {
                    writeln!(f, "    {k}")?;
                }
                if keys.len() > shown {
                    writeln!(f, "    ... and {} more", keys.len() - shown)?;
                }
                write!(
                    f,
                    "  refusing to substitute defaults: a config this reader cannot\n  \
                     fully understand would silently produce a DIFFERENT model."
                )
            }
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub hidden: i32,       // 7168
    pub n_layers: i32,     // 93
    pub vocab: i32,        // 163840
    pub rms_eps: f32,      // 1e-5

    pub kda_heads: i32,    // 96
    pub kda_head_dim: i32, // 128, and d_k == d_v
    pub conv_k: i32,       // 4, depthwise, causal, SiLU fused
    pub gate_lb: f32,      // -5.0, the decay lower bound

    pub n_heads: i32,      // 96
    pub q_lora: i32,       // 1536
    pub kv_lora: i32,      // 512
    pub qk_nope: i32,      // 128
    pub qk_rope: i32,      // 64, PRESENT BUT NEVER ROTATED
    pub v_head: i32,       // 128
    pub mla_out_gate: bool,

    pub n_experts: i32,    // 896
    pub topk: i32,         // 16
    pub n_shared: i32,     // 2, full width, added UNWEIGHTED
    pub latent: i32,       // 3584, the routed-expert width
    pub moe_inter: i32,    // 3072
    pub routed_scale: f32, // 1.0
    pub moe_renorm: bool,
    pub latent_norm: bool, // RMSNorm on the AGGREGATE, not per expert

    pub first_dense: i32,  // 1
    pub dense_inter: i32,  // 33792

    pub attn_res_block: i32, // 12

    pub situ_b1: f32,      // 4.0
    pub situ_b2: f32,      // 25.0

    pub full_attn: Vec<i32>,

    pub nested: bool,
}

impl Cfg {
    #[inline]
    pub fn is_mla(&self, layer: i32) -> bool {
        self.full_attn.contains(&(layer + 1))
    }

    #[inline]
    pub fn is_kda(&self, layer: i32) -> bool {
        !self.is_mla(layer)
    }

    #[inline]
    pub fn is_dense(&self, layer: i32) -> bool {
        layer < self.first_dense
    }

    pub fn summary(&self, whence: &str) -> String {
        format!(
            "config: {whence} ({} shape) | hidden={} layers={} vocab={} | {} MLA + {} KDA | \
             experts {} top{} shared{} | latent={}",
            if self.nested { "nested" } else { "flat" },
            self.hidden,
            self.n_layers,
            self.vocab,
            self.full_attn.len(),
            self.n_layers as usize - self.full_attn.len(),
            self.n_experts,
            self.topk,
            self.n_shared,
            self.latent,
        )
    }
}

struct Src<'a> {
    txt: Option<&'a Value>,
    lin: Option<&'a Value>,
    root: &'a Value,
    nested: bool,
    missing: Vec<String>,
}

impl<'a> Src<'a> {
    fn find(&self, primary: &str, alias: Option<&str>) -> Option<&'a Value> {
        for obj in [self.txt, self.lin, Some(self.root)].into_iter().flatten() {
            for name in [Some(primary), alias].into_iter().flatten() {
                if let Some(v) = obj.get(name) {
                    return Some(v);
                }
            }
        }
        None
    }

    fn miss(&mut self, name: &str) {
        self.missing.push(name.to_string());
    }

    fn i(&mut self, primary: &str, alias: Option<&str>) -> i32 {
        match self.find(primary, alias).and_then(Value::as_f64) {
            Some(n) => n as i32,
            None => {
                self.miss(primary);
                0
            }
        }
    }

    fn f(&mut self, primary: &str, alias: Option<&str>) -> f32 {
        match self.find(primary, alias).and_then(Value::as_f64) {
            Some(n) => n as f32,
            None => {
                self.miss(primary);
                0.0
            }
        }
    }

    fn b(&mut self, primary: &str, alias: Option<&str>, dflt: bool) -> bool {
        match self.find(primary, alias) {
            None => dflt,
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_f64().is_some_and(|v| v != 0.0),
            Some(_) => dflt,
        }
    }
}

pub fn load(root: &Value, whence: &str) -> Result<Cfg, Error> {
    let txt = root.get("text_config");
    let base = txt.unwrap_or(root);
    let mut s = Src {
        txt,
        lin: base.get("linear_attn_config"),
        root,
        nested: txt.is_some(),
        missing: Vec::new(),
    };

    let hidden = s.i("hidden_size", None);
    let n_layers = s.i("num_hidden_layers", None);
    let vocab = s.i("vocab_size", None);
    let rms_eps = s.f("rms_norm_eps", None);

    let kda_heads = s.i("num_heads", Some("kda_num_heads"));
    let kda_head_dim = s.i("head_dim", Some("kda_head_dim"));
    let conv_k = s.i("short_conv_kernel_size", None);
    let gate_lb = s.f("gate_lower_bound", None);

    let n_heads = s.i("num_attention_heads", None);
    let q_lora = s.i("q_lora_rank", None);
    let kv_lora = s.i("kv_lora_rank", None);
    let qk_nope = s.i("qk_nope_head_dim", None);
    let qk_rope = s.i("qk_rope_head_dim", None);
    let v_head = s.i("v_head_dim", None);
    let mla_out_gate = s.b("mla_use_output_gate", None, true);

    let n_experts = s.i("num_experts", None);
    let topk = s.i("num_experts_per_token", None);
    let n_shared = s.i("num_shared_experts", None);
    let latent = s.i("routed_expert_hidden_size", None);
    let moe_inter = s.i("moe_intermediate_size", None);
    let routed_scale = s.f("routed_scaling_factor", None);
    let moe_renorm = s.b("moe_renormalize", None, true);
    let latent_norm = s.b("latent_moe_use_norm", None, true);

    let first_dense = s.i("first_k_dense_replace", None);
    let dense_inter = s.i("intermediate_size", None);
    let attn_res_block = s.i("attn_res_block_size", None);

    let situ_b1 = s.f("activation_situ_beta", Some("situ_beta"));
    let situ_b2 = s.f("activation_situ_linear_beta", Some("situ_linear_beta"));

    let mut full_attn: Vec<i32> = Vec::new();
    match s.find("full_attn_layers", None).and_then(Value::as_array) {
        None => s.miss("full_attn_layers"),
        Some(a) if a.is_empty() => s.miss("full_attn_layers"),
        Some(a) if a.len() > MAX_FULL_ATTN => {
            return Err(Error::Structural(format!(
                "k3_cfg: {whence} lists {} full-attention layers, buffer holds {MAX_FULL_ATTN}",
                a.len()
            )))
        }
        Some(a) => {
            for (i, v) in a.iter().enumerate() {
                let Some(n) = v.as_f64() else {
                    return Err(Error::Structural(format!(
                        "k3_cfg: {whence} full_attn_layers[{i}] is not a number"
                    )));
                };
                full_attn.push(n as i32);
            }
        }
    }

    if !s.missing.is_empty() {
        return Err(Error::Missing {
            whence: whence.to_string(),
            keys: s.missing,
        });
    }

    let c = Cfg {
        hidden, n_layers, vocab, rms_eps,
        kda_heads, kda_head_dim, conv_k, gate_lb,
        n_heads, q_lora, kv_lora, qk_nope, qk_rope, v_head, mla_out_gate,
        n_experts, topk, n_shared, latent, moe_inter, routed_scale,
        moe_renorm, latent_norm,
        first_dense, dense_inter, attn_res_block,
        situ_b1, situ_b2,
        full_attn,
        nested: s.nested,
    };

    let bad = |m: String| Err(Error::Structural(m));

    if c.n_layers <= 0 || c.hidden <= 0 || c.vocab <= 0 {
        return bad(format!("k3_cfg: {whence} has non-positive layers/hidden/vocab"));
    }
    if c.full_attn.len() as i32 >= c.n_layers {
        return bad(format!(
            "k3_cfg: {whence} marks {} of {} layers as full attention, leaving no KDA layers",
            c.full_attn.len(),
            c.n_layers
        ));
    }
    for (i, &l) in c.full_attn.iter().enumerate() {
        if l < 1 || l > c.n_layers {
            return bad(format!(
                "k3_cfg: {whence} full_attn_layers[{i}] = {l} is outside 1..{} \
                 (the list is ONE-based)",
                c.n_layers
            ));
        }
    }
    if c.topk > MAX_TOPK {
        return bad(format!(
            "k3_cfg: {whence} selects top-{}, but this build supports at most {MAX_TOPK}\n  \
             (MAX_TOPK bounds the fixed-size routing arrays)",
            c.topk
        ));
    }
    if c.topk > c.n_experts {
        return bad(format!(
            "k3_cfg: {whence} selects {} of {} experts",
            c.topk, c.n_experts
        ));
    }
    if c.attn_res_block <= 0 {
        return bad(format!(
            "k3_cfg: {whence} has attn_res_block_size {}; layer_idx % 0 would divide by zero",
            c.attn_res_block
        ));
    }
    if c.conv_k < 1 {
        return bad(format!(
            "k3_cfg: {whence} has short_conv_kernel_size {}",
            c.conv_k
        ));
    }

    Ok(c)
}

pub fn load_file(path: &Path) -> Result<Cfg, Error> {
    let whence = path.display().to_string();
    let meta = std::fs::metadata(path).map_err(|e| Error::Io(format!("{whence}: {e}")))?;
    if meta.len() > (1 << 28) {
        return Err(Error::Io(format!(
            "{whence}: implausible config size {}",
            meta.len()
        )));
    }
    let txt = std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{whence}: {e}")))?;
    let root: Value =
        serde_json::from_str(&txt).map_err(|_| Error::Parse(format!("{whence}: not valid JSON")))?;
    load(&root, &whence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flat() -> Value {
        json!({
            "hidden_size": 128, "num_hidden_layers": 13, "vocab_size": 256,
            "rms_norm_eps": 1e-05,
            "kda_num_heads": 4, "kda_head_dim": 16,
            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
            "num_attention_heads": 4, "q_lora_rank": 64, "kv_lora_rank": 32,
            "qk_nope_head_dim": 24, "qk_rope_head_dim": 8, "v_head_dim": 16,
            "mla_use_output_gate": true,
            "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 2,
            "routed_expert_hidden_size": 64, "moe_intermediate_size": 48,
            "routed_scaling_factor": 1.0, "moe_renormalize": true,
            "latent_moe_use_norm": true,
            "first_k_dense_replace": 1, "intermediate_size": 96,
            "attn_res_block_size": 3,
            "situ_beta": 4.0, "situ_linear_beta": 25.0,
            "full_attn_layers": [4, 8, 12, 13],
        })
    }

    #[test]
    fn reads_the_flat_shape() {
        let c = load(&flat(), "inline").expect("flat config should load");
        assert!(!c.nested);
        assert_eq!(c.hidden, 128);
        assert_eq!(c.kda_heads, 4);
        assert_eq!(c.full_attn, vec![4, 8, 12, 13]);
    }

    #[test]
    fn reads_the_nested_shape_and_prefers_the_linear_block_for_num_heads() {
        let nested = json!({
            "text_config": {
                "hidden_size": 128, "num_hidden_layers": 13, "vocab_size": 256,
                "rms_norm_eps": 1e-05,
                "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
                "num_attention_heads": 4,
                "q_lora_rank": 64, "kv_lora_rank": 32,
                "qk_nope_head_dim": 24, "qk_rope_head_dim": 8, "v_head_dim": 16,
                "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 2,
                "routed_expert_hidden_size": 64, "moe_intermediate_size": 48,
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 1, "intermediate_size": 96,
                "attn_res_block_size": 3,
                "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
                "linear_attn_config": {
                    "num_heads": 7, "head_dim": 16,
                    "full_attn_layers": [4, 8, 12, 13],
                },
            }
        });
        let c = load(&nested, "inline").expect("nested config should load");
        assert!(c.nested);
        assert_eq!(c.kda_heads, 7, "linear_attn_config.num_heads must win over the root");
        assert_eq!(c.n_heads, 4, "MLA head count still comes from the LM level");
        assert_eq!(c.full_attn, vec![4, 8, 12, 13]);
    }

    #[test]
    fn a_missing_key_is_an_error_and_every_miss_is_reported() {
        let mut v = flat();
        let o = v.as_object_mut().unwrap();
        o.remove("situ_beta");
        o.remove("hidden_size");
        o.remove("num_experts");

        match load(&v, "inline") {
            Err(Error::Missing { keys, .. }) => {
                assert!(keys.iter().any(|k| k == "activation_situ_beta"));
                assert!(keys.iter().any(|k| k == "hidden_size"));
                assert!(keys.iter().any(|k| k == "num_experts"));
                assert_eq!(keys.len(), 3, "all misses reported together, not just the first");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_layer_map_is_a_missing_key_not_an_all_kda_model() {
        let mut v = flat();
        v["full_attn_layers"] = json!([]);
        match load(&v, "inline") {
            Err(Error::Missing { keys, .. }) => {
                assert_eq!(keys, vec!["full_attn_layers".to_string()])
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn optional_booleans_keep_their_defaults() {
        let mut v = flat();
        let o = v.as_object_mut().unwrap();
        o.remove("mla_use_output_gate");
        o.remove("moe_renormalize");
        o.remove("latent_moe_use_norm");
        let c = load(&v, "inline").expect("booleans are genuinely optional");
        assert!(c.mla_out_gate && c.moe_renorm && c.latent_norm);
    }

    #[test]
    fn numeric_booleans_are_accepted() {
        let mut v = flat();
        v["mla_use_output_gate"] = json!(0);
        assert!(!load(&v, "inline").unwrap().mla_out_gate);
    }

    #[test]
    fn structural_checks_reject_impossible_configs() {
        type Mutate = Box<dyn Fn(&mut Value)>;
        let cases: Vec<(&str, Mutate)> = vec![
            ("non-positive", Box::new(|v: &mut Value| v["hidden_size"] = json!(0))),
            ("leaving no KDA layers", Box::new(|v: &mut Value| {
                v["num_hidden_layers"] = json!(4)
            })),
            ("is outside 1..", Box::new(|v: &mut Value| {
                v["full_attn_layers"] = json!([0, 8, 12, 13])
            })),
            ("selects 9 of 8 experts", Box::new(|v: &mut Value| {
                v["num_experts_per_token"] = json!(9)
            })),
            ("divide by zero", Box::new(|v: &mut Value| {
                v["attn_res_block_size"] = json!(0)
            })),
            ("short_conv_kernel_size 0", Box::new(|v: &mut Value| {
                v["short_conv_kernel_size"] = json!(0)
            })),
        ];
        for (needle, mutate) in cases {
            let mut v = flat();
            mutate(&mut v);
            match load(&v, "inline") {
                Err(Error::Structural(m)) => {
                    assert!(m.contains(needle), "expected {needle:?} in {m:?}")
                }
                other => panic!("expected Structural containing {needle:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn topk_above_the_routing_array_bound_is_rejected() {
        let mut v = flat();
        v["num_experts"] = json!(1024);
        v["num_experts_per_token"] = json!(MAX_TOPK + 1);
        match load(&v, "inline") {
            Err(Error::Structural(m)) => assert!(m.contains("supports at most 64")),
            other => panic!("expected Structural, got {other:?}"),
        }
    }

    #[test]
    fn layer_predicates_are_one_based() {
        let c = load(&flat(), "inline").unwrap();
        assert!(c.is_mla(3) && c.is_mla(7) && c.is_mla(11) && c.is_mla(12));
        assert!(c.is_mla(11) && c.is_mla(12), "the last two layers are both MLA");
        assert!(!c.is_mla(0) && c.is_kda(0), "layer 0 is KDA");
        assert!(c.is_dense(0) && !c.is_dense(1), "only layer 0 is dense");
        let nmla = (0..c.n_layers).filter(|&l| c.is_mla(l)).count();
        assert_eq!(nmla, 4);
        assert_eq!((0..c.n_layers).filter(|&l| c.is_kda(l)).count(), 9);
    }
}
