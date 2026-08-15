// SPDX-License-Identifier: Apache-2.0
//
// What distinguishes one MoE family from another, as data.
//
// THE OBSERVATION THIS FILE IS BUILT ON
//     Every MoE architecture llama.cpp supports -- llama/mixtral, qwen2moe, qwen3moe,
//     glm4moe, olmoe, granitemoe, dbrx, phimoe, ernie4_5-moe, bailingmoe, hunyuan-moe,
//     smallthinker -- stores its routed experts under the SAME four names:
//
//         blk.N.ffn_gate_inp        the router
//         blk.N.ffn_{gate,up,down}_exps   stacked 3-D expert tensors
//
//     plus an optional `_shexp` set for a shared expert. So the expensive half of this
//     engine already serves all of them: the GGUF container reader, the k-quant kernels,
//     the stacked expert addressing, the streaming cache with its sweep-aware eviction and
//     GPU victim tier, and the batched routing union.
//
//     What actually differs between families is small and mostly boolean: whether queries
//     and keys get a per-head norm, whether there is a shared expert, whether the router
//     renormalises its top-k, which name the post-attention norm goes by. Encoding that as
//     a TABLE rather than as one module per family is what makes "support another MoE
//     model" a data change.
//
// WHAT THIS TABLE DOES NOT CLAIM
//     An entry here means the geometry and tensor layout are understood -- NOT that the
//     forward pass has been verified against a reference. Those are different claims, and
//     conflating them is how a model loads, runs, and is quietly wrong. `verified` says
//     which is which, and `capability.rs` reports only the verified ones as runnable.

/// How one MoE family differs from the common shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    /// `general.architecture`, verbatim, and the prefix its hparams use.
    pub name: &'static str,
    /// Per-head RMSNorm on q and k before rope (`attn_q_norm` / `attn_k_norm`).
    pub qk_norm: bool,
    /// A shared expert that runs for every token alongside the routed ones.
    pub shared_expert: bool,
    /// Renormalise the top-k router probabilities to sum to 1.
    pub norm_topk: bool,
    /// The name the pre-FFN norm goes by. Most say `ffn_norm`; the Qwen3.5/3.6 line says
    /// `post_attention_norm` for the same tensor.
    pub ffn_norm: &'static str,
    /// Query and its output gate fused into one projection, interleaved per head.
    pub fused_q_gate: bool,
    /// A hybrid stack mixing linear-attention and full-attention blocks.
    pub hybrid: bool,
    /// Multi-head LATENT attention: q and kv are projected through a low-rank bottleneck
    /// rather than to full per-head vectors, so the KV cache holds one shared row per
    /// token instead of per-head keys and values. Nothing about the common GQA block
    /// applies.
    pub mla: bool,
    /// No routed experts at all: one feed-forward per block, which every token runs.
    ///
    /// The streaming cache still serves it -- a dense FFN is one expert per layer that is
    /// always selected -- but nothing may assume `n_experts > 0` from an entry being here.
    pub dense: bool,
    /// Has THIS family's flag combination been diffed against a reference?
    ///
    /// Sharing the common block is necessary but not sufficient. `qwen3moe` exercises
    /// `qk_norm = true, shared_expert = false`; a family with the opposite flags runs
    /// different lines and needs its own fixture before it may be advertised.
    pub verified: bool,
}

const COMMON: Arch = Arch {
    name: "",
    qk_norm: false,
    shared_expert: false,
    norm_topk: true,
    ffn_norm: "ffn_norm",
    fused_q_gate: false,
    hybrid: false,
    mla: false,
    dense: false,
    verified: false,
};

/// Every MoE family this build knows the shape of.
///
/// Ordered roughly by how close each is to the common case, so the diffs are legible.
pub const ARCHES: &[Arch] = &[
    // Mixtral and every llama-shaped MoE: plain GQA, no qk-norm, no shared expert.
    Arch { name: "llama", ..COMMON },
    // Qwen3 MoE: adds per-head q/k norms. No shared expert -- Qwen2 had one, Qwen3 dropped it.
    // Diffed against llama.cpp on a synthetic fixture (tools/make_tiny_qwen3moe.py):
    // result_norm -28.417568 vs -28.406910, f32 drift over 4 blocks. This is the entry
    // that verifies the COMMON BLOCK itself.
    Arch { name: "qwen3moe", qk_norm: true, verified: true, ..COMMON },
    // Qwen2 MoE: shared expert, no qk-norm.
    Arch { name: "qwen2moe", shared_expert: true, ..COMMON },
    // GLM-4 MoE: qk-norm and a shared expert.
    Arch { name: "glm4moe", qk_norm: true, shared_expert: true, ..COMMON },
    // OLMoE: qk-norm, no shared expert.
    Arch { name: "olmoe", qk_norm: true, ..COMMON },
    Arch { name: "granitemoe", ..COMMON },
    Arch { name: "phimoe", ..COMMON },
    Arch { name: "dbrx", ..COMMON },
    Arch { name: "ernie4_5-moe", shared_expert: true, ..COMMON },
    Arch { name: "bailingmoe", shared_expert: true, ..COMMON },
    Arch { name: "hunyuan-moe", shared_expert: true, ..COMMON },
    // The two with a verified forward pass. Both needed their own block implementation:
    // deepseek2/V4 for MLA plus Hyper-Connections, qwen35moe for the gated-delta-net
    // hybrid. Everything above them is the common shape and shares one block.
    Arch {
        name: "qwen35moe",
        qk_norm: true,
        shared_expert: true,
        ffn_norm: "post_attention_norm",
        fused_q_gate: true,
        hybrid: true,
        verified: true,
        ..COMMON
    },
    Arch { name: "deepseek_v4", shared_expert: true, mla: true, verified: true, ..COMMON },
    // The DENSE sibling of qwen35moe, and the reason this table's name is now slightly
    // wrong: Qwen3.8-27B has no experts at all. It is here because everything else about it
    // -- the hybrid stack, the fused q/gate, the tensor names -- is qwen35moe's, so the
    // question "what shape is this family?" has the same answer and one block serves both.
    // Diffed at ratio 3 against llama.cpp's `qwen35.cpp`: result_output +0.179502 vs our
    // +0.179739 (tests/tools/make_tiny_qwen35.py --dense --ratio 3).
    Arch {
        name: "qwen35",
        qk_norm: true,
        ffn_norm: "post_attention_norm",
        fused_q_gate: true,
        hybrid: true,
        dense: true,
        verified: true,
        ..COMMON
    },
];

pub fn find(name: &str) -> Option<&'static Arch> {
    ARCHES.iter().find(|a| a.name == name)
}

/// Architectures whose forward pass has been diffed against a reference.
pub fn verified() -> impl Iterator<Item = &'static Arch> {
    ARCHES.iter().filter(|a| a.verified)
}

impl Arch {
    /// Can the common GQA-plus-MoE block serve this family, or does it need its own?
    ///
    /// Hybrid, fused-gate and MLA are each a different BLOCK rather than a variation on
    /// the common one: alternating linear and full attention, fusing the output gate into
    /// the query projection, and projecting through a latent bottleneck all change the
    /// shape of the computation, not just a flag inside it.
    pub fn uses_common_block(&self) -> bool {
        !self.hybrid && !self.fused_q_gate && !self.mla
    }

    /// The expert tensor names, which are shared across every family in the table. This is
    /// the whole reason one engine can serve all of them.
    pub fn expert_names(&self, layer: usize) -> [String; 3] {
        [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's whole value is that the expert side is common. If a family ever needs
    /// different expert names, the streaming cache stops serving it and this must be the
    /// place that says so.
    #[test]
    fn every_family_shares_the_expert_tensor_names() {
        let want = crate::cache::gguf_expert_src(7, 0);
        for a in ARCHES {
            let n = a.expert_names(7);
            match &want {
                crate::cache::ExpertSrc::Stacked { names, .. } => {
                    assert_eq!(&n, names, "{} must use the shared expert layout", a.name);
                }
                _ => panic!("gguf expert source must be stacked"),
            }
        }
    }

    /// "We know the shape" and "we have checked the arithmetic" are different claims.
    /// Only the second may be reported as runnable.
    #[test]
    fn only_architectures_with_a_reference_diff_are_marked_verified() {
        let v: Vec<&str> = verified().map(|a| a.name).collect();
        assert_eq!(
            v,
            vec!["qwen3moe", "qwen35moe", "deepseek_v4", "qwen35"],
            "diffed against a reference"
        );
        // Sharing the common block is NOT verification: these run different lines of it.
        for n in ["llama", "qwen2moe", "glm4moe"] {
            let a = find(n).unwrap();
            assert!(a.uses_common_block() && !a.verified,
                    "{n} shares the block but its flag combination is untested");
        }
        // And the capability report must agree, or a model would be advertised on the
        // strength of a table entry alone.
        for a in ARCHES {
            assert_eq!(
                a.verified,
                crate::capability::RUNS.contains(&a.name),
                "{} disagrees between the arch table and the capability report",
                a.name
            );
        }
    }

    /// The two verified families are exactly the ones that needed their own block; every
    /// other entry is a flag-difference on the common one.
    #[test]
    fn the_common_block_covers_everything_except_the_two_exotic_stacks() {
        assert!(!find("qwen35moe").unwrap().uses_common_block(), "hybrid + fused gate");
        assert!(!find("deepseek_v4").unwrap().uses_common_block(), "MLA");
        for n in ["llama", "qwen3moe", "qwen2moe", "glm4moe", "olmoe", "dbrx"] {
            assert!(find(n).unwrap().uses_common_block(), "{n} is the common shape");
        }
    }

    /// Family differences must actually differ, or the table is decoration.
    #[test]
    fn the_table_records_real_differences() {
        // Qwen3 dropped Qwen2's shared expert and added per-head q/k norms.
        let (q2, q3) = (find("qwen2moe").unwrap(), find("qwen3moe").unwrap());
        assert!(q2.shared_expert && !q2.qk_norm);
        assert!(!q3.shared_expert && q3.qk_norm);
        // Mixtral is the plainest of all.
        let l = find("llama").unwrap();
        assert!(!l.qk_norm && !l.shared_expert && l.uses_common_block());
        // Qwen3.5/3.6 call the pre-FFN norm something else.
        assert_eq!(find("qwen35moe").unwrap().ffn_norm, "post_attention_norm");
        assert_eq!(q3.ffn_norm, "ffn_norm");
    }

    #[test]
    fn an_unknown_architecture_is_not_silently_accepted() {
        assert!(find("mamba").is_none());
        assert!(find("").is_none());
    }
}

#[cfg(test)]
mod dense_tests {
    use super::*;

    /// `qwen35` is the dense sibling, and the table has to say so without implying it is
    /// somehow an MoE with zero experts. Anything reading `dense` must not then go on to
    /// index a router.
    #[test]
    fn the_dense_family_is_marked_dense_and_shares_the_hybrid_shape() {
        let d = find("qwen35").expect("qwen35 is in the table");
        assert!(d.dense, "Qwen3.8-27B has no routed experts at all");
        assert!(!d.uses_common_block(), "hybrid plus fused q/gate needs its own block");
        // It is the SAME stack as qwen35moe everywhere except the feed-forward, which is
        // the claim that lets one block implementation serve both.
        let m = find("qwen35moe").unwrap();
        assert_eq!((d.hybrid, d.fused_q_gate, d.qk_norm), (m.hybrid, m.fused_q_gate, m.qk_norm));
        assert_eq!(d.ffn_norm, m.ffn_norm, "both call it post_attention_norm");
        // And it differs in exactly the expected places.
        assert!(!d.shared_expert && m.shared_expert, "a dense FFN needs no shared expert");
        assert!(!m.dense, "the MoE sibling is not dense");
    }

    /// Every other family in the table is routed. If a second dense one is added, it must
    /// be a deliberate act -- the streaming cache addresses the two differently
    /// (`gguf_dense_ffn_src` vs `gguf_expert_src`) and picking the wrong one is silent.
    #[test]
    fn qwen35_is_the_only_dense_entry() {
        let dense: Vec<&str> = ARCHES.iter().filter(|a| a.dense).map(|a| a.name).collect();
        assert_eq!(dense, vec!["qwen35"]);
    }
}
