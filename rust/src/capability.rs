// SPDX-License-Identifier: Apache-2.0
//
// "Can this build run that model, and if not, exactly what is missing?"
//
// WHY THIS EXISTS
//     Adding a model has meant reading its header by hand and knowing which of half a
//     dozen quirks apply. That does not scale past one person. Worse, the failure mode
//     without it is silence: a filename says `Q4_K_M`, the file actually holds 132 Q5_0
//     tensors, and the only reason anyone noticed was a manual check. Twice more the same
//     day: a `Q2_K` build contained no Q2_K at all, and a config claimed
//     `num_nextn_predict_layers: 1` while three MTP stages shipped.
//
//     So the engine has to answer the question itself, from the bytes, and say what it
//     cannot do rather than discovering it mid-generation. Every blocker below names a
//     concrete missing capability, never "unsupported".
//
// THE RULE THIS ENCODES
//     A tensor type in the block table can have its OFFSETS computed; that is not the same
//     as being decodable. Scanning a file proves its layout is understood, not that its
//     weights can be read. Those two are reported separately because conflating them is
//     how you get a model that loads and produces nonsense.

use std::path::{Path, PathBuf};

use crate::st::{Dtype, St};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Safetensors,
    Gguf,
}

impl Format {
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Safetensors => "safetensors",
            Format::Gguf => "gguf",
        }
    }
}

pub struct Report {
    pub path: PathBuf,
    pub format: Format,
    /// `general.architecture` (GGUF) or `model_type` (safetensors), verbatim.
    pub arch: String,
    pub files: usize,
    pub n_tensors: usize,
    pub bytes: u64,
    /// Routed-expert bytes, which stream, versus everything else, which stays resident.
    /// The split decides whether a model is bandwidth-bound or fits.
    pub expert_bytes: u64,
    /// (dtype, count, bytes, has_kernel) over every tensor. The flag is the whole
    /// point: a type can be scannable without being decodable.
    pub dtypes: Vec<(&'static str, usize, u64, bool)>,
    /// Why this build cannot run it. Empty means it can.
    pub blockers: Vec<String>,
    pub notes: Vec<String>,
}

/// Does a matmul or dequantiser exist for this type, as opposed to merely a block-size
/// entry that lets its offsets be computed?
pub fn has_kernel(d: Dtype) -> bool {
    match d {
        // Read directly, or widened by st.rs.
        Dtype::F32 | Dtype::F16 | Dtype::Bf16 | Dtype::U8 | Dtype::I8 | Dtype::I8R
        | Dtype::I64 | Dtype::F8E4M3 | Dtype::F8E8M0 => true,
        // k-quants with both a dequantiser and a fused matmul, each verified bit-identical
        // against llama.cpp's reference on real weights.
        Dtype::Q3K | Dtype::Q4K | Dtype::Q5K | Dtype::Q6K => true,
        // Scannable, not decodable: the block table gives their size so a mixed file's
        // offsets resolve, but nothing can read their weights yet.
        Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 | Dtype::Q8_0
        | Dtype::Q2K | Dtype::IQ4NL => false,
    }
}

/// Architectures with a VERIFIED forward pass, not merely a readable container.
///
/// `qwen35moe` joined on 2026-08-12, when every block was diffed against llama.cpp's
/// reference trace on a synthetic fixture and the end-to-end logits matched. Leaving it out
/// was not conservative: `registry` hides blocked models from `/v1/models` and `/api/tags`,
/// and `rustlm serve` will not auto-select one, so a stale entry here makes a working
/// engine unreachable.
pub const RUNS: [&str; 4] = ["deepseek_v4", "qwen35moe", "qwen3moe", "qwen35"];

/// A tensor that STREAMS off disk, rather than a trunk weight that stays resident.
///
/// For a Mixture-of-Experts model those are the routed experts, named the same way in both
/// ecosystems. For the dense `qwen35` they are the per-layer feed-forward, which this
/// engine streams for exactly the same reason -- it is 62% of Qwen3.8-27B and cannot be
/// resident on the machines this is built for.
///
/// The architecture has to be part of the question. `blk.N.ffn_gate.weight` is a streamed
/// tensor in a dense qwen35 and does not exist at all in an MoE one, and a name-only rule
/// would report a dense checkpoint as 100% resident -- which is the opposite of true and
/// would make the memory advice in `inspect` wrong.
fn streams(name: &str, arch: &str) -> bool {
    if name.contains(".experts.") || name.contains("_exps.") {
        return true;
    }
    arch == "qwen35"
        && name.starts_with("blk.")
        && ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
            .iter()
            .any(|t| name.ends_with(t))
}

/// True for a tensor in a trailing MTP/next-token-prediction block that the runner excludes
/// (`block_count` counts it, `n_layers` drops it -- see `qwen35::Cfg`). Its weights are never
/// loaded, so their decodability must not count against the verdict: Qwen3.8-27B ships its
/// `blk.64.nextn.eh_proj` as Q8_0, and without this it would be reported as unrunnable over a
/// tensor the forward pass never touches.
fn excluded_nextn(name: &str, first_excluded: usize) -> bool {
    name.strip_prefix("blk.")
        .and_then(|r| r.split('.').next())
        .and_then(|i| i.parse::<usize>().ok())
        .is_some_and(|i| i >= first_excluded)
}

impl Report {
    pub fn inspect(path: &Path) -> Result<Report, String> {
        let dir = if path.is_dir() { path.to_path_buf() } else {
            path.parent().map(Path::to_path_buf).ok_or("path has no parent directory")?
        };
        let st = St::open(&dir).map_err(|e| e.to_string())?;
        let format = if st.meta.is_some() { Format::Gguf } else { Format::Safetensors };
        // Which tensors stream depends on the architecture, so it has to be known before
        // the accounting loop rather than after it.
        let gguf_arch: String = st
            .meta
            .as_ref()
            .and_then(|m| m.get("general.architecture"))
            .and_then(crate::gguf::Value::as_str)
            .unwrap_or("")
            .to_string();

        // First block index the runner EXCLUDES: block_count minus the MTP/nextn prediction
        // layers. Tensors at or past it are the discarded head and must not affect the
        // decodability verdict, only the true file size.
        let first_excluded = st
            .meta
            .as_ref()
            .and_then(|m| {
                let g = |k: &str| m.get(&format!("{gguf_arch}.{k}")).and_then(crate::gguf::Value::as_u);
                Some(g("block_count")?.saturating_sub(g("nextn_predict_layers").unwrap_or(0)) as usize)
            })
            .unwrap_or(usize::MAX);

        let mut dtypes: Vec<(&'static str, usize, u64, bool)> = Vec::new();
        let (mut bytes, mut expert_bytes) = (0u64, 0u64);
        let mut nextn_skipped = 0usize;
        for t in &st.tensors {
            bytes += t.nbytes as u64; // the real file size counts every tensor
            if streams(&t.name, &gguf_arch) {
                expert_bytes += t.nbytes as u64;
            }
            // The excluded MTP head is never loaded, so it does not gate runnability.
            if excluded_nextn(&t.name, first_excluded) {
                nextn_skipped += 1;
                continue;
            }
            let n = t.dtype.name();
            match dtypes.iter_mut().find(|(k, ..)| *k == n) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += t.nbytes as u64;
                }
                None => dtypes.push((n, 1, t.nbytes as u64, has_kernel(t.dtype))),
            }
        }
        dtypes.sort_by_key(|(_, c, ..)| std::cmp::Reverse(*c));

        let mut blockers = Vec::new();
        let mut notes = Vec::new();
        if nextn_skipped > 0 {
            notes.push(format!(
                "{nextn_skipped} tensor(s) belong to the trailing MTP/nextn head (blocks >= \
                 {first_excluded}); the runner drops them, so they do not gate runnability"
            ));
        }

        // Undecodable types are the commonest blocker, and the count matters: one stray
        // tensor is a different problem from half the model.
        for (name, count, b, ok) in &dtypes {
            if !ok {
                blockers.push(format!(
                    "{count} tensor(s) are {name} ({:.2} GB) and this build has no kernel \
                     for it -- their offsets resolve but their weights cannot be read",
                    *b as f64 / 1e9
                ));
            }
        }

        // Architecture, from whichever place the format keeps it.
        let arch = match &st.meta {
            Some(_) => gguf_arch.clone(),
            None => {
                let cfg = dir.join("config.json");
                match crate::arch::detect_file(&cfg) {
                    Ok(f) => f.as_str().to_string(),
                    Err(e) => {
                        blockers.push(e.lines().next().unwrap_or("unreadable config").into());
                        String::new()
                    }
                }
            }
        };
        if format == Format::Gguf {
            match &st.meta {
                Some(m) => match crate::arch::gguf_to_json(m) {
                    Ok(_) => notes.push(format!(
                        "gguf metadata maps to a config for architecture {arch:?}"
                    )),
                    Err(e) => blockers.push(e),
                },
                None => blockers.push("gguf file carried no metadata".into()),
            }
            // Known-but-unported is a different answer from unknown, and saying which
            // saves the reader guessing.
            if !arch.is_empty() && !RUNS.contains(&arch.as_str()) {
                // Say what is missing SPECIFICALLY. "Unsupported" tells a user nothing
                // about whether their model is one flag away or a research project.
                match crate::moearch::find(&arch) {
                    Some(a) if a.uses_common_block() => blockers.push(format!(
                        "architecture {arch:?} is the common GQA+MoE shape and its layout is \
                         known (qk_norm {}, shared_expert {}, ffn norm {:?}), but its \
                         forward pass has not been diffed against a reference yet. The \
                         container, quant kernels and expert streaming already work for it.",
                        a.qk_norm, a.shared_expert, a.ffn_norm
                    )),
                    Some(a) => blockers.push(format!(
                        "architecture {arch:?} needs its own block, not the common one ({}), \
                         and it has not been implemented",
                        if a.mla { "latent attention" }
                        else if a.hybrid { "hybrid linear/full attention" }
                        else { "a fused query-gate projection" }
                    )),
                    None => blockers.push(format!(
                        "architecture {arch:?} is not in the MoE family table at all -- its \
                         geometry has never been read here"
                    )),
                }
            }
        }

        if expert_bytes > 0 {
            notes.push(format!(
                "{:.1} GB of {} ({:.0}% of the checkpoint) -- these stream; \
                 the remaining {:.1} GB is resident",
                expert_bytes as f64 / 1e9,
                if arch == "qwen35" { "dense feed-forward" } else { "routed experts" },
                expert_bytes as f64 / bytes as f64 * 100.0,
                (bytes - expert_bytes) as f64 / 1e9
            ));
        }

        Ok(Report {
            path: dir,
            format,
            arch,
            files: st.paths.len(),
            n_tensors: st.tensors.len(),
            bytes,
            expert_bytes,
            dtypes,
            blockers,
            notes,
        })
    }

    pub fn runnable(&self) -> bool {
        self.blockers.is_empty()
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.path.display())?;
        writeln!(
            f,
            "  format     : {} ({} file(s), {} tensors, {:.2} GB)",
            self.format.as_str(),
            self.files,
            self.n_tensors,
            self.bytes as f64 / 1e9
        )?;
        writeln!(f, "  arch       : {}", if self.arch.is_empty() { "?" } else { &self.arch })?;
        writeln!(f, "  dtypes     :")?;
        for (n, c, b, ok) in &self.dtypes {
            writeln!(
                f,
                "      {n:<8} x{c:<5} {:>7.2} GB   {}",
                *b as f64 / 1e9,
                if *ok { "decodable" } else { "NO KERNEL -- offsets only" }
            )?;
        }
        for n in &self.notes {
            writeln!(f, "  note       : {n}")?;
        }
        if self.runnable() {
            writeln!(f, "  VERDICT    : runnable")?;
        } else {
            writeln!(f, "  VERDICT    : cannot run, {} blocker(s)", self.blockers.len())?;
            for b in &self.blockers {
                writeln!(f, "    - {b}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every dtype must be classified deliberately. A new variant added to `Dtype` without
    /// a decision here would otherwise default into whichever arm the compiler picked,
    /// and "we can read this" is exactly the claim that must never be made by accident.
    #[test]
    fn every_dtype_is_classified_and_the_k_quants_are_the_readable_ones() {
        assert!(has_kernel(Dtype::Q4K) && has_kernel(Dtype::Q5K));
        assert!(has_kernel(Dtype::Q6K) && has_kernel(Dtype::Q3K));
        assert!(has_kernel(Dtype::F32) && has_kernel(Dtype::F8E4M3));
        // Scannable but not decodable -- offsets resolve, weights do not.
        assert!(!has_kernel(Dtype::Q2K), "Q2_K has a block size but no dequantiser");
        assert!(!has_kernel(Dtype::IQ4NL), "IQ4_NL is a codebook format, not implemented");
        assert!(!has_kernel(Dtype::Q5_0) && !has_kernel(Dtype::Q8_0));
    }

    /// A model whose forward pass is verified must be reported runnable. This is not
    /// cosmetic: `registry` hides blocked models from `/v1/models` and `/api/tags`, and
    /// `rustlm serve` will not auto-select one, so a stale blocker makes a working engine
    /// unreachable.
    #[test]
    fn architectures_with_a_verified_forward_pass_are_not_blocked() {
        for arch in ["deepseek_v4", "qwen35moe", "qwen3moe"] {
            assert!(RUNS.contains(&arch), "{arch} has a forward pass and must not be blocked");
        }
        assert!(!RUNS.contains(&"llama"), "an unported architecture must still be blocked");
    }

    #[test]
    fn expert_tensors_are_told_apart_from_trunk_ones() {
        // Both naming conventions the engine reads.
        assert!(streams("layers.2.ffn.experts.137.w1.weight", "qwen35moe"), "safetensors");
        assert!(streams("blk.7.ffn_gate_exps.weight", "qwen35moe"), "gguf stacked");
        // The shared expert is resident, not routed, so it must NOT count as streaming.
        assert!(!streams("blk.7.ffn_gate_shexp.weight", "qwen35moe"), "shared expert is trunk");
        assert!(!streams("blk.0.attn_qkv.weight", "qwen35moe"));
        assert!(!streams("token_embd.weight", "qwen35moe"));

        // The dense architecture streams its feed-forward, and ONLY under that arch: the
        // same three names under an MoE arch would be a checkpoint we do not understand,
        // and counting them there would silently misreport the resident/streamed split.
        for t in ["blk.0.ffn_gate.weight", "blk.31.ffn_up.weight", "blk.7.ffn_down.weight"] {
            assert!(streams(t, "qwen35"), "{t} streams in a dense qwen35");
            assert!(!streams(t, "qwen35moe"), "{t} must not count under an MoE arch");
            assert!(!streams(t, "llama"), "{t} must not count for an unsupported arch");
        }
        // The routed names still stream under the dense arch check, since the rule is a
        // union and an arch mismatch there would be a corrupt file rather than a layout.
        assert!(streams("blk.7.ffn_gate_exps.weight", "qwen35"));
        // Near-misses that must NOT be swept in.
        assert!(!streams("blk.0.ffn_gate_shexp.weight", "qwen35"), "shared expert is trunk");
        assert!(!streams("blk.0.ffn_gate_inp.weight", "qwen35"), "the router is trunk");
        assert!(!streams("ffn_gate.weight", "qwen35"), "must be inside a block");
    }
}
