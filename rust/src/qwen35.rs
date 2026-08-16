// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 / Qwen3.6 MoE (`qwen35moe`) -- a hybrid linear-attention / full-attention stack.
//
// SHAPE OF THE ARCHITECTURE, read from the checkpoint rather than assumed
//     Qwen3.6-35B-A3B: 40 blocks, hidden 2048, vocab 248320, 256 experts top-8.
//     `full_attention_interval: 4`, so blocks 3, 7, 11, ... are full attention and the
//     other thirty are linear attention -- a gated delta net, the same family as Kimi K3's
//     KDA, which is why `ops::shortconv` / `kda_decay` / `kda_step` already apply.
//
//     Qwen3.5-122B-A10B is the SAME architecture at 48 blocks and hidden 3072, so
//     everything here carries over; the 35B exists to make an iteration cost seconds.
//
// THE TWO BLOCK TYPES
//     attention  attn_q [8192, 2048]  <- TWICE 16 heads x 256, because q and its output
//                                        GATE are fused into one projection
//                attn_k / attn_v [512, 2048]   2 kv heads x 256, so GQA 8:1
//                attn_q_norm / attn_k_norm [256]  per-head-dim RMS, no learned gain scale
//     linear     attn_qkv [8192, 2048] fused, plus attn_gate and the ssm_* set
//
// WHAT IS IMPLEMENTED HERE SO FAR
//     The IO path only: embedding lookup, final norm, vocabulary head. That is deliberate
//     -- it is the smallest slice that exercises GGUF loading, the k-quant kernels and the
//     real 21 GB checkpoint together, so the plumbing is proven before forty layers of
//     arithmetic are stacked on top of it. The layer stack lands next, component by
//     component, each checked against a reference the way V4's attention was.

use crate::gguf::{self, Meta, Q4K_BLOCK, Q6K_BLOCK, QK_K};
use crate::ops::W;
use crate::st::{Dtype, St};

/// Every dimension this architecture needs, read from the checkpoint's own metadata.
///
/// WHY THIS IS A STRUCT AND NOT A PILE OF LOOKUPS
/// ```text
///     Two shapes here are indistinguishable from a plausible wrong guess. `attn_qkv` is
///     [8192, 2048] and 8192 decomposes as 16*128 + 16*128 + 32*128 -- q and k have SIXTEEN
///     heads, v has THIRTY-TWO, and reading it as three equal parts produces a model that
///     still emits fluent text. Likewise `attn_q` is [8192, 2048] in a full-attention block
///     and that is 2 * 16 * 256, because q and its output gate are fused.
/// ```
///
/// GEOMETRY, verified against Qwen3.6-35B-A3B's GGUF header
/// ```text
///     40 blocks, hidden 2048, vocab 248320, rms_eps 1e-6.
///     `full_attention_interval` 4, so blocks 3, 7, 11, ... are full attention (10 of them)
///     and the other thirty are linear attention -- a gated delta net.
///
///     full attention   16 heads x 256, 2 kv heads x 256 (GQA 8:1), rope over the first
///                      64 dims only, freq_base 1e7.
///     linear attention 16 q heads and 16 k heads x 128, 32 v heads x 128; conv kernel 4
///                      over all 8192 mixed channels; state 128x128 per v head.
///
///     Qwen3.5-122B-A10B is the SAME architecture at 48 blocks and hidden 3072, which is
///     why nothing here is hardcoded.
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Cfg {
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    /// Blocks where `(l + 1) % interval == 0` are full attention.
    pub full_attn_interval: usize,
    // -- full attention --
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Rope covers only the first `n_rot` of `head_dim`; the rest passes through.
    pub n_rot: usize,
    pub rope_base: f32,
    // -- linear attention (gated delta net) --
    /// Total value width; `d_inner / n_v_heads` is the per-head value dim.
    pub d_inner: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub d_state: usize,
    pub conv_kernel: usize,
    // -- feed-forward: routed experts, or one dense FFN --
    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    /// The dense sibling's FFN width (`qwen35.feed_forward_length`), or 0 for the MoE.
    ///
    /// `qwen35` and `qwen35moe` are the SAME stack -- identical tensor names for every
    /// attention and SSM weight, identical fused q/gate, identical mrope sections -- and
    /// they differ only in what sits after `post_attention_norm`: 256 routed experts of
    /// width 512, or one dense FFN of width 17408. So this is a field rather than a second
    /// `Cfg`, and `is_dense()` is the only branch the forward pass needs.
    pub dense_inter: usize,
    /// Extra trailing blocks that are NOT part of the stack.
    ///
    /// Qwen3.8-27B reports `block_count` 65 while its `layer_types` lists 64: block 64 is
    /// the multi-token-prediction head, carrying its own attention, its own FFN and an
    /// `nextn.eh_proj`. Running it as layer 65 would be wrong in the quietest possible way
    /// -- every tensor it asks for exists, so the loader would succeed and the model would
    /// simply be a different model. It is excluded from `n_layers` and kept here because a
    /// speculative drafter is exactly what a dense model on a slow disk wants.
    pub mtp_layers: usize,
}

impl Cfg {
    /// One dense FFN per block instead of a router and a bank of experts.
    pub fn is_dense(&self) -> bool {
        self.n_experts == 0
    }

    /// The FFN width actually in use, whichever kind this is.
    pub fn ffn_width(&self) -> usize {
        if self.is_dense() {
            self.dense_inter
        } else {
            self.moe_inter
        }
    }
}

impl Cfg {
    pub fn from_meta(m: &Meta) -> Result<Cfg, String> {
        let arch = m
            .get("general.architecture")
            .and_then(gguf::Value::as_str)
            .ok_or("no general.architecture")?;
        // `qwen35` is the DENSE sibling of `qwen35moe`, not a different architecture: same
        // hybrid stack, same tensor names, same fused q/gate, same mrope sections. Only the
        // feed-forward differs. Both are read here so that half is written once.
        if arch != "qwen35moe" && arch != "qwen35" {
            return Err(format!("{arch:?} is neither qwen35moe nor qwen35"));
        }
        let dense = arch == "qwen35";
        // Absent keys are an ERROR, never a default. A silently-defaulted head count or
        // rope base yields a model that loads, runs, and is wrong -- the exact failure
        // this codebase keeps rediscovering.
        let u = |k: &str| -> Result<usize, String> {
            m.get(&format!("{arch}.{k}"))
                .and_then(gguf::Value::as_u)
                .map(|v| v as usize)
                .ok_or_else(|| format!("{arch}.{k} is missing from the gguf metadata"))
        };
        let f = |k: &str| -> Result<f32, String> {
            m.get(&format!("{arch}.{k}"))
                .and_then(gguf::Value::as_f)
                .map(|v| v as f32)
                .ok_or_else(|| format!("{arch}.{k} is missing from the gguf metadata"))
        };
        // Optional ONLY where absence is meaningful: a dense build has no expert keys and
        // an MoE build has no `feed_forward_length`. Everything else stays mandatory.
        let opt = |k: &str| -> usize {
            m.get(&format!("{arch}.{k}")).and_then(gguf::Value::as_u).unwrap_or(0) as usize
        };
        let hidden = u("embedding_length")?;
        let d_inner = u("ssm.inner_size")?;
        let n_v_heads = u("ssm.time_step_rank")?;
        let head_dim = u("attention.key_length")?;
        if u("attention.value_length")? != head_dim {
            return Err("attention key_length and value_length differ, which this \
                        implementation does not handle".into());
        }
        if d_inner % n_v_heads != 0 {
            return Err(format!(
                "ssm.inner_size {d_inner} is not divisible by {n_v_heads} value heads"
            ));
        }
        // The vocabulary is not in the metadata under a dimension key; it is the row count
        // of the embedding, which the caller knows from the tensor.
        // The MTP head is a trailing block that `block_count` includes and `layer_types`
        // does not. Subtracting it is what keeps the stack 64 blocks rather than 65.
        let mtp_layers = opt("nextn_predict_layers");
        let block_count = u("block_count")?;
        let n_layers = block_count.checked_sub(mtp_layers).filter(|n| *n > 0).ok_or_else(|| {
            format!("block_count {block_count} minus {mtp_layers} prediction layers is empty")
        })?;
        Ok(Cfg {
            n_layers,
            hidden,
            vocab: 0,
            eps: f("attention.layer_norm_rms_epsilon")?,
            full_attn_interval: u("full_attention_interval")?,
            n_heads: u("attention.head_count")?,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim,
            n_rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")?,
            d_inner,
            n_k_heads: u("ssm.group_count")?,
            n_v_heads,
            d_state: u("ssm.state_size")?,
            conv_kernel: u("ssm.conv_kernel")?,
            n_experts: if dense { 0 } else { u("expert_count")? },
            topk: if dense { 0 } else { u("expert_used_count")? },
            moe_inter: if dense { 0 } else { u("expert_feed_forward_length")? },
            // Qwen3.6's shared expert; the dense build has none, because its whole FFN is
            // the shared path.
            shared_inter: if dense { 0 } else { u("expert_shared_feed_forward_length")? },
            dense_inter: if dense { u("feed_forward_length")? } else { 0 },
            mtp_layers,
        })
    }

    /// Blocks 3, 7, 11, ... are full attention; every other block is a gated delta net.
    ///
    /// Off by one here silently swaps thirty linear blocks with ten attention ones, and
    /// every tensor name it then asks for exists in the file for the OTHER block type --
    /// so the loader would not complain either.
    pub fn is_full_attn(&self, layer: usize) -> bool {
        self.full_attn_interval > 0 && (layer + 1) % self.full_attn_interval == 0
    }

    /// Per-head value width for the linear-attention state.
    pub fn head_v_dim(&self) -> usize {
        self.d_inner / self.n_v_heads
    }

    /// Width of the fused `attn_qkv` projection in a linear block:
    /// q and k at `n_k_heads x d_state`, v at `d_inner`.
    pub fn qkv_width(&self) -> usize {
        2 * self.n_k_heads * self.d_state + self.d_inner
    }

    /// Routed-expert bytes moved per token, given bytes per expert. The whole reason this
    /// model is worth the port: 8 of 256 experts across 40 layers.
    pub fn expert_reads_per_token(&self) -> usize {
        self.topk * self.n_layers
    }
}

pub struct Io {
    /// The embedding table `[vocab][hidden]` is NOT held resident: at Q4_K it is 715 MB on
    /// the 27B, yet a decode step needs exactly ONE row of it. It is streamed a row at a time
    /// from `embed_file` at `embed_off + id*embed_stride`, freeing that RAM for the expert
    /// arena at the cost of a ~3 KB read per token (which the OS page cache absorbs for
    /// repeated ids). The head, by contrast, IS resident: every token reads all of it.
    embed_file: std::fs::File,
    embed_off: u64,
    embed_stride: usize,
    embed_dtype: Dtype,
    norm: Vec<f32>,
    head: Vec<u8>,
    head_dtype: Dtype,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
}

fn raw(st: &St, name: &str) -> Result<(Vec<u8>, Dtype), String> {
    let t = st.find(name).ok_or_else(|| format!("missing {name}"))?;
    let mut b = vec![0u8; t.nbytes as usize];
    st.read(t, &mut b);
    Ok((b, t.dtype))
}

/// Bytes per row of a 2-D tensor whose row width is `hidden`.
///
/// Returns `None` for a type with no kernel, rather than a plausible-looking stride: a
/// wrong stride reads a misaligned window of the neighbouring row and still produces
/// finite numbers.
fn row_stride(d: Dtype, hidden: usize) -> Option<usize> {
    match d {
        Dtype::F32 => Some(hidden * 4),
        Dtype::F16 | Dtype::Bf16 => Some(hidden * 2),
        Dtype::Q4K => Some(hidden / QK_K * Q4K_BLOCK),
        Dtype::Q6K => Some(hidden / QK_K * Q6K_BLOCK),
        _ => None,
    }
}

/// Public so the layer loader can validate a dtype at load time rather than
/// discovering it mid-token.
pub fn weight_of(bytes: &[u8], d: Dtype) -> Option<W<'_>> {
    Some(match d {
        Dtype::Q4K => W::Q4K(bytes),
        Dtype::Q5K => W::Q5K(bytes),
        Dtype::Q6K => W::Q6K(bytes),
        Dtype::F32 => W::F32(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / 4)
        }),
        Dtype::Bf16 => W::Bf16(unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / 2)
        }),
        _ => return None,
    })
}

impl Io {
    pub fn load(st: &St, hidden: usize, vocab: usize, eps: f32) -> Result<Io, String> {
        // The embedding is streamed, so we keep only its location + a private read handle to
        // its shard rather than its 715 MB of bytes.
        let et = st.find("token_embd.weight").ok_or("missing token_embd.weight")?;
        let (embed_dtype, embed_off, embed_shard) = (et.dtype, et.off as u64, et.shard);
        let embed_stride = row_stride(embed_dtype, hidden).ok_or_else(|| {
            format!("token_embd.weight is {}, which has no kernel", embed_dtype.name())
        })?;
        let embed_file = std::fs::File::open(&st.paths[embed_shard])
            .map_err(|e| format!("reopen shard for streamed embedding: {e}"))?;

        let (head, head_dtype) = raw(st, "output.weight")?;
        let nt = st.find("output_norm.weight").ok_or("missing output_norm.weight")?;
        let mut norm = vec![0f32; nt.numel() as usize];
        st.read_f32(nt, &mut norm);

        if row_stride(head_dtype, hidden).is_none() {
            return Err(format!("output.weight is {}, which has no kernel", head_dtype.name()));
        }
        if hidden % QK_K != 0 {
            return Err(format!(
                "hidden {hidden} is not a whole number of {QK_K}-element super-blocks, so a \
                 row cannot be dequantised in isolation"
            ));
        }
        Ok(Io { embed_file, embed_off, embed_stride, embed_dtype, norm, head, head_dtype, hidden, vocab, eps })
    }

    /// One row of the embedding table, widened. Streamed from disk (one `pread` of
    /// `embed_stride` bytes), not sliced from a resident table.
    pub fn embed_row(&self, id: u32, out: &mut [f32]) -> Result<(), String> {
        use std::os::unix::fs::FileExt;
        if id as usize >= self.vocab {
            return Err(format!("token id {id} is outside a {}-entry vocabulary", self.vocab));
        }
        let stride = self.embed_stride;
        let base = self.embed_off + id as u64 * stride as u64;
        let mut buf = vec![0u8; stride];
        let mut got = 0usize;
        while got < stride {
            match self.embed_file.read_at(&mut buf[got..], base + got as u64) {
                Ok(0) => return Err(format!("short read on token_embd row {id}")),
                Ok(n) => got += n,
                Err(e) => return Err(format!("token_embd row {id}: {e}")),
            }
        }
        let src = &buf[..];
        let nb = self.hidden / QK_K;
        match self.embed_dtype {
            Dtype::Q4K => gguf::q4k_dequant(out, src, nb),
            Dtype::Q6K => gguf::q6k_dequant(out, src, nb),
            Dtype::F32 => {
                for (i, v) in out[..self.hidden].iter_mut().enumerate() {
                    *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
                }
            }
            Dtype::Bf16 => {
                for (i, v) in out[..self.hidden].iter_mut().enumerate() {
                    *v = crate::st::bf16_to_f32(u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]));
                }
            }
            d => return Err(format!("embedding is {}", d.name())),
        }
        Ok(())
    }

    /// Final RMS norm and the vocabulary head. `x` is the last block's output.
    pub fn logits(&self, x: &[f32], out: &mut [f32]) -> Result<(), String> {
        let mut normed = vec![0f32; self.hidden];
        crate::ops::rmsnorm(&mut normed, x, &self.norm, self.hidden, self.eps);
        let w = weight_of(&self.head, self.head_dtype).ok_or("output head has no kernel")?;
        crate::ops::mmw(out, &normed, w, self.hidden, self.vocab);
        Ok(())
    }

    /// The final norm's gain, so a caller can reproduce `result_norm` for a reference diff.
    pub fn norm(&self) -> &[f32] {
        &self.norm
    }

    /// The vocabulary head as an [`ops::W`] handle, for the training backward pass to run
    /// `grad_hidden = Wᵀ·grad_logits` through [`crate::ops::wt`]. Same weight `logits` uses.
    pub fn head_w(&self) -> Option<W<'_>> {
        weight_of(&self.head, self.head_dtype)
    }

    /// Resident IO bytes. The embedding is streamed, so it is deliberately NOT counted here —
    /// only the head and the final norm actually occupy RAM.
    pub fn bytes(&self) -> usize {
        self.head.len() + self.norm.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real header of Qwen3.6-35B-A3B, so the geometry assertions below are pinned to
    /// a checkpoint rather than to my arithmetic.
    fn qwen36_35b() -> Meta {
        use crate::gguf::Value::{Str, F, U};
        let mut m = Meta::new();
        m.insert("general.architecture".into(), Str("qwen35moe".into()));
        for (k, v) in [
            ("block_count", 40u64), ("embedding_length", 2048), ("full_attention_interval", 4),
            ("attention.head_count", 16), ("attention.head_count_kv", 2),
            ("attention.key_length", 256), ("attention.value_length", 256),
            ("rope.dimension_count", 64),
            ("ssm.inner_size", 4096), ("ssm.group_count", 16), ("ssm.time_step_rank", 32),
            ("ssm.state_size", 128), ("ssm.conv_kernel", 4),
            ("expert_count", 256), ("expert_used_count", 8),
            ("expert_feed_forward_length", 512), ("expert_shared_feed_forward_length", 512),
        ] {
            m.insert(format!("qwen35moe.{k}"), U(v));
        }
        m.insert("qwen35moe.rope.freq_base".into(), F(10_000_000.0));
        m.insert("qwen35moe.attention.layer_norm_rms_epsilon".into(), F(1e-6));
        m
    }

    #[test]
    fn the_config_reproduces_the_checkpoints_geometry() {
        let c = Cfg::from_meta(&qwen36_35b()).unwrap();
        assert_eq!((c.n_layers, c.hidden), (40, 2048));
        assert_eq!(c.head_v_dim(), 128, "4096 value width over 32 heads");
        // 8192 is the width of attn_qkv in the header, and it is NOT three equal parts:
        // q and k are 16 heads of 128, v is 32 heads of 128.
        assert_eq!(c.qkv_width(), 8192);
        assert_eq!(2 * c.n_k_heads * c.d_state, 4096, "q and k together");
        assert_eq!(c.d_inner, 4096, "v alone");
        assert_ne!(c.qkv_width() / 3, c.d_inner, "an equal three-way split would be wrong");
        // Rope is partial: 64 of the 256 head dims.
        assert!(c.n_rot < c.head_dim, "rope must not cover the whole head");
    }

    /// The streamed embedding must return BYTE-IDENTICAL rows to a full-tensor read + dequant.
    /// A wrong offset/stride reads a neighbouring row and still yields finite, fluent-looking
    /// numbers, so this asserts equality against an independent reference rather than trusting
    /// it. Runs only when `K3_MODEL` points at a real checkpoint dir (skips otherwise).
    #[test]
    fn streamed_embedding_row_is_bit_identical_to_a_full_read() {
        let Some(dir) = std::env::var_os("K3_MODEL") else {
            return; // no model available in this environment
        };
        let st = St::open(std::path::Path::new(&dir)).expect("open model");
        let meta = st.meta.as_ref().expect("gguf meta");
        let cfg = Cfg::from_meta(meta).expect("cfg");
        let vt = st.find("token_embd.weight").expect("token_embd");
        let hidden = cfg.hidden;
        let vocab = vt.numel() as usize / hidden;
        let io = Io::load(&st, hidden, vocab, cfg.eps).expect("io");

        // reference: read the WHOLE embedding tensor once, dequant selected rows directly.
        let mut all = vec![0u8; vt.nbytes as usize];
        st.read(vt, &mut all);
        let stride = row_stride(vt.dtype, hidden).expect("stride");
        let nb = hidden / QK_K;
        let dequant = |src: &[u8], out: &mut [f32]| match vt.dtype {
            Dtype::Q4K => gguf::q4k_dequant(out, src, nb),
            Dtype::Q6K => gguf::q6k_dequant(out, src, nb),
            d => panic!("test does not cover embedding dtype {}", d.name()),
        };

        for &id in &[0u32, 1, 2, 100, 12345, (vocab / 2) as u32, (vocab - 1) as u32] {
            let mut want = vec![0f32; hidden];
            dequant(&all[id as usize * stride..][..stride], &mut want);
            let mut got = vec![0f32; hidden];
            io.embed_row(id, &mut got).expect("embed_row");
            assert_eq!(want, got, "streamed embedding row {id} differs from a full read");
        }
    }

    /// Ten full-attention blocks at 3, 7, 11, ... and thirty linear ones. Off by one here
    /// swaps the two block types, and the tensors it would then ask for all exist -- for
    /// the other kind of block -- so nothing downstream would complain.
    #[test]
    fn full_attention_lands_on_every_fourth_block_starting_at_three() {
        let c = Cfg::from_meta(&qwen36_35b()).unwrap();
        let full: Vec<usize> = (0..c.n_layers).filter(|&l| c.is_full_attn(l)).collect();
        assert_eq!(&full[..4], &[3, 7, 11, 15]);
        assert_eq!(full.len(), 10, "10 attention blocks");
        assert_eq!(c.n_layers - full.len(), 30, "30 gated-delta-net blocks");
        assert!(!c.is_full_attn(0), "block 0 is linear -- it has ssm_* tensors");
    }

    /// A missing key must be an error. Defaulting a head count or a rope base yields a
    /// model that loads, runs, and is quietly wrong.
    /// The real header of Qwen3.8-27B, the DENSE sibling. Every value here was read out of
    /// `unsloth/Qwen3.8-27B-GGUF` Q4_K_M with a range request, not inferred.
    fn qwen38_27b() -> Meta {
        use crate::gguf::Value::{Str, F, U};
        let mut m = Meta::new();
        m.insert("general.architecture".into(), Str("qwen35".into()));
        for (k, v) in [
            // 65 blocks with ONE prediction layer, so the stack is 64.
            ("block_count", 65u64), ("nextn_predict_layers", 1),
            ("embedding_length", 5120), ("feed_forward_length", 17408),
            ("full_attention_interval", 4),
            ("attention.head_count", 24), ("attention.head_count_kv", 4),
            ("attention.key_length", 256), ("attention.value_length", 256),
            ("rope.dimension_count", 64),
            ("ssm.inner_size", 6144), ("ssm.group_count", 16), ("ssm.time_step_rank", 48),
            ("ssm.state_size", 128), ("ssm.conv_kernel", 4),
        ] {
            m.insert(format!("qwen35.{k}"), U(v));
        }
        m.insert("qwen35.rope.freq_base".into(), F(10_000_000.0));
        m.insert("qwen35.attention.layer_norm_rms_epsilon".into(), F(1e-6));
        m
    }

    /// The dense sibling parses, and parses as DENSE -- no experts invented, no expert
    /// keys demanded.
    #[test]
    fn the_dense_sibling_parses_with_the_same_reader() {
        let c = Cfg::from_meta(&qwen38_27b()).unwrap();
        assert!(c.is_dense(), "no expert_count means dense");
        assert_eq!(c.ffn_width(), 17408, "one FFN of 17408, not experts of 512");
        assert_eq!((c.n_experts, c.topk, c.shared_inter), (0, 0, 0), "nothing invented");
        assert_eq!((c.hidden, c.n_heads, c.n_kv_heads), (5120, 24, 4));

        // The head ratio is THREE here (16 key heads over 48 value heads), where the 35B is
        // 2 and the 122B is 4. `linear_block` maps value head hv to key head hv % nk, so a
        // ratio it has never seen is exactly the case that mapping has to get right.
        assert_eq!((c.n_k_heads, c.n_v_heads), (16, 48));
        assert_eq!(c.n_v_heads / c.n_k_heads, 3, "a head ratio neither fixture covers");
        assert_eq!(c.head_v_dim(), 128, "6144 value width over 48 heads");
        // q and k at 16 x 128 each, v at 6144.
        assert_eq!(c.qkv_width(), 16 * 128 * 2 + 6144);
    }

    /// The quietest possible way to be wrong: run the multi-token-prediction head as if it
    /// were layer 65. Every tensor it names exists (`blk.64.attn_q`, `blk.64.ffn_gate`, ...)
    /// so the loader would succeed and simply compute a different model.
    #[test]
    fn the_prediction_head_is_excluded_from_the_stack() {
        let c = Cfg::from_meta(&qwen38_27b()).unwrap();
        assert_eq!(c.n_layers, 64, "65 blocks minus 1 prediction layer");
        assert_eq!(c.mtp_layers, 1, "kept, because it is a ready-made speculative drafter");
        // An MoE build declares none, and must be unaffected.
        assert_eq!(Cfg::from_meta(&qwen36_35b()).unwrap().mtp_layers, 0);
        assert_eq!(Cfg::from_meta(&qwen36_35b()).unwrap().n_layers, 40);
        // And a checkpoint claiming to be all prediction layers is refused, not wrapped
        // around into a huge `n_layers` by underflow.
        let mut m = qwen38_27b();
        m.insert("qwen35.nextn_predict_layers".into(), crate::gguf::Value::U(65));
        assert!(Cfg::from_meta(&m).unwrap_err().contains("is empty"));
    }

    /// The two arches share every attention and SSM dimension, which is the whole reason
    /// one block implementation serves both. If this stops holding, the shared forward
    /// pass stops being justified.
    #[test]
    fn dense_and_moe_differ_only_after_the_post_attention_norm() {
        let (moe, den) = (Cfg::from_meta(&qwen36_35b()).unwrap(), Cfg::from_meta(&qwen38_27b()).unwrap());
        assert_eq!(moe.full_attn_interval, den.full_attn_interval);
        assert_eq!((moe.head_dim, den.head_dim), (256, 256));
        assert_eq!((moe.n_rot, den.n_rot), (64, 64));
        assert_eq!((moe.d_state, den.d_state), (128, 128));
        assert_eq!((moe.conv_kernel, den.conv_kernel), (4, 4));
        assert_eq!(moe.rope_base, den.rope_base);
        // Both put full attention on blocks 3, 7, 11, ...
        for l in [3usize, 7, 11, 63] {
            assert_eq!(moe.is_full_attn(l), den.is_full_attn(l), "block {l}");
        }
        assert!(!moe.is_dense() && den.is_dense(), "and differ in exactly one thing");
    }

    #[test]
    fn missing_metadata_is_refused_rather_than_defaulted() {
        let mut m = qwen36_35b();
        m.remove("qwen35moe.rope.freq_base");
        let e = Cfg::from_meta(&m).unwrap_err();
        assert!(e.contains("rope.freq_base"), "the error must name the key: {e}");

        let mut m = qwen36_35b();
        m.insert("general.architecture".into(), crate::gguf::Value::Str("llama".into()));
        assert!(Cfg::from_meta(&m).is_err(), "a different architecture must be refused");
    }

    /// 8 experts of 40 layers is 320 reads a token -- the number the cache has to serve,
    /// and the reason this model is worth porting at all.
    #[test]
    fn expert_reads_per_token_is_the_cache_workload() {
        let c = Cfg::from_meta(&qwen36_35b()).unwrap();
        assert_eq!(c.expert_reads_per_token(), 320);
        // 2.04 MB an expert, from the real tensor sizes: gate and up are Q4_K 589824 B,
        // down is Q6_K 860160 B.
        let per_expert = 589_824 + 589_824 + 860_160;
        assert_eq!(c.expert_reads_per_token() * per_expert, 652_738_560, "653 MB a token");
    }

    /// The gate is per v-HEAD, one scalar for a whole 128x128 state matrix -- not per
    /// channel like K3's KDA. And `a` is stored negative, so the decay stays in (0, 1].
    #[test]
    fn the_gated_deltanet_decay_is_per_head_and_lands_in_the_unit_interval() {
        let (alpha, dt, a) = ([0.5f32, -2.0, 3.0], [0.1f32, 0.2, -0.3], [-1.0f32, -0.5, -2.0]);
        let mut decay = [0f32; 3];
        crate::ops::gdn_decay(&mut decay, &alpha, &dt, &a, 3);
        for (h, d) in decay.iter().enumerate() {
            assert!(*d > 0.0 && *d <= 1.0, "head {h} decay {d} escaped (0, 1]");
        }
        // exp(softplus(0.6) * -1.0)
        // a[0] = -1.0, so this is exp(softplus(0.5 + 0.1) * -1.0).
        let want = (-crate::ops::softplus(0.6)).exp();
        assert!((decay[0] - want).abs() < 1e-6, "{} vs {want}", decay[0]);
        // A less negative `a` must forget MORE slowly.
        assert!(decay[1] > decay[2], "a=-0.5 should retain more than a=-2.0");
    }

    /// l2_norm, not RMS norm: no division by sqrt(n) and no learned gain. Confusing the
    /// two rescales q and k by sqrt(128) = 11.3x, which the delta rule partly absorbs.
    #[test]
    fn l2_norm_makes_each_head_a_unit_vector() {
        let x: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
        let mut y = vec![0f32; 8];
        crate::ops::l2norm_heads(&mut y, &x, 2, 4, 0.0);
        for h in 0..2 {
            let n: f32 = y[h * 4..][..4].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-6, "head {h} has norm {n}, expected 1");
        }
        // RMS norm would leave norm sqrt(4) = 2, not 1.
    }

    /// A row stride is bytes-per-row, and getting it wrong reads a misaligned window of a
    /// neighbouring row -- finite, plausible, wrong. Pinned against the real geometry of
    /// Qwen3.6-35B: hidden 2048 is 8 super-blocks.
    #[test]
    fn row_strides_match_the_block_formats() {
        assert_eq!(row_stride(Dtype::Q4K, 2048), Some(8 * 144), "8 super-blocks of 144 B");
        assert_eq!(row_stride(Dtype::Q6K, 2048), Some(8 * 210));
        assert_eq!(row_stride(Dtype::F32, 2048), Some(8192));
        assert_eq!(row_stride(Dtype::Bf16, 2048), Some(4096));
        // 122B geometry: hidden 3072 is 12 super-blocks.
        assert_eq!(row_stride(Dtype::Q4K, 3072), Some(12 * 144));
    }

    /// A type with no dequantiser must yield no stride at all. Returning a guess here is
    /// how an unreadable tensor becomes silently-wrong weights.
    #[test]
    fn a_type_without_a_kernel_has_no_stride() {
        assert_eq!(row_stride(Dtype::Q2K, 2048), None);
        assert_eq!(row_stride(Dtype::IQ4NL, 2048), None);
        assert_eq!(row_stride(Dtype::Q5_0, 2048), None);
    }

    /// The whole embedding table for Qwen3.6-35B is 248320 x 2048. One row is 1152 bytes
    /// at Q4_K; the table is 286 MB packed and would be 1.9 GB widened, which is why the
    /// lookup dequantises a row rather than the tensor.
    #[test]
    fn one_embedding_row_is_a_fraction_of_the_table() {
        let (vocab, hidden) = (248_320usize, 2048usize);
        let row = row_stride(Dtype::Q4K, hidden).unwrap();
        assert_eq!(row, 1152);
        assert_eq!(vocab * row, 286_064_640, "packed table");
        assert_eq!(vocab * hidden * 4, 2_034_237_440, "widened, which is why we do not");
    }
}
