// SPDX-License-Identifier: Apache-2.0

use crate::ops::{self, W};

/// `compress_ratios`, transcribed verbatim from the released config.json of
/// deepseek-ai/DeepSeek-V4-Flash-0731.
///
/// NOTE THE LENGTH: 46 entries for `num_hidden_layers` 43. The three trailing zeros are
/// NOT the last three decoder layers. The release ships 48 shards -- shard 1 is `embed`,
/// shards 2..=44 are layers 0..=42, shard 45 is `norm` + `head`, and shards 46..=48 are
/// three further layer-sized blocks beyond the decoder stack. The array is indexed over
/// all 46 blocks, so decoder layers 40, 41 and 42 take indices 40, 41, 42 -- which are
/// `4`, `128`, `4`, not zero.
///
/// Reading the trailing zeros as the last three decoder layers turns those three into
/// pure sliding-window attention: no Compressor, no Indexer, and YaRN silently disabled.
/// Three of 43 layers then attend over a 128-token window instead of the whole compressed
/// history. It does not crash and it does not look wrong in the output.
pub const COMPRESS_RATIOS: [usize; 46] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
    0, 0,
];

/// `COMPRESS_RATIOS[layer]`, bounds-checked. Panics rather than defaulting to zero: a
/// zero here is a valid ratio meaning "pure sliding window", so an out-of-range layer
/// would silently become a plausible configuration instead of an error.
pub fn compress_ratio(layer: usize) -> usize {
    COMPRESS_RATIOS[layer]
}

pub struct AttnDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub window: usize,
    /// compress_ratios[layer]. Zero means pure sliding window: no Compressor, no
    /// Indexer, and YaRN disabled in favour of the base rope_theta.
    pub compress_ratio: usize,
    pub eps: f32,
}

pub struct AttnW<'a> {
    pub wq_a: W<'a>,
    pub q_norm: &'a [f32],
    pub wq_b: W<'a>,
    pub wkv: W<'a>,
    pub kv_norm: &'a [f32],
    pub wo_a: W<'a>,
    pub wo_b: W<'a>,
    pub attn_sink: &'a [f32],
}

pub struct Rope {
    /// [pos][dim/2] interleaved (cos, sin).
    pub cs: Vec<f32>,
    pub half: usize,
}

// YaRN: interpolate the low-frequency half of the spectrum by `factor` and blend toward
// the untouched high frequencies across a linear ramp between the two correction
// dimensions. original_seq_len == 0 disables it, which is what layers with
// compress_ratio == 0 use.
#[allow(clippy::too_many_arguments)]
pub fn precompute_rope(
    dim: usize,
    seqlen: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Rope {
    let half = dim / 2;
    let mut freqs: Vec<f64> = (0..half)
        .map(|i| 1.0 / (base as f64).powf(2.0 * i as f64 / dim as f64))
        .collect();

    if original_seq_len > 0 {
        let corr = |rot: f64| {
            dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = corr(beta_fast as f64).floor().max(0.0);
        let mut high = corr(beta_slow as f64).ceil().min(dim as f64 - 1.0);
        if (high - low).abs() < f64::EPSILON {
            high += 0.001;
        }
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor as f64 * (1.0 - smooth) + *f * smooth;
        }
    }

    let mut cs = vec![0f32; seqlen * half * 2];
    for t in 0..seqlen {
        for i in 0..half {
            let a = t as f64 * freqs[i];
            cs[(t * half + i) * 2] = a.cos() as f32;
            cs[(t * half + i) * 2 + 1] = a.sin() as f32;
        }
    }
    Rope { cs, half }
}

// torch views the last axis as complex pairs, so element 2i and 2i+1 rotate together.
// Splitting the axis in half instead (the Llama convention) is a different rotation
// that still produces plausible attention.
pub fn apply_rope(x: &mut [f32], rope: &Rope, pos: usize, inverse: bool) {
    for i in 0..rope.half {
        let c = rope.cs[(pos * rope.half + i) * 2];
        let s = {
            let s = rope.cs[(pos * rope.half + i) * 2 + 1];
            if inverse { -s } else { s }
        };
        let (re, im) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = re * c - im * s;
        x[2 * i + 1] = re * s + im * c;
    }
}

/// Causal sliding window: token `t` attends to `max(0, t-window+1) ..= t`.
pub fn window_indices(t: usize, window: usize) -> std::ops::RangeInclusive<usize> {
    t.saturating_sub(window - 1)..=t
}

/// The window slots for token `t` as the reference lays them out, padded with `-1` so
/// every token has the same width.
pub fn window_row(t: usize, window: usize, t_len: usize) -> Vec<i64> {
    let width = window.min(t_len);
    let lo = t.saturating_sub(window - 1);
    (0..width).map(|k| if lo + k <= t { (lo + k) as i64 } else { -1 }).collect()
}

// Online softmax over the selected positions, with a learnable sink. The sink adds
// exp(sink[h] - max) to the DENOMINATOR ONLY -- it has no value vector, so it lets a
// head attend to nothing. Giving it a value row is a plausible and wrong reading of
// kernel.py:346.
pub fn sparse_attn_row(
    o: &mut [f32],
    q: &[f32],
    kv: &[f32],
    idxs: &[i64],
    sink: f32,
    head_dim: usize,
    scale: f32,
) {
    let mut m = sink;
    let mut sc = Vec::with_capacity(idxs.len());
    // A negative index is a masked slot: the window is shorter than its buffer, or a
    // compressed block is not causal yet. It must contribute nothing, not row 0.
    for &ti in idxs {
        if ti < 0 {
            sc.push(f32::NEG_INFINITY);
            continue;
        }
        let t = ti as usize;
        let row = &kv[t * head_dim..][..head_dim];
        let mut d = 0.0f64;
        for i in 0..head_dim {
            d += q[i] as f64 * row[i] as f64;
        }
        let s = d as f32 * scale;
        if s > m {
            m = s;
        }
        sc.push(s);
    }
    let mut z = (sink - m).exp() as f64;
    for s in sc.iter_mut() {
        *s = if s.is_infinite() && s.is_sign_negative() { 0.0 } else { (*s - m).exp() };
        z += *s as f64;
    }
    o[..head_dim].fill(0.0);
    for (k, &ti) in idxs.iter().enumerate() {
        if ti < 0 {
            continue;
        }
        let p = (sc[k] as f64 / z) as f32;
        let row = &kv[ti as usize * head_dim..][..head_dim];
        for i in 0..head_dim {
            o[i] += p * row[i];
        }
    }
}

/// One decoder layer's attention for the `compress_ratio == 0` case: sliding window
/// only, prefill from position 0. `x` is [T][hidden], `out` is [T][hidden].
///
/// NOT implemented: layers with a non-zero compress_ratio, which add a Compressor
/// (gated pooling over `ratio` tokens, overlapping when ratio == 4) and, at ratio 4,
/// an Indexer that selects the top `index_topk` compressed positions. `assert` guards
/// the entry rather than silently running a windowed approximation of a compressed
/// layer.
pub fn attention_window(
    out: &mut [f32],
    x: &[f32],
    w: &AttnW,
    d: &AttnDims,
    t_len: usize,
    rope: &Rope,
) {
    assert_eq!(d.compress_ratio, 0, "use attention_prefill for compressed layers");
    attention_prefill(out, x, w, d, t_len, rope, None);
}

/// Everything a compressed layer needs beyond the sliding window. `indexer` is present
/// only at ratio 4; the ratio-128 layers take every causally available block.
pub struct Compressed<'a> {
    pub w: &'a CompressorW<'a>,
    pub d: &'a CompressorDims,
    pub indexer: Option<(&'a IndexerW<'a>, &'a IndexerDims, &'a CompressorW<'a>, &'a CompressorDims)>,
}

pub fn attention_prefill(
    out: &mut [f32],
    x: &[f32],
    w: &AttnW,
    d: &AttnDims,
    t_len: usize,
    rope: &Rope,
    comp: Option<&Compressed>,
) {
    assert_eq!(
        d.compress_ratio == 0,
        comp.is_none(),
        "compress_ratio and the Compressed context must agree"
    );
    let (e, h_n, hd, rd) = (d.hidden, d.n_heads, d.head_dim, d.rope_head_dim);
    let scale = (hd as f32).powf(-0.5);
    let gsz = h_n * hd / d.o_groups; // 4096

    let mut qr1 = vec![0f32; d.q_lora_rank];
    // The Indexer reuses the attention block's normalised q-LoRA, so keep every token's.
    let mut qr = vec![0f32; t_len * d.q_lora_rank];
    let mut q = vec![0f32; t_len * h_n * hd];
    let mut kv = vec![0f32; t_len * hd];
    let mut kvt = vec![0f32; hd];

    for t in 0..t_len {
        let xt = &x[t * e..][..e];
        ops::mmw(&mut qr1, xt, w.wq_a, e, d.q_lora_rank);
        let qn = &mut qr[t * d.q_lora_rank..][..d.q_lora_rank];
        ops::rmsnorm(qn, &qr1, w.q_norm, d.q_lora_rank, d.eps);
        let qn = qn.to_vec();
        ops::mmw(&mut q[t * h_n * hd..][..h_n * hd], &qn, w.wq_b, d.q_lora_rank, h_n * hd);

        // A second RMS scaling on q, PER HEAD and with no learned gain. Skipping it
        // leaves attention that still normalises and is wrong.
        for h in 0..h_n {
            let qh = &mut q[(t * h_n + h) * hd..][..hd];
            let mut ss = 0.0f64;
            for v in qh.iter() {
                ss += *v as f64 * *v as f64;
            }
            let inv = (1.0 / (ss / hd as f64 + d.eps as f64).sqrt()) as f32;
            for v in qh.iter_mut() {
                *v *= inv;
            }
            apply_rope(&mut qh[hd - rd..], rope, t, false);
        }

        // One shared KV head serves every query head (num_key_value_heads = 1).
        ops::mmw(&mut kvt, xt, w.wkv, e, hd);
        let src = kvt.clone();
        ops::rmsnorm(&mut kvt, &src, w.kv_norm, hd, d.eps);
        apply_rope(&mut kvt[hd - rd..], rope, t, false);
        kv[t * hd..][..hd].copy_from_slice(&kvt);
    }

    // Compressed blocks are appended AFTER the token window, so their indices carry an
    // offset of the token count. Getting that offset wrong points attention at tokens.
    let mut rows: Vec<Vec<i64>> = (0..t_len).map(|t| window_row(t, d.window, t_len)).collect();
    if let Some(c) = comp {
        let kvc = compress_prefill(x, c.w, c.d, t_len, rope);
        let nblk = if kvc.is_empty() { 0 } else { kvc.len() / hd };
        let offset = t_len;
        let extra = match c.indexer {
            Some((iw, id, icw, icd)) => {
                let ikvc = compress_prefill(x, icw, icd, t_len, rope);
                indexer_prefill(&qr, x, &ikvc, iw, id, t_len, rope, offset)
            }
            None => compress_topk_prefill(t_len, nblk, c.d.ratio, offset),
        };
        for (t, r) in rows.iter_mut().enumerate() {
            r.extend_from_slice(&extra[t]);
        }
        kv.extend_from_slice(&kvc);
    }

    let mut o = vec![0f32; h_n * hd];
    let mut go = vec![0f32; d.o_groups * d.o_lora_rank];
    for t in 0..t_len {
        let idxs = &rows[t];
        for h in 0..h_n {
            sparse_attn_row(
                &mut o[h * hd..][..hd],
                &q[(t * h_n + h) * hd..][..hd],
                &kv,
                idxs,
                w.attn_sink[h],
                hd,
                scale,
            );
            // The output is de-rotated: the same rope, conjugated.
            apply_rope(&mut o[h * hd + hd - rd..][..rd], rope, t, true);
        }

        // Grouped o-LoRA. wo_a is [o_groups * o_lora_rank][gsz], read as one
        // [o_lora_rank][gsz] block per group: group g sees only its own slice of o.
        for g in 0..d.o_groups {
            let osl = &o[g * gsz..][..gsz];
            let dst = &mut go[g * d.o_lora_rank..][..d.o_lora_rank];
            // Group g sees only its own [o_lora_rank][gsz] block of wo_a.
            let rows = d.o_lora_rank;
            let sub = match w.wo_a {
                W::F32(m) => W::F32(&m[g * rows * gsz..][..rows * gsz]),
                W::Bf16(m) => W::Bf16(&m[g * rows * gsz..][..rows * gsz]),
                W::F8Block { w: m, scale, block } => {
                    let sb_in = gsz.div_ceil(block);
                    W::F8Block {
                        w: &m[g * rows * gsz..][..rows * gsz],
                        scale: &scale[(g * rows / block) * sb_in..],
                        block,
                    }
                }
                W::I8(_) => unimplemented!("wo_a is never the draft format"),
                // A k-quant packs its scales inside each 256-element super-block, so a
                // row-offset sub-slice is only meaningful when the group width is a whole
                // number of super-blocks -- and DeepSeek-V4 is never read from GGUF.
                W::Q4K(_) | W::Q5K(_) | W::Q6K(_) => unimplemented!("wo_a is never a k-quant"),
            };
            ops::mmw(dst, osl, sub, gsz, rows);
        }
        ops::mmw(&mut out[t * e..][..e], &go, w.wo_b, d.o_groups * d.o_lora_rank, e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotates_adjacent_pairs_not_split_halves() {
        let rope = precompute_rope(4, 4, 0, 10000.0, 1.0, 32.0, 1.0);
        let mut x = [1.0f32, 0.0, 1.0, 0.0];
        apply_rope(&mut x, &rope, 1, false);
        // (1, 0) rotated by position 1's angle for pair 0 lands on (cos, sin).
        let (c0, s0) = (rope.cs[(rope.half) * 2], rope.cs[(rope.half) * 2 + 1]);
        assert!((x[0] - c0).abs() < 1e-6 && (x[1] - s0).abs() < 1e-6, "pair 0");
        // Pair 1 uses its OWN, lower frequency. A split-halves implementation would
        // have applied pair 0's angle here.
        let (c1, s1) = (rope.cs[(rope.half + 1) * 2], rope.cs[(rope.half + 1) * 2 + 1]);
        assert!((x[2] - c1).abs() < 1e-6 && (x[3] - s1).abs() < 1e-6, "pair 1");
        assert!((c0 - c1).abs() > 1e-6, "the two pairs must differ or this proves nothing");
    }

    #[test]
    fn inverse_rope_undoes_the_forward_rotation() {
        let rope = precompute_rope(8, 16, 0, 10000.0, 1.0, 32.0, 1.0);
        let orig: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut x = orig.clone();
        apply_rope(&mut x, &rope, 5, false);
        assert!(x.iter().zip(&orig).any(|(a, b)| (a - b).abs() > 1e-3), "rope did something");
        apply_rope(&mut x, &rope, 5, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "inverse must undo forward");
        }
    }

    #[test]
    fn yarn_scales_low_frequencies_and_leaves_high_ones() {
        // Compared far out in the sequence: near position 0 both angles are tiny and
        // cos() rounds to 1.0 in f32 whether YaRN ran or not.
        const T: usize = 4096;
        let plain = precompute_rope(64, T, 0, 160000.0, 16.0, 32.0, 1.0);
        let yarn = precompute_rope(64, T, 65536, 160000.0, 16.0, 32.0, 1.0);
        let h = plain.half;
        let at = |r: &Rope, i: usize| r.cs[((T - 1) * h + i) * 2];
        assert!(
            (at(&plain, 0) - at(&yarn, 0)).abs() < 1e-6,
            "the highest frequency sits below the correction range and is untouched"
        );
        assert!(
            (at(&plain, h - 1) - at(&yarn, h - 1)).abs() > 1e-5,
            "the lowest frequency is interpolated by `factor`"
        );
    }

    #[test]
    fn window_is_causal_and_bounded() {
        assert_eq!(window_indices(0, 128).collect::<Vec<_>>(), vec![0]);
        assert_eq!(window_indices(5, 3).collect::<Vec<_>>(), vec![3, 4, 5]);
        assert_eq!(window_indices(200, 128).count(), 128);
        assert_eq!(*window_indices(200, 128).end(), 200, "the current token is included");
    }

    // The sink has no value vector. With a large sink every probability shrinks toward
    // zero but the output direction is unchanged -- attending to nothing, not to a row.
    #[test]
    fn attn_sink_scales_the_denominator_only() {
        let hd = 4;
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let kv = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let idxs = [0i64, 1];
        let mut lo = vec![0f32; hd];
        let mut hi = vec![0f32; hd];
        sparse_attn_row(&mut lo, &q, &kv, &idxs, -50.0, hd, 1.0);
        sparse_attn_row(&mut hi, &q, &kv, &idxs, 5.0, hd, 1.0);
        let nlo: f32 = lo.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nhi: f32 = hi.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(nhi < nlo, "a larger sink must shrink the output, not redirect it");
        let dot: f32 = lo.iter().zip(&hi).map(|(a, b)| a * b).sum();
        assert!((dot / (nlo * nhi) - 1.0).abs() < 1e-4, "direction must be unchanged");
    }
}

// ------------------------------------------------------------- compressor ----

pub struct CompressorDims {
    pub hidden: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub ratio: usize,
    /// The Indexer's compressor applies a Hadamard rotation before its FP4 simulation.
    pub rotate: bool,
    pub eps: f32,
}

impl CompressorDims {
    /// Overlapping windows are used at ratio 4 only.
    pub fn overlap(&self) -> bool {
        self.ratio == 4
    }
    pub fn coff(&self) -> usize {
        1 + usize::from(self.overlap())
    }
}

pub struct CompressorW<'a> {
    /// [ratio][coff * head_dim] absolute position embedding, added to the GATE.
    pub ape: &'a [f32],
    pub wkv: W<'a>,
    pub wgate: W<'a>,
    pub norm: &'a [f32],
}

/// In-place fast Walsh-Hadamard transform over `n` (a power of two), scaled by
/// n^-0.5 to match `fast_hadamard_transform.hadamard_transform(x, scale=d**-0.5)`.
pub fn hadamard(x: &mut [f32], n: usize) {
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h << 1;
        }
        h <<= 1;
    }
    let s = (n as f32).powf(-0.5);
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// Prefill compression: fold `t_len` tokens into `t_len / ratio` blocks of `head_dim`.
/// Returns the compressed KV, [nblk][head_dim]. The tail of `t_len % ratio` tokens is
/// carried into decode state by the reference and contributes no block here.
pub fn compress_prefill(
    x: &[f32],
    w: &CompressorW,
    d: &CompressorDims,
    t_len: usize,
    rope: &Rope,
) -> Vec<f32> {
    let (e, hd, r) = (d.hidden, d.head_dim, d.ratio);
    let cd = d.coff() * hd;
    let cutoff = t_len - t_len % r;
    let nblk = cutoff / r;
    if nblk == 0 {
        return Vec::new();
    }

    let mut kv = vec![0f32; t_len * cd];
    let mut sc = vec![0f32; t_len * cd];
    for t in 0..t_len {
        ops::mmw(&mut kv[t * cd..][..cd], &x[t * e..][..e], w.wkv, e, cd);
        ops::mmw(&mut sc[t * cd..][..cd], &x[t * e..][..e], w.wgate, e, cd);
        // The APE is added to the GATE, never to the value.
        for i in 0..cd {
            sc[t * cd + i] += w.ape[(t % r) * cd + i];
        }
    }

    // Gather each block's slots. Without overlap a block is its own `ratio` tokens.
    // With overlap it is 2*ratio slots: the upper half of this block's dims, plus the
    // lower half of the PREVIOUS block's - which is what makes the windows overlap.
    let slots = d.coff() * r;
    let mut bkv = vec![0f32; slots * hd];
    let mut bsc = vec![0f32; slots * hd];
    let mut out = vec![0f32; nblk * hd];

    for b in 0..nblk {
        if d.overlap() {
            bkv.fill(0.0);
            bsc.fill(f32::NEG_INFINITY); // masked slots must not win the softmax
            for j in 0..r {
                let src = (b * r + j) * cd;
                bkv[(r + j) * hd..][..hd].copy_from_slice(&kv[src + hd..][..hd]);
                bsc[(r + j) * hd..][..hd].copy_from_slice(&sc[src + hd..][..hd]);
                if b > 0 {
                    let prev = ((b - 1) * r + j) * cd;
                    bkv[j * hd..][..hd].copy_from_slice(&kv[prev..][..hd]);
                    bsc[j * hd..][..hd].copy_from_slice(&sc[prev..][..hd]);
                }
            }
        } else {
            for j in 0..r {
                let src = (b * r + j) * cd;
                bkv[j * hd..][..hd].copy_from_slice(&kv[src..][..hd]);
                bsc[j * hd..][..hd].copy_from_slice(&sc[src..][..hd]);
            }
        }

        // Softmax runs over the SLOT axis independently per dimension: the gate picks,
        // for each channel, which token in the block to take it from.
        let o = &mut out[b * hd..][..hd];
        for i in 0..hd {
            let mut m = f32::NEG_INFINITY;
            for j in 0..slots {
                let v = bsc[j * hd + i];
                if v > m {
                    m = v;
                }
            }
            let mut z = 0.0f64;
            let mut acc = 0.0f64;
            for j in 0..slots {
                let p = if bsc[j * hd + i] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (bsc[j * hd + i] - m).exp()
                };
                z += p as f64;
                acc += p as f64 * bkv[j * hd + i] as f64;
            }
            o[i] = (acc / z) as f32;
        }
    }

    let rd = d.rope_head_dim;
    let mut normed = vec![0f32; hd];
    for b in 0..nblk {
        let src = out[b * hd..][..hd].to_vec();
        ops::rmsnorm(&mut normed, &src, w.norm, hd, d.eps);
        // Block b sits at absolute position b * ratio.
        apply_rope(&mut normed[hd - rd..], rope, b * r, false);
        if d.rotate {
            hadamard(&mut normed, hd);
        }
        out[b * hd..][..hd].copy_from_slice(&normed);
    }
    out
}

// ---------------------------------------------------------------- indexer ----

pub struct IndexerDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub index_topk: usize,
    pub ratio: usize,
}

pub struct IndexerW<'a> {
    pub wq_b: W<'a>,
    pub weights_proj: W<'a>,
}

/// Which compressed blocks each token attends to. `-1` marks a slot that is masked
/// (either not yet causal, or fewer blocks exist than `index_topk`).
pub fn indexer_prefill(
    qr: &[f32],
    x: &[f32],
    kvc: &[f32],
    w: &IndexerW,
    d: &IndexerDims,
    t_len: usize,
    rope: &Rope,
    offset: usize,
) -> Vec<Vec<i64>> {
    let (e, h_n, hd, rd) = (d.hidden, d.n_heads, d.head_dim, d.rope_head_dim);
    let nblk = kvc.len().checked_div(hd).unwrap_or(0);
    let softmax_scale = (hd as f32).powf(-0.5);
    let wscale = softmax_scale * (h_n as f32).powf(-0.5);

    let mut q = vec![0f32; h_n * hd];
    let mut wt = vec![0f32; h_n];
    let mut res = Vec::with_capacity(t_len);

    for t in 0..t_len {
        ops::mmw(&mut q, &qr[t * d.q_lora_rank..][..d.q_lora_rank], w.wq_b, d.q_lora_rank, h_n * hd);
        for h in 0..h_n {
            let qh = &mut q[h * hd..][..hd];
            apply_rope(&mut qh[hd - rd..], rope, t, false);
            // The Indexer rotates BOTH q and its compressed kv; the plain attention
            // compressor does not. Rotating one side only silently changes the scores.
            hadamard(qh, hd);
        }
        ops::mmw(&mut wt, &x[t * e..][..e], w.weights_proj, e, h_n);
        for v in wt.iter_mut() {
            *v *= wscale;
        }

        // Block b is causal for token t only when b < (t+1)/ratio.
        let visible = (t + 1) / d.ratio;
        let mut score = vec![f32::NEG_INFINITY; nblk];
        for (b, s) in score.iter_mut().enumerate().take(visible.min(nblk)) {
            let row = &kvc[b * hd..][..hd];
            let mut acc = 0.0f64;
            for h in 0..h_n {
                let qh = &q[h * hd..][..hd];
                let mut dp = 0.0f64;
                for i in 0..hd {
                    dp += qh[i] as f64 * row[i] as f64;
                }
                // relu BEFORE weighting, then summed over heads.
                acc += dp.max(0.0) * wt[h] as f64;
            }
            *s = acc as f32;
        }

        let keep = d.index_topk.min(nblk);
        let mut order: Vec<usize> = (0..nblk).collect();
        order.sort_by(|&a, &b| {
            score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        let row: Vec<i64> = order
            .iter()
            .take(keep)
            .map(|&b| if b < visible { (b + offset) as i64 } else { -1 })
            .collect();
        res.push(row);
    }
    res
}

/// The ratio-128 layers have no Indexer: every causally available block is used.
pub fn compress_topk_prefill(t_len: usize, nblk: usize, ratio: usize, offset: usize) -> Vec<Vec<i64>> {
    (0..t_len)
        .map(|t| {
            let visible = (t + 1) / ratio;
            (0..nblk)
                .map(|b| if b < visible { (b + offset) as i64 } else { -1 })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod mask_tests {
    use super::*;

    // A masked slot must contribute nothing. Padding the window with index 0 instead
    // would silently give every short row extra attention on the first token.
    #[test]
    fn masked_slots_contribute_nothing() {
        let hd = 4;
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let kv = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (mut a, mut b) = (vec![0f32; hd], vec![0f32; hd]);
        sparse_attn_row(&mut a, &q, &kv, &[0, 1], -10.0, hd, 1.0);
        sparse_attn_row(&mut b, &q, &kv, &[0, 1, -1, -1], -10.0, hd, 1.0);
        for (p, s) in a.iter().zip(&b) {
            assert_eq!(p.to_bits(), s.to_bits(), "padding with -1 must change nothing");
        }
        let mut c = vec![0f32; hd];
        sparse_attn_row(&mut c, &q, &kv, &[0, 1, 0, 0], -10.0, hd, 1.0);
        assert_ne!(a[0].to_bits(), c[0].to_bits(), "padding with 0 is not the same thing");
    }

    #[test]
    fn window_row_pads_to_a_constant_width() {
        assert_eq!(window_row(0, 4, 10), vec![0, -1, -1, -1]);
        assert_eq!(window_row(2, 4, 10), vec![0, 1, 2, -1]);
        assert_eq!(window_row(5, 4, 10), vec![2, 3, 4, 5]);
        // Short sequences narrow the row rather than padding forever.
        assert_eq!(window_row(1, 8, 3), vec![0, 1, -1]);
    }
}

// ------------------------------------------------------------ full forward ----

use crate::cache::{Cache, ExpertQ};
use crate::ops::{Glu, Scoring};
use crate::st::St;

pub struct MoeDimsV4 {
    pub hidden: usize,
    pub moe_inter: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// Layers below this route by token id instead of by score.
    pub n_hash_layers: usize,
}

pub struct MoeWV4<'a> {
    pub gate: &'a [f32],
    /// `noaux_tc` selection bias. Absent on the hash-routed layers.
    pub bias: Option<&'a [f32]>,
    /// [vocab][topk] expert ids, for the hash-routed layers.
    pub tid2eid: Option<&'a [i32]>,
    pub sh1: W<'a>,
    pub sh3: W<'a>,
    pub sh2: W<'a>,
}

/// One expert in latent-free form: gate and up at `moe_inter`, then down. The routing
/// weight is applied BEFORE the down-projection, unlike K3 which weights the output.
fn expert_fwd(
    out: &mut [f32],
    x: &[f32],
    q: &ExpertQ,
    d: &MoeDimsV4,
    weight: f32,
    gu: &mut [f32],
    act: &mut [f32],
) {
    let (e, i_n) = (d.hidden, d.moe_inter);
    ops::matmul_mxfp4(&mut gu[..i_n], x, q.p1, q.s1, e, i_n, ops::MXFP4_GROUP);
    ops::matmul_mxfp4(&mut gu[i_n..], x, q.p3, q.s3, e, i_n, ops::MXFP4_GROUP);
    ops::glu(act, gu, i_n, Glu::SwigluClamped { limit: d.swiglu_limit });
    for v in act.iter_mut().take(i_n) {
        *v *= weight;
    }
    ops::matmul_mxfp4(out, act, q.p2, q.s2, i_n, e, ops::MXFP4_GROUP);
}

/// The MoE for one token. Streams its routed experts through `cache`.
#[allow(clippy::too_many_arguments)]
pub fn moe_v4(
    out: &mut [f32],
    x: &[f32],
    w: &MoeWV4,
    d: &MoeDimsV4,
    layer: usize,
    token_id: u32,
    st: &St,
    cache: &mut Cache,
    names: fn(usize, usize) -> crate::cache::ExpertNames,
) -> Result<(), String> {
    let (idx, wt) = route_v4(x, w, d, layer, token_id);

    // Hand the whole top-k over first so the reads can overlap; without this the loop
    // misses, blocks on a multi-megabyte read, computes, and misses again.
    let want: Vec<usize> = idx.iter().map(|&i| i as usize).collect();
    cache.prefetch_many(st, layer, &want, names);
    moe_v4_routed(out, x, w, d, layer, &idx, &wt, st, cache, names)
}

/// The routing half, lifted out so k speculative tokens can be routed before any expert
/// is fetched.
pub fn route_v4(
    x: &[f32],
    w: &MoeWV4,
    d: &MoeDimsV4,
    layer: usize,
    token_id: u32,
) -> (Vec<i32>, Vec<f32>) {
    let e = d.hidden;
    let mut idx = vec![0i32; d.topk];
    let mut wt = vec![0f32; d.topk];
    if layer < d.n_hash_layers {
        if let Some(t2e) = w.tid2eid {
            ops::router_hashed(&mut idx, &mut wt, x, w.gate, t2e, token_id as usize, e,
                               d.n_experts, d.topk, d.route_scale, Scoring::SqrtSoftplus);
            return (idx, wt);
        }
    }
    ops::router_scored(&mut idx, &mut wt, x, w.gate, w.bias, e, d.n_experts, d.topk,
                       true, d.route_scale, Scoring::SqrtSoftplus);
    (idx, wt)
}

/// The expert half, given an already-computed routing.
#[allow(clippy::too_many_arguments)]
pub fn moe_v4_routed(
    out: &mut [f32],
    x: &[f32],
    w: &MoeWV4,
    d: &MoeDimsV4,
    layer: usize,
    idx: &[i32],
    wt: &[f32],
    st: &St,
    cache: &mut Cache,
    names: fn(usize, usize) -> crate::cache::ExpertNames,
) -> Result<(), String> {
    let (e, i_n) = (d.hidden, d.moe_inter);

    let mut gu = vec![0f32; 2 * i_n];
    let mut act = vec![0f32; i_n];
    let mut edn = vec![0f32; e];
    out[..e].fill(0.0);
    for j in 0..d.topk {
        let ex = idx[j] as usize;
        let slot = cache
            .get(st, layer, ex, &names(layer, ex))
            .ok_or_else(|| format!("layer {layer} expert {ex} failed to load; this token is CORRUPT"))?;
        expert_fwd(&mut edn, x, &cache.expert(slot), d, wt[j], &mut gu, &mut act);
        for i in 0..e {
            out[i] += edn[i];
        }
    }

    // The shared expert runs on the ORIGINAL input, unweighted and unscaled.
    let si = i_n;
    let mut sgu = vec![0f32; 2 * si];
    let mut sact = vec![0f32; si];
    let mut sdn = vec![0f32; e];
    ops::mmw(&mut sgu[..si], x, w.sh1, e, si);
    ops::mmw(&mut sgu[si..], x, w.sh3, e, si);
    ops::glu(&mut sact, &sgu, si, Glu::SwigluClamped { limit: d.swiglu_limit });
    ops::mmw(&mut sdn, &sact, w.sh2, si, e);
    for i in 0..e {
        out[i] += sdn[i];
    }
    Ok(())
}

pub struct LayerV4<'a> {
    pub attn: AttnW<'a>,
    pub attn_dims: &'a AttnDims,
    pub compressed: Option<Compressed<'a>>,
    pub hc: crate::ops::HcLayer<'a>,
    pub attn_norm: &'a [f32],
    pub ffn_norm: &'a [f32],
    pub moe: MoeWV4<'a>,
}

/// One decoder layer: Hyper-Connections around attention, then around the MoE.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward(
    res: &mut crate::ops::HyperConnResidual,
    l: &LayerV4,
    d: &MoeDimsV4,
    layer: usize,
    ids: &[u32],
    t_len: usize,
    rope: &Rope,
    st: &St,
    cache: &mut Cache,
    names: fn(usize, usize) -> crate::cache::ExpertNames,
    rms_eps: f32,
) -> Result<(), String> {
    use crate::ops::{Residual, Sub};
    let e = d.hidden;
    let mut modin = vec![0f32; t_len * e];
    let mut hin = vec![0f32; t_len * e];
    let mut tmp = vec![0f32; t_len * e];

    let carry = res.pre(Sub::Attn, &mut modin);
    for t in 0..t_len {
        ops::rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], l.attn_norm, e, rms_eps);
    }
    attention_prefill(&mut tmp, &hin, &l.attn, l.attn_dims, t_len, rope, l.compressed.as_ref());
    res.post(Sub::Attn, &tmp, carry);

    let carry = res.pre(Sub::Mlp, &mut modin);
    for t in 0..t_len {
        ops::rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], l.ffn_norm, e, rms_eps);
    }
    for t in 0..t_len {
        let (a, b) = (&hin[t * e..][..e].to_vec(), &mut tmp[t * e..][..e]);
        moe_v4(b, a, &l.moe, d, layer, ids[t], st, cache, names)?;
    }
    res.post(Sub::Mlp, &tmp, carry);
    Ok(())
}

/// Per-layer decode state. `kv` and `qr` grow by one row per generated token, so the
/// projections for tokens already seen are computed once rather than once per step.
#[derive(Default, Clone)]
pub struct LayerState {
    /// [n][head_dim] the shared KV row per token, post-norm and post-rope.
    pub kv: Vec<f32>,
    /// [n][q_lora_rank] normalised q-LoRA per token. The Indexer reuses it, so it has to
    /// survive the step that produced it.
    pub qr: Vec<f32>,
    /// Compressed blocks, and the token count they were built from.
    ///
    /// A block folds a COMPLETED span of `ratio` tokens, so once built it never changes:
    /// rebuilding it every step is pure waste, and the waste multiplies when a batch of k
    /// tokens calls this k times. Rebuilding only when a new block completes takes the
    /// compressor from once per token to once per `ratio` tokens -- 4x at ratio 4, 128x
    /// at ratio 128.
    pub kvc: Vec<f32>,
    pub ikvc: Vec<f32>,
    pub built_from: usize,
}

impl LayerState {
    pub fn len(&self, hd: usize) -> usize {
        self.kv.len().checked_div(hd).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }
    pub fn clear(&mut self) {
        self.kv.clear();
        self.qr.clear();
        self.kvc.clear();
        self.ikvc.clear();
        self.built_from = 0;
    }

    /// Drop everything past `n` tokens. Speculative verification writes KV for tokens it
    /// may then reject; without this the cache would keep entries for tokens that never
    /// happened and every later token would attend to a future that was discarded.
    pub fn truncate(&mut self, n: usize, hd: usize, q_lora: usize) {
        self.kv.truncate(n * hd);
        self.qr.truncate(n * q_lora);
        // Blocks built from rolled-back tokens are invalid; drop them and let the next
        // call rebuild from whatever history survives.
        if self.built_from > n {
            self.kvc.clear();
            self.ikvc.clear();
            self.built_from = 0;
        }
    }
}

/// One decode step: attention for the LAST token of `hin` only, reusing cached KV for
/// every earlier token.
///
/// `hin` is the whole normalised-input history, [n][hidden], because the Compressor folds
/// spans of tokens and cannot be reconstructed from the current one. The attention itself
/// touches exactly one query row.
pub fn attention_decode(
    out: &mut [f32],
    hin: &[f32],
    w: &AttnW,
    d: &AttnDims,
    rope: &Rope,
    comp: Option<&Compressed>,
    s: &mut LayerState,
) {
    assert_eq!(
        d.compress_ratio == 0,
        comp.is_none(),
        "compress_ratio and the Compressed context must agree"
    );
    let (e, h_n, hd, rd) = (d.hidden, d.n_heads, d.head_dim, d.rope_head_dim);
    let n = hin.len() / e;
    let p = n - 1;
    let scale = (hd as f32).powf(-0.5);
    let gsz = h_n * hd / d.o_groups;
    let xt = &hin[p * e..][..e];

    // q for this token only.
    let mut qr1 = vec![0f32; d.q_lora_rank];
    ops::mmw(&mut qr1, xt, w.wq_a, e, d.q_lora_rank);
    let mut qn = vec![0f32; d.q_lora_rank];
    ops::rmsnorm(&mut qn, &qr1, w.q_norm, d.q_lora_rank, d.eps);
    s.qr.extend_from_slice(&qn);
    let mut q = vec![0f32; h_n * hd];
    ops::mmw(&mut q, &qn, w.wq_b, d.q_lora_rank, h_n * hd);
    for h in 0..h_n {
        let qh = &mut q[h * hd..][..hd];
        let mut ss = 0.0f64;
        for v in qh.iter() {
            ss += *v as f64 * *v as f64;
        }
        let inv = (1.0 / (ss / hd as f64 + d.eps as f64).sqrt()) as f32;
        for v in qh.iter_mut() {
            *v *= inv;
        }
        apply_rope(&mut qh[hd - rd..], rope, p, false);
    }

    // KV for this token, appended to the cache.
    let mut kvt = vec![0f32; hd];
    ops::mmw(&mut kvt, xt, w.wkv, e, hd);
    let src = kvt.clone();
    ops::rmsnorm(&mut kvt, &src, w.kv_norm, hd, d.eps);
    apply_rope(&mut kvt[hd - rd..], rope, p, false);
    s.kv.extend_from_slice(&kvt);
    debug_assert_eq!(s.len(hd), n, "the cache must hold exactly one row per token");

    let mut kv = s.kv.clone();
    let mut idxs = window_row(p, d.window, n);
    if let Some(c) = comp {
        // Rebuild only when another block has completed. A block covers a finished span
        // of `ratio` tokens and never changes afterwards, so this is exact, not an
        // approximation -- and it is what makes batching worth anything: the previous
        // version rebuilt the whole compressor once per token in the batch.
        let ratio = c.d.ratio.max(1);
        if s.kvc.is_empty() || n / ratio > s.built_from / ratio {
            s.kvc = compress_prefill(hin, c.w, c.d, n, rope);
            if let Some((_, _, icw, icd)) = c.indexer {
                s.ikvc = compress_prefill(hin, icw, icd, n, rope);
            }
            s.built_from = n;
        }
        let nblk = if s.kvc.is_empty() { 0 } else { s.kvc.len() / hd };
        let extra = match c.indexer {
            Some((iw, id, _, _)) => {
                indexer_prefill(&s.qr, hin, &s.ikvc, iw, id, n, rope, n)
            }
            None => compress_topk_prefill(n, nblk, ratio, n),
        };
        idxs.extend_from_slice(&extra[p]);
        kv.extend_from_slice(&s.kvc);
    }

    let mut o = vec![0f32; h_n * hd];
    let mut go = vec![0f32; d.o_groups * d.o_lora_rank];
    for h in 0..h_n {
        sparse_attn_row(
            &mut o[h * hd..][..hd],
            &q[h * hd..][..hd],
            &kv,
            &idxs,
            w.attn_sink[h],
            hd,
            scale,
        );
        apply_rope(&mut o[h * hd + hd - rd..][..rd], rope, p, true);
    }
    for g in 0..d.o_groups {
        let osl = &o[g * gsz..][..gsz];
        let dst = &mut go[g * d.o_lora_rank..][..d.o_lora_rank];
        let rows = d.o_lora_rank;
        let sub = match w.wo_a {
            W::F32(m) => W::F32(&m[g * rows * gsz..][..rows * gsz]),
            W::Bf16(m) => W::Bf16(&m[g * rows * gsz..][..rows * gsz]),
            W::F8Block { w: m, scale, block } => {
                let sb_in = gsz.div_ceil(block);
                W::F8Block {
                    w: &m[g * rows * gsz..][..rows * gsz],
                    scale: &scale[(g * rows / block) * sb_in..],
                    block,
                }
            }
            W::I8(_) => unimplemented!("wo_a is never the draft format"),
                // A k-quant packs its scales inside each 256-element super-block, so a
                // row-offset sub-slice is only meaningful when the group width is a whole
                // number of super-blocks -- and DeepSeek-V4 is never read from GGUF.
                W::Q4K(_) | W::Q5K(_) | W::Q6K(_) => unimplemented!("wo_a is never a k-quant"),
        };
        ops::mmw(dst, osl, sub, gsz, rows);
    }
    ops::mmw(out, &go, w.wo_b, d.o_groups * d.o_lora_rank, e);
}

/// One decoder layer for a single token, reusing cached KV. `hin_hist` is this layer's
/// normalised attention-input history and grows by one row per call.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_decode(
    res: &mut crate::ops::HyperConnResidual,
    l: &LayerV4,
    d: &MoeDimsV4,
    layer: usize,
    id: u32,
    rope: &Rope,
    st: &St,
    cache: &mut Cache,
    names: fn(usize, usize) -> crate::cache::ExpertNames,
    rms_eps: f32,
    state: &mut LayerState,
    hin_hist: &mut Vec<f32>,
) -> Result<(), String> {
    use crate::ops::{Residual, Sub};
    let e = d.hidden;
    let mut modin = vec![0f32; e];
    let mut hin = vec![0f32; e];
    let mut tmp = vec![0f32; e];

    let carry = res.pre(Sub::Attn, &mut modin);
    ops::rmsnorm(&mut hin, &modin, l.attn_norm, e, rms_eps);
    hin_hist.extend_from_slice(&hin);
    attention_decode(&mut tmp, hin_hist, &l.attn, l.attn_dims, rope, l.compressed.as_ref(), state);
    res.post(Sub::Attn, &tmp, carry);

    let carry = res.pre(Sub::Mlp, &mut modin);
    ops::rmsnorm(&mut hin, &modin, l.ffn_norm, e, rms_eps);
    moe_v4(&mut tmp, &hin, &l.moe, d, layer, id, st, cache, names)?;
    res.post(Sub::Mlp, &tmp, carry);
    Ok(())
}

/// Attention for the last `k` tokens of `hin`, appending each to the KV cache in turn.
///
/// Equivalent to calling `attention_decode` k times -- the cache makes each step see
/// exactly the tokens before it, so the result is causal and identical to decoding them
/// one at a time. Batched here so the CALLER can route all k tokens together and fetch
/// the union of their experts once, which is the whole point of speculative decoding on
/// a bandwidth-bound device.
pub fn attention_decode_batch(
    out: &mut [f32],
    hin: &[f32],
    w: &AttnW,
    d: &AttnDims,
    rope: &Rope,
    comp: Option<&Compressed>,
    s: &mut LayerState,
    k: usize,
) {
    let e = d.hidden;
    let n = hin.len() / e;
    debug_assert!(k <= n, "cannot decode more tokens than the history holds");
    for j in (0..k).rev() {
        let upto = (n - j) * e;
        attention_decode(&mut out[(k - 1 - j) * e..][..e], &hin[..upto], w, d, rope, comp, s);
    }
}

/// One decoder layer over `k` candidate tokens at once.
///
/// The saving over k separate decode steps is entirely in the MoE: all k tokens are
/// routed first, the UNION of their experts is fetched once, and only then are the
/// per-token expert products computed. On a bandwidth-bound device that is the whole
/// game -- k tokens cost |union| expert reads instead of 6k, and routing is concentrated
/// enough that the union is far smaller than the sum.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_spec(
    res: &mut crate::ops::HyperConnResidual,
    l: &LayerV4,
    d: &MoeDimsV4,
    layer: usize,
    ids: &[u32],
    rope: &Rope,
    st: &St,
    cache: &mut Cache,
    names: fn(usize, usize) -> crate::cache::ExpertNames,
    rms_eps: f32,
    state: &mut LayerState,
    hin_hist: &mut Vec<f32>,
) -> Result<(), String> {
    use crate::ops::{Residual, Sub};
    let (e, k) = (d.hidden, ids.len());
    let mut modin = vec![0f32; k * e];
    let mut hin = vec![0f32; k * e];
    let mut tmp = vec![0f32; k * e];

    let carry = res.pre(Sub::Attn, &mut modin);
    for t in 0..k {
        ops::rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], l.attn_norm, e, rms_eps);
    }
    hin_hist.extend_from_slice(&hin);
    attention_decode_batch(&mut tmp, hin_hist, &l.attn, l.attn_dims, rope,
                           l.compressed.as_ref(), state, k);
    res.post(Sub::Attn, &tmp, carry);

    let carry = res.pre(Sub::Mlp, &mut modin);
    for t in 0..k {
        ops::rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], l.ffn_norm, e, rms_eps);
    }
    // Route every candidate, then fetch the union once.
    let mut sel = Vec::with_capacity(k);
    let mut union: Vec<usize> = Vec::with_capacity(k * d.topk);
    for t in 0..k {
        let (idx, wt) = route_v4(&hin[t * e..][..e], &l.moe, d, layer, ids[t]);
        for &i in &idx {
            let ex = i as usize;
            if !union.contains(&ex) {
                union.push(ex);
            }
        }
        sel.push((idx, wt));
    }
    cache.prefetch_many(st, layer, &union, names);
    for t in 0..k {
        let (idx, wt) = &sel[t];
        moe_v4_routed(&mut tmp[t * e..][..e], &hin[t * e..][..e], &l.moe, d, layer,
                      idx, wt, st, cache, names)?;
    }
    res.post(Sub::Mlp, &tmp, carry);
    Ok(())
}
