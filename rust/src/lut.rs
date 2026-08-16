//! LUT-GEMM (T-MAC-style) table-lookup matmul for 4-bit weights.
//!
//! The idea, and why it is bit-exact rather than approximate: a dot product `Σ qᵢ·xᵢ` with
//! 4-bit weights `q ∈ [0,15]` can be computed WITHOUT any multiply. Decompose each nibble
//! into its four bit-planes, `q = Σ_{p<4} 2ᵖ·bₚ`, so
//! ```text
//!   Σ qᵢ xᵢ = Σ_p 2ᵖ (Σ_i bᵢₚ xᵢ)
//! ```
//! and each inner term `Σ bᵢₚ xᵢ` is a *subset-sum* of the activations. Group the
//! contraction dimension into tiles of 4: for one group of four activations there are only
//! `2⁴ = 16` possible subset-sums, so precompute those 16 values ONCE (the LUT, built from
//! the activations and reused across every output row), then each group of four weight bits
//! is a 4-bit index into that table — one lookup + one add replaces four multiplies.
//!
//! This is exact in the INTEGER domain: `q` are integers, the activations are int8-quantised
//! (Q8-style), and integer addition is associative, so the LUT reassociation is bit-for-bit
//! the naive integer dot — no rounding is introduced by the transform itself. That is the
//! whole appeal over dequant-and-multiply: same arithmetic, restructured onto table lookups
//! (which on AVX2 become one `pshufb` doing 32 lookups per instruction). The per-subblock
//! `d·scale` and the Q4_K min term are applied in f32 afterwards, identically to the scalar
//! kernel. See `dense-speedup-levers` memory for why this is the chosen lossless lever.
//!
//! This module is the SCALAR reference: it proves the decomposition is exact and packs the
//! weight bit-planes the way the AVX2 kernel will consume them. The SIMD kernel is verified
//! against this, which is verified against the naive dot — the fixture-diff discipline.

/// The group size. Four activations → a 16-entry subset-sum table → one AVX2 `pshufb`
/// (16-byte table, 4-bit indices, 32 lookups/instruction). Not a tunable: it is the width
/// `pshufb` dictates.
pub const G: usize = 4;
/// Bits per Q4_K weight nibble.
pub const NBITS: usize = 4;

/// Naive integer dot of 4-bit weights against int8 activations: `Σ q[i]·x[i]`. The reference
/// the LUT path must equal exactly. `q[i] ∈ [0,15]`; accumulated in i32 (max over 256 elems
/// is 15·127·256 ≈ 487k, comfortably inside i32).
pub fn nibble_dot_naive(q: &[u8], x: &[i8]) -> i32 {
    debug_assert_eq!(q.len(), x.len());
    let mut acc = 0i32;
    for i in 0..q.len() {
        acc += q[i] as i32 * x[i] as i32;
    }
    acc
}

/// Build the per-group subset-sum LUT from the activations: `lut[g][mask] = Σ_{j: mask has
/// bit j} x[4g+j]`. Built once per activation vector, reused across every output row — this
/// is what makes the LUT path's multiply count independent of the output dimension.
///
/// `n` must be a multiple of `G`. Returns `n/G` tables of 16 i32 each.
pub fn build_lut(x: &[i8]) -> Vec<[i32; 16]> {
    let n = x.len();
    debug_assert_eq!(n % G, 0, "LUT groups are whole tiles of {G}");
    let ngroup = n / G;
    let mut lut = vec![[0i32; 16]; ngroup];
    for (g, tbl) in lut.iter_mut().enumerate() {
        let x4 = &x[g * G..][..G];
        // mask 0..16 selects a subset of the 4 activations. Built by the standard
        // subset-sum recurrence so each entry is one add off a smaller subset.
        for mask in 1..16usize {
            let low = mask & mask.wrapping_neg(); // lowest set bit
            let j = low.trailing_zeros() as usize;
            tbl[mask] = tbl[mask ^ low] + x4[j] as i32;
        }
    }
    lut
}

/// The LUT dot: `Σ q[i]·x[i]` via bit-plane subset-sum lookups, given a prebuilt `lut`.
///
/// For each of the 4 bit-planes, gather each group's 4 weight bits into a 4-bit index, look
/// up that group's subset-sum, sum the groups, and shift-accumulate by the plane weight.
/// Bit-for-bit equal to [`nibble_dot_naive`] because it is the same integer sum reassociated.
pub fn nibble_dot_lut(q: &[u8], lut: &[[i32; 16]]) -> i32 {
    let ngroup = lut.len();
    debug_assert_eq!(q.len(), ngroup * G);
    let mut acc = 0i32;
    for p in 0..NBITS {
        let mut plane = 0i32;
        for (g, tbl) in lut.iter().enumerate() {
            let base = g * G;
            // index = the p-th bit of each of the 4 nibbles in this group.
            let idx = (((q[base] >> p) & 1) as usize)
                | ((((q[base + 1] >> p) & 1) as usize) << 1)
                | ((((q[base + 2] >> p) & 1) as usize) << 2)
                | ((((q[base + 3] >> p) & 1) as usize) << 3);
            plane += tbl[idx];
        }
        acc += plane << p;
    }
    acc
}

/// Convenience: build the LUT and dot in one call (the per-row-reused LUT is the point, so
/// prefer [`build_lut`] + [`nibble_dot_lut`] in a real matmul; this is for tests).
pub fn nibble_dot(q: &[u8], x: &[i8]) -> i32 {
    nibble_dot_lut(q, &build_lut(x))
}

// --- Q4_K-aware full matvec via the LUT (scalar reference for the AVX2 kernel) ---

const QK_K: usize = 256;
const Q4K_BLOCK: usize = 144;
const SUB: usize = 32; // elements per Q4_K sub-block (one 6-bit scale/min each)

/// The Q4_K 6-bit scale/min unpack — a local copy of the gguf packing so this module is
/// self-contained. Sub-blocks 0-3 come straight from bytes 0-7; 4-7 are split across bytes
/// 8-11 and the top two bits of 0-7.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// The int8-quantised form of one activation vector, precomputed once and reused across every
/// output row — the LUTs, the per-sub-block scales, and the per-sub-block int8 sums (for the
/// Q4_K min term). This is the "activation side" of LUT-GEMM.
pub struct ActLut {
    /// Per sub-block (32 elems): subset-sum LUT over its 8 groups of 4.
    lut: Vec<[[i32; 16]; SUB / G]>,
    /// Per sub-block: `absmax/127`, the int8 scale.
    sx: Vec<f32>,
    /// Per sub-block: `Σ q8`, for the affine min term.
    sumq8: Vec<i32>,
}

impl ActLut {
    /// Quantise `x` (length a multiple of 32) sub-block by sub-block to int8 (Q8-style, one
    /// absmax scale per 32) and build the per-sub-block LUTs. `round-to-nearest`, ties away
    /// from zero via `+0.5` on the magnitude — pin this to whatever the AVX2 path uses.
    pub fn build(x: &[f32]) -> ActLut {
        assert_eq!(x.len() % SUB, 0, "activation length must be a whole number of sub-blocks");
        let nsub = x.len() / SUB;
        let mut lut = Vec::with_capacity(nsub);
        let mut sx = Vec::with_capacity(nsub);
        let mut sumq8 = Vec::with_capacity(nsub);
        for s in 0..nsub {
            let xs = &x[s * SUB..][..SUB];
            let amax = xs.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let mut q8 = [0i8; SUB];
            let mut sum = 0i32;
            for l in 0..SUB {
                // round-to-nearest, ties away from zero, clamped to int8.
                let v = (xs[l] * inv).round().clamp(-127.0, 127.0) as i32;
                q8[l] = v as i8;
                sum += v;
            }
            // one LUT per group of 4 within the sub-block
            let mut sub_lut = [[0i32; 16]; SUB / G];
            for (g, tbl) in sub_lut.iter_mut().enumerate() {
                let x4 = &q8[g * G..][..G];
                for mask in 1..16usize {
                    let low = mask & mask.wrapping_neg();
                    let j = low.trailing_zeros() as usize;
                    tbl[mask] = tbl[mask ^ low] + x4[j] as i32;
                }
            }
            lut.push(sub_lut);
            sx.push(scale);
            sumq8.push(sum);
        }
        ActLut { lut, sx, sumq8 }
    }
}

/// `y[r] = row r of the Q4_K matrix, dotted with x`, computed via LUT lookups over int8
/// activations. NOT bit-identical to [`crate::gguf::matmul_q4k`] (that uses f32 activations);
/// it is the Q8-activation computation — exact in the integer core, differing from the f32
/// kernel only by the int8 activation rounding, which is the same tradeoff llama.cpp's k-quant
/// path makes. `k_in` is a multiple of `QK_K`.
pub fn matmul_q4k_lut(y: &mut [f32], act: &ActLut, src: &[u8], k_in: usize, rows: usize) {
    assert_eq!(k_in % QK_K, 0, "k-quant rows are whole super-blocks");
    let nb = k_in / QK_K; // super-blocks per row
    let stride = nb * Q4K_BLOCK;
    for r in 0..rows {
        let row = &src[r * stride..][..stride];
        let mut acc = 0f64;
        for b in 0..nb {
            let blk = &row[b * Q4K_BLOCK..][..Q4K_BLOCK];
            let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
            let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            for sub in 0..8 {
                let s = b * 8 + sub;
                let (p, half) = (sub / 2, sub & 1);
                // extract this sub-block's 32 nibbles
                let mut q = [0u8; SUB];
                for l in 0..SUB {
                    q[l] = (qs[p * SUB + l] >> (4 * half)) & 0xF;
                }
                let (sc, m) = scale_min_k4(sub, scales);
                let idot = nibble_dot_lut(&q, &act.lut[s]) as f64;
                let sx = act.sx[s] as f64;
                // Σ(d·sc·q − dmin·m)·x ≈ sx·(d·sc·idot − dmin·m·Σq8)
                acc += sx * (d * sc as f64 * idot - dmin * m as f64 * act.sumq8[s] as f64);
            }
        }
        y[r] = acc as f32;
    }
}

// --- AVX2 pshufb kernel: 32 output rows' integer dot for one sub-block at once ---
//
// For a fixed sub-block (32 activations = 8 groups of 4) with its 8 subset-sum LUTs, and 32
// weight rows whose per-(group,plane) 4-bit indices are pre-packed, compute all 32 rows'
// `Σ nibble·q8` in registers. Each (group, plane) is ONE `pshufb` doing 32 table lookups.
//
// LUT values are int16 (subset-sum of four int8 ∈ [-508,508]), so each lookup is two
// `pshufb` (low byte, high byte) reassembled by `unpack_epi8`. The unpack permutation lands
// rows 0-7 / 8-15 / 16-23 / 24-31 in natural order across four int32 accumulators, so no
// final shuffle is needed. Bit-identical to the scalar [`nibble_dot_lut`] per row.

/// Per-(group, plane) 4-bit indices for 32 rows: `idx[group][plane][row]`.
pub type SubIdx = [[[u8; 32]; NBITS]; SUB / G];

/// The i16-LUT form, kept as the directly-testable reference for the AVX2 core (the
/// production path uses [`subblock_idot_pre`] with precomputed byte tables).
#[cfg(all(target_arch = "x86_64", test))]
#[target_feature(enable = "avx2")]
unsafe fn subblock_idot_x32(lut16: &[[i16; 16]; SUB / G], idx: &SubIdx) -> [i32; 32] {
    use std::arch::x86_64::*;
    // Four int32 accumulators: rows 0-7, 8-15, 16-23, 24-31.
    let mut a0 = _mm256_setzero_si256();
    let mut a1 = _mm256_setzero_si256();
    let mut a2 = _mm256_setzero_si256();
    let mut a3 = _mm256_setzero_si256();

    for g in 0..(SUB / G) {
        // Split this group's 16 int16 LUT entries into low/high byte tables, broadcast to
        // both 128-bit lanes (pshufb is lane-local, so each lane needs the same 16-byte table).
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        for k in 0..16 {
            lo[k] = (lut16[g][k] & 0xFF) as u8;
            hi[k] = (lut16[g][k] >> 8) as u8;
        }
        let lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(lo.as_ptr().cast()));
        let lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi.as_ptr().cast()));

        for p in 0..NBITS {
            // Shift amount is a runtime value, so use the per-lane variable shift.
            let sh = _mm256_set1_epi32(p as i32);
            let iv = _mm256_loadu_si256(idx[g][p].as_ptr().cast());
            let lb = _mm256_shuffle_epi8(lut_lo, iv); // low bytes of 32 lookups
            let hb = _mm256_shuffle_epi8(lut_hi, iv); // high bytes
            // reassemble int16: interleave low/high bytes (little-endian i16 per row)
            let ulo = _mm256_unpacklo_epi8(lb, hb); // int16 rows [0-7 | 16-23]
            let uhi = _mm256_unpackhi_epi8(lb, hb); // int16 rows [8-15 | 24-31]
            // widen to int32 (sign-extend) and shift-accumulate by the plane weight.
            let w0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ulo)); // rows 0-7
            let w2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ulo, 1)); // rows 16-23
            let w1 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(uhi)); // rows 8-15
            let w3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(uhi, 1)); // rows 24-31
            a0 = _mm256_add_epi32(a0, _mm256_sllv_epi32(w0, sh));
            a1 = _mm256_add_epi32(a1, _mm256_sllv_epi32(w1, sh));
            a2 = _mm256_add_epi32(a2, _mm256_sllv_epi32(w2, sh));
            a3 = _mm256_add_epi32(a3, _mm256_sllv_epi32(w3, sh));
        }
    }

    let mut out = [0i32; 32];
    _mm256_storeu_si256(out.as_mut_ptr().cast(), a0);
    _mm256_storeu_si256(out[8..].as_mut_ptr().cast(), a1);
    _mm256_storeu_si256(out[16..].as_mut_ptr().cast(), a2);
    _mm256_storeu_si256(out[24..].as_mut_ptr().cast(), a3);
    out
}

/// Q4_K weights repacked for the AVX2 LUT kernel: per-(tile, super-block, sub-block) the
/// 4-bit `pshufb` indices for 32 rows, plus the per-(row, sub-block) affine scales `d·sc`
/// and `dmin·m` pulled out of the k-quant blocks. Built ONCE at load (weights are static);
/// the per-token cost is only the activation LUT + the lookups. ~2x the weight size on disk,
/// the standard LUT-GEMM preprocessing tradeoff.
pub struct Q4kLutW {
    rows: usize,
    nb: usize,    // super-blocks per row
    ntile: usize, // ceil(rows/32)
    idx: Vec<SubIdx>, // [tile*nb*8 + b*8 + sub]
    aw: Vec<f32>,     // [row*(nb*8) + s] = d·sc
    bw: Vec<f32>,     // [row*(nb*8) + s] = dmin·m
}

impl Q4kLutW {
    /// `(k_in, out)` this repack was built for — the caller checks these match its matmul so a
    /// shape mismatch falls back to the safe path instead of reading out of bounds.
    pub fn dims(&self) -> (usize, usize) {
        (self.nb * QK_K, self.rows)
    }

    pub fn repack(src: &[u8], k_in: usize, rows: usize) -> Q4kLutW {
        assert_eq!(k_in % QK_K, 0);
        let nb = k_in / QK_K;
        let ntile = rows.div_ceil(32);
        let stride = nb * Q4K_BLOCK;
        let mut idx = vec![[[[0u8; 32]; NBITS]; SUB / G]; ntile * nb * 8];
        let mut aw = vec![0f32; rows * nb * 8];
        let mut bw = vec![0f32; rows * nb * 8];
        for t in 0..ntile {
            for rr in 0..32 {
                let row = t * 32 + rr;
                if row >= rows {
                    break;
                }
                let rb = &src[row * stride..][..stride];
                for b in 0..nb {
                    let blk = &rb[b * Q4K_BLOCK..][..Q4K_BLOCK];
                    let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                    let scales = &blk[4..16];
                    let qs = &blk[16..144];
                    for sub in 0..8 {
                        let (sc, m) = scale_min_k4(sub, scales);
                        let s = b * 8 + sub;
                        aw[row * (nb * 8) + s] = d * sc as f32;
                        bw[row * (nb * 8) + s] = dmin * m as f32;
                        let (p_sb, half) = (sub / 2, sub & 1);
                        let cell = &mut idx[t * nb * 8 + b * 8 + sub];
                        for g in 0..(SUB / G) {
                            for p in 0..NBITS {
                                let mut ix = 0u8;
                                for j in 0..G {
                                    let nib = (qs[p_sb * SUB + G * g + j] >> (4 * half)) & 0xF;
                                    ix |= ((nib >> p) & 1) << j;
                                }
                                cell[g][p][rr] = ix;
                            }
                        }
                    }
                }
            }
        }
        Q4kLutW { rows, nb, ntile, idx, aw, bw }
    }
}

/// `y = W·x` via the AVX2 pshufb LUT kernel. Bit-identical to the scalar [`matmul_q4k_lut`]
/// (same int8 activations, same integer dot), just computed 32 rows at a time through
/// `pshufb`. Falls back to the scalar path when AVX2 is unavailable.
pub fn matmul_q4k_lut_avx2(y: &mut [f32], act: &ActLut, w: &Q4kLutW) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            matmul_q4k_lut_avx2_inner(y, act, w);
            return;
        }
    }
    // scalar fallback: reuse the reference by reconstructing the byte layout would be costly;
    // instead recompute the integer dot per sub-block from the repacked indices in scalar.
    matmul_q4k_lut_from_idx_scalar(y, act, w);
}

/// The i16 activation LUTs, one per sub-block, cast from the ActLut's i32 tables once per
/// matmul (subset-sums of four int8 fit i16). Activation-side, shared across all row tiles.
fn act_lut16(act: &ActLut) -> Vec<[[i16; 16]; SUB / G]> {
    act.lut
        .iter()
        .map(|sub| {
            let mut o = [[0i16; 16]; SUB / G];
            for g in 0..(SUB / G) {
                for k in 0..16 {
                    o[g][k] = sub[g][k] as i16;
                }
            }
            o
        })
        .collect()
}

/// Precomputed low/high byte LUT tables for one sub-block's 8 groups — activation-side, built
/// once and reused across every row tile (this hoist is what makes the kernel worth running).
struct SubTables {
    lo: [[u8; 16]; SUB / G],
    hi: [[u8; 16]; SUB / G],
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn subblock_idot_pre(t: &SubTables, idx: &SubIdx) -> [i32; 32] {
    use std::arch::x86_64::*;
    let mut a0 = _mm256_setzero_si256();
    let mut a1 = _mm256_setzero_si256();
    let mut a2 = _mm256_setzero_si256();
    let mut a3 = _mm256_setzero_si256();
    for g in 0..(SUB / G) {
        let lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(t.lo[g].as_ptr().cast()));
        let lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(t.hi[g].as_ptr().cast()));
        for p in 0..NBITS {
            let sh = _mm256_set1_epi32(p as i32);
            let iv = _mm256_loadu_si256(idx[g][p].as_ptr().cast());
            let lb = _mm256_shuffle_epi8(lut_lo, iv);
            let hb = _mm256_shuffle_epi8(lut_hi, iv);
            let ulo = _mm256_unpacklo_epi8(lb, hb);
            let uhi = _mm256_unpackhi_epi8(lb, hb);
            let w0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(ulo));
            let w2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(ulo, 1));
            let w1 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(uhi));
            let w3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(uhi, 1));
            a0 = _mm256_add_epi32(a0, _mm256_sllv_epi32(w0, sh));
            a1 = _mm256_add_epi32(a1, _mm256_sllv_epi32(w1, sh));
            a2 = _mm256_add_epi32(a2, _mm256_sllv_epi32(w2, sh));
            a3 = _mm256_add_epi32(a3, _mm256_sllv_epi32(w3, sh));
        }
    }
    let mut out = [0i32; 32];
    _mm256_storeu_si256(out.as_mut_ptr().cast(), a0);
    _mm256_storeu_si256(out[8..].as_mut_ptr().cast(), a1);
    _mm256_storeu_si256(out[16..].as_mut_ptr().cast(), a2);
    _mm256_storeu_si256(out[24..].as_mut_ptr().cast(), a3);
    out
}

#[cfg(target_arch = "x86_64")]
fn matmul_q4k_lut_avx2_inner(y: &mut [f32], act: &ActLut, w: &Q4kLutW) {
    use rayon::prelude::*;
    let lut16 = act_lut16(act);
    let nsub = w.nb * 8;
    // Precompute the byte tables for every sub-block ONCE (activation-side, tile-independent),
    // shared read-only across the rayon workers.
    let tables: Vec<SubTables> = lut16
        .iter()
        .map(|sub| {
            let mut t = SubTables { lo: [[0u8; 16]; SUB / G], hi: [[0u8; 16]; SUB / G] };
            for g in 0..(SUB / G) {
                for k in 0..16 {
                    t.lo[g][k] = (sub[g][k] & 0xFF) as u8;
                    t.hi[g][k] = (sub[g][k] >> 8) as u8;
                }
            }
            t
        })
        .collect();

    // One rayon task per 32-row tile — tiles write disjoint outputs, so this is the same
    // embarrassingly-parallel fan-out the baseline kernel uses, and lets a 16-thread vs
    // 16-thread comparison be fair. Tile OUTER / sub-block INNER keeps the accumulator hot.
    y.par_chunks_mut(32).enumerate().for_each(|(t, ychunk)| {
        let base = t * 32;
        let nrow = ychunk.len();
        let mut yacc = [0f64; 32];
        for b in 0..w.nb {
            for sub in 0..8 {
                let s = b * 8 + sub;
                let (sx, sumq8) = (act.sx[s] as f64, act.sumq8[s] as f64);
                // SAFETY: avx2 verified by the dispatcher; all threads share this CPU.
                let idot = unsafe { subblock_idot_pre(&tables[s], &w.idx[t * nsub + b * 8 + sub]) };
                for rr in 0..nrow {
                    let row = base + rr;
                    let a = w.aw[row * nsub + s] as f64;
                    let bb = w.bw[row * nsub + s] as f64;
                    yacc[rr] += sx * (a * idot[rr] as f64 - bb * sumq8);
                }
            }
        }
        for rr in 0..nrow {
            ychunk[rr] = yacc[rr] as f32;
        }
    });
}

/// Scalar equivalent of the AVX2 kernel, driven from the SAME repacked indices — the
/// portable fallback, and a second independent path the AVX2 result can be checked against.
fn matmul_q4k_lut_from_idx_scalar(y: &mut [f32], act: &ActLut, w: &Q4kLutW) {
    let lut16 = act_lut16(act);
    let nsub = w.nb * 8;
    for t in 0..w.ntile {
        for rr in 0..32 {
            let row = t * 32 + rr;
            if row >= w.rows {
                break;
            }
            let mut acc = 0f64;
            for b in 0..w.nb {
                for sub in 0..8 {
                    let s = b * 8 + sub;
                    let cell = &w.idx[t * nsub + b * 8 + sub];
                    // integer dot for this row via the bit-plane formula
                    let mut idot = 0i32;
                    for p in 0..NBITS {
                        let mut plane = 0i32;
                        for g in 0..(SUB / G) {
                            plane += lut16[s][g][cell[g][p][rr] as usize] as i32;
                        }
                        idot += plane << p;
                    }
                    let a = w.aw[row * nsub + s] as f64;
                    let bb = w.bw[row * nsub + s] as f64;
                    acc += act.sx[s] as f64 * (a * idot as f64 - bb * act.sumq8[s] as f64);
                }
            }
            y[row] = acc as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// The AVX2 pshufb core must equal the scalar bit-plane formula EXACTLY for all 32 rows.
    /// This is the load-bearing correctness gate for the SIMD kernel: a wrong lane, a wrong
    /// unpack, or a sign-extension bug produces a plausible-but-wrong integer, so only exact
    /// per-row equality proves it.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_core_equals_scalar_formula() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut s = 0xBEEFu64;
        for _trial in 0..20 {
            // realizable-range int16 LUTs (subset-sum of four int8) and 0..15 indices
            let mut lut = [[0i16; 16]; SUB / G];
            for g in 0..(SUB / G) {
                for k in 0..16 {
                    lut[g][k] = (lcg(&mut s) as i32 % 1017 - 508) as i16;
                }
            }
            let mut idx: SubIdx = [[[0u8; 32]; NBITS]; SUB / G];
            for g in 0..(SUB / G) {
                for p in 0..NBITS {
                    for r in 0..32 {
                        idx[g][p][r] = (lcg(&mut s) & 0xF) as u8;
                    }
                }
            }
            // scalar reference: acc[row] = Σ_p 2^p Σ_g lut[g][idx[g][p][row]]
            let mut want = [0i32; 32];
            for (r, w) in want.iter_mut().enumerate() {
                let mut acc = 0i32;
                for p in 0..NBITS {
                    let mut plane = 0i32;
                    for g in 0..(SUB / G) {
                        plane += lut[g][idx[g][p][r] as usize] as i32;
                    }
                    acc += plane << p;
                }
                *w = acc;
            }
            let got = unsafe { subblock_idot_x32(&lut, &idx) };
            assert_eq!(got, want, "AVX2 core diverged from scalar on trial {_trial}");
        }
    }

    /// The LUT dot must equal the naive integer dot BIT-FOR-BIT (both integer, so `==`), for
    /// every length and random content. This is the mutation-resistant core assertion: a
    /// wrong bit-plane index or a wrong subset-sum still yields a plausible finite integer,
    /// so only exact equality proves the transform.
    #[test]
    fn lut_equals_naive_exactly() {
        let mut s = 12345u64;
        for &n in &[4usize, 8, 32, 256, 512] {
            let q: Vec<u8> = (0..n).map(|_| (lcg(&mut s) & 0xF) as u8).collect();
            let x: Vec<i8> = (0..n).map(|_| (lcg(&mut s) as i32 % 255 - 127) as i8).collect();
            let a = nibble_dot_naive(&q, &x);
            let b = nibble_dot(&q, &x);
            assert_eq!(a, b, "n={n}: LUT {b} != naive {a}");
        }
    }

    /// Boundary values: all-zero, all-15 nibbles, extreme int8, so the i32 accumulation and
    /// the top bit-plane (p=3, weight 8) are exercised at their limits.
    #[test]
    fn lut_handles_extremes() {
        let n = 256;
        // (nibble, activation) generators, exercising the i32 accumulation and the top
        // bit-plane (p=3, weight 8) at their limits.
        let q_gen: [fn(usize) -> u8; 4] = [|_| 0, |_| 15, |_| 15, |i| (i % 16) as u8];
        let x_gen: [fn(usize) -> i8; 4] =
            [|_| 127, |_| 127, |_| -128, |i| (i as i32 % 255 - 127) as i8];
        for (qf, xf) in q_gen.into_iter().zip(x_gen) {
            let q: Vec<u8> = (0..n).map(qf).collect();
            let x: Vec<i8> = (0..n).map(xf).collect();
            assert_eq!(nibble_dot_naive(&q, &x), nibble_dot(&q, &x));
        }
    }

    /// The AVX2 matvec must equal the scalar-from-index path BIT-FOR-BIT (they share the
    /// repacked weights and scale representation, so any difference is a SIMD bug), and must
    /// track the f32 kernel within the int8-activation bound. Row counts not divisible by 32
    /// exercise the tail-tile padding.
    #[test]
    fn avx2_matvec_matches_scalar_and_tracks_f32() {
        for &(k_in, rows) in &[(QK_K * 3, 100usize), (QK_K * 2, 128), (QK_K, 33)] {
            let x: Vec<f32> = (0..k_in).map(|i| ((i as f64 * 0.017).sin() * 0.6) as f32).collect();
            let src = rand_q4k(rows * (k_in / QK_K), 424242);
            let act = ActLut::build(&x);
            let w = Q4kLutW::repack(&src, k_in, rows);

            let mut y_avx = vec![0f32; rows];
            matmul_q4k_lut_avx2(&mut y_avx, &act, &w);
            let mut y_idx = vec![0f32; rows];
            matmul_q4k_lut_from_idx_scalar(&mut y_idx, &act, &w);
            for r in 0..rows {
                assert_eq!(
                    y_avx[r].to_bits(),
                    y_idx[r].to_bits(),
                    "k_in={k_in} rows={rows} row {r}: AVX2 {} != scalar {}",
                    y_avx[r],
                    y_idx[r]
                );
            }

            let mut y_ref = vec![0f32; rows];
            crate::gguf::matmul_q4k(&mut y_ref, &x, &src, k_in, rows);
            let scale = y_ref.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-6);
            let rms = (y_ref.iter().zip(&y_avx).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>()
                / rows as f64)
                .sqrt() as f32
                / scale;
            assert!(rms < 0.02, "k_in={k_in} rows={rows}: AVX2 drifted from f32, rms {rms}");
        }
    }

    /// A random-but-valid Q4_K super-block stream, f16 scale fields pinned finite (0.25 /
    /// 0.125) exactly as the gguf test fixtures do, so a stray NaN cannot make a comparison
    /// vacuously pass.
    fn rand_q4k(nblocks: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![0u8; nblocks * Q4K_BLOCK];
        let mut s = seed | 1;
        for b in v.iter_mut() {
            *b = (lcg(&mut s) & 0xFF) as u8;
        }
        for i in 0..nblocks {
            let blk = &mut v[i * Q4K_BLOCK..][..Q4K_BLOCK];
            blk[0..2].copy_from_slice(&0x3400u16.to_le_bytes()); // d = 0.25
            blk[2..4].copy_from_slice(&0x3000u16.to_le_bytes()); // dmin = 0.125
        }
        v
    }

    /// The full LUT matvec vs the engine's f32 Q4_K kernel. Not bit-identical (int8 vs f32
    /// activations) — this bounds the error the int8 activation introduces, and confirms the
    /// LUT pipeline (nibble unpack + scales + min term) is wired correctly. A wiring bug would
    /// blow the error up by orders of magnitude, so a tight bound is a real correctness gate.
    #[test]
    fn lut_matvec_tracks_the_f32_kernel() {
        let (k_in, rows) = (QK_K * 4, 128usize);
        let x: Vec<f32> = (0..k_in).map(|i| ((i as f64 * 0.013).sin() * 0.7) as f32).collect();
        let src = rand_q4k(rows * (k_in / QK_K), 20260816);

        let mut refy = vec![0f32; rows];
        crate::gguf::matmul_q4k(&mut refy, &x, &src, k_in, rows);

        let act = ActLut::build(&x);
        let mut luty = vec![0f32; rows];
        matmul_q4k_lut(&mut luty, &act, &src, k_in, rows);

        let scale = refy.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-6);
        let mut max_rel = 0f32;
        let mut sse = 0f64;
        for r in 0..rows {
            let e = (luty[r] - refy[r]).abs();
            max_rel = max_rel.max(e / scale);
            sse += (e as f64).powi(2);
        }
        let rms_rel = (sse / rows as f64).sqrt() as f32 / scale;
        // int8 activation over 256-wide rows: error is a small fraction of the row scale.
        assert!(
            rms_rel < 0.02 && max_rel < 0.05,
            "LUT matvec drifted from f32 kernel: rms_rel {rms_rel:.4}, max_rel {max_rel:.4}"
        );
    }

    /// Teeth: a single wrong nibble (or a corrupted LUT) MUST change the result, or the test
    /// proves nothing. Mutating one weight bit changes the naive dot; the LUT must track it.
    #[test]
    fn a_single_flipped_bit_changes_the_result() {
        let mut s = 777u64;
        let n = 64;
        let mut q: Vec<u8> = (0..n).map(|_| (lcg(&mut s) & 0xF) as u8).collect();
        let x: Vec<i8> = (0..n).map(|_| (lcg(&mut s) as i32 % 255 - 127) as i8).collect();
        let before = nibble_dot(&q, &x);
        // Flip bit 3 of a nibble whose activation is non-zero so the change is observable.
        let i = (0..n).find(|&i| x[i] != 0).unwrap();
        q[i] ^= 0b1000;
        let after = nibble_dot(&q, &x);
        assert_ne!(before, after, "flipping a weight bit must change the dot");
        // And it must still match naive after the mutation.
        assert_eq!(nibble_dot_naive(&q, &x), after);
    }
}
