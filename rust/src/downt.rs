//! Neuron-major transpose of the FFN `down` weight, so activation sparsity actually skips
//! work instead of just zeroing it.
//!
//! `down` is stored `[hidden, inter]` row-major: neuron `i`'s output direction is COLUMN `i`,
//! strided by `inter`. A k-quant super-block holds 256 CONTIGUOUS values along a row, so a
//! column touches one value out of each of `hidden/256`-worth of blocks — to skip neuron `i`
//! you would still have to decode every block. Column sparsity in a row-major matrix buys
//! nothing (see `dense-speedup-levers`).
//!
//! The fix is to store `downᵀ = [inter, hidden]`: now neuron `i` is a CONTIGUOUS ROW, and the
//! down projection `y[h] = Σ_i aᵢ·down[h,i]` becomes `y = Σ_i aᵢ·downᵀ[i,:]` — a weighted sum
//! of rows. Skipping neuron `i` (its `aᵢ` zeroed by the certificate) means its row is never
//! decoded. That is the whole speed win of certified sparsity.
//!
//! Storage is Q8: one int8 per weight plus one f32 scale per 32, quantised from the exact
//! dequantised `down` values. That adds a small (~8-bit) requant error on top of the model's
//! own Q4_K/Q6_K — folded into the certificate's perturbation budget, so the emitted token is
//! still argmax-exact. The repack is a one-time per-layer cost at first use; the transposed
//! weight is then resident and the down projection stops streaming entirely.

const QK: usize = 32; // elements per Q8 block
const BLK: usize = 4 + QK; // f32 scale + 32 int8

/// `downᵀ` for one layer: `inter` rows of `hidden` weights, each row Q8-quantised in blocks
/// of 32. Row `i` is neuron `i`; skipping it skips its decode.
pub struct DownT {
    pub hidden: usize,
    pub inter: usize,
    row_bytes: usize,
    data: Vec<u8>,
}

/// Dequantise one Q8 block into `out[..32]`.
#[inline]
fn deq_block(blk: &[u8], out: &mut [f32]) {
    let scale = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
    for l in 0..QK {
        out[l] = blk[4 + l] as i8 as f32 * scale;
    }
}

/// Quantise `x[..32]` into one Q8 block (absmax scale, round to nearest).
#[inline]
fn q_block(x: &[f32], blk: &mut [u8]) {
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let scale = amax / 127.0;
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    blk[0..4].copy_from_slice(&scale.to_le_bytes());
    for l in 0..QK {
        blk[4 + l] = ((x[l] * inv).round().clamp(-127.0, 127.0) as i32) as i8 as u8;
    }
}

impl DownT {
    /// Repack a k-quant `down` weight (`[hidden, inter]`, row-major, `block_bytes`/`dq` its
    /// quant) into the neuron-major Q8 transpose. `inter` and `hidden` must be multiples of
    /// 32 and 256 respectively (Q8 blocks / super-blocks).
    pub fn repack(
        down_bytes: &[u8],
        hidden: usize,
        inter: usize,
        block_bytes: usize,
        dq: fn(&mut [f32], &[u8], usize),
    ) -> DownT {
        assert_eq!(hidden % QK, 0, "hidden must be a whole number of Q8 blocks");
        assert_eq!(inter % 256, 0, "inter must be a whole number of super-blocks");
        let nb = inter / 256;
        let src_stride = nb * block_bytes;

        // Dequantise down row by row and scatter into a transposed f32 buffer.
        let mut t = vec![0f32; inter * hidden];
        let mut row = vec![0f32; inter];
        for h in 0..hidden {
            dq(&mut row, &down_bytes[h * src_stride..][..src_stride], nb);
            for i in 0..inter {
                t[i * hidden + h] = row[i];
            }
        }

        // Quantise each transposed row (neuron) to Q8.
        let row_bytes = hidden / QK * BLK;
        let mut data = vec![0u8; inter * row_bytes];
        for i in 0..inter {
            let src = &t[i * hidden..][..hidden];
            let dst = &mut data[i * row_bytes..][..row_bytes];
            for b in 0..hidden / QK {
                q_block(&src[b * QK..][..QK], &mut dst[b * BLK..][..BLK]);
            }
        }
        DownT { hidden, inter, row_bytes, data }
    }

    /// Resident bytes of the transposed weight.
    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    /// The down projection `y[h] = Σ_i a[i]·downᵀ[i,h]`, decoding ONLY the rows whose `a[i]`
    /// is non-zero. `a` is the (already certified-sparse) activation vector; skipped neurons
    /// have `a[i] == 0` and their row is never touched — the realised speed win.
    ///
    /// Parallel over kept neurons (each rayon task folds a partial `hidden` accumulator, then
    /// they are reduced), so this competes with the baseline down matmul on equal thread
    /// count rather than losing the skip to single-threading.
    pub fn matvec_skip(&self, a: &[f32], y: &mut [f32]) {
        use rayon::prelude::*;
        debug_assert_eq!(a.len(), self.inter);
        debug_assert_eq!(y.len(), self.hidden);
        let (hidden, row_bytes, data) = (self.hidden, self.row_bytes, &self.data);
        let acc = (0..self.inter)
            .into_par_iter()
            .filter(|&i| a[i] != 0.0)
            .fold(
                || vec![0f64; hidden],
                |mut acc, i| {
                    let ai = a[i] as f64;
                    let rowb = &data[i * row_bytes..][..row_bytes];
                    let mut blk = [0f32; QK];
                    for b in 0..hidden / QK {
                        deq_block(&rowb[b * BLK..][..BLK], &mut blk);
                        let base = b * QK;
                        for l in 0..QK {
                            acc[base + l] += ai * blk[l] as f64;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0f64; hidden],
                |mut a, b| {
                    for h in 0..hidden {
                        a[h] += b[h];
                    }
                    a
                },
            );
        for h in 0..hidden {
            y[h] = acc[h] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a random Q6_K `down` [hidden, inter] and check the transposed Q8 matvec matches a
    // dense f32 reference within the Q8 quant bound — and that zeroing an activation exactly
    // removes that neuron's contribution (the skip is correct, not approximate).
    fn rand_q6k(nblocks: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![0u8; nblocks * 210];
        let mut s = seed | 1;
        for b in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        for i in 0..nblocks {
            // Q6_K scale is the last 2 bytes of the 210-byte block; pin it finite.
            v[i * 210 + 208..i * 210 + 210].copy_from_slice(&0x3400u16.to_le_bytes());
        }
        v
    }

    #[test]
    fn transposed_matvec_tracks_dense_and_skips_exactly() {
        let (hidden, inter) = (256usize, 512usize);
        let src = rand_q6k(hidden * (inter / 256), 12321);
        let dt = DownT::repack(&src, hidden, inter, 210, crate::gguf::q6k_dequant);

        let mut s = 5u64;
        let a: Vec<f32> = (0..inter)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 40) as i64 as f32 / (1i64 << 30) as f32
            })
            .collect();

        // dense reference: y[h] = Σ_i a[i]·down[h,i], via the engine's own kernel.
        let mut dense = vec![0f32; hidden];
        crate::gguf::matmul_q6k(&mut dense, &a, &src, inter, hidden);

        let mut y = vec![0f32; hidden];
        dt.matvec_skip(&a, &mut y);
        let scale = dense.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
        let rms = (dense.iter().zip(&y).map(|(&d, &v)| ((d - v) as f64).powi(2)).sum::<f64>()
            / hidden as f64)
            .sqrt() as f32
            / scale;
        assert!(rms < 0.02, "transposed Q8 matvec drifted from dense: rms {rms}");

        // Skipping a neuron must remove EXACTLY its contribution: zero a[i], recompute, and
        // check the delta equals a full-vs-that-neuron difference.
        let mut a2 = a.clone();
        let victim = 7usize;
        a2[victim] = 0.0;
        let mut y2 = vec![0f32; hidden];
        dt.matvec_skip(&a2, &mut y2);
        // reconstruct neuron `victim`'s contribution from downᵀ and compare.
        let mut only = vec![0f32; hidden];
        let mut solo = vec![0f32; inter];
        solo[victim] = a[victim];
        dt.matvec_skip(&solo, &mut only);
        for h in 0..hidden {
            let got = y[h] - y2[h];
            assert!((got - only[h]).abs() < 1e-3, "skip of neuron {victim} not exact at {h}");
        }
    }
}
