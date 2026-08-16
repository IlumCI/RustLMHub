// SPDX-License-Identifier: Apache-2.0

use rayon::prelude::*;

use crate::libm::sigmoidf;

pub const MXFP4_GROUP: usize = 32;
pub const MAX_TOPK: usize = 64;

#[derive(Clone, Copy)]
pub enum W<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
    I8(&'a [u8]),
    /// FP8 e4m3 with one E8M0 scale per `block` x `block` tile, read in place.
    /// Dequantising instead would turn DeepSeek-V4's 8.29 GB trunk into ~33 GB and
    /// take it out of a 16 GB machine, which is the whole point of not doing it.
    F8Block { w: &'a [u8], scale: &'a [u8], block: usize },
    /// GGUF k-quant super-blocks. One slice, not two: unlike MXFP4 and FP8-block, a
    /// k-quant carries its scales INSIDE each 256-element super-block, so there is no
    /// separate scale tensor to offset alongside.
    Q4K(&'a [u8]),
    Q5K(&'a [u8]),
    Q6K(&'a [u8]),
}

/// int8 (Q8-activation) path for k-quant matmuls: ~2.3-2.55x faster, near-bitwise (the
/// activation rounds to int8). Default OFF so the engine stays bit-exact against the f32
/// reference (tests, golden diffs, training). `serve` flips it ON by default via [`set_int8`]
/// because a served model wants speed; `RUSTLM_INT8=1`/`=0` overrides either way.
///
/// State: 0 = uninitialised (seed from env on first read), 1 = off, 2 = on.
static INT8_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn int8_enabled() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match INT8_STATE.load(Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var_os("RUSTLM_INT8").map(|v| v != "0").unwrap_or(false);
            INT8_STATE.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
    }
}

/// Force the int8 path on or off, overriding the env default. `serve` calls this to default
/// int8 ON. An explicit `RUSTLM_INT8=0` should still win, so callers pass the env-resolved
/// value rather than a bare `true`.
pub fn set_int8(on: bool) {
    INT8_STATE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

pub fn mmw(y: &mut [f32], x: &[f32], w: W, k_in: usize, out: usize) {
    match w {
        W::F32(m) => matmul(y, x, m, k_in, out),
        W::Bf16(m) => matmul_bf16(y, x, m, k_in, out),
        W::I8(m) => matmul_q8(y, x, m, k_in, out),
        W::F8Block { w, scale, block } => matmul_fp8_block(y, x, w, scale, k_in, out, block),
        W::Q4K(m) if int8_enabled() => {
            let act = crate::gguf::Q8Act::quantize(x, k_in);
            crate::gguf::matmul_q4k_q8(y, &act, m, k_in, out);
        }
        W::Q4K(m) => crate::gguf::matmul_q4k(y, x, m, k_in, out),
        W::Q6K(m) if int8_enabled() => {
            let act = crate::gguf::Q8Act::quantize(x, k_in);
            crate::gguf::matmul_q6k_q8(y, &act, m, k_in, out);
        }
        W::Q5K(m) => crate::gguf::matmul_q5k(y, x, m, k_in, out),
        W::Q6K(m) => crate::gguf::matmul_q6k(y, x, m, k_in, out),
    }
}

/// The same projection applied to `ntok` activation vectors, both TOKEN-MAJOR.
///
/// A prompt chunk drives every token through the same weights, so the weight can be
/// decoded once for the whole chunk instead of once per token. Measured 1.6-2.2x on Q4_K
/// at the shapes qwen35moe uses, bit-identical to the loop it replaces (`tests/kbench.rs`,
/// `batched_kquant`).
///
/// The k-quant types have a real batched kernel because that is where the decode cost is.
/// Everything else falls back to the obvious loop -- correct, and no slower than what the
/// caller would have written -- so a caller may use this unconditionally without having to
/// know which dtype it holds.
pub fn mmw_many(y: &mut [f32], x: &[f32], w: W, k_in: usize, out: usize, ntok: usize) {
    debug_assert_eq!(x.len(), ntok * k_in);
    debug_assert_eq!(y.len(), ntok * out);
    match w {
        W::Q4K(m) if int8_enabled() => {
            // Quantise each token once, reuse across the row set.
            for t in 0..ntok {
                let act = crate::gguf::Q8Act::quantize(&x[t * k_in..][..k_in], k_in);
                crate::gguf::matmul_q4k_q8(&mut y[t * out..][..out], &act, m, k_in, out);
            }
        }
        W::Q4K(m) => crate::gguf::matmul_q4k_many(y, x, m, k_in, out, ntok),
        W::Q6K(m) if int8_enabled() => {
            for t in 0..ntok {
                let act = crate::gguf::Q8Act::quantize(&x[t * k_in..][..k_in], k_in);
                crate::gguf::matmul_q6k_q8(&mut y[t * out..][..out], &act, m, k_in, out);
            }
        }
        W::Q5K(m) => crate::gguf::matmul_q5k_many(y, x, m, k_in, out, ntok),
        W::Q6K(m) => crate::gguf::matmul_q6k_many(y, x, m, k_in, out, ntok),
        _ => {
            for t in 0..ntok {
                mmw(&mut y[t * out..][..out], &x[t * k_in..][..k_in], w, k_in, out);
            }
        }
    }
}

/// `grad_x = Wᵀ · grad_y` -- the backward matvec, the adjoint of [`mmw`]. `k_in` is the
/// input width (length of `grad_x`), `out` the output width (length of `grad_y`, and the
/// number of rows of `W`). Overwrites `grad_x`.
///
/// The k-quant types have the transpose-free streaming kernel (see
/// [`crate::gguf::out_prod_q4k`]); the dense types fall back to the obvious column reduction,
/// which for an in-memory f32/bf16/i8 weight is already sequential enough. A caller may use
/// this unconditionally without knowing the dtype, exactly like `mmw`.
pub fn wt(grad_x: &mut [f32], grad_y: &[f32], w: W, k_in: usize, out: usize) {
    debug_assert!(grad_x.len() >= k_in);
    debug_assert!(grad_y.len() >= out);
    match w {
        W::Q4K(m) => crate::gguf::out_prod_q4k(grad_x, grad_y, m, k_in, out),
        W::Q5K(m) => crate::gguf::out_prod_q5k(grad_x, grad_y, m, k_in, out),
        W::Q6K(m) => crate::gguf::out_prod_q6k(grad_x, grad_y, m, k_in, out),
        W::F32(m) => {
            for gx in grad_x[..k_in].iter_mut() {
                *gx = 0.0;
            }
            for r in 0..out {
                let g = grad_y[r] as f64;
                let row = &m[r * k_in..][..k_in];
                for i in 0..k_in {
                    grad_x[i] = (grad_x[i] as f64 + g * row[i] as f64) as f32;
                }
            }
        }
        W::Bf16(m) => {
            for gx in grad_x[..k_in].iter_mut() {
                *gx = 0.0;
            }
            for r in 0..out {
                let g = grad_y[r] as f64;
                let row = &m[r * k_in..][..k_in];
                for i in 0..k_in {
                    grad_x[i] = (grad_x[i] as f64 + g * bf16f(row[i]) as f64) as f32;
                }
            }
        }
        _ => unimplemented!("wt: backward matvec not implemented for this weight dtype"),
    }
}

/// The batched backward matvec: `ntok` gradient vectors through the same weight, TOKEN-MAJOR
/// (`grad_y[t*out + r]`, `grad_x[t*k_in + i]`). The adjoint of [`mmw_many`], and the batched
/// form of [`wt`] -- it decodes each frozen row once for the whole batch, which is what keeps
/// a 248k-row vocabulary head from dominating a training step. Bit-identical to calling `wt`
/// per token. k-quant types have the real batched kernel; others fall back to the per-token
/// loop, correct and no slower than the caller would write.
pub fn wt_many(grad_x: &mut [f32], grad_y: &[f32], w: W, k_in: usize, out: usize, ntok: usize) {
    debug_assert!(grad_x.len() >= ntok * k_in);
    debug_assert!(grad_y.len() >= ntok * out);
    match w {
        W::Q4K(m) => crate::gguf::out_prod_q4k_many(grad_x, grad_y, m, k_in, out, ntok),
        W::Q5K(m) => crate::gguf::out_prod_q5k_many(grad_x, grad_y, m, k_in, out, ntok),
        W::Q6K(m) => crate::gguf::out_prod_q6k_many(grad_x, grad_y, m, k_in, out, ntok),
        _ => {
            for t in 0..ntok {
                wt(&mut grad_x[t * k_in..][..k_in], &grad_y[t * out..][..out], w, k_in, out);
            }
        }
    }
}

#[inline]
pub fn bf16f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Accumulation width for RMSNorm. This is a per-ARCHITECTURE property, not a tuning
/// knob: K3's reference upcasts to f64 and DeepSeek's does `x.float()` then `.mean()` in
/// f32, and the two give different last bits. Hardcoding either one silently binds the
/// engine to one model, which is why it lives in the architecture descriptor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Acc {
    F64,
    F32,
}

/// K3's form, and the one the 14 op fixtures gate.
pub fn rmsnorm(y: &mut [f32], x: &[f32], w: &[f32], n: usize, eps: f32) {
    rmsnorm_acc(y, x, w, n, eps, Acc::F64)
}

pub fn rmsnorm_acc(y: &mut [f32], x: &[f32], w: &[f32], n: usize, eps: f32, acc: Acc) {
    // f64: 7168 squared terms in f32 loses real precision, and every downstream
    // comparison against the K3 reference depends on it.
    let inv = match acc {
        Acc::F64 => {
            let mut ss = 0.0f64;
            for i in 0..n {
                ss += x[i] as f64 * x[i] as f64;
            }
            (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32
        }
        Acc::F32 => {
            let mut ss = 0.0f32;
            for i in 0..n {
                ss += x[i] * x[i];
            }
            1.0 / (ss / n as f32 + eps).sqrt()
        }
    };
    for i in 0..n {
        y[i] = w[i] * x[i] * inv;
    }
}

/// Backward of [`rmsnorm`] (the f64-accumulated K3 form): given `grad_y = dL/dy` where
/// `y = rmsnorm(x)`, write `grad_x = dL/dx`.
///
/// With `inv = 1/sqrt(mean(x²)+eps)` and `y_i = w_i·x_i·inv`, differentiating through the
/// shared `inv` gives
/// ```text
///   grad_x_j = inv·w_j·grad_y_j − (inv³/n)·x_j·Σ_i(grad_y_i·w_i·x_i)
/// ```
/// The reduction `S = Σ grad_y·w·x` and `inv` are taken in f64 to match the forward's
/// precision, so this is the exact adjoint of the forward norm and is gradient-checked
/// (`train::tests::rmsnorm_backward_gradient_check`).
pub fn rmsnorm_backward(grad_x: &mut [f32], grad_y: &[f32], x: &[f32], w: &[f32], n: usize, eps: f32) {
    debug_assert!(grad_x.len() >= n && grad_y.len() >= n && x.len() >= n && w.len() >= n);
    let mut ss = 0.0f64;
    for i in 0..n {
        ss += x[i] as f64 * x[i] as f64;
    }
    let inv = 1.0 / (ss / n as f64 + eps as f64).sqrt();
    let mut s = 0.0f64;
    for i in 0..n {
        s += grad_y[i] as f64 * w[i] as f64 * x[i] as f64;
    }
    let c = inv * inv * inv / n as f64;
    for j in 0..n {
        grad_x[j] = (inv * w[j] as f64 * grad_y[j] as f64 - c * x[j] as f64 * s) as f32;
    }
}

/// Gemma's form: the stored gain is an OFFSET, so the multiplier is `1 + w`. MiniMax-M3
/// sets `use_gemma_norm`. Using the plain form on a gemma-norm checkpoint scales every
/// channel by roughly its own weight instead of one plus it -- bounded, plausible, wrong.
pub fn rmsnorm_gemma(y: &mut [f32], x: &[f32], w: &[f32], n: usize, eps: f32, acc: Acc) {
    let mut t = vec![0f32; n];
    let ones = vec![1f32; n];
    rmsnorm_acc(&mut t, x, &ones, n, eps, acc);
    for i in 0..n {
        y[i] = t[i] * (1.0 + w[i]);
    }
}

pub fn situ_glu(y: &mut [f32], x: &[f32], n: usize, b1: f32, b2: f32) {
    let (gate, up) = (&x[..n], &x[n..2 * n]);
    for i in 0..n {
        let g = gate[i];
        // The sigmoid takes the UNCAPPED gate. Feeding it the capped value instead
        // still yields a bounded, plausible function and is WRONG.
        let a = b1 * (g / b1).tanh() * sigmoidf(g);
        let u = b2 * (up[i] / b2).tanh();
        y[i] = a * u;
    }
}

pub fn shortconv(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    mut state: Option<&mut [f32]>,
    channels: usize,
    k: usize,
    t_len: usize,
) {
    let hist = k - 1;
    let mut buf = vec![0f32; hist];
    for c in 0..channels {
        if hist > 0 {
            match state.as_deref() {
                Some(s) => buf.copy_from_slice(&s[c * hist..][..hist]),
                None => buf.fill(0.0),
            }
        }
        for t in 0..t_len {
            let cur = x[t * channels + c];
            // Taps are ordered oldest..newest: w[k-1] multiplies the CURRENT input.
            let mut acc = w[c * k + hist] * cur;
            for j in 0..hist {
                acc += w[c * k + j] * buf[j];
            }
            for j in 0..hist.saturating_sub(1) {
                buf[j] = buf[j + 1];
            }
            if hist > 0 {
                buf[hist - 1] = cur;
            }
            y[t * channels + c] = acc * sigmoidf(acc); // SiLU, fused
        }
        if hist > 0 {
            if let Some(s) = state.as_deref_mut() {
                s[c * hist..][..hist].copy_from_slice(&buf);
            }
        }
    }
}

pub fn kda_decay(
    g: &mut [f32],
    alpha: &mut [f32],
    z: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    h_n: usize,
    d_n: usize,
    lb: f32,
) {
    for h in 0..h_n {
        // PER HEAD. The checkpoint stores head_dim floats but only the first H are
        // nonzero. Indexing this per channel is a silent, fatal error.
        let a = a_log[h].exp();
        for d in 0..d_n {
            let i = h * d_n + d;
            let u = a * (z[i] + dt_bias[i]);
            let gi = lb * sigmoidf(u);
            g[i] = gi;
            alpha[i] = gi.exp();
        }
    }
}

pub fn kda_step(
    s: &mut [f32],
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: f32,
    dk: usize,
    dv: usize,
) {
    // 1. channel-wise decay: scale ROW i of S by alpha[i]. The gate is per key
    //    channel, not a scalar, which is what "channel-wise forget gate" means.
    for i in 0..dk {
        let a = alpha[i];
        for j in 0..dv {
            s[i * dv + j] *= a;
        }
    }

    // 2. read the state along k: u = S^T k
    let mut u = vec![0f32; dv];
    for i in 0..dk {
        let ki = k[i];
        if ki == 0.0 {
            continue;
        }
        for j in 0..dv {
            u[j] += ki * s[i * dv + j];
        }
    }

    // 3. rank-one delta write. (v - u) is the prediction error: this is what makes
    //    it a DELTA rule rather than plain accumulation.
    for i in 0..dk {
        let ki = k[i];
        if ki == 0.0 {
            continue;
        }
        for j in 0..dv {
            s[i * dv + j] += ki * beta * (v[j] - u[j]);
        }
    }

    // 4. output from the ALREADY UPDATED state: o = S^T q
    o[..dv].fill(0.0);
    for i in 0..dk {
        let qi = q[i];
        if qi == 0.0 {
            continue;
        }
        for j in 0..dv {
            o[j] += qi * s[i * dv + j];
        }
    }
}

// Sixteen f64 accumulators partitioned by i%16, reduced as a fixed tree. The split is
// written out rather than left to the compiler because it fixes a summation ORDER:
// matmul_bf16 and both AVX2 paths reproduce this exact partition and this exact tree,
// which is what makes the implementations agree bit for bit.
#[inline]
fn reduce16(a: &[f64; 16]) -> f64 {
    let b0 = (a[0] + a[4]) + (a[8] + a[12]);
    let b1 = (a[1] + a[5]) + (a[9] + a[13]);
    let b2 = (a[2] + a[6]) + (a[10] + a[14]);
    let b3 = (a[3] + a[7]) + (a[11] + a[15]);
    (b0 + b1) + (b2 + b3)
}

/// Below this many output rows a matmul runs serially, reproducing the OpenMP build's
/// `if (out > 64)` clause (`k3_ops.c:246`). Under it the threading costs more than the
/// arithmetic, and the engine calls these kernels on small projections constantly.
const PAR_MIN_ROWS: usize = 64;

/// Rows per parallel chunk, aiming at a few chunks per thread so rayon can steal.
///
/// OpenMP's `schedule(static)` hands each thread one contiguous block; this hands out
/// smaller blocks. The difference is scheduling only. Every row reads its own slice of
/// the weight matrix and writes its own element of `y`, so no partition of the row space
/// can change the arithmetic — which is what makes replacing the schedule safe.
/// The row count above which a matmul fans out, and the chunking it uses. Exposed so
/// `gguf`'s k-quant kernels partition rows exactly as every other kernel here does --
/// the thread-independence tests depend on there being ONE such policy, not two.
pub fn par_min_rows() -> usize {
    PAR_MIN_ROWS
}

pub fn row_chunk_pub(rows: usize) -> usize {
    row_chunk(rows)
}

fn row_chunk(rows: usize) -> usize {
    rows.div_ceil(rayon::current_num_threads().max(1) * 4).max(1)
}

pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], k_in: usize, out: usize) {
    if out > PAR_MIN_ROWS {
        let chunk = row_chunk(out);
        y[..out].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            matmul_serial(yc, x, &w[base * k_in..], k_in, yc.len());
        });
        return;
    }
    matmul_serial(y, x, w, k_in, out);
}

fn matmul_serial(y: &mut [f32], x: &[f32], w: &[f32], k_in: usize, out: usize) {
    for o in 0..out {
        let row = &w[o * k_in..][..k_in];
        let mut a = [0f64; 16];
        let mut i = 0;
        while i + 15 < k_in {
            for l in 0..16 {
                a[l] = (row[i + l] as f64).mul_add(x[i + l] as f64, a[l]);
            }
            i += 16;
        }
        let mut acc = reduce16(&a);
        while i < k_in {
            acc = (row[i] as f64).mul_add(x[i] as f64, acc);
            i += 1;
        }
        y[o] = acc as f32;
    }
}

fn matmul_bf16_scalar(y: &mut [f32], x: &[f32], w: &[u16], k_in: usize, out: usize) {
    for o in 0..out {
        let row = &w[o * k_in..][..k_in];
        let mut a = [0f64; 16];
        let mut i = 0;
        while i + 15 < k_in {
            for l in 0..16 {
                a[l] = (bf16f(row[i + l]) as f64).mul_add(x[i + l] as f64, a[l]);
            }
            i += 16;
        }
        let mut acc = reduce16(&a);
        while i < k_in {
            acc = (bf16f(row[i]) as f64).mul_add(x[i] as f64, acc);
            i += 1;
        }
        y[o] = acc as f32;
    }
}

// Lane l of v_k holds elements with index = 4k+l (mod 16), the same partition as the
// scalar accumulators. (v0+v1)+(v2+v3) lanewise therefore yields exactly the scalar
// b0..b3, and the cross-lane (a0+a1)+(a2+a3) closes the identical tree.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_bf16_avx2(y: &mut [f32], x: &[f32], w: &[u16], k_in: usize, out: usize) {
    use std::arch::x86_64::*;
    for o in 0..out {
        let row = w.as_ptr().add(o * k_in);
        let mut v0 = _mm256_setzero_pd();
        let mut v1 = _mm256_setzero_pd();
        let mut v2 = _mm256_setzero_pd();
        let mut v3 = _mm256_setzero_pd();
        let mut i = 0;
        // bf16 -> f32 is a 16-bit left shift: widen u16 to u32, shift, reinterpret.
        let widen = |p: *const u16| -> __m256d {
            let h = _mm_loadl_epi64(p as *const __m128i);
            _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h), 16)))
        };
        while i + 15 < k_in {
            v0 = _mm256_fmadd_pd(widen(row.add(i)), _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(i))), v0);
            v1 = _mm256_fmadd_pd(widen(row.add(i + 4)), _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(i + 4))), v1);
            v2 = _mm256_fmadd_pd(widen(row.add(i + 8)), _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(i + 8))), v2);
            v3 = _mm256_fmadd_pd(widen(row.add(i + 12)), _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(i + 12))), v3);
            i += 16;
        }
        let vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
        let mut a = [0f64; 4];
        _mm256_storeu_pd(a.as_mut_ptr(), vt);
        let mut acc = (a[0] + a[1]) + (a[2] + a[3]);
        while i < k_in {
            acc = (bf16f(*row.add(i)) as f64).mul_add(x[i] as f64, acc);
            i += 1;
        }
        y[o] = acc as f32;
    }
}

pub fn matmul_bf16(y: &mut [f32], x: &[f32], w: &[u16], k_in: usize, out: usize) {
    if out > PAR_MIN_ROWS {
        let chunk = row_chunk(out);
        y[..out].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            matmul_bf16_serial(yc, x, &w[base * k_in..], k_in, yc.len());
        });
        return;
    }
    matmul_bf16_serial(y, x, w, k_in, out);
}

/// The SIMD dispatch, and the unit of work a thread runs. The feature detection sits
/// here rather than outside the parallel region so the AVX2 body stays inside a single
/// `#[target_feature]` function -- a rayon closure is a distinct function and does not
/// inherit the caller's enabled features.
fn matmul_bf16_serial(y: &mut [f32], x: &[f32], w: &[u16], k_in: usize, out: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul_bf16_avx2(y, x, w, k_in, out) };
            return;
        }
    }
    matmul_bf16_scalar(y, x, w, k_in, out);
}

// Draft-only (K3_WI8): no cross-path determinism contract, so this accumulates in f32.
pub fn matmul_q8(y: &mut [f32], x: &[f32], w: &[u8], k_in: usize, out: usize) {
    if out > PAR_MIN_ROWS {
        let chunk = row_chunk(out);
        let rowb = 4 + k_in;
        y[..out].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            matmul_q8_serial(yc, x, &w[base * rowb..], k_in, yc.len());
        });
        return;
    }
    matmul_q8_serial(y, x, w, k_in, out);
}

fn matmul_q8_serial(y: &mut [f32], x: &[f32], w: &[u8], k_in: usize, out: usize) {
    let rowb = 4 + k_in;
    for o in 0..out {
        let row = &w[o * rowb..][..rowb];
        let scale = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
        let q = &row[4..];
        let (mut a0, mut a1, mut a2, mut a3) = (0f32, 0f32, 0f32, 0f32);
        let mut i = 0;
        while i + 3 < k_in {
            a0 += q[i] as i8 as f32 * x[i];
            a1 += q[i + 1] as i8 as f32 * x[i + 1];
            a2 += q[i + 2] as i8 as f32 * x[i + 2];
            a3 += q[i + 3] as i8 as f32 * x[i + 3];
            i += 4;
        }
        let mut acc = (a0 + a1) + (a2 + a3);
        while i < k_in {
            acc += q[i] as i8 as f32 * x[i];
            i += 1;
        }
        y[o] = acc * scale;
    }
}

// OCP MX E2M1: index by the 4-bit code; bit 3 is the sign.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The nibble-pair table, exposed so the QDQ sweep can decode packed bytes without
/// duplicating the low/high-nibble convention.
pub fn e2m1_pair_table() -> &'static [[f32; 2]; 256] {
    e2m1_pair()
}

fn e2m1_pair() -> &'static [[f32; 2]; 256] {
    use std::sync::OnceLock;
    static T: OnceLock<[[f32; 2]; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0f32; 2]; 256];
        for (b, slot) in t.iter_mut().enumerate() {
            // low nibble = EVEN element, high nibble = ODD element
            *slot = [E2M1[b & 0x0F], E2M1[b >> 4]];
        }
        t
    })
}

fn e8m0() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(crate::libm::e8m0_table)
}

// Deliberately NOT bit-identical to dequantise-then-matmul: this sums each group of 32
// and applies that group's scale before accumulating. Every product is exact in f64
// (E2M1 carries 3 mantissa bits, x carries 24, so 27 of 53 are needed), so only the
// additions round and the difference is ~1 ULP of f64 against a 1e-6 requirement.
pub fn matmul_mxfp4(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    k_in: usize,
    rows: usize,
    group: usize,
) {
    if rows > PAR_MIN_ROWS {
        let chunk = row_chunk(rows);
        // Two strides, not one: an expert's nibbles and its per-group E8M0 scales live in
        // separate tensors, so a chunk has to be offset into both.
        let (pcols, ngrp) = (k_in / 2, k_in.div_ceil(group));
        y[..rows].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            matmul_mxfp4_serial(
                yc,
                x,
                &packed[base * pcols..],
                &scales[base * ngrp..],
                k_in,
                yc.len(),
                group,
            );
        });
        return;
    }
    matmul_mxfp4_serial(y, x, packed, scales, k_in, rows, group);
}

// The MoE expert kernel, and the largest single consumer of per-token compute: a routed
// expert is three of these. The scalar path keeps eight f64 accumulators partitioned by
// i%8 and folds them as (s0+s4)+(s1+s5) | (s2+s6)+(s3+s7). Lane j of u0+u1 is exactly
// s[j]+s[j+4], so this reproduces that tree and is bit-identical, not merely close.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn e2m1_decode8(mags: std::arch::x86_64::__m256, code: std::arch::x86_64::__m256i)
    -> std::arch::x86_64::__m256
{
    use std::arch::x86_64::*;
    // bits 0-2 index the magnitude (permutevar8x32_ps masks to 3 bits itself);
    // bit 3 becomes the f32 sign. Code 8 therefore yields -0.0, which is E2M1[8].
    let mag = _mm256_permutevar8x32_ps(mags, code);
    let sgn = _mm256_slli_epi32(_mm256_and_si256(code, _mm256_set1_epi32(8)), 28);
    _mm256_or_ps(mag, _mm256_castsi256_ps(sgn))
}

// The nibble decode happens IN REGISTERS. The previous version expanded each byte through
// a 256-entry table into a `wf[64]` staging buffer and then reloaded it -- an 8-byte load
// and 8-byte store per byte, which cost more than the dot product it fed.
//
// E2M1 is clean sign-magnitude, so `permutevar8x32_ps` is an exact 8-entry in-register
// LUT. The decoded f32 values are bit-identical to `E2M1[code]`, and `cvtps_pd` is exact,
// so this stays bit-identical to matmul_mxfp4_scalar -- Gate D is unweakened.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_mxfp4_avx2(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    k_in: usize,
    rows: usize,
    group: usize,
) {
    use std::arch::x86_64::*;
    let pcols = k_in / 2;
    let ngrp = k_in.div_ceil(group);
    let gbyte = group / 2;
    let pair = e2m1_pair();
    let e8 = e8m0();
    let mags = _mm256_setr_ps(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    let mask_f = _mm256_set1_epi32(0x0F);

    for r in 0..rows {
        let pr = &packed[r * pcols..][..pcols];
        let sr = &scales[r * ngrp..][..ngrp];
        let mut acc = 0.0f64;
        for g in 0..ngrp {
            let sb = sr[g];
            if sb == 255 {
                continue;
            }
            let pb = &pr[g * gbyte..];
            let xg = &x[g * group..];
            let n = (k_in - g * group).min(group);

            let (mut u0, mut u1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let xp = xg.as_ptr();
            let mut i = 0;
            // 8 packed bytes -> 16 values per iteration. The i%8 partition into u0/u1 is
            // preserved exactly -- 0-3 to u0, 4-7 to u1, 8-11 to u0, 12-15 to u1 -- which
            // is the order matmul_mxfp4_scalar's eight accumulators fold in.
            while i + 15 < n {
                let raw =
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64(pb.as_ptr().add(i >> 1) as *const __m128i));
                let ev = e2m1_decode8(mags, _mm256_and_si256(raw, mask_f));
                let od = e2m1_decode8(mags, _mm256_and_si256(_mm256_srli_epi32(raw, 4), mask_f));
                // Interleave to natural order: low nibble is the EVEN element.
                let a = _mm256_unpacklo_ps(ev, od);
                let b = _mm256_unpackhi_ps(ev, od);
                for (q, base) in [
                    (_mm256_permute2f128_ps(a, b, 0x20), i),
                    (_mm256_permute2f128_ps(a, b, 0x31), i + 8),
                ] {
                    u0 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_castps256_ps128(q)),
                        _mm256_cvtps_pd(_mm_loadu_ps(xp.add(base))),
                        u0,
                    );
                    u1 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm256_extractf128_ps(q, 1)),
                        _mm256_cvtps_pd(_mm_loadu_ps(xp.add(base + 4))),
                        u1,
                    );
                }
                i += 16;
            }
            let mut a = [0f64; 4];
            _mm256_storeu_pd(a.as_mut_ptr(), _mm256_add_pd(u0, u1));
            let mut sub = (a[0] + a[1]) + (a[2] + a[3]);
            // Tail for a partial final group. Scalar, via the same table the scalar path
            // uses, so the values match there too.
            while i < n {
                let pv = pair[*pb.get_unchecked(i >> 1) as usize];
                let w = if i & 1 == 0 { pv[0] } else { pv[1] };
                sub = (w as f64).mul_add(*xg.get_unchecked(i) as f64, sub);
                i += 1;
            }
            acc += sub * e8[sb as usize] as f64;
        }
        *y.get_unchecked_mut(r) = acc as f32;
    }
}

fn matmul_mxfp4_serial(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    k_in: usize,
    rows: usize,
    group: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul_mxfp4_avx2(y, x, packed, scales, k_in, rows, group) };
            return;
        }
    }
    matmul_mxfp4_scalar(y, x, packed, scales, k_in, rows, group);
}

fn matmul_mxfp4_scalar(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    k_in: usize,
    rows: usize,
    group: usize,
) {
    let pcols = k_in / 2;
    let ngrp = k_in.div_ceil(group);
    let gbyte = group / 2;
    let pair = e2m1_pair();
    let e8 = e8m0();

    for r in 0..rows {
        let pr = &packed[r * pcols..][..pcols];
        let sr = &scales[r * ngrp..][..ngrp];
        let mut acc = 0.0f64;

        for g in 0..ngrp {
            let sb = sr[g];
            if sb == 255 {
                continue; // NaN scale by OCP MX spec: contribute nothing
            }
            let pb = &pr[g * gbyte..];
            let xg = &x[g * group..];
            let n = (k_in - g * group).min(group);

            let mut wf = [0f32; 64];
            let half = n >> 1;
            for j in 0..half {
                let pv = pair[pb[j] as usize];
                wf[2 * j] = pv[0];
                wf[2 * j + 1] = pv[1];
            }
            if n & 1 != 0 {
                wf[n - 1] = pair[pb[half] as usize][0];
            }

            // Eight f64 lanes partitioned by i%8, reduced as (s0+s4)+(s1+s5) pairs; the
            // AVX2 path uses the identical partition so both agree bit for bit.
            let mut s = [0f64; 8];
            let mut i = 0;
            while i + 7 < n {
                for l in 0..8 {
                    s[l] = (wf[i + l] as f64).mul_add(xg[i + l] as f64, s[l]);
                }
                i += 8;
            }
            let (b0, b1) = (s[0] + s[4], s[1] + s[5]);
            let (b2, b3) = (s[2] + s[6], s[3] + s[7]);
            let mut sub = (b0 + b1) + (b2 + b3);
            while i < n {
                sub = (wf[i] as f64).mul_add(xg[i] as f64, sub);
                i += 1;
            }
            acc += sub * e8[sb as usize] as f64;
        }
        y[r] = acc as f32;
    }
}

pub fn mxfp4_dequant(
    out: &mut [f32],
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    pcols: usize,
    group: usize,
) {
    let width = pcols * 2;
    let ngrp = width.div_ceil(group);
    let e8 = e8m0();
    for r in 0..rows {
        let pr = &packed[r * pcols..][..pcols];
        let sr = &scales[r * ngrp..][..ngrp];
        let orow = &mut out[r * width..][..width];
        for g in 0..ngrp {
            let mult = e8[sr[g] as usize];
            let lo = g * group;
            let hi = (lo + group).min(width);
            for i in lo..hi {
                let byte = pr[i >> 1];
                // low nibble = EVEN element. Reversing this gives right values in wrong
                // places, which every statistical check would pass.
                let nib = if i & 1 != 0 { byte >> 4 } else { byte & 0x0F };
                orow[i] = E2M1[nib as usize] * mult;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn router(
    idx: &mut [i32],
    wt: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    hidden: usize,
    n_experts: usize,
    topk: usize,
    renorm: bool,
    routed_scale: f32,
) {
    let mut score = vec![0f32; n_experts];
    let mut choice = vec![0f32; n_experts];
    // k3_ops.c:417-421: each iteration writes only its own score[e] and choice[e], and
    // the accumulation order INSIDE an expert is untouched -- thread t still sums
    // i = 0..hidden-1 in sequence into its own double. Splitting the outer loop therefore
    // cannot change a single bit. Only the scoring parallelises; the top-k below is a
    // sequential selection whose tie-breaking is part of the contract.
    score
        .par_iter_mut()
        .zip(choice.par_iter_mut())
        .enumerate()
        .for_each(|(e, (sc, ch))| {
            let row = &w[e * hidden..][..hidden];
            let mut acc = 0.0f64;
            for i in 0..hidden {
                acc += row[i] as f64 * x[i] as f64;
            }
            *sc = 1.0 / (1.0 + (-(acc as f32)).exp());
            *ch = *sc + bias.map_or(0.0, |b| b[e]);
        });

    // top-k by repeated max. Marking taken entries -inf keeps ties deterministic in
    // first-index order, matching a stable selection.
    for j in 0..topk {
        let mut best = -1i32;
        let mut bv = f32::NEG_INFINITY;
        for e in 0..n_experts {
            if choice[e] > bv {
                bv = choice[e];
                best = e as i32;
            }
        }
        if best < 0 {
            idx[j] = 0;
            wt[j] = 0.0;
            continue;
        }
        idx[j] = best;
        wt[j] = score[best as usize]; // UNBIASED score, not choice[best]
        choice[best as usize] = f32::NEG_INFINITY;
    }

    if renorm && topk > 1 {
        let s: f64 = wt[..topk].iter().map(|&v| v as f64).sum();
        let inv = (1.0 / (s + 1e-20)) as f32;
        for j in 0..topk {
            wt[j] *= inv;
        }
    }
    for j in 0..topk {
        wt[j] *= routed_scale;
    }
}

pub fn attn_res(out: &mut [f32], src: &[f32], fold: &[f32], nsrc: usize, n: usize, eps: f32) {
    let mut score = vec![0f32; nsrc];
    for s in 0..nsrc {
        let v = &src[s * n..][..n];
        let mut ss = 0.0f64;
        for i in 0..n {
            ss += v[i] as f64 * v[i] as f64;
        }
        let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
        // key is the NORMALISED source; fold already carries norm.weight*proj.weight
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += (v[i] * inv) as f64 * fold[i] as f64;
        }
        score[s] = acc as f32;
    }

    let mut m = score[0];
    for s in 1..nsrc {
        if score[s] > m {
            m = score[s];
        }
    }
    let mut z = 0.0f64;
    for s in 0..nsrc {
        score[s] = (score[s] - m).exp();
        z += score[s] as f64;
    }

    out[..n].fill(0.0);
    for s in 0..nsrc {
        let p = (score[s] as f64 / z) as f32;
        let v = &src[s * n..][..n]; // the RAW source, not the key
        for i in 0..n {
            out[i] += p * v[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // bf16 with a finite exponent, i.e. what a checkpoint actually contains. Arbitrary
    // u16 patterns include the NaN exponent, and the two paths then differ only in NaN
    // payload -- which neither this port nor the C build promises to preserve.
    fn bf16_weight(i: usize) -> u16 {
        let v = ((i as f32) * 0.7391).sin() * 2.0;
        (v.to_bits() >> 16) as u16
    }

    #[test]
    fn matmul_bf16_paths_agree_bitwise() {
        // The C build compiles exactly one of these; here both exist, so the identity
        // the C only asserts in a comment is actually executed. Sizes straddle the
        // 16-element block so the scalar tail is exercised at every remainder.
        for k_in in [1usize, 7, 15, 16, 17, 31, 32, 64, 77, 128, 129] {
            let out = 5;
            let w: Vec<u16> = (0..k_in * out).map(bf16_weight).collect();
            let x: Vec<f32> = (0..k_in).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut a = vec![0f32; out];
            let mut b = vec![0f32; out];
            matmul_bf16_scalar(&mut a, &x, &w, k_in, out);
            matmul_bf16(&mut b, &x, &w, k_in, out);
            for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                assert_eq!(p.to_bits(), q.to_bits(), "k_in={k_in} row {i}: {p} vs {q}");
            }
        }
    }

    #[test]
    fn matmul_and_matmul_bf16_agree_on_bf16_representable_input() {
        // bf16 widens exactly, so multiplying from bf16 storage must compute with the
        // same values an f32 copy would have supplied.
        let (k_in, out) = (64, 3);
        let wb: Vec<u16> = (0..k_in * out).map(|i| 0x3F00 + (i as u16 % 97)).collect();
        let wf: Vec<f32> = wb.iter().map(|&h| bf16f(h)).collect();
        let x: Vec<f32> = (0..k_in).map(|i| (i as f32 * 0.11).cos()).collect();
        let mut a = vec![0f32; out];
        let mut b = vec![0f32; out];
        matmul(&mut a, &x, &wf, k_in, out);
        matmul_bf16(&mut b, &x, &wb, k_in, out);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn mxfp4_low_nibble_is_the_even_element() {
        // byte 0x21 -> low nibble 1 (0.5) at index 0, high nibble 2 (1.0) at index 1.
        let mut out = [0f32; 2];
        mxfp4_dequant(&mut out, &[0x21], &[127], 1, 1, 32);
        assert_eq!(out, [0.5, 1.0], "reversing the nibbles passes every statistical check");
    }

    #[test]
    fn mxfp4_nan_scale_zeroes_its_group() {
        let mut out = [0f32; 2];
        mxfp4_dequant(&mut out, &[0x21], &[255], 1, 1, 32);
        assert_eq!(out, [0.0, 0.0]);
        let mut y = [0f32; 1];
        matmul_mxfp4(&mut y, &[1.0, 1.0], &[0x21], &[255], 2, 1, 32);
        assert_eq!(y[0], 0.0);
    }

    #[test]
    fn router_ties_break_in_first_index_order() {
        let (hidden, n_experts, topk) = (2, 4, 2);
        let x = [0.0f32, 0.0];
        let w = vec![0f32; n_experts * hidden]; // every logit identical
        let mut idx = [0i32; 2];
        let mut wt = [0f32; 2];
        router(&mut idx, &mut wt, &x, &w, None, hidden, n_experts, topk, false, 1.0);
        assert_eq!(idx, [0, 1]);
    }

    #[test]
    fn router_weight_is_the_unbiased_score() {
        let (hidden, n_experts, topk) = (1, 2, 1);
        let x = [0.0f32];
        let w = vec![0f32; 2];
        // Bias picks expert 1, but the returned weight must be its sigmoid(0) = 0.5.
        let bias = [0.0f32, 10.0];
        let mut idx = [0i32; 1];
        let mut wt = [0f32; 1];
        router(&mut idx, &mut wt, &x, &w, Some(&bias), hidden, n_experts, topk, false, 1.0);
        assert_eq!(idx[0], 1);
        assert_eq!(wt[0], 0.5);
    }

    #[test]
    fn situ_glu_sigmoid_sees_the_uncapped_gate() {
        // At g = 50 with b1 = 4, tanh(g/b1) saturates but sigmoid(50) != sigmoid(4).
        let (b1, b2) = (4.0f32, 25.0f32);
        let x = [50.0f32, 1.0];
        let mut y = [0f32; 1];
        situ_glu(&mut y, &x, 1, b1, b2);
        let capped = b1 * (50.0f32 / b1).tanh() * sigmoidf(b1 * (50.0f32 / b1).tanh());
        let want = b1 * (50.0f32 / b1).tanh() * sigmoidf(50.0);
        let u = b2 * (1.0f32 / b2).tanh();
        assert_eq!(y[0], want * u);
        assert_ne!(y[0], capped * u, "the capped form is plausible and wrong");
    }
}

pub struct MlaW<'a> {
    pub q_a: W<'a>,
    pub q_b: W<'a>,
    pub kv_a: W<'a>,
    pub kv_b: W<'a>,
    pub o: W<'a>,
    pub g: Option<W<'a>>,
    pub q_a_norm: &'a [f32],
    pub kv_a_norm: &'a [f32],
}

pub struct MlaDims {
    pub hidden: usize,
    pub n_heads: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub rms_eps: f32,
}

// kvc == None recomputes all keys and values from x and caches nothing. Both paths must
// produce identical output.
#[allow(clippy::too_many_arguments)]
pub fn mla_cached(
    out: &mut [f32],
    x: &[f32],
    w: &MlaW,
    d: &MlaDims,
    t_len: usize,
    kv: Option<(&mut [f32], &mut [f32])>,
    cached: usize,
) {
    let e = d.hidden;
    let h_n = d.n_heads;
    let (qn, qr, vh) = (d.qk_nope, d.qk_rope, d.v_head);
    let qh = qn + qr;
    let kvw = d.kv_lora + qr;
    let kvd = qn + vh;
    let scale = 1.0 / (qh as f32).sqrt(); // over qh, not qn
    let have_cache = kv.is_some();
    let cached = if have_cache { cached } else { 0 };
    let last = cached + t_len - 1;

    let mut q = vec![0f32; t_len * h_n * qh];
    let mut ct = vec![0f32; kvw];
    let mut ql = vec![0f32; d.q_lora];
    let mut acc = vec![0f32; h_n * vh];
    let mut gbuf = vec![0f32; h_n * vh];
    let mut sc = vec![0f32; last + 1];

    let mut own_kv = vec![0f32; if have_cache { 0 } else { t_len * h_n * kvd }];
    let mut own_rope = vec![0f32; if have_cache { 0 } else { t_len * qr }];
    let (kvc, ropec) = match kv {
        Some((a, b)) => (a, b),
        None => (own_kv.as_mut_slice(), own_rope.as_mut_slice()),
    };

    for t in 0..t_len {
        let p = cached + t;
        let xt = &x[t * e..][..e];
        mmw(&mut ql, xt, w.q_a, e, d.q_lora);
        let qln = ql.clone();
        rmsnorm(&mut ql, &qln, w.q_a_norm, d.q_lora, d.rms_eps);
        mmw(&mut q[t * h_n * qh..][..h_n * qh], &ql, w.q_b, d.q_lora, h_n * qh);

        // ONE projection emits the compressed latent AND the shared rope slot.
        mmw(&mut ct, xt, w.kv_a, e, kvw);
        let ctn = ct.clone();
        // the norm covers the latent only, never the rope slot
        rmsnorm(&mut ct, &ctn, w.kv_a_norm, d.kv_lora, d.rms_eps);
        ropec[p * qr..][..qr].copy_from_slice(&ct[d.kv_lora..][..qr]);
        let ctl = ct[..d.kv_lora].to_vec();
        mmw(&mut kvc[p * h_n * kvd..][..h_n * kvd], &ctl, w.kv_b, d.kv_lora, h_n * kvd);
    }

    for t in 0..t_len {
        let p = cached + t;
        for h in 0..h_n {
            let qt = &q[(t * h_n + h) * qh..][..qh];
            let mut m = f32::NEG_INFINITY;
            for s in 0..=p {
                let ks = &kvc[s * h_n * kvd + h * kvd..][..kvd];
                let kr = &ropec[s * qr..][..qr];
                let mut dd = 0.0f64;
                for i in 0..qn {
                    dd += qt[i] as f64 * ks[i] as f64;
                }
                // The rope slot is UNROTATED but still scored, and the SAME values serve
                // every head. Dropping this term is the silent bug.
                for i in 0..qr {
                    dd += qt[qn + i] as f64 * kr[i] as f64;
                }
                sc[s] = dd as f32 * scale;
                if sc[s] > m {
                    m = sc[s];
                }
            }
            let mut z = 0.0f64;
            for s in 0..=p {
                sc[s] = (sc[s] - m).exp();
                z += sc[s] as f64;
            }
            let o = &mut acc[h * vh..][..vh];
            o.fill(0.0);
            for s in 0..=p {
                let pr = (sc[s] as f64 / z) as f32;
                let vs = &kvc[s * h_n * kvd + h * kvd + qn..][..vh];
                for j in 0..vh {
                    o[j] += pr * vs[j];
                }
            }
        }

        // Gate BEFORE o_proj, and no norm on it, unlike KDA which norms first.
        if let Some(gw) = w.g {
            mmw(&mut gbuf, &x[t * e..][..e], gw, e, h_n * vh);
            for i in 0..h_n * vh {
                acc[i] *= 1.0 / (1.0 + (-gbuf[i]).exp());
            }
        }
        mmw(&mut out[t * e..][..e], &acc, w.o, h_n * vh, e);
    }
}

pub struct MoeW<'a> {
    pub gate: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub w1: &'a [f32],
    pub w3: &'a [f32],
    pub w2: &'a [f32],
    pub latent_norm: &'a [f32],
    pub down: W<'a>,
    pub up: W<'a>,
    pub sh1: W<'a>,
    pub sh3: W<'a>,
    pub sh2: W<'a>,
}

pub struct MoeDims {
    pub hidden: usize,
    pub latent: usize,
    pub moe_inter: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub n_shared: usize,
    pub routed_scale: f32,
    pub renorm: bool,
    pub latent_norm: bool,
    pub rms_eps: f32,
    pub situ_b1: f32,
    pub situ_b2: f32,
}

// The ORDER is load bearing: route on the FULL hidden width before any projection,
// down-project, run experts IN LATENT SPACE and sum weighted, RMSNorm the AGGREGATE
// (never per expert), up-project, then add the shared expert computed on the ORIGINAL
// input with NO routing weight and NO scaling.
pub fn moe(out: &mut [f32], x: &[f32], w: &MoeW, d: &MoeDims, t_len: usize) {
    let (e, l, i_n) = (d.hidden, d.latent, d.moe_inter);
    let si = i_n * d.n_shared;

    let mut z = vec![0f32; l];
    let mut acc_l = vec![0f32; l];
    let mut gu = vec![0f32; 2 * i_n];
    let mut act = vec![0f32; i_n];
    let mut edn = vec![0f32; l];
    let mut sgu = vec![0f32; 2 * si];
    let mut sact = vec![0f32; si];
    let mut sdn = vec![0f32; e];
    let mut idx = vec![0i32; d.topk];
    let mut wt = vec![0f32; d.topk];

    for t in 0..t_len {
        let xt = &x[t * e..][..e];
        let ot = &mut out[t * e..][..e];

        router(
            &mut idx, &mut wt, xt, w.gate, w.bias, e, d.n_experts, d.topk, d.renorm,
            d.routed_scale,
        );

        mmw(&mut z, xt, w.down, e, l);
        acc_l.fill(0.0);
        for j in 0..d.topk {
            let ex = idx[j] as usize;
            let e1 = &w.w1[ex * i_n * l..][..i_n * l];
            let e3 = &w.w3[ex * i_n * l..][..i_n * l];
            let e2 = &w.w2[ex * l * i_n..][..l * i_n];
            matmul(&mut gu[..i_n], &z, e1, l, i_n);
            matmul(&mut gu[i_n..], &z, e3, l, i_n);
            situ_glu(&mut act, &gu, i_n, d.situ_b1, d.situ_b2);
            matmul(&mut edn, &act, e2, i_n, l);
            let wj = wt[j];
            for i in 0..l {
                acc_l[i] += wj * edn[i];
            }
        }

        if d.latent_norm {
            let a = acc_l.clone();
            rmsnorm(&mut acc_l, &a, w.latent_norm, l, d.rms_eps);
        }
        mmw(ot, &acc_l, w.up, l, e);

        mmw(&mut sgu[..si], xt, w.sh1, e, si);
        mmw(&mut sgu[si..], xt, w.sh3, e, si);
        situ_glu(&mut sact, &sgu, si, d.situ_b1, d.situ_b2);
        mmw(&mut sdn, &sact, w.sh2, si, e);
        for i in 0..e {
            ot[i] += sdn[i];
        }
    }
}

pub struct KdaW<'a> {
    pub q: W<'a>,
    pub k: W<'a>,
    pub v: W<'a>,
    pub q_conv: &'a [f32],
    pub k_conv: &'a [f32],
    pub v_conv: &'a [f32],
    pub f_a: W<'a>,
    pub f_b: W<'a>,
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub b: W<'a>,
    pub g: W<'a>,
    pub o_norm: &'a [f32],
    pub o: W<'a>,
}

pub struct KdaDims {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_k: usize,
    pub gate_lb: f32,
    pub rms_eps: f32,
}

// L2 normalisation over the last dimension. The reference uses the SUM of squares with
// eps inside the rsqrt, NOT the mean: using the mean scales every q and k by sqrt(d_k)
// and quietly changes the attention temperature.
fn l2norm(v: &mut [f32], n: usize, eps: f32) {
    let mut ss = 0.0f64;
    for i in 0..n {
        ss += v[i] as f64 * v[i] as f64;
    }
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for i in 0..n {
        v[i] *= inv;
    }
}

// state layout: [H*D*D recurrent] then [3 * P * (conv_k-1) shortconv history].
pub fn kda_state_len(d: &KdaDims) -> usize {
    let p = d.heads * d.head_dim;
    d.heads * d.head_dim * d.head_dim + 3 * p * (d.conv_k - 1)
}

pub fn kda_layer(
    out: &mut [f32],
    x: &[f32],
    w: &KdaW,
    d: &KdaDims,
    t_len: usize,
    state: &mut [f32],
) {
    let (e, h_n, dd) = (d.hidden, d.heads, d.head_dim);
    let p = h_n * dd;
    let k_sz = d.conv_k;
    let hist = k_sz - 1;

    let mut q = vec![0f32; t_len * p];
    let mut k = vec![0f32; t_len * p];
    let mut v = vec![0f32; t_len * p];
    let mut z = vec![0f32; t_len * p];
    let mut al = vec![0f32; t_len * p];
    let mut bt = vec![0f32; t_len * h_n];
    let mut o = vec![0f32; t_len * p];
    let mut gb = vec![0f32; p];
    let mut fa = vec![0f32; dd];

    for t in 0..t_len {
        let xt = &x[t * e..][..e];
        mmw(&mut q[t * p..][..p], xt, w.q, e, p);
        mmw(&mut k[t * p..][..p], xt, w.k, e, p);
        mmw(&mut v[t * p..][..p], xt, w.v, e, p);
        mmw(&mut bt[t * h_n..][..h_n], xt, w.b, e, h_n);
        // ONE shared low-rank pair feeds every head: [E->D] then [D->H*D]
        mmw(&mut fa, xt, w.f_a, e, dd);
        mmw(&mut z[t * p..][..p], &fa, w.f_b, dd, p);
    }

    let (rec, conv) = state.split_at_mut(h_n * dd * dd);
    let (cq, rest) = conv.split_at_mut(p * hist);
    let (ck, cv) = rest.split_at_mut(p * hist);
    let qi = q.clone();
    shortconv(&mut q, &qi, w.q_conv, Some(cq), p, k_sz, t_len);
    let ki = k.clone();
    shortconv(&mut k, &ki, w.k_conv, Some(ck), p, k_sz, t_len);
    let vi = v.clone();
    shortconv(&mut v, &vi, w.v_conv, Some(cv), p, k_sz, t_len);

    // L2Norm on q and k ONLY, per head. v is deliberately left alone.
    for t in 0..t_len {
        for h in 0..h_n {
            l2norm(&mut q[t * p + h * dd..][..dd], dd, 1e-6);
            l2norm(&mut k[t * p + h * dd..][..dd], dd, 1e-6);
        }
    }

    for t in 0..t_len {
        for h in 0..h_n {
            bt[t * h_n + h] = sigmoidf(bt[t * h_n + h]);
        }
        let zi = z[t * p..][..p].to_vec();
        let (zs, als) = (&mut z[t * p..][..p], &mut al[t * p..][..p]);
        kda_decay(zs, als, &zi, w.a_log, w.dt_bias, h_n, dd, d.gate_lb);
    }

    let qscale = 1.0 / (dd as f32).sqrt();
    // k3_ops.c:869-873: the recurrence is parallelised over HEADS. Per-head arithmetic is
    // untouched -- the sequential sweep over t stays sequential inside each head, which is
    // what the recurrence requires -- and each head owns its own dd*dd slice of the
    // recurrent state, so the results are bit-identical to the serial form. It is 0.4% of
    // FLOPs but, serial, a majority of non-matmul wall time at high core counts.
    //
    // One difference from the C, forced by the layout rather than chosen: `o` is
    // [t][h][dd], so a head's outputs are strided and cannot be handed out as a mutable
    // sub-slice. Each head fills a contiguous [t][dd] block of its own and they are
    // scattered back below. The scatter is a copy; no arithmetic happens in it.
    let mut oh = vec![0f32; h_n * t_len * dd];
    rec.par_chunks_mut(dd * dd)
        .zip(oh.par_chunks_mut(t_len * dd))
        .enumerate()
        .for_each(|(h, (rec_h, oh_h))| {
            let mut wh = vec![0f32; dd];
            let mut ostep = vec![0f32; dd];
            for t in 0..t_len {
                let off = t * p + h * dd;
                for i in 0..dd {
                    wh[i] = q[off + i] * qscale;
                }
                kda_step(
                    rec_h,
                    &mut ostep,
                    &wh,
                    &k[off..off + dd],
                    &v[off..off + dd],
                    &al[off..off + dd],
                    bt[t * h_n + h],
                    dd,
                    dd,
                );
                oh_h[t * dd..][..dd].copy_from_slice(&ostep);
            }
        });
    for h in 0..h_n {
        for t in 0..t_len {
            let src = &oh[h * t_len * dd + t * dd..][..dd];
            o[t * p + h * dd..][..dd].copy_from_slice(src);
        }
    }

    // head-wise RMSNorm, THEN the gate, THEN the output projection
    for t in 0..t_len {
        let xt = &x[t * e..][..e];
        for h in 0..h_n {
            let src = o[t * p + h * dd..][..dd].to_vec();
            rmsnorm(&mut o[t * p + h * dd..][..dd], &src, w.o_norm, dd, d.rms_eps);
        }
        mmw(&mut gb, xt, w.g, e, p);
        for i in 0..p {
            o[t * p + i] *= sigmoidf(gb[i]);
        }
        let ot = o[t * p..][..p].to_vec();
        mmw(&mut out[t * e..][..e], &ot, w.o, p, e);
    }
}

pub enum Attn<'a> {
    Kda(&'a KdaW<'a>, &'a KdaDims),
    Mla(&'a MlaW<'a>, &'a MlaDims),
}

pub enum Mlp<'a> {
    Moe(&'a MoeW<'a>, &'a MoeDims),
    Dense {
        gate: W<'a>,
        up: W<'a>,
        down: W<'a>,
        inter: usize,
        b1: f32,
        b2: f32,
    },
}

pub struct LayerW<'a> {
    pub in_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub attn_res_norm: &'a [f32],
    pub attn_res_proj: &'a [f32],
    pub mlp_res_norm: &'a [f32],
    pub mlp_res_proj: &'a [f32],
    pub attn: Attn<'a>,
    pub mlp: Mlp<'a>,
}

pub fn fold_residual(norm: &[f32], proj: &[f32]) -> Vec<f32> {
    // The norm gain and the scoring projection collapse to ONE vector.
    norm.iter().zip(proj).map(|(a, b)| a * b).collect()
}

// Architecture-neutral: the residual scheme is a trait, so K3's block snapshots and
// DeepSeek-V4's Hyper-Connections are two implementations rather than two code paths.
pub fn decoder_layer<R: Residual>(
    res: &mut R,
    w: &LayerW,
    hidden: usize,
    t_len: usize,
    state: &mut [f32],
    rms_eps: f32,
) {
    decoder_layer_inc(res, w, hidden, t_len, state, rms_eps, None, 0)
}

pub fn decoder_layer_inc<R: Residual>(
    res: &mut R,
    w: &LayerW,
    hidden: usize,
    t_len: usize,
    state: &mut [f32],
    rms_eps: f32,
    kv: Option<(&mut [f32], &mut [f32])>,
    cached: usize,
) {
    let e = hidden;
    let mut modin = vec![0f32; t_len * e];
    let mut hin = vec![0f32; t_len * e];
    let mut tmp = vec![0f32; t_len * e];

    let carry = res.pre(Sub::Attn, &mut modin);
    for t in 0..t_len {
        rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], w.in_norm, e, rms_eps);
    }
    match &w.attn {
        Attn::Kda(kw, kd) => kda_layer(&mut tmp, &hin, kw, kd, t_len, state),
        Attn::Mla(mw, md) => mla_cached(&mut tmp, &hin, mw, md, t_len, kv, cached),
    }
    res.post(Sub::Attn, &tmp, carry);

    let carry = res.pre(Sub::Mlp, &mut modin);
    for t in 0..t_len {
        rmsnorm(&mut hin[t * e..][..e], &modin[t * e..][..e], w.post_norm, e, rms_eps);
    }
    match &w.mlp {
        Mlp::Moe(mw, md) => moe(&mut tmp, &hin, mw, md, t_len),
        Mlp::Dense { gate, up, down, inter, b1, b2 } => {
            let mut dgu = vec![0f32; 2 * inter];
            let mut act = vec![0f32; *inter];
            for t in 0..t_len {
                mmw(&mut dgu[..*inter], &hin[t * e..][..e], *gate, e, *inter);
                mmw(&mut dgu[*inter..], &hin[t * e..][..e], *up, e, *inter);
                situ_glu(&mut act, &dgu, *inter, *b1, *b2);
                mmw(&mut tmp[t * e..][..e], &act, *down, *inter, e);
            }
        }
    }
    res.post(Sub::Mlp, &tmp, carry);
}

// ---------------------------------------------------------------- multi-model ----
// Seams derived from two real architectures rather than guessed from one. See
// docs/MULTI_MODEL.md for the full comparison.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Glu {
    // K3: the sigmoid takes the UNCAPPED gate.
    SiTu { b1: f32, b2: f32 },
    // DeepSeek-V4 (inference/model.py Expert.forward). The clamp is ASYMMETRIC:
    // up is bounded both sides, gate only from above. Clamping gate's min too is a
    // plausible, bounded, wrong function.
    SwigluClamped { limit: f32 },
    // MiniMax-M3 `swigluoai`: the gate's sigmoid argument is scaled by alpha, and the up
    // branch carries a +1 bias. Dropping either leaves a bounded, plausible activation.
    SwigluOai { alpha: f32, limit: f32 },
}

pub fn glu(y: &mut [f32], x: &[f32], n: usize, k: Glu) {
    match k {
        Glu::SiTu { b1, b2 } => situ_glu(y, x, n, b1, b2),
        Glu::SwigluClamped { limit } => {
            let (gate, up) = (&x[..n], &x[n..2 * n]);
            for i in 0..n {
                let mut g = gate[i];
                let mut u = up[i];
                if limit.is_finite() && limit > 0.0 {
                    u = u.clamp(-limit, limit);
                    g = g.min(limit);
                }
                y[i] = (g * sigmoidf(g)) * u;
            }
        }
        Glu::SwigluOai { alpha, limit } => {
            let (gate, up) = (&x[..n], &x[n..2 * n]);
            for i in 0..n {
                let mut g = gate[i];
                let mut u = up[i];
                if limit.is_finite() && limit > 0.0 {
                    u = u.clamp(-limit, limit);
                    g = g.min(limit);
                }
                y[i] = (g * sigmoidf(alpha * g)) * (u + 1.0);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scoring {
    Sigmoid,
    SqrtSoftplus,
    Softmax,
}

// torch's F.softplus reverts to the identity above threshold=20 for stability, and the
// reference relies on that default.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Gated DeltaNet forget gate, as Qwen3.5/3.6 computes it.
///
/// `decay[h] = exp(softplus(alpha[h] + dt_bias[h]) * a[h])`, where `a` is stored already
/// negative in the checkpoint, so the product is <= 0 and the decay lands in (0, 1].
///
/// TWO differences from `kda_decay`, and both are silent if got wrong:
///
///   * The formula. K3's KDA is `lb * sigmoid(exp(a_log) * (z + dt_bias))`; this is
/// ```text
///     `exp(softplus(...) * a)`. Both produce a number in (0, 1) that decays a state, so
///     a mix-up degrades quality without ever failing.
/// ```
///   * The SHAPE. K3's gate is per key CHANNEL -- one value per state row. This one is per
/// ```text
///     v-HEAD: llama.cpp reshapes it to leading dimension 1 and broadcasts, so a single
///     scalar scales the whole per-head state matrix. Indexing it per channel reads
///     neighbouring heads' gates and still decays plausibly.
/// ```
pub fn gdn_decay(decay: &mut [f32], alpha: &[f32], dt_bias: &[f32], a: &[f32], h_n: usize) {
    for h in 0..h_n {
        decay[h] = (softplus(alpha[h] + dt_bias[h]) * a[h]).exp();
    }
}

/// L2 normalise each head's vector: `x / sqrt(sum(x^2) + eps)`.
///
/// Note this is `ggml_l2_norm`, NOT RMS norm: there is no division by `sqrt(n)` and no
/// learned gain. Substituting rmsnorm rescales q and k by sqrt(head_dim) -- 11.3x here --
/// which the delta rule partly absorbs, so the model still reads fluently.
pub fn l2norm_heads(y: &mut [f32], x: &[f32], heads: usize, dim: usize, eps: f32) {
    for h in 0..heads {
        let (src, dst) = (&x[h * dim..][..dim], &mut y[h * dim..][..dim]);
        let mut ss = 0.0f64;
        for v in src {
            ss += *v as f64 * *v as f64;
        }
        let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s * inv;
        }
    }
}

impl Scoring {
    fn apply(self, logits: &mut [f32]) {
        match self {
            Scoring::Sigmoid => {
                for v in logits.iter_mut() {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
            Scoring::SqrtSoftplus => {
                for v in logits.iter_mut() {
                    *v = softplus(*v).sqrt();
                }
            }
            Scoring::Softmax => {
                let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut z = 0.0f64;
                for v in logits.iter_mut() {
                    *v = (*v - m).exp();
                    z += *v as f64;
                }
                for v in logits.iter_mut() {
                    *v = (*v as f64 / z) as f32;
                }
            }
        }
    }
}

// K3's router with the score function lifted out. Both architectures add the bias for
// SELECTION only and return the unbiased score as the weight (noaux_tc); both renorm
// over the selected set and then scale.
#[allow(clippy::too_many_arguments)]
pub fn router_scored(
    idx: &mut [i32],
    wt: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    hidden: usize,
    n_experts: usize,
    topk: usize,
    renorm: bool,
    routed_scale: f32,
    scoring: Scoring,
) {
    let mut score = vec![0f32; n_experts];
    for e in 0..n_experts {
        let row = &w[e * hidden..][..hidden];
        let mut acc = 0.0f64;
        for i in 0..hidden {
            acc += row[i] as f64 * x[i] as f64;
        }
        score[e] = acc as f32;
    }
    scoring.apply(&mut score);
    let mut choice: Vec<f32> =
        score.iter().enumerate().map(|(e, s)| s + bias.map_or(0.0, |b| b[e])).collect();

    for j in 0..topk {
        let mut best = -1i32;
        let mut bv = f32::NEG_INFINITY;
        for e in 0..n_experts {
            if choice[e] > bv {
                bv = choice[e];
                best = e as i32;
            }
        }
        if best < 0 {
            idx[j] = 0;
            wt[j] = 0.0;
            continue;
        }
        idx[j] = best;
        wt[j] = score[best as usize];
        choice[best as usize] = f32::NEG_INFINITY;
    }

    if renorm && topk > 1 {
        let s: f64 = wt[..topk].iter().map(|&v| v as f64).sum();
        let inv = (1.0 / (s + 1e-20)) as f32;
        for j in 0..topk {
            wt[j] *= inv;
        }
    }
    for j in 0..topk {
        wt[j] *= routed_scale;
    }
}

// DeepSeek-V4's first n_hash_layers route by TOKEN ID, not by score: the expert set is
// a lookup, and only the weights come from the scores (inference/model.py Gate.forward).
#[allow(clippy::too_many_arguments)]
pub fn router_hashed(
    idx: &mut [i32],
    wt: &mut [f32],
    x: &[f32],
    w: &[f32],
    tid2eid: &[i32],
    token_id: usize,
    hidden: usize,
    n_experts: usize,
    topk: usize,
    routed_scale: f32,
    scoring: Scoring,
) {
    let mut score = vec![0f32; n_experts];
    for e in 0..n_experts {
        let row = &w[e * hidden..][..hidden];
        let mut acc = 0.0f64;
        for i in 0..hidden {
            acc += row[i] as f64 * x[i] as f64;
        }
        score[e] = acc as f32;
    }
    scoring.apply(&mut score);
    for j in 0..topk {
        idx[j] = tid2eid[token_id * topk + j];
        wt[j] = score[idx[j] as usize];
    }
    let s: f64 = wt[..topk].iter().map(|&v| v as f64).sum();
    let inv = (1.0 / (s + 1e-20)) as f32;
    for j in 0..topk {
        wt[j] *= inv * routed_scale;
    }
}

// FP8 e4m3 weights with one E8M0 scale per [block x block] tile, the format DeepSeek-V4
// uses for every non-expert matrix. Scale is [ceil(out/block)][ceil(in/block)].
//
// NOT bit-identical to the reference: inference/model.py quantises the ACTIVATION to
// fp8 before the gemm (act_quant, activation_scheme "dynamic") and accumulates in fp32.
// This dequantises the weight and dots against the full-precision activation in f64,
// which is strictly more accurate and therefore different. Matching the reference
// exactly would mean reproducing the activation quantisation too.
#[allow(clippy::too_many_arguments)]
pub fn matmul_fp8_block(
    y: &mut [f32],
    x: &[f32],
    w: &[u8],
    scale: &[u8],
    k_in: usize,
    out: usize,
    block: usize,
) {
    // Every DeepSeek-V4 attention projection is this kernel, so leaving it serial leaves
    // the whole trunk serial. It is not one of the C engine's seven OpenMP regions
    // because it has no C counterpart -- V4 is FP8 where K3 is bf16.
    //
    // The chunk must be a multiple of `block`: the scale grid is 128x128, so a chunk
    // starting mid-block would need a fractional scale-row offset and would silently
    // pair rows with the wrong scales.
    if out > PAR_MIN_ROWS {
        let chunk = row_chunk(out).next_multiple_of(block).max(block);
        let sb_in = k_in.div_ceil(block);
        y[..out].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            matmul_fp8_block_serial(
                yc,
                x,
                &w[base * k_in..],
                &scale[(base / block) * sb_in..],
                k_in,
                yc.len(),
                block,
            );
        });
        return;
    }
    matmul_fp8_block_serial(y, x, w, scale, k_in, out, block);
}

// Lane l of v_k holds elements at position 4k+l within each 16-wide step, which is the
// same partition reduce16 folds: (v0+v1)+(v2+v3) lanewise gives exactly its b0..b3, and
// the cross-lane (a0+a1)+(a2+a3) closes the identical tree. So this is bit-identical to
// the scalar path, not merely close -- asserted by a test, because that is the only way
// to know.
//
// The dequant is a 256-entry table lookup rather than arithmetic: e4m3 has no bit trick
// the way bf16 does (bf16 is just a shift), and the lookup is an L1 hit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_fp8_block_avx2(
    y: &mut [f32],
    x: &[f32],
    w: &[u8],
    scale: &[u8],
    k_in: usize,
    out: usize,
    block: usize,
) {
    use std::arch::x86_64::*;
    let sb_in = k_in.div_ceil(block);
    let e4m3 = crate::st::e4m3_table();
    let xp = x.as_ptr();
    for o in 0..out {
        let row = w.as_ptr().add(o * k_in);
        let srow = scale.as_ptr().add((o / block) * sb_in);
        let mut acc = 0.0f64;
        for gb in 0..sb_in {
            let sc = crate::st::e8m0_to_f32(*srow.add(gb)) as f64;
            if sc == 0.0 {
                continue;
            }
            let lo = gb * block;
            let hi = (lo + block).min(k_in);
            let (mut v0, mut v1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let (mut v2, mut v3) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let mut buf = [0f32; 16];
            let mut i = lo;
            while i + 15 < hi {
                for l in 0..16 {
                    *buf.get_unchecked_mut(l) = *e4m3.get_unchecked(*row.add(i + l) as usize);
                }
                let b = buf.as_ptr();
                v0 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(b)),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i))),
                    v0,
                );
                v1 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(b.add(4))),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 4))),
                    v1,
                );
                v2 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(b.add(8))),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 8))),
                    v2,
                );
                v3 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(b.add(12))),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 12))),
                    v3,
                );
                i += 16;
            }
            let vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
            let mut a = [0f64; 4];
            _mm256_storeu_pd(a.as_mut_ptr(), vt);
            let mut sub = (a[0] + a[1]) + (a[2] + a[3]);
            while i < hi {
                sub = (*e4m3.get_unchecked(*row.add(i) as usize) as f64)
                    .mul_add(*x.get_unchecked(i) as f64, sub);
                i += 1;
            }
            acc += sub * sc;
        }
        *y.get_unchecked_mut(o) = acc as f32;
    }
}

fn matmul_fp8_block_serial(
    y: &mut [f32],
    x: &[f32],
    w: &[u8],
    scale: &[u8],
    k_in: usize,
    out: usize,
    block: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul_fp8_block_avx2(y, x, w, scale, k_in, out, block) };
            return;
        }
    }
    matmul_fp8_block_scalar(y, x, w, scale, k_in, out, block);
}

fn matmul_fp8_block_scalar(
    y: &mut [f32],
    x: &[f32],
    w: &[u8],
    scale: &[u8],
    k_in: usize,
    out: usize,
    block: usize,
) {
    let sb_in = k_in.div_ceil(block);
    let e4m3 = crate::st::e4m3_table();
    for o in 0..out {
        let row = &w[o * k_in..][..k_in];
        let srow = &scale[(o / block) * sb_in..][..sb_in];
        let mut acc = 0.0f64;
        for gb in 0..sb_in {
            let sc = crate::st::e8m0_to_f32(srow[gb]) as f64;
            if sc == 0.0 {
                continue;
            }
            let lo = gb * block;
            let hi = (lo + block).min(k_in);
            let mut a = [0f64; 16];
            let mut i = lo;
            while i + 15 < hi {
                for l in 0..16 {
                    a[l] = (e4m3[row[i + l] as usize] as f64).mul_add(x[i + l] as f64, a[l]);
                }
                i += 16;
            }
            let mut sub = reduce16(&a);
            while i < hi {
                sub = (e4m3[row[i] as usize] as f64).mul_add(x[i] as f64, sub);
                i += 1;
            }
            acc += sub * sc;
        }
        y[o] = acc as f32;
    }
}

// ------------------------------------------------------------------ residual ----
// How a module's output rejoins the hidden stream. K3 keeps a stack of block snapshots
// and scores them with AttnRes; DeepSeek-V4 keeps hc_mult parallel copies and mixes
// them with Sinkhorn-normalised weights. Same job, different shape -- so it is a trait
// with pre/post rather than a parameter list on decoder_layer.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Sub {
    Attn,
    Mlp,
}

pub trait Residual {
    /// Whatever `pre` must hand `post` for the same sub-module.
    type Carry;
    /// Reduce the carried state to this sub-module's input, [T][hidden].
    fn pre(&mut self, sub: Sub, input: &mut [f32]) -> Self::Carry;
    /// Rejoin the sub-module's output, [T][hidden].
    fn post(&mut self, sub: Sub, output: &[f32], carry: Self::Carry);
    /// The hidden state a following layer reads, [T][hidden].
    fn hidden(&self) -> &[f32];
}

/// K3: Block Attention Residuals.
pub struct AttnResidual {
    h: Vec<f32>,
    pref: Vec<f32>,
    /// [T][maxb][hidden]
    blocks: Vec<f32>,
    n_blocks: usize,
    have_prefix: bool,
    t_len: usize,
    hidden: usize,
    maxb: usize,
    eps: f32,
    /// Per-layer, set by `begin_layer`. norm gain and scoring projection, folded.
    fold_a: Vec<f32>,
    fold_m: Vec<f32>,
    layer_idx: usize,
    attn_res_block: usize,
    src: Vec<f32>,
}

impl AttnResidual {
    pub fn new(h0: &[f32], t_len: usize, hidden: usize, maxb: usize, eps: f32) -> Self {
        AttnResidual {
            h: h0[..t_len * hidden].to_vec(),
            pref: vec![0.0; t_len * hidden],
            blocks: vec![0.0; t_len * maxb * hidden],
            n_blocks: 0,
            have_prefix: true,
            t_len,
            hidden,
            maxb,
            eps,
            fold_a: Vec::new(),
            fold_m: Vec::new(),
            layer_idx: 0,
            attn_res_block: 1,
            src: vec![0.0; (maxb + 1) * hidden],
        }
    }

    pub fn begin_layer(
        &mut self,
        layer_idx: usize,
        attn_res_block: usize,
        fold_a: &[f32],
        fold_m: &[f32],
    ) {
        self.layer_idx = layer_idx;
        self.attn_res_block = attn_res_block;
        self.fold_a.clear();
        self.fold_a.extend_from_slice(fold_a);
        self.fold_m.clear();
        self.fold_m.extend_from_slice(fold_m);
        self.pref.copy_from_slice(&self.h);
        self.have_prefix = true;
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// The one model-level aggregator, beyond the two per layer: output_attn_res_{norm,proj}.
    /// The tensor census counts exactly one of these and skipping it is silent.
    pub fn finish(&mut self, norm: &[f32], proj: &[f32]) -> &[f32] {
        let fold = fold_residual(norm, proj);
        self.aggregate(&fold);
        &self.h
    }

    fn aggregate(&mut self, fold: &[f32]) {
        let (e, nb) = (self.hidden, self.n_blocks);
        for t in 0..self.t_len {
            for b in 0..nb {
                self.src[b * e..][..e]
                    .copy_from_slice(&self.blocks[(t * self.maxb + b) * e..][..e]);
            }
            self.src[nb * e..][..e].copy_from_slice(&self.pref[t * e..][..e]);
            attn_res(&mut self.h[t * e..][..e], &self.src, fold, nb + 1, e, self.eps);
        }
    }
}

impl Residual for AttnResidual {
    type Carry = ();

    fn pre(&mut self, sub: Sub, input: &mut [f32]) {
        match sub {
            Sub::Attn => {
                // aggregation before attention, only when snapshots already exist
                if self.n_blocks > 0 {
                    let f = std::mem::take(&mut self.fold_a);
                    self.aggregate(&f);
                    self.fold_a = f;
                }
                // block boundary: snapshot the running residual, then CLEAR it. This sits
                // between the aggregate and the module because the aggregate reads the
                // pre-snapshot prefix.
                if self.layer_idx % self.attn_res_block == 0 {
                    let e = self.hidden;
                    for t in 0..self.t_len {
                        let d = (t * self.maxb + self.n_blocks) * e;
                        self.blocks[d..d + e].copy_from_slice(&self.pref[t * e..][..e]);
                    }
                    self.n_blocks += 1;
                    self.have_prefix = false;
                }
            }
            // aggregation before the MLP. NO emptiness guard in the reference.
            Sub::Mlp => {
                let f = std::mem::take(&mut self.fold_m);
                self.aggregate(&f);
                self.fold_m = f;
            }
        }
        input[..self.t_len * self.hidden].copy_from_slice(&self.h);
    }

    fn post(&mut self, _sub: Sub, output: &[f32], _c: ()) {
        let n = self.t_len * self.hidden;
        if self.have_prefix {
            for i in 0..n {
                self.pref[i] += output[i];
            }
        } else {
            self.pref[..n].copy_from_slice(&output[..n]);
            self.have_prefix = true;
        }
        self.h.copy_from_slice(&self.pref);
    }

    fn hidden(&self) -> &[f32] {
        &self.h
    }
}

/// DeepSeek-V4: Hyper-Connections. `mixes` is split into pre weights, post weights and
/// a combination matrix; the matrix is softmaxed then Sinkhorn-normalised toward doubly
/// stochastic (inference/kernel.py, hc_split_sinkhorn_kernel).
pub fn hc_split_sinkhorn(
    mixes: &[f32],
    hc_scale: &[f32; 3],
    hc_base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut pre = vec![0f32; hc];
    let mut post = vec![0f32; hc];
    let mut comb = vec![0f32; hc * hc];

    for j in 0..hc {
        pre[j] = sigmoidf(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
        // post carries a factor of 2 the other two do not.
        post[j] = 2.0 * sigmoidf(mixes[j + hc] * hc_scale[1] + hc_base[j + hc]);
    }
    for j in 0..hc {
        for k in 0..hc {
            let o = j * hc + k + 2 * hc;
            comb[j * hc + k] = mixes[o] * hc_scale[2] + hc_base[o];
        }
    }

    // softmax over each row, then + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..][..hc];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            z += *v;
        }
        for v in row.iter_mut() {
            *v = *v / z + eps;
        }
    }
    // one column normalisation, then iters-1 rounds of row-then-column
    let col_norm = |c: &mut Vec<f32>| {
        for k in 0..hc {
            let s: f32 = (0..hc).map(|j| c[j * hc + k]).sum();
            for j in 0..hc {
                c[j * hc + k] /= s + eps;
            }
        }
    };
    let row_norm = |c: &mut Vec<f32>| {
        for j in 0..hc {
            let s: f32 = c[j * hc..][..hc].iter().sum();
            for k in 0..hc {
                c[j * hc + k] /= s + eps;
            }
        }
    };
    col_norm(&mut comb);
    for _ in 0..iters.saturating_sub(1) {
        row_norm(&mut comb);
        col_norm(&mut comb);
    }
    (pre, post, comb)
}

/// The final Hyper-Connections reduce: `hc` copies -> 1, by a learned sigmoid gate.
///
/// NOT a mean. `pre` here is `sigmoid(mixes * scale + base) + hc_eps`, a per-copy gate the
/// checkpoint ships parameters for (`hc_head_fn [hc][hc*d]`, `hc_head_base [hc]`,
/// `hc_head_scale [1]`). Averaging instead is a plausible-looking error that survives
/// inspection: the reduce is followed by an RMSNorm, which cancels any GLOBAL rescaling,
/// so a mean differs from the real thing only in how it weights the four copies relative
/// to one another -- enough to move logits, not enough to stop the text reading fluently.
///
/// `x` is one token's [hc][hidden]; `out` is [hidden].
#[allow(clippy::too_many_arguments)]
pub fn hc_head(
    out: &mut [f32],
    x: &[f32],
    fnw: &[f32],
    base: &[f32],
    scale: f32,
    hidden: usize,
    hc: usize,
    norm_eps: f32,
    hc_eps: f32,
) {
    let hcd = hc * hidden;
    let mut ss = 0.0f64;
    for v in &x[..hcd] {
        ss += *v as f64 * *v as f64;
    }
    let rsqrt = (1.0 / (ss / hcd as f64 + norm_eps as f64).sqrt()) as f32;
    out[..hidden].fill(0.0);
    for j in 0..hc {
        let row = &fnw[j * hcd..][..hcd];
        let mut acc = 0.0f64;
        for i in 0..hcd {
            acc += row[i] as f64 * x[i] as f64;
        }
        let g = sigmoidf(acc as f32 * rsqrt * scale + base[j]) + hc_eps;
        let src = &x[j * hidden..][..hidden];
        for i in 0..hidden {
            out[i] += g * src[i];
        }
    }
}

pub struct HcLayer<'a> {
    pub attn_fn: &'a [f32],
    pub attn_base: &'a [f32],
    pub attn_scale: [f32; 3],
    pub ffn_fn: &'a [f32],
    pub ffn_base: &'a [f32],
    pub ffn_scale: [f32; 3],
}

/// Owned per-layer Hyper-Connections weights. Owned rather than borrowed because the
/// residual outlives any one layer's weights: it carries the stream across all 43.
#[derive(Default, Clone)]
pub struct HcOwned {
    pub attn_fn: Vec<f32>,
    pub attn_base: Vec<f32>,
    pub attn_scale: [f32; 3],
    pub ffn_fn: Vec<f32>,
    pub ffn_base: Vec<f32>,
    pub ffn_scale: [f32; 3],
}

impl HcOwned {
    pub fn from(l: &HcLayer) -> HcOwned {
        HcOwned {
            attn_fn: l.attn_fn.to_vec(),
            attn_base: l.attn_base.to_vec(),
            attn_scale: l.attn_scale,
            ffn_fn: l.ffn_fn.to_vec(),
            ffn_base: l.ffn_base.to_vec(),
            ffn_scale: l.ffn_scale,
        }
    }
}

pub struct HyperConnResidual {
    /// [T][hc][hidden]
    x: Vec<f32>,
    residual: Vec<f32>,
    flat: Vec<f32>,
    t_len: usize,
    hidden: usize,
    hc: usize,
    norm_eps: f32,
    hc_eps: f32,
    iters: usize,
    layer: HcOwned,
    /// The reduced [T][hidden] view a following layer would read.
    view: Vec<f32>,
}

pub struct HcCarry {
    post: Vec<f32>,
    comb: Vec<f32>,
}

impl HyperConnResidual {
    pub fn new(
        x0: &[f32],
        t_len: usize,
        hidden: usize,
        hc: usize,
        norm_eps: f32,
        hc_eps: f32,
        iters: usize,
    ) -> Self {
        HyperConnResidual {
            x: x0[..t_len * hc * hidden].to_vec(),
            residual: vec![0.0; t_len * hc * hidden],
            flat: vec![0.0; hc * hidden],
            t_len,
            hidden,
            hc,
            norm_eps,
            hc_eps,
            iters,
            layer: HcOwned::default(),
            view: vec![0.0; t_len * hidden],
        }
    }

    pub fn begin_layer(&mut self, layer: &HcLayer) {
        self.layer = HcOwned::from(layer);
    }

    pub fn state(&self) -> &[f32] {
        &self.x
    }
}

impl Residual for HyperConnResidual {
    type Carry = HcCarry;

    fn pre(&mut self, sub: Sub, input: &mut [f32]) -> HcCarry {
        let (fnw, base, scale) = match sub {
            Sub::Attn => (
                self.layer.attn_fn.clone(),
                self.layer.attn_base.clone(),
                self.layer.attn_scale,
            ),
            Sub::Mlp => (
                self.layer.ffn_fn.clone(),
                self.layer.ffn_base.clone(),
                self.layer.ffn_scale,
            ),
        };
        assert!(!fnw.is_empty(), "begin_layer before pre");
        let (e, hc) = (self.hidden, self.hc);
        let hcd = hc * e;
        let mix_hc = (2 + hc) * hc;

        self.residual.copy_from_slice(&self.x);
        let mut post_all = vec![0f32; self.t_len * hc];
        let mut comb_all = vec![0f32; self.t_len * hc * hc];

        for t in 0..self.t_len {
            self.flat.copy_from_slice(&self.x[t * hcd..][..hcd]);
            // rsqrt over the FLATTENED hc*d vector, not per copy.
            let mut ss = 0.0f64;
            for v in &self.flat {
                ss += *v as f64 * *v as f64;
            }
            let rsqrt = (1.0 / (ss / hcd as f64 + self.norm_eps as f64).sqrt()) as f32;

            let mut mixes = vec![0f32; mix_hc];
            for m in 0..mix_hc {
                let row = &fnw[m * hcd..][..hcd];
                let mut acc = 0.0f64;
                for i in 0..hcd {
                    acc += row[i] as f64 * self.flat[i] as f64;
                }
                mixes[m] = acc as f32 * rsqrt;
            }
            let (pre, post, comb) =
                hc_split_sinkhorn(&mixes, &scale, &base, hc, self.iters, self.hc_eps);

            let o = &mut input[t * e..][..e];
            o.fill(0.0);
            for j in 0..hc {
                let pj = pre[j];
                let src = &self.flat[j * e..][..e];
                for i in 0..e {
                    o[i] += pj * src[i];
                }
            }
            post_all[t * hc..][..hc].copy_from_slice(&post);
            comb_all[t * hc * hc..][..hc * hc].copy_from_slice(&comb);
        }
        HcCarry { post: post_all, comb: comb_all }
    }

    fn post(&mut self, _sub: Sub, output: &[f32], c: HcCarry) {
        let (e, hc) = (self.hidden, self.hc);
        for t in 0..self.t_len {
            let out = &output[t * e..][..e];
            for k in 0..hc {
                let dst = &mut self.x[(t * hc + k) * e..][..e];
                let pk = c.post[t * hc + k];
                for i in 0..e {
                    dst[i] = pk * out[i];
                }
                // comb is summed over its FIRST index: y[k] = post[k]*out + sum_j comb[j][k]*res[j].
                // Transposing this is a bounded, plausible mix and is wrong.
                for j in 0..hc {
                    let w = c.comb[t * hc * hc + j * hc + k];
                    let res = &self.residual[(t * hc + j) * e..][..e];
                    for i in 0..e {
                        dst[i] += w * res[i];
                    }
                }
            }
        }
    }

    fn hidden(&self) -> &[f32] {
        &self.view
    }
}

impl AttnResidual {
    /// Seed the block stack from a fixture's `block_residual_in`, laid out [T][nb_in][E].
    pub fn seed_blocks(&mut self, bri: &[f32], nb_in: usize) {
        let e = self.hidden;
        for t in 0..self.t_len {
            for b in 0..nb_in {
                self.blocks[(t * self.maxb + b) * e..][..e]
                    .copy_from_slice(&bri[(t * nb_in + b) * e..][..e]);
            }
        }
        self.n_blocks = nb_in;
    }
}

// ------------------------------------------------------- threading, gated per kernel ---
//
// docs/RUST_PORT.md:281: OpenMP's `schedule(static)` becomes rayon "over the same chunk
// boundaries -- but verify that per kernel rather than assuming it."
//
// These run each parallelised kernel under a one-thread pool and under an eight-thread
// pool and require the results to be BITWISE equal, not merely close. That is a stronger
// property than the op fixtures give: a fixture proves the kernel agrees with the torch
// reference inside a tolerance, and would still pass if a chunking mistake perturbed the
// last bits. Thread-count independence is what the C build claims at k3_ops.c:870 and it
// is the property the whole memory ladder rests on -- output is byte-identical across
// twelve memory budgets, which cannot survive a kernel whose result depends on how the
// work was divided.
#[cfg(test)]
mod threading {
    use super::*;

    fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    /// Compare as bit patterns. `assert_eq!` on f32 would accept a NaN mismatch and,
    /// more to the point, reads as a tolerance check to anyone skimming.
    fn same_bits(what: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (&p, &q)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{what}: element {i} differs between 1 and 8 threads: {p:e} vs {q:e}"
            );
        }
    }

    fn x_of(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.017).sin()).collect()
    }

    // Every case uses out/rows > PAR_MIN_ROWS so the parallel path is actually taken; a
    // test below the threshold would compare the serial path against itself and pass
    // whatever the chunking did.
    const OUT: usize = 320;
    const K_IN: usize = 256;

    #[test]
    fn matmul_is_independent_of_thread_count() {
        let x = x_of(K_IN);
        let w: Vec<f32> = (0..K_IN * OUT).map(|i| ((i as f32) * 0.003).cos()).collect();
        let run = || {
            let mut y = vec![0f32; OUT];
            matmul(&mut y, &x, &w, K_IN, OUT);
            y
        };
        same_bits("matmul", &in_pool(1, run), &in_pool(8, run));
    }

    #[test]
    fn matmul_bf16_is_independent_of_thread_count() {
        let x = x_of(K_IN);
        let w: Vec<u16> = (0..K_IN * OUT).map(|i| 0x3F00u16.wrapping_add(i as u16 % 97)).collect();
        let run = || {
            let mut y = vec![0f32; OUT];
            matmul_bf16(&mut y, &x, &w, K_IN, OUT);
            y
        };
        same_bits("matmul_bf16", &in_pool(1, run), &in_pool(8, run));
    }

    // Also worth pinning across threads even though the draft path carries no
    // cross-path determinism contract: it accumulates in f32, where a reassociation
    // would show up far sooner than it would in the f64 kernels.
    #[test]
    fn matmul_q8_is_independent_of_thread_count() {
        let x = x_of(K_IN);
        let rowb = 4 + K_IN;
        let mut w = vec![0u8; rowb * OUT];
        for r in 0..OUT {
            w[r * rowb..r * rowb + 4].copy_from_slice(&(0.01f32 + r as f32 * 1e-4).to_le_bytes());
            for i in 0..K_IN {
                w[r * rowb + 4 + i] = ((r * 31 + i * 7) % 251) as u8;
            }
        }
        let run = || {
            let mut y = vec![0f32; OUT];
            matmul_q8(&mut y, &x, &w, K_IN, OUT);
            y
        };
        same_bits("matmul_q8", &in_pool(1, run), &in_pool(8, run));
    }

    // The one with two strides: nibbles and per-group E8M0 scales live in separate
    // tensors, so a chunk has to be offset into both. Getting only one of them right
    // gives every row past the first chunk the wrong scales -- which stays finite and
    // plausible, so this test exists to catch exactly that.
    #[test]
    fn matmul_mxfp4_is_independent_of_thread_count() {
        let group = MXFP4_GROUP;
        let rows = OUT;
        let x = x_of(K_IN);
        // `i % 256` would alias the same way the scales did: a row is 128 packed bytes, so
        // row r and row r+2k start at the same value and dropping the packed offset went
        // undetected. 251 is coprime with the row stride.
        let packed: Vec<u8> = (0..K_IN / 2 * rows).map(|i| (i * 7 % 251) as u8).collect();
        // Two constraints on this data, both learned by breaking the kernel on purpose.
        //
        // Keep every scale away from 255: that byte is a NaN scale by the OCP MX spec and
        // makes its whole group contribute nothing, which would mask a striding bug.
        //
        // And make the period coprime with the chunk sizes. An earlier version used
        // `120 + i % 16` with ngrp = 8, which makes a row's scales depend only on its
        // PARITY -- so with both chunk sizes even, dropping the scale offset entirely
        // still produced identical output and the test passed on a broken kernel. 13 is
        // coprime with ngrp and with the row counts either pool picks.
        let ngrp = K_IN.div_ceil(group);
        let scales: Vec<u8> = (0..ngrp * rows).map(|i| (120 + i % 13) as u8).collect();
        let run = || {
            let mut y = vec![0f32; rows];
            matmul_mxfp4(&mut y, &x, &packed, &scales, K_IN, rows, group);
            y
        };
        same_bits("matmul_mxfp4", &in_pool(1, run), &in_pool(8, run));
    }

    // The router returns indices as well as weights, and the top-k selection that
    // consumes the parallel scores is a sequential scan whose tie-breaking is part of the
    // contract. Check both halves.
    // The V4 trunk's kernel. Its scale grid is 128x128, so a chunk boundary that is not
    // a multiple of `block` pairs rows with the wrong scales -- finite, plausible, wrong.
    #[test]
    fn matmul_fp8_block_is_independent_of_thread_count() {
        let (k_in, out, block) = (512usize, 384usize, 128usize);
        let x = x_of(k_in);
        // Avoid the e4m3 NaN bytes (0x7F/0xFF) so every element contributes.
        let w: Vec<u8> = (0..k_in * out).map(|i| ((i * 7) % 126) as u8).collect();
        let sb = k_in.div_ceil(block) * out.div_ceil(block);
        // Period coprime with the block grid, or a stride bug aliases and hides.
        let scale: Vec<u8> = (0..sb).map(|i| (120 + i % 13) as u8).collect();
        let run = || {
            let mut y = vec![0f32; out];
            matmul_fp8_block(&mut y, &x, &w, &scale, k_in, out, block);
            y
        };
        same_bits("matmul_fp8_block", &in_pool(1, run), &in_pool(8, run));
    }

    // Thread-count independence is NOT enough on its own. Both pools can choose the same
    // chunk size, in which case a stride bug perturbs both runs identically and the
    // comparison passes on a broken kernel -- observed, by dropping the scale offset and
    // watching the test stay green. Comparing the parallel path against the serial one
    // is the property that actually pins the offsets.
    #[test]
    fn the_parallel_matmuls_agree_with_their_serial_paths_bitwise() {
        let (k_in, out, block) = (512usize, 384usize, 128usize);
        let x = x_of(k_in);
        assert!(out > PAR_MIN_ROWS, "the parallel path must actually be taken");

        let w: Vec<u8> = (0..k_in * out).map(|i| ((i * 7) % 126) as u8).collect();
        let sb = k_in.div_ceil(block) * out.div_ceil(block);
        let scale: Vec<u8> = (0..sb).map(|i| (120 + i % 13) as u8).collect();
        let (mut p, mut q) = (vec![0f32; out], vec![0f32; out]);
        matmul_fp8_block(&mut p, &x, &w, &scale, k_in, out, block);
        matmul_fp8_block_serial(&mut q, &x, &w, &scale, k_in, out, block);
        same_bits("matmul_fp8_block", &p, &q);

        let wf: Vec<f32> = (0..k_in * out).map(|i| ((i as f32) * 0.003).cos()).collect();
        let (mut p, mut q) = (vec![0f32; out], vec![0f32; out]);
        matmul(&mut p, &x, &wf, k_in, out);
        matmul_serial(&mut q, &x, &wf, k_in, out);
        same_bits("matmul", &p, &q);

        let wb: Vec<u16> = (0..k_in * out).map(|i| 0x3F00u16.wrapping_add(i as u16 % 97)).collect();
        let (mut p, mut q) = (vec![0f32; out], vec![0f32; out]);
        matmul_bf16(&mut p, &x, &wb, k_in, out);
        matmul_bf16_serial(&mut q, &x, &wb, k_in, out);
        same_bits("matmul_bf16", &p, &q);

        let g = MXFP4_GROUP;
        let packed: Vec<u8> = (0..k_in / 2 * out).map(|i| (i * 7 % 251) as u8).collect();
        let ngrp = k_in.div_ceil(g);
        let sc: Vec<u8> = (0..ngrp * out).map(|i| (120 + i % 13) as u8).collect();
        let (mut p, mut q) = (vec![0f32; out], vec![0f32; out]);
        matmul_mxfp4(&mut p, &x, &packed, &sc, k_in, out, g);
        matmul_mxfp4_serial(&mut q, &x, &packed, &sc, k_in, out, g);
        same_bits("matmul_mxfp4", &p, &q);
    }

    // The two new SIMD paths. docs/RUST_PORT.md:273 -- transliterate, do not improve:
    // the AVX2 path must be BIT-IDENTICAL to the scalar one, not merely close, because
    // every downstream comparison is an equality and not a tolerance.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_avx2_paths_are_bit_identical_to_the_scalar_ones() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("no avx2 on this host; nothing to compare");
            return;
        }
        let (k_in, out, block) = (512usize, 192usize, 128usize);
        let x = x_of(k_in);

        let w: Vec<u8> = (0..k_in * out).map(|i| ((i * 7) % 126) as u8).collect();
        let sb = k_in.div_ceil(block) * out.div_ceil(block);
        let sc: Vec<u8> = (0..sb).map(|i| (120 + i % 13) as u8).collect();
        let (mut a, mut b) = (vec![0f32; out], vec![0f32; out]);
        unsafe { matmul_fp8_block_avx2(&mut a, &x, &w, &sc, k_in, out, block) };
        matmul_fp8_block_scalar(&mut b, &x, &w, &sc, k_in, out, block);
        same_bits("matmul_fp8_block avx2 vs scalar", &a, &b);

        let g = MXFP4_GROUP;
        let packed: Vec<u8> = (0..k_in / 2 * out).map(|i| (i * 7 % 251) as u8).collect();
        let ngrp = k_in.div_ceil(g);
        let msc: Vec<u8> = (0..ngrp * out).map(|i| (120 + i % 13) as u8).collect();
        let (mut a, mut b) = (vec![0f32; out], vec![0f32; out]);
        unsafe { matmul_mxfp4_avx2(&mut a, &x, &packed, &msc, k_in, out, g) };
        matmul_mxfp4_scalar(&mut b, &x, &packed, &msc, k_in, out, g);
        same_bits("matmul_mxfp4 avx2 vs scalar", &a, &b);
    }

    // A NaN scale (255) must make its whole group contribute nothing, on both paths.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn a_nan_scale_is_skipped_identically_by_both_mxfp4_paths() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let (k_in, out, g) = (256usize, 128usize, MXFP4_GROUP);
        let x = x_of(k_in);
        let packed: Vec<u8> = (0..k_in / 2 * out).map(|i| (i * 7 % 251) as u8).collect();
        let ngrp = k_in.div_ceil(g);
        let mut sc: Vec<u8> = (0..ngrp * out).map(|i| (120 + i % 13) as u8).collect();
        for r in 0..out {
            sc[r * ngrp + 1] = 255;
        }
        let (mut a, mut b) = (vec![0f32; out], vec![0f32; out]);
        unsafe { matmul_mxfp4_avx2(&mut a, &x, &packed, &sc, k_in, out, g) };
        matmul_mxfp4_scalar(&mut b, &x, &packed, &sc, k_in, out, g);
        same_bits("mxfp4 nan-scale", &a, &b);
        assert!(a.iter().all(|v| v.is_finite()), "a NaN scale must not poison the row");
    }

    #[test]
    fn router_is_independent_of_thread_count() {
        let (hidden, n_experts, topk) = (64usize, 256usize, 8usize);
        let x = x_of(hidden);
        let w: Vec<f32> = (0..hidden * n_experts).map(|i| ((i as f32) * 0.011).sin()).collect();
        let bias: Vec<f32> = (0..n_experts).map(|e| ((e as f32) * 0.3).cos() * 0.05).collect();
        let run = || {
            let mut idx = vec![0i32; topk];
            let mut wt = vec![0f32; topk];
            router(&mut idx, &mut wt, &x, &w, Some(&bias), hidden, n_experts, topk, true, 1.0);
            (idx, wt)
        };
        let (ia, wa) = in_pool(1, run);
        let (ib, wb) = in_pool(8, run);
        assert_eq!(ia, ib, "router selected different experts under a different thread count");
        same_bits("router weights", &wa, &wb);
    }
}

pub const PREFILL_CHUNK: usize = 64;

pub fn route_chunk(
    x: &[f32],
    gate: &[f32],
    bias: Option<&[f32]>,
    d: &MoeDims,
    t0: usize,
    n: usize,
) -> (Vec<i32>, Vec<f32>, Vec<usize>) {
    let (mut idx, mut wt) = (vec![0i32; n * d.topk], vec![0f32; n * d.topk]);
    for t in 0..n {
        router(
            &mut idx[t * d.topk..][..d.topk],
            &mut wt[t * d.topk..][..d.topk],
            &x[(t0 + t) * d.hidden..][..d.hidden],
            gate,
            bias,
            d.hidden,
            d.n_experts,
            d.topk,
            d.renorm,
            d.routed_scale,
        );
    }
    let mut seen = vec![false; d.n_experts];
    let mut uniq = Vec::with_capacity(n * d.topk);
    for &e in &idx {
        let e = e as usize;
        if e < d.n_experts && !seen[e] {
            seen[e] = true;
            uniq.push(e);
        }
    }
    (idx, wt, uniq)
}

#[cfg(test)]
mod prefill {
    use super::*;

    fn dims(n_experts: usize, topk: usize) -> MoeDims {
        MoeDims {
            hidden: 32,
            latent: 32,
            moe_inter: 32,
            n_experts,
            topk,
            n_shared: 1,
            routed_scale: 1.0,
            renorm: true,
            latent_norm: false,
            rms_eps: 1e-6,
            situ_b1: 4.0,
            situ_b2: 25.0,
        }
    }

    fn fixture(n: usize, d: &MoeDims) -> (Vec<f32>, Vec<f32>) {
        let x = (0..n * d.hidden).map(|i| ((i as f32) * 0.07).sin()).collect();
        let g = (0..d.n_experts * d.hidden).map(|i| ((i as f32) * 0.013).cos()).collect();
        (x, g)
    }

    #[test]
    fn the_union_is_deduplicated_and_covers_every_selection() {
        let d = dims(64, 6);
        let n = PREFILL_CHUNK;
        let (x, g) = fixture(n, &d);
        let (idx, _, uniq) = route_chunk(&x, &g, None, &d, 0, n);
        let mut sorted = uniq.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), uniq.len(), "an expert was fetched twice");
        for &e in &idx {
            assert!(uniq.contains(&(e as usize)), "expert {e} routed to but never fetched");
        }
    }

    #[test]
    fn the_union_is_smaller_than_the_request_count() {
        let d = dims(64, 6);
        let n = PREFILL_CHUNK;
        let (x, g) = fixture(n, &d);
        let (idx, _, uniq) = route_chunk(&x, &g, None, &d, 0, n);
        assert!(uniq.len() < idx.len(), "{} unique of {} requests", uniq.len(), idx.len());
    }

    #[test]
    fn routing_a_chunk_matches_routing_each_token_alone() {
        let d = dims(64, 6);
        let n = 8;
        let (x, g) = fixture(n, &d);
        let (idx, wt, _) = route_chunk(&x, &g, None, &d, 0, n);
        for t in 0..n {
            let (mut i1, mut w1) = (vec![0i32; d.topk], vec![0f32; d.topk]);
            router(&mut i1, &mut w1, &x[t * d.hidden..][..d.hidden], &g, None,
                   d.hidden, d.n_experts, d.topk, d.renorm, d.routed_scale);
            assert_eq!(&idx[t * d.topk..][..d.topk], &i1[..]);
            for (a, b) in wt[t * d.topk..][..d.topk].iter().zip(&w1) {
                assert_eq!(a.to_bits(), b.to_bits(), "token {t} weight moved");
            }
        }
    }

    #[test]
    fn an_offset_chunk_routes_its_own_tokens() {
        let d = dims(32, 4);
        let (x, g) = fixture(16, &d);
        let (a, _, _) = route_chunk(&x, &g, None, &d, 8, 8);
        let (b, _, _) = route_chunk(&x[8 * d.hidden..], &g, None, &d, 0, 8);
        assert_eq!(a, b, "t0 must offset into x, not be ignored");
    }
}

#[derive(Clone, Copy)]
pub struct Eq<'a> {
    pub p1: &'a [u8],
    pub s1: &'a [u8],
    pub p3: &'a [u8],
    pub s3: &'a [u8],
    pub p2: &'a [u8],
    pub s2: &'a [u8],
}

pub fn moe_packed(
    out: &mut [f32],
    x: &[f32],
    w: &MoeW,
    d: &MoeDims,
    t_len: usize,
    idx: &[i32],
    wt: &[f32],
    experts: &[Eq],
) {
    let (e, l, i_n) = (d.hidden, d.latent, d.moe_inter);
    let si = i_n * d.n_shared;
    let g = MXFP4_GROUP;

    let mut z = vec![0f32; l];
    let mut acc_l = vec![0f32; l];
    let mut gu = vec![0f32; 2 * i_n];
    let mut act = vec![0f32; i_n];
    let mut edn = vec![0f32; l];
    let mut sgu = vec![0f32; 2 * si];
    let mut sact = vec![0f32; si];
    let mut sdn = vec![0f32; e];

    for t in 0..t_len {
        let xt = &x[t * e..][..e];
        let ot = &mut out[t * e..][..e];

        mmw(&mut z, xt, w.down, e, l);
        acc_l.fill(0.0);
        for j in 0..d.topk {
            let q = &experts[t * d.topk + j];
            matmul_mxfp4(&mut gu[..i_n], &z, q.p1, q.s1, l, i_n, g);
            matmul_mxfp4(&mut gu[i_n..], &z, q.p3, q.s3, l, i_n, g);
            situ_glu(&mut act, &gu, i_n, d.situ_b1, d.situ_b2);
            matmul_mxfp4(&mut edn, &act, q.p2, q.s2, i_n, l, g);
            let wj = wt[t * d.topk + j];
            for i in 0..l {
                acc_l[i] += wj * edn[i];
            }
        }
        let _ = idx;

        if d.latent_norm {
            let a = acc_l.clone();
            rmsnorm(&mut acc_l, &a, w.latent_norm, l, d.rms_eps);
        }
        mmw(ot, &acc_l, w.up, l, e);

        mmw(&mut sgu[..si], xt, w.sh1, e, si);
        mmw(&mut sgu[si..], xt, w.sh3, e, si);
        situ_glu(&mut sact, &sgu, si, d.situ_b1, d.situ_b2);
        mmw(&mut sdn, &sact, w.sh2, si, e);
        for i in 0..e {
            ot[i] += sdn[i];
        }
    }
}

#[cfg(test)]
mod arch_kernels {
    use super::*;

    // MiniMax-M3's swigluoai differs from DeepSeek's clamped SwiGLU in two places, and
    // dropping either leaves a bounded, plausible activation.
    #[test]
    fn swigluoai_scales_the_gate_sigmoid_by_alpha() {
        let (n, a) = (4usize, 1.702f32);
        let x: Vec<f32> = vec![0.7, -0.4, 1.3, -1.1, 0.2, 0.9, -0.5, 1.4];
        let mut oai = vec![0f32; n];
        let mut plain = vec![0f32; n];
        glu(&mut oai, &x, n, Glu::SwigluOai { alpha: a, limit: 7.0 });
        glu(&mut plain, &x, n, Glu::SwigluOai { alpha: 1.0, limit: 7.0 });
        assert!(oai.iter().zip(&plain).any(|(p, q)| (p - q).abs() > 1e-6), "alpha did nothing");
        for i in 0..n {
            let g = x[i];
            let want = (g * sigmoidf(a * g)) * (x[n + i] + 1.0);
            assert!((oai[i] - want).abs() < 1e-6, "{i}: {} vs {want}", oai[i]);
        }
    }

    #[test]
    fn swigluoai_carries_the_up_bias_of_one() {
        let n = 2usize;
        // up = -1 exactly: with the +1 bias the product is zero, without it it is not.
        let x = vec![1.0f32, 1.0, -1.0, -1.0];
        let mut y = vec![0f32; n];
        glu(&mut y, &x, n, Glu::SwigluOai { alpha: 1.702, limit: 7.0 });
        for v in &y {
            assert!(v.abs() < 1e-6, "up + 1 must vanish at up = -1, got {v}");
        }
    }

    #[test]
    fn deepseek_and_oai_swiglu_are_not_the_same_function() {
        let n = 4usize;
        let x: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.6).sin()).collect();
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        glu(&mut a, &x, n, Glu::SwigluClamped { limit: 7.0 });
        glu(&mut b, &x, n, Glu::SwigluOai { alpha: 1.702, limit: 7.0 });
        assert!(a.iter().zip(&b).any(|(p, q)| (p - q).abs() > 1e-6));
    }

    // Gemma's gain is an OFFSET: the multiplier is 1 + w, not w. On a checkpoint whose
    // weights sit near zero the plain form scales every channel to near zero, which is
    // stable and wrong rather than obviously broken.
    #[test]
    fn gemma_norm_treats_the_gain_as_an_offset() {
        let n = 6usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.4).collect();
        let w = vec![0.0f32; n];
        let mut g = vec![0f32; n];
        let mut p = vec![0f32; n];
        rmsnorm_gemma(&mut g, &x, &w, n, 1e-6, Acc::F32);
        rmsnorm_acc(&mut p, &x, &w, n, 1e-6, Acc::F32);
        assert!(p.iter().all(|v| v.abs() < 1e-9), "a zero gain zeroes the plain form");
        assert!(g.iter().any(|v| v.abs() > 0.1), "1 + 0 = 1, so gemma is the identity gain");
        let ones = vec![1.0f32; n];
        let mut q = vec![0f32; n];
        rmsnorm_acc(&mut q, &x, &ones, n, 1e-6, Acc::F32);
        for i in 0..n {
            assert!((g[i] - q[i]).abs() < 1e-6, "gemma(w=0) must equal plain(w=1)");
        }
    }

    // The seam docs/MULTI_MODEL.md called "still wrong". The two widths must actually
    // differ, or moving it into the descriptor bought nothing.
    #[test]
    fn the_two_rmsnorm_accumulators_give_different_last_bits() {
        let n = 4096usize;
        // Many small terms: the f32 accumulator loses low-order bits the f64 one keeps.
        let x: Vec<f32> = (0..n).map(|i| 1e-3 + (i as f32 * 1e-7)).collect();
        let w = vec![1.0f32; n];
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        rmsnorm_acc(&mut a, &x, &w, n, 1e-6, Acc::F64);
        rmsnorm_acc(&mut b, &x, &w, n, 1e-6, Acc::F32);
        assert!(
            a.iter().zip(&b).any(|(p, q)| p.to_bits() != q.to_bits()),
            "if these agreed bitwise the accumulation width would not be worth a field"
        );
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-4, "and they must still be close");
        }
    }

    #[test]
    fn rmsnorm_defaults_to_the_k3_width() {
        let n = 512usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let w = vec![0.7f32; n];
        let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
        rmsnorm(&mut a, &x, &w, n, 1e-5);
        rmsnorm_acc(&mut b, &x, &w, n, 1e-5, Acc::F64);
        assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
}
