//! Argmax-margin-certified activation sparsity for SwiGLU feed-forward blocks.
//!
//! Contextual sparsity (skip the FFN neurons that barely fire) normally trades a little
//! quality for speed. SwiGLU has no hard zeros, so any skip is lossy — UNLESS you can prove
//! the skip cannot change what the model emits. That proof is what this module provides.
//!
//! The FFN is `y = down( silu(gate·x) ⊙ (up·x) )`. Write `aᵢ = silu(gateᵢ)·upᵢ` for neuron
//! `i`; its exact contribution to the output is `aᵢ · down[:,i]`. Skipping a set `S` perturbs
//! the FFN output by
//! ```text
//!   ‖Σ_{i∈S} aᵢ·down[:,i]‖₂  ≤  Σ_{i∈S} |aᵢ|·‖down[:,i]‖₂   (triangle inequality)
//! ```
//! Call `cᵢ = |aᵢ|·‖down[:,i]‖₂` the neuron's contribution bound (the column norms are a
//! one-time offline precompute). The **certificate** is: propagate that output perturbation
//! bound to the logits — for a downstream Lipschitz factor `L` (operator-norm bound of
//! everything between this FFN and the logits, incl. the head), `‖Δlogits‖∞ ≤ L·Σcᵢ`. If
//! `L·Σcᵢ < margin/2`, where `margin` is the current top-1 minus top-2 logit gap, then **no
//! logit can overtake the top one**, so the greedy argmax — the emitted token — is bitwise
//! identical to the dense model. The skip is then token-exact lossless, not approximate.
//!
//! This module proves the two load-bearing facts and nothing more: (1) the certified bound is
//! SOUND — the real perturbation never exceeds it — and (2) when the bound is under the
//! margin, the argmax is preserved. A wrong mask still produces fluent text, so soundness is
//! asserted, never assumed. Wiring (precompute column norms, choose `L`, apply in the FFN
//! forward) is the integration step; see `dense-speedup-levers`.

#[inline]
fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

/// Per-neuron SwiGLU activation `aᵢ = silu(gateᵢ)·upᵢ`.
pub fn activations(gate: &[f32], up: &[f32]) -> Vec<f32> {
    debug_assert_eq!(gate.len(), up.len());
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

/// L2 norm of each column of `down` (`[hidden, inter]`, row-major). Column `i` is neuron
/// `i`'s output direction, strided by `inter`. Precompute once at load — it does not depend
/// on the activations.
pub fn down_col_norms(down: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
    let mut n = vec![0f64; inter];
    for h in 0..hidden {
        let row = &down[h * inter..][..inter];
        for i in 0..inter {
            n[i] += row[i] as f64 * row[i] as f64;
        }
    }
    n.iter().map(|&v| v.sqrt() as f32).collect()
}

/// Column norms of a k-quant `down` weight (`[hidden, inter]`, row-major), dequantised row by
/// row so no full-tensor f32 buffer is materialised. `block_bytes` and `dq` select the quant
/// (the `down` weight is Q6_K on some layers and Q4_K on others in a `_M` build, so the caller
/// must pass the right pair). Computed ONCE per layer at first use and cached — it is a
/// property of the frozen weight, not the activations.
pub fn down_col_norms_q(
    bytes: &[u8],
    hidden: usize,
    inter: usize,
    block_bytes: usize,
    dq: fn(&mut [f32], &[u8], usize),
) -> Vec<f32> {
    const QK_K: usize = 256;
    let nb = inter / QK_K;
    let stride = nb * block_bytes;
    let mut sq = vec![0f64; inter];
    let mut row = vec![0f32; inter];
    for h in 0..hidden {
        dq(&mut row, &bytes[h * stride..][..stride], nb);
        for i in 0..inter {
            sq[i] += row[i] as f64 * row[i] as f64;
        }
    }
    sq.iter().map(|&v| v.sqrt() as f32).collect()
}

/// Per-neuron contribution bound `cᵢ = |aᵢ|·‖down[:,i]‖₂`.
pub fn contribs(a: &[f32], col_norm: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), col_norm.len());
    a.iter().zip(col_norm).map(|(&ai, &ni)| ai.abs() * ni).collect()
}

/// Choose which neurons to KEEP so the certified output-perturbation bound (`Σ` of skipped
/// `cᵢ`) stays `≤ budget`. Greedy and optimal for this objective: skip the smallest-`cᵢ`
/// neurons first, since that removes the most neurons per unit of spent budget.
///
/// Returns `(keep, bound)` where `keep[i]` is false for skipped neurons and `bound` is the
/// certified L2 upper bound on `‖dense_out − sparse_out‖` (the sum of skipped `cᵢ`).
pub fn certified_keep_mask(contribs: &[f32], budget: f32) -> (Vec<bool>, f32) {
    let n = contribs.len();
    let mut order: Vec<usize> = (0..n).collect();
    // ascending by contribution; skip from the smallest up.
    order.sort_by(|&a, &b| contribs[a].total_cmp(&contribs[b]));
    let mut keep = vec![true; n];
    let mut spent = 0f64;
    for &i in &order {
        let c = contribs[i] as f64;
        if spent + c <= budget as f64 {
            spent += c;
            keep[i] = false;
        } else {
            break; // sorted ascending, so nothing further fits either
        }
    }
    (keep, spent as f32)
}

/// The certified-sparse down projection: `y[h] = Σ_{i: keep[i]} aᵢ · down[h,i]`. Skipped
/// neurons are never touched — in a streaming engine that also means their `down` column is
/// never decoded or read, which is where the speed comes from.
pub fn sparse_down(a: &[f32], down: &[f32], keep: &[bool], hidden: usize, inter: usize) -> Vec<f32> {
    let mut y = vec![0f32; hidden];
    for h in 0..hidden {
        let row = &down[h * inter..][..inter];
        let mut acc = 0f64;
        for i in 0..inter {
            if keep[i] {
                acc += a[i] as f64 * row[i] as f64;
            }
        }
        y[h] = acc as f32;
    }
    y
}

/// The top-1 minus top-2 gap of a logit vector — the margin the certificate must beat
/// (halved). Returns `(argmax, margin)`.
pub fn logit_margin(logits: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            second = best.1;
            best = (i, v);
        } else if v > second {
            second = v;
        }
    }
    (best.0, best.1 - second)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 40) as i64 as f32 / (1i64 << 24) as f32 - 0.5
    }

    fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| {
                let row = &w[r * cols..][..cols];
                row.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32
            })
            .collect()
    }

    /// SOUNDNESS: the real L2 perturbation from skipping neurons must NEVER exceed the
    /// certified bound. This is the whole guarantee — if it can be violated, the certificate
    /// is worthless and the sparsity is silently lossy.
    #[test]
    fn the_certified_bound_is_never_violated() {
        let (hidden, inter) = (48usize, 96usize);
        let mut s = 7u64;
        for _ in 0..40 {
            let gate: Vec<f32> = (0..inter).map(|_| lcg(&mut s) * 3.0).collect();
            let up: Vec<f32> = (0..inter).map(|_| lcg(&mut s) * 2.0).collect();
            let down: Vec<f32> = (0..hidden * inter).map(|_| lcg(&mut s)).collect();

            let a = activations(&gate, &up);
            let norms = down_col_norms(&down, hidden, inter);
            let c = contribs(&a, &norms);

            let dense = sparse_down(&a, &down, &vec![true; inter], hidden, inter);
            for &budget in &[0.0f32, 0.5, 2.0, 10.0, 1e9] {
                let (keep, bound) = certified_keep_mask(&c, budget);
                let sparse = sparse_down(&a, &down, &keep, hidden, inter);
                let err = (dense.iter().zip(&sparse).map(|(&d, &sp)| ((d - sp) as f64).powi(2)).sum::<f64>()).sqrt() as f32;
                assert!(
                    err <= bound + 1e-3,
                    "budget {budget}: real error {err} exceeds certified bound {bound}"
                );
                assert!(bound <= budget + 1e-3, "spent bound {bound} exceeds budget {budget}");
            }
        }
    }

    /// ARGMAX PRESERVATION: when the certificate holds (propagated bound < margin/2), the
    /// greedy token is IDENTICAL to dense. This is the end-to-end lossless claim, on a real
    /// FFN -> head -> logits chain.
    #[test]
    fn a_valid_certificate_preserves_the_argmax() {
        let (hidden, inter, vocab) = (64usize, 128usize, 40usize);
        let mut s = 99u64;
        let mut certified = 0;
        for _ in 0..60 {
            let gate: Vec<f32> = (0..inter).map(|_| lcg(&mut s) * 2.5).collect();
            let up: Vec<f32> = (0..inter).map(|_| lcg(&mut s) * 2.0).collect();
            let down: Vec<f32> = (0..hidden * inter).map(|_| lcg(&mut s) * 0.5).collect();
            let head: Vec<f32> = (0..vocab * hidden).map(|_| lcg(&mut s) * 0.5).collect();

            let a = activations(&gate, &up);
            let norms = down_col_norms(&down, hidden, inter);
            let c = contribs(&a, &norms);

            let dense_h = sparse_down(&a, &down, &vec![true; inter], hidden, inter);
            let dense_logits = matvec(&head, &dense_h, vocab, hidden);
            let (dense_arg, margin) = logit_margin(&dense_logits);

            // Downstream Lipschitz factor from FFN output to logits: here the head alone,
            // so ‖Δlogits‖∞ ≤ (max row L2 of head) · ‖Δh‖₂ (Cauchy–Schwarz).
            let l = (0..vocab)
                .map(|r| head[r * hidden..][..hidden].iter().map(|&w| (w as f64).powi(2)).sum::<f64>().sqrt())
                .fold(0f64, f64::max) as f32;
            // Safe FFN-output budget so the logit perturbation stays under margin/2.
            let budget = (margin / 2.0) / l.max(1e-6);

            let (keep, _bound) = certified_keep_mask(&c, budget);
            let n_skip = keep.iter().filter(|&&k| !k).count();
            let sparse_h = sparse_down(&a, &down, &keep, hidden, inter);
            let sparse_logits = matvec(&head, &sparse_h, vocab, hidden);
            let (sparse_arg, _) = logit_margin(&sparse_logits);

            assert_eq!(
                dense_arg, sparse_arg,
                "certified skip of {n_skip} neurons changed the argmax ({dense_arg} -> {sparse_arg})"
            );
            if n_skip > 0 {
                certified += 1;
            }
        }
        assert!(certified > 0, "the certificate never fired — test proves nothing");
    }

    /// TEETH: an OVER-budget skip (ignoring the certificate) must sometimes flip the argmax,
    /// proving the certificate is what protects it, not luck. Uses a deliberately huge budget.
    #[test]
    fn an_uncertified_skip_can_flip_the_argmax() {
        let (hidden, inter, vocab) = (64usize, 128usize, 40usize);
        let mut s = 12345u64;
        let mut flips = 0;
        for _ in 0..60 {
            let gate: Vec<f32> = (0..inter).map(|_| lcg(&mut s) * 2.5).collect();
            let up: Vec<f32> = (0..inter).map(|_| lcg(&mut s) * 2.0).collect();
            let down: Vec<f32> = (0..hidden * inter).map(|_| lcg(&mut s) * 0.5).collect();
            let head: Vec<f32> = (0..vocab * hidden).map(|_| lcg(&mut s) * 0.5).collect();
            let a = activations(&gate, &up);
            let norms = down_col_norms(&down, hidden, inter);
            let c = contribs(&a, &norms);
            let dense_h = sparse_down(&a, &down, &vec![true; inter], hidden, inter);
            let (dense_arg, _) = logit_margin(&matvec(&head, &dense_h, vocab, hidden));
            // Skip aggressively: budget = 100x the total contribution -> skip everything.
            let (keep, _) = certified_keep_mask(&c, c.iter().sum::<f32>() * 100.0);
            let sparse_h = sparse_down(&a, &down, &keep, hidden, inter);
            let (sparse_arg, _) = logit_margin(&matvec(&head, &sparse_h, vocab, hidden));
            if dense_arg != sparse_arg {
                flips += 1;
            }
        }
        assert!(flips > 0, "uncertified skipping never flipped the argmax — teeth check is vacuous");
    }

    /// The greedy mask must skip the SMALLEST contributors and be monotone in budget: more
    /// budget skips a superset. A wrong ordering would skip high-impact neurons and blow the
    /// bound.
    #[test]
    fn larger_budget_skips_a_superset() {
        let c = vec![5.0f32, 1.0, 3.0, 0.5, 2.0];
        let (k_small, _) = certified_keep_mask(&c, 1.6); // fits 0.5 + 1.0
        let (k_big, _) = certified_keep_mask(&c, 3.6); // fits 0.5 + 1.0 + 2.0
        assert_eq!(k_small, vec![true, false, true, false, true]);
        assert_eq!(k_big, vec![true, false, true, false, false]);
        // every neuron skipped by the smaller budget is skipped by the larger.
        for i in 0..c.len() {
            if !k_small[i] {
                assert!(!k_big[i], "neuron {i} un-skipped by a larger budget");
            }
        }
    }
}
