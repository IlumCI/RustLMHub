// SPDX-License-Identifier: Apache-2.0
//
// Kernels the third and fourth architectures need and the first two did not: grouped-query
// attention, partial and interleaved rope, and block-sparse row selection.

use crate::ops::{rmsnorm_acc, Acc};

/// How the rotation pairs the head dimension.
///
/// `Adjacent` treats the last axis as complex, pairing (0,1), (2,3), ... `Halves` pairs i
/// with i + dim/2, the Llama convention. Both are valid rotations and both produce
/// working attention, so a checkpoint read with the wrong one degrades rather than fails.
/// GLM-5.1 sets `rope_interleave: true`, meaning `Adjacent`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pairing {
    Adjacent,
    Halves,
}

pub struct Rope {
    pub cs: Vec<f32>,
    pub rot: usize,
    pub pairing: Pairing,
}

/// `rot` is how many of `head_dim` leading elements rotate. MiniMax-M3 sets
/// `partial_rotary_factor: 0.5` with `rotary_dim: 64` over a 128-wide head, so half of
/// every head is left unrotated; rotating all of it is a different position encoding that
/// still trains-looking output.
pub fn rope(head_dim: usize, rot: usize, seqlen: usize, theta: f32, pairing: Pairing) -> Rope {
    assert!(rot <= head_dim && rot % 2 == 0, "rotary width must be even and fit the head");
    let half = rot / 2;
    let mut cs = vec![0f32; seqlen * half * 2];
    for p in 0..seqlen {
        for i in 0..half {
            let f = 1.0f64 / (theta as f64).powf(2.0 * i as f64 / rot as f64);
            let a = p as f64 * f;
            cs[(p * half + i) * 2] = a.cos() as f32;
            cs[(p * half + i) * 2 + 1] = a.sin() as f32;
        }
    }
    Rope { cs, rot, pairing }
}

pub fn apply_rope(x: &mut [f32], r: &Rope, pos: usize, inverse: bool) {
    let half = r.rot / 2;
    // Name the limit. Indexing past the table is a bounds panic from inside a hot loop
    // with no clue what went wrong; a conversation simply grew past the length the table
    // was built for, and the caller needs to hear that rather than "index out of bounds".
    assert!(
        half == 0 || (pos + 1) * half * 2 <= r.cs.len(),
        "rope table holds {} positions but position {pos} was asked for -- size it from \
         the model's context length, not a constant",
        r.cs.len() / half.max(1) / 2
    );
    for i in 0..half {
        let c = r.cs[(pos * half + i) * 2];
        let s = if inverse { -r.cs[(pos * half + i) * 2 + 1] } else { r.cs[(pos * half + i) * 2 + 1] };
        let (a, b) = match r.pairing {
            Pairing::Adjacent => (2 * i, 2 * i + 1),
            Pairing::Halves => (i, i + half),
        };
        let (xa, xb) = (x[a], x[b]);
        x[a] = xa * c - xb * s;
        x[b] = xa * s + xb * c;
    }
}

pub struct GqaDims {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub acc: Acc,
    /// Per-head RMSNorm on q and k before scoring (`use_qk_norm`, `qk_norm_type per_head`).
    pub qk_norm: bool,
}

/// Grouped-query attention over a causal prefix.
///
/// `q` is [T][n_heads][head_dim]; `k` and `v` are [T][n_kv_heads][head_dim]. Head h reads
/// kv head `h / (n_heads / n_kv_heads)`. Getting that division backwards -- `h % n_kv` --
/// pairs every head with the wrong group while keeping every shape identical, which is
/// the failure this signature exists to make hard.
///
/// `keep[t]` optionally restricts which source positions token t may attend to; `None`
/// means the full causal prefix. Entries must be < = t and negative slots are skipped.
#[allow(clippy::too_many_arguments)]
fn gqa_core(
    one_row: bool,
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: &GqaDims,
    t_len: usize,
    qn: Option<&[f32]>,
    kn: Option<&[f32]>,
    keep: Option<&[Vec<i64>]>,
) {
    let (h_n, kv_n, hd) = (d.n_heads, d.n_kv_heads, d.head_dim);
    assert!(kv_n > 0 && h_n % kv_n == 0, "n_heads must be a multiple of n_kv_heads");
    let group = h_n / kv_n;
    let scale = 1.0f32 / (hd as f32).sqrt();

    let mut qh = vec![0f32; hd];
    let mut kh = vec![0f32; hd];
    // Single-row mode visits ONLY the final position; `t` remains its true position so
    // causality still admits the whole prefix. Skipping this is what made decode O(len^2).
    let first_t = if one_row { t_len - 1 } else { 0 };
    for t in first_t..t_len {
        // Where this step's query lives, and where its output goes. Both coincide with `t`
        // for a full sweep and are row 0 for `gqa_last`.
        let q_row = if one_row { 0 } else { t };
        let out_row = q_row;
        for h in 0..h_n {
            let kvh = h / group;
            let qs = &q[(q_row * h_n + h) * hd..][..hd];
            if d.qk_norm {
                rmsnorm_acc(&mut qh, qs, qn.unwrap_or(&vec![1.0; hd]), hd, d.eps, d.acc);
            } else {
                qh.copy_from_slice(qs);
            }

            let src: Vec<usize> = match keep {
                Some(rows) => rows[t].iter().filter(|&&s| s >= 0).map(|&s| s as usize).collect(),
                None => (0..=t).collect(),
            };
            if src.is_empty() {
                out[(out_row * h_n + h) * hd..][..hd].fill(0.0);
                continue;
            }

            let mut sc = Vec::with_capacity(src.len());
            let mut mx = f32::NEG_INFINITY;
            for &s in &src {
                let ks = &k[(s * kv_n + kvh) * hd..][..hd];
                if d.qk_norm {
                    rmsnorm_acc(&mut kh, ks, kn.unwrap_or(&vec![1.0; hd]), hd, d.eps, d.acc);
                } else {
                    kh.copy_from_slice(ks);
                }
                let mut dot = 0.0f32;
                for i in 0..hd {
                    dot += qh[i] * kh[i];
                }
                let z = dot * scale;
                if z > mx {
                    mx = z;
                }
                sc.push(z);
            }
            let mut den = 0.0f32;
            for z in sc.iter_mut() {
                *z = (*z - mx).exp();
                den += *z;
            }
            let o = &mut out[(out_row * h_n + h) * hd..][..hd];
            o.fill(0.0);
            for (j, &s) in src.iter().enumerate() {
                let w = sc[j] / den;
                let vs = &v[(s * kv_n + kvh) * hd..][..hd];
                for i in 0..hd {
                    o[i] += w * vs[i];
                }
            }
        }
    }
}

/// Grouped-query attention over a causal prefix, every position scored.
#[allow(clippy::too_many_arguments)]
pub fn gqa(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: &GqaDims,
    t_len: usize,
    qn: Option<&[f32]>,
    kn: Option<&[f32]>,
    keep: Option<&[Vec<i64>]>,
) {
    gqa_core(false, out, q, k, v, d, t_len, qn, kn, keep);
}

/// Score ONE query -- the newest token -- against the whole `t_len`-long prefix.
///
/// Decode needs exactly the last row of the attention matrix. Getting it by running the
/// full sweep and discarding everything else makes each token O(len^2) where it is O(len):
/// at 4k context that is four thousand times the work, all of it thrown away. `q` holds a
/// single row and `out` receives a single row; `k` and `v` are the full prefix.
#[allow(clippy::too_many_arguments)]
pub fn gqa_last(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: &GqaDims,
    t_len: usize,
    qn: Option<&[f32]>,
    kn: Option<&[f32]>,
) {
    assert!(t_len > 0, "there is no last row of an empty prefix");
    // One iteration, at position t_len-1 so causality still admits the whole prefix.
    gqa_core(true, out, q, k, v, d, t_len, qn, kn, None);
}

/// Which source positions token `t` keeps under MiniMax-M3's block-sparse attention:
/// the first `init` blocks, the last `local` blocks, and the `topk` highest-scoring of
/// the rest. Blocks are `bsz` positions wide.
///
/// A block is eligible for token t only when it starts at or before t; including the
/// block t sits in is correct (its later positions are masked by causality inside `gqa`),
/// but including any block that starts after t lets a token attend to its own future.
pub fn sparse_rows(
    t_len: usize,
    bsz: usize,
    topk: usize,
    init: usize,
    local: usize,
    score: impl Fn(usize, usize) -> f32,
) -> Vec<Vec<i64>> {
    let nblk = t_len.div_ceil(bsz);
    let mut rows = Vec::with_capacity(t_len);
    for t in 0..t_len {
        let last = t / bsz;
        let mut keep: Vec<usize> = Vec::new();
        for b in 0..=last.min(nblk.saturating_sub(1)) {
            if b < init || b + local > last {
                keep.push(b);
            }
        }
        let mut rest: Vec<usize> =
            (0..=last).filter(|b| !keep.contains(b)).collect();
        rest.sort_by(|&a, &b| {
            score(t, b).partial_cmp(&score(t, a)).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        keep.extend(rest.into_iter().take(topk));
        keep.sort_unstable();
        let mut row: Vec<i64> = Vec::new();
        for b in keep {
            for p in b * bsz..((b + 1) * bsz).min(t_len) {
                if p <= t {
                    row.push(p as i64);
                }
            }
        }
        rows.push(row);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `gqa_last` must equal `gqa`'s final row exactly. Two implementations of attention
    /// is one too many, so this pins them together -- and it is an EQUALITY, not a
    /// tolerance, because they run the same code over the same reduction order.
    #[test]
    fn gqa_last_reproduces_the_final_row_of_the_full_sweep() {
        let d = GqaDims { n_heads: 4, n_kv_heads: 2, head_dim: 8, eps: 1e-6,
                          acc: crate::ops::Acc::F64, qk_norm: false };
        let t_len = 7;
        let f = |i: usize, m: usize| ((i * 37 + m * 11) % 23) as f32 * 0.05 - 0.5;
        let q: Vec<f32> = (0..t_len * d.n_heads * d.head_dim).map(|i| f(i, 1)).collect();
        let k: Vec<f32> = (0..t_len * d.n_kv_heads * d.head_dim).map(|i| f(i, 2)).collect();
        let v: Vec<f32> = (0..t_len * d.n_kv_heads * d.head_dim).map(|i| f(i, 3)).collect();

        let mut full = vec![0f32; t_len * d.n_heads * d.head_dim];
        gqa(&mut full, &q, &k, &v, &d, t_len, None, None, None);

        let row = d.n_heads * d.head_dim;
        let mut one = vec![0f32; row];
        gqa_last(&mut one, &q[(t_len - 1) * row..][..row], &k, &v, &d, t_len, None, None);
        assert_eq!(one, full[(t_len - 1) * row..].to_vec(), "single-row must be bit-identical");
        // And it must NOT be the same as an earlier row, or the test would pass vacuously
        // on an implementation that ignored the prefix.
        assert_ne!(one, full[..row].to_vec());
    }

    fn dims(h: usize, kv: usize, hd: usize) -> GqaDims {
        GqaDims { n_heads: h, n_kv_heads: kv, head_dim: hd, eps: 1e-6, acc: Acc::F32, qk_norm: false }
    }

    // The grouping. h / group, not h % kv: both keep every shape identical and one pairs
    // every query head with the wrong keys.
    #[test]
    fn query_heads_map_to_kv_heads_by_division_not_modulo() {
        let (h_n, kv_n, hd, t) = (8usize, 2usize, 4usize, 1usize);
        let d = dims(h_n, kv_n, hd);
        let q = vec![1.0f32; t * h_n * hd];
        let mut k = vec![0f32; t * kv_n * hd];
        let mut v = vec![0f32; t * kv_n * hd];
        // kv head 0 carries value 10, kv head 1 carries value 20.
        for i in 0..hd {
            k[i] = 1.0;
            v[i] = 10.0;
            k[hd + i] = 1.0;
            v[hd + i] = 20.0;
        }
        let mut out = vec![0f32; t * h_n * hd];
        gqa(&mut out, &q, &k, &v, &d, t, None, None, None);
        // Heads 0..3 belong to kv head 0, heads 4..7 to kv head 1.
        for h in 0..h_n {
            let want = if h < 4 { 10.0 } else { 20.0 };
            assert_eq!(out[h * hd], want, "head {h} read the wrong kv group");
        }
    }

    #[test]
    fn mha_is_the_special_case_where_every_head_owns_its_kv() {
        let (h_n, hd, t) = (4usize, 4usize, 3usize);
        let d = dims(h_n, h_n, hd);
        let q: Vec<f32> = (0..t * h_n * hd).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..t * h_n * hd).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..t * h_n * hd).map(|i| i as f32 * 0.01).collect();
        let mut out = vec![0f32; t * h_n * hd];
        gqa(&mut out, &q, &k, &v, &d, t, None, None, None);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // Causality: token 0 can only see itself, so its output is exactly v[0].
    #[test]
    fn attention_is_causal() {
        let (h_n, hd, t) = (2usize, 4usize, 4usize);
        let d = dims(h_n, 1, hd);
        let q: Vec<f32> = (0..t * h_n * hd).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..t * hd).map(|i| (i as f32 * 0.7).cos()).collect();
        let mut v = vec![0f32; t * hd];
        for i in 0..hd {
            v[i] = 5.0;
        }
        let mut out = vec![0f32; t * h_n * hd];
        gqa(&mut out, &q, &k, &v, &d, t, None, None, None);
        for h in 0..h_n {
            assert!((out[h * hd] - 5.0).abs() < 1e-5, "token 0 must see only itself");
        }
    }

    #[test]
    fn softmax_weights_sum_to_one() {
        let (h_n, hd, t) = (1usize, 3usize, 5usize);
        let d = dims(h_n, 1, hd);
        let q: Vec<f32> = (0..t * hd).map(|i| (i as f32 * 0.4).sin()).collect();
        let k: Vec<f32> = (0..t * hd).map(|i| (i as f32 * 0.9).cos()).collect();
        // Constant v: a convex combination of identical rows is that row.
        let v = vec![3.5f32; t * hd];
        let mut out = vec![0f32; t * hd];
        gqa(&mut out, &q, &k, &v, &d, t, None, None, None);
        for x in &out {
            assert!((x - 3.5).abs() < 1e-4, "weights did not sum to 1: {x}");
        }
    }

    // Adjacent vs Halves are different rotations, both norm-preserving.
    #[test]
    fn the_two_rope_pairings_differ_but_both_preserve_norm() {
        let hd = 8;
        let x0: Vec<f32> = (0..hd).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let n0: f32 = x0.iter().map(|v| v * v).sum();
        let mut a = x0.clone();
        let mut b = x0.clone();
        apply_rope(&mut a, &rope(hd, hd, 4, 10000.0, Pairing::Adjacent), 3, false);
        apply_rope(&mut b, &rope(hd, hd, 4, 10000.0, Pairing::Halves), 3, false);
        let na: f32 = a.iter().map(|v| v * v).sum();
        let nb: f32 = b.iter().map(|v| v * v).sum();
        assert!((na - n0).abs() < 1e-4 && (nb - n0).abs() < 1e-4, "rotation must preserve norm");
        assert!(a.iter().zip(&b).any(|(p, q)| (p - q).abs() > 1e-6), "the pairings must differ");
    }

    #[test]
    fn rope_inverse_undoes_rope() {
        let hd = 8;
        let r = rope(hd, hd, 6, 10000.0, Pairing::Adjacent);
        let x0: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut x = x0.clone();
        apply_rope(&mut x, &r, 5, false);
        apply_rope(&mut x, &r, 5, true);
        for (a, b) in x.iter().zip(&x0) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // partial_rotary_factor 0.5: the upper half of each head must come out untouched.
    #[test]
    fn partial_rotary_leaves_the_tail_of_the_head_alone() {
        let (hd, rot) = (8usize, 4usize);
        let r = rope(hd, rot, 4, 5e6, Pairing::Adjacent);
        let x0: Vec<f32> = (0..hd).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let mut x = x0.clone();
        apply_rope(&mut x, &r, 2, false);
        assert_eq!(&x[rot..], &x0[rot..], "only the first rotary_dim elements rotate");
        assert!(x[..rot].iter().zip(&x0[..rot]).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn a_block_sparse_row_never_reaches_past_its_own_token() {
        let rows = sparse_rows(40, 8, 1, 1, 1, |_, b| b as f32);
        for (t, r) in rows.iter().enumerate() {
            for &p in r {
                assert!(p >= 0 && (p as usize) <= t, "token {t} attended to {p}");
            }
            assert!(!r.is_empty(), "token {t} kept nothing");
            assert!(r.contains(&(t as i64)), "token {t} must keep itself");
        }
    }

    #[test]
    fn the_first_and_last_blocks_are_always_kept() {
        let rows = sparse_rows(64, 8, 0, 1, 1, |_, _| 0.0);
        let last = rows.last().unwrap();
        assert!(last.contains(&0), "init block must survive with topk = 0");
        assert!(last.contains(&63), "local block must survive with topk = 0");
    }

    #[test]
    fn selection_rows_are_sorted_and_free_of_duplicates() {
        let rows = sparse_rows(50, 8, 2, 1, 1, |t, b| ((t + b) % 5) as f32);
        for r in &rows {
            let mut s = r.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(&s, r, "a position was kept twice or out of order");
        }
    }

    // The keep-list has to actually restrict attention, or block-sparse is dense.
    #[test]
    fn a_keep_list_restricts_what_attention_reads() {
        let (hd, t) = (4usize, 4usize);
        let d = dims(1, 1, hd);
        let q = vec![0f32; t * hd];
        let k = vec![0f32; t * hd];
        let mut v = vec![0f32; t * hd];
        for p in 0..t {
            for i in 0..hd {
                v[p * hd + i] = p as f32;
            }
        }
        let mut all = vec![0f32; t * hd];
        gqa(&mut all, &q, &k, &v, &d, t, None, None, None);
        // Token 3 over {0,1,2,3} averages to 1.5; restricted to {3} it is exactly 3.
        assert!((all[3 * hd] - 1.5).abs() < 1e-5, "got {}", all[3 * hd]);
        let keep = vec![vec![0i64], vec![1], vec![2], vec![3]];
        let mut only = vec![0f32; t * hd];
        gqa(&mut only, &q, &k, &v, &d, t, None, None, Some(&keep));
        assert!((only[3 * hd] - 3.0).abs() < 1e-5, "got {}", only[3 * hd]);
    }
}
