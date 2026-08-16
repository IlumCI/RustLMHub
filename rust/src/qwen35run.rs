// SPDX-License-Identifier: Apache-2.0
//
// The qwen35moe forward pass: a hybrid gated-delta-net / full-attention stack.
//
// PROVENANCE
//     The operation order here follows llama.cpp's `src/models/qwen35moe.cpp`
//     (`build_layer_attn_linear` and `build_layer_attn`), read rather than inferred. That
//     matters because three of the details below are invisible when wrong -- the model
//     keeps emitting fluent text -- and I had two of them wrong from tensor shapes alone.
//
// THE THREE THAT ARE SILENT WHEN WRONG
//   1. In a full-attention block the output gate is INTERLEAVED PER HEAD inside `attn_q`:
//      the row stride is `2 * head_dim`, query at offset 0 and gate at offset `head_dim`.
//      `attn_q` is [8192, 2048] and 8192 = 2 * 16 * 256, so reading it as "queries then
//      gates" also consumes exactly the tensor and also produces a working model.
//   2. `attn_qkv` in a linear block is NOT three equal parts. Its 8192 columns are
//      16 q heads x 128, 16 k heads x 128, 32 v heads x 128.
//   3. q and k in a linear block get `l2_norm`, not RMS norm: no `sqrt(n)` divisor and no
//      learned gain. Substituting rmsnorm scales both by sqrt(128) = 11.3x, and the delta
//      rule partly absorbs it.
//
// TEXT-ONLY ROPE
//     The reference calls `ggml_rope_multi` (IMRoPE) with a 4-entry `sections` array. That
//     is the multimodal rope: it splits the rotary dims into sections carrying separate
//     position streams for text, height and width. With no image or video in the context
//     every section carries the SAME position -- the token index -- and multi-section rope
//     reduces exactly to ordinary rope at that position. So plain rope is not an
//     approximation here, it is the same function on this input. It stops being so the
//     moment a vision token appears, which is why this is written down.

use crate::gqa::{self, GqaDims, Rope};
use crate::ops::{self, Acc, W};
use crate::qwen35::Cfg;

/// Per-layer recurrent state for a gated-delta-net block.
///
/// Unlike a KV cache this does NOT grow with context: the whole history lives in a fixed
/// `n_v_heads * d_state * head_v_dim` matrix. That is the entire reason 30 of the 40
/// blocks cost no memory per token, and why a 262144-token context is advertised at all.
#[derive(Clone)]
pub struct LinState {
    /// `[conv_kernel - 1][channels]`, the depthwise convolution's history.
    pub conv: Vec<f32>,
    /// `[n_v_heads][d_state * head_v_dim]`, row-major per head.
    pub s: Vec<f32>,
}

impl LinState {
    pub fn new(c: &Cfg) -> LinState {
        LinState {
            conv: vec![0f32; (c.conv_kernel - 1) * c.qkv_width()],
            s: vec![0f32; c.n_v_heads * c.d_state * c.head_v_dim()],
        }
    }
    /// A zero-cost placeholder for a FULL-ATTENTION layer, which never runs the delta-net and
    /// so never reads this state. The recurrent state is ~3 MB per layer, so allocating it for
    /// the attention layers that ignore it wastes ~52 MB on the 27B (16 of 64 layers).
    pub fn empty() -> LinState {
        LinState { conv: Vec::new(), s: Vec::new() }
    }
}

/// Weights of one gated-delta-net block, borrowed in their stored quantised form.
pub struct LinW<'a> {
    pub attn_norm: &'a [f32],
    pub qkv: W<'a>,
    /// `attn_gate` -- the `z` branch of the gated output norm.
    pub z: W<'a>,
    pub alpha: W<'a>,
    pub beta: W<'a>,
    /// `ssm_a`, stored already negative so `exp(softplus(..) * a)` lands in (0, 1].
    pub a: &'a [f32],
    pub dt_bias: &'a [f32],
    /// `[channels][kernel]` depthwise.
    pub conv1d: &'a [f32],
    /// `[head_v_dim]`, the gain of the gated RMS norm.
    pub norm: &'a [f32],
    pub out: W<'a>,
}

/// One decode step of a gated-delta-net block. `x` is the post-`attn_norm` input.
///
/// Returns into `out` the block's contribution, before the residual add.
pub fn linear_block(out: &mut [f32], x: &[f32], w: &LinW, c: &Cfg, st: &mut LinState) {
    linear_block_many(out, x, w, c, st, 1)
}

/// The same block over a CHUNK of `ntok` tokens, token-major.
///
/// WHAT IS BATCHED AND WHAT CANNOT BE
/// ```text
///     The four input projections and the output projection are the same weights for every
///     token, so they run once for the chunk -- see `ops::mmw_many`, which decodes each
///     quantised row once instead of once per token. That is where prefill time goes:
///     `qkv` alone is 2048x8192, and thirty of these blocks make it 21% of all the
///     arithmetic in the model.
///
///     The middle cannot be batched at any width, and that is a property of the
///     architecture rather than of this code. The depthwise convolution carries `kw - 1`
///     steps of history and the delta rule carries the recurrent state, so token `t + 1`
///     cannot start until token `t` has updated both. They stay a serial loop, and it is
///     the projections around them that get the width.
/// ```
///
/// `linear_block` is this function at `ntok = 1`, so there is exactly one implementation
/// of the arithmetic and a batched prompt cannot diverge from a serial one.
pub fn linear_block_many(outs: &mut [f32], xs: &[f32], w: &LinW, c: &Cfg, st: &mut LinState, ntok: usize) {
    linear_block_many_ck(outs, xs, w, c, st, ntok, None)
}

/// As [`linear_block_many`], but if `ckpt0` is provided, the recurrent state AFTER the first
/// token (and before the second) is cloned into it — the checkpoint a speculative verify needs
/// to roll back to "first token only" when the drafted second token is rejected. A gated
/// delta-net state cannot be sliced like a KV cache, so this snapshot is the only cheap way
/// back; without it a rejection would re-stream the whole model to reprocess the accepted token.
#[allow(clippy::too_many_arguments)]
pub fn linear_block_many_ck(
    outs: &mut [f32],
    xs: &[f32],
    w: &LinW,
    c: &Cfg,
    st: &mut LinState,
    ntok: usize,
    mut ckpt0: Option<&mut LinState>,
) {
    let (hid, kw) = (c.hidden, c.conv_kernel);
    let (dk, dv) = (c.d_state, c.head_v_dim());
    let (nk, nv) = (c.n_k_heads, c.n_v_heads);
    let width = c.qkv_width();

    // Sums matching llama-eval-callback's labels, so the first divergence against the
    // reference trace can be read off directly. Env-gated: costs nothing when unset.
    let dbg = std::env::var_os("Q35_SUMS").is_some();
    let sum = |v: &[f32]| v.iter().map(|a| *a as f64).sum::<f64>();

    let mut hs = vec![0f32; ntok * hid];
    for t in 0..ntok {
        ops::rmsnorm(&mut hs[t * hid..][..hid], &xs[t * hid..][..hid], w.attn_norm, hid, c.eps);
    }
    let mut qkv_all = vec![0f32; ntok * width];
    ops::mmw_many(&mut qkv_all, &hs, w.qkv, hid, width, ntok);
    let mut z_all = vec![0f32; ntok * c.d_inner];
    ops::mmw_many(&mut z_all, &hs, w.z, hid, c.d_inner, ntok);
    // Forget gate and write strength, both per V-HEAD -- one scalar for a whole dk x dv
    // state matrix, not one per channel. See ops::gdn_decay.
    let mut araw_all = vec![0f32; ntok * nv];
    ops::mmw_many(&mut araw_all, &hs, w.alpha, hid, nv, ntok);
    let mut beta_all = vec![0f32; ntok * nv];
    ops::mmw_many(&mut beta_all, &hs, w.beta, hid, nv, ntok);

    // Everything from here to the output projection is per token and in order, because the
    // conv history and the recurrent state both advance one token at a time.
    let mut gs = vec![0f32; ntok * c.d_inner];
    for t in 0..ntok {
        let h = &hs[t * hid..][..hid];
        let qkv = &qkv_all[t * width..][..width];
        let z = &z_all[t * c.d_inner..][..c.d_inner];
        if dbg {
            eprintln!("  attn_norm      {:.6}", sum(h));
            eprintln!("  qkv_mixed      {:.6}", sum(qkv));
        }
        let mut decay = vec![0f32; nv];
        ops::gdn_decay(&mut decay, &araw_all[t * nv..][..nv], w.dt_bias, w.a, nv);
        let mut beta = beta_all[t * nv..][..nv].to_vec();
        for b in beta.iter_mut() {
            *b = crate::libm::sigmoidf(*b);
        }
        linear_core(&mut gs[t * c.d_inner..][..c.d_inner], qkv, z, &decay, &beta, w, c, st,
                    dk, dv, nk, nv, width, kw, dbg);
        // Snapshot the recurrent state after the FIRST token, for a speculative rollback.
        if t == 0 {
            if let Some(cp) = ckpt0.as_deref_mut() {
                *cp = st.clone();
            }
        }
    }
    ops::mmw_many(outs, &gs, w.out, c.d_inner, hid, ntok);
    if dbg {
        for t in 0..ntok {
            eprintln!("  linear_attn_out{:.6}", sum(&outs[t * hid..][..hid]));
        }
    }
}

/// The serial middle of a gated-delta-net block: conv, delta rule, gated norm.
///
/// Split out only so `linear_block_many` can read as "batch, walk, batch". Everything here
/// touches `st` and must run in token order.
#[allow(clippy::too_many_arguments)]
fn linear_core(
    g: &mut [f32],
    qkv: &[f32],
    z: &[f32],
    decay: &[f32],
    beta: &[f32],
    w: &LinW,
    c: &Cfg,
    st: &mut LinState,
    dk: usize,
    dv: usize,
    nk: usize,
    nv: usize,
    width: usize,
    kw: usize,
    dbg: bool,
) {
    let sum = |v: &[f32]| v.iter().map(|a| *a as f64).sum::<f64>();
    // Causal depthwise conv over ALL mixed channels at once. The conv runs before the
    // q/k/v split, which is why its channel count is the full 8192.
    //
    // `shortconv` applies SiLU ITSELF -- the activation is fused into the tap loop. An
    // extra `x * sigmoid(x)` here is a double SiLU on every channel; it stays bounded and
    // monotone, so nothing errors, and the model emits confident nonsense.
    let mut conv = vec![0f32; width];
    // t_len 1: one decode step. The state carries the previous kw-1 steps.
    ops::shortconv(&mut conv, qkv, w.conv1d, Some(&mut st.conv), width, kw, 1);
    if dbg { eprintln!("  conv_out_silu  {:.6}", sum(&conv)); }

    // q | k | v, with q and k SIXTEEN heads and v THIRTY-TWO.
    let (q_off, k_off, v_off) = (0, nk * dk, 2 * nk * dk);
    let mut q = vec![0f32; nk * dk];
    let mut k = vec![0f32; nk * dk];
    ops::l2norm_heads(&mut q, &conv[q_off..][..nk * dk], nk, dk, c.eps);
    // The query carries the usual 1/sqrt(d) attention scale. L2-normalising q and k makes
    // their dot product a cosine in [-1, 1], so this is easy to believe unnecessary and
    // easy to omit -- the model runs and stays well-scaled without it, because the gated
    // RMSNorm downstream reabsorbs most of a constant factor.
    //
    // Recovered by arithmetic, not by reading: with a zero initial state the recurrence is
    // exactly o = beta * (q.k) * v, and our sum was 0.007609 against llama.cpp's 0.001903.
    // The ratio is 3.9985 = sqrt(16) = sqrt(d_state).
    let qscale = 1.0 / (dk as f32).sqrt();
    for v in q.iter_mut() {
        *v *= qscale;
    }
    ops::l2norm_heads(&mut k, &conv[k_off..][..nk * dk], nk, dk, c.eps);
    let v = &conv[v_off..][..c.d_inner];
    if dbg {
        eprintln!("  q_predelta     {:.6}", sum(&q));
        eprintln!("  k_predelta     {:.6}", sum(&k));
        eprintln!("  v_predelta     {:.6}", sum(v));
        eprintln!("  beta_sigmoid   {:.6}  gate {:.6}  z {:.6}", sum(beta), sum(decay), sum(z));
    }

    // Value head `hv` reads key head `hv % nk` -- MODULO, not divide.
    //
    // The reference widens q and k from `nk` heads to `nv` with `ggml_repeat_4d`, and
    // ggml_repeat TILES rather than blocking: it maps index i to `i % n_src`, so 2 key
    // heads over 4 value heads give [0, 1, 0, 1], not [0, 0, 1, 1]. I had this backwards
    // and wrote a test asserting the wrong one, because both mappings have identical
    // shapes and both produce a running model.
    //
    // Caught by diffing `attn_output` against llama-eval-callback on the tiny fixture:
    // every input to the recurrence matched to f32 rounding and the output did not.
    let mut o = vec![0f32; c.d_inner];
    // kda_step takes a per-channel decay; ours is one scalar broadcast over the head. It
    // is filled rather than reimplemented so the delta rule has exactly one implementation
    // in this crate, already covered by the KDA fixtures.
    let mut alpha = vec![0f32; dk];
    for hv in 0..nv {
        alpha.fill(decay[hv]);
        let kh = hv % nk;
        // MUTATION HOOK, not dead code. `head_v_dim == d_state` is a constraint of this
        // architecture -- llama.cpp derives the value width from d_state and refuses a
        // model where they differ -- so the per-head state is always SQUARE and a
        // transposed state cannot be caught by a shape check.
        //
        // Setting Q35_MUTATE_TRANSPOSE flips the state's indexing so the fixture diff can
        // prove it WOULD catch one. Measured on the 3-token fixture: correct +0.00074800
        // (= llama.cpp exactly), transposed +0.00864300. Note the FIRST token is identical
        // either way (+0.001902) because the state starts at zero -- a single-token diff
        // cannot see this, which is why the reference prompt has three.
        if std::env::var_os("Q35_MUTATE_TRANSPOSE").is_some() {
            let base = hv * dk * dv;
            let mut t = vec![0f32; dk * dv];
            for i in 0..dk { for j in 0..dv { t[j * dk + i] = st.s[base + i * dv + j]; } }
            st.s[base..base + dk * dv].copy_from_slice(&t);
        }
        ops::kda_step(
            &mut st.s[hv * dk * dv..][..dk * dv],
            &mut o[hv * dv..][..dv],
            &q[kh * dk..][..dk],
            &k[kh * dk..][..dk],
            &v[hv * dv..][..dv],
            &alpha,
            beta[hv],
            dk,
            dv,
        );
    }

    if dbg {
        // With a zero initial state the recurrence collapses to o_hv = beta_hv * (q.k) * v_hv,
        // so the whole block reduces to arithmetic we can solve for the head pairing.
        for a in 0..nk {
            for b in 0..nk {
                let d: f32 = (0..dk).map(|i| q[a * dk + i] * k[b * dk + i]).sum();
                eprintln!("    dot(q{a},k{b}) = {d:.6}");
            }
        }
        for hv in 0..nv {
            let sv: f32 = v[hv * dv..][..dv].iter().sum();
            eprintln!("    head {hv}: beta {:.6} sum_v {sv:.6}", beta[hv]);
        }
    }
    if dbg { eprintln!("  attn_output    {:.6}   [reference 0.001903]", sum(&o)); }
    // Gated RMS norm: rms_norm(o_head, gain) * silu(z_head), per value head.
    //
    // BOTH ORDERS HAVE BEEN TRIED against the llama.cpp reference trace and NEITHER
    // reproduces it: this one gives final_output 0.460749 and gating-before-normalising
    // gives 0.914849, where the reference is 0.574183. Since q, k, v, beta, gate and z all
    // match the reference exactly at this point, the discrepancy is inside the delta rule
    // or this norm, and it is not merely their order. Left as the cited form
    // (rms_norm then gate) until the fused `__fgdn_ar__` kernel's semantics are pinned
    // down; see tools/make_tiny_qwen35.py for how to regenerate the reference.
    for hv in 0..nv {
        ops::rmsnorm(&mut g[hv * dv..][..dv], &o[hv * dv..][..dv], w.norm, dv, c.eps);
    }
    for (gi, zi) in g.iter_mut().zip(z.iter()) {
        *gi *= *zi * crate::libm::sigmoidf(*zi);
    }
    if dbg { eprintln!("  final_output   {:.6}", sum(g)); }
}

/// Weights of one full-attention block.
pub struct AttnW<'a> {
    pub attn_norm: &'a [f32],
    /// `attn_q`: query and output gate INTERLEAVED per head, stride `2 * head_dim`.
    pub qg: &'a QW,
    pub k: &'a QW,
    pub v: &'a QW,
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub o: &'a QW,
}

/// Rolling KV for a full-attention block. Only 10 of 40 blocks have one.
#[derive(Clone)]
pub struct AttnState {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

impl AttnState {
    pub fn new() -> AttnState {
        AttnState { k: Vec::new(), v: Vec::new(), len: 0 }
    }
    /// Drop KV entries back to `new_len` tokens. `kv_width = n_kv_heads * head_dim` is the
    /// per-token stride of both `k` and `v`. Used to roll a rejected speculative token's KV
    /// out of the cache without recomputing the accepted prefix.
    pub fn truncate_to(&mut self, new_len: usize, kv_width: usize) {
        if new_len < self.len {
            self.k.truncate(new_len * kv_width);
            self.v.truncate(new_len * kv_width);
            self.len = new_len;
        }
    }
}

impl Default for AttnState {
    fn default() -> Self {
        AttnState::new()
    }
}

/// Pull the queries out of the fused `attn_q` output, discarding the interleaved gates.
///
/// Layout is `[q_0, gate_0, q_1, gate_1, ...]` with each part `head_dim` wide -- NOT all
/// queries followed by all gates. Both readings consume the tensor exactly.
pub fn take_q(dst: &mut [f32], qg: &[f32], n_heads: usize, head_dim: usize) {
    for h in 0..n_heads {
        let src = &qg[h * 2 * head_dim..][..head_dim];
        dst[h * head_dim..][..head_dim].copy_from_slice(src);
    }
}

/// The other half of the same interleaving.
pub fn take_gate(dst: &mut [f32], qg: &[f32], n_heads: usize, head_dim: usize) {
    for h in 0..n_heads {
        let src = &qg[h * 2 * head_dim + head_dim..][..head_dim];
        dst[h * head_dim..][..head_dim].copy_from_slice(src);
    }
}

/// One decode step of a full-attention block at position `pos`.
pub fn attn_block(
    out: &mut [f32],
    x: &[f32],
    w: &AttnW,
    c: &Cfg,
    st: &mut AttnState,
    rope: &Rope,
    pos: usize,
) {
    attn_block_many(out, x, w, c, st, rope, pos, 1)
}

/// The same block over a CHUNK of `ntok` tokens starting at `pos0`, token-major.
///
/// Same shape as `linear_block_many` and for the same reason: the q/gate, k, v and output
/// projections are chunk-wide, while the attention itself stays serial because each token
/// attends over a KV cache the previous token just extended. Ten of these blocks make the
/// projections around 11% of the model's arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn attn_block_many(
    outs: &mut [f32],
    xs: &[f32],
    w: &AttnW,
    c: &Cfg,
    st: &mut AttnState,
    rope: &Rope,
    pos0: usize,
    ntok: usize,
) {
    let (hid, hd) = (c.hidden, c.head_dim);
    let (nh, nkv) = (c.n_heads, c.n_kv_heads);

    let mut hs = vec![0f32; ntok * hid];
    for t in 0..ntok {
        ops::rmsnorm(&mut hs[t * hid..][..hid], &xs[t * hid..][..hid], w.attn_norm, hid, c.eps);
    }
    let mut qg_all = vec![0f32; ntok * 2 * nh * hd];
    w.qg.mm(&mut qg_all, &hs, hid, 2 * nh * hd, ntok);
    let mut k_all = vec![0f32; ntok * nkv * hd];
    let mut v_all = vec![0f32; ntok * nkv * hd];
    w.k.mm(&mut k_all, &hs, hid, nkv * hd, ntok);
    w.v.mm(&mut v_all, &hs, hid, nkv * hd, ntok);

    let dbg = std::env::var_os("Q35_SUMS").is_some();
    let sm = |v: &[f32]| v.iter().map(|a| *a as f64).sum::<f64>();
    let d = GqaDims {
        n_heads: nh,
        n_kv_heads: nkv,
        head_dim: hd,
        eps: c.eps,
        acc: Acc::F64,
        qk_norm: false,
    };
    let mut ctxs = vec![0f32; ntok * nh * hd];
    let mut q = vec![0f32; nh * hd];
    let mut gate = vec![0f32; nh * hd];
    let mut tmp = vec![0f32; hd];
    for t in 0..ntok {
        let qg = &qg_all[t * 2 * nh * hd..][..2 * nh * hd];
        take_q(&mut q, qg, nh, hd);
        take_gate(&mut gate, qg, nh, hd);
        let k = &mut k_all[t * nkv * hd..][..nkv * hd];

        // Per-head RMS norm on q and k BEFORE rope. Doing it after rotates unnormalised
        // vectors and then normalises away part of the angle.
        for i in 0..nh {
            ops::rmsnorm(&mut tmp, &q[i * hd..][..hd], w.q_norm, hd, c.eps);
            q[i * hd..][..hd].copy_from_slice(&tmp);
        }
        for i in 0..nkv {
            ops::rmsnorm(&mut tmp, &k[i * hd..][..hd], w.k_norm, hd, c.eps);
            k[i * hd..][..hd].copy_from_slice(&tmp);
        }
        for i in 0..nh {
            gqa::apply_rope(&mut q[i * hd..][..hd], rope, pos0 + t, false);
        }
        for i in 0..nkv {
            gqa::apply_rope(&mut k[i * hd..][..hd], rope, pos0 + t, false);
        }
        if dbg {
            eprintln!("  [ATTN] q_normed {:.6} k_normed {:.6}", sm(&q), sm(k));
        }
        st.k.extend_from_slice(k);
        st.v.extend_from_slice(&v_all[t * nkv * hd..][..nkv * hd]);
        st.len += 1;

        // One query against the whole prefix -- O(len), not O(len^2). `gqa_last` is pinned
        // bit-identical to `gqa`'s final row by a test, so this is a cost change only.
        let last = &mut ctxs[t * nh * hd..][..nh * hd];
        gqa::gqa_last(last, &q, &st.k, &st.v, &d, st.len, None, None);
        if dbg { eprintln!("  [ATTN] attn_pregate {:.6}", sm(last)); }
        // Sigmoid output gate, applied to the attention result and only then projected.
        for (o, g) in last.iter_mut().zip(gate.iter()) {
            *o *= crate::libm::sigmoidf(*g);
        }
        if dbg { eprintln!("  [ATTN] attn_gated {:.6}", sm(last)); }
    }
    w.o.mm(outs, &ctxs, nh * hd, hid, ntok);
    if dbg {
        for t in 0..ntok {
            eprintln!("  [ATTN] attn_output {:.6}", sm(&outs[t * hid..][..hid]));
        }
    }
}

/// The dense feed-forward of one block, over a CHUNK of `ntok` tokens.
///
/// WHY THIS IS THE SAME CODE AS AN EXPERT
/// ```text
///     A dense FFN is arithmetically one expert with a router weight of 1 that every token
///     selects. So this is `moe_many` with the routing removed: normalise, fetch, apply.
///     `expert_fwd_many` is reused verbatim rather than copied, which is what keeps SwiGLU,
///     the gate/up pairing and the batched decode identical between the two architectures
///     -- there is no second implementation to drift.
///
///     It streams through the same `Cache` too. Qwen3.8-27B's FFN is 10.53 GB of a 17.10 GB
///     checkpoint, so on an 11 GB machine it is exactly the part that cannot be resident,
///     and the layer sweep it is fetched on is the access pattern that cache was built for.
///     One difference in its favour: for a dense model the next fetch is not predicted, it
///     is KNOWN -- layer l is always followed by layer l+1 -- so the prefetch is exact and
///     the read overlaps compute completely.
/// ```
///
/// Bit-identical to `ntok` single-token calls, because `expert_fwd_many` is.
#[allow(clippy::too_many_arguments)]
pub fn ffn_many(
    out: &mut [f32],
    xs: &[f32],
    m: &Moe,
    c: &Cfg,
    dt: &[Dtype; 3],
    st: &St,
    cache: &mut crate::cache::Cache,
    layer: usize,
    k: usize,
) -> Result<(), String> {
    let (hid, inter) = (c.hidden, c.dense_inter);
    let mut hs = vec![0f32; k * hid];
    for t in 0..k {
        ops::rmsnorm(&mut hs[t * hid..][..hid], &xs[t * hid..][..hid], &m.post_norm, hid, c.eps);
    }
    // Expert 0 of a one-expert layer. `gguf_dense_ffn_src` names the UNSTACKED tensors and
    // takes them whole; see `cache::ExpertSrc::Whole` for why slicing them would be a
    // silent 1/17408th of a model rather than an error.
    let src = crate::cache::gguf_dense_ffn_src(layer, 0);
    let slot = cache
        .get(st, layer, 0, &src)
        .ok_or_else(|| format!("layer {layer} feed-forward could not be cached"))?;
    // Overlap the NEXT layer's FFN read with THIS layer's compute. The dense layer order is
    // fixed (l -> l+1), so the prediction is exact: the 165 MB read streams on the prefetch
    // thread while the int8 matmuls below run on a CPU that would otherwise sit idle waiting
    // on the disk. Decode is disk-bandwidth-bound with the cores ~84% idle, so this hides the
    // compute behind the read for free. A no-op unless a Prefetcher is attached; the request
    // lands in the worker's own buffer, so `slot` (this layer) stays valid through the matmul.
    if layer + 1 < c.n_layers {
        cache.prefetch_hint(layer + 1, &[0], crate::cache::gguf_dense_ffn_src);
    }
    // Certified activation sparsity (opt-in via Q35_CERT=<budget fraction>): skip the FFN
    // neurons whose bounded contribution sums under `frac` of the layer's total, provably
    // bounding the output perturbation. Correctness + skip-rate measurement path; realising
    // the SPEED needs the down^T neuron-major repack so skipped columns are not decoded.
    if let Ok(bs) = std::env::var("Q35_CERT") {
        let frac: f32 = bs.parse().unwrap_or(0.01);
        return ffn_many_cert(out, &hs, &cache.expert(slot), dt, hid, inter, layer, k, frac);
    }
    // Router weight 1.0: every token takes this path, at full strength.
    let ones = vec![1.0f32; k];
    expert_fwd_many(out, &hs, &cache.expert(slot), dt, hid, inter, &ones, k)
}

/// Per-layer down-column norms (lazy, cached — a property of the frozen weight) and the
/// running skip-rate counters for the certified-sparsity measurement path.
static CERT_NORMS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<Vec<f32>>>>> =
    std::sync::OnceLock::new();
static CERT_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CERT_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn cert_col_norms(
    layer: usize,
    down_bytes: &[u8],
    down_dt: Dtype,
    hidden: usize,
    inter: usize,
) -> Result<std::sync::Arc<Vec<f32>>, String> {
    let map = CERT_NORMS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(n) = map.lock().unwrap().get(&layer) {
        return Ok(n.clone());
    }
    // The `down` weight's quant varies per layer in a `_M` build, so pick the matching block
    // size and dequant. Anything else is an error, not a silent wrong stride.
    #[allow(clippy::type_complexity)]
    let (block, dq): (usize, fn(&mut [f32], &[u8], usize)) = match down_dt {
        Dtype::Q4K => (crate::gguf::Q4K_BLOCK, crate::gguf::q4k_dequant),
        Dtype::Q5K => (crate::gguf::Q5K_BLOCK, crate::gguf::q5k_dequant),
        Dtype::Q6K => (crate::gguf::Q6K_BLOCK, crate::gguf::q6k_dequant),
        d => return Err(format!("certified sparsity: down weight is {}, no dequant", d.name())),
    };
    let norms =
        std::sync::Arc::new(crate::certsparse::down_col_norms_q(down_bytes, hidden, inter, block, dq));
    map.lock().unwrap().insert(layer, norms.clone());
    Ok(norms)
}

/// Cumulative (skipped, total) FFN neuron counts across all certified-sparsity calls, for the
/// caller to print the realised skip rate.
pub fn cert_report() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CERT_SKIPPED.load(Relaxed), CERT_TOTAL.load(Relaxed))
}

/// Per-layer neuron-major transposed `down` (lazy, cached), so skipped neurons' rows are
/// never decoded in the down projection.
static CERT_DOWNT: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<crate::downt::DownT>>>> =
    std::sync::OnceLock::new();

fn cert_downt(
    layer: usize,
    down_bytes: &[u8],
    down_dt: Dtype,
    hidden: usize,
    inter: usize,
) -> Result<std::sync::Arc<crate::downt::DownT>, String> {
    let map = CERT_DOWNT.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Some(d) = map.lock().unwrap().get(&layer) {
        return Ok(d.clone());
    }
    #[allow(clippy::type_complexity)]
    let (block, dq): (usize, fn(&mut [f32], &[u8], usize)) = match down_dt {
        Dtype::Q4K => (crate::gguf::Q4K_BLOCK, crate::gguf::q4k_dequant),
        Dtype::Q5K => (crate::gguf::Q5K_BLOCK, crate::gguf::q5k_dequant),
        Dtype::Q6K => (crate::gguf::Q6K_BLOCK, crate::gguf::q6k_dequant),
        d => return Err(format!("certified sparsity: down weight is {}, no dequant", d.name())),
    };
    let dt = std::sync::Arc::new(crate::downt::DownT::repack(down_bytes, hidden, inter, block, dq));
    map.lock().unwrap().insert(layer, dt.clone());
    Ok(dt)
}

/// Certified-sparse dense FFN: gate/up/SwiGLU as usual, then skip the neurons whose bounded
/// contribution `Σ|aᵢ|·‖down[:,i]‖` stays under `frac` of the layer total, before the down
/// projection. The skipped activations are zeroed (so the output stays within the certified
/// bound); the down matmul is still full-width here, so this measures the achievable skip
/// rate rather than yet realising its speed. Uses the verified [`crate::certsparse`] core.
#[allow(clippy::too_many_arguments)]
fn ffn_many_cert(
    outs: &mut [f32],
    xs: &[f32],
    q: &crate::cache::ExpertQ,
    dt: &[Dtype; 3],
    hidden: usize,
    inter: usize,
    layer: usize,
    ntok: usize,
    frac: f32,
) -> Result<(), String> {
    let mut gate = vec![0f32; ntok * inter];
    let mut up = vec![0f32; ntok * inter];
    ops::mmw_many(&mut gate, xs, w("gate", q.p1, dt[0])?, hidden, inter, ntok);
    ops::mmw_many(&mut up, xs, w("up", q.p3, dt[1])?, hidden, inter, ntok);

    let norms = cert_col_norms(layer, q.p2, dt[2], hidden, inter)?;
    let downt = cert_downt(layer, q.p2, dt[2], hidden, inter)?;

    let mut gu = vec![0f32; 2 * inter];
    let mut act = vec![0f32; inter];
    for t in 0..ntok {
        gu[..inter].copy_from_slice(&gate[t * inter..][..inter]);
        gu[inter..].copy_from_slice(&up[t * inter..][..inter]);
        ops::glu(&mut act, &gu, inter, SWIGLU);
        // Certified skip: zero the neurons whose bounded contribution fits the budget.
        let c = crate::certsparse::contribs(&act, &norms);
        let budget = frac * c.iter().sum::<f32>();
        let (keep, _bound) = crate::certsparse::certified_keep_mask(&c, budget);
        let mut skipped = 0u64;
        for i in 0..inter {
            if !keep[i] {
                act[i] = 0.0;
                skipped += 1;
            }
        }
        use std::sync::atomic::Ordering::Relaxed;
        CERT_SKIPPED.fetch_add(skipped, Relaxed);
        CERT_TOTAL.fetch_add(inter as u64, Relaxed);
        // Down projection through the neuron-major transpose: skipped neurons' rows are
        // never decoded, so the skip is now a real compute saving, not just a zero-multiply.
        downt.matvec_skip(&act, &mut outs[t * hidden..][..hidden]);
    }
    Ok(())
}

// ============================ MoEfication Stage-0 capture ============================
//
// The go/no-go measurement (see the MoEfication plan): for a chosen set of layers, capture
// per token the FFN input `hs` (post-norm — the router's future training input) and the
// per-neuron contribution `c_j = |silu(gate·hs)_j · (up·hs)_j| · ‖down_col_j‖₂` (the CETT
// quantity, reused from `certsparse`). These stream to disk and feed the OFFLINE clustering
// + oracle-recall + arena simulation that decide whether a routed FFN can reach ~40% active
// without dropping the neurons that matter — BEFORE any routed FFN is built. Nothing here
// changes a shipped decode path; it only adds a measurement tap around the existing FFN.

/// Disk sink for the capture: one raw-`f32`-little-endian file pair per captured layer
/// (`layer_{l}_c.f32` = `[ntok × inter]`, `layer_{l}_hs.f32` = `[ntok × hidden]`). Written
/// token by token so the whole calibration set never sits in RAM. `cap` bounds the tokens
/// captured per layer; the down-column norms are lazily precomputed once per layer (a
/// property of the frozen weight) exactly as the certified-sparsity path does.
pub struct MoefyCapture {
    layers: std::collections::HashSet<usize>,
    cap: usize,
    ntok: usize,
    #[allow(clippy::type_complexity)]
    w: std::collections::HashMap<usize, (std::io::BufWriter<std::fs::File>, std::io::BufWriter<std::fs::File>)>,
    norms: std::collections::HashMap<usize, std::sync::Arc<Vec<f32>>>,
}

impl MoefyCapture {
    pub fn new(dir: &std::path::Path, layers: &[usize], cap: usize) -> Result<MoefyCapture, String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let mut w = std::collections::HashMap::new();
        for &l in layers {
            let cf = std::fs::File::create(dir.join(format!("layer_{l}_c.f32"))).map_err(|e| e.to_string())?;
            let hf = std::fs::File::create(dir.join(format!("layer_{l}_hs.f32"))).map_err(|e| e.to_string())?;
            w.insert(l, (std::io::BufWriter::new(cf), std::io::BufWriter::new(hf)));
        }
        Ok(MoefyCapture { layers: layers.iter().copied().collect(), cap, ntok: 0, w, norms: std::collections::HashMap::new() })
    }

    /// Tokens captured so far (identical across every captured layer).
    pub fn captured(&self) -> usize {
        self.ntok
    }
    /// True once the per-layer token budget is met — the driver should stop feeding.
    pub fn full(&self) -> bool {
        self.ntok >= self.cap
    }

    /// Down-column L2 norms for `layer`, computed once and cached. Dtype-dispatched because a
    /// `_M` build stores `down` as Q6_K on most layers and Q4_K on a few (a wrong stride is a
    /// silent wrong answer, so it is an error, not a guess).
    fn norms_for(&mut self, layer: usize, down_bytes: &[u8], dt: Dtype, hidden: usize, inter: usize) -> Result<std::sync::Arc<Vec<f32>>, String> {
        if let Some(n) = self.norms.get(&layer) {
            return Ok(n.clone());
        }
        #[allow(clippy::type_complexity)]
        let (block, dq): (usize, fn(&mut [f32], &[u8], usize)) = match dt {
            Dtype::Q4K => (crate::gguf::Q4K_BLOCK, crate::gguf::q4k_dequant),
            Dtype::Q5K => (crate::gguf::Q5K_BLOCK, crate::gguf::q5k_dequant),
            Dtype::Q6K => (crate::gguf::Q6K_BLOCK, crate::gguf::q6k_dequant),
            d => return Err(format!("moefy capture: down weight is {}, no dequant", d.name())),
        };
        let n = std::sync::Arc::new(crate::certsparse::down_col_norms_q(down_bytes, hidden, inter, block, dq));
        self.norms.insert(layer, n.clone());
        Ok(n)
    }

    fn write_token(&mut self, layer: usize, hs: &[f32], c: &[f32]) -> Result<(), String> {
        use std::io::Write;
        let (cw, hw) = self.w.get_mut(&layer).unwrap();
        let mut cb = Vec::with_capacity(c.len() * 4);
        for &v in c {
            cb.extend_from_slice(&v.to_le_bytes());
        }
        cw.write_all(&cb).map_err(|e| e.to_string())?;
        let mut hb = Vec::with_capacity(hs.len() * 4);
        for &v in hs {
            hb.extend_from_slice(&v.to_le_bytes());
        }
        hw.write_all(&hb).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Flush every writer. Call once at the end; dropping alone would flush but swallow errors.
    pub fn finish(&mut self) -> Result<(), String> {
        use std::io::Write;
        for (cw, hw) in self.w.values_mut() {
            cw.flush().map_err(|e| e.to_string())?;
            hw.flush().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Full-stack forward over a chunk of `ids`, tapping the FFN internals at the sink's captured
/// layers. The residual arithmetic is IDENTICAL to [`step_many`] — the captured FFN is the
/// same gate/up/SwiGLU/down at router weight 1.0 — so the activations recorded are the ones
/// the real model computes. No logits are produced; this is a measurement pass. Advances
/// `s.pos` and the sink's token count (by the number of tokens captured this call).
pub fn capture_ffn_stats(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    ids: &[u32],
    sink: &mut MoefyCapture,
) -> Result<(), String> {
    let c = &t.cfg;
    if !c.is_dense() {
        return Err("moefy capture targets a DENSE qwen35 FFN".into());
    }
    let k = ids.len();
    if k == 0 {
        return Err("capture needs at least one token".into());
    }
    let hid = c.hidden;
    // Tokens to tap this call: the first `take` positions of the chunk, so every captured
    // layer records the SAME token set and the per-layer files stay row-aligned.
    let take = (sink.cap - sink.ntok).min(k);
    let mut xs = vec![0f32; k * hid];
    for (i, id) in ids.iter().enumerate() {
        t.io.embed_row(*id, &mut xs[i * hid..][..hid])?;
    }
    let mut branch = vec![0f32; k * hid];
    for l in 0..c.n_layers {
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => linear_block_many(&mut branch, &xs, &w.view(), c, &mut s.lin[l], k),
            Block::Full(w) => {
                attn_block_many(&mut branch, &xs, &w.view(), c, &mut s.attn[l], &t.rope, s.pos, k)
            }
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
        if take > 0 && sink.layers.contains(&l) {
            capture_layer_ffn(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k, take, sink)?;
        } else {
            ffn_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
    }
    s.pos += k;
    sink.ntok += take;
    Ok(())
}

/// The dense FFN at one captured layer: compute `hs`, gate/up, SwiGLU `act` and the real
/// down projection into `out` (bit-identical to `ffn_many`'s non-cert path), and along the
/// way tap the first `take` tokens' `(hs, c)` into the sink. `c` reuses `certsparse::contribs`.
#[allow(clippy::too_many_arguments)]
fn capture_layer_ffn(
    out: &mut [f32],
    xs: &[f32],
    m: &Moe,
    c: &Cfg,
    dt: &[Dtype; 3],
    st: &St,
    cache: &mut crate::cache::Cache,
    layer: usize,
    k: usize,
    take: usize,
    sink: &mut MoefyCapture,
) -> Result<(), String> {
    let (hid, inter) = (c.hidden, c.dense_inter);
    let mut hs = vec![0f32; k * hid];
    for tk in 0..k {
        ops::rmsnorm(&mut hs[tk * hid..][..hid], &xs[tk * hid..][..hid], &m.post_norm, hid, c.eps);
    }
    let src = crate::cache::gguf_dense_ffn_src(layer, 0);
    let slot = cache
        .get(st, layer, 0, &src)
        .ok_or_else(|| format!("layer {layer} feed-forward could not be cached"))?;
    let q = cache.expert(slot);
    let mut gate = vec![0f32; k * inter];
    let mut up = vec![0f32; k * inter];
    ops::mmw_many(&mut gate, &hs, w("gate", q.p1, dt[0])?, hid, inter, k);
    ops::mmw_many(&mut up, &hs, w("up", q.p3, dt[1])?, hid, inter, k);
    let norms = sink.norms_for(layer, q.p2, dt[2], hid, inter)?;
    let mut act = vec![0f32; k * inter];
    let mut gu = vec![0f32; 2 * inter];
    for tk in 0..k {
        gu[..inter].copy_from_slice(&gate[tk * inter..][..inter]);
        gu[inter..].copy_from_slice(&up[tk * inter..][..inter]);
        ops::glu(&mut act[tk * inter..][..inter], &gu, inter, SWIGLU);
    }
    for tk in 0..take {
        let cc = crate::certsparse::contribs(&act[tk * inter..][..inter], &norms);
        sink.write_token(layer, &hs[tk * hid..][..hid], &cc)?;
    }
    // The real down projection (router weight 1.0), so the residual the rest of the stack
    // sees is exactly `ffn_many`'s.
    ops::mmw_many(out, &act, w("down", q.p2, dt[2])?, inter, hid, k);
    Ok(())
}

// ============================ Native MTP / speculative-decode head ============================
//
// Block `n_layers` of the checkpoint (block 64 here, `nextn_predict_layers=1`) is a full
// transformer block wrapped by the next-n projection: it predicts the token TWO ahead from the
// main model's hidden state at position t plus the embedding of the token at t+1. That makes it
// a drafter for speculative decoding, whose acceptance rate against the main model's own greedy
// tokens is what determines the achievable speedup. Weights are ~230 MB, held resident.

pub struct MtpHead {
    attn: FullBlock,
    post_norm: Vec<f32>,
    gate: QW,
    up: QW,
    down: QW,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    shared_head_norm: Vec<f32>,
    /// eh_proj dequantised from Q8_0 to f32, row-major `[hidden][2*hidden]`.
    eh_proj: Vec<f32>,
}

impl MtpHead {
    pub fn load(st: &St, cfg: &Cfg) -> Result<MtpHead, String> {
        let l = cfg.n_layers; // the MTP block sits at index n_layers, excluded from the stack
        let p = |s: &str| format!("blk.{l}.{s}");
        let attn = FullBlock {
            attn_norm: f32s(st, &p("attn_norm.weight"))?,
            qg: QW::load(st, &p("attn_q.weight"))?,
            k: QW::load(st, &p("attn_k.weight"))?,
            v: QW::load(st, &p("attn_v.weight"))?,
            q_norm: f32s(st, &p("attn_q_norm.weight"))?,
            k_norm: f32s(st, &p("attn_k_norm.weight"))?,
            o: QW::load(st, &p("attn_output.weight"))?,
        };
        // eh_proj is Q8_0, which has no in-place matmul kernel; dequantise once (~210 MB f32).
        let ehn = st.find(&p("nextn.eh_proj.weight")).ok_or("missing eh_proj")?;
        let mut ehb = vec![0u8; ehn.nbytes as usize];
        st.read(ehn, &mut ehb);
        let eh_elems = ehn.numel() as usize;
        let mut eh_proj = vec![0f32; eh_elems];
        crate::gguf::q8_0_dequant(&mut eh_proj, &ehb, eh_elems / 32);
        Ok(MtpHead {
            attn,
            post_norm: f32s(st, &p("post_attention_norm.weight"))?,
            gate: QW::load(st, &p("ffn_gate.weight"))?,
            up: QW::load(st, &p("ffn_up.weight"))?,
            down: QW::load(st, &p("ffn_down.weight"))?,
            enorm: f32s(st, &p("nextn.enorm.weight"))?,
            hnorm: f32s(st, &p("nextn.hnorm.weight"))?,
            shared_head_norm: f32s(st, &p("nextn.shared_head_norm.weight"))?,
            eh_proj,
        })
    }

    /// Draft the token TWO ahead for each position, batched: given the main model's per-position
    /// final hidden `hidden[n*hidden]` and `next_ids[t]` = the token at position t+1, returns the
    /// greedy draft `draft[t]` (the head's prediction of the token at t+2). The eh_proj input
    /// order defaults to `[emb ; hidden]`; `MTP_ORDER=he` flips it (a wrong order collapses the
    /// acceptance rate to ~0, which is the built-in correctness check).
    pub fn draft_many(
        &self,
        io: &crate::qwen35::Io,
        cfg: &Cfg,
        rope: &Rope,
        hidden: &[f32],
        next_ids: &[u32],
    ) -> Result<Vec<u32>, String> {
        let hid = cfg.hidden;
        let inter = cfg.dense_inter;
        let n = next_ids.len();
        let he_order = std::env::var("MTP_ORDER").map(|s| s == "he").unwrap_or(false);

        // 1. combined input  x_t = eh_proj( concat(enorm(emb(next_t)), hnorm(hidden_t)) )
        let mut comb = vec![0f32; n * hid];
        let (mut e, mut en, mut hn) = (vec![0f32; hid], vec![0f32; hid], vec![0f32; hid]);
        let mut cat = vec![0f32; 2 * hid];
        for t in 0..n {
            io.embed_row(next_ids[t], &mut e)?;
            ops::rmsnorm(&mut en, &e, &self.enorm, hid, cfg.eps);
            ops::rmsnorm(&mut hn, &hidden[t * hid..][..hid], &self.hnorm, hid, cfg.eps);
            let (first, second) = if he_order { (&hn, &en) } else { (&en, &hn) };
            cat[..hid].copy_from_slice(first);
            cat[hid..].copy_from_slice(second);
            ops::mmw(&mut comb[t * hid..][..hid], &cat, crate::ops::W::F32(&self.eh_proj), 2 * hid, hid);
        }

        // 2. one transformer block over the combined inputs (fresh KV, causal, rope from 0)
        let mut sa = AttnState::new();
        let mut branch = vec![0f32; n * hid];
        attn_block_many(&mut branch, &comb, &self.attn.view(), cfg, &mut sa, rope, 0, n);
        for i in 0..n * hid {
            comb[i] += branch[i];
        }
        let mut hs = vec![0f32; n * hid];
        for t in 0..n {
            ops::rmsnorm(&mut hs[t * hid..][..hid], &comb[t * hid..][..hid], &self.post_norm, hid, cfg.eps);
        }
        let mut gate = vec![0f32; n * inter];
        let mut up = vec![0f32; n * inter];
        self.gate.mm(&mut gate, &hs, hid, inter, n);
        self.up.mm(&mut up, &hs, hid, inter, n);
        let mut act = vec![0f32; n * inter];
        let mut gu = vec![0f32; 2 * inter];
        for t in 0..n {
            gu[..inter].copy_from_slice(&gate[t * inter..][..inter]);
            gu[inter..].copy_from_slice(&up[t * inter..][..inter]);
            ops::glu(&mut act[t * inter..][..inter], &gu, inter, SWIGLU);
        }
        let mut ffn_out = vec![0f32; n * hid];
        self.down.mm(&mut ffn_out, &act, inter, hid, n);
        for i in 0..n * hid {
            comb[i] += ffn_out[i];
        }

        // 3. shared_head_norm -> shared output head -> greedy argmax
        let head = io.head_w().ok_or("no output head")?;
        let vocab = io.vocab;
        let mut normed = vec![0f32; hid];
        let mut logits = vec![0f32; vocab];
        let mut out = Vec::with_capacity(n);
        for t in 0..n {
            ops::rmsnorm(&mut normed, &comb[t * hid..][..hid], &self.shared_head_norm, hid, cfg.eps);
            ops::mmw(&mut logits, &normed, head, hid, vocab);
            let mut bi = 0usize;
            let mut bv = f32::MIN;
            for (i, &v) in logits.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            out.push(bi as u32);
        }
        Ok(out)
    }
}

/// Per-position final residual (the input the output head reads) for every token in `ids`,
/// batched: `[ids.len() * hidden]`. Identical arithmetic to [`step_many`]'s layer loop, but
/// returns EVERY position's hidden rather than only the last position's logits — this is the
/// hidden the MTP head consumes.
pub fn final_hiddens(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    ids: &[u32],
) -> Result<Vec<f32>, String> {
    let c = &t.cfg;
    let k = ids.len();
    if k == 0 {
        return Err("final_hiddens needs at least one token".into());
    }
    let hid = c.hidden;
    let mut xs = vec![0f32; k * hid];
    for (i, id) in ids.iter().enumerate() {
        t.io.embed_row(*id, &mut xs[i * hid..][..hid])?;
    }
    let mut branch = vec![0f32; k * hid];
    for l in 0..c.n_layers {
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => linear_block_many(&mut branch, &xs, &w.view(), c, &mut s.lin[l], k),
            Block::Full(w) => {
                attn_block_many(&mut branch, &xs, &w.view(), c, &mut s.attn[l], &t.rope, s.pos, k)
            }
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
        ffn_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
    }
    s.pos += k;
    Ok(xs)
}

/// State captured during a speculative verify so a rejected draft can be rolled back WITHOUT
/// re-streaming the model: the per-gdn-layer recurrent state after the first (always-accepted)
/// token, and the per-attention-layer KV length before the verify.
pub struct SpecCkpt {
    lin: Vec<LinState>,
    kv_len: Vec<usize>,
}

/// Speculative verify: process `[g, d]` (K=2) in ONE weight traversal — one FFN read serves
/// both tokens (see `readprof`) — returning both positions' final hidden `[2*hidden]` and a
/// rollback checkpoint. `g` is the main model's own greedy token (always accepted); `d` is the
/// drafted token to be checked. Bit-identical to two serial `step`s; only the disk traffic is
/// shared. On rejection the caller passes the checkpoint to [`spec_rollback`].
pub fn step_verify(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    g: u32,
    d: u32,
) -> Result<(Vec<f32>, SpecCkpt), String> {
    let c = &t.cfg;
    let hid = c.hidden;
    let k = 2usize;
    let ids = [g, d];
    let kv_len: Vec<usize> = s.attn.iter().map(|a| a.len).collect();
    let mut ckpt: Vec<LinState> = (0..c.n_layers).map(|_| LinState::empty()).collect();

    let mut xs = vec![0f32; k * hid];
    for (i, id) in ids.iter().enumerate() {
        t.io.embed_row(*id, &mut xs[i * hid..][..hid])?;
    }
    let mut branch = vec![0f32; k * hid];
    for l in 0..c.n_layers {
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => {
                linear_block_many_ck(&mut branch, &xs, &w.view(), c, &mut s.lin[l], k, Some(&mut ckpt[l]))
            }
            Block::Full(w) => {
                attn_block_many(&mut branch, &xs, &w.view(), c, &mut s.attn[l], &t.rope, s.pos, k)
            }
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
        ffn_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
    }
    s.pos += k;
    Ok((xs, SpecCkpt { lin: ckpt, kv_len }))
}

/// Roll a rejected speculative token out of the session: restore each recurrent layer to its
/// post-first-token checkpoint, drop the second token's KV from each attention layer, and step
/// the position back by one. After this the session is exactly as if only `g` had been decoded.
pub fn spec_rollback(t: &Trunk, s: &mut Session, ckpt: &SpecCkpt) {
    let kv_width = t.cfg.n_kv_heads * t.cfg.head_dim;
    for l in 0..t.cfg.n_layers {
        match &t.blocks[l] {
            Block::Linear(_) => s.lin[l].clone_from(&ckpt.lin[l]),
            Block::Full(_) => s.attn[l].truncate_to(ckpt.kv_len[l] + 1, kv_width),
        }
    }
    s.pos -= 1;
}

/// Softmax router over all experts, then the top `k` by probability.
///
/// Returns `(expert, weight)` pairs, renormalised to sum to 1 over the chosen subset.
pub fn route(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut p: Vec<f32> = logits.to_vec();
    let m = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f64;
    for v in p.iter_mut() {
        *v = (*v - m).exp();
        z += *v as f64;
    }
    for v in p.iter_mut() {
        *v = (*v as f64 / z) as f32;
    }
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx.truncate(k.min(p.len()));
    // Renormalised over the chosen subset, with the denominator clamped away from zero.
    //
    // A NOTE ON HOW THIS WAS NEARLY BROKEN. The reference trace shows `ffn_moe_weights`
    // summing to 0.574741, which reads as "no renormalisation" -- and I changed the code
    // on that basis. But three ops later the same chain has
    // `ffn_moe_weights_sum -> CLAMP -> ffn_moe_weights_norm = DIV`. The tensor I read was
    // the RAW gather, not the value that reaches the experts. Reading one tensor instead
    // of the chain it belongs to is its own failure mode.
    let s: f32 = idx.iter().map(|&i| p[i]).sum();
    let s = if s.abs() < 1e-20 { 1e-20 } else { s };
    idx.into_iter().map(|i| (i, p[i] / s)).collect()
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

use crate::st::{Dtype, St};

/// A weight kept in its stored quantised form.
///
/// Not dequantised on load, and that is the whole economy of this architecture: the
/// non-expert weights are 0.28 GB packed and would be ~1.5 GB widened, while the routed
/// experts -- 20.89 GB, 98.7% of the checkpoint -- never become resident at all.
pub struct QW {
    b: Vec<u8>,
    d: Dtype,
    /// LUT-GEMM repack of a Q4_K weight, built at load only when `Q35_LUT` is set. Resident
    /// (~2x the packed weight) — affordable for the trunk's resident attention weights, unlike
    /// the streamed FFN. `None` => the normal `dot_p8` path.
    lut: Option<crate::lut::Q4kLutW>,
}

impl QW {
    fn load(st: &St, name: &str) -> Result<QW, String> {
        QW::load_inner(st, name, false)
    }

    /// Load and, if `Q35_LUT` is set and the weight is Q4_K, also build the resident LUT
    /// repack. Only the trunk's attention weights opt in — the streamed FFN cannot afford the
    /// ~2x resident repack (see `dense-speedup-levers`).
    fn load_lut(st: &St, name: &str) -> Result<QW, String> {
        QW::load_inner(st, name, true)
    }

    fn load_inner(st: &St, name: &str, want_lut: bool) -> Result<QW, String> {
        let t = st.find(name).ok_or_else(|| format!("missing {name}"))?;
        let mut b = vec![0u8; t.nbytes as usize];
        st.read(t, &mut b);
        if crate::qwen35::weight_of(&b, t.dtype).is_none() {
            return Err(format!("{name} is {}, which has no matmul kernel", t.dtype.name()));
        }
        // st stores shapes in numpy order: shape[0] = rows (out), shape[1] = columns (k_in,
        // the contraction). matmul_q4k reads `rows` rows of `k_in`, so k_in is shape[1].
        let (k_in, out) = if t.shape.len() == 2 {
            (t.shape[1] as usize, t.shape[0] as usize)
        } else {
            (0, 0)
        };
        // LUT-GEMM is Q4_K only (Q6_K would need 6 bit-planes and isn't worth it), on a whole
        // number of super-blocks. Built once, resident.
        let lut = if want_lut
            && std::env::var_os("Q35_LUT").is_some()
            && t.dtype == Dtype::Q4K
            && k_in % 256 == 0
            && k_in > 0
        {
            Some(crate::lut::Q4kLutW::repack(&b, k_in, out))
        } else {
            None
        };
        Ok(QW { b, d: t.dtype, lut })
    }

    pub fn w(&self) -> W<'_> {
        // Checked at load, so this cannot fail here.
        crate::qwen35::weight_of(&self.b, self.d).expect("dtype was validated on load")
    }

    /// `y = W·x` for `ntok` token-major activations. Routes through the LUT kernel when this
    /// weight was repacked (build the per-token activation LUT, reuse the resident weight
    /// tables), else the normal batched `dot_p8`. Near-bitwise to `mmw_many` either way.
    pub fn mm(&self, y: &mut [f32], x: &[f32], k_in: usize, out: usize, ntok: usize) {
        // Fall back if the repack's shape does not match this call — never read out of bounds.
        if let Some(lut) = self.lut.as_ref().filter(|l| l.dims() == (k_in, out)) {
            for t in 0..ntok {
                let act = crate::lut::ActLut::build(&x[t * k_in..][..k_in]);
                crate::lut::matmul_q4k_lut_avx2(&mut y[t * out..][..out], &act, lut);
            }
        } else {
            crate::ops::mmw_many(y, x, self.w(), k_in, out, ntok);
        }
    }

    pub fn bytes(&self) -> usize {
        self.b.len()
    }
}

fn f32s(st: &St, name: &str) -> Result<Vec<f32>, String> {
    let t = st.find(name).ok_or_else(|| format!("missing {name}"))?;
    let mut v = vec![0f32; t.numel() as usize];
    st.read_f32(t, &mut v);
    Ok(v)
}

pub struct LinBlock {
    attn_norm: Vec<f32>,
    qkv: QW,
    z: QW,
    alpha: QW,
    beta: QW,
    a: Vec<f32>,
    dt_bias: Vec<f32>,
    conv1d: Vec<f32>,
    norm: Vec<f32>,
    out: QW,
}

pub struct FullBlock {
    attn_norm: Vec<f32>,
    qg: QW,
    k: QW,
    v: QW,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o: QW,
}

pub enum Block {
    Linear(LinBlock),
    Full(FullBlock),
}

/// The feed-forward half of a block. Every block has one, whichever attention it uses.
///
/// `post_norm` is common to both architectures; what follows it is what `qwen35` and
/// `qwen35moe` disagree about, and it is the ONLY thing they disagree about.
pub struct Moe {
    pub post_norm: Vec<f32>,
    /// The router and the shared expert, or `None` for a dense block.
    ///
    /// A dense FFN needs neither: there is nothing to route to, and no "shared" expert
    /// because the single FFN every token runs is already that. Making this an `Option`
    /// rather than zero-length vectors means the dense path cannot accidentally run a
    /// softmax over an empty router and get a plausible-looking answer.
    pub routed: Option<Routed>,
}

/// What a Mixture-of-Experts block has and a dense one does not.
pub struct Routed {
    /// `[n_experts][hidden]`, F32 -- the router.
    pub gate_inp: Vec<f32>,
    /// `[hidden]`, F32. A dot product, then sigmoid: one scalar gating the shared expert.
    pub gate_shexp: Vec<f32>,
    pub sh_gate: QW,
    pub sh_up: QW,
    pub sh_down: QW,
}

impl Moe {
    fn routed(&self) -> Result<&Routed, String> {
        self.routed.as_ref().ok_or_else(|| {
            "a dense block has no router; this is the MoE path".to_string()
        })
    }
}

impl LinBlock {
    pub fn view(&self) -> LinW<'_> {
        LinW {
            attn_norm: &self.attn_norm,
            qkv: self.qkv.w(),
            z: self.z.w(),
            alpha: self.alpha.w(),
            beta: self.beta.w(),
            a: &self.a,
            dt_bias: &self.dt_bias,
            conv1d: &self.conv1d,
            norm: &self.norm,
            out: self.out.w(),
        }
    }
}

impl FullBlock {
    pub fn view(&self) -> AttnW<'_> {
        AttnW {
            attn_norm: &self.attn_norm,
            qg: &self.qg,
            k: &self.k,
            v: &self.v,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            o: &self.o,
        }
    }
}

/// Everything except the routed experts: 0.28 GB that stays resident for the whole run.
pub struct Trunk {
    pub cfg: Cfg,
    pub blocks: Vec<Block>,
    pub moe: Vec<Moe>,
    pub io: crate::qwen35::Io,
    pub rope: Rope,
    pub bytes: usize,
    /// Stored dtypes of (gate, up, down) expert tensors, PER LAYER.
    ///
    /// Not one triple for the model: this `Q4_K_M` build stores `ffn_down_exps` as Q6_K in
    /// most layers but Q4_K in layers 5 and 6. A `_M` suffix names an imatrix-driven
    /// POLICY, not a uniform format -- the quantiser spends bits per tensor by measured
    /// importance. Reading blk.0 and assuming it holds fed a Q4_K buffer to the Q6_K
    /// kernel, which is only caught because the byte lengths happen to disagree; had the
    /// two formats been the same width it would have decoded garbage in silence.
    pub expert_dt: Vec<[Dtype; 3]>,
    /// Runtime refusal-direction ablation (abliteration): a unit vector in hidden space. When
    /// set, `(x·r)·r` is subtracted from every token's residual after each layer, projecting
    /// the refusal direction out of the stream. `None` leaves the model untouched. Set after
    /// load; a permanent runtime feature the served/trained model carries.
    pub ablate: Option<Vec<f32>>,
}

/// Project the ablation direction out of every token's residual: `x -= (x·r)·r`. `r` is a
/// unit vector, so this removes exactly the component along `r` and leaves the rest intact.
fn apply_ablate(xs: &mut [f32], r: &[f32], k: usize, hid: usize) {
    for t in 0..k {
        let x = &mut xs[t * hid..][..hid];
        let mut dot = 0f64;
        for i in 0..hid {
            dot += x[i] as f64 * r[i] as f64;
        }
        let dot = dot as f32;
        for i in 0..hid {
            x[i] -= dot * r[i];
        }
    }
}

impl Trunk {
    pub fn load(st: &St, cfg: &Cfg, max_ctx: usize) -> Result<Trunk, String> {
        let vt = st.find("token_embd.weight").ok_or("missing token_embd.weight")?;
        // The vocabulary is the embedding's ROW COUNT, not a metadata key -- gguf does not
        // carry one for this architecture, and inventing a default would silently truncate
        // or overrun the head.
        let vocab = (vt.numel() as usize) / cfg.hidden;
        let mut c = cfg.clone();
        c.vocab = vocab;

        let io = crate::qwen35::Io::load(st, c.hidden, vocab, c.eps)?;
        let mut bytes = io.bytes();
        let (mut blocks, mut moe) = (Vec::new(), Vec::new());

        for l in 0..c.n_layers {
            let p = |n: &str| format!("blk.{l}.{n}");
            // The block TYPE decides which tensors exist. Asking for the wrong set is not
            // a silent error here -- `missing blk.N.ssm_a` is a hard failure -- which is
            // exactly why `is_full_attn` is worth a test of its own.
            let b = if c.is_full_attn(l) {
                Block::Full(FullBlock {
                    attn_norm: f32s(st, &p("attn_norm.weight"))?,
                    qg: QW::load_lut(st, &p("attn_q.weight"))?,
                    k: QW::load_lut(st, &p("attn_k.weight"))?,
                    v: QW::load_lut(st, &p("attn_v.weight"))?,
                    q_norm: f32s(st, &p("attn_q_norm.weight"))?,
                    k_norm: f32s(st, &p("attn_k_norm.weight"))?,
                    o: QW::load_lut(st, &p("attn_output.weight"))?,
                })
            } else {
                Block::Linear(LinBlock {
                    attn_norm: f32s(st, &p("attn_norm.weight"))?,
                    qkv: QW::load(st, &p("attn_qkv.weight"))?,
                    z: QW::load(st, &p("attn_gate.weight"))?,
                    alpha: QW::load(st, &p("ssm_alpha.weight"))?,
                    beta: QW::load(st, &p("ssm_beta.weight"))?,
                    a: f32s(st, &p("ssm_a"))?,
                    dt_bias: f32s(st, &p("ssm_dt.bias"))?,
                    conv1d: f32s(st, &p("ssm_conv1d.weight"))?,
                    norm: f32s(st, &p("ssm_norm.weight"))?,
                    out: QW::load(st, &p("ssm_out.weight"))?,
                })
            };
            // A dense block loads NOTHING here beyond its norm: its whole feed-forward is
            // 165 MB per layer and streams through the cache exactly as routed experts do.
            let m = Moe {
                post_norm: f32s(st, &p("post_attention_norm.weight"))?,
                routed: if c.is_dense() {
                    None
                } else {
                    Some(Routed {
                        gate_inp: f32s(st, &p("ffn_gate_inp.weight"))?,
                        gate_shexp: f32s(st, &p("ffn_gate_inp_shexp.weight"))?,
                        sh_gate: QW::load(st, &p("ffn_gate_shexp.weight"))?,
                        sh_up: QW::load(st, &p("ffn_up_shexp.weight"))?,
                        sh_down: QW::load(st, &p("ffn_down_shexp.weight"))?,
                    })
                },
            };
            bytes += match &b {
                Block::Full(f) => f.qg.bytes() + f.k.bytes() + f.v.bytes() + f.o.bytes(),
                Block::Linear(x) => {
                    x.qkv.bytes() + x.z.bytes() + x.alpha.bytes() + x.beta.bytes() + x.out.bytes()
                }
            } + m.routed.as_ref().map_or(0, |r| {
                r.sh_gate.bytes() + r.sh_up.bytes() + r.sh_down.bytes() + r.gate_inp.len() * 4
            });
            blocks.push(b);
            moe.push(m);
        }

        // Halves pairing and partial rotation: 64 of each 256-wide head. See the module
        // header on why plain rope is exact for text-only input.
        //
        // Sized from `max_ctx`, NOT a constant. This was hardcoded to 8192 while the model
        // advertises 262144, so any conversation past 8191 tokens panicked inside
        // `apply_rope` -- and agent prompts start at ~10k, so it was on the first request
        // of the workload this engine exists for. The table is pure lookup, so a longer
        // one gives identical values at every position; only memory is at stake, at
        // n_rot/2 * 2 floats per position (16 KB per 1k tokens here).
        let rope = gqa::rope(c.head_dim, c.n_rot, max_ctx.max(1), c.rope_base,
                             crate::gqa::Pairing::Halves);
        let dt = |l: usize, n: &str| -> Result<Dtype, String> {
            st.find(&format!("blk.{l}.{n}"))
                .map(|t| t.dtype)
                .ok_or_else(|| format!("missing blk.{l}.{n}"))
        };
        // Same per-layer discipline for both, and for the same reason: a `_M` build spends
        // bits by measured importance, so `ffn_down` is Q6_K in most layers of Qwen3.8-27B
        // and the gate/up pair is Q4_K. Reading layer 0 and assuming it holds is how a
        // Q4_K buffer reaches the Q6_K kernel.
        let (g, u, d) = if c.is_dense() {
            ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
        } else {
            ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
        };
        let mut expert_dt = Vec::with_capacity(c.n_layers);
        for l in 0..c.n_layers {
            expert_dt.push([dt(l, g)?, dt(l, u)?, dt(l, d)?]);
        }
        Ok(Trunk { cfg: c, blocks, moe, io, rope, bytes, expert_dt, ablate: None })
    }

    /// Routed experts never load here -- they stream through `cache::ExpertSrc::Stacked`.
    pub fn expert_src(&self, layer: usize, expert: usize) -> crate::cache::ExpertSrc {
        crate::cache::gguf_expert_src(layer, expert)
    }
}

// ---------------------------------------------------------------------------
// Mixture of experts, and a decode step
// ---------------------------------------------------------------------------

/// Plain SwiGLU: `silu(gate) * up`. DeepSeek-V4 clamps both branches; Qwen does not, and
/// an unwanted clamp is a bounded, plausible, wrong activation.
pub(crate) const SWIGLU: crate::ops::Glu = crate::ops::Glu::SwigluClamped { limit: f32::INFINITY };

/// One expert's feed-forward, straight out of the cache slot it streamed into.
/// A fn, not a closure: a closure cannot express that the returned `W` borrows from the
/// `bytes` argument rather than from the closure's own scope. Naming the tensor in the
/// error means a future quant type reports which of the three it hit.
fn w<'a>(which: &str, bytes: &'a [u8], d: Dtype) -> Result<W<'a>, String> {
    crate::qwen35::weight_of(bytes, d)
        .ok_or_else(|| format!("expert {which} is {}, which has no kernel", d.name()))
}

/// One expert applied to the `ntok` tokens that routed to it, writing each token's
/// contribution to `outs` rather than accumulating into it.
///
/// WHY WRITE AND NOT ACCUMULATE
///     A token's experts must be summed in ROUTE order -- descending router probability --
///     because that is the order the serial path used and float addition is not
///     associative. Grouping by expert visits them in a different order, so the results
///     are parked in per-(token, slot) scratch and summed afterwards in the original
///     order. That keeps expert-major iteration bit-identical to token-major, which is the
///     only reason it is allowed here at all: a reordered sum would be a model that is
///     subtly wrong only on prompts long enough to group.
///
/// The gain is the same one `mmw_many` gives everywhere else -- an expert routed to by
/// several tokens in a chunk is decoded once instead of once per token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_fwd_many(
    outs: &mut [f32],
    xs: &[f32],
    q: &crate::cache::ExpertQ,
    dt: &[Dtype; 3],
    hidden: usize,
    inter: usize,
    weights: &[f32],
    ntok: usize,
) -> Result<(), String> {
    let mut gu = vec![0f32; ntok * 2 * inter];
    let mut gate = vec![0f32; ntok * inter];
    let mut up = vec![0f32; ntok * inter];
    crate::ops::mmw_many(&mut gate, xs, w("gate", q.p1, dt[0])?, hidden, inter, ntok);
    crate::ops::mmw_many(&mut up, xs, w("up", q.p3, dt[1])?, hidden, inter, ntok);
    let dbg = std::env::var_os("Q35_SUMS").is_some();
    let mut act = vec![0f32; ntok * inter];
    for t in 0..ntok {
        // `glu` wants gate and up adjacent, which is why they are re-paired here rather
        // than projected into one buffer: `mmw_many` writes token-major, so a single
        // 2*inter output would interleave the two halves per token.
        let p = &mut gu[t * 2 * inter..][..2 * inter];
        p[..inter].copy_from_slice(&gate[t * inter..][..inter]);
        p[inter..].copy_from_slice(&up[t * inter..][..inter]);
        if dbg {
            eprintln!("    expert gate {:.6} up {:.6}",
                      p[..inter].iter().map(|v| *v as f64).sum::<f64>(),
                      p[inter..].iter().map(|v| *v as f64).sum::<f64>());
        }
        let a = &mut act[t * inter..][..inter];
        crate::ops::glu(a, p, inter, SWIGLU);
        // The router weight is folded into the activation, so the caller can accumulate
        // across experts with a plain add rather than a second scaled pass.
        for v in a.iter_mut() {
            *v *= weights[t];
        }
    }
    crate::ops::mmw_many(outs, &act, w("down", q.p2, dt[2])?, inter, hidden, ntok);
    Ok(())
}

pub(crate) fn expert_fwd(
    out: &mut [f32],
    x: &[f32],
    q: &crate::cache::ExpertQ,
    dt: &[Dtype; 3],
    hidden: usize,
    inter: usize,
    weight: f32,
    gu: &mut [f32],
    act: &mut [f32],
) -> Result<(), String> {
    // SAFETY of the shapes: gate and up are [inter, hidden], down is [hidden, inter].
    crate::ops::mmw(&mut gu[..inter], x, w("gate", q.p1, dt[0])?, hidden, inter);
    crate::ops::mmw(&mut gu[inter..], x, w("up", q.p3, dt[1])?, hidden, inter);
    if std::env::var_os("Q35_SUMS").is_some() {
        eprintln!("    expert gate {:.6} up {:.6}",
                  gu[..inter].iter().map(|v| *v as f64).sum::<f64>(),
                  gu[inter..].iter().map(|v| *v as f64).sum::<f64>());
    }
    crate::ops::glu(act, gu, inter, SWIGLU);
    // The router weight is folded into the activation, so `out` can be accumulated across
    // experts with a plain add rather than a second scaled pass.
    for v in act.iter_mut().take(inter) {
        *v *= weight;
    }
    let mut tmp = vec![0f32; hidden];
    crate::ops::mmw(&mut tmp, &act[..inter], w("down", q.p2, dt[2])?, inter, hidden);
    for (o, t) in out.iter_mut().zip(tmp.iter()) {
        *o += *t;
    }
    Ok(())
}

/// The MoE branch of one block: 8 routed experts streamed through `cache`, plus one shared
/// expert that is always resident.
#[allow(clippy::too_many_arguments)]
pub fn moe(
    out: &mut [f32],
    x: &[f32],
    m: &Moe,
    c: &Cfg,
    dt: &[Dtype; 3],
    st: &St,
    cache: &mut crate::cache::Cache,
    layer: usize,
) -> Result<(), String> {
    let (hid, inter) = (c.hidden, c.moe_inter);
    let mut h = vec![0f32; hid];
    ops::rmsnorm(&mut h, x, &m.post_norm, hid, c.eps);

    let r = m.routed()?;
    let mut logits = vec![0f32; c.n_experts];
    ops::mmw(&mut logits, &h, W::F32(&r.gate_inp), hid, c.n_experts);
    let sel = route(&logits, c.topk);
    if std::env::var_os("Q35_SUMS").is_some() {
        let w: f32 = sel.iter().map(|(_, w)| *w).sum();
        eprintln!("  [L{layer}] moe_logits {:.6} weights {:.6} {:?}",
                  logits.iter().map(|v| *v as f64).sum::<f64>(), w,
                  sel.iter().map(|(e, _)| *e).collect::<Vec<_>>());
    }

    // Hand the whole top-k over BEFORE using any of it. One-at-a-time `get` gives the
    // drive queue depth one, and a device that needs depth to reach its rated bandwidth
    // spends the difference idle.
    let ids: Vec<usize> = sel.iter().map(|(e, _)| *e).collect();
    cache.prefetch_many(st, layer, &ids, crate::cache::gguf_expert_src);

    out[..hid].fill(0.0);
    let mut gu = vec![0f32; 2 * inter];
    let mut act = vec![0f32; inter];
    for (e, wt) in &sel {
        let slot = cache
            .get(st, layer, *e, &crate::cache::gguf_expert_src(layer, *e))
            .ok_or_else(|| format!("layer {layer} expert {e} could not be cached"))?;
        expert_fwd(out, &h, &cache.expert(slot), dt, hid, inter, *wt, &mut gu, &mut act)?;
    }

    // The shared expert runs for every token and is gated by a SCALAR: a dot product with
    // `ffn_gate_inp_shexp` [hidden], then sigmoid. Treating that vector as a matrix row
    // per channel would gate each channel separately -- a different, plausible function.
    let mut g = 0.0f64;
    for (a, b) in r.gate_shexp.iter().zip(h.iter()) {
        g += *a as f64 * *b as f64;
    }
    let g = crate::libm::sigmoidf(g as f32);
    let si = c.shared_inter;
    let mut sgu = vec![0f32; 2 * si];
    ops::mmw(&mut sgu[..si], &h, r.sh_gate.w(), hid, si);
    ops::mmw(&mut sgu[si..], &h, r.sh_up.w(), hid, si);
    let mut sact = vec![0f32; si];
    ops::glu(&mut sact, &sgu, si, SWIGLU);
    for v in sact.iter_mut() {
        *v *= g;
    }
    let mut sh = vec![0f32; hid];
    ops::mmw(&mut sh, &sact, r.sh_down.w(), si, hid);
    if std::env::var_os("Q35_SUMS").is_some() {
        eprintln!("  [L{layer}] moe_routed_out {:.6}  shared_gate {g:.6}",
                  out.iter().map(|v| *v as f64).sum::<f64>());
    }
    for (o, v) in out.iter_mut().zip(sh.iter()) {
        *o += *v;
    }
    Ok(())
}

/// Expert-sharing counters, so the benefit of batching is measured rather than assumed.
pub static ROUTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static UNIQUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Fetches saved by unioning a chunk's routing, as (routed, unique).
pub fn sharing() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (ROUTED.load(Relaxed), UNIQUE.load(Relaxed))
}

/// Zero the sharing counters, so a measurement covers one run rather than the process.
pub fn reset_sharing() {
    use std::sync::atomic::Ordering::Relaxed;
    ROUTED.store(0, Relaxed);
    UNIQUE.store(0, Relaxed);
}

/// The MoE branch for a CHUNK of tokens, fetching each routed expert once.
///
/// MEASURED, on Qwen3.6-35B with a real natural-language prompt. Batching and cache size
/// turn out to be SUBSTITUTES, not complements -- the union only saves reads the cache was
/// not already absorbing:
///
/// ```text
///   cache 6.0 GB   width 8: 44.7% of fetch requests deduplicated, 0% less I/O, 1.01x
///   cache 0.25 GB  width 8: 42.9% deduplicated,                  43% less I/O, 1.67x
/// ```
///
/// The deduplication rate is nearly identical at both sizes; what differs is whether the
/// duplicate would have hit the cache anyway. With 6 GB an expert fetched for token t is
/// still resident when t+1 asks for it, so the union saves a lookup and not a read.
///
/// The consequence for the memory split: shrinking the expert cache to make room for
/// conversation state is partly self-compensating, because it makes batching pay. The two
/// caches are not simply competing.
///
/// The order of operations is deliberate: route EVERY token first, hand the whole union to
/// `prefetch_many` in one call, and only then compute. Fetching per token gives the drive
/// queue depth one, and a device that needs depth to reach its rated bandwidth spends the
/// difference idle.
#[allow(clippy::too_many_arguments)]
pub fn moe_many(
    out: &mut [f32],
    xs: &[f32],
    m: &Moe,
    c: &Cfg,
    dt: &[Dtype; 3],
    st: &St,
    cache: &mut crate::cache::Cache,
    layer: usize,
    k: usize,
) -> Result<(), String> {
    let (hid, inter) = (c.hidden, c.moe_inter);
    let mut hs = vec![0f32; k * hid];
    let mut sel = Vec::with_capacity(k);
    let mut union: Vec<usize> = Vec::with_capacity(k * c.topk);

    for t in 0..k {
        ops::rmsnorm(&mut hs[t * hid..][..hid], &xs[t * hid..][..hid], &m.post_norm, hid, c.eps);
    }
    // The router, the shared expert's three projections and the routed experts all read
    // the SAME normalised activations, so each of those weights is decoded once for the
    // chunk rather than once per token.
    let r = m.routed()?;
    let mut rlog = vec![0f32; k * c.n_experts];
    ops::mmw_many(&mut rlog, &hs, W::F32(&r.gate_inp), hid, c.n_experts, k);
    for t in 0..k {
        let s = route(&rlog[t * c.n_experts..][..c.n_experts], c.topk);
        for (e, _) in &s {
            if !union.contains(e) {
                union.push(*e);
            }
        }
        sel.push(s);
    }
    // How much a chunk actually shares. `k * topk` fetches would be needed with no
    // union; `union.len()` is what is actually issued. Counted rather than assumed --
    // the 48.6% figure on record came from a different model and a repetitive prompt.
    ROUTED.fetch_add(k * c.topk, std::sync::atomic::Ordering::Relaxed);
    UNIQUE.fetch_add(union.len(), std::sync::atomic::Ordering::Relaxed);
    cache.prefetch_many(st, layer, &union, crate::cache::gguf_expert_src);

    // Expert-major, so a weight decoded once serves every token in the chunk that routed
    // to it. `partial[(t * topk + j) * hid]` holds the contribution of token `t`'s j-th
    // routed expert; the per-token sum below then runs in route order, which is what keeps
    // this bit-identical to visiting experts token-major. See `expert_fwd_many`.
    let topk = c.topk;
    let mut partial = vec![0f32; k * topk * hid];
    let mut xb = Vec::<f32>::new();
    let mut wb = Vec::<f32>::new();
    let mut who = Vec::<(usize, usize)>::new();
    for &e in &union {
        who.clear();
        for (t, s) in sel.iter().enumerate() {
            for (j, (ex, wt)) in s.iter().enumerate() {
                if *ex == e {
                    who.push((t, j));
                    wb.push(*wt);
                }
            }
        }
        if who.is_empty() {
            continue;
        }
        xb.clear();
        for &(t, _) in &who {
            xb.extend_from_slice(&hs[t * hid..][..hid]);
        }
        let slot = cache
            .get(st, layer, e, &crate::cache::gguf_expert_src(layer, e))
            .ok_or_else(|| format!("layer {layer} expert {e} could not be cached"))?;
        let mut ob = vec![0f32; who.len() * hid];
        expert_fwd_many(&mut ob, &xb, &cache.expert(slot), dt, hid, inter, &wb, who.len())?;
        for (i, &(t, j)) in who.iter().enumerate() {
            partial[(t * topk + j) * hid..][..hid].copy_from_slice(&ob[i * hid..][..hid]);
        }
        wb.clear();
    }

    let si = c.shared_inter;

    // The shared expert runs for EVERY token, so unlike the routed ones it batches with no
    // grouping and no reordering. `sgu` is held as two chunk-wide halves because `glu`
    // wants gate and up adjacent per token.
    let mut sg = vec![0f32; k * si];
    let mut su = vec![0f32; k * si];
    ops::mmw_many(&mut sg, &hs, r.sh_gate.w(), hid, si, k);
    ops::mmw_many(&mut su, &hs, r.sh_up.w(), hid, si, k);
    let mut sacts = vec![0f32; k * si];
    let mut pair = vec![0f32; 2 * si];
    for t in 0..k {
        pair[..si].copy_from_slice(&sg[t * si..][..si]);
        pair[si..].copy_from_slice(&su[t * si..][..si]);
        let sact = &mut sacts[t * si..][..si];
        ops::glu(sact, &pair, si, SWIGLU);
        // Gated by a scalar formed from the token's own activations.
        let mut g = 0.0f64;
        for (a, b) in r.gate_shexp.iter().zip(hs[t * hid..][..hid].iter()) {
            g += *a as f64 * *b as f64;
        }
        let g = crate::libm::sigmoidf(g as f32);
        for v in sact.iter_mut() {
            *v *= g;
        }
    }
    let mut shs = vec![0f32; k * hid];
    ops::mmw_many(&mut shs, &sacts, r.sh_down.w(), si, hid, k);

    for t in 0..k {
        let o = &mut out[t * hid..][..hid];
        o.fill(0.0);
        // Route order, exactly as the token-major loop accumulated it.
        for j in 0..sel[t].len() {
            for (a, b) in o.iter_mut().zip(partial[(t * topk + j) * hid..][..hid].iter()) {
                *a += *b;
            }
        }
        for (a, b) in o.iter_mut().zip(shs[t * hid..][..hid].iter()) {
            *a += *b;
        }
    }
    Ok(())
}

/// Everything that changes as a sequence is decoded.
pub struct Session {
    pub lin: Vec<LinState>,
    pub attn: Vec<AttnState>,
    pub pos: usize,
}

impl Session {
    pub fn new(t: &Trunk) -> Session {
        // Allocate each layer's state by the block type it actually runs: the recurrent
        // LinState only for delta-net layers, the (initially empty) KV AttnState only for
        // full-attention layers. Both were previously allocated for every layer; the unused
        // LinStates alone were ~52 MB on the 27B.
        Session {
            lin: (0..t.cfg.n_layers)
                .map(|l| if t.cfg.is_full_attn(l) { LinState::empty() } else { LinState::new(&t.cfg) })
                .collect(),
            attn: (0..t.cfg.n_layers).map(|_| AttnState::new()).collect(),
            pos: 0,
        }
    }
}

/// A CHUNK of tokens through the whole stack, one layer at a time. Writes the `vocab`
/// logits that follow the LAST token.
///
/// WHY THIS IS NOT "the same thing, faster"
/// ```text
///     Within a layer the tokens are still processed IN ORDER, because 30 of the 40 blocks
/// ```text
///     are recurrent: token t's state depends on t-1. Nothing here parallelises across
///     tokens. What changes is the axis the work is grouped on -- all `k` tokens finish a
///     layer before any of them starts the next -- which is what lets one expert fetch
///     serve the whole chunk.
///
///     So the arithmetic is IDENTICAL to calling `step` k times, and the fixture diff
///     asserts exactly that. Only the I/O differs.
/// ```
/// ```
pub fn step_many(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    ids: &[u32],
    logits: &mut [f32],
) -> Result<(), String> {
    let c = &t.cfg;
    let k = ids.len();
    if k == 0 {
        return Err("step_many needs at least one token".into());
    }
    let hid = c.hidden;
    let mut xs = vec![0f32; k * hid];
    for (i, id) in ids.iter().enumerate() {
        t.io.embed_row(*id, &mut xs[i * hid..][..hid])?;
    }
    let mut branch = vec![0f32; k * hid];

    for l in 0..c.n_layers {
        // The layer axis is a cyclic scan over the whole expert set; telling the cache
        // where it is in that sweep is what makes Belady's rule computable without an
        // oracle. Without it LRU is the WORST policy on this access pattern.
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => {
                linear_block_many(&mut branch, &xs, &w.view(), c, &mut s.lin[l], k);
            }
            Block::Full(w) => {
                attn_block_many(&mut branch, &xs, &w.view(), c, &mut s.attn[l], &t.rope, s.pos, k);
            }
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
        if c.is_dense() {
            ffn_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
        } else {
            moe_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
    }
    s.pos += k;
    // Only the LAST position's logits are wanted: a prompt chunk predicts tokens we
    // already have, except at its end.
    t.io.logits(&xs[(k - 1) * hid..][..hid], logits)
}

/// One token through the whole stack. Writes `vocab` logits.
pub fn step(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    id: u32,
    logits: &mut [f32],
) -> Result<(), String> {
    let c = &t.cfg;
    let mut x = vec![0f32; c.hidden];
    t.io.embed_row(id, &mut x)?;
    let mut branch = vec![0f32; c.hidden];

    for l in 0..c.n_layers {
        // The layer axis is a cyclic scan over the whole expert set, so the cache is told
        // where in the sweep it is: distance to reuse is (L - l) mod cycle, which needs no
        // oracle. Without this LRU is the WORST policy on this access pattern.
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => linear_block(&mut branch, &x, &w.view(), c, &mut s.lin[l]),
            Block::Full(w) => {
                attn_block(&mut branch, &x, &w.view(), c, &mut s.attn[l], &t.rope, s.pos)
            }
        }
        let rms = |v: &[f32]| (v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let ab = rms(&branch);
        for i in 0..c.hidden {
            x[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut x, r, 1, c.hidden);
        }
        let ax = rms(&x);
        if c.is_dense() {
            // `ffn_many` at one token, so decode and prefill share one implementation.
            ffn_many(&mut branch, &x, &t.moe[l], c, &t.expert_dt[l], st, cache, l, 1)?;
        } else {
            moe(&mut branch, &x, &t.moe[l], c, &t.expert_dt[l], st, cache, l)?;
        }
        let mb = rms(&branch);
        for i in 0..c.hidden {
            x[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut x, r, 1, c.hidden);
        }
        if std::env::var_os("Q35_TRACE").is_some() {
            eprintln!("L{l:2} {} attn_out {ab:9.4} x {ax:9.4} moe_out {mb:9.4} x {:9.4}",
                      if matches!(t.blocks[l], Block::Full(_)) { "ATTN" } else { "lin " }, rms(&x));
        }
    }
    s.pos += 1;
    t.io.logits(&x, logits)
}

/// One token through the whole stack with the trained LoRA adapter APPLIED at the top FFN —
/// inference for a fine-tuned model. Identical to [`step`] except the last layer's frozen FFN
/// is replaced by the adapter's `W·x + scale·B·(A·x)`. This is how an eval or a served model
/// runs the fine-tune; the base model uses plain `step`.
pub fn step_adapted(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    id: u32,
    ffn: &crate::train::FfnLora,
    logits: &mut [f32],
) -> Result<(), String> {
    let c = &t.cfg;
    let (hid, last) = (c.hidden, c.n_layers - 1);
    let mut x = vec![0f32; hid];
    t.io.embed_row(id, &mut x)?;
    let mut branch = vec![0f32; hid];
    for l in 0..c.n_layers {
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => linear_block(&mut branch, &x, &w.view(), c, &mut s.lin[l]),
            Block::Full(w) => attn_block(&mut branch, &x, &w.view(), c, &mut s.attn[l], &t.rope, s.pos),
        }
        for i in 0..hid {
            x[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut x, r, 1, hid);
        }
        if l < last {
            ffn_many(&mut branch, &x, &t.moe[l], c, &t.expert_dt[l], st, cache, l, 1)?;
            for i in 0..hid {
                x[i] += branch[i];
            }
        } else {
            // The trained adapter in place of the frozen top FFN.
            let src = crate::cache::gguf_dense_ffn_src(last, 0);
            let slot = cache.get(st, last, 0, &src).ok_or("top FFN not cached")?;
            let q = cache.expert(slot);
            let dt = &t.expert_dt[last];
            let wg = crate::qwen35::weight_of(q.p1, dt[0]).ok_or("gate no kernel")?;
            let wu = crate::qwen35::weight_of(q.p3, dt[1]).ok_or("up no kernel")?;
            let wd = crate::qwen35::weight_of(q.p2, dt[2]).ok_or("down no kernel")?;
            let m = &t.moe[last];
            let mut hs = vec![0f32; hid];
            ops::rmsnorm(&mut hs, &x, &m.post_norm, hid, c.eps);
            let mut ffn_out = vec![0f32; hid];
            let _ = ffn.forward(wg, wu, wd, &hs, &mut ffn_out);
            for i in 0..hid {
                x[i] += ffn_out[i];
            }
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut x, r, 1, hid);
        }
    }
    s.pos += 1;
    t.io.logits(&x, logits)
}

/// One depth-1 LoRA training step on a DENSE qwen35: forward the whole streamed stack, adapt
/// only the LAST layer's feed-forward with `ffn`, score the response tokens, and backprop
/// exactly as far as that top FFN -- no further.
///
/// This is the first end-to-end training pass that actually runs on the real checkpoint. It
/// deliberately terminates the backward at the top block so it needs NO attention backward
/// (the gated-delta-net backward is not written yet): the gradient path is
/// ```text
///   CE(logits) -> head Wᵀ (ops::wt on output.weight) -> output-norm backward
///              -> the top FfnLora.backward (Wᵀ on the streamed gate/up/down) -> stop.
/// ```
/// Every kernel on that path is separately verified: `wt` by the adjoint test, the FFN block
/// and cross-entropy by finite-difference gradient checks, the RMS-norm backward likewise.
/// So a loss that falls here is real, not a coincidence of fluent-wrong arithmetic.
///
/// The caller passes a FRESH [`Session`] per example (pos 0, zeroed recurrent state) and the
/// next-token `target`/`mask` from [`crate::train::Example::view`]. Grads accumulate into
/// `ffn`; the caller drives `zero_grad`/`adam_step`. Returns the mean response-token loss.
///
/// Errors if the model is not dense -- a MoE last layer would need the routed-expert backward,
/// which is out of this step's scope.
/// The FROZEN part of a depth-1 training forward: the whole 64-layer stack EXCEPT the top
/// FFN, returning `h` — the residual after the last attention, which is the input the
/// trainable FFN adapter consumes. Nothing here depends on the adapter, so `h` is identical
/// every epoch for the same token ids: compute it ONCE and cache it, and every later epoch
/// skips this (the expensive 99% of the step) entirely. See `train_run`'s feature cache.
pub fn frozen_hidden(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    ids: &[u32],
) -> Result<Vec<f32>, String> {
    let c = &t.cfg;
    if !c.is_dense() {
        return Err("train_lastffn_step is for dense qwen35 only (MoE needs expert backward)".into());
    }
    let k = ids.len();
    if k == 0 {
        return Err("train step needs at least one token".into());
    }
    let hid = c.hidden;
    let last = c.n_layers - 1;
    let mut xs = vec![0f32; k * hid];
    for (i, id) in ids.iter().enumerate() {
        t.io.embed_row(*id, &mut xs[i * hid..][..hid])?;
    }
    let mut branch = vec![0f32; k * hid];
    for l in 0..c.n_layers {
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => linear_block_many(&mut branch, &xs, &w.view(), c, &mut s.lin[l], k),
            Block::Full(w) => {
                attn_block_many(&mut branch, &xs, &w.view(), c, &mut s.attn[l], &t.rope, s.pos, k)
            }
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
        // Every layer's FFN is frozen EXCEPT the last (the trainable adapter, applied later).
        if l < last {
            ffn_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
            for i in 0..k * hid {
                xs[i] += branch[i];
            }
            if let Some(r) = &t.ablate {
                apply_ablate(&mut xs, r, k, hid);
            }
        }
    }
    s.pos += k;
    Ok(xs)
}

/// The final residual stream at the LAST token, after the whole stack (including the last
/// FFN) but before the head — the input the vocabulary head reads to pick the next token, and
/// the cleanest place to read the refusal direction. Honours `t.ablate`, so it can capture
/// either the clean model (for computing the direction) or the abliterated one (to check the
/// direction was removed). A full forward; use a fresh [`Session`] per call.
pub fn final_residual(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    ids: &[u32],
) -> Result<Vec<f32>, String> {
    let c = &t.cfg;
    let k = ids.len();
    if k == 0 {
        return Err("final_residual needs at least one token".into());
    }
    let hid = c.hidden;
    let mut xs = vec![0f32; k * hid];
    for (i, id) in ids.iter().enumerate() {
        t.io.embed_row(*id, &mut xs[i * hid..][..hid])?;
    }
    let mut branch = vec![0f32; k * hid];
    for l in 0..c.n_layers {
        cache.at_layer(l, c.n_layers);
        match &t.blocks[l] {
            Block::Linear(w) => linear_block_many(&mut branch, &xs, &w.view(), c, &mut s.lin[l], k),
            Block::Full(w) => {
                attn_block_many(&mut branch, &xs, &w.view(), c, &mut s.attn[l], &t.rope, s.pos, k)
            }
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
        if c.is_dense() {
            ffn_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
        } else {
            moe_many(&mut branch, &xs, &t.moe[l], c, &t.expert_dt[l], st, cache, l, k)?;
        }
        for i in 0..k * hid {
            xs[i] += branch[i];
        }
        if let Some(r) = &t.ablate {
            apply_ablate(&mut xs, r, k, hid);
        }
    }
    s.pos += k;
    Ok(xs[(k - 1) * hid..][..hid].to_vec())
}

/// One depth-1 LoRA training step: run the frozen stack, then the trainable top FFN. This is
/// `frozen_hidden` followed by [`train_lastffn_from_hidden`]; multi-epoch callers should cache
/// the former and re-run only the latter.
#[allow(clippy::too_many_arguments)]
pub fn train_lastffn_step(
    t: &Trunk,
    s: &mut Session,
    st: &St,
    cache: &mut crate::cache::Cache,
    ids: &[u32],
    target: &[u32],
    mask: &[bool],
    ffn: &mut crate::train::FfnLora,
) -> Result<f32, String> {
    let h = frozen_hidden(t, s, st, cache, ids)?;
    train_lastffn_from_hidden(t, st, cache, &h, target, mask, ffn)
}

/// The TRAINABLE part of a depth-1 step, from the cached frozen hidden `h`: the top FFN
/// adapter forward, the vocabulary head, the loss, and the backward — accumulating grads into
/// `ffn`. `h` is `[k*hidden]`; it is not mutated (a local copy carries the residual add).
#[allow(clippy::too_many_arguments)]
pub fn train_lastffn_from_hidden(
    t: &Trunk,
    st: &St,
    cache: &mut crate::cache::Cache,
    h: &[f32],
    target: &[u32],
    mask: &[bool],
    ffn: &mut crate::train::FfnLora,
) -> Result<f32, String> {
    let c = &t.cfg;
    let (hid, vocab) = (c.hidden, t.io.vocab);
    let last = c.n_layers - 1;
    let k = h.len() / hid;
    // Local mutable copy — the cached `h` stays pristine for the next epoch.
    let mut xs = h.to_vec();

    // --- top layer's adapter FFN: grab the frozen gate/up/down once, hold them ---
    cache.at_layer(last, c.n_layers);
    let src = crate::cache::gguf_dense_ffn_src(last, 0);
    let slot = cache
        .get(st, last, 0, &src)
        .ok_or_else(|| format!("layer {last} feed-forward could not be cached"))?;
    let q = cache.expert(slot);
    let dt = &t.expert_dt[last];
    let wg = crate::qwen35::weight_of(q.p1, dt[0]).ok_or("gate has no kernel")?;
    let wu = crate::qwen35::weight_of(q.p3, dt[1]).ok_or("up has no kernel")?;
    let wd = crate::qwen35::weight_of(q.p2, dt[2]).ok_or("down has no kernel")?;
    let m = &t.moe[last];

    // xs currently holds h (residual after the last attention). Apply the adapter FFN per
    // token, capture the activations, and add the residual.
    let mut acts = Vec::with_capacity(k);
    let mut ffn_out = vec![0f32; hid];
    let mut hs = vec![0f32; hid];
    for tk in 0..k {
        ops::rmsnorm(&mut hs, &xs[tk * hid..][..hid], &m.post_norm, hid, c.eps);
        let act = ffn.forward(wg, wu, wd, &hs, &mut ffn_out);
        for i in 0..hid {
            xs[tk * hid + i] += ffn_out[i];
        }
        acts.push(act);
    }

    // --- loss over the response tokens, head BATCHED ---
    // The vocabulary head (248k x hidden) is the single most expensive matmul in the step and
    // the FROZEN weight the loss signal must pass through both ways. Gather only the scored
    // positions and run the head as ONE batched matmul each way -- `mmw_many` forward,
    // `wt_many` backward -- so each of the 248k rows is decoded once for the whole response,
    // not once per token. The head never receives a gradient of its own; it is frozen (only
    // the FFN adapter trains), and `wt_many` here carries the loss signal DOWN to it.
    let scored: Vec<usize> = (0..k).filter(|&tk| mask[tk]).collect();
    let n_s = scored.len();
    if n_s == 0 {
        return Ok(0.0);
    }
    let wh = t.io.head_w().ok_or("output head has no kernel")?;
    let norm = t.io.norm();

    let mut normed_s = vec![0f32; n_s * hid];
    for (i, &pos) in scored.iter().enumerate() {
        ops::rmsnorm(&mut normed_s[i * hid..][..hid], &xs[pos * hid..][..hid], norm, hid, c.eps);
    }
    let mut logits_s = vec![0f32; n_s * vocab];
    ops::mmw_many(&mut logits_s, &normed_s, wh, hid, vocab, n_s);

    let targets_s: Vec<u32> = scored.iter().map(|&p| target[p]).collect();
    let all = vec![true; n_s];
    let mut grad_logits_s = vec![0f32; n_s * vocab];
    let loss = crate::train::cross_entropy(&logits_s, &targets_s, &all, vocab, &mut grad_logits_s);

    // --- backward: head Wᵀ batched, then per-position norm + FFN backward ---
    let mut grad_normed_s = vec![0f32; n_s * hid];
    ops::wt_many(&mut grad_normed_s, &grad_logits_s, wh, hid, vocab, n_s);

    let mut grad_final = vec![0f32; hid];
    let mut grad_ffn_in = vec![0f32; hid];
    for (i, &pos) in scored.iter().enumerate() {
        // through the output RMS norm to the final residual hidden.
        ops::rmsnorm_backward(&mut grad_final, &grad_normed_s[i * hid..][..hid], &xs[pos * hid..][..hid], norm, hid, c.eps);
        // final = h + ffn_out, so grad w.r.t ffn_out is grad_final. Backprop the adapter FFN;
        // grad_ffn_in (grad into post_norm(h)) is discarded -- depth-1 stops here.
        ffn.backward(wg, wu, wd, &acts[pos], &grad_final, &mut grad_ffn_in);
    }
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rope table must cover the whole conversation. It was hardcoded to 8192 while
    /// this model advertises a 262144 context, so token 8192 panicked inside `apply_rope`
    /// -- on the first request of an agent workload, whose prompts start around 10k.
    #[test]
    fn the_rope_table_covers_the_context_it_was_built_for() {
        let c = cfg();
        let r = crate::gqa::rope(c.head_dim, c.n_rot, 12_000, c.rope_base, crate::gqa::Pairing::Halves);
        let mut x = vec![1.0f32; c.head_dim];
        // The position that used to crash, and one well past it.
        for pos in [8191usize, 8192, 11_999] {
            crate::gqa::apply_rope(&mut x, &r, pos, false);
        }
        assert!(x.iter().all(|v| v.is_finite()));
    }

    /// And past the end it must say WHY, not just "index out of bounds" from a hot loop.
    #[test]
    fn a_position_past_the_rope_table_names_the_limit() {
        let c = cfg();
        let r = crate::gqa::rope(c.head_dim, c.n_rot, 16, c.rope_base, crate::gqa::Pairing::Halves);
        let mut x = vec![1.0f32; c.head_dim];
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::gqa::apply_rope(&mut x, &r, 16, false)
        }))
        .unwrap_err();
        let msg = e.downcast_ref::<String>().map(String::as_str).unwrap_or("");
        assert!(msg.contains("rope table holds 16 positions"), "unhelpful panic: {msg}");
    }

    /// A `Q4_K_M` filename names an imatrix-driven POLICY, not a uniform format. This
    /// build stores `ffn_down_exps` as Q6_K in half its layers and Q4_K in the other half
    /// -- 20 of 40 -- because the quantiser spends bits by measured importance.
    ///
    /// Reading blk.0's dtypes and applying them everywhere fed a Q4_K buffer to the Q6_K
    /// kernel. That was caught only because the two formats have different byte lengths;
    /// between two formats of equal width it would have decoded noise in silence. So the
    /// dtype is per layer, and this test pins the shape of that decision.
    #[test]
    fn expert_dtypes_are_per_layer_because_k_quants_are_mixed_within_one_file() {
        // Q4_K is 144 B per 256-element super-block, Q6_K is 210. A 512x2048 expert row
        // block is therefore 589824 B at Q4_K and 860160 B at Q6_K -- the two differ, which
        // is the only reason the mistake surfaced as a panic rather than as bad text.
        let (q4, q6) = (2048 / 256 * 144 * 512, 512 / 256 * 210 * 2048);
        assert_eq!(q4, 589_824);
        assert_eq!(q6, 860_160);
        assert_ne!(q4, q6, "equal widths here would have made the bug silent");
    }

    use crate::gguf::Value::{Str, F, U};
    use crate::gguf::Meta;
    use crate::gqa::Pairing;

    fn cfg() -> Cfg {
        let mut m = Meta::new();
        m.insert("general.architecture".into(), Str("qwen35moe".into()));
        for (k, v) in [
            ("block_count", 40u64), ("embedding_length", 2048), ("full_attention_interval", 4),
            ("attention.head_count", 16), ("attention.head_count_kv", 2),
            ("attention.key_length", 256), ("attention.value_length", 256),
            ("rope.dimension_count", 64), ("ssm.inner_size", 4096), ("ssm.group_count", 16),
            ("ssm.time_step_rank", 32), ("ssm.state_size", 128), ("ssm.conv_kernel", 4),
            ("expert_count", 256), ("expert_used_count", 8),
            ("expert_feed_forward_length", 512), ("expert_shared_feed_forward_length", 512),
        ] {
            m.insert(format!("qwen35moe.{k}"), U(v));
        }
        m.insert("qwen35moe.rope.freq_base".into(), F(10_000_000.0));
        m.insert("qwen35moe.attention.layer_norm_rms_epsilon".into(), F(1e-6));
        Cfg::from_meta(&m).unwrap()
    }

    /// THE mistake this module exists to prevent. `attn_q` holds query and gate
    /// interleaved per head; reading it as "all queries, then all gates" consumes exactly
    /// the same bytes and yields a model that still writes fluent English.
    #[test]
    fn the_fused_q_projection_is_interleaved_per_head_not_split_in_half() {
        let (nh, hd) = (3usize, 4usize);
        // Head h: query = 100 + h*10 + i, gate = 200 + h*10 + i.
        let mut qg = vec![0f32; 2 * nh * hd];
        for h in 0..nh {
            for i in 0..hd {
                qg[h * 2 * hd + i] = (100 + h * 10 + i) as f32;
                qg[h * 2 * hd + hd + i] = (200 + h * 10 + i) as f32;
            }
        }
        let mut q = vec![0f32; nh * hd];
        let mut g = vec![0f32; nh * hd];
        take_q(&mut q, &qg, nh, hd);
        take_gate(&mut g, &qg, nh, hd);
        assert_eq!(q, vec![100., 101., 102., 103., 110., 111., 112., 113., 120., 121., 122., 123.]);
        assert_eq!(g, vec![200., 201., 202., 203., 210., 211., 212., 213., 220., 221., 222., 223.]);
        // The wrong reading -- first half is queries -- would have taken gate values into
        // q from head 1 onward. Assert the two differ, so this test cannot pass vacuously.
        assert_ne!(&q[hd..2 * hd], &qg[hd..2 * hd], "halves-split would read head 0's gate");
    }

    /// Router weights must sum to one over the chosen experts. Without renormalisation
    /// they sum to whatever mass the top-k happened to capture, uniformly attenuating the
    /// whole FFN branch.
    #[test]
    fn routing_picks_the_top_k_and_renormalises_them() {
        let logits = [0.1f32, 5.0, 0.2, 4.0, 3.0];
        let sel = route(&logits, 3);
        assert_eq!(sel.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1, 3, 4]);
        // llama.cpp gathers the raw probabilities and THEN divides by their clamped sum
        // (ffn_moe_weights -> SUM_ROWS -> CLAMP -> DIV), so the weights reaching the
        // experts sum to 1.
        let sum: f32 = sel.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights must sum to 1, got {sum}");
        assert!(sel[0].1 > sel[1].1 && sel[1].1 > sel[2].1, "ordered by probability");
    }

    /// Ties must resolve by index, or two runs of the same prompt can route differently
    /// and the reproducibility guarantee quietly stops holding.
    #[test]
    fn routing_breaks_ties_by_index() {
        let sel = route(&[1.0f32, 1.0, 1.0, 1.0], 2);
        assert_eq!(sel.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 1]);
    }

    /// The recurrent state is fixed-size: it does NOT grow with context. That is what
    /// makes 30 of the 40 blocks cost nothing per token.
    #[test]
    fn the_linear_state_does_not_grow_with_context() {
        let c = cfg();
        let st = LinState::new(&c);
        assert_eq!(st.s.len(), 32 * 128 * 128, "one 128x128 matrix per value head");
        assert_eq!(st.conv.len(), 3 * 8192, "kernel 4 keeps 3 previous steps");
        // 2 MB a layer, 30 layers -- 63 MB total, independent of sequence length.
        assert_eq!(st.s.len() * 4, 2_097_152);
    }

    /// q and k are widened from `nk` heads to `nv` by `ggml_repeat_4d`, which TILES:
    /// index i maps to `i % n_src`. Two key heads over four value heads therefore pair as
    /// [0, 1, 0, 1], NOT [0, 0, 1, 1].
    ///
    /// This test previously asserted the opposite, which is how the bug survived: both
    /// mappings keep every shape identical and both yield a model that runs. Only a diff
    /// against llama.cpp's `attn_output` on the tiny fixture told them apart.
    #[test]
    fn value_heads_read_key_heads_by_modulo_because_repeat_tiles() {
        let c = cfg();
        let by_mod: Vec<usize> = (0..c.n_v_heads).map(|hv| hv % c.n_k_heads).collect();
        let by_div: Vec<usize> = (0..c.n_v_heads).map(|hv| hv / (c.n_v_heads / c.n_k_heads)).collect();
        assert_eq!(&by_mod[..4], &[0, 1, 2, 3], "32 v heads over 16 k heads: tiled");
        assert_ne!(by_mod, by_div, "the two mappings must not be confusable");
        assert_eq!(*by_mod.iter().max().unwrap(), c.n_k_heads - 1);
    }

    /// Rope must leave 192 of each 256-wide head untouched. Rotating the whole head is a
    /// different position encoding that still trains-looking output.
    #[test]
    fn rope_is_partial_over_the_head() {
        let c = cfg();
        let r = gqa::rope(c.head_dim, c.n_rot, 8, c.rope_base, Pairing::Halves);
        let mut x = vec![1.0f32; c.head_dim];
        gqa::apply_rope(&mut x, &r, 3, false);
        assert!(x[..c.n_rot].iter().any(|v| (*v - 1.0).abs() > 1e-6), "the first 64 rotate");
        assert!(
            x[c.n_rot..].iter().all(|v| (*v - 1.0).abs() < 1e-9),
            "dims past n_rot must pass through untouched"
        );
    }
}

