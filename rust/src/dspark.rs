// SPDX-License-Identifier: Apache-2.0
//
// DSpark: DeepSeek-V4-Flash's trained block drafter.
//
// The checkpoint stores it under the `mtp.*` namespace, which is legacy naming -- this is
// not DeepSeek-V3's chain of single-token MTP heads. `inference/model.py` is explicit:
//
//     class DSparkBlock(Block):
//         """DSpark stage stored under the mtp.* checkpoint namespace."""
//         self.main_proj = Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)
//
// with `dspark_target_layer_ids: [40, 41, 42]`, which is why `main_proj` is [4096, 12288]
// rather than the [4096, 8192] a V3-style head would use: its input is the concatenation
// of the MAIN model's hidden states at layers 40, 41 and 42, each hc-averaged.
//
// Three differences from a classic drafter, all of which change the code:
//
//   1. It drafts a BLOCK of `dspark_block_size` (5) tokens in one pass, not one token per
//      pass. The block is seeded [real_token, noise, noise, noise, noise] and the noise
//      slots are filled in by attention, which is BIDIRECTIONAL within the block --
//      `get_dspark_topk_idxs` hands every one of the 5 queries the same key set, so
//      position 1 sees position 4. It is not causal and must not be made causal.
//
//   2. Its KV cache is keyed to the MAIN model's positions, not its own. Each stage
//      derives one KV row per main-model position from `main_x`, and the draft block
//      attends to a 128-wide window of those plus its own 5 rows. So every accepted
//      token has to be pushed through `push` whether or not a draft is wanted, which is
//      why the prompt is walked through it too.
//
//   3. It has a Markov head and a confidence head. The Markov head is a rank-256 bigram
//      correction applied autoregressively ACROSS the block: the logits at position i are
//      biased by an embedding of the token chosen at position i-1, so the 5 tokens are
//      not independent even though the attention that produced them was parallel.
//
// Drafting is greedy here, where the reference samples. That is deliberate and it is not
// an approximation: the verifier is greedy argmax, so a greedy draft is the draft most
// likely to be accepted. Correctness never rests on the drafter -- `layer_forward_spec`
// accepts only the prefix the main model itself predicts, so a bad draft costs a wasted
// read and never a wrong token.

use crate::arch::Spec;
use crate::cache::{v4_dspark_expert_names, Cache};
use crate::ops::{HcLayer, HyperConnResidual, W};
use crate::st::St;
use crate::v4::*;
use crate::v4run::{f32s, fp8, raw, Fp8, Layer, Trunk, BLOCK, HC};

/// The main-model layers whose hidden states feed `main_proj`, in checkpoint order.
/// `config.json: dspark_target_layer_ids`. The concatenation order is the order of this
/// list, so permuting it produces a plausible-looking draft from the wrong projection.
pub const TARGET_LAYERS: [usize; 3] = [40, 41, 42];

/// How many `mtp.N` stages the checkpoint actually ships.
///
/// NOT `config.json: num_nextn_predict_layers`, which says 1 for DeepSeek-V4-Flash while
/// three blocks are stored -- and only the third carries the output glue, so trusting the
/// config builds a stage stack two thirds too short and then fails on a missing tensor.
pub fn n_stages(st: &St) -> usize {
    (0..).take_while(|i| st.find(&format!("mtp.{i}.attn_norm.weight")).is_some()).count()
}

pub struct DSpark {
    stages: Vec<Layer>,
    /// Stage 0 only: projects the concatenated main hidden states down to one hidden.
    main_proj: Fp8,
    main_norm: Vec<f32>,
    /// Last stage only.
    norm: Vec<f32>,
    hc_head_fn: Vec<f32>,
    hc_head_base: Vec<f32>,
    hc_head_scale: f32,
    /// [vocab][rank] bf16. `markov_w1` is an embedding, `markov_w2` projects back to vocab.
    markov_w1: Vec<u8>,
    markov_w2: Vec<u8>,
    /// [hidden + rank] f32, from a bf16 tensor.
    conf: Vec<f32>,
    /// One KV row per MAIN-model position, per stage: [stage][n][head_dim].
    kv: Vec<Vec<f32>>,
    pub block: usize,
    pub noise: u32,
    pub rank: usize,
    pub bytes: usize,
}

pub struct Draft {
    /// The `block` drafted tokens, in order, following the token `draft` was given.
    pub ids: Vec<u32>,
    /// The head's own confidence per drafted position. Higher is more certain; the scale
    /// is whatever training left it at, so it is only ever compared against itself.
    pub conf: Vec<f32>,
    /// [block][hidden] the hc_head output, before `norm` and the vocabulary head. Kept
    /// because it is the last point where every upstream stage -- main_proj, the KV ring,
    /// the block attention, all three MoEs, hc_head -- is still visible as one number
    /// that an independent implementation can be compared against.
    pub hx: Vec<f32>,
}

fn bf16_row(w: &[u8], row: usize, n: usize, dst: &mut [f32]) {
    let p: &[u16] = unsafe { std::slice::from_raw_parts(w.as_ptr().cast(), w.len() / 2) };
    for i in 0..n {
        dst[i] = crate::st::bf16_to_f32(p[row * n + i]);
    }
}

fn bf16_w(w: &[u8]) -> W<'_> {
    W::Bf16(unsafe { std::slice::from_raw_parts(w.as_ptr().cast(), w.len() / 2) })
}

impl DSpark {
    /// `n_stages` is how many `mtp.N` blocks the checkpoint ships. The last one carries the
    /// output glue (`norm`, `hc_head_*`, both Markov tables, the confidence projection);
    /// the first carries `main_proj`/`main_norm`. Both are asserted by the loads below
    /// failing rather than by a silent default.
    pub fn load(
        st: &St,
        s: &Spec,
        n_stages: usize,
        block: usize,
        noise: u32,
    ) -> Result<DSpark, String> {
        if n_stages == 0 {
            return Err("DSpark needs at least one mtp stage".into());
        }
        let mut stages = Vec::with_capacity(n_stages);
        let mut bytes = 0usize;
        for i in 0..n_stages {
            // ratio 0: DSparkAttention asserts compress_ratio == 0, so no Compressor and
            // no Indexer -- and the checkpoint ships neither for mtp.*.
            let lay = Layer::load(st, s, &format!("mtp.{i}"), 0, None)?;
            bytes += lay.bytes();
            stages.push(lay);
        }
        let last = n_stages - 1;
        let main_proj = fp8(st, "mtp.0.main_proj")?;
        let markov_w1 = raw(st, &format!("mtp.{last}.markov_head.markov_w1.weight"))?;
        let markov_w2 = raw(st, &format!("mtp.{last}.markov_head.markov_w2.weight"))?;
        let confraw = raw(st, &format!("mtp.{last}.confidence_head.proj.weight"))?;
        let mut conf = vec![0f32; confraw.len() / 2];
        bf16_row(&confraw, 0, conf.len(), &mut conf);
        let rank = markov_w1.len() / 2 / s.vocab;
        bytes += main_proj.bytes() + markov_w1.len() + markov_w2.len();
        Ok(DSpark {
            stages,
            main_proj,
            main_norm: f32s(st, "mtp.0.main_norm.weight")?,
            norm: f32s(st, &format!("mtp.{last}.norm.weight"))?,
            hc_head_fn: f32s(st, &format!("mtp.{last}.hc_head_fn"))?,
            hc_head_base: f32s(st, &format!("mtp.{last}.hc_head_base"))?,
            hc_head_scale: f32s(st, &format!("mtp.{last}.hc_head_scale"))?[0],
            markov_w1,
            markov_w2,
            conf,
            kv: vec![Vec::new(); n_stages],
            block,
            noise,
            rank,
            bytes,
        })
    }

    /// One stage's per-main-position KV rows, [n][head_dim], post-norm and post-rope.
    pub fn kv_rows(&self, stage: usize) -> &[f32] {
        &self.kv[stage]
    }

    pub fn n_positions(&self, head_dim: usize) -> usize {
        self.kv[0].len() / head_dim
    }

    /// Roll the per-stage KV back to `n` main-model positions. Speculative verification
    /// writes rows for tokens it may reject; leaving them would let later drafts attend to
    /// a history that never happened.
    pub fn truncate(&mut self, n: usize, head_dim: usize) {
        for k in self.kv.iter_mut() {
            k.truncate(n * head_dim);
        }
    }

    /// `main_x`, shared by all stages: `main_norm(main_proj(concat(h40, h41, h42)))`.
    pub fn main_x(&self, main_hidden: &[f32], e: usize, eps: f32) -> Vec<f32> {
        let mut p = vec![0f32; e];
        crate::ops::mmw(&mut p, main_hidden, self.main_proj.w(), main_hidden.len(), e);
        let mut out = vec![0f32; e];
        crate::ops::rmsnorm(&mut out, &p, &self.main_norm, e, eps);
        out
    }

    /// Record one main-model position. Every accepted token must pass through here, in
    /// order, including prompt tokens -- the reference does exactly this in its prefill
    /// branch, where `DSparkBlock.forward` calls the attention for its cache side effect
    /// and returns `x` untouched.
    pub fn push(
        &mut self,
        main_hidden: &[f32],
        pos: usize,
        rope: &Rope,
        e: usize,
        head_dim: usize,
        rope_head_dim: usize,
        eps: f32,
    ) {
        let mx = self.main_x(main_hidden, e, eps);
        for (i, lay) in self.stages.iter().enumerate() {
            let mut kv = vec![0f32; head_dim];
            crate::ops::mmw(&mut kv, &mx, lay.wkv.w(), e, head_dim);
            let src = kv.clone();
            crate::ops::rmsnorm(&mut kv, &src, &lay.kv_norm, head_dim, eps);
            apply_rope(&mut kv[head_dim - rope_head_dim..], rope, pos, false);
            self.kv[i].extend_from_slice(&kv);
        }
    }

    /// Draft `block` tokens to follow `token`, which sits at main-model position `pos`.
    ///
    /// `push` must already have been called for `pos`, so the stage caches hold
    /// `pos + 1` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        trunk: &Trunk,
        s: &Spec,
        md: &MoeDimsV4,
        token: u32,
        pos: usize,
        rope: &Rope,
        st: &St,
        cache: &mut Cache,
        eps: f32,
    ) -> Result<Draft, String> {
        let (e, hd, k) = (s.hidden, s.head_dim, self.block);
        debug_assert_eq!(self.n_positions(hd), pos + 1, "push must precede draft");

        // The block is seeded with the real token and `block-1` copies of the noise token;
        // the noise slots carry no information and are resolved by the bidirectional
        // attention below. Seeding them with anything else -- the real token repeated, the
        // previous tokens -- drafts fluently and wrongly.
        let ids: Vec<u32> = (0..k).map(|i| if i == 0 { token } else { self.noise }).collect();
        let mut x0 = vec![0f32; k * HC * e];
        let mut row = vec![0f32; e];
        for (t, &id) in ids.iter().enumerate() {
            trunk.embed_row(id, e, &mut row);
            for c in 0..HC {
                x0[(t * HC + c) * e..][..e].copy_from_slice(&row);
            }
        }
        let mut res = HyperConnResidual::new(&x0, k, e, HC, eps, eps, 20);

        for si in 0..self.stages.len() {
            let lay = &self.stages[si];
            let ad = AttnDims {
                hidden: e,
                n_heads: s.n_heads,
                head_dim: hd,
                rope_head_dim: 64,
                q_lora_rank: 1024,
                o_lora_rank: 1024,
                o_groups: 8,
                window: BLOCK,
                compress_ratio: 0,
                eps,
            };
            let attn = AttnW {
                wq_a: lay.wq_a.w(),
                q_norm: &lay.q_norm,
                wq_b: lay.wq_b.w(),
                wkv: lay.wkv.w(),
                kv_norm: &lay.kv_norm,
                wo_a: lay.wo_a.w(),
                wo_b: lay.wo_b.w(),
                attn_sink: &lay.sink,
            };
            let hc = HcLayer {
                attn_fn: &lay.hc_attn[0],
                attn_base: &lay.hc_attn[1],
                attn_scale: [lay.hc_attn[2][0], lay.hc_attn[2][1], lay.hc_attn[2][2]],
                ffn_fn: &lay.hc_ffn[0],
                ffn_base: &lay.hc_ffn[1],
                ffn_scale: [lay.hc_ffn[2][0], lay.hc_ffn[2][1], lay.hc_ffn[2][2]],
            };
            let moe = MoeWV4 {
                gate: &lay.gate,
                bias: lay.bias.as_deref(),
                tid2eid: lay.tid2eid.as_deref(),
                sh1: lay.sh1.w(),
                sh3: lay.sh3.w(),
                sh2: lay.sh2.w(),
            };
            res.begin_layer(&hc);

            use crate::ops::{Residual, Sub};
            let mut modin = vec![0f32; k * e];
            let mut hin = vec![0f32; k * e];
            let mut tmp = vec![0f32; k * e];

            let carry = res.pre(Sub::Attn, &mut modin);
            for t in 0..k {
                crate::ops::rmsnorm(
                    &mut hin[t * e..][..e],
                    &modin[t * e..][..e],
                    &lay.attn_norm,
                    e,
                    eps,
                );
            }
            block_attn(&mut tmp, &hin, &attn, &ad, rope, &self.kv[si], pos, k);
            res.post(Sub::Attn, &tmp, carry);

            let carry = res.pre(Sub::Mlp, &mut modin);
            for t in 0..k {
                crate::ops::rmsnorm(
                    &mut hin[t * e..][..e],
                    &modin[t * e..][..e],
                    &lay.ffn_norm,
                    e,
                    eps,
                );
            }
            // Same union trick as the verifier: route all `k` block positions, fetch the
            // union once. The stages are the same three every step, so after the first
            // draft their experts are largely cache-resident.
            let layer = crate::cache::V4_LAYERS + si;
            cache.at_layer(layer, s.n_layers + self.stages.len());
            let mut sel = Vec::with_capacity(k);
            let mut union: Vec<usize> = Vec::with_capacity(k * md.topk);
            for t in 0..k {
                let (idx, wt) = route_v4(&hin[t * e..][..e], &moe, md, layer, ids[t]);
                for &i in &idx {
                    if !union.contains(&(i as usize)) {
                        union.push(i as usize);
                    }
                }
                sel.push((idx, wt));
            }
            cache.prefetch_many(st, layer, &union, v4_dspark_expert_names);
            for t in 0..k {
                let (idx, wt) = &sel[t];
                moe_v4_routed(
                    &mut tmp[t * e..][..e],
                    &hin[t * e..][..e],
                    &moe,
                    md,
                    layer,
                    idx,
                    wt,
                    st,
                    cache,
                    v4_dspark_expert_names,
                )?;
            }
            res.post(Sub::Mlp, &tmp, carry);
        }

        // forward_head: hc_head -> norm -> vocabulary head, then the Markov chain.
        let sstate = res.state();
        let mut hx = vec![0f32; k * e];
        let mut logits = vec![0f32; s.vocab];
        let mut normed = vec![0f32; e];
        let mut out = Vec::with_capacity(k);
        let mut conf = Vec::with_capacity(k);
        let mut emb = vec![0f32; self.rank];
        let mut bias = vec![0f32; s.vocab];
        let mut prev = token;
        for t in 0..k {
            crate::ops::hc_head(
                &mut hx[t * e..][..e],
                &sstate[t * HC * e..][..HC * e],
                &self.hc_head_fn,
                &self.hc_head_base,
                self.hc_head_scale,
                e,
                HC,
                eps,
                eps,
            );
            crate::ops::rmsnorm(&mut normed, &hx[t * e..][..e], &self.norm, e, eps);
            crate::ops::mmw(&mut logits, &normed, trunk.head_w(), e, s.vocab);
            // The bias depends on the token chosen at t-1, so this loop is sequential
            // even though the attention that produced `hx` was not.
            bf16_row(&self.markov_w1, prev as usize, self.rank, &mut emb);
            crate::ops::mmw(&mut bias, &emb, bf16_w(&self.markov_w2), self.rank, s.vocab);
            let next = logits
                .iter()
                .zip(&bias)
                .enumerate()
                .max_by(|a, b| {
                    (a.1 .0 + a.1 .1)
                        .partial_cmp(&(b.1 .0 + b.1 .1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32)
                .ok_or("empty logits")?;
            // confidence_head reads the PRE-norm hc_head output concatenated with the
            // Markov embedding of this position's input token.
            let mut c = 0.0f64;
            for i in 0..e {
                c += self.conf[i] as f64 * hx[t * e + i] as f64;
            }
            for i in 0..self.rank {
                c += self.conf[e + i] as f64 * emb[i] as f64;
            }
            conf.push(c as f32);
            out.push(next);
            prev = next;
        }
        Ok(Draft { ids: out, conf, hx })
    }
}

/// DSparkAttention for one draft block.
///
/// `kv_hist` is this stage's per-main-position cache, [n][head_dim], already normalised
/// and rope'd; `pos` is the main-model position of the seed token, so `n == pos + 1`.
///
/// Two things here differ from `attention_decode` and both are load-bearing:
///
///   * The key set is the SAME for all `k` queries -- the 128-wide window of main
///     positions ending at `pos`, then all `k` block rows. Block position 1 therefore
///     attends to block position 4. Making this causal would turn the drafter back into
///     an autoregressive one and destroy the block prediction it was trained for.
///   * The window ends at `pos`, the seed's position, not at each query's own position,
///     because the cached rows describe the main model's history and stop there.
#[allow(clippy::too_many_arguments)]
fn block_attn(
    out: &mut [f32],
    hin: &[f32],
    w: &AttnW,
    d: &AttnDims,
    rope: &Rope,
    kv_hist: &[f32],
    pos: usize,
    k: usize,
) {
    let (e, h_n, hd, rd) = (d.hidden, d.n_heads, d.head_dim, d.rope_head_dim);
    let n = pos + 1;
    let scale = (hd as f32).powf(-0.5);
    let gsz = h_n * hd / d.o_groups;

    let mut kv = kv_hist[..n * hd].to_vec();
    let mut q = vec![0f32; k * h_n * hd];
    for t in 0..k {
        let xt = &hin[t * e..][..e];
        let p = pos + 1 + t;

        let mut qr = vec![0f32; d.q_lora_rank];
        crate::ops::mmw(&mut qr, xt, w.wq_a, e, d.q_lora_rank);
        let mut qn = vec![0f32; d.q_lora_rank];
        crate::ops::rmsnorm(&mut qn, &qr, w.q_norm, d.q_lora_rank, d.eps);
        let qt = &mut q[t * h_n * hd..][..h_n * hd];
        crate::ops::mmw(qt, &qn, w.wq_b, d.q_lora_rank, h_n * hd);
        for h in 0..h_n {
            let qh = &mut qt[h * hd..][..hd];
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

        let mut kvt = vec![0f32; hd];
        crate::ops::mmw(&mut kvt, xt, w.wkv, e, hd);
        let src = kvt.clone();
        crate::ops::rmsnorm(&mut kvt, &src, w.kv_norm, hd, d.eps);
        apply_rope(&mut kvt[hd - rd..], rope, p, false);
        kv.extend_from_slice(&kvt);
    }

    let mut idxs = window_row(pos, d.window, n);
    idxs.extend((0..k).map(|i| (n + i) as i64));

    let mut o = vec![0f32; h_n * hd];
    let mut go = vec![0f32; d.o_groups * d.o_lora_rank];
    for t in 0..k {
        let p = pos + 1 + t;
        for h in 0..h_n {
            sparse_attn_row(
                &mut o[h * hd..][..hd],
                &q[t * h_n * hd + h * hd..][..hd],
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
            crate::ops::mmw(dst, osl, sub, gsz, rows);
        }
        crate::ops::mmw(&mut out[t * e..][..e], &go, w.wo_b, d.o_groups * d.o_lora_rank, e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every query in the block must see every other block position AND the whole window,
    // which is what `get_dspark_topk_idxs` builds. A causal reading of the block -- the
    // natural instinct, and what `window_row` alone would give -- silently drops the last
    // rows and turns a block drafter into an autoregressive one.
    #[test]
    fn the_block_attends_to_itself_bidirectionally() {
        let (pos, win, k) = (7usize, 128usize, 5usize);
        let n = pos + 1;
        let mut idxs = window_row(pos, win, n);
        idxs.extend((0..k).map(|i| (n + i) as i64));
        assert_eq!(idxs.len(), n + k, "8 history rows plus the 5 block rows");
        assert!(idxs.iter().all(|&i| i >= 0), "nothing is masked at this length");
        assert_eq!(&idxs[n..], &[8, 9, 10, 11, 12], "the block rows follow the history");
    }

    // Past 128 main positions the window slides, and it ends at the SEED's position --
    // not at each query's own position, which is what a decoder-layer window would do.
    #[test]
    fn the_window_ends_at_the_seed_and_holds_at_128() {
        let (pos, win, k) = (500usize, 128usize, 5usize);
        let n = pos + 1;
        let mut idxs = window_row(pos, win, n);
        assert_eq!(idxs.len(), win);
        assert_eq!(idxs[0], (pos - win + 1) as i64);
        assert_eq!(idxs[win - 1], pos as i64);
        idxs.extend((0..k).map(|i| (n + i) as i64));
        assert_eq!(idxs.len(), win + k);
    }

    // The seed occupies slot 0 and every other slot is the noise token. Repeating the
    // real token instead is the plausible-looking error: it drafts, and it drafts wrong.
    #[test]
    fn the_block_is_seeded_with_one_real_token_and_noise() {
        let (token, noise, k) = (1234u32, 128799u32, 5usize);
        let ids: Vec<u32> = (0..k).map(|i| if i == 0 { token } else { noise }).collect();
        assert_eq!(ids, vec![1234, 128799, 128799, 128799, 128799]);
    }
}
