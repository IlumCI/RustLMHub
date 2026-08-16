//! Streamed LoRA fine-tuning: the training half of the engine.
//!
//! Inference here already runs a model larger than RAM by streaming its frozen weights off
//! disk through [`crate::cache`]. Training reuses that same machinery, plus two pieces the
//! forward pass never needed:
//!
//!   - [`crate::cache::Dir::Down`] -- Belady-by-layer eviction, mirrored, so the BACKWARD
//!     sweep (layers n..0) does not evict exactly the states it is about to revisit.
//!   - [`crate::ops::wt`] / [`crate::gguf::out_prod_q4k`] -- `grad_x = Wᵀ·grad_y`, the
//!     transpose-free backward matvec through a frozen k-quant weight, verified as the exact
//!     adjoint of the forward matmul.
//!
//! On top of those, this module is the standard SFT arithmetic, kept deliberately plain:
//! a LoRA adapter ([`Lora`]) over one frozen projection, an Adam step on the adapter's own
//! parameters (the only things that get gradients -- W stays frozen, so training costs the
//! LoRA `4N` FLOPs/token, not the `6N` of a full fine-tune), grad accumulation over a wide
//! batch to amortise the weight I/O, and a cross-entropy loss masked to response tokens.
//!
//! The whole point of the engine is that a wrong implementation still produces fluent text,
//! so the backward pass is not trusted -- it is finite-difference gradient-checked against
//! the forward pass (`gradient_check_matches_finite_difference`), which is the training
//! analog of the adjoint test that gates `wt`.

use crate::ops::{self, W};

/// A LoRA adapter sitting on one frozen linear `y = W·x`, where `W` is `[out, k_in]`.
///
/// The adapter adds a low-rank correction `y += (alpha/r) · B·(A·x)` with `A` = `[r, k_in]`
/// and `B` = `[out, r]`. Only `A` and `B` carry gradients; `W` is streamed and frozen. `A`
/// is seeded small and `B` starts at zero, so the adapter is the identity at step 0 and the
/// model begins training exactly as the pretrained one -- the standard LoRA initialisation.
///
/// Resident cost is `r·(k_in + out)` f32 weights plus the same again twice over for Adam's
/// two moment buffers and the grad accumulator: for `r=32` on a 5120-wide projection that is
/// ~0.6 MB per adapter, so a top-10-layer adapter set is tens of MB -- resident, while the
/// billions of frozen parameters stream past.
pub struct Lora {
    pub r: usize,
    pub k_in: usize,
    pub out: usize,
    /// `alpha/r`, the fixed LoRA scale applied to the low-rank branch.
    pub scale: f32,
    /// `[r * k_in]`, row-major (`a[j*k_in + i]`). The down-projection.
    pub a: Vec<f32>,
    /// `[out * r]`, row-major (`b[o*r + j]`). The up-projection, zero at init.
    pub b: Vec<f32>,
    // Grad accumulators, same shapes as a/b. Summed across a grad-accumulation batch and
    // consumed by `adam_step`.
    ga: Vec<f32>,
    gb: Vec<f32>,
    // Adam first/second moments, same shapes. Resident for the adapter only.
    ma: Vec<f32>,
    va: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
}

impl Lora {
    /// `alpha` is the LoRA scaling numerator (the branch is scaled by `alpha/r`). `seed`
    /// drives the deterministic init of `A`; `B` is zero.
    pub fn new(r: usize, k_in: usize, out: usize, alpha: f32, seed: u64) -> Lora {
        assert!(r > 0 && k_in > 0 && out > 0, "LoRA dims must be positive");
        // Kaiming-flavoured small init on A, scaled by 1/sqrt(k_in). The exact distribution
        // is not load-bearing (B=0 makes the branch zero regardless at step 0); determinism
        // is, so runs reproduce.
        let sd = (1.0 / k_in as f64).sqrt();
        let mut s = seed | 1;
        let mut randn = || {
            // Two LCG draws -> a crude Box-Muller-free bounded gaussian-ish value. Bounded,
            // finite, deterministic -- enough for an init.
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (s >> 11) as f64 / (1u64 << 53) as f64;
            ((u - 0.5) + (v - 0.5)) * sd // triangular, mean 0
        };
        let a: Vec<f32> = (0..r * k_in).map(|_| randn() as f32).collect();
        Lora {
            r,
            k_in,
            out,
            scale: alpha / r as f32,
            a,
            b: vec![0.0; out * r],
            ga: vec![0.0; r * k_in],
            gb: vec![0.0; out * r],
            ma: vec![0.0; r * k_in],
            va: vec![0.0; r * k_in],
            mb: vec![0.0; out * r],
            vb: vec![0.0; out * r],
        }
    }

    /// Resident parameter + optimiser-state bytes for this adapter.
    pub fn resident_bytes(&self) -> usize {
        // a,b,ga,gb,ma,va,mb,vb -- eight buffers of the adapter's parameter count.
        8 * (self.a.len() + self.b.len()) / 2 * std::mem::size_of::<f32>()
    }

    /// `A·x` -> `ax[r]`. Cached by the caller and handed back to [`Lora::backward`] so the
    /// down-projection is not recomputed.
    fn ax(&self, x: &[f32], ax: &mut [f32]) {
        for j in 0..self.r {
            let row = &self.a[j * self.k_in..][..self.k_in];
            let mut acc = 0f64;
            for i in 0..self.k_in {
                acc += row[i] as f64 * x[i] as f64;
            }
            ax[j] = acc as f32;
        }
    }

    /// Forward: `y = W·x + scale·B·(A·x)`. Writes `y[out]` and the cached `ax[r]`.
    ///
    /// `W` is the frozen streamed weight; the low-rank branch is the only trainable part.
    pub fn forward(&self, w: W, x: &[f32], y: &mut [f32], ax_buf: &mut [f32]) {
        debug_assert_eq!(x.len(), self.k_in);
        debug_assert_eq!(y.len(), self.out);
        debug_assert_eq!(ax_buf.len(), self.r);
        // Frozen path: y = W x.
        ops::mmw(y, x, w, self.k_in, self.out);
        // Low-rank path: y += scale * B (A x).
        self.ax(x, ax_buf);
        for o in 0..self.out {
            let brow = &self.b[o * self.r..][..self.r];
            let mut acc = 0f64;
            for j in 0..self.r {
                acc += brow[j] as f64 * ax_buf[j] as f64;
            }
            y[o] += self.scale * acc as f32;
        }
    }

    /// Backward through one adapter. Given `grad_y[out]`, the input `x` and the cached
    /// `ax[r]` from [`Lora::forward`]:
    ///   - accumulates `grad_A`, `grad_B` into the internal accumulators, and
    ///   - returns `grad_x[k_in] = Wᵀ·grad_y + scale·Aᵀ·(Bᵀ·grad_y)` into `grad_x`.
    ///
    /// The `Wᵀ·grad_y` term is the streamed transpose-free matvec ([`ops::wt`]); the rest is
    /// dense arithmetic on the tiny resident adapter. This is what lets the gradient keep
    /// flowing DOWN the stack to lower adapted layers without ever materialising `Wᵀ`.
    pub fn backward(&mut self, w: W, x: &[f32], ax: &[f32], grad_y: &[f32], grad_x: &mut [f32]) {
        debug_assert_eq!(grad_y.len(), self.out);
        debug_assert_eq!(grad_x.len(), self.k_in);

        // grad w.r.t. the down-projection output: g_ax = scale * Bᵀ grad_y   ([r]).
        // Simultaneously accumulate grad_B[o,j] += scale * grad_y[o] * ax[j].
        let mut g_ax = vec![0f64; self.r];
        for o in 0..self.out {
            let gy = grad_y[o] as f64 * self.scale as f64;
            if gy == 0.0 {
                continue;
            }
            let brow = &self.b[o * self.r..][..self.r];
            let gbrow = &mut self.gb[o * self.r..][..self.r];
            for j in 0..self.r {
                g_ax[j] += gy * brow[j] as f64;
                gbrow[j] += (gy * ax[j] as f64) as f32;
            }
        }

        // grad_A[j,i] += g_ax[j] * x[i]; and the low-rank contribution to grad_x:
        // grad_x_lora = Aᵀ g_ax.
        let mut grad_x_lora = vec![0f64; self.k_in];
        for j in 0..self.r {
            let gj = g_ax[j];
            if gj == 0.0 {
                continue;
            }
            let arow = &self.a[j * self.k_in..][..self.k_in];
            let garow = &mut self.ga[j * self.k_in..][..self.k_in];
            for i in 0..self.k_in {
                garow[i] += (gj * x[i] as f64) as f32;
                grad_x_lora[i] += gj * arow[i] as f64;
            }
        }

        // Frozen path: grad_x_w = Wᵀ grad_y, the streamed backward matvec.
        ops::wt(grad_x, grad_y, w, self.k_in, self.out);
        // grad_x = grad_x_w + grad_x_lora.
        for i in 0..self.k_in {
            grad_x[i] = (grad_x[i] as f64 + grad_x_lora[i]) as f32;
        }
    }

    /// Zero the grad accumulators. Call once at the start of each accumulation batch.
    pub fn zero_grad(&mut self) {
        self.ga.iter_mut().for_each(|g| *g = 0.0);
        self.gb.iter_mut().for_each(|g| *g = 0.0);
    }

    /// One Adam step consuming the accumulated grads. `t` is the 1-based step number (for
    /// bias correction). `scale` divides the grad -- pass the grad-accumulation count so the
    /// update is the mean, not the sum, over the batch.
    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, t: u64, scale: f32) {
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        let inv = 1.0 / scale;
        let step =
            |p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32]| {
                for k in 0..p.len() {
                    let gr = g[k] * inv;
                    m[k] = b1 * m[k] + (1.0 - b1) * gr;
                    v[k] = b2 * v[k] + (1.0 - b2) * gr * gr;
                    let mh = m[k] / bc1;
                    let vh = v[k] / bc2;
                    p[k] -= lr * mh / (vh.sqrt() + eps);
                }
            };
        step(&mut self.a, &self.ga, &mut self.ma, &mut self.va);
        step(&mut self.b, &self.gb, &mut self.mb, &mut self.vb);
    }
}

/// Cross-entropy loss over a sequence, masked to the positions where `mask[t]` is true, and
/// its gradient w.r.t. the logits.
///
/// `logits` is `[seq * vocab]` row-major; `target[t]` is the gold next-token id at position
/// `t`; `mask[t]` selects which positions count. Prompt tokens are masked OUT so the model
/// is trained only to produce the response, not to re-predict the instruction -- the SFT
/// convention. Returns the mean loss over unmasked positions and fills `grad[seq*vocab]`
/// with `(softmax - onehot)/n_unmasked` at unmasked positions, zero elsewhere.
///
/// Numerically stable: the max-subtraction and the f64 sum are the usual guard against a
/// large logit overflowing `exp`, which on a 150k-vocab model is not hypothetical.
pub fn cross_entropy(
    logits: &[f32],
    target: &[u32],
    mask: &[bool],
    vocab: usize,
    grad: &mut [f32],
) -> f32 {
    let seq = target.len();
    debug_assert_eq!(logits.len(), seq * vocab);
    debug_assert_eq!(mask.len(), seq);
    debug_assert_eq!(grad.len(), seq * vocab);
    grad.iter_mut().for_each(|g| *g = 0.0);

    let n = mask.iter().filter(|&&m| m).count().max(1) as f64;
    let mut total = 0f64;
    for t in 0..seq {
        if !mask[t] {
            continue;
        }
        let row = &logits[t * vocab..][..vocab];
        let grow = &mut grad[t * vocab..][..vocab];
        let mut mx = f32::NEG_INFINITY;
        for &v in row {
            mx = mx.max(v);
        }
        let mut denom = 0f64;
        for &v in row {
            denom += ((v - mx) as f64).exp();
        }
        let tgt = target[t] as usize;
        // loss = -log softmax[tgt] = -(row[tgt]-mx) + log denom.
        total += -((row[tgt] - mx) as f64) + denom.ln();
        // grad = (softmax - onehot) / n.
        for k in 0..vocab {
            let p = ((row[k] - mx) as f64).exp() / denom;
            grow[k] = ((p - if k == tgt { 1.0 } else { 0.0 }) / n) as f32;
        }
    }
    (total / n) as f32
}

#[inline]
fn sigmoidf(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// A SwiGLU feed-forward block with a LoRA adapter on each of its three projections --
/// gate, up, and down. This is the FFN half of a transformer block, and the first real
/// backward *block* of the streamed training pass: the 27B is dense, so its per-layer
/// feed-forward is exactly this, and the whole block is verified by one gradient check
/// (`ffn_gradient_check`) rather than trusted.
///
/// Forward (matching `qwen35run::expert_fwd_many` with the unclamped SwiGLU):
/// ```text
///   g = W_gate·x + lora,   u = W_up·x + lora
///   a = silu(g) * u                    (silu(z) = z·σ(z))
///   y = W_down·a + lora
/// ```
/// The frozen `W_*` stream off disk; only the three adapters train. Backward reuses
/// [`Lora::backward`] three times -- so the frozen `Wᵀ·grad` terms are the verified
/// [`crate::ops::wt`] kernel -- with the SiLU derivative between the down and gate/up stages.
pub struct FfnLora {
    pub gate: Lora,
    pub up: Lora,
    pub down: Lora,
    pub hidden: usize,
    pub inter: usize,
}

/// Cached forward intermediates for one token, consumed by [`FfnLora::backward`]. Holding
/// them is the activation-checkpointing cost the backward pass trades memory for -- here per
/// call, in the full model per (layer, token).
pub struct FfnAct {
    g: Vec<f32>,       // gate pre-activation [inter]
    u: Vec<f32>,       // up projection [inter]
    a: Vec<f32>,       // silu(g)*u [inter]
    ax_gate: Vec<f32>, // gate adapter's A·x [r]
    ax_up: Vec<f32>,   // up adapter's A·x [r]
    ax_down: Vec<f32>, // down adapter's A·a [r]
    x: Vec<f32>,       // the input [hidden]
}

impl FfnLora {
    pub fn new(hidden: usize, inter: usize, r: usize, alpha: f32, seed: u64) -> FfnLora {
        FfnLora {
            gate: Lora::new(r, hidden, inter, alpha, seed),
            up: Lora::new(r, hidden, inter, alpha, seed ^ 0x9e37),
            down: Lora::new(r, inter, hidden, alpha, seed ^ 0x1234_5678),
            hidden,
            inter,
        }
    }

    /// Forward through the block. `wg`, `wu`, `wd` are the three frozen weights (gate, up,
    /// down). Returns `y[hidden]` and the cached activations for backward.
    pub fn forward(&self, wg: W, wu: W, wd: W, x: &[f32], y: &mut [f32]) -> FfnAct {
        let (h, n) = (self.hidden, self.inter);
        debug_assert_eq!(x.len(), h);
        let mut g = vec![0f32; n];
        let mut u = vec![0f32; n];
        let mut ax_gate = vec![0f32; self.gate.r];
        let mut ax_up = vec![0f32; self.up.r];
        self.gate.forward(wg, x, &mut g, &mut ax_gate);
        self.up.forward(wu, x, &mut u, &mut ax_up);
        let mut a = vec![0f32; n];
        for i in 0..n {
            a[i] = g[i] * sigmoidf(g[i]) * u[i]; // silu(g)*u
        }
        let mut ax_down = vec![0f32; self.down.r];
        self.down.forward(wd, &a, y, &mut ax_down);
        FfnAct { g, u, a, ax_gate, ax_up, ax_down, x: x.to_vec() }
    }

    /// Backward through the block. Given `grad_y[hidden]` and the cached [`FfnAct`],
    /// accumulates all three adapters' grads and returns `grad_x[hidden]`.
    pub fn backward(&mut self, wg: W, wu: W, wd: W, act: &FfnAct, grad_y: &[f32], grad_x: &mut [f32]) {
        let (h, n) = (self.hidden, self.inter);
        // Down: grad_y -> grad_a, accumulate down's adapter grads.
        let mut grad_a = vec![0f32; n];
        self.down.backward(wd, &act.a, &act.ax_down, grad_y, &mut grad_a);

        // SiLU: a = silu(g)*u.  grad_g = grad_a * silu'(g) * u,  grad_u = grad_a * silu(g).
        let mut grad_g = vec![0f32; n];
        let mut grad_u = vec![0f32; n];
        for i in 0..n {
            let s = sigmoidf(act.g[i]);
            let silu = act.g[i] * s;
            let silu_prime = s * (1.0 + act.g[i] * (1.0 - s)); // σ + g·σ·(1-σ)
            grad_g[i] = grad_a[i] * silu_prime * act.u[i];
            grad_u[i] = grad_a[i] * silu;
        }

        // gate and up each send grad back to x; sum the two contributions.
        let mut grad_x_gate = vec![0f32; h];
        let mut grad_x_up = vec![0f32; h];
        self.gate.backward(wg, &act.x, &act.ax_gate, &grad_g, &mut grad_x_gate);
        self.up.backward(wu, &act.x, &act.ax_up, &grad_u, &mut grad_x_up);
        for i in 0..h {
            grad_x[i] = grad_x_gate[i] + grad_x_up[i];
        }
    }

    pub fn zero_grad(&mut self) {
        self.gate.zero_grad();
        self.up.zero_grad();
        self.down.zero_grad();
    }

    pub fn adam_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, t: u64, scale: f32) {
        self.gate.adam_step(lr, b1, b2, eps, t, scale);
        self.up.adam_step(lr, b1, b2, eps, t, scale);
        self.down.adam_step(lr, b1, b2, eps, t, scale);
    }

    /// Serialise the trained adapters (A/B of gate, up, down) to a flat little-endian f32
    /// file with a small header. Only the adapter is saved -- the frozen model is unchanged
    /// on disk -- so a fine-tune is a few MB, not a few GB.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut buf = Vec::new();
        let hdr = [
            0x4c6f4141u32, // "LoAA" magic
            self.hidden as u32,
            self.inter as u32,
            self.gate.r as u32,
        ];
        for h in hdr {
            buf.extend_from_slice(&h.to_le_bytes());
        }
        for mat in [&self.gate.a, &self.gate.b, &self.up.a, &self.up.b, &self.down.a, &self.down.b] {
            for &v in mat {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, &buf).map_err(|e| format!("{path}: {e}"))
    }

    /// Load adapters saved by [`FfnLora::save`] into an adapter of matching shape. The header
    /// dims must match exactly -- loading an adapter trained for a different block shape is
    /// the kind of silent mismatch this engine refuses, so it is an error, not a reshape.
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        let buf = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let rd = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        if buf.len() < 16 || rd(&buf, 0) != 0x4c6f4141 {
            return Err(format!("{path}: not a LoAA adapter file"));
        }
        let (h, n, r) = (rd(&buf, 4) as usize, rd(&buf, 8) as usize, rd(&buf, 12) as usize);
        if (h, n, r) != (self.hidden, self.inter, self.gate.r) {
            return Err(format!(
                "{path}: adapter is (hidden {h}, inter {n}, r {r}) but this block is \
                 (hidden {}, inter {}, r {})",
                self.hidden, self.inter, self.gate.r
            ));
        }
        let mut off = 16;
        let take = |mat: &mut [f32], off: &mut usize| -> Result<(), String> {
            for v in mat.iter_mut() {
                if *off + 4 > buf.len() {
                    return Err(format!("{path}: truncated adapter payload"));
                }
                *v = f32::from_le_bytes([buf[*off], buf[*off + 1], buf[*off + 2], buf[*off + 3]]);
                *off += 4;
            }
            Ok(())
        };
        take(&mut self.gate.a, &mut off)?;
        take(&mut self.gate.b, &mut off)?;
        take(&mut self.up.a, &mut off)?;
        take(&mut self.up.b, &mut off)?;
        take(&mut self.down.a, &mut off)?;
        take(&mut self.down.b, &mut off)?;
        Ok(())
    }
}

/// One tokenised training example: the full token sequence and the length of its prompt
/// prefix. The response is `ids[prompt_len..]`, and the loss is trained only there.
#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    pub ids: Vec<u32>,
    /// Number of leading tokens that are prompt (system + user + the assistant-open marker).
    /// Everything from here on is the assistant response and is what the model is trained to
    /// produce.
    pub prompt_len: usize,
}

impl Example {
    /// The next-token training view: `input[t]` predicts `target[t]`, and `mask[t]` selects
    /// the positions whose *prediction* falls inside the response.
    ///
    /// Position `t` predicts `ids[t+1]`, so it is trained iff `ids[t+1]` is a response token,
    /// i.e. `t + 1 >= prompt_len`. The prompt tokens are still fed as context; they are just
    /// not scored -- the SFT convention, so the model learns to WRITE the response, not to
    /// re-predict the instruction it was given.
    pub fn view(&self) -> (Vec<u32>, Vec<u32>, Vec<bool>) {
        let n = self.ids.len().saturating_sub(1);
        let input = self.ids[..n].to_vec();
        let target = self.ids[1..=n].to_vec();
        let mask = (0..n).map(|t| t + 1 >= self.prompt_len).collect();
        (input, target, mask)
    }

    /// Response tokens actually scored -- the denominator of the per-example loss and the
    /// figure the token-budget arithmetic is built on.
    pub fn scored_tokens(&self) -> usize {
        self.ids.len().saturating_sub(self.prompt_len)
    }
}

/// Length of the longest shared prefix of two token sequences.
///
/// Masking assumes the prompt tokens are a prefix of the full tokenisation, which is *almost*
/// always true -- but BPE can merge across the boundary between the assistant-open marker and
/// the first response character, shifting one token. Taking the common-prefix length rather
/// than trusting `prompt_ids.len()` is robust to that: at worst one boundary token is counted
/// as prompt, which changes the loss by one position out of hundreds and never mislabels a
/// mid-response token as prompt.
pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Render + tokenise one (system, user, response) triple into an [`Example`] using the
/// model's OWN template and tokenizer.
///
/// Tokenising with anything other than the GGUF's vocab would silently desynchronise the ids
/// from the weights -- the canonical fluent-wrong failure -- so this takes the real
/// [`crate::chat::Template`] and [`crate::tok::Tok`]. It renders twice: the prompt with the
/// generation prompt appended (`<|im_start|>assistant\n`), and the full conversation with the
/// response as the assistant turn. `prompt_len` is the common-prefix length of the two
/// tokenisations.
pub fn build_example(
    tmpl: &crate::chat::Template,
    tok: &crate::tok::Tok,
    system: &str,
    user: &str,
    response: &str,
) -> Result<Example, String> {
    use serde_json::json;
    let no_tools: Vec<serde_json::Value> = Vec::new();

    let prompt_msgs = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user}),
    ];
    let full_msgs = vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user}),
        json!({"role": "assistant", "content": response}),
    ];

    let prompt_str = tmpl.render(&prompt_msgs, &no_tools, true)?;
    let full_str = tmpl.render(&full_msgs, &no_tools, false)?;

    // add_special=false: the template already carries the special markers; letting the
    // tokenizer add BOS/EOS again would double them and shift every mask boundary.
    let prompt_ids = tok.encode(&prompt_str, false)?;
    let full_ids = tok.encode(&full_str, false)?;

    if full_ids.len() <= prompt_ids.len() {
        return Err(format!(
            "empty response after tokenising: {} full tokens vs {} prompt tokens",
            full_ids.len(),
            prompt_ids.len()
        ));
    }
    let prompt_len = common_prefix_len(&prompt_ids, &full_ids);
    Ok(Example { ids: full_ids, prompt_len })
}

/// Build a training [`Example`] from a normalised [`crate::dataset::Record`], handling any
/// source format uniformly: supervised records go through the chat template with the prompt
/// masked; a `text_only` record (mask_prompt=false) is tokenised raw and trained on its whole
/// length (language-model completion, no masking).
pub fn example_from_record(
    tmpl: &crate::chat::Template,
    tok: &crate::tok::Tok,
    rec: &crate::dataset::Record,
) -> Result<Example, String> {
    if rec.mask_prompt {
        build_example(tmpl, tok, &rec.system, &rec.prompt, &rec.response)
    } else {
        let ids = tok.encode(&rec.response, true)?;
        if ids.is_empty() {
            return Err("empty text record".into());
        }
        Ok(Example { ids, prompt_len: 0 })
    }
}

/// A raw deduped record as produced by `tools/prep_redteam.py`.
#[derive(Clone, Debug)]
pub struct Raw {
    pub system: String,
    pub user: String,
    pub response: String,
}

/// Parse the prepared JSONL (`{system, user, response, ...}` per line). Extra fields are
/// ignored; a line missing any of the three required strings is an error rather than a
/// silently-skipped row, so a malformed export cannot quietly shrink the training set.
pub fn load_jsonl(path: &str) -> Result<Vec<Raw>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        let get = |k: &str| -> Result<String, String> {
            v.get(k)
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| format!("{path}:{}: missing string field {k:?}", i + 1))
        };
        out.push(Raw { system: get("system")?, user: get("user")?, response: get("response")? });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic exactly-representable f32 weight matrix, so the finite-difference
    /// gradient check is not fighting quant rounding. The quantised `wt` path is verified
    /// separately by the adjoint test in `gguf`.
    fn dense_w(out: usize, k_in: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..out * k_in)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // small integers / 16 -> exact in f32.
                (((s >> 40) as i64 % 17 - 8) as f32) / 16.0
            })
            .collect()
    }

    fn vecf(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as i64 % 21 - 10) as f32 / 32.0
            })
            .collect()
    }

    /// The backward pass must match finite differences of the forward pass -- the training
    /// analog of the adjoint test. A random linear functional `L = Σ c[o]·y[o]` has gradient
    /// `dL/dθ` computed by `backward` with `grad_y = c`; we compare it, for a sample of
    /// entries of A, B and x, against the central difference `(L(θ+ε) − L(θ−ε))/2ε`.
    ///
    /// This one check covers the entire hand-written autodiff chain at once: grad_B, grad_A,
    /// the low-rank contribution to grad_x, AND the frozen `Wᵀ·grad_y` term (here on an f32
    /// weight, so FD is clean).
    #[test]
    fn gradient_check_matches_finite_difference() {
        let (k_in, out, r) = (24usize, 16usize, 4usize);
        let wv = dense_w(out, k_in, 111);
        let w = W::F32(&wv);
        let x = vecf(k_in, 222);
        let c = vecf(out, 333); // the functional weights == grad_y

        let mut lora = Lora::new(r, k_in, out, 8.0, 444);
        // Give B some non-zero content or its gradient path is untested at init.
        lora.b = vecf(out * r, 555);

        // Analytic grads via backward with grad_y = c.
        let mut y = vec![0f32; out];
        let mut ax = vec![0f32; r];
        lora.forward(w, &x, &mut y, &mut ax);
        let mut grad_x = vec![0f32; k_in];
        lora.zero_grad();
        lora.backward(w, &x, &ax, &c, &mut grad_x);
        let (ga, gb) = (lora.ga.clone(), lora.gb.clone());

        // L(θ) = Σ c[o] y[o].
        let loss = |lora: &Lora, x: &[f32]| -> f64 {
            let mut y = vec![0f32; out];
            let mut ax = vec![0f32; r];
            lora.forward(w, x, &mut y, &mut ax);
            y.iter().zip(&c).map(|(&yo, &co)| yo as f64 * co as f64).sum()
        };

        let eps = 1e-2f32; // large-ish: f32 forward, small integer weights
        let check = |analytic: f32, fd: f64, what: &str, idx: usize| {
            let rel = (analytic as f64 - fd).abs() / fd.abs().max(1e-3);
            assert!(
                rel < 2e-2,
                "{what}[{idx}]: analytic {analytic} vs finite-diff {fd}, rel {rel}"
            );
        };

        // Sample a few A entries.
        for &idx in &[0usize, 5, 23, r * k_in - 1] {
            let mut lp = Lora::new(r, k_in, out, 8.0, 444);
            lp.b = lora.b.clone();
            lp.a = lora.a.clone();
            lp.a[idx] += eps;
            let hi = loss(&lp, &x);
            lp.a[idx] -= 2.0 * eps;
            let lo = loss(&lp, &x);
            check(ga[idx], (hi - lo) / (2.0 * eps as f64), "grad_A", idx);
        }
        // Sample a few B entries.
        for &idx in &[0usize, 3, out * r - 1] {
            let mut lp = Lora::new(r, k_in, out, 8.0, 444);
            lp.a = lora.a.clone();
            lp.b = lora.b.clone();
            lp.b[idx] += eps;
            let hi = loss(&lp, &x);
            lp.b[idx] -= 2.0 * eps;
            let lo = loss(&lp, &x);
            check(gb[idx], (hi - lo) / (2.0 * eps as f64), "grad_B", idx);
        }
        // And grad_x (covers the frozen Wᵀ term + the low-rank Aᵀ term together).
        for &idx in &[0usize, 7, k_in - 1] {
            let mut xp = x.clone();
            xp[idx] += eps;
            let hi = loss(&lora, &xp);
            xp[idx] -= 2.0 * eps;
            let lo = loss(&lora, &xp);
            check(grad_x[idx], (hi - lo) / (2.0 * eps as f64), "grad_x", idx);
        }
    }

    /// The whole SwiGLU FFN block backward -- three adapters plus the SiLU derivative and the
    /// residual-free composition -- must match finite differences. If the SiLU derivative or
    /// the gate/up grad split were wrong, the block would still produce plausible activations;
    /// only the gradient would be wrong, and only against FD does that show.
    #[test]
    fn ffn_gradient_check() {
        let (h, n, r) = (16usize, 20usize, 4usize);
        let wg = dense_w(n, h, 11);
        let wu = dense_w(n, h, 22);
        let wd = dense_w(h, n, 33);
        let (gw, uw, dw) = (W::F32(&wg), W::F32(&wu), W::F32(&wd));
        let x = vecf(h, 44);
        let c = vecf(h, 55); // functional weights == grad_y

        let mut ffn = FfnLora::new(h, n, r, 8.0, 66);
        // Non-zero B on each adapter so every grad path is exercised (B=0 at init hides them).
        ffn.gate.b = vecf(n * r, 1);
        ffn.up.b = vecf(n * r, 2);
        ffn.down.b = vecf(h * r, 3);

        let mut y = vec![0f32; h];
        let act = ffn.forward(gw, uw, dw, &x, &mut y);
        let mut grad_x = vec![0f32; h];
        ffn.zero_grad();
        ffn.backward(gw, uw, dw, &act, &c, &mut grad_x);

        let (gga, ggb) = (ffn.gate.ga.clone(), ffn.gate.gb.clone());
        let (uga, ugb) = (ffn.up.ga.clone(), ffn.up.gb.clone());
        let (dga, dgb) = (ffn.down.ga.clone(), ffn.down.gb.clone());

        let loss = |ffn: &FfnLora, x: &[f32]| -> f64 {
            let mut y = vec![0f32; h];
            let _ = ffn.forward(gw, uw, dw, x, &mut y);
            y.iter().zip(&c).map(|(&yo, &co)| yo as f64 * co as f64).sum()
        };
        let eps = 1e-2f32;
        let chk = |analytic: f32, hi: f64, lo: f64, what: &str, i: usize| {
            let fd = (hi - lo) / (2.0 * eps as f64);
            let rel = (analytic as f64 - fd).abs() / fd.abs().max(1e-3);
            assert!(rel < 3e-2, "{what}[{i}]: analytic {analytic} vs fd {fd}, rel {rel}");
        };

        // Perturb one weight of a given adapter matrix and read the FD. `grads` is the
        // analytic gradient for that matrix, whose length also picks the probe indices, so an
        // index can never fall outside a matrix regardless of its shape.
        let probe = |sel: fn(&mut FfnLora) -> &mut Vec<f32>, grads: &[f32], what: &str| {
            let len = grads.len();
            for &idx in &[0usize, len / 2, len - 1] {
                let mut fp = FfnLora::new(h, n, r, 8.0, 66);
                fp.gate.b = ffn.gate.b.clone();
                fp.up.b = ffn.up.b.clone();
                fp.down.b = ffn.down.b.clone();
                fp.gate.a = ffn.gate.a.clone();
                fp.up.a = ffn.up.a.clone();
                fp.down.a = ffn.down.a.clone();
                sel(&mut fp)[idx] += eps;
                let hi = loss(&fp, &x);
                sel(&mut fp)[idx] -= 2.0 * eps;
                let lo = loss(&fp, &x);
                chk(grads[idx], hi, lo, what, idx);
            }
        };

        probe(|f| &mut f.gate.a, &gga, "gate.A");
        probe(|f| &mut f.gate.b, &ggb, "gate.B");
        probe(|f| &mut f.up.a, &uga, "up.A");
        probe(|f| &mut f.up.b, &ugb, "up.B");
        probe(|f| &mut f.down.a, &dga, "down.A");
        probe(|f| &mut f.down.b, &dgb, "down.B");
        // grad_x through the whole block (covers all three frozen Wᵀ terms + SiLU).
        for &i in &[0usize, 9, h - 1] {
            let mut xp = x.clone();
            xp[i] += eps;
            let hi = loss(&ffn, &xp);
            xp[i] -= 2.0 * eps;
            let lo = loss(&ffn, &xp);
            chk(grad_x[i], hi, lo, "grad_x", i);
        }
    }

    /// RMS-norm backward must match finite differences of the forward norm -- it is on the
    /// training gradient path (output norm) and, like every backward here, is not trusted.
    #[test]
    fn rmsnorm_backward_gradient_check() {
        let n = 24usize;
        let x = vecf(n, 71);
        let w = vecf(n, 72);
        let c = vecf(n, 73); // functional weights == grad_y
        let eps = 1e-6f32;

        let mut y = vec![0f32; n];
        crate::ops::rmsnorm(&mut y, &x, &w, n, eps);
        let mut grad_x = vec![0f32; n];
        crate::ops::rmsnorm_backward(&mut grad_x, &c, &x, &w, n, eps);

        let loss = |x: &[f32]| -> f64 {
            let mut y = vec![0f32; n];
            crate::ops::rmsnorm(&mut y, x, &w, n, eps);
            y.iter().zip(&c).map(|(&yo, &co)| yo as f64 * co as f64).sum()
        };
        let e = 1e-3f32;
        for &i in &[0usize, 11, n - 1] {
            let mut xp = x.clone();
            xp[i] += e;
            let hi = loss(&xp);
            xp[i] -= 2.0 * e;
            let lo = loss(&xp);
            let fd = (hi - lo) / (2.0 * e as f64);
            let rel = (grad_x[i] as f64 - fd).abs() / fd.abs().max(1e-3);
            assert!(rel < 2e-2, "grad_x[{i}]: analytic {} vs fd {fd}, rel {rel}", grad_x[i]);
        }
    }

    #[test]
    fn adapter_save_load_round_trips() {
        let mut a = FfnLora::new(8, 12, 4, 8.0, 5);
        a.gate.a = vecf(4 * 8, 1);
        a.down.b = vecf(8 * 4, 2);
        let path = std::env::temp_dir().join("k3_adapter.loaa");
        let ps = path.to_str().unwrap();
        a.save(ps).unwrap();

        let mut b = FfnLora::new(8, 12, 4, 8.0, 999); // different seed -> different init
        b.load(ps).unwrap();
        assert_eq!(a.gate.a, b.gate.a);
        assert_eq!(a.down.b, b.down.b);
        assert_eq!(a.up.a, b.up.a);

        // Wrong shape must be refused, not silently reshaped.
        let mut wrong = FfnLora::new(8, 16, 4, 8.0, 1);
        assert!(wrong.load(ps).is_err(), "a shape mismatch must error");
        let _ = std::fs::remove_file(&path);
    }

    /// Adam on the adapter must actually descend: a full-batch loop on a fixed target should
    /// drive the loss down monotonically-ish and far. If the optimiser or the grad sign were
    /// wrong, the loss would rise or stall -- a coarse but decisive end-to-end check.
    #[test]
    fn adam_descends_a_regression_target() {
        let (k_in, out, r) = (12usize, 8usize, 4usize);
        let wv = dense_w(out, k_in, 1);
        let w = W::F32(&wv);
        let x = vecf(k_in, 2);
        let goal = vecf(out, 3); // target output vector

        let mut lora = Lora::new(r, k_in, out, 8.0, 4);
        lora.b = vecf(out * r, 5);

        let mse_and_grad = |lora: &mut Lora| -> f32 {
            let mut y = vec![0f32; out];
            let mut ax = vec![0f32; r];
            lora.forward(w, &x, &mut y, &mut ax);
            // grad_y of 0.5*Σ(y-goal)^2 is (y-goal).
            let gy: Vec<f32> = y.iter().zip(&goal).map(|(&yo, &g)| yo - g).collect();
            let loss: f32 = gy.iter().map(|&e| 0.5 * e * e).sum();
            let mut gx = vec![0f32; k_in];
            lora.zero_grad();
            lora.backward(w, &x, &ax, &gy, &mut gx);
            loss
        };

        let first = mse_and_grad(&mut lora);
        // (grad already accumulated by the call above for step 1)
        let mut t = 0u64;
        let mut last = first;
        for _ in 0..400 {
            t += 1;
            lora.adam_step(5e-2, 0.9, 0.999, 1e-8, t, 1.0);
            last = mse_and_grad(&mut lora);
        }
        assert!(last < first * 0.05, "loss {first} -> {last}: Adam did not descend");
    }

    /// The response mask must cover EXACTLY the response tokens: every prompt position
    /// excluded, every response position included, aligned to next-token prediction. An
    /// off-by-one here trains the model on the wrong half of the sequence and is invisible in
    /// the output -- fluent, wrong.
    #[test]
    fn the_mask_selects_exactly_the_response() {
        // 4 prompt tokens, 3 response tokens (ids 100,101,102).
        let ex = Example { ids: vec![1, 2, 3, 4, 100, 101, 102], prompt_len: 4 };
        assert_eq!(ex.scored_tokens(), 3);
        let (input, target, mask) = ex.view();
        assert_eq!(input, vec![1, 2, 3, 4, 100, 101]);
        assert_eq!(target, vec![2, 3, 4, 100, 101, 102]);
        // Position t predicts target[t]; trained iff target[t] is a response token.
        // target = [2,3,4,100,101,102] -> the last three are response.
        assert_eq!(mask, vec![false, false, false, true, true, true]);
        // The number of trained positions equals the response length.
        assert_eq!(mask.iter().filter(|&&m| m).count(), ex.scored_tokens());
    }

    /// A boundary BPE merge (prompt is NOT a clean prefix of full) must degrade gracefully:
    /// common_prefix_len finds the real split rather than trusting the prompt length.
    #[test]
    fn a_boundary_merge_shifts_by_at_most_one() {
        let prompt = [10u32, 11, 12, 99]; // last token is the assistant-open "\n"
        // In `full`, the boundary token 99 merged with the response start into 77.
        let full = [10u32, 11, 12, 77, 55, 56];
        let plen = common_prefix_len(&prompt, &full);
        assert_eq!(plen, 3, "the common prefix stops at the merged boundary token");
        let ex = Example { ids: full.to_vec(), prompt_len: plen };
        // The merged token 77 is now (conservatively) treated as the first response token.
        // No mid-response token is ever mislabelled as prompt, which is the property that
        // matters.
        assert_eq!(ex.scored_tokens(), 3);
    }

    #[test]
    fn jsonl_round_trips_and_rejects_malformed_rows() {
        let dir = std::env::temp_dir();
        let good = dir.join("k3_train_good.jsonl");
        std::fs::write(
            &good,
            "{\"system\":\"s1\",\"user\":\"u1\",\"response\":\"r1\",\"extra\":7}\n\
             \n\
             {\"system\":\"s2\",\"user\":\"u2\",\"response\":\"r2\"}\n",
        )
        .unwrap();
        let rows = load_jsonl(good.to_str().unwrap()).unwrap();
        assert_eq!(rows.len(), 2, "blank line skipped, both records kept");
        assert_eq!(rows[0].response, "r1");
        assert_eq!(rows[1].user, "u2");

        let bad = dir.join("k3_train_bad.jsonl");
        std::fs::write(&bad, "{\"system\":\"s\",\"user\":\"u\"}\n").unwrap();
        assert!(
            load_jsonl(bad.to_str().unwrap()).is_err(),
            "a row missing `response` must error, not silently shrink the set"
        );
        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
    }

    /// Cross-entropy: a confident correct prediction is near-zero loss with near-zero grad;
    /// masking must exclude prompt positions entirely (zero grad there, and they do not
    /// dilute the mean).
    #[test]
    fn cross_entropy_masks_the_prompt() {
        let (seq, vocab) = (4usize, 6usize);
        let mut logits = vec![0f32; seq * vocab];
        // Make each position confidently predict token (t+1)%vocab.
        for t in 0..seq {
            logits[t * vocab + (t + 1) % vocab] = 20.0;
        }
        let target: Vec<u32> = (0..seq).map(|t| ((t + 1) % vocab) as u32).collect();
        // Mask out the first two (the "prompt"); train only the last two.
        let mask = vec![false, false, true, true];
        let mut grad = vec![0f32; seq * vocab];
        let loss = cross_entropy(&logits, &target, &mask, vocab, &mut grad);
        assert!(loss < 1e-6, "confident correct predictions -> ~0 loss, got {loss}");

        // Masked positions have exactly zero gradient.
        for t in 0..2 {
            for k in 0..vocab {
                assert_eq!(grad[t * vocab + k], 0.0, "masked position {t} must not get grad");
            }
        }
        // A wrong target at an unmasked position produces a real, non-zero loss and grad.
        let mut tgt2 = target.clone();
        tgt2[2] = 0; // position 2 confidently predicts token 3, gold now says 0
        let mut grad2 = vec![0f32; seq * vocab];
        let loss2 = cross_entropy(&logits, &tgt2, &mask, vocab, &mut grad2);
        assert!(loss2 > 1.0, "a confident wrong prediction must cost, got {loss2}");
    }
}
