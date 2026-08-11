// SPDX-License-Identifier: Apache-2.0
//
// Token sampling, with the determinism this project's gates depend on.
//
// WHY THE SEED IS NOT OPTIONAL
//     Every correctness check here rests on running the same thing twice and diffing:
//     speculative decoding is proven exact by `diff <(gen) <(gen)`, quantisation damage is
//     measured the same way, and `rust-golden` asserts byte-identical logits. Sampling
//     from an unseeded source would delete that whole apparatus.
//
//     So every request carries a seed -- supplied, or generated and reported back -- and
//     the same seed with the same prompt reproduces exactly. Nothing here reads the clock
//     or the OS entropy pool on its own.
//
// TEMPERATURE ZERO IS EXACTLY THE OLD PATH
//     `pick` returns the plain argmax when `temperature <= 0`, before any filter runs and
//     without touching the RNG. That is not an optimisation: it is what lets every
//     existing measurement and gate keep reproducing bit for bit after sampling exists.
//
// !! THE HAZARD, recorded where someone will hit it !!
//     Speculative decoding accepts a draft only when it equals the GREEDY argmax
//     (`v4run.rs`), and `dspark.rs` says greedy drafting is exact *because* the verifier is
//     greedy. With `temperature > 0` that identity breaks: sampling the verifier
//     independently of the draft silently changes the output distribution while still
//     reading fluently. Either both must be drawn from the same seeded stream, or
//     speculation must be off. The server takes the second option for now -- see
//     `SampleParams::speculation_safe`.

/// PCG-XSH-RR 64/32, the "minimal" variant from O'Neill's PCG paper.
///
/// Hand-rolled rather than pulled in: it is thirty lines, the crate carries four
/// dependencies on purpose, and a generator whose whole job is reproducibility should be
/// something we can read.
///
/// The multiplier and the shift/rotate schedule are the published constants. Note what is
/// NOT claimed: these have not been pinned against the reference implementation's output
/// vectors, so the tests below check reproducibility, stream independence and uniformity
/// rather than asserting specific numbers I would have written on both sides.
#[derive(Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, stream: u64) -> Pcg32 {
        let mut r = Pcg32 { state: 0, inc: (stream << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1). 24 bits of mantissa, so every value is exactly representable and
    /// the result never rounds to 1.0 -- which would put a sample past the end of the CDF.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Clone, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        // Greedy by default, so an unconfigured request reproduces the engine's historical
        // behaviour exactly rather than something merely similar.
        SampleParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

impl SampleParams {
    /// Whether speculative decoding may be left on. See the hazard note at the top of this
    /// file: the accept test is an equality against the greedy argmax, which only holds
    /// when the verifier is greedy.
    pub fn speculation_safe(&self) -> bool {
        self.temperature <= 0.0
    }
}

pub struct Sampler {
    rng: Pcg32,
    pub p: SampleParams,
}

impl Sampler {
    pub fn new(p: SampleParams) -> Sampler {
        Sampler { rng: Pcg32::new(p.seed, 0xda3e_39cb_94b9_5bdb), p }
    }

    /// Choose the next token. `history` is the tokens so far, for the repetition penalty.
    ///
    /// Filter order is penalty -> temperature -> top-k -> top-p -> min-p -> draw. Order is
    /// not arbitrary and changes results: applying top-p before temperature nucleates on
    /// the unscaled distribution, which is a different sampler with the same knob names.
    pub fn pick(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        // Greedy: no filters, no RNG draw, no divergence from the pre-sampling engine.
        if self.p.temperature <= 0.0 {
            return argmax(logits);
        }

        if self.p.repeat_penalty != 1.0 && self.p.repeat_last_n > 0 {
            let from = history.len().saturating_sub(self.p.repeat_last_n);
            for &t in &history[from..] {
                if let Some(l) = logits.get_mut(t as usize) {
                    // Divide when positive, multiply when negative: scaling a negative
                    // logit down would REWARD the token it is meant to discourage.
                    *l = if *l > 0.0 { *l / self.p.repeat_penalty } else { *l * self.p.repeat_penalty };
                }
            }
        }

        let inv = 1.0 / self.p.temperature;
        for l in logits.iter_mut() {
            *l *= inv;
        }

        // Sort candidates by logit, descending. The vocabulary is ~250k, so this is the
        // expensive part of sampling -- and still trivial beside a forward pass.
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));

        if self.p.top_k > 0 && self.p.top_k < idx.len() {
            idx.truncate(self.p.top_k);
        }

        // Softmax over the surviving candidates, shifted by the max for stability.
        let top = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx.iter().map(|&i| (logits[i as usize] - top).exp()).collect();
        let z: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= z;
        }

        // min-p first: it is a floor relative to the best candidate, so it must see the
        // full distribution rather than whatever top-p already removed.
        if self.p.min_p > 0.0 {
            let floor = self.p.min_p * probs[0];
            let keep = probs.iter().take_while(|&&p| p >= floor).count().max(1);
            idx.truncate(keep);
            probs.truncate(keep);
        }

        if self.p.top_p < 1.0 {
            let mut acc = 0.0;
            let mut keep = 0usize;
            for p in &probs {
                acc += *p;
                keep += 1;
                if acc >= self.p.top_p {
                    break;
                }
            }
            idx.truncate(keep.max(1));
            probs.truncate(keep.max(1));
        }

        // Draw from the surviving mass. Renormalising is what makes the draw correct after
        // truncation; sampling against the original total would bias toward the tail.
        let total: f32 = probs.iter().sum();
        let mut r = self.rng.next_f32() * total;
        for (k, p) in probs.iter().enumerate() {
            r -= *p;
            if r <= 0.0 {
                return idx[k];
            }
        }
        // Floating-point slack only: the loop above consumes the whole mass in exact
        // arithmetic. Falling back to the most likely candidate keeps this total.
        idx[0]
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole file exists for: same seed, same stream, every time.
    #[test]
    fn the_same_seed_reproduces_the_same_stream() {
        let a: Vec<u32> = (0..64).scan(Pcg32::new(42, 54), |r, _| Some(r.next_u32())).collect();
        let b: Vec<u32> = (0..64).scan(Pcg32::new(42, 54), |r, _| Some(r.next_u32())).collect();
        assert_eq!(a, b);
        let c: Vec<u32> = (0..64).scan(Pcg32::new(43, 54), |r, _| Some(r.next_u32())).collect();
        assert_ne!(a, c, "a different seed must give a different stream");
        let d: Vec<u32> = (0..64).scan(Pcg32::new(42, 55), |r, _| Some(r.next_u32())).collect();
        assert_ne!(a, d, "the stream parameter must actually select a stream");
    }

    /// Not a proof of quality -- a sanity floor. A generator stuck on a constant, or one
    /// whose rotate is wrong enough to collapse the range, fails this.
    #[test]
    fn the_generator_is_broadly_uniform() {
        let mut r = Pcg32::new(7, 0);
        let mut bins = [0u32; 16];
        const N: u32 = 160_000;
        for _ in 0..N {
            let f = r.next_f32();
            assert!((0.0..1.0).contains(&f), "next_f32 must stay in [0, 1), got {f}");
            bins[(f * 16.0) as usize] += 1;
        }
        let expect = N as f64 / 16.0;
        for (i, b) in bins.iter().enumerate() {
            let dev = (*b as f64 - expect).abs() / expect;
            assert!(dev < 0.10, "bin {i} deviates {:.1}% from uniform", dev * 100.0);
        }
    }

    /// Temperature zero must be the OLD path exactly -- no filters, no RNG draw. This is
    /// what lets every existing gate keep reproducing after sampling exists.
    #[test]
    fn temperature_zero_is_argmax_and_never_touches_the_rng() {
        let mut logits = vec![0.1, 0.9, 0.3, 0.7];
        let p = SampleParams { temperature: 0.0, top_k: 1, top_p: 0.1, ..Default::default() };
        let mut s = Sampler::new(p);
        let before = s.rng.state;
        assert_eq!(s.pick(&mut logits, &[]), 1);
        assert_eq!(s.rng.state, before, "greedy must not advance the generator");
        // And the filters, which would otherwise have been applied, changed nothing.
        assert_eq!(logits, vec![0.1, 0.9, 0.3, 0.7], "greedy must not rescale logits");
    }

    #[test]
    fn speculation_is_only_declared_safe_when_greedy() {
        assert!(SampleParams { temperature: 0.0, ..Default::default() }.speculation_safe());
        assert!(!SampleParams { temperature: 0.7, ..Default::default() }.speculation_safe());
    }

    /// top_k = 1 is greedy by another route: whatever the temperature, only the best
    /// candidate survives the filter.
    #[test]
    fn top_k_one_always_yields_the_best_token() {
        for seed in [1u64, 2, 3, 99] {
            let mut logits = vec![0.1, 5.0, 0.3, 4.9];
            let p = SampleParams { temperature: 2.0, top_k: 1, seed, ..Default::default() };
            assert_eq!(Sampler::new(p).pick(&mut logits, &[]), 1, "seed {seed}");
        }
    }

    /// The repetition penalty must push a token DOWN whether its logit is positive or
    /// negative. Multiplying a negative logit by the penalty raises it, rewarding exactly
    /// the token it was meant to suppress.
    #[test]
    fn the_repetition_penalty_lowers_both_signs() {
        let p = SampleParams {
            temperature: 1.0,
            repeat_penalty: 2.0,
            repeat_last_n: 8,
            ..Default::default()
        };
        let mut logits = vec![4.0f32, -4.0];
        let mut s = Sampler::new(p);
        s.pick(&mut logits, &[0, 1]);
        assert!(logits[0] < 4.0, "a positive logit must fall: {}", logits[0]);
        assert!(logits[1] < -4.0, "a negative logit must fall too: {}", logits[1]);
    }

    /// A given seed and prompt must produce one answer, repeatedly -- the property a
    /// server has to promise for anything to be debuggable.
    #[test]
    fn sampling_is_reproducible_for_a_fixed_seed() {
        let draw = || {
            let p = SampleParams { temperature: 0.8, seed: 12345, ..Default::default() };
            let mut s = Sampler::new(p);
            (0..32)
                .map(|i| {
                    let mut l: Vec<f32> = (0..50).map(|k| ((k * 7 + i) % 13) as f32 * 0.3).collect();
                    s.pick(&mut l, &[])
                })
                .collect::<Vec<u32>>()
        };
        assert_eq!(draw(), draw());
    }
}
