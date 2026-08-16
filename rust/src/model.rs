// SPDX-License-Identifier: Apache-2.0
//
// One interface over every architecture, so the server does not know which it is running.
//
// WHY THE SEAM IS HERE AND NOT LOWER
//     `serve.rs` was written against `v4run` concretely -- `Engine`, `Session`, `Params`
//     and `generate_on` all appear in its signatures -- so the working fast model could not
//     be served at all. The obvious fix is a trait over "run a forward pass", but the
//     obvious *placement* is wrong: `v4run::generate_on` does batched prefill and
//     speculative verification inside its loop, and a one-token `step` would throw both
//     away.
//
//     So the seam is `feed(ids) -> logits for the last position`. An implementation may
//     batch the whole slice, verify speculation inside it, or walk one token at a time --
//     the caller cannot tell, and does not need to. That is also exactly the shape prefix
//     caching wants: "advance this conversation by these tokens".
//
// WHAT LIVES ABOVE THE TRAIT
//     Sampling, stop sequences and streaming. Those are identical for every architecture,
//     and they currently exist only inside `v4run::generate_on`. Duplicating them per
//     architecture would mean two places for the sampling-versus-speculation hazard to be
//     got wrong, so they are lifted here instead.

use crate::sample::{SampleParams, Sampler};

/// What the server needs from a loaded model, whichever stack implements it.
pub trait Model {
    fn vocab(&self) -> usize;

    /// The token that ends a turn, if the checkpoint declares one.
    fn eos(&self) -> Option<u32>;

    fn tok(&self) -> Option<&crate::tok::Tok>;

    /// The model's own chat template, or `Plain` for a base model.
    fn template(&self) -> &crate::chat::Template;

    /// Advance the conversation by `ids` and write the logits that follow the LAST one.
    ///
    /// Implementations are free to batch. `ids` is never empty.
    fn feed(&mut self, ids: &[u32], logits: &mut [f32]) -> Result<(), String>;

    /// Forget the conversation.
    fn reset(&mut self);

    /// Restore the longest cached prefix of `ids` and return how many leading tokens are
    /// already applied, so the caller feeds only the remainder.
    ///
    /// The default resets and returns 0: an implementation with no cache is simply always
    /// cold, and `generate` needs no special case for it.
    fn begin(&mut self, ids: &[u32]) -> usize {
        let _ = ids;
        self.reset();
        0
    }

    /// Offer the state produced by `ids` to the cache. Called after the prompt is applied,
    /// not after generation, so what is stored is exactly a prefix a later turn can match.
    fn commit(&mut self, ids: &[u32]) {
        let _ = ids;
    }

    /// How many tokens this conversation already holds, for context-limit reporting.
    fn pos(&self) -> usize;

    /// (expert GB, conversation GB, rebalances) for a backend with an adaptive split.
    fn split_report(&self) -> Option<(f64, f64, u64)> {
        None
    }
}

/// Why generation stopped. The caller reports this as `finish_reason`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Finish {
    Eos,
    Length,
    Stop,
    /// The sink asked to stop -- a stop sequence matched, or the client hung up.
    Sink,
}

impl Finish {
    pub fn as_str(self) -> &'static str {
        match self {
            // OpenAI collapses "the model chose to end" and "a stop string matched" into
            // one value; the distinction is kept internally because only one of them means
            // the model was interrupted.
            Finish::Eos | Finish::Stop | Finish::Sink => "stop",
            Finish::Length => "length",
        }
    }
}

pub struct GenParams {
    pub max_tokens: usize,
    pub sample: Option<SampleParams>,
}

/// Prompt-then-decode, over any `Model`.
///
/// `prompt` is fed in one call so an implementation that batches can do so; generated
/// tokens are then fed one at a time, because each depends on the last.
///
/// `sink` returns false to stop -- a stop sequence, or a disconnected client. At seconds
/// per token, continuing after either would burn minutes producing output nobody reads.
pub fn generate(
    m: &mut dyn Model,
    prompt: &[u32],
    p: &GenParams,
    sink: &mut dyn FnMut(u32, &str) -> bool,
) -> Result<Finish, String> {
    if prompt.is_empty() {
        return Err("cannot generate from an empty prompt".into());
    }
    let mut logits = vec![0f32; m.vocab()];
    // Reuse whatever of this conversation is already computed. `begin` returns how many
    // leading tokens it restored; the rest is prefill.
    let done = m.begin(prompt);
    debug_assert!(done < prompt.len(), "begin must leave at least one token to feed");
    m.feed(&prompt[done..], &mut logits)?;
    m.commit(prompt);

    let mut sampler = p.sample.clone().map(Sampler::new);
    // `history` is what the repetition penalty looks back over, so it must include the
    // prompt and not just what has been generated.
    let mut history: Vec<u32> = prompt.to_vec();

    for _ in 0..p.max_tokens {
        let id = match sampler.as_mut() {
            Some(s) => s.pick(&mut logits, &history),
            None => argmax(&logits),
        };
        if Some(id) == m.eos() {
            return Ok(Finish::Eos);
        }
        let piece = match m.tok() {
            Some(t) => t.piece(id)?,
            None => String::new(),
        };
        if !sink(id, &piece) {
            return Ok(Finish::Sink);
        }
        history.push(id);
        m.feed(&[id], &mut logits)?;
        if !logits.iter().all(|v| v.is_finite()) {
            return Err(format!("non-finite logits after token {id}"));
        }
    }
    Ok(Finish::Length)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

// ---------------------------------------------------------------------------
// qwen35moe
// ---------------------------------------------------------------------------

/// A loaded `qwen35moe` model: the resident trunk, the streaming expert cache, and the
/// conversation state.
pub struct Qwen35<'a> {
    st: &'a crate::st::St,
    trunk: crate::qwen35run::Trunk,
    cache: crate::cache::Cache,
    sess: crate::qwen35run::Session,
    tok: crate::tok::Tok,
    template: crate::chat::Template,
    eos: u32,
    prefix: crate::prefix::PrefixCache<crate::prefix::Qwen35State>,
    fp: crate::prefix::Fingerprint,
    bal: crate::prefix::Balancer,
    topk: usize,
    /// Prompt-chunk width. `None` derives it from the cache; a value overrides.
    width: Option<usize>,
}

impl<'a> Qwen35<'a> {
    pub fn load(
        st: &'a crate::st::St,
        cache_gb: f64,
        conv_gb: f64,
        max_ctx: usize,
    ) -> Result<Qwen35<'a>, String> {
        let meta = st.meta.as_ref().ok_or("qwen35moe needs gguf metadata")?;
        let cfg = crate::qwen35::Cfg::from_meta(meta)?;
        let trunk = crate::qwen35run::Trunk::load(st, &cfg, max_ctx)?;

        // Slot size is the MAX over every layer, never layer 0's. This build quantises
        // ffn_down_exps to Q6_K in half its layers and Q4_K in the other half, so a slot
        // sized from one layer is too small for the rest -- and the cache refuses the
        // admission rather than corrupting it, which at least fails loudly.
        // For a dense model the cache's unit is a whole LAYER's feed-forward rather than
        // one expert of many -- 165 MB against 2 MB -- but nothing else about the cache
        // changes: same slots, same sweep-aware eviction, same admission check.
        let src = if cfg.is_dense() {
            crate::cache::gguf_dense_ffn_src
        } else {
            crate::cache::gguf_expert_src
        };
        let mut slot = 0usize;
        for l in 0..cfg.n_layers {
            let r = crate::cache::locate(st, &src(l, 0))
                .ok_or_else(|| format!("cannot locate layer {l} feed-forward"))?;
            slot = slot.max(crate::cache::slot_need(&r));
        }
        // One "expert" per layer, always selected. `prefill_width`'s occupancy formula then
        // reduces to U(k) = 1, which is exactly right: a chunk of any width draws the same
        // single FFN per layer, so width is bounded by activations alone.
        let (n_exp, topk) =
            if cfg.is_dense() { (1, 1) } else { (cfg.n_experts, cfg.topk) };
        // Arena budget. `cache_gb <= 0` means AUTO: size to the largest swap-safe slice of RAM
        // now that the trunk is resident (so MemAvailable already excludes it). An explicit
        // value is honoured but CLAMPED down if it would swap -- a server that pages the trunk
        // out onto the same disk it streams from thrashes instead of failing, which is worse
        // than a smaller cache. `total_stream` (every streamed unit) is the useful ceiling.
        let total_stream = (cfg.n_layers as i64) * (n_exp as i64) * (slot as i64);
        let min_budget = ((topk + 1) * slot) as i64;
        let margin = std::env::var("RUSTLM_MEM_MARGIN_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|g| (g * 1e9) as u64)
            .unwrap_or(2_500_000_000);
        let avail = crate::cache::mem_available_bytes().unwrap_or(0) as f64 / 1e9;
        let budget = if cache_gb <= 0.0 {
            let b = crate::cache::auto_budget_bytes(margin, min_budget, total_stream, 5_000_000_000);
            println!(
                "expert cache: auto {:.2} GB (largest swap-safe; MemAvailable {:.1} GB - {:.1} GB margin)",
                b as f64 / 1e9, avail, margin as f64 / 1e9
            );
            b
        } else {
            let req = (cache_gb * 1e9) as i64;
            let safe = crate::cache::auto_budget_bytes(margin, min_budget, total_stream, req);
            if req > safe {
                println!(
                    "expert cache: --cache-gb {cache_gb:.1} risks swap; clamped to {:.2} GB (swap-safe max)",
                    safe as f64 / 1e9
                );
                safe
            } else {
                req
            }
        };
        let cache = crate::cache::Cache::new(budget, slot, n_exp, topk)?;

        let tok = crate::tok::Tok::from_gguf(meta)?;
        let eos = tok.eos;
        let template = crate::chat::Template::from_gguf(meta);
        let sess = crate::qwen35run::Session::new(&trunk);
        let fp = crate::prefix::Fingerprint::of(&cfg);
        Ok(Qwen35 {
            st,
            trunk,
            cache,
            sess,
            tok,
            template,
            eos,
            prefix: crate::prefix::PrefixCache::new((conv_gb * 1e9) as usize),
            fp,
            // The two caches share whatever they were given between them. Uses the RESOLVED
            // arena budget (after auto-sizing/clamping), not the raw request.
            bal: crate::prefix::Balancer::new(budget as usize + (conv_gb * 1e9) as usize, 8),
            topk: cfg.topk,
            width: None,
        })
    }

    pub fn cfg(&self) -> &crate::qwen35::Cfg {
        &self.trunk.cfg
    }

    /// Override the prompt-chunk width. Zero restores the derived one.
    pub fn set_width(&mut self, w: usize) {
        self.width = if w == 0 { None } else { Some(w) };
    }

    /// The width `feed` will actually use.
    pub fn width(&self) -> usize {
        self.width.unwrap_or_else(|| {
            let c = &self.trunk.cfg;
            crate::v4run::prefill_width(
                self.cache.nslot(),
                c.n_experts.max(1),
                c.topk.max(1),
                c.hidden,
                c.n_layers,
                c.ffn_width(),
            )
        })
    }

    pub fn nslot(&self) -> usize {
        self.cache.nslot()
    }

    /// (bytes read, hits, misses) since the last reset -- what a prefill measurement needs.
    pub fn io_stats(&self) -> (u64, u64, u64) {
        (self.cache.bytes_read, self.cache.hits, self.cache.misses)
    }

    pub fn reset_io_stats(&mut self) {
        self.cache.reset_stats();
    }

    pub fn bytes(&self) -> usize {
        self.trunk.bytes
    }

    pub fn prefix_stats(&self) -> (u64, u64, u64, usize) {
        (self.prefix.hits, self.prefix.misses, self.prefix.tokens_saved, self.prefix.used())
    }

    /// (expert bytes, conversation budget, moves made) -- for `/health`, so the split is
    /// never a mystery to whoever is wondering why a request was slow.
    pub fn split(&self) -> (usize, usize, u64) {
        (self.cache.bytes(), self.prefix.budget(), self.bal.moves)
    }

    /// Decide whether to move a slab between the caches. Called once per request.
    ///
    /// The expert resize is DESTRUCTIVE -- it drops every cached expert -- so this only
    /// fires on interval boundaries and only on a clear signal. See `Balancer`.
    fn rebalance(&mut self) {
        use crate::prefix::Move;
        let (e, c) = (self.cache.bytes(), self.prefix.budget());
        let mv = self.bal.observe(e, c, self.cache.window_hit_rate(), self.prefix.window_hit_rate());
        match mv {
            Move::Hold => return,
            Move::ToConversations(n) => {
                // Shrink the experts FIRST, so the machine never holds both at once. On a
                // 15 GB box the overlap is what would trigger the OOM this engine has an
                // admission check to avoid.
                if self.cache.resize((e - n) as i64, self.topk) {
                    self.prefix.set_budget(c + n);
                    eprintln!("  [split] experts {:.1} -> {:.1} GB, conversations {:.1} -> {:.1} GB",
                              e as f64 / 1e9, (e - n) as f64 / 1e9,
                              c as f64 / 1e9, (c + n) as f64 / 1e9);
                }
            }
            Move::ToExperts(n) => {
                self.prefix.set_budget(c - n);
                if !self.cache.resize((e + n) as i64, self.topk) {
                    // The resize was refused, so give the memory back rather than leaving
                    // it owned by nobody.
                    self.prefix.set_budget(c);
                } else {
                    eprintln!("  [split] experts {:.1} -> {:.1} GB, conversations {:.1} -> {:.1} GB",
                              e as f64 / 1e9, (e + n) as f64 / 1e9,
                              c as f64 / 1e9, (c - n) as f64 / 1e9);
                }
            }
        }
        self.cache.reset_stats();
        self.prefix.reset_stats();
    }
}

impl Model for Qwen35<'_> {
    fn vocab(&self) -> usize {
        self.trunk.cfg.vocab
    }
    fn eos(&self) -> Option<u32> {
        Some(self.eos)
    }
    fn tok(&self) -> Option<&crate::tok::Tok> {
        Some(&self.tok)
    }
    fn template(&self) -> &crate::chat::Template {
        &self.template
    }
    fn pos(&self) -> usize {
        self.sess.pos
    }
    fn split_report(&self) -> Option<(f64, f64, u64)> {
        let (e, c, m) = self.split();
        Some((e as f64 / 1e9, c as f64 / 1e9, m))
    }
    fn reset(&mut self) {
        self.sess = crate::qwen35run::Session::new(&self.trunk);
    }

    fn begin(&mut self, ids: &[u32]) -> usize {
        match self.prefix.find(ids, self.fp) {
            Some(i) => {
                let s = self.prefix.get(i);
                let n = s.ids.len();
                s.state.restore(&mut self.sess);
                eprintln!(
                    "  [prefix] {n}/{} tokens restored, {} to prefill",
                    ids.len(),
                    ids.len() - n
                );
                n
            }
            None => {
                self.reset();
                0
            }
        }
    }

    fn commit(&mut self, ids: &[u32]) {
        // Rebalance here, not in `begin`: by now this request's hits and misses are on
        // record, so the decision uses the window that just closed.
        self.rebalance();
        // Snapshot AT the prompt boundary. A later turn's prompt begins with exactly this
        // sequence, so this is the position a match can be made at; snapshotting mid-
        // generation would store a prefix no future request contains.
        let st = crate::prefix::Qwen35State::take(&self.sess);
        self.prefix.insert(crate::prefix::Snapshot::new(st, ids, self.fp));
    }

    /// Batched, in chunks sized to the expert cache.
    ///
    /// The width comes from the same occupancy bound the V4 engine uses: a chunk's union
    /// of routed experts must fit in a quarter of the cache, or the tail of the chunk
    /// evicts the experts its head just fetched and the batching costs more than it saves.
    fn feed(&mut self, ids: &[u32], logits: &mut [f32]) -> Result<(), String> {
        let w = self.width().max(1);
        for chunk in ids.chunks(w) {
            crate::qwen35run::step_many(
                &self.trunk,
                &mut self.sess,
                self.st,
                &mut self.cache,
                chunk,
                logits,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that returns fixed logits, so the loop above can be tested without a
    /// checkpoint. `feed` records what it was given, which is how the batching contract
    /// gets asserted at all.
    struct Fake {
        fed: Vec<Vec<u32>>,
        next: u32,
        eos: Option<u32>,
        tmpl: crate::chat::Template,
    }

    impl Fake {
        fn new(next: u32, eos: Option<u32>) -> Fake {
            Fake { fed: Vec::new(), next, eos, tmpl: crate::chat::Template::Plain }
        }
    }

    impl Model for Fake {
        fn vocab(&self) -> usize {
            8
        }
        fn eos(&self) -> Option<u32> {
            self.eos
        }
        fn tok(&self) -> Option<&crate::tok::Tok> {
            None
        }
        fn template(&self) -> &crate::chat::Template {
            &self.tmpl
        }
        fn pos(&self) -> usize {
            self.fed.iter().map(Vec::len).sum()
        }
        fn reset(&mut self) {
            self.fed.clear();
        }
        fn feed(&mut self, ids: &[u32], logits: &mut [f32]) -> Result<(), String> {
            self.fed.push(ids.to_vec());
            logits.fill(0.0);
            logits[self.next as usize] = 1.0;
            Ok(())
        }
    }

    /// The prompt must arrive as ONE call. An implementation that batches can only do so
    /// if the loop hands it the whole slice, and feeding token-by-token here would quietly
    /// disable batched prefill for every architecture.
    #[test]
    fn the_whole_prompt_is_fed_in_a_single_call() {
        let mut m = Fake::new(5, None);
        let mut sink = |_: u32, _: &str| true;
        generate(&mut m, &[1, 2, 3, 4], &GenParams { max_tokens: 2, sample: None }, &mut sink)
            .unwrap();
        assert_eq!(m.fed[0], vec![1, 2, 3, 4], "the prompt must not be split");
        // Generated tokens follow one at a time -- each depends on the last.
        assert_eq!(&m.fed[1..], &[vec![5], vec![5]]);
    }

    #[test]
    fn eos_stops_generation_and_is_not_emitted() {
        let mut m = Fake::new(5, Some(5));
        let mut seen = Vec::new();
        let mut sink = |id: u32, _: &str| {
            seen.push(id);
            true
        };
        let f = generate(&mut m, &[1], &GenParams { max_tokens: 9, sample: None }, &mut sink)
            .unwrap();
        assert_eq!(f, Finish::Eos);
        assert!(seen.is_empty(), "the eos token itself must never reach the client");
    }

    #[test]
    fn max_tokens_is_a_hard_cap() {
        let mut m = Fake::new(5, None);
        let mut n = 0;
        let mut sink = |_: u32, _: &str| {
            n += 1;
            true
        };
        let f = generate(&mut m, &[1], &GenParams { max_tokens: 3, sample: None }, &mut sink)
            .unwrap();
        assert_eq!((f, n), (Finish::Length, 3));
    }

    /// A sink returning false must stop immediately. At seconds per token, continuing
    /// after a stop sequence or a hung-up client burns minutes for nothing.
    #[test]
    fn a_sink_that_returns_false_stops_at_once() {
        let mut m = Fake::new(5, None);
        let mut n = 0;
        let mut sink = |_: u32, _: &str| {
            n += 1;
            false
        };
        let f = generate(&mut m, &[1], &GenParams { max_tokens: 99, sample: None }, &mut sink)
            .unwrap();
        assert_eq!((f, n), (Finish::Sink, 1));
    }

    #[test]
    fn an_empty_prompt_is_refused_rather_than_generating_from_nothing() {
        let mut m = Fake::new(5, None);
        let mut sink = |_: u32, _: &str| true;
        assert!(generate(&mut m, &[], &GenParams { max_tokens: 1, sample: None }, &mut sink)
            .is_err());
    }

    /// OpenAI reports a stop sequence and an eos identically; internally they differ
    /// because only one means the model was interrupted.
    #[test]
    fn finish_reasons_map_onto_the_two_openai_values() {
        assert_eq!(Finish::Eos.as_str(), "stop");
        assert_eq!(Finish::Stop.as_str(), "stop");
        assert_eq!(Finish::Sink.as_str(), "stop");
        assert_eq!(Finish::Length.as_str(), "length");
    }
}
