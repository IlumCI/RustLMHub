// SPDX-License-Identifier: Apache-2.0
//
// Reusing conversation state across requests.
//
// THE PROBLEM, MEASURED
//     RustLM Code's first request carried 9862 prompt tokens. Every request re-prefills
//     from token zero, and at this engine's prefill rate that is hours before the first
//     output token. But an agent's turn N+1 is turn N plus a short append: measured on the
//     real Qwen chat template, turn 2 shares 22 of its 39 tokens with turn 1, and with a
//     10k-token system prompt the shared fraction approaches 98%.
//
//     So the work is already done; it is being thrown away between requests.
//
// WHY A RECURRENT MODEL MAKES THIS BOTH EASIER AND HARDER
//     Easier: 30 of qwen35moe's 40 blocks hold FIXED-size state -- 66 MB no matter how long
//     the conversation -- so a whole 10k-token conversation snapshots for ~475 MB, about 8
//     of them in a 4 GB budget. On a pure transformer the same context costs 4x that.
//
//     Harder: a recurrent state CANNOT BE SLICED. A KV cache for N tokens contains the KV
//     for every shorter prefix, so it can be truncated for free. A gated-delta-net state
//     has absorbed all N tokens into one matrix and cannot be rewound. There is no such
//     thing as "the state at token 40" recoverable from the state at token 100.
//
//     That is why entries are matched at the EXACT position they were taken and the rest
//     is replayed, rather than trimmed. It is the checkpoint-and-replay design from
//     "Sparse Prefix Caching for Hybrid and Recurrent LLM Serving" (arXiv 2605.05219).
//
// !! THE FAILURE MODE THIS FILE IS BUILT AROUND !!
//     Restoring the wrong state does not crash. It produces a fluent continuation of a
//     conversation that never happened, and nothing downstream can tell. Our own C engine
//     said the same thing about its `--save-state`:
//
//         "restoring state built by a different architecture would produce fluent, wrong
//          output with nothing to indicate it, which is the one failure mode this engine
//          refuses to have."
//
//     So a hit requires BOTH an exact token-by-token comparison of the whole stored prefix
//     and a matching config fingerprint. A hash is used only to skip obvious misses; it is
//     never sufficient on its own, because a collision would silently swap two
//     conversations.

/// Whatever a given architecture carries between tokens.
///
/// Generic because the two engines carry different things: qwen35moe holds fixed-size
/// recurrent matrices plus a small KV, while DeepSeek-V4 holds an MLA KV cache, the
/// q-LoRA rows and a per-layer hidden history. Only two properties are needed here --
/// it can be copied, and it can say how big it is.
pub trait State: Clone {
    fn bytes(&self) -> usize;
}

/// qwen35moe: 30 fixed recurrent states + 10 growing KVs.
#[derive(Clone)]
pub struct Qwen35State {
    pub lin: Vec<crate::qwen35run::LinState>,
    pub attn: Vec<crate::qwen35run::AttnState>,
    pub pos: usize,
}

impl State for Qwen35State {
    fn bytes(&self) -> usize {
        self.lin.iter().map(|l| (l.conv.len() + l.s.len()) * 4).sum::<usize>()
            + self.attn.iter().map(|a| (a.k.len() + a.v.len()) * 4).sum::<usize>()
    }
}

impl Qwen35State {
    pub fn take(sess: &crate::qwen35run::Session) -> Qwen35State {
        // `extend_from_slice` grows by doubling, so a live KV can hold twice its used
        // bytes. Snapshots outlive their request and are charged to a budget, so the slack
        // is dropped rather than paid for.
        let mut attn = sess.attn.clone();
        for a in attn.iter_mut() {
            a.k.shrink_to_fit();
            a.v.shrink_to_fit();
        }
        Qwen35State { lin: sess.lin.clone(), attn, pos: sess.pos }
    }

    pub fn restore(&self, sess: &mut crate::qwen35run::Session) {
        sess.lin = self.lin.clone();
        sess.attn = self.attn.clone();
        sess.pos = self.pos;
    }
}

/// DeepSeek-V4: the MLA KV cache, the q-LoRA rows, and the per-layer hidden history.
///
/// `hin` is the dominant term -- `hidden` floats per token per layer -- which is why a V4
/// session is roughly 1 MB per token where a qwen35moe one is 40 KB.
#[derive(Clone)]
pub struct V4State {
    pub state: Vec<crate::v4::LayerState>,
    pub hin: Vec<Vec<f32>>,
    pub pos: usize,
}

impl State for V4State {
    fn bytes(&self) -> usize {
        self.state
            .iter()
            .map(|l| (l.kv.len() + l.qr.len() + l.kvc.len() + l.ikvc.len()) * 4)
            .sum::<usize>()
            + self.hin.iter().map(|h| h.len() * 4).sum::<usize>()
    }
}

impl V4State {
    pub fn take(sess: &crate::v4run::Session) -> V4State {
        V4State { state: sess.state.clone(), hin: sess.hin.clone(), pos: sess.pos }
    }

    /// Restore into a session whose `ids` are the FULL new prompt.
    ///
    /// `prompt_len` is set to the whole prompt and `pos` to what was restored, so
    /// `generate_on`'s `prompt_phase = pos < prompt_len` stays true and the appended
    /// tokens are BATCH-prefilled rather than decoded one at a time.
    pub fn restore(&self, sess: &mut crate::v4run::Session, ids: Vec<u32>) {
        sess.prompt_len = ids.len();
        sess.ids = ids;
        sess.state = self.state.clone();
        sess.hin = self.hin.clone();
        sess.pos = self.pos;
    }
}


/// Identifies the geometry a snapshot was built under.
///
/// Transcribed from `K3StateHdr::fp` in the C engine: restoring state across a different
/// model, quantisation or head layout is the failure this exists to make impossible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fingerprint(pub u64);

impl Fingerprint {
    pub fn of(c: &crate::qwen35::Cfg) -> Fingerprint {
        // FNV-1a over the dimensions that change what a state MEANS. Two checkpoints with
        // identical geometry are interchangeable; anything else is not.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in [
            c.n_layers, c.hidden, c.vocab, c.full_attn_interval, c.n_heads, c.n_kv_heads,
            c.head_dim, c.n_rot, c.d_inner, c.n_k_heads, c.n_v_heads, c.d_state,
            c.conv_kernel, c.n_experts, c.topk, c.moe_inter, c.shared_inter,
        ] {
            for b in (v as u64).to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        Fingerprint(h)
    }

    /// The same idea for a safetensors `Spec`. Restoring V4 state into a differently
    /// shaped model is the identical hazard.
    pub fn of_spec(s: &crate::arch::Spec) -> Fingerprint {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in [
            s.n_layers, s.hidden, s.vocab, s.n_heads, s.n_kv_heads, s.head_dim,
            s.n_experts, s.topk, s.moe_inter,
        ] {
            for b in (v as u64).to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        }
        Fingerprint(h)
    }
}

/// One conversation's state at an exact token position.
pub struct Snapshot<S: State> {
    /// The EXACT tokens this state was produced by. Compared element by element on a hit.
    pub ids: Vec<u32>,
    pub fp: Fingerprint,
    pub state: S,
    bytes: usize,
    used_at: u64,
}

impl<S: State> Snapshot<S> {
    pub fn new(state: S, ids: &[u32], fp: Fingerprint) -> Snapshot<S> {
        let bytes = state.bytes() + ids.len() * 4;
        Snapshot { ids: ids.to_vec(), fp, state, bytes, used_at: 0 }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Conversation snapshots, evicted whole under a byte budget.
///
/// Eviction is LRU over ENTIRE conversations rather than over positions within one: half a
/// conversation is worth nothing, because the recurrent half cannot be sliced back to a
/// shorter prefix.
pub struct PrefixCache<S: State> {
    entries: Vec<Snapshot<S>>,
    budget: usize,
    used: usize,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    pub tokens_saved: u64,
    pub evictions: u64,
}

impl<S: State> PrefixCache<S> {
    pub fn new(budget_bytes: usize) -> PrefixCache<S> {
        PrefixCache {
            entries: Vec::new(),
            budget: budget_bytes,
            used: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            tokens_saved: 0,
            evictions: 0,
        }
    }

    pub fn used(&self) -> usize {
        self.used
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The longest stored prefix of `ids`, if any.
    ///
    /// A candidate qualifies only when its ENTIRE stored token sequence is a prefix of the
    /// request, verified element by element. Anything less would restore a state built from
    /// tokens the request does not contain.
    ///
    /// A snapshot covering every token of `ids` is deliberately rejected: there would be
    /// nothing left to feed, and `generate` needs at least one forward pass to produce the
    /// logits it samples from.
    pub fn find(&mut self, ids: &[u32], fp: Fingerprint) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.fp != fp || e.ids.len() >= ids.len() {
                continue;
            }
            if e.ids[..] != ids[..e.ids.len()] {
                continue;
            }
            if best.is_none_or(|b| e.ids.len() > self.entries[b].ids.len()) {
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                self.clock += 1;
                self.entries[i].used_at = self.clock;
                self.hits += 1;
                self.tokens_saved += self.entries[i].ids.len() as u64;
                Some(i)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    pub fn get(&self, i: usize) -> &Snapshot<S> {
        &self.entries[i]
    }

    /// Store a snapshot, evicting least-recently-used conversations to stay in budget.
    ///
    /// An entry larger than the whole budget is dropped rather than stored, because
    /// admitting it would evict everything else and then not fit either.
    pub fn insert(&mut self, mut snap: Snapshot<S>) {
        if snap.bytes > self.budget {
            return;
        }
        // Replace an entry for the same prefix rather than accumulating duplicates.
        if let Some(i) = self.entries.iter().position(|e| e.ids == snap.ids && e.fp == snap.fp) {
            self.used -= self.entries[i].bytes;
            self.entries.swap_remove(i);
        }
        while self.used + snap.bytes > self.budget && !self.entries.is_empty() {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used_at)
                .map(|(i, _)| i)
                .expect("non-empty");
            self.used -= self.entries[victim].bytes;
            self.entries.swap_remove(victim);
            self.evictions += 1;
        }
        self.clock += 1;
        snap.used_at = self.clock;
        self.used += snap.bytes;
        self.entries.push(snap);
    }

    /// Change the budget, evicting least-recently-used conversations to fit.
    ///
    /// Unlike the expert cache this keeps what still fits: a conversation snapshot is
    /// self-contained, so shrinking is a pure eviction and costs nothing else.
    pub fn set_budget(&mut self, bytes: usize) {
        self.budget = bytes;
        while self.used > self.budget && !self.entries.is_empty() {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used_at)
                .map(|(i, _)| i)
                .expect("non-empty");
            self.used -= self.entries[victim].bytes;
            self.entries.swap_remove(victim);
            self.evictions += 1;
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Hit rate over the window since `reset_stats`.
    pub fn window_hit_rate(&self) -> Option<f64> {
        let n = self.hits + self.misses;
        (n > 0).then(|| self.hits as f64 / n as f64)
    }

    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.tokens_saved = 0;
        self.evictions = 0;
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.used = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Meta;
    use crate::gguf::Value::{Str, F, U};

    fn cfg() -> crate::qwen35::Cfg {
        let mut m = Meta::new();
        m.insert("general.architecture".into(), Str("qwen35moe".into()));
        for (k, v) in [
            ("block_count", 4u64), ("embedding_length", 256), ("full_attention_interval", 4),
            ("attention.head_count", 2), ("attention.head_count_kv", 1),
            ("attention.key_length", 64), ("attention.value_length", 64),
            ("rope.dimension_count", 16), ("ssm.inner_size", 64), ("ssm.group_count", 2),
            ("ssm.time_step_rank", 4), ("ssm.state_size", 16), ("ssm.conv_kernel", 4),
            ("expert_count", 4), ("expert_used_count", 2),
            ("expert_feed_forward_length", 32), ("expert_shared_feed_forward_length", 32),
        ] {
            m.insert(format!("qwen35moe.{k}"), U(v));
        }
        m.insert("qwen35moe.rope.freq_base".into(), F(10_000_000.0));
        m.insert("qwen35moe.attention.layer_norm_rms_epsilon".into(), F(1e-6));
        crate::qwen35::Cfg::from_meta(&m).unwrap()
    }

    fn snap(c: &crate::qwen35::Cfg, ids: &[u32]) -> Snapshot<Qwen35State> {
        let mut s = crate::qwen35run::Session {
            lin: (0..c.n_layers).map(|_| crate::qwen35run::LinState::new(c)).collect(),
            attn: (0..c.n_layers).map(|_| crate::qwen35run::AttnState::new()).collect(),
            pos: ids.len(),
        };
        // Make the state distinguishable, so a restore that silently did nothing fails.
        s.lin[0].s[0] = ids.len() as f32;
        Snapshot::new(Qwen35State::take(&s), ids, Fingerprint::of(c))
    }

    #[test]
    fn the_longest_matching_prefix_wins() {
        let c = cfg();
        let mut p: PrefixCache<Qwen35State> = PrefixCache::new(1 << 30);
        p.insert(snap(&c, &[1, 2]));
        p.insert(snap(&c, &[1, 2, 3, 4]));
        p.insert(snap(&c, &[9, 9]));
        let i = p.find(&[1, 2, 3, 4, 5], Fingerprint::of(&c)).expect("should hit");
        assert_eq!(p.get(i).ids, vec![1, 2, 3, 4], "the longer prefix must win");
    }

    /// The whole point. A stored sequence that is NOT a prefix must never match, however
    /// much it overlaps -- restoring it would continue a conversation that never happened.
    #[test]
    fn a_divergent_conversation_never_matches() {
        let c = cfg();
        let mut p: PrefixCache<Qwen35State> = PrefixCache::new(1 << 30);
        p.insert(snap(&c, &[1, 2, 3, 4]));
        // Shares three tokens, differs at the fourth.
        assert!(p.find(&[1, 2, 3, 9, 9], Fingerprint::of(&c)).is_none());
        // A superset in the other direction is not a prefix either.
        assert!(p.find(&[0, 1, 2, 3, 4], Fingerprint::of(&c)).is_none());
    }

    /// State from a different geometry must be refused. This is the failure the C engine
    /// named: it produces fluent, wrong output with nothing to indicate it.
    #[test]
    fn a_different_model_geometry_is_refused() {
        let c = cfg();
        let mut other = cfg();
        other.n_v_heads = 8;
        assert_ne!(Fingerprint::of(&c), Fingerprint::of(&other));
        let mut p: PrefixCache<Qwen35State> = PrefixCache::new(1 << 30);
        p.insert(snap(&c, &[1, 2, 3]));
        assert!(p.find(&[1, 2, 3, 4], Fingerprint::of(&other)).is_none());
        assert!(p.find(&[1, 2, 3, 4], Fingerprint::of(&c)).is_some());
    }

    /// An exact-length match leaves nothing to feed, and `generate` needs one forward pass
    /// to produce logits. Returning it would hand back a model with no logits at all.
    #[test]
    fn a_snapshot_covering_the_whole_request_is_not_used() {
        let c = cfg();
        let mut p: PrefixCache<Qwen35State> = PrefixCache::new(1 << 30);
        p.insert(snap(&c, &[1, 2, 3]));
        assert!(p.find(&[1, 2, 3], Fingerprint::of(&c)).is_none());
        assert!(p.find(&[1, 2, 3, 4], Fingerprint::of(&c)).is_some());
    }

    #[test]
    fn restoring_replaces_the_session_state() {
        let c = cfg();
        let s = snap(&c, &[1, 2, 3, 4, 5]);
        let mut sess = crate::qwen35run::Session {
            lin: (0..c.n_layers).map(|_| crate::qwen35run::LinState::new(&c)).collect(),
            attn: (0..c.n_layers).map(|_| crate::qwen35run::AttnState::new()).collect(),
            pos: 999,
        };
        s.state.restore(&mut sess);
        assert_eq!(sess.pos, 5, "position must come from the snapshot");
        assert_eq!(sess.lin[0].s[0], 5.0, "recurrent state must actually be copied");
    }

    /// The budget is a hard limit; LRU decides who goes. Half a conversation is worthless,
    /// so entries leave whole.
    #[test]
    fn the_budget_is_enforced_by_evicting_whole_conversations() {
        let c = cfg();
        let one = snap(&c, &[1]).bytes();
        let mut p: PrefixCache<Qwen35State> = PrefixCache::new(one * 2 + one / 2);
        p.insert(snap(&c, &[1, 1]));
        p.insert(snap(&c, &[2, 2]));
        assert_eq!(p.len(), 2);
        p.find(&[1, 1, 7], Fingerprint::of(&c)); // touch the first, so the second is LRU
        p.insert(snap(&c, &[3, 3]));
        assert!(p.used() <= p.budget, "the budget must hold");
        assert!(p.evictions >= 1);
        assert!(p.find(&[1, 1, 7], Fingerprint::of(&c)).is_some(), "the touched one stays");
    }

    /// An entry that cannot fit at all must be dropped, not admitted after clearing the
    /// cache it then fails to fit into anyway.
    #[test]
    fn an_oversized_entry_is_dropped_without_evicting_anything() {
        let c = cfg();
        let mut p: PrefixCache<Qwen35State> = PrefixCache::new(1 << 30);
        p.insert(snap(&c, &[1, 2]));
        let n = p.len();
        p.budget = 8;
        p.insert(snap(&c, &[3, 4]));
        assert_eq!(p.len(), n, "an unfittable entry must not evict the cache it cannot join");
    }

    #[test]
    fn reinserting_the_same_prefix_replaces_rather_than_duplicates() {
        let c = cfg();
        let mut p: PrefixCache<Qwen35State> = PrefixCache::new(1 << 30);
        p.insert(snap(&c, &[1, 2]));
        let used = p.used();
        p.insert(snap(&c, &[1, 2]));
        assert_eq!((p.len(), p.used()), (1, used), "no duplicate, no double accounting");
    }
}

// ---------------------------------------------------------------------------
// Splitting RAM between the two caches
// ---------------------------------------------------------------------------

/// Moves memory between the expert cache and the conversation cache.
///
/// THE TWO ARE NOT SIMPLY COMPETITORS
/// ```text
///     Measured on Qwen3.6-35B: batched prefill deduplicates ~45% of expert fetches at
///     either cache size, but the I/O saving only appears when the cache is SMALL --
///     6 GB gave 0% less I/O, 0.25 GB gave 43% and 1.67x. With a generous cache the
///     duplicate would have hit anyway.
///
///     So shrinking the expert cache is partly self-compensating: it is exactly the regime
///     where batching starts paying. That is why this controller is willing to take memory
///     from the experts at all, and why the step is a fraction rather than a doubling.
/// ```
///
/// WHY WINDOWED AND NOT LIFETIME
/// ```text
///     `Cache` counts hits for the life of the process. A lifetime average moves ever more
///     slowly as the process ages, so a controller driven by it would stop responding to
///     workload changes exactly when a long-running server needs it most. Both sides are
///     reset each interval.
/// ```
pub struct Balancer {
    /// Total RAM the two caches may share.
    pub total: usize,
    /// Requests between decisions. Rebalancing per request would thrash the expert cache,
    /// whose resize is destructive.
    pub interval: u32,
    /// Fraction of the total moved in one step.
    pub step: f64,
    /// Never starve either side below this fraction of the total.
    pub floor: f64,
    seen: u32,
    pub moves: u64,
}

/// What a rebalance decided, so it can be reported rather than being a mystery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Move {
    /// Not an interval boundary, or nothing to decide on yet.
    Hold,
    /// Give the conversation cache more, taking it from the experts.
    ToConversations(usize),
    /// The reverse.
    ToExperts(usize),
}

impl Balancer {
    pub fn new(total: usize, interval: u32) -> Balancer {
        Balancer { total, interval, step: 0.15, floor: 0.15, seen: 0, moves: 0 }
    }

    /// Call once per request. `expert` and `conv` are windowed hit rates, `None` when that
    /// cache saw no traffic in the window.
    ///
    /// The rule: grow whichever side is missing more, because a miss is what costs time on
    /// this machine -- an expert miss is a disk read, a conversation miss is a full
    /// re-prefill. A side with no traffic makes no claim, and a side already at the floor
    /// cannot give.
    pub fn observe(
        &mut self,
        expert_bytes: usize,
        conv_bytes: usize,
        expert: Option<f64>,
        conv: Option<f64>,
    ) -> Move {
        self.seen += 1;
        if self.seen < self.interval {
            return Move::Hold;
        }
        self.seen = 0;
        let (Some(e), Some(c)) = (expert, conv) else { return Move::Hold };

        let slab = (self.total as f64 * self.step) as usize;
        let floor = (self.total as f64 * self.floor) as usize;
        // Miss rates, because the cost of a miss is what is being minimised.
        let (em, cm) = (1.0 - e, 1.0 - c);
        // A dead band: a near-tie is not evidence, and acting on noise would thrash a
        // cache whose resize throws away everything it held.
        if (em - cm).abs() < 0.05 {
            return Move::Hold;
        }
        if cm > em && expert_bytes.saturating_sub(slab) >= floor {
            self.moves += 1;
            Move::ToConversations(slab)
        } else if em > cm && conv_bytes.saturating_sub(slab) >= floor {
            self.moves += 1;
            Move::ToExperts(slab)
        } else {
            Move::Hold
        }
    }
}

#[cfg(test)]
mod balance_tests {
    use super::*;

    fn b() -> Balancer {
        Balancer::new(10_000, 1)
    }

    /// The starving cache grows. A conversation miss costs a full re-prefill, which on
    /// this machine is the most expensive thing that can happen.
    #[test]
    fn memory_moves_towards_whichever_cache_is_missing_more() {
        let m = b().observe(8_000, 2_000, Some(0.90), Some(0.10));
        assert!(matches!(m, Move::ToConversations(_)), "{m:?}");
        let m = b().observe(2_000, 8_000, Some(0.10), Some(0.90));
        assert!(matches!(m, Move::ToExperts(_)), "{m:?}");
    }

    /// A near-tie is noise, not evidence. Acting on it would thrash the expert cache,
    /// whose resize discards everything it held.
    #[test]
    fn a_near_tie_holds_rather_than_thrashing() {
        assert_eq!(b().observe(5_000, 5_000, Some(0.50), Some(0.52)), Move::Hold);
    }

    /// Neither side may be starved below the floor, however badly it is missing.
    #[test]
    fn no_cache_is_taken_below_the_floor() {
        // Experts already at the floor cannot give, even though conversations miss more.
        let m = b().observe(1_500, 8_500, Some(0.99), Some(0.01));
        assert_eq!(m, Move::Hold, "the floor must hold even under maximum pressure");
    }

    /// A cache with no traffic in the window makes no claim -- otherwise a cold start,
    /// where the conversation cache has seen nothing, would immediately raid the experts.
    #[test]
    fn a_cache_with_no_traffic_does_not_get_a_vote() {
        assert_eq!(b().observe(8_000, 2_000, Some(0.9), None), Move::Hold);
        assert_eq!(b().observe(8_000, 2_000, None, Some(0.1)), Move::Hold);
    }

    /// Decisions happen on interval boundaries, not every request.
    #[test]
    fn rebalancing_waits_for_the_interval() {
        let mut bal = Balancer::new(10_000, 3);
        assert_eq!(bal.observe(8_000, 2_000, Some(0.9), Some(0.1)), Move::Hold);
        assert_eq!(bal.observe(8_000, 2_000, Some(0.9), Some(0.1)), Move::Hold);
        assert!(matches!(bal.observe(8_000, 2_000, Some(0.9), Some(0.1)), Move::ToConversations(_)));
        // And the counter restarts, so it does not fire every request thereafter.
        assert_eq!(bal.observe(8_000, 2_000, Some(0.9), Some(0.1)), Move::Hold);
    }
}
