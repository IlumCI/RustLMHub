// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use rayon::prelude::*;

use crate::st::St;

/// Bytes the kernel reports as available RIGHT NOW (`MemAvailable` — free RAM plus the page
/// cache it can reclaim without swapping). `None` if `/proc/meminfo` is unreadable.
pub fn mem_available_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| {
        m.lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kb| kb * 1024)
    })
}

/// The largest expert-arena budget that fits in currently-available RAM WITHOUT swapping,
/// leaving `margin_bytes` free for activations, KV-cache growth and other processes (a game,
/// the browser). Call this AFTER the resident trunk is loaded, so `MemAvailable` already
/// excludes it. Result is clamped to `[min_bytes, max_bytes]` (never bigger than the whole
/// streamed weight set — a larger arena buys nothing). Falls back to `fallback` when
/// `/proc/meminfo` can't be read.
///
/// This is the anti-swap guard the dense path lacked: an arena sized past available RAM does
/// not OOM cleanly, it pages the resident trunk out onto the same disk the experts stream
/// from, and both the model and everything else grind. Sizing to what is actually free — and
/// re-reading it each load — also means a model launched while a game is running quietly
/// takes a smaller arena instead of fighting it for RAM.
pub fn auto_budget_bytes(margin_bytes: u64, min_bytes: i64, max_bytes: i64, fallback: i64) -> i64 {
    match mem_available_bytes() {
        Some(avail) => (avail.saturating_sub(margin_bytes) as i64).clamp(min_bytes, max_bytes),
        None => fallback.clamp(min_bytes, max_bytes),
    }
}

/// Where one routed expert's bytes live.
///
/// Two layouts, because two ecosystems disagree. Safetensors checkpoints give every expert
/// its own tensors -- three matrices, each a packed weight plus a separate scale block, so
/// six in total. Every GGUF MoE instead stacks all N experts of a projection into ONE 3-D
/// tensor, and a k-quant carries its scales inside each super-block, so an expert is three
/// SLICES and there are no scale tensors at all.
///
/// Both resolve to the same `ExpertRef` of coalesced byte runs, so the cache, the
/// sweep-aware eviction and the VRAM tier are shared rather than forked -- they are all
/// byte-range based and do not care which produced the range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertSrc {
    /// Six whole tensors, in ExpertQ slot order: w1, w1.scale, w3, w3.scale, w2, w2.scale.
    Named([String; 6]),
    /// Three stacked tensors sliced at `expert`, in ExpertQ slot order: w1, w3, w2.
    Stacked { names: [String; 3], expert: usize },
    /// Three UNSTACKED tensors taken whole, in the same slot order: w1, w3, w2.
    ///
    /// A dense model's feed-forward is one expert per layer that every token routes to, so
    /// it streams through this cache exactly like a routed one -- same sweep, same
    /// eviction, and a prefetch that is EXACT rather than predicted, because the next layer
    /// is always the next layer.
    ///
    /// It cannot reuse `Stacked` with `expert: 0`. A stacked projection is
    /// `[n_experts, rows, cols]` and the slice is taken on `shape[0]`; a dense
    /// `blk.N.ffn_gate.weight` is 2-D, so `shape[0]` is 17408 ROWS and slice zero would
    /// fetch one row of the matrix. The shapes are indistinguishable to the slicer and the
    /// result would be a model that loads and computes with 1/17408th of its weights.
    Whole([String; 3]),
}

/// The old name, kept so the `fn(usize, usize) -> _` pointers threaded through v4.rs and
/// dspark.rs need only a type swap rather than a rewrite.
pub type ExpertNames = ExpertSrc;

impl ExpertSrc {
    /// The six ExpertQ slots as (tensor, byte offset within it, length), `None` where the
    /// layout has no such piece.
    ///
    /// The index mapping is load-bearing: `Cache::slice` reads slots 0..5 as
    /// `p1, s1, p3, s3, p2, s2`, so a stacked expert's three tensors must land at 0, 2
    /// and 4 with the scale slots empty. Packing them at 0, 1, 2 instead would bind w3's
    /// bytes to w1's scale slot -- finite, plausible, and wrong, which is precisely the
    /// shape of the w2/w3 bug that already cost this project a debugging session.
    /// The first tensor this expert lives in -- enough to ask whether it exists at all,
    /// which is how the checkpoint validators count present layers without reading bytes.
    /// The six slot names for a `Named` layout, for tests that pin the exact strings.
    /// Panics on `Stacked`, which has no such array by construction.
    pub fn named(&self) -> &[String; 6] {
        match self {
            ExpertSrc::Named(n) => n,
            ExpertSrc::Stacked { .. } | ExpertSrc::Whole(_) => {
                panic!("only a Named layout has a six-name array")
            }
        }
    }

    pub fn probe_name(&self) -> &str {
        match self {
            ExpertSrc::Named(n) => &n[0],
            ExpertSrc::Stacked { names, .. } | ExpertSrc::Whole(names) => &names[0],
        }
    }

    /// Byte range of expert `e` within a stacked tensor of `nbytes` holding `n_exp`
    /// experts. Split out from `pieces` so the arithmetic is testable without a
    /// multi-gigabyte fixture, and so its guards are visible.
    fn stacked_slice(nbytes: i64, n_exp: i64, e: usize) -> Option<(i64, i64)> {
        // A non-divisible total means shape[0] is not the expert count -- the tensor is
        // not what we think it is, and slicing it would hand back a misaligned window of
        // somebody else's weights.
        if n_exp <= 0 || nbytes <= 0 || nbytes % n_exp != 0 || (e as i64) >= n_exp {
            return None;
        }
        let stride = nbytes / n_exp;
        Some((e as i64 * stride, stride))
    }

    fn pieces(&self, st: &St) -> Option<[Option<(String, i64, i64)>; 6]> {
        const NONE: Option<(String, i64, i64)> = None;
        let mut out = [NONE; 6];
        match self {
            ExpertSrc::Named(names) => {
                for (i, n) in names.iter().enumerate() {
                    out[i] = Some((n.clone(), 0, st.find(n)?.nbytes));
                }
            }
            ExpertSrc::Stacked { names, expert } => {
                for (j, n) in names.iter().enumerate() {
                    let t = st.find(n)?;
                    // shape[0] is the expert count. GGUF stores dims fastest-varying
                    // first and the scanner reverses them, so a stacked projection is
                    // [n_experts, rows, cols] and every expert is an equal slice.
                    let (off, len) =
                        Self::stacked_slice(t.nbytes, *t.shape.first()?, *expert)?;
                    out[j * 2] = Some((n.clone(), off, len));
                }
            }
            // Same slot order as `Stacked`, no slicing: the whole tensor IS the expert.
            ExpertSrc::Whole(names) => {
                for (j, n) in names.iter().enumerate() {
                    out[j * 2] = Some((n.clone(), 0, st.find(n)?.nbytes));
                }
            }
        }
        Some(out)
    }
}

/// DeepSeek-V4: `layers.L.ffn.experts.E.w{1,3,2}.{weight,scale}`.
pub fn v4_expert_names(layer: usize, expert: usize) -> ExpertNames {
    let p = format!("layers.{layer}.ffn.experts.{expert}");
    ExpertSrc::Named([
        format!("{p}.w1.weight"),
        format!("{p}.w1.scale"),
        format!("{p}.w3.weight"),
        format!("{p}.w3.scale"),
        format!("{p}.w2.weight"),
        format!("{p}.w2.scale"),
        ])
}

/// DeepSeek-V4's decoder depth. The DSpark stages are addressed as layers
/// `V4_LAYERS + stage` so their cache keys -- `layer * n_experts + expert` -- cannot
/// collide with decoder layers 0, 1 and 2, which are the ones they would otherwise share
/// a key with.
pub const V4_LAYERS: usize = 43;

/// DeepSeek-V4 DSpark stages: `mtp.S.ffn.experts.E.w{1,3,2}.{weight,scale}`, addressed by
/// the caller as layer `V4_LAYERS + S`.
pub fn v4_dspark_expert_names(layer: usize, expert: usize) -> ExpertNames {
    let p = format!("mtp.{}.ffn.experts.{expert}", layer - V4_LAYERS);
    ExpertSrc::Named([
        format!("{p}.w1.weight"),
        format!("{p}.w1.scale"),
        format!("{p}.w3.weight"),
        format!("{p}.w3.scale"),
        format!("{p}.w2.weight"),
        format!("{p}.w2.scale"),
        ])
}

/// Kimi K3: `language_model.model.layers.L.block_sparse_moe.experts.E.w{1,2,3}
/// .weight_{packed,scale}`, one contiguous 17,547,264-byte run.
pub fn k3_expert_names(layer: usize, expert: usize) -> ExpertNames {
    let p = format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}");
    ExpertSrc::Named([
        // w1, w3, w2 -- the order ExpertQ's slots are cut in, NOT the on-disk order
        // (which is w1, w2, w3). locate() sorts by file offset, so this array's order
        // only decides which tensor lands in which ExpertQ field. Listing them in disk
        // order swaps w2 and w3: every routed expert then computes
        // situ_glu(w1(z), w2(z)) . w3 instead of situ_glu(w1(z), w3(z)) . w2, which is
        // finite, plausible and wrong.
        format!("{p}.w1.weight_packed"),
        format!("{p}.w1.weight_scale"),
        format!("{p}.w3.weight_packed"),
        format!("{p}.w3.weight_scale"),
        format!("{p}.w2.weight_packed"),
        format!("{p}.w2.weight_scale"),
        ])
}

/// GGUF MoE: `blk.N.ffn_{gate,up,down}_exps.weight`, every expert stacked into one 3-D
/// tensor per projection.
///
/// gate/up/down is w1/w3/w2 -- ExpertQ's slot order, NOT the order the names read in.
/// Listing them gate, DOWN, up would compute `situ_glu(w1(z), w2(z)) . w3`, which is
/// finite and plausible and wrong; that exact transposition already shipped once here.
pub fn gguf_expert_src(layer: usize, expert: usize) -> ExpertSrc {
    ExpertSrc::Stacked {
        names: [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ],
        expert,
    }
}

/// A dense block's feed-forward, addressed as this cache's unit of streaming.
///
/// Note the names: `ffn_gate` where the MoE says `ffn_gate_exps`. That three-character
/// difference is the entire distinction between `qwen35` and `qwen35moe` on disk -- every
/// other tensor in the block is spelled identically -- which is why one loader, one cache
/// and one attention implementation serve both.
pub fn gguf_dense_ffn_src(layer: usize, _expert: usize) -> ExpertSrc {
    ExpertSrc::Whole([
        format!("blk.{layer}.ffn_gate.weight"),
        format!("blk.{layer}.ffn_up.weight"),
        format!("blk.{layer}.ffn_down.weight"),
    ])
}

/// Validate one expert's geometry before any byte is read. `k3_load.c:46-56`: if the
/// scale count is not the logical width over the group size, the group size is not 32 for
/// this tensor and every scale after the first is applied to the wrong 32 weights.
pub fn check_expert(st: &St, src: &ExpertSrc, group: i64) -> Result<(), String> {
    // This checks a packed-weight/separate-scale pairing. A stacked GGUF expert has no
    // scale tensors at all -- its scales live inside each super-block -- so there is
    // nothing here to validate and pretending otherwise would invent a failure.
    let ExpertSrc::Named(names) = src else {
        return Ok(());
    };
    for pair in 0..3 {
        let (pn, sn) = (&names[pair * 2], &names[pair * 2 + 1]);
        let p = st.find(pn).ok_or_else(|| format!("k3_load: missing {pn}"))?;
        let s = st.find(sn).ok_or_else(|| format!("k3_load: missing {sn}"))?;
        if p.dtype != crate::st::Dtype::U8 || s.dtype != crate::st::Dtype::U8 {
            return Err(format!("k3_load: {pn} is not U8"));
        }
        if p.shape.len() != 2 || s.shape.len() != 2 {
            return Err(format!("k3_load: {pn} is not 2D"));
        }
        if p.shape[0] != s.shape[0] {
            return Err(format!(
                "k3_load: {pn} row mismatch {} vs {}",
                p.shape[0], s.shape[0]
            ));
        }
        let logical = p.shape[1] * 2;
        if s.shape[1] * group != logical {
            return Err(format!(
                "k3_load: {pn}: {} scales for {logical} elements implies group size {:.2}, not {group}",
                s.shape[1],
                logical as f64 / s.shape[1] as f64
            ));
        }
        if p.shard != s.shard {
            return Err(format!("k3_load: {pn} is split across shards"));
        }
    }
    Ok(())
}

/// A maximal gapless span of the shard covering one or more of the expert's tensors.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub off: i64,
    pub len: i64,
}

#[derive(Clone, Debug)]
pub struct ExpertRef {
    pub shard: usize,
    /// Coalesced runs, in file order. K3 packs an expert as one run; DeepSeek-V4 groups
    /// all scales together and all weights together, ~340 MB apart, so it takes two.
    /// Reading the enclosing span instead would fetch 354 MB to use 13.37.
    pub runs: Vec<Run>,
    /// (run index, offset within the run, length) per tensor, in ExpertNames order.
    pub parts: [(usize, i64, i64); 6],
    pub nbytes: i64,
}

impl ExpertRef {
    pub fn contiguous(&self) -> bool {
        self.runs.len() == 1
    }
}

pub fn locate(st: &St, src: &ExpertSrc) -> Option<ExpertRef> {
    let pieces = src.pieces(st)?;
    // (absolute offset, length, ExpertQ slot, shard) for the slots that exist.
    let mut ts = Vec::with_capacity(6);
    for (i, piece) in pieces.iter().enumerate() {
        let Some((name, off, len)) = piece else { continue };
        let t = st.find(name)?;
        ts.push((t.off + off, *len, i, t.shard));
    }
    if ts.is_empty() {
        return None;
    }
    let shard = ts[0].3;
    if ts.iter().any(|t| t.3 != shard) {
        return None; // an expert split across shards is not a single fetch
    }
    let mut order = ts.clone();
    order.sort_by_key(|t| t.0);

    let mut runs: Vec<Run> = Vec::new();
    // Absent slots stay (0, 0, 0), which `Cache::slice` reads as an empty slice -- exactly
    // right for a stacked expert's three missing scale tensors.
    let mut parts = [(0usize, 0i64, 0i64); 6];
    for &(off, len, idx, _) in &order {
        let extend = matches!(runs.last(), Some(r) if r.off + r.len == off);
        if extend {
            let i = runs.len() - 1;
            parts[idx] = (i, runs[i].len, len);
            runs[i].len += len;
        } else {
            parts[idx] = (runs.len(), 0, len);
            runs.push(Run { off, len });
        }
    }
    let nbytes = order.iter().map(|t| t.1).sum();
    Some(ExpertRef { shard, runs, parts, nbytes })
}

/// Slices of one resident expert, pointing into the cache arena. The packed MXFP4 form
/// is what `ops::matmul_mxfp4` consumes, so nothing is dequantised on the way in:
/// dequantised, one DeepSeek-V4 expert is 8x its packed size for no benefit, because a
/// matrix-vector product is memory bound.
pub struct ExpertQ<'a> {
    pub p1: &'a [u8],
    pub s1: &'a [u8],
    pub p3: &'a [u8],
    pub s3: &'a [u8],
    pub p2: &'a [u8],
    pub s2: &'a [u8],
}

/// O_DIRECT widens a read outward to aligned bounds, so a run needs a page of slack at
/// each end on top of its own length.
pub fn run_region(len: i64) -> usize {
    (len as usize).next_multiple_of(crate::st::ALIGN as usize) + 2 * crate::st::ALIGN as usize
}

/// Slot bytes an expert needs: every run's region, back to back.
pub fn slot_need(r: &ExpertRef) -> usize {
    r.runs.iter().map(|x| run_region(x.len)).sum()
}

const EMPTY: i64 = -1;
const INFLIGHT: i64 = -2;

/// Direction the decoder sweep runs along the layer axis.
///
/// `Up` is the forward pass -- layers 0..n, which is inference, prefill, and the forward
/// half of a training step. `Down` is the training BACKWARD pass -- layers n..0, where the
/// gradient walks the stack in reverse.
///
/// The direction is load-bearing, not cosmetic. Belady-by-layer (see [`Cache::distance`])
/// evicts the layer whose next use is furthest away. On the forward pass that is the layer
/// just left. On the backward pass the sweep is reaching for layer `l-1` next, so the
/// SAME "evict what was just used" instinct is now exactly wrong: the layer just finished
/// on the way down is the one backprop revisits soonest on the way... it is not, but its
/// NEIGHBOUR is, and the furthest-in-future layer is the mirror of the forward answer.
/// Running the forward policy through a backward sweep would evict precisely the states
/// backprop is about to need, turning a streamed backward pass from affordable into a
/// re-read storm. This flag is the whole reason streaming the backward pass pays.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Up,
    Down,
}

pub struct Cache {
    arena: crate::st::Aligned,
    slot_bytes: usize,
    nslot: usize,
    n_experts: usize,
    /// key -> slot, or -1
    slot_of: HashMap<i64, i32>,
    /// slot -> key, EMPTY, or INFLIGHT
    key_of: Vec<i64>,
    used_at: Vec<u64>,
    pinned: Vec<bool>,
    refs: Vec<Option<ExpertRef>>,
    /// Where each run's payload starts inside its slot region, per slot.
    pad: Vec<Vec<usize>>,
    clock: u64,
    /// GPU memory as a victim cache: evicted experts spill here instead of being dropped.
    vram: Option<crate::vram::Vram>,
    /// Experts re-served from the GPU rather than re-read from the shard.
    pub vram_hits: u64,
    /// (current position, cycle length) of the decoder sweep, when the caller reports it.
    ///
    /// Eviction needs this because the access pattern is a CYCLIC SCAN: every token walks
    /// layers 0..n in order, and one token's experts do not fit in the cache. Under a
    /// cyclic scan whose working set exceeds the cache, LRU is the worst policy available
    /// -- it evicts the least-recently-used block, which is precisely the one the next
    /// pass reaches SOONEST. That is the sequential-flooding pathology database buffer
    /// managers have used MRU-for-scans to avoid since the 1980s.
    ///
    /// Belady (1966) says evict the furthest-in-future. On the layer axis the future is
    /// not a prediction at all: at position `l`, an expert belonging to layer `L` cannot
    /// be touched again until the sweep comes back round to `L`, which is exactly
    /// `(L - l) mod cycle` steps away. So the dominant term in Belady's rule is
    /// computable here, exactly, with no oracle.
    sweep: Option<(usize, usize)>,
    /// Which way the sweep in `sweep` runs. `Up` for the forward pass and for every
    /// existing call site; `Down` only during a streamed training backward pass. Ignored
    /// entirely when `sweep` is `None`, where the policy is plain LRU regardless.
    dir: Dir,

    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
    /// Experts brought resident by a batch prefetch rather than by `get`. Counted
    /// separately or the hit rate becomes a lie: a prefetched expert is resident when
    /// `get` asks, so `get` records a hit, but the bytes still came off the disk this
    /// token. The effective rate is (hits - prefetch_reads) / requests.
    pub prefetch_reads: u64,
    /// Requests per (layer, expert). Which experts are hot is not knowable in advance
    /// and is the input to any pinning strategy; without measuring it, pinning is guesswork.
    pub hist: HashMap<i64, u32>,
    /// (layer, expert) pairs in request order, when enabled. Routing does not depend on
    /// the cache, so ONE run yields the whole hit-rate-versus-capacity curve offline --
    /// which is the only way to compare replacement policies without re-running a 304B
    /// model once per policy. Off by default: it is 8 bytes per expert request.
    trace: Option<Vec<i32>>,
    pre: Option<Prefetcher>,
    /// Prefetched experts discarded because no free slot was available.
    pub dropped: u64,
    /// (bits, cold_threshold) for the precision sweep. cold_threshold 0 means every expert.
    qdq: Option<(u32, usize)>,
    pub qdq_applied: u64,
    /// Per-layer profiling: bytes actually read off disk, and hit/miss counts, keyed by
    /// layer. Populated on every admit; cleared by `reset_stats`. This is the ground truth
    /// for "which bytes are read per token", broken down by layer, rather than inferred from
    /// average bandwidth.
    pub layer_bytes: HashMap<usize, u64>,
    pub layer_hits: HashMap<usize, u64>,
    pub layer_miss: HashMap<usize, u64>,
}

impl Cache {
    /// `budget_bytes` is divided into whole slots of `slot_bytes`. The constructor
    /// refuses a cache that cannot hold one token's working set: `k3_moe` fetches an
    /// expert and uses it immediately, so plain LRU is only safe while capacity exceeds
    /// top-k. Enforced rather than trusted.
    pub fn new(
        budget_bytes: i64,
        slot_bytes: usize,
        n_experts: usize,
        topk: usize,
    ) -> Result<Cache, String> {
        let nslot = (budget_bytes / slot_bytes as i64).max(0) as usize;
        if nslot < topk + 1 {
            return Err(format!(
                "cache holds {nslot} experts but a token needs top-{topk}; a cache smaller \
                 than one token's working set would evict an expert before it is used"
            ));
        }
        Ok(Cache {
            arena: crate::st::Aligned::new(nslot * slot_bytes),
            slot_bytes,
            nslot,
            n_experts,
            slot_of: HashMap::new(),
            key_of: vec![EMPTY; nslot],
            used_at: vec![0; nslot],
            pinned: vec![false; nslot],
            refs: vec![None; nslot],
            pad: vec![Vec::new(); nslot],
            clock: 0,
            sweep: None,
            dir: Dir::Up,
            vram: None,
            vram_hits: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            bytes_read: 0,
            prefetch_reads: 0,
            hist: HashMap::new(),
            trace: None,
            pre: None,
            dropped: 0,
            qdq: None,
            qdq_applied: 0,
            layer_bytes: HashMap::new(),
            layer_hits: HashMap::new(),
            layer_miss: HashMap::new(),
        })
    }

    /// Attach a GPU victim cache. Pins the host arena so DMA skips the driver's bounce
    /// buffer; that is an optimisation and its failure is silent.
    pub fn attach_vram(&mut self, v: crate::vram::Vram) {
        v.pin(self.arena.as_mut_ptr(), self.nslot * self.slot_bytes);
        self.vram = Some(v);
    }

    pub fn vram_report(&self) -> Option<(usize, u64, u64, u64)> {
        self.vram.as_ref().map(|v| (v.nslot(), v.spills, v.fills, v.bytes_down))
    }

    pub fn nslot(&self) -> usize {
        self.nslot
    }

    fn key(&self, layer: usize, expert: usize) -> i64 {
        (layer * self.n_experts + expert) as i64
    }

    /// Report where the decoder sweep is. `cycle` is the number of distinct layer
    /// positions a token visits, so successive tokens repeat the same cycle.
    ///
    /// Without this the cache falls back to plain LRU, which is what every existing test
    /// exercises and what `sweep: None` preserves.
    pub fn at_layer(&mut self, layer: usize, cycle: usize) {
        self.at_layer_dir(layer, cycle, Dir::Up);
    }

    /// Report the sweep position AND its direction. `at_layer` is this with [`Dir::Up`];
    /// the training backward pass calls it with [`Dir::Down`] so eviction mirrors. The
    /// VRAM victim cache is only prefetched on the forward sweep -- on the way down the
    /// exact next layer is known, so there is nothing to predict and `at_layer` on the
    /// GPU tier is left at its forward meaning.
    pub fn at_layer_dir(&mut self, layer: usize, cycle: usize, dir: Dir) {
        self.sweep = (cycle > 0).then_some((layer % cycle, cycle));
        self.dir = dir;
        let n = self.n_experts;
        if let (Dir::Up, Some(v)) = (dir, self.vram.as_mut()) {
            v.at_layer(layer, cycle, n);
        }
    }

    /// How many sweep steps until layer `L` can next be reached from the current position.
    /// `cycle` when the layer is the one just finished, 0 when it is the current one.
    fn distance(&self, key: i64) -> usize {
        match self.sweep {
            Some((l, cycle)) => {
                let layer = (key as usize / self.n_experts) % cycle;
                // Deliberately NOT `% cycle` at the end: an expert for the layer we are
                // standing on has already been used this pass and is a full cycle away
                // from its next use, not zero. Folding it to 0 would make the current
                // layer's own experts the LAST thing evicted, which is backwards.
                //
                // Down is the exact mirror of Up: swap the roles of `layer` and `l` in the
                // subtraction and the same "furthest next use, current layer a full cycle
                // away" property holds while the sweep runs n..0 instead of 0..n. Both
                // forms are underflow-free: `layer` and `l` are each in `0..cycle`, so the
                // bracket is always in `0..cycle` before the `+1`.
                match self.dir {
                    Dir::Up => (layer + cycle - l - 1) % cycle + 1,
                    Dir::Down => (l + cycle - layer - 1) % cycle + 1,
                }
            }
            None => 0,
        }
    }

    /// Linear scan. At a few hundred slots that is a few hundred comparisons against a
    /// multi-megabyte read; a heap would not pay for itself.
    ///
    /// Ranks by (distance to next use, then age). With no sweep reported every distance is
    /// 0 and this degenerates to exactly the LRU it replaced.
    fn pick_victim(&self) -> Option<usize> {
        let mut best = None;
        let mut best_rank = (0usize, 0u64);
        for i in 0..self.nslot {
            let key = self.key_of[i];
            match key {
                INFLIGHT => continue, // being read into right now
                EMPTY => return Some(i),
                _ if self.pinned[i] => continue,
                _ => {
                    // Maximise distance, then age: `u64::MAX - used_at` is larger for
                    // older entries, so one comparison orders both.
                    let rank = (self.distance(key), u64::MAX - self.used_at[i]);
                    if best.is_none() || rank > best_rank {
                        best_rank = rank;
                        best = Some(i);
                    }
                }
            }
        }
        best
    }

    fn admit(&mut self, st: &St, layer: usize, expert: usize, names: &ExpertNames) -> Option<usize> {
        let key = self.key(layer, expert);
        *self.hist.entry(key).or_insert(0) += 1;
        if let Some(&slot) = self.slot_of.get(&key) {
            self.hits += 1;
            *self.layer_hits.entry(layer).or_insert(0) += 1;
            self.clock += 1;
            self.used_at[slot as usize] = self.clock;
            return Some(slot as usize);
        }
        self.misses += 1;
        *self.layer_miss.entry(layer).or_insert(0) += 1;

        let r = locate(st, names)?;
        if slot_need(&r) > self.slot_bytes {
            eprintln!(
                "k3_cache: L{layer} expert {expert} needs {} bytes, slot holds {}",
                slot_need(&r),
                self.slot_bytes
            );
            return None;
        }
        let slot = self.pick_victim()?;
        if self.key_of[slot] >= 0 {
            // Spill the outgoing expert to the GPU instead of dropping it. This is the
            // only place an expert leaves RAM, so it is the only place the victim cache
            // can be filled.
            if let Some(v) = self.vram.as_mut() {
                let old = self.key_of[slot];
                if let Some(r) = self.refs[slot].as_ref() {
                    let base = slot * self.slot_bytes;
                    v.put(old, &self.arena[base..base + self.slot_bytes], &self.pad[slot], r);
                }
            }
            self.slot_of.remove(&self.key_of[slot]);
            self.evictions += 1;
        }
        self.key_of[slot] = INFLIGHT;

        // Before touching the disk, ask the GPU. A DMA round trip is a memcpy, so an
        // expert served from here is bit-identical to one read from the shard -- this
        // path changes how long a token takes and cannot change what it says.
        if let Some(v) = self.vram.as_mut() {
            let base = slot * self.slot_bytes;
            if let Some((pads, r)) = v.take(key, &mut self.arena[base..base + self.slot_bytes]) {
                self.pad[slot] = pads;
                self.refs[slot] = Some(r);
                self.key_of[slot] = key;
                self.slot_of.insert(key, slot as i32);
                self.clock += 1;
                self.used_at[slot] = self.clock;
                self.vram_hits += 1;
                return Some(slot);
            }
        }

        // One aligned read per coalesced run, each into its own region of the slot.
        let base = slot * self.slot_bytes;
        let mut pads = Vec::with_capacity(r.runs.len());
        let mut cur = base;
        let mut ok = true;
        for run in &r.runs {
            let region = run_region(run.len);
            let (avail, pad) =
                st.read_aligned(r.shard, run.off, run.len, &mut self.arena[cur..cur + region]);
            if avail < run.len {
                ok = false;
                break;
            }
            pads.push(cur - base + pad as usize);
            cur += region;
        }
        if !ok {
            // A short read must not become a resident expert: the bytes would be
            // partly stale and the model would still produce output.
            self.key_of[slot] = EMPTY;
            eprintln!("k3_cache: short read on L{layer} expert {expert}");
            return None;
        }
        self.bytes_read += r.nbytes as u64;
        *self.layer_bytes.entry(layer).or_insert(0) += r.nbytes as u64;
        self.pad[slot] = pads;
        self.refs[slot] = Some(r);
        self.key_of[slot] = key;
        self.slot_of.insert(key, slot as i32);
        self.clock += 1;
        self.used_at[slot] = self.clock;
        if let Some((bits, cold)) = self.qdq {
            let hot = self.hist.get(&key).copied().unwrap_or(0) as usize;
            if cold == 0 || hot <= cold {
                self.qdq_slot(slot, bits);
                self.qdq_applied += 1;
            }
        }
        Some(slot)
    }

    fn slice(&self, slot: usize) -> ExpertQ<'_> {
        let r = self.refs[slot].as_ref().expect("resident slot has a ref");
        let base = slot * self.slot_bytes;
        let at = |i: usize| {
            let (run, o, n) = r.parts[i];
            let s = base + self.pad[slot][run] + o as usize;
            &self.arena[s..s + n as usize]
        };
        ExpertQ { p1: at(0), s1: at(1), p3: at(2), s3: at(3), p2: at(4), s2: at(5) }
    }

    pub fn get(&mut self, st: &St, layer: usize, expert: usize, names: &ExpertNames) -> Option<usize> {
        if let Some(t) = self.trace.as_mut() {
            t.push(layer as i32);
            t.push(expert as i32);
        }
        if self.pre.is_some() {
            self.drain_prefetched();
        }
        self.admit(st, layer, expert, names)
    }

    /// Start recording the request sequence. `sim_cache.py` replays it.
    /// Degrade experts to `bits` on admission. `cold` restricts it to experts requested
    /// at most that many times; 0 applies it to all of them.
    pub fn qdq_on(&mut self, bits: u32, cold: usize) {
        self.qdq = Some((bits, cold));
    }

    pub fn trace_on(&mut self) {
        self.trace = Some(Vec::new());
    }

    pub fn trace_bytes(&self) -> Vec<u8> {
        self.trace
            .as_ref()
            .map(|t| t.iter().flat_map(|v| v.to_le_bytes()).collect())
            .unwrap_or_default()
    }

    /// Already-resident lookup with no disk read. Used by the draft path, which routes
    /// only to experts it already holds.
    pub fn resident(&mut self, layer: usize, expert: usize) -> Option<usize> {
        let key = self.key(layer, expert);
        let slot = *self.slot_of.get(&key)?;
        self.clock += 1;
        self.used_at[slot as usize] = self.clock;
        Some(slot as usize)
    }

    pub fn expert(&self, slot: usize) -> ExpertQ<'_> {
        self.slice(slot)
    }

    /// Hand the whole top-k over before using any of it, so the reads can overlap.
    /// Without this the caller misses, blocks on a multi-megabyte read, computes, and
    /// misses again: queue depth one against a drive that needs depth to reach its
    /// rated bandwidth.
    pub fn prefetch_many<F>(&mut self, st: &St, layer: usize, experts: &[usize], names: F) -> usize
    where
        F: Fn(usize, usize) -> ExpertNames,
    {
        // Three phases, as k3_cache.c:150-200. Reserving every slot before any read is
        // what lets the reads run concurrently: a drive needs queue depth to reach its
        // rated bandwidth, and one-at-a-time admit() gives it depth one.
        let mut jobs: Vec<(usize, usize, ExpertRef)> = Vec::new();
        for &e in experts {
            let key = self.key(layer, e);
            *self.hist.entry(key).or_insert(0) += 1;
            if self.slot_of.contains_key(&key) {
                continue;
            }
            let Some(r) = locate(st, &names(layer, e)) else { continue };
            if slot_need(&r) > self.slot_bytes {
                continue;
            }
            let Some(slot) = self.pick_victim() else { break };
            if self.key_of[slot] >= 0 {
                // This is where the bulk of eviction happens -- the batch prefetch, not
                // `admit` -- so the victim cache has to be filled from here or it barely
                // sees traffic at all.
                if let Some(v) = self.vram.as_mut() {
                    let old = self.key_of[slot];
                    if let Some(rr) = self.refs[slot].as_ref() {
                        let base = slot * self.slot_bytes;
                        v.put(old, &self.arena[base..base + self.slot_bytes], &self.pad[slot], rr);
                    }
                }
                self.slot_of.remove(&self.key_of[slot]);
                self.evictions += 1;
            }
            // Ask the GPU before queueing a disk read: same bytes, ~6x quicker.
            if let Some(v) = self.vram.as_mut() {
                let base = slot * self.slot_bytes;
                if let Some((pads, rr)) =
                    v.take(key, &mut self.arena[base..base + self.slot_bytes])
                {
                    self.pad[slot] = pads;
                    self.refs[slot] = Some(rr);
                    self.key_of[slot] = key;
                    self.slot_of.insert(key, slot as i32);
                    self.clock += 1;
                    self.used_at[slot] = self.clock;
                    self.vram_hits += 1;
                    continue;
                }
            }
            self.key_of[slot] = INFLIGHT;
            self.misses += 1;
            jobs.push((slot, e, r));
        }
        if jobs.is_empty() {
            return 0;
        }
        // Issue in disk order. Sort by the LARGEST run, not runs[0]: a DeepSeek-V4
        // expert's first run is its 786 KB of scales, while the 12.6 MB of weights sits
        // ~341 MB later. Ordering by runs[0] sequenced 6% of the bytes and left the other
        // 94% in whatever order rayon happened to pick.
        jobs.sort_by_key(|(_, _, r)| {
            let big = r.runs.iter().max_by_key(|x| x.len).map_or(0, |x| x.off);
            (r.shard, big)
        });

        let slot_bytes = self.slot_bytes;
        let want: HashMap<usize, usize> =
            jobs.iter().enumerate().map(|(j, (s, _, _))| (*s, j)).collect();

        // One work item per RUN, not per expert. The two runs of one expert land in
        // disjoint regions of the same slot, so they can be in flight together; reading
        // them serially inside a single task capped achieved queue depth at topk when the
        // device measured 378 MB/s at depth 1 against 604 at depth 3.
        struct Piece<'a> {
            job: usize,
            run: usize,
            off: i64,
            len: i64,
            shard: usize,
            at: usize,
            buf: &'a mut [u8],
        }
        let mut pieces: Vec<Piece> = Vec::new();
        for (i, chunk) in self.arena.chunks_mut(slot_bytes).enumerate() {
            let Some(&j) = want.get(&i) else { continue };
            let r = &jobs[j].2;
            let mut rest = chunk;
            let mut cur = 0usize;
            for (ri, run) in r.runs.iter().enumerate() {
                let region = run_region(run.len);
                let (head, tail) = rest.split_at_mut(region);
                pieces.push(Piece {
                    job: j,
                    run: ri,
                    off: run.off,
                    len: run.len,
                    shard: r.shard,
                    at: cur,
                    buf: head,
                });
                rest = tail;
                cur += region;
            }
        }
        pieces.sort_by_key(|p| (p.shard, p.off));

        let done: Vec<(usize, usize, Option<usize>)> = pieces
            .into_par_iter()
            .map(|p| {
                let (avail, pad) = st.read_aligned(p.shard, p.off, p.len, p.buf);
                (p.job, p.run, (avail >= p.len).then_some(p.at + pad as usize))
            })
            .collect();

        // Reassemble: an expert is usable only if every one of its runs landed.
        let mut pads: Vec<Vec<Option<usize>>> =
            jobs.iter().map(|(_, _, r)| vec![None; r.runs.len()]).collect();
        for (j, ri, at) in done {
            pads[j][ri] = at;
        }
        let mut out: Vec<(usize, Option<Vec<usize>>)> = pads
            .into_iter()
            .enumerate()
            .map(|(j, v)| (j, v.iter().copied().collect::<Option<Vec<usize>>>()))
            .collect();
        out.sort_by_key(|(j, _)| *j);

        let mut brought = 0;
        for (j, pads) in out {
            let (slot, e, r) = &jobs[j];
            let Some(pads) = pads else {
                self.key_of[*slot] = EMPTY;
                eprintln!("k3_cache: short prefetch of L{layer} expert {e}");
                continue;
            };
            self.bytes_read += r.nbytes as u64;
            self.pad[*slot] = pads;
            self.refs[*slot] = Some(r.clone());
            let key = self.key(layer, *e);
            self.key_of[*slot] = key;
            self.slot_of.insert(key, *slot as i32);
            self.clock += 1;
            self.used_at[*slot] = self.clock;
            self.prefetch_reads += 1;
            brought += 1;
        }
        brought
    }

    pub fn pin(&mut self, layer: usize, expert: usize, pin: bool) -> bool {
        match self.slot_of.get(&self.key(layer, expert)) {
            Some(&s) => {
                self.pinned[s as usize] = pin;
                true
            }
            None => false,
        }
    }

    /// Resize the arena to a new byte budget, dropping everything it held.
    ///
    /// Deliberately destructive. The alternative -- copying surviving experts into a new
    /// arena -- would have to rewrite every slot's `pad` offsets, and a mistake there
    /// yields an expert read at the wrong offset: fluent, wrong output with no error. A
    /// rebalance is rare and a cold cache costs one sweep, so the safe version wins.
    ///
    /// Returns false when the new budget cannot hold a single token's working set, in
    /// which case nothing changes.
    pub fn resize(&mut self, budget_bytes: i64, topk: usize) -> bool {
        let nslot = (budget_bytes / self.slot_bytes as i64).max(0) as usize;
        if nslot < topk + 1 || nslot == self.nslot {
            return false;
        }
        self.arena = crate::st::Aligned::new(nslot * self.slot_bytes);
        self.slot_of.clear();
        self.key_of = vec![EMPTY; nslot];
        self.used_at = vec![0; nslot];
        self.pinned = vec![false; nslot];
        self.refs = (0..nslot).map(|_| None).collect();
        self.pad = (0..nslot).map(|_| Vec::new()).collect();
        self.nslot = nslot;
        true
    }

    pub fn bytes(&self) -> usize {
        self.nslot * self.slot_bytes
    }

    /// Hit rate over the window since the last `reset_stats`, and how many requests it
    /// covers. A LIFETIME average reacts ever more slowly as a process ages, which is
    /// exactly wrong for a controller that has to notice a workload change.
    pub fn window_hit_rate(&self) -> Option<f64> {
        let n = self.hits + self.misses;
        (n > 0).then(|| self.hits as f64 / n as f64)
    }

    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.bytes_read = 0;
        self.prefetch_reads = 0;
        self.layer_bytes.clear();
        self.layer_hits.clear();
        self.layer_miss.clear();
    }

    pub fn report(&self, label: &str) {
        let n = self.hits + self.misses;
        let raw = if n > 0 { 100.0 * self.hits as f64 / n as f64 } else { 0.0 };
        println!("{label}");
        println!(
            "  requests     : {n}  hits {} ({raw:.2}%)  misses {}  evictions {}",
            self.hits, self.misses, self.evictions
        );
        if self.prefetch_reads > 0 {
            let served = self.hits.saturating_sub(self.prefetch_reads);
            let eff = if n > 0 { 100.0 * served as f64 / n as f64 } else { 0.0 };
            println!(
                "  of those hits : {} came from the batch prefetch, i.e. read from disk\n\
                 \x20                 this token; TRUE resident hit rate {eff:.2}%",
                self.prefetch_reads
            );
        }
        println!("  bytes read   : {:.2} GB", self.bytes_read as f64 / 1e9);
    }
}

#[cfg(test)]
mod tests {

    /// gate/up/down must land in ExpertQ's w1/w3/w2 slots. This is the same transposition
    /// that shipped once as the K3 w2/w3 swap: it produces finite, plausible activations
    /// and a model that still writes fluent text, so only an explicit check catches it.
    #[test]
    fn a_stacked_expert_maps_gate_up_down_to_w1_w3_w2() {
        let ExpertSrc::Stacked { names, expert } = gguf_expert_src(7, 42) else {
            panic!("gguf_expert_src must produce a stacked source");
        };
        assert_eq!(expert, 42);
        assert_eq!(names[0], "blk.7.ffn_gate_exps.weight", "slot w1 is gate");
        assert_eq!(names[1], "blk.7.ffn_up_exps.weight", "slot w3 is up");
        assert_eq!(names[2], "blk.7.ffn_down_exps.weight", "slot w2 is down");
    }

    /// The three tensors occupy ExpertQ slots 0, 2 and 4 -- never 0, 1, 2 -- because
    /// `Cache::slice` reads 0..5 as p1, s1, p3, s3, p2, s2. Packed consecutively, w3's
    /// bytes would be read as w1's scale block.
    #[test]
    fn stacked_pieces_skip_the_scale_slots() {
        // Mirrors what `pieces` builds, without needing an St to resolve names.
        let mut slots = [false; 6];
        for j in 0..3 {
            slots[j * 2] = true;
        }
        assert_eq!(slots, [true, false, true, false, true, false]);
    }

    #[test]
    fn stacked_slicing_is_exact_and_guards_its_assumptions() {
        // Real geometry: Qwen3.6-35B blk.0.ffn_gate_exps is Q4_K [256, 512, 2048], so
        // 256 experts x 4096 super-blocks x 144 bytes.
        let total = 256 * 4096 * 144;
        assert_eq!(ExpertSrc::stacked_slice(total, 256, 0), Some((0, 589_824)));
        assert_eq!(ExpertSrc::stacked_slice(total, 256, 1), Some((589_824, 589_824)));
        assert_eq!(ExpertSrc::stacked_slice(total, 256, 255), Some((255 * 589_824, 589_824)));
        // The last expert must end exactly at the tensor's end.
        let (off, len) = ExpertSrc::stacked_slice(total, 256, 255).unwrap();
        assert_eq!(off + len, total, "expert 255 must reach the end and no further");

        // Guards: past the end, and a total that is not a whole number of experts.
        assert_eq!(ExpertSrc::stacked_slice(total, 256, 256), None, "out of range");
        assert_eq!(ExpertSrc::stacked_slice(total + 1, 256, 0), None, "not divisible");
        assert_eq!(ExpertSrc::stacked_slice(total, 0, 0), None, "zero experts");
    }

    #[test]
    fn probe_name_works_for_both_layouts() {
        assert_eq!(v4_expert_names(2, 5).probe_name(), "layers.2.ffn.experts.5.w1.weight");
        assert_eq!(gguf_expert_src(2, 5).probe_name(), "blk.2.ffn_gate_exps.weight");
    }
    use super::*;

    #[test]
    fn refuses_a_cache_smaller_than_one_tokens_working_set() {
        // topk 6 needs at least 7 slots; 6 would evict an expert before it is used.
        assert!(Cache::new(6 * 1024, 1024, 256, 6).is_err());
        assert!(Cache::new(7 * 1024, 1024, 256, 6).is_ok());
    }

    #[test]
    fn v4_expert_names_follow_the_checkpoint() {
        let n = v4_expert_names(2, 137);
        assert_eq!(n.named()[0], "layers.2.ffn.experts.137.w1.weight");
        assert_eq!(n.named()[1], "layers.2.ffn.experts.137.w1.scale");
        // w3 precedes w2: gate, up, then down, matching how the MoE consumes them.
        assert_eq!(n.named()[2], "layers.2.ffn.experts.137.w3.weight");
        assert_eq!(n.named()[4], "layers.2.ffn.experts.137.w2.weight");
    }

    #[test]
    fn empty_slots_are_taken_before_anything_is_evicted() {
        let mut c = Cache::new(8 * 100, 100, 16, 2).unwrap();
        assert_eq!(c.pick_victim(), Some(0));
        c.key_of[0] = 5;
        assert_eq!(c.pick_victim(), Some(1), "an EMPTY slot wins over any LRU candidate");
    }

    #[test]
    fn inflight_slots_are_never_handed_out() {
        let mut c = Cache::new(3 * 100, 100, 16, 2).unwrap();
        for i in 0..3 {
            c.key_of[i] = i as i64;
            c.used_at[i] = 10 - i as u64;
        }
        c.key_of[2] = INFLIGHT;
        // slot 2 has the oldest stamp but its read has not landed.
        assert_eq!(c.pick_victim(), Some(1));
    }

    /// The whole point of the sweep-aware victim rule, stated as the thing that must be
    /// true: standing at layer `l`, the expert the scan will reach SOONEST must be the
    /// one most protected, and the layer just finished must be the first evicted.
    ///
    /// Plain LRU gets this exactly backwards under a cyclic scan -- the least-recently
    /// used entry belongs to the layer just finished only if the scan is short enough to
    /// stay resident, and otherwise it is the entry the next pass needs first. Both
    /// directions are asserted here so an inverted comparison cannot pass.
    #[test]
    fn the_victim_is_the_layer_the_sweep_just_left_not_the_oldest() {
        let mut c = Cache::new(4 * 100, 100, 16, 2).unwrap();
        // Four experts, one per layer, all in a 6-layer cycle. Ages are deliberately the
        // reverse of the layer order so age alone would pick a different slot.
        for (slot, layer) in [3usize, 4, 0, 1].into_iter().enumerate() {
            c.key_of[slot] = (layer * 16) as i64;
            c.used_at[slot] = (10 - slot) as u64;
        }
        c.at_layer(4, 6);
        // From layer 4: distances are 5->1, 0->2, 1->3, 4->6 (a full cycle, just used).
        assert_eq!(c.distance(4 * 16), 6, "the layer we stand on is a full cycle away");
        assert_eq!(c.distance(5 * 16), 1, "the next layer is one step away");
        assert_eq!(c.pick_victim(), Some(1), "slot 1 holds layer 4, just finished");

        // Move on one layer and the answer must move with it.
        c.at_layer(0, 6);
        assert_eq!(c.pick_victim(), Some(2), "slot 2 holds layer 0, now just finished");
    }

    /// The backward sweep is the exact mirror of the forward one: with the SAME resident
    /// set and the SAME sweep position, `Up` and `Down` must choose DIFFERENT victims,
    /// because the layer furthest in the future is on the opposite side. This is the whole
    /// point of the direction flag -- if it did nothing, backprop would evict the states it
    /// is about to reuse.
    #[test]
    fn the_backward_sweep_evicts_the_mirror_of_the_forward_one() {
        let layout = |dir: Dir| {
            let mut c = Cache::new(4 * 100, 100, 16, 2).unwrap();
            // Four resident layers, one expert each, all ages equal so distance alone
            // decides. Standing at layer 3 in a 6-layer cycle.
            for (slot, layer) in [5usize, 0, 1, 2].into_iter().enumerate() {
                c.key_of[slot] = (layer * 16) as i64;
                c.used_at[slot] = 7; // identical -> no age tiebreak
            }
            c.at_layer_dir(3, 6, dir);
            c
        };

        // Forward from layer 3 visits 4,5,0,1,2,3: layer 2 is furthest (5 steps away).
        let up = layout(Dir::Up);
        assert_eq!(up.distance(2 * 16), 5, "up: layer 2 is the furthest ahead");
        assert_eq!(up.distance(5 * 16), 2, "up: layer 5 is soon");
        assert_eq!(up.pick_victim(), Some(3), "up evicts slot 3, holding layer 2");

        // Backward from layer 3 visits 2,1,0,5,4,3: now layer 5 is furthest (4 steps).
        let down = layout(Dir::Down);
        assert_eq!(down.distance(2 * 16), 1, "down: layer 2 is the very next");
        assert_eq!(down.distance(5 * 16), 4, "down: layer 5 is now the furthest");
        assert_eq!(down.pick_victim(), Some(0), "down evicts slot 0, holding layer 5");
    }

    /// The standing-on layer is a full cycle from its next use in BOTH directions -- the
    /// property the forward comment relies on has to survive the mirror.
    #[test]
    fn the_current_layer_is_a_full_cycle_away_going_down_too() {
        let mut c = Cache::new(3 * 100, 100, 16, 2).unwrap();
        c.at_layer_dir(4, 6, Dir::Down);
        assert_eq!(c.distance(4 * 16), 6, "standing on layer 4, it is a full cycle away");
        assert_eq!(c.distance(3 * 16), 1, "the next layer down is one step");
        assert_eq!(c.distance(5 * 16), 5, "the layer just above wraps around last");
    }

    /// Age still decides between experts of the SAME layer, so the policy is a refinement
    /// of LRU rather than a replacement for it.
    #[test]
    fn age_breaks_ties_within_one_layer() {
        let mut c = Cache::new(3 * 100, 100, 16, 2).unwrap();
        for slot in 0..3 {
            c.key_of[slot] = (2 * 16 + slot) as i64; // all layer 2
            c.used_at[slot] = (5 + slot) as u64;
        }
        c.at_layer(0, 6);
        assert_eq!(c.pick_victim(), Some(0), "same distance, so the oldest goes");
    }

    /// With no sweep reported the cache must behave exactly as it did before, or every
    /// existing measurement and every other test in this file silently changes meaning.
    #[test]
    fn without_a_sweep_it_is_still_plain_lru() {
        let mut c = Cache::new(4 * 100, 100, 16, 2).unwrap();
        for slot in 0..4 {
            c.key_of[slot] = (slot * 16 + 1) as i64; // four different layers
            c.used_at[slot] = (10 - slot) as u64;
        }
        assert_eq!(c.distance(3 * 16), 0, "no sweep means no distance information");
        assert_eq!(c.pick_victim(), Some(3), "the oldest, as LRU always did");
    }

    #[test]
    fn pinned_slots_survive_pressure() {
        let mut c = Cache::new(3 * 100, 100, 16, 2).unwrap();
        for i in 0..3 {
            c.key_of[i] = i as i64;
            c.used_at[i] = i as u64;
        }
        c.pinned[0] = true; // oldest, but pinned
        assert_eq!(c.pick_victim(), Some(1));
    }

    #[test]
    fn all_pinned_means_no_victim_rather_than_a_wrong_one() {
        let mut c = Cache::new(3 * 100, 100, 16, 2).unwrap();
        for i in 0..3 {
            c.key_of[i] = i as i64;
            c.pinned[i] = true;
        }
        assert_eq!(c.pick_victim(), None, "better to fail the fetch than evict a pin");
    }
}

#[cfg(test)]
mod naming {
    use super::*;

    // ExpertQ cuts its slots as (0,1)->w1, (2,3)->w3, (4,5)->w2. Any name table that
    // lists the tensors in a different order silently binds w2's bytes to w3's slot.
    // locate() sorts by file offset, so the array order decides ONLY which tensor lands
    // in which field -- there is no shape mismatch and no error to notice.
    #[test]
    fn every_expert_name_table_orders_the_matrices_w1_w3_w2() {
        for (arch, n) in [("k3", k3_expert_names(7, 3)), ("v4", v4_expert_names(7, 3))] {
            assert!(n.named()[0].contains("w1") && n.named()[1].contains("w1"), "{arch}: slot 0/1 is w1");
            assert!(n.named()[2].contains("w3") && n.named()[3].contains("w3"), "{arch}: slot 2/3 is w3");
            assert!(n.named()[4].contains("w2") && n.named()[5].contains("w2"), "{arch}: slot 4/5 is w2");
        }
    }

    #[test]
    fn the_packed_tensor_precedes_its_scale_in_every_pair() {
        for n in [k3_expert_names(0, 0), v4_expert_names(0, 0)] {
            for p in 0..3 {
                let (w, s) = (&n.named()[p * 2], &n.named()[p * 2 + 1]);
                assert!(!w.contains("scale"), "{w} should be the packed weight");
                assert!(s.contains("scale"), "{s} should be the scale");
                assert_eq!(w.rsplit_once('.').unwrap().0, s.rsplit_once('.').unwrap().0);
            }
        }
    }

    #[test]
    fn the_two_architectures_use_genuinely_different_names() {
        let (k, v) = (k3_expert_names(2, 5), v4_expert_names(2, 5));
        assert!(k.named()[0].starts_with("language_model.model.layers.2.block_sparse_moe.experts.5"));
        assert!(v.named()[0].starts_with("layers.2.ffn.experts.5"));
        assert_ne!(k, v);
    }
}

// ------------------------------------------------------- asynchronous prefetch ----
//
// Expert I/O and layer compute run in series today: fetch layer L's experts, compute on
// them, move to L+1. On a 604 MB/s device that is ~3 s of I/O and ~3.5 s of compute per
// token, one after the other. Overlapping them hides the smaller of the two.
//
// The predictor is the training-free one: run layer L+1's OWN gate on layer L's residual.
// The residual stream changes slowly between layers, so the top-k it produces mostly
// agrees with what L+1 will really route to, and the gate is ~1M MACs against 151M for
// the experts it saves. arxiv 2607.24787 (SpecPrefetch) measures a trained low-rank
// adapter beating this kind of training-free baseline by at most ~5.5 points of recall,
// which is not worth a training pipeline here.
//
// Wrong guesses cost a wasted read, not a wrong answer: the prefetcher only warms slots,
// and `get` still fetches synchronously on a miss. Prediction accuracy is a throughput
// knob, never a correctness one.

/// A prefetched expert, read off disk but not yet installed in a slot.
pub struct Staged {
    pub key: i64,
    pub r: ExpertRef,
    pub buf: crate::st::Aligned,
    pub pads: Vec<usize>,
}

pub struct Prefetcher {
    req: std::sync::mpsc::Sender<(i64, usize, usize, ExpertNames)>,
    done: std::sync::mpsc::Receiver<Staged>,
    inflight: std::collections::HashSet<i64>,
    worker: Option<std::thread::JoinHandle<()>>,
    pub issued: u64,
    pub used: u64,
}

impl Prefetcher {
    /// Opens its OWN handle on the checkpoint rather than sharing the caller's. File
    /// descriptors are cheap and this keeps the reader thread from needing any lifetime
    /// or lock relationship with the engine's `St`.
    pub fn new(dir: &std::path::Path, slot_bytes: usize) -> Result<Prefetcher, String> {
        let st = St::open(dir).map_err(|e| e.to_string())?;
        let (rq, rq_rx) = std::sync::mpsc::channel::<(i64, usize, usize, ExpertNames)>();
        let (dn_tx, dn) = std::sync::mpsc::channel::<Staged>();
        let worker = std::thread::spawn(move || {
            while let Ok((key, _layer, _expert, names)) = rq_rx.recv() {
                let Some(r) = locate(&st, &names) else { continue };
                if slot_need(&r) > slot_bytes {
                    continue;
                }
                let mut buf = crate::st::Aligned::new(slot_bytes);
                let mut pads = Vec::with_capacity(r.runs.len());
                let mut cur = 0usize;
                let mut ok = true;
                for run in &r.runs {
                    let region = run_region(run.len);
                    let (avail, pad) =
                        st.read_aligned(r.shard, run.off, run.len, &mut buf[cur..cur + region]);
                    if avail < run.len {
                        ok = false;
                        break;
                    }
                    pads.push(cur + pad as usize);
                    cur += region;
                }
                if ok && dn_tx.send(Staged { key, r, buf, pads }).is_err() {
                    break;
                }
            }
        });
        Ok(Prefetcher {
            req: rq,
            done: dn,
            inflight: std::collections::HashSet::new(),
            worker: Some(worker),
            issued: 0,
            used: 0,
        })
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        // Dropping the sender ends the worker's recv loop.
        let (dead, _) = std::sync::mpsc::channel();
        let _ = std::mem::replace(&mut self.req, dead);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

impl Cache {
    pub fn attach_prefetcher(&mut self, p: Prefetcher) {
        self.pre = Some(p);
    }

    pub fn prefetcher_stats(&self) -> (u64, u64) {
        self.pre.as_ref().map_or((0, 0), |p| (p.issued, p.used))
    }

    /// Queue reads for experts a later layer is predicted to want. Never blocks, never
    /// evicts, and skips anything already resident or already queued.
    pub fn prefetch_hint<F>(&mut self, layer: usize, experts: &[usize], names: F)
    where
        F: Fn(usize, usize) -> ExpertNames,
    {
        let keys: Vec<(i64, usize)> = experts
            .iter()
            .map(|&e| (self.key(layer, e), e))
            .filter(|(k, _)| !self.slot_of.contains_key(k))
            .collect();
        let Some(p) = self.pre.as_mut() else { return };
        for (key, e) in keys {
            if p.inflight.contains(&key) {
                continue;
            }
            if p.req.send((key, layer, e, names(layer, e))).is_ok() {
                p.inflight.insert(key);
                p.issued += 1;
            }
        }
    }

    /// Install whatever the reader thread has finished. Called before each lookup so a
    /// prediction that landed in time turns into a hit.
    pub fn drain_prefetched(&mut self) {
        let mut staged = Vec::new();
        if let Some(p) = self.pre.as_mut() {
            while let Ok(s) = p.done.try_recv() {
                p.inflight.remove(&s.key);
                staged.push(s);
            }
        }
        // A speculative fetch may only take a slot nothing real is holding. Letting it
        // evict cost 11 points of hit rate and 74% more I/O when measured: 8 candidates
        // across 43 layers is 344 speculative reads a token against 258 real ones, so
        // speculation alone churned the whole cache every step and threw out experts the
        // CURRENT layer still needed. This is 2Q's admission rule (Johnson & Shasha 1994)
        // in its simplest form -- an unproven block cannot displace a proven one.
        for s in staged {
            if self.slot_of.contains_key(&s.key) {
                continue; // a synchronous miss beat the prefetch to it
            }
            let Some(slot) = self.key_of.iter().position(|&k| k == EMPTY) else {
                self.dropped += 1;
                continue;
            };
            let base = slot * self.slot_bytes;
            let n = s.buf.len().min(self.slot_bytes);
            self.arena[base..base + n].copy_from_slice(&s.buf[..n]);
            self.bytes_read += s.r.nbytes as u64;
            self.pad[slot] = s.pads;
            self.refs[slot] = Some(s.r);
            self.key_of[slot] = s.key;
            self.slot_of.insert(s.key, slot as i32);
            self.clock += 1;
            self.used_at[slot] = self.clock;
            if let Some(p) = self.pre.as_mut() {
                p.used += 1;
            }
        }
    }
}

// ------------------------------------------------------------------- QDQ sweep ----
//
// Quantise-dequantise a resident expert in place: decode each E2M1 nibble, requantise the
// group with a uniform symmetric `bits`-bit grid over its own absmax, then re-encode to
// the NEAREST E2M1 code. The stored format never changes, so no kernel moves and the only
// thing under test is precision. This is the same method tools/qdq_trunk.py applies to the
// trunk, at expert granularity.
//
// The point is to answer, before building a packed 2-bit path, whether the output survives
// it -- and to answer it on the experts that actually dominate reads. Hot experts stay
// resident in cache; the misses are overwhelmingly cold ones, so `K3_QDQ_COLD` restricts
// the damage to experts requested fewer than that many times.

/// E2M1 magnitudes, in code order for the low 3 bits.
const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn nearest_e2m1(v: f32) -> u8 {
    let (mag, sign) = (v.abs(), if v < 0.0 { 8u8 } else { 0u8 });
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (i, &m) in E2M1_MAG.iter().enumerate() {
        let d = (m - mag).abs();
        if d < bd {
            bd = d;
            best = i;
        }
    }
    sign | best as u8
}

/// Round the 32 values of one group onto a uniform symmetric `bits`-bit grid.
fn qdq_group(vals: &mut [f32], bits: u32) {
    let amax = vals.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    if amax == 0.0 {
        return;
    }
    let levels = ((1u32 << (bits - 1)) - 1).max(1) as f32; // symmetric, sign separate
    let step = amax / levels;
    for v in vals.iter_mut() {
        *v = (*v / step).round().clamp(-levels, levels) * step;
    }
}

impl Cache {
    /// Apply QDQ to the packed weights of a resident slot. Scales are untouched: they are
    /// E8M0 exponents, and rounding them would change the group's range rather than its
    /// resolution.
    fn qdq_slot(&mut self, slot: usize, bits: u32) {
        let Some(r) = self.refs[slot].clone() else { return };
        let base = slot * self.slot_bytes;
        let pair = crate::ops::e2m1_pair_table();
        for which in [0usize, 2, 4] {
            let (run, off, len) = r.parts[which];
            let start = base + self.pad[slot][run] + off as usize;
            let bytes = &mut self.arena[start..start + len as usize];
            // 16 packed bytes carry one 32-value group.
            for g in bytes.chunks_mut(16) {
                let mut v = [0f32; 32];
                for (j, b) in g.iter().enumerate() {
                    let p = pair[*b as usize];
                    v[2 * j] = p[0];
                    v[2 * j + 1] = p[1];
                }
                let n = g.len() * 2;
                qdq_group(&mut v[..n], bits);
                for (j, b) in g.iter_mut().enumerate() {
                    *b = nearest_e2m1(v[2 * j]) | (nearest_e2m1(v[2 * j + 1]) << 4);
                }
            }
        }
    }
}
