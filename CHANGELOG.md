# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The Rust port and the multi-model generalisation it exists for. The C engine is unchanged
and still passes `make test`; it is the differential oracle the port is graded against.

### Added

- **GGUF container reader and the Q4_K super-block format** (`rust/src/gguf.rs`). The
  engine could previously only read what a lab published as safetensors — in practice
  BF16 or FP8, the two formats least suited to a machine that is bandwidth-bound on
  storage. Every community that actually runs large models on small hardware publishes
  GGUF, for exactly the reason this port measured: a DeepSeek-V4 expert costs 13.37 MB at
  4.25 bits and 88 MB in BF16.

  `scan` emits the same `st::Tensor` records the safetensors scanner does, so everything
  downstream is unchanged — `St::find`, `St::read_aligned`, the expert cache, the
  sweep-aware eviction and the VRAM tier are all byte-range based and do not care what
  produced the range. Block-quantised tensors are read as raw bytes and dequantised per
  block, exactly as MXFP4 already is; `read_f32` refuses them loudly rather than widening
  across a super-block boundary whose scales live at its head.

  **Q4_K, Q5_K and Q6_K** dequantise; all three are verified against `gguf`, the llama.cpp
  project's own reference implementation (`tools/verify_gguf.py` with
  `rust/src/bin/gguf_dump.rs`), on `Qwen2.5-0.5B-Instruct` at Q4_K_M and Q5_K_M:
  **290/290 tensors** agree on dtype, shape, byte offset and byte length, and every
  dequantisable tensor is **bit-identical** — 4.36M values each, 24 tensors per file.

  Each kernel is mutation-tested, and each mutation is the plausible wrong reading:

  | kernel | mutation | corrupted |
  |---|---|---|
  | Q4_K | the eight 6-bit scales read uniformly | 47% |
  | Q5_K | `qh` advanced like `qs` instead of shifting the bit mask | 37% |
  | Q6_K | the four 32-element groups read consecutively, not interleaved | 24% |
  | Q6_K | quants not recentred by −32 | 25% |

  Every one produces finite, in-range, plausible floats — `0.0077` where `0.0045` belongs —
  which is why the comparison is bitwise against a foreign implementation rather than a
  tolerance against our own.

  **Fused matmuls** `matmul_q4k` / `matmul_q5k` / `matmul_q6k` dequantise straight into the
  dot product, never into a buffer. Q4_K and Q5_K are affine (`d*sc*q - dmin*m`), so the min
  does **not** factor out and each 32-element sub-block needs two running sums —
  `d*sc*sum(q*x) - dmin*m*sum(x)`. Q6_K has no min but has *signed* 8-bit scales and quants
  recentred by −32. Verified on real quantised weights against `dequantize(W) @ x` in numpy,
  max relative error 5.9e-8 across every Q4_K/Q5_K/Q6_K tensor in both test files, plus
  bitwise parallel-vs-serial and thread-count independence. Mutation-tested: dropping the
  affine min term moves a row by only ~1% (`2173.34` where `2151.72` belongs) and is caught.

  **Scale folding: one reduction per super-block, not per scale group.** The obvious form
  applies each sub-block's scale to its own dot product, which costs a full horizontal
  reduction per 16 or 32 elements. Q6_K's scale changes every 16, so that was *sixteen*
  reductions per super-block against two accumulate steps each — it measured 5.08 GMAC/s
  where Q4_K managed 14.15. Folding the scale into the weights instead (`w[i] = sc_j * q_i`,
  exact because `|sc*q| ≤ 4064 < 2^24`) allows one reduction per 256 elements and preserves
  `dot_p8`'s precondition, so the AVX2 path stays bit-identical to the scalar one:

  | kernel | before | after |
  |---|---|---|
  | Q4_K | 14.15 GMAC/s | **16.59** |
  | Q5_K | 12.90 | **14.58** |
  | Q6_K | 5.08 | **11.66** |

  Blended for a `Q4_K_M` recipe (⅔ Q4_K, ⅓ Q6_K by weight): 8.87 → **14.53 GMAC/s, 1.64×**.
  Still bit-identical to the `gguf` reference on every tensor in both test files.

  **AVX2 dot, and why the vector path can be exact.** `matmul_mxfp4`'s vector path is bit-identical to
  its scalar one only because of a carefully chosen `i % 8` lane partition that reproduces
  the scalar fold tree exactly; reproducing that for an affine form with two accumulators is
  separate work, and a fast path merely *close* to the scalar one would break the
  bit-exactness contract every other kernel here holds. Scalar first, measured, then
  vectorised if it proves to be on the critical path.

  Note that llama.cpp's "Q4_K_M" is a recipe, not a type: the Qwen2.5-0.5B build is 132
  Q5_0 tensors, 121 F32, 13 Q8_0, 12 Q6_K and only 12 Q4_K. The block-size table covers
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K so mixed files scan; only Q4_K
  dequantises so far, and an unreadable type is named in the error rather than guessed at.

- **GPU memory as an expert victim cache** (`rust/src/vram.rs`), behind `K3_VRAM_GB`.
  Not a compute offload — compute is ~8% of a token and no arithmetic moves a
  bandwidth bound. What the card has is 3.7 GB of idle memory on a link two orders of
  magnitude faster than the USB SSD, and it costs *zero system RAM*, which matters
  because the 8.29 GB resident trunk leaves under 2 GB for experts on a 15 GB machine.
  Evicted experts spill to the device instead of being dropped, and a later miss is
  served by DMA (~2–5 ms) rather than a re-read (~27 ms). The CUDA driver API is reached
  by `dlopen`, so a machine with no GPU or no driver still builds and runs — the tier
  just reports why it is off. A slot round-trips verbatim, so an expert served from VRAM
  is bit-identical to one read from disk: this can change how long a token takes and
  cannot change what it says.

  Measured, 1.6 GB RAM cache + 3.0 GB VRAM, output identical to both baselines:
  steady state 8.96 → **7.97 s/token** (−11.0% against LRU), disk reads
  36.48 → **31.78 GB** (−12.9%), with 9.29 GB served from the GPU.

  Two findings on the way, both the same pathology at different levels. Wiring the spill
  only into `admit` caught 39 of 2557 evictions, because the batch prefetch — not
  `admit` — does the bulk of the evicting. And with the spill fixed but the device tier
  evicting by LRU, 2557 spills produced 50 hits: under a cyclic scan the
  least-recently-spilled entry is exactly the one the sweep is about to ask for, so every
  spill overwrote the entry that was about to hit. Giving the device tier the same
  distance rule as the host cache took it from 50 hits to 694.

- **Sweep-aware expert eviction** (`Cache::at_layer`), replacing plain LRU. Expert access
  is a *cyclic scan*: every token walks the same 43 layers in order, and one token's
  experts (3.45 GB) do not fit in the cache. Under a cyclic scan whose working set exceeds
  the cache, LRU is the worst available policy — the least-recently-used entry is the one
  the next pass reaches soonest. Belady (1966) says evict furthest-in-future, and on the
  *layer* axis that is not a prediction at all: at position `l`, an expert for layer `L`
  cannot be touched again for exactly `(L - l) mod cycle` steps. Victims are now ranked by
  that distance, with age as the tiebreak; with no sweep reported it degenerates to exactly
  the LRU it replaced. Measured, 1.6 GB cache, output identical: steady state
  8.96 → 8.67 s/token (−3.2%), 36.48 → 35.78 GB read (−1.9%), resident hit rate
  6.30% → 7.33%. The gain is small because the cache holds less than half of one token's
  working set, so almost no cross-token reuse is *possible* at this size regardless of
  policy — the effect should grow sharply once the cache exceeds 3.45 GB.

- **DSpark, DeepSeek-V4-Flash's trained block drafter** (`rust/src/dspark.rs`), behind
  `K3_DSPARK=1`. The `mtp.*` tensors are not DeepSeek-V3-style MTP heads: `main_proj` is
  [4096, 12288] because it consumes the concatenated hc-mean hidden states of main-model
  layers 40, 41 and 42 (`dspark_target_layer_ids`). It drafts a block of five tokens in
  one pass, seeded `[real_token, noise, noise, noise, noise]` with bidirectional attention
  inside the block, over a KV cache keyed to the *main* model's positions; a rank-256
  Markov head then biases each position by the token chosen at the one before it, and a
  confidence head scores each draft. Verified end to end against an independent numpy
  parse of the same shards (`tools/verify_dspark.py`, with `rust/src/bin/dspark.rs`):
  `main_x`, all three stages' KV rows, the `hc_head` output and the confidence all agree
  to ~1e-6, and the drafted token ids are identical.

  Measured on the real checkpoint, 1.6 GB cache, 8 tokens, all three token-identical to
  serial decode: serial 94.8 s / 36.48 GB read; DSpark at k=2 103.2 s / 37.71 GB, 67%
  of drafts accepted; DSpark at k=5 168.4 s / 51.26 GB, 27% accepted. **It is a net loss
  on this hardware** and stays opt-in: widening the verification batch is paid in expert
  bytes on a saturated device, and the three DSpark stages add ~1.2 GB of their own
  traffic to a 119-slot cache that cannot hold them. `K3_DSPARK_K` caps the draft length
  (default 2) and `K3_DSPARK_CONF` cuts it at the first position the confidence head
  doubts.

- **The Rust port** (`rust/`), one crate, one file per C translation unit. Ported and
  gated against real bytes: the config reader (byte-identical to C on the 93-layer
  config), the safetensors reader (1,565/1,565 tensors against an independent Python
  parse), all ~40 kernels (all 14 op fixtures), and the expert cache (byte-exact under
  571 evictions). Plus `libm.rs`, whose parity gate sweeps every one of the 4,278,190,082
  non-NaN `f32` bit patterns against the C symbols the engine links, settling the one
  genuine cross-language risk. See [docs/RUST_PORT.md](docs/RUST_PORT.md).
- **A second architecture, DeepSeek-V4-Flash** (304B), read from real checkpoint shards:
  MLA with o-LoRA, attention sink, Compressor and Indexer; Hyper-Connections residual;
  hash routing; FP8-block matmul; clamped SwiGLU; `sqrt(softplus)` scoring. Its 256 routed
  experts turn out to use exactly K3's MXFP4 layout, so `matmul_mxfp4` reads a 304B model
  unchanged — covering 277B of its 304B parameters. The forward pass is wired end to end
  but **has never produced a token**: no machine in CI holds all 48 shards.
  See [docs/MULTI_MODEL.md](docs/MULTI_MODEL.md).
- Generalisation seams in `ops.rs`, each gated so that widening them did not move K3's
  arithmetic: `W` (tagged weight format), `Glu`, `Scoring`, `router_scored`,
  `router_hashed`, and a `Residual` trait with K3's block-snapshot and DeepSeek-V4's
  Hyper-Connections implementations. A test asserts `router_scored` reproduces the K3
  router **bitwise** on sigmoid.
- **Phase 1 of the port is complete.** The remaining modules are in: `io.rs` (O_DIRECT
  trunk streaming with a pinned prefix and an asynchronous reader), `bind.rs` (tensor
  binding from shards or from a packed run, every element count checked before a byte is
  read), `tok_k3.rs` (K3's tiktoken `.model` and the Kimi pre-tokenizer, with the Unicode
  class tables regenerated from the C rather than substituted with Rust's `char` methods),
  and `bin/k3_run.rs` (the decode loop). `decoder_layer_inc` threads the MLA KV cache,
  `route_chunk` provides chunk-union dedup and `moe_packed` consumes streamed experts in
  their packed MXFP4 form.
- **`make rust-golden`**, a whole-stack gate on a tiny real checkpoint: C and Rust logits
  byte-identical, Rust against the torch reference elementwise, greedy and incremental
  decode agreeing with the C engine token for token, and identical ids across trunk
  budgets from fully streamed to fully pinned.
- **Threading in the port**, `rayon` in place of OpenMP for all seven of the C engine's
  parallel regions: the four matmuls, the router's scoring loop, the KDA recurrence over
  heads, and the concurrent expert prefetch (reserve every slot, sort by disk offset, read
  into disjoint arena regions, publish only what arrived). Each arithmetic one is gated by
  running the kernel under one thread and under eight and requiring **bitwise** equality,
  which is stronger than the op fixtures' tolerance. A 4096×4096 bf16 matmul goes from
  3.05 ms to 0.56 ms on 16 threads.
- The `RING >= 2` invariant is now structural rather than documented. `k3_trunk.c:274`
  records a run where a one-slot ring let the reader thread overwrite the layer the main
  thread was computing on, producing fluent wrong tokens with no diagnostic. In `io.rs`
  the reader owns a slot only by having it moved into it over a channel, so it cannot
  hold the one the caller is using.
- CI now builds and gates the port: `make rust`, `rust-test`, `rust-parity`, `rust-diff`,
  and clippy with warnings denied. Previously no job in the workflow mentioned cargo, so
  the active development surface was entirely ungated. The declared `rust-version = 1.85`
  floor is now verified by its own job rather than asserted.

- **An architecture descriptor** (`rust/src/arch.rs`): the checkpoint's own `config.json`
  chooses the family via `model_type` / `architectures[0]`, and config and tensor names
  become data rather than code — the refactor `docs/RUST_PORT.md:19` names as the payoff
  and `README.md` called "the next refactor". Four families are described: Kimi K3,
  DeepSeek-V4-Flash, GLM-5.1 and MiniMax-M3. An unrecognised `model_type` is refused by
  name rather than guessed at.
- **`rust/src/bin/run.rs`**, one binary for every architecture. It reads the descriptor,
  validates the checkpoint, and refuses an incomplete one by naming what is absent.
- The three published `config.json` files are checked in under `tests/fixtures/arch/` and
  parsed by `rust/tests/arch_configs.rs`. A descriptor that only agrees with hand-written
  fixtures agrees with itself and nothing else; this one is additionally checked against
  the measurements, reproducing `docs/MULTI_MODEL.md`'s 13.37 MB per routed expert and
  3.45 GB per token from `config.json` alone.
- **A resident trunk for DeepSeek-V4** (`rust/src/v4run.rs`). The trunk is loaded once and
  kept in its stored FP8/bf16 form; the previous loop re-read all 8.29 GB from disk on
  every token and re-prefilled the whole sequence each step.
- **Kernels for the third and fourth architectures** (`rust/src/gqa.rs`): grouped-query
  attention, the two rope pairings (adjacent vs halves), partial rotary, and block-sparse
  row selection; plus `Glu::SwigluOai` and `ops::rmsnorm_gemma` in `ops.rs`. Each is
  mutation-tested — broken on purpose, with the test watched to fail.

- **Native MXFP4 SIMD decode.** The AVX2 expert kernel expanded each packed byte through a
  256-entry table into a `wf[64]` staging buffer and reloaded it -- an 8-byte load and
  8-byte store per byte, costing more than the dot product it fed. E2M1 is clean
  sign-magnitude, so `_mm256_permutevar8x32_ps` serves as an exact 8-entry in-register LUT
  and the sign is one OR. **11.29 -> 18.97 GMAC/s (1.68x) at DeepSeek-V4's real shapes,
  and bit-identical to the scalar path** -- no gate weakened, `rust-golden` still shows K3
  byte-identical to the C engine.
- **Expert reads issued per RUN rather than per expert.** A DeepSeek-V4 expert is two runs
  ~341 MB apart; reading them serially inside one task capped queue depth at top-k on a
  device measuring 378 MB/s at depth 1 against 604 at depth 3. Jobs are also now sorted by
  the *largest* run: sorting by `runs[0]` sequenced the 786 KB of scales and left the
  12.6 MB of weights -- 94% of the bytes -- unordered. Cold prompt pass 131.5 -> 101.9 s.
- **Speculative expert prefetch**, off by default (`K3_PREFETCH=1`). Predicts layer L+1's
  experts by running its own gate on layer L's residual, and reads them on a background
  thread. Measured slower than no prefetch on a bandwidth-saturated device even though it
  improves hit rate and bytes read; see docs/MULTI_MODEL.md for the numbers and why.
- **`tools/sim_policy.py`**, which replays a real expert trace under OPT, LRU, LFU, LRU-2,
  2Q and ARC. On DeepSeek-V4 it shows a better policy is worth 1-3 points at usable cache
  sizes -- and that LRU collapses to **0.00%** below the knee, exactly matching the
  engine's measured hit rate at a 2 GB cache.
- **`--dump-cache-trace`**, so one run yields the whole hit-rate-versus-capacity curve
  offline instead of re-running a 304B model once per policy.

### Fixed

Three defects in the DeepSeek-V4 path, all found while building DSpark, and all of the
same kind: the model kept producing fluent, plausible text with each of them in place.

- **The router bias was never loaded, for any V4 layer.** The checkpoint stores it as
  `ffn.gate.bias`; the loader asked for `ffn.gate.e_score_correction_bias` — the
  DeepSeek-V3/HF spelling — and swallowed the miss with `.ok()`. `Gate.forward` adds that
  bias before the top-k, so for forty of the forty-three decoder layers the engine had
  been selecting the wrong six experts on every token since V4 first ran. Now it tries
  both names and *errors* if neither is present, except on the three hash-routed layers
  where `Gate.__init__` legitimately sets it to `None`. Fixing it changed the sample
  output from a loop (`" Paris.", "The capital of"`) to `" Paris. The capital of Spain
  is Madrid"`.

- **`hc_head` was a plain mean.** DeepSeek-V4 reduces its four Hyper-Connection copies to
  one with a learned sigmoid gate, and the checkpoint ships `hc_head_fn [4, 16384]`,
  `hc_head_base [4]` and `hc_head_scale [1]` for it. Both the library decode loop and
  `bin/v4_run` averaged the copies instead. The reduce is followed by an RMSNorm, which
  cancels any *global* rescaling, so the error showed up only as a wrong relative
  weighting between the four copies — enough to move logits, not enough to disturb the
  prose. Now `ops::hc_head`, shared with DSpark's own output stage.

- **Speculative decoding never executed.** The decode loop tested `pos < ids.len()` for
  "still consuming the prompt", but every emitted token is appended to `ids`, so the test
  stayed true forever and the loop fed its own output back one token at a time down the
  prompt path. Output was correct — that path is plain serial decode — but `k` was always
  1, so the n-gram drafter, the batched union fetch and every measurement that depended on
  them were dead code. The test is now against `prompt_len`.

- **The K3 expert name table bound w2's bytes to w3's slot.** `ExpertQ` cuts its slots as
  w1, w3, w2; the table listed them in on-disk order, w1, w2, w3. Every routed expert then
  computed `situ_glu(w1(z), w2(z)) · w3`. There was no shape mismatch and no error: the
  argmax still agreed with the reference and the correlation was 0.988. Only an
  elementwise comparison against an independent implementation found it, which is what
  `make rust-golden` now runs on every build.
- **DeepSeek-V4 `compress_ratios` was wrong for layers 40, 41 and 42.** The released array
  has 46 entries for 43 decoder layers — the three trailing zeros belong to the
  layer-sized blocks past the decoder stack, not to the last three layers. Reading them as
  the last three gave those layers a ratio of 0, which is a *valid* ratio meaning pure
  sliding-window attention: no crash, no shape mismatch, and three of 43 layers quietly
  attending over a 128-token window instead of the whole compressed history. The table is
  now transcribed verbatim as `v4::COMPRESS_RATIOS` rather than recomputed from a formula,
  with five tests pinning it.
- **RMSNorm accumulation width is no longer hardcoded.** `docs/MULTI_MODEL.md` named this
  as the one seam still wrong: K3's reference upcasts to f64 and DeepSeek's does not, and
  the kernel assumed f64 for everyone. It is now `Spec::rms_acc` and `ops::rmsnorm_acc`.
- **`tools/verify_v4_{attn,compress}.py` read the wrong shard on a full download.** Both
  took `paths[0]`, which was the layer's shard only because the documented workflow
  downloads one shard at a time; against all 48 that is the embedding shard and every
  lookup raised KeyError, which reads as a corrupt download. They now scan for the shard
  holding the requested layer.
- `docs/MULTI_MODEL.md` claimed the DeepSeek-V4 forward pass and expert cache were
  unwired; both were implemented. `docs/RUST_PORT.md`'s status table marked one of five
  finished modules as done and described a layout that never existed.
  `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` did not mention `rust/` at all, and
  `docs/README.md` indexed neither port document.

## [1.0.0] - 2026-08-07

Verified end to end on the full released checkpoint, and made substantially faster, with
byte-identical output preserved at every step. The first-run experience, which was broken
on a clean clone, now works.

### Added

- **`--preset auto`**: sizes the trunk and expert-cache budgets from the machine's own free
  RAM, trunk-first, so a user need not pick a preset by hand. A gigabyte given to the trunk
  is worth far more than a gigabyte of expert cache, and auto pins accordingly, capping the
  pin below the RAM ceiling after a heavy-pin regression was measured.
- **Chunk-union prefill**: a batched-prefill MoE that fetches each unique routed expert once
  per chunk instead of once per token, measured to read about half the expert bytes on a
  prompt, with the generated token bit-identical to the per-token path.
- **Conversation resume** (`--save-state` / `--load-state`): carries the recurrent state and
  KV cache to disk so a second turn resumes instead of re-reading the whole prompt, measured
  3.9x faster on turn two with identical output. Refuses to restore state from a different
  architecture.
- **`--spec N`**: speculative decode by n-gram drafting with batched greedy verification;
  output is exactly the serial greedy decode by construction.
- `--tf-check`, teacher-forced agreement over an id sequence in one sweep, for measuring
  draft quality; `tools/qdq_trunk.py` and `tools/int8_trunk.py` for deriving quantized
  trunks.

### Changed

- **Fused matmul kernels** (fp32, bf16, MXFP4): sixteen partitioned accumulators with
  explicitly fused products, taking the trunk matmul to its memory floor (about eight times
  less per-token compute) while keeping the scalar and AVX2 paths bitwise identical.
- **KDA recurrence parallelised over heads**, bit-identical to the serial form.
- `scripts/k3-doctor.sh` per-preset speed expectations refreshed to the v1.0.0 numbers, with
  the streaming presets noted as disk-bound and the resident tier as compute-bound.

### Fixed

- All shell scripts are committed executable; the first documented command no longer fails
  with Permission denied on a clean clone.
- `scripts/download-model.sh` uses the current `hf` CLI and pins an immutable revision with
  checksum verification; it no longer attempts a pip install that cannot succeed on the
  target OS, and refuses to start without free space for the checkpoint.
- `scripts/k3-doctor.sh` no longer fails a machine that can build and test the engine; the
  memory floor is a warning about running the checkpoint, not a hard stop.
- The config-refusal fixtures the docs describe now exist and are gated in `make test`,
  ctest and CI; the tokenizer leg reports NOT RUN rather than passing silently; CI runs
  `make test` rather than a hand-picked subset.
- A silent-corruption path in the MLA KV overflow and one in the single-slot trunk reader,
  both of which could emit a plausible wrong token, now abort or are prevented.
- The MXFP4 packer alignment and the tiny-checkpoint scale rule.

### Research notes, not shipped as features

- Lossless trunk compression and a quantized-self-draft hybrid were both built and measured,
  and both turned out to help only narrow regimes. The findings and prototypes are kept in
  [`docs/notes/`](docs/notes/).

## [0.1.0] - 2026-07-31

First public release.

### Added

- Full 93-layer Kimi K3 inference: 69 KDA + 24 Gated MLA layers, 896 routed experts with
  top-16 selection, SiTU-GLU, Attention Residuals, native MXFP4 expert weights.
- **Trunk streaming**, which turns the memory budget into a dial rather than a floor. The
  model runs in 8 GB and in 224 GB and produces byte-identical output at every budget
  measured in between.
- MXFP4 matmul that consumes packed nibbles directly, never materialising a dequantised
  expert.
- BPE tokenizer in C, reading the released `tiktoken.model` directly, text in, text out
  with no external step.
- Config reader that loads the checkpoint's own `config.json` and **refuses** a config it
  cannot fully understand rather than defaulting missing fields.
- Incremental decode with a KV cache and carried recurrent state, verified to produce the
  same tokens as full recompute.
- Named memory presets (`--preset laptop|desktop|workstation|server|max`) derived from
  the measured memory ladder.
- `scripts/k3-doctor.sh`, reports whether a machine can run the model, which preset
  fits, and how fast its storage is.
- `scripts/download-model.sh`, fetches the checkpoint and verifies it byte-exactly
  against the published total, because a partial download produces wrong output silently.
- Test suite that runs entirely without model weights: op fixtures, expert cache,
  safetensors reader, config reader, and end-to-end oracle gates (teacher forcing,
  greedy decode, and incremental decode).
- CI: build matrix across GCC and Clang, warnings-as-errors, ASan and UBSan, Python and
  shell lint. Tokenizer parity is built and reported but CANNOT gate on a clean
  checkout, because it needs the vocabulary that ships with the model weights; run
  `make tok` locally against a downloaded checkpoint.

### Known limitations

- No chunked prefill, so long prompts are impractical despite a 32k context ceiling.
- Greedy decoding only; no chat template; no serving layer; no vision; CPU only.

See [docs/ROADMAP.md](docs/ROADMAP.md).

[Unreleased]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/releases/tag/v0.1.0
