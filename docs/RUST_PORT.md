# The Rust port

A 1:1 port of the C99 engine to Rust, gated on **byte-identical output** at
every step. This document records what has been decided, what has been
measured, and what is left.

The C build is not being replaced as it goes. It stays in the tree and stays
buildable, because it is the differential oracle the port is graded against.
Deleting it before the port is finished would remove the only thing that can
prove the port is correct.

## Why

Three reasons, in the order they actually matter:

1. **Multi-model support.** Nothing about MoE streaming, expert caching or
   trunk pinning is Kimi-K3-specific. DeepSeek R1, GLM-5.2 and MiniMax M3 are
   all MoE, all too large for consumer RAM, and all would benefit from this
   engine. The architecture-specific parts are the tensor-name table in
   `src/model/k3_bind.h` and the layer dispatch in `src/core/k3_ops.c`; both
   can become data rather than code. That refactor is the payoff, and the port
   is the natural moment to do it.
2. **Memory safety where it is actually load-bearing.** The I/O layer does raw
   pointer arithmetic into `O_DIRECT` buffers, hands a ring buffer to a reader
   thread, and depends on a `RING >= 2` invariant whose violation is *silent* —
   `k3_trunk.c:274` documents a run that produced fluent, wrong tokens with no
   diagnostic of any kind. That class of bug is what the borrow checker is for.
3. **Dependency management.** `cargo` in place of four vendored headers.

Speed is deliberately **not** on this list. The workload is device-bandwidth
bound at every memory budget except the largest (`docs/TUNING.md`), and the
trunk matmul is already at its memory floor. A language change cannot move
either number. See "What the port is not for" below.

## The contract

Byte-identical output against the C build, gated per kernel.

This is not a stylistic preference. The whole test suite rests on it: output is
byte-identical across all twelve memory budgets today, and that property is what
lets `benchmarks/memory-ladder.sh` attribute a change in seconds-per-token to
the change under test rather than to numerical drift. Give it up and every
performance comparison in `docs/` becomes unfalsifiable.

Two things make the contract cheaper to hold in Rust than it would be in most
languages:

- **Rust never contracts FMA implicitly.** The C build needs
  `-ffp-contract=off` (Makefile:86) to stop the compiler folding `a*b+c` into a
  single fused instruction with different rounding. Rust has no equivalent
  footgun, so that half of the problem does not exist.
- **`core::arch::x86_64` maps 1:1 onto the existing intrinsics.** The AVX2
  kernels transliterate; `_mm256_fmadd_pd` is `_mm256_fmadd_pd`.

## Measured: the libm question is closed

The one genuine cross-language risk was the transcendentals. IEEE-754 mandates
correct rounding for `sqrt` but says nothing about `exp` or `tanh`, so two
conforming implementations may legitimately differ in the last ulp. A drift
there would invalidate every downstream bit-exactness gate — and inside a
93-layer stack of softmaxes, a last-ulp drift does not stay a last-ulp drift.

`rust/src/libm.rs` settles it by exhaustion. `make rust-parity` sweeps **every one
of the 4,278,190,082 non-NaN `f32` bit patterns** and compares each Rust method
against the C symbol the engine links:

| function      | call sites in `k3_ops.c`                                    | result |
|---------------|-------------------------------------------------------------|--------|
| `expf`        | `sigmoidf_:102`, `kda_decay:167,173`, softmaxes `:366,:481`, gate `:382`, router `:429` | identical |
| `tanhf`       | `situ_glu:116,117`                                            | identical |
| `sqrtf`       | `mla scale:308`, `kda qscale:865`                             | identical |
| `sqrt` (f64)  | `rmsnorm:97,471`, `l2norm_:796`                               | identical |
| `ldexpf(1,n)` | `K3_E8M0:1208`, `mxfp4 dequant:1346`                          | identical over n ∈ [-127, 127] |

**No vendored libm and no FFI are required.** Rust's `f32::exp` lowers to
`llvm.exp.f32`, which on this target becomes a call to the same glibc `expf`,
so the agreement is structural rather than lucky. It is asserted anyway: that
reasoning stops holding the moment someone retargets to musl or aarch64, which
is exactly when a silent drift would be hardest to find. The gate is cheap
(~16 s on four cores) and runs in CI, in the `rust` job of `.github/workflows/ci.yml`.

`ldexpf` is replaced by a pure-Rust `exp2i` that writes the exponent field
directly. Note the subnormal branch: the E8M0 table's lowest entry is `2^-127`
and binary32's smallest *normal* is `2^-126`, so that entry cannot be built by
writing an exponent field alone.

## Layout

**One crate, one file per C translation unit.** File-for-file correspondence
is load-bearing rather than cosmetic: the port is graded by running each
module against the C build's test for the same module, and a reviewer can diff
`src/ops.rs` against `src/core/k3_ops.c` side by side. A crate per module
bought nothing at this size except nine manifests to keep in sync.

```
rust/
  Cargo.toml
  src/
    lib.rs      module map and the crate-wide lints  (was include/k3/k3.h)  [done]
    libm.rs     scalar math + the parity gate        (new)                  [done]
    cfg.rs      config reader, both JSON shapes      (k3_cfg.h)             [done]
    st.rs       safetensors reader                   (k3_st.c)              [done]
    fmt.rs      output formatting                    (new)                  [done]
    tok.rs      HF tokenizer.json                    (third_party/tok*.h)   [done]
    tok_k3.rs   K3 tiktoken .model + Kimi pretokenizer (k3_tok.h)           [done]
    ops.rs      the ~40 kernels                      (k3_ops.c)             [done]
    cache.rs    expert cache                         (k3_cache.c)           [done]
    v4.rs       DeepSeek-V4 architecture             (new)                  [done]
    io.rs       O_DIRECT streaming, async reader     (k3_trunk.c + k3_load.c) [done]
    bind.rs     tensor-name binding                  (k3_bind.c)             [done]
    bin/
      v4_run.rs      DeepSeek-V4 decode loop and CLI                        [done]
      k3_run.rs      decode loop and CLI            (k3_run.c)             [done]
      test_cfg.rs    mirrors tests/unit/test_cfg.c                          [done]
      test_st.rs     mirrors tests/unit/test_st.c                           [done]
      libm_parity.rs exhaustive libm gate                                   [done]
      v4_{attn,compress,expert,cache}.rs  verification harnesses            [done]
  tests/        integration tests over tests/fixtures/
```

There are two tokenizers because the checkpoints ship two formats. `tok.rs` wraps the
`tokenizers` crate for HF `tokenizer.json`, which covers DeepSeek-V4, GLM-5.1 and
MiniMax-M3; `tok_k3.rs` reads K3's `tiktoken.model` directly, because K3 ships no
`tokenizer.json`. Its Unicode class tables are regenerated from
`third_party/tok_unicode{,_o200k}.h` rather than substituted with Rust's `char` methods:
`char::is_alphabetic` is Alphabetic, not `\p{L}`, and the difference would silently
retokenize part of the vocabulary. `fmt.rs` and `v4.rs` have no C counterpart.

Not a single `.rs`: `ops.rs` alone is ~1,100 lines, and folding `st`/`io`/
`cache`/`bind` in on top would give one 3,000-line file that no longer maps
onto anything in the C tree — which is the property that makes "port this
file, gate it, move on" work at all. The C project drew the same line.

Order is bottom-up, and each crate is gated by the C suite's existing test for
the same module before the next one starts.

| # | C source | Rust module | Gate | Status |
|---|----------|-------------|------|--------|
| 1 | `include/k3/k3_cfg.h` | `cfg` | `test_cfg` | **done** |
| 2 | `src/io/k3_st.c` | `st` | `test_st`, `verify_st.py` | **done** — 1,565/1,565 tensors |
| 3 | `third_party/tok*.h` | `tok`, `tok_k3` | `test_tok`, `tok_parity.py` | **done** — HF JSON and the tiktoken `.model` |
| 4 | `src/core/k3_ops.c` | `ops` | `test_ops`, 14 fixtures | **done** — all 14 |
| 5 | `src/cache/k3_cache.c` | `cache` | `test_cache` | **done** — byte-exact under 571 evictions |
| 6 | `src/io/k3_{load,trunk}.c` | `io` | `test_expert` | **done** — ring, async reader, O_DIRECT |
| 7 | `src/model/k3_bind.c` | `bind` | `test_real_layer` | **done** — shards and packed-run paths |
| 8 | `src/cli/k3_run.c` | `bin/k3_run` | golden logits, `cmp_logits.py` | **done** — byte-identical to C |

**Phase 1 is complete.** `make rust-golden` builds a tiny real checkpoint (BF16 trunk,
MXFP4 experts, HF names, torch reference) and gates the whole stack rather than one
module at a time:

```
  ok    logits byte-identical, C vs Rust, full stack
  ok    Rust logits match the torch reference (5.223827e-06)
  ok    greedy decode agrees with the C engine, 8 tokens
  ok    incremental decode agrees (KV cache + carried KDA state)
  ok    trunk budget 0.0005 GB: identical ids      <- nothing pinned, every layer streams
  ok    trunk budget 0.05 GB: identical ids        <- all 13 pinned
```

The last pair is the memory-ladder claim reproduced in the port: memory buys speed, not
capability. It is also the property an asynchronous reader writing over a live slot breaks
silently, which is why `RING >= 2` is structural in `io.rs` rather than a comment — the
reader owns a slot only by having it moved into it over a channel, so it cannot hold the
one the caller is computing on.

**What the golden gate caught.** The expert name table listed w1, w2, w3 in disk order,
but `ExpertQ` cuts its slots as w1, w3, w2, so every routed expert computed
`situ_glu(w1(z), w2(z)) · w3`. The argmax still agreed with the reference and the
correlation was 0.988. Nothing but an elementwise comparison against an independent
implementation would have found it, and the module fixtures could not: `matmul_mxfp4` was
correct, the cache was correct, and the bug lived in the seam between them.

Two things that are not steps in this table and are easy to miss:

- **Threading — all seven regions done.** The six arithmetic ones in `k3_ops.c` are
  ported to `rayon`: the four matmuls (chunked over output rows, with the `if (out > 64)`
  clause reproduced as `PAR_MIN_ROWS`), the router's scoring loop, and the KDA recurrence
  over heads. The seventh, `k3_cache.c:180`, is the concurrent expert prefetch: it
  reserves every slot, sorts by disk offset, reads into disjoint arena regions in
  parallel, and publishes only what arrived.

  Each one is gated by running the kernel under a one-thread pool and an eight-thread
  pool and requiring **bitwise** equality — see `ops::threading` and the two
  `kda_layer_*_is_independent_of_thread_count` tests. That is stronger than the op
  fixtures, which would accept a chunking mistake that only perturbed the last bits.

  Two of those tests were vacuous when first written, and it is worth recording why: the
  synthetic weights had a period that aliased with the chunk sizes, so a row's data
  depended only on its parity, and dropping a stride offset entirely still produced
  identical output under both pools. Every offset is now confirmed by breaking it on
  purpose and watching the test fail. Measured on a 4096×4096 bf16 matmul: 3.05 ms
  serial, 0.56 ms on 16 threads. Sub-linear, as `docs/TUNING.md` would predict for a
  kernel already near its memory floor.
- **Kernels.** `decoder_layer_inc` now threads the MLA KV cache through, `route_chunk`
  provides the chunk-union dedup that `k3_moe_prefill` exists for, and `moe_packed`
  consumes streamed experts in their packed MXFP4 form. The `k3_*_scratch` sizing
  functions are deliberately not ported: they exist so C can hand-allocate one arena, and
  Rust's kernels allocate their own scratch.

**The Python tooling in `tools/` is not being ported.** It is the oracle, and
it keeps working against the Rust binary unchanged because it compares files,
not internals: `emit_fixtures.py`, `k3_ref.py`, `cmp_logits.py`,
`sim_cache.py`, `verify_{kda,mla,expert,st,real_layer}.py`.

## Scope: what is load-bearing, and what is not

File line counts overstate the work by a wide margin. The tree is 10,832 lines
across C sources, of which **7,507 are code and 2,479 are comment** — the
engine itself is 4,398 lines of code, not the 6,798 its files total. Filtering
further by what actually has to exist to run a forward pass:

### Tier 1 — the reusable engine (~2,940 code lines)

Everything a *any* MoE model needs. This is what the port is for.

| part | code lines |
|---|---:|
| `k3_st.c` safetensors reader | 465 |
| `k3_trunk.c` streaming + async reader | 426 |
| `k3_bind.c` tensor binding (becomes arch-generic) | 385 |
| `k3_cache.c` expert cache | 323 |
| `k3_load.c` expert fetch | 133 |
| `k3_ops.c` architecture-neutral kernels | ~660 |
| `k3_cfg.h` config reader — **done** | 187 |
| the decode loop out of `k3_run.c` | ~211 |
| tokenizer glue | ~150 |

### Tier 2 — Kimi-K3-specific (~325 code lines)

Real work, but it does **not** transfer to the other targets: DeepSeek R1 and
GLM-5.2 use SwiGLU rather than SiTU-GLU, standard attention rather than KDA,
and FP8 rather than MXFP4. After the Phase 2 refactor these become one
architecture's implementation of a trait, sitting beside R1's rather than
being replaced by it.

- KDA: `shortconv` 33, `kda_decay` 16, `kda_step` 43, `kda_layer` 88
- `situ_glu` 14
- MXFP4: `matmul_mxfp4` 87, `mxfp4_dequant` 31, E8M0/pair tables 13

### Tier 3 — not ported (~800 code lines)

`k3_run.c`'s `main()` is 934 lines, of which only ~211 run a model:

| lines | section | disposition |
|---:|---|---|
| 13 | the generate loop | port |
| 36 | incremental decode | port |
| 51 | buffer allocation | port |
| 111 | prompt + tokenize | port |
| 103 | setup / arg handling | thin version |
| 339 | `--tf-check` + final reporting | skip |
| 135 | pre-flight reporting | skip |
| 75 | `--preset auto` budget solver | later |
| 71 | `--spec` speculative decode | later |

Also skipped: `benchmarks/bench_kernels.c` (92) and `tests/unit/scale_test.c`
(159) — the latter is a memory-sizing calculator, not a correctness gate.

### `third_party/` needs almost no hand-porting (957 code lines)

- `json.h` (135) → `serde_json`. **Done**, zero lines written.
- `tok_unicode{,_o200k}.h` (376) → generated data tables, per their own
  header comment. Mechanical regeneration, not translation.
- `tok.h` (446) → the strongest candidate for the `tokenizers` crate, which is
  needed regardless: R1, GLM-5.2 and M3 all ship `tokenizer.json`. K3's
  tiktoken `.model` is a base64-token-plus-rank text format and needs roughly
  a 50-line loader.

### Tests: 9 binaries, ~78 assertion sites

Not thousands. `test_ops.c` is 720 lines because it is a *harness* that loops
over the 14 JSON fixtures in `tests/fixtures/ops/` — and **the fixtures need
no porting at all**, because a Rust harness reads the same files. `cfg`
already demonstrates this: its tests run against the untouched `ref_k3.json`
and `cfg/*.json`.

Worth porting, in order: `test_ops.c` (gates ~40 kernels individually) and
`k3_model.c` (the full-model oracle, three gates) carry nearly all the value.
`test_st.c` and `test_cache.c` follow. `test_expert.c` and
`test_real_layer.c` need a real checkpoint and wait for the loader. The Rust
versions will be *smaller* than the C: much of `test_ops.c` is manual `json.h`
traversal that `serde_json` collapses into a derive.

### What crates replace, and what they must not touch

Rust is more compact than C in exactly the places C was doing work a standard
library would otherwise do, and not one line more compact in the kernels.

| module | C code | Rust est | what changes |
|---|---:|---:|---|
| `st` | 465 | **~180** | `str_`, `i64_`, `skip_value`, `scan_shard` plus an FNV-1a open-addressed index and a string pool — ~308 lines that exist only because C has no JSON parser and no hash map — become `serde_json` + `HashMap` |
| `tok` | 822 | **~120** | `tiktoken-rs` or `tokenizers`, gated by `tok_parity.py` |
| `bind` | 385 | ~280 | the tensor-name table becomes data |
| `cache` | 323 | ~250 | the pin/prefetch semantics do not fit the `lru` crate |
| `io` | 559 | ~400 | `rustix`; the ring buffer and reader thread stay hand-written |
| `cli` | 211 | ~150 | `clap` |
| `ops` | 985 | **~1,100** | **nothing.** Bit-exact transliteration; expect slightly *more* than the C from `unsafe` blocks |
| tests | — | ~450 | serde derives replace manual fixture traversal |
| **total** | | **~2,930** | |

Two dependencies to refuse on purpose:

- **`safetensors`.** It wants the file as a byte slice or an mmap, and
  `k3_st.h:22` refuses mmap deliberately: "Pages read into a buffer the engine
  owns never become file-backed mappings counted against the process, so peak
  RSS tracks what is actually resident rather than the whole 1.56 TB
  checkpoint." The 8.24 GB peak-RSS figure on the README depends on that
  choice. Parse the header with a `pread` and `serde_json`; keep
  `k3_st_read_aligned`'s O_DIRECT widening hand-written.
- **`half`** for bf16. The conversion is `(h as u32) << 16` — bf16 *is* the
  top 16 bits of an f32, with no rebias and no rounding (`k3_st.h:105`). A
  dependency for one shift is a dependency that can silently change rounding.

And one rule for the kernels: compaction that hides the reduction order is a
regression even when it compiles and passes. Generic-over-dtype matmuls are
tempting — the four `k3_matmul*` variants are visibly parallel — but they
accumulate in different types (`__m256d` for bf16, `__m256` for q8) with
different unroll factors, and the contract lives in exactly those details.

### Revised cost

|  | first estimate | filtered | with crates + one crate |
|---|---:|---:|---:|
| Rust LOC, Phase 1 | ~8,000 | ~4,700 | **~2,930** |
| Sessions, Phase 1 | 20–30 | 12–18 | **9–13** |
| Sessions, all phases | 40–60 | 28–40 | **22–32** |

The first estimate over-quoted by roughly 40%, by counting `main()`'s
reporting code and the fixture-driven test harnesses as engine logic. The
second still assumed a hand-written JSON scanner, hash index and tokenizer.
Note where the remaining time sits: `ops.rs` is a third of the total line
count and gets no help from any of this.

## Translation rules

Three rules carry the bit-exactness contract. All three are easy to violate
while making the code look nicer.

- **SIMD.** Transliterate; do not improve. In particular, keep the
  four-accumulator structure in `k3_matmul_bf16` (`k3_ops.c:1083`) and the
  `__m256d` accumulation. Accumulating in double costs roughly 2× throughput
  and was chosen deliberately — the comment at `:1053` reads "THE AVX2 PATH IS
  BIT-IDENTICAL TO THE SCALAR PATH, not merely close." Optimising it is a
  separate, later, opt-in decision.
- **Threading.** `#pragma omp parallel for schedule(static)` becomes `rayon`
  over the same chunk boundaries. Each output row is independent, so the
  reduction order is preserved — but verify that per kernel rather than
  assuming it.
- **I/O.** Keep the 2 MB-aligned `O_DIRECT` allocation and the
  `MADV_HUGEPAGE` hint (`k3_trunk.c:356-382`); the comment there explains that
  without them the pinning cost migrates into the `pread` loop and the
  benchmark reports a device rate that flatters the disk. Keep the `RING >= 2`
  invariant and its comment.

## Running it

```sh
make rust           # build
make rust-test      # the port's own suite, against tests/fixtures/
make rust-parity    # exhaustive libm gate, ~16 s
make rust-diff      # run C and Rust on the same fixtures and diff the output
```

`make test` (the C suite) is unchanged and must stay green throughout.

## What the port is not for

Worth stating plainly, because "rewrite it in Rust for speed" is the default
assumption and it is wrong here.

`docs/TUNING.md` says storage bandwidth is the ceiling, not the CPU, and that
a 6× difference in device bandwidth is a 6× difference in throughput — larger
than any tuning decision in the document. Of the five rungs on the memory
ladder, four are device-bound; only the 128 GB trunk-resident row (5.59
s/token) is compute-bound, and of that, 2.48 s/token is trunk matmul already at
its memory floor. Assembly reorders instructions; it does not make DRAM faster.

The measured headroom that *does* exist is elsewhere:

- **Expert cache policy.** `docs/data/expert-cache-capacity.txt` puts it best:
  at 64 GB, Belady reaches 61.74% where LRU manages 36.24%. That 25.5-point gap
  is available to a better policy at the *same* memory.
  [FlashMoE](https://arxiv.org/abs/2601.17063) closes most of a comparable gap
  with a lightweight recency+frequency predictor (+51% hit rate, 2.6× speedup).
  `tools/sim_cache.py` and `tests/fixtures/expert_trace.bin` make this testable
  offline at zero inference cost. Heed that file's own caveat: the trace
  re-prefills, so its hit rates are an upper bound.
- **Smaller models.** K3 moves ~135 GB/token. DeepSeek R1 activates 37B
  params, ~37 GB/token at native FP8 — and its full checkpoint is ~671 GB
  against K3's 1.56 TB, so on a 128–256 GB box the trunk goes fully resident
  and the streaming regime disappears entirely. There is a routing reason too:
  `expert-cache-capacity.txt` notes K3's router is trained with Quantile
  Balancing *specifically to flatten expert usage*, which is precisely what
  defeats an LRU cache. DeepSeek's is not flattened the same way.

A bare-metal kernel was considered and dropped. The I/O path already uses
`O_DIRECT`, 2 MB-aligned hugepage buffers, coalesced `pread`s and a
prefetching reader thread, so there is essentially no kernel overhead left to
bypass — a multi-megabyte read at 4.6 GB/s costs milliseconds of device time
against ~1–2 µs of syscall. The remaining bare-metal-only wins are worth low
single digits, against an operating system's worth of new code and the loss of
the portability the README advertises.
