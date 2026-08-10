# Roadmap

Ordered by value, with the reasoning stated so the order can be argued with.

**Two roadmaps, and this is the second one.** The active work is the Rust port and the
multi-model generalisation it exists for; that plan lives in
[RUST_PORT.md](RUST_PORT.md) and [MULTI_MODEL.md](MULTI_MODEL.md), and it is where new
work goes. In short:

| | |
|---|---|
| ~~finish the port~~ | **done.** All eight steps of the gate ladder; `make rust-golden` gates the whole stack against the C engine and the torch reference |
| ~~generalise~~ | **done.** `rust/src/arch.rs` reads the architecture from the checkpoint's own `config.json`; `rust/src/bin/run.rs` is one binary for all of them |
| more models | DeepSeek-V4-Flash: attention, Compressor, Indexer and experts verified against numpy on real shards; a full generation still needs the 166.9 GB release. GLM-5.1 and MiniMax-M3: descriptors and kernels exist and are gated, layer wiring unproven |

What is left on the port, in order: an end-to-end DeepSeek-V4 generation on the full
release; true incremental decode for V4, whose attention currently recomputes over the
window each step; and the GLM-5.1 and MiniMax-M3 layer wiring, verified per component
against a reference the way DeepSeek-V4's was.

Everything below is the **C engine**. It remains the reference implementation and the
differential oracle, and `make test` must stay green throughout the port; but these items
are not being worked on ahead of the list above.

## 1. Chunked prefill, the highest-value missing piece

Prefill currently runs as a single forward pass over the whole prompt, and attention in
the 24 MLA layers is quadratic in sequence length. The practical effect: the context
ceiling is 32k tokens, but a 21k-token prompt does not complete in reasonable time.

Raising the ceiling was necessary and is done. This is the part that makes it usable.

## 2. Re-run the published campaign under the replicating harness

The measured noise floor is 33%, and almost every published figure is a single sample.
The harness itself is done, `benchmarks/memory-ladder.sh` and `benchmarks/split-sweep.sh`
already default to 3 repeats and report mean, sd and spread. What remains is re-running
the 12 ladder rungs and the 12 splits under it and replacing the single-sample tables in
docs/data/ with replicated ones.

## 3. Thread scaling

`OMP_NUM_THREADS` has never been swept on this engine. The workload is I/O bound at low
memory budgets, so the useful thread count is probably well below the core count, and
on memory-bound workloads throughput often *declines* past a point. Unknown here.

## 4. SIMD in the KDA recurrence

The bf16 trunk matmul and the MXFP4 expert matmul already have hand-written AVX2 paths
(`src/core/k3_ops.c`), each written to reproduce the scalar reduction order exactly. The
KDA recurrence does not: it is still plain scalar C, and it is the largest remaining
un-vectorised kernel on the non-I/O path.

## 5. Sampling

Greedy only today. Adding temperature and top-p is small, but note the trade-off: greedy
decoding is what makes output identical across memory budgets, which is a property the
test-suite depends on. Sampling must be opt-in and off by default.

## 6. Chat template

K3 ships an XTML chat format. Without it the engine produces base-model continuations,
which is why asking a question gets the question completed rather than answered.

## 7. Vision

K3 is natively multimodal. The vision tower is 27 layers and ~0.4B parameters, small,
and its weights are ~0.9 GB, a fraction of a percent of the checkpoint. Self-contained
enough to be tractable.

## 8. Serving

No HTTP API. Deliberately last: it is product surface, and everything above changes what
would be served.

## Explicitly not planned

**A precision dial for the trunk.** The trunk is streamed losslessly rather than
quantised, and that is a design decision, not an omission. Post-hoc int4 measures ~17%
mean relative weight error on K3 attention tensors against ~1% for int8; streaming
costs time, which more memory buys back, while rounding costs accuracy, which nothing
buys back.

**GPU support.** Out of scope for this project.
