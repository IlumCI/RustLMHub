# Multi-model: what actually varies

This engine was built for one model. Making it serve others is not a matter of adding
configuration knobs — it is a matter of finding out which parts are architecture and
which are Kimi K3.

You cannot generalise from one example. So this document is written against **two**
architectures the port has now read real weights from: Kimi K3 (2.78T) and
DeepSeek-V4-Flash (304B). Everything below is measured from checkpoints and reference
implementations, not inferred.

## Where they differ, and where they do not

| | Kimi K3 | DeepSeek-V4-Flash | |
|---|---|---|---|
| routed-expert format | MXFP4 g32 + E8M0 | MXFP4 g32 + E8M0 | **same** |
| shared expert | `n_shared`, unweighted | 1, unweighted | **same** |
| router bias role | selection only | selection only (`noaux_tc`) | **same** |
| renorm + scale | yes, `routed_scale` | yes, `route_scale` 1.5 | **same** |
| dense / attention weights | bf16 | FP8 e4m3, 128×128 E8M0 blocks | differ |
| FFN activation | SiTU-GLU | SwiGLU, clamped at 10 | differ |
| router scoring | sigmoid | `sqrt(softplus(·))` | differ |
| routing source | scores | scores, **or token id** (layers 0–2) | differ |
| expert space | latent (down/up project) | full hidden width | differ |
| residual / combine | AttnRes block snapshots | Hyper-Connections, 4 copies | differ |
| attention | KDA \| Gated MLA | MLA + o-LoRA + sink + compressor + indexer | differ |
| RMSNorm accumulation | f64 | f32 | differ |

**The single most valuable finding is the first row.** DeepSeek-V4 stores its 256 routed
experts in exactly the layout K3 uses: E2M1 nibbles two per byte, one E8M0 scale per
group of 32. `inference/kernel.py` pins the constants — `block_size` 32,
`float8_e8m0fnu`, `fp4_max` 6.0. So `ops::matmul_mxfp4` and `ops::mxfp4_dequant`,
written for a 2.78T model, read a 304B one **unchanged**, and that covers 277B of
DeepSeek-V4's 304B parameters.

## Seams that exist now

Small, checked, and derived from the table above.

- **`ops::W`** — tagged weight matrix. `F32`, `Bf16`, `I8` (per-row draft), and now the
  DeepSeek-V4 formats via `matmul_fp8_block`.
- **`ops::Glu`** — `SiTu { b1, b2 }` and `SwigluClamped { limit }`.
- **`ops::Scoring`** — `Sigmoid`, `SqrtSoftplus`, `Softmax`, lifted out of the router.
- **`ops::router_scored`** — the K3 router with the score function as a parameter.
  A test asserts it reproduces `ops::router` **bitwise** on sigmoid, so generalising it
  cannot have changed a model that already passes its fixtures.
- **`ops::router_hashed`** — DeepSeek-V4's first `n_hash_layers` pick experts from a
  `tid2eid` table indexed by token id and use the scores only for weights.

## Traps, each now an executable test

- **SwiGLU's clamp is asymmetric.** `up` is bounded both sides; `gate` only from above
  (`inference/model.py`, `Expert.forward`). Clamping `gate`'s minimum too yields a
  bounded, plausible, wrong activation.
- **`softplus` goes linear above 20.** torch's default threshold; reproduce it or the
  scores drift on large logits.
- **Hash routing ignores the scores when selecting.** A scores-based fallback would
  route layers 0–2 of DeepSeek-V4 to entirely the wrong experts and still produce
  fluent-looking output.
- **MXFP4 nibble order is unverifiable statistically.** Low nibble is the even element
  in both engines. Reversing it changed 7,571,924 of 8,388,608 elements on a real
  DeepSeek expert while leaving every per-group statistic identical. Settling it needs a
  reference forward pass. Recorded rather than assumed.
- **FP8-block matmul is not bit-identical to the reference.** `inference/model.py`
  quantises the *activation* to fp8 before the gemm (`act_quant`, `activation_scheme:
  dynamic`) and accumulates in fp32. The port dequantises the weight and dots against
  the full-precision activation in f64 — strictly more accurate, and therefore
  different. Matching exactly means reproducing the activation quantisation too.

## The residual seam, now a trait

`ops::Residual` has `pre(sub, input) -> Carry` and `post(sub, output, carry)`, with an
associated `Carry` for whatever the two halves must pass between them. Two
implementations:

- **`AttnResidual`** — K3. Keeps the block-snapshot stack, the running prefix and the
  `have_prefix` flag. The boundary snapshot lives at the end of `pre(Attn)`, because the
  aggregate has to read the prefix *before* it is snapshotted and cleared.
  `Carry = ()`.
- **`HyperConnResidual`** — DeepSeek-V4. Keeps `hc_mult` parallel copies. `pre` flattens
  them, computes `mixes = hc_fn · flat * rsqrt`, splits it through
  `hc_split_sinkhorn` into pre/post/comb, and reduces the copies to one.
  `Carry = (post, comb)`.

`ops::decoder_layer` is now generic over `Residual` and takes neither `maxb` nor
`attn_res_block`. **All 14 K3 op fixtures still pass**, so generalising the residual did
not move K3's arithmetic.

Traps found while implementing Hyper-Connections, each an executable test:

- **`hc_post` sums over `comb`'s FIRST index**: `y[k] = post[k]·out + Σ_j comb[j][k]·res[j]`.
  Transposing it mixes the same copies with the same weights and is wrong.
- **`post` is `2·sigmoid(·)` where `pre` is `sigmoid(·) + eps`.** Dropping the factor of
  two halves every module contribution and leaves a stable, plausible stream. It cannot
  be caught by a value range: with the real `hc_attn_scale[1]` of 0.019 every sigmoid
  lands below 0.5, so `post` stays under 1 either way. The test compares the ratio.
- **Sinkhorn ends on a column normalisation**, so columns sum to 1 exactly and rows only
  approach it (measured 0.925–1.067 after 20 iterations). Asserting both would be wrong.
- **The `rsqrt` is over the flattened `hc*d` vector**, not per copy.

## The architecture descriptor

`rust/src/arch.rs` is the refactor `docs/RUST_PORT.md:19` names as the payoff: config and
tensor names become **data**, read from the checkpoint's own `config.json`, with the
family chosen by `model_type` and `architectures[0]`. `rust/src/bin/run.rs` is one binary
for all of them.

Four families are described. Three of the four `config.json` files are checked in verbatim
under `tests/fixtures/arch/` and parsed by `rust/tests/arch_configs.rs`, because a
descriptor is only worth anything if it reads the files the models actually ship — a
hand-written approximation agrees with itself and with nothing else.

| | Kimi K3 | DeepSeek-V4-Flash | GLM-5.1 | MiniMax-M3 |
|---|---|---|---|---|
| `model_type` | *(absent)* | `deepseek_v4` | `glm_moe_dsa` | `minimax_m3_vl` |
| hidden / layers | 7168 / 93 | 4096 / 43 | 6144 / 78 | 6144 / 60 |
| experts, top-k | 896, 16 | 256, 6 | 256, 8 | 128, 4 |
| scoring | sigmoid | `sqrt(softplus)` | sigmoid | sigmoid |
| activation | SiTU-GLU | SwiGLU clamp 10 | SwiGLU | `swigluoai` α1.702 lim 7 |
| dense layers | `first_dense` | *(none)* | `first_k_dense_replace` | `moe_layer_freq[]` |
| attention | KDA \| Gated MLA | MLA + compressor + indexer | MLA + DSA indexer | GQA 64/4 + block-sparse |
| RMSNorm accumulation | **f64** | f32 | f32 | f32 |
| norm form | plain | plain | plain | **gemma**, gain is `1 + w` |

Four things the descriptor had to be taught, each of which reads as a plausible config
otherwise:

- **Dense layers have three spellings and they are not interchangeable.** `first_dense`,
  `first_k_dense_replace` and a per-layer `moe_layer_freq` array. DeepSeek-V4 has no dense
  layer at all — its first three route by *token id* rather than by score, which is a
  routing change, not a dense one.
- **MiniMax-M3 spells the routed-expert width `intermediate_size`.** Everywhere else that
  key IS the dense MLP width, and M3 puts the dense width in `dense_intermediate_size`.
  The reader refused rather than defaulting, which is how this was found.
- **Every text field of M3 lives under `text_config`**, because it is multimodal. `cfg.rs`
  already solved the same nested-vs-flat problem for K3.
- **All three ship `num_nextn_predict_layers`.** Those blocks are not decoder layers, and
  counting them as such is precisely what made `compress_ratios` look three entries long.

The descriptor is checked against the measurements rather than only against itself:
`expert_bytes` reproduces this document's 13.37 MB per routed expert and 3.45 GB per token
from `config.json` alone.

## Seams that are still wrong

~~**RMSNorm accumulation width is a per-architecture property.**~~ **Fixed.** It is now
`Spec::rms_acc`, and `ops::rmsnorm_acc` takes it. `ops::rmsnorm` keeps the f64 form, which
is what the 14 K3 fixtures gate. A test asserts the two widths genuinely disagree in the
last bits on a realistic vector — if they agreed, the field would not be worth having.

`ops::rmsnorm_gemma` is the other norm form: MiniMax-M3 sets `use_gemma_norm`, where the
stored gain is an OFFSET and the multiplier is `1 + w`. On a checkpoint whose norm weights
sit near zero, using the plain form scales every channel to near zero — stable, and
wrong.

## What DeepSeek-V4 still needs

| piece | status |
|---|---|
| safetensors reader, `F8_E4M3` / `F8_E8M0` / `I8` / `I64` | done, verified on real shards |
| expert dequant and matmul | done, bit-exact on real bytes |
| FP8-block matmul | done, agrees with numpy on real attention weights |
| GLU, scoring, hash routing | done |
| Hyper-Connections residual | done, matches the reference on real hc weights |
| attention, `compress_ratio == 0` layers | done, verified against numpy on real layer-0 weights |
| Compressor, both window shapes | done, verified on real layer-2 and layer-3 weights |
| Indexer (top-k block selection) | done, verified on real layer-2 weights |
| compressed KV wired into attention | done, verified on real layer-2 weights |
| tokenizer (`tokenizer.json`) | done, round-trips CJK / ZWJ emoji / accents / code |
| generation CLI (`v4_run`) | done, prefill + greedy decode wired end to end |
| expert cache | done, byte-exact under eviction on real experts |
| full forward pass wiring | done; **never run on a complete checkpoint** |
| resident trunk | done — loaded once in stored FP8/bf16 form, ~8.3 GB |
| incremental decode | prefill-style: each step recomputes over the window |

## Attention

`compress_ratios` is `[0, 0, 4, 128, 4, 128, …, 4, 0, 0, 0]`. Layers with ratio **0** —
including layer 0, the one whose weights fit on this disk — are pure sliding-window
attention: no Compressor, no Indexer, and YaRN disabled in favour of the base
`rope_theta`. That path is implemented in `src/v4.rs` and verified.

**The array is 46 long and `num_hidden_layers` is 43, and that is the trap.** The three
trailing zeros are not the last three decoder layers. The release ships 48 shards: shard 1
is `embed`, shards 2–44 are layers 0–42, shard 45 is `norm` plus `head`, and shards 46–48
are three further layer-sized blocks past the decoder stack. The array indexes all 46, so
decoder layers 40, 41 and 42 take indices 40, 41, 42 — which are `4`, `128`, `4`.

Reading the ellipsis above as "and 0 for the last three" is what the port did, and it gave
those three layers a ratio of 0. Zero is a *valid* ratio meaning pure sliding window, so
there is no error and no shape mismatch: three of 43 layers quietly attend over a
128-token window instead of the whole compressed history, and the model stays fluent. The
table is now transcribed verbatim as `v4::COMPRESS_RATIOS` rather than recomputed from a
formula, and five tests in `rust/tests/deepseek_v4.rs` pin it — including one asserting
that layers 0 and 1 are the *only* zeros in the decoder stack.

`tools/verify_v4_attn.py` reparses the shard with numpy and `ml_dtypes` and follows
`inference/model.py`'s `Attention.forward`. On **real layer-0 weights**, T=6:

```
max |diff|                                        8.464e-06
max |diff| / max |out|                            1.353e-06
max relative error, 15,549 elements above 10% of scale   8.927e-06
```

That is f32 accumulation noise. A small synthetic fixture reruns the same code path in
CI without the 3.5 GB shard.

Five places to be plausibly wrong, each now a test:

- **Rope pairs adjacent elements**, because torch views the last axis as complex.
  Splitting the axis in half — the Llama convention — is a different rotation that still
  produces working attention.
- **q is RMS-scaled twice**: once through `q_norm` with a learned gain, then again per
  head after `wq_b` with no gain at all. Dropping the second still normalises.
- **`attn_sink` contributes to the denominator only.** It has no value row, so a head
  can attend to nothing. Giving it one is a plausible misreading of `kernel.py:346`;
  the test asserts a larger sink shrinks the output without changing its direction.
- **The output is de-rotated** by the same rope conjugated, before the o-projection.
- **The o-projection is grouped**: `o` is cut into `o_groups` slices and each sees only
  its own `[o_lora_rank][gsz]` block of `wo_a`. Treating `wo_a` as one dense matrix has
  the right shape and mixes the wrong heads.

### Compressor and Indexer

Both are implemented and verified on real weights — layer 2 (ratio 4, overlapping
windows, with an Indexer) and layer 3 (ratio 128, plain windows, no Indexer):

```
layer 2, T=12 -> 3 blocks
  compressor   max |diff| / max|out|   2.150e-06
  indexer kv   max |diff| / max|out|   2.083e-06   (Hadamard-rotated)
  indexer topk 12/12 tokens select the identical block order
layer 3, T=256 -> 2 blocks
  compressor   max |diff| / max|out|   1.023e-05
```

Six more places to be plausibly wrong, each a test:

- **The gate softmax runs over the SLOT axis, per channel.** It picks, for each
  dimension independently, which token in the block that dimension comes from — not a
  distribution over channels.
- **The APE is added to the gate, never to the value.**
- **At ratio 4 the windows overlap**: a block takes the upper half of its own tokens'
  dims and the lower half of the *previous* block's. Block 0 has no previous block, so
  its lower slots are masked with `-inf` rather than zero-weighted — a zero there would
  still be a valid probability and would dilute every channel.
- **A compressed block sits at rope position `b * ratio`,** not `b`.
- **The Indexer rotates both its query and its compressed KV** by a Hadamard transform.
  Rotating one side only leaves finite scores and a different ranking.
- **`relu` comes before the head weighting,** and the result is summed over heads.
- **A block is causal for token `t` only when `b < (t+1)/ratio`.** Off by one in either
  direction and a token either sees a block built partly from its own future, or misses
  the last block it should have.

The wiring is done. `attention_prefill` concatenates the compressed blocks after the
token window and attends over both; compressed block `b` lands at index `T + b`, which
is the offset the Indexer already emits. Verified end to end on real weights:

```
layer 0, ratio 0,  T=6    max relative error   8.927e-06
layer 2, ratio 4,  T=12   max relative error   2.018e-05   (31,458 elements above 10% of scale)
```

Index rows are padded with `-1` rather than repeated, and `sparse_attn_row` skips
negative slots. Padding with index 0 instead would silently give every short row extra
attention on the first token — a test covers exactly that difference.

**The FP4 simulation matters more here than anywhere else.** `inference/model.py` calls
`fp4_act_quant` on the Indexer's query and compressed KV; this port does not, so it is
more precise than the released kernel. Everywhere else that only shifts the last bits,
but the Indexer's output is a *discrete* top-k selection, so two nearly-equal scores can
be ordered differently by a quantisation the port never performs. The 12/12 agreement
above is between the port and a reference that makes the same omission.

## Running it

```sh
cargo run --release --bin v4_run -- \
    --model /path/to/DeepSeek-V4-Flash-0731 \
    --tokenizer /path/to/tokenizer.json \
    --prompt "The capital of France is" --n 16
```

The tokenizer is the `tokenizers` crate reading the released `tokenizer.json` — the same
format GLM-5.2 and MiniMax M3 ship, so it covers every target except K3, which keeps
`third_party/tok.h` until its tiktoken `.model` gets a loader.

`v4_run` **refuses an incomplete checkpoint** and names what is absent, rather than
running. A missing layer reads as zeros and the model still emits fluent, wrong text;
there is no error to notice at run time. On a single-shard directory it reports 465
missing tensors and stops.

The forward pass is wired: `generate()` prefills the prompt, runs all 43 layers through
`v4::layer_forward`, streams routed experts through `cache::Cache`, reduces the
Hyper-Connections copies and greedy-decodes. **It has never produced a token**, because no
machine in CI holds all 48 shards.

Two things stand between that and a usable first token, and neither is a correctness gap:

- **The trunk is re-read from disk every token.** `generate()` calls `raw()`/`f32s()` per
  tensor, per layer, inside the step loop, so all 8.29 GB moves off the device once per
  step instead of being loaded once and staying resident. This is what `k3_bind.c` solves
  for K3 and what the port has no equivalent of yet.
- **There is no incremental decode.** `v4.rs` implements `attention_prefill`,
  `compress_prefill` and `indexer_prefill` and nothing else, so each step rebuilds
  `HyperConnResidual` over the full sequence and recomputes attention from position zero.
  A first token is reachable; sixteen is quadratic on top of the re-read above.

Note what is *not* needed. K3 streams a 108.81 GB trunk because it cannot be resident;
DeepSeek-V4's is 8.29 GB and simply loads once. The trunk-streaming machinery in
`k3_trunk.c` has no DeepSeek-V4 equivalent to port.

## The expert cache

`src/cache.rs` holds routed experts in their packed MXFP4 form — dequantised, one
DeepSeek-V4 expert is 8× larger for no benefit, because `matmul_mxfp4` consumes the
packed bytes and a matrix-vector product is memory bound. LRU with pinning, and the
constructor refuses a cache smaller than one token's working set rather than trusting
the caller.

**DeepSeek-V4 experts are not contiguous, and that changes the read strategy.** K3 packs
an expert's six tensors as one run. DeepSeek groups all three *scales* together and all
three *weights* together, about 340 MB apart in the shard:

```
w1.scale    off  32,992,856  len    262,144   |
w2.scale    off  33,255,000  len    262,144   |  786 KB run
w3.scale    off  33,517,144  len    262,144   |
w1.weight   off 374,830,168  len  4,194,304   |
w2.weight   off 379,024,472  len  4,194,304   |  12.6 MB run
w3.weight   off 383,218,776  len  4,194,304   |
```

So an expert is 13.37 MB spread over a 354 MB span. Reading the span would fetch 26×
what it uses; reading six tensors separately would be six seeks. `locate` coalesces into
maximal gapless runs — one for K3, two for DeepSeek-V4 — and the cache issues one
aligned read per run.

Measured on the real layer-2 shard, 256 experts, a 400 MB cache (29 slots):

| stream | hits | bytes read | evictions | byte check |
|---|---:|---:|---:|---|
| sweep, deliberately cache-hostile | 0.00% | 8.02 GB | 571 | 86/86 identical |
| skewed routing (80% from a tenth) | **78.17%** | 1.75 GB | 102 | 86/86 identical |

Every sampled expert is compared against an independent direct read of the same tensors,
so eviction cannot quietly hand back a stale slot.

**O_DIRECT needs the buffer address page-aligned**, not just the offset and length. A
plain `Vec<u8>` is not, and every `pread` against one fails with `EINVAL` — which
surfaces as a short read, not an error. `st::Aligned` allocates the arena through
`Layout::from_size_align`; the C engine uses `posix_memalign` for the same reason.

Note what is *not* needed here. K3 streams a 108.81 GB trunk because it cannot be
resident; DeepSeek-V4's is 8.29 GB and simply loads. The trunk-streaming machinery in
`k3_trunk.c` has no DeepSeek-V4 equivalent to port.

## Hardware

DeepSeek-V4-Flash suits a small machine far better than K3 does, and the reason is the
trunk. Measured from the layer-0 shard header and multiplied out over 43 layers:

| | Kimi K3 | DeepSeek-V4-Flash |
|---|---:|---:|
| trunk, must be resident or streamed | 108.81 GB | **8.29 GB** |
| routed experts, streamed | 1.45 TB | 147.2 GB |
| bytes read per token | ~135 GB | **3.45 GB** |
| checkpoint on disk | 1.56 TB | 166.9 GB |

**The trunk fits in RAM.** That is the whole difference. K3 only reaches its
trunk-resident regime at 128 GB, where it runs 5.59 s/token; below that it streams
108 GB every step and takes 26.5 s/token. DeepSeek-V4's trunk is 8.29 GB — 6.17 GB of
per-layer weights plus roughly 1 GB each for `embed` and the head — so a 16 GB machine
starts in the fast regime rather than climbing toward it.

Per-token traffic then drops by a factor of **39**: 3.45 GB against K3's ~135 GB. One
routed expert is 13.37 MB across `w1`/`w2`/`w3` including scales, and a token activates
6 of 256 in each of 43 layers.

### A 16 GB / 1 TB target

```
resident   trunk + embed + head                 8.3 GB
            engine, activations, KV cache       ~1 GB
            expert cache                       ~5 GB   (~370 of 11,008 experts)
streamed   routed experts                    147.2 GB on disk, 3.45 GB per token
```

The KV cache is not a concern here: compressed KV plus a 128-token window is about
1.2 MB per layer, ~51 MB across the model at 4k context. That is what "Flash" buys.

Expert caching should also work better than it does on K3.
`docs/data/expert-cache-capacity.txt` notes K3's router is trained with Quantile
Balancing *specifically to flatten expert usage*, which is what defeats LRU. DeepSeek-V4
uses `noaux_tc` with a selection bias and routes the first three layers by token id, so
usage is not deliberately flattened.

At 3.45 GB/token, a drive doing 2 GB/s gives roughly 1.7 s/token and one doing 3.5 GB/s
about 1.0 s/token, before any cache hits. Both are estimates from byte counts, not
measurements — `./scripts/k3-doctor.sh` measures the actual device, and the number it
reports is the one that matters.

**The 4 GB of VRAM has no use here.** This engine is CPU-only by design, and the
reference implementation needs `tilelang` and `fast_hadamard_transform` on a GPU with
far more memory than that. Streaming from NVMe on the CPU is the path that fits.

### Verifying at small scale

Shard *N+1* holds layer *N*, at ~3.5 GB each, so a single download is enough to check a
layer end to end. That is how everything in this document was verified.

## Kernels the third and fourth architectures need

GLM-5.1 reuses more than it adds: MLA plus a DSA indexer, sigmoid scoring and bf16
weights, so `v4::indexer_prefill`, `ops::router_scored` and `ops::matmul_bf16` apply
unchanged. Its one new trap is rope pairing.

MiniMax-M3 needs genuinely new kernels, in `rust/src/gqa.rs` and `ops.rs`. Each is a place
where the wrong choice produces working attention rather than an error, so each is a test:

- **Grouped-query attention.** Query head `h` reads kv head `h / (n_heads / n_kv_heads)`.
  Writing `h % n_kv_heads` keeps every shape identical and pairs every head with the wrong
  group. M3 is 64 query heads over 4 kv heads; there is no GQA anywhere else in the tree.
- **Rope pairing.** `Adjacent` treats the last axis as complex — (0,1), (2,3), … —
  and `Halves` pairs `i` with `i + dim/2`. Both are norm-preserving rotations and both
  produce working attention. GLM-5.1 sets `rope_interleave: true` *and*
  `indexer_rope_interleave: true`.
- **Partial rotary.** M3 sets `partial_rotary_factor: 0.5` with `rotary_dim: 64` over a
  128-wide head, so half of every head is deliberately left unrotated. Rotating all of it
  is a different position encoding, not a broken one.
- **`swigluoai`.** The gate's sigmoid argument is scaled by `alpha` (1.702) and the up
  branch carries a `+1` bias. Dropping either leaves a bounded, plausible activation; the
  test pins `up = -1` where the bias makes the product vanish exactly.
- **Gemma norm.** `use_gemma_norm` means the stored gain is an OFFSET, so the multiplier
  is `1 + w`. On weights near zero the plain form scales every channel to near zero:
  stable, and wrong.
- **Block-sparse rows.** Keep the first `init` blocks, the last `local` blocks, and the
  `topk` best of the rest. A block is eligible only when it starts at or before the token;
  including the block the token sits in is right — causality masks its tail — and
  including any later block lets a token read its own future.

Each of these was mutation-tested: the kernel was broken on purpose and the test watched
to fail. That step matters here more than usual, because two earlier thread-independence
tests passed on a deliberately broken kernel when the synthetic data happened to alias
with the chunk boundaries.

**What is not done for these two.** The descriptor parses their released configs and the
kernels exist and are gated, but neither has been run against real weights: GLM-5.1 and
MiniMax-M3 are large downloads and nothing here has read a byte of either. `run` says so
rather than pretending — it validates the checkpoint, then reports that the layer wiring
is absent. On the evidence of DeepSeek-V4, where the expert name table and
`compress_ratios` were both wrong in ways no fixture caught, that wiring should be assumed
wrong until a reference forward pass says otherwise.

## Speculative expert prefetch: measured, and it does not help here

`SpecPrefetch` (arXiv 2607.24787) predicts layer *L+1*'s experts from layer *L*'s router
input and prefetches them asynchronously, reporting 8–20% throughput on mobile NVMe. The
port implements the training-free form of that idea — run layer L+1's own gate on layer
L's residual, which is exact apart from using the previous layer's hidden state, and costs
~1M MACs against the 151M the experts cost. It is off by default. `K3_PREFETCH=1` enables it.

Measured on DeepSeek-V4-Flash, 5 GB expert cache, model on a 604 MB/s USB SSD:

| | off | on, evicting | on, non-evicting |
|---|---:|---:|---:|
| s/token | **7.0** | 11.2 | 8.3 |
| GB read per token | 3.91 | 6.80 | **3.57** |
| true resident hit rate | 19.1% | 8.4% | **21.1%** |
| evictions | 1381 | 3709 | 1764 |

Two findings, in order of how much they cost to learn.

**Speculation must never evict.** The first version let a prefetched expert take any slot.
Eight candidates across 43 layers is 344 speculative reads per token against 258 real
ones, so speculation alone churned a 373-slot cache every step and threw out experts the
*current* layer still needed: hit rate fell by more than half and I/O rose 74%. Admitting
prefetched blocks only into free slots — 2Q's admission rule (Johnson & Shasha, 1994), in
its simplest form — fixed that completely and beat the baseline on both hit rate and bytes.

**And it was still slower, because prefetching hides latency and this workload is
bandwidth-bound.** At 604 MB/s with ~3.6 GB of experts per token, the device is busy
roughly 6 s of every 7 s step. There is no idle I/O window to overlap into, so the reader
thread only competes with the real reads for a saturated device, and every mispredicted
candidate is pure waste. Better hit rate and fewer bytes did not translate into less time.

This is an Amdahl argument that should have been run before writing the code: overlap can
only recover the *smaller* of the two costs, and here compute is ~1.5 s against ~6 s of
I/O. The prefetch path is kept because it should win on a device with headroom — the same
model on a 3 GB/s NVMe would be compute-bound, and then the ordering reverses.

**What this says about where the time goes.** Expert bytes per token are set by the
architecture (top-6 of 256, 13.37 MB each, 43 layers = 3.45 GB). On this hardware the only
levers that move the total are reading *fewer* bytes — a higher hit rate, which needs RAM
the machine does not have — or reading them *faster*, which needs different storage.
Neither is a software problem.

## Where the time actually goes

Measured on DeepSeek-V4-Flash, 3 GB expert cache, model on a 604 MB/s USB SSD:

```
bytes read            29.59 GB / 6 tokens  =  4.93 GB per token
4.93 GB / 604 MB/s                         =  8.2 s of I/O
measured                                      8.8 s per token
```

**I/O is ~92% of a token.** That number is what decides which optimisations are worth
building, and it is why a 1.68x speedup of the expert kernel moved the end-to-end figure
by only ~8%: Amdahl caps kernel work at the ~8% of the token that is compute.

The corollary is that the only levers which move the total are the ones that change *bytes
per token* -- a higher cache hit rate, or fewer expert reads per generated token -- or the
rate at which those bytes arrive. Optimising CPU utilisation is actively misleading here;
650% of 1600% can mean six cores working and ten stalled on a saturated device.
