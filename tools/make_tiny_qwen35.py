#!/usr/bin/env python3
"""Write a tiny, all-F32 qwen35moe GGUF so our forward pass can be diffed against llama.cpp.

WHY THIS EXISTS
    The 35B checkpoint produces fluent-looking garbage, which means a SEMANTIC bug: right
    shapes, right magnitudes, wrong function. Bisecting that by eye across 40 blocks and
    21 GB has not converged. A few-megabyte model with the same architecture can be run
    under `llama-eval-callback`, which dumps every intermediate tensor, so the first
    divergence can be located exactly.

WHY F32 AND NOT Q4_K
    This isolates the ARCHITECTURE from the quantisation. The k-quant kernels are already
    verified bit-identical against llama.cpp's reference on real weights; the block wiring
    is not. Mixing both means a mismatch has two possible causes.

VERIFIED REFERENCE VALUES (regenerate with the commands below and compare)

    ratio 2 -- Qwen3.6-35B's shape (16 key heads, 32 value heads)
        python3 tools/make_tiny_qwen35.py /tmp/tiny/m.gguf
        llama-eval-callback -m /tmp/tiny/m.gguf -p "ABC" -n 1 -ngl 0   result_output -0.730464
        Q35_SUMS=1 q35chk /tmp/tiny 1 0.05 "ABC"                       ours          -0.730195

    ratio 4 -- Qwen3.5-122B's shape (16 key heads, 64 value heads)
        python3 tools/make_tiny_qwen35.py /tmp/tiny4/m.gguf --ratio 4
        llama-eval-callback -m /tmp/tiny4/m.gguf -p "ABC" -n 1 -ngl 0  result_output +1.012012
        Q35_SUMS=1 q35chk /tmp/tiny4 1 0.05 "ABC"                      ours          +1.011608

    Both matter. The value-head -> key-head mapping is `hv % n_k_heads` because
    ggml_repeat TILES, and at ratio 2 that is [0,1,0,1] while the plausible wrong answer
    `hv / group` is [0,0,1,1] -- they differ, so ratio 2 catches it. But a THIRD mapping
    could agree with modulo at ratio 2 and diverge at 4, which is why the 122B's geometry
    needed its own fixture rather than an argument from the 35B's.

WHY IT IS SAFE TO RUN llama.cpp ON THIS
    It is ~10 MB. The rule against running external runtimes here is about the multi-
    gigabyte checkpoints, which mmap past physical memory and thrash the machine.
"""
import struct, sys, hashlib

# GGUF value type tags.
U32, F32T, STR, ARR, U64 = 4, 6, 8, 9, 10
GGML_F32 = 0
ALIGN = 32

# Small on purpose, but every ratio that matters is preserved:
#   * 4 blocks with full_attention_interval 4 -> block 3 is full attention, 0/1/2 linear.
#   * n_v_heads = 2 * n_k_heads, so the value-head -> key-head mapping is exercised.
#   * head_dim > n_rot, so partial rope is exercised.
#   * top-2 of 4 experts, so routing and renormalisation are exercised.
#
# ON d_state == head_v_dim, WHICH LOOKS LIKE A GAP AND IS NOT
#     Both are 16 here, and both are 128 in the real 35B, so the per-head state matrix is
#     always SQUARE -- which would seem to let a transposed state escape detection.
#
#     It cannot be tested by making them differ: llama.cpp derives the value width from
#     d_state, not from ssm.inner_size, and refuses to load a model where they disagree
#     ("tensor 'blk.0.attn_qkv.weight' has wrong shape; expected 256,128, got 256,192").
#     head_v_dim == d_state is a constraint of the architecture, not an accident.
#
#     A transpose is nonetheless excluded by the end-to-end diff, because q, k and v play
#     DIFFERENT roles regardless of matching dimensions: the true readout is
#     o[j] = beta * (q.k) * v[j], while a transposed state gives k[j] * beta * (q.v).
#     Those differ for generic inputs, so the verified match already rules it out.
HP = dict(
    block_count=4, hidden=256, vocab=64, full_attention_interval=4,
    n_heads=2, n_kv_heads=1, head_dim=64, n_rot=16, rope_base=10000000.0, eps=1e-6,
    d_state=16, n_k_heads=2, n_v_heads=4, d_inner=64, conv_kernel=4,
    n_experts=4, topk=2, moe_inter=32, shared_inter=32,
)
# Optional second geometry: `--ratio 4` widens the value heads so each key head serves
# FOUR of them, which is Qwen3.5-122B's shape (16 key heads, 64 value heads). The ratio-2
# fixture cannot distinguish the tiling from any other mapping that happens to agree at
# ratio 2, and the 122B is advertised as runnable on the strength of a ratio-2 diff.
#
# head_v_dim must equal d_state (llama.cpp derives the value width from d_state and refuses
# a model where they disagree), so widening the ratio means widening d_inner in step.
if "--ratio" in sys.argv:
    r = int(sys.argv[sys.argv.index("--ratio") + 1])
    HP["n_v_heads"] = HP["n_k_heads"] * r
    HP["d_inner"] = HP["n_v_heads"] * HP["d_state"]

HP["head_v_dim"] = HP["d_inner"] // HP["n_v_heads"]
HP["qkv_width"] = 2 * HP["n_k_heads"] * HP["d_state"] + HP["d_inner"]
# Must sum to n_rot/2 pairs, mirroring the real model's [11, 11, 10, 0].
HP["rope_sections"] = [3, 3, 2, 0]


class Rng:
    """xorshift64*, so the file is reproducible without depending on numpy's version."""
    def __init__(self, seed): self.s = seed or 88172645463325252
    def u64(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = x
        return (x * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF
    def normal(self, n, scale):
        # Sum of 4 uniforms, centred: enough for weights that only need to be well-scaled.
        out = []
        for _ in range(n):
            s = sum(((self.u64() >> 11) / (1 << 53)) for _ in range(4)) - 2.0
            out.append(s * scale)
        return out


def s_(v):        return struct.pack("<Q", len(v)) + v.encode()
def kv(k, t, b):  return s_(k) + struct.pack("<I", t) + b
def kv_u32(k, v): return kv(k, U32, struct.pack("<I", v))
def kv_f32(k, v): return kv(k, F32T, struct.pack("<f", v))
def kv_str(k, v): return kv(k, STR, s_(v))
def kv_arr_str(k, vs):
    return kv(k, ARR, struct.pack("<IQ", STR, len(vs)) + b"".join(s_(v) for v in vs))
def kv_arr_i32(k, vs):
    return kv(k, ARR, struct.pack("<IQ", 5, len(vs)) + b"".join(struct.pack("<i", v) for v in vs))


def main(path):
    rng = Rng(20260812)
    H, V = HP["hidden"], HP["vocab"]
    tensors = []   # (name, ne_fastest_first, data_bytes)

    def add(name, ne, count, scale=0.02, ones=False):
        vals = [1.0] * count if ones else rng.normal(count, scale)
        tensors.append((name, ne, struct.pack("<%df" % count, *vals)))

    add("token_embd.weight", [H, V], H * V)
    add("output_norm.weight", [H], H, ones=True)
    add("output.weight", [H, V], H * V)

    for l in range(HP["block_count"]):
        p = f"blk.{l}."
        add(p + "attn_norm.weight", [H], H, ones=True)
        add(p + "post_attention_norm.weight", [H], H, ones=True)
        if (l + 1) % HP["full_attention_interval"] == 0:
            nq = 2 * HP["n_heads"] * HP["head_dim"]      # query AND gate, interleaved
            nkv = HP["n_kv_heads"] * HP["head_dim"]
            add(p + "attn_q.weight", [H, nq], H * nq)
            add(p + "attn_k.weight", [H, nkv], H * nkv)
            add(p + "attn_v.weight", [H, nkv], H * nkv)
            add(p + "attn_q_norm.weight", [HP["head_dim"]], HP["head_dim"], ones=True)
            add(p + "attn_k_norm.weight", [HP["head_dim"]], HP["head_dim"], ones=True)
            no = HP["n_heads"] * HP["head_dim"]
            add(p + "attn_output.weight", [no, H], no * H)
        else:
            W, DI, NV = HP["qkv_width"], HP["d_inner"], HP["n_v_heads"]
            add(p + "attn_qkv.weight", [H, W], H * W)
            add(p + "attn_gate.weight", [H, DI], H * DI)
            add(p + "ssm_alpha.weight", [H, NV], H * NV)
            add(p + "ssm_beta.weight", [H, NV], H * NV)
            # ssm_a is stored ALREADY NEGATIVE: decay = exp(softplus(..) * a) must land in
            # (0, 1]. Positive values here would make the recurrence diverge.
            tensors.append((p + "ssm_a", [NV],
                            struct.pack("<%df" % NV, *[-abs(v) - 0.5 for v in rng.normal(NV, 1.0)])))
            add(p + "ssm_dt.bias", [NV], NV, scale=0.1)
            add(p + "ssm_conv1d.weight", [HP["conv_kernel"], W], HP["conv_kernel"] * W, scale=0.3)
            add(p + "ssm_norm.weight", [HP["head_v_dim"]], HP["head_v_dim"], ones=True)
            add(p + "ssm_out.weight", [DI, H], DI * H)

        E, MI, SI = HP["n_experts"], HP["moe_inter"], HP["shared_inter"]
        add(p + "ffn_gate_inp.weight", [H, E], H * E)
        add(p + "ffn_gate_inp_shexp.weight", [H], H)
        add(p + "ffn_gate_exps.weight", [H, MI, E], H * MI * E)
        add(p + "ffn_up_exps.weight", [H, MI, E], H * MI * E)
        add(p + "ffn_down_exps.weight", [MI, H, E], MI * H * E)
        add(p + "ffn_gate_shexp.weight", [H, SI], H * SI)
        add(p + "ffn_up_shexp.weight", [H, SI], H * SI)
        add(p + "ffn_down_shexp.weight", [SI, H], SI * H)

    # A byte-level vocabulary: token i is the single character chr(i+33). Trivially
    # invertible, so a prompt maps to a known id sequence with no BPE ambiguity at all.
    toks = [chr(33 + i) for i in range(V - 3)] + ["<|s|>", "<|e|>", "<|p|>"]
    kvs = [
        kv_str("general.architecture", "qwen35moe"),
        kv_str("general.name", "tiny-qwen35moe-fixture"),
        kv_u32("general.file_type", 0),
        kv_u32("qwen35moe.block_count", HP["block_count"]),
        kv_u32("qwen35moe.embedding_length", H),
        kv_u32("qwen35moe.context_length", 512),
        kv_u32("qwen35moe.full_attention_interval", HP["full_attention_interval"]),
        kv_u32("qwen35moe.attention.head_count", HP["n_heads"]),
        kv_u32("qwen35moe.attention.head_count_kv", HP["n_kv_heads"]),
        kv_u32("qwen35moe.attention.key_length", HP["head_dim"]),
        kv_u32("qwen35moe.attention.value_length", HP["head_dim"]),
        kv_f32("qwen35moe.attention.layer_norm_rms_epsilon", HP["eps"]),
        kv_u32("qwen35moe.rope.dimension_count", HP["n_rot"]),
        kv_f32("qwen35moe.rope.freq_base", HP["rope_base"]),
        # MANDATORY: llama.cpp refuses to load without it. Four sections partitioning the
        # n_rot/2 rotary PAIRS among the position components (time, height, width, unused).
        # The real 35B declares [11, 11, 10, 0], summing to 32 = 64/2. For text-only input
        # all components carry the same position, so each pair still rotates by
        # pos * inv_freq[i] and multi-section rope reduces exactly to ordinary rope.
        kv_arr_i32("qwen35moe.rope.dimension_sections", HP["rope_sections"]),
        kv_u32("qwen35moe.ssm.conv_kernel", HP["conv_kernel"]),
        kv_u32("qwen35moe.ssm.group_count", HP["n_k_heads"]),
        kv_u32("qwen35moe.ssm.inner_size", HP["d_inner"]),
        kv_u32("qwen35moe.ssm.state_size", HP["d_state"]),
        kv_u32("qwen35moe.ssm.time_step_rank", HP["n_v_heads"]),
        kv_u32("qwen35moe.expert_count", HP["n_experts"]),
        kv_u32("qwen35moe.expert_used_count", HP["topk"]),
        kv_u32("qwen35moe.expert_feed_forward_length", HP["moe_inter"]),
        kv_u32("qwen35moe.expert_shared_feed_forward_length", HP["shared_inter"]),
        kv_str("tokenizer.ggml.model", "gpt2"),
        kv_str("tokenizer.ggml.pre", "qwen35"),
        kv_arr_str("tokenizer.ggml.tokens", toks),
        kv_arr_i32("tokenizer.ggml.token_type", [1] * (V - 3) + [3, 3, 3]),
        kv_arr_str("tokenizer.ggml.merges", []),
        kv_u32("tokenizer.ggml.bos_token_id", V - 3),
        kv_u32("tokenizer.ggml.eos_token_id", V - 2),
        kv_u32("general.alignment", ALIGN),
    ]
    # Counted, never asserted. A hardcoded KV count that drifts by one desynchronises the
    # whole file: the reader consumes the wrong number of pairs and then reads tensor info
    # out of the middle of a string, reporting nonsense like "9 dimensions".
    meta, n_kv = b"".join(kvs), len(kvs)
    info, off = b"", 0
    for name, ne, data in tensors:
        info += s_(name) + struct.pack("<I", len(ne))
        for d in ne:
            info += struct.pack("<Q", d)
        info += struct.pack("<IQ", GGML_F32, off)
        off += len(data)
        off = (off + ALIGN - 1) // ALIGN * ALIGN

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), n_kv) + meta + info
    pad = (-len(head)) % ALIGN
    out = bytearray(head + b"\0" * pad)
    for _, _, data in tensors:
        out += data
        out += b"\0" * ((-len(data)) % ALIGN)

    with open(path, "wb") as f:
        f.write(out)
    print(f"{path}: {len(out)/1e6:.2f} MB, {len(tensors)} tensors, sha256 "
          f"{hashlib.sha256(out).hexdigest()[:16]}")
    print("hparams:", " ".join(f"{k}={v}" for k, v in HP.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny-qwen35moe.gguf")
