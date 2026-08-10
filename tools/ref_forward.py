#!/usr/bin/env python
"""
ref_forward.py - a full 93-layer reference forward pass over the REAL Kimi K3 checkpoint,
in torch and numpy, producing GOLDEN LOGITS.

WHY THIS IS THE TEST THAT MATTERS
    Everything else in this project validates a piece:
      - the op fixtures validate each kernel against torch, on tiny synthetic weights;
      - the full-model oracle validates COMPOSITION end to end, but on a 13-layer toy;
      - tools/conform_all.py validates every one of the 93 real layers against torch on
        REAL weights, but each layer in ISOLATION, fed a fresh random input.

    None of those proves the thing a user actually cares about: that stacking the real 93
    layers produces the right logits. The residual stream, the AttnRes aggregator that
    spans layers and resets every attn_res_block, the embedding, the final norm and the
    lm_head are all untested at real scale, and an error in any of them is invisible to a
    per-layer check because each layer is still individually correct.

    This script closes that gap. It runs the whole model the way the reference defines it
    (k3_ref.K3Model.forward, mirrored statement for statement below), from the same shard
    bytes the C engine reads, and writes the final-position logits. Comparing those to the
    C engine's logits ELEMENTWISE -- not argmax -- is the definitive conformance test.
    Argmax alone is not sufficient: it hides near-ties, so two engines can agree on every
    token while their logit vectors differ substantially. Elementwise comparison is what
    turns "picks the same words" into "computes the same function".

HOW IT FITS IN 64 GB
    The always-active weights are 113.49 GB at bf16 and would be ~227 GB in fp32, so
    nothing is held resident. Each layer's weights are loaded, used and freed before the
    next layer starts, and the MoE
    dequantises only the experts the router actually selected. Peak memory is one layer.

usage: ref_forward.py <shard_dir> <ids> [out.json]
       ids is a comma-separated list of token ids, e.g. 3,4,5,6,7
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from k3_ref import (K3Config, GatedMLA, KimiDeltaAttention,       # noqa: E402
                    apply_attn_res, situ_glu)
from verify_real_layer import Shards, dequant                              # noqa: E402

PRE = "language_model.model."

# ---------------------------------------------------------------- int8 simulation ----
# Set by --int8. When on, every LARGE matrix is round-tripped through int8 with one fp32
# scale per 32 elements along the contraction dimension, exactly the scheme a C kernel
# would use, before it is multiplied. Nothing else changes, so the resulting logits show
# what an int8 trunk would actually produce end to end.
#
# WHAT IS DELIBERATELY NOT QUANTISED, and why:
#   routers (gate.weight, e_score_correction_bias) - the router's output SELECTS experts.
#     A 0.5% perturbation near a top-16 boundary does not shift the answer slightly, it
#     swaps in a different expert, which is a discrete change of computation. Routers are
#     ~1.2 GB across 92 layers, so keeping them exact is nearly free.
#   every *norm* weight, A_log, dt_bias, conv1d, *_res_proj - all tiny vectors that are
#     read elementwise rather than through a dot product, so quantising them buys nothing
#     and costs precision where it is most visible.
#   embed and lm_head - left exact. They sit at the two ends of the network and perturb
#     the logits most directly, so quantising them would confound the very measurement
#     this flag exists to make.
INT8 = False
GROUP = 32


def q8(w: np.ndarray) -> np.ndarray:
    """Round-trip a matrix through int8 with one fp32 scale per GROUP elements along the
    LAST axis, which is the contraction dimension for every matmul here."""
    if not INT8 or w.ndim != 2 or w.shape[-1] < GROUP:
        return w
    r, c = w.shape
    g = c // GROUP
    body, tail = w[:, : g * GROUP], w[:, g * GROUP:]
    b = body.reshape(r, g, GROUP).astype(np.float32)
    scale = np.abs(b).max(axis=2, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    qb = np.rint(b / scale).clip(-127, 127)
    out = (qb * scale).reshape(r, g * GROUP)
    if tail.size:
        out = np.concatenate([out, tail], axis=1)
    return out.astype(np.float32)


def rms(x: torch.Tensor, w: np.ndarray, eps: float) -> torch.Tensor:
    wt = torch.from_numpy(w.astype(np.float32))
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * wt


def load_attention(sh: Shards, c: K3Config, L: int, is_mla: bool):
    """Same tensor names and the same A_log truncation as tools/verify_real_layer.py."""
    lp = PRE + "layers.%d." % L
    sp = lp + "self_attn."
    # Big projections go through q8. It is a no-op for anything that is not a 2-D matrix
    # of at least GROUP columns, so the 1-D norms, A_log and dt_bias below pass through
    # untouched without needing a special case.
    g = lambda n: torch.from_numpy(q8(sh.get(sp + n).astype(np.float32)))   # noqa: E731
    if is_mla:
        m = GatedMLA(c)
        m.q_a_proj.weight.copy_(g("q_a_proj.weight"))
        m.q_a_layernorm.weight.copy_(g("q_a_layernorm.weight"))
        m.q_b_proj.weight.copy_(g("q_b_proj.weight"))
        m.kv_a_proj_with_mqa.weight.copy_(g("kv_a_proj_with_mqa.weight"))
        m.kv_a_layernorm.weight.copy_(g("kv_a_layernorm.weight"))
        m.kv_b_proj.weight.copy_(g("kv_b_proj.weight"))
        m.o_proj.weight.copy_(g("o_proj.weight"))
        if m.g_proj is not None:
            m.g_proj.weight.copy_(g("g_proj.weight"))
        return m
    m = KimiDeltaAttention(c)
    for n in ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj",
              "f_a_proj", "f_b_proj", "b_proj"):
        getattr(m, n).weight.copy_(g(n + ".weight"))
    for nm, mod in (("q", m.q_conv1d), ("k", m.k_conv1d), ("v", m.v_conv1d)):
        w = sh.get(sp + "%s_conv1d.weight" % nm).astype(np.float32)
        mod.weight.copy_(torch.from_numpy(w).reshape(mod.weight.shape))
    alog = sh.get(sp + "A_log").astype(np.float32)          # ships 128, PER HEAD
    m.A_log.copy_(torch.from_numpy(alog[: c.kda_num_heads]))
    m.dt_bias.copy_(torch.from_numpy(sh.get(sp + "dt_bias").astype(np.float32)))
    m.o_norm.weight.copy_(torch.from_numpy(sh.get(sp + "o_norm.weight").astype(np.float32)))
    return m


def mlp_forward(sh: Shards, c: K3Config, L: int, hn: torch.Tensor) -> torch.Tensor:
    """Dense MLP for layer 0, latent MoE for the rest. Mirrors verify_real_layer.py."""
    lp = PRE + "layers.%d." % L
    x = hn.numpy().astype(np.float32)                                   # [T, H]
    T = x.shape[0]

    if L < c.first_k_dense_replace:
        dp = lp + "mlp."
        dg = q8(sh.get(dp + "gate_proj.weight").astype(np.float32))
        du = q8(sh.get(dp + "up_proj.weight").astype(np.float32))
        dd = q8(sh.get(dp + "down_proj.weight").astype(np.float32))
        a = situ_glu(torch.from_numpy(np.concatenate([x @ dg.T, x @ du.T], axis=1)),
                     c.situ_beta, c.situ_linear_beta).numpy()
        return torch.from_numpy(a @ dd.T)

    mp = lp + "block_sparse_moe."
    gate_w = sh.get(mp + "gate.weight").astype(np.float32)
    bias = sh.get(mp + "gate.e_score_correction_bias").astype(np.float32)
    logits = x @ gate_w.T
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    biased = scores + bias                       # bias steers SELECTION only
    idx = np.argsort(-biased, axis=1)[:, : c.num_experts_per_token]
    w = np.take_along_axis(scores, idx, axis=1)
    w = w / (w.sum(1, keepdims=True) + 1e-20)

    down = q8(sh.get(mp + "routed_expert_down_proj.weight").astype(np.float32))
    up = q8(sh.get(mp + "routed_expert_up_proj.weight").astype(np.float32))
    lnorm = sh.get(mp + "routed_expert_norm.weight").astype(np.float32)
    z = x @ down.T                                                      # [T, latent]

    acc = np.zeros((T, c.routed_expert_hidden_size), dtype=np.float32)
    for e in sorted({int(v) for row in idx for v in row}):
        base = mp + "experts.%d." % e
        w1, w3, w2 = dequant(sh, base + "w1"), dequant(sh, base + "w3"), dequant(sh, base + "w2")
        for t in range(T):
            hit = np.where(idx[t] == e)[0]
            if hit.size == 0:
                continue
            a = situ_glu(torch.from_numpy(np.stack([z[t] @ w1.T, z[t] @ w3.T]).reshape(1, -1)),
                         c.situ_beta, c.situ_linear_beta).numpy().ravel()
            acc[t] += float(w[t, hit[0]]) * (a @ w2.T)
        del w1, w3, w2

    # RMSNorm on the AGGREGATE, not per expert.
    r = np.sqrt((acc.astype(np.float64) ** 2).mean(1, keepdims=True) + c.rms_norm_eps)
    y = ((acc / r) * lnorm).astype(np.float32) @ up.T

    s1 = q8(sh.get(mp + "shared_experts.gate_proj.weight").astype(np.float32))
    s3 = q8(sh.get(mp + "shared_experts.up_proj.weight").astype(np.float32))
    s2 = q8(sh.get(mp + "shared_experts.down_proj.weight").astype(np.float32))
    sa = situ_glu(torch.from_numpy(np.concatenate([x @ s1.T, x @ s3.T], axis=1)),
                  c.situ_beta, c.situ_linear_beta).numpy()
    return torch.from_numpy(y + sa @ s2.T)                 # shared added UNWEIGHTED


@torch.no_grad()
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    global INT8
    argv = [a for a in sys.argv if a != "--int8"]
    INT8 = len(argv) != len(sys.argv)
    shard_dir = argv[1]
    ids = [int(v) for v in argv[2].split(",") if v != ""]
    outp = argv[3] if len(argv) > 3 else "ref_logits.json"
    if INT8:
        print("INT8 SIMULATION: large matrices round-tripped through int8/g%d.\n"
              "  Routers, norms, biases, A_log, dt_bias, conv and res_proj stay exact.\n"
              "  embed and lm_head stay exact." % GROUP)

    c = K3Config()
    sh = Shards(shard_dir)
    T, H, eps = len(ids), c.hidden_size, c.rms_norm_eps
    print("reference forward: %d layers, %d prompt ids, hidden %d, vocab %d"
          % (c.num_hidden_layers, T, H, c.vocab_size))

    # ---- embedding: read only the referenced rows, not the full 2.35 GB table ----
    p, dt, shape, off, _nb = sh.index[PRE + "embed_tokens.weight"]
    itemsize = 2 if dt == "BF16" else 4
    h = np.empty((T, H), dtype=np.float32)
    with open(p, "rb") as f:
        for i, tok in enumerate(ids):
            f.seek(off + tok * H * itemsize)
            raw = f.read(H * itemsize)
            if dt == "BF16":
                h[i] = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                h[i] = np.frombuffer(raw, dtype=np.float32)
    h = torch.from_numpy(h).reshape(1, T, H)
    print("embedded %d ids (%s)" % (T, dt))

    # ---- 93 layers, mirroring k3_ref.DecoderLayer.forward statement for statement ----
    block_residual = h.new_zeros(T, 0, H)
    t_start = time.time()
    for L in range(c.num_hidden_layers):
        t0 = time.time()
        lp = PRE + "layers.%d." % L
        is_mla = c.is_mla(L)
        prefix_sum = h

        if block_residual.shape[1] > 0:
            h = apply_attn_res(
                prefix_sum.view(-1, H), block_residual,
                torch.from_numpy(sh.get(lp + "self_attention_res_proj.weight")
                                 .astype(np.float32)).squeeze(0),
                torch.from_numpy(sh.get(lp + "self_attention_res_norm.weight")
                                 .astype(np.float32)), eps).view(1, T, H)

        if L % c.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, H).unsqueeze(1)], dim=1)
            prefix_sum = None                                    # reset, k3_ref:998

        hn = rms(h, sh.get(lp + "input_layernorm.weight"), eps)
        attn = load_attention(sh, c, L, is_mla)
        att, _ = attn(hn)
        del attn
        prefix_sum = att if prefix_sum is None else prefix_sum + att

        h = apply_attn_res(
            prefix_sum.view(-1, H), block_residual,
            torch.from_numpy(sh.get(lp + "mlp_res_proj.weight").astype(np.float32)).squeeze(0),
            torch.from_numpy(sh.get(lp + "mlp_res_norm.weight").astype(np.float32)),
            eps).view(1, T, H)
        hpn = rms(h, sh.get(lp + "post_attention_layernorm.weight"), eps)
        mo = mlp_forward(sh, c, L, hpn.view(T, H)).view(1, T, H)
        h = mo if prefix_sum is None else prefix_sum + mo

        print("  L%-3d %-3s %-5s  %6.1f s   |h| max %.6f" %
              (L, "MLA" if is_mla else "KDA",
               "dense" if L < c.first_k_dense_replace else "MoE",
               time.time() - t0, float(h.abs().max())), flush=True)
        if not torch.isfinite(h).all():
            print("NON-FINITE hidden state at layer %d -- aborting" % L)
            return 1

    # ---- model-level AttnRes, final norm, lm_head ----
    h = apply_attn_res(
        h.view(-1, H), block_residual,
        torch.from_numpy(sh.get(PRE + "output_attn_res_proj.weight")
                         .astype(np.float32)).squeeze(0),
        torch.from_numpy(sh.get(PRE + "output_attn_res_norm.weight")
                         .astype(np.float32)), eps).view(1, T, H)
    h = rms(h, sh.get(PRE + "norm.weight"), eps)

    last = h.view(T, H)[-1].numpy().astype(np.float32)
    lm_name = "lm_head.weight" if sh.has("lm_head.weight") else PRE + "lm_head.weight"
    if not sh.has(lm_name):
        lm_name = "language_model.lm_head.weight"
    p, dt, shape, off, _nb = sh.index[lm_name]
    V = shape[0]
    logits = np.empty(V, dtype=np.float32)
    step = 4096
    itemsize = 2 if dt == "BF16" else 4
    with open(p, "rb") as f:
        f.seek(off)
        for s in range(0, V, step):
            n = min(step, V - s)
            raw = f.read(n * H * itemsize)
            if dt == "BF16":
                blk = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                blk = np.frombuffer(raw, dtype=np.float32)
            logits[s:s + n] = blk.reshape(n, H) @ last

    tok = int(np.argmax(logits))
    print("\nfinal position: argmax token %d, logit %.6f, mean %.6f, max %.6f"
          % (tok, float(logits[tok]), float(logits.mean()), float(logits.max())))
    print("total %.1f s" % (time.time() - t_start))

    with open(outp, "w") as f:
        json.dump({"prompt_ids": ids, "argmax": tok, "vocab": int(V),
                   "logits_bits": np.asarray(logits, dtype=np.float32).view(np.uint32).tolist()},
                  f)
    print("wrote %s" % outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
