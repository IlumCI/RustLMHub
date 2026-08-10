#!/usr/bin/env python
"""
verify_real_layer.py - does the C engine compute a REAL Kimi K3 layer correctly?

THE ONE THING THIS TESTS THAT NOTHING ELSE DOES
    Every earlier test ran on weights this project generated. That validates the
    arithmetic and says nothing about the BINDING: whether k3_bind attached
    q_proj.weight to the q slot rather than the k slot, whether a matrix needed
    transposing, whether A_log was taken per head or per channel. Those failures all
    produce finite, plausible, wrong numbers, and no self-consistent test can see them.

    So this reads the same shards independently, in Python, builds the reference module
    from tools/k3_ref.py, and compares stage by stage. A disagreement localises to a
    stage, and within a stage to a handful of tensors.

STAGES
    A  KDA attention output      15 attention tensors, the recurrence, the decay
    B  router indices + weights  gate.weight and e_score_correction_bias
    C  MoE output                latent down/up, the aggregate norm, the shared expert,
                                 and the streamed MXFP4 routed experts

TOLERANCE
    The C engine accumulates in double and torch in float32, so exact equality is not
    expected and demanding it would be a false alarm. What matters is that the error
    sits at float32 rounding for the width involved, not at the level a swapped tensor
    would produce. A wrong binding does not miss by 1e-6; it misses by ~1.

usage: verify_real_layer.py <shard_dir> [real_layer.json]
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from k3_ref import (K3Config, GatedMLA, KimiDeltaAttention, RMSNorm,   # noqa: E402
                    situ_glu)

PRE = "language_model.model."
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


class Shards:
    """A deliberately separate safetensors reader. Sharing code with the C side, or
    even with tools/verify_st.py's parse, would weaken the independence."""

    def __init__(self, d):
        self.index = {}
        self.files = {}
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".safetensors"):
                continue
            p = os.path.join(d, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n).decode("utf-8"))
            base = 8 + n
            for name, e in hdr.items():
                if name == "__metadata__":
                    continue
                a, b = e["data_offsets"]
                self.index[name] = (p, e["dtype"], tuple(e["shape"]), base + a, b - a)

    def raw(self, name):
        p, dt, shape, off, nb = self.index[name]
        with open(p, "rb") as f:
            f.seek(off)
            return f.read(nb), dt, shape

    def get(self, name) -> np.ndarray:
        buf, dt, shape = self.raw(name)
        if dt == "F32":
            a = np.frombuffer(buf, dtype=np.float32)
        elif dt == "BF16":
            a = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "U8":
            a = np.frombuffer(buf, dtype=np.uint8)
        else:
            raise ValueError(dt)
        return a.reshape(shape)

    def has(self, name):
        return name in self.index


def dequant(sh: Shards, base: str) -> np.ndarray:
    """MXFP4 -> float32, done here with numpy so it does not share a line of code with
    the C implementation being checked."""
    p = sh.get(base + ".weight_packed")
    s = sh.get(base + ".weight_scale")
    R, C = p.shape
    lo, hi = (p & 0x0F), (p >> 4)
    out = np.empty((R, 2 * C), dtype=np.float32)
    out[:, 0::2] = E2M1[lo]          # low nibble is the EVEN element
    out[:, 1::2] = E2M1[hi]
    mult = np.where(s == 255, 0.0, np.exp2(s.astype(np.int32) - 127).astype(np.float32))
    out *= np.repeat(mult, 32, axis=1)[:, : 2 * C]
    return out


def bits_to_f32(v):
    return np.array(v, dtype=np.uint32).view(np.float32)


def cmp(label, a, b, width, fails, note=""):
    """Report agreement in units of float32 epsilon scaled by the accumulation width,
    which is the right yardstick for a dot product of that length."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        fails.append("%s: shape %s vs %s" % (label, a.shape, b.shape))
        return
    diff = np.abs(a - b)
    scale = max(float(np.abs(b).max()), 1e-30)
    rel = float(diff.max()) / scale
    # float32 has ~1.2e-7 resolution; a dot product of `width` terms accumulates error
    # roughly with sqrt(width). Anything within ~50x that is rounding, not a bug.
    budget = 1.2e-7 * np.sqrt(width) * 50
    ok = rel <= budget
    print("  %-28s max|diff| %.3e   rel %.3e   budget %.1e   %s%s"
          % (label, float(diff.max()), rel, budget, "OK" if ok else "MISMATCH", note))
    if not ok:
        i = int(np.argmax(diff))
        fails.append("%s: rel %.3e exceeds %.1e; worst at [%d] C=%.9g torch=%.9g"
                     % (label, rel, budget, i, a[i], b[i]))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    dump_p = sys.argv[2] if len(sys.argv) > 2 else "real_layer.json"

    D = json.load(open(dump_p, encoding="utf-8"))
    L, T, H, topk = D["layer"], D["T"], D["hidden"], D["topk"]
    print("checking layer %d, T=%d, hidden=%d, top-%d\n" % (L, T, H, topk))

    sh = Shards(d)
    c = K3Config()
    torch.set_grad_enabled(False)
    fails = []

    lp = PRE + "layers.%d." % L
    x = bits_to_f32(D["x"]).reshape(1, T, H)
    xn_c = bits_to_f32(D["x_norm"]).reshape(1, T, H)

    # ---- the input norm, as a cheap first check that x itself round-tripped ---------
    inw = torch.from_numpy(sh.get(lp + "input_layernorm.weight").astype(np.float32))
    n = RMSNorm(H, c.rms_norm_eps)
    n.weight.copy_(inw)
    xn_t = n(torch.from_numpy(x))
    print("stage A0  input RMSNorm")
    cmp("x_norm", xn_c, xn_t.numpy(), H, fails)

    # ---- stage A: attention ---------------------------------------------------------
    if D.get("is_mla"):
        print("\nstage A   Gated MLA attention (8 tensors)")
        mla = GatedMLA(c)
        sp = lp + "self_attn."
        mla.q_a_proj.weight.copy_(torch.from_numpy(sh.get(sp + "q_a_proj.weight").astype(np.float32)))
        mla.q_a_layernorm.weight.copy_(
            torch.from_numpy(sh.get(sp + "q_a_layernorm.weight").astype(np.float32)))
        mla.q_b_proj.weight.copy_(torch.from_numpy(sh.get(sp + "q_b_proj.weight").astype(np.float32)))
        mla.kv_a_proj_with_mqa.weight.copy_(
            torch.from_numpy(sh.get(sp + "kv_a_proj_with_mqa.weight").astype(np.float32)))
        mla.kv_a_layernorm.weight.copy_(
            torch.from_numpy(sh.get(sp + "kv_a_layernorm.weight").astype(np.float32)))
        mla.kv_b_proj.weight.copy_(torch.from_numpy(sh.get(sp + "kv_b_proj.weight").astype(np.float32)))
        mla.o_proj.weight.copy_(torch.from_numpy(sh.get(sp + "o_proj.weight").astype(np.float32)))
        if mla.g_proj is not None:
            mla.g_proj.weight.copy_(torch.from_numpy(sh.get(sp + "g_proj.weight").astype(np.float32)))
        print("  q_b [%d] = %d heads x (nope %d + rope %d); kv_b [%d] = %d x (nope %d + v %d)"
              % (sh.index[sp + "q_b_proj.weight"][2][0], c.num_attention_heads,
                 c.qk_nope_head_dim, c.qk_rope_head_dim,
                 sh.index[sp + "kv_b_proj.weight"][2][0], c.num_attention_heads,
                 c.qk_nope_head_dim, c.v_head_dim))
        print("  softmax scale = (nope+rope)^-0.5 = %d^-0.5; the rope dims are carried "
              "but NEVER rotated" % (c.qk_nope_head_dim + c.qk_rope_head_dim))
        out_t, _ = mla(torch.from_numpy(xn_c.reshape(1, T, H)))
        att_c = bits_to_f32(D["attn_out"]).reshape(1, T, H)
        cmp("attn_out", att_c, out_t.numpy(), H, fails)
        # No early return: an MLA layer has a MoE block too, and stages B and C below
        # check it exactly as they do for a KDA layer.
        return stages_bc(D, sh, lp, c, T, H, topk, fails)

    print("\nstage A   KDA attention (15 tensors)")
    att = KimiDeltaAttention(c)
    att.q_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.q_proj.weight").astype(np.float32)))
    att.k_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.k_proj.weight").astype(np.float32)))
    att.v_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.v_proj.weight").astype(np.float32)))
    att.g_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.g_proj.weight").astype(np.float32)))
    att.o_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.o_proj.weight").astype(np.float32)))
    att.f_a_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.f_a_proj.weight").astype(np.float32)))
    att.f_b_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.f_b_proj.weight").astype(np.float32)))
    att.b_proj.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.b_proj.weight").astype(np.float32)))
    for nm, mod in (("q", att.q_conv1d), ("k", att.k_conv1d), ("v", att.v_conv1d)):
        # Ships rank 3 as [P][1][kernel]; the reference module may hold it as [P][kernel]
        # or [P][1][kernel]. Same bytes in the same order either way, so reshape to
        # whatever the module declares rather than assuming.
        w = sh.get(lp + "self_attn.%s_conv1d.weight" % nm).astype(np.float32)
        mod.weight.copy_(torch.from_numpy(w).reshape(mod.weight.shape))
    # A_log ships [128] on a 96-head model; the tail is zero and it is PER HEAD.
    alog = sh.get(lp + "self_attn.A_log").astype(np.float32)
    print("  A_log shipped %d values, tail beyond %d heads all zero: %s"
          % (alog.size, c.kda_num_heads, bool(np.all(alog[c.kda_num_heads:] == 0))))
    att.A_log.copy_(torch.from_numpy(alog[: c.kda_num_heads]))
    att.dt_bias.copy_(torch.from_numpy(sh.get(lp + "self_attn.dt_bias").astype(np.float32)))
    att.o_norm.weight.copy_(torch.from_numpy(sh.get(lp + "self_attn.o_norm.weight").astype(np.float32)))

    out_t, _ = att(torch.from_numpy(xn_c.reshape(1, T, H)))
    att_c = bits_to_f32(D["attn_out"]).reshape(1, T, H)
    cmp("attn_out", att_c, out_t.numpy(), H, fails)

    return stages_bc(D, sh, lp, c, T, H, topk, fails)


def stages_bc(D, sh, lp, c, T, H, topk, fails):
    """Routing and MoE. Shared by both attention kinds: a KDA layer and an MLA layer
    have identical MoE blocks, so verifying it twice with two copies of the code would
    only create a way for the two to drift."""
    moe_in = bits_to_f32(D["moe_in"]).reshape(T, H)

    # ---- dense layer: no router, no experts -----------------------------------------
    # Layer 0 is the only dense layer in Kimi K3. Its MLP is a single 33792-wide
    # gate/up/SiTU-GLU/down, not a MoE block, so the checkpoint carries mlp.* rather than
    # block_sparse_moe.*. It needs its own branch here: routing code applied to a dense
    # layer dereferences a gate weight that does not exist.
    if "dense_out" in D:
        print("\nstage B   (skipped: dense layer, no router)")
        print("stage C   dense MLP")
        dp = lp + "mlp."
        dg = sh.get(dp + "gate_proj.weight").astype(np.float32)
        du = sh.get(dp + "up_proj.weight").astype(np.float32)
        dd = sh.get(dp + "down_proj.weight").astype(np.float32)
        print("  dense intermediate is %d (config dense_inter)" % dg.shape[0])
        g = moe_in @ dg.T
        u = moe_in @ du.T
        a = situ_glu(torch.from_numpy(np.concatenate([g, u], axis=1)),
                     c.situ_beta, c.situ_linear_beta).numpy()
        y = a @ dd.T
        dense_c = bits_to_f32(D["dense_out"]).reshape(T, H)
        cmp("dense_out", dense_c, y, H, fails)
        print()
        if fails:
            print("FAILED: %d problem(s)" % len(fails))
            for fl in fails:
                print("   " + fl)
            return 1
        print("VERIFIED: the C engine reproduces the real Kimi K3 DENSE layer (layer %d) "
              "from the real checkpoint, at float32 rounding, across attention and the "
              "%d-wide MLP. The bindings are correct." % (D["layer"], dg.shape[0]))
        return 0

    # ---- stage B: routing -----------------------------------------------------------
    print("\nstage B   router")
    mp = lp + "block_sparse_moe."
    gate_w = sh.get(mp + "gate.weight").astype(np.float32)
    bias = sh.get(mp + "gate.e_score_correction_bias").astype(np.float32)
    logits = moe_in @ gate_w.T
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    biased = scores + bias
    idx_t = np.argsort(-biased, axis=1)[:, :topk]
    w_t = np.take_along_axis(scores, idx_t, axis=1)
    w_t = w_t / (w_t.sum(1, keepdims=True) + 1e-20)

    idx_c = np.array(D["route_idx"], dtype=np.int64).reshape(T, topk)
    wt_c = bits_to_f32(D["route_wt"]).reshape(T, topk)
    same_sets = all(set(idx_c[t].tolist()) == set(idx_t[t].tolist()) for t in range(T))
    print("  selected expert SETS identical for all %d tokens: %s" % (T, same_sets))
    if not same_sets:
        for t in range(T):
            a, b = set(idx_c[t].tolist()), set(idx_t[t].tolist())
            if a != b:
                fails.append("token %d: C picked %s, torch picked %s"
                             % (t, sorted(a - b), sorted(b - a)))
    # Compare weights after sorting by expert id, since top-k order is unspecified.
    oc = np.argsort(idx_c, axis=1)
    ot = np.argsort(idx_t, axis=1)
    cmp("route_wt (id-sorted)", np.take_along_axis(wt_c, oc, 1),
        np.take_along_axis(w_t, ot, 1), H, fails)

    # ---- stage C: MoE ---------------------------------------------------------------
    print("\nstage C   MoE, latent path + shared expert + streamed MXFP4 experts")
    down = sh.get(mp + "routed_expert_down_proj.weight").astype(np.float32)
    up = sh.get(mp + "routed_expert_up_proj.weight").astype(np.float32)
    lnorm = sh.get(mp + "routed_expert_norm.weight").astype(np.float32)
    z = moe_in @ down.T                                          # [T, latent]

    Lw = c.routed_expert_hidden_size
    acc = np.zeros((T, Lw), dtype=np.float32)
    needed = sorted({int(e) for row in idx_t for e in row})
    print("  dequantising %d distinct experts with numpy" % len(needed))
    for e in needed:
        base = mp + "experts.%d." % e
        w1 = dequant(sh, base + "w1")
        w3 = dequant(sh, base + "w3")
        w2 = dequant(sh, base + "w2")
        for t in range(T):
            hit = np.where(idx_t[t] == e)[0]
            if hit.size == 0:
                continue
            g = z[t] @ w1.T
            u = z[t] @ w3.T
            a = situ_glu(torch.from_numpy(np.stack([g, u]).reshape(1, -1)),
                         c.situ_beta, c.situ_linear_beta).numpy().ravel()
            acc[t] += float(w_t[t, hit[0]]) * (a @ w2.T)
        del w1, w3, w2

    # RMSNorm on the AGGREGATE, not per expert.
    rms = np.sqrt((acc.astype(np.float64) ** 2).mean(1, keepdims=True) + c.rms_norm_eps)
    accn = (acc / rms) * lnorm
    y = accn.astype(np.float32) @ up.T

    sh1 = sh.get(mp + "shared_experts.gate_proj.weight").astype(np.float32)
    sh3 = sh.get(mp + "shared_experts.up_proj.weight").astype(np.float32)
    sh2 = sh.get(mp + "shared_experts.down_proj.weight").astype(np.float32)
    print("  shared expert intermediate is %d = num_shared_experts %d x moe_inter %d"
          % (sh1.shape[0], c.num_shared_experts, c.moe_intermediate_size))
    sg = moe_in @ sh1.T
    su = moe_in @ sh3.T
    sa = situ_glu(torch.from_numpy(np.concatenate([sg, su], axis=1)),
                  c.situ_beta, c.situ_linear_beta).numpy()
    y = y + sa @ sh2.T                                            # added UNWEIGHTED

    moe_c = bits_to_f32(D["moe_out"]).reshape(T, H)
    cmp("moe_out", moe_c, y, Lw, fails)

    print()
    if fails:
        print("FAILED: %d problem(s)" % len(fails))
        for f in fails:
            print("   " + f)
        return 1
    print("VERIFIED: the C engine reproduces a real Kimi K3 %s layer from the real "
          "checkpoint, at float32 rounding, across attention, routing and the "
          "streamed-expert MoE. The bindings are correct."
          % ("MLA" if D.get("is_mla") else "KDA"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
