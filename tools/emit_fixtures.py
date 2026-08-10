#!/usr/bin/env python
"""
emit_fixtures.py - per-operation reference fixtures for the C engine.

WHY PER-OP FIXTURES
    The full-model oracle (ref_k3.json) is pass or fail. When it fails it tells you
    nothing about WHERE. These fixtures let each piece be validated on its own, in
    the order it should be built, so a bug is localised to one function instead of
    to a 93-layer stack.

FORMAT
    Plain JSON with flat float arrays, deliberately. The engine ships a minimal
    JSON reader (third_party/json.h), so the C engine can load these with code it
    already has, and a human can open one in an editor when a test fails.

    Every entry has: "shape", "data" (row-major flat), and where useful "note".

COVERAGE, and the reason each one exists
    rmsnorm       trivial, but the eps placement and the float upcast are easy to get
                  wrong and everything downstream depends on it.
    situ_glu      evaluated across |z| from 0.1 to 1000 SPECIFICALLY so both soft caps
                  saturate. In the near-linear regime beta*tanh(z/beta) == z and a
                  wrong implementation passes. Also asserts the |out| <= 100 bound.
    shortconv     includes the carried state, which is the piece a decode loop forgets.
    kda_decay     the full chain z -> g -> alpha, with a spread of A_log so the PER-HEAD
                  indexing is actually exercised. An implementation that indexed A_log
                  per CHANNEL would pass a single-head test and fail this one.
    kda_recur1 /
    kda_recur8    the recurrence in ISOLATION, fed directly with q, k, v, alpha and beta
                  so it can be validated without the surrounding projections. One step
                  pins the decay-then-write-then-read order; eight steps are where an
                  ordering error compounds far enough to be unmistakable.
    kda_layer1 /
    kda_layer8    the whole KDA layer including projections, ShortConv and the output
                  gate, at one token and at eight.
    router        index SET equality first, then weights. The frozen bias is non-zero
                  and reorders the top-k, so an engine ignoring it fails here.
    attnres       nine sources, keys normalised and values raw.
    mla           the NoPE path including the 64 unrotated dims and the head broadcast.
    moe           the full Stable LatentMoE block: route, down-project, run experts in
                  latent space, norm the aggregate, up-project, add the shared expert.
    layer_kda /
    layer_mla     a whole decoder layer, the last stop before the full model.

    The authoritative list is MANIFEST.json's "order" array, which is generated from
    what this script actually wrote. If this comment and the manifest disagree, the
    manifest is right.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3_ref import (KimiDeltaAttention, GatedMLA, LatentMoE, RMSNorm,
                    ShortConv, DecoderLayer, apply_attn_res, situ_glu, tiny_config)

from _paths import FIX_OPS as OUT
os.makedirs(OUT, exist_ok=True)


def T(x: torch.Tensor, note: str = "") -> dict:
    d = {"shape": list(x.shape), "data": [round(float(v), 7) for v in x.flatten().tolist()]}
    if note:
        d["note"] = note
    return d


def weights(mod, prefix: str = "") -> dict:
    """Export every named parameter of a module.

    Call this for EVERY module whose output a fixture records. A fixture that stores only
    inputs and outputs is unreproducible: the module was randomly initialised, so no C
    implementation can ever match it. Such a fixture provides zero coverage while looking
    exactly like a test that passes, the failure mode is a green tick, not a red one.
    """
    out = {}
    for n, p in mod.named_parameters():
        out[(prefix + n).replace(".", "_")] = T(p.detach(), "shape %s" % list(p.shape))
    return out


# Every fixture dump() writes, in the order it was written. The manifest's "order" list
# is built from this rather than maintained by hand: a hand-written list drifts the first
# time a fixture is renamed, added or dropped, and the drift is invisible because the
# manifest is documentation that nothing validates.
WRITTEN: list[str] = []


def dump(name: str, obj: dict):
    p = os.path.join(OUT, name + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    WRITTEN.append(name)
    n = sum(len(v["data"]) for v in obj.values() if isinstance(v, dict) and "data" in v)
    print("  %-16s %6d floats  %s" % (name, n, p))


@torch.no_grad()
def main():
    torch.manual_seed(7)
    c = tiny_config()
    print("emitting fixtures into %s" % OUT)

    # ---------- rmsnorm ----------
    n = RMSNorm(c.hidden_size, c.rms_norm_eps)
    torch.nn.init.normal_(n.weight, mean=1.0, std=0.1)
    x = torch.randn(3, c.hidden_size)
    dump("rmsnorm", {"eps": c.rms_norm_eps, "weight": T(n.weight), "in": T(x),
                     "out": T(n(x), "accumulate the mean of squares in double; eps is "
                                    "added INSIDE the rsqrt, not outside")})

    # ---------- situ_glu, driven into saturation ----------
    mags = [0.1, 1.0, 4.0, 25.0, 100.0, 1000.0]
    rows = []
    for m in mags:
        rows.append(torch.cat([torch.full((8,), m), torch.full((8,), -m)]))
    xs = torch.stack(rows)                                  # [6, 16] -> gate|up split
    ys = situ_glu(xs, c.situ_beta, c.situ_linear_beta)
    assert ys.abs().max() <= c.situ_beta * c.situ_linear_beta + 1e-3
    dump("situ_glu", {"beta": c.situ_beta, "linear_beta": c.situ_linear_beta,
                      "magnitudes": mags, "in": T(xs),
                      "out": T(ys, "sigmoid takes the UNCAPPED gate. bound is "
                                   "beta*linear_beta = 100, asserted here"),
                      "max_abs": round(float(ys.abs().max()), 6)})

    # ---------- shortconv, with carried state ----------
    P = c.kda_num_heads * c.kda_head_dim
    sc = ShortConv(P, c.short_conv_kernel_size)
    torch.nn.init.normal_(sc.weight, std=0.4)
    x1 = torch.randn(1, 5, P)
    y_full, st = sc(x1)
    x2 = torch.randn(1, 1, P)
    y_next, _ = sc(x2, st)
    dump("shortconv", {"kernel": c.short_conv_kernel_size, "channels": P,
                       "weight": T(sc.weight, "[channels, 1, kernel], depthwise"),
                       "in": T(x1), "out": T(y_full, "SiLU is FUSED inside the conv"),
                       "state_after": T(st, "the (kernel-1) past inputs a decode step "
                                            "must carry, per channel"),
                       "in_next": T(x2), "out_next": T(y_next)})

    # ---------- kda decay chain ----------
    kda = KimiDeltaAttention(c)
    for p in kda.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.3)
    kda.A_log.copy_(torch.linspace(-0.4, 1.2, c.kda_num_heads))
    torch.nn.init.normal_(kda.dt_bias, std=0.5)
    xd = torch.randn(1, 4, c.hidden_size)
    g = kda.decay(xd)
    # The low-rank projection output, BEFORE dt_bias is added, which is exactly what the
    # C function takes as its `z` argument. Without this the fixture only let the test
    # check that the reference's own alpha lay in range, which says nothing about the C
    # code. Exporting z lets t_decay actually drive k3_kda_decay forward and compare.
    z_pre = kda.f_b_proj(kda.f_a_proj(xd)).float()
    dump("kda_decay", {"lower_bound": c.gate_lower_bound,
                       "z": T(z_pre, "f_b_proj(f_a_proj(x)), BEFORE dt_bias. This is the "
                                     "`z` argument of k3_kda_decay, which adds dt_bias "
                                     "itself. Drives the C function forward."),
                       "A_log": T(kda.A_log, "PER HEAD. The checkpoint stores 128 "
                                             "values but only the first num_heads are "
                                             "nonzero; the rest are padding."),
                       "dt_bias": T(kda.dt_bias, "per (head, channel)"),
                       "in": T(xd), "g": T(g, "g = lower_bound*sigmoid(exp(A_log[h])*"
                                              "(z+dt_bias)), in (lower_bound, 0]"),
                       "alpha": T(g.exp(), "alpha = exp(g). Mathematically the range is "
                                           "the OPEN interval (e^lb, 1), but in float32 "
                                           "sigmoid underflows to exactly 0 once its "
                                           "argument drops below about -87.3, so g == -0.0 "
                                           "and alpha == 1.0 EXACTLY for strongly "
                                           "retaining channels. A C engine will hit the "
                                           "same saturation and must not treat alpha == 1 "
                                           "as an error. The usable bound is (e^lb, 1]."),
                       "alpha_min": round(float(g.exp().min()), 8),
                       "alpha_max": round(float(g.exp().max()), 8),
                       "n_alpha_saturated_at_1": int((g == 0.0).sum()),
                       "n_alpha_total": int(g.numel())})

    # ---------- kda recurrence, ISOLATED ----------
    # Feeds the recurrence directly with q, k, v, alpha and beta, so k3_kda_step can be
    # validated with no projections in the way.
    #
    # A fixture recording only in/out/state would be useless here: without the module
    # weights a C engine cannot reproduce the inputs, so every comparison would be
    # against numbers it had no way to compute. Exporting the recurrence's actual
    # arguments is what makes this test capable of failing, and this is where the
    # ordering that matters lives (decay, then the delta write, then read the UPDATED
    # state).
    H, D = c.kda_num_heads, c.kda_head_dim
    for nm, Tn in (("kda_recur1", 1), ("kda_recur8", 8)):
        torch.manual_seed(99)
        q = torch.randn(Tn, H, D)
        k = torch.randn(Tn, H, D)
        vv = torch.randn(Tn, H, D)
        al = torch.rand(Tn, H, D) * 0.99 + 0.005        # inside (e^-5, 1)
        bt = torch.rand(Tn, H)
        S = torch.zeros(H, D, D)
        outs = []
        for t in range(Tn):                             # naive.py:59-63, exactly
            S = S * al[t].unsqueeze(-1)
            u = torch.einsum('hk,hkv->hv', k[t], S)
            S = S + torch.einsum('hk,hv->hkv', k[t], bt[t].unsqueeze(-1) * (vv[t] - u))
            outs.append(torch.einsum('hk,hkv->hv', q[t] * (D ** -0.5), S))
        dump(nm, {"T": Tn, "H": H, "d_k": D, "d_v": D, "q_scale": D ** -0.5,
                  "q": T(q, "NOT pre-scaled here; the C routine receives q already "
                            "multiplied by d_k^-0.5, as naive.py:52 does"),
                  "k": T(k), "v": T(vv), "alpha": T(al), "beta": T(bt, "per-head SCALAR"),
                  "out": T(torch.stack(outs), "o = S^T q from the ALREADY UPDATED state"),
                  "state_out": T(S, "[H, d_k, d_v] row-major")})

    # ---------- full KDA layer, weights included so C can reproduce it ----------
    for nm, Tn in (("kda_layer1", 1), ("kda_layer8", 8)):
        xs2 = torch.randn(1, Tn, c.hidden_size)
        y, st2 = kda(xs2)
        w = {"q_proj": kda.q_proj.weight, "k_proj": kda.k_proj.weight,
             "v_proj": kda.v_proj.weight, "g_proj": kda.g_proj.weight,
             "o_proj": kda.o_proj.weight, "f_a": kda.f_a_proj.weight,
             "f_b": kda.f_b_proj.weight, "b_proj": kda.b_proj.weight,
             "q_conv": kda.q_conv1d.weight, "k_conv": kda.k_conv1d.weight,
             "v_conv": kda.v_conv1d.weight, "A_log": kda.A_log,
             "dt_bias": kda.dt_bias, "o_norm": kda.o_norm.weight}
        d = {"T": Tn, "H": H, "d_k": D, "conv_k": c.short_conv_kernel_size,
             "lower_bound": c.gate_lower_bound, "rms_eps": c.rms_norm_eps,
             "in": T(xs2), "out": T(y), "state_out": T(st2[0])}
        for kk, vvv in w.items():
            d[kk] = T(vvv)
        dump(nm, d)

    # ---------- router ----------
    # The gate weights must be SMALL. With std=0.3 on a 128-wide input the logits are
    # large, sigmoid saturates, and the scores spread across almost all of (0,1); a
    # modest bias then cannot reorder anything and the fixture silently stops testing
    # the bias at all. Real trained routers produce tightly clustered scores, which is
    # exactly the regime where the frozen bias decides the top-k. std=0.02 reproduces
    # that. The assert below makes the weakness impossible to reintroduce unnoticed.
    moe = LatentMoE(c)
    torch.nn.init.normal_(moe.gate.weight, std=0.02)
    moe.e_score_correction_bias.copy_(torch.linspace(-0.15, 0.15, c.num_experts))
    xr = torch.randn(6, c.hidden_size)
    idx, w = moe.route(xr)
    idx_nobias = torch.topk(torch.sigmoid(torch.nn.functional.linear(
        xr.float(), moe.gate.weight.float())), c.num_experts_per_token, -1).indices
    changed = int((idx.sort(-1).values != idx_nobias.sort(-1).values).any(-1).sum())
    assert changed > 0, (
        "router fixture is USELESS: the frozen bias never changed the top-k, so an "
        "engine that ignored e_score_correction_bias entirely would still pass. "
        "Lower the gate weight std until the scores cluster.")
    dump("router", {"top_k": c.num_experts_per_token, "n_experts": c.num_experts,
                    "gate_weight": T(moe.gate.weight), "bias": T(moe.e_score_correction_bias),
                    "in": T(xr), "topk_idx": T(idx.float(), "selection uses BIASED scores"),
                    "topk_weight": T(w, "weights gathered from UNBIASED sigmoid scores, "
                                        "renormalised, then scaled"),
                    "rows_where_bias_changed_selection": changed})

    # ---------- attnres ----------
    D = c.hidden_size
    prefix = torch.randn(4, D)
    blocks = torch.randn(4, 8, D)                 # 8 completed blocks + prefix = 9 sources
    pw, nw = torch.randn(D), torch.randn(D).abs() + 0.5
    dump("attnres", {"eps": c.rms_norm_eps, "n_sources": 9,
                     "prefix_sum": T(prefix), "block_residual": T(blocks),
                     "proj_weight": T(pw), "norm_weight": T(nw),
                     "fold_hint": T(nw * pw, "norm.weight * proj.weight collapses to ONE "
                                             "vector; fold it at load time"),
                     "out": T(apply_attn_res(prefix, blocks, pw, nw, c.rms_norm_eps),
                              "keys are RMSNormed, VALUES ARE RAW")})

    # ---------- mla ----------
    mla = GatedMLA(c)
    for p in mla.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.3)
    xm = torch.randn(1, 6, c.hidden_size)
    ym, cache = mla(xm)
    d_mla = {"scale": mla.scale, "qk_nope": c.qk_nope_head_dim,
             "qk_rope": c.qk_rope_head_dim, "kv_lora": c.kv_lora_rank,
             "q_lora": c.q_lora_rank, "v_head": c.v_head_dim,
             "n_heads": c.num_attention_heads, "rms_eps": c.rms_norm_eps,
             "in": T(xm), "out": T(ym),
                 "kv_latent": T(cache[0], "stored per token"),
                 "k_rope": T(cache[1], "the 64-wide slot exists but is NEVER rotated; "
                                       "one head, broadcast to all")}
    d_mla.update(weights(mla))
    dump("mla", d_mla)

    # ---------- whole layers ----------
    for nm, li in (("layer_kda", 1), ("layer_mla", 3)):
        lay = DecoderLayer(c, li)
        for p in lay.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.2)
        xl = torch.randn(1, 4, c.hidden_size)
        br = torch.randn(4, 2, c.hidden_size)
        out, br2, _ = lay(xl, br)
        d_lay = {"layer_idx": li, "is_mla": lay.is_mla,
                 "attn_res_block_size": c.attn_res_block_size,
                 "rms_eps": c.rms_norm_eps, "hidden": c.hidden_size,
                 "in": T(xl), "block_residual_in": T(br),
                 "out": T(out), "block_residual_out": T(br2,
                           "grows only on layers where idx % block_size == 0")}
        d_lay.update(weights(lay))
        dump(nm, d_lay)

    # ---------- full LatentMoE block ----------
    # Verified against modeling_kimi_linear.py:815-838. Order is load bearing:
    # gate on FULL width, down-project to latent, experts in latent space, RMSNorm on
    # the AGGREGATE (not per expert), up-project, then add the shared expert computed
    # on the ORIGINAL full-width input with NO weight and NO scaling.
    moe2 = LatentMoE(c)
    for p_ in moe2.parameters():
        if p_.dim() >= 2:
            torch.nn.init.normal_(p_, std=0.25)
    torch.nn.init.normal_(moe2.gate.weight, std=0.02)      # cluster scores, see router
    moe2.e_score_correction_bias.copy_(torch.linspace(-0.15, 0.15, c.num_experts))
    for e in list(moe2.experts) + [moe2.shared]:
        torch.nn.init.normal_(e.w1.weight, std=1.1)        # drive SiTU into its caps
        torch.nn.init.normal_(e.w3.weight, std=1.1)
    xmo = torch.randn(1, 3, c.hidden_size)
    ymo = moe2(xmo)
    d_moe = {"hidden": c.hidden_size, "latent": c.routed_expert_hidden_size,
             "moe_inter": c.moe_intermediate_size, "n_experts": c.num_experts,
             "top_k": c.num_experts_per_token, "n_shared": c.num_shared_experts,
             "routed_scale": c.routed_scaling_factor, "renorm": int(c.moe_renormalize),
             "latent_norm": int(c.latent_moe_use_norm), "rms_eps": c.rms_norm_eps,
             "situ_b1": c.situ_beta, "situ_b2": c.situ_linear_beta,
             "in": T(xmo), "out": T(ymo)}
    d_moe.update(weights(moe2))
    dump("moe", d_moe)

    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({"config": c.__dict__,
                   "order": list(WRITTEN),
                   "tolerance": {"fp32_abs": 1e-5, "fp32_rel": 1e-4},
                   "note": "Validate in the listed order: each fixture depends only on "
                           "kernels earlier in the list, so the first failure is the "
                           "root cause rather than a downstream symptom. Every fixture "
                           "is self-contained (weights, inputs, expected outputs). The "
                           "order is generated from what emit_fixtures.py wrote, not "
                           "maintained by hand."}, f, indent=1)
    print("\nwrote MANIFEST.json  (%d fixtures: %s)" % (len(WRITTEN), ", ".join(WRITTEN)))


if __name__ == "__main__":
    main()
