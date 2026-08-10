#!/usr/bin/env python
"""make_tiny_checkpoint.py - turn the tiny oracle model into a ./bin/k3 checkpoint.

The engine's streaming loader (k3_load.c) refuses any expert whose logical width is
not divisible by the MXFP4 group size 32. The default tiny_config() uses
moe_intermediate_size=48 (48/32 = 1.5 groups) so it can never be loaded by bin/k3.
This script regenerates the tiny model with moe_intermediate_size=64 (2 groups),
then lays it out exactly as the released checkpoint is laid out:

  <out>/config.json            flat shape, the aliases k3_cfg.h accepts
  <out>/model.safetensors      BF16 trunk + MXFP4 experts, HF dotted names
  <out>/ref_logits.json        torch reference: final-position logits for the
                               prompt, so tools/cmp_logits.py can score bin/k3.

The init recipe mirrors tools/make_k3_oracle.py::build() so the SiTU caps engage
and the routing bias reorders selection, exactly like the oracle fixture.

The dtype of every tensor mirrors src/model/k3_bind.c's reqw/reqn split exactly:
  reqw  -> stored F32  (kernels read elementwise: norms, res projs, convs,
          A_log, dt_bias, o_norm, router gate + bias, latent norm)
  reqn  -> stored BF16 (large matrices read through k3_mmw only)
  experts -> MXFP4 U8 (packed + scale), low-nibble-first, per-32 scale

Usage: make_tiny_checkpoint.py <out_dir> [--seed N] [--prompt-ids a,b,c]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from k3_ref import K3Model, tiny_config  # noqa: E402

GROUP = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

REQN_LEAVES = {
    "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj", "self_attn.o_proj", "self_attn.g_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.f_a_proj", "self_attn.f_b_proj", "self_attn.b_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "block_sparse_moe.routed_expert_down_proj",
    "block_sparse_moe.routed_expert_up_proj",
    "block_sparse_moe.shared_experts.gate_proj",
    "block_sparse_moe.shared_experts.up_proj",
    "block_sparse_moe.shared_experts.down_proj",
}
REQN_TOP = {"embed_tokens", "lm_head"}


# ---------------------------------------------------------------- safetensors
def write_safetensors(path: str, tensors: dict):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        dt = {np.dtype("uint8"): "U8", np.dtype("uint16"): "BF16",
              np.dtype("float32"): "F32"}[arr.dtype]
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def bf16_of(w: torch.Tensor) -> np.ndarray:
    return (w.detach().float().cpu().numpy().view(np.uint32) >> 16).astype(np.uint16)


def bf16_roundtrip(w: np.ndarray) -> np.ndarray:
    return ((w.view(np.uint32) >> 16).astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------- MXFP4 quant
def mxfp4_quant(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-32 block: scale = nearest pow2 >= max|w| (E8M0 biased-127); nibble =
    nearest E2M1 value. Returns (packed U8 [R, C/2], scales U8 [R, C/32])."""
    w = w.astype(np.float32)
    R, C = w.shape
    assert C % GROUP == 0, "logical width %d not divisible by group %d" % (C, GROUP)
    g = C // GROUP
    blocks = w.reshape(R, g, GROUP)
    amax = np.abs(blocks).max(axis=2)
    # OCP MXFP4: the shared exponent puts the block maximum near the top of the E2M1
    # range, whose largest value is 6.0 = 1.5 * 2^2, hence the -2.
    #
    # This previously read max(ceil(log2 amax), 0). The clamp is the problem. It makes
    # 2^0 the smallest scale, so the smallest representable non-zero magnitude is 0.5 *
    # 1 = 0.5, and anything under 0.25 rounds to zero. Routed w2 keeps the base
    # N(0, 0.02) init while only w1 and w3 are re-scaled, so its block maxima are around
    # 0.03 to 0.07 and EVERY element of every routed w2 quantised to exactly zero. The C
    # engine and the torch reference then agreed perfectly, because both computed zero,
    # and a fixture built that way cannot fail: a wrong top-k, a reversed nibble order,
    # dropped routing weights or wrong SiTU betas all produce identical output.
    #
    # Checked against tests/fixtures/mxfp4.json, which holds real bytes from the released
    # checkpoint: the clamped rule reproduces 0 of its 7168 scale exponents, this one
    # reproduces the majority. It cannot reproduce all of them, because the fixture's
    # amax can only be recovered from already-quantised values.
    exp = (np.floor(np.log2(np.maximum(amax, 1e-30))) - 2).astype(np.int32)
    scales = (exp + 127).astype(np.uint8)
    mult = np.exp2(exp.astype(np.float32)).reshape(R, g, 1)
    x = blocks / mult
    pos = np.abs(x)
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    idx = np.argmin(np.abs(pos[..., None] - lut), axis=-1).astype(np.uint8)
    nib = np.where(x < 0, idx | 0x8, idx).astype(np.uint8)
    packed = (nib[:, :, 0::2] | (nib[:, :, 1::2] << 4)).reshape(R, g * GROUP // 2)

    # A matrix that quantises to all zeros still round-trips perfectly, still matches the
    # reference exactly, and tests nothing at all. Refuse to emit one: the whole point of
    # this checkpoint is to be a fixture that CAN fail.
    if not np.any(packed & 0x77):          # 0x77 masks both nibbles' magnitude bits
        raise SystemExit(
            "mxfp4_quant: every element quantised to zero (block maxima %.4g..%.4g).\n"
            "  The scale rule cannot represent values this small, so this tensor would\n"
            "  make the reference and the engine agree by both computing nothing.\n"
            "  Re-scale the tensor's initialisation, or widen the scale rule."
            % (float(amax.min()), float(amax.max())))

    # E8M0 is unsigned: an exponent outside -127..127 cannot be encoded and would wrap.
    if exp.min() < -127 or exp.max() > 127:
        raise SystemExit("mxfp4_quant: exponent %d..%d is outside the E8M0 range"
                         % (int(exp.min()), int(exp.max())))
    return packed, scales


def mxfp4_dequant(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    R, C = packed.shape
    lo, hi = (packed & 0x0F), (packed >> 4)
    out = np.empty((R, 2 * C), dtype=np.float32)
    out[:, 0::2] = E2M1[lo]
    out[:, 1::2] = E2M1[hi]
    mult = np.where(scales == 255, 0.0, np.exp2(scales.astype(np.int32) - 127).astype(np.float32))
    out *= np.repeat(mult, GROUP, axis=1)[:, : 2 * C]
    return out


# ---------------------------------------------------------------- name mapping
def engine_name(name: str) -> str:
    """k3_ref state_dict key -> the HF dotted name k3_bind.c / k3_load.c expect."""
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    if name == "embed_tokens.weight":
        return "language_model.model.embed_tokens.weight"
    if name.startswith("layers."):
        pre = "language_model.model." + name
        # k3_ref's LatentMoE attrs map onto the released naming
        pre = pre.replace(".mlp.e_score_correction_bias",
                          ".block_sparse_moe.gate.e_score_correction_bias")
        pre = pre.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
        pre = pre.replace(".mlp.down.weight",
                          ".block_sparse_moe.routed_expert_down_proj.weight")
        pre = pre.replace(".mlp.up.weight",
                          ".block_sparse_moe.routed_expert_up_proj.weight")
        pre = pre.replace(".mlp.norm.weight",
                          ".block_sparse_moe.routed_expert_norm.weight")
        pre = pre.replace(".mlp.experts.", ".block_sparse_moe.experts.")
        pre = pre.replace(".mlp.shared.w1.weight",
                          ".block_sparse_moe.shared_experts.gate_proj.weight")
        pre = pre.replace(".mlp.shared.w3.weight",
                          ".block_sparse_moe.shared_experts.up_proj.weight")
        pre = pre.replace(".mlp.shared.w2.weight",
                          ".block_sparse_moe.shared_experts.down_proj.weight")
        return pre
    # model-level norms / attn-res
    return "language_model.model." + name


def is_expert(name: str) -> bool:
    return ".mlp.experts." in name and name.endswith(".weight")


def is_reqn(eng: str, name: str) -> bool:
    if name in ("embed_tokens.weight", "lm_head.weight"):
        return True
    body = eng[len("language_model.model.layers."):]
    # body: "0.self_attn.q_proj.weight" -> strip the layer index and the
    # trailing ".weight" so it matches the REQN_LEAVES entries
    rest = body.split(".", 1)[1] if "." in body else body
    return rest.endswith(".weight") and rest[:-len(".weight")] in REQN_LEAVES


# ---------------------------------------------------------------- build
def build(seed: int):
    torch.manual_seed(seed)
    cfg = tiny_config(moe_intermediate_size=64)
    model = K3Model(cfg).to(torch.float32).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            elif "weight" in name:
                p.fill_(1.0)
            else:
                p.zero_()
        for layer in model.layers:
            mlp = layer.mlp
            experts = list(getattr(mlp, "experts", [])) + [getattr(mlp, "shared", None)]
            for e in experts:
                if e is None:
                    continue
                torch.nn.init.normal_(e.w1.weight, std=1.2)
                torch.nn.init.normal_(e.w3.weight, std=1.2)
            if hasattr(mlp, "gate_proj"):
                torch.nn.init.normal_(mlp.gate_proj.weight, std=1.2)
                torch.nn.init.normal_(mlp.up_proj.weight, std=1.2)
            if hasattr(mlp, "e_score_correction_bias"):
                n = mlp.e_score_correction_bias.numel()
                mlp.e_score_correction_bias.copy_(torch.linspace(-0.15, 0.15, n))
            if hasattr(layer.self_attn, "A_log"):
                h = layer.self_attn.A_log.numel()
                layer.self_attn.A_log.copy_(torch.linspace(-0.4, 1.2, h))
                torch.nn.init.normal_(layer.self_attn.dt_bias, std=0.5)
                torch.nn.init.normal_(layer.self_attn.q_conv1d.weight, std=0.4)
                torch.nn.init.normal_(layer.self_attn.k_conv1d.weight, std=0.4)
                torch.nn.init.normal_(layer.self_attn.v_conv1d.weight, std=0.4)
    return cfg, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt-ids", default="3,7,11,5,9")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    cfg, model = build(a.seed)
    ids = [int(v) for v in a.prompt_ids.split(",") if v != ""]
    print("model: hidden %d, layers %d, vocab %d, experts %d, moe_inter %d" %
          (cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size,
           cfg.num_experts, cfg.moe_intermediate_size))
    print("layer map: %d MLA (%s), first %d dense" %
          (len(cfg.full_attn_layers), cfg.full_attn_layers, cfg.first_k_dense_replace))

    # ---- transform the model's weights to exactly what the shards will hold, so
    # the torch forward is the TRUE answer for the bytes bin/k3 reads ----
    with torch.no_grad():
        for name, p in model.named_parameters():
            w = p.detach().float().cpu().numpy()
            eng = engine_name(name)
            if is_expert(name):
                packed, scales = mxfp4_quant(w)
                p.copy_(torch.from_numpy(mxfp4_dequant(packed, scales)))
            elif is_reqn(eng, name):
                p.copy_(torch.from_numpy(bf16_roundtrip(w)))
            # else: keep exact fp32 (reqw path)

    # ---- torch reference forward over the transformed weights ----
    with torch.no_grad():
        logits = model.forward(torch.tensor([ids]))[0][0, -1].float().cpu().numpy()
    tok = int(np.argmax(logits))
    with open(os.path.join(a.out_dir, "ref_logits.json"), "w") as f:
        json.dump({"prompt_ids": ids, "argmax": tok, "vocab": int(cfg.vocab_size),
                   "logits_bits": np.asarray(logits, dtype=np.float32).view(np.uint32).tolist()},
                  f)
    print("reference: argmax %d, wrote ref_logits.json" % tok)

    # ---- flat config.json (aliases accepted by k3_cfg.h) ----
    config = {
        "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size, "rms_norm_eps": cfg.rms_norm_eps,
        "kda_num_heads": cfg.kda_num_heads, "kda_head_dim": cfg.kda_head_dim,
        "short_conv_kernel_size": cfg.short_conv_kernel_size,
        "gate_lower_bound": cfg.gate_lower_bound,
        "num_attention_heads": cfg.num_attention_heads, "q_lora_rank": cfg.q_lora_rank,
        "kv_lora_rank": cfg.kv_lora_rank, "qk_nope_head_dim": cfg.qk_nope_head_dim,
        "qk_rope_head_dim": cfg.qk_rope_head_dim, "v_head_dim": cfg.v_head_dim,
        "mla_use_output_gate": bool(cfg.mla_use_output_gate),
        "num_experts": cfg.num_experts, "num_experts_per_token": cfg.num_experts_per_token,
        "num_shared_experts": cfg.num_shared_experts,
        "routed_expert_hidden_size": cfg.routed_expert_hidden_size,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "routed_scaling_factor": cfg.routed_scaling_factor,
        "moe_renormalize": bool(cfg.moe_renormalize),
        "latent_moe_use_norm": bool(cfg.latent_moe_use_norm),
        "first_k_dense_replace": cfg.first_k_dense_replace,
        "intermediate_size": cfg.intermediate_size,
        "attn_res_block_size": cfg.attn_res_block_size,
        "situ_beta": cfg.situ_beta, "situ_linear_beta": cfg.situ_linear_beta,
        "full_attn_layers": cfg.full_attn_layers,
    }
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    # ---- one safetensors shard, laid out so every layer's TRUNK tensors form one
    # contiguous run (pack_trunk.py requires this; the released checkpoint does
    # the same). Order: per layer, trunk tensors then that layer's experts, then
    # the model-level tensors. dict insertion order is preserved. ----
    by_layer: dict[int, dict[str, tuple]] = {}
    for name, w in sorted(model.state_dict().items()):
        eng = engine_name(name)
        ww = w.numpy()
        if name.startswith("layers."):
            L = int(name.split(".")[1])
        else:
            L = -1
        if is_expert(name):
            packed, scales = mxfp4_quant(ww)
            base = eng[:-len(".weight")]
            entry = [(base + ".weight_packed", packed),
                     (base + ".weight_scale", scales)]
            print("  %-78s %s %s  +  %s %s" % (
                base + ".weight_packed", packed.dtype, list(packed.shape),
                scales.dtype, list(scales.shape)))
        elif name.endswith(".self_attn.A_log") and not cfg.is_mla(L):
            # engine binds want=kda_head_dim(16) take=kda_heads(4); real checkpoint
            # ships 16 and zeroes the tail
            full = np.zeros(cfg.kda_head_dim, dtype=np.float32)
            full[:cfg.kda_num_heads] = ww
            entry = [(eng, full)]
            print("  %-78s %s %s" % (eng, full.dtype, list(full.shape)))
        elif is_reqn(eng, name):
            entry = [(eng, bf16_of(w))]
            print("  %-78s %s %s" % (eng, entry[0][1].dtype, list(entry[0][1].shape)))
        else:
            entry = [(eng, ww.astype(np.float32))]
            print("  %-78s %s %s" % (eng, ww.dtype, list(ww.shape)))
        by_layer.setdefault(L, {})[name] = entry

    tensors: dict = {}
    for L in range(cfg.num_hidden_layers):
        layer_items = by_layer[L]
        trunk_names = sorted(n for n in layer_items if not is_expert(n))
        expert_names = sorted(n for n in layer_items if is_expert(n))
        for name in trunk_names + expert_names:
            for k, v in layer_items[name]:
                tensors[k] = v
    for name in sorted(by_layer.get(-1, {})):
        for k, v in by_layer[-1][name]:
            tensors[k] = v
    write_safetensors(os.path.join(a.out_dir, "model.safetensors"), tensors)
    print("\nwrote %s/model.safetensors (%d tensors)" % (a.out_dir, len(tensors)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
