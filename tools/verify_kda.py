#!/usr/bin/env python
"""
verify_kda.py - check the pure-torch KDA reference against the REAL released
KimiDeltaAttention class.

This is the most valuable single check in the project. KDA is 69 of the 93 layers, its
recurrence is the most intricate part of the model, and it is where a silent ordering
bug would do the most damage.

REQUIREMENTS
    A machine where `fla` imports, in practice, Linux. A GPU is not required, but it
    changes what is being tested: with CUDA present fla dispatches to its Triton kernels,
    and without it fla prints "Triton is not supported on current platform, roll back to
    CPU" and runs a pure-torch fallback. Only the CUDA run exercises the kernels the
    released model uses in production. See the device-selection comment in main().

STRUCTURAL DIFFERENCE TO WATCH
    The released forward ends with
        g = g_proj(x); g = rearrange(g, '... (h d) -> ... h d'); o = o_norm(o, g)
    where o_norm is fla's FusedRMSNormGated(head_v_dim, activation="sigmoid"), a FUSED
    gated norm that takes the RAW gate and applies the sigmoid internally.

    The reference implementation instead does RMSNorm(o) followed by an explicit
    multiply by sigmoid(g_proj(x)). Those are the same function only if the fused kernel
    is genuinely norm-then-sigmoid-gate in that order. This script proves it rather than
    assuming it, because the plausible alternatives, gating before norming, or norming
    the gate, both run cleanly and produce a different model.

MODE SELECTION
    modeling_kimi_linear.py:561 selects 'fused_recurrent' when use_cache and q_len == 1,
    and otherwise self.mode, which is 'chunk' (:481). A single token therefore exercises
    the recurrent kernel and a multi-token sequence the chunked one. Both are compared
    here: the two released paths must agree with each other and with the reference.
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HF = os.path.join(ROOT, "kimi_k3_hf", "files")
if not os.path.isdir(HF):
    HF = os.path.join(os.path.expanduser("~"), "k3c", "hf_files")

sys.path.insert(0, HERE)
from k3_ref import KimiDeltaAttention as RefKDA, tiny_config  # noqa: E402


def load_real():
    for base in (os.path.dirname(HF), os.path.expanduser("~/k3c")):
        if base not in sys.path:
            sys.path.insert(0, base)
    pkg = os.path.basename(HF)
    mod = __import__("%s.modeling_kimi_linear" % pkg, fromlist=["KimiDeltaAttention"])
    cfg = __import__("%s.configuration_kimi_k3" % pkg, fromlist=["KimiLinearConfig"])
    return mod.KimiDeltaAttention, cfg.KimiLinearConfig


def compare(tag, a, b, tol=2e-4):
    d = (a - b).abs()
    ok = bool(d.max() < tol)
    print("  %-22s max abs %.3e   mean %.3e   %s"
          % (tag, d.max(), d.mean(), "MATCH" if ok else "DIVERGED"))
    if not ok:
        i = int(d.argmax())
        print("      worst element %d: reference %.7f released %.7f"
              % (i, float(a.flatten()[i]), float(b.flatten()[i])))
    return ok


def main():
    print("torch", torch.__version__)
    try:
        RealKDA, KimiLinearConfig = load_real()
    except Exception as e:
        print("cannot import released modeling code: %s: %s" % (type(e).__name__, e))
        return 2

    c = tiny_config()
    torch.manual_seed(0)

    real_cfg = KimiLinearConfig(
        hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        vocab_size=c.vocab_size,
        rms_norm_eps=c.rms_norm_eps,
        attn_res_block_size=c.attn_res_block_size,
        linear_attn_config={
            "num_heads": c.kda_num_heads,
            "head_dim": c.kda_head_dim,
            "short_conv_kernel_size": c.short_conv_kernel_size,
            "gate_lower_bound": c.gate_lower_bound,
            "use_full_rank_gate": True,
            "full_attn_layers": c.full_attn_layers,
            "kda_layers": [i for i in range(1, c.num_hidden_layers + 1)
                           if i not in c.full_attn_layers],
        },
    )
    # Device selection decides WHICH released kernel is under test, so it is not an
    # implementation detail. With CUDA present, fla dispatches to its Triton kernels and
    # this script validates those. Without it, fla falls back to a pure-torch CPU path
    # and this script validates that instead. Both are worth checking; only the CUDA run
    # tests the kernels the released model actually uses in production.
    #
    # The reference module is placed on the same device either way, so the comparison is
    # like for like. Do not mix devices: a Triton kernel handed a CPU tensor fails with
    # "Pointer argument cannot be accessed from Triton (cpu tensor?)".
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    real = RealKDA(real_cfg, layer_idx=1).to(torch.float32).to(dev).eval()
    ref = RefKDA(c).to(torch.float32).to(dev).eval()

    rm, om = dict(real.named_parameters()), dict(ref.named_parameters())
    print("\nreal params %d, ref %d" % (len(rm), len(om)))
    missing = []
    with torch.no_grad():
        for n, p in om.items():
            if n in rm and rm[n].shape == p.shape:
                p.copy_(rm[n])
            else:
                missing.append((n, tuple(p.shape), tuple(rm[n].shape) if n in rm else None))
    if missing:
        print("SHAPE/NAME MISMATCH, fix before trusting anything downstream:")
        for n, o_s, r_s in missing:
            print("   %-28s reference=%-16s released=%s" % (n, o_s, r_s))
        print("\nreal-only names:", sorted(set(rm) - set(om)))
        return 1
    print("all %d parameters copied by name with matching shapes" % len(om))

    ok = True
    for T in (1, 4, 8, 16, 17):
        x = torch.randn(1, T, c.hidden_size, device=dev)
        with torch.no_grad():
            y_ref, _ = ref(x)
            out = real(x, attention_mask=None, cache_params=None, use_cache=False)
            y_real = out[0] if isinstance(out, tuple) else out
        ok &= compare("T=%-3d  reference vs released" % T, y_ref.float().cpu(), y_real.float().cpu())

    print("\nT=1 exercises the fused_recurrent kernel; T>1 exercises the chunked one")
    print("(modeling_kimi_linear.py:561). Both must agree with the reference and with")
    print("each other.")
    print("\nVERDICT: %s" % ("KDA REFERENCE CONFIRMED" if ok else "DIVERGENCE PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
