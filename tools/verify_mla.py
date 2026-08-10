#!/usr/bin/env python
"""
verify_mla.py - check the pure-torch MLA reference against the REAL released
KimiMLAAttention class.

WHY THIS IS POSSIBLE AT ALL
    modeling_kimi_linear.py imports fla at module scope, and fla's KDA kernels are
    Triton-backed, so the whole file fails to import on a machine without Triton. But
    GatedMLA itself uses no fla, it is plain torch. So on a host where fla imports, the
    real KimiMLAAttention class can be instantiated directly, loaded with the reference
    implementation's weights, and compared output for output.

    This closes the most important gap in the validation chain. Without it, the C engine
    is checked against the pure-torch reference and the reference is checked only against
    itself, so a shared misreading of the architecture would pass every test in the
    project. This script checks the reference against the released code.

TWO DIFFERENCES THAT ARE EXPECTED AND ARE NOT BUGS
    1. KV cache layout. KimiDynamicCache.update (modeling_kimi_linear.py:157-173)
       stores the EXPANDED key and value states: 96 heads x (192 + 128) floats per token
       per layer. The reference caches the COMPRESSED latent, 512 + 64 per token per
       layer, and re-expands on use, 53x less memory, and what any serving engine does.
       The two are mathematically identical, so outputs must still agree to tolerance.
    2. This script runs without a cache at all (one forward pass over the full
       sequence), which removes difference 1 from the comparison entirely.

USAGE
    python verify_mla.py

    Requires torch, fla, and the released modeling code on the import path.
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
sys.path.insert(0, HF)

from k3_ref import GatedMLA, tiny_config  # noqa: E402


def main():
    print("torch", torch.__version__)
    # The released modeling file uses RELATIVE imports (from .configuration_kimi_k3
    # import ...), so it must be loaded as a package, not as a top-level module. It
    # also targets transformers 4.56.2 exactly: on 5.x, OutputRecorder has moved and
    # the import fails. config.json records the version it was written against.
    try:
        import transformers
        print("transformers", transformers.__version__)
        for base in (os.path.dirname(HF), os.path.expanduser("~/k3c")):
            if base not in sys.path:
                sys.path.insert(0, base)
        pkg = os.path.basename(HF)
        mod = __import__("%s.modeling_kimi_linear" % pkg, fromlist=["KimiMLAAttention"])
        cfgmod = __import__("%s.configuration_kimi_k3" % pkg, fromlist=["KimiLinearConfig"])
        KimiMLAAttention = mod.KimiMLAAttention
        KimiLinearConfig = cfgmod.KimiLinearConfig
    except Exception as e:
        print("cannot import the released modeling code: %s: %s" % (type(e).__name__, e))
        print("Needs: transformers==4.56.2, flash-linear-attention, fla-core, triton,")
        print("an __init__.py inside the modeling directory, and its PARENT on sys.path.")
        return 2

    c = tiny_config()
    torch.manual_seed(0)

    real_cfg = KimiLinearConfig(
        hidden_size=c.hidden_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        vocab_size=c.vocab_size,
        q_lora_rank=c.q_lora_rank,
        kv_lora_rank=c.kv_lora_rank,
        qk_nope_head_dim=c.qk_nope_head_dim,
        qk_rope_head_dim=c.qk_rope_head_dim,
        v_head_dim=c.v_head_dim,
        rms_norm_eps=c.rms_norm_eps,
        mla_use_nope=True,
        mla_use_output_gate=c.mla_use_output_gate,
        attn_res_block_size=c.attn_res_block_size,
    )
    real_cfg._attn_implementation = "eager"

    real = KimiMLAAttention(real_cfg, layer_idx=3).to(torch.float32).eval()
    ref = GatedMLA(c).to(torch.float32).eval()

    # Copy the released weights into the reference module by parameter name. A name that
    rm = dict(real.named_parameters())
    om = dict(ref.named_parameters())
    print("\nreal parameters : %d" % len(rm))
    print("reference parameters : %d" % len(om))
    missing, copied = [], 0
    with torch.no_grad():
        for n, p in om.items():
            if n in rm and rm[n].shape == p.shape:
                p.copy_(rm[n]); copied += 1
            else:
                missing.append((n, tuple(p.shape),
                                tuple(rm[n].shape) if n in rm else None))
    print("copied %d, mismatched/missing %d" % (copied, len(missing)))
    for n, ref_shape, real_shape in missing:
        print("   %-34s reference=%-18s released=%s" % (n, ref_shape, real_shape))
    if missing:
        print("\nSHAPE MISMATCH means the reference module is structurally different")
        print("from the released one. Fix that before trusting anything downstream.")
        return 1

    # CAUSALITY IS THE CALLER'S JOB. eager_attention_forward (modeling_kimi_linear.py
    # :325-326) only applies a mask "if attention_mask is not None", so passing None
    # makes the real module run BIDIRECTIONAL attention. The causal mask is built by
    # KimiLinearModel.forward via create_causal_mask and handed down. The reference
    # masks internally, so without this the two cannot agree, and the mismatch would
    # look like a model bug when it is purely a harness artefact.
    Tq = 7
    neg = torch.finfo(torch.float32).min
    causal = torch.full((1, 1, Tq, Tq), neg)
    causal = torch.triu(causal, diagonal=1)          # 0 on/below the diagonal

    x = torch.randn(1, Tq, c.hidden_size)
    with torch.no_grad():
        y_ref, _ = ref(x)
        y_real = real(x, attention_mask=causal, position_ids=None, past_key_values=None)
        if isinstance(y_real, tuple):
            y_real = y_real[0]

    d = (y_ref - y_real).abs()
    rel = d / (y_real.abs() + 1e-6)
    print("\n=== REFERENCE MLA vs THE RELEASED KimiMLAAttention ===")
    print("  shapes      : ref %s  real %s" % (list(y_ref.shape), list(y_real.shape)))
    print("  max abs diff: %.3e" % d.max())
    print("  mean abs    : %.3e" % d.mean())
    print("  max rel diff: %.3e" % rel.max())
    ok = bool(d.max() < 1e-4)
    print("  VERDICT     : %s" % ("MATCH" if ok else "DIVERGED"))
    if not ok:
        i = int(d.argmax())
        print("  worst element %d: ref %.7f real %.7f" %
              (i, float(y_ref.flatten()[i]), float(y_real.flatten()[i])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
