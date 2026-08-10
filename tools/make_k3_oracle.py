#!/usr/bin/env python
"""
make_k3_oracle.py - build a tiny Kimi K3 and emit the reference arrays a C engine
must reproduce EXACTLY.

Generates the tiny-model oracle fixture: a reduced-dimension K3 whose tensor graph
matches the real architecture exactly, giving the engine its teacher-forcing and
20/20 greedy targets. Without an equivalent, a Kimi K3 engine has nothing to prove
itself against.

WHAT IT EMITS  (tests/fixtures/ref_k3.json)
    config       the exact tiny config, so the C loader can be driven from it
    prompt_ids   the fixed input token ids
    full_ids     prompt followed by N greedily decoded tokens
    tf_pred      argmax at every position of ONE teacher-forced forward over full_ids
    logits_head  the first 8 logits at each position, for a numeric (not just argmax)
                 comparison when token identity fails and you need to see how far off

WHY THE TINY MODEL IS SHAPED THE WAY IT IS
    13 layers with attn_res_block_size 3 puts AttnRes boundaries at layers
    0, 3, 6, 9 and 12, giving five snapshots and four COMPLETE blocks. The AttnRes
    failure modes (wrong source set, forgetting that the embedding is a source,
    mishandling the prefix-sum reset) cannot appear until at least two blocks are
    complete and a third is in progress. Thirteen is therefore the smallest depth
    that can distinguish a correct implementation from a plausible wrong one; a
    shallower fixture passes both.

    full_attn_layers [4, 8, 12, 13] is ONE-BASED, mirroring the released config, so
    zero-based layers 3, 7 and 11 are MLA and layer 12 is a second MLA at the top.
    That reproduces the real model's two-consecutive-MLA tail, which is a layer-map
    edge case an implementer can easily get wrong.

    first_k_dense_replace 1 makes layer 0 dense AND KDA, exactly as in the real model.

WHY THE WEIGHTS ARE INITIALISED THE WAY THEY ARE
    Default small init would leave SiTU-GLU untested: with beta 4 and beta_2 25 and
    weights near zero, beta*tanh(z/beta) == z to within float noise and the entire
    soft cap is dead code. So the expert gate and up projections are deliberately
    scaled up until both caps engage. Likewise e_score_correction_bias is set to a
    spread of values so the frozen routing bias actually CHANGES which experts are
    selected relative to no bias; with a zero bias, a C engine that ignored the bias
    entirely would still pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3_ref import K3Model, tiny_config

from _paths import FIXTURES as FIX
os.makedirs(FIX, exist_ok=True)


def build(seed: int = 1234):
    torch.manual_seed(seed)
    cfg = tiny_config()
    model = K3Model(cfg).to(torch.float32).eval()

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            elif "weight" in name:
                p.fill_(1.0)                       # norm gains
            else:
                p.zero_()

        for layer in model.layers:
            # Make SiTU-GLU's soft caps actually engage rather than sit in the
            # near-linear regime where beta*tanh(z/beta) == z.
            mlp = layer.mlp
            experts = list(getattr(mlp, "experts", [])) + [getattr(mlp, "shared", None)]
            for e in experts:
                if e is None:
                    continue
                torch.nn.init.normal_(e.w1.weight, std=1.2)
                torch.nn.init.normal_(e.w3.weight, std=1.2)
            if hasattr(mlp, "gate_proj"):          # the dense layer
                torch.nn.init.normal_(mlp.gate_proj.weight, std=1.2)
                torch.nn.init.normal_(mlp.up_proj.weight, std=1.2)

            # A frozen routing bias that genuinely reorders the top-k. If this were
            # zero, an engine that ignored the bias would still pass the oracle.
            if hasattr(mlp, "e_score_correction_bias"):
                n = mlp.e_score_correction_bias.numel()
                mlp.e_score_correction_bias.copy_(torch.linspace(-0.15, 0.15, n))

            # A_log spread so exp(A_log) is not all 1.0 and the per-head decay scale
            # is actually exercised per head.
            if hasattr(layer.self_attn, "A_log"):
                h = layer.self_attn.A_log.numel()
                layer.self_attn.A_log.copy_(torch.linspace(-0.4, 1.2, h))
                torch.nn.init.normal_(layer.self_attn.dt_bias, std=0.5)
                torch.nn.init.normal_(layer.self_attn.q_conv1d.weight, std=0.4)
                torch.nn.init.normal_(layer.self_attn.k_conv1d.weight, std=0.4)
                torch.nn.init.normal_(layer.self_attn.v_conv1d.weight, std=0.4)
    return cfg, model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt-len", type=int, default=12)
    ap.add_argument("--new", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(FIX, "ref_k3.json"))
    ap.add_argument("--save-weights", default=os.path.join(FIX, "tiny_k3.pt"))
    a = ap.parse_args()

    cfg, model = build(a.seed)
    torch.manual_seed(a.seed + 1)
    prompt = torch.randint(0, cfg.vocab_size, (1, a.prompt_len))

    full = model.greedy(prompt, a.new)                       # greedy decode
    logits, _ = model.forward(full)                          # ONE teacher-forced pass
    tf_pred = logits.argmax(-1)

    ref = {
        "note": "Reference fixture for a pure-C Kimi K3 engine. Produced by k3_ref.py, "
                "a plain-PyTorch reimplementation of modeling_kimi_linear.py. "
                "tf_pred[i] predicts full_ids[i+1].",
        "seed": a.seed,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "prompt_ids": prompt[0].tolist(),
        "full_ids": full[0].tolist(),
        "tf_pred": tf_pred[0].tolist(),
        "logits_head": [[round(float(x), 6) for x in row[:8]] for row in logits[0]],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1)
    torch.save(model.state_dict(), a.save_weights)

    n_match = sum(1 for i in range(len(ref["full_ids"]) - 1)
                  if ref["tf_pred"][i] == ref["full_ids"][i + 1])
    print("wrote %s" % a.out)
    print("  prompt_ids : %d" % len(ref["prompt_ids"]))
    print("  full_ids   : %d" % len(ref["full_ids"]))
    print("  tf_pred    : %d" % len(ref["tf_pred"]))
    print("  weights    : %s" % a.save_weights)
    print()
    print("self-check, tf_pred[i] == full_ids[i+1] over the generated span:")
    print("  %d / %d positions agree" % (n_match, len(ref["full_ids"]) - 1))
    print("  (positions inside the PROMPT are expected to disagree: the model is")
    print("   random and the prompt is random, so it does not predict its own input.)")
    print()
    print("generated continuation: %s" % ref["full_ids"][a.prompt_len:])


if __name__ == "__main__":
    main()
