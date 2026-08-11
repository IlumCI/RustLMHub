#!/usr/bin/env python
"""
roofline.py - which wall is each model actually standing against, on THIS machine.

WHY THIS EXISTS
    Every optimisation in this engine has been worth either a lot or nothing, and which one
    depended entirely on whether the model was bandwidth-bound or compute-bound. The AVX2
    MXFP4 kernel was 1.68x and bought ~8% end to end, because DeepSeek-V4 spends 92% of a
    token waiting on the disk. The same kernel on a model that fits in RAM would be worth
    the full 1.68x.

    So before building anything else, put every target model on the same axes and read off
    which wall it hits. This turns "is this worth doing" into arithmetic.

THE TWO REGIMES
    streaming  bytes/token / disk_bw  >  macs/token / kernel_rate     -> disk-bound
    resident   the other way round                                    -> compute-bound

    The interesting models sit near the crossover, and the interesting OPTIMISATION is the
    one that moves a model across it: shrinking the streamed bytes (precision tiering,
    caching) converts a disk-bound model into a compute-bound one, at which point the
    kernel work starts to pay and not before.

MACHINE NUMBERS ARE MEASURED, NOT SPEC SHEETS
    kernel_rate comes from `gguf_dump <file> <tensor> /dev/null bench`, disk from
    tools/devbw.py. Anything not measured is marked as such.

usage: roofline.py [--disk-mbs 500] [--kernel-gmacs 14.15] [--ram-gb 15] [--vram-gb 3.0]
"""
from __future__ import annotations

import argparse

# Measured on this machine. Update these rather than editing the tables below.
#
#   kernel: gguf_dump q4km.gguf blk.11.ffn_down.weight /dev/null bench 40
#             Q4_K 16.59 GMAC/s   Q5_K 14.58   Q6_K 11.66   (16 rayon threads)
#           Blended for a Q4_K_M recipe (2/3 Q4_K, 1/3 Q6_K by weight): 14.53 GMAC/s.
#           Was 8.87 blended before the scale-folding restructure -- Q6_K alone went
#           5.08 -> 11.66 by reducing once per 256-element super-block instead of once
#           per 16-element scale group.
#   disk:   Samsung T7 over a USB link negotiated at 5 Gbps (Gen1), so ~500 MB/s is the
#           ceiling regardless of what the drive can do; it is a 10 Gbps device.
DEFAULTS = dict(disk_mbs=500.0, kernel_gmacs=14.53, ram_gb=15.0, vram_gb=3.0)

# Expert access skew, measured post-bugfix on two ~155-token traces (DeepSeek-V4):
#   phil  hot 5% -> 34.6%   10% -> 47.5%   20% -> 63.4%   50% -> 87.5%
#   code  hot 5% -> 34.0%   10% -> 47.8%   20% -> 64.8%   50% -> 88.4%
# The earlier 32-token pre-bugfix trace said 58.4% at 20%, which confirms the warning that
# short traces FLATTEN a distribution: 5x the data moved it up ~6 points. The true value is
# probably a little higher still. Measured on DeepSeek-V4; whether it transfers to a
# different depth and top-k (Qwen3.5: 48 layers, top-8) is unknown.
HOT_SHARE, HOT_HITS = 0.20, 0.64
# Q3_K (3.44 bits) against Q4_K_M's ~4.85 average -- NOT the 2 bits originally assumed.
# Q2_K was ruled out on evidence: llama.cpp's own quantiser refuses to emit it for small
# tensors, and the in-tree qdq quantiser at 2 bits collapses to crude ternary.
COLD_BITS_RATIO = 3.44 / 4.85


class Model:
    """A model's per-token cost, in the only two units that decide anything."""

    def __init__(self, name, total_gb, macs_g, expert_bytes_gb, trunk_gb, note=""):
        self.name = name
        self.total_gb = total_gb                # whole checkpoint on disk
        self.macs_g = macs_g                    # GMAC per generated token
        self.expert_bytes_gb = expert_bytes_gb  # routed-expert bytes per token
        self.trunk_gb = trunk_gb                # resident, touched every token
        self.note = note


# Per-token figures derived from each checkpoint's own config/header. Expert bytes are
# (bits/param) x (params per expert) x (experts per token), which is the quantity the whole
# streaming design exists to reduce.
MODELS = [
    Model("Qwen2.5-0.5B Q4_K_M", 0.40, 0.40, 0.0, 0.40,
          "dense; fits entirely, so nothing streams"),
    Model("Qwen3-30B-A3B Q4_K_M", 18.0, 3.0, 1.8, 2.0,
          "3B active; total is near the RAM+VRAM budget -> the crossover case"),
    Model("Qwen3.5-122B-A10B Q4_K_M", 74.2, 6.0, 2.35, 5.0,
          "48 blocks, top-8 of 256; 36 SSM + 12 attention"),
    Model("DeepSeek-V4-Flash MXFP4", 166.9, 12.7, 3.45, 7.98,
          "measured end to end at ~8.0 s/token"),
]


def analyse(m, mach):
    cache_gb = max(0.0, mach["ram_gb"] + mach["vram_gb"] - m.trunk_gb - 1.0)
    # Resident fraction: what share of the routed-expert corpus can be held at once. The
    # trunk is always resident, so it does not stream.
    expert_corpus = max(1e-9, m.total_gb - m.trunk_gb)
    resident = min(1.0, cache_gb / expert_corpus)
    # Hit rate is NOT the resident fraction -- a cyclic scan over a working set larger than
    # the cache reuses almost nothing, which is exactly what the LRU simulation showed
    # (0.00% at 119 and 224 slots). Treat resident coverage as an upper bound and take a
    # measured-ish 0.5 factor below the knee.
    hit = resident if resident >= 0.99 else resident * 0.5
    streamed = m.expert_bytes_gb * (1.0 - hit)

    t_io = streamed / (mach["disk_mbs"] / 1000.0)
    t_cpu = m.macs_g / mach["kernel_gmacs"]
    total = max(t_io, t_cpu) + min(t_io, t_cpu) * 0.15  # partial overlap, mostly serial
    bound = "DISK" if t_io > t_cpu else "COMPUTE"
    return dict(cache_gb=cache_gb, resident=resident, streamed=streamed,
                t_io=t_io, t_cpu=t_cpu, total=total, bound=bound)


def main():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    a = vars(p.parse_args())
    mach = {k: a[k] for k in DEFAULTS}

    print(f"machine: disk {mach['disk_mbs']:.0f} MB/s | kernel {mach['kernel_gmacs']:.2f} "
          f"GMAC/s | RAM {mach['ram_gb']:.0f} GB + VRAM {mach['vram_gb']:.1f} GB\n")
    print(f"{'model':<30} {'GB/tok':>7} {'io s':>7} {'cpu s':>7} {'s/tok':>7} "
          f"{'tok/s':>7}  bound")
    print("-" * 82)
    rows = []
    for m in MODELS:
        r = analyse(m, mach)
        rows.append((m, r))
        print(f"{m.name:<30} {r['streamed']:>7.2f} {r['t_io']:>7.2f} {r['t_cpu']:>7.2f} "
              f"{r['total']:>7.2f} {1/r['total']:>7.2f}  {r['bound']}")

    print("\nwhat each lever is worth, per model")
    print("-" * 82)
    for m, r in rows:
        # Track 2: an 8x kernel only touches t_cpu.
        vnni = max(r["t_io"], r["t_cpu"] / 8) + min(r["t_io"], r["t_cpu"] / 8) * 0.15
        # Track 3b decomposes into two effects that must be scored SEPARATELY, because one
        # is robust and the other rests on an assumption the data does not yet support:
        #   residency  - keep the hottest HOT_SHARE of experts resident, serving HOT_HITS
        #                of accesses. MEASURED, weakly: 20% -> 58.4% on a 32-token trace,
        #                log-log slope -0.67 rather than Zipf's -1.0.
        #   shrink     - store the cold tail at 2 bits instead of ~4.85. Pure arithmetic,
        #                independent of any skew assumption, and therefore the reliable half.
        resid_io = r["t_io"] * (1.0 - HOT_HITS)
        shrink_io = r["t_io"] * COLD_BITS_RATIO
        tier_io = r["t_io"] * (1.0 - HOT_HITS) * COLD_BITS_RATIO
        f = lambda io, cpu: max(io, cpu) + min(io, cpu) * 0.15
        tier = f(tier_io, r["t_cpu"])
        both = f(tier_io, r["t_cpu"] / 8)
        print(f"{m.name:<30} now {r['total']:>6.2f}s | VNNI {r['total']/vnni:>5.2f}x | "
              f"residency {r['total']/f(resid_io, r['t_cpu']):>5.2f}x | "
              f"shrink {r['total']/f(shrink_io, r['t_cpu']):>5.2f}x | "
              f"tier {r['total']/tier:>5.2f}x | +VNNI {r['total']/both:>5.2f}x")

    print(f"""
reading it

  SEQUENCING. VNNI is worth 8x on a model that fits and ~1% on one that streams. If the
  goal is >=120B models, the kernel work is NOT the first move -- it only starts to pay
  once the I/O term has been cut, because it can never reduce the term that dominates.

  THE SKEW IS REAL BUT MODEST: the hottest 20% of experts serve {HOT_HITS*100:.0f}% of accesses,
  not the 80% a Zipf prior would suggest. Two ~155-token post-bugfix traces agree to
  within 1.4 points, so this is now measured rather than assumed.

  DOMAIN-ADAPTIVE TIERING IS DEAD, and the control is what killed it. Hot-set overlap
  between two maximally different topics (analytic philosophy vs compilers) is jaccard
  31.3% at the top 20%. That looks like strong specialisation -- until you split ONE
  topic's trace in half and measure it against itself: 27.0%. Cross-domain overlap is
  indistinguishable from, in fact slightly higher than, same-domain overlap. There is no
  domain signal; the variation is sampling noise. Holding "the code experts" resident
  would buy nothing over holding the globally hottest ones.

  THE SHRINK HALF IS ROBUST. Storing the cold tail at 2 bits rather than 4.85 is
  arithmetic, not a distributional bet, and it carries most of the tiering gain on its own.
  If the skew turns out to be flat, tiering still works -- it just becomes a compression
  story rather than a caching one.

  Q6_K IS THE CHEAP WIN NOBODY LOOKED AT. Q4_K_M uses it for every ffn_down, and it
  measured 5.08 GMAC/s against Q4_K's 14.15. A third of expert weight runs 2.8x slower
  than the rest; fixing it is worth ~1.3x on the compute term before any VNNI work.""")


if __name__ == "__main__":
    main()
