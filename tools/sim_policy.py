#!/usr/bin/env python
"""
sim_policy.py - which replacement policy is worth implementing, measured on a real trace.

sim_cache.py answers "how much RAM", assuming LRU. This answers the other question:
given the RAM you actually have, how much is left on the table by the POLICY?

The policies are the grounded ones, not recent ones. Expert access here is heavily
skewed (a few experts are hit almost every token) with a scan through 43 layers on top,
which is precisely the workload LRU was long ago shown to handle worst:

  OPT / Belady (1966)   the offline optimum. No online policy can beat it.
  LRU                   what the engine implements today.
  LFU                   pure frequency. Strong under skew, pathological under phase change.
  LRU-K (1993)          O'Neil/Roussopoulos/Weikum. Evicts by the Kth-most-recent
                        reference, so a single scan touch cannot promote a cold block
                        over a genuinely hot one. K=2 is the standard choice.
  2Q (1994)             Johnson/Shasha. A1 admission queue plus Am hot queue: a block
                        must be referenced TWICE before it earns cache residency, which
                        makes it scan-resistant with none of LRU-K's bookkeeping.
  ARC (2003)            Megiddo/Modha. Adapts the recency/frequency split online, no
                        tuning parameter. The one to beat.

usage: sim_policy.py trace.bin [--expert-mb 13.37] [--sizes 2,4,5,6,8,16]
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, Counter, defaultdict

import numpy as np


def load(path):
    raw = np.fromfile(path, dtype=np.int32)
    if len(raw) % 2:
        raise SystemExit("trace is not an even number of int32")
    lay, exp = raw[0::2].astype(np.int64), raw[1::2].astype(np.int64)
    return ((lay << 20) | exp).tolist()


def opt(trace, cap):
    """Belady. Evict whatever is used furthest in the future."""
    nxt = defaultdict(list)
    for i, k in enumerate(trace):
        nxt[k].append(i)
    pos = {k: 0 for k in nxt}
    live, hits = set(), 0
    for i, k in enumerate(trace):
        pos[k] += 1
        if k in live:
            hits += 1
            continue
        if len(live) >= cap:
            # furthest next use, treating "never again" as infinity
            worst, wd = None, -1
            for c in live:
                p = pos[c]
                d = nxt[c][p] if p < len(nxt[c]) else 1 << 60
                if d > wd:
                    worst, wd = c, d
            live.discard(worst)
        live.add(k)
    return hits


def lru(trace, cap):
    d, hits = OrderedDict(), 0
    for k in trace:
        if k in d:
            d.move_to_end(k)
            hits += 1
            continue
        if len(d) >= cap:
            d.popitem(last=False)
        d[k] = 1
    return hits


def lfu(trace, cap):
    freq, live, hits = Counter(), set(), 0
    for k in trace:
        freq[k] += 1
        if k in live:
            hits += 1
            continue
        if len(live) >= cap:
            live.discard(min(live, key=lambda c: freq[c]))
        live.add(k)
    return hits


def lru_k(trace, cap, K=2):
    """Evict by the Kth-most-recent reference; a first touch cannot displace a hot block."""
    hist = defaultdict(list)
    live, hits = set(), 0
    for t, k in enumerate(trace):
        hist[k].append(t)
        if len(hist[k]) > K:
            hist[k].pop(0)
        if k in live:
            hits += 1
            continue
        if len(live) >= cap:
            # oldest Kth reference wins eviction; blocks with <K references go first
            worst, wv = None, None
            for c in live:
                h = hist[c]
                v = (0, h[0]) if len(h) < K else (1, h[0])
                if wv is None or v < wv:
                    worst, wv = c, v
            live.discard(worst)
        live.add(k)
    return hits


def twoq(trace, cap):
    """2Q: A1in admission FIFO + Am hot LRU. Kin/Kout at the paper's defaults."""
    kin, kout = max(1, cap // 4), max(1, cap // 2)
    a1, am, a1out = OrderedDict(), OrderedDict(), OrderedDict()
    hits = 0
    for k in trace:
        if k in am:
            am.move_to_end(k)
            hits += 1
            continue
        if k in a1:
            hits += 1
            continue  # stays in A1 -- promotion only on a hit AFTER eviction
        if k in a1out:
            del a1out[k]
            if len(am) + len(a1) >= cap:
                if am:
                    am.popitem(last=False)
                elif a1:
                    a1.popitem(last=False)
            am[k] = 1
            continue
        if len(a1) >= kin:
            old, _ = a1.popitem(last=False)
            a1out[old] = 1
            if len(a1out) > kout:
                a1out.popitem(last=False)
        elif len(am) + len(a1) >= cap and am:
            am.popitem(last=False)
        a1[k] = 1
    return hits


def arc(trace, cap):
    """Megiddo/Modha adaptive replacement: T1/T2 real, B1/B2 ghost, p self-tunes."""
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p, hits = 0, 0

    def replace(k):
        nonlocal p
        if t1 and (len(t1) > p or (k in b2 and len(t1) == p)):
            old, _ = t1.popitem(last=False)
            b1[old] = 1
        elif t2:
            old, _ = t2.popitem(last=False)
            b2[old] = 1

    for k in trace:
        if k in t1:
            del t1[k]
            t2[k] = 1
            hits += 1
            continue
        if k in t2:
            t2.move_to_end(k)
            hits += 1
            continue
        if k in b1:
            p = min(cap, p + max(1, len(b2) // max(1, len(b1))))
            replace(k)
            del b1[k]
            t2[k] = 1
            continue
        if k in b2:
            p = max(0, p - max(1, len(b1) // max(1, len(b2))))
            replace(k)
            del b2[k]
            t2[k] = 1
            continue
        if len(t1) + len(b1) >= cap:
            if len(t1) < cap:
                b1.popitem(last=False)
                replace(k)
            else:
                t1.popitem(last=False)
        elif len(t1) + len(t2) + len(b1) + len(b2) >= cap:
            if len(t1) + len(t2) + len(b1) + len(b2) >= 2 * cap and b2:
                b2.popitem(last=False)
            replace(k)
        t1[k] = 1
    return hits


def sweep(trace, cap, cycle=43):
    """Layer-distance Belady, the policy src/cache.rs now implements.

    Expert access is a cyclic scan over `cycle` layers, so the dominant term of Belady's
    rule needs no oracle: from layer l, an expert for layer L is (L - l) mod cycle steps
    from its next possible use. Evict the largest such distance; break ties by LRU. Note
    the +1/-1: the layer we are STANDING on was just used and is a full cycle away, not
    zero -- folding it to zero protects exactly the wrong entries."""
    live, hits = {}, 0
    for t, k in enumerate(trace):
        if k in live:
            live[k] = t
            hits += 1
            continue
        if len(live) >= cap:
            l = (k >> 20) % cycle
            worst, wr = None, None
            for c, ts in live.items():
                d = ((c >> 20) % cycle + cycle - l - 1) % cycle + 1
                r = (d, -ts)
                if wr is None or r > wr:
                    worst, wr = c, r
            del live[worst]
        live[k] = t
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expert-mb", type=float, default=13.37)
    ap.add_argument("--sizes", default="2,4,5,6,8,12,16,24,32")
    a = ap.parse_args()

    trace = load(a.trace)
    n = len(trace)
    distinct = len(set(trace))
    eb = a.expert_mb * 1e6
    print(f"trace: {n} requests, {distinct} distinct experts "
          f"({distinct * eb / 1e9:.2f} GB to hold them all)")
    print(f"compulsory misses: {distinct}  ->  ceiling {100 * (n - distinct) / n:.2f}% "
          f"for ANY policy at ANY size\n")

    pols = [("LRU", lru), ("LFU", lfu), ("LRU-2", lru_k), ("2Q", twoq),
            ("ARC", arc), ("SWEEP", sweep), ("OPT", opt)]
    print(f"{'CACHE':>7} {'SLOTS':>7} " + " ".join(f"{p:>8}" for p, _ in pols)
          + "   best-vs-LRU")
    print("-" * 78)
    for gb in [float(s) for s in a.sizes.split(",")]:
        cap = int(gb * 1e9 / eb)
        if cap < 1:
            continue
        res = [(nm, 100.0 * f(trace, cap) / n) for nm, f in pols]
        base = dict(res)["LRU"]
        online = [v for nm, v in res if nm != "OPT"]
        gain = max(online) - base
        print(f"{gb:>6.0f}G {cap:>7} " + " ".join(f"{v:>7.2f}%" for _, v in res)
              + f"   {gain:+.2f} pt")


if __name__ == "__main__":
    main()
