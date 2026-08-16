#!/usr/bin/env python3
"""MoEfication Stage-0 go/no-go analysis.

Consumes the artefacts written by `moefy_capture` (per-layer `c` = neuron contribution
`|silu(gate)*up| * ||down_col||`, and `hs` = FFN input) and answers the two gates the plan
front-loads BEFORE any routed FFN is built:

  Gate A — does the neuron structure exist?  Cluster each layer's neurons into balanced
    expert groups and measure the ORACLE activation-mass recall at top-k: with a perfect
    router, what fraction of each token's contribution mass falls inside the selected
    groups?  Compared against the per-neuron oracle (the ceiling: is the model even that
    sparse?) and a random-grouping control (does clustering actually help?).
      PASS if grouped-oracle recall >= --recall-gate (default 0.90).

  Gate B — would the arena cross?  From the grouped-oracle routing decisions, measure the
    load imbalance and the resident hot-set size, and check whether trunk + shared + hot
    routed experts fit the arena budget (the mechanism that turns disk-streaming into
    resident hits).

This is measurement only; it decides whether Stage 1 is worth building.

  moefy_analyze.py CAPTURE_DIR [--experts 68] [--topk 27] [--shared-groups 16]
     [--active 0.40] [--recall-gate 0.90] [--pca 64] [--holdout 0.3]
     [--arena-gb 9] [--trunk-gb 6.5] [--neuron-bytes 9960]
"""
import argparse, json, os, sys
import numpy as np

SWEEP_FRACS = [0.40, 0.50, 0.60, 0.70, 0.80]
# FFN share of decode weight traffic (Qwen3.8-27B): decode speedup ~= 1/(1-share+share*active)
FFN_SHARE = 0.62


def load_layer(d, layer, n, inter):
    c = np.fromfile(os.path.join(d, f"layer_{layer}_c.f32"), dtype="<f4")
    avail = c.size // inter
    if avail < n:
        sys.exit(f"layer {layer}: only {avail} tokens on disk, asked for {n}")
    return c[: n * inter].reshape(n, inter)


def tokens_on_disk(d, layers, inter):
    """Min captured tokens across layer files — lets a PARTIAL (still-running) capture be
    analysed for a directional read before the full run finishes."""
    m = None
    for L in layers:
        p = os.path.join(d, f"layer_{L}_c.f32")
        if not os.path.exists(p):
            return 0
        t = os.path.getsize(p) // (inter * 4)
        m = t if m is None else min(m, t)
    return m or 0


def per_neuron_oracle(C, active):
    """Ceiling: mean over tokens of (mass in the top `active` fraction of neurons)/total."""
    k = max(1, int(round(active * C.shape[1])))
    tot = C.sum(1) + 1e-30
    # partial sort: top-k per row
    part = np.partition(C, C.shape[1] - k, axis=1)[:, -k:]
    return (part.sum(1) / tot).mean()


def balanced_kmeans(X, k, size, iters=12, seed=0):
    """k clusters of exactly `size` rows each (k*size == len(X)). Greedy balanced assignment
    by ascending point-centroid distance; Lloyd updates between rounds."""
    n, d = X.shape
    assert n == k * size, f"{n} != {k}*{size}"
    rng = np.random.default_rng(seed)
    cen = X[rng.choice(n, k, replace=False)].astype(np.float32).copy()
    assign = np.full(n, -1)
    for _ in range(iters):
        # squared distances [n,k] in float32
        d2 = (
            (X * X).sum(1)[:, None]
            - 2.0 * X @ cen.T
            + (cen * cen).sum(1)[None, :]
        ).astype(np.float32)
        order = np.argsort(d2, axis=None, kind="stable")  # flat, ascending
        cap = np.full(k, size, dtype=np.int32)
        assign = np.full(n, -1)
        left = n
        for flat in order:
            i, cc = divmod(int(flat), k)
            if assign[i] == -1 and cap[cc] > 0:
                assign[i] = cc
                cap[cc] -= 1
                left -= 1
                if left == 0:
                    break
        for cc in range(k):
            cen[cc] = X[assign == cc].mean(0)
    return assign


def grouped_recall(C, groups, k):
    """Mean over tokens of (mass in the top-k highest-mass groups)/total. `groups` is a
    neuron->group id array; groups need not be equal size for this to be well-defined."""
    g = groups.max() + 1
    # group-mass matrix [tokens, g] via add.at
    gm = np.zeros((C.shape[0], g), dtype=np.float64)
    np.add.at(gm.T, groups, C.T)  # gm[:, groups[j]] += C[:, j]
    tot = C.sum(1) + 1e-30
    topk_mass = np.sort(gm, axis=1)[:, -k:].sum(1)
    # which groups are selected per token (for Gate B)
    sel = np.argsort(gm, axis=1)[:, -k:]
    return (topk_mass / tot).mean(), sel, gm


def neuron_features(C_train, pca):
    """Per-neuron activation-pattern feature. Binary co-activation (>= per-layer 60th pct,
    i.e. ~40% 'on'), neurons as rows, tokens as columns, then PCA to `pca` dims."""
    tau = np.quantile(C_train, 0.60)
    B = (C_train >= tau).astype(np.float32).T  # [neurons, tokens]
    B = B - B.mean(1, keepdims=True)
    if pca and pca < min(B.shape):
        from sklearn.decomposition import PCA
        B = PCA(n_components=pca, random_state=0).fit_transform(B)
    return np.ascontiguousarray(B, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--experts", type=int, default=68)
    ap.add_argument("--topk", type=int, default=27)
    ap.add_argument("--shared-groups", type=int, default=16, help="top-activation groups pinned always-on")
    ap.add_argument("--active", type=float, default=0.40, help="target active fraction (for the ceiling)")
    ap.add_argument("--recall-gate", type=float, default=0.90)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--holdout", type=float, default=0.30)
    ap.add_argument("--arena-gb", type=float, default=9.0)
    ap.add_argument("--trunk-gb", type=float, default=6.5)
    ap.add_argument("--neuron-bytes", type=int, default=9960, help="gate+up+down^T bytes per neuron")
    ap.add_argument("--n-layers", type=int, default=64, help="fallback if meta.json lacks n_layers")
    ap.add_argument("--max-tokens", type=int, default=0, help="cap tokens used (0 = all on disk)")
    ap.add_argument("--layers", default="", help="comma list; needed only if meta.json absent (partial run)")
    ap.add_argument("--inter", type=int, default=17408, help="used only if meta.json absent")
    args = ap.parse_args()

    mp = os.path.join(args.capture_dir, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    else:  # still-running capture: meta.json is written only at the end
        meta = {"inter": args.inter, "layers": [int(x) for x in args.layers.split(",") if x.strip()]}
        if not meta["layers"]:
            sys.exit("meta.json absent (partial run) — pass --layers")
    inter = meta["inter"]
    layers = meta["layers"]
    # Use whatever is actually on disk (supports analysing a still-running capture).
    n = min(meta.get("n_captured", 1 << 30), tokens_on_disk(args.capture_dir, layers, inter))
    if args.max_tokens:
        n = min(n, args.max_tokens)
    esz = inter // args.experts
    if esz * args.experts != inter:
        sys.exit(f"inter {inter} not divisible by --experts {args.experts} (use 17/34/68 for 17408)")
    n_hold = max(1, int(round(args.holdout * n)))
    n_train = n - n_hold
    print(f"# MoEfication Stage-0 analysis  ({meta.get('model','?')})")
    print(f"# tokens={n} (train {n_train} / hold {n_hold})  inter={inter}  layers={layers}")
    print(f"# config: {args.experts} experts x {esz} neurons, top-{args.topk} "
          f"(~{100*args.topk/args.experts:.0f}% active), gate={args.recall_gate}")
    if n_train < 32:
        print("!! very few tokens — numbers are a smoke check, not a verdict.")

    rows = []
    for L in layers:
        C = load_layer(args.capture_dir, L, n, inter)
        Ctr, Cho = C[:n_train], C[n_train:]
        if not np.isfinite(C).all():
            print(f"layer {L}: !! non-finite contributions"); continue

        ceiling = per_neuron_oracle(Cho, args.active)

        feats = neuron_features(Ctr, args.pca)
        groups = balanced_kmeans(feats, args.experts, esz, seed=0)
        rec, sel, gm = grouped_recall(Cho, groups, args.topk)

        rng = np.random.default_rng(1)
        rgroups = np.repeat(np.arange(args.experts), esz)
        rgroups = rng.permutation(rgroups)
        rrec, _, _ = grouped_recall(Cho, rgroups, args.topk)

        # shared-expert variant: pin the top --shared-groups by TRAIN activation rate,
        # cluster the rest, recall = shared mass + top-(topk-shared) routed mass.
        mu = (Ctr >= np.quantile(Ctr, 0.60)).mean(0)
        n_shared = args.shared_groups * esz
        shared_idx = np.argsort(mu)[-n_shared:]
        is_shared = np.zeros(inter, bool); is_shared[shared_idx] = True
        rest = np.where(~is_shared)[0]
        rgrp = args.experts - args.shared_groups
        rest_groups = balanced_kmeans(feats[rest], rgrp, esz, seed=2)
        shared_mass = Cho[:, shared_idx].sum(1)
        gm_r = np.zeros((Cho.shape[0], rgrp))
        np.add.at(gm_r.T, rest_groups, Cho[:, rest].T)
        k_routed = args.topk - args.shared_groups
        routed_mass = np.sort(gm_r, axis=1)[:, -k_routed:].sum(1)
        tot = Cho.sum(1) + 1e-30
        rec_shared = ((shared_mass + routed_mass) / tot).mean()

        # Gate B: per-group selection frequency across held tokens.
        freq = np.bincount(sel.reshape(-1), minlength=args.experts) / Cho.shape[0]
        hot = int((freq >= 0.80).sum())     # groups selected >80% of tokens
        cold = int((freq <= 0.05).sum())

        # active-fraction sweep: for each target active %, the per-neuron ceiling and the
        # grouped-oracle recall (reusing this layer's clustering). Answers "how much active
        # does THIS model need to preserve the mass?".
        sweep = {}
        for frac in SWEEP_FRACS:
            kk = max(1, min(args.experts, int(round(frac * args.experts))))
            gr, _, _ = grouped_recall(Cho, groups, kk)
            sweep[frac] = (per_neuron_oracle(Cho, frac), gr)

        rows.append((L, ceiling, rec, rec_shared, rrec, hot, cold, freq, sweep))
        print(f"\nlayer {L:2d}:  per-neuron ceiling@{int(100*args.active)}% = {ceiling:.3f}")
        print(f"          grouped oracle top-{args.topk}          = {rec:.3f}   "
              f"{'PASS' if rec>=args.recall_gate else 'fail'}")
        print(f"          + shared expert ({args.shared_groups} grp)     = {rec_shared:.3f}   "
              f"{'PASS' if rec_shared>=args.recall_gate else 'fail'}")
        print(f"          random-group control          = {rrec:.3f}   (clustering gain {rec-rrec:+.3f})")
        print(f"          load imbalance: {hot} hot (>80%), {cold} cold (<5%) of {args.experts} groups")

    if not rows:
        sys.exit("no layers analysed")

    # ---- verdict ----
    mean_ceiling = np.mean([r[1] for r in rows])
    mean_rec = np.mean([r[2] for r in rows])
    mean_shared = np.mean([r[3] for r in rows])
    best = max(mean_rec, mean_shared)
    print("\n" + "=" * 64)
    print(f"GATE A (structure):  mean grouped-oracle recall = {mean_rec:.3f}"
          f"  (+shared {mean_shared:.3f}),  ceiling {mean_ceiling:.3f}")
    a_pass = best >= args.recall_gate
    print(f"   -> {'PASS' if a_pass else 'FAIL'} at gate {args.recall_gate:.2f}"
          f"   [{'build' if a_pass else 'STOP — no router beats this; keep int8'}]")
    if mean_ceiling < args.recall_gate:
        print(f"   !! per-neuron ceiling itself < gate: the model is not {int(100*args.active)}% "
              f"sparse — MoEfication cannot work at this active fraction regardless of clustering.")

    # Gate B: hot-set resident estimate. The shared expert and the pinned hot routed groups
    # are resident in EVERY layer, so per-layer bytes multiply by n_layers.
    nlayers = meta.get("n_layers", args.n_layers)
    mean_hot = np.mean([r[5] for r in rows])
    shared_gb = args.shared_groups * esz * args.neuron_bytes * nlayers / 1e9
    hot_gb = mean_hot * esz * args.neuron_bytes * nlayers / 1e9
    resident = args.trunk_gb + shared_gb + hot_gb
    print(f"\nGATE B (arena crossing):  mean hot groups/layer = {mean_hot:.1f}  (x{nlayers} layers)")
    print(f"   resident estimate = trunk {args.trunk_gb:.1f} + shared {shared_gb:.1f} + "
          f"hot {hot_gb:.1f} = {resident:.1f} GB  vs arena {args.arena_gb:.1f} GB")
    b_fit = resident <= args.arena_gb
    print(f"   -> {'FITS' if b_fit else 'TIGHT/OVER'}: "
          f"{'the disk->resident crossing is reachable' if b_fit else 'crossing may not fire; win ~nominal only'}")
    # ---- active-fraction sweep: how much active does this model actually need? ----
    print("\nACTIVE-FRACTION SWEEP  (mean over sampled layers)")
    print(f"  {'active':>7} {'ceiling':>8} {'grouped':>8} {'decode x':>9}  vs int8 1.48x")
    def dec_speedup(active):  # memory-bound decode ceiling
        return 1.0 / (1.0 - FFN_SHARE + FFN_SHARE * active)
    cross90 = None
    for frac in SWEEP_FRACS:
        ceil = np.mean([r[8][frac][0] for r in rows])
        grp = np.mean([r[8][frac][1] for r in rows])
        sp = dec_speedup(frac)
        note = "  <-- beats int8" if (grp >= 0.90 and sp > 1.48) else ("" if grp < 0.90 else "  (lossy vs int8)")
        if cross90 is None and grp >= 0.90:
            cross90 = (frac, sp)
        print(f"  {int(frac*100):>6}% {ceil:>8.3f} {grp:>8.3f} {sp:>8.2f}x{note}")
    print("  (ceiling = perfect per-neuron oracle; grouped = 68-group top-k oracle w/ a perfect router)")
    if cross90 is None:
        print("  -> grouped recall never reaches 0.90 in [40,80]% — routing cannot preserve quality here.")
    else:
        frac, sp = cross90
        verdict = "and still beats int8" if sp > 1.48 else "but that is <= int8's lossless 1.48x"
        print(f"  -> need >= {int(frac*100)}% active for 0.90 grouped recall -> ~{sp:.2f}x decode, {verdict}.")

    print("=" * 64)
    print("NOTE: recall/arena are per the sampled layers only; a full run needs all 64.")


if __name__ == "__main__":
    main()
