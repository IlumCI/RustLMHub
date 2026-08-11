#!/usr/bin/env python
"""
verify_dspark.py - check the Rust DSpark drafter against an independent numpy parse.

WHY THIS EXISTS
    src/dspark.rs implements DeepSeek-V4-Flash's trained block drafter, and almost every
    part of it is a place to be plausibly wrong while still producing tokens:

      * main_proj consumes the CONCATENATION of the main model's hidden states at layers
        40, 41 and 42. Permuting them, or using the hc_head reduce where the reference
        uses a plain mean, gives a working projection of the wrong thing.
      * The KV cache is keyed to the MAIN model's positions. Roping those rows at the
        draft block's positions instead of their own is a one-character error.
      * Attention inside the draft block is BIDIRECTIONAL -- every one of the 5 queries
        gets the same key set. A causal reading drafts fluently and badly.
      * The window ends at the SEED's position, not at each query's own position.
      * The Markov head biases position i's logits by an embedding of the token chosen at
        i-1, so it is autoregressive across a block that was otherwise computed in
        parallel. Dropping the chain leaves 5 independent guesses that still decode.
      * hc_head is a learned sigmoid gate, not a mean over the 4 copies.

    This reparses the same shards with numpy, follows inference/model.py
    (DSparkBlock, DSparkAttention, DSparkMarkovHead, DSparkConfidenceHead) and
    inference/kernel.py (hc_split_sinkhorn), and compares against what the Rust binary
    dumped. Different implementation, same weights.

WHAT IT DELIBERATELY DOES NOT DO
    inference/model.py calls act_quant on kv[..., :-rope_head_dim], simulating the FP8
    quantisation of the activations. Neither this reference nor src/dspark.rs does, so
    both are strictly more precise than the released kernel and agree with each other
    rather than with it. Same choice, and same reason, as verify_v4_attn.py.

usage: verify_dspark.py <shard_dir> <rust_out.bin>
       (produce rust_out.bin with: dspark <shard_dir> <T> <seed> <cache_gb> <out.bin>)
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

import numpy as np

# The narrow dtypes come from torch rather than from anything written here: bf16 is a
# pure bit-shift and needs no library, but e4m3 and e8m0 have subnormals and a NaN
# encoding, and hand-rolling those would make this a second copy of the implementation
# it is supposed to be checking. ml_dtypes is used when present; torch is the fallback
# because this machine's python is PEP 668-managed and has torch but not ml_dtypes.
try:
    import ml_dtypes

    def narrow_f32(u8, kind):
        dt = {"e4m3": ml_dtypes.float8_e4m3fn, "e8m0": ml_dtypes.float8_e8m0fnu}[kind]
        return u8.view(dt).astype(np.float32)
except ImportError:
    try:
        import torch
    except ImportError:
        sys.exit("needs ml_dtypes or torch")

    def narrow_f32(u8, kind):
        dt = {"e4m3": torch.float8_e4m3fn, "e8m0": torch.float8_e8m0fnu}[kind]
        return torch.from_numpy(u8.copy()).view(dt).float().numpy()


def bf16_f32(u8):
    """bf16 IS the top 16 bits of the f32 with the same value, so this is the definition
    rather than a conversion -- no library, no rounding, nothing to get wrong."""
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)

E, HC, H, HD, RD, QL, OL, G, WIN = 4096, 4, 64, 512, 64, 1024, 1024, 8, 128
EPS, HC_EPS, SINKHORN_ITERS = 1e-6, 1e-6, 20
MOE_INTER, NEXP, TOPK, ROUTE_SCALE, SWIGLU_LIMIT = 2048, 256, 6, 1.5, 10.0
ROPE_THETA = 10000.0          # compress_ratio == 0 disables YaRN and uses the base theta
NOISE = 128799
TARGETS = 3                   # len(dspark_target_layer_ids)

# E2M1: three magnitude bits, one sign bit. kernel.py pins fp4_max = 6.0.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


class Shards:
    """Every tensor in the directory, by name, read straight from the mmapped shard."""

    def __init__(self, d):
        self.idx = {}
        self.files = {}
        for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            f = open(p, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            self.files[p] = (f, 8 + n)
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.idx[k] = (p, v)
        if not self.idx:
            sys.exit(f"no .safetensors in {d}")

    def raw(self, name):
        if name not in self.idx:
            sys.exit(f"missing tensor {name}")
        p, v = self.idx[name]
        f, base = self.files[p]
        f.seek(base + v["data_offsets"][0])
        n = v["data_offsets"][1] - v["data_offsets"][0]
        return np.frombuffer(f.read(n), dtype=np.uint8), v["shape"]

    def deq(self, nm, block=128):
        """FP8 e4m3 weight with one E8M0 scale per block x block tile."""
        W, ws = self.raw(nm + ".weight")
        S, ss = self.raw(nm + ".scale")
        w = narrow_f32(W, "e4m3").reshape(ws)
        s = narrow_f32(S, "e8m0").reshape(ss)
        s = np.where(S.reshape(ss) == 255, np.float32(0), s)   # a NaN scale zeroes its tile
        s = np.repeat(np.repeat(s, block, 0), block, 1)[: ws[0], : ws[1]]
        return w * s

    def mxfp4(self, nm, group=32):
        """MXFP4 expert: E2M1 nibbles, two per byte, one E8M0 scale per group of 32."""
        W, ws = self.raw(nm + ".weight")
        S, ss = self.raw(nm + ".scale")
        rows, pcols = ws
        b = W.reshape(rows, pcols)
        lo, hi = b & 0xF, b >> 4
        # element 2i is the LOW nibble of byte i, 2i+1 the high one.
        codes = np.empty((rows, pcols * 2), dtype=np.uint8)
        codes[:, 0::2], codes[:, 1::2] = lo, hi
        mag = E2M1[codes & 7]
        vals = np.where(codes >= 8, -mag, mag)
        s = narrow_f32(S, "e8m0").reshape(ss)
        s = np.where(S.reshape(ss) == 255, np.float32(0), s)
        return vals * np.repeat(s, group, axis=1)[:, : pcols * 2]

    def bf(self, nm):
        W, ws = self.raw(nm)
        return bf16_f32(W).reshape(ws)

    def rows(self, nm, rows):
        """A few rows of a big 2-D table, without ever materialising it as f32.

        embed.weight and head.weight are 129280 x 4096 -- 1 GB each stored, 2.1 GB each
        as f32. Converting them whole is what makes a verifier freeze a 15 GB laptop
        whose swap lives on the same USB device the shards are being read from."""
        W, ws = self.raw(nm)
        if self.idx[nm][1]["dtype"] == "BF16":
            return bf16_f32(W.view(np.uint16).reshape(ws)[rows].view(np.uint8))
        return W.view(np.float32).reshape(ws)[rows]

    def matvec_big(self, nm, x, chunk=16384):
        """x @ W.T for a big row-major W, a chunk of output rows at a time."""
        W, ws = self.raw(nm)
        bf = self.idx[nm][1]["dtype"] == "BF16"
        w = W.view(np.uint16).reshape(ws) if bf else W.view(np.float32).reshape(ws)
        out = np.empty((x.shape[0], ws[0]), dtype=np.float32)
        for i in range(0, ws[0], chunk):
            blk = w[i:i + chunk]
            blk = bf16_f32(blk.view(np.uint8)).reshape(blk.shape) if bf else blk
            out[:, i:i + chunk] = x @ blk.T
        return out

    def f32(self, nm):
        W, ws = self.raw(nm)
        return W.view(np.float32).reshape(ws)


def rms(x, w, eps=EPS):
    v = (x.astype(np.float64) ** 2).mean(-1, keepdims=True)
    return (w * (x / np.sqrt(v + eps))).astype(np.float32)


def rope_tables(dim, T, theta):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    a = np.outer(np.arange(T), freqs)
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hc_split_sinkhorn(mixes, scale, base, hc=HC, iters=SINKHORN_ITERS, eps=HC_EPS):
    """kernel.py:372. `post` carries a factor of 2 the other two do not."""
    pre = sigmoid(mixes[:hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * sigmoid(mixes[hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[2 * hc:] * scale[2] + base[2 * hc:]).reshape(hc, hc)
    m = comb.max(-1, keepdims=True)
    ex = np.exp(comb - m)
    comb = ex / ex.sum(-1, keepdims=True) + eps
    comb = comb / (comb.sum(0, keepdims=True) + eps)         # one column pass first
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(0, keepdims=True) + eps)
    return pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)


def hc_pre(x, fnw, scale, base):
    """x: [T][hc][E] -> reduced [T][E], plus the post/comb carried to hc_post."""
    T = x.shape[0]
    flat = x.reshape(T, HC * E).astype(np.float32)
    rsq = 1.0 / np.sqrt((flat.astype(np.float64) ** 2).mean(-1) + EPS)
    out = np.zeros((T, E), dtype=np.float32)
    posts, combs = [], []
    for t in range(T):
        mixes = (fnw.astype(np.float64) @ flat[t].astype(np.float64)) * rsq[t]
        pre, post, comb = hc_split_sinkhorn(mixes.astype(np.float32), scale, base)
        out[t] = (pre[:, None] * x[t]).sum(0)
        posts.append(post)
        combs.append(comb)
    return out, np.stack(posts), np.stack(combs)


def hc_post(out, residual, post, comb):
    """y[k] = post[k]*out + sum_j comb[j][k]*residual[j] -- summed over comb's FIRST index."""
    y = post[:, :, None] * out[:, None, :]
    y += np.einsum("tjk,tjd->tkd", comb, residual)
    return y.astype(np.float32)


def hc_head(x, fnw, scale, base):
    """The learned final reduce. NOT a mean over the hc copies."""
    T = x.shape[0]
    flat = x.reshape(T, HC * E).astype(np.float32)
    rsq = 1.0 / np.sqrt((flat.astype(np.float64) ** 2).mean(-1) + EPS)
    out = np.zeros((T, E), dtype=np.float32)
    for t in range(T):
        mixes = (fnw.astype(np.float64) @ flat[t].astype(np.float64)) * rsq[t]
        g = sigmoid(mixes.astype(np.float32) * scale + base) + HC_EPS
        out[t] = (g[:, None] * x[t]).sum(0)
    return out


def swiglu(x, w1, w3, w2):
    gate = (x @ w1.T).astype(np.float32)
    up = (x @ w3.T).astype(np.float32)
    up = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    gate = np.minimum(gate, SWIGLU_LIMIT)
    return (gate * sigmoid(gate) * up) @ w2.T


def moe(sh, stage, x):
    """MoE.forward with sqrtsoftplus scoring, and a bias that shifts SELECTION only.

    Looped over EXPERTS, not over tokens, so each expert is dequantised once per stage
    rather than once per token that picked it -- 3 GB of re-reads per stage otherwise,
    off the same USB device everything else is streaming from."""
    P = f"mtp.{stage}.ffn"
    gate_w = sh.bf(f"{P}.gate.weight")
    bias = sh.f32(f"{P}.gate.bias")
    z = x @ gate_w.T
    scores = np.sqrt(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0))   # softplus, stably
    shifted = scores + bias
    y = np.zeros_like(x, dtype=np.float32)
    picks = {}
    for t in range(x.shape[0]):
        idx = np.argsort(-shifted[t], kind="stable")[:TOPK]
        w = scores[t, idx]
        w = w / w.sum() * ROUTE_SCALE
        for j, ex in enumerate(idx):
            picks.setdefault(int(ex), []).append((t, float(w[j])))
    for ex in sorted(picks):
        q = f"{P}.experts.{ex}"
        w1, w3, w2 = sh.mxfp4(q + ".w1"), sh.mxfp4(q + ".w3"), sh.mxfp4(q + ".w2")
        ts = [t for t, _ in picks[ex]]
        ws = np.array([w for _, w in picks[ex]], dtype=np.float32)
        y[ts] += ws[:, None] * swiglu(x[ts], w1, w3, w2)
    S = f"{P}.shared_experts"
    y += swiglu(x, sh.deq(S + ".w1"), sh.deq(S + ".w3"), sh.deq(S + ".w2"))
    return y.astype(np.float32)


def block_attn(sh, stage, hin, kv_hist, pos, C, S):
    """DSparkAttention for one draft block. `hin` is [k][E]; kv_hist is [n][HD]."""
    P = f"mtp.{stage}.attn"
    wq_a, wq_b = sh.deq(P + ".wq_a"), sh.deq(P + ".wq_b")
    wkv = sh.deq(P + ".wkv")
    wo_a, wo_b = sh.deq(P + ".wo_a"), sh.deq(P + ".wo_b")
    q_norm, kv_norm = sh.bf(P + ".q_norm.weight"), sh.bf(P + ".kv_norm.weight")
    sink = sh.f32(P + ".attn_sink")
    k = hin.shape[0]
    n = pos + 1

    def rope(x, p, inv=False):
        re, im = x[..., 0::2].copy(), x[..., 1::2].copy()
        c, s = C[p], (-S[p] if inv else S[p])
        x[..., 0::2] = re * c - im * s
        x[..., 1::2] = re * s + im * c
        return x

    qr = rms(hin @ wq_a.T, q_norm)
    q = (qr @ wq_b.T).reshape(k, H, HD)
    # A SECOND RMS scaling, per head and with no learned gain.
    q *= (1.0 / np.sqrt((q.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS)).astype(np.float32)
    kvb = rms(hin @ wkv.T, kv_norm)
    for t in range(k):
        p = pos + 1 + t                     # the block sits AFTER the seed's position
        q[t, :, HD - RD:] = rope(q[t, :, HD - RD:], p)
        kvb[t, HD - RD:] = rope(kvb[t, HD - RD:].copy(), p)

    kv = np.concatenate([kv_hist[:n], kvb], 0)
    # The same key set for every query: the window ending at the SEED, then all k block
    # rows. Bidirectional inside the block by construction.
    rows = list(range(max(0, pos - WIN + 1), pos + 1)) + [n + i for i in range(k)]
    sel = kv[rows]

    out = np.zeros((k, E), dtype=np.float32)
    for t in range(k):
        sc = (q[t].astype(np.float64) @ sel.astype(np.float64).T) * (HD ** -0.5)
        m = np.maximum(sc.max(-1), sink)
        ex = np.exp(sc - m[:, None])
        # The sink adds to the DENOMINATOR only -- it has no value row.
        z = ex.sum(-1) + np.exp(sink - m)
        o = ((ex / z[:, None]) @ sel.astype(np.float64)).astype(np.float32)
        o[:, HD - RD:] = rope(o[:, HD - RD:], pos + 1 + t, inv=True)
        go = np.einsum("gd,grd->gr", o.reshape(G, -1), wo_a.reshape(G, OL, -1))
        out[t] = go.reshape(-1) @ wo_b.T
    return out


def reference(shard_dir, T, seed, k):
    sh = Shards(shard_dir)
    n_stages = 0
    while f"mtp.{n_stages}.attn_norm.weight" in sh.idx:
        n_stages += 1
    C, S = rope_tables(RD, T + k + 2, ROPE_THETA)

    # The same deterministic main_hidden the Rust binary builds.
    mh = (np.sin(np.arange(T * TARGETS * E) * 0.001) * 0.05).astype(np.float32)
    mh = mh.reshape(T, TARGETS * E)

    main_proj = sh.deq("mtp.0.main_proj")
    main_norm = sh.bf("mtp.0.main_norm.weight")
    main_x = rms(mh @ main_proj.T, main_norm)

    def rope(x, p):
        re, im = x[..., 0::2].copy(), x[..., 1::2].copy()
        x[..., 0::2] = re * C[p] - im * S[p]
        x[..., 1::2] = re * S[p] + im * C[p]
        return x

    kvs = []
    for s in range(n_stages):
        P = f"mtp.{s}.attn"
        kv = rms(main_x @ sh.deq(P + ".wkv").T, sh.bf(P + ".kv_norm.weight"))
        for p in range(T):
            kv[p, HD - RD:] = rope(kv[p, HD - RD:].copy(), p)   # each row at ITS OWN position
        kvs.append(kv)

    ids = np.array([seed] + [NOISE] * (k - 1), dtype=np.int64)
    x = np.repeat(sh.rows("embed.weight", ids)[:, None, :], HC, axis=1)   # [k][hc][E]

    pos = T - 1
    for s in range(n_stages):
        p = f"mtp.{s}"
        residual = x
        red, post, comb = hc_pre(x, sh.f32(p + ".hc_attn_fn"), sh.f32(p + ".hc_attn_scale"),
                                 sh.f32(p + ".hc_attn_base"))
        a = block_attn(sh, s, rms(red, sh.bf(p + ".attn_norm.weight")), kvs[s], pos, C, S)
        x = hc_post(a, residual, post, comb)

        residual = x
        red, post, comb = hc_pre(x, sh.f32(p + ".hc_ffn_fn"), sh.f32(p + ".hc_ffn_scale"),
                                 sh.f32(p + ".hc_ffn_base"))
        f = moe(sh, s, rms(red, sh.bf(p + ".ffn_norm.weight")))
        x = hc_post(f, residual, post, comb)

    last = n_stages - 1
    hx = hc_head(x, sh.f32(f"mtp.{last}.hc_head_fn"), sh.f32(f"mtp.{last}.hc_head_scale"),
                 sh.f32(f"mtp.{last}.hc_head_base"))
    normed = rms(hx, sh.bf(f"mtp.{last}.norm.weight"))
    if "head.scale" in sh.idx:
        logits = normed.astype(np.float32) @ sh.deq("head").T
    else:
        logits = sh.matvec_big("head.weight", normed.astype(np.float32))

    w1 = sh.bf(f"mtp.{last}.markov_head.markov_w1.weight")
    w2 = sh.bf(f"mtp.{last}.markov_head.markov_w2.weight")
    conf_w = sh.bf(f"mtp.{last}.confidence_head.proj.weight")[0]
    out_ids, conf, prev = [], [], seed
    for t in range(k):
        emb = w1[prev]
        # The chain: position t is biased by the token chosen at t-1.
        nxt = int(np.argmax(logits[t] + emb @ w2.T))   # rank-256 bigram bias
        conf.append(float(np.concatenate([hx[t], emb]).astype(np.float64) @ conf_w.astype(np.float64)))
        out_ids.append(nxt)
        prev = nxt
    return main_x[-1], kvs, hx, np.array(conf, dtype=np.float32), np.array(out_ids)


def cmp(label, a, b, fails, tol=2e-2):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        print(f"  {label:26s} SHAPE {a.shape} vs {b.shape}")
        fails.append(label)
        return
    scale = max(np.abs(b).max(), 1e-9)
    rel = np.abs(a - b).max() / scale
    ok = rel < tol
    print(f"  {label:26s} max|d|/scale = {rel:.2e}   {'ok' if ok else 'MISMATCH'}")
    if not ok:
        fails.append(label)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.strip().splitlines()[-2].strip())
    shard_dir, dump = sys.argv[1], sys.argv[2]
    raw = open(dump, "rb").read()
    n_stages, T, k, e, hd, rank, vocab = struct.unpack("<7I", raw[:28])
    if (e, hd) != (E, HD):
        sys.exit(f"dump says hidden={e} head_dim={hd}, this reference is fixed at {E}/{HD}")
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    off = 28

    def take(n):
        nonlocal off
        v = np.frombuffer(raw, dtype=np.float32, count=n, offset=off)
        off += 4 * n
        return v

    r_main_x = take(E)
    r_kv = [take(T * HD).reshape(T, HD) for _ in range(n_stages)]
    r_hx = take(k * E).reshape(k, E)
    r_conf = take(k)
    r_ids = np.frombuffer(raw, dtype=np.uint32, count=k, offset=off)
    if seed is None:
        sys.exit("pass the same seed token the binary was given as argv[3]")

    print(f"dump: {n_stages} stages, T={T}, block={k}, rank={rank}, vocab={vocab}")
    print(f"rust draft: {list(r_ids)}")
    fails = []
    m_main_x, m_kv, m_hx, m_conf, m_ids = reference(shard_dir, T, seed, k)
    print(f"numpy draft: {list(m_ids)}")

    cmp("main_x", r_main_x, m_main_x, fails)
    for s in range(n_stages):
        cmp(f"stage {s} kv rows", r_kv[s], m_kv[s], fails)
    cmp("hc_head output", r_hx, m_hx, fails)
    cmp("confidence", r_conf, m_conf, fails)
    if list(r_ids) != [int(i) for i in m_ids]:
        print(f"  {'drafted ids':26s} {list(r_ids)} vs {list(m_ids)}   MISMATCH")
        fails.append("drafted ids")
    else:
        print(f"  {'drafted ids':26s} identical                    ok")

    if fails:
        print(f"\nFAIL: {', '.join(fails)}")
        return 1
    print("\nOK: the Rust drafter matches an independent parse of the same weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
