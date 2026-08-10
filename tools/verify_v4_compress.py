#!/usr/bin/env python
"""
verify_v4_compress.py - check the Rust Compressor and Indexer against numpy.

WHY THIS EXISTS
    DeepSeek-V4's compressed layers are where the architecture is least like anything
    else in this repository, and every piece has a plausible wrong reading:

      - the gate softmax runs over the SLOT axis per channel, not over channels;
      - the APE is added to the gate, never to the value;
      - at ratio 4 the windows OVERLAP: a block takes the upper half of its own tokens'
        dims and the lower half of the PREVIOUS block's, so block 0's lower slots are
        masked rather than zero-weighted;
      - a compressed block sits at rope position b * ratio, not b;
      - the Indexer rotates BOTH its query and its compressed kv by a Hadamard
        transform, and applies relu BEFORE weighting and summing over heads;
      - a block is causal for token t only when b < (t + 1) // ratio.

    Each of those still produces finite, plausible activations when done wrong.

WHAT IT DELIBERATELY DOES NOT DO
    inference/model.py simulates FP4 on the Indexer's q and kv (fp4_act_quant) and FP8
    on the plain compressor's non-rope dims. Neither this reference nor src/v4.rs does.

    That matters more here than it does for attention. The Indexer's output is a
    DISCRETE top-k selection, so two nearly-equal scores can be ordered differently by
    a quantisation this port does not perform. Score agreement is checked numerically;
    index agreement is reported separately and is expected to hold only where the score
    gaps are comfortable.

usage: verify_v4_compress.py <shard_dir> <layer> <T> <out_prefix>
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

import numpy as np

try:
    import ml_dtypes
except ImportError:
    sys.exit("needs ml_dtypes: pip install ml_dtypes")

HIDDEN, HD, RD, IHD, QL, NH = 4096, 512, 64, 128, 1024, 64
EPS = 1e-6
COMPRESS_ROPE_THETA, ORIG_SEQ, FACTOR, BFAST, BSLOW = 160000.0, 65536, 16.0, 32.0, 1.0


def readers(shard_dir, layer=2):
    paths = sorted(glob.glob(os.path.join(shard_dir, "*.safetensors")))
    # The full download starts with the embedding shard, so paths[0] holds none of this
    # layer's tensors; pick the shard that actually does.
    marker = f"layers.{layer}.attn.compressor.wkv.weight"
    for _p in paths:
        _f = open(_p, "rb")
        _n = struct.unpack("<Q", _f.read(8))[0]
        if marker in json.loads(_f.read(_n)):
            _f.close()
            paths = [_p]
            break
        _f.close()
    f = open(paths[0], "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n

    def raw(nm):
        v = hdr[nm]
        f.seek(base + v["data_offsets"][0])
        return np.frombuffer(f.read(v["data_offsets"][1] - v["data_offsets"][0]), dtype=np.uint8), v["shape"]

    def bf(nm):
        W, ws = raw(nm)
        return W.view(ml_dtypes.bfloat16).astype(np.float32).reshape(ws)

    def f32(nm):
        W, ws = raw(nm)
        return W.view(np.float32).reshape(ws)

    def deq(nm, block=128):
        W, ws = raw(nm + ".weight")
        S, ss = raw(nm + ".scale")
        w = W.view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(ws)
        s = S.view(ml_dtypes.float8_e8m0fnu).astype(np.float32).reshape(ss)
        s = np.where(S.reshape(ss) == 255, np.float32(0), s)
        s = np.repeat(np.repeat(s, block, 0), block, 1)[: ws[0], : ws[1]]
        return w * s

    return hdr, bf, f32, deq


def yarn_rope(dim, T):
    freqs = 1.0 / (COMPRESS_ROPE_THETA ** (np.arange(0, dim, 2) / dim))
    corr = lambda rot: dim * np.log(ORIG_SEQ / (rot * 2 * np.pi)) / (2 * np.log(COMPRESS_ROPE_THETA))
    low, high = max(np.floor(corr(BFAST)), 0), min(np.ceil(corr(BSLOW)), dim - 1)
    if high == low:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    smooth = 1 - ramp
    freqs = freqs / FACTOR * (1 - smooth) + freqs * smooth
    a = np.outer(np.arange(T), freqs)
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def rope(x, C, S, pos, inv=False):
    re, im = x[..., 0::2].copy(), x[..., 1::2].copy()
    c, s = C[pos], (-S[pos] if inv else S[pos])
    x[..., 0::2] = re * c - im * s
    x[..., 1::2] = re * s + im * c
    return x


def hadamard(x):
    """FWHT over the last axis, scaled by n^-0.5."""
    n = x.shape[-1]
    y = x.astype(np.float64).copy()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        y = y.reshape(*y.shape[:-3], n)
        h *= 2
    return (y * n ** -0.5).astype(np.float32)


def rms(x, w):
    v = (x.astype(np.float64) ** 2).mean(-1, keepdims=True)
    return (w * (x / np.sqrt(v + EPS))).astype(np.float32)


def compress(x, wkv, wgate, ape, norm, ratio, hd, C, S, rotate):
    T = x.shape[0]
    overlap = ratio == 4
    cd = (2 if overlap else 1) * hd
    cutoff = T - T % ratio
    nblk = cutoff // ratio
    if nblk == 0:
        return np.zeros((0, hd), dtype=np.float32)

    kv = (x @ wkv.T)[:cutoff].reshape(nblk, ratio, cd)
    sc = (x @ wgate.T)[:cutoff].reshape(nblk, ratio, cd) + ape        # APE on the GATE

    if overlap:
        k2 = np.zeros((nblk, 2 * ratio, hd), dtype=np.float32)
        s2 = np.full((nblk, 2 * ratio, hd), -np.inf, dtype=np.float32)
        k2[:, ratio:] = kv[:, :, hd:]          # own tokens, upper half of the dims
        s2[:, ratio:] = sc[:, :, hd:]
        k2[1:, :ratio] = kv[:-1, :, :hd]       # PREVIOUS block, lower half
        s2[1:, :ratio] = sc[:-1, :, :hd]
        kv, sc = k2, s2

    m = sc.max(axis=1, keepdims=True)
    e = np.where(np.isneginf(sc), 0.0, np.exp(sc - m))
    out = (kv * (e / e.sum(axis=1, keepdims=True))).sum(axis=1).astype(np.float32)

    out = rms(out, norm)
    for b in range(nblk):
        out[b, hd - RD:] = rope(out[b, hd - RD:].copy(), C, S, b * ratio)   # position b*ratio
    if rotate:
        out = hadamard(out)
    return out


def main():
    if len(sys.argv) < 5:
        sys.exit("usage: verify_v4_compress.py <shard_dir> <layer> <T> <out_prefix>")
    shard_dir, layer, T, pfx = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    hdr, bf, f32, deq = readers(shard_dir, layer)
    p = lambda n: f"layers.{layer}.{n}"
    has_idx = p("attn.indexer.wq_b.weight") in hdr
    ratio = 4 if has_idx else 128

    C, S = yarn_rope(RD, max(T, 2))
    x = (np.sin(np.arange(T * HIDDEN) * 0.001) * 0.05).astype(np.float32).reshape(T, HIDDEN)

    want = compress(x, bf(p("attn.compressor.wkv.weight")), bf(p("attn.compressor.wgate.weight")),
                    f32(p("attn.compressor.ape")), bf(p("attn.compressor.norm.weight")),
                    ratio, HD, C, S, False)
    got = np.fromfile(pfx + "_kvc.bin", dtype=np.uint32).view(np.float32).reshape(-1, HD)
    print(f"layer {layer}: ratio {ratio}, T={T} -> {want.shape[0]} blocks")
    ok = True
    if want.size:
        d = np.abs(got - want)
        sc = np.abs(want).max()
        print(f"  compressor   max |diff| {d.max():.3e}   / max|out| {d.max()/sc:.3e}")
        ok &= d.max() / sc < 1e-4

    if has_idx:
        ikvc = compress(x, bf(p("attn.indexer.compressor.wkv.weight")),
                        bf(p("attn.indexer.compressor.wgate.weight")),
                        f32(p("attn.indexer.compressor.ape")),
                        bf(p("attn.indexer.compressor.norm.weight")),
                        4, IHD, C, S, True)
        gi = np.fromfile(pfx + "_ikvc.bin", dtype=np.uint32).view(np.float32).reshape(-1, IHD)
        d = np.abs(gi - ikvc)
        sc = np.abs(ikvc).max()
        print(f"  indexer kv   max |diff| {d.max():.3e}   / max|out| {d.max()/sc:.3e}   (Hadamard-rotated)")
        ok &= d.max() / sc < 1e-4

        qr = rms(x @ deq(p("attn.wq_a")).T, bf(p("attn.q_norm.weight")))
        q = (qr @ deq(p("attn.indexer.wq_b")).T).reshape(T, NH, IHD)
        for t in range(T):
            q[t, :, IHD - RD:] = rope(q[t, :, IHD - RD:], C, S, t)
        q = hadamard(q)
        wts = (x @ bf(p("attn.indexer.weights_proj.weight")).T) * (IHD ** -0.5 * NH ** -0.5)

        nblk = ikvc.shape[0]
        score = np.einsum("thd,bd->thb", q.astype(np.float64), ikvc.astype(np.float64))
        score = (np.maximum(score, 0) * wts[:, :, None]).sum(1)          # relu, then weight, then sum heads
        vis = (np.arange(1, T + 1) // ratio)[:, None] > np.arange(nblk)[None, :]
        score = np.where(vis, score, -np.inf)
        want_idx = []
        for t in range(T):
            order = sorted(range(nblk), key=lambda b: (-score[t, b], b))
            want_idx.append([b if vis[t, b] else -1 for b in order[: min(512, nblk)]])
        got_idx = np.fromfile(pfx + "_idx.bin", dtype=np.uint32).view(np.float32).astype(int).reshape(T, -1)
        same = sum(1 for t in range(T) if list(got_idx[t]) == want_idx[t])
        print(f"  indexer topk {same}/{T} tokens select the identical block order")
        if same != T:
            for t in range(T):
                if list(got_idx[t]) != want_idx[t]:
                    print(f"    t={t}: rust {list(got_idx[t])}  numpy {want_idx[t]}")
        ok &= same == T

    print("\n" + ("VERIFIED: Compressor and Indexer match an independent numpy parse."
                  if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
