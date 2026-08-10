#!/usr/bin/env python
"""
verify_v4_attn.py - check the Rust DeepSeek-V4 attention against an independent parse.

WHY THIS EXISTS
    src/v4.rs implements layer-0 attention -- the compress_ratio == 0 case, which is
    pure sliding window with no Compressor and no Indexer. Every piece of it is a place
    to be plausibly wrong: the rope pairing, the second per-head RMS scaling on q, the
    attention sink contributing to the denominator only, the inverse rope on the output,
    and the grouped o-LoRA. All of those still produce finite, reasonable-looking
    activations when implemented incorrectly.

    This reparses the same shard with numpy and ml_dtypes -- different implementations,
    written by different people -- follows inference/model.py Attention.forward, and
    compares the two outputs against the scale of the output itself.

WHAT IT DELIBERATELY DOES NOT DO
    inference/model.py calls act_quant on kv[..., :-rope_head_dim], simulating the FP8
    quantisation-aware training. Neither this reference nor src/v4.rs does, so both are
    strictly more precise than the released kernel and agree with each other rather than
    with it. Closing that gap means reproducing the activation quantisation on both
    sides; it is recorded in docs/MULTI_MODEL.md rather than hidden here.

usage: verify_v4_attn.py <shard_dir> <layer> <T> <rust_out.bin>
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np

try:
    import ml_dtypes
except ImportError:
    sys.exit("needs ml_dtypes: pip install ml_dtypes")

E, H, HD, RD, QL, OL, G, WIN, EPS = 4096, 64, 512, 64, 1024, 1024, 8, 128, 1e-6
ROPE_THETA = 10000.0          # compress_ratio == 0 disables YaRN and uses the base theta


def open_shard(d, layer=0):
    """Open the shard that actually holds `layer`.

    The released checkpoint puts layer N in shard N+2, so a single-shard directory was
    enough to verify one layer and this used to take paths[0]. Against the full 48-shard
    download paths[0] is the embedding shard, and every tensor lookup then raises
    KeyError -- which reads as a corrupt download rather than as looking in the wrong
    file. Scan for the marker instead.
    """
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not paths:
        sys.exit(f"no .safetensors in {d}")
    marker = f"layers.{layer}.attn.wq_a.weight"
    for p in paths:
        f = open(p, "rb")
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        if marker in hdr:
            return f, hdr, 8 + n
        f.close()
    sys.exit(f"{marker} is in none of the {len(paths)} shard(s) under {d}")


def make_readers(f, hdr, base):
    def raw(nm):
        v = hdr[nm]
        f.seek(base + v["data_offsets"][0])
        n = v["data_offsets"][1] - v["data_offsets"][0]
        return np.frombuffer(f.read(n), dtype=np.uint8), v["shape"]

    def deq(nm, block=128):
        """FP8 e4m3 weight with one E8M0 scale per block x block tile."""
        W, ws = raw(nm + ".weight")
        S, ss = raw(nm + ".scale")
        w = W.view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(ws)
        s = S.view(ml_dtypes.float8_e8m0fnu).astype(np.float32).reshape(ss)
        s = np.where(S.reshape(ss) == 255, np.float32(0), s)   # NaN scale zeroes its tile
        s = np.repeat(np.repeat(s, block, 0), block, 1)[: ws[0], : ws[1]]
        return w * s

    def bf(nm):
        W, ws = raw(nm)
        return W.view(ml_dtypes.bfloat16).astype(np.float32).reshape(ws)

    def f32(nm):
        W, ws = raw(nm)
        return W.view(np.float32).reshape(ws)

    return deq, bf, f32


def rms(x, w, eps=EPS):
    v = (x.astype(np.float64) ** 2).mean(-1, keepdims=True)
    return (w * (x / np.sqrt(v + eps))).astype(np.float32)


def rope_tables(dim, T, theta):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    a = np.outer(np.arange(T), freqs)
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def load_compress_ref():
    """Reuse the Compressor/Indexer reference that was validated on layers 2 and 3."""
    import importlib.util, os
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("v4c", os.path.join(here, "verify_v4_compress.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def reference(shard_dir, T, layer="0"):
    f, hdr, base = open_shard(shard_dir, layer)
    deq, bf, f32 = make_readers(f, hdr, base)

    P = lambda n: f"layers.{layer}.{n}"
    has_comp = P("attn.compressor.wkv.weight") in hdr
    has_idx = P("attn.indexer.wq_b.weight") in hdr
    ratio = 0 if not has_comp else (4 if has_idx else 128)

    wq_a = deq(P("attn.wq_a"))
    wq_b = deq(P("attn.wq_b"))
    wkv = deq(P("attn.wkv"))
    wo_a = deq(P("attn.wo_a"))
    wo_b = deq(P("attn.wo_b"))
    q_norm = bf(P("attn.q_norm.weight"))
    kv_norm = bf(P("attn.kv_norm.weight"))
    sink = f32(P("attn.attn_sink"))

    v4c = load_compress_ref()
    if ratio:
        # A compressed layer enables YaRN on compress_rope_theta.
        C, S = v4c.yarn_rope(RD, max(T, 2))
    else:
        C, S = rope_tables(RD, max(T, 2), ROPE_THETA)

    def rope(x, pos, inv=False):
        # torch views the last axis as complex pairs: element 2i and 2i+1 rotate together.
        re, im = x[..., 0::2].copy(), x[..., 1::2].copy()
        c, s = C[pos], (-S[pos] if inv else S[pos])
        x[..., 0::2] = re * c - im * s
        x[..., 1::2] = re * s + im * c
        return x

    # The same deterministic input the Rust binary builds.
    x = (np.sin(np.arange(T * E) * 0.001) * 0.05).astype(np.float32).reshape(T, E)

    qr = rms(x @ wq_a.T, q_norm)
    q = (qr @ wq_b.T).reshape(T, H, HD)
    # A SECOND RMS scaling, per head and with no learned gain.
    q *= (1.0 / np.sqrt((q.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS)).astype(np.float32)
    for t in range(T):
        q[t, :, HD - RD:] = rope(q[t, :, HD - RD:], t)

    # One shared KV head serves every query head.
    kv = rms(x @ wkv.T, kv_norm)
    for t in range(T):
        kv[t, HD - RD:] = rope(kv[t, HD - RD:].copy(), t)

    # Compressed blocks are appended AFTER the token window, so they carry offset T.
    rows = [list(range(max(0, t - WIN + 1), t + 1)) for t in range(T)]
    if ratio:
        kvc = v4c.compress(x, bf(P("attn.compressor.wkv.weight")), bf(P("attn.compressor.wgate.weight")),
                           f32(P("attn.compressor.ape")), bf(P("attn.compressor.norm.weight")),
                           ratio, HD, C, S, False)
        nblk = kvc.shape[0]
        if has_idx:
            ikvc = v4c.compress(x, bf(P("attn.indexer.compressor.wkv.weight")),
                                bf(P("attn.indexer.compressor.wgate.weight")),
                                f32(P("attn.indexer.compressor.ape")),
                                bf(P("attn.indexer.compressor.norm.weight")), 4, 128, C, S, True)
            iq = (qr @ deq(P("attn.indexer.wq_b")).T).reshape(T, 64, 128)
            for t in range(T):
                iq[t, :, 128 - RD:] = rope(iq[t, :, 128 - RD:], t)
            iq = v4c.hadamard(iq)
            wts = (x @ bf(P("attn.indexer.weights_proj.weight")).T) * (128 ** -0.5 * 64 ** -0.5)
            isc = np.einsum("thd,bd->thb", iq.astype(np.float64), ikvc.astype(np.float64))
            isc = (np.maximum(isc, 0) * wts[:, :, None]).sum(1)
            vis = (np.arange(1, T + 1) // ratio)[:, None] > np.arange(ikvc.shape[0])[None, :]
            isc = np.where(vis, isc, -np.inf)
            for t in range(T):
                order = sorted(range(ikvc.shape[0]), key=lambda b: (-isc[t, b], b))
                rows[t] += [T + b for b in order[: min(512, ikvc.shape[0])] if vis[t, b]]
        else:
            for t in range(T):
                rows[t] += [T + b for b in range(nblk) if b < (t + 1) // ratio]
        kv = np.concatenate([kv, kvc], axis=0)

    scale = HD ** -0.5
    o = np.zeros((T, H, HD), dtype=np.float32)
    for t in range(T):
        idx = np.array(rows[t], dtype=int)
        sc = (q[t].astype(np.float64) @ kv[idx].astype(np.float64).T) * scale
        m = np.maximum(sc.max(1), sink.astype(np.float64))
        e = np.exp(sc - m[:, None])
        z = e.sum(1) + np.exp(sink - m)                       # sink: denominator only
        o[t] = (e / z[:, None]).astype(np.float32) @ kv[idx]
    for t in range(T):
        o[t, :, HD - RD:] = rope(o[t, :, HD - RD:], t, inv=True)   # de-rotate the output

    gsz = H * HD // G
    oa = wo_a.reshape(G, OL, gsz)                             # grouped o-LoRA
    og = np.einsum("tgd,grd->tgr", o.reshape(T, G, gsz).astype(np.float64), oa.astype(np.float64))
    return (og.reshape(T, G * OL) @ wo_b.T.astype(np.float64)).astype(np.float32)


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__.strip().splitlines()[-1])
    shard_dir, layer, T, rust_bin = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

    want = reference(shard_dir, T, layer)
    got = np.fromfile(rust_bin, dtype=np.uint32).view(np.float32).reshape(T, E)

    d = np.abs(got - want)
    scale = np.abs(want).max()
    print(f"DeepSeek-V4 layer {layer} attention, T={T}, {E} dims")
    print(f"  numpy range   [{want.min():.6f}, {want.max():.6f}]")
    print(f"  rust  range   [{got.min():.6f}, {got.max():.6f}]")
    print(f"  max |diff|    {d.max():.3e}")
    print(f"  mean |diff|   {d.mean():.3e}")
    # Relative to the OUTPUT SCALE, not to each element: dividing by a near-zero element
    # reports a huge relative error for an absolute difference at f32 epsilon.
    print(f"  max |diff| / max |out| = {d.max() / scale:.3e}")
    big = np.abs(want) > 0.1 * scale
    print(f"  max relative error on the {big.sum()} elements above 10% of scale: "
          f"{(d[big] / np.abs(want[big])).max():.3e}")

    ok = d.max() / scale < 1e-4
    print("\n" + ("VERIFIED: the Rust attention matches an independent numpy parse "
                  "to f32 accumulation noise." if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
