#!/usr/bin/env python
"""
verify_st.py - check the C safetensors reader against an independent Python parse.

WHY THIS EXISTS
    k3_st.c contains a hand-written JSON scanner. Hand-written parsers are the classic
    place to hide an off-by-one that never crashes: a data_offsets base computed as
    8 + hlen - 1, a shape dimension dropped, a dtype mapped to the wrong width. Every
    one of those still returns bytes, so the model still runs and is simply wrong.

    This reparses the same files with Python's json (a different implementation, written
    by different people) and compares, for EVERY tensor: dtype, shape, absolute byte
    offset, byte count. Then, for any tensor the C side dumped values for, it re-reads
    the raw bytes with numpy and compares the widened float32 BIT PATTERNS exactly.

    Bit patterns rather than approximate equality, because widening bf16 or f16 to f32
    is a lossless integer operation. There is no tolerance to negotiate. If a single bit
    differs, the converter is wrong.

usage: verify_st.py <shard_dir> [st_index.json] [st_values.json]
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

WIDTH = {"U8": 1, "BF16": 2, "F16": 2, "F32": 4}


def parse_shard(path):
    """Independent parse: read the length prefix, json.loads the header, return
    name -> (dtype, shape, absolute_offset, nbytes)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
        size = os.path.getsize(path)
    base = 8 + n
    out = {}
    for name, e in hdr.items():
        if name == "__metadata__":
            continue
        s, t = e["data_offsets"]
        out[name] = (e["dtype"], tuple(e["shape"]), base + s, t - s)
    return out, base, size


def widen(raw: bytes, dtype: str) -> np.ndarray:
    """The reference widening, done with numpy rather than by hand."""
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == "U8":
        return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    if dtype == "BF16":
        # bf16 IS the top 16 bits of an f32, so widening is a shift. Building it this
        # way rather than via any bfloat16 library keeps the reference independent of
        # whatever the C code does.
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    raise ValueError(dtype)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    idx_path = sys.argv[2] if len(sys.argv) > 2 else "st_index.json"
    val_path = sys.argv[3] if len(sys.argv) > 3 else "st_values.json"

    shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    print("verifying %d shard(s) in %s" % (len(shards), d))

    truth = {}
    for sh in shards:
        t, base, size = parse_shard(os.path.join(d, sh))
        print("  %-40s header ends at %-10d  %d tensors  %.2f GB"
              % (sh, base, len(t), size / 1e9))
        for k, v in t.items():
            if k in truth:
                print("     DUPLICATE across shards: %s" % k)
            truth[k] = v + (sh,)

    got = json.load(open(idx_path, encoding="utf-8"))["tensors"]
    print("\npython sees %d tensors, C reported %d" % (len(truth), len(got)))

    fails, checked = [], 0
    for name, (dt, shape, off, nb, sh) in truth.items():
        c = got.get(name)
        if c is None:
            fails.append("%s: MISSING from the C index" % name)
            continue
        checked += 1
        if c["dtype"] != dt:
            fails.append("%s: dtype C=%s python=%s" % (name, c["dtype"], dt))
        if tuple(c["shape"]) != shape:
            fails.append("%s: shape C=%s python=%s" % (name, tuple(c["shape"]), shape))
        if c["off"] != off:
            fails.append("%s: OFFSET C=%d python=%d (delta %d)"
                         % (name, c["off"], off, c["off"] - off))
        if c["nbytes"] != nb:
            fails.append("%s: nbytes C=%d python=%d" % (name, c["nbytes"], nb))
        if c["shard"] != sh:
            fails.append("%s: shard C=%s python=%s" % (name, c["shard"], sh))
        # Independent cross-check of the size invariant itself.
        want = int(np.prod(shape)) * WIDTH[dt] if shape else WIDTH[dt]
        if nb != want:
            fails.append("%s: header is self-inconsistent, %d bytes vs shape implying %d"
                         % (name, nb, want))
    for name in got:
        if name not in truth:
            fails.append("%s: C invented a tensor python does not see" % name)

    print("compared %d tensors on dtype, shape, absolute offset, nbytes and shard"
          % checked)

    # ---- values -------------------------------------------------------------------
    nval = 0
    if os.path.exists(val_path):
        vals = json.load(open(val_path, encoding="utf-8"))
        print("\nchecking %d dumped tensor(s), float32 bit patterns, zero tolerance"
              % len(vals))
        for name, v in vals.items():
            dt, shape, off, nb, sh = truth[name]
            with open(os.path.join(d, sh), "rb") as f:
                f.seek(off)
                raw = f.read(nb)
            ref = widen(raw, dt)
            if len(ref) != v["n"]:
                fails.append("%s: element count C=%d python=%d" % (name, v["n"], len(ref)))
                continue
            rb = ref.view(np.uint32)
            head = np.array(v["first_bits"], dtype=np.uint32)
            tail = np.array(v["last_bits"], dtype=np.uint32)
            ts = v["tail_start"]
            nbad_h = int((rb[:len(head)] != head).sum())
            nbad_t = int((rb[ts:ts + len(tail)] != tail).sum())
            nval += 1
            if nbad_h or nbad_t:
                i = int(np.argmax(rb[:len(head)] != head)) if nbad_h else 0
                fails.append("%s: %d/%d head and %d/%d tail bit mismatches; "
                             "first at [%d] C=0x%08x python=0x%08x (%g vs %g)"
                             % (name, nbad_h, len(head), nbad_t, len(tail), i,
                                head[i], rb[i],
                                head[i:i + 1].view(np.float32)[0],
                                ref[i]))
            else:
                print("  %-72s %-5s %8d elems  EXACT (%d head + %d tail bits)"
                      % (name[:72], dt, len(ref), len(head), len(tail)))
    else:
        print("\n(no %s, skipping the value check)" % val_path)

    print()
    if fails:
        print("FAILED: %d problem(s)" % len(fails))
        for f in fails[:40]:
            print("   " + f)
        if len(fails) > 40:
            print("   ... and %d more" % (len(fails) - 40))
        return 1
    print("VERIFIED: the C index matches an independent Python parse on all %d tensors, "
          "and all %d dumped tensors are bit-exact." % (checked, nval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
