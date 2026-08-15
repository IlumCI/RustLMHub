#!/usr/bin/env python3
"""Split one GGUF into N parts, the way llama-gguf-split does.

WHY THIS EXISTS
    `fetch.rs` learned to download split checkpoints on the claim that the READER already
    handled them -- `St::open` sorts a directory of .gguf files, scans each with its own
    shard index and merges the tensor tables, taking metadata from the first. That claim
    was inferred from the code, not observed, and "the loader silently binds half a model"
    is precisely the failure this project keeps refusing to accept on inference.

    Every split checkpoint worth testing against is hundreds of gigabytes. So this makes a
    three-megabyte one instead, and the whole chain -- inspect, load, forward pass -- can
    be run on it in a second.

LAYOUT
    Each part is a complete, independently-parseable GGUF: its own magic, its own metadata
    and its own tensor table listing ONLY the tensors it carries, with offsets relative to
    its own data section. Part one additionally keeps all the original metadata, which is
    why it is the part that can be probed for architecture and hparams.

    `split.no`, `split.count` and `split.tensors.count` are added to every part, matching
    llama.cpp's convention.

    usage: split_gguf.py <in.gguf> <out-dir> [n_parts]
"""

import os
import struct
import sys

T_U32, T_STR, T_ARR, T_U16 = 4, 8, 9, 2


def rd_str(d, o):
    (n,) = struct.unpack_from("<Q", d, o)
    o += 8
    return d[o : o + n].decode("utf8", "replace"), o + n


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def rd_val(d, o, t):
    if t == T_STR:
        return rd_str(d, o)
    if t == T_ARR:
        (et,) = struct.unpack_from("<I", d, o)
        o += 4
        (n,) = struct.unpack_from("<Q", d, o)
        o += 8
        vs = []
        for _ in range(n):
            v, o = rd_val(d, o, et)
            vs.append(v)
        return (et, vs), o
    f = SCALAR[t]
    (v,) = struct.unpack_from("<" + f, d, o)
    return v, o + struct.calcsize("<" + f)


def wr_str(s):
    b = s.encode("utf8")
    return struct.pack("<Q", len(b)) + b


def wr_val(t, v):
    if t == T_STR:
        return wr_str(v)
    if t == T_ARR:
        et, vs = v
        out = struct.pack("<I", et) + struct.pack("<Q", len(vs))
        for x in vs:
            out += wr_val(et, x)
        return out
    return struct.pack("<" + SCALAR[t], v)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, outdir = sys.argv[1], sys.argv[2]
    nparts = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    d = open(src, "rb").read()
    magic, ver = struct.unpack_from("<II", d, 0)
    assert magic == 0x46554747, f"not a gguf: {magic:#x}"
    o = 8
    n_tensors, n_kv = struct.unpack_from("<QQ", d, o)
    o += 16

    kvs = []
    for _ in range(n_kv):
        k, o = rd_str(d, o)
        (t,) = struct.unpack_from("<I", d, o)
        o += 4
        v, o = rd_val(d, o, t)
        kvs.append((k, t, v))
    align = next((v for k, _, v in kvs if k == "general.alignment"), 32)

    tensors = []
    for _ in range(n_tensors):
        name, o = rd_str(d, o)
        (nd,) = struct.unpack_from("<I", d, o)
        o += 4
        dims = struct.unpack_from("<" + "Q" * nd, d, o)
        o += 8 * nd
        (ty,) = struct.unpack_from("<I", d, o)
        o += 4
        (off,) = struct.unpack_from("<Q", d, o)
        o += 8
        tensors.append([name, list(dims), ty, off])

    data_start = (o + align - 1) // align * align
    # Byte length of each tensor, from the gap to the next one (the file's own accounting,
    # so this needs no block-size table and cannot disagree with the writer).
    ends = sorted([t[3] for t in tensors]) + [len(d) - data_start]
    size_of = {}
    for t in tensors:
        nxt = next(e for e in ends if e > t[3])
        size_of[t[0]] = nxt - t[3]

    os.makedirs(outdir, exist_ok=True)
    per = (len(tensors) + nparts - 1) // nparts
    groups = [tensors[i : i + per] for i in range(0, len(tensors), per)] or [[]]
    nparts = len(groups)
    base = os.path.basename(src).removesuffix(".gguf")

    for i, group in enumerate(groups):
        # Part one keeps the full metadata; later parts carry only the split keys, which is
        # what makes part one the only one worth probing for architecture.
        meta = list(kvs) if i == 0 else []
        meta += [
            ("split.no", T_U16, i),
            ("split.count", T_U16, nparts),
            ("split.tensors.count", T_U32, len(tensors)),
        ]
        head = struct.pack("<II", magic, ver) + struct.pack("<QQ", len(group), len(meta))
        for k, t, v in meta:
            head += wr_str(k) + struct.pack("<I", t) + wr_val(t, v)

        # Offsets are relative to THIS part's data section, so they must be recomputed.
        local, cur = [], 0
        for name, dims, ty, off in group:
            local.append((name, dims, ty, cur, off, size_of[name]))
            cur = (cur + size_of[name] + align - 1) // align * align

        tbl = b""
        for name, dims, ty, newoff, _, _ in local:
            tbl += wr_str(name) + struct.pack("<I", len(dims))
            tbl += struct.pack("<" + "Q" * len(dims), *dims)
            tbl += struct.pack("<I", ty) + struct.pack("<Q", newoff)

        body = bytearray()
        for _, _, _, newoff, oldoff, nbytes in local:
            body.extend(b"\0" * (newoff - len(body)))
            body.extend(d[data_start + oldoff : data_start + oldoff + nbytes])

        hdr = head + tbl
        pad = (len(hdr) + align - 1) // align * align - len(hdr)
        path = os.path.join(outdir, f"{base}-{i + 1:05d}-of-{nparts:05d}.gguf")
        with open(path, "wb") as f:
            f.write(hdr + b"\0" * pad + bytes(body))
        print(f"  {os.path.basename(path)}  {len(group)} tensors, {os.path.getsize(path)} B")

    print(f"{nparts} parts from {len(tensors)} tensors in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
