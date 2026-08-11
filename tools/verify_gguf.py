#!/usr/bin/env python
"""
verify_gguf.py - check src/gguf.rs against the official `gguf` package.

WHY THIS EXISTS
    A container reader is all offsets, and an offset error is silent: every tensor after
    the mistake is read from the wrong place, decodes to plausible-looking floats, and
    produces fluent nonsense. The block-size table is worse, because `nbytes` is derived
    from it -- one wrong entry desynchronises the rest of the file.

    Q4_K itself has a specific trap. Its twelve scale bytes hold eight 6-bit scales and
    eight 6-bit mins, and the packing is NOT uniform: sub-blocks 0-3 take clean six-bit
    fields, while 4-7 take four low bits from bytes 8-11 and two high bits from the tops
    of bytes 0-7. Reading all eight uniformly yields in-range scales and is wrong for half
    of every tensor.

    So this compares against `gguf`, the reference implementation from the llama.cpp
    project -- a different implementation by different people from the same spec -- on:
      * every tensor's name, shape, byte offset and byte length
      * the dequantised values of every Q4_K tensor in the file

usage: verify_gguf.py <file.gguf> <path-to-gguf_dump-binary>
"""
from __future__ import annotations

import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


# Kernels src/gguf.rs can dequantise. Anything else in the file is scanned for its
# offsets but not decoded, so listing it here would compare an empty file.
KERNELS = ("Q3_K", "Q4_K", "Q5_K", "Q6_K")


def rust_container(binary, path):
    out = subprocess.run([binary, path], capture_output=True, text=True, check=True).stdout
    tensors = {}
    for line in out.splitlines():
        if not line.startswith("T\t"):
            continue
        _, name, dtype, shape, off, nbytes = line.split("\t")
        tensors[name] = (dtype, eval(shape), int(off), int(nbytes))
    return tensors


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: verify_gguf.py <file.gguf> <gguf_dump binary>")
    path, binary = sys.argv[1], sys.argv[2]

    r = GGUFReader(path)
    mine = rust_container(binary, path)
    print(f"reference: {len(r.tensors)} tensors    rust: {len(mine)} tensors")

    fails = []
    if len(r.tensors) != len(mine):
        fails.append("tensor count")

    q4k = []
    for t in r.tensors:
        if t.name not in mine:
            fails.append(f"{t.name}: missing from the rust scan")
            continue
        dtype, shape, off, nbytes = mine[t.name]
        # GGUF stores dims fastest-varying first; the engine is outer-first, so the rust
        # side reverses. Undo that here rather than assuming either is "right".
        ref_shape = [int(x) for x in reversed(list(t.shape))]
        if dtype != t.tensor_type.name:
            fails.append(f"{t.name}: dtype {dtype} vs {t.tensor_type.name}")
        if shape != ref_shape:
            fails.append(f"{t.name}: shape {shape} vs {ref_shape}")
        if nbytes != int(t.n_bytes):
            fails.append(f"{t.name}: nbytes {nbytes} vs {t.n_bytes}")
        # data_offset in the reference is absolute in the file, same as ours.
        if off != int(t.data_offset):
            fails.append(f"{t.name}: offset {off} vs {t.data_offset}")
        if t.tensor_type.name in KERNELS:
            q4k.append(t)

    print(f"container: {len(mine)} tensors compared, {len(fails)} mismatch(es)")
    for f in fails[:10]:
        print(f"    {f}")

    print(f"\nquantised tensors this build can dequantise ({'/'.join(KERNELS)}): {len(q4k)}")
    if not q4k:
        print("  (none -- this file cannot exercise the kernel)")
    for t in q4k:
        want = dequantize(t.data, t.tensor_type).astype(np.float32).ravel()
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            subprocess.run([binary, path, t.name, tmp.name], capture_output=True, check=True)
            got = np.fromfile(tmp.name, dtype=np.float32)
        if got.shape != want.shape:
            print(f"  {t.name:34s} SHAPE {got.shape} vs {want.shape}")
            fails.append(t.name)
            continue
        # A dequantiser is exact arithmetic on exact integers: this must be bitwise, not
        # close. Any tolerance here would hide a wrong scale for half the sub-blocks.
        bad = int((got != want).sum())
        if bad:
            i = int(np.argmax(got != want))
            print(f"  {t.name:34s} {bad}/{got.size} differ; first at {i}: {got[i]} vs {want[i]}")
            fails.append(t.name)
        else:
            print(f"  {t.name:34s} {t.tensor_type.name:5s} {got.size:>9} values bit-identical")

    # The fused matmuls, on real quantised weights. Not bitwise: the fused form factors
    # the affine term per sub-block while numpy multiplies it out per element, so they sum
    # in different orders. The tolerance is tight enough that dropping the min term (a ~1%
    # error, entirely plausible-looking) fails.
    print(f"\nfused matmul on real weights ({len(q4k)} tensors)")
    for t in q4k:
        rows, k_in = [int(v) for v in reversed(list(t.shape))]
        x = (np.sin(np.arange(k_in) * 0.017) * 0.4).astype(np.float32)
        w = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(rows, k_in)
        want = w.astype(np.float64) @ x.astype(np.float64)
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            subprocess.run([binary, path, t.name, tmp.name, "matmul"],
                           capture_output=True, check=True)
            got = np.fromfile(tmp.name, dtype=np.float32).astype(np.float64)
        scale = np.maximum(np.abs(want), 1e-3)
        rel = float((np.abs(got - want) / scale).max())
        ok = rel < 1e-5
        print(f"  {t.name:34s} {t.tensor_type.name:5s} max rel {rel:.2e}  "
              f"{'ok' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"{t.name} matmul")

    if fails:
        print(f"\nFAIL: {len(fails)} problem(s)")
        return 1
    print("\nOK: container and every dequantised type match the reference exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
