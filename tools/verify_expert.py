#!/usr/bin/env python
"""
verify_expert.py - close the last link in the loading chain.

THE CHAIN, AND THE LINK THIS CHECKS
    tools/fetch_mxfp4.py pulled real bytes for
        language_model.model.layers.1.block_sparse_moe.experts.0.w1
    straight out of the released checkpoint over HTTP range reads, and wrote
    fixtures/mxfp4.json. test_ops already proves k3_mxfp4_dequant reproduces that
    fixture bit for bit.

    But that proves the ARITHMETIC, not the READER. The fixture's bytes were located by
    Python, using Python's parse of the header. Nothing yet proves the C code finds the
    same bytes in a local shard file.

    So this compares three things against the fixture, in increasing order of what a
    failure would mean:
      packed bytes  - if these differ, k3_st computed a wrong file offset
      scale bytes   - if only these differ, the scale tensor offset is wrong, which is
                      the subtler bug: weights would be right and magnitudes wrong
      float bits    - the composed result, exact

    Zero tolerance throughout. Both sides are reading identical bytes off identical
    weights; there is nothing here that could legitimately differ.

usage: verify_expert.py [expert_dump.json] [fixtures/mxfp4.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIX_OPS

DEF_FX = os.path.join(FIX_OPS, "mxfp4.json")


def main():
    dump_p = sys.argv[1] if len(sys.argv) > 1 else "expert_dump.json"
    fx_p = sys.argv[2] if len(sys.argv) > 2 else DEF_FX
    d = json.load(open(dump_p, encoding="utf-8"))
    fx = json.load(open(fx_p, encoding="utf-8"))

    print("C dump   : layer %d expert 0 w1, %d rows x %d packed cols"
          % (d["layer"], d["rows"], d["pcols"]))
    print("fixture  : %s" % fx["source"])
    print("           %d rows, %d packed cols, logical width %d, group %d"
          % (fx["rows"], fx["packed_cols"], fx["logical_width"], fx["group_size"]))

    if not fx["source"].endswith("layers.%d.block_sparse_moe.experts.0.w1" % d["layer"]):
        print("\nMISMATCHED SOURCES: the fixture is for %s but the dump is layer %d."
              % (fx["source"], d["layer"]))
        print("Re-run test_expert with the layer the fixture was built from.")
        return 2

    fails = []

    # ---- 1. raw packed bytes: did the reader find the right place in the file? -------
    cp = np.array(d["packed"], dtype=np.uint8)
    fp = np.array(fx["packed"]["data"], dtype=np.uint8)[: cp.size]
    nbad = int((cp != fp).sum())
    if nbad:
        i = int(np.argmax(cp != fp))
        fails.append("packed bytes: %d/%d differ, first at [%d] C=%d fixture=%d"
                     % (nbad, cp.size, i, cp[i], fp[i]))
    else:
        print("\n  packed bytes : %d/%d IDENTICAL  -> k3_st found the right offset"
              % (cp.size, cp.size))

    # ---- 2. scale bytes -------------------------------------------------------------
    cs = np.array(d["scales"], dtype=np.uint8)
    fs = np.array(fx["scales"]["data"], dtype=np.uint8)[: cs.size]
    nbad = int((cs != fs).sum())
    if nbad:
        i = int(np.argmax(cs != fs))
        fails.append("scale bytes: %d/%d differ, first at [%d] C=%d fixture=%d"
                     % (nbad, cs.size, i, cs[i], fs[i]))
    else:
        print("  scale bytes  : %d/%d IDENTICAL  -> the scale tensor offset is right"
              % (cs.size, cs.size))

    # ---- 3. dequantised floats, compared as bit patterns ----------------------------
    cb = np.array(d["bits"], dtype=np.uint32)
    ref = np.array(fx["expected"]["data"], dtype=np.float32)[: cb.size]
    rb = ref.view(np.uint32)
    nbad = int((cb != rb).sum())
    if nbad:
        i = int(np.argmax(cb != rb))
        cf = cb[i : i + 1].view(np.float32)[0]
        fails.append("float bits: %d/%d differ, first at [%d] C=%.9g (0x%08x) "
                     "fixture=%.9g (0x%08x)" % (nbad, cb.size, i, cf, cb[i], ref[i], rb[i]))
    else:
        print("  float bits   : %d/%d IDENTICAL  -> dequantisation is exact end to end"
              % (cb.size, cb.size))

    # ---- the nibble-order trap, checked rather than assumed -------------------------
    swapped = np.array(fx["expected_swapped_nibbles"]["first64"], dtype=np.float32)
    mine = cb[:64].view(np.float32)
    if np.array_equal(mine, swapped):
        fails.append("the output matches the SWAPPED nibble order: every adjacent pair "
                     "of weights is transposed. Statistics look perfect and the model "
                     "is wrong.")
    else:
        agree = int((mine == swapped).sum())
        print("  nibble order : differs from the swapped order in %d of the first 64 "
              "(a coincidental match on %d)" % (64 - agree, agree))

    print()
    if fails:
        print("FAILED:")
        for f in fails:
            print("   " + f)
        return 1
    print("VERIFIED END TO END: bytes read from the local shard by the C reader are "
          "identical to bytes fetched independently from the released checkpoint, and "
          "dequantise bit-exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
