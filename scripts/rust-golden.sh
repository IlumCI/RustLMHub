#!/bin/sh
# Golden gate: step 8 of docs/RUST_PORT.md's ladder, on a tiny REAL checkpoint.
#
# Builds a BF16 trunk + MXFP4 expert checkpoint with HF names and a torch reference, then
# checks four things the module fixtures cannot:
#
#   1. C and Rust dump byte-identical logits over the whole stack.
#   2. The Rust dump matches the torch reference elementwise.
#   3. Greedy decode and incremental decode agree with the C engine token for token.
#   4. Output is identical across trunk budgets, from fully streamed to fully pinned.
#
# (1) is what caught w2/w3 swapped in the expert name table: the argmax still agreed and
# the correlation was 0.988, so only an elementwise comparison against an independent
# implementation would have found it. (4) is the project's headline claim -- memory buys
# speed, not capability -- and it is the one an async reader writing over a live slot
# breaks silently.
set -eu
OUT="${1:-build/golden}"
IDS=3,7,11,5,9
GEN=8

if ! python3 -c "import torch, safetensors" 2>/dev/null; then
    echo "  NOT RUN  rust-golden needs torch + safetensors to build the tiny checkpoint"
    exit 0
fi
mkdir -p "$OUT"
python3 tools/make_tiny_checkpoint.py "$OUT/tiny" --prompt-ids "$IDS" >/dev/null
python3 tools/pack_trunk.py "$OUT/tiny" "$OUT/trunk" 13 >/dev/null

K3=./bin/k3
RS=./rust/target/release/k3_run

"$K3" "$OUT/tiny" --ids "$IDS" --gen 1 --cache-gb 0.5 --dump-logits "$OUT/c.bin"  >/dev/null 2>&1
"$RS" "$OUT/tiny" --ids "$IDS" --gen 1 --cache-gb 0.5 --dump-logits "$OUT/rs.bin" >/dev/null 2>&1
if cmp -s "$OUT/c.bin" "$OUT/rs.bin"; then
    echo "  ok    logits byte-identical, C vs Rust, full stack"
else
    echo "  FAIL  the two engines' logits disagree"; exit 1
fi
python3 tools/cmp_logits.py "$OUT/rs.bin" "$OUT/tiny/ref_logits.json" >"$OUT/cmp.txt" 2>&1 || {
    tail -3 "$OUT/cmp.txt"; echo "  FAIL  Rust logits do not match the torch reference"; exit 1; }
echo "  ok    Rust logits match the torch reference ($(grep 'relative to max' "$OUT/cmp.txt" | awk '{print $5}'))"

"$K3" "$OUT/tiny" --ids "$IDS" --gen $GEN --cache-gb 0.5 --out "$OUT/c.json" >/dev/null 2>&1
WANT=$(sed 's/.*"full_ids":\[//; s/\].*//' "$OUT/c.json")
check() {
    got=$(cat "$2")
    if [ "$got" = "$WANT" ]; then echo "  ok    $1"; else
        echo "  FAIL  $1"; echo "        want $WANT"; echo "        got  $got"; exit 1; fi
}
"$RS" "$OUT/tiny" --ids "$IDS" --gen $GEN --cache-gb 0.5 --out "$OUT/full.txt" >/dev/null 2>&1
check "greedy decode agrees with the C engine, $GEN tokens" "$OUT/full.txt"
"$RS" "$OUT/tiny" --ids "$IDS" --gen $GEN --incremental --cache-gb 0.5 --out "$OUT/inc.txt" >/dev/null 2>&1
check "incremental decode agrees (KV cache + carried KDA state)" "$OUT/inc.txt"

# 0.0005 GB pins nothing, so every layer streams through the ring and the asynchronous
# reader runs; 0.05 GB pins all 13. Identical ids at both ends is the memory-ladder claim.
for gb in 0.0005 0.001 0.002 0.05; do
    "$RS" "$OUT/tiny" --trunk "$OUT/trunk" --ids "$IDS" --gen $GEN --cache-gb 0.5 \
          --trunk-gb $gb --out "$OUT/t.txt" >/dev/null 2>&1
    check "trunk budget $gb GB: identical ids" "$OUT/t.txt"
done
