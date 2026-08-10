#!/usr/bin/env bash
# Run the same prompt at three very different memory budgets and diff the output.
#
#   examples/02-memory-budgets.sh ~/k3model ~/k3trunk
#
# The point of this example is that the token ids come out IDENTICAL. Memory buys speed,
# not capability.
set -euo pipefail
MODEL="${1:?usage: 02-memory-budgets.sh <model_dir> <trunk_dir>}"
TRUNK="${2:?}"
IDS=1008,10484,318,15383,387          # "The capital of France is"
OUT=$(mktemp -d)

for preset in laptop desktop server; do
    echo "=== $preset ==="
    ./bin/k3 "$MODEL" --trunk "$TRUNK" --preset "$preset" \
        --ids "$IDS" --gen 8 --incremental --out "$OUT/$preset.json" \
        | grep -E 's/token average|PEAK RSS'
done

echo
echo "=== token ids at each budget ==="
for preset in laptop desktop server; do
    printf '%-12s ' "$preset"
    python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['generated_ids'])" \
        "$OUT/$preset.json"
done
echo
echo "If those three lines are not identical, that is a bug worth reporting."
rm -rf "$OUT"
