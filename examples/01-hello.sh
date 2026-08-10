#!/usr/bin/env bash
# The smallest useful run: text in, text out.
#
#   examples/01-hello.sh ~/k3model ~/k3trunk
#
# Note the model has no chat template, so it CONTINUES the prompt rather than replying
# to it. "Hello! My name is" gets completed, not answered. That is expected.
set -euo pipefail
MODEL="${1:?usage: 01-hello.sh <model_dir> <trunk_dir>}"
TRUNK="${2:?}"

./bin/k3 "$MODEL" \
    --trunk "$TRUNK" --preset server \
    --tok "$MODEL" \
    --prompt "Hello! My name is" \
    --gen 24 --incremental
