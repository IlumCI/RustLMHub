#!/usr/bin/env bash
# Feed a file as the prompt instead of a command-line string.
#
#   examples/03-long-prompt.sh ~/k3model ~/k3trunk somefile.txt
#
# Use --prompt-file rather than --prompt for anything non-ASCII: argv is re-encoded by
# the shell, so a UTF-8 string can reach the engine as different bytes than you typed.
#
# Be aware that prefill is not chunked and MLA attention is quadratic in sequence length,
# so a long file is slow. See docs/ROADMAP.md.
set -euo pipefail
MODEL="${1:?usage: 03-long-prompt.sh <model_dir> <trunk_dir> <file>}"
TRUNK="${2:?}"
FILE="${3:?}"

wc -c "$FILE"
./bin/k3 "$MODEL" --trunk "$TRUNK" --preset server \
    --tok "$MODEL" --prompt-file "$FILE" --gen 16 --incremental
