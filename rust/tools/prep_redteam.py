#!/usr/bin/env python3
"""Prepare Euroswarms/redteaming-man for a streamed LoRA fine-tune.

The raw set (hack.jsonl, 4454 rows) near-duplicates every response across instruction
framings: the SAME assistant answer appears under a `concept_explain` and a
`technique_howto` phrasing (and occasionally more). Training on all 4454 would spend half
the compute re-teaching identical content, so this dedupes by response.

Policy, deliberately conservative:
  * group rows by exact response text;
  * keep at most `--framings` rows per group (default 2), preferring DISTINCT `row_kind`s
    so the kept pair is "same knowledge, two instruction phrasings" -- which helps the model
    follow the instruction rather than memorise one surface form, the one thing the near-dup
    structure is actually good for;
  * drop the 3rd+ framing of a response entirely.

Output is a flat JSONL of {id, system, user, response, row_kind, category}. Tokenisation and
the chat-template render happen in Rust against the MODEL's own tokenizer (see
`train::data`), never here -- a Python tokenizer would not match the GGUF's vocab, and a
mismatch is exactly the silent-wrong failure this project refuses.

Usage:
    prep_redteam.py hack.jsonl prepared.jsonl [--framings 2] [--seed 0]
"""
import sys, json, argparse, hashlib
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--framings", type=int, default=2, help="max rows kept per unique response")
    ap.add_argument("--seed", type=int, default=0, help="tie-break ordering, for reproducibility")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src)]
    # Group by exact response text.
    groups = defaultdict(list)
    for r in rows:
        groups[r["response"]].append(r)

    # Deterministic order: sort group keys, and within a group order by row_kind then a
    # stable hash so the same rows are kept every run regardless of input order.
    def md(r):
        m = r["metadata"]
        return json.loads(m) if isinstance(m, str) else m

    kept = []
    for resp in sorted(groups):
        members = groups[resp]
        # Prefer distinct row_kinds: greedily take one of each kind first, then fill.
        by_kind = defaultdict(list)
        for r in members:
            by_kind[md(r).get("row_kind", "")].append(r)
        ordered = []
        # round-robin across kinds so the first `framings` are as diverse as possible
        kinds = sorted(by_kind)
        while any(by_kind[k] for k in kinds) and len(ordered) < len(members):
            for k in kinds:
                if by_kind[k]:
                    # stable within-kind order by content hash
                    by_kind[k].sort(key=lambda r: hashlib.sha1((r["user"]).encode()).hexdigest())
                    ordered.append(by_kind[k].pop(0))
        for r in ordered[: args.framings]:
            m = md(r)
            kept.append(
                {
                    "id": hashlib.sha1((r["system"] + "\x00" + r["user"]).encode()).hexdigest()[:16],
                    "system": r["system"],
                    "user": r["user"],
                    "response": r["response"],
                    "row_kind": m.get("row_kind", ""),
                    "category": m.get("category", ""),
                }
            )

    with open(args.dst, "w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    # Report -- concrete numbers, not vibes.
    tot_chars = sum(len(k["response"]) + len(k["system"]) + len(k["user"]) for k in kept)
    approx_tok = tot_chars // 4  # ~4 chars/token, a rough planning figure only
    print(f"raw rows              : {len(rows)}")
    print(f"unique responses      : {len(groups)}")
    print(f"kept (<= {args.framings} framings) : {len(kept)}")
    print(f"kept row_kinds        : {dict(Counter(k['row_kind'] for k in kept))}")
    print(f"kept categories       : {dict(Counter(k['category'] for k in kept))}")
    print(f"approx tokens/epoch   : ~{approx_tok:,} (planning estimate; real count is Rust-side)")


if __name__ == "__main__":
    main()
