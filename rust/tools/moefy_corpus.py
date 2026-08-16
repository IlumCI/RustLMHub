#!/usr/bin/env python3
"""Assemble a diverse MoEfication calibration corpus from local HF datasets + repo code.

Routing tunes to the calibration distribution, so this deliberately mixes web, math,
reasoning traces, and code (the shape of the deployment) rather than one domain. Documents
are interleaved and separated by blank lines so the fixed-length capture windows straddle
domains. Output is plain UTF-8 text for `moefy_capture`.

  moefy_corpus.py --out calib.txt [--per 40] [--repo-code DIR]
"""
import argparse, glob, os, random, sys

HF = os.path.expanduser("~/.cache/huggingface/datasets")

# (glob for an arrow shard, list of text columns to join per row)
SOURCES = [
    ("ag_news/*/*/*/ag_news-train.arrow", ["text"]),
    ("openai___gsm8k/*/*/*/gsm8k-train.arrow", ["question", "answer"]),
    ("Crownelius___opus-4.6-reasoning-3300x/*/*/*/*-train.arrow", None),  # None = all str cols
    ("nvidia___open_code_instruct/*/*/*/*-00000-*.arrow", None),
    ("HuggingFaceH4___instruction-dataset/*/*/*/*.arrow", None),
]


def rows_from(arrow_glob, cols, per, rng):
    import datasets
    paths = sorted(glob.glob(os.path.join(HF, arrow_glob)))
    if not paths:
        print(f"  (skip, no shard: {arrow_glob})", file=sys.stderr)
        return []
    ds = datasets.Dataset.from_file(paths[0])
    use = cols if cols else [c for c in ds.column_names]
    idx = rng.sample(range(len(ds)), min(per, len(ds)))
    out = []
    for i in idx:
        r = ds[i]
        parts = []
        for c in use:
            v = r.get(c)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # chat-style: join message contents
                parts.append("\n".join(str(m.get("content", "")) for m in v))
        if parts:
            out.append("\n".join(parts))
    print(f"  {arrow_glob.split('/')[0]:40s} {len(out):3d} docs", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=40, help="docs per source")
    ap.add_argument("--repo-code", default="src", help="dir to sample .rs files from")
    ap.add_argument("--code-files", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    docs = []
    for g, cols in SOURCES:
        docs += rows_from(g, cols, args.per, rng)

    # real code from the repo itself
    rs = glob.glob(os.path.join(args.repo_code, "**/*.rs"), recursive=True)
    for p in rng.sample(rs, min(args.code_files, len(rs))):
        docs.append(open(p, encoding="utf-8", errors="ignore").read()[:4000])
    print(f"  repo .rs                                 {min(args.code_files, len(rs)):3d} docs", file=sys.stderr)

    rng.shuffle(docs)
    text = "\n\n".join(docs)
    open(args.out, "w", encoding="utf-8").write(text)
    print(f"corpus: {len(docs)} docs, {len(text)} chars -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
