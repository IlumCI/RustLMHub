#!/usr/bin/env python
"""
tok.py - text to token ids and back, for driving the C engine.

WHY THE TOKENIZER IS IN PYTHON AND THE ENGINE IS IN C
    Kimi K3 uses a tiktoken BPE with 163,584 ranks plus 256 special tokens. Writing that
    in C is a self-contained project of its own and would prove nothing about the
    inference engine, which is what this work is about. So tokenisation happens here,
    ids go to the engine, and ids come back here to be decoded. The boundary is explicit
    rather than hidden.

    The engine therefore does text in and text out only in the sense that this script
    wraps it. That is stated plainly rather than glossed over.

FORMAT
    tiktoken.model is one "base64(token_bytes) rank" pair per line. The chat template
    and special tokens come from tokenizer_config.json.

usage:
    tok.py encode "some text"       -> prints comma separated ids
    tok.py decode 1,2,3             -> prints the text
    tok.py chat "some text"         -> ids with the chat template applied
"""
from __future__ import annotations

import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def find_files():
    """Locate the released tokenizer files. Public: tok_parity.py calls this too.

    A single hardcoded path works in a full checkout and nowhere else. On a machine where
    only the tokenizer files were copied across, a hardcoded join raises FileNotFoundError
    from inside the oracle, which surfaces as a tokenizer PARITY FAILURE rather than a
    missing-path error -- the most misleading possible framing, since the tokenizer is
    fine.

    Order: explicit K3_TOK_FILES, the in-repo location, then conventional spots. On
    failure every candidate is listed, so the fix is obvious rather than guessed at.
    """
    cands = []
    env = os.environ.get("K3_TOK_FILES")
    if env:
        cands.append(env)
    cands += [
        os.path.join(ROOT, "kimi_k3_hf", "files"),
        os.path.join(os.path.dirname(ROOT), "kimi_k3_hf", "files"),
        os.path.expanduser("~/k3/hf"),
        os.path.expanduser("~/k3model"),
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, "tiktoken.model")):
            return c
    raise SystemExit(
        "cannot find tiktoken.model; looked in:\n  " + "\n  ".join(cands) +
        "\nset K3_TOK_FILES to the directory holding tiktoken.model")


FILES = find_files()


def split_pattern() -> str:
    """Read TikTokenTokenizer.pat_str out of the released tokenizer.

    It is written as a multi-line '|'.join([...]) of eight raw strings, so scraping it
    with line matching is fragile and importing the module would drag in transformers,
    tokenizers and a relative import. Parsing the source with ast gets the exact value
    with no dependencies and no guessing: a tokenizer whose split pattern differs even
    slightly produces different ids, and the engine would then be fed a prompt the model
    never saw in that form.
    """
    import ast

    src = open(os.path.join(FILES, "tokenization_kimi.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pat_str" for t in node.targets):
            continue
        v = node.value
        # "sep".join([...]) of string literals
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                and v.func.attr == "join"):
            sep = ast.literal_eval(v.func.value)
            parts = ast.literal_eval(v.args[0])
            return sep.join(parts)
        return ast.literal_eval(v)          # a plain string literal
    raise SystemExit("pat_str not found in tokenization_kimi.py")


def load():
    import tiktoken

    ranks = {}
    with open(os.path.join(FILES, "tiktoken.model"), encoding="utf-8") as f:
        for line in f:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)

    cfg = json.load(open(os.path.join(FILES, "tokenizer_config.json"), encoding="utf-8"))
    special = {}
    for _id, entry in (cfg.get("added_tokens_decoder") or {}).items():
        special[entry["content"]] = int(_id)

    pat = split_pattern()

    enc = tiktoken.Encoding(name="kimi_k3", pat_str=pat, mergeable_ranks=ranks,
                            special_tokens=special)
    return enc, cfg, special


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode, arg = sys.argv[1], sys.argv[2]
    enc, cfg, special = load()

    if mode == "encode":
        ids = enc.encode(arg, allowed_special=set(special))
        print(",".join(str(i) for i in ids))
    elif mode == "chat":
        # Apply the released chat template if there is one, so the model sees the format
        # it was trained on. Without it a chat model answers as if mid-document.
        tmpl = cfg.get("chat_template")
        if tmpl:
            from jinja2 import Template
            text = Template(tmpl).render(
                messages=[{"role": "user", "content": arg}],
                add_generation_prompt=True, bos_token=cfg.get("bos_token", ""),
            )
        else:
            text = arg
        ids = enc.encode(text, allowed_special=set(special))
        sys.stderr.write("templated prompt:\n%s\n---\n" % text)
        print(",".join(str(i) for i in ids))
    elif mode == "decode":
        ids = [int(x) for x in arg.replace(" ", "").split(",") if x != ""]
        rev = {v: k for k, v in special.items()}
        out, buf = [], []
        for i in ids:
            if i in rev:
                if buf:
                    out.append(enc.decode(buf)); buf = []
                out.append(rev[i])
            else:
                buf.append(i)
        if buf:
            out.append(enc.decode(buf))
        print("".join(out))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
