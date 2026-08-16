#!/bin/sh
# RustLMHub companion installer for `rustlm code`.
#
# `rustlm code` dispatches to a SEPARATE binary called `rustlm-code`, the way git dispatches to
# git-*. That binary is a standalone terminal coding agent licensed GPL-3.0, forked from claurst
# (github.com/kuberwastaken/claurst) and maintained at github.com/IlumCI/rustlm-code. It is NOT
# part of the RustLMHub engine and is NOT covered by the RustLMHub license. This script builds it
# from source and installs it in the SAME directory as your `rustlm` binary, which is where
# `rustlm code` looks for it first, so the two work together.
#
#   ./install-code.sh                 build and install next to rustlm (or ~/.local/bin)
#   ./install-code.sh --system        install to /usr/local/bin (uses sudo)
#   ./install-code.sh --prefix DIR    install to DIR/bin
#   ./install-code.sh --src DIR       path to a rustlm-code checkout (the dir containing src-rust/)
#   ./install-code.sh --clone         git clone the fork if no local source is found
#   ./install-code.sh --repo URL      clone URL (default https://github.com/IlumCI/rustlm-code)
#   ./install-code.sh --no-build      install an already-built binary only

set -eu

# --- locate this script's directory (robust to symlinks) ---
SELF=$(command -v "$0" || echo "$0")
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$SELF")" && pwd -P)

# --- defaults / args ---
PREFIX=""
SYSTEM=0
SRC=""
CLONE=0
BUILD=1
REPO="https://github.com/IlumCI/rustlm-code"
while [ $# -gt 0 ]; do
  case "$1" in
    --system) SYSTEM=1 ;;
    --prefix) shift; PREFIX="${1:-}" ;;
    --src)    shift; SRC="${1:-}" ;;
    --clone)  CLONE=1 ;;
    --repo)   shift; REPO="${1:-}" ;;
    --no-build) BUILD=0 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- find the rustlm-code source (must contain src-rust/Cargo.toml) ---
has_src() { [ -f "$1/src-rust/Cargo.toml" ]; }

# A stable, user-writable location for a fetched copy. Used when there is no local checkout
# (for example when this script is run piped through `curl | sh`, where SCRIPT_DIR is not the
# repo). Override with RUSTLM_CODE_SRC.
DEFAULT_SRC="${RUSTLM_CODE_SRC:-$HOME/.local/share/rustlm-code}"

if [ -z "$SRC" ]; then
  # Prefer a checkout that is a sibling of this repo under RustLMHub/, then a previous clone.
  for c in "$SCRIPT_DIR/../rustlm-code" "$SCRIPT_DIR/rustlm-code" "$DEFAULT_SRC"; do
    if has_src "$c"; then SRC="$c"; break; fi
  done
fi

if [ -z "$SRC" ] || ! has_src "$SRC"; then
  if [ "$CLONE" -eq 1 ]; then
    SRC="${SRC:-$DEFAULT_SRC}"
    echo "cloning $REPO -> $SRC"
    rm -rf "$SRC" 2>/dev/null || true
    git clone --depth 1 "$REPO" "$SRC"
  else
    echo "error: no rustlm-code source found." >&2
    echo "  Pass --src DIR pointing at a checkout (the dir containing src-rust/)," >&2
    echo "  or pass --clone to fetch it from $REPO." >&2
    exit 1
  fi
fi
SRC=$(CDPATH= cd -- "$SRC" && pwd -P)
echo "rustlm-code source: $SRC"

# --- decide where to install ---
# Prefer the directory that already holds `rustlm`, so `rustlm code` finds this binary as a
# sibling (its first lookup) and it is guaranteed to be on the same PATH.
if [ -n "$PREFIX" ]; then
  BINDIR="$PREFIX/bin"
elif [ "$SYSTEM" -eq 1 ]; then
  BINDIR="/usr/local/bin"
elif command -v rustlm >/dev/null 2>&1; then
  BINDIR=$(dirname -- "$(command -v rustlm)")
  echo "found rustlm at $BINDIR/rustlm; installing beside it"
else
  BINDIR="$HOME/.local/bin"
fi

mkdir -p "$BINDIR" 2>/dev/null || true
SUDO=""
if [ ! -w "$BINDIR" ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    echo "note: $BINDIR is not writable, using sudo to install"
  else
    echo "error: $BINDIR is not writable and sudo is unavailable. Use --prefix DIR." >&2
    exit 1
  fi
fi

# --- build ---
BIN="$SRC/src-rust/target/release/rustlm-code"
if [ "$BUILD" -eq 1 ]; then
  command -v cargo >/dev/null 2>&1 || { echo "error: cargo (Rust) is required to build. Install rustup." >&2; exit 1; }
  echo "building rustlm-code (release, this can take several minutes on a first build)"
  ( cd "$SRC/src-rust" && cargo build --release --no-default-features -p rustlm-code )
fi
[ -f "$BIN" ] || { echo "error: binary not found at $BIN. Run without --no-build, or build it first." >&2; exit 1; }

# --- install ---
echo "installing rustlm-code to $BINDIR"
$SUDO install -m 0755 "$BIN" "$BINDIR/rustlm-code"

# --- PATH check ---
case ":$PATH:" in
  *":$BINDIR:"*) : ;;
  *)
    echo "note: $BINDIR is not on your PATH. Add it, for example:"
    echo "  echo 'export PATH=\"$BINDIR:\$PATH\"' >> \"$HOME/.profile\" && . \"$HOME/.profile\""
    ;;
esac

# --- verify ---
echo
echo "installed: $BINDIR/rustlm-code"
"$BINDIR/rustlm-code" --version 2>/dev/null | head -1 || true
if command -v rustlm >/dev/null 2>&1; then
  echo "wired up: 'rustlm code' will now launch it."
else
  echo "note: 'rustlm' was not found on PATH. Install the engine first with ./install.sh, then 'rustlm code' will use this binary."
fi
echo
echo "license: rustlm-code is GPL-3.0 (a fork of claurst, github.com/kuberwastaken/claurst)."
echo "It is a separate program from the RustLMHub engine and is not under the RustLMHub license."
