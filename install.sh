#!/bin/sh
# RustLMHub installer — builds the Rust engine + workbench and installs the binaries.
# Works on Linux and macOS (Darwin), x86_64 and arm64.
#
#   ./install.sh                 build (release) and install to ~/.local/bin
#   ./install.sh --system        install to /usr/local/bin (uses sudo; includes the tools)
#   ./install.sh --prefix DIR    install to DIR/bin
#   ./install.sh --tools         also install the dev/inspection tools (implied by sudo)
#   ./install.sh --no-build      install already-built binaries only
#
# The four platform binaries (rustlm, rustlm_tui, train_run, eval_run) are installed into the
# SAME directory on purpose: `rustlm train` dispatches to `rustlm_tui`, which shells out to
# `train_run`/`eval_run`, all found as siblings.

set -eu

# --- locate the rust workspace relative to this script (robust to symlinks) ---
SELF=$(command -v "$0" || echo "$0")
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$SELF")" && pwd -P)
RUST_DIR="$SCRIPT_DIR/rust"
[ -f "$RUST_DIR/Cargo.toml" ] || { echo "error: cannot find rust/Cargo.toml next to this script ($RUST_DIR)"; exit 1; }

# --- args ---
PREFIX=""
SYSTEM=0
TOOLS=0
BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --system) SYSTEM=1 ;;
    --tools)  TOOLS=1 ;;
    --no-build) BUILD=0 ;;
    --prefix) shift; PREFIX="${1:-}";;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
  shift
done

OS=$(uname -s)
ARCH=$(uname -m)
echo "RustLMHub installer  —  $OS/$ARCH"

# --- prerequisites ---
if [ "$BUILD" -eq 1 ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found. Install Rust (>= 1.85):"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
  fi
  RUSTV=$(rustc --version 2>/dev/null | awk '{print $2}')
  echo "using rustc $RUSTV"
fi

# --- pick the install directory ---
if [ -n "$PREFIX" ]; then
  BINDIR="$PREFIX/bin"
elif [ "$SYSTEM" -eq 1 ]; then
  BINDIR="/usr/local/bin"
else
  BINDIR="$HOME/.local/bin"
fi

# Does writing to BINDIR need sudo?
SUDO=""
mkdir -p "$BINDIR" 2>/dev/null || true
if [ ! -w "$BINDIR" ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
    echo "note: $BINDIR is not writable — will use sudo to install"
  else
    echo "error: $BINDIR is not writable and sudo is unavailable. Use --prefix DIR."
    exit 1
  fi
fi

# --- the binaries ---
# Core platform (must be co-located for the sibling dispatch).
CORE="rustlm rustlm_tui train_run eval_run"
# Optional inspection/prep tools.
DEVTOOLS="dataprep q35chk gguf_dump"

# A system-wide install (--system, or any dir that needed sudo) is meant to be complete, so
# it includes the tools automatically. A user-local install stays lean unless --tools is given.
if [ "$SYSTEM" -eq 1 ] || [ -n "$SUDO" ]; then
  TOOLS=1
fi

SET="$CORE"
[ "$TOOLS" -eq 1 ] && SET="$CORE $DEVTOOLS"

# --- build ---
if [ "$BUILD" -eq 1 ]; then
  echo "building (release): $SET"
  BINARGS=""
  for b in $SET; do BINARGS="$BINARGS --bin $b"; done
  # shellcheck disable=SC2086
  ( cd "$RUST_DIR" && cargo build --release $BINARGS )
fi

TARGET="$RUST_DIR/target/release"

# --- install ---
echo "installing to $BINDIR"
for b in $SET; do
  if [ ! -x "$TARGET/$b" ]; then
    echo "error: $TARGET/$b not found — build it first (drop --no-build)"; exit 1
  fi
  $SUDO install -m 0755 "$TARGET/$b" "$BINDIR/$b"
  echo "  installed $b"
done

# --- PATH hint + smoke test ---
echo
case ":$PATH:" in
  *":$BINDIR:"*) : ;;  # already on PATH
  *)
    echo "note: $BINDIR is not on your PATH. Add it:"
    RC="$HOME/.bashrc"; [ "$OS" = "Darwin" ] && RC="$HOME/.zshrc"
    echo "  echo 'export PATH=\"$BINDIR:\$PATH\"' >> $RC && . $RC"
    ;;
esac

echo
echo "done. Try:"
echo "  rustlm list           # what's registered"
echo "  rustlm train          # the fine-tune & eval workbench (TUI)"
"$BINDIR/rustlm" 2>/dev/null | head -1 || true
