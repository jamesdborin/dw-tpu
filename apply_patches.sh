#!/usr/bin/env bash
# Apply the DSV4 TPU v6e patch series.
#
#   ./apply_patches.sh                 # patch tpu-inference (must be clean, at the pinned commit)
#   ./apply_patches.sh --reset         # discard local changes in tpu-inference first
#   ./apply_patches.sh --tokamax PY    # also patch the tokamax installed in python env PY
#   ./apply_patches.sh --all           # also apply patches/tpu-inference-ext (0007-0503)
#
# Patches are applied in filename order: patches/tpu-inference first, then (with --all)
# patches/tpu-inference-ext. Every patch except 0001-0003 is a no-op unless its
# environment flag is set (see README.md).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
RESET=0 ALL=0 PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --reset) RESET=1 ;;
    --all) ALL=1 ;;
    --tokamax) PY=${2:?--tokamax needs a python executable}; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

TI="$ROOT/tpu-inference"
PIN=$(git -C "$ROOT" ls-files -s tpu-inference | awk '{print $2}')
if [ "$RESET" = 1 ]; then
  git -C "$TI" checkout -q "$PIN" && git -C "$TI" reset -q --hard "$PIN" && git -C "$TI" clean -qfd
fi
if [ "$(git -C "$TI" rev-parse HEAD)" != "$PIN" ] || [ -n "$(git -C "$TI" status --porcelain)" ]; then
  echo "tpu-inference is not a clean checkout of $PIN; rerun with --reset" >&2
  exit 1
fi
SERIES=("$ROOT"/patches/tpu-inference/*.patch)
[ "$ALL" = 1 ] && SERIES+=("$ROOT"/patches/tpu-inference-ext/*.patch)
for p in "${SERIES[@]}"; do
  git -C "$TI" apply --whitespace=nowarn "$p"
  echo "applied $(basename "$(dirname "$p")")/$(basename "$p")"
done

if [ -n "$PY" ]; then
  SITE=$("$PY" -c 'import os, tokamax; print(os.path.dirname(os.path.dirname(tokamax.__file__)))')
  VER=$("$PY" -c 'import importlib.metadata as m; print(m.version("tokamax"))')
  [ "$VER" = "0.0.16.dev20261009" ] || { echo "expected tokamax 0.0.16.dev20261009, found $VER" >&2; exit 1; }
  for p in "$ROOT"/patches/tokamax/*.patch; do
    if patch -d "$SITE" -p1 -R --dry-run -s -f < "$p" >/dev/null 2>&1; then
      echo "already applied tokamax/$(basename "$p")"
    else
      patch -d "$SITE" -p1 -s -f < "$p"
      echo "applied tokamax/$(basename "$p")"
    fi
  done
fi
