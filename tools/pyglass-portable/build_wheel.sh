#!/usr/bin/env bash
# Build a *portable* pyglass wheel: clone upstream, apply the portability patch,
# and build in-place (no build isolation, so setup.py's ../glass / ../third_party
# references resolve), then repair to a redistributable wheel.
#
# The patch (portable.patch) makes pyglass run across compute envs without
# losing speed/quality (see README.md):
#   * AVX2 baseline on x86_64 (runs on all Haswell+ / 2013+, not just AVX-512)
#   * NEON on arm64 (Apple Silicon, Graviton)
#   * fixes Linux/x86-only assumptions (MADV_HUGEPAGE, unguarded immintrin.h
#     includes, a strict-clang narrowing, a missing arm helpa u2 kernel)
#
# Usage:  tools/pyglass-portable/build_wheel.sh [output_dir]
# Env:    PYGLASS_REF (upstream git ref, default the pinned commit)
#         GLASS_MARCH (override the x86 SIMD baseline, e.g. "-march=native")
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/wheelhouse}"
REF="${PYGLASS_REF:-d2296ec447d2374ee8f88c6d3b85be1b1e434ad3}"
PY="${PYTHON:-python3}"

# pyglass needs a modern toolchain (C++20, fp8 E5M2 types); the macOS *system*
# python3 (3.9 / old SDK) fails to compile it. Require >= 3.10.
ver="$("$PY" -c 'import sys;print("%d%02d"%sys.version_info[:2])')"
if [ "$ver" -lt 310 ]; then
  echo "error: need Python >= 3.10 (got $("$PY" --version 2>&1)). Set PYTHON=." >&2
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo ">> cloning zilliztech/pyglass @ $REF"
git clone --quiet https://github.com/zilliztech/pyglass.git "$work/pyglass"
git -C "$work/pyglass" checkout --quiet "$REF"

echo ">> applying portable.patch"
git -C "$work/pyglass" apply "$HERE/portable.patch"

echo ">> building wheel (in-place via bdist_wheel, so ../glass resolves)"
"$PY" -m pip install -q --upgrade pip pybind11 numpy setuptools wheel delocate auditwheel 2>/dev/null || true
mkdir -p "$OUT"
( cd "$work/pyglass/python" && "$PY" setup.py bdist_wheel -d "$OUT" )

echo ">> repairing wheel for portability"
case "$(uname -s)" in
  Darwin) "$PY" -m delocate.cmd.delocate_wheel -w "$OUT" "$OUT"/glass-*.whl 2>/dev/null || \
            echo "   (delocate skipped/failed — raw wheel kept)";;
  Linux)  for w in "$OUT"/glass-*.whl; do
            auditwheel repair "$w" -w "$OUT" 2>/dev/null || echo "   (auditwheel skipped for $w)"
          done;;
esac

echo ">> done. wheels in: $OUT"
ls -la "$OUT"/*.whl
