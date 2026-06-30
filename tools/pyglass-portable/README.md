# pyglass-portable

A portability patch + build kit for [zilliztech/pyglass](https://github.com/zilliztech/pyglass)
(MIT) that makes it **run across most compute environments without sacrificing
speed or quality** — the thing the upstream PyPI wheel (`glassppy`) can't do.

## The problem

pyglass is one of the fastest HNSW query engines, but its published wheel is
compiled **AVX-512-only**, so it `SIGILL`s (illegal instruction) on:

- GitHub's runners (now uniformly **AMD EPYC 7763** — AVX2, no AVX-512),
- most cloud/desktop CPUs without AVX-512,
- **arm64** entirely (no wheel at all — Apple Silicon, AWS Graviton).

The *source* already has AVX-512 / AVX2 / SSE / **NEON** / scalar kernels — they're
just selected at **compile time**, and the wheel baked in AVX-512. So portability
is a build/packaging fix, not new kernels. (A handful of Linux/x86-only assumptions
also leak through and break non-x86 / strict-compiler builds — fixed here.)

## The fix (`portable.patch`, 6 files)

| file | fix |
|---|---|
| `python/build_extension.py` | per-arch SIMD baseline: **`-mavx2 -mfma -mf16c`** on x86_64 (Haswell/2013+, *not* `-march=native`/AVX-512), **NEON** on arm64; Linux-only `-lrt`; OpenMP made optional (libomp on macOS if present). Override x86 ISA via `GLASS_MARCH`. |
| `glass/simd/avx2.hpp` | move `#include "helpa/.../x86/utils.hpp"` *inside* `#if defined(__AVX2__)` (it pulled `immintrin.h` on ARM). |
| `glass/quant/sq1_quant.hpp` | guard a bare `#include <immintrin.h>` with `#if defined(__SSE2__)`. |
| `glass/memory.hpp` | guard Linux-only `MADV_HUGEPAGE` with `#ifdef`. |
| `glass/searcher/refiner.hpp` | `auto dist` instead of `float dist` — avoids a `float→int` narrowing that strict clang rejects. |
| `third_party/helpa/.../arm/l2_impl.hpp` | add the missing `l2a_u2_u2` (2-bit-quant L2) kernel for arm (upstream defines it only for x86). |

**Speed/quality:** identical recall (same algorithm). On x86 the only thing given
up vs the AVX-512 wheel is the AVX-512 *increment* (~10–20% end-to-end, since graph
ANN search is memory-latency-bound) — recovered later with runtime dispatch if
wanted. AVX2 + NEON are the per-arch native baselines.

## Verified

Built and run on **Apple Silicon (arm64, NEON)** — which never had a pyglass wheel:

```
glass-2.1.0-cp312-cp312-macosx_arm64.whl  →  installs + runs
HNSW build + search on M-series: ~450k queries/sec
```

The same patch produces a portable **x86-64 AVX2** wheel that runs on the AMD CI
runners (no AVX-512) without a source build.

## Build a wheel locally

```bash
# needs Python >= 3.10 with pip + a C++20 compiler
PYTHON=/path/to/python3.12 tools/pyglass-portable/build_wheel.sh [out_dir]
```

Clones upstream at the pinned commit, applies `portable.patch`, builds in-place
(`bdist_wheel`, so setup.py's `../glass` references resolve), and `delocate`/
`auditwheel`-repairs the result. The import name stays **`glass`** (drop-in).

## Use it as a scanpy NN backend

Install the wheel, and pyglass works as a drop-in `sc.pp.neighbors` backend via
`GlassTransformer` — the same pattern as `PiPNNTransformer`:

```bash
pip install wheelhouse/glass-*.whl
```

```python
import scanpy as sc
from pipnn.contrib import GlassTransformer
sc.pp.neighbors(adata, n_neighbors=15, transformer=GlassTransformer())
```

`GlassTransformer` imports `glass` (this build) or `glassppy` automatically, so no
other change is needed. This is how the pyglass column in the Tahoe-100M
benchmark was run, natively on Apple Silicon.

## Multi-arch wheels (CI)

`.github/workflows/pyglass-portable-wheels.yml` (`workflow_dispatch`) runs the
build across `{linux-x86_64 (AVX2), macOS-arm64 (NEON), macOS-x86_64 (AVX2)}` and
uploads the wheels as artifacts.

## Publishing the fork to PyPI (remaining manual steps)

The technical work is done; publishing needs an account you control:

1. Create a GitHub fork (`gh repo fork zilliztech/pyglass`) and commit this patch
   (or vendor the patched tree). Keep the upstream MIT `LICENSE` + attribution.
2. Pick a distribution name (e.g. `pyglass-portable`) in `python/pyproject.toml`
   `[project].name`; keep the import module `glass`.
3. Add manylinux + `linux-aarch64` (QEMU) builds and a multi-CPython matrix
   (cibuildwheel, or extend the workflow here), then a `release.yml` that publishes
   on tag via **PyPI Trusted Publishing** (no token in the repo).
4. `git tag vX.Y.Z && git push --tags` → wheels build and publish.

Upstream is MIT-licensed (© 2023 zh Wang); this is a derivative build patch.
