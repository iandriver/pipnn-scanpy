# PiPNN

Fast graph-based approximate nearest-neighbor indexing for single-cell data,
implementing the **PiPNN** algorithm (Rubel et al., *PiPNN: Ultra-Scalable
Graph-Based Nearest Neighbor Indexing*, arXiv:2602.21247) — including its
**HashPrune** online residualized-LSH pruning — as a Rust core with Python
bindings that plug directly into scanpy.

```python
import scanpy as sc
from pipnn import PiPNNTransformer

sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca",
                transformer=PiPNNTransformer())
# adata.obsp['distances'] / ['connectivities'] now populated; UMAP/Leiden work as usual.
```

## Build (development)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python maturin numpy scipy scikit-learn scanpy pynndescent pytest
.venv/bin/maturin develop --release
.venv/bin/pytest tests/
```

## What's implemented

The full PiPNN build pipeline, in Rust with `rayon` parallelism:

- **Randomized Ball Carving** partitioning (paper Alg 5): near-linear
  bounded-branching recursion, plus a `fanout`-overlap halo (via a coarse `√t`
  super-group index over leaf centroids) so replication stays ≈`fanout`.
- **Leaf GEMM** all-pairs distances (`‖x−y‖² = ‖x‖²+‖y‖²−2XYᵀ`, paper §4.2).
- **HashPrune** online residualized-LSH pruning (paper Alg 3) with the 8-byte
  reservoir slot; candidates stream straight into per-point reservoirs (the only
  persistent build state) — history-independent, so the build is deterministic.
- **RobustPrune** (Alg 2) to a degree-`R` navigable graph.
- **BeamSearch** (Alg 1) self-query, **warm-started** from each point's reservoir
  candidates and using a per-thread reusable scratch (no per-query allocation).

Performance is portable-SIMD (`wide::f32x8` → NEON/AVX) throughout, and the
partition + halo are near-linear (bounded branching + a coarse `√t` centroid
index). Small inputs (`n ≤ 4096`) use an exact brute-force path that doubles as
the recall oracle. For held-out queries, a `transform(X_new)` path is future work.

## Performance

See **[Scaling](#scaling-pipnn-vs-pynndescent)** below for the full table — PiPNN
builds the kNN graph **faster than pynndescent through ~200k cells**, is on par at
400k, and within ~12% at 800k, at recall@15 ≈ 1.0 throughout (~4 GB at 400k,
~6.5 GB at 800k). Tune `n_jobs` to bound thread-level transient memory, and
`beam_L` / `c_max` to trade a little recall for speed/memory.

## Comparison notebook

`notebooks/pipnn_vs_pynndescent.ipynb` compares PiPNN vs pynndescent vs
[pyglass](https://github.com/zilliztech/pyglass) vs exact on real single-cell
data, all through the same `sc.pp.neighbors(transformer=...)` hook: build time,
recall@k, side-by-side UMAP embeddings, and Leiden clustering agreement (ARI). It
ships pre-executed with plots. To re-run:

```bash
.venv/bin/python -m ipykernel install --user --name pipnn-venv --display-name "PiPNN (venv)"
.venv/bin/python build_notebook.py            # regenerate
.venv/bin/jupyter lab notebooks/pipnn_vs_pynndescent.ipynb   # "PiPNN (venv)" kernel
```

Backends are auto-discovered (`bench/bench_lib.py`): any that import are included,
so `glass` appears wherever pyglass is installed.

### Benchmark methodology (important)

Timings are reported as **cold** (first build) *and* **warm** (median of repeated
steady-state builds). pynndescent compiles numba kernels on its first call, so its
cold time is heavily JIT-inflated; the **warm** number is the fair comparison.
PiPNN (Rust) and glass (C++) have no JIT, so cold ≈ warm.

Representative result (20k cells, 50 PCs, warm): PiPNN `sc.pp.neighbors` ≈ 0.49s
vs pynndescent ≈ 0.51s (the previously-quoted "4.7×" was almost entirely numba
JIT — the corrected warm numbers are ~par at this size; PiPNN's advantage grows
with `n`). recall@15 0.9997 vs 0.9948; ARI-to-exact 0.94 vs 0.92.

### pyglass on Apple Silicon

pyglass ships only manylinux x86_64 wheels (`glassppy`, CPython 3.10) and its
source assumes x86 intrinsics, so it does not run natively on arm64 macOS.
`python/pipnn/contrib/glass.py` (`GlassTransformer`) activates automatically
wherever `glassppy`/`glass` imports.

The bundled `docker/Dockerfile` (linux/amd64, py3.10) builds a complete 4-backend
image — it installs `glassppy`, compiles `pipnn`, and runs `docker_compare.py`:

```bash
docker build --platform linux/amd64 -t pipnn-bench -f docker/Dockerfile .
docker run --platform linux/amd64 --rm -v "$PWD/docker/out:/out" pipnn-bench
```

**Run this on x86_64 hardware** (a native Linux box or CI). On an arm64 host the
container runs under qemu emulation, where glass's SIMD/OpenMP code is ~1000×
slower (an `n=1000` build did not finish in 14 min) — verified that glassppy
installs and the `GlassTransformer` API matches, but the benchmark is not
runnable under emulation. The native notebook (PiPNN/pynndescent/exact) is the
authoritative timing comparison on Apple Silicon.

## Scaling: PiPNN vs pynndescent

`bench/bench_scaling.py` sweeps dataset size and times the kNN-graph build (warm,
min-of-N — pynndescent JIT excluded) with recall@15 vs exact. Synthetic 50-d data,
constant cluster density, all cores.

![PiPNN vs pynndescent build-time and recall scaling, 5k–800k cells](bench/scaling.png)

*Left: warm build time vs n (log-log). Solid = PiPNN now; dashed = PiPNN before
this work's optimizations (~4× slower at 800k). The shaded band marks where PiPNN
is the faster builder — up to the crossover at ~346k cells. Right: recall@15 —
PiPNN holds ≈1.0 at every size while pynndescent (defaults) falls to ~0.82
(≈ +16 points at scale).*

| cells | PiPNN build | pynndescent build | speedup | PiPNN recall | pynndescent recall | peak RSS |
|------:|------------:|------------------:|--------:|-------------:|-------------------:|---------:|
| 5k    | **0.06s** | 0.46s | **7.8×** | 1.000 | 0.973 | 0.8 GB |
| 25k   | **0.15s** | 0.47s | **3.2×** | 1.000 | 0.868 | 1.4 GB |
| 50k   | **0.29s** | 0.57s | **2.0×** | 0.997 | 0.818 | 1.6 GB |
| 100k  | **0.60s** | 0.79s | **1.3×** | 0.998 | 0.825 | 2.2 GB |
| 200k  | **1.23s** | 1.35s | **1.1×** | 0.998 | 0.830 | 2.9 GB |
| 400k  | 2.55s | 2.49s | ~1.0× | 0.997 | 0.833 | 4.1 GB |
| 800k  | 5.47s | 4.82s | 0.88× | 0.997 | 0.832 | 6.6 GB |

### Tradeoffs vs pynndescent

| | **PiPNN** | **pynndescent** |
|---|---|---|
| **Recall@15** (defaults) | **≈1.0 at all sizes** | 0.97 → **~0.82** as n grows |
| **Warm build** | faster ≤200k, ~tied 400k, ~12% slower 800k | flat/low; scales slightly better past ~400k |
| **Cold (first) build** | same as warm | **+5–10 s numba JIT** on first call |
| **Determinism** | **deterministic** (seed → identical graph) | randomized/approximate |
| **Runtime deps** | self-contained Rust wheel (no JIT) | numba + llvmlite |
| **Maturity** | new | battle-tested, scanpy default |

**When PiPNN wins:** you want near-exact neighbors (recall matters for
clustering/UMAP fidelity), reproducible graphs, fast first-call (notebooks,
CI, many small datasets), or you're at ≤ a few hundred thousand cells.
**When pynndescent is fine:** atlas-scale (≳1M cells) where its slightly better
warm scaling helps and lower recall is acceptable — though to *match* PiPNN's
recall it must raise its build parameters, which narrows or erases the speed gap.

To trade PiPNN's recall for more speed, lower `beam_L` (e.g. 64 → 40 ≈ recall
0.99) or set `PIPNN_QUERY=reservoir` (fastest, ~0.93 recall).

### Matched-recall comparison

The default-vs-default table above is not apples-to-apples on quality: pynndescent
is running at recall ~0.82 there. pynndescent's recall *is* tunable — chiefly by
building a wider graph (`n_neighbors` ≫ k, then keep the top-k), the direct analog
of PiPNN's internal over-search. `bench/bench_matched_recall.py` tunes pynndescent
up to PiPNN's recall and re-times it:

| cells | PiPNN | pynndescent (default) | pynndescent (matched recall) | slowdown at matched recall |
|------:|------:|----------------------:|-----------------------------:|---------------------------:|
| 50k   | 0.29s @ 0.997 | 0.55s @ 0.82 | 1.08s @ 0.999 | **3.8×** |
| 100k  | 0.58s @ 0.998 | 0.79s @ 0.83 | 2.09s @ 0.999 | **3.6×** |
| 200k  | 1.25s @ 0.998 | 1.39s @ 0.83 | 4.58s @ 0.999 | **3.7×** |
| 400k  | 2.52s @ 0.997 | 2.51s @ 0.83 | 9.90s @ 0.999 | **3.9×** |

![Recall vs build time at 100k — PiPNN dominates pynndescent's quality curve](bench/matched_recall.png)

**At equal recall (~0.999), PiPNN is ~3.6–3.9× faster than pynndescent at every
size.** Note 400k: default-vs-default they *tie* (2.5s each) — but only because
pynndescent is at 0.83 recall; hold it to 0.997 and it needs 9.9s. The Pareto plot
(recall vs build time at 100k) shows PiPNN sitting in the upper-left "high recall,
low time" corner while pynndescent must spend 3–5× the time climbing to the same
recall.

These build-time gains came from five **recall-neutral** optimizations, found by profiling
(`PIPNN_PROFILE=1` prints per-stage timings), not guesswork:
1. **Leaf candidate selection** via quickselect, replacing an `O(s²·ℓ_max)`
   insertion sort (the real hot spot — partitioning never was).
2. **Portable SIMD** (`wide::f32x8` → NEON/AVX) for the squared-L2 kernel that
   BeamSearch, RobustPrune, carving, and the halo all bottom out in.
3. **`beam_L` default 100 → 64** (recall ~0.997, faster query).
4. **De-quadratified partitioning**: both the ball-carving assignment and the
   overlap halo were `O(n²/c_max)` (a fat single level / a global centroid scan);
   bounded branching + a coarse `√t` super-group index over leaf centroids make
   them near-linear.
5. **Self-query**: BeamSearch reused a `vec![false; n]` per query (`O(n²)` zeroing
   across all `n` queries) — now a per-thread reusable scratch reset only on
   touched entries, warm-started from each point's reservoir candidates. Cut the
   800k query ~2×.

### Native HNSW backend (three-way comparison)

pyglass (HNSW/NSG) can't run on arm64 macOS, so for a real graph-ANN comparison we
also ship a compact **HNSW** built natively in the Rust crate — the algorithm
pyglass implements — reusing the SIMD kernel, RobustPrune (α=1 = HNSW's neighbor
heuristic), and a parallel hnswlib-style concurrent build. Use it like any backend:

```python
from pipnn.contrib import HnswTransformer
sc.pp.neighbors(adata, n_neighbors=15, transformer=HnswTransformer())
```

It's auto-included in `bench/bench_lib.py`, giving PiPNN vs HNSW vs pynndescent vs
exact on the same hardware (warm steady-state, 50-d, library defaults):

| backend | 50k build | recall | 100k build | recall |
|---|---|---|---|---|
| **PiPNN** | **0.34s** | **0.997** | **0.69s** | **0.998** |
| HNSW (this crate) | 0.43s | 0.982 | 0.83s | 0.943 |
| pynndescent | 0.62s | 0.817 | 0.89s | 0.826 |
| exact | 0.62s | 1.000 | 2.35s | 1.000 |

PiPNN is fastest *and* highest-recall; our HNSW lands in between — competitive
speed at much higher recall than pynndescent's defaults. (HNSW's parallel build is
non-deterministic, unlike PiPNN's; `HnswTransformer` exposes `m`, `ef_construction`,
`ef_search` to trade recall for speed.)

#### Scalar quantization (SQ8) — and why it doesn't help here

`HnswTransformer(quantize="sq8")` stores per-dimension **8-bit codes** (4× smaller
than f32) and builds/searches on them — pyglass's core speed trick. It's correctly
implemented (emitted distances are always recomputed exactly), but on single-cell
data it is **not** worth it:

| n=100k | exact f32 | SQ8 | | vector memory |
|---|---|---|---|---|
| d=50  | 0.79s @ 0.977 | 1.09s @ 0.912 | (slower, lower recall) | 20 MB → 5 MB |
| d=256 | 2.43s @ 0.940 | 3.96s @ 0.878 | (slower, lower recall) | 102 MB → 26 MB |

SQ8 cuts vector memory 4× but is **slower** at both dims. The reason is
fundamental to a *portable* implementation: our exact path is a tight `f32x8` SIMD
kernel over contiguous floats, whereas SQ8 needs a scalar `u8 → f32` gather per
block (portable SIMD can't widen `u8` lanes without arch-specific NEON/AVX
intrinsics), and the memory saving doesn't overcome that. SQ8's real wins
(pyglass/FAISS) come from hand-tuned integer kernels at very high `d` / billion
scale. **Takeaway for single-cell:** at PCA dimensions, exact SIMD f32 beats
quantization on *both* speed and recall — which is why PiPNN uses exact distances.

## Metrics

`euclidean` (default, matches scanpy on PCA space) and `cosine`.
