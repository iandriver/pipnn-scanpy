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

- **Randomized Ball Carving** partitioning (paper Alg 5), as a two-stage
  disjoint-carve + bounded-halo scheme so total replication is ≈`fanout`
  (not `fanout^depth`).
- **Leaf GEMM** all-pairs distances (`‖x−y‖² = ‖x‖²+‖y‖²−2XYᵀ`, paper §4.2).
- **HashPrune** online residualized-LSH pruning (paper Alg 3) with the 8-byte
  reservoir slot; candidates stream straight into per-point reservoirs (the only
  persistent build state) — history-independent, so the build is deterministic.
- **RobustPrune** (Alg 2) to a degree-`R` navigable graph.
- **BeamSearch** (Alg 1) self-query, seeded at each point for robust self-kNN.

Small inputs (`n ≤ 4096`) use an exact brute-force path that doubles as the
recall oracle. For very large/held-out queries, a global navigable index +
held-out `transform(X_new)` is future work.

## Performance (50-d, M3-class laptop, 16 threads)

| n        | build+query | recall@15 | peak RSS |
|----------|-------------|-----------|----------|
| 100k     | ~1.8 s      | 0.9999    | ~0.4 GB* |
| 500k     | ~13 s       | 0.988     | ~2.0 GB* |

`*` PiPNN-only; benchmark-process peaks are higher because they also hold the
sklearn exact-NN ground truth. At 100k, PiPNN built **~2.7× faster than
pynndescent** (cold start) at equal-or-better recall.

### Scaling notes

Comfortable to ~1M cells (~4–5 GB). Beyond that, two known costs need the
Phase-8 optimizations: the halo step is `O(n²/c_max)`, and the per-leaf GEMM
transient grows with leaf size. Raise `c_max` / lower `fanout` to trade recall
for memory, or set `n_jobs` to bound thread-level transient memory.

## Comparison notebook

`notebooks/pipnn_vs_pynndescent.ipynb` compares PiPNN vs pynndescent vs exact on
real single-cell data, all through the same `sc.pp.neighbors(transformer=...)`
hook: build time, recall@k, side-by-side UMAP embeddings, and Leiden clustering
agreement (ARI). It ships pre-executed with plots. To re-run:

```bash
.venv/bin/python -m ipykernel install --user --name pipnn-venv --display-name "PiPNN (venv)"
.venv/bin/jupyter lab notebooks/pipnn_vs_pynndescent.ipynb   # select the "PiPNN (venv)" kernel
```

Representative result (20k cells, 50 PCs): PiPNN `sc.pp.neighbors` 1.7s vs
pynndescent 7.7s (**4.7×**), recall@15 0.9997 vs 0.9948, ARI-to-exact 0.94 vs 0.92.
(`build_notebook.py` regenerates the notebook.)

## Metrics

`euclidean` (default, matches scanpy on PCA space) and `cosine`.
