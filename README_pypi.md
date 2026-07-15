# pipnn

Fast, graph-based approximate-nearest-neighbor indexing for building the k-NN
graph in single-cell / `scanpy` workflows — a Rust implementation (with PyO3
bindings) of the **PiPNN** method. It plugs into `scanpy` as a drop-in
`sc.pp.neighbors` backend and also exposes a small standalone API. It builds the
graph quickly at recall close to exact, and scales to tens of millions of cells.

> Independent implementation of the PiPNN algorithm. Please cite the paper:
> Tobias Rubel, Richard Wen, Laxman Dhulipala, Lars Gottesbüren, Rajesh Jayaram,
> Jakub Łącki. *PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor Indexing.*
> [arXiv:2602.21247](https://arxiv.org/abs/2602.21247)

## Install

```bash
uv pip install pipnn      # or: pip install pipnn
```

Prebuilt wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64), and Windows
(x86_64); Python 3.9+. No Rust toolchain needed to install.

## Quickstart

**As a scanpy backend** (drop-in for the default neighbor step):

```python
import scanpy as sc
from pipnn import PiPNNTransformer

sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca",
                transformer=PiPNNTransformer())
# then sc.tl.umap / sc.tl.leiden as usual
```

**Standalone** (just the k-NN graph):

```python
import numpy as np
from pipnn import self_knn_graph

X = np.random.default_rng(0).normal(size=(100_000, 50)).astype("float32")
indices, distances = self_knn_graph(X, n_neighbors=15)
# indices, distances: shape (n, k+1), self edge first, sorted by distance
```

## What it does

- Produces the exact-distance k-NN graph `scanpy` expects — a scipy CSR of shape
  `(n, n)` with `k+1` entries per row, the self edge first.
- `euclidean` (default, matches scanpy on PCA space) and `cosine` metrics.
- Parallel Rust build; deterministic given a fixed `random_state`.
- Optional comparison backends under `pipnn.contrib` behind the same transformer
  API: a native `HnswTransformer`, `FaissTransformer` (`pip install pipnn[faiss]`),
  and `GlassTransformer` (pyglass).

## Benchmarks & source

Full benchmarks — SIFT, and the Tahoe-100M single-cell atlas from 1M to 10M cells
(build time, recall, and peak memory vs FAISS, pyglass, and pynndescent) — plus
development notes and the comparison harness, are in the repository:

**https://github.com/iandriver/pipnn-scanpy**

MIT licensed.
