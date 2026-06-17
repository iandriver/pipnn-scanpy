"""Generate notebooks/pipnn_vs_pynndescent.ipynb (backend-list driven)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
c = []
md = lambda s: c.append(new_markdown_cell(s))
code = lambda s: c.append(new_code_cell(s))


md("""# PiPNN vs. pynndescent vs. pyglass in scanpy

Compares ANN backends for `sc.pp.neighbors`, all plugged in through the *same*
sklearn `KNeighborsTransformer` hook — so swapping backend is a one-line change:

```python
sc.pp.neighbors(adata, transformer=PiPNNTransformer())        # this repo (Rust, arXiv:2602.21247)
sc.pp.neighbors(adata, transformer=PyNNDescentTransformer())  # scanpy default
sc.pp.neighbors(adata, transformer=GlassTransformer())        # pyglass (zilliz HNSW/NSG)
```

**Timing methodology.** We report **cold** (first build, which for pynndescent
includes one-time numba JIT compilation) *and* **warm** (median of repeated
steady-state builds). The warm number is the fair head-to-head; the cold number
shows the first-call cost a user actually feels.

> `pyglass` has no macOS/arm64 build, so on Apple Silicon the `glass` column is
> absent here; it appears automatically wherever `glass` imports (x86_64 Linux —
> see the Docker run in the repo). The methodology and all other backends are
> unaffected.
""")

code("""%matplotlib inline
import sys, time, inspect, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../bench")
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

import bench_lib as bl
sc.settings.verbosity = 0
RNG = np.random.default_rng(0)
K = 15
print("scanpy", sc.__version__)""")

md("""## 1. Data + preprocessing

Real single-cell data, subsampled for a snappy notebook, run through the standard
scanpy recipe (normalize → log1p → HVG → scale → PCA-50). Point `DATA` at any
`.h5ad`; set `SUBSAMPLE=None` for all cells.""")

code("""DATA = "/Users/iandriver/Downloads/SingleRust/data/bench_input.h5ad"
SUBSAMPLE = 20_000

a = ad.read_h5ad(DATA)
a.var_names_make_unique()
if SUBSAMPLE and a.n_obs > SUBSAMPLE:
    a = a[RNG.choice(a.n_obs, SUBSAMPLE, replace=False)].copy()
sc.pp.filter_genes(a, min_cells=3)
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=2000)
a = a[:, a.var.highly_variable].copy()
sc.pp.scale(a, max_value=10)
sc.tl.pca(a, n_comps=50)
a.obsm["X_pca"] = a.obsm["X_pca"].astype(np.float32)
X = np.ascontiguousarray(a.obsm["X_pca"], dtype=np.float32)
print(f"{a.n_obs} cells x {X.shape[1]} PCs")""")

md("""## 2. How a custom NN backend plugs into scanpy

scanpy (≥1.10) calls `transformer.fit_transform(X)` and expects a sparse `(n,n)`
**distance** graph (k+1 entries/row, self first); it computes the UMAP
connectivities itself. So every backend is interchangeable. Here's the PiPNN
transformer's core — it only returns the kNN distance graph:""")

code("""from pipnn import PiPNNTransformer
print(inspect.getsource(PiPNNTransformer.transform))""")

md("""## 3. Build the neighbor graph with each backend (cold vs. warm)

`bench_lib.available_backends` returns every backend whose library is installed
(`glass` included automatically where importable). We time a **cold** build then
take the **median of 3 warm** builds.""")

code("""backends = bl.available_backends(K)
print("backends:", list(backends.keys()))

results = {}
for name, make in backends.items():
    results[name] = bl.timed_neighbors(make, a, K, repeats=3)

print(f"\\n{'backend':14s} {'cold(s)':>9s} {'warm(s)':>9s} {'conn_nnz':>10s}")
for name, r in results.items():
    print(f"{name:14s} {r['cold']:9.2f} {r['warm_median']:9.2f} {r['conn_nnz']:10d}")
print("\\n(warm = median steady-state; cold includes numba JIT for pynndescent)")""")

md("""## 4. Recall@k vs. exact, and the timing picture

Recall is the fraction of each cell's true k nearest neighbors recovered. The
timing chart shows cold vs. warm side by side — note how much of pynndescent's
cold time is one-time JIT.""")

code("""exact_idx = bl.exact_knn(X, K)
for name, r in results.items():
    r["recall"] = bl.recall_from_obsp(r["adata"], exact_idx, K)
    print(f"{name:14s} recall@{K} = {r['recall']:.4f}")""")

code("""names = list(results.keys())
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar(x - w/2, [results[n]['cold'] for n in names], w, label='cold (incl. JIT)', color='#bbbbbb')
ax[0].bar(x + w/2, [results[n]['warm_median'] for n in names], w, label='warm (steady-state)', color='#2a9d8f')
ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_ylabel('sc.pp.neighbors (s)')
ax[0].set_title('Build time: cold vs warm'); ax[0].legend()
ax[1].bar(x, [results[n]['recall'] for n in names], color='#e76f51')
ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylim(0.9, 1.001)
ax[1].set_title(f'recall@{K} vs exact'); ax[1].set_ylabel('recall')
plt.tight_layout(); plt.show()""")

md("""## 5. UMAP embeddings, side by side

Each backend's graph → its own UMAP, all colored by the **same** reference labels
(Leiden on the exact graph). Matching colors in the same places = the embedding is
reproduced despite the approximate graph.""")

code("""sc.tl.leiden(results["exact"]["adata"], flavor="igraph", n_iterations=2, key_added="ref")
ref = results["exact"]["adata"].obs["ref"].values
codes = results["exact"]["adata"].obs["ref"].cat.codes.values
cmap = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ref))))

for name, r in results.items():
    r["adata"].obs["ref"] = ref
    sc.tl.umap(r["adata"])

names = list(results.keys())
fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 5))
if len(names) == 1: axes = [axes]
for ax, name in zip(axes, names):
    U = results[name]["adata"].obsm["X_umap"]
    ax.scatter(U[:, 0], U[:, 1], c=cmap[codes], s=2, linewidths=0)
    ax.set_title(f"{name}  (recall={results[name]['recall']:.3f})")
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()""")

md("""## 6. Clustering agreement (ARI vs. exact)

Leiden on each backend's graph, scored against the exact-graph clustering. ARI ≈ 1
means the downstream biology is unchanged by swapping ANN backend.""")

code("""for name, r in results.items():
    sc.tl.leiden(r["adata"], flavor="igraph", n_iterations=2, key_added="leiden")
ref_leiden = results["exact"]["adata"].obs["leiden"]
print(f"{'backend':14s} {'#clusters':>9s} {'ARI vs exact':>13s}")
for name, r in results.items():
    ari = adjusted_rand_score(ref_leiden, r["adata"].obs["leiden"])
    print(f"{name:14s} {r['adata'].obs['leiden'].nunique():9d} {ari:13.4f}")""")

md("""## 7. Neighbor-graph structure

Distance distributions and connectivity counts should be near-identical across
backends — confirming the graphs are interchangeable downstream.""")

code("""fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
for (name, r), col in zip(results.items(), colors):
    ax[0].hist(r["adata"].obsp["distances"].data, bins=60, histtype="step",
               color=col, label=name, density=True)
ax[0].set_title("kNN distance distribution"); ax[0].set_xlabel("distance"); ax[0].legend()
ax[1].bar(list(results.keys()), [r["conn_nnz"] for r in results.values()], color=colors)
ax[1].set_title("connectivities nnz (graph density)")
plt.tight_layout(); plt.show()""")

md("""## 8. Reading the results

- **One-line swap.** Every backend is a `transformer=` argument; connectivities,
  UMAP, and Leiden are identical machinery downstream.
- **Warm vs cold matters.** pynndescent's cold time is inflated by numba JIT; the
  warm numbers are the fair comparison. PiPNN (Rust) and glass (C++) have no JIT,
  so their cold ≈ warm. PiPNN's build-time advantage grows with `n` (the win is at
  100k–1M+ cells, where exact brute force is O(n²)).
- **Quality.** All backends recover the embedding and clustering at high recall /
  ARI, so the choice is about build speed and scaling, not result quality.

### Backends under the hood
- **PiPNN** (Rust): ball carving → leaf GEMM → HashPrune (residualized-LSH
  reservoirs) → RobustPrune → BeamSearch.
- **pynndescent** (numba): NN-Descent from random-projection-tree init.
- **pyglass** (C++): HNSW/NSG graph with SIMD + product/scalar quantization.
""")

nb["cells"] = c
nb.metadata["kernelspec"] = {"name": "pipnn-venv", "display_name": "PiPNN (venv)", "language": "python"}
with open("notebooks/pipnn_vs_pynndescent.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/pipnn_vs_pynndescent.ipynb with", len(c), "cells")
