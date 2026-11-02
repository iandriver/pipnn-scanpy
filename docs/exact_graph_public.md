# Public exact-search and downstream example

This case uses all 2,638 cells and 50 stored PCs from
`scanpy.datasets.pbmc3k_processed()`. It compares search methods on the same
coordinates, then separates neighbor changes from graph-weighting changes.
The [recorded public report](examples/public_knn/report.md)
contains figures and per-group results. It covers only this public dataset.

Four-thread settings, median of three fresh-process warm repeats on macOS arm64:

| Builder | Seconds | Whole-worker peak RSS | Recall@15 |
|---|---:|---:|---:|
| FAISS exhaustive | 0.0030 | 69 MB | 100% |
| sklearn brute-force | 0.0098 | 156 MB | 100% |
| PiPNN direct, dense, partition limit 512, eight passes | 0.0397 | 166 MB | 99.9823% |
| Scanpy-configured PyNNDescent, forced ANN | 0.4595 | 629 MB | 97.0634% |

Scanpy already automatically selects exact search for a dataset this small;
the ANN controls deliberately force ANN. The recorded run used randomized serial ordering for all six methods. Whole-process memory includes unequal dependency overhead, and
other jobs ran on this interactive machine. This is an integration example,
not a dedicated-machine speed ranking or a large-data benchmark.

The two exhaustive methods agree on all neighbor IDs. For the 15-cell
megakaryocyte annotation, direct construction and exact search both recover all
neighbors. Every graph uses self plus 15 nonself neighbors; the shared Scanpy
connectivity function therefore receives `n_neighbors=16`.

With fixed UMAP weighting, direct-versus-exact Leiden agreement averages 0.995
ARI across three same-seed comparisons; exact-graph seed stability averages
0.988. Holding exact neighbors fixed, Scanpy Jaccard weighting gives ARI 0.948
against UMAP weighting, and unweighted union edges give 0.919. Mutual-only edges
leave 645 cells isolated. Graph design can change the result even when neighbor
search is exhaustive. These differences do not establish which graph is
biologically better; the annotations came from the original tutorial analysis.

## Reproduce from this branch

On Linux or macOS, use Python 3.12 and a Rust toolchain. With `uv` installed:

```bash
git clone --branch codex/public-knn-reproduction https://github.com/iandriver/pipnn-scanpy.git
cd pipnn-scanpy
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python -r examples/requirements-public-knn.lock.txt
env -u CONDA_PREFIX .venv/bin/maturin develop --release --uv
.venv/bin/python bench/bench_exact_graph.py prepare
.venv/bin/python bench/bench_exact_graph.py search --threads 4
.venv/bin/python bench/bench_exact_graph.py graphs --threads 4
.venv/bin/python bench/audit_exact_disagreements.py
.venv/bin/python bench/report_exact_graph.py
```

The first preparation downloads the public dataset. Results and PNG/SVG figures
are written to `bench/results/public_knn/`; inputs are cached under
`bench/data/public_knn/`. Use a new output directory when changing settings;
`search` resumes completed worker receipts. These options require a source
build and are not in the released v0.1.1 wheel. The existing index path remains
the default. Direct construction is opt-in:

```python
from pipnn import PiPNNTransformer
transformer = PiPNNTransformer(
    algorithm="direct", kernel="dense", n_neighbors=15,
    c_max=512, runs=8, n_jobs=4, random_state=0,
)
# sc.pp.neighbors(adata, use_rep="X_pca", transformer=transformer)
```

`kernel="fused"` is also available in direct mode. It avoids a square leaf
distance matrix; its speed/memory tradeoff depends on the workload. This small
case cannot establish that tradeoff.

The complete tested environment is pinned in
`examples/requirements-public-knn.lock.txt`; receipts record versions, code and
input fingerprints. The fresh environment passed 67 Python tests and 11 Rust
tests, followed by all 18 search measurements and 10 graph checks. The explicit union-SNN variant uses Jaccard overlap of the outgoing
15-neighbor lists on union-kNN support, without pruning. Scanpy's Jaccard kernel
averages directed weights on symmetrization, giving half the Jaccard weight to
an edge appearing in just one direction. These are deliberately separate
variants. All clustering uses igraph Leiden, resolution 1, two iterations and
seeds 0/1/2. Selected distances are recomputed in float64 then cast to float32
before the shared UMAP connectivity call, so graph differences reflect neighbor
IDs rather than backend distance arithmetic.

## Measurement definitions

Timing excludes imports, loading and method warmup/JIT. It includes native
construction/search and each API's output; common self-edge normalization and
evaluation are outside the timer. Peak process RSS includes runtime, input,
warmup, scratch and output, with different dependency overhead by method.
Scanpy-configured PyNNDescent uses the actual transformer, including CSR
assembly and index compression; the bare-library control only extracts its
neighbor graph. These paths are labeled separately.

Sklearn is the accuracy reference. Every disagreement between the two exhaustive
implementations is checked against float64 exhaustive distances. Selected
pair distances are recomputed in float64 and cast to float32 before common
UMAP weighting. The connectivity function receives self plus 15 neighbors
(`n_neighbors=16`), avoiding a hidden change in the self-inclusive count. This
convention is checked against Scanpy's transformer hook in the tests.

Graph checks include edge overlap, relative absolute weight change, connected
components, per-group internal fragmentation and cross-annotation weight.
A rare group is below 1% of cells. Per-group connections are descriptive;
annotation boundaries are not an independent biological truth. Clustering
uses igraph Leiden, resolution 1, two iterations and seeds 0/1/2. The comparison
does not validate marker genes or biological outcomes.
