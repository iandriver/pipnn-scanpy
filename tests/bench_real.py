"""Real single-cell benchmark via the shared bench_lib (warmup + median timing).

    .venv/bin/python tests/bench_real.py <file.h5ad> [k]

Preprocesses (normalize -> log1p -> HVG -> scale -> PCA-50) then times + scores
every available backend (PiPNN, pynndescent, glass-if-installed, exact) with
cold-vs-warm timing and recall@k vs exact.
"""

import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import bench_lib as bl  # noqa: E402


def preprocess(path):
    a = ad.read_h5ad(path)
    a.var_names_make_unique()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=50)
    a.obsm["X_pca"] = a.obsm["X_pca"].astype(np.float32)
    return a


def main():
    path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    a = preprocess(path)
    print(f"{Path(path).name}: {a.n_obs} cells x {a.obsm['X_pca'].shape[1]} PCs\n")
    rows = bl.run_comparison(a, k=k, repeats=3)
    bl.print_table(rows)


if __name__ == "__main__":
    main()
