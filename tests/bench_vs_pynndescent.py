"""Synthetic-scale benchmark via the shared bench_lib (warmup + median timing).

    .venv/bin/python tests/bench_vs_pynndescent.py [n] [d] [k]

Times + scores every available backend (PiPNN, pynndescent, glass-if-installed,
exact) on clustered synthetic data with cold-vs-warm timing and recall@k.
"""

import sys
from pathlib import Path

import numpy as np
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import bench_lib as bl  # noqa: E402


def make_data(n, d, n_clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d)) * 6
    sizes = [n // n_clusters] * n_clusters
    sizes[-1] += n - sum(sizes)
    X = np.vstack([c + rng.normal(size=(s, d)) for c, s in zip(centers, sizes)])
    return X.astype(np.float32)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    X = make_data(n, d)
    a = ad.AnnData(X)
    a.obsm["X_pca"] = X
    print(f"n={n} d={d} k={k}\n")
    rows = bl.run_comparison(a, k=k, repeats=3)
    bl.print_table(rows)


if __name__ == "__main__":
    main()
