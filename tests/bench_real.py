"""Real single-cell benchmark: standard scanpy preprocessing -> PCA -> PiPNN.

    .venv/bin/python tests/bench_real.py <file.h5ad> [k]

Reports PiPNN vs pynndescent build time and PiPNN recall@k vs exact kNN on the
real PCA embedding, then runs the full sc.pp.neighbors + UMAP + Leiden pipeline.
"""

import sys
import time
import resource

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors

from pipnn import PiPNNTransformer, self_knn_graph


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / 1e9) if sys.platform == "darwin" else (r / 1e6)


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


def recall(approx_idx, exact_idx):
    n, kp1 = exact_idx.shape
    k = kp1 - 1
    return sum(
        len(set(approx_idx[i, 1:]) & set(exact_idx[i, 1:])) for i in range(n)
    ) / (n * k)


def main():
    path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    t = time.time()
    a = preprocess(path)
    X = np.ascontiguousarray(a.obsm["X_pca"], dtype=np.float32)
    print(f"{path.split('/')[-1]}: {a.n_obs} cells x {X.shape[1]} PCs "
          f"(preprocess {time.time()-t:.1f}s)", flush=True)

    t = time.time()
    idx, _ = self_knn_graph(X, n_neighbors=k, random_state=0)
    t_pip = time.time() - t
    print(f"[pipnn]  build+query: {t_pip:6.2f}s  peakRSS={rss_gb():.2f}GB", flush=True)

    try:
        from pynndescent import PyNNDescentTransformer
        t = time.time()
        PyNNDescentTransformer(n_neighbors=k, metric="euclidean").fit_transform(X)
        t_pyn = time.time() - t
        print(f"[pynnd]  build+query: {t_pyn:6.2f}s  speedup={t_pyn/t_pip:.2f}x", flush=True)
    except Exception as e:
        print(f"[pynnd]  skipped: {e}", flush=True)

    m = min(a.n_obs, 20000)
    sub = np.random.default_rng(0).choice(a.n_obs, m, replace=False)
    _, exact = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[sub])
    print(f"[pipnn]  recall@{k} (on {m} cells): {recall(idx[sub], exact):.4f}", flush=True)

    # Full pipeline through the scanpy transformer hook.
    t = time.time()
    sc.pp.neighbors(a, n_neighbors=k, use_rep="X_pca", transformer=PiPNNTransformer(n_neighbors=k))
    sc.tl.leiden(a, flavor="igraph", n_iterations=2)
    print(f"[scanpy] neighbors+leiden: {time.time()-t:.2f}s  "
          f"clusters={a.obs['leiden'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
