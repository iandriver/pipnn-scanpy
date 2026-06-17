"""Shared benchmark utilities: warmup + median-of-N timing, cold-vs-warm capture,
thread matching, recall, and a registry of available scanpy NN backends.

The key methodology fix vs. naive timing: the **first** build of a backend pays
one-time costs (notably pynndescent's numba JIT compilation). We report that as
``cold`` but base the comparison on ``warm`` — the median of subsequent steady-
state builds — so PiPNN vs. pynndescent vs. glass is apples-to-apples.
"""

from __future__ import annotations

import os
import statistics
import time

import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


def match_threads(n_threads: int | None):
    """Pin OpenMP/numba thread counts so all backends use the same core count.

    Call before importing pynndescent/glass for full effect; also pass the same
    value as each transformer's ``n_jobs``.
    """
    if n_threads and n_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["NUMBA_NUM_THREADS"] = str(n_threads)


# ---------------------------------------------------------------------------
# Backend registry — each entry present only if its library imports.
# A "backend" is a callable (n_neighbors, n_jobs) -> fresh transformer instance.
# ---------------------------------------------------------------------------

def available_backends(k: int, n_jobs: int = -1):
    backends = {}

    from pipnn import PiPNNTransformer
    backends["PiPNN"] = lambda: PiPNNTransformer(n_neighbors=k, n_jobs=n_jobs)

    try:
        from pynndescent import PyNNDescentTransformer
        backends["pynndescent"] = lambda: PyNNDescentTransformer(
            n_neighbors=k, metric="euclidean", n_jobs=n_jobs)
    except Exception:
        pass

    try:
        from pipnn.contrib import GlassTransformer
        GlassTransformer._import_glass()  # probe importability (glassppy or glass)
        backends["glass"] = lambda: GlassTransformer(n_neighbors=k, n_jobs=n_jobs)
    except Exception:
        pass

    from sklearn.neighbors import KNeighborsTransformer
    backends["exact"] = lambda: KNeighborsTransformer(
        n_neighbors=k, mode="distance", n_jobs=(None if n_jobs in (-1, 0) else n_jobs))

    return backends


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _build(adata, make_transformer, k):
    a = adata.copy()
    sc.pp.neighbors(a, n_neighbors=k, use_rep="X_pca", transformer=make_transformer())
    return a


def timed_neighbors(make_transformer, adata, k, repeats: int = 3):
    """Return dict with cold (first build, incl. JIT) and warm steady-state times.

    ``cold`` = first build wall time. ``warm_median``/``warm_min`` = median/min of
    ``repeats`` subsequent builds. The returned ``adata`` is from the warm run.
    """
    t0 = time.perf_counter()
    a = _build(adata, make_transformer, k)
    cold = time.perf_counter() - t0

    warm = []
    for _ in range(repeats):
        t = time.perf_counter()
        a = _build(adata, make_transformer, k)
        warm.append(time.perf_counter() - t)

    return {
        "cold": cold,
        "warm_median": statistics.median(warm),
        "warm_min": min(warm),
        "adata": a,
        "conn_nnz": int(a.obsp["connectivities"].nnz),
    }


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

def exact_knn(X, k):
    """Exact kNN indices (excluding self), shape (n, k)."""
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return idx[:, 1:]


def knn_from_obsp(adata, k):
    """Per-row top-k neighbor ids (excluding self) from obsp['distances']."""
    D = adata.obsp["distances"].tocsr()
    n = D.shape[0]
    out = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        s, e = D.indptr[i], D.indptr[i + 1]
        cols, vals = D.indices[s:e], D.data[s:e]
        order = np.argsort(vals)
        nbrs = [c for c in cols[order] if c != i][:k]
        out[i, : len(nbrs)] = nbrs
    return out

def recall_from_obsp(adata, exact_idx, k):
    approx = knn_from_obsp(adata, k)
    n = approx.shape[0]
    return float(np.mean([
        len(set(approx[i]) & set(exact_idx[i])) / k for i in range(n)
    ]))


# ---------------------------------------------------------------------------
# One-call comparison
# ---------------------------------------------------------------------------

def run_comparison(adata, k=15, repeats=3, n_jobs=-1, recall_sample=20000, rng_seed=0):
    """Time + recall every available backend on ``adata.obsm['X_pca']``."""
    X = np.ascontiguousarray(adata.obsm["X_pca"], dtype=np.float32)
    n = X.shape[0]
    sub = np.random.default_rng(rng_seed).choice(n, min(n, recall_sample), replace=False)
    exact = exact_knn(X, k)

    rows = {}
    for name, make in available_backends(k, n_jobs).items():
        r = timed_neighbors(make, adata, k, repeats=repeats)
        # exact is indexed by original cell id; compare on the sampled cells.
        r["recall"] = _recall_subset(r["adata"], exact, sub, k)
        rows[name] = r
    return rows


def _recall_subset(adata, exact_idx, sub, k):
    approx = knn_from_obsp(adata, k)
    return float(np.mean([
        len(set(approx[i]) & set(exact_idx[i])) / k for i in sub
    ]))


def print_table(rows):
    print(f"{'backend':14s} {'cold(s)':>9s} {'warm(s)':>9s} {'recall@k':>9s} {'conn_nnz':>10s}")
    for name, r in rows.items():
        print(f"{name:14s} {r['cold']:9.2f} {r['warm_median']:9.2f} "
              f"{r['recall']:9.4f} {r['conn_nnz']:10d}")
    print("\n(warm = median steady-state; cold includes one-time JIT/setup)")
