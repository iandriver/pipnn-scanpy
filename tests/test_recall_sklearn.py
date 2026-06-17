"""Recall@k of the PiPNN self-kNN against exact sklearn brute force."""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from pipnn import self_knn_graph


def _recall(approx_idx, exact_idx):
    """Mean fraction of true neighbors recovered, excluding the self column."""
    n, k = exact_idx.shape
    hits = 0
    for i in range(n):
        a = set(approx_idx[i, 1:].tolist())  # drop self at col 0
        e = set(exact_idx[i, 1:].tolist())
        hits += len(a & e)
    return hits / (n * (k - 1))


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_recall_synthetic(metric):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 50)).astype(np.float32)
    k = 15

    idx, _ = self_knn_graph(X, n_neighbors=k, metric=metric, random_state=0)

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, exact = nn.kneighbors(X)

    recall = _recall(idx, exact)
    # Phase 0 is exact brute force → recall should be ~1.0.
    assert recall >= 0.95, f"recall {recall:.3f} below threshold ({metric})"


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_recall_graph_path(metric):
    # n above the brute-force threshold → exercises the real PiPNN graph build
    # + BeamSearch self-query.
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(8, 50)) * 5
    X = np.vstack([c + rng.normal(size=(1000, 50)) for c in centers]).astype(np.float32)
    k = 15

    idx, _ = self_knn_graph(X, n_neighbors=k, metric=metric, random_state=0)

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, exact = nn.kneighbors(X)

    recall = _recall(idx, exact)
    assert recall >= 0.95, f"graph-path recall {recall:.3f} below threshold ({metric})"


def test_determinism():
    # Same seed → byte-identical neighbor graph (HashPrune is history-independent
    # and the build is deterministic).
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6000, 50)).astype(np.float32)
    a_idx, a_dist = self_knn_graph(X, n_neighbors=15, random_state=42)
    b_idx, b_dist = self_knn_graph(X, n_neighbors=15, random_state=42)
    assert np.array_equal(a_idx, b_idx)
    assert np.array_equal(a_dist, b_dist)
