"""Tests for the external-query path (PiPNNIndex / query_knn) and the multi-run
build knob — the ANN-benchmark machinery, distinct from the self-kNN transformer.
"""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from pipnn import PiPNNIndex, query_knn, self_knn_graph


@pytest.fixture(scope="module")
def base_query():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(8, 48)) * 6
    X = np.vstack([c + rng.normal(size=(700, 48)) for c in centers]).astype(np.float32)
    Q = (rng.normal(size=(200, 48)) * 6)[rng.integers(0, 8, 200)] \
        + rng.normal(size=(200, 48))
    return X.astype(np.float32), Q.astype(np.float32)


def _recall(approx, gt, k):
    return np.mean([len(set(approx[i]) & set(gt[i])) / k for i in range(len(gt))])


def test_query_knn_shapes_and_recall(base_query):
    X, Q = base_query
    k = 10
    idx, dist = query_knn(X, Q, n_neighbors=k, runs=3, beam_L=512, random_state=0)
    assert idx.shape == (len(Q), k)
    assert dist.shape == (len(Q), k)
    # distances sorted ascending per row
    assert np.all(np.diff(dist, axis=1) >= -1e-4)
    _, gt = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Q)
    # External queries on a small clustered set should be reachable with a wide
    # beam — not perfect (graph navigability is the documented limitation), but
    # clearly better than chance.
    assert _recall(idx, gt, k) >= 0.80


def test_persistent_index_matches_oneshot(base_query):
    X, Q = base_query
    k = 10
    one_idx, _ = query_knn(X, Q, n_neighbors=k, runs=2, beam_L=256, random_state=0)
    index = PiPNNIndex(X, runs=2, random_state=0)
    persist_idx, _ = index.query(Q, n_neighbors=k, beam_L=256)
    # Same build + same search params → identical results (determinism).
    assert np.array_equal(one_idx, persist_idx)


def test_more_runs_help_self_knn_recall(base_query):
    """The multi-run knob should be monotone-ish for self-kNN recall."""
    X, _ = base_query
    k = 10
    _, gt = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    gt = gt[:, 1:]
    r1 = _recall(self_knn_graph(X, n_neighbors=k, runs=1, random_state=0)[0][:, 1:], gt, k)
    r3 = _recall(self_knn_graph(X, n_neighbors=k, runs=3, random_state=0)[0][:, 1:], gt, k)
    assert r3 >= r1 - 1e-6
    assert r3 >= 0.95


def test_query_dim_mismatch_raises(base_query):
    X, Q = base_query
    with pytest.raises(Exception):
        PiPNNIndex(X, random_state=0).query(Q[:, :10], n_neighbors=5, beam_L=64)
