"""Pin the scanpy/sklearn KNeighborsTransformer contract.

These assertions must stay green through every implementation phase — they are
what guarantees scanpy's downstream UMAP/Leiden math works.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone

from pipnn import PiPNNTransformer


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    # 3 well-separated gaussian blobs in 8-d.
    centers = rng.normal(size=(3, 8)) * 10
    X = np.vstack([c + rng.normal(size=(200, 8)) for c in centers])
    return X.astype(np.float32)


def test_output_shape_and_type(data):
    k = 15
    g = PiPNNTransformer(n_neighbors=k).fit_transform(data)
    n = data.shape[0]
    assert isinstance(g, csr_matrix)
    assert g.shape == (n, n)


def test_k_plus_one_explicit_entries_per_row(data):
    k = 15
    g = PiPNNTransformer(n_neighbors=k).fit_transform(data)
    counts = np.diff(g.indptr)
    # Exactly k+1 stored entries per row (self + k neighbors).
    assert np.all(counts == k + 1)


def test_self_edge_first_and_explicit_zero(data):
    k = 10
    g = PiPNNTransformer(n_neighbors=k).fit_transform(data)
    n = data.shape[0]
    for i in range(0, n, 37):
        start = g.indptr[i]
        # First stored neighbor of row i is the point itself...
        assert g.indices[start] == i, f"row {i}: self edge not first"
        # ...stored as an explicit 0.0 (present in .data, not pruned).
        assert g.data[start] == 0.0


def test_distances_sorted_ascending(data):
    k = 10
    g = PiPNNTransformer(n_neighbors=k).fit_transform(data)
    n = data.shape[0]
    for i in range(0, n, 53):
        row = g.data[g.indptr[i]:g.indptr[i + 1]]
        assert np.all(np.diff(row) >= -1e-6), f"row {i} distances not sorted"


def test_get_params_roundtrip_and_clone(data):
    t = PiPNNTransformer(n_neighbors=20, metric="cosine")
    assert t.get_params()["n_neighbors"] == 20
    t2 = clone(t)
    assert t2.get_params()["metric"] == "cosine"


def test_cosine_metric_runs(data):
    g = PiPNNTransformer(n_neighbors=10, metric="cosine").fit_transform(data)
    assert g.shape[0] == data.shape[0]
    # cosine distance is in [0, 2]
    assert g.data.min() >= -1e-6
    assert g.data.max() <= 2.0 + 1e-4
