"""Pin the scanpy/sklearn KNeighborsTransformer contract.

These assertions must stay green through every implementation phase — they are
what guarantees scanpy's downstream UMAP/Leiden math works.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone

from pipnn import PiPNNTransformer


@pytest.fixture(params=[{}, {"algorithm": "direct", "kernel": "dense"},
                        {"algorithm": "direct", "kernel": "fused"}])
def options(request):
    return request.param


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    # 3 well-separated gaussian blobs in 8-d.
    centers = rng.normal(size=(3, 8)) * 10
    X = np.vstack([c + rng.normal(size=(200, 8)) for c in centers])
    return X.astype(np.float32)


def test_output_shape_and_type(data, options):
    k = 15
    g = PiPNNTransformer(n_neighbors=k, **options).fit_transform(data)
    n = data.shape[0]
    assert isinstance(g, csr_matrix)
    assert g.shape == (n, n)


def test_k_plus_one_explicit_entries_per_row(data, options):
    k = 15
    g = PiPNNTransformer(n_neighbors=k, **options).fit_transform(data)
    counts = np.diff(g.indptr)
    # Exactly k+1 stored entries per row (self + k neighbors).
    assert np.all(counts == k + 1)


def test_self_edge_first_and_explicit_zero(data, options):
    k = 10
    g = PiPNNTransformer(n_neighbors=k, **options).fit_transform(data)
    n = data.shape[0]
    for i in range(0, n, 37):
        start = g.indptr[i]
        # First stored neighbor of row i is the point itself...
        assert g.indices[start] == i, f"row {i}: self edge not first"
        # ...stored as an explicit 0.0 (present in .data, not pruned).
        assert g.data[start] == 0.0


def test_distances_sorted_ascending(data, options):
    k = 10
    g = PiPNNTransformer(n_neighbors=k, **options).fit_transform(data)
    n = data.shape[0]
    for i in range(0, n, 53):
        row = g.data[g.indptr[i]:g.indptr[i + 1]]
        assert np.all(np.diff(row) >= -1e-6), f"row {i} distances not sorted"


def test_get_params_roundtrip_and_clone(data, options):
    t = PiPNNTransformer(n_neighbors=20, metric="cosine", **options)
    assert t.get_params()["n_neighbors"] == 20
    t2 = clone(t)
    assert t2.get_params() == t.get_params()


def test_cosine_metric_runs(data, options):
    g = PiPNNTransformer(n_neighbors=10, metric="cosine", **options).fit_transform(data)
    assert g.shape[0] == data.shape[0]
    # cosine distance is in [0, 2]
    assert g.data.min() >= -1e-6
    assert g.data.max() <= 2.0 + 1e-4


@pytest.mark.parametrize("options", [
    {"algorithm": "unknown"}, {"kernel": "unknown"}, {"kernel": "fused"},
])
def test_invalid_mode_rejected_on_fit(data, options):
    t = PiPNNTransformer(**options)  # sklearn constructors store, fit validates
    with pytest.raises(ValueError):
        t.fit(data)


def test_default_routes_to_existing_backend(data, monkeypatch):
    from pipnn import _pipnn

    original = _pipnn.build_and_self_knn
    calls = []

    def index(*args):
        calls.append(args)
        return original(*args)

    def direct(*args):
        pytest.fail("default transformer called direct builder")

    monkeypatch.setattr(_pipnn, "build_and_self_knn", index)
    monkeypatch.setattr(_pipnn, "build_direct_knn", direct)
    default = PiPNNTransformer(n_jobs=2).fit_transform(data)
    explicit = PiPNNTransformer(algorithm="index", n_jobs=2).fit_transform(data)
    assert len(calls) == 2
    np.testing.assert_array_equal(default.indices, explicit.indices)
    np.testing.assert_array_equal(default.data, explicit.data)


@pytest.mark.parametrize("kernel", ["dense", "fused"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_direct_matches_standalone(data, kernel, metric, monkeypatch):
    from pipnn import _pipnn, direct_knn_graph

    def index(*args):
        pytest.fail("direct transformer called index builder")

    monkeypatch.setattr(_pipnn, "build_and_self_knn", index)
    params = dict(n_neighbors=10, metric=metric, c_max=128, fanout=2,
                  runs=3, n_jobs=2, random_state=17, kernel=kernel)
    ids, distances = direct_knn_graph(data, **params)
    graph = PiPNNTransformer(algorithm="direct", **params).fit_transform(data)
    np.testing.assert_array_equal(graph.indices.reshape(ids.shape), ids)
    np.testing.assert_array_equal(graph.data.reshape(ids.shape), distances)
