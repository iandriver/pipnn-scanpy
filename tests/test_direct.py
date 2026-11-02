"""Contracts for the experimental direct graph API."""
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from pipnn import direct_knn_graph


def test_full_leaf_exact_and_original_distance_units():
    x = np.random.default_rng(23).normal(size=(301, 9)).astype(np.float32)
    x[200] = x[0]
    ids, distances = direct_knn_graph(x, 15, c_max=1024, n_jobs=2)
    truth_d, truth_i = NearestNeighbors(n_neighbors=17, algorithm='brute').fit(x).kneighbors(x)
    for i in range(len(x)):
        assert ids[i, 0] == i and distances[i, 0] == 0
        expected = truth_i[i][truth_i[i] != i][:15]
        assert set(ids[i, 1:]) == set(expected)
    actual = np.linalg.norm(x[:, None] - x[ids], axis=2)
    np.testing.assert_allclose(distances, actual, atol=2e-6, rtol=2e-6)


def test_invalid_input_is_rejected():
    for x in [np.empty((0, 4)), np.empty((20, 0)), np.full((20, 4), np.nan)]:
        with pytest.raises(ValueError):
            direct_knn_graph(x, 3)
    with pytest.raises(ValueError):
        direct_knn_graph(np.ones((20, 4)), 20)
    with pytest.raises(ValueError):
        direct_knn_graph(np.ones((20, 4)), 3, runs=0)


def test_partitioned_path_and_zero_neighbors():
    x = np.random.default_rng(24).normal(size=(4500, 8)).astype(np.float32)
    a, da = direct_knn_graph(x, 15, c_max=256, runs=2, n_jobs=1)
    b, db = direct_knn_graph(x, 15, c_max=256, runs=2, n_jobs=4)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(da, db)
    assert all(len(set(row)) == 16 for row in a)
    self_ids, self_ds = direct_knn_graph(x, 0, n_jobs=2)
    np.testing.assert_array_equal(self_ids[:, 0], np.arange(len(x)))
    assert not self_ds.any()
