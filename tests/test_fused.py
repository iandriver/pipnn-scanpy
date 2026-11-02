"""Fused scoring contracts independent of real-data performance results."""
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from pipnn import direct_knn_graph


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_fused_full_leaf_against_exact(metric):
    x = np.random.default_rng(52).normal(size=(131, 50)).astype(np.float32)
    ids, distances = direct_knn_graph(x, 15, metric=metric, kernel='fused', n_jobs=2)
    truth_d, truth_i = NearestNeighbors(n_neighbors=16, metric=metric).fit(x).kneighbors(x)
    for i in range(len(x)):
        assert ids[i, 0] == i
        assert set(ids[i, 1:]) == set(truth_i[i][truth_i[i] != i][:15])
    if metric == 'euclidean':
        actual = np.linalg.norm(x[:, None] - x[ids], axis=2)
    else:
        normed = x / np.linalg.norm(x, axis=1, keepdims=True)
        actual = 1 - (normed[:, None] * normed[ids]).sum(axis=2)
    np.testing.assert_allclose(distances, actual, atol=2e-6, rtol=2e-6)


def test_fused_partition_merge_reproduces_dense_and_is_thread_deterministic():
    x = np.random.default_rng(53).normal(size=(4500, 17)).astype(np.float32)
    kwargs = dict(n_neighbors=15, c_max=256, runs=2)
    dense = direct_knn_graph(x, kernel='dense', n_jobs=1, **kwargs)
    fused = direct_knn_graph(x, kernel='fused', n_jobs=1, **kwargs)
    threaded = direct_knn_graph(x, kernel='fused', n_jobs=4, **kwargs)
    for reference, candidate, parallel in zip(dense, fused, threaded):
        np.testing.assert_array_equal(reference, candidate)
        np.testing.assert_array_equal(candidate, parallel)


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match='kernel'):
        direct_knn_graph(np.ones((20, 5)), 3, kernel='unknown')
