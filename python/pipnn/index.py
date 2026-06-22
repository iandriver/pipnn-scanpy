"""Standalone PiPNN index wrapper (without the sklearn/scanpy machinery).

Convenience for direct k-NN graph construction outside an AnnData workflow.
"""

from __future__ import annotations

import numpy as np

from . import _pipnn

__all__ = ["self_knn_graph", "query_knn", "PiPNNIndex"]


class PiPNNIndex:
    """Persistent PiPNN index: build once, query an external set many times.

    The shape the ANN-benchmark harness needs — build cost is paid once and only
    the per-query ``beam_L`` sweep is timed.

        idx = PiPNNIndex(base, runs=3)
        ids, dists = idx.query(queries, n_neighbors=10, beam_L=256)

    ``runs`` is the build-side recall knob; ``beam_L`` the search-side one.
    """

    def __init__(
        self,
        X,
        *,
        metric: str = "euclidean",
        m: int = 12,
        l_max: int = 96,
        R: int = 64,
        alpha: float = 1.2,
        fanout: int = 2,
        c_min: int = 256,
        c_max: int = 2048,
        runs: int = 1,
        n_jobs: int = -1,
        random_state: int = 0,
    ):
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_jobs = 0 if n_jobs in (None, -1) else int(n_jobs)
        self._idx = _pipnn.PiPNNIndex(
            X, str(metric), int(m), int(l_max), int(R), float(alpha),
            int(fanout), int(c_min), int(c_max), int(runs), int(n_jobs),
            int(random_state),
        )

    def query(self, queries, n_neighbors: int = 10, *, beam_L: int = 64):
        """Return ``(indices, distances)`` of shape ``(n_queries, n_neighbors)``."""
        Q = np.ascontiguousarray(queries, dtype=np.float32)
        n_q = Q.shape[0]
        indices, distances, stride = self._idx.query(Q, int(n_neighbors), int(beam_L))
        return indices.reshape(n_q, stride), distances.reshape(n_q, stride)


def self_knn_graph(
    X,
    n_neighbors: int = 15,
    *,
    metric: str = "euclidean",
    m: int = 12,
    l_max: int = 96,
    R: int = 64,
    alpha: float = 1.2,
    beam_L: int = 64,
    fanout: int = 2,
    c_min: int = 256,
    c_max: int = 2048,
    runs: int = 1,
    n_jobs: int = -1,
    random_state: int = 0,
):
    """Build the index over ``X`` and return ``(indices, distances)`` arrays.

    Both are shape ``(n, n_neighbors + 1)``; column 0 is the self edge.

    ``runs`` is the number of independent Randomized-Ball-Carving passes whose
    candidates are unioned (the paper's recall/speed knob; more runs → higher
    recall at ~linear cost).
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    n_jobs = 0 if n_jobs in (None, -1) else int(n_jobs)
    indices, distances, stride = _pipnn.build_and_self_knn(
        X, int(n_neighbors), str(metric), int(m), int(l_max), int(R),
        float(alpha), int(beam_L), int(fanout), int(c_min), int(c_max),
        int(runs), int(n_jobs), int(random_state),
    )
    return indices.reshape(n, stride), distances.reshape(n, stride)


def query_knn(
    X,
    queries,
    n_neighbors: int = 15,
    *,
    metric: str = "euclidean",
    m: int = 12,
    l_max: int = 96,
    R: int = 64,
    alpha: float = 1.2,
    beam_L: int = 64,
    fanout: int = 2,
    c_min: int = 256,
    c_max: int = 2048,
    runs: int = 1,
    n_jobs: int = -1,
    random_state: int = 0,
):
    """Build a PiPNN index over base ``X`` and query an external ``queries`` set.

    Returns ``(indices, distances)`` of shape ``(n_queries, n_neighbors)`` (no
    self edge). This is the ANN-benchmark path (held-out queries). ``runs`` is
    the build-side recall knob; ``beam_L`` is the search-side one.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    Q = np.ascontiguousarray(queries, dtype=np.float32)
    n_q = Q.shape[0]
    n_jobs = 0 if n_jobs in (None, -1) else int(n_jobs)
    indices, distances, stride = _pipnn.build_and_query(
        X, Q, int(n_neighbors), str(metric), int(m), int(l_max), int(R),
        float(alpha), int(beam_L), int(fanout), int(c_min), int(c_max),
        int(runs), int(n_jobs), int(random_state),
    )
    return indices.reshape(n_q, stride), distances.reshape(n_q, stride)
