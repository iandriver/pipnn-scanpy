"""Standalone PiPNN index wrapper (without the sklearn/scanpy machinery).

Convenience for direct k-NN graph construction outside an AnnData workflow.
"""

from __future__ import annotations

import numpy as np

from . import _pipnn

__all__ = ["self_knn_graph"]


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
    n_jobs: int = -1,
    random_state: int = 0,
):
    """Build the index over ``X`` and return ``(indices, distances)`` arrays.

    Both are shape ``(n, n_neighbors + 1)``; column 0 is the self edge.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    n_jobs = 0 if n_jobs in (None, -1) else int(n_jobs)
    indices, distances, stride = _pipnn.build_and_self_knn(
        X, int(n_neighbors), str(metric), int(m), int(l_max), int(R),
        float(alpha), int(beam_L), int(fanout), int(c_min), int(c_max),
        int(n_jobs), int(random_state),
    )
    return indices.reshape(n, stride), distances.reshape(n, stride)
