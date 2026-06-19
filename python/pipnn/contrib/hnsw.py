"""scanpy-compatible k-NN transformer backed by the native Rust HNSW backend.

A compact HNSW (the algorithm pyglass implements) built into `pipnn._pipnn`, so it
runs on any platform the wheel does — including arm64 macOS, where pyglass can't.
Useful as a graph-ANN comparison baseline alongside PiPNN and pynndescent:

    import scanpy as sc
    from pipnn.contrib import HnswTransformer
    sc.pp.neighbors(adata, n_neighbors=15, transformer=HnswTransformer())

Returns the same CSR layout PiPNN does (shape (n, n), k+1 entries/row, self edge
first as an explicit zero, sorted by distance).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from .. import _pipnn

__all__ = ["HnswTransformer"]


class HnswTransformer(TransformerMixin, BaseEstimator):
    """k-NN transformer using the native Rust HNSW index.

    Parameters
    ----------
    n_neighbors : int
        Neighbors per point (excluding self).
    metric : {"euclidean", "cosine"}
    m : int
        Max out-degree of the graph (layer 0 uses ``2*m``).
    ef_construction : int
        Candidate-list width during build (higher = better recall, slower).
    ef_search : int
        Candidate-list width during query (clamped to ``>= n_neighbors + 1``).
    quantize : {"none", "sq8"}
        ``"sq8"`` builds/searches on per-dimension 8-bit codes (4× smaller
        vectors, pyglass-style) — faster/leaner, slightly lower recall. Emitted
        distances are always exact.
    n_jobs : int
        Thread count for the query (``-1``/``0`` = all cores).
    random_state : int
        Seed for the level-assignment RNG.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        *,
        metric: str = "euclidean",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        quantize: str = "none",
        n_jobs: int = -1,
        random_state: int = 0,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.quantize = quantize
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float32, accept_sparse=False, order="C")
        n = X.shape[0]
        if self.n_neighbors >= n:
            raise ValueError(f"n_neighbors={self.n_neighbors} must be < n_samples={n}")
        n_jobs = 0 if self.n_jobs in (None, -1) else int(self.n_jobs)
        idx, dist, stride = _pipnn.hnsw_self_knn(
            X, int(self.n_neighbors), str(self.metric),
            int(self.m), int(self.ef_construction), int(self.ef_search),
            str(self.quantize), int(n_jobs), int(self.random_state),
        )
        self._indices = idx
        self._distances = dist
        self._stride = stride
        self.n_samples_fit_ = n
        self._n_features_out = n
        return self

    def transform(self, X):
        n = self.n_samples_fit_
        stride = self._stride
        indptr = np.arange(0, n * stride + 1, stride, dtype=np.int32)
        return csr_matrix(
            (
                self._distances.astype(np.float64, copy=False),
                self._indices.astype(np.int32, copy=False),
                indptr,
            ),
            shape=(n, n),
        )

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"requires_fit": True}
