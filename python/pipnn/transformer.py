"""scanpy-compatible k-NN transformer backed by the PiPNN Rust core.

Usage with scanpy (drop-in replacement for the default neighbor backend)::

    import scanpy as sc
    from pipnn import PiPNNTransformer

    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca",
                    transformer=PiPNNTransformer())

scanpy calls ``transformer.fit_transform(X)`` and then computes the fuzzy
connectivities itself, so this class only has to return the k-NN *distance*
graph in the exact CSR layout sklearn's ``KNeighborsTransformer`` produces:

* shape ``(n, n)``, ``mode='distance'``;
* ``n_neighbors + 1`` explicit entries per row, **including the self edge**
  (the point itself at distance 0.0, placed first);
* the self 0.0 is a *stored* explicit zero (we build the CSR from raw
  ``(data, indices, indptr)`` so scipy will not drop it);
* per-row entries ordered by ascending distance (self first).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from . import _pipnn

__all__ = ["PiPNNTransformer"]


class PiPNNTransformer(TransformerMixin, BaseEstimator):
    """k-NN transformer using the PiPNN graph index.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors per point (excluding the self edge), matching
        scanpy's ``n_neighbors``.
    metric : {"euclidean", "cosine"}
        Distance metric. ``"euclidean"`` (default) matches scanpy on PCA space.
    m, l_max, R, alpha, beam_L, fanout, c_min, c_max :
        PiPNN build/search hyperparameters (see the paper); defaults follow the
        paper's recommendations.
    n_jobs : int
        Thread count for the Rust build (``-1`` / ``0`` = all cores).
    random_state : int
        Seed for the RNG (ball carving + LSH hyperplanes). Builds are
        deterministic given a fixed seed.
    """

    def __init__(
        self,
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
        # sklearn contract: __init__ only stores args verbatim, no validation.
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.m = m
        self.l_max = l_max
        self.R = R
        self.alpha = alpha
        self.beam_L = beam_L
        self.fanout = fanout
        self.c_min = c_min
        self.c_max = c_max
        self.n_jobs = n_jobs
        self.random_state = random_state

    # -- sklearn estimator API -------------------------------------------------

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float32, accept_sparse=False, order="C")
        n = X.shape[0]
        if self.n_neighbors >= n:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must be < n_samples={n}"
            )

        n_jobs = 0 if self.n_jobs in (None, -1) else int(self.n_jobs)
        indices, distances, stride = _pipnn.build_and_self_knn(
            X,
            int(self.n_neighbors),
            str(self.metric),
            int(self.m),
            int(self.l_max),
            int(self.R),
            float(self.alpha),
            int(self.beam_L),
            int(self.fanout),
            int(self.c_min),
            int(self.c_max),
            int(n_jobs),
            int(self.random_state),
        )

        self._indices = indices
        self._distances = distances
        self._stride = stride
        self.n_samples_fit_ = n
        self._n_features_out = n
        return self

    def transform(self, X):
        n = self.n_samples_fit_
        stride = self._stride
        # Uniform stride per row → indptr is a plain arange.
        indptr = np.arange(0, n * stride + 1, stride, dtype=np.int32)
        graph = csr_matrix(
            (
                self._distances.astype(np.float64, copy=False),
                self._indices.astype(np.int32, copy=False),
                indptr,
            ),
            shape=(n, n),
        )
        # NB: do NOT sort_indices() — that would reorder columns and break the
        # "self edge first" invariant scanpy relies on. Distance order (self
        # first) is preserved as produced by the core.
        return graph

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    # sklearn tag: this transformer always requires fitting.
    def _more_tags(self):
        return {"requires_fit": True}
