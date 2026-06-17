"""scanpy-compatible k-NN transformer backed by pyglass (zilliz 'glass').

Mirrors :class:`pipnn.PiPNNTransformer` so it is a drop-in alternative backend:

    import scanpy as sc
    from pipnn.contrib import GlassTransformer
    sc.pp.neighbors(adata, n_neighbors=15, transformer=GlassTransformer())

pyglass (https://github.com/zilliztech/pyglass) is a graph ANN library (HNSW/NSG
with SIMD + quantization). Its ``batch_search`` returns neighbor ids and
(quantizer-dependent) distances; to keep the graph identical in convention to the
other backends we recompute **exact** distances from the data for the returned
neighbors, then emit the same CSR layout scanpy expects (shape ``(n, n)``,
``k+1`` entries/row, self edge first as an explicit zero).

pyglass has no macOS/arm64 build; this backend activates wherever ``glass``
imports (manylinux x86_64). The import is lazy, so importing this module never
fails — only constructing/fitting without ``glass`` raises a clear error.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

__all__ = ["GlassTransformer"]

_INSTALL_HINT = (
    "pyglass ('glass') is not importable. It ships only as manylinux x86_64 "
    "(`pip install glassppy`, CPython 3.10) or a from-source build "
    "(https://github.com/zilliztech/pyglass); there is no macOS/arm64 wheel."
)


class GlassTransformer(TransformerMixin, BaseEstimator):
    """k-NN transformer using a pyglass HNSW/NSG graph index.

    Parameters
    ----------
    n_neighbors : int
        Neighbors per point (excluding self), matching scanpy's ``n_neighbors``.
    metric : {"euclidean", "cosine"}
        ``euclidean`` maps to glass ``"L2"``; ``cosine`` L2-normalizes rows then
        uses ``"L2"`` (monotone with angular distance).
    index_type : {"HNSW", "NSG"}
    R, L : int
        Graph degree / build beam (glass ``Index`` params).
    ef : int
        Search beam width (``set_ef``); clamped to ``>= n_neighbors + 1``.
    quant : str
        Build-time quantizer (glass ``Index(quant=...)``). ``"FP32"`` = exact.
    quantizer : str
        Search-time quantizer (glass ``Searcher(quantizer=...)``). ``"FP32"`` = exact.
    n_jobs : int
        Thread count (``glass.set_num_threads``); ``-1``/``None`` = all cores.
    random_state : int
        Accepted for API parity (glass build is not seeded here).
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        *,
        metric: str = "euclidean",
        index_type: str = "HNSW",
        R: int = 32,
        L: int = 50,
        ef: int = 64,
        quant: str = "FP32",
        quantizer: str = "FP32",
        n_jobs: int = -1,
        random_state: int = 0,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.index_type = index_type
        self.R = R
        self.L = L
        self.ef = ef
        self.quant = quant
        self.quantizer = quantizer
        self.n_jobs = n_jobs
        self.random_state = random_state

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _import_glass():
        # The PyPI wheel installs as `glassppy`; a from-source build installs as
        # `glass`. Accept either.
        for name in ("glassppy", "glass"):
            try:
                return __import__(name)
            except Exception:
                continue
        raise ImportError(_INSTALL_HINT)

    def _prep(self, X):
        X = check_array(X, dtype=np.float32, accept_sparse=False, order="C")
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = (X / norms).astype(np.float32, copy=False)
        return np.ascontiguousarray(X, dtype=np.float32)

    # -- sklearn estimator API -------------------------------------------------

    def fit(self, X, y=None):
        glass = self._import_glass()
        Xp = self._prep(X)
        n = Xp.shape[0]
        if self.n_neighbors >= n:
            raise ValueError(f"n_neighbors={self.n_neighbors} must be < n_samples={n}")

        if self.n_jobs not in (None, -1):
            glass.set_num_threads(int(self.n_jobs))

        index = glass.Index(
            index_type=self.index_type, metric="L2",
            quant=self.quant, R=int(self.R), L=int(self.L),
        )
        graph = index.build(Xp)
        searcher = glass.Searcher(
            graph=graph, data=Xp, metric="L2", quantizer=self.quantizer,
        )
        searcher.set_ef(max(int(self.ef), self.n_neighbors + 1))
        searcher.optimize()

        self._searcher = searcher
        self._X = Xp
        self.n_samples_fit_ = n
        self._n_features_out = n
        return self

    def transform(self, X):
        n = self.n_samples_fit_
        k = self.n_neighbors
        stride = k + 1
        Xp = self._X

        ids, _ = self._searcher.batch_search(Xp, stride)
        ids = np.asarray(ids, dtype=np.int64).reshape(n, -1)

        out_idx = np.empty((n, stride), dtype=np.int32)
        out_dist = np.empty((n, stride), dtype=np.float64)

        # Recompute exact distances for the returned neighbors (chunked to bound
        # memory), and force the self edge to slot 0.
        CHUNK = 8192
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            block = ids[s:e]                      # (b, stride)
            nb = Xp[block]                        # (b, stride, d)
            diff = nb - Xp[s:e, None, :]          # (b, stride, d)
            d2 = np.einsum("bsd,bsd->bs", diff, diff)  # squared L2
            if self.metric == "cosine":
                emit = 0.5 * d2                   # unit vectors: 1 - cos = d2/2
            else:
                emit = np.sqrt(np.maximum(d2, 0.0))
            for r in range(e - s):
                i = s + r
                row_ids = block[r].copy()
                row_d = emit[r].copy()
                # ensure self present and first
                pos = np.where(row_ids == i)[0]
                if pos.size:
                    j = pos[0]
                else:
                    # replace the farthest if glass omitted self
                    j = int(np.argmax(row_d))
                    row_ids[j] = i
                    row_d[j] = 0.0
                row_ids[j], row_ids[0] = row_ids[0], row_ids[j]
                row_d[j], row_d[0] = row_d[0], row_d[j]
                # sort neighbors [1:] by distance ascending, keep self first
                order = np.argsort(row_d[1:], kind="stable") + 1
                out_idx[i, 0] = i
                out_dist[i, 0] = 0.0
                out_idx[i, 1:] = row_ids[order]
                out_dist[i, 1:] = row_d[order]

        indptr = np.arange(0, n * stride + 1, stride, dtype=np.int32)
        graph = csr_matrix(
            (out_dist.reshape(-1), out_idx.reshape(-1), indptr), shape=(n, n)
        )
        return graph

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"requires_fit": True}
