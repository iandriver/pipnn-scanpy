"""scanpy-compatible k-NN transformer backed by FAISS (facebookresearch/faiss).

FAISS is the industry-standard similarity-search library; `faiss-cpu` ships native
macOS-arm64 wheels, so this runs as-is (no Rust). Useful as a battle-tested
cross-check alongside PiPNN, our native HNSW, and pynndescent.

    import scanpy as sc
    from pipnn.contrib import FaissTransformer
    sc.pp.neighbors(adata, n_neighbors=15,
                    transformer=FaissTransformer(index_type="hnsw"))

`index_type`: ``"hnsw"`` (IndexHNSWFlat), ``"ivfpq"`` (IndexIVFPQ — clustered +
product-quantized), or ``"flat"`` (IndexFlatL2 — exact, BLAS). Returns the same CSR
layout as the other backends (shape ``(n, n)``, ``k+1``/row, self edge first as an
explicit zero, sorted); emitted distances are always recomputed exactly.

Note: `import faiss` here resolves to the top-level FAISS package (Python 3
absolute imports), not this module.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

__all__ = ["FaissTransformer"]

_HINT = "FAISS not importable. Install with `pip install faiss-cpu`."


def _largest_divisor_leq(d: int, cap: int) -> int:
    """Largest divisor of d that is <= cap (PQ needs pq_m | d), >= 1."""
    for m in range(min(cap, d), 0, -1):
        if d % m == 0:
            return m
    return 1


class FaissTransformer(TransformerMixin, BaseEstimator):
    """k-NN transformer using a FAISS index.

    Parameters
    ----------
    n_neighbors : int
    metric : {"euclidean", "cosine"}
    index_type : {"hnsw", "ivfpq", "flat"}
    m, ef_construction, ef_search : int
        HNSW params.
    nlist, nprobe, pq_m : int or None
        IVF-PQ params (defaults derived from n and d).
    n_jobs : int
        FAISS OpenMP threads (``-1``/``0`` = all cores).
    random_state : int
        Accepted for API parity.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        *,
        metric: str = "euclidean",
        index_type: str = "hnsw",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        nlist: int | None = None,
        nprobe: int = 16,
        pq_m: int | None = None,
        n_jobs: int = -1,
        random_state: int = 0,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.n_jobs = n_jobs
        self.random_state = random_state

    @staticmethod
    def _import_faiss():
        try:
            import faiss  # top-level package, not this module
        except Exception as e:  # pragma: no cover
            raise ImportError(_HINT) from e
        return faiss

    def _prep(self, X):
        X = check_array(X, dtype=np.float32, accept_sparse=False, order="C")
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = (X / norms).astype(np.float32, copy=False)
        return np.ascontiguousarray(X, dtype=np.float32)

    def fit(self, X, y=None):
        faiss = self._import_faiss()
        Xp = self._prep(X)
        n, d = Xp.shape
        if self.n_neighbors >= n:
            raise ValueError(f"n_neighbors={self.n_neighbors} must be < n_samples={n}")
        if self.n_jobs not in (None, -1, 0):
            faiss.omp_set_num_threads(int(self.n_jobs))

        it = self.index_type
        if it == "hnsw":
            index = faiss.IndexHNSWFlat(d, int(self.m))
            index.hnsw.efConstruction = int(self.ef_construction)
            index.add(Xp)
            index.hnsw.efSearch = max(int(self.ef_search), self.n_neighbors + 1)
        elif it == "flat":
            index = faiss.IndexFlatL2(d)
            index.add(Xp)
        elif it == "ivfpq":
            nlist = self.nlist or int(np.clip(round(4 * np.sqrt(n)), 16, 4096))
            nlist = min(int(nlist), n)
            pq_m = int(self.pq_m or _largest_divisor_leq(d, 32))
            index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, pq_m, 8)
            index.train(Xp)
            index.add(Xp)
            index.nprobe = int(self.nprobe)
        else:
            raise ValueError(f"unknown index_type: {it!r}")

        self._index = index
        self._X = Xp
        self.n_samples_fit_ = n
        self._n_features_out = n
        return self

    def transform(self, X):
        n = self.n_samples_fit_
        k = self.n_neighbors
        stride = k + 1
        Xp = self._X

        _, ids = self._index.search(Xp, stride)
        ids = np.asarray(ids, dtype=np.int64)
        # FAISS pads missing results (rare, e.g. IVF) with -1; replace each with
        # its own row index (treated as the self edge below).
        if (ids < 0).any():
            ids[ids < 0] = np.where(ids < 0)[0]

        out_idx = np.empty((n, stride), dtype=np.int32)
        out_dist = np.empty((n, stride), dtype=np.float64)
        CHUNK = 8192
        for s in range(0, n, CHUNK):
            e = min(s + CHUNK, n)
            block = ids[s:e]
            nb = Xp[block]
            diff = nb - Xp[s:e, None, :]
            d2 = np.einsum("bsd,bsd->bs", diff, diff)
            emit = 0.5 * d2 if self.metric == "cosine" else np.sqrt(np.maximum(d2, 0.0))
            for r in range(e - s):
                i = s + r
                row_ids = block[r].copy()
                row_d = emit[r].copy()
                pos = np.where(row_ids == i)[0]
                if pos.size:
                    j = pos[0]
                else:
                    j = int(np.argmax(row_d))
                    row_ids[j] = i
                    row_d[j] = 0.0
                row_ids[j], row_ids[0] = row_ids[0], row_ids[j]
                row_d[j], row_d[0] = row_d[0], row_d[j]
                order = np.argsort(row_d[1:], kind="stable") + 1
                out_idx[i, 0] = i
                out_dist[i, 0] = 0.0
                out_idx[i, 1:] = row_ids[order]
                out_dist[i, 1:] = row_d[order]

        indptr = np.arange(0, n * stride + 1, stride, dtype=np.int32)
        return csr_matrix((out_dist.reshape(-1), out_idx.reshape(-1), indptr), shape=(n, n))

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"requires_fit": True}
