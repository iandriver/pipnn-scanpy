"""Contract + recall checks across every available NN backend.

PiPNN always runs; glass runs only where pyglass is importable (e.g. the Docker
x86_64 image), so this same test validates GlassTransformer there.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import bench_lib as bl  # noqa: E402


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(6, 32)) * 6
    X = np.vstack([c + rng.normal(size=(400, 32)) for c in centers]).astype(np.float32)
    return X


# One (name, make_transformer) per available backend.
BACKENDS = list(bl.available_backends(15).items())
IDS = [name for name, _ in BACKENDS]


@pytest.mark.parametrize("name,make", BACKENDS, ids=IDS)
def test_csr_contract(name, make, data):
    k = 15
    g = make().fit_transform(data)
    n = data.shape[0]
    assert isinstance(g, csr_matrix)
    assert g.shape == (n, n)
    counts = np.diff(g.indptr)
    assert np.all(counts == k + 1), f"{name}: expected k+1 entries/row"
    for i in range(0, n, 53):
        s = g.indptr[i]
        assert g.indices[s] == i, f"{name}: self edge not first in row {i}"
        # self edge must be the explicit, near-zero first entry (sklearn's exact
        # backend yields ~5e-7 from float32 sqrt rather than a hard 0.0).
        assert g.data[s] <= 1e-4, f"{name}: self edge not ~zero ({g.data[s]})"
        row = g.data[g.indptr[i]:g.indptr[i + 1]]
        assert np.all(np.diff(row) >= -1e-6), f"{name}: distances not sorted"


@pytest.mark.parametrize("name,make", BACKENDS, ids=IDS)
def test_recall_vs_exact(name, make, data):
    k = 15
    g = make().fit_transform(data)
    approx = bl.knn_from_obsp_like(g, k) if hasattr(bl, "knn_from_obsp_like") else None
    # derive neighbors directly from the returned CSR
    n = data.shape[0]
    approx = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        s, e = g.indptr[i], g.indptr[i + 1]
        cols, vals = g.indices[s:e], g.data[s:e]
        order = np.argsort(vals)
        approx[i] = [c for c in cols[order] if c != i][:k]

    _, exact = NearestNeighbors(n_neighbors=k + 1).fit(data).kneighbors(data)
    exact = exact[:, 1:]
    recall = np.mean([len(set(approx[i]) & set(exact[i])) / k for i in range(n)])
    assert recall >= 0.90, f"{name}: recall {recall:.3f} too low"
