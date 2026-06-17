"""End-to-end: PiPNNTransformer plugged into sc.pp.neighbors drives UMAP/Leiden."""

import numpy as np
import pytest

sc = pytest.importorskip("scanpy")
import anndata as ad  # noqa: E402

from pipnn import PiPNNTransformer  # noqa: E402


@pytest.fixture
def adata():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 50)) * 8
    X = np.vstack([c + rng.normal(size=(150, 50)) for c in centers]).astype(np.float32)
    a = ad.AnnData(X)
    a.obsm["X_pca"] = X  # treat the matrix as the PCA rep
    return a


def test_neighbors_populates_obsp(adata):
    sc.pp.neighbors(
        adata, n_neighbors=15, use_rep="X_pca",
        transformer=PiPNNTransformer(n_neighbors=15),
    )
    assert "distances" in adata.obsp
    assert "connectivities" in adata.obsp
    assert "neighbors" in adata.uns
    n = adata.n_obs
    assert adata.obsp["distances"].shape == (n, n)
    assert adata.obsp["connectivities"].shape == (n, n)
    # connectivities should be a non-trivial symmetric-ish graph
    assert adata.obsp["connectivities"].nnz > 0


def test_umap_and_leiden_run(adata):
    sc.pp.neighbors(
        adata, n_neighbors=15, use_rep="X_pca",
        transformer=PiPNNTransformer(n_neighbors=15),
    )
    sc.tl.umap(adata)
    assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)

    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
    n_clusters = adata.obs["leiden"].nunique()
    # 4 blobs → expect a small number of clusters, definitely > 1
    assert 1 < n_clusters < 30
