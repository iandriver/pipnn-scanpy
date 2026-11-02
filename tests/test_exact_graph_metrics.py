"""Check experiment metrics against small graphs with known answers."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'exact_graph', Path(__file__).parents[1] / 'bench' / 'bench_exact_graph.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_support_and_jaccard_weights():
    ids = np.array([[0,1,2],[1,0,2],[2,0,3],[3,0,2]])
    x = np.zeros((4,1))
    union = bench.weights(x, ids, 'union')
    mutual = bench.weights(x, ids, 'mutual')
    snn = bench.weights(x, ids, 'snn')
    assert union.nnz == 10
    assert mutual.nnz == 6
    assert mutual[0,3] == 0 and union[0,3] == 1
    assert snn[0,1] == pytest.approx(1/3)
    assert snn[0,2] == 0
    assert (snn != snn.T).nnz == 0
    assert not snn.diagonal().any()


def test_group_fragmentation_and_graph_difference():
    ids = np.array([[0,1,2],[1,0,2],[2,0,3],[3,0,2]])
    g = bench.weights(np.zeros((4,1)), ids, 'mutual')
    labels = np.array([0,0,1,1])
    summary = bench.graph_summary(g, labels, ['a','b'])
    assert summary['components'] == 1
    for group in summary['by_type'].values():
        assert group['internal_components'] == 1
        assert group['internal_isolated'] == 0
        assert group['internal_largest_fraction'] == 1
        assert group['cross_type_weight_fraction'] == pytest.approx(1/3)
    modified = g.copy(); modified[0,2] = 0; modified[2,0] = 0
    modified.eliminate_zeros()
    d = bench.graph_difference(modified, g, labels, ['a','b'])
    assert d['edge_jaccard'] == pytest.approx(2/3)
    assert d['relative_weight_l1'] == pytest.approx(1/3)
    assert bench.graph_summary(modified,labels,['a','b'])['components'] == 2


def test_self_normalization_does_not_drop_nonself_neighbor():
    x = np.zeros((20,1))
    ids = np.array([np.roll(np.arange(20),-i)[:16] for i in range(20)])
    # Put the self edge last to catch implicit first-column removal.
    ids = np.roll(ids, -1, axis=1)
    ds = np.ones_like(ids,dtype=float)
    for i,row in enumerate(ids): ds[i,row==i] = 0
    normalized,distances = bench.normalize(x,ids,ds)
    np.testing.assert_array_equal(normalized[:,0],np.arange(20))
    assert not distances[:,0].any()
    for before,after in zip(ids,normalized): assert set(before)==set(after)
    ids[0,1]=ids[0,0]
    with pytest.raises(ValueError): bench.normalize(x,ids,ds)


def test_common_weighting_matches_scanpy_transformer_hook():
    sc = pytest.importorskip('scanpy')
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    x = np.random.default_rng(19).normal(size=(40,4)).astype(np.float32)
    ds,ids = NearestNeighbors(n_neighbors=16,algorithm='brute').fit(x).kneighbors(x)
    # Match the experiment's common float64 distance evaluation.
    ds = np.linalg.norm(x[:,None].astype(np.float64)-x[ids].astype(np.float64),axis=2).astype(np.float32)
    g = csr_matrix((ds.ravel(),ids.ravel(),np.arange(0,40*16+1,16)),shape=(40,40))

    class FrozenGraph:
        def get_params(self): return {'n_neighbors':16}
        def fit_transform(self,x): return g

    a = sc.AnnData(x)
    sc.pp.neighbors(a,use_rep='X',transformer=FrozenGraph())
    expected = bench.weights(x,ids,'umap')
    assert abs(a.obsp['connectivities']-expected).max() < 1e-6


def test_scanpy_jaccard_averages_one_way_edges():
    pytest.importorskip('scanpy')
    ids = np.array([[0,1,2],[1,0,2],[2,0,3],[3,0,2]])
    x = np.zeros((4,1))
    built_in = bench.weights(x,ids,'scanpy_jaccard')
    union_snn = bench.weights(x,ids,'snn')
    assert built_in[0,1] == pytest.approx(1/3)  # reciprocal
    assert built_in[0,3] == pytest.approx(1/6)  # one direction only
    assert union_snn[0,3] == pytest.approx(1/3)
    assert (built_in != built_in.T).nnz == 0
