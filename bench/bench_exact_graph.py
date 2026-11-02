"""Exact versus approximate k-NN and downstream graph checks on public PBMC3k.

Run prepare, search, then graphs. Labels are used only for evaluation.
Workers run serially in fresh processes. See docs/exact_graph_public.md.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(x):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def prepare(a):
    import numpy as np
    root = a.data
    root.mkdir(parents=True, exist_ok=True)
    # Scanpy downloads the public example if it is not already cached.
    import scanpy as sc
    sc.settings.datasetdir = a.public_cache
    public = sc.datasets.pbmc3k_processed()
    x = np.asarray(public.obsm['X_pca'], dtype=np.float32)
    names, y = np.unique(public.obs['louvain'].astype(str), return_inverse=True)
    tasks = [('pbmc3k', x, y, names.tolist(), 'public scanpy.datasets.pbmc3k_processed')]
    manifest = {}
    for name, x, y, categories, source in tasks:
        path = root / name; path.mkdir(exist_ok=True)
        np.save(path / 'X.npy', np.ascontiguousarray(x, dtype=np.float32))
        np.save(path / 'labels.npy', y)
        m = dict(n=len(x), d=x.shape[1], k=15, categories=categories,
                 counts=np.bincount(y, minlength=len(categories)).tolist(),
                 input_sha256=digest(x), labels_sha256=digest(y), source=source,
                 subset_seed=20261102, rare_threshold=0.01)
        write(path / 'manifest.json', m); manifest[name] = m
    write(a.output / 'datasets.json', manifest)


def build(x, method, threads, seed=0):
    import numpy as np
    k = 15
    if method in ['faiss', 'sklearn']:
        if method == 'faiss':
            import faiss
            faiss.omp_set_num_threads(threads)
            index = faiss.IndexFlatL2(x.shape[1]); index.add(x)
            ds, ids = index.search(x, k + 1)
            ds = np.sqrt(np.maximum(ds, 0))
        else:
            from sklearn.neighbors import NearestNeighbors
            index = NearestNeighbors(n_neighbors=k+1, algorithm='brute',
                                     metric='euclidean', n_jobs=threads).fit(x)
            ds, ids = index.kneighbors(x)
    elif method in ['direct', 'index']:
        from pipnn import PiPNNTransformer
        t = PiPNNTransformer(n_neighbors=k, algorithm='direct' if method == 'direct' else 'index',
                             kernel='dense', c_max=512 if method == 'direct' else 2048,
                             runs=8 if method == 'direct' else 1,
                             beam_L=64, n_jobs=threads, random_state=seed)
        g = t.fit_transform(x)
        ids = g.indices.reshape(len(x), k+1); ds = g.data.reshape(len(x), k+1)
    elif method == 'scanpy_pynn':
        from pynndescent import PyNNDescentTransformer
        # Match Scanpy's default transformer settings for self + 15 neighbors.
        t = PyNNDescentTransformer(n_neighbors=k+1, metric='euclidean',
            n_trees=min(64,5+round(len(x)**.5/20)), n_iters=max(5,round(np.log2(len(x)))),
            random_state=seed,n_jobs=threads,low_memory=True)
        g = t.fit_transform(x)
        width = int(np.diff(g.indptr)[0])
        ids = g.indices.reshape(len(x),width);ds=g.data.reshape(len(x),width)
        order = np.argsort(ds,axis=1,kind='stable')
        ids=np.take_along_axis(ids,order,axis=1);ds=np.take_along_axis(ds,order,axis=1)
    else:
        from pynndescent import NNDescent
        index = NNDescent(x, n_neighbors=k+1, metric='euclidean', random_state=seed,
                          n_jobs=threads, low_memory=True)
        ids, ds = index.neighbor_graph
    return np.asarray(ids), np.asarray(ds)


def normalize(x, ids, ds):
    import numpy as np
    # Always return self + 15 other unique IDs. Preserve native distances here.
    n = len(x); out = np.empty((n,16), dtype=np.int32); dist = np.empty((n,16), dtype=np.float32)
    out[:,0] = np.arange(n); dist[:,0] = 0
    for i, row in enumerate(ids):
        keep = np.flatnonzero((row != i) & (row >= 0))[:15]
        if len(keep) != 15 or len(np.unique(row[keep])) != 15:
            raise ValueError(f'underfilled/duplicate row {i}')
        out[i,1:] = row[keep]; dist[i,1:] = ds[i,keep]
    assert out.min() >= 0 and out.max() < n and np.isfinite(dist).all()
    assert np.all(dist >= 0)
    return out, dist


def search_worker(a):
    import numpy as np
    from threadpoolctl import threadpool_limits
    m = json.loads((a.data / a.dataset / 'manifest.json').read_text())
    x = np.load(a.data / a.dataset / 'X.npy')
    assert digest(x) == m['input_sha256']
    with threadpool_limits(a.threads):
        # Compile/JIT and method initialization are outside the warm timer.
        warm = build(x[:min(5000,len(x))], a.method, a.threads)
        del warm
        start = time.perf_counter()
        ids, ds = build(x, a.method, a.threads)
        seconds = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = (peak if sys.platform == 'darwin' else peak * 1024) / 1e6
    ids, ds = normalize(x, ids, ds)
    assert np.all(np.diff(ds, axis=1) >= -1e-4)
    dest = a.output / a.dataset; dest.mkdir(exist_ok=True)
    stem = dest / f'{a.method}_{a.repeat}'
    np.save(stem.with_suffix('.npy'), ids)
    result = dict(method=a.method, seconds=seconds, peak_process_mb=peak_mb,
                  graph_ids_sha256=digest(ids), input_sha256=m['input_sha256'])
    write(stem.with_suffix('.json'), result)
    print(a.dataset, a.method, a.repeat, round(seconds,3), round(peak_mb,1), flush=True)


def orchestrate(a, phase):
    import numpy as np
    datasets = json.loads((a.output / 'datasets.json').read_text())
    if a.dataset:
        datasets = {a.dataset: datasets[a.dataset]}
    rng = np.random.default_rng(20261102)
    for name in datasets:
        if phase == 'search':
            for r in range(a.repeats):
                for method in rng.permutation(['faiss','sklearn','direct','index','scanpy_pynn','pynndescent']):
                    dest = a.output / name / f'{method}_{r}.json'
                    if dest.exists(): continue
                    cmd = [sys.executable, __file__, 'search-worker', '--dataset', name,
                           '--method', method, '--repeat', str(r)]
                    cmd += ['--data',str(a.data),'--output',str(a.output),'--threads',str(a.threads)]
                    subprocess.run(cmd, check=True)
        else:
            cmd = [sys.executable,__file__,'graph-worker','--dataset',name,
                   '--data',str(a.data),'--output',str(a.output),'--threads',str(a.threads)]
            subprocess.run(cmd,check=True)


def graph_summary(g, labels, categories):
    import numpy as np
    from scipy.sparse.csgraph import connected_components
    g = g.tocsr(); n = g.shape[0]
    components, c = connected_components(g, directed=False)
    by_type = {}
    total_weight = np.asarray(g.sum(axis=1)).ravel()
    for i, name in enumerate(categories):
        mask = labels == i; rows = np.flatnonzero(mask)
        if not len(rows): continue
        within = g[rows][:,rows]
        degree = np.diff(within.indptr)
        nc, cc = connected_components(within, directed=False)
        internal = float(within.sum()); total = float(total_weight[mask].sum())
        by_type[name] = dict(cells=len(rows), internal_components=int(nc),
                            internal_isolated=int((degree == 0).sum()),
                            internal_largest_fraction=float(np.bincount(cc).max()/len(rows)),
                            cross_type_weight_fraction=(1-internal/total) if total else None)
    return dict(edges_undirected=int(g.nnz//2), components=int(components),
                largest_component_fraction=float(np.bincount(c).max()/n),
                isolated_cells=int((np.diff(g.indptr)==0).sum()), by_type=by_type)


def graph_difference(g, ref, labels, categories):
    import numpy as np
    delta = abs(g-ref).tocsr(); denominator = float(ref.sum())
    support = g.copy(); support.data[:] = 1
    base = ref.copy(); base.data[:] = 1
    intersection = support.multiply(base).nnz
    union = support.nnz + base.nnz - intersection
    by_type = {}
    for i,name in enumerate(categories):
        rows=np.flatnonzero(labels==i)
        if not len(rows): continue
        denom=float(ref[rows].sum())
        by_type[name] = float(delta[rows].sum()/denom) if denom else None
    return dict(edge_jaccard=float(intersection/union) if union else 1.0,
                relative_weight_l1=float(delta.sum()/denominator) if denominator else 0.0,
                by_type_relative_weight_l1=by_type)


def cluster(g, labels, categories, reference=None):
    import numpy as np
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    a = sc.AnnData(shape=(len(labels),0)); results=[]; assignments=[]
    for seed in [0,1,2]:
        sc.tl.leiden(a, adjacency=g, flavor='igraph', directed=False,
                     n_iterations=2, resolution=1, random_state=seed, key_added='cluster')
        y = a.obs['cluster'].cat.codes.to_numpy().copy(); assignments.append(y)
        by_type={}
        for i,name in enumerate(categories):
            take=labels==i
            if not take.any(): continue
            ncl = np.bincount(y[take],minlength=y.max()+1)
            best=int(ncl.argmax()); size=int((y==best).sum())
            by_type[name]=dict(best_cluster_completeness=float(ncl[best]/take.sum()),
                               best_cluster_purity=float(ncl[best]/size),
                               occupied_clusters=int((ncl>0).sum()))
        results.append(dict(seed=seed,clusters=int(len(set(y))),
                            ari_to_annotations=float(adjusted_rand_score(labels,y)),
                            nmi_to_annotations=float(normalized_mutual_info_score(labels,y)),
                            ari_to_exact_same_seed=(float(adjusted_rand_score(reference[seed],y))
                                                    if reference is not None else 1.0),
                            by_type=by_type))
    stability=[float(adjusted_rand_score(assignments[i],assignments[j]))
               for i,j in [(0,1),(0,2),(1,2)]]
    return dict(seeds=results,within_graph_seed_ari=stability),assignments


def weights(x, ids, kind):
    import numpy as np
    from scipy.sparse import csr_matrix
    n=len(x); k=ids.shape[1]-1
    if kind=='umap':
        from scanpy.neighbors._connectivity import umap
        # Recompute every selected distance in float64, then cast to float32:
        # differences now reflect candidate IDs, not backend distance arithmetic.
        distances=np.empty(ids.shape,dtype=np.float32)
        for start in range(0,n,10000):
            q=ids[start:start+10000]
            distances[start:start+len(q)]=np.linalg.norm(
                x[start:start+len(q),None,:].astype(np.float64)-x[q].astype(np.float64),axis=2)
        # Ensure neighbor rows are ordered by these common distances.
        order=np.argsort(distances[:,1:],axis=1,kind='stable')+1
        ix=np.take_along_axis(ids[:,1:],order-1,axis=1)
        ds=np.take_along_axis(distances[:,1:],order-1,axis=1)
        ids=np.column_stack([np.arange(n),ix]); distances=np.column_stack([np.zeros(n),ds])
        return umap(ids,distances,n_obs=n,n_neighbors=k+1).tocsr()
    if kind=='scanpy_jaccard':
        from scanpy.neighbors._connectivity import jaccard
        return jaccard(ids,n_obs=n,n_neighbors=k+1).tocsr()
    rows=np.repeat(np.arange(n),k)
    b=csr_matrix((np.ones(n*k,dtype=np.float32),(rows,ids[:,1:].ravel())),shape=(n,n))
    if kind=='union': return b.maximum(b.T).tocsr()
    if kind=='mutual': return b.minimum(b.T).tocsr()
    # Jaccard shared-neighbor weights on union-kNN support only (no pruning).
    # This definition is explicit; it is not a reimplementation of Seurat defaults.
    shared=(b@b.T).tocsr()
    support=b.maximum(b.T)
    shared=shared.multiply(support).tocsr()
    shared.data=shared.data/(2*k-shared.data)
    shared.setdiag(0);shared.eliminate_zeros()
    return shared


def graph_worker(a):
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.sparse import save_npz
    data=a.data/a.dataset; dest=a.output/a.dataset
    m=json.loads((data/'manifest.json').read_text())
    x=np.load(data/'X.npy'); labels=np.load(data/'labels.npy'); categories=m['categories']
    refids=np.load(dest/'sklearn_0.npy')
    # Uniform plus type-stratified float64 audit against the full reference.
    rng=np.random.default_rng(20261102)
    qs=set(rng.choice(len(x),min(64,len(x)),replace=False).tolist())
    for label in np.unique(labels):
        rows=np.flatnonzero(labels==label);qs.update(rng.choice(rows,min(8,len(rows)),replace=False).tolist())
    qs=np.array(sorted(qs)); mismatches=[]
    for start in range(0,len(qs),16):
        q=qs[start:start+16];d=cdist(x[q].astype(np.float64),x.astype(np.float64),'sqeuclidean')
        d[np.arange(len(q)),q]=np.inf
        exact=np.argpartition(d,14,axis=1)[:,:15]
        for local,(row,truth) in enumerate(zip(q,exact)):
            if set(truth)!=set(refids[row,1:]):
                cutoff=float(np.max(d[local,truth]))
                excess=float(np.max(d[local,refids[row,1:]])-cutoff)
                mismatches.append(dict(row=int(row),different_ids=len(set(truth)-set(refids[row,1:])),
                                       cutoff_squared_distance=cutoff, selection_excess_squared=excess))
    write(dest/'float64_audit.json',dict(queries=len(qs),mismatches=mismatches))
    # Search quality for every cell, and exact-implementation disagreements.
    recall={}
    for method in ['faiss','sklearn','direct','index','scanpy_pynn','pynndescent']:
        pred=np.load(dest/f'{method}_0.npy');r=np.empty(len(x))
        for start in range(0,len(x),10000):
            p=pred[start:start+10000,1:]; t=refids[start:start+len(p),1:]
            r[start:start+len(p)]=(p[:,:,None]==t[:,None,:]).any(axis=2).sum(axis=1)/15
        recall[method]=dict(overall=float(r.mean()),
                            by_type={name:dict(cells=int((labels==i).sum()),recall=float(r[labels==i].mean()))
                                     for i,name in enumerate(categories) if (labels==i).any()})
    # Warm connectivity/Leiden initialization outside the downstream timers.
    warmx=x[:200]; wi,wd=normalize(warmx,*build(warmx,'faiss',a.threads))
    wg=weights(warmx,wi,'umap'); cluster(wg,labels[:200],categories)
    start=time.perf_counter();ref=weights(x,refids,'umap');weight_seconds=time.perf_counter()-start
    save_npz(dest/'graph_sklearn_umap.npz',ref)
    start=time.perf_counter();refcluster,refassign=cluster(ref,labels,categories)
    cluster_seconds=time.perf_counter()-start
    checks={'sklearn_umap':dict(weight_seconds=weight_seconds, three_seed_cluster_seconds=cluster_seconds,
                             graph=graph_summary(ref,labels,categories),clustering=refcluster)}
    for method in ['faiss','direct','index','scanpy_pynn','pynndescent']:
        start=time.perf_counter();g=weights(x,np.load(dest/f'{method}_0.npy'),'umap')
        weight_seconds=time.perf_counter()-start
        save_npz(dest/f'graph_{method}_umap.npz',g)
        start=time.perf_counter();cl,assign=cluster(g,labels,categories,refassign)
        cluster_seconds=time.perf_counter()-start
        checks[f'{method}_umap']=dict(weight_seconds=weight_seconds, three_seed_cluster_seconds=cluster_seconds,
                                      graph=graph_summary(g,labels,categories),
                                      difference_to_exact=graph_difference(g,ref,labels,categories),clustering=cl)
        print(a.dataset,method,'graphs checked',flush=True)
    # Hold exact neighbors fixed and change only weighting/support policy.
    for kind in ['union','mutual','snn','scanpy_jaccard']:
        start=time.perf_counter();g=weights(x,refids,kind);weight_seconds=time.perf_counter()-start
        save_npz(dest/f'graph_sklearn_{kind}.npz',g)
        start=time.perf_counter();cl,assign=cluster(g,labels,categories,refassign)
        cluster_seconds=time.perf_counter()-start
        checks[f'sklearn_{kind}']=dict(weight_seconds=weight_seconds, three_seed_cluster_seconds=cluster_seconds,
                                    graph=graph_summary(g,labels,categories),
                                    difference_to_exact=graph_difference(g,ref,labels,categories),clustering=cl)
        print(a.dataset,kind,'weighting checked',flush=True)
    write(dest/'downstream.json',dict(reference='sklearn',recall=recall,checks=checks))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','search','graphs','search-worker','graph-worker'])
    p.add_argument('--data',type=Path,default=Path('bench/data/public_knn'))
    p.add_argument('--output',type=Path,default=Path('bench/results/public_knn'))
    p.add_argument('--public-cache',type=Path,default=Path('bench/data/public_knn/public_cache'))
    p.add_argument('--dataset',default='');p.add_argument('--method',default='faiss',choices=['faiss','sklearn','direct','index','scanpy_pynn','pynndescent'])
    p.add_argument('--repeat',type=int,default=0);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if a.threads<1 or a.repeats<1: p.error('threads and repeats must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS',
                'NUMBA_NUM_THREADS','LOKY_MAX_CPU_COUNT']:
        os.environ[key]=str(a.threads)
    if a.phase=='prepare': prepare(a)
    elif a.phase in ['search','graphs']: orchestrate(a,a.phase)
    elif a.phase=='search-worker': search_worker(a)
    else: graph_worker(a)
