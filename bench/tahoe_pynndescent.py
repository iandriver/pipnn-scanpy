"""One pynndescent self-kNN build on a Tahoe PCA slice (run in the isolated
pynndescent venv; see the driver in the README/commit). Prints a JSON line
{n, s, recall, peak_gb} so the caller can merge it into tahoe_bench.json.

pynndescent is scanpy's DEFAULT NN backend but is memory-hungry at scale — this
runs one size per process so an OOM kills only that process, letting the driver
find where it tops out. Needs a WRITABLE array (its numba kernels reject the
readonly mmap views the other backends tolerate).

    /tmp/pynn_venv/bin/python bench/tahoe_pynndescent.py --n 1000000
"""

import argparse
import json
import os
import resource
import time

import numpy as np

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "tahoe_cache")
K = 15


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--source", type=int, default=5_000_000)
    args = ap.parse_args()

    from pynndescent import PyNNDescentTransformer

    full = np.load(os.path.join(CACHE, f"X_pca_{args.source}.npy"), mmap_mode="r")
    X = np.array(full[:args.n], dtype=np.float32)  # writable copy (numba needs it)

    # exact GT for a 4000-cell subsample, via FAISS flat
    import faiss
    sub = np.random.default_rng(1).choice(args.n, min(args.n, 4000), replace=False)
    fi = faiss.IndexFlatL2(X.shape[1]); fi.add(X)
    _, I = fi.search(X[sub], K + 1)
    ex = np.array([I[j][I[j] != i][:K] for j, i in enumerate(sub)])
    del fi

    # warm the numba JIT (excluded from the timed build), then time fit_transform
    PyNNDescentTransformer(n_neighbors=K, metric="euclidean").fit_transform(X[:2000])
    t = time.perf_counter()
    tr = PyNNDescentTransformer(n_neighbors=K, metric="euclidean")
    g = tr.fit_transform(X).tocsr()  # CSR (n,n), k+1/row (self included)
    dt = time.perf_counter() - t

    # recall: extract neighbors for the 4000 subsample rows only (from the CSR)
    def nbrs(i):
        s, e = g.indptr[i], g.indptr[i + 1]
        cols, vals = g.indices[s:e], g.data[s:e]
        order = np.argsort(vals)
        return [c for c in cols[order] if c != i][:K]

    rec = float(np.mean([len(set(nbrs(i)) & set(ex[j])) / K for j, i in enumerate(sub)]))

    print("RESULT " + json.dumps({"n": args.n, "s": dt, "recall": rec, "peak_gb": rss_gb()}),
          flush=True)


if __name__ == "__main__":
    main()
