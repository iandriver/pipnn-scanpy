"""Build-time + recall benchmark: PiPNN vs pynndescent vs sklearn-exact.

Run directly (not under pytest) for timing:

    .venv/bin/python tests/bench_vs_pynndescent.py [n] [d] [k]
"""

import sys
import time

import numpy as np
import resource

from sklearn.neighbors import NearestNeighbors
from pipnn import self_knn_graph


def make_data(n, d, n_clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d)) * 6
    sizes = [n // n_clusters] * n_clusters
    sizes[-1] += n - sum(sizes)
    X = np.vstack([c + rng.normal(size=(s, d)) for c, s in zip(centers, sizes)])
    return X.astype(np.float32)


def recall(approx_idx, exact_idx):
    n, kp1 = exact_idx.shape
    k = kp1 - 1
    hits = 0
    for i in range(n):
        hits += len(set(approx_idx[i, 1:].tolist()) & set(exact_idx[i, 1:].tolist()))
    return hits / (n * k)


def peak_rss_mb():
    # ru_maxrss is bytes on macOS, kB on Linux.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    print(f"n={n} d={d} k={k}")

    X = make_data(n, d)
    print(f"data: {X.nbytes/1e6:.1f} MB")

    # PiPNN
    t = time.time()
    idx, _ = self_knn_graph(X, n_neighbors=k, random_state=0)
    t_pip = time.time() - t
    print(f"[pipnn]  build+query: {t_pip:6.2f}s   peakRSS={peak_rss_mb():.0f}MB")

    # pynndescent (via its transformer)
    try:
        from pynndescent import PyNNDescentTransformer

        t = time.time()
        g = PyNNDescentTransformer(n_neighbors=k, metric="euclidean").fit_transform(X)
        t_pyn = time.time() - t
        print(f"[pynnd]  build+query: {t_pyn:6.2f}s   speedup={t_pyn/t_pip:.2f}x")
    except Exception as e:  # pragma: no cover
        print(f"[pynnd]  skipped: {e}")

    # Exact ground truth (subsample if huge)
    m = min(n, 20_000)
    sub = np.random.default_rng(1).choice(n, m, replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, exact = nn.kneighbors(X[sub])
    rec = recall(idx[sub], exact)
    print(f"[pipnn]  recall@{k} (on {m} sampled): {rec:.4f}")


if __name__ == "__main__":
    main()
