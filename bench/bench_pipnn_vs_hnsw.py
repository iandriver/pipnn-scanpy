"""PiPNN vs the native Rust HNSW backend across dataset sizes (5k–800k).

Warm min-of-N timing of the kNN-graph build (the part that differs), recall@15 vs
exact. Both are graph-ANN built in the same crate on the same hardware, library
defaults. Synthetic 50-d data, constant cluster density.

    .venv/bin/python bench/bench_pipnn_vs_hnsw.py [maxN]

Output: bench/pipnn_vs_hnsw.json, bench/pipnn_vs_hnsw.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import resource
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
K = 15


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / 1e9) if sys.platform == "darwin" else (r / 1e6)


def make_data(n, d=50, seed=0):
    rng = np.random.default_rng(seed)
    nc = max(8, n // 5000)
    C = rng.normal(size=(nc, d)) * 6
    sz = [n // nc] * nc
    sz[-1] += n - sum(sz)
    X = np.vstack([c + rng.normal(size=(s, d)) for c, s in zip(C, sz)])
    return np.ascontiguousarray(X, dtype=np.float32)


def warm(fn, repeats):
    fn()  # discard warmup
    out = [None]

    def timed():
        t = time.perf_counter()
        out[0] = fn()
        return time.perf_counter() - t

    return min(timed() for _ in range(repeats)), out[0]


def recall_pairs(idx_rows, exact_sub, sub):
    return float(np.mean([
        len(set(idx_rows[i][1:]) & set(exact_sub[j])) / K for j, i in enumerate(sub)
    ]))


def recall_csr(g, exact_sub, sub):
    g = g.tocsr()
    out = {}
    for j, i in enumerate(sub):
        s, e = g.indptr[i], g.indptr[i + 1]
        order = np.argsort(g.data[s:e])
        nbrs = [c for c in g.indices[s:e][order] if c != i][:K]
        out[i] = nbrs
    return float(np.mean([len(set(out[i]) & set(exact_sub[j])) / K for j, i in enumerate(sub)]))


def main():
    maxN = int(sys.argv[1]) if len(sys.argv) > 1 else 800_000
    sizes = [s for s in [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 800_000]
             if s <= maxN]

    from pipnn import self_knn_graph
    from pipnn.contrib import HnswTransformer

    rows = []
    print(f"{'n':>8} | {'PiPNN s':>8} {'rec':>6} | {'HNSW s':>8} {'rec':>6} | {'peakGB':>6}")
    for n in sizes:
        X = make_data(n)
        sub = np.random.default_rng(1).choice(n, min(n, 4000), replace=False)
        _, ex = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X[sub])
        ex = ex[:, 1:]
        reps = 3 if n <= 100_000 else 2

        t_pip, (idx, _) = warm(lambda: self_knn_graph(X, n_neighbors=K, random_state=0), reps)
        r_pip = recall_pairs(idx, ex, sub)

        t_h, g = warm(lambda: HnswTransformer(n_neighbors=K).fit_transform(X), reps)
        r_h = recall_csr(g, ex, sub)

        rows.append(dict(n=n, pipnn_s=t_pip, pipnn_rec=r_pip, hnsw_s=t_h, hnsw_rec=r_h,
                         peak_gb=rss_gb()))
        print(f"{n:8d} | {t_pip:8.3f} {r_pip:6.3f} | {t_h:8.3f} {r_h:6.3f} | {rss_gb():6.2f}",
              flush=True)
        with open(HERE / "pipnn_vs_hnsw.json", "w") as f:
            json.dump(rows, f, indent=2)

    _plot(rows)
    print("\nwrote bench/pipnn_vs_hnsw.json and bench/pipnn_vs_hnsw.png")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    ax[0].plot(n, [r["pipnn_s"] for r in rows], "o-", color="#2a9d8f", label="PiPNN")
    ax[0].plot(n, [r["hnsw_s"] for r in rows], "^-", color="#8856a7", label="HNSW (native)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("cells (n)"); ax[0].set_ylabel("warm build time (s)")
    ax[0].set_title("kNN graph build time vs n (warm, log-log)")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()

    ax[1].plot(n, [r["pipnn_rec"] for r in rows], "o-", color="#2a9d8f", label="PiPNN")
    ax[1].plot(n, [r["hnsw_rec"] for r in rows], "^-", color="#8856a7", label="HNSW (native)")
    ax[1].set_xscale("log"); ax[1].set_ylim(0.80, 1.005)
    ax[1].set_xlabel("cells (n)"); ax[1].set_ylabel("recall@15 vs exact")
    ax[1].set_title("Recall vs n (library defaults)")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend()

    fig.suptitle("PiPNN vs native HNSW — kNN graph build (50-d, warm steady-state, all cores)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(HERE / "pipnn_vs_hnsw.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
