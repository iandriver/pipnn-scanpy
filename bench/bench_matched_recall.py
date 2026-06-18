"""Matched-recall comparison: how long does pynndescent take to reach PiPNN's recall?

pynndescent at its defaults is fast but its recall@15 drops to ~0.82 at scale.
Its recall is tunable — chiefly by **building a wider graph** (`n_neighbors` larger
than the requested k, then truncating to the top-k), the direct analog of PiPNN's
internal over-search. This benchmark reports, per dataset size:

* PiPNN (default)                  — time, recall
* pynndescent (default)            — time, recall  (the fast/low-recall point)
* pynndescent (tuned to PiPNN's recall) — time, recall, and the slowdown vs PiPNN

It also writes a recall-vs-build-time Pareto plot at one size.

    .venv/bin/python bench/bench_matched_recall.py [maxN]

Output: bench/matched_recall.json, bench/matched_recall.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent

# pynndescent "high quality" knobs (less pruning, more descent); recall is then
# driven up by BUILD_K (build a wider graph, keep the top-15).
QUAL = dict(diversify_prob=0.0, pruning_degree_multiplier=3.0, n_iters=20, max_candidates=60)
K = 15


def make_data(n, d=50, seed=0):
    rng = np.random.default_rng(seed)
    nc = max(8, n // 5000)
    C = rng.normal(size=(nc, d)) * 6
    sz = [n // nc] * nc
    sz[-1] += n - sum(sz)
    X = np.vstack([c + rng.normal(size=(s, d)) for c, s in zip(C, sz)])
    return np.ascontiguousarray(X, dtype=np.float32)


def warm(fn, repeats):
    fn()
    return min(_timed(fn) for _ in range(repeats))


def _timed(fn):
    t = time.perf_counter()
    fn._last = fn()
    return time.perf_counter() - t


def knn_from_csr(g, k):
    g = g.tocsr()
    n = g.shape[0]
    out = np.full((n, k), -1, np.int64)
    for i in range(n):
        s, e = g.indptr[i], g.indptr[i + 1]
        c, v = g.indices[s:e], g.data[s:e]
        out[i] = [x for x in c[np.argsort(v)] if x != i][:k]
    return out


def recall_csr(g, ex, sub):
    a = knn_from_csr(g, K)
    return float(np.mean([len(set(a[i]) & set(ex[j])) / K for j, i in enumerate(sub)]))


def recall_pairs(idx, ex, sub):
    return float(np.mean([len(set(idx[i, 1:]) & set(ex[j])) / K for j, i in enumerate(sub)]))


def main():
    maxN = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    sizes = [s for s in [50_000, 100_000, 200_000, 400_000, 800_000] if s <= maxN]

    from pipnn import self_knn_graph
    from pynndescent import PyNNDescentTransformer

    rows = []
    print(f"{'n':>8} | {'PiPNN s':>8} {'rec':>6} | {'pynn-def s':>10} {'rec':>6} | "
          f"{'pynn-match s':>12} {'rec':>6} {'build_k':>7} | {'slowdown':>8}")
    for n in sizes:
        X = make_data(n)
        sub = np.random.default_rng(1).choice(n, 4000, replace=False)
        _, ex = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X[sub])
        ex = ex[:, 1:]
        reps = 3 if n <= 100_000 else 2

        # PiPNN
        gp = [None]
        def run_pip():
            gp[0] = self_knn_graph(X, n_neighbors=K, random_state=0)
            return gp[0]
        t_pip = warm(run_pip, reps)
        r_pip = recall_pairs(gp[0][0], ex, sub)

        # pynndescent default
        gd = [None]
        def run_def():
            gd[0] = PyNNDescentTransformer(n_neighbors=K, metric="euclidean").fit_transform(X)
            return gd[0]
        t_def = warm(run_def, reps)
        r_def = recall_csr(gd[0], ex, sub)

        # pynndescent tuned: raise build_k until recall >= PiPNN's.
        target = r_pip - 0.002
        chosen = None
        for bk in (30, 45, 60, 90):
            g = PyNNDescentTransformer(n_neighbors=bk, metric="euclidean", **QUAL).fit_transform(X)
            if recall_csr(g, ex, sub) >= target or bk == 90:
                chosen = bk
                break
        gm = [None]
        def run_match():
            gm[0] = PyNNDescentTransformer(n_neighbors=chosen, metric="euclidean", **QUAL).fit_transform(X)
            return gm[0]
        t_match = warm(run_match, reps)
        r_match = recall_csr(gm[0], ex, sub)

        slow = t_match / t_pip
        rows.append(dict(n=n, pipnn_s=t_pip, pipnn_rec=r_pip, pynn_def_s=t_def, pynn_def_rec=r_def,
                         pynn_match_s=t_match, pynn_match_rec=r_match, build_k=chosen, slowdown=slow))
        print(f"{n:8d} | {t_pip:8.2f} {r_pip:6.3f} | {t_def:10.2f} {r_def:6.3f} | "
              f"{t_match:12.2f} {r_match:6.3f} {chosen:7d} | {slow:7.1f}x", flush=True)
        with open(HERE / "matched_recall.json", "w") as f:
            json.dump(rows, f, indent=2)

    _pareto(rows)
    print("\nwrote bench/matched_recall.json and bench/matched_recall.png")


def _pareto(rows, pareto_n=100_000):
    """Recall-vs-time Pareto: pynndescent quality sweep vs PiPNN's single point."""
    from pynndescent import PyNNDescentTransformer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = pareto_n
    X = make_data(n)
    sub = np.random.default_rng(1).choice(n, 4000, replace=False)
    _, ex = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X[sub])
    ex = ex[:, 1:]

    from pipnn import self_knn_graph
    self_knn_graph(X, n_neighbors=K, random_state=0)
    t = time.perf_counter(); idx, _ = self_knn_graph(X, n_neighbors=K, random_state=0)
    pip_t = time.perf_counter() - t
    pip_r = recall_pairs(idx, ex, sub)

    pts = []
    for bk in (15, 30, 45, 60, 90):
        PyNNDescentTransformer(n_neighbors=bk, metric="euclidean", **QUAL).fit_transform(X)
        t = time.perf_counter()
        g = PyNNDescentTransformer(n_neighbors=bk, metric="euclidean", **QUAL).fit_transform(X)
        pts.append((time.perf_counter() - t, recall_csr(g, ex, sub), bk))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "s-", color="#e76f51",
            label="pynndescent (build_k sweep)")
    for tt, rr, bk in pts:
        ax.annotate(f"k={bk}", (tt, rr), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.scatter([pip_t], [pip_r], s=160, marker="*", color="#2a9d8f", zorder=5,
               label=f"PiPNN (default)")
    ax.annotate("PiPNN", (pip_t, pip_r), textcoords="offset points", xytext=(8, 6),
                fontsize=10, color="#2a9d8f", fontweight="bold")
    ax.axhline(pip_r, color="#2a9d8f", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("warm build time (s)"); ax.set_ylabel("recall@15 vs exact")
    ax.set_title(f"Recall vs build time at n={n:,} (50-d)\nupper-left is better (high recall, low time)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(HERE / "matched_recall.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
