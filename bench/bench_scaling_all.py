"""Graph-ANN scaling sweep (5k–800k): PiPNN vs our HNSW vs FAISS HNSW vs pynndescent.

Warm min-of-N build time + recall@15 vs exact, 50-d synthetic, constant cluster
density, library defaults. (Quantized/exact backends omitted — see bench_xcompare.py
for the full per-size spread.)

    .venv/bin/python bench/bench_scaling_all.py [maxN]

Output: bench/scaling_all.json, bench/scaling_all.png
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
    fn()
    out = [None]

    def timed():
        t = time.perf_counter()
        out[0] = fn()
        return time.perf_counter() - t

    return min(timed() for _ in range(repeats)), out[0]


def rows_from_csr(g, n):
    g = g.tocsr()
    out = np.full((n, K), -1, np.int64)
    for i in range(n):
        s, e = g.indptr[i], g.indptr[i + 1]
        order = np.argsort(g.data[s:e])
        nbrs = [c for c in g.indices[s:e][order] if c != i][:K]
        out[i, : len(nbrs)] = nbrs
    return out


def recall(rows, exact_sub, sub):
    return float(np.mean([
        len(set(rows[i]) & set(exact_sub[j])) / K for j, i in enumerate(sub)
    ]))


def build_backends():
    """name -> (build_fn(X) -> result, extract_fn(result, n) -> (n,K) neighbor ids)."""
    b = {}
    from pipnn import self_knn_graph
    b["PiPNN"] = (
        lambda X: self_knn_graph(X, n_neighbors=K, random_state=0),
        lambda r, n: np.asarray(r[0])[:, 1:K + 1],  # drop self col 0
    )
    from pipnn.contrib import HnswTransformer
    b["HNSW (ours)"] = (
        lambda X: HnswTransformer(n_neighbors=K).fit_transform(X), rows_from_csr)
    try:
        from pipnn.contrib import FaissTransformer
        import faiss  # noqa: F401
        b["FAISS HNSW"] = (
            lambda X: FaissTransformer(n_neighbors=K, index_type="hnsw").fit_transform(X),
            rows_from_csr)
    except Exception:
        pass
    try:
        from pynndescent import PyNNDescentTransformer
        b["pynndescent"] = (
            lambda X: PyNNDescentTransformer(n_neighbors=K, metric="euclidean").fit_transform(X),
            rows_from_csr)
    except Exception:
        pass
    return b


def main():
    maxN = int(sys.argv[1]) if len(sys.argv) > 1 else 800_000
    sizes = [s for s in [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 800_000]
             if s <= maxN]
    backends = build_backends()
    names = list(backends.keys())

    rows = []
    print("n        | " + " | ".join(f"{nm:>22}" for nm in names))
    for n in sizes:
        X = make_data(n)
        sub = np.random.default_rng(1).choice(n, min(n, 4000), replace=False)
        _, ex = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X[sub])
        ex = ex[:, 1:]
        reps = 3 if n <= 100_000 else 2

        row = {"n": n}
        cells = []
        for nm, (build, extract) in backends.items():
            t, res = warm(lambda b=build: b(X), reps)
            r = recall(extract(res, n), ex, sub)
            row[nm] = {"s": t, "recall": r}
            cells.append(f"{t:7.2f}s {r:.3f}")
        row["peak_gb"] = rss_gb()
        rows.append(row)
        print(f"{n:8d} | " + " | ".join(f"{c:>22}" for c in cells), flush=True)
        with open(HERE / "scaling_all.json", "w") as f:
            json.dump(rows, f, indent=2)

    _plot(rows, names)
    print("\nwrote bench/scaling_all.json and bench/scaling_all.png")


def _plot(rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n"] for r in rows]
    styles = {"PiPNN": ("o-", "#2a9d8f"), "HNSW (ours)": ("^-", "#8856a7"),
              "FAISS HNSW": ("s-", "#3a7ca5"), "pynndescent": ("D-", "#e76f51")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for nm in names:
        mk, col = styles.get(nm, ("x-", "#666"))
        ax[0].plot(n, [r[nm]["s"] for r in rows], mk, color=col, label=nm)
        ax[1].plot(n, [r[nm]["recall"] for r in rows], mk, color=col, label=nm)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("cells (n)"); ax[0].set_ylabel("warm build time (s)")
    ax[0].set_title("kNN graph build time vs n (warm, log-log)")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
    ax[1].set_xscale("log"); ax[1].set_ylim(0.78, 1.005)
    ax[1].set_xlabel("cells (n)"); ax[1].set_ylabel("recall@15 vs exact")
    ax[1].set_title("Recall vs n (library defaults)")
    ax[1].grid(True, which="both", alpha=0.3); ax[1].legend()
    fig.suptitle("Graph-ANN scaling: PiPNN vs HNSW vs FAISS HNSW vs pynndescent (50-d, warm)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(HERE / "scaling_all.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
