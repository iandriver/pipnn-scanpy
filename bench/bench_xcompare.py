"""Cross-comparison: PiPNN vs our HNSW vs FAISS (HNSW / IVF-PQ / Flat) vs
pynndescent vs exact — one size, warm steady-state, time + recall + a bar plot.

    .venv/bin/python bench/bench_xcompare.py [n]

Output: bench/xcompare.json, bench/xcompare.png
"""

import sys
import time
from pathlib import Path

import numpy as np
import anndata as ad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_lib as bl  # noqa: E402

K = 15
ORDER = ["PiPNN", "hnsw", "faiss-hnsw", "faiss-ivfpq", "faiss-flat", "pynndescent", "exact"]


def make_data(n, d=50, seed=0):
    rng = np.random.default_rng(seed)
    nc = max(8, n // 5000)
    C = rng.normal(size=(nc, d)) * 6
    sz = [n // nc] * nc
    sz[-1] += n - sum(sz)
    X = np.vstack([c + rng.normal(size=(s, d)) for c, s in zip(C, sz)])
    return np.ascontiguousarray(X, dtype=np.float32)


def backends(k, n_jobs=-1):
    b = bl.available_backends(k, n_jobs)
    try:
        from pipnn.contrib import FaissTransformer
        import faiss  # noqa: F401
        b["faiss-flat"] = lambda: FaissTransformer(n_neighbors=k, index_type="flat", n_jobs=n_jobs)
    except Exception:
        pass
    return {name: b[name] for name in ORDER if name in b}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    X = make_data(n)
    a = ad.AnnData(X)
    a.obsm["X_pca"] = X
    sub = np.random.default_rng(1).choice(n, min(n, 4000), replace=False)
    exact = bl.exact_knn(X, K)

    rows = []
    print(f"n={n}  k={K}\n{'backend':14s} {'warm(s)':>8s} {'cold(s)':>8s} {'recall':>8s}")
    for name, make in backends(K, -1).items():
        r = bl.timed_neighbors(make, a, K, repeats=2)
        rec = bl._recall_subset(r["adata"], exact, sub, K)
        rows.append(dict(name=name, warm=r["warm_median"], cold=r["cold"], recall=rec))
        print(f"{name:14s} {r['warm_median']:8.3f} {r['cold']:8.3f} {rec:8.4f}", flush=True)
        import json
        with open(HERE / "xcompare.json", "w") as f:
            json.dump({"n": n, "rows": rows}, f, indent=2)

    _plot(rows, n)
    print("\nwrote bench/xcompare.json and bench/xcompare.png")


def _plot(rows, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    colors = ["#2a9d8f" if nm == "PiPNN" else "#8856a7" if nm == "hnsw"
              else "#3a7ca5" if nm.startswith("faiss") else "#e76f51" if nm == "pynndescent"
              else "#999999" for nm in names]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].bar(x, [r["warm"] for r in rows], color=colors)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right")
    ax[0].set_ylabel("warm build time (s)"); ax[0].set_title(f"Build time (lower=better), n={n:,}")
    ax[0].grid(True, axis="y", alpha=0.3)
    ax[1].bar(x, [r["recall"] for r in rows], color=colors)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30, ha="right")
    ax[1].set_ylim(0.5, 1.01); ax[1].set_ylabel("recall@15 vs exact")
    ax[1].set_title("Recall (higher=better)")
    ax[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"kNN graph build: PiPNN vs HNSW vs FAISS vs pynndescent (50-d, warm)", y=1.02)
    plt.tight_layout()
    plt.savefig(HERE / "xcompare.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
