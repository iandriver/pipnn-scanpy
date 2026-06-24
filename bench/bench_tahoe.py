"""scanpy self-kNN-graph benchmark on real Tahoe-100M cells (1M / 2M / 3M).

Three fast graph-ANN builders on the cached Tahoe PCA(50) embedding
(bench/tahoe_prep.py): PiPNN vs pyglass (portable build) vs FAISS HNSW. Times the
full k-NN-graph construction (build + self-query, k=15 — what scanpy needs) and
measures recall@15 vs exact on a 4000-cell subsample.

    .venv/bin/python bench/bench_tahoe.py --sizes 1000000 2000000 3000000

Output: bench/tahoe_bench.json, bench/tahoe_bench.png
"""

import argparse
import gc
import json
import os
import time

import numpy as np

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "tahoe_cache")
K = 15


def rss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if os.uname().sysname == "Darwin" else r / 1e6


def exact_neighbors(X, sub):
    """Exact k nearest (excl. self) for the subsample rows, via FAISS flat."""
    import faiss
    idx = faiss.IndexFlatL2(X.shape[1])
    idx.add(X)
    _, I = idx.search(X[sub], K + 1)
    out = np.empty((len(sub), K), np.int64)
    for j, i in enumerate(sub):
        row = I[j][I[j] != i][:K]
        out[j] = row
    del idx
    return out


def recall(ids, exact, sub):
    return float(np.mean([len(set(ids[i]) & set(exact[j])) / K for j, i in enumerate(sub)]))


def run_pipnn(X):
    from pipnn import self_knn_graph
    t = time.perf_counter()
    r = self_knn_graph(X, n_neighbors=K, random_state=0)
    dt = time.perf_counter() - t
    ids = np.asarray(r[0])[:, 1:K + 1]  # drop self col 0
    return dt, ids


def run_faiss(X):
    import faiss
    n, d = X.shape
    t = time.perf_counter()
    index = faiss.IndexHNSWFlat(d, 16)
    index.hnsw.efConstruction = 200
    index.add(X)
    index.hnsw.efSearch = max(64, K + 1)
    _, I = index.search(X, K + 1)
    dt = time.perf_counter() - t
    ids = np.empty((n, K), np.int64)
    for i in range(n):
        row = I[i][I[i] != i][:K]
        ids[i, :len(row)] = row
    del index
    return dt, ids


def run_glass(X):
    import glass
    n, d = X.shape
    t = time.perf_counter()
    index = glass.Index("HNSW", "L2", "FP32", 32, 200)
    g = index.build(X)
    s = glass.Searcher(g, X, "L2", "FP32")
    s.set_ef(max(64, K + 1))
    s.optimize()
    I = np.asarray(s.batch_search(X, K + 1)[0]).reshape(n, -1)
    dt = time.perf_counter() - t
    ids = np.empty((n, K), np.int64)
    for i in range(n):
        row = I[i][I[i] != i][:K]
        ids[i, :len(row)] = row
    del index, g, s
    return dt, ids


BACKENDS = [("PiPNN", run_pipnn), ("pyglass", run_glass), ("FAISS HNSW", run_faiss)]


def _worker(q, fn, X, sub, exact):
    t, ids = fn(X)
    q.put((t, recall(ids, exact, sub), rss_gb()))


def run_isolated(fn, X, sub, exact):
    """Run a backend in a forked child. PiPNN (rayon) and pyglass/FAISS (libomp)
    deadlock when sharing one process (multiple OpenMP/threading runtimes on
    macOS); a subprocess per backend isolates them and also frees memory between
    runs. X is inherited copy-on-write (no serialization)."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, fn, X, sub, exact))
    p.start()
    p.join()
    if p.exitcode != 0 or q.empty():
        print(f"    [backend child exited {p.exitcode}]")
        return None, None, None
    return q.get()


def _exact_worker(q, X, sub):
    q.put(exact_neighbors(X, sub))


def exact_isolated(X, sub):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_exact_worker, args=(q, X, sub))
    p.start()
    res = q.get()  # drain before join (array can exceed pipe buffer)
    p.join()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 2_000_000, 3_000_000])
    ap.add_argument("--source", type=int, default=3_000_000, help="which X_pca_<N>.npy to slice")
    ap.add_argument("--cooldown", type=int, default=0,
                    help="seconds to idle before each backend (thermal headroom on laptops)")
    args = ap.parse_args()

    src = os.path.join(CACHE, f"X_pca_{args.source}.npy")
    full = np.load(src, mmap_mode="r")
    print(f"loaded {src}: {full.shape}\n")

    rows = []
    for n in args.sizes:
        if n > len(full):
            print(f"skip {n:,} (only {len(full):,} cells available)")
            continue
        X = np.ascontiguousarray(full[:n], dtype=np.float32)
        sub = np.random.default_rng(1).choice(n, min(n, 4000), replace=False)
        ex = exact_isolated(X, sub)
        row = {"n": n}
        print(f"=== n={n:,} (d={X.shape[1]}) ===", flush=True)
        for name, fn in BACKENDS:
            if args.cooldown:
                time.sleep(args.cooldown)  # shed heat so build times aren't throttled
            t, rec, peak = run_isolated(fn, X, sub, ex)
            if t is None:
                continue
            row[name] = {"s": t, "recall": rec, "peak_gb": peak}
            print(f"  {name:11s} {t:7.2f}s  recall@{K}={rec:.4f}  [child peak {peak:.1f}GB]", flush=True)
            gc.collect()
        rows.append(row)
        del X, ex
        gc.collect()
        with open(os.path.join(HERE, "tahoe_bench.json"), "w") as f:
            json.dump(rows, f, indent=2)

    _plot(rows)
    print("\nwrote bench/tahoe_bench.{json,png}")


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"PiPNN": ("o-", "#2a9d8f"), "pyglass": ("^-", "#e76f51"),
              "FAISS HNSW": ("s-", "#3a7ca5"), "pynndescent": ("D-", "#9467bd")}
    names = [nm for nm in styles if any(nm in r for r in rows)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for nm in names:
        mk, c = styles[nm]
        pts = [r for r in rows if nm in r]  # a backend may stop early (e.g. OOM)
        xs = [r["n"] for r in pts]
        ax[0].plot(xs, [r[nm]["s"] for r in pts], mk, color=c, label=nm)
        ax[1].plot(xs, [r[nm]["recall"] for r in pts], mk, color=c, label=nm)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("cells"); ax[0].set_ylabel("kNN-graph build time (s)")
    ax[0].set_title("Self-kNN graph build time"); ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()
    ax[1].set_xscale("log")
    ax[1].set_xlabel("cells"); ax[1].set_ylabel(f"recall@{K} vs exact")
    ax[1].set_title("Recall"); ax[1].grid(True, which="both", alpha=0.3); ax[1].legend()
    fig.suptitle("Tahoe-100M scanpy kNN: PiPNN vs pyglass vs FAISS HNSW vs pynndescent "
                 "(real cells, 50-d PCA)", y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "tahoe_bench.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
