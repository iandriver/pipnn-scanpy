"""ANN-benchmark iso-recall harness on real SIFT/GIST (held-out queries).

This is the *paper's* regime — not self-kNN on synthetic PCA data, but external
queries against a base index on standard high-dimensional datasets, scored vs.
the provided exact ground truth. Each backend builds once (build time reported
separately) and we sweep its per-query recall/speed knob, then plot the
recall-vs-QPS Pareto curve (the standard ANN-benchmarks view).

    .venv/bin/python bench/bench_ann_isorecall.py --dataset sift [--n-sub 200000]

Datasets are the ann-benchmarks HDF5 files (train/test/neighbors), cached under
bench/data/. `sift` = SIFT1M (128-d, 1M base, 10k queries); `gist` = GIST1M
(960-d). `--n-sub` subsamples the base for a faster/smaller run (ground truth is
then recomputed exactly for the subsample).

Knobs swept (the recall levers):
  * PiPNN       — beam_L (search width) at fixed `runs` (build passes), on a
                  persistent index so only the query is timed.
  * FAISS HNSW  — efSearch on a prebuilt IndexHNSWFlat (the reference baseline).

Output: bench/ann_isorecall_<dataset>.json, bench/ann_isorecall_<dataset>.png
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
K = 10  # ann-benchmarks convention: recall@10

URLS = {
    "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "gist": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
}


def fetch(dataset: str) -> Path:
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{dataset}-ann-benchmarks.hdf5"
    if path.exists():
        return path
    url = URLS[dataset]
    print(f"downloading {url} -> {path} (large; cached after first run)")
    tmp = path.with_suffix(".part")
    # ann-benchmarks.com is behind Cloudflare, which 403s urllib's default UA.
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:7.1f} / {total/1e6:7.1f} MB", end="", flush=True)
    print()
    tmp.rename(path)
    return path


def load(dataset: str, n_sub):
    import h5py

    with h5py.File(fetch(dataset), "r") as f:
        base = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        query = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt = np.ascontiguousarray(f["neighbors"][:], dtype=np.int64)[:, :K]
    if n_sub and n_sub < len(base):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(base), n_sub, replace=False)
        base = np.ascontiguousarray(base[idx])
        # Provided GT indexes the full base → invalid after subsample; recompute
        # exactly (cached per dataset+n_sub, the slow part of a rerun).
        cache = DATA / f"{dataset}-gt-{n_sub}.npy"
        if cache.exists():
            gt = np.load(cache)
        else:
            from sklearn.neighbors import NearestNeighbors

            print(f"  subsampled base to {n_sub}; recomputing exact GT for queries...")
            _, gt = NearestNeighbors(n_neighbors=K).fit(base).kneighbors(query)
            np.save(cache, gt.astype(np.int64))
    return base, query, gt.astype(np.int64)


def recall(approx, gt):
    return float(np.mean([len(set(approx[i]) & set(gt[i])) / K for i in range(len(gt))]))


def time_query(fn, repeats=3):
    fn()  # warm
    return min((lambda t=time.perf_counter(): (fn(), time.perf_counter() - t)[1])()
               for _ in range(repeats))


def run_pipnn(base, query, gt, runs, beams):
    from pipnn import PiPNNIndex

    tb = time.perf_counter()
    idx = PiPNNIndex(base, runs=runs, random_state=0)
    build_s = time.perf_counter() - tb
    print(f"  [PiPNN runs={runs} build {build_s:.2f}s]")
    out = []
    for bl in beams:
        fn = lambda bl=bl: idx.query(query, n_neighbors=K, beam_L=bl)[0]
        t = time_query(fn)
        out.append({"label": f"runs={runs},beam={bl}", "recall": recall(fn(), gt),
                    "qps": len(query) / t, "build_s": build_s})
        print(f"  PiPNN runs={runs} beam_L={bl:5d} | recall={out[-1]['recall']:.4f} "
              f"| {out[-1]['qps']:8.0f} q/s")
    return out


def run_faiss_hnsw(base, query, gt, efs):
    import faiss

    d = base.shape[1]
    index = faiss.IndexHNSWFlat(d, 16)
    index.hnsw.efConstruction = 200
    tb = time.perf_counter()
    index.add(base)
    build_s = time.perf_counter() - tb
    print(f"  [FAISS HNSW build {build_s:.2f}s]")
    out = []
    for ef in efs:
        index.hnsw.efSearch = int(ef)
        fn = lambda: index.search(query, K)[1]
        t = time_query(fn)
        out.append({"label": f"ef={ef}", "recall": recall(fn(), gt),
                    "qps": len(query) / t, "build_s": build_s})
        print(f"  FAISS HNSW ef={ef:5d} | recall={out[-1]['recall']:.4f} "
              f"| {out[-1]['qps']:8.0f} q/s")
    return out


def run_glass_hnsw(base, query, gt, efs):
    """pyglass (zilliztech/pyglass) HNSW — x86_64 only; skipped where `glass`
    isn't importable (e.g. macOS/arm64). One of the fastest HNSW query engines,
    so it's the toughest query-side baseline."""
    for name in ("glassppy", "glass"):
        try:
            glass = __import__(name)
            break
        except Exception:
            glass = None
    if glass is None:
        print("  [pyglass not importable — skipped (x86_64 manylinux wheel only)]")
        return None

    import os
    glass.set_num_threads(os.cpu_count() or 1)
    index = glass.Index(index_type="HNSW", metric="L2", quant="FP32", R=32, L=200)
    tb = time.perf_counter()
    graph = index.build(base)
    build_s = time.perf_counter() - tb
    print(f"  [pyglass HNSW build {build_s:.2f}s]")
    searcher = glass.Searcher(graph=graph, data=base, metric="L2", quantizer="FP32")
    out = []
    for ef in efs:
        searcher.set_ef(max(int(ef), K + 1))
        searcher.optimize()  # tunes prefetch for this ef (setup, not timed)

        def fn():
            ids, _ = searcher.batch_search(query, K)
            return np.asarray(ids, dtype=np.int64).reshape(len(query), -1)[:, :K]
        t = time_query(fn)
        out.append({"label": f"ef={ef}", "recall": recall(fn(), gt),
                    "qps": len(query) / t, "build_s": build_s})
        print(f"  pyglass HNSW ef={ef:5d} | recall={out[-1]['recall']:.4f} "
              f"| {out[-1]['qps']:8.0f} q/s")
    return out


def run_glass_isolated(base, query, gt, efs):
    """Run the pyglass arm in a child process so its precompiled-SIMD SIGILL
    (its manylinux wheel may use AVX-512 the runner CPU lacks) can't take down
    the whole harness. Returns the result list, or None if glass is absent or
    the child crashed/errored."""
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def worker(qq):
        try:
            qq.put(run_glass_hnsw(base, query, gt, efs))
        except Exception as e:  # pragma: no cover - exercised only on x86 CI
            qq.put({"__error__": repr(e)})

    p = ctx.Process(target=worker, args=(q,))
    p.start()
    p.join()
    if p.exitcode != 0:  # SIGILL etc. → negative exitcode / 132
        print(f"  [pyglass child crashed (exit {p.exitcode}) — likely a CPU/SIMD "
              f"mismatch in the wheel; skipped]")
        return None
    res = q.get() if not q.empty() else None
    if isinstance(res, dict) and "__error__" in res:
        print(f"  [pyglass error: {res['__error__']}] — skipped")
        return None
    return res


def _write(results, dataset):
    with open(HERE / f"ann_isorecall_{dataset}.json", "w") as f:
        json.dump(results, f, indent=2)
    _plot(results, dataset)
    print(f"wrote bench/ann_isorecall_{dataset}.{{json,png}}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(URLS), default="sift")
    ap.add_argument("--n-sub", type=int, default=200_000,
                    help="subsample base to this size (0 = full 1M)")
    args = ap.parse_args()

    base, query, gt = load(args.dataset, args.n_sub or None)
    n, d = base.shape
    print(f"\n{args.dataset}: base={n}x{d}  queries={len(query)}  recall@{K}\n",
          flush=True)

    results = {"dataset": args.dataset, "n": n, "d": d, "nq": len(query)}
    print("PiPNN (persistent index; runs = build passes, beam_L = search width):",
          flush=True)
    results["pipnn"] = (
        run_pipnn(base, query, gt, runs=1, beams=[64, 256, 1024])
        + run_pipnn(base, query, gt, runs=3, beams=[256, 1024, 4096])
    )
    print("\nFAISS HNSW (reference graph-ANN; efSearch swept):", flush=True)
    results["faiss_hnsw"] = run_faiss_hnsw(base, query, gt, efs=[16, 32, 64, 128, 256])
    # Write now so the PiPNN-vs-FAISS artifact survives even if pyglass crashes.
    _write(results, args.dataset)

    print("\npyglass HNSW (x86_64 only; isolated subprocess):", flush=True)
    glass_res = run_glass_isolated(base, query, gt, efs=[16, 32, 64, 128, 256])
    if glass_res:
        results["glass_hnsw"] = glass_res
        _write(results, args.dataset)  # re-write with the third arm


def _plot(results, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8),
                           gridspec_kw={"width_ratios": [2, 1]})
    # Series present in this run (pyglass only on x86_64).
    series = [("PiPNN", "pipnn", "o-", "#2a9d8f"),
              ("FAISS HNSW", "faiss_hnsw", "s-", "#3a7ca5")]
    if "glass_hnsw" in results:
        series.append(("pyglass HNSW", "glass_hnsw", "^-", "#e76f51"))

    # Left: recall-vs-QPS Pareto (query side — higher/right is better).
    for name, key, mk, col in series:
        pts = sorted(results[key], key=lambda p: p["recall"])
        ax[0].plot([p["recall"] for p in pts], [p["qps"] for p in pts], mk,
                   color=col, label=name)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(f"recall@{K} (vs exact)")
    ax[0].set_ylabel("queries / sec (log) — higher is better")
    ax[0].set_title("Query throughput at iso-recall")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend()

    # Right: index build time (the side where PiPNN's GEMM construction wins).
    builds = [
        ("PiPNN\n(runs=1)", next(p["build_s"] for p in results["pipnn"]
                                 if "runs=1" in p["label"]), "#2a9d8f"),
        ("PiPNN\n(runs=3)", next(p["build_s"] for p in results["pipnn"]
                                 if "runs=3" in p["label"]), "#1d7268"),
        ("FAISS\nHNSW", results["faiss_hnsw"][0]["build_s"], "#3a7ca5"),
    ]
    if "glass_hnsw" in results:
        builds.append(("pyglass\nHNSW", results["glass_hnsw"][0]["build_s"], "#e76f51"))
    bars = ax[1].bar([b[0] for b in builds], [b[1] for b in builds],
                     color=[b[2] for b in builds])
    ax[1].bar_label(bars, fmt="%.1fs", padding=3)
    ax[1].set_ylabel("index build time (s) — lower is better")
    ax[1].set_title("Build time (whole base)")
    ax[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"PiPNN vs FAISS HNSW on {dataset} "
                 f"({results['n']}×{results['d']}, {results['nq']} held-out queries)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(HERE / f"ann_isorecall_{dataset}.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
