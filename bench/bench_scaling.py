"""Scaling sweep: PiPNN vs pynndescent build time + recall across dataset sizes.

Times the kNN-graph build (transformer.fit_transform — the part that differs
between backends) with warmup + median-of-N steady-state timing, so pynndescent's
one-time numba JIT is excluded. Writes a table, a JSON, and a log-log scaling plot.

    .venv/bin/python bench/bench_scaling.py [maxN]

Output: bench/scaling_results.json, bench/scaling.png
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import resource

HERE = Path(__file__).resolve().parent


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / 1e9) if sys.platform == "darwin" else (r / 1e6)


def make_data(n, d=50, seed=0):
    # Constant cluster *density* (~5000 cells/cluster) so the data structure is
    # the same shape at every n — otherwise PiPNN's partition regime shifts as
    # cluster size crosses c_max and the timing curve gets jagged.
    rng = np.random.default_rng(seed)
    n_clusters = max(8, n // 5000)
    centers = rng.normal(size=(n_clusters, d)) * 6
    sizes = [n // n_clusters] * n_clusters
    sizes[-1] += n - sum(sizes)
    X = np.vstack([c + rng.normal(size=(s, d)) for c, s in zip(centers, sizes)])
    return np.ascontiguousarray(X, dtype=np.float32)


def warm_time(fn, repeats):
    """One discarded warmup build, then the MIN of `repeats` timed builds.

    Min is the standard steady-state estimator: it rejects upward noise from
    other processes / GC, leaving the best achievable build time.
    """
    fn()  # warmup (compiles numba kernels for pynndescent)
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return min(ts), out


def neighbors_from_csr(g, k):
    n = g.shape[0]
    out = np.full((n, k), -1, dtype=np.int64)
    g = g.tocsr()
    for i in range(n):
        s, e = g.indptr[i], g.indptr[i + 1]
        cols, vals = g.indices[s:e], g.data[s:e]
        order = np.argsort(vals)
        out[i] = [c for c in cols[order] if c != i][:k]
    return out


def recall_on(g, exact_sub, sub, k):
    approx = neighbors_from_csr(g, k)
    return float(np.mean([
        len(set(approx[i]) & set(exact_sub[j])) / k for j, i in enumerate(sub)
    ]))


def main():
    maxN = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    k = 15
    sizes = [s for s in [5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000,
                         800_000, 1_600_000] if s <= maxN]

    from pipnn import PiPNNTransformer
    from pynndescent import PyNNDescentTransformer
    from sklearn.neighbors import NearestNeighbors

    rows = []
    print(f"{'n':>9s} {'pipnn(s)':>9s} {'pynnd(s)':>9s} {'speedup':>8s} "
          f"{'pip_rec':>8s} {'pyn_rec':>8s} {'peakGB':>7s}", flush=True)

    for n in sizes:
        X = make_data(n)
        repeats = 3 if n <= 100_000 else 2

        t_pip, g_pip = warm_time(
            lambda: PiPNNTransformer(n_neighbors=k).fit_transform(X), repeats)
        t_pyn, g_pyn = warm_time(
            lambda: PyNNDescentTransformer(n_neighbors=k, metric="euclidean").fit_transform(X),
            repeats)

        # recall vs exact on a sample (fit on all, query the sample only).
        m = min(n, 4000)
        sub = np.random.default_rng(1).choice(n, m, replace=False)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, ex = nn.kneighbors(X[sub])
        ex = ex[:, 1:]
        r_pip = recall_on(g_pip, ex, sub, k)
        r_pyn = recall_on(g_pyn, ex, sub, k)

        row = dict(n=n, pipnn_s=t_pip, pynnd_s=t_pyn, speedup=t_pyn / t_pip,
                   pipnn_recall=r_pip, pynnd_recall=r_pyn, peak_gb=rss_gb())
        rows.append(row)
        print(f"{n:9d} {t_pip:9.3f} {t_pyn:9.3f} {row['speedup']:7.2f}x "
              f"{r_pip:8.4f} {r_pyn:8.4f} {row['peak_gb']:7.2f}", flush=True)

        with open(HERE / "scaling_results.json", "w") as f:
            json.dump(rows, f, indent=2)

    _plot(rows)
    print("\nwrote bench/scaling_results.json and bench/scaling.png")


def _crossover(ns, pip, pyn):
    """Interpolate (in log-log) the n where PiPNN and pynndescent build times meet."""
    import math
    for i in range(len(ns) - 1):
        d0, d1 = pip[i] - pyn[i], pip[i + 1] - pyn[i + 1]
        if d0 == 0:
            return ns[i]
        if d0 < 0 <= d1 or d1 < 0 <= d0:  # sign change → crossover in this segment
            r0 = math.log(pip[i] / pyn[i])
            r1 = math.log(pip[i + 1] / pyn[i + 1])
            t = (0 - r0) / (r1 - r0)
            return math.exp(math.log(ns[i]) + t * (math.log(ns[i + 1]) - math.log(ns[i])))
    return None


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n"] for r in rows]
    pip = [r["pipnn_s"] for r in rows]
    pyn = [r["pynnd_s"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    a0, a1 = ax

    # "before" PiPNN curve for the before/after comparison.
    old_path = HERE / "scaling_results_before.json"
    old = json.load(open(old_path)) if old_path.exists() else None

    # --- left: build time vs n (log-log) ---
    cross = _crossover(n, pip, pyn)
    if cross:
        # Shade the region where PiPNN is the faster builder.
        a0.axvspan(min(n) * 0.8, cross, color="#2a9d8f", alpha=0.06, zorder=0)
        a0.axvline(cross, color="#888", ls=":", lw=1, zorder=1)
        a0.annotate(f"crossover ≈{cross/1000:.0f}k\n← PiPNN faster   pynndescent faster →",
                    xy=(cross, min(pip) * 1.6), ha="center", va="bottom",
                    fontsize=8.5, color="#444")

    if old:
        on = [r["n"] for r in old]
        a0.plot(on, [r["pipnn_s"] for r in old], "o--", color="#9ecfc7",
                label="PiPNN (before optim.)", zorder=2)
        # Annotate the before/after gap at the largest shared n.
        nb, sb = on[-1], old[-1]["pipnn_s"]
        if nb == n[-1]:
            sa = pip[-1]
            a0.annotate("", xy=(nb, sa), xytext=(nb, sb),
                        arrowprops=dict(arrowstyle="<->", color="#2a9d8f", lw=1.4))
            a0.annotate(f"{sb/sa:.1f}× faster", xy=(nb, (sa * sb) ** 0.5),
                        xytext=(-6, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=9, color="#2a9d8f", fontweight="bold")

    a0.plot(n, pip, "o-", color="#2a9d8f", label="PiPNN", zorder=3)
    a0.plot(n, pyn, "s-", color="#e76f51", label="pynndescent", zorder=3)
    a0.set_xscale("log"); a0.set_yscale("log")
    a0.set_xlabel("cells (n)"); a0.set_ylabel("warm build time (s)")
    a0.set_title("kNN graph build time vs n  (warm, JIT excluded)")
    a0.grid(True, which="both", alpha=0.3); a0.legend(loc="upper left", fontsize=9)

    # --- right: recall vs n ---
    rp = [r["pipnn_recall"] for r in rows]
    ry = [r["pynnd_recall"] for r in rows]
    a1.plot(n, rp, "o-", color="#2a9d8f", label="PiPNN")
    a1.plot(n, ry, "s-", color="#e76f51", label="pynndescent")
    # Shade + label the recall gap at the largest n.
    a1.fill_between(n, ry, rp, color="#2a9d8f", alpha=0.06)
    a1.annotate(f"≈ +{(rp[-1]-ry[-1])*100:.0f} pts\nrecall", xy=(n[-1], (rp[-1] + ry[-1]) / 2),
                xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                fontsize=8.5, color="#444")
    a1.set_xscale("log"); a1.set_ylim(0.55, 1.02)
    a1.set_xlabel("cells (n)"); a1.set_ylabel("recall@15 vs exact")
    a1.set_title("Recall vs n  (library defaults)")
    a1.grid(True, which="both", alpha=0.3); a1.legend(loc="lower left", fontsize=9)

    fig.suptitle("PiPNN vs pynndescent — single-cell kNN graph build (50-d, warm steady-state, all cores)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(HERE / "scaling.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
