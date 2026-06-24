"""Comprehensive dashboard for the Tahoe-100M scanpy-kNN benchmark.

Reads bench/tahoe_bench.json and plots every relevant axis side by side:
build time, recall, peak memory, and throughput. → bench/tahoe_dashboard.png

    .venv/bin/python bench/tahoe_plot_all.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
STYLES = {
    "PiPNN": ("o-", "#2a9d8f"),
    "pyglass": ("^-", "#e76f51"),
    "FAISS HNSW": ("s-", "#3a7ca5"),
    "pynndescent": ("D-", "#9467bd"),
}


def main():
    rows = json.load(open(os.path.join(HERE, "tahoe_bench.json")))
    names = [nm for nm in STYLES if any(nm in r for r in rows)]

    def series(nm, field):
        pts = [r for r in rows if nm in r]
        xs = [r["n"] for r in pts]
        if field == "throughput":
            ys = [r["n"] / r[nm]["s"] / 1e6 for r in pts]  # M cells / s
        elif field == "spm":
            ys = [r[nm]["s"] / (r["n"] / 1e6) for r in pts]  # s per M cells
        else:
            ys = [r[nm][field] for r in pts]
        return xs, ys

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    panels = [
        (ax[0, 0], "s", "kNN-graph build time (s)", "Build time — lower is better",
         dict(xlog=True, ylog=True)),
        (ax[0, 1], "recall", "recall@15 vs exact", "Recall — higher is better",
         dict(xlog=True)),
        (ax[1, 0], "peak_gb", "peak memory (GB)", "Peak memory — lower is better",
         dict(xlog=True)),
        (ax[1, 1], "throughput", "throughput (million cells / s)",
         "Throughput — higher is better", dict(xlog=True)),
    ]
    for a, field, ylabel, title, opt in panels:
        for nm in names:
            mk, c = STYLES[nm]
            xs, ys = series(nm, field)
            a.plot(xs, ys, mk, color=c, label=nm)
        if opt.get("xlog"):
            a.set_xscale("log")
        if opt.get("ylog"):
            a.set_yscale("log")
        a.set_xlabel("cells")
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=9)

    # mark the 48 GB machine ceiling on the memory panel
    ax[1, 0].axhline(48, ls=":", color="#888", lw=1)
    ax[1, 0].text(rows[0]["n"], 44, "48 GB machine RAM", fontsize=8, color="#888")

    fig.suptitle("Tahoe-100M scanpy kNN — PiPNN vs pyglass vs FAISS HNSW vs pynndescent "
                 "(real cells, 50-d PCA, 18-core M-series)", y=1.0, fontsize=13)
    plt.tight_layout()
    out = os.path.join(HERE, "tahoe_dashboard.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
