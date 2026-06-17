"""4-backend comparison run INSIDE the linux/amd64 container (pyglass available).

Outputs a table + JSON (/out/results.json) with cold/warm timing, recall@k,
connectivity nnz, Leiden cluster counts, and ARI-vs-exact for PiPNN, pynndescent,
glass, and exact.

NOTE: under qemu emulation on an arm64 host, timings are indicative only;
recall / ARI / cluster structure are valid.
"""

import json
import os
import sys

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "bench")
import bench_lib as bl

sc.settings.verbosity = 0


def make_data(n=8000, d=50, n_clusters=12, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d)) * 6
    sizes = [n // n_clusters] * n_clusters
    sizes[-1] += n - sum(sizes)
    X = np.vstack([c + rng.normal(size=(s, d)) for c, s in zip(centers, sizes)])
    return X.astype(np.float32)


def main():
    n = int(os.environ.get("BENCH_N", "8000"))
    k = 15
    X = make_data(n=n)
    a = ad.AnnData(X)
    a.obsm["X_pca"] = X
    print(f"n={a.n_obs} d={X.shape[1]} k={k}")
    print("backends:", list(bl.available_backends(k).keys()), flush=True)

    rows = bl.run_comparison(a, k=k, repeats=3)

    # Leiden ARI vs exact.
    for name, r in rows.items():
        sc.tl.leiden(r["adata"], flavor="igraph", n_iterations=2, key_added="leiden")
    ref = rows["exact"]["adata"].obs["leiden"]

    out = {}
    print(f"\n{'backend':14s} {'cold':>7s} {'warm':>7s} {'recall':>8s} "
          f"{'nnz':>9s} {'clusters':>9s} {'ARI':>7s}")
    for name, r in rows.items():
        ari = adjusted_rand_score(ref, r["adata"].obs["leiden"])
        nclust = int(r["adata"].obs["leiden"].nunique())
        print(f"{name:14s} {r['cold']:7.2f} {r['warm_median']:7.2f} "
              f"{r['recall']:8.4f} {r['conn_nnz']:9d} {nclust:9d} {ari:7.4f}")
        out[name] = {
            "cold_s": r["cold"], "warm_s": r["warm_median"], "recall": r["recall"],
            "conn_nnz": r["conn_nnz"], "clusters": nclust, "ari_vs_exact": ari,
        }
    print("\n(timings are qemu-emulated x86 — indicative only)")

    os.makedirs("/out", exist_ok=True)
    with open("/out/results.json", "w") as f:
        json.dump({"n": n, "k": k, "backends": out}, f, indent=2)
    print("wrote /out/results.json")


if __name__ == "__main__":
    main()
