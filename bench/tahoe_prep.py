"""Stream Tahoe-100M cells → scanpy-style PCA(50) embedding, cached for the NN bench.

Tahoe-100M (tahoebio/Tahoe-100M on HF) stores per-cell sparse raw counts:
`genes` (int64 ids, vocab 62710) + `expressions` (float counts). Each cell starts
with a constant sentinel (gene id 1, expr -2.0) which we drop.

Pipeline (memory-bounded — never materializes the full cell×gene matrix):
  1. download enough 28k-cell parquet shards to cover N cells
  2. pick a 2000-gene HVG panel from a sample (per-gene variance of log1p-CPM)
  3. IncrementalPCA(50): fit on a subset, transform all N in dense HVG chunks
Output: bench/tahoe_cache/X_pca_<N>.npy  (float32, N×50)

    .venv/bin/python bench/tahoe_prep.py --n 3000000
"""

import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.decomposition import IncrementalPCA

REPO = "tahoebio/Tahoe-100M"
CACHE = os.path.join(os.path.dirname(__file__), "tahoe_cache")
N_SHARDS = 3388
CELLS_PER_SHARD = 28225
VOCAB = 62713  # max gene id + slack
TARGET_SUM = 1e4


def rss_gb():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if os.uname().sysname == "Darwin" else r / 1e6


def shard_path(i):
    fn = f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
    return hf_hub_download(REPO, fn, repo_type="dataset", local_dir=CACHE)


def iter_cells(n, shards):
    """Yield (genes np.int64, expr np.float32) per cell, up to n, dropping the
    sentinel (expr <= 0). Zero-copy: slices flat arrow values via list offsets
    instead of building Python lists per cell (~5x faster)."""
    seen = 0
    for si in range(shards):
        pf = pq.ParquetFile(shard_path(si))
        for rg in range(pf.num_row_groups):
            b = pf.read_row_group(rg, columns=["genes", "expressions"])
            gc = b.column("genes").combine_chunks()
            ec = b.column("expressions").combine_chunks()
            goff = gc.offsets.to_numpy()
            gval = gc.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            eval_ = ec.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            for i in range(len(goff) - 1):
                s, e = goff[i], goff[i + 1]
                ev = eval_[s:e]
                keep = ev > 0
                yield gval[s:e][keep], ev[keep]
                seen += 1
                if seen >= n:
                    return


def shards_for(n):
    return min(N_SHARDS, (n + CELLS_PER_SHARD - 1) // CELLS_PER_SHARD + 1)


def pick_hvg(n_sample, n_hvg, shards):
    """Top-`n_hvg` genes by variance of log1p-CPM over a streamed sample."""
    s = np.zeros(VOCAB, np.float64)
    s2 = np.zeros(VOCAB, np.float64)
    cnt = 0
    for g, e in iter_cells(n_sample, shards):
        tot = e.sum()
        if tot <= 0:
            continue
        v = np.log1p(e / tot * TARGET_SUM).astype(np.float64)
        s[g] += v
        s2[g] += v * v
        cnt += 1
    mean = s / cnt
    var = s2 / cnt - mean * mean
    var[:2] = -1  # exclude sentinel ids 0,1
    panel = np.sort(np.argsort(var)[-n_hvg:]).astype(np.int64)
    print(f"  HVG panel: {len(panel)} genes from {cnt} sampled cells")
    return panel


def chunks_dense(n, panel, shards, chunk=100_000):
    """Yield dense (rows × len(panel)) log1p-CPM blocks over the HVG panel."""
    col_of = np.full(VOCAB, -1, np.int64)
    col_of[panel] = np.arange(len(panel))
    buf = np.zeros((chunk, len(panel)), np.float32)
    r = 0
    for g, e in iter_cells(n, shards):
        tot = e.sum()
        if tot > 0:
            cols = col_of[g]
            m = cols >= 0
            if m.any():
                buf[r, cols[m]] = np.log1p(e[m] / tot * TARGET_SUM)
        r += 1
        if r == chunk:
            yield buf[:r].copy()
            buf[:] = 0
            r = 0
    if r:
        yield buf[:r].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--pcs", type=int, default=50)
    ap.add_argument("--fit-cells", type=int, default=500_000)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    shards = shards_for(args.n)
    out = os.path.join(CACHE, f"X_pca_{args.n}.npy")
    if os.path.exists(out):
        print(f"cached: {out}")
        return

    t0 = time.perf_counter()
    print(f"[1/3] HVG panel (sample 300k of {shards} shards needed for {args.n:,} cells)")
    panel_f = os.path.join(CACHE, f"hvg_{args.hvg}.npy")
    panel = np.load(panel_f) if os.path.exists(panel_f) else pick_hvg(300_000, args.hvg, shards)
    np.save(panel_f, panel)

    print(f"[2/3] IncrementalPCA fit on {args.fit_cells:,} cells")
    ipca = IncrementalPCA(n_components=args.pcs)
    for blk in chunks_dense(args.fit_cells, panel, shards):
        ipca.partial_fit(blk)
    print(f"  fit done [{time.perf_counter()-t0:.0f}s, peak {rss_gb():.1f}GB]")

    print(f"[3/3] transform {args.n:,} cells")
    X = np.empty((args.n, args.pcs), np.float32)
    r = 0
    for blk in chunks_dense(args.n, panel, shards):
        X[r:r + len(blk)] = ipca.transform(blk).astype(np.float32)
        r += len(blk)
        if r % 500_000 < 100_000:
            print(f"  {r:,}/{args.n:,}  [peak {rss_gb():.1f}GB]", flush=True)
    np.save(out, X[:r])
    print(f"wrote {out}  shape={X[:r].shape}  [{time.perf_counter()-t0:.0f}s total, peak {rss_gb():.1f}GB]")


if __name__ == "__main__":
    main()
