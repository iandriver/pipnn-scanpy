//! PiPNN index construction (paper Alg 4).
//!
//! Partition (Randomized Ball Carving) → per-leaf candidate edges → per-point
//! HashPrune reservoir → RobustPrune → symmetrized navigable CSR graph.

use std::sync::Mutex;

use rayon::prelude::*;

use crate::dataset::{Dataset, Id};
use crate::graph::{approx_medoid, Graph};
use crate::hashprune::{Hyperplanes, Reservoir};
use crate::leaf::leaf_sq_dists;
use crate::params::BuildParams;
use crate::partition::partition;
use crate::robust_prune::robust_prune;

/// Build the navigable graph index over `data`.
pub fn build_index(data: &Dataset, p: &BuildParams) -> Graph {
    build_index_with_cands(data, p, 0).0
}

/// Build the index and additionally return, per point, its top-`m_cands`
/// reservoir candidates `(id, sq_dist)` sorted ascending (empty when
/// `m_cands == 0`). These are each point's nearest in-build candidates — the
/// direct, high-recall answer to *self*-kNN, before RobustPrune discards the
/// close-but-redundant ones for navigability. See [`crate::batch_query::knn_self_reservoir`].
#[allow(unused_assignments)] // final `lap!` reassigns the timer it never reads
pub fn build_index_with_cands(
    data: &Dataset,
    p: &BuildParams,
    m_cands: usize,
) -> (Graph, Vec<Vec<(Id, f32)>>) {
    let n = data.n;
    if n == 0 {
        return (Graph::from_adjacency(Vec::new(), 0), Vec::new());
    }

    // Optional stage profiling: set PIPNN_PROFILE=1 to print per-stage timings.
    let prof = std::env::var("PIPNN_PROFILE").is_ok();
    let mark = std::time::Instant::now();
    macro_rules! lap {
        ($label:expr, $t:expr) => {
            if prof {
                eprintln!(
                    "[pipnn] {:<14} {:>7.3}s",
                    $label,
                    $t.elapsed().as_secs_f64()
                );
                $t = std::time::Instant::now();
            }
        };
    }
    let mut t = mark;

    // HashPrune setup: hyperplanes + precomputed sketch S = X·Hᵀ (n × m). The
    // sketch is a function of the points only, so it is shared across all runs.
    let hp = Hyperplanes::new(p.m, data.d, p.seed);
    let sketch = hp.sketch_all(data);
    let m = p.m;
    let cand_cap = p.l_max;
    lap!("sketch", t);

    // Per-point HashPrune reservoirs — the only persistent state (8·ℓ_max·n
    // bytes). We never materialize the full candidate-edge list, which would
    // dwarf it. HashPrune is history-independent, so neither the per-leaf
    // insertion order under lock contention nor the order of runs affects the
    // final reservoir contents (build stays deterministic).
    let reservoirs: Vec<Mutex<Reservoir>> = (0..n)
        .map(|_| Mutex::new(Reservoir::new(p.l_max)))
        .collect();

    // Alg 4 line 2 + lines 3–5, repeated over `runs` independent partitions
    // ("runs" knob). Each pass carves a fresh random partition (distinct seed)
    // and streams its per-leaf candidate edges into the shared reservoirs; the
    // union across passes recovers true neighbors that any single partition
    // splits across a leaf boundary. recall rises with `runs` at ~linear cost.
    let runs = p.runs.max(1);
    for run in 0..runs {
        // Distinct seed per pass (golden-ratio mix) → independent partitions;
        // run 0 reproduces the single-pass build exactly when runs == 1.
        let pr = BuildParams {
            seed: p
                .seed
                .wrapping_add((run as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            ..*p
        };
        let leaves = partition(data, &pr);
        if prof {
            let tot: usize = leaves.iter().map(|l| l.len()).sum();
            eprintln!(
                "[pipnn] run {}/{} leaves={} repl={:.2}x",
                run + 1,
                runs,
                leaves.len(),
                tot as f64 / n as f64
            );
        }

        leaves.par_iter().for_each(|leaf| {
            let s = leaf.len();
            if s <= 1 {
                return;
            }
            let d2 = leaf_sq_dists(data, leaf);
            let cap = cand_cap.min(s - 1);
            // Reusable scratch of (dist, id) pairs for one row's candidates.
            let mut scratch: Vec<(f32, Id)> = Vec::with_capacity(s);
            for a in 0..s {
                let pa = leaf[a] as usize;
                let row = &d2[a * s..(a + 1) * s];
                // Gather this row's candidates (excluding self).
                scratch.clear();
                for b in 0..s {
                    if b != a {
                        scratch.push((row[b], leaf[b]));
                    }
                }
                // Select the `cap` nearest in O(len) via quickselect, then sort
                // just those (paper §4.2 keeps each point's nearest in-leaf
                // candidates). This replaces an O(s²·cap) insertion sort — the
                // leaf-build hot spot.
                if scratch.len() > cap {
                    scratch.select_nth_unstable_by(cap, |x, y| x.0.partial_cmp(&y.0).unwrap());
                    scratch.truncate(cap);
                }
                scratch.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

                let sa = &sketch[pa * m..(pa + 1) * m];
                let mut res = reservoirs[pa].lock().unwrap();
                for &(dist, dst) in &scratch {
                    let sc = &sketch[dst as usize * m..(dst as usize + 1) * m];
                    let code = hp.code_from_sketch(sa, sc);
                    res.insert(dst, code, dist);
                }
            }
        });
    }
    lap!("leaf+hashprune", t);

    // Alg 4 line 8 ("PruneNode"): RobustPrune each reservoir to ≤ R out-edges,
    // written into one flat fixed-stride buffer (`out_idx`, stride r; `out_deg`
    // valid entries/node) rather than a `Vec<Vec>` — millions of tiny per-point
    // allocations were a large share of peak RSS. Optionally stash each point's
    // top-`m_cands` nearest candidates first (the self-kNN seed); RobustPrune
    // mutates/consumes the sorted list.
    let r = p.r;
    let mut out_idx = vec![Id::MAX; n * r];
    let mut out_deg = vec![0u32; n];
    let self_cands: Vec<Vec<(Id, f32)>> = reservoirs
        .into_par_iter()
        .zip(out_idx.par_chunks_mut(r))
        .zip(out_deg.par_iter_mut())
        .enumerate()
        .map(|(x, ((res, slot), deg))| {
            let mut c = res.into_inner().unwrap().into_sorted();
            let cands = if m_cands > 0 {
                c.iter().take(m_cands).copied().collect()
            } else {
                Vec::new()
            };
            let adj = robust_prune(data, x as Id, &mut c, p.alpha, r);
            slot[..adj.len()].copy_from_slice(&adj);
            *deg = adj.len() as u32;
            cands
        })
        .collect();
    lap!("robustprune", t);

    // Symmetrize (Vamana reverse edges) + degree-cap, assembling the CSR graph
    // directly from flat buffers — no Vec<Vec> clone/triple.
    let entry = approx_medoid(data);
    let graph = symmetrize_flat(data, &out_idx, &out_deg, n, r, entry);
    lap!("symmetrize", t);
    (graph, self_cands)
}

/// Add reverse edges, dedup, and cap each node to its `r` closest neighbors,
/// assembling the CSR [`Graph`] directly from the flat fixed-stride out-edge
/// buffer (`out_idx`, stride `r`; `out_deg[x]` valid entries/node). All scratch
/// is flat (counting-sort) instead of per-node `Vec`s, keeping peak RSS low.
fn symmetrize_flat(
    data: &Dataset,
    out_idx: &[Id],
    out_deg: &[u32],
    n: usize,
    r: usize,
    entry: Id,
) -> Graph {
    // 1. Degree of the symmetric multigraph: own out-edges + incoming reverses.
    let mut cnt = vec![0u32; n];
    for x in 0..n {
        let dx = out_deg[x] as usize;
        cnt[x] += dx as u32;
        for &y in &out_idx[x * r..x * r + dx] {
            cnt[y as usize] += 1;
        }
    }
    // 2. Counting-sort fill into one flat buffer `tidx` (≤ 2·n·R entries).
    let mut toff = vec![0usize; n + 1];
    for i in 0..n {
        toff[i + 1] = toff[i] + cnt[i] as usize;
    }
    let mut tidx = vec![0 as Id; toff[n]];
    let mut cur: Vec<usize> = toff[..n].to_vec();
    for x in 0..n {
        let dx = out_deg[x] as usize;
        for &y in &out_idx[x * r..x * r + dx] {
            tidx[cur[x]] = y;
            cur[x] += 1;
            let yi = y as usize;
            tidx[cur[yi]] = x as Id;
            cur[yi] += 1;
        }
    }
    drop(cur);
    drop(cnt);

    // 3. Per-node dedup + drop self + keep the r closest → fixed-stride `fpad`.
    //    Reusable per-thread scratch (no per-node allocation).
    let mut fpad = vec![Id::MAX; n * r];
    let mut fdeg = vec![0u32; n];
    fpad.par_chunks_mut(r)
        .zip(fdeg.par_iter_mut())
        .enumerate()
        .for_each_init(
            || (Vec::<Id>::with_capacity(2 * r), Vec::<(f32, Id)>::with_capacity(2 * r)),
            |(buf, withd), (x, (slot, deg))| {
                buf.clear();
                buf.extend(
                    tidx[toff[x]..toff[x + 1]]
                        .iter()
                        .copied()
                        .filter(|&y| y as usize != x),
                );
                buf.sort_unstable();
                buf.dedup();
                let k = if buf.len() > r {
                    withd.clear();
                    withd.extend(buf.iter().map(|&y| (data.sq_dist(x, y as usize), y)));
                    withd.select_nth_unstable_by(r, |a, b| a.0.partial_cmp(&b.0).unwrap());
                    for (i, &(_, y)) in withd[..r].iter().enumerate() {
                        slot[i] = y;
                    }
                    r
                } else {
                    slot[..buf.len()].copy_from_slice(buf);
                    buf.len()
                };
                *deg = k as u32;
            },
        );
    drop(tidx);
    drop(toff);

    // 4. Compact the fixed-stride buffer into the final CSR (one pass).
    let mut indptr = vec![0usize; n + 1];
    for i in 0..n {
        indptr[i + 1] = indptr[i] + fdeg[i] as usize;
    }
    let mut indices = vec![0 as Id; indptr[n]];
    for x in 0..n {
        let k = fdeg[x] as usize;
        let dst = indptr[x];
        indices[dst..dst + k].copy_from_slice(&fpad[x * r..x * r + k]);
    }
    Graph { indptr, indices, entry }
}
