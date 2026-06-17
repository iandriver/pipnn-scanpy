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
    let n = data.n;
    if n == 0 {
        return Graph::from_adjacency(Vec::new(), 0);
    }

    // Optional stage profiling: set PIPNN_PROFILE=1 to print per-stage timings.
    let prof = std::env::var("PIPNN_PROFILE").is_ok();
    let mark = std::time::Instant::now();
    macro_rules! lap {
        ($label:expr, $t:expr) => {
            if prof {
                eprintln!("[pipnn] {:<14} {:>7.3}s", $label, $t.elapsed().as_secs_f64());
                $t = std::time::Instant::now();
            }
        };
    }
    let mut t = mark;

    // Alg 4 line 2: partition into overlapping leaves.
    let leaves = partition(data, p);
    if prof {
        let tot: usize = leaves.iter().map(|l| l.len()).sum();
        eprintln!("[pipnn] leaves={} repl={:.2}x", leaves.len(), tot as f64 / n as f64);
    }
    lap!("partition", t);

    // HashPrune setup: hyperplanes + precomputed sketch S = X·Hᵀ (n × m).
    let hp = Hyperplanes::new(p.m, data.d, p.seed);
    let sketch = hp.sketch_all(data);
    let m = p.m;
    let cand_cap = p.l_max;
    lap!("sketch", t);

    // Alg 4 lines 3–5 ("Pick" + "Prune_And_Add_Edges"): stream each leaf's
    // candidate edges directly into per-point HashPrune reservoirs. The reservoir
    // is the only persistent state (8·ℓ_max·n bytes) — we never materialize the
    // full candidate-edge list, which would dwarf it. HashPrune is
    // history-independent, so the per-leaf insertion order under lock contention
    // does not affect the final reservoir contents (build stays deterministic).
    let reservoirs: Vec<Mutex<Reservoir>> =
        (0..n).map(|_| Mutex::new(Reservoir::new(p.l_max))).collect();

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
            // Select the `cap` nearest in O(len) via quickselect, then sort just
            // those (paper §4.2 keeps each point's nearest in-leaf candidates).
            // This replaces an O(s²·cap) insertion sort — the leaf-build hot spot.
            if scratch.len() > cap {
                scratch.select_nth_unstable_by(cap, |x, y| {
                    x.0.partial_cmp(&y.0).unwrap()
                });
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
    lap!("leaf+hashprune", t);

    // Alg 4 line 8 ("PruneNode"): RobustPrune each reservoir to ≤ R out-edges.
    let out_adj: Vec<Vec<Id>> = reservoirs
        .into_par_iter()
        .enumerate()
        .map(|(x, res)| {
            let mut c = res.into_inner().unwrap().into_sorted();
            robust_prune(data, x as Id, &mut c, p.alpha, p.r)
        })
        .collect();
    lap!("robustprune", t);

    // Symmetrize for navigability (Vamana-style reverse edges), then cap degree.
    let final_adj = symmetrize(data, &out_adj, p.r);
    let entry = approx_medoid(data);
    lap!("symmetrize", t);
    Graph::from_adjacency(final_adj, entry)
}

/// Add reverse edges and cap each node's degree at `r` by keeping its `r`
/// closest neighbors. Keeps the graph strongly connected for BeamSearch.
fn symmetrize(data: &Dataset, out_adj: &[Vec<Id>], r: usize) -> Vec<Vec<Id>> {
    let n = out_adj.len();
    let mut sets: Vec<Vec<Id>> = out_adj.to_vec();
    for (x, nbrs) in out_adj.iter().enumerate() {
        for &y in nbrs {
            sets[y as usize].push(x as Id);
        }
    }
    (0..n)
        .into_par_iter()
        .map(|x| {
            let mut v = sets[x].clone();
            v.sort_unstable();
            v.dedup();
            v.retain(|&y| y as usize != x);
            if v.len() > r {
                let mut withd: Vec<(f32, Id)> =
                    v.iter().map(|&y| (data.sq_dist(x, y as usize), y)).collect();
                withd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
                withd.truncate(r);
                withd.into_iter().map(|(_, y)| y).collect()
            } else {
                v
            }
        })
        .collect()
}
