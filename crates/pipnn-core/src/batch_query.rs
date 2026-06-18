//! Batched self-kNN over the built graph via BeamSearch (the transformer path).
//!
//! Queries the index with every point's own vector and returns `k + 1` results
//! per row (self edge first), in the layout the Python transformer reshapes into
//! a scipy CSR.

use rayon::prelude::*;

use crate::bruteforce::SelfKnn;
use crate::dataset::{Dataset, Id};
use crate::graph::Graph;
use crate::params::SearchParams;
use crate::search::{beam_search, Scratch};

/// Self-kNN for all points by graph search.
///
/// `seeds` optionally provides each point's reservoir candidates (warm start);
/// when present the beam is seeded with them plus the point itself, so the
/// search starts at the answer and converges with far less exploration. Falls
/// back to seeding from the point alone otherwise. Uses a per-thread reusable
/// [`Scratch`] to avoid `O(n)` allocation per query.
pub fn knn_self_graph(
    data: &Dataset,
    graph: &Graph,
    seeds: Option<&[Vec<(Id, f32)>]>,
    k: usize,
    s: &SearchParams,
) -> SelfKnn {
    let n = data.n;
    let stride = (k + 1).min(n.max(1));
    let beam_l = s.beam_l.max(stride);

    let mut indices = vec![0 as Id; n * stride];
    let mut distances = vec![0.0f32; n * stride];

    indices
        .par_chunks_mut(stride)
        .zip(distances.par_chunks_mut(stride))
        .enumerate()
        .map_init(
            || (Scratch::new(n), Vec::<Id>::with_capacity(64)),
            |(scratch, seed_buf), (i, (idx_row, dist_row))| {
                seed_buf.clear();
                seed_buf.push(i as Id);
                if let Some(cands) = seeds {
                    for &(id, _) in &cands[i] {
                        seed_buf.push(id);
                    }
                }
                let mut res = beam_search(data, graph, seed_buf, data.row(i), beam_l, scratch);

                // Guarantee the self edge is present and first.
                if !res.iter().any(|&(id, _)| id as usize == i) {
                    res.insert(0, (i as Id, 0.0));
                }
                if let Some(pos) = res.iter().position(|&(id, _)| id as usize == i) {
                    let self_entry = res.remove(pos);
                    res.insert(0, self_entry);
                }
                res.truncate(stride);
                while res.len() < stride {
                    res.push((i as Id, 0.0));
                }
                for (slot, &(id, d2)) in res.iter().enumerate() {
                    idx_row[slot] = id;
                    dist_row[slot] = data.metric.emit(d2);
                }
            },
        )
        .for_each(|_| {});

    SelfKnn {
        indices,
        distances,
        stride,
    }
}

/// Self-kNN directly from each point's saved reservoir candidates (the nearest
/// points found during build), optionally refined with a 1-hop graph expansion.
///
/// For *self*-kNN this is both faster and more accurate than BeamSearch: the
/// reservoir already holds each point's nearest in-build candidates, whereas the
/// graph has been RobustPruned (close-but-redundant edges dropped for navigability).
/// With `refine`, we also pull in the graph neighbors of the point and of its few
/// nearest candidates — recovering true neighbors a point missed but its neighbors
/// found — for a small extra cost.
pub fn knn_self_reservoir(
    data: &Dataset,
    graph: &Graph,
    self_cands: &[Vec<(Id, f32)>],
    k: usize,
    refine: bool,
) -> SelfKnn {
    let n = data.n;
    let stride = (k + 1).min(n.max(1));
    let mut indices = vec![0 as Id; n * stride];
    let mut distances = vec![0.0f32; n * stride];

    indices
        .par_chunks_mut(stride)
        .zip(distances.par_chunks_mut(stride))
        .enumerate()
        .for_each(|(i, (idx_row, dist_row))| {
            let mut cand: Vec<Id> = Vec::with_capacity(64);
            for &(id, _) in &self_cands[i] {
                cand.push(id);
            }
            if refine {
                for &nb in graph.neighbors(i) {
                    cand.push(nb);
                }
                // 1-hop from the few nearest candidates.
                let seeds = self_cands[i].len().min(6);
                for c in 0..seeds {
                    let cid = self_cands[i][c].0 as usize;
                    for &nb in graph.neighbors(cid) {
                        cand.push(nb);
                    }
                }
            }
            cand.sort_unstable();
            cand.dedup();

            // Exact distances, excluding self.
            let mut scored: Vec<(f32, Id)> = cand
                .iter()
                .filter(|&&id| id as usize != i)
                .map(|&id| (data.sq_dist(i, id as usize), id))
                .collect();
            if scored.len() > k {
                scored.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
                scored.truncate(k);
            }
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            idx_row[0] = i as Id;
            dist_row[0] = 0.0;
            let mut slot = 1;
            for &(d2, id) in &scored {
                if slot >= stride {
                    break;
                }
                idx_row[slot] = id;
                dist_row[slot] = data.metric.emit(d2);
                slot += 1;
            }
            while slot < stride {
                idx_row[slot] = i as Id;
                dist_row[slot] = 0.0;
                slot += 1;
            }
        });

    SelfKnn {
        indices,
        distances,
        stride,
    }
}
