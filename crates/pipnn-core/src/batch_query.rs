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
use crate::search::beam_search;

/// Self-kNN for all points by graph search.
pub fn knn_self_graph(data: &Dataset, graph: &Graph, k: usize, s: &SearchParams) -> SelfKnn {
    let n = data.n;
    let stride = (k + 1).min(n.max(1));
    // Beam must be at least wide enough to return `stride` results.
    let beam_l = s.beam_l.max(stride);

    let mut indices = vec![0 as Id; n * stride];
    let mut distances = vec![0.0f32; n * stride];

    indices
        .par_chunks_mut(stride)
        .zip(distances.par_chunks_mut(stride))
        .enumerate()
        .for_each(|(i, (idx_row, dist_row))| {
            // Self-kNN: the query *is* node i, so seed the search at i and
            // explore its own graph neighborhood. This is both faster and
            // robust to a graph that is disconnected across well-separated
            // clusters (a global entry point could not reach other components).
            let mut res = beam_search(data, graph, i as Id, data.row(i), beam_l);
            // Guarantee the self edge is present and first (graph search may
            // converge without landing exactly on i if the graph is sparse).
            if !res.iter().any(|&(id, _)| id as usize == i) {
                res.insert(0, (i as Id, 0.0));
            }
            // res is sorted ascending by squared distance; move self to front.
            if let Some(pos) = res.iter().position(|&(id, _)| id as usize == i) {
                let self_entry = res.remove(pos);
                res.insert(0, self_entry);
            }
            res.truncate(stride);
            // Pad (only possible for pathological tiny/degenerate inputs).
            while res.len() < stride {
                res.push((i as Id, 0.0));
            }
            for (slot, &(id, d2)) in res.iter().enumerate() {
                idx_row[slot] = id;
                dist_row[slot] = data.metric.emit(d2);
            }
        });

    SelfKnn {
        indices,
        distances,
        stride,
    }
}
