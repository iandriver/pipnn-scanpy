//! Final navigable graph in CSR adjacency form.

use crate::dataset::{Dataset, Id};

pub struct Graph {
    /// Length `n + 1`; out-neighbors of `i` are `indices[indptr[i]..indptr[i+1]]`.
    pub indptr: Vec<usize>,
    pub indices: Vec<Id>,
    /// Fixed BeamSearch entry point (the dataset medoid approximation).
    pub entry: Id,
}

impl Graph {
    #[inline]
    pub fn neighbors(&self, i: usize) -> &[Id] {
        &self.indices[self.indptr[i]..self.indptr[i + 1]]
    }

    pub fn n(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Assemble a CSR graph from per-node adjacency lists.
    pub fn from_adjacency(adj: Vec<Vec<Id>>, entry: Id) -> Graph {
        let n = adj.len();
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        indptr.push(0);
        for nbrs in &adj {
            indices.extend_from_slice(nbrs);
            indptr.push(indices.len());
        }
        Graph {
            indptr,
            indices,
            entry,
        }
    }
}

/// Approximate medoid: the point closest to the coordinate-wise mean.
/// Cheap, deterministic, and a fine fixed entry point for BeamSearch.
pub fn approx_medoid(data: &Dataset) -> Id {
    let n = data.n;
    let d = data.d;
    let mut mean = vec![0.0f32; d];
    for i in 0..n {
        let row = data.row(i);
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    let inv = 1.0 / n as f32;
    for m in mean.iter_mut() {
        *m *= inv;
    }
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for i in 0..n {
        let dd = data.sq_dist_to(i, &mean);
        if dd < best_d {
            best_d = dd;
            best = i;
        }
    }
    best as Id
}
