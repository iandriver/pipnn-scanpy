//! Phase 0 ground-truth self-kNN by exact brute force.
//!
//! Used to pin the scanpy/sklearn integration contract before the real PiPNN
//! graph exists, and later as the correctness oracle in tests. Output layout is
//! exactly what the transformer needs: row-major `n * (k+1)`, with each row's
//! self-edge first (distance 0) followed by the `k` nearest neighbors in
//! ascending distance order.

use rayon::prelude::*;

use crate::dataset::{Dataset, Id};

/// Result of a self-kNN query in the layout the Python transformer consumes.
pub struct SelfKnn {
    /// `n * stride` neighbor ids, row-major.
    pub indices: Vec<Id>,
    /// `n * stride` emitted distances, row-major, aligned with `indices`.
    pub distances: Vec<f32>,
    /// Number of entries per row (`k + 1`, including the self edge).
    pub stride: usize,
}

/// Exact self-kNN. Returns `k` neighbors plus the self edge per point.
pub fn knn_self_bruteforce(data: &Dataset, k: usize) -> SelfKnn {
    let n = data.n;
    let stride = (k + 1).min(n.max(1));
    let mut indices = vec![0 as Id; n * stride];
    let mut distances = vec![0.0f32; n * stride];

    indices
        .par_chunks_mut(stride)
        .zip(distances.par_chunks_mut(stride))
        .enumerate()
        .for_each(|(i, (idx_row, dist_row))| {
            // (sq_dist, id) heap-free top-`stride` via a small sorted insert.
            let mut best: Vec<(f32, Id)> = Vec::with_capacity(stride + 1);
            for j in 0..n {
                let d2 = data.sq_dist(i, j);
                // Insert into the running top-`stride` smallest.
                if best.len() < stride {
                    best.push((d2, j as Id));
                    best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                } else if d2 < best[stride - 1].0 {
                    best[stride - 1] = (d2, j as Id);
                    best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }
            }
            // Guarantee the self edge sits at slot 0 (scanpy requires the first
            // neighbor of each row to be the point itself). Duplicate points can
            // tie at distance 0, so place self explicitly rather than trusting
            // sort order.
            if let Some(pos) = best.iter().position(|&(_, id)| id == i as Id) {
                best.swap(0, pos);
            }
            for (slot, &(d2, id)) in best.iter().enumerate() {
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
