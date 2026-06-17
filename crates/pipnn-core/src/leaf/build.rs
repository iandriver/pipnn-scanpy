//! Pick(leaf): emit each member's top in-leaf candidate edges.
//!
//! Computes the leaf's all-pairs squared distances via GEMM, then for every
//! member keeps its `cand_cap` nearest in-leaf neighbors as candidate edges.
//! These are merged per-point across overlapping leaves in [`crate::build`] and
//! pruned by HashPrune, so a per-leaf cap of `cand_cap` (≈ ℓ_max) loses nothing.

use crate::dataset::{Dataset, Id};

use super::gemm::leaf_sq_dists;

/// A directed candidate edge with its squared distance.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub src: Id,
    pub dst: Id,
    pub d2: f32,
}

/// Produce candidate edges for one leaf. Each member contributes up to
/// `cand_cap` edges to its nearest in-leaf neighbors (excluding itself).
pub fn pick_leaf(data: &Dataset, members: &[Id], cand_cap: usize) -> Vec<Edge> {
    let s = members.len();
    if s <= 1 {
        return Vec::new();
    }
    let d2 = leaf_sq_dists(data, members);
    let cap = cand_cap.min(s - 1);
    let mut edges = Vec::with_capacity(s * cap);

    let mut best: Vec<(f32, Id)> = Vec::with_capacity(cap + 1);
    for a in 0..s {
        best.clear();
        let row = &d2[a * s..(a + 1) * s];
        for b in 0..s {
            if b == a {
                continue;
            }
            let dist = row[b];
            if best.len() < cap {
                best.push((dist, members[b]));
                best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            } else if dist < best[cap - 1].0 {
                best[cap - 1] = (dist, members[b]);
                best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            }
        }
        let src = members[a];
        for &(dist, dst) in &best {
            edges.push(Edge { src, dst, d2: dist });
        }
    }
    edges
}
