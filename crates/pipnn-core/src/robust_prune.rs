//! RobustPrune (paper Alg 2 / Vamana): select ≤ R diverse out-neighbors.
//!
//! Greedily takes the closest remaining candidate `y`, keeps the edge `(x, y)`,
//! then discards any candidate `z` that `y` already "covers" — i.e.
//! `α·‖y, z‖ < ‖x, z‖` — pruning redundant edges in similar directions.

use crate::dataset::{Dataset, Id};

/// Run RobustPrune for point `x` over `candidates` given as `(id, sq_dist_to_x)`.
/// Returns up to `r` selected neighbor ids (closest first).
pub fn robust_prune(
    data: &Dataset,
    x: Id,
    candidates: &mut Vec<(Id, f32)>,
    alpha: f32,
    r: usize,
) -> Vec<Id> {
    // Closest first; the loop consumes from the front.
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    // Drop any self-references defensively.
    candidates.retain(|&(id, _)| id != x);

    let mut out: Vec<Id> = Vec::with_capacity(r);
    let mut alive = vec![true; candidates.len()];

    for i in 0..candidates.len() {
        if out.len() >= r {
            break;
        }
        if !alive[i] {
            continue;
        }
        let (y, _) = candidates[i];
        out.push(y);

        // Prune candidates dominated by y: α·d(y,z) < d(x,z).
        for j in (i + 1)..candidates.len() {
            if !alive[j] {
                continue;
            }
            let (z, dxz) = candidates[j];
            let dyz = data.sq_dist(y as usize, z as usize);
            // Compare in squared space: α·‖y,z‖ < ‖x,z‖  ⇔  α²·d²(y,z) < d²(x,z).
            if (alpha * alpha) * dyz < dxz {
                alive[j] = false;
            }
        }
    }
    out
}
