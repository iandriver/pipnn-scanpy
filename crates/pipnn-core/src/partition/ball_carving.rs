//! Randomized Ball Carving (paper Alg 5) with bounded overlap.
//!
//! Done in two stages so total replication is exactly ≈`fanout`, independent of
//! recursion depth:
//!
//! 1. **Disjoint carving** — recursively assign each point to its single nearest
//!    leader, recursing on balls still larger than `c_max`. Produces a disjoint
//!    cover (replication 1×). Applying overlap at *every* level instead would
//!    replicate a point `fanout^depth` times — catastrophic for memory/time.
//! 2. **Halo overlap** — each point additionally joins its `fanout−1` nearest
//!    *other* leaves (by leaf centroid), so boundary points get candidates from
//!    several leaves and the final graph stays connected across leaf boundaries.

use rayon::prelude::*;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::dataset::{Dataset, Id};
use crate::params::BuildParams;

/// A leaf partition: the ids of its member points.
pub type Leaf = Vec<Id>;

/// Partition all points into overlapping leaves of size ≈ `c_max`.
pub fn partition(data: &Dataset, p: &BuildParams) -> Vec<Leaf> {
    let all: Vec<Id> = (0..data.n as Id).collect();
    let mut leaves = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(p.seed ^ 0x9E37_79B9_7F4A_7C15);
    carve_disjoint(data, p, all, 0, &mut rng, &mut leaves);

    if p.fanout > 1 && leaves.len() > 1 {
        add_halo(data, p, &mut leaves);
    }
    leaves
}

/// Stage 1: disjoint recursive ball carving (each point → nearest leader).
fn carve_disjoint(
    data: &Dataset,
    p: &BuildParams,
    points: Vec<Id>,
    depth: usize,
    rng: &mut ChaCha8Rng,
    out: &mut Vec<Leaf>,
) {
    if points.len() <= p.c_max {
        out.push(points);
        return;
    }

    // Sample leaders so disjoint balls land near c_max/2 — but **cap** the count
    // (bounded branching) so each level is O(|P|·BRANCH), not O(|P|²/c_max). The
    // recursion still produces ~2n/c_max leaves overall; only the per-level
    // assignment cost is bounded. Overlap is added later by the (sub-quadratic)
    // global halo, so coarser per-level splits don't cost recall.
    const BRANCH: usize = 64;
    let n_leaders = ((2 * points.len()) / p.c_max).clamp(2, BRANCH).min(points.len());
    let mut shuffled = points.clone();
    shuffled.shuffle(rng);
    let leaders: Vec<Id> = shuffled[..n_leaders].to_vec();

    // Assign each point to its single nearest leader.
    let mut balls: Vec<Leaf> = vec![Vec::new(); n_leaders];
    for &x in &points {
        let mut best_li = 0usize;
        let mut best_d = f32::INFINITY;
        for (li, &l) in leaders.iter().enumerate() {
            let d2 = data.sq_dist(x as usize, l as usize);
            if d2 < best_d {
                best_d = d2;
                best_li = li;
            }
        }
        balls[best_li].push(x);
    }

    let parent_len = points.len();
    let shrink_threshold = (parent_len as f64 * 0.9) as usize;
    const MAX_DEPTH: usize = 24;
    for ball in balls.into_iter() {
        if ball.is_empty() {
            continue;
        }
        if ball.len() <= p.c_max {
            out.push(ball);
        } else if ball.len() <= shrink_threshold && depth < MAX_DEPTH {
            carve_disjoint(data, p, ball, depth + 1, rng, out);
        } else {
            // Degenerate (one leader grabbed ~everything): random-chunk to ensure
            // termination.
            let mut s = ball;
            s.shuffle(rng);
            for chunk in s.chunks(p.c_max) {
                out.push(chunk.to_vec());
            }
        }
    }
}

/// Top-`extra` nearest centroids to `row` among `cands` (centroid indices),
/// excluding `home`. Returns the chosen centroid indices.
#[inline]
fn nearest_centroids(
    row: &[f32],
    centroids: &[f32],
    d: usize,
    extra: usize,
    home: u32,
    cands: &[u32],
) -> Vec<u32> {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(extra + 1);
    for &ci in cands {
        if ci == home {
            continue;
        }
        let c = &centroids[ci as usize * d..(ci as usize + 1) * d];
        let dd = crate::metric::sq_l2(row, c);
        if best.len() < extra {
            best.push((dd, ci));
            best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        } else if dd < best[extra - 1].0 {
            best[extra - 1] = (dd, ci);
            best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
    }
    best.into_iter().map(|(_, ci)| ci).collect()
}

/// Stage 2: add each point to its `fanout-1` nearest other leaves (by centroid).
///
/// A brute scan of all `t` centroids per point is `O(n·t·d) = O(n²·d/c_max)` —
/// the build's super-linear tail. For large `t` we instead build a one-level
/// coarse index over the centroids: cluster them into `√t` super-groups, route
/// each point to its nearest few super-groups, and scan only those members. That
/// is `O(n·√t·d)`, sub-quadratic, with negligible recall cost (the halo only
/// needs *approximately* nearby leaves for overlap, not exact ones).
fn add_halo(data: &Dataset, p: &BuildParams, leaves: &mut Vec<Leaf>) {
    let t = leaves.len();
    let d = data.d;
    let extra = p.fanout.saturating_sub(1).min(t.saturating_sub(1));
    if extra == 0 {
        return;
    }

    // Leaf centroids and each point's home leaf (disjoint → exactly one).
    let mut centroids = vec![0.0f32; t * d];
    let mut home: Vec<u32> = vec![0; data.n];
    for (li, leaf) in leaves.iter().enumerate() {
        let c = &mut centroids[li * d..(li + 1) * d];
        for &pt in leaf {
            home[pt as usize] = li as u32;
            let row = data.row(pt as usize);
            for j in 0..d {
                c[j] += row[j];
            }
        }
        let inv = 1.0 / leaf.len().max(1) as f32;
        for v in c.iter_mut() {
            *v *= inv;
        }
    }

    const BRUTE_T: usize = 256;
    let additions: Vec<(u32, Id)> = if t <= BRUTE_T {
        // Few leaves → exact brute scan is cheap.
        let all: Vec<u32> = (0..t as u32).collect();
        (0..data.n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let row = data.row(i);
                nearest_centroids(row, &centroids, d, extra, home[i], &all)
                    .into_iter()
                    .map(move |ci| (ci, i as Id))
            })
            .collect()
    } else {
        // Coarse index over the centroids: `√t` super-groups.
        let g = (t as f64).sqrt().ceil() as usize;
        let mut rng = ChaCha8Rng::seed_from_u64(p.seed ^ 0xA5A5_5A5A_1234_9876);
        let mut sl: Vec<usize> = (0..t).collect();
        sl.shuffle(&mut rng);
        sl.truncate(g);

        // Assign each centroid to its nearest super-leader, then form super-centroids.
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); g];
        let mut super_c = vec![0.0f32; g * d];
        for ci in 0..t {
            let c = &centroids[ci * d..(ci + 1) * d];
            let mut bg = 0usize;
            let mut bd = f32::INFINITY;
            for (gi, &s) in sl.iter().enumerate() {
                let dd = crate::metric::sq_l2(c, &centroids[s * d..(s + 1) * d]);
                if dd < bd {
                    bd = dd;
                    bg = gi;
                }
            }
            members[bg].push(ci as u32);
            for j in 0..d {
                super_c[bg * d + j] += c[j];
            }
        }
        for gi in 0..g {
            let inv = 1.0 / members[gi].len().max(1) as f32;
            for v in &mut super_c[gi * d..(gi + 1) * d] {
                *v *= inv;
            }
        }

        // Each point: nearest `NG` super-groups, then scan their member centroids.
        const NG: usize = 3;
        (0..data.n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let row = data.row(i);
                let mut bestg: Vec<(f32, usize)> = Vec::with_capacity(NG + 1);
                for gi in 0..g {
                    let dd = crate::metric::sq_l2(row, &super_c[gi * d..(gi + 1) * d]);
                    if bestg.len() < NG {
                        bestg.push((dd, gi));
                        bestg.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    } else if dd < bestg[NG - 1].0 {
                        bestg[NG - 1] = (dd, gi);
                        bestg.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    }
                }
                let mut cand: Vec<u32> = Vec::new();
                for &(_, gi) in &bestg {
                    cand.extend_from_slice(&members[gi]);
                }
                nearest_centroids(row, &centroids, d, extra, home[i], &cand)
                    .into_iter()
                    .map(move |ci| (ci, i as Id))
            })
            .collect()
    };

    for (li, pt) in additions {
        leaves[li as usize].push(pt);
    }
}
