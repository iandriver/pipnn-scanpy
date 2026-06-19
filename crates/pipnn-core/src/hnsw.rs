//! Compact HNSW (Malkov & Yashunin) — a native graph-ANN comparison backend.
//!
//! This is the core algorithm pyglass implements. We build it from scratch in
//! Rust (pyglass's C++ does not compile on arm64) reusing the crate's pieces:
//! the SIMD [`crate::metric::sq_l2`] kernel, and [`crate::robust_prune`] as the
//! "select neighbors heuristic" (HNSW Alg 4 is exactly α=1 RobustPrune).
//!
//! Build is parallel (hnswlib-style concurrent insertion with per-node locks), so
//! the timing comparison against PiPNN is fair. Like all parallel HNSW builds it
//! is non-deterministic (insertion order varies); the query side is lock-free.
//!
//! Layout: `links[node][layer]` is `node`'s neighbor list at `layer` (layer 0 is
//! densest, degree bound `m0 = 2·m`; higher layers use `m`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Mutex, RwLock};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::bruteforce::SelfKnn;
use crate::dataset::{Dataset, Id};
use crate::robust_prune::robust_prune;

#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams { m: 16, ef_construction: 200, ef_search: 64, seed: 0 }
    }
}

impl HnswParams {
    #[inline]
    fn m_at(&self, layer: usize) -> usize {
        if layer == 0 {
            2 * self.m
        } else {
            self.m
        }
    }
}

pub struct Hnsw {
    links: Vec<Vec<Vec<Id>>>,
    entry: Id,
    max_layer: usize,
}

/// Reusable visited set via generation stamps (O(1) reset, no per-query alloc).
struct Visited {
    stamp: Vec<u32>,
    gen: u32,
}

impl Visited {
    fn new(n: usize) -> Self {
        Visited { stamp: vec![0; n.max(1)], gen: 0 }
    }
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.gen = 1;
        }
    }
    #[inline]
    fn visit(&mut self, id: usize) -> bool {
        if self.stamp[id] == self.gen {
            false
        } else {
            self.stamp[id] = self.gen;
            true
        }
    }
}

// Squared-L2 ≥ 0 → raw f32 bits are order-preserving; key the heaps by u32.
#[inline]
fn key(d: f32) -> u32 {
    d.to_bits()
}

/// Greedy search at one `layer` from `entry_pts`, returning up to `ef` closest
/// nodes to `q` as `(id, sq_dist)`. `nbrs(node, layer)` yields a node's neighbors
/// (a clone, so it works for both the locked build graph and the plain query graph).
fn search_layer<F>(
    data: &Dataset,
    q: &[f32],
    entry_pts: &[Id],
    ef: usize,
    layer: usize,
    vis: &mut Visited,
    nbrs: &F,
) -> Vec<(Id, f32)>
where
    F: Fn(usize, usize) -> Vec<Id>,
{
    vis.reset();
    let mut cand: BinaryHeap<Reverse<(u32, Id)>> = BinaryHeap::new();
    let mut w: BinaryHeap<(u32, Id)> = BinaryHeap::new();

    for &ep in entry_pts {
        let d = data.sq_dist_to(ep as usize, q);
        vis.visit(ep as usize);
        cand.push(Reverse((key(d), ep)));
        w.push((key(d), ep));
    }

    while let Some(Reverse((cd, c))) = cand.pop() {
        if w.len() >= ef {
            if let Some(&(wd, _)) = w.peek() {
                if cd > wd {
                    break;
                }
            }
        }
        for e in nbrs(c as usize, layer) {
            let ei = e as usize;
            if !vis.visit(ei) {
                continue;
            }
            let d = data.sq_dist_to(ei, q);
            let worst = w.peek().map(|&(wd, _)| wd).unwrap_or(u32::MAX);
            if w.len() < ef || key(d) < worst {
                cand.push(Reverse((key(d), e)));
                w.push((key(d), e));
                if w.len() > ef {
                    w.pop();
                }
            }
        }
    }
    w.into_iter().map(|(k, id)| (id, f32::from_bits(k))).collect()
}

#[inline]
fn nearest(w: &[(Id, f32)]) -> Option<Id> {
    w.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|&(id, _)| id)
}

#[inline]
fn random_layer(rng: &mut ChaCha8Rng, ml: f64) -> usize {
    let u: f64 = rng.gen::<f64>().max(1e-12);
    ((-u.ln() * ml).floor() as usize).min(24)
}

impl Hnsw {
    /// Build the HNSW index over `data` with parallel insertion.
    pub fn build(data: &Dataset, p: &HnswParams) -> Hnsw {
        let n = data.n;
        if n == 0 {
            return Hnsw { links: Vec::new(), entry: 0, max_layer: 0 };
        }

        // Per-node top layer, fixed up-front (seeded) so layer assignment is stable.
        let mut rng = ChaCha8Rng::seed_from_u64(p.seed ^ 0x484E_5357_u64);
        let ml = 1.0f64 / (p.m as f64).ln();
        let node_layer: Vec<usize> = (0..n).map(|_| random_layer(&mut rng, ml)).collect();

        // Locked adjacency for concurrent insertion; converted to plain Vecs after.
        let blinks: Vec<Vec<Mutex<Vec<Id>>>> = node_layer
            .iter()
            .map(|&l| (0..=l).map(|_| Mutex::new(Vec::new())).collect())
            .collect();
        // Global entry point + its layer.
        let entry_state = RwLock::new((0u32, node_layer[0]));

        // Neighbor accessor for the build graph: lock, clone, unlock (one lock at
        // a time — never nested — so there is no lock-ordering deadlock).
        let nbrs = |node: usize, layer: usize| -> Vec<Id> {
            blinks[node][layer].lock().unwrap().clone()
        };

        // Insert nodes 1..n in parallel (node 0 is the initial entry, no edges).
        (1..n).into_par_iter().for_each_init(
            || Visited::new(n),
            |vis, node| {
                let q = data.row(node);
                let (mut ep, max_layer) = *entry_state.read().unwrap();
                let l = node_layer[node];

                // Descend layers above l (ef = 1).
                let mut lc = max_layer;
                while lc > l {
                    let w = search_layer(data, q, &[ep], 1, lc, vis, &nbrs);
                    if let Some(id) = nearest(&w) {
                        ep = id;
                    }
                    lc = lc.saturating_sub(1);
                    if lc == 0 && l == 0 {
                        break;
                    }
                }

                // Connect from min(l, max_layer) down to 0.
                for lc in (0..=l.min(max_layer)).rev() {
                    let w = search_layer(data, q, &[ep], p.ef_construction, lc, vis, &nbrs);
                    let m = p.m_at(lc);
                    let mut cands = w.clone();
                    let selected = robust_prune(data, node as Id, &mut cands, 1.0, m);

                    // Set this node's own links (drop the guard before touching others).
                    *blinks[node][lc].lock().unwrap() = selected.clone();

                    // Backward links + prune over-full neighbors, one lock at a time.
                    for &nb in &selected {
                        let mut nbl = blinks[nb as usize][lc].lock().unwrap();
                        nbl.push(node as Id);
                        if nbl.len() > m {
                            let mut nbc: Vec<(Id, f32)> = nbl
                                .iter()
                                .map(|&e| (e, data.sq_dist(nb as usize, e as usize)))
                                .collect();
                            *nbl = robust_prune(data, nb, &mut nbc, 1.0, m);
                        }
                    }

                    if let Some(id) = nearest(&w) {
                        ep = id;
                    }
                }

                // Promote the entry point if this node reaches a new top layer.
                if l > max_layer {
                    let mut g = entry_state.write().unwrap();
                    if l > g.1 {
                        *g = (node as Id, l);
                    }
                }
            },
        );

        // Freeze into the lock-free query structure.
        let links: Vec<Vec<Vec<Id>>> = blinks
            .into_iter()
            .map(|layers| layers.into_iter().map(|mx| mx.into_inner().unwrap()).collect())
            .collect();
        let (entry, max_layer) = *entry_state.read().unwrap();
        Hnsw { links, entry, max_layer }
    }

    /// k nearest neighbors of `q` (as `(id, sq_dist)`, ascending).
    fn search(&self, data: &Dataset, q: &[f32], k: usize, ef: usize, vis: &mut Visited) -> Vec<(Id, f32)> {
        let nbrs = |node: usize, layer: usize| -> Vec<Id> { self.links[node][layer].clone() };
        let mut ep = self.entry;
        let mut lc = self.max_layer;
        while lc >= 1 {
            let w = search_layer(data, q, &[ep], 1, lc, vis, &nbrs);
            if let Some(id) = nearest(&w) {
                ep = id;
            }
            lc -= 1;
        }
        let mut w = search_layer(data, q, &[ep], ef.max(k), 0, vis, &nbrs);
        w.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        w.truncate(k);
        w
    }
}

/// Self-kNN over a freshly built HNSW index, in the transformer's `SelfKnn` layout.
pub fn knn_self_hnsw(data: &Dataset, p: &HnswParams, k: usize) -> SelfKnn {
    let n = data.n;
    let stride = (k + 1).min(n.max(1));
    let hnsw = Hnsw::build(data, p);
    let ef = p.ef_search.max(stride);

    let mut indices = vec![0 as Id; n * stride];
    let mut distances = vec![0.0f32; n * stride];

    indices
        .par_chunks_mut(stride)
        .zip(distances.par_chunks_mut(stride))
        .enumerate()
        .map_init(
            || Visited::new(n),
            |vis, (i, (idx_row, dist_row))| {
                let mut res = hnsw.search(data, data.row(i), stride, ef, vis);
                if !res.iter().any(|&(id, _)| id as usize == i) {
                    res.insert(0, (i as Id, 0.0));
                }
                if let Some(pos) = res.iter().position(|&(id, _)| id as usize == i) {
                    let s = res.remove(pos);
                    res.insert(0, s);
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

    SelfKnn { indices, distances, stride }
}
