//! Compact HNSW (Malkov & Yashunin) — a native graph-ANN comparison backend.
//!
//! The core algorithm pyglass implements, built from scratch in Rust (pyglass's
//! C++ does not compile on arm64). Reuses the SIMD [`crate::metric::sq_l2`] kernel;
//! the neighbor-selection heuristic is α=1 RobustPrune (HNSW Alg 4). Build is
//! parallel (hnswlib-style concurrent insertion with per-node locks, never
//! nested → deadlock-free); the query is lock-free.
//!
//! **Scalar quantization (SQ8).** Optionally the graph is built and searched on
//! per-dimension 8-bit codes ([`Sq8`]) — pyglass's trick: codes are 4× smaller
//! than f32, so HNSW's random neighbor lookups touch far less cache. The returned
//! neighbors' distances are always recomputed exactly for the output.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Mutex, RwLock};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::bruteforce::SelfKnn;
use crate::dataset::{Dataset, Id};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    None,
    Sq8,
}

#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub quant: Quant,
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams { m: 16, ef_construction: 200, ef_search: 64, quant: Quant::None, seed: 0 }
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

/// Per-dimension 8-bit scalar quantizer. `code[i*d+k]` in `0..=255`; the squared
/// distance is `Σ_k scale2[k]·(code_i[k] − code_j[k])²`.
pub struct Sq8 {
    codes: Vec<u8>,
    scale2: Vec<f32>,
    d: usize,
}

impl Sq8 {
    pub fn new(data: &Dataset) -> Sq8 {
        let (n, d) = (data.n, data.d);
        let mut min = vec![f32::INFINITY; d];
        let mut max = vec![f32::NEG_INFINITY; d];
        for i in 0..n {
            let row = data.row(i);
            for k in 0..d {
                if row[k] < min[k] {
                    min[k] = row[k];
                }
                if row[k] > max[k] {
                    max[k] = row[k];
                }
            }
        }
        let mut step = vec![1.0f32; d];
        for k in 0..d {
            let r = max[k] - min[k];
            step[k] = if r > 0.0 { r / 255.0 } else { 1.0 };
        }
        let mut codes = vec![0u8; n * d];
        for i in 0..n {
            let row = data.row(i);
            let out = &mut codes[i * d..(i + 1) * d];
            for k in 0..d {
                let q = ((row[k] - min[k]) / step[k]).round();
                out[k] = q.clamp(0.0, 255.0) as u8;
            }
        }
        let scale2 = step.iter().map(|&s| s * s).collect();
        Sq8 { codes, scale2, d }
    }

    /// Approximate squared distance between two stored codes (SIMD `f32x8`).
    #[inline]
    pub fn dist(&self, i: usize, j: usize) -> f32 {
        use wide::f32x8;
        let d = self.d;
        let ci = &self.codes[i * d..(i + 1) * d];
        let cj = &self.codes[j * d..(j + 1) * d];
        let mut acc = f32x8::ZERO;
        let mut k = 0;
        while k + 8 <= d {
            let a = f32x8::from(std::array::from_fn::<f32, 8, _>(|t| ci[k + t] as f32));
            let b = f32x8::from(std::array::from_fn::<f32, 8, _>(|t| cj[k + t] as f32));
            let s = f32x8::from(<[f32; 8]>::try_from(&self.scale2[k..k + 8]).unwrap());
            let dq = a - b;
            acc += s * dq * dq;
            k += 8;
        }
        let mut r = acc.reduce_add();
        while k < d {
            let dq = ci[k] as f32 - cj[k] as f32;
            r += self.scale2[k] * dq * dq;
            k += 1;
        }
        r
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

// Squared distances ≥ 0 → raw f32 bits are order-preserving; key the heaps by u32.
#[inline]
fn key(d: f32) -> u32 {
    d.to_bits()
}

/// Greedy search at one `layer` from `entry_pts`. `dist(qid, cand)` is the
/// distance from the fixed query point `qid` to a candidate; `nbrs(node, layer)`
/// yields a node's neighbors. Returns up to `ef` closest `(id, dist)`.
fn search_layer<D, N>(
    dist: &D,
    qid: usize,
    entry_pts: &[Id],
    ef: usize,
    layer: usize,
    vis: &mut Visited,
    nbrs: &N,
) -> Vec<(Id, f32)>
where
    D: Fn(usize, usize) -> f32,
    N: Fn(usize, usize) -> Vec<Id>,
{
    vis.reset();
    let mut cand: BinaryHeap<Reverse<(u32, Id)>> = BinaryHeap::new();
    let mut w: BinaryHeap<(u32, Id)> = BinaryHeap::new();

    for &ep in entry_pts {
        let d = dist(qid, ep as usize);
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
            let d = dist(qid, ei);
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

/// HNSW select-neighbors heuristic (Alg 4, α=1): keep the `m` closest candidates
/// to `q`, skipping any candidate already "covered" by a closer selected one.
fn select_neighbors<D: Fn(usize, usize) -> f32>(
    dist: &D,
    q: usize,
    cands: &mut Vec<(Id, f32)>,
    m: usize,
) -> Vec<Id> {
    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
    let mut out: Vec<Id> = Vec::with_capacity(m);
    for &(c, dc) in cands.iter() {
        if c as usize == q {
            continue;
        }
        if out.len() >= m {
            break;
        }
        if out.iter().all(|&s| dist(s as usize, c as usize) >= dc) {
            out.push(c);
        }
    }
    out
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
    /// Build the HNSW with parallel insertion, using `dist(a, b)` for all distances.
    pub fn build<D>(n: usize, p: &HnswParams, dist: &D) -> Hnsw
    where
        D: Fn(usize, usize) -> f32 + Sync,
    {
        if n == 0 {
            return Hnsw { links: Vec::new(), entry: 0, max_layer: 0 };
        }

        let mut rng = ChaCha8Rng::seed_from_u64(p.seed ^ 0x484E_5357_u64);
        let ml = 1.0f64 / (p.m as f64).ln();
        let node_layer: Vec<usize> = (0..n).map(|_| random_layer(&mut rng, ml)).collect();

        let blinks: Vec<Vec<Mutex<Vec<Id>>>> = node_layer
            .iter()
            .map(|&l| (0..=l).map(|_| Mutex::new(Vec::new())).collect())
            .collect();
        let entry_state = RwLock::new((0u32, node_layer[0]));

        // Neighbor accessor for the build graph: lock, clone, unlock (one lock at
        // a time — never nested — so there is no lock-ordering deadlock).
        let nbrs = |node: usize, layer: usize| -> Vec<Id> {
            blinks[node][layer].lock().unwrap().clone()
        };

        (1..n).into_par_iter().for_each_init(
            || Visited::new(n),
            |vis, node| {
                let (mut ep, max_layer) = *entry_state.read().unwrap();
                let l = node_layer[node];

                let mut lc = max_layer;
                while lc > l {
                    let w = search_layer(dist, node, &[ep], 1, lc, vis, &nbrs);
                    if let Some(id) = nearest(&w) {
                        ep = id;
                    }
                    if lc == 0 {
                        break;
                    }
                    lc -= 1;
                }

                for lc in (0..=l.min(max_layer)).rev() {
                    let w = search_layer(dist, node, &[ep], p.ef_construction, lc, vis, &nbrs);
                    let m = p.m_at(lc);
                    let mut cands = w.clone();
                    let selected = select_neighbors(dist, node, &mut cands, m);

                    *blinks[node][lc].lock().unwrap() = selected.clone();

                    for &nb in &selected {
                        let mut nbl = blinks[nb as usize][lc].lock().unwrap();
                        nbl.push(node as Id);
                        if nbl.len() > m {
                            let mut nbc: Vec<(Id, f32)> = nbl
                                .iter()
                                .map(|&e| (e, dist(nb as usize, e as usize)))
                                .collect();
                            *nbl = select_neighbors(dist, nb as usize, &mut nbc, m);
                        }
                    }

                    if let Some(id) = nearest(&w) {
                        ep = id;
                    }
                }

                if l > max_layer {
                    let mut g = entry_state.write().unwrap();
                    if l > g.1 {
                        *g = (node as Id, l);
                    }
                }
            },
        );

        let links: Vec<Vec<Vec<Id>>> = blinks
            .into_iter()
            .map(|layers| layers.into_iter().map(|mx| mx.into_inner().unwrap()).collect())
            .collect();
        let (entry, max_layer) = *entry_state.read().unwrap();
        Hnsw { links, entry, max_layer }
    }

    /// Top-`k` neighbors of query point `qid` (`(id, dist)`, ascending), by `dist`.
    fn search<D>(&self, dist: &D, qid: usize, k: usize, ef: usize, vis: &mut Visited) -> Vec<(Id, f32)>
    where
        D: Fn(usize, usize) -> f32,
    {
        let nbrs = |node: usize, layer: usize| -> Vec<Id> { self.links[node][layer].clone() };
        let mut ep = self.entry;
        let mut lc = self.max_layer;
        while lc >= 1 {
            let w = search_layer(dist, qid, &[ep], 1, lc, vis, &nbrs);
            if let Some(id) = nearest(&w) {
                ep = id;
            }
            lc -= 1;
        }
        let mut w = search_layer(dist, qid, &[ep], ef.max(k), 0, vis, &nbrs);
        w.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        w.truncate(k);
        w
    }
}

/// Self-kNN over a freshly built HNSW index, in the transformer's `SelfKnn` layout.
/// With `Quant::Sq8`, the graph is built/searched on 8-bit codes; the emitted
/// neighbor distances are always recomputed exactly.
pub fn knn_self_hnsw(data: &Dataset, p: &HnswParams, k: usize) -> SelfKnn {
    match p.quant {
        Quant::Sq8 => {
            let q = Sq8::new(data);
            build_and_query(data, p, k, &|a, b| q.dist(a, b))
        }
        Quant::None => build_and_query(data, p, k, &|a, b| data.sq_dist(a, b)),
    }
}

fn build_and_query<D>(data: &Dataset, p: &HnswParams, k: usize, dist: &D) -> SelfKnn
where
    D: Fn(usize, usize) -> f32 + Sync,
{
    let n = data.n;
    let stride = (k + 1).min(n.max(1));
    let hnsw = Hnsw::build(n, p, dist);
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
                // Search returns candidates by the build metric (possibly SQ8);
                // re-rank/emit with EXACT distances.
                let cands = hnsw.search(dist, i, stride, ef, vis);
                let mut res: Vec<(Id, f32)> =
                    cands.iter().map(|&(id, _)| (id, data.sq_dist(i, id as usize))).collect();
                if !res.iter().any(|&(id, _)| id as usize == i) {
                    res.push((i as Id, 0.0));
                }
                res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
                // Self (dist 0) is now first.
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
