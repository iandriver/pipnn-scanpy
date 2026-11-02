//! Direct self-kNN: union nearest candidates from overlapping leaves.
//!
//! No directional hashing, navigable graph, RobustPrune, or graph search.
//! Dense leaves use GEMM; fused leaves stream scores into bounded candidate
//! lists. Both share exact candidate reranking and the per-point nearest-k merge.

use std::sync::Mutex;
use rayon::prelude::*;
use crate::{BuildParams, Dataset, Id, SelfKnn};
use crate::leaf::leaf_sq_dists;
use crate::partition::partition;

fn order(a: &(f32, Id), b: &(f32, Id)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
}

pub fn knn_self_direct(data: &Dataset, p: &BuildParams, k: usize) -> SelfKnn {
    knn_self_direct_with_kernel(data, p, k, false)
}

/// Same partitioning and candidate merge, with an optional fused leaf scorer.
pub fn knn_self_direct_with_kernel(data: &Dataset, p: &BuildParams, k: usize, fused: bool) -> SelfKnn {
    let n = data.n;
    let k = k.min(n.saturating_sub(1));
    let stride = k + 1;
    let rows: Vec<Mutex<Vec<(f32, Id)>>> = (0..n)
        .map(|_| Mutex::new(Vec::with_capacity(2 * k))).collect();
    if k > 0 {
        for run in 0..p.runs.max(1) {
            let pr = BuildParams {
                seed: p.seed.wrapping_add((run as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                ..*p
            };
            let leaves = partition(data, &pr);
            leaves.par_iter().for_each(|leaf| {
                let s = leaf.len();
                if s < 2 { return; }
                let consume = |a: usize, scratch: &mut Vec<(f32, Id)>| {
                    let src = leaf[a];
                    for (dist, dst) in scratch.iter_mut() {
                        *dist = data.sq_dist(src as usize, *dst as usize);
                    }
                    scratch.sort_unstable_by(order);
                    scratch.truncate(k);
                    let mut row = rows[src as usize].lock().unwrap();
                    row.extend_from_slice(scratch);
                    row.sort_unstable_by(order);
                    row.dedup_by_key(|x| x.1);
                    row.truncate(k);
                };
                if fused {
                    crate::leaf::fused::for_each_topk(data, leaf, 2 * k, consume);
                } else {
                    // Preserve the prior dense scorer and shortlist rule.
                    let d2 = leaf_sq_dists(data, leaf);
                    let mut scratch = Vec::with_capacity(s);
                    for (a, &src) in leaf.iter().enumerate() {
                        scratch.clear();
                        scratch.extend(leaf.iter().enumerate().filter(|&(_, &dst)| dst != src)
                            .map(|(b, &dst)| (d2[a * s + b], dst)));
                        let shortlist = (2 * k).min(scratch.len());
                        if scratch.len() > shortlist {
                            scratch.select_nth_unstable_by(shortlist, order);
                            scratch.truncate(shortlist);
                        }
                        consume(a, &mut scratch);
                    }
                }
            });
        }
    }
    let mut indices = vec![0; n * stride];
    let mut distances = vec![0.0; n * stride];
    let fallback = std::sync::atomic::AtomicUsize::new(0);
    rows.into_par_iter().zip(indices.par_chunks_mut(stride))
        .zip(distances.par_chunks_mut(stride)).enumerate()
        .for_each(|(i, ((row, ids), ds))| {
            let mut row = row.into_inner().unwrap();
            if row.len() < k {
                // A singleton/tiny leaf must not produce invented neighbors.
                // Complete only underfilled rows by exact search; do not use
                // search to improve recall of an already complete row.
                fallback.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                row = (0..n).filter(|&j| j != i)
                    .map(|j| (data.sq_dist(i, j), j as Id)).collect();
                if row.len() > k {
                    row.select_nth_unstable_by(k, order);
                    row.truncate(k);
                }
                row.sort_unstable_by(order);
            }
            ids[0] = i as Id;
            for (j, &(d2, id)) in row.iter().enumerate() {
                ids[j + 1] = id;
                ds[j + 1] = data.metric.emit(d2);
            }
        });
    if std::env::var_os("PIPNN_PROFILE").is_some() {
        eprintln!("[pipnn] DIRECT exact_completion_rows={}", fallback.load(std::sync::atomic::Ordering::Relaxed));
    }
    SelfKnn { indices, distances, stride }
}
