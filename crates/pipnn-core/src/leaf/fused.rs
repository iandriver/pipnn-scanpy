//! Float32 scoring with immediate bounded top-k selection.
//!
//! Candidate points are packed into eight-wide panels. Four queries share
//! each panel load; completed dot products go straight into bounded heaps.
//! Each unordered pair is evaluated once and updates both endpoints. No
//! s-by-s matrix is written. This is a portable floating-point
//! kernel, not the paper's quantized integer kernel.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use wide::f32x8;

use crate::{Dataset, Id};

#[derive(Clone, Copy)]
struct Candidate(f32, Id);

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.cmp(other) == Ordering::Equal }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

#[inline]
fn insert(heap: &mut BinaryHeap<Candidate>, cap: usize, candidate: Candidate) {
    if heap.len() < cap {
        heap.push(candidate);
    } else if candidate < *heap.peek().unwrap() {
        *heap.peek_mut().unwrap() = candidate;
    }
}

/// For each leaf row, pass its nearest `cap` scores to `consume`.
///
/// The callback may mutate the scratch list, which is reused for the next
/// row. Both dense and fused direct paths rerank these shortlisted candidates
/// with the original squared-distance function before merging nearest-k.
pub fn for_each_topk(
    data: &Dataset,
    members: &[Id],
    cap: usize,
    mut consume: impl FnMut(usize, &mut Vec<(f32, Id)>),
) {
    let (s, d) = (members.len(), data.d);
    let cap = cap.min(s.saturating_sub(1));
    if cap == 0 { return; }
    let panels = s.div_ceil(8);
    let mut packed = vec![f32x8::ZERO; panels * d];
    let mut norms = vec![f32x8::ZERO; panels];
    for panel in 0..panels {
        norms[panel] = f32x8::new(std::array::from_fn(|lane| {
            let j = panel * 8 + lane;
            if j < s { data.sq_norms[members[j] as usize] } else { 0.0 }
        }));
        for dim in 0..d {
            packed[panel * d + dim] = f32x8::new(std::array::from_fn(|lane| {
                let j = panel * 8 + lane;
                if j < s { data.row(members[j] as usize)[dim] } else { 0.0 }
            }));
        }
    }
    let mut heaps: Vec<BinaryHeap<Candidate>> =
        (0..s).map(|_| BinaryHeap::with_capacity(cap)).collect();
    let mut scratch = Vec::with_capacity(cap);
    for q in (0..s).step_by(4) {
        let count = (s - q).min(4);
        let query: [&[f32]; 4] = std::array::from_fn(|r| {
            data.row(members[(q + r).min(s - 1)] as usize)
        });
        let qnorm: [f32x8; 4] = std::array::from_fn(|r| {
            f32x8::splat(data.sq_norms[members[(q + r).min(s - 1)] as usize])
        });
        for panel in q / 8..panels {
            let (mut a, mut b, mut c, mut e) =
                (f32x8::ZERO, f32x8::ZERO, f32x8::ZERO, f32x8::ZERO);
            for dim in 0..d {
                let v = packed[panel * d + dim];
                a = v.mul_add(f32x8::splat(query[0][dim]), a);
                b = v.mul_add(f32x8::splat(query[1][dim]), b);
                c = v.mul_add(f32x8::splat(query[2][dim]), c);
                e = v.mul_add(f32x8::splat(query[3][dim]), e);
            }
            let scores = [a, b, c, e];
            for r in 0..count {
                let distances = (qnorm[r] + norms[panel] - scores[r] * f32x8::splat(2.0))
                    .max(f32x8::ZERO).to_array();
                for (lane, distance) in distances.into_iter().enumerate() {
                    let j = panel * 8 + lane;
                    if j < s && j > q + r && members[j] != members[q + r] {
                        insert(&mut heaps[q + r], cap, Candidate(distance, members[j]));
                        insert(&mut heaps[j], cap, Candidate(distance, members[q + r]));
                    }
                }
            }
        }
    }
    for (row, heap) in heaps.iter_mut().enumerate() {
        scratch.clear();
        scratch.extend(heap.drain().map(|Candidate(score, id)| (score, id)));
        consume(row, &mut scratch);
    }
}
