//! BeamSearch / greedy graph search (paper Alg 1).
//!
//! From a seed set, repeatedly expand the closest not-yet-visited node in the
//! beam, adding its neighbors, and truncate the beam to the `L` closest to `q`.
//! Returns the beam (≤ `L` closest points found), each with its squared distance.
//!
//! Two efficiency points that matter when running `n` self-queries:
//! * **Reusable scratch.** `visited`/`in_beam` are kept in a per-thread
//!   [`Scratch`] and only the touched entries are reset between queries —
//!   instead of a fresh `vec![false; n]` per query (which is `O(n²)` zeroing
//!   across all queries).
//! * **Warm-start seeds.** Seeding the beam with each point's nearest reservoir
//!   candidates starts the search at the answer, so far fewer expansions are
//!   needed for the same recall.

use crate::dataset::{Dataset, Id};
use crate::graph::Graph;

/// Per-thread reusable search state (avoids per-query `O(n)` allocation/zeroing).
pub struct Scratch {
    visited: Vec<bool>,
    in_beam: Vec<bool>,
    touched: Vec<u32>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Scratch {
            visited: vec![false; n],
            in_beam: vec![false; n],
            touched: Vec::with_capacity(256),
        }
    }

    #[inline]
    fn mark_beam(&mut self, id: u32) {
        if !self.in_beam[id as usize] {
            self.in_beam[id as usize] = true;
            self.touched.push(id);
        }
    }

    /// Reset only the entries touched by the last query.
    fn clear(&mut self) {
        for &id in &self.touched {
            self.visited[id as usize] = false;
            self.in_beam[id as usize] = false;
        }
        self.touched.clear();
    }
}

/// Greedy beam search from `seeds`. Returns up to `beam_l` closest points to `q`,
/// sorted ascending by squared distance. `scratch` is reused across calls.
pub fn beam_search(
    data: &Dataset,
    graph: &Graph,
    seeds: &[Id],
    q: &[f32],
    beam_l: usize,
    scratch: &mut Scratch,
) -> Vec<(Id, f32)> {
    let mut beam: Vec<(f32, Id)> = Vec::with_capacity(beam_l + 1);
    for &s in seeds {
        if !scratch.in_beam[s as usize] {
            beam.push((data.sq_dist_to(s as usize, q), s));
            scratch.mark_beam(s);
        }
    }
    beam.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if beam.len() > beam_l {
        for &(_, id) in &beam[beam_l..] {
            scratch.in_beam[id as usize] = false;
        }
        beam.truncate(beam_l);
    }

    loop {
        // Alg 1 line 4: closest unvisited node in the beam.
        let mut pick: Option<usize> = None;
        for (k, &(_, id)) in beam.iter().enumerate() {
            if !scratch.visited[id as usize] {
                pick = Some(k);
                break; // beam is sorted → first unvisited is the closest
            }
        }
        let Some(k) = pick else { break };
        let p = beam[k].1;
        scratch.visited[p as usize] = true;

        // Line 5: add neighbors of p. Prefetch the next neighbor's vector while
        // scoring the current one — graph edges point all over the dataset, so
        // these are scattered random loads and hiding the latency is a big win.
        let nbrs = graph.neighbors(p as usize);
        for (k, &nbr) in nbrs.iter().enumerate() {
            if let Some(&next) = nbrs.get(k + 1) {
                data.prefetch_row(next as usize);
            }
            let ni = nbr as usize;
            if scratch.in_beam[ni] {
                continue;
            }
            let d2 = data.sq_dist_to(ni, q);
            if beam.len() < beam_l || d2 < beam[beam.len() - 1].0 {
                let pos = beam
                    .binary_search_by(|probe| {
                        probe.0.partial_cmp(&d2).unwrap().then(std::cmp::Ordering::Less)
                    })
                    .unwrap_or_else(|e| e);
                beam.insert(pos, (d2, nbr));
                scratch.mark_beam(nbr);
                // Lines 7–8: truncate to L closest.
                if beam.len() > beam_l {
                    let (_, evicted) = beam.pop().unwrap();
                    scratch.in_beam[evicted as usize] = false;
                }
            }
        }
    }

    let out: Vec<(Id, f32)> = beam.into_iter().map(|(d2, id)| (id, d2)).collect();
    scratch.clear();
    out
}
