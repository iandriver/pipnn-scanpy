//! BeamSearch / greedy graph search (paper Alg 1).
//!
//! From entry `s`, repeatedly expand the closest not-yet-visited node in the
//! beam, adding its neighbors, and truncate the beam to the `L` closest to `q`.
//! Returns the beam (≤ `L` closest points found), each with its squared
//! distance to `q`.

use crate::dataset::{Dataset, Id};
use crate::graph::Graph;

/// Returns up to `beam_l` closest points to `q`, sorted ascending by distance.
pub fn beam_search(
    data: &Dataset,
    graph: &Graph,
    entry: Id,
    q: &[f32],
    beam_l: usize,
) -> Vec<(Id, f32)> {
    let n = data.n;
    // `beam`: (d2, id) kept sorted ascending, capped at beam_l.
    let mut beam: Vec<(f32, Id)> = Vec::with_capacity(beam_l + 1);
    // visited[id]: already expanded; in_beam[id]: present in beam (avoid dups).
    let mut visited = vec![false; n];
    let mut in_beam = vec![false; n];

    let s = entry as usize;
    beam.push((data.sq_dist_to(s, q), entry));
    in_beam[s] = true;

    loop {
        // Alg 1 line 4: closest unvisited node in the beam.
        let mut pick: Option<usize> = None;
        for (k, &(_, id)) in beam.iter().enumerate() {
            if !visited[id as usize] {
                pick = Some(k);
                break; // beam is sorted → first unvisited is the closest
            }
        }
        let Some(k) = pick else { break };
        let p = beam[k].1;
        visited[p as usize] = true;

        // Lines 5: add neighbors of p.
        for &nbr in graph.neighbors(p as usize) {
            let ni = nbr as usize;
            if in_beam[ni] {
                continue;
            }
            let d2 = data.sq_dist_to(ni, q);
            // Insert into the sorted beam if it qualifies.
            if beam.len() < beam_l || d2 < beam[beam.len() - 1].0 {
                let pos = beam
                    .binary_search_by(|probe| {
                        probe.0.partial_cmp(&d2).unwrap().then(std::cmp::Ordering::Less)
                    })
                    .unwrap_or_else(|e| e);
                beam.insert(pos, (d2, nbr));
                in_beam[ni] = true;
                // Line 7–8: truncate to L closest.
                if beam.len() > beam_l {
                    let (_, evicted) = beam.pop().unwrap();
                    in_beam[evicted as usize] = false;
                }
            }
        }
    }

    beam.into_iter().map(|(d2, id)| (id, d2)).collect()
}
