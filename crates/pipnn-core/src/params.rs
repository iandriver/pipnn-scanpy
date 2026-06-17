//! Build- and search-time parameters with paper-recommended defaults.

use crate::metric::Metric;

#[derive(Clone, Copy, Debug)]
pub struct BuildParams {
    pub metric: Metric,
    /// Number of LSH hyperplanes `m` for HashPrune residual codes (8–16; 12 default).
    pub m: usize,
    /// Per-point reservoir capacity `ℓ_max` (64–192; 96 default).
    pub l_max: usize,
    /// Max out-degree `R` of the final graph after RobustPrune (64 default).
    pub r: usize,
    /// RobustPrune slack `α` (≈1.2).
    pub alpha: f32,
    /// Ball-carving overlap: each point joins its `fanout` nearest leaders
    /// (the replication degree per recursion level; ≈2 → ~2× total overlap).
    pub fanout: usize,
    /// Minimum leaf size; nodes at/below this stop recursing.
    pub c_min: usize,
    /// Maximum leaf size; nodes above this keep recursing.
    pub c_max: usize,
    /// RNG seed (determinism / history-independence checks).
    pub seed: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        BuildParams {
            metric: Metric::L2,
            m: 12,
            l_max: 96,
            r: 64,
            alpha: 1.2,
            fanout: 2,
            c_min: 256,
            c_max: 2048,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    /// Beam width `L` for greedy BeamSearch (≥ k; 100 default).
    pub beam_l: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams { beam_l: 100 }
    }
}
