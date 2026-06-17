//! Distance metrics.
//!
//! Both supported metrics reduce to *squared Euclidean* ranking on a suitably
//! prepared vector, which is what lets us use a single GEMM kernel everywhere:
//!
//! * `L2`     – rank by squared L2; emitted distance is `sqrt(d2)`.
//! * `Cosine` – vectors are L2-normalized at ingest, so for unit vectors
//!              `‖x−y‖² = 2 − 2·cos(x,y)`; we rank by squared L2 and emit the
//!              cosine distance `1 − cos = d2 / 2`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
}

impl Metric {
    pub fn parse(s: &str) -> Option<Metric> {
        match s.to_ascii_lowercase().as_str() {
            "l2" | "euclidean" | "euclidian" => Some(Metric::L2),
            "cosine" | "angular" => Some(Metric::Cosine),
            _ => None,
        }
    }

    /// Whether rows must be L2-normalized at ingest for this metric.
    pub fn needs_normalize(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// Convert a (non-negative) squared-L2 ranking value into the user-facing
    /// distance that scanpy / sklearn expect to see in the output graph.
    #[inline]
    pub fn emit(self, d2: f32) -> f32 {
        let d2 = d2.max(0.0);
        match self {
            Metric::L2 => d2.sqrt(),
            // unit vectors: d2 = 2 - 2cos  =>  cosine_distance = 1 - cos = d2/2
            Metric::Cosine => 0.5 * d2,
        }
    }
}

/// Squared Euclidean distance between two equal-length slices.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    // Auto-vectorizes well; a hand-written SIMD kernel is a Phase 8 optimization.
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}

/// L2-normalize a vector in place (no-op for the zero vector).
pub fn normalize_in_place(v: &mut [f32]) {
    let mut norm = 0.0f32;
    for &x in v.iter() {
        norm += x * x;
    }
    norm = norm.sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}
