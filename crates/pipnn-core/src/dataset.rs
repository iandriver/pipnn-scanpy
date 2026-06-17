//! Owned view over an `(n, d)` row-major `f32` matrix plus precomputed row norms.
//!
//! For `L2` the caller's buffer is borrowed without copying. For `Cosine` we
//! must store an L2-normalized copy, so the matrix is owned in that case. The
//! `Cow` keeps the common (L2) path zero-copy.

use std::borrow::Cow;

use crate::metric::{normalize_in_place, Metric};

pub type Id = u32;

pub struct Dataset<'a> {
    data: Cow<'a, [f32]>,
    pub n: usize,
    pub d: usize,
    pub metric: Metric,
    /// Squared L2 norm of each row, ‖x_i‖². Used for the GEMM distance identity
    /// `‖x−y‖² = ‖x‖² + ‖y‖² − 2·x·y`.
    pub sq_norms: Vec<f32>,
}

impl<'a> Dataset<'a> {
    /// Build a dataset from a borrowed row-major slice of length `n*d`.
    pub fn new(data: &'a [f32], n: usize, d: usize, metric: Metric) -> Self {
        assert_eq!(data.len(), n * d, "data length must equal n*d");
        let cow: Cow<'a, [f32]> = if metric.needs_normalize() {
            let mut owned = data.to_vec();
            for row in owned.chunks_mut(d.max(1)) {
                normalize_in_place(row);
            }
            Cow::Owned(owned)
        } else {
            Cow::Borrowed(data)
        };

        let sq_norms = cow
            .chunks(d.max(1))
            .map(|row| row.iter().map(|&x| x * x).sum::<f32>())
            .collect();

        Dataset {
            data: cow,
            n,
            d,
            metric,
            sq_norms,
        }
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.d;
        &self.data[start..start + self.d]
    }

    #[inline]
    pub fn flat(&self) -> &[f32] {
        &self.data
    }

    /// Squared-L2 distance between two stored rows (the ranking quantity).
    #[inline]
    pub fn sq_dist(&self, i: usize, j: usize) -> f32 {
        crate::metric::sq_l2(self.row(i), self.row(j))
    }

    /// Squared-L2 distance between a stored row and an arbitrary query vector.
    #[inline]
    pub fn sq_dist_to(&self, i: usize, q: &[f32]) -> f32 {
        crate::metric::sq_l2(self.row(i), q)
    }
}
