//! Residualized LSH hyperplanes for HashPrune (paper §3, Eq. 1).
//!
//! For point `p` and candidate `c`, the individualized code is
//! `h_p(c) = ⊕_{i=1..m} [ H_i·(c − p) ≥ 0 ]` over `m` fixed random hyperplanes
//! through the origin, applied to the *residual* `(c − p)`.
//!
//! Computing `H_i·(c − p)` directly is `O(m·d)` per candidate. Instead we
//! precompute the **sketch** `S = X·Hᵀ` (shape `n × m`) once, after which
//! `H_i·(c − p) = S[c][i] − S[p][i]`, giving `O(m)` per candidate with no `d`.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

use crate::dataset::Dataset;

pub struct Hyperplanes {
    /// `m × d` row-major matrix of Gaussian hyperplane normals.
    mat: Vec<f32>,
    pub m: usize,
    pub d: usize,
}

impl Hyperplanes {
    /// Sample `m` random hyperplanes (standard normal entries) in `d` dims.
    pub fn new(m: usize, d: usize, seed: u64) -> Self {
        assert!(m <= 16, "m must be <= 16 (code packs into u16)");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut mat = vec![0.0f32; m * d];
        for x in mat.iter_mut() {
            // Box–Muller for a standard normal from two uniforms.
            let u1: f32 = rng.gen::<f32>().max(1e-9);
            let u2: f32 = rng.gen::<f32>();
            *x = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        }
        Hyperplanes { mat, m, d }
    }

    #[inline]
    fn plane(&self, i: usize) -> &[f32] {
        &self.mat[i * self.d..(i + 1) * self.d]
    }

    /// Precompute `S = X·Hᵀ`, the `n × m` sketch matrix (row-major).
    pub fn sketch_all(&self, data: &Dataset) -> Vec<f32> {
        let n = data.n;
        let m = self.m;
        let mut s = vec![0.0f32; n * m];
        for i in 0..n {
            let row = data.row(i);
            let out = &mut s[i * m..(i + 1) * m];
            for j in 0..m {
                out[j] = dot(self.plane(j), row);
            }
        }
        s
    }

    /// Code for candidate `c` w.r.t. point `p` from their precomputed sketches:
    /// bit `i` set iff `S[c][i] − S[p][i] ≥ 0`.
    #[inline]
    pub fn code_from_sketch(&self, sketch_p: &[f32], sketch_c: &[f32]) -> u16 {
        let mut code: u16 = 0;
        for i in 0..self.m {
            if sketch_c[i] - sketch_p[i] >= 0.0 {
                code |= 1 << i;
            }
        }
        code
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}
