//! Dense all-pairs squared-L2 distances within a leaf via GEMM (paper §4.2).
//!
//! Uses the identity `‖x − y‖² = ‖x‖² + ‖y‖² − 2·x·y`: compute the Gram block
//! `G = X·Xᵀ` with a single matrix multiply, then add the precomputed row norms.
//! For a leaf of `s` points in `d` dims this is one `s×d · d×s` GEMM.

use crate::dataset::Dataset;

/// All-pairs squared distances among the rows listed in `members`.
///
/// Returns a flat `s × s` row-major matrix `D2` where `D2[a*s + b]` is the
/// squared distance between `members[a]` and `members[b]`. The diagonal is
/// clamped to 0.
pub fn leaf_sq_dists(data: &Dataset, members: &[u32]) -> Vec<f32> {
    let s = members.len();
    let d = data.d;

    // Pack the leaf's rows contiguously for a cache-friendly GEMM.
    let mut xs = vec![0.0f32; s * d];
    for (a, &id) in members.iter().enumerate() {
        xs[a * d..(a + 1) * d].copy_from_slice(data.row(id as usize));
    }

    // gram = X · Xᵀ  (s × s), row-major.
    let mut gram = vec![0.0f32; s * s];
    unsafe {
        // C[s×s] = 1.0 · A[s×d] · B[d×s] + 0.0 · C
        // A = xs (row-major: rsa=d, csa=1); B = xsᵀ (row-major xs viewed as
        // d×s column-major: rsb=1, csb=d); C row-major: rsc=s, csc=1.
        matrixmultiply::sgemm(
            s, d, s, 1.0,
            xs.as_ptr(), d as isize, 1,
            xs.as_ptr(), 1, d as isize,
            0.0,
            gram.as_mut_ptr(), s as isize, 1,
        );
    }

    // D2[a,b] = ‖x_a‖² + ‖x_b‖² − 2·gram[a,b]
    let norms: Vec<f32> = members
        .iter()
        .map(|&id| data.sq_norms[id as usize])
        .collect();
    let mut d2 = gram;
    for a in 0..s {
        let na = norms[a];
        let base = a * s;
        for b in 0..s {
            let v = na + norms[b] - 2.0 * d2[base + b];
            d2[base + b] = if a == b { 0.0 } else { v.max(0.0) };
        }
    }
    d2
}
