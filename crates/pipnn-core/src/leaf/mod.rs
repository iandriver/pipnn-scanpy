//! Leaf building (paper §4.2 / Alg 4 "Pick"): per-leaf GEMM all-pairs distances,
//! then per-point top in-leaf candidate edges feeding the HashPrune reservoirs.

pub mod build;
pub mod gemm;

pub use build::{pick_leaf, Edge};
pub use gemm::leaf_sq_dists;
