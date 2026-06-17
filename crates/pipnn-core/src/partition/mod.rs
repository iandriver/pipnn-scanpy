//! Overlapping partitioning of the dataset into leaves (paper Alg 5).

pub mod ball_carving;

pub use ball_carving::{partition, Leaf};
