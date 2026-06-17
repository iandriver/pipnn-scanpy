//! HashPrune: online pruning via residualized LSH (paper §3).
//!
//! [`Hyperplanes`] produces per-point residual codes; [`Reservoir`] maintains
//! each point's bounded, history-independent candidate set keyed by those codes.

pub mod hyperplanes;
pub mod reservoir;

pub use hyperplanes::Hyperplanes;
pub use reservoir::{Reservoir, Slot};
