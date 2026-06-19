//! pipnn-core: the PiPNN approximate-nearest-neighbor algorithm.
//!
//! Build pipeline (paper Alg 4): Randomized Ball Carving partitioning →
//! per-leaf GEMM all-pairs + HashPrune online pruning → RobustPrune final graph,
//! queried by greedy BeamSearch.

pub mod batch_query;
pub mod bruteforce;
pub mod build;
pub mod dataset;
pub mod graph;
pub mod hashprune;
pub mod hnsw;
pub mod leaf;
pub mod metric;
pub mod params;
pub mod partition;
pub mod robust_prune;
pub mod search;

pub use batch_query::{knn_self_graph, knn_self_reservoir};
pub use bruteforce::{knn_self_bruteforce, SelfKnn};
pub use build::{build_index, build_index_with_cands};
pub use dataset::{Dataset, Id};
pub use graph::Graph;
pub use hashprune::{Hyperplanes, Reservoir};
pub use hnsw::{knn_self_hnsw, Hnsw, HnswParams};
pub use metric::Metric;
pub use params::{BuildParams, SearchParams};
