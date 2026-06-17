//! PyO3 bindings: the `pipnn._pipnn` extension module.
//!
//! Responsibilities kept deliberately thin: zero-copy ingest of the `(n, d)`
//! float32 matrix from numpy, parameter marshalling, GIL release around the
//! Rust compute, and handing back flat numpy arrays the Python transformer
//! reshapes into a scipy CSR.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pipnn_core::{
    build_index_with_cands, knn_self_bruteforce, knn_self_graph, knn_self_reservoir, BuildParams,
    Dataset, Metric, SearchParams,
};

/// Build the index and return self-kNN as flat arrays.
///
/// Returns `(indices, distances, stride)` where `indices`/`distances` are
/// length `n * stride` row-major and `stride == k + 1` (self edge first).
///
/// Phase 0 routes through exact brute force; later phases swap in the PiPNN
/// graph build + BeamSearch behind this same signature.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    x, n_neighbors, metric="euclidean",
    m=12, l_max=96, r=64, alpha=1.2, beam_l=64,
    fanout=2, c_min=256, c_max=2048, n_jobs=0, seed=0,
))]
fn build_and_self_knn<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    n_neighbors: usize,
    metric: &str,
    m: usize,
    l_max: usize,
    r: usize,
    alpha: f32,
    beam_l: usize,
    fanout: usize,
    c_min: usize,
    c_max: usize,
    n_jobs: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>, usize)> {
    let metric = Metric::parse(metric)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported metric: {metric}")))?;

    let arr = x.as_array();
    let (n, d) = (arr.shape()[0], arr.shape()[1]);
    if n == 0 {
        return Err(PyValueError::new_err("empty input matrix"));
    }
    // Contiguous row-major copy into a Vec the Rust side owns (numpy may be
    // non-contiguous; a single copy keeps the hot path simple).
    let flat: Vec<f32> = arr.iter().copied().collect();

    let bp = BuildParams {
        metric,
        m,
        l_max,
        r,
        alpha,
        fanout,
        c_min,
        c_max,
        seed,
    };
    let sp = SearchParams { beam_l };

    // Optionally bound the rayon pool to `n_jobs` threads (0 = all cores).
    let pool = if n_jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs)
            .build()
            .ok()
    } else {
        None
    };

    let knn = py.detach(|| {
        let data = Dataset::new(&flat, n, d, metric);
        let run = || {
            // Tiny inputs: skip the graph and answer exactly (also the recall oracle).
            if n <= bp.c_max.min(2048) && n <= 4096 {
                knn_self_bruteforce(&data, n_neighbors)
            } else {
                let prof = std::env::var("PIPNN_PROFILE").is_ok();
                // Query method (override with PIPNN_QUERY). Default "warm":
                // BeamSearch seeded from each point's reservoir candidates (high
                // recall, fast). "reservoir": candidates + 1-hop only (fastest,
                // lower recall). "beam": cold BeamSearch from the point alone.
                let method = std::env::var("PIPNN_QUERY").unwrap_or_else(|_| "warm".into());
                let m_cands = (n_neighbors + 1).max(32).min(bp.l_max);

                let t = std::time::Instant::now();
                let (graph, cands) = build_index_with_cands(&data, &bp, m_cands);
                let tb = t.elapsed().as_secs_f64();
                let t2 = std::time::Instant::now();
                let r = match method.as_str() {
                    "reservoir" => knn_self_reservoir(&data, &graph, &cands, n_neighbors, true),
                    "beam" => knn_self_graph(&data, &graph, None, n_neighbors, &sp),
                    _ => knn_self_graph(&data, &graph, Some(&cands), n_neighbors, &sp),
                };
                if prof {
                    eprintln!(
                        "[pipnn] BUILD={:.3}s QUERY={:.3}s [{}]",
                        tb,
                        t2.elapsed().as_secs_f64(),
                        method
                    );
                }
                r
            }
        };
        match &pool {
            Some(p) => p.install(run),
            None => run(),
        }
    });

    let stride = knn.stride;
    Ok((
        knn.indices.into_pyarray(py),
        knn.distances.into_pyarray(py),
        stride,
    ))
}

#[pymodule]
fn _pipnn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_and_self_knn, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
