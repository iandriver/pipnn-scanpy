//! Recall of the PiPNN graph self-kNN vs exact brute force on synthetic data.

use pipnn_core::{
    build_index, knn_self_bruteforce, knn_self_graph, knn_self_hnsw, BuildParams, Dataset,
    HnswParams, Metric, SearchParams,
};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic LCG-based pseudo-gaussian-ish data.
    let mut s = seed.wrapping_add(0x1234_5678);
    let mut v = vec![0.0f32; n * d];
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 33) as f32) / (1u64 << 31) as f32; // ~[0,2)
        *x = u - 1.0;
    }
    v
}

fn recall_at_k(approx: &[u32], exact: &[u32], n: usize, stride: usize) -> f64 {
    let k = stride - 1;
    let mut hits = 0usize;
    for i in 0..n {
        let a = &approx[i * stride + 1..i * stride + stride];
        let e = &exact[i * stride + 1..i * stride + stride];
        for &x in a {
            if e.contains(&x) {
                hits += 1;
            }
        }
    }
    hits as f64 / (n * k) as f64
}

fn gaussian(n: usize, d: usize, seed: u64) -> Vec<f32> {
    // Box–Muller gaussian, matching the distribution that triggered the hang.
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut nextu = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((s >> 33) as f32) / (1u64 << 31) as f32 * 0.5).clamp(1e-7, 1.0 - 1e-7)
    };
    let mut v = vec![0.0f32; n * d];
    let mut i = 0;
    while i < v.len() {
        let u1 = nextu();
        let u2 = nextu();
        let r = (-2.0 * u1.ln()).sqrt();
        v[i] = r * (std::f32::consts::TAU * u2).cos();
        if i + 1 < v.len() {
            v[i + 1] = r * (std::f32::consts::TAU * u2).sin();
        }
        i += 2;
    }
    v
}

#[test]
fn recall_hnsw_vs_bruteforce() {
    let (n, d, k) = (10000usize, 50usize, 15usize);
    let flat = gaussian(n, d, 1);
    let data = Dataset::new(&flat, n, d, Metric::L2);
    let p = HnswParams::default();

    let approx = knn_self_hnsw(&data, &p, k);
    let exact = knn_self_bruteforce(&data, k);
    assert_eq!(approx.stride, exact.stride);
    let r = recall_at_k(&approx.indices, &exact.indices, n, approx.stride);
    println!("hnsw recall@{k} = {r:.4}");
    assert!(r >= 0.95, "hnsw recall {r:.4} too low");
}

#[test]
fn recall_graph_gaussian_10k() {
    let (n, d, k) = (10000usize, 50usize, 15usize);
    let flat = gaussian(n, d, 1);
    let data = Dataset::new(&flat, n, d, Metric::L2);
    let bp = BuildParams::default();
    let sp = SearchParams { beam_l: 100 };
    let graph = build_index(&data, &bp);
    let approx = knn_self_graph(&data, &graph, None, k, &sp);
    let exact = knn_self_bruteforce(&data, k);
    let r = recall_at_k(&approx.indices, &exact.indices, n, approx.stride);
    println!("gaussian recall@{k} = {r:.4}");
    assert!(r >= 0.85, "recall {r:.4} too low");
}

#[test]
fn recall_graph_vs_bruteforce() {
    let (n, d, k) = (5000usize, 50usize, 15usize);
    let flat = synth(n, d, 7);
    let data = Dataset::new(&flat, n, d, Metric::L2);

    let bp = BuildParams::default();
    let sp = SearchParams { beam_l: 100 };

    let graph = build_index(&data, &bp);
    let approx = knn_self_graph(&data, &graph, None, k, &sp);
    let exact = knn_self_bruteforce(&data, k);

    assert_eq!(approx.stride, exact.stride);
    let r = recall_at_k(&approx.indices, &exact.indices, n, approx.stride);
    println!("recall@{k} = {r:.4}");
    assert!(r >= 0.90, "recall {r:.4} too low");
}
