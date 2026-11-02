use pipnn_core::{knn_self_direct, BuildParams, Dataset, Metric};

fn matrix(n: usize, d: usize) -> Vec<f32> {
    (0..n * d).map(|i| (((i * 7919 + i * i * 13) % 65521) as f32) / 65521.0).collect()
}

#[test]
fn full_leaf_matches_exact_with_duplicate_vectors_and_self_first() {
    let (n, d, k) = (257, 7, 15);
    let mut x = matrix(n, d);
    for j in 0..d { x[20 * d + j] = x[j]; }
    for metric in [Metric::L2, Metric::Cosine] {
        let data = Dataset::new(&x, n, d, metric);
        let p = BuildParams { c_max: 1024, metric, ..Default::default() };
        let result = knn_self_direct(&data, &p, k);
        for i in 0..n {
            assert_eq!(result.indices[i * (k + 1)], i as u32);
            let mut exact: Vec<_> = (0..n).filter(|&j| j != i)
                .map(|j| (data.sq_dist(i, j), j as u32)).collect();
            exact.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for j in 0..k {
                assert_eq!(result.indices[i * (k + 1) + j + 1], exact[j].1);
                assert_eq!(result.distances[i * (k + 1) + j + 1], metric.emit(exact[j].0));
            }
        }
    }
}

#[test]
fn overlap_union_is_deterministic_across_thread_counts() {
    let x = matrix(2000, 13);
    let data = Dataset::new(&x, 2000, 13, Metric::L2);
    let p = BuildParams { c_max: 100, fanout: 3, runs: 3, ..Default::default() };
    let run = |threads| rayon::ThreadPoolBuilder::new().num_threads(threads).build()
        .unwrap().install(|| knn_self_direct(&data, &p, 15));
    let a = run(1);
    let b = run(4);
    assert_eq!(a.indices, b.indices);
    assert_eq!(a.distances, b.distances);
    for (i, row) in a.indices.chunks(a.stride).enumerate() {
        assert_eq!(row[0], i as u32);
        let mut unique = row.to_vec(); unique.sort_unstable(); unique.dedup();
        assert_eq!(unique.len(), row.len());
    }
}

#[test]
fn tiny_leaves_get_real_neighbors_instead_of_padding() {
    let x = matrix(80, 5);
    let data = Dataset::new(&x, 80, 5, Metric::L2);
    let p = BuildParams { c_max: 2, fanout: 1, ..Default::default() };
    let result = knn_self_direct(&data, &p, 10);
    for i in 0..80 {
        let mut exact: Vec<_> = (0..80).filter(|&j| j != i)
            .map(|j| (data.sq_dist(i, j), j as u32)).collect();
        exact.sort_unstable_by(|a,b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(&result.indices[i * 11 + 1..(i + 1) * 11],
            &exact[..10].iter().map(|x| x.1).collect::<Vec<_>>());
    }
}
