use pipnn_core::{knn_self_direct_with_kernel, BuildParams, Dataset, Metric};

#[test]
fn fused_leaf_matches_exact_with_panel_and_query_tails() {
    for d in [1, 7, 50] {
        let n = 131;
        let x: Vec<f32> = (0..n * d)
            .map(|i| ((i * 7919 + i * i * 13) % 65521) as f32 / 65521.0)
            .collect();
        for metric in [Metric::L2, Metric::Cosine] {
            // One-dimensional positive cosine vectors all have the same
            // direction; normalization roundoff cannot define a unique exact
            // neighbor-ID ordering. Tied vectors are covered separately below.
            if d == 1 && metric == Metric::Cosine { continue; }
            let data = Dataset::new(&x, n, d, metric);
            let params = BuildParams { c_max: 512, metric, ..Default::default() };
            let out = knn_self_direct_with_kernel(&data, &params, 15, true);
            for i in 0..n {
                let mut exact: Vec<_> = (0..n).filter(|&j| j != i)
                    .map(|j| (data.sq_dist(i, j), j as u32)).collect();
                exact.sort_unstable_by(|a,b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                assert_eq!(out.indices[i*16], i as u32);
                for j in 0..15 {
                    assert_eq!(out.indices[i*16+j+1], exact[j].1, "d={d}, row={i}");
                    assert_eq!(out.distances[i*16+j+1], metric.emit(exact[j].0));
                }
            }
        }
    }
}

#[test]
fn duplicate_vectors_use_id_ties_and_exclude_self() {
    let x = vec![0.0; 40 * 7];
    let data = Dataset::new(&x, 40, 7, Metric::L2);
    let out = knn_self_direct_with_kernel(&data, &BuildParams::default(), 15, true);
    for i in 0..40 {
        let expected: Vec<_> = std::iter::once(i as u32)
            .chain((0..40).filter(|&j| j != i).take(15).map(|j| j as u32)).collect();
        assert_eq!(&out.indices[i*16..(i+1)*16], expected);
    }
    assert!(out.distances.iter().all(|&d| d == 0.0));
}
