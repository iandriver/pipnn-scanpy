//! HashPrune is history-independent and the build is deterministic: a fixed seed
//! yields a byte-identical graph, and a reservoir's contents do not depend on
//! insertion order.

use pipnn_core::{build_index, BuildParams, Dataset, Metric, Reservoir};

fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x1234_5678);
    let mut v = vec![0.0f32; n * d];
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0;
    }
    v
}

#[test]
fn build_is_deterministic() {
    let (n, d) = (8000usize, 50usize);
    let flat = synth(n, d, 5);
    let data = Dataset::new(&flat, n, d, Metric::L2);
    let bp = BuildParams::default();

    let g1 = build_index(&data, &bp);
    let g2 = build_index(&data, &bp);

    assert_eq!(g1.indptr, g2.indptr, "indptr differs across builds");
    assert_eq!(g1.indices, g2.indices, "adjacency differs across builds");
    assert_eq!(g1.entry, g2.entry);
}

#[test]
fn reservoir_is_order_independent() {
    // Insert the same candidates in two different orders; the resulting sets
    // (id, hash) must match.
    let cands: Vec<(u32, u16, f32)> = vec![
        (10, 0b0001, 5.0),
        (11, 0b0010, 3.0),
        (12, 0b0001, 2.0), // same bucket as id 10, closer → should win bucket 0b0001
        (13, 0b0100, 8.0),
        (14, 0b1000, 1.0),
        (15, 0b0010, 9.0), // same bucket as 11, farther → should lose
    ];

    let mut a = Reservoir::new(4);
    for &(id, h, n) in &cands {
        a.insert(id, h, n);
    }
    let mut b = Reservoir::new(4);
    for &(id, h, n) in cands.iter().rev() {
        b.insert(id, h, n);
    }

    let sa = a.into_sorted();
    let sb = b.into_sorted();
    assert_eq!(sa, sb, "reservoir contents depend on insertion order");
    // The 4 closest distinct-bucket reps: 14@1, 12@2, 11@3, 13@8.
    let ids: Vec<u32> = sa.iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, vec![14, 12, 11, 13]);
}
