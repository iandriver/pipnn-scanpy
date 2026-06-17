//! Per-point reservoir for HashPrune (paper §3, Alg 3).
//!
//! Each slot is 8 bytes — `id: u32` + `hash: u16` + `norm: bf16` — matching the
//! paper's memory layout (8·ℓ_max·n total). The stored `norm` is the squared-L2
//! residual `‖c − p‖²` (monotone with distance; avoids a sqrt), in bf16.
//!
//! Insertion is **history-independent**: the final slot set depends only on the
//! candidate set, not the order they were inserted (paper's stated property).
//! That holds because (1) within a hash bucket we always keep the closest
//! representative, and (2) across buckets we keep the ℓ_max closest of those
//! representatives — both order-independent given distinct (norm, id) keys.

use half::bf16;

use crate::dataset::Id;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub id: Id,    // 4 bytes
    pub hash: u16, // 2 bytes
    pub norm: u16, // 2 bytes: bf16 bit pattern of ‖c − p‖²
}

const _: () = assert!(std::mem::size_of::<Slot>() == 8);

impl Slot {
    #[inline]
    fn norm_f32(&self) -> f32 {
        bf16::from_bits(self.norm).to_f32()
    }
}

pub struct Reservoir {
    slots: Vec<Slot>,
    l_max: usize,
}

impl Reservoir {
    pub fn new(l_max: usize) -> Self {
        Reservoir {
            slots: Vec::with_capacity(l_max),
            l_max,
        }
    }

    #[inline]
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Insert candidate `id` with LSH code `hash` and residual squared-norm
    /// `norm_sq` (= ‖c − p‖²). Self-edges (norm 0) should not be inserted.
    pub fn insert(&mut self, id: Id, hash: u16, norm_sq: f32) {
        let norm_b = bf16::from_f32(norm_sq);
        let norm_q = norm_b.to_f32(); // quantized value used for all comparisons

        // (1) Same-hash bucket already present → keep the closer representative.
        for s in self.slots.iter_mut() {
            if s.hash == hash {
                let cur = s.norm_f32();
                // Deterministic tie-break by id keeps the result order-independent.
                if norm_q < cur || (norm_q == cur && id < s.id) {
                    s.id = id;
                    s.norm = norm_b.to_bits();
                }
                return;
            }
        }

        // (2) Room available → just add.
        if self.slots.len() < self.l_max {
            self.slots.push(Slot {
                id,
                hash,
                norm: norm_b.to_bits(),
            });
            return;
        }

        // (3) Full and new bucket → evict the farthest if the candidate is closer.
        let mut worst = 0usize;
        let mut worst_norm = self.slots[0].norm_f32();
        let mut worst_id = self.slots[0].id;
        for (k, s) in self.slots.iter().enumerate().skip(1) {
            let nf = s.norm_f32();
            if nf > worst_norm || (nf == worst_norm && s.id > worst_id) {
                worst = k;
                worst_norm = nf;
                worst_id = s.id;
            }
        }
        if norm_q < worst_norm || (norm_q == worst_norm && id < worst_id) {
            self.slots[worst] = Slot {
                id,
                hash,
                norm: norm_b.to_bits(),
            };
        }
    }

    /// Drain the reservoir into `(id, norm_sq)` pairs sorted by ascending norm.
    pub fn into_sorted(self) -> Vec<(Id, f32)> {
        let mut v: Vec<(Id, f32)> = self
            .slots
            .into_iter()
            .map(|s| (s.id, s.norm_f32()))
            .collect();
        v.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then_with(|| a.0.cmp(&b.0))
        });
        v
    }
}
