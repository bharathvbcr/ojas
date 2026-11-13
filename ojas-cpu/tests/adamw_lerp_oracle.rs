//! ft-9c13f472 D5: the AdamW first moment `m = lerp(m, g, 1 - beta1)` on the
//! CPU, against bits an independent reference computed.
//!
//! The expected bits come from `backups/6a/sem-lane/lerp_oracle.py` (numpy
//! float32 and float64 forms of ATen's scalar lerp, `Lerp.h:24-34`; torch
//! 2.12.1's `torch.lerp` agrees on every row below, and differs on one other
//! row only by the fused multiply-add of its vectorized CPU kernel):
//!
//! - Fast, f32: `w = f32(1 - beta1)`; `m + w * (g - m)` if `w < 0.5`, else
//!   `g - (g - m) * (1f - w)`, every operation rounded to f32.
//! - Exact, f64: the same with `w = 1 - beta1` in f64, rounded once to f32.
//!
//! Row: `(beta1, g bits, m bits, Fast bits, Exact bits)`. beta1 below 0.5 is
//! where the high-branch coefficient `1 - w` differs from `beta1`.

use ojas_core::{AdamWConfig, BackendModule, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

const ROWS: [(f64, u32, u32, u32, u32); 36] = [
    (0.0, 0x3e99999a, 0xbf333333, 0x3e99999a, 0x3e99999a),
    (0.0, 0x3f9e064b, 0x39a84be4, 0x3f9e064b, 0x3f9e064b),
    (0.0, 0xbd4ccccd, 0x3f666666, 0xbd4ccccd, 0xbd4ccccd),
    (0.0, 0x3b23d70a, 0xbaa3d70a, 0x3b23d70a, 0x3b23d70a),
    (0.001, 0x3e99999a, 0xbf333333, 0x3e991688, 0x3e991688),
    (0.001, 0x3f9e064b, 0x39a84be4, 0x3f9dddda, 0x3f9dddd9),
    (0.001, 0xbd4ccccd, 0x3f666666, 0xbd48e8ab, 0xbd48e8a7),
    (0.001, 0x3b23d70a, 0xbaa3d70a, 0x3b239820, 0x3b239820),
    (0.3, 0x3e99999a, 0xbf333333, 0x00000000, 0x324ccccd),
    (0.3, 0x3f9e064b, 0x39a84be4, 0x3f5d4252, 0x3f5d4252),
    (0.3, 0xbd4ccccd, 0x3f666666, 0x3e70a3d7, 0x3e70a3d7),
    (0.3, 0x3b23d70a, 0xbaa3d70a, 0x3ab43957, 0x3ab43958),
    (0.499, 0x3e99999a, 0xbf333333, 0xbe4bc6a8, 0xbe4bc6a7),
    (0.499, 0x3f9e064b, 0x39a84be4, 0x3f1e61b3, 0x3f1e61b3),
    (0.499, 0xbd4ccccd, 0x3f666666, 0x3ed91d15, 0x3ed91d14),
    (0.499, 0x3b23d70a, 0xbaa3d70a, 0x3a24d2b2, 0x3a24d2b3),
    (0.5, 0x3e99999a, 0xbf333333, 0xbe4ccccc, 0xbe4ccccc),
    (0.5, 0x3f9e064b, 0x39a84be4, 0x3f1e10d0, 0x3f1e10d0),
    (0.5, 0xbd4ccccd, 0x3f666666, 0x3ed99999, 0x3ed99999),
    (0.5, 0x3b23d70a, 0xbaa3d70a, 0x3a23d70a, 0x3a23d70a),
    (
        0.5000000009313226,
        0x3e99999a,
        0xbf333333,
        0xbe4ccccc,
        0xbe4ccccc,
    ),
    (
        0.5000000009313226,
        0x3f9e064b,
        0x39a84be4,
        0x3f1e10d0,
        0x3f1e10d0,
    ),
    (
        0.5000000009313226,
        0xbd4ccccd,
        0x3f666666,
        0x3ed99999,
        0x3ed99999,
    ),
    (
        0.5000000009313226,
        0x3b23d70a,
        0xbaa3d70a,
        0x3a23d70a,
        0x3a23d70a,
    ),
    (
        0.4999999990686774,
        0x3e99999a,
        0xbf333333,
        0xbe4ccccc,
        0xbe4ccccc,
    ),
    (
        0.4999999990686774,
        0x3f9e064b,
        0x39a84be4,
        0x3f1e10d0,
        0x3f1e10d0,
    ),
    (
        0.4999999990686774,
        0xbd4ccccd,
        0x3f666666,
        0x3ed99999,
        0x3ed99999,
    ),
    (
        0.4999999990686774,
        0x3b23d70a,
        0xbaa3d70a,
        0x3a23d70a,
        0x3a23d70a,
    ),
    (0.9, 0x3e99999a, 0xbf333333, 0xbf199999, 0xbf199999),
    (0.9, 0x3f9e064b, 0x39a84be4, 0x3dfd6e56, 0x3dfd6e56),
    (0.9, 0xbd4ccccd, 0x3f666666, 0x3f4e147a, 0x3f4e147b),
    (0.9, 0x3b23d70a, 0xbaa3d70a, 0xba656041, 0xba656041),
    (0.999, 0x3e99999a, 0xbf333333, 0xbf32f1aa, 0xbf32f1aa),
    (0.999, 0x3f9e064b, 0x39a84be4, 0x3acbd966, 0x3acbd966),
    (0.999, 0xbd4ccccd, 0x3f666666, 0x3f662824, 0x3f662824),
    (0.999, 0x3b23d70a, 0xbaa3d70a, 0xbaa35936, 0xbaa35936),
];

/// The new first moment of a one-element step.
fn moment(numerics: Numerics, beta1: f64, g: u32, m: u32) -> u32 {
    let budget = Budget::new(1 << 20);
    let backend = CpuBackend::new(budget.clone()).with_numerics(numerics);
    let one = |v: f32| Tensor::from_f32(&[v], &[1], &budget).unwrap();
    let cfg = AdamWConfig {
        beta1,
        ..AdamWConfig::nanolab(1e-3, 0.0)
    };
    let (mut p, grad) = (one(0.5), one(f32::from_bits(g)));
    let (mut m1, mut m2) = (one(f32::from_bits(m)), one(0.0));
    backend
        .adamw_step(&mut p, &grad, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    m1.to_f32_vec().unwrap()[0].to_bits()
}

#[test]
fn fast_first_moment_equals_the_independent_f32_oracle_bits() {
    for (beta1, g, m, fast, _) in ROWS {
        assert_eq!(
            moment(Numerics::Fast, beta1, g, m),
            fast,
            "beta1 {beta1} g {g:#010x} m {m:#010x}"
        );
    }
}

#[test]
fn exact_first_moment_equals_the_independent_f64_oracle_bits() {
    for (beta1, g, m, _, exact) in ROWS {
        assert_eq!(
            moment(Numerics::Exact, beta1, g, m),
            exact,
            "beta1 {beta1} g {g:#010x} m {m:#010x}"
        );
    }
}
