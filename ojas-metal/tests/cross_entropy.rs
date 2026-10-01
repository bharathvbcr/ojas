//! Cross-entropy forward and backward against the CPU reference at the
//! paired bench's vocabulary (50,304), with ignored and sharp rows, and its
//! error contract: a NaN or infinity anywhere in the logits, including a row
//! whose target is ignored, is refused.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};

const V: usize = 50304;

fn targets(rows: usize, seed: u64) -> Vec<u32> {
    ids(rows, seed, V as u32)
}

#[test]
fn matches_cpu_at_the_bench_vocabulary() {
    let (m, c) = (metal(), cpu());
    let rows = 384;
    for (scale, ignore) in [(4.0f32, None), (4.0, Some(7u32)), (80.0, None)] {
        let logits = rand(&[rows, V], 121 + scale as u64, scale);
        let mut t = targets(rows, 122);
        if let Some(ig) = ignore {
            for i in (0..rows).step_by(5) {
                t[i] = ig;
            }
        }
        let ht = host_u32(&t, &[rows]);
        let (dl, dt) = (up(&m, &logits), up(&m, &ht));
        let tag = format!("ce scale {scale} ignore {ignore:?}");
        same_tensor(
            &format!("{tag} loss"),
            &ok(&tag, m.cross_entropy_mean_forward(&dl, &dt, ignore)),
            &ok(&tag, c.cross_entropy_mean_forward(&logits, &ht, ignore)),
            1e-5,
            1e-5,
        );
        same_tensor(
            &format!("{tag} grad"),
            &ok(&tag, m.cross_entropy_mean_backward(&dl, &dt, ignore)),
            &ok(&tag, c.cross_entropy_mean_backward(&logits, &ht, ignore)),
            1e-9,
            1e-4,
        );
    }
}

#[test]
fn is_deterministic() {
    let m = metal();
    let rows = 64;
    let dl = up(&m, &rand(&[rows, V], 5, 4.0));
    let dt = up(&m, &host_u32(&targets(rows, 6), &[rows]));
    let bits = |t: &Tensor| -> Vec<u32> { down(t).iter().map(|v| v.to_bits()).collect() };
    let l0 = bits(&ok("fwd", m.cross_entropy_mean_forward(&dl, &dt, None)));
    let g0 = bits(&ok("bwd", m.cross_entropy_mean_backward(&dl, &dt, None)));
    for _ in 0..3 {
        assert_eq!(bits(&ok("fwd", m.cross_entropy_mean_forward(&dl, &dt, None))), l0);
        assert_eq!(bits(&ok("bwd", m.cross_entropy_mean_backward(&dl, &dt, None))), g0);
    }
}

#[test]
fn a_non_finite_logit_anywhere_is_refused_even_in_an_ignored_row() {
    let m = metal();
    let rows = 6;
    let ignore = 3u32;
    let mut t = targets(rows, 9);
    t[2] = ignore;
    let dt = up(&m, &host_u32(&t, &[rows]));
    // (row, column): inside a counted row, the last element of the last row,
    // and inside the ignored row 2.
    for (row, col) in [(0usize, 17usize), (rows - 1, V - 1), (2, 40000)] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut l = values(rows * V, 10, 4.0);
            l[row * V + col] = bad;
            let dl = up(&m, &host(&l, &[rows, V]));
            let f = m.cross_entropy_mean_forward(&dl, &dt, Some(ignore));
            assert!(
                matches!(f, Err(OjasError::NonFinite { op: "cross_entropy_mean_forward" })),
                "fwd ({row}, {col}) = {bad}: {f:?}"
            );
            let g = m.cross_entropy_mean_backward(&dl, &dt, Some(ignore));
            assert!(
                matches!(g, Err(OjasError::NonFinite { op: "cross_entropy_mean_backward" })),
                "bwd ({row}, {col}) = {bad}: {g:?}"
            );
        }
    }
}

#[test]
fn a_non_finite_logit_outranks_an_out_of_range_target() {
    let m = metal();
    let mut l = values(2 * V, 11, 1.0);
    l[V + 3] = f32::NAN;
    let dl = up(&m, &host(&l, &[2, V]));
    let dt = up(&m, &host_u32(&[1, V as u32 + 5], &[2]));
    let g = m.cross_entropy_mean_backward(&dl, &dt, None);
    assert!(matches!(g, Err(OjasError::NonFinite { .. })), "{g:?}");
    let finite = up(&m, &rand(&[2, V], 12, 1.0));
    let g = m.cross_entropy_mean_backward(&finite, &dt, None);
    assert!(matches!(g, Err(OjasError::OutOfRange { .. })), "{g:?}");
}
