//! The elementwise ops' non-finite checks, on both paths:
//! - **input**: a NaN or Inf in any input is reported as that op's fault;
//! - **output**: finite inputs whose result overflows are reported too. These
//!   are the only cases where an output check fires on its own; every
//!   NaN-in test also trips the output check, so it cannot tell whether the
//!   output check ran.
//!
//! Plus bit identity with the CPU reference for the ops whose result is one
//! IEEE add or multiply per element, which any correct kernel must match
//! exactly.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};
use ojas_metal::MetalBackend;

/// What `sync` reports: `None` for `Ok`, the op for `NonFinite`.
fn pending(m: &MetalBackend) -> Option<&'static str> {
    match m.sync() {
        Ok(()) => None,
        Err(OjasError::NonFinite { op }) => Some(op),
        Err(e) => panic!("sync: {e:?}"),
    }
}

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// `n` copies of `fill`, with `last` as the final element.
fn ends_with(n: usize, fill: f32, last: f32) -> Vec<f32> {
    let mut v = vec![fill; n];
    v[n - 1] = last;
    v
}

const N: usize = 1027;

#[test]
fn finite_inputs_whose_result_overflows_are_reported() {
    let m = metal();
    let one = up(&m, &host(&vec![1.0f32; N], &[N]));
    let big = up(&m, &host(&ends_with(N, 1.0, f32::MAX), &[N]));
    let four = up(&m, &host(&vec![4.0f32; N], &[N]));

    let y = ok("mul", m.mul_forward(&big, &four));
    assert_eq!(
        pending(&m),
        Some("mul_forward"),
        "MAX * 4 overflowed unreported"
    );
    drop(y);

    // grad_a = gy * b overflows; grad_b = gy * a does not.
    let g = ok("mul bwd a", m.mul_backward(&one, &big, &four));
    assert_eq!(
        pending(&m),
        Some("mul_backward"),
        "grad_a overflowed unreported"
    );
    drop(g);
    // grad_b = gy * a overflows; grad_a = gy * b does not.
    let g = ok("mul bwd b", m.mul_backward(&big, &one, &four));
    assert_eq!(
        pending(&m),
        Some("mul_backward"),
        "grad_b overflowed unreported"
    );
    drop(g);

    let y = ok("add", m.residual_add_forward(&big, &big));
    assert_eq!(
        pending(&m),
        Some("residual_add_forward"),
        "MAX + MAX overflowed unreported"
    );
    drop(y);

    // silu'(3) is about 1.088, so MAX * silu'(3) overflows.
    let three = up(&m, &host(&vec![3.0f32; N], &[N]));
    let g = ok("silu bwd", m.silu_backward(&three, &big));
    assert_eq!(
        pending(&m),
        Some("silu_backward"),
        "gy * silu'(x) overflowed unreported"
    );
    drop(g);

    // The same ops on values that stay finite report nothing.
    let y = ok("mul clean", m.mul_forward(&one, &four));
    let z = ok("add clean", m.residual_add_forward(&one, &four));
    let g = ok("silu bwd clean", m.silu_backward(&three, &four));
    let h = ok("mul bwd clean", m.mul_backward(&one, &four, &four));
    assert_eq!(pending(&m), None, "finite results reported a fault");
    drop((y, z, g, h));
}

#[test]
fn a_non_finite_input_at_the_end_of_an_offset_view_is_reported() {
    let m = metal();
    // A [3, N] buffer; the view is row 1, starting N floats in. N = 1027
    // keeps the start off any 16-byte boundary.
    let row = |last: f32| {
        let mut v = vec![0.5f32; 3 * N];
        v[2 * N - 1] = last;
        let base = up(&m, &host(&v, &[3 * N]));
        ok("view", base.view(&[N], &[1], N * 4))
    };
    let clean = up(&m, &host(&vec![0.5f32; N], &[N]));
    let lam = up(&m, &host(&[0.25], &[1]));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x = row(bad);
        // Each op runs and is synced on its own: a sync reports only the first
        // fault recorded since the last one.
        type Run<'a> = Box<dyn Fn() -> Result<Vec<Tensor>, OjasError> + 'a>;
        let cases: Vec<(&str, Run<'_>)> = vec![
            (
                "silu_forward",
                Box::new(|| m.silu_forward(&x).map(|t| vec![t])),
            ),
            (
                "silu_backward",
                Box::new(|| m.silu_backward(&clean, &x).map(|t| vec![t])),
            ),
            (
                "mul_forward",
                Box::new(|| m.mul_forward(&clean, &x).map(|t| vec![t])),
            ),
            (
                "mul_backward",
                Box::new(|| m.mul_backward(&x, &clean, &clean).map(|(a, b)| vec![a, b])),
            ),
            (
                "residual_add_forward",
                Box::new(|| m.residual_add_forward(&clean, &x).map(|t| vec![t])),
            ),
            (
                "residual_add_backward",
                Box::new(|| {
                    m.residual_add_backward(&x, &clean, &clean)
                        .map(|(a, b)| vec![a, b])
                }),
            ),
            (
                "value_residual_blend_forward",
                Box::new(|| {
                    m.value_residual_blend_forward(&clean, &x, &lam)
                        .map(|t| vec![t])
                }),
            ),
            (
                "value_residual_blend_backward",
                Box::new(|| {
                    m.value_residual_blend_backward(&x, &clean, &lam, &clean)
                        .map(|g| vec![g.value, g.value0, g.lambda])
                }),
            ),
        ];
        for (op, run) in cases {
            let kept = ok(op, run());
            assert_eq!(
                pending(&m),
                Some(op),
                "{op}: {bad} at the view's last element"
            );
            drop(kept);
        }
    }
}

#[test]
fn add_and_mul_match_the_cpu_reference_bit_for_bit() {
    let (m, c) = (metal(), cpu());
    let n = 4096 * 3 + 5;
    let (a, b, g) = (
        host(&values(n, 11, 8.0), &[n]),
        host(&values(n, 12, 8.0), &[n]),
        host(&values(n, 13, 8.0), &[n]),
    );
    let (da, db, dg) = (up(&m, &a), up(&m, &b), up(&m, &g));

    let want = ok("cpu add", c.residual_add_forward(&a, &b));
    let got = ok("metal add", m.residual_add_forward(&da, &db));
    assert_eq!(bits(&got), bits(&want), "residual_add_forward");

    let (wx, wy) = ok("cpu add bwd", c.residual_add_backward(&a, &b, &g));
    let (gx, gy) = ok("metal add bwd", m.residual_add_backward(&da, &db, &dg));
    assert_eq!(bits(&gx), bits(&wx), "residual_add_backward grad_x");
    assert_eq!(bits(&gy), bits(&wy), "residual_add_backward grad_y");

    let want = ok("cpu mul", c.mul_forward(&a, &b));
    let got = ok("metal mul", m.mul_forward(&da, &db));
    assert_eq!(bits(&got), bits(&want), "mul_forward");

    let (wa, wb) = ok("cpu mul bwd", c.mul_backward(&a, &b, &g));
    let (ga, gb) = ok("metal mul bwd", m.mul_backward(&da, &db, &dg));
    assert_eq!(bits(&ga), bits(&wa), "mul_backward grad_a");
    assert_eq!(bits(&gb), bits(&wb), "mul_backward grad_b");
    assert_eq!(pending(&m), None);
}
