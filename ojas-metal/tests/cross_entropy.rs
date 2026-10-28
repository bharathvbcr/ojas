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
        assert_eq!(
            bits(&ok("fwd", m.cross_entropy_mean_forward(&dl, &dt, None))),
            l0
        );
        assert_eq!(
            bits(&ok("bwd", m.cross_entropy_mean_backward(&dl, &dt, None))),
            g0
        );
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
            deferred(
                &m,
                &format!("fwd ({row}, {col}) = {bad}"),
                f,
                "cross_entropy_mean_forward",
            );
            let g = m.cross_entropy_mean_backward(&dl, &dt, Some(ignore));
            deferred(
                &m,
                &format!("bwd ({row}, {col}) = {bad}"),
                g,
                "cross_entropy_mean_backward",
            );
        }
    }
}

/// The targets are checked on the host, so an out-of-range target is
/// refused at the call even when a logit is non-finite (the precedence the
/// deferred-fault contract flips); the call records nothing, so nothing is
/// pending afterwards.
#[test]
fn an_out_of_range_target_is_refused_at_once_even_beside_a_non_finite_logit() {
    let m = metal();
    let mut l = values(2 * V, 11, 1.0);
    l[V + 3] = f32::NAN;
    let dl = up(&m, &host(&l, &[2, V]));
    let dt = up(&m, &host_u32(&[1, V as u32 + 5], &[2]));
    let g = m.cross_entropy_mean_backward(&dl, &dt, None);
    assert!(matches!(g, Err(OjasError::OutOfRange { .. })), "{g:?}");
    assert!(m.sync().is_ok(), "the refused call must record nothing");
    let finite = up(&m, &rand(&[2, V], 12, 1.0));
    let g = m.cross_entropy_mean_backward(&finite, &dt, None);
    assert!(matches!(g, Err(OjasError::OutOfRange { .. })), "{g:?}");
}

/// `|got - want|` in units of the f32 spacing at `want`.
fn ulps(got: f32, want: f64) -> f64 {
    let w = want as f32;
    let spacing = if w == 0.0 || !w.is_normal() {
        f64::from(f32::from_bits(1))
    } else {
        let next = f32::from_bits(w.abs().to_bits() + 1);
        f64::from(next) - f64::from(w.abs())
    };
    (f64::from(got) - want).abs() / spacing
}

/// Paper shape `[9, -9, -9]`, target 0, margin 18. The f64 oracle does not
/// recompute `p - 1`. A second ignored row keeps the mean divisor at 1.
#[test]
fn a_row_past_the_fp32_cliff_keeps_a_nonzero_target_gradient() {
    let row = [9.0f64, -9.0, -9.0];
    let max_o = row[1];
    let sum_o = 2.0 * (row[1] - max_o).exp();
    let delta = max_o - row[0];
    let q = sum_o * delta.exp();
    let want_loss = q.ln_1p();
    let scale = delta.exp() / (1.0 + q);
    let want = [
        -sum_o * scale,
        (row[1] - max_o).exp() * scale,
        (row[2] - max_o).exp() * scale,
    ];

    let m = metal();
    let logits = host(&[9.0, -9.0, -9.0, 1.0, 2.0, 3.0], &[2, 3]);
    let targets = host_u32(&[0, 99], &[2]);
    let dl = up(&m, &logits);
    let dt = up(&m, &targets);
    let loss = down(&ok(
        "cliff forward",
        m.cross_entropy_mean_forward(&dl, &dt, Some(99)),
    ));
    let grad = down(&ok(
        "cliff backward",
        m.cross_entropy_mean_backward(&dl, &dt, Some(99)),
    ));
    assert!(grad[0] != 0.0, "target gradient collapsed: {grad:?}");
    let loss_ulps = ulps(loss[0], want_loss);
    assert!(
        loss_ulps <= 16.0,
        "loss {} vs {want_loss} ({loss_ulps} ulps)",
        loss[0]
    );
    for (j, (&got, &want)) in grad[..3].iter().zip(&want).enumerate() {
        let u = ulps(got, want);
        assert!(u <= 16.0, "grad[{j}] {got} vs {want} ({u} ulps)");
    }
    let sum: f64 = grad[..3].iter().map(|g| f64::from(*g)).sum();
    let peak = grad[..3].iter().fold(0.0f32, |m, g| m.max(g.abs()));
    let next = f32::from_bits(peak.to_bits() + 1);
    let spacing = f64::from(next) - f64::from(peak);
    assert!(
        sum.abs() <= 8.0 * spacing,
        "row sum {sum} of {grad:?} is more than 8 ulps of the peak {peak}"
    );
    assert!(grad[3..].iter().all(|g| *g == 0.0), "ignored row {grad:?}");
}
