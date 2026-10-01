//! RMSNorm and QK-norm forward and backward against the CPU reference at
//! the paired bench's shapes (`[4096, 768]`, and `[4, 1024, 12, 64]` = 49,152
//! rows of 64), at odd widths, and their error contract.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};

const EPS: f32 = 1e-6;
const PW_ATOL: f32 = 1e-6;

fn check(shape: &[usize], seed: u64) {
    let (m, c) = (metal(), cpu());
    let dim = *shape.last().expect("rank >= 1");
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let x = rand(shape, seed, 2.0);
    let w = rand(&[dim], seed + 1, 1.5);
    let g = rand(shape, seed + 2, 1.0);
    let (dx, dw, dg) = (up(&m, &x), up(&m, &w), up(&m, &g));
    let tag = format!("rms {shape:?}");
    let tol = PW_ATOL * dim as f32;
    same_tensor(
        &tag,
        &ok(&tag, m.rms_norm_forward(&dx, &dw, EPS)),
        &ok(&tag, c.rms_norm_forward(&x, &w, EPS)),
        tol,
        1e-4,
    );
    let (wx, ww) = ok(&tag, c.rms_norm_backward(&x, &w, &g, EPS));
    let (gx, gw) = ok(&tag, m.rms_norm_backward(&dx, &dw, &dg, EPS));
    same_tensor(&format!("{tag} dx"), &gx, &wx, tol, 1e-4);
    // dw sums `rows` terms; f32 sums in a different order differ by about
    // sqrt(rows) ulps of the running magnitude.
    same_tensor(
        &format!("{tag} dw"),
        &gw,
        &ww,
        PW_ATOL * (rows as f32).sqrt() * 8.0,
        1e-4,
    );
}

#[test]
fn rms_matches_cpu_at_bench_and_odd_shapes() {
    check(&[4096, 768], 1);
    check(&[4, 1024, 12, 64], 2);
    check(&[1000, 33], 3);
    check(&[3, 1], 4);
    check(&[65, 1], 5);
    check(&[7, 4099], 6);
}

#[test]
fn qk_norm_backward_matches_cpu_at_the_bench_shape() {
    let (m, c) = (metal(), cpu());
    let s = [4usize, 1024, 12, 64];
    let rows = 4 * 1024 * 12;
    let (q, k) = (rand(&s, 41, 1.0), rand(&s, 42, 1.0));
    let (qw, kw) = (rand(&[64], 43, 1.0), rand(&[64], 44, 1.0));
    let (gq, gk) = (rand(&s, 45, 1.0), rand(&s, 46, 1.0));
    let d: Vec<Tensor> = [&q, &k, &qw, &kw, &gq, &gk].iter().map(|t| up(&m, t)).collect();
    let want = ok("cpu", c.rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &gk, EPS));
    let got = ok(
        "metal",
        m.rms_qk_norm_backward(&d[0], &d[1], &d[2], &d[3], &d[4], &d[5], EPS),
    );
    let tol = PW_ATOL * 64.0;
    let wtol = PW_ATOL * (rows as f32).sqrt() * 8.0;
    same_tensor("dq", &got.0, &want.0, tol, 1e-4);
    same_tensor("dk", &got.1, &want.1, tol, 1e-4);
    same_tensor("dqw", &got.2, &want.2, wtol, 1e-4);
    same_tensor("dkw", &got.3, &want.3, wtol, 1e-4);
}

#[test]
fn norms_are_deterministic() {
    let m = metal();
    let s = [4096usize, 768];
    let (x, w, g) = (
        up(&m, &rand(&s, 1, 1.0)),
        up(&m, &rand(&[768], 2, 1.0)),
        up(&m, &rand(&s, 3, 1.0)),
    );
    let bits = |t: &Tensor| -> Vec<u32> { down(t).iter().map(|v| v.to_bits()).collect() };
    let y0 = bits(&ok("fwd", m.rms_norm_forward(&x, &w, EPS)));
    let (gx0, gw0) = ok("bwd", m.rms_norm_backward(&x, &w, &g, EPS));
    let (gx0, gw0) = (bits(&gx0), bits(&gw0));
    for _ in 0..3 {
        assert_eq!(bits(&ok("fwd", m.rms_norm_forward(&x, &w, EPS))), y0);
        let (gx, gw) = ok("bwd", m.rms_norm_backward(&x, &w, &g, EPS));
        assert_eq!(bits(&gx), gx0, "dx differs run to run");
        assert_eq!(bits(&gw), gw0, "dw differs run to run");
    }
}

#[test]
fn non_finite_inputs_and_overflowing_squares_are_refused() {
    let m = metal();
    let s = [300usize, 64];
    let n = 300 * 64;
    for which in 0..3 {
        let mut ins = [values(n, 1, 1.0), values(64, 2, 1.0), values(n, 3, 1.0)];
        let last = ins[which].len() - 1;
        ins[which][last] = f32::NAN;
        let x = up(&m, &host(&ins[0], &s));
        let w = up(&m, &host(&ins[1], &[64]));
        let g = up(&m, &host(&ins[2], &s));
        if which != 2 {
            let r = m.rms_norm_forward(&x, &w, EPS);
            assert!(
                matches!(r, Err(OjasError::NonFinite { op: "rms_norm_forward" })),
                "fwd input {which}: {r:?}"
            );
        }
        let r = m.rms_norm_backward(&x, &w, &g, EPS);
        assert!(
            matches!(r, Err(OjasError::NonFinite { op: "rms_norm_backward" })),
            "bwd input {which}: {r:?}"
        );
    }
    // A finite row whose sum of squares overflows f32.
    let mut xs = values(n, 4, 1.0);
    xs[n - 64..].iter_mut().for_each(|v| *v = 3.0e19);
    let (x, w, g) = (
        up(&m, &host(&xs, &s)),
        up(&m, &rand(&[64], 5, 1.0)),
        up(&m, &rand(&s, 6, 1.0)),
    );
    let r = m.rms_norm_forward(&x, &w, EPS);
    assert!(
        matches!(r, Err(OjasError::NonFinite { op: "rms_norm_forward" })),
        "fwd overflow: {r:?}"
    );
    let r = m.rms_norm_backward(&x, &w, &g, EPS);
    assert!(
        matches!(r, Err(OjasError::NonFinite { op: "rms_norm_backward" })),
        "bwd overflow: {r:?}"
    );
}
