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
            deferred(&m, &format!("fwd input {which}"), r, "rms_norm_forward");
        }
        let r = m.rms_norm_backward(&x, &w, &g, EPS);
        deferred(&m, &format!("bwd input {which}"), r, "rms_norm_backward");
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
    deferred(&m, "fwd overflow", r, "rms_norm_forward");
    let r = m.rms_norm_backward(&x, &w, &g, EPS);
    deferred(&m, "bwd overflow", r, "rms_norm_backward");
}

/// QK-norm on Metal computes the CPU reference's composition, `rms_norm_*`
/// on q and then on k: the results are those two calls' bits. Shape and
/// placement are checked for both pairs before anything is recorded, so k's
/// shape or placement refusal leaves nothing pending; a budget refusal of
/// k comes from the composition, which records q first, so a NaN in q is
/// then named by the next sync.
#[test]
fn qk_norm_is_the_composition_and_refuses_in_its_order() {
    use ojas_core::Budget;
    use ojas_metal::MetalBackend;

    let bits = |t: &Tensor| -> Vec<u32> { down(t).iter().map(|x| x.to_bits()).collect() };
    let (rows, dim) = (32usize, 16usize);
    let s = [4usize, 8, dim];
    let n = rows * dim;
    let make = |m: &MetalBackend, seed: u64, nan: bool| {
        let mut v = values(n, seed, 1.0);
        if nan {
            v[n - 1] = f32::NAN;
        }
        up(m, &host(&v, &s))
    };
    let m = metal();
    let (q, k, gq, gk) = (make(&m, 1, false), make(&m, 2, false), make(&m, 3, false), make(&m, 4, false));
    let (qw, kw) = (up(&m, &rand(&[dim], 5, 1.0)), up(&m, &rand(&[dim], 6, 1.0)));

    // Results: exactly the two single calls.
    let (fq, fk) = ok("qk fwd", m.rms_qk_norm_forward(&q, &k, &qw, &kw, EPS));
    assert_eq!(bits(&fq), bits(&ok("q", m.rms_norm_forward(&q, &qw, EPS))));
    assert_eq!(bits(&fk), bits(&ok("k", m.rms_norm_forward(&k, &kw, EPS))));
    let (bq, bk, bqw, bkw) = ok("qk bwd", m.rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &gk, EPS));
    let (sq, sqw) = ok("q bwd", m.rms_norm_backward(&q, &qw, &gq, EPS));
    let (sk, skw) = ok("k bwd", m.rms_norm_backward(&k, &kw, &gk, EPS));
    for (what, a, b) in [("gq", &bq, &sq), ("gk", &bk, &sk), ("gqw", &bqw, &sqw), ("gkw", &bkw, &skw)] {
        assert_eq!(bits(a), bits(b), "{what}");
    }

    // Order of refusals.
    let nan_q = make(&m, 1, true);
    let nan_k = make(&m, 2, true);
    let bad_kw = up(&m, &rand(&[dim + 1], 7, 1.0));
    let host_k = rand(&s, 8, 1.0);
    let fwd = |q: &Tensor, k: &Tensor, kw: &Tensor| m.rms_qk_norm_forward(q, k, &qw, kw, EPS).map(drop);
    let bwd = |q: &Tensor, k: &Tensor, kw: &Tensor| {
        m.rms_qk_norm_backward(q, k, &qw, kw, &gq, &gk, EPS).map(drop)
    };
    // A refused call records nothing (docs/shape-contract.md): both pairs
    // are validated, then both placed, before q is recorded, so neither k's
    // shape nor k's placement leaves q's NaN pending. A call that runs
    // reports q's NaN at the next sync.
    let pending = |b: &MetalBackend| match b.sync() {
        Ok(()) => None,
        Err(OjasError::NonFinite { op }) => Some(op),
        Err(e) => panic!("sync: {e:?}"),
    };
    const FWD: Option<&str> = Some("rms_norm_forward");
    const BWD: Option<&str> = Some("rms_norm_backward");
    type Call<'a> = Box<dyn Fn() -> Result<(), OjasError> + 'a>;
    let cases: Vec<(&str, Call<'_>, &str, Option<&str>)> = vec![
        ("fwd: q NaN", Box::new(|| fwd(&nan_q, &k, &kw)), "ok", FWD),
        ("fwd: k NaN", Box::new(|| fwd(&q, &nan_k, &kw)), "ok", FWD),
        ("fwd: q NaN, k shape", Box::new(|| fwd(&nan_q, &k, &bad_kw)), "shape", None),
        ("fwd: q NaN, k on host", Box::new(|| fwd(&nan_q, &host_k, &kw)), "placement", None),
        ("fwd: q fine, k shape", Box::new(|| fwd(&q, &k, &bad_kw)), "shape", None),
        ("bwd: q NaN", Box::new(|| bwd(&nan_q, &k, &kw)), "ok", BWD),
        ("bwd: k NaN", Box::new(|| bwd(&q, &nan_k, &kw)), "ok", BWD),
        ("bwd: q NaN, k shape", Box::new(|| bwd(&nan_q, &k, &bad_kw)), "shape", None),
        ("bwd: q NaN, k on host", Box::new(|| bwd(&nan_q, &host_k, &kw)), "placement", None),
        ("bwd: q fine, k on host", Box::new(|| bwd(&q, &host_k, &kw)), "placement", None),
    ];
    for (what, call, want, want_pending) in cases {
        let r = call();
        let fine = match want {
            "ok" => r.is_ok(),
            "shape" => matches!(r, Err(OjasError::Shape { .. })),
            _ => matches!(r, Err(OjasError::Placement { .. })),
        };
        assert!(fine, "{what}: {r:?}");
        assert_eq!(pending(&m), want_pending, "{what}: the next sync");
        assert_eq!(pending(&m), None, "{what}: reported once");
    }

    // Budgets. Inputs: q, k, gq, gk and the two weights.
    let f = 4u64;
    let inputs = f * (4 * n + 2 * dim) as u64;
    let out_fwd = f * n as u64;
    let out_bwd = f * (n + dim) as u64;
    let scratch_bwd = f * (rows + rows.div_ceil(64) * dim) as u64;
    let with_cap = |cap: u64, nan: bool| {
        let b = ok("metal", MetalBackend::new(Budget::new(cap)));
        let q = make(&b, 1, nan);
        let k = make(&b, 2, false);
        let (gq, gk) = (make(&b, 3, false), make(&b, 4, false));
        let (qw, kw) = (up(&b, &rand(&[dim], 5, 1.0)), up(&b, &rand(&[dim], 6, 1.0)));
        let f = b.rms_qk_norm_forward(&q, &k, &qw, &kw, EPS).map(drop);
        let f_pending = pending(&b);
        let g = b.rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &gk, EPS).map(drop);
        let g_pending = pending(&b);
        (f, f_pending, g, g_pending)
    };
    // Room for q's output only: k is refused for budget at the call; q's
    // NaN, recorded first, is what the next sync names.
    let (f, fp, _, _) = with_cap(inputs + out_fwd, false);
    assert!(matches!(f, Err(OjasError::CapacityExceeded { .. })), "{f:?}");
    assert_eq!(fp, None);
    let (f, fp, _, _) = with_cap(inputs + out_fwd, true);
    assert!(matches!(f, Err(OjasError::CapacityExceeded { .. })), "{f:?}");
    assert_eq!(fp, FWD);
    // The composition's peak is both outputs and one side's scratch; a call
    // that held both sides' scratch at once would not fit.
    let (f, fp, g, gp) = with_cap(inputs + 2 * out_bwd + scratch_bwd, false);
    ok("fwd at the composition's peak", f);
    ok("bwd at the composition's peak", g);
    assert_eq!((fp, gp), (None, None));
    let (_, _, g, gp) = with_cap(inputs + 2 * out_bwd + scratch_bwd, true);
    ok("bwd records", g);
    assert_eq!(gp, BWD);
}
