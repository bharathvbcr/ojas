//! `Backend::linear_cross_entropy_mean` on Metal (T3, gate G3): the mean
//! cross-entropy of `input @ weight^T` and both gradients, with at most
//! `chunk.rows * chunk.cols` logits alive at once.
//!
//! Tolerances are normwise: `max |got - want| <= tol * max |want|` per
//! tensor (and `|got - want| <= tol * |want|` for the scalar loss).
//!
//! Against the unfused composition (G3a) the loss is within 1e-6 (in
//! practice identical), and so are the gradients when the chunk does not
//! split the problem. A chunk that splits the vocabulary or the rows sums
//! each gradient tile by tile where the composition makes one GEMM over the
//! whole axis; both are f32 roundings of the same sum, and a softmax
//! gradient cancels, so they differ by a few 1e-6 of the gradient's largest
//! entry, about what the composition itself is off from an f64 evaluation.
//! There the gate is that the fused gradients are at least as accurate
//! against f64 as the composition's, and within 1e-5 of them.
#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, Budget, CeChunk, LinearCe, OjasError, Tensor};
use ojas_metal::MetalBackend;

const OP: &str = "linear_cross_entropy_mean";

struct Problem {
    x: Tensor,
    w: Tensor,
    t: Tensor,
    ignore: Option<u32>,
}

/// `n` rows of width `d` over a vocabulary of `v`. With `ignore`, the
/// ignore index is vocabulary id 1, a valid id, so only the ignore rule
/// (not the range check) can skip those rows; every fifth target is set to
/// it.
fn problem(n: usize, d: usize, v: usize, seed: u64, ignore: bool) -> Problem {
    let mut ids = ids(n, seed + 2, v as u32);
    let ignore = ignore.then_some(1u32.min(v as u32 - 1));
    if let Some(ig) = ignore {
        for i in (0..n).step_by(5) {
            ids[i] = ig;
        }
    }
    Problem {
        x: rand(&[n, d], seed, 1.0),
        w: rand(&[v, d], seed + 1, 0.5),
        t: host_u32(&ids, &[n]),
        ignore,
    }
}

struct Grads {
    loss: f32,
    gx: Vec<f32>,
    gw: Vec<f32>,
}

fn scalar(t: &Tensor) -> f32 {
    let v = down(t);
    assert_eq!(v.len(), 1, "loss is one value");
    v[0]
}

/// linear_forward -> cross_entropy_mean_forward/backward -> linear_backward.
fn composed<B: Backend>(be: &B, p: &Problem, place: impl Fn(&Tensor) -> Tensor) -> Grads {
    let (x, w, t) = (place(&p.x), place(&p.w), place(&p.t));
    let logits = ok("linear", be.linear_forward(&x, &w));
    let loss = ok("ce fwd", be.cross_entropy_mean_forward(&logits, &t, p.ignore));
    let g = ok("ce bwd", be.cross_entropy_mean_backward(&logits, &t, p.ignore));
    drop(logits);
    let (gx, gw) = ok("linear bwd", be.linear_backward(&x, &w, &g));
    let host = |t: &Tensor| ok("to host", t.to_host(&Budget::new(8 * GIB)));
    Grads {
        loss: ok("loss", host(&loss).to_f32_vec())[0],
        gx: ok("gx", host(&gx).to_f32_vec()),
        gw: ok("gw", host(&gw).to_f32_vec()),
    }
}

/// The loss and both gradients in f64, from the definition.
fn exact(p: &Problem) -> (f64, Vec<f64>, Vec<f64>) {
    let x = ok("x", p.x.to_f32_vec());
    let w = ok("w", p.w.to_f32_vec());
    let t = ok("t", p.t.to_u32_vec());
    let (n, d) = (p.x.shape()[0], p.x.shape()[1]);
    let v = p.w.shape()[0];
    let valid: Vec<bool> = t.iter().map(|&id| Some(id) != p.ignore && (id as usize) < v).collect();
    let count = valid.iter().filter(|&&b| b).count() as f64;
    let (mut loss, mut gx, mut gw) = (0.0f64, vec![0.0f64; n * d], vec![0.0f64; v * d]);
    for r in 0..n {
        if !valid[r] {
            continue;
        }
        let logits: Vec<f64> = (0..v)
            .map(|c| (0..d).map(|k| f64::from(x[r * d + k]) * f64::from(w[c * d + k])).sum())
            .collect();
        let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = logits.iter().map(|l| (l - mx).exp()).sum();
        loss += mx + total.ln() - logits[t[r] as usize];
        for c in 0..v {
            let mut g = (logits[c] - mx).exp() / total / count;
            if c == t[r] as usize {
                g -= 1.0 / count;
            }
            for k in 0..d {
                gx[r * d + k] += g * f64::from(w[c * d + k]);
                gw[c * d + k] += g * f64::from(x[r * d + k]);
            }
        }
    }
    (loss / count, gx, gw)
}

/// `max |got - want| / max |want|` against an f64 reference.
fn err64(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let worst = got
        .iter()
        .zip(want)
        .fold(0.0f64, |a, (g, w)| a.max((f64::from(*g) - w).abs()));
    worst / scale
}

fn fused(m: &MetalBackend, p: &Problem, chunk: CeChunk) -> Grads {
    let out = ok(
        "fused",
        m.linear_cross_entropy_mean(&up(m, &p.x), &up(m, &p.w), &up(m, &p.t), p.ignore, chunk, true),
    );
    let LinearCe {
        loss,
        grad_input,
        grad_weight,
    } = out;
    let (Some(gx), Some(gw)) = (grad_input, grad_weight) else {
        panic!("want_grad set but a gradient is missing");
    };
    assert_eq!(gx.shape(), p.x.shape(), "grad_input shape");
    assert_eq!(gw.shape(), p.w.shape(), "grad_weight shape");
    Grads {
        loss: scalar(&loss),
        gx: down(&gx),
        gw: down(&gw),
    }
}

fn normwise(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0.0f32, |a, x| a.max(x.abs()));
    let (mut worst, mut at) = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite at {i}");
        let e = (g - w).abs();
        if e > worst {
            (worst, at) = (e, i);
        }
    }
    assert!(
        worst <= tol * scale,
        "{what}: max error {worst:e} at {at} (got {} want {}) > {tol:e} * {scale:e}",
        got[at],
        want[at]
    );
}

/// G3a: the loss within 1e-6; the gradients bit-equal to the composition's
/// when the chunk covers the problem, else no less accurate against f64 and
/// within 1e-5 normwise.
fn agrees_with_composition(what: &str, p: &Problem, c: CeChunk, got: &Grads, comp: &Grads) {
    let (n, v) = (p.x.shape()[0], p.w.shape()[0]);
    assert!(
        (got.loss - comp.loss).abs() <= 1e-6 * comp.loss.abs(),
        "{what}: loss {} vs {}",
        got.loss,
        comp.loss
    );
    if c.rows >= n && c.cols >= v {
        assert_eq!(got.gx, comp.gx, "{what}: untiled grad_input");
        assert_eq!(got.gw, comp.gw, "{what}: untiled grad_weight");
        return;
    }
    let (_, gx64, gw64) = exact(p);
    for (name, g, cm, e) in [("grad_input", &got.gx, &comp.gx, &gx64), ("grad_weight", &got.gw, &comp.gw, &gw64)] {
        let (eg, ec) = (err64(g, e), err64(cm, e));
        assert!(
            eg <= 1.25 * ec + 1e-7,
            "{what} {name}: fused is {eg:.2e} from f64, the composition {ec:.2e}"
        );
        normwise(&format!("{what} {name}"), g, cm, 1e-5);
    }
}

fn same_grads(what: &str, got: &Grads, want: &Grads, tol: f32) {
    assert!(
        (got.loss - want.loss).abs() <= tol * want.loss.abs(),
        "{what}: loss {} vs {}",
        got.loss,
        want.loss
    );
    normwise(&format!("{what} grad_input"), &got.gx, &want.gx, tol);
    normwise(&format!("{what} grad_weight"), &got.gw, &want.gw, tol);
}

const fn chunk(rows: usize, cols: usize) -> CeChunk {
    CeChunk { rows, cols }
}

#[test]
fn equals_the_unfused_composition_on_metal() {
    let m = metal();
    let cases = [
        // (n, d, v, chunks)
        (64usize, 32usize, 300usize, vec![chunk(16, 128), chunk(64, 300), chunk(7, 100)]),
        (100, 33, 257, vec![chunk(7, 100), chunk(1, 257), chunk(100, 1), chunk(3, 64)]),
        (37, 64, 1000, vec![chunk(37, 999), chunk(36, 1000), chunk(usize::MAX, usize::MAX)]),
        (5, 8, 11, vec![chunk(1, 1), chunk(2, 3), chunk(1000, 1000)]),
    ];
    for (i, (n, d, v, chunks)) in cases.into_iter().enumerate() {
        for ignore in [false, true] {
            let p = problem(n, d, v, 10 * i as u64, ignore);
            let want = composed(&m, &p, |t| up(&m, t));
            for c in chunks.iter().copied() {
                let got = fused(&m, &p, c);
                let what = format!("n{n} d{d} v{v} {c:?} ignore {ignore}");
                agrees_with_composition(&what, &p, c, &got, &want);
            }
        }
    }
}

#[test]
fn without_gradients_returns_only_the_loss() {
    let m = metal();
    let p = problem(50, 16, 200, 3, true);
    let want = composed(&m, &p, |t| up(&m, t));
    let out = ok(
        "fused",
        m.linear_cross_entropy_mean(
            &up(&m, &p.x),
            &up(&m, &p.w),
            &up(&m, &p.t),
            p.ignore,
            chunk(8, 64),
            false,
        ),
    );
    assert!(out.grad_input.is_none() && out.grad_weight.is_none());
    let loss = scalar(&out.loss);
    assert!((loss - want.loss).abs() <= 1e-6 * want.loss.abs(), "{loss} vs {}", want.loss);
}

#[test]
fn runs_under_a_budget_smaller_than_the_logits() {
    let (n, d, v) = (512usize, 64usize, 4096usize);
    let logits_bytes = (n * v * 4) as u64;
    let cap = logits_bytes / 2;
    let small = ok("metal", MetalBackend::new(Budget::new(cap)));
    let p = problem(n, d, v, 41, true);
    let (x, w, t) = (up(&small, &p.x), up(&small, &p.w), up(&small, &p.t));
    // The composition materializes [N, V] logits and is refused.
    let r = small.linear_forward(&x, &w);
    assert!(matches!(r, Err(OjasError::CapacityExceeded { .. })), "{r:?}");
    // A chunk as large as the problem is charged as such and refused too.
    let r = small.linear_cross_entropy_mean(&x, &w, &t, p.ignore, chunk(n, v), true);
    assert!(matches!(r, Err(OjasError::CapacityExceeded { .. })), "{r:?}");
    let live = ok("live", small.budget().live_bytes());
    let out = ok(
        "fused",
        small.linear_cross_entropy_mean(&x, &w, &t, p.ignore, chunk(64, 512), true),
    );
    let got = Grads {
        loss: scalar(&out.loss),
        gx: down(out.grad_input.as_ref().expect("grad_input")),
        gw: down(out.grad_weight.as_ref().expect("grad_weight")),
    };
    drop(out);
    assert_eq!(ok("live", small.budget().live_bytes()), live, "scratch leaked");
    let big = metal();
    let want = composed(&big, &p, |t| up(&big, t));
    agrees_with_composition("small budget", &p, chunk(64, 512), &got, &want);
}

#[test]
fn matches_the_cpu_reference_at_the_bench_vocabulary() {
    let (n, d, v) = (4096usize, 768usize, 50304usize);
    let m = metal();
    let c = cpu();
    let p = problem(n, d, v, 77, true);
    let got = fused(&m, &p, chunk(1024, 8192));
    let want = composed(&c, &p, |t| t.clone());
    same_grads("bench shape vs cpu", &got, &want, 1e-4);
}

#[test]
fn is_deterministic() {
    let m = metal();
    let p = problem(300, 64, 2000, 5, true);
    let bits = |g: &Grads| {
        let mut b: Vec<u32> = vec![g.loss.to_bits()];
        b.extend(g.gx.iter().map(|x| x.to_bits()));
        b.extend(g.gw.iter().map(|x| x.to_bits()));
        b
    };
    let first = bits(&fused(&m, &p, chunk(64, 512)));
    for _ in 0..3 {
        assert_eq!(bits(&fused(&m, &p, chunk(64, 512))), first);
    }
}

#[test]
fn refusals() {
    let m = metal();
    let (n, d, v) = (6usize, 8usize, 20usize);
    let p = problem(n, d, v, 9, false);
    let (x, w, t) = (up(&m, &p.x), up(&m, &p.w), up(&m, &p.t));
    let c = chunk(4, 8);
    let call = |x: &Tensor, w: &Tensor, t: &Tensor, ig: Option<u32>, c: CeChunk| {
        m.linear_cross_entropy_mean(x, w, t, ig, c, true)
    };
    let nonfinite = |r: Result<LinearCe, OjasError>, what: &str| {
        assert!(matches!(r, Err(OjasError::NonFinite { op: OP })), "{what}: {r:?}");
        assert!(m.sync().is_ok(), "{what}: a host refusal records nothing");
    };
    let out_of_range = |r: Result<LinearCe, OjasError>, what: &str| {
        assert!(matches!(r, Err(OjasError::OutOfRange { op: OP, .. })), "{what}: {r:?}");
        assert!(m.sync().is_ok(), "{what}: a host refusal records nothing");
    };
    // Every target ignored: decided on the host, at the call.
    let all = up(&m, &host_u32(&vec![7; n], &[n]));
    nonfinite(call(&x, &w, &all, Some(7), c), "all ignored");
    // A target out of range.
    let mut bad_ids = ok("ids", p.t.to_u32_vec());
    bad_ids[n - 1] = v as u32;
    let bad_t = up(&m, &host_u32(&bad_ids, &[n]));
    out_of_range(call(&x, &w, &bad_t, None, c), "bad target");
    // Non-finite inputs, at the last element: found on the device and named
    // by the next sync. The targets are checked on the host first, so a bad
    // target is refused at the call even beside a non-finite input.
    let poison = |src: &Tensor, val: f32| {
        let mut data = ok("data", src.to_f32_vec());
        let last = data.len() - 1;
        data[last] = val;
        up(&m, &host(&data, src.shape()))
    };
    for val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        deferred(&m, "x", call(&poison(&p.x, val), &w, &t, None, c), OP);
        deferred(&m, "w", call(&x, &poison(&p.w, val), &t, None, c), OP);
        out_of_range(call(&x, &poison(&p.w, val), &bad_t, None, c), "w with a bad target");
    }
    // Logits that overflow f32: a device fault with valid targets, and a
    // bad target refused at the call.
    let huge = |shape: &[usize]| up(&m, &host(&vec![1e20; shape.iter().product()], shape));
    deferred(&m, "overflow", call(&huge(&[n, d]), &huge(&[v, d]), &t, None, c), OP);
    out_of_range(call(&huge(&[n, d]), &huge(&[v, d]), &bad_t, None, c), "overflow with a bad target");
    // Shapes and placement.
    for bad in [chunk(0, 4), chunk(4, 0)] {
        let r = call(&x, &w, &t, None, bad);
        assert!(matches!(r, Err(OjasError::Shape { .. })), "{bad:?}: {r:?}");
    }
    let r = call(&x, &up(&m, &rand(&[v, d + 1], 1, 1.0)), &t, None, c);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    let r = call(&p.x, &w, &t, None, c);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    let r = call(&x, &w, &x, None, c);
    assert!(matches!(r, Err(OjasError::Dtype { .. })), "{r:?}");
    // Nothing above leaked a reservation.
    drop((x, w, t, all, bad_t));
    assert_eq!(ok("live", m.budget().live_bytes()), 0);
}
