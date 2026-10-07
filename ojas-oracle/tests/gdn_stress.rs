//! Adversarial and stress checks of the CPU gated delta rule
//! (`Backend::chunked_gdn_forward` / `_backward`), beyond the goldens in
//! `gdn.rs`:
//!
//! - a sweep of shapes, including every extent of 1 and the lengths either
//!   side of each 64-token checkpoint, against the f64 forward;
//! - central differences at the edge shapes;
//! - inputs that break naive arithmetic: rows of norm 1e30 (whose f32 sum
//!   of squares overflows), all-zero rows, a decay that forgets everything,
//!   no decay, `beta = 0`;
//! - the bits do not depend on numerics, thread count or concurrent callers;
//! - the budget is exact: no charge outlives the call, a cap one byte under
//!   the op's need refuses, and a cancel mid-op leaves nothing charged.
//!
//! The long-sequence check is `#[ignore]`: run it with
//! `cargo test --release -p ojas-oracle --test gdn_stress -- --ignored`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_core::{Backend, Budget, GdnInputs, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_oracle::gdn::{forward_f64, Inputs, Shape};

/// `max |cpu - f64| / max |f64|` per tensor.
const BOUND: f64 = 1e-5;

fn exact(cap: u64) -> CpuBackend {
    CpuBackend::new(Budget::new(cap)).with_numerics(Numerics::Exact)
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n).map(|_| lo + (hi - lo) * self.unit()).collect()
    }

    fn pick(&mut self, xs: &[usize]) -> usize {
        xs[((self.unit() * xs.len() as f64) as usize).min(xs.len() - 1)]
    }
}

#[derive(Clone)]
struct P {
    s: Shape,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    s0: Vec<f64>,
}

impl P {
    fn random(s: Shape, seed: u64) -> Self {
        let mut r = Lcg(seed);
        let n = s.rows();
        Self {
            s,
            q: r.vec(n * s.dk, -1.0, 1.0),
            k: r.vec(n * s.dk, -1.0, 1.0),
            v: r.vec(n * s.dv, -1.0, 1.0),
            g: r.vec(n, -1.0, -0.05),
            beta: r.vec(n, 0.1, 0.9),
            s0: r.vec(s.state_len(), -0.5, 0.5),
        }
    }

    fn oracle(&self, s0: bool) -> Inputs<'_> {
        Inputs {
            s: self.s,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: s0.then_some(self.s0.as_slice()),
        }
    }
}

/// Host tensors of `p`, charged to `b`.
struct T4 {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Tensor,
}

impl T4 {
    fn new(b: &CpuBackend, p: &P) -> Self {
        let Shape {
            b: nb,
            t,
            h,
            dk,
            dv,
        } = p.s;
        let f = |x: &[f64], shape: &[usize]| {
            let y: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            Tensor::from_f32(&y, shape, b.budget()).unwrap()
        };
        Self {
            q: f(&p.q, &[nb, t, h, dk]),
            k: f(&p.k, &[nb, t, h, dk]),
            v: f(&p.v, &[nb, t, h, dv]),
            g: f(&p.g, &[nb, t, h]),
            beta: f(&p.beta, &[nb, t, h]),
            s0: f(&p.s0, &[nb, h, dk, dv]),
        }
    }

    fn inputs(&self, s0: bool) -> GdnInputs<'_> {
        GdnInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            initial_state: s0.then_some(&self.s0),
        }
    }
}

fn rel(what: &str, got: &Tensor, want: &[f64]) -> f64 {
    let got = got.to_f32_vec().unwrap();
    assert_eq!(got.len(), want.len(), "{what}: length");
    let peak = want.iter().fold(0.0f64, |m, w| m.max(w.abs())).max(1e-30);
    let worst = got.iter().zip(want).fold(0.0f64, |m, (&g, &w)| {
        assert!(g.is_finite(), "{what}: non-finite {g}");
        m.max((f64::from(g) - w).abs())
    });
    worst / peak
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Every output and gradient's bits, forward then backward, for an
/// all-ones output gradient and (with an initial state) a final-state
/// gradient of ones.
fn all_bits(b: &CpuBackend, x: &T4, s0: bool) -> Vec<Vec<u32>> {
    let fwd = b.chunked_gdn_forward(x.inputs(s0)).unwrap();
    let ones = |t: &Tensor| {
        Tensor::from_f32(
            &vec![1.0; t.shape().iter().product()],
            t.shape(),
            b.budget(),
        )
        .unwrap()
    };
    let d_o = ones(&fwd.output);
    let d_fin = ones(&fwd.final_state);
    let gr = b
        .chunked_gdn_backward(x.inputs(s0), &fwd.checkpoints, &d_o, s0.then_some(&d_fin))
        .unwrap();
    let mut out = vec![
        bits(&fwd.output),
        bits(&fwd.final_state),
        bits(&fwd.checkpoints),
        bits(&gr.q),
        bits(&gr.k),
        bits(&gr.v),
        bits(&gr.g),
        bits(&gr.beta),
    ];
    out.extend(gr.initial_state.as_ref().map(bits));
    out
}

#[test]
fn a_sweep_of_shapes_matches_the_f64_forward() {
    let mut r = Lcg(2026);
    let mut worst = 0.0f64;
    let mut cases = 0;
    for i in 0..60u64 {
        let s = Shape {
            b: r.pick(&[1, 2, 3]),
            t: r.pick(&[1, 2, 63, 64, 65, 127, 128, 129, 200]),
            h: r.pick(&[1, 2, 5]),
            dk: r.pick(&[1, 2, 3, 8, 17]),
            dv: r.pick(&[1, 2, 5, 16]),
        };
        let p = P::random(s, 100 + i);
        let b = exact(1 << 30);
        let x = T4::new(&b, &p);
        for s0 in [false, true] {
            let fwd = b.chunked_gdn_forward(x.inputs(s0)).unwrap();
            let want = forward_f64(&p.oracle(s0)).unwrap();
            for (n, got, w) in [
                ("o", &fwd.output, &want.o),
                ("fin", &fwd.final_state, &want.fin),
            ] {
                let e = rel(&format!("{s:?} s0 {s0} {n}"), got, w);
                assert!(e <= BOUND, "{s:?} s0 {s0} {n}: {e:e}");
                worst = worst.max(e);
            }
            cases += 1;
        }
    }
    assert_eq!(cases, 120);
    eprintln!("gdn sweep: 120 cases, worst {worst:e}");
}

/// Central differences at the shapes most likely to be off by one: every
/// extent 1, a chunk exactly full, and two checkpoints crossed.
#[test]
fn gradients_match_central_differences_at_edge_shapes() {
    const H: f64 = 1e-5;
    const GRAD_BOUND: f64 = 3e-4;
    for (i, s) in [
        Shape {
            b: 1,
            t: 1,
            h: 1,
            dk: 1,
            dv: 1,
        },
        Shape {
            b: 1,
            t: 64,
            h: 1,
            dk: 2,
            dv: 1,
        },
        Shape {
            b: 3,
            t: 2,
            h: 1,
            dk: 1,
            dv: 3,
        },
        Shape {
            b: 1,
            t: 129,
            h: 2,
            dk: 2,
            dv: 2,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let p = P::random(s, 500 + i as u64);
        let mut r = Lcg(900 + i as u64);
        let w_o = r.vec(s.rows() * s.dv, -1.0, 1.0);
        let w_f = r.vec(s.state_len(), -1.0, 1.0);
        let loss = |p: &P| {
            let f = forward_f64(&p.oracle(true)).unwrap();
            f.o.iter().zip(&w_o).map(|(a, b)| a * b).sum::<f64>()
                + f.fin.iter().zip(&w_f).map(|(a, b)| a * b).sum::<f64>()
        };
        let b = exact(1 << 30);
        let x = T4::new(&b, &p);
        let fwd = b.chunked_gdn_forward(x.inputs(true)).unwrap();
        let t = |d: &[f64], shape: &[usize]| {
            let y: Vec<f32> = d.iter().map(|&v| v as f32).collect();
            Tensor::from_f32(&y, shape, b.budget()).unwrap()
        };
        let gr = b
            .chunked_gdn_backward(
                x.inputs(true),
                &fwd.checkpoints,
                &t(&w_o, fwd.output.shape()),
                Some(&t(&w_f, fwd.final_state.shape())),
            )
            .unwrap();
        let ds0 = gr.initial_state.as_ref().unwrap();
        type Pick = fn(&mut P) -> &mut Vec<f64>;
        let slots: [(&str, Pick, &Tensor); 6] = [
            ("q", |p| &mut p.q, &gr.q),
            ("k", |p| &mut p.k, &gr.k),
            ("v", |p| &mut p.v, &gr.v),
            ("g", |p| &mut p.g, &gr.g),
            ("beta", |p| &mut p.beta, &gr.beta),
            ("s0", |p| &mut p.s0, ds0),
        ];
        for (name, pick, analytic) in slots {
            let mut base = p.clone();
            let n = pick(&mut base).len();
            let numeric: Vec<f64> = (0..n)
                .map(|j| {
                    let mut a = p.clone();
                    pick(&mut a)[j] += H;
                    let mut c = p.clone();
                    pick(&mut c)[j] -= H;
                    (loss(&a) - loss(&c)) / (2.0 * H)
                })
                .collect();
            // At Dk = 1, l2norm is ~sign(x): its derivative is
            // eps / (x^2 + eps)^1.5 (about 1e-5 here), and the kernel forms it
            // as dy (1 - y^2) with 1 - y^2 ~ 4e-6 from an f32 dy that carries
            // ~1e-7 relative rounding. So dq and dk are known only to a few
            // percent of their own (tiny) size; 5% still fails a zero or a
            // wrong-signed gradient. Every other gradient, and dq and dk at
            // Dk > 1, keep GRAD_BOUND.
            let bound = if s.dk == 1 && (name == "q" || name == "k") {
                5e-2
            } else {
                GRAD_BOUND
            };
            let e = rel(&format!("{s:?} d{name}"), analytic, &numeric);
            assert!(e <= bound, "{s:?} d{name}: {e:e} > {bound:e}");
        }
    }
}

/// `l2norm` is scale-free, so rows scaled by 1e30 (whose `f32` sum of
/// squares overflows) give the output and every gradient but `dq`, `dk` of
/// the unscaled rows; `dq`, `dk` scale by 1e-30. A naive `f32` sum of
/// squares normalizes those rows to zeros.
#[test]
fn rows_whose_f32_sum_of_squares_overflows_still_normalize() {
    let s = Shape {
        b: 1,
        t: 70,
        h: 2,
        dk: 8,
        dv: 4,
    };
    let p = P::random(s, 31);
    let mut big = p.clone();
    for x in big.q.iter_mut().chain(big.k.iter_mut()) {
        *x *= 1e30;
    }
    let b = exact(1 << 30);
    let (small, large) = (T4::new(&b, &p), T4::new(&b, &big));
    let a = b.chunked_gdn_forward(small.inputs(true)).unwrap();
    let z = b.chunked_gdn_forward(large.inputs(true)).unwrap();
    let want = forward_f64(&p.oracle(true)).unwrap();
    assert!(rel("scaled o", &z.output, &want.o) <= BOUND);
    assert!(rel("scaled fin", &z.final_state, &want.fin) <= BOUND);
    let peak = z
        .output
        .to_f32_vec()
        .unwrap()
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak > 1e-3, "the scaled rows normalized to zeros");
    let ones = Tensor::from_f32(&vec![1.0; s.rows() * s.dv], a.output.shape(), b.budget()).unwrap();
    let ga = b
        .chunked_gdn_backward(small.inputs(true), &a.checkpoints, &ones, None)
        .unwrap();
    let gz = b
        .chunked_gdn_backward(large.inputs(true), &z.checkpoints, &ones, None)
        .unwrap();
    for (n, x, y) in [
        ("dv", &ga.v, &gz.v),
        ("dg", &ga.g, &gz.g),
        ("dbeta", &ga.beta, &gz.beta),
    ] {
        let want: Vec<f64> = x
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert!(rel(n, y, &want) <= 1e-5, "{n}");
    }
    for (n, x, y) in [("dq", &ga.q, &gz.q), ("dk", &ga.k, &gz.k)] {
        let want: Vec<f64> = x
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * 1e-30)
            .collect();
        assert!(rel(n, y, &want) <= 1e-5, "{n}");
    }
}

/// Inputs at the edges of the recurrence, each against what it must give:
/// all-zero `q` reads nothing; `beta = 0` writes nothing (the state is `s0`
/// decayed); a decay of `e^-200` forgets everything before the token.
#[test]
fn degenerate_gates_and_rows_give_the_closed_form() {
    let s = Shape {
        b: 1,
        t: 66,
        h: 1,
        dk: 3,
        dv: 2,
    };
    let b = exact(1 << 30);
    let ones = |t: &Tensor| {
        Tensor::from_f32(
            &vec![1.0; t.shape().iter().product()],
            t.shape(),
            b.budget(),
        )
        .unwrap()
    };

    // q = 0: the output is exactly zero, and backward is finite.
    let mut p = P::random(s, 41);
    p.q.iter_mut().for_each(|x| *x = 0.0);
    let x = T4::new(&b, &p);
    let f = b.chunked_gdn_forward(x.inputs(true)).unwrap();
    assert!(f.output.to_f32_vec().unwrap().iter().all(|&v| v == 0.0));
    b.chunked_gdn_backward(x.inputs(true), &f.checkpoints, &ones(&f.output), None)
        .unwrap();

    // beta = 0 and g = 0: the state never changes, so S_T = s0 exactly and
    // every checkpoint is s0.
    let mut p = P::random(s, 42);
    p.beta.iter_mut().for_each(|x| *x = 0.0);
    p.g.iter_mut().for_each(|x| *x = 0.0);
    let x = T4::new(&b, &p);
    let f = b.chunked_gdn_forward(x.inputs(true)).unwrap();
    let s0 = x.s0.to_f32_vec().unwrap();
    assert_eq!(f.final_state.to_f32_vec().unwrap(), s0);
    let ck = f.checkpoints.to_f32_vec().unwrap();
    assert_eq!(ck.len(), 2 * s0.len());
    assert!(ck.chunks(s0.len()).all(|c| c == s0.as_slice()));

    // g = -200: exp underflows to 0, so token t's state is k^ (beta v)^T
    // alone, and the output and its f64 reference agree.
    let mut p = P::random(s, 43);
    p.g.iter_mut().for_each(|x| *x = -200.0);
    let x = T4::new(&b, &p);
    let f = b.chunked_gdn_forward(x.inputs(true)).unwrap();
    let want = forward_f64(&p.oracle(true)).unwrap();
    assert!(rel("forget o", &f.output, &want.o) <= BOUND);
    let gr = b
        .chunked_gdn_backward(x.inputs(true), &f.checkpoints, &ones(&f.output), None)
        .unwrap();
    // Nothing reaches s0 through a zero decay.
    let ds0 = gr.initial_state.unwrap().to_f32_vec().unwrap();
    assert!(ds0.iter().all(|&v| v == 0.0), "{ds0:?}");
}

/// Exact and Fast, one thread or eighteen, and eight callers at once on one
/// pooled backend: the same bits, because tasks split whole heads and the
/// kernel has one arithmetic.
#[test]
fn bits_do_not_depend_on_numerics_threads_or_concurrent_callers() {
    // 2 x 6 heads x 300 tokens x 256 state values x 3 passes is past the
    // 2^21 units at which the forward splits across the pool.
    let s = Shape {
        b: 2,
        t: 300,
        h: 6,
        dk: 16,
        dv: 16,
    };
    let p = P::random(s, 61);
    let serial = exact(1 << 31);
    let want = all_bits(&serial, &T4::new(&serial, &p), true);
    for threads in [2usize, 3, 7, 18] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let b = CpuBackend::with_threads(Budget::new(1 << 31), threads)
                .unwrap()
                .with_numerics(numerics);
            let got = all_bits(&b, &T4::new(&b, &p), true);
            assert!(got == want, "threads {threads} {numerics:?} changed bits");
        }
    }
    let shared = Arc::new(CpuBackend::with_threads(Budget::new(1 << 32), 4).unwrap());
    let p = Arc::new(p);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (b, p) = (Arc::clone(&shared), Arc::clone(&p));
            std::thread::spawn(move || all_bits(&b, &T4::new(&b, &p), true))
        })
        .collect();
    for h in handles {
        assert!(
            h.join().unwrap() == want,
            "a concurrent caller changed bits"
        );
    }
}

/// The op's charges end with the call; a cap a little under its need is
/// `CapacityExceeded` with nothing left charged; a cancel between heads is
/// returned and leaves nothing charged.
#[test]
fn the_budget_is_released_on_success_refusal_and_cancel() {
    let s = Shape {
        b: 1,
        t: 130,
        h: 4,
        dk: 16,
        dv: 16,
    };
    let p = P::random(s, 71);
    let b = exact(1 << 30);
    let x = T4::new(&b, &p);
    let base = b.budget().live_bytes().unwrap();
    b.budget().reset_peak();
    let f = b.chunked_gdn_forward(x.inputs(true)).unwrap();
    let outs: u64 = [&f.output, &f.final_state, &f.checkpoints]
        .iter()
        .map(|t| 4 * t.shape().iter().product::<usize>() as u64)
        .sum();
    assert_eq!(
        b.budget().live_bytes().unwrap(),
        base + outs,
        "only the outputs stay charged"
    );
    let fwd_peak = b.budget().peak_bytes() - base;
    assert!(
        fwd_peak > outs,
        "scratch was charged: peak {fwd_peak} outputs {outs}"
    );
    drop(f);
    assert_eq!(b.budget().live_bytes().unwrap(), base);

    // A cap one byte under the forward's peak need refuses and charges nothing.
    let tight = Budget::new(fwd_peak - 1);
    let b2 = CpuBackend::new(tight.clone()).with_numerics(Numerics::Exact);
    let x2 = T4::new(&b, &p);
    let r = b2.chunked_gdn_forward(x2.inputs(true));
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
    assert_eq!(tight.live_bytes().unwrap(), 0);
    // ...and exactly the need succeeds.
    let fits = Budget::new(fwd_peak);
    let b3 = CpuBackend::new(fits.clone()).with_numerics(Numerics::Exact);
    let f3 = b3.chunked_gdn_forward(x2.inputs(true)).unwrap();
    drop(f3);
    assert_eq!(fits.live_bytes().unwrap(), 0);

    // A cancel after the first head, forward and backward.
    let f = b.chunked_gdn_forward(x.inputs(true)).unwrap();
    let d_o = Tensor::from_f32(&vec![1.0; s.rows() * s.dv], f.output.shape(), b.budget()).unwrap();
    let live = b.budget().live_bytes().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    b.set_cancel(move || {
        if seen.fetch_add(1, Ordering::SeqCst) >= 1 {
            Err(OjasError::Backend {
                id: ojas_core::BackendId::Cpu,
                detail: "cancelled".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(b.chunked_gdn_forward(x.inputs(true)).is_err());
    assert_eq!(b.budget().live_bytes().unwrap(), live);
    calls.store(0, Ordering::SeqCst);
    assert!(b
        .chunked_gdn_backward(x.inputs(true), &f.checkpoints, &d_o, None)
        .is_err());
    assert_eq!(b.budget().live_bytes().unwrap(), live);
    b.set_cancel(|| Ok(()));
    b.chunked_gdn_backward(x.inputs(true), &f.checkpoints, &d_o, None)
        .unwrap();
}

/// Mismatched checkpoints, the wrong output gradient and non-finite
/// backward operands are each refused with the error the contract names.
#[test]
fn backward_refuses_wrong_checkpoints_and_non_finite_gradients() {
    let s = Shape {
        b: 1,
        t: 65,
        h: 1,
        dk: 2,
        dv: 2,
    };
    let p = P::random(s, 81);
    let b = exact(1 << 30);
    let x = T4::new(&b, &p);
    let f = b.chunked_gdn_forward(x.inputs(false)).unwrap();
    let t = |v: f32, shape: &[usize]| {
        Tensor::from_f32(&vec![v; shape.iter().product()], shape, b.budget()).unwrap()
    };
    let d_o = t(1.0, f.output.shape());
    // A checkpoint tensor for T = 64 (one chunk) is the wrong shape.
    let short = t(0.0, &[1, 1, 1, 2, 2]);
    assert!(matches!(
        b.chunked_gdn_backward(x.inputs(false), &short, &d_o, None),
        Err(OjasError::Shape {
            op: "chunked_gdn_backward",
            ..
        })
    ));
    let mut ck = f.checkpoints.to_f32_vec().unwrap();
    ck[3] = f32::NAN;
    let bad_ck = Tensor::from_f32(&ck, f.checkpoints.shape(), b.budget()).unwrap();
    assert!(matches!(
        b.chunked_gdn_backward(x.inputs(false), &bad_ck, &d_o, None),
        Err(OjasError::NonFinite {
            op: "chunked_gdn_backward"
        })
    ));
    let inf = t(f32::INFINITY, f.final_state.shape());
    assert!(matches!(
        b.chunked_gdn_backward(x.inputs(false), &f.checkpoints, &d_o, Some(&inf)),
        Err(OjasError::NonFinite {
            op: "chunked_gdn_backward"
        })
    ));
    // A transposed view of `v` (same shape, column-major over T and Dv) is
    // refused, not read as if it were row-major.
    let view = x.v.view(&[1, 65, 1, 2], &[130, 1, 2, 65], 0).unwrap();
    assert!(!view.is_contiguous().unwrap());
    let r = b.chunked_gdn_forward(GdnInputs {
        v: &view,
        ..x.inputs(false)
    });
    assert!(
        matches!(
            r,
            Err(OjasError::Shape {
                op: "chunked_gdn_forward",
                ..
            })
        ),
        "a strided view was accepted: {r:?}"
    );
}

/// T = 4096 at a Qwen-like head shape: f32 against f64 over a long
/// recurrence, crossing 64 checkpoints, with threads.
#[test]
#[ignore = "long; run with --release -- --ignored"]
fn a_long_sequence_stays_within_bound_of_the_f64_forward() {
    let s = Shape {
        b: 1,
        t: 4096,
        h: 2,
        dk: 64,
        dv: 64,
    };
    let p = P::random(s, 91);
    let b = CpuBackend::with_threads(Budget::new(1 << 32), 4).unwrap();
    let x = T4::new(&b, &p);
    let f = b.chunked_gdn_forward(x.inputs(true)).unwrap();
    let want = forward_f64(&p.oracle(true)).unwrap();
    let eo = rel("long o", &f.output, &want.o);
    let ef = rel("long fin", &f.final_state, &want.fin);
    eprintln!("gdn T=4096: o {eo:e} fin {ef:e}");
    assert!(eo <= 1e-4 && ef <= 1e-4);
    let serial = exact(1 << 32);
    assert!(all_bits(&serial, &T4::new(&serial, &p), true) == all_bits(&b, &x, true));
}
