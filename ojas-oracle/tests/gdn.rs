//! `Backend::chunked_gdn_forward` / `_backward` on the CPU, against
//! transformers' float64 goldens, central differences of the f64 forward
//! (`ojas_oracle::gdn::forward_f64`), and that forward's states at each
//! checkpoint.

use ojas_core::{Backend, Budget, GdnGrad, GdnInputs, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_oracle::gdn::{forward_f64, golden, Inputs, Shape, GOLDEN_SEQ};

/// f32 against the f64 goldens, `max |got - want| / max |want|` per tensor.
/// The goldens' own f32 transformers run is within 2.2e-7 of them
/// (`manifest.json`); this kernel sums in a different order.
const GOLDEN_BOUND: f64 = 1e-5;

fn exact() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact)
}

fn threaded_fast() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 30), 4).unwrap()
}

fn t32(b: &CpuBackend, data: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    Tensor::from_f32(&f, shape, b.budget()).unwrap()
}

fn rel(name: &str, got: &Tensor, want: &[f64]) -> f64 {
    let got = got.to_f32_vec().unwrap();
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    if scale == 0.0 {
        // An exactly zero reference must be met exactly.
        assert!(
            got.iter().all(|&g| g == 0.0),
            "{name}: reference is zero, got is not"
        );
        return 0.0;
    }
    let worst = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&g, &w)| m.max((f64::from(g) - w).abs()));
    worst / scale
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Host tensors for one problem, built on `b`.
struct Problem {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Option<Tensor>,
}

impl Problem {
    fn new(b: &CpuBackend, p: &Inputs<'_>) -> Self {
        let Shape {
            b: nb,
            t,
            h,
            dk,
            dv,
        } = p.s;
        Self {
            q: t32(b, p.q, &[nb, t, h, dk]),
            k: t32(b, p.k, &[nb, t, h, dk]),
            v: t32(b, p.v, &[nb, t, h, dv]),
            g: t32(b, p.g, &[nb, t, h]),
            beta: t32(b, p.beta, &[nb, t, h]),
            s0: p.s0.map(|s0| t32(b, s0, &[nb, h, dk, dv])),
        }
    }

    fn inputs(&self) -> GdnInputs<'_> {
        GdnInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            initial_state: self.s0.as_ref(),
        }
    }
}

fn grad_tensors(g: &GdnGrad) -> Vec<&Tensor> {
    let mut v = vec![&g.q, &g.k, &g.v, &g.g, &g.beta];
    v.extend(g.initial_state.as_ref());
    v
}

#[test]
fn cpu_matches_the_published_goldens_at_every_checkpoint_boundary() {
    let mut worst = 0.0f64;
    for t in GOLDEN_SEQ {
        let gold = golden(t).unwrap();
        let s = gold.s;
        let mut runs = Vec::new();
        for b in [exact(), threaded_fast()] {
            let p = Problem::new(&b, &gold.inputs());
            let d_o = t32(&b, &gold.d_o, &[s.b, t, s.h, s.dv]);
            let dfin = t32(&b, &gold.dfin, &[s.b, s.h, s.dk, s.dv]);
            let fwd = b.chunked_gdn_forward(p.inputs()).unwrap();
            assert_eq!(
                fwd.checkpoints.shape(),
                [s.b, s.h, t.div_ceil(64), s.dk, s.dv]
            );
            let grad = b
                .chunked_gdn_backward(p.inputs(), &fwd.checkpoints, &d_o, Some(&dfin))
                .unwrap();
            runs.push((fwd, grad));
        }
        let (fwd, grad) = &runs[0];
        let ds0 = grad
            .initial_state
            .as_ref()
            .expect("ds0 with an initial state");
        for (name, got, want) in [
            ("o", &fwd.output, &gold.o),
            ("fin", &fwd.final_state, &gold.fin),
            ("dq", &grad.q, &gold.dq),
            ("dk", &grad.k, &gold.dk),
            ("dv", &grad.v, &gold.dv),
            ("dg", &grad.g, &gold.dg),
            ("dbeta", &grad.beta, &gold.dbeta),
            ("ds0", ds0, &gold.ds0),
        ] {
            let r = rel(name, got, want);
            assert!(r <= GOLDEN_BOUND, "T {t} {name}: {r:e} > {GOLDEN_BOUND:e}");
            worst = worst.max(r);
        }
        // One kernel: Fast on four threads has Exact's serial bits.
        let (f2, g2) = &runs[1];
        for (a, b) in [
            (&fwd.output, &f2.output),
            (&fwd.final_state, &f2.final_state),
            (&fwd.checkpoints, &f2.checkpoints),
        ]
        .into_iter()
        .chain(grad_tensors(grad).into_iter().zip(grad_tensors(g2)))
        {
            assert_eq!(bits(a), bits(b), "T {t}: threads or numerics changed bits");
        }
    }
    eprintln!("gdn goldens: worst relative error {worst:e}");
}

/// A small deterministic generator, so the test needs no dependency.
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
}

/// Owned operands of a random problem: `g` is a log decay in `[-1, -0.05]`
/// and `beta` in `(0.1, 0.9)`, as a sigmoid gate gives.
struct Random {
    s: Shape,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    s0: Vec<f64>,
}

impl Random {
    fn new(s: Shape, seed: u64) -> Self {
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

    fn inputs(&self, with_s0: bool) -> Inputs<'_> {
        Inputs {
            s: self.s,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: with_s0.then_some(self.s0.as_slice()),
        }
    }
}

/// Every input's gradient against central differences of the f64 forward,
/// at a length that crosses two checkpoints (T = 67 > 64), so the backward's
/// recompute from a saved state is on the path being checked.
#[test]
fn cpu_backward_matches_central_differences_across_checkpoints() {
    const H: f64 = 1e-5;
    const BOUND: f64 = 2e-4;
    let s = Shape {
        b: 2,
        t: 67,
        h: 2,
        dk: 5,
        dv: 3,
    };
    let base = Random::new(s, 7);
    let mut r = Lcg(99);
    let w_o = r.vec(s.rows() * s.dv, -1.0, 1.0);
    let w_f = r.vec(s.state_len(), -1.0, 1.0);
    let loss = |x: &Inputs<'_>| -> f64 {
        let f = forward_f64(x).unwrap();
        let a: f64 = f.o.iter().zip(&w_o).map(|(o, w)| o * w).sum();
        let b: f64 = f.fin.iter().zip(&w_f).map(|(o, w)| o * w).sum();
        a + b
    };
    let b = exact();
    let p = Problem::new(&b, &base.inputs(true));
    let fwd = b.chunked_gdn_forward(p.inputs()).unwrap();
    let grad = b
        .chunked_gdn_backward(
            p.inputs(),
            &fwd.checkpoints,
            &t32(&b, &w_o, &[s.b, s.t, s.h, s.dv]),
            Some(&t32(&b, &w_f, &[s.b, s.h, s.dk, s.dv])),
        )
        .unwrap();
    let ds0 = grad.initial_state.as_ref().unwrap();
    for (slot, name, analytic) in [
        (0, "q", &grad.q),
        (1, "k", &grad.k),
        (2, "v", &grad.v),
        (3, "g", &grad.g),
        (4, "beta", &grad.beta),
        (5, "s0", ds0),
    ] {
        let mut numeric = Vec::new();
        let len = match slot {
            0 => base.q.len(),
            1 => base.k.len(),
            2 => base.v.len(),
            3 => base.g.len(),
            4 => base.beta.len(),
            _ => base.s0.len(),
        };
        for i in 0..len {
            let at = |d: f64| {
                let mut x = clone_random(&base);
                let buf = match slot {
                    0 => &mut x.q,
                    1 => &mut x.k,
                    2 => &mut x.v,
                    3 => &mut x.g,
                    4 => &mut x.beta,
                    _ => &mut x.s0,
                };
                buf[i] += d;
                loss(&x.inputs(true))
            };
            numeric.push((at(H) - at(-H)) / (2.0 * H));
        }
        let r = rel(name, analytic, &numeric);
        assert!(r <= BOUND, "d{name}: {r:e} > {BOUND:e}");
    }
}

fn clone_random(r: &Random) -> Random {
    Random {
        s: r.s,
        q: r.q.clone(),
        k: r.k.clone(),
        v: r.v.clone(),
        g: r.g.clone(),
        beta: r.beta.clone(),
        s0: r.s0.clone(),
    }
}

/// Checkpoint `c` is the state entering token `64 c`: zeros (or `s0`) for
/// `c = 0`, and the f64 forward's final state over the first `64 c` tokens
/// after that. Without an initial state the backward returns no `ds0`.
#[test]
fn checkpoints_are_the_states_entering_each_chunk() {
    let s = Shape {
        b: 1,
        t: 200,
        h: 3,
        dk: 4,
        dv: 2,
    };
    let x = Random::new(s, 3);
    let b = exact();
    for with_s0 in [false, true] {
        let p = Problem::new(&b, &x.inputs(with_s0));
        let fwd = b.chunked_gdn_forward(p.inputs()).unwrap();
        let f64_fwd = forward_f64(&x.inputs(with_s0)).unwrap();
        assert!(rel("o", &fwd.output, &f64_fwd.o) <= GOLDEN_BOUND);
        assert!(rel("fin", &fwd.final_state, &f64_fwd.fin) <= GOLDEN_BOUND);
        let ck = fwd.checkpoints.to_f32_vec().unwrap();
        let per = s.dk * s.dv;
        let nc = s.t.div_ceil(64);
        assert_eq!(nc, 4);
        for c in 0..nc {
            let want: Vec<f64> = if c == 0 {
                if with_s0 {
                    x.s0.clone()
                } else {
                    vec![0.0; s.state_len()]
                }
            } else {
                // The prefix of 64 c tokens, every row of every head.
                let tp = 64 * c;
                let cut = |xs: &[f64], w: usize| xs[..s.b * tp * s.h * w].to_vec();
                let pre = Random {
                    s: Shape { t: tp, ..s },
                    q: cut(&x.q, s.dk),
                    k: cut(&x.k, s.dk),
                    v: cut(&x.v, s.dv),
                    g: cut(&x.g, 1),
                    beta: cut(&x.beta, 1),
                    s0: x.s0.clone(),
                };
                forward_f64(&pre.inputs(with_s0)).unwrap().fin
            };
            for head in 0..s.b * s.h {
                for e in 0..per {
                    let got = f64::from(ck[(head * nc + c) * per + e]);
                    let w = want[head * per + e];
                    assert!(
                        (got - w).abs() <= 1e-5 * w.abs().max(1.0),
                        "s0 {with_s0} checkpoint {c} head {head} [{e}]: {got} vs {w}"
                    );
                }
            }
        }
        let d_o =
            Tensor::from_f32(&vec![1.0; s.rows() * s.dv], fwd.output.shape(), b.budget()).unwrap();
        let grad = b
            .chunked_gdn_backward(p.inputs(), &fwd.checkpoints, &d_o, None)
            .unwrap();
        assert_eq!(grad.initial_state.is_some(), with_s0);
    }
}

#[test]
fn non_finite_operands_are_refused_after_the_shape_check() {
    let s = Shape {
        b: 1,
        t: 3,
        h: 1,
        dk: 2,
        dv: 2,
    };
    let mut x = Random::new(s, 5);
    x.g[1] = f64::NAN;
    let b = exact();
    let p = Problem::new(&b, &x.inputs(false));
    assert!(matches!(
        b.chunked_gdn_forward(p.inputs()),
        Err(OjasError::NonFinite {
            op: "chunked_gdn_forward"
        })
    ));
    // A misshapen beta beside the NaN is the shape error.
    let wrong = Tensor::from_f32(&[0.5; 4], &[1, 4, 1], b.budget()).unwrap();
    let bad = GdnInputs {
        beta: &wrong,
        ..p.inputs()
    };
    assert!(matches!(
        b.chunked_gdn_forward(bad),
        Err(OjasError::Shape {
            op: "chunked_gdn_forward",
            ..
        })
    ));
    // A decay that overflows the state is refused, not returned.
    let mut y = Random::new(s, 5);
    y.g = vec![80.0; s.rows()];
    let p = Problem::new(&b, &y.inputs(false));
    assert!(matches!(
        b.chunked_gdn_forward(p.inputs()),
        Err(OjasError::NonFinite { .. })
    ));
}

#[test]
fn a_budget_too_small_for_the_outputs_is_refused() {
    let s = Shape {
        b: 1,
        t: 70,
        h: 2,
        dk: 8,
        dv: 8,
    };
    let x = Random::new(s, 11);
    let roomy = exact();
    let p = Problem::new(&roomy, &x.inputs(false));
    // Inputs are already charged to `roomy`; a fresh backend with a tight
    // cap must refuse rather than allocate past it.
    let tight = CpuBackend::new(Budget::new(4096)).with_numerics(Numerics::Exact);
    assert!(matches!(
        tight.chunked_gdn_forward(p.inputs()),
        Err(OjasError::CapacityExceeded { .. })
    ));
}
