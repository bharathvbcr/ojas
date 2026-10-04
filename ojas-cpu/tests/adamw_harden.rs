//! The counted AdamW element loops stay on the scalar formula.
//!
//! Fast is bit-identical to an f32 reimplementation of `AdamCoeffs32::elem`
//! at 1 and 6 threads. Exact is finite and bit-identical to itself across
//! those thread counts. A non-finite element, including one past a full
//! block, leaves the parameter and both moments unchanged and does not
//! charge the budget. Length 0 is refused by the layout check.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use ojas_core::{check_adamw, AdamWConfig, Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const THREADS: [usize; 2] = [1, 6];
/// `1 << 17` is the element-block size. `+ 3` is a scalar tail on the next block.
const LENGTHS: [usize; 6] = [1, 7, 8, 9, 17, (1 << 17) + 3];
const STEP: u64 = 7;

/// f32 scalars of one Fast step, formed the way `AdamCoeffs::fast` forms them:
/// f64 arithmetic in `check_adamw`, then one cast. `low` and `decays` are the
/// f64 flags, not recomputed from the f32 values.
struct FastAdam {
    beta1: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2_sqrt: f32,
    eps: f32,
    decay: f32,
    decays: bool,
    low: bool,
}

fn fast_adam(config: AdamWConfig, step: u64) -> FastAdam {
    let (_, bc1, bc2) = check_adamw(config, step).expect("adamw config");
    let one_minus_b1 = 1.0 - config.beta1;
    FastAdam {
        beta1: config.beta1 as f32,
        beta2: config.beta2 as f32,
        one_minus_b1: one_minus_b1 as f32,
        one_minus_b2: (1.0 - config.beta2) as f32,
        step_size: (config.lr / bc1) as f32,
        bc2_sqrt: bc2.sqrt() as f32,
        eps: config.eps as f32,
        decay: (1.0 - config.lr * config.weight_decay) as f32,
        decays: config.weight_decay != 0.0,
        low: one_minus_b1 < 0.5,
    }
}

/// Scalar body of `AdamCoeffs32::elem`.
fn fast_elem(c: &FastAdam, p: f32, g: f32, m: f32, v: f32) -> (f32, f32, f32) {
    let m = if c.low {
        m + c.one_minus_b1 * (g - m)
    } else {
        g - (g - m) * c.beta1
    };
    let v = c.beta2 * v + c.one_minus_b2 * g * g;
    let denom = v.sqrt() / c.bc2_sqrt + c.eps;
    let delta = (-c.step_size) * m / denom;
    let mut q = p;
    if c.decays {
        q *= c.decay;
    }
    q += delta;
    let new_p = if !c.decays && delta == 0.0 { p } else { q };
    (new_p, m, v)
}

fn fast_ref(c: &FastAdam, p: &[f32], g: &[f32], m: &[f32], v: &[f32]) -> [Vec<f32>; 3] {
    let mut ps = Vec::with_capacity(p.len());
    let mut ms = Vec::with_capacity(p.len());
    let mut vs = Vec::with_capacity(p.len());
    for i in 0..p.len() {
        let (np, nm, nv) = fast_elem(c, p[i], g[i], m[i], v[i]);
        ps.push(np);
        ms.push(nm);
        vs.push(nv);
    }
    [ps, ms, vs]
}

fn configs() -> [AdamWConfig; 4] {
    [
        AdamWConfig::nanolab(1e-3, 0.0),
        AdamWConfig::nanolab(1e-3, 0.1),
        AdamWConfig {
            beta1: 0.3,
            ..AdamWConfig::nanolab(1e-3, 0.0)
        },
        AdamWConfig {
            beta1: 0.3,
            ..AdamWConfig::nanolab(1e-3, 0.1)
        },
    ]
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = ((s >> 40) as u32) & 0x00ff_ffff;
        let x = (bits as f32) / (0x0100_0000 as f32) - 0.5;
        out.push(x * 0.02);
    }
    out
}

/// Modest finite inputs, plus a negative zero that a zero step must keep
/// when weight decay is off, and a large-but-finite gradient.
fn inputs(n: usize, seed: u64) -> [Vec<f32>; 4] {
    let mut p = noise(n, seed);
    let mut g = noise(n, seed ^ 0x1111);
    let mut m = noise(n, seed ^ 0x2222);
    let mut v = noise(n, seed ^ 0x3333);
    for value in &mut v {
        *value = value.abs() + 1e-4;
    }
    if n > 0 {
        p[0] = -0.0;
        g[0] = 0.0;
        m[0] = 0.0;
        v[0] = 0.0;
    }
    if n > 1 {
        p[1] = f32::MAX;
        g[1] = 0.0;
        m[1] = 0.0;
        v[1] = 0.0;
    }
    if n > 2 {
        g[2] = 1e10;
    }
    if n > 4 {
        p[4] = f32::from_bits(1);
    }
    [p, g, m, v]
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what} length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what} [{i}]: got {a:?} ({:#010x}) want {b:?} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

struct Ran {
    result: Result<(), OjasError>,
    param: Vec<f32>,
    moment1: Vec<f32>,
    moment2: Vec<f32>,
    live_before: u64,
    live_after: u64,
}

fn run(
    threads: usize,
    numerics: Numerics,
    p: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    config: AdamWConfig,
) -> Ran {
    let budget = Budget::new(u64::MAX);
    let backend = CpuBackend::with_threads(budget.clone(), threads)
        .expect("threads")
        .with_numerics(numerics);
    let shape = [p.len()];
    let mut param = Tensor::from_f32(p, &shape, &budget).expect("param");
    let grad = Tensor::from_f32(g, &shape, &budget).expect("grad");
    let mut moment1 = Tensor::from_f32(m, &shape, &budget).expect("moment1");
    let mut moment2 = Tensor::from_f32(v, &shape, &budget).expect("moment2");
    let live_before = budget.live_bytes().expect("live");
    let result = backend.adamw_step(&mut param, &grad, &mut moment1, &mut moment2, STEP, config);
    let live_after = budget.live_bytes().expect("live");
    Ran {
        result,
        param: param.to_f32_vec().expect("param out"),
        moment1: moment1.to_f32_vec().expect("moment1 out"),
        moment2: moment2.to_f32_vec().expect("moment2 out"),
        live_before,
        live_after,
    }
}

fn label(n: usize, threads: usize, numerics: Numerics, config: AdamWConfig) -> String {
    format!(
        "n={n} threads={threads} {numerics:?} wd={} beta1={}",
        config.weight_decay, config.beta1
    )
}

#[test]
fn fast_matches_scalar_f32_and_exact_matches_itself_across_threads() {
    for &n in &LENGTHS {
        let [p, g, m, v] = inputs(n, 0xa11);
        for config in configs() {
            let coeffs = fast_adam(config, STEP);
            let want = fast_ref(&coeffs, &p, &g, &m, &v);
            let mut exact: Option<[Vec<f32>; 3]> = None;
            for &threads in &THREADS {
                for numerics in [Numerics::Fast, Numerics::Exact] {
                    let what = label(n, threads, numerics, config);
                    let ran = run(threads, numerics, &p, &g, &m, &v, config);
                    ran.result.expect(&what);
                    assert_eq!(ran.live_before, ran.live_after, "{what} budget");
                    if numerics == Numerics::Fast {
                        assert_bits(&format!("{what} param"), &ran.param, &want[0]);
                        assert_bits(&format!("{what} moment1"), &ran.moment1, &want[1]);
                        assert_bits(&format!("{what} moment2"), &ran.moment2, &want[2]);
                    } else {
                        assert!(
                            ran.param
                                .iter()
                                .chain(&ran.moment1)
                                .chain(&ran.moment2)
                                .all(|x| x.is_finite()),
                            "{what} exact produced a non-finite"
                        );
                        match &exact {
                            None => exact = Some([ran.param, ran.moment1, ran.moment2]),
                            Some(prev) => {
                                assert_bits(
                                    &format!("{what} param vs threads {}", THREADS[0]),
                                    &ran.param,
                                    &prev[0],
                                );
                                assert_bits(
                                    &format!("{what} moment1 vs threads {}", THREADS[0]),
                                    &ran.moment1,
                                    &prev[1],
                                );
                                assert_bits(
                                    &format!("{what} moment2 vs threads {}", THREADS[0]),
                                    &ran.moment2,
                                    &prev[2],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn empty_length_is_refused_and_a_successful_empty_would_match() {
    let budget = Budget::new(u64::MAX);
    let backend = CpuBackend::with_threads(budget.clone(), 1)
        .expect("threads")
        .with_numerics(Numerics::Fast);
    let shape = [0usize];
    let mut param = Tensor::from_f32(&[], &shape, &budget).expect("param");
    let grad = Tensor::from_f32(&[], &shape, &budget).expect("grad");
    let mut moment1 = Tensor::from_f32(&[], &shape, &budget).expect("m1");
    let mut moment2 = Tensor::from_f32(&[], &shape, &budget).expect("m2");
    let live = budget.live_bytes().expect("live");
    let result = backend.adamw_step(
        &mut param,
        &grad,
        &mut moment1,
        &mut moment2,
        STEP,
        AdamWConfig::nanolab(1e-3, 0.1),
    );
    assert_eq!(budget.live_bytes().expect("live"), live);
    match result {
        Err(OjasError::Shape { .. }) => {
            assert!(param.to_f32_vec().expect("p").is_empty());
            assert!(moment1.to_f32_vec().expect("m1").is_empty());
            assert!(moment2.to_f32_vec().expect("m2").is_empty());
        }
        Ok(()) => {
            assert!(param.to_f32_vec().expect("p").is_empty());
        }
        Err(other) => panic!("length 0: {other:?}"),
    }
}

#[test]
fn nonfinite_last_element_leaves_tensors_and_budget_unchanged() {
    type PoisonFn = fn(&mut [Vec<f32>; 4]);
    let poisons: [(&str, PoisonFn); 4] = [
        ("grad nan", |t| {
            let n = t[0].len();
            t[1][n - 1] = f32::NAN;
        }),
        ("moment2 +inf", |t| {
            let n = t[0].len();
            t[3][n - 1] = f32::INFINITY;
        }),
        ("grad 1e30", |t| {
            let n = t[0].len();
            t[1][n - 1] = 1e30;
        }),
        ("moment2 negative", |t| {
            let n = t[0].len();
            t[3][n - 1] = -1.0;
        }),
    ];
    for &n in &LENGTHS {
        let base = inputs(n, 0xb22);
        for (name, poison) in poisons {
            let mut tensors = base.clone();
            poison(&mut tensors);
            let [p, g, m, v] = tensors;
            for config in [
                AdamWConfig::nanolab(1e-3, 0.0),
                AdamWConfig::nanolab(1e-3, 0.1),
            ] {
                for &threads in &THREADS {
                    for numerics in [Numerics::Fast, Numerics::Exact] {
                        let what = format!("{name} {}", label(n, threads, numerics, config));
                        let ran = run(threads, numerics, &p, &g, &m, &v, config);
                        assert!(
                            matches!(ran.result, Err(OjasError::NonFinite { .. })),
                            "{what}: {:?}",
                            ran.result
                        );
                        assert_eq!(ran.live_before, ran.live_after, "{what} budget");
                        assert_bits(&format!("{what} param"), &ran.param, &p);
                        assert_bits(&format!("{what} moment1"), &ran.moment1, &m);
                        assert_bits(&format!("{what} moment2"), &ran.moment2, &v);
                    }
                }
            }
        }
    }
}

#[test]
fn concurrent_adamw_steps_on_one_backend_do_not_deadlock() {
    let n = (1 << 17) + 3;
    let [p, g, m, v] = inputs(n, 0xc33);
    let config = AdamWConfig::nanolab(1e-3, 0.1);
    let want = fast_ref(&fast_adam(config, STEP), &p, &g, &m, &v);
    let (done, wait) = mpsc::channel();
    let runner = thread::Builder::new()
        .name("adamw-pair".to_string())
        .spawn(move || {
            let budget = Budget::new(u64::MAX);
            let backend = CpuBackend::with_threads(budget.clone(), 6)
                .expect("threads")
                .with_numerics(Numerics::Fast);
            thread::scope(|scope| {
                for _ in 0..2 {
                    let backend = &backend;
                    let budget = &budget;
                    let (p, g, m, v) = (&p, &g, &m, &v);
                    let want = &want;
                    scope.spawn(move || {
                        let shape = [n];
                        let mut param = Tensor::from_f32(p, &shape, budget).expect("param");
                        let grad = Tensor::from_f32(g, &shape, budget).expect("grad");
                        let mut moment1 = Tensor::from_f32(m, &shape, budget).expect("m1");
                        let mut moment2 = Tensor::from_f32(v, &shape, budget).expect("m2");
                        backend
                            .adamw_step(&mut param, &grad, &mut moment1, &mut moment2, STEP, config)
                            .expect("concurrent step");
                        assert_bits(
                            "concurrent param",
                            &param.to_f32_vec().expect("p"),
                            &want[0],
                        );
                        assert_bits(
                            "concurrent moment1",
                            &moment1.to_f32_vec().expect("m1"),
                            &want[1],
                        );
                        assert_bits(
                            "concurrent moment2",
                            &moment2.to_f32_vec().expect("m2"),
                            &want[2],
                        );
                    });
                }
            });
            let _ = done.send(());
        })
        .expect("spawn");
    match wait.recv_timeout(Duration::from_secs(60)) {
        Ok(()) => runner.join().expect("concurrent adamw"),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            runner.join().expect("concurrent adamw");
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("two adamw_step calls on one backend did not finish in 60s");
        }
    }
}
