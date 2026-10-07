//! Exact-mode bits under concurrent callers at every pool width from 1 to 18.
//!
//! `numerics.rs` compares widths one call at a time; here several OS threads
//! share one backend and call at once, and the test's own handle is dropped
//! while they run, so the pool is torn down by whichever caller finishes
//! last. Every output must equal the 1-thread bits. Shapes are above the
//! split thresholds (rows of 2^15 values, AdamW chunks of 2^14, more than one
//! GEMM tile, several attention heads), so the work is spread on the pool.

use std::sync::{mpsc, Barrier};
use std::time::Duration;

use ojas_core::{AdamWConfig, Backend, Budget, Numerics, Tensor, RMS_NORM_EPS};
use ojas_cpu::CpuBackend;

mod common;
use common::{bits, SplitMix64};

struct Inputs {
    x: Tensor,
    w: Tensor,
    gy: Tensor,
    rx: Tensor,
    rw: Tensor,
    rgy: Tensor,
    q: Tensor,
    sgy: Tensor,
    p: Vec<f32>,
    g: Tensor,
    m1: Vec<f32>,
    m2: Vec<f32>,
    logits: Tensor,
    targets: Tensor,
}

const P_LEN: usize = 1 << 16;
const P_SHAPE: [usize; 2] = [256, 256];

fn inputs() -> Inputs {
    let b = Budget::new(1 << 30);
    let mut rng = SplitMix64(0x9001);
    let t = |rng: &mut SplitMix64, shape: &[usize], scale: f32| {
        let n: usize = shape.iter().product();
        Tensor::from_f32(&rng.vec(n, scale), shape, &b).unwrap()
    };
    let (rows, kin, nout) = (160usize, 256usize, 192usize);
    let sdpa = [2usize, 4, 96, 32];
    let vocab = 512usize;
    let targets: Vec<u32> = (0..128u32).map(|i| (i * 37) % vocab as u32).collect();
    Inputs {
        x: t(&mut rng, &[rows, kin], 0.5),
        w: t(&mut rng, &[nout, kin], 0.5),
        gy: t(&mut rng, &[rows, nout], 0.5),
        rx: t(&mut rng, &[512, 128], 1.0),
        rw: t(&mut rng, &[128], 1.0),
        rgy: t(&mut rng, &[512, 128], 1.0),
        q: t(&mut rng, &sdpa, 1.0),
        sgy: t(&mut rng, &sdpa, 1.0),
        p: rng.vec(P_LEN, 0.1),
        g: t(&mut rng, &P_SHAPE, 0.1),
        m1: rng.vec(P_LEN, 0.01),
        m2: rng.vec(P_LEN, 0.01).iter().map(|v| v.abs()).collect(),
        logits: t(&mut rng, &[128, vocab], 2.0),
        targets: Tensor::from_u32(&targets, &[128], &b).unwrap(),
    }
}

fn outputs(cpu: &CpuBackend, inp: &Inputs) -> Vec<(&'static str, Vec<u32>)> {
    let f = |t: &Tensor| bits(&t.to_f32_vec().unwrap());
    let mut out = Vec::new();
    out.push((
        "linear_fwd",
        f(&cpu.linear_forward(&inp.x, &inp.w).unwrap()),
    ));
    let (gx, gw) = cpu.linear_backward(&inp.x, &inp.w, &inp.gy).unwrap();
    out.push(("linear_gx", f(&gx)));
    out.push(("linear_gw", f(&gw)));
    out.push((
        "rms_fwd",
        f(&cpu
            .rms_norm_forward(&inp.rx, &inp.rw, RMS_NORM_EPS)
            .unwrap()),
    ));
    let (rgx, rgw) = cpu
        .rms_norm_backward(&inp.rx, &inp.rw, &inp.rgy, RMS_NORM_EPS)
        .unwrap();
    out.push(("rms_gx", f(&rgx)));
    out.push(("rms_gw", f(&rgw)));
    out.push((
        "sdpa_fwd",
        f(&cpu
            .causal_sdpa_forward(&inp.q, &inp.q, &inp.q, None)
            .map(|(y, _)| y)
            .unwrap()),
    ));
    let (sq, sk, sv) = cpu
        .causal_sdpa_backward_recompute(&inp.q, &inp.q, &inp.q, &inp.sgy, None)
        .unwrap();
    out.push(("sdpa_gq", f(&sq)));
    out.push(("sdpa_gk", f(&sk)));
    out.push(("sdpa_gv", f(&sv)));
    out.push((
        "ce_grad",
        f(&cpu
            .cross_entropy_mean_backward(&inp.logits, &inp.targets, None)
            .unwrap()),
    ));
    let b = cpu.budget();
    let mut p = Tensor::from_f32(&inp.p, &P_SHAPE, b).unwrap();
    let mut m1 = Tensor::from_f32(&inp.m1, &P_SHAPE, b).unwrap();
    let mut m2 = Tensor::from_f32(&inp.m2, &P_SHAPE, b).unwrap();
    cpu.adamw_step(
        &mut p,
        &inp.g,
        &mut m1,
        &mut m2,
        4,
        AdamWConfig::nanolab(1e-3, 0.1),
    )
    .unwrap();
    out.push(("adamw_p", f(&p)));
    out.push(("adamw_m1", f(&m1)));
    out.push(("adamw_m2", f(&m2)));
    out
}

fn backend(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 30), threads)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

/// `callers` threads share one `threads`-wide backend; the original handle
/// drops once they have all started.
fn concurrent_round(
    inp: &Inputs,
    want: &[(&'static str, Vec<u32>)],
    threads: usize,
    callers: usize,
    iters: usize,
) {
    let cpu = backend(threads);
    let start = Barrier::new(callers + 1);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for caller in 0..callers {
            let (mine, start) = (cpu.clone(), &start);
            handles.push(
                std::thread::Builder::new()
                    .stack_size(8 << 20)
                    .spawn_scoped(scope, move || {
                        start.wait();
                        for iter in 0..iters {
                            let got = outputs(&mine, inp);
                            assert_eq!(got.len(), want.len());
                            for ((name, want), (_, got)) in want.iter().zip(got) {
                                assert!(
                                    *want == got,
                                    "{name}: {threads} threads, caller {caller}, iter {iter} \
                                     differs from 1 thread"
                                );
                            }
                        }
                    })
                    .expect("caller spawn"),
            );
        }
        start.wait();
        drop(cpu);
        for handle in handles {
            handle.join().expect("caller failed");
        }
    });
}

/// Run on a watchdog thread so a deadlock fails the test instead of hanging it.
fn within(secs: u64, body: impl FnOnce() + Send + 'static) {
    let (done, wait) = mpsc::channel();
    let runner = std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            body();
            let _ = done.send(());
        })
        .expect("watchdog spawn");
    match wait.recv_timeout(Duration::from_secs(secs)) {
        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => runner.join().expect("stress body"),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("pool stress did not finish in {secs}s (deadlock or lost wake-up)")
        }
    }
}

fn run(rounds: usize, callers: usize, iters: usize, secs: u64) {
    within(secs, move || {
        let inp = inputs();
        let want = outputs(&backend(1), &inp);
        for _ in 0..rounds {
            for threads in 1..=18 {
                concurrent_round(&inp, &want, threads, callers, iters);
            }
        }
    });
}

#[test]
fn exact_bits_from_concurrent_callers_match_one_thread_at_every_width() {
    run(1, 3, 1, 300);
}

/// 50 rounds at every width, more callers and iterations.
/// `cargo test -p ojas-cpu --release --test pool_stress -- --ignored`.
#[test]
#[ignore = "soak: about 12 s in release; run explicitly"]
fn exact_bits_soak() {
    run(50, 6, 2, 7200);
}

/// `start_workers` spawns the pool up front, is idempotent, and leaves the
/// pool computing the same bits as one started lazily; a serial backend has
/// nothing to start.
#[test]
fn starting_workers_early_is_idempotent_and_changes_no_bits() {
    CpuBackend::new(Budget::new(1 << 20))
        .start_workers()
        .unwrap();
    let eager = CpuBackend::with_threads(Budget::new(1 << 30), 4)
        .unwrap()
        .with_numerics(Numerics::Exact);
    eager.start_workers().unwrap();
    eager.start_workers().unwrap();
    let lazy = CpuBackend::with_threads(Budget::new(1 << 30), 4)
        .unwrap()
        .with_numerics(Numerics::Exact);
    let mut rng = SplitMix64(7);
    let (rows, dim) = (256, 512);
    let x: Vec<f32> = (0..rows * dim).map(|_| rng.unit()).collect();
    let w: Vec<f32> = (0..dim).map(|_| rng.unit()).collect();
    let run = |cpu: &CpuBackend| {
        let x = Tensor::from_f32(&x, &[rows, dim], cpu.budget()).unwrap();
        let w = Tensor::from_f32(&w, &[dim], cpu.budget()).unwrap();
        let y = cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap();
        bits(&y.to_f32_vec().unwrap())
    };
    assert_eq!(run(&eager), run(&lazy));
}
