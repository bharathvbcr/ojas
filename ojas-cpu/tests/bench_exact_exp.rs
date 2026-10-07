//! Wall time of the CPU ops whose exponential is `exp_exact` (or was libm
//! `f32::exp` before it): exact attention and cross-entropy, exact SiLU
//! (`sigmoid`), and Fast attention at T 256, which takes the same softmax.
//! One thread, min of 15 after two warmup calls. Ignored by default. For an
//! A/B, build the binary at both commits and run them alternately:
//!
//! ```text
//! cargo test -p ojas-cpu --release --test bench_exact_exp -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{f32t, u32t, SplitMix64};

const REPS: usize = 15;

fn time(name: &str, mut f: impl FnMut() -> Result<(), OjasError>) {
    f().unwrap();
    f().unwrap();
    let mut best = Duration::MAX;
    for _ in 0..REPS {
        let start = Instant::now();
        f().unwrap();
        best = best.min(start.elapsed());
    }
    println!("OJAS_EXP {name} {:.1}", best.as_secs_f64() * 1e6);
}

fn tensor(cpu: &CpuBackend, rng: &mut SplitMix64, shape: &[usize], scale: f32) -> Tensor {
    f32t(cpu, &rng.vec(shape.iter().product(), scale), shape)
}

fn sdpa(cpu: &CpuBackend, label: &str, t: usize) {
    let mut rng = SplitMix64(7);
    let shape = [1, 12, t, 64];
    let q = tensor(cpu, &mut rng, &shape, 1.0);
    let k = tensor(cpu, &mut rng, &shape, 1.0);
    let v = tensor(cpu, &mut rng, &shape, 1.0);
    let gy = tensor(cpu, &mut rng, &shape, 0.01);
    time(&format!("{label}_sdpa_fwd_t{t}"), || {
        cpu.causal_sdpa_forward(&q, &k, &v, None)
            .map(|(y, _)| y)
            .map(drop)
    });
    time(&format!("{label}_sdpa_bwd_t{t}"), || {
        cpu.causal_sdpa_backward_recompute(&q, &k, &v, &gy, None)
            .map(drop)
    });
}

#[test]
#[ignore]
fn exact_exp_op_times() {
    let budget = Budget::new(1 << 32);
    let exact = CpuBackend::with_threads(budget.clone(), 1)
        .unwrap()
        .with_numerics(Numerics::Exact);
    let fast = CpuBackend::with_threads(budget, 1)
        .unwrap()
        .with_numerics(Numerics::Fast);

    sdpa(&exact, "exact", 256);
    sdpa(&fast, "fast", 256);

    let mut rng = SplitMix64(11);
    let (rows, vocab) = (64, 50304);
    let logits = tensor(&exact, &mut rng, &[rows, vocab], 4.0);
    let targets: Vec<u32> = (0..rows).map(|_| rng.below(vocab) as u32).collect();
    let targets = u32t(&exact, &targets, &[rows]);
    time("exact_ce_fwd", || {
        exact
            .cross_entropy_mean_forward(&logits, &targets, None)
            .map(drop)
    });
    time("exact_ce_bwd", || {
        exact
            .cross_entropy_mean_backward(&logits, &targets, None)
            .map(drop)
    });

    let x = tensor(&exact, &mut rng, &[1024, 2048], 4.0);
    time("exact_silu_fwd", || exact.silu_forward(&x).map(drop));
}
