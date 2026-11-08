//! `adamw_step` on one parameter of `2^20` elements: the per-call cost
//! including the host-side checks it makes before recording.
//!
//! `cargo run -p ojas-wgpu --release --example adamw_step_bench`
//!
//! Prints `adamw_step_us <min over runs of the mean per-call time>`. Each run
//! records `STEPS` calls and syncs once at the end, so the sync is spread over
//! the calls and the number is the training-path cost, not a round trip.

use std::time::Instant;

use ojas_core::{AdamWConfig, BackendCore, BackendFloat, BackendModule, Budget, Tensor};
use ojas_wgpu::WgpuBackend;

const N: usize = 1 << 20;
const STEPS: usize = 50;
const RUNS: usize = 5;

fn main() {
    let budget = Budget::new(1 << 30);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let up = |v: f32| {
        gpu.upload(&Tensor::from_f32(&vec![v; N], &[N], &budget).unwrap())
            .unwrap()
    };
    let (mut p, g, mut m, mut v) = (up(1.0), up(0.5), up(0.0), up(0.0));
    let cfg = AdamWConfig::nanolab(1e-3, 0.1);
    for step in 1..=5u64 {
        gpu.adamw_step(&mut p, &g, &mut m, &mut v, step, cfg)
            .unwrap();
    }
    gpu.sync().unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..RUNS {
        let t = Instant::now();
        for step in 0..STEPS as u64 {
            gpu.adamw_step(&mut p, &g, &mut m, &mut v, 6 + step, cfg)
                .unwrap();
        }
        gpu.sync().unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e6 / STEPS as f64);
    }
    println!("adamw_step_us {best:.2}");
}
