//! ft-1185100b adversarial bench: the cost a CALLER pays for `ToyProvider::adamw_step`, the same
//! file run at dd41efd (one pass per entry) and at b36c2fb (check pass + update pass).
//! Prints `BENCH n=2^k min-of-5 ms <x>` lines; the driver script interleaves the two binaries.
//! Run: `cargo test --release -p ojas-train --test host_adamw_bench -- --ignored --nocapture`.

use std::time::Instant;

use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::recipe::{BETA1, EPS, WEIGHT_DECAY};
use ojas_train::lappi::step::{AdamWHyper, StepProvider};

#[test]
#[ignore = "timing; run by hand"]
fn host_adamw_step_cost() {
    const HIDDEN: usize = 64;
    for k in [16u32, 20, 24] {
        let n = 1usize << k;
        let mut p = ToyProvider::new(n / HIDDEN, HIDDEN, 0, 9).expect("toy");
        let entries = p.parameters().len();
        for (i, g) in p.grads_mut().iter_mut().enumerate() {
            for (j, x) in g.iter_mut().enumerate() {
                *x = (((i + j) % 977) as f32 - 488.0) * 1e-3;
            }
        }
        let h = AdamWHyper {
            lr: 1e-3,
            beta1: BETA1,
            beta2: 0.999,
            eps: EPS,
            grad_scale: 1.0,
        };
        let ones = vec![1.0; entries];
        let wd = vec![WEIGHT_DECAY; entries];
        let mut best = f64::MAX;
        for step in 1..=6u64 {
            let t = Instant::now();
            p.adamw_step(&h, step, &ones, &wd).expect("step");
            let dt = t.elapsed().as_secs_f64() * 1e3;
            // The first step touches cold moment pages; keep it out of the minimum.
            if step > 1 {
                best = best.min(dt);
            }
        }
        println!("BENCH n=2^{k} min-of-5 ms {best:.4}");
    }
}
