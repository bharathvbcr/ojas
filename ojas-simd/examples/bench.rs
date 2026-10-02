//! Single-call GFLOP/s of each backend.
//!
//! ```text
//! cargo run --release -p ojas-simd --example bench [--features accelerate]
//! ```
//!
//! `sgemm_tile` runs on the calling thread. Accelerate threads internally;
//! set `VECLIB_MAXIMUM_THREADS=1` to measure it on one thread.

use std::time::{Duration, Instant};

use ojas_simd::Backend;

fn fill(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// Best wall time of repeated calls, after one warm-up, within a time budget.
fn best(mut f: impl FnMut()) -> Duration {
    f();
    let budget = Instant::now();
    let mut best = Duration::MAX;
    let mut reps = 0;
    while reps < 3 || (budget.elapsed() < Duration::from_secs(2) && reps < 50) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        reps += 1;
    }
    best
}

fn main() {
    let shapes = [
        (256usize, 256usize, 256usize),
        (512, 768, 768),
        (2048, 2048, 2048),
    ];
    println!("detected backend: {}", ojas_simd::backend_name());
    println!(
        "{:<14} {:>16} {:>10} {:>10}",
        "backend", "m x n x k", "ms", "GFLOP/s"
    );
    for &(m, n, k) in &shapes {
        let a = fill(m * k, 1);
        let b = fill(k * n, 2);
        let mut c = vec![0.0f32; m * n];
        let flops = 2.0 * (m * n * k) as f64;
        let shape = format!("{m}x{n}x{k}");
        for bk in Backend::ALL.into_iter().filter(|b| b.is_available()) {
            let t = best(|| {
                ojas_simd::sgemm_tile_with(bk, m, n, k, &a, k, 1, &b, n, 1, &mut c, n, false)
                    .unwrap();
            });
            let s = t.as_secs_f64();
            println!(
                "{:<14} {:>16} {:>10.3} {:>10.1}",
                bk.name(),
                shape,
                s * 1e3,
                flops / s / 1e9
            );
        }
        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            let t = best(|| {
                ojas_simd::sgemm_accelerate(m, n, k, &a, k, 1, &b, n, 1, &mut c, n, false).unwrap();
            });
            let s = t.as_secs_f64();
            println!(
                "{:<14} {:>16} {:>10.3} {:>10.1}",
                "accelerate",
                shape,
                s * 1e3,
                flops / s / 1e9
            );
        }
    }
}
