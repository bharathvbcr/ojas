//! Release A/B of the panic guard on the inline path, which every small op
//! takes. `raw` is the pre-guard inline body, `guarded` is what ships. The two
//! run interleaved in one binary; each reports the minimum of `SAMPLES`
//! batches of `ITERS` calls, plus the spread of the batch times.
//!
//! `cargo test --release -p ojas-cpu --lib guard_bench -- --ignored --nocapture`

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use ojas_core::{Numerics, OjasError};

use super::{guarded, Exec, Pool};

const SAMPLES: usize = 9;
const ITERS: usize = 20_000;

fn per_call_ns(mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    t.elapsed().as_secs_f64() * 1e9 / ITERS as f64
}

fn report(what: &str, raw: &mut [f64], with: &mut [f64]) {
    raw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    with.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (r, g) = (raw[0], with[0]);
    println!(
        "{what}: raw min {r:.1} ns (max {:.1}), guarded min {g:.1} ns (max {:.1}), delta {:+.1} ns ({:+.1}%)",
        raw[SAMPLES - 1],
        with[SAMPLES - 1],
        g - r,
        (g - r) / r * 100.0
    );
}

#[test]
#[ignore = "release measurement"]
fn guard_overhead_on_the_inline_path() {
    let data: Arc<Vec<f32>> = Arc::new((0..64).map(|i| i as f32).collect());
    let pool = Arc::new(Pool::new(1).unwrap());
    let exec = Exec {
        pool: &pool,
        numerics: Numerics::Fast,
    };
    for count in [1usize, 8] {
        let (mut raw, mut with) = (Vec::new(), Vec::new());
        for _ in 0..SAMPLES {
            let d = Arc::clone(&data);
            raw.push(per_call_ns(|| {
                let d = Arc::clone(&d);
                let out: Vec<f32> = (0..count)
                    .map(|i| d.iter().sum::<f32>() + i as f32)
                    .collect();
                black_box(out);
            }));
            let d = Arc::clone(&data);
            with.push(per_call_ns(|| {
                let d = Arc::clone(&d);
                let out = exec.map(count, 0, 1, move |i| d.iter().sum::<f32>() + i as f32);
                black_box(out.unwrap());
            }));
        }
        report(
            &format!("Exec::map inline, {count} tasks"),
            &mut raw,
            &mut with,
        );
    }
    let (mut raw, mut with) = (Vec::new(), Vec::new());
    for _ in 0..SAMPLES {
        raw.push(per_call_ns(|| {
            let out: Result<Vec<f32>, OjasError> = (0..8)
                .map(|i| Ok(data.iter().sum::<f32>() + i as f32))
                .collect();
            black_box(out.unwrap());
        }));
        with.push(per_call_ns(|| {
            let out = guarded(|| {
                (0..8)
                    .map(|i| Ok(data.iter().sum::<f32>() + i as f32))
                    .collect::<Result<Vec<f32>, OjasError>>()
            });
            black_box(out.unwrap());
        }));
    }
    report(
        "scoped/fill_parts inline shape, 8 tasks",
        &mut raw,
        &mut with,
    );
}
