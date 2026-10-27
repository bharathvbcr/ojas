//! GroupNorm on SAM's pixel decoder shapes, `[1, 256, 144, 144]` and
//! `[1, 256, 288, 288]` with 8 groups and eps 1e-5, three ways:
//!
//! - (a) `MetalBackend::group_norm_forward` on device-resident x, weight and
//!   bias, a sync around the call only;
//! - (b) `CpuBackend::group_norm_forward` on the same values;
//! - (c) what SpikeTrack does without the op: download x, the CPU
//!   `group_norm_forward`, upload the result.
//!
//! `cargo run -p ojas-metal --release --example group_norm_bench [warmups] [runs]`
//!
//! Defaults 5 and 25. Each row prints the minimum and the median in
//! milliseconds.

#[cfg(target_os = "macos")]
mod bench {
    use std::time::Instant;

    use ojas_core::{Backend, Budget, Tensor};
    use ojas_cpu::CpuBackend;
    use ojas_metal::MetalBackend;

    const GROUPS: usize = 8;
    const EPS: f32 = 1e-5;

    fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 40) as f32 / (1u64 << 24) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect()
    }

    fn tensor(data: &[f32], shape: &[usize], budget: &Budget) -> Tensor {
        Tensor::from_f32(data, shape, budget).expect("host tensor")
    }

    /// Minimum and median milliseconds of `runs` timed calls after `warmups`.
    fn time(warmups: usize, runs: usize, mut f: impl FnMut()) -> (f64, f64) {
        for _ in 0..warmups {
            f();
        }
        let mut ms: Vec<f64> = (0..runs)
            .map(|_| {
                let start = Instant::now();
                f();
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        (ms[0], ms[ms.len() / 2])
    }

    pub fn run() {
        let args: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|a| a.parse().ok())
            .collect();
        let (warmups, runs) = (
            args.first().copied().unwrap_or(5),
            args.get(1).copied().unwrap_or(25),
        );
        let budget = Budget::new(16 << 30);
        let metal = MetalBackend::new(budget.clone()).expect("Metal device");
        let cpu = CpuBackend::new(budget.clone());
        println!("shape,path,min_ms,median_ms");
        for hw in [144usize, 288] {
            let shape = [1usize, 256, hw, hw];
            let x = tensor(&values(256 * hw * hw, 1, 3.0), &shape, &budget);
            let w = tensor(&values(256, 2, 2.0), &[256], &budget);
            let b = tensor(&values(256, 3, 1.0), &[256], &budget);
            let (dx, dw, db) = (
                metal.upload(&x).expect("upload x"),
                metal.upload(&w).expect("upload w"),
                metal.upload(&b).expect("upload b"),
            );
            metal.sync().expect("sync");
            let name = format!("[1,256,{hw},{hw}]");

            // The first call on this device, before any warm-up: pipeline use,
            // first allocations and the first command buffer.
            metal.sync().expect("sync");
            let first = Instant::now();
            drop(
                metal
                    .group_norm_forward(&dx, &dw, &db, GROUPS, EPS)
                    .expect("metal group_norm"),
            );
            metal.sync().expect("sync");
            println!(
                "{name},a metal first call,{:.3},",
                first.elapsed().as_secs_f64() * 1e3
            );
            let (a_min, a_med) = timed_cell(warmups, runs, || {
                metal.sync().expect("sync");
                let start = Instant::now();
                let y = metal
                    .group_norm_forward(&dx, &dw, &db, GROUPS, EPS)
                    .expect("metal group_norm");
                metal.sync().expect("sync");
                drop(y);
                start.elapsed().as_secs_f64() * 1e3
            });
            println!("{name},a metal device-resident,{a_min:.3},{a_med:.3}");

            let (b_min, b_med) = time(warmups, runs, || {
                drop(
                    cpu.group_norm_forward(&x, &w, &b, GROUPS, EPS)
                        .expect("cpu group_norm"),
                );
            });
            println!("{name},b cpu,{b_min:.3},{b_med:.3}");

            let (c_min, c_med) = time(warmups, runs, || {
                let host = metal.download(&dx).expect("download x");
                let y = cpu
                    .group_norm_forward(&host, &w, &b, GROUPS, EPS)
                    .expect("cpu group_norm");
                let up = metal.upload(&y).expect("upload y");
                metal.sync().expect("sync");
                drop(up);
            });
            println!("{name},c download + cpu + upload,{c_min:.3},{c_med:.3}");
        }
    }

    /// Like [`time`], for a closure that returns its own measured
    /// milliseconds (so the sync before the timer is not counted).
    fn timed_cell(warmups: usize, runs: usize, mut f: impl FnMut() -> f64) -> (f64, f64) {
        for _ in 0..warmups {
            f();
        }
        let mut ms: Vec<f64> = (0..runs).map(|_| f()).collect();
        ms.sort_by(f64::total_cmp);
        (ms[0], ms[ms.len() / 2])
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("group_norm_bench needs macOS");
}
