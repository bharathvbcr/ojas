//! What does a host <-> device copy cost on this machine's unified memory,
//! against the host-memory floor? `MetalBackend::upload` and `Tensor::to_host`
//! on one f32 tensor of 1, 16, 66 and 170 MB (decimal; the sizes of SpikeTrack's
//! mask uploads, its readbacks and its large frame tensors), beside two plain
//! host arms:
//!
//! - `memcpy`: `copy_from_slice` of the tensor's bytes into a warm, reused
//!   buffer. The floor for one pass over the data.
//! - `clone`: `to_vec` of the same bytes, a fresh allocation plus one pass.
//!   The floor for a readback that returns a new host buffer.
//! - `upload`: `MetalBackend::upload` of the host tensor. The result is
//!   dropped after the timer stops.
//! - `to_host`: `Tensor::to_host` of a device tensor uploaded once before the
//!   timed rounds. The result is dropped after the timer stops.
//!
//! `cargo run -p ojas-metal --release --example readback_bench [warmups] [runs] [only]`
//!
//! Defaults 3 and 15; `only` is a size in MB (`16`). A round runs every arm
//! once; odd rounds walk the arms in reverse, so no arm is always first or
//! last. Rows give the minimum, median and p90 of `runs` timed rounds in
//! milliseconds, GB/s at the minimum and the median (the bytes of one pass over
//! the tensor), and `x_clone`, the arm's minimum over the `clone` arm's minimum:
//! how many times the floor for a fresh host result the copy costs. Before any
//! timing, one `to_host` of each size is compared bit for bit with the tensor
//! that was uploaded, and the example stops if a bit differs.
//!
//! The 1-minute load average (`sysctl vm.loadavg`) is printed at the start and
//! the end, and the last line is the verdict: `clean` when both loads are at
//! most `MAX_LOAD`, else `repeat`. A run under load is a sample, not a result.
//!
//! The GPU streaming-bandwidth arm of the unify plan is not here; the example
//! measures the copies, not the device.

#[cfg(target_os = "macos")]
mod bench {
    use std::hint::black_box;
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{Backend, Budget, Tensor};
    use ojas_metal::MetalBackend;

    /// The 1-minute load above which a run is repeated.
    const MAX_LOAD: f64 = 6.0;
    /// Tensor sizes in decimal megabytes.
    const SIZES_MB: [usize; 4] = [1, 16, 66, 170];
    /// Budget of the backend: a size's host tensor, device tensor, upload
    /// result and readback are all live at once, about 4 x 170 MB at most.
    const BUDGET_BYTES: u64 = 8 << 30;

    #[derive(Clone, Copy, PartialEq)]
    enum Arm {
        Memcpy,
        ToVec,
        Upload,
        ToHost,
    }

    const ARMS: [Arm; 4] = [Arm::Memcpy, Arm::ToVec, Arm::Upload, Arm::ToHost];

    impl Arm {
        fn name(self) -> &'static str {
            match self {
                Arm::Memcpy => "memcpy",
                Arm::ToVec => "clone",
                Arm::Upload => "upload",
                Arm::ToHost => "to_host",
            }
        }
    }

    /// `(1-minute, 5-minute, 15-minute)` load averages from `sysctl vm.loadavg`.
    fn load() -> [f64; 3] {
        let out = Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .expect("sysctl vm.loadavg");
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let v: Vec<f64> = text
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        assert!(v.len() >= 3, "unparsable vm.loadavg: {text:?}");
        [v[0], v[1], v[2]]
    }

    /// Finite values in [-1, 1) with varied bits, so no copy can be skipped
    /// as all zero and no check refuses a NaN.
    fn values(n: usize) -> Vec<f32> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 40) as f32 / (1u64 << 24) as f32;
                2.0 * u - 1.0
            })
            .collect()
    }

    fn quantile(sorted: &[f64], q: f64) -> f64 {
        let i = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[i]
    }

    struct Case {
        src: Vec<f32>,
        dst: Vec<f32>,
        host: Tensor,
        dev: Tensor,
    }

    /// One timed run of `arm`, in milliseconds. Whatever the arm allocates is
    /// dropped after the clock stops.
    fn time_arm(arm: Arm, be: &MetalBackend, case: &mut Case) -> f64 {
        match arm {
            Arm::Memcpy => {
                let t = Instant::now();
                case.dst.copy_from_slice(&case.src);
                black_box(&case.dst);
                t.elapsed().as_secs_f64() * 1e3
            }
            Arm::ToVec => {
                let t = Instant::now();
                let v = case.src.to_vec();
                black_box(&v);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                drop(v);
                ms
            }
            Arm::Upload => {
                let t = Instant::now();
                let d = be.upload(&case.host).expect("upload");
                let ms = t.elapsed().as_secs_f64() * 1e3;
                black_box(&d);
                drop(d);
                ms
            }
            Arm::ToHost => {
                let t = Instant::now();
                let h = case.dev.to_host(be.budget()).expect("to_host");
                let ms = t.elapsed().as_secs_f64() * 1e3;
                black_box(&h);
                drop(h);
                ms
            }
        }
    }

    fn run_size(be: &MetalBackend, mb: usize, warmups: usize, runs: usize) {
        let n = mb * 1_000_000 / 4;
        let bytes = (n * 4) as f64;
        let src = values(n);
        let host = Tensor::from_f32(&src, &[n], be.budget()).expect("host tensor");
        let dev = be.upload(&host).expect("upload");
        let back = dev.to_host(be.budget()).expect("to_host");
        let same = back
            .f32_slice()
            .expect("host slice")
            .iter()
            .zip(&src)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "{mb} MB: to_host differs from the uploaded tensor");
        drop(back);
        let mut case = Case {
            dst: vec![0.0; n],
            src,
            host,
            dev,
        };

        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(runs); ARMS.len()];
        for round in 0..warmups + runs {
            let mut order: Vec<usize> = (0..ARMS.len()).collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for i in order {
                let ms = time_arm(ARMS[i], be, &mut case);
                if round >= warmups {
                    samples[i].push(ms);
                }
            }
        }

        let stats: Vec<[f64; 3]> = samples
            .iter_mut()
            .map(|s| {
                s.sort_by(|a, b| a.total_cmp(b));
                [s[0], quantile(s, 0.5), quantile(s, 0.9)]
            })
            .collect();
        let clone_min = stats[1][0];
        for (i, arm) in ARMS.iter().enumerate() {
            let [min, med, p90] = stats[i];
            println!(
                "{mb:>4} MB  {:<8} min {min:>9.3} ms  med {med:>9.3} ms  p90 {p90:>9.3} ms  \
                 {:>6.2} GB/s @min  {:>6.2} GB/s @med  x_clone {:>5.2}",
                arm.name(),
                bytes / min / 1e6,
                bytes / med / 1e6,
                min / clone_min,
            );
        }
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let warmups: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
        let runs: usize = args
            .get(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(15)
            .max(1);
        let only = args.get(3).cloned();

        let be = MetalBackend::new(Budget::new(BUDGET_BYTES)).expect("open the Metal device");
        let start_load = load();
        println!("# device {}", be.device_name());
        println!(
            "# load_start {:.2} {:.2} {:.2}",
            start_load[0], start_load[1], start_load[2]
        );
        println!(
            "# warmups {warmups} runs {runs} (decimal MB, f32, one pass = the tensor's bytes)"
        );
        for mb in SIZES_MB {
            if only
                .as_deref()
                .is_some_and(|o| o.parse::<usize>().ok() != Some(mb))
            {
                continue;
            }
            run_size(&be, mb, warmups, runs);
        }
        let end_load = load();
        println!(
            "# load_end {:.2} {:.2} {:.2}",
            end_load[0], end_load[1], end_load[2]
        );
        println!(
            "# verdict {}",
            if start_load[0] <= MAX_LOAD && end_load[0] <= MAX_LOAD {
                "clean"
            } else {
                "repeat"
            }
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("readback_bench needs macOS");
}
