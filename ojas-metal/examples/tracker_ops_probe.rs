//! SpikeTrack memory-attention small ops, timed one by one
//! (bench/results/2026-11-03-tracker-ops-baseline).
//!
//! `tracker_ops_probe <warmups> <runs> <tag>` times each arm with a device
//! sync on both sides of every call, odd rounds in reverse order, and prints
//! per arm the min, p10, median and p90 in ms, the dispatches one call
//! encoded, and the permute and finite-check pipelines among them. The arms
//! are the shapes of `SpikeTrack/bench/results/ab-coreml-tracker/run2-ops`:
//!
//! - `perm_head36k`: `permute` `[1,36400,8,32]` by `[0,2,1,3]`, the head swap
//!   of the memory keys and values (`memory_attention.rs` `to_bhtd`);
//! - `perm_ident36k`: the identity permute of the same tensor, a straight
//!   copy through the same planner (the copy-speed control for the head swap);
//! - `perm_generic36k`: the head swap on the generic one-thread-per-element
//!   kernel (`permute_generic`);
//! - `perm_head5k`: the head swap at the current frame's 5,184 tokens;
//! - `resadd36k`: `residual_add_forward` `[1,36400,256]` + `[1,36400,256]`,
//!   a bandwidth reference (two reads and a write of the same size);
//! - `lin36k`, `linb36k`, `badd36k`: `linear_forward` and
//!   `linear_bias_forward` `[36400,256] x [256,256]`, and the bias add alone
//!   (`broadcast_add_forward` `[36400,256] + [256]`): `img_k`, `ca_k`, `ca_v`;
//! - `lin5k_up`, `linb5k_up`: the same at the FFN's `[5184,256] x [2048,256]`.

#[cfg(target_os = "macos")]
mod probe {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{Backend, Budget, Tensor};
    use ojas_metal::MetalBackend;

    const KERNELS: [&str; 4] = [
        "ojas_permute_runs",
        "ojas_permute",
        "ojas_permute_tile",
        "ojas_check_finite",
    ];

    fn load() -> String {
        let out = Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .expect("sysctl vm.loadavg");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .trim_matches(|c| c == '{' || c == '}')
            .trim()
            .to_string()
    }

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

    fn pct(sorted: &[f64], p: f64) -> f64 {
        let i = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[i]
    }

    type Arm<'a> = (&'static str, Box<dyn Fn() + 'a>);

    pub fn run() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let num = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
        let (warmups, runs) = (num(0, 3), num(1, 15));
        let tag = args.get(2).cloned().unwrap_or_default();
        let budget = Budget::new(24 << 30);
        let m = MetalBackend::new(budget.clone()).expect("Metal device");
        let up = |shape: &[usize], seed: u64, scale: f32| {
            let n = shape.iter().product();
            let t = Tensor::from_f32(&values(n, seed, scale), shape, &budget).expect("host");
            m.upload(&t).expect("upload")
        };
        let load_start = load();

        let heads36 = up(&[1, 36400, 8, 32], 1, 1.0);
        let heads5 = up(&[1, 5184, 8, 32], 2, 1.0);
        let a36 = up(&[1, 36400, 256], 3, 1.0);
        let b36 = up(&[1, 36400, 256], 4, 1.0);
        let x36 = up(&[36400, 256], 5, 1.0);
        let w256 = up(&[256, 256], 6, 0.05);
        let bias256 = up(&[256], 7, 0.1);
        let x5 = up(&[5184, 256], 8, 1.0);
        let w2048 = up(&[2048, 256], 9, 0.05);
        let bias2048 = up(&[2048], 10, 0.1);
        let swap = [0usize, 2, 1, 3];
        let ident = [0usize, 1, 2, 3];

        let arms: Vec<Arm<'_>> = vec![
            (
                "perm_head36k",
                Box::new(|| drop(m.permute(&heads36, &swap).expect("perm"))),
            ),
            (
                "perm_ident36k",
                Box::new(|| drop(m.permute(&heads36, &ident).expect("ident"))),
            ),
            (
                "perm_generic36k",
                Box::new(|| drop(m.permute_generic(&heads36, &swap).expect("generic"))),
            ),
            (
                "perm_head5k",
                Box::new(|| drop(m.permute(&heads5, &swap).expect("perm5"))),
            ),
            (
                "resadd36k",
                Box::new(|| drop(m.residual_add_forward(&a36, &b36).expect("resadd"))),
            ),
            (
                "lin36k",
                Box::new(|| drop(m.linear_forward(&x36, &w256).expect("lin"))),
            ),
            (
                "linb36k",
                Box::new(|| drop(m.linear_bias_forward(&x36, &w256, &bias256).expect("linb"))),
            ),
            (
                "badd36k",
                Box::new(|| drop(m.broadcast_add_forward(&x36, &bias256).expect("badd"))),
            ),
            (
                "lin5k_up",
                Box::new(|| drop(m.linear_forward(&x5, &w2048).expect("lin5"))),
            ),
            (
                "linb5k_up",
                Box::new(|| {
                    drop(
                        m.linear_bias_forward(&x5, &w2048, &bias2048)
                            .expect("linb5"),
                    )
                }),
            ),
        ];
        let counts = || {
            let mut v = vec![m.dispatches()];
            v.extend(KERNELS.iter().map(|k| m.kernel_dispatches(k)));
            v
        };
        let mut ms: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut per_call: Vec<Vec<u64>> = vec![Vec::new(); arms.len()];
        for round in 0..warmups + runs {
            for step in 0..arms.len() {
                let i = if round % 2 == 0 {
                    step
                } else {
                    arms.len() - 1 - step
                };
                m.sync().expect("sync");
                let before = counts();
                let t = Instant::now();
                (arms[i].1)();
                m.sync().expect("sync");
                let dt = t.elapsed().as_secs_f64() * 1e3;
                let after = counts();
                if round == 0 {
                    per_call[i] = after.iter().zip(&before).map(|(a, b)| a - b).collect();
                }
                if round >= warmups {
                    ms[i].push(dt);
                }
            }
        }
        let load_end = load();
        for (i, arm) in arms.iter().enumerate() {
            let mut s = ms[i].clone();
            s.sort_by(f64::total_cmp);
            println!(
                "ops\t{tag}\t{}\tmin {:.3}\tp10 {:.3}\tmedian {:.3}\tp90 {:.3}\tn {}\tdispatches,runs,generic,tile,check {:?}\tload_start {}\tload_end {}",
                arm.0,
                s[0],
                pct(&s, 0.1),
                pct(&s, 0.5),
                pct(&s, 0.9),
                s.len(),
                per_call[i],
                load_start,
                load_end
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    probe::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("tracker_ops_probe needs macOS and Metal");
}
