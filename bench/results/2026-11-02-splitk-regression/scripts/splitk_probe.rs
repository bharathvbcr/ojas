//! Split-K regression probe (bench/results/2026-11-02-splitk-regression).
//!
//! One fixed source, copied into every bisect build's `ojas-metal/examples/`,
//! so only the library differs between binaries. Times three arms at
//! q [1,8,5184,32] x k,v [1,8,36320,32], f32, no bias, non-causal:
//! - `split-K`: `attention_forward_saving_splitk` (the arm under test);
//! - `dispatcher`: `attention_forward_saving` (the same shape on the route the
//!   dispatcher picks; a same-process control at the same shape);
//! - `fc1`: `linear_forward` f32 [5184,1024] x [4736,1024] (the GEMM control).
//! Each call has a device sync on both sides. Odd rounds walk the arms in
//! reverse. Usage: `splitk_probe <warmups> <runs> [tag]`.

#[cfg(target_os = "macos")]
mod bench {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{AttentionOpts, Backend, Budget, Tensor};
    use ojas_metal::MetalBackend;

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
        let metal = MetalBackend::new(budget.clone()).expect("Metal device");
        let host =
            |data: &[f32], shape: &[usize]| Tensor::from_f32(data, shape, &budget).expect("host");
        let up = |t: &Tensor| metal.upload(t).expect("upload");
        let load_start = load();

        let (b, h, tq, tk, d) = (1usize, 8usize, 5184usize, 36320usize, 32usize);
        let q = up(&host(&values(b * h * tq * d, 101, 1.0), &[b, h, tq, d]));
        let k = up(&host(&values(b * h * tk * d, 102, 1.0), &[b, h, tk, d]));
        let v = up(&host(&values(b * h * tk * d, 103, 1.0), &[b, h, tk, d]));
        let (cm, ck, cn) = (5184usize, 1024usize, 4736usize);
        let x = up(&host(&values(cm * ck, 1, 1.0), &[cm, ck]));
        let w = up(&host(&values(cn * ck, 2, 0.05), &[cn, ck]));
        let opts = || AttentionOpts {
            causal: false,
            ..Default::default()
        };

        // Correctness before timing: split-K against the dispatcher's route.
        let (a, _) = metal
            .attention_forward_saving_splitk(&q, &k, &v, opts())
            .expect("split-K");
        let (r, _) = metal
            .attention_forward_saving(&q, &k, &v, opts())
            .expect("dispatcher");
        metal.sync().expect("sync");
        let av = a.to_host(&budget).expect("host").to_f32_vec().expect("f32");
        let rv = r.to_host(&budget).expect("host").to_f32_vec().expect("f32");
        let max_abs = av
            .iter()
            .zip(&rv)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        drop((a, r));

        let arms: Vec<Arm<'_>> = vec![
            (
                "fc1",
                Box::new(|| drop(metal.linear_forward(&x, &w).expect("fc1"))),
            ),
            (
                "split-K",
                Box::new(|| {
                    drop(
                        metal
                            .attention_forward_saving_splitk(&q, &k, &v, opts())
                            .expect("split-K"),
                    )
                }),
            ),
            (
                "dispatcher",
                Box::new(|| {
                    drop(
                        metal
                            .attention_forward_saving(&q, &k, &v, opts())
                            .expect("dispatcher"),
                    )
                }),
            ),
        ];
        let mut ms: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..warmups + runs {
            for step in 0..arms.len() {
                let i = if round % 2 == 0 {
                    step
                } else {
                    arms.len() - 1 - step
                };
                metal.sync().expect("sync");
                let t = Instant::now();
                (arms[i].1)();
                metal.sync().expect("sync");
                if round >= warmups {
                    ms[i].push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        let load_end = load();
        for (i, arm) in arms.iter().enumerate() {
            let mut s = ms[i].clone();
            s.sort_by(f64::total_cmp);
            println!(
                "probe\t{tag}\t{}\tp10 {:.3}\tmedian {:.3}\tp90 {:.3}\tn {}\tmaxabs_vs_dispatcher {:.2e}\tload_start {}\tload_end {}",
                arm.0,
                pct(&s, 0.1),
                pct(&s, 0.5),
                pct(&s, 0.9),
                s.len(),
                max_abs,
                load_start,
                load_end
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("splitk_probe needs macOS and Metal");
}
