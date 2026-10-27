//! Day-0 kill check of the TensorOps cross-attention work: the effective rate of
//! the existing flash forward `ojas_attn_fwd_d32` (`Backend::causal_sdpa_forward`,
//! windowless) at large T, with tessl's bidirectional `encoder_attn`
//! (`attention_forward` at Tq = Tk) as the engine reference and the fc1 linear as
//! the control. The rules are `bench/results/2026-10-27-attention-mpp/rules.md`.
//!
//! `cargo run -p ojas-metal --release --example attention_causal_rate_bench [warmups] [runs] [only]`
//!
//! Defaults 5 and 25; `only` is a substring of a row name. Rows give p10, median
//! and p90 of the A pass (ms), the A' median, the A-vs-A' gap and the pair's
//! spread, `ok` when the gap is within the spread, and the effective TFLOP/s at
//! the median: useful work over time, with 4 H D T (T + 1) / 2 flops for the
//! causal arm (the visible pairs) and 4 H D T^2 for `encoder_attn`. The 1-minute
//! load average is printed at the start and the end, the last lines are the kill
//! check and the verdict (`clean` or `repeat`).

#[cfg(target_os = "macos")]
mod bench {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{AttentionOpts, Backend, Budget, Tensor};
    use ojas_metal::MetalBackend;

    const MAX_LOAD: f64 = 6.0;
    const CONTROL_TFLOPS: (f64, f64) = (6.3 * 0.97, 6.3 * 1.03);
    /// The kill rule's rate and the T it is read at.
    const KILL_TFLOPS: f64 = 4.3;
    const KILL_T: usize = 15584;
    const H: usize = 8;
    const D: usize = 32;
    const TS: [usize; 3] = [5184, 15584, 36320];

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

    fn pct(v: &[f64], q: f64) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[((s.len() - 1) as f64 * q).round() as usize]
    }

    type Arm<'a> = (&'static str, f64, Box<dyn Fn() + 'a>);

    /// Every round: each arm once, then each again; odd rounds reverse the arm
    /// order. Prints one line per arm; returns each arm's A-pass median and
    /// the number of NOISY pairs.
    fn run_row(
        metal: &MetalBackend,
        label: &str,
        arms: &[Arm<'_>],
        warmups: usize,
        runs: usize,
    ) -> (Vec<f64>, usize) {
        let count = arms.len();
        let mut samples: Vec<[Vec<f64>; 2]> =
            (0..count).map(|_| [Vec::new(), Vec::new()]).collect();
        for round in 0..warmups + runs {
            for pass in [0usize, 1] {
                for step in 0..count {
                    let i = if round % 2 == 0 {
                        step
                    } else {
                        count - 1 - step
                    };
                    metal.sync().expect("sync");
                    let start = Instant::now();
                    (arms[i].2)();
                    metal.sync().expect("sync");
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    if round >= warmups {
                        samples[i][pass].push(ms);
                    }
                }
            }
        }
        let (mut medians, mut noisy) = (Vec::new(), 0);
        for ((name, flops, _), s) in arms.iter().zip(&samples) {
            let (ma, mb) = (pct(&s[0], 0.5), pct(&s[1], 0.5));
            let spread = (pct(&s[0], 0.9) - pct(&s[0], 0.1)).max(pct(&s[1], 0.9) - pct(&s[1], 0.1));
            let gap = (ma - mb).abs();
            let ok = gap <= spread;
            if !ok {
                noisy += 1;
            }
            println!(
                "{label}\t{name}\t{:.3}\t{ma:.3}\t{:.3}\t{mb:.3}\t{gap:.3}\t{spread:.3}\t{}\t{:.2} TFLOP/s",
                pct(&s[0], 0.1),
                pct(&s[0], 0.9),
                if ok { "ok" } else { "NOISY" },
                flops / (ma * 1e-3) / 1e12
            );
            medians.push(ma);
        }
        (medians, noisy)
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let num = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
        let (warmups, runs) = (num(0, 5), num(1, 25));
        let only = args.get(2).cloned().unwrap_or_default();
        let budget = Budget::new(24 << 30);
        let metal = MetalBackend::new(budget.clone()).expect("Metal device");
        assert!(
            metal.has_tensorops(),
            "the TensorOps kernels need TensorOps"
        );
        let host =
            |data: &[f32], shape: &[usize]| Tensor::from_f32(data, shape, &budget).expect("host");
        let up = |t: &Tensor| metal.upload(t).expect("upload");
        let start_load = load();
        println!(
            "# load_start {:.2} {:.2} {:.2}",
            start_load[0], start_load[1], start_load[2]
        );
        println!(
            "row\tarm\tp10_ms\tmedian_ms\tp90_ms\tmedian_ms_A'\tgap_ms\tspread_ms\tpair\teffective"
        );
        let mut noisy_total = 0;
        let mut control_ok = true;
        let mut kill_rate: Option<f64> = None;

        let (cm, ck, cn) = (5184usize, 1024usize, 4736usize);
        {
            let x = up(&host(&values(cm * ck, 1, 1.0), &[cm, ck]));
            let w = up(&host(&values(cn * ck, 2, 0.05), &[cn, ck]));
            let flops = 2.0 * cm as f64 * ck as f64 * cn as f64;
            let arms: Vec<Arm<'_>> = vec![(
                "fc1 linear f32 (control)",
                flops,
                Box::new(|| drop(metal.linear_forward(&x, &w).expect("fc1"))),
            )];
            let (med, noisy) = run_row(
                &metal,
                "control fc1 [5184,1024]x[4736,1024]",
                &arms,
                warmups,
                runs,
            );
            noisy_total += noisy;
            let tflops = flops / (med[0] * 1e-3) / 1e12;
            if !(CONTROL_TFLOPS.0..=CONTROL_TFLOPS.1).contains(&tflops) {
                println!(
                    "# control {tflops:.2} TFLOP/s outside {:.2} to {:.2}",
                    CONTROL_TFLOPS.0, CONTROL_TFLOPS.1
                );
                control_ok = false;
            }
        }

        for (ti, &t) in TS.iter().enumerate() {
            let label = format!("self-attention [1,{H},{t},{D}]");
            if !label.contains(&only) {
                continue;
            }
            let n = H * t * D;
            let (q, k, v) = (
                up(&host(&values(n, 10 + ti as u64, 1.0), &[1, H, t, D])),
                up(&host(&values(n, 20 + ti as u64, 1.0), &[1, H, t, D])),
                up(&host(&values(n, 30 + ti as u64, 1.0), &[1, H, t, D])),
            );
            let causal_flops = 4.0 * (H * D) as f64 * (t as f64) * (t as f64 + 1.0) / 2.0;
            let full_flops = 4.0 * (H * D) as f64 * (t as f64) * (t as f64);
            let count = |f: &dyn Fn()| {
                metal.sync().expect("sync");
                let before = metal.dispatches();
                f();
                metal.sync().expect("sync");
                metal.dispatches() - before
            };
            let causal = || drop(metal.causal_sdpa_forward(&q, &k, &v, None).expect("causal"));
            let encoder = || {
                drop(
                    metal
                        .attention_forward(
                            &q,
                            &k,
                            &v,
                            AttentionOpts {
                                causal: false,
                                window: None,
                                bias: None,
                                key_lens: None,
                            },
                        )
                        .expect("encoder_attn"),
                )
            };
            println!(
                "# {label} dispatches per call: causal {} encoder_attn route {}",
                count(&causal),
                count(&encoder)
            );
            let arms: Vec<Arm<'_>> = vec![
                ("causal ojas_attn_fwd_d32", causal_flops, Box::new(causal)),
                (
                    "encoder_attn (Tq = Tk, bidirectional)",
                    full_flops,
                    Box::new(encoder),
                ),
            ];
            let (med, noisy) = run_row(&metal, &label, &arms, warmups, runs);
            noisy_total += noisy;
            if t == KILL_T {
                kill_rate = Some(causal_flops / (med[0] * 1e-3) / 1e12);
            }
        }

        let end_load = load();
        println!(
            "# load_end {:.2} {:.2} {:.2}",
            end_load[0], end_load[1], end_load[2]
        );
        let loads_ok = start_load[0] <= MAX_LOAD && end_load[0] <= MAX_LOAD;
        println!("# noisy_pairs {noisy_total}");
        let clean = loads_ok && noisy_total == 0 && control_ok;
        if let Some(rate) = kill_rate {
            println!(
                "# kill-check T = {KILL_T}: causal ojas_attn_fwd_d32 {rate:.2} TFLOP/s effective against {KILL_TFLOPS} -> {} (read on a clean run only)",
                if rate >= KILL_TFLOPS { "passes" } else { "(b) dies" }
            );
        }
        println!("# verdict {}", if clean { "clean" } else { "repeat" });
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("attention_causal_rate_bench needs macOS");
}
