//! Stage 2 of the tiled general attention (the simdgroup-matrix partials kernel
//! `ojas_xattn_mma_part_f32_dD`) against the scalar split-K partials kernel it
//! replaced and tessl's pad-q route, at SpikeTrack's cross-attention shapes.
//!
//! `cargo run -p ojas-metal --release --example attention_stage2_bench [warmups] [runs] [only]`
//!
//! Defaults 5 and 25; `only` is a substring of a row name. The rules are
//! `bench/results/2026-10-27-attention-stage2/rules.md`. Arms:
//! - `stage 2`: `attention_forward_saving`, the dispatcher's kernel;
//! - `split-K`: `attention_forward_saving_splitk`, the scalar partials kernel;
//! - `pad-q`: `attention_forward_padq`, tessl's `encoder_attn` over Q padded
//!   to `Tk` rows, with the pad ratio bound lifted (where no bias, key lengths
//!   or head dim 16 rule it out).
//!
//! Every arm of a row runs twice per round (A, then A'), odd rounds in reverse
//! order. Rows give p10, median and p90 of the A pass in ms, the A' median,
//! the A-vs-A' gap and the pair's spread, and `ok` when the gap is within the
//! spread. Before any timing every row is checked: stage 2 against split-K and
//! pad-q over the whole output and against the CPU on the first head plane, and
//! the dispatch counts of stage 2 and split-K must match. The 1-minute load
//! average is printed at the start and the end, the fc1 linear is timed as the
//! control row, and the last line is the verdict (`clean` or `repeat`).

#[cfg(target_os = "macos")]
mod bench {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{AttentionOpts, Backend, Budget, Numerics, Tensor};
    use ojas_cpu::CpuBackend;
    use ojas_metal::MetalBackend;

    const MAX_LOAD: f64 = 6.0;
    const CONTROL_TFLOPS: (f64, f64) = (6.3 * 0.97, 6.3 * 1.03);

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

    struct Row {
        name: &'static str,
        h: usize,
        tq: usize,
        tk: usize,
        d: usize,
        bias: bool,
        padq: bool,
        /// Whether this row is a crossover row (paired-win counts printed).
        cross: bool,
    }

    const ROWS: &[Row] = &[
        Row {
            name: "memory cross m=5200",
            h: 8,
            tq: 5184,
            tk: 5200,
            d: 32,
            bias: false,
            padq: true,
            cross: true,
        },
        Row {
            name: "memory cross m=10400",
            h: 8,
            tq: 5184,
            tk: 10400,
            d: 32,
            bias: false,
            padq: true,
            cross: true,
        },
        Row {
            name: "memory cross m=10624",
            h: 8,
            tq: 5184,
            tk: 10624,
            d: 32,
            bias: false,
            padq: true,
            cross: true,
        },
        Row {
            name: "memory cross m=15584",
            h: 8,
            tq: 5184,
            tk: 15584,
            d: 32,
            bias: false,
            padq: true,
            cross: true,
        },
        Row {
            name: "memory cross m=36320",
            h: 8,
            tq: 5184,
            tk: 36320,
            d: 32,
            bias: false,
            padq: true,
            cross: false,
        },
        Row {
            name: "detector bias [1,8,201,32]x5184",
            h: 8,
            tq: 201,
            tk: 5184,
            d: 32,
            bias: true,
            padq: false,
            cross: false,
        },
        Row {
            name: "decoder D16 [1,8,80,16]x5184",
            h: 8,
            tq: 80,
            tk: 5184,
            d: 16,
            bias: false,
            padq: false,
            cross: false,
        },
    ];

    type Arm<'a> = (&'static str, Box<dyn Fn() + 'a>);

    struct Samples {
        /// `[pass][round]` timed milliseconds of one arm.
        ms: [Vec<f64>; 2],
    }

    fn pct(v: &[f64], q: f64) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[((s.len() - 1) as f64 * q).round() as usize]
    }

    /// Every round: each arm once, then each again; odd rounds reverse the
    /// arm order. Returns the samples per arm.
    fn run_arms(
        metal: &MetalBackend,
        arms: &[Arm<'_>],
        warmups: usize,
        runs: usize,
    ) -> Vec<Samples> {
        let count = arms.len();
        let mut out: Vec<Samples> = (0..count)
            .map(|_| Samples {
                ms: [Vec::new(), Vec::new()],
            })
            .collect();
        for round in 0..warmups + runs {
            for pass in 0..2 {
                for step in 0..count {
                    let i = if round % 2 == 0 {
                        step
                    } else {
                        count - 1 - step
                    };
                    metal.sync().expect("sync");
                    let start = Instant::now();
                    (arms[i].1)();
                    metal.sync().expect("sync");
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    if round >= warmups {
                        out[i].ms[pass].push(ms);
                    }
                }
            }
        }
        out
    }

    /// Prints each arm's row and returns `(median, p10, p90)` of the A pass
    /// and the number of NOISY pairs.
    fn report(
        label: &str,
        arms: &[Arm<'_>],
        samples: &[Samples],
        extra: impl Fn(f64) -> String,
    ) -> (Vec<(f64, f64, f64)>, usize) {
        let mut noisy = 0;
        let mut stats = Vec::new();
        for ((name, _), s) in arms.iter().zip(samples) {
            let (a, b) = (&s.ms[0], &s.ms[1]);
            let (ma, mb) = (pct(a, 0.5), pct(b, 0.5));
            let spread = (pct(a, 0.9) - pct(a, 0.1)).max(pct(b, 0.9) - pct(b, 0.1));
            let gap = (ma - mb).abs();
            let ok = gap <= spread;
            if !ok {
                noisy += 1;
            }
            println!(
                "{label}\t{name}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{gap:.3}\t{spread:.3}\t{}\t{}",
                pct(a, 0.1),
                ma,
                pct(a, 0.9),
                mb,
                if ok { "ok" } else { "NOISY" },
                extra(ma)
            );
            stats.push((ma, pct(a, 0.1), pct(a, 0.9)));
        }
        (stats, noisy)
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn cpu_agrees(got: &[f32], want: &[f32], tk: usize) -> Result<(), String> {
        let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let worst = max_abs(got, want);
        if worst > 1e-4 * peak.max(1e-12) {
            return Err(format!("max abs {worst} over max |ref| {peak}"));
        }
        for (i, (x, y)) in got.iter().zip(want).enumerate() {
            if (x - y).abs() > 4e-5 * (tk as f32).sqrt() + 1e-4 * y.abs() {
                return Err(format!("element {i}: {x} against {y}"));
            }
        }
        Ok(())
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let num = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
        let (warmups, runs) = (num(0, 5), num(1, 25));
        let only = args.get(2).cloned().unwrap_or_default();
        let budget = Budget::new(24 << 30);
        let metal = MetalBackend::new(budget.clone()).expect("Metal device");
        assert!(metal.has_tensorops(), "the pad-q arm needs TensorOps");
        let cpu = CpuBackend::new(Budget::new(24 << 30)).with_numerics(Numerics::Exact);
        let host =
            |data: &[f32], shape: &[usize]| Tensor::from_f32(data, shape, &budget).expect("host");
        let up = |t: &Tensor| metal.upload(t).expect("upload");
        let start_load = load();
        println!(
            "# load_start {:.2} {:.2} {:.2}",
            start_load[0], start_load[1], start_load[2]
        );
        println!(
            "row\tarm\tp10_ms\tmedian_ms\tp90_ms\tmedian_ms_A'\tgap_ms\tspread_ms\tpair\tnote"
        );
        let mut noisy_total = 0;
        let mut failed = false;

        // Control: fc1.
        let (cm, ck, cn) = (5184usize, 1024usize, 4736usize);
        {
            let x = up(&host(&values(cm * ck, 1, 1.0), &[cm, ck]));
            let w = up(&host(&values(cn * ck, 2, 0.05), &[cn, ck]));
            let arms: Vec<Arm<'_>> = vec![(
                "fc1 linear f32 (control)",
                Box::new(|| drop(metal.linear_forward(&x, &w).expect("fc1"))),
            )];
            let flops = 2.0 * cm as f64 * ck as f64 * cn as f64;
            let samples = run_arms(&metal, &arms, warmups, runs);
            let (stats, noisy) = report(
                "control fc1 [5184,1024]x[4736,1024]",
                &arms,
                &samples,
                |ms| format!("{:.2} TFLOP/s", flops / (ms * 1e-3) / 1e12),
            );
            noisy_total += noisy;
            let tflops = flops / (stats[0].0 * 1e-3) / 1e12;
            if !(CONTROL_TFLOPS.0..=CONTROL_TFLOPS.1).contains(&tflops) {
                println!(
                    "# control {tflops:.2} TFLOP/s outside {:.2} to {:.2}",
                    CONTROL_TFLOPS.0, CONTROL_TFLOPS.1
                );
                failed = true;
            }
        }

        for (ri, r) in ROWS
            .iter()
            .enumerate()
            .filter(|(_, r)| r.name.contains(&only))
        {
            let (b, h, tq, tk, d) = (1usize, r.h, r.tq, r.tk, r.d);
            let seed = 10 + 7 * ri as u64;
            let qd = values(b * h * tq * d, seed, 1.0);
            let kd = values(b * h * tk * d, seed + 1, 1.0);
            let vd = values(b * h * tk * d, seed + 2, 1.0);
            let bd = r.bias.then(|| values(b * h * tq * tk, seed + 3, 2.0));
            let (q, k, v) = (
                up(&host(&qd, &[b, h, tq, d])),
                up(&host(&kd, &[b, h, tk, d])),
                up(&host(&vd, &[b, h, tk, d])),
            );
            let bias = bd.as_ref().map(|x| up(&host(x, &[b, h, tq, tk])));
            let opts = || AttentionOpts {
                causal: false,
                window: None,
                bias: bias.as_ref(),
                key_lens: None,
            };

            // Correctness and dispatch counts before any timing.
            let (s2, s2_lse) = metal
                .attention_forward_saving(&q, &k, &v, opts())
                .expect("stage 2");
            let (sk, sk_lse) = metal
                .attention_forward_saving_splitk(&q, &k, &v, opts())
                .expect("split-K");
            metal.sync().expect("sync");
            let s2v = {
                let t = s2.to_host(&budget).expect("host");
                t.to_f32_vec().expect("f32")
            };
            let skv = sk
                .to_host(&budget)
                .expect("host")
                .to_f32_vec()
                .expect("f32");
            let s2l = s2_lse
                .to_host(&budget)
                .expect("host")
                .to_f32_vec()
                .expect("f32");
            let skl = sk_lse
                .to_host(&budget)
                .expect("host")
                .to_f32_vec()
                .expect("f32");
            let (d_out, d_lse) = (max_abs(&s2v, &skv), max_abs(&s2l, &skl));
            let mut note = format!("stage2-vs-splitK max abs out {d_out:.2e} lse {d_lse:.2e}");
            if d_out > 1e-5 || d_lse > 1e-4 {
                println!("# CORRECTNESS FAIL {}: {note}", r.name);
                failed = true;
            }
            if r.padq {
                let pq = metal
                    .attention_forward_padq(&q, &k, &v, opts())
                    .expect("pad-q");
                metal.sync().expect("sync");
                let pqv = pq
                    .to_host(&budget)
                    .expect("host")
                    .to_f32_vec()
                    .expect("f32");
                let dp = max_abs(&s2v, &pqv);
                note.push_str(&format!(" vs pad-q {dp:.2e}"));
                if dp > 1e-5 {
                    println!("# CORRECTNESS FAIL {}: {note}", r.name);
                    failed = true;
                }
            }
            // The first head plane against the CPU.
            let plane = |data: &[f32], rows: usize| host(&data[..rows * d], &[1, 1, rows, d]);
            let cbias = bd.as_ref().map(|x| host(&x[..tq * tk], &[1, 1, tq, tk]));
            let (cq, ck_, cv) = (plane(&qd, tq), plane(&kd, tk), plane(&vd, tk));
            let (want, want_lse) = cpu
                .attention_forward_saving(
                    &cq,
                    &ck_,
                    &cv,
                    AttentionOpts {
                        causal: false,
                        window: None,
                        bias: cbias.as_ref(),
                        key_lens: None,
                    },
                )
                .expect("cpu");
            let want = want.to_f32_vec().expect("f32");
            let want_lse = want_lse.to_f32_vec().expect("f32");
            for (what, got, w, tkk) in [
                ("output", &s2v[..tq * d], &want, tk),
                ("lse", &s2l[..tq], &want_lse, tk),
            ] {
                if let Err(e) = cpu_agrees(got, w, tkk) {
                    println!("# CORRECTNESS FAIL {} vs CPU {what}: {e}", r.name);
                    failed = true;
                }
            }
            let count = |splitk: bool| {
                metal.sync().expect("sync");
                let fresh = (
                    up(&host(&qd, &[b, h, tq, d])),
                    up(&host(&kd, &[b, h, tk, d])),
                    up(&host(&vd, &[b, h, tk, d])),
                );
                let fresh_bias = bd.as_ref().map(|x| up(&host(x, &[b, h, tq, tk])));
                let o = AttentionOpts {
                    causal: false,
                    window: None,
                    bias: fresh_bias.as_ref(),
                    key_lens: None,
                };
                metal.sync().expect("sync");
                let before = metal.dispatches();
                let out = if splitk {
                    metal.attention_forward_saving_splitk(&fresh.0, &fresh.1, &fresh.2, o)
                } else {
                    metal.attention_forward_saving(&fresh.0, &fresh.1, &fresh.2, o)
                }
                .expect("count");
                metal.sync().expect("sync");
                drop(out);
                metal.dispatches() - before
            };
            let (n2, nk) = (count(false), count(true));
            note.push_str(&format!(" dispatches stage2 {n2} split-K {nk}"));
            if n2 != nk {
                println!("# DISPATCH FAIL {}: {note}", r.name);
                failed = true;
            }
            drop((s2, sk, s2_lse, sk_lse));

            let mut arms: Vec<Arm<'_>> = vec![
                (
                    "stage 2",
                    Box::new(|| {
                        drop(
                            metal
                                .attention_forward_saving(&q, &k, &v, opts())
                                .expect("s2"),
                        )
                    }),
                ),
                (
                    "split-K",
                    Box::new(|| {
                        drop(
                            metal
                                .attention_forward_saving_splitk(&q, &k, &v, opts())
                                .expect("sk"),
                        )
                    }),
                ),
            ];
            if r.padq {
                arms.push((
                    "pad-q",
                    Box::new(|| {
                        drop(
                            metal
                                .attention_forward_padq(&q, &k, &v, opts())
                                .expect("pq"),
                        )
                    }),
                ));
            }
            let samples = run_arms(&metal, &arms, warmups, runs);
            let flops = 4.0 * (h * tq * tk * d) as f64;
            let (stats, noisy) = report(r.name, &arms, &samples, |ms| {
                format!(
                    "{:.2} TFLOP/s on the useful work",
                    flops / (ms * 1e-3) / 1e12
                )
            });
            noisy_total += noisy;
            let (s2m, skm) = (stats[0], stats[1]);
            let noise_sk = (s2m.2 - s2m.1).max(skm.2 - skm.1);
            println!(
                "# row {} | {note} | stage2 - splitK = {:.3} ms (noise bar {:.3}) {}",
                r.name,
                s2m.0 - skm.0,
                noise_sk,
                if s2m.0 - skm.0 > noise_sk {
                    "REGRESSION"
                } else {
                    "no regression"
                }
            );
            if r.padq {
                let pqm = stats[2];
                let noise_pq = (s2m.2 - s2m.1).max(pqm.2 - pqm.1);
                let wins = (0..2)
                    .flat_map(|p| {
                        samples[0].ms[p]
                            .iter()
                            .zip(&samples[2].ms[p])
                            .map(|(a, b)| a < b)
                            .collect::<Vec<_>>()
                    })
                    .filter(|&w| w)
                    .count();
                let total = samples[0].ms[0].len() + samples[0].ms[1].len();
                println!(
                    "# row {} | pad-q - stage2 = {:.3} ms (noise bar {:.3}), stage 2 faster in {wins}/{total} paired samples; better of split-K and pad-q = {:.3} ms",
                    r.name,
                    pqm.0 - s2m.0,
                    noise_pq,
                    skm.0.min(pqm.0)
                );
                if r.cross {
                    println!(
                        "# cross {} | stage 2 beats pad-q beyond noise: {}",
                        r.name,
                        pqm.0 - s2m.0 > noise_pq && wins * 10 >= total * 9
                    );
                }
            }
        }

        let end_load = load();
        println!(
            "# load_end {:.2} {:.2} {:.2}",
            end_load[0], end_load[1], end_load[2]
        );
        let loads_ok = start_load[0] <= MAX_LOAD && end_load[0] <= MAX_LOAD;
        println!("# noisy_pairs {noisy_total}");
        println!(
            "# verdict {}",
            if loads_ok && noisy_total == 0 && !failed {
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
    eprintln!("attention_stage2_bench needs macOS");
}
