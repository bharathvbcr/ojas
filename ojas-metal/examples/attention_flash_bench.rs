//! The TensorOps flash kernel on cross-attention (`ojas_attn_fwd_d32` run with
//! `cross`) against the split-K kernels and tessl's pad-q route, at
//! SpikeTrack's cross-attention shapes. The rules are
//! `bench/results/2026-10-27-attention-mpp/rules.md`.
//!
//! `cargo run -p ojas-metal --release --example attention_flash_bench [warmups] [runs] [only]`
//!
//! Defaults 5 and 25; `only` is a substring of a row name. Arms of the gate,
//! bias and head-dim-16 rows:
//! - `flash` (head dim 32 with no bias: the rows the route covers) or
//!   `dispatcher` (the bias and head-dim-16 rows, which stay split-K):
//!   `attention_forward_saving`, the dispatcher's kernel;
//! - `split-K`: `attention_forward_saving_splitk`;
//! - `pad-q`: `attention_forward_padq`, tessl's `encoder_attn` over Q padded
//!   to `Tk` rows, with the pad ratio bound lifted (where no bias, key lengths
//!   or head dim 16 rule it out);
//! - `scalar` (the bias rows only): `attention_forward_saving_scalar`, the
//!   general `ojas_attn_gen_fwd` the tiled kernel replaces. Its output is held
//!   to the dispatcher's before any timing (max abs and max rel to the peak
//!   are printed, the reductions differ so they are not bit-equal), the
//!   kernels it encoded over the timed rounds are counted (the scalar kernel
//!   once per call, no split-K and no flash), and the row prints the
//!   dispatcher / scalar ratio against the pair's noise bar.
//!
//! The `sweep tq=.. m=..` rows (rules.md addendum 2) time `flash (forced)`
//! (`attention_forward_saving_flash`), `split-K` and the `dispatcher` for Tq in
//! 1 to 2048 over Tk 5184 and 15584, and the last `# floor rule` line derives
//! `FLASH_MIN_GROUPS` from them by the rule.
//!
//! Every arm of a row runs twice per round (A, then A'), odd rounds in reverse
//! order. Rows give p10, median and p90 of the A pass in ms, the A' median,
//! the A-vs-A' gap and the pair's spread, and `ok` when the gap is within the
//! spread. Before any timing every row is checked: the dispatcher's output
//! against split-K, the scalar kernel (flash rows) and pad-q over the whole
//! output and against the CPU on the first head plane, and its dispatch count
//! (a flash call is the scalar call's, split-K's is one more, a split-K row's
//! equals split-K's). The 1-minute load average is printed at the start and
//! the end, the fc1 linear is timed as the control row, and the last line is
//! the verdict (`clean` or `repeat`). The `# stand-in` lines are the rules'
//! pre-stated substitution when no clean window exists: the interleaved
//! dispatcher / pad-q ratio times the quiet pad-q median of
//! `bench/results/2026-10-27-attention-stage2`, beside the pad-q arm against
//! that quiet anchor.

#[cfg(target_os = "macos")]
mod bench {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{AttentionOpts, Backend, Budget, Numerics, Tensor};
    use ojas_cpu::CpuBackend;
    use ojas_metal::MetalBackend;

    const MAX_LOAD: f64 = 6.0;
    const CONTROL_TFLOPS: (f64, f64) = (6.3 * 0.97, 6.3 * 1.03);
    /// Quiet pad-q medians (ms) at q `[1, 8, 5184, 32]` over m keys: stage 2's
    /// clean attempt 2, `bench/results/2026-10-27-attention-stage2/bench-output.txt`
    /// (load 3.9 to 5.8, fc1 control 6.33 TFLOP/s).
    const QUIET_PADQ_MS: [(usize, f64); 5] = [
        (5200, 5.601),
        (10400, 21.056),
        (10624, 21.863),
        (15584, 46.836),
        (36320, 281.832),
    ];
    /// The least factor by which the dispatcher's call on a flash row must
    /// beat the same call forced onto the scalar kernel for the flash kernel
    /// to count as observed (the frame profile has the scalar kernel at 429 ms
    /// for m = 10400 against about 25 ms for the TensorOps route).
    const FLASH_VS_SCALAR_FLOOR: f64 = 3.0;
    /// The gate's per-call bounds (ms) at m = 15584 and 36320.
    const GATES_MS: [(usize, f64); 2] = [(15584, 20.0), (36320, 45.0)];

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
        name: String,
        h: usize,
        tq: usize,
        tk: usize,
        d: usize,
        bias: bool,
        padq: bool,
        /// Whether this row is a crossover row (paired-win counts printed).
        cross: bool,
        /// Whether this is a short-Q sweep row (flash forced against split-K
        /// and the dispatcher; sets `FLASH_MIN_GROUPS` by the rules).
        sweep: bool,
    }

    /// The sweep's query lengths and key lengths (rules.md, addendum 2).
    const SWEEP_TQ: [usize; 8] = [1, 8, 32, 64, 201, 512, 1024, 2048];
    const SWEEP_TK: [usize; 2] = [5184, 15584];

    fn rows() -> Vec<Row> {
        let gate = |name: &str, tk: usize, cross: bool| Row {
            name: name.to_string(),
            h: 8,
            tq: 5184,
            tk,
            d: 32,
            bias: false,
            padq: true,
            cross,
            sweep: false,
        };
        let mut rows = vec![
            gate("memory cross m=5200", 5200, true),
            gate("memory cross m=10400", 10400, true),
            gate("memory cross m=10624", 10624, true),
            gate("memory cross m=15584", 15584, true),
            gate("memory cross m=36320", 36320, false),
            Row {
                name: "detector bias [1,8,201,32]x5184".to_string(),
                h: 8,
                tq: 201,
                tk: 5184,
                d: 32,
                bias: true,
                padq: false,
                cross: false,
                sweep: false,
            },
            Row {
                name: "decoder D16 [1,8,80,16]x5184".to_string(),
                h: 8,
                tq: 80,
                tk: 5184,
                d: 16,
                bias: false,
                padq: false,
                cross: false,
                sweep: false,
            },
        ];
        for &tk in &SWEEP_TK {
            for &tq in &SWEEP_TQ {
                rows.push(Row {
                    name: format!("sweep tq={tq} m={tk}"),
                    h: 8,
                    tq,
                    tk,
                    d: 32,
                    bias: false,
                    padq: false,
                    cross: false,
                    sweep: true,
                });
            }
        }
        rows
    }

    /// Threadgroups the flash kernel launches: 32 query rows of each plane.
    fn flash_groups(b: usize, h: usize, tq: usize) -> u64 {
        (tq.div_ceil(32) * b * h) as u64
    }

    /// The attention kernels one call encoded, by pipeline name.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Ran {
        flash: u64,
        scalar: u64,
        part: u64,
        combine: u64,
    }

    impl Ran {
        const FLASH: Ran = Ran {
            flash: 1,
            scalar: 0,
            part: 0,
            combine: 0,
        };
        const SCALAR: Ran = Ran {
            flash: 0,
            scalar: 1,
            part: 0,
            combine: 0,
        };
        const SPLITK: Ran = Ran {
            flash: 0,
            scalar: 0,
            part: 1,
            combine: 1,
        };
    }

    fn snapshot(m: &MetalBackend) -> Ran {
        Ran {
            flash: m.kernel_dispatches("ojas_attn_fwd_d32"),
            scalar: m.kernel_dispatches("ojas_attn_gen_fwd"),
            part: m.kernel_dispatches("ojas_xattn_part_d32")
                + m.kernel_dispatches("ojas_xattn_part_d16"),
            combine: m.kernel_dispatches("ojas_xattn_combine_d32")
                + m.kernel_dispatches("ojas_xattn_combine_d16"),
        }
    }

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

    /// The largest `|got - want|` over the largest `|want|`.
    fn rel_to_peak(got: &[f32], want: &[f32]) -> f32 {
        let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        max_abs(got, want) / peak.max(1e-12)
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

        let table = rows();
        // (flash groups, tk, flash won) of every sweep row timed.
        let mut sweep_results: Vec<(u64, usize, bool)> = Vec::new();
        let sweep_rows = table.iter().filter(|r| r.sweep).count();
        for (ri, r) in table
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

            // The rows the flash route covers (head dim 32, no bias), and
            // whether the dispatcher takes them (threadgroup floor).
            let flash_row = !r.bias && d == 32;
            let groups = flash_groups(b, h, tq);
            let dispatcher_flash = flash_row && groups >= MetalBackend::flash_min_groups();
            // Correctness and dispatch counts before any timing.
            let (s2, s2_lse) = metal
                .attention_forward_saving(&q, &k, &v, opts())
                .expect("dispatcher");
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
            let mut note = format!("dispatcher-vs-splitK max abs out {d_out:.2e} lse {d_lse:.2e}");
            if d_out > 1e-5 || d_lse > 1e-4 {
                println!("# CORRECTNESS FAIL {}: {note}", r.name);
                failed = true;
            }
            // The scalar kernel is the oracle of the flash rows and the timed
            // baseline of the bias rows.
            if flash_row || r.bias {
                let (sc, sc_lse) = metal
                    .attention_forward_saving_scalar(&q, &k, &v, opts())
                    .expect("scalar");
                metal.sync().expect("sync");
                let scv = sc
                    .to_host(&budget)
                    .expect("host")
                    .to_f32_vec()
                    .expect("f32");
                let scl = sc_lse
                    .to_host(&budget)
                    .expect("host")
                    .to_f32_vec()
                    .expect("f32");
                note.push_str(&format!(
                    " vs scalar out {:.2e} lse {:.2e}",
                    max_abs(&s2v, &scv),
                    max_abs(&s2l, &scl)
                ));
                if r.bias {
                    note.push_str(&format!(
                        " (rel to peak out {:.2e} lse {:.2e})",
                        rel_to_peak(&s2v, &scv),
                        rel_to_peak(&s2l, &scl)
                    ));
                }
                for (what, got, w) in [("output", &s2v, &scv), ("lse", &s2l, &scl)] {
                    if let Err(e) = cpu_agrees(got, w, tk) {
                        println!("# CORRECTNESS FAIL {} vs scalar {what}: {e}", r.name);
                        failed = true;
                    }
                }
            }
            if r.sweep {
                let (fl, fl_lse) = metal
                    .attention_forward_saving_flash(&q, &k, &v, opts())
                    .expect("forced flash");
                metal.sync().expect("sync");
                let flv = fl
                    .to_host(&budget)
                    .expect("host")
                    .to_f32_vec()
                    .expect("f32");
                let fll = fl_lse
                    .to_host(&budget)
                    .expect("host")
                    .to_f32_vec()
                    .expect("f32");
                let (fo, fl_l) = (max_abs(&flv, &skv), max_abs(&fll, &skl));
                note.push_str(&format!(
                    " forced-flash-vs-splitK out {fo:.2e} lse {fl_l:.2e}"
                ));
                if fo > 1e-5 || fl_l > 1e-4 {
                    println!("# CORRECTNESS FAIL {}: {note}", r.name);
                    failed = true;
                }
                for (what, got, w) in [("output", &flv, &s2v), ("lse", &fll, &s2l)] {
                    if let Err(e) = cpu_agrees(got, w, tk) {
                        println!(
                            "# CORRECTNESS FAIL {} forced flash vs dispatcher {what}: {e}",
                            r.name
                        );
                        failed = true;
                    }
                }
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
            // Dispatch counts and the kernels encoded, observed on fresh
            // uploads: `kernel_dispatches` by pipeline name, not the planner.
            let count =
                |which: &str| -> (u64, Ran) {
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
                    let (before, kernels) = (metal.dispatches(), snapshot(&metal));
                    let out =
                        match which {
                            "splitk" => metal
                                .attention_forward_saving_splitk(&fresh.0, &fresh.1, &fresh.2, o),
                            "scalar" => metal
                                .attention_forward_saving_scalar(&fresh.0, &fresh.1, &fresh.2, o),
                            "flash" => metal
                                .attention_forward_saving_flash(&fresh.0, &fresh.1, &fresh.2, o),
                            _ => metal.attention_forward_saving(&fresh.0, &fresh.1, &fresh.2, o),
                        }
                        .expect("count");
                    metal.sync().expect("sync");
                    drop(out);
                    let now = snapshot(&metal);
                    (
                        metal.dispatches() - before,
                        Ran {
                            flash: now.flash - kernels.flash,
                            scalar: now.scalar - kernels.scalar,
                            part: now.part - kernels.part,
                            combine: now.combine - kernels.combine,
                        },
                    )
                };
            let ((n_auto, k_auto), (n_splitk, k_splitk), (n_scalar, k_scalar)) =
                (count("auto"), count("splitk"), count("scalar"));
            note.push_str(&format!(
                " dispatches dispatcher {n_auto} split-K {n_splitk} scalar {n_scalar}; kernels dispatcher {k_auto:?} split-K {k_splitk:?} scalar {k_scalar:?}"
            ));
            let mut dispatch_ok = k_auto
                == if dispatcher_flash {
                    Ran::FLASH
                } else {
                    Ran::SPLITK
                }
                && k_splitk == Ran::SPLITK
                && k_scalar == Ran::SCALAR
                && n_splitk == n_scalar + 1
                && (n_auto == if dispatcher_flash { n_scalar } else { n_splitk });
            if flash_row {
                let (n_flash, k_flash) = count("flash");
                note.push_str(&format!(" forced flash {n_flash} {k_flash:?}"));
                dispatch_ok &= k_flash == Ran::FLASH && n_flash == n_scalar;
            }
            if !dispatch_ok {
                println!("# DISPATCH FAIL {}: {note}", r.name);
                failed = true;
            }
            // The kernel counts above are the observation of which kernel ran;
            // time is a second, independent observable on the gate rows: the
            // dispatcher's call against the same call forced onto the scalar
            // kernel, on these inputs, best of three after a warm-up. Scalar is
            // many times slower than the TensorOps kernel at these shapes (the
            // frame profile has it at 429 ms for m = 10400), so a ratio under
            // the floor means the dispatcher did not run the flash kernel.
            if dispatcher_flash && !r.sweep {
                let time_call = |scalar: bool| -> f64 {
                    let mut best = f64::INFINITY;
                    for round in 0..4 {
                        metal.sync().expect("sync");
                        let start = Instant::now();
                        let out = if scalar {
                            metal.attention_forward_saving_scalar(&q, &k, &v, opts())
                        } else {
                            metal.attention_forward_saving(&q, &k, &v, opts())
                        }
                        .expect("observable");
                        metal.sync().expect("sync");
                        let ms = start.elapsed().as_secs_f64() * 1e3;
                        drop(out);
                        if round > 0 {
                            best = best.min(ms);
                        }
                    }
                    best
                };
                let (ms_auto, ms_scalar) = (time_call(false), time_call(true));
                let ratio = ms_scalar / ms_auto;
                let confirmed = ratio >= FLASH_VS_SCALAR_FLOOR;
                println!(
                    "# observable {} | the flash kernel ran if the dispatcher's call is at least {FLASH_VS_SCALAR_FLOOR}x faster than the same call forced onto the scalar kernel: dispatcher {ms_auto:.2} ms, forced scalar {ms_scalar:.2} ms, ratio {ratio:.1}x -> {}",
                    r.name,
                    if confirmed { "confirmed" } else { "NOT CONFIRMED" }
                );
                if !confirmed {
                    println!("# DISPATCH FAIL {}: flash kernel not observed", r.name);
                    failed = true;
                }
                note.push_str(&format!(" flash-vs-scalar time ratio {ratio:.1}x"));
            }
            drop((s2, sk, s2_lse, sk_lse));

            let mut arms: Vec<Arm<'_>> = vec![
                if r.sweep {
                    (
                        "flash (forced)",
                        Box::new(|| {
                            drop(
                                metal
                                    .attention_forward_saving_flash(&q, &k, &v, opts())
                                    .expect("flash"),
                            )
                        }),
                    )
                } else {
                    (
                        if dispatcher_flash {
                            "flash"
                        } else {
                            "dispatcher"
                        },
                        Box::new(|| {
                            drop(
                                metal
                                    .attention_forward_saving(&q, &k, &v, opts())
                                    .expect("s2"),
                            )
                        }),
                    )
                },
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
            if r.sweep {
                arms.push((
                    "dispatcher",
                    Box::new(|| {
                        drop(
                            metal
                                .attention_forward_saving(&q, &k, &v, opts())
                                .expect("s2"),
                        )
                    }),
                ));
            }
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
            // The bias rows' baseline, timed in the same interleaved rounds.
            let scalar_arm = r.bias.then(|| {
                arms.push((
                    "scalar",
                    Box::new(|| {
                        drop(
                            metal
                                .attention_forward_saving_scalar(&q, &k, &v, opts())
                                .expect("scalar"),
                        )
                    }),
                ));
                arms.len() - 1
            });
            metal.sync().expect("sync");
            let encoded_before = snapshot(&metal);
            let samples = run_arms(&metal, &arms, warmups, runs);
            let encoded = snapshot(&metal);
            let flops = 4.0 * (h * tq * tk * d) as f64;
            let (stats, noisy) = report(&r.name, &arms, &samples, |ms| {
                format!(
                    "{:.2} TFLOP/s on the useful work",
                    flops / (ms * 1e-3) / 1e12
                )
            });
            noisy_total += noisy;
            if r.sweep {
                // flash wins when below split-K by more than the larger
                // p90 - p10 spread of the two (a tie goes to split-K); the
                // dispatcher is held to split-K's time where it routes there
                // (same kernel) and to the best of the two where it routes
                // to flash.
                let (fl, sk_, dp) = (stats[0], stats[1], stats[2]);
                let bar = (fl.2 - fl.1).max(sk_.2 - sk_.1);
                let won = sk_.0 - fl.0 > bar;
                let dbar = (dp.2 - dp.1).max(sk_.2 - sk_.1);
                println!(
                    "# sweep {} | groups {groups} | flash - splitK = {:.3} ms (noise bar {bar:.3}): flash {} | dispatcher routed {} | dispatcher - splitK = {:.3} ms (noise bar {dbar:.3}) {}",
                    r.name,
                    fl.0 - sk_.0,
                    if won { "WINS" } else { "does not win" },
                    if dispatcher_flash { "flash" } else { "split-K" },
                    dp.0 - sk_.0,
                    if !dispatcher_flash && dp.0 - sk_.0 > dbar {
                        "REGRESSION"
                    } else {
                        "no regression"
                    }
                );
                sweep_results.push((groups, tk, won));
                continue;
            }
            let (s2m, skm) = (stats[0], stats[1]);
            if let Some(si) = scalar_arm {
                // Each arm ran `2 * (warmups + runs)` calls: the dispatcher
                // and split-K are both split-K on a bias row, the scalar arm
                // is the scalar kernel, so over the rounds the scalar kernel
                // was encoded once per scalar call, split-K's partials and
                // combine twice per call pair, and flash never.
                let calls = (2 * (warmups + runs)) as u64;
                let observed = Ran {
                    flash: encoded.flash - encoded_before.flash,
                    scalar: encoded.scalar - encoded_before.scalar,
                    part: encoded.part - encoded_before.part,
                    combine: encoded.combine - encoded_before.combine,
                };
                let want = Ran {
                    flash: 0,
                    scalar: calls,
                    part: 2 * calls,
                    combine: 2 * calls,
                };
                println!(
                    "# kernels {} | timed rounds encoded {observed:?}, expected {want:?} ({calls} calls per arm)",
                    r.name
                );
                if observed != want {
                    println!(
                        "# DISPATCH FAIL {}: the scalar arm did not run the scalar kernel once per call, or another arm ran a kernel it should not",
                        r.name
                    );
                    failed = true;
                }
                let sm = stats[si];
                let noise_sc = (s2m.2 - s2m.1).max(sm.2 - sm.1);
                println!(
                    "# row {} | scalar - dispatcher = {:.3} ms (noise bar {:.3}), dispatcher / scalar = {:.3} (scalar {:.1}x slower) {}",
                    r.name,
                    sm.0 - s2m.0,
                    noise_sc,
                    s2m.0 / sm.0,
                    sm.0 / s2m.0,
                    if sm.0 - s2m.0 > noise_sc {
                        "dispatcher faster beyond noise"
                    } else {
                        "NOT beyond noise"
                    }
                );
            }
            let noise_sk = (s2m.2 - s2m.1).max(skm.2 - skm.1);
            println!(
                "# row {} | {note} | dispatcher - splitK = {:.3} ms (noise bar {:.3}) {}",
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
                    "# row {} | pad-q - dispatcher = {:.3} ms (noise bar {:.3}), dispatcher faster in {wins}/{total} paired samples; better of split-K and pad-q = {:.3} ms",
                    r.name,
                    pqm.0 - s2m.0,
                    noise_pq,
                    skm.0.min(pqm.0)
                );
                if r.cross {
                    println!(
                        "# cross {} | dispatcher beats pad-q beyond noise: {}",
                        r.name,
                        pqm.0 - s2m.0 > noise_pq && wins * 10 >= total * 9
                    );
                }
                // The pre-stated substitution: the interleaved ratio times
                // the quiet pad-q median, beside the pad-q arm against its
                // quiet anchor (a high ratio there says the run was slow).
                if let Some(&(_, anchor)) = QUIET_PADQ_MS.iter().find(|(m, _)| *m == tk) {
                    let ratio = s2m.0 / pqm.0;
                    let gate = GATES_MS.iter().find(|(m, _)| *m == tk).map(|&(_, g)| g);
                    println!(
                        "# stand-in {} | dispatcher / pad-q = {ratio:.3} x quiet pad-q {anchor:.3} ms = {:.2} ms{}; pad-q arm {:.2} ms against its quiet anchor {anchor:.2} ms (x{:.2})",
                        r.name,
                        ratio * anchor,
                        gate.map_or(String::new(), |g| format!(
                            " (gate {g} ms: {})",
                            if ratio * anchor <= g { "met" } else { "missed" }
                        )),
                        pqm.0,
                        pqm.0 / anchor
                    );
                }
            }
        }

        if sweep_results.len() == sweep_rows {
            // rules.md addendum 2: the smallest sweep group count g such that
            // flash wins every row of both Tk with at least g groups; 513 if
            // no sweep row qualifies.
            let mut counts: Vec<u64> = sweep_results.iter().map(|r| r.0).collect();
            counts.sort_unstable();
            counts.dedup();
            let floor = counts
                .iter()
                .copied()
                .find(|&g| sweep_results.iter().filter(|r| r.0 >= g).all(|r| r.2))
                .unwrap_or(513);
            println!(
                "# floor rule | FLASH_MIN_GROUPS = {floor} by the sweep (current constant {}); valid only if the run's verdict is clean",
                MetalBackend::flash_min_groups()
            );
        } else if !sweep_results.is_empty() {
            println!(
                "# floor rule | not computed: {} of {sweep_rows} sweep rows timed",
                sweep_results.len()
            );
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
    eprintln!("attention_flash_bench needs macOS");
}
