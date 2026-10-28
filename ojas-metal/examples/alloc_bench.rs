//! Does allocation churn explain why SpikeTrack's frame profile prices the
//! large-output ops at 1.8x to 2.7x their quiet single-call medians
//! (`bench/results/2026-10-27-alloc/rules.md`, card ft-580e6c40)? Four of the
//! profile's rows, each under the same arms, on a quiet Metal device:
//!
//! - `A` steady state: the op alone, as `frame_ops_bench` times it;
//! - `B` big churn: before each call (untimed) device tensors of other sizes
//!   (three of about 170 MB and one of 760 MB) are uploaded and dropped, so
//!   more than the pool cache cap of buffers is freed at once;
//! - `B2` small churn: before each call, 1.25 GiB of distinct buffers each half
//!   the op's output size, so the pool cache is full of buffers smaller than the
//!   op's output and the output's freed buffer is released, not cached
//!   (tessl `BufferPool::recycle`);
//! - `C28` big churn on a second backend opened with SpikeTrack's production
//!   budget, 28 GiB (`image_cmd.rs` BUDGET_BYTES, `vos_gate.rs:62`); the main
//!   backend's 24 GiB is the profile's (`frame_profile.rs:163`) and is arm B;
//! - `E` fresh: `trim_pool` before each call (untimed), so the op's output and
//!   scratch buffers are new allocations every call, with no other churn;
//! - `Dsync` and `Dnosync`: eight consecutive calls with a sync after each
//!   versus one sync after all eight, per call.
//!
//! Every arm runs twice per round (A and A'), odd rounds reversed, with the
//! fc1 control as in `frame_ops_bench`. Each timed sample is preceded by one
//! untimed pass of the same arm so the pool holds that arm's own state, not the
//! previous arm's. Per sample the example records `MetalBackend::waits()`,
//! `dispatches()` and `wait_counts()` deltas around the timed region only.
//!
//! `cargo run -p ojas-metal --release --example alloc_bench [warmups] [runs] [only]`

#[cfg(target_os = "macos")]
mod bench {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{Backend, Budget, Conv2dParams, ConvTranspose2dParams, Tensor};
    use ojas_metal::{MetalBackend, WaitCounts};

    const MAX_LOAD: f64 = 6.0;
    const CONTROL_TFLOPS: f64 = 6.3;
    const CONTROL_TOL: f64 = 0.03;
    const ATTEMPTS: usize = 3;
    /// Calls per sample of the sync-cost arms.
    const D_CALLS: usize = 8;
    /// Swap-ins an attempt may see and still be clean (rules.md amendment; Pageouts must be zero).
    const SWAPINS_CAP: u64 = 1000;
    /// The memory cap a 24 GiB backend opens with: min(1 GiB, budget / 4), device.rs:88 and :587.
    const DEFAULT_MEM_CAP: u64 = 1 << 30;
    /// The raised cap of the `Mraise` arm: above the 8 conv calls' ~6.8 GB between waits.
    const RAISED_MEM_CAP: u64 = 8 << 30;
    /// Bytes the small churn frees per call: past the 1 GiB pool cap and memory cap.
    const SMALL_CHURN_BYTES: u64 = 5 << 28;
    /// Big-churn tensors, in f32 elements: 168, 172, 176 and 760 MB.
    const BIG_CHURN: [usize; 4] = [42_000_000, 43_000_000, 44_000_000, 190_000_000];
    const FC1: (usize, usize, usize) = (5184, 1024, 4736);

    #[derive(Clone, Copy)]
    enum Kind {
        Conv3,
        ConvT,
        Interp,
        LinearBias,
    }

    struct Row {
        name: &'static str,
        kind: Kind,
        shapes: &'static [&'static [usize]],
    }

    /// The profile's shapes (`frame-metal.log`, tracker frame F = 7).
    const ROWS: &[Row] = &[
        Row {
            name: "conv2d [1, 256, 288, 288] x [256, 256, 3, 3] x [256] stride 1 pad 1",
            kind: Kind::Conv3,
            shapes: &[&[1, 256, 288, 288], &[256, 256, 3, 3], &[256]],
        },
        Row {
            name: "conv_transpose2d [1, 512, 144, 144] x [512, 256, 2, 2] x [256] stride 2",
            kind: Kind::ConvT,
            shapes: &[&[1, 512, 144, 144], &[512, 256, 2, 2], &[256]],
        },
        Row {
            name: "interpolate_bilinear [1, 32, 1008, 1008] -> [1152, 1152]",
            kind: Kind::Interp,
            shapes: &[&[1, 32, 1008, 1008]],
        },
        Row {
            name: "linear_bias [1, 36400, 256] x [256, 256] x [256]",
            kind: Kind::LinearBias,
            shapes: &[&[1, 36400, 256], &[256, 256], &[256]],
        },
    ];

    fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
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

    /// `vm_stat` Pageouts and Swapins (cumulative page counts) and `vm.swapusage`.
    fn vm_pressure() -> (u64, u64, String) {
        let vm = Command::new("vm_stat").output().expect("vm_stat");
        let text = String::from_utf8_lossy(&vm.stdout).into_owned();
        let field = |name: &str| -> u64 {
            text.lines()
                .find_map(|l| l.trim().strip_prefix(name))
                .and_then(|r| {
                    r.trim_start_matches(':')
                        .trim()
                        .trim_end_matches('.')
                        .parse()
                        .ok()
                })
                .unwrap_or_else(|| panic!("vm_stat has no {name}: {text:?}"))
        };
        let swap = Command::new("sysctl")
            .args(["-n", "vm.swapusage"])
            .output()
            .expect("sysctl vm.swapusage");
        (
            field("Pageouts"),
            field("Swapins"),
            String::from_utf8_lossy(&swap.stdout)
                .trim()
                .replace('\t', " "),
        )
    }

    struct Stat {
        min: f64,
        p10: f64,
        med: f64,
        p90: f64,
    }

    fn stat(mut ms: Vec<f64>) -> Stat {
        ms.sort_by(f64::total_cmp);
        let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
        Stat {
            min: ms[0],
            p10: at(0.10),
            med: at(0.50),
            p90: at(0.90),
        }
    }

    /// One timed sample: milliseconds per call and the counter deltas per call.
    #[derive(Clone, Copy)]
    struct Sample {
        ms: f64,
        waits: f64,
        dispatches: f64,
        by: [f64; 9],
    }

    const TRIGGERS: [&str; 9] = [
        "upload",
        "read",
        "sync",
        "clip_norm",
        "mem_cap",
        "working_set",
        "slab_full",
        "recycle",
        "freed",
    ];

    fn by_trigger(c: &WaitCounts) -> [u64; 9] {
        [
            c.upload,
            c.read,
            c.sync,
            c.clip_norm,
            c.mem_cap,
            c.working_set,
            c.slab_full,
            c.recycle,
            c.freed,
        ]
    }

    struct Arm<'a> {
        name: String,
        bytes: u64,
        mb: &'a MetalBackend,
        /// Untimed, before the timed region (churn, trim, a cap change).
        prep: Box<dyn Fn() + 'a>,
        /// Untimed, after the timed region and its counters (undoes `prep`).
        post: Box<dyn Fn() + 'a>,
        call: Box<dyn Fn() + 'a>,
        reps: usize,
        sync_each: bool,
        /// Whether this arm's pair gates the attempt's cleanliness.
        gates: bool,
    }

    fn sample(arm: &Arm<'_>) -> Sample {
        (arm.prep)();
        let (w0, d0, c0) = (arm.mb.waits(), arm.mb.dispatches(), arm.mb.wait_counts());
        let start = Instant::now();
        for _ in 0..arm.reps {
            (arm.call)();
            if arm.sync_each {
                arm.mb.sync().expect("sync");
            }
        }
        arm.mb.sync().expect("sync");
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let (w1, d1, c1) = (arm.mb.waits(), arm.mb.dispatches(), arm.mb.wait_counts());
        let n = arm.reps as f64;
        (arm.post)();
        let (b0, b1) = (by_trigger(&c0), by_trigger(&c1));
        let mut by = [0.0; 9];
        for i in 0..9 {
            by[i] = (b1[i] - b0[i]) as f64 / n;
        }
        Sample {
            ms: ms / n,
            waits: (w1 - w0) as f64 / n,
            dispatches: (d1 - d0) as f64 / n,
            by,
        }
    }

    /// Upload `hosts` all at once, then drop them all.
    fn churn(mb: &MetalBackend, hosts: &[Tensor]) {
        let live: Vec<Tensor> = hosts
            .iter()
            .map(|h| mb.upload(h).expect("churn upload"))
            .collect();
        drop(live);
    }

    fn mean(v: impl Iterator<Item = f64>) -> f64 {
        let (s, n) = v.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
        s / n.max(1) as f64
    }

    fn attempt(
        label: &str,
        attempt: usize,
        arms: &[Arm<'_>],
        control: &Arm<'_>,
        warmups: usize,
        runs: usize,
    ) -> bool {
        let all: Vec<&Arm<'_>> = arms.iter().chain(std::iter::once(control)).collect();
        let count = all.len();
        let load_start = load();
        let vm0 = vm_pressure();
        let mut samples: Vec<Vec<Sample>> = vec![Vec::with_capacity(runs); 2 * count];
        for round in 0..warmups + runs {
            for pass in 0..2 {
                for step in 0..count {
                    let i = if round % 2 == 0 {
                        step
                    } else {
                        count - 1 - step
                    };
                    // Untimed pass of the same arm: the pool holds its own state.
                    let _ = sample(all[i]);
                    let s = sample(all[i]);
                    if round >= warmups {
                        samples[pass * count + i].push(s);
                    }
                }
            }
        }
        let load_end = load();
        let vm1 = vm_pressure();
        let pageouts = vm1.0.saturating_sub(vm0.0);
        let swapins = vm1.1.saturating_sub(vm0.1);
        let mut noisy_gating = 0;
        let mut control_tflops = 0.0;
        for (i, arm) in all.iter().enumerate() {
            let a = stat(samples[i].iter().map(|s| s.ms).collect());
            let b = stat(samples[count + i].iter().map(|s| s.ms).collect());
            let gap = (a.med - b.med).abs();
            let spread = (a.p90 - a.p10).max(b.p90 - b.p10);
            let ok = gap <= spread;
            if !ok && arm.gates {
                noisy_gating += 1;
            }
            if i == count - 1 {
                let flops = 2.0 * FC1.0 as f64 * FC1.1 as f64 * FC1.2 as f64;
                control_tflops = flops / (a.med * 1e-3) / 1e12;
            }
            for (r, tag, off) in [(&a, "", 0), (&b, "'", count)] {
                let ss = &samples[off + i];
                let waits = mean(ss.iter().map(|s| s.waits));
                let wmax = ss.iter().map(|s| s.waits).fold(0.0, f64::max);
                let disp = mean(ss.iter().map(|s| s.dispatches));
                let trig: Vec<String> = (0..9)
                    .filter_map(|t| {
                        let m = mean(ss.iter().map(|s| s.by[t]));
                        (m > 0.0).then(|| format!("{}={:.2}", TRIGGERS[t], m))
                    })
                    .collect();
                println!(
                    "{label}\t{attempt}\t{}{tag}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{gap:.4}\t{spread:.4}\t{}\twaits_mean {waits:.2}\twaits_max {wmax:.0}\tdispatches_mean {disp:.2}\t{}",
                    arm.name,
                    arm.bytes,
                    r.min,
                    r.p10,
                    r.med,
                    r.p90,
                    if ok { "ok" } else if arm.gates { "NOISY" } else { "noisy (not gating)" },
                    trig.join(" ")
                );
            }
        }
        for (i, arm) in all.iter().enumerate() {
            for pass in 0..2 {
                for (round, s) in samples[pass * count + i].iter().enumerate() {
                    let trig: Vec<String> = (0..9)
                        .filter(|&t| s.by[t] > 0.0)
                        .map(|t| format!("{}={:.2}", TRIGGERS[t], s.by[t]))
                        .collect();
                    println!(
                        "# sample\t{label}\t{attempt}\t{}\t{pass}\t{round}\t{:.4}\t{:.2}\t{:.2}\t{}",
                        arm.name,
                        s.ms,
                        s.waits,
                        s.dispatches,
                        trig.join(" ")
                    );
                }
            }
        }
        // Rounds in which the control, run in the same round, stayed within
        // CONTROL_TOL of its own median in both passes: a contended round shows up
        // in the control. Per arm, the median over those rounds and over all rounds
        // (both passes together).
        let ctl_all: Vec<f64> = (0..2)
            .flat_map(|p| samples[p * count + count - 1].iter().map(|s| s.ms))
            .collect();
        let ctl_med = stat(ctl_all).med;
        let quiet: Vec<bool> = (0..runs)
            .map(|r| {
                (0..2).all(|p| {
                    (samples[p * count + count - 1][r].ms / ctl_med - 1.0).abs() <= CONTROL_TOL
                })
            })
            .collect();
        let n_quiet = quiet.iter().filter(|&&q| q).count();
        for (i, arm) in all.iter().enumerate() {
            let both = |only_quiet: bool| -> f64 {
                let v: Vec<f64> = (0..runs)
                    .filter(|&r| !only_quiet || quiet[r])
                    .flat_map(|r| [samples[i][r].ms, samples[count + i][r].ms])
                    .collect();
                if v.is_empty() {
                    f64::NAN
                } else {
                    stat(v).med
                }
            };
            println!(
                "# quiet_rounds\t{label}\t{attempt}\t{}\t{n_quiet} of {runs} rounds\tmedian_quiet_ms {:.4}\tmedian_all_ms {:.4}",
                arm.name,
                both(true),
                both(false)
            );
        }
        let load_ok = load_start[0] <= MAX_LOAD && load_end[0] <= MAX_LOAD;
        let control_ok = (control_tflops / CONTROL_TFLOPS - 1.0).abs() <= CONTROL_TOL;
        let pressure = pageouts != 0 || swapins > SWAPINS_CAP;
        let clean = load_ok && control_ok && noisy_gating == 0 && !pressure;
        println!(
            "# attempt\t{label}\t{attempt}\tload_start {:.2} {:.2} {:.2}\tload_end {:.2} {:.2} {:.2}\tcontrol_tflops {control_tflops:.2}\tcontrol_ok {control_ok}\tnoisy_pairs_gating {noisy_gating}\tpageouts_delta {pageouts}\tswapins_delta {swapins}\tswap_before {}\tswap_after {}\tverdict {}",
            load_start[0],
            load_start[1],
            load_start[2],
            load_end[0],
            load_end[1],
            load_end[2],
            vm0.2,
            vm1.2,
            if clean {
                "clean"
            } else if pressure {
                "pressure"
            } else {
                "repeat"
            }
        );
        clean
    }

    fn op_call<'a>(mb: &'a MetalBackend, kind: Kind, t: &'a [Tensor]) -> Box<dyn Fn() + 'a> {
        match kind {
            Kind::Conv3 => {
                let mut p = Conv2dParams::new();
                p.stride = [1, 1];
                p.padding = [1, 1];
                Box::new(move || {
                    drop(
                        mb.conv2d_forward(&t[0], &t[1], Some(&t[2]), p)
                            .expect("conv2d"),
                    )
                })
            }
            Kind::ConvT => {
                let mut p = ConvTranspose2dParams::new();
                p.stride = [2, 2];
                Box::new(move || {
                    drop(
                        mb.conv_transpose2d_forward(&t[0], &t[1], Some(&t[2]), p)
                            .expect("convT"),
                    )
                })
            }
            Kind::Interp => Box::new(move || {
                drop(
                    mb.interpolate_bilinear_forward(&t[0], [1152, 1152], false)
                        .expect("interpolate"),
                )
            }),
            Kind::LinearBias => Box::new(move || {
                drop(
                    mb.linear_bias_forward(&t[0], &t[1], &t[2])
                        .expect("linear_bias"),
                )
            }),
        }
    }

    fn out_bytes(mb: &MetalBackend, kind: Kind, t: &[Tensor]) -> u64 {
        let out = match kind {
            Kind::Conv3 => {
                let mut p = Conv2dParams::new();
                p.stride = [1, 1];
                p.padding = [1, 1];
                mb.conv2d_forward(&t[0], &t[1], Some(&t[2]), p)
                    .expect("conv2d")
            }
            Kind::ConvT => {
                let mut p = ConvTranspose2dParams::new();
                p.stride = [2, 2];
                mb.conv_transpose2d_forward(&t[0], &t[1], Some(&t[2]), p)
                    .expect("convT")
            }
            Kind::Interp => mb
                .interpolate_bilinear_forward(&t[0], [1152, 1152], false)
                .expect("interpolate"),
            Kind::LinearBias => mb
                .linear_bias_forward(&t[0], &t[1], &t[2])
                .expect("linear_bias"),
        };
        4 * numel(out.shape()) as u64
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let num = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
        let (warmups, runs) = (num(0, 5), num(1, 25));
        let only = args.get(2).cloned().unwrap_or_default();
        let budget24 = Budget::new(24 << 30);
        let main_owner = MetalBackend::new(budget24.clone()).expect("Metal device");
        let main = &main_owner;
        let b28_owner = MetalBackend::new(Budget::new(28 << 30)).expect("Metal device, 28 GiB");
        let b28 = &b28_owner;
        assert!(main.has_tensorops(), "the control needs TensorOps");
        for (tag, m) in [("24 GiB", main), ("28 GiB", b28)] {
            let mem = m.memory().expect("memory");
            println!(
                "# backend {tag}: recommended_working_set {} pool_cache_cap {} allocated {}",
                mem.recommended_working_set, mem.pool_cache_cap, mem.allocated
            );
        }
        let host = |data: &[f32], shape: &[usize]| {
            Tensor::from_f32(data, shape, &budget24).expect("host tensor")
        };
        let dev = |m: &MetalBackend, shape: &[usize], seed: u64, scale: f32| {
            m.upload(&host(&values(numel(shape), seed, scale), shape))
                .expect("upload")
        };
        let ctl_x = dev(main, &[FC1.0, FC1.1], 91, 1.0);
        let ctl_w = dev(main, &[FC1.2, FC1.1], 92, 0.05);
        let control = Arm {
            name: "control: f32 linear fc1".into(),
            bytes: 4 * (FC1.0 * FC1.1 + FC1.2 * FC1.1 + FC1.0 * FC1.2) as u64,
            mb: main,
            prep: Box::new(|| {}),
            call: Box::new(|| {
                drop(main.linear_forward(&ctl_x, &ctl_w).expect("control"));
            }),
            reps: 1,
            sync_each: true,
            post: Box::new(|| {}),
            gates: true,
        };
        let big: Vec<Tensor> = BIG_CHURN
            .iter()
            .map(|&n| host(&vec![0.0f32; n], &[n]))
            .collect();
        let s = load();
        println!("# load_start {:.2} {:.2} {:.2}", s[0], s[1], s[2]);
        println!("# columns: row\tattempt\tarm\tbytes\tmin_ms\tp10_ms\tmedian_ms\tp90_ms\tgap_ms\tspread_ms\tpair\twaits_mean\twaits_max\tdispatches_mean\ttriggers (mean per call, nonzero)");
        let (mut clean_rows, mut total_rows) = (0, 0);
        for row in ROWS {
            if !only.is_empty() && !only.split('|').any(|k| row.name.contains(k)) {
                continue;
            }
            total_rows += 1;
            let t: Vec<Tensor> = row
                .shapes
                .iter()
                .enumerate()
                .map(|(i, sh)| dev(main, sh, 10 + i as u64, if i == 0 { 1.0 } else { 0.05 }))
                .collect();
            let t28: Vec<Tensor> = row
                .shapes
                .iter()
                .enumerate()
                .map(|(i, sh)| dev(b28, sh, 10 + i as u64, if i == 0 { 1.0 } else { 0.05 }))
                .collect();
            let obytes = out_bytes(main, row.kind, &t);
            let in_bytes: u64 = row.shapes.iter().map(|s| 4 * numel(s) as u64).sum();
            let bytes = in_bytes + obytes;
            // Small churn: pieces half the output size, each a distinct 16 KiB bucket.
            let piece = (obytes / 2 / 4) as usize;
            let count = SMALL_CHURN_BYTES.div_ceil(piece as u64 * 4) as usize;
            let small: Vec<Tensor> = (0..count)
                .map(|i| {
                    let n = piece + i * 4096;
                    host(&vec![0.0f32; n], &[n])
                })
                .collect();
            println!(
                "# row\t{}\toutput_bytes {obytes}\tsmall_churn_pieces {count} of {} elements",
                row.name, piece
            );
            let call_main = op_call(main, row.kind, &t);
            let call_main_b = op_call(main, row.kind, &t);
            let call_main_b2 = op_call(main, row.kind, &t);
            let call_main_e = op_call(main, row.kind, &t);
            let call_main_ds = op_call(main, row.kind, &t);
            let call_main_dn = op_call(main, row.kind, &t);
            let call_main_md = op_call(main, row.kind, &t);
            let call_main_mr = op_call(main, row.kind, &t);
            let call_28 = op_call(b28, row.kind, &t28);
            let arms: Vec<Arm<'_>> = vec![
                Arm {
                    name: "A steady".into(),
                    bytes,
                    mb: main,
                    prep: Box::new(|| {}),
                    call: call_main,
                    reps: 1,
                    sync_each: true,
                    post: Box::new(|| {}),
                    gates: true,
                },
                Arm {
                    name: "B big churn (24 GiB)".into(),
                    bytes,
                    mb: main,
                    prep: Box::new(|| churn(main, &big)),
                    call: call_main_b,
                    reps: 1,
                    sync_each: true,
                    post: Box::new(|| {}),
                    gates: false,
                },
                Arm {
                    name: "B2 small churn (24 GiB)".into(),
                    bytes,
                    mb: main,
                    prep: Box::new(|| churn(main, &small)),
                    call: call_main_b2,
                    reps: 1,
                    sync_each: true,
                    post: Box::new(|| {}),
                    gates: false,
                },
                Arm {
                    name: "C28 big churn (28 GiB backend)".into(),
                    bytes,
                    mb: b28,
                    prep: Box::new(|| churn(b28, &big)),
                    call: call_28,
                    reps: 1,
                    sync_each: true,
                    post: Box::new(|| {}),
                    gates: false,
                },
                Arm {
                    name: "E fresh (trim_pool before each call)".into(),
                    bytes,
                    mb: main,
                    prep: Box::new(|| main.trim_pool().expect("trim_pool")),
                    call: call_main_e,
                    reps: 1,
                    sync_each: true,
                    post: Box::new(|| {}),
                    gates: false,
                },
                Arm {
                    name: format!("Mdef x{D_CALLS}"),
                    bytes,
                    mb: main,
                    prep: Box::new(|| {}),
                    call: call_main_md,
                    reps: D_CALLS,
                    sync_each: false,
                    post: Box::new(|| {}),
                    gates: false,
                },
                Arm {
                    name: format!("Mraise x{D_CALLS} (mem_cap 8 GiB)"),
                    bytes,
                    mb: main,
                    prep: Box::new(|| main.set_mem_cap(RAISED_MEM_CAP).expect("set_mem_cap")),
                    call: call_main_mr,
                    reps: D_CALLS,
                    sync_each: false,
                    post: Box::new(|| main.set_mem_cap(DEFAULT_MEM_CAP).expect("set_mem_cap")),
                    gates: false,
                },
                Arm {
                    name: format!("Dsync x{D_CALLS}"),
                    bytes,
                    mb: main,
                    prep: Box::new(|| {}),
                    call: call_main_ds,
                    reps: D_CALLS,
                    sync_each: true,
                    post: Box::new(|| {}),
                    gates: false,
                },
                Arm {
                    name: format!("Dnosync x{D_CALLS}"),
                    bytes,
                    mb: main,
                    prep: Box::new(|| {}),
                    call: call_main_dn,
                    reps: D_CALLS,
                    sync_each: false,
                    post: Box::new(|| {}),
                    gates: false,
                },
            ];
            let mut clean = false;
            for n in 1..=ATTEMPTS {
                if attempt(row.name, n, &arms, &control, warmups, runs) {
                    clean = true;
                    break;
                }
            }
            if clean {
                clean_rows += 1;
            }
            println!("# row\t{}\tclean {clean}", row.name);
        }
        let e = load();
        println!("# load_end {:.2} {:.2} {:.2}", e[0], e[1], e[2]);
        println!("# rows_clean {clean_rows} of {total_rows}");
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("alloc_bench needs macOS");
}
