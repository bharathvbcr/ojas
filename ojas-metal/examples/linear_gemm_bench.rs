//! Linear and convolution GEMMs of SpikeTrack's SAM 3.1 shapes on Metal, with
//! device-resident tensors: ms, achieved TFLOP/s, the f32 lane against the
//! bf16 lane, the bias add's share, and the convolution's im2col share.
//!
//! `cargo run -p ojas-metal --release --example linear_gemm_bench [warmups] [runs] [only]`
//!
//! Defaults 5 and 25; `only` is a substring of a shape name. Each round runs
//! every arm of a shape once, in the same order, so the arms see the same
//! machine state (interleaved); a shape's samples are the `runs` timed rounds
//! after the `warmups`. Rows give min, p10, median and p90 milliseconds and
//! TFLOP/s at the median (`2 m k n / ms`), tab-separated. The `bias add` row's
//! TFLOP/s is the linear's flop count over the add's time, not a rate: read its
//! milliseconds only.
//!
//! Arms of a linear `x [m, k] @ w [n, k]^T`:
//! - `f32`: `linear_forward` on f32 operands, tessl's exact-f32 NT GEMM
//!   (`matmul2d_tensorops_nt_f32`, a 32 x 32 tile on one simdgroup);
//! - `f32+bias`: `linear_bias_forward`, that GEMM and then `broadcast_add`;
//! - `bias add`: `broadcast_add_forward` alone on the `[m, n]` product;
//! - `bf16 lane`: `linear_forward` on operands rounded and tagged bf16-exact
//!   once (what `Autocast` hands the backend), tessl's cooperative bf16 NT
//!   GEMM (`matmul2d_tensorops_nt_bf16_f32`, 128 x 64, four simdgroups),
//!   which narrows both operands to bf16 inside the call;
//! - `bf16 lane+bias`: `linear_bias_forward` on the same operands;
//! - `autocast`: `Autocast<&MetalBackend>::linear_forward` in a bf16 region
//!   on untagged f32 operands: the two operand roundings, the bf16 GEMM and
//!   the output rounding, as a model run under `Autocast` pays them.
//!
//! Convolution shapes add the `conv2d` arm (im2col, the exact-f32 GEMM and the
//! layout pass) beside the `f32` linear of the same GEMM, so the difference is
//! the im2col and layout share. Further arms: `conv2d bf16-exact` (the same
//! convolution on tagged operands, the bf16 lane) with its GEMM alone, and,
//! where the kernel is 1 x 1 at stride 1 without padding, the same two through
//! `conv2d_via_im2col`, the route before such a convolution skipped im2col (at
//! that shape the plain `conv2d` arms, and the `o=8` probe, run the skip).
//!
//! Every shape first prints one `COUNT` line per arm: the compute dispatches
//! one call makes, and how many of them are `ojas_im2col`, `ojas_conv_nchw`
//! and `ojas_bcast_binary`. The `command overhead` rows time no work: `sync
//! only`, one minimal command (`fill` of one element), and sixteen of them
//! before one sync.

#[cfg(target_os = "macos")]
mod bench {
    use std::time::Instant;

    use ojas_core::{Autocast, AutocastMode, Backend, Budget, Conv2dParams, DType, Tensor};
    use ojas_metal::MetalBackend;

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

    struct Linear {
        name: &'static str,
        m: usize,
        k: usize,
        n: usize,
    }

    struct Conv {
        name: &'static str,
        shape: [usize; 4],
        out: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    }

    /// The linears the frame profile lists (SpikeTrack bench/results/profile),
    /// and the ones the task names.
    const LINEARS: &[Linear] = &[
        Linear {
            name: "vit fc1 [5184,1024]x[4736,1024]",
            m: 5184,
            k: 1024,
            n: 4736,
        },
        Linear {
            name: "vit fc2 [5184,4736]x[1024,4736]",
            m: 5184,
            k: 4736,
            n: 1024,
        },
        Linear {
            name: "vit qkv [5184,1024]x[3072,1024]",
            m: 5184,
            k: 1024,
            n: 3072,
        },
        Linear {
            name: "vit proj/q/k/v [5184,1024]x[1024,1024]",
            m: 5184,
            k: 1024,
            n: 1024,
        },
        Linear {
            name: "window proj [9*576,1024]x[1024,1024]",
            m: 9 * 576,
            k: 1024,
            n: 1024,
        },
        Linear {
            name: "256 [5184,256]x[256,256]",
            m: 5184,
            k: 256,
            n: 256,
        },
        Linear {
            name: "256 [10400,256]x[256,256]",
            m: 10400,
            k: 256,
            n: 256,
        },
        Linear {
            name: "256 up [5184,256]x[2048,256]",
            m: 5184,
            k: 256,
            n: 2048,
        },
        Linear {
            name: "256 down [5184,2048]x[256,2048]",
            m: 5184,
            k: 2048,
            n: 256,
        },
        Linear {
            name: "256 query [201,256]x[256,256]",
            m: 201,
            k: 256,
            n: 256,
        },
        Linear {
            name: "256 tiny [4,256]x[256,256]",
            m: 4,
            k: 256,
            n: 256,
        },
    ];

    const CONVS: &[Conv] = &[
        Conv {
            name: "patch embed [1,3,1008,1008] k14 s14",
            shape: [1, 3, 1008, 1008],
            out: 1024,
            kernel: 14,
            stride: 14,
            padding: 0,
        },
        Conv {
            name: "neck 3x3 p1 [1,256,288,288]",
            shape: [1, 256, 288, 288],
            out: 256,
            kernel: 3,
            stride: 1,
            padding: 1,
        },
        Conv {
            name: "neck 3x3 p1 [1,256,144,144]",
            shape: [1, 256, 144, 144],
            out: 256,
            kernel: 3,
            stride: 1,
            padding: 1,
        },
        Conv {
            name: "neck 3x3 p1 [1,256,72,72]",
            shape: [1, 256, 72, 72],
            out: 256,
            kernel: 3,
            stride: 1,
            padding: 1,
        },
        // The detector's 1 x 1 (a plain [82944,256] x [256,256] GEMM) and the
        // memory encoder's 3 x 3 (12.2 GFLOP), from SpikeTrack's op profile.
        Conv {
            name: "1x1 [1,256,288,288]",
            shape: [1, 256, 288, 288],
            out: 256,
            kernel: 1,
            stride: 1,
            padding: 0,
        },
        Conv {
            name: "memory encode 3x3 p1 [1,32,1152,1152] o16",
            shape: [1, 32, 1152, 1152],
            out: 16,
            kernel: 3,
            stride: 1,
            padding: 1,
        },
    ];

    type Arm<'a> = (&'static str, Box<dyn Fn() + 'a>);

    struct Stats {
        min: f64,
        p10: f64,
        med: f64,
        p90: f64,
    }

    fn stats(mut ms: Vec<f64>) -> Stats {
        ms.sort_by(f64::total_cmp);
        let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
        Stats {
            min: ms[0],
            p10: at(0.10),
            med: at(0.50),
            p90: at(0.90),
        }
    }

    /// Interleaved rounds over `arms`: every round runs each arm once.
    fn run_arms(
        metal: &MetalBackend,
        label: &str,
        flops: f64,
        arms: &[Arm<'_>],
        warmups: usize,
        runs: usize,
    ) {
        // What each arm dispatches, from one counted call before the timed
        // rounds: every compute dispatch (tessl's GEMMs and casts included)
        // and the Ojas kernels the convolution's passes are named by.
        for (name, f) in arms {
            metal.sync().expect("sync");
            let at = |k: &str| metal.kernel_dispatches(k);
            let before = (
                metal.dispatches(),
                at("ojas_im2col"),
                at("ojas_conv_nchw"),
                at("ojas_bcast_binary"),
            );
            f();
            metal.sync().expect("sync");
            println!(
                "COUNT\t{label}\t{name}\tdispatches {}\tim2col {}\tconv_nchw {}\tbcast {}",
                metal.dispatches() - before.0,
                at("ojas_im2col") - before.1,
                at("ojas_conv_nchw") - before.2,
                at("ojas_bcast_binary") - before.3,
            );
        }
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(runs); arms.len()];
        for round in 0..warmups + runs {
            for (i, (_, f)) in arms.iter().enumerate() {
                metal.sync().expect("sync");
                let start = Instant::now();
                f();
                metal.sync().expect("sync");
                let ms = start.elapsed().as_secs_f64() * 1e3;
                if round >= warmups {
                    samples[i].push(ms);
                }
            }
        }
        for ((name, _), s) in arms.iter().zip(samples) {
            let s = stats(s);
            let tflops = flops / (s.med * 1e-3) / 1e12;
            println!(
                "{label}\t{name}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
                s.min, s.p10, s.med, s.p90, tflops
            );
        }
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
            "the bf16 lane needs TensorOps: this device has none"
        );
        let host =
            |data: &[f32], shape: &[usize]| Tensor::from_f32(data, shape, &budget).expect("host");
        let up = |t: &Tensor| metal.upload(t).expect("upload");
        println!("shape\tarm\tmin_ms\tp10_ms\tmedian_ms\tp90_ms\tmedian_tflops");

        for l in LINEARS.iter().filter(|l| l.name.contains(&only)) {
            let (x, w, b) = (
                up(&host(&values(l.m * l.k, 1, 1.0), &[l.m, l.k])),
                up(&host(&values(l.n * l.k, 2, 0.05), &[l.n, l.k])),
                up(&host(&values(l.n, 3, 1.0), &[l.n])),
            );
            let ac = Autocast::new(&metal);
            let (xt, wt) = (
                ac.cast_bf16(&x).expect("round x"),
                ac.cast_bf16(&w).expect("round w"),
            );
            assert!(xt.bf16_exact() && wt.bf16_exact(), "operands are tagged");
            let _region = ac.autocast_region(AutocastMode::Bf16).expect("region");
            let y = metal.linear_forward(&x, &w).expect("product");
            let arms: Vec<Arm<'_>> = vec![
                (
                    "f32",
                    Box::new(|| drop(metal.linear_forward(&x, &w).expect("f32"))),
                ),
                (
                    "f32+bias",
                    Box::new(|| drop(metal.linear_bias_forward(&x, &w, &b).expect("f32+bias"))),
                ),
                (
                    "bias add",
                    Box::new(|| drop(metal.broadcast_add_forward(&y, &b).expect("bias add"))),
                ),
                (
                    "bf16 lane",
                    Box::new(|| drop(metal.linear_forward(&xt, &wt).expect("bf16"))),
                ),
                (
                    "bf16 lane+bias",
                    Box::new(|| drop(metal.linear_bias_forward(&xt, &wt, &b).expect("bf16+bias"))),
                ),
                (
                    "autocast",
                    Box::new(|| drop(ac.linear_forward(&x, &w).expect("autocast"))),
                ),
            ];
            let flops = 2.0 * l.m as f64 * l.k as f64 * l.n as f64;
            run_arms(&metal, l.name, flops, &arms, warmups, runs);
        }

        for c in CONVS.iter().filter(|c| c.name.contains(&only)) {
            let [n, ch, h, wd] = c.shape;
            let pad = c.padding;
            let (oh, ow) = (
                (h + 2 * pad - c.kernel) / c.stride + 1,
                (wd + 2 * pad - c.kernel) / c.stride + 1,
            );
            let (m, k) = (n * oh * ow, ch * c.kernel * c.kernel);
            let x = up(&host(&values(n * ch * h * wd, 4, 1.0), &c.shape));
            let wt = up(&host(
                &values(c.out * k, 5, 0.05),
                &[c.out, ch, c.kernel, c.kernel],
            ));
            let bias = up(&host(&values(c.out, 6, 1.0), &[c.out]));
            // The GEMM inside the convolution: the im2col matrix [m, k] by the
            // weight [out, k], as a linear.
            let cols = up(&host(&values(m * k, 7, 1.0), &[m, k]));
            let w2 = up(&host(&values(c.out * k, 5, 0.05), &[c.out, k]));
            let params = Conv2dParams {
                stride: [c.stride, c.stride],
                padding: [pad, pad],
                ..Conv2dParams::new()
            };
            // Probes that isolate the passes of the convolution: the same
            // im2col matrix with 8 output channels (its GEMM is a thin read of
            // the matrix), the thin GEMM alone, the write floor of the matrix
            // and of the output, and the product's transpose.
            let w8 = up(&host(&values(8 * k, 8, 0.05), &[8, ch, c.kernel, c.kernel]));
            let w8_2 = up(&host(&values(8 * k, 8, 0.05), &[8, k]));
            let prod = up(&host(&values(m * c.out, 9, 1.0), &[m, c.out]));
            // The same convolution on bf16-exact operands (what `Autocast`
            // hands the backend), and its GEMM alone on the bf16 lane.
            let ac = Autocast::new(&metal);
            let (xt, wtt) = (
                ac.cast_bf16(&x).expect("round x"),
                ac.cast_bf16(&wt).expect("round w"),
            );
            let (colst, w2t) = (
                ac.cast_bf16(&cols).expect("round cols"),
                ac.cast_bf16(&w2).expect("round w2"),
            );
            assert!(xt.bf16_exact() && wtt.bf16_exact(), "operands are tagged");
            let mut arms: Vec<Arm<'_>> = vec![
                (
                    "probe conv2d o=8 (im2col + thin GEMM; pointwise at 1x1)",
                    Box::new(|| drop(metal.conv2d_forward(&x, &w8, None, params).expect("p1"))),
                ),
                (
                    "probe GEMM o=8 (thin read of the matrix)",
                    Box::new(|| drop(metal.linear_forward(&cols, &w8_2).expect("p2"))),
                ),
                (
                    "probe fill [m,k] (write floor of the matrix)",
                    Box::new(|| drop(metal.fill(&[m, k], DType::F32, 1.0).expect("p3"))),
                ),
                (
                    "probe fill [m,o] (write floor of the output)",
                    Box::new(|| drop(metal.fill(&[m, c.out], DType::F32, 1.0).expect("p4"))),
                ),
                (
                    "probe permute [m,o] -> [o,m]",
                    Box::new(|| drop(metal.permute(&prod, &[1, 0]).expect("p5"))),
                ),
                (
                    "conv2d f32 (im2col + GEMM + layout)",
                    Box::new(|| {
                        drop(
                            metal
                                .conv2d_forward(&x, &wt, Some(&bias), params)
                                .expect("conv"),
                        )
                    }),
                ),
                (
                    "GEMM alone f32 (the conv's [m,k]x[n,k])",
                    Box::new(|| drop(metal.linear_forward(&cols, &w2).expect("gemm"))),
                ),
            ];
            arms.push((
                "conv2d bf16-exact (im2col + bf16 GEMM + layout)",
                Box::new(|| {
                    drop(
                        metal
                            .conv2d_forward(&xt, &wtt, Some(&bias), params)
                            .expect("conv bf16"),
                    )
                }),
            ));
            arms.push((
                "GEMM alone bf16 lane (the conv's [m,k]x[n,k])",
                Box::new(|| drop(metal.linear_forward(&colst, &w2t).expect("gemm bf16"))),
            ));
            // A 1 x 1 stride 1 unpadded convolution skips im2col; the old
            // route stays callable for the comparison.
            if c.kernel == 1 && c.stride == 1 && pad == 0 {
                arms.push((
                    "conv2d f32 via im2col (the route before the skip)",
                    Box::new(|| {
                        drop(
                            metal
                                .conv2d_via_im2col(&x, &wt, Some(&bias), params)
                                .expect("conv via im2col"),
                        )
                    }),
                ));
                arms.push((
                    "conv2d bf16-exact via im2col (the route before the skip)",
                    Box::new(|| {
                        drop(
                            metal
                                .conv2d_via_im2col(&xt, &wtt, Some(&bias), params)
                                .expect("conv bf16 via im2col"),
                        )
                    }),
                ));
            }
            let flops = 2.0 * m as f64 * k as f64 * c.out as f64;
            run_arms(&metal, c.name, flops, &arms, warmups, runs);
        }

        // Per-command overhead: a command that does no work, on the same
        // worker, beside the same command repeated 16 times before one sync.
        // `(16x - 1x) / 15` is the cost of one more queued command; the `sync
        // only` row is the timing and wait floor of every other row.
        if "command overhead".contains(&only) {
            let arms: Vec<Arm<'_>> = vec![
                ("sync only (no command)", Box::new(|| {})),
                (
                    "fill [1] x1 (one minimal command)",
                    Box::new(|| drop(metal.fill(&[1], DType::F32, 1.0).expect("fill"))),
                ),
                (
                    "fill [1] x16 (16 minimal commands, one sync)",
                    Box::new(|| {
                        for _ in 0..16 {
                            drop(metal.fill(&[1], DType::F32, 1.0).expect("fill"));
                        }
                    }),
                ),
            ];
            run_arms(&metal, "command overhead", 0.0, &arms, warmups, runs);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("linear_gemm_bench needs macOS");
}
