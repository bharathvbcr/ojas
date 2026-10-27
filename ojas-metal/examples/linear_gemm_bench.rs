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
//! the im2col and layout share.

#[cfg(target_os = "macos")]
mod bench {
    use std::time::Instant;

    use ojas_core::{Autocast, AutocastMode, Backend, Budget, Conv2dParams, Tensor};
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
            let arms: Vec<Arm<'_>> = vec![
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
            let flops = 2.0 * m as f64 * k as f64 * c.out as f64;
            run_arms(&metal, c.name, flops, &arms, warmups, runs);
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
