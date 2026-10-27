//! The memory-bound ops of SpikeTrack's SAM 3.1 frame on Metal, measured with a
//! load average and a repeat of every arm: the bias add at the linear shapes
//! (`broadcast_add_forward` on the `[m, n]` product), `permute` at the shapes
//! `st-model` calls it with, and the f32 and bf16 linears as a control.
//!
//! `cargo run -p ojas-metal --release --example memory_bound_bench [warmups] [runs] [only]`
//!
//! Defaults 5 and 25; `only` is a substring of a shape name. A round runs every
//! arm once, then every arm a second time (the repeat, `A'`); odd rounds walk
//! the arms in reverse, so no arm is always first or last. Rows give the
//! minimum, p10, median and p90 of `runs` timed rounds in milliseconds, the
//! bytes the op must move (a read of the input and a write of the output; the
//! bias add adds its bias), GB/s at the median, TFLOP/s for the linears, and
//! for each arm its gap to its repeat (`|median - median'|`), the larger p90 -
//! p10 spread of the pair, and `ok` when the gap is within the spread.
//!
//! The 1-minute load average (`sysctl vm.loadavg`) is printed at the start and
//! the end, and the last line is the verdict: `clean` when both loads are at
//! most `MAX_LOAD` and every pair is `ok`, else `repeat`.

#[cfg(target_os = "macos")]
mod bench {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{Autocast, Backend, Budget, DType, Tensor};
    use ojas_metal::MetalBackend;

    /// The 1-minute load above which a run is repeated.
    const MAX_LOAD: f64 = 6.0;

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

    struct Linear {
        name: &'static str,
        m: usize,
        k: usize,
        n: usize,
    }

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
            name: "vit proj [5184,1024]x[1024,1024]",
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
    ];

    /// A `permute` of `shape` by `dims` and where `st-model` calls it, with the
    /// per-frame call count of the 2026-10-27 profile where it lists one.
    struct Perm {
        shape: &'static [usize],
        dims: &'static [usize],
        site: &'static str,
    }

    const PERMS: &[Perm] = &[
        Perm {
            shape: &[9, 576, 16, 64],
            dims: &[0, 2, 1, 3],
            site: "backbone.rs:613 windowed q/k/v to [B,H,T,D]",
        },
        Perm {
            shape: &[9, 16, 576, 64],
            dims: &[0, 2, 1, 3],
            site: "backbone.rs:616 windowed attention out",
        },
        Perm {
            shape: &[1, 5184, 16, 64],
            dims: &[0, 2, 1, 3],
            site: "backbone.rs:613 global q/k/v",
        },
        Perm {
            shape: &[1, 16, 5184, 64],
            dims: &[0, 2, 1, 3],
            site: "backbone.rs:616 global attention out",
        },
        Perm {
            shape: &[1, 3, 24, 3, 24, 1024],
            dims: &[0, 1, 3, 2, 4, 5],
            site: "spatial.rs:635 window_partition (backbone.rs:604)",
        },
        Perm {
            shape: &[1, 3, 3, 24, 24, 1024],
            dims: &[0, 1, 3, 2, 4, 5],
            site: "spatial.rs:674 window_merge (backbone.rs:621)",
        },
        Perm {
            shape: &[1, 1024, 72, 72],
            dims: &[0, 2, 3, 1],
            site: "backbone.rs:507 patch embed to NHWC",
        },
        Perm {
            shape: &[1, 72, 72, 1024],
            dims: &[0, 3, 1, 2],
            site: "backbone.rs:635 neck input to NCHW",
        },
        Perm {
            shape: &[1, 256, 72, 72],
            dims: &[0, 2, 3, 1],
            site: "memory_encoder.rs:107 NHWC",
        },
        Perm {
            shape: &[1, 72, 72, 256],
            dims: &[0, 3, 1, 2],
            site: "memory_encoder.rs:110 NCHW",
        },
        Perm {
            shape: &[1, 256, 5184],
            dims: &[0, 2, 1],
            site: "detector.rs:280, interactive.rs:86, mask_decoder.rs:164 NCHW to tokens",
        },
        Perm {
            shape: &[1, 5184, 256],
            dims: &[0, 2, 1],
            site: "detector.rs:361, mask_decoder.rs:172 tokens to NCHW",
        },
        Perm {
            shape: &[256, 82944],
            dims: &[1, 0],
            site: "detector.rs:372 instance masks [D, 288*288] to [82944, 256]",
        },
        Perm {
            shape: &[32, 82944],
            dims: &[1, 0],
            site: "mask_decoder.rs:199 upscaled masks to [n, 32]",
        },
        Perm {
            shape: &[1, 5184, 8, 32],
            dims: &[0, 2, 1, 3],
            site: "layers.rs:116 attention heads, D32",
        },
        Perm {
            shape: &[1, 8, 5184, 32],
            dims: &[0, 2, 1, 3],
            site: "layers.rs:122 attention out, D32",
        },
        Perm {
            shape: &[1, 10400, 8, 32],
            dims: &[0, 2, 1, 3],
            site: "layers.rs:116 memory keys/values, D32",
        },
        Perm {
            shape: &[1, 8, 201, 32],
            dims: &[0, 2, 1, 3],
            site: "layers.rs:122 detector queries out",
        },
        Perm {
            shape: &[1, 32, 16, 64],
            dims: &[0, 2, 1, 3],
            site: "text.rs:129 text heads",
        },
    ];

    type Arm<'a> = (String, u64, Box<dyn Fn() + 'a>);

    struct Row {
        name: String,
        min: f64,
        p10: f64,
        med: f64,
        p90: f64,
    }

    fn row(name: String, mut ms: Vec<f64>) -> Row {
        ms.sort_by(f64::total_cmp);
        let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
        Row {
            name,
            min: ms[0],
            p10: at(0.10),
            med: at(0.50),
            p90: at(0.90),
        }
    }

    /// Every round: each arm once, then each arm again as its repeat. Odd
    /// rounds walk the arms in reverse. Returns the number of noisy pairs.
    fn run_arms(
        metal: &MetalBackend,
        label: &str,
        flops: f64,
        arms: &[Arm<'_>],
        warmups: usize,
        runs: usize,
    ) -> usize {
        let count = arms.len();
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(runs); 2 * count];
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
                    (arms[i].2)();
                    metal.sync().expect("sync");
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    if round >= warmups {
                        samples[pass * count + i].push(ms);
                    }
                }
            }
        }
        let mut noisy = 0;
        for (i, (name, bytes, _)) in arms.iter().enumerate() {
            let a = row(name.clone(), samples[i].clone());
            let b = row(format!("{name}'"), samples[count + i].clone());
            let gap = (a.med - b.med).abs();
            let spread = (a.p90 - a.p10).max(b.p90 - b.p10);
            let ok = gap <= spread;
            if !ok {
                noisy += 1;
            }
            for r in [&a, &b] {
                let gbs = *bytes as f64 / (r.med * 1e-3) / 1e9;
                let tflops = if flops > 0.0 {
                    format!("{:.2}", flops / (r.med * 1e-3) / 1e12)
                } else {
                    "-".to_string()
                };
                println!(
                    "{label}\t{}\t{bytes}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{gbs:.1}\t{tflops}\t{gap:.3}\t{spread:.3}\t{}",
                    r.name,
                    r.min,
                    r.p10,
                    r.med,
                    r.p90,
                    if ok { "ok" } else { "NOISY" }
                );
            }
        }
        noisy
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
        let start_load = load();
        println!(
            "# load_start {:.2} {:.2} {:.2}",
            start_load[0], start_load[1], start_load[2]
        );
        println!(
            "shape\tarm\tbytes\tmin_ms\tp10_ms\tmedian_ms\tp90_ms\tgb_per_s\ttflops\tgap_ms\tspread_ms\tpair"
        );
        let mut noisy = 0;

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
            let y = metal.linear_forward(&x, &w).expect("product");
            let elems = (l.m * l.n) as u64;
            let moved = 2 * 4 * elems;
            let arms: Vec<Arm<'_>> = vec![
                (
                    "bias add".to_string(),
                    moved + 4 * l.n as u64,
                    Box::new(|| drop(metal.broadcast_add_forward(&y, &b).expect("bias add"))),
                ),
                (
                    "fill (write floor)".to_string(),
                    4 * elems,
                    Box::new(|| drop(metal.fill(&[l.m, l.n], DType::F32, 1.0).expect("fill"))),
                ),
                (
                    "linear f32 (control)".to_string(),
                    4 * (l.m * l.k + l.n * l.k) as u64 + 4 * elems,
                    Box::new(|| drop(metal.linear_forward(&x, &w).expect("f32"))),
                ),
                (
                    "linear bf16 lane (control)".to_string(),
                    4 * (l.m * l.k + l.n * l.k) as u64 + 4 * elems,
                    Box::new(|| drop(metal.linear_forward(&xt, &wt).expect("bf16"))),
                ),
            ];
            let flops = 2.0 * l.m as f64 * l.k as f64 * l.n as f64;
            noisy += run_arms(&metal, l.name, flops, &arms, warmups, runs);
        }

        for p in PERMS {
            let name = format!("permute {:?} by {:?} ({})", p.shape, p.dims, p.site);
            if !name.contains(&only) {
                continue;
            }
            let numel: usize = p.shape.iter().product();
            let x = up(&host(&values(numel, 4, 1.0), p.shape));
            // The identity permute runs the same scan and a copy with no index
            // arithmetic to speak of: the floor of a permute call.
            let identity: Vec<usize> = (0..p.shape.len()).collect();
            // `relu` is the streaming reference: one read and one write of the
            // same bytes with no index arithmetic.
            let arms: Vec<Arm<'_>> = vec![
                (
                    "permute".to_string(),
                    8 * numel as u64,
                    Box::new(|| drop(metal.permute(&x, p.dims).expect("permute"))),
                ),
                (
                    "permute identity (copy floor)".to_string(),
                    8 * numel as u64,
                    Box::new(|| drop(metal.permute(&x, &identity).expect("identity"))),
                ),
                (
                    "permute (generic kernel)".to_string(),
                    8 * numel as u64,
                    Box::new(|| drop(metal.permute_generic(&x, p.dims).expect("generic"))),
                ),
                (
                    "relu (streaming reference)".to_string(),
                    8 * numel as u64,
                    Box::new(|| drop(metal.relu_forward(&x).expect("relu"))),
                ),
            ];
            noisy += run_arms(&metal, &name, 0.0, &arms, warmups, runs);
        }

        let end_load = load();
        println!(
            "# load_end {:.2} {:.2} {:.2}",
            end_load[0], end_load[1], end_load[2]
        );
        let loads_ok = start_load[0] <= MAX_LOAD && end_load[0] <= MAX_LOAD;
        println!("# noisy_pairs {noisy}");
        println!(
            "# verdict {}",
            if loads_ok && noisy == 0 {
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
    eprintln!("memory_bound_bench needs macOS");
}
