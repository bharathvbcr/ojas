//! SAM 3.1 ViT attention shapes on Metal: `MetalBackend::attention_forward`
//! against the same attention routed through tessl's `encoder_attn`.
//!
//! `cargo run -p ojas-metal --release --example attention_vision_bench [reps]`
//!
//! Shapes, q = k = v, f32, non-causal, no bias, no key lengths, scale
//! 1/sqrt(64), `[B, H, T, D]`:
//! - windowed: `[9, 16, 576, 64]`
//! - global: `[1, 16, 5184, 64]`
//!
//! Two arms on identical host data:
//! - **ojas**: `attention_forward` on device tensors uploaded once, then
//!   `Backend::sync`.
//! - **tessl**: what a route through `tessl::embedgemma2::encoder_attn` costs
//!   end to end: `[B, H, T, D]` to `[B, T, H, D]` permutes of q, k and v,
//!   the kernel, the output permuted back, then `synchronize`. The kernel
//!   takes a per-sequence length buffer; it is all `T`, built once outside
//!   the timed region, because it depends only on `(B, T)` and a routed path
//!   would cache it.
//!
//! `MetalBackend` keeps its tessl runtime on a private device thread, so its
//! tensors cannot be handed to tessl. The tessl arm therefore runs on a
//! `GpuRuntime` of its own, on the same device, with the crate's own overlay
//! metallib loaded, and records each permute as `MetalBackend::permute` does:
//! one `ojas_check_finite` pass over the input (the op refuses a NaN or
//! infinity), then the `ojas_permute` kernel with the same bindings
//! (`fn permute` in `ojas-metal/src/device.rs`). Intermediates are allocated
//! fresh each run, as every `MetalBackend` op allocates its output. Both arms
//! end in one wait for the device; neither downloads inside the timer. The
//! ojas arm also pays the device-thread channel, which the tessl arm does
//! not (`bench/results/2026-10-01-floor/` measures it in microseconds).
//!
//! Correctness first, with nothing timed until it passes. Each arm's output
//! is compared with `CpuBackend` under `Numerics::Exact` (the whole windowed
//! shape; heads 0 and H-1 of the global shape, whose CPU cost is
//! prohibitive in full), and the two arms are compared with each other over
//! the whole output. A comparison passes when `max|a-b| / max|ref|` is at
//! most 1e-4 and every element satisfies the test suite's rule
//! `|a-b| <= 4e-5 sqrt(T) + 1e-4 |ref|`. Each arm's dispatch count is read
//! too: ojas from `MetalBackend::dispatches`, tessl from `infer_trace`.
//!
//! Timing: warm-ups, then `reps` rounds of A, B, A', B' (A' and B' repeat the
//! identical configuration, which is the noise floor). Every run is wall
//! clock around submit and wait.
//!
//! Routing rule, fixed before the first run: route the unmasked `Tq = Tk`,
//! `D` in {64, 256} case through tessl iff, at the global shape, the median
//! of the tessl arm (B and B' pooled) is below the median of the ojas arm
//! (A and A' pooled) by more than the A-vs-A spread, which is p90 - p10 of
//! the pooled A and A' samples.

#[cfg(target_os = "macos")]
mod bench {
    use std::sync::Arc;
    use std::time::Instant;

    use ojas_core::{AttentionOpts, Backend, Budget, Numerics, Tensor};
    use ojas_cpu::CpuBackend;
    use ojas_metal::MetalBackend;
    use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_u32, Binder};
    use tessl::embedgemma2::{
        encoder_attn, encoder_attn_kernel, EncoderAttnDims, EncoderAttnKernel,
    };
    use tessl::infer_trace;
    use tessl::runtime::GpuRuntime;
    use tessl::tensor::GpuBuffer;

    type R<T> = Result<T, String>;

    const GIB: u64 = 1 << 30;
    /// `CHECK_PER_THREAD` in `ojas-metal/src/device.rs`.
    const CHECK_PER_THREAD: usize = 4;
    const WARMUPS: usize = 5;
    /// Relative bound of the stop gate, and the suite's absolute term's scale.
    const REL: f32 = 1e-4;
    const ATOL_SCALE: f32 = 4e-5;

    #[derive(Clone, Copy)]
    struct Shape {
        name: &'static str,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
    }

    impl Shape {
        fn elems(&self) -> usize {
            self.b * self.h * self.t * self.d
        }
        fn dims(&self) -> [usize; 4] {
            [self.b, self.h, self.t, self.d]
        }
    }

    const SHAPES: [Shape; 2] = [
        Shape {
            name: "windowed",
            b: 9,
            h: 16,
            t: 576,
            d: 64,
        },
        Shape {
            name: "global",
            b: 1,
            h: 16,
            t: 5184,
            d: 64,
        },
    ];

    fn e<E: std::fmt::Debug>(what: &str) -> impl FnOnce(E) -> String + '_ {
        move |err| format!("{what}: {err:?}")
    }

    fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn host(data: &[f32], shape: &[usize]) -> R<Tensor> {
        Tensor::from_f32(data, shape, &Budget::new(8 * GIB)).map_err(e("host tensor"))
    }

    fn download(t: &Tensor) -> R<Vec<f32>> {
        t.to_host(&Budget::new(8 * GIB))
            .and_then(|h| h.to_f32_vec())
            .map_err(e("download"))
    }

    /// Host operands of a shape. q and k span [-2, 2) so the scores
    /// (std about 1.3 after the scale) give a peaked softmax and an output
    /// the comparison can discriminate; v spans [-1, 1).
    struct Host {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
    }

    impl Host {
        fn new(s: &Shape) -> Self {
            let n = s.elems();
            Host {
                q: values(n, 11, 2.0),
                k: values(n, 12, 2.0),
                v: values(n, 13, 1.0),
            }
        }
    }

    /// The tessl arm: permute, `encoder_attn`, permute back.
    struct Routed {
        rt: Arc<GpuRuntime>,
        shape: Shape,
        status: GpuBuffer,
        lens: GpuBuffer,
        q: GpuBuffer,
        k: GpuBuffer,
        v: GpuBuffer,
    }

    impl Routed {
        fn new(rt: &Arc<GpuRuntime>, s: Shape, h: &Host) -> R<Self> {
            let bytes = 4 * s.elems();
            let put = |data: &[f32]| -> R<GpuBuffer> {
                let buf = rt.alloc_buffer(bytes)?;
                buf.try_write_f32(data)?;
                Ok(buf)
            };
            let status = rt.alloc_buffer(8 * 4)?;
            status.try_contents_u32()?.fill(0);
            Ok(Routed {
                rt: rt.clone(),
                shape: s,
                status,
                lens: rt.alloc_buffer_from_u32(&vec![s.t as u32; s.b])?,
                q: put(&h.q)?,
                k: put(&h.k)?,
                v: put(&h.v)?,
            })
        }

        /// `MetalBackend::permute` by `[0, 2, 1, 3]`: the finite check on the
        /// input, then `ojas_permute` into a fresh buffer.
        fn permute(&self, x: &GpuBuffer, shape: [usize; 4]) -> R<GpuBuffer> {
            let n = shape.iter().product::<usize>();
            let y = self.rt.alloc_buffer(4 * n)?;
            let strides = [
                shape[1] * shape[2] * shape[3],
                shape[2] * shape[3],
                shape[3],
                1,
            ];
            let oshape = [shape[0], shape[2], shape[1], shape[3]].map(|v| v as u32);
            let istride = [strides[0], strides[2], strides[1], strides[3]].map(|v| v as u32);
            let bytes =
                |v: [u32; 4]| -> Vec<u8> { v.iter().flat_map(|w| w.to_ne_bytes()).collect() };
            let (shape_bytes, stride_bytes) = (bytes(oshape), bytes(istride));
            let n32 = u32::try_from(n).map_err(|_| "element count over u32".to_string())?;
            let check = self.rt.pipeline("ojas_check_finite")?;
            dispatch_1d(
                &self.rt,
                &check,
                n.div_ceil(CHECK_PER_THREAD),
                |b: &mut Binder<'_>| {
                    set_gpu_buf(b, x, 0);
                    set_gpu_buf(b, &self.status, 1);
                    set_u32(b, n32, 2);
                    set_u32(b, 0, 3);
                },
            )?;
            let perm = self.rt.pipeline("ojas_permute")?;
            dispatch_1d(&self.rt, &perm, n, |b: &mut Binder<'_>| {
                set_gpu_buf(b, x, 0);
                set_gpu_buf(b, &y, 1);
                set_u32(b, n32, 2);
                set_u32(b, 4, 3);
                b.bind_bytes(&shape_bytes, 4);
                b.bind_bytes(&stride_bytes, 5);
            })?;
            Ok(y)
        }

        fn dims(&self) -> EncoderAttnDims {
            let s = &self.shape;
            EncoderAttnDims {
                batch: s.b as u32,
                seq: s.t as u32,
                heads: s.h as u32,
                heads_kv: s.h as u32,
                head_dim: s.d as u32,
                window: 0,
                scale: 1.0 / (s.d as f32).sqrt(),
            }
        }

        /// One routed call to device-complete; the output is `[B, H, T, D]`.
        fn run(&self) -> R<GpuBuffer> {
            let s = &self.shape;
            let bhtd = s.dims();
            let qt = self.permute(&self.q, bhtd)?;
            let kt = self.permute(&self.k, bhtd)?;
            let vt = self.permute(&self.v, bhtd)?;
            let o = self.rt.alloc_buffer(4 * s.elems())?;
            encoder_attn(&self.rt, &qt, &kt, &vt, &o, &self.lens, self.dims())?;
            let out = self.permute(&o, [s.b, s.t, s.h, s.d])?;
            self.rt.synchronize()?;
            Ok(out)
        }
    }

    /// The ojas arm.
    struct Direct<'a> {
        m: &'a MetalBackend,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    }

    impl Direct<'_> {
        fn run(&self) -> R<Tensor> {
            let out = self
                .m
                .attention_forward(
                    &self.q,
                    &self.k,
                    &self.v,
                    AttentionOpts {
                        causal: false,
                        window: None,
                        bias: None,
                        key_lens: None,
                    },
                )
                .map_err(e("attention_forward"))?;
            self.m.sync().map_err(e("sync"))?;
            Ok(out)
        }
    }

    /// Milliseconds of `f`, its result dropped after the clock stops.
    fn timed<T>(mut f: impl FnMut() -> R<T>) -> R<f64> {
        let t0 = Instant::now();
        let out = f()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(out);
        Ok(ms)
    }

    struct Diff {
        max_abs: f32,
        max_ref: f32,
        /// Worst `|a-b| / (atol + REL |ref|)` over the elements; at most 1 passes.
        worst: f32,
        nonfinite: usize,
    }

    impl Diff {
        fn rel(&self) -> f32 {
            self.max_abs / self.max_ref
        }
        fn ok(&self) -> bool {
            self.nonfinite == 0 && self.max_ref > 0.0 && self.rel() <= REL && self.worst <= 1.0
        }
    }

    fn diff(got: &[f32], want: &[f32], atol: f32) -> Diff {
        let mut d = Diff {
            max_abs: 0.0,
            max_ref: 0.0,
            worst: 0.0,
            nonfinite: 0,
        };
        for (&a, &b) in got.iter().zip(want) {
            if !a.is_finite() || !b.is_finite() {
                d.nonfinite += 1;
                continue;
            }
            let x = (a - b).abs();
            d.max_abs = d.max_abs.max(x);
            d.max_ref = d.max_ref.max(b.abs());
            d.worst = d.worst.max(x / (atol + REL * b.abs()));
        }
        d
    }

    fn report_diff(label: &str, d: &Diff) {
        println!(
            "| {label} | {:.3e} | {:.4} | {:.3e} | {:.3} | {} |",
            d.max_abs,
            d.max_ref,
            d.rel(),
            d.worst,
            if d.ok() { "pass" } else { "FAIL" }
        );
    }

    fn pct(sorted: &[f64], p: f64) -> f64 {
        sorted[((sorted.len() - 1) as f64 * p).round() as usize]
    }

    struct Stats {
        min: f64,
        p10: f64,
        median: f64,
        p90: f64,
        max: f64,
    }

    fn stats(samples: &[f64]) -> Stats {
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        Stats {
            min: s[0],
            p10: pct(&s, 0.1),
            median: pct(&s, 0.5),
            p90: pct(&s, 0.9),
            max: s[s.len() - 1],
        }
    }

    fn row(name: &str, n: usize, s: &Stats) {
        println!(
            "| {name} | {n} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            s.min,
            s.median,
            s.p10,
            s.p90,
            s.max,
            s.p90 - s.p10
        );
    }

    fn gpu_util() -> String {
        match std::process::Command::new("ioreg")
            .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
            .output()
        {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout);
                let tag = "\"Device Utilization %\"=";
                text.find(tag)
                    .and_then(|i| {
                        text[i + tag.len()..]
                            .split(|c: char| !c.is_ascii_digit())
                            .next()
                            .map(|d| format!("{d}%"))
                    })
                    .unwrap_or_else(|| "not reported".to_string())
            }
            Err(err) => format!("ioreg failed: {err}"),
        }
    }

    fn load() -> String {
        match std::process::Command::new("uptime").output() {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout).into_owned();
                s.split("load average")
                    .nth(1)
                    .map(|t| t.trim_start_matches(['s', ':', ' ']).trim().to_string())
                    .unwrap_or(s)
            }
            Err(err) => format!("uptime failed: {err}"),
        }
    }

    /// Both arms agree with the CPU reference and with each other, and each
    /// ran its kernels. Returns the failure text, if any.
    fn check(m: &MetalBackend, s: &Shape, h: &Host, direct: &Direct, routed: &Routed) -> R<bool> {
        let atol = ATOL_SCALE * (s.t as f32).sqrt();
        let cpu = CpuBackend::new(Budget::new(8 * GIB)).with_numerics(Numerics::Exact);
        let d0 = m.dispatches();
        let ojas = download(&direct.run()?)?;
        let ojas_dispatches = m.dispatches() - d0;
        infer_trace::set_enabled(true);
        let before = infer_trace::snapshot();
        let out = routed.run()?;
        let after = infer_trace::snapshot();
        infer_trace::set_enabled(false);
        let tessl = out.try_contents_f32()?.to_vec();
        println!(
            "\n{}: dispatches per call: ojas {ojas_dispatches} (MetalBackend::dispatches), tessl {} (infer_trace; 4 x [check + permute] + 1 kernel = 9 expected)",
            s.name,
            after.dispatches - before.dispatches
        );
        println!("\n| {} comparison | max abs diff | max abs ref | max abs diff / max abs ref | worst suite ratio | verdict |", s.name);
        println!("|---|---:|---:|---:|---:|---|");
        let mut pass = true;
        let mut cmp = |label: &str, got: &[f32], want: &[f32]| {
            let d = diff(got, want, atol);
            report_diff(label, &d);
            pass &= d.ok();
        };
        cmp("ojas vs tessl, whole output", &ojas, &tessl);
        // CPU reference: the whole shape when it is cheap, else head planes.
        let plane = s.t * s.d;
        let planes: Vec<usize> = if s.b * s.h <= 144 && s.t <= 1024 {
            (0..s.b * s.h).collect()
        } else {
            vec![0, s.h - 1]
        };
        if planes.len() == s.b * s.h {
            let want = cpu
                .attention_forward(
                    &host(&h.q, &s.dims())?,
                    &host(&h.k, &s.dims())?,
                    &host(&h.v, &s.dims())?,
                    AttentionOpts {
                        causal: false,
                        window: None,
                        bias: None,
                        key_lens: None,
                    },
                )
                .map_err(e("cpu attention_forward"))?;
            let want = want.to_f32_vec().map_err(e("cpu output"))?;
            cmp("ojas vs CPU Exact, whole output", &ojas, &want);
            cmp("tessl vs CPU Exact, whole output", &tessl, &want);
        } else {
            for p in planes {
                let at = p * plane..(p + 1) * plane;
                let one = [1, 1, s.t, s.d];
                let want = cpu
                    .attention_forward(
                        &host(&h.q[at.clone()], &one)?,
                        &host(&h.k[at.clone()], &one)?,
                        &host(&h.v[at.clone()], &one)?,
                        AttentionOpts {
                            causal: false,
                            window: None,
                            bias: None,
                            key_lens: None,
                        },
                    )
                    .map_err(e("cpu attention_forward"))?;
                let want = want.to_f32_vec().map_err(e("cpu output"))?;
                cmp(
                    &format!("ojas vs CPU Exact, plane {p}"),
                    &ojas[at.clone()],
                    &want,
                );
                cmp(&format!("tessl vs CPU Exact, plane {p}"), &tessl[at], &want);
            }
        }
        Ok(pass)
    }

    pub fn main() -> R<()> {
        let reps: usize = std::env::args()
            .nth(1)
            .and_then(|a| a.parse().ok())
            .unwrap_or(25);
        if reps < 15 {
            return Err(format!("reps {reps} is under the 15 the method requires"));
        }
        let m = MetalBackend::new(Budget::new(8 * GIB)).map_err(e("MetalBackend::new"))?;
        let rt = GpuRuntime::new()?;
        // The pool cap `Worker::open` gives a backend opened with 8 GiB.
        rt.set_pool_cache_cap_bytes(GIB as usize);
        rt.set_async_encode(true)?;
        rt.add_metallib_bytes(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/ojas_per_head_gate.metallib"
        )))?;
        println!(
            "device: {}, tensorops {}",
            m.device_name(),
            rt.has_tensorops()
        );
        println!(
            "tessl kernel picked at D=64: {:?}",
            encoder_attn_kernel(&rt, 64)
        );
        if encoder_attn_kernel(&rt, 64) != EncoderAttnKernel::Tiled {
            return Err("tessl would not run the tiled kernel here; the verdict would be about another kernel".into());
        }
        println!("start: GPU {}, load {}", gpu_util(), load());
        println!("{reps} reps of A, B, A', B' after {WARMUPS} warm-ups of each");

        let mut verdict = None;
        for s in SHAPES {
            println!(
                "\n## {} [B={}, H={}, T={}, D={}]",
                s.name, s.b, s.h, s.t, s.d
            );
            let h = Host::new(&s);
            let dev = |data: &[f32]| -> R<Tensor> {
                m.upload(&host(data, &s.dims())?).map_err(e("upload"))
            };
            let direct = Direct {
                m: &m,
                q: dev(&h.q)?,
                k: dev(&h.k)?,
                v: dev(&h.v)?,
            };
            let routed = Routed::new(&rt, s, &h)?;
            if !check(&m, &s, &h, &direct, &routed)? {
                return Err(format!(
                    "{}: the arms disagree beyond the bound; no timings",
                    s.name
                ));
            }
            for _ in 0..WARMUPS {
                timed(|| direct.run())?;
                timed(|| routed.run())?;
            }
            let (mut a, mut b, mut a2, mut b2) = (vec![], vec![], vec![], vec![]);
            for _ in 0..reps {
                a.push(timed(|| direct.run())?);
                b.push(timed(|| routed.run())?);
                a2.push(timed(|| direct.run())?);
                b2.push(timed(|| routed.run())?);
            }
            let pool = |x: &[f64], y: &[f64]| [x, y].concat();
            let (ao, bo) = (pool(&a, &a2), pool(&b, &b2));
            println!("\n| arm | n | min ms | median | p10 | p90 | max | p90 - p10 |");
            println!("|---|---:|---:|---:|---:|---:|---:|---:|");
            row("ojas A", reps, &stats(&a));
            row("ojas A' (same config)", reps, &stats(&a2));
            row("tessl B (with permutes)", reps, &stats(&b));
            row("tessl B' (same config)", reps, &stats(&b2));
            let (sa, sb) = (stats(&ao), stats(&bo));
            row("ojas A + A' pooled", ao.len(), &sa);
            row("tessl B + B' pooled", bo.len(), &sb);
            let mut paired: Vec<f64> = a.iter().zip(&a2).map(|(x, y)| (x - y).abs()).collect();
            paired.sort_by(f64::total_cmp);
            let spread = sa.p90 - sa.p10;
            println!(
                "\nnoise floor (A vs A'): |median A - median A'| = {:.3} ms, median per-round |A - A'| = {:.3} ms, pooled p90 - p10 = {:.3} ms",
                (stats(&a).median - stats(&a2).median).abs(),
                pct(&paired, 0.5),
                spread
            );
            println!(
                "tessl median {:.3} ms vs ojas median {:.3} ms: ojas - tessl = {:.3} ms (tessl/ojas = {:.3}), spread {:.3} ms",
                sb.median,
                sa.median,
                sa.median - sb.median,
                sb.median / sa.median,
                spread
            );
            if s.name == "global" {
                verdict = Some((sa.median - sb.median > spread, sa.median, sb.median, spread));
            }
        }
        println!("\nend: GPU {}, load {}", gpu_util(), load());
        if let Some((route, ojas, tessl, spread)) = verdict {
            println!(
                "\nrouting rule (global shape): tessl median {tessl:.3} ms, ojas median {ojas:.3} ms, A-vs-A spread {spread:.3} ms: ojas - tessl = {:.3} ms {} spread -> {}",
                ojas - tessl,
                if route { ">" } else { "<=" },
                if route { "ROUTE through tessl" } else { "DO NOT ROUTE (keep ojas_attn_gen_fwd)" }
            );
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(err) = bench::main() {
        eprintln!("attention_vision_bench: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("attention_vision_bench needs macOS and the `metal` feature");
}
