//! SpikeTrack (SAM 3.1) attention shapes on Metal: the scalar
//! `ojas_attn_gen_fwd` behind `MetalBackend::attention_forward` against
//! tessl's tiled `encoder_attn`, where a route exists, plus the cost of the
//! shapes that stay on the scalar kernel. The results decide which Metal
//! kernels get built next.
//!
//! `cargo run -p ojas-metal --release --example attention_spiketrack_bench [reps] [out_dir] [rows]`
//!
//! `reps` defaults to 25 (at least 15). `out_dir`, when given, receives
//! `rounds.csv` (every timed run) and `rows.jsonl` (one summary per row).
//! `rows` is a comma list of row ids (default: all, in this order):
//! `a b c1 c2 c3 d e f33 f20 g g20 bx`. `bx` is the extra
//! `[16, 8, 200, 32]` detector shape with the largest batch the Budget lets
//! through (16, 8, 4, 2). The crossover rows `x5200 x10400 x10624 x15616`
//! (alias `xover`, all four) are opt-in and described at the end of this
//! header.
//!
//! The scalar arm is genuinely scalar: this crate has no routing to
//! `encoder_attn` at the commit this was run from (no
//! `routes_to_encoder_attn` in `ojas-metal/src`), so every `ojas` arm is
//! `ojas_attn_gen_fwd`.
//!
//! Rows, `[B, H, T, D]`, f32, scale 1/sqrt(D):
//! - a: self-attention `[1, 8, 5184, 32]`, no mask, x10 per frame (6 detector
//!   encoder + 4 memory attention)
//! - b: detector cross-attention, q `[1, 8, 201, 32]`, k and v
//!   `[1, 8, 5184, 32]`, bias `[1, 8, 201, 5184]`, x6; scalar only
//! - c1..c3: memory cross-attention, q `[1, 8, 5184, 32]`, k and v
//!   `[1, 8, m, 32]` for m = 5200, 15616, 36544, x4
//! - d: decoder, q `[1, 8, 16, 16]`, k and v `[1, 8, 4096, 16]`; scalar only
//! - e: decoder self-attention `[1, 8, 201, 32]`, x6
//! - f33, f20: encoder to text, q `[1, 8, 5184, 32]`, k and v
//!   `[1, 8, 33, 32]`, key_lens `[33]` / `[20]`, x6; scalar only
//! - g, g20: decoder to text, q `[1, 8, 201, 32]`, k and v `[1, 8, 33, 32]`,
//!   no mask / key_lens `[20]`, x6; scalar only
//!
//! Arms:
//! - **ojas**: `attention_forward` on device tensors uploaded once, then
//!   `Backend::sync`.
//! - **tessl**: what a route through `tessl::embedgemma2::encoder_attn` costs
//!   end to end: the `[B, H, T, D]` to `[B, T, H, D]` permutes of q, k and v
//!   (each preceded by the finite check `MetalBackend::permute` runs), the
//!   kernel, the output permuted back, then `synchronize`. The length buffer
//!   is cached outside the timer, as a routed path would cache it.
//! - **tessl, q padded** (rows c only, `Tq < Tk`): `encoder_attn` takes one
//!   sequence length for q and k, so q is padded with zero rows to `Tk` and
//!   `lens = Tk`; the output is sliced back to `Tq` rows. Two variants, both
//!   timed: *fused* writes the permuted q straight into the padded buffer
//!   (one device copy for the zero tail) and reads the first `Tq` rows in
//!   the closing permute; *explicit* permutes q to `Tq` rows, copies it into
//!   a padded buffer, and copies the first `Tq` output rows to a buffer
//!   before the closing permute (copies are tessl's `copy_u32` kernel).
//!   The explicit variant is the one the rule is judged on. The zero source
//!   is cached outside the timer.
//!
//! `MetalBackend` keeps its tessl runtime on a private device thread, so its
//! tensors cannot be handed to tessl; the tessl arms run on a `GpuRuntime` of
//! their own on the same device, as `attention_vision_bench` does.
//!
//! Correctness first, with nothing timed until it passes: every arm is
//! compared with `CpuBackend` under `Numerics::Exact` on the same inputs
//! (every head plane when cheap, else the first and last, and only the first
//! when a plane is over 3e9 multiply-adds), and the arms with each other over
//! the whole output. A comparison passes when `max|a-b| / max|ref|` is at
//! most 1e-4 and every element satisfies `|a-b| <= 4e-5 sqrt(Tk) + 1e-4
//! |ref|`. Each arm's dispatch count is read and compared with what the
//! intended path issues (tessl: `infer_trace`; ojas: `MetalBackend::dispatches`).
//!
//! Timing: 5 warm-ups of each arm, then `reps` rounds, each of two passes
//! over the arms in order (arm 1, arm 2, ..., arm 1', arm 2', ...). The
//! primed pass repeats the identical configuration, so arm 1 vs arm 1' is the
//! A-vs-A noise floor. Every run is wall clock around submit and wait.
//!
//! Rules (fixed before the first measurement; summary.md states them):
//! see `bench/results/2026-10-27-attention-spiketrack/summary.md`.
//!
//! # Crossover rows: tiled kernel against the pad-q route
//!
//! `x<m>` times memory cross-attention, q `[1, 8, 5184, 32]` and k and v
//! `[1, 8, m, 32]` (no bias, no key lengths, non-causal), through the two
//! `MetalBackend` entry points that force each kernel:
//! - **tiled**: `attention_forward_saving`, which never routes to tessl and
//!   runs the split-K tiled kernel at head dim 32 (its log-sum-exp output is
//!   dropped);
//! - **pad-q**: `attention_forward`, which pads Q to Tk and runs tessl's
//!   `encoder_attn` where `routes_to_encoder_attn` says so.
//!
//! Each arm owns its uploads. Before any timing, for every m: the dispatch
//! counts of both arms, on fresh uploads and again on the same tensors, must
//! equal those of a small yardstick shape that is known to take each route
//! (`[1, 8, 40, 32]` against 80 keys, plus the causal scalar call on 40
//! keys), the two arms must agree within 1e-5 max abs over the whole output,
//! and both must agree with `CpuBackend` under `Numerics::Exact`. Timing is
//! 5 warm-ups of each arm, then `reps` rounds of two passes; odd rounds
//! reverse the arm order inside each pass, so no arm is always second. The
//! verdict rules are `bench/results/2026-10-27-tiled-vs-padq/rules.md`.

#[cfg(target_os = "macos")]
mod bench {
    use std::fmt::Write as _;
    use std::sync::Arc;
    use std::time::Instant;

    use ojas_core::{AttentionOpts, Backend, Budget, Numerics, Tensor};
    use ojas_cpu::CpuBackend;
    use ojas_metal::MetalBackend;
    use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_gpu_buf_offset, set_u32, Binder};
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
    const REL: f32 = 1e-4;
    const ATOL_SCALE: f32 = 4e-5;
    /// Bias-kernel threshold: 10% of the CoreML port's frame on this M5 Pro,
    /// 0.272 s backbone + 0.0775 s track step = 0.3495 s (SpikeTrack
    /// bench/results/baselines/coreml.json, median of 5, contended).
    const BIAS_THRESHOLD_MS: f64 = 35.0;
    /// The provisional threshold the rule was first stated with: 10% of the
    /// port's published 2.75 s (2.05 s backbone + 0.7 s tracker, SpikeTrack
    /// HANDOFF.md:16).
    const BIAS_THRESHOLD_PROVISIONAL_MS: f64 = 275.0;
    const ALL_ROWS: [&str; 12] = [
        "a", "b", "c1", "c2", "c3", "d", "e", "f33", "f20", "g", "g20", "bx",
    ];

    #[derive(Clone, Copy)]
    struct Case {
        b: usize,
        h: usize,
        tq: usize,
        tk: usize,
        d: usize,
        bias: bool,
        lens: Option<u32>,
    }

    impl Case {
        fn q_dims(&self) -> [usize; 4] {
            [self.b, self.h, self.tq, self.d]
        }
        fn kv_dims(&self) -> [usize; 4] {
            [self.b, self.h, self.tk, self.d]
        }
        fn bias_dims(&self) -> [usize; 4] {
            [self.b, self.h, self.tq, self.tk]
        }
        fn q_elems(&self) -> usize {
            self.q_dims().iter().product()
        }
        fn kv_elems(&self) -> usize {
            self.kv_dims().iter().product()
        }
        fn bias_elems(&self) -> usize {
            self.bias_dims().iter().product()
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Pad {
        None,
        Fused,
        Explicit,
    }

    #[derive(Clone, Copy)]
    enum Rule {
        /// Route through tessl iff the median margin beats the A-vs-A spread
        /// and the sign holds in every paired round.
        Route,
        /// A tiled bias kernel is warranted iff per-frame cost exceeds this.
        Threshold(f64),
        /// No rule; the cost is reported.
        Info,
    }

    struct Row {
        id: &'static str,
        title: &'static str,
        case: Case,
        tessl: Vec<Pad>,
        /// Layers per frame, `None` when the count is not known.
        layers: Option<u32>,
        rule: Rule,
    }

    fn rows() -> Vec<Row> {
        let c = |tq, tk, d, bias, lens| Case {
            b: 1,
            h: 8,
            tq,
            tk,
            d,
            bias,
            lens,
        };
        let mem = |id, title, m| Row {
            id,
            title,
            case: c(5184, m, 32, false, None),
            tessl: vec![Pad::Fused, Pad::Explicit],
            layers: Some(4),
            rule: Rule::Route,
        };
        vec![
            Row {
                id: "a",
                title: "D32 self-attention [1,8,5184,32]",
                case: c(5184, 5184, 32, false, None),
                tessl: vec![Pad::None],
                layers: Some(10),
                rule: Rule::Route,
            },
            Row {
                id: "b",
                title: "detector cross-attention with bias q [1,8,201,32] kv [1,8,5184,32] bias [1,8,201,5184]",
                case: c(201, 5184, 32, true, None),
                tessl: vec![],
                layers: Some(6),
                rule: Rule::Threshold(BIAS_THRESHOLD_MS),
            },
            mem("c1", "memory cross-attention m=5200 (frame 1)", 5200),
            mem("c2", "memory cross-attention m=15616", 15616),
            mem("c3", "memory cross-attention m=36544 (max)", 36544),
            Row {
                id: "d",
                title: "decoder D16 q [1,8,16,16] kv [1,8,4096,16]",
                case: c(16, 4096, 16, false, None),
                tessl: vec![],
                layers: None,
                rule: Rule::Info,
            },
            Row {
                id: "e",
                title: "decoder self-attention [1,8,201,32]",
                case: c(201, 201, 32, false, None),
                tessl: vec![Pad::None],
                layers: Some(6),
                rule: Rule::Route,
            },
            Row {
                id: "f33",
                title: "encoder to text q [1,8,5184,32] kv [1,8,33,32] key_lens [33]",
                case: c(5184, 33, 32, false, Some(33)),
                tessl: vec![],
                layers: Some(6),
                rule: Rule::Info,
            },
            Row {
                id: "f20",
                title: "encoder to text q [1,8,5184,32] kv [1,8,33,32] key_lens [20]",
                case: c(5184, 33, 32, false, Some(20)),
                tessl: vec![],
                layers: Some(6),
                rule: Rule::Info,
            },
            Row {
                id: "g",
                title: "decoder to text q [1,8,201,32] kv [1,8,33,32]",
                case: c(201, 33, 32, false, None),
                tessl: vec![],
                layers: Some(6),
                rule: Rule::Info,
            },
            Row {
                id: "g20",
                title: "decoder to text q [1,8,201,32] kv [1,8,33,32] key_lens [20]",
                case: c(201, 33, 32, false, Some(20)),
                tessl: vec![],
                layers: Some(6),
                rule: Rule::Info,
            },
        ]
    }

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

    fn host_lens(len: u32, batch: usize) -> R<Tensor> {
        Tensor::from_u32(&vec![len; batch], &[batch], &Budget::new(GIB)).map_err(e("host lens"))
    }

    fn download(t: &Tensor) -> R<Vec<f32>> {
        t.to_host(&Budget::new(8 * GIB))
            .and_then(|h| h.to_f32_vec())
            .map_err(e("download"))
    }

    /// Host operands. q and k span [-2, 2) so the scores give a peaked
    /// softmax and an output the comparison can discriminate; v and the bias
    /// span [-1, 1).
    struct Host {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        bias: Option<Vec<f32>>,
    }

    impl Host {
        fn new(c: &Case) -> Self {
            Host {
                q: values(c.q_elems(), 11, 2.0),
                k: values(c.kv_elems(), 12, 2.0),
                v: values(c.kv_elems(), 13, 1.0),
                bias: c.bias.then(|| values(c.bias_elems(), 14, 1.0)),
            }
        }
    }

    trait Arm {
        fn run(&self) -> R<()>;
    }

    /// Which `MetalBackend` entry point a [`Direct`] arm calls.
    #[derive(Clone, Copy, PartialEq)]
    enum Entry {
        /// `attention_forward`: tessl's `encoder_attn` (Q padded to Tk) where
        /// `routes_to_encoder_attn` says so, else the general kernel.
        Forward,
        /// `attention_forward_saving`: never routes; the split-K tiled kernel
        /// at head dim 16 or 32, non-causal.
        Saving,
        /// `attention_forward_saving` with `causal: true`: only the scalar
        /// kernel runs it. The dispatch yardstick of the crossover rows.
        Scalar,
    }

    /// The ojas arm.
    struct Direct<'a> {
        m: &'a MetalBackend,
        entry: Entry,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Option<Tensor>,
        lens: Option<Tensor>,
    }

    impl<'a> Direct<'a> {
        fn new(m: &'a MetalBackend, c: &Case, h: &Host) -> R<Self> {
            let dev = |data: &[f32], shape: &[usize]| -> R<Tensor> {
                m.upload(&host(data, shape)?).map_err(e("upload"))
            };
            Ok(Direct {
                m,
                entry: Entry::Forward,
                q: dev(&h.q, &c.q_dims())?,
                k: dev(&h.k, &c.kv_dims())?,
                v: dev(&h.v, &c.kv_dims())?,
                bias: h
                    .bias
                    .as_ref()
                    .map(|b| dev(b, &c.bias_dims()))
                    .transpose()?,
                lens: c
                    .lens
                    .map(|l| m.upload(&host_lens(l, c.b)?).map_err(e("upload lens")))
                    .transpose()?,
            })
        }

        fn with_entry(self, entry: Entry) -> Self {
            Direct { entry, ..self }
        }

        fn out(&self) -> R<Tensor> {
            let opts = AttentionOpts {
                causal: self.entry == Entry::Scalar,
                window: None,
                bias: self.bias.as_ref(),
                key_lens: self.lens.as_ref(),
            };
            let out = match self.entry {
                Entry::Forward => self
                    .m
                    .attention_forward(&self.q, &self.k, &self.v, opts)
                    .map_err(e("attention_forward"))?,
                Entry::Saving | Entry::Scalar => {
                    self.m
                        .attention_forward_saving(&self.q, &self.k, &self.v, opts)
                        .map_err(e("attention_forward_saving"))?
                        .0
                }
            };
            self.m.sync().map_err(e("sync"))?;
            Ok(out)
        }
    }

    impl Arm for Direct<'_> {
        fn run(&self) -> R<()> {
            self.out().map(drop)
        }
    }

    /// The tessl arm: permute, `encoder_attn`, permute back, with q padded
    /// to the key length when `pad` is not `None`.
    struct Routed {
        rt: Arc<GpuRuntime>,
        case: Case,
        pad: Pad,
        status: GpuBuffer,
        lens: GpuBuffer,
        zeros: Option<GpuBuffer>,
        q: GpuBuffer,
        k: GpuBuffer,
        v: GpuBuffer,
    }

    impl Routed {
        fn new(rt: &Arc<GpuRuntime>, c: Case, pad: Pad, h: &Host) -> R<Self> {
            if pad == Pad::None && c.tq != c.tk {
                return Err("encoder_attn without padding needs Tq == Tk".into());
            }
            if pad != Pad::None && (c.b != 1 || c.tq >= c.tk) {
                return Err("the pad route is built for B = 1 and Tq < Tk".into());
            }
            let put = |data: &[f32]| -> R<GpuBuffer> {
                let buf = rt.alloc_buffer(4 * data.len())?;
                buf.try_write_f32(data)?;
                Ok(buf)
            };
            let status = rt.alloc_buffer(8 * 4)?;
            status.try_contents_u32()?.fill(0);
            let seq = if pad == Pad::None { c.tq } else { c.tk };
            let zeros = if pad == Pad::None {
                None
            } else {
                Some(rt.alloc_buffer_from_u32(&vec![0u32; (c.tk - c.tq) * c.h * c.d])?)
            };
            Ok(Routed {
                rt: rt.clone(),
                case: c,
                pad,
                status,
                lens: rt.alloc_buffer_from_u32(&vec![seq as u32; c.b])?,
                zeros,
                q: put(&h.q)?,
                k: put(&h.k)?,
                v: put(&h.v)?,
            })
        }

        /// `MetalBackend::permute` by `[0, 2, 1, 3]` over the first
        /// `prod(shape)` elements of `x`: the finite check, then
        /// `ojas_permute` into `y`.
        fn permute_into(&self, x: &GpuBuffer, shape: [usize; 4], y: &GpuBuffer) -> R<()> {
            let n = shape.iter().product::<usize>();
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
                set_gpu_buf(b, y, 1);
                set_u32(b, n32, 2);
                set_u32(b, 4, 3);
                b.bind_bytes(&shape_bytes, 4);
                b.bind_bytes(&stride_bytes, 5);
            })
        }

        fn permute(&self, x: &GpuBuffer, shape: [usize; 4]) -> R<GpuBuffer> {
            let y = self.rt.alloc_buffer(4 * shape.iter().product::<usize>())?;
            self.permute_into(x, shape, &y)?;
            Ok(y)
        }

        /// `dst[dst_off..] = src[src_off..]`, `n` f32 words, by tessl's
        /// bitwise `copy_u32` kernel.
        fn copy(
            &self,
            src: &GpuBuffer,
            src_off: usize,
            dst: &GpuBuffer,
            dst_off: usize,
            n: usize,
        ) -> R<()> {
            let n32 = u32::try_from(n).map_err(|_| "copy over u32".to_string())?;
            let p = self.rt.pipeline("copy_u32")?;
            dispatch_1d(&self.rt, &p, n, |b: &mut Binder<'_>| {
                set_gpu_buf_offset(b, src, 4 * src_off, 0);
                set_gpu_buf_offset(b, dst, 4 * dst_off, 1);
                set_u32(b, n32, 2);
            })
        }

        /// Dispatches one run issues: 2 per permute (check + permute), one
        /// per copy, one kernel.
        fn expected_dispatches(&self) -> usize {
            match self.pad {
                Pad::None => 4 * 2 + 1,
                Pad::Fused => 4 * 2 + 1 + 1,
                Pad::Explicit => 4 * 2 + 3 + 1,
            }
        }

        /// One call to device-complete; the buffer holds `[B, H, Tq, D]`
        /// in its first `B * H * Tq * D` words.
        fn out(&self) -> R<GpuBuffer> {
            let c = &self.case;
            let rows = if self.pad == Pad::None { c.tq } else { c.tk };
            let row_words = c.h * c.d;
            let kt = self.permute(&self.k, c.kv_dims())?;
            let vt = self.permute(&self.v, c.kv_dims())?;
            let zeros = |this: &Self, y: &GpuBuffer| -> R<()> {
                let z = this.zeros.as_ref().ok_or("no zero source")?;
                this.copy(z, 0, y, c.tq * row_words, (c.tk - c.tq) * row_words)
            };
            let qp = match self.pad {
                Pad::None => self.permute(&self.q, c.q_dims())?,
                Pad::Fused => {
                    let y = self.rt.alloc_buffer(4 * rows * row_words)?;
                    zeros(self, &y)?;
                    self.permute_into(&self.q, c.q_dims(), &y)?;
                    y
                }
                Pad::Explicit => {
                    let qt = self.permute(&self.q, c.q_dims())?;
                    let y = self.rt.alloc_buffer(4 * rows * row_words)?;
                    zeros(self, &y)?;
                    self.copy(&qt, 0, &y, 0, c.tq * row_words)?;
                    y
                }
            };
            let o = self.rt.alloc_buffer(4 * c.b * rows * row_words)?;
            let dims = EncoderAttnDims {
                batch: c.b as u32,
                seq: rows as u32,
                heads: c.h as u32,
                heads_kv: c.h as u32,
                head_dim: c.d as u32,
                window: 0,
                scale: 1.0 / (c.d as f32).sqrt(),
            };
            encoder_attn(&self.rt, &qp, &kt, &vt, &o, &self.lens, dims)?;
            let src = match self.pad {
                Pad::Explicit => {
                    let os = self.rt.alloc_buffer(4 * c.tq * row_words)?;
                    self.copy(&o, 0, &os, 0, c.tq * row_words)?;
                    os
                }
                _ => o,
            };
            let out = self.permute(&src, [c.b, c.tq, c.h, c.d])?;
            self.rt.synchronize()?;
            Ok(out)
        }
    }

    impl Arm for Routed {
        fn run(&self) -> R<()> {
            self.out().map(drop)
        }
    }

    fn timed(arm: &dyn Arm) -> R<f64> {
        let t0 = Instant::now();
        arm.run()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    }

    struct Diff {
        max_abs: f32,
        max_ref: f32,
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

    fn pct(sorted: &[f64], p: f64) -> f64 {
        sorted[((sorted.len() - 1) as f64 * p).round() as usize]
    }

    #[derive(Clone, Copy)]
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

    fn print_row(name: &str, n: usize, s: &Stats) {
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

    fn command_line(cmd: &str, args: &[&str]) -> Option<String> {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
    }

    fn gpu_util() -> String {
        let text = command_line(
            "ioreg",
            &["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
        )
        .unwrap_or_default();
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

    fn load() -> String {
        let s = command_line("uptime", &[]).unwrap_or_default();
        s.split("load average")
            .nth(1)
            .map(|t| t.trim_start_matches(['s', ':', ' ']).trim().to_string())
            .unwrap_or(s)
    }

    /// One arm's max error against a reference, for the record.
    #[derive(Default)]
    struct Errors {
        /// (label, max abs diff, max abs diff / max abs ref, worst suite ratio, pass)
        rows: Vec<(String, f32, f32, f32, bool)>,
    }

    impl Errors {
        fn cmp(&mut self, label: &str, got: &[f32], want: &[f32], atol: f32) {
            let d = diff(got, want, atol);
            println!(
                "| {label} | {:.3e} | {:.4} | {:.3e} | {:.3} | {} |",
                d.max_abs,
                d.max_ref,
                d.rel(),
                d.worst,
                if d.ok() { "pass" } else { "FAIL" }
            );
            self.rows
                .push((label.to_string(), d.max_abs, d.rel(), d.worst, d.ok()));
        }
        fn pass(&self) -> bool {
            self.rows.iter().all(|r| r.4)
        }
    }

    fn label(p: Pad) -> &'static str {
        match p {
            Pad::None => "tessl (with permutes)",
            Pad::Fused => "tessl, q padded, fused slice",
            Pad::Explicit => "tessl, q padded, explicit pad+slice copies",
        }
    }

    /// All arms agree with the CPU reference and with each other, and each
    /// ran the intended kernels. Returns the errors and the dispatch counts.
    fn check(
        m: &MetalBackend,
        c: &Case,
        h: &Host,
        direct: &Direct,
        routed: &[Routed],
    ) -> R<(Errors, String)> {
        let atol = ATOL_SCALE * (c.tk as f32).sqrt();
        let d0 = m.dispatches();
        let ojas = download(&direct.out()?)?;
        let ojas_dispatches = m.dispatches() - d0;
        let mut dispatch_note = format!("ojas {ojas_dispatches} (MetalBackend::dispatches)");
        let n = c.q_elems();
        let mut outs: Vec<(&'static str, Vec<f32>)> = vec![];
        for r in routed {
            infer_trace::set_enabled(true);
            let before = infer_trace::snapshot();
            let out = r.out()?;
            let after = infer_trace::snapshot();
            infer_trace::set_enabled(false);
            let got = after.dispatches - before.dispatches;
            let words = out.try_contents_f32()?;
            if words.len() < n {
                return Err(format!(
                    "tessl output holds {} words, need {n}",
                    words.len()
                ));
            }
            outs.push((label(r.pad), words[..n].to_vec()));
            let want = r.expected_dispatches();
            let _ = write!(
                dispatch_note,
                ", {} {got} (infer_trace; expected {want})",
                label(r.pad)
            );
            if got as usize != want {
                return Err(format!(
                    "{}: {got} dispatches, the intended path issues {want}",
                    label(r.pad)
                ));
            }
        }
        println!("dispatches per call: {dispatch_note}");
        println!("\n| comparison | max abs diff | max abs ref | max abs diff / max abs ref | worst suite ratio | verdict |");
        println!("|---|---:|---:|---:|---:|---|");
        let mut errs = Errors::default();
        for (name, got) in &outs {
            errs.cmp(&format!("ojas vs {name}, whole output"), &ojas, got, atol);
        }
        let mut arms: Vec<(&str, &[f32])> = vec![("ojas", &ojas)];
        arms.extend(outs.iter().map(|(name, got)| (*name, got.as_slice())));
        compare_cpu(c, h, &arms, &mut errs)?;
        Ok((errs, dispatch_note))
    }

    /// Each of `arms` (a label and a whole `[B, H, Tq, D]` output) against
    /// `CpuBackend` under `Numerics::Exact` on the same inputs: every head
    /// plane when cheap, else the first and last, and only the first when a
    /// plane is over 3e9 multiply-adds.
    fn compare_cpu(c: &Case, h: &Host, arms: &[(&str, &[f32])], errs: &mut Errors) -> R<()> {
        let atol = ATOL_SCALE * (c.tk as f32).sqrt();
        let cpu = CpuBackend::new(Budget::new(8 * GIB)).with_numerics(Numerics::Exact);
        let plane_q = c.tq * c.d;
        let planes_n = c.b * c.h;
        let plane_cost = c.tq * c.tk * c.d;
        let planes: Vec<usize> = if planes_n * plane_cost <= 300_000_000 {
            (0..planes_n).collect()
        } else if plane_cost > 3_000_000_000 {
            vec![0]
        } else {
            vec![0, planes_n - 1]
        };
        for p in planes {
            let one = |data: &[f32], plane: usize, rows: usize, cols: usize| {
                host(
                    &data[plane * rows * cols..(plane + 1) * rows * cols],
                    &[1, 1, rows, cols],
                )
            };
            let bias = h.bias.as_ref().map(|b| one(b, p, c.tq, c.tk)).transpose()?;
            let lens = c.lens.map(|l| host_lens(l, 1)).transpose()?;
            let want = cpu
                .attention_forward(
                    &one(&h.q, p, c.tq, c.d)?,
                    &one(&h.k, p, c.tk, c.d)?,
                    &one(&h.v, p, c.tk, c.d)?,
                    AttentionOpts {
                        causal: false,
                        window: None,
                        bias: bias.as_ref(),
                        key_lens: lens.as_ref(),
                    },
                )
                .map_err(e("cpu attention_forward"))?;
            let want = want.to_f32_vec().map_err(e("cpu output"))?;
            let at = p * plane_q..(p + 1) * plane_q;
            for (name, got) in arms {
                errs.cmp(
                    &format!("{name} vs CPU Exact, plane {p}"),
                    &got[at.clone()],
                    &want,
                    atol,
                );
            }
        }
        Ok(())
    }

    /// One row's measured result, for the per-frame totals.
    struct Measured {
        id: &'static str,
        layers: Option<u32>,
        scalar: f64,
        /// Median of the rule arm (the last tessl arm) and whether the rule
        /// recommends the route.
        tessl: Option<(f64, bool)>,
    }

    struct Sink {
        csv: Option<std::fs::File>,
        jsonl: Option<std::fs::File>,
    }

    impl Sink {
        fn write(file: &mut Option<std::fs::File>, text: &str) -> R<()> {
            use std::io::Write;
            if let Some(f) = file {
                f.write_all(text.as_bytes()).map_err(e("write results"))?;
                f.flush().map_err(e("flush results"))?;
            }
            Ok(())
        }
    }

    fn json_stats(s: &Stats) -> String {
        format!(
            "{{\"min\":{:.4},\"median\":{:.4},\"p10\":{:.4},\"p90\":{:.4},\"max\":{:.4}}}",
            s.min, s.median, s.p10, s.p90, s.max
        )
    }

    fn run_row(
        m: &MetalBackend,
        rt: &Arc<GpuRuntime>,
        sink: &mut Sink,
        reps: usize,
        row: &Row,
    ) -> R<Measured> {
        let Row {
            id,
            title,
            case,
            ref tessl,
            layers,
            rule,
        } = *row;
        let (id, case) = (id, case);
        println!(
            "\n## {id}: {title} [B={}, H={}, Tq={}, Tk={}, D={}{}{}]",
            case.b,
            case.h,
            case.tq,
            case.tk,
            case.d,
            if case.bias { ", bias" } else { "" },
            case.lens
                .map_or(String::new(), |l| format!(", key_lens [{l}]"))
        );
        let start = format!("GPU {}, load {}", gpu_util(), load());
        let h = Host::new(&case);
        let direct = Direct::new(m, &case, &h)?;
        let routed: Vec<Routed> = tessl
            .iter()
            .map(|&p| Routed::new(rt, case, p, &h))
            .collect::<R<_>>()?;
        let (errs, dispatch_note) = check(m, &case, &h, &direct, &routed)?;
        if !errs.pass() {
            return Err(format!(
                "{id}: an arm disagrees beyond the bound; no timings"
            ));
        }
        let mut arms: Vec<(String, &dyn Arm)> = vec![("ojas scalar".to_string(), &direct)];
        for r in &routed {
            arms.push((label(r.pad).to_string(), r));
        }
        for _ in 0..WARMUPS {
            for (_, a) in &arms {
                timed(*a)?;
            }
        }
        // samples[arm][pass][round]
        let mut samples = vec![[vec![], vec![]]; arms.len()];
        for round in 0..reps {
            // `pass` is a sample label as well as an index into the pair.
            #[allow(clippy::needless_range_loop)]
            for pass in 0..2 {
                for (i, (_, a)) in arms.iter().enumerate() {
                    let ms = timed(*a)?;
                    samples[i][pass].push(ms);
                    Sink::write(
                        &mut sink.csv,
                        &format!("{id},\"{}\",{pass},{round},{ms:.4}\n", arms[i].0),
                    )?;
                }
            }
        }
        let end = format!("GPU {}, load {}", gpu_util(), load());
        println!("\n| arm | n | min ms | median | p10 | p90 | max | p90 - p10 |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|");
        let mut pooled = vec![];
        for (i, (name, _)) in arms.iter().enumerate() {
            print_row(name, reps, &stats(&samples[i][0]));
            print_row(&format!("{name} (repeat)"), reps, &stats(&samples[i][1]));
            let p = [samples[i][0].as_slice(), samples[i][1].as_slice()].concat();
            let st = stats(&p);
            print_row(&format!("{name} pooled"), p.len(), &st);
            pooled.push(st);
        }
        let base = pooled[0];
        let (m0, m1) = (stats(&samples[0][0]).median, stats(&samples[0][1]).median);
        let spread = base.p90 - base.p10;
        println!(
            "\nnoise floor (ojas scalar vs its repeat): |median gap| = {:.3} ms, pooled p90 - p10 = {spread:.3} ms",
            (m0 - m1).abs()
        );
        println!("contention: start {start}; end {end}");
        let mut json = String::new();
        let _ = write!(
            json,
            "{{\"row\":\"{id}\",\"title\":\"{title}\",\"shape\":{{\"b\":{},\"h\":{},\"tq\":{},\"tk\":{},\"d\":{},\"bias\":{},\"key_lens\":{}}},\"reps\":{reps},\"start\":\"{start}\",\"end\":\"{end}\",\"dispatches\":\"{dispatch_note}\",\"max_abs_err\":{:.3e},\"max_rel_err\":{:.3e},\"noise_median_gap_ms\":{:.4},\"noise_spread_ms\":{spread:.4},\"arms\":{{",
            case.b,
            case.h,
            case.tq,
            case.tk,
            case.d,
            case.bias,
            case.lens.map_or("null".to_string(), |l| l.to_string()),
            errs.rows.iter().map(|r| r.1).fold(0.0f32, f32::max),
            errs.rows.iter().map(|r| r.2).fold(0.0f32, f32::max),
            (m0 - m1).abs(),
        );
        for (i, (name, _)) in arms.iter().enumerate() {
            let sep = if i == 0 { "" } else { "," };
            let _ = write!(
                json,
                "{sep}\"{name}\":{{\"pooled\":{},\"pass0\":{},\"pass1\":{}}}",
                json_stats(&pooled[i]),
                json_stats(&stats(&samples[i][0])),
                json_stats(&stats(&samples[i][1]))
            );
        }
        json.push_str("},\"verdicts\":[");
        let mut tessl_out = None;
        for (i, r) in routed.iter().enumerate() {
            let a = i + 1;
            let margin = base.median - pooled[a].median;
            let wins = (0..2)
                .flat_map(|p| (0..reps).map(move |r| (p, r)))
                .filter(|&(p, r)| samples[a][p][r] < samples[0][p][r])
                .count();
            let total = 2 * reps;
            let route = margin > spread && wins == total;
            println!(
                "{}: median {:.3} ms vs scalar {:.3} ms: margin {margin:.3} ms (ratio {:.4}), spread {spread:.3} ms, faster in {wins}/{total} paired rounds -> {}",
                label(r.pad),
                pooled[a].median,
                base.median,
                pooled[a].median / base.median,
                if route { "ROUTE" } else { "DO NOT ROUTE" }
            );
            let sep = if i == 0 { "" } else { "," };
            let _ = write!(
                json,
                "{sep}{{\"arm\":\"{}\",\"margin_ms\":{margin:.4},\"ratio\":{:.5},\"spread_ms\":{spread:.4},\"wins\":{wins},\"paired\":{total},\"margin_over_median_gap\":{},\"route\":{route}}}",
                label(r.pad),
                pooled[a].median / base.median,
                margin > (m0 - m1).abs()
            );
            tessl_out = Some((pooled[a].median, route));
        }
        json.push(']');
        if let Some(n) = layers {
            let per = |ms: f64| ms * n as f64;
            println!(
                "per layer {:.3} ms x {n} layers = {:.3} ms per frame (scalar, contended)",
                base.median,
                per(base.median)
            );
            if let Some((t, _)) = tessl_out {
                println!(
                    "per layer {t:.3} ms x {n} layers = {:.3} ms per frame (tessl arm, contended)",
                    per(t)
                );
            }
            if let Rule::Threshold(limit) = rule {
                let frame = per(base.median);
                println!(
                    "bias rule: per-frame {frame:.3} ms {} {limit:.0} ms (10% of the measured 0.3495 s CoreML frame) and {} {BIAS_THRESHOLD_PROVISIONAL_MS:.0} ms (provisional, 10% of 2.75 s) -> tiled bias kernel {} against the measured threshold",
                    if frame > limit { ">" } else { "<=" },
                    if frame > BIAS_THRESHOLD_PROVISIONAL_MS { ">" } else { "<=" },
                    if frame > limit { "WARRANTED" } else { "NOT WARRANTED" }
                );
                let _ = write!(
                    json,
                    ",\"per_frame_ms\":{frame:.4},\"threshold_ms\":{limit},\"warranted\":{}",
                    frame > limit
                );
            }
        }
        json.push_str("}\n");
        Sink::write(&mut sink.jsonl, &json)?;
        Ok(Measured {
            id,
            layers,
            scalar: base.median,
            tessl: tessl_out,
        })
    }

    /// The extra `[B, 8, 200, 32]` detector shape at the largest batch the
    /// Budget lets through.
    fn run_bias_batch(
        m: &MetalBackend,
        rt: &Arc<GpuRuntime>,
        sink: &mut Sink,
        reps: usize,
    ) -> R<()> {
        for b in [16usize, 8, 4, 2] {
            let case = Case {
                b,
                h: 8,
                tq: 200,
                tk: 5184,
                d: 32,
                bias: true,
                lens: None,
            };
            let h = Host::new(&case);
            let probe = Direct::new(m, &case, &h).and_then(|d| d.run());
            match probe {
                Ok(()) => {
                    println!(
                        "\nbias batch {b}: bias {} bytes accepted by the Budget",
                        4 * case.bias_elems()
                    );
                    drop(h);
                    let row = Row {
                        id: "bx",
                        title: "detector cross-attention with bias, default shape, largest batch that fits",
                        case,
                        tessl: vec![],
                        layers: Some(6),
                        rule: Rule::Info,
                    };
                    let r = run_row(m, rt, sink, reps, &row)?;
                    println!(
                        "per frame at batch {b}: per-layer {:.3} ms x 6 / {b} = {:.3} ms (informational)",
                        r.scalar,
                        r.scalar * 6.0 / b as f64
                    );
                    return Ok(());
                }
                Err(err) => println!(
                    "\nbias batch {b} ({} bytes): refused: {err}",
                    4 * case.bias_elems()
                ),
            }
        }
        Err("no bias batch from 16 down to 2 fit".into())
    }

    /// The crossover rows' ids and key lengths; q is `XOVER_TQ` rows.
    const XOVER_ROWS: [(&str, usize); 4] = [
        ("x5200", 5200),
        ("x10400", 10400),
        ("x10624", 10624),
        ("x15616", 15616),
    ];
    const XOVER_TQ: usize = 5184;
    /// The two arms must agree within this max abs error over the whole
    /// output before anything is timed.
    const XOVER_GATE_ABS: f32 = 1e-5;
    /// A winner must be faster in at least this share of the paired
    /// `(round, pass)` samples (9 in 10).
    const XOVER_SIGN_TENTHS: usize = 9;
    /// Fable's chosen range for the routing bound, as `m / 5184`, and the
    /// span within which it stands (rules.md, R2 and R3).
    const FABLE_RANGE: (f64, f64) = (2.25, 2.5);
    const R2_RANGE: (f64, f64) = (2.05, 2.5);
    const R3_RANGE: (f64, f64) = (1.5, 3.0);
    const TILED: &str = "tiled (attention_forward_saving)";
    const PADQ: &str = "pad-q (attention_forward)";

    fn xover_case(tq: usize, tk: usize) -> Case {
        Case {
            b: 1,
            h: 8,
            tq,
            tk,
            d: 32,
            bias: false,
            lens: None,
        }
    }

    /// Dispatches of the first call on fresh uploads and of the second call
    /// on the same tensors.
    type Counts = (u64, u64);

    fn counts(m: &MetalBackend, arm: &Direct) -> R<Counts> {
        let mut seen = [0u64; 2];
        for slot in &mut seen {
            m.sync().map_err(e("sync"))?;
            let before = m.dispatches();
            arm.out().map(drop)?;
            *slot = m.dispatches() - before;
        }
        Ok((seen[0], seen[1]))
    }

    /// What each route issues on a small shape that is known to take it:
    /// the causal scalar call on `[1, 8, 40, 32]` against 40 keys, and the
    /// tiled and pad-q calls on the same q against 80 keys.
    struct Yardstick {
        scalar: Counts,
        tiled: Counts,
        padq: Counts,
    }

    fn yardstick(m: &MetalBackend) -> R<Yardstick> {
        let arm = |tk: usize, entry: Entry| -> R<Counts> {
            let case = xover_case(40, tk);
            let h = Host::new(&case);
            counts(m, &Direct::new(m, &case, &h)?.with_entry(entry))
        };
        let y = Yardstick {
            scalar: arm(40, Entry::Scalar)?,
            tiled: arm(80, Entry::Saving)?,
            padq: arm(80, Entry::Forward)?,
        };
        println!(
            "yardstick dispatches (first call, repeat): scalar {:?}, tiled {:?}, pad-q {:?}",
            y.scalar, y.tiled, y.padq
        );
        if y.tiled != (y.scalar.0 + 1, y.scalar.1 + 1) {
            return Err(format!(
                "the tiled yardstick {:?} is not the scalar kernel's {:?} plus one dispatch",
                y.tiled, y.scalar
            ));
        }
        if y.padq.0 <= y.tiled.0 || y.padq.1 <= y.tiled.1 {
            return Err(format!(
                "the pad-q yardstick {:?} does not issue more dispatches than the tiled {:?}: \
                 the call did not take the pad route",
                y.padq, y.tiled
            ));
        }
        Ok(y)
    }

    /// One m's correctness and dispatch gate, passed.
    struct Gate {
        tiled: Counts,
        padq: Counts,
        arms_max_abs: f32,
        cpu_max_abs: f32,
    }

    fn gate(m: &MetalBackend, y: &Yardstick, tk: usize) -> R<Gate> {
        let case = xover_case(XOVER_TQ, tk);
        println!("\n## gate x{tk}: q [1,8,{XOVER_TQ},32], k and v [1,8,{tk},32]");
        let h = Host::new(&case);
        let tiled = Direct::new(m, &case, &h)?.with_entry(Entry::Saving);
        let padq = Direct::new(m, &case, &h)?.with_entry(Entry::Forward);
        let (tc, pc) = (counts(m, &tiled)?, counts(m, &padq)?);
        println!(
            "dispatches (first call, repeat): tiled {tc:?} (yardstick {:?}), pad-q {pc:?} (yardstick {:?})",
            y.tiled, y.padq
        );
        if tc != y.tiled || pc != y.padq {
            return Err(format!(
                "m = {tk}: dispatch counts tiled {tc:?} / pad-q {pc:?} are not the routes' {:?} / {:?}",
                y.tiled, y.padq
            ));
        }
        let a = download(&tiled.out()?)?;
        let b = download(&padq.out()?)?;
        let d = diff(&a, &b, 0.0);
        println!(
            "tiled vs pad-q, whole output: max abs diff {:.3e} (bound {XOVER_GATE_ABS:.0e}), max abs {:.4}, non-finite {}",
            d.max_abs, d.max_ref, d.nonfinite
        );
        if d.nonfinite != 0 || d.max_abs > XOVER_GATE_ABS {
            return Err(format!(
                "m = {tk}: the arms disagree by {:.3e} (non-finite {}), over {XOVER_GATE_ABS:.0e}; no timings",
                d.max_abs, d.nonfinite
            ));
        }
        println!("\n| comparison | max abs diff | max abs ref | max abs diff / max abs ref | worst suite ratio | verdict |");
        println!("|---|---:|---:|---:|---:|---|");
        let mut errs = Errors::default();
        compare_cpu(&case, &h, &[("tiled", &a), ("pad-q", &b)], &mut errs)?;
        if !errs.pass() {
            return Err(format!(
                "m = {tk}: an arm disagrees with the CPU beyond the bound; no timings"
            ));
        }
        Ok(Gate {
            tiled: tc,
            padq: pc,
            arms_max_abs: d.max_abs,
            cpu_max_abs: errs.rows.iter().map(|r| r.1).fold(0.0f32, f32::max),
        })
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Winner {
        Tiled,
        PadQ,
        Tie,
    }

    impl Winner {
        fn name(self) -> &'static str {
            match self {
                Winner::Tiled => "tiled",
                Winner::PadQ => "pad-q",
                Winner::Tie => "tie",
            }
        }
    }

    /// `margin` is pad-q's median minus tiled's, so a positive margin favours
    /// tiled. A winner needs a margin over `bar` and to be faster in at least
    /// 9 in 10 of the `total` paired samples.
    fn judge(
        margin: f64,
        bar: f64,
        tiled_faster: usize,
        padq_faster: usize,
        total: usize,
    ) -> Winner {
        let steady = |n: usize| n * 10 >= total * XOVER_SIGN_TENTHS;
        if margin > bar && steady(tiled_faster) {
            Winner::Tiled
        } else if -margin > bar && steady(padq_faster) {
            Winner::PadQ
        } else {
            Winner::Tie
        }
    }

    /// One m's timing, for the R1 to R4 outcome.
    struct Crossover {
        tk: usize,
        tiled: Stats,
        padq: Stats,
        /// The larger of the two arms' pooled p90 - p10.
        noise: f64,
        /// The larger of the two arms' A-vs-A' median gaps.
        gap: f64,
        primary: Winner,
        secondary: Winner,
        gate: Gate,
    }

    impl Crossover {
        fn ratio(&self) -> f64 {
            self.tk as f64 / XOVER_TQ as f64
        }
        /// Pad-q's median minus tiled's: positive when tiled is faster.
        fn margin(&self) -> f64 {
            self.padq.median - self.tiled.median
        }
    }

    fn run_crossover_row(
        m: &MetalBackend,
        sink: &mut Sink,
        reps: usize,
        tk: usize,
        gate: Gate,
    ) -> R<Crossover> {
        let id = format!("x{tk}");
        let case = xover_case(XOVER_TQ, tk);
        println!(
            "\n## {id}: memory cross-attention m = {tk}, ratio {:.4}",
            tk as f64 / XOVER_TQ as f64
        );
        let start = format!("GPU {}, load {}", gpu_util(), load());
        let h = Host::new(&case);
        let tiled = Direct::new(m, &case, &h)?.with_entry(Entry::Saving);
        let padq = Direct::new(m, &case, &h)?.with_entry(Entry::Forward);
        drop(h);
        let arms: [(&str, &dyn Arm); 2] = [(TILED, &tiled), (PADQ, &padq)];
        for _ in 0..WARMUPS {
            for (_, a) in &arms {
                timed(*a)?;
            }
        }
        // samples[arm][pass][round]
        let mut samples = [[vec![], vec![]], [vec![], vec![]]];
        for round in 0..reps {
            // Odd rounds run the arms in reverse inside each pass.
            let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            // `pass` is a sample label as well as an index into the pair.
            #[allow(clippy::needless_range_loop)]
            for pass in 0..2 {
                for i in order {
                    let ms = timed(arms[i].1)?;
                    samples[i][pass].push(ms);
                    Sink::write(
                        &mut sink.csv,
                        &format!("{id},\"{}\",{pass},{round},{ms:.4}\n", arms[i].0),
                    )?;
                }
            }
        }
        let end = format!("GPU {}, load {}", gpu_util(), load());
        println!("\n| arm | n | min ms | median | p10 | p90 | max | p90 - p10 |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|");
        let mut pooled = vec![];
        let mut gaps = vec![];
        for (i, (name, _)) in arms.iter().enumerate() {
            let (s0, s1) = (stats(&samples[i][0]), stats(&samples[i][1]));
            print_row(name, reps, &s0);
            print_row(&format!("{name} (repeat)"), reps, &s1);
            let p = [samples[i][0].as_slice(), samples[i][1].as_slice()].concat();
            let st = stats(&p);
            print_row(&format!("{name} pooled"), p.len(), &st);
            pooled.push(st);
            gaps.push((s0.median - s1.median).abs());
        }
        let (ts, ps) = (pooled[0], pooled[1]);
        let noise = (ts.p90 - ts.p10).max(ps.p90 - ps.p10);
        let gap = gaps[0].max(gaps[1]);
        let total = 2 * reps;
        let paired = |a: usize, b: usize| {
            (0..2)
                .flat_map(|p| (0..reps).map(move |r| (p, r)))
                .filter(|&(p, r)| samples[a][p][r] < samples[b][p][r])
                .count()
        };
        let (tiled_faster, padq_faster) = (paired(0, 1), paired(1, 0));
        let margin = ps.median - ts.median;
        let primary = judge(margin, noise, tiled_faster, padq_faster, total);
        let secondary = judge(margin, gap, tiled_faster, padq_faster, total);
        println!(
            "\nmedian margin (pad-q - tiled) {margin:.3} ms; noise {noise:.3} ms (larger pooled p90 - p10); A-vs-A' median gap {gap:.3} ms (larger of the two arms); tiled faster in {tiled_faster}/{total} paired samples, pad-q in {padq_faster}/{total}"
        );
        println!(
            "winner beyond noise: {} (primary, bar = noise); {} (secondary, bar = A-vs-A' gap)",
            primary.name(),
            secondary.name()
        );
        println!("contention: start {start}; end {end}");
        let json = format!(
            "{{\"row\":\"{id}\",\"m\":{tk},\"ratio\":{:.5},\"reps\":{reps},\"start\":\"{start}\",\"end\":\"{end}\",\"dispatches\":{{\"tiled\":[{},{}],\"padq\":[{},{}]}},\"gate_arms_max_abs\":{:.3e},\"gate_cpu_max_abs\":{:.3e},\"noise_ms\":{noise:.4},\"aa_gap_ms\":{gap:.4},\"margin_ms\":{margin:.4},\"tiled_faster\":{tiled_faster},\"padq_faster\":{padq_faster},\"paired\":{total},\"winner_primary\":\"{}\",\"winner_secondary\":\"{}\",\"arms\":{{\"tiled\":{{\"pooled\":{},\"pass0\":{},\"pass1\":{}}},\"padq\":{{\"pooled\":{},\"pass0\":{},\"pass1\":{}}}}}}}\n",
            tk as f64 / XOVER_TQ as f64,
            gate.tiled.0,
            gate.tiled.1,
            gate.padq.0,
            gate.padq.1,
            gate.arms_max_abs,
            gate.cpu_max_abs,
            primary.name(),
            secondary.name(),
            json_stats(&ts),
            json_stats(&stats(&samples[0][0])),
            json_stats(&stats(&samples[0][1])),
            json_stats(&ps),
            json_stats(&stats(&samples[1][0])),
            json_stats(&stats(&samples[1][1])),
        );
        Sink::write(&mut sink.jsonl, &json)?;
        Ok(Crossover {
            tk,
            tiled: ts,
            padq: ps,
            noise,
            gap,
            primary,
            secondary,
            gate,
        })
    }

    /// Where the routing bound lies, by rules R1 to R4 (rules.md).
    #[derive(Clone, Copy, Debug)]
    enum Estimate {
        /// Tiled's raw median is faster at the smallest m.
        Below,
        /// The first zero of the piecewise-linear margin over the raw
        /// medians, as `m / 5184`.
        At(f64),
        /// Pad-q's raw median is faster at every m.
        Above,
    }

    fn estimate(rows: &[Crossover]) -> Estimate {
        if rows[0].margin() > 0.0 {
            return Estimate::Below;
        }
        for w in rows.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if a.margin() <= 0.0 && b.margin() > 0.0 {
                let t = -a.margin() / (b.margin() - a.margin());
                return Estimate::At(a.ratio() + t * (b.ratio() - a.ratio()));
            }
        }
        Estimate::Above
    }

    /// The R1 to R4 outcome over every m, in the order of rules.md. The rules
    /// were committed before the first timing; this only applies them.
    fn outcome(rows: &[Crossover], sink: &mut Sink) -> R<()> {
        println!("\n## verdict table");
        println!("| m | ratio | tiled median ms | pad-q median ms | noise ms | A-vs-A' gap ms | winner (primary) | winner (secondary) | dispatches tiled / pad-q (first, repeat) |");
        println!("|---:|---:|---:|---:|---:|---:|---|---|---|");
        for r in rows {
            println!(
                "| {} | {:.4} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {:?} / {:?} |",
                r.tk,
                r.ratio(),
                r.tiled.median,
                r.padq.median,
                r.noise,
                r.gap,
                r.primary.name(),
                r.secondary.name(),
                r.gate.tiled,
                r.gate.padq
            );
        }
        let hi = rows.iter().position(|r| r.primary == Winner::Tiled);
        let lo = rows
            .iter()
            .enumerate()
            .filter(|(i, r)| r.primary == Winner::PadQ && hi.is_none_or(|h| *i < h))
            .map(|(i, _)| i)
            .next_back();
        let non_monotone = hi.is_some_and(|h| rows[h..].iter().any(|r| r.primary == Winner::PadQ));
        let name = |i: Option<usize>| {
            i.map_or("none".to_string(), |i| {
                format!("m = {} (ratio {:.4})", rows[i].tk, rows[i].ratio())
            })
        };
        println!(
            "\nR1: pad-q wins beyond noise up to {}; tiled wins beyond noise from {}; the ties between them: {}",
            name(lo),
            name(hi),
            rows.iter()
                .enumerate()
                .filter(|(i, r)| r.primary == Winner::Tie
                    && lo.is_none_or(|l| *i > l)
                    && hi.is_none_or(|h| *i < h))
                .map(|(_, r)| r.tk.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        if non_monotone {
            println!("R1: pad-q wins beyond noise at an m above the first tiled win: non-monotone, the bracket is UNRESOLVED");
        }
        let bound = lo.map(|i| rows[i].ratio());
        match bound {
            Some(b) => println!(
                "R1 routing bound (pad-q only where it beat tiled beyond noise; ties favour tiled): ratio {b:.4}, below tiled's first win {}",
                hi.map_or("(none on the grid)".to_string(), |h| format!("{:.4}", rows[h].ratio()))
            ),
            None => println!("R1 routing bound: pad-q beat tiled beyond noise at no m; route nothing to pad-q"),
        }
        let est = estimate(rows);
        println!("crossover point estimate (zero of the piecewise-linear raw-median margin, inferred): {est:?}");
        let flag_r3 = match est {
            Estimate::Below => true,
            Estimate::Above => true,
            Estimate::At(x) => !(R3_RANGE.0..=R3_RANGE.1).contains(&x),
        };
        let established = match est {
            Estimate::Below => rows[0].primary == Winner::Tiled,
            Estimate::Above => rows[rows.len() - 1].primary == Winner::PadQ,
            Estimate::At(_) => true,
        };
        match est {
            Estimate::At(x) if (R2_RANGE.0..=R2_RANGE.1).contains(&x) => println!(
                "R2: crossover ratio {x:.4} is inside [{}, {}]: Fable's chosen range {}-{} STANDS; gap to the range ends: {:+.4} from {}, {:+.4} from {}; the bracket is {}",
                R2_RANGE.0, R2_RANGE.1, FABLE_RANGE.0, FABLE_RANGE.1,
                x - FABLE_RANGE.0, FABLE_RANGE.0, x - FABLE_RANGE.1, FABLE_RANGE.1,
                format_args!("({}, {}]", name(lo), name(hi))
            ),
            Estimate::At(x) if !flag_r3 => println!(
                "R2: crossover ratio {x:.4} is outside [{}, {}] but inside [{}, {}]: neither R2 nor R3; the range moves to the measured bracket ({}, {}]",
                R2_RANGE.0, R2_RANGE.1, R3_RANGE.0, R3_RANGE.1, name(lo), name(hi)
            ),
            _ => println!("R2: not applicable (the crossover is outside [{}, {}])", R3_RANGE.0, R3_RANGE.1),
        }
        if flag_r3 {
            println!(
                "R3: RESET THE CONSTANT (crossover {est:?} is outside [{}, {}]); {}",
                R3_RANGE.0,
                R3_RANGE.1,
                if established {
                    "established beyond noise at the end of the grid"
                } else {
                    "not established beyond noise: the end-of-grid row is a tie"
                }
            );
        } else {
            println!("R3: not triggered");
        }
        if rows[0].primary == Winner::Tiled {
            println!(
                "R4: tiled beats pad-q at m = {} beyond noise: DELETE PAD-Q",
                rows[0].tk
            );
        } else {
            println!(
                "R4: not triggered at m = {} (winner {}, raw margin {:+.3} ms)",
                rows[0].tk,
                rows[0].primary.name(),
                rows[0].margin()
            );
        }
        let g = &rows[0].gate;
        let (fewer, other) = if g.tiled.0 <= g.padq.0 {
            ("tiled", "pad-q")
        } else {
            ("pad-q", "tiled")
        };
        println!(
            "tie-break: ties favour {fewer} ({} dispatches against {other}'s {}); scratch is not measured here, see summary.md",
            g.tiled.0.min(g.padq.0),
            g.tiled.0.max(g.padq.0)
        );
        let json = format!(
            "{{\"row\":\"verdict\",\"r1_lo_m\":{},\"r1_hi_m\":{},\"routing_bound_ratio\":{},\"non_monotone\":{non_monotone},\"estimate\":\"{est:?}\",\"r3_reset\":{flag_r3},\"r3_established\":{established},\"r4_delete_padq\":{}}}\n",
            lo.map_or("null".to_string(), |i| rows[i].tk.to_string()),
            hi.map_or("null".to_string(), |i| rows[i].tk.to_string()),
            bound.map_or("null".to_string(), |b| format!("{b:.5}")),
            rows[0].primary == Winner::Tiled,
        );
        Sink::write(&mut sink.jsonl, &json)
    }

    /// The crossover rows: every gate first, then the timings, then the
    /// outcome. A subset of the four prints the table without an outcome.
    fn run_crossover(m: &MetalBackend, sink: &mut Sink, reps: usize, tks: &[usize]) -> R<()> {
        if !m.has_tensorops() {
            return Err("no TensorOps: attention_forward would not take the pad route".into());
        }
        let y = yardstick(m)?;
        let mut gates = vec![];
        for &tk in tks {
            gates.push(gate(m, &y, tk)?);
        }
        let mut rows = vec![];
        for (&tk, g) in tks.iter().zip(gates) {
            rows.push(run_crossover_row(m, sink, reps, tk, g)?);
        }
        if rows.len() == XOVER_ROWS.len() {
            outcome(&rows, sink)
        } else {
            println!(
                "\nR1 to R4 need all four m; this run had {}: no outcome",
                rows.len()
            );
            Ok(())
        }
    }

    pub fn main() -> R<()> {
        let args: Vec<String> = std::env::args().collect();
        let reps: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(25);
        if reps < 15 {
            return Err(format!("reps {reps} is under the 15 the method requires"));
        }
        let mut sink = Sink {
            csv: None,
            jsonl: None,
        };
        if let Some(dir) = args.get(2) {
            std::fs::create_dir_all(dir).map_err(e("create out_dir"))?;
            let open = |name: &str| {
                std::fs::File::create(std::path::Path::new(dir).join(name)).map_err(e("create"))
            };
            sink.csv = Some(open("rounds.csv")?);
            sink.jsonl = Some(open("rows.jsonl")?);
            Sink::write(&mut sink.csv, "row,arm,pass,round,ms\n")?;
        }
        let wanted: Vec<&str> = match args.get(3) {
            Some(list) => list
                .split(',')
                .flat_map(|id| {
                    if id == "xover" {
                        XOVER_ROWS.iter().map(|(id, _)| *id).collect()
                    } else {
                        vec![id]
                    }
                })
                .collect(),
            None => ALL_ROWS.to_vec(),
        };
        for w in &wanted {
            if !ALL_ROWS.contains(w) && !XOVER_ROWS.iter().any(|(id, _)| id == w) {
                return Err(format!("unknown row {w}"));
            }
        }
        let m = MetalBackend::new(Budget::new(8 * GIB)).map_err(e("MetalBackend::new"))?;
        println!("device: {}", m.device_name());
        let xover: Vec<usize> = XOVER_ROWS
            .iter()
            .filter(|(id, _)| wanted.contains(id))
            .map(|&(_, tk)| tk)
            .collect();
        let old_rows = wanted.iter().any(|w| ALL_ROWS.contains(w));
        // Every run checks that tessl would run its tiled kernel: the pad-q
        // arm of the crossover rows is that kernel, and the other rows time it.
        let rt = GpuRuntime::new()?;
        rt.set_pool_cache_cap_bytes(GIB as usize);
        rt.set_async_encode(true)?;
        rt.add_metallib_bytes(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/ojas_per_head_gate.metallib"
        )))?;
        println!("tessl runtime: tensorops {}", rt.has_tensorops());
        println!(
            "tessl kernel picked at D=32: {:?}",
            encoder_attn_kernel(&rt, 32)
        );
        if encoder_attn_kernel(&rt, 32) != EncoderAttnKernel::Tiled {
            return Err("tessl would not run the tiled kernel here; the verdict would be about another kernel".into());
        }
        println!("start: GPU {}, load {}", gpu_util(), load());
        println!(
            "{reps} reps of two passes over the arms after {WARMUPS} warm-ups of each; rows {wanted:?}"
        );
        let mut done = vec![];
        for row in rows() {
            if wanted.contains(&row.id) {
                done.push(run_row(&m, &rt, &mut sink, reps, &row)?);
            }
        }
        if wanted.contains(&"bx") {
            run_bias_batch(&m, &rt, &mut sink, reps)?;
        }
        if !xover.is_empty() {
            run_crossover(&m, &mut sink, reps, &xover)?;
        }
        println!("\nend: GPU {}, load {}", gpu_util(), load());
        if old_rows {
            totals(&done);
        }
        Ok(())
    }

    /// Per-frame totals over the rows that carry a layer count, for the
    /// frame-1, mid and max memory lengths. `d` has no layer count and is
    /// left out. `f20`, `g20` and `bx` are variants of rows already counted.
    fn totals(done: &[Measured]) {
        let by = |id: &str| done.iter().find(|r| r.id == id);
        let counted = ["a", "b", "e", "f33", "g"];
        let fixed: Vec<&Measured> = counted.iter().filter_map(|id| by(id)).collect();
        if fixed.len() != counted.len() {
            println!("\nper-frame totals: not all of a, b, e, f33, g ran; none printed");
            return;
        }
        println!("\n## per-frame totals (median ms x layers, contended)");
        println!("| memory length | scalar everywhere | best route per rule | rows counted |");
        println!("|---|---:|---:|---|");
        for mem in ["c1", "c2", "c3"] {
            let Some(c) = by(mem) else { continue };
            let (mut scalar, mut routed) = (0.0, 0.0);
            for r in fixed.iter().copied().chain([c]) {
                let n = f64::from(r.layers.unwrap_or(0));
                scalar += r.scalar * n;
                routed += match r.tessl {
                    Some((t, true)) => t,
                    _ => r.scalar,
                } * n;
            }
            println!(
                "| {mem} | {scalar:.1} | {routed:.1} | a x10, b x6, {mem} x4, e x6, f33 x6, g x6 |"
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(err) = bench::main() {
        eprintln!("attention_spiketrack_bench: {err}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("attention_spiketrack_bench needs macOS and the `metal` feature");
}
