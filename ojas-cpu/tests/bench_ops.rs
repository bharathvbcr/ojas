//! Per-op wall time of the nanolab training step on `CpuBackend` (Fast),
//! paired with `ojas-cpu/benches/torch_ops.py` (torch oracle and timer) and
//! summarized by `ojas-cpu/benches/summarize_ops.py`. Ignored by default.
//!
//! Shapes are the nanolab block: d_model 768, 12 heads of 64, T 1024, B 1,
//! SwiGLU hidden 2048 (`docs/framework-design.md` §2), vocab 50304.
//!
//! `OJAS_BENCH_MODE` selects what runs:
//! - `dump`: write every case's inputs and its 6-thread outputs as raw
//!   little-endian files under `OJAS_BENCH_DIR/<case>/`, listed in
//!   `manifest.txt`. The torch script reads them for parity. Nothing is timed.
//! - `time` (default): min-of-N wall time for each case, direction and thread
//!   count, after two warmup calls. One `OJAS_OP` line per measurement.
//! - `loop`: repeat one case and direction (`OJAS_BENCH_OPS`, `OJAS_BENCH_DIRSEL`)
//!   while `/usr/bin/sample` profiles this process into `OJAS_BENCH_SAMPLE_OUT`.
//!
//! `OJAS_BENCH_OPS=a,b` keeps only those cases; `OJAS_BENCH_THREADS=6,18` sets
//! the thread counts (the first is used by `dump` and `loop`).
//!
//! ```text
//! OJAS_BENCH_MODE=dump cargo test -p ojas-cpu --release --test bench_ops -- --ignored --nocapture --test-threads=1
//! cargo test -p ojas-cpu --release --test bench_ops -- --ignored --nocapture --test-threads=1
//! ```

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ojas_core::{
    AdamWConfig, Backend, Budget, DType, MuonNs5Config, Numerics, OjasError, Tensor, RMS_NORM_EPS,
};
use ojas_cpu::CpuBackend;

mod common;
use common::SplitMix64;

const T: usize = 1024;
const D: usize = 768;
const NH: usize = 12;
const HD: usize = 64;
const FF: usize = 2048;
const V: usize = 50304;
const LAYERS: usize = 12;
/// `[B, T, H, D]` to `[B, H, T, D]`; it is its own inverse.
const SWAP_TH: [usize; 4] = [0, 2, 1, 3];
/// Accounting cap only. Peak resident memory is measured separately.
const BUDGET_BYTES: u64 = 12 << 30;

type Outs = Vec<(String, Tensor)>;
type Run = Box<dyn Fn(&CpuBackend, &mut [Tensor]) -> Result<Outs, OjasError>>;

struct Dir {
    name: &'static str,
    n: usize,
    run: Run,
    /// The op writes its inputs in place (optimizer steps, clip). Every call
    /// then runs on a fresh copy of the case's inputs, made outside the timer
    /// as `torch_ops.py`'s `prep` does, so each call does the first call's
    /// work: a reused clip input is already clipped, and its scale pass would
    /// stop running after one call.
    mutates: bool,
}

struct Case {
    name: &'static str,
    shape: String,
    names: Vec<String>,
    tensors: Vec<Tensor>,
    dirs: Vec<Dir>,
}

fn dir<F>(name: &'static str, n: usize, f: F) -> Dir
where
    F: Fn(&CpuBackend, &mut [Tensor]) -> Result<Outs, OjasError> + 'static,
{
    Dir {
        name,
        n,
        run: Box::new(f),
        mutates: false,
    }
}

/// [`dir`] for an op that writes its inputs in place; see [`Dir::mutates`].
fn dir_mut<F>(name: &'static str, n: usize, f: F) -> Dir
where
    F: Fn(&CpuBackend, &mut [Tensor]) -> Result<Outs, OjasError> + 'static,
{
    Dir {
        mutates: true,
        ..dir(name, n, f)
    }
}

/// One call of `d`, timed. A mutating direction gets a fresh copy of
/// `tensors`, made and charged before the timer starts.
fn call(d: &Dir, cpu: &CpuBackend, tensors: &mut [Tensor]) -> (Duration, Result<Outs, OjasError>) {
    let mut copy;
    let ins = if d.mutates {
        copy = tensors
            .iter()
            .map(|t| Tensor::from_f32(&t.to_f32_vec()?, t.shape(), cpu.budget()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|e| panic!("{}: input copy: {e:?}", d.name));
        copy.as_mut_slice()
    } else {
        tensors
    };
    let t0 = Instant::now();
    let outs = (d.run)(cpu, ins);
    (t0.elapsed(), outs)
}

fn out(name: &str, t: Tensor) -> (String, Tensor) {
    (name.to_string(), t)
}

struct Ctx {
    budget: Budget,
    rng: SplitMix64,
    names: Vec<String>,
    tensors: Vec<Tensor>,
}

impl Ctx {
    fn new(budget: &Budget, case: &str) -> Self {
        // FNV-1a of the case name, so each case's data does not depend on
        // which other cases were selected.
        let seed = case.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
        Self {
            budget: budget.clone(),
            rng: SplitMix64(seed),
            names: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Uniform in `[offset - scale, offset + scale)`; returns its input index.
    fn f32(&mut self, name: &str, shape: &[usize], scale: f32, offset: f32) -> usize {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| offset + scale * self.rng.unit()).collect();
        self.push(name, Tensor::from_f32(&data, shape, &self.budget).unwrap())
    }

    fn ids(&mut self, name: &str, shape: &[usize], below: usize) -> usize {
        let n: usize = shape.iter().product();
        let data: Vec<u32> = (0..n).map(|_| self.rng.below(below) as u32).collect();
        self.push(name, Tensor::from_u32(&data, shape, &self.budget).unwrap())
    }

    /// Half-split RoPE tables `[T, HD]`, base 10000, both halves equal.
    fn rope_tables(&mut self) -> (usize, usize) {
        let mut cos = vec![0.0f32; T * HD];
        let mut sin = vec![0.0f32; T * HD];
        for t in 0..T {
            for i in 0..HD / 2 {
                let freq = 1.0 / 10000f64.powf((2 * i) as f64 / HD as f64);
                let angle = t as f64 * freq;
                for slot in [i, i + HD / 2] {
                    cos[t * HD + slot] = angle.cos() as f32;
                    sin[t * HD + slot] = angle.sin() as f32;
                }
            }
        }
        let c = Tensor::from_f32(&cos, &[T, HD], &self.budget).unwrap();
        let s = Tensor::from_f32(&sin, &[T, HD], &self.budget).unwrap();
        (self.push("cos", c), self.push("sin", s))
    }

    fn push(&mut self, name: &str, t: Tensor) -> usize {
        self.names.push(name.to_string());
        self.tensors.push(t);
        self.tensors.len() - 1
    }

    fn case(self, name: &'static str, shape: String, dirs: Vec<Dir>) -> Case {
        Case {
            name,
            shape,
            names: self.names,
            tensors: self.tensors,
            dirs,
        }
    }
}

/// Row-major view of a contiguous tensor with a new shape; no copy.
fn view(t: &Tensor, shape: &[usize]) -> Result<Tensor, OjasError> {
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    t.view(shape, &strides, t.byte_offset())
}

/// `rows` is `T` for a training step and 1 for a decode step.
fn linear_case(
    b: &Budget,
    name: &'static str,
    rows: usize,
    kin: usize,
    nout: usize,
    n: usize,
) -> Case {
    let mut c = Ctx::new(b, name);
    let x = c.f32("x", &[rows, kin], 1.0, 0.0);
    let w = c.f32("w", &[nout, kin], 0.035, 0.0);
    let gy = c.f32("gy", &[rows, nout], 0.01, 0.0);
    c.case(
        name,
        format!("{rows}x{kin}->{nout}"),
        vec![
            dir("fwd", n, move |cpu, t| {
                Ok(vec![out("y", cpu.linear_forward(&t[x], &t[w])?)])
            }),
            dir("bwd", n, move |cpu, t| {
                let (gx, gw) = cpu.linear_backward(&t[x], &t[w], &t[gy])?;
                Ok(vec![out("gx", gx), out("gw", gw)])
            }),
        ],
    )
}

fn embedding_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "embedding");
    let table = c.f32("table", &[V, D], 0.035, 0.0);
    let ids = c.ids("ids", &[1, T], V);
    let gy = c.f32("gy", &[1, T, D], 0.01, 0.0);
    c.case(
        "embedding",
        format!("[{V},{D}][1,{T}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                Ok(vec![out("y", cpu.embedding_forward(&t[table], &t[ids])?)])
            }),
            dir("bwd", 5, move |cpu, t| {
                Ok(vec![out(
                    "gtable",
                    cpu.embedding_backward(&t[table], &t[ids], &t[gy])?,
                )])
            }),
        ],
    )
}

fn rms_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "rms_norm");
    let x = c.f32("x", &[T, D], 1.0, 0.0);
    let w = c.f32("w", &[D], 0.1, 1.0);
    let gy = c.f32("gy", &[T, D], 0.01, 0.0);
    c.case(
        "rms_norm",
        format!("[{T},{D}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                Ok(vec![out(
                    "y",
                    cpu.rms_norm_forward(&t[x], &t[w], RMS_NORM_EPS)?,
                )])
            }),
            dir("bwd", 20, move |cpu, t| {
                let (gx, gw) = cpu.rms_norm_backward(&t[x], &t[w], &t[gy], RMS_NORM_EPS)?;
                Ok(vec![out("gx", gx), out("gw", gw)])
            }),
        ],
    )
}

fn qk_norm_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "qk_norm");
    let q = c.f32("q", &[1, T, NH, HD], 1.0, 0.0);
    let k = c.f32("k", &[1, T, NH, HD], 1.0, 0.0);
    let wq = c.f32("wq", &[HD], 0.1, 1.0);
    let wk = c.f32("wk", &[HD], 0.1, 1.0);
    let gq = c.f32("gq", &[1, T, NH, HD], 0.01, 0.0);
    let gk = c.f32("gk", &[1, T, NH, HD], 0.01, 0.0);
    c.case(
        "qk_norm",
        format!("2x[1,{T},{NH},{HD}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                let (qn, kn) =
                    cpu.rms_qk_norm_forward(&t[q], &t[k], &t[wq], &t[wk], RMS_NORM_EPS)?;
                Ok(vec![out("qn", qn), out("kn", kn)])
            }),
            dir("bwd", 20, move |cpu, t| {
                let (a, bb, cc, dd) = cpu.rms_qk_norm_backward(
                    &t[q],
                    &t[k],
                    &t[wq],
                    &t[wk],
                    &t[gq],
                    &t[gk],
                    RMS_NORM_EPS,
                )?;
                Ok(vec![
                    out("gq", a),
                    out("gk", bb),
                    out("gwq", cc),
                    out("gwk", dd),
                ])
            }),
        ],
    )
}

fn rope_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "rope");
    let x = c.f32("x", &[1, T, NH, HD], 1.0, 0.0);
    let gy = c.f32("gy", &[1, T, NH, HD], 0.01, 0.0);
    let (cos, sin) = c.rope_tables();
    c.case(
        "rope",
        format!("[1,{T},{NH},{HD}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                Ok(vec![out(
                    "y",
                    cpu.rope_half_split_forward(&t[x], &t[cos], &t[sin])?,
                )])
            }),
            dir("bwd", 20, move |cpu, t| {
                Ok(vec![out(
                    "gx",
                    cpu.rope_half_split_backward(&t[gy], &t[cos], &t[sin])?,
                )])
            }),
        ],
    )
}

fn sdpa_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "sdpa");
    let q = c.f32("q", &[1, NH, T, HD], 1.0, 0.0);
    let k = c.f32("k", &[1, NH, T, HD], 1.0, 0.0);
    let v = c.f32("v", &[1, NH, T, HD], 1.0, 0.0);
    let gy = c.f32("gy", &[1, NH, T, HD], 0.01, 0.0);
    c.case(
        "sdpa",
        format!("[1,{NH},{T},{HD}]"),
        vec![
            dir("fwd", 10, move |cpu, t| {
                Ok(vec![out(
                    "y",
                    cpu.causal_sdpa_forward(&t[q], &t[k], &t[v])?,
                )])
            }),
            dir("bwd", 10, move |cpu, t| {
                let (gq, gk, gv) = cpu.causal_sdpa_backward(&t[q], &t[k], &t[v], &t[gy])?;
                Ok(vec![out("gq", gq), out("gk", gk), out("gv", gv)])
            }),
        ],
    )
}

fn gate_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "gate");
    let x = c.f32("x", &[1, T, D], 1.0, 0.0);
    let w = c.f32("w", &[NH, D], 0.035, 0.0);
    let bias = c.f32("b", &[NH], 0.1, 0.0);
    let attn = c.f32("attn", &[1, T, NH, HD], 1.0, 0.0);
    let gy = c.f32("gy", &[1, T, NH, HD], 0.01, 0.0);
    // The scale is what a Fast forward keeps. Building it on the first call
    // (a warmup) keeps it out of the timed backward, the same split as
    // torch's untimed forward plus timed `autograd.grad`.
    let scales = OnceLock::new();
    c.case(
        "gate",
        format!("x[1,{T},{D}] attn[1,{T},{NH},{HD}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                let (y, scale) =
                    cpu.per_head_sigmoid_gate_forward_saving(&t[x], &t[w], &t[bias], &t[attn])?;
                drop(scale);
                Ok(vec![out("y", y)])
            }),
            dir("bwd", 20, move |cpu, t| {
                let scale = scales.get_or_init(|| {
                    cpu.per_head_sigmoid_gate_forward_saving(&t[x], &t[w], &t[bias], &t[attn])
                        .expect("gate forward")
                        .1
                        .expect("Fast gate forward keeps the per-head scale")
                });
                let g = cpu.per_head_sigmoid_gate_backward_saved(
                    &t[x], &t[w], &t[bias], &t[attn], &t[gy], scale,
                )?;
                Ok(vec![
                    out("gx", g.input),
                    out("gw", g.weight),
                    out("gb", g.bias),
                    out("gattn", g.attn_out),
                ])
            }),
        ],
    )
}

fn vres_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "vres");
    let v = c.f32("v", &[1, T, NH, HD], 1.0, 0.0);
    let v0 = c.f32("v0", &[1, T, NH, HD], 1.0, 0.0);
    let lam = c.f32("lam", &[1], 0.3, 0.0);
    let gy = c.f32("gy", &[1, T, NH, HD], 0.01, 0.0);
    c.case(
        "vres",
        format!("[1,{T},{NH},{HD}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                Ok(vec![out(
                    "y",
                    cpu.value_residual_blend_forward(&t[v], &t[v0], &t[lam])?,
                )])
            }),
            dir("bwd", 20, move |cpu, t| {
                let g = cpu.value_residual_blend_backward(&t[v], &t[v0], &t[lam], &t[gy])?;
                Ok(vec![
                    out("gv", g.value),
                    out("gv0", g.value0),
                    out("glam", g.lambda),
                ])
            }),
        ],
    )
}

fn silu_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "silu");
    let x = c.f32("x", &[T, FF], 4.0, 0.0);
    let gy = c.f32("gy", &[T, FF], 0.01, 0.0);
    c.case(
        "silu",
        format!("[{T},{FF}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                Ok(vec![out("y", cpu.silu_forward(&t[x])?)])
            }),
            dir("bwd", 20, move |cpu, t| {
                Ok(vec![out("gx", cpu.silu_backward(&t[x], &t[gy])?)])
            }),
        ],
    )
}

fn mul_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "mul");
    let a = c.f32("a", &[T, FF], 1.0, 0.0);
    let bb = c.f32("b", &[T, FF], 1.0, 0.0);
    let gy = c.f32("gy", &[T, FF], 0.01, 0.0);
    c.case(
        "mul",
        format!("[{T},{FF}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                Ok(vec![out("y", cpu.mul_forward(&t[a], &t[bb])?)])
            }),
            dir("bwd", 20, move |cpu, t| {
                let (ga, gb) = cpu.mul_backward(&t[a], &t[bb], &t[gy])?;
                Ok(vec![out("ga", ga), out("gb", gb)])
            }),
        ],
    )
}

fn add_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "add");
    let x = c.f32("x", &[T, D], 1.0, 0.0);
    let y = c.f32("y", &[T, D], 1.0, 0.0);
    let gy = c.f32("gy", &[T, D], 0.01, 0.0);
    c.case(
        "add",
        format!("[{T},{D}]"),
        vec![
            dir("fwd", 20, move |cpu, t| {
                Ok(vec![out("z", cpu.residual_add_forward(&t[x], &t[y])?)])
            }),
            dir("bwd", 20, move |cpu, t| {
                let (gx, gy2) = cpu.residual_add_backward(&t[x], &t[y], &t[gy])?;
                Ok(vec![out("gx", gx), out("gy", gy2)])
            }),
        ],
    )
}

/// Permute has no backward method: its gradient is the same permute
/// ([`SWAP_TH`] is its own inverse), so the forward row covers both.
fn permute_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "permute");
    let x = c.f32("x", &[1, T, NH, HD], 1.0, 0.0);
    c.case(
        "permute",
        format!("[1,{T},{NH},{HD}]->(0,2,1,3)"),
        vec![dir("fwd", 20, move |cpu, t| {
            Ok(vec![out("y", cpu.permute(&t[x], &SWAP_TH)?)])
        })],
    )
}

fn ce_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "ce");
    let logits = c.f32("logits", &[T, V], 2.0, 0.0);
    let targets = c.ids("targets", &[T], V);
    c.case(
        "ce",
        format!("[{T},{V}]"),
        vec![
            dir("fwd", 5, move |cpu, t| {
                Ok(vec![out(
                    "loss",
                    cpu.cross_entropy_mean_forward(&t[logits], &t[targets], None)?,
                )])
            }),
            dir("bwd", 5, move |cpu, t| {
                Ok(vec![out(
                    "glogits",
                    cpu.cross_entropy_mean_backward(&t[logits], &t[targets], None)?,
                )])
            }),
        ],
    )
}

/// nanolab AdamW group: lr 6e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.
/// Moments start non-zero so the step is not the first-step special case.
fn adamw_case(b: &Budget, name: &'static str, rows: usize, cols: usize, n: usize) -> Case {
    let mut c = Ctx::new(b, name);
    let p = c.f32("p", &[rows, cols], 0.035, 0.0);
    let g = c.f32("g", &[rows, cols], 0.01, 0.0);
    let m1 = c.f32("m1", &[rows, cols], 0.001, 0.0);
    let m2 = c.f32("m2", &[rows, cols], 0.5e-5, 0.5e-5);
    let _ = (p, g, m1, m2);
    c.case(
        name,
        format!("[{rows},{cols}]"),
        vec![dir_mut("step", n, move |cpu, t| {
            let [p, g, m1, m2] = t else {
                unreachable!("adamw case has four inputs")
            };
            cpu.adamw_step(p, g, m1, m2, 0, AdamWConfig::nanolab(6e-4, 0.0))?;
            Ok(vec![
                out("p", p.clone()),
                out("m1", m1.clone()),
                out("m2", m2.clone()),
            ])
        })],
    )
}

/// nanolab Muon default: lr 0.025, momentum 0.99, weight decay 0.1, Nesterov.
fn muon_case(b: &Budget, name: &'static str, rows: usize, cols: usize, n: usize) -> Case {
    let mut c = Ctx::new(b, name);
    let p = c.f32("p", &[rows, cols], 0.035, 0.0);
    let g = c.f32("g", &[rows, cols], 0.01, 0.0);
    let m = c.f32("mom", &[rows, cols], 0.01, 0.0);
    let _ = (p, g, m);
    c.case(
        name,
        format!("[{rows},{cols}]"),
        vec![dir_mut("step", n, move |cpu, t| {
            let [p, g, m] = t else {
                unreachable!("muon case has three inputs")
            };
            cpu.muon_ns5_step(p, g, m, MuonNs5Config::nanolab_default())?;
            Ok(vec![out("p", p.clone()), out("mom", m.clone())])
        })],
    )
}

/// Every nanolab parameter shape, in `named_parameters` order: 123.7M values.
fn nanolab_param_shapes() -> Vec<(String, Vec<usize>)> {
    let mut shapes = vec![("tok_emb".to_string(), vec![V, D])];
    for i in 0..LAYERS {
        let p = |s: &str| format!("b{i}.{s}");
        shapes.extend([
            (p("norm1"), vec![D]),
            (p("q"), vec![D, D]),
            (p("k"), vec![D, D]),
            (p("v"), vec![D, D]),
            (p("o"), vec![D, D]),
            (p("q_norm"), vec![HD]),
            (p("k_norm"), vec![HD]),
            (p("gate_w"), vec![NH, D]),
            (p("gate_b"), vec![NH]),
            (p("vr_lambda"), vec![1]),
            (p("norm2"), vec![D]),
            (p("ffn_gate"), vec![FF, D]),
            (p("ffn_up"), vec![FF, D]),
            (p("ffn_down"), vec![D, FF]),
        ]);
    }
    shapes.push(("norm_f".to_string(), vec![D]));
    shapes
}

/// Global-norm clip at nanolab's `grad_clip` 1.0. The gradients' norm is
/// about 6.4, so the scaling path runs on every call.
fn clip_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "clip");
    let shapes = nanolab_param_shapes();
    let total: usize = shapes
        .iter()
        .map(|(_, s)| s.iter().product::<usize>())
        .sum();
    for (name, shape) in &shapes {
        c.f32(name, shape, 1e-3, 0.0);
    }
    // Index of layer 0's q gradient, returned for parity.
    let q0 = 2;
    let last = shapes.len() - 1;
    c.case(
        "clip",
        format!("{} tensors, {total} values", shapes.len()),
        vec![dir_mut("step", 5, move |cpu, t| {
            let norm = cpu.clip_grad_norm(t, 1.0)?;
            assert!(norm > 1.0, "clip input norm {norm} would not scale");
            let norm_t = Tensor::from_f32(&[norm], &[], cpu.budget())?;
            Ok(vec![
                out("norm", norm_t),
                out("b0_q", t[q0].clone()),
                out("norm_f", t[last].clone()),
            ])
        })],
    )
}

/// Input indices of the block case, in the order `block_case` creates them.
struct BlockIx {
    x: usize,
    wn1: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    wqn: usize,
    wkn: usize,
    cos: usize,
    sin: usize,
    v0: usize,
    lam: usize,
    wg: usize,
    bg: usize,
    wn2: usize,
    wfg: usize,
    wfu: usize,
    wd: usize,
    gy: usize,
}

/// Forward activations the hand-written backward reads.
struct Acts {
    h: Tensor,
    q4: Tensor,
    k4: Tensor,
    v4: Tensor,
    qh: Tensor,
    kh: Tensor,
    vh: Tensor,
    yb: Tensor,
    h3: Tensor,
    g2: Tensor,
    x1: Tensor,
    h2: Tensor,
    a: Tensor,
    u: Tensor,
    s: Tensor,
    m: Tensor,
    out: Tensor,
}

/// nanolab `Attention.forward` and the SwiGLU FFN (`docs/framework-design.md`
/// §2): norm1, q/k/v, QK-norm, RoPE, value residual, permute, causal SDPA,
/// permute back, per-head gate reading `h`, o_proj, residual, norm2, SwiGLU,
/// residual.
fn block_forward(cpu: &CpuBackend, t: &[Tensor], ix: &BlockIx) -> Result<Acts, OjasError> {
    let x = &t[ix.x];
    let h = cpu.rms_norm_forward(x, &t[ix.wn1], RMS_NORM_EPS)?;
    let q = cpu.linear_forward(&h, &t[ix.wq])?;
    let k = cpu.linear_forward(&h, &t[ix.wk])?;
    let v = cpu.linear_forward(&h, &t[ix.wv])?;
    let q4 = view(&q, &[1, T, NH, HD])?;
    let k4 = view(&k, &[1, T, NH, HD])?;
    let v4 = view(&v, &[1, T, NH, HD])?;
    let (qn, kn) = cpu.rms_qk_norm_forward(&q4, &k4, &t[ix.wqn], &t[ix.wkn], RMS_NORM_EPS)?;
    let qr = cpu.rope_half_split_forward(&qn, &t[ix.cos], &t[ix.sin])?;
    let kr = cpu.rope_half_split_forward(&kn, &t[ix.cos], &t[ix.sin])?;
    let vr = cpu.value_residual_blend_forward(&v4, &t[ix.v0], &t[ix.lam])?;
    let qh = cpu.permute(&qr, &SWAP_TH)?;
    let kh = cpu.permute(&kr, &SWAP_TH)?;
    let vh = cpu.permute(&vr, &SWAP_TH)?;
    let y = cpu.causal_sdpa_forward(&qh, &kh, &vh)?;
    let yb = cpu.permute(&y, &SWAP_TH)?;
    let h3 = view(&h, &[1, T, D])?;
    let g = cpu.per_head_sigmoid_gate_forward(&h3, &t[ix.wg], &t[ix.bg], &yb)?;
    let g2 = view(&g, &[T, D])?;
    let o = cpu.linear_forward(&g2, &t[ix.wo])?;
    let x1 = cpu.residual_add_forward(x, &o)?;
    let h2 = cpu.rms_norm_forward(&x1, &t[ix.wn2], RMS_NORM_EPS)?;
    let a = cpu.linear_forward(&h2, &t[ix.wfg])?;
    let u = cpu.linear_forward(&h2, &t[ix.wfu])?;
    let s = cpu.silu_forward(&a)?;
    let m = cpu.mul_forward(&s, &u)?;
    let dn = cpu.linear_forward(&m, &t[ix.wd])?;
    let out = cpu.residual_add_forward(&x1, &dn)?;
    Ok(Acts {
        h,
        q4,
        k4,
        v4,
        qh,
        kh,
        vh,
        yb,
        h3,
        g2,
        x1,
        h2,
        a,
        u,
        s,
        m,
        out,
    })
}

/// The adjoint of [`block_forward`] for an upstream gradient `t[ix.gy]`.
fn block_backward(
    cpu: &CpuBackend,
    t: &[Tensor],
    ix: &BlockIx,
    z: &Acts,
) -> Result<Outs, OjasError> {
    let gy = &t[ix.gy];
    // out = x1 + down(m)
    let (gm, gwd) = cpu.linear_backward(&z.m, &t[ix.wd], gy)?;
    let (gs, gu) = cpu.mul_backward(&z.s, &z.u, &gm)?;
    let ga = cpu.silu_backward(&z.a, &gs)?;
    let (gh2a, gwfg) = cpu.linear_backward(&z.h2, &t[ix.wfg], &ga)?;
    let (gh2u, gwfu) = cpu.linear_backward(&z.h2, &t[ix.wfu], &gu)?;
    let gh2 = cpu.residual_add_forward(&gh2a, &gh2u)?;
    let (gx1n, gwn2) = cpu.rms_norm_backward(&z.x1, &t[ix.wn2], &gh2, RMS_NORM_EPS)?;
    let gx1 = cpu.residual_add_forward(gy, &gx1n)?;
    // x1 = x + o_proj(gate(h, attention))
    let (gg2, gwo) = cpu.linear_backward(&z.g2, &t[ix.wo], &gx1)?;
    let gg = view(&gg2, &[1, T, NH, HD])?;
    let gate = cpu.per_head_sigmoid_gate_backward(&z.h3, &t[ix.wg], &t[ix.bg], &z.yb, &gg)?;
    let gyh = cpu.permute(&gate.attn_out, &SWAP_TH)?;
    let (gqh, gkh, gvh) = cpu.causal_sdpa_backward(&z.qh, &z.kh, &z.vh, &gyh)?;
    let gqr = cpu.permute(&gqh, &SWAP_TH)?;
    let gkr = cpu.permute(&gkh, &SWAP_TH)?;
    let gvr = cpu.permute(&gvh, &SWAP_TH)?;
    let gqn = cpu.rope_half_split_backward(&gqr, &t[ix.cos], &t[ix.sin])?;
    let gkn = cpu.rope_half_split_backward(&gkr, &t[ix.cos], &t[ix.sin])?;
    let (gq4, gk4, gwqn, gwkn) = cpu.rms_qk_norm_backward(
        &z.q4,
        &z.k4,
        &t[ix.wqn],
        &t[ix.wkn],
        &gqn,
        &gkn,
        RMS_NORM_EPS,
    )?;
    let vrg = cpu.value_residual_blend_backward(&z.v4, &t[ix.v0], &t[ix.lam], &gvr)?;
    let (ghq, gwq) = cpu.linear_backward(&z.h, &t[ix.wq], &view(&gq4, &[T, D])?)?;
    let (ghk, gwk) = cpu.linear_backward(&z.h, &t[ix.wk], &view(&gk4, &[T, D])?)?;
    let (ghv, gwv) = cpu.linear_backward(&z.h, &t[ix.wv], &view(&vrg.value, &[T, D])?)?;
    let gh = cpu.residual_add_forward(&ghq, &ghk)?;
    let gh = cpu.residual_add_forward(&gh, &ghv)?;
    let gh = cpu.residual_add_forward(&gh, &view(&gate.input, &[T, D])?)?;
    let (gxn, gwn1) = cpu.rms_norm_backward(&t[ix.x], &t[ix.wn1], &gh, RMS_NORM_EPS)?;
    let gx = cpu.residual_add_forward(&gx1, &gxn)?;
    Ok(vec![
        out("out", z.out.clone()),
        out("gx", gx),
        out("gwq", gwq),
        out("gwk", gwk),
        out("gwv", gwv),
        out("gwo", gwo),
        out("gwd", gwd),
        out("gwfg", gwfg),
        out("gwfu", gwfu),
        out("gwg", gate.weight),
        out("gbg", gate.bias),
        out("gwqn", gwqn),
        out("gwkn", gwkn),
        out("gwn1", gwn1),
        out("gwn2", gwn2),
        out("glam", vrg.lambda),
        out("gv0", vrg.value0),
    ])
}

fn block_case(b: &Budget) -> Case {
    let mut c = Ctx::new(b, "block");
    let x = c.f32("x", &[T, D], 1.0, 0.0);
    let wn1 = c.f32("wn1", &[D], 0.1, 1.0);
    let wq = c.f32("wq", &[D, D], 0.035, 0.0);
    let wk = c.f32("wk", &[D, D], 0.035, 0.0);
    let wv = c.f32("wv", &[D, D], 0.035, 0.0);
    let wo = c.f32("wo", &[D, D], 0.035, 0.0);
    let wqn = c.f32("wqn", &[HD], 0.1, 1.0);
    let wkn = c.f32("wkn", &[HD], 0.1, 1.0);
    let (cos, sin) = c.rope_tables();
    let v0 = c.f32("v0", &[1, T, NH, HD], 1.0, 0.0);
    let lam = c.f32("lam", &[1], 0.3, 0.0);
    let wg = c.f32("wg", &[NH, D], 0.035, 0.0);
    let bg = c.f32("bg", &[NH], 0.1, 0.0);
    let wn2 = c.f32("wn2", &[D], 0.1, 1.0);
    let wfg = c.f32("wfg", &[FF, D], 0.035, 0.0);
    let wfu = c.f32("wfu", &[FF, D], 0.035, 0.0);
    let wd = c.f32("wd", &[D, FF], 0.035, 0.0);
    let gy = c.f32("gy", &[T, D], 0.01, 0.0);
    let ix = std::rc::Rc::new(BlockIx {
        x,
        wn1,
        wq,
        wk,
        wv,
        wo,
        wqn,
        wkn,
        cos,
        sin,
        v0,
        lam,
        wg,
        bg,
        wn2,
        wfg,
        wfu,
        wd,
        gy,
    });
    let ix_f = std::rc::Rc::clone(&ix);
    c.case(
        "block",
        format!("T{T} d{D} {NH}x{HD} ff{FF}"),
        vec![
            dir("fwd", 10, move |cpu, t| {
                let z = block_forward(cpu, t, &ix_f)?;
                Ok(vec![out("out", z.out)])
            }),
            dir("step", 10, move |cpu, t| {
                let z = block_forward(cpu, t, &ix)?;
                block_backward(cpu, t, &ix, &z)
            }),
        ],
    )
}

type Builder = fn(&Budget) -> Case;

fn builders() -> Vec<(&'static str, Builder)> {
    vec![
        ("linear_qkvo", |b| {
            linear_case(b, "linear_qkvo", T, D, D, 20)
        }),
        ("linear_up", |b| linear_case(b, "linear_up", T, D, FF, 20)),
        ("linear_down", |b| {
            linear_case(b, "linear_down", T, FF, D, 20)
        }),
        ("linear_lmhead", |b| {
            linear_case(b, "linear_lmhead", T, D, V, 5)
        }),
        ("linear_dec_qkvo", |b| {
            linear_case(b, "linear_dec_qkvo", 1, D, D, 200)
        }),
        ("linear_dec_up", |b| {
            linear_case(b, "linear_dec_up", 1, D, FF, 200)
        }),
        ("linear_dec_down", |b| {
            linear_case(b, "linear_dec_down", 1, FF, D, 200)
        }),
        ("embedding", embedding_case),
        ("rms_norm", rms_case),
        ("qk_norm", qk_norm_case),
        ("rope", rope_case),
        ("sdpa", sdpa_case),
        ("gate", gate_case),
        ("vres", vres_case),
        ("silu", silu_case),
        ("mul", mul_case),
        ("add", add_case),
        ("permute", permute_case),
        ("ce", ce_case),
        ("adamw_768x768", |b| {
            adamw_case(b, "adamw_768x768", D, D, 10)
        }),
        ("adamw_3072x768", |b| {
            adamw_case(b, "adamw_3072x768", 3072, D, 10)
        }),
        ("adamw_50304x768", |b| {
            adamw_case(b, "adamw_50304x768", V, D, 12)
        }),
        ("muon_768x768", |b| muon_case(b, "muon_768x768", D, D, 10)),
        ("muon_2048x768", |b| muon_case(b, "muon_2048x768", FF, D, 5)),
        ("muon_3072x768", |b| {
            muon_case(b, "muon_3072x768", 3072, D, 5)
        }),
        ("clip", clip_case),
        ("block", block_case),
    ]
}

fn env_list(key: &str) -> Option<Vec<String>> {
    std::env::var(key).ok().map(|v| {
        v.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

fn thread_counts() -> Vec<usize> {
    match env_list("OJAS_BENCH_THREADS") {
        Some(list) => list
            .iter()
            .map(|t| t.parse().expect("OJAS_BENCH_THREADS entry is not a number"))
            .collect(),
        None => vec![6, 18],
    }
}

fn selected() -> Vec<(&'static str, Builder)> {
    let all = builders();
    match env_list("OJAS_BENCH_OPS") {
        None => all,
        Some(keep) => {
            for k in &keep {
                assert!(
                    all.iter().any(|(name, _)| name == k),
                    "OJAS_BENCH_OPS names unknown case {k}"
                );
            }
            all.into_iter()
                .filter(|(name, _)| keep.iter().any(|k| k == name))
                .collect()
        }
    }
}

fn bench_dir() -> PathBuf {
    match std::env::var("OJAS_BENCH_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("../target-lane-measure/bench_ops"),
    }
}

fn backend(budget: &Budget, threads: usize) -> CpuBackend {
    let cpu = CpuBackend::with_threads(budget.clone(), threads).unwrap();
    assert_eq!(
        cpu.numerics(),
        Numerics::Fast,
        "CpuBackend default is not Fast"
    );
    cpu
}

fn shape_str(shape: &[usize]) -> String {
    if shape.is_empty() {
        "_".to_string()
    } else {
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Raw little-endian payload plus one manifest line.
fn dump_tensor(dir: &Path, manifest: &mut impl std::io::Write, kind: &str, name: &str, t: &Tensor) {
    let file = dir.join(format!("{kind}.{name}.bin"));
    let mut w = std::io::BufWriter::new(std::fs::File::create(&file).unwrap());
    let dtype = match t.dtype() {
        DType::F32 => {
            for v in t.to_f32_vec().unwrap() {
                w.write_all(&v.to_le_bytes()).unwrap();
            }
            "f32"
        }
        DType::U32 => {
            for v in t.to_u32_vec().unwrap() {
                w.write_all(&v.to_le_bytes()).unwrap();
            }
            "u32"
        }
        other => panic!("{name}: cannot dump {other:?}"),
    };
    w.flush().unwrap();
    writeln!(manifest, "{kind} {name} {dtype} {}", shape_str(t.shape())).unwrap();
}

fn run_dump(budget: &Budget) {
    let threads = thread_counts()[0];
    let cpu = backend(budget, threads);
    let root = bench_dir();
    for (name, build) in selected() {
        let mut case = build(budget);
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = Vec::new();
        writeln!(
            manifest,
            "case {} threads={threads} shape={}",
            case.name, case.shape
        )
        .unwrap();
        for (n, t) in case.names.iter().zip(&case.tensors) {
            dump_tensor(&dir, &mut manifest, "in", n, t);
        }
        for d in &case.dirs {
            let outs = call(d, &cpu, &mut case.tensors)
                .1
                .unwrap_or_else(|e| panic!("{name}.{}: {e:?}", d.name));
            for (n, t) in &outs {
                dump_tensor(&dir, &mut manifest, "out", &format!("{}.{n}", d.name), t);
            }
        }
        std::fs::write(dir.join("manifest.txt"), manifest).unwrap();
        println!("OJAS_DUMP case={name} dir={}", dir.display());
    }
}

fn run_time(budget: &Budget) {
    let backends: Vec<CpuBackend> = thread_counts()
        .into_iter()
        .map(|t| backend(budget, t))
        .collect();
    println!(
        "OJAS_META pid={} numerics=Fast threads={:?}",
        std::process::id(),
        backends.iter().map(CpuBackend::threads).collect::<Vec<_>>()
    );
    for (name, build) in selected() {
        let mut case = build(budget);
        for cpu in &backends {
            for d in &case.dirs {
                for _ in 0..2 {
                    std::hint::black_box(call(d, cpu, &mut case.tensors).1.unwrap());
                }
                let mut ns = Vec::with_capacity(d.n);
                for _ in 0..d.n {
                    let (dt, outs) = call(d, cpu, &mut case.tensors);
                    let outs = outs.unwrap();
                    std::hint::black_box(&outs);
                    drop(outs);
                    ns.push(dt.as_nanos());
                }
                ns.sort_unstable();
                println!(
                    "OJAS_OP op={name} dir={} shape={} threads={} n={} min_ms={:.4} med_ms={:.4}",
                    d.name,
                    case.shape.replace(' ', "_"),
                    cpu.threads(),
                    d.n,
                    ns[0] as f64 / 1e6,
                    ns[ns.len() / 2] as f64 / 1e6
                );
            }
        }
    }
}

fn run_loop(budget: &Budget) {
    let cases = selected();
    assert_eq!(
        cases.len(),
        1,
        "loop mode needs exactly one OJAS_BENCH_OPS case"
    );
    let dir_name = std::env::var("OJAS_BENCH_DIRSEL").expect("loop mode needs OJAS_BENCH_DIRSEL");
    let sample_out =
        std::env::var("OJAS_BENCH_SAMPLE_OUT").expect("loop mode needs OJAS_BENCH_SAMPLE_OUT");
    let secs: u64 = std::env::var("OJAS_BENCH_LOOP_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let cpu = backend(budget, thread_counts()[0]);
    let (name, build) = cases[0];
    let mut case = build(budget);
    let d = case
        .dirs
        .iter()
        .find(|d| d.name == dir_name)
        .unwrap_or_else(|| panic!("{name} has no direction {dir_name}"));
    for _ in 0..2 {
        std::hint::black_box(call(d, &cpu, &mut case.tensors).1.unwrap());
    }
    let mut child = std::process::Command::new("/usr/bin/sample")
        .arg(std::process::id().to_string())
        .arg(secs.to_string())
        .arg("-f")
        .arg(&sample_out)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn /usr/bin/sample");
    let start = Instant::now();
    let mut calls = 0u64;
    let mut busy = Duration::ZERO;
    while child.try_wait().unwrap().is_none()
        && start.elapsed() < Duration::from_secs(secs * 4 + 10)
    {
        let (dt, outs) = call(d, &cpu, &mut case.tensors);
        std::hint::black_box(outs.unwrap());
        busy += dt;
        calls += 1;
    }
    let status = child.wait().unwrap();
    println!(
        "OJAS_LOOP op={name} dir={dir_name} threads={} calls={calls} mean_ms={:.4} sample_status={status} out={sample_out}",
        cpu.threads(),
        busy.as_secs_f64() * 1e3 / calls.max(1) as f64
    );
}

#[test]
#[ignore]
fn bench_ops() {
    let budget = Budget::new(BUDGET_BYTES);
    match std::env::var("OJAS_BENCH_MODE").as_deref() {
        Ok("dump") => run_dump(&budget),
        Ok("loop") => run_loop(&budget),
        Ok("time") | Err(_) => run_time(&budget),
        Ok(other) => panic!("OJAS_BENCH_MODE {other} is not dump, time or loop"),
    }
}
