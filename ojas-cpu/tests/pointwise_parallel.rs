//! The pointwise, gate, value-residual, norm and permute kernels on the
//! worker pool.
//!
//! - Under `Numerics::Exact` every output is bit-identical to the serial
//!   scalar loops these ops ran before they used the pool (copied below as
//!   the reference), at every thread count.
//! - Under `Numerics::Fast` every output is within a stated tolerance of an
//!   `f64` reference and its bits do not depend on the thread count.
//! - A NaN or infinity in any operand is refused as `NonFinite` before
//!   anything is charged, so a backend with no budget at all still reports it.
//!
//! Tolerances are normalized the way `ojas-cpu/benches/torch_ops.py`
//! normalizes them: `max |got - ref| / max |ref|` over the tensor.

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_nonfinite, bits, SplitMix64};

const THREADS: [usize; 4] = [1, 2, 7, 18];

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

/// `max |got - want| / max |want|`.
fn normalized(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - w).abs()));
    if scale == 0.0 {
        err
    } else {
        err / scale
    }
}

fn assert_close(what: &str, got: &[f32], want: &[f64], tol: f64) {
    let err = normalized(got, want);
    assert!(err <= tol, "{what}: normalized error {err:e} > {tol:e}");
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---- The serial scalar loops the ops ran before the pool (Exact bits) ----

fn sigmoid_ref(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn silu_fwd_ref(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * sigmoid_ref(v)).collect()
}

fn silu_bwd_ref(x: &[f32], gy: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(gy)
        .map(|(&v, &g)| {
            let s = sigmoid_ref(v);
            g * s * (1.0 + v * (1.0 - s))
        })
        .collect()
}

fn vres_fwd_ref(value: &[f32], value0: &[f32], lambda: f32) -> Vec<f32> {
    let s = sigmoid_ref(lambda);
    value
        .iter()
        .zip(value0)
        .map(|(v, v0)| (1.0 - s) * v + s * v0)
        .collect()
}

fn vres_bwd_ref(
    value: &[f32],
    value0: &[f32],
    lambda: f32,
    gy: &[f32],
) -> (Vec<f32>, Vec<f32>, f32) {
    let s = sigmoid_ref(lambda);
    let mut gv = Vec::new();
    let mut gv0 = Vec::new();
    let mut gs = 0.0f32;
    for i in 0..value.len() {
        gv.push((1.0 - s) * gy[i]);
        gv0.push(s * gy[i]);
        gs += (value0[i] - value[i]) * gy[i];
    }
    (gv, gv0, gs * s * (1.0 - s))
}

struct Gate {
    rows: usize,
    din: usize,
    heads: usize,
    dh: usize,
}

fn gate_value_ref(x: &[f32], w: &[f32], b: &[f32], row: usize, head: usize, l: &Gate) -> f32 {
    let mut z = b[head];
    for i in 0..l.din {
        z += x[row * l.din + i] * w[head * l.din + i];
    }
    sigmoid_ref(z)
}

fn gate_fwd_ref(x: &[f32], w: &[f32], b: &[f32], attn: &[f32], l: &Gate) -> Vec<f32> {
    let mut y = vec![0.0f32; attn.len()];
    for row in 0..l.rows {
        for head in 0..l.heads {
            let g = gate_value_ref(x, w, b, row, head, l);
            for d in 0..l.dh {
                let ai = (row * l.heads + head) * l.dh + d;
                y[ai] = attn[ai] * g;
            }
        }
    }
    y
}

type GateGrads = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

#[allow(clippy::needless_range_loop)] // Verbatim copy of the pre-pool loop.
fn gate_bwd_ref(x: &[f32], w: &[f32], b: &[f32], attn: &[f32], gy: &[f32], l: &Gate) -> GateGrads {
    let mut gx = vec![0.0f32; x.len()];
    let mut gw = vec![0.0f32; w.len()];
    let mut gb = vec![0.0f32; l.heads];
    let mut ga = vec![0.0f32; attn.len()];
    for row in 0..l.rows {
        for head in 0..l.heads {
            let g = gate_value_ref(x, w, b, row, head, l);
            let mut grad_g = 0.0f32;
            for d in 0..l.dh {
                let ai = (row * l.heads + head) * l.dh + d;
                ga[ai] = gy[ai] * g;
                grad_g += gy[ai] * attn[ai];
            }
            let grad_z = grad_g * g * (1.0 - g);
            gb[head] += grad_z;
            for i in 0..l.din {
                let xi = row * l.din + i;
                let wi = head * l.din + i;
                gx[xi] += grad_z * w[wi];
                gw[wi] += grad_z * x[xi];
            }
        }
    }
    (gx, gw, gb, ga)
}

fn rstd_ref(row: &[f32], eps: f32) -> f32 {
    let mut sum_sq = 0.0f32;
    for &value in row {
        sum_sq += value * value;
    }
    1.0 / (sum_sq / row.len() as f32 + eps).sqrt()
}

fn rms_fwd_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let dim = w.len();
    let mut y = vec![0.0f32; x.len()];
    for (src, dst) in x.chunks(dim).zip(y.chunks_mut(dim)) {
        let rstd = rstd_ref(src, eps);
        for col in 0..dim {
            dst[col] = src[col] * rstd * w[col];
        }
    }
    y
}

fn rms_bwd_ref(x: &[f32], w: &[f32], gy: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let dim = w.len();
    let rows = x.len() / dim;
    let inv_dim = 1.0f32 / dim as f32;
    let mut gx = vec![0.0f32; x.len()];
    let mut rstds = Vec::new();
    for row in 0..rows {
        let src = &x[row * dim..(row + 1) * dim];
        let g = &gy[row * dim..(row + 1) * dim];
        let rstd = rstd_ref(src, eps);
        let mut dot = 0.0f32;
        for col in 0..dim {
            dot += (g[col] * w[col]) * (src[col] * rstd);
        }
        let mean = dot * inv_dim;
        for col in 0..dim {
            let dxhat = g[col] * w[col];
            let xhat = src[col] * rstd;
            gx[row * dim + col] = (dxhat - xhat * mean) * rstd;
        }
        rstds.push(rstd);
    }
    let mut gw = vec![0.0f32; dim];
    for (row, &r) in rstds.iter().enumerate() {
        for col in 0..dim {
            gw[col] += gy[row * dim + col] * (x[row * dim + col] * r);
        }
    }
    (gx, gw)
}

// ---- f64 references ----

fn gate_fwd64(x: &[f32], w: &[f32], b: &[f32], attn: &[f32], l: &Gate) -> Vec<f64> {
    let mut y = vec![0.0f64; attn.len()];
    for row in 0..l.rows {
        for head in 0..l.heads {
            let mut z = f64::from(b[head]);
            for i in 0..l.din {
                z += f64::from(x[row * l.din + i]) * f64::from(w[head * l.din + i]);
            }
            let g = sigmoid64(z);
            for d in 0..l.dh {
                let ai = (row * l.heads + head) * l.dh + d;
                y[ai] = f64::from(attn[ai]) * g;
            }
        }
    }
    y
}

type GateGrads64 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn gate_bwd64(x: &[f32], w: &[f32], b: &[f32], attn: &[f32], gy: &[f32], l: &Gate) -> GateGrads64 {
    let mut gx = vec![0.0f64; x.len()];
    let mut gw = vec![0.0f64; w.len()];
    let mut gb = vec![0.0f64; l.heads];
    let mut ga = vec![0.0f64; attn.len()];
    for row in 0..l.rows {
        for head in 0..l.heads {
            let mut z = f64::from(b[head]);
            for i in 0..l.din {
                z += f64::from(x[row * l.din + i]) * f64::from(w[head * l.din + i]);
            }
            let g = sigmoid64(z);
            let mut grad_g = 0.0f64;
            for d in 0..l.dh {
                let ai = (row * l.heads + head) * l.dh + d;
                ga[ai] = f64::from(gy[ai]) * g;
                grad_g += f64::from(gy[ai]) * f64::from(attn[ai]);
            }
            let grad_z = grad_g * g * (1.0 - g);
            gb[head] += grad_z;
            for i in 0..l.din {
                let xi = row * l.din + i;
                let wi = head * l.din + i;
                gx[xi] += grad_z * f64::from(w[wi]);
                gw[wi] += grad_z * f64::from(x[xi]);
            }
        }
    }
    (gx, gw, gb, ga)
}

fn rms_fwd64(x: &[f32], w: &[f32], eps: f32) -> Vec<f64> {
    let dim = w.len();
    let mut y = Vec::with_capacity(x.len());
    for src in x.chunks(dim) {
        let ms = src
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / dim as f64;
        let rstd = 1.0 / (ms + f64::from(eps)).sqrt();
        for col in 0..dim {
            y.push(f64::from(src[col]) * rstd * f64::from(w[col]));
        }
    }
    y
}

fn rms_bwd64(x: &[f32], w: &[f32], gy: &[f32], eps: f32) -> (Vec<f64>, Vec<f64>) {
    let dim = w.len();
    let rows = x.len() / dim;
    let mut gx = Vec::with_capacity(x.len());
    let mut gw = vec![0.0f64; dim];
    for row in 0..rows {
        let src = &x[row * dim..(row + 1) * dim];
        let g = &gy[row * dim..(row + 1) * dim];
        let ms = src
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / dim as f64;
        let rstd = 1.0 / (ms + f64::from(eps)).sqrt();
        let mut dot = 0.0f64;
        for col in 0..dim {
            dot += f64::from(g[col]) * f64::from(w[col]) * f64::from(src[col]) * rstd;
        }
        let mean = dot / dim as f64;
        for col in 0..dim {
            let dxhat = f64::from(g[col]) * f64::from(w[col]);
            let xhat = f64::from(src[col]) * rstd;
            gx.push((dxhat - xhat * mean) * rstd);
            gw[col] += f64::from(g[col]) * xhat;
        }
    }
    (gx, gw)
}

// ---- Fixtures ----

/// Above every parallel threshold: about ten row chunks on 18 threads.
const EW: [usize; 2] = [640, 515];

struct GateCase {
    l: Gate,
    x: Vec<f32>,
    w: Vec<f32>,
    b: Vec<f32>,
    attn: Vec<f32>,
    gy: Vec<f32>,
}

impl GateCase {
    fn new(seed: u64, batch: usize, time: usize, din: usize, heads: usize, dh: usize) -> Self {
        let mut rng = SplitMix64(seed);
        let rows = batch * time;
        Self {
            x: rng.vec(rows * din, 1.0),
            w: rng.vec(heads * din, 1.0 / (din as f32).sqrt()),
            b: rng.vec(heads, 0.1),
            attn: rng.vec(rows * heads * dh, 1.0),
            gy: rng.vec(rows * heads * dh, 0.01),
            l: Gate {
                rows,
                din,
                heads,
                dh,
            },
        }
    }

    fn shapes(&self) -> [Vec<usize>; 3] {
        let rows = self.l.rows;
        [
            vec![rows, self.l.din],
            vec![rows, self.l.heads, self.l.dh],
            vec![self.l.heads, self.l.din],
        ]
    }

    fn tensors(&self) -> [Tensor; 5] {
        let [xs, attn_s, ws] = self.shapes();
        [
            t(&self.x, &xs),
            t(&self.w, &ws),
            t(&self.b, &[self.l.heads]),
            t(&self.attn, &attn_s),
            t(&self.gy, &attn_s),
        ]
    }

    fn run(&self, cpu: &CpuBackend) -> (Vec<f32>, GateGrads) {
        let [x, w, b, attn, gy] = self.tensors();
        let y = cpu
            .per_head_sigmoid_gate_forward(&x, &w, &b, &attn)
            .unwrap();
        let g = cpu
            .per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &gy)
            .unwrap();
        (
            v(&y),
            (v(&g.input), v(&g.weight), v(&g.bias), v(&g.attn_out)),
        )
    }
}

/// `(small, below 2^21 multiply-adds, above it)`. Under Fast the last one is
/// one Accelerate call on macOS, whose bits Apple does not pin, so it is
/// held to the tolerance only. Since the macOS cutoff became
/// `ojas_cpu::FAST_WHOLE_CALL_MACS` (2^13), the first two also reach
/// Accelerate there; the thread-count test still passes because one
/// Accelerate call is made whatever the pool size.
fn gate_cases() -> [GateCase; 3] {
    [
        GateCase::new(11, 2, 150, 96, 6, 32),
        GateCase::new(12, 1, 1024, 160, 12, 64),
        GateCase::new(13, 1, 1024, 300, 12, 64),
    ]
}

// ---- Exact: bits of the scalar loops at every thread count ----

#[test]
fn exact_gate_matches_the_scalar_loops_at_every_thread_count() {
    for case in gate_cases() {
        let l = &case.l;
        let want_y = gate_fwd_ref(&case.x, &case.w, &case.b, &case.attn, l);
        let (gx, gw, gb, ga) = gate_bwd_ref(&case.x, &case.w, &case.b, &case.attn, &case.gy, l);
        for threads in THREADS {
            let (y, (rx, rw, rb, ra)) = case.run(&backend(threads, Numerics::Exact));
            let at = format!("rows {} din {} threads {threads}", l.rows, l.din);
            assert_eq!(bits(&y), bits(&want_y), "gate y {at}");
            assert_eq!(bits(&rx), bits(&gx), "gate grad_x {at}");
            assert_eq!(bits(&rw), bits(&gw), "gate grad_w {at}");
            assert_eq!(bits(&rb), bits(&gb), "gate grad_b {at}");
            assert_eq!(bits(&ra), bits(&ga), "gate grad_attn {at}");
        }
    }
}

#[test]
fn exact_silu_mul_add_match_the_scalar_loops_at_every_thread_count() {
    let mut rng = SplitMix64(21);
    let n = EW[0] * EW[1];
    let x = rng.vec(n, 6.0);
    let y = rng.vec(n, 1.0);
    let gy = rng.vec(n, 0.01);
    let (xt, yt, gt) = (t(&x, &EW), t(&y, &EW), t(&gy, &EW));
    let silu_y = silu_fwd_ref(&x);
    let silu_g = silu_bwd_ref(&x, &gy);
    let mul_y: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a * b).collect();
    let mul_ga: Vec<f32> = gy.iter().zip(&y).map(|(g, b)| g * b).collect();
    let mul_gb: Vec<f32> = gy.iter().zip(&x).map(|(g, a)| g * a).collect();
    let add_y: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    for threads in THREADS {
        // mul and add are one rounding per element; Fast is held to the
        // same bits.
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let cpu = backend(threads, numerics);
            let at = format!("threads {threads} {numerics:?}");
            if numerics == Numerics::Exact {
                assert_eq!(
                    bits(&v(&cpu.silu_forward(&xt).unwrap())),
                    bits(&silu_y),
                    "silu y {at}"
                );
                assert_eq!(
                    bits(&v(&cpu.silu_backward(&xt, &gt).unwrap())),
                    bits(&silu_g),
                    "silu grad {at}"
                );
            }
            assert_eq!(
                bits(&v(&cpu.mul_forward(&xt, &yt).unwrap())),
                bits(&mul_y),
                "mul y {at}"
            );
            let (ga, gb) = cpu.mul_backward(&xt, &yt, &gt).unwrap();
            assert_eq!(bits(&v(&ga)), bits(&mul_ga), "mul grad_a {at}");
            assert_eq!(bits(&v(&gb)), bits(&mul_gb), "mul grad_b {at}");
            assert_eq!(
                bits(&v(&cpu.residual_add_forward(&xt, &yt).unwrap())),
                bits(&add_y),
                "add z {at}"
            );
            let (gx, gy2) = cpu.residual_add_backward(&xt, &yt, &gt).unwrap();
            assert_eq!(bits(&v(&gx)), bits(&gy), "add grad_x {at}");
            assert_eq!(bits(&v(&gy2)), bits(&gy), "add grad_y {at}");
            assert_eq!(gx.shape(), &EW[..]);
            assert_eq!(gy2.shape(), &EW[..]);
        }
    }
}

fn vres_fixture(seed: u64) -> (Vec<f32>, Vec<f32>, f32, Vec<f32>, Vec<usize>) {
    // The bench shape and distributions (bench_ops.rs `vres_case`).
    let shape = vec![1, 1024, 12, 64];
    let n = shape.iter().product();
    let mut rng = SplitMix64(seed);
    let value = rng.vec(n, 1.0);
    let value0 = rng.vec(n, 1.0);
    let lambda = 0.3 * rng.unit();
    let gy = rng.vec(n, 0.01);
    (value, value0, lambda, gy, shape)
}

#[test]
fn exact_value_residual_matches_the_scalar_loops_at_every_thread_count() {
    let (value, value0, lambda, gy, shape) = vres_fixture(31);
    let want_y = vres_fwd_ref(&value, &value0, lambda);
    let (gv, gv0, gl) = vres_bwd_ref(&value, &value0, lambda, &gy);
    let (vt, v0t, lt, gt) = (
        t(&value, &shape),
        t(&value0, &shape),
        t(&[lambda], &[1]),
        t(&gy, &shape),
    );
    for threads in THREADS {
        let cpu = backend(threads, Numerics::Exact);
        let y = cpu.value_residual_blend_forward(&vt, &v0t, &lt).unwrap();
        assert_eq!(bits(&v(&y)), bits(&want_y), "vres y threads {threads}");
        let g = cpu
            .value_residual_blend_backward(&vt, &v0t, &lt, &gt)
            .unwrap();
        assert_eq!(
            bits(&v(&g.value)),
            bits(&gv),
            "vres grad_v threads {threads}"
        );
        assert_eq!(
            bits(&v(&g.value0)),
            bits(&gv0),
            "vres grad_v0 threads {threads}"
        );
        // Exact keeps the ascending-index f32 reduction (lib.rs, ojas-core
        // `Numerics::Exact`).
        assert_eq!(
            v(&g.lambda)[0].to_bits(),
            gl.to_bits(),
            "vres grad_lambda threads {threads}"
        );
        assert_eq!(g.lambda.shape(), &[1]);
    }
}

/// RMSNorm keeps the ascending row sums under both contracts (norm.rs sums
/// eight rows side by side without reordering any row), so Fast is held to
/// the scalar loops' bits too. Row counts that are and are not multiples of
/// eight, split across tasks at every thread count.
#[test]
fn rms_norm_matches_the_scalar_loops_under_both_contracts_at_every_thread_count() {
    let mut rng = SplitMix64(41);
    for shape in [
        vec![700usize, 192],
        vec![1, 200, 4, 64],
        vec![37, 768],
        vec![3, 5],
    ] {
        let dim = *shape.last().unwrap();
        let n: usize = shape.iter().product();
        let x = rng.vec(n, 1.0);
        let w: Vec<f32> = rng.vec(dim, 0.1).iter().map(|d| 1.0 + d).collect();
        let gy = rng.vec(n, 0.01);
        let want_y = rms_fwd_ref(&x, &w, 1e-6);
        let (want_gx, want_gw) = rms_bwd_ref(&x, &w, &gy, 1e-6);
        let (xt, wt, gt) = (t(&x, &shape), t(&w, &[dim]), t(&gy, &shape));
        for threads in THREADS {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let cpu = backend(threads, numerics);
                let at = format!("{shape:?} threads {threads} {numerics:?}");
                let y = cpu.rms_norm_forward(&xt, &wt, 1e-6).unwrap();
                assert_eq!(bits(&v(&y)), bits(&want_y), "rms y {at}");
                let (gx, gw) = cpu.rms_norm_backward(&xt, &wt, &gt, 1e-6).unwrap();
                assert_eq!(bits(&v(&gx)), bits(&want_gx), "rms grad_x {at}");
                assert_eq!(bits(&v(&gw)), bits(&want_gw), "rms grad_w {at}");
            }
        }
    }
}

// ---- Fast: tolerance against f64, and bits independent of threads ----

/// Every output of every op under Fast, at one thread count.
fn fast_outputs(threads: usize) -> Vec<(String, Vec<f32>)> {
    let cpu = backend(threads, Numerics::Fast);
    let mut out = Vec::new();
    let mut rng = SplitMix64(51);
    let n = EW[0] * EW[1];
    let x = rng.vec(n, 6.0);
    let gy = rng.vec(n, 0.01);
    let (xt, gt) = (t(&x, &EW), t(&gy, &EW));
    out.push(("silu y".to_string(), v(&cpu.silu_forward(&xt).unwrap())));
    out.push((
        "silu grad".to_string(),
        v(&cpu.silu_backward(&xt, &gt).unwrap()),
    ));
    let (value, value0, lambda, vgy, shape) = vres_fixture(52);
    let (vt, v0t, lt, vgt) = (
        t(&value, &shape),
        t(&value0, &shape),
        t(&[lambda], &[1]),
        t(&vgy, &shape),
    );
    out.push((
        "vres y".to_string(),
        v(&cpu.value_residual_blend_forward(&vt, &v0t, &lt).unwrap()),
    ));
    let g = cpu
        .value_residual_blend_backward(&vt, &v0t, &lt, &vgt)
        .unwrap();
    out.push(("vres grad_v".to_string(), v(&g.value)));
    out.push(("vres grad_v0".to_string(), v(&g.value0)));
    out.push(("vres grad_lambda".to_string(), v(&g.lambda)));
    for shape in [vec![700usize, 192], vec![1, 1024, 12, 64]] {
        let dim = *shape.last().unwrap();
        let n: usize = shape.iter().product();
        let x = rng.vec(n, 1.0);
        let w: Vec<f32> = rng.vec(dim, 0.1).iter().map(|d| 1.0 + d).collect();
        let gy = rng.vec(n, 0.01);
        let (xt, wt, gt) = (t(&x, &shape), t(&w, &[dim]), t(&gy, &shape));
        out.push((
            format!("rms y {shape:?}"),
            v(&cpu.rms_norm_forward(&xt, &wt, 1e-6).unwrap()),
        ));
        let (gx, gw) = cpu.rms_norm_backward(&xt, &wt, &gt, 1e-6).unwrap();
        out.push((format!("rms grad_x {shape:?}"), v(&gx)));
        out.push((format!("rms grad_w {shape:?}"), v(&gw)));
    }
    // The two gate shapes below 2^21 multiply-adds (see `gate_cases`; on
    // macOS they are whole Accelerate calls, made the same at every pool size).
    for case in gate_cases().into_iter().take(2) {
        let (y, (gx, gw, gb, ga)) = case.run(&cpu);
        let at = format!("rows {} din {}", case.l.rows, case.l.din);
        out.push((format!("gate y {at}"), y));
        out.push((format!("gate grad_x {at}"), gx));
        out.push((format!("gate grad_w {at}"), gw));
        out.push((format!("gate grad_b {at}"), gb));
        out.push((format!("gate grad_attn {at}"), ga));
    }
    out
}

#[test]
fn fast_bits_do_not_depend_on_the_thread_count() {
    let base = fast_outputs(THREADS[0]);
    for threads in &THREADS[1..] {
        for ((name, want), (_, got)) in base.iter().zip(fast_outputs(*threads)) {
            assert_eq!(bits(&got), bits(want), "{name}: threads {threads} vs 1");
        }
    }
}

#[test]
fn fast_gate_is_within_tolerance_of_f64() {
    // torch_ops.py TOL["gate"].
    const TOL: f64 = 1e-5;
    for case in gate_cases() {
        let l = &case.l;
        let want_y = gate_fwd64(&case.x, &case.w, &case.b, &case.attn, l);
        let (gx, gw, gb, ga) = gate_bwd64(&case.x, &case.w, &case.b, &case.attn, &case.gy, l);
        for threads in [1usize, 7] {
            let (y, (rx, rw, rb, ra)) = case.run(&backend(threads, Numerics::Fast));
            let at = format!("rows {} din {} threads {threads}", l.rows, l.din);
            assert_close(&format!("gate y {at}"), &y, &want_y, TOL);
            assert_close(&format!("gate grad_x {at}"), &rx, &gx, TOL);
            assert_close(&format!("gate grad_w {at}"), &rw, &gw, TOL);
            assert_close(&format!("gate grad_b {at}"), &rb, &gb, TOL);
            assert_close(&format!("gate grad_attn {at}"), &ra, &ga, TOL);
        }
    }
}

#[test]
fn fast_silu_is_within_tolerance_of_f64() {
    // Per element: |got - want| <= REL * scale + FLOOR, where scale is the sum
    // of the magnitudes the formula combines, so the zero of the backward
    // factor `1 + x (1 - s)` near x = -1.28 does not demand a relative error
    // the f32 inputs cannot carry. FLOOR is the smallest normal f32: results
    // below it are subnormal and carry fewer bits.
    const REL: f64 = 1e-6;
    const FLOOR: f64 = f32::MIN_POSITIVE as f64;
    let mut x: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1e-30,
        -1e-30,
        1e-45,
        -1e-45,
        5.0,
        -5.0,
        15.0,
        -15.0,
        20.0,
        -20.0,
        87.3,
        -87.3,
        88.7,
        -88.7,
        89.0,
        -89.0,
        103.9,
        -103.9,
        104.0,
        -104.0,
        120.0,
        -120.0,
        1e30,
        -1e30,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
    ];
    // A dense sweep over [-110, 110].
    let steps = 1 << 18;
    for i in 0..=steps {
        x.push(-110.0 + 220.0 * (i as f32) / (steps as f32));
    }
    let n = x.len();
    let gy: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -0.37 })
        .collect();
    let (xt, gt) = (t(&x, &[n]), t(&gy, &[n]));
    for threads in [1usize, 7] {
        let cpu = backend(threads, Numerics::Fast);
        let y = v(&cpu.silu_forward(&xt).unwrap());
        let g = v(&cpu.silu_backward(&xt, &gt).unwrap());
        for i in 0..n {
            let xv = f64::from(x[i]);
            let s = sigmoid64(xv);
            let want_y = xv * s;
            let err_y = (f64::from(y[i]) - want_y).abs();
            assert!(
                err_y <= REL * want_y.abs() + FLOOR,
                "silu({}) = {} want {want_y:e} (threads {threads})",
                x[i],
                y[i]
            );
            let gv = f64::from(gy[i]);
            let want_g = gv * s * (1.0 + xv * (1.0 - s));
            let scale = (gv * s).abs() * (1.0 + (xv * (1.0 - s)).abs());
            let err_g = (f64::from(g[i]) - want_g).abs();
            assert!(
                err_g <= REL * scale + FLOOR,
                "silu'({}) = {} want {want_g:e} (threads {threads})",
                x[i],
                g[i]
            );
        }
    }
}

#[test]
fn fast_value_residual_and_its_lambda_gradient_are_within_tolerance_of_f64() {
    // The lambda gradient is a sum of about 786k products. Summed serially in
    // f32 it was 3.2e-5 off the f64 sum at this shape; torch is 1.4e-7.
    const LAMBDA_REL: f64 = 1e-6;
    const TOL: f64 = 1e-6;
    for seed in [61u64, 62, 63] {
        let (value, value0, lambda, gy, shape) = vres_fixture(seed);
        let s = sigmoid64(f64::from(lambda));
        let want_y: Vec<f64> = value
            .iter()
            .zip(&value0)
            .map(|(&a, &b)| (1.0 - s) * f64::from(a) + s * f64::from(b))
            .collect();
        let want_gv: Vec<f64> = gy.iter().map(|&g| (1.0 - s) * f64::from(g)).collect();
        let want_gv0: Vec<f64> = gy.iter().map(|&g| s * f64::from(g)).collect();
        let grad_s: f64 = (0..value.len())
            .map(|i| (f64::from(value0[i]) - f64::from(value[i])) * f64::from(gy[i]))
            .sum();
        let want_gl = grad_s * s * (1.0 - s);
        let (vt, v0t, lt, gt) = (
            t(&value, &shape),
            t(&value0, &shape),
            t(&[lambda], &[1]),
            t(&gy, &shape),
        );
        for threads in [1usize, 7] {
            let cpu = backend(threads, Numerics::Fast);
            let y = v(&cpu.value_residual_blend_forward(&vt, &v0t, &lt).unwrap());
            assert_close("vres y", &y, &want_y, TOL);
            let g = cpu
                .value_residual_blend_backward(&vt, &v0t, &lt, &gt)
                .unwrap();
            assert_close("vres grad_v", &v(&g.value), &want_gv, TOL);
            assert_close("vres grad_v0", &v(&g.value0), &want_gv0, TOL);
            let gl = f64::from(v(&g.lambda)[0]);
            let rel = (gl - want_gl).abs() / want_gl.abs();
            assert!(
                rel <= LAMBDA_REL,
                "seed {seed} threads {threads}: grad_lambda {gl:e} vs f64 {want_gl:e}, relative error {rel:e} > {LAMBDA_REL:e}"
            );
        }
    }
}

/// The tolerance above discriminates: the ascending-index f32 sum, which is
/// what `Numerics::Exact` keeps and what `Numerics::Fast` ran before its
/// f64 block sum, misses it at this shape.
#[test]
fn the_ascending_f32_lambda_sum_misses_the_fast_tolerance() {
    let mut worst = 0.0f64;
    for seed in [61u64, 62, 63] {
        let (value, value0, lambda, gy, shape) = vres_fixture(seed);
        let s = sigmoid64(f64::from(lambda));
        let grad_s: f64 = (0..value.len())
            .map(|i| (f64::from(value0[i]) - f64::from(value[i])) * f64::from(gy[i]))
            .sum();
        let want = grad_s * s * (1.0 - s);
        let (vt, v0t, lt, gt) = (
            t(&value, &shape),
            t(&value0, &shape),
            t(&[lambda], &[1]),
            t(&gy, &shape),
        );
        let g = backend(7, Numerics::Exact)
            .value_residual_blend_backward(&vt, &v0t, &lt, &gt)
            .unwrap();
        let (_, _, serial) = vres_bwd_ref(&value, &value0, lambda, &gy);
        assert_eq!(v(&g.lambda)[0].to_bits(), serial.to_bits());
        worst = worst.max((f64::from(serial) - want).abs() / want.abs());
    }
    assert!(
        worst > 1e-6,
        "ascending f32 lambda sum within {worst:e} of f64"
    );
}

#[test]
fn fast_rms_norm_is_within_tolerance_of_f64() {
    // torch_ops.py TOL["rms_norm"].
    const TOL: f64 = 1e-5;
    let mut rng = SplitMix64(71);
    for shape in [vec![700usize, 192], vec![1, 1024, 12, 64], vec![33, 768]] {
        let dim = *shape.last().unwrap();
        let n: usize = shape.iter().product();
        let x = rng.vec(n, 1.0);
        let w: Vec<f32> = rng.vec(dim, 0.1).iter().map(|d| 1.0 + d).collect();
        let gy = rng.vec(n, 0.01);
        let (xt, wt, gt) = (t(&x, &shape), t(&w, &[dim]), t(&gy, &shape));
        let want_y = rms_fwd64(&x, &w, 1e-6);
        let (want_gx, want_gw) = rms_bwd64(&x, &w, &gy, 1e-6);
        for threads in [1usize, 7] {
            let cpu = backend(threads, Numerics::Fast);
            let y = v(&cpu.rms_norm_forward(&xt, &wt, 1e-6).unwrap());
            assert_close(&format!("rms y {shape:?}"), &y, &want_y, TOL);
            let (gx, gw) = cpu.rms_norm_backward(&xt, &wt, &gt, 1e-6).unwrap();
            assert_close(&format!("rms grad_x {shape:?}"), &v(&gx), &want_gx, TOL);
            assert_close(&format!("rms grad_w {shape:?}"), &v(&gw), &want_gw, TOL);
        }
    }
}

// ---- Permute ----

/// `permute(dims).contiguous()` by explicit index arithmetic.
fn permute_ref(x: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    let rank = shape.len();
    let mut in_strides = vec![1usize; rank];
    for axis in (0..rank.saturating_sub(1)).rev() {
        in_strides[axis] = in_strides[axis + 1] * shape[axis + 1];
    }
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let mut rest = flat;
        let mut src = 0;
        for axis in (0..rank).rev() {
            let i = rest % out_shape[axis];
            rest /= out_shape[axis];
            src += i * in_strides[dims[axis]];
        }
        out.push(x[src]);
    }
    out
}

#[test]
fn permute_moves_bits_for_runs_and_single_values_at_every_thread_count() {
    let mut rng = SplitMix64(81);
    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![1, 1024, 12, 64], vec![0, 2, 1, 3]),
        (vec![2, 37, 5, 3], vec![0, 2, 1, 3]),
        (vec![3, 4, 5, 6], vec![1, 0, 2, 3]),
        (vec![3, 4, 5, 6], vec![2, 0, 1, 3]),
        (vec![3, 4, 5, 6], vec![0, 1, 3, 2]),
        (vec![3, 4, 5, 6], vec![3, 2, 1, 0]),
        (vec![3, 4, 5, 6], vec![0, 1, 2, 3]),
        (vec![64, 128, 64], vec![2, 0, 1]),
        (vec![7, 9], vec![1, 0]),
        (vec![11], vec![0]),
    ];
    for (shape, dims) in cases {
        let n: usize = shape.iter().product();
        let x = rng.vec(n, 1.0);
        let want = permute_ref(&x, &shape, &dims);
        let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
        let xt = t(&x, &shape);
        for threads in THREADS {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let y = backend(threads, numerics).permute(&xt, &dims).unwrap();
                assert_eq!(y.shape(), out_shape.as_slice());
                assert_eq!(
                    bits(&v(&y)),
                    bits(&want),
                    "{shape:?} {dims:?} threads {threads}"
                );
            }
        }
    }
}

#[test]
fn permute_accepts_rank_zero_and_empty_axes() {
    let cpu = backend(7, Numerics::Fast);
    let scalar = t(&[2.5], &[]);
    assert_eq!(v(&cpu.permute(&scalar, &[]).unwrap()), vec![2.5]);
    let empty = Tensor::zeros(&[2, 0, 3], ojas_core::DType::F32, &Budget::new(u64::MAX)).unwrap();
    let y = cpu.permute(&empty, &[1, 2, 0]).unwrap();
    assert_eq!(y.shape(), &[0, 3, 2]);
    assert!(v(&y).is_empty());
}

// ---- NaN and infinity: refused before any charge ----

fn poison(mut data: Vec<f32>, at: usize, value: f32) -> Vec<f32> {
    let i = at % data.len();
    data[i] = value;
    data
}

type Probe = Box<dyn Fn(&CpuBackend, &[Tensor]) -> Result<(), OjasError>>;
/// `(op name, operand data and shapes, op)`.
type NanCase = (&'static str, Vec<(Vec<f32>, Vec<usize>)>, Probe);

#[test]
fn a_nonfinite_operand_is_refused_before_any_charge() {
    let mut rng = SplitMix64(91);
    let ew = [64usize, 520];
    let n = ew[0] * ew[1];
    let gate = GateCase::new(92, 1, 64, 48, 4, 16);
    let [gxs, gas, gws] = gate.shapes();
    let cases: Vec<NanCase> = vec![
        (
            "silu_forward",
            vec![(rng.vec(n, 1.0), ew.to_vec())],
            Box::new(|c, a| c.silu_forward(&a[0]).map(|_| ())),
        ),
        (
            "silu_backward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
            ],
            Box::new(|c, a| c.silu_backward(&a[0], &a[1]).map(|_| ())),
        ),
        (
            "mul_forward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
            ],
            Box::new(|c, a| c.mul_forward(&a[0], &a[1]).map(|_| ())),
        ),
        (
            "mul_backward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
            ],
            Box::new(|c, a| c.mul_backward(&a[0], &a[1], &a[2]).map(|_| ())),
        ),
        (
            "residual_add_forward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
            ],
            Box::new(|c, a| c.residual_add_forward(&a[0], &a[1]).map(|_| ())),
        ),
        (
            "residual_add_backward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
            ],
            Box::new(|c, a| c.residual_add_backward(&a[0], &a[1], &a[2]).map(|_| ())),
        ),
        (
            "value_residual_blend_forward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
                (vec![0.2], vec![1]),
            ],
            Box::new(|c, a| {
                c.value_residual_blend_forward(&a[0], &a[1], &a[2])
                    .map(|_| ())
            }),
        ),
        (
            "value_residual_blend_backward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(n, 1.0), ew.to_vec()),
                (vec![0.2], vec![1]),
                (rng.vec(n, 1.0), ew.to_vec()),
            ],
            Box::new(|c, a| {
                c.value_residual_blend_backward(&a[0], &a[1], &a[2], &a[3])
                    .map(|_| ())
            }),
        ),
        (
            "rms_norm_forward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(ew[1], 1.0), vec![ew[1]]),
            ],
            Box::new(|c, a| c.rms_norm_forward(&a[0], &a[1], 1e-6).map(|_| ())),
        ),
        (
            "rms_norm_backward",
            vec![
                (rng.vec(n, 1.0), ew.to_vec()),
                (rng.vec(ew[1], 1.0), vec![ew[1]]),
                (rng.vec(n, 1.0), ew.to_vec()),
            ],
            Box::new(|c, a| c.rms_norm_backward(&a[0], &a[1], &a[2], 1e-6).map(|_| ())),
        ),
        (
            "per_head_sigmoid_gate_forward",
            vec![
                (gate.x.clone(), gxs.clone()),
                (gate.w.clone(), gws.clone()),
                (gate.b.clone(), vec![gate.l.heads]),
                (gate.attn.clone(), gas.clone()),
            ],
            Box::new(|c, a| {
                c.per_head_sigmoid_gate_forward(&a[0], &a[1], &a[2], &a[3])
                    .map(|_| ())
            }),
        ),
        (
            "per_head_sigmoid_gate_backward",
            vec![
                (gate.x.clone(), gxs),
                (gate.w.clone(), gws),
                (gate.b.clone(), vec![gate.l.heads]),
                (gate.attn.clone(), gas.clone()),
                (gate.gy.clone(), gas),
            ],
            Box::new(|c, a| {
                c.per_head_sigmoid_gate_backward(&a[0], &a[1], &a[2], &a[3], &a[4])
                    .map(|_| ())
            }),
        ),
        (
            "permute",
            vec![(rng.vec(n, 1.0), vec![4, 16, 2, 260])],
            Box::new(|c, a| c.permute(&a[0], &[0, 2, 1, 3]).map(|_| ())),
        ),
    ];
    for (name, operands, op) in &cases {
        for bad_at in 0..operands.len() {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                // The bad value sits deep in the last task's range.
                let args: Vec<Tensor> = operands
                    .iter()
                    .enumerate()
                    .map(|(i, (data, shape))| {
                        let data = if i == bad_at {
                            poison(data.clone(), data.len().saturating_sub(3), bad)
                        } else {
                            data.clone()
                        };
                        t(&data, shape)
                    })
                    .collect();
                for threads in [1usize, 7] {
                    for numerics in [Numerics::Exact, Numerics::Fast] {
                        let empty = CpuBackend::with_threads(Budget::new(0), threads)
                            .unwrap()
                            .with_numerics(numerics);
                        let result = op(&empty, &args);
                        if !matches!(result, Err(OjasError::NonFinite { .. })) {
                            panic!(
                                "{name}: {bad} in operand {bad_at} ({threads} threads, {numerics:?}): expected NonFinite, got {result:?}"
                            );
                        }
                        assert_eq!(empty.budget().live_bytes().unwrap(), 0, "{name}");
                    }
                }
            }
        }
    }
}

/// Finite operands whose gate logit overflows are refused, not turned into
/// a gate of 0 or 1.
#[test]
fn a_gate_logit_that_overflows_is_refused() {
    let x = t(&[1e30; 8], &[2, 4]);
    let w = t(&[1e30; 8], &[2, 4]);
    let b = t(&[0.0; 2], &[2]);
    let attn = t(&[1.0; 12], &[2, 2, 3]);
    for threads in [1usize, 7] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let cpu = backend(threads, numerics);
            let before = cpu.budget().live_bytes().unwrap();
            assert_nonfinite(cpu.per_head_sigmoid_gate_forward(&x, &w, &b, &attn));
            assert_eq!(cpu.budget().live_bytes().unwrap(), before);
            assert_nonfinite(cpu.per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &attn));
            assert_eq!(cpu.budget().live_bytes().unwrap(), before);
        }
    }
}

/// Fast `grad_attn` is a finite `grad_y` times a sigmoid clamped into
/// `[0, 1]`, so it is recorded finite. An overflowing per-head dot still
/// refuses: `grad_bias` and the GEMM gradients stay scanned, and the charge
/// is released.
#[test]
fn fast_gate_backward_trusts_the_attn_product_and_still_scans_the_reduction() {
    let cpu = backend(6, Numerics::Fast);
    let x = t(&[0.0; 4], &[2, 2]);
    let w = t(&[0.0; 4], &[2, 2]);
    let b = t(&[0.0; 2], &[2]);
    let attn = t(&[1.0; 4], &[2, 2, 1]);
    let gy = t(&[f32::MAX, -f32::MAX, f32::MAX, -f32::MAX], &[2, 2, 1]);
    let g = cpu
        .per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &gy)
        .unwrap();
    assert!(g
        .attn_out
        .all_finite_cached(|_| panic!("grad_attn scanned again"))
        .unwrap());
    let y = v(&g.attn_out);
    assert!(
        y.iter()
            .zip([1.0f32, -1.0, 1.0, -1.0])
            .all(|(got, sign)| got.is_finite() && got.signum() == sign),
        "{y:?}"
    );
    drop(g);

    let attn = t(&[1e20; 16], &[2, 2, 4]);
    let gy = t(&[1e20; 16], &[2, 2, 4]);
    let before = cpu.budget().live_bytes().unwrap();
    assert_nonfinite(cpu.per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &gy));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}

/// Fast backward with the scale the forward kept matches the recompute,
/// including a finite scale outside `[0, 1]` (clamped, not copied into the
/// formula raw). A non-finite saved scale is refused and the charge released.
/// Exact does not read the saved scale.
#[test]
fn saved_gate_scale_matches_the_recompute_and_a_non_finite_scale_releases() {
    let fast = backend(6, Numerics::Fast);
    let case = GateCase::new(21, 1, 32, 16, 4, 8);
    let [x, w, b, attn, gy] = case.tensors();
    let (y_kept, scales) = fast
        .per_head_sigmoid_gate_forward_saving(&x, &w, &b, &attn)
        .unwrap();
    let y = fast
        .per_head_sigmoid_gate_forward(&x, &w, &b, &attn)
        .unwrap();
    assert_eq!(
        bits(&v(&y_kept)),
        bits(&v(&y)),
        "keeping the scale changed y"
    );
    let scales = scales.expect("fast forward keeps the per-head scale");
    let with = fast
        .per_head_sigmoid_gate_backward_saved(&x, &w, &b, &attn, &gy, &scales)
        .unwrap();
    let without = fast
        .per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &gy)
        .unwrap();
    for (name, a, b) in [
        ("gx", &with.input, &without.input),
        ("gw", &with.weight, &without.weight),
        ("gb", &with.bias, &without.bias),
        ("ga", &with.attn_out, &without.attn_out),
    ] {
        assert_eq!(bits(&v(a)), bits(&v(b)), "{name}");
    }
    assert!(with
        .attn_out
        .all_finite_cached(|_| panic!("saved-scale grad_attn was scanned"))
        .unwrap());

    let mut raw = v(&scales);
    raw[0] = 2.0;
    raw[1] = -0.5;
    let mut clamped = raw.clone();
    clamped[0] = 1.0;
    clamped[1] = 0.0;
    let shape = scales.shape().to_vec();
    let outside = t(&raw, &shape);
    let inside = t(&clamped, &shape);
    let from_outside = fast
        .per_head_sigmoid_gate_backward_saved(&x, &w, &b, &attn, &gy, &outside)
        .unwrap();
    let from_inside = fast
        .per_head_sigmoid_gate_backward_saved(&x, &w, &b, &attn, &gy, &inside)
        .unwrap();
    assert_eq!(bits(&v(&from_outside.input)), bits(&v(&from_inside.input)));
    assert_eq!(
        bits(&v(&from_outside.attn_out)),
        bits(&v(&from_inside.attn_out))
    );

    let n = scales.num_elements().unwrap();
    let before = fast.budget().live_bytes().unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let scale = t(&vec![bad; n], &shape);
        assert_nonfinite(fast.per_head_sigmoid_gate_backward_saved(&x, &w, &b, &attn, &gy, &scale));
        assert_eq!(fast.budget().live_bytes().unwrap(), before, "{bad}");
    }
    let short = t(&[0.5], &[1]);
    let err = fast
        .per_head_sigmoid_gate_backward_saved(&x, &w, &b, &attn, &gy, &short)
        .unwrap_err();
    assert!(matches!(err, OjasError::Shape { .. }), "{err:?}");
    assert_eq!(fast.budget().live_bytes().unwrap(), before);

    let exact = backend(2, Numerics::Exact);
    let (ey, none) = exact
        .per_head_sigmoid_gate_forward_saving(&x, &w, &b, &attn)
        .unwrap();
    assert!(none.is_none(), "exact forward kept a scale");
    let ey0 = exact
        .per_head_sigmoid_gate_forward(&x, &w, &b, &attn)
        .unwrap();
    assert_eq!(bits(&v(&ey)), bits(&v(&ey0)));
    let junk = t(&vec![0.0; n], &shape);
    let ignored = exact
        .per_head_sigmoid_gate_backward_saved(&x, &w, &b, &attn, &gy, &junk)
        .unwrap();
    let recomputed = exact
        .per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &gy)
        .unwrap();
    assert_eq!(bits(&v(&ignored.input)), bits(&v(&recomputed.input)));
    assert_eq!(bits(&v(&ignored.weight)), bits(&v(&recomputed.weight)));
    assert_eq!(bits(&v(&ignored.bias)), bits(&v(&recomputed.bias)));
    assert_eq!(bits(&v(&ignored.attn_out)), bits(&v(&recomputed.attn_out)));
}
