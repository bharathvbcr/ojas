//! Grouped-query attention (H != Hkv) through the Tape, checked by central
//! differences of an f64 oracle.
//!
//! The graph is `x [B, T, C]` projected by `wq [H*D, C]`, `wk`, `wv
//! [Hkv*D, C]`, reshaped to `[B, T, H, D]` and permuted to `[B, H, T, D]`,
//! causal attention (whole prefix, or a sliding window), and the loss
//! `sum(r * y)` for a fixed `r`. The f64 oracle computes the same loss from
//! the same values; its central differences are the reference for every
//! leaf's tape gradient: on `CpuBackend` (Exact), and on Metal and wgpu,
//! which are also held to the CPU tape's gradients.
//!
//! A missing device fails unless `OJAS_ALLOW_NO_GPU=1`.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Backend, BackendId, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const B: usize = 2;
const T: usize = 7;
const C: usize = 5;
const H: usize = 4;
const HKV: usize = 2;
const D: usize = 3;
/// Whole prefix, a window inside one row block, and one past `T` (the
/// whole prefix again).
const WINDOWS: [Option<usize>; 4] = [None, Some(1), Some(3), Some(T + 2)];
const STEP: f64 = 1e-4;
/// The tape's f32 gradients against f64 central differences.
const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-3;
/// A device tape's gradients against the CPU tape's.
const DEVICE_TOL: f32 = 2e-4;
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

/// Deterministic values in `[-amp, amp)`.
fn values(n: usize, seed: u64, amp: f64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// The leaves: `x`, `wq`, `wk`, `wv`, and the loss weights `r`.
struct Inputs {
    x: Vec<f64>,
    wq: Vec<f64>,
    wk: Vec<f64>,
    wv: Vec<f64>,
    r: Vec<f64>,
}

impl Inputs {
    fn new() -> Self {
        Self {
            x: values(B * T * C, 1, 1.0),
            wq: values(H * D * C, 2, 0.8),
            wk: values(HKV * D * C, 3, 0.8),
            wv: values(HKV * D * C, 4, 0.8),
            r: values(B * H * T * D, 5, 1.0),
        }
    }

    /// The same values rounded to `f32` and back, so the oracle sees what
    /// the tape sees.
    fn rounded(&self) -> Self {
        let r = |v: &[f64]| v.iter().map(|&x| f64::from(x as f32)).collect();
        Self {
            x: r(&self.x),
            wq: r(&self.wq),
            wk: r(&self.wk),
            wv: r(&self.wv),
            r: r(&self.r),
        }
    }
}

/// `x [B, T, C]` times `w [heads*D, C]`ᵀ, as `[B, heads, T, D]`.
fn project(x: &[f64], w: &[f64], heads: usize) -> Vec<f64> {
    let mut out = vec![0.0; B * heads * T * D];
    for b in 0..B {
        for t in 0..T {
            for h in 0..heads {
                for d in 0..D {
                    let row = h * D + d;
                    let dot: f64 = (0..C)
                        .map(|c| x[(b * T + t) * C + c] * w[row * C + c])
                        .sum();
                    out[((b * heads + h) * T + t) * D + d] = dot;
                }
            }
        }
    }
    out
}

/// The f64 loss `sum(r * attention(q, k, v))`.
fn oracle(i: &Inputs, window: Option<usize>) -> f64 {
    let q = project(&i.x, &i.wq, H);
    let k = project(&i.x, &i.wk, HKV);
    let v = project(&i.x, &i.wv, HKV);
    let scale = 1.0 / (D as f64).sqrt();
    let rep = H / HKV;
    let mut loss = 0.0;
    for b in 0..B {
        for h in 0..H {
            let kvh = b * HKV + h / rep;
            for t in 0..T {
                let first = window.map_or(0, |w| (t + 1).saturating_sub(w));
                let qr = &q[((b * H + h) * T + t) * D..][..D];
                let s: Vec<f64> = (first..=t)
                    .map(|j| {
                        let kr = &k[(kvh * T + j) * D..][..D];
                        qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f64>() * scale
                    })
                    .collect();
                let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - max).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..D {
                    let y: f64 = (first..=t)
                        .zip(&e)
                        .map(|(j, p)| p / z * v[(kvh * T + j) * D + d])
                        .sum();
                    loss += i.r[((b * H + h) * T + t) * D + d] * y;
                }
            }
        }
    }
    loss
}

/// Central differences of the oracle for every leaf: `[x, wq, wk, wv]`.
fn numeric(i: &Inputs, window: Option<usize>) -> [Vec<f64>; 4] {
    let with = |which: usize, p: &[f64]| {
        let mut j = Inputs {
            x: i.x.clone(),
            wq: i.wq.clone(),
            wk: i.wk.clone(),
            wv: i.wv.clone(),
            r: i.r.clone(),
        };
        match which {
            0 => j.x = p.to_vec(),
            1 => j.wq = p.to_vec(),
            2 => j.wk = p.to_vec(),
            _ => j.wv = p.to_vec(),
        }
        Ok(oracle(&j, window))
    };
    [&i.x, &i.wq, &i.wk, &i.wv]
        .into_iter()
        .enumerate()
        .map(|(which, v)| central_diff(v, STEP, |p| with(which, p)).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn host(v: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::from_f32(&f, shape, &Budget::new(1 << 24)).unwrap()
}

/// `[B, T, heads*D]` as `[B, heads, T, D]` on the tape.
fn heads<Bk: Backend>(tape: &mut Tape<Bk>, y: Var, heads: usize) -> Result<Var, OjasError> {
    let y = tape.reshape(y, &[B, T, heads, D])?;
    tape.permute(y, &[0, 2, 1, 3])
}

/// The tape's gradients of `[x, wq, wk, wv]`, read back to the host.
fn tape_grads<Bk: Backend>(
    backend: Bk,
    i: &Inputs,
    window: Option<usize>,
) -> Result<[Vec<f32>; 4], OjasError> {
    let mut tape = Tape::new(backend);
    let x = tape.leaf(host(&i.x, &[B, T, C]))?;
    let wq = tape.leaf(host(&i.wq, &[H * D, C]))?;
    let wk = tape.leaf(host(&i.wk, &[HKV * D, C]))?;
    let wv = tape.leaf(host(&i.wv, &[HKV * D, C]))?;
    let r = tape.leaf(host(&i.r, &[B, H, T, D]))?;
    let q = tape.linear(x, wq)?;
    let q = heads(&mut tape, q, H)?;
    let k = tape.linear(x, wk)?;
    let k = heads(&mut tape, k, HKV)?;
    let v = tape.linear(x, wv)?;
    let v = heads(&mut tape, v, HKV)?;
    let y = tape.causal_sdpa(q, k, v, window)?;
    let loss = tape.mul(y, r)?;
    tape.backward(loss)?;
    let mut out: Vec<Vec<f32>> = Vec::new();
    for var in [x, wq, wk, wv] {
        let g = tape.grad(var).ok_or(OjasError::OutOfRange {
            op: "gqa_tape",
            detail: "leaf has no gradient".to_string(),
        })?;
        let g = if tape.backend().id() == BackendId::Cpu {
            g.clone()
        } else {
            tape.backend().download(g)?
        };
        out.push(g.to_f32_vec()?);
    }
    Ok(out.try_into().unwrap())
}

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact)
}

const NAMES: [&str; 4] = ["x", "wq", "wk", "wv"];

/// The CPU tape's gradients of every leaf against the oracle's central
/// differences, for each window.
#[test]
fn cpu_tape_gqa_matches_central_differences() {
    let i = Inputs::new().rounded();
    for window in WINDOWS {
        let want = numeric(&i, window);
        let got = tape_grads(cpu(), &i, window).unwrap();
        for ((g, n), name) in got.iter().zip(&want).zip(NAMES) {
            gradients_match(g, n, ATOL, RTOL)
                .unwrap_or_else(|e| panic!("cpu window {window:?} {name}: {e}"));
        }
    }
}

/// The tape on `backend` against the oracle and against the CPU tape.
fn device_matches<Bk: Backend>(name: &str, backend: &Bk) {
    let i = Inputs::new().rounded();
    for window in WINDOWS {
        let want = numeric(&i, window);
        let reference = tape_grads(cpu(), &i, window).unwrap();
        let got = tape_grads(backend, &i, window).unwrap();
        for (((g, n), c), leaf) in got.iter().zip(&want).zip(&reference).zip(NAMES) {
            gradients_match(g, n, ATOL, RTOL)
                .unwrap_or_else(|e| panic!("{name} window {window:?} {leaf} vs fd: {e}"));
            for (j, (a, b)) in g.iter().zip(c).enumerate() {
                assert!(
                    (a - b).abs() <= DEVICE_TOL * (1.0 + b.abs()),
                    "{name} window {window:?} {leaf}[{j}]: {a} vs cpu {b}"
                );
            }
        }
    }
}

fn skip_or_fail(what: &str, err: &dyn std::fmt::Display) {
    if std::env::var(ALLOW_NO_GPU).as_deref() == Ok("1") {
        eprintln!("SKIP ({ALLOW_NO_GPU}=1): {what}: {err}");
    } else {
        panic!("{what} failed: {err}. Set {ALLOW_NO_GPU}=1 to skip explicitly");
    }
}

#[test]
fn wgpu_tape_gqa_matches_central_differences_and_cpu() {
    match ojas_wgpu::WgpuBackend::open(Budget::new(1 << 30)) {
        Ok(gpu) => device_matches("wgpu", &gpu),
        Err(err) => skip_or_fail("WgpuBackend::open", &err),
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tape_gqa_matches_central_differences_and_cpu() {
    match ojas_metal::MetalBackend::new(Budget::new(1 << 30)) {
        Ok(gpu) => device_matches("metal", &gpu),
        Err(err) => skip_or_fail("MetalBackend::new", &err),
    }
}
