//! Tape `permute` and a multi-head attention block built on it (audit F1).
//!
//! RoPE and the per-head gate take `[B, T, H, D]`; causal attention takes
//! `[B, H, T, D]`. With `H > 1` those are different memory orders, so a
//! reshape cannot connect them and a permute must. The block below is
//! gradchecked against an f64 forward that indexes `[B, T, H, D]` directly,
//! with no permute of its own, so a wrong axis order fails the comparison.
//! Same harness as `gradcheck.rs`: `central_diff` and `gradients_match`.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{inverse_permutation, Backend, Budget, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const H_STEP: f64 = 1e-3;
const ATOL: f64 = 2e-3;
const RTOL: f64 = 2e-2;

const B: usize = 2;
const T: usize = 5;
const NH: usize = 3;
const D: usize = 4;
const C: usize = NH * D;
const BTHD: [usize; 4] = [B, T, NH, D];
/// `[B, T, H, D]` to `[B, H, T, D]`; it is its own inverse.
const SWAP_TH: [usize; 4] = [0, 2, 1, 3];

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 24))
}

fn tensor(cpu: &CpuBackend, data: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    Tensor::from_f32(&f, shape, cpu.budget()).unwrap()
}

fn check(name: &str, analytic: &[f32], numeric: &[f64]) {
    gradients_match(analytic, numeric, ATOL, RTOL).unwrap_or_else(|e| panic!("{name}: {e}"));
}

/// Deterministic values in `[-scale, scale)` with no dependency.
fn values(seed: u64, n: usize, scale: f64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            scale * (2.0 * unit - 1.0)
        })
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn at(b: usize, t: usize, h: usize, d: usize) -> usize {
    ((b * T + t) * NH + h) * D + d
}

/// Half-split RoPE on `[B, T, H, D]` with `cos`/`sin` of shape `[T, D]`.
fn rope_ref(x: &[f64], cos: &[f64], sin: &[f64]) -> Vec<f64> {
    let half = D / 2;
    let mut y = vec![0.0; x.len()];
    for b in 0..B {
        for t in 0..T {
            for h in 0..NH {
                for i in 0..half {
                    let lo = x[at(b, t, h, i)];
                    let hi = x[at(b, t, h, i + half)];
                    y[at(b, t, h, i)] = lo * cos[t * D + i] - hi * sin[t * D + i];
                    y[at(b, t, h, i + half)] =
                        hi * cos[t * D + i + half] + lo * sin[t * D + i + half];
                }
            }
        }
    }
    y
}

struct Inputs {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    x: Vec<f64>,
    w: Vec<f64>,
    bias: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    weight: Vec<f64>,
}

fn inputs() -> Inputs {
    let n = B * T * NH * D;
    let mut cos = vec![0.0; T * D];
    let mut sin = vec![0.0; T * D];
    for t in 0..T {
        for i in 0..D / 2 {
            let freq = 1.0 / 10000f64.powf((2 * i) as f64 / D as f64);
            let angle = t as f64 * freq;
            for slot in [i, i + D / 2] {
                cos[t * D + slot] = angle.cos();
                sin[t * D + slot] = angle.sin();
            }
        }
    }
    Inputs {
        q: values(1, n, 0.8),
        k: values(2, n, 0.8),
        v: values(3, n, 1.0),
        x: values(4, B * T * C, 1.0),
        w: values(5, NH * C, 0.5),
        bias: values(6, NH, 0.3),
        cos,
        sin,
        weight: values(7, n, 1.0),
    }
}

/// The block's output, `[B, T, H, D]`, in f64 with no permute: attention
/// for head `h` reads `q[b, t, h, :]` and `k[b, j, h, :]` in place.
fn block_ref(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    x: &[f64],
    w: &[f64],
    bias: &[f64],
    inp: &Inputs,
) -> Vec<f64> {
    let qr = rope_ref(q, &inp.cos, &inp.sin);
    let kr = rope_ref(k, &inp.cos, &inp.sin);
    let scale = 1.0 / (D as f64).sqrt();
    let mut out = vec![0.0; q.len()];
    for b in 0..B {
        for h in 0..NH {
            for t in 0..T {
                let scores: Vec<f64> = (0..=t)
                    .map(|j| {
                        (0..D)
                            .map(|d| qr[at(b, t, h, d)] * kr[at(b, j, h, d)])
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let z: f64 = exps.iter().sum();
                let mut zg = bias[h];
                for c in 0..C {
                    zg += x[(b * T + t) * C + c] * w[h * C + c];
                }
                let gate = sigmoid(zg);
                for d in 0..D {
                    let attn: f64 = (0..=t).map(|j| exps[j] / z * v[at(b, j, h, d)]).sum();
                    out[at(b, t, h, d)] = attn * gate;
                }
            }
        }
    }
    out
}

fn loss_ref(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    x: &[f64],
    w: &[f64],
    bias: &[f64],
    inp: &Inputs,
) -> f64 {
    block_ref(q, k, v, x, w, bias, inp)
        .iter()
        .zip(&inp.weight)
        .map(|(o, r)| o * r)
        .sum()
}

struct Block {
    tape: Tape,
    q: Var,
    k: Var,
    v: Var,
    x: Var,
    w: Var,
    bias: Var,
    out: Var,
    loss: Var,
}

/// q, k `[B,T,H,D]` → RoPE → permute to `[B,H,T,D]` → causal SDPA →
/// permute back → per-head gate. `loss` is `out * weight`, whose sum is the
/// scalar the backward seed of ones differentiates.
fn block(inp: &Inputs) -> Block {
    let cpu = cpu();
    let mut tape = Tape::new(cpu.clone());
    let q = tape.leaf(tensor(&cpu, &inp.q, &BTHD)).unwrap();
    let k = tape.leaf(tensor(&cpu, &inp.k, &BTHD)).unwrap();
    let v = tape.leaf(tensor(&cpu, &inp.v, &BTHD)).unwrap();
    let x = tape.leaf(tensor(&cpu, &inp.x, &[B, T, C])).unwrap();
    let w = tape.leaf(tensor(&cpu, &inp.w, &[NH, C])).unwrap();
    let bias = tape.leaf(tensor(&cpu, &inp.bias, &[NH])).unwrap();
    let weight = tape.leaf(tensor(&cpu, &inp.weight, &BTHD)).unwrap();
    let cos = tensor(&cpu, &inp.cos, &[T, D]);
    let sin = tensor(&cpu, &inp.sin, &[T, D]);
    let qr = tape.rope(q, cos.clone(), sin.clone()).unwrap();
    let kr = tape.rope(k, cos, sin).unwrap();
    let qh = tape.permute(qr, &SWAP_TH).unwrap();
    let kh = tape.permute(kr, &SWAP_TH).unwrap();
    let vh = tape.permute(v, &SWAP_TH).unwrap();
    let y = tape.causal_sdpa(qh, kh, vh, None).unwrap();
    let y = tape.permute(y, &SWAP_TH).unwrap();
    let out = tape.per_head_gate(x, w, bias, y).unwrap();
    let loss = tape.mul(out, weight).unwrap();
    Block {
        tape,
        q,
        k,
        v,
        x,
        w,
        bias,
        out,
        loss,
    }
}

fn grad(block: &Block, var: Var) -> Vec<f32> {
    block.tape.grad(var).unwrap().to_f32_vec().unwrap()
}

#[test]
fn multi_head_block_forward_matches_the_scalar_reference() {
    let inp = inputs();
    let blk = block(&inp);
    let got = blk.tape.value(blk.out).unwrap();
    assert_eq!(got.shape(), &BTHD);
    let got = got.to_f32_vec().unwrap();
    let want = block_ref(&inp.q, &inp.k, &inp.v, &inp.x, &inp.w, &inp.bias, &inp);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let diff = (f64::from(*g) - w).abs();
        assert!(
            diff <= 1e-5 + 1e-5 * w.abs(),
            "out[{i}]: tape {g} ref {w} diff {diff}"
        );
    }
}

#[test]
fn multi_head_block_gradients_match_central_differences() {
    let inp = inputs();
    let mut blk = block(&inp);
    let loss = blk.loss;
    blk.tape.backward(loss).unwrap();
    let nq = central_diff(&inp.q, H_STEP, |p| {
        Ok(loss_ref(p, &inp.k, &inp.v, &inp.x, &inp.w, &inp.bias, &inp))
    })
    .unwrap();
    let nk = central_diff(&inp.k, H_STEP, |p| {
        Ok(loss_ref(&inp.q, p, &inp.v, &inp.x, &inp.w, &inp.bias, &inp))
    })
    .unwrap();
    let nv = central_diff(&inp.v, H_STEP, |p| {
        Ok(loss_ref(&inp.q, &inp.k, p, &inp.x, &inp.w, &inp.bias, &inp))
    })
    .unwrap();
    let nx = central_diff(&inp.x, H_STEP, |p| {
        Ok(loss_ref(&inp.q, &inp.k, &inp.v, p, &inp.w, &inp.bias, &inp))
    })
    .unwrap();
    let nw = central_diff(&inp.w, H_STEP, |p| {
        Ok(loss_ref(&inp.q, &inp.k, &inp.v, &inp.x, p, &inp.bias, &inp))
    })
    .unwrap();
    let nb = central_diff(&inp.bias, H_STEP, |p| {
        Ok(loss_ref(&inp.q, &inp.k, &inp.v, &inp.x, &inp.w, p, &inp))
    })
    .unwrap();
    check("q", &grad(&blk, blk.q), &nq);
    check("k", &grad(&blk, blk.k), &nk);
    check("v", &grad(&blk, blk.v), &nv);
    check("x", &grad(&blk, blk.x), &nx);
    check("gate weight", &grad(&blk, blk.w), &nw);
    check("gate bias", &grad(&blk, blk.bias), &nb);
}

/// A bare permute has a gradient of ones whatever the axis order, so the
/// permuted value is multiplied by a fixed non-uniform weight: the gradient
/// is then that weight permuted back, exactly.
#[test]
fn permute_gradient_is_the_upstream_gradient_permuted_back() {
    let cpu = cpu();
    let shape = [2usize, 3, 4];
    let dims = [2usize, 0, 1];
    let out_shape = [4usize, 2, 3];
    let x0 = values(11, 24, 1.0);
    let r0 = values(12, 24, 2.0);
    let mut tape = Tape::new(cpu.clone());
    let x = tape.leaf(tensor(&cpu, &x0, &shape)).unwrap();
    let r = tape.leaf(tensor(&cpu, &r0, &out_shape)).unwrap();
    let y = tape.permute(x, &dims).unwrap();
    assert_eq!(tape.value(y).unwrap().shape(), &out_shape);
    let z = tape.mul(y, r).unwrap();
    tape.backward(z).unwrap();
    let gx = tape.grad(x).unwrap();
    assert_eq!(gx.shape(), &shape);
    // Output element (a, b, c) is input element (b, c, a).
    let sum_ref = |p: &[f64]| {
        let mut s = 0.0;
        for a in 0..4 {
            for b in 0..2 {
                for c in 0..3 {
                    s += p[(b * 3 + c) * 4 + a] * r0[(a * 2 + b) * 3 + c];
                }
            }
        }
        s
    };
    let numeric = central_diff(&x0, H_STEP, |p| Ok(sum_ref(p))).unwrap();
    check("permute", &gx.to_f32_vec().unwrap(), &numeric);
    // d(y * r)/dy is r exactly, so the gradient is r permuted back, bit for bit.
    let want = tape
        .backend()
        .permute(tape.value(r).unwrap(), &inverse_permutation(&dims))
        .unwrap();
    let to_bits = |t: &Tensor| -> Vec<u32> {
        t.to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(to_bits(gx), to_bits(&want));
}

#[test]
fn permute_on_the_tape_refuses_bad_axes_and_records_nothing() {
    let cpu = cpu();
    let mut tape = Tape::new(cpu.clone());
    let x = tape.leaf(tensor(&cpu, &[0.5; 6], &[2, 3])).unwrap();
    for dims in [&[0usize][..], &[0, 0], &[0, 2], &[1, 0, 2]] {
        match tape.permute(x, dims) {
            Err(OjasError::Shape { .. }) => {}
            other => panic!("{dims:?}: expected Shape, got {other:?}"),
        }
    }
    // Only the leaf is on the tape: the next var is index 1.
    let y = tape.permute(x, &[1, 0]).unwrap();
    assert_eq!(y, Var(1));
    assert!(tape.permute(Var(9), &[1, 0]).is_err());
}
