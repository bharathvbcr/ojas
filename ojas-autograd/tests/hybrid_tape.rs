//! Qwen3.5's hybrid-layer ops on the tape: `causal_conv1d_silu`,
//! `gated_rms_norm` and `rope_partial` record their forward, and backward
//! hands every input exactly the gradient the backend's backward chain
//! gives, bit for bit, directly and inside a checkpointed segment. The ops'
//! own numerics are checked in `ojas-cpu/tests/hybrid.rs`.

use ojas_autograd::{Tape, Var};
use ojas_core::{mrope_text_tables, Backend, Budget, MropeSection, Numerics, Tensor};
use ojas_cpu::CpuBackend;

const B: usize = 2;
const T: usize = 9;
const H: usize = 2;
const D: usize = 8;
const C: usize = H * D;
const K: usize = 4;
const R: usize = 4;
const EPS: f32 = 1e-6;

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact)
}

fn vals(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

struct Inputs {
    x: Tensor,
    w: Tensor,
    z: Tensor,
    nw: Tensor,
    r: Tensor,
    cos: Tensor,
    sin: Tensor,
}

fn inputs(be: &CpuBackend) -> Inputs {
    let b = be.budget();
    let f = |seed, n, lo, hi, shape: &[usize]| {
        Tensor::from_f32(&vals(seed, n, lo, hi), shape, b).unwrap()
    };
    let qwen = MropeSection {
        section: [1, 1, 0],
        interleaved: true,
    };
    let (cos, sin) = mrope_text_tables(0, T, qwen, R, 1e7, b).unwrap();
    Inputs {
        x: f(1, B * T * C, -2.0, 2.0, &[B, T, C]),
        w: f(2, C * K, -1.0, 1.0, &[C, K]),
        z: f(3, B * T * C, -2.0, 2.0, &[B, T, C]),
        nw: f(4, D, 0.5, 1.5, &[D]),
        r: f(5, B * T * C, -1.0, 1.0, &[B, T, H, D]),
        cos,
        sin,
    }
}

/// conv1d + SiLU, gated RMSNorm per head, partial RoPE, then `* r`.
fn hybrid(t: &mut Tape, v: [Var; 4], i: &Inputs) -> Result<Var, ojas_core::OjasError> {
    let [x, w, z, nw] = v;
    let y = t.causal_conv1d_silu(x, w)?;
    let y = t.reshape(y, &[B, T, H, D])?;
    let z = t.reshape(z, &[B, T, H, D])?;
    let y = t.gated_rms_norm(y, z, nw, EPS)?;
    t.rope_partial(y, i.cos.clone(), i.sin.clone())
}

fn tape_grads(checkpoint: bool) -> [Vec<u32>; 4] {
    let be = cpu();
    let i = inputs(&be);
    let mut t = Tape::new(be);
    let v = [
        t.leaf(i.x.clone()).unwrap(),
        t.leaf(i.w.clone()).unwrap(),
        t.leaf(i.z.clone()).unwrap(),
        t.leaf(i.nw.clone()).unwrap(),
    ];
    let r = t.leaf(i.r.clone()).unwrap();
    let y = if checkpoint {
        t.checkpoint(|t| hybrid(t, v, &i).map(|y| vec![y])).unwrap()[0]
    } else {
        hybrid(&mut t, v, &i).unwrap()
    };
    let loss = t.mul(y, r).unwrap();
    t.backward(loss).unwrap();
    v.map(|v| bits(t.grad(v).unwrap()))
}

#[test]
fn tape_gradients_are_the_backend_backward_chain() {
    let be = cpu();
    let i = inputs(&be);
    // Forward, as the tape records it.
    let y1 = be.causal_conv1d_silu_forward(&i.x, &i.w).unwrap();
    let y1h = y1.reshape(&[B, T, H, D]).unwrap();
    let zh = i.z.reshape(&[B, T, H, D]).unwrap();
    // d(y * r)/dy is r; then back through each op.
    let g_rope = be.rope_partial_backward(&i.r, &i.cos, &i.sin).unwrap();
    let g = be
        .gated_rms_norm_backward(&y1h, &zh, &i.nw, &g_rope, EPS)
        .unwrap();
    let gy1 = g.input.reshape(&[B, T, C]).unwrap();
    let (gx, gw) = be.causal_conv1d_silu_backward(&i.x, &i.w, &gy1).unwrap();
    let want = [
        bits(&gx),
        bits(&gw),
        bits(&g.gate.reshape(&[B, T, C]).unwrap()),
        bits(&g.weight),
    ];
    assert_eq!(tape_grads(false), want);
}

#[test]
fn inside_a_checkpointed_segment_the_gradients_are_the_same_bits() {
    assert_eq!(tape_grads(true), tape_grads(false));
}
