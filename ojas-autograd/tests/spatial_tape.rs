//! `Tape::conv2d`, `Tape::max_pool2d` and `Tape::conv_transpose2d`: a small
//! encoder-decoder (conv with bias, pool, transposed conv without bias)
//! recorded on the tape must give the gradients of the backend's backward
//! calls chained by hand, bit for bit, recorded directly or inside a
//! checkpointed segment. The values of those backwards are checked against
//! f64 adjoints in `ojas-cpu/tests/spatial.rs`.

use ojas_autograd::{Tape, Var};
use ojas_core::{Backend, Budget, Conv2dParams, ConvTranspose2dParams, Numerics, Tensor};
use ojas_cpu::CpuBackend;

const N: usize = 2;
const C: usize = 3;
const HW: usize = 8;
const MID: usize = 4;
const OUT: usize = 2;
const CONV: Conv2dParams = Conv2dParams {
    stride: [1, 1],
    padding: [1, 1],
    dilation: [1, 1],
    groups: 1,
};
const UP: ConvTranspose2dParams = ConvTranspose2dParams {
    stride: [2, 2],
    padding: [0, 0],
    output_padding: [0, 0],
};

fn cpu() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 24), 2)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

/// Deterministic values in `[-1, 1)`.
fn vals(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            -1.0 + 2.0 * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn tensor(be: &CpuBackend, seed: u64, shape: &[usize]) -> Tensor {
    Tensor::from_f32(&vals(seed, shape.iter().product()), shape, be.budget()).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

struct Params {
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
}

fn params(be: &CpuBackend) -> Params {
    Params {
        x: tensor(be, 1, &[N, C, HW, HW]),
        w1: tensor(be, 2, &[MID, C, 3, 3]),
        b1: tensor(be, 3, &[MID]),
        w2: tensor(be, 4, &[MID, OUT, 2, 2]),
    }
}

/// conv 3x3 p1 (+ bias) -> max pool 2x2 -> transposed conv 2x2 s2.
fn net(tape: &mut Tape<CpuBackend>, v: [Var; 4]) -> Var {
    let [x, w1, b1, w2] = v;
    let h = tape.conv2d(x, w1, Some(b1), CONV).unwrap();
    let p = tape.max_pool2d(h, [2, 2], [2, 2]).unwrap();
    tape.conv_transpose2d(p, w2, None, UP).unwrap()
}

/// The same chain through the backend: `(dx, dw1, db1, dw2)` for a seed of
/// ones.
fn by_hand(be: &CpuBackend, p: &Params) -> [Tensor; 4] {
    let h = be.conv2d_forward(&p.x, &p.w1, Some(&p.b1), CONV).unwrap();
    let pooled = be.max_pool2d_forward(&h, [2, 2], [2, 2]).unwrap();
    let y = be
        .conv_transpose2d_forward(&pooled, &p.w2, None, UP)
        .unwrap();
    let n: usize = y.shape().iter().product();
    let ones = Tensor::from_f32(&vec![1.0; n], y.shape(), be.budget()).unwrap();
    let up = be
        .conv_transpose2d_backward(&pooled, &p.w2, None, &ones, UP)
        .unwrap();
    assert!(up.bias.is_none());
    let gh = be
        .max_pool2d_backward(&h, &up.input, [2, 2], [2, 2])
        .unwrap();
    let down = be
        .conv2d_backward(&p.x, &p.w1, Some(&p.b1), &gh, CONV)
        .unwrap();
    [down.input, down.weight, down.bias.unwrap(), up.weight]
}

fn leaves(tape: &mut Tape<CpuBackend>, p: &Params) -> [Var; 4] {
    [&p.x, &p.w1, &p.b1, &p.w2].map(|t| tape.leaf(t.clone()).unwrap())
}

#[test]
fn the_tape_chains_the_spatial_backwards() {
    let be = cpu();
    let p = params(&be);
    let want = by_hand(&be, &p);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &p);
    let y = net(&mut tape, v);
    assert_eq!(tape.value(y).unwrap().shape(), &[N, OUT, HW, HW]);
    tape.backward(y).unwrap();
    for (var, want) in v.iter().zip(&want) {
        let got = tape.grad(*var).expect("every leaf takes a gradient");
        assert_eq!(got.shape(), want.shape());
        assert_eq!(bits(got), bits(want));
    }
}

#[test]
fn a_checkpointed_spatial_segment_gives_the_same_bits() {
    let be = cpu();
    let p = params(&be);
    let want = by_hand(&be, &p);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &p);
    let out = tape.checkpoint(|t| Ok(vec![net(t, v)])).unwrap();
    tape.backward(out[0]).unwrap();
    for (var, want) in v.iter().zip(&want) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want));
    }
}

#[test]
fn a_frozen_image_takes_no_gradient_and_the_weights_still_do() {
    let be = cpu();
    let p = params(&be);
    let want = by_hand(&be, &p);
    let mut tape = Tape::new(be.clone());
    let x = tape.frozen_leaf(p.x.clone()).unwrap();
    let [w1, b1, w2] = [&p.w1, &p.b1, &p.w2].map(|t| tape.leaf(t.clone()).unwrap());
    let y = net(&mut tape, [x, w1, b1, w2]);
    tape.backward(y).unwrap();
    assert!(tape.grad(x).is_none());
    for (var, want) in [w1, b1, w2].iter().zip(&want[1..]) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want));
    }
}
