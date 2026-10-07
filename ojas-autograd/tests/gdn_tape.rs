//! `Tape::chunked_gdn`: the record keeps the forward's checkpoints and its
//! backward hands every input the gradient `Backend::chunked_gdn_backward`
//! gives for an all-ones output seed. The op's own numerics are checked in
//! `ojas-oracle/tests/gdn.rs` (published goldens and central differences).

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, GdnInputs, Numerics, Tensor};
use ojas_cpu::CpuBackend;

const B: usize = 1;
const T: usize = 70;
const H: usize = 2;
const DK: usize = 4;
const DV: usize = 3;

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact)
}

/// Deterministic values in `[lo, hi)`.
fn vals(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8;
            lo + (hi - lo) * (x as f32 / (1u32 << 24) as f32)
        })
        .collect()
}

fn t(b: &CpuBackend, data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_f32(&data, shape, b.budget()).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

struct Ops {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
}

fn ops(b: &CpuBackend) -> Ops {
    Ops {
        q: t(b, vals(B * T * H * DK, 1, -1.0, 1.0), &[B, T, H, DK]),
        k: t(b, vals(B * T * H * DK, 2, -1.0, 1.0), &[B, T, H, DK]),
        v: t(b, vals(B * T * H * DV, 3, -1.0, 1.0), &[B, T, H, DV]),
        g: t(b, vals(B * T * H, 4, -1.0, -0.05), &[B, T, H]),
        beta: t(b, vals(B * T * H, 5, 0.1, 0.9), &[B, T, H]),
    }
}

#[test]
fn tape_gradients_are_the_backend_backward_of_an_all_ones_seed() {
    let b = cpu();
    let x = ops(&b);
    let mut tape = Tape::new(b.clone());
    let (q, k, v, g, beta) = (
        tape.leaf(x.q.clone()).unwrap(),
        tape.leaf(x.k.clone()).unwrap(),
        tape.leaf(x.v.clone()).unwrap(),
        tape.leaf(x.g.clone()).unwrap(),
        tape.leaf(x.beta.clone()).unwrap(),
    );
    let y = tape.chunked_gdn(q, k, v, g, beta).unwrap();
    tape.backward(y).unwrap();

    let inputs = GdnInputs {
        q: &x.q,
        k: &x.k,
        v: &x.v,
        g: &x.g,
        beta: &x.beta,
        initial_state: None,
    };
    let fwd = b.chunked_gdn_forward(inputs).unwrap();
    assert_eq!(bits(tape.value(y).unwrap()), bits(&fwd.output));
    let ones = t(&b, vec![1.0; B * T * H * DV], &[B, T, H, DV]);
    let want = b
        .chunked_gdn_backward(inputs, &fwd.checkpoints, &ones, None)
        .unwrap();
    for (name, var, w) in [
        ("q", q, &want.q),
        ("k", k, &want.k),
        ("v", v, &want.v),
        ("g", g, &want.g),
        ("beta", beta, &want.beta),
    ] {
        assert_eq!(bits(tape.grad(var).unwrap()), bits(w), "d{name}");
    }
    assert!(want.initial_state.is_none());
}

/// One variable used as both `q` and `k` receives the sum of both
/// gradients, as any shared input does on the tape.
#[test]
fn a_variable_used_as_q_and_k_gets_both_gradients() {
    let b = cpu();
    let x = ops(&b);
    let mut tape = Tape::new(b.clone());
    let qk = tape.leaf(x.q.clone()).unwrap();
    let (v, g, beta) = (
        tape.leaf(x.v.clone()).unwrap(),
        tape.leaf(x.g.clone()).unwrap(),
        tape.leaf(x.beta.clone()).unwrap(),
    );
    let y = tape.chunked_gdn(qk, qk, v, g, beta).unwrap();
    tape.backward(y).unwrap();

    let inputs = GdnInputs {
        q: &x.q,
        k: &x.q,
        v: &x.v,
        g: &x.g,
        beta: &x.beta,
        initial_state: None,
    };
    let fwd = b.chunked_gdn_forward(inputs).unwrap();
    let ones = t(&b, vec![1.0; B * T * H * DV], &[B, T, H, DV]);
    let want = b
        .chunked_gdn_backward(inputs, &fwd.checkpoints, &ones, None)
        .unwrap();
    let sum = b.residual_add_forward(&want.q, &want.k).unwrap();
    // The record hands `q`'s gradient over first, then `k`'s is added to it.
    assert_eq!(bits(tape.grad(qk).unwrap()), bits(&sum));
}
