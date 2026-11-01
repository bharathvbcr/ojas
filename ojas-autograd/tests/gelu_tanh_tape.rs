//! `Tape::gelu_tanh`: the tanh-approximate GELU on the tape.
//!
//! - A linear layer -> GELU(tanh) -> weighting recorded on the tape gives the
//!   gradients of the backend's backward calls chained by hand, bit for bit,
//!   recorded directly or inside a checkpointed segment.
//! - The same gradients are held to central differences of an `f64` forward
//!   written with libm `tanh` (torch's formula), so a gradient routed to the
//!   wrong input, or a derivative of the wrong function, fails here.
//! - A variable that is not on the tape is refused and pushes nothing.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const TOKENS: usize = 3;
const IN: usize = 6;
const OUT: usize = 5;
const NAMES: [&str; 4] = ["x", "w", "b", "r"];

fn cpu() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 24), 2)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

/// Deterministic values in `[-amp, amp)`.
fn vals(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (-1.0 + 2.0 * ((s >> 40) as f32 / (1u64 << 24) as f32)) * amp
        })
        .collect()
}

fn tensor(be: &CpuBackend, seed: u64, shape: &[usize], amp: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_f32(&vals(seed, n, amp), shape, be.budget()).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `x`, `w`, `b`, `r`. The pre-activations reach about `+-3`, where the tanh
/// and erf forms differ and neither tail is flat.
fn leaves_data(be: &CpuBackend) -> [Tensor; 4] {
    [
        tensor(be, 1, &[2, TOKENS, IN], 2.0),
        tensor(be, 2, &[OUT, IN], 1.0),
        tensor(be, 3, &[OUT], 1.0),
        tensor(be, 4, &[2, TOKENS, OUT], 1.0),
    ]
}

fn net<B: Backend>(tape: &mut Tape<B>, v: &[Var; 4]) -> Var {
    let [x, w, b, r] = *v;
    let h = tape.linear_bias(x, w, b).unwrap();
    let h = tape.gelu_tanh(h).unwrap();
    tape.mul(h, r).unwrap()
}

/// The same chain through the backend: the gradient of every leaf for a seed
/// of ones.
fn by_hand(be: &CpuBackend, k: &[Tensor; 4]) -> [Tensor; 4] {
    let [x, w, b, r] = k;
    let h1 = be.linear_bias_forward(x, w, b).unwrap();
    let h2 = be.gelu_tanh_forward(&h1).unwrap();
    let ones = Tensor::from_f32(
        &vec![1.0; h2.shape().iter().product()],
        h2.shape(),
        be.budget(),
    )
    .unwrap();
    let (gh2, gr) = be.mul_backward(&h2, r, &ones).unwrap();
    let g1 = be.gelu_tanh_backward(&h1, &gh2).unwrap();
    let lin = be.linear_bias_backward(x, w, b, &g1).unwrap();
    [lin.input, lin.weight, lin.bias, gr]
}

fn leaves<B: Backend>(tape: &mut Tape<B>, k: &[Tensor; 4]) -> [Var; 4] {
    std::array::from_fn(|i| tape.leaf(k[i].clone()).unwrap())
}

#[test]
fn the_tape_chains_the_gelu_tanh_backward() {
    let be = cpu();
    let k = leaves_data(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v);
    assert_eq!(tape.value(z).unwrap().shape(), &[2, TOKENS, OUT]);
    tape.backward(z).unwrap();
    for ((var, want), name) in v.iter().zip(&want).zip(NAMES) {
        let got = tape
            .grad(*var)
            .unwrap_or_else(|| panic!("{name} takes a gradient"));
        assert_eq!(got.shape(), want.shape(), "{name}");
        assert_eq!(bits(got), bits(want), "{name}");
    }
}

#[test]
fn a_checkpointed_gelu_tanh_segment_gives_the_same_bits() {
    let be = cpu();
    let k = leaves_data(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let out = tape.checkpoint(|t| Ok(vec![net(t, &v)])).unwrap();
    tape.backward(out[0]).unwrap();
    for ((var, want), name) in v.iter().zip(&want).zip(NAMES) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want), "{name}");
    }
}

/// `sum(gelu_tanh(x @ w^T + b) * r)` in `f64` with libm `tanh`.
fn net_f64(p: &[f64]) -> f64 {
    let sizes = [2 * TOKENS * IN, OUT * IN, OUT, 2 * TOKENS * OUT];
    let mut at = 0;
    let [x, w, b, r]: [&[f64]; 4] = std::array::from_fn(|i| {
        let part = &p[at..at + sizes[i]];
        at += sizes[i];
        part
    });
    let kappa = (2.0 / std::f64::consts::PI).sqrt();
    let mut total = 0.0;
    for row in 0..2 * TOKENS {
        for o in 0..OUT {
            let dot: f64 = (0..IN).map(|i| x[row * IN + i] * w[o * IN + i]).sum();
            let z = dot + b[o];
            let gelu = 0.5 * z * (1.0 + (kappa * (z + 0.044715 * z * z * z)).tanh());
            total += gelu * r[row * OUT + o];
        }
    }
    total
}

#[test]
fn the_tapes_gelu_tanh_gradients_match_central_differences() {
    let be = cpu();
    let k = leaves_data(&be);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v);
    tape.backward(z).unwrap();
    let flat: Vec<f64> = k
        .iter()
        .flat_map(|t| t.to_f32_vec().unwrap())
        .map(f64::from)
        .collect();
    let numeric = central_diff(&flat, 1e-6, |p| Ok(net_f64(p))).unwrap();
    let mut at = 0;
    for (var, name) in v.iter().zip(NAMES) {
        let got = tape.grad(*var).unwrap().to_f32_vec().unwrap();
        gradients_match(&got, &numeric[at..at + got.len()], 2e-4, 2e-3)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        at += got.len();
    }
}

#[test]
fn a_variable_off_the_tape_is_refused_and_pushes_nothing() {
    let be = cpu();
    let mut tape = Tape::new(be.clone());
    let a = tape.leaf(tensor(&be, 41, &[2, 3], 1.0)).unwrap();
    match tape.gelu_tanh(Var(99)) {
        Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, "Tape::gelu_tanh"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
    assert!(
        tape.value(Var(1)).is_err(),
        "a refused record pushed a node"
    );
    // The next record still lands where it should.
    assert_eq!(tape.gelu_tanh(a).unwrap(), Var(1));
}
