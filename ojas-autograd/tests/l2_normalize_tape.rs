//! `Tape::l2_normalize`: `x / max(||x||, eps)` over the last axis, on the tape.
//!
//! - An `l2_normalize` -> weighting recorded on the tape gives the gradients
//!   of the backend's backward calls chained by hand, bit for bit, recorded
//!   directly and inside a checkpointed segment.
//! - The same gradients are held to central differences of an `f64` forward,
//!   over rows on both sides of the clamp: `eps` here is `0.5`, large enough
//!   that an ordinary `f32` row can sit below it and a central difference
//!   stays on one side of the `max`.
//! - The clamp branch is pinned exactly. Below `eps` the forward is
//!   `x / eps` with a constant divisor, so it is linear and the gradient is
//!   `grad / eps` with no projection term; subtracting the projection there
//!   (the shape the forward's own formula has above `eps`) is the error this
//!   test exists to catch. The tie `||x|| == eps` belongs to the clamp
//!   branch, as `max(norm, eps)` returning `eps` does.
//! - A variable that is not on the tape is refused and pushes nothing.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const ROWS: usize = 5;
const DIM: usize = 4;
/// Far above the `1e-12` of the embedding path, so a row of ordinary `f32`
/// values can fall below it and central differences can reach the branch.
const EPS: f32 = 0.5;
const NAMES: [&str; 2] = ["x", "r"];

/// Norms 2.0, 1.0, 0.0, 0.1 and sqrt(10): two above the clamp, a zero row
/// and a tiny row below it, and one more above.
const X: [f32; ROWS * DIM] = [
    1.0, -1.0, 1.0, -1.0, //
    0.6, 0.8, 0.0, 0.0, //
    0.0, 0.0, 0.0, 0.0, //
    0.05, -0.05, 0.05, -0.05, //
    2.0, 1.0, -2.0, 1.0,
];
/// Which rows of [`X`] are at or below [`EPS`] and so take `grad / eps`.
const CLAMPED: [usize; 2] = [2, 3];

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

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn leaves_data(be: &CpuBackend) -> [Tensor; 2] {
    [
        Tensor::from_f32(&X, &[ROWS, DIM], be.budget()).unwrap(),
        Tensor::from_f32(&vals(7, ROWS * DIM, 1.0), &[ROWS, DIM], be.budget()).unwrap(),
    ]
}

fn net<B: Backend>(tape: &mut Tape<B>, v: &[Var; 2]) -> Var {
    let [x, r] = *v;
    let unit = tape.l2_normalize(x, EPS).unwrap();
    tape.mul(unit, r).unwrap()
}

/// The same chain through the backend: every leaf's gradient for a seed of
/// ones.
fn by_hand(be: &CpuBackend, k: &[Tensor; 2]) -> [Tensor; 2] {
    let [x, r] = k;
    let unit = be.l2_normalize_forward(x, EPS).unwrap();
    let ones = Tensor::from_f32(&[1.0; ROWS * DIM], &[ROWS, DIM], be.budget()).unwrap();
    let (gu, gr) = be.mul_backward(&unit, r, &ones).unwrap();
    let gx = be.l2_normalize_backward(x, &gu, EPS).unwrap();
    [gx, gr]
}

fn leaves<B: Backend>(tape: &mut Tape<B>, k: &[Tensor; 2]) -> [Var; 2] {
    std::array::from_fn(|i| tape.leaf(k[i].clone()).unwrap())
}

#[test]
fn the_tape_chains_the_l2_normalize_backward() {
    let be = cpu();
    let k = leaves_data(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v);
    assert_eq!(tape.value(z).unwrap().shape(), &[ROWS, DIM]);
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
fn a_checkpointed_l2_normalize_segment_gives_the_same_bits() {
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

/// `sum_{row, d} r * x / max(||x_row||, eps)` in `f64`.
fn net_f64(p: &[f64]) -> f64 {
    let (x, r) = p.split_at(ROWS * DIM);
    let mut total = 0.0;
    for row in 0..ROWS {
        let xs = &x[row * DIM..(row + 1) * DIM];
        let norm = xs.iter().map(|v| v * v).sum::<f64>().sqrt();
        let denom = norm.max(f64::from(EPS));
        for d in 0..DIM {
            total += (xs[d] / denom) * r[row * DIM + d];
        }
    }
    total
}

#[test]
fn the_tapes_l2_normalize_gradients_match_central_differences() {
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
    let numeric = central_diff(&flat, 1e-5, |p| Ok(net_f64(p))).unwrap();
    let mut at = 0;
    for (var, name) in v.iter().zip(NAMES) {
        let got = tape.grad(*var).unwrap().to_f32_vec().unwrap();
        gradients_match(&got, &numeric[at..at + got.len()], 2e-5, 2e-3)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        at += got.len();
    }
}

/// The two rows below `eps` take `grad / eps` exactly: the forward divided
/// them by a constant, so their Jacobian is `I / eps` and no projection is
/// subtracted. The rows above `eps` are not `grad / eps`, so this does not
/// pass by accident.
#[test]
fn the_rows_below_eps_take_the_gradient_divided_by_eps() {
    let be = cpu();
    let k = leaves_data(&be);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v);
    tape.backward(z).unwrap();
    let gx = tape.grad(v[0]).unwrap().to_f32_vec().unwrap();
    // The gradient reaching the normalize is `r` itself (the weighting's
    // other factor), so the clamped rows are `r / eps`.
    let r = k[1].to_f32_vec().unwrap();
    for row in CLAMPED {
        for d in 0..DIM {
            let i = row * DIM + d;
            let want = (f64::from(r[i]) / f64::from(EPS)) as f32;
            assert_eq!(
                gx[i].to_bits(),
                want.to_bits(),
                "row {row} d {d}: {} != {want}",
                gx[i]
            );
        }
    }
    for row in 0..ROWS {
        if CLAMPED.contains(&row) {
            continue;
        }
        let differs = (0..DIM).any(|d| {
            let i = row * DIM + d;
            gx[i].to_bits() != ((f64::from(r[i]) / f64::from(EPS)) as f32).to_bits()
        });
        assert!(differs, "row {row} is above eps but took grad / eps");
    }
}

/// `||x|| == eps` is the clamp branch, as `max(norm, eps)` returning `eps`
/// is. The projection branch would give a different row here, so the tie is
/// pinned and not left to whichever comparison the code happens to use.
#[test]
fn a_norm_exactly_at_eps_takes_the_clamp_branch() {
    let be = cpu();
    let x = Tensor::from_f32(&[0.5, 0.0, 0.0, 0.0], &[1, DIM], be.budget()).unwrap();
    let grad = Tensor::from_f32(&[1.0, 2.0, -3.0, 4.0], &[1, DIM], be.budget()).unwrap();
    let got = be
        .l2_normalize_backward(&x, &grad, EPS)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let g = grad.to_f32_vec().unwrap();
    for d in 0..DIM {
        let want = (f64::from(g[d]) / f64::from(EPS)) as f32;
        assert_eq!(got[d].to_bits(), want.to_bits(), "d {d}: {} ", got[d]);
    }
    // The projection branch at this row would subtract `y (y . g)` with
    // `y = [1, 0, 0, 0]`, so component 0 would be 0 rather than 2.
    assert_ne!(got[0], 0.0);
}

#[test]
fn a_variable_off_the_tape_is_refused_and_pushes_nothing() {
    let be = cpu();
    let mut tape = Tape::new(be.clone());
    let a = tape
        .leaf(Tensor::from_f32(&X, &[ROWS, DIM], be.budget()).unwrap())
        .unwrap();
    match tape.l2_normalize(Var(99), EPS) {
        Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, "Tape::l2_normalize"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
    assert!(
        tape.value(Var(1)).is_err(),
        "a refused record pushed a node"
    );
    // The next record still lands where it should.
    assert_eq!(tape.l2_normalize(a, EPS).unwrap(), Var(1));
}
