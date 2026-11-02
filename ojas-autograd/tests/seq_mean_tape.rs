//! `Tape::seq_mean`: the masked mean over time on the tape.
//!
//! - A `seq_mean` -> weighting recorded on the tape gives the gradients of
//!   the backend's backward calls chained by hand, bit for bit, recorded
//!   directly and inside a checkpointed segment.
//! - The same gradients are held to central differences of an `f64` forward
//!   that divides by `min(key_lens[b], T)`, so a divisor taken from the raw
//!   length, a gradient spread onto the padding rows, or one scaled by the
//!   wrong count fails here.
//! - The padding rows take exactly `+0.0`, and a sequence of length 0 takes
//!   a block of zeros.
//! - A variable that is not on the tape is refused and pushes nothing.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const BATCH: usize = 4;
const SEQ: usize = 4;
const DIM: usize = 3;
/// A length past the end of the sequence, an empty one, a padded one and a
/// full one: every branch of `min(key_lens[b], T)` in one batch.
const LENS: [u32; BATCH] = [6, 0, 2, 4];
const NAMES: [&str; 2] = ["x", "r"];

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

fn lens(be: &CpuBackend) -> Tensor {
    Tensor::from_u32(&LENS, &[BATCH], be.budget()).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `x` `[B, T, D]` and the weighting `r` `[B, D]`.
fn leaves_data(be: &CpuBackend) -> [Tensor; 2] {
    [
        tensor(be, 11, &[BATCH, SEQ, DIM], 2.0),
        tensor(be, 12, &[BATCH, DIM], 1.0),
    ]
}

fn net<B: Backend>(tape: &mut Tape<B>, v: &[Var; 2], key_lens: Tensor) -> Var {
    let [x, r] = *v;
    let pooled = tape.seq_mean(x, key_lens).unwrap();
    tape.mul(pooled, r).unwrap()
}

/// The same chain through the backend: every leaf's gradient for a seed of
/// ones.
fn by_hand(be: &CpuBackend, k: &[Tensor; 2]) -> [Tensor; 2] {
    let [x, r] = k;
    let key_lens = lens(be);
    let pooled = be.seq_mean_forward(x, &key_lens).unwrap();
    let ones = Tensor::from_f32(&[1.0; BATCH * DIM], &[BATCH, DIM], be.budget()).unwrap();
    let (gp, gr) = be.mul_backward(&pooled, r, &ones).unwrap();
    let gx = be.seq_mean_backward(x, &key_lens, &gp).unwrap();
    [gx, gr]
}

fn leaves<B: Backend>(tape: &mut Tape<B>, k: &[Tensor; 2]) -> [Var; 2] {
    std::array::from_fn(|i| tape.leaf(k[i].clone()).unwrap())
}

#[test]
fn the_tape_chains_the_seq_mean_backward() {
    let be = cpu();
    let k = leaves_data(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v, lens(&be));
    assert_eq!(tape.value(z).unwrap().shape(), &[BATCH, DIM]);
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
fn a_checkpointed_seq_mean_segment_gives_the_same_bits() {
    let be = cpu();
    let k = leaves_data(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let key_lens = lens(&be);
    let out = tape
        .checkpoint(|t| Ok(vec![net(t, &v, key_lens.clone())]))
        .unwrap();
    tape.backward(out[0]).unwrap();
    for ((var, want), name) in v.iter().zip(&want).zip(NAMES) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want), "{name}");
    }
}

/// `sum_{b, d} r[b, d] * mean_{t < min(len[b], T)} x[b, t, d]` in `f64`.
fn net_f64(p: &[f64]) -> f64 {
    let (x, r) = p.split_at(BATCH * SEQ * DIM);
    let mut total = 0.0;
    for (b, &len) in LENS.iter().enumerate() {
        let rows = (len as usize).min(SEQ);
        for d in 0..DIM {
            if rows == 0 {
                continue;
            }
            let sum: f64 = (0..rows).map(|t| x[(b * SEQ + t) * DIM + d]).sum();
            total += (sum / rows as f64) * r[b * DIM + d];
        }
    }
    total
}

#[test]
fn the_tapes_seq_mean_gradients_match_central_differences() {
    let be = cpu();
    let k = leaves_data(&be);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v, lens(&be));
    tape.backward(z).unwrap();
    let flat: Vec<f64> = k
        .iter()
        .flat_map(|t| t.to_f32_vec().unwrap())
        .map(f64::from)
        .collect();
    let numeric = central_diff(&flat, 1e-4, |p| Ok(net_f64(p))).unwrap();
    let mut at = 0;
    for (var, name) in v.iter().zip(NAMES) {
        let got = tape.grad(*var).unwrap().to_f32_vec().unwrap();
        gradients_match(&got, &numeric[at..at + got.len()], 2e-6, 2e-4)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        at += got.len();
    }
}

/// The analytical gradient of `x`, row by row: `1/min(len, T)` of the
/// incoming gradient on the rows the mean read and `+0.0` past them. A
/// length over `T` divides by `T`, which is the divisor the forward used.
#[test]
fn the_seq_mean_gradient_is_zero_on_the_padding_rows() {
    let be = cpu();
    let k = leaves_data(&be);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v, lens(&be));
    tape.backward(z).unwrap();
    let gx = tape.grad(v[0]).unwrap().to_f32_vec().unwrap();
    let r = k[1].to_f32_vec().unwrap();
    for (b, &len) in LENS.iter().enumerate() {
        let rows = (len as usize).min(SEQ);
        for t in 0..SEQ {
            for d in 0..DIM {
                let got = gx[(b * SEQ + t) * DIM + d];
                if t < rows {
                    let want = (f64::from(r[b * DIM + d]) / rows as f64) as f32;
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "b {b} t {t} d {d}: {got} != {want}"
                    );
                } else {
                    assert_eq!(got.to_bits(), 0.0f32.to_bits(), "b {b} t {t} d {d}: {got}");
                }
            }
        }
    }
}

#[test]
fn a_variable_off_the_tape_is_refused_and_pushes_nothing() {
    let be = cpu();
    let mut tape = Tape::new(be.clone());
    let a = tape.leaf(tensor(&be, 41, &[BATCH, SEQ, DIM], 1.0)).unwrap();
    match tape.seq_mean(Var(99), lens(&be)) {
        Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, "Tape::seq_mean"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
    assert!(
        tape.value(Var(1)).is_err(),
        "a refused record pushed a node"
    );
    // The next record still lands where it should.
    assert_eq!(tape.seq_mean(a, lens(&be)).unwrap(), Var(1));
}
