//! `Tape::rms_qkv_norm`: EmbeddingGemma 2's per-head Q/K/V RMSNorm on the
//! tape, composed from three `Tape::rms_norm` records rather than from a
//! fourth hand-written RMSNorm gradient.
//!
//! - The three outputs are `Backend::rms_qkv_norm_forward`'s outputs bit for
//!   bit, with learned Q/K weights and weightless, so the composition is the
//!   same forward the backend op computes and not a near-copy of it.
//! - The gradients are the backend's `rms_norm_backward` chained by hand, bit
//!   for bit, for every leg: this is what "composes, does not duplicate"
//!   means operationally.
//! - The same gradients are held to central differences of an `f64` forward.
//! - A weightless leg's ones vector is a frozen leaf, so it takes no
//!   gradient and no caller has to discard one.
//! - A variable that is not on the tape is refused and pushes nothing.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const B: usize = 1;
const HQ: usize = 2;
const HKV: usize = 1;
const T: usize = 3;
const D: usize = 4;
const EPS: f32 = 1e-6;

const QN: usize = B * HQ * T * D;
const KN: usize = B * HKV * T * D;
const RN: usize = B * (HQ + 2 * HKV) * T * D;

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

fn t(be: &CpuBackend, seed: u64, shape: &[usize], amp: f32) -> Tensor {
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

fn ones(be: &CpuBackend, n: usize) -> Tensor {
    Tensor::from_f32(&vec![1.0; n], &[n], be.budget()).unwrap()
}

/// `q`, `k`, `v`, `q_weight`, `k_weight`, and the weighting `r` over the
/// three outputs concatenated on the head axis.
fn leaves_data(be: &CpuBackend) -> [Tensor; 6] {
    [
        t(be, 51, &[B, HQ, T, D], 2.0),
        t(be, 52, &[B, HKV, T, D], 2.0),
        t(be, 53, &[B, HKV, T, D], 2.0),
        t(be, 54, &[D], 1.0),
        t(be, 55, &[D], 1.0),
        t(be, 56, &[B, HQ + 2 * HKV, T, D], 1.0),
    ]
}

/// The three outputs concatenated on the head axis and weighted, so one
/// backward walk reaches every leg.
fn net<Be: Backend>(tape: &mut Tape<Be>, v: &[Var; 6], weighted: bool) -> Var {
    let [q, k, val, qw, kw, r] = *v;
    let (wq, wk) = if weighted {
        (Some(qw), Some(kw))
    } else {
        (None, None)
    };
    let (qn, kn, vn) = tape.rms_qkv_norm(q, k, val, wq, wk, EPS).unwrap();
    let all = tape.concat(&[qn, kn, vn], 1).unwrap();
    tape.mul(all, r).unwrap()
}

/// Every leg's gradient through the backend's own `rms_norm_backward`, with
/// the ones vector a weightless leg uses.
fn by_hand(be: &CpuBackend, k: &[Tensor; 6], weighted: bool) -> [Tensor; 6] {
    let [q, kt, vt, qw, kw, r] = k;
    let one = ones(be, D);
    let (wq, wk) = if weighted { (qw, kw) } else { (&one, &one) };
    let qn = be.rms_norm_forward(q, wq, EPS).unwrap();
    let kn = be.rms_norm_forward(kt, wk, EPS).unwrap();
    let vn = be.rms_norm_forward(vt, &one, EPS).unwrap();
    let all = be.concat(&[&qn, &kn, &vn], 1).unwrap();
    let seed = Tensor::from_f32(&vec![1.0; RN], all.shape(), be.budget()).unwrap();
    let (gall, gr) = be.mul_backward(&all, r, &seed).unwrap();
    // The concat's gradient is its slices, in head order.
    let gq_out = be.slice(&gall, 1, 0, HQ).unwrap();
    let gk_out = be.slice(&gall, 1, HQ, HKV).unwrap();
    let gv_out = be.slice(&gall, 1, HQ + HKV, HKV).unwrap();
    let (gq, gqw) = be.rms_norm_backward(q, wq, &gq_out, EPS).unwrap();
    let (gk, gkw) = be.rms_norm_backward(kt, wk, &gk_out, EPS).unwrap();
    let (gv, _) = be.rms_norm_backward(vt, &one, &gv_out, EPS).unwrap();
    [gq, gk, gv, gqw, gkw, gr]
}

fn leaves<Be: Backend>(tape: &mut Tape<Be>, k: &[Tensor; 6]) -> [Var; 6] {
    std::array::from_fn(|i| tape.leaf(k[i].clone()).unwrap())
}

#[test]
fn the_tapes_rms_qkv_forward_is_the_backend_ops_output_bit_for_bit() {
    let be = cpu();
    let k = leaves_data(&be);
    for weighted in [true, false] {
        let (wq, wk) = if weighted {
            (Some(&k[3]), Some(&k[4]))
        } else {
            (None, None)
        };
        let (wq_t, wk_t) = (wq.cloned(), wk.cloned());
        let (want_q, want_k, want_v) = be
            .rms_qkv_norm_forward(&k[0], &k[1], &k[2], wq_t.as_ref(), wk_t.as_ref(), EPS)
            .expect("backend rms_qkv_norm_forward");
        let mut tape = Tape::new(be.clone());
        let v = leaves(&mut tape, &k);
        let (aq, ak, av) = if weighted {
            tape.rms_qkv_norm(v[0], v[1], v[2], Some(v[3]), Some(v[4]), EPS)
                .unwrap()
        } else {
            tape.rms_qkv_norm(v[0], v[1], v[2], None, None, EPS)
                .unwrap()
        };
        for ((got, want), name) in [aq, ak, av]
            .iter()
            .zip([&want_q, &want_k, &want_v])
            .zip(["q", "k", "v"])
        {
            let got = tape.value(*got).unwrap();
            assert_eq!(got.shape(), want.shape(), "{name} weighted {weighted}");
            assert_eq!(bits(got), bits(want), "{name} weighted {weighted}");
        }
    }
}

#[test]
fn the_tape_chains_the_rms_norm_backward_on_every_leg() {
    let be = cpu();
    let k = leaves_data(&be);
    for weighted in [true, false] {
        let want = by_hand(&be, &k, weighted);
        let mut tape = Tape::new(be.clone());
        let v = leaves(&mut tape, &k);
        let z = net(&mut tape, &v, weighted);
        tape.backward(z).unwrap();
        let names = ["q", "k", "v", "q_weight", "k_weight", "r"];
        for (i, name) in names.iter().enumerate() {
            // A weightless call leaves the Q and K weight leaves untouched.
            if !weighted && (i == 3 || i == 4) {
                assert!(
                    tape.grad(v[i]).is_none(),
                    "{name} took a gradient from a weightless call"
                );
                continue;
            }
            let got = tape
                .grad(v[i])
                .unwrap_or_else(|| panic!("{name} takes a gradient"));
            assert_eq!(got.shape(), want[i].shape(), "{name} weighted {weighted}");
            assert_eq!(bits(got), bits(&want[i]), "{name} weighted {weighted}");
        }
    }
}

/// `sum(concat(qn, kn, vn) * r)` in `f64`, with
/// `rstd = 1 / sqrt(mean(x^2) + eps)` per last-axis row.
fn net_f64(p: &[f64]) -> f64 {
    let (q, rest) = p.split_at(QN);
    let (k, rest) = rest.split_at(KN);
    let (v, rest) = rest.split_at(KN);
    let (qw, rest) = rest.split_at(D);
    let (kw, r) = rest.split_at(D);
    let one = vec![1.0f64; D];
    let mut total = 0.0;
    // The concat's head order: HQ q heads, then the k head, then the v head.
    let legs: [(&[f64], &[f64], usize); 3] = [(q, qw, 0), (k, kw, HQ), (v, &one, HQ + HKV)];
    for (data, weight, head0) in legs {
        let heads = data.len() / (T * D);
        for h in 0..heads {
            for row in 0..T {
                let at = (h * T + row) * D;
                let xs = &data[at..at + D];
                let ms = xs.iter().map(|e| e * e).sum::<f64>() / D as f64;
                let rstd = 1.0 / (ms + f64::from(EPS)).sqrt();
                for d in 0..D {
                    let ri = ((head0 + h) * T + row) * D + d;
                    total += xs[d] * rstd * weight[d] * r[ri];
                }
            }
        }
    }
    total
}

#[test]
fn the_tapes_rms_qkv_gradients_match_central_differences() {
    let be = cpu();
    let k = leaves_data(&be);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v, true);
    tape.backward(z).unwrap();
    let flat: Vec<f64> = k
        .iter()
        .flat_map(|t| t.to_f32_vec().unwrap())
        .map(f64::from)
        .collect();
    assert_eq!(flat.len(), QN + 2 * KN + 2 * D + RN);
    let numeric = central_diff(&flat, 1e-3, |p| Ok(net_f64(p))).unwrap();
    let mut at = 0;
    let names = ["q", "k", "v", "q_weight", "k_weight", "r"];
    for (var, name) in v.iter().zip(names) {
        let got = tape.grad(*var).unwrap().to_f32_vec().unwrap();
        // The RMSNorm's reciprocal square root is an `f32` computation, so
        // these are the tolerances `tests/gradcheck.rs` holds it to.
        gradients_match(&got, &numeric[at..at + got.len()], 2e-3, 2e-2)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        at += got.len();
    }
}

#[test]
fn a_variable_off_the_tape_is_refused_and_pushes_nothing() {
    let be = cpu();
    let k = leaves_data(&be);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let before = tape.value(v[5]).unwrap().shape().to_vec();
    match tape.rms_qkv_norm(Var(99), v[1], v[2], None, None, EPS) {
        Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, "Tape::rms_qkv_norm"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
    // A weight that is not on the tape is refused the same way, and neither
    // refusal leaves a node behind.
    match tape.rms_qkv_norm(v[0], v[1], v[2], Some(Var(99)), None, EPS) {
        Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, "Tape::rms_qkv_norm"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
    assert!(
        tape.value(Var(6)).is_err(),
        "a refused record pushed a node"
    );
    assert_eq!(tape.value(v[5]).unwrap().shape(), before);
}
