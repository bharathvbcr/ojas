//! `Tape::linear_q`: a linear over a frozen, resident affine-quantised weight (card ft-5f743c19).
//!
//! The record holds the packed weight and dequantises it to a transient in the forward and again
//! in the backward, so a tape over `linear_q` must give the same values and input gradients, bit
//! for bit, as a tape over a frozen dense leaf holding the host dequantisation of the same
//! quantisation, with and without `checkpoint` (a replay re-runs the record against moved
//! indices), and the same gradient as the backward calls chained by hand
//! (`dequant_affine_bf16`, then `linear_backward_input`, per layer).
//!
//! The dense reference uses the host dequantiser, so this is a composition check, not an oracle
//! for the dequantiser itself (W2's mlx goldens are).

mod common;

use common::Resident;
use ojas_autograd::{Tape, Var};
use ojas_core::{
    dequantize_affine_bf16, f32_to_bf16, quantize_affine, AffineSpec, Backend, BackendCore,
    BackendId, BackendModule, Budget, OjasError, PlacedAffine, Tensor,
};
use ojas_cpu::CpuBackend;

const ROWS: usize = 6;
const D0: usize = 128;
const D1: usize = 192;

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 28))
}

/// Deterministic values in `[lo, hi)`.
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

/// A bf16 weight `[out, inn]` quantised to q8, group 64, as `mx.quantize` does.
fn quantised(seed: u64, out: usize, inn: usize, budget: &Budget) -> ojas_core::AffineQuantized {
    let bf: Vec<u16> = vals(seed, out * inn, -1.5, 1.5)
        .into_iter()
        .map(f32_to_bf16)
        .collect();
    let w = Tensor::from_bf16_bits(&bf, &[out, inn], budget).unwrap();
    quantize_affine(&w, AffineSpec::q8(), budget).unwrap()
}

struct Weights {
    q1: ojas_core::AffineQuantized,
    q2: ojas_core::AffineQuantized,
}

fn weights(budget: &Budget) -> Weights {
    Weights {
        q1: quantised(11, D1, D0, budget),
        q2: quantised(12, D0, D1, budget),
    }
}

fn x_leaf<B: Backend>(tape: &mut Tape<B>) -> Var {
    let budget = tape.backend().budget().clone();
    tape.leaf(Tensor::from_f32(&vals(7, ROWS * D0, -1.0, 1.0), &[ROWS, D0], &budget).unwrap())
        .unwrap()
}

/// The result `(y2 bits, dx bits)` of the two-layer chain over packed weights, optionally with
/// both linears in one checkpointed segment.
fn packed_chain(w: &Weights, checkpointed: bool) -> (Vec<u32>, Vec<u32>) {
    let be = cpu();
    let (p1, p2) = (
        w.q1.clone().place(&be).unwrap(),
        w.q2.clone().place(&be).unwrap(),
    );
    let mut tape = Tape::new(be);
    let x = x_leaf(&mut tape);
    let y2 = if checkpointed {
        tape.checkpoint(|t| {
            let a = t.linear_q(x, &p1)?;
            Ok(vec![t.linear_q(a, &p2)?])
        })
        .unwrap()[0]
    } else {
        let a = tape.linear_q(x, &p1).unwrap();
        tape.linear_q(a, &p2).unwrap()
    };
    tape.backward(y2).unwrap();
    (
        bits(tape.value(y2).unwrap()),
        bits(tape.grad(x).expect("x takes a gradient")),
    )
}

/// The same chain over frozen dense leaves holding the host dequantisation.
fn dense_chain(w: &Weights, checkpointed: bool) -> (Vec<u32>, Vec<u32>) {
    let be = cpu();
    let budget = be.budget().clone();
    let d1 = dequantize_affine_bf16(&w.q1, &budget).unwrap();
    let d2 = dequantize_affine_bf16(&w.q2, &budget).unwrap();
    let mut tape = Tape::new(be);
    let x = x_leaf(&mut tape);
    let (f1, f2) = (tape.frozen_leaf(d1).unwrap(), tape.frozen_leaf(d2).unwrap());
    let y2 = if checkpointed {
        tape.checkpoint(|t| {
            let a = t.linear(x, f1)?;
            Ok(vec![t.linear(a, f2)?])
        })
        .unwrap()[0]
    } else {
        let a = tape.linear(x, f1).unwrap();
        tape.linear(a, f2).unwrap()
    };
    tape.backward(y2).unwrap();
    (
        bits(tape.value(y2).unwrap()),
        bits(tape.grad(x).expect("x takes a gradient")),
    )
}

#[test]
fn linear_q_equals_a_frozen_dense_leaf_bitwise_without_checkpointing() {
    let w = weights(&Budget::new(1 << 28));
    let (packed, dense) = (packed_chain(&w, false), dense_chain(&w, false));
    assert_eq!(packed.0, dense.0, "forward values differ");
    assert_eq!(packed.1, dense.1, "input gradients differ");
    // Not vacuous: the chain produces non-trivial values and gradients.
    assert!(packed.0.iter().any(|&b| b != 0) && packed.1.iter().any(|&b| b != 0));
}

/// Inside a checkpointed segment the replay re-records the two linears against moved indices; a
/// missed `moved` arm would read the wrong `x` and the gradient would differ.
#[test]
fn linear_q_equals_a_frozen_dense_leaf_bitwise_under_checkpoint() {
    let w = weights(&Budget::new(1 << 28));
    let (packed, dense) = (packed_chain(&w, true), dense_chain(&w, true));
    assert_eq!(packed.0, dense.0);
    assert_eq!(packed.1, dense.1);
    // And checkpointing changes nothing about the values.
    assert_eq!(packed, packed_chain(&w, false));
}

/// AGENTS.md step 6: the tape's gradient is the backward chained by hand, bit for bit.
#[test]
fn linear_q_gradient_is_the_hand_chained_backward() {
    let w = weights(&Budget::new(1 << 28));
    let be = cpu();
    let (p1, p2) = (
        w.q1.clone().place(&be).unwrap(),
        w.q2.clone().place(&be).unwrap(),
    );
    let mut tape = Tape::new(be.clone());
    let x = x_leaf(&mut tape);
    let a = tape.linear_q(x, &p1).unwrap();
    let y2 = tape.linear_q(a, &p2).unwrap();
    tape.backward(y2).unwrap();

    let deq = |p: &PlacedAffine| {
        be.dequant_affine_bf16(p.packed(), p.scales(), p.biases(), p.spec())
            .unwrap()
    };
    let (w1, w2) = (deq(&p1), deq(&p2));
    let budget = be.budget().clone();
    let seed = Tensor::from_f32(&[1.0f32; ROWS * D0], &[ROWS, D0], &budget).unwrap();
    let g1 = be
        .linear_backward_input(tape.value(a).unwrap(), &w2, &seed)
        .unwrap();
    let gx = be
        .linear_backward_input(tape.value(x).unwrap(), &w1, &g1)
        .unwrap();
    assert_eq!(bits(tape.grad(x).unwrap()), bits(&gx));
}

/// The forward value is `linear_forward` over the dequantised weight: no extra rounding.
#[test]
fn linear_q_forward_is_linear_forward_over_the_dequantisation() {
    let w = weights(&Budget::new(1 << 28));
    let be = cpu();
    let p1 = w.q1.clone().place(&be).unwrap();
    let mut tape = Tape::new(be.clone());
    let x = x_leaf(&mut tape);
    let y = tape.linear_q(x, &p1).unwrap();
    let dense = be
        .dequant_affine_bf16(p1.packed(), p1.scales(), p1.biases(), p1.spec())
        .unwrap();
    let want = be.linear_forward(tape.value(x).unwrap(), &dense).unwrap();
    assert_eq!(bits(tape.value(y).unwrap()), bits(&want));
    assert_eq!(tape.value(y).unwrap().shape(), [ROWS, D1]);
}

/// A frozen weight takes no gradient, and the dequantised weight is never a value of the tape.
#[test]
fn linear_q_records_one_value_and_no_dense_weight() {
    let w = weights(&Budget::new(1 << 28));
    let be = cpu();
    let p1 = w.q1.clone().place(&be).unwrap();
    let mut tape = Tape::new(be);
    let x = x_leaf(&mut tape);
    let before = (0..).take_while(|&i| tape.value(Var(i)).is_ok()).count();
    let y = tape.linear_q(x, &p1).unwrap();
    let after = (0..).take_while(|&i| tape.value(Var(i)).is_ok()).count();
    assert_eq!((before, after), (1, 2), "one record, one value");
    assert_eq!(y, Var(1));
    for i in 0..after {
        let v = tape.value(Var(i)).unwrap();
        assert_ne!(
            v.shape(),
            [D1, D0],
            "a dense [out, in] weight is on the tape"
        );
    }
}

// ---- refusals: before any dispatch, and nothing is recorded -----------------------------------

#[test]
fn an_input_of_the_wrong_width_is_a_shape_refusal_and_records_nothing() {
    let w = weights(&Budget::new(1 << 28));
    let be = cpu();
    let p1 = w.q1.clone().place(&be).unwrap();
    let mut tape = Tape::new(be);
    let budget = tape.backend().budget().clone();
    let wrong = tape
        .leaf(Tensor::from_f32(&[0.5f32; ROWS * (D0 + 1)], &[ROWS, D0 + 1], &budget).unwrap())
        .unwrap();
    match tape.linear_q(wrong, &p1) {
        Err(OjasError::Shape { op, detail }) => {
            assert_eq!(op, "Tape::linear_q");
            assert!(detail.contains("128"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
    assert!(tape.value(Var(1)).is_err(), "a refusal records nothing");
}

#[test]
fn a_variable_that_is_not_on_the_tape_is_out_of_range() {
    let w = weights(&Budget::new(1 << 28));
    let be = cpu();
    let p1 = w.q1.clone().place(&be).unwrap();
    let mut tape = Tape::new(be);
    match tape.linear_q(Var(9), &p1) {
        Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, "Tape::linear_q"),
        other => panic!("{other:?}"),
    }
}

/// D9: a weight is placed once, where the tape computes. Host parts on a device tape and device
/// parts on a host tape are both `Placement`, with nothing recorded.
#[test]
fn a_weight_placed_elsewhere_is_a_placement_refusal_both_ways() {
    let w = weights(&Budget::new(1 << 28));
    // Host parts on a device tape (the double reports Wgpu).
    let host = w.q1.clone().place(&cpu()).unwrap();
    assert_eq!(host.backend(), BackendId::Cpu);
    let mut device_tape = Tape::new(Resident::new());
    let x = x_leaf(&mut device_tape);
    match device_tape.linear_q(x, &host) {
        Err(OjasError::Placement {
            op,
            expected,
            found,
        }) => {
            assert_eq!(op, "Tape::linear_q");
            assert_eq!((expected, found), (Some(BackendId::Wgpu), None));
        }
        other => panic!("{other:?}"),
    }
    assert!(device_tape.value(Var(1)).is_err());

    // Device parts on a host tape.
    let on_device = w.q1.clone().place(&Resident::new()).unwrap();
    assert_eq!(on_device.backend(), BackendId::Wgpu);
    assert!(
        on_device.packed().device().is_some(),
        "the parts were uploaded"
    );
    let mut host_tape = Tape::new(cpu());
    let x = x_leaf(&mut host_tape);
    match host_tape.linear_q(x, &on_device) {
        Err(OjasError::Placement {
            op,
            expected,
            found,
        }) => {
            assert_eq!(op, "Tape::linear_q");
            assert_eq!((expected, found), (None, Some(BackendId::Wgpu)));
        }
        other => panic!("{other:?}"),
    }
    assert!(host_tape.value(Var(1)).is_err());
}

/// A rank-1 input whose only extent equals the weight's width is refused before any dispatch, as
/// the doc says, whatever a backend's linear would do with it.
#[test]
fn a_rank_one_input_is_a_shape_refusal_and_records_nothing() {
    let w = weights(&Budget::new(1 << 28));
    let be = cpu();
    let p1 = w.q1.clone().place(&be).unwrap();
    let mut tape = Tape::new(be);
    let budget = tape.backend().budget().clone();
    let flat = tape
        .leaf(Tensor::from_f32(&[0.5f32; D0], &[D0], &budget).unwrap())
        .unwrap();
    match tape.linear_q(flat, &p1) {
        Err(OjasError::Shape { op, .. }) => assert_eq!(op, "Tape::linear_q"),
        other => panic!("{other:?}"),
    }
    assert!(tape.value(Var(1)).is_err());
}
