//! Frozen leaves (`Tape::frozen_leaf`) and `Backend::linear_backward_input`.
//!
//! A frozen leaf has no gradient (absent, not zero) and costs no weight
//! gradient: a linear with a frozen weight runs the dX-only backward, whose
//! result equals the full path's `grad_input` bit for bit and which never
//! allocates the weight-gradient buffer (measured on the backend's budget).

mod common;

use common::{bits, data};
use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

const BIG: u64 = 1 << 28;

fn t(budget: &Budget, seed: u32, shape: &[usize]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&data(seed, n), shape, budget).unwrap()
}

/// Highest live bytes above the starting level while `f` runs, and the live
/// bytes `f`'s result keeps.
fn measure<R>(budget: &Budget, f: impl FnOnce() -> R) -> (R, u64, u64) {
    let before = budget.live_bytes().unwrap();
    budget.reset_peak();
    let out = f();
    let peak = budget.peak_bytes() - before;
    let kept = budget.live_bytes().unwrap() - before;
    (out, peak, kept)
}

#[test]
fn dx_only_equals_the_full_path_bit_for_bit_and_allocates_no_dw() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for (rows, kin, nout) in [(1, 64, 256), (3, 40, 17), (96, 128, 512), (130, 70, 90)] {
            let budget = Budget::new(BIG);
            let be = CpuBackend::new(budget.clone()).with_numerics(numerics);
            let x = t(&budget, 1, &[rows, kin]);
            let w = t(&budget, 2, &[nout, kin]);
            let g = t(&budget, 3, &[rows, nout]);
            let dx_bytes = (rows * kin * 4) as u64;
            let dw_bytes = (nout * kin * 4) as u64;
            let ((full_dx, _full_dw), full_peak, full_kept) =
                measure(&budget, || be.linear_backward(&x, &w, &g).unwrap());
            let (only, peak, kept) =
                measure(&budget, || be.linear_backward_input(&x, &w, &g).unwrap());
            assert_eq!(
                bits(&only),
                bits(&full_dx),
                "{numerics:?} {rows}x{kin}x{nout}"
            );
            assert_eq!(kept, dx_bytes, "only dX stays live");
            assert_eq!(full_kept, dx_bytes + dw_bytes);
            assert!(peak >= dx_bytes);
            assert!(
                peak <= full_peak - dw_bytes,
                "dX-only peak {peak} must be the full peak {full_peak} less dW {dw_bytes}"
            );
        }
    }
}

#[test]
fn dx_only_accepts_the_bf16_weight_the_full_path_accepts() {
    let budget = Budget::new(BIG);
    let be = CpuBackend::new(budget.clone());
    let x = t(&budget, 1, &[4, 32]);
    let w = t(&budget, 2, &[48, 32]);
    let g = t(&budget, 3, &[4, 48]);
    let wb = be.to_bf16(&w).unwrap();
    let (full_dx, _) = be.linear_backward(&x, &wb, &g).unwrap();
    let only = be.linear_backward_input(&x, &wb, &g).unwrap();
    assert_eq!(bits(&only), bits(&full_dx));
}

#[test]
fn dx_only_refuses_what_the_full_path_refuses() {
    let budget = Budget::new(BIG);
    let be = CpuBackend::new(budget.clone());
    let x = t(&budget, 1, &[4, 32]);
    let w = t(&budget, 2, &[48, 32]);
    let bad = t(&budget, 3, &[4, 47]);
    assert!(be.linear_backward(&x, &w, &bad).is_err());
    assert!(be.linear_backward_input(&x, &w, &bad).is_err());
    let mut nan = data(3, 4 * 48);
    nan[5] = f32::NAN;
    let g = Tensor::from_f32(&nan, &[4, 48], &budget).unwrap();
    assert!(be.linear_backward(&x, &w, &g).is_err());
    assert!(be.linear_backward_input(&x, &w, &g).is_err());
}

/// `loss = sum(linear(linear(x, w1), w2))` where `frozen` picks which
/// leaves are recorded frozen.
fn run(frozen_w1: bool, frozen_w2: bool) -> (Vec<Option<Vec<u32>>>, u64) {
    let budget = Budget::new(BIG);
    let mut tape = Tape::new(CpuBackend::new(budget.clone()));
    let x = tape.leaf(t(&budget, 1, &[6, 16])).unwrap();
    let mk = |tape: &mut Tape<CpuBackend>, seed, shape: &[usize], frozen| {
        let value = t(&budget, seed, shape);
        if frozen {
            tape.frozen_leaf(value).unwrap()
        } else {
            tape.leaf(value).unwrap()
        }
    };
    let w1 = mk(&mut tape, 2, &[24, 16], frozen_w1);
    let w2 = mk(&mut tape, 3, &[8, 24], frozen_w2);
    let h = tape.linear(x, w1).unwrap();
    let y = tape.linear(h, w2).unwrap();
    let rows = y;
    let before = budget.live_bytes().unwrap();
    tape.backward(rows).unwrap();
    let grown = budget.live_bytes().unwrap() - before;
    let grads = [x, w1, w2]
        .iter()
        .map(|&v| tape.grad(v).map(bits))
        .collect();
    (grads, grown)
}

#[test]
fn a_frozen_leaf_has_no_gradient_and_the_others_are_unchanged() {
    let (full, full_bytes) = run(false, false);
    let (frozen, frozen_bytes) = run(true, true);
    assert!(full.iter().all(Option::is_some));
    assert_eq!(frozen[0], full[0], "dX unchanged bit for bit");
    assert!(
        frozen[1].is_none() && frozen[2].is_none(),
        "absent, not zero"
    );
    assert_eq!(full_bytes - frozen_bytes, ((24 * 16 + 8 * 24) * 4) as u64);
    let (mixed, _) = run(true, false);
    assert!(mixed[1].is_none());
    assert_eq!(mixed[0], full[0]);
    assert_eq!(mixed[2], full[2]);
}

#[test]
fn take_grad_of_a_frozen_leaf_is_none() {
    let budget = Budget::new(BIG);
    let mut tape = Tape::new(CpuBackend::new(budget.clone()));
    let x = tape.leaf(t(&budget, 1, &[2, 4])).unwrap();
    let w = tape.frozen_leaf(t(&budget, 2, &[3, 4])).unwrap();
    let y = tape.linear(x, w).unwrap();
    tape.backward(y).unwrap();
    assert!(tape.take_grad(w).is_none());
    assert!(tape.take_grad(x).is_some());
}

#[test]
fn a_frozen_leaf_used_twice_and_through_views_takes_no_gradient() {
    let budget = Budget::new(BIG);
    let mut tape = Tape::new(CpuBackend::new(budget.clone()));
    let x = tape.leaf(t(&budget, 1, &[2, 4])).unwrap();
    let w = tape.frozen_leaf(t(&budget, 2, &[4, 4])).unwrap();
    let view = tape.reshape(w, &[4, 4]).unwrap();
    let a = tape.linear(x, w).unwrap();
    let b = tape.linear(a, view).unwrap();
    tape.backward(b).unwrap();
    assert!(tape.grad(w).is_none() && tape.grad(view).is_none());
    assert!(tape.grad(x).is_some());

    // The same tensor recorded as a trainable leaf is a separate leaf.
    let mut tape = Tape::new(CpuBackend::new(budget.clone()));
    let x = tape.leaf(t(&budget, 1, &[2, 4])).unwrap();
    let frozen = tape.frozen_leaf(t(&budget, 2, &[4, 4])).unwrap();
    let live = tape.leaf(t(&budget, 2, &[4, 4])).unwrap();
    let a = tape.linear(x, frozen).unwrap();
    let b = tape.linear(a, live).unwrap();
    tape.backward(b).unwrap();
    assert!(tape.grad(frozen).is_none());
    assert!(tape.grad(live).is_some());
}

#[test]
fn a_graph_of_only_frozen_leaves_backwards_without_allocating() {
    let budget = Budget::new(BIG);
    let mut tape = Tape::new(CpuBackend::new(budget.clone()));
    let x = tape.frozen_leaf(t(&budget, 1, &[2, 4])).unwrap();
    let w = tape.frozen_leaf(t(&budget, 2, &[3, 4])).unwrap();
    let y = tape.linear(x, w).unwrap();
    let ((), peak, kept) = measure(&budget, || tape.backward(y).unwrap());
    assert_eq!((peak, kept), (0, 0));
    assert!(tape.grad(x).is_none() && tape.grad(w).is_none() && tape.grad(y).is_none());
}

#[test]
fn a_frozen_embedding_table_costs_no_table_sized_gradient() {
    let budget = Budget::new(BIG);
    let vocab = 512;
    let dim = 16;
    let table_bytes = (vocab * dim * 4) as u64;
    let peak_of = |frozen: bool| {
        let mut tape = Tape::new(CpuBackend::new(budget.clone()));
        let value = t(&budget, 1, &[vocab, dim]);
        let table = if frozen {
            tape.frozen_leaf(value).unwrap()
        } else {
            tape.leaf(value).unwrap()
        };
        let w = tape.leaf(t(&budget, 2, &[8, dim])).unwrap();
        let ids = Tensor::from_u32(&[3, 7, 7], &[3], &budget).unwrap();
        let e = tape.embedding(table, ids).unwrap();
        let y = tape.linear(e, w).unwrap();
        let ((), peak, _) = measure(&budget, || tape.backward(y).unwrap());
        assert_eq!(tape.grad(table).is_some(), !frozen);
        peak
    };
    assert!(peak_of(false) >= table_bytes);
    assert!(peak_of(true) < table_bytes);
}

#[test]
fn a_frozen_leaf_inside_a_checkpoint_body_is_refused() {
    let budget = Budget::new(BIG);
    let mut tape = Tape::new(CpuBackend::new(budget.clone()));
    let x = tape.leaf(t(&budget, 1, &[2, 4])).unwrap();
    let w = t(&budget, 2, &[3, 4]);
    let refused = tape.checkpoint(|tp| {
        let f = tp.frozen_leaf(w)?;
        Ok(vec![tp.linear(x, f)?])
    });
    assert!(refused.is_err());
}

#[test]
fn a_checkpointed_segment_over_a_frozen_weight_matches_the_direct_walk() {
    let budget = Budget::new(BIG);
    let direct = {
        let mut tape = Tape::new(CpuBackend::new(budget.clone()));
        let x = tape.leaf(t(&budget, 1, &[2, 4])).unwrap();
        let w = tape.frozen_leaf(t(&budget, 2, &[4, 4])).unwrap();
        let y = tape.linear(x, w).unwrap();
        tape.backward(y).unwrap();
        bits(tape.grad(x).unwrap())
    };
    let mut tape = Tape::new(CpuBackend::new(budget.clone()));
    let x = tape.leaf(t(&budget, 1, &[2, 4])).unwrap();
    let w = tape.frozen_leaf(t(&budget, 2, &[4, 4])).unwrap();
    let y = tape.checkpoint(|tp| Ok(vec![tp.linear(x, w)?])).unwrap()[0];
    tape.backward(y).unwrap();
    assert_eq!(bits(tape.grad(x).unwrap()), direct);
    assert!(tape.grad(w).is_none());
}
