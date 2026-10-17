//! `Tape::linear_cross_entropy_value`: the fused cross entropy's value with
//! nothing recorded and no gradient asked for (card ft-b5e64557, the
//! report-only full-vocabulary loss).

mod common;

use common::{bits, data};
use ojas_autograd::{Tape, Var};
use ojas_core::{Budget, CeChunk, Tensor};
use ojas_cpu::CpuBackend;

const BIG: u64 = 1 << 30;
const CHUNK: CeChunk = CeChunk {
    rows: 2,
    cols: 1024,
};

fn t(budget: &Budget, seed: u32, shape: &[usize]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&data(seed, n), shape, budget).unwrap()
}

fn tape(budget: &Budget) -> Tape<CpuBackend> {
    Tape::new(CpuBackend::new(budget.clone()))
}

fn targets(budget: &Budget, n: usize, v: usize) -> Tensor {
    let tg: Vec<u32> = (0..n).map(|i| ((i * 5 + 1) % v) as u32).collect();
    Tensor::from_u32(&tg, &[n], budget).unwrap()
}

/// Bytes the budget's peak rose above its level before `f`.
fn peak_of<R>(budget: &Budget, f: impl FnOnce() -> R) -> (R, u64) {
    let before = budget.live_bytes().unwrap();
    budget.reset_peak();
    let out = f();
    (out, budget.peak_bytes().saturating_sub(before))
}

#[test]
fn the_value_equals_the_recorded_loss_bit_for_bit() {
    let (n, d, v) = (5usize, 8usize, 37usize);
    for head_frozen in [false, true] {
        for ignore in [None, Some(1u32)] {
            let budget = Budget::new(BIG);
            let mut tp = tape(&budget);
            let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
            let w = if head_frozen {
                tp.frozen_leaf(t(&budget, 2, &[v, d])).unwrap()
            } else {
                tp.leaf(t(&budget, 2, &[v, d])).unwrap()
            };
            let tg = targets(&budget, n, v);
            let value = tp
                .linear_cross_entropy_value(x, w, &tg, ignore, CHUNK)
                .unwrap();
            let recorded = tp
                .linear_cross_entropy(x, w, tg.clone(), ignore, CHUNK)
                .unwrap();
            assert_eq!(
                bits(&value),
                bits(tp.value(recorded).unwrap()),
                "frozen={head_frozen} ignore={ignore:?}"
            );
        }
    }
}

#[test]
fn nothing_is_recorded_and_backward_on_other_losses_is_unaffected() {
    let (n, d, v) = (4usize, 8usize, 12usize);
    let run = |with_value: bool| {
        let budget = Budget::new(BIG);
        let mut tp = tape(&budget);
        let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
        let w = tp.leaf(t(&budget, 2, &[v, d])).unwrap();
        let tg = targets(&budget, n, v);
        let before = tp.leaf(t(&budget, 9, &[1, 1])).unwrap();
        if with_value {
            tp.linear_cross_entropy_value(x, w, &tg, None, CHUNK)
                .unwrap();
        }
        let after = tp.leaf(t(&budget, 9, &[1, 1])).unwrap();
        // The tape grew by exactly the one leaf: no node was pushed.
        assert_eq!(after, Var(before.0 + 1), "with_value={with_value}");
        let loss = tp.linear_cross_entropy(x, w, tg, None, CHUNK).unwrap();
        tp.backward(loss).unwrap();
        (bits(tp.grad(x).unwrap()), bits(tp.grad(w).unwrap()))
    };
    assert_eq!(run(false), run(true));
}

/// The value call charges neither a head-sized gradient (trainable head) nor
/// an input-sized one (trainable input), where the recording path pays both.
#[test]
fn the_value_call_allocates_no_gradient() {
    let (n, d, v) = (64usize, 64usize, 8192usize);
    let head_bytes = (v * d * 4) as u64;
    let input_bytes = (n * d * 4) as u64;
    let peak = |value_only: bool, head_frozen: bool| {
        let budget = Budget::new(BIG);
        let mut tp = tape(&budget);
        let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
        let w = if head_frozen {
            tp.frozen_leaf(t(&budget, 2, &[v, d])).unwrap()
        } else {
            tp.leaf(t(&budget, 2, &[v, d])).unwrap()
        };
        let tg = targets(&budget, n, v);
        if value_only {
            let (_, p) = peak_of(&budget, || {
                tp.linear_cross_entropy_value(x, w, &tg, None, CHUNK)
                    .unwrap()
            });
            p
        } else {
            let (loss, fwd) = peak_of(&budget, || {
                tp.linear_cross_entropy(x, w, tg.clone(), None, CHUNK)
                    .unwrap()
            });
            let (_, bwd) = peak_of(&budget, || tp.backward(loss).unwrap());
            fwd.max(bwd)
        }
    };
    // Trainable head: recording pays a [V, d] gradient, the value call not.
    assert!(peak(false, false) >= head_bytes);
    assert!(peak(true, false) < head_bytes);
    // Frozen head, trainable input: recording still pays [N, d]; the value
    // call pays less than that on top of what the loss itself needs.
    assert!(
        peak(true, true) + input_bytes <= peak(false, true),
        "value-only {} B, recording {} B, input gradient {input_bytes} B",
        peak(true, true),
        peak(false, true)
    );
}

#[test]
fn a_variable_off_the_tape_and_a_bad_target_are_refused() {
    let (n, d, v) = (3usize, 4usize, 6usize);
    let budget = Budget::new(BIG);
    let mut tp = tape(&budget);
    let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
    let w = tp.frozen_leaf(t(&budget, 2, &[v, d])).unwrap();
    let tg = targets(&budget, n, v);
    assert!(tp
        .linear_cross_entropy_value(Var(99), w, &tg, None, CHUNK)
        .is_err());
    let out_of_range = Tensor::from_u32(&[0, 1, v as u32], &[n], &budget).unwrap();
    assert!(tp
        .linear_cross_entropy_value(x, w, &out_of_range, None, CHUNK)
        .is_err());
}

/// Every refusal of the recording path is the value path's refusal, and a
/// refused value call leaves the tape and the budget as they were.
#[test]
fn refusals_match_the_recording_path_and_leak_nothing() {
    let (n, d, v) = (3usize, 4usize, 6usize);
    let budget = Budget::new(BIG);
    let mut tp = tape(&budget);
    let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
    let w = tp.frozen_leaf(t(&budget, 2, &[v, d])).unwrap();
    let all_ignored = Tensor::from_u32(&[2, 2, 2], &[n], &budget).unwrap();
    let wrong_dtype = Tensor::from_f32(&[0.0, 1.0, 2.0], &[n], &budget).unwrap();
    let wrong_len = Tensor::from_u32(&[0, 1], &[2], &budget).unwrap();
    let cases: [(&Tensor, Option<u32>); 3] = [
        (&all_ignored, Some(2)),
        (&wrong_dtype, None),
        (&wrong_len, None),
    ];
    for (i, (tg, ignore)) in cases.into_iter().enumerate() {
        let live = budget.live_bytes().unwrap();
        let len = tp.leaf(t(&budget, 9, &[1, 1])).unwrap().0;
        let value_err = tp
            .linear_cross_entropy_value(x, w, tg, ignore, CHUNK)
            .expect_err("value path must refuse");
        let record_err = tp
            .linear_cross_entropy(x, w, tg.clone(), ignore, CHUNK)
            .expect_err("recording path must refuse");
        assert_eq!(
            std::mem::discriminant(&value_err),
            std::mem::discriminant(&record_err),
            "case {i}: {value_err:?} vs {record_err:?}"
        );
        // Nothing was pushed by either refusal.
        assert_eq!(
            tp.leaf(t(&budget, 9, &[1, 1])).unwrap().0,
            len + 1,
            "case {i}"
        );
        // The only bytes still live are the two 1x1 leaves just made.
        assert_eq!(budget.live_bytes().unwrap(), live + 8, "case {i}");
    }
}

/// Chunk sizes of 1 and larger than the vocabulary, and a call made inside a
/// checkpointed body, give the recording path's loss and gradients.
#[test]
fn chunk_extremes_and_a_call_inside_a_checkpoint_agree() {
    let (n, d, v) = (5usize, 8usize, 37usize);
    for chunk in [
        CeChunk { rows: 1, cols: 1 },
        CeChunk {
            rows: 1000,
            cols: 100_000,
        },
    ] {
        let budget = Budget::new(BIG);
        let mut tp = tape(&budget);
        let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
        let w = tp.leaf(t(&budget, 2, &[v, d])).unwrap();
        let tg = targets(&budget, n, v);
        let value = tp
            .linear_cross_entropy_value(x, w, &tg, None, chunk)
            .unwrap();
        let recorded = tp
            .linear_cross_entropy(x, w, tg.clone(), None, chunk)
            .unwrap();
        assert_eq!(bits(&value), bits(tp.value(recorded).unwrap()));
    }

    let run = |segmented: bool| {
        let budget = Budget::new(BIG);
        let mut tp = tape(&budget);
        let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
        let w = tp.leaf(t(&budget, 2, &[v, d])).unwrap();
        let tg = targets(&budget, n, v);
        let loss = if segmented {
            let tgc = tg.clone();
            tp.checkpoint(|s| {
                s.linear_cross_entropy_value(x, w, &tgc, None, CHUNK)?;
                Ok(vec![s.linear_cross_entropy(x, w, tgc, None, CHUNK)?])
            })
            .unwrap()[0]
        } else {
            tp.linear_cross_entropy(x, w, tg, None, CHUNK).unwrap()
        };
        tp.backward(loss).unwrap();
        (bits(tp.grad(x).unwrap()), bits(tp.grad(w).unwrap()))
    };
    assert_eq!(run(false), run(true));
}
