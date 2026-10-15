//! Adversarial tests for `Tape::frozen_leaf` and
//! `Backend::linear_backward_input` (card ft-6fb212ce). A test that exposes a
//! real break is `#[ignore = "BREAK: ..."]`; the rest are regression tests.

mod common;

use common::{bits, data};
use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, CeChunk, Numerics, Tensor};
use ojas_cpu::CpuBackend;

const BIG: u64 = 1 << 30;

fn t(budget: &Budget, seed: u32, shape: &[usize]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&data(seed, n), shape, budget).unwrap()
}

fn tape(budget: &Budget) -> Tape<CpuBackend> {
    Tape::new(CpuBackend::new(budget.clone()))
}

fn peak_of<R>(budget: &Budget, f: impl FnOnce() -> R) -> (R, u64) {
    let before = budget.live_bytes().unwrap();
    budget.reset_peak();
    let out = f();
    (out, budget.peak_bytes().saturating_sub(before))
}

/// dX of the dX-only path is bit-identical to `linear_backward`'s at shapes
/// that miss every tile edge, under both numerics.
#[test]
fn dx_only_is_bit_identical_at_awkward_shapes() {
    let shapes = [
        (1, 1, 1),
        (1, 1, 7),
        (1, 7, 1),
        (2, 1, 1),
        (1, 513, 1),
        (1, 2048, 2048),
        (1, 2049, 17),
        (3, 31, 5),
        (4, 65, 33),
        (17, 129, 63),
        (255, 257, 129),
        (64, 2048, 1024),
        (512, 1024, 2048),
    ];
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for (rows, kin, nout) in shapes {
            let budget = Budget::new(BIG);
            let be = CpuBackend::new(budget.clone()).with_numerics(numerics);
            let x = t(&budget, 11, &[rows, kin]);
            let w = t(&budget, 12, &[nout, kin]);
            let g = t(&budget, 13, &[rows, nout]);
            let (full, _) = be.linear_backward(&x, &w, &g).unwrap();
            let only = be.linear_backward_input(&x, &w, &g).unwrap();
            assert_eq!(only.shape(), full.shape());
            assert_eq!(bits(&only), bits(&full), "{numerics:?} {rows}x{kin}x{nout}");
        }
    }
}

/// The dX-only path must give the same bits whether the input is f32 or
/// holds values the full path would treat identically; `input` is not read.
#[test]
fn dx_only_does_not_depend_on_input_values() {
    let budget = Budget::new(BIG);
    let be = CpuBackend::new(budget.clone());
    let w = t(&budget, 2, &[9, 6]);
    let g = t(&budget, 3, &[4, 9]);
    let a = be
        .linear_backward_input(&t(&budget, 1, &[4, 6]), &w, &g)
        .unwrap();
    let b = be
        .linear_backward_input(&t(&budget, 99, &[4, 6]), &w, &g)
        .unwrap();
    assert_eq!(bits(&a), bits(&b));
}

/// An input that holds NaN: the full path refuses it, so does dX-only
/// (a frozen weight must not make a poisoned activation pass silently).
#[test]
fn dx_only_with_a_nan_input_matches_the_full_paths_verdict() {
    let budget = Budget::new(BIG);
    let be = CpuBackend::new(budget.clone());
    let mut xs = data(1, 4 * 6);
    xs[3] = f32::NAN;
    let x = Tensor::from_f32(&xs, &[4, 6], &budget).unwrap();
    let w = t(&budget, 2, &[9, 6]);
    let g = t(&budget, 3, &[4, 9]);
    let full = be.linear_backward(&x, &w, &g).is_err();
    let only = be.linear_backward_input(&x, &w, &g).is_err();
    assert_eq!(full, only, "full path errs: {full}, dX-only errs: {only}");
}

/// A weight with inf: both must refuse, or both succeed with equal bits.
#[test]
fn dx_only_with_a_non_finite_weight_matches_the_full_path() {
    let budget = Budget::new(BIG);
    let be = CpuBackend::new(budget.clone());
    let x = t(&budget, 1, &[4, 6]);
    let mut ws = data(2, 9 * 6);
    ws[7] = f32::INFINITY;
    let w = Tensor::from_f32(&ws, &[9, 6], &budget).unwrap();
    let g = t(&budget, 3, &[4, 9]);
    match (
        be.linear_backward(&x, &w, &g),
        be.linear_backward_input(&x, &w, &g),
    ) {
        (Ok((a, _)), Ok(b)) => assert_eq!(bits(&a), bits(&b)),
        (Err(_), Err(_)) => {}
        (f, o) => panic!(
            "verdicts differ: full ok={} only ok={}",
            f.is_ok(),
            o.is_ok()
        ),
    }
}

/// A budget that fits dX-only but not the full path: the frozen path must
/// run, and the full path must refuse. Pins "no dW allocation" as a hard
/// bound, not a relative peak.
#[test]
fn dx_only_runs_in_a_budget_too_small_for_dw() {
    let (rows, kin, nout) = (2usize, 256usize, 512usize);
    let setup = Budget::new(BIG);
    let x = t(&setup, 1, &[rows, kin]);
    let w = t(&setup, 2, &[nout, kin]);
    let g = t(&setup, 3, &[rows, nout]);
    // The dX-only peak on the backend's own (otherwise empty) budget.
    let probe = Budget::new(BIG);
    let be = CpuBackend::new(probe.clone());
    let (r, peak) = peak_of(&probe, || be.linear_backward_input(&x, &w, &g));
    r.unwrap();
    let dw = (nout * kin * 4) as u64;
    // A cap of exactly that peak: dX-only fits, the full path (which also
    // holds dW) cannot.
    let tight = CpuBackend::new(Budget::new(peak));
    assert!(tight.linear_backward_input(&x, &w, &g).is_ok());
    assert!(
        tight.linear_backward(&x, &w, &g).is_err(),
        "full path must not fit in the dX-only peak {peak} (dW is {dw})"
    );
}

/// backward twice gives bit-identical gradients, with frozen leaves still
/// absent.
#[test]
fn backward_twice_is_stable_and_frozen_stays_absent() {
    let budget = Budget::new(BIG);
    let mut tp = tape(&budget);
    let x = tp.leaf(t(&budget, 1, &[3, 8])).unwrap();
    let w = tp.frozen_leaf(t(&budget, 2, &[8, 8])).unwrap();
    let w2 = tp.leaf(t(&budget, 3, &[4, 8])).unwrap();
    let h = tp.linear(x, w).unwrap();
    let y = tp.linear(h, w2).unwrap();
    tp.backward(y).unwrap();
    let (gx, g2) = (bits(tp.grad(x).unwrap()), bits(tp.grad(w2).unwrap()));
    tp.backward(y).unwrap();
    assert_eq!(bits(tp.grad(x).unwrap()), gx);
    assert_eq!(bits(tp.grad(w2).unwrap()), g2);
    assert!(tp.grad(w).is_none());
    // take_grad empties; a third backward refills only the trainable ones.
    assert!(tp.take_grad(w).is_none());
    assert!(tp.take_grad(x).is_some());
    assert!(tp.take_grad(x).is_none());
    tp.backward(y).unwrap();
    assert!(tp.grad(x).is_some() && tp.grad(w).is_none());
}

/// backward on a frozen node itself, with a trainable leaf elsewhere on the
/// tape: no error, and no gradient left on the frozen node.
#[test]
fn backward_from_a_frozen_root_leaves_no_gradient() {
    let budget = Budget::new(BIG);
    let mut tp = tape(&budget);
    let live = tp.leaf(t(&budget, 1, &[2, 2])).unwrap();
    let f = tp.frozen_leaf(t(&budget, 2, &[2, 2])).unwrap();
    tp.backward(f).unwrap();
    assert!(tp.grad(f).is_none());
    assert!(tp.grad(live).is_none());
}

/// A frozen leaf through reshape and permute into mul/add/rms with a
/// trainable partner: the trainable gradients equal an all-trainable run's.
#[test]
fn trainable_gradients_ignore_whether_the_partner_is_frozen() {
    let run = |frozen: bool| {
        let budget = Budget::new(BIG);
        let mut tp = tape(&budget);
        let x = tp.leaf(t(&budget, 1, &[4, 6])).unwrap();
        let fv = t(&budget, 2, &[6, 4]);
        let f = if frozen {
            tp.frozen_leaf(fv).unwrap()
        } else {
            tp.leaf(fv).unwrap()
        };
        let p = tp.permute(f, &[1, 0]).unwrap(); // [4, 6]
        let r = tp.reshape(p, &[4, 6]).unwrap();
        let m = tp.mul(x, r).unwrap();
        let a = tp.add(m, x).unwrap();
        let nw = if frozen {
            tp.frozen_leaf(t(&budget, 3, &[6])).unwrap()
        } else {
            tp.leaf(t(&budget, 3, &[6])).unwrap()
        };
        let n = tp.rms_norm(a, nw, 1e-6).unwrap();
        let w = tp.leaf(t(&budget, 4, &[5, 6])).unwrap();
        let y = tp.linear(n, w).unwrap();
        tp.backward(y).unwrap();
        (
            bits(tp.grad(x).unwrap()),
            bits(tp.grad(w).unwrap()),
            [f, p, r, nw].map(|v| tp.grad(v).is_some()),
        )
    };
    let (gx_t, gw_t, has_t) = run(false);
    let (gx_f, gw_f, has_f) = run(true);
    assert_eq!(gx_t, gx_f);
    assert_eq!(gw_t, gw_f);
    assert!(has_t[0]);
    // f, p (permute), r (reshape of permute) and the norm weight.
    assert_eq!(has_f, [false, false, false, false]);
}

/// A frozen embedding table feeding a frozen head through fused CE with a
/// trainable adapter in between: gradients of the trainable leaf match the
/// all-trainable run, the frozen ones are absent.
#[test]
fn frozen_embedding_and_fused_head_leave_the_trainable_gradient_unchanged() {
    let (n, d, v) = (6usize, 8usize, 20usize);
    let run = |frozen: bool| {
        let budget = Budget::new(BIG);
        let mut tp = tape(&budget);
        let mk = |tp: &mut Tape<CpuBackend>, seed, shape: &[usize]| {
            let value = t(&budget, seed, shape);
            if frozen {
                tp.frozen_leaf(value).unwrap()
            } else {
                tp.leaf(value).unwrap()
            }
        };
        let table = mk(&mut tp, 1, &[v, d]);
        let head = mk(&mut tp, 2, &[v, d]);
        let adapter = tp.leaf(t(&budget, 3, &[d, d])).unwrap();
        let ids: Vec<u32> = (0..n).map(|i| ((i * 3) % v) as u32).collect();
        let e = tp
            .embedding(table, Tensor::from_u32(&ids, &[n], &budget).unwrap())
            .unwrap();
        let x = tp.linear(e, adapter).unwrap();
        let tg: Vec<u32> = (0..n).map(|i| ((i * 5 + 1) % v) as u32).collect();
        let loss = tp
            .linear_cross_entropy(
                x,
                head,
                Tensor::from_u32(&tg, &[n], &budget).unwrap(),
                None,
                CeChunk { rows: 2, cols: 7 },
            )
            .unwrap();
        tp.backward(loss).unwrap();
        (
            bits(tp.grad(adapter).unwrap()),
            tp.grad(table).is_some(),
            tp.grad(head).is_some(),
        )
    };
    let (a_t, table_t, head_t) = run(false);
    let (a_f, table_f, head_f) = run(true);
    assert_eq!(a_t, a_f);
    assert!(table_t && head_t);
    assert!(!table_f && !head_f);
}

/// A frozen head in the fused cross entropy must not allocate a head-sized
/// gradient: the tied Qwen3.5 lm_head is vocab x hidden (about 2 GiB at
/// 248k x 2048 in f32) and is always frozen under LoRA.
#[test]
fn a_frozen_fused_head_allocates_no_head_sized_gradient() {
    let (n, d, v) = (4usize, 64usize, 8192usize);
    let head_bytes = (v * d * 4) as u64;
    let peak = |frozen: bool| {
        let budget = Budget::new(BIG);
        let mut tp = tape(&budget);
        let head = if frozen {
            tp.frozen_leaf(t(&budget, 2, &[v, d])).unwrap()
        } else {
            tp.leaf(t(&budget, 2, &[v, d])).unwrap()
        };
        let x = tp.leaf(t(&budget, 3, &[n, d])).unwrap();
        let tg: Vec<u32> = (0..n).map(|i| ((i * 5 + 1) % v) as u32).collect();
        let (loss, fwd_peak) = peak_of(&budget, || {
            tp.linear_cross_entropy(
                x,
                head,
                Tensor::from_u32(&tg, &[n], &budget).unwrap(),
                None,
                CeChunk {
                    rows: 2,
                    cols: 1024,
                },
            )
            .unwrap()
        });
        let (_, bwd_peak) = peak_of(&budget, || tp.backward(loss).unwrap());
        fwd_peak.max(bwd_peak)
    };
    let live = peak(false);
    let frozen = peak(true);
    assert!(live >= head_bytes);
    assert!(
        frozen < head_bytes,
        "frozen head peak {frozen} B is not below one head-sized gradient {head_bytes} B"
    );
}

/// A fused cross entropy with a frozen *input* and trainable head still
/// trains the head.
#[test]
fn frozen_input_to_fused_ce_keeps_the_head_gradient() {
    let budget = Budget::new(BIG);
    let (n, d, v) = (4usize, 8usize, 12usize);
    let mut tp = tape(&budget);
    let x = tp.frozen_leaf(t(&budget, 3, &[n, d])).unwrap();
    let head = tp.leaf(t(&budget, 2, &[v, d])).unwrap();
    let tg: Vec<u32> = (0..n).map(|i| (i % v) as u32).collect();
    let loss = tp
        .linear_cross_entropy(
            x,
            head,
            Tensor::from_u32(&tg, &[n], &budget).unwrap(),
            None,
            CeChunk { rows: 2, cols: 5 },
        )
        .unwrap();
    tp.backward(loss).unwrap();
    assert!(tp.grad(head).is_some());
    assert!(tp.grad(x).is_none());
}

/// Trainable AND frozen views of the very same tensor allocation: freezing
/// one record must not strip or alias the other's gradient.
#[test]
fn frozen_and_trainable_records_of_one_allocation_stay_separate() {
    let budget = Budget::new(BIG);
    let shared = t(&budget, 2, &[6, 6]);
    let mut tp = tape(&budget);
    let x = tp.leaf(t(&budget, 1, &[3, 6])).unwrap();
    let fz = tp.frozen_leaf(shared.clone()).unwrap();
    let lv = tp.leaf(shared.clone()).unwrap();
    let a = tp.linear(x, fz).unwrap();
    let b = tp.linear(a, lv).unwrap();
    tp.backward(b).unwrap();
    let g_live = tp.take_grad(lv).expect("trainable twin has a gradient");
    // A gradient taken from the twin must not have written through the
    // shared allocation.
    assert_eq!(bits(&shared), bits(&t(&budget, 2, &[6, 6])));
    assert!(tp.take_grad(fz).is_none());
    assert_eq!(g_live.shape(), &[6, 6]);
}

/// A frozen leaf whose gradient would be shared (Add hands one gradient to
/// both inputs): with one trainable and one frozen input of the add, the
/// trainable leaf's gradient must still own its allocation.
#[test]
fn add_of_frozen_and_trainable_hands_out_an_owned_gradient() {
    let budget = Budget::new(BIG);
    let mut tp = tape(&budget);
    let a = tp.leaf(t(&budget, 1, &[3, 3])).unwrap();
    let f = tp.frozen_leaf(t(&budget, 2, &[3, 3])).unwrap();
    let s = tp.add(a, f).unwrap();
    tp.backward(s).unwrap();
    let ga = tp.take_grad(a).unwrap();
    assert!(!ga.shares_allocation());
    assert!(bits(&ga).iter().all(|&b| b == 1.0f32.to_bits()));
    assert!(tp.take_grad(f).is_none());
}

/// A graph whose only leaves are frozen, with a checkpoint segment: not
/// an error (the "no allocation" promise excludes checkpoints, which is
/// the documented caveat).
#[test]
fn frozen_only_graph_with_a_checkpoint_does_not_error() {
    let budget = Budget::new(BIG);
    let mut tp = tape(&budget);
    let x = tp.frozen_leaf(t(&budget, 1, &[2, 4])).unwrap();
    let w = tp.frozen_leaf(t(&budget, 2, &[4, 4])).unwrap();
    let y = tp.checkpoint(|b| Ok(vec![b.linear(x, w)?])).unwrap()[0];
    tp.backward(y).unwrap();
    assert!(tp.grad(x).is_none() && tp.grad(w).is_none());
}

/// Out-of-range and stale vars on grad/take_grad/backward are not panics.
#[test]
fn grads_on_unknown_vars_are_none_not_panics() {
    let budget = Budget::new(BIG);
    let mut tp = tape(&budget);
    let f = tp.frozen_leaf(t(&budget, 1, &[2, 2])).unwrap();
    tp.clear();
    assert!(tp.grad(f).is_none());
    assert!(tp.take_grad(f).is_none());
    assert!(tp.backward(f).is_err());
}

/// Gradient-seeded backward (1/K) on a frozen/trainable mix scales the
/// trainable gradient exactly and still has no frozen gradient.
#[test]
fn seeded_backward_with_a_frozen_weight_scales_exactly() {
    let budget = Budget::new(BIG);
    let run = |seed: f32| {
        let mut tp = tape(&budget);
        let x = tp.leaf(t(&budget, 1, &[3, 8])).unwrap();
        let w = tp.frozen_leaf(t(&budget, 2, &[8, 8])).unwrap();
        let y = tp.linear(x, w).unwrap();
        tp.backward_seeded(y, seed).unwrap();
        assert!(tp.grad(w).is_none());
        tp.grad(x).unwrap().to_f32_vec().unwrap()
    };
    let one = run(1.0);
    let quarter = run(0.25);
    for (a, b) in one.iter().zip(&quarter) {
        assert_eq!((a * 0.25).to_bits(), b.to_bits());
    }
}

/// A backend that does not override `linear_backward_input` (the `Resident`
/// doubles implement only `linear_backward`) runs the trait default: a
/// frozen weight gives the trainable path's dX bit for bit and no weight
/// gradient is kept, on both the host and the device placement path.
#[test]
fn a_backend_without_the_override_runs_the_trait_default_correctly() {
    use common::Resident;
    let budget = Budget::new(BIG);
    for device in [false, true] {
        let run = |frozen: bool| {
            let backend = if device {
                Resident::new()
            } else {
                Resident::host()
            };
            let mut tp = Tape::new(backend);
            let x = tp.leaf(t(&budget, 1, &[5, 12])).unwrap();
            let wv = t(&budget, 2, &[7, 12]);
            let w = if frozen {
                tp.frozen_leaf(wv).unwrap()
            } else {
                tp.leaf(wv).unwrap()
            };
            let y = tp.linear(x, w).unwrap();
            tp.backward(y).unwrap();
            let dx = tp.grad(x).unwrap().to_host(&budget).unwrap();
            (bits(&dx), tp.grad(w).is_some())
        };
        let (live, live_has) = run(false);
        let (frozen, frozen_has) = run(true);
        assert_eq!(live, frozen, "device={device}");
        assert!(live_has && !frozen_has, "device={device}");
    }
}
