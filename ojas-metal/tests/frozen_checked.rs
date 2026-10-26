//! `MetalBackend::freeze_checked`: a frozen parameter is scanned for NaN and
//! infinity once, at load, and not again by every matmul, gather or fused
//! cross-entropy that reads it (card ft-2bbced18, criterion 2).
//!
//! The proof is a dispatch count: the scan is one `ojas_check_finite`
//! dispatch per operand per op, so an op over a frozen weight encodes exactly
//! one dispatch fewer than the same op over an identical unfrozen weight, and
//! its bits are the same. The mark's invariants are tested beside it: every
//! op that writes a buffer refuses a frozen one and leaves it untouched, and
//! a frozen buffer that was not finite is reported at the next sync and then
//! scanned by every later op as before.
//!
//! Needs a Metal device; run under the mac_heavy lock.
#![cfg(target_os = "macos")]

mod common;

use common::{deferred, down, host, host_u32, ids, metal, ok, rand, up};
use ojas_core::{AdamWConfig, Backend, CeChunk, CeGrads, DType, OjasError, Tensor};
use ojas_metal::MetalBackend;

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// Dispatches `op` encodes, and its result.
fn counted<T>(m: &MetalBackend, op: impl FnOnce() -> Result<T, OjasError>) -> (T, u64) {
    let before = m.dispatches();
    let out = ok("counted op", op());
    (out, m.dispatches() - before)
}

/// An uploaded weight and an identical twin, the twin frozen.
fn twins(m: &MetalBackend, shape: &[usize], seed: u64, bf16: bool) -> (Tensor, Tensor) {
    let host = rand(shape, seed, 1.0);
    let make = |m: &MetalBackend| {
        let t = up(m, &host);
        if bf16 {
            ok("to_bf16", m.to_bf16(&t))
        } else {
            t
        }
    };
    let (live, frozen) = (make(m), make(m));
    ok("freeze", m.freeze_checked(&frozen));
    ok("sync", m.sync());
    (live, frozen)
}

#[test]
fn freezing_is_one_scan_and_a_linear_over_a_frozen_weight_skips_it() {
    let m = metal();
    let (rows, kin, nout) = (17usize, 129usize, 63usize);
    for bf16 in [false, true] {
        let x = up(&m, &rand(&[rows, kin], 1, 1.0));
        let g = up(&m, &rand(&[rows, nout], 3, 1.0));
        let host = rand(&[nout, kin], 2, 1.0);
        let mk = |m: &MetalBackend| {
            let t = up(m, &host);
            if bf16 {
                ok("to_bf16", m.to_bf16(&t))
            } else {
                t
            }
        };
        let (live, frozen) = (mk(&m), mk(&m));
        ok("sync", m.sync());
        let (_, scan) = counted(&m, || m.freeze_checked(&frozen));
        assert_eq!(scan, 1, "bf16={bf16}: freezing is one scan of the weight");

        let (y_live, d_live) = counted(&m, || m.linear_forward(&x, &live));
        let (y_frozen, d_frozen) = counted(&m, || m.linear_forward(&x, &frozen));
        assert_eq!(d_live, d_frozen + 1, "bf16={bf16}: forward skips one scan");
        // A second use is no dearer than the first: nothing is rescanned.
        let (_, d_again) = counted(&m, || m.linear_forward(&x, &frozen));
        assert_eq!(d_again, d_frozen, "bf16={bf16}: a repeat use rescans");
        ok("sync", m.sync());
        assert_eq!(bits(&y_live), bits(&y_frozen), "bf16={bf16}: forward bits");

        let (dx_live, d_live) = counted(&m, || m.linear_backward_input(&x, &live, &g));
        let (dx_frozen, d_frozen) = counted(&m, || m.linear_backward_input(&x, &frozen, &g));
        assert_eq!(d_live, d_frozen + 1, "bf16={bf16}: dX skips one scan");
        ok("sync", m.sync());
        assert_eq!(bits(&dx_live), bits(&dx_frozen), "bf16={bf16}: dX bits");
    }
}

#[test]
fn a_frozen_bf16_table_skips_its_scan_in_the_gather_and_the_fused_head() {
    let m = metal();
    let (n, d, v) = (8usize, 32usize, 64usize);
    let (live, frozen) = twins(&m, &[v, d], 5, true);
    let tokens = up(&m, &host_u32(&ids(n, 9, v as u32), &[n]));
    let x = up(&m, &rand(&[n, d], 6, 1.0));

    let (e_live, d_live) = counted(&m, || m.embedding_forward(&live, &tokens));
    let (e_frozen, d_frozen) = counted(&m, || m.embedding_forward(&frozen, &tokens));
    assert_eq!(d_live, d_frozen + 1, "the gather skips the table's scan");
    ok("sync", m.sync());
    assert_eq!(bits(&e_live), bits(&e_frozen));

    let chunk = CeChunk { rows: n, cols: v };
    let head_only_input = CeGrads {
        input: true,
        weight: false,
    };
    let (l_live, d_live) = counted(&m, || {
        m.linear_cross_entropy_mean(&x, &live, &tokens, None, chunk, head_only_input)
    });
    let (l_frozen, d_frozen) = counted(&m, || {
        m.linear_cross_entropy_mean(&x, &frozen, &tokens, None, chunk, head_only_input)
    });
    assert_eq!(
        d_live,
        d_frozen + 1,
        "the fused head skips the table's scan"
    );
    ok("sync", m.sync());
    assert_eq!(bits(&l_live.loss), bits(&l_frozen.loss));
    assert_eq!(
        bits(l_live.grad_input.as_ref().expect("grad_input")),
        bits(l_frozen.grad_input.as_ref().expect("grad_input"))
    );
    assert!(l_frozen.grad_weight.is_none(), "no [V, d] gradient exists");
}

#[test]
fn every_op_that_writes_a_buffer_refuses_a_frozen_one_and_leaves_it_alone() {
    let m = metal();
    let shape = [4usize, 8];
    let mut p = up(&m, &rand(&shape, 1, 1.0));
    let g = up(&m, &rand(&shape, 2, 1.0));
    let mut m1 = up(&m, &rand(&shape, 3, 0.1));
    let mut m2 = up(&m, &rand(&shape, 4, 0.1));
    ok("freeze", m.freeze_checked(&p));
    ok("sync", m.sync());
    let before = bits(&p);
    let refused = |what: &str, r: Result<(), OjasError>| match r {
        Err(OjasError::Backend { detail, .. }) if detail.contains("frozen") => {}
        other => panic!("{what}: expected a frozen-parameter refusal, got {other:?}"),
    };

    let cfg = AdamWConfig::nanolab(1e-3, 0.0);
    refused(
        "adamw p",
        m.adamw_step(&mut p, &g, &mut m1, &mut m2, 0, cfg),
    );
    let mut p2 = up(&m, &rand(&shape, 5, 1.0));
    let mut frozen_moment = up(&m, &rand(&shape, 6, 0.1));
    ok("freeze moment", m.freeze_checked(&frozen_moment));
    refused(
        "adamw moment",
        m.adamw_step(&mut p2, &g, &mut frozen_moment, &mut m2, 0, cfg),
    );
    refused("accumulate_grad", m.accumulate_grad(&mut p, &g));
    refused("scale_grad", m.scale_grad(&mut p, 0.5));

    ok("sync", m.sync());
    assert_eq!(bits(&p), before, "a refused write leaves the weight alone");
    // The refusals are typed errors, not a poisoned backend.
    let x = up(&m, &rand(&[2, 8], 7, 1.0));
    ok("still usable", m.linear_forward(&x, &p));
    ok("sync", m.sync());
}

#[test]
fn a_frozen_weight_that_is_not_finite_is_reported_and_scanned_again() {
    let m = metal();
    let (rows, kin, nout) = (3usize, 8usize, 5usize);
    let mut data = vec![0.5f32; nout * kin];
    data[11] = f32::NAN;
    let w = up(&m, &host(&data, &[nout, kin]));
    ok("freeze records, it does not refuse", m.freeze_checked(&w));
    match m.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "freeze_checked"),
        other => panic!("sync must name the freeze: {other:?}"),
    }
    ok("the fault is reported once", m.sync());
    // The mark is withdrawn: the next op scans the weight and faults itself.
    let x = up(&m, &rand(&[rows, kin], 1, 1.0));
    let r = m.linear_forward(&x, &w);
    deferred(
        &m,
        "forward over a non-finite frozen weight",
        r,
        "linear_forward",
    );
}

#[test]
fn only_float_tensors_can_be_frozen() {
    let m = metal();
    let tokens = up(&m, &host_u32(&[1, 2, 3], &[3]));
    match m.freeze_checked(&tokens) {
        Err(OjasError::Dtype {
            expected: DType::F32,
            got: DType::U32,
            ..
        }) => {}
        other => panic!("a U32 tensor is not a parameter: {other:?}"),
    }
}

#[test]
fn a_widened_copy_of_a_frozen_weight_is_the_callers_to_train_and_is_scanned_again() {
    let m = metal();
    let shape = [4usize, 8];
    let (_, frozen) = twins(&m, &shape, 11, true);
    // `to_f32` is public: its result is a new buffer the caller owns.
    let mut p = ok("to_f32", m.to_f32(&frozen));
    let g = up(&m, &rand(&shape, 12, 1.0));
    let zeros = host(&[0.0; 32], &shape);
    let (mut m1, mut m2) = (up(&m, &zeros), up(&m, &zeros));
    ok("sync", m.sync());
    let before = bits(&p);
    ok(
        "a widened copy is not a frozen parameter",
        m.adamw_step(
            &mut p,
            &g,
            &mut m1,
            &mut m2,
            0,
            AdamWConfig::nanolab(1e-3, 0.0),
        ),
    );
    ok("sync", m.sync());
    assert_ne!(bits(&p), before, "the step moved the copy");
    // The write withdrew the copy's mark: a matmul over it scans it, as over
    // an identical plain upload.
    let plain = up(&m, &host(&down(&p), &shape));
    ok("sync", m.sync());
    let x = up(&m, &rand(&[2, 8], 7, 1.0));
    let (_, d_copy) = counted(&m, || m.linear_forward(&x, &p));
    let (_, d_plain) = counted(&m, || m.linear_forward(&x, &plain));
    assert_eq!(d_copy, d_plain, "a written copy is scanned again");
    ok("sync", m.sync());
}

#[test]
fn freezing_a_frozen_window_again_scans_and_records_nothing() {
    let m = metal();
    let w = up(&m, &rand(&[4, 8], 21, 1.0));
    ok("freeze", m.freeze_checked(&w));
    ok("sync", m.sync());
    let (_, again) = counted(&m, || m.freeze_checked(&w));
    assert_eq!(again, 0, "a repeat freeze rescans");
    // A widened copy frozen on purpose is a frozen parameter: refused writes.
    let (_, bf) = twins(&m, &[4, 8], 22, true);
    let mut copy = ok("to_f32", m.to_f32(&bf));
    ok("freeze the copy", m.freeze_checked(&copy));
    ok("sync", m.sync());
    match m.scale_grad(&mut copy, 0.5) {
        Err(OjasError::Backend { detail, .. }) if detail.contains("frozen") => {}
        other => panic!("a deliberately frozen copy is not written: {other:?}"),
    }
}

#[test]
fn a_frozen_window_marks_only_that_window_of_a_shared_buffer() {
    let m = metal();
    let (kin, rows) = (16usize, 8usize);
    let host_w = rand(&[rows, kin], 31, 1.0);
    let (full_live, full_frozen) = (up(&m, &host_w), up(&m, &host_w));
    let half = rows / 2;
    let head = |t: &Tensor| ok("head view", t.narrow(0, &[half, kin], &[kin, 1]));
    let tail = |t: &Tensor| {
        ok(
            "tail view",
            t.narrow(half * kin * 4, &[half, kin], &[kin, 1]),
        )
    };
    ok("freeze head rows", m.freeze_checked(&head(&full_frozen)));
    ok("sync", m.sync());
    let x = up(&m, &rand(&[3, kin], 32, 1.0));
    let d = |w: &Tensor| counted(&m, || m.linear_forward(&x, w)).1;
    assert_eq!(
        d(&head(&full_live)),
        d(&head(&full_frozen)) + 1,
        "head skips"
    );
    assert_eq!(
        d(&tail(&full_live)),
        d(&tail(&full_frozen)),
        "tail is scanned"
    );
    assert_eq!(
        d(&full_live),
        d(&full_frozen),
        "the whole buffer is scanned"
    );
    ok("sync", m.sync());
}

#[test]
fn ops_that_ran_on_the_mark_before_its_fault_surfaced_do_not_hide_it() {
    let m = metal();
    let mut data = vec![0.5f32; 5 * 8];
    data[3] = f32::INFINITY;
    let w = up(&m, &host(&data, &[5, 8]));
    ok("freeze", m.freeze_checked(&w));
    // No sync: the mark is live while the scan's verdict is still on the device.
    let x = up(&m, &rand(&[3, 8], 1, 1.0));
    ok("queued over the mark", m.linear_forward(&x, &w));
    match m.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "freeze_checked"),
        other => panic!("the freeze's fault comes first: {other:?}"),
    }
    let r = m.linear_forward(&x, &w);
    deferred(&m, "scanned again after the report", r, "linear_forward");
}

fn halved(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| (v * 0.5).to_bits()).collect()
}

#[test]
fn scale_grad_of_a_shared_frozen_grad_replaces_it_and_never_writes_the_buffer() {
    let m = metal();
    let mut p = up(&m, &rand(&[4, 8], 41, 1.0));
    ok("freeze", m.freeze_checked(&p));
    ok("sync", m.sync());
    let before = bits(&p);
    let want = halved(&p);
    let keep = p.clone();
    ok(
        "a shared grad is replaced, not refused",
        m.scale_grad(&mut p, 0.5),
    );
    ok("sync", m.sync());
    assert_eq!(
        bits(&keep),
        before,
        "the shared frozen buffer is not written"
    );
    assert_eq!(bits(&p), want, "the replacement holds the products");
}

#[test]
fn scale_grad_of_a_widened_copy_writes_it_and_withdraws_its_mark() {
    let m = metal();
    let shape = [4usize, 8];
    let (_, frozen) = twins(&m, &shape, 42, true);
    let mut p = ok("to_f32", m.to_f32(&frozen));
    ok("sync", m.sync());
    let want = halved(&p);
    ok(
        "a widened copy is the caller's to scale",
        m.scale_grad(&mut p, 0.5),
    );
    ok("sync", m.sync());
    assert_eq!(bits(&p), want, "the copy was scaled in place");
    let plain = up(&m, &host(&down(&p), &shape));
    ok("sync", m.sync());
    let x = up(&m, &rand(&[2, 8], 7, 1.0));
    let (_, d_copy) = counted(&m, || m.linear_forward(&x, &p));
    let (_, d_plain) = counted(&m, || m.linear_forward(&x, &plain));
    assert_eq!(d_copy, d_plain, "a scaled copy is scanned again");
    ok("sync", m.sync());
}

#[test]
fn scale_grad_refuses_a_non_finite_scale_at_the_call_and_leaves_the_grad() {
    let m = metal();
    let mut g = up(&m, &rand(&[4, 8], 43, 1.0));
    ok("sync", m.sync());
    let before = bits(&g);
    for s in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        match m.scale_grad(&mut g, s) {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "scale_grad"),
            other => panic!("scale {s}: expected NonFinite at the call, got {other:?}"),
        }
    }
    ok("sync", m.sync());
    assert_eq!(bits(&g), before, "a refused scale writes nothing");
}

#[test]
fn scale_grad_overflow_is_deferred_to_the_sync_like_other_in_place_ops() {
    let m = metal();
    let mut g = up(&m, &host(&[3.0e38, 1.0], &[2]));
    ok("sync", m.sync());
    ok(
        "the product is checked on the device",
        m.scale_grad(&mut g, 4.0),
    );
    match m.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "scale_grad"),
        other => panic!("the overflow is reported at sync: {other:?}"),
    }
}
