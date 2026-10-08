//! `accumulate_grad` (T2): `acc += grad` in place, all-or-nothing.
//!
//! The trait default composes `residual_add_forward` and replaces `acc` with
//! a new tensor. The CPU override adds into `acc`'s own allocation when it
//! is uniquely owned, so it needs no scratch and no charge (since
//! 2026-10-02; before, one `acc`-sized scratch) where the default needs a
//! new `acc`-sized tensor.

mod common;

use std::time::Instant;

use common::{assert_capacity, assert_nonfinite, assert_shape, bits, SplitMix64};
use ojas_core::{Backend, Budget, DType, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn f(x: &Tensor) -> Vec<f32> {
    x.to_f32_vec().unwrap()
}

#[test]
fn sum_matches_residual_add_bit_for_bit() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let mut rng = SplitMix64(0xacc1);
    for shape in [vec![1usize], vec![7, 13], vec![768, 768], vec![3, 4, 1025]] {
        let n: usize = shape.iter().product();
        let a = rng.vec(n, 2.0);
        let g = rng.vec(n, 2.0);
        let want = be
            .residual_add_forward(&t(&a, &shape), &t(&g, &shape))
            .unwrap();
        let mut acc = t(&a, &shape);
        be.accumulate_grad(&mut acc, &t(&g, &shape)).unwrap();
        assert_eq!(acc.shape(), shape.as_slice());
        assert_eq!(bits(&f(&acc)), bits(&f(&want)), "{shape:?}");
    }
}

/// A uniquely owned `acc` that is a window into a larger allocation keeps
/// that allocation: the sum is written in place. (The trait default returns
/// a fresh tensor at offset 0 sized to the window.)
#[test]
fn a_unique_acc_is_written_in_place() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let big = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);
    let mut acc = big.view(&[2, 2], &[2, 1], 8).unwrap();
    drop(big);
    let (offset, storage) = (acc.byte_offset(), acc.storage_len());
    be.accumulate_grad(&mut acc, &t(&[0.5, 0.25, -1.0, 2.0], &[2, 2]))
        .unwrap();
    assert_eq!(f(&acc), vec![3.5, 4.25, 4.0, 8.0]);
    assert_eq!((acc.byte_offset(), acc.storage_len()), (offset, storage));
}

/// The override on a uniquely owned `acc` charges nothing (since
/// 2026-10-02): it runs while the default's result holds the whole budget.
/// The composed default (`residual_add`, which reads its operands in place
/// since 2026-10-01) builds a new tensor of exactly `acc`'s size, and so
/// does the override for a shared `acc`, which it cannot write: one
/// `acc`-sized charge, and one f32 less refuses both with every handle
/// unchanged. All paths give the same bits.
#[test]
fn in_place_charges_nothing_and_a_replacement_needs_one_acc_sized_charge() {
    let n = 4096usize;
    let room = 4 * n as u64;
    let be = CpuBackend::new(Budget::new(room));
    let mut acc = t(&vec![1.0; n], &[n]);
    let grad = t(&vec![0.5; n], &[n]);
    let sum = be.residual_add_forward(&acc, &grad).unwrap();
    assert_eq!(be.budget().live_bytes().unwrap(), room);
    let (offset, storage) = (acc.byte_offset(), acc.storage_len());
    // The default's result holds the whole budget; adding in place needs
    // none of it.
    be.accumulate_grad(&mut acc, &grad).unwrap();
    assert_eq!(bits(&f(&acc)), bits(&f(&sum)));
    assert_eq!((acc.byte_offset(), acc.storage_len()), (offset, storage));
    assert_eq!(be.budget().live_bytes().unwrap(), room);
    drop(sum);
    assert_eq!(be.budget().live_bytes().unwrap(), 0);

    let tight = CpuBackend::new(Budget::new(room - 4));
    assert_capacity(tight.residual_add_forward(&acc, &grad));
    let before = bits(&f(&acc));
    let mut shared = acc.clone();
    assert_capacity(tight.accumulate_grad(&mut shared, &grad));
    assert_eq!(bits(&f(&shared)), before);
    assert_eq!(bits(&f(&acc)), before);
    assert_eq!(tight.budget().live_bytes().unwrap(), 0);
    let exact = CpuBackend::new(Budget::new(room));
    exact.accumulate_grad(&mut shared, &grad).unwrap();
    assert!(f(&shared).iter().all(|&x| x == 2.0));
    assert_eq!(bits(&f(&acc)), before);
    assert_eq!(exact.budget().live_bytes().unwrap(), room);
}

/// A shared `acc` cannot be written; it is replaced by a new tensor holding
/// the sum, as the default does, and the other handle keeps the old values.
#[test]
fn a_shared_acc_is_replaced_and_the_other_handle_is_unchanged() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let mut acc = t(&[1.0, 2.0, 3.0], &[3]);
    let other = acc.clone();
    be.accumulate_grad(&mut acc, &t(&[1.0, 1.0, 1.0], &[3]))
        .unwrap();
    assert_eq!(f(&acc), vec![2.0, 3.0, 4.0]);
    assert_eq!(f(&other), vec![1.0, 2.0, 3.0]);
    // A view of a live tensor shares too.
    let big = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let mut view = big.view(&[2], &[1], 8).unwrap();
    be.accumulate_grad(&mut view, &t(&[10.0, 10.0], &[2]))
        .unwrap();
    assert_eq!(f(&view), vec![13.0, 14.0]);
    assert_eq!(f(&big), vec![1.0, 2.0, 3.0, 4.0]);
}

/// NaN or infinity in `grad` or `acc`, and a finite sum that overflows, are
/// `NonFinite` with `acc`'s bits unchanged. So are every other refusal.
#[test]
fn every_refusal_leaves_acc_unchanged() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let init = [1.0f32, 3.0e38, -2.0, 0.5];
    let check = |acc: &Tensor, result: Result<(), OjasError>, what: &str| {
        assert_eq!(bits(&f(acc)), bits(&init), "{what}: acc changed");
        result
    };
    let mut acc = t(&init, &[4]);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let r = be.accumulate_grad(&mut acc, &t(&[0.0, 0.0, bad, 0.0], &[4]));
        assert_nonfinite(check(&acc, r, "non-finite grad"));
    }
    let r = be.accumulate_grad(&mut acc, &t(&[0.0, 3.0e38, 0.0, 0.0], &[4]));
    assert_nonfinite(check(&acc, r, "overflow"));
    let r = be.accumulate_grad(&mut acc, &t(&[0.0; 3], &[3]));
    assert_shape(check(&acc, r, "length"));
    let r = be.accumulate_grad(&mut acc, &t(&[0.0; 4], &[2, 2]));
    assert_shape(check(&acc, r, "shape"));
    let gu = Tensor::from_u32(&[0; 4], &[4], &Budget::new(u64::MAX)).unwrap();
    let r = be.accumulate_grad(&mut acc, &gu);
    assert!(matches!(
        check(&acc, r, "dtype"),
        Err(OjasError::Dtype {
            expected: DType::F32,
            ..
        })
    ));
    // Only a shared `acc` is charged (its replacement); short of room it
    // is refused with both handles unchanged.
    let tight = CpuBackend::new(Budget::new(8));
    let mut shared = acc.clone();
    let r = tight.accumulate_grad(&mut shared, &t(&[0.0; 4], &[4]));
    assert_capacity(check(&shared, r, "budget"));
    assert_eq!(bits(&f(&acc)), bits(&init));
    assert_eq!(tight.budget().live_bytes().unwrap(), 0);
    // NaN already in `acc` is refused before any charge.
    let mut nan_acc = t(&[f32::NAN, 0.0], &[2]);
    let none = CpuBackend::new(Budget::new(0));
    assert_nonfinite(none.accumulate_grad(&mut nan_acc, &t(&[0.0, 0.0], &[2])));
    assert!(f(&nan_acc)[0].is_nan());
}

/// Long enough that the check and the add are each cut into several pieces
/// on 6 threads (and one on 1): unique and shared `acc` both equal
/// `residual_add_forward` bit for bit; a NaN in the last piece, or an
/// overflow in the first, is refused with `acc` unchanged, because the
/// whole check finishes before the first write.
#[test]
fn several_pieces_sum_like_residual_add_and_refuse_before_any_write() {
    let n = 300_001usize;
    let mut rng = SplitMix64(0xacc3);
    let a = rng.vec(n, 2.0);
    let g = rng.vec(n, 2.0);
    for threads in [1, 6] {
        let be = CpuBackend::with_threads(Budget::new(u64::MAX), threads).unwrap();
        let want = bits(&f(&be.residual_add_forward(&t(&a, &[n]), &t(&g, &[n])).unwrap()));
        let mut acc = t(&a, &[n]);
        be.accumulate_grad(&mut acc, &t(&g, &[n])).unwrap();
        assert_eq!(bits(&f(&acc)), want, "unique, {threads} threads");
        let mut shared = t(&a, &[n]);
        let keep = shared.clone();
        be.accumulate_grad(&mut shared, &t(&g, &[n])).unwrap();
        assert_eq!(bits(&f(&shared)), want, "shared, {threads} threads");
        assert_eq!(bits(&f(&keep)), bits(&a));

        let mut nan_last = g.clone();
        nan_last[n - 1] = f32::NAN;
        let mut acc = t(&a, &[n]);
        assert_nonfinite(be.accumulate_grad(&mut acc, &t(&nan_last, &[n])));
        assert_eq!(bits(&f(&acc)), bits(&a), "NaN last, {threads} threads");
        let mut big = a.clone();
        big[0] = f32::MAX;
        let mut over = g.clone();
        over[0] = f32::MAX;
        let mut acc = t(&big, &[n]);
        assert_nonfinite(be.accumulate_grad(&mut acc, &t(&over, &[n])));
        assert_eq!(bits(&f(&acc)), bits(&big), "overflow, {threads} threads");
    }
}

/// `scale_grad` (the tape's loss seed): each value is one rounding of
/// `value * scale`, the host loop's bits, at one and several pieces. A
/// unique tensor is scaled where it is with nothing charged; a shared one is
/// replaced and the other handle keeps the old values; a NaN, an overflow
/// or a non-finite scale is `NonFinite`.
#[test]
fn scale_grad_is_the_host_product_in_place_or_replaced() {
    let mut rng = SplitMix64(0xacc4);
    for n in [7usize, 300_001] {
        let v = rng.vec(n, 3.0);
        let want: Vec<f32> = v.iter().map(|x| x * 0.25).collect();
        for threads in [1, 6] {
            let be = CpuBackend::with_threads(Budget::new(4 * n as u64), threads).unwrap();
            let mut g = t(&v, &[n]);
            let (offset, storage) = (g.byte_offset(), g.storage_len());
            be.scale_grad(&mut g, 0.25).unwrap();
            assert_eq!(bits(&f(&g)), bits(&want), "{n} in place, {threads} threads");
            assert_eq!((g.byte_offset(), g.storage_len()), (offset, storage));
            assert_eq!(be.budget().live_bytes().unwrap(), 0, "in place charged");

            let mut g = t(&v, &[n]);
            let keep = g.clone();
            be.scale_grad(&mut g, 0.25).unwrap();
            assert_eq!(bits(&f(&g)), bits(&want), "{n} shared, {threads} threads");
            assert_eq!(bits(&f(&keep)), bits(&v));
            assert_eq!(be.budget().live_bytes().unwrap(), 4 * n as u64);
        }
    }
    let be = CpuBackend::new(Budget::new(u64::MAX));
    assert_nonfinite(be.scale_grad(&mut t(&[1.0, f32::NAN], &[2]), 0.5));
    assert_nonfinite(be.scale_grad(&mut t(&[1.0, 3.0e38], &[2]), 4.0));
    assert_nonfinite(be.scale_grad(&mut t(&[1.0, 2.0], &[2]), f32::INFINITY));
    assert_nonfinite(be.scale_grad(&mut t(&[1.0, 2.0], &[2]), f32::NAN));
}

/// Interleaved A/B, min of N: the trait default's body
/// (`*acc = residual_add_forward(acc, grad)`) against the override.
/// `cargo test --release -p ojas-cpu --test framework_accum -- --ignored --nocapture`
#[test]
#[ignore]
fn accumulate_grad_ab() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let mut rng = SplitMix64(0xacc2);
    for (name, shape) in [("768x768", [768usize, 768]), ("50304x768", [50304, 768])] {
        let n = shape[0] * shape[1];
        let grad = t(&rng.vec(n, 1e-3), &shape);
        let mut a = t(&rng.vec(n, 1.0), &shape);
        let mut b = t(&f(&a), &shape);
        let reps = if n > 1 << 22 { 9 } else { 41 };
        let (mut best_default, mut best_override) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..reps {
            let start = Instant::now();
            a = be.residual_add_forward(&a, &grad).unwrap();
            best_default = best_default.min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            be.accumulate_grad(&mut b, &grad).unwrap();
            best_override = best_override.min(start.elapsed().as_secs_f64());
        }
        assert_eq!(bits(&f(&a)), bits(&f(&b)));
        println!(
            "accumulate_grad {name}: default (residual_add_forward) {:.3} ms, in place {:.3} ms, {:.2}x",
            best_default * 1e3,
            best_override * 1e3,
            best_default / best_override
        );
    }
}
