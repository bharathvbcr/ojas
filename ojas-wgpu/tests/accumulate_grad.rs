//! `Backend::accumulate_grad` on wgpu (T2, gate G2): `acc += grad` in the
//! accumulator's own buffer when it is the sole owner, bit-identical to the
//! CPU reference (`Numerics::Exact`); a shared accumulator gets a new buffer
//! and the other handle is untouched. Non-finite sums are deferred like every
//! wgpu fault: the op returns `Ok` and the next `sync` reports `NonFinite`
//! naming `accumulate_grad`, after which `acc` is invalid (the trait's rule
//! for a deferring backend). Shape, dtype and placement errors are
//! synchronous and leave `acc` unchanged.

mod common;

use std::sync::Arc;

use common::*;
use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

const OP: &str = "accumulate_grad";

fn bits(g: &WgpuBackend, t: &Tensor) -> Vec<u32> {
    g.download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn buffer_ptr(t: &Tensor) -> *const () {
    match t.device_buffer() {
        Some(b) => Arc::as_ptr(b) as *const (),
        None => panic!("not a device tensor"),
    }
}

fn scaled(seed: u64, shape: &[usize], s: f32) -> Tensor {
    let n = shape.iter().product();
    let v: Vec<f32> = data(seed, n).iter().map(|x| x * s).collect();
    Tensor::from_f32(&v, shape, host_budget()).unwrap()
}

#[test]
fn matches_cpu_bit_for_bit_in_the_same_buffer() {
    let g = fresh();
    let c = cpu();
    for (i, shape) in [vec![1], vec![3, 4099], vec![768, 768], vec![4, 12, 64]]
        .into_iter()
        .enumerate()
    {
        let seed = 10 * i as u64;
        let (a, gr) = (scaled(seed, &shape, 3.0), scaled(seed + 1, &shape, 2.0));
        let mut want = a.clone();
        c.accumulate_grad(&mut want, &gr).unwrap();
        let mut acc = g.upload(&a).unwrap();
        let before = buffer_ptr(&acc);
        g.accumulate_grad(&mut acc, &g.upload(&gr).unwrap())
            .unwrap();
        assert_eq!(buffer_ptr(&acc), before, "{shape:?}: not written in place");
        g.sync().unwrap();
        let want: Vec<u32> = want
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(bits(&g, &acc), want, "{shape:?}: differs from the CPU");
    }
}

#[test]
fn repeated_accumulation_matches_cpu() {
    let g = fresh();
    let c = cpu();
    let shape = [5, 77];
    let mut want = scaled(1, &shape, 1.0);
    let mut acc = g.upload(&want).unwrap();
    for step in 0..6u64 {
        let gr = scaled(100 + step, &shape, 0.5);
        c.accumulate_grad(&mut want, &gr).unwrap();
        g.accumulate_grad(&mut acc, &g.upload(&gr).unwrap())
            .unwrap();
    }
    g.sync().unwrap();
    let want: Vec<u32> = want
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert_eq!(bits(&g, &acc), want);
}

#[test]
fn runs_in_place_under_a_budget_with_no_room_for_a_new_sum() {
    let n = 3 * 4099;
    // Exactly acc and grad: a path that allocates the sum is refused.
    let g = WgpuBackend::with_context(gpu().context().clone(), Budget::new(2 * 4 * n as u64));
    let mut acc = g.upload(&scaled(1, &[3, 4099], 1.0)).unwrap();
    let gr = g.upload(&scaled(2, &[3, 4099], 1.0)).unwrap();
    g.accumulate_grad(&mut acc, &gr).unwrap();
    g.sync().unwrap();
}

#[test]
fn a_shared_accumulator_gets_a_new_buffer_and_the_other_handle_is_untouched() {
    let g = fresh();
    let shape = [5, 33];
    let (a, gr) = (scaled(3, &shape, 1.0), scaled(4, &shape, 1.0));
    let mut acc = g.upload(&a).unwrap();
    let other = acc.clone();
    let before = bits(&g, &other);
    g.accumulate_grad(&mut acc, &g.upload(&gr).unwrap())
        .unwrap();
    assert_ne!(buffer_ptr(&acc), buffer_ptr(&other));
    assert_eq!(bits(&g, &other), before, "the shared handle was written");
    let want: Vec<u32> = a
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip(gr.to_f32_vec().unwrap())
        .map(|(x, y)| (x + y).to_bits())
        .collect();
    assert_eq!(bits(&g, &acc), want);
    assert!(
        acc.device_buffer_mut().is_ok(),
        "acc must end uniquely owned"
    );
}

#[test]
fn a_non_finite_sum_surfaces_at_sync_naming_accumulate_grad() {
    let g = fresh();
    let shape = [3, 4099];
    let n = 3 * 4099;
    let mut cases: Vec<(f32, f32)> = Vec::new();
    for p in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        cases.push((p, 1.0));
        cases.push((1.0, p));
    }
    cases.push((f32::MAX, f32::MAX));
    for shared in [false, true] {
        for &(av, gv) in &cases {
            let mut a = data(5, n);
            let mut gr = data(6, n);
            a[n - 1] = av;
            gr[n - 1] = gv;
            let mut acc = g
                .upload(&Tensor::from_f32(&a, &shape, host_budget()).unwrap())
                .unwrap();
            let keep = shared.then(|| acc.clone());
            let r = g.accumulate_grad(
                &mut acc,
                &g.upload(&Tensor::from_f32(&gr, &shape, host_budget()).unwrap())
                    .unwrap(),
            );
            assert!(r.is_ok(), "the fault is deferred: {r:?}");
            match g.sync() {
                Err(OjasError::NonFinite { op }) => {
                    assert_eq!(op, OP, "shared {shared}, acc {av}, grad {gv}")
                }
                other => panic!("shared {shared}, acc {av}, grad {gv}: {other:?}"),
            }
            drop(keep);
        }
    }
    // Clean afterwards.
    let mut acc = g.upload(&scaled(7, &shape, 1.0)).unwrap();
    g.accumulate_grad(&mut acc, &g.upload(&scaled(8, &shape, 1.0)).unwrap())
        .unwrap();
    g.sync().unwrap();
}

#[test]
fn refusals_are_synchronous_and_leave_acc_unchanged() {
    let g = fresh();
    let mut acc = g.upload(&scaled(1, &[4, 6], 1.0)).unwrap();
    let before = bits(&g, &acc);
    let ptr = buffer_ptr(&acc);
    let wrong = g.upload(&scaled(2, &[6, 4], 1.0)).unwrap();
    assert!(matches!(
        g.accumulate_grad(&mut acc, &wrong),
        Err(OjasError::Shape { .. })
    ));
    let host_grad = scaled(3, &[4, 6], 1.0);
    assert!(matches!(
        g.accumulate_grad(&mut acc, &host_grad),
        Err(OjasError::Placement { .. })
    ));
    let ids = g
        .upload(&Tensor::from_u32(&[0; 24], &[4, 6], host_budget()).unwrap())
        .unwrap();
    assert!(matches!(
        g.accumulate_grad(&mut acc, &ids),
        Err(OjasError::Dtype { .. })
    ));
    assert_eq!(buffer_ptr(&acc), ptr);
    g.sync().unwrap();
    assert_eq!(bits(&g, &acc), before);
}

// The optimizer's reported scratch (`optimizer_scratch_bytes`) against the
// budget's measured peak; it lives here because it shares this file's
// own-backend setup and adds no test binary.

/// A gradient at byte offset 0, or at an offset the device can bind in
/// place, charges the reported figure less the one gradient copy `bind`
/// makes for a view it cannot bind; a gradient view at such an offset
/// charges the reported figure exactly. The figure is therefore never below
/// what a step takes.
#[test]
fn reported_optimizer_scratch_bounds_the_measured_peak() {
    use ojas_core::{AdamWConfig, MuonNs5Config, OptimizerKind};
    for (rows, cols) in [(1, 1), (3, 5), (64, 17), (17, 64), (96, 96)] {
        let g = fresh();
        let align = u64::from(g.context().limits().min_storage_buffer_offset_alignment);
        let copy = (rows * cols * 4) as u64;
        let what = format!("[{rows}, {cols}]");
        for (kind, offset) in [
            (OptimizerKind::MuonNs5, false),
            (OptimizerKind::MuonNs5, true),
            (OptimizerKind::AdamW, false),
            (OptimizerKind::AdamW, true),
        ] {
            let reported = g
                .optimizer_scratch_bytes(kind, rows, cols)
                .unwrap()
                .unwrap();
            let mut p = g.upload(&host(1, &[rows, cols])).unwrap();
            let mut m1 = g.upload(&host(3, &[rows, cols])).unwrap();
            // AdamW's second moment is a running mean of squares: >= 0.
            let zeros = Tensor::zeros(&[rows, cols], ojas_core::DType::F32, host_budget()).unwrap();
            let mut m2 = g.upload(&zeros).unwrap();
            // An offset gradient is rows 1.. of a taller tensor.
            let tall = g.upload(&host(2, &[rows + 1, cols])).unwrap();
            let grad = if offset {
                tall.narrow(cols * 4, &[rows, cols], &[cols, 1]).unwrap()
            } else {
                g.upload(&host(2, &[rows, cols])).unwrap()
            };
            let live = g.budget().live_bytes().unwrap();
            g.budget().reset_peak();
            match kind {
                OptimizerKind::MuonNs5 => g
                    .muon_ns5_step(&mut p, &grad, &mut m1, MuonNs5Config::nanolab_default())
                    .unwrap(),
                OptimizerKind::AdamW => g
                    .adamw_step(
                        &mut p,
                        &grad,
                        &mut m1,
                        &mut m2,
                        0,
                        AdamWConfig::nanolab(1e-3, 0.0),
                    )
                    .unwrap(),
            }
            g.sync().unwrap();
            let took = g.budget().peak_bytes() - live;
            let copied = offset && !(cols as u64 * 4).is_multiple_of(align);
            let expect = if copied { reported } else { reported - copy };
            assert_eq!(took, expect, "{what} {kind:?} offset {offset}");
        }
    }
}

/// AdamW decides on the device and writes in place: beyond its operands a
/// step charges only its 16-byte status word, at every size. Before, it
/// copied the parameter and both moments to scratch each step (`3 * 4n`
/// more bytes) so it could restore them on a fault.
#[test]
fn adamw_step_copies_no_state() {
    use ojas_core::AdamWConfig;
    for n in [1usize, 255, 4096, 70_001] {
        let g = fresh();
        let mut p = g.upload(&host(11, &[n])).unwrap();
        let grad = g.upload(&host(12, &[n])).unwrap();
        let mut m1 = g.upload(&host(13, &[n])).unwrap();
        let zeros = Tensor::zeros(&[n], ojas_core::DType::F32, host_budget()).unwrap();
        let mut m2 = g.upload(&zeros).unwrap();
        g.sync().unwrap();
        let before = g.context().stats();
        let live = g.budget().live_bytes().unwrap();
        g.budget().reset_peak();
        g.adamw_step(
            &mut p,
            &grad,
            &mut m1,
            &mut m2,
            1,
            AdamWConfig::nanolab(1e-3, 0.1),
        )
        .unwrap();
        g.sync().unwrap();
        assert_eq!(g.budget().peak_bytes() - live, 16, "n {n}");
        let after = g.context().stats();
        assert_eq!(after.dispatches - before.dispatches, 2, "check, then apply");
    }
}
