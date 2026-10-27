//! The fixed host cost of a wgpu dispatch, counted (`CacheStats`).
//!
//! Before: every dispatch created a 64-byte mapped uniform buffer and a bind
//! group, each under its own blocking error-scope pop, and every read made a
//! fresh `MAP_READ` staging buffer under another; a view at a byte offset
//! was copied into scratch on every op. Now parameters go to a ring bound at
//! a dynamic offset, bind groups and staging buffers are reused, and an
//! aligned offset view is bound in place.

mod common;

use common::*;
use ojas_core::{Backend, Tensor};

#[test]
fn a_steady_dispatch_creates_nothing_and_pops_no_error_scope() {
    let g = fresh();
    let x = g.upload(&host(1, &[4096])).unwrap();
    // Warm: the pipeline, the pool, the bind group and the staging buffer.
    for _ in 0..3 {
        drop(g.silu_forward(&x).unwrap());
        g.sync().unwrap();
    }
    let before = g.context().stats();
    const N: u64 = 50;
    for _ in 0..N {
        drop(g.silu_forward(&x).unwrap());
        g.sync().unwrap();
    }
    let after = g.context().stats();
    let d = |f: fn(&ojas_wgpu::CacheStats) -> u64| f(&after) - f(&before);
    assert_eq!(d(|s| s.dispatches), N);
    assert_eq!(d(|s| s.reads), N);
    assert_eq!(d(|s| s.scope_pops), 0, "{before:?} -> {after:?}");
    assert_eq!(d(|s| s.bind_groups), 0, "{before:?} -> {after:?}");
    assert_eq!(d(|s| s.staging_creates), 0, "{before:?} -> {after:?}");
    // Every op dispatch, and every read's fault hold, found its group.
    assert_eq!(d(|s| s.bind_hits), 2 * N, "{before:?} -> {after:?}");
    // The output came from the pool each time.
    assert_eq!(d(|s| s.pool_hits), N);
}

/// An upload is one fresh mapped-at-creation buffer under one blocking
/// out-of-memory scope pop, with or without work recorded, and no dispatch.
/// It does not take a pooled buffer: a queue write lands before recorded
/// work that may still read that buffer's old contents. The cost is
/// accepted (`docs/reports/BUILD_REPORT-4b.md`): a trainer step uploads its
/// ids and targets, two per micro-batch.
#[test]
fn an_upload_is_one_fresh_buffer_and_one_scope_pop() {
    let g = fresh();
    let x = g.upload(&host(1, &[4096])).unwrap();
    for _ in 0..3 {
        drop(g.silu_forward(&x).unwrap());
        g.sync().unwrap();
    }
    let ids: Vec<u32> = (0..4096u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let t = Tensor::from_u32(&ids, &[ids.len()], host_budget()).unwrap();
    for recorded in [false, true] {
        let pending = recorded.then(|| g.silu_forward(&x).unwrap());
        const N: u64 = 8;
        let before = g.context().stats();
        let ups: Vec<Tensor> = (0..N).map(|_| g.upload(&t).unwrap()).collect();
        let after = g.context().stats();
        let d = |f: fn(&ojas_wgpu::CacheStats) -> u64| f(&after) - f(&before);
        assert_eq!(d(|s| s.uploads), N, "recorded {recorded}");
        assert_eq!(
            d(|s| s.scope_pops),
            N,
            "recorded {recorded}: {before:?} -> {after:?}"
        );
        assert_eq!(d(|s| s.pool_hits), 0, "recorded {recorded}");
        assert_eq!(d(|s| s.dispatches), 0, "recorded {recorded}");
        drop(pending);
        for u in &ups {
            let back = g.download(u).unwrap();
            assert_eq!(
                back.u32_slice().unwrap(),
                ids.as_slice(),
                "recorded {recorded}"
            );
        }
    }
}

#[test]
fn offset_views_bind_in_place_when_the_device_allows_it() {
    // A fused projection's q, k and v as three row blocks of one buffer.
    let g = fresh();
    let c = cpu();
    let (rows, d) = (8usize, 64usize);
    let fused = host(7, &[3 * rows * d]);
    let dev = g.upload(&fused).unwrap();
    let align = u64::from(g.context().limits().min_storage_buffer_offset_alignment);
    // The view's result must be the same kernel's result on exactly the
    // slice's values: bit for bit against the slice uploaded on its own (a
    // binding at the wrong offset or length reads other values), and close
    // to the CPU like every parity test (the GPU's `exp` is not the CPU's,
    // so the two may differ in the last place).
    let check = |name: &str, view: &Tensor, slice: &Tensor| {
        let got = g.silu_forward(view).unwrap();
        let alone = g.silu_forward(&g.upload(slice).unwrap()).unwrap();
        assert_eq!(
            g.download(&got).unwrap().to_f32_vec().unwrap(),
            g.download(&alone).unwrap().to_f32_vec().unwrap(),
            "{name}"
        );
        close(name, &got, &c.silu_forward(slice).unwrap());
    };
    // The slice's values in a tensor of their own, at byte 0.
    let own = |off: usize, shape: &[usize], strides: &[usize]| {
        let v = fused
            .narrow(off, shape, strides)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        Tensor::from_f32(&v, shape, host_budget()).unwrap()
    };
    let before = g.context().stats().offset_copies;
    for i in 0..3 {
        let off = i * rows * d * 4;
        assert_eq!(
            off as u64 % align,
            0,
            "the slices are aligned on this device"
        );
        let view = dev.narrow(off, &[rows, d], &[d, 1]).unwrap();
        check(&format!("slice {i}"), &view, &own(off, &[rows, d], &[d, 1]));
    }
    assert_eq!(
        g.context().stats().offset_copies,
        before,
        "aligned slices were copied"
    );

    // One word in: not a storage-offset multiple, so compacted, and still right.
    let view = dev.narrow(4, &[rows * d], &[1]).unwrap();
    check("one word in", &view, &own(4, &[rows * d], &[1]));
    assert_eq!(g.context().stats().offset_copies, before + 1);
}

#[test]
fn a_gradient_view_at_an_offset_is_normed_in_place() {
    // The second half of a buffer, bound at its offset, is what the norm
    // reads: the first half (all 1) must not leak in.
    let g = fresh();
    let mut v = vec![1.0f32; 64];
    v[32..].fill(3.0);
    let dev = g
        .upload(&Tensor::from_f32(&v, &[64], host_budget()).unwrap())
        .unwrap();
    let before = g.context().stats().offset_copies;
    let mut grads = vec![dev.narrow(128, &[32], &[1]).unwrap()];
    let norm = g.clip_grad_norm(&mut grads, 1.0e9).unwrap();
    let want = (32.0f32 * 9.0).sqrt();
    assert!((norm - want).abs() <= 1e-5 * want, "{norm} vs {want}");
    assert_eq!(g.context().stats().offset_copies, before);
}

/// The global norm reads every gradient once, `k` gradients per dispatch
/// (`k` from the device's storage-binding limit), plus one finish; the
/// scale is `ceil(len / k)` more. Before: an abs-max and a sum-of-squares
/// dispatch per gradient plus two finishes, and one scale per gradient.
#[test]
fn clip_dispatches_scale_with_slot_groups_not_tensors() {
    let g = fresh();
    let c = cpu();
    let limit = g.context().limits().max_storage_buffers_per_shader_stage;
    let k = ojas_kernels::clip_slots(limit).unwrap() as u64;
    // 170 tensors, as nanolab's parameter list, at sizes that straddle the
    // 4096-value chunk.
    let sizes: Vec<usize> = (0..170).map(|i| 1 + (i * 977) % 9000).collect();
    let mut hg: Vec<Tensor> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| host(300 + i as u64, &[n]))
        .collect();
    let mut dg: Vec<Tensor> = hg.iter().map(|t| g.upload(t).unwrap()).collect();
    g.sync().unwrap();
    let before = g.context().stats().dispatches;
    let gn = g.clip_grad_norm(&mut dg, 1.0e9).unwrap();
    let norm_only = g.context().stats().dispatches - before;
    assert_eq!(norm_only, 170u64.div_ceil(k) + 1, "k {k}");
    let before = g.context().stats().dispatches;
    let gn2 = g.clip_grad_norm(&mut dg, gn * 0.5).unwrap();
    g.sync().unwrap();
    let with_scale = g.context().stats().dispatches - before;
    assert_eq!(with_scale, 2 * 170u64.div_ceil(k) + 1, "k {k}");
    let cn = c.clip_grad_norm(&mut hg, 1.0e9).unwrap();
    assert!((gn - cn).abs() <= 1e-5 * cn, "{gn} vs {cn}");
    assert!((gn2 - cn).abs() <= 1e-5 * cn, "{gn2} vs {cn}");
    c.clip_grad_norm(&mut hg, cn * 0.5).unwrap();
    for (i, (d, h)) in dg.iter().zip(&hg).enumerate() {
        close(&format!("scaled {i}"), d, h);
    }
}

/// `residual_add_backward` is one kernel, as Metal's `ojas_add_bwd`: it
/// writes the gradient to both outputs and checks `x`, `y` and the gradient
/// on the way. Before: three `check_finite` dispatches and two buffer copies.
#[test]
fn residual_add_backward_is_one_dispatch_that_checks_every_input() {
    const OP: &str = "residual_add_backward";
    let g = fresh();
    let c = cpu();
    let shape = [3usize, 4099];
    let n = 3 * 4099;
    let ins = [host(11, &shape), host(12, &shape), host(13, &shape)];
    let dev: Vec<Tensor> = ins.iter().map(|t| g.upload(t).unwrap()).collect();
    g.sync().unwrap();
    let before = g.context().stats().dispatches;
    let (gx, gy) = g.residual_add_backward(&dev[0], &dev[1], &dev[2]).unwrap();
    assert_eq!(g.context().stats().dispatches - before, 1);
    let want = ins[2].to_f32_vec().unwrap();
    let (cx, cy) = c.residual_add_backward(&ins[0], &ins[1], &ins[2]).unwrap();
    for (name, got, cpu) in [("grad_x", &gx, &cx), ("grad_y", &gy, &cy)] {
        assert_eq!(got.shape(), &shape[..], "{name}");
        let got = g.download(got).unwrap().to_f32_vec().unwrap();
        assert_eq!(got, want, "{name}");
        assert_eq!(got, cpu.to_f32_vec().unwrap(), "{name} vs cpu");
    }
    // A non-finite value in any one input, at the last element, is the op's
    // own fault by the next sync.
    for slot in 0..3 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let dev: Vec<Tensor> = ins
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let mut v = t.to_f32_vec().unwrap();
                    if i == slot {
                        v[n - 1] = bad;
                    }
                    g.upload(&Tensor::from_f32(&v, &shape, host_budget()).unwrap())
                        .unwrap()
                })
                .collect();
            let r = g.residual_add_backward(&dev[0], &dev[1], &dev[2]);
            assert!(r.is_ok(), "the fault is deferred: {r:?}");
            match g.sync() {
                Err(ojas_core::OjasError::NonFinite { op }) => {
                    assert_eq!(op, OP, "input {slot}, {bad}")
                }
                other => panic!("input {slot}, {bad}: {other:?}"),
            }
        }
    }
    g.residual_add_backward(&dev[0], &dev[1], &dev[2]).unwrap();
    g.sync().unwrap();
}

/// `scale_grad` on a gradient that solely owns its buffer scales it in
/// place: one dispatch and no transient the size of the gradient. Before:
/// the trait default uploaded the scalar and two ones vectors, built the
/// full broadcast with two k = 1 GEMMs and replaced the gradient with a
/// `mul_forward`, two gradients' bytes on top of the one held (about
/// 2 x 154 MB at a 50304 x 768 CE weight gradient).
#[test]
fn scale_grad_of_a_sole_owner_is_in_place_and_charges_no_transient() {
    const OP: &str = "scale_grad";
    let g = fresh();
    let shape = [64usize, 1000];
    let bytes = (64 * 1000 * 4) as u64;
    let x = host(21, &shape);
    let mut grad = g.upload(&x).unwrap();
    g.sync().unwrap();
    let live = g.budget().live_bytes().unwrap();
    g.budget().reset_peak();
    let before = g.context().stats();
    g.scale_grad(&mut grad, 0.25).unwrap();
    g.sync().unwrap();
    let after = g.context().stats();
    let took = g.budget().peak_bytes() - live;
    // The slot table alone: two words for the one tensor. No copy of the
    // gradient, and nothing for unused slots (the dispatch binds exactly one).
    assert_eq!(took, 8, "peak rose {took} B over a {bytes} B gradient");
    assert_eq!(after.dispatches - before.dispatches, 1);
    let want: Vec<f32> = x.to_f32_vec().unwrap().iter().map(|v| v * 0.25).collect();
    assert_eq!(g.download(&grad).unwrap().to_f32_vec().unwrap(), want);
    assert!(
        grad.device_buffer_mut().is_ok(),
        "grad must stay the sole owner"
    );

    // A non-finite scale is refused at the call and writes nothing.
    for bad in [f32::NAN, f32::INFINITY] {
        match g.scale_grad(&mut grad, bad) {
            Err(ojas_core::OjasError::NonFinite { op }) => assert_eq!(op, OP),
            other => panic!("{bad}: {other:?}"),
        }
    }
    assert_eq!(g.download(&grad).unwrap().to_f32_vec().unwrap(), want);

    // A product that overflows is the op's own fault by the next sync, on
    // both paths.
    for shared in [false, true] {
        let mut big = g
            .upload(&Tensor::from_f32(&[1.0, f32::MAX], &[2], host_budget()).unwrap())
            .unwrap();
        let keep = shared.then(|| big.clone());
        assert!(g.scale_grad(&mut big, 4.0).is_ok(), "the fault is deferred");
        match g.sync() {
            Err(ojas_core::OjasError::NonFinite { op }) => assert_eq!(op, OP, "shared {shared}"),
            other => panic!("shared {shared}: {other:?}"),
        }
        drop(keep);
    }
    g.sync().unwrap();
}

/// A shared gradient is replaced by the product; the other handle keeps
/// the old values and its buffer is not written.
#[test]
fn scale_grad_of_a_shared_gradient_replaces_it_and_leaves_the_other_handle() {
    let g = fresh();
    let x = host(22, &[3, 4099]);
    let mut grad = g.upload(&x).unwrap();
    let keep = grad.clone();
    g.scale_grad(&mut grad, -0.5).unwrap();
    let want: Vec<f32> = x.to_f32_vec().unwrap().iter().map(|v| v * -0.5).collect();
    assert_eq!(g.download(&grad).unwrap().to_f32_vec().unwrap(), want);
    assert_eq!(
        g.download(&keep).unwrap().to_f32_vec().unwrap(),
        x.to_f32_vec().unwrap()
    );
}

/// A gradient that solely owns its buffer but starts past byte 0 cannot be
/// scaled in place (the multi-tensor kernel binds whole buffers), so it is
/// replaced by the product at byte 0; the rows before it are not written.
#[test]
fn scale_grad_of_a_sole_owner_at_an_offset_replaces_it() {
    let g = fresh();
    let (rows, cols) = (4usize, 1000usize);
    let x = host(23, &[rows, cols]);
    let tall = g.upload(&x).unwrap();
    let mut grad = tall
        .narrow(cols * 4, &[rows - 1, cols], &[cols, 1])
        .unwrap();
    drop(tall);
    assert!(
        grad.device_buffer_mut().is_ok(),
        "the view is the sole owner"
    );
    g.scale_grad(&mut grad, 3.0).unwrap();
    assert_eq!(grad.byte_offset(), 0);
    let all = x.to_f32_vec().unwrap();
    let want: Vec<f32> = all[cols..].iter().map(|v| v * 3.0).collect();
    assert_eq!(g.download(&grad).unwrap().to_f32_vec().unwrap(), want);
}

/// Values near f32's top: the squares overflow, the norm does not. Every
/// chunk keeps its own max, and the finish rescales to the global one.
#[test]
fn clip_norm_of_huge_values_is_finite_and_right() {
    let g = fresh();
    let big = 1.0e30f32;
    let mut a = vec![big; 5000];
    a[4999] = -2.0 * big;
    let b = vec![0.5f32 * big; 3];
    let mut dg = vec![
        g.upload(&Tensor::from_f32(&a, &[5000], host_budget()).unwrap())
            .unwrap(),
        g.upload(&Tensor::from_f32(&b, &[3], host_budget()).unwrap())
            .unwrap(),
    ];
    let norm = g.clip_grad_norm(&mut dg, f32::MAX).unwrap();
    let want = (f64::from(big) * (4999.0f64 + 4.0 + 3.0 * 0.25).sqrt()) as f32;
    assert!(norm.is_finite(), "{norm}");
    assert!((norm - want).abs() <= 1e-5 * want, "{norm} vs {want}");
}

/// A non-finite value in any slot of any dispatch group is the clip's own
/// `NonFinite`, and nothing is scaled.
#[test]
fn a_nan_in_a_late_slot_group_is_found_and_scales_nothing() {
    let g = fresh();
    let mut hg: Vec<Tensor> = (0..40).map(|i| host(900 + i, &[100])).collect();
    let mut v = hg[37].to_f32_vec().unwrap();
    v[99] = f32::NAN;
    hg[37] = Tensor::from_f32(&v, &[100], host_budget()).unwrap();
    let mut dg: Vec<Tensor> = hg.iter().map(|t| g.upload(t).unwrap()).collect();
    match g.clip_grad_norm(&mut dg, 1e-3) {
        Err(ojas_core::OjasError::NonFinite { op }) => assert_eq!(op, "clip_grad_norm"),
        other => panic!("{other:?}"),
    }
    let first = g.download(&dg[0]).unwrap().to_f32_vec().unwrap();
    assert_eq!(first, hg[0].to_f32_vec().unwrap(), "a gradient was scaled");
}

/// Scratch is held until its commands are submitted, so a run of ops whose
/// widened bf16 operands sum past the budget needs a submit before the
/// budget is exhausted, not only after [`ojas_wgpu::FLUSH_AT`] dispatches.
/// A call the budget cannot carry beside the pending scratch submits it and
/// goes on; the submit count rises with the bytes. Before: the third call
/// returned `CapacityExceeded`.
///
/// The output is charged in `out` before either widen, and every output is
/// kept. The cap is the operands, all eight outputs, and exactly one call's
/// widened scratch. A second call's scratch does not fit beside that, so
/// the widen refusal submits and retries. An extra `widen / 4` used to let
/// two scratches stay resident, and the next `out` then failed with the
/// recorder still full (`call 4 of 8`) because `out` is not scratch and
/// does not submit.
#[test]
fn pending_scratch_that_outgrows_the_budget_triggers_a_submit() {
    use ojas_core::{Budget, DType};
    let (rows, k, n) = (256usize, 1024usize, 1024usize);
    const CALLS: usize = 8;
    let widen = ((rows * k + n * k) * 4) as u64;
    let out = (rows * n * 4) as u64;
    let operands = ((rows * k + n * k) * 2) as u64;
    let cap = operands + CALLS as u64 * out + widen;
    let g = ojas_wgpu::WgpuBackend::open(Budget::new(cap)).expect("wgpu adapter");
    let bits = |seed: u64, shape: &[usize]| {
        let v = data(seed, shape.iter().product());
        let b: Vec<u16> = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
        let t = Tensor::from_bf16_bits(&b, shape, host_budget()).unwrap();
        assert_eq!(t.dtype(), DType::Bf16);
        g.upload(&t).unwrap()
    };
    let x = bits(1, &[rows, k]);
    let w = bits(2, &[n, k]);
    let before = g.context().stats();
    let mut ys = Vec::new();
    for i in 0..CALLS {
        ys.push(
            g.linear_forward(&x, &w)
                .unwrap_or_else(|e| panic!("call {i} of {CALLS}: {e}")),
        );
    }
    g.sync().unwrap();
    let after = g.context().stats();
    let submits = after.submits - before.submits;
    eprintln!(
        "{CALLS} bf16 linear_forward: {submits} submits, {} dispatches",
        after.dispatches - before.dispatches
    );
    assert!(
        submits >= 2,
        "{CALLS} calls of {widen} scratch bytes under {cap}: {submits} submits"
    );
    assert!(
        after.dispatches - before.dispatches < ojas_wgpu::FLUSH_AT as u64,
        "the dispatch count alone would have flushed"
    );
    assert_eq!(ys.len(), CALLS);
}
