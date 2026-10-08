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

#[test]
fn offset_views_bind_in_place_when_the_device_allows_it() {
    // A fused projection's q, k and v as three row blocks of one buffer.
    let g = fresh();
    let c = cpu();
    let (rows, d) = (8usize, 64usize);
    let fused = host(7, &[3 * rows * d]);
    let dev = g.upload(&fused).unwrap();
    let align = u64::from(g.context().limits().min_storage_buffer_offset_alignment);
    let before = g.context().stats().offset_copies;
    for i in 0..3 {
        let off = i * rows * d * 4;
        assert_eq!(off as u64 % align, 0, "the slices are aligned on this device");
        let view = dev.narrow(off, &[rows, d], &[d, 1]).unwrap();
        let got = g.silu_forward(&view).unwrap();
        let want = c
            .silu_forward(&fused.narrow(off, &[rows, d], &[d, 1]).unwrap())
            .unwrap();
        assert_eq!(
            g.download(&got).unwrap().to_f32_vec().unwrap(),
            want.to_f32_vec().unwrap(),
            "slice {i}"
        );
    }
    assert_eq!(g.context().stats().offset_copies, before, "aligned slices were copied");

    // One word in: not a storage-offset multiple, so compacted, and still right.
    let view = dev.narrow(4, &[rows * d], &[1]).unwrap();
    let got = g.silu_forward(&view).unwrap();
    let want = c
        .silu_forward(&fused.narrow(4, &[rows * d], &[1]).unwrap())
        .unwrap();
    assert_eq!(
        g.download(&got).unwrap().to_f32_vec().unwrap(),
        want.to_f32_vec().unwrap()
    );
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
        g.upload(&Tensor::from_f32(&a, &[5000], host_budget()).unwrap()).unwrap(),
        g.upload(&Tensor::from_f32(&b, &[3], host_budget()).unwrap()).unwrap(),
    ];
    let norm = g.clip_grad_norm(&mut dg, f32::MAX).unwrap();
    let want = (f64::from(big)
        * (4999.0f64 + 4.0 + 3.0 * 0.25).sqrt()) as f32;
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
