//! Folded (2-D) dispatch grids: `fold_grid` rounds a workgroup count up to
//! `max x rows`, so a kernel launched on one has spare workgroups past the
//! real count. Capping `max_compute_workgroups_per_dimension` at a few
//! groups folds the grid at sizes a test can afford, where the real trigger
//! is a gradient of more than 65,535 x 4,096 elements (the Qwen3.5-2B
//! embedding).
//!
//! `clip_grad_norm` is held to an f64 reference and, bit for bit, to the
//! uncapped device (the chunk association does not depend on the grid), and
//! must repeat bit for bit. The sum stage and Muon's norm are covered by the
//! unit tests in `ojas-wgpu/src/backend.rs`, which can reach them without
//! the pointwise kernels that a cap this small also refuses.

mod common;

use common::*;
use ojas_core::{Backend, Tensor};
use ojas_wgpu::WgpuBackend;

const CHUNK: usize = 4096;

/// `sizes.len()` gradients; the last chunk of the last one is a thousand times
/// larger than the rest, so a dropped last chunk moves the norm.
fn grads(seed: u64, sizes: &[usize]) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| data(seed + i as u64, n))
        .collect();
    if let Some(last) = out.last_mut() {
        let from = (last.len() - 1) / CHUNK * CHUNK;
        for v in &mut last[from..] {
            *v *= 1000.0;
        }
    }
    out
}

fn reference(vs: &[Vec<f32>]) -> f64 {
    vs.iter()
        .flatten()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt()
}

fn norm_on(g: &WgpuBackend, vs: &[Vec<f32>]) -> f32 {
    let mut ts: Vec<Tensor> = vs
        .iter()
        .map(|v| {
            g.upload(&Tensor::from_f32(v, &[v.len()], host_budget()).unwrap())
                .unwrap()
        })
        .collect();
    // No clipping: the scale kernel is covered by its own test.
    g.clip_grad_norm(&mut ts, f32::MAX).unwrap()
}

fn check(g: &WgpuBackend, name: &str, vs: &[Vec<f32>], runs: usize) {
    let want = reference(vs);
    let first = norm_on(g, vs);
    let rel = (f64::from(first) - want).abs() / want;
    assert!(
        rel < 1e-5,
        "{name}: norm {first} vs f64 {want} (rel {rel:.2e})"
    );
    // The uncapped device associates the same chunks, so the bits agree.
    let open = norm_on(gpu(), vs);
    assert_eq!(
        first.to_bits(),
        open.to_bits(),
        "{name}: capped vs uncapped"
    );
    for run in 1..runs {
        let again = norm_on(g, vs);
        assert_eq!(
            first.to_bits(),
            again.to_bits(),
            "{name}: run {run} gave {again}, run 0 gave {first}"
        );
    }
}

/// The brief's repro: a cap of 2 and 3 chunks, a (2, 2) grid with one spare.
#[test]
fn three_chunks_under_a_cap_of_two_repeat_bit_for_bit() {
    let g = capped(2);
    check(&g, "12288", &grads(1, &[3 * CHUNK]), 20);
    check(
        &g,
        "12289 (4 chunks, no spare)",
        &grads(2, &[3 * CHUNK + 1]),
        20,
    );
    check(&g, "8193 (3 chunks)", &grads(3, &[2 * CHUNK + 1]), 20);
}

/// Every chunk count a cap folds, at exact multiples, one past and one short,
/// over a single tensor and several sharing a dispatch.
#[test]
fn every_chunk_count_under_small_caps_matches_the_reference() {
    for cap in [2u32, 3, 4] {
        let g = capped(cap);
        let most = (cap * cap) as usize;
        for chunks in 1..=most {
            for n in [chunks * CHUNK, chunks * CHUNK - 1, (chunks - 1) * CHUNK + 1] {
                if n == 0 {
                    continue;
                }
                check(
                    &g,
                    &format!("cap {cap} n {n}"),
                    &grads(10 + n as u64, &[n]),
                    3,
                );
            }
        }
        // Tensors that share one dispatch, a tail inside the last.
        if cap >= 3 {
            let sizes = [CHUNK + 7, 1, 2 * CHUNK, CHUNK + 1];
            check(&g, &format!("cap {cap} mixed"), &grads(99, &sizes), 5);
        }
        // One element.
        check(&g, "one element", &grads(5, &[1]), 3);
    }
}

/// Past the folded grid's reach the host refuses; nothing is clamped.
#[test]
fn a_gradient_past_the_folded_grid_is_refused() {
    let g = capped(2);
    let v = grads(7, &[4 * CHUNK + 1]);
    let mut ts = vec![g
        .upload(&Tensor::from_f32(&v[0], &[v[0].len()], host_budget()).unwrap())
        .unwrap()];
    assert!(g.clip_grad_norm(&mut ts, f32::MAX).is_err());
}

/// The pointwise and row kernels launch on `lanes()`/`groups()` too: under a
/// cap of 2 they fold at 513 lanes (three groups on a (2, 2) grid), and each
/// must leave its spare workgroup inert, at every boundary of the grid.
#[test]
fn pointwise_and_row_kernels_leave_spare_workgroups_inert_under_a_cap_of_two() {
    let g = capped(2);
    let c = cpu();
    let dev = |v: &[f32], shape: &[usize]| {
        g.upload(&Tensor::from_f32(v, shape, host_budget()).unwrap())
            .unwrap()
    };
    let back = |t: &Tensor| g.download(t).unwrap().to_f32_vec().unwrap();
    for n in [1usize, 255, 256, 257, 512, 513, 768, 769, 1023, 1024] {
        let (a, b, gy) = (data(1, n), data(2, n), data(3, n));
        let (ha, hb, hg) = (
            Tensor::from_f32(&a, &[n], host_budget()).unwrap(),
            Tensor::from_f32(&b, &[n], host_budget()).unwrap(),
            Tensor::from_f32(&gy, &[n], host_budget()).unwrap(),
        );
        let (da, db, dg) = (dev(&a, &[n]), dev(&b, &[n]), dev(&gy, &[n]));
        let want = |t: Tensor| t.to_f32_vec().unwrap();
        close_vec(
            &format!("silu {n}"),
            &back(&g.silu_forward(&da).unwrap()),
            &want(c.silu_forward(&ha).unwrap()),
        );
        close_vec(
            &format!("silu' {n}"),
            &back(&g.silu_backward(&da, &dg).unwrap()),
            &want(c.silu_backward(&ha, &hg).unwrap()),
        );
        close_vec(
            &format!("mul {n}"),
            &back(&g.mul_forward(&da, &db).unwrap()),
            &want(c.mul_forward(&ha, &hb).unwrap()),
        );
        close_vec(
            &format!("add {n}"),
            &back(&g.residual_add_forward(&da, &db).unwrap()),
            &want(c.residual_add_forward(&ha, &hb).unwrap()),
        );
        let (ga, gb) = g.mul_backward(&da, &db, &dg).unwrap();
        let (ca, cb) = c.mul_backward(&ha, &hb, &hg).unwrap();
        close_vec(&format!("mul' a {n}"), &back(&ga), &want(ca));
        close_vec(&format!("mul' b {n}"), &back(&gb), &want(cb));
    }
    // Rows: dim 4 packs 64 rows per workgroup, dim 256 one.
    for (rows, dim) in [(130usize, 4usize), (129, 4), (3, 256), (4, 256), (2, 256)] {
        let (x, w, gy) = (data(4, rows * dim), data(5, dim), data(6, rows * dim));
        let (hx, hw, hg) = (
            Tensor::from_f32(&x, &[rows, dim], host_budget()).unwrap(),
            Tensor::from_f32(&w, &[dim], host_budget()).unwrap(),
            Tensor::from_f32(&gy, &[rows, dim], host_budget()).unwrap(),
        );
        let (dx, dw, dg) = (
            dev(&x, &[rows, dim]),
            dev(&w, &[dim]),
            dev(&gy, &[rows, dim]),
        );
        let eps = 1e-6;
        close_vec(
            &format!("rms {rows}x{dim}"),
            &back(&g.rms_norm_forward(&dx, &dw, eps).unwrap()),
            &c.rms_norm_forward(&hx, &hw, eps)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
        let (gx, _) = g.rms_norm_backward(&dx, &dw, &dg, eps).unwrap();
        let (cx, _) = c.rms_norm_backward(&hx, &hw, &hg, eps).unwrap();
        close_vec(
            &format!("rms' {rows}x{dim}"),
            &back(&gx),
            &cx.to_f32_vec().unwrap(),
        );
    }
}

/// The Qwen3.5-2B embedding gradient, 248,320 x 2,048 (2 GiB): 124,160 chunks
/// fold to a (65,535, 2) grid with 6,910 spare workgroups on a real device.
#[test]
#[ignore = "2 GiB gradient; run with --ignored"]
fn the_qwen_2b_embedding_gradient_keeps_its_last_chunk() {
    let n = 248_320usize * 2_048;
    let mut v = data(2026, 1 << 20);
    v.resize(n, 0.0);
    for i in (1 << 20)..n {
        v[i] = v[i & ((1 << 20) - 1)];
    }
    for x in &mut v[n - CHUNK..] {
        *x *= 10.0;
    }
    let g = gpu();
    let want = reference(std::slice::from_ref(&v));
    let first = norm_on(g, std::slice::from_ref(&v));
    let rel = (f64::from(first) - want).abs() / want;
    assert!(rel < 1e-5, "norm {first} vs f64 {want} (rel {rel:.2e})");
    for run in 1..5 {
        let again = norm_on(g, std::slice::from_ref(&v));
        assert_eq!(first.to_bits(), again.to_bits(), "run {run}");
    }
}
