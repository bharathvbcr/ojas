//! `clip_grad_norm` as a multi-tensor pass: the norm is tessl's `sq_norm`,
//! two kernels in one binder scope for any number of gradients, then 24 gradients per
//! `ojas_scale_multi`. Before 0e3254f each gradient took a finite check and
//! two two-stage reductions (5 dispatches) and two more to scale; from it
//! until tessl's norm was wired in, 24 gradients per `ojas_norm_multi`
//! dispatch and one finish.
#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};

fn vec1(v: &[f32]) -> Tensor {
    host(v, &[v.len()])
}

#[test]
fn clip_norm_is_one_scope_and_scale_goes_by_slot_groups() {
    let (m, c) = (metal(), cpu());
    let sizes: Vec<usize> = (0..170).map(|i| 1 + (i * 977) % 9000).collect();
    let mut hg: Vec<Tensor> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| vec1(&values(n, 300 + i as u64, 1.0)))
        .collect();
    let mut dg: Vec<Tensor> = hg.iter().map(|t| m.upload(t).unwrap()).collect();
    m.sync().unwrap();
    let before = m.dispatches();
    let gn = m.clip_grad_norm(&mut dg, 1.0e9).unwrap();
    // `dispatches` counts tessl binder scopes: the norm's two kernels are
    // one, whatever the gradient count.
    assert_eq!(m.dispatches() - before, 1);
    let before = m.dispatches();
    let gn2 = m.clip_grad_norm(&mut dg, gn * 0.5).unwrap();
    m.sync().unwrap();
    assert_eq!(m.dispatches() - before, 1 + 170u64.div_ceil(24));
    let cn = c.clip_grad_norm(&mut hg, 1.0e9).unwrap();
    assert!((gn - cn).abs() <= 1e-5 * cn, "{gn} vs {cn}");
    assert!((gn2 - cn).abs() <= 1e-5 * cn, "{gn2} vs {cn}");
    c.clip_grad_norm(&mut hg, cn * 0.5).unwrap();
    for (i, (d, h)) in dg.iter().zip(&hg).enumerate() {
        let got = m.download(d).unwrap().to_f32_vec().unwrap();
        let want = h.to_f32_vec().unwrap();
        for (j, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 + 1e-5 * b.abs(),
                "{i}[{j}]: {a} vs {b}"
            );
        }
    }
}

/// Gradient counts at each edge of a 24-slot group: a last group of 1, 23,
/// a full one, and a full one plus 1. The slots a short group leaves unused
/// bind the worker's spare words, never a gradient again. Every scaled
/// value against the CPU, and the dispatch count.
#[test]
fn every_last_group_size_scales_like_the_cpu() {
    let (m, c) = (metal(), cpu());
    for count in [1usize, 23, 24, 25, 47] {
        let mut hg: Vec<Tensor> = (0..count)
            .map(|i| vec1(&values(1 + (i * 613) % 5000, 500 + i as u64, 1.0)))
            .collect();
        let mut dg: Vec<Tensor> = hg.iter().map(|t| m.upload(t).unwrap()).collect();
        let cn = c.clip_grad_norm(&mut hg.clone(), 1.0e9).unwrap();
        m.sync().unwrap();
        let before = m.dispatches();
        let gn = m.clip_grad_norm(&mut dg, cn * 0.5).unwrap();
        m.sync().unwrap();
        assert_eq!(
            m.dispatches() - before,
            1 + count.div_ceil(24) as u64,
            "count {count}"
        );
        assert!((gn - cn).abs() <= 1e-5 * cn, "count {count}: {gn} vs {cn}");
        c.clip_grad_norm(&mut hg, cn * 0.5).unwrap();
        for (i, (d, h)) in dg.iter().zip(&hg).enumerate() {
            let got = m.download(d).unwrap().to_f32_vec().unwrap();
            let want = h.to_f32_vec().unwrap();
            for (j, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-6 + 1e-5 * b.abs(),
                    "count {count}: {i}[{j}]: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn clip_norm_of_huge_values_is_finite_and_right() {
    let m = metal();
    let big = 1.0e30f32;
    let mut a = vec![big; 5000];
    a[4999] = -2.0 * big;
    let mut dg = vec![
        m.upload(&vec1(&a)).unwrap(),
        m.upload(&vec1(&[0.5 * big; 3])).unwrap(),
    ];
    let norm = m.clip_grad_norm(&mut dg, f32::MAX).unwrap();
    let want = (f64::from(big) * (4999.0f64 + 4.0 + 0.75).sqrt()) as f32;
    assert!(norm.is_finite(), "{norm}");
    assert!((norm - want).abs() <= 1e-5 * want, "{norm} vs {want}");
}

#[test]
fn a_nan_in_a_late_slot_group_is_found_and_scales_nothing() {
    let m = metal();
    let mut hv: Vec<Vec<f32>> = (0..60).map(|i| values(100, 900 + i, 1.0)).collect();
    hv[53][99] = f32::NAN;
    let mut dg: Vec<Tensor> = hv.iter().map(|v| m.upload(&vec1(v)).unwrap()).collect();
    match m.clip_grad_norm(&mut dg, 1e-3) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "clip_grad_norm"),
        other => panic!("{other:?}"),
    }
    let first = m.download(&dg[0]).unwrap().to_f32_vec().unwrap();
    assert_eq!(first, hv[0], "a gradient was scaled");
}

#[test]
fn views_at_offsets_are_normed_where_they_sit() {
    let m = metal();
    let mut v = vec![1.0f32; 64];
    v[32..].fill(3.0);
    let dev = m.upload(&vec1(&v)).unwrap();
    let mut dg = vec![dev.narrow(128, &[32], &[1]).unwrap()];
    let norm = m.clip_grad_norm(&mut dg, 1.0e9).unwrap();
    let want = (32.0f32 * 9.0).sqrt();
    assert!((norm - want).abs() <= 1e-5 * want, "{norm} vs {want}");
}

/// The norm does not depend on the gradients' order or grouping: its sum is
/// exact in integers before one rounding, so 300 gradients (more chunks than
/// a 256-lane finish has lanes, magnitudes over 2^-15..2^14) give the same
/// bits forward, reversed and interleaved, and the f64 norm to 1e-6. A guard
/// for that invariant, not a regression test: the f32-finish norm before
/// tessl's was wired in also passed it (2026-10-17).
#[test]
fn the_norm_is_the_same_bits_in_any_order() {
    let m = metal();
    let hv: Vec<Vec<f32>> = (0..300usize)
        .map(|i| {
            let scale = 2f32.powi((i * 7 % 30) as i32 - 15);
            values(4096, 700 + i as u64, scale)
        })
        .collect();
    let norm = |order: &[usize]| {
        let mut dg: Vec<Tensor> = order
            .iter()
            .map(|&i| m.upload(&vec1(&hv[i])).unwrap())
            .collect();
        m.clip_grad_norm(&mut dg, f32::MAX).unwrap()
    };
    let fwd: Vec<usize> = (0..300).collect();
    let rev: Vec<usize> = (0..300).rev().collect();
    let mixed: Vec<usize> = (0..150).flat_map(|i| [i, 299 - i]).collect();
    let (a, b, c) = (norm(&fwd), norm(&rev), norm(&mixed));
    assert_eq!(a.to_bits(), b.to_bits(), "reversed: {a} vs {b}");
    assert_eq!(a.to_bits(), c.to_bits(), "interleaved: {a} vs {c}");
    let want = hv
        .iter()
        .flatten()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    assert!((f64::from(a) - want).abs() <= 1e-6 * want, "{a} vs {want}");
}

/// Gradients near 1e-30, whose squares are below f32's subnormals, give
/// their norm, not 0.
#[test]
fn clip_norm_of_tiny_values_is_right() {
    let m = metal();
    let v: Vec<f32> = values(9000, 41, 1.0).iter().map(|x| x * 1.0e-30).collect();
    let mut dg = vec![m.upload(&vec1(&v)).unwrap()];
    let norm = m.clip_grad_norm(&mut dg, 1.0).unwrap();
    let want = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    assert!(
        norm > 0.0 && (f64::from(norm) - want).abs() <= 1e-6 * want,
        "{norm} vs {want}"
    );
}

/// An empty gradient is refused at the boundary, before anything is
/// dispatched or scaled.
#[test]
fn an_empty_gradient_is_refused_and_scales_nothing() {
    let m = metal();
    let first = values(100, 77, 1.0);
    let mut dg = vec![m.upload(&vec1(&first)).unwrap(), vec1(&[])];
    m.sync().unwrap();
    let before = m.dispatches();
    match m.clip_grad_norm(&mut dg, 1e-3) {
        Err(OjasError::Shape { op, .. }) => assert_eq!(op, "clip_grad_norm"),
        other => panic!("{other:?}"),
    }
    assert_eq!(m.dispatches() - before, 0);
    assert_eq!(m.download(&dg[0]).unwrap().to_f32_vec().unwrap(), first);
}
