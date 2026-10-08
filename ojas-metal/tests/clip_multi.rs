//! `clip_grad_norm` as one multi-tensor pass: 24 gradients per
//! `ojas_norm_multi` dispatch, one finish, 24 per `ojas_scale_multi`.
//! Before, each gradient took a finite check and two two-stage reductions
//! (5 dispatches) and two more to scale (the scale and its check).

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};

fn vec1(v: &[f32]) -> Tensor {
    host(v, &[v.len()])
}

#[test]
fn clip_dispatches_scale_with_slot_groups_not_tensors() {
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
    assert_eq!(m.dispatches() - before, 170u64.div_ceil(24) + 1);
    let before = m.dispatches();
    let gn2 = m.clip_grad_norm(&mut dg, gn * 0.5).unwrap();
    m.sync().unwrap();
    assert_eq!(m.dispatches() - before, 2 * 170u64.div_ceil(24) + 1);
    let cn = c.clip_grad_norm(&mut hg, 1.0e9).unwrap();
    assert!((gn - cn).abs() <= 1e-5 * cn, "{gn} vs {cn}");
    assert!((gn2 - cn).abs() <= 1e-5 * cn, "{gn2} vs {cn}");
    c.clip_grad_norm(&mut hg, cn * 0.5).unwrap();
    for (i, (d, h)) in dg.iter().zip(&hg).enumerate() {
        let got = m.download(d).unwrap().to_f32_vec().unwrap();
        let want = h.to_f32_vec().unwrap();
        for (j, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-6 + 1e-5 * b.abs(), "{i}[{j}]: {a} vs {b}");
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
