//! RMSNorm on wgpu: every row width, including the narrow rows that share a
//! workgroup, against `CpuBackend` (`Numerics::Exact`); and `rms_qk_norm`
//! against the composition it replaces (two `rms_norm` calls): the same
//! bits, the same error for the same bad operand, and the same deferred
//! fault when q is recorded before k is refused.

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor, RMS_NORM_EPS};
use ojas_wgpu::WgpuBackend;

fn bits(g: &WgpuBackend, t: &Tensor) -> Vec<u32> {
    g.download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn every_width_and_partial_workgroup_matches_cpu() {
    let c = cpu();
    let g = fresh();
    let dims = [
        1usize, 2, 3, 16, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 768, 1000,
    ];
    for (i, &dim) in dims.iter().enumerate() {
        for &rows in &[1usize, 3, 4, 5, 17, 300] {
            let s = 1000 + 10 * i as u64 + rows as u64;
            let x = host(s, &[rows, dim]);
            let w = host(s + 1, &[dim]);
            let gy = host(s + 2, &[rows, dim]);
            let tag = format!("rms {rows}x{dim}");
            let y = g
                .rms_norm_forward(&g.upload(&x).unwrap(), &g.upload(&w).unwrap(), RMS_NORM_EPS)
                .unwrap();
            let (dx, dw) = g
                .rms_norm_backward(
                    &g.upload(&x).unwrap(),
                    &g.upload(&w).unwrap(),
                    &g.upload(&gy).unwrap(),
                    RMS_NORM_EPS,
                )
                .unwrap();
            g.sync().unwrap_or_else(|e| panic!("{tag}: {e:?}"));
            let down = |t: &Tensor| g.download(t).unwrap().to_f32_vec().unwrap();
            close_vec(
                &format!("{tag} y"),
                &down(&y),
                &c.rms_norm_forward(&x, &w, RMS_NORM_EPS)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap(),
            );
            let (cx, cw) = c.rms_norm_backward(&x, &w, &gy, RMS_NORM_EPS).unwrap();
            close_vec(&format!("{tag} dx"), &down(&dx), &cx.to_f32_vec().unwrap());
            close_vec(&format!("{tag} dw"), &down(&dw), &cw.to_f32_vec().unwrap());
        }
    }
}

#[test]
fn a_non_finite_row_in_a_shared_workgroup_is_reported() {
    let g = fresh();
    for (rows, dim, bad) in [(8usize, 64usize, 5usize), (9, 16, 8), (3, 3, 0)] {
        let mut x = data(7, rows * dim);
        x[bad * dim + dim / 2] = f32::NAN;
        let x = g
            .upload(&Tensor::from_f32(&x, &[rows, dim], host_budget()).unwrap())
            .unwrap();
        let w = g.upload(&host(8, &[dim])).unwrap();
        g.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap();
        match g.sync() {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "rms_norm_forward"),
            other => panic!("{rows}x{dim} row {bad}: {other:?}"),
        }
    }
    // A row whose sum of squares overflows: rstd is 0 and every output is a
    // finite 0, so only the denominator check can report it.
    for (rows, dim, bad) in [
        (8usize, 64usize, 6usize),
        (9, 16, 3),
        (2, 1, 1),
        (3, 300, 2),
    ] {
        let mut x = data(9, rows * dim);
        for v in &mut x[bad * dim..(bad + 1) * dim] {
            *v = 3.0e30;
        }
        let x = g
            .upload(&Tensor::from_f32(&x, &[rows, dim], host_budget()).unwrap())
            .unwrap();
        let w = g.upload(&host(10, &[dim])).unwrap();
        g.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap();
        match g.sync() {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "rms_norm_forward"),
            other => panic!("overflow {rows}x{dim} row {bad}: {other:?}"),
        }
    }
}

/// q `[B, T, H, D]` and k `[B, T, Hkv, D]` with grouped heads.
fn qk(seed: u64, t: usize, h: usize, hkv: usize, d: usize) -> [Tensor; 6] {
    [
        host(seed, &[2, t, h, d]),
        host(seed + 1, &[2, t, hkv, d]),
        host(seed + 2, &[d]),
        host(seed + 3, &[d]),
        host(seed + 4, &[2, t, h, d]),
        host(seed + 5, &[2, t, hkv, d]),
    ]
}

#[test]
fn qk_norm_is_bit_identical_to_two_rms_norm_calls() {
    let g = fresh();
    for (i, &(t, h, hkv, d)) in [
        (1usize, 12usize, 12usize, 64usize),
        (17, 6, 2, 64),
        (33, 4, 1, 128),
        (5, 3, 3, 13),
    ]
    .iter()
    .enumerate()
    {
        let [q, k, qw, kw, gq, gk] =
            qk(2000 + 10 * i as u64, t, h, hkv, d).map(|x| g.upload(&x).unwrap());
        let (yq, yk) = g
            .rms_qk_norm_forward(&q, &k, &qw, &kw, RMS_NORM_EPS)
            .unwrap();
        let wq = g.rms_norm_forward(&q, &qw, RMS_NORM_EPS).unwrap();
        let wk = g.rms_norm_forward(&k, &kw, RMS_NORM_EPS).unwrap();
        assert_eq!(bits(&g, &yq), bits(&g, &wq), "fwd q {t} {h} {hkv} {d}");
        assert_eq!(bits(&g, &yk), bits(&g, &wk), "fwd k {t} {h} {hkv} {d}");
        let got = g
            .rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &gk, RMS_NORM_EPS)
            .unwrap();
        let (dq, dqw) = g.rms_norm_backward(&q, &qw, &gq, RMS_NORM_EPS).unwrap();
        let (dk, dkw) = g.rms_norm_backward(&k, &kw, &gk, RMS_NORM_EPS).unwrap();
        for (name, a, b) in [
            ("dq", &got.0, &dq),
            ("dk", &got.1, &dk),
            ("dqw", &got.2, &dqw),
            ("dkw", &got.3, &dkw),
        ] {
            assert_eq!(bits(&g, a), bits(&g, b), "bwd {name} {t} {h} {hkv} {d}");
        }
    }
}

fn err_text<T: std::fmt::Debug>(r: Result<T, OjasError>) -> String {
    format!("{:?}", r.map(|_| ()).unwrap_err())
}

/// The shape contract (docs/shape-contract.md): `rms_qk_norm_*_dims` checks
/// both pairs before either norm runs, so a bad q is refused with q's own
/// error, a bad k with k's, and a refused call records nothing, not even a
/// clean q's work. A k refused after the validator (here, on the host) is
/// planned before anything is recorded, so it records nothing either.
#[test]
fn qk_norm_refuses_in_the_composition_order() {
    let g = fresh();
    let [q, k, qw, kw, gq, gk] = qk(3000, 5, 4, 2, 32).map(|x| g.upload(&x).unwrap());
    let bad_w = g.upload(&host(3010, &[31])).unwrap();
    let mut qn = data(3011, 2 * 5 * 4 * 32);
    qn[7] = f32::NAN;
    let q_nan = g
        .upload(&Tensor::from_f32(&qn, &[2, 5, 4, 32], host_budget()).unwrap())
        .unwrap();
    let ids = g
        .upload(&Tensor::from_u32(&[0; 32], &[32], host_budget()).unwrap())
        .unwrap();

    // A bad q: the error q's own call gives; nothing recorded.
    let fused = err_text(g.rms_qk_norm_forward(&q, &k, &bad_w, &bad_w, RMS_NORM_EPS));
    let alone = err_text(g.rms_norm_forward(&q, &bad_w, RMS_NORM_EPS));
    assert_eq!(fused, alone);
    let fused = err_text(g.rms_qk_norm_backward(&q, &k, &ids, &kw, &gq, &gk, RMS_NORM_EPS));
    let alone = err_text(g.rms_norm_backward(&q, &ids, &gq, RMS_NORM_EPS));
    assert_eq!(fused, alone);
    g.sync().unwrap();

    // A bad k after a clean q: k's error, and a clean sync.
    let fused = err_text(g.rms_qk_norm_forward(&q, &k, &qw, &bad_w, RMS_NORM_EPS));
    let alone = err_text(g.rms_norm_forward(&k, &bad_w, RMS_NORM_EPS));
    assert_eq!(fused, alone);
    let fused = err_text(g.rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &q, RMS_NORM_EPS));
    let alone = err_text(g.rms_norm_backward(&k, &kw, &q, RMS_NORM_EPS));
    assert_eq!(fused, alone);
    g.sync().unwrap();

    // A bad k after a non-finite q: k's error, and nothing recorded, so the
    // sync is clean (round 4 recorded q first; the contract refuses first).
    let fused = err_text(g.rms_qk_norm_forward(&q_nan, &k, &qw, &bad_w, RMS_NORM_EPS));
    assert_eq!(
        fused,
        err_text(g.rms_norm_forward(&k, &bad_w, RMS_NORM_EPS))
    );
    g.sync().unwrap();
    let fused = err_text(g.rms_qk_norm_backward(&q_nan, &k, &qw, &kw, &gq, &q, RMS_NORM_EPS));
    assert_eq!(
        fused,
        err_text(g.rms_norm_backward(&k, &kw, &q, RMS_NORM_EPS))
    );
    g.sync().unwrap();

    // A well-formed k on the host: Placement, and still nothing recorded.
    let k_host = g.download(&k).unwrap();
    let gk_host = g.download(&gk).unwrap();
    assert!(matches!(
        g.rms_qk_norm_forward(&q_nan, &k_host, &qw, &kw, RMS_NORM_EPS),
        Err(OjasError::Placement { .. })
    ));
    g.sync().unwrap();
    assert!(matches!(
        g.rms_qk_norm_backward(&q_nan, &k_host, &qw, &kw, &gq, &gk_host, RMS_NORM_EPS),
        Err(OjasError::Placement { .. })
    ));
    g.sync().unwrap();
}
