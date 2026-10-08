//! `sgemm_accelerate` and `ssyrk_accelerate` against the f64 reference.
//! Built only with `--features accelerate` on macOS.
//!
//! Accelerate's summation order is not ours, so there is no bit comparison with
//! `sgemm_tile`. The tolerance is the same `γ_{k+1}` bound, with a factor of 2
//! for headroom.
#![cfg(all(feature = "accelerate", target_os = "macos"))]

mod common;

use common::*;
use ojas_simd::{sgemm_accelerate, ssyrk_accelerate, Operand as Op, SimdError};

fn run(case: &Case) -> Result<Vec<f32>, SimdError> {
    let mut c = case.c0.to_vec();
    sgemm_accelerate(
        case.m,
        case.n,
        case.k,
        &case.a.buf,
        case.a.rs,
        case.a.cs,
        &case.b.buf,
        case.b.rs,
        case.b.cs,
        &mut c,
        case.c_rs,
        case.accumulate,
    )?;
    Ok(c)
}

const BLAS_KINDS: [Kind; 4] = [
    Kind::RowMajor,
    Kind::ColMajor,
    Kind::PaddedRow(3),
    Kind::PaddedCol(2),
];

#[test]
fn matches_reference_on_blas_layouts() {
    let mut rng = Rng::new(21);
    let shapes = [
        (1, 1, 1),
        (73, 11, 13),
        (127, 67, 131),
        (1, 9, 5),
        (9, 1, 5),
        (7, 8, 1),
        (300, 257, 513),
    ];
    for &(m, n, k) in &shapes {
        for ka in BLAS_KINDS {
            for kb in BLAS_KINDS {
                let a = operand(&mut rng, m, k, ka);
                let b = operand(&mut rng, k, n, kb);
                let (c0, c_rs) = output(&mut rng, m, n, 2);
                for accumulate in [false, true] {
                    let case = Case {
                        m,
                        n,
                        k,
                        a: &a,
                        b: &b,
                        c0: &c0,
                        c_rs,
                        accumulate,
                    };
                    let got = run(&case).unwrap();
                    case.check(
                        &got,
                        2.0,
                        &format!("accelerate {m}x{n}x{k} {ka:?}/{kb:?} acc={accumulate}"),
                    );
                }
            }
        }
    }
}

#[test]
fn beta_zero_ignores_nan_in_c_and_k_zero() {
    let mut rng = Rng::new(22);
    let (m, n, k) = (6, 5, 4);
    let a = operand(&mut rng, m, k, Kind::RowMajor);
    let b = operand(&mut rng, k, n, Kind::RowMajor);
    let mut c = vec![f32::NAN; m * n];
    sgemm_accelerate(
        m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c, n, false,
    )
    .unwrap();
    assert!(c.iter().all(|v| v.is_finite()), "beta = 0 read C: {c:?}");

    let mut c = vec![2.0f32; m * n];
    sgemm_accelerate(m, n, 0, &[], 1, 1, &[], 1, 1, &mut c, n, true).unwrap();
    assert_eq!(c, vec![2.0; m * n]);
    sgemm_accelerate(m, n, 0, &[], 1, 1, &[], 1, 1, &mut c, n, false).unwrap();
    assert_eq!(c, vec![0.0; m * n]);
}

#[test]
fn run_to_run_repeatable() {
    let mut rng = Rng::new(23);
    let (m, n, k) = (384, 320, 512);
    let a = operand(&mut rng, m, k, Kind::RowMajor);
    let b = operand(&mut rng, k, n, Kind::ColMajor);
    let (c0, c_rs) = output(&mut rng, m, n, 0);
    let case = Case {
        m,
        n,
        k,
        a: &a,
        b: &b,
        c0: &c0,
        c_rs,
        accumulate: true,
    };
    let first = run(&case).unwrap();
    for _ in 0..5 {
        assert_bits_eq(&first, &run(&case).unwrap(), "accelerate repeat");
    }
}

#[test]
fn refuses_what_blas_cannot_address() {
    let mut rng = Rng::new(24);
    let (m, n, k) = (4, 5, 3);
    let b = operand(&mut rng, k, n, Kind::RowMajor);
    for ka in [Kind::Strided, Kind::BroadcastRows] {
        let a = operand(&mut rng, m, k, ka);
        let mut c = vec![1.0f32; m * n];
        let r = sgemm_accelerate(
            m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c, n, false,
        );
        assert_eq!(
            r,
            Err(SimdError::UnsupportedLayout { operand: Op::A }),
            "{ka:?}"
        );
        assert_eq!(c, vec![1.0; m * n]);
    }
    let a = operand(&mut rng, m, k, Kind::RowMajor);
    let mut c = vec![1.0f32; m * n];
    let r = sgemm_accelerate(
        m,
        n,
        k,
        &a.buf,
        a.rs,
        a.cs,
        &b.buf[..b.buf.len() - 1],
        b.rs,
        b.cs,
        &mut c,
        n,
        false,
    );
    assert!(matches!(
        r,
        Err(SimdError::BufferTooShort { operand: Op::B, .. })
    ));
    let r = sgemm_accelerate(
        m,
        n,
        k,
        &a.buf,
        a.rs,
        a.cs,
        &b.buf,
        b.rs,
        b.cs,
        &mut c,
        n - 1,
        false,
    );
    assert_eq!(r, Err(SimdError::OverlappingOutputRows { n, c_rs: n - 1 }));
    let big = i32::MAX as usize + 1;
    // Zero strides keep the buffers at one element while k exceeds CBLAS int.
    let r = sgemm_accelerate(1, 1, big, &[1.0], 1, 0, &[1.0], 0, 1, &mut [0.0], 1, false);
    assert_eq!(r, Err(SimdError::DimensionTooLarge { value: big }));
}

/// `a` read as its transpose: the same buffer with the strides swapped.
fn transposed(a: &Operand) -> Operand {
    Operand {
        buf: a.buf.clone(),
        rs: a.cs,
        cs: a.rs,
        rows: a.cols,
        cols: a.rows,
    }
}

#[test]
fn ssyrk_matches_reference_and_is_exactly_symmetric() {
    let mut rng = Rng::new(25);
    for &(n, k) in &[(1, 1), (2, 7), (73, 11), (128, 300), (257, 64)] {
        for ka in BLAS_KINDS {
            let a = operand(&mut rng, n, k, ka);
            let at = transposed(&a);
            for pad in [0, 3] {
                let (c0, c_rs) = output(&mut rng, n, n, pad);
                let mut got = c0.clone();
                ssyrk_accelerate(n, k, &a.buf, a.rs, a.cs, &mut got, c_rs).unwrap();
                let case = Case {
                    m: n,
                    n,
                    k,
                    a: &a,
                    b: &at,
                    c0: &c0,
                    c_rs,
                    accumulate: false,
                };
                let what = format!("ssyrk {n}x{k} {ka:?} pad={pad}");
                case.check(&got, 2.0, &what);
                for i in 0..n {
                    for j in 0..i {
                        assert_eq!(
                            got[i * c_rs + j].to_bits(),
                            got[j * c_rs + i].to_bits(),
                            "{what}: C[{i},{j}] is not C[{j},{i}]"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn ssyrk_ignores_c_and_zeroes_at_k_zero() {
    let mut rng = Rng::new(26);
    let (n, k) = (6, 4);
    let a = operand(&mut rng, n, k, Kind::RowMajor);
    let mut c = vec![f32::NAN; n * n];
    ssyrk_accelerate(n, k, &a.buf, a.rs, a.cs, &mut c, n).unwrap();
    assert!(c.iter().all(|v| v.is_finite()), "ssyrk read C: {c:?}");
    let mut c = vec![2.0f32; n * n];
    ssyrk_accelerate(n, 0, &[], 1, 1, &mut c, n).unwrap();
    assert_eq!(c, vec![0.0; n * n]);
    ssyrk_accelerate(0, k, &[], 1, 1, &mut [], 1).unwrap();
}

#[test]
fn ssyrk_refuses_what_blas_cannot_address() {
    let mut rng = Rng::new(27);
    let (n, k) = (4, 3);
    for ka in [Kind::Strided, Kind::BroadcastRows] {
        let a = operand(&mut rng, n, k, ka);
        let mut c = vec![1.0f32; n * n];
        let r = ssyrk_accelerate(n, k, &a.buf, a.rs, a.cs, &mut c, n);
        assert!(
            matches!(r, Err(SimdError::UnsupportedLayout { .. })),
            "{ka:?}: {r:?}"
        );
        assert_eq!(c, vec![1.0; n * n]);
    }
    let a = operand(&mut rng, n, k, Kind::RowMajor);
    let mut c = vec![1.0f32; n * n];
    let r = ssyrk_accelerate(n, k, &a.buf[..a.buf.len() - 1], a.rs, a.cs, &mut c, n);
    assert!(matches!(r, Err(SimdError::BufferTooShort { .. })), "{r:?}");
    let r = ssyrk_accelerate(n, k, &a.buf, a.rs, a.cs, &mut c, n - 1);
    assert_eq!(r, Err(SimdError::OverlappingOutputRows { n, c_rs: n - 1 }));
    assert_eq!(c, vec![1.0; n * n]);
}
