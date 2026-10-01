//! Correctness, determinism and partition-invariance tests for `sgemm_tile`.
//!
//! Tolerance for every reference check: `|got - ref| <= (k+1) * 2^-24 *
//! (Σ_p |A[i,p]·B[p,j]| + |C0[i,j]|)` against an f64 reference (see
//! `common::Case::check`). Backends are also required to agree bit for bit.

mod common;

use common::*;
use ojas_simd::{sgemm_tile, sgemm_tile_with, Backend};

fn check_shape(rng: &mut Rng, m: usize, n: usize, k: usize, ka: Kind, kb: Kind, pad: usize) {
    let a = operand(rng, m, k, ka);
    let b = operand(rng, k, n, kb);
    let (c0, c_rs) = output(rng, m, n, pad);
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
        case.run_all(&format!(
            "{m}x{n}x{k} A={ka:?} B={kb:?} pad={pad} acc={accumulate}"
        ));
    }
}

#[test]
fn detected_backend_is_available_and_named() {
    let d = Backend::detect();
    assert!(d.is_available());
    assert_eq!(ojas_simd::backend_name(), d.name());
    #[cfg(target_arch = "aarch64")]
    assert_eq!(ojas_simd::backend_name(), "neon");
    assert!(Backend::Portable.is_available());
}

#[test]
fn unavailable_backend_is_an_error() {
    for bk in Backend::ALL.into_iter().filter(|b| !b.is_available()) {
        let mut c = [0.0f32; 1];
        let r = sgemm_tile_with(bk, 1, 1, 1, &[1.0], 1, 1, &[1.0], 1, 1, &mut c, 1, false);
        assert_eq!(
            r,
            Err(ojas_simd::SimdError::BackendUnavailable { backend: bk })
        );
        assert_eq!(c, [0.0]);
    }
}

#[test]
fn one_by_one() {
    let mut rng = Rng::new(1);
    check_shape(&mut rng, 1, 1, 1, Kind::RowMajor, Kind::RowMajor, 0);
    let mut c = [10.0f32];
    sgemm_tile(1, 1, 1, &[3.0], 1, 1, &[4.0], 1, 1, &mut c, 1, true).unwrap();
    assert_eq!(c, [22.0]);
    sgemm_tile(1, 1, 1, &[3.0], 1, 1, &[4.0], 1, 1, &mut c, 1, false).unwrap();
    assert_eq!(c, [12.0]);
}

#[test]
#[cfg_attr(miri, ignore)]
fn primes_all_layouts() {
    let mut rng = Rng::new(2);
    for &(m, n, k) in &[(73, 11, 13), (127, 67, 131)] {
        for ka in KINDS {
            for kb in KINDS {
                check_shape(&mut rng, m, n, k, ka, kb, 2);
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn tile_boundaries() {
    let mut rng = Rng::new(3);
    // Register tiles: MR in {6, 8}, NR in {8, 12, 16}.
    for mr in [6usize, 8] {
        for nr in [8usize, 12, 16] {
            for (dm, dn) in [(-1i64, -1i64), (0, 0), (1, 1), (0, 1), (1, 0)] {
                let m = (mr as i64 + dm) as usize;
                let n = (nr as i64 + dn) as usize;
                check_shape(&mut rng, m, n, 7, Kind::RowMajor, Kind::RowMajor, 1);
                check_shape(
                    &mut rng,
                    2 * m + 1,
                    2 * n + 1,
                    5,
                    Kind::ColMajor,
                    Kind::Strided,
                    0,
                );
            }
        }
    }
    // KC = 512 depth blocks.
    for k in [511, 512, 513, 1024, 1025] {
        check_shape(&mut rng, 9, 13, k, Kind::RowMajor, Kind::ColMajor, 0);
    }
    // MC block rows (128, 126 for MR = 6) and NC block columns (960).
    for m in [126, 127, 128, 129, 257] {
        check_shape(&mut rng, m, 5, 3, Kind::PaddedRow(1), Kind::RowMajor, 0);
    }
    for n in [959, 960, 961, 1921] {
        check_shape(&mut rng, 3, n, 3, Kind::RowMajor, Kind::PaddedCol(2), 4);
    }
}

#[test]
fn small_shapes_all_layouts() {
    // Small enough for Miri: every layout on a shape that spans partial tiles.
    let mut rng = Rng::new(4);
    for ka in KINDS {
        for kb in KINDS {
            check_shape(&mut rng, 9, 10, 3, ka, kb, 1);
        }
    }
}

#[test]
fn transposed_operands_need_no_copy() {
    // A is given as Aᵀ (k × m row-major), B as Bᵀ (n × k row-major).
    let (m, n, k) = (5, 7, 4);
    let at: Vec<f32> = (0..k * m).map(|x| x as f32 * 0.5 - 3.0).collect();
    let bt: Vec<f32> = (0..n * k).map(|x| 1.0 - x as f32 * 0.25).collect();
    let mut c = vec![0.0f32; m * n];
    for bk in available_backends() {
        sgemm_tile_with(bk, m, n, k, &at, 1, m, &bt, 1, k, &mut c, n, false).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| at[p * m + i] * bt[j * k + p]).sum();
                assert!((c[i * n + j] - want).abs() <= 1e-4, "{bk:?} C[{i},{j}]");
            }
        }
    }
}

#[test]
fn k_zero_zeroes_or_keeps_c() {
    for bk in available_backends() {
        let mut c = vec![f32::NAN, 7.0, SENTINEL, 1.5, -2.0];
        // m=2, n=2, c_rs=3: valid slots 0,1,3,4; slot 2 is padding.
        sgemm_tile_with(bk, 2, 2, 0, &[], 1, 1, &[], 1, 1, &mut c, 3, true).unwrap();
        assert_eq!(c[1..].to_vec(), vec![7.0, SENTINEL, 1.5, -2.0]);
        assert!(c[0].is_nan());
        sgemm_tile_with(bk, 2, 2, 0, &[], 1, 1, &[], 1, 1, &mut c, 3, false).unwrap();
        assert_eq!(c, vec![0.0, 0.0, SENTINEL, 0.0, 0.0], "{bk:?}");
    }
}

#[test]
fn empty_m_or_n_writes_nothing() {
    for bk in available_backends() {
        let mut c: Vec<f32> = vec![];
        sgemm_tile_with(bk, 0, 5, 3, &[], 3, 1, &[], 5, 1, &mut c, 5, false).unwrap();
        sgemm_tile_with(bk, 4, 0, 3, &[], 3, 1, &[], 0, 1, &mut c, 0, false).unwrap();
        let mut c = vec![3.0f32; 4];
        sgemm_tile_with(bk, 0, 0, 9, &[], 0, 0, &[], 0, 0, &mut c, 0, false).unwrap();
        assert_eq!(c, vec![3.0; 4]);
    }
}

#[test]
fn accumulate_false_never_reads_c() {
    let mut rng = Rng::new(5);
    let (m, n, k) = (13, 17, 9);
    let a = operand(&mut rng, m, k, Kind::RowMajor);
    let b = operand(&mut rng, k, n, Kind::RowMajor);
    let garbage = vec![f32::NAN; m * n];
    let zeros = vec![0.0f32; m * n];
    for bk in available_backends() {
        let mut c1 = garbage.clone();
        let mut c2 = zeros.clone();
        sgemm_tile_with(
            bk, m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c1, n, false,
        )
        .unwrap();
        sgemm_tile_with(
            bk, m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c2, n, false,
        )
        .unwrap();
        assert_bits_eq(&c1, &c2, &format!("{bk:?}"));
        assert!(c1.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn nan_and_inf_propagate() {
    // Shapes are not multiples of any tile, so padding lanes meet Inf too.
    let (m, n, k) = (5, 7, 3);
    let mut rng = Rng::new(6);
    for bk in available_backends() {
        // NaN in A[2,1] poisons all of row 2.
        let mut a = operand(&mut rng, m, k, Kind::RowMajor);
        let b = operand(&mut rng, k, n, Kind::RowMajor);
        a.set(2, 1, f32::NAN);
        let mut c = vec![0.0f32; m * n];
        sgemm_tile_with(
            bk, m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c, n, false,
        )
        .unwrap();
        for i in 0..m {
            for j in 0..n {
                assert_eq!(c[i * n + j].is_nan(), i == 2, "{bk:?} NaN C[{i},{j}]");
            }
        }

        // +Inf in B[0,4] with A[:,0] of known sign gives ±Inf in column 4, and
        // a zero A[3,0] gives 0*Inf = NaN.
        let mut a = operand(&mut rng, m, k, Kind::ColMajor);
        let mut b = operand(&mut rng, k, n, Kind::RowMajor);
        for i in 0..m {
            a.set(i, 0, if i % 2 == 0 { 0.5 } else { -0.5 });
        }
        a.set(3, 0, 0.0);
        b.set(0, 4, f32::INFINITY);
        let mut c = vec![0.0f32; m * n];
        sgemm_tile_with(
            bk, m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c, n, false,
        )
        .unwrap();
        for i in 0..m {
            for j in 0..n {
                let v = c[i * n + j];
                if j != 4 {
                    assert!(v.is_finite(), "{bk:?} C[{i},{j}] = {v}");
                } else if i == 3 {
                    assert!(v.is_nan(), "{bk:?} 0*Inf C[{i},{j}] = {v}");
                } else if i % 2 == 0 {
                    assert_eq!(v, f32::INFINITY, "{bk:?} C[{i},{j}]");
                } else {
                    assert_eq!(v, f32::NEG_INFINITY, "{bk:?} C[{i},{j}]");
                }
            }
        }

        // NaN already in C propagates when accumulating.
        let a = operand(&mut rng, m, k, Kind::RowMajor);
        let b = operand(&mut rng, k, n, Kind::RowMajor);
        let mut c = vec![1.0f32; m * n];
        c[8] = f32::NAN;
        c[9] = f32::INFINITY;
        sgemm_tile_with(
            bk, m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c, n, true,
        )
        .unwrap();
        assert!(c[8].is_nan());
        assert_eq!(c[9], f32::INFINITY);
        assert_eq!(c.iter().filter(|v| !v.is_finite()).count(), 2, "{bk:?}");
    }
}

/// Splits `0..len` into random contiguous pieces.
fn random_cuts(rng: &mut Rng, len: usize) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut s = 0;
    while s < len {
        let w = 1 + rng.below((len - s).min(23));
        out.push((s, w));
        s += w;
    }
    out
}

/// Computes C as independent M×N sub-tile calls by offsetting the slices.
#[allow(clippy::too_many_arguments)]
fn tiled(
    bk: Backend,
    rng: &mut Rng,
    m: usize,
    n: usize,
    k: usize,
    a: &Operand,
    b: &Operand,
    c: &mut [f32],
    c_rs: usize,
    accumulate: bool,
) {
    let rows = random_cuts(rng, m);
    let cols = random_cuts(rng, n);
    for &(i0, h) in &rows {
        for &(j0, w) in &cols {
            sgemm_tile_with(
                bk,
                h,
                w,
                k,
                &a.buf[i0 * a.rs..],
                a.rs,
                a.cs,
                &b.buf[j0 * b.cs..],
                b.rs,
                b.cs,
                &mut c[i0 * c_rs + j0..],
                c_rs,
                accumulate,
            )
            .unwrap();
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn tile_partition_invariance() {
    let mut rng = Rng::new(7);
    for &(m, n, k) in &[(37, 41, 300), (127, 67, 131), (64, 96, 513), (9, 200, 17)] {
        for (ka, kb) in [
            (Kind::RowMajor, Kind::RowMajor),
            (Kind::ColMajor, Kind::Strided),
            (Kind::PaddedRow(2), Kind::ColMajor),
        ] {
            let a = operand(&mut rng, m, k, ka);
            let b = operand(&mut rng, k, n, kb);
            let (c0, c_rs) = output(&mut rng, m, n, 3);
            for accumulate in [false, true] {
                for bk in available_backends() {
                    let mut whole = c0.clone();
                    sgemm_tile_with(
                        bk, m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut whole, c_rs,
                        accumulate,
                    )
                    .unwrap();
                    for trial in 0..4 {
                        let mut parts = c0.clone();
                        tiled(bk, &mut rng, m, n, k, &a, &b, &mut parts, c_rs, accumulate);
                        assert_bits_eq(
                            &whole,
                            &parts,
                            &format!(
                                "{bk:?} {m}x{n}x{k} {ka:?}/{kb:?} acc={accumulate} trial {trial}"
                            ),
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn tile_partition_invariance_across_threads() {
    let mut rng = Rng::new(8);
    let (m, n, k) = (150, 130, 700);
    let a = operand(&mut rng, m, k, Kind::RowMajor);
    let b = operand(&mut rng, k, n, Kind::ColMajor);
    let mut whole = vec![0.0f32; m * n];
    sgemm_tile(
        m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut whole, n, false,
    )
    .unwrap();

    // Row bands on separate threads, each band further cut by columns.
    let mut parts = vec![0.0f32; m * n];
    let bands = random_cuts(&mut rng, m);
    let col_cuts = random_cuts(&mut rng, n);
    std::thread::scope(|s| {
        let mut rest: &mut [f32] = &mut parts;
        let mut consumed = 0;
        for &(i0, h) in &bands {
            let (band, tail) = std::mem::take(&mut rest).split_at_mut((i0 + h - consumed) * n);
            rest = tail;
            consumed = i0 + h;
            let (a, b, col_cuts) = (&a, &b, &col_cuts);
            s.spawn(move || {
                for &(j0, w) in col_cuts {
                    sgemm_tile(
                        h,
                        w,
                        k,
                        &a.buf[i0 * a.rs..],
                        a.rs,
                        a.cs,
                        &b.buf[j0 * b.cs..],
                        b.rs,
                        b.cs,
                        &mut band[j0..],
                        n,
                        false,
                    )
                    .unwrap();
                }
            });
        }
    });
    assert_bits_eq(&whole, &parts, "threaded tiles vs one call");
}

#[test]
#[cfg_attr(miri, ignore)]
fn run_to_run_determinism() {
    let mut rng = Rng::new(9);
    let (m, n, k) = (200, 190, 600);
    let a = operand(&mut rng, m, k, Kind::RowMajor);
    let b = operand(&mut rng, k, n, Kind::RowMajor);
    let (c0, c_rs) = output(&mut rng, m, n, 0);
    for bk in available_backends() {
        let run = |c0: &[f32]| {
            let mut c = c0.to_vec();
            sgemm_tile_with(
                bk, m, n, k, &a.buf, a.rs, a.cs, &b.buf, b.rs, b.cs, &mut c, c_rs, true,
            )
            .unwrap();
            c
        };
        let first = run(&c0);
        for _ in 0..3 {
            assert_bits_eq(&first, &run(&c0), &format!("{bk:?} repeat"));
        }
        // Concurrent calls, each with its own thread-local scratch.
        let results: Vec<Vec<f32>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..4).map(|_| s.spawn(|| run(&c0))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for r in &results {
            assert_bits_eq(&first, r, &format!("{bk:?} concurrent"));
        }
    }
}

/// The chain is exactly `acc = fma(a, b, acc)` from `+0.0` (or old C), in
/// ascending `p`. Checked against a scalar `mul_add` loop bit for bit.
#[test]
fn bits_match_scalar_fma_chain() {
    let mut rng = Rng::new(10);
    // k spans three KC = 512 blocks, so C is reloaded between blocks.
    let (m, n, k) = (11, 14, 1100);
    let a = operand(&mut rng, m, k, Kind::RowMajor);
    let b = operand(&mut rng, k, n, Kind::RowMajor);
    let (c0, c_rs) = output(&mut rng, m, n, 0);
    for accumulate in [false, true] {
        let mut want = c0.clone();
        for i in 0..m {
            for j in 0..n {
                let mut acc = if accumulate { c0[i * c_rs + j] } else { 0.0 };
                for p in 0..k {
                    acc = a.get(i, p).mul_add(b.get(p, j), acc);
                }
                want[i * c_rs + j] = acc;
            }
        }
        for bk in available_backends() {
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
            assert_bits_eq(&want, &case.run(bk), &format!("{bk:?} acc={accumulate}"));
        }
    }
}
