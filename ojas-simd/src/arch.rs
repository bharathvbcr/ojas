//! The only module in the crate that contains `unsafe` code.
//!
//! Each entry point takes slices and asserts the lengths its raw-pointer loop
//! touches before entering `unsafe`. The driver only passes packed panels it
//! sized itself, so those assertions never fire, but no caller is trusted for
//! memory safety.
//!
//! The intrinsic blocks also wrap value-only intrinsics. Before Rust 1.87
//! those were `unsafe fn`, and the workspace MSRV is 1.82.

use crate::gemm::MicroKernel;
use crate::layout::Problem;
use crate::{Backend, SimdError};

/// Asserts that `kc` packed steps of an `mr × nr` tile fit in the slices.
#[cfg_attr(
    not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        target_arch = "x86_64"
    )),
    allow(dead_code)
)]
fn assert_panels(kc: usize, mr: usize, nr: usize, a: &[f32], b: &[f32], acc: &[f32]) {
    let a_need = kc.checked_mul(mr).expect("kc*MR overflows");
    let b_need = kc.checked_mul(nr).expect("kc*NR overflows");
    assert!(a.len() >= a_need, "packed A panel too short");
    assert!(b.len() >= b_need, "packed B panel too short");
    assert_eq!(acc.len(), mr * nr, "accumulator tile has the wrong size");
}

// ---------------------------------------------------------------- NEON ----

pub(crate) fn neon_available() -> bool {
    cfg!(all(target_arch = "aarch64", target_feature = "neon"))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::{assert_panels, MicroKernel};
    use core::arch::aarch64::{float32x4_t, vfmaq_laneq_f32, vld1q_f32, vst1q_f32};

    /// 8×12 tile: 24 accumulator registers, 2 for A, 3 for B.
    #[derive(Clone, Copy)]
    pub(crate) struct Neon;

    impl MicroKernel for Neon {
        const MR: usize = 8;
        const NR: usize = 12;

        fn run(self, kc: usize, a: &[f32], b: &[f32], acc: &mut [f32]) {
            assert_panels(kc, 8, 12, a, b, acc);
            // SAFETY: the `target_feature = "neon"` cfg on this module means
            // the whole crate is compiled with NEON. `assert_panels` has just
            // checked `a.len() >= kc*8`, `b.len() >= kc*12` and
            // `acc.len() == 96`, which is everything `kernel_8x12` dereferences.
            unsafe { kernel_8x12(kc, a.as_ptr(), b.as_ptr(), acc.as_mut_ptr()) }
        }
    }

    /// # Safety
    ///
    /// NEON must be available. `a` must be readable for `kc*8` floats, `b`
    /// for `kc*12`, and `c` readable and writable for 96.
    #[target_feature(enable = "neon")]
    #[allow(unused_unsafe)]
    unsafe fn kernel_8x12(kc: usize, a: *const f32, b: *const f32, c: *mut f32) {
        let mut acc: [[float32x4_t; 3]; 8];
        // SAFETY: `c` is valid for 96 floats; row `r` reads `c[r*12 .. r*12+12]`.
        unsafe {
            acc = core::array::from_fn(|r| {
                core::array::from_fn(|q| vld1q_f32(c.add(r * 12 + q * 4)))
            });
        }
        for p in 0..kc {
            // SAFETY: `p < kc`, so `p*8 + 8 <= kc*8` floats of `a` and
            // `p*12 + 12 <= kc*12` floats of `b` are in bounds.
            unsafe {
                let a0 = vld1q_f32(a.add(p * 8));
                let a1 = vld1q_f32(a.add(p * 8 + 4));
                let b0 = vld1q_f32(b.add(p * 12));
                let b1 = vld1q_f32(b.add(p * 12 + 4));
                let b2 = vld1q_f32(b.add(p * 12 + 8));
                macro_rules! row {
                    ($r:literal, $av:ident, $lane:literal) => {
                        acc[$r][0] = vfmaq_laneq_f32::<$lane>(acc[$r][0], b0, $av);
                        acc[$r][1] = vfmaq_laneq_f32::<$lane>(acc[$r][1], b1, $av);
                        acc[$r][2] = vfmaq_laneq_f32::<$lane>(acc[$r][2], b2, $av);
                    };
                }
                row!(0, a0, 0);
                row!(1, a0, 1);
                row!(2, a0, 2);
                row!(3, a0, 3);
                row!(4, a1, 0);
                row!(5, a1, 1);
                row!(6, a1, 2);
                row!(7, a1, 3);
            }
        }
        // SAFETY: as for the loads above. `c` is writable for 96 floats.
        unsafe {
            for (r, row) in acc.iter().enumerate() {
                for (q, v) in row.iter().enumerate() {
                    vst1q_f32(c.add(r * 12 + q * 4), *v);
                }
            }
        }
    }
}

pub(crate) fn run_neon(p: &Problem, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), SimdError> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        crate::gemm::gemm(neon::Neon, p, a, b, c);
        Ok(())
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        let _ = (p, a, b, c);
        Err(SimdError::BackendUnavailable {
            backend: Backend::Neon,
        })
    }
}

// ----------------------------------------------------------- AVX2 + FMA ----

pub(crate) fn avx2_fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{assert_panels, MicroKernel};
    use core::arch::x86_64::{
        __m256, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps,
    };

    /// 6×16 tile: 12 accumulator registers, 2 for B, 1 broadcast of A.
    ///
    /// Only [`Avx2Fma::detect`] constructs one, so holding a value proves the
    /// CPU supports AVX2 and FMA.
    #[derive(Clone, Copy)]
    pub(crate) struct Avx2Fma {
        _detected: (),
    }

    impl Avx2Fma {
        pub(crate) fn detect() -> Option<Avx2Fma> {
            super::avx2_fma_available().then_some(Avx2Fma { _detected: () })
        }
    }

    impl MicroKernel for Avx2Fma {
        const MR: usize = 6;
        const NR: usize = 16;

        fn run(self, kc: usize, a: &[f32], b: &[f32], acc: &mut [f32]) {
            assert_panels(kc, 6, 16, a, b, acc);
            // SAFETY: `self` exists only if `detect` saw AVX2 and FMA at
            // runtime. `assert_panels` checked `a.len() >= kc*6`,
            // `b.len() >= kc*16` and `acc.len() == 96`.
            unsafe { kernel_6x16(kc, a.as_ptr(), b.as_ptr(), acc.as_mut_ptr()) }
        }
    }

    /// # Safety
    ///
    /// AVX2 and FMA must be available. `a` must be readable for `kc*6` floats,
    /// `b` for `kc*16`, and `c` readable and writable for 96.
    #[target_feature(enable = "avx2,fma")]
    #[allow(unused_unsafe)]
    unsafe fn kernel_6x16(kc: usize, a: *const f32, b: *const f32, c: *mut f32) {
        let mut acc: [[__m256; 2]; 6];
        // SAFETY: `c` is valid for 96 floats; row `r` reads `c[r*16 .. r*16+16]`.
        unsafe {
            acc = core::array::from_fn(|r| {
                core::array::from_fn(|q| _mm256_loadu_ps(c.add(r * 16 + q * 8)))
            });
        }
        for p in 0..kc {
            // SAFETY: `p < kc`, so `b[p*16 .. p*16+16]` and `a[p*6 .. p*6+6]`
            // are in bounds.
            unsafe {
                let b0 = _mm256_loadu_ps(b.add(p * 16));
                let b1 = _mm256_loadu_ps(b.add(p * 16 + 8));
                for (r, row) in acc.iter_mut().enumerate() {
                    let av = _mm256_broadcast_ss(&*a.add(p * 6 + r));
                    row[0] = _mm256_fmadd_ps(av, b0, row[0]);
                    row[1] = _mm256_fmadd_ps(av, b1, row[1]);
                }
            }
        }
        // SAFETY: `c` is writable for 96 floats.
        unsafe {
            for (r, row) in acc.iter().enumerate() {
                for (q, v) in row.iter().enumerate() {
                    _mm256_storeu_ps(c.add(r * 16 + q * 8), *v);
                }
            }
        }
    }
}

pub(crate) fn run_avx2_fma(
    p: &Problem,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), SimdError> {
    #[cfg(target_arch = "x86_64")]
    if let Some(kern) = avx2::Avx2Fma::detect() {
        crate::gemm::gemm(kern, p, a, b, c);
        return Ok(());
    }
    let _ = (p, a, b, c);
    Err(SimdError::BackendUnavailable {
        backend: Backend::Avx2Fma,
    })
}

// ----------------------------------------------------------- Accelerate ----

#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod accelerate {
    use core::ffi::c_int;

    pub(super) const CBLAS_ROW_MAJOR: c_int = 101;
    pub(super) const CBLAS_NO_TRANS: c_int = 111;
    pub(super) const CBLAS_TRANS: c_int = 112;

    // Signature from the macOS SDK's vecLib `cblas.h`. CBLAS enums are C
    // `int`.
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        pub(super) fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }
}

/// Elements a row-major BLAS operand of `rows × cols` with leading dimension
/// `ld` spans: `(rows-1)*ld + cols`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn blas_span(rows: i32, cols: i32, ld: i32) -> usize {
    let (rows, cols, ld) = (rows as usize, cols as usize, ld as usize);
    assert!(ld >= cols.max(1), "leading dimension below column count");
    (rows - 1)
        .checked_mul(ld)
        .and_then(|r| r.checked_add(cols))
        .expect("BLAS span overflows")
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn accelerate_sgemm(
    call: &crate::layout::BlasCall,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    use accelerate::*;
    assert!(call.m > 0 && call.n > 0 && call.k > 0, "empty BLAS call");
    let (a_rows, a_cols) = if call.a.trans {
        (call.k, call.m)
    } else {
        (call.m, call.k)
    };
    let (b_rows, b_cols) = if call.b.trans {
        (call.n, call.k)
    } else {
        (call.k, call.n)
    };
    assert!(
        a.len() >= blas_span(a_rows, a_cols, call.a.ld),
        "A shorter than BLAS span"
    );
    assert!(
        b.len() >= blas_span(b_rows, b_cols, call.b.ld),
        "B shorter than BLAS span"
    );
    assert!(
        c.len() >= blas_span(call.m, call.n, call.ldc),
        "C shorter than BLAS span"
    );
    let trans = |t: bool| if t { CBLAS_TRANS } else { CBLAS_NO_TRANS };
    // SAFETY: the asserts above check that every element `cblas_sgemm` may
    // address for these dimensions and leading dimensions lies inside `a`, `b`
    // and `c`. All dimensions are positive and each `ld` is at least its row
    // length. `c` is an exclusive borrow, so it cannot alias `a` or `b`.
    // Accelerate keeps no pointer after the call returns.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            trans(call.a.trans),
            trans(call.b.trans),
            call.m,
            call.n,
            call.k,
            1.0,
            a.as_ptr(),
            call.a.ld,
            b.as_ptr(),
            call.b.ld,
            call.beta,
            c.as_mut_ptr(),
            call.ldc,
        );
    }
}
