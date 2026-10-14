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
        const MR: usize = crate::Backend::Neon.tile().0;
        const NR: usize = crate::Backend::Neon.tile().1;

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
        const MR: usize = crate::Backend::Avx2Fma.tile().0;
        const NR: usize = crate::Backend::Avx2Fma.tile().1;

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
    use core::ffi::{c_int, c_long, c_ulong};

    pub(super) const CBLAS_ROW_MAJOR: c_int = 101;
    pub(super) const CBLAS_NO_TRANS: c_int = 111;
    pub(super) const CBLAS_TRANS: c_int = 112;
    pub(super) const CBLAS_UPPER: c_int = 121;

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

        // `cblas.h`: `C = alpha A Aᵀ + beta C` (`NoTrans`, `A` is `n × k`)
        // or `alpha Aᵀ A + beta C` (`Trans`, `A` is `k × n`), only the
        // `uplo` triangle of the `n × n` `C` read or written.
        pub(super) fn cblas_ssyrk(
            order: c_int,
            uplo: c_int,
            trans: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );

        // vecLib `vDSP.h`: `C[n] = A[n] * B[n]`, stride in elements.
        // `vDSP_Stride` is `long` and `vDSP_Length` is `unsigned long` on
        // LP64 macOS (the arm64-ilp32 stride is not a macOS ABI).
        pub(super) fn vDSP_vmul(
            a: *const f32,
            ia: c_long,
            b: *const f32,
            ib: c_long,
            c: *mut f32,
            ic: c_long,
            n: c_ulong,
        );
        pub(super) fn vDSP_vadd(
            a: *const f32,
            ia: c_long,
            b: *const f32,
            ib: c_long,
            c: *mut f32,
            ic: c_long,
            n: c_ulong,
        );

        // vecLib `vDSP.h`: matrix move. `__M` is the column count and `__N`
        // is the row count (the header's loop is `C[n][m] = A[n][m]` with
        // `n < N` and `m < M`). `__TA` and `__TC` are the element counts from
        // one row to the next. Lengths are `vDSP_Length` (`unsigned long`),
        // passed by value.
        pub(super) fn vDSP_mmov(
            a: *const f32,
            c: *mut f32,
            m_cols: c_ulong,
            n_rows: c_ulong,
            ta: c_ulong,
            tc: c_ulong,
        );

        // vecLib `vForce.h`:
        // `void vvexpf(float *y, const float *x, const int *n);`
        // `y[i] = exp(x[i])`. The count is a pointer to C `int`.
        pub(super) fn vvexpf(y: *mut f32, x: *const f32, n: *const c_int);
    }

    const _: () = assert!(
        core::mem::size_of::<usize>() == core::mem::size_of::<c_ulong>(),
        "vDSP_Length is unsigned long"
    );
    const _: () = assert!(
        core::mem::size_of::<c_long>() == 8,
        "vDSP_Stride is long on LP64 macOS"
    );
    const _: () = assert!(core::mem::size_of::<c_int>() == 4, "vForce count is C int");
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

/// The upper triangle (`j >= i`) of `C[n×n] = A · Aᵀ` through one
/// `cblas_ssyrk`, `beta = 0`. `call` is the validated `A · Aᵀ` GEMM of
/// [`crate::ssyrk_accelerate`]: `call.a` addresses `A` (`n × k`), and a
/// transposed `A` is passed as its `k × n` storage with `Trans`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn accelerate_ssyrk_upper(call: &crate::layout::BlasCall, a: &[f32], c: &mut [f32]) {
    use accelerate::*;
    assert!(call.m == call.n, "ssyrk output is square");
    assert!(call.n > 0 && call.k > 0, "empty BLAS call");
    assert!(call.beta == 0.0, "ssyrk here never accumulates");
    let (a_rows, a_cols) = if call.a.trans {
        (call.k, call.n)
    } else {
        (call.n, call.k)
    };
    assert!(
        a.len() >= blas_span(a_rows, a_cols, call.a.ld),
        "A shorter than BLAS span"
    );
    assert!(
        c.len() >= blas_span(call.n, call.n, call.ldc),
        "C shorter than BLAS span"
    );
    let trans = if call.a.trans {
        CBLAS_TRANS
    } else {
        CBLAS_NO_TRANS
    };
    // SAFETY: the asserts above check that every element `cblas_ssyrk` may
    // address for these dimensions and leading dimensions lies inside `a`
    // and `c` (it touches a subset of the full `n × n` span of `c`). All
    // dimensions are positive and each `ld` is at least its row length. `c`
    // is an exclusive borrow, so it cannot alias `a`. Accelerate keeps no
    // pointer after the call returns.
    unsafe {
        cblas_ssyrk(
            CBLAS_ROW_MAJOR,
            CBLAS_UPPER,
            trans,
            call.n,
            call.k,
            1.0,
            a.as_ptr(),
            call.a.ld,
            0.0,
            c.as_mut_ptr(),
            call.ldc,
        );
    }
}

/// Which stride-1 two-vector vDSP kernel to run.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
enum VdspOp {
    /// `c[i] = a[i] * b[i]`.
    Mul,
    /// `c[i] = a[i] + b[i]`.
    Add,
}

/// One stride-1 two-vector vDSP op. `a`, `b`, and the `c` region have the
/// same non-zero length and do not overlap.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn vdsp_call(op: VdspOp, a: &[f32], b: &[f32], c: *mut f32) {
    use accelerate::{vDSP_vadd, vDSP_vmul};
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(!a.is_empty());
    // `usize` and `vDSP_Length` (`unsigned long`) are the same width on this
    // target; the asserts next to the declarations lock that in.
    let n = a.len() as core::ffi::c_ulong;
    // SAFETY: the public wrappers check equal lengths, a non-overlapping
    // output, and that `c` addresses `a.len()` writable elements. Stride is
    // 1. vDSP writes every element and does not read `c`, and it does not
    // keep the pointers after the call. Every bit pattern is a valid `f32`.
    unsafe {
        match op {
            VdspOp::Mul => vDSP_vmul(a.as_ptr(), 1, b.as_ptr(), 1, c, 1, n),
            VdspOp::Add => vDSP_vadd(a.as_ptr(), 1, b.as_ptr(), 1, c, 1, n),
        }
    }
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vdsp_vmul(a: &[f32], b: &[f32], c: &mut [f32]) {
    vdsp_call(VdspOp::Mul, a, b, c.as_mut_ptr());
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vdsp_vadd(a: &[f32], b: &[f32], c: &mut [f32]) {
    vdsp_call(VdspOp::Add, a, b, c.as_mut_ptr());
}

/// Writes `n` results into `dst`'s spare capacity and grows `dst` by `n`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn vdsp_append(op: VdspOp, a: &[f32], b: &[f32], dst: &mut Vec<f32>) {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    debug_assert!(n > 0);
    debug_assert!(dst.capacity() - dst.len() >= n);
    let c = dst.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    vdsp_call(op, a, b, c);
    // SAFETY: `vdsp_call` wrote `n` initialized `f32`s at the start of spare
    // capacity, and this function does not reallocate between that write and
    // `set_len`.
    unsafe { dst.set_len(dst.len() + n) }
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vdsp_vmul_append(a: &[f32], b: &[f32], dst: &mut Vec<f32>) {
    vdsp_append(VdspOp::Mul, a, b, dst);
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vdsp_vadd_append(a: &[f32], b: &[f32], dst: &mut Vec<f32>) {
    vdsp_append(VdspOp::Add, a, b, dst);
}

/// Move `rows` rows of `cols` columns. `src_stride` and `dst_stride` are the
/// element counts from one row to the next. The public wrapper has checked
/// that both strides are at least `cols` when `rows > 1`, that `src` covers
/// `(rows - 1) * src_stride + cols` elements, and that `dst` covers the same
/// span for `dst_stride`. The two spans do not overlap. `rows` and `cols`
/// are non-zero.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vdsp_mmov(
    src: &[f32],
    dst: *mut f32,
    rows: usize,
    cols: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    debug_assert!(rows > 0 && cols > 0);
    debug_assert!(rows == 1 || src_stride >= cols);
    debug_assert!(rows == 1 || dst_stride >= cols);
    let src_span = (rows - 1) * src_stride + cols;
    debug_assert!(src.len() >= src_span);
    // SAFETY: the public wrapper checked the spans and rejected an overlap.
    // `vDSP_mmov` writes `C[n][m] = A[n][m]` for `n < rows` and `m < cols`,
    // with row pitches `src_stride` and `dst_stride`, and does not keep
    // either pointer. Column `m` is contiguous, so the last row occupies
    // `cols` elements, not a full pitch. Every bit pattern is a valid `f32`,
    // and the kernel moves bits: it does not round.
    unsafe {
        accelerate::vDSP_mmov(
            src.as_ptr(),
            dst,
            cols as core::ffi::c_ulong,
            rows as core::ffi::c_ulong,
            src_stride as core::ffi::c_ulong,
            dst_stride as core::ffi::c_ulong,
        );
    }
}

/// Writes `rows * cols` contiguous floats (`dst` row stride `cols`) into
/// `dst`'s spare capacity and grows `dst` by that count.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vdsp_mmov_append(
    src: &[f32],
    dst: &mut Vec<f32>,
    rows: usize,
    cols: usize,
    src_stride: usize,
) {
    let n = rows * cols;
    debug_assert!(n > 0);
    debug_assert!(dst.capacity() - dst.len() >= n);
    let dest = dst.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    vdsp_mmov(src, dest, rows, cols, src_stride, cols);
    // SAFETY: `vdsp_mmov` wrote `n` initialized `f32`s at the start of spare
    // capacity, and this function does not reallocate between that write and
    // `set_len`.
    unsafe { dst.set_len(dst.len() + n) }
}

/// `y[i] = exp(x[i])`. `y` and `x` have the same non-zero length, that
/// length fits in `c_int`, and the slices do not overlap.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vvexpf(y: &mut [f32], x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    debug_assert!(!y.is_empty());
    debug_assert!(y.len() <= core::ffi::c_int::MAX as usize);
    let n = y.len() as core::ffi::c_int;
    // SAFETY: `y` and `x` are the same length `n`, `n` fits in `c_int`, and
    // the public wrapper has rejected an overlap. `n` is a local that lives
    // for this call. The header defines the write as element `i` of `y`
    // from element `i` of `x`, and vForce does not keep either pointer.
    // Every bit pattern is a valid `f32`.
    unsafe {
        accelerate::vvexpf(y.as_mut_ptr(), x.as_ptr(), &n);
    }
}

/// In-place `y[i] = exp(y[i])`. `y` is non-empty and its length fits in `c_int`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub(crate) fn vvexpf_inplace(y: &mut [f32]) {
    debug_assert!(!y.is_empty());
    debug_assert!(y.len() <= core::ffi::c_int::MAX as usize);
    let n = y.len() as core::ffi::c_int;
    let p = y.as_mut_ptr();
    // SAFETY: `y` has length `n` and `n` fits in `c_int`. The header sets
    // `y[i]` from `x[i]` alone, so the same address is a valid input and
    // output. `n` is a local that lives for this call. vForce does not keep
    // the pointer. Every bit pattern is a valid `f32`.
    unsafe {
        accelerate::vvexpf(p, p, &n);
    }
}

/// Append one batch of the nanolab head pairs onto `dst`.
///
/// The public wrapper has checked that `src` holds
/// `1024 * 768` elements and that `dst` has that much spare capacity, not
/// overlapping `src`. Group width 2, token tile 32. Each token's 128 source
/// floats are loaded once and stored, 64 into each head block. One head is
/// one 8×8 block: `ldnp` loads it, `trn1`/`trn2` transpose it in registers
/// and transpose it again (a transpose is an involution, so the lanes match
/// the load), and `stnp` writes every lane. Bits are unchanged, including −0.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) fn split_nanolab_head_pairs_append(src: &[f32], dst: &mut Vec<f32>) {
    const N: usize = 1024 * 768;
    debug_assert!(src.len() >= N);
    debug_assert!(dst.capacity() - dst.len() >= N);
    let dest = dst.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    // SAFETY: the public wrapper checked the source span, the spare
    // capacity, and that those ranges do not overlap. `split_nanolab_pairs`
    // stores `N` initialized `f32`s at `dest` and does not reallocate. Neon
    // is enabled for this target.
    unsafe {
        split_nanolab_pairs(src.as_ptr(), dest);
        dst.set_len(dst.len() + N);
    }
}

/// Where a worker band is in its life. `Pending` is the only state a band
/// can be written from; every other state is final once left, except that
/// `Running` ends in `Done` or `Failed`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BandState {
    /// Not started. A write may still begin.
    Pending,
    /// A write is storing lanes. The scope must not end until it finishes.
    Running,
    /// Every lane of the band is stored.
    Done,
    /// The scope ended first. A late write stores nothing.
    Cancelled,
    /// The write unwound part way. Its lanes are not all stored.
    Failed,
}

/// The shared state of one worker band. It holds no pointer, so it is
/// ordinary `Send + Sync` data that may outlive the call.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) struct BandCell {
    state: std::sync::Mutex<BandState>,
    changed: std::sync::Condvar,
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl BandCell {
    pub(crate) fn new() -> Self {
        Self {
            state: std::sync::Mutex::new(BandState::Pending),
            changed: std::sync::Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BandState> {
        // The state is a plain enum that every writer leaves valid, so a
        // poisoned lock still holds a meaningful value.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn set(&self, to: BandState) {
        *self.lock() = to;
        self.changed.notify_all();
    }

    /// Run `store` only if the band is still `Pending`. It is `Running`
    /// while `store` runs, `Done` if it returns and `Failed` if it unwinds.
    /// Returns whether `store` ran.
    pub(crate) fn run(&self, store: impl FnOnce()) -> bool {
        {
            let mut state = self.lock();
            if *state != BandState::Pending {
                return false;
            }
            *state = BandState::Running;
        }
        struct Unwound<'a>(&'a BandCell);
        impl Drop for Unwound<'_> {
            fn drop(&mut self) {
                self.0.set(BandState::Failed);
            }
        }
        let unwound = Unwound(self);
        store();
        std::mem::forget(unwound);
        self.set(BandState::Done);
        true
    }

    /// End the scope: a `Pending` band becomes `Cancelled` without waiting,
    /// a `Running` band is waited for. Returns the state left behind, which
    /// is `Done`, `Cancelled` or `Failed`.
    pub(crate) fn close(&self) -> BandState {
        let mut state = self.lock();
        if *state == BandState::Pending {
            *state = BandState::Cancelled;
        }
        while *state == BandState::Running {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *state
    }
}

/// Ends a [`BandCell`] when `with_nanolab_token_bands` leaves, by return or
/// by unwind, so no worker write is running or can start afterwards.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) struct BandScope(pub(crate) std::sync::Arc<BandCell>);

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl Drop for BandScope {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// The raw ends of one band: where its tokens are read and stored.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[derive(Clone, Copy)]
struct BandSpan {
    src: *const f32,
    dst: *mut f32,
    begin: usize,
    count: usize,
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl BandSpan {
    /// # Safety
    ///
    /// `src` must be readable for the whole batch and `dst` writable for it,
    /// for as long as this call runs, and no other write may touch this
    /// span's destination lanes meanwhile.
    unsafe fn store(self) {
        // SAFETY: the caller's contract. Neon is enabled on the target.
        unsafe { split_nanolab_token_range(self.src, self.dst, self.begin, self.count) }
    }
}

/// The caller's band of one [`crate::with_nanolab_token_bands`] call.
///
/// `'band` is a brand: it is chosen by the call, not the caller, so the
/// band cannot be kept after `body` returns, and the
/// [`NanolabBandWritten`] that [`Self::write`] returns names this band's
/// call alone. Dropping the band stores nothing.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub struct NanolabCallerBand<'band> {
    span: BandSpan,
    // Invariant in `'band`, so the brand cannot be shortened or lengthened
    // to match another call's.
    brand: std::marker::PhantomData<fn(&'band ()) -> &'band ()>,
}

/// Proof that a [`NanolabCallerBand`] stored every lane of its tokens.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub struct NanolabBandWritten<'band> {
    brand: std::marker::PhantomData<fn(&'band ()) -> &'band ()>,
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl<'band> NanolabCallerBand<'band> {
    pub(crate) fn new(src: *const f32, dst: *mut f32, begin: usize, count: usize) -> Self {
        Self {
            span: BandSpan {
                src,
                dst,
                begin,
                count,
            },
            brand: std::marker::PhantomData,
        }
    }

    /// Store this band on the calling thread. The 8x8 body is
    /// [`transpose_8x8_block`].
    pub fn write(self) -> NanolabBandWritten<'band> {
        // SAFETY: the brand ties this band to `body`, which
        // `with_nanolab_token_bands` runs while it holds the `&mut Vec`
        // and the `&[f32]` it checked. It checked the batch, the spare
        // capacity and that the two bands partition `0..1024`, so this
        // band's destination lanes are its own. This consumes the band.
        unsafe { self.span.store() };
        NanolabBandWritten {
            brand: std::marker::PhantomData,
        }
    }
}

/// The worker's band of one [`crate::with_nanolab_token_bands`] call.
///
/// It may move to another thread and outlive the call. Past the call
/// [`Self::write`] stores nothing: the call ends the band first, and a
/// write only starts from the `Pending` state. Dropping the band stores
/// nothing.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub struct NanolabWorkerBand {
    span: BandSpan,
    cell: std::sync::Arc<BandCell>,
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
impl NanolabWorkerBand {
    pub(crate) fn new(
        span_of: (*const f32, *mut f32, usize, usize),
        cell: std::sync::Arc<BandCell>,
    ) -> Self {
        let (src, dst, begin, count) = span_of;
        Self {
            span: BandSpan {
                src,
                dst,
                begin,
                count,
            },
            cell,
        }
    }

    /// Store this band if its call is still running, and do nothing if the
    /// call has ended. Returns whether the band was stored.
    pub fn write(self) -> bool {
        // SAFETY: `run` calls this only while the band is `Pending` and then
        // marks it `Running`. The scope's `close` does not return while a
        // band is `Running`, and a `Pending` band it finds is `Cancelled`
        // and never runs. So the spans' pointers are live for this call, as
        // in `NanolabCallerBand::write`, and the bands partition `0..1024`.
        self.cell.run(|| unsafe { self.span.store() })
    }
}

// SAFETY: the raw pointers in `span` are dereferenced only inside
// `BandCell::run`, which `BandScope` outwaits, so a band sent to another
// thread cannot touch freed memory. The two bands of one call store
// disjoint lanes and only read `src`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe impl Send for NanolabWorkerBand {}

/// Grow `dst` by `n` after both bands have stored their lanes.
///
/// The caller has seen a [`NanolabBandWritten`] and the worker band `Done`:
/// `n` initialized floats are in spare capacity.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) fn commit_nanolab_spare(dst: &mut Vec<f32>, n: usize) {
    debug_assert!(dst.capacity() - dst.len() >= n);
    // SAFETY: the caller wrote `n` initialized floats into spare capacity
    // and this does not reallocate.
    unsafe { dst.set_len(dst.len() + n) }
}

/// # Safety
///
/// Neon must be available. `src` must be readable for `1024 * 768` floats.
/// `dst` must be writable for `12 * 1024 * 64` floats. The ranges must not
/// overlap.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn split_nanolab_pairs(src: *const f32, dst: *mut f32) {
    // SAFETY: the caller checked the full 1024-token batch.
    unsafe { split_nanolab_token_range(src, dst, 0, 1024) }
}

/// # Safety
///
/// Neon must be available. `begin + count` must be at most 1024. `src` must
/// be readable for `1024 * 768` floats. `dst` must be writable for
/// `12 * 1024 * 64` floats. A second call with the same pointers is sound
/// only when its token range does not overlap this one. The ranges must not
/// overlap each other as source and destination.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn split_nanolab_token_range(src: *const f32, dst: *mut f32, begin: usize, count: usize) {
    const TOKENS: usize = 1024;
    const HEADS: usize = 12;
    const DIM: usize = 64;
    const STRIDE: usize = HEADS * DIM;
    const GROUP: usize = 2;
    const TILE: usize = 32;
    let end = begin + count;
    debug_assert!(end <= TOKENS);
    let groups = HEADS / GROUP;
    let head_elems = TOKENS * DIM;
    let mut token0 = begin;
    while token0 < end {
        let tn = TILE.min(end - token0);
        for g in 0..groups {
            // SAFETY: `token0 < 1024`, `g < 6`, so the source offset is at
            // most `1023 * 768 + 10 * 64` and the 128-float load stays
            // inside `1024 * 768`. Each destination head block is
            // `1024 * 64` floats, and `g * 2 + 1` is at most 11. Each head
            // is one 8×8 block. `stnp` of a full `q` register writes all
            // four lanes.
            unsafe {
                let mut sp = src.add(token0 * STRIDE + g * GROUP * DIM);
                let mut d0 = dst.add((g * GROUP) * head_elems + token0 * DIM);
                let mut d1 = dst.add((g * GROUP + 1) * head_elems + token0 * DIM);
                for _ in 0..tn {
                    transpose_8x8_block(sp, d0);
                    transpose_8x8_block(sp.add(DIM), d1);
                    sp = sp.add(STRIDE);
                    d0 = d0.add(DIM);
                    d1 = d1.add(DIM);
                }
            }
        }
        token0 += tn;
    }
}

/// One 64-float head as an 8×8 block.
///
/// `src` is readable for 64 floats and `dst` is writable for 64. The
/// regions do not overlap. Both addresses are 16-byte aligned. `ldnp`
/// loads each row of eight floats. `trn1`/`trn2` transpose that block,
/// then transpose it again so the stored lanes are the loaded lanes.
/// `stnp` writes every lane.
///
/// # Safety
///
/// Neon must be available. `src` and `dst` must be 16-byte aligned, in
/// bounds for 64 floats, and non-overlapping.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn transpose_8x8_block(src: *const f32, dst: *mut f32) {
    // SAFETY: the caller guarantees the 64-float windows, alignment, and
    // that they do not overlap. Neon is enabled on this function. A
    // transpose applied twice is the identity, so every bit, including −0,
    // is stored where it was loaded.
    unsafe {
        core::arch::asm!(
            "ldnp q0, q1, [{src}, #0]",
            "ldnp q2, q3, [{src}, #32]",
            "ldnp q4, q5, [{src}, #64]",
            "ldnp q6, q7, [{src}, #96]",
            "ldnp q8, q9, [{src}, #128]",
            "ldnp q10, q11, [{src}, #160]",
            "ldnp q12, q13, [{src}, #192]",
            "ldnp q14, q15, [{src}, #224]",
            "trn1 v16.4s, v0.4s, v2.4s",
            "trn2 v17.4s, v0.4s, v2.4s",
            "trn1 v18.4s, v4.4s, v6.4s",
            "trn2 v19.4s, v4.4s, v6.4s",
            "trn1 v0.2d, v16.2d, v18.2d",
            "trn1 v2.2d, v17.2d, v19.2d",
            "trn2 v4.2d, v16.2d, v18.2d",
            "trn2 v6.2d, v17.2d, v19.2d",
            "trn1 v16.4s, v1.4s, v3.4s",
            "trn2 v17.4s, v1.4s, v3.4s",
            "trn1 v18.4s, v5.4s, v7.4s",
            "trn2 v19.4s, v5.4s, v7.4s",
            "trn1 v1.2d, v16.2d, v18.2d",
            "trn1 v3.2d, v17.2d, v19.2d",
            "trn2 v5.2d, v16.2d, v18.2d",
            "trn2 v7.2d, v17.2d, v19.2d",
            "trn1 v16.4s, v8.4s, v10.4s",
            "trn2 v17.4s, v8.4s, v10.4s",
            "trn1 v18.4s, v12.4s, v14.4s",
            "trn2 v19.4s, v12.4s, v14.4s",
            "trn1 v8.2d, v16.2d, v18.2d",
            "trn1 v10.2d, v17.2d, v19.2d",
            "trn2 v12.2d, v16.2d, v18.2d",
            "trn2 v14.2d, v17.2d, v19.2d",
            "trn1 v16.4s, v9.4s, v11.4s",
            "trn2 v17.4s, v9.4s, v11.4s",
            "trn1 v18.4s, v13.4s, v15.4s",
            "trn2 v19.4s, v13.4s, v15.4s",
            "trn1 v9.2d, v16.2d, v18.2d",
            "trn1 v11.2d, v17.2d, v19.2d",
            "trn2 v13.2d, v16.2d, v18.2d",
            "trn2 v15.2d, v17.2d, v19.2d",
            "trn1 v24.4s, v0.4s, v2.4s",
            "trn2 v25.4s, v0.4s, v2.4s",
            "trn1 v26.4s, v4.4s, v6.4s",
            "trn2 v27.4s, v4.4s, v6.4s",
            "trn1 v16.2d, v24.2d, v26.2d",
            "trn1 v17.2d, v25.2d, v27.2d",
            "trn2 v18.2d, v24.2d, v26.2d",
            "trn2 v19.2d, v25.2d, v27.2d",
            "trn1 v24.4s, v1.4s, v3.4s",
            "trn2 v25.4s, v1.4s, v3.4s",
            "trn1 v26.4s, v5.4s, v7.4s",
            "trn2 v27.4s, v5.4s, v7.4s",
            "trn1 v20.2d, v24.2d, v26.2d",
            "trn1 v21.2d, v25.2d, v27.2d",
            "trn2 v22.2d, v24.2d, v26.2d",
            "trn2 v23.2d, v25.2d, v27.2d",
            "trn1 v24.4s, v8.4s, v10.4s",
            "trn2 v25.4s, v8.4s, v10.4s",
            "trn1 v26.4s, v12.4s, v14.4s",
            "trn2 v27.4s, v12.4s, v14.4s",
            "trn1 v28.2d, v24.2d, v26.2d",
            "trn1 v29.2d, v25.2d, v27.2d",
            "trn2 v30.2d, v24.2d, v26.2d",
            "trn2 v31.2d, v25.2d, v27.2d",
            "trn1 v24.4s, v9.4s, v11.4s",
            "trn2 v25.4s, v9.4s, v11.4s",
            "trn1 v26.4s, v13.4s, v15.4s",
            "trn2 v27.4s, v13.4s, v15.4s",
            "trn1 v8.2d, v24.2d, v26.2d",
            "trn1 v10.2d, v25.2d, v27.2d",
            "trn2 v12.2d, v24.2d, v26.2d",
            "trn2 v14.2d, v25.2d, v27.2d",
            "stnp q16, q20, [{dst}, #0]",
            "stnp q17, q21, [{dst}, #32]",
            "stnp q18, q22, [{dst}, #64]",
            "stnp q19, q23, [{dst}, #96]",
            "stnp q28, q8, [{dst}, #128]",
            "stnp q29, q10, [{dst}, #160]",
            "stnp q30, q12, [{dst}, #192]",
            "stnp q31, q14, [{dst}, #224]",
            src = in(reg) src,
            dst = in(reg) dst,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            out("v24") _,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack, preserves_flags),
        );
    }
}

/// Floats one pass of [`store_row_blocks`] moves.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const EMBED_BLOCK_FLOATS: usize = 64;

/// # Safety
///
/// Neon must be available. `blocks` must be at least 1. `src` must be
/// readable for `blocks * 64` floats and `dst` writable for as many. The
/// ranges must not overlap. `stnp` of a `q` pair keeps the non-temporal
/// hint when `dst` is 32-byte aligned; a 16-byte aligned address still
/// stores every lane.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn store_row_blocks(src: *const f32, dst: *mut f32, blocks: usize) {
    // SAFETY: `blocks` is at least 1. Each pass reads and writes 64 floats
    // (256 bytes) and then advances both pointers by that many bytes, so
    // `blocks` passes cover the row and stop on its exact end. Neon is a
    // baseline feature of this target, so the `q` registers are available
    // without a separate `target_feature` gate.
    unsafe {
        core::arch::asm!(
            "2:",
            "ldp q0, q1, [{src}, #0]",
            "stnp q0, q1, [{dst}, #0]",
            "ldp q0, q1, [{src}, #32]",
            "stnp q0, q1, [{dst}, #32]",
            "ldp q0, q1, [{src}, #64]",
            "stnp q0, q1, [{dst}, #64]",
            "ldp q0, q1, [{src}, #96]",
            "stnp q0, q1, [{dst}, #96]",
            "ldp q0, q1, [{src}, #128]",
            "stnp q0, q1, [{dst}, #128]",
            "ldp q0, q1, [{src}, #160]",
            "stnp q0, q1, [{dst}, #160]",
            "ldp q0, q1, [{src}, #192]",
            "stnp q0, q1, [{dst}, #192]",
            "ldp q0, q1, [{src}, #224]",
            "stnp q0, q1, [{dst}, #224]",
            "add {src}, {src}, #256",
            "add {dst}, {dst}, #256",
            "subs {blocks}, {blocks}, #1",
            "b.ne 2b",
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            blocks = inout(reg) blocks => _,
            out("v0") _,
            out("v1") _,
            options(nostack),
        );
    }
}

/// Each row is loaded with `ldp` and stored with `stnp`, 64 floats a
/// pass. Nothing is written before the first `stnp`, and `dst`'s length
/// grows only after every row is stored. Bits are unchanged, including −0.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) fn gather_embedding_rows(table: &[f32], row: usize, ids: &[u32], dst: &mut Vec<f32>) {
    let n = ids.len() * row;
    debug_assert!(row > 0 && row.is_multiple_of(EMBED_BLOCK_FLOATS));
    debug_assert!(n == 0 || table.len().is_multiple_of(row));
    debug_assert!(dst.capacity() - dst.len() >= n);
    if n == 0 {
        return;
    }
    let blocks = row / EMBED_BLOCK_FLOATS;
    let dest = dst.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    let base = table.as_ptr();
    // SAFETY: the public wrapper checked the row width, the source span,
    // every id, the spare capacity, and that those ranges do not overlap.
    // Each `store_row_blocks` writes `row` initialized `f32`s. This does
    // not reallocate between those stores and `set_len`. Neon is a
    // baseline feature of this target. Every bit pattern is a valid `f32`.
    unsafe {
        for (i, &id) in ids.iter().enumerate() {
            store_row_blocks(base.add(id as usize * row), dest.add(i * row), blocks);
        }
        dst.set_len(dst.len() + n);
    }
}

/// Writes `gates[h] * head` into `dst`'s spare capacity and grows `dst`.
///
/// `attn.len() == gates.len() * head_dim`, `head_dim > 0`, `attn` is
/// non-empty, spare capacity holds `attn.len()` elements, and that spare
/// range does not overlap `attn` or `gates`.
///
/// Each product is one rounding of `a * g`, including `−0`. The release
/// build turns the loop-invariant scale into a splat `fmul` and stores it
/// with `stp`/`str`.
pub(crate) fn scale_heads_append(attn: &[f32], gates: &[f32], head_dim: usize, dst: &mut Vec<f32>) {
    let n = attn.len();
    debug_assert!(n > 0 && head_dim > 0);
    debug_assert_eq!(gates.len() * head_dim, n);
    debug_assert!(dst.capacity() - dst.len() >= n);
    let dest = dst.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    let src = attn.as_ptr();
    // SAFETY: spare capacity holds `n` elements and does not overlap `attn`
    // or `gates` (the public wrapper checked both). The loop writes `n`
    // initialized `f32`s and does not reallocate. `set_len` then publishes
    // them. Every bit pattern is a valid `f32`.
    unsafe {
        let mut written = 0usize;
        for &g in gates {
            let head = src.add(written);
            let out = dest.add(written);
            for k in 0..head_dim {
                out.add(k).write(head.add(k).read() * g);
            }
            written += head_dim;
        }
        debug_assert_eq!(written, n);
        dst.set_len(dst.len() + n);
    }
}

/// Output filled by storing `-|x|` and then `vvexpf` on those lanes.
///
/// `try_new` reserves `n` elements and does not zero-fill them. Each
/// [`Self::write_chunk`] stores `-|src[i]|` into every lane of that chunk
/// before it reads any of them, then `vvexpf` overwrites those lanes, then
/// `finish` sees initialized memory. [`Self::into_vec`] returns the vector
/// only after every chunk has been stored. Dropping the buffer earlier frees
/// the allocation with length 0, so uninitialized lanes are not read.
///
/// Chunks must be contiguous, non-overlapping, and cover `0..n` in order.
/// Two `write_chunk` calls may run at once only on different chunks.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub struct NegAbsExp {
    data: Vec<f32>,
    ptr: *mut f32,
    n: usize,
    chunks: Vec<(usize, usize)>,
    state: Vec<std::sync::atomic::AtomicU8>,
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
impl NegAbsExp {
    /// Reserve `n` elements, uninitialized. `chunks` is `(start, len)` pairs.
    ///
    /// # Errors
    ///
    /// [`SimdError::OutputLength`] when the chunks do not cover `0..n`.
    /// [`SimdError::LengthTooLarge`] when a chunk end overflows.
    /// [`SimdError::ReserveFailed`] when the allocation cannot be reserved;
    /// nothing is stored.
    pub fn try_new(n: usize, chunks: &[(usize, usize)]) -> Result<Self, SimdError> {
        let mut at = 0usize;
        for &(start, len) in chunks {
            if start != at {
                return Err(SimdError::OutputLength {
                    output: at,
                    expected: n,
                });
            }
            at = at
                .checked_add(len)
                .ok_or(SimdError::LengthTooLarge { len })?;
        }
        if at != n {
            return Err(SimdError::OutputLength {
                output: at,
                expected: n,
            });
        }
        let mut data = Vec::new();
        if data.try_reserve_exact(n).is_err() {
            return Err(SimdError::ReserveFailed { len: n });
        }
        let ptr = data.as_mut_ptr();
        let state = (0..chunks.len())
            .map(|_| std::sync::atomic::AtomicU8::new(0))
            .collect();
        Ok(Self {
            data,
            ptr,
            n,
            chunks: chunks.to_vec(),
            state,
        })
    }

    /// Store `exp(-|src[i]|)` into chunk `index`, then run `finish` on that
    /// initialized slice. `finish`'s bool is returned.
    ///
    /// `src` must be the chunk's length and must not overlap this buffer.
    /// A second call on the same chunk is [`SimdError::OverlappingOutput`]
    /// and does not write.
    ///
    /// # Errors
    ///
    /// [`SimdError::OutputLength`], [`SimdError::OverlappingOutput`].
    pub fn write_chunk<F>(&self, index: usize, src: &[f32], finish: F) -> Result<bool, SimdError>
    where
        F: FnOnce(&mut [f32]) -> bool,
    {
        use std::sync::atomic::Ordering;
        let Some(&(start, len)) = self.chunks.get(index) else {
            return Err(SimdError::OutputLength {
                output: index,
                expected: self.chunks.len(),
            });
        };
        if src.len() != len {
            return Err(SimdError::OutputLength {
                output: src.len(),
                expected: len,
            });
        }
        if chunk_overlaps(src.as_ptr(), src.len(), self.ptr, self.n) {
            return Err(SimdError::OverlappingOutput);
        }
        if self.state[index]
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(SimdError::OverlappingOutput);
        }
        if len > 0 {
            // SAFETY: this chunk was claimed, so no other call writes it.
            // `start..start+len` lies inside the `n`-element allocation.
            // Each lane is stored with `-|x|` before any load. `vvexpf` then
            // reads those stores and writes `exp` over every lane. `finish`
            // runs only after that. The pointer is not kept.
            let finite = unsafe {
                let p = self.ptr.add(start);
                for (i, &v) in src.iter().enumerate() {
                    p.add(i).write(-v.abs());
                }
                const MAX: usize = i32::MAX as usize;
                let mut off = 0usize;
                while off < len {
                    let n = (len - off).min(MAX);
                    vvexpf_inplace(std::slice::from_raw_parts_mut(p.add(off), n));
                    off += n;
                }
                finish(std::slice::from_raw_parts_mut(p, len))
            };
            self.state[index].store(2, Ordering::Release);
            Ok(finite)
        } else {
            // SAFETY: length 0. The slice is not dereferenced. `self.ptr` is
            // the allocation's base, which is non-null and aligned.
            let finite = unsafe { finish(std::slice::from_raw_parts_mut(self.ptr, 0)) };
            self.state[index].store(2, Ordering::Release);
            Ok(finite)
        }
    }

    /// The filled vector, or [`SimdError::Incomplete`] if a chunk was not stored.
    ///
    /// On `Err` the allocation is dropped with length 0.
    pub fn into_vec(mut self) -> Result<Vec<f32>, SimdError> {
        use std::sync::atomic::Ordering;
        let done = self
            .state
            .iter()
            .filter(|s| s.load(Ordering::Acquire) == 2)
            .count();
        if done != self.state.len() {
            return Err(SimdError::Incomplete {
                done,
                expected: self.state.len(),
            });
        }
        // SAFETY: every chunk stored `exp` into each of its lanes before
        // `finish` ran, and the chunks cover `0..n` without gaps. `finish`
        // only wrote initialized `f32`s. `capacity >= n`. `f32` has no drop.
        unsafe { self.data.set_len(self.n) };
        Ok(std::mem::take(&mut self.data))
    }
}

// SAFETY: `ptr` addresses `data`'s heap allocation. `data.len()` stays 0
// until `into_vec`, so `Drop` does not read those lanes. `write_chunk`
// claims its chunk before storing, and the chunks do not overlap, so
// concurrent calls do not alias. The `Vec` header is not mutated until
// `into_vec` takes `self`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
unsafe impl Send for NegAbsExp {}
#[cfg(all(feature = "accelerate", target_os = "macos"))]
unsafe impl Sync for NegAbsExp {}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn chunk_overlaps(a: *const f32, a_len: usize, b: *mut f32, b_len: usize) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let bytes = std::mem::size_of::<f32>();
    let Some(a_bytes) = a_len.checked_mul(bytes) else {
        return true;
    };
    let Some(b_bytes) = b_len.checked_mul(bytes) else {
        return true;
    };
    let a0 = a as usize;
    let b0 = b as usize;
    let Some(a1) = a0.checked_add(a_bytes) else {
        return true;
    };
    let Some(b1) = b0.checked_add(b_bytes) else {
        return true;
    };
    a0 < b1 && b0 < a1
}

/// Every source slice is `len` long.
fn check_sources(srcs: &[&[f32]], len: usize) -> Result<(), SimdError> {
    match srcs.iter().find(|src| src.len() != len) {
        Some(src) => Err(SimdError::OutputLength {
            output: src.len(),
            expected: len,
        }),
        None => Ok(()),
    }
}

/// An `f32` buffer reserved without zero-fill and published only after every
/// chunk has been stored.
///
/// `try_new` reserves `n` elements and does not write them. Each
/// [`Self::write_chunk`] (or its `_map` and `_pair_map` forms) stores a value
/// into every lane of one chunk itself, from a per-lane closure, and marks
/// the chunk complete only after the last lane. The caller never holds the
/// uninitialized slice. [`Self::into_vec`] then [`Vec::set_len`]s, which is
/// the `assume_init` of lanes that were all stored. Dropping the buffer
/// earlier frees the allocation with length 0, so uninitialized lanes are
/// not read.
///
/// Chunks must be contiguous, non-overlapping, and cover `0..n` in order.
/// Two `write_chunk` calls may run at once only on different chunks.
pub struct ReservedF32 {
    data: Vec<f32>,
    ptr: *mut f32,
    n: usize,
    chunks: Vec<(usize, usize)>,
    state: Vec<std::sync::atomic::AtomicU8>,
}

impl ReservedF32 {
    /// Reserve `n` elements, uninitialized. `chunks` is `(start, len)` pairs.
    ///
    /// # Errors
    ///
    /// [`SimdError::OutputLength`] when the chunks do not cover `0..n`.
    /// [`SimdError::LengthTooLarge`] when a chunk end overflows.
    /// [`SimdError::ReserveFailed`] when the allocation cannot be reserved;
    /// nothing is stored.
    pub fn try_new(n: usize, chunks: &[(usize, usize)]) -> Result<Self, SimdError> {
        let mut at = 0usize;
        for &(start, len) in chunks {
            if start != at {
                return Err(SimdError::OutputLength {
                    output: at,
                    expected: n,
                });
            }
            at = at
                .checked_add(len)
                .ok_or(SimdError::LengthTooLarge { len })?;
        }
        if at != n {
            return Err(SimdError::OutputLength {
                output: at,
                expected: n,
            });
        }
        let mut data = Vec::new();
        if data.try_reserve_exact(n).is_err() {
            return Err(SimdError::ReserveFailed { len: n });
        }
        let ptr = data.as_mut_ptr();
        let state = (0..chunks.len())
            .map(|_| std::sync::atomic::AtomicU8::new(0))
            .collect();
        Ok(Self {
            data,
            ptr,
            n,
            chunks: chunks.to_vec(),
            state,
        })
    }

    /// Claim chunk `index` and return its `(start, len)`.
    fn claim(&self, index: usize) -> Result<(usize, usize), SimdError> {
        use std::sync::atomic::Ordering;
        let Some(&(start, len)) = self.chunks.get(index) else {
            return Err(SimdError::OutputLength {
                output: index,
                expected: self.chunks.len(),
            });
        };
        if self.state[index]
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(SimdError::OverlappingOutput);
        }
        Ok((start, len))
    }

    /// Store `lane(i)` into lane `i` of chunk `index`, for `i` in
    /// `0..chunk_len`, in order.
    ///
    /// This function does the storing, so a chunk is marked complete only
    /// after every one of its lanes has been written. `lane` cannot read the
    /// output. If `lane` panics the chunk stays unmarked and
    /// [`Self::into_vec`] is [`SimdError::Incomplete`].
    ///
    /// A second call on the same chunk is [`SimdError::OverlappingOutput`]
    /// and does not call `lane`. An index past the chunk list is
    /// [`SimdError::OutputLength`] and does not call `lane`.
    pub fn write_chunk<F>(&self, index: usize, mut lane: F) -> Result<(), SimdError>
    where
        F: FnMut(usize) -> f32,
    {
        use std::sync::atomic::Ordering;
        let (start, len) = self.claim(index)?;
        // SAFETY: this chunk was claimed, so no other call writes it.
        // `start..start+len` lies inside the `n`-element allocation (a
        // zero-length chunk at `n` is the one-past-end pointer, which is
        // not written). Each lane is stored once and none is read. The
        // pointer is not kept. `into_vec` assumes the lanes are initialized
        // only for chunks marked below, which is after this loop stored all
        // `len` of them.
        unsafe {
            let p = self.ptr.add(start);
            for i in 0..len {
                p.add(i).write(lane(i));
            }
        }
        self.state[index].store(2, Ordering::Release);
        Ok(())
    }

    /// Store `lane([srcs[0][i], srcs[1][i], ..])` into lane `i` of chunk
    /// `index`: [`Self::write_chunk`] over `K` source slices of the chunk's
    /// length, which are checked first. Because the lengths are known to be
    /// equal the per-lane reads need no bounds check, so the loop is the
    /// same machine code as a zip of the slices.
    ///
    /// A source of another length is [`SimdError::OutputLength`] and the
    /// chunk is not claimed. Otherwise as [`Self::write_chunk`].
    #[allow(clippy::needless_range_loop)] // `i` indexes every source slice
    pub fn write_chunk_map<const K: usize, F>(
        &self,
        index: usize,
        srcs: [&[f32]; K],
        mut lane: F,
    ) -> Result<(), SimdError>
    where
        F: FnMut([f32; K]) -> f32,
    {
        use std::sync::atomic::Ordering;
        let len = self.chunk_len(index)?;
        check_sources(&srcs, len)?;
        let (start, len) = self.claim(index)?;
        // SAFETY: as in `write_chunk`. The sources are `len` long, so
        // `srcs[j][i]` is in range for `i < len`.
        unsafe {
            let p = self.ptr.add(start);
            for i in 0..len {
                p.add(i).write(lane(std::array::from_fn(|j| srcs[j][i])));
            }
        }
        self.state[index].store(2, Ordering::Release);
        Ok(())
    }

    /// [`Self::write_chunk_map`] into chunk `index` of this buffer and of
    /// `other` at once: lane `i` of this chunk is `lane(..).0` and lane `i`
    /// of `other`'s is `lane(..).1`.
    ///
    /// `other` must be a different buffer with the same chunk at `index`.
    /// The same buffer is [`SimdError::OverlappingOutput`] and a different
    /// chunk is [`SimdError::OutputLength`]; neither calls `lane`. A chunk
    /// of either buffer that was already claimed is
    /// [`SimdError::OverlappingOutput`] and nothing stays claimed by this
    /// call.
    #[allow(clippy::needless_range_loop)] // `i` indexes every source slice
    pub fn write_chunk_pair_map<const K: usize, F>(
        &self,
        other: &Self,
        index: usize,
        srcs: [&[f32]; K],
        mut lane: F,
    ) -> Result<(), SimdError>
    where
        F: FnMut([f32; K]) -> (f32, f32),
    {
        use std::sync::atomic::Ordering;
        if std::ptr::eq(self, other) {
            return Err(SimdError::OverlappingOutput);
        }
        let len = self.chunk_len(index)?;
        let other_len = other.chunk_len(index)?;
        if other.chunks[index] != self.chunks[index] {
            return Err(SimdError::OutputLength {
                output: other_len,
                expected: len,
            });
        }
        check_sources(&srcs, len)?;
        let (start, len) = self.claim(index)?;
        if let Err(err) = other.claim(index) {
            self.state[index].store(0, Ordering::Release);
            return Err(err);
        }
        // SAFETY: as in `write_chunk`, for both buffers. They are distinct
        // allocations, so the two pointers do not alias, both chunks are
        // claimed and have this `start` and `len`, and the sources are
        // `len` long.
        unsafe {
            let p = self.ptr.add(start);
            let q = other.ptr.add(start);
            for i in 0..len {
                let (u, v) = lane(std::array::from_fn(|j| srcs[j][i]));
                p.add(i).write(u);
                q.add(i).write(v);
            }
        }
        self.state[index].store(2, Ordering::Release);
        other.state[index].store(2, Ordering::Release);
        Ok(())
    }

    /// The length of chunk `index`, without claiming it.
    fn chunk_len(&self, index: usize) -> Result<usize, SimdError> {
        self.chunks
            .get(index)
            .map(|&(_, len)| len)
            .ok_or(SimdError::OutputLength {
                output: index,
                expected: self.chunks.len(),
            })
    }

    /// The filled vector, or [`SimdError::Incomplete`] if a chunk was not stored.
    ///
    /// On `Err` the allocation is dropped with length 0.
    pub fn into_vec(mut self) -> Result<Vec<f32>, SimdError> {
        use std::sync::atomic::Ordering;
        let done = self
            .state
            .iter()
            .filter(|s| s.load(Ordering::Acquire) == 2)
            .count();
        if done != self.state.len() {
            return Err(SimdError::Incomplete {
                done,
                expected: self.state.len(),
            });
        }
        // SAFETY: every chunk is marked 2, and `write_chunk` and
        // `write_chunk_map` and `write_chunk_pair_map` mark a chunk only after their own loop stored
        // an `f32` into each of its lanes. The chunks cover `0..n` without
        // gaps. `capacity >= n`. `f32` has
        // no drop. `set_len` is the `assume_init` of those lanes.
        unsafe { self.data.set_len(self.n) };
        Ok(std::mem::take(&mut self.data))
    }
}

// SAFETY: `ptr` addresses `data`'s heap allocation. `data.len()` stays 0
// until `into_vec`, so `Drop` does not read those lanes. `write_chunk`
// claims its chunk before storing, and the chunks do not overlap, so
// concurrent calls do not alias. The `Vec` header is not mutated until
// `into_vec` takes `self`.
unsafe impl Send for ReservedF32 {}
unsafe impl Sync for ReservedF32 {}

/// Fill `signs` (length 0, capacity at least `z.len()`) and store `-|z[i]|`
/// when every lane is finite.
///
/// `false` means a chunk held NaN or an infinity. That chunk is not stored,
/// `signs.len()` stays 0, and earlier chunks of `z` may already hold `-|x|`.
/// On `true`, every sign byte is written before [`Vec::set_len`].
pub(crate) fn store_neg_abs_signs(z: &mut [f32], signs: &mut Vec<u8>) -> bool {
    let n = z.len();
    debug_assert!(signs.capacity() >= n);
    debug_assert_eq!(signs.len(), 0);
    // SAFETY: `signs` has room for `n` bytes and its length is 0, so this
    // write does not alias a live element. `z` has `n` initialized lanes.
    // `set_len` publishes sign bytes only after every lane was finite.
    // `u8` has no destructor.
    unsafe {
        let ok = write_neg_abs_signs(z.as_mut_ptr(), signs.as_mut_ptr(), n);
        if ok {
            signs.set_len(n);
        }
        ok
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn write_neg_abs_signs(z: *mut f32, signs: *mut u8, n: usize) -> bool {
    use core::arch::aarch64::{
        vabsq_f32, vcltq_f32, vcombine_u16, vcombine_u8, vdupq_n_f32, vdupq_n_u32, vld1q_f32,
        vmaxvq_u32, vmovn_u16, vmovn_u32, vnegq_f32, vorrq_u32, vshrq_n_u32, vst1_u8, vst1q_f32,
        vst1q_u8,
    };
    // SAFETY: `target_feature = "neon"` is enabled on this function. `z`
    // holds `n` lanes and `signs` has room for `n` bytes. A chunk is loaded
    // once. Its magnitude bits are compared with the infinity exponent
    // (`f32::is_finite`) before any sign or `-|x|` of that chunk is stored.
    // A non-finite lane returns without storing that chunk. `-0 < 0` is
    // false, and `vneg(vabs(±0))` is `-0`.
    let zero = vdupq_n_f32(0.0);
    let mag = vdupq_n_u32(0x7fff_ffff);
    let inf_bits = vdupq_n_u32(0x7f80_0000);
    let mut i = 0usize;
    while i + 16 <= n {
        unsafe {
            let x0 = vld1q_f32(z.add(i));
            let x1 = vld1q_f32(z.add(i + 4));
            let x2 = vld1q_f32(z.add(i + 8));
            let x3 = vld1q_f32(z.add(i + 12));
            let bad = vorrq_u32(
                vorrq_u32(
                    quad_nonfinite(x0, mag, inf_bits),
                    quad_nonfinite(x1, mag, inf_bits),
                ),
                vorrq_u32(
                    quad_nonfinite(x2, mag, inf_bits),
                    quad_nonfinite(x3, mag, inf_bits),
                ),
            );
            if vmaxvq_u32(bad) != 0 {
                return false;
            }
            let c0 = vcltq_f32(x0, zero);
            let c1 = vcltq_f32(x1, zero);
            let c2 = vcltq_f32(x2, zero);
            let c3 = vcltq_f32(x3, zero);
            let y0 = vnegq_f32(vabsq_f32(x0));
            let y1 = vnegq_f32(vabsq_f32(x1));
            let y2 = vnegq_f32(vabsq_f32(x2));
            let y3 = vnegq_f32(vabsq_f32(x3));
            let p01 = vcombine_u16(
                vmovn_u32(vshrq_n_u32(c0, 31)),
                vmovn_u32(vshrq_n_u32(c1, 31)),
            );
            let p23 = vcombine_u16(
                vmovn_u32(vshrq_n_u32(c2, 31)),
                vmovn_u32(vshrq_n_u32(c3, 31)),
            );
            let bytes = vcombine_u8(vmovn_u16(p01), vmovn_u16(p23));
            vst1q_u8(signs.add(i), bytes);
            vst1q_f32(z.add(i), y0);
            vst1q_f32(z.add(i + 4), y1);
            vst1q_f32(z.add(i + 8), y2);
            vst1q_f32(z.add(i + 12), y3);
        }
        i += 16;
    }
    while i + 4 <= n {
        unsafe {
            let x = vld1q_f32(z.add(i));
            if vmaxvq_u32(quad_nonfinite(x, mag, inf_bits)) != 0 {
                return false;
            }
            let c = vcltq_f32(x, zero);
            let y = vnegq_f32(vabsq_f32(x));
            let low = vmovn_u32(vshrq_n_u32(c, 31));
            let narrow = vmovn_u16(vcombine_u16(low, low));
            let mut tmp = [0u8; 8];
            vst1_u8(tmp.as_mut_ptr(), narrow);
            core::ptr::copy_nonoverlapping(tmp.as_ptr(), signs.add(i), 4);
            vst1q_f32(z.add(i), y);
        }
        i += 4;
    }
    while i < n {
        unsafe {
            let v = z.add(i).read();
            if !v.is_finite() {
                return false;
            }
            signs.add(i).write(u8::from(v < 0.0));
            z.add(i).write(-v.abs());
        }
        i += 1;
    }
    true
}

/// All-ones in a lane whose magnitude bits are at least the infinity
/// exponent, else zero. That is `!x.is_finite()` per lane.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
fn quad_nonfinite(
    x: core::arch::aarch64::float32x4_t,
    mag: core::arch::aarch64::uint32x4_t,
    inf_bits: core::arch::aarch64::uint32x4_t,
) -> core::arch::aarch64::uint32x4_t {
    use core::arch::aarch64::{vandq_u32, vcgeq_u32, vreinterpretq_u32_f32};
    vcgeq_u32(vandq_u32(vreinterpretq_u32_f32(x), mag), inf_bits)
}

/// The NEON path's chunks (16 lanes, then groups of 4, then single lanes),
/// so a non-finite lane stops before any lane of its chunk is stored, as
/// [`crate::store_neg_abs_signs`] documents.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
unsafe fn write_neg_abs_signs(z: *mut f32, signs: *mut u8, n: usize) -> bool {
    let mut i = 0usize;
    while i < n {
        let width = if i + 16 <= n {
            16
        } else if i + 4 <= n {
            4
        } else {
            1
        };
        // SAFETY: `i + width <= n`, and both pointers cover `n` elements.
        unsafe {
            for j in i..i + width {
                if !z.add(j).read().is_finite() {
                    return false;
                }
            }
            for j in i..i + width {
                let v = z.add(j).read();
                signs.add(j).write(u8::from(v < 0.0));
                z.add(j).write(-v.abs());
            }
        }
        i += width;
    }
    true
}

#[cfg(all(test, target_arch = "aarch64", target_feature = "neon"))]
mod band_cell_tests {
    use super::{BandCell, BandScope, BandState};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{mpsc, Arc};

    #[test]
    fn a_pending_band_is_cancelled_without_waiting_and_a_late_run_does_nothing() {
        let cell = BandCell::new();
        assert_eq!(cell.close(), BandState::Cancelled);
        let mut ran = false;
        assert!(!cell.run(|| ran = true));
        assert!(!ran);
        assert_eq!(cell.close(), BandState::Cancelled);
    }

    #[test]
    fn a_completed_run_is_done_and_runs_once() {
        let cell = BandCell::new();
        let mut count = 0;
        assert!(cell.run(|| count += 1));
        assert!(!cell.run(|| count += 1));
        assert_eq!(count, 1);
        assert_eq!(cell.close(), BandState::Done);
    }

    #[test]
    fn a_run_that_unwinds_is_failed_and_does_not_hang_close() {
        let cell = BandCell::new();
        let outcome = catch_unwind(AssertUnwindSafe(|| cell.run(|| panic!("store fails"))));
        assert!(outcome.is_err());
        assert_eq!(cell.close(), BandState::Failed);
    }

    #[test]
    fn close_waits_for_a_running_band() {
        let cell = Arc::new(BandCell::new());
        let (started_tx, started) = mpsc::channel();
        let (release_tx, release) = mpsc::channel::<()>();
        let worker = {
            let cell = Arc::clone(&cell);
            std::thread::spawn(move || {
                cell.run(|| {
                    started_tx.send(()).unwrap();
                    release.recv().unwrap();
                })
            })
        };
        started.recv().unwrap();
        let closer = {
            let cell = Arc::clone(&cell);
            std::thread::spawn(move || cell.close())
        };
        // `close` cannot return while the store is still running.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!closer.is_finished());
        release_tx.send(()).unwrap();
        assert!(worker.join().unwrap());
        assert_eq!(closer.join().unwrap(), BandState::Done);
    }

    #[test]
    fn the_scope_guard_ends_the_band_on_unwind() {
        let cell = Arc::new(BandCell::new());
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _scope = BandScope(Arc::clone(&cell));
            panic!("body fails");
        }));
        assert!(outcome.is_err());
        assert!(!cell.run(|| unreachable!("a cancelled band never runs")));
    }
}
