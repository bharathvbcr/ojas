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
    use core::ffi::{c_int, c_long, c_ulong};

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

/// An `f32` buffer reserved without zero-fill and published only after every
/// chunk has been stored.
///
/// `try_new` reserves `n` elements and does not write them. Each
/// [`Self::write_chunk`] hands that chunk to the caller as
/// `&mut [MaybeUninit<f32>]`. The caller must store every lane before
/// returning, including a one-element chunk. [`Self::into_vec`] then
/// [`Vec::set_len`]s, which is the `assume_init` of those lanes. Dropping
/// the buffer earlier frees the allocation with length 0, so uninitialized
/// lanes are not read.
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

    /// Run `write` on chunk `index`. `write` must store an `f32` into every
    /// lane of the slice before it returns, and must not read a lane first.
    /// Its return value is this function's `Ok` value.
    ///
    /// A second call on the same chunk is [`SimdError::OverlappingOutput`]
    /// and does not write. An index past the chunk list is
    /// [`SimdError::OutputLength`] and does not write.
    pub fn write_chunk<R, F>(&self, index: usize, write: F) -> Result<R, SimdError>
    where
        F: FnOnce(&mut [std::mem::MaybeUninit<f32>]) -> R,
    {
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
        // SAFETY: this chunk was claimed, so no other call writes it.
        // `start..start+len` lies inside the `n`-element allocation (a
        // zero-length chunk at `n` is the one-past-end pointer and is not
        // dereferenced). The slice is `MaybeUninit`, so the caller stores
        // with `MaybeUninit::write` and this block does not read the lanes.
        // The pointer is not kept. `into_vec` is what assumes the lanes are
        // initialized, and only after every chunk has returned from here.
        let result = unsafe {
            let p = self.ptr.add(start).cast::<std::mem::MaybeUninit<f32>>();
            write(std::slice::from_raw_parts_mut(p, len))
        };
        self.state[index].store(2, Ordering::Release);
        Ok(result)
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
        // SAFETY: every chunk's `write` returned, and the contract of
        // `write_chunk` is that it stored an `f32` into each lane first.
        // The chunks cover `0..n` without gaps. `capacity >= n`. `f32` has
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

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
unsafe fn write_neg_abs_signs(z: *mut f32, signs: *mut u8, n: usize) -> bool {
    let mut i = 0usize;
    while i < n {
        // SAFETY: `i < n`, and both pointers cover `n` elements.
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
