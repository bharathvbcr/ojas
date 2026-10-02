//! Fast-tier `f32` GEMM for ojas.
//!
//! `ojas-cpu` is the Exact tier: no FMA, ascending-`k` reductions. This crate
//! is the Fast tier. It uses FMA and SIMD, and with the `accelerate` feature
//! it can call Apple's BLAS.
//!
//! # Determinism contract (`sgemm_tile`, `sgemm_tile_with`)
//!
//! Each output element is one fused multiply-add chain in ascending `p`:
//!
//! ```text
//! acc = if accumulate { C[i,j] } else { +0.0 }
//! for p in 0..k { acc = fma(A[i,p], B[p,j], acc) }   // one rounding per step
//! C[i,j] = acc
//! ```
//!
//! Every backend (NEON, AVX2+FMA, portable `f32::mul_add`) runs this chain.
//! The internal blocking, register tiling, and padding depend on no element's
//! value, and each SIMD lane is an independent chain. So:
//!
//! - the bits of `C[i,j]` depend only on row `i` of `A`, column `j` of `B`,
//!   `k`, `accumulate` and the old `C[i,j]`;
//! - splitting `C` into M×N tiles across calls or threads gives bits identical
//!   to one call, provided every call passes the full `k`;
//! - runs are bit-identical, and so are the three backends (the tests check
//!   this). NaN payloads are not part of the contract.
//!
//! This differs from `ojas-cpu`: an FMA chain rounds once per step, and the
//! Exact tier rounds the product and the sum separately.
//!
//! [`sgemm_accelerate`] does not follow this contract. See its docs.
//!
//! # Safety
//!
//! Every length and stride is validated with checked arithmetic before any
//! `unsafe` code runs. A bad shape returns [`SimdError`]. It never panics and
//! never reads out of bounds. All `unsafe` code is in the private `arch` module.

#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

#[cfg(all(feature = "accelerate", not(target_os = "macos")))]
compile_error!(
    "the `accelerate` feature of ojas-simd links Apple's Accelerate framework and is macOS-only"
);

#[allow(unsafe_code)]
mod arch;
mod gemm;
mod layout;

use core::fmt;

/// Names one matrix argument in a [`SimdError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    /// The left input, `m × k`.
    A,
    /// The right input, `k × n`.
    B,
    /// The output, `m × n`.
    C,
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Operand::A => "A",
            Operand::B => "B",
            Operand::C => "C",
        })
    }
}

/// Why a GEMM call was refused. A refused call has not modified `C`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimdError {
    /// The index of the operand's last element, `(rows-1)*rs + (cols-1)*cs`,
    /// does not fit in `usize`.
    ExtentOverflow {
        /// The operand whose extent overflowed.
        operand: Operand,
    },
    /// The slice is shorter than the extent its shape and strides require.
    BufferTooShort {
        /// The operand that is too short.
        operand: Operand,
        /// Required length, `(rows-1)*rs + (cols-1)*cs + 1`.
        required: usize,
        /// Actual slice length.
        len: usize,
    },
    /// `m > 1` and `c_rs < n`, so two output elements would share a slot.
    OverlappingOutputRows {
        /// Output columns.
        n: usize,
        /// Output row stride.
        c_rs: usize,
    },
    /// The requested backend is not compiled in or not supported by this CPU.
    BackendUnavailable {
        /// The backend that was requested.
        backend: Backend,
    },
    /// [`sgemm_accelerate`] only: BLAS needs a unit stride on one axis and a
    /// leading dimension at least the length of the other axis.
    UnsupportedLayout {
        /// The operand BLAS cannot address.
        operand: Operand,
    },
    /// [`sgemm_accelerate`] only: a dimension or leading dimension exceeds
    /// `i32::MAX`, the largest value CBLAS `int` can hold.
    DimensionTooLarge {
        /// The value that does not fit.
        value: usize,
    },
}

impl fmt::Display for SimdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdError::ExtentOverflow { operand } => {
                write!(f, "extent of operand {operand} overflows usize")
            }
            SimdError::BufferTooShort {
                operand,
                required,
                len,
            } => write!(
                f,
                "operand {operand} needs {required} elements but the slice has {len}"
            ),
            SimdError::OverlappingOutputRows { n, c_rs } => write!(
                f,
                "output row stride {c_rs} is smaller than n = {n}, so rows overlap"
            ),
            SimdError::BackendUnavailable { backend } => {
                write!(f, "backend {} is not available on this CPU", backend.name())
            }
            SimdError::UnsupportedLayout { operand } => write!(
                f,
                "operand {operand} has no unit stride with a valid leading dimension for BLAS"
            ),
            SimdError::DimensionTooLarge { value } => {
                write!(f, "dimension {value} exceeds the CBLAS int range")
            }
        }
    }
}

impl std::error::Error for SimdError {}

/// A micro-kernel family for [`sgemm_tile_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// aarch64 NEON 8×12 register tile (`vfmaq_laneq_f32`).
    Neon,
    /// x86_64 AVX2 + FMA 6×16 register tile, chosen by runtime detection.
    Avx2Fma,
    /// Safe Rust 8×8 tile using `f32::mul_add`. Available on every target.
    /// Where the CPU has no FMA unit, `mul_add` calls a correctly rounded
    /// software `fmaf`. That is slow, but the bits match the other backends.
    Portable,
}

impl Backend {
    /// Every backend, available or not.
    pub const ALL: [Backend; 3] = [Backend::Neon, Backend::Avx2Fma, Backend::Portable];

    /// Stable name: `"neon"`, `"avx2-fma"` or `"portable-fma"`.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Neon => "neon",
            Backend::Avx2Fma => "avx2-fma",
            Backend::Portable => "portable-fma",
        }
    }

    /// Whether this backend can run in this process.
    pub fn is_available(self) -> bool {
        match self {
            Backend::Neon => arch::neon_available(),
            Backend::Avx2Fma => arch::avx2_fma_available(),
            Backend::Portable => true,
        }
    }

    /// The backend [`sgemm_tile`] uses: NEON on aarch64, AVX2+FMA when the
    /// CPU reports both features, otherwise portable.
    pub fn detect() -> Backend {
        if arch::neon_available() {
            Backend::Neon
        } else if arch::avx2_fma_available() {
            Backend::Avx2Fma
        } else {
            Backend::Portable
        }
    }
}

/// Name of the backend [`sgemm_tile`] dispatches to in this process:
/// `"neon"`, `"avx2-fma"` or `"portable-fma"`.
///
/// [`sgemm_accelerate`] is never chosen implicitly, so its name
/// (`"accelerate"`) is not returned here.
pub fn backend_name() -> &'static str {
    Backend::detect().name()
}

/// `C[m×n] (+)= A[m×k] · B[k×n]` on the [`Backend::detect`] backend.
///
/// Addressing, with every stride counted in elements:
///
/// - `A[i,p] = a[i*a_rs + p*a_cs]`
/// - `B[p,j] = b[p*b_rs + j*b_cs]`
/// - `C[i,j] = c[i*c_rs + j]`. Columns of `C` are contiguous.
///
/// A transposed input needs no copy. For example, `Aᵀ` stored row-major as
/// `k × m` is `a_rs = 1, a_cs = m`. A stride of 0 broadcasts a row or column.
/// `a` and `b` are shared borrows and `c` is exclusive, so the inputs cannot
/// alias the output.
///
/// With `accumulate == false`, the old contents of `C` are never read. NaN or
/// garbage there does not leak into the result. With `k == 0`, `C` is zeroed
/// when `accumulate` is false and untouched when it is true. When `m == 0` or
/// `n == 0` nothing is written.
///
/// The bits of each element depend only on that element's inputs and `k`. See
/// the crate-level determinism contract. A caller may split `M` and `N` into
/// tiles across threads by offsetting the slices. Every tile must pass the
/// full `k`.
///
/// Scratch: each thread keeps one reusable packing buffer of at most
/// `(MC + NC) * KC = (128 + 960) * 512` floats (2.125 MiB). It is sized to the
/// largest call that thread has made.
///
/// # Errors
///
/// [`SimdError::BufferTooShort`] or [`SimdError::ExtentOverflow`] when an
/// operand's last addressed element is outside its slice.
/// [`SimdError::OverlappingOutputRows`] when `m > 1 && c_rs < n`. `C` is
/// unmodified on error.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_tile(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    c: &mut [f32],
    c_rs: usize,
    accumulate: bool,
) -> Result<(), SimdError> {
    sgemm_tile_with(
        Backend::detect(),
        m,
        n,
        k,
        a,
        a_rs,
        a_cs,
        b,
        b_rs,
        b_cs,
        c,
        c_rs,
        accumulate,
    )
}

/// [`sgemm_tile`] on an explicit backend. The semantics and output bits are
/// identical.
///
/// # Errors
///
/// As for [`sgemm_tile`], plus [`SimdError::BackendUnavailable`] when
/// `!backend.is_available()`. Shape errors are reported first.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_tile_with(
    backend: Backend,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    c: &mut [f32],
    c_rs: usize,
    accumulate: bool,
) -> Result<(), SimdError> {
    let p = layout::Problem::validate(m, n, k, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, accumulate)?;
    match backend {
        Backend::Portable => {
            gemm::gemm(gemm::Portable, &p, a, b, c);
            Ok(())
        }
        Backend::Neon => arch::run_neon(&p, a, b, c),
        Backend::Avx2Fma => arch::run_avx2_fma(&p, a, b, c),
    }
}

/// `C[m×n] (+)= A[m×k] · B[k×n]` through Accelerate's `cblas_sgemm`.
///
/// The arguments and addressing are those of [`sgemm_tile`]. Each operand
/// must be BLAS-addressable: either its column stride is 1 and its row stride
/// is at least its column count, or its row stride is 1 (the transposed case)
/// and its column stride is at least its row count. A stride along an axis of
/// length 1 is ignored. `C` needs `c_rs >= n`. Zero-stride broadcasts are
/// refused. `accumulate` maps to `beta = 1.0` or `beta = 0.0`. With
/// `beta = 0.0`, Accelerate does not read `C`, so NaN there is not propagated
/// (the tests check this).
///
/// Determinism: Accelerate chooses its own blocking, summation order, kernel
/// (NEON, AMX or SME) and internal thread count. For a fixed problem on one
/// machine and OS build the tests observe bit-identical repeats. The order is
/// not specified and can change across macOS releases. The result depends on
/// the whole `(m, n, k)` shape, so it is **not** tile-partition invariant and
/// does not match the [`sgemm_tile`] bits. Call it once per whole problem, not
/// per tile. Do not call it from inside a parallel tile loop either:
/// Accelerate already threads internally, and nesting oversubscribes the
/// cores. Accuracy is still within the usual `γ_k` error bound.
///
/// # Errors
///
/// The [`sgemm_tile`] shape errors, [`SimdError::UnsupportedLayout`], and
/// [`SimdError::DimensionTooLarge`] when a dimension or leading dimension
/// exceeds `i32::MAX`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn sgemm_accelerate(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    c: &mut [f32],
    c_rs: usize,
    accumulate: bool,
) -> Result<(), SimdError> {
    let p = layout::Problem::validate(m, n, k, a, a_rs, a_cs, b, b_rs, b_cs, c, c_rs, accumulate)?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        gemm::zero_unless_accumulate(&p, c);
        return Ok(());
    }
    let call = layout::BlasCall::from_problem(&p)?;
    arch::accelerate_sgemm(&call, a, b, c);
    Ok(())
}
