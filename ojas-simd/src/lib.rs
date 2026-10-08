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
//! With the `accelerate` feature on macOS, [`vdsp_vmul`], [`vdsp_vadd`],
//! and their append forms call the stride-1 vDSP kernels. Each output is
//! one rounding of `a * b` or `a + b`. [`vdsp_mmov`] and
//! [`vdsp_mmov_append`] copy rows of a matrix; that copy is a move, so
//! every bit is preserved, including −0. [`vvexpf`] and [`vvexpf_inplace`]
//! call vForce `vvexpf` (`y[i] = exp(x[i])`). Those functions are not
//! compiled on any other target. [`store_neg_abs_signs`] loads each chunk
//! of `z` once, refuses a non-finite lane, records `z < 0`, and stores
//! `-|z|` before the next chunk.
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

/// Why a call was refused. A refused call has not modified the output.
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
    /// Two-vector vDSP: the inputs differ in length.
    MismatchedLengths {
        /// Length of the first input.
        a: usize,
        /// Length of the second input.
        b: usize,
    },
    /// vDSP or [`vvexpf`]: the output is not the input length. For the
    /// append forms, `output` is spare capacity and `expected` is the input
    /// length.
    OutputLength {
        /// Length of the output slice, or spare capacity of the append buffer.
        output: usize,
        /// Length the output must have.
        expected: usize,
    },
    /// vDSP or [`vvexpf`]: the output overlaps an input.
    OverlappingOutput,
    /// [`vvexpf`]: the length does not fit in a C `int`.
    LengthTooLarge {
        /// The length that does not fit.
        len: usize,
    },
    /// [`ReservedF32`] (and `NegAbsExp` on macOS): the allocation could not
    /// be reserved. Nothing was stored.
    ReserveFailed {
        /// Element count that was requested.
        len: usize,
    },
    /// [`ReservedF32::into_vec`] (and `NegAbsExp::into_vec` on macOS): not
    /// every chunk has been stored.
    Incomplete {
        /// Chunks whose lanes were stored.
        done: usize,
        /// Chunks the buffer was split into.
        expected: usize,
    },
    /// [`store_neg_abs_signs`]: a lane was NaN or an infinity.
    /// `signs` was not published. Lanes before that chunk may hold `-|x|`.
    NonFinite,
    /// [`gather_embedding_rows`]: the row width is not a positive multiple
    /// of the kernel's 64-float block.
    RowWidth {
        /// The row width that was asked for.
        row: usize,
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
            SimdError::MismatchedLengths { a, b } => {
                write!(f, "vDSP inputs have lengths {a} and {b}")
            }
            SimdError::OutputLength { output, expected } => {
                write!(f, "output has {output} elements, needs {expected}")
            }
            SimdError::OverlappingOutput => {
                write!(f, "output overlaps an input")
            }
            SimdError::LengthTooLarge { len } => {
                write!(f, "vector length {len} exceeds the C int range")
            }
            SimdError::ReserveFailed { len } => {
                write!(f, "could not reserve {len} elements")
            }
            SimdError::Incomplete { done, expected } => {
                write!(f, "stored {done} of {expected} chunks")
            }
            SimdError::NonFinite => write!(f, "a value is not finite"),
            SimdError::RowWidth { row } => {
                write!(f, "row width {row} is not a positive multiple of 64")
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

    /// The `(MR, NR)` register tile of this backend's micro-kernel. The
    /// kernels take their tile from here, so [`sgemm_tile_scratch`] cannot
    /// drift from them.
    pub(crate) const fn tile(self) -> (usize, usize) {
        match self {
            Backend::Neon => (8, 12),
            Backend::Avx2Fma => (6, 16),
            Backend::Portable => (8, 8),
        }
    }

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
/// largest call that thread has made; [`sgemm_tile_scratch`] gives the floats
/// one call needs.
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

/// Floats of the packing buffer one [`sgemm_tile`] call of `m × n × k` needs
/// on the calling thread, on the [`Backend::detect`] backend. 0 when any
/// dimension is 0, where no buffer is touched.
///
/// The buffer is thread-local and kept for the thread's lifetime, grown to
/// the largest call. A caller that charges a budget for tiles run on fresh
/// threads charges this once per thread; a thread that already holds a
/// buffer this large allocates nothing. At most 557,056 floats (2.125 MiB).
pub fn sgemm_tile_scratch(m: usize, n: usize, k: usize) -> usize {
    if m == 0 || n == 0 || k == 0 {
        return 0;
    }
    let (mr, nr) = Backend::detect().tile();
    let blocks = gemm::Blocks::new(mr, nr, m, n, k);
    blocks.a_len + blocks.b_len
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

/// True when the half-open element ranges `[a, a+a_len)` and `[c, c+c_len)` overlap.
fn ranges_overlap(a: *const f32, a_len: usize, c: *const f32, c_len: usize) -> bool {
    if a_len == 0 || c_len == 0 {
        return false;
    }
    let bytes = core::mem::size_of::<f32>();
    let Some(a_bytes) = a_len.checked_mul(bytes) else {
        return true;
    };
    let Some(c_bytes) = c_len.checked_mul(bytes) else {
        return true;
    };
    let a0 = a as usize;
    let c0 = c as usize;
    let Some(a1) = a0.checked_add(a_bytes) else {
        return true;
    };
    let Some(c1) = c0.checked_add(c_bytes) else {
        return true;
    };
    a0 < c1 && c0 < a1
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn vdsp_slices(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    call: fn(&[f32], &[f32], &mut [f32]),
) -> Result<(), SimdError> {
    if a.len() != b.len() {
        return Err(SimdError::MismatchedLengths {
            a: a.len(),
            b: b.len(),
        });
    }
    if c.len() != a.len() {
        return Err(SimdError::OutputLength {
            output: c.len(),
            expected: a.len(),
        });
    }
    if ranges_overlap(a.as_ptr(), a.len(), c.as_ptr(), c.len())
        || ranges_overlap(b.as_ptr(), b.len(), c.as_ptr(), c.len())
    {
        return Err(SimdError::OverlappingOutput);
    }
    if a.is_empty() {
        return Ok(());
    }
    call(a, b, c);
    Ok(())
}

#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn vdsp_into_spare(
    a: &[f32],
    b: &[f32],
    dst: &mut Vec<f32>,
    call: fn(&[f32], &[f32], &mut Vec<f32>),
) -> Result<(), SimdError> {
    if a.len() != b.len() {
        return Err(SimdError::MismatchedLengths {
            a: a.len(),
            b: b.len(),
        });
    }
    let n = a.len();
    let spare = dst.capacity() - dst.len();
    if spare < n {
        return Err(SimdError::OutputLength {
            output: spare,
            expected: n,
        });
    }
    if n == 0 {
        return Ok(());
    }
    let dest = dst.as_ptr().wrapping_add(dst.len());
    if ranges_overlap(a.as_ptr(), n, dest, n) || ranges_overlap(b.as_ptr(), n, dest, n) {
        return Err(SimdError::OverlappingOutput);
    }
    call(a, b, dst);
    Ok(())
}

/// `c[i] = a[i] * b[i]` through Accelerate `vDSP_vmul`, stride 1.
///
/// `a`, `b`, and `c` must be the same length. `c` must not overlap `a` or
/// `b`. An empty length does not call vDSP. Each finite product is one IEEE
/// rounding, including a tail that is not a multiple of a vector width.
/// NaN and infinity propagate; a finite overflow becomes infinity. This
/// function does not scan for them.
///
/// Compiled only with the `accelerate` feature on macOS.
///
/// # Errors
///
/// [`SimdError::MismatchedLengths`], [`SimdError::OutputLength`], or
/// [`SimdError::OverlappingOutput`]. A refusal does not write `c`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vdsp_vmul(a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), SimdError> {
    vdsp_slices(a, b, c, arch::vdsp_vmul)
}

/// `c[i] = a[i] + b[i]` through Accelerate `vDSP_vadd`, stride 1.
///
/// The contract is [`vdsp_vmul`]'s, with addition in place of multiplication.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vdsp_vadd(a: &[f32], b: &[f32], c: &mut [f32]) -> Result<(), SimdError> {
    vdsp_slices(a, b, c, arch::vdsp_vadd)
}

/// Appends `a[i] * b[i]` onto `dst`.
///
/// `dst` must already have room for `a.len()` more elements (`reserve` first).
/// On success those elements are initialized and `dst.len()` grows by
/// `a.len()`. On error `dst` is unchanged. The new elements must not overlap
/// `a` or `b`. An empty input leaves `dst` as it was and does not call vDSP.
///
/// # Errors
///
/// As for [`vdsp_vmul`]. [`SimdError::OutputLength`] reports spare capacity
/// in `output`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vdsp_vmul_append(a: &[f32], b: &[f32], dst: &mut Vec<f32>) -> Result<(), SimdError> {
    vdsp_into_spare(a, b, dst, arch::vdsp_vmul_append)
}

/// Appends `a[i] + b[i]` onto `dst`. See [`vdsp_vmul_append`].
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vdsp_vadd_append(a: &[f32], b: &[f32], dst: &mut Vec<f32>) -> Result<(), SimdError> {
    vdsp_into_spare(a, b, dst, arch::vdsp_vadd_append)
}

/// Element count from the start of row 0 to just past the last copied column.
///
/// A single row ignores `stride` (the pitch is never multiplied). More than
/// one row requires `stride >= cols`, so the copied rows do not overlap.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn mmov_span(rows: usize, cols: usize, stride: usize) -> Result<usize, SimdError> {
    if rows > 1 && stride < cols {
        return Err(SimdError::OverlappingOutputRows {
            n: cols,
            c_rs: stride,
        });
    }
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    let pitch = if rows == 1 { 0 } else { stride };
    (rows - 1)
        .checked_mul(pitch)
        .and_then(|span| span.checked_add(cols))
        .ok_or(SimdError::LengthTooLarge { len: rows })
}

/// Copy `rows` rows of `cols` columns through Accelerate `vDSP_mmov`.
///
/// vecLib passes the column count as `__M` and the row count as `__N`:
///
/// ```text
/// for n in 0..rows {
///     for m in 0..cols {
///         dst[n * dst_stride + m] = src[n * src_stride + m];
///     }
/// }
/// ```
///
/// `src_stride` and `dst_stride` are those pitches, in elements, passed by
/// value. Both are at least `cols` when `rows > 1`. `src` must cover
/// `(rows - 1) * src_stride + cols` elements, and `dst` the same span for
/// `dst_stride`. The two spans must not overlap. An empty `rows` or `cols`
/// does not call vDSP and does not write `dst`.
///
/// The kernel moves bits. −0, NaN payloads, and subnormals are unchanged.
///
/// Compiled only with the `accelerate` feature on macOS.
///
/// # Errors
///
/// [`SimdError::OverlappingOutputRows`] when a pitch is shorter than `cols`.
/// [`SimdError::BufferTooShort`] when a slice is shorter than the span.
/// [`SimdError::OverlappingOutput`] when the spans overlap.
/// [`SimdError::LengthTooLarge`] when the span overflows. A refusal does not
/// write `dst`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vdsp_mmov(
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Result<(), SimdError> {
    let src_span = mmov_span(rows, cols, src_stride)?;
    let dst_span = mmov_span(rows, cols, dst_stride)?;
    if src_span == 0 {
        return Ok(());
    }
    if src.len() < src_span {
        return Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: src_span,
            len: src.len(),
        });
    }
    if dst.len() < dst_span {
        return Err(SimdError::BufferTooShort {
            operand: Operand::C,
            required: dst_span,
            len: dst.len(),
        });
    }
    if ranges_overlap(src.as_ptr(), src_span, dst.as_ptr(), dst_span) {
        return Err(SimdError::OverlappingOutput);
    }
    arch::vdsp_mmov(src, dst.as_mut_ptr(), rows, cols, src_stride, dst_stride);
    Ok(())
}

/// Appends `rows` contiguous rows of `cols` columns onto `dst`.
///
/// The source pitch is `src_stride`. Each destination row is `cols` elements
/// and the rows are packed, so `dst` grows by `rows * cols`. `dst` must
/// already have room for those elements (`reserve` first). On success they
/// are initialized. On error `dst` is unchanged. The new elements must not
/// overlap `src`. An empty `rows` or `cols` leaves `dst` as it was and does
/// not call vDSP.
///
/// This is the same move as [`vdsp_mmov`], with destination pitch `cols`.
///
/// # Errors
///
/// As for [`vdsp_mmov`]. [`SimdError::OutputLength`] reports spare capacity
/// in `output`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vdsp_mmov_append(
    src: &[f32],
    dst: &mut Vec<f32>,
    rows: usize,
    cols: usize,
    src_stride: usize,
) -> Result<(), SimdError> {
    let src_span = mmov_span(rows, cols, src_stride)?;
    if src_span == 0 {
        return Ok(());
    }
    let n = rows
        .checked_mul(cols)
        .ok_or(SimdError::LengthTooLarge { len: rows })?;
    if src.len() < src_span {
        return Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: src_span,
            len: src.len(),
        });
    }
    let spare = dst.capacity() - dst.len();
    if spare < n {
        return Err(SimdError::OutputLength {
            output: spare,
            expected: n,
        });
    }
    let dest = dst.as_ptr().wrapping_add(dst.len());
    if ranges_overlap(src.as_ptr(), src_span, dest, n) {
        return Err(SimdError::OverlappingOutput);
    }
    arch::vdsp_mmov_append(src, dst, rows, cols, src_stride);
    Ok(())
}

/// Append one batch of nanolab head pairs onto `dst`.
///
/// The source is 1024 tokens of 12 heads of 64 floats, packed, so the pitch
/// is 768 and `src` must cover `1024 * 768` floats. Head `h` of token `t`
/// is the 64 floats at `t * 768 + h * 64`. The destination is 12 contiguous
/// blocks of `1024 * 64` floats, head `h` then token `t`.
///
/// Adjacent heads are loaded once per token (128 floats) and stored into
/// the two head blocks. Each head is one 8×8 block transposed in registers
/// and transposed back, so the stored lanes match the load. Tokens are
/// visited in tiles of 32, one pair of heads at a time, so two destination
/// blocks are live. Nothing is zero-filled: the 786_432 floats are written
/// into spare capacity and `dst` grows by that count. Every lane is stored.
/// −0, NaN payloads, and subnormals are unchanged. This does not call vDSP.
///
/// `dst` must already have room for those elements. On error `dst` is
/// unchanged. The new elements must not overlap `src`.
///
/// Compiled only on aarch64 with NEON.
///
/// # Errors
///
/// [`SimdError::BufferTooShort`] when `src` is shorter than one batch.
/// [`SimdError::OutputLength`] when spare capacity is short (`output` is
/// the spare count). [`SimdError::OverlappingOutput`] when the spare range
/// overlaps `src`. A refusal does not write `dst`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn split_nanolab_head_pairs_append(src: &[f32], dst: &mut Vec<f32>) -> Result<(), SimdError> {
    const N: usize = 1024 * 768;
    if src.len() < N {
        return Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: N,
            len: src.len(),
        });
    }
    let spare = dst.capacity() - dst.len();
    if spare < N {
        return Err(SimdError::OutputLength {
            output: spare,
            expected: N,
        });
    }
    let dest = dst.as_ptr().wrapping_add(dst.len());
    if ranges_overlap(src.as_ptr(), N, dest, N) {
        return Err(SimdError::OverlappingOutput);
    }
    arch::split_nanolab_head_pairs_append(src, dst);
    Ok(())
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use arch::NanolabTokenBand;

/// Run `body` on two disjoint time-row bands of one nanolab batch.
///
/// `caller_tokens` is the caller's prefix of the 1024 tokens. The other band
/// is the remaining suffix. `body` receives `(caller, worker)`. Return
/// `Ok(true)` only after both bands have been `write`n, including a write
/// that ran on another thread: this then grows `dst` by the batch. Return
/// `Ok(false)` to leave `dst` unchanged so the caller can take the one-call
/// path. `Err` also leaves `dst` unchanged.
///
/// The worker band must be finished before `body` returns `Ok(true)`. Nothing
/// is zero-filled. −0 is preserved. `dst` must already have room for
/// `1024 * 768` elements.
///
/// # Errors
///
/// [`SimdError::LengthTooLarge`] when `caller_tokens` is `0` or at least 1024.
/// [`SimdError::BufferTooShort`] when `src` is shorter than one batch.
/// [`SimdError::OutputLength`] when spare capacity is short.
/// [`SimdError::OverlappingOutput`] when the spare range overlaps `src`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn with_nanolab_token_bands<E>(
    src: &[f32],
    dst: &mut Vec<f32>,
    caller_tokens: usize,
    body: impl FnOnce(NanolabTokenBand, NanolabTokenBand) -> Result<bool, E>,
) -> Result<Result<bool, E>, SimdError> {
    const TOKENS: usize = 1024;
    const N: usize = TOKENS * 768;
    if caller_tokens == 0 || caller_tokens >= TOKENS {
        return Err(SimdError::LengthTooLarge { len: caller_tokens });
    }
    if src.len() < N {
        return Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: N,
            len: src.len(),
        });
    }
    let spare = dst.capacity() - dst.len();
    if spare < N {
        return Err(SimdError::OutputLength {
            output: spare,
            expected: N,
        });
    }
    let dest = dst.as_ptr().wrapping_add(dst.len());
    if ranges_overlap(src.as_ptr(), N, dest, N) {
        return Err(SimdError::OverlappingOutput);
    }
    let spare_dst = dst.spare_capacity_mut().as_mut_ptr().cast::<f32>();
    let src_ptr = src.as_ptr();
    let caller = NanolabTokenBand::new(src_ptr, spare_dst, 0, caller_tokens);
    let worker = NanolabTokenBand::new(src_ptr, spare_dst, caller_tokens, TOKENS - caller_tokens);
    let outcome = body(caller, worker);
    if matches!(outcome, Ok(true)) {
        // `body` returned `Ok(true)` only after both bands stored their
        // tokens. Those ranges partition the batch, so all `N` spare lanes
        // are initialized.
        arch::commit_nanolab_spare(dst, N);
    }
    Ok(outcome)
}

/// Append gathered embedding rows of `row` floats onto `dst`.
///
/// Row `i` of the output is `table[ids[i] * row ..][..row]`, in id order.
/// `row` must be a positive multiple of 64 and `table.len()` a multiple of
/// `row`. The copy is a move: `ldp` loads each 64-float block of a row and
/// `stnp` stores it into spare capacity, so the kernel has no tail. Nothing
/// is zero-filled. `dst` grows by `ids.len() * row` only after every row
/// is stored. −0, NaN payloads, and subnormals are unchanged. (Until
/// 2026-10-07 this took 768-float rows only.)
///
/// `dst` must already have room for those elements. On error `dst` is
/// unchanged. The new elements must not overlap `table`.
///
/// Compiled only on aarch64 with NEON.
///
/// # Errors
///
/// [`SimdError::RowWidth`] when `row` is not a positive multiple of 64.
/// [`SimdError::BufferTooShort`] when `table` is not a whole number of rows,
/// or an id selects a row past `table`. [`SimdError::LengthTooLarge`] when
/// `ids.len() * row` overflows. [`SimdError::OutputLength`] when spare
/// capacity is short (`output` is the spare count). [`SimdError::OverlappingOutput`]
/// when the spare range overlaps `table`. A refusal does not write `dst`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn gather_embedding_rows(
    table: &[f32],
    row: usize,
    ids: &[u32],
    dst: &mut Vec<f32>,
) -> Result<(), SimdError> {
    if row == 0 || !row.is_multiple_of(64) {
        return Err(SimdError::RowWidth { row });
    }
    if !table.len().is_multiple_of(row) {
        return Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: table.len().div_ceil(row) * row,
            len: table.len(),
        });
    }
    let vocab = table.len() / row;
    let n = match ids.len().checked_mul(row) {
        Some(n) => n,
        None => return Err(SimdError::LengthTooLarge { len: ids.len() }),
    };
    if n == 0 {
        return Ok(());
    }
    for &id in ids {
        if id as usize >= vocab {
            let required = (id as usize).saturating_add(1).saturating_mul(row);
            return Err(SimdError::BufferTooShort {
                operand: Operand::A,
                required,
                len: table.len(),
            });
        }
    }
    let spare = dst.capacity() - dst.len();
    if spare < n {
        return Err(SimdError::OutputLength {
            output: spare,
            expected: n,
        });
    }
    let dest = dst.as_ptr().wrapping_add(dst.len());
    if ranges_overlap(table.as_ptr(), table.len(), dest, n) {
        return Err(SimdError::OverlappingOutput);
    }
    arch::gather_embedding_rows(table, row, ids, dst);
    Ok(())
}

/// `y[i] = exp(x[i])` through Accelerate vForce `vvexpf`.
///
/// The macOS SDK declares
/// `void vvexpf(float *y, const float *x, const int *n)`:
/// `y[i]` is set to `exp(x[i])`, and `n` points at the element count.
///
/// `y` and `x` must be the same length, and that length must fit in a C
/// `int`. `y` must not overlap `x`; the same buffer is [`vvexpf_inplace`].
/// An empty length does not call vForce. vForce may flush denormal inputs
/// and its exact finite results can differ across OS versions. NaN and
/// infinity follow the usual `exp` closure: a NaN stays a NaN, `+inf` stays
/// `+inf`, and `-inf` becomes `+0`. This function does not scan for them.
///
/// Compiled only with the `accelerate` feature on macOS.
///
/// # Errors
///
/// [`SimdError::OutputLength`], [`SimdError::OverlappingOutput`], or
/// [`SimdError::LengthTooLarge`]. A refusal does not write `y`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vvexpf(y: &mut [f32], x: &[f32]) -> Result<(), SimdError> {
    if y.len() != x.len() {
        return Err(SimdError::OutputLength {
            output: y.len(),
            expected: x.len(),
        });
    }
    vvexpf_count(y.len())?;
    if ranges_overlap(x.as_ptr(), x.len(), y.as_ptr(), y.len()) {
        return Err(SimdError::OverlappingOutput);
    }
    if y.is_empty() {
        return Ok(());
    }
    arch::vvexpf(y, x);
    Ok(())
}

/// In-place [`vvexpf`]: `y[i] = exp(y[i])`.
///
/// An empty length does not call vForce. A length that does not fit in a C
/// `int` is [`SimdError::LengthTooLarge`] and does not write `y`.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub fn vvexpf_inplace(y: &mut [f32]) -> Result<(), SimdError> {
    vvexpf_count(y.len())?;
    if y.is_empty() {
        return Ok(());
    }
    arch::vvexpf_inplace(y);
    Ok(())
}

/// One pass: load, test finite, record `z[i] < 0`, store `z[i] = -|z[i]|`.
///
/// Each 16-wide chunk (then each remaining group of 4, then each tail
/// lane) is checked before any lane of it is stored, on every backend; on
/// NEON it is loaded once. A lane that is not finite stops the loop before
/// that chunk is stored. Earlier chunks may already hold `-|x|`. Their
/// sign bytes are not published: `signs` stays at length 0. A finite chunk
/// stores a sign byte and `-|x|` before the next load. A sign byte is `1`
/// when the loaded value is `< 0` and `0` otherwise: `-0` compares equal
/// to `+0`, so it is not recorded as negative. `-|±0|` is stored as `-0`.
///
/// # Errors
///
/// [`SimdError::ReserveFailed`] when `signs` cannot grow to `z.len()`.
/// Nothing is stored. [`SimdError::NonFinite`] when a lane is NaN or an
/// infinity. `signs` has length 0.
pub fn store_neg_abs_signs(z: &mut [f32], signs: &mut Vec<u8>) -> Result<(), SimdError> {
    let n = z.len();
    if signs.capacity() < n {
        let mut fresh = Vec::new();
        if fresh.try_reserve_exact(n).is_err() {
            return Err(SimdError::ReserveFailed { len: n });
        }
        *signs = fresh;
    } else {
        signs.clear();
    }
    if !arch::store_neg_abs_signs(z, signs) {
        debug_assert_eq!(signs.len(), 0);
        return Err(SimdError::NonFinite);
    }
    Ok(())
}

pub use arch::ReservedF32;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
pub use arch::NegAbsExp;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn vvexpf_count(n: usize) -> Result<(), SimdError> {
    if n > core::ffi::c_int::MAX as usize {
        Err(SimdError::LengthTooLarge { len: n })
    } else {
        Ok(())
    }
}

/// Appends `gates[h] * attn` of that head onto `dst`.
///
/// Head `h` is `attn[h * head_dim .. (h + 1) * head_dim]`, scaled by
/// `gates[h]`. Each product is one rounding of `a * g`, written once into
/// `dst`'s spare capacity. `dst` must already have room for `attn.len()`
/// more elements. On success `dst.len()` grows by that count. On error
/// `dst` is unchanged. An empty `attn` leaves `dst` as it was.
///
/// This is not `vDSP_vsmul`. One loop writes every head. Each product is
/// one rounding of `a * g`, including `−0`. The release build stores it
/// with `stp`/`str`.
///
/// # Errors
///
/// [`SimdError::MismatchedLengths`] when `attn.len()` is not
/// `gates.len() * head_dim` (a `head_dim` of 0 requires an empty `attn`).
/// [`SimdError::LengthTooLarge`] when that product overflows.
/// [`SimdError::OutputLength`] when spare capacity is short (`output` is
/// the spare count). [`SimdError::OverlappingOutput`] when the spare range
/// overlaps `attn` or `gates`. A refusal does not write `dst`.
pub fn scale_heads_append(
    attn: &[f32],
    gates: &[f32],
    head_dim: usize,
    dst: &mut Vec<f32>,
) -> Result<(), SimdError> {
    let n = if head_dim == 0 {
        if attn.is_empty() {
            return Ok(());
        }
        return Err(SimdError::MismatchedLengths {
            a: attn.len(),
            b: 0,
        });
    } else {
        match gates.len().checked_mul(head_dim) {
            Some(n) if n == attn.len() => n,
            Some(n) => {
                return Err(SimdError::MismatchedLengths {
                    a: attn.len(),
                    b: n,
                });
            }
            None => return Err(SimdError::LengthTooLarge { len: head_dim }),
        }
    };
    let spare = dst.capacity() - dst.len();
    if spare < n {
        return Err(SimdError::OutputLength {
            output: spare,
            expected: n,
        });
    }
    if n == 0 {
        return Ok(());
    }
    let dest = dst.as_ptr().wrapping_add(dst.len());
    if ranges_overlap(attn.as_ptr(), n, dest, n)
        || ranges_overlap(gates.as_ptr(), gates.len(), dest, n)
    {
        return Err(SimdError::OverlappingOutput);
    }
    arch::scale_heads_append(attn, gates, head_dim, dst);
    Ok(())
}
