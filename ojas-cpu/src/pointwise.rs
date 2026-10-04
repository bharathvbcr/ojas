//! Embedding, pointwise ops, per-head gate, value residual, and cross-entropy.
//!
//! SiLU, the value-residual blend, the gate under
//! [`Numerics::Exact`], and the gate backward run in contiguous chunks on
//! scoped threads, each chunk written in place into a zero-filled output
//! ([`scoped::chunks_into_n`]). Mul forward and mul backward reserve without
//! that zero-fill and adopt after every lane is stored. Mul tests each stored
//! product in that loop and does not reread the chunk. The blend's forward
//! pass still scans each chunk on the thread that wrote it
//! ([`crate::validate::fill_outs_chunked`]).
//! Add forward does not
//! spawn: it appends each sum once into a reserved output on the calling
//! thread ([`extend_sums`]; on macOS Fast add is one `vDSP_vadd` into that
//! buffer). Fast gate forward's broadcast does not spawn either: each head
//! is scaled by that head's sigmoid into a reserved output
//! ([`scale_heads`]).
//! No output value depends on which chunk computed it,
//! so the bits do not depend on the thread count. Under [`Numerics::Exact`]
//! every value is the scalar formula below, evaluated as written (libm
//! `exp`, no `mul_add`). Under [`Numerics::Fast`] SiLU backward and
//! cross-entropy use the branch-free [`crate::exp`] so their loops vectorize.
//! Fast SiLU forward uses that same exponential off macOS; on macOS it uses
//! Accelerate `vvexpf` for `e^{-|x|}` and then the same sigmoid pair. The
//! gate's logit sigmoid does the same under Fast on macOS. The
//! value-residual `lambda`
//! gradient sums `f64` terms over fixed blocks, and the gate's per-head dot
//! uses [`dot_lanes`].
//!
//! The gate logits `input · Wᵀ + b` and the gate's input and weight
//! gradients are products on the GEMM core ([`crate::gemm`]). Fast backward
//! uses the scale the forward kept, when the caller passes it, and does not
//! run that logit product again.

use std::borrow::Cow;
use std::mem::MaybeUninit;

use ojas_core::{
    BackendId, Budget, CeDims, EmbeddingDims, GateDims, Numerics, OjasError, Reservation, Scratch,
    Tensor,
};

use crate::exp::{exp, exp_sub_store, exp_sub_sum};
use crate::gemm::{fma, gemm, gemm_out, scratch as gemm_scratch, Mat};
use crate::pool::scoped;
use crate::pool::{Exec, ROW_MIN_ELEMS};
use crate::validate::{
    all_finite, check_f32, f32_values, fill_rows, nonfinite, nonfinite_first, payload_bytes,
    product, room_for, shape, trusted_finite_f32, u32_values,
};

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// The rows of `table` (`[vocab, dim]`, read in place) that `ids` pick,
/// copied value for value into one charged tensor shaped `dims.out_shape`.
pub(crate) fn embedding_forward(
    op: &'static str,
    budget: &Budget,
    table: &[f32],
    ids: &[u32],
    dims: &EmbeddingDims,
) -> Result<Tensor, OjasError> {
    check_ids(op, ids, dims.vocab)?;
    let row = dims.dim;
    let mut out = Scratch::<f32>::try_alloc(product(op, &[dims.tokens, row])?, budget)?;
    for (dst, id) in out.as_mut_slice().chunks_exact_mut(row).zip(ids) {
        let start = id_index(*id) * row;
        let src = table
            .get(start..start + row)
            .ok_or_else(|| outside(op, "embedding row exceeds the table"))?;
        dst.copy_from_slice(src);
    }
    Tensor::from_scratch(out, &dims.out_shape)
}

/// The table gradient: each row of `grad` (`ids.shape ++ [dim]`, read in
/// place) is added into row `ids[n]` of one zeroed, charged `[vocab, dim]`
/// tensor, in token order. A table row is the f32 sum from 0 of its tokens'
/// rows in ascending token order, the order Metal's scatter also uses.
///
/// A sum that overflows is [`OjasError::NonFinite`].
pub(crate) fn embedding_backward(
    op: &'static str,
    budget: &Budget,
    ids: &[u32],
    grad: &[f32],
    dims: &EmbeddingDims,
) -> Result<Tensor, OjasError> {
    check_ids(op, ids, dims.vocab)?;
    let dim = dims.dim;
    let mut out = Scratch::<f32>::try_alloc(product(op, &[dims.vocab, dim])?, budget)?;
    let table = out.as_mut_slice();
    let mut finite = true;
    for (src, id) in grad.chunks_exact(dim).zip(ids) {
        let start = id_index(*id) * dim;
        let dst = table
            .get_mut(start..start + dim)
            .ok_or_else(|| outside(op, "embedding row exceeds the table"))?;
        for (d, s) in dst.iter_mut().zip(src) {
            let sum = *d + *s;
            finite &= sum.is_finite();
            *d = sum;
        }
    }
    if !finite {
        return Err(nonfinite(op));
    }
    Tensor::from_scratch(out, &dims.out_shape)
}

fn id_index(id: u32) -> usize {
    id as usize
}

fn outside(op: &'static str, detail: &str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: detail.to_string(),
    }
}

/// Every id below `vocab`, read in place.
fn check_ids(op: &'static str, ids: &[u32], vocab: usize) -> Result<(), OjasError> {
    for (n, &id) in ids.iter().enumerate() {
        if id_index(id) >= vocab {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("token id {id} at {n} is outside vocab {vocab}"),
            });
        }
    }
    Ok(())
}

/// `(sigmoid(x), sigmoid(-x))` under [`Numerics::Fast`]. Both come from
/// `e = e^-|x|`, so `1 - sigmoid(x)` is not formed by cancellation.
#[inline(always)]
fn sigmoid_pair_fast(x: f32) -> (f32, f32) {
    let e = exp(-x.abs());
    let big = 1.0 / (1.0 + e);
    let small = e * big;
    if x >= 0.0 {
        (big, small)
    } else {
        (small, big)
    }
}

fn same_len(op: &'static str, lens: &[usize], what: &str) -> Result<(), OjasError> {
    if lens.windows(2).all(|w| w[0] == w[1]) {
        Ok(())
    } else {
        Err(shape(op, format!("{what} lengths differ: {lens:?}")))
    }
}

/// `x * sigmoid(x)` as a tensor of `shape`.
///
/// Exact is libm [`sigmoid`] through [`fill_rows`], which zero-fills the
/// output. Fast off macOS is the same fill with [`crate::exp`]. Fast on
/// macOS does not zero-fill: the output is reserved in `ojas-simd`, every
/// lane is stored with `-|x|` before it is read, `vvexpf` overwrites those
/// lanes, then the [`sigmoid_pair_fast`] pair runs on that memory. The
/// vector is moved into the scratch under the same `len * 4` charge. A
/// refusal drops it and releases the charge. Exact is unchanged.
pub(crate) fn silu_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    #[cfg(target_os = "macos")]
    if exec.numerics == Numerics::Fast {
        return silu_forward_fast(op, budget, exec, x, shape);
    }
    silu_forward_rows(op, budget, exec, x, shape)
}

/// [`fill_rows`] SiLU. Exact, and Fast off macOS. The output is zero-filled
/// and then overwritten.
fn silu_forward_rows(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let numerics = exec.numerics;
    fill_rows(op, budget, exec, shape, x.len(), 1, |range, dst| {
        let src = &x[range];
        match numerics {
            Numerics::Exact => {
                for (out, &v) in dst.iter_mut().zip(src) {
                    *out = v * sigmoid(v);
                }
            }
            Numerics::Fast => silu_fast(op, src, dst)?,
        }
        Ok(())
    })
}

/// Fast macOS SiLU. The scratch charge matches [`Scratch::try_alloc`]:
/// `n * 4` bytes, taken before the allocation and released on every refusal.
#[cfg(target_os = "macos")]
fn silu_forward_fast(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let n = x.len();
    if product(op, shape)? != product(op, &[n, 1])? {
        return Err(crate::validate::shape(
            op,
            format!("output {shape:?} does not hold {n} rows of 1"),
        ));
    }
    let bytes = (n as u64)
        .checked_mul(4)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: format!("{n} * 4 overflows u64"),
        })?;
    let reservation = budget.try_reserve(bytes)?;
    let min_chunk = scoped::min_rows(1);
    let pieces = if n == 0 {
        0
    } else {
        (n / min_chunk.max(1)).clamp(1, exec.pool.threads().saturating_mul(2))
    };
    let parts = if n == 0 {
        Vec::new()
    } else {
        crate::pool::ranges(n, pieces)
    };
    let chunks: Vec<(usize, usize)> = parts.iter().map(|r| (r.start, r.end - r.start)).collect();
    let buf = match ojas_simd::NegAbsExp::try_new(n, &chunks) {
        Ok(buf) => buf,
        Err(ojas_simd::SimdError::ReserveFailed { .. }) => {
            drop(reservation);
            return Err(OjasError::CapacityExceeded {
                requested: bytes,
                cap: budget.cap_bytes(),
                live: budget.live_bytes()?,
            });
        }
        Err(err) => {
            drop(reservation);
            return Err(vdsp_refused(op, err));
        }
    };
    let flags = match scoped::map(exec, chunks.len(), |i| {
        let (start, len) = chunks[i];
        let src = &x[start..start + len];
        let finite = buf
            .write_chunk(i, src, |dst| {
                for (d, &v) in dst.iter_mut().zip(src) {
                    let e = *d;
                    let big = 1.0 / (1.0 + e);
                    let small = e * big;
                    let s = if v >= 0.0 { big } else { small };
                    *d = v * s;
                }
                all_finite(dst)
            })
            .map_err(|err| vdsp_refused(op, err))?;
        Ok(finite)
    }) {
        Ok(flags) => flags,
        Err(err) => {
            drop(buf);
            drop(reservation);
            return Err(err);
        }
    };
    if flags.iter().any(|finite| !finite) {
        drop(buf);
        drop(reservation);
        return Err(nonfinite(op));
    }
    let vec = match buf.into_vec() {
        Ok(vec) => vec,
        Err(err) => {
            drop(reservation);
            return Err(vdsp_refused(op, err));
        }
    };
    let scratch = Scratch::<f32>::try_adopt(vec, reservation)?;
    trusted_finite_f32(op, scratch, shape)
}

/// Fast `x * sigmoid(x)` into `dst`, the same length as `src`.
fn silu_fast(op: &'static str, src: &[f32], dst: &mut [f32]) -> Result<(), OjasError> {
    #[cfg(target_os = "macos")]
    {
        silu_fast_vvexpf(op, src, dst)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = op;
        silu_fast_poly(src, dst);
        Ok(())
    }
}

/// [`sigmoid_pair_fast`] evaluated in place. Fast SiLU forward off macOS.
#[cfg(not(target_os = "macos"))]
fn silu_fast_poly(src: &[f32], dst: &mut [f32]) {
    for (out, &v) in dst.iter_mut().zip(src) {
        *out = v * sigmoid_pair_fast(v).0;
    }
}

/// `e^{-|x|}` via in-place `vvexpf`, then the [`sigmoid_pair_fast`] pair.
/// One call per chunk: `vvexpf` is elementwise, so a different chunk length
/// does not change an element's bits.
#[cfg(target_os = "macos")]
fn silu_fast_vvexpf(op: &'static str, src: &[f32], dst: &mut [f32]) -> Result<(), OjasError> {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &v) in dst.iter_mut().zip(src) {
        *d = -v.abs();
    }
    // The count argument is a C `int`. A pool chunk is far smaller; a
    // single-threaded tensor can be larger, so split there.
    const MAX: usize = i32::MAX as usize;
    for chunk in dst.chunks_mut(MAX) {
        ojas_simd::vvexpf_inplace(chunk).map_err(|err| vdsp_refused(op, err))?;
    }
    for (d, &v) in dst.iter_mut().zip(src) {
        let e = *d;
        let big = 1.0 / (1.0 + e);
        let small = e * big;
        let s = if v >= 0.0 { big } else { small };
        *d = v * s;
    }
    Ok(())
}

/// `grad_y * s * (1 + x (1 - s))` with `s = sigmoid(x)` as a tensor of
/// `shape` ([`fill_rows`]).
pub(crate) fn silu_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    grad_y: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    same_len(op, &[x.len(), grad_y.len()], "silu input and grad")?;
    let numerics = exec.numerics;
    fill_rows(op, budget, exec, shape, x.len(), 1, |range, dst| {
        let pairs = x[range.clone()].iter().zip(&grad_y[range]);
        match numerics {
            Numerics::Exact => {
                for (out, (&v, &g)) in dst.iter_mut().zip(pairs) {
                    let s = sigmoid(v);
                    *out = g * s * (1.0 + v * (1.0 - s));
                }
            }
            Numerics::Fast => {
                for (out, (&v, &g)) in dst.iter_mut().zip(pairs) {
                    let (s, rest) = sigmoid_pair_fast(v);
                    *out = g * s * fma(v, rest, 1.0);
                }
            }
        }
        Ok(())
    })
}

/// `a * b` as a tensor of `shape`, one rounding per value under either
/// contract.
///
/// The operands are read where they are and each value is written once.
/// The output is reserved without a zero-fill ([`ojas_simd::ReservedF32`]):
/// [`mul_store`] writes every lane, including a one-element tail, and the
/// finite test is that stored product. Chunk flags AND-reduce. The vector
/// is adopted under the `len * 4` charge only when every flag is true. A
/// refusal drops the buffer and releases the charge. A failed charge does
/// not allocate the buffer.
///
/// The pass is split into chunks of [`ROW_MIN_ELEMS`] values on scoped
/// threads, the same cut as [`fill_outs_chunked_trusted`] with width 1
/// (`(n / ROW_MIN_ELEMS).clamp(1, threads * 2)`). One chunk, or one thread,
/// stays on the calling thread. Fewer chunks lost on backward, so backward
/// keeps that count. Forward does too, except at 6 threads for a length
/// from `1024 * 1024` through `1024 * 2048` inclusive, which uses 4 pieces.
/// Release, same process, alternating, min of 24 after 2 warmups: 4 pieces
/// beat the 12-piece cut at both ends of that band and lost at double the
/// upper end, which stays on the live cut. Each value is the same single
/// rounding wherever it is computed, so the bits do not depend on the split.
/// A serial one-write was
/// slower for this shape at 6 threads, and so was `vDSP_vmul` on those
/// chunks and a serial `vDSP_vmul` into spare capacity (interleaved,
/// `[1024, 2048]`, the zip at about 0.20–0.22 ms and vDSP level with it).
/// Folding the finite test into the zip, same shape and 6 threads, min of 20
/// after 2 warmups interleaved against the reread, was 0.183 ms forward
/// against 0.216 ms and 0.307 ms backward against 0.346 ms. Mul stays on the
/// parallel zip under both contracts. Backward adopts both gradients the
/// same way and keeps the wider chunk count.
pub(crate) fn mul_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let n = a.len();
    same_len(op, &[n, b.len(), product(op, shape)?], "mul operand")?;
    let bytes = (n as u64)
        .checked_mul(4)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: format!("{n} * 4 overflows u64"),
        })?;
    let reservation = budget.try_reserve(bytes)?;
    // Width-1 cut of `chunks_into_n` / `fill_outs_chunked_trusted`, except
    // 6 threads on [1024*1024, 1024*2048]. Forward-only, release, same
    // process, alternating, min of 24 after 2 warmups against the live cut
    // (12 pieces at these lengths): 4 pieces was 0.0655 ms vs 0.0828 ms at
    // the low end and 0.1227 ms vs 0.1383 ms at 1024*2048. At double
    // (1024*4096) the live cut was faster, 0.2552 ms vs 0.2943 ms, so the
    // band stops at the upper end.
    let min_chunk = scoped::min_rows(1);
    let wide = exec.pool.threads().saturating_mul(2);
    let live = (n / min_chunk.max(1)).clamp(1, wide);
    let pieces = if exec.pool.threads() == 6 && ((1024 * 1024)..=(1024 * 2048)).contains(&n) {
        4
    } else {
        live
    };
    let chunks: Vec<(usize, usize)> = crate::pool::ranges(n, pieces)
        .into_iter()
        .map(|r| (r.start, r.end - r.start))
        .collect();
    let buf = match ojas_simd::ReservedF32::try_new(n, &chunks) {
        Ok(buf) => buf,
        Err(ojas_simd::SimdError::ReserveFailed { .. }) => {
            drop(reservation);
            return Err(OjasError::CapacityExceeded {
                requested: bytes,
                cap: budget.cap_bytes(),
                live: budget.live_bytes()?,
            });
        }
        Err(err) => {
            drop(reservation);
            return Err(reserved_refused(op, err));
        }
    };
    let flags = match scoped::map(exec, chunks.len(), |i| {
        let (start, len) = chunks[i];
        let end = start + len;
        buf.write_chunk(i, |dst| mul_store(dst, &a[start..end], &b[start..end]))
            .map_err(|err| reserved_refused(op, err))
    }) {
        Ok(flags) => flags,
        Err(err) => {
            drop(buf);
            drop(reservation);
            return Err(err);
        }
    };
    if flags.iter().any(|finite| !finite) {
        drop(buf);
        drop(reservation);
        return Err(nonfinite(op));
    }
    let vec = match buf.into_vec() {
        Ok(vec) => vec,
        Err(err) => {
            drop(reservation);
            return Err(reserved_refused(op, err));
        }
    };
    let scratch = Scratch::<f32>::try_adopt(vec, reservation)?;
    trusted_finite_f32(op, scratch, shape)
}

fn reserved_refused(op: &'static str, err: ojas_simd::SimdError) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail: format!("{op}: {err}"),
    }
}

/// `out[i] = a[i] * b[i]`. True only when every stored product is finite.
///
/// Every lane is stored, including a one-element tail. The finite test is
/// `p.to_bits() & MAGNITUDE` on the product register that is stored, not a
/// reload of `out`.
fn mul_store(out: &mut [MaybeUninit<f32>], a: &[f32], b: &[f32]) -> bool {
    debug_assert_eq!(out.len(), a.len());
    debug_assert_eq!(out.len(), b.len());
    let mut top = 0u32;
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        let p = x * y;
        o.write(p);
        top = top.max(p.to_bits() & MAGNITUDE);
    }
    top < NON_FINITE
}

/// `(grad_y * b, grad_y * a)` as tensors of `a_shape` and `b_shape` (see
/// [`mul_forward`]), both filled in one split and one pass over `grad_y`.
///
/// Both outputs are reserved without a zero-fill ([`ojas_simd::ReservedF32`]),
/// the same adopt path as [`mul_forward`]. Charges are taken before either
/// buffer is allocated. [`mul_grad_lanes`] stores every lane of both,
/// including a one-element tail and an empty chunk, and the finite test is
/// those stored products. Neither vector is adopted unless every chunk flag
/// is true: a non-finite product drops both buffers and releases both
/// charges, and nothing uninitialized is read. The split stays
/// `(n / ROW_MIN_ELEMS).clamp(1, threads * 2)`, including where forward
/// uses fewer pieces.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mul_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    a: &[f32],
    b: &[f32],
    grad_y: &[f32],
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(Tensor, Tensor), OjasError> {
    let n = grad_y.len();
    same_len(
        op,
        &[
            a.len(),
            b.len(),
            n,
            product(op, a_shape)?,
            product(op, b_shape)?,
        ],
        "mul grad",
    )?;
    let bytes = (n as u64)
        .checked_mul(4)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: format!("{n} * 4 overflows u64"),
        })?;
    let reservation_a = budget.try_reserve(bytes)?;
    let reservation_b = match budget.try_reserve(bytes) {
        Ok(reservation) => reservation,
        Err(err) => {
            drop(reservation_a);
            return Err(err);
        }
    };
    // Width-1 cut of `chunks_into_n`. Forward uses 4 pieces on a length
    // band at 6 threads; this does not.
    let min_chunk = scoped::min_rows(1);
    let pieces = (n / min_chunk.max(1)).clamp(1, exec.pool.threads().saturating_mul(2));
    let chunks: Vec<(usize, usize)> = crate::pool::ranges(n, pieces)
        .into_iter()
        .map(|r| (r.start, r.end - r.start))
        .collect();
    let buf_a = match ojas_simd::ReservedF32::try_new(n, &chunks) {
        Ok(buf) => buf,
        Err(err) => {
            drop(reservation_a);
            drop(reservation_b);
            return Err(reserved_alloc_err(op, budget, bytes, err)?);
        }
    };
    let buf_b = match ojas_simd::ReservedF32::try_new(n, &chunks) {
        Ok(buf) => buf,
        Err(err) => {
            drop(buf_a);
            drop(reservation_a);
            drop(reservation_b);
            return Err(reserved_alloc_err(op, budget, bytes, err)?);
        }
    };
    let flags = match scoped::map(exec, chunks.len(), |i| {
        let (start, len) = chunks[i];
        let end = start + len;
        mul_grad_chunk(
            &buf_a,
            &buf_b,
            i,
            &a[start..end],
            &b[start..end],
            &grad_y[start..end],
        )
        .map_err(|err| reserved_refused(op, err))
    }) {
        Ok(flags) => flags,
        Err(err) => {
            drop(buf_a);
            drop(buf_b);
            drop(reservation_a);
            drop(reservation_b);
            return Err(err);
        }
    };
    if flags.iter().any(|finite| !finite) {
        drop(buf_a);
        drop(buf_b);
        drop(reservation_a);
        drop(reservation_b);
        return Err(nonfinite(op));
    }
    let vec_a = match buf_a.into_vec() {
        Ok(vec) => vec,
        Err(err) => {
            drop(buf_b);
            drop(reservation_a);
            drop(reservation_b);
            return Err(reserved_refused(op, err));
        }
    };
    let vec_b = match buf_b.into_vec() {
        Ok(vec) => vec,
        Err(err) => {
            drop(vec_a);
            drop(reservation_a);
            drop(reservation_b);
            return Err(reserved_refused(op, err));
        }
    };
    let scratch_a = Scratch::<f32>::try_adopt(vec_a, reservation_a)?;
    let scratch_b = Scratch::<f32>::try_adopt(vec_b, reservation_b)?;
    let grad_a = trusted_finite_f32(op, scratch_a, a_shape)?;
    let grad_b = trusted_finite_f32(op, scratch_b, b_shape)?;
    Ok((grad_a, grad_b))
}

/// Store one chunk of both gradients. Every lane of both slices is written
/// before this returns.
fn mul_grad_chunk(
    buf_a: &ojas_simd::ReservedF32,
    buf_b: &ojas_simd::ReservedF32,
    index: usize,
    a: &[f32],
    b: &[f32],
    g: &[f32],
) -> Result<bool, ojas_simd::SimdError> {
    buf_a.write_chunk(index, |ga| {
        buf_b.write_chunk(index, |gb| mul_grad_lanes(ga, gb, a, b, g))
    })?
}

/// A simd reserve that failed after the charges were released, or any other
/// `ReservedF32` refusal.
fn reserved_alloc_err(
    op: &'static str,
    budget: &Budget,
    bytes: u64,
    err: ojas_simd::SimdError,
) -> Result<OjasError, OjasError> {
    match err {
        ojas_simd::SimdError::ReserveFailed { .. } => Ok(OjasError::CapacityExceeded {
            requested: bytes,
            cap: budget.cap_bytes(),
            live: budget.live_bytes()?,
        }),
        other => Ok(reserved_refused(op, other)),
    }
}

/// `ga[i] = g[i] * b[i]` and `gb[i] = g[i] * a[i]`, one read of `g`.
///
/// True only when every stored product is finite. Every lane of both outputs
/// is stored, including a one-element tail. The finite test is
/// `to_bits() & MAGNITUDE` on the product registers that are stored, not a
/// reload of either output.
fn mul_grad_lanes(
    ga: &mut [MaybeUninit<f32>],
    gb: &mut [MaybeUninit<f32>],
    a: &[f32],
    b: &[f32],
    g: &[f32],
) -> bool {
    debug_assert_eq!(ga.len(), gb.len());
    debug_assert_eq!(ga.len(), a.len());
    debug_assert_eq!(ga.len(), b.len());
    debug_assert_eq!(ga.len(), g.len());
    let mut top = 0u32;
    for i in 0..ga.len() {
        let gv = g[i];
        let pa = gv * b[i];
        let pb = gv * a[i];
        ga[i].write(pa);
        gb[i].write(pb);
        top = top
            .max(pa.to_bits() & MAGNITUDE)
            .max(pb.to_bits() & MAGNITUDE);
    }
    top < NON_FINITE
}

/// Floats per stack tile of [`extend_sums`].
///
/// 4096 is 16 KiB. On `[1024, 768]` at 6 threads it was faster than 1024
/// and faster than the zero-filled parallel zip. Each tile is one `a + b`
/// per element, scanned, then appended, so the output element is written
/// once.
const ADD_FWD_TILE: usize = 4096;

/// `a + b` appended in [`ADD_FWD_TILE`] chunks, including a short tail.
///
/// Returns whether every sum is finite. `a` and `b` are the same length.
/// Each sum is one rounding of `a + b`.
fn append_sums(a: &[f32], b: &[f32], dst: &mut Vec<f32>) -> bool {
    let mut tile = [0f32; ADD_FWD_TILE];
    let mut finite = true;
    let mut ac = a.chunks_exact(ADD_FWD_TILE);
    let mut bc = b.chunks_exact(ADD_FWD_TILE);
    for (ca, cb) in ac.by_ref().zip(bc.by_ref()) {
        for k in 0..ADD_FWD_TILE {
            tile[k] = ca[k] + cb[k];
        }
        finite &= all_finite(&tile);
        dst.extend_from_slice(&tile);
    }
    let ra = ac.remainder();
    let rb = bc.remainder();
    let n = ra.len();
    debug_assert_eq!(n, rb.len());
    if n > 0 {
        for k in 0..n {
            tile[k] = ra[k] + rb[k];
        }
        finite &= all_finite(&tile[..n]);
        dst.extend_from_slice(&tile[..n]);
    }
    finite
}

/// One output of `a + b`, charged and reserved, never zero-filled.
///
/// A non-finite sum drops the scratch (the charge comes off) and is
/// [`OjasError::NonFinite`] before a tensor is returned. A finite output is
/// recorded finite, so the next op does not scan it again.
fn extend_sums(
    op: &'static str,
    budget: &Budget,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let n = a.len();
    let mut finite = true;
    let scratch = Scratch::<f32>::try_extend(n, budget, |dst| {
        finite = append_sums(a, b, dst);
    })?;
    if !finite {
        drop(scratch);
        return Err(nonfinite(op));
    }
    let out = Tensor::from_scratch(scratch, shape)?;
    if !out.all_finite_cached(|_| Ok(true))? {
        return Err(nonfinite(op));
    }
    Ok(out)
}

/// `a + b` as a tensor of `shape`, one rounding per value.
///
/// Exact, and Fast off macOS, append on the calling thread ([`extend_sums`]).
/// The pool is not used: at `[1024, 768]` on 6 threads the zero-fill and the
/// scoped spawn cost more than the extra arithmetic of six threads. On macOS
/// Fast is one `vDSP_vadd` into that same reserved buffer. Interleaved at
/// `[1024, 768]` on 6 threads that was 0.080 ms against 0.117 ms for the
/// scalar tiles. Writing vDSP into the stack tile and then appending was
/// slower than the scalar tiles (0.120 ms). Scanning each vDSP chunk before
/// the next, at 2,097,152 floats, did not beat one `vDSP_vadd` plus one scan
/// (release, 6 threads, min of 20 after 2 warmups): warmed whole forward
/// 0.206 ms, scan 0.081 ms. Chunks of 16384, 65536, 262144, and 1048576 were
/// not faster on both the minimum and the median. The bits do not depend on
/// the thread count.
pub(crate) fn add_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    same_len(
        op,
        &[a.len(), b.len(), product(op, shape)?],
        "residual add operand",
    )?;
    match exec.numerics {
        Numerics::Exact => extend_sums(op, budget, a, b, shape),
        Numerics::Fast => add_fast(op, budget, a, b, shape),
    }
}

fn add_fast(
    op: &'static str,
    budget: &Budget,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    #[cfg(target_os = "macos")]
    {
        add_vdsp(op, budget, a, b, shape)
    }
    #[cfg(not(target_os = "macos"))]
    {
        extend_sums(op, budget, a, b, shape)
    }
}

#[cfg(target_os = "macos")]
fn add_vdsp(
    op: &'static str,
    budget: &Budget,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    finish_extend(op, budget, a.len(), shape, |dst| {
        ojas_simd::vdsp_vadd_append(a, b, dst).map_err(|err| vdsp_refused(op, err))?;
        Ok(all_finite(dst))
    })
}

#[cfg(target_os = "macos")]
fn vdsp_refused(op: &'static str, err: ojas_simd::SimdError) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail: format!("{op}: {err}"),
    }
}

/// Charge `n` elements and let `write` append them. `Ok(false)` is a
/// non-finite result: the buffer is dropped and the charge released.
fn finish_extend(
    op: &'static str,
    budget: &Budget,
    n: usize,
    shape: &[usize],
    write: impl FnOnce(&mut Vec<f32>) -> Result<bool, OjasError>,
) -> Result<Tensor, OjasError> {
    let mut outcome = None;
    let scratch = Scratch::<f32>::try_extend(n, budget, |dst| {
        outcome = Some(write(dst));
    });
    match outcome {
        Some(Err(err)) => return Err(err),
        Some(Ok(false)) => {
            drop(scratch);
            return Err(nonfinite(op));
        }
        Some(Ok(true)) => {}
        None => {
            return match scratch {
                Err(err) => Err(err),
                Ok(_) => Err(OjasError::Backend {
                    id: BackendId::Cpu,
                    detail: format!("{op}: output was not written"),
                }),
            };
        }
    }
    let scratch = scratch?;
    let out = Tensor::from_scratch(scratch, shape)?;
    if !out.all_finite_cached(|_| Ok(true))? {
        return Err(nonfinite(op));
    }
    Ok(out)
}

/// Both gradients of `x + y` are `grad_y` itself. The caller has run
/// [`ojas_core::residual_add_backward_dims`]; every operand is then checked
/// in place (layout, a NaN or infinity) before anything is charged; `x` and
/// `y` are read for nothing else and are not copied. The gradient is read
/// once, in [`ADD_BWD_TILE`] chunks, and that chunk is appended into both
/// reserved outputs ([`Scratch::try_extend_pair`]): each is its own
/// allocation, charged `n * 4`. The outputs are not zero-filled, and the
/// gradient is not read again. The bits are `grad_y`'s bits, including `-0`.
/// A gradient a
/// caller rewrites in place (a clip) must not share storage with `grad_y`
/// or with the other gradient. Sharing would also hand a trainer a gradient
/// still aliased by the other input's gradient. A failed second reserve
/// releases the first charge.
pub(crate) fn add_backward(
    op: &'static str,
    budget: &Budget,
    x: &Tensor,
    y: &Tensor,
    grad_y: &Tensor,
) -> Result<(Tensor, Tensor), OjasError> {
    for t in [x, y, grad_y] {
        check_f32(op, t)?;
    }
    let src = grad_y.f32_slice()?;
    let (grad_x, grad_y_out) = Scratch::<f32>::try_extend_pair(src.len(), budget, |a, b| {
        copy_tile_into_both(src, a, b);
    })?;
    Ok((
        Tensor::from_scratch(grad_x, x.shape())?,
        Tensor::from_scratch(grad_y_out, y.shape())?,
    ))
}

/// `f32`s per stack tile. 4096 is 16 KiB, which stays in L1: the gradient
/// chunk is loaded once and then appended to each output.
const ADD_BWD_TILE: usize = 4096;

fn copy_tile_into_both(src: &[f32], a: &mut Vec<f32>, b: &mut Vec<f32>) {
    let mut tile = [0f32; ADD_BWD_TILE];
    let mut chunks = src.chunks_exact(ADD_BWD_TILE);
    for chunk in chunks.by_ref() {
        tile.copy_from_slice(chunk);
        a.extend_from_slice(&tile);
        b.extend_from_slice(&tile);
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
        let k = rest.len();
        tile[..k].copy_from_slice(rest);
        a.extend_from_slice(&tile[..k]);
        b.extend_from_slice(&tile[..k]);
    }
}

/// `weight` is `[n_head, d_model]` (nn.Linear with bias).
/// `input` is `[..., d_model]`, `attn` is `[..., n_head, head_dim]`.
/// Output is `attn * sigmoid(input @ W^T + bias)`, with the sigmoid
/// broadcast over the head dimension. The output has the shape of `attn`.
///
/// The logits are one product on the GEMM core. The broadcast multiply is
/// `attn * sigmoid`, one rounding per value. Under [`Numerics::Exact`] it
/// runs in row chunks on the pool. Under [`Numerics::Fast`] it is
/// [`scale_heads`]: a serial reserve-and-scale, no spawn and no zero-fill.
/// Under [`Numerics::Exact`] each logit is
/// `((b + x_0 w_0) + x_1 w_1) + ...` in ascending input index without
/// `mul_add`, the order of the scalar loop this replaced: the product runs
/// over `[1, x]` and `[b, w]`, whose first term `0 + 1·b` is `b`.
/// Under [`Numerics::Fast`] on macOS each finite logit is [`sigmoid`] via
/// Accelerate `vvexpf` (`e^{-|z|}`, then the same stable pair). Exact, and
/// Fast off macOS, stay on libm [`sigmoid`]. Interleaved at the nanolab
/// shape (`x` `[1, 1024, 768]`, 12 heads of 64) on 6 threads, min of 20
/// after 2 warmups: libm sigmoid 0.023 ms and this path 0.015 ms; the whole
/// forward 0.180 ms against 0.173 ms. The Fast broadcast was then still the
/// parallel fill (about 0.09 ms, of which spawn about 0.03, zero-fill about
/// 0.012, and the multiply about 0.005). One serial write of each `a * g`
/// into spare capacity, interleaved the same way (min of 24 after 2
/// warmups), made the whole forward 0.158 ms against 0.178 ms for that fill.
/// On macOS Fast each stored sigmoid is then clamped into `[0, 1]` (a
/// non-finite scale is refused, not turned into 0 or 1) and the output scan
/// is skipped. Attention was already scanned on the way in, and a finite
/// value times a scale in `[0, 1]` cannot overflow. Alternating at this
/// shape on 6 threads, min of 24 after 2 warmups: 0.1237 ms against 0.1493 ms
/// with the scan, and the clamp changed no bit of that output.
///
/// `dims` comes from [`ojas_core::per_head_sigmoid_gate_forward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    attn: &[f32],
    dims: GateDims,
    out_shape: &[usize],
    keep_scales: bool,
) -> Result<(Tensor, Option<Tensor>), OjasError> {
    gate_lengths(op, &dims, input, weight, bias, attn)?;
    let keep = keep_scales && exec.numerics == Numerics::Fast;
    let z_len = product(op, &[dims.rows, dims.heads])?;
    // The scale's only charge, taken before the logit hold. That hold covers
    // the product scratch; it does not cover these bytes a second time.
    let scale_charge = if keep {
        Some(budget.try_reserve(payload_bytes(op, z_len)?)?)
    } else {
        None
    };
    let logit = logit_work(op, exec, &dims)?;
    let scratch = if keep {
        logit
            .checked_sub(z_len)
            .ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: "logit scratch shorter than the saved scale".to_string(),
            })?
    } else {
        logit
    };
    // The logits phase, held while the output is charged and written.
    let _hold = room_for(op, budget, scratch)?;
    let mut gates = gate_values(op, exec, &dims, input, weight, bias)?;
    let (heads, dh) = (dims.heads, dims.head_dim);
    let width = heads * dh;
    let y = if exec.numerics == Numerics::Fast {
        fast_gate_broadcast(
            op, budget, attn, &mut gates, dh, dims.rows, width, out_shape,
        )?
    } else {
        fill_rows(op, budget, exec, out_shape, dims.rows, width, |range, y| {
            let src = &attn[range.start * width..range.end * width];
            let g = &gates[range.start * heads..range.end * heads];
            for ((dst, src), &g) in y.chunks_exact_mut(dh).zip(src.chunks_exact(dh)).zip(g) {
                for (out, &a) in dst.iter_mut().zip(src) {
                    *out = a * g;
                }
            }
            Ok(())
        })?
    };
    // Exact never keeps the scale: its backward still recomputes the logits.
    // Fast keeps the buffer `gate_values` already wrote, moved into the
    // reservation taken above. No element is copied and nothing is reserved
    // again.
    let scales = if let Some(reservation) = scale_charge {
        Some(persist_gate_scales(&dims, gates, reservation)?)
    } else {
        drop(gates);
        None
    };
    Ok((y, scales))
}

/// Move `gates` (`[rows, heads]`, already the sigmoid) into `reservation`.
///
/// `reservation` is the buffer's only charge. No element is copied and
/// nothing is zero-filled or reserved again. A length that is not
/// `rows * heads` drops the buffer and releases the charge.
fn persist_gate_scales(
    dims: &GateDims,
    gates: Vec<f32>,
    reservation: Reservation,
) -> Result<Tensor, OjasError> {
    let scratch = Scratch::<f32>::try_adopt(gates, reservation)?;
    Tensor::from_scratch(scratch, &[dims.rows, dims.heads])
}

/// Fast broadcast: reserve `attn.len()` and scale each head into that spare
/// capacity, one write per value ([`ojas_simd::scale_heads_append`]).
///
/// Nothing is zero-filled and the pool is not used. When `scan_output` is
/// set, a non-finite product drops the scratch, so the charge is released,
/// and is [`OjasError::NonFinite`]; a finite output is recorded finite.
/// The macOS Fast forward passes `false` only after `bound_sigmoid_scales`:
/// attention was already scanned on the way in, and every scale that reaches
/// the multiply is in `[0, 1]`, so the product cannot overflow and is
/// recorded finite without a second pass. A `shape` whose element count is
/// not `rows * width` is refused before the reserve, the same check as
/// [`fill_rows`].
#[allow(clippy::too_many_arguments)]
fn scale_heads(
    op: &'static str,
    budget: &Budget,
    attn: &[f32],
    gates: &[f32],
    head_dim: usize,
    rows: usize,
    width: usize,
    out_shape: &[usize],
    scan_output: bool,
) -> Result<Tensor, OjasError> {
    if product(op, out_shape)? != product(op, &[rows, width])? {
        return Err(shape(
            op,
            format!("output {out_shape:?} does not hold {rows} rows of {width}"),
        ));
    }
    finish_extend(op, budget, attn.len(), out_shape, |dst| {
        ojas_simd::scale_heads_append(attn, gates, head_dim, dst).map_err(|err| {
            OjasError::Backend {
                id: BackendId::Cpu,
                detail: format!("{op}: {err}"),
            }
        })?;
        if scan_output {
            Ok(all_finite(dst))
        } else {
            debug_assert!(
                gates
                    .iter()
                    .all(|g| g.is_finite() && (0.0..=1.0).contains(g)),
                "{op}: output scan skipped with a scale outside [0, 1]"
            );
            Ok(true)
        }
    })
}

/// macOS Fast broadcast: clamp each finite sigmoid into `[0, 1]` and skip
/// the output scan. Off macOS the stored scales are multiplied as they are
/// and the products are scanned.
#[allow(clippy::too_many_arguments)]
fn fast_gate_broadcast(
    op: &'static str,
    budget: &Budget,
    attn: &[f32],
    gates: &mut [f32],
    head_dim: usize,
    rows: usize,
    width: usize,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    #[cfg(target_os = "macos")]
    {
        // `per_head_sigmoid_gate_forward` is the only caller of
        // [`gate_forward`]. It scans `attn` with
        // [`crate::validate::f32_operands`] before that call, so every
        // attention value here is finite. The scales were just stored by
        // [`gate_values`]. A non-finite scale is refused here, before the
        // output is reserved, and the logit charge in [`gate_forward`]
        // drops with this error. A finite scale outside `[0, 1]` is clamped
        // to that interval (`-0` stays `-0`). The multiply is then a finite
        // value times a scale in `[0, 1]`, which cannot overflow, so the
        // broadcast is not scanned.
        bound_sigmoid_scales(op, gates)?;
        scale_heads(
            op, budget, attn, gates, head_dim, rows, width, out_shape, false,
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        scale_heads(
            op, budget, attn, gates, head_dim, rows, width, out_shape, true,
        )
    }
}

/// After the sigmoid is stored: refuse a non-finite scale without writing a
/// number over it, and clamp every finite scale into `[0, 1]`.
///
/// `NaN` and both infinities fail [`all_finite`] and leave `scales`
/// unchanged. `-0` compares equal to `0` and is not rewritten.
#[cfg(target_os = "macos")]
fn bound_sigmoid_scales(op: &'static str, scales: &mut [f32]) -> Result<(), OjasError> {
    if !all_finite(scales) {
        return Err(nonfinite(op));
    }
    for v in scales.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        } else if *v > 1.0 {
            *v = 1.0;
        }
    }
    Ok(())
}

/// `[grad_input, grad_weight, grad_bias, grad_attn]`: the caller's zeroed
/// outputs, written in place.
pub(crate) type GateGrads<'a> = [&'a mut [f32]; 4];

/// `gz = (sum_d grad_y · attn) · g · (1 - g)` per row and head, then
/// `grad_input = gz · W` and `grad_weight = gzᵀ · input` on the GEMM core,
/// `grad_bias` the column sums of `gz` in ascending row order, and
/// `grad_attn = grad_y · g`. Under [`Numerics::Exact`] every sum ascends
/// from `+0.0` without `mul_add`, which is the order of the scalar loop this
/// replaced; under [`Numerics::Fast`] the per-head dot uses
/// [`dot_lanes`]. `grad_attn` and `gz` are filled in row chunks on scoped
/// threads, and the two products write straight into their outputs
/// ([`gemm_out`]); `grad_bias` sums into its zeroed output. `dims` comes
/// from [`ojas_core::per_head_sigmoid_gate_backward_dims`].
///
/// Returns whether `grad_attn` is finite by construction. That is true only
/// on macOS [`Numerics::Fast`]: `g` is the `vvexpf` sigmoid, clamped into
/// `[0, 1]` by [`bound_sigmoid_scales`] (a non-finite scale is refused and
/// is not stored as 0 or 1). The caller has already refused a non-finite
/// `grad_y`, and a finite value times a scale in `[0, 1]` cannot overflow,
/// so that product is not scanned. `grad_bias` and the two GEMM gradients
/// still can overflow and are still scanned. Exact never clamps and this
/// returns false. Alternating the whole backward at the nanolab shape on 6
/// threads, min of 24 after 2 warmups: 0.2974 ms against 0.3226 ms with the
/// `grad_attn` scan. The clamp changed no bit of that backward.
///
/// A Fast caller can pass the scale the forward kept (`[rows, heads]`,
/// already clamped on macOS). That skips the logit product. The scale is
/// borrowed when every value is finite and in `[0, 1]`; a finite value
/// outside that interval is copied and clamped, and a non-finite scale
/// refuses and releases the scratch charge. `grad_bias` and the two GEMM
/// gradients are still scanned. Exact ignores a saved scale and recomputes.
/// Alternating at this shape on 6 threads, min of 20 after 2 warmups:
/// forward 0.1447 ms against 0.1451 ms without keeping the scale, backward
/// 0.2501 ms against 0.3359 ms recomputing, and the pair 0.3996 ms against
/// 0.4928 ms.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [input, weight, bias, attn, grad_y]: [&[f32]; 5],
    dims: GateDims,
    saved_scales: Option<&[f32]>,
    [grad_x, grad_w, grad_b, grad_attn]: GateGrads<'_>,
) -> Result<bool, OjasError> {
    gate_lengths(op, &dims, input, weight, bias, attn)?;
    if grad_y.len() != attn.len() {
        return Err(shape(op, "gate grad length does not match attn"));
    }
    if grad_x.len() != input.len()
        || grad_w.len() != weight.len()
        || grad_b.len() != bias.len()
        || grad_attn.len() != attn.len()
    {
        return Err(shape(op, "gate gradient length does not match its operand"));
    }
    let GateDims {
        rows,
        d_model: din,
        heads,
        head_dim: dh,
    } = dims;
    let gz_len = product(op, &[rows, heads])?;
    // Exact ignores a saved scale and recomputes, the same as no scale.
    let saved = match saved_scales {
        Some(saved) if exec.numerics == Numerics::Fast => Some(saved),
        _ => None,
    };
    if let Some(saved) = saved {
        if saved.len() != gz_len {
            return Err(shape(op, "gate scale length does not match rows * heads"));
        }
    }
    // Charged at once: the logits phase, or with a saved Fast scale room for
    // one copy of it (taken only when a finite value lies outside `[0, 1]`),
    // plus `gz` and the larger gradient product's scratch. The four
    // gradients are the caller's. The saved scale itself is the caller's tensor.
    let grads =
        gemm_scratch(op, exec, rows, heads, din)?.max(gemm_scratch(op, exec, heads, rows, din)?);
    let work = if saved.is_some() {
        [gz_len, gz_len, grads]
    } else {
        [logit_work(op, exec, &dims)?, gz_len, grads]
    }
    .into_iter()
    .try_fold(0usize, |total, n| add(op, total, n))?;
    let _hold = room_for(op, budget, work)?;
    let (gates_buf, attn_proven) = if let Some(saved) = saved {
        // A non-finite scale refuses here. Dropping `_hold` releases the charge.
        fast_saved_scales(op, exec.numerics, saved)?
    } else {
        let mut gates = gate_values(op, exec, &dims, input, weight, bias)?;
        let proven = clamp_fast_scales(op, exec.numerics, &mut gates)?;
        (Cow::Owned(gates), proven)
    };
    let gates = gates_buf.as_ref();
    let numerics = exec.numerics;
    let width = heads * dh;
    let min_rows = scoped::min_rows(width);
    let mut gz = vec![0.0f32; gz_len];
    scoped::chunks_into_n(
        exec,
        [grad_attn, &mut gz],
        rows,
        [width, heads],
        min_rows,
        |range, [grad_attn, gz]| {
            let attn = &attn[range.start * width..range.end * width];
            let gy = &grad_y[range.start * width..range.end * width];
            let g = &gates[range.start * heads..range.end * heads];
            let heads_in_range = grad_attn
                .chunks_exact_mut(dh)
                .zip(attn.chunks_exact(dh))
                .zip(gy.chunks_exact(dh))
                .zip(g.iter().zip(gz.iter_mut()));
            for (((dst, a), gy), (&g, gz)) in heads_in_range {
                for (out, &v) in dst.iter_mut().zip(gy) {
                    *out = v * g;
                }
                let grad_g = match numerics {
                    Numerics::Exact => dot_ascending(gy, a),
                    Numerics::Fast => dot_lanes(gy, a),
                };
                *gz = grad_g * g * (1.0 - g);
            }
            Ok(())
        },
    )?;
    for row in gz.chunks_exact(heads) {
        for (slot, &v) in grad_b.iter_mut().zip(row) {
            *slot += v;
        }
    }
    let gz = Mat::row_major(&gz, rows, heads);
    // grad_x[row, i] = sum_head gz[row, head] * W[head, i], head from 0.
    gemm_out(op, exec, &gz, &Mat::row_major(weight, heads, din), grad_x)?;
    // grad_w[head, i] = sum_row gz[row, head] * x[row, i], row from 0.
    gemm_out(op, exec, &gz.t(), &Mat::row_major(input, rows, din), grad_w)?;
    Ok(attn_proven)
}

/// macOS Fast only: refuse a non-finite sigmoid and clamp every finite one
/// into `[0, 1]`, the same rule as the forward broadcast. `true` means every
/// scale that reaches the `grad_attn` multiply is in that interval.
fn clamp_fast_scales(
    op: &'static str,
    numerics: Numerics,
    gates: &mut [f32],
) -> Result<bool, OjasError> {
    #[cfg(target_os = "macos")]
    if numerics == Numerics::Fast {
        bound_sigmoid_scales(op, gates)?;
        debug_assert!(
            gates
                .iter()
                .all(|g| g.is_finite() && (0.0..=1.0).contains(g)),
            "{op}: grad_attn treated as finite with a scale outside [0, 1]"
        );
        return Ok(true);
    }
    let _ = (op, numerics, gates);
    Ok(false)
}

/// A Fast scale saved by the forward. Non-finite refuses. On macOS a finite
/// value outside `[0, 1]` is copied and clamped, the same rule as
/// [`bound_sigmoid_scales`]; a scale already in range is borrowed, and
/// `true` means `grad_attn` is finite by construction. Off macOS nothing is
/// clamped and the flag is false.
fn fast_saved_scales<'a>(
    op: &'static str,
    numerics: Numerics,
    saved: &'a [f32],
) -> Result<(Cow<'a, [f32]>, bool), OjasError> {
    if !all_finite(saved) {
        return Err(nonfinite(op));
    }
    #[cfg(target_os = "macos")]
    if numerics == Numerics::Fast {
        if saved.iter().any(|g| *g < 0.0 || *g > 1.0) {
            let mut owned = saved.to_vec();
            bound_sigmoid_scales(op, &mut owned)?;
            return Ok((Cow::Owned(owned), true));
        }
        debug_assert!(
            saved
                .iter()
                .all(|g| g.is_finite() && (0.0..=1.0).contains(g)),
            "{op}: grad_attn treated as finite with a scale outside [0, 1]"
        );
        return Ok((Cow::Borrowed(saved), true));
    }
    let _ = numerics;
    Ok((Cow::Borrowed(saved), false))
}

fn add(op: &'static str, a: usize, b: usize) -> Result<usize, OjasError> {
    a.checked_add(b).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "scratch length overflows".to_string(),
    })
}

/// `sigmoid(input · Wᵀ + b)` as `[rows, heads]`. A logit that is not finite
/// is refused, not turned into a gate of 0 or 1. The charge for the logits
/// is the caller's [`room_for`] hold, released when this returns an error.
fn gate_values(
    op: &'static str,
    exec: Exec<'_>,
    dims: &GateDims,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, OjasError> {
    let (rows, din, heads) = (dims.rows, dims.d_model, dims.heads);
    let mut z = match exec.numerics {
        Numerics::Exact => {
            // `[1, x]` and `[b, w]`: the product's first term is the bias.
            let width = din + 1;
            let mut ones_x = Vec::with_capacity(product(op, &[rows, width])?);
            for row in input.chunks_exact(din) {
                ones_x.push(1.0f32);
                ones_x.extend_from_slice(row);
            }
            let mut bias_w = Vec::with_capacity(product(op, &[heads, width])?);
            for (row, &b) in weight.chunks_exact(din).zip(bias) {
                bias_w.push(b);
                bias_w.extend_from_slice(row);
            }
            let a = Mat::row_major(&ones_x, rows, width);
            let w = Mat::row_major(&bias_w, heads, width);
            gemm(op, exec, &a, &w.t())?
        }
        Numerics::Fast => {
            let a = Mat::row_major(input, rows, din);
            let w = Mat::row_major(weight, heads, din);
            let mut z = gemm(op, exec, &a, &w.t())?;
            for row in z.chunks_exact_mut(heads) {
                for (v, &b) in row.iter_mut().zip(bias) {
                    *v += b;
                }
            }
            z
        }
    };
    map_logits(op, logit_sigmoid(exec.numerics), &mut z)?;
    Ok(z)
}

/// Which sigmoid turns gate logits into gates.
#[derive(Clone, Copy)]
enum LogitSigmoid {
    /// libm [`sigmoid`]. Exact, and Fast off macOS.
    Libm,
    /// Accelerate `vvexpf` of `e^{-|z|}`, then the stable pair. macOS Fast.
    #[cfg(target_os = "macos")]
    Vvexpf,
}

fn logit_sigmoid(numerics: Numerics) -> LogitSigmoid {
    #[cfg(target_os = "macos")]
    if numerics == Numerics::Fast {
        return LogitSigmoid::Vvexpf;
    }
    let _ = numerics;
    LogitSigmoid::Libm
}

fn map_logits(op: &'static str, which: LogitSigmoid, z: &mut [f32]) -> Result<(), OjasError> {
    match which {
        LogitSigmoid::Libm => {
            for v in z.iter_mut() {
                if !v.is_finite() {
                    return Err(nonfinite(op));
                }
                *v = sigmoid(*v);
            }
            Ok(())
        }
        #[cfg(target_os = "macos")]
        LogitSigmoid::Vvexpf => map_logits_vvexpf(op, z),
    }
}

/// Finite logits only. The sign store tests each loaded lane and refuses a
/// non-finite one before `vvexpf`, and before that lane is stored. Earlier
/// lanes of this scratch may already hold `-|x|`. The error drops `z`, so
/// the logit charge in [`gate_forward`] is released and no gate is published.
#[cfg(target_os = "macos")]
fn map_logits_vvexpf(op: &'static str, z: &mut [f32]) -> Result<(), OjasError> {
    let mut negative = Vec::new();
    match ojas_simd::store_neg_abs_signs(z, &mut negative) {
        Ok(()) => {}
        Err(ojas_simd::SimdError::NonFinite) => return Err(nonfinite(op)),
        Err(err) => return Err(vdsp_refused(op, err)),
    }
    // `z` now holds `-|logit|` and `negative` is `logit < 0` (`-0` is not
    // negative, matching [`sigmoid`]). One `vvexpf`: it is elementwise, so
    // splitting only at the C `int` limit does not change a gate's bits.
    const MAX: usize = i32::MAX as usize;
    for chunk in z.chunks_mut(MAX) {
        ojas_simd::vvexpf_inplace(chunk).map_err(|err| vdsp_refused(op, err))?;
    }
    for (&neg, v) in negative.iter().zip(z.iter_mut()) {
        let e = *v;
        let big = 1.0 / (1.0 + e);
        *v = if neg != 0 { e * big } else { big };
    }
    Ok(())
}

/// `sum a[i] * b[i]` from `+0.0` in ascending `i`, separate multiply and add.
fn dot_ascending(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |acc, (x, y)| acc + x * y)
}

/// Lanes of the [`Numerics::Fast`] reductions.
const LANES: usize = 8;

/// `sum a[i] * b[i]` under [`Numerics::Fast`]: [`LANES`] partial sums with
/// `mul_add` (lane `l` takes `i ≡ l mod LANES` in ascending order), added in
/// lane order, then the tail in ascending order. The order depends only on
/// the length, so the loop vectorizes and the bits do not depend on the
/// thread count.
fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.iter().zip(cb) {
        for lane in 0..LANES {
            acc[lane] = fma(x[lane], y[lane], acc[lane]);
        }
    }
    let mut total = acc.iter().fold(0.0f32, |s, &v| s + v);
    for (&x, &y) in ra.iter().zip(rb) {
        total = fma(x, y, total);
    }
    total
}

/// The copies hold as many values as `dims` says they do.
fn gate_lengths(
    op: &'static str,
    dims: &GateDims,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    attn: &[f32],
) -> Result<(), OjasError> {
    if input.len() != product(op, &[dims.rows, dims.d_model])?
        || weight.len() != product(op, &[dims.heads, dims.d_model])?
        || bias.len() != dims.heads
        || attn.len() != product(op, &[dims.rows, dims.heads, dims.head_dim])?
    {
        return Err(shape(op, "gate data length does not match shape"));
    }
    Ok(())
}

/// Floats [`gate_values`] allocates: the logits, the logit product's
/// scratch and, under [`Numerics::Exact`], the `[1, x]` and `[b, w]`
/// operands.
fn logit_work(op: &'static str, exec: Exec<'_>, dims: &GateDims) -> Result<usize, OjasError> {
    let z = product(op, &[dims.rows, dims.heads])?;
    match exec.numerics {
        Numerics::Exact => {
            let width = add(op, dims.d_model, 1)?;
            let ones_x = product(op, &[dims.rows, width])?;
            let bias_w = product(op, &[dims.heads, width])?;
            let work = gemm_scratch(op, exec, dims.rows, width, dims.heads)?;
            [ones_x, bias_w, work]
                .into_iter()
                .try_fold(z, |total, n| add(op, total, n))
        }
        Numerics::Fast => add(
            op,
            z,
            gemm_scratch(op, exec, dims.rows, dims.d_model, dims.heads)?,
        ),
    }
}

/// `y = (1 - sigmoid(lambda)) * value + sigmoid(lambda) * value0` as a
/// tensor of `shape`, split and written in place like [`mul_forward`].
pub(crate) fn value_residual_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    value: &[f32],
    value0: &[f32],
    lambda: f32,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    same_len(
        op,
        &[value.len(), value0.len(), product(op, shape)?],
        "value residual operand",
    )?;
    let s = sigmoid(lambda);
    if !s.is_finite() {
        return Err(nonfinite(op));
    }
    fill_rows(op, budget, exec, shape, value.len(), 1, |range, out| {
        for ((o, &v), &v0) in out
            .iter_mut()
            .zip(&value[range.clone()])
            .zip(&value0[range])
        {
            *o = (1.0 - s) * v + s * v0;
        }
        Ok(())
    })
}

/// Values per block of the [`Numerics::Fast`] `lambda` gradient sum. Block
/// boundaries are fixed by the length alone, never by the task split.
const LAMBDA_BLOCK: usize = 1 << 12;

/// Writes `grad_value` and `grad_value0` into the caller's outputs and
/// returns `grad_lambda`.
///
/// It runs in [`LAMBDA_BLOCK`] chunks on scoped threads, each task filling
/// its blocks of both gradients in place and, under [`Numerics::Fast`],
/// returning its blocks' `grad_lambda` sums.
///
/// `grad_lambda = s (1 - s) sum_i (value0[i] - value[i]) grad_y[i]` with
/// `s = sigmoid(lambda)`, a sum over every element.
/// - [`Numerics::Exact`]: the sum ascends from index 0 in `f32` with
///   separate multiply and add, the crate's reduction contract (lib.rs and
///   `ojas_core::Numerics`), on the calling thread after the gradients; its
///   error grows with the element count.
/// - [`Numerics::Fast`]: each term is formed in `f64` (exact whenever
///   `value` and `value0` are within a factor 2^29 of each other, otherwise
///   rounded at 2^-53), summed over fixed [`LAMBDA_BLOCK`]-value blocks in
///   [`LANES`] `f64` partial sums, the block sums added in block order, and
///   `s (1 - s)` applied in `f64` before the one rounding to `f32`. At the
///   nanolab shape (786k values) that is within 1e-6 of the `f64` sum where
///   the ascending `f32` sum was 3.2e-5 off.
pub(crate) fn value_residual_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [value, value0, grad_y]: [&[f32]; 3],
    lambda: f32,
    [grad_v, grad_v0]: [&mut [f32]; 2],
) -> Result<f32, OjasError> {
    let n = value.len();
    same_len(
        op,
        &[n, value0.len(), grad_y.len(), grad_v.len(), grad_v0.len()],
        "value residual grad",
    )?;
    let s = sigmoid(lambda);
    let blocks = n.div_ceil(LAMBDA_BLOCK);
    // One f64 (two f32) per block; the gradients are the caller's.
    let _sums = room_for(op, budget, product(op, &[blocks, 2])?)?;
    let numerics = exec.numerics;
    let min_blocks = ROW_MIN_ELEMS.div_ceil(LAMBDA_BLOCK);
    let sums = scoped::chunks_into_n(
        exec,
        [grad_v, grad_v0],
        blocks,
        [LAMBDA_BLOCK; 2],
        min_blocks,
        |block_range, [grad_v, grad_v0]| {
            let range = block_range.start * LAMBDA_BLOCK..(block_range.end * LAMBDA_BLOCK).min(n);
            let gy = &grad_y[range.clone()];
            for ((gv, gv0), &g) in grad_v.iter_mut().zip(grad_v0.iter_mut()).zip(gy) {
                *gv = (1.0 - s) * g;
                *gv0 = s * g;
            }
            Ok(match numerics {
                Numerics::Exact => Vec::new(),
                Numerics::Fast => range
                    .step_by(LAMBDA_BLOCK)
                    .map(|start| {
                        let end = (start + LAMBDA_BLOCK).min(n);
                        diff_dot_f64(&value[start..end], &value0[start..end], &grad_y[start..end])
                    })
                    .collect::<Vec<f64>>(),
            })
        },
    )?;
    let grad_lambda = match numerics {
        Numerics::Exact => {
            let mut grad_s = 0.0f32;
            for ((&v, &v0), &g) in value.iter().zip(value0).zip(grad_y) {
                grad_s += (v0 - v) * g;
            }
            grad_s * s * (1.0 - s)
        }
        Numerics::Fast => {
            // The block sums in block order: task order, then each task's.
            let grad_s = sums.iter().flatten().fold(0.0f64, |total, &b| total + b);
            let s = 1.0 / (1.0 + (-f64::from(lambda)).exp());
            (grad_s * s * (1.0 - s)) as f32
        }
    };
    Ok(grad_lambda)
}

/// `sum_i (value0[i] - value[i]) * grad_y[i]` with every term formed in
/// `f64`, in [`LANES`] partial sums added in lane order, then the tail.
fn diff_dot_f64(value: &[f32], value0: &[f32], grad_y: &[f32]) -> f64 {
    let (cv, rv) = value.as_chunks::<LANES>();
    let (cv0, rv0) = value0.as_chunks::<LANES>();
    let (cg, rg) = grad_y.as_chunks::<LANES>();
    let mut acc = [0.0f64; LANES];
    for ((v, v0), g) in cv.iter().zip(cv0).zip(cg) {
        for lane in 0..LANES {
            acc[lane] += (f64::from(v0[lane]) - f64::from(v[lane])) * f64::from(g[lane]);
        }
    }
    let mut total = acc.iter().fold(0.0f64, |s, &v| s + v);
    for ((&v, &v0), &g) in rv.iter().zip(rv0).zip(rg) {
        total += (f64::from(v0) - f64::from(v)) * f64::from(g);
    }
    total
}

/// Logits per cross-entropy task. A task's row count is fixed by the
/// vocabulary alone, so the partition does not depend on the thread count.
const CE_BLOCK_ELEMS: usize = 1 << 18;

/// Rows per cross-entropy task.
fn ce_block_rows(vocab: usize) -> usize {
    (CE_BLOCK_ELEMS / vocab.max(1)).max(1)
}

/// Count of valid targets, after checking that every target other than
/// `ignore` is inside the vocabulary. No valid target is
/// [`OjasError::NonFinite`]: the mean has no denominator (torch gives NaN),
/// and it is not a finite loss of 0.
fn ce_valid(
    op: &'static str,
    targets: &[u32],
    vocab: usize,
    ignore: Option<u32>,
) -> Result<u32, OjasError> {
    let mut valid: u32 = 0;
    for (n, &target) in targets.iter().enumerate() {
        if ignore == Some(target) {
            continue;
        }
        if id_index(target) >= vocab {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("target {target} at {n} is outside vocab {vocab}"),
            });
        }
        valid = valid.checked_add(1).ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "valid target count overflows".to_string(),
        })?;
    }
    if valid == 0 {
        return Err(nonfinite(op));
    }
    Ok(valid)
}

/// Magnitude bits of an f32 (the sign cleared); a value is NaN or infinite
/// exactly when they are at least [`NON_FINITE`].
const MAGNITUDE: u32 = 0x7fff_ffff;
const NON_FINITE: u32 = 0x7f80_0000;

/// Largest value of a row, and whether every value is finite: the scalar
/// loop under Exact, lane maxima under Fast. Every value is read either
/// way, so this pass is also the row's NaN scan. Among finite values the
/// order of comparisons cannot change the maximum.
fn row_max(row: &[f32], numerics: Numerics) -> (f32, bool) {
    let value = |v: &f32| *v;
    let bits = |v: &f32| v.to_bits() & MAGNITUDE;
    match numerics {
        Numerics::Exact => {
            let (mut max, mut top) = (f32::NEG_INFINITY, 0u32);
            for word in row {
                let v = value(word);
                if v > max {
                    max = v;
                }
                top = top.max(bits(word));
            }
            (max, top < NON_FINITE)
        }
        Numerics::Fast => {
            let (chunks, rest) = row.as_chunks::<LANES>();
            let mut acc = [f32::NEG_INFINITY; LANES];
            let mut top = [0u32; LANES];
            for chunk in chunks {
                for ((a, t), word) in acc.iter_mut().zip(top.iter_mut()).zip(chunk) {
                    *a = a.max(value(word));
                    *t = (*t).max(bits(word));
                }
            }
            let mut max = acc.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let mut top = top.iter().fold(0u32, |m, &t| m.max(t));
            for word in rest {
                max = max.max(value(word));
                top = top.max(bits(word));
            }
            (max, top < NON_FINITE)
        }
    }
}

/// `sum_j e^(x_j - max)` of one row, each term also written to `store` when
/// it is given. Exact: libm `exp` and an ascending f32 sum, the reference
/// order. Fast: [`crate::exp`] in its fixed lane order.
fn row_exp_sum(
    op: &'static str,
    row: &[f32],
    max: f32,
    numerics: Numerics,
    store: Option<&mut [f32]>,
) -> Result<f32, OjasError> {
    let sum = match (numerics, store) {
        (Numerics::Exact, store) => {
            let mut sum = 0.0f32;
            let mut store = store;
            for (col, word) in row.iter().enumerate() {
                let e = (*word - max).exp();
                if !e.is_finite() {
                    return Err(nonfinite(op));
                }
                if let Some(out) = store.as_deref_mut() {
                    out[col] = e;
                }
                sum += e;
            }
            sum
        }
        (Numerics::Fast, Some(out)) => exp_sub_store(row, max, out),
        (Numerics::Fast, None) => exp_sub_sum(row, max),
    };
    if !(sum.is_finite() && sum > 0.0) {
        return Err(nonfinite(op));
    }
    Ok(sum)
}

/// A mean cross-entropy over `logits` `[rows, vocab]` and `targets`
/// `[rows]`, both read in place, with `valid` rows counted by
/// [`ce_valid`].
#[derive(Clone, Copy)]
pub(crate) struct CeInput<'a> {
    pub logits: &'a [f32],
    pub targets: &'a [u32],
    pub vocab: usize,
    pub ignore: Option<u32>,
    pub valid: u32,
}

impl CeInput<'_> {
    /// One row's loss term `max + ln(sum) - x[target]`, or `None` for an
    /// ignored row. With `grad`, the row's `(softmax - onehot) / valid` is
    /// written into it. Every logit of the row is checked finite, ignored
    /// rows included.
    fn row(
        &self,
        op: &'static str,
        numerics: Numerics,
        n: usize,
        grad: Option<&mut [f32]>,
    ) -> Result<Option<f32>, OjasError> {
        let row = self
            .logits
            .get(n * self.vocab..(n + 1) * self.vocab)
            .ok_or_else(|| outside(op, "cross-entropy row exceeds logits"))?;
        let target = self.targets[n];
        if self.ignore == Some(target) {
            return if all_finite(row) {
                Ok(None)
            } else {
                Err(nonfinite(op))
            };
        }
        let (max, finite) = row_max(row, numerics);
        if !finite {
            return Err(nonfinite(op));
        }
        let class = target as usize;
        let picked = row[class];
        let sum = match grad {
            None => row_exp_sum(op, row, max, numerics, None)?,
            Some(dst) => {
                let sum = row_exp_sum(op, row, max, numerics, Some(&mut *dst))?;
                let denom = self.valid as f32;
                for slot in dst.iter_mut() {
                    let p = *slot / sum;
                    *slot = p / denom;
                }
                dst[class] -= 1.0 / denom;
                sum
            }
        };
        Ok(Some(max + sum.ln() - picked))
    }

    /// The mean loss, in fixed blocks of rows on the pool's threads; with
    /// `grad` (`[rows, vocab]` values, zeroed) every valid row's gradient is
    /// written into it once and ignored rows keep their zeros.
    ///
    /// Each valid row's term comes back separately and the terms are added
    /// in increasing row order, so the loss bits do not depend on the blocks
    /// or the thread count, and no gradient value depends on which thread
    /// wrote it. Any non-finite logit, sum or loss is
    /// [`OjasError::NonFinite`]; this pass reads every logit, so it is also
    /// the logits' NaN scan.
    pub(crate) fn mean(
        &self,
        op: &'static str,
        exec: Exec<'_>,
        grad: Option<&mut [f32]>,
    ) -> Result<f32, OjasError> {
        let numerics = exec.numerics;
        let block = ce_block_rows(self.vocab);
        let rows = self.targets.len();
        let rows_of = |b: usize| b * block..((b + 1) * block).min(rows);
        let terms: Vec<Vec<f32>> = match grad {
            None => scoped::map(exec, rows.div_ceil(block), |b| {
                let mut terms = Vec::with_capacity(block);
                for n in rows_of(b) {
                    terms.extend(self.row(op, numerics, n, None)?);
                }
                Ok(terms)
            })?,
            Some(grad) => scoped::fill(exec, grad, block * self.vocab, |b, chunk| {
                let mut terms = Vec::with_capacity(block);
                for (n, dst) in rows_of(b).zip(chunk.chunks_exact_mut(self.vocab)) {
                    terms.extend(self.row(op, numerics, n, Some(dst))?);
                }
                Ok(terms)
            })?,
        };
        let mut total = 0.0f32;
        for term in terms.iter().flatten() {
            total += term;
        }
        let loss = total / self.valid as f32;
        if !loss.is_finite() {
            return Err(nonfinite(op));
        }
        Ok(loss)
    }
}

/// The checked [`CeInput`] of a cross-entropy call whose shapes
/// `cross_entropy_mean_*_dims` accepted: the logits' layout, then the
/// targets' layout, range and valid count, all read in place. The logits'
/// NaN scan is [`CeInput::mean`]'s pass, so each refusal here first scans
/// them ([`nonfinite_first`]): a NaN outranks it, as in argument order.
pub(crate) fn ce_input<'a>(
    op: &'static str,
    exec: Exec<'_>,
    logits: &'a Tensor,
    targets: &'a Tensor,
    dims: CeDims,
    ignore: Option<u32>,
) -> Result<CeInput<'a>, OjasError> {
    let logits_w = f32_values(op, logits)?;
    let scan_first = |err| nonfinite_first(op, exec, &[logits], err);
    let targets_w = u32_values(op, targets).map_err(scan_first)?;
    let valid = ce_valid(op, targets_w, dims.vocab, ignore).map_err(scan_first)?;
    Ok(CeInput {
        logits: logits_w,
        targets: targets_w,
        vocab: dims.vocab,
        ignore,
        valid,
    })
}

/// [`CeInput::mean`] with its gradient written once into a charged tensor
/// shaped `shape`: `out` holds `rows * vocab` zeroed f32 values. The loss is
/// formed too and a non-finite loss is refused, as the forward refuses it.
///
/// Every value is `(e / sum) / valid` with `e = e^(x - max)` in `[0, 1]` and
/// `sum >= 1` (the maximum's own term is exactly 1), minus `1 / valid` at
/// the target, so the gradient is finite by construction and is not scanned
/// again.
pub(crate) fn cross_entropy_grad(
    op: &'static str,
    exec: Exec<'_>,
    input: CeInput<'_>,
    mut out: Scratch<f32>,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    input.mean(op, exec, Some(out.as_mut_slice()))?;
    Tensor::from_scratch(out, shape)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::pool::Pool;

    /// A `shape` that disagrees with the operands is refused before
    /// anything is charged, for every elementwise forward pass: none of
    /// them may write a prefix of the operands into a smaller output or
    /// leave a larger one partly zero.
    #[test]
    fn elementwise_forwards_refuse_a_shape_that_disagrees_with_their_operands() {
        let pool = Arc::new(Pool::new(2).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let budget = Budget::new(1 << 20);
        let x = [1.0f32; 6];
        for bad in [&[2usize, 2][..], &[2, 4], &[7]] {
            let results = [
                ("mul", mul_forward("t", &budget, exec, &x, &x, bad)),
                ("add", add_forward("t", &budget, exec, &x, &x, bad)),
                (
                    "value_residual",
                    value_residual_forward("t", &budget, exec, &x, &x, 0.5, bad),
                ),
            ];
            for (name, got) in results {
                assert!(
                    matches!(got, Err(OjasError::Shape { .. })),
                    "{name} {bad:?}: {got:?}"
                );
            }
            assert_eq!(budget.live_bytes().unwrap(), 0);
        }
        let good = value_residual_forward("t", &budget, exec, &x, &x, 0.5, &[2, 3]).unwrap();
        assert_eq!(good.f32_slice().unwrap(), &x[..]);
    }

    /// An output large enough for the output scan to split into blocks
    /// across threads still refuses an overflow in its last value only, and
    /// releases its charge; with no overflow it is recorded finite.
    #[test]
    fn a_multi_block_output_refuses_an_overflow_in_its_last_value() {
        let pool = Arc::new(Pool::new(4).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let budget = Budget::new(1 << 26);
        let n = (1 << 21) + 3;
        let mut a = vec![1.0f32; n];
        let mut b = vec![2.0f32; n];
        let ok = mul_forward("t", &budget, exec, &a, &b, &[n]).unwrap();
        assert!(ok.f32_slice().unwrap().iter().all(|&v| v == 2.0));
        drop(ok);
        a[n - 1] = f32::MAX;
        b[n - 1] = f32::MAX;
        let got = mul_forward("t", &budget, exec, &a, &b, &[n]);
        assert!(matches!(got, Err(OjasError::NonFinite { .. })), "{got:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    /// The serial broadcast matches `a * g` per head, including a head longer
    /// than 4096. A non-finite product is `NonFinite` and releases the charge.
    /// A shape that is not `rows * width` is refused before the reserve.
    #[test]
    fn scale_heads_matches_scalar_products_and_releases_a_non_finite_output() {
        let budget = Budget::new(1 << 22);
        let check = |head_dim: usize, heads: usize| {
            let n = heads * head_dim;
            let attn: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 3.0).collect();
            let gates: Vec<f32> = (0..heads).map(|h| 0.1 + (h as f32) * 0.01).collect();
            let mut want = Vec::with_capacity(n);
            for (head, &g) in attn.chunks_exact(head_dim).zip(&gates) {
                for &a in head {
                    want.push(a * g);
                }
            }
            let got = scale_heads(
                "t",
                &budget,
                &attn,
                &gates,
                head_dim,
                heads,
                head_dim,
                &[n],
                true,
            )
            .unwrap();
            assert_eq!(got.f32_slice().unwrap(), want.as_slice());
            drop(got);
            assert_eq!(budget.live_bytes().unwrap(), 0);
        };
        check(100, 5);
        check(100, 41);
        check(5000, 2);
        check(64, 64);
        check(64, 12 * 1024);

        // `−0` across the streaming widths: 32-lane groups, an 8-lane pair,
        // a 4-lane pair, and the scalar tail. `assert_eq` on `f32` treats
        // `−0` as `+0`, so this compares bits.
        let patterns = [
            -0.0f32,
            0.0,
            1.0,
            -1.0,
            0.5,
            f32::MIN_POSITIVE,
            -2.5,
            f32::from_bits(0x8000_0001),
        ];
        let gates = [-0.0f32, 0.0, 1.0];
        for head_dim in [3usize, 4, 7, 8, 12, 32, 36, 64] {
            let n = gates.len() * head_dim;
            let attn: Vec<f32> = (0..n).map(|i| patterns[i % patterns.len()]).collect();
            let mut want = Vec::with_capacity(n);
            for (head, &g) in attn.chunks_exact(head_dim).zip(&gates) {
                for &a in head {
                    want.push((a * g).to_bits());
                }
            }
            let got = scale_heads(
                "t",
                &budget,
                &attn,
                &gates,
                head_dim,
                gates.len(),
                head_dim,
                &[n],
                true,
            )
            .unwrap();
            let bits: Vec<u32> = got.f32_slice().unwrap().iter().map(|v| v.to_bits()).collect();
            assert_eq!(bits, want, "head_dim {head_dim}");
            drop(got);
            assert_eq!(budget.live_bytes().unwrap(), 0);
        }

        let mut attn = vec![1.0f32; 64 * 3];
        attn[64 * 3 - 1] = f32::NAN;
        let got = scale_heads(
            "t",
            &budget,
            &attn,
            &[0.5, 0.25, 0.125],
            64,
            3,
            64,
            &[3, 64],
            true,
        );
        assert!(matches!(got, Err(OjasError::NonFinite { .. })), "{got:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let shape = scale_heads("t", &budget, &[1.0; 4], &[1.0, 1.0], 2, 2, 2, &[3], true);
        assert!(matches!(shape, Err(OjasError::Shape { .. })), "{shape:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    /// A non-finite scale is refused before any clamp writes a number over
    /// it, and before the broadcast reserves its output. Finite values
    /// outside `[0, 1]` become the nearer endpoint; `-0` stays `-0`.
    #[cfg(target_os = "macos")]
    #[test]
    fn bound_sigmoid_scales_refuses_non_finite_and_clamps_finite_outliers() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut scales = [0.25f32, bad, 1.5, -0.5];
            let err = bound_sigmoid_scales("t", &mut scales).unwrap_err();
            assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
            assert_eq!(scales[0].to_bits(), 0.25f32.to_bits());
            assert_eq!(scales[1].to_bits(), bad.to_bits());
            assert_eq!(scales[2].to_bits(), 1.5f32.to_bits());
            assert_eq!(scales[3].to_bits(), (-0.5f32).to_bits());
        }
        let mut ok = [-0.0, 0.0, 1.0, 0.5, 1.5, -0.25, 2.0];
        bound_sigmoid_scales("t", &mut ok).unwrap();
        assert_eq!(ok[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(ok[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(ok[2].to_bits(), 1.0f32.to_bits());
        assert_eq!(ok[3].to_bits(), 0.5f32.to_bits());
        assert_eq!(ok[4].to_bits(), 1.0f32.to_bits());
        assert_eq!(ok[5].to_bits(), 0.0f32.to_bits());
        assert_eq!(ok[6].to_bits(), 1.0f32.to_bits());
    }

    /// Clamping happens before the multiply, so `f32::MAX` times a scale of
    /// 2 becomes `f32::MAX` and is recorded finite. The same product scanned
    /// without the clamp is `NonFinite` and releases the charge. A `NaN`
    /// scale refuses without reserving the output; a logit hold taken
    /// around that call is the only charge, and dropping it releases it.
    #[cfg(target_os = "macos")]
    #[test]
    fn clamped_scales_skip_the_output_scan_and_a_nan_scale_releases_the_charge() {
        let budget = Budget::new(1 << 20);
        let attn = [f32::MAX, -3.0, 1.0];
        let mut gates = [2.0f32, -4.0, -0.0];
        bound_sigmoid_scales("t", &mut gates).unwrap();
        let got = scale_heads("t", &budget, &attn, &gates, 1, 3, 1, &[3], false).unwrap();
        let y = got.f32_slice().unwrap();
        assert_eq!(y[0].to_bits(), f32::MAX.to_bits());
        assert_eq!(y[1].to_bits(), (-3.0f32 * 0.0).to_bits());
        assert_eq!(y[2].to_bits(), (1.0f32 * -0.0).to_bits());
        assert!(got
            .all_finite_cached(|_| panic!("recorded finite, so not scanned again"))
            .unwrap());
        drop(got);
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let raw = scale_heads("t", &budget, &[f32::MAX], &[2.0], 1, 1, 1, &[1], true);
        assert!(matches!(raw, Err(OjasError::NonFinite { .. })), "{raw:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let hold = room_for("t", &budget, 16).unwrap();
        let held = budget.live_bytes().unwrap();
        let mut nan_gates = [0.2f32, f32::NAN, 0.4];
        let refused = fast_gate_broadcast(
            "t",
            &budget,
            &[1.0, 2.0, 3.0],
            &mut nan_gates,
            1,
            3,
            1,
            &[3],
        );
        assert!(
            matches!(refused, Err(OjasError::NonFinite { .. })),
            "{refused:?}"
        );
        assert_eq!(budget.live_bytes().unwrap(), held);
        assert_eq!(nan_gates[0].to_bits(), 0.2f32.to_bits());
        assert!(nan_gates[1].is_nan());
        assert_eq!(nan_gates[2].to_bits(), 0.4f32.to_bits());
        drop(hold);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    /// A charge that does not fit never keeps a buffer. An empty product is
    /// an empty tensor. A one-element `-0` stays, and a one-element overflow
    /// releases the charge.
    #[test]
    fn mul_forward_adopts_only_a_fully_stored_finite_buffer() {
        let pool = Arc::new(Pool::new(6).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let budget = Budget::new(0);
        let err = mul_forward("t", &budget, exec, &[1.0, 2.0], &[3.0, 4.0], &[2]).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let budget = Budget::new(1 << 20);
        let empty = mul_forward("t", &budget, exec, &[], &[], &[0]).unwrap();
        assert!(empty.f32_slice().unwrap().is_empty());
        drop(empty);
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let neg = mul_forward("t", &budget, exec, &[-0.0], &[1.0], &[1]).unwrap();
        assert_eq!(neg.f32_slice().unwrap()[0].to_bits(), (-0.0f32).to_bits());
        drop(neg);

        let err = mul_forward("t", &budget, exec, &[f32::MAX], &[f32::MAX], &[1]).unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    /// Both gradients are adopted together. A charge that cannot cover the
    /// second output allocates neither buffer. An empty pair and a one-element
    /// `-0` in both products are published. A non-finite product in either
    /// output, including the last element of a multi-chunk pair, publishes
    /// neither and releases both charges.
    #[test]
    fn mul_backward_adopts_both_outputs_or_neither() {
        let pool = Arc::new(Pool::new(6).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Exact,
        };
        let budget = Budget::new(0);
        let err = mul_backward("t", &budget, exec, &[1.0], &[2.0], &[3.0], &[1], &[1]).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let budget = Budget::new(4);
        let err = mul_backward("t", &budget, exec, &[1.0], &[2.0], &[3.0], &[1], &[1]).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let budget = Budget::new(1 << 26);
        let (ga, gb) = mul_backward("t", &budget, exec, &[], &[], &[], &[0], &[0]).unwrap();
        assert!(ga.f32_slice().unwrap().is_empty());
        assert!(gb.f32_slice().unwrap().is_empty());
        drop((ga, gb));
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let (ga, gb) =
            mul_backward("t", &budget, exec, &[-0.0], &[-0.0], &[1.0], &[1], &[1]).unwrap();
        assert_eq!(ga.f32_slice().unwrap()[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(gb.f32_slice().unwrap()[0].to_bits(), (-0.0f32).to_bits());
        drop((ga, gb));

        let err = mul_backward(
            "t",
            &budget,
            exec,
            &[1.0, 1.0],
            &[1.0, f32::INFINITY],
            &[1.0, 1.0],
            &[2],
            &[2],
        )
        .unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let err = mul_backward(
            "t",
            &budget,
            exec,
            &[f32::INFINITY, 1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[2],
            &[2],
        )
        .unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        let n = (1 << 15) * 2 + 1;
        let mut a = vec![1.0f32; n];
        let b = vec![1.0f32; n];
        let g = vec![1.0f32; n];
        a[n - 1] = f32::NAN;
        let err = mul_backward("t", &budget, exec, &a, &b, &g, &[n], &[n]).unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);

        a[n - 1] = 1.0;
        let (mut ga, gb) = mul_backward("t", &budget, exec, &a, &b, &g, &[n], &[n]).unwrap();
        let gb_bits: Vec<u32> = gb
            .f32_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        ga.f32_slice_mut().unwrap()[0] = 42.0;
        assert_eq!(
            gb.f32_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            gb_bits
        );
        drop((ga, gb));
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    /// The scale a Fast forward keeps is one charge: the logit hold already
    /// covers that buffer, and adopting it must not reserve it again. After
    /// the call the live bytes are the output plus that one scale.
    #[test]
    fn saved_gate_scale_is_charged_once() {
        let budget = Budget::new(1 << 28);
        let pool = Arc::new(Pool::new(1).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let dims = GateDims {
            rows: 2,
            d_model: 4,
            heads: 2,
            head_dim: 2,
        };
        let input = vec![0.1f32; dims.rows * dims.d_model];
        let weight = vec![0.02f32; dims.heads * dims.d_model];
        let bias = vec![0.0f32; dims.heads];
        let attn = vec![0.3f32; dims.rows * dims.heads * dims.head_dim];
        let (y, scales) = gate_forward(
            "t",
            &budget,
            exec,
            &input,
            &weight,
            &bias,
            &attn,
            dims,
            &[dims.rows, dims.heads, dims.head_dim],
            true,
        )
        .unwrap();
        let scales = scales.expect("fast forward keeps the scale");
        let y_bytes = payload_bytes("t", y.num_elements().unwrap()).unwrap();
        let z_bytes = payload_bytes("t", scales.num_elements().unwrap()).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), y_bytes + z_bytes);
        let logit = payload_bytes("t", logit_work("t", exec, &dims).unwrap()).unwrap();
        let peak = budget.peak_bytes();
        let once = logit + y_bytes;
        assert!(
            peak <= once,
            "saved scale charged twice: peak {peak} > logit hold + output {once} (extra {})",
            peak.saturating_sub(once)
        );
    }

    /// NEON signs and `-|z|` match the scalar loop, including a tail and
    /// `-0` (`-0 < 0` is false; `-|±0|` is stored as `-0`). A non-finite
    /// logit in this buffer is refused before the chunk that holds it is
    /// stored.
    #[cfg(target_os = "macos")]
    #[test]
    fn fast_gate_neg_abs_matches_scalar_including_negative_zero() {
        let mut values = vec![
            -0.0,
            0.0,
            -1.0,
            2.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            40.0,
            -40.0,
            0.5,
            -0.5,
            f32::from_bits(0x8000_0001),
            3.0,
        ];
        values.extend((0..20).map(|i| (i as f32) - 9.5));
        for n in [0usize, 1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, values.len()] {
            let src = &values[..n];
            let mut scalar = src.to_vec();
            let mut signs = Vec::with_capacity(n);
            for v in scalar.iter_mut() {
                signs.push(*v < 0.0);
                *v = -v.abs();
            }
            const MAX: usize = i32::MAX as usize;
            for chunk in scalar.chunks_mut(MAX) {
                ojas_simd::vvexpf_inplace(chunk).unwrap();
            }
            for (&neg, v) in signs.iter().zip(scalar.iter_mut()) {
                let e = *v;
                let big = 1.0 / (1.0 + e);
                *v = if neg { e * big } else { big };
            }
            let mut neon = src.to_vec();
            map_logits_vvexpf("t", &mut neon).unwrap();
            let scalar_bits: Vec<u32> = scalar.iter().map(|v| v.to_bits()).collect();
            let neon_bits: Vec<u32> = neon.iter().map(|v| v.to_bits()).collect();
            assert_eq!(neon_bits, scalar_bits, "len {n}");
        }
        let mut bad = [1.0f32, f32::NAN, -2.0, f32::INFINITY];
        let before: Vec<u32> = bad.iter().map(|v| v.to_bits()).collect();
        let err = map_logits_vvexpf("t", &mut bad).unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(bad.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);

        let mut zeros = [-0.0f32, 0.0, -0.0, 1.0];
        let mut signs = Vec::new();
        ojas_simd::store_neg_abs_signs(&mut zeros, &mut signs).unwrap();
        assert_eq!(signs, [0, 0, 0, 0]);
        assert_eq!(zeros[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(zeros[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(zeros[2].to_bits(), (-0.0f32).to_bits());
        assert_eq!(zeros[3].to_bits(), (-1.0f32).to_bits());
    }
}
