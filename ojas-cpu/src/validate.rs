use ojas_core::{Budget, DType, OjasError, Reservation, Tensor};

use crate::pool::scoped;
use crate::pool::Exec;

/// Host copy of an f32 input. `_charge` holds the copy's bytes against the
/// backend budget until this value (or the struct it was split from) drops.
pub(crate) struct F32V {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    _charge: Reservation,
}

pub(crate) fn shape(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op,
        detail: detail.into(),
    }
}

pub(crate) fn nonfinite(op: &'static str) -> OjasError {
    OjasError::NonFinite { op }
}

/// Validate one f32 input, then copy it to the host and charge the copy.
///
/// The crate forbids `unsafe`, so a tensor's bytes cannot be borrowed as
/// `&[f32]`; every op works on a decoded copy. The copy is as large as the
/// input, so it is reserved against `budget` before it is made and the
/// reservation travels in [`F32V::charge`]. Without it a cap sized for the
/// output alone admits an op that holds input, copy and output at once.
///
/// An op with several inputs uses [`f32_inputs`], so that every operand is
/// validated before the first one is charged.
pub(crate) fn f32_in(op: &'static str, budget: &Budget, t: &Tensor) -> Result<F32V, OjasError> {
    let [v] = f32_inputs(op, budget, [t])?;
    Ok(v)
}

/// Validate every tensor in `ts`, then copy and charge each in order.
///
/// Validation (dtype, layout, emptiness, placement, a NaN or infinity) runs
/// over all operands before any budget charge, so an invalid operand is
/// reported as invalid whatever the budget holds, rather than one operand's
/// copy being refused for room before another operand's NaN is seen.
pub(crate) fn f32_inputs<const N: usize>(
    op: &'static str,
    budget: &Budget,
    ts: [&Tensor; N],
) -> Result<[F32V; N], OjasError> {
    f32_input_list(op, budget, &ts)?
        .try_into()
        .map_err(|_| shape(op, "input count changed while copying"))
}

/// [`f32_inputs`] for a count known only at run time, such as the
/// gradients a clip reads. Same order: validate all, then copy each.
pub(crate) fn f32_input_list(
    op: &'static str,
    budget: &Budget,
    ts: &[&Tensor],
) -> Result<Vec<F32V>, OjasError> {
    let mut lens = Vec::with_capacity(ts.len());
    for t in ts {
        lens.push(check_f32(op, t)?);
    }
    let mut out = Vec::with_capacity(ts.len());
    for (len, t) in lens.into_iter().zip(ts) {
        out.push(copy_checked(op, budget, t, len)?);
    }
    Ok(out)
}

/// The charged host copy of `t`, a tensor of `len` values already checked by
/// [`check_f32`] or [`check_f32s`].
pub(crate) fn copy_checked(
    op: &'static str,
    budget: &Budget,
    t: &Tensor,
    len: usize,
) -> Result<F32V, OjasError> {
    let charge = room_for(op, budget, len)?;
    Ok(F32V {
        data: t.to_f32_vec()?,
        shape: t.shape().to_vec(),
        _charge: charge,
    })
}

/// Host copy of a u32 input, charged like [`F32V`].
pub(crate) struct U32V {
    pub data: Vec<u32>,
    _charge: Reservation,
}

/// [`f32_in`] for token ids and targets. A u32 is four bytes, as an f32 is.
/// An op that also reads f32 inputs calls [`check_u32`] before
/// [`f32_inputs`], so the ids are validated before anything is charged.
pub(crate) fn u32_in(op: &'static str, budget: &Budget, t: &Tensor) -> Result<U32V, OjasError> {
    let n = check_u32(op, t)?;
    let charge = room_for(op, budget, n)?;
    let data = t.to_u32_vec()?;
    Ok(U32V {
        data,
        _charge: charge,
    })
}

/// Everything [`u32_in`] refuses, without copying or charging.
pub(crate) fn check_u32(op: &'static str, t: &Tensor) -> Result<usize, OjasError> {
    let (n, _) = check_layout(op, t, DType::U32)?;
    Ok(n)
}

/// Everything [`f32_in`] refuses, without copying or charging. The NaN and
/// infinity scan reads the tensor's bytes in place: an f32 is non-finite
/// exactly when its eight exponent bits are all set.
pub(crate) fn check_f32(op: &'static str, t: &Tensor) -> Result<usize, OjasError> {
    let (n, bytes) = check_layout(op, t, DType::F32)?;
    if !bytes_all_finite(bytes) {
        return Err(nonfinite(op));
    }
    Ok(n)
}

/// Magnitude bits of an f32 (the sign cleared).
const MAGNITUDE: u32 = 0x7fff_ffff;
/// An f32 is NaN or infinite exactly when its magnitude bits are at least
/// this: all eight exponent bits set.
const NON_FINITE: u32 = 0x7f80_0000;
/// Values per block of the finite scans. A block is folded to its largest
/// magnitude without an early exit, which vectorizes to one mask and one
/// unsigned max per four values; a block with a bad value ends the scan.
const SCAN_BLOCK: usize = 1024;

/// [`all_finite`] over native-endian f32 bytes, read in place. A trailing
/// partial value (a length not a multiple of four) is not inspected; every
/// caller passes a whole f32 window.
pub(crate) fn bytes_all_finite(bytes: &[u8]) -> bool {
    let top = |chunk: &[u8]| {
        chunk.as_chunks::<4>().0.iter().fold(0u32, |top, word| {
            top.max(u32::from_ne_bytes(*word) & MAGNITUDE)
        })
    };
    let (blocks, rest) = bytes.as_chunks::<{ 4 * SCAN_BLOCK }>();
    blocks.iter().all(|block| top(block) < NON_FINITE) && top(rest) < NON_FINITE
}

/// Bytes per task of [`check_f32s`]' parallel scan.
const PAR_SCAN_BYTES: usize = 4 << 20;

/// [`check_f32`] on every tensor of `ts`, returning each one's values as
/// native-endian words read in place.
///
/// The layouts are checked in argument order ([`f32_layouts`]) and the NaN
/// scans then run as one parallel pass over fixed [`PAR_SCAN_BYTES`]
/// blocks. The result is the error checking each operand in turn would give.
pub(crate) fn check_f32s<'t>(
    op: &'static str,
    exec: Exec<'_>,
    ts: &[&'t Tensor],
) -> Result<Vec<&'t [[u8; 4]]>, OjasError> {
    let words = f32_layouts(op, exec, ts)?;
    let windows: Vec<&[u8]> = words.iter().map(|w| w.as_flattened()).collect();
    if !windows_finite(exec, &windows)? {
        return Err(nonfinite(op));
    }
    Ok(words)
}

/// The layout checks of [`check_f32`] on every tensor of `ts`, in argument
/// order, without the NaN scan: for an op whose compute pass reads every
/// value and refuses a non-finite one itself.
///
/// A layout refusal of operand `i` is reported only after operands `0..i`
/// are scanned, so a NaN in an earlier operand still outranks it, as
/// checking each operand in turn would give. Every later refusal such an op
/// makes before its compute pass goes through [`nonfinite_first`].
pub(crate) fn f32_layouts<'t>(
    op: &'static str,
    exec: Exec<'_>,
    ts: &[&'t Tensor],
) -> Result<Vec<&'t [[u8; 4]]>, OjasError> {
    let mut words = Vec::with_capacity(ts.len());
    for (i, t) in ts.iter().enumerate() {
        match f32_words(op, t) {
            Ok(w) => words.push(w),
            Err(err) => return Err(nonfinite_first(op, exec, &ts[..i], err)),
        }
    }
    Ok(words)
}

/// `err`, unless one of `ts` holds a NaN or an infinity: then
/// [`OjasError::NonFinite`], the error the NaN scan would have given first.
/// An op that folds its NaN scan into its compute pass sends every refusal
/// it makes before that pass through here, so the observable order is the
/// scan-first order.
pub(crate) fn nonfinite_first(
    op: &'static str,
    exec: Exec<'_>,
    ts: &[&Tensor],
    err: OjasError,
) -> OjasError {
    match check_f32s(op, exec, ts) {
        Ok(_) => err,
        Err(scan) => scan,
    }
}

/// [`check_f32s`] for one tensor.
pub(crate) fn f32_checked<'t>(
    op: &'static str,
    exec: Exec<'_>,
    t: &'t Tensor,
) -> Result<&'t [[u8; 4]], OjasError> {
    check_f32s(op, exec, &[t])?
        .pop()
        .ok_or_else(|| shape(op, "input count changed while checking"))
}

/// The values of a contiguous host `F32` tensor as native-endian words,
/// read in place, after the layout checks of [`check_f32`] but without its
/// scan: for a tensor already scanned in this call.
pub(crate) fn f32_words<'t>(op: &'static str, t: &'t Tensor) -> Result<&'t [[u8; 4]], OjasError> {
    let (_, bytes) = check_layout(op, t, DType::F32)?;
    as_words(op, bytes)
}

/// The values of a contiguous host `U32` tensor as native-endian words, read
/// in place after [`check_u32`]'s checks.
pub(crate) fn u32_words<'t>(op: &'static str, t: &'t Tensor) -> Result<&'t [[u8; 4]], OjasError> {
    let (_, bytes) = check_layout(op, t, DType::U32)?;
    as_words(op, bytes)
}

fn as_words<'b>(op: &'static str, bytes: &'b [u8]) -> Result<&'b [[u8; 4]], OjasError> {
    match bytes.as_chunks::<4>() {
        (words, []) => Ok(words),
        _ => Err(shape(op, "window is not a whole number of 4-byte values")),
    }
}

/// [`bytes_all_finite`] over every window, in [`PAR_SCAN_BYTES`] blocks on
/// the pool's threads.
fn windows_finite(exec: Exec<'_>, windows: &[&[u8]]) -> Result<bool, OjasError> {
    let blocks: Vec<&[u8]> = windows
        .iter()
        .flat_map(|w| w.chunks(PAR_SCAN_BYTES))
        .collect();
    let finite = scoped::map(exec, blocks.len(), |i| Ok(bytes_all_finite(blocks[i])))?;
    Ok(finite.into_iter().all(|ok| ok))
}

/// Dtype, layout, non-emptiness and placement. Returns the element count
/// and the contiguous window, borrowed, not copied, so device memory is a
/// Placement error and not a budget refusal under a tight cap.
fn check_layout<'t>(
    op: &'static str,
    t: &'t Tensor,
    dtype: DType,
) -> Result<(usize, &'t [u8]), OjasError> {
    if t.dtype() != dtype {
        return Err(OjasError::Dtype {
            op,
            expected: dtype,
            got: t.dtype(),
        });
    }
    if !t.is_contiguous()? {
        return Err(shape(op, "non-contiguous view is not supported"));
    }
    let n = t.num_elements()?;
    if n == 0 || t.shape().contains(&0) {
        return Err(shape(op, "empty tensor"));
    }
    Ok((n, t.contiguous_bytes()?))
}

pub(crate) fn product(op: &'static str, dims: &[usize]) -> Result<usize, OjasError> {
    let mut n = 1usize;
    for &dim in dims {
        n = n.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "shape product overflows".to_string(),
        })?;
    }
    Ok(n)
}

pub(crate) fn payload_bytes(op: &'static str, elements: usize) -> Result<u64, OjasError> {
    let bytes = elements
        .checked_mul(DType::F32.size())
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "byte length overflows".to_string(),
        })?;
    u64::try_from(bytes).map_err(|_| OjasError::OutOfRange {
        op,
        detail: "byte length does not fit in u64".to_string(),
    })
}

/// Charge `bytes` against the budget for the duration of the guard.
///
/// In-place steps do not allocate a second tensor, but they build their new
/// values in temporary `Vec`s on the process heap before writing them. The
/// guard covers the most of those that are live at once (optim.rs
/// `adamw_scratch`, `muon_scratch`) and is released before returning, so a
/// cap without that room refuses the step instead of writing.
pub(crate) fn headroom(
    op: &'static str,
    budget: &Budget,
    bytes: u64,
) -> Result<Reservation, OjasError> {
    let _ = op;
    budget.try_reserve(bytes)
}

/// Charge `elements` f32 values and keep the charge until the guard drops.
///
/// Hold the guard across the scratch `Vec`. Dropping it in this function,
/// before that `Vec` exists, lets a second caller reserve the same bytes.
/// An output's guard travels with it in [`F32Out`] until [`alloc_out`] has
/// copied it into a tensor.
pub(crate) fn room_for(
    op: &'static str,
    budget: &Budget,
    elements: usize,
) -> Result<Reservation, OjasError> {
    let _ = op;
    budget.try_reserve(payload_bytes(op, elements)?)
}

/// Prove every in-place target accepts a write before any target changes.
///
/// `targets` pairs each output tensor with the values it holds now; only
/// their length is used. [`Tensor::ensure_writable_f32`] runs the checks
/// `write_f32` would (dtype, length, contiguous host view, sole ownership)
/// without writing, so a target whose storage is shared with a clone or a
/// view refuses here exactly as the real write would. The caller holds
/// `&mut` on every target for the whole step, so no new handle to them can
/// appear between this check and the real writes, which then cannot fail.
pub(crate) fn claim_writable(targets: &mut [(&mut Tensor, &[f32])]) -> Result<(), OjasError> {
    for (tensor, current) in targets.iter_mut() {
        tensor.ensure_writable_f32(current.len())?;
    }
    Ok(())
}

pub(crate) fn alloc_f32(
    op: &'static str,
    budget: &Budget,
    data: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    if !all_finite(data) {
        return Err(nonfinite(op));
    }
    Tensor::from_f32(data, shape, budget)
}

/// A computed f32 result and the charge for its buffer: [`F32V`]'s
/// counterpart for outputs. The charge is taken before the buffer is
/// allocated and travels with it.
pub(crate) struct F32Out {
    pub data: Vec<f32>,
    pub charge: Reservation,
}

/// [`alloc_f32`] for a charged result. The tensor is a copy, so for a moment
/// the result is on the heap twice; `out.charge` stays held until the tensor
/// exists and the buffer is freed, and both copies are charged.
pub(crate) fn alloc_out(
    op: &'static str,
    budget: &Budget,
    out: F32Out,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let F32Out { data, charge } = out;
    let tensor = alloc_f32(op, budget, &data, shape);
    drop(data);
    drop(charge);
    tensor
}

/// No NaN or infinity in `data`, scanned in [`SCAN_BLOCK`] blocks.
pub(crate) fn all_finite(data: &[f32]) -> bool {
    let top = |chunk: &[f32]| {
        chunk
            .iter()
            .fold(0u32, |top, value| top.max(value.to_bits() & MAGNITUDE))
    };
    let (blocks, rest) = data.as_chunks::<SCAN_BLOCK>();
    blocks.iter().all(|block| top(block) < NON_FINITE) && top(rest) < NON_FINITE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every non-finite class is caught at every position (first, inside a
    /// whole block, in the tail past the last block); every finite edge value
    /// passes. The f32 and in-place byte scans agree.
    #[test]
    fn finite_scans_catch_every_nonfinite_at_every_position() {
        let bad = [
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffc0_0000),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let good = [
            0.0,
            -0.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            1.0,
        ];
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>();
        for len in [1usize, 7, 255, 256, 257, 1023, 1024, 1025, 4100] {
            let mut data: Vec<f32> = (0..len).map(|i| good[i % good.len()]).collect();
            assert!(all_finite(&data), "len {len}");
            assert!(bytes_all_finite(&bytes(&data)), "len {len}");
            for at in [0, len / 2, len - 1] {
                for &value in &bad {
                    let keep = data[at];
                    data[at] = value;
                    assert!(
                        !all_finite(&data),
                        "len {len} at {at} {:#x}",
                        value.to_bits()
                    );
                    assert!(
                        !bytes_all_finite(&bytes(&data)),
                        "bytes len {len} at {at} {:#x}",
                        value.to_bits()
                    );
                    data[at] = keep;
                }
            }
        }
        assert!(all_finite(&[]));
        assert!(bytes_all_finite(&[]));
    }

    #[test]
    fn room_for_keeps_the_charge_until_the_guard_drops() {
        let budget = Budget::new(100);
        let guard = room_for("room_for", &budget, 10).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 40);
        assert!(matches!(
            budget.try_reserve(80),
            Err(OjasError::CapacityExceeded { .. })
        ));
        drop(guard);
        assert_eq!(budget.live_bytes().unwrap(), 0);
        assert!(budget.try_reserve(100).is_ok());
    }
}
