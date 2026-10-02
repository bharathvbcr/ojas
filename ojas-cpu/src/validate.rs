use std::sync::Arc;

use ojas_core::{Budget, DType, OjasError, Reservation, Scratch, Tensor};

/// The serial finite scan has one owner, ojas-core; every scan here, and
/// every scan passed to [`Tensor::all_finite_cached`], applies it.
pub(crate) use ojas_core::f32_all_finite as all_finite;

use crate::pool::scoped;
use crate::pool::Exec;

pub(crate) fn shape(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op,
        detail: detail.into(),
    }
}

pub(crate) fn nonfinite(op: &'static str) -> OjasError {
    OjasError::NonFinite { op }
}

/// The layout checks of a host `U32` tensor (ids and targets), read in
/// place.
pub(crate) fn check_u32(op: &'static str, t: &Tensor) -> Result<usize, OjasError> {
    Ok(u32_values(op, t)?.len())
}

/// Every check of a host `F32` operand: dtype, layout, emptiness,
/// placement, then the NaN and infinity scan, reading the values in place.
/// Operands are never copied or charged. The scan is skipped when the
/// storage is already known finite ([`Tensor::all_finite_cached`]).
pub(crate) fn check_f32(op: &'static str, t: &Tensor) -> Result<usize, OjasError> {
    let values = f32_values(op, t)?;
    if !t.all_finite_cached(|w| Ok(all_finite(w)))? {
        return Err(nonfinite(op));
    }
    Ok(values.len())
}

/// Values per task of [`check_f32s`]' parallel scan (4 MiB).
const PAR_SCAN_VALUES: usize = 1 << 20;

/// [`check_f32`] on every tensor of `ts`, returning each one's values, read
/// in place.
///
/// The layouts are checked in argument order ([`f32_layouts`]) and then
/// each operand's NaN scan, in fixed [`PAR_SCAN_VALUES`] blocks on the
/// pool, unless its storage is already known finite
/// ([`Tensor::all_finite_cached`]). The result is the error checking each
/// operand in turn would give.
pub(crate) fn check_f32s<'t>(
    op: &'static str,
    exec: Exec<'_>,
    ts: &[&'t Tensor],
) -> Result<Vec<&'t [f32]>, OjasError> {
    let values = f32_layouts(op, exec, ts)?;
    for t in ts {
        if !t.all_finite_cached(|w| window_finite(exec, w))? {
            return Err(nonfinite(op));
        }
    }
    Ok(values)
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
) -> Result<Vec<&'t [f32]>, OjasError> {
    let mut values = Vec::with_capacity(ts.len());
    for (i, t) in ts.iter().enumerate() {
        match f32_values(op, t) {
            Ok(v) => values.push(v),
            Err(err) => return Err(nonfinite_first(op, exec, &ts[..i], err)),
        }
    }
    Ok(values)
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

/// [`check_f32s`] for a fixed operand count, as an array: every operand's
/// checks and NaN scan in argument order, then each one's values read in
/// place. Nothing is copied or charged.
pub(crate) fn f32_operands<'t, const N: usize>(
    op: &'static str,
    exec: Exec<'_>,
    ts: [&'t Tensor; N],
) -> Result<[&'t [f32]; N], OjasError> {
    check_f32s(op, exec, &ts)?
        .try_into()
        .map_err(|_| shape(op, "input count changed while checking"))
}

/// A checked host `F32` operand that the pool's `'static` tasks read in
/// place: one clone of the tensor (its shape and strides, and one more
/// reference to its storage), shared by every task through an `Arc`. No
/// value is copied or charged.
///
/// While the clone lives the storage has more than one owner, and every
/// host mutation needs sole ownership ([`Tensor::f32_slice_mut`]), so the
/// values cannot change under a task. `Pool::run` waits for every task and
/// frees what they captured before it returns, after an error or a panic as
/// after success (pool.rs `captured_buffers_are_freed_before_run_returns`),
/// so no clone outlives the op that made it and a later in-place step on
/// the same tensor is not refused. A `Shared` never leaves a kernel: kernels
/// return plain values.
///
/// Sharing checks only the layout; the op's NaN scan is its entry check
/// ([`shared_operands`], or [`f32_operands`] over every operand when only
/// some are shared).
///
/// [`crate::gemm::Mat`] borrows instead, through [`scoped`] threads, because
/// its operands include buffers that are not tensors.
#[derive(Clone)]
pub(crate) struct Shared {
    op: &'static str,
    t: Arc<Tensor>,
}

impl Shared {
    /// `t` shared with the pool's tasks after its layout checks.
    pub(crate) fn new(op: &'static str, t: &Tensor) -> Result<Self, OjasError> {
        f32_values(op, t)?;
        Ok(Self {
            op,
            t: Arc::new(t.clone()),
        })
    }

    /// The values, read in place. The layout was checked when this was
    /// made and cannot change, so this repeats checks that pass; a refusal
    /// is still returned, never assumed away.
    pub(crate) fn values(&self) -> Result<&[f32], OjasError> {
        f32_values(self.op, &self.t)
    }
}

/// [`f32_operands`] as [`Shared`] operands, for kernels whose tasks run on
/// the pool: the same checks and NaN scan, in argument order.
pub(crate) fn shared_operands<const N: usize>(
    op: &'static str,
    exec: Exec<'_>,
    ts: [&Tensor; N],
) -> Result<[Shared; N], OjasError> {
    check_f32s(op, exec, &ts)?;
    let shared: Vec<Shared> = ts
        .iter()
        .map(|t| Shared::new(op, t))
        .collect::<Result<_, _>>()?;
    shared
        .try_into()
        .map_err(|_| shape(op, "input count changed while sharing"))
}

/// [`check_f32s`] for one tensor.
pub(crate) fn f32_checked<'t>(
    op: &'static str,
    exec: Exec<'_>,
    t: &'t Tensor,
) -> Result<&'t [f32], OjasError> {
    check_f32s(op, exec, &[t])?
        .pop()
        .ok_or_else(|| shape(op, "input count changed while checking"))
}

/// The values of a contiguous host `F32` tensor, read in place, after the
/// layout checks of [`check_f32`] but without its scan: for a tensor already
/// scanned in this call.
pub(crate) fn f32_values<'t>(op: &'static str, t: &'t Tensor) -> Result<&'t [f32], OjasError> {
    check_layout(op, t, DType::F32)?;
    t.f32_slice()
}

/// The values of a contiguous host `U32` tensor, read in place after
/// [`check_u32`]'s checks.
pub(crate) fn u32_values<'t>(op: &'static str, t: &'t Tensor) -> Result<&'t [u32], OjasError> {
    check_layout(op, t, DType::U32)?;
    t.u32_slice()
}

/// [`all_finite`] over `window`, in [`PAR_SCAN_VALUES`] blocks on the
/// pool's threads.
fn window_finite(exec: Exec<'_>, window: &[f32]) -> Result<bool, OjasError> {
    let blocks: Vec<&[f32]> = window.chunks(PAR_SCAN_VALUES).collect();
    let finite = scoped::map(exec, blocks.len(), |i| Ok(all_finite(blocks[i])))?;
    Ok(finite.into_iter().all(|ok| ok))
}

/// Dtype, layout and non-emptiness, in that order. The caller then borrows
/// the window ([`Tensor::f32_slice`] or [`Tensor::u32_slice`]), which reports
/// device memory as a Placement error, not a budget refusal under a tight
/// cap.
fn check_layout(op: &'static str, t: &Tensor, dtype: DType) -> Result<(), OjasError> {
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
    Ok(())
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
/// The Muon step does not allocate a second tensor, but it builds its new
/// values in temporary `Vec`s on the process heap before writing them. The
/// guard covers the most of those that are live at once (optim.rs
/// `muon_scratch`) and is released before returning, so a cap without that
/// room refuses the step instead of writing. AdamW stores in place and needs
/// none (optim.rs `adamw_in_place`).
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

/// A computed f32 result and the charge for its buffer. The charge is taken before the buffer is
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

/// A new `F32` tensor of `shape` whose values `fill` writes in place into
/// zeroed memory: charged before it is allocated, moved into the tensor
/// with no copy, and scanned for a NaN or infinity (an overflow is
/// refused). A refusal from `fill` is returned and the memory released. The
/// scan is the tensor's [`Tensor::all_finite_cached`], so the next op that
/// takes it as an operand does not scan it again.
pub(crate) fn fill_out(
    op: &'static str,
    budget: &Budget,
    shape: &[usize],
    fill: impl FnOnce(&mut [f32]) -> Result<(), OjasError>,
) -> Result<Tensor, OjasError> {
    let [out] = fill_outs(op, budget, [shape], |[out]| fill(out))?;
    Ok(out)
}

/// [`fill_out`] for `N` outputs that one pass writes together: every output
/// is charged, in order, before `fill` runs, and each is scanned and
/// recorded finite after.
pub(crate) fn fill_outs<const N: usize>(
    op: &'static str,
    budget: &Budget,
    shapes: [&[usize]; N],
    fill: impl FnOnce([&mut [f32]; N]) -> Result<(), OjasError>,
) -> Result<[Tensor; N], OjasError> {
    let mut outs = Vec::with_capacity(N);
    for shape in shapes {
        outs.push(Scratch::<f32>::try_alloc(product(op, shape)?, budget)?);
    }
    let mut outs: [Scratch<f32>; N] = outs
        .try_into()
        .map_err(|_| shape(op, "output count changed while charging"))?;
    fill(outs.each_mut().map(Scratch::as_mut_slice))?;
    let mut tensors = Vec::with_capacity(N);
    for (out, shape) in outs.into_iter().zip(shapes) {
        let out = Tensor::from_scratch(out, shape)?;
        if !out.all_finite_cached(|w| Ok(all_finite(w)))? {
            return Err(nonfinite(op));
        }
        tensors.push(out);
    }
    tensors
        .try_into()
        .map_err(|_| shape(op, "output count changed while filling"))
}

#[cfg(test)]
mod tests {
    use super::*;

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
