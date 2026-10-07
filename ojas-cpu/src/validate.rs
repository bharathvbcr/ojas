use std::ops::Range;
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

/// Values per task of the parallel finite scan, `window_finite` (4 MiB).
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

/// A checked matmul-class operand's `f32` values: an `F32` tensor's read in
/// place, or a `Bf16` tensor's widened exactly (`bits << 16`) into charged
/// scratch that lives as long as this value.
pub(crate) enum Wide<'t> {
    Read(&'t [f32]),
    Widened(Scratch<f32>),
}

impl Wide<'_> {
    pub(crate) fn values(&self) -> &[f32] {
        match self {
            Wide::Read(values) => values,
            Wide::Widened(scratch) => scratch.as_slice(),
        }
    }
}

/// Values per block of the bf16 finite scan and of the widening pass.
const WIDEN_BLOCK: usize = 1 << 17;

/// The operands of a matmul-class op ([`ojas_core::Backend::bf16_operands`]),
/// each `F32` or `Bf16`.
///
/// With no `Bf16` operand this is [`f32_operands`]. Otherwise every
/// operand's layout is checked in argument order, then each one's NaN and
/// infinity scan in argument order (a `Bf16` operand's on its bits, before
/// anything is charged), then each `Bf16` operand is widened into scratch
/// charged to `budget`. Widening is exact, so the kernel sees the values a
/// bf16-rounded `F32` operand would hold and gives the same bits.
pub(crate) fn matmul_operands<'t, const N: usize>(
    op: &'static str,
    exec: Exec<'_>,
    budget: &Budget,
    ts: [&'t Tensor; N],
) -> Result<[Wide<'t>; N], OjasError> {
    if ts.iter().all(|t| t.dtype() != DType::Bf16) {
        return Ok(f32_operands(op, exec, ts)?.map(Wide::Read));
    }
    for t in ts {
        check_layout(op, t, matmul_dtype(t))?;
    }
    for t in ts {
        let finite = match t.dtype() {
            DType::Bf16 => bf16_window_finite(exec, t.bf16_slice()?)?,
            _ => t.all_finite_cached(|w| window_finite(exec, w))?,
        };
        if !finite {
            return Err(nonfinite(op));
        }
    }
    let mut out = Vec::with_capacity(N);
    for t in ts {
        out.push(match t.dtype() {
            DType::Bf16 => Wide::Widened(widen_bf16(exec, budget, t.bf16_slice()?)?),
            _ => Wide::Read(t.f32_slice()?),
        });
    }
    out.try_into()
        .map_err(|_| shape(op, "input count changed while checking"))
}

/// `Bf16` for a bf16 tensor, else `F32`, the dtype a matmul operand is
/// checked against.
fn matmul_dtype(t: &Tensor) -> DType {
    match t.dtype() {
        DType::Bf16 => DType::Bf16,
        _ => DType::F32,
    }
}

/// No bf16 NaN or infinity (all eight exponent bits set) in `bits`, in
/// [`WIDEN_BLOCK`] blocks on scoped threads.
fn bf16_window_finite(exec: Exec<'_>, bits: &[u16]) -> Result<bool, OjasError> {
    const EXPONENT: u16 = 0x7f80;
    let blocks: Vec<&[u16]> = bits.chunks(WIDEN_BLOCK).collect();
    let finite = scoped::map(exec, blocks.len(), |i| {
        Ok(blocks[i].iter().all(|b| b & EXPONENT != EXPONENT))
    })?;
    Ok(finite.into_iter().all(|ok| ok))
}

/// `bits` widened to `f32` in new scratch charged to `budget`.
fn widen_bf16(exec: Exec<'_>, budget: &Budget, bits: &[u16]) -> Result<Scratch<f32>, OjasError> {
    let mut wide = Scratch::<f32>::try_alloc(bits.len(), budget)?;
    scoped::fill(exec, wide.as_mut_slice(), WIDEN_BLOCK, |i, chunk| {
        let from = &bits[i * WIDEN_BLOCK..i * WIDEN_BLOCK + chunk.len()];
        for (dst, &b) in chunk.iter_mut().zip(from) {
            *dst = ojas_core::bf16_to_f32(b);
        }
        Ok(())
    })?;
    Ok(wide)
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
/// ([`f32_operands`] over every operand).
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

/// [`all_finite`] over `window`, in [`PAR_SCAN_VALUES`] blocks on scoped
/// threads; one block runs on the calling thread with no spawn. Operands
/// ([`check_f32s`]) and new outputs ([`fill_outs`]) are scanned with it.
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
/// Outputs are charged by [`fill_out`] / [`fill_outs`] instead.
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

/// A new `F32` tensor of `shape` whose values `fill` writes in place into
/// zeroed memory: charged before it is allocated, moved into the tensor
/// with no copy, and scanned for a NaN or infinity (an overflow is
/// refused). A refusal from `fill` is returned and the memory released. The
/// scan is the operand scan, `window_finite`, recorded as the tensor's
/// [`Tensor::all_finite_cached`], so the next op that takes it as an
/// operand does not scan it again.
pub(crate) fn fill_out(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    shape: &[usize],
    fill: impl FnOnce(&mut [f32]) -> Result<(), OjasError>,
) -> Result<Tensor, OjasError> {
    let [out] = fill_outs(op, budget, exec, [shape], |[out]| fill(out))?;
    Ok(out)
}

/// [`fill_out`] for `N` outputs that one pass writes together: every output
/// is charged, in order, before `fill` runs, and each is scanned and
/// recorded finite after.
pub(crate) fn fill_outs<const N: usize>(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    shapes: [&[usize]; N],
    fill: impl FnOnce([&mut [f32]; N]) -> Result<(), OjasError>,
) -> Result<[Tensor; N], OjasError> {
    let mut outs = charge_outs(op, budget, shapes)?;
    fill(outs.each_mut().map(Scratch::as_mut_slice))?;
    into_tensors(op, outs, shapes, |w| window_finite(exec, w))
}

/// [`fill_outs`] for a pass split into contiguous pieces: items `0..len`
/// are cut as [`scoped::chunks_into_n`] cuts them (`widths`, `min_chunk`),
/// and `fill` writes each piece's values of every output. Each piece is then
/// scanned for a NaN or infinity on the thread that wrote it, while its
/// values are still in that core's cache, and no second pass over the
/// outputs follows: the pieces cover every output, so their verdicts
/// together are each output's [`Tensor::all_finite_cached`]. Returns the
/// outputs and `fill`'s results in piece order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_outs_chunked<const N: usize, R, F>(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    shapes: [&[usize]; N],
    len: usize,
    widths: [usize; N],
    min_chunk: usize,
    fill: F,
) -> Result<([Tensor; N], Vec<R>), OjasError>
where
    R: Send,
    F: Fn(Range<usize>, [&mut [f32]; N]) -> Result<R, OjasError> + Sync,
{
    fill_outs_chunked_trusted(
        op,
        budget,
        exec,
        shapes,
        len,
        widths,
        min_chunk,
        |range, mut parts| {
            let result = fill(range, parts.each_mut().map(|part| &mut **part))?;
            Ok((result, parts.iter().all(|part| all_finite(part))))
        },
    )
}

/// [`fill_outs_chunked`] when `fill` already tested every value it stored.
///
/// The bool is that piece's verdict: true only if every stored element was
/// finite. The written piece is not read again. A false verdict from any
/// piece refuses the op and drops the charged outputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_outs_chunked_trusted<const N: usize, R, F>(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    shapes: [&[usize]; N],
    len: usize,
    widths: [usize; N],
    min_chunk: usize,
    fill: F,
) -> Result<([Tensor; N], Vec<R>), OjasError>
where
    R: Send,
    F: Fn(Range<usize>, [&mut [f32]; N]) -> Result<(R, bool), OjasError> + Sync,
{
    let mut outs = charge_outs(op, budget, shapes)?;
    let pieces = scoped::chunks_into_n(
        exec,
        outs.each_mut().map(Scratch::as_mut_slice),
        len,
        widths,
        min_chunk,
        |range, mut parts| fill(range, parts.each_mut().map(|part| &mut **part)),
    )?;
    let finite = pieces.iter().all(|(_, finite)| *finite);
    if !finite {
        return Err(nonfinite(op));
    }
    let tensors = into_tensors(op, outs, shapes, |_| Ok(true))?;
    Ok((tensors, pieces.into_iter().map(|(r, _)| r).collect()))
}

/// A new tensor of `shape`, `[rows, width]` values, filled in chunks of
/// whole rows of about [`crate::pool::ROW_MIN_ELEMS`] values as
/// [`scoped::rows_into`] cuts them, `fill` writing rows `range` into
/// `part`, and each chunk scanned on the thread that wrote it
/// ([`fill_outs_chunked`]). A `shape` whose element count is not
/// `rows * width` is refused before anything is charged; its layout is the
/// caller's (`[batch, time, dim]` with `rows = batch * time`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_rows<F>(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    shape: &[usize],
    rows: usize,
    width: usize,
    fill: F,
) -> Result<Tensor, OjasError>
where
    F: Fn(Range<usize>, &mut [f32]) -> Result<(), OjasError> + Sync,
{
    if product(op, shape)? != product(op, &[rows, width])? {
        return Err(self::shape(
            op,
            format!("output {shape:?} does not hold {rows} rows of {width}"),
        ));
    }
    let ([out], _) = fill_outs_chunked(
        op,
        budget,
        exec,
        [shape],
        rows,
        [width],
        scoped::min_rows(width),
        |range, [part]| fill(range, part),
    )?;
    Ok(out)
}

/// One zeroed output per shape, each charged before it is allocated, in
/// order.
fn charge_outs<const N: usize>(
    op: &'static str,
    budget: &Budget,
    shapes: [&[usize]; N],
) -> Result<[Scratch<f32>; N], OjasError> {
    let mut outs = Vec::with_capacity(N);
    for shape in shapes {
        outs.push(Scratch::<f32>::try_alloc(product(op, shape)?, budget)?);
    }
    outs.try_into()
        .map_err(|_| shape(op, "output count changed while charging"))
}

/// One filled scratch, scanned with [`window_finite`] and recorded finite
/// only when that scan passes.
pub(crate) fn scanned_f32(
    op: &'static str,
    exec: Exec<'_>,
    scratch: Scratch<f32>,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let [tensor] = into_tensors(op, [scratch], [shape], |w| window_finite(exec, w))?;
    Ok(tensor)
}

/// One filled scratch whose every element was already checked finite by the
/// writer. Recorded finite with no second pass.
pub(crate) fn trusted_finite_f32(
    op: &'static str,
    scratch: Scratch<f32>,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let [tensor] = into_tensors(op, [scratch], [shape], |_| Ok(true))?;
    Ok(tensor)
}

/// The filled outputs as tensors, each refused if `scan` finds a NaN or an
/// infinity in it and otherwise recorded finite.
fn into_tensors<const N: usize>(
    op: &'static str,
    outs: [Scratch<f32>; N],
    shapes: [&[usize]; N],
    scan: impl Fn(&[f32]) -> Result<bool, OjasError>,
) -> Result<[Tensor; N], OjasError> {
    let mut tensors = Vec::with_capacity(N);
    for (out, shape) in outs.into_iter().zip(shapes) {
        let out = Tensor::from_scratch(out, shape)?;
        if !out.all_finite_cached(&scan)? {
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

    /// Every piece's scan counts: a NaN written into any one piece, of
    /// either output, is refused with every charge released, whatever the
    /// thread count; with none, both outputs are recorded finite, so the next
    /// op's scan never runs.
    #[test]
    fn fill_outs_chunked_refuses_a_nan_in_any_piece_and_records_finite_otherwise() {
        use crate::pool::Pool;
        use ojas_core::Numerics;
        let n = 1000;
        for threads in [1usize, 3, 6] {
            let pool = Arc::new(Pool::new(threads).unwrap());
            let exec = Exec {
                pool: &pool,
                numerics: Numerics::Fast,
            };
            let budget = Budget::new(1 << 20);
            let run = |poison: Option<(usize, usize)>| {
                fill_outs_chunked(
                    "t",
                    &budget,
                    exec,
                    [&[n][..], &[n, 2]],
                    n,
                    [1, 2],
                    10,
                    |range, [x, y]| {
                        for (k, i) in range.clone().enumerate() {
                            x[k] = i as f32;
                            y[2 * k] = 1.0;
                            y[2 * k + 1] = 2.0;
                        }
                        if let Some((out, i)) = poison {
                            if range.contains(&i) {
                                let k = i - range.start;
                                if out == 0 {
                                    x[k] = f32::NAN;
                                } else {
                                    y[2 * k + 1] = f32::INFINITY;
                                }
                            }
                        }
                        Ok(range.len())
                    },
                )
            };
            for poison in [(0, 0), (0, n - 1), (1, n / 2), (1, n - 1)] {
                let got = run(Some(poison));
                assert!(
                    matches!(got, Err(OjasError::NonFinite { .. })),
                    "{threads} threads, {poison:?}: {got:?}"
                );
                assert_eq!(budget.live_bytes().unwrap(), 0);
            }
            let ([x, y], lens) = run(None).unwrap();
            assert_eq!(lens.iter().sum::<usize>(), n);
            assert_eq!(x.f32_slice().unwrap()[n - 1], (n - 1) as f32);
            assert_eq!(y.f32_slice().unwrap()[2 * n - 1], 2.0);
            for t in [&x, &y] {
                assert!(t
                    .all_finite_cached(|_| panic!("recorded finite, so not scanned again"))
                    .unwrap());
            }
        }
    }

    /// A shape whose element count is not `rows * width` is refused before
    /// anything is charged or written, including one row short, the case
    /// [`scoped::chunks_into_n`] itself accepts for block-shaped items. The
    /// layout of `shape` is the caller's: `[4, 3]` holds 3 rows of 4.
    #[test]
    fn fill_rows_refuses_a_shape_that_is_not_rows_of_width() {
        use crate::pool::Pool;
        use ojas_core::Numerics;
        let pool = Arc::new(Pool::new(2).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let budget = Budget::new(1 << 20);
        for bad in [&[3usize, 5][..], &[2, 4], &[11], &[13]] {
            let got = fill_rows("t", &budget, exec, bad, 3, 4, |_, _| {
                panic!("nothing is written for {bad:?}")
            });
            assert!(
                matches!(got, Err(OjasError::Shape { .. })),
                "{bad:?}: {got:?}"
            );
            assert_eq!(budget.live_bytes().unwrap(), 0);
        }
        let ok = fill_rows("t", &budget, exec, &[3, 4], 3, 4, |range, part| {
            part.fill(range.start as f32);
            Ok(())
        })
        .unwrap();
        assert_eq!(ok.f32_slice().unwrap().len(), 12);
    }
}
