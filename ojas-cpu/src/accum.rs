//! `acc += grad` in place, for gradient accumulation across micro-batches
//! and for a tape's gradient fan-in.

use ojas_core::{accumulate_grad_dims, Budget, OjasError, Tensor};

use crate::pool::{scoped, Exec};
use crate::validate::{f32_layouts, fill_outs_chunked_trusted, nonfinite};

/// [`ojas_core::Backend::accumulate_grad`] on the CPU.
///
/// Both operands are read in place. Every sum is checked finite before
/// anything is written or charged: a NaN or infinity in either operand, or
/// a sum that overflows, is `NonFinite` with `acc` untouched. That check is
/// one pass over both operands, cut into the pieces [`scoped::pieces`]
/// gives at [`scoped::min_rows`]`(1)` values each and run on the pool: a
/// sum is finite only if both of its operands are, so it is also the
/// operands' NaN scan (before 2026-10-07 that was three serial passes: each
/// operand's scan, then the sums').
///
/// A uniquely owned `acc` is then added to in place, in the same pieces,
/// with no buffer and no charge, and recorded finite (since 2026-10-02;
/// before, the sum was built in a charged buffer and copied in). Writing
/// cannot refuse once it starts: the sums are known finite and a piece that
/// does not panic does not fail. In place there is no second pass that could
/// undo a written piece, so the check runs first rather than inside the add.
/// An `acc` whose allocation is shared (a clone or a view of it exists)
/// cannot be written, so it is replaced by a new tensor holding the sum,
/// written in those pieces straight into the tensor, as the trait default
/// does; the other handles keep the old values. Each sum is one rounding of
/// `acc + grad` wherever it runs, so the bits do not depend on the cut.
pub(crate) fn accumulate_grad(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    acc: &mut Tensor,
    grad: &Tensor,
) -> Result<(), OjasError> {
    // `ojas_core::shapes` owns the rules; it runs before any value is read
    // or anything is charged.
    accumulate_grad_dims(acc, grad)?;
    let min_chunk = scoped::min_rows(1);
    let len = {
        let values = f32_layouts(op, exec, &[&*acc, grad])?;
        let (a, g) = (values[0], values[1]);
        let pieces = scoped::pieces(exec, a.len(), min_chunk);
        let finite = scoped::map(exec, pieces.len(), |i| {
            let range = pieces[i].clone();
            Ok(sums_finite(&a[range.clone()], &g[range]))
        })?;
        if !finite.into_iter().all(|piece| piece) {
            return Err(nonfinite(op));
        }
        a.len()
    };
    let g = grad.f32_slice()?;
    match acc.ensure_writable_f32(len) {
        Ok(()) => {
            scoped::chunks_into(
                exec,
                acc.f32_slice_mut()?,
                len,
                1,
                min_chunk,
                |range, part| {
                    for (a, g) in part.iter_mut().zip(&g[range]) {
                        *a += g;
                    }
                    Ok(())
                },
            )?;
            // Every sum was checked finite above.
            acc.all_finite_cached(|_| Ok(true))?;
            Ok(())
        }
        // `f32_layouts` already proved `acc` is a contiguous host F32 tensor
        // of `len` elements, so the one `Shape` refusal left is shared
        // storage.
        Err(OjasError::Shape { .. }) => {
            let shape = acc.shape().to_vec();
            let a = acc.f32_slice()?;
            let ([sum], _) = fill_outs_chunked_trusted(
                op,
                budget,
                exec,
                [&shape],
                len,
                [1],
                min_chunk,
                |range, [out]| {
                    let sums = out.iter_mut().zip(&a[range.clone()]).zip(&g[range]);
                    for ((o, &a), &g) in sums {
                        *o = a + g;
                    }
                    Ok(((), true))
                },
            )?;
            *acc = sum;
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Every `a[i] + g[i]` is finite. Each block of [`FINITE_BLOCK`] sums is
/// tested without a branch, so the loop vectorises (an element-wise
/// short-circuit does not), and the first non-finite block ends the scan.
fn sums_finite(a: &[f32], g: &[f32]) -> bool {
    a.chunks(FINITE_BLOCK)
        .zip(g.chunks(FINITE_BLOCK))
        .all(|(a, g)| {
            a.iter()
                .zip(g)
                .fold(true, |finite, (a, g)| finite & (a + g).is_finite())
        })
}

/// Sums [`sums_finite`] tests per branch.
const FINITE_BLOCK: usize = 64;

/// [`ojas_core::Backend::scale_grad`] on the CPU: `grad *= scale`.
///
/// One pass in the [`scoped::pieces`] cut, each product one rounding of
/// `value * scale` (the bits of the host loop the tape ran before
/// 2026-10-07, which copied the gradient out, scaled the copy and built a
/// new tensor from it). A uniquely owned `grad` is written in place with no
/// charge and recorded finite; a shared one is replaced by a new tensor
/// holding the products, and the other handles keep the old values. Every
/// product is tested as it is stored: a NaN or infinity in `grad`, or a
/// product that overflows, is `NonFinite`, and an in-place `grad`'s values
/// are then unspecified (the trait's contract: the caller discards it).
pub(crate) fn scale_grad(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    grad: &mut Tensor,
    scale: f32,
) -> Result<(), OjasError> {
    if !scale.is_finite() {
        return Err(nonfinite(op));
    }
    let len = f32_layouts(op, exec, &[&*grad])?[0].len();
    let min_chunk = scoped::min_rows(1);
    match grad.ensure_writable_f32(len) {
        Ok(()) => {
            let finite =
                scoped::chunks_into(exec, grad.f32_slice_mut()?, len, 1, min_chunk, |_, part| {
                    let mut finite = true;
                    for value in part {
                        *value *= scale;
                        finite &= value.is_finite();
                    }
                    Ok(finite)
                })?;
            if !finite.into_iter().all(|piece| piece) {
                return Err(nonfinite(op));
            }
            grad.all_finite_cached(|_| Ok(true))?;
            Ok(())
        }
        Err(OjasError::Shape { .. }) => {
            let shape = grad.shape().to_vec();
            let g = grad.f32_slice()?;
            let ([out], _) = fill_outs_chunked_trusted(
                op,
                budget,
                exec,
                [&shape],
                len,
                [1],
                min_chunk,
                |range, [out]| {
                    let mut finite = true;
                    for (o, &value) in out.iter_mut().zip(&g[range]) {
                        *o = value * scale;
                        finite &= o.is_finite();
                    }
                    Ok(((), finite))
                },
            )?;
            *grad = out;
            Ok(())
        }
        Err(err) => Err(err),
    }
}
