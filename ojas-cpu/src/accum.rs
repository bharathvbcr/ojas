//! `acc += grad` in place, for gradient accumulation across micro-batches.

use ojas_core::{accumulate_grad_dims, Budget, OjasError, Tensor};

use crate::pool::Exec;
use crate::validate::{check_f32, fill_out, nonfinite};

/// [`ojas_core::Backend::accumulate_grad`] on the CPU.
///
/// Both operands are read in place. Every sum is checked finite before
/// anything is written or charged: a NaN in `grad`, or a sum that
/// overflows, is `NonFinite` with `acc` untouched. A uniquely owned `acc`
/// is then added to in place, with no buffer and no charge (since
/// 2026-10-02; before, the sum was built in a charged buffer and copied in).
/// An `acc` whose allocation is shared (a clone or a view of it exists)
/// cannot be written, so it is replaced by a new tensor holding the sum,
/// written straight into the tensor ([`fill_out`]), as the trait default
/// does; the other handles keep the old values.
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
    let len = check_f32(op, acc)?;
    check_f32(op, grad)?;
    let g = grad.f32_slice()?;
    if !acc
        .f32_slice()?
        .iter()
        .zip(g)
        .all(|(a, g)| (a + g).is_finite())
    {
        return Err(nonfinite(op));
    }
    match acc.ensure_writable_f32(len) {
        Ok(()) => {
            for (a, g) in acc.f32_slice_mut()?.iter_mut().zip(g) {
                *a += g;
            }
            Ok(())
        }
        // `check_f32` already proved `acc` is a contiguous host F32 tensor of
        // `len` elements, so the one `Shape` refusal left is shared storage.
        Err(OjasError::Shape { .. }) => {
            let a = acc.f32_slice()?;
            let sum = fill_out(op, budget, exec, acc.shape(), |out| {
                for ((o, &a), &g) in out.iter_mut().zip(a).zip(g) {
                    *o = a + g;
                }
                Ok(())
            })?;
            *acc = sum;
            Ok(())
        }
        Err(err) => Err(err),
    }
}
