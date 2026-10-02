//! `acc += grad` in place, for gradient accumulation across micro-batches.

use ojas_core::{residual_add_forward_dims, Budget, OjasError, Tensor};

use crate::validate::{all_finite, alloc_out, check_f32, nonfinite, room_for, F32Out};

/// `ojas_core::shapes` has no validator for `accumulate_grad`. Its rules are
/// `residual_add_forward_dims`'s (two `F32` tensors of one shape), and every
/// refusal is reported under `op`, as Metal and wgpu report it. Runs before
/// any value is read or anything is charged.
fn accumulate_grad_dims(op: &'static str, acc: &Tensor, grad: &Tensor) -> Result<usize, OjasError> {
    residual_add_forward_dims(acc, grad).map_err(|err| match err {
        OjasError::Shape { detail, .. } => OjasError::Shape { op, detail },
        OjasError::Dtype { expected, got, .. } => OjasError::Dtype { op, expected, got },
        OjasError::OutOfRange { detail, .. } => OjasError::OutOfRange { op, detail },
        OjasError::NonFinite { .. } => OjasError::NonFinite { op },
        other => other,
    })
}

/// [`ojas_core::Backend::accumulate_grad`] on the CPU.
///
/// The sum is built in one charged buffer from `acc`'s values and `grad`'s
/// bytes, read in place, and checked finite before `acc` changes: a NaN in
/// `grad`, or a sum that overflows, is `NonFinite` with `acc` untouched. A
/// uniquely owned `acc` is then overwritten in place. An `acc` whose
/// allocation is shared (a clone or a view of it exists) cannot be written,
/// so it is replaced by a new tensor holding the sum, as the trait default
/// does; the other handles keep the old values.
pub(crate) fn accumulate_grad(
    op: &'static str,
    budget: &Budget,
    acc: &mut Tensor,
    grad: &Tensor,
) -> Result<(), OjasError> {
    accumulate_grad_dims(op, acc, grad)?;
    let len = check_f32(op, acc)?;
    check_f32(op, grad)?;
    let charge = room_for(op, budget, len)?;
    let (a, _) = acc.contiguous_bytes()?.as_chunks::<4>();
    let (g, _) = grad.contiguous_bytes()?.as_chunks::<4>();
    let sum: Vec<f32> = a
        .iter()
        .zip(g)
        .map(|(a, g)| f32::from_ne_bytes(*a) + f32::from_ne_bytes(*g))
        .collect();
    if !all_finite(&sum) {
        return Err(nonfinite(op));
    }
    match acc.ensure_writable_f32(len) {
        Ok(()) => acc.write_f32(&sum),
        // `check_f32` already proved `acc` is a contiguous host F32 tensor of
        // `len` elements, so the one `Shape` refusal left is shared storage.
        Err(OjasError::Shape { .. }) => {
            let shape = acc.shape().to_vec();
            *acc = alloc_out(op, budget, F32Out { data: sum, charge }, &shape)?;
            Ok(())
        }
        Err(err) => Err(err),
    }
}
