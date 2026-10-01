//! Axis permutation: values move, nothing is computed.

use ojas_core::{permute_output_shape, Budget, DType, OjasError, Scratch, Tensor};

use crate::validate::{bytes_all_finite, nonfinite, shape};

const F32_BYTES: usize = 4;

/// `torch.permute(input, dims).contiguous()` for a host f32 tensor.
///
/// The input is read in place from its contiguous window, so no host copy
/// of it is made or charged. The output is a [`Scratch`] charged to
/// `budget` and handed to the tensor without a second copy. Each value is
/// moved as its four bytes, so the output bits are the input bits.
///
/// Refused: a dtype other than f32, device memory (from the window read),
/// a strided view, invalid `dims` (see [`permute_output_shape`]), and a
/// NaN or infinity, as every other CPU op refuses one. Rank 0 and axes of
/// length 0 are accepted.
pub(crate) fn permute(
    op: &'static str,
    budget: &Budget,
    input: &Tensor,
    dims: &[usize],
) -> Result<Tensor, OjasError> {
    if input.dtype() != DType::F32 {
        return Err(OjasError::Dtype {
            op,
            expected: DType::F32,
            got: input.dtype(),
        });
    }
    let in_shape = input.shape();
    let out_shape = permute_output_shape(op, in_shape, dims)?;
    if !input.is_contiguous()? {
        return Err(shape(op, "non-contiguous view is not supported"));
    }
    let src = input.contiguous_bytes()?;
    if !bytes_all_finite(src) {
        return Err(nonfinite(op));
    }
    let mut out = Scratch::<u8>::try_alloc(src.len(), budget)?;
    gather(src, in_shape, dims, &out_shape, out.as_mut_slice());
    Tensor::from_scratch(out, &out_shape, DType::F32)
}

/// Write `dst` in output row-major order. Output axis `i` steps the source
/// by the input stride of axis `dims[i]`, so an odometer over the output
/// index carries the source offset without a divide per element.
fn gather(src: &[u8], in_shape: &[usize], dims: &[usize], out_shape: &[usize], dst: &mut [u8]) {
    if dst.is_empty() {
        return;
    }
    let rank = out_shape.len();
    let mut in_strides = vec![1usize; in_shape.len()];
    for axis in (0..in_shape.len().saturating_sub(1)).rev() {
        in_strides[axis] = in_strides[axis + 1] * in_shape[axis + 1];
    }
    let step: Vec<usize> = dims.iter().map(|&axis| in_strides[axis]).collect();
    let mut index = vec![0usize; rank];
    let mut from = 0usize;
    for out in dst.as_chunks_mut::<F32_BYTES>().0 {
        let at = from * F32_BYTES;
        out.copy_from_slice(&src[at..at + F32_BYTES]);
        // Advance the odometer from the last output axis.
        for axis in (0..rank).rev() {
            index[axis] += 1;
            from += step[axis];
            if index[axis] < out_shape[axis] {
                break;
            }
            from -= step[axis] * out_shape[axis];
            index[axis] = 0;
        }
    }
}
