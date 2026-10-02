//! Axis permutation: values move, nothing is computed.

use ojas_core::{permute_output_shape, Budget, DType, OjasError, Scratch, Tensor};

use crate::validate::{all_finite, nonfinite, shape};

/// `torch.permute(input, dims).contiguous()` for a host f32 tensor.
///
/// The input is read in place from its contiguous window, so no host copy
/// of it is made or charged. The output is a [`Scratch`] charged to
/// `budget` and handed to the tensor without a second copy. Values are
/// copied, never computed, so the output bits are the input bits.
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
    let src = input.f32_slice()?;
    if !all_finite(src) {
        return Err(nonfinite(op));
    }
    let mut out = Scratch::<f32>::try_alloc(src.len(), budget)?;
    gather(src, in_shape, dims, &out_shape, out.as_mut_slice());
    Tensor::from_scratch(out, &out_shape)
}

/// Write `dst` in output row-major order. Output axis `i` steps the source
/// by the input stride of axis `dims[i]`, so an odometer over the output
/// index carries the source offset without a divide per element.
///
/// Trailing output axes that are the same input axes in the same order
/// (`dims[i] == i` for every `i >= keep`) are one contiguous run in both
/// tensors, so the odometer walks only the leading `keep` axes and each
/// step copies a whole run: `(0, 2, 1, 3)` on `[B, T, H, D]` copies `D`
/// values at a time, and the identity is one copy.
fn gather(src: &[f32], in_shape: &[usize], dims: &[usize], out_shape: &[usize], dst: &mut [f32]) {
    if dst.is_empty() {
        return;
    }
    let mut in_strides = vec![1usize; in_shape.len()];
    for axis in (0..in_shape.len().saturating_sub(1)).rev() {
        in_strides[axis] = in_strides[axis + 1] * in_shape[axis + 1];
    }
    let mut keep = out_shape.len();
    while keep > 0 && dims[keep - 1] == keep - 1 {
        keep -= 1;
    }
    let run: usize = out_shape[keep..].iter().product::<usize>();
    let step: Vec<usize> = dims[..keep].iter().map(|&axis| in_strides[axis]).collect();
    let mut odometer = Odometer {
        index: vec![0usize; keep],
        from: 0,
        step: &step,
        extent: &out_shape[..keep],
    };
    if run == 1 {
        // One value per step; a run-length copy this short would be a
        // `memcpy` call per value.
        for out in dst.iter_mut() {
            *out = src[odometer.from];
            odometer.advance();
        }
    } else {
        for out in dst.chunks_exact_mut(run) {
            let at = odometer.from;
            out.copy_from_slice(&src[at..at + run]);
            odometer.advance();
        }
    }
}

/// Source offset, in values, of the current output position over the
/// leading output axes, advanced from the last axis.
struct Odometer<'a> {
    index: Vec<usize>,
    from: usize,
    step: &'a [usize],
    extent: &'a [usize],
}

impl Odometer<'_> {
    fn advance(&mut self) {
        for axis in (0..self.index.len()).rev() {
            self.index[axis] += 1;
            self.from += self.step[axis];
            if self.index[axis] < self.extent[axis] {
                return;
            }
            self.from -= self.step[axis] * self.extent[axis];
            self.index[axis] = 0;
        }
    }
}
