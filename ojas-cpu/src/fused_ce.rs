//! Mean cross-entropy of `x · Wᵀ` and both gradients, without the `[N, V]`
//! logits.
//!
//! The rows of `x` are walked in `chunk.rows` blocks and, inside each, the
//! vocabulary in `chunk.cols` blocks; one `[rows, cols]` logit tile exists at
//! a time. Each tile is the [`crate::gemm`] product `x_c · W_cᵀ`, so Fast and
//! Exact numerics apply as they do to `linear_forward`. A row block makes
//! three passes over the vocabulary, recomputing its tiles each time:
//!
//! 1. the row maximum;
//! 2. `sum exp(l - max)` in ascending vocabulary order, and the target logit;
//! 3. with `want_grad`, `G = (softmax - onehot) / n_valid` formed in the tile
//!    in place, then `grad_x[rows] += G · W_c` and `grad_w[cols] += Gᵀ · x_c`.
//!
//! The pass-1/pass-2 split (rather than an online, rescaled sum) keeps every
//! loss term the f32 sequence `cross_entropy_mean_forward` computes, and the
//! gradient tile is the one `cross_entropy_mean_backward` writes (ignored
//! rows are zero rows, not skipped). The gradient products go through
//! [`gemm_acc`], which continues each output's sum from the value already in
//! the zeroed gradient, so `grad_x` summed over vocabulary chunks and
//! `grad_w` summed over row chunks take the steps one GEMM over all of `V`
//! (or `N`) takes. Under [`ojas_core::Numerics::Exact`] the exact GEMM kernel
//! does not depend on the tiling, so the loss and both gradients equal the
//! composed `linear_forward → cross_entropy_mean_forward/backward →
//! linear_backward` bit for bit, for every chunk. Under Fast they agree to
//! rounding: a whole-call product (Accelerate on macOS) picks its own order.
//!
//! The exponentials and the softmax tile run on the calling thread; the
//! GEMMs use the pool.
//!
//! The valid-row count is global. An all-ignored batch is
//! [`OjasError::NonFinite`], and a target outside the vocabulary that is not
//! `ignore_index` is [`OjasError::OutOfRange`], as in the unfused path.

use ojas_core::{linear_ce_dims, Budget, CeChunk, LinearCe, LinearCeDims, OjasError, Tensor};

use crate::gemm::{gemm, gemm_acc, scratch, Mat};
use crate::pool::Exec;
use crate::validate::{
    all_finite, alloc_f32, alloc_out, check_u32, f32_operands, nonfinite, product, room_for, shape,
    u32_values, F32Out,
};

const OP: &str = "linear_cross_entropy_mean";

/// [`ojas_core::Backend::linear_cross_entropy_mean`] on the CPU.
///
/// Every operand is validated before the first charge, then read in place:
/// the input rows and weight blocks of each tile are views, not copies. The
/// gradient buffers and the tile scratch are each charged before they are
/// allocated; each gradient's charge is held until its tensor copy exists.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_cross_entropy_mean(
    budget: &Budget,
    exec: Exec<'_>,
    input: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    ignore: Option<u32>,
    chunk: CeChunk,
    want_grad: bool,
) -> Result<LinearCe, OjasError> {
    let dims = linear_ce_dims(input, weight, targets, chunk)?;
    check_u32(OP, targets)?;
    let [x, w] = f32_operands(OP, exec, [input, weight])?;
    let t = u32_values(OP, targets)?;
    let (loss, grads) = fused(budget, exec, x, w, t, dims, ignore, chunk, want_grad)?;
    let loss = alloc_f32(OP, budget, &[loss], &[])?;
    let (grad_input, grad_weight) = match grads {
        Some((gx, gw)) => (
            Some(alloc_out(OP, budget, gx, input.shape())?),
            Some(alloc_out(OP, budget, gw, weight.shape())?),
        ),
        None => (None, None),
    };
    Ok(LinearCe {
        loss,
        grad_input,
        grad_weight,
    })
}

/// Valid rows, after checking every non-ignored target is inside the
/// vocabulary. Zero valid rows is `NonFinite` (the mean has no denominator).
fn valid_rows(targets: &[u32], vocab: usize, ignore: Option<u32>) -> Result<u32, OjasError> {
    let mut valid: u32 = 0;
    for (n, &target) in targets.iter().enumerate() {
        if ignore == Some(target) {
            continue;
        }
        if usize::try_from(target)
            .ok()
            .filter(|id| *id < vocab)
            .is_none()
        {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("target {target} at {n} is outside vocab {vocab}"),
            });
        }
        valid = valid.checked_add(1).ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "valid target count overflows".to_string(),
        })?;
    }
    if valid == 0 {
        return Err(nonfinite(OP));
    }
    Ok(valid)
}

/// Floats live at once besides the inputs and the two gradients: one logit
/// tile, the larger of the two [`gemm_acc`] output snapshots (a split
/// product copies its `[rows, d]` or `[cols, d]` output), three per-row
/// statistics, and the largest [`scratch`] of the three GEMMs. The row block
/// of `x` and the vocabulary block of `W` are views of the inputs, not
/// copies. Every block is at most `rows x cols`, so the full-size block
/// bounds the partial ones.
fn tile_scratch(
    exec: Exec<'_>,
    rows: usize,
    cols: usize,
    dim: usize,
    want_grad: bool,
) -> Result<usize, OjasError> {
    let tile = product(OP, &[rows, cols])?;
    let stats = product(OP, &[rows, 3])?;
    let mut gemm_work = scratch(OP, exec, rows, dim, cols)?;
    let mut grad_product = 0;
    if want_grad {
        gemm_work = gemm_work
            .max(scratch(OP, exec, rows, cols, dim)?)
            .max(scratch(OP, exec, cols, rows, dim)?);
        grad_product = product(OP, &[rows, dim])?.max(product(OP, &[cols, dim])?);
    }
    [tile, stats, gemm_work, grad_product]
        .into_iter()
        .try_fold(0usize, |sum, n| sum.checked_add(n))
        .ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "fused cross-entropy scratch length overflows".to_string(),
        })
}

/// Rows `c0..c1` of `W` as their own matrix, and the logit tile `x_c · W_cᵀ`.
/// A non-finite logit is `NonFinite`, as `linear_forward` reports it.
fn logit_tile<'w>(
    exec: Exec<'_>,
    x_block: &Mat,
    w: &'w [f32],
    cols: std::ops::Range<usize>,
    dim: usize,
) -> Result<(Vec<f32>, Mat<'w>), OjasError> {
    let w_block = Mat::row_major(&w[cols.start * dim..cols.end * dim], cols.len(), dim);
    let logits = gemm(OP, exec, x_block, &w_block.t())?;
    if !all_finite(&logits) {
        return Err(nonfinite(OP));
    }
    Ok((logits, w_block))
}

/// The scalar loss, and `(grad_x, grad_w)` with their charges when
/// `want_grad` is set.
#[allow(clippy::too_many_arguments)]
fn fused(
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    w: &[f32],
    targets: &[u32],
    dims: LinearCeDims,
    ignore: Option<u32>,
    chunk: CeChunk,
    want_grad: bool,
) -> Result<(f32, Option<(F32Out, F32Out)>), OjasError> {
    let LinearCeDims {
        rows: n,
        model_dim: d,
        vocab: v,
    } = dims;
    if x.len() != product(OP, &[n, d])? || w.len() != product(OP, &[v, d])? || targets.len() != n {
        return Err(shape(
            OP,
            "fused cross-entropy data length does not match shape",
        ));
    }
    let denom = valid_rows(targets, v, ignore)? as f32;
    let (rows, cols) = (chunk.rows.min(n), chunk.cols.min(v));
    // Each gradient's charge is taken before its buffer exists and leaves
    // with it; the tile charge ends when this function returns.
    let grad_charges = if want_grad {
        Some((room_for(OP, budget, n * d)?, room_for(OP, budget, v * d)?))
    } else {
        None
    };
    let _tiles = room_for(OP, budget, tile_scratch(exec, rows, cols, d, want_grad)?)?;
    let (mut gx, mut gw) = if want_grad {
        (vec![0.0f32; n * d], vec![0.0f32; v * d])
    } else {
        (Vec::new(), Vec::new())
    };
    let vocab_blocks = || (0..v).step_by(cols).map(move |c0| c0..(c0 + cols).min(v));
    let mut total = 0.0f32;
    for r0 in (0..n).step_by(rows) {
        let r1 = (r0 + rows).min(n);
        let r = r1 - r0;
        let x_block = Mat::row_major(&x[r0 * d..r1 * d], r, d);
        let block_targets = &targets[r0..r1];
        let ignored = |i: usize| ignore == Some(block_targets[i]);
        let mut maxes = vec![f32::NEG_INFINITY; r];
        let mut sums = vec![0.0f32; r];
        let mut picked = vec![0.0f32; r];

        for range in vocab_blocks() {
            let (logits, _) = logit_tile(exec, &x_block, w, range.clone(), d)?;
            for (max, row) in maxes.iter_mut().zip(logits.chunks_exact(range.len())) {
                for &value in row {
                    if value > *max {
                        *max = value;
                    }
                }
            }
        }

        for range in vocab_blocks() {
            let (logits, _) = logit_tile(exec, &x_block, w, range.clone(), d)?;
            for (i, row) in logits.chunks_exact(range.len()).enumerate() {
                if ignored(i) {
                    continue;
                }
                for &value in row {
                    let e = (value - maxes[i]).exp();
                    if !e.is_finite() {
                        return Err(nonfinite(OP));
                    }
                    sums[i] += e;
                }
                let class = block_targets[i] as usize;
                if range.contains(&class) {
                    picked[i] = row[class - range.start];
                }
            }
        }
        for i in 0..r {
            if ignored(i) {
                continue;
            }
            let sum = sums[i];
            if !(sum.is_finite() && sum > 0.0) {
                return Err(nonfinite(OP));
            }
            total += maxes[i] + sum.ln() - picked[i];
        }

        if !want_grad {
            continue;
        }
        for range in vocab_blocks() {
            let width = range.len();
            let (mut g, w_block) = logit_tile(exec, &x_block, w, range.clone(), d)?;
            for (i, row) in g.chunks_exact_mut(width).enumerate() {
                if ignored(i) {
                    row.fill(0.0);
                    continue;
                }
                let (max, sum) = (maxes[i], sums[i]);
                for value in row.iter_mut() {
                    let p = (*value - max).exp() / sum;
                    *value = p / denom;
                }
                let class = block_targets[i] as usize;
                if range.contains(&class) {
                    row[class - range.start] -= 1.0 / denom;
                }
            }
            let g = Mat::row_major(&g, r, width);
            gemm_acc(OP, exec, &g, &w_block, &mut gx[r0 * d..r1 * d])?;
            gemm_acc(
                OP,
                exec,
                &g.t(),
                &x_block,
                &mut gw[range.start * d..range.end * d],
            )?;
        }
    }
    let loss = total / denom;
    if !loss.is_finite() {
        return Err(nonfinite(OP));
    }
    let grads = grad_charges.map(|(cx, cw)| {
        (
            F32Out {
                data: gx,
                charge: cx,
            },
            F32Out {
                data: gw,
                charge: cw,
            },
        )
    });
    Ok((loss, grads))
}
