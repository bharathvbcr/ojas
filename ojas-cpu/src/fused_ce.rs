//! Mean cross-entropy of `x · Wᵀ` and the gradients asked for, without the
//! `[N, V]` logits.
//!
//! The rows of `x` are walked in `chunk.rows` blocks and, inside each, the
//! vocabulary in `chunk.cols` blocks; one `[rows, cols]` logit tile exists at
//! a time. Each tile is the [`crate::gemm`] product `x_c · W_cᵀ`, so Fast and
//! Exact numerics apply as they do to `linear_forward`. A row block makes
//! three passes over the vocabulary, recomputing its tiles each time when
//! the vocabulary takes more than one block:
//!
//! 1. the row maximum over every class except the target;
//! 2. `sum exp(l - max)` of those classes, in ascending vocabulary order, and
//!    the target logit;
//! 3. with a gradient asked for, `G` formed in the tile in place (each other
//!    class `exp(l - max) * scale / n_valid`, the target the negation of
//!    their sum), then `grad_x[rows] += G · W_c` and/or
//!    `grad_w[cols] += Gᵀ · x_c`, each only when its gradient is asked for
//!    (a frozen tied head takes no `[V, d]` output and no `Gᵀ · x_c`).
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
//! When `chunk.cols` covers the vocabulary there is one tile per row block,
//! so it is computed once and the three passes reuse it. The per-row work of
//! each pass (maximum, exponential sums, softmax tile) is split by rows
//! across scoped threads, each row on one thread in its own order, so the
//! split changes no bit; the GEMMs use the pool.
//!
//! A `Bf16` weight (the tied table held at 2 bytes) is read in place: each
//! vocabulary block is widened exactly into an `f32` block of `chunk.cols *
//! d` values when its tile is made, charged with the tile scratch, and
//! dropped with the tile. The GEMMs then see the values a bf16-exact `F32`
//! weight holds, so the bits are that weight's, and no `f32` copy of the
//! whole table exists.
//!
//! The valid-row count is global. An all-ignored batch is
//! [`OjasError::NonFinite`], and a target outside the vocabulary that is not
//! `ignore_index` is [`OjasError::OutOfRange`], as in the unfused path.

use std::ops::Range;

use ojas_core::{
    excluded_ce, exp_exact, linear_ce_dims, Budget, CeChunk, CeGrads, DType, LinearCe,
    LinearCeDims, OjasError, Tensor,
};

use crate::gemm::{gemm, gemm_acc, scratch, Mat};
use crate::pool::scoped::{self, min_rows};
use crate::pool::Exec;
use crate::validate::{
    all_finite, alloc_f32, check_u32, f32_operands, fill_outs, nonfinite, product, room_for,
    scan_matmul_operand, shape, u32_values, widen_bf16_into, Wide,
};

const OP: &str = "linear_cross_entropy_mean";

/// [`ojas_core::Backend::linear_cross_entropy_mean`] on the CPU.
///
/// Every operand is validated before the first charge, then read in place:
/// the input rows and weight blocks of each tile are views, not copies (a
/// `Bf16` weight's block is widened, see the module docs). The gradients
/// are written straight into their output tensors (charged before they are
/// allocated, never copied), and the tile scratch is charged too.
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_cross_entropy_mean(
    budget: &Budget,
    exec: Exec<'_>,
    input: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    ignore: Option<u32>,
    chunk: CeChunk,
    grads: CeGrads,
) -> Result<LinearCe, OjasError> {
    let dims = linear_ce_dims(input, weight, targets, chunk)?;
    check_u32(OP, targets)?;
    let [x] = f32_operands(OP, exec, [input])?;
    let w = match weight.dtype() {
        DType::Bf16 => {
            scan_matmul_operand(OP, exec, weight)?;
            Weight::Bf16(weight.bf16_slice()?)
        }
        _ => Weight::F32(f32_operands(OP, exec, [weight])?[0]),
    };
    let t = u32_values(OP, targets)?;
    // The targets are values too: checked before the gradients are charged.
    let targets = Targets {
        ids: t,
        ignore,
        valid: valid_rows(t, dims.vocab, ignore)?,
    };
    let mut loss = 0.0f32;
    let mut run = |gx: Option<&mut [f32]>, gw: Option<&mut [f32]>| -> Result<(), OjasError> {
        loss = fused(budget, exec, x, w, &targets, dims, chunk, [gx, gw])?;
        Ok(())
    };
    let (grad_input, grad_weight) = match (grads.input, grads.weight) {
        (true, true) => {
            let shapes = [input.shape(), weight.shape()];
            let [gx, gw] = fill_outs(OP, budget, exec, shapes, |[gx, gw]| run(Some(gx), Some(gw)))?;
            (Some(gx), Some(gw))
        }
        (true, false) => {
            let [gx] = fill_outs(OP, budget, exec, [input.shape()], |[gx]| {
                run(Some(gx), None)
            })?;
            (Some(gx), None)
        }
        (false, true) => {
            let [gw] = fill_outs(OP, budget, exec, [weight.shape()], |[gw]| {
                run(None, Some(gw))
            })?;
            (None, Some(gw))
        }
        (false, false) => {
            run(None, None)?;
            (None, None)
        }
    };
    let loss = alloc_f32(OP, budget, &[loss], &[])?;
    Ok(LinearCe {
        loss,
        grad_input,
        grad_weight,
    })
}

/// The `[V, d]` weight, read in place: `F32` values, or a `Bf16` weight's
/// bits, widened one vocabulary block at a time ([`logit_tile`]).
#[derive(Clone, Copy)]
enum Weight<'w> {
    F32(&'w [f32]),
    Bf16(&'w [u16]),
}

impl Weight<'_> {
    fn len(&self) -> usize {
        match self {
            Weight::F32(w) => w.len(),
            Weight::Bf16(w) => w.len(),
        }
    }
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
/// statistics, and the largest [`scratch`] of the GEMMs that run (the logit
/// product always, `G · W_c` for `grads.input`, `Gᵀ · x_c` for
/// `grads.weight`); with a
/// `widened` (`Bf16`) weight, one `[cols, d]` widened vocabulary block. The
/// row block of `x` and an `F32` weight's block are views of the inputs,
/// not copies. Every block is at most `rows x cols`, so the full-size block
/// bounds the partial ones.
fn tile_scratch(
    exec: Exec<'_>,
    rows: usize,
    cols: usize,
    dim: usize,
    grads: CeGrads,
    widened: bool,
) -> Result<usize, OjasError> {
    let tile = product(OP, &[rows, cols])?;
    let block = if widened {
        product(OP, &[cols, dim])?
    } else {
        0
    };
    let stats = product(OP, &[rows, 3])?;
    let mut gemm_work = scratch(OP, exec, rows, dim, cols)?;
    let mut grad_product = 0;
    if grads.input {
        gemm_work = gemm_work.max(scratch(OP, exec, rows, cols, dim)?);
        grad_product = product(OP, &[rows, dim])?;
    }
    if grads.weight {
        gemm_work = gemm_work.max(scratch(OP, exec, cols, rows, dim)?);
        grad_product = grad_product.max(product(OP, &[cols, dim])?);
    }
    [tile, stats, gemm_work, grad_product, block]
        .into_iter()
        .try_fold(0usize, |sum, n| sum.checked_add(n))
        .ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "fused cross-entropy scratch length overflows".to_string(),
        })
}

/// Rows `c0..c1` of `W` (a view of an `F32` weight, or a `Bf16` weight's
/// rows widened into a new block, which [`tile_scratch`] charged), and the
/// logit tile `x_c · W_cᵀ`. A non-finite logit is `NonFinite`, as
/// `linear_forward` reports it.
fn logit_tile<'w>(
    exec: Exec<'_>,
    x_block: &Mat,
    w: Weight<'w>,
    cols: std::ops::Range<usize>,
    dim: usize,
) -> Result<(Vec<f32>, Wide<'w>), OjasError> {
    let rows = cols.start * dim..cols.end * dim;
    let w_block = match w {
        Weight::F32(w) => Wide::Read(&w[rows]),
        Weight::Bf16(bits) => {
            let mut wide = vec![0.0f32; rows.len()];
            widen_bf16_into(exec, &bits[rows], &mut wide)?;
            Wide::Owned(wide)
        }
    };
    let w_mat = Mat::row_major(w_block.values(), cols.len(), dim);
    let logits = gemm(OP, exec, x_block, &w_mat.t())?;
    if !all_finite(&logits) {
        return Err(nonfinite(OP));
    }
    Ok((logits, w_block))
}

/// The target ids, the ignored id, and the count of the others
/// ([`valid_rows`]).
#[derive(Clone, Copy)]
struct Targets<'a> {
    ids: &'a [u32],
    ignore: Option<u32>,
    valid: u32,
}

/// The scalar loss. Each of `[grad_x, grad_w]` that is present is
/// accumulated into the caller's zeroed output.
#[allow(clippy::too_many_arguments)]
fn fused(
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    w: Weight<'_>,
    targets: &Targets<'_>,
    dims: LinearCeDims,
    chunk: CeChunk,
    [mut gx, mut gw]: [Option<&mut [f32]>; 2],
) -> Result<f32, OjasError> {
    let Targets {
        ids: targets,
        ignore,
        valid,
    } = *targets;
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
    if gx.as_ref().is_some_and(|g| g.len() != x.len())
        || gw.as_ref().is_some_and(|g| g.len() != w.len())
    {
        return Err(shape(
            OP,
            "fused cross-entropy gradient length does not match its operand",
        ));
    }
    let grads = CeGrads {
        input: gx.is_some(),
        weight: gw.is_some(),
    };
    let denom = valid as f32;
    let (rows, cols) = (chunk.rows.min(n), chunk.cols.min(v));
    // The tile charge ends when this function returns; the gradients are
    // the caller's.
    let widened = matches!(w, Weight::Bf16(_));
    let scratch = tile_scratch(exec, rows, cols, d, grads, widened)?;
    let _tiles = room_for(OP, budget, scratch)?;
    let vocab_blocks = || (0..v).step_by(cols).map(move |c0| c0..(c0 + cols).min(v));
    let single = cols == v;
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
        // With one vocabulary block the tile is the same product in every
        // pass, so it is computed once and kept: same bits, same one tile.
        let mut kept = None;

        for range in vocab_blocks() {
            let (logits, w_block) = logit_tile(exec, &x_block, w, range.clone(), d)?;
            row_maxes(
                exec,
                &logits,
                range.clone(),
                block_targets,
                ignore,
                &mut maxes,
            )?;
            if single {
                kept = Some((logits, w_block));
            }
        }

        for range in vocab_blocks() {
            let computed;
            let logits = match &kept {
                Some((logits, _)) => logits,
                None => {
                    computed = logit_tile(exec, &x_block, w, range.clone(), d)?.0;
                    &computed
                }
            };
            let rows = Rows {
                targets: block_targets,
                ignore,
                maxes: &maxes,
            };
            exp_sums(exec, logits, range, rows, [&mut sums, &mut picked])?;
        }
        for i in 0..r {
            if ignored(i) {
                continue;
            }
            let ex = excluded_ce(maxes[i], sums[i], picked[i]);
            if !ex.loss.is_finite() {
                return Err(nonfinite(OP));
            }
            total += ex.loss;
        }

        if !grads.any() {
            continue;
        }
        for range in vocab_blocks() {
            let width = range.len();
            let (mut g, w_block) = match kept.take() {
                Some(tile) => tile,
                None => logit_tile(exec, &x_block, w, range.clone(), d)?,
            };
            let rows = Rows {
                targets: block_targets,
                ignore,
                maxes: &maxes,
            };
            softmax_grad(exec, &mut g, range.clone(), rows, &sums, &picked, denom)?;
            let g = Mat::row_major(&g, r, width);
            let w_block = Mat::row_major(w_block.values(), width, d);
            if let Some(gx) = gx.as_deref_mut() {
                gemm_acc(OP, exec, &g, &w_block, &mut gx[r0 * d..r1 * d])?;
            }
            if let Some(gw) = gw.as_deref_mut() {
                gemm_acc(
                    OP,
                    exec,
                    &g.t(),
                    &x_block,
                    &mut gw[range.start * d..range.end * d],
                )?;
            }
        }
    }
    let loss = total / denom;
    if !loss.is_finite() {
        return Err(nonfinite(OP));
    }
    Ok(loss)
}

/// A row block's targets, the ignored id, and each row's maximum logit.
#[derive(Clone, Copy)]
struct Rows<'a> {
    targets: &'a [u32],
    ignore: Option<u32>,
    maxes: &'a [f32],
}

impl Rows<'_> {
    fn ignored(&self, i: usize) -> bool {
        self.ignore == Some(self.targets[i])
    }
}

/// Pass 1: each row's running maximum over this tile, skipping the target
/// column when the row is not ignored. Rows split across [`scoped`] threads;
/// a row's maximum is the same whichever thread scans it. An ignored target
/// may lie outside the vocabulary and is never used as an index.
fn row_maxes(
    exec: Exec<'_>,
    logits: &[f32],
    range: Range<usize>,
    targets: &[u32],
    ignore: Option<u32>,
    maxes: &mut [f32],
) -> Result<(), OjasError> {
    let width = range.len();
    let r = maxes.len();
    scoped::chunks_into(exec, maxes, r, 1, min_rows(width), |span, part| {
        let tile = &logits[span.start * width..span.end * width];
        for (offset, (max, row)) in part.iter_mut().zip(tile.chunks_exact(width)).enumerate() {
            let i = span.start + offset;
            let class = targets[i] as usize;
            let skip = if ignore != Some(targets[i]) && range.contains(&class) {
                Some(class - range.start)
            } else {
                None
            };
            for (col, &value) in row.iter().enumerate() {
                if Some(col) == skip {
                    continue;
                }
                if value > *max {
                    *max = value;
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// Pass 2: each valid row's `sum exp(l - max)` continued over this tile in
/// ascending vocabulary order, excluding the target column, and its target
/// logit if the tile holds it. Rows split across [`scoped`] threads; each
/// row's sum is one thread's sequence, so the bits do not depend on the
/// split. A row whose other-set is empty has a non-finite max and adds
/// nothing.
fn exp_sums(
    exec: Exec<'_>,
    logits: &[f32],
    range: Range<usize>,
    rows: Rows<'_>,
    [sums, picked]: [&mut [f32]; 2],
) -> Result<(), OjasError> {
    let (r, width) = (sums.len(), range.len());
    let outs = [sums, picked];
    scoped::chunks_into_n(
        exec,
        outs,
        r,
        [1, 1],
        min_rows(width),
        |part, [sums, picked]| {
            for (local, i) in part.enumerate() {
                if rows.ignored(i) {
                    continue;
                }
                let row = &logits[i * width..(i + 1) * width];
                let class = rows.targets[i] as usize;
                let local_class = class.checked_sub(range.start).filter(|c| *c < width);
                if rows.maxes[i].is_finite() {
                    for (col, &value) in row.iter().enumerate() {
                        if Some(col) == local_class {
                            continue;
                        }
                        let e = exp_exact(value - rows.maxes[i]);
                        if !e.is_finite() {
                            return Err(nonfinite(OP));
                        }
                        sums[local] += e;
                    }
                }
                if let Some(col) = local_class {
                    picked[local] = row[col];
                }
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// Pass 3: the logit tile turned in place into the target-excluded gradient
/// divided by `denom`. The target column, when this tile holds it, is the
/// negation of the other-class sum; every other column is
/// `exp(logit - max) * scale`. An empty other-set writes zeros and does not
/// exponentiate against a non-finite max. Ignored rows are zero. Rows split
/// across [`scoped`] threads; every value is computed alone.
fn softmax_grad(
    exec: Exec<'_>,
    g: &mut [f32],
    range: Range<usize>,
    rows: Rows<'_>,
    sums: &[f32],
    picked: &[f32],
    denom: f32,
) -> Result<(), OjasError> {
    let (r, width) = (sums.len(), range.len());
    scoped::chunks_into(exec, g, r, width, min_rows(width), |part, tile| {
        for (i, row) in part.zip(tile.chunks_exact_mut(width)) {
            if rows.ignored(i) {
                row.fill(0.0);
                continue;
            }
            let (max, sum) = (rows.maxes[i], sums[i]);
            let class = rows.targets[i] as usize;
            let scale = excluded_ce(max, sum, picked[i]).scale / denom;
            for (col, value) in row.iter_mut().enumerate() {
                if range.start + col == class {
                    *value = -sum * scale;
                } else if sum > 0.0 {
                    *value = exp_exact(*value - max) * scale;
                } else {
                    *value = 0.0;
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}
