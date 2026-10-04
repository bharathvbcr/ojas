//! `y = x @ W^T` and its gradients, as strided views over one GEMM core.
//!
//! `W` is `[out, in]`, the `nn.Linear(..., bias=False)` layout. The forward
//! is `x · Wᵀ`, `grad_x` is `g · W`, and `grad_w` is `gᵀ · x`; the transposes
//! are strides that [`crate::gemm`] packs through, not copies. Under
//! [`ojas_core::Numerics::Exact`] each output sums its reduction axis from
//! index 0 upward in `f32` without `mul_add`, and the bits do not depend on
//! the thread count.

use ojas_core::{Budget, LinearDims, OjasError};

use crate::gemm::{fma, gemm_out, scratch, single_task, whole_call, Mat, TASK_MACS};
use crate::pool::{scoped, Exec};
use crate::validate::{product, room_for, shape};

/// Side of one transpose tile. The tile is loaded as contiguous source rows
/// and stored as contiguous output rows. Writing `out[col * rows + row]` in
/// a row-major scan stores with stride `rows`; on the tall Muon matrix that
/// stride is 12 KiB, so each store misses.
const TRANSPOSE_TILE: usize = 32;

pub(crate) fn transpose(
    op: &'static str,
    a: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, OjasError> {
    let len = product(op, &[rows, cols])?;
    if a.len() != len {
        return Err(shape(
            op,
            format!("transpose data len {} != {rows}*{cols}", a.len()),
        ));
    }
    let mut out = vec![0.0f32; len];
    if rows == 0 || cols == 0 {
        return Ok(out);
    }
    let mut tile = [0.0f32; TRANSPOSE_TILE * TRANSPOSE_TILE];
    let mut row0 = 0;
    while row0 < rows {
        let th = TRANSPOSE_TILE.min(rows - row0);
        let mut col0 = 0;
        while col0 < cols {
            let tw = TRANSPOSE_TILE.min(cols - col0);
            for i in 0..th {
                let from = (row0 + i) * cols + col0;
                tile[i * TRANSPOSE_TILE..i * TRANSPOSE_TILE + tw]
                    .copy_from_slice(&a[from..from + tw]);
            }
            for j in 0..tw {
                let to = (col0 + j) * rows + row0;
                for i in 0..th {
                    out[to + i] = tile[i * TRANSPOSE_TILE + j];
                }
            }
            col0 += TRANSPOSE_TILE;
        }
        row0 += TRANSPOSE_TILE;
    }
    Ok(out)
}

/// `x` is `[rows, in]`, `weight` is `[out, in]`, and the output `y`
/// (`[rows, out]`, zeroed, charged by the caller) is written in place;
/// `dims` comes from [`ojas_core::linear_forward_dims`]. The operands are
/// read where they are (a tensor's own storage), so beside the output only
/// the GEMM scratch is charged.
pub(crate) fn linear_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    weight: &[f32],
    dims: &LinearDims,
    y: &mut [f32],
) -> Result<(), OjasError> {
    let (rows, kin, nout) = (dims.rows, dims.in_features, dims.out_features);
    if x.len() != product(op, &[rows, kin])?
        || weight.len() != product(op, &[nout, kin])?
        || y.len() != product(op, &[rows, nout])?
    {
        return Err(shape(op, "linear data length does not match shape"));
    }
    let _scratch = room_for(op, budget, scratch(op, exec, rows, kin, nout)?)?;
    let x = Mat::row_major(x, rows, kin);
    let w = Mat::row_major(weight, nout, kin);
    gemm_out(op, exec, &x, &w.t(), y)
}

/// `grad_x` (`[rows, in]`) and `grad_w` (`[out, in]`) for `grad_y`
/// `[rows, out]`, written into `grads` (zeroed, charged by the caller);
/// `dims` comes from [`ojas_core::linear_backward_dims`]. The operands are
/// read where they are, as in [`linear_forward`].
///
/// When the forward has one row and the weight gradient would be a Fast
/// whole call, that product is an outer product of two contiguous vectors
/// ([`rank1_weight_grad`]). `cblas_sgemm` on that shape spent most of the
/// call in `memset` (release `sample` of `linear_dec_up` backward).
///
/// Returns whether every weight-gradient element was checked finite while it
/// was stored. That is only the one-row Fast path; other paths return
/// `false` and the caller scans `grad_w`. A non-finite weight gradient is
/// [`OjasError::NonFinite`] and nothing is published.
pub(crate) fn linear_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [x, weight, grad_y]: [&[f32]; 3],
    dims: &LinearDims,
    [grad_x, grad_w]: [&mut [f32]; 2],
) -> Result<bool, OjasError> {
    let (rows, kin, nout) = (dims.rows, dims.in_features, dims.out_features);
    if x.len() != product(op, &[rows, kin])?
        || weight.len() != product(op, &[nout, kin])?
        || grad_y.len() != product(op, &[rows, nout])?
        || grad_x.len() != product(op, &[rows, kin])?
        || grad_w.len() != product(op, &[nout, kin])?
    {
        return Err(shape(
            op,
            "linear backward data length does not match shape",
        ));
    }
    // Two products that each fit in one task run side by side, so their
    // scratch is live at once; larger ones split themselves in turn. A whole
    // `ojas-simd` call must not run inside a pool task. On macOS a Fast
    // product of at least `TASK_MACS / 2` is always a whole call, so only
    // Exact pairs there; off macOS Fast pairs too.
    let pair = exec.pool.threads() > 1
        && single_task(exec, rows, kin, nout)
        && single_task(exec, nout, kin, rows)
        && !whole_call(exec.numerics, rows, nout, kin)
        && rows.saturating_mul(nout).saturating_mul(kin) >= TASK_MACS / 2;
    let (sx, sw) = (
        scratch(op, exec, rows, nout, kin)?,
        scratch(op, exec, nout, rows, kin)?,
    );
    let work = if pair {
        sx.checked_add(sw).ok_or_else(|| overflow(op))?
    } else {
        sx.max(sw)
    };
    let _work = room_for(op, budget, work)?;
    // One forward row: grad_w[col, inner] = g[col] * x[inner]. Above the
    // whole-call cutoff that went through `cblas_sgemm`, which memsets the
    // gradient (the same size as `W`) before writing the products. The
    // vectors are already contiguous, so write the Fast chain directly.
    // `grad_x` is still a reduction over `out` and stays on the GEMM.
    let direct_w = rows == 1 && whole_call(exec.numerics, nout, rows, kin);
    let x_mat = Mat::row_major(x, rows, kin);
    let w = Mat::row_major(weight, nout, kin);
    let g = Mat::row_major(grad_y, rows, nout);
    // grad_x[row, inner] = sum_col g[row, col] * W[col, inner], col from 0.
    // grad_w[col, inner] = sum_row g[row, col] * x[row, inner], row from 0.
    // A paired product fits in one task, so it runs whole on its scoped
    // thread, which reads the borrowed operands and writes its own output.
    if pair {
        scoped::fill_parts(exec, vec![grad_x, grad_w], |i, out| {
            if i == 0 {
                gemm_out(op, exec, &g, &w, out)
            } else {
                gemm_out(op, exec, &g.t(), &x_mat, out)
            }
        })?;
        return Ok(false);
    } else if direct_w {
        gemm_out(op, exec, &g, &w, grad_x)?;
        // The same one check a whole call makes before it starts.
        exec.pool.cancel_hook()()?;
        // Every output was just written. The flag is that check, so the
        // caller does not scan `grad_w` again.
        if !rank1_weight_grad(grad_y, x, grad_w) {
            return Err(crate::validate::nonfinite(op));
        }
        return Ok(true);
    } else {
        gemm_out(op, exec, &g, &w, grad_x)?;
        gemm_out(op, exec, &g.t(), &x_mat, grad_w)?;
    }
    Ok(false)
}

/// `grad_w[col, inner] = fma(g[col], x[inner], +0.0)` for a single forward
/// row. That is the Fast ascending chain with `k = 1`: one product per
/// output, from `+0.0`, no second rounding and no panel pack.
fn rank1_weight_grad(g: &[f32], x: &[f32], grad_w: &mut [f32]) -> bool {
    let kin = x.len();
    let mut finite = true;
    for (dst, &s) in grad_w.chunks_exact_mut(kin).zip(g) {
        for (d, &xv) in dst.iter_mut().zip(x) {
            *d = fma(s, xv, 0.0);
            finite &= d.is_finite();
        }
    }
    finite
}

fn overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "linear scratch length overflows".to_string(),
    }
}

#[cfg(test)]
mod transpose_tests {
    use super::transpose;

    fn index_swap(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = a[row * cols + col];
            }
        }
        out
    }

    #[test]
    fn transpose_matches_the_index_swap_including_partial_tiles() {
        let shapes = [
            (0, 4),
            (3, 0),
            (1, 1),
            (1, 7),
            (7, 1),
            (3, 2),
            (32, 32),
            (33, 17),
            (40, 70),
            (70, 40),
            (768, 3),
            (3, 768),
        ];
        let mut seed = 0x9e37_79b9u64;
        for (rows, cols) in shapes {
            let n = rows * cols;
            let mut a = Vec::with_capacity(n);
            for _ in 0..n {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                a.push(f32::from_bits((seed >> 32) as u32));
            }
            if !a.is_empty() {
                a[0] = -0.0;
            }
            let got = transpose("transpose", &a, rows, cols).unwrap();
            let want = index_swap(&a, rows, cols);
            assert_eq!(got.len(), want.len(), "{rows}x{cols}");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "{rows}x{cols} [{i}]");
            }
        }
        assert!(transpose("transpose", &[1.0], 2, 2).is_err());
    }
}
