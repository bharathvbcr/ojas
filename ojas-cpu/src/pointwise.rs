//! Embedding, pointwise ops, per-head gate, value residual, and cross-entropy.
//!
//! SiLU, mul, add, the value-residual blend and the gate's broadcast and
//! backward run in contiguous chunks on scoped threads, each chunk written
//! in place into the output ([`scoped::chunks_into_n`]); mul, add and the
//! blend's forward pass also scan each chunk for a NaN or infinity on the
//! thread that wrote it ([`fill_outs_chunked`]). No output value depends on
//! which chunk computed it, so the bits do not depend on the thread count. Under [`Numerics::Exact`]
//! every value is the scalar formula below, evaluated as written (libm
//! `exp`, no `mul_add`). Under [`Numerics::Fast`] SiLU and cross-entropy use
//! the branch-free [`crate::exp`] so their loops vectorize, the value-residual `lambda`
//! gradient sums `f64` terms over fixed blocks, and the gate's per-head dot
//! uses [`dot_lanes`].
//!
//! The gate logits `input · Wᵀ + b` and the gate's input and weight
//! gradients are products on the GEMM core ([`crate::gemm`]).

use std::ops::Range;

use ojas_core::{Budget, CeDims, EmbeddingDims, GateDims, Numerics, OjasError, Scratch, Tensor};

use crate::exp::{exp, exp_sub_store, exp_sub_sum};
use crate::gemm::{fma, gemm, gemm_out, scratch as gemm_scratch, Mat};
use crate::pool::scoped;
use crate::pool::{Exec, ROW_MIN_ELEMS};
use crate::validate::{
    all_finite, check_f32, f32_values, fill_outs_chunked, nonfinite, nonfinite_first, product,
    room_for, shape, u32_values,
};

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// The rows of `table` (`[vocab, dim]`, read in place) that `ids` pick,
/// copied value for value into one charged tensor shaped `dims.out_shape`.
///
/// The output is the only allocation and its charge. The caller has scanned
/// the table, so the copied values are finite and nothing is scanned again.
pub(crate) fn embedding_forward(
    op: &'static str,
    budget: &Budget,
    table: &[f32],
    ids: &[u32],
    dims: &EmbeddingDims,
) -> Result<Tensor, OjasError> {
    check_ids(op, ids, dims.vocab)?;
    let row = dims.dim;
    let mut out = Scratch::<f32>::try_alloc(product(op, &[dims.tokens, row])?, budget)?;
    for (dst, id) in out.as_mut_slice().chunks_exact_mut(row).zip(ids) {
        let start = id_index(*id) * row;
        let src = table
            .get(start..start + row)
            .ok_or_else(|| outside(op, "embedding row exceeds the table"))?;
        dst.copy_from_slice(src);
    }
    Tensor::from_scratch(out, &dims.out_shape)
}

/// The table gradient: each row of `grad` (`ids.shape ++ [dim]`, read in
/// place) is added into row `ids[n]` of one zeroed, charged `[vocab, dim]`
/// tensor, in token order. A table row is the f32 sum from 0 of its tokens'
/// rows in ascending token order, the order Metal's scatter also uses.
///
/// A sum that overflows is [`OjasError::NonFinite`].
pub(crate) fn embedding_backward(
    op: &'static str,
    budget: &Budget,
    ids: &[u32],
    grad: &[f32],
    dims: &EmbeddingDims,
) -> Result<Tensor, OjasError> {
    check_ids(op, ids, dims.vocab)?;
    let dim = dims.dim;
    let mut out = Scratch::<f32>::try_alloc(product(op, &[dims.vocab, dim])?, budget)?;
    let table = out.as_mut_slice();
    let mut finite = true;
    for (src, id) in grad.chunks_exact(dim).zip(ids) {
        let start = id_index(*id) * dim;
        let dst = table
            .get_mut(start..start + dim)
            .ok_or_else(|| outside(op, "embedding row exceeds the table"))?;
        for (d, s) in dst.iter_mut().zip(src) {
            let sum = *d + *s;
            finite &= sum.is_finite();
            *d = sum;
        }
    }
    if !finite {
        return Err(nonfinite(op));
    }
    Tensor::from_scratch(out, &dims.out_shape)
}

fn id_index(id: u32) -> usize {
    id as usize
}

fn outside(op: &'static str, detail: &str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: detail.to_string(),
    }
}

/// Every id below `vocab`, read in place.
fn check_ids(op: &'static str, ids: &[u32], vocab: usize) -> Result<(), OjasError> {
    for (n, &id) in ids.iter().enumerate() {
        if id_index(id) >= vocab {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("token id {id} at {n} is outside vocab {vocab}"),
            });
        }
    }
    Ok(())
}

/// `(sigmoid(x), sigmoid(-x))` under [`Numerics::Fast`]. Both come from
/// `e = e^-|x|`, so `1 - sigmoid(x)` is not formed by cancellation.
#[inline(always)]
fn sigmoid_pair_fast(x: f32) -> (f32, f32) {
    let e = exp(-x.abs());
    let big = 1.0 / (1.0 + e);
    let small = e * big;
    if x >= 0.0 {
        (big, small)
    } else {
        (small, big)
    }
}

fn same_len(op: &'static str, lens: &[usize], what: &str) -> Result<(), OjasError> {
    if lens.windows(2).all(|w| w[0] == w[1]) {
        Ok(())
    } else {
        Err(shape(op, format!("{what} lengths differ: {lens:?}")))
    }
}

/// `x * sigmoid(x)`, written into `y` (charged by the caller) in place.
pub(crate) fn silu_forward(exec: Exec<'_>, x: &[f32], y: &mut [f32]) -> Result<(), OjasError> {
    let numerics = exec.numerics;
    scoped::rows_into(exec, y, x.len(), 1, |range, dst| {
        let src = &x[range];
        match numerics {
            Numerics::Exact => {
                for (out, &v) in dst.iter_mut().zip(src) {
                    *out = v * sigmoid(v);
                }
            }
            Numerics::Fast => {
                for (out, &v) in dst.iter_mut().zip(src) {
                    *out = v * sigmoid_pair_fast(v).0;
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// `grad_y * s * (1 + x (1 - s))` with `s = sigmoid(x)`, written into
/// `grad_x` (charged by the caller) in place.
pub(crate) fn silu_backward(
    op: &'static str,
    exec: Exec<'_>,
    x: &[f32],
    grad_y: &[f32],
    grad_x: &mut [f32],
) -> Result<(), OjasError> {
    same_len(op, &[x.len(), grad_y.len()], "silu input and grad")?;
    let numerics = exec.numerics;
    scoped::rows_into(exec, grad_x, x.len(), 1, |range, dst| {
        let pairs = x[range.clone()].iter().zip(&grad_y[range]);
        match numerics {
            Numerics::Exact => {
                for (out, (&v, &g)) in dst.iter_mut().zip(pairs) {
                    let s = sigmoid(v);
                    *out = g * s * (1.0 + v * (1.0 - s));
                }
            }
            Numerics::Fast => {
                for (out, (&v, &g)) in dst.iter_mut().zip(pairs) {
                    let (s, rest) = sigmoid_pair_fast(v);
                    *out = g * s * fma(v, rest, 1.0);
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// `a * b` as a tensor of `shape`, one rounding per value under either
/// contract.
///
/// Mul and add are one memory pass: the operands are read where they are
/// and each value is written once, straight into the output tensor's
/// buffer, so nothing is copied or joined. Since 2026-10-02 the pass is
/// split into chunks of [`ROW_MIN_ELEMS`] values written in place on scoped
/// threads, each chunk scanned for a NaN or infinity by the thread that
/// wrote it ([`fill_outs_chunked`]); one chunk, or one thread, stays on the
/// calling thread. Each value is the same single
/// rounding wherever it is computed, so the bits do not depend on the
/// split. (A split on the persistent pool could not write into one buffer
/// and cost a joined copy: at `[1024, 2048]` on 6 threads it measured 1.04
/// ms for mul forward against 0.92 ms on one thread, when the operands
/// were still copied first.)
pub(crate) fn mul_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    same_len(op, &[a.len(), b.len(), product(op, shape)?], "mul operand")?;
    elementwise(op, budget, exec, shape, |range, out| {
        for ((o, &x), &y) in out.iter_mut().zip(&a[range.clone()]).zip(&b[range]) {
            *o = x * y;
        }
    })
}

/// `(grad_y * b, grad_y * a)` as tensors of `a_shape` and `b_shape` (see
/// [`mul_forward`]), both filled in one split.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mul_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    a: &[f32],
    b: &[f32],
    grad_y: &[f32],
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(Tensor, Tensor), OjasError> {
    let n = grad_y.len();
    same_len(
        op,
        &[
            a.len(),
            b.len(),
            n,
            product(op, a_shape)?,
            product(op, b_shape)?,
        ],
        "mul grad",
    )?;
    let ([grad_a, grad_b], _) = fill_outs_chunked(
        op,
        budget,
        exec,
        [a_shape, b_shape],
        n,
        [1, 1],
        ROW_MIN_ELEMS,
        |range, [ga, gb]| {
            let (a, b, g) = (&a[range.clone()], &b[range.clone()], &grad_y[range]);
            for ((o, &bv), &g) in ga.iter_mut().zip(b).zip(g) {
                *o = g * bv;
            }
            for ((o, &av), &g) in gb.iter_mut().zip(a).zip(g) {
                *o = g * av;
            }
            Ok(())
        },
    )?;
    Ok((grad_a, grad_b))
}

/// `a + b` as a tensor of `shape` (see [`mul_forward`]).
pub(crate) fn add_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    a: &[f32],
    b: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    same_len(
        op,
        &[a.len(), b.len(), product(op, shape)?],
        "residual add operand",
    )?;
    elementwise(op, budget, exec, shape, |range, out| {
        for ((o, &x), &y) in out.iter_mut().zip(&a[range.clone()]).zip(&b[range]) {
            *o = x + y;
        }
    })
}

/// A new tensor of `shape` whose values `fill` writes in chunks of
/// [`ROW_MIN_ELEMS`] on scoped threads, values `range` into `part`, each
/// chunk scanned for a NaN or infinity by the thread that wrote it
/// ([`fill_outs_chunked`]). The caller has checked that its operands hold
/// `shape`'s element count.
fn elementwise(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    shape: &[usize],
    fill: impl Fn(Range<usize>, &mut [f32]) + Sync,
) -> Result<Tensor, OjasError> {
    let n = product(op, shape)?;
    let ([out], _) = fill_outs_chunked(
        op,
        budget,
        exec,
        [shape],
        n,
        [1],
        ROW_MIN_ELEMS,
        |range, [part]| {
            fill(range, part);
            Ok(())
        },
    )?;
    Ok(out)
}

/// Both gradients of `x + y` are `grad_y` itself. The caller has run
/// [`ojas_core::residual_add_backward_dims`]; every operand is then checked
/// in place (layout, a NaN or infinity) before anything is charged; `x` and
/// `y` are read for nothing else and are not copied. Each
/// gradient is one charged byte copy of `grad_y`'s window handed to its
/// tensor without a second copy, so its bits are `grad_y`'s bits. They are
/// copies, not views: a gradient a caller rewrites in place (a clip) must
/// not share storage with `grad_y` or with the other gradient.
pub(crate) fn add_backward(
    op: &'static str,
    budget: &Budget,
    x: &Tensor,
    y: &Tensor,
    grad_y: &Tensor,
) -> Result<(Tensor, Tensor), OjasError> {
    for t in [x, y, grad_y] {
        check_f32(op, t)?;
    }
    let src = grad_y.f32_slice()?;
    let copy = |shape: &[usize]| -> Result<Tensor, OjasError> {
        let mut out = Scratch::<f32>::try_alloc(src.len(), budget)?;
        out.as_mut_slice().copy_from_slice(src);
        Tensor::from_scratch(out, shape)
    };
    let grad_x = copy(x.shape())?;
    let grad_y_out = copy(y.shape())?;
    Ok((grad_x, grad_y_out))
}

/// `weight` is `[n_head, d_model]` (nn.Linear with bias).
/// `input` is `[..., d_model]`, `attn` is `[..., n_head, head_dim]`.
/// Output is `attn * sigmoid(input @ W^T + bias)`, with the sigmoid
/// broadcast over the head dimension. The output has the shape of `attn`.
///
/// The logits are one product on the GEMM core and the broadcast multiply
/// runs in row chunks on the pool. Under [`Numerics::Exact`] each logit is
/// `((b + x_0 w_0) + x_1 w_1) + ...` in ascending input index without
/// `mul_add`, the order of the scalar loop this replaced: the product runs
/// over `[1, x]` and `[b, w]`, whose first term `0 + 1·b` is `b`.
///
/// `dims` comes from [`ojas_core::per_head_sigmoid_gate_forward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    attn: &[f32],
    dims: GateDims,
    y: &mut [f32],
) -> Result<(), OjasError> {
    gate_lengths(op, &dims, input, weight, bias, attn)?;
    if y.len() != attn.len() {
        return Err(shape(op, "gate output length does not match attn"));
    }
    // The logits phase; the output is the caller's, written in place.
    let _hold = room_for(op, budget, logit_work(op, exec, &dims)?)?;
    let gates = gate_values(op, exec, &dims, input, weight, bias)?;
    let (heads, dh) = (dims.heads, dims.head_dim);
    let width = heads * dh;
    scoped::rows_into(exec, y, dims.rows, width, |range, y| {
        let src = &attn[range.start * width..range.end * width];
        let g = &gates[range.start * heads..range.end * heads];
        for ((dst, src), &g) in y.chunks_exact_mut(dh).zip(src.chunks_exact(dh)).zip(g) {
            for (out, &a) in dst.iter_mut().zip(src) {
                *out = a * g;
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// `[grad_input, grad_weight, grad_bias, grad_attn]`: the caller's zeroed
/// outputs, written in place.
pub(crate) type GateGrads<'a> = [&'a mut [f32]; 4];

/// `gz = (sum_d grad_y · attn) · g · (1 - g)` per row and head, then
/// `grad_input = gz · W` and `grad_weight = gzᵀ · input` on the GEMM core,
/// `grad_bias` the column sums of `gz` in ascending row order, and
/// `grad_attn = grad_y · g`. Under [`Numerics::Exact`] every sum ascends
/// from `+0.0` without `mul_add`, which is the order of the scalar loop this
/// replaced; under [`Numerics::Fast`] the per-head dot uses
/// [`dot_lanes`]. `grad_attn` and `gz` are filled in row chunks on scoped
/// threads, and the two products write straight into their outputs
/// ([`gemm_out`]); `grad_bias` sums into its zeroed output. `dims` comes
/// from [`ojas_core::per_head_sigmoid_gate_backward_dims`].
pub(crate) fn gate_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [input, weight, bias, attn, grad_y]: [&[f32]; 5],
    dims: GateDims,
    [grad_x, grad_w, grad_b, grad_attn]: GateGrads<'_>,
) -> Result<(), OjasError> {
    gate_lengths(op, &dims, input, weight, bias, attn)?;
    if grad_y.len() != attn.len() {
        return Err(shape(op, "gate grad length does not match attn"));
    }
    if grad_x.len() != input.len()
        || grad_w.len() != weight.len()
        || grad_b.len() != bias.len()
        || grad_attn.len() != attn.len()
    {
        return Err(shape(op, "gate gradient length does not match its operand"));
    }
    let GateDims {
        rows,
        d_model: din,
        heads,
        head_dim: dh,
    } = dims;
    let gz_len = product(op, &[rows, heads])?;
    // Charged at once: the logits phase, `gz`, and the larger gradient
    // product's scratch. The four gradients are the caller's.
    let grads =
        gemm_scratch(op, exec, rows, heads, din)?.max(gemm_scratch(op, exec, heads, rows, din)?);
    let work = [logit_work(op, exec, &dims)?, gz_len, grads]
        .into_iter()
        .try_fold(0usize, |total, n| add(op, total, n))?;
    let _hold = room_for(op, budget, work)?;
    let gates = gate_values(op, exec, &dims, input, weight, bias)?;
    let numerics = exec.numerics;
    let width = heads * dh;
    let min_rows = (ROW_MIN_ELEMS / width.max(1)).max(1);
    let mut gz = vec![0.0f32; gz_len];
    scoped::chunks_into_n(
        exec,
        [grad_attn, &mut gz],
        rows,
        [width, heads],
        min_rows,
        |range, [grad_attn, gz]| {
            let attn = &attn[range.start * width..range.end * width];
            let gy = &grad_y[range.start * width..range.end * width];
            let g = &gates[range.start * heads..range.end * heads];
            let heads_in_range = grad_attn
                .chunks_exact_mut(dh)
                .zip(attn.chunks_exact(dh))
                .zip(gy.chunks_exact(dh))
                .zip(g.iter().zip(gz.iter_mut()));
            for (((dst, a), gy), (&g, gz)) in heads_in_range {
                for (out, &v) in dst.iter_mut().zip(gy) {
                    *out = v * g;
                }
                let grad_g = match numerics {
                    Numerics::Exact => dot_ascending(gy, a),
                    Numerics::Fast => dot_lanes(gy, a),
                };
                *gz = grad_g * g * (1.0 - g);
            }
            Ok(())
        },
    )?;
    for row in gz.chunks_exact(heads) {
        for (slot, &v) in grad_b.iter_mut().zip(row) {
            *slot += v;
        }
    }
    let gz = Mat::row_major(&gz, rows, heads);
    // grad_x[row, i] = sum_head gz[row, head] * W[head, i], head from 0.
    gemm_out(op, exec, &gz, &Mat::row_major(weight, heads, din), grad_x)?;
    // grad_w[head, i] = sum_row gz[row, head] * x[row, i], row from 0.
    gemm_out(op, exec, &gz.t(), &Mat::row_major(input, rows, din), grad_w)
}

fn add(op: &'static str, a: usize, b: usize) -> Result<usize, OjasError> {
    a.checked_add(b).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "scratch length overflows".to_string(),
    })
}

/// `sigmoid(input · Wᵀ + b)` as `[rows, heads]`. A logit that is not finite
/// is refused, not turned into a gate of 0 or 1.
fn gate_values(
    op: &'static str,
    exec: Exec<'_>,
    dims: &GateDims,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, OjasError> {
    let (rows, din, heads) = (dims.rows, dims.d_model, dims.heads);
    let mut z = match exec.numerics {
        Numerics::Exact => {
            // `[1, x]` and `[b, w]`: the product's first term is the bias.
            let width = din + 1;
            let mut ones_x = Vec::with_capacity(product(op, &[rows, width])?);
            for row in input.chunks_exact(din) {
                ones_x.push(1.0f32);
                ones_x.extend_from_slice(row);
            }
            let mut bias_w = Vec::with_capacity(product(op, &[heads, width])?);
            for (row, &b) in weight.chunks_exact(din).zip(bias) {
                bias_w.push(b);
                bias_w.extend_from_slice(row);
            }
            let a = Mat::row_major(&ones_x, rows, width);
            let w = Mat::row_major(&bias_w, heads, width);
            gemm(op, exec, &a, &w.t())?
        }
        Numerics::Fast => {
            let a = Mat::row_major(input, rows, din);
            let w = Mat::row_major(weight, heads, din);
            let mut z = gemm(op, exec, &a, &w.t())?;
            for row in z.chunks_exact_mut(heads) {
                for (v, &b) in row.iter_mut().zip(bias) {
                    *v += b;
                }
            }
            z
        }
    };
    for v in z.iter_mut() {
        if !v.is_finite() {
            return Err(nonfinite(op));
        }
        *v = sigmoid(*v);
    }
    Ok(z)
}

/// `sum a[i] * b[i]` from `+0.0` in ascending `i`, separate multiply and add.
fn dot_ascending(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |acc, (x, y)| acc + x * y)
}

/// Lanes of the [`Numerics::Fast`] reductions.
const LANES: usize = 8;

/// `sum a[i] * b[i]` under [`Numerics::Fast`]: [`LANES`] partial sums with
/// `mul_add` (lane `l` takes `i ≡ l mod LANES` in ascending order), added in
/// lane order, then the tail in ascending order. The order depends only on
/// the length, so the loop vectorizes and the bits do not depend on the
/// thread count.
fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.iter().zip(cb) {
        for lane in 0..LANES {
            acc[lane] = fma(x[lane], y[lane], acc[lane]);
        }
    }
    let mut total = acc.iter().fold(0.0f32, |s, &v| s + v);
    for (&x, &y) in ra.iter().zip(rb) {
        total = fma(x, y, total);
    }
    total
}

/// The copies hold as many values as `dims` says they do.
fn gate_lengths(
    op: &'static str,
    dims: &GateDims,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    attn: &[f32],
) -> Result<(), OjasError> {
    if input.len() != product(op, &[dims.rows, dims.d_model])?
        || weight.len() != product(op, &[dims.heads, dims.d_model])?
        || bias.len() != dims.heads
        || attn.len() != product(op, &[dims.rows, dims.heads, dims.head_dim])?
    {
        return Err(shape(op, "gate data length does not match shape"));
    }
    Ok(())
}

/// Floats [`gate_values`] allocates: the logits, the logit product's
/// scratch and, under [`Numerics::Exact`], the `[1, x]` and `[b, w]`
/// operands.
fn logit_work(op: &'static str, exec: Exec<'_>, dims: &GateDims) -> Result<usize, OjasError> {
    let z = product(op, &[dims.rows, dims.heads])?;
    match exec.numerics {
        Numerics::Exact => {
            let width = add(op, dims.d_model, 1)?;
            let ones_x = product(op, &[dims.rows, width])?;
            let bias_w = product(op, &[dims.heads, width])?;
            let work = gemm_scratch(op, exec, dims.rows, width, dims.heads)?;
            [ones_x, bias_w, work]
                .into_iter()
                .try_fold(z, |total, n| add(op, total, n))
        }
        Numerics::Fast => add(
            op,
            z,
            gemm_scratch(op, exec, dims.rows, dims.d_model, dims.heads)?,
        ),
    }
}

/// `y = (1 - sigmoid(lambda)) * value + sigmoid(lambda) * value0` as a
/// tensor of `shape`, split and written in place like [`mul_forward`].
pub(crate) fn value_residual_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    value: &[f32],
    value0: &[f32],
    lambda: f32,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    same_len(
        op,
        &[value.len(), value0.len(), product(op, shape)?],
        "value residual operand",
    )?;
    let s = sigmoid(lambda);
    if !s.is_finite() {
        return Err(nonfinite(op));
    }
    elementwise(op, budget, exec, shape, |range, out| {
        for ((o, &v), &v0) in out
            .iter_mut()
            .zip(&value[range.clone()])
            .zip(&value0[range])
        {
            *o = (1.0 - s) * v + s * v0;
        }
    })
}

/// Values per block of the [`Numerics::Fast`] `lambda` gradient sum. Block
/// boundaries are fixed by the length alone, never by the task split.
const LAMBDA_BLOCK: usize = 1 << 12;

/// Writes `grad_value` and `grad_value0` into the caller's outputs and
/// returns `grad_lambda`.
///
/// It runs in [`LAMBDA_BLOCK`] chunks on scoped threads, each task filling
/// its blocks of both gradients in place and, under [`Numerics::Fast`],
/// returning its blocks' `grad_lambda` sums.
///
/// `grad_lambda = s (1 - s) sum_i (value0[i] - value[i]) grad_y[i]` with
/// `s = sigmoid(lambda)`, a sum over every element.
/// - [`Numerics::Exact`]: the sum ascends from index 0 in `f32` with
///   separate multiply and add, the crate's reduction contract (lib.rs and
///   `ojas_core::Numerics`), on the calling thread after the gradients; its
///   error grows with the element count.
/// - [`Numerics::Fast`]: each term is formed in `f64` (exact whenever
///   `value` and `value0` are within a factor 2^29 of each other, otherwise
///   rounded at 2^-53), summed over fixed [`LAMBDA_BLOCK`]-value blocks in
///   [`LANES`] `f64` partial sums, the block sums added in block order, and
///   `s (1 - s)` applied in `f64` before the one rounding to `f32`. At the
///   nanolab shape (786k values) that is within 1e-6 of the `f64` sum where
///   the ascending `f32` sum was 3.2e-5 off.
pub(crate) fn value_residual_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [value, value0, grad_y]: [&[f32]; 3],
    lambda: f32,
    [grad_v, grad_v0]: [&mut [f32]; 2],
) -> Result<f32, OjasError> {
    let n = value.len();
    same_len(
        op,
        &[n, value0.len(), grad_y.len(), grad_v.len(), grad_v0.len()],
        "value residual grad",
    )?;
    let s = sigmoid(lambda);
    let blocks = n.div_ceil(LAMBDA_BLOCK);
    // One f64 (two f32) per block; the gradients are the caller's.
    let _sums = room_for(op, budget, product(op, &[blocks, 2])?)?;
    let numerics = exec.numerics;
    let min_blocks = ROW_MIN_ELEMS.div_ceil(LAMBDA_BLOCK);
    let sums = scoped::chunks_into_n(
        exec,
        [grad_v, grad_v0],
        blocks,
        [LAMBDA_BLOCK; 2],
        min_blocks,
        |block_range, [grad_v, grad_v0]| {
            let range = block_range.start * LAMBDA_BLOCK..(block_range.end * LAMBDA_BLOCK).min(n);
            let gy = &grad_y[range.clone()];
            for ((gv, gv0), &g) in grad_v.iter_mut().zip(grad_v0.iter_mut()).zip(gy) {
                *gv = (1.0 - s) * g;
                *gv0 = s * g;
            }
            Ok(match numerics {
                Numerics::Exact => Vec::new(),
                Numerics::Fast => range
                    .step_by(LAMBDA_BLOCK)
                    .map(|start| {
                        let end = (start + LAMBDA_BLOCK).min(n);
                        diff_dot_f64(&value[start..end], &value0[start..end], &grad_y[start..end])
                    })
                    .collect::<Vec<f64>>(),
            })
        },
    )?;
    let grad_lambda = match numerics {
        Numerics::Exact => {
            let mut grad_s = 0.0f32;
            for ((&v, &v0), &g) in value.iter().zip(value0).zip(grad_y) {
                grad_s += (v0 - v) * g;
            }
            grad_s * s * (1.0 - s)
        }
        Numerics::Fast => {
            // The block sums in block order: task order, then each task's.
            let grad_s = sums.iter().flatten().fold(0.0f64, |total, &b| total + b);
            let s = 1.0 / (1.0 + (-f64::from(lambda)).exp());
            (grad_s * s * (1.0 - s)) as f32
        }
    };
    Ok(grad_lambda)
}

/// `sum_i (value0[i] - value[i]) * grad_y[i]` with every term formed in
/// `f64`, in [`LANES`] partial sums added in lane order, then the tail.
fn diff_dot_f64(value: &[f32], value0: &[f32], grad_y: &[f32]) -> f64 {
    let (cv, rv) = value.as_chunks::<LANES>();
    let (cv0, rv0) = value0.as_chunks::<LANES>();
    let (cg, rg) = grad_y.as_chunks::<LANES>();
    let mut acc = [0.0f64; LANES];
    for ((v, v0), g) in cv.iter().zip(cv0).zip(cg) {
        for lane in 0..LANES {
            acc[lane] += (f64::from(v0[lane]) - f64::from(v[lane])) * f64::from(g[lane]);
        }
    }
    let mut total = acc.iter().fold(0.0f64, |s, &v| s + v);
    for ((&v, &v0), &g) in rv.iter().zip(rv0).zip(rg) {
        total += (f64::from(v0) - f64::from(v)) * f64::from(g);
    }
    total
}

/// Logits per cross-entropy task. A task's row count is fixed by the
/// vocabulary alone, so the partition does not depend on the thread count.
const CE_BLOCK_ELEMS: usize = 1 << 18;

/// Rows per cross-entropy task.
fn ce_block_rows(vocab: usize) -> usize {
    (CE_BLOCK_ELEMS / vocab.max(1)).max(1)
}

/// Count of valid targets, after checking that every target other than
/// `ignore` is inside the vocabulary. No valid target is
/// [`OjasError::NonFinite`]: the mean has no denominator (torch gives NaN),
/// and it is not a finite loss of 0.
fn ce_valid(
    op: &'static str,
    targets: &[u32],
    vocab: usize,
    ignore: Option<u32>,
) -> Result<u32, OjasError> {
    let mut valid: u32 = 0;
    for (n, &target) in targets.iter().enumerate() {
        if ignore == Some(target) {
            continue;
        }
        if id_index(target) >= vocab {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("target {target} at {n} is outside vocab {vocab}"),
            });
        }
        valid = valid.checked_add(1).ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "valid target count overflows".to_string(),
        })?;
    }
    if valid == 0 {
        return Err(nonfinite(op));
    }
    Ok(valid)
}

/// Magnitude bits of an f32 (the sign cleared); a value is NaN or infinite
/// exactly when they are at least [`NON_FINITE`].
const MAGNITUDE: u32 = 0x7fff_ffff;
const NON_FINITE: u32 = 0x7f80_0000;

/// Largest value of a row, and whether every value is finite: the scalar
/// loop under Exact, lane maxima under Fast. Every value is read either
/// way, so this pass is also the row's NaN scan. Among finite values the
/// order of comparisons cannot change the maximum.
fn row_max(row: &[f32], numerics: Numerics) -> (f32, bool) {
    let value = |v: &f32| *v;
    let bits = |v: &f32| v.to_bits() & MAGNITUDE;
    match numerics {
        Numerics::Exact => {
            let (mut max, mut top) = (f32::NEG_INFINITY, 0u32);
            for word in row {
                let v = value(word);
                if v > max {
                    max = v;
                }
                top = top.max(bits(word));
            }
            (max, top < NON_FINITE)
        }
        Numerics::Fast => {
            let (chunks, rest) = row.as_chunks::<LANES>();
            let mut acc = [f32::NEG_INFINITY; LANES];
            let mut top = [0u32; LANES];
            for chunk in chunks {
                for ((a, t), word) in acc.iter_mut().zip(top.iter_mut()).zip(chunk) {
                    *a = a.max(value(word));
                    *t = (*t).max(bits(word));
                }
            }
            let mut max = acc.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let mut top = top.iter().fold(0u32, |m, &t| m.max(t));
            for word in rest {
                max = max.max(value(word));
                top = top.max(bits(word));
            }
            (max, top < NON_FINITE)
        }
    }
}

/// `sum_j e^(x_j - max)` of one row, each term also written to `store` when
/// it is given. Exact: libm `exp` and an ascending f32 sum, the reference
/// order. Fast: [`crate::exp`] in its fixed lane order.
fn row_exp_sum(
    op: &'static str,
    row: &[f32],
    max: f32,
    numerics: Numerics,
    store: Option<&mut [f32]>,
) -> Result<f32, OjasError> {
    let sum = match (numerics, store) {
        (Numerics::Exact, store) => {
            let mut sum = 0.0f32;
            let mut store = store;
            for (col, word) in row.iter().enumerate() {
                let e = (*word - max).exp();
                if !e.is_finite() {
                    return Err(nonfinite(op));
                }
                if let Some(out) = store.as_deref_mut() {
                    out[col] = e;
                }
                sum += e;
            }
            sum
        }
        (Numerics::Fast, Some(out)) => exp_sub_store(row, max, out),
        (Numerics::Fast, None) => exp_sub_sum(row, max),
    };
    if !(sum.is_finite() && sum > 0.0) {
        return Err(nonfinite(op));
    }
    Ok(sum)
}

/// A mean cross-entropy over `logits` `[rows, vocab]` and `targets`
/// `[rows]`, both read in place, with `valid` rows counted by
/// [`ce_valid`].
#[derive(Clone, Copy)]
pub(crate) struct CeInput<'a> {
    pub logits: &'a [f32],
    pub targets: &'a [u32],
    pub vocab: usize,
    pub ignore: Option<u32>,
    pub valid: u32,
}

impl CeInput<'_> {
    /// One row's loss term `max + ln(sum) - x[target]`, or `None` for an
    /// ignored row. With `grad`, the row's `(softmax - onehot) / valid` is
    /// written into it. Every logit of the row is checked finite, ignored
    /// rows included.
    fn row(
        &self,
        op: &'static str,
        numerics: Numerics,
        n: usize,
        grad: Option<&mut [f32]>,
    ) -> Result<Option<f32>, OjasError> {
        let row = self
            .logits
            .get(n * self.vocab..(n + 1) * self.vocab)
            .ok_or_else(|| outside(op, "cross-entropy row exceeds logits"))?;
        let target = self.targets[n];
        if self.ignore == Some(target) {
            return if all_finite(row) {
                Ok(None)
            } else {
                Err(nonfinite(op))
            };
        }
        let (max, finite) = row_max(row, numerics);
        if !finite {
            return Err(nonfinite(op));
        }
        let class = target as usize;
        let picked = row[class];
        let sum = match grad {
            None => row_exp_sum(op, row, max, numerics, None)?,
            Some(dst) => {
                let sum = row_exp_sum(op, row, max, numerics, Some(&mut *dst))?;
                let denom = self.valid as f32;
                for slot in dst.iter_mut() {
                    let p = *slot / sum;
                    *slot = p / denom;
                }
                dst[class] -= 1.0 / denom;
                sum
            }
        };
        Ok(Some(max + sum.ln() - picked))
    }

    /// The mean loss, in fixed blocks of rows on the pool's threads; with
    /// `grad` (`[rows, vocab]` values, zeroed) every valid row's gradient is
    /// written into it once and ignored rows keep their zeros.
    ///
    /// Each valid row's term comes back separately and the terms are added
    /// in increasing row order, so the loss bits do not depend on the blocks
    /// or the thread count, and no gradient value depends on which thread
    /// wrote it. Any non-finite logit, sum or loss is
    /// [`OjasError::NonFinite`]; this pass reads every logit, so it is also
    /// the logits' NaN scan.
    pub(crate) fn mean(
        &self,
        op: &'static str,
        exec: Exec<'_>,
        grad: Option<&mut [f32]>,
    ) -> Result<f32, OjasError> {
        let numerics = exec.numerics;
        let block = ce_block_rows(self.vocab);
        let rows = self.targets.len();
        let rows_of = |b: usize| b * block..((b + 1) * block).min(rows);
        let terms: Vec<Vec<f32>> = match grad {
            None => scoped::map(exec, rows.div_ceil(block), |b| {
                let mut terms = Vec::with_capacity(block);
                for n in rows_of(b) {
                    terms.extend(self.row(op, numerics, n, None)?);
                }
                Ok(terms)
            })?,
            Some(grad) => scoped::fill(exec, grad, block * self.vocab, |b, chunk| {
                let mut terms = Vec::with_capacity(block);
                for (n, dst) in rows_of(b).zip(chunk.chunks_exact_mut(self.vocab)) {
                    terms.extend(self.row(op, numerics, n, Some(dst))?);
                }
                Ok(terms)
            })?,
        };
        let mut total = 0.0f32;
        for term in terms.iter().flatten() {
            total += term;
        }
        let loss = total / self.valid as f32;
        if !loss.is_finite() {
            return Err(nonfinite(op));
        }
        Ok(loss)
    }
}

/// The checked [`CeInput`] of a cross-entropy call whose shapes
/// `cross_entropy_mean_*_dims` accepted: the logits' layout, then the
/// targets' layout, range and valid count, all read in place. The logits'
/// NaN scan is [`CeInput::mean`]'s pass, so each refusal here first scans
/// them ([`nonfinite_first`]): a NaN outranks it, as in argument order.
pub(crate) fn ce_input<'a>(
    op: &'static str,
    exec: Exec<'_>,
    logits: &'a Tensor,
    targets: &'a Tensor,
    dims: CeDims,
    ignore: Option<u32>,
) -> Result<CeInput<'a>, OjasError> {
    let logits_w = f32_values(op, logits)?;
    let scan_first = |err| nonfinite_first(op, exec, &[logits], err);
    let targets_w = u32_values(op, targets).map_err(scan_first)?;
    let valid = ce_valid(op, targets_w, dims.vocab, ignore).map_err(scan_first)?;
    Ok(CeInput {
        logits: logits_w,
        targets: targets_w,
        vocab: dims.vocab,
        ignore,
        valid,
    })
}

/// [`CeInput::mean`] with its gradient written once into a charged tensor
/// shaped `shape`: `out` holds `rows * vocab` zeroed f32 values. The loss is
/// formed too and a non-finite loss is refused, as the forward refuses it.
///
/// Every value is `(e / sum) / valid` with `e = e^(x - max)` in `[0, 1]` and
/// `sum >= 1` (the maximum's own term is exactly 1), minus `1 / valid` at
/// the target, so the gradient is finite by construction and is not scanned
/// again.
pub(crate) fn cross_entropy_grad(
    op: &'static str,
    exec: Exec<'_>,
    input: CeInput<'_>,
    mut out: Scratch<f32>,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    input.mean(op, exec, Some(out.as_mut_slice()))?;
    Tensor::from_scratch(out, shape)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::pool::Pool;

    /// A `shape` that disagrees with the operands is refused before
    /// anything is charged, for every elementwise forward pass: none of
    /// them may write a prefix of the operands into a smaller output or
    /// leave a larger one partly zero.
    #[test]
    fn elementwise_forwards_refuse_a_shape_that_disagrees_with_their_operands() {
        let pool = Arc::new(Pool::new(2).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let budget = Budget::new(1 << 20);
        let x = [1.0f32; 6];
        for bad in [&[2usize, 2][..], &[2, 4], &[7]] {
            let results = [
                ("mul", mul_forward("t", &budget, exec, &x, &x, bad)),
                ("add", add_forward("t", &budget, exec, &x, &x, bad)),
                (
                    "value_residual",
                    value_residual_forward("t", &budget, exec, &x, &x, 0.5, bad),
                ),
            ];
            for (name, got) in results {
                assert!(
                    matches!(got, Err(OjasError::Shape { .. })),
                    "{name} {bad:?}: {got:?}"
                );
            }
            assert_eq!(budget.live_bytes().unwrap(), 0);
        }
        let good = value_residual_forward("t", &budget, exec, &x, &x, 0.5, &[2, 3]).unwrap();
        assert_eq!(good.f32_slice().unwrap(), &x[..]);
    }

    /// An output large enough for the output scan to split into blocks
    /// across threads still refuses an overflow in its last value only, and
    /// releases its charge; with no overflow it is recorded finite.
    #[test]
    fn a_multi_block_output_refuses_an_overflow_in_its_last_value() {
        let pool = Arc::new(Pool::new(4).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let budget = Budget::new(1 << 26);
        let n = (1 << 21) + 3;
        let mut a = vec![1.0f32; n];
        let mut b = vec![2.0f32; n];
        let ok = mul_forward("t", &budget, exec, &a, &b, &[n]).unwrap();
        assert!(ok.f32_slice().unwrap().iter().all(|&v| v == 2.0));
        drop(ok);
        a[n - 1] = f32::MAX;
        b[n - 1] = f32::MAX;
        let got = mul_forward("t", &budget, exec, &a, &b, &[n]);
        assert!(matches!(got, Err(OjasError::NonFinite { .. })), "{got:?}");
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }
}
