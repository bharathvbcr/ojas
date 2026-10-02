//! Embedding, pointwise ops, per-head gate, value residual, and cross-entropy.
//!
//! SiLU, the value-residual backward pass and the gate's broadcast run in
//! contiguous chunks on the pool and join the chunks in order; mul, add and
//! the value-residual forward pass run in place on the calling thread (see
//! [`mul_forward`]). No output value depends on which chunk computed it, so
//! the bits do not depend on the thread count. Under [`Numerics::Exact`]
//! every value is the scalar formula below, evaluated as written (libm
//! `exp`, no `mul_add`). Under [`Numerics::Fast`] SiLU and cross-entropy use
//! the branch-free [`crate::exp`] so their loops vectorize, the value-residual `lambda`
//! gradient sums `f64` terms over fixed blocks, and the gate's per-head dot
//! uses [`dot_lanes`].
//!
//! The gate logits `input · Wᵀ + b` and the gate's input and weight
//! gradients are products on the GEMM core ([`crate::gemm`]).

use std::ops::Range;
use std::sync::Arc;

use ojas_core::{Budget, CeDims, EmbeddingDims, GateDims, Numerics, OjasError, Scratch, Tensor};

use crate::exp::{exp, exp_sub_store, exp_sub_sum};
use crate::gemm::{fma, gemm, scratch as gemm_scratch, Mat};
use crate::pool::scoped;
use crate::pool::{Exec, ROW_MIN_ELEMS};
use crate::validate::{
    all_finite, check_f32, f32_values, nonfinite, nonfinite_first, product, room_for, shape,
    u32_values,
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

/// `f(range)` over `0..len` in contiguous chunks of at least
/// [`ROW_MIN_ELEMS`] values, joined in order.
fn elementwise<F>(exec: Exec<'_>, len: usize, f: F) -> Result<Vec<f32>, OjasError>
where
    F: Fn(Range<usize>) -> Vec<f32> + Send + Sync + 'static,
{
    exec.rows(len, 1, move |range| Ok(f(range)))
}

/// Concatenate task parts in order. One part is moved, not copied.
fn join2(parts: Vec<(Vec<f32>, Vec<f32>)>, len_a: usize, len_b: usize) -> (Vec<f32>, Vec<f32>) {
    if parts.len() == 1 {
        if let Some(only) = parts.into_iter().next() {
            return only;
        }
        return (Vec::new(), Vec::new());
    }
    let mut a = Vec::with_capacity(len_a);
    let mut b = Vec::with_capacity(len_b);
    for (pa, pb) in parts {
        a.extend_from_slice(&pa);
        b.extend_from_slice(&pb);
    }
    (a, b)
}

fn same_len(op: &'static str, lens: &[usize], what: &str) -> Result<(), OjasError> {
    if lens.windows(2).all(|w| w[0] == w[1]) {
        Ok(())
    } else {
        Err(shape(op, format!("{what} lengths differ: {lens:?}")))
    }
}

/// `x * sigmoid(x)`. The output's task parts are charged here; the caller
/// charges the joined output.
pub(crate) fn silu_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Vec<f32>,
) -> Result<Vec<f32>, OjasError> {
    let n = x.len();
    let _parts = room_for(op, budget, n)?;
    let x = Arc::new(x);
    let numerics = exec.numerics;
    elementwise(exec, n, move |range| {
        let src = &x[range];
        match numerics {
            Numerics::Exact => src.iter().map(|&v| v * sigmoid(v)).collect(),
            Numerics::Fast => src.iter().map(|&v| v * sigmoid_pair_fast(v).0).collect(),
        }
    })
}

/// `grad_y * s * (1 + x (1 - s))` with `s = sigmoid(x)`.
pub(crate) fn silu_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Vec<f32>,
    grad_y: Vec<f32>,
) -> Result<Vec<f32>, OjasError> {
    same_len(op, &[x.len(), grad_y.len()], "silu input and grad")?;
    let n = x.len();
    let _parts = room_for(op, budget, n)?;
    let (x, grad_y) = (Arc::new(x), Arc::new(grad_y));
    let numerics = exec.numerics;
    elementwise(exec, n, move |range| {
        let pairs = x[range.clone()].iter().zip(&grad_y[range]);
        match numerics {
            Numerics::Exact => pairs
                .map(|(&v, &g)| {
                    let s = sigmoid(v);
                    g * s * (1.0 + v * (1.0 - s))
                })
                .collect(),
            Numerics::Fast => pairs
                .map(|(&v, &g)| {
                    let (s, rest) = sigmoid_pair_fast(v);
                    g * s * fma(v, rest, 1.0)
                })
                .collect(),
        }
    })
}

/// `a * b`, one rounding per value under either contract.
///
/// Mul, add and the value-residual blend are one memory pass over input
/// copies the caller has already made and charged, so they are computed in
/// place in one of those copies on the calling thread: no output buffer is
/// allocated and nothing is joined. A pool split cannot write into one
/// buffer (tasks are `'static` and the crate forbids `unsafe`), so it costs
/// a joined copy of the output; at `[1024, 2048]` on 6 threads the split
/// measured 1.04 ms for mul forward against 0.92 ms in place (min of 3
/// interleaved runs, M5 Pro).
pub(crate) fn mul_forward(
    op: &'static str,
    a: Vec<f32>,
    b: Vec<f32>,
) -> Result<Vec<f32>, OjasError> {
    same_len(op, &[a.len(), b.len()], "mul operand")?;
    let mut a = a;
    for (x, &y) in a.iter_mut().zip(&b) {
        *x *= y;
    }
    Ok(a)
}

/// `(grad_y * b, grad_y * a)`, written over `a` and `b` (see [`mul_forward`]).
pub(crate) fn mul_backward(
    op: &'static str,
    a: Vec<f32>,
    b: Vec<f32>,
    grad_y: Vec<f32>,
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    same_len(op, &[a.len(), b.len(), grad_y.len()], "mul grad")?;
    let (mut a, mut b) = (a, b);
    for ((x, y), &g) in a.iter_mut().zip(b.iter_mut()).zip(&grad_y) {
        let (av, bv) = (*x, *y);
        *x = g * bv;
        *y = g * av;
    }
    Ok((a, b))
}

/// `a + b`, written over `a` (see [`mul_forward`]).
pub(crate) fn add_forward(
    op: &'static str,
    a: Vec<f32>,
    b: Vec<f32>,
) -> Result<Vec<f32>, OjasError> {
    same_len(op, &[a.len(), b.len()], "residual add operand")?;
    let mut a = a;
    for (x, &y) in a.iter_mut().zip(&b) {
        *x += y;
    }
    Ok(a)
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
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: &[f32],
    attn: Vec<f32>,
    dims: GateDims,
) -> Result<Vec<f32>, OjasError> {
    gate_lengths(op, &dims, &input, &weight, bias, &attn)?;
    // The logits phase, then the output's task parts; the caller charges
    // the joined output.
    let work = add(op, logit_work(op, exec, &dims)?, attn.len())?;
    let _hold = room_for(op, budget, work)?;
    let gates = gate_values(op, exec, &dims, Arc::new(input), Arc::new(weight), bias)?;
    let attn = Arc::new(attn);
    let (heads, dh) = (dims.heads, dims.head_dim);
    let width = heads * dh;
    exec.rows(dims.rows, width, move |range| {
        let mut y = vec![0.0f32; range.len() * width];
        let src = &attn[range.start * width..range.end * width];
        let g = &gates[range.start * heads..range.end * heads];
        for ((dst, src), &g) in y.chunks_exact_mut(dh).zip(src.chunks_exact(dh)).zip(g) {
            for (out, &a) in dst.iter_mut().zip(src) {
                *out = a * g;
            }
        }
        Ok(y)
    })
}

/// `(grad_input, grad_weight, grad_bias, grad_attn)`.
type GateGrads = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// `gz = (sum_d grad_y · attn) · g · (1 - g)` per row and head, then
/// `grad_input = gz · W` and `grad_weight = gzᵀ · input` on the GEMM core,
/// `grad_bias` the column sums of `gz` in ascending row order, and
/// `grad_attn = grad_y · g`. Under [`Numerics::Exact`] every sum ascends
/// from `+0.0` without `mul_add`, which is the order of the scalar loop this
/// replaced; under [`Numerics::Fast`] the per-head dot uses
/// [`dot_lanes`]. `dims` comes from
/// [`ojas_core::per_head_sigmoid_gate_backward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: &[f32],
    attn: Vec<f32>,
    grad_y: Vec<f32>,
    dims: GateDims,
) -> Result<GateGrads, OjasError> {
    gate_lengths(op, &dims, &input, &weight, bias, &attn)?;
    if grad_y.len() != attn.len() {
        return Err(shape(op, "gate grad length does not match attn"));
    }
    let GateDims {
        rows,
        d_model: din,
        heads,
        head_dim: dh,
    } = dims;
    let attn_len = attn.len();
    let gz_len = product(op, &[rows, heads])?;
    // Charged at once: the logits phase; grad_attn's task parts (the caller
    // charges the joined grad_attn); gz's parts and joined copy; the larger
    // gradient product's scratch; and the three other gradients.
    let grads =
        gemm_scratch(op, exec, rows, heads, din)?.max(gemm_scratch(op, exec, heads, rows, din)?);
    let work = [
        logit_work(op, exec, &dims)?,
        attn.len(),
        gz_len,
        gz_len,
        grads,
        input.len(),
        weight.len(),
        heads,
    ]
    .into_iter()
    .try_fold(0usize, |total, n| add(op, total, n))?;
    let _hold = room_for(op, budget, work)?;
    let (input, weight) = (Arc::new(input), Arc::new(weight));
    let gates = gate_values(
        op,
        exec,
        &dims,
        Arc::clone(&input),
        Arc::clone(&weight),
        bias,
    )?;
    let (attn, grad_y) = (Arc::new(attn), Arc::new(grad_y));
    let numerics = exec.numerics;
    let width = heads * dh;
    let min_rows = (ROW_MIN_ELEMS / width.max(1)).max(1);
    let parts = exec.chunks(rows, min_rows, move |range| {
        let mut grad_attn = vec![0.0f32; range.len() * width];
        let mut gz = vec![0.0f32; range.len() * heads];
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
        (grad_attn, gz)
    })?;
    let (grad_attn, gz) = join2(parts, attn_len, gz_len);
    let mut grad_b = vec![0.0f32; heads];
    for row in gz.chunks_exact(heads) {
        for (slot, &v) in grad_b.iter_mut().zip(row) {
            *slot += v;
        }
    }
    let gz = Mat::row_major(Arc::new(gz), rows, heads);
    // grad_x[row, i] = sum_head gz[row, head] * W[head, i], head from 0.
    let grad_x = gemm(op, exec, &gz, &Mat::row_major(weight, heads, din))?;
    // grad_w[head, i] = sum_row gz[row, head] * x[row, i], row from 0.
    let grad_w = gemm(op, exec, &gz.t(), &Mat::row_major(input, rows, din))?;
    Ok((grad_x, grad_w, grad_b, grad_attn))
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
    input: Arc<Vec<f32>>,
    weight: Arc<Vec<f32>>,
    bias: &[f32],
) -> Result<Arc<Vec<f32>>, OjasError> {
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
            let a = Mat::row_major(Arc::new(ones_x), rows, width);
            let w = Mat::row_major(Arc::new(bias_w), heads, width);
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
    Ok(Arc::new(z))
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

/// `y = (1 - sigmoid(lambda)) * value + sigmoid(lambda) * value0`, written
/// over `value` (see [`mul_forward`]).
pub(crate) fn value_residual_forward(
    op: &'static str,
    value: Vec<f32>,
    value0: Vec<f32>,
    lambda: f32,
) -> Result<Vec<f32>, OjasError> {
    same_len(op, &[value.len(), value0.len()], "value residual operand")?;
    let s = sigmoid(lambda);
    if !s.is_finite() {
        return Err(nonfinite(op));
    }
    let mut value = value;
    for (v, &v0) in value.iter_mut().zip(&value0) {
        *v = (1.0 - s) * *v + s * v0;
    }
    Ok(value)
}

/// Values per block of the [`Numerics::Fast`] `lambda` gradient sum. Block
/// boundaries are fixed by the length alone, never by the task split.
const LAMBDA_BLOCK: usize = 1 << 12;

/// `(grad_value, grad_value0, grad_lambda)`.
///
/// Unlike [`mul_forward`] this runs in block chunks on the pool: the `f64`
/// sum below makes it compute-bound enough that the split pays for the
/// joined copy of both gradients.
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
    value: Vec<f32>,
    value0: Vec<f32>,
    lambda: f32,
    grad_y: Vec<f32>,
) -> Result<(Vec<f32>, Vec<f32>, f32), OjasError> {
    same_len(
        op,
        &[value.len(), value0.len(), grad_y.len()],
        "value residual grad",
    )?;
    let s = sigmoid(lambda);
    let n = value.len();
    let blocks = n.div_ceil(LAMBDA_BLOCK);
    // Two gradients' task parts and one f64 (two f32) per block; the caller
    // charges the joined gradients.
    let work = add(op, product(op, &[n, 2])?, product(op, &[blocks, 2])?)?;
    let _parts = room_for(op, budget, work)?;
    let (value, value0, grad_y) = (Arc::new(value), Arc::new(value0), Arc::new(grad_y));
    let numerics = exec.numerics;
    let parts = {
        let (value, value0, grad_y) =
            (Arc::clone(&value), Arc::clone(&value0), Arc::clone(&grad_y));
        let min_blocks = ROW_MIN_ELEMS.div_ceil(LAMBDA_BLOCK);
        exec.chunks(blocks, min_blocks, move |block_range| {
            let range = block_range.start * LAMBDA_BLOCK..(block_range.end * LAMBDA_BLOCK).min(n);
            let gy = &grad_y[range.clone()];
            let grad_v = gy.iter().map(|g| (1.0 - s) * g).collect();
            let grad_v0 = gy.iter().map(|g| s * g).collect();
            let sums = match numerics {
                Numerics::Exact => Vec::new(),
                Numerics::Fast => range
                    .clone()
                    .step_by(LAMBDA_BLOCK)
                    .map(|start| {
                        let end = (start + LAMBDA_BLOCK).min(n);
                        diff_dot_f64(&value[start..end], &value0[start..end], &grad_y[start..end])
                    })
                    .collect(),
            };
            (grad_v, grad_v0, sums)
        })?
    };
    let mut block_sums = Vec::with_capacity(blocks);
    let pairs = parts
        .into_iter()
        .map(|(gv, gv0, sums)| {
            block_sums.extend_from_slice(&sums);
            (gv, gv0)
        })
        .collect();
    let (grad_v, grad_v0) = join2(pairs, n, n);
    let grad_lambda = match numerics {
        Numerics::Exact => {
            let mut grad_s = 0.0f32;
            for ((&v, &v0), &g) in value.iter().zip(value0.iter()).zip(grad_y.iter()) {
                grad_s += (v0 - v) * g;
            }
            grad_s * s * (1.0 - s)
        }
        Numerics::Fast => {
            let grad_s = block_sums.iter().fold(0.0f64, |total, &b| total + b);
            let s = 1.0 / (1.0 + (-f64::from(lambda)).exp());
            (grad_s * s * (1.0 - s)) as f32
        }
    };
    Ok((grad_v, grad_v0, grad_lambda))
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
