//! Embedding, pointwise ops, per-head gate, value residual, and cross-entropy.

use std::sync::Arc;

use ojas_core::{Budget, OjasError};

use crate::pool::{Exec, ROW_MIN_ELEMS};
use crate::validate::{all_finite, flat, get, nonfinite, product, room_for, same_shape, shape};

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

pub(crate) fn embedding_forward(
    op: &'static str,
    budget: &Budget,
    table: &[f32],
    table_shape: &[usize],
    ids: &[u32],
    id_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), OjasError> {
    let (vocab, dim) = table_rank(op, table_shape)?;
    check_ids(op, ids, vocab)?;
    let mut out_shape = id_shape.to_vec();
    out_shape.push(dim);
    let out_len = product(op, &[ids.len(), dim])?;
    let _hold = room_for(op, budget, out_len)?;
    let mut out = vec![0.0f32; out_len];
    for (n, &id) in ids.iter().enumerate() {
        let row = id as usize;
        for col in 0..dim {
            out[flat(op, n, col, dim)?] = get(op, table, flat(op, row, col, dim)?)?;
        }
    }
    Ok((out, out_shape))
}

pub(crate) fn embedding_backward(
    op: &'static str,
    budget: &Budget,
    table_shape: &[usize],
    ids: &[u32],
    id_shape: &[usize],
    grad_y: &[f32],
    grad_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let (vocab, dim) = table_rank(op, table_shape)?;
    check_ids(op, ids, vocab)?;
    let mut expect = id_shape.to_vec();
    expect.push(dim);
    if grad_shape != expect.as_slice() {
        return Err(shape(
            op,
            format!("embedding grad shape {grad_shape:?} != {expect:?}"),
        ));
    }
    let table_len = product(op, &[vocab, dim])?;
    let _hold = room_for(op, budget, table_len)?;
    let mut grad_table = vec![0.0f32; table_len];
    for (n, &id) in ids.iter().enumerate() {
        let row = id as usize;
        for col in 0..dim {
            let dst = flat(op, row, col, dim)?;
            grad_table[dst] += get(op, grad_y, flat(op, n, col, dim)?)?;
        }
    }
    Ok(grad_table)
}

fn table_rank(op: &'static str, shape_v: &[usize]) -> Result<(usize, usize), OjasError> {
    if shape_v.len() != 2 {
        return Err(shape(
            op,
            format!("embedding table rank {} != 2 [vocab, dim]", shape_v.len()),
        ));
    }
    Ok((shape_v[0], shape_v[1]))
}

fn check_ids(op: &'static str, ids: &[u32], vocab: usize) -> Result<(), OjasError> {
    let vocab_u64 = u64::try_from(vocab).map_err(|_| OjasError::OutOfRange {
        op,
        detail: "vocab does not fit in u64".to_string(),
    })?;
    for (n, &id) in ids.iter().enumerate() {
        if u64::from(id) >= vocab_u64 {
            return Err(OjasError::OutOfRange {
                op,
                detail: format!("token id {id} at {n} is outside vocab {vocab}"),
            });
        }
    }
    Ok(())
}

pub(crate) fn silu_forward(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * sigmoid(v)).collect()
}

pub(crate) fn silu_backward(
    op: &'static str,
    x: &[f32],
    grad_y: &[f32],
) -> Result<Vec<f32>, OjasError> {
    if grad_y.len() != x.len() {
        return Err(shape(op, "silu grad length does not match input"));
    }
    let mut out = Vec::with_capacity(x.len());
    for (&v, &g) in x.iter().zip(grad_y.iter()) {
        let s = sigmoid(v);
        out.push(g * s * (1.0 + v * (1.0 - s)));
    }
    Ok(out)
}

pub(crate) fn mul_forward(op: &'static str, a: &[f32], b: &[f32]) -> Result<Vec<f32>, OjasError> {
    if a.len() != b.len() {
        return Err(shape(op, "mul operands differ in length"));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

pub(crate) fn mul_backward(
    op: &'static str,
    a: &[f32],
    b: &[f32],
    grad_y: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    if a.len() != b.len() || grad_y.len() != a.len() {
        return Err(shape(op, "mul grad length does not match inputs"));
    }
    let grad_a: Vec<f32> = grad_y.iter().zip(b.iter()).map(|(g, bv)| g * bv).collect();
    let grad_b: Vec<f32> = grad_y.iter().zip(a.iter()).map(|(g, av)| g * av).collect();
    Ok((grad_a, grad_b))
}

pub(crate) fn add_forward(op: &'static str, a: &[f32], b: &[f32]) -> Result<Vec<f32>, OjasError> {
    if a.len() != b.len() {
        return Err(shape(op, "residual add operands differ in length"));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

/// `weight` is `[n_head, d_model]` (nn.Linear with bias).
/// `input` is `[..., d_model]`, `attn` is `[..., n_head, head_dim]`.
/// Output is `attn * sigmoid(input @ W^T + bias)`, with the sigmoid
/// broadcast over the head dimension. The output has the shape of `attn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_forward(
    op: &'static str,
    budget: &Budget,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: &[f32],
    bias_shape: &[usize],
    attn: &[f32],
    attn_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let layout = gate_layout(op, input_shape, weight_shape, bias_shape, attn_shape)?;
    let _hold = room_for(op, budget, attn.len())?;
    let mut y = vec![0.0f32; attn.len()];
    for row in 0..layout.rows {
        for head in 0..layout.heads {
            let g = gate_value(op, input, weight, bias, row, head, &layout)?;
            for d in 0..layout.dh {
                let ai = attn_index(op, row, head, d, &layout)?;
                y[ai] = get(op, attn, ai)? * g;
            }
        }
    }
    Ok(y)
}

/// `sigmoid(input[row] . weight[head] + bias[head])`.
fn gate_value(
    op: &'static str,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    row: usize,
    head: usize,
    layout: &GateLayout,
) -> Result<f32, OjasError> {
    let mut z = get(op, bias, head)?;
    for i in 0..layout.din {
        z += get(op, input, flat(op, row, i, layout.din)?)?
            * get(op, weight, flat(op, head, i, layout.din)?)?;
    }
    if !z.is_finite() {
        return Err(nonfinite(op));
    }
    Ok(sigmoid(z))
}

/// `(grad_input, grad_weight, grad_bias, grad_attn)`.
type GateGrads = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

#[allow(clippy::too_many_arguments)]
pub(crate) fn gate_backward(
    op: &'static str,
    budget: &Budget,
    input: &[f32],
    input_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    bias: &[f32],
    bias_shape: &[usize],
    attn: &[f32],
    attn_shape: &[usize],
    grad_y: &[f32],
    grad_shape: &[usize],
) -> Result<GateGrads, OjasError> {
    same_shape(op, attn_shape, grad_shape)?;
    let layout = gate_layout(op, input_shape, weight_shape, bias_shape, attn_shape)?;
    let scratch = input
        .len()
        .checked_add(weight.len())
        .and_then(|n| n.checked_add(layout.heads))
        .and_then(|n| n.checked_add(attn.len()))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "gate scratch length overflows".to_string(),
        })?;
    let _hold = room_for(op, budget, scratch)?;
    let mut grad_x = vec![0.0f32; input.len()];
    let mut grad_w = vec![0.0f32; weight.len()];
    let mut grad_b = vec![0.0f32; layout.heads];
    let mut grad_attn = vec![0.0f32; attn.len()];
    for row in 0..layout.rows {
        for (head, grad_bias) in grad_b.iter_mut().enumerate() {
            let g = gate_value(op, input, weight, bias, row, head, &layout)?;
            let mut grad_g = 0.0f32;
            for d in 0..layout.dh {
                let ai = attn_index(op, row, head, d, &layout)?;
                let gy = get(op, grad_y, ai)?;
                grad_attn[ai] = gy * g;
                grad_g += gy * get(op, attn, ai)?;
            }
            let grad_z = grad_g * g * (1.0 - g);
            *grad_bias += grad_z;
            for i in 0..layout.din {
                let xi = flat(op, row, i, layout.din)?;
                let wi = flat(op, head, i, layout.din)?;
                grad_x[xi] += grad_z * get(op, weight, wi)?;
                grad_w[wi] += grad_z * get(op, input, xi)?;
            }
        }
    }
    Ok((grad_x, grad_w, grad_b, grad_attn))
}

struct GateLayout {
    rows: usize,
    din: usize,
    heads: usize,
    dh: usize,
}

fn gate_layout(
    op: &'static str,
    input_shape: &[usize],
    weight_shape: &[usize],
    bias_shape: &[usize],
    attn_shape: &[usize],
) -> Result<GateLayout, OjasError> {
    if weight_shape.len() != 2 {
        return Err(shape(op, "gate weight must be [n_head, d_model]"));
    }
    let din = match input_shape.last().copied() {
        Some(dim) => dim,
        None => return Err(shape(op, "gate input rank 0")),
    };
    if weight_shape[1] != din {
        return Err(shape(
            op,
            format!("gate weight in {} != input dim {din}", weight_shape[1]),
        ));
    }
    let heads = weight_shape[0];
    if bias_shape.len() != 1 || bias_shape[0] != heads {
        return Err(shape(op, format!("gate bias {bias_shape:?} != [{heads}]")));
    }
    if attn_shape.len() != input_shape.len() + 1 {
        return Err(shape(op, "gate attn rank must be input rank + 1"));
    }
    let dh = match attn_shape.last().copied() {
        Some(dim) => dim,
        None => return Err(shape(op, "gate attn rank 0")),
    };
    if attn_shape[attn_shape.len() - 2] != heads {
        return Err(shape(op, "gate attn head axis != weight rows"));
    }
    if attn_shape[..attn_shape.len() - 2] != input_shape[..input_shape.len() - 1] {
        return Err(shape(op, "gate attn prefix does not match input prefix"));
    }
    let rows = product(op, &input_shape[..input_shape.len() - 1])?;
    if rows == 0 {
        return Err(shape(op, "empty tensor"));
    }
    Ok(GateLayout {
        rows,
        din,
        heads,
        dh,
    })
}

fn attn_index(
    op: &'static str,
    row: usize,
    head: usize,
    d: usize,
    layout: &GateLayout,
) -> Result<usize, OjasError> {
    let per_row = layout
        .heads
        .checked_mul(layout.dh)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "gate row stride overflows".to_string(),
        })?;
    row.checked_mul(per_row)
        .and_then(|base| base.checked_add(head.checked_mul(layout.dh)?))
        .and_then(|base| base.checked_add(d))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "gate index overflows".to_string(),
        })
}

/// `y = (1 - sigmoid(lambda)) * value + sigmoid(lambda) * value0`.
pub(crate) fn value_residual_forward(
    op: &'static str,
    value: &[f32],
    value0: &[f32],
    lambda: f32,
) -> Result<(Vec<f32>, f32), OjasError> {
    if value.len() != value0.len() {
        return Err(shape(op, "value residual operands differ in length"));
    }
    let s = sigmoid(lambda);
    if !s.is_finite() {
        return Err(nonfinite(op));
    }
    let y = value
        .iter()
        .zip(value0.iter())
        .map(|(v, v0)| (1.0 - s) * v + s * v0)
        .collect();
    Ok((y, s))
}

pub(crate) fn value_residual_backward(
    op: &'static str,
    value: &[f32],
    value0: &[f32],
    lambda: f32,
    grad_y: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, f32), OjasError> {
    if value.len() != value0.len() || grad_y.len() != value.len() {
        return Err(shape(op, "value residual grad length does not match"));
    }
    let s = sigmoid(lambda);
    let mut grad_v = Vec::with_capacity(value.len());
    let mut grad_v0 = Vec::with_capacity(value.len());
    let mut grad_s = 0.0f32;
    for i in 0..value.len() {
        let gy = get(op, grad_y, i)?;
        grad_v.push((1.0 - s) * gy);
        grad_v0.push(s * gy);
        grad_s += (value0[i] - value[i]) * gy;
    }
    let grad_lambda = grad_s * s * (1.0 - s);
    Ok((grad_v, grad_v0, grad_lambda))
}

/// Mean cross-entropy. `ignore` positions are dropped. The denominator is the
/// valid-row count. An empty valid set matches torch and is
/// [`OjasError::NonFinite`] (the mean is NaN). It is not a finite loss of 0.
/// Rows run in chunks on the pool. Each valid row's loss term comes back
/// separately and the terms are added in increasing row order on the
/// caller, so the loss bits do not depend on the chunking.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cross_entropy(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    logits: Vec<f32>,
    logit_shape: &[usize],
    targets: Vec<u32>,
    target_shape: &[usize],
    ignore: Option<u32>,
) -> Result<(f32, Vec<f32>), OjasError> {
    let vocab = match logit_shape.last().copied() {
        Some(dim) => dim,
        None => return Err(shape(op, "cross-entropy logits rank 0")),
    };
    let prefix = &logit_shape[..logit_shape.len() - 1];
    if target_shape != prefix {
        return Err(shape(
            op,
            format!("targets {target_shape:?} != logits prefix {prefix:?}"),
        ));
    }
    let rows = product(op, prefix)?;
    if rows != targets.len() {
        return Err(shape(op, "target count does not match logits prefix"));
    }
    let mut n_valid: u32 = 0;
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
                op,
                detail: format!("target {target} at {n} is outside vocab {vocab}"),
            });
        }
        n_valid = n_valid
            .checked_add(1)
            .ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: "valid target count overflows".to_string(),
            })?;
    }
    if n_valid == 0 {
        return Err(nonfinite(op));
    }
    let denom = n_valid as f32;
    let grad_len = product(op, &[rows, vocab])?;
    // One exp buffer stays live with the row gradient. The stored result of
    // the forward pass is only the scalar loss.
    let scratch = grad_len
        .checked_add(vocab)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "cross-entropy scratch length overflows".to_string(),
        })?;
    let _hold = room_for(op, budget, scratch)?;
    let (logits, targets) = (Arc::new(logits), Arc::new(targets));
    let min_rows = (ROW_MIN_ELEMS / vocab.max(1)).max(1);
    let parts = exec.chunks(rows, min_rows, move |range| {
        let mut grad = vec![0.0f32; range.len() * vocab];
        let mut terms = Vec::with_capacity(range.len());
        let mut exps = vec![0.0f32; vocab];
        for n in range.clone() {
            let target = targets[n];
            if ignore == Some(target) {
                continue;
            }
            let class = target as usize;
            let mut max_logit = f32::NEG_INFINITY;
            let row =
                logits
                    .get(n * vocab..(n + 1) * vocab)
                    .ok_or_else(|| OjasError::OutOfRange {
                        op,
                        detail: "cross-entropy row exceeds logits".to_string(),
                    })?;
            for &value in row {
                if value > max_logit {
                    max_logit = value;
                }
            }
            let mut sum = 0.0f32;
            for (col, &value) in row.iter().enumerate() {
                let e = (value - max_logit).exp();
                if !e.is_finite() {
                    return Err(nonfinite(op));
                }
                exps[col] = e;
                sum += e;
            }
            if !(sum.is_finite() && sum > 0.0) {
                return Err(nonfinite(op));
            }
            terms.push(max_logit + sum.ln() - row[class]);
            let base = (n - range.start) * vocab;
            for col in 0..vocab {
                let p = exps[col] / sum;
                grad[base + col] = p / denom;
            }
            grad[base + class] -= 1.0 / denom;
        }
        Ok((grad, terms))
    })?;
    let mut grad = Vec::with_capacity(rows.saturating_mul(vocab));
    let mut total = 0.0f32;
    for part in parts {
        let (rows_grad, terms) = part?;
        grad.extend_from_slice(&rows_grad);
        for term in terms {
            total += term;
        }
    }
    let loss = total / denom;
    if !loss.is_finite() || !all_finite(&grad) {
        return Err(nonfinite(op));
    }
    Ok((loss, grad))
}
