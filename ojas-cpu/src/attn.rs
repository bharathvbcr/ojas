//! Causal scaled dot-product attention.
//!
//! Layout is `[batch, heads, time, head_dim]`, the same layout
//! `F.scaled_dot_product_attention` uses after nanolab transposes. Scale is
//! `1/sqrt(head_dim)`. Query `t` attends to keys `0..=t` only. This function
//! does not apply the Metal head-dimension cap.

use ojas_core::{sdpa_scale, OjasError};

use crate::validate::{get, nonfinite, shape};

pub(crate) fn causal_sdpa_forward(
    op: &'static str,
    q: &[f32],
    q_shape: &[usize],
    k: &[f32],
    k_shape: &[usize],
    v: &[f32],
    v_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let dims = sdpa_dims(op, q_shape, k_shape, v_shape)?;
    let mut out = vec![0.0f32; q.len()];
    for b in 0..dims.batch {
        for h in 0..dims.heads {
            for t in 0..dims.time {
                let probs = causal_probs(op, q, k, b, h, t, &dims)?;
                for d in 0..dims.dim {
                    let mut acc = 0.0f32;
                    for j in 0..=t {
                        acc += get(op, &probs, j)? * at(op, v, b, h, j, d, &dims)?;
                    }
                    out[index(op, b, h, t, d, &dims)?] = acc;
                }
            }
        }
    }
    Ok(out)
}

/// `(grad_q, grad_k, grad_v)`.
type SdpaGrads = (Vec<f32>, Vec<f32>, Vec<f32>);

#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_sdpa_backward(
    op: &'static str,
    q: &[f32],
    q_shape: &[usize],
    k: &[f32],
    k_shape: &[usize],
    v: &[f32],
    v_shape: &[usize],
    grad_y: &[f32],
    grad_shape: &[usize],
) -> Result<SdpaGrads, OjasError> {
    if grad_shape != q_shape {
        return Err(shape(
            op,
            format!("sdpa grad shape {grad_shape:?} != query {q_shape:?}"),
        ));
    }
    let dims = sdpa_dims(op, q_shape, k_shape, v_shape)?;
    let mut grad_q = vec![0.0f32; q.len()];
    let mut grad_k = vec![0.0f32; k.len()];
    let mut grad_v = vec![0.0f32; v.len()];
    for b in 0..dims.batch {
        for h in 0..dims.heads {
            for t in 0..dims.time {
                let probs = causal_probs(op, q, k, b, h, t, &dims)?;
                let mut dprobs = vec![0.0f32; t + 1];
                for (j, slot) in dprobs.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..dims.dim {
                        dot += at(op, grad_y, b, h, t, d, &dims)? * at(op, v, b, h, j, d, &dims)?;
                    }
                    *slot = dot;
                }
                let mut expected = 0.0f32;
                for j in 0..=t {
                    expected += get(op, &probs, j)? * get(op, &dprobs, j)?;
                }
                for j in 0..=t {
                    let p = get(op, &probs, j)?;
                    let ds = p * (get(op, &dprobs, j)? - expected);
                    for d in 0..dims.dim {
                        let qi = index(op, b, h, t, d, &dims)?;
                        let ki = index(op, b, h, j, d, &dims)?;
                        grad_q[qi] += dims.scale * ds * get(op, k, ki)?;
                        grad_k[ki] += dims.scale * ds * get(op, q, qi)?;
                        grad_v[ki] += p * at(op, grad_y, b, h, t, d, &dims)?;
                    }
                }
            }
        }
    }
    Ok((grad_q, grad_k, grad_v))
}

struct Dims {
    batch: usize,
    heads: usize,
    time: usize,
    dim: usize,
    scale: f32,
}

fn sdpa_dims(
    op: &'static str,
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
) -> Result<Dims, OjasError> {
    if q_shape.len() != 4 {
        return Err(shape(
            op,
            format!("sdpa query rank {} != 4 [B, H, T, D]", q_shape.len()),
        ));
    }
    if k_shape != q_shape || v_shape != q_shape {
        return Err(shape(
            op,
            format!("sdpa shapes q {q_shape:?} k {k_shape:?} v {v_shape:?} differ"),
        ));
    }
    let dim = q_shape[3];
    let dim_u32 = u32::try_from(dim).map_err(|_| OjasError::OutOfRange {
        op,
        detail: format!("head dim {dim} does not fit in u32"),
    })?;
    let scale = sdpa_scale(dim_u32)?;
    Ok(Dims {
        batch: q_shape[0],
        heads: q_shape[1],
        time: q_shape[2],
        dim,
        scale,
    })
}

fn causal_probs(
    op: &'static str,
    q: &[f32],
    k: &[f32],
    b: usize,
    h: usize,
    t: usize,
    dims: &Dims,
) -> Result<Vec<f32>, OjasError> {
    let mut scores = vec![0.0f32; t + 1];
    let mut max_score = f32::NEG_INFINITY;
    for (j, slot) in scores.iter_mut().enumerate() {
        let mut dot = 0.0f32;
        for d in 0..dims.dim {
            dot += at(op, q, b, h, t, d, dims)? * at(op, k, b, h, j, d, dims)?;
        }
        let score = dot * dims.scale;
        if !score.is_finite() {
            return Err(nonfinite(op));
        }
        *slot = score;
        if score > max_score {
            max_score = score;
        }
    }
    let mut sum = 0.0f32;
    let mut probs = vec![0.0f32; t + 1];
    for j in 0..=t {
        let e = (scores[j] - max_score).exp();
        if !e.is_finite() {
            return Err(nonfinite(op));
        }
        probs[j] = e;
        sum += e;
    }
    if !(sum.is_finite() && sum > 0.0) {
        return Err(nonfinite(op));
    }
    for p in &mut probs {
        *p /= sum;
    }
    Ok(probs)
}

fn index(
    op: &'static str,
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    dims: &Dims,
) -> Result<usize, OjasError> {
    let stride_h = dims
        .time
        .checked_mul(dims.dim)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "sdpa stride overflows".to_string(),
        })?;
    let stride_b = dims
        .heads
        .checked_mul(stride_h)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "sdpa batch stride overflows".to_string(),
        })?;
    b.checked_mul(stride_b)
        .and_then(|base| base.checked_add(h.checked_mul(stride_h)?))
        .and_then(|base| base.checked_add(t.checked_mul(dims.dim)?))
        .and_then(|base| base.checked_add(d))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "sdpa index overflows".to_string(),
        })
}

fn at(
    op: &'static str,
    data: &[f32],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    dims: &Dims,
) -> Result<f32, OjasError> {
    get(op, data, index(op, b, h, t, d, dims)?)
}
