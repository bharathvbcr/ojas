//! Embed, RMSNorm, and causal attention in f32.
//!
//! These are the three kernels the decoder uses. They are local because
//! `ojas-cpu` is owned by another lane and is not a dependency here.

use ojas_core::{sdpa_scale, DType, OjasError, Tensor};

pub fn copy_f32(tensor: &Tensor, shape: &[usize]) -> Result<Vec<f32>, OjasError> {
    if tensor.dtype() != DType::F32 {
        return Err(OjasError::Dtype {
            op: "copy_f32",
            expected: DType::F32,
            got: tensor.dtype(),
        });
    }
    if tensor.shape() != shape {
        return Err(OjasError::Shape {
            op: "copy_f32",
            detail: format!("shape {:?} != {:?}", tensor.shape(), shape),
        });
    }
    let bytes = tensor.contiguous_bytes()?;
    if bytes.len() % 4 != 0 {
        return Err(OjasError::Shape {
            op: "copy_f32",
            detail: "f32 storage is not a multiple of 4".into(),
        });
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut a = [0u8; 4];
        a.copy_from_slice(chunk);
        out.push(f32::from_ne_bytes(a));
    }
    Ok(out)
}

pub fn embed(table: &[f32], n_embd: usize, ids: &[u32], out: &mut [f32]) -> Result<(), OjasError> {
    if n_embd == 0 || table.len() % n_embd != 0 {
        return Err(OjasError::Shape {
            op: "embed",
            detail: "embedding table is not divisible by n_embd".into(),
        });
    }
    let vocab = table.len() / n_embd;
    let vocab_u = u32::try_from(vocab).map_err(|_| OjasError::OutOfRange {
        op: "embed",
        detail: "vocab does not fit in u32".into(),
    })?;
    if out.len() != ids.len().saturating_mul(n_embd) {
        return Err(OjasError::Shape {
            op: "embed",
            detail: "embedding output length mismatch".into(),
        });
    }
    for (t, &id) in ids.iter().enumerate() {
        if id >= vocab_u {
            return Err(OjasError::OutOfRange {
                op: "embed",
                detail: format!("token id {id} >= vocab {vocab_u}"),
            });
        }
        let src = (id as usize) * n_embd;
        let dst = t * n_embd;
        out[dst..dst + n_embd].copy_from_slice(&table[src..src + n_embd]);
    }
    Ok(())
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<(), OjasError> {
    if x.len() != weight.len() || x.len() != out.len() || x.is_empty() {
        return Err(OjasError::Shape {
            op: "rms_norm",
            detail: format!(
                "lengths x {} weight {} out {}",
                x.len(),
                weight.len(),
                out.len()
            ),
        });
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(OjasError::OutOfRange {
            op: "rms_norm",
            detail: format!("eps {eps} is not a non-negative finite value"),
        });
    }
    let mut sum_sq = 0.0f64;
    for &v in x {
        sum_sq += f64::from(v) * f64::from(v);
    }
    let mean_sq = sum_sq / (x.len() as f64);
    let denom = mean_sq + f64::from(eps);
    if !(denom.is_finite() && denom > 0.0) {
        return Err(OjasError::NonFinite { op: "rms_norm" });
    }
    let inv = (1.0 / denom.sqrt()) as f32;
    if !inv.is_finite() {
        return Err(OjasError::NonFinite { op: "rms_norm" });
    }
    for i in 0..x.len() {
        let val = x[i] * inv * weight[i];
        if !val.is_finite() {
            return Err(OjasError::NonFinite { op: "rms_norm" });
        }
        out[i] = val;
    }
    Ok(())
}

pub fn linear(
    x: &[f32],
    weight: &[f32],
    in_dim: usize,
    out_dim: usize,
    y: &mut [f32],
) -> Result<(), OjasError> {
    if x.len() != in_dim || y.len() != out_dim {
        return Err(OjasError::Shape {
            op: "linear",
            detail: "linear input or output length mismatch".into(),
        });
    }
    let need = out_dim
        .checked_mul(in_dim)
        .ok_or_else(|| OjasError::OutOfRange {
            op: "linear",
            detail: "weight length overflows".into(),
        })?;
    if weight.len() != need {
        return Err(OjasError::Shape {
            op: "linear",
            detail: format!("weight len {} != {out_dim}*{in_dim}", weight.len()),
        });
    }
    if in_dim == 0 {
        y.fill(0.0);
        return Ok(());
    }
    for (yo, row) in y.iter_mut().zip(weight.chunks_exact(in_dim)) {
        let val = dot(x, row);
        if !val.is_finite() {
            return Err(OjasError::NonFinite { op: "linear" });
        }
        *yo = val;
    }
    Ok(())
}

/// One query against every cached key. `q` is `[n_embd]`. `k` and `v` are
/// `[seq, n_embd]` packed head-major inside each position.
pub fn attend_one(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>, OjasError> {
    if n_head == 0 || head_dim == 0 {
        return Err(OjasError::OutOfRange {
            op: "attend_one",
            detail: "n_head or head_dim is 0".into(),
        });
    }
    let n_embd = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| OjasError::OutOfRange {
            op: "attend_one",
            detail: "n_embd overflows".into(),
        })?;
    if q.len() != n_embd || k.len() != v.len() || k.len() % n_embd != 0 {
        return Err(OjasError::Shape {
            op: "attend_one",
            detail: "q, k, v layout mismatch".into(),
        });
    }
    let seq = k.len() / n_embd;
    if seq == 0 {
        return Err(OjasError::Shape {
            op: "attend_one",
            detail: "attention has no keys".into(),
        });
    }
    let scale = sdpa_scale(u32::try_from(head_dim).map_err(|_| OjasError::OutOfRange {
        op: "attend_one",
        detail: "head_dim exceeds u32".into(),
    })?)?;
    let mut out = vec![0.0f32; n_embd];
    for head in 0..n_head {
        let hs = head * head_dim;
        let qh = &q[hs..hs + head_dim];
        let mut scores = vec![0.0f32; seq];
        for (t, score) in scores.iter_mut().enumerate() {
            let base = t * n_embd + hs;
            *score = scale * dot(qh, &k[base..base + head_dim]);
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Err(OjasError::NonFinite { op: "attend_one" });
        }
        softmax(&mut scores)?;
        for (t, &p) in scores.iter().enumerate() {
            let base = t * n_embd + hs;
            let vh = &v[base..base + head_dim];
            for (o, &vd) in out[hs..hs + head_dim].iter_mut().zip(vh) {
                *o += p * vd;
            }
        }
    }
    Ok(out)
}

/// Causal self-attention over a full sequence. Position `t` attends to `0..=t`.
pub fn causal_self_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>, OjasError> {
    let n_embd = n_head
        .checked_mul(head_dim)
        .ok_or_else(|| OjasError::OutOfRange {
            op: "causal_self_attention",
            detail: "n_embd overflows".into(),
        })?;
    if n_embd == 0 || q.len() != k.len() || q.len() != v.len() || q.len() % n_embd != 0 {
        return Err(OjasError::Shape {
            op: "causal_self_attention",
            detail: "q, k, v layout mismatch".into(),
        });
    }
    let seq = q.len() / n_embd;
    let mut out = vec![0.0f32; q.len()];
    for t in 0..seq {
        let qs = t * n_embd;
        let ke = (t + 1) * n_embd;
        let row = attend_one(&q[qs..qs + n_embd], &k[..ke], &v[..ke], n_head, head_dim)?;
        out[qs..qs + n_embd].copy_from_slice(&row);
    }
    Ok(out)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax(scores: &mut [f32]) -> Result<(), OjasError> {
    if scores.is_empty() {
        return Err(OjasError::Shape {
            op: "softmax",
            detail: "empty scores".into(),
        });
    }
    let mut max = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > max {
            max = s;
        }
    }
    if !max.is_finite() {
        return Err(OjasError::NonFinite { op: "softmax" });
    }
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(OjasError::NonFinite { op: "softmax" });
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::Budget;

    #[test]
    fn embed_and_rms_and_causal_attention() {
        let table = [10.0f32, 11.0, 20.0, 21.0];
        let mut out = [0.0f32; 4];
        embed(&table, 2, &[1, 0], &mut out).unwrap();
        assert_eq!(out, [20.0, 21.0, 10.0, 11.0]);
        assert!(embed(&table, 2, &[2], &mut [0.0; 2]).is_err());

        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut y = [0.0f32; 2];
        let eps = 1e-6f32;
        rms_norm(&x, &w, eps, &mut y).unwrap();
        let scale = 1.0 / (12.5f32 + eps).sqrt();
        assert!((y[0] - 3.0 * scale).abs() < 1e-5);
        assert!((y[1] - 4.0 * scale).abs() < 1e-5);

        let q = [1.0f32, 1.0];
        let k = [1.0f32, 1.0];
        let v = [2.0f32, 4.0];
        let attn = causal_self_attention(&q, &k, &v, 1, 1).unwrap();
        assert!((attn[0] - 2.0).abs() < 1e-5);
        assert!((attn[1] - 3.0).abs() < 1e-5);

        let budget = Budget::new(1 << 20);
        let tensor = Tensor::from_f32(&[1.5, -2.0], &[2], &budget).unwrap();
        assert_eq!(copy_f32(&tensor, &[2]).unwrap(), vec![1.5, -2.0]);
    }

    #[test]
    fn linear_rows_and_zero_inputs() {
        let mut y = [9.0f32; 2];
        linear(&[1.0, 2.0], &[1.0, 0.0, 3.0, -1.0], 2, 2, &mut y).unwrap();
        assert_eq!(y, [1.0, 1.0]);
        let mut y = [9.0f32; 3];
        linear(&[], &[], 0, 3, &mut y).unwrap();
        assert_eq!(y, [0.0; 3]);
        assert!(linear(&[1.0], &[1.0], 1, 2, &mut [0.0; 2]).is_err());
    }

    #[test]
    fn softmax_refuses_nan_and_all_negative_infinity() {
        assert!(softmax(&mut [f32::NAN, 0.0]).is_err());
        assert!(softmax(&mut [f32::NEG_INFINITY; 2]).is_err());
        let mut s = [0.0f32, 0.0];
        softmax(&mut s).unwrap();
        assert_eq!(s, [0.5, 0.5]);
    }

    #[test]
    fn rms_norm_refuses_non_finite_and_handles_large_f64_sum_sq() {
        let mut y = [0.0f32; 2];
        assert!(rms_norm(&[f32::NAN, 1.0], &[1.0, 1.0], 1e-6, &mut y).is_err());
        assert!(rms_norm(&[f32::INFINITY, 1.0], &[1.0, 1.0], 1e-6, &mut y).is_err());
        assert!(rms_norm(&[1.0, 1.0], &[f32::NAN, 1.0], 1e-6, &mut y).is_err());

        // 1e19 squared is 1e38, which would overflow f32 when summed over multiple elements,
        // but succeeds with f64 accumulation.
        let large_x = [1e19f32, -1e19f32];
        let w = [1.0f32, 1.0f32];
        let mut out = [0.0f32; 2];
        rms_norm(&large_x, &w, 1e-6, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        let scale = 1.0 / (2e38f64 / 2.0 + 1e-6).sqrt() as f32;
        assert!((out[0] - 1e19 * scale).abs() < 1e13);
        assert!((out[1] - (-1e19 * scale)).abs() < 1e13);
    }

    #[test]
    fn linear_refuses_non_finite_outputs() {
        let mut y = [0.0f32; 1];
        assert!(linear(&[f32::NAN], &[1.0], 1, 1, &mut y).is_err());
        assert!(linear(&[1.0], &[f32::INFINITY], 1, 1, &mut y).is_err());
    }
}
