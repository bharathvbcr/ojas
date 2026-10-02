//! Decode-only f32 kernels: a one-row linear and one query against the KV
//! cache. Embedding and RMSNorm go through `ojas_cpu::CpuBackend`.

use ojas_core::{sdpa_scale, OjasError};
#[cfg(test)]
use ojas_core::{DType, Tensor};

#[cfg(test)]
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
    Ok(tensor.f32_slice()?.to_vec())
}

/// `y[row] = dot(x, weight[row])` for each of `out_dim` rows of the
/// row-major `[out_dim, in_dim]` `weight`. Each dot product accumulates in
/// index order from `0.0`. A non-finite output is [`OjasError::NonFinite`].
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
    if in_dim == 0 {
        return Err(OjasError::Shape {
            op: "linear",
            detail: "in_dim is 0".into(),
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
    for (yo, row) in y.iter_mut().zip(weight.chunks_exact(in_dim)) {
        let val = dot(x, row);
        if !val.is_finite() {
            return Err(OjasError::NonFinite { op: "linear" });
        }
        *yo = val;
    }
    Ok(())
}

/// One query against every cached key. `q` is `[n_head * head_dim]`. `k`
/// and `v` are `[seq, n_kv_head * head_dim]`, head-major inside each
/// position. Query head `h` reads KV head `h / (n_head / n_kv_head)`, which
/// is torch's `repeat_interleave(rep, dim=1)` in nanolab's GQA path.
pub fn attend_one(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>, OjasError> {
    if n_head == 0 || n_kv_head == 0 || head_dim == 0 {
        return Err(OjasError::OutOfRange {
            op: "attend_one",
            detail: "n_head, n_kv_head or head_dim is 0".into(),
        });
    }
    if !n_head.is_multiple_of(n_kv_head) {
        return Err(OjasError::Shape {
            op: "attend_one",
            detail: format!("n_head {n_head} is not a multiple of n_kv_head {n_kv_head}"),
        });
    }
    let rep = n_head / n_kv_head;
    let width = |heads: usize| {
        heads
            .checked_mul(head_dim)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "attend_one",
                detail: "head width overflows".into(),
            })
    };
    let q_width = width(n_head)?;
    let kv_width = width(n_kv_head)?;
    if q.len() != q_width || k.len() != v.len() || !k.len().is_multiple_of(kv_width) {
        return Err(OjasError::Shape {
            op: "attend_one",
            detail: "q, k, v layout mismatch".into(),
        });
    }
    let seq = k.len() / kv_width;
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
    let mut out = vec![0.0f32; q_width];
    for head in 0..n_head {
        let hs = head * head_dim;
        let ks = (head / rep) * head_dim;
        let qh = &q[hs..hs + head_dim];
        let mut scores = vec![0.0f32; seq];
        for (t, score) in scores.iter_mut().enumerate() {
            let base = t * kv_width + ks;
            *score = scale * dot(qh, &k[base..base + head_dim]);
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Err(OjasError::NonFinite { op: "attend_one" });
        }
        softmax(&mut scores)?;
        for (t, &p) in scores.iter().enumerate() {
            let base = t * kv_width + ks;
            let vh = &v[base..base + head_dim];
            for (o, &vd) in out[hs..hs + head_dim].iter_mut().zip(vh) {
                *o += p * vd;
            }
        }
    }
    Ok(out)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
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
    fn attend_one_mixes_every_cached_key_and_copy_f32_checks_shape() {
        let attn = attend_one(&[1.0], &[1.0, 1.0], &[2.0, 4.0], 1, 1, 1).unwrap();
        assert!((attn[0] - 3.0).abs() < 1e-5, "{attn:?}");
        assert!(attend_one(&[1.0], &[], &[], 1, 1, 1).is_err());
        assert!(attend_one(&[1.0, 1.0], &[1.0], &[1.0], 1, 1, 1).is_err());
        assert!(attend_one(&[1.0; 3], &[1.0], &[1.0], 3, 2, 1).is_err());

        // Two query heads share one KV head: both read the same values.
        let gqa = attend_one(&[1.0, -1.0], &[1.0, 1.0], &[2.0, 4.0], 2, 1, 1).unwrap();
        assert!(
            (gqa[0] - 3.0).abs() < 1e-5 && (gqa[1] - 3.0).abs() < 1e-5,
            "{gqa:?}"
        );

        let budget = Budget::new(1 << 20);
        let tensor = Tensor::from_f32(&[1.5, -2.0], &[2], &budget).unwrap();
        assert_eq!(copy_f32(&tensor, &[2]).unwrap(), vec![1.5, -2.0]);
        assert!(copy_f32(&tensor, &[1, 2]).is_err());
    }

    #[test]
    fn linear_rows_and_zero_inputs() {
        let mut y = [9.0f32; 2];
        linear(&[1.0, 2.0], &[1.0, 0.0, 3.0, -1.0], 2, 2, &mut y).unwrap();
        assert_eq!(y, [1.0, 1.0]);
        let mut y = [9.0f32; 3];
        assert!(linear(&[], &[], 0, 3, &mut y).is_err());
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
    fn linear_refuses_non_finite_outputs() {
        let mut y = [0.0f32; 1];
        assert!(linear(&[f32::NAN], &[1.0], 1, 1, &mut y).is_err());
        assert!(linear(&[1.0], &[f32::INFINITY], 1, 1, &mut y).is_err());
    }
}
