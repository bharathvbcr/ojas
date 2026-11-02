//! Gauge-projected attention backward, full-range target-excluded cross-entropy,
//! symmetric per-channel `INT8` linear kernels with `i32` accumulator overflow guards
//! ($K \le 131,071$), and Wanda `2:4` structured-sparse slow-weight compression.

use ojas_core::{BackendId, Budget, DType, OjasError, Tensor};

/// Maximum safe inner dimension $K$ for symmetric `i8 x i8 -> i32` GEMM accumulation
/// without saturation: $\lfloor (2^{31} - 1) / 127^2 \rfloor = 133,128 \implies K_{\max} = 131,071 = 2^{17} - 1$.
pub const I8_ACCUMULATOR_MAX_K: usize = 131_071;

fn shape_err(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op,
        detail: detail.into(),
    }
}

fn nonfinite_err(op: &'static str) -> OjasError {
    OjasError::NonFinite { op }
}

/// Validates that `k_in <= I8_ACCUMULATOR_MAX_K` so `i8 x i8 -> i32` accumulation
/// can never wrap or saturate `i32::MAX`.
pub fn check_i8_accumulator_bound(op: &'static str, k_in: usize) -> Result<(), OjasError> {
    if k_in > I8_ACCUMULATOR_MAX_K {
        return Err(OjasError::OutOfRange {
            op,
            detail: format!(
                "inner dimension K={k_in} exceeds safe i32 accumulator bound I8_ACCUMULATOR_MAX_K={I8_ACCUMULATOR_MAX_K}"
            ),
        });
    }
    Ok(())
}

/// Full-range target-excluded cross-entropy mean forward + backward (`other_sum` activated
/// whenever `picked == max` across all margins $m \in (0, \infty)$).
///
/// Eliminates the `1.0f32` mantissa absorption window ($m \in [6.0, 20.78]$) where standard
/// `1.0 / sum - 1.0` loses up to 20 bits of precision before `sum == 1.0` collapses.
pub fn cross_entropy_mean_backward_exact_margin(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: Option<u32>,
    budget: &Budget,
) -> Result<(f32, Tensor), OjasError> {
    const OP: &str = "cross_entropy_mean_backward_exact_margin";
    if logits.dtype() != DType::F32 {
        return Err(OjasError::Dtype {
            op: OP,
            expected: DType::F32,
            got: logits.dtype(),
        });
    }
    if targets.dtype() != DType::U32 {
        return Err(OjasError::Dtype {
            op: OP,
            expected: DType::U32,
            got: targets.dtype(),
        });
    }
    let l_shape = logits.shape();
    if l_shape.len() != 2 {
        return Err(shape_err(OP, "logits must have rank 2 [rows, vocab]"));
    }
    let (rows, vocab) = (l_shape[0], l_shape[1]);
    if targets.shape() != [rows] || vocab == 0 {
        return Err(shape_err(
            OP,
            "targets shape must match [rows] and vocab > 0",
        ));
    }
    let logit_vals = logits.to_f32_vec()?;
    let target_vals = targets.to_u32_vec()?;

    let mut valid_count = 0usize;
    for (r, &t) in target_vals.iter().enumerate() {
        if ignore_index == Some(t) {
            continue;
        }
        if (t as usize) >= vocab {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("target {t} at row {r} exceeds vocab {vocab}"),
            });
        }
        valid_count += 1;
    }
    if valid_count == 0 {
        return Err(nonfinite_err(OP));
    }

    let denom = valid_count as f32;
    let mut grad = vec![0.0f32; rows * vocab];
    let mut total_loss = 0.0f32;

    for r in 0..rows {
        let row = &logit_vals[r * vocab..(r + 1) * vocab];
        if !row.iter().all(|v| v.is_finite()) {
            return Err(nonfinite_err(OP));
        }
        let t = target_vals[r];
        if ignore_index == Some(t) {
            continue;
        }
        let target_col = t as usize;
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let picked = row[target_col];
        let g_row = &mut grad[r * vocab..(r + 1) * vocab];

        if picked == max_val && vocab > 1 {
            // Target-excluded summation from 0.0f32: avoids 1.0f32 mantissa absorption across all margins m > 0.
            let mut s_other = 0.0f32;
            for (c, &z) in row.iter().enumerate() {
                if c != target_col {
                    let e = (z - max_val).exp();
                    g_row[c] = e;
                    s_other += e;
                }
            }
            if !s_other.is_finite() {
                return Err(nonfinite_err(OP));
            }
            let d = 1.0f32 + s_other;
            for (c, slot) in g_row.iter_mut().enumerate() {
                if c != target_col {
                    *slot = (*slot / d) / denom;
                }
            }
            g_row[target_col] = -(s_other / d) / denom;
            total_loss += s_other.ln_1p();
        } else {
            let mut sum = 0.0f32;
            for (c, &z) in row.iter().enumerate() {
                let e = (z - max_val).exp();
                g_row[c] = e;
                sum += e;
            }
            if !(sum.is_finite() && sum > 0.0) {
                return Err(nonfinite_err(OP));
            }
            for slot in g_row.iter_mut() {
                *slot = (*slot / sum) / denom;
            }
            g_row[target_col] -= 1.0 / denom;
            total_loss += max_val + sum.ln() - picked;
        }
    }

    let mean_loss = total_loss / denom;
    if !mean_loss.is_finite() {
        return Err(nonfinite_err(OP));
    }
    let grad_tensor = Tensor::from_f32(&grad, l_shape, budget)?;
    Ok((mean_loss, grad_tensor))
}

/// Gauge-projected causal SDPA backward across all sequence lengths ($T \le 256$ and $T > 256$).
///
/// Computes $\delta_i = \sum_{j \in \text{keys}(i)} P_{ij} dP_{ij}$ directly from the backward's
/// recomputed $P_{ij}$ and $dP_{ij}$ instead of $\langle dO_i, O_i \rangle$, guaranteeing
/// $\sum_j dS_{ij} = 0$ and key-shift gauge invariance $K \leftarrow K + \mathbf{1} c^\top$
/// even when the forward output tensor `O` was rounded to `bf16` or discarded.
pub fn causal_sdpa_backward_gauge_projected(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    lse: &Tensor,
    grad_output: &Tensor,
    window: Option<usize>,
    budget: &Budget,
) -> Result<(Tensor, Tensor, Tensor), OjasError> {
    const OP: &str = "causal_sdpa_backward_gauge_projected";
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4
        || k_shape.len() != 4
        || v_shape != k_shape
        || grad_output.shape() != q_shape
    {
        return Err(shape_err(OP, "invalid SDPA tensor shapes"));
    }
    let (batch, heads, time, dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let kv_heads = k_shape[1];
    if k_shape[0] != batch
        || k_shape[2] != time
        || k_shape[3] != dim
        || kv_heads == 0
        || heads % kv_heads != 0
    {
        return Err(shape_err(
            OP,
            "incompatible GQA head or sequence dimensions",
        ));
    }
    if lse.shape() != [batch, heads, time] {
        return Err(shape_err(OP, "lse shape must be [B, H, T]"));
    }
    if window == Some(0) {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: "window cannot be 0".to_string(),
        });
    }

    let q_vals = q.to_f32_vec()?;
    let k_vals = k.to_f32_vec()?;
    let v_vals = v.to_f32_vec()?;
    let lse_vals = lse.to_f32_vec()?;
    let gy_vals = grad_output.to_f32_vec()?;

    let scale = 1.0f32 / (dim as f32).sqrt();
    let rep = heads / kv_heads;
    let head_stride = time * dim;

    let mut grad_q = vec![0.0f32; q_vals.len()];
    let mut grad_k = vec![0.0f32; k_vals.len()];
    let mut grad_v = vec![0.0f32; v_vals.len()];
    let mut probs = vec![0.0f32; time];
    let mut dprobs = vec![0.0f32; time];

    for b in 0..batch {
        for h in 0..heads {
            let kv_h = h / rep;
            let q_plane = (b * heads + h) * head_stride;
            let kv_plane = (b * kv_heads + kv_h) * head_stride;
            let lse_plane = (b * heads + h) * time;

            for t in 0..time {
                let q_row = &q_vals[q_plane + t * dim..q_plane + (t + 1) * dim];
                let gy_row = &gy_vals[q_plane + t * dim..q_plane + (t + 1) * dim];
                let row_lse = lse_vals[lse_plane + t];
                if !row_lse.is_finite() {
                    return Err(nonfinite_err(OP));
                }
                let start_j = match window {
                    Some(w) => (t + 1).saturating_sub(w),
                    None => 0,
                };
                let keys = start_j..=t;

                // Gauge-projected row baseline: delta = sum_j P_{tj} * dP_{tj}
                let mut delta = 0.0f32;
                for j in keys.clone() {
                    let k_row = &k_vals[kv_plane + j * dim..kv_plane + (j + 1) * dim];
                    let v_row = &v_vals[kv_plane + j * dim..kv_plane + (j + 1) * dim];
                    let score: f32 =
                        q_row.iter().zip(k_row).map(|(&x, &y)| x * y).sum::<f32>() * scale;
                    let p = (score - row_lse).exp();
                    let dp: f32 = gy_row.iter().zip(v_row).map(|(&g, &vv)| g * vv).sum();
                    if !(p.is_finite() && dp.is_finite()) {
                        return Err(nonfinite_err(OP));
                    }
                    probs[j] = p;
                    dprobs[j] = dp;
                    delta += p * dp;
                }

                for j in keys {
                    let p = probs[j];
                    let ds = scale * p * (dprobs[j] - delta);
                    let k_row = &k_vals[kv_plane + j * dim..kv_plane + (j + 1) * dim];
                    let gq_row = &mut grad_q[q_plane + t * dim..q_plane + (t + 1) * dim];
                    let gk_row = &mut grad_k[kv_plane + j * dim..kv_plane + (j + 1) * dim];
                    let gv_row = &mut grad_v[kv_plane + j * dim..kv_plane + (j + 1) * dim];
                    for d_idx in 0..dim {
                        gq_row[d_idx] += ds * k_row[d_idx];
                        gk_row[d_idx] += ds * q_row[d_idx];
                        gv_row[d_idx] += p * gy_row[d_idx];
                    }
                }
            }
        }
    }

    Ok((
        Tensor::from_f32(&grad_q, q_shape, budget)?,
        Tensor::from_f32(&grad_k, k_shape, budget)?,
        Tensor::from_f32(&grad_v, v_shape, budget)?,
    ))
}

/// Symmetric per-output-channel `INT8` (`[-127, 127]`) quantized linear weight matrix.
#[derive(Clone, Debug)]
pub struct QuantizedWeightI8 {
    pub n_out: usize,
    pub k_in: usize,
    pub weights_i8: Vec<i8>,
    pub scales: Vec<f32>,
}

impl QuantizedWeightI8 {
    /// Quantizes a 2D `F32` weight tensor `[n_out, k_in]` into symmetric per-channel `INT8`
    /// after verifying the safe `i32` accumulator bound $K \le 131,071$.
    pub fn quantize(weight: &Tensor, _budget: &Budget) -> Result<Self, OjasError> {
        const OP: &str = "QuantizedWeightI8::quantize";
        if weight.dtype() != DType::F32 {
            return Err(OjasError::Dtype {
                op: OP,
                expected: DType::F32,
                got: weight.dtype(),
            });
        }
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(shape_err(OP, "weight must have rank 2 [n_out, k_in]"));
        }
        let (n_out, k_in) = (shape[0], shape[1]);
        check_i8_accumulator_bound(OP, k_in)?;
        let vals = weight.to_f32_vec()?;
        if !vals.iter().all(|v| v.is_finite()) {
            return Err(nonfinite_err(OP));
        }

        let mut weights_i8 = Vec::with_capacity(vals.len());
        let mut scales = Vec::with_capacity(n_out);
        for row in vals.chunks_exact(k_in) {
            let abs_max = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            scales.push(scale);
            let inv = if abs_max > 0.0 { 127.0 / abs_max } else { 0.0 };
            for &v in row {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                weights_i8.push(q);
            }
        }
        Ok(Self {
            n_out,
            k_in,
            weights_i8,
            scales,
        })
    }

    /// Reconstructs the dequantized `f32` weight matrix `[n_out, k_in]`.
    pub fn dequantize_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_out * self.k_in);
        for (r, &scale) in self.scales.iter().enumerate() {
            let row = &self.weights_i8[r * self.k_in..(r + 1) * self.k_in];
            for &q in row {
                out.push((q as f32) * scale);
            }
        }
        out
    }

    /// Executes dynamic per-row `INT8` input quantization + `i8 x i8 -> i32` GEMM accumulation
    /// + `f32` rescaling, enforcing `k_in <= I8_ACCUMULATOR_MAX_K`.
    pub fn linear_i8_forward(&self, input: &Tensor, budget: &Budget) -> Result<Tensor, OjasError> {
        const OP: &str = "QuantizedWeightI8::linear_i8_forward";
        check_i8_accumulator_bound(OP, self.k_in)?;
        let in_shape = input.shape();
        if in_shape.is_empty() || *in_shape.last().unwrap() != self.k_in {
            return Err(shape_err(OP, "input last dimension does not match k_in"));
        }
        let x_vals = input.to_f32_vec()?;
        if !x_vals.iter().all(|v| v.is_finite()) {
            return Err(nonfinite_err(OP));
        }
        let rows = x_vals.len() / self.k_in;
        let mut out = vec![0.0f32; rows * self.n_out];
        let mut x_q = vec![0i8; self.k_in];

        for (r, x_row) in x_vals.chunks_exact(self.k_in).enumerate() {
            let abs_max = x_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let x_scale = if abs_max > 0.0 { abs_max / 127.0 } else { 0.0 };
            let x_inv = if abs_max > 0.0 { 127.0 / abs_max } else { 0.0 };
            for (dst, &v) in x_q.iter_mut().zip(x_row) {
                *dst = (v * x_inv).round().clamp(-127.0, 127.0) as i8;
            }
            let out_row = &mut out[r * self.n_out..(r + 1) * self.n_out];
            for (o_idx, &w_scale) in self.scales.iter().enumerate() {
                let w_row = &self.weights_i8[o_idx * self.k_in..(o_idx + 1) * self.k_in];
                let mut acc_i32: i32 = 0;
                for (&xq, &wq) in x_q.iter().zip(w_row) {
                    acc_i32 = acc_i32
                        .checked_add((xq as i32) * (wq as i32))
                        .ok_or_else(|| OjasError::Backend {
                            id: BackendId::Cpu,
                            detail: format!("{OP}: i32 accumulator overflow"),
                        })?;
                }
                out_row[o_idx] = (acc_i32 as f32) * (x_scale * w_scale);
            }
        }

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(self.n_out);
        Tensor::from_f32(&out, &out_shape, budget)
    }
}

/// Wanda-style activation-weighted `2:4` semi-structured sparse weight matrix (`[n_out, k_in]`).
/// Every 4-element group along `k_in` retains at most 2 non-zero entries.
#[derive(Clone, Debug)]
pub struct Sparse24Weight {
    pub n_out: usize,
    pub k_in: usize,
    pub pruned_f32: Vec<f32>,
    pub mask_bits: Vec<u8>,
}

impl Sparse24Weight {
    /// Applies `2:4` structured pruning along `k_in` (requiring `k_in % 4 == 0`),
    /// ranking each 4-tuple by Wanda importance $S_{ij} = |W_{ij}| \cdot \|X_{\cdot, j}\|_2$.
    pub fn prune_24_wanda(weight: &Tensor, act_norms: Option<&[f32]>) -> Result<Self, OjasError> {
        const OP: &str = "Sparse24Weight::prune_24_wanda";
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(shape_err(OP, "weight must have rank 2 [n_out, k_in]"));
        }
        let (n_out, k_in) = (shape[0], shape[1]);
        if k_in == 0 || k_in % 4 != 0 {
            return Err(shape_err(
                OP,
                "k_in must be a positive multiple of 4 for 2:4 sparsity",
            ));
        }
        if let Some(norms) = act_norms {
            if norms.len() != k_in {
                return Err(shape_err(OP, "act_norms length must equal k_in"));
            }
        }
        let vals = weight.to_f32_vec()?;
        if !vals.iter().all(|v| v.is_finite()) {
            return Err(nonfinite_err(OP));
        }

        let mut pruned_f32 = vec![0.0f32; vals.len()];
        let groups_per_row = k_in / 4;
        let mut mask_bits = vec![0u8; n_out * groups_per_row];

        for r in 0..n_out {
            let row = &vals[r * k_in..(r + 1) * k_in];
            let dst_row = &mut pruned_f32[r * k_in..(r + 1) * k_in];
            for g in 0..groups_per_row {
                let base = g * 4;
                let mut scores: [(f32, usize); 4] = std::array::from_fn(|lane| {
                    let act = act_norms.map_or(1.0, |n| n[base + lane].abs().max(1e-6));
                    (row[base + lane].abs() * act, lane)
                });
                // Sort descending by Wanda score; tie-break deterministically by lower lane index.
                scores.sort_by(|a, b| {
                    b.0.partial_cmp(&a.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.1.cmp(&b.1))
                });
                let keep0 = scores[0].1;
                let keep1 = scores[1].1;
                dst_row[base + keep0] = row[base + keep0];
                dst_row[base + keep1] = row[base + keep1];
                mask_bits[r * groups_per_row + g] = (1u8 << keep0) | (1u8 << keep1);
            }
        }

        Ok(Self {
            n_out,
            k_in,
            pruned_f32,
            mask_bits,
        })
    }

    /// Verifies that every 4-element block has at most 2 non-zero entries and returns the zero fraction.
    pub fn verify_24_invariant(&self) -> bool {
        self.pruned_f32
            .chunks_exact(4)
            .all(|block| block.iter().filter(|&&v| v != 0.0).count() <= 2)
    }

    /// Executes `2:4` semi-structured linear forward (`input @ pruned_f32^T`).
    pub fn linear_24_forward(&self, input: &Tensor, budget: &Budget) -> Result<Tensor, OjasError> {
        let w_tensor = Tensor::from_f32(&self.pruned_f32, &[self.n_out, self.k_in], budget)?;
        let cpu = crate::CpuBackend::new(budget.clone());
        ojas_core::Backend::linear_forward(&cpu, input, &w_tensor)
    }

    /// Quantizes the `2:4`-pruned slow weights into symmetric per-channel `INT8`
    /// (`prune-then-quantize` pipeline for sleep consolidation).
    pub fn quantize_i8(&self, budget: &Budget) -> Result<QuantizedWeightI8, OjasError> {
        let w_tensor = Tensor::from_f32(&self.pruned_f32, &[self.n_out, self.k_in], budget)?;
        QuantizedWeightI8::quantize(&w_tensor, budget)
    }
}
