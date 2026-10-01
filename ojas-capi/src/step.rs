//! One training step on the caller's payload.
//!
//! Mode [`MODE_LOGITS`] is mean cross-entropy of the supplied logits.
//! Mode [`MODE_TOKENS`] is a linear map of the u16 tokens (`weight[c] = c + 1`)
//! followed by that same cross-entropy. Both paths call [`ojas_cpu::CpuBackend`].
//! The gradient norm is `clip_grad_norm` of the cross-entropy backward, and
//! `lr` is the learning rate AdamW applied to a one-element parameter.
//! Nothing is written into the session.

use ojas_core::{AdamWConfig, Backend, Budget, DType, Tensor};
use ojas_cpu::CpuBackend;

pub const MODE_HEADER: u32 = 0;
pub const MODE_LOGITS: u32 = 1;
pub const MODE_TOKENS: u32 = 2;

/// Token-mode targets are u16, so no class past this index can be a target.
pub const MAX_TOKEN_CLASSES: u32 = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub lr: f32,
}

pub struct StepInput<'a> {
    pub mode: u32,
    pub batch: u32,
    pub seq: u32,
    pub n_classes: u32,
    pub step: u32,
    pub lr: f32,
    pub logits: &'a [f32],
    pub tokens: &'a [u16],
    pub targets_u32: &'a [u32],
    pub targets_u16: &'a [u16],
}

pub fn step(input: StepInput<'_>) -> Result<StepStats, String> {
    if input.mode == MODE_HEADER {
        return Err("step: shape: step payload is missing logits".to_string());
    }
    let rows = rows(input.batch, input.seq)?;
    if !input.lr.is_finite() || input.lr < 0.0 {
        return Err("step: out of range: lr must be finite and >= 0".to_string());
    }
    if input.n_classes == 0 {
        return Err("step: shape: n_classes is 0".to_string());
    }
    let n_classes = input.n_classes as usize;
    let budget = Budget::new(1 << 30);
    let cpu = CpuBackend::new(budget.clone());
    let logits = match input.mode {
        MODE_LOGITS => {
            if input.logits.len() != rows * n_classes {
                return Err(format!(
                    "step: shape: logits len {} != {rows}*{n_classes}",
                    input.logits.len()
                ));
            }
            if input.targets_u32.len() != rows {
                return Err(format!(
                    "step: shape: targets len {} != {rows}",
                    input.targets_u32.len()
                ));
            }
            Tensor::from_f32(input.logits, &[rows, n_classes], &budget).map_err(show)?
        }
        MODE_TOKENS => {
            if input.n_classes > MAX_TOKEN_CLASSES {
                return Err(format!(
                    "step: shape: n_classes {} exceeds {MAX_TOKEN_CLASSES} for u16 targets",
                    input.n_classes
                ));
            }
            if input.tokens.len() != rows || input.targets_u16.len() != rows {
                return Err(format!(
                    "step: shape: token buffer len {}/{} != {rows}",
                    input.tokens.len(),
                    input.targets_u16.len()
                ));
            }
            let x: Vec<f32> = input.tokens.iter().map(|&t| f32::from(t)).collect();
            let weight: Vec<f32> = (0..n_classes).map(|c| (c as f32) + 1.0).collect();
            let x = Tensor::from_f32(&x, &[rows, 1], &budget).map_err(show)?;
            let weight = Tensor::from_f32(&weight, &[n_classes, 1], &budget).map_err(show)?;
            cpu.linear_forward(&x, &weight).map_err(show)?
        }
        other => return Err(format!("step: shape: unknown mode {other}")),
    };
    let targets: Vec<u32> = if input.mode == MODE_LOGITS {
        input.targets_u32.to_vec()
    } else {
        input.targets_u16.iter().map(|&t| u32::from(t)).collect()
    };
    let targets = Tensor::from_u32(&targets, &[rows], &budget).map_err(show)?;
    let loss_t = cpu
        .cross_entropy_mean_forward(&logits, &targets, None)
        .map_err(show)?;
    let mut grad = cpu
        .cross_entropy_mean_backward(&logits, &targets, None)
        .map_err(show)?;
    let grad_norm = cpu.clip_grad_norm(std::slice::from_mut(&mut grad), 1.0).map_err(show)?;
    let loss = scalar(&loss_t)?;
    // The learning rate is applied to a one-element parameter whose gradient
    // is the scalar loss, so a non-finite lr fails inside ojas-cpu.
    let mut param = Tensor::from_f32(&[0.0], &[1], &budget).map_err(show)?;
    let grad_scalar = Tensor::from_f32(&[loss], &[1], &budget).map_err(show)?;
    let mut moment1 = Tensor::zeros(&[1], DType::F32, &budget).map_err(show)?;
    let mut moment2 = Tensor::zeros(&[1], DType::F32, &budget).map_err(show)?;
    let config = AdamWConfig::nanolab(f64::from(input.lr), 0.0);
    cpu.adamw_step(
        &mut param,
        &grad_scalar,
        &mut moment1,
        &mut moment2,
        u64::from(input.step),
        config,
    )
    .map_err(show)?;
    if !loss.is_finite() || !grad_norm.is_finite() {
        return Err("step: non-finite value".to_string());
    }
    Ok(StepStats {
        loss,
        grad_norm,
        lr: input.lr,
    })
}

pub(crate) fn rows(batch: u32, seq: u32) -> Result<usize, String> {
    if batch == 0 || seq == 0 {
        return Err("step: shape: batch and seq must be non-zero".to_string());
    }
    usize::try_from(batch)
        .ok()
        .and_then(|b| usize::try_from(seq).ok().and_then(|s| b.checked_mul(s)))
        .ok_or_else(|| "step: shape: batch * seq overflows".to_string())
}

fn scalar(tensor: &Tensor) -> Result<f32, String> {
    let values = tensor.to_f32_vec().map_err(show)?;
    if values.len() != 1 {
        return Err(format!("step: shape: loss rank has {} values", values.len()));
    }
    Ok(values[0])
}

fn show(err: ojas_core::OjasError) -> String {
    format!("step: {err}")
}
