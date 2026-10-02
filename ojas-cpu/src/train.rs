//! Gradient accumulation and the nanolab `muon_ns5_adamw` parameter split.
//!
//! Accumulation sums micro-batch gradients in `f32` in the order they are
//! added, then divides by the count once. It does not scale each micro-batch
//! before the sum. A count of 0 is an error. The count must stay exact in
//! `f32` (`<= 2^24`).
//!
//! Grouping matches `optim.py` `_split_params` / `build_optimizers` for
//! `muon_ns5_adamw`, without a width multiplier (the tiny reference uses
//! `width_mult = 1`):
//! - `ndim >= 2` and not an embedding: Muon NS5, decoupled weight decay 0.1
//! - embeddings, including the tied head: AdamW, weight decay 0
//! - biases, norm weights, and every other vector: AdamW, weight decay 0
//!
//! One [`HybridOptimizer::step`] consumes the already-reduced gradient and
//! advances every group. A non-finite gradient, or a step counter of
//! `u64::MAX`, returns before any parameter or moment is written.

use std::sync::Arc;

use ojas_core::{clip_scale, next_step, AdamWConfig, MuonNs5Config, Numerics, OjasError};

use crate::optim::{adamw, muon_ns5, total_norm};
use crate::pool::{Exec, Pool};
use crate::schedule::scaled_lr;
use crate::validate::{all_finite, nonfinite};

/// Muon decoupled weight decay. Nanolab `weight_decay` default, 2-D only.
pub const MUON_WEIGHT_DECAY: f64 = 0.1;

/// AdamW weight decay on embeddings and vectors in the Muon hybrid.
pub const ADAM_HYBRID_WEIGHT_DECAY: f64 = 0.0;

/// Nanolab `muon_momentum` default used by this hybrid step.
pub const MUON_MOMENTUM: f64 = 0.99;

/// Largest accumulation count that is an exact `f32`.
const ACCUM_COUNT_MAX: u64 = 1 << 24;

/// Which optimizer owns a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimGroup {
    /// Hidden matrix. Muon NS5, weight decay [`MUON_WEIGHT_DECAY`].
    MuonMatrix,
    /// Token embedding or tied head. AdamW, weight decay 0.
    AdamEmbedding,
    /// Bias, norm scale, or any other vector. AdamW, weight decay 0.
    AdamVector,
}

/// `embedding` wins over rank: an embedding matrix does not go to Muon.
pub fn optim_group(ndim: usize, embedding: bool) -> OptimGroup {
    if embedding {
        OptimGroup::AdamEmbedding
    } else if ndim >= 2 {
        OptimGroup::MuonMatrix
    } else {
        OptimGroup::AdamVector
    }
}

/// Running sum of micro-batch gradients.
#[derive(Clone, Debug)]
pub struct GradAccumulator {
    sum: Vec<f32>,
    count: u64,
}

impl GradAccumulator {
    pub fn new(width: usize) -> Result<Self, OjasError> {
        if width == 0 {
            return Err(OjasError::Shape {
                op: "grad_accum",
                detail: "empty tensor".to_string(),
            });
        }
        Ok(Self {
            sum: zero_f32("grad_accum", width)?,
            count: 0,
        })
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Add one micro-batch gradient. On error the sum and count stay as they were.
    pub fn add(&mut self, grad: &[f32]) -> Result<(), OjasError> {
        const OP: &str = "grad_accum";
        if grad.len() != self.sum.len() {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("grad len {} != {}", grad.len(), self.sum.len()),
            });
        }
        if !all_finite(grad) {
            return Err(nonfinite(OP));
        }
        let next_count = self
            .count
            .checked_add(1)
            .ok_or_else(|| OjasError::OutOfRange {
                op: OP,
                detail: "accumulation count overflows".to_string(),
            })?;
        if next_count > ACCUM_COUNT_MAX {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("accumulation count {next_count} is not an exact f32"),
            });
        }
        let mut next = Vec::new();
        try_reserve_f32(OP, &mut next, self.sum.len())?;
        next.extend_from_slice(&self.sum);
        for (slot, value) in next.iter_mut().zip(grad.iter()) {
            *slot += *value;
            if !slot.is_finite() {
                return Err(nonfinite(OP));
            }
        }
        self.sum = next;
        self.count = next_count;
        Ok(())
    }

    /// Divide the sum by the count once. Count 0 is an error. The sum is kept.
    pub fn mean(&self) -> Result<Vec<f32>, OjasError> {
        const OP: &str = "grad_accum";
        if self.count == 0 {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "accumulation count 0".to_string(),
            });
        }
        let k = self.count as f32;
        let mut out = Vec::new();
        try_reserve_f32(OP, &mut out, self.sum.len())?;
        for value in &self.sum {
            let scaled = *value / k;
            if !scaled.is_finite() {
                return Err(nonfinite(OP));
            }
            out.push(scaled);
        }
        Ok(out)
    }
}

/// Sum `parts` in order, then divide by `parts.len()` once.
///
/// An empty slice is accumulation count 0.
pub fn mean_micrograds(parts: &[&[f32]]) -> Result<Vec<f32>, OjasError> {
    if parts.is_empty() {
        return Err(OjasError::OutOfRange {
            op: "grad_accum",
            detail: "accumulation count 0".to_string(),
        });
    }
    let mut acc = GradAccumulator::new(parts[0].len())?;
    for part in parts {
        acc.add(part)?;
    }
    acc.mean()
}

/// Global-norm clip. When the coefficient is 1 the gradient bits are unchanged.
/// A failure leaves `grads` as it was.
pub fn clip_grads(grads: &mut [Vec<f32>], max_norm: f32) -> Result<f32, OjasError> {
    const OP: &str = "clip_grad_norm";
    if grads.is_empty() {
        return Err(OjasError::Shape {
            op: OP,
            detail: "empty tensor".to_string(),
        });
    }
    let norm = total_norm(OP, grads)?;
    let scale = clip_scale(max_norm, norm)?;
    if scale < 1.0 {
        let mut scaled = Vec::with_capacity(grads.len());
        for part in grads.iter() {
            let mut next = Vec::with_capacity(part.len());
            for value in part {
                let y = *value * scale;
                if !y.is_finite() {
                    return Err(nonfinite(OP));
                }
                next.push(y);
            }
            scaled.push(next);
        }
        for (grad, next) in grads.iter_mut().zip(scaled) {
            *grad = next;
        }
    }
    Ok(norm)
}

/// One parameter plus the moments the hybrid step reads and writes.
#[derive(Clone, Debug)]
pub struct HybridParam {
    pub group: OptimGroup,
    pub initial_lr: f64,
    /// Matrix rows. Unused for AdamW groups.
    pub rows: usize,
    /// Matrix columns. Unused for AdamW groups.
    pub cols: usize,
    pub shape: Vec<usize>,
    pub param: Vec<f32>,
    pub grad: Vec<f32>,
    pub moment1: Vec<f32>,
    pub moment2: Vec<f32>,
}

impl HybridParam {
    pub fn new(
        group: OptimGroup,
        initial_lr: f64,
        shape: &[usize],
        param: Vec<f32>,
    ) -> Result<Self, OjasError> {
        const OP: &str = "hybrid_step";
        let width = shape_product(OP, shape)?;
        if param.len() != width {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("param len {} != shape product {width}", param.len()),
            });
        }
        if !initial_lr.is_finite() {
            return Err(nonfinite(OP));
        }
        let (rows, cols) = match group {
            OptimGroup::MuonMatrix => {
                if shape.len() != 2 {
                    return Err(OjasError::Shape {
                        op: OP,
                        detail: "muon parameter must be a matrix".to_string(),
                    });
                }
                (shape[0], shape[1])
            }
            OptimGroup::AdamEmbedding | OptimGroup::AdamVector => (0, 0),
        };
        Ok(Self {
            group,
            initial_lr,
            rows,
            cols,
            shape: shape.to_vec(),
            param,
            grad: zero_f32(OP, width)?,
            moment1: zero_f32(OP, width)?,
            moment2: zero_f32(OP, width)?,
        })
    }
}

/// Muon matrices plus AdamW embeddings and vectors. One step index for every group.
#[derive(Clone, Debug)]
pub struct HybridOptimizer {
    /// Completed steps. The next call passes this value to AdamW as `step_before`.
    pub step_index: u64,
    pub momentum: f64,
    pub nesterov: bool,
    pub params: Vec<HybridParam>,
}

impl HybridOptimizer {
    pub fn new(params: Vec<HybridParam>) -> Result<Self, OjasError> {
        if params.is_empty() {
            return Err(OjasError::Shape {
                op: "hybrid_step",
                detail: "no parameters".to_string(),
            });
        }
        Ok(Self {
            step_index: 0,
            momentum: MUON_MOMENTUM,
            nesterov: true,
            params,
        })
    }

    /// Apply `multiplier` to each parameter's initial LR and take one step.
    ///
    /// The gradient must already be the reduced micro-batch mean. Nothing is
    /// written if any gradient is non-finite or the step counter cannot advance.
    pub fn step(&mut self, multiplier: f64) -> Result<(), OjasError> {
        const OP: &str = "hybrid_step";
        let next = next_step(self.step_index)?;
        if !(multiplier.is_finite() && self.momentum.is_finite()) {
            return Err(nonfinite(OP));
        }
        for param in &self.params {
            check_finite_len(
                OP,
                &param.param,
                &param.grad,
                &param.moment1,
                &param.moment2,
            )?;
        }
        let pool = Arc::new(Pool::serial());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Exact,
        };
        let mut staged: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::with_capacity(self.params.len());
        for param in &self.params {
            let lr = scaled_lr(param.initial_lr, multiplier)?;
            let (new_p, new_m, new_v) = match param.group {
                OptimGroup::MuonMatrix => {
                    let config = MuonNs5Config {
                        lr,
                        momentum: self.momentum,
                        weight_decay: MUON_WEIGHT_DECAY,
                        nesterov: self.nesterov,
                    };
                    let (new_p, new_m) = muon_ns5(
                        exec,
                        param.param.clone(),
                        param.grad.clone(),
                        param.moment1.clone(),
                        param.rows,
                        param.cols,
                        config,
                    )?;
                    (new_p, new_m, param.moment2.clone())
                }
                OptimGroup::AdamEmbedding | OptimGroup::AdamVector => {
                    let config = AdamWConfig::nanolab(lr, ADAM_HYBRID_WEIGHT_DECAY);
                    adamw(
                        &param.param,
                        &param.grad,
                        &param.moment1,
                        &param.moment2,
                        self.step_index,
                        config,
                    )?
                }
            };
            staged.push((new_p, new_m, new_v));
        }
        for (param, (new_p, new_m, new_v)) in self.params.iter_mut().zip(staged) {
            param.param = new_p;
            param.moment1 = new_m;
            param.moment2 = new_v;
        }
        self.step_index = next;
        Ok(())
    }
}

fn check_finite_len(
    op: &'static str,
    param: &[f32],
    grad: &[f32],
    moment1: &[f32],
    moment2: &[f32],
) -> Result<(), OjasError> {
    if param.len() != grad.len() || param.len() != moment1.len() || param.len() != moment2.len() {
        return Err(OjasError::Shape {
            op,
            detail: "hybrid tensors differ in length".to_string(),
        });
    }
    if ![param, grad, moment1, moment2].into_iter().all(all_finite) {
        return Err(nonfinite(op));
    }
    Ok(())
}

fn zero_f32(op: &'static str, width: usize) -> Result<Vec<f32>, OjasError> {
    let mut data = Vec::new();
    try_reserve_f32(op, &mut data, width)?;
    data.resize(width, 0.0);
    Ok(data)
}

fn try_reserve_f32(op: &'static str, data: &mut Vec<f32>, len: usize) -> Result<(), OjasError> {
    let bytes = u64::try_from(len)
        .ok()
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "allocation byte length overflows".to_string(),
        })?;
    if data.try_reserve_exact(len).is_err() {
        return Err(OjasError::CapacityExceeded {
            requested: bytes,
            cap: bytes,
            live: 0,
        });
    }
    Ok(())
}

fn shape_product(op: &'static str, shape: &[usize]) -> Result<usize, OjasError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(OjasError::Shape {
            op,
            detail: "empty tensor".to_string(),
        });
    }
    let mut n = 1usize;
    for dim in shape {
        n = n.checked_mul(*dim).ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "parameter length overflows".to_string(),
        })?;
    }
    Ok(n)
}
