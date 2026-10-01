//! Nanolab cosine schedule, as a multiplier on each group's initial learning rate.
//!
//! `schedules.py` `CosineSchedule` returns an absolute LR. `apply_lr` then
//! divides by the reference peak and multiplies each group's `initial_lr`, so
//! the Muon and AdamW groups keep their ratio. This module returns that
//! multiplier directly. The floor is 0.1 of the peak.
//!
//! While `step < warmup`, the multiplier is `(step + 1) / warmup` (the
//! `_warmup_mult` used inside the warmup branch). After warmup,
//!
//! ```text
//! t = min(1, (step - warmup) / max(1, total - warmup))
//! multiplier = 0.1 + 0.9 * 0.5 * (1 + cos(pi * t))
//! ```
//!
//! Warmup 0 is refused. A step that does not fit in an exact `f64` integer,
//! including `u64::MAX`, is refused.

use ojas_core::OjasError;

/// Floor as a fraction of the peak. Nanolab `lr_floor_frac` default.
pub const COSINE_FLOOR_FRAC: f64 = 0.1;

/// Largest integer that is exact in `f64`. `step + 1` must stay inside it.
const F64_INT_MAX: u64 = 1 << 53;

/// Fixed-length cosine schedule.
#[derive(Clone, Copy, Debug)]
pub struct CosineSchedule {
    warmup_steps: u64,
    total_steps: u64,
}

impl CosineSchedule {
    /// `warmup_steps` and `total_steps` must both be non-zero and exact in `f64`.
    pub fn new(warmup_steps: u64, total_steps: u64) -> Result<Self, OjasError> {
        if warmup_steps == 0 {
            return Err(OjasError::OutOfRange {
                op: "cosine_schedule",
                detail: "warmup 0".to_string(),
            });
        }
        if total_steps == 0 {
            return Err(OjasError::OutOfRange {
                op: "cosine_schedule",
                detail: "total steps 0".to_string(),
            });
        }
        exact_step("cosine_schedule", warmup_steps)?;
        exact_step("cosine_schedule", total_steps)?;
        Ok(Self {
            warmup_steps,
            total_steps,
        })
    }

    pub fn warmup_steps(self) -> u64 {
        self.warmup_steps
    }

    pub fn total_steps(self) -> u64 {
        self.total_steps
    }

    /// Multiplier applied to a group's initial learning rate at `step` (0-based).
    pub fn multiplier(self, step: u64) -> Result<f64, OjasError> {
        const OP: &str = "cosine_schedule";
        if step.checked_add(1).is_none() {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "step counter is u64::MAX".to_string(),
            });
        }
        exact_step(OP, step)?;
        exact_step(OP, step + 1)?;
        let step_f = step as f64;
        let warmup = self.warmup_steps as f64;
        if step < self.warmup_steps {
            return Ok((step_f + 1.0) / warmup);
        }
        let span = (self.total_steps as f64) - warmup;
        let denom = if span < 1.0 { 1.0 } else { span };
        let t = ((step_f - warmup) / denom).clamp(0.0, 1.0);
        let floor = COSINE_FLOOR_FRAC;
        let multiplier = floor + (1.0 - floor) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
        if !multiplier.is_finite() {
            return Err(OjasError::NonFinite { op: OP });
        }
        Ok(multiplier)
    }
}

/// `initial_lr * multiplier`. Both factors and the product must be finite.
pub fn scaled_lr(initial_lr: f64, multiplier: f64) -> Result<f64, OjasError> {
    const OP: &str = "cosine_schedule";
    if !(initial_lr.is_finite() && multiplier.is_finite()) {
        return Err(OjasError::NonFinite { op: OP });
    }
    let lr = initial_lr * multiplier;
    if !lr.is_finite() {
        return Err(OjasError::NonFinite { op: OP });
    }
    Ok(lr)
}

fn exact_step(op: &'static str, step: u64) -> Result<(), OjasError> {
    if step > F64_INT_MAX {
        Err(OjasError::OutOfRange {
            op,
            detail: format!("step {step} is not an exact f64 integer"),
        })
    } else {
        Ok(())
    }
}
