//! Nanolab cosine and warmup-stable-decay schedules, as a multiplier on each
//! group's initial learning rate. [`LrSchedule`] selects one; [`WsdSchedule`]
//! documents its own formula.
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

/// Nanolab `wsd_decay_frac` default: the last fifth of the run decays.
pub const WSD_DECAY_FRAC: f64 = 0.2;

/// Nanolab warmup-stable-decay (`schedules.py` `WSDSchedule`), as a
/// multiplier on each group's initial learning rate.
///
/// While `step < warmup`, the multiplier is `(step + 1) / warmup`. After
/// warmup,
///
/// ```text
/// decay_steps = trunc(decay_frac * total)
/// decay_start = total - decay_steps
/// step <  decay_start: 1
/// step >= decay_start: floor + (1 - floor) * (1 - (step - decay_start) / max(1, decay_steps))
/// ```
///
/// with `floor` = [`COSINE_FLOOR_FRAC`] (nanolab's `lr_floor_frac`, shared by
/// its cosine and WSD schedules). Each line is the f64 operation nanolab
/// performs, in its order, so the bits match it.
///
/// Nanolab does not clamp the decay: past `total` its multiplier falls below
/// the floor and then below zero. Here a step past `total` is refused instead,
/// so the schedule is defined on `0..=total` and never below the floor. Warmup
/// 0, total 0, and a `decay_frac` outside `[0, 1]` are refused, as are steps
/// that are not exact `f64` integers.
#[derive(Clone, Copy, Debug)]
pub struct WsdSchedule {
    warmup_steps: u64,
    total_steps: u64,
    decay_frac: f64,
}

impl WsdSchedule {
    pub fn new(warmup_steps: u64, total_steps: u64, decay_frac: f64) -> Result<Self, OjasError> {
        const OP: &str = "wsd_schedule";
        if warmup_steps == 0 {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "warmup 0".to_string(),
            });
        }
        if total_steps == 0 {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "total steps 0".to_string(),
            });
        }
        if !(decay_frac.is_finite() && (0.0..=1.0).contains(&decay_frac)) {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("decay fraction {decay_frac} is outside [0, 1]"),
            });
        }
        exact_step(OP, warmup_steps)?;
        exact_step(OP, total_steps)?;
        Ok(Self {
            warmup_steps,
            total_steps,
            decay_frac,
        })
    }

    pub fn warmup_steps(self) -> u64 {
        self.warmup_steps
    }

    pub fn total_steps(self) -> u64 {
        self.total_steps
    }

    pub fn decay_frac(self) -> f64 {
        self.decay_frac
    }

    /// Multiplier applied to a group's initial learning rate at `step` (0-based).
    pub fn multiplier(self, step: u64) -> Result<f64, OjasError> {
        const OP: &str = "wsd_schedule";
        if step.checked_add(1).is_none() {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "step counter is u64::MAX".to_string(),
            });
        }
        exact_step(OP, step)?;
        exact_step(OP, step + 1)?;
        if step > self.total_steps {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!(
                    "step {step} is past the schedule's {} steps",
                    self.total_steps
                ),
            });
        }
        if step < self.warmup_steps {
            return Ok((step as f64 + 1.0) / self.warmup_steps as f64);
        }
        // `int(frac * total)`: the product is in `[0, total]`, so the cast
        // truncates exactly as Python's `int` does.
        let decay_steps = (self.decay_frac * self.total_steps as f64) as u64;
        let decay_start = self.total_steps - decay_steps;
        if step < decay_start {
            return Ok(1.0);
        }
        let frac = (step - decay_start) as f64 / decay_steps.max(1) as f64;
        let floor = COSINE_FLOOR_FRAC;
        let multiplier = floor + (1.0 - floor) * (1.0 - frac);
        if !multiplier.is_finite() {
            return Err(OjasError::NonFinite { op: OP });
        }
        Ok(multiplier)
    }
}

/// The learning-rate schedules a trainer can run: nanolab's `cosine` and
/// `wsd`.
#[derive(Clone, Copy, Debug)]
pub enum LrSchedule {
    Cosine(CosineSchedule),
    Wsd(WsdSchedule),
}

impl LrSchedule {
    pub fn warmup_steps(self) -> u64 {
        match self {
            Self::Cosine(s) => s.warmup_steps(),
            Self::Wsd(s) => s.warmup_steps(),
        }
    }

    pub fn total_steps(self) -> u64 {
        match self {
            Self::Cosine(s) => s.total_steps(),
            Self::Wsd(s) => s.total_steps(),
        }
    }

    /// The wrapped schedule's multiplier, with its refusals.
    pub fn multiplier(self, step: u64) -> Result<f64, OjasError> {
        match self {
            Self::Cosine(s) => s.multiplier(step),
            Self::Wsd(s) => s.multiplier(step),
        }
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
