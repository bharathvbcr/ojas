//! The learning-rate schedule: a port of `LRSchedule` (`python/qd_train/run_control.py:309-392`)
//! and of the one recipe `tools/real_ft_run.py:1823-1825` builds from it.
//!
//! **A pure function of the step.** `lr_at(step)` depends on the step and the four fields only,
//! so a resumed run recomputes the rate rather than restoring it. It refuses a step past the
//! end rather than clamping: a loop that asks for step `total_steps` has a different run length
//! in mind than the schedule, and a clamped value would hide that.
//!
//! **Bit equality with Python, and where it holds.** Every operation is the one Python performs,
//! in Python's order, in f64: `peak * (step + 1) / warmup` (left to right, two roundings),
//! `(step - warmup) / span` (both integers exact in f64 below 2^53, one correctly rounded
//! division), `0.5 * (1.0 + cos(pi * progress))`, `min + (peak - min) * decay`. `math.cos` and
//! `f64::cos` both call the platform libm, so the fixture test
//! (`tests/schedule_oracle.rs`, values dumped by `tools/qd_train_oracle_trainer.py`) is a
//! same-host claim: on this Mac both sides are Apple's libm. On Linux the crate builds and the
//! warmup half still matches bit for bit; the cosine half is glibc's and is compared there only
//! if the fixture is regenerated on that host.

use crate::pyjson::{Json, PyJsonError};

/// `run_control.MAX_LOSS_POINTS`: one loss point per optimizer step bounds the schedule too.
pub const MAX_LOSS_POINTS: u64 = 5_000_000;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    #[error("{0}")]
    Invalid(String),
    #[error(
        "step {step} is past the end of a {total}-step schedule. The loop and the schedule \
         disagree about how long this run is; that is a bug in one of them, not a rate to clamp."
    )]
    PastEnd { step: u64, total: u64 },
}

/// Linear warmup, then cosine decay to `min_lr`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrSchedule {
    peak_lr: f64,
    total_steps: u64,
    warmup_steps: u64,
    min_lr: f64,
}

impl LrSchedule {
    /// The checks of `LRSchedule.__post_init__`, in its order.
    pub fn new(peak_lr: f64, total_steps: u64, warmup_steps: u64, min_lr: f64) -> Result<Self, ScheduleError> {
        let bad = |m: String| Err(ScheduleError::Invalid(m));
        if !peak_lr.is_finite() {
            return bad(format!("peak_lr must be finite, got {peak_lr}"));
        }
        if !min_lr.is_finite() {
            return bad(format!("min_lr must be finite, got {min_lr}"));
        }
        if peak_lr <= 0.0 {
            return bad(format!("peak_lr must be positive, got {peak_lr}"));
        }
        if min_lr < 0.0 {
            return bad(format!("min_lr must be non-negative, got {min_lr}"));
        }
        if min_lr > peak_lr {
            return bad(format!(
                "min_lr {min_lr} exceeds peak_lr {peak_lr}: the schedule would decay upwards"
            ));
        }
        if total_steps == 0 {
            return bad(format!("total_steps must be positive, got {total_steps}"));
        }
        if total_steps > MAX_LOSS_POINTS {
            return bad(format!(
                "total_steps {total_steps} exceeds MAX_LOSS_POINTS {MAX_LOSS_POINTS}: the loss log \
                 this schedule implies would not be bounded"
            ));
        }
        if warmup_steps >= total_steps {
            return bad(format!(
                "warmup_steps {warmup_steps} >= total_steps {total_steps}: the run would end before \
                 the warmup does, so the peak rate is never reached and the recipe's stated peak_lr \
                 describes a rate this run never uses"
            ));
        }
        Ok(Self {
            peak_lr,
            total_steps,
            warmup_steps,
            min_lr,
        })
    }

    /// `tools/real_ft_run._control`'s schedule: peak `lr`, warmup `max(1, steps // 20)`, floor
    /// `lr / 10`. The `max(1, ...)` matters below 20 steps: rung (b)'s 20-step run warms up for
    /// one step, not zero.
    pub fn real_ft(lr: f64, steps: u64) -> Result<Self, ScheduleError> {
        Self::new(lr, steps, (steps / 20).max(1), lr / 10.0)
    }

    pub fn peak_lr(&self) -> f64 {
        self.peak_lr
    }
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
    pub fn warmup_steps(&self) -> u64 {
        self.warmup_steps
    }
    pub fn min_lr(&self) -> f64 {
        self.min_lr
    }

    /// The rate for `step`, 0-based over optimizer steps.
    pub fn lr_at(&self, step: u64) -> Result<f64, ScheduleError> {
        if step >= self.total_steps {
            return Err(ScheduleError::PastEnd {
                step,
                total: self.total_steps,
            });
        }
        // u64 -> f64 is exact for every value MAX_LOSS_POINTS admits.
        if step < self.warmup_steps {
            return Ok(self.peak_lr * (step + 1) as f64 / self.warmup_steps as f64);
        }
        let span = self.total_steps - self.warmup_steps;
        let progress = (step - self.warmup_steps) as f64 / span as f64;
        let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        Ok(self.min_lr + (self.peak_lr - self.min_lr) * decay)
    }

    /// `LRSchedule.to_json`, as the checkpoint and the manifest record it.
    pub fn to_json(&self) -> Result<Json, PyJsonError> {
        Json::obj([
            ("peak_lr", Json::Float(self.peak_lr)),
            ("total_steps", Json::Int(to_i64(self.total_steps)?)),
            ("warmup_steps", Json::Int(to_i64(self.warmup_steps)?)),
            ("min_lr", Json::Float(self.min_lr)),
        ])
    }
}

fn to_i64(x: u64) -> Result<i64, PyJsonError> {
    i64::try_from(x).map_err(|e| PyJsonError(format!("{x}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_what_lrschedule_refuses() {
        assert!(LrSchedule::new(0.0, 10, 1, 0.0).is_err());
        assert!(LrSchedule::new(f64::NAN, 10, 1, 0.0).is_err());
        assert!(LrSchedule::new(1e-5, 10, 1, -1.0).is_err());
        assert!(LrSchedule::new(1e-5, 10, 1, 1e-4).is_err(), "decays upwards");
        assert!(LrSchedule::new(1e-5, 0, 0, 0.0).is_err());
        assert!(LrSchedule::new(1e-5, 10, 10, 0.0).is_err(), "warmup never ends");
        assert!(LrSchedule::new(1e-5, MAX_LOSS_POINTS + 1, 1, 0.0).is_err());
        assert!(LrSchedule::new(1e-5, 10, 0, 0.0).is_ok(), "no warmup is a schedule");
    }

    #[test]
    fn refuses_a_step_past_the_end_rather_than_clamping() {
        let s = LrSchedule::real_ft(1e-5, 20).unwrap();
        assert!(s.lr_at(19).is_ok());
        assert_eq!(s.lr_at(20), Err(ScheduleError::PastEnd { step: 20, total: 20 }));
    }

    #[test]
    fn real_ft_warms_up_for_at_least_one_step() {
        // real_ft_run.py:1824 is `max(1, steps // 20)`: below 20 steps, `steps // 20` is 0 and
        // the max makes it 1. A port that dropped the max would start at the peak.
        for steps in [1u64, 2, 19] {
            let s = LrSchedule::real_ft(1e-5, steps);
            if steps == 1 {
                assert!(s.is_err(), "warmup 1 >= total 1 is refused, as LRSchedule refuses it");
            } else {
                assert_eq!(s.unwrap().warmup_steps(), 1);
            }
        }
        assert_eq!(LrSchedule::real_ft(1e-5, 20).unwrap().warmup_steps(), 1);
        assert_eq!(LrSchedule::real_ft(1e-5, 1505).unwrap().warmup_steps(), 75);
        let s = LrSchedule::real_ft(1e-5, 100).unwrap();
        assert_eq!(s.min_lr(), 1e-5 / 10.0);
        assert_eq!(s.lr_at(4).unwrap(), 1e-5, "the last warmup step is the peak exactly");
    }
}
