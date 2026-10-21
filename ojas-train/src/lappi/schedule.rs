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

use std::fmt;

use ojas_io::json::JsonValue as Value;

use super::pyjson::{PyJsonError, float, obj};

/// `run_control.MAX_LOSS_POINTS`: one loss point per optimizer step bounds the schedule too.
pub const MAX_LOSS_POINTS: u64 = 5_000_000;

#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    Invalid(String),
    PastEnd { step: u64, total: u64 },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(m) => f.write_str(m),
            Self::PastEnd { step, total } => write!(
                f,
                "step {step} is past the end of a {total}-step schedule. The loop and the schedule \
                 disagree about how long this run is; that is a bug in one of them, not a rate to clamp."
            ),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// How the rate falls from the peak to `min_lr` once the warmup ends. The warmup is the same
/// linear ramp for every kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decay {
    /// Lappi's schedule: `0.5 * (1 + cos(pi * x))`. What [`LrSchedule::new`] builds.
    Cosine,
    /// The peak, held to the last step.
    Constant,
    /// `1 - x`: straight down toward `min_lr`, which the last step approaches but does not reach.
    Linear,
    /// `gamma^k`, `k = (step - warmup) / every` (integer division): the rate is multiplied by
    /// `gamma` every `every` steps. Evaluated with `f64::powi`, not with a Python `**` oracle;
    /// the two may differ in the last bit.
    Step { every: u64, gamma: f64 },
}

impl Decay {
    /// The decay a schedule object records: the inverse of [`LrSchedule::to_json`]'s `"decay"`
    /// member. Absent is cosine (what Python's four-key form means); an unknown kind, a
    /// missing or ill-typed field, a member the kind does not have, or a step decay that
    /// [`Decay::validate`] refuses is an error.
    pub fn from_json(schedule: &Value) -> Result<Self, ScheduleError> {
        let bad = |m: String| ScheduleError::Invalid(m);
        let d = match schedule.field_opt("decay") {
            Ok(None) => return Ok(Self::Cosine),
            Ok(Some(d)) => d,
            Err(e) => return Err(bad(format!("schedule is not an object: {e}"))),
        };
        let member = |key: &str| match d.field_opt(key) {
            Ok(Some(v)) => Ok(v),
            _ => Err(bad(format!("decay has no {key:?}"))),
        };
        let kind = member("kind")?
            .as_str()
            .map_err(|_| bad("decay kind is not a string".into()))?;
        let (decay, allowed): (Self, &[&str]) = match kind {
            "constant" => (Self::Constant, &["kind"]),
            "linear" => (Self::Linear, &["kind"]),
            "step" => (
                Self::Step {
                    every: member("every")?
                        .as_u64()
                        .map_err(|_| bad("decay every is not a non-negative integer".into()))?,
                    gamma: member("gamma")?
                        .as_f64()
                        .map_err(|_| bad("decay gamma is not a number".into()))?,
                },
                &["kind", "every", "gamma"],
            ),
            other => {
                return Err(bad(format!(
                    "decay kind {other:?} is not constant, linear or step"
                )));
            }
        };
        d.deny_unknown_keys(allowed)
            .map_err(|e| bad(format!("decay {kind:?}: {e}")))?;
        decay.validate()?;
        Ok(decay)
    }

    /// The one owner of the decay's own rule, called by [`LrSchedule::with_decay`] and by
    /// [`Decay::from_json`]: a step decay needs `every > 0` and a `gamma` that is finite and
    /// in `(0, 1]` (so `-0.0`, and a literal that underflows to `0.0`, are refused).
    pub fn validate(&self) -> Result<(), ScheduleError> {
        if let Self::Step { every, gamma } = *self {
            if every == 0 {
                return Err(ScheduleError::Invalid(
                    "step decay: every must be positive, got 0".into(),
                ));
            }
            if !(gamma.is_finite() && gamma > 0.0 && gamma <= 1.0) {
                return Err(ScheduleError::Invalid(format!(
                    "step decay: gamma must be finite and in (0, 1], got {gamma}"
                )));
            }
        }
        Ok(())
    }

    /// The `"decay"` member [`LrSchedule::to_json`] writes for a kind other than cosine.
    fn to_json(self) -> Result<Option<Value>, PyJsonError> {
        Ok(match self {
            Self::Cosine => None,
            Self::Constant => Some(obj([("kind", Value::from("constant"))])?),
            Self::Linear => Some(obj([("kind", Value::from("linear"))])?),
            Self::Step { every, gamma } => Some(obj([
                ("kind", Value::from("step")),
                ("every", Value::from(every)),
                ("gamma", float(gamma)?),
            ])?),
        })
    }
}

/// Linear warmup, then a [`Decay`] (cosine by default) to `min_lr`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrSchedule {
    peak_lr: f64,
    total_steps: u64,
    warmup_steps: u64,
    min_lr: f64,
    decay: Decay,
}

impl LrSchedule {
    /// The checks of `LRSchedule.__post_init__`, in its order; cosine decay.
    pub fn new(
        peak_lr: f64,
        total_steps: u64,
        warmup_steps: u64,
        min_lr: f64,
    ) -> Result<Self, ScheduleError> {
        Self::with_decay(peak_lr, total_steps, warmup_steps, min_lr, Decay::Cosine)
    }

    /// [`LrSchedule::new`] with the decay given. Every check of `new` applies to every kind;
    /// a step decay is also refused for `every == 0` and for a `gamma` that is not finite or
    /// is outside `(0, 1]`.
    pub fn with_decay(
        peak_lr: f64,
        total_steps: u64,
        warmup_steps: u64,
        min_lr: f64,
        decay: Decay,
    ) -> Result<Self, ScheduleError> {
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
        decay.validate()?;
        Ok(Self {
            peak_lr,
            total_steps,
            warmup_steps,
            min_lr,
            decay,
        })
    }

    /// `tools/real_ft_run._control`'s schedule: peak `lr`, warmup `max(1, steps // 20)`, floor
    /// `lr / 10`. The `max(1, ...)` matters below 20 steps: rung (b)'s 20-step run warms up for
    /// one step, not zero.
    ///
    /// Exactly [`LrSchedule::real_ft_with_floor`] at `min_lr = lr / 10`, the floor every run
    /// without `--min-lr` trains on (`crates/qd-train-metal/src/run.rs` calls this).
    pub fn real_ft(lr: f64, steps: u64) -> Result<Self, ScheduleError> {
        Self::real_ft_with_floor(lr, steps, lr / 10.0)
    }

    /// `tools/real_ft_run._control` under `--min-lr min_lr` (v5's `recipe.added`, `--min-lr 0`):
    /// the same peak and warmup as [`LrSchedule::real_ft`], the cosine floor given rather than
    /// `lr / 10`. With a layer-wise split the lower layers run at their scale times this rate,
    /// so a zero floor is a zero floor for them too.
    pub fn real_ft_with_floor(lr: f64, steps: u64, min_lr: f64) -> Result<Self, ScheduleError> {
        Self::new(lr, steps, (steps / 20).max(1), min_lr)
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
    pub fn decay(&self) -> Decay {
        self.decay
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
        let decay = match self.decay {
            Decay::Cosine => 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()),
            Decay::Constant => 1.0,
            Decay::Linear => 1.0 - progress,
            Decay::Step { every, gamma } => {
                let k = i32::try_from((step - self.warmup_steps) / every).map_err(|_| {
                    ScheduleError::Invalid(format!(
                        "step decay exponent for step {step} does not fit an i32"
                    ))
                })?;
                gamma.powi(k)
            }
        };
        Ok(self.min_lr + (self.peak_lr - self.min_lr) * decay)
    }

    /// `LRSchedule.to_json`, as the checkpoint and the manifest record it. A cosine schedule
    /// writes exactly Python's four keys; any other decay adds a `"decay"` object (Python's
    /// reader never sees one, because Lappi's runs are cosine).
    pub fn to_json(&self) -> Result<Value, PyJsonError> {
        let mut pairs = vec![
            ("peak_lr", float(self.peak_lr)?),
            ("total_steps", Value::from(self.total_steps)),
            ("warmup_steps", Value::from(self.warmup_steps)),
            ("min_lr", float(self.min_lr)?),
        ];
        if let Some(d) = self.decay.to_json()? {
            pairs.push(("decay", d));
        }
        obj(pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_what_lrschedule_refuses() {
        assert!(LrSchedule::new(0.0, 10, 1, 0.0).is_err());
        assert!(LrSchedule::new(f64::NAN, 10, 1, 0.0).is_err());
        assert!(LrSchedule::new(1e-5, 10, 1, -1.0).is_err());
        assert!(
            LrSchedule::new(1e-5, 10, 1, 1e-4).is_err(),
            "decays upwards"
        );
        assert!(LrSchedule::new(1e-5, 0, 0, 0.0).is_err());
        assert!(
            LrSchedule::new(1e-5, 10, 10, 0.0).is_err(),
            "warmup never ends"
        );
        assert!(LrSchedule::new(1e-5, MAX_LOSS_POINTS + 1, 1, 0.0).is_err());
        assert!(
            LrSchedule::new(1e-5, 10, 0, 0.0).is_ok(),
            "no warmup is a schedule"
        );
    }

    #[test]
    fn refuses_a_step_past_the_end_rather_than_clamping() {
        let s = LrSchedule::real_ft(1e-5, 20).unwrap();
        assert!(s.lr_at(19).is_ok());
        assert_eq!(
            s.lr_at(20),
            Err(ScheduleError::PastEnd {
                step: 20,
                total: 20
            })
        );
    }

    #[test]
    fn real_ft_warms_up_for_at_least_one_step() {
        // real_ft_run.py:1824 is `max(1, steps // 20)`: below 20 steps, `steps // 20` is 0 and
        // the max makes it 1. A port that dropped the max would start at the peak.
        for steps in [1u64, 2, 19] {
            let s = LrSchedule::real_ft(1e-5, steps);
            if steps == 1 {
                assert!(
                    s.is_err(),
                    "warmup 1 >= total 1 is refused, as LRSchedule refuses it"
                );
            } else {
                assert_eq!(s.unwrap().warmup_steps(), 1);
            }
        }
        assert_eq!(LrSchedule::real_ft(1e-5, 20).unwrap().warmup_steps(), 1);
        assert_eq!(LrSchedule::real_ft(1e-5, 1505).unwrap().warmup_steps(), 75);
        let s = LrSchedule::real_ft(1e-5, 100).unwrap();
        assert_eq!(s.min_lr(), 1e-5 / 10.0);
        assert_eq!(
            s.lr_at(4).unwrap(),
            1e-5,
            "the last warmup step is the peak exactly"
        );
    }

    #[test]
    fn real_ft_with_floor_is_real_ft_but_for_the_floor() {
        for steps in [2u64, 20, 200, 9683] {
            let default = LrSchedule::real_ft(1e-5, steps).unwrap();
            let same = LrSchedule::real_ft_with_floor(1e-5, steps, 1e-5 / 10.0).unwrap();
            assert_eq!(default, same, "real_ft is real_ft_with_floor at lr / 10");
            let zero = LrSchedule::real_ft_with_floor(1e-5, steps, 0.0).unwrap();
            assert_eq!(zero.warmup_steps(), default.warmup_steps());
            assert_eq!(zero.min_lr(), 0.0);
            let last = zero.lr_at(steps - 1).unwrap();
            assert!(
                last > 0.0,
                "the last step of a zero floor is still a positive rate"
            );
            assert!(last <= default.lr_at(steps - 1).unwrap());
        }
        assert!(LrSchedule::real_ft_with_floor(1e-5, 200, -1e-9).is_err());
        assert!(LrSchedule::real_ft_with_floor(1e-5, 200, f64::NAN).is_err());
        assert!(
            LrSchedule::real_ft_with_floor(1e-5, 200, 2e-5).is_err(),
            "above the peak"
        );
    }
}
