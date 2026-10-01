//! The run's bookkeeping, ported from `python/qd_train/run_control.py`: the wall-clock cap and
//! the clock it is measured on, the loss log and its digest, and the consumed-batch digest.
//!
//! Each piece keeps the property its Python original exists for:
//!
//! * [`WallClockCap`] is mandatory and bounded by [`MAX_CAP_S`] (the program's own 40 h,
//!   read-only under rule 2); [`RunClock`] refuses a clock that steps backwards rather than
//!   clamping it, because a clamped clock extends the cap silently.
//! * [`LossLog`] is strictly increasing in optimizer step, and its digest is sha256 over the
//!   exact numbers (`float.hex`), so two runs agree iff their digests agree -- and the digest
//!   is the one Python computes over the same points (`tests/pyjson_oracle.rs`).
//! * [`ConsumedPrefix`] hashes what the run actually ate, length-prefixed part by part, under
//!   the same domain string, so a resume onto a different corpus order is refused.

use std::time::Instant;

use sha2::{Digest, Sha256};

use serde_json::Value;

use crate::pyjson::{dumps, float_hex, obj, PyJsonError, CANONICAL_ASCII};

/// `run_control.MAX_CAP_S`: 40 h. A cap above the program's cap is not a cap.
pub const MAX_CAP_S: f64 = 40.0 * 3600.0;

/// `real_ft_run.WALL_CLOCK_CAP_S`, the cap a run carries when none is given.
pub const DEFAULT_CAP_S: f64 = 1_800.0;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RunControlError {
    #[error("{0}")]
    Invalid(String),
    #[error(
        "the clock went backwards: {now}s elapsed now against {high_water}s already observed. \
         Clamping the difference away would extend the cap by the size of the step back; fix \
         the clock, the cap cannot be trusted until you do."
    )]
    ClockBackwards { now: f64, high_water: f64 },
}

/// The hard stop: positive, finite, at most [`MAX_CAP_S`]. There is no value meaning "no cap".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallClockCap {
    cap_s: f64,
}

impl WallClockCap {
    pub fn new(cap_s: f64) -> Result<Self, RunControlError> {
        if !cap_s.is_finite() {
            return Err(RunControlError::Invalid(format!("cap_s must be finite, got {cap_s}")));
        }
        if cap_s <= 0.0 {
            return Err(RunControlError::Invalid(format!("cap_s must be positive, got {cap_s}")));
        }
        if cap_s > MAX_CAP_S {
            return Err(RunControlError::Invalid(format!(
                "cap_s {cap_s}s exceeds MAX_CAP_S {MAX_CAP_S}s (40 h), the program's own cap. Rule 2: \
                 a kill criterion is read-only. Report that the cap was hit; do not raise it."
            )));
        }
        Ok(Self { cap_s })
    }

    pub fn cap_s(&self) -> f64 {
        self.cap_s
    }

    pub fn expired(&self, elapsed_s: f64) -> bool {
        elapsed_s >= self.cap_s
    }
}

/// A monotonic source of seconds. Injected so a test drives the cap in microseconds.
pub trait Clock {
    fn now_s(&self) -> f64;
}

/// `time.monotonic`'s counterpart.
pub struct MonotonicClock {
    origin: Instant,
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self { origin: Instant::now() }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_s(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

/// Elapsed time against a cap, with `RunControl.elapsed_s`'s refusal of a clock that steps back.
pub struct RunClock<'c> {
    clock: &'c dyn Clock,
    t0: f64,
    high_water: f64,
    cap: WallClockCap,
}

impl<'c> RunClock<'c> {
    pub fn start(clock: &'c dyn Clock, cap: WallClockCap) -> Result<Self, RunControlError> {
        let t0 = clock.now_s();
        if !t0.is_finite() {
            return Err(RunControlError::Invalid(format!("clock reading at start is {t0}")));
        }
        Ok(Self {
            clock,
            t0,
            high_water: 0.0,
            cap,
        })
    }

    pub fn elapsed_s(&mut self) -> Result<f64, RunControlError> {
        let now = self.clock.now_s();
        if !now.is_finite() {
            return Err(RunControlError::Invalid(format!("clock reading is {now}")));
        }
        let elapsed = now - self.t0;
        if elapsed < self.high_water {
            return Err(RunControlError::ClockBackwards {
                now: elapsed,
                high_water: self.high_water,
            });
        }
        self.high_water = elapsed;
        Ok(elapsed)
    }

    /// Polled by the loop at an optimizer-step boundary only.
    pub fn expired(&mut self) -> Result<bool, RunControlError> {
        let e = self.elapsed_s()?;
        Ok(self.cap.expired(e))
    }

    pub fn cap(&self) -> WallClockCap {
        self.cap
    }
}

/// One optimizer step's loss, tagged with the batch it came from (`LossPoint`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossPoint {
    pub optimizer_step: u64,
    pub epoch: u64,
    pub batch_index: u64,
    pub loss: f64,
}

impl LossPoint {
    fn validate(&self) -> Result<(), RunControlError> {
        if !self.loss.is_finite() {
            return Err(RunControlError::Invalid(format!("LossPoint.loss must be finite, got {}", self.loss)));
        }
        if self.loss < 0.0 {
            return Err(RunControlError::Invalid(format!(
                "LossPoint.loss must be non-negative, got {}. A negative cross-entropy is not a small \
                 loss; it is a masking or normalisation bug.",
                self.loss
            )));
        }
        Ok(())
    }

    /// `LossPoint.to_json`: `{"epoch", "index", "loss_hex", "step"}`.
    pub fn to_json(&self) -> Result<Value, PyJsonError> {
        obj([
            ("step", Value::from(self.optimizer_step)),
            ("epoch", Value::from(self.epoch)),
            ("index", Value::from(self.batch_index)),
            ("loss_hex", Value::from(float_hex(self.loss)?)),
        ])
    }
}

/// The loss trajectory, append-only and strictly increasing in optimizer step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LossLog {
    points: Vec<LossPoint>,
}

impl LossLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(&mut self, point: LossPoint) -> Result<(), RunControlError> {
        point.validate()?;
        if self.points.len() as u64 >= crate::schedule::MAX_LOSS_POINTS {
            return Err(RunControlError::Invalid(
                "loss log reached MAX_LOSS_POINTS; a loop that logs more steps than any schedule \
                 declares is not terminating"
                    .into(),
            ));
        }
        if let Some(last) = self.points.last()
            && point.optimizer_step <= last.optimizer_step
        {
            return Err(RunControlError::Invalid(format!(
                "optimizer step {} follows {}: the loss log is strictly increasing in step",
                point.optimizer_step, last.optimizer_step
            )));
        }
        self.points.push(point);
        Ok(())
    }

    pub fn points(&self) -> &[LossPoint] {
        &self.points
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn last(&self) -> Option<&LossPoint> {
        self.points.last()
    }

    pub fn to_json(&self) -> Result<Value, PyJsonError> {
        Ok(Value::Array(self.points.iter().map(LossPoint::to_json).collect::<Result<_, _>>()?))
    }

    /// `LossLog.digest`: sha256 over `json.dumps([...], sort_keys=True, separators=(",", ":"))`.
    pub fn digest(&self) -> Result<String, PyJsonError> {
        let body = dumps(&self.to_json()?, CANONICAL_ASCII)?;
        Ok(hex(&Sha256::digest(body.as_bytes())))
    }
}

/// The running sha256 over consumed batches (`ConsumedPrefix`, domain `qd-consumed-prefix-v1`).
/// Every part is length-prefixed, so `fold(["ab"])`, `fold(["a", "b"])` and two folds cannot
/// collide.
#[derive(Clone)]
pub struct ConsumedPrefix {
    h: Sha256,
    n: u64,
}

impl ConsumedPrefix {
    pub const DOMAIN: &'static [u8] = b"qd-consumed-prefix-v1";

    pub fn new() -> Self {
        let mut h = Sha256::new();
        h.update(Self::DOMAIN);
        Self { h, n: 0 }
    }

    /// Add one consumed batch. Called exactly once per batch, in consumption order.
    pub fn fold(&mut self, parts: &[&[u8]]) -> Result<(), RunControlError> {
        let count = u32::try_from(parts.len())
            .map_err(|_| RunControlError::Invalid(format!("{} parts do not fit the 4-byte count", parts.len())))?;
        self.h.update([0u8]);
        self.h.update(count.to_be_bytes());
        for part in parts {
            let len = u64::try_from(part.len()).map_err(|e| RunControlError::Invalid(e.to_string()))?;
            self.h.update(len.to_be_bytes());
            self.h.update(part);
        }
        self.n += 1;
        Ok(())
    }

    pub fn n_folded(&self) -> u64 {
        self.n
    }

    pub fn hexdigest(&self) -> String {
        hex(&self.h.clone().finalize())
    }
}

impl Default for ConsumedPrefix {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct Fake(Cell<f64>);
    impl Clock for Fake {
        fn now_s(&self) -> f64 {
            self.0.get()
        }
    }

    #[test]
    fn the_cap_is_mandatory_bounded_and_refuses_a_clock_that_steps_back() {
        assert!(WallClockCap::new(0.0).is_err());
        assert!(WallClockCap::new(f64::INFINITY).is_err());
        assert!(WallClockCap::new(MAX_CAP_S + 1.0).is_err());
        let clock = Fake(Cell::new(100.0));
        let mut rc = RunClock::start(&clock, WallClockCap::new(10.0).unwrap()).unwrap();
        clock.0.set(105.0);
        assert!(!rc.expired().unwrap());
        clock.0.set(110.0);
        assert!(rc.expired().unwrap(), "elapsed == cap is expired, as Python's >=");
        clock.0.set(104.0);
        assert!(matches!(rc.elapsed_s(), Err(RunControlError::ClockBackwards { .. })));
    }

    #[test]
    fn the_loss_log_is_strictly_increasing_and_refuses_what_is_not_a_loss() {
        let mut log = LossLog::new();
        let p = |s, l| LossPoint {
            optimizer_step: s,
            epoch: 0,
            batch_index: s,
            loss: l,
        };
        log.append(p(0, 1.5)).unwrap();
        assert!(log.append(p(0, 1.0)).is_err(), "a repeated step");
        assert!(log.append(p(1, f64::NAN)).is_err());
        assert!(log.append(p(1, -0.5)).is_err());
        log.append(p(1, 0.75)).unwrap();
        assert_eq!(
            dumps(&log.to_json().unwrap(), CANONICAL_ASCII).unwrap(),
            "[{\"epoch\":0,\"index\":0,\"loss_hex\":\"0x1.8000000000000p+0\",\"step\":0},\
             {\"epoch\":0,\"index\":1,\"loss_hex\":\"0x1.8000000000000p-1\",\"step\":1}]"
        );
    }

    #[test]
    fn consumed_prefix_parts_are_length_prefixed() {
        let mut a = ConsumedPrefix::new();
        a.fold(&[b"ab"]).unwrap();
        let mut b = ConsumedPrefix::new();
        b.fold(&[b"a", b"b"]).unwrap();
        let mut c = ConsumedPrefix::new();
        c.fold(&[b"a"]).unwrap();
        c.fold(&[b"b"]).unwrap();
        assert_ne!(a.hexdigest(), b.hexdigest());
        assert_ne!(b.hexdigest(), c.hexdigest());
        assert_ne!(a.hexdigest(), c.hexdigest());
        assert_eq!(c.n_folded(), 2);
    }
}
