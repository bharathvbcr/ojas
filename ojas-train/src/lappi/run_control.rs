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
//!
//! * [`tensor_ref_digest`] and [`sidecar_digest`] are `TensorRef.digest` and `_sidecar_digest`,
//!   the repo's canonical digest of a tensor set, which pins the span-head init file
//!   (Amendment 2 (iv); `tests/head_init.rs` checks them against Python's own functions).
//!
//! The consumed-batch digest (`ConsumedPrefix`) has one owner, beside the `Batch` it folds:
//! [`crate::shards::ConsumedPrefix`].

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

pub(crate) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// `run_control._TENSOR_REF_DOMAIN`.
pub const TENSOR_REF_DOMAIN: &[u8] = b"qd-tensor-ref-v1";
/// `run_control._SIDECAR_DOMAIN`.
pub const SIDECAR_DOMAIN: &[u8] = b"qd-checkpoint-sidecar-v1";

/// `TensorRef.digest` (`run_control.py:1140-1158`): sha256 over the domain, then the dtype name,
/// the shape and the bytes, each length-prefixed big-endian (u32 for the name and the rank, u64
/// for every axis and the byte count). `data` is the tensor's little-endian bytes. A byte count
/// that disagrees with the shape is refused, as `TensorRef` refuses it.
pub fn tensor_ref_digest(dtype: &str, shape: &[usize], element_bytes: usize, data: &[u8]) -> Result<String, RunControlError> {
    let elems = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .and_then(|n| n.checked_mul(element_bytes))
        .ok_or_else(|| RunControlError::Invalid(format!("shape {shape:?} overflows")))?;
    if elems != data.len() {
        return Err(RunControlError::Invalid(format!(
            "a {dtype} tensor of shape {shape:?} is {elems} bytes and this one carries {}",
            data.len()
        )));
    }
    let name_len = u32::try_from(dtype.len()).map_err(|e| RunControlError::Invalid(e.to_string()))?;
    let rank = u32::try_from(shape.len()).map_err(|e| RunControlError::Invalid(e.to_string()))?;
    let mut h = Sha256::new();
    h.update(TENSOR_REF_DOMAIN);
    h.update(name_len.to_be_bytes());
    h.update(dtype.as_bytes());
    h.update(rank.to_be_bytes());
    for &axis in shape {
        h.update((axis as u64).to_be_bytes());
    }
    h.update((data.len() as u64).to_be_bytes());
    h.update(data);
    Ok(hex(&h.finalize()))
}

/// `run_control._sidecar_digest` (`run_control.py:1308-1322`): sha256 over the domain, the
/// count (u64 big-endian), then for each name in sorted order its UTF-8 length (u64), the name
/// and the 32 raw bytes of its tensor digest. Python sorts `str` by code point and UTF-8 byte
/// order is code-point order, so sorting the Rust strings gives the same sequence. A repeated
/// name, or a digest that is not 64 lowercase hex characters, is refused.
pub fn sidecar_digest(tensors: &[(String, String)]) -> Result<String, RunControlError> {
    let mut sorted: Vec<&(String, String)> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(w) = sorted.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(RunControlError::Invalid(format!("{} is named twice in one tensor set", w[0].0)));
    }
    let mut h = Sha256::new();
    h.update(SIDECAR_DOMAIN);
    h.update((sorted.len() as u64).to_be_bytes());
    for (name, digest) in sorted {
        let raw = unhex32(digest).ok_or_else(|| RunControlError::Invalid(format!("{name}: digest {digest:?} is not a sha256")))?;
        h.update((name.len() as u64).to_be_bytes());
        h.update(name.as_bytes());
        h.update(raw);
    }
    Ok(hex(&h.finalize()))
}

fn unhex32(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (nibble(b[2 * i])? << 4) | nibble(b[2 * i + 1])?;
    }
    Some(out)
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
}
