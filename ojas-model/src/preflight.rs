//! A Metal-aware memory pre-flight for a Qwen3.5 LoRA step.
//!
//! `mac_heavy.sh`'s RSS cap and memguard's footprint cap do not see Metal
//! buffers, so a step that fits neither cap's accounting can still be killed
//! (or refused mid-step) on the device. This module answers "does the step
//! fit at the run's widest bucket" before the first GPU step, and refuses
//! with the shape when it does not. Two checks, one refusal type:
//!
//! - [`plan`]: an analytic estimate. [`Calibration`] fits
//!   `peak(T) = c0 + c1 * T` for one snapshot from committed bench rows
//!   (`bench/results/2026-10-17-lora-step/rows.jsonl`: arm `tape-lora`,
//!   checkpointing `on`, status `ok`, field `step_peak_bytes`). Nothing is
//!   hard-coded: the numbers come from the rows file or from points the
//!   caller supplies. The fit is ordinary least squares (exact for two rows),
//!   with the intercept raised just enough that the estimate never falls
//!   below a measured peak at a calibration point.
//! - [`dry_run`]: runs the caller's step once at the widest bucket under a
//!   child [`Budget`] of the backend's budget and maps
//!   [`OjasError::CapacityExceeded`] to the same [`PreflightRefusal`], with
//!   the measured peak.
//!
//! Neither check replaces the other. The analytic plan needs no device; the
//! dry run measures what the step really charges, but only what is charged
//! through the child budget it is handed.
//!
//! Wiring this before step 0 is phase A's job (the `StepProvider`); this
//! module only provides the checks.
//!
//! What is checked, and what is not:
//! - `live_bytes` is device memory already held by someone else that the
//!   calibrated peak does not contain (another process's Metal buffers). The
//!   calibrated peak already contains this run's resident weights.
//! - A bucket outside the calibrated token range is allowed and flagged
//!   [`Plan::extrapolated`]; the line is not validated there.
//! - The model is linear in tokens at fixed checkpointing, rank and
//!   micro-batch, as measured. It says nothing about another rank, another
//!   checkpointing mode or another snapshot, and refuses a snapshot with no
//!   rows.

use ojas_core::{Backend, Budget, OjasError};
use ojas_io::parse_json;
use std::fmt;
use std::path::Path;

/// Largest rows file read, in bytes.
pub const MAX_ROWS_BYTES: u64 = 4 << 20;

/// The bench arm and checkpointing mode whose rows calibrate the plan.
const ROW_ARM: &str = "tape-lora";
const ROW_CHECKPOINTING: &str = "on";
const ROW_STATUS_OK: &str = "ok";

/// Why a pre-flight refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalReason {
    /// The analytic estimate exceeds the cap minus live bytes.
    Estimate,
    /// The cap is below the snapshot's resident bytes: not even the weights fit.
    ResidentExceedsCap,
    /// The dry run's own charge was refused by the budget.
    DryRun,
}

/// The step does not fit; the shape that did not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreflightRefusal {
    /// Tokens the caller asked about.
    pub tokens: u64,
    /// The bucket width planned (`tokens` rounded up to the run's buckets).
    pub bucket: u64,
    /// Bytes the step needs: the analytic peak, the resident bytes, or for a
    /// dry run the live bytes plus the charge that was refused.
    pub estimated_bytes: u64,
    /// The cap it was held against.
    pub cap_bytes: u64,
    /// Bytes already live outside the estimate.
    pub live_bytes: u64,
    pub reason: RefusalReason,
    /// The bucket lies outside the calibrated token range.
    pub extrapolated: bool,
    /// Highest charge the dry run's child budget held before refusing.
    /// `None` for an analytic refusal.
    pub measured_peak_bytes: Option<u64>,
}

impl fmt::Display for PreflightRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.reason {
            RefusalReason::Estimate => "estimated peak",
            RefusalReason::ResidentExceedsCap => "resident weights",
            RefusalReason::DryRun => "dry-run charge",
        };
        write!(
            f,
            "preflight refused: step at {} tokens (bucket {}) needs {} bytes ({what}), \
             cap {} bytes with {} live",
            self.tokens, self.bucket, self.estimated_bytes, self.cap_bytes, self.live_bytes
        )?;
        if self.extrapolated {
            write!(f, "; estimate extrapolated beyond the calibrated rows")?;
        }
        if let Some(peak) = self.measured_peak_bytes {
            write!(f, "; measured peak before refusal {peak} bytes")?;
        }
        Ok(())
    }
}

/// A pre-flight that did not pass, and why.
#[derive(Debug)]
pub enum PreflightError {
    /// The step does not fit.
    Refused(PreflightRefusal),
    /// The rows file or points are unusable: unreadable, malformed, a
    /// duplicate token count, a non-monotone peak, too few points.
    Calibration { detail: String },
    /// No usable row for this snapshot. Never a pass.
    NoRows { snapshot: String },
    /// Tokens, buckets or cap are not usable (zero or negative tokens, empty
    /// bucket list, tokens above every bucket).
    InvalidInput { detail: String },
    /// A byte count did not fit `u64`.
    Overflow { detail: String },
    /// The step closure failed with something other than a capacity refusal.
    Step(OjasError),
}

impl fmt::Display for PreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreflightError::Refused(r) => write!(f, "{r}"),
            PreflightError::Calibration { detail } => {
                write!(f, "preflight calibration: {detail}")
            }
            PreflightError::NoRows { snapshot } => {
                write!(f, "preflight: no calibration rows for snapshot {snapshot}")
            }
            PreflightError::InvalidInput { detail } => write!(f, "preflight: {detail}"),
            PreflightError::Overflow { detail } => {
                write!(f, "preflight: u64 overflow: {detail}")
            }
            PreflightError::Step(err) => write!(f, "preflight dry run: step failed: {err}"),
        }
    }
}

impl std::error::Error for PreflightError {}

/// One measured point: the device peak of a step at `tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub tokens: u64,
    pub peak_bytes: u64,
}

/// `peak(T) = c0 + c1 * T` for one snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    snapshot: String,
    resident_bytes: u64,
    c0: f64,
    c1: f64,
    min_tokens: u64,
    max_tokens: u64,
}

impl Calibration {
    /// Fit from measured points.
    ///
    /// At least two points are required. Token counts must be distinct and
    /// the peak must strictly increase with tokens, or the rows contradict a
    /// per-token cost and this is a [`PreflightError::Calibration`]. A fit with
    /// a negative intercept is refused too: a step cannot cost less than
    /// nothing at zero tokens, so the points do not describe this model.
    pub fn from_points(
        snapshot: &str,
        resident_bytes: u64,
        points: &[Point],
    ) -> Result<Self, PreflightError> {
        if points.len() < 2 {
            return Err(calibration(format!(
                "{} point(s) for {snapshot}; a slope needs at least two",
                points.len()
            )));
        }
        let mut sorted = points.to_vec();
        sorted.sort_by_key(|p| p.tokens);
        for pair in sorted.windows(2) {
            if pair[0].tokens == pair[1].tokens {
                return Err(calibration(format!(
                    "duplicate rows for {snapshot} at {} tokens",
                    pair[0].tokens
                )));
            }
            if pair[1].peak_bytes <= pair[0].peak_bytes {
                return Err(calibration(format!(
                    "non-monotone rows for {snapshot}: {} tokens peak {} bytes, \
                     then {} tokens peak {} bytes",
                    pair[0].tokens, pair[0].peak_bytes, pair[1].tokens, pair[1].peak_bytes
                )));
            }
        }
        let n = sorted.len() as f64;
        let mean_x = sorted.iter().map(|p| p.tokens as f64).sum::<f64>() / n;
        let mean_y = sorted.iter().map(|p| p.peak_bytes as f64).sum::<f64>() / n;
        let sxx: f64 = sorted
            .iter()
            .map(|p| (p.tokens as f64 - mean_x).powi(2))
            .sum();
        let sxy: f64 = sorted
            .iter()
            .map(|p| (p.tokens as f64 - mean_x) * (p.peak_bytes as f64 - mean_y))
            .sum();
        let c1 = sxy / sxx;
        let c0 = mean_y - c1 * mean_x;
        if !(c1.is_finite() && c0.is_finite() && c1 > 0.0) {
            return Err(calibration(format!(
                "fit for {snapshot} is not a positive finite slope (c0 {c0}, c1 {c1})"
            )));
        }
        if c0 < 0.0 {
            return Err(calibration(format!(
                "fit for {snapshot} has a negative intercept {c0}"
            )));
        }
        let mut fit = Self {
            snapshot: snapshot.to_owned(),
            resident_bytes,
            c0,
            c1,
            min_tokens: sorted[0].tokens,
            max_tokens: sorted[sorted.len() - 1].tokens,
        };
        fit.cover_measured(&sorted)?;
        Ok(fit)
    }

    /// Least squares splits its error both ways, so with three or more points
    /// the line can pass below a measured peak, and an estimate below a
    /// measurement is a pass that then runs out of memory. Raise the intercept
    /// until the rounded estimate is at least every measured peak (rounding
    /// of large values included). The slope is untouched.
    fn cover_measured(&mut self, points: &[Point]) -> Result<(), PreflightError> {
        for _ in 0..64 {
            let mut deficit = 0.0_f64;
            for p in points {
                let fitted = self.peak_bytes(p.tokens)?;
                if fitted < p.peak_bytes {
                    deficit = deficit.max((p.peak_bytes - fitted) as f64);
                }
            }
            if deficit == 0.0 {
                return Ok(());
            }
            let ulp = self.c0.abs() * f64::EPSILON;
            self.c0 += deficit.max(ulp);
        }
        Err(calibration(format!(
            "fit for {} cannot be raised to cover its measured peaks",
            self.snapshot
        )))
    }

    /// Fit from the text of a `rows.jsonl`, using rows of `snapshot` that are
    /// arm `tape-lora`, checkpointing `on` and status `ok`. Other rows
    /// (refused, other arms, other snapshots) are skipped. No such row is
    /// [`PreflightError::NoRows`].
    pub fn from_rows_text(text: &str, snapshot: &str) -> Result<Self, PreflightError> {
        let mut points = Vec::new();
        let mut resident: Option<u64> = None;
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let at = |what: &str, err: &dyn fmt::Display| {
                calibration(format!("rows line {}: {what}: {err}", index + 1))
            };
            let row = parse_json(line).map_err(|e| at("not JSON", &e))?;
            let text_field = |key: &str| -> Result<String, PreflightError> {
                row.get_str(key).map(str::to_owned).map_err(|e| at(key, &e))
            };
            // The arm is read first: a row of another arm owns its own
            // schema and need not carry this arm's fields.
            if text_field("arm")? != ROW_ARM {
                continue;
            }
            if text_field("snapshot")? != snapshot
                || text_field("checkpointing")? != ROW_CHECKPOINTING
                || text_field("status")? != ROW_STATUS_OK
            {
                continue;
            }
            let number = |key: &str| -> Result<u64, PreflightError> {
                row.get_u64(key).map_err(|e| at(key, &e))
            };
            let row_resident = number("resident_bytes")?;
            match resident {
                Some(seen) if seen != row_resident => {
                    return Err(calibration(format!(
                        "rows for {snapshot} disagree on resident_bytes: {seen} and {row_resident}"
                    )));
                }
                _ => resident = Some(row_resident),
            }
            points.push(Point {
                tokens: number("tokens")?,
                peak_bytes: number("step_peak_bytes")?,
            });
        }
        let Some(resident) = resident else {
            return Err(PreflightError::NoRows {
                snapshot: snapshot.to_owned(),
            });
        };
        Self::from_points(snapshot, resident, &points)
    }

    /// Fit from a `rows.jsonl` on disk. A missing, oversized or unreadable
    /// file is a [`PreflightError::Calibration`], never an empty pass.
    pub fn from_rows_path(path: &Path, snapshot: &str) -> Result<Self, PreflightError> {
        let shown = path.display();
        let len = std::fs::metadata(path)
            .map_err(|e| calibration(format!("{shown}: {e}")))?
            .len();
        if len > MAX_ROWS_BYTES {
            return Err(calibration(format!(
                "{shown}: {len} bytes exceeds the {MAX_ROWS_BYTES} byte limit"
            )));
        }
        let text =
            std::fs::read_to_string(path).map_err(|e| calibration(format!("{shown}: {e}")))?;
        Self::from_rows_text(&text, snapshot)
    }

    pub fn snapshot(&self) -> &str {
        &self.snapshot
    }

    /// Resident bytes of the snapshot's weights and adapters, from the rows.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Smallest and largest calibrated token counts.
    pub fn token_range(&self) -> (u64, u64) {
        (self.min_tokens, self.max_tokens)
    }

    /// Fixed bytes `c0` and bytes per token `c1` of the fit.
    pub fn coefficients(&self) -> (f64, f64) {
        (self.c0, self.c1)
    }

    /// The fitted peak at `tokens`, rounded up to a whole byte.
    pub fn peak_bytes(&self, tokens: u64) -> Result<u64, PreflightError> {
        let value = (self.c0 + self.c1 * tokens as f64).ceil();
        // `u64::MAX as f64` rounds up to 2^64, so `>=` also excludes it.
        if !value.is_finite() || value < 0.0 || value >= u64::MAX as f64 {
            return Err(PreflightError::Overflow {
                detail: format!("fitted peak {value} at {tokens} tokens does not fit u64"),
            });
        }
        Ok(value as u64)
    }
}

/// The analytic plan for a step that fits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub tokens: u64,
    pub bucket: u64,
    pub estimated_bytes: u64,
    pub cap_bytes: u64,
    pub live_bytes: u64,
    /// The bucket lies outside the calibrated token range.
    pub extrapolated: bool,
}

/// The smallest of `buckets` that holds `tokens`.
///
/// Zero or negative tokens, an empty list, a zero-width bucket, and tokens
/// above every bucket are [`PreflightError::InvalidInput`].
pub fn round_to_bucket(tokens: i64, buckets: &[u64]) -> Result<u64, PreflightError> {
    let tokens = positive_tokens(tokens)?;
    if buckets.is_empty() {
        return Err(invalid("the bucket list is empty".to_owned()));
    }
    if buckets.contains(&0) {
        return Err(invalid("a bucket width of 0".to_owned()));
    }
    buckets
        .iter()
        .copied()
        .filter(|&width| width >= tokens)
        .min()
        .ok_or_else(|| {
            invalid(format!(
                "{tokens} tokens exceed the widest bucket {}",
                buckets.iter().copied().max().unwrap_or(0)
            ))
        })
}

/// Plan the step at the run's widest bucket against `cap_bytes`.
///
/// `tokens` is the widest sequence in the run; it is rounded up to `buckets`.
/// The step fits when the fitted peak is at most `cap_bytes - live_bytes`.
/// Otherwise (or when `cap_bytes` is below the snapshot's resident bytes) the
/// refusal carries the shape. `live_bytes` above the cap leaves no room and
/// refuses.
pub fn plan(
    calibration: &Calibration,
    tokens: i64,
    buckets: &[u64],
    cap_bytes: u64,
    live_bytes: u64,
) -> Result<Plan, PreflightError> {
    let bucket = round_to_bucket(tokens, buckets)?;
    let tokens = positive_tokens(tokens)?;
    let (min, max) = calibration.token_range();
    let extrapolated = bucket < min || bucket > max;
    let refuse = |reason, estimated_bytes| {
        PreflightError::Refused(PreflightRefusal {
            tokens,
            bucket,
            estimated_bytes,
            cap_bytes,
            live_bytes,
            reason,
            extrapolated,
            measured_peak_bytes: None,
        })
    };
    if cap_bytes < calibration.resident_bytes() {
        return Err(refuse(
            RefusalReason::ResidentExceedsCap,
            calibration.resident_bytes(),
        ));
    }
    let estimated_bytes = calibration.peak_bytes(bucket)?;
    let room = cap_bytes.saturating_sub(live_bytes);
    if estimated_bytes > room {
        return Err(refuse(RefusalReason::Estimate, estimated_bytes));
    }
    Ok(Plan {
        tokens,
        bucket,
        estimated_bytes,
        cap_bytes,
        live_bytes,
        extrapolated,
    })
}

/// What a passing dry run measured.
#[derive(Debug)]
pub struct DryRun<R> {
    /// The step closure's value.
    pub value: R,
    /// Highest charge the child budget held.
    pub peak_bytes: u64,
}

/// Run `step_fn` once under a child [`Budget`] of `backend`'s budget.
///
/// `widest_tokens` is the widest bucket of the run, already rounded (see
/// [`round_to_bucket`]); it is reported as both `tokens` and `bucket` in a
/// refusal. The child's cap is the parent's cap, so the parent's live bytes
/// count against it. The step must charge its allocations through the budget
/// it is handed: a charge made on any other budget is invisible here.
/// [`OjasError::CapacityExceeded`] becomes a
/// [`PreflightRefusal`] with [`RefusalReason::DryRun`]: `estimated_bytes` is
/// the refused charge's `live + requested`, saturating at `u64::MAX`, and
/// `measured_peak_bytes` is the child's peak. A request so large that
/// `live + requested` overflows `u64` (which `Budget` reports as
/// [`OjasError::OutOfRange`]) is the same refusal with `u64::MAX` as the
/// estimate and the child's own live bytes. Any other step error is
/// [`PreflightError::Step`]. Zero or negative `widest_tokens` is refused
/// before the closure runs.
pub fn dry_run<B, R>(
    backend: &B,
    widest_tokens: i64,
    step_fn: impl FnOnce(&Budget) -> Result<R, OjasError>,
) -> Result<DryRun<R>, PreflightError>
where
    B: Backend + ?Sized,
{
    let tokens = positive_tokens(widest_tokens)?;
    let parent = backend.budget();
    let child = parent.child(parent.cap_bytes());
    match step_fn(&child) {
        Ok(value) => Ok(DryRun {
            value,
            peak_bytes: child.peak_bytes(),
        }),
        Err(OjasError::CapacityExceeded {
            requested,
            cap,
            live,
        }) => Err(PreflightError::Refused(PreflightRefusal {
            tokens,
            bucket: tokens,
            estimated_bytes: live.saturating_add(requested),
            cap_bytes: cap,
            live_bytes: live,
            reason: RefusalReason::DryRun,
            extrapolated: false,
            measured_peak_bytes: Some(child.peak_bytes()),
        })),
        Err(OjasError::OutOfRange {
            op: BUDGET_OVERFLOW_OP,
            ..
        }) => Err(PreflightError::Refused(PreflightRefusal {
            tokens,
            bucket: tokens,
            estimated_bytes: u64::MAX,
            cap_bytes: child.cap_bytes(),
            live_bytes: child.live_bytes().map_err(PreflightError::Step)?,
            reason: RefusalReason::DryRun,
            extrapolated: false,
            measured_peak_bytes: Some(child.peak_bytes()),
        })),
        Err(other) => Err(PreflightError::Step(other)),
    }
}

/// The `op` [`Budget::try_reserve`] reports when `live + requested` overflows
/// `u64`. Such a request can never fit under any cap, so the dry run treats it
/// as a capacity refusal with a saturated estimate.
const BUDGET_OVERFLOW_OP: &str = "Budget::try_reserve";

fn positive_tokens(tokens: i64) -> Result<u64, PreflightError> {
    u64::try_from(tokens)
        .ok()
        .filter(|&t| t > 0)
        .ok_or_else(|| invalid(format!("tokens must be positive, got {tokens}")))
}

fn calibration(detail: String) -> PreflightError {
    PreflightError::Calibration { detail }
}

fn invalid(detail: String) -> PreflightError {
    PreflightError::InvalidInput { detail }
}
