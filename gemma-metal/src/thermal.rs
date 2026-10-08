//! Sustained-load ("thermal") gate.
//!
//! Every other bench in this crate measures a burst: warm up, run 32 steps,
//! divide. That answers "how fast is a token when the GPU is cold" and says
//! nothing about the shape a user actually sees — a session that runs for
//! minutes while the die heats and the OS walks the clocks down.
//!
//! This gate runs one workload continuously for a fixed wall-clock duration,
//! chops it into fixed-length windows, and compares the tail of the run against
//! the head.
//!
//! # Separating decline from noise
//!
//! The first version of this gate failed the mini graph at `tail/head = 0.89`.
//! It was not throttling: a build and two editors were saturating the CPU, and
//! the mini graph is dispatch-bound, so window rates wandered between 24 and 50
//! tok/s with no trend at all. A gate that calls host contention a thermal
//! failure is a gate people learn to ignore, and one that quietly passes a real
//! throttle is worse. So the verdict is a documented decision procedure over
//! two independent statistics:
//!
//! - **`sustained_ratio`** — median of the last third over median of the first
//!   third. A *directional* statistic. Scheduling noise perturbs both thirds
//!   equally, so it does not move this; a part that slows down does.
//! - **`cv`** — coefficient of variation across windows. A *dispersion*
//!   statistic that says whether the run was clean enough to certify. Its
//!   default ceiling is calibrated from measured runs: quiet 0.039–0.075,
//!   contended 0.100–0.149.
//!
//! Order matters, and it is asymmetric on purpose: a depressed tail is reported
//! as a FAIL even when the run is noisy (noise is not an alibi for a real
//! decline), but a *stable* tail on a noisy run is `Skipped`, never `Pass` — the
//! run did not earn a clean bill of health. See [`evaluate`].
//!
//! # Honesty properties
//!
//! 1. **A check that could not run never reports the same result as a check
//!    that ran and passed.** [`Verdict`] is four-valued — `Pass`, `Fail`,
//!    `Skipped`, `Error` — and [`ThermalReport::passed`] is true for exactly
//!    one of them.
//! 2. **The verdict is a pure function of the samples.** [`evaluate`] takes
//!    windows and returns a report; it touches no clock and no GPU, so every
//!    edge case below has a unit test.
//! 3. **Degenerate input fails loud.** Zero-token windows, a zero-throughput
//!    head, non-finite samples and an out-of-range config are distinct named
//!    failures, never a silent pass.
//!
//! The window series doubles as a leak detector: RSS is sampled once per window
//! and growth across the run is gated.
//!
//! ```no_run
//! use gemma_metal::thermal::{run_sustained, ThermalGateConfig};
//! let cfg = ThermalGateConfig::default();          // 60 s, 5 s windows
//! let mut n = 0u64;
//! let report = run_sustained(&cfg, "demo", |batch| { n += batch as u64; Ok(batch) });
//! println!("{}", report.summary_line());
//! ```

use crate::diag;
use std::time::Instant;

/// Largest batch the calibrator will hand a work closure in one call.
const MAX_BATCH: usize = 8192;

/// Calibration aims for a call this fraction of a window, so a window holds
/// enough calls to average over but few enough to not overrun badly.
const CALIB_WINDOW_FRACTION: f64 = 1.0 / 8.0;

/// Wall-clock ceiling on batch calibration, independent of `warmup_secs`.
const CALIB_MAX_SECS: f64 = 4.0;

/// A trailing window shorter than this fraction of `window_secs` is discarded
/// rather than compared: a partial window holds fewer samples, so its rate is
/// noisier than the windows it would be judged against.
const MIN_WINDOW_FRACTION: f64 = 0.9;

/// Knobs for one sustained-load run. See [`ThermalGateConfig::validate`] for the
/// legal ranges — construction never panics; running an invalid config yields
/// [`Verdict::Error`].
#[derive(Clone, Debug)]
pub struct ThermalGateConfig {
    /// Wall-clock seconds of *measured* load, after warmup.
    pub duration_secs: f64,
    /// Length of one throughput window.
    pub window_secs: f64,
    /// Discarded lead-in: page faults, shader warm-up, clock boost.
    pub warmup_secs: f64,
    /// Gate on `tail_tok_s / head_tok_s` (median of last third over first third).
    pub min_sustained_ratio: f64,
    /// Gate on the worst two-window rolling mean over the **median** rolling
    /// mean. Two adjacent windows must be depressed together, so one scheduling
    /// stall cannot fail the run while a real sag still does; dividing by the
    /// median rather than the best pair keeps the statistic comparable across
    /// run lengths. Measured on this host: clean runs score 0.905–0.969, runs
    /// with a genuine multi-window sag score ≤ 0.730.
    pub min_rolling_ratio: f64,
    /// Dispersion ceiling. Above this the run is too noisy to *certify* — a
    /// stable-looking result becomes `Skipped`, not `Pass`. `None` disables the
    /// check, which is what a caller does when it knows the host is quiet.
    ///
    /// Calibrated against measured runs on an M5 Pro, not guessed: quiet runs
    /// land at cv 0.039–0.075, runs with visible host contention at 0.100–0.149.
    /// The default sits in that gap. Set it too loose and contended runs reach
    /// the sag gate, where ordinary scheduling produces a two-window dip and the
    /// gate reports FAIL for something that is not the GPU's fault.
    pub max_cv: Option<f64>,
    /// Fewer measured windows than this and the run is `Skipped`, not `Pass`.
    pub min_windows: usize,
    /// Optional absolute floor: every window must clear it.
    pub floor_tok_s: Option<f64>,
    /// Optional leak gate on RSS growth between the first and last window.
    pub max_rss_growth_mib: Option<f64>,
    /// Ceiling on the batch the calibrator may hand the work closure. Callers
    /// with a hard per-call limit — a KV cache that holds N tokens — set this so
    /// calibration stops at a batch the workload can actually honour instead of
    /// doubling to [`MAX_BATCH`] against a closure that silently clamps.
    pub max_batch: Option<usize>,
}

impl Default for ThermalGateConfig {
    fn default() -> Self {
        Self {
            duration_secs: 60.0,
            window_secs: 5.0,
            // Apple GPUs boost before settling; a short warmup leaves the boost
            // inside window 0 and manufactures a decline that is not thermal.
            warmup_secs: 10.0,
            min_sustained_ratio: 0.90,
            min_rolling_ratio: 0.85,
            max_cv: Some(0.09),
            // Each third of the run needs enough windows for its median to mean
            // something; below ~8 the ratio is one sample against one sample.
            min_windows: 8,
            floor_tok_s: None,
            max_rss_growth_mib: Some(512.0),
            max_batch: None,
        }
    }
}

impl ThermalGateConfig {
    /// Short run for smoke tests and CI, where a 60 s gate is not affordable.
    /// Same statistics, less wall clock — and correspondingly less power to
    /// resolve a small decline.
    pub fn quick() -> Self {
        Self {
            duration_secs: 24.0,
            window_secs: 2.0,
            warmup_secs: 4.0,
            min_windows: 8,
            ..Self::default()
        }
    }

    /// Every problem with this config, not just the first.
    ///
    /// Ratios above 1.0 are rejected deliberately: a gate demanding the tail be
    /// *faster* than the head can never pass on hardware that throttles, and is
    /// far more likely a typo than an intent.
    pub fn validate(&self) -> std::result::Result<(), String> {
        let mut errs: Vec<String> = Vec::new();
        let finite_pos = |v: f64| v.is_finite() && v > 0.0;

        if !finite_pos(self.duration_secs) {
            errs.push(format!(
                "duration_secs must be finite and > 0 (got {})",
                self.duration_secs
            ));
        }
        if !finite_pos(self.window_secs) {
            errs.push(format!(
                "window_secs must be finite and > 0 (got {})",
                self.window_secs
            ));
        }
        if finite_pos(self.duration_secs)
            && finite_pos(self.window_secs)
            && self.window_secs > self.duration_secs
        {
            errs.push(format!(
                "window_secs ({}) exceeds duration_secs ({}) — no window can complete",
                self.window_secs, self.duration_secs
            ));
        }
        if !self.warmup_secs.is_finite() || self.warmup_secs < 0.0 {
            errs.push(format!(
                "warmup_secs must be finite and >= 0 (got {})",
                self.warmup_secs
            ));
        }
        for (name, v) in [
            ("min_sustained_ratio", self.min_sustained_ratio),
            ("min_rolling_ratio", self.min_rolling_ratio),
        ] {
            if !v.is_finite() || v <= 0.0 || v > 1.0 {
                errs.push(format!("{name} must be finite and in (0, 1] (got {v})"));
            }
        }
        if let Some(cv) = self.max_cv {
            if !cv.is_finite() || cv <= 0.0 {
                errs.push(format!("max_cv must be finite and > 0 when set (got {cv})"));
            }
        }
        if self.min_windows == 0 {
            errs.push("min_windows must be >= 1 — a verdict needs at least one sample".into());
        }
        if let Some(f) = self.floor_tok_s {
            if !f.is_finite() || f < 0.0 {
                errs.push(format!("floor_tok_s must be finite and >= 0 (got {f})"));
            }
        }
        if let Some(m) = self.max_rss_growth_mib {
            if !m.is_finite() || m < 0.0 {
                errs.push(format!(
                    "max_rss_growth_mib must be finite and >= 0 (got {m})"
                ));
            }
        }
        if self.max_batch == Some(0) {
            errs.push("max_batch must be >= 1 when set — a batch of 0 does no work".into());
        }
        // A duration that cannot fit `min_windows` windows can only ever return
        // `Skipped`. Reject it up front rather than burning the wall clock to
        // discover it. Windows end on the first call that crosses the boundary,
        // so budget the trailing partial that `MIN_WINDOW_FRACTION` discards.
        if finite_pos(self.duration_secs) && finite_pos(self.window_secs) && self.min_windows > 0 {
            let fits = (self.duration_secs / self.window_secs).floor() as usize;
            if fits < self.min_windows {
                errs.push(format!(
                    "duration_secs {} / window_secs {} fits {fits} window(s), below min_windows {}",
                    self.duration_secs, self.window_secs, self.min_windows
                ));
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs.join("; "))
        }
    }
}

/// The OS's own view of thermal pressure, from `NSProcessInfo.thermalState`.
///
/// This is the one signal in this module that is *not* derived from throughput.
/// It is what lets the gate say "it slowed down **and** the OS reported heat"
/// rather than merely "it slowed down" — the exact ambiguity that made an early
/// version of this gate call CPU contention a thermal failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HostThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
    /// A value Apple added after these bindings were generated. Reported rather
    /// than folded into `Nominal`, so a future OS cannot silently downgrade real
    /// pressure to "no pressure".
    Unknown(isize),
}

impl HostThermalState {
    pub fn as_str(&self) -> &'static str {
        match self {
            HostThermalState::Nominal => "nominal",
            HostThermalState::Fair => "fair",
            HostThermalState::Serious => "serious",
            HostThermalState::Critical => "critical",
            HostThermalState::Unknown(_) => "unknown",
        }
    }

    /// True when the OS reports pressure that would actually cap clocks.
    ///
    /// `Unknown` counts as pressure: an unrecognised state is more likely a new
    /// level above `Critical` than a new one below `Nominal`, and the safe
    /// reading of "I do not know" is never "everything is fine".
    pub fn is_pressured(&self) -> bool {
        matches!(
            self,
            HostThermalState::Serious | HostThermalState::Critical | HostThermalState::Unknown(_)
        )
    }
}

/// Read `NSProcessInfo.thermalState` and `isLowPowerModeEnabled`.
///
/// Returns `None` only if the Objective-C call is unavailable, which on Apple
/// silicon it is not — but the gate treats a missing reading as missing, never
/// as `Nominal`.
fn host_thermal_sample() -> (Option<HostThermalState>, Option<bool>) {
    use objc2_foundation::{NSProcessInfo, NSProcessInfoThermalState};
    let info = NSProcessInfo::processInfo();
    let raw = info.thermalState();
    let state = match raw {
        NSProcessInfoThermalState::Nominal => HostThermalState::Nominal,
        NSProcessInfoThermalState::Fair => HostThermalState::Fair,
        NSProcessInfoThermalState::Serious => HostThermalState::Serious,
        NSProcessInfoThermalState::Critical => HostThermalState::Critical,
        other => HostThermalState::Unknown(other.0),
    };
    (Some(state), Some(info.isLowPowerModeEnabled()))
}

/// One measured window of sustained load.
#[derive(Clone, Debug, PartialEq)]
pub struct ThermalWindow {
    pub index: usize,
    /// Measured wall-clock length; not exactly `window_secs` — a window ends on
    /// the first work call that crosses the boundary.
    pub secs: f64,
    pub tokens: u64,
    pub tok_s: f64,
    /// Process RSS sampled at the end of the window, when `ps` was readable.
    pub rss_mib: Option<f64>,
    /// 1-minute host load average sampled at the end of the window, when
    /// readable. Diagnostic only — `cv` is what gates on noise, because load
    /// average says nothing about whether *this* run was disturbed.
    pub load_avg: Option<f64>,
    /// `NSProcessInfo.thermalState` at the end of the window.
    pub thermal_state: Option<HostThermalState>,
    /// `NSProcessInfo.isLowPowerModeEnabled` at the end of the window. Low Power
    /// Mode caps clocks by policy, which looks identical to a thermal decline
    /// and is not one.
    pub low_power: Option<bool>,
}

impl ThermalWindow {
    /// Build a window from raw counts, deriving `tok_s`.
    ///
    /// A non-positive or non-finite duration yields `tok_s = 0.0` rather than an
    /// infinity: [`evaluate`] then flags the window as stalled instead of
    /// propagating a NaN through the medians.
    pub fn new(index: usize, secs: f64, tokens: u64, rss_mib: Option<f64>) -> Self {
        Self::with_load(index, secs, tokens, rss_mib, None)
    }

    pub fn with_load(
        index: usize,
        secs: f64,
        tokens: u64,
        rss_mib: Option<f64>,
        load_avg: Option<f64>,
    ) -> Self {
        Self::sampled(index, secs, tokens, rss_mib, load_avg, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sampled(
        index: usize,
        secs: f64,
        tokens: u64,
        rss_mib: Option<f64>,
        load_avg: Option<f64>,
        thermal_state: Option<HostThermalState>,
        low_power: Option<bool>,
    ) -> Self {
        let tok_s = if secs.is_finite() && secs > 0.0 {
            tokens as f64 / secs
        } else {
            0.0
        };
        Self {
            index,
            secs,
            tokens,
            tok_s,
            rss_mib,
            load_avg,
            thermal_state,
            low_power,
        }
    }
}

/// Outcome of a sustained-load run. Four-valued on purpose: `Skipped` and
/// `Error` must never be readable as `Pass`.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Ran to completion and cleared every gate.
    Pass,
    /// Ran to completion and broke at least one gate.
    Fail(Vec<String>),
    /// Ran, but produced too little or too noisy a signal to judge. Not a pass.
    Skipped(String),
    /// The workload or the config broke. Not a pass, and not a throttle result.
    Error(String),
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail(_) => "FAIL",
            Verdict::Skipped(_) => "SKIPPED",
            Verdict::Error(_) => "ERROR",
        }
    }

    /// Human-readable cause, empty for `Pass`.
    pub fn detail(&self) -> String {
        match self {
            Verdict::Pass => String::new(),
            Verdict::Fail(rs) => rs.join("; "),
            Verdict::Skipped(r) | Verdict::Error(r) => r.clone(),
        }
    }
}

/// Full result of a sustained-load run: the samples, the derived statistics, and
/// the verdict.
#[derive(Clone, Debug)]
pub struct ThermalReport {
    pub label: String,
    pub verdict: Verdict,
    pub windows: Vec<ThermalWindow>,
    /// Windows dropped for being a short trailing remainder.
    pub dropped_partial_windows: usize,
    pub head_tok_s: f64,
    pub tail_tok_s: f64,
    pub best_tok_s: f64,
    pub worst_tok_s: f64,
    pub mean_tok_s: f64,
    /// `tail_tok_s / head_tok_s`. 0.0 when it could not be computed.
    pub sustained_ratio: f64,
    /// Worst two-window rolling mean over the median rolling mean. 0.0 when
    /// uncomputable.
    pub rolling_ratio: f64,
    /// Raw single-window `worst_tok_s / best_tok_s`, reported as data — the gate
    /// uses [`Self::rolling_ratio`], which one stall cannot dominate.
    pub worst_ratio: f64,
    /// Coefficient of variation of the window rates (population sd / mean).
    pub cv: f64,
    pub total_tokens: u64,
    pub measured_secs: f64,
    pub rss_growth_mib: Option<f64>,
    /// Batch size the calibrator settled on, for reproducing the run.
    pub batch: usize,
    /// Qualitative label **derived from the measured ratio** — this is not a
    /// reading of any OS or SMC thermal sensor. For that, see
    /// [`Self::peak_thermal_state`].
    pub throughput_state: &'static str,
    /// Highest `NSProcessInfo.thermalState` observed across the run — the OS's
    /// own reading, not a throughput inference.
    pub peak_thermal_state: Option<HostThermalState>,
    /// True if Low Power Mode was enabled in any window.
    pub low_power_any: Option<bool>,
    /// What the OS reading says about the measured result: whether a decline was
    /// corroborated as thermal, or contradicted. Never changes the verdict —
    /// throughput decides *whether* it slowed, this says *why*.
    pub attribution: String,
}

impl ThermalReport {
    /// True for `Verdict::Pass` and nothing else.
    pub fn passed(&self) -> bool {
        matches!(self.verdict, Verdict::Pass)
    }

    /// One line for stdout / CI logs.
    pub fn summary_line(&self) -> String {
        let detail = self.verdict.detail();
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!(" — {detail}")
        };
        let os = self.peak_thermal_state.map(|s| s.as_str()).unwrap_or("n/a");
        format!(
            "thermal[{}] {}: sustained={:.3} (head {:.1} → tail {:.1} tok/s), rolling={:.3}, \
             cv={:.3}, windows={} ({:.0}s), state={}, os_thermal={}{}",
            self.label,
            self.verdict.as_str(),
            self.sustained_ratio,
            self.head_tok_s,
            self.tail_tok_s,
            self.rolling_ratio,
            self.cv,
            self.windows.len(),
            self.measured_secs,
            self.throughput_state,
            os,
            tail
        )
    }

    /// Artifact body, matching the `bench/results/*.json` conventions.
    pub fn to_json(&self, cfg: &ThermalGateConfig) -> serde_json::Value {
        serde_json::json!({
            "artifact": "thermal_gate",
            "label": self.label,
            "verdict": self.verdict.as_str(),
            "verdict_detail": self.verdict.detail(),
            "passed": self.passed(),
            "sustained_ratio": self.sustained_ratio,
            "rolling_ratio": self.rolling_ratio,
            "worst_ratio": self.worst_ratio,
            "cv": self.cv,
            "head_tok_s": self.head_tok_s,
            "tail_tok_s": self.tail_tok_s,
            "best_tok_s": self.best_tok_s,
            "worst_tok_s": self.worst_tok_s,
            "mean_tok_s": self.mean_tok_s,
            "total_tokens": self.total_tokens,
            "measured_secs": self.measured_secs,
            "windows": self.windows.iter().map(|w| serde_json::json!({
                "index": w.index,
                "secs": w.secs,
                "tokens": w.tokens,
                "tok_s": w.tok_s,
                "rss_mib": w.rss_mib,
                "load_avg": w.load_avg,
                "thermal_state": w.thermal_state.map(|s| s.as_str()),
                "low_power": w.low_power,
            })).collect::<Vec<_>>(),
            "dropped_partial_windows": self.dropped_partial_windows,
            "rss_growth_mib": self.rss_growth_mib,
            "batch": self.batch,
            "throughput_state": self.throughput_state,
            "peak_thermal_state": self.peak_thermal_state.map(|s| s.as_str()),
            "low_power_any": self.low_power_any,
            "attribution": self.attribution,
            "config": {
                "duration_secs": cfg.duration_secs,
                "window_secs": cfg.window_secs,
                "warmup_secs": cfg.warmup_secs,
                "min_sustained_ratio": cfg.min_sustained_ratio,
                "min_rolling_ratio": cfg.min_rolling_ratio,
                "max_cv": cfg.max_cv,
                "min_windows": cfg.min_windows,
                "floor_tok_s": cfg.floor_tok_s,
                "max_rss_growth_mib": cfg.max_rss_growth_mib,
                "max_batch": cfg.max_batch,
            },
            "notes": [
                "Sustained-load gate: continuous decode chopped into fixed windows.",
                "sustained_ratio = median(last third) / median(first third) — directional.",
                "cv = population sd / mean over window rates — dispersion; high cv turns a \
                 stable-looking run into SKIPPED, never PASS, but never excuses a low \
                 sustained_ratio.",
                "rolling_ratio = min(2-window means) / median(2-window means): one stall cannot \
                 fail the run, and the statistic does not drift with run length the way a \
                 min/max would. worst_ratio is raw single-window data only.",
                "throughput_state is DERIVED FROM THROUGHPUT; peak_thermal_state is the OS's own \
                 NSProcessInfo.thermalState reading. `attribution` combines them and is advisory: \
                 it never changes the verdict.",
                "SKIPPED and ERROR are not passes; passed=true only for verdict=PASS.",
            ],
        })
    }
}

fn median(input: &[f64]) -> f64 {
    if input.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = input.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Population coefficient of variation. Zero for fewer than two samples or a
/// non-positive mean — callers must not read "0.0" as "perfectly clean" without
/// also checking the window count.
fn coefficient_of_variation(rates: &[f64]) -> f64 {
    if rates.len() < 2 {
        return 0.0;
    }
    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;
    if !mean.is_finite() || mean <= 0.0 {
        return 0.0;
    }
    let var = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd.is_finite() {
        sd / mean
    } else {
        0.0
    }
}

/// Means of every adjacent pair. Empty for fewer than two samples.
fn rolling2(rates: &[f64]) -> Vec<f64> {
    if rates.len() < 2 {
        return Vec::new();
    }
    rates.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect()
}

/// Explain a measured result against the OS's own thermal reading.
///
/// Deliberately advisory: it never changes the verdict. Throughput decides
/// *whether* the part slowed down; this decides *what to blame*, which is the
/// question a reader actually has when a gate goes red. `NSProcessInfo` is
/// coarse and lags, so letting it veto a measured decline would trade a
/// false alarm for a missed regression.
/// What the run established about throughput, derived from the verdict itself
/// rather than from the raw gate predicates.
///
/// The distinction matters: a run can trip the sag predicate and still be
/// `Skipped` because the dispersion check short-circuited first. Reading the
/// predicate alone once produced "this decline is NOT thermal" printed beside a
/// SKIPPED verdict that had established no decline at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeclineFinding {
    /// The verdict is a FAIL on a throughput arm.
    Declined,
    /// The run completed and throughput held.
    Held,
    /// No throughput conclusion was reached (skipped, errored).
    Inconclusive,
}

/// Explain a measured result against the OS's own thermal reading.
///
/// Deliberately advisory: it never changes the verdict. Throughput decides
/// *whether* the part slowed down; this decides *what to blame*, which is the
/// question a reader actually has when a gate goes red. `NSProcessInfo` is
/// coarse and lags, so letting it veto a measured decline would trade a false
/// alarm for a missed regression.
fn attribute(
    finding: DeclineFinding,
    peak: Option<HostThermalState>,
    low_power_any: Option<bool>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    match (finding, peak) {
        (_, None) => parts
            .push("OS thermal state unavailable — attribution is throughput-only".to_string()),
        (DeclineFinding::Declined, Some(state)) if state.is_pressured() => parts.push(format!(
            "OS reported thermal pressure ({}) during the run — the decline is corroborated as thermal",
            state.as_str()
        )),
        (DeclineFinding::Declined, Some(state)) => parts.push(format!(
            "OS reported no thermal pressure (peak {}) — this decline is NOT thermal; look at host contention or a code regression",
            state.as_str()
        )),
        (DeclineFinding::Held, Some(state)) if state.is_pressured() => parts.push(format!(
            "held rate under OS-reported thermal pressure ({})",
            state.as_str()
        )),
        (DeclineFinding::Held, Some(state)) => parts.push(format!(
            "no decline, and no OS thermal pressure (peak {})",
            state.as_str()
        )),
        (DeclineFinding::Inconclusive, Some(state)) if state.is_pressured() => parts.push(format!(
            "no throughput verdict was reached, and the OS reported thermal pressure (peak {}) during the run",
            state.as_str()
        )),
        (DeclineFinding::Inconclusive, Some(state)) => parts.push(format!(
            "no throughput verdict was reached; the OS reported no thermal pressure (peak {}), so whatever moved the numbers was not heat",
            state.as_str()
        )),
    }
    if low_power_any == Some(true) {
        parts.push(
            "Low Power Mode was enabled — clocks are capped by policy, not by heat".to_string(),
        );
    }
    parts.join("; ")
}

/// Label derived from the sustained ratio. Named for what it measures —
/// throughput — because nothing here reads a temperature.
fn throughput_state(ratio: f64) -> &'static str {
    if !ratio.is_finite() || ratio <= 0.0 {
        "unknown"
    } else if ratio >= 0.97 {
        "stable"
    } else if ratio >= 0.90 {
        "mild-decline"
    } else if ratio >= 0.75 {
        "declining"
    } else {
        "severe-decline"
    }
}

/// Turn samples into a verdict. Pure: no clock, no GPU, no I/O.
///
/// Decision procedure, in order:
/// 1. invalid config → `Error`
/// 2. workload failure → `Error`
/// 3. too few windows → `Skipped`
/// 4. a stalled window, or an unusable head → `Fail`
/// 5. `sustained_ratio` below threshold → `Fail` (a real decline; noise is not
///    an alibi, so this is checked *before* the dispersion test)
/// 6. `cv` above `max_cv` → `Skipped` (stable-looking, but not certifiable)
/// 7. `rolling_ratio` / floor / RSS gates → `Fail`
/// 8. otherwise → `Pass`
pub fn evaluate(
    cfg: &ThermalGateConfig,
    label: &str,
    windows: Vec<ThermalWindow>,
    dropped_partial_windows: usize,
    batch: usize,
    run_error: Option<String>,
) -> ThermalReport {
    let rates: Vec<f64> = windows.iter().map(|w| w.tok_s).collect();
    let total_tokens: u64 = windows.iter().map(|w| w.tokens).sum();
    // Fold from an explicit +0.0 rather than `.sum()`: std's `Sum for f64` uses
    // -0.0 as its identity, so an empty window list would report "-0s" in the
    // summary line and `-0.0` in the artifact JSON.
    let measured_secs: f64 = windows
        .iter()
        .map(|w| w.secs)
        .filter(|s| s.is_finite())
        .fold(0.0, |a, b| a + b);

    // Split into thirds; with fewer than 3 windows head and tail overlap, which
    // is honest — the ratio is then near 1 and `min_windows` is what gates.
    let n = rates.len();
    let third = (n / 3).max(1);
    let head_tok_s = median(&rates[..third.min(n)]);
    let tail_tok_s = median(&rates[n.saturating_sub(third)..]);
    let best_tok_s = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let worst_tok_s = rates.iter().copied().fold(f64::INFINITY, f64::min);
    let best_tok_s = if best_tok_s.is_finite() {
        best_tok_s
    } else {
        0.0
    };
    let worst_tok_s = if worst_tok_s.is_finite() {
        worst_tok_s
    } else {
        0.0
    };
    let mean_tok_s = if measured_secs > 0.0 {
        total_tokens as f64 / measured_secs
    } else {
        0.0
    };

    let ratio = |num: f64, den: f64| -> f64 {
        if den.is_finite() && den > 0.0 && num.is_finite() {
            num / den
        } else {
            0.0
        }
    };
    let sustained_ratio = ratio(tail_tok_s, head_tok_s);
    let worst_ratio = ratio(worst_tok_s, best_tok_s);
    let roll = rolling2(&rates);
    // Worst pair against the *median* pair, not the best pair. min/max compounds
    // two order statistics, so it drifts down as the window count rises purely
    // because `max` has more chances to be high — a 300 s run would fail where
    // a 60 s run of the same material passes. The median is stable in n, so this
    // ratio means the same thing at any run length.
    let rolling_ratio = if roll.is_empty() {
        worst_ratio
    } else {
        ratio(
            roll.iter().copied().fold(f64::INFINITY, f64::min),
            median(&roll),
        )
    };
    let cv = coefficient_of_variation(&rates);

    let peak_thermal_state = windows.iter().filter_map(|w| w.thermal_state).max();
    let low_power_any = if windows.iter().any(|w| w.low_power.is_some()) {
        Some(windows.iter().any(|w| w.low_power == Some(true)))
    } else {
        None
    };
    // Exactly the predicates the verdict gates on, evaluated once and reused
    // below, so the attribution wording can never disagree with the pass/fail it
    // explains. An earlier version derived `declined` from the sustained arm
    // alone and printed "no decline" underneath a rolling-arm FAIL.
    let head_usable = head_tok_s.is_finite() && head_tok_s > 0.0;
    let sustained_fail = head_usable && sustained_ratio < cfg.min_sustained_ratio;
    let rolling_fail = head_usable && rolling_ratio < cfg.min_rolling_ratio;
    // Attribution is filled in once the verdict is known — see the end of this
    // function. Deriving it from `sustained_fail`/`rolling_fail` alone printed
    // wording that contradicted verdicts which short-circuited before those arms
    // were reached.

    let rss_growth_mib = match (
        windows.first().and_then(|w| w.rss_mib),
        windows.last().and_then(|w| w.rss_mib),
    ) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some(b - a),
        _ => None,
    };

    let mut report = ThermalReport {
        label: label.to_string(),
        verdict: Verdict::Pass,
        windows,
        dropped_partial_windows,
        head_tok_s,
        tail_tok_s,
        best_tok_s,
        worst_tok_s,
        mean_tok_s,
        sustained_ratio,
        rolling_ratio,
        worst_ratio,
        cv,
        total_tokens,
        measured_secs,
        rss_growth_mib,
        batch,
        throughput_state: throughput_state(sustained_ratio),
        peak_thermal_state,
        low_power_any,
        attribution: String::new(),
    };

    let verdict = 'decide: {
        // 1–2: a broken config or workload is never reported as a throttle result,
        // and never as a pass.
        if let Err(e) = cfg.validate() {
            break 'decide Verdict::Error(format!("invalid config: {e}"));
        }
        if let Some(e) = run_error {
            break 'decide Verdict::Error(format!(
                "workload failed after {} window(s): {e}",
                report.windows.len()
            ));
        }
        // 3: too little signal.
        if report.windows.len() < cfg.min_windows {
            break 'decide Verdict::Skipped(format!(
                "only {} measured window(s), need {} — inconclusive, not a pass",
                report.windows.len(),
                cfg.min_windows
            ));
        }

        let mut fails: Vec<String> = Vec::new();

        // 4: a stalled or nonsensical window invalidates the rate series, so name it
        // before reporting ratios computed from it.
        let stalled: Vec<usize> = report
            .windows
            .iter()
            .filter(|w| w.tokens == 0 || !w.tok_s.is_finite() || w.tok_s <= 0.0)
            .map(|w| w.index)
            .collect();
        if !stalled.is_empty() {
            fails.push(format!(
            "{} window(s) produced no work (indices {stalled:?}) — the workload stalled under load",
            stalled.len()
        ));
        }
        if !head_usable {
            fails.push(format!(
                "head throughput was {head_tok_s} — cannot form a sustained ratio"
            ));
        }

        // 5: directional decline. Checked before the dispersion test so a noisy run
        // cannot launder a genuine slowdown into `Skipped`.
        if sustained_fail {
            fails.push(format!(
            "sustained ratio {sustained_ratio:.3} < {:.3} (head {head_tok_s:.1} → tail {tail_tok_s:.1} tok/s)",
            cfg.min_sustained_ratio
        ));
        }

        // 6: dispersion. Only reachable when nothing above failed — a clean-looking
        // but noisy run is inconclusive, not a pass.
        if fails.is_empty() {
            if let Some(max_cv) = cfg.max_cv {
                if cv > max_cv {
                    break 'decide Verdict::Skipped(format!(
                    "window throughput varied by cv={cv:.3} (limit {max_cv:.3}) with no decline in \
                     the tail — the host was too noisy to certify stability; rerun quiet"
                ));
                }
            }
        }

        // 7: remaining gates.
        if rolling_fail {
            fails.push(format!(
            "rolling 2-window ratio {rolling_ratio:.3} < {:.3} (best {best_tok_s:.1}, worst {worst_tok_s:.1} tok/s) \
             — two adjacent windows sagged together vs the median window",
            cfg.min_rolling_ratio
        ));
        }
        if let Some(floor) = cfg.floor_tok_s {
            let below: Vec<usize> = report
                .windows
                .iter()
                .filter(|w| w.tok_s < floor)
                .map(|w| w.index)
                .collect();
            if !below.is_empty() {
                fails.push(format!(
                "{} window(s) below floor {floor:.1} tok/s (indices {below:?}, worst {worst_tok_s:.1})",
                below.len()
            ));
            }
        }
        if let (Some(cap), Some(growth)) = (cfg.max_rss_growth_mib, rss_growth_mib) {
            if growth > cap {
                fails.push(format!(
                "RSS grew {growth:.1} MiB across the run, over the {cap:.1} MiB cap — probable leak"
            ));
            }
        }

        if fails.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail(fails)
        }
    };

    // Attribution follows the verdict that was actually reached, so the wording
    // can never claim more (or less) than the gate concluded.
    let finding = match &verdict {
        Verdict::Fail(_) if sustained_fail || rolling_fail => DeclineFinding::Declined,
        Verdict::Fail(_) | Verdict::Pass => DeclineFinding::Held,
        Verdict::Skipped(_) | Verdict::Error(_) => DeclineFinding::Inconclusive,
    };
    report.attribution = attribute(finding, peak_thermal_state, low_power_any);
    report.verdict = verdict;
    report
}

/// 1-minute host load average, when readable.
fn load_avg_1m() -> Option<f64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    // `{ 2.71 3.04 3.11 }`
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace().find_map(|t| {
        t.trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
            .parse::<f64>()
            .ok()
    })
}

/// Drive `work` under continuous load and judge the result.
///
/// `work(batch)` runs `batch` units and returns how many it actually completed
/// (tokens, steps — whatever the caller counts). Returning `Err` aborts the run
/// with [`Verdict::Error`]; returning `Ok(0)` repeatedly surfaces as a stalled
/// window, which fails.
///
/// The batch size is calibrated during warmup so one call lands near an eighth
/// of a window: long enough to amortize the timer, short enough that windows do
/// not overrun. Warmup samples are discarded.
pub fn run_sustained<F>(cfg: &ThermalGateConfig, label: &str, mut work: F) -> ThermalReport
where
    F: FnMut(usize) -> std::result::Result<usize, String>,
{
    if let Err(e) = cfg.validate() {
        // Report through `evaluate` so the `Error` shape is identical whichever
        // path rejected the run.
        return evaluate(cfg, label, Vec::new(), 0, 0, Some(e));
    }

    // --- Phase 1: calibrate batch. Bounded by the batch ceiling (at most 13
    // doublings) and by CALIB_MAX_SECS, so this terminates even if `work`
    // returns instantly.
    let target_call = cfg.window_secs * CALIB_WINDOW_FRACTION;
    let batch_cap = cfg.max_batch.unwrap_or(MAX_BATCH).clamp(1, MAX_BATCH);
    let mut batch = 1usize.min(batch_cap);
    let calib_start = Instant::now();
    loop {
        let t = Instant::now();
        if let Err(e) = work(batch) {
            return evaluate(
                cfg,
                label,
                Vec::new(),
                0,
                batch,
                Some(format!("during calibration: {e}")),
            );
        }
        let d = t.elapsed().as_secs_f64();
        if d >= target_call
            || batch >= batch_cap
            || calib_start.elapsed().as_secs_f64() >= CALIB_MAX_SECS
        {
            break;
        }
        batch = batch.saturating_mul(2).min(batch_cap);
    }
    diag::log(
        "thermal",
        format_args!(
            "{label}: calibrated batch={batch} (target {target_call:.3}s/call, calib {:.2}s)",
            calib_start.elapsed().as_secs_f64()
        ),
    );

    // --- Phase 2: remaining warmup at the chosen batch, discarded.
    let warm_start = Instant::now();
    while warm_start.elapsed().as_secs_f64() < cfg.warmup_secs {
        if let Err(e) = work(batch) {
            return evaluate(
                cfg,
                label,
                Vec::new(),
                0,
                batch,
                Some(format!("during warmup: {e}")),
            );
        }
    }

    // --- Phase 3: measured windows.
    let mut windows: Vec<ThermalWindow> = Vec::new();
    let mut dropped_partial = 0usize;
    let mut run_error: Option<String> = None;
    let run_start = Instant::now();
    let mut index = 0usize;

    'outer: while run_start.elapsed().as_secs_f64() < cfg.duration_secs {
        let w_start = Instant::now();
        let mut tokens: u64 = 0;
        loop {
            match work(batch) {
                Ok(n) => tokens = tokens.saturating_add(n as u64),
                Err(e) => {
                    run_error = Some(e);
                    break 'outer;
                }
            }
            if w_start.elapsed().as_secs_f64() >= cfg.window_secs
                || run_start.elapsed().as_secs_f64() >= cfg.duration_secs
            {
                break;
            }
        }
        let secs = w_start.elapsed().as_secs_f64();
        // Drop a short trailing remainder: it holds fewer samples than the
        // windows it would be compared against, and letting it into the tail
        // median fabricates a decline that did not happen.
        if secs < cfg.window_secs * MIN_WINDOW_FRACTION {
            dropped_partial += 1;
            break;
        }
        let (thermal_state, low_power) = host_thermal_sample();
        let w = ThermalWindow::sampled(
            index,
            secs,
            tokens,
            diag::rss_mib(),
            load_avg_1m(),
            thermal_state,
            low_power,
        );
        diag::log(
            "thermal",
            format_args!(
                "{label}: window {index} {:.2}s {tokens} tok {:.2} tok/s rss={:?} load={:?} \
                 os_thermal={:?} low_power={:?}",
                w.secs, w.tok_s, w.rss_mib, w.load_avg, w.thermal_state, w.low_power
            ),
        );
        windows.push(w);
        index += 1;
    }

    evaluate(cfg, label, windows, dropped_partial, batch, run_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(index: usize, tok_s: f64) -> ThermalWindow {
        // 5 s windows, tokens derived so `tok_s` is (very nearly) the rate.
        ThermalWindow::new(index, 5.0, (tok_s * 5.0).round() as u64, None)
    }

    fn wins(rates: &[f64]) -> Vec<ThermalWindow> {
        rates.iter().enumerate().map(|(i, &r)| win(i, r)).collect()
    }

    /// 9 windows so thirds are 3 wide; cv/rolling gates on by default.
    fn cfg9() -> ThermalGateConfig {
        ThermalGateConfig {
            duration_secs: 45.0,
            window_secs: 5.0,
            warmup_secs: 0.0,
            min_sustained_ratio: 0.90,
            min_rolling_ratio: 0.80,
            max_cv: Some(0.12),
            min_windows: 9,
            floor_tok_s: None,
            max_rss_growth_mib: None,
            max_batch: None,
        }
    }

    #[test]
    fn flat_run_passes() {
        let r = evaluate(&cfg9(), "flat", wins(&[40.0; 9]), 0, 1, None);
        assert!(r.passed(), "{}", r.summary_line());
        assert!((r.sustained_ratio - 1.0).abs() < 1e-3);
        assert_eq!(r.throughput_state, "stable");
        assert!(r.cv < 0.01);
    }

    #[test]
    fn throttled_run_fails_on_sustained_ratio() {
        let r = evaluate(
            &cfg9(),
            "throttle",
            wins(&[40.0, 40.0, 39.0, 36.0, 33.0, 30.0, 27.0, 26.0, 25.0]),
            0,
            1,
            None,
        );
        assert!(!r.passed());
        assert!(matches!(r.verdict, Verdict::Fail(_)));
        assert!(
            r.verdict.detail().contains("sustained ratio"),
            "{}",
            r.verdict.detail()
        );
    }

    #[test]
    fn a_noisy_run_never_launders_a_real_decline_into_skipped() {
        // cv is far over the limit *and* the tail is genuinely down. Step 5 runs
        // before step 6, so this must be FAIL, not SKIPPED.
        let r = evaluate(
            &cfg9(),
            "noisy-throttle",
            wins(&[60.0, 20.0, 58.0, 40.0, 15.0, 38.0, 20.0, 8.0, 18.0]),
            0,
            1,
            None,
        );
        assert!(r.cv > 0.12, "cv={}", r.cv);
        assert!(
            matches!(r.verdict, Verdict::Fail(_)),
            "{}",
            r.summary_line()
        );
        assert!(r.verdict.detail().contains("sustained ratio"));
    }

    #[test]
    fn a_noisy_but_trendless_run_is_skipped_not_passed() {
        // This is the mini-graph-under-a-busy-laptop case that motivated the
        // dispersion test: rates wander, the tail is not down, and the run
        // simply cannot certify stability.
        let r = evaluate(
            &cfg9(),
            "noisy-flat",
            wins(&[24.0, 50.0, 30.0, 47.0, 26.0, 44.0, 28.0, 48.0, 31.0]),
            0,
            1,
            None,
        );
        assert!(r.cv > 0.12, "cv={}", r.cv);
        assert!(!r.passed());
        assert!(
            matches!(r.verdict, Verdict::Skipped(_)),
            "{}",
            r.summary_line()
        );
        assert!(
            r.verdict.detail().contains("too noisy"),
            "{}",
            r.verdict.detail()
        );
    }

    #[test]
    fn disabling_max_cv_lets_a_noisy_flat_run_through() {
        let mut cfg = cfg9();
        cfg.max_cv = None;
        cfg.min_rolling_ratio = 0.1; // isolate the dispersion behaviour
        let r = evaluate(
            &cfg,
            "noisy-uncapped",
            wins(&[24.0, 50.0, 30.0, 47.0, 26.0, 44.0, 28.0, 48.0, 31.0]),
            0,
            1,
            None,
        );
        assert!(r.passed(), "{}", r.summary_line());
    }

    #[test]
    fn one_stall_does_not_fail_the_run_but_a_two_window_sag_does() {
        // Single deep dip: rolling means smooth it, cv stays under the limit.
        let single = evaluate(
            &cfg9(),
            "one-dip",
            wins(&[40.0, 40.0, 40.0, 34.0, 40.0, 40.0, 40.0, 40.0, 40.0]),
            0,
            1,
            None,
        );
        assert!(single.passed(), "{}", single.summary_line());

        // Two adjacent windows down by the same amount: a real sag.
        let double = evaluate(
            &cfg9(),
            "two-dip",
            wins(&[40.0, 40.0, 40.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0]),
            0,
            1,
            None,
        );
        assert!(!double.passed(), "{}", double.summary_line());
        assert!(
            double.verdict.detail().contains("rolling")
                || double.verdict.detail().contains("too noisy"),
            "{}",
            double.verdict.detail()
        );
    }

    #[test]
    fn too_few_windows_is_skipped_not_passed() {
        let r = evaluate(&cfg9(), "short", wins(&[40.0; 8]), 0, 1, None);
        assert!(!r.passed(), "8 windows under min_windows=9 must not pass");
        assert!(matches!(r.verdict, Verdict::Skipped(_)));
        assert_eq!(r.verdict.as_str(), "SKIPPED");
    }

    #[test]
    fn zero_windows_is_skipped_not_passed() {
        let r = evaluate(&cfg9(), "empty", Vec::new(), 0, 1, None);
        assert!(!r.passed());
        assert!(matches!(r.verdict, Verdict::Skipped(_)));
        assert_eq!(r.sustained_ratio, 0.0);
        assert_eq!(r.best_tok_s, 0.0);
        assert_eq!(r.mean_tok_s, 0.0);
        assert_eq!(r.cv, 0.0);
    }

    #[test]
    fn workload_error_outranks_good_windows() {
        let r = evaluate(
            &cfg9(),
            "err",
            wins(&[40.0; 9]),
            0,
            1,
            Some("Metal encode failed".into()),
        );
        assert!(!r.passed());
        assert!(matches!(r.verdict, Verdict::Error(_)));
        assert!(r.verdict.detail().contains("Metal encode failed"));
    }

    #[test]
    fn stalled_window_fails_loudly() {
        let mut ws = wins(&[40.0; 9]);
        ws[3] = ThermalWindow::new(3, 5.0, 0, None);
        let r = evaluate(&cfg9(), "stall", ws, 0, 1, None);
        assert!(!r.passed());
        assert!(
            r.verdict.detail().contains("stalled"),
            "{}",
            r.verdict.detail()
        );
    }

    #[test]
    fn zero_duration_window_does_not_produce_infinite_rate() {
        assert_eq!(ThermalWindow::new(0, 0.0, 100, None).tok_s, 0.0);
        assert_eq!(ThermalWindow::new(0, f64::NAN, 100, None).tok_s, 0.0);
        assert_eq!(ThermalWindow::new(0, -1.0, 100, None).tok_s, 0.0);
    }

    #[test]
    fn floor_gate_reports_offending_windows() {
        let mut cfg = cfg9();
        cfg.floor_tok_s = Some(38.0);
        let r = evaluate(
            &cfg,
            "floor",
            wins(&[40.0, 40.0, 37.0, 40.0, 40.0, 40.0, 40.0, 40.0, 39.0]),
            0,
            1,
            None,
        );
        assert!(!r.passed());
        assert!(
            r.verdict.detail().contains("below floor"),
            "{}",
            r.verdict.detail()
        );
        assert!(r.verdict.detail().contains("[2]"), "{}", r.verdict.detail());
    }

    #[test]
    fn rss_growth_gate_catches_a_leak() {
        let mut cfg = cfg9();
        cfg.max_rss_growth_mib = Some(100.0);
        let ws: Vec<_> = (0..9)
            .map(|i| ThermalWindow::new(i, 5.0, 200, Some(1000.0 + 50.0 * i as f64)))
            .collect();
        let r = evaluate(&cfg, "leak", ws, 0, 1, None);
        assert_eq!(r.rss_growth_mib, Some(400.0));
        assert!(!r.passed());
        assert!(
            r.verdict.detail().contains("probable leak"),
            "{}",
            r.verdict.detail()
        );
    }

    #[test]
    fn stable_rss_passes_the_leak_gate() {
        let mut cfg = cfg9();
        cfg.max_rss_growth_mib = Some(100.0);
        let ws: Vec<_> = (0..9)
            .map(|i| ThermalWindow::new(i, 5.0, 200, Some(1000.0 + (i % 2) as f64 * 3.0)))
            .collect();
        let r = evaluate(&cfg, "noleak", ws, 0, 1, None);
        assert!(r.passed(), "{}", r.summary_line());
    }

    #[test]
    fn missing_rss_samples_do_not_fabricate_a_verdict() {
        let mut cfg = cfg9();
        cfg.max_rss_growth_mib = Some(1.0);
        let r = evaluate(&cfg, "norss", wins(&[40.0; 9]), 0, 1, None);
        assert_eq!(r.rss_growth_mib, None);
        assert!(
            r.passed(),
            "an unreadable RSS must not fail the run: {}",
            r.verdict.detail()
        );
    }

    #[test]
    fn invalid_config_is_error_not_pass() {
        let mut cfg = cfg9();
        cfg.window_secs = 0.0;
        let r = evaluate(&cfg, "badcfg", wins(&[40.0; 9]), 0, 1, None);
        assert!(!r.passed());
        assert!(matches!(r.verdict, Verdict::Error(_)));
    }

    #[test]
    fn config_validation_names_every_problem() {
        let cfg = ThermalGateConfig {
            duration_secs: -1.0,
            window_secs: 0.0,
            warmup_secs: -3.0,
            min_sustained_ratio: 1.5,
            min_rolling_ratio: 0.0,
            max_cv: Some(0.0),
            min_windows: 0,
            floor_tok_s: Some(f64::NAN),
            max_rss_growth_mib: Some(-2.0),
            max_batch: Some(0),
        };
        let e = cfg.validate().unwrap_err();
        for expect in [
            "duration_secs",
            "window_secs",
            "warmup_secs",
            "min_sustained_ratio",
            "min_rolling_ratio",
            "max_cv",
            "min_windows",
            "floor_tok_s",
            "max_rss_growth_mib",
            "max_batch",
        ] {
            assert!(e.contains(expect), "validate() missed {expect}: {e}");
        }
    }

    #[test]
    fn config_rejects_a_duration_that_cannot_reach_min_windows() {
        let cfg = ThermalGateConfig {
            duration_secs: 10.0,
            window_secs: 5.0,
            min_windows: 4, // only 2 windows fit
            ..ThermalGateConfig::default()
        };
        assert!(cfg.validate().unwrap_err().contains("min_windows"));
    }

    #[test]
    fn defaults_and_quick_are_self_consistent() {
        ThermalGateConfig::default()
            .validate()
            .expect("default config must be runnable");
        ThermalGateConfig::quick()
            .validate()
            .expect("quick config must be runnable");
    }

    #[test]
    fn run_sustained_drives_a_closure_and_passes() {
        let cfg = ThermalGateConfig {
            duration_secs: 2.0,
            window_secs: 0.2,
            warmup_secs: 0.05,
            min_windows: 8,
            max_cv: Some(0.35), // sleep-based work under a busy CI host
            max_rss_growth_mib: None,
            ..ThermalGateConfig::default()
        };
        let mut calls = 0usize;
        let r = run_sustained(&cfg, "closure", |batch| {
            calls += 1;
            std::thread::sleep(std::time::Duration::from_micros(200 * batch as u64));
            Ok(batch)
        });
        assert!(
            r.passed() || matches!(r.verdict, Verdict::Skipped(_)),
            "a flat sleep workload must pass or be inconclusive, never fail: {}",
            r.summary_line()
        );
        assert!(r.windows.len() >= 8, "windows={}", r.windows.len());
        assert!(calls > 0 && r.batch >= 1 && r.total_tokens > 0);
    }

    #[test]
    fn run_sustained_surfaces_a_mid_run_failure_as_error() {
        let cfg = ThermalGateConfig {
            duration_secs: 2.0,
            window_secs: 0.2,
            warmup_secs: 0.0,
            min_windows: 1,
            max_rss_growth_mib: None,
            ..ThermalGateConfig::default()
        };
        let mut n = 0usize;
        let r = run_sustained(&cfg, "boom", |batch| {
            n += 1;
            std::thread::sleep(std::time::Duration::from_millis(5));
            if n > 20 {
                Err("GPU command buffer error".into())
            } else {
                Ok(batch)
            }
        });
        assert!(!r.passed());
        assert!(
            matches!(r.verdict, Verdict::Error(_)),
            "{}",
            r.summary_line()
        );
        assert!(r.verdict.detail().contains("GPU command buffer error"));
    }

    #[test]
    fn run_sustained_with_instant_work_terminates_at_the_batch_cap() {
        let cfg = ThermalGateConfig {
            duration_secs: 0.8,
            window_secs: 0.2,
            warmup_secs: 0.0,
            min_windows: 2,
            max_rss_growth_mib: None,
            ..ThermalGateConfig::default()
        };
        let r = run_sustained(&cfg, "instant", Ok);
        assert!(r.batch <= MAX_BATCH);
        assert!(r.windows.len() >= 2, "{}", r.summary_line());
    }

    #[test]
    fn run_sustained_honours_max_batch() {
        let cfg = ThermalGateConfig {
            duration_secs: 0.8,
            window_secs: 0.2,
            warmup_secs: 0.0,
            min_windows: 2,
            max_batch: Some(4),
            max_rss_growth_mib: None,
            ..ThermalGateConfig::default()
        };
        let mut seen_max = 0usize;
        let r = run_sustained(&cfg, "capped", |batch| {
            seen_max = seen_max.max(batch);
            Ok(batch)
        });
        assert_eq!(r.batch, 4, "calibration must stop at max_batch");
        assert_eq!(
            seen_max, 4,
            "the closure must never be handed more than max_batch"
        );
    }

    #[test]
    fn run_sustained_rejects_an_invalid_config_without_running() {
        let cfg = ThermalGateConfig {
            duration_secs: 5.0,
            window_secs: 10.0,
            ..ThermalGateConfig::default()
        };
        let mut called = false;
        let r = run_sustained(&cfg, "nogo", |b| {
            called = true;
            Ok(b)
        });
        assert!(!called, "an invalid config must not execute the workload");
        assert!(matches!(r.verdict, Verdict::Error(_)));
    }

    #[test]
    fn json_body_marks_skipped_as_not_passed() {
        let cfg = cfg9();
        let r = evaluate(&cfg, "j", vec![win(0, 40.0)], 0, 1, None);
        let v = r.to_json(&cfg);
        assert_eq!(v["verdict"], "SKIPPED");
        assert_eq!(v["passed"], false);
        assert_eq!(v["artifact"], "thermal_gate");
        assert!(v["config"]["window_secs"].as_f64().is_some());
        assert!(v["cv"].as_f64().is_some());
    }

    #[test]
    fn an_empty_run_reports_positive_zero_seconds() {
        // std's `Sum for f64` folds from -0.0, which formats as "-0s" and
        // serialises as -0.0. Neither belongs in an artifact.
        let cfg = cfg9();
        let r = evaluate(&cfg, "empty", Vec::new(), 0, 0, None);
        assert_eq!(r.measured_secs, 0.0);
        assert!(
            r.measured_secs.is_sign_positive(),
            "measured_secs was negative zero: {:?}",
            r.measured_secs
        );
        assert!(
            !r.summary_line().contains("-0s"),
            "summary line leaked a negative zero: {}",
            r.summary_line()
        );
        assert_eq!(r.to_json(&cfg)["measured_secs"].as_f64(), Some(0.0));
    }

    fn wins_with(rates: &[f64], state: HostThermalState, low_power: bool) -> Vec<ThermalWindow> {
        rates
            .iter()
            .enumerate()
            .map(|(i, &r)| {
                ThermalWindow::sampled(
                    i,
                    5.0,
                    (r * 5.0).round() as u64,
                    None,
                    None,
                    Some(state),
                    Some(low_power),
                )
            })
            .collect()
    }

    const DECLINE: [f64; 9] = [40.0, 40.0, 39.0, 36.0, 33.0, 30.0, 27.0, 26.0, 25.0];

    #[test]
    fn a_decline_with_os_thermal_pressure_is_attributed_to_heat() {
        let r = evaluate(
            &cfg9(),
            "hot",
            wins_with(&DECLINE, HostThermalState::Serious, false),
            0,
            1,
            None,
        );
        assert!(matches!(r.verdict, Verdict::Fail(_)));
        assert_eq!(r.peak_thermal_state, Some(HostThermalState::Serious));
        assert!(
            r.attribution.contains("corroborated as thermal"),
            "{}",
            r.attribution
        );
    }

    #[test]
    fn a_decline_without_os_thermal_pressure_is_attributed_elsewhere() {
        // The case that motivated reading NSProcessInfo at all: throughput fell,
        // the OS says the machine is cool, so the cause is contention or a
        // regression — and the report must say so rather than let "thermal gate
        // FAILED" imply heat.
        let r = evaluate(
            &cfg9(),
            "cool",
            wins_with(&DECLINE, HostThermalState::Nominal, false),
            0,
            1,
            None,
        );
        assert!(matches!(r.verdict, Verdict::Fail(_)));
        assert!(r.attribution.contains("NOT thermal"), "{}", r.attribution);
        assert!(r.attribution.contains("contention"), "{}", r.attribution);
    }

    #[test]
    fn os_reading_never_overrides_the_measured_verdict() {
        // Nominal must not rescue a real decline into a PASS, and pressure must
        // not fail a run that held its rate. Attribution is advisory only.
        let cool_decline = evaluate(
            &cfg9(),
            "cool",
            wins_with(&DECLINE, HostThermalState::Nominal, false),
            0,
            1,
            None,
        );
        assert!(
            !cool_decline.passed(),
            "nominal OS state must not rescue a decline"
        );

        let hot_flat = evaluate(
            &cfg9(),
            "hot-flat",
            wins_with(&[40.0; 9], HostThermalState::Critical, false),
            0,
            1,
            None,
        );
        assert!(hot_flat.passed(), "{}", hot_flat.summary_line());
        assert!(
            hot_flat
                .attribution
                .contains("held rate under OS-reported thermal pressure"),
            "{}",
            hot_flat.attribution
        );
    }

    #[test]
    fn low_power_mode_is_called_out_separately_from_heat() {
        let r = evaluate(
            &cfg9(),
            "lpm",
            wins_with(&DECLINE, HostThermalState::Nominal, true),
            0,
            1,
            None,
        );
        assert_eq!(r.low_power_any, Some(true));
        assert!(
            r.attribution.contains("Low Power Mode"),
            "{}",
            r.attribution
        );
        assert!(r.attribution.contains("not by heat"), "{}", r.attribution);
    }

    #[test]
    fn a_missing_os_reading_is_not_reported_as_nominal() {
        let r = evaluate(&cfg9(), "noos", wins(&DECLINE), 0, 1, None);
        assert_eq!(r.peak_thermal_state, None);
        assert_eq!(r.low_power_any, None);
        assert!(r.attribution.contains("unavailable"), "{}", r.attribution);
        assert!(
            !r.attribution.contains("no thermal pressure"),
            "an unavailable reading must not claim the machine was cool: {}",
            r.attribution
        );
    }

    #[test]
    fn peak_thermal_state_takes_the_worst_window_not_the_last() {
        let mut ws = wins_with(&[40.0; 9], HostThermalState::Nominal, false);
        ws[4].thermal_state = Some(HostThermalState::Critical);
        let r = evaluate(&cfg9(), "peak", ws, 0, 1, None);
        assert_eq!(r.peak_thermal_state, Some(HostThermalState::Critical));
    }

    #[test]
    fn an_unrecognised_os_state_counts_as_pressure_not_as_calm() {
        // A level Apple adds later is far more likely above Critical than below
        // Nominal, so "I do not know" must never read as "everything is fine".
        assert!(HostThermalState::Unknown(9).is_pressured());
        assert!(!HostThermalState::Nominal.is_pressured());
        assert!(!HostThermalState::Fair.is_pressured());
        assert!(HostThermalState::Serious.is_pressured());
        assert!(HostThermalState::Critical.is_pressured());
        // Ord must rank it as the worst so `peak` cannot be masked by Nominal.
        assert!(HostThermalState::Unknown(9) > HostThermalState::Critical);
    }

    #[test]
    fn the_os_thermal_probe_answers_on_this_host() {
        // Not a mock: this is the real NSProcessInfo call. It must return a
        // reading (any of the four) and a Low Power Mode flag.
        let (state, low_power) = host_thermal_sample();
        let state = state.expect("NSProcessInfo.thermalState returned nothing on macOS");
        assert!(
            matches!(
                state,
                HostThermalState::Nominal
                    | HostThermalState::Fair
                    | HostThermalState::Serious
                    | HostThermalState::Critical
            ),
            "unexpected thermal state {state:?}"
        );
        assert!(low_power.is_some());
        eprintln!(
            "host thermal state = {} low_power = {low_power:?}",
            state.as_str()
        );
    }

    #[test]
    fn json_carries_the_os_reading_and_attribution() {
        let cfg = cfg9();
        let r = evaluate(
            &cfg,
            "j2",
            wins_with(&[40.0; 9], HostThermalState::Fair, false),
            0,
            1,
            None,
        );
        let v = r.to_json(&cfg);
        assert_eq!(v["peak_thermal_state"], "fair");
        assert_eq!(v["low_power_any"], false);
        assert!(v["attribution"].as_str().is_some_and(|a| !a.is_empty()));
        assert_eq!(v["windows"][0]["thermal_state"], "fair");
        assert_eq!(v["windows"][0]["low_power"], false);
    }

    #[test]
    fn a_single_fast_window_cannot_fail_the_run() {
        // The defect this guards: min/max compounds two order statistics, so one
        // unusually *fast* window raises `max` and drags the ratio down. A run
        // then gets likelier to fail the longer it runs and the luckier one
        // window gets — which is why the real 300 s endurance run scored 0.691
        // min/max against 0.905-0.927 for 11-window runs of the same material.
        // Dividing by the median makes the statistic depend on the sag, not on
        // the best moment.
        let base = [55.0, 55.0, 55.0, 48.0, 48.0, 55.0, 55.0, 55.0, 55.0];
        let mut spiked = base;
        spiked[6] = 65.0; // one fast window, nothing else changed

        let r_base = evaluate(&cfg9(), "base", wins(&base), 0, 1, None);
        let r_spike = evaluate(&cfg9(), "spiked", wins(&spiked), 0, 1, None);

        assert!(r_base.passed(), "{}", r_base.summary_line());
        assert!(
            r_spike.passed(),
            "one fast window must not fail an otherwise identical run: {}",
            r_spike.summary_line()
        );
        assert!(
            (r_base.rolling_ratio - r_spike.rolling_ratio).abs() < 0.03,
            "rolling ratio moved {:.3} -> {:.3} because a single window was fast",
            r_base.rolling_ratio,
            r_spike.rolling_ratio
        );
    }

    #[test]
    fn rolling_gate_still_catches_a_real_sag_in_a_long_run() {
        // A genuine multi-window depression must fail regardless of length.
        let mut rates = vec![55.0f64; 27];
        for r in rates.iter_mut().skip(10).take(3) {
            *r = 38.0;
        }
        let r = evaluate(&cfg9(), "sag", wins(&rates), 0, 1, None);
        assert!(!r.passed(), "{}", r.summary_line());
        assert!(
            r.verdict.detail().contains("rolling"),
            "{}",
            r.verdict.detail()
        );
    }

    #[test]
    fn attribution_agrees_with_a_rolling_only_failure() {
        // Regression: `declined` once came from the sustained arm alone, so a
        // rolling-arm FAIL printed "no decline" beside it.
        let mut rates = vec![55.0f64; 27];
        for r in rates.iter_mut().skip(10).take(3) {
            *r = 38.0;
        }
        let ws: Vec<ThermalWindow> = rates
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                ThermalWindow::sampled(
                    i,
                    5.0,
                    (x * 5.0).round() as u64,
                    None,
                    None,
                    Some(HostThermalState::Nominal),
                    Some(false),
                )
            })
            .collect();
        let r = evaluate(&cfg9(), "rolling-only", ws, 0, 1, None);
        assert!(
            matches!(r.verdict, Verdict::Fail(_)),
            "{}",
            r.summary_line()
        );
        assert!(
            !r.verdict.detail().contains("sustained ratio"),
            "this run must fail on the rolling arm only: {}",
            r.verdict.detail()
        );
        assert!(
            !r.attribution.contains("no decline"),
            "attribution contradicted a FAIL verdict: {}",
            r.attribution
        );
        assert!(r.attribution.contains("NOT thermal"), "{}", r.attribution);
    }

    #[test]
    fn a_noise_skipped_run_does_not_claim_a_decline_or_deny_one() {
        // Regression: a run whose sag predicate trips but whose verdict
        // short-circuits to SKIPPED on dispersion printed "this decline is NOT
        // thermal" — asserting a decline the gate never concluded.
        let mut cfg = cfg9();
        cfg.max_cv = Some(0.05);
        // Two *adjacent* low windows, so the sag predicate trips (alternating
        // windows would be smoothed away by the rolling mean, which is the whole
        // point of that statistic), with dispersion above the ceiling and a flat
        // head-vs-tail so the sustained arm stays quiet.
        let rates: [f64; 9] = [55.0, 55.0, 55.0, 30.0, 30.0, 55.0, 55.0, 55.0, 55.0];
        let ws: Vec<ThermalWindow> = rates
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                ThermalWindow::sampled(
                    i,
                    5.0,
                    (x * 5.0).round() as u64,
                    None,
                    None,
                    Some(HostThermalState::Nominal),
                    Some(false),
                )
            })
            .collect();
        let r = evaluate(&cfg, "noise", ws, 0, 1, None);
        assert!(
            matches!(r.verdict, Verdict::Skipped(_)),
            "{}",
            r.summary_line()
        );
        assert!(
            r.rolling_ratio < cfg.min_rolling_ratio,
            "this fixture must trip the sag predicate to be meaningful (got {:.3})",
            r.rolling_ratio
        );
        assert!(
            r.attribution.contains("no throughput verdict was reached"),
            "attribution overclaimed on a SKIPPED run: {}",
            r.attribution
        );
        assert!(
            !r.attribution.contains("this decline is NOT thermal"),
            "{}",
            r.attribution
        );
        assert!(
            !r.attribution.contains("no decline, and"),
            "{}",
            r.attribution
        );
    }

    #[test]
    fn an_errored_run_makes_no_thermal_claim() {
        let r = evaluate(
            &cfg9(),
            "err",
            wins_with(&[40.0; 9], HostThermalState::Nominal, false),
            0,
            1,
            Some("Metal encode failed".into()),
        );
        assert!(matches!(r.verdict, Verdict::Error(_)));
        assert!(
            r.attribution.contains("no throughput verdict was reached"),
            "{}",
            r.attribution
        );
    }

    #[test]
    fn median_handles_even_and_odd_lengths() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[3.0]), 3.0);
        assert_eq!(median(&[1.0, 3.0]), 2.0);
        assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn cv_and_rolling_helpers_handle_degenerate_input() {
        assert_eq!(coefficient_of_variation(&[]), 0.0);
        assert_eq!(coefficient_of_variation(&[5.0]), 0.0);
        assert_eq!(coefficient_of_variation(&[0.0, 0.0]), 0.0);
        assert!((coefficient_of_variation(&[10.0, 10.0]) - 0.0).abs() < 1e-12);
        assert!((coefficient_of_variation(&[8.0, 12.0]) - 0.2).abs() < 1e-12);
        assert!(rolling2(&[]).is_empty());
        assert!(rolling2(&[1.0]).is_empty());
        assert_eq!(rolling2(&[1.0, 3.0, 5.0]), vec![2.0, 4.0]);
    }
}
