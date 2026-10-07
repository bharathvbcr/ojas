//! Check outcomes and comparisons, shared by the `#[ignore]` device tests and
//! the rung-0 binary, so both judge a device result by the same code.
//!
//! A check that could not run ([`Status::NotRun`]) or panicked
//! ([`Status::Panicked`]) is never [`Status::Pass`], and [`overall`] is
//! `Pass` only when every check ran and passed.

use crate::error::CudaError;
use crate::json::{Json, JsonObj};

/// How one check ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Ran, and met its criterion.
    Pass,
    /// Ran, and did not.
    Fail,
    /// Panicked (inside cudarc, e.g. a missing symbol); caught and recorded.
    Panicked,
    /// Did not run; the detail says why.
    NotRun,
}

impl Status {
    /// `pass`, `fail`, `panicked`, `not_run`.
    pub fn name(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "fail",
            Status::Panicked => "panicked",
            Status::NotRun => "not_run",
        }
    }
}

/// One named check and its evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// Stable name, e.g. `k0.cast_f32_to_bf16`.
    pub name: String,
    /// Outcome.
    pub status: Status,
    /// Human-readable evidence or failure reason.
    pub detail: String,
    /// Numbers behind the outcome.
    pub metrics: JsonObj,
}

impl Check {
    fn new(name: &str, status: Status, detail: impl Into<String>) -> Self {
        Check {
            name: name.to_string(),
            status,
            detail: detail.into(),
            metrics: JsonObj::new(),
        }
    }

    /// A passing check.
    pub fn pass(name: &str, detail: impl Into<String>) -> Self {
        Check::new(name, Status::Pass, detail)
    }

    /// A failing check.
    pub fn fail(name: &str, detail: impl Into<String>) -> Self {
        Check::new(name, Status::Fail, detail)
    }

    /// A check that did not run.
    pub fn not_run(name: &str, reason: impl Into<String>) -> Self {
        Check::new(name, Status::NotRun, reason)
    }

    /// A check that panicked, with the panic payload's text.
    pub fn panicked(name: &str, payload: impl Into<String>) -> Self {
        Check::new(name, Status::Panicked, payload)
    }

    /// A check whose setup or device call failed.
    pub fn from_error(name: &str, err: &CudaError) -> Self {
        Check::new(name, Status::Fail, format!("{} error: {err}", err.kind()))
    }

    /// Attach a metric.
    pub fn with(mut self, key: &str, value: impl Into<Json>) -> Self {
        self.metrics.push(key, value);
        self
    }

    /// The report object.
    pub fn to_json(&self) -> Json {
        JsonObj::new()
            .with("name", self.name.as_str())
            .with("status", self.status.name())
            .with("detail", self.detail.as_str())
            .with("metrics", self.metrics.clone())
            .into()
    }
}

/// `Pass` iff there is at least one check and every check passed.
pub fn overall(checks: &[Check]) -> Status {
    if checks.is_empty() {
        return Status::NotRun;
    }
    if checks.iter().any(|c| c.status == Status::Panicked) {
        return Status::Panicked;
    }
    if checks.iter().any(|c| c.status == Status::Fail) {
        return Status::Fail;
    }
    if checks.iter().any(|c| c.status == Status::NotRun) {
        return Status::NotRun;
    }
    Status::Pass
}

/// A bitwise comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitDiff {
    /// Elements compared.
    pub compared: usize,
    /// Elements whose bits differ (a length difference counts every extra one).
    pub mismatches: usize,
    /// The first mismatch: index, got bits, want bits.
    pub first: Option<(usize, u64, u64)>,
}

/// `bits` maps a value to its bit pattern; `is_nan` says whether it is a NaN.
/// Two NaNs match whatever their payloads (Fable's NaN ruling, 2026-10-02:
/// kernels do not canonicalise, NaN-equivalence lives here, and bitwise
/// claims are for finite values). Every other pair must match bit for bit,
/// so `-0.0` against `+0.0`, or a NaN against a number, is a mismatch.
fn diff_by<T: Copy>(
    got: &[T],
    want: &[T],
    bits: impl Fn(T) -> u64,
    is_nan: impl Fn(T) -> bool,
) -> BitDiff {
    let mut d = BitDiff {
        compared: got.len().min(want.len()),
        mismatches: got.len().abs_diff(want.len()),
        first: None,
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if bits(g) != bits(w) && !(is_nan(g) && is_nan(w)) {
            d.mismatches += 1;
            if d.first.is_none() {
                d.first = Some((i, bits(g), bits(w)));
            }
        }
    }
    d
}

/// Compare f32 slices by their bit patterns; any NaN matches any NaN.
pub fn diff_bits_f32(got: &[f32], want: &[f32]) -> BitDiff {
    diff_by(got, want, |x| u64::from(x.to_bits()), f32::is_nan)
}

/// Compare bf16 bit slices; any bf16 NaN matches any bf16 NaN.
pub fn diff_bits_bf16(got: &[u16], want: &[u16]) -> BitDiff {
    diff_by(got, want, u64::from, |b| b & 0x7fff > 0x7f80)
}

/// A bitwise check: passes iff the lengths agree and no element differs.
pub fn bitwise_check(name: &str, d: BitDiff, want_len: usize) -> Check {
    let check = if d.mismatches == 0 && d.compared == want_len {
        Check::pass(name, format!("{} elements bit-identical", d.compared))
    } else {
        let first = d.first.map_or("length differs".to_string(), |(i, g, w)| {
            format!("first at {i}: got {g:#x}, want {w:#x}")
        });
        Check::fail(
            name,
            format!("{} of {want_len} elements differ; {first}", d.mismatches),
        )
    };
    check
        .with("compared", d.compared)
        .with("mismatches", d.mismatches)
}

/// Error of an f32 result against a float64 reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TolReport {
    /// Largest `|got - want|` over finite outputs.
    pub max_abs_err: f64,
    /// Where it occurs.
    pub argmax: usize,
    /// Outputs that are NaN or infinite.
    pub nonfinite: usize,
    /// Largest `|want|`, for relative bounds.
    pub max_abs_ref: f64,
}

/// Measure `got` against `want`. Lengths must match.
pub fn tolerance_vs_f64(got: &[f32], want: &[f64]) -> Result<TolReport, CudaError> {
    if got.len() != want.len() {
        return Err(CudaError::invalid(
            "tolerance_vs_f64",
            format!("{} results against {} references", got.len(), want.len()),
        ));
    }
    let mut r = TolReport {
        max_abs_err: 0.0,
        argmax: 0,
        nonfinite: 0,
        max_abs_ref: 0.0,
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        r.max_abs_ref = r.max_abs_ref.max(w.abs());
        if !g.is_finite() {
            r.nonfinite += 1;
            continue;
        }
        let e = (f64::from(g) - w).abs();
        if e > r.max_abs_err {
            r.max_abs_err = e;
            r.argmax = i;
        }
    }
    Ok(r)
}

/// A tolerance check: passes iff every output is finite and the largest
/// error is strictly below `bound` (tessl's `err < atol`). `source` cites
/// where the bound comes from.
pub fn tolerance_check(name: &str, r: TolReport, bound: f64, source: &str) -> Check {
    let check = if r.nonfinite == 0 && r.max_abs_err < bound {
        Check::pass(
            name,
            format!("max |err| {:.3e} < {bound:.3e} ({source})", r.max_abs_err),
        )
    } else {
        Check::fail(
            name,
            format!(
                "max |err| {:.3e} at {} against bound {bound:.3e} ({source}); {} non-finite",
                r.max_abs_err, r.argmax, r.nonfinite
            ),
        )
    };
    check
        .with("max_abs_err", r.max_abs_err)
        .with("argmax", r.argmax)
        .with("nonfinite", r.nonfinite)
        .with("max_abs_ref", r.max_abs_ref)
        .with("bound", bound)
        .with("bound_source", source)
}

/// The text of a caught panic payload.
pub fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_not_run_or_panicked_check_never_makes_the_whole_pass() {
        let pass = Check::pass("a", "ok");
        assert_eq!(overall(std::slice::from_ref(&pass)), Status::Pass);
        assert_eq!(
            overall(&[pass.clone(), Check::not_run("b", "no gpu")]),
            Status::NotRun
        );
        assert_eq!(
            overall(&[pass.clone(), Check::panicked("c", "boom")]),
            Status::Panicked
        );
        assert_eq!(overall(&[pass, Check::fail("d", "bad")]), Status::Fail);
        assert_eq!(overall(&[]), Status::NotRun);
    }

    /// Fable's NaN ruling (2026-10-02): any NaN matches any NaN, payload and
    /// sign included; every other value matches only its own bits, so the
    /// sign of zero and a NaN against a number still count.
    #[test]
    fn bit_diffs_match_any_nan_and_count_everything_else() {
        let nan_a = f32::from_bits(0x7fc0_0000);
        let nan_b = f32::from_bits(0xffc0_0001);
        let d = diff_bits_f32(&[1.0, nan_a, -0.0, nan_a], &[1.0, nan_b, 0.0, 2.0]);
        assert_eq!(d.mismatches, 2);
        assert_eq!(d.first, Some((2, 0x8000_0000, 0)));
        // bf16: 0x7fc0 and 0xffc1 are both NaN; 0x7f80 (+inf) is not.
        let b = diff_bits_bf16(&[0x7fc0, 0x7f80, 0x7fc0], &[0xffc1, 0x7fc0, 0x7fc0]);
        assert_eq!((b.mismatches, b.first), (1, Some((1, 0x7f80, 0x7fc0))));
        let short = diff_bits_bf16(&[1, 2], &[1, 2, 3]);
        assert_eq!((short.compared, short.mismatches), (2, 1));
        assert_eq!(bitwise_check("x", short, 3).status, Status::Fail);
        let same = diff_bits_bf16(&[1, 2, 3], &[1, 2, 3]);
        assert_eq!(bitwise_check("x", same, 3).status, Status::Pass);
    }

    #[test]
    fn tolerance_fails_on_non_finite_and_at_the_bound() {
        let r = tolerance_vs_f64(&[1.0, 2.0], &[1.0, 2.5]).unwrap();
        assert_eq!((r.max_abs_err, r.argmax), (0.5, 1));
        assert_eq!(tolerance_check("t", r, 0.5, "s").status, Status::Fail);
        assert_eq!(tolerance_check("t", r, 0.51, "s").status, Status::Pass);
        let nan = tolerance_vs_f64(&[f32::NAN], &[0.0]).unwrap();
        assert_eq!(tolerance_check("t", nan, 1.0, "s").status, Status::Fail);
        assert!(tolerance_vs_f64(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn panic_payloads_become_text() {
        let caught = std::panic::catch_unwind(|| panic!("Missing symbol cublasGemmEx"));
        let payload = caught.unwrap_err();
        assert_eq!(panic_text(payload.as_ref()), "Missing symbol cublasGemmEx");
    }
}
