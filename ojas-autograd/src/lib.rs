//! Tape over a [`Backend`](ojas_core::Backend), plus an f64 central-difference gradcheck.
//!
//! On [`BackendId::Cpu`](ojas_core::BackendId::Cpu) the tape keeps the tensors it
//! is given. On any other backend, inputs and the backward seed are uploaded
//! and reshape is a view, so backward does not read device memory back.
//! Gradcheck compares the f32 gradients to a central difference of an f64 forward.

#![forbid(unsafe_code)]

mod tape;
mod tiny;

pub use tape::{Tape, Var};
pub use tiny::{reduce_micrograds, TinyTrain, TokenBatch};

use ojas_core::OjasError;

/// Central difference of a scalar f64 function. `h` is the step on each coordinate.
pub fn central_diff<F>(x: &[f64], h: f64, mut f: F) -> Result<Vec<f64>, OjasError>
where
    F: FnMut(&[f64]) -> Result<f64, OjasError>,
{
    if !(h.is_finite() && h > 0.0) {
        return Err(OjasError::OutOfRange {
            op: "central_diff",
            detail: "step h must be finite and positive".to_string(),
        });
    }
    let mut point = x.to_vec();
    let mut grad = vec![0.0f64; x.len()];
    for i in 0..x.len() {
        let original = point[i];
        point[i] = original + h;
        let pos = f(&point)?;
        point[i] = original - h;
        let neg = f(&point)?;
        point[i] = original;
        let slope = (pos - neg) / (2.0 * h);
        if !slope.is_finite() {
            return Err(OjasError::NonFinite { op: "central_diff" });
        }
        grad[i] = slope;
    }
    Ok(grad)
}

/// Compare an f32 analytic gradient to an f64 finite difference.
///
/// A non-finite value on either side never matches, and the tolerances must
/// be finite and non-negative: `diff > tol` is false for NaN, so a NaN that
/// reached the comparison would otherwise pass.
pub fn gradients_match(
    analytical: &[f32],
    numeric: &[f64],
    atol: f64,
    rtol: f64,
) -> Result<(), String> {
    if analytical.len() != numeric.len() {
        return Err(format!(
            "grad length {} != {}",
            analytical.len(),
            numeric.len()
        ));
    }
    for (name, value) in [("atol", atol), ("rtol", rtol)] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(format!("{name} {value} must be finite and non-negative"));
        }
    }
    let mut worst = 0.0f64;
    for (index, (got, expect)) in analytical.iter().zip(numeric.iter()).enumerate() {
        let got = f64::from(*got);
        if !got.is_finite() || !expect.is_finite() {
            return Err(format!(
                "index {index}: analytic {got} numeric {expect} is not finite"
            ));
        }
        let diff = (got - expect).abs();
        worst = worst.max(diff);
        let tol = atol + rtol * got.abs().max(expect.abs());
        if diff > tol {
            return Err(format!(
                "index {index}: analytic {got} numeric {expect} diff {diff} tol {tol} worst {worst}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::gradients_match;

    #[test]
    fn a_non_finite_gradient_never_matches() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = gradients_match(&[bad, 1.0], &[0.5, 1.0], 1e-3, 1e-3);
            assert!(err.is_err(), "analytic {bad} matched");
            let err = gradients_match(&[1.0, bad], &[1.0, 0.5], 1e-3, 1e-3);
            assert!(err.is_err(), "analytic {bad} at the last index matched");
        }
        for bad in [f64::NAN, f64::INFINITY] {
            let err = gradients_match(&[0.5, 1.0], &[bad, 1.0], 1e-3, 1e-3);
            assert!(err.is_err(), "numeric {bad} matched");
        }
    }

    #[test]
    fn a_tolerance_that_cannot_be_compared_is_refused() {
        for (atol, rtol) in [
            (f64::NAN, 1e-3),
            (1e-3, f64::NAN),
            (-1e-3, 1e-3),
            (1e-3, -1.0),
        ] {
            let err = gradients_match(&[0.5], &[9.0], atol, rtol);
            assert!(err.is_err(), "atol {atol} rtol {rtol} accepted");
        }
        assert!(gradients_match(&[0.5], &[0.5], f64::INFINITY, 0.0).is_err());
    }

    #[test]
    fn finite_gradients_compare_by_tolerance() {
        assert!(gradients_match(&[0.5, 1.0], &[0.5, 1.0], 0.0, 0.0).is_ok());
        assert!(gradients_match(&[0.5, 1.0], &[0.5005, 1.0], 1e-3, 0.0).is_ok());
        assert!(gradients_match(&[0.5, 1.0], &[0.6, 1.0], 1e-3, 1e-3).is_err());
        assert!(gradients_match(&[0.5], &[0.5, 1.0], 1.0, 1.0).is_err());
    }
}
