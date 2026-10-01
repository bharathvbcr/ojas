//! Tape over the CPU reference, plus an f64 central-difference gradcheck.
//!
//! Backward calls the `ojas-cpu` implementations. Gradcheck compares those
//! f32 gradients to a central difference of an f64 forward.

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
    let mut worst = 0.0f64;
    for (index, (got, expect)) in analytical.iter().zip(numeric.iter()).enumerate() {
        let got = f64::from(*got);
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
