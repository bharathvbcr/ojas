//! Compare any two [`ojas_core::Backend`] values. The tolerance is per op.
//! Bit-exact agreement is not assumed across GPU backends.

use ojas_core::{Backend, OjasError, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityOp {
    Linear,
}

pub fn max_abs(got: &[f32], expect: &[f32]) -> Result<f64, OjasError> {
    if got.len() != expect.len() {
        return Err(OjasError::Shape {
            op: "parity",
            detail: format!("len {} != {}", got.len(), expect.len()),
        });
    }
    Ok(got
        .iter()
        .zip(expect)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0_f64, f64::max))
}

/// SplitMix64 values in `[-scale, scale)`. Local so the harness adds no crate.
pub fn splitmix_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let mantissa = (z >> 40) as f32 / (1u64 << 24) as f32;
            scale * (2.0 * mantissa - 1.0)
        })
        .collect()
}

pub fn linear_close(
    reference: &impl Backend,
    other: &impl Backend,
    rows: usize,
    kin: usize,
    nout: usize,
    seed: u64,
    tol: f64,
) -> Result<(), OjasError> {
    let x = splitmix_f32(seed, rows * kin, 0.5);
    let w = splitmix_f32(seed ^ 0xA5, nout * kin, 0.5);
    let xt = Tensor::from_f32(&x, &[rows, kin], reference.budget())?;
    let wt = Tensor::from_f32(&w, &[nout, kin], reference.budget())?;
    let xo = Tensor::from_f32(&x, &[rows, kin], other.budget())?;
    let wo = Tensor::from_f32(&w, &[nout, kin], other.budget())?;
    let a = reference.linear_forward(&xt, &wt)?;
    let b = other.linear_forward(&xo, &wo)?;
    let err = max_abs(&b.to_f32_vec()?, &a.to_f32_vec()?)?;
    if err > tol {
        return Err(OjasError::Backend {
            id: other.id(),
            detail: format!("linear parity error {err} exceeds {tol}"),
        });
    }
    let _ = ParityOp::Linear;
    Ok(())
}
