//! Compare any two [`ojas_core::Backend`] values. The tolerance is per op.
//! Bit-exact agreement is not assumed across GPU backends.
//!
//! Inputs are built on the host and moved with each backend's own
//! [`Backend::upload`]; results come back with [`Backend::download`]. A
//! device-resident backend therefore runs its kernels, and a deferred
//! device fault surfaces as that download's error.

use ojas_core::{Backend, OjasError, Tensor};

use crate::SplitMix64;

/// Largest `|got - expect|`. A non-finite value on either side is
/// `f64::INFINITY`, so no tolerance accepts it: a NaN difference must never
/// read as agreement.
pub fn max_abs(got: &[f32], expect: &[f32]) -> Result<f64, OjasError> {
    if got.len() != expect.len() {
        return Err(OjasError::Shape {
            op: "parity",
            detail: format!("len {} != {}", got.len(), expect.len()),
        });
    }
    let mut worst = 0.0f64;
    for (a, b) in got.iter().zip(expect) {
        if !(a.is_finite() && b.is_finite()) {
            return Ok(f64::INFINITY);
        }
        worst = worst.max((f64::from(*a) - f64::from(*b)).abs());
    }
    Ok(worst)
}

/// `n` [`SplitMix64`] values in `[-scale, scale)`, from `seed`.
pub fn splitmix_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| scale * rng.next_signed_f32()).collect()
}

/// `linear_forward` on `other` against `reference` for `[rows, kin] x
/// [nout, kin]^T`. The error is the absolute difference over the whole
/// output, so `tol` is in the output's units.
pub fn linear_close(
    reference: &impl Backend,
    other: &impl Backend,
    rows: usize,
    kin: usize,
    nout: usize,
    seed: u64,
    tol: f64,
) -> Result<(), OjasError> {
    let product = |a: usize, b: usize| {
        a.checked_mul(b).ok_or_else(|| OjasError::OutOfRange {
            op: "linear_close",
            detail: format!("{a} * {b} overflows"),
        })
    };
    let x = splitmix_f32(seed, product(rows, kin)?, 0.5);
    let w = splitmix_f32(seed ^ 0xA5, product(nout, kin)?, 0.5);
    let run = |backend: &dyn Backend| -> Result<Vec<f32>, OjasError> {
        let xt = backend.upload(&Tensor::from_f32(&x, &[rows, kin], backend.budget())?)?;
        let wt = backend.upload(&Tensor::from_f32(&w, &[nout, kin], backend.budget())?)?;
        let y = backend.linear_forward(&xt, &wt)?;
        backend.download(&y)?.to_f32_vec()
    };
    let want = run(reference)?;
    let got = run(other)?;
    let err = max_abs(&got, &want)?;
    // A NaN tolerance accepts nothing.
    if !matches!(
        err.partial_cmp(&tol),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    ) {
        return Err(OjasError::Backend {
            id: other.id(),
            detail: format!("linear parity error {err} exceeds {tol}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bits of the inline generator `splitmix_f32` carried before it moved
    /// onto [`SplitMix64`].
    #[test]
    fn splitmix_f32_keeps_its_bits() {
        let bits = |v: Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(splitmix_f32(7, 4, 0.5)),
            [0xbde1_a0f8, 0xbef7_6788, 0x3ecd_3080, 0x3da9_d758]
        );
        assert_eq!(
            bits(splitmix_f32(0, 3, 1.0)),
            [0x3f44_4150, 0xbe0c_3b10, 0xbf72_7746]
        );
        assert!(splitmix_f32(9, 0, 1.0).is_empty());
    }

    #[test]
    fn non_finite_values_never_pass_parity() {
        // Pre-fix, `fold(0.0, f64::max)` dropped every NaN difference, so an
        // all-NaN kernel output measured 0.0 and passed any tolerance.
        let want = [1.0f32, -2.0, 0.5];
        for got in [
            [f32::NAN; 3],
            [1.0, f32::NAN, 0.5],
            [1.0, -2.0, f32::INFINITY],
            [f32::NEG_INFINITY, -2.0, 0.5],
        ] {
            let err = max_abs(&got, &want).unwrap();
            assert_eq!(err, f64::INFINITY, "{got:?} measured {err}");
            let err = max_abs(&want, &got).unwrap();
            assert_eq!(err, f64::INFINITY, "reference {got:?} measured {err}");
        }
        assert_eq!(max_abs(&want, &want).unwrap(), 0.0);
        assert_eq!(max_abs(&[1.0, 2.0], &[1.5, 2.0]).unwrap(), 0.5);
        assert!(max_abs(&[1.0], &want).is_err());
    }
}
