//! `erf` and the exact GELU built on it, the error function behind
//! `torch.nn.functional.gelu(x, approximate="none")`.
//!
//! Like [`crate::exp_exact`] this is plain `f64` arithmetic: no fused
//! multiply-add and no libm call (libm `erff` differs between platforms), so
//! the bits are the same everywhere and under every thread count.
//!
//! `erf(z) = 2/sqrt(pi) * e^(-z^2) * sum_n 2^n z^(2n+1) / (2n+1)!!` for
//! `z >= 0`. Every term is positive, so the sum does not cancel, and the
//! only error is the `f64` rounding of about a hundred terms plus the
//! exponential's (`2^-50` relative). From [`SATURATE`] up the `f64` value of
//! `erf` is `1.0`.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI};

use crate::exp_exact::exp_in_range_f64;

/// `erf(6) = 1 - 2.2e-17`, which is `1.0` in `f64`.
const SATURATE: f64 = 6.0;
/// The series stops when a term is below this fraction of the sum.
const STOP: f64 = 1e-17;
/// Terms the series may add. `z = 6` needs about 110.
const MAX_TERMS: u32 = 400;

/// `erf(z)` in `f64`, within `1e-14` relative of the true value (the
/// exponential's `2^-50` and the rounding of the sum).
/// NaN stays NaN, `+-inf` and `|z| >= 6` are `+-1.0`, and the sign of zero
/// is kept.
pub fn erf_f64(z: f64) -> f64 {
    if z.is_nan() {
        return z;
    }
    let a = z.abs();
    if a >= SATURATE {
        return 1.0f64.copysign(z);
    }
    let two_a2 = 2.0 * a * a;
    let mut term = a;
    let mut sum = a;
    let mut denom = 3.0;
    for _ in 0..MAX_TERMS {
        if term <= sum * STOP {
            break;
        }
        term = term * two_a2 / denom;
        sum += term;
        denom += 2.0;
    }
    // The rounding of the sum can leave a value an ulp above 1 near the cut.
    let value = (FRAC_2_SQRT_PI * exp_in_range_f64(-(a * a)) * sum).min(1.0);
    value.copysign(z)
}

/// `erf(x)` rounded once to `f32`, the same bits on every platform.
pub fn erf_exact(x: f32) -> f32 {
    erf_f64(f64::from(x)) as f32
}

/// The exact GELU `x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, formed in
/// `f64` and rounded once to `f32`, the same bits on every platform. `1 + erf`
/// is rounded in `f64`, not `f32` as torch does, so a negative `x`'s tail
/// (`|gelu| < 1e-6`, below `x = -4.9`) is good to about `1e-16` absolute
/// instead of `6e-8`; from `x = -8.5` on, `erf(x / sqrt(2))` is `-1.0` and the
/// value is `-0.0`, as in torch. NaN stays NaN, and `-inf` is NaN
/// (`-inf * 0`), as in torch.
pub fn gelu_erf_exact(x: f32) -> f32 {
    let xd = f64::from(x);
    let erf = erf_f64(xd * FRAC_1_SQRT_2);
    (0.5 * xd * (1.0 + erf)) as f32
}

/// `1 / sqrt(2 pi)`, the standard normal density at 0.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// [`exp_in_range_f64`] takes arguments down to `-104`; `e^-208` is `5e-91`.
const EXP_RANGE: f64 = 104.0;

/// `e^-h` for `h >= 0` in `f64`, past `exp_in_range_f64`'s range as the
/// product of two of its values. An `h` above `2 * EXP_RANGE` is `0.0`: there
/// `x * phi(x)` is below `4e-90`, so even a gradient of `3.4e38` scales it to
/// below `1e-50`, which is under the smallest `f32`.
fn exp_neg(h: f64) -> f64 {
    if h <= EXP_RANGE {
        exp_in_range_f64(-h)
    } else if h <= 2.0 * EXP_RANGE {
        exp_in_range_f64(-EXP_RANGE) * exp_in_range_f64(EXP_RANGE - h)
    } else {
        0.0
    }
}

/// `dy * d/dx gelu(x)` for the exact GELU: `dy * (Phi(x) + x * phi(x))`,
/// `Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))` and `phi(x) = e^(-x^2 / 2) /
/// sqrt(2 pi)`, formed in `f64` and rounded once to `f32`, the same bits on
/// every platform. The density's exponential is evaluated where torch's `f32`
/// one has underflowed to 0 (`|x|` above 14.4), so a gradient near the
/// `f32` maximum still sees the term it scales. A negative `x`'s tail
/// carries the `1e-14` absolute error of `1 + erf`, which an `f32` gradient
/// above `1e-7` does not see. NaN stays NaN, and an infinite `x` is NaN
/// (`inf * 0`), as in torch.
pub fn gelu_erf_backward_exact(x: f32, dy: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let xd = f64::from(x);
    let cdf = 0.5 * (1.0 + erf_f64(xd * FRAC_1_SQRT_2));
    let pdf = FRAC_1_SQRT_2PI * exp_neg(0.5 * xd * xd);
    (f64::from(dy) * (cdf + xd * pdf)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(x, math.erf(x))` from Python's libm, within one `f64` ulp.
    const ERF: [(f64, f64); 24] = [
        (0.0, 0.0),
        (1e-08, 1.1283791670955126e-08),
        (0.01, 0.011283415555849616),
        (0.1, 0.1124629160182849),
        (0.25, 0.2763263901682369),
        (0.5, 0.5204998778130465),
        (0.75, 0.7111556336535151),
        (1.0, 0.8427007929497149),
        (1.25, 0.9229001282564582),
        (1.5, 0.9661051464753108),
        (1.75, 0.9866716712191824),
        (2.0, 0.9953222650189527),
        (2.5, 0.999593047982555),
        (3.0, 0.9999779095030014),
        (3.5, 0.9999992569016276),
        (4.0, 0.9999999845827421),
        (4.5, 0.9999999998033839),
        (5.0, 0.9999999999984626),
        (5.5, 0.9999999999999927),
        (5.9, 0.9999999999999999),
        (-0.3, -0.3286267594591274),
        (-1.0, -0.8427007929497149),
        (-2.2, -0.9981371537020182),
        (-4.7, -0.9999999999700474),
    ];

    /// `(x, 0.5 * x * (1 + math.erf(x / sqrt(2))))`.
    const GELU: [(f64, f64); 20] = [
        (-8.0, -4.884981308350689e-15),
        (-6.0, -5.919525869479969e-09),
        (-5.0, -1.4332578593401202e-06),
        (-4.0, -0.00012668496733247991),
        (-3.0, -0.00404969409489031),
        (-2.0, -0.04550026389635842),
        (-1.0, -0.15865525393145707),
        (-0.5, -0.15426876936299344),
        (-0.1, -0.0460172162722971),
        (-0.001, -0.000499601057786089),
        (0.0, 0.0),
        (0.001, 0.000500398942213911),
        (0.1, 0.053982783727702904),
        (0.5, 0.34573123063700656),
        (1.0, 0.8413447460685429),
        (2.0, 1.9544997361036416),
        (3.0, 2.99595030590511),
        (4.0, 3.9998733150326675),
        (6.0, 5.999999994080474),
        (10.0, 10.0),
    ];

    #[test]
    fn erf_matches_libm_to_a_few_ulps() {
        for (x, want) in ERF {
            let got = erf_f64(x);
            let tol = 1e-14 * want.abs().max(1e-300);
            assert!((got - want).abs() <= tol, "erf({x}) = {got}, libm {want}");
        }
    }

    #[test]
    fn erf_is_odd_and_keeps_the_sign_of_zero() {
        for (x, _) in ERF {
            assert_eq!(erf_f64(-x).to_bits(), (-erf_f64(x)).to_bits(), "x = {x}");
        }
        assert_eq!(erf_f64(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(erf_f64(-0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn erf_saturates_and_passes_nan() {
        for z in [6.0, 7.5, 1e30, f64::INFINITY] {
            assert_eq!(erf_f64(z), 1.0);
            assert_eq!(erf_f64(-z), -1.0);
        }
        assert!(erf_f64(f64::NAN).is_nan());
        // Just under the cut the series is already within an ulp of 1.
        assert!((erf_f64(5.999) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn erf_is_monotone_and_within_one() {
        let mut prev = -1.0;
        let mut z = -6.0;
        while z <= 6.0 {
            let v = erf_f64(z);
            assert!(v >= prev - 1e-14 && (-1.0..=1.0).contains(&v), "z = {z}");
            prev = v;
            z += 0.0173;
        }
    }

    #[test]
    fn gelu_matches_libm_erf_form_to_f32() {
        for (x, want) in GELU {
            let got = f64::from(gelu_erf_exact(x as f32));
            // One `f32` rounding, plus the `f64` rounding of `1 + erf`.
            let tol = 2e-7 * want.abs() + 1e-15;
            assert!((got - want).abs() <= tol, "gelu({x}) = {got}, libm {want}");
        }
    }

    #[test]
    fn gelu_limits_match_torch() {
        assert_eq!(gelu_erf_exact(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(gelu_erf_exact(-0.0).to_bits(), (-0.0f32).to_bits());
        // Far negative: torch's `x * 0.5 * (1 + erf)` is `-0.0`.
        assert_eq!(gelu_erf_exact(-9.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(gelu_erf_exact(-1e30).to_bits(), (-0.0f32).to_bits());
        // Far positive: exactly `x`.
        assert_eq!(gelu_erf_exact(9.0), 9.0);
        assert_eq!(gelu_erf_exact(3.0e38), 3.0e38);
        assert!(gelu_erf_exact(f32::NAN).is_nan());
        assert!(gelu_erf_exact(f32::NEG_INFINITY).is_nan());
    }

    /// `gelu` in `f64`, the reference the derivative is differenced from.
    fn gelu_f64(x: f64) -> f64 {
        0.5 * x * (1.0 + erf_f64(x * FRAC_1_SQRT_2))
    }

    #[test]
    fn gelu_derivative_matches_a_central_difference_of_gelu() {
        let h = 1e-5;
        let mut x = -7.0f64;
        while x <= 7.0 {
            let want = (gelu_f64(x + h) - gelu_f64(x - h)) / (2.0 * h);
            let got = f64::from(gelu_erf_backward_exact(x as f32, 1.0));
            // The difference quotient's own error is about 1e-9 in the tails.
            assert!(
                (got - want).abs() <= 2e-7 * want.abs() + 1e-8,
                "x = {x}: {got} vs {want}"
            );
            x += 0.173;
        }
    }

    #[test]
    fn gelu_derivative_has_its_closed_form_values() {
        // gelu'(0) = 1/2; gelu'(+-1) = Phi(1) +- phi(1).
        assert_eq!(gelu_erf_backward_exact(0.0, 1.0), 0.5);
        assert_eq!(gelu_erf_backward_exact(-0.0, 1.0), 0.5);
        let (cdf, pdf) = (0.841_344_746_068_542_9, 0.241_970_724_519_143_37);
        let got = f64::from(gelu_erf_backward_exact(1.0, 1.0));
        assert!((got - (cdf + pdf)).abs() < 1e-7, "{got}");
        let got = f64::from(gelu_erf_backward_exact(-1.0, 1.0));
        assert!((got - (1.0 - cdf - pdf)).abs() < 1e-7, "{got}");
        // The gradient scales the derivative.
        assert_eq!(
            gelu_erf_backward_exact(0.0, -3.0),
            -3.0 * gelu_erf_backward_exact(0.0, 1.0)
        );
    }

    #[test]
    fn gelu_derivative_saturates_and_keeps_the_underflowed_density_term() {
        // Far positive: exactly the gradient. Far negative: `-x * phi(x)` is
        // `-9e-18` at -9 and below the smallest `f32` from -15 on, so zero.
        for x in [9.0f32, 15.0, 30.0, 1e30, f32::MAX] {
            assert_eq!(gelu_erf_backward_exact(x, 0.75), 0.75, "x = {x}");
        }
        for x in [15.0f32, 30.0, 1e30, f32::MAX] {
            assert_eq!(gelu_erf_backward_exact(-x, 0.75), 0.0, "x = -{x}");
        }
        let nine = f64::from(gelu_erf_backward_exact(-9.0, 1.0));
        let want = -9.0 * 0.398_942_280_401_432_7 * (-40.5f64).exp();
        assert!(
            (nine - want).abs() <= 1e-6 * want.abs(),
            "{nine:e} vs {want:e}"
        );
        // Past |x| = 14.4 torch's f32 exponential is 0; the term survives a
        // gradient near the f32 maximum: -x * phi(x) * dy.
        for (x, dy) in [(15.0f64, 1e38f32), (18.0, 3.0e38), (18.5, 3.4e38)] {
            let want = -x * 0.398_942_280_401_432_7 * (-0.5 * x * x).exp() * f64::from(dy);
            let got = f64::from(gelu_erf_backward_exact(-x as f32, dy));
            assert!(
                (got - want).abs() <= 1e-6 * want.abs(),
                "x = -{x}: {got} vs {want}"
            );
        }
        // Where the term is below an f32 it is a signed zero, not garbage.
        assert_eq!(gelu_erf_backward_exact(-25.0, 3.4e38), 0.0);
        assert_eq!(gelu_erf_backward_exact(-1e30, f32::MAX), 0.0);
    }

    #[test]
    fn gelu_derivative_passes_nan_and_refuses_nothing_else() {
        assert!(gelu_erf_backward_exact(f32::NAN, 1.0).is_nan());
        assert!(gelu_erf_backward_exact(1.0, f32::NAN).is_nan());
        assert!(gelu_erf_backward_exact(f32::INFINITY, 1.0).is_nan());
        assert!(gelu_erf_backward_exact(f32::NEG_INFINITY, 1.0).is_nan());
        // An overflowing product is inf, for the caller to refuse.
        assert_eq!(gelu_erf_backward_exact(1.5, f32::MAX), f32::INFINITY);
    }

    #[test]
    fn gelu_is_the_erf_form_not_the_tanh_form() {
        // gelu_erf(1) = 0.8413447460685429, gelu_tanh(1) = 0.8411919906082768.
        let got = f64::from(gelu_erf_exact(1.0));
        assert!((got - 0.841_344_746_068_542_9).abs() < 1e-7, "{got}");
        assert!((got - 0.841_191_990_608_276_8).abs() > 1e-4);
    }
}
