//! The tanh-approximate GELU, torch's `F.gelu(x, approximate="tanh")`
//! (`gelu_pytorch_tanh`), and its derivative.
//!
//! Like [`crate::gelu_erf_exact`] this is plain `f64` arithmetic over
//! [`crate::exp_exact`]'s exponential: no fused multiply-add and no libm call
//! (libm `tanh` differs between platforms), so the bits are the same
//! everywhere and under every thread count.
//!
//! With `u = 2 * sqrt(2 / pi) * (x + 0.044715 * x^3)` and `s = sigmoid(u) =
//! 1 / (1 + e^-u)`, the identity `1 + tanh(u / 2) = 2 s` turns torch's
//! `0.5 * x * (1 + tanh(u / 2))` into `x * s`, which has no cancellation in
//! the negative tail (`1 + tanh` of a large negative argument is a difference
//! of two numbers near 1). The derivative is `s + x * s * (1 - s) * u'` with
//! `u' = 2 * sqrt(2 / pi) * (1 + 3 * 0.044715 * x^2)`, and `s * (1 - s) =
//! e^-|u| / (1 + e^-|u|)^2` for either sign of `u`.

use crate::erf_exact::exp_neg;

/// `sqrt(2 / pi)`, torch's `kBeta`.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// Torch's `kKappa`, the cubic coefficient.
const CUBIC: f64 = 0.044_715;

/// `(s, s * (1 - s))` for `s = sigmoid(u)`, from `e^-|u|` alone so neither
/// the large positive nor the large negative argument loses its tail. An
/// `|u|` past `2 * 104` has `e^-|u|` below `1e-90`, which `exp_neg` takes as
/// 0: `s` is then exactly 0 or 1. `u` must not be NaN.
fn sigmoid_pair(u: f64) -> (f64, f64) {
    let e = exp_neg(u.abs());
    let denom = 1.0 + e;
    let s = if u >= 0.0 { 1.0 / denom } else { e / denom };
    (s, e / (denom * denom))
}

/// The argument `u = 2 * sqrt(2 / pi) * (x + 0.044715 * x^3)` of the sigmoid.
fn sigmoid_arg(x: f64) -> f64 {
    2.0 * SQRT_2_OVER_PI * (x + CUBIC * x * x * x)
}

/// The tanh-approximate GELU `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x +
/// 0.044715 * x^3)))` formed in `f64` as `x * sigmoid(u)` and rounded once to
/// `f32`, the same bits on every platform. The negative tail keeps its
/// relative accuracy and underflows to `-0.0` as torch's does (from about
/// `x = -10.8`, where `|gelu|` is below half the smallest `f32` subnormal);
/// a large positive `x` is exactly `x`. NaN stays NaN and `-inf`
/// is NaN (`-inf * 0`), as in torch.
pub fn gelu_tanh_exact(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let xd = f64::from(x);
    let (s, _) = sigmoid_pair(sigmoid_arg(xd));
    (xd * s) as f32
}

/// `dy * d/dx gelu_tanh(x)`, formed in `f64` as `dy * (s + x * s * (1 - s) *
/// u')` and rounded once to `f32`, the same bits on every platform. Far
/// positive `x` gives exactly `dy` and far negative `x` a signed zero. NaN
/// stays NaN and an infinite `x` is NaN (`inf * 0`), as in torch. A product
/// that overflows `f32` is `inf`, for the caller to refuse.
pub fn gelu_tanh_backward_exact(x: f32, dy: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let xd = f64::from(x);
    let (s, slope) = sigmoid_pair(sigmoid_arg(xd));
    let du = 2.0 * SQRT_2_OVER_PI * (1.0 + 3.0 * CUBIC * xd * xd);
    (f64::from(dy) * (s + xd * slope * du)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torch's formula in `f64` with libm `tanh`, accurate (`1e-16` absolute
    /// in `1 + tanh`) wherever `x` is not far into a tail.
    fn gelu_f64(x: f64) -> f64 {
        0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + CUBIC * x * x * x)).tanh())
    }

    /// The derivative written as torch's backward writes it.
    fn gelu_slope_f64(x: f64) -> f64 {
        let t = (SQRT_2_OVER_PI * (x + CUBIC * x * x * x)).tanh();
        0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * CUBIC * x * x)
    }

    fn grid() -> Vec<f64> {
        let mut v = Vec::new();
        let mut x = -6.0f64;
        while x <= 6.0 {
            v.push(x);
            x += 0.0731;
        }
        v
    }

    #[test]
    fn forward_matches_torch_s_formula_to_f32() {
        for x in grid() {
            let xf = x as f32;
            let want = gelu_f64(f64::from(xf));
            let got = f64::from(gelu_tanh_exact(xf));
            // One f32 rounding of the value; the reference's own 1e-16.
            let tol = 1.2e-7 * want.abs() + 1e-14;
            assert!((got - want).abs() <= tol, "x = {x}: {got} vs {want}");
        }
    }

    #[test]
    fn forward_has_its_closed_form_values() {
        assert_eq!(gelu_tanh_exact(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(gelu_tanh_exact(-0.0).to_bits(), (-0.0f32).to_bits());
        // gelu_tanh(1) = 0.8411919906082768 (the erf form is 0.8413447...).
        let got = f64::from(gelu_tanh_exact(1.0));
        assert!((got - 0.841_191_990_608_276_8).abs() < 1e-7, "{got}");
        // The function is x * sigmoid(u): odd up to the sigmoid's symmetry,
        // gelu(x) - gelu(-x) = x exactly.
        for x in [0.3f32, 1.0, 2.5, 4.0] {
            let sum = f64::from(gelu_tanh_exact(x)) - f64::from(gelu_tanh_exact(-x));
            assert!((sum - f64::from(x)).abs() <= 2e-7 * f64::from(x), "x = {x}");
        }
    }

    #[test]
    fn forward_limits_match_torch() {
        // Far negative: torch's `x * 0.5 * (1 + tanh)` is -0.0.
        for x in [-14.0f32, -20.0, -1e30, f32::MIN] {
            assert_eq!(gelu_tanh_exact(x).to_bits(), (-0.0f32).to_bits(), "x = {x}");
        }
        // Far positive: exactly x.
        for x in [9.0f32, 30.0, 3.0e38, f32::MAX] {
            assert_eq!(gelu_tanh_exact(x), x, "x = {x}");
        }
        assert!(gelu_tanh_exact(f32::NAN).is_nan());
        assert!(gelu_tanh_exact(f32::NEG_INFINITY).is_nan());
        assert_eq!(gelu_tanh_exact(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn forward_keeps_the_negative_tail_where_one_plus_tanh_would_cancel() {
        // x = -8: u = -2 * 0.79788 * (8 + 22.9) is about -49.3, so s is
        // 3.6e-22 and gelu is -2.9e-21. `1 + tanh` in f64 is 0 there.
        let u = sigmoid_arg(-8.0);
        let want = -8.0 * (u.exp() / (1.0 + u.exp()));
        let got = f64::from(gelu_tanh_exact(-8.0));
        assert!(
            want < 0.0 && (got - want).abs() <= 1.2e-7 * want.abs(),
            "{got:e} vs {want:e}"
        );
        // In the f32 subnormal range the value is a rounded nonzero, not a
        // flush to zero: x = -10.5 is about -7.5e-43, a few hundred
        // subnormal steps.
        let u = sigmoid_arg(-10.5);
        let want = -10.5 * (u.exp() / (1.0 + u.exp()));
        let got = f64::from(gelu_tanh_exact(-10.5));
        assert!(got != 0.0, "x = -10.5 flushed to zero (want {want:e})");
        assert!((got - want).abs() <= 0.5 * f64::from(f32::from_bits(1)) + 1e-7 * want.abs());
    }

    #[test]
    fn backward_matches_torch_s_derivative_to_f32() {
        for x in grid() {
            let xf = x as f32;
            let want = gelu_slope_f64(f64::from(xf));
            let got = f64::from(gelu_tanh_backward_exact(xf, 1.0));
            let tol = 1.2e-7 * want.abs() + 1e-13;
            assert!((got - want).abs() <= tol, "x = {x}: {got} vs {want}");
        }
    }

    #[test]
    fn backward_matches_a_central_difference_of_the_forward() {
        let h = 1e-5;
        for x in grid() {
            let want = (gelu_f64(x + h) - gelu_f64(x - h)) / (2.0 * h);
            let got = f64::from(gelu_tanh_backward_exact(x as f32, 1.0));
            // The quotient's own error is about 1e-10; f32 rounds the result.
            assert!(
                (got - want).abs() <= 2e-7 * want.abs() + 1e-8,
                "x = {x}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn backward_has_its_closed_form_values_and_scales_with_the_gradient() {
        // gelu_tanh'(0) = 1/2.
        assert_eq!(gelu_tanh_backward_exact(0.0, 1.0), 0.5);
        assert_eq!(gelu_tanh_backward_exact(-0.0, 1.0), 0.5);
        assert_eq!(
            gelu_tanh_backward_exact(0.7, -3.0),
            -3.0 * gelu_tanh_backward_exact(0.7, 1.0)
        );
        assert_eq!(gelu_tanh_backward_exact(0.7, 0.0), 0.0);
    }

    #[test]
    fn backward_saturates_and_passes_nan() {
        for x in [9.0f32, 30.0, 1e30, f32::MAX] {
            assert_eq!(gelu_tanh_backward_exact(x, 0.75), 0.75, "x = {x}");
        }
        for x in [-20.0f32, -30.0, -1e30, f32::MIN] {
            assert_eq!(gelu_tanh_backward_exact(x, 0.75), 0.0, "x = {x}");
        }
        assert!(gelu_tanh_backward_exact(f32::NAN, 1.0).is_nan());
        assert!(gelu_tanh_backward_exact(1.0, f32::NAN).is_nan());
        assert!(gelu_tanh_backward_exact(f32::INFINITY, 1.0).is_nan());
        assert!(gelu_tanh_backward_exact(f32::NEG_INFINITY, 1.0).is_nan());
        // An overflowing product is inf, for the caller to refuse.
        assert_eq!(gelu_tanh_backward_exact(3.0, f32::MAX), f32::INFINITY);
    }

    #[test]
    fn gelu_is_the_tanh_form_not_the_erf_form() {
        let tanh = f64::from(gelu_tanh_exact(1.0));
        let erf = f64::from(crate::gelu_erf_exact(1.0));
        assert!((tanh - erf).abs() > 1e-4);
    }
}
