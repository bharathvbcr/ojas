//! Branch-free exponentials for [`ojas_core::Numerics::Fast`]: `2^x` for the
//! attention softmax ([`exp2_affine`]), `e^x` for SiLU ([`exp`]) and
//! cross-entropy ([`exp_sub_sum`], [`exp_sub_store`]). One polynomial and one
//! scaling step serve all of them; `f32::exp` is a libm call per element and
//! does not vectorize.
//!
//! `2^x`: the range reduction is exact, `x = n + r` with `n` rounded to
//! nearest and `|r| <= 1/2`. `2^r` is the degree-7 Taylor polynomial of
//! `e^(r·ln 2)` (truncation below 2^-27 relative) in Horner form, and `2^n`
//! is applied as two factors `2^(n>>1) · 2^(n - (n>>1))`, each a normal
//! number, so results in the subnormal range are rounded once and overflow
//! goes to `+inf`.
//!
//! `e^x`: `n = round(x·log2 e)` and `r = x - n·ln 2` with `ln 2` split in two
//! (Cody-Waite), both steps exact for the clamped range, then `2^(r·log2 e)`
//! on the same polynomial (`|r·log2 e| <= 1/2`, the product rounded once) and
//! the same scaling.
//!
//! Error against `f64` rounded to `f32`, over every finite `f32` input (the
//! full-range sweeps below): at most [`MAX_ULP`] units in the last place
//! where the result is a normal `f32`, at most one subnormal step (`2^-149`)
//! below that, `+inf` where `f64` overflows `f32` and `+0.0` where it
//! underflows to zero. NaN stays NaN.

use crate::gemm::fma;

/// `1.5 · 2^23`: adding and subtracting it rounds to the nearest integer.
const ROUND: f32 = 12_582_912.0;
/// The `2^x` clamp keeps `n` in `[-151, 129]`: `2^-151` rounds to 0 and
/// `2^129` overflows.
const LO: f32 = -151.0;
const HI: f32 = 129.0;
/// The `e^x` clamp, the same `n` range: `-151·ln 2 > -104.7` and
/// `129·ln 2 < 89.5`, so `n` rounds into `[-151, 129]`.
const E_LO: f32 = -104.7;
const E_HI: f32 = 89.5;
const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// `ln 2` split so `n · LN2_HI` is exact for `|n| <= 2^15`.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;

/// Bound asserted by the full-range tests, in units in the last place.
#[cfg(test)]
const MAX_ULP: u32 = 2;

/// `(ln 2)^k / k!` for `k = 7, 6, .., 0`, rounded to `f32`: Horner order.
const COEFFS: [f32; 8] = [
    1.525_273_4e-5,
    1.540_353e-4,
    1.333_355_8e-3,
    9.618_129e-3,
    5.550_411e-2,
    2.402_265e-1,
    6.931_472e-1,
    1.0,
];

/// Lane sums a slice kernel keeps: element `i` of a whole chunk of
/// `LANES` adds into lane `i % LANES`. Each lane is evaluated in full
/// before the next, which the compiler turns into `LANES / 4` vector chains
/// (measured 23% faster single-threaded than evaluating every lane step by
/// step, which it vectorized across chunks with lane gathers instead).
const LANES: usize = 16;

/// `xs[i] = 2^(xs[i]·mul + add)`, returning the sum of the results in a
/// fixed order: [`LANES`] lane sums over consecutive elements, combined in
/// lane order, then the tail in index order.
pub(crate) fn exp2_affine(xs: &mut [f32], mul: f32, add: f32) -> f32 {
    let (chunks, rest) = xs.as_chunks_mut::<LANES>();
    let mut acc = [0.0f32; LANES];
    for chunk in chunks {
        for (a, x) in acc.iter_mut().zip(chunk.iter_mut()) {
            *x = exp2(fma(*x, mul, add));
            *a += *x;
        }
    }
    let mut total = fold(&acc);
    for x in rest {
        *x = exp2(fma(*x, mul, add));
        total += *x;
    }
    total
}

/// `sum_i e^(x_i - shift)` over native-endian f32 words, in the fixed order
/// of [`exp2_affine`]. `x_i - shift` is one f32 subtraction, as the scalar
/// path forms it. Nothing is stored.
pub(crate) fn exp_sub_sum(row: &[[u8; 4]], shift: f32) -> f32 {
    let (chunks, rest) = row.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for chunk in chunks {
        for (a, word) in acc.iter_mut().zip(chunk) {
            *a += exp(f32::from_ne_bytes(*word) - shift);
        }
    }
    let mut total = fold(&acc);
    for word in rest {
        total += exp(f32::from_ne_bytes(*word) - shift);
    }
    total
}

/// [`exp_sub_sum`] that also stores each `e^(x_i - shift)` in `out` as
/// native-endian words. `out` is as long as `row`.
pub(crate) fn exp_sub_store(row: &[[u8; 4]], shift: f32, out: &mut [[u8; 4]]) -> f32 {
    let (chunks, rest) = row.as_chunks::<LANES>();
    let (out_chunks, out_rest) = out.as_chunks_mut::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (chunk, dst) in chunks.iter().zip(out_chunks) {
        for ((a, word), slot) in acc.iter_mut().zip(chunk).zip(dst.iter_mut()) {
            let e = exp(f32::from_ne_bytes(*word) - shift);
            *a += e;
            *slot = e.to_ne_bytes();
        }
    }
    let mut total = fold(&acc);
    for (word, slot) in rest.iter().zip(out_rest) {
        let e = exp(f32::from_ne_bytes(*word) - shift);
        total += e;
        *slot = e.to_ne_bytes();
    }
    total
}

/// Lane sums combined in lane order.
#[inline(always)]
fn fold(acc: &[f32; LANES]) -> f32 {
    acc.iter().fold(0.0f32, |s, &v| s + v)
}

/// `2^x`.
#[inline(always)]
pub(crate) fn exp2(x: f32) -> f32 {
    let (t, r) = reduce2(x);
    scale(poly(r), t)
}

/// `e^x`.
#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    let (t, r) = reduce_e(x);
    scale(poly(r), t)
}

#[inline(always)]
fn poly(r: f32) -> f32 {
    let mut p = COEFFS[0];
    for &c in &COEFFS[1..] {
        p = fma(p, r, c);
    }
    p
}

/// Clamp, then `(t, r)` with `t = ROUND + n` exactly (its low mantissa bits
/// hold `n`, the nearest integer) and `r = x - n`, also exact.
#[inline(always)]
fn reduce2(x: f32) -> (f32, f32) {
    // A NaN fails both comparisons and stays NaN through the arithmetic.
    let x = if x < LO { LO } else { x };
    let x = if x > HI { HI } else { x };
    let t = x + ROUND;
    (t, x - (t - ROUND))
}

/// Clamp, then `(t, r)` with `t = ROUND + n`, `n = round(x · log2 e)`, and
/// `r = (x - n · ln 2) · log2 e`, so `e^x = 2^n · 2^r`.
#[inline(always)]
fn reduce_e(x: f32) -> (f32, f32) {
    let x = if x < E_LO { E_LO } else { x };
    let x = if x > E_HI { E_HI } else { x };
    let t = fma(x, LOG2_E, ROUND);
    let n = t - ROUND;
    let r = fma(-n, LN2_HI, x);
    let r = fma(-n, LN2_LO, r);
    (t, r * LOG2_E)
}

/// `p · 2^n` for the `n` held in `t`, as two normal factors.
#[inline(always)]
fn scale(p: f32, t: f32) -> f32 {
    let n = t.to_bits().wrapping_sub(ROUND.to_bits()) as i32;
    let n1 = n >> 1;
    p * pow2(n1) * pow2(n.wrapping_sub(n1))
}

/// `2^n` for `n` in `[-126, 127]`.
#[inline(always)]
fn pow2(n: i32) -> f32 {
    f32::from_bits((n.wrapping_add(127) as u32) << 23)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `got` against `want` (`f64` rounded to `f32`) under the module bound.
    fn check(name: &str, x: f32, got: f32, want: f32, worst: &mut (u32, f32)) {
        if x.is_nan() {
            assert!(got.is_nan(), "{name}(NaN) = {got}");
            return;
        }
        if want == f32::INFINITY || want == f32::MAX {
            // Within half an ulp of f32::MAX or beyond it.
            assert!(
                got == f32::INFINITY || got == f32::MAX,
                "{name}({x:e}) = {got:e}, want {want:e}"
            );
            return;
        }
        assert!(got.is_finite() && got >= 0.0, "{name}({x:e}) = {got:e}");
        if want < f32::MIN_POSITIVE {
            assert!(
                (got - want).abs() <= f32::from_bits(1),
                "{name}({x:e}) = {got:e}, want {want:e} (subnormal)"
            );
            return;
        }
        let u = got.to_bits().abs_diff(want.to_bits());
        if u > worst.0 {
            *worst = (u, x);
        }
        assert!(
            u <= MAX_ULP,
            "{name}({x:e}) = {got:e}, want {want:e}: {u} ulp"
        );
    }

    fn check2(x: f32, worst: &mut (u32, f32)) {
        check("exp2", x, exp2(x), (x as f64).exp2() as f32, worst);
    }

    fn check_e(x: f32, worst: &mut (u32, f32)) {
        check("exp", x, exp(x), (x as f64).exp() as f32, worst);
    }

    /// Every 61st bit pattern of both signs (about 70 million inputs, every
    /// exponent and a spread of mantissas), plus the boundaries.
    #[test]
    fn exp2_matches_f64_over_the_whole_finite_range() {
        let mut worst = (0u32, 0.0f32);
        let mut bits = 0u32;
        while bits < 0x7f80_0000 {
            check2(f32::from_bits(bits), &mut worst);
            check2(-f32::from_bits(bits), &mut worst);
            bits += 61;
        }
        for x in [
            0.0f32,
            -0.0,
            0.5,
            -0.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            LO,
            HI,
            -126.0,
            -126.5,
            -149.0,
            -149.5,
            -150.0,
            -150.5,
            127.0,
            127.5,
            127.999_99,
            128.0,
        ] {
            check2(x, &mut worst);
        }
        assert_eq!(exp2(0.0), 1.0);
        assert_eq!(exp2(-1.0), 0.5);
        assert_eq!(exp2(-149.0), f32::from_bits(1));
        assert_eq!(exp2(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp2(f32::INFINITY), f32::INFINITY);
        assert!(exp2(f32::NAN).is_nan());
        println!("exp2 worst {} ulp at {:e}", worst.0, worst.1);
    }

    /// Every 7th bit pattern of `[-45, 0]`, the softmax domain that carries
    /// the weight (within `2^-45` of the row maximum): about 160 million.
    #[test]
    fn exp2_is_within_bound_densely_in_minus_45_to_0() {
        let mut worst = (0u32, 0.0f32);
        let top = (-45.0f32).to_bits();
        let mut bits = 0x8000_0000u32;
        while bits <= top {
            check2(f32::from_bits(bits), &mut worst);
            bits += 7;
        }
        println!("exp2 worst {} ulp at {:e}", worst.0, worst.1);
    }

    /// `e^x` over every 61st bit pattern of both signs, plus the clamp and
    /// subnormal boundaries.
    #[test]
    fn exp_matches_f64_over_the_whole_finite_range() {
        let mut worst = (0u32, 0.0f32);
        let mut bits = 0u32;
        while bits < 0x7f80_0000 {
            check_e(f32::from_bits(bits), &mut worst);
            check_e(-f32::from_bits(bits), &mut worst);
            bits += 61;
        }
        for x in [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            E_LO,
            E_HI,
            -87.336_55,
            -87.4,
            -103.0,
            -103.28,
            -103.98,
            -104.0,
            -104.5,
            -104.66,
            88.0,
            88.722_83,
            88.73,
            89.0,
        ] {
            check_e(x, &mut worst);
        }
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert!(exp(f32::NAN).is_nan());
        println!("exp worst {} ulp at {:e}", worst.0, worst.1);
    }

    /// Every 7th bit pattern of `[-104, 0]`, the domain of SiLU's
    /// `e^-|x|` and of every softmax term: about 270 million.
    #[test]
    fn exp_is_within_bound_densely_in_minus_104_to_0() {
        let mut worst = (0u32, 0.0f32);
        let top = (-104.0f32).to_bits();
        let mut bits = 0x8000_0000u32;
        while bits <= top {
            check_e(f32::from_bits(bits), &mut worst);
            bits += 7;
        }
        println!("exp worst {} ulp at {:e}", worst.0, worst.1);
    }

    /// The slice kernels are the scalar functions lane by lane, and their
    /// sums are the documented fixed order.
    #[test]
    fn slice_kernels_are_the_scalar_kernels_and_a_fixed_order_sum() {
        let order = |want: &[f32]| {
            let mut lanes = [0.0f32; LANES];
            let whole = want.len() / LANES * LANES;
            for (i, &v) in want[..whole].iter().enumerate() {
                lanes[i % LANES] += v;
            }
            lanes
                .iter()
                .chain(&want[whole..])
                .fold(0.0f32, |s, &v| s + v)
        };
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 1000] {
            let base: Vec<f32> = (0..len).map(|i| ((i * 37) % 101) as f32 * -0.37).collect();
            let mut xs = base.clone();
            let total = exp2_affine(&mut xs, 0.18, -0.25);
            let want: Vec<f32> = base.iter().map(|&x| exp2(fma(x, 0.18, -0.25))).collect();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&xs), bits(&want), "exp2_affine len {len}");
            assert_eq!(total.to_bits(), order(&want).to_bits(), "len {len}");

            let words: Vec<[u8; 4]> = base.iter().map(|x| x.to_ne_bytes()).collect();
            let shift = -3.5f32;
            let want: Vec<f32> = base.iter().map(|&x| exp(x - shift)).collect();
            let sum = exp_sub_sum(&words, shift);
            assert_eq!(
                sum.to_bits(),
                order(&want).to_bits(),
                "exp_sub_sum len {len}"
            );
            let mut out = vec![[0u8; 4]; len];
            let stored = exp_sub_store(&words, shift, &mut out);
            assert_eq!(stored.to_bits(), sum.to_bits(), "exp_sub_store len {len}");
            let got: Vec<f32> = out.iter().map(|w| f32::from_ne_bytes(*w)).collect();
            assert_eq!(bits(&got), bits(&want), "exp_sub_store values len {len}");
        }
    }
}
