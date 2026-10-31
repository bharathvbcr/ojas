//! `f32_to_bf16` and `bf16_to_f32` against an independent value-based rounding.
//!
//! The oracle picks the nearer of the two bf16 neighbours by f64 arithmetic,
//! ties to the even mantissa, and shares no code with the integer sequence
//! under test. These checks lived in ojas-cuda's host bf16 module until that
//! module was deleted in favour of these functions (the one host rule every
//! ojas crate shares); they moved here, and the sweep below was added so the
//! comparison is exhaustive in the sense that matters.
//!
//! Round-to-nearest-even on the dropped low half depends only on the high 16
//! bits (which fix the sign, the exponent, the kept mantissa and its parity)
//! and on whether the low 16 bits are below, equal to or above `0x8000`; a
//! NaN is told apart from an infinity by whether any mantissa bit is set. So
//! all 65,536 high halves, each with low halves from every one of those
//! classes, cover every distinct rounding decision in the f32 space.

use ojas_core::{bf16_to_f32, f32_to_bf16};

/// The nearer bf16 neighbour of a non-NaN `x`, by value; ties to even.
fn nearest_bf16_by_value(x: f32) -> u16 {
    assert!(!x.is_nan());
    if x.is_infinite() {
        return (x.to_bits() >> 16) as u16;
    }
    let sign = (x.to_bits() >> 16) as u16 & 0x8000;
    let mag = x.abs();
    let lo = (mag.to_bits() >> 16) as u16; // truncation toward zero
    let lo_v = f64::from(bf16_to_f32(lo));
    let hi = lo + 1;
    // The neighbour above 0x7f7f is 0x7f80 (inf); RNE treats it as 2^128.
    let hi_v = if hi == 0x7f80 {
        2f64.powi(128)
    } else {
        f64::from(bf16_to_f32(hi))
    };
    let m = f64::from(mag);
    let pick = if m - lo_v < hi_v - m {
        lo
    } else if m - lo_v > hi_v - m {
        hi
    } else if lo & 1 == 0 {
        lo
    } else {
        hi
    };
    sign | pick
}

/// The documented NaN rule: sign and top payload kept, quiet bit set.
fn expected_nan(bits: u32) -> u16 {
    ((bits >> 16) as u16) | 0x0040
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn widening_is_exact_for_all_65536_bf16_patterns() {
    for b in 0..=u16::MAX {
        let x = bf16_to_f32(b);
        assert_eq!(x.to_bits(), u32::from(b) << 16, "bf16 {b:#06x}");
    }
}

#[test]
fn every_non_nan_bf16_value_round_trips_and_every_nan_stays_a_nan() {
    for b in 0..=u16::MAX {
        let x = bf16_to_f32(b);
        if x.is_nan() {
            let back = f32_to_bf16(x);
            assert!(bf16_to_f32(back).is_nan(), "bf16 {b:#06x}");
            // A bf16 NaN is an f32 NaN with a zero low half: the rule only
            // sets the quiet bit.
            assert_eq!(back, b | 0x0040, "bf16 {b:#06x}");
        } else {
            assert_eq!(f32_to_bf16(x), b, "bf16 {b:#06x}");
        }
    }
}

#[test]
fn every_rounding_class_of_every_high_half_matches_the_value_oracle() {
    const LOWS: [u32; 8] = [
        0x0000, 0x0001, 0x4000, 0x7fff, 0x8000, 0x8001, 0xc000, 0xffff,
    ];
    let mut checked = 0usize;
    for hi in 0..=u32::from(u16::MAX) {
        for low in LOWS {
            let bits = (hi << 16) | low;
            let x = f32::from_bits(bits);
            let got = f32_to_bf16(x);
            if x.is_nan() {
                assert_eq!(got, expected_nan(bits), "NaN {bits:#010x}");
                assert!(
                    bf16_to_f32(got).is_nan(),
                    "NaN {bits:#010x} became a number"
                );
            } else {
                assert_eq!(got, nearest_bf16_by_value(x), "{x:e} ({bits:#010x})");
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 65_536 * LOWS.len());
}

#[test]
fn ties_go_to_even_and_carries_propagate() {
    // 1 + 2^-8 is exactly half way between 1.0 (even) and 1 + 2^-7.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
    // 1 + 3*2^-8 is half way between 1 + 2^-7 (odd) and 1 + 2^-6 (even).
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    // Just past half way rounds up.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
    // A mantissa of all ones carries into the exponent.
    assert_eq!(f32_to_bf16(f32::from_bits(0x3fff_ffff)), 0x4000);
}

#[test]
fn overflow_rounds_to_infinity_and_infinities_stay() {
    assert_eq!(f32_to_bf16(f32::MAX), 0x7f80);
    assert_eq!(f32_to_bf16(-f32::MAX), 0xff80);
    assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
    assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
    // The largest finite bf16 plus less than half an ulp stays finite.
    assert_eq!(f32_to_bf16(f32::from_bits(0x7f7f_7fff)), 0x7f7f);
}

#[test]
fn nans_keep_sign_and_top_payload_and_gain_the_quiet_bit() {
    // Payload only in the low half: truncation would give infinity.
    assert_eq!(f32_to_bf16(f32::from_bits(0x7f80_0001)), 0x7fc0);
    assert_eq!(f32_to_bf16(f32::from_bits(0xff80_0001)), 0xffc0);
    // Top payload bits survive; a signalling NaN comes back quiet.
    assert_eq!(f32_to_bf16(f32::from_bits(0x7fa5_1234)), 0x7fa5 | 0x0040);
    assert_eq!(f32_to_bf16(f32::from_bits(0xff81_0000)), 0xffc1);
    assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
}

#[test]
fn signed_zeros_and_subnormals_are_kept() {
    assert_eq!(f32_to_bf16(0.0), 0x0000);
    assert_eq!(f32_to_bf16(-0.0), 0x8000);
    // Smallest f32 subnormal rounds to +0 (below half the smallest bf16 subnormal).
    assert_eq!(f32_to_bf16(f32::from_bits(1)), 0x0000);
    // Smallest bf16 subnormal survives.
    assert_eq!(f32_to_bf16(f32::from_bits(0x0001_0000)), 0x0001);
    // Exactly half the smallest bf16 subnormal is a tie to the even zero.
    assert_eq!(f32_to_bf16(f32::from_bits(0x0000_8000)), 0x0000);
    assert_eq!(f32_to_bf16(f32::from_bits(0x8000_8000)), 0x8000);
    // Three halves of it is a tie to the even 2.
    assert_eq!(f32_to_bf16(f32::from_bits(0x0001_8000)), 0x0002);
}

#[test]
fn random_bit_patterns_match_the_value_oracle() {
    let mut state = 0x5eed_b16f_2026_1001u64;
    let mut checked = 0usize;
    for _ in 0..2_000_000 {
        let bits = (splitmix64(&mut state) >> 32) as u32;
        let x = f32::from_bits(bits);
        if x.is_nan() {
            assert_eq!(f32_to_bf16(x), expected_nan(bits), "NaN {bits:#010x}");
            continue;
        }
        assert_eq!(
            f32_to_bf16(x),
            nearest_bf16_by_value(x),
            "x = {x:e} ({bits:#010x})"
        );
        checked += 1;
    }
    assert!(checked > 1_990_000, "only {checked} inputs checked");
}
