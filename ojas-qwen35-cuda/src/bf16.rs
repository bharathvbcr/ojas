//! bf16 on the host: round-to-nearest-even from f32, and the exact widening.
//!
//! The algorithm is tessl's (`tessl/src/tensor.rs:641-654`), bit for bit, and
//! the device cast kernel (`qd_cast_f32_to_bf16` in [`crate::kernels`]) is the
//! same integer sequence. A NaN keeps its sign and its top payload bits and is
//! quieted (bit 6 of the bf16 set), so a NaN whose payload sits entirely in
//! the discarded low half stays a NaN. Finite values round to nearest, ties
//! to even; a value at or past `max + half an ulp` rounds to infinity.

/// f32 to bf16 bits, round-to-nearest-even, NaNs quieted.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        // NaN: keep sign and top payload, set the quiet bit. The `as` keeps
        // the low 16 bits of a value already shifted into them.
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Largest non-NaN input is 0xff80_0000 (-inf); + 0x8000 cannot wrap.
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// bf16 bits to f32: exact (bf16 is the top half of an f32).
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// `x` rounded to the nearest bf16 and widened back to f32.
pub fn round_to_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

/// [`f32_to_bf16_bits`] over a slice.
pub fn f32_slice_to_bf16(xs: &[f32]) -> Vec<u16> {
    xs.iter().copied().map(f32_to_bf16_bits).collect()
}

/// [`bf16_bits_to_f32`] over a slice.
pub fn bf16_slice_to_f32(bs: &[u16]) -> Vec<f32> {
    bs.iter().copied().map(bf16_bits_to_f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An independent rounding: pick the nearer of the two bf16 neighbours by
    /// f64 arithmetic, ties to the even mantissa. Shares no code with the
    /// integer trick above.
    fn nearest_bf16_by_value(x: f32) -> u16 {
        assert!(!x.is_nan());
        if x.is_infinite() {
            return (x.to_bits() >> 16) as u16;
        }
        let sign = (x.to_bits() >> 16) as u16 & 0x8000;
        let mag = x.abs();
        let lo = (mag.to_bits() >> 16) as u16; // truncation toward zero
        let lo_v = f64::from(bf16_bits_to_f32(lo));
        let hi = lo + 1;
        // The neighbour above 0x7f7f is 0x7f80 (inf); RNE treats it as 2^128.
        let hi_v = if hi == 0x7f80 {
            2f64.powi(128)
        } else {
            f64::from(bf16_bits_to_f32(hi))
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

    use crate::inputs::splitmix64;

    #[test]
    fn ties_go_to_even_and_carries_propagate() {
        // 1 + 2^-8 is exactly half way between 1.0 (even) and 1 + 2^-7.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        // 1 + 3*2^-8 is half way between 1 + 2^-7 (odd) and 1 + 2^-6 (even).
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
        // Just past half way rounds up.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
        // A mantissa of all ones carries into the exponent.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3fff_ffff)), 0x4000);
    }

    #[test]
    fn overflow_rounds_to_infinity_and_infinities_stay() {
        assert_eq!(f32_to_bf16_bits(f32::MAX), 0x7f80);
        assert_eq!(f32_to_bf16_bits(-f32::MAX), 0xff80);
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16_bits(f32::NEG_INFINITY), 0xff80);
        // The largest finite bf16 plus less than half an ulp stays finite.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x7f7f_7fff)), 0x7f7f);
    }

    #[test]
    fn nans_stay_nans_with_the_quiet_bit_and_sign() {
        // Payload only in the low half: truncation would give infinity.
        let low_payload = f32::from_bits(0x7f80_0001);
        assert!(low_payload.is_nan());
        assert_eq!(f32_to_bf16_bits(low_payload), 0x7fc0);
        let negative = f32::from_bits(0xff80_0001);
        assert_eq!(f32_to_bf16_bits(negative), 0xffc0);
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn signed_zeros_and_subnormals_are_kept() {
        assert_eq!(f32_to_bf16_bits(0.0), 0x0000);
        assert_eq!(f32_to_bf16_bits(-0.0), 0x8000);
        // Smallest f32 subnormal rounds to +0 (below half the smallest bf16 subnormal).
        assert_eq!(f32_to_bf16_bits(f32::from_bits(1)), 0x0000);
        // Smallest bf16 subnormal survives.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x0001_0000)), 0x0001);
    }

    #[test]
    fn every_bf16_value_round_trips_exactly() {
        for b in 0..=u16::MAX {
            let x = bf16_bits_to_f32(b);
            if x.is_nan() {
                assert!(bf16_bits_to_f32(f32_to_bf16_bits(x)).is_nan());
            } else {
                assert_eq!(f32_to_bf16_bits(x), b, "bf16 {b:#06x}");
            }
        }
    }

    #[test]
    fn integer_trick_matches_the_value_rounding_on_two_million_inputs() {
        let mut state = 0x5eed_b16f_2026_1001u64;
        let mut checked = 0usize;
        for _ in 0..2_000_000 {
            let x = f32::from_bits((splitmix64(&mut state) >> 32) as u32);
            if x.is_nan() {
                continue;
            }
            assert_eq!(
                f32_to_bf16_bits(x),
                nearest_bf16_by_value(x),
                "x = {x:e} ({:#010x})",
                x.to_bits()
            );
            checked += 1;
        }
        // Every exact tie in the low half, across exponents.
        for hi in (0..=0x7f7fu32).step_by(7) {
            let x = f32::from_bits((hi << 16) | 0x8000);
            assert_eq!(f32_to_bf16_bits(x), nearest_bf16_by_value(x), "{x:e}");
            checked += 1;
        }
        assert!(checked > 1_990_000, "only {checked} inputs checked");
    }
}
