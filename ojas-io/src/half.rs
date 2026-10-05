//! bf16 and f16 bit conversions, without a `half` dependency.
//!
//! Decoding (`*_to_f32`) is exact: every bf16 and f16 value is an f32.
//! Encoding (`f32_to_*`) rounds to nearest, ties to even, and overflows to
//! a signed infinity. NaN keeps its sign and the payload bits that fit (bf16:
//! the top 7 mantissa bits; f16: mantissa `>> 13`) and gets the quiet bit
//! (bf16 `0x0040`, f16 `0x0200`) set, so a NaN whose payload sits only in
//! the dropped low bits stays NaN instead of turning into infinity. A quiet
//! NaN therefore round-trips bit for bit; a signalling NaN comes back quiet.

/// `bits << 16`. The rounding lives in `ojas_core` so training and checkpoints share one implementation.
pub fn bf16_to_f32(bits: u16) -> f32 {
    ojas_core::bf16_to_f32(bits)
}

/// Round to nearest even on the dropped 16 bits.
pub fn f32_to_bf16(value: f32) -> u16 {
    ojas_core::f32_to_bf16(value)
}

/// Exact, including subnormals, signed zeros, infinities and NaN payloads.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let man = u32::from(bits & 0x03FF);
    match (exp, man) {
        (0, 0) => f32::from_bits(sign),
        // man * 2^-24 is exact in f32: man < 2^10 and the result is normal.
        (0, _) => f32::from_bits(sign | (man as f32 * f32::from_bits(0x3380_0000)).to_bits()),
        (31, 0) => f32::from_bits(sign | 0x7F80_0000),
        (31, _) => f32::from_bits(sign | 0x7F80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

/// Round to nearest even; `|value| >= 65520` becomes infinity, values at or
/// below `2^-25` in magnitude become a signed zero.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7FFF_FFFF;
    if abs > 0x7F80_0000 {
        return sign | 0x7C00 | 0x0200 | ((abs >> 13) & 0x03FF) as u16;
    }
    // f16 biased exponent of this value.
    let e = (abs >> 23) as i32 - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal result: (1.m) * 2^(e - 14) in units of 2^-24.
        let m = (abs & 0x007F_FFFF) | 0x0080_0000;
        let shift = (14 - e) as u32;
        let r = round_shift(m, shift);
        return sign | r as u16;
    }
    let h = ((e as u32) << 10) | ((abs >> 13) & 0x03FF);
    let rem = abs & 0x1FFF;
    // A carry out of the mantissa bumps the exponent; out of e = 30 it lands
    // exactly on 0x7C00, infinity.
    let h = if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h + 1
    } else {
        h
    };
    sign | h as u16
}

/// `m >> shift` rounded to nearest even. `1 <= shift <= 24`.
fn round_shift(m: u32, shift: u32) -> u32 {
    let r = m >> shift;
    let rem = m & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if rem > half || (rem == half && r & 1 == 1) {
        r + 1
    } else {
        r
    }
}
