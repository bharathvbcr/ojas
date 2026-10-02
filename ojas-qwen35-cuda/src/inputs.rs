//! Deterministic test inputs, with no dependency.
//!
//! - [`splitmix64`] / [`splitmix_f32`] are ojas's generator
//!   (`ojas-kernels/src/harness.rs:31-45`), so the same seed gives the same
//!   values here and in ojas's parity harness.
//! - [`tessl_ragged_a`] / [`tessl_ragged_b`] are the inputs of tessl's
//!   `gemm_bf16_nn_ragged_mn` test (`tessl/src/gemm.rs:2349-2350`), so the
//!   GEMM shapes borrowed from that test also use its values.

/// One SplitMix64 step.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `n` values in `[-scale, scale)`, as ojas's `splitmix_f32`.
pub fn splitmix_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let z = splitmix64(&mut state);
            // The top 24 bits are exact in f32.
            let mantissa = (z >> 40) as f32 / (1u64 << 24) as f32;
            scale * (2.0 * mantissa - 1.0)
        })
        .collect()
}

/// `n` raw 32-bit patterns: every class of f32 (NaN, inf, subnormal, zero)
/// appears, which is what a bit-exact cast test needs.
pub fn splitmix_bits(seed: u64, n: usize) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| (splitmix64(&mut state) >> 32) as u32)
        .collect()
}

/// f32 values whose bit patterns sweep the rounding edge cases of an f32 to
/// bf16 cast: zeros, subnormals, exact ties, ties plus one, the overflow
/// boundary, infinities and NaNs with payloads in either half.
pub fn bf16_cast_edge_cases() -> Vec<f32> {
    let bits: [u32; 24] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0000_8000,
        0x0001_0000,
        0x0001_8000,
        0x3f80_0000,
        0x3f80_8000,
        0x3f81_8000,
        0x3f80_8001,
        0x3f80_7fff,
        0x3fff_ffff,
        0x7f7f_7fff,
        0x7f7f_8000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc0_0000,
        0x7fbf_ffff,
        0xffff_ffff,
    ];
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

/// tessl's A pattern for an `[rows, cols]`-shaped operand of `len` values:
/// `(i % 251) / 256 - 0.49`.
pub fn tessl_ragged_a(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 251) as f32) / 256.0 - 0.49)
        .collect()
}

/// tessl's B pattern: `(i % 241) / 256 - 0.47`.
pub fn tessl_ragged_b(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % 241) as f32) / 256.0 - 0.47)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_f32_stays_in_range_and_repeats_by_seed() {
        let a = splitmix_f32(7, 1000, 0.5);
        let b = splitmix_f32(7, 1000, 0.5);
        assert_eq!(a, b);
        assert!(a.iter().all(|x| (-0.5..0.5).contains(x)));
        assert_ne!(a, splitmix_f32(8, 1000, 0.5));
    }

    #[test]
    fn edge_cases_cover_every_class() {
        let xs = bf16_cast_edge_cases();
        assert!(xs.iter().any(|x| x.is_nan()));
        assert!(xs.iter().any(|x| x.is_infinite()));
        assert!(xs.iter().any(|x| x.is_subnormal()));
        assert!(xs.iter().any(|x| *x == 0.0 && x.is_sign_negative()));
    }

    #[test]
    fn tessl_patterns_match_their_formulas() {
        let a = tessl_ragged_a(260);
        assert_eq!(a[0], -0.49);
        assert_eq!(a[251], -0.49);
        assert_eq!(a[250], 250.0 / 256.0 - 0.49);
        let b = tessl_ragged_b(242);
        assert_eq!(b[241], -0.47);
    }
}
