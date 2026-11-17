//! Seeded value generators: one xorshift64 stream per call, so a test's
//! inputs are the same on every run and every machine.
//!
//! The stream is `s = seed * 0x9e3779b97f4a7c15 | 1`, then per value
//! `s ^= s << 13; s ^= s >> 7; s ^= s << 17`. The `f32` generators map the top
//! 24 bits of `s` to `u` in `[0, 1)`; [`vals_range_f64`] maps the top 53 bits.
//!
//! The mappings below round differently, so they are not interchangeable:
//! `vals(seed, n, amp)` is `(-1 + 2u) * amp`, while `vals_range(seed, n, -amp,
//! amp)` is `-amp + 2 amp u`. A test pinned to one keeps that one.

fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

fn unit24_f32(s: u64) -> f32 {
    (s >> 40) as f32 / (1u64 << 24) as f32
}

/// `n` values in `[-amp, amp)`, as `(-1 + 2u) * amp`.
pub fn vals(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut next = stream(seed);
    (0..n)
        .map(|_| (-1.0 + 2.0 * unit24_f32(next())) * amp)
        .collect()
}

/// `n` values in `[-1, 1)`, as `-1 + 2u`.
pub fn vals_unit(seed: u64, n: usize) -> Vec<f32> {
    let mut next = stream(seed);
    (0..n).map(|_| -1.0 + 2.0 * unit24_f32(next())).collect()
}

/// `n` values in `[lo, hi)`, as `lo + (hi - lo) * u`.
pub fn vals_range(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut next = stream(seed);
    (0..n)
        .map(|_| lo + (hi - lo) * unit24_f32(next()))
        .collect()
}

/// `n` `f64` values in `[-amp, amp)` from 24-bit steps, as `(2u - 1) * amp`.
pub fn vals_f64(seed: u64, n: usize, amp: f64) -> Vec<f64> {
    let mut next = stream(seed);
    (0..n)
        .map(|_| ((next() >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) * amp)
        .collect()
}

/// `n` `f64` values in `[lo, hi)` from 53-bit steps, as `lo + (hi - lo) * u`.
pub fn vals_range_f64(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut next = stream(seed);
    (0..n)
        .map(|_| lo + (hi - lo) * ((next() >> 11) as f64 / (1u64 << 53) as f64))
        .collect()
}
