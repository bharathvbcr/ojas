//! Each generator reproduces, bit for bit, the per-file copy it replaced.
//! The oracles below are those copies' bodies, kept verbatim.

use ojas_test_support::vals::{vals, vals_f64, vals_range, vals_range_f64, vals_unit};
use ojas_test_support::SplitMix64;

fn state(seed: u64) -> u64 {
    seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1
}

fn old_amp(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = state(seed);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (-1.0 + 2.0 * ((s >> 40) as f32 / (1u64 << 24) as f32)) * amp
        })
        .collect()
}

fn old_amp_flipped(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = state(seed);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn old_unit(seed: u64, n: usize) -> Vec<f32> {
    let mut s = state(seed);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            -1.0 + 2.0 * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn old_range(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = state(seed);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn old_f64(seed: u64, n: usize, amp: f64) -> Vec<f64> {
    let mut s = state(seed);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn old_range_f64(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut s = state(seed);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect()
}

fn old_pins(seed: u64, n: usize) -> Vec<f64> {
    let mut s = state(seed);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f64::from((((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32)
        })
        .collect()
}

fn bits32(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn bits64(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

const SEEDS: [u64; 5] = [0, 1, 7, 1337, u64::MAX];

#[test]
fn the_f32_generators_match_the_copies_they_replaced() {
    for seed in SEEDS {
        for amp in [1.0f32, 0.5, 3.0] {
            assert_eq!(
                bits32(&vals(seed, 257, amp)),
                bits32(&old_amp(seed, 257, amp))
            );
            assert_eq!(
                bits32(&vals(seed, 257, amp)),
                bits32(&old_amp_flipped(seed, 257, amp))
            );
        }
        assert_eq!(bits32(&vals_unit(seed, 257)), bits32(&old_unit(seed, 257)));
        for (lo, hi) in [(-1.0f32, 1.0f32), (0.5, 1.5), (-2.0, 2.0)] {
            assert_eq!(
                bits32(&vals_range(seed, 257, lo, hi)),
                bits32(&old_range(seed, 257, lo, hi))
            );
        }
    }
}

#[test]
fn the_f64_generators_match_the_copies_they_replaced() {
    for seed in SEEDS {
        assert_eq!(
            bits64(&vals_f64(seed, 257, 1.5)),
            bits64(&old_f64(seed, 257, 1.5))
        );
        for (lo, hi) in [(-1.0f64, 1.0f64), (0.5, 1.5), (-3.0, 3.0)] {
            assert_eq!(
                bits64(&vals_range_f64(seed, 257, lo, hi)),
                bits64(&old_range_f64(seed, 257, lo, hi))
            );
        }
        // `spatial_pins.rs` rounds the 53-bit `[-1, 1)` stream to f32.
        let rounded: Vec<f64> = vals_range_f64(seed, 257, -1.0, 1.0)
            .into_iter()
            .map(|x| f64::from(x as f32))
            .collect();
        assert_eq!(bits64(&rounded), bits64(&old_pins(seed, 257)));
    }
}

#[test]
fn an_amplitude_is_not_a_range_and_the_two_stay_separate() {
    let amp = vals(3, 4096, 0.3);
    let range = vals_range(3, 4096, -0.3, 0.3);
    assert_ne!(
        bits32(&amp),
        bits32(&range),
        "the roundings differ somewhere"
    );
}

#[test]
fn splitmix64_reproduces_the_published_stream() {
    // First outputs of splitmix64 seeded 0 (Vigna's reference implementation).
    let mut r = SplitMix64(0);
    assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(r.next_u64(), 0x06c4_5d18_8009_454f);
}

#[test]
fn splitmix64_helpers_stay_in_range() {
    let mut r = SplitMix64::new(9);
    for _ in 0..1000 {
        assert!((-1.0..1.0).contains(&r.unit()));
        assert!(r.below(7) < 7);
    }
    assert_eq!(r.vec(5, 2.0).len(), 5);
}
