//! `store_neg_abs_signs` against the scalar `z < 0` then `z = -|z|` loop.
//! `-0` is not `< 0`. `-|±0|` is stored as `-0`. The tail is included.
//! A non-finite lane stops before its chunk is stored and does not publish
//! signs.

use ojas_simd::{store_neg_abs_signs, SimdError};

fn scalar(z: &mut [f32]) -> Vec<u8> {
    let mut signs = Vec::with_capacity(z.len());
    for v in z.iter_mut() {
        signs.push(u8::from(*v < 0.0));
        *v = -v.abs();
    }
    signs
}

fn check(src: &[f32]) {
    let mut expect_z = src.to_vec();
    let expect_s = scalar(&mut expect_z);
    let mut got_z = src.to_vec();
    let mut got_s = Vec::new();
    store_neg_abs_signs(&mut got_z, &mut got_s).unwrap();
    assert_eq!(got_s, expect_s, "signs len {}", src.len());
    let expect_bits: Vec<u32> = expect_z.iter().map(|v| v.to_bits()).collect();
    let got_bits: Vec<u32> = got_z.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, expect_bits, "stored len {}", src.len());
}

#[test]
fn signs_and_neg_abs_match_scalar_including_negative_zero_and_the_tail() {
    check(&[]);
    check(&[-0.0]);
    check(&[0.0, -0.0]);
    let mut pattern = vec![
        -0.0,
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        40.0,
        -40.0,
        f32::from_bits(0x8000_0001),
        f32::MAX,
        -f32::MAX,
        3.25,
        -7.5,
        0.125,
    ];
    for n in 1..=pattern.len() {
        check(&pattern[..n]);
    }
    pattern.extend((0..48).map(|i| (i as f32) * 0.25 - 6.0));
    check(&pattern);
    assert_eq!((-0.0f32).to_bits(), 0x8000_0000);
    let mut zeros = [0.0f32, -0.0];
    let mut signs = Vec::new();
    store_neg_abs_signs(&mut zeros, &mut signs).unwrap();
    assert_eq!(signs, vec![0, 0]);
    assert_eq!(zeros[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(zeros[1].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn reuse_overwrites_a_previous_sign_buffer() {
    let mut z = [-1.0f32, 2.0, -0.0, 4.0];
    let mut signs = Vec::new();
    signs.reserve_exact(8);
    signs.extend_from_slice(&[9, 9, 9, 9]);
    store_neg_abs_signs(&mut z, &mut signs).unwrap();
    assert_eq!(signs, vec![1, 0, 0, 0]);
    assert_eq!(z[0].to_bits(), (-1.0f32).to_bits());
    assert_eq!(z[1].to_bits(), (-2.0f32).to_bits());
    assert_eq!(z[2].to_bits(), (-0.0f32).to_bits());
    assert_eq!(z[3].to_bits(), (-4.0f32).to_bits());
}

fn bits(z: &[f32]) -> Vec<u32> {
    z.iter().map(|v| v.to_bits()).collect()
}

/// How far [`store_neg_abs_signs`] stores before the chunk that holds
/// index `at`: 16-wide chunks, then groups of 4, then one tail lane at a time.
fn stored_before(n: usize, at: usize) -> usize {
    let wide = (n / 16) * 16;
    if at < wide {
        return (at / 16) * 16;
    }
    let mut i = wide;
    while i + 4 <= n {
        if at < i + 4 {
            return i;
        }
        i += 4;
    }
    at
}

/// A non-finite lane stops before that chunk is stored. Sign bytes are not
/// published. Lanes in earlier chunks may already hold `-|x|`.
#[test]
fn a_nonfinite_lane_stops_before_its_chunk_and_does_not_publish_signs() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for n in [1usize, 4, 6, 8, 17, 20] {
            for at in 0..n {
                let mut z = vec![1.0f32; n];
                z[at] = bad;
                let before = bits(&z);
                let mut signs = Vec::new();
                signs.reserve_exact(n + 8);
                signs.extend_from_slice(&[9, 9, 9, 9]);
                let err = store_neg_abs_signs(&mut z, &mut signs).unwrap_err();
                assert_eq!(err, SimdError::NonFinite, "n {n} at {at} {bad}");
                assert!(signs.is_empty(), "n {n} at {at}");
                let stop = stored_before(n, at);
                assert_eq!(bits(&z[stop..]), before[stop..], "n {n} at {at}");
                assert!(
                    z[..stop].iter().all(|v| v.to_bits() == (-1.0f32).to_bits()),
                    "n {n} at {at} stop {stop}"
                );
            }
        }
    }
}
