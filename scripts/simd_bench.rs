#![feature(portable_simd)]
//! Nightly gate only. This file is not a Cargo target, so stable
//! `clippy --all-targets` never compiles `std::simd`.

use std::simd::f32x8;

fn main() {
    let left = f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let right = f32x8::splat(0.5);
    let got = left + right;
    let mut expect = [0.0f32; 8];
    for (i, slot) in expect.iter_mut().enumerate() {
        *slot = left.to_array()[i] + 0.5;
    }
    assert_eq!(got.to_array(), expect);
    println!("portable_simd f32x8 add matches the scalar lane");
}
