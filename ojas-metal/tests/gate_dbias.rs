//! The per-head gate's bias gradient on Metal sums `d_pre` over rows in
//! ascending order from 0, as the CPU reference does, bit for bit.
//!
//! This carries the check the retired tiny step's overlay tests made on
//! `ojas_per_head_gate_dbias`, through `MetalBackend` instead of a private
//! dispatch. With `x`, `W` and the bias all zero every gate is exactly 0.5.
//! With head dimension 1 and `attn = 1`, `d_pre[r, h] = dy[r, h] * 0.5 * 0.5`,
//! which is exact, so `d_bias[h]` is the f32 fold of `dy / 4` down the rows.
//! Values of +/-3e7 sit among small ones, so a sum taken in any other order
//! (pairwise, reversed, a chunk dropped or repeated, the wrong stride) lands
//! on other bits. Row counts straddle the 32-lane strides, the 8-row unrolled
//! adds and the 2048-row threadgroup blocks.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::Backend;
use ojas_metal::MetalBackend;

fn dy(r: usize, h: usize) -> f32 {
    match (r * 7 + h * 3) % 5 {
        0 => 3.0e7,
        2 => -3.0e7,
        k => 0.37 + (r % 11) as f32 * 0.13 + k as f32 + h as f32 * 0.01,
    }
}

/// One gate-backward input: `dy` with a zero `x`, `W` and bias, so
/// `d_pre = dy / 4`.
struct Case {
    rows: usize,
    heads: usize,
    gy: Vec<f32>,
}

impl Case {
    fn new(rows: usize, heads: usize) -> Self {
        let gy = (0..rows * heads)
            .map(|i| dy(i / heads, i % heads))
            .collect();
        Self { rows, heads, gy }
    }

    /// The f32 fold of `dy / 4` down the rows, ascending from 0.
    fn want(&self) -> Vec<f32> {
        (0..self.heads)
            .map(|h| (0..self.rows).fold(0.0f32, |s, r| s + self.gy[r * self.heads + h] * 0.25))
            .collect()
    }

    fn run(&self, m: &MetalBackend) -> Vec<f32> {
        let (rows, heads) = (self.rows, self.heads);
        let x = up(m, &host(&vec![0.0; rows], &[rows, 1]));
        let w = up(m, &host(&vec![0.0; heads], &[heads, 1]));
        let bias = up(m, &host(&vec![0.0; heads], &[heads]));
        let attn = up(m, &host(&vec![1.0; rows * heads], &[rows, heads, 1]));
        let gy = up(m, &host(&self.gy, &[rows, heads, 1]));
        let grads = ok(
            "gate backward",
            m.per_head_sigmoid_gate_backward(&x, &w, &bias, &attn, &gy),
        );
        ok("sync", m.sync());
        down(&grads.bias)
    }
}

#[test]
fn the_bias_gradient_sums_rows_in_ascending_order_bit_for_bit() {
    let m = metal();
    for heads in [1usize, 3, 12] {
        for rows in [
            1usize, 31, 32, 33, 64, 65, 127, 128, 129, 1024, 2047, 2048, 2049, 4096, 4097,
        ] {
            let case = Case::new(rows, heads);
            let (got, want) = (case.run(&m), case.want());
            assert_eq!(
                bits(&got),
                bits(&want),
                "rows {rows}, heads {heads}: {got:?} vs {want:?}"
            );
        }
    }
}

/// The same inputs give the same bits on every dispatch: the retired tiny
/// step's overlay tests repeated the gate kernel 1000 times for this.
#[test]
fn the_bias_gradient_is_the_same_bits_on_every_dispatch() {
    let m = metal();
    let case = Case::new(4097, 12);
    let first = bits(&case.run(&m));
    assert_eq!(first, bits(&case.want()), "first dispatch");
    for round in 1..200 {
        assert_eq!(bits(&case.run(&m)), first, "dispatch {round}");
    }
}
