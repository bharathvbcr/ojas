//! `ojas_check_finite`, the pass `MetalBackend` runs over every op's inputs
//! and outputs, must see every element of the window it is given and
//! nothing outside it. Each thread checks several elements, so a single
//! non-finite value is placed at the start, the middle and the end, and at
//! every element of small and odd lengths, around thread and threadgroup
//! sizes, and in views whose window starts at an offset that is not a
//! multiple of 16 bytes.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};
use ojas_metal::MetalBackend;

/// What `sync` reports: `None` for `Ok`, the op for `NonFinite`.
fn pending(m: &MetalBackend) -> Option<&'static str> {
    match m.sync() {
        Ok(()) => None,
        Err(OjasError::NonFinite { op }) => Some(op),
        Err(e) => panic!("sync: {e:?}"),
    }
}

/// `silu_forward` checks its input and its output (same length), so a
/// skipped index would be skipped by both checks and the fault would go
/// unreported.
fn silu_reports(m: &MetalBackend, x: &Tensor) -> Option<&'static str> {
    let y = ok("silu records", m.silu_forward(x));
    let got = pending(m);
    drop(y);
    got
}

#[test]
fn one_non_finite_value_is_found_at_every_tested_position() {
    let m = metal();
    let lengths = [
        1usize,
        2,
        3,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        4095,
        4096,
        4097,
        65_537,
        (1 << 20) + 3,
    ];
    for n in lengths {
        let mut positions: Vec<usize> = if n <= 33 {
            (0..n).collect()
        } else {
            vec![0, 1, 7, 8, 31, 32, n / 2, n - 9, n - 8, n - 2, n - 1]
        };
        positions.sort_unstable();
        positions.dedup();
        for p in positions {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut v = vec![0.5f32; n];
                v[p] = bad;
                let x = up(&m, &host(&v, &[n]));
                assert_eq!(
                    silu_reports(&m, &x),
                    Some("silu_forward"),
                    "length {n}, {bad} at {p} was not found"
                );
            }
        }
        let clean = up(&m, &host(&vec![0.5f32; n], &[n]));
        assert_eq!(
            silu_reports(&m, &clean),
            None,
            "length {n}, clean, reported a fault"
        );
    }
}

#[test]
fn a_view_checks_only_its_own_window() {
    let m = metal();
    // Rows of 5 floats: row r starts at byte 20 r, never a multiple of 16
    // for r = 1, 2, 3.
    let (rows, width) = (4usize, 5usize);
    for r in 1..rows {
        let mut v = vec![0.5f32; rows * width];
        // A NaN in every other row, just before and just after this window.
        v[r * width - 1] = f32::NAN;
        if (r + 1) * width < v.len() {
            v[(r + 1) * width] = f32::NAN;
        }
        let base = up(&m, &host(&v, &[rows * width]));
        let row = ok("row view", base.view(&[width], &[1], r * width * 4));
        assert_eq!(
            silu_reports(&m, &row),
            None,
            "row {r} saw a NaN outside its window"
        );
        for p in [0, width / 2, width - 1] {
            let mut w = v.clone();
            w[r * width + p] = f32::NAN;
            let base = up(&m, &host(&w, &[rows * width]));
            let row = ok("row view", base.view(&[width], &[1], r * width * 4));
            assert_eq!(
                silu_reports(&m, &row),
                Some("silu_forward"),
                "row {r}: NaN at {p} inside the window was not found"
            );
        }
    }
}
