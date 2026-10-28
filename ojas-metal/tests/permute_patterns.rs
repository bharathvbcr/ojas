//! `MetalBackend::permute` on the kernel the pattern planner picks
//! (`src/permute.rs`): row copies, a transpose tile, or the generic kernel.
//! Every output must equal the CPU backend's and `permute_generic`'s, bit for
//! bit, at the shapes SpikeTrack calls `permute` with and at odd ones; the
//! planner must pick the kernel it is meant to; a call is one dispatch more
//! than a scan, as the generic kernel's is; and a non-finite input is refused
//! as today.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeSet;

use common::*;
use ojas_core::{Backend, Tensor};
use ojas_metal::MetalBackend;

/// Row-major strides of `shape`, in elements.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Values that exercise bit preservation, then random data: negative zero,
/// subnormals of both signs, extremes, and distinct payload bits.
fn data(n: usize, seed: u64) -> Vec<f32> {
    let specials = [
        (-0.0f32).to_bits(),
        1,
        0x807F_FFFF,
        f32::MAX.to_bits(),
        f32::MIN.to_bits(),
        f32::MIN_POSITIVE.to_bits(),
        0x3F80_0001,
        0xBEAA_AAAB,
        0x0000_0100,
        0x7F7F_FFFE,
    ];
    let mut v = values(n, seed, 1.0);
    for (i, slot) in v.iter_mut().enumerate().step_by(7) {
        *slot = f32::from_bits(specials[(i / 7) % specials.len()]);
    }
    v
}

/// The planner's choice, the new kernel's output and the generic kernel's,
/// all three compared with the CPU's by bits. Returns the pattern.
fn check(m: &MetalBackend, c: &impl Backend, x: &Tensor, dx: &Tensor, dims: &[usize]) -> String {
    let what = format!("{:?} by {dims:?}", x.shape());
    let want = bits(&ok("cpu", ok("cpu", c.permute(x, dims)).to_f32_vec()));
    let pattern = ok("pattern", m.permute_pattern(dx, dims)).to_string();
    let got = ok("permute", m.permute(dx, dims));
    let generic = ok("generic", m.permute_generic(dx, dims));
    ok("sync", m.sync());
    assert_eq!(
        bits(&down(&got)),
        want,
        "{what}: {pattern} differs from the CPU"
    );
    assert_eq!(
        bits(&down(&generic)),
        want,
        "{what}: the generic kernel differs from the CPU"
    );
    pattern
}

fn input(m: &MetalBackend, shape: &[usize], seed: u64) -> (Tensor, Tensor) {
    let x = host(&data(shape.iter().product(), seed), shape);
    let dx = up(m, &x);
    (x, dx)
}

/// Every call site of `permute` in `st-model`, with the pattern it must take
/// (`bench/results/2026-10-27-memory-bound`).
const CALL_SITES: &[(&[usize], &[usize], &str)] = &[
    (&[9, 576, 16, 64], &[0, 2, 1, 3], "runs"),
    (&[9, 16, 576, 64], &[0, 2, 1, 3], "runs"),
    (&[1, 5184, 16, 64], &[0, 2, 1, 3], "runs"),
    (&[1, 16, 5184, 64], &[0, 2, 1, 3], "runs"),
    (&[1, 3, 24, 3, 24, 1024], &[0, 1, 3, 2, 4, 5], "runs"),
    (&[1, 3, 3, 24, 24, 1024], &[0, 1, 3, 2, 4, 5], "runs"),
    (&[1, 1024, 72, 72], &[0, 2, 3, 1], "transpose"),
    (&[1, 72, 72, 1024], &[0, 3, 1, 2], "transpose"),
    (&[1, 256, 72, 72], &[0, 2, 3, 1], "transpose"),
    (&[1, 72, 72, 256], &[0, 3, 1, 2], "transpose"),
    (&[1, 256, 5184], &[0, 2, 1], "transpose"),
    (&[1, 5184, 256], &[0, 2, 1], "transpose"),
    (&[256, 82944], &[1, 0], "transpose"),
    (&[32, 82944], &[1, 0], "transpose"),
    (&[1, 5184, 8, 32], &[0, 2, 1, 3], "runs"),
    (&[1, 8, 5184, 32], &[0, 2, 1, 3], "runs"),
    (&[1, 10400, 8, 32], &[0, 2, 1, 3], "runs"),
    (&[1, 8, 201, 32], &[0, 2, 1, 3], "runs"),
    (&[1, 32, 16, 64], &[0, 2, 1, 3], "runs"),
    (&[1, 16, 32, 64], &[0, 2, 1, 3], "runs"),
];

#[test]
fn every_st_model_call_site_takes_its_pattern_and_matches_cpu_and_generic_bit_for_bit() {
    let (m, c) = (metal(), cpu());
    for (i, (shape, dims, want)) in CALL_SITES.iter().enumerate() {
        let (x, dx) = input(&m, shape, 100 + i as u64);
        let pattern = check(&m, &c, &x, &dx, dims);
        assert_eq!(pattern, *want, "{shape:?} by {dims:?}");
    }
}

#[test]
fn every_permutation_of_shapes_that_reach_each_kernel_matches() {
    let (m, c) = (metal(), cpu());
    // Distinct extents so a swapped pair of axes cannot pass; runs of 16 and
    // 8 floats, tiles of 40 x 36 (edge tiles in both axes), a run of 12 that
    // is a whole number of vectors, and extents with no vector run at all.
    let shapes: &[&[usize]] = &[
        &[2, 3, 5, 16],
        &[2, 3, 5, 48],
        &[2, 3, 5, 12],
        &[3, 5, 8, 2],
        &[40, 36],
        &[3, 40, 36],
        &[2, 9, 12],
        &[5, 7, 3],
        &[2, 3, 4, 5, 8],
    ];
    let mut seen = BTreeSet::new();
    for (i, shape) in shapes.iter().enumerate() {
        let (x, dx) = input(&m, shape, 200 + i as u64);
        for dims in permutations(shape.len()) {
            seen.insert(check(&m, &c, &x, &dx, &dims));
        }
    }
    // The set is only a test of the kernels if each ran.
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        ["generic", "runs", "transpose"]
    );
}

#[test]
fn ranks_one_to_eight_with_size_one_axes_match() {
    let (m, c) = (metal(), cpu());
    // Axes of 8, 1, 4, 16, 1, 2, 8, 4 floats: size-1 axes, runs that merge
    // across them, and every rank from 1 to 8.
    let axes = [8usize, 1, 4, 16, 1, 2, 8, 4];
    let mut rng = 0x1234_5678_9ABC_DEF1u64;
    for rank in 1..=8 {
        let shape = &axes[..rank];
        let (x, dx) = input(&m, shape, 300 + rank as u64);
        let all = permutations(rank);
        let picked: Vec<&Vec<usize>> = if all.len() <= 120 {
            all.iter().collect()
        } else {
            (0..120)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    &all[(rng % all.len() as u64) as usize]
                })
                .collect()
        };
        for dims in picked {
            check(&m, &c, &x, &dx, dims);
        }
    }
    // The identity of a long vector is one run.
    let (x, dx) = input(&m, &[1024], 310);
    assert_eq!(check(&m, &c, &x, &dx, &[0]), "runs");
}

#[test]
fn offset_views_read_their_own_window_on_either_kernel() {
    let (m, c) = (metal(), cpu());
    let shape = [3usize, 8, 4, 16];
    let n: usize = shape.iter().product();
    let x = host(&data(n, 400), &shape);
    for (lead, want) in [
        (0, "runs"),
        (4, "runs"),
        (8, "runs"),
        (2, "generic"),
        (6, "generic"),
    ] {
        let mut padded = values(lead, 401, 1.0);
        padded.extend_from_slice(&ok("host", x.to_f32_vec()));
        padded.extend(values(5, 402, 1.0));
        let big = up(&m, &host(&padded, &[padded.len()]));
        let view = ok("narrow", big.narrow(lead * 4, &shape, &strides(&shape)));
        let pattern = check(&m, &c, &x, &view, &[0, 2, 1, 3]);
        assert_eq!(pattern, want, "a view {lead} floats in");
    }
    // The same through a transpose and through the generic kernel.
    let shape = [2usize, 40, 36];
    let n: usize = shape.iter().product();
    let x = host(&data(n, 410), &shape);
    for lead in [0usize, 1, 4, 5] {
        let mut padded = values(lead, 411, 1.0);
        padded.extend_from_slice(&ok("host", x.to_f32_vec()));
        padded.extend(values(5, 412, 1.0));
        let big = up(&m, &host(&padded, &[padded.len()]));
        let view = ok("narrow", big.narrow(lead * 4, &shape, &strides(&shape)));
        assert_eq!(check(&m, &c, &x, &view, &[0, 2, 1]), "transpose");
    }
}

/// The generic kernel is a scan and a kernel (two dispatches); the row-copy
/// and transpose kernels test what they read, so a call is one dispatch.
#[test]
fn a_pattern_call_is_one_dispatch_and_the_generic_call_is_a_scan_and_a_kernel() {
    let m = metal();
    let count = |dx: &Tensor, dims: &[usize], generic: bool| {
        ok("sync", m.sync());
        let before = m.dispatches();
        let y = if generic {
            ok("generic", m.permute_generic(dx, dims))
        } else {
            ok("permute", m.permute(dx, dims))
        };
        ok("sync", m.sync());
        drop(y);
        m.dispatches() - before
    };
    for (i, (shape, dims, want)) in CALL_SITES.iter().enumerate().step_by(3) {
        let (_, dx) = input(&m, shape, 500 + i as u64);
        assert_eq!(ok("pattern", m.permute_pattern(&dx, dims)), *want);
        assert_eq!(
            count(&dx, dims, true),
            2,
            "{shape:?}: scan and ojas_permute"
        );
        assert_eq!(count(&dx, dims, false), 1, "{shape:?} by {dims:?}: {want}");
    }
    // A shape on the generic kernel keeps its scan.
    let (_, dx) = input(&m, &[5, 7, 3], 520);
    assert_eq!(ok("pattern", m.permute_pattern(&dx, &[2, 1, 0])), "generic");
    assert_eq!(count(&dx, &[2, 1, 0], false), 2);
}

/// The fused finite test sees every word: a NaN or infinity at any single
/// position of a small input, on each kernel, is a deferred fault.
#[test]
fn a_non_finite_word_at_any_position_is_seen_by_every_kernel() {
    let m = metal();
    for (shape, dims, want) in [
        (&[2usize, 3, 4, 16][..], &[0usize, 2, 1, 3][..], "runs"),
        (&[2, 9, 10][..], &[0, 2, 1][..], "transpose"),
        (&[3, 5, 7][..], &[2, 1, 0][..], "generic"),
    ] {
        let n: usize = shape.iter().product();
        for at in 0..n {
            let bad = if at % 2 == 0 {
                f32::NAN
            } else {
                f32::NEG_INFINITY
            };
            let mut v = values(n, 700, 1.0);
            v[at] = bad;
            let dx = up(&m, &host(&v, shape));
            assert_eq!(ok("pattern", m.permute_pattern(&dx, dims)), want);
            deferred(
                &m,
                &format!("{shape:?} position {at} {bad}"),
                m.permute(&dx, dims),
                "permute",
            );
        }
    }
}

#[test]
fn a_non_finite_input_is_refused_as_before_on_every_kernel() {
    let m = metal();
    for (shape, dims) in [
        (&[3usize, 8, 4, 16][..], &[0usize, 2, 1, 3][..]), // runs
        (&[2, 40, 36][..], &[0, 2, 1][..]),                // transpose
        (&[5, 7, 3][..], &[2, 1, 0][..]),                  // generic
    ] {
        for bad in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7FC0_1234),
        ] {
            let mut v = values(shape.iter().product(), 600, 1.0);
            let last = v.len() - 1;
            v[last / 2] = bad;
            v[last] = bad;
            let dx = up(&m, &host(&v, shape));
            deferred(
                &m,
                &format!("{shape:?} {bad}"),
                m.permute(&dx, dims),
                "permute",
            );
        }
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for p in permutations(n - 1) {
        for pos in 0..=p.len() {
            let mut q = p.clone();
            q.insert(pos, n - 1);
            out.push(q);
        }
    }
    out
}
