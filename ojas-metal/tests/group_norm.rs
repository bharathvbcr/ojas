//! `group_norm_forward` on Metal (`ojas_gn_sum`, `ojas_gn_sq`,
//! `ojas_gn_apply`: f32 two-pass statistics over many threadgroups per
//! `(image, group)` row) against the CPU backend's `f64` statistics: odd
//! shapes (groups 1, C and between; spatial sizes that are not multiples of
//! the threadgroup), SAM's pixel decoder at `[1, 256, 144, 144]` and
//! `[1, 256, 288, 288]` with 8 groups, every refusal as the validator
//! refuses it, non-finite faults under the op name, offset views, the
//! budget, and the dispatch count.
//!
//! The two backends differ by f32 summation error in the statistics (a row
//! sums up to 2.65 million values), so parity is a tolerance, not bits; the
//! kernel itself repeats bit for bit.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{group_norm_forward_dims, Backend, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

const EPS: f32 = 1e-5;
const ATOL: f32 = 2e-5;
const RTOL: f32 = 1e-4;

struct Case {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    groups: usize,
}

fn operands(c: &Case, seed: u64, mean: f32) -> [Tensor; 3] {
    let x: Vec<f32> = values(c.n * c.c * c.h * c.w, seed, 3.0)
        .into_iter()
        .map(|v| v + mean)
        .collect();
    [
        host(&x, &[c.n, c.c, c.h, c.w]),
        rand(&[c.c], seed + 1, 2.0),
        rand(&[c.c], seed + 2, 1.0),
    ]
}

fn check(m: &MetalBackend, c: &Case, eps: f32, seed: u64) {
    let tag = format!(
        "[{},{},{},{}] groups {} eps {eps:e}",
        c.n, c.c, c.h, c.w, c.groups
    );
    let [x, w, b] = operands(c, seed, 0.5);
    let want = ok("cpu", cpu().group_norm_forward(&x, &w, &b, c.groups, eps));
    let (dx, dw, db) = (up(m, &x), up(m, &w), up(m, &b));
    let got = ok("metal", m.group_norm_forward(&dx, &dw, &db, c.groups, eps));
    ok("sync", m.sync());
    same_tensor(&tag, &got, &want, ATOL, RTOL);
    let again = ok("again", m.group_norm_forward(&dx, &dw, &db, c.groups, eps));
    ok("sync", m.sync());
    assert_eq!(down(&again), down(&got), "{tag}: not repeatable");
}

#[test]
fn odd_shapes_and_every_group_count_match_cpu() {
    let m = metal();
    for (i, (n, c, h, w, g)) in [
        (2usize, 6usize, 5usize, 7usize, 3usize),
        // groups 1: the whole image; groups C: instance norm.
        (1, 8, 3, 3, 1),
        (1, 8, 3, 3, 8),
        (3, 12, 1, 1, 4),
        // Spatial 258: not a multiple of the 256-thread group.
        (1, 4, 2, 129, 2),
        (2, 32, 17, 19, 8),
        // A group of one value: variance 0, so eps alone.
        (1, 3, 1, 1, 3),
        // Rows long enough to be cut into several chunks (width 8192 and 65536).
        (1, 16, 64, 64, 8),
        (2, 16, 128, 128, 4),
    ]
    .into_iter()
    .enumerate()
    {
        check(
            &m,
            &Case {
                n,
                c,
                h,
                w,
                groups: g,
            },
            EPS,
            10 + 3 * i as u64,
        );
    }
    check(
        &m,
        &Case {
            n: 2,
            c: 8,
            h: 9,
            w: 11,
            groups: 4,
        },
        1e-3,
        50,
    );
}

#[test]
fn the_pixel_decoder_sizes_match_cpu() {
    let m = metal();
    for (i, hw) in [144usize, 288].into_iter().enumerate() {
        check(
            &m,
            &Case {
                n: 1,
                c: 256,
                h: hw,
                w: hw,
                groups: 8,
            },
            EPS,
            100 + i as u64,
        );
    }
}

#[test]
fn a_row_with_a_mean_far_from_zero_keeps_the_cpu_answer_to_f32_statistics() {
    // Mean 20 over 2 million values: the f32 sums carry about 1e-7 relative
    // error each, so the output agrees to a few 1e-6, within the bound.
    let (m, c) = (metal(), cpu());
    let case = Case {
        n: 1,
        c: 64,
        h: 128,
        w: 128,
        groups: 8,
    };
    let [x, w, b] = operands(&case, 70, 20.0);
    let want = ok("cpu", c.group_norm_forward(&x, &w, &b, 8, EPS));
    let got = ok(
        "metal",
        m.group_norm_forward(&up(&m, &x), &up(&m, &w), &up(&m, &b), 8, EPS),
    );
    ok("sync", m.sync());
    same_tensor("mean 20", &got, &want, 1e-4, 1e-4);
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let case = Case {
        n: 2,
        c: 6,
        h: 5,
        w: 5,
        groups: 3,
    };
    let [x, w, b] = operands(&case, 80, 0.0);
    let want = ok("cpu", c.group_norm_forward(&x, &w, &b, 3, EPS));
    let got = ok(
        "metal",
        m.group_norm_forward(
            &offset_view(&m, &x, 3, 81),
            &offset_view(&m, &w, 5, 82),
            &offset_view(&m, &b, 7, 83),
            3,
            EPS,
        ),
    );
    ok("sync", m.sync());
    same_tensor("views", &got, &want, ATOL, RTOL);
}

#[test]
fn a_malformed_call_is_refused_as_the_validator_refuses_it() {
    let m = metal();
    let x = rand(&[2, 6, 4, 4], 90, 1.0);
    let w = rand(&[6], 91, 1.0);
    let b = rand(&[6], 92, 1.0);
    let cases: Vec<(&str, Tensor, Tensor, Tensor, usize, f32)> = vec![
        (
            "rank 3",
            rand(&[2, 6, 16], 1, 1.0),
            w.clone(),
            b.clone(),
            3,
            EPS,
        ),
        ("groups 0", x.clone(), w.clone(), b.clone(), 0, EPS),
        (
            "groups not dividing",
            x.clone(),
            w.clone(),
            b.clone(),
            4,
            EPS,
        ),
        (
            "weight shape",
            x.clone(),
            rand(&[5], 2, 1.0),
            b.clone(),
            3,
            EPS,
        ),
        (
            "bias shape",
            x.clone(),
            w.clone(),
            rand(&[6, 1], 3, 1.0),
            3,
            EPS,
        ),
        ("eps 0", x.clone(), w.clone(), b.clone(), 3, 0.0),
        ("eps negative", x.clone(), w.clone(), b.clone(), 3, -1e-5),
        ("eps NaN", x.clone(), w.clone(), b.clone(), 3, f32::NAN),
        (
            "eps infinite",
            x.clone(),
            w.clone(),
            b.clone(),
            3,
            f32::INFINITY,
        ),
        (
            "u32 input",
            host_u32(&[0; 192], &[2, 6, 4, 4]),
            w.clone(),
            b.clone(),
            3,
            EPS,
        ),
        (
            "empty",
            host(&[], &[2, 6, 0, 4]),
            w.clone(),
            b.clone(),
            3,
            EPS,
        ),
    ];
    for (what, x, w, b, groups, eps) in cases {
        let want = group_norm_forward_dims(&x, &w, &b, groups, eps).expect_err(what);
        let err = m
            .group_norm_forward(&x, &w, &b, groups, eps)
            .expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (host)");
        let device_ok = x.dtype() == ojas_core::DType::F32 && !x.shape().contains(&0);
        if device_ok && w.shape().iter().all(|&d| d > 0) {
            let r = m.group_norm_forward(&up(&m, &x), &up(&m, &w), &up(&m, &b), groups, eps);
            let err = r.expect_err(what);
            assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (device)");
        }
        ok("sync", m.sync());
    }
    // A valid call on host operands is refused for placement, after the shape.
    let r = m.group_norm_forward(&x, &w, &b, 3, EPS);
    assert!(
        matches!(
            r,
            Err(OjasError::Placement {
                op: "group_norm_forward",
                ..
            })
        ),
        "{r:?}"
    );
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    let case = Case {
        n: 1,
        c: 4,
        h: 6,
        w: 6,
        groups: 2,
    };
    let shapes: [&[usize]; 3] = [&[1, 4, 6, 6], &[4], &[4]];
    for (which, name) in ["input", "weight", "bias"].into_iter().enumerate() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut ops = operands(&case, 95, 0.0);
            let mut data = ok("host", ops[which].to_f32_vec());
            data[3] = bad;
            ops[which] = host(&data, shapes[which]);
            let dev: Vec<Tensor> = ops.iter().map(|t| up(&m, t)).collect();
            let r = m.group_norm_forward(&dev[0], &dev[1], &dev[2], 2, EPS);
            deferred(&m, &format!("{bad} in {name}"), r, "group_norm_forward");
        }
    }
    // Finite operands whose result overflows: the output scan catches it.
    let [x, _, b] = operands(&case, 96, 0.0);
    let w = host(&[3.0e38; 4], &[4]);
    let r = m.group_norm_forward(&up(&m, &x), &up(&m, &w), &up(&m, &b), 2, EPS);
    deferred(&m, "overflow", r, "group_norm_forward");
}

/// The partial sums are charged: a budget that holds the operands, the
/// output and all but one value of the partials is refused with nothing left
/// charged, and one with room for them runs.
#[test]
fn the_partial_sums_are_charged_to_the_budget() {
    // [1, 16, 64, 64], 8 groups: 8 rows of 8192 values, 8 chunks each, so
    // 2 * 8 * 8 = 128 partial values (512 bytes).
    let case = Case {
        n: 1,
        c: 16,
        h: 64,
        w: 64,
        groups: 8,
    };
    let [x, w, b] = operands(&case, 97, 0.0);
    let inputs = (x.num_elements().unwrap() + 2 * 16) * 4;
    let output = x.num_elements().unwrap() * 4;
    let partials = 128 * 4;
    for (room, runs) in [
        (inputs + output + partials - 4, false),
        (inputs + output + partials, true),
    ] {
        let tight = MetalBackend::new(Budget::new(room as u64))
            .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
        let (dx, dw, db) = (up(&tight, &x), up(&tight, &w), up(&tight, &b));
        let baseline = ok("live", tight.budget().live_bytes());
        let r = tight.group_norm_forward(&dx, &dw, &db, 8, EPS);
        ok("sync", tight.sync());
        if runs {
            let y = ok("fits", r);
            drop(y);
        } else {
            assert!(
                matches!(r, Err(OjasError::CapacityExceeded { .. })),
                "one value short: {r:?}"
            );
        }
        assert_eq!(
            ok("live", tight.budget().live_bytes()),
            baseline,
            "nothing is left charged (room {room})"
        );
    }
}

/// Three kernels and the same operand and output scans as `layer_norm_forward`
/// (one tessl kernel): two dispatches more.
#[test]
fn group_norm_is_three_kernels() {
    let m = metal();
    let case = Case {
        n: 1,
        c: 16,
        h: 32,
        w: 32,
        groups: 8,
    };
    let [x, w, b] = operands(&case, 98, 0.0);
    let (dx, dw, db) = (up(&m, &x), up(&m, &w), up(&m, &b));
    let count = |f: &dyn Fn()| {
        ok("sync", m.sync());
        let before = m.dispatches();
        f();
        ok("sync", m.sync());
        m.dispatches() - before
    };
    let gn = count(&|| drop(ok("gn", m.group_norm_forward(&dx, &dw, &db, 8, EPS))));
    // LayerNorm over rows of 16 with a [16] affine, the same three operands.
    // Fresh buffers, so no scan of the operands above is skipped.
    let lx = up(&m, &rand(&[1024, 16], 99, 1.0));
    let (lw, lb) = (up(&m, &w), up(&m, &b));
    let ln = count(&|| drop(ok("ln", m.layer_norm_forward(&lx, &lw, &lb, EPS))));
    assert_eq!(gn, ln + 2, "group norm {gn}, layer norm {ln}");
}
