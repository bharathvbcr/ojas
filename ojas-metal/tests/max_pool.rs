//! `max_pool2d_forward` and `max_pool2d_backward` on Metal (`ojas_max_pool2d_fwd`
//! and `ojas_max_pool2d_bwd`) against the CPU backend and against torch.
//!
//! Max pooling does no arithmetic on the values it passes, and the backward
//! adds one `grad_output` per window an input wins, in ascending window order
//! (a gather, no atomics). So parity is bitwise, not toleranced: SAM's neck
//! shape, odd and floor-mode shapes, overlapping and strided windows, and the
//! tie rule (the first maximum of a window in `(kh, kw)` order wins, `+0.0`
//! and `-0.0` tying) each match the CPU bit for bit. The torch golden
//! (`ojas-oracle/fixtures/max_pool/golden.txt`, written by
//! `ojas-oracle/python/max_pool_fixture.py`) holds both backends to torch's
//! bits on small cases, ties included.
//!
//! Every comparison also checks the kernel ran: the backend's per-kernel
//! dispatch count rises by one for the op.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{
    max_pool2d_backward_dims, max_pool2d_forward_dims, Backend, DType, OjasError, Tensor,
};
use ojas_metal::MetalBackend;

const FWD: &str = "ojas_max_pool2d_fwd";
const BWD: &str = "ojas_max_pool2d_bwd";

/// Forward and backward of one case on the CPU and on Metal, bit for bit.
fn check(m: &MetalBackend, what: &str, x: &Tensor, k: [usize; 2], s: [usize; 2]) {
    let c = cpu();
    let dx = up(m, x);

    let want = ok("cpu forward", c.max_pool2d_forward(x, k, s));
    let before = m.kernel_dispatches(FWD);
    let got = ok("metal forward", m.max_pool2d_forward(&dx, k, s));
    ok("sync", m.sync());
    assert_eq!(
        m.kernel_dispatches(FWD),
        before + 1,
        "{what}: forward kernel"
    );
    same_bits(&format!("{what} forward"), &got, &want);

    let dy = rand(want.shape(), 99, 1.0);
    let want_dx = ok("cpu backward", c.max_pool2d_backward(x, &dy, k, s));
    let before = m.kernel_dispatches(BWD);
    let got_dx = ok(
        "metal backward",
        m.max_pool2d_backward(&dx, &up(m, &dy), k, s),
    );
    ok("sync", m.sync());
    assert_eq!(
        m.kernel_dispatches(BWD),
        before + 1,
        "{what}: backward kernel"
    );
    same_bits(&format!("{what} backward"), &got_dx, &want_dx);
}

/// Values drawn from {0, 1, 2}: most windows tie.
fn ties(shape: &[usize], seed: u64) -> Tensor {
    let n = shape.iter().product();
    let data: Vec<f32> = values(n, seed, 1.0)
        .iter()
        .map(|v| (v * 1.5).round() + 1.0)
        .collect();
    host(&data, shape)
}

#[test]
fn sam_neck_shape_matches_cpu() {
    let m = metal();
    // The 2x2 stride-2 pool over a 256-channel 288 x 288 feature map.
    check(
        &m,
        "neck [1,256,288,288]",
        &rand(&[1, 256, 288, 288], 1, 2.0),
        [2, 2],
        [2, 2],
    );
}

#[test]
fn odd_floor_overlapping_and_strided_shapes_match_cpu() {
    let m = metal();
    let cases: [(&str, [usize; 4], [usize; 2], [usize; 2]); 12] = [
        (
            "odd 7x9 k2 s2 (floor remainder)",
            [2, 3, 7, 9],
            [2, 2],
            [2, 2],
        ),
        ("odd 97x61 k2 s2", [2, 3, 97, 61], [2, 2], [2, 2]),
        ("k3 s1 overlapping", [1, 4, 9, 11], [3, 3], [1, 1]),
        ("k3 s2 overlapping", [2, 2, 33, 31], [3, 3], [2, 2]),
        ("rect k2x3 s1x2", [1, 3, 5, 8], [2, 3], [1, 2]),
        ("rect k3x2 s2x1", [1, 3, 8, 5], [3, 2], [2, 1]),
        (
            "stride past the kernel k2 s3",
            [1, 2, 11, 10],
            [2, 2],
            [3, 3],
        ),
        ("1x1 kernel", [2, 2, 5, 6], [1, 1], [1, 1]),
        ("1x1 kernel stride 2", [1, 2, 6, 7], [1, 1], [2, 2]),
        ("window is the image", [3, 2, 4, 5], [4, 5], [1, 1]),
        ("single pixel image", [2, 3, 1, 1], [1, 1], [1, 1]),
        ("one row", [1, 2, 1, 9], [1, 3], [1, 2]),
    ];
    for (i, (what, shape, k, s)) in cases.into_iter().enumerate() {
        check(&m, what, &rand(&shape, 10 + i as u64, 3.0), k, s);
    }
}

#[test]
fn ties_go_to_the_first_maximum_as_on_the_cpu() {
    let m = metal();
    // Constant planes: every input ties, the window's first input wins.
    check(
        &m,
        "constant",
        &host(&[1.5; 2 * 3 * 8 * 8], &[2, 3, 8, 8]),
        [2, 2],
        [2, 2],
    );
    check(
        &m,
        "constant k3 s1",
        &host(&[-4.0; 2 * 7 * 7], &[1, 2, 7, 7]),
        [3, 3],
        [1, 1],
    );
    // Three distinct values: most windows tie, some do not.
    check(&m, "ints k2 s2", &ties(&[2, 4, 12, 12], 3), [2, 2], [2, 2]);
    check(&m, "ints k3 s1", &ties(&[1, 3, 10, 9], 4), [3, 3], [1, 1]);
    check(&m, "ints k3 s2", &ties(&[2, 2, 15, 13], 5), [3, 3], [2, 2]);
    check(
        &m,
        "ints k2x3 s1x2",
        &ties(&[1, 2, 7, 11], 6),
        [2, 3],
        [1, 2],
    );
}

#[test]
fn positive_and_negative_zero_tie_and_keep_the_winners_own_bits() {
    let m = metal();
    let n = 2 * 3 * 6 * 6;
    let data: Vec<f32> = ids(n, 7, 2)
        .iter()
        .map(|&b| if b == 0 { 0.0 } else { -0.0 })
        .collect();
    // `0.0 > -0.0` is false, so the first zero of a window wins whatever its
    // sign, and the forward writes that zero's bits.
    check(
        &m,
        "signed zeros k2 s2",
        &host(&data, &[2, 3, 6, 6]),
        [2, 2],
        [2, 2],
    );
    check(
        &m,
        "signed zeros k3 s1",
        &host(&data, &[2, 3, 6, 6]),
        [3, 3],
        [1, 1],
    );
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[2, 3, 7, 6], 20, 2.0);
    let dy = rand(&[2, 3, 3, 3], 21, 1.0);
    let want = ok("cpu", c.max_pool2d_forward(&x, [2, 2], [2, 2]));
    let got = ok(
        "metal",
        m.max_pool2d_forward(&offset_view(&m, &x, 3, 22), [2, 2], [2, 2]),
    );
    ok("sync", m.sync());
    same_bits("view forward", &got, &want);
    let want = ok("cpu", c.max_pool2d_backward(&x, &dy, [2, 2], [2, 2]));
    let got = ok(
        "metal",
        m.max_pool2d_backward(
            &offset_view(&m, &x, 3, 23),
            &offset_view(&m, &dy, 5, 24),
            [2, 2],
            [2, 2],
        ),
    );
    ok("sync", m.sync());
    same_bits("view backward", &got, &want);
}

/// Torch's `F.max_pool2d` and its autograd gradient, from
/// `ojas-oracle/fixtures/max_pool/golden.txt`: both backends match torch's
/// bits, ties and signed zeros included.
#[test]
fn both_backends_match_the_torch_golden_bit_for_bit() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../ojas-oracle/fixtures/max_pool/golden.txt"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("ojas-maxpool-golden-v1"));
    assert!(lines.next().is_some_and(|l| l.starts_with("torch ")));

    let (m, c) = (metal(), cpu());
    let mut cases = 0;
    while let Some(head) = lines.next() {
        let f: Vec<&str> = head.split_whitespace().collect();
        assert_eq!(f[0], "case", "{head}");
        let name = f[1];
        let n: Vec<usize> = f[2..].iter().map(|v| v.parse().expect("integer")).collect();
        let (k, s) = ([n[4], n[5]], [n[6], n[7]]);
        let mut tensor = |label: &str| -> Tensor {
            let line = lines.next().expect("tensor line");
            let f: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(&f[..3], ["t", label, "f32"], "{name}: {line}");
            let dims: Vec<usize> = f[3].split(',').map(|v| v.parse().expect("dim")).collect();
            let data: Vec<f32> = f[4..]
                .iter()
                .map(|h| f32::from_bits(u32::from_str_radix(h, 16).expect("hex")))
                .collect();
            host(&data, &dims)
        };
        let (x, gy, y, dx) = (tensor("x"), tensor("gy"), tensor("y"), tensor("dx"));
        assert_eq!(lines.next(), Some("end"), "{name}");

        let cpu_y = ok("cpu forward", c.max_pool2d_forward(&x, k, s));
        assert_eq!(
            bits(&down_host(&cpu_y)),
            bits(&down_host(&y)),
            "{name}: cpu forward"
        );
        let cpu_dx = ok("cpu backward", c.max_pool2d_backward(&x, &gy, k, s));
        assert_eq!(
            bits(&down_host(&cpu_dx)),
            bits(&down_host(&dx)),
            "{name}: cpu backward"
        );

        let got = ok("metal forward", m.max_pool2d_forward(&up(&m, &x), k, s));
        ok("sync", m.sync());
        same_bits(&format!("{name} forward vs torch"), &got, &y);
        let got = ok(
            "metal backward",
            m.max_pool2d_backward(&up(&m, &x), &up(&m, &gy), k, s),
        );
        ok("sync", m.sync());
        same_bits(&format!("{name} backward vs torch"), &got, &dx);
        cases += 1;
    }
    assert!(cases >= 10, "only {cases} golden cases");
}

fn down_host(t: &Tensor) -> Vec<f32> {
    ok("host values", t.to_f32_vec())
}

#[test]
fn a_malformed_call_is_refused_as_the_validator_refuses_it() {
    let m = metal();
    let x = rand(&[1, 2, 6, 6], 30, 1.0);
    let cases: Vec<(&str, Tensor, [usize; 2], [usize; 2])> = vec![
        ("rank 3", rand(&[2, 6, 6], 1, 1.0), [2, 2], [2, 2]),
        ("zero kernel", x.clone(), [0, 2], [2, 2]),
        ("zero stride", x.clone(), [2, 2], [2, 0]),
        ("kernel taller than the image", x.clone(), [7, 2], [1, 1]),
        ("kernel wider than the image", x.clone(), [2, 7], [1, 1]),
        (
            "u32 input",
            host_u32(&[0; 72], &[1, 2, 6, 6]),
            [2, 2],
            [2, 2],
        ),
    ];
    for (what, x, k, s) in cases {
        let want = max_pool2d_forward_dims(&x, k, s).expect_err(what);
        let err = m.max_pool2d_forward(&x, k, s).expect_err(what);
        assert_eq!(
            format!("{err:?}"),
            format!("{want:?}"),
            "{what} forward (host)"
        );
        let dy = rand(&[1, 2, 3, 3], 31, 1.0);
        let want = max_pool2d_backward_dims(&x, &dy, k, s).expect_err(what);
        let err = m.max_pool2d_backward(&x, &dy, k, s).expect_err(what);
        assert_eq!(
            format!("{err:?}"),
            format!("{want:?}"),
            "{what} backward (host)"
        );
        if x.dtype() == DType::F32 && x.shape().iter().all(|&d| d > 0) {
            let err = m.max_pool2d_forward(&up(&m, &x), k, s).expect_err(what);
            let want = max_pool2d_forward_dims(&x, k, s).expect_err(what);
            assert_eq!(
                format!("{err:?}"),
                format!("{want:?}"),
                "{what} forward (device)"
            );
        }
    }
    // A gradient of the wrong shape or dtype.
    for (what, dy) in [
        ("wrong shape", rand(&[1, 2, 3, 4], 32, 1.0)),
        ("u32 gradient", host_u32(&[0; 18], &[1, 2, 3, 3])),
    ] {
        let want = max_pool2d_backward_dims(&x, &dy, [2, 2], [2, 2]).expect_err(what);
        let err = m
            .max_pool2d_backward(&up(&m, &x), &dy, [2, 2], [2, 2])
            .expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what}");
    }
    ok("sync", m.sync());
}

#[test]
fn host_operands_are_refused_for_placement_after_the_shape_check() {
    let m = metal();
    let x = rand(&[1, 1, 4, 4], 40, 1.0);
    let dy = rand(&[1, 1, 2, 2], 41, 1.0);
    let r = m.max_pool2d_forward(&x, [2, 2], [2, 2]);
    assert!(
        matches!(
            r,
            Err(OjasError::Placement {
                op: "max_pool2d_forward",
                ..
            })
        ),
        "{r:?}"
    );
    // A device input beside a host gradient is a placement error too.
    let r = m.max_pool2d_backward(&up(&m, &x), &dy, [2, 2], [2, 2]);
    assert!(
        matches!(
            r,
            Err(OjasError::Placement {
                op: "max_pool2d_backward",
                ..
            })
        ),
        "{r:?}"
    );
}

/// The CPU refuses a non-finite value wherever it sits, including the
/// floor-mode remainder no window reads; Metal records the fault.
#[test]
fn a_non_finite_input_is_a_deferred_fault_under_the_op_name() {
    let (m, c) = (metal(), cpu());
    let (h, w) = (7usize, 9usize);
    // (row, col) of the planted value: inside a window, and in the last row and
    // column that a 2x2 stride-2 pooling of 7 x 9 never reads.
    for (row, col) in [(0usize, 0usize), (3, 4), (6, 8), (6, 0), (0, 8)] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut data = values(2 * h * w, 50, 1.0);
            data[h * w + row * w + col] = bad;
            let x = host(&data, &[1, 2, h, w]);
            let cpu_result = c.max_pool2d_forward(&x, [2, 2], [2, 2]);
            assert!(
                matches!(
                    cpu_result,
                    Err(OjasError::NonFinite {
                        op: "max_pool2d_forward"
                    })
                ),
                "the CPU refuses {bad} at ({row}, {col}): {cpu_result:?}"
            );
            let r = m.max_pool2d_forward(&up(&m, &x), [2, 2], [2, 2]);
            deferred(
                &m,
                &format!("{bad} at ({row}, {col})"),
                r,
                "max_pool2d_forward",
            );
        }
    }
}

#[test]
fn a_non_finite_backward_operand_or_result_is_a_deferred_fault() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[1, 2, 7, 9], 60, 1.0);
    let dy = rand(&[1, 2, 3, 4], 61, 1.0);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut xs = ok("x", x.to_f32_vec());
        xs[7 * 9 + 6 * 9 + 8] = bad; // channel 1, a pixel no window reads
        let xb = host(&xs, x.shape());
        assert!(matches!(
            c.max_pool2d_backward(&xb, &dy, [2, 2], [2, 2]),
            Err(OjasError::NonFinite {
                op: "max_pool2d_backward"
            })
        ));
        let r = m.max_pool2d_backward(&up(&m, &xb), &up(&m, &dy), [2, 2], [2, 2]);
        deferred(&m, &format!("x {bad}"), r, "max_pool2d_backward");

        let mut ds = ok("dy", dy.to_f32_vec());
        ds[5] = bad;
        let db = host(&ds, dy.shape());
        assert!(matches!(
            c.max_pool2d_backward(&x, &db, [2, 2], [2, 2]),
            Err(OjasError::NonFinite {
                op: "max_pool2d_backward"
            })
        ));
        let r = m.max_pool2d_backward(&up(&m, &x), &up(&m, &db), [2, 2], [2, 2]);
        deferred(&m, &format!("dy {bad}"), r, "max_pool2d_backward");
    }
    // Finite operands whose gather sum overflows: the centre input of a 4x4
    // image wins all four 3x3 windows, and 4 * 3e38 is infinite in f32.
    let mut data = vec![0.0f32; 16];
    data[5] = 1.0;
    let big = host(&data, &[1, 1, 4, 4]);
    let dy = host(&[3.0e38; 4], &[1, 1, 2, 2]);
    assert!(matches!(
        c.max_pool2d_backward(&big, &dy, [3, 3], [1, 1]),
        Err(OjasError::NonFinite {
            op: "max_pool2d_backward"
        })
    ));
    let r = m.max_pool2d_backward(&up(&m, &big), &up(&m, &dy), [3, 3], [1, 1]);
    deferred(&m, "overflowing sum", r, "max_pool2d_backward");
}

#[test]
fn only_the_output_is_charged() {
    // No scratch: a budget holding the operands and the result exactly runs.
    let x = rand(&[1, 1, 64, 64], 70, 1.0);
    let dy = rand(&[1, 1, 32, 32], 71, 1.0);
    let forward = (64 * 64 + 32 * 32) * 4;
    let backward = (64 * 64 + 32 * 32 + 64 * 64) * 4;
    let tight = |bytes: usize| {
        MetalBackend::new(ojas_core::Budget::new(bytes as u64))
            .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"))
    };
    let t = tight(forward);
    let dx = ok("upload", t.upload(&x));
    let y = ok("tight forward", t.max_pool2d_forward(&dx, [2, 2], [2, 2]));
    ok("sync", t.sync());
    assert_eq!(y.shape(), [1, 1, 32, 32]);

    let t = tight(backward);
    let (dx, dd) = (ok("upload", t.upload(&x)), ok("upload", t.upload(&dy)));
    let g = ok(
        "tight backward",
        t.max_pool2d_backward(&dx, &dd, [2, 2], [2, 2]),
    );
    ok("sync", t.sync());
    assert_eq!(g.shape(), [1, 1, 64, 64]);

    // One byte short refuses, with the op's budget error and nothing run.
    let t = tight(forward - 4);
    let dx = ok("upload", t.upload(&x));
    let r = t.max_pool2d_forward(&dx, [2, 2], [2, 2]);
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
}
