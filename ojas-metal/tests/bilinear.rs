//! `interpolate_bilinear_forward` on Metal (`ojas_bilinear`) against the CPU
//! backend, both `align_corners` settings: SAM's mask upsampling
//! (`[1, 32, 288, 288]` to 1152 x 1152 and 1008 x 1008), a video frame size
//! (288 to 1080 x 1920), down-sampling, identity and degenerate sizes.
//! Each thread forms its taps from the host's f32 axis scales with the CPU's
//! operations, so parity is bitwise; there are no axis tables to charge (the
//! CPU charges two to the budget), and the only allocation is the output.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{interpolate_bilinear_forward_dims, Backend, Tensor};
use ojas_metal::MetalBackend;

/// Bitwise in practice (`-ffp-contract=off`, the CPU's expressions); the
/// bound only keeps a compiler change from becoming a spurious failure.
const ATOL: f32 = 1e-6;
const RTOL: f32 = 1e-6;

fn check(m: &MetalBackend, what: &str, x: &Tensor, out: [usize; 2]) {
    let c = cpu();
    let dx = up(m, x);
    for align in [false, true] {
        let tag = format!("{what} align_corners {align}");
        let want = ok("cpu", c.interpolate_bilinear_forward(x, out, align));
        let got = ok("metal", m.interpolate_bilinear_forward(&dx, out, align));
        ok("sync", m.sync());
        same_tensor(&tag, &got, &want, ATOL, RTOL);
    }
}

#[test]
fn sam_mask_upsampling_matches_cpu() {
    let m = metal();
    let x = rand(&[1, 32, 288, 288], 1, 2.0);
    check(&m, "288 -> 1152", &x, [1152, 1152]);
    check(&m, "288 -> 1008", &x, [1008, 1008]);
}

#[test]
fn a_video_frame_size_matches_cpu() {
    let m = metal();
    let x = rand(&[1, 32, 288, 288], 2, 2.0);
    check(&m, "288 -> 1080x1920", &x, [1080, 1920]);
}

#[test]
fn downsampling_matches_cpu() {
    let m = metal();
    check(
        &m,
        "1152 -> 288",
        &rand(&[1, 8, 1152, 1152], 3, 2.0),
        [288, 288],
    );
    check(
        &m,
        "1080x1920 -> 288x512",
        &rand(&[1, 4, 1080, 1920], 4, 2.0),
        [288, 512],
    );
    check(
        &m,
        "odd 97x61 -> 33x20",
        &rand(&[2, 3, 97, 61], 5, 2.0),
        [33, 20],
    );
}

#[test]
fn small_odd_identity_and_degenerate_sizes_match_cpu() {
    let m = metal();
    check(&m, "1x1 -> 5x7", &rand(&[1, 3, 1, 1], 10, 1.0), [5, 7]);
    check(&m, "5x7 -> 5x7", &rand(&[2, 3, 5, 7], 11, 1.0), [5, 7]);
    check(&m, "5x7 -> 1x1", &rand(&[1, 2, 5, 7], 12, 1.0), [1, 1]);
    check(&m, "5x7 -> 1x9", &rand(&[1, 2, 5, 7], 13, 1.0), [1, 9]);
    check(&m, "5x7 -> 9x1", &rand(&[1, 2, 5, 7], 14, 1.0), [9, 1]);
    check(
        &m,
        "3x3 -> 1000x1000",
        &rand(&[1, 1, 3, 3], 15, 1.0),
        [1000, 1000],
    );
    check(
        &m,
        "17x13 -> 18x14",
        &rand(&[3, 2, 17, 13], 16, 1.0),
        [18, 14],
    );
    check(
        &m,
        "2x2 -> 3x3",
        &host(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]),
        [3, 3],
    );
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[2, 3, 6, 5], 20, 2.0);
    for align in [false, true] {
        let want = ok("cpu", c.interpolate_bilinear_forward(&x, [11, 9], align));
        let got = ok(
            "metal",
            m.interpolate_bilinear_forward(&offset_view(&m, &x, 3, 21), [11, 9], align),
        );
        ok("sync", m.sync());
        same_tensor(&format!("views align {align}"), &got, &want, ATOL, RTOL);
    }
}

#[test]
fn a_malformed_call_is_refused_as_the_validator_refuses_it() {
    let m = metal();
    let x = rand(&[1, 2, 4, 4], 30, 1.0);
    let cases: Vec<(&str, Tensor, [usize; 2])> = vec![
        ("rank 3", rand(&[2, 4, 4], 1, 1.0), [8, 8]),
        ("zero target height", x.clone(), [0, 8]),
        ("zero target width", x.clone(), [8, 0]),
        ("u32 input", host_u32(&[0; 32], &[1, 2, 4, 4]), [8, 8]),
        ("empty input", host(&[], &[1, 2, 0, 4]), [8, 8]),
    ];
    for (what, x, out) in cases {
        for align in [false, true] {
            let want = interpolate_bilinear_forward_dims(&x, out).expect_err(what);
            let err = m
                .interpolate_bilinear_forward(&x, out, align)
                .expect_err(what);
            assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (host)");
            if x.dtype() == ojas_core::DType::F32 && x.shape().iter().all(|&d| d > 0) {
                let err = m
                    .interpolate_bilinear_forward(&up(&m, &x), out, align)
                    .expect_err(what);
                assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (device)");
            }
        }
        ok("sync", m.sync());
    }
}

#[test]
fn host_operands_are_refused_for_placement_after_the_shape_check() {
    let m = metal();
    let r = m.interpolate_bilinear_forward(&rand(&[1, 1, 3, 3], 40, 1.0), [6, 6], false);
    assert!(
        matches!(
            r,
            Err(ojas_core::OjasError::Placement {
                op: "interpolate_bilinear_forward",
                ..
            })
        ),
        "{r:?}"
    );
}

#[test]
fn a_non_finite_input_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for align in [false, true] {
            let mut data = values(2 * 5 * 5, 50, 1.0);
            data[7] = bad;
            let x = up(&m, &host(&data, &[1, 2, 5, 5]));
            let r = m.interpolate_bilinear_forward(&x, [9, 9], align);
            deferred(
                &m,
                &format!("{bad} align {align}"),
                r,
                "interpolate_bilinear_forward",
            );
        }
    }
}

#[test]
fn only_the_output_is_charged() {
    // The CPU charges its two axis tables; the kernel forms taps itself, so
    // a budget that holds the input and the output exactly is enough.
    let x = rand(&[1, 1, 64, 64], 60, 1.0);
    let out = [128usize, 96];
    let bytes = (64 * 64 + out[0] * out[1]) * 4;
    let tight = ojas_metal::MetalBackend::new(ojas_core::Budget::new(bytes as u64))
        .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
    let dx = ok("upload", tight.upload(&x));
    let y = ok("tight", tight.interpolate_bilinear_forward(&dx, out, false));
    ok("sync", tight.sync());
    assert_eq!(y.shape(), [1, 1, 128, 96]);
}
