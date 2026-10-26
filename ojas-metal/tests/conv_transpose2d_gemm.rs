//! `conv_transpose2d_forward` on Metal against the CPU backend for a kernel
//! equal to its stride without padding (the SAM neck's 2 x 2 stride 2): a
//! permute, tessl's exact f32 GEMM and a layout pass. Every other
//! configuration is `Unsupported` and names the op and the configuration; a
//! malformed call is refused as the validator refuses it; a non-finite
//! operand is a deferred fault under `conv_transpose2d_forward`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{conv_transpose2d_forward_dims, Backend, ConvTranspose2dParams, OjasError, Tensor};
use ojas_metal::MetalBackend;

const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-4;

fn params(k: [usize; 2]) -> ConvTranspose2dParams {
    ConvTranspose2dParams {
        stride: k,
        ..ConvTranspose2dParams::new()
    }
}

/// CPU and Metal agree on `conv_transpose2d_forward` of `x` `[n, c, h, w]`
/// by a `[c, o, kh, kw]` weight, with or without a bias.
fn check(
    m: &MetalBackend,
    what: &str,
    x: &[usize],
    w: &[usize],
    with_bias: bool,
    p: ConvTranspose2dParams,
    seed: u64,
) {
    let c = cpu();
    let x = rand(x, seed, 1.0);
    let w = rand(w, seed + 1, 0.1);
    let bias = with_bias.then(|| rand(&[w.shape()[1]], seed + 2, 1.0));
    let want = ok("cpu", c.conv_transpose2d_forward(&x, &w, bias.as_ref(), p));
    let dev_bias = bias.as_ref().map(|b| up(m, b));
    let got = ok(
        "metal",
        m.conv_transpose2d_forward(&up(m, &x), &up(m, &w), dev_bias.as_ref(), p),
    );
    ok("sync", m.sync());
    same_tensor(what, &got, &want, ATOL, RTOL);
}

#[test]
fn the_neck_upsamplers_match_cpu_at_the_real_channel_counts() {
    let m = metal();
    let k2 = params([2, 2]);
    check(
        &m,
        "convT 1024->512 18",
        &[1, 1024, 18, 18],
        &[1024, 512, 2, 2],
        true,
        k2,
        1,
    );
    check(
        &m,
        "convT 512->256 36",
        &[1, 512, 36, 36],
        &[512, 256, 2, 2],
        true,
        k2,
        2,
    );
}

#[test]
fn the_neck_chain_with_gelu_matches_cpu() {
    let (m, c) = (metal(), cpu());
    let k2 = params([2, 2]);
    let x = rand(&[1, 1024, 12, 12], 5, 1.0);
    let (w1, b1) = (rand(&[1024, 512, 2, 2], 6, 0.1), rand(&[512], 7, 1.0));
    let (w2, b2) = (rand(&[512, 256, 2, 2], 8, 0.1), rand(&[256], 9, 1.0));
    let h = ok("cpu 1", c.conv_transpose2d_forward(&x, &w1, Some(&b1), k2));
    let h = ok("cpu gelu", c.gelu_erf_forward(&h));
    let want = ok("cpu 2", c.conv_transpose2d_forward(&h, &w2, Some(&b2), k2));
    let h = ok(
        "metal 1",
        m.conv_transpose2d_forward(&up(&m, &x), &up(&m, &w1), Some(&up(&m, &b1)), k2),
    );
    let h = ok("metal gelu", m.gelu_erf_forward(&h));
    let got = ok(
        "metal 2",
        m.conv_transpose2d_forward(&h, &up(&m, &w2), Some(&up(&m, &b2)), k2),
    );
    ok("sync", m.sync());
    same_tensor("neck chain", &got, &want, 1e-4, 1e-3);
}

#[test]
fn other_kernels_equal_to_the_stride_and_odd_shapes_match_cpu() {
    let m = metal();
    check(
        &m,
        "k2 no bias",
        &[2, 3, 5, 7],
        &[3, 4, 2, 2],
        false,
        params([2, 2]),
        10,
    );
    check(
        &m,
        "k3",
        &[1, 2, 4, 4],
        &[2, 5, 3, 3],
        true,
        params([3, 3]),
        11,
    );
    check(
        &m,
        "k2x3",
        &[2, 3, 3, 5],
        &[3, 2, 2, 3],
        true,
        params([2, 3]),
        12,
    );
    check(
        &m,
        "k1",
        &[1, 3, 4, 4],
        &[3, 6, 1, 1],
        true,
        params([1, 1]),
        13,
    );
    check(
        &m,
        "one pixel",
        &[1, 1, 1, 1],
        &[1, 1, 2, 2],
        true,
        params([2, 2]),
        14,
    );
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[1, 3, 4, 4], 20, 1.0);
    let w = rand(&[3, 4, 2, 2], 21, 0.1);
    let b = rand(&[4], 22, 1.0);
    let p = params([2, 2]);
    let want = ok("cpu", c.conv_transpose2d_forward(&x, &w, Some(&b), p));
    let got = ok(
        "metal",
        m.conv_transpose2d_forward(
            &offset_view(&m, &x, 3, 30),
            &offset_view(&m, &w, 5, 31),
            Some(&offset_view(&m, &b, 7, 32)),
            p,
        ),
    );
    ok("sync", m.sync());
    same_tensor("views", &got, &want, ATOL, RTOL);
}

#[test]
fn host_operands_are_refused_for_placement_after_the_shape_check() {
    let m = metal();
    let x = rand(&[1, 2, 3, 3], 40, 1.0);
    let w = rand(&[2, 3, 2, 2], 41, 0.1);
    let r = m.conv_transpose2d_forward(&x, &w, None, params([2, 2]));
    assert!(
        matches!(
            r,
            Err(OjasError::Placement {
                op: "conv_transpose2d_forward",
                ..
            })
        ),
        "{r:?}"
    );
}

#[test]
fn a_configuration_the_gemm_does_not_cover_is_unsupported_and_named() {
    let m = metal();
    let x = up(&m, &rand(&[1, 4, 6, 6], 50, 1.0));
    let w = up(&m, &rand(&[4, 4, 2, 2], 51, 1.0));
    let w3 = up(&m, &rand(&[4, 4, 3, 3], 52, 1.0));
    let cases: Vec<(&str, &Tensor, ConvTranspose2dParams)> = vec![
        ("stride 1", &w, ConvTranspose2dParams::new()),
        ("stride 3 over a 2x2 kernel", &w, params([3, 3])),
        ("stride 2 over a 3x3 kernel", &w3, params([2, 2])),
        (
            "padding",
            &w,
            ConvTranspose2dParams {
                padding: [1, 0],
                ..params([2, 2])
            },
        ),
        (
            "output padding",
            &w,
            ConvTranspose2dParams {
                output_padding: [1, 0],
                ..params([2, 2])
            },
        ),
    ];
    for (what, w, p) in cases {
        // The validator accepts it: only Metal declines.
        ok(what, conv_transpose2d_forward_dims(&x, w, None, p));
        match m.conv_transpose2d_forward(&x, w, None, p) {
            Err(OjasError::Unsupported { op, ref detail }) => {
                assert_eq!(op, "conv_transpose2d_forward", "{what}");
                assert!(
                    detail.contains("stride") && detail.contains("padding"),
                    "{what}: the detail names the configuration: {detail}"
                );
            }
            other => panic!("{what}: expected Unsupported, got {other:?}"),
        }
        ok("sync", m.sync());
    }
}

#[test]
fn a_malformed_call_is_refused_as_the_validator_refuses_it() {
    let m = metal();
    let x = rand(&[1, 4, 6, 6], 60, 1.0);
    let w = rand(&[4, 3, 2, 2], 61, 1.0);
    let p = params([2, 2]);
    let cases: Vec<(&str, Tensor, Tensor, Option<Tensor>, ConvTranspose2dParams)> = vec![
        ("rank 3 input", rand(&[4, 6, 6], 1, 1.0), w.clone(), None, p),
        (
            "weight in-channels",
            x.clone(),
            rand(&[5, 3, 2, 2], 2, 1.0),
            None,
            p,
        ),
        (
            "bias length",
            x.clone(),
            w.clone(),
            Some(rand(&[4], 3, 1.0)),
            p,
        ),
        ("zero stride", x.clone(), w.clone(), None, params([0, 2])),
        (
            "output_padding at the stride",
            x.clone(),
            w.clone(),
            None,
            ConvTranspose2dParams {
                output_padding: [2, 0],
                ..p
            },
        ),
        (
            "u32 input",
            host_u32(&[0; 4 * 6 * 6], &[1, 4, 6, 6]),
            w.clone(),
            None,
            p,
        ),
    ];
    for (what, x, w, bias, p) in cases {
        let want = conv_transpose2d_forward_dims(&x, &w, bias.as_ref(), p).expect_err(what);
        let host_err = m
            .conv_transpose2d_forward(&x, &w, bias.as_ref(), p)
            .expect_err(what);
        assert_eq!(
            format!("{host_err:?}"),
            format!("{want:?}"),
            "{what} (host)"
        );
        if x.dtype() == ojas_core::DType::F32 {
            let dev_bias = bias.as_ref().map(|b| up(&m, b));
            let r = m.conv_transpose2d_forward(&up(&m, &x), &up(&m, &w), dev_bias.as_ref(), p);
            let err = r.expect_err(what);
            assert_eq!(format!("{err:?}"), format!("{want:?}"), "{what} (device)");
        }
        ok("sync", m.sync());
    }
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    let p = params([2, 2]);
    let shapes: [&[usize]; 3] = [&[1, 2, 3, 3], &[2, 3, 2, 2], &[3]];
    for (which, name) in ["input", "weight", "bias"].into_iter().enumerate() {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut ops: Vec<Tensor> = shapes
                .iter()
                .enumerate()
                .map(|(i, s)| rand(s, 70 + i as u64, 1.0))
                .collect();
            let mut data = ok("host", ops[which].to_f32_vec());
            data[1] = bad;
            ops[which] = host(&data, shapes[which]);
            let dev: Vec<Tensor> = ops.iter().map(|t| up(&m, t)).collect();
            let r = m.conv_transpose2d_forward(&dev[0], &dev[1], Some(&dev[2]), p);
            deferred(
                &m,
                &format!("{bad} in {name}"),
                r,
                "conv_transpose2d_forward",
            );
        }
    }
}

/// The neck at its real first-stage size (72 x 72, 1024 -> 512 and
/// 512 -> 256 after the first upsample); the CPU reference takes a while, so
/// it runs on request.
#[test]
#[ignore = "slow CPU reference; run with --ignored"]
fn the_neck_upsamplers_at_72_match_cpu() {
    let m = metal();
    let k2 = params([2, 2]);
    check(
        &m,
        "convT 1024->512 72",
        &[1, 1024, 72, 72],
        &[1024, 512, 2, 2],
        true,
        k2,
        100,
    );
    check(
        &m,
        "convT 512->256 144",
        &[1, 512, 144, 144],
        &[512, 256, 2, 2],
        true,
        k2,
        101,
    );
}
