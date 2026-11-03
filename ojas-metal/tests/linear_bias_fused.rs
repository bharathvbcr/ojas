//! `linear_bias_forward` under the f32 lane with a `[out]` bias runs the
//! exact-f32 GEMM and then adds the bias in place (`ojas_bias_rows`), where
//! it used to run `linear_forward` and then `broadcast_add_forward`. That
//! saves the output scan and a second output buffer, and it must not move a
//! bit: SpikeTrack's memory attention runs its projections through this op,
//! and its parity gates compare outputs byte for byte.
//!
//! So every case holds the fused op to the two composed ops, bit for bit, on
//! the same device and inputs: SpikeTrack's projection shapes
//! (`[36400,256] x [256,256]`, the FFN's `[5184,256] x [2048,256]`), ragged
//! row and column counts, a batched input and a view at an unaligned offset.
//! It also checks what ran, and that an overflowing sum is the op's own
//! deferred fault.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::Backend;
use ojas_metal::MetalBackend;

const FUSED: &str = "ojas_bias_rows";
const BCAST: &str = "ojas_bcast_binary";

fn assert_same_bits(what: &str, got: &ojas_core::Tensor, want: &ojas_core::Tensor) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    assert_eq!(
        bits(&down(got)),
        bits(&down(want)),
        "{what}: fused output is not bit-identical to linear_forward + broadcast_add_forward"
    );
}

fn case(m: &MetalBackend, x_shape: &[usize], out: usize, seed: u64) {
    let kin = *x_shape.last().expect("rank >= 1");
    let x = up(m, &rand(x_shape, seed, 1.0));
    let w = up(m, &rand(&[out, kin], seed + 1, 0.1));
    let b = up(m, &rand(&[out], seed + 2, 1.0));
    let (fused0, bcast0) = (m.kernel_dispatches(FUSED), m.kernel_dispatches(BCAST));
    let got = ok("fused", m.linear_bias_forward(&x, &w, &b));
    ok("sync", m.sync());
    assert_eq!(
        (
            m.kernel_dispatches(FUSED) - fused0,
            m.kernel_dispatches(BCAST) - bcast0
        ),
        (1, 0),
        "{x_shape:?} x [{out},{kin}]: ({FUSED}, {BCAST}) dispatches"
    );
    let product = ok("linear", m.linear_forward(&x, &w));
    let want = ok("bcast", m.broadcast_add_forward(&product, &b));
    ok("sync", m.sync());
    assert_same_bits(&format!("{x_shape:?} x [{out},{kin}]"), &got, &want);
}

#[test]
fn fused_f32_linear_bias_equals_the_composed_ops_bit_for_bit() {
    let m = metal();
    case(&m, &[36400, 256], 256, 1);
    case(&m, &[5184, 256], 2048, 2);
    case(&m, &[5184, 2048], 256, 3);
    case(&m, &[1, 5184, 256], 256, 4);
    case(&m, &[2, 37, 19], 33, 5);
    case(&m, &[1, 1], 1, 6);
    case(&m, &[65, 7], 3, 7);
}

#[test]
fn fused_f32_linear_bias_reads_an_unaligned_input_view() {
    let m = metal();
    let x = offset_view(&m, &rand(&[33, 24], 11, 1.0), 3, 12);
    let w = up(&m, &rand(&[40, 24], 13, 0.1));
    let b = up(&m, &rand(&[40], 14, 1.0));
    let got = ok("fused", m.linear_bias_forward(&x, &w, &b));
    let want = ok(
        "bcast",
        m.broadcast_add_forward(&ok("linear", m.linear_forward(&x, &w)), &b),
    );
    ok("sync", m.sync());
    assert_same_bits("offset view", &got, &want);
}

#[test]
fn an_overflowing_fused_sum_is_the_ops_deferred_fault() {
    let m = metal();
    // x @ w^T is 3e38 + 3e38 in column 0: the product overflows, so the sum
    // with any finite bias does too.
    let x = up(&m, &host(&[3.0e38, 3.0e38], &[1, 2]));
    let w = up(&m, &host(&[1.0, 1.0, 0.0, 0.0], &[2, 2]));
    let b = up(&m, &host(&[0.5, 0.5], &[2]));
    deferred(
        &m,
        "overflowing product",
        m.linear_bias_forward(&x, &w, &b),
        "linear_bias_forward",
    );
    // A finite product that the bias pushes past f32::MAX.
    let x = up(&m, &host(&[3.0e38, 0.0], &[1, 2]));
    let w = up(&m, &host(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let b = up(&m, &host(&[3.0e38, 0.0], &[2]));
    deferred(
        &m,
        "overflowing bias add",
        m.linear_bias_forward(&x, &w, &b),
        "linear_bias_forward",
    );
}
