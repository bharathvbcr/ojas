//! `rope_interleaved_forward` and `_backward` on Metal (`ojas_rope`, mode 2)
//! against the CPU backend: the SAM ViT shapes with the axial 2D tables
//! (windowed scale 1, global scale 24/72), small odd shapes, offset device
//! views, the round trip, and refusals that match the validator exactly.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{
    axial_rope_2d_tables, rope_interleaved_backward_dims, rope_interleaved_forward_dims, Backend,
    Budget, Tensor,
};

/// The kernel forms each product pair exactly as the CPU does
/// (`-ffp-contract=off`), so parity is bitwise; the bound only keeps a
/// future compiler change from turning into a spurious failure.
const ATOL: f32 = 1e-6;
const RTOL: f32 = 1e-6;

fn axial(head_dim: usize, h: usize, w: usize, scale: f32) -> (Tensor, Tensor) {
    ok(
        "axial tables",
        axial_rope_2d_tables(head_dim, h, w, 10_000.0, scale, &Budget::new(8 * GIB)),
    )
}

/// Both directions on both backends for `x` `[b, t, heads, dim]`.
fn check(m: &ojas_metal::MetalBackend, what: &str, x: &Tensor, cos: &Tensor, sin: &Tensor) {
    let c = cpu();
    let (dx, dc, ds) = (up(m, x), up(m, cos), up(m, sin));
    let want = ok("cpu fwd", c.rope_interleaved_forward(x, cos, sin));
    let got = ok("metal fwd", m.rope_interleaved_forward(&dx, &dc, &ds));
    same_tensor(&format!("{what} forward"), &got, &want, ATOL, RTOL);
    let want = ok("cpu bwd", c.rope_interleaved_backward(x, cos, sin));
    let got = ok("metal bwd", m.rope_interleaved_backward(&dx, &dc, &ds));
    same_tensor(&format!("{what} backward"), &got, &want, ATOL, RTOL);
    ok("sync", m.sync());
}

#[test]
fn sam_windowed_shape_matches_cpu() {
    let m = metal();
    // 9 windows of 24 x 24 tokens, 16 heads of 64, scale 1.
    let (cos, sin) = axial(64, 24, 24, 1.0);
    let x = rand(&[9, 576, 16, 64], 1, 2.0);
    check(&m, "[9,576,16,64]", &x, &cos, &sin);
}

#[test]
fn sam_global_shape_matches_cpu() {
    let m = metal();
    // The 72 x 72 grid with positions scaled 24/72, as the global layers do.
    let (cos, sin) = axial(64, 72, 72, 24.0 / 72.0);
    let x = rand(&[1, 5184, 16, 64], 2, 2.0);
    check(&m, "[1,5184,16,64]", &x, &cos, &sin);
}

#[test]
fn small_and_odd_shapes_match_cpu() {
    let m = metal();
    // (batch, time, heads, dim): one pair, an odd time and head count, a
    // dim that is not a multiple of 4 (checkpoint tables are free-form).
    for (i, (b, t, h, d)) in [
        (1usize, 1usize, 1usize, 2usize),
        (1, 3, 1, 6),
        (2, 5, 3, 10),
        (3, 7, 5, 34),
        (1, 1, 16, 64),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = 100 + 10 * i as u64;
        let x = rand(&[b, t, h, d], seed, 3.0);
        let cos = rand(&[t, d / 2], seed + 1, 1.0);
        let sin = rand(&[t, d / 2], seed + 2, 1.0);
        check(&m, &format!("[{b},{t},{h},{d}]"), &x, &cos, &sin);
    }
}

#[test]
fn offset_device_views_give_the_cpu_answer() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[2, 4, 3, 8], 7, 2.0);
    let cos = rand(&[4, 4], 8, 1.0);
    let sin = rand(&[4, 4], 9, 1.0);
    let want = ok("cpu", c.rope_interleaved_forward(&x, &cos, &sin));
    let got = ok(
        "metal",
        m.rope_interleaved_forward(
            &offset_view(&m, &x, 3, 20),
            &offset_view(&m, &cos, 5, 21),
            &offset_view(&m, &sin, 7, 22),
        ),
    );
    ok("sync", m.sync());
    same_tensor("views", &got, &want, ATOL, RTOL);
}

#[test]
fn backward_inverts_forward() {
    let m = metal();
    let (cos, sin) = axial(64, 8, 8, 1.0);
    // The axial tables are unit-magnitude, so the rotation is orthogonal.
    let x = rand(&[2, 64, 4, 64], 11, 1.5);
    let (dx, dc, ds) = (up(&m, &x), up(&m, &cos), up(&m, &sin));
    let y = ok("fwd", m.rope_interleaved_forward(&dx, &dc, &ds));
    let back = ok("bwd", m.rope_interleaved_backward(&y, &dc, &ds));
    ok("sync", m.sync());
    same_tensor("round trip", &back, &x, 1e-5, 1e-5);
}

#[test]
fn a_malformed_call_is_refused_as_the_cpu_refuses_it() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[1, 3, 2, 8], 1, 1.0);
    let t4 = rand(&[3, 4], 2, 1.0);
    let cases: Vec<(&str, Tensor, Tensor, Tensor)> = vec![
        ("rank 3 x", rand(&[3, 2, 8], 3, 1.0), t4.clone(), t4.clone()),
        (
            "odd head dim",
            rand(&[1, 3, 2, 7], 4, 1.0),
            t4.clone(),
            t4.clone(),
        ),
        (
            "tables of half width",
            x.clone(),
            rand(&[3, 2], 5, 1.0),
            rand(&[3, 2], 6, 1.0),
        ),
        (
            "tables of the wrong time",
            x.clone(),
            rand(&[2, 4], 7, 1.0),
            rand(&[2, 4], 8, 1.0),
        ),
        (
            "cos and sin differ",
            x.clone(),
            t4.clone(),
            rand(&[3, 5], 9, 1.0),
        ),
        (
            "u32 tables",
            x.clone(),
            host_u32(&[0; 12], &[3, 4]),
            host_u32(&[0; 12], &[3, 4]),
        ),
    ];
    for (name, x, cos, sin) in cases {
        let want = rope_interleaved_forward_dims(&x, &cos, &sin).expect_err(name);
        assert!(
            matches!(
                c.rope_interleaved_forward(&x, &cos, &sin),
                Err(ref e) if format!("{e:?}") == format!("{want:?}")
            ),
            "{name}: the CPU must return the validator's error"
        );
        // Host operands (placement is checked after the shape) and device ones.
        let host_err = m.rope_interleaved_forward(&x, &cos, &sin).expect_err(name);
        assert_eq!(
            format!("{host_err:?}"),
            format!("{want:?}"),
            "{name} (host)"
        );
        if x.shape().iter().all(|&d| d > 0) && x.dtype() == ojas_core::DType::F32 {
            let r = m.rope_interleaved_forward(&up(&m, &x), &up(&m, &cos), &up(&m, &sin));
            let err = r.expect_err(name);
            assert_eq!(format!("{err:?}"), format!("{want:?}"), "{name} (device)");
        }
        let want_back = rope_interleaved_backward_dims(&x, &cos, &sin).expect_err(name);
        let back = m.rope_interleaved_backward(&x, &cos, &sin).expect_err(name);
        assert_eq!(
            format!("{back:?}"),
            format!("{want_back:?}"),
            "{name} (backward)"
        );
        ok("sync", m.sync());
    }
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_under_the_op_name() {
    let m = metal();
    let (cos, sin) = (rand(&[3, 4], 1, 1.0), rand(&[3, 4], 2, 1.0));
    let mut data = values(3 * 2 * 8, 3, 1.0);
    data[5] = f32::NAN;
    let x = up(&m, &host(&data, &[1, 3, 2, 8]));
    let (dc, ds) = (up(&m, &cos), up(&m, &sin));
    deferred(
        &m,
        "nan x",
        m.rope_interleaved_forward(&x, &dc, &ds),
        "rope_interleaved_forward",
    );
    deferred(
        &m,
        "nan x bwd",
        m.rope_interleaved_backward(&x, &dc, &ds),
        "rope_interleaved_backward",
    );
    let mut table = values(12, 4, 1.0);
    table[2] = f32::INFINITY;
    let bad = up(&m, &host(&table, &[3, 4]));
    let x = up(&m, &rand(&[1, 3, 2, 8], 5, 1.0));
    deferred(
        &m,
        "inf table",
        m.rope_interleaved_forward(&x, &bad, &ds),
        "rope_interleaved_forward",
    );
}
