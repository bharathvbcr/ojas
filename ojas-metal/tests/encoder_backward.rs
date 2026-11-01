//! The encoder backwards on Metal: `layer_norm`, `channel_layer_norm`,
//! `gelu_erf`, `softmax`, `broadcast_add`, `broadcast_mul` and `linear_bias`
//! (`kernels/encoder_bwd.metal`, with tessl's GEMMs for the linear's `dx` and
//! `dw`).
//!
//! - Torch: the cases of `ojas-oracle/fixtures/encoder_bwd/golden.txt`, through
//!   the same function the CPU is held to (`ojas_oracle::encoder_golden`).
//! - The CPU: at SAM ViT shapes (5184 rows of 1024, an MLP of 4096), and over
//!   the thread-count, split and axis patterns the kernels choose between.
//! - The kernels ran: dispatch counts by kernel name.
//! - Reductions repeat bit for bit.
//! - A NaN or infinity, a malformed call and an over-budget call are refused as
//!   on the CPU.
//!
//! Tolerances (stated before measuring; see `backend/enc_bwd_host.rs` for the
//! reduction order and bound). Errors are `max |got - want| / max |want|`.
//! Metal sums in double-single and rounds once, where the CPU sums in `f64` and
//! rounds once, so a sum differs by about one `f32` ulp of itself; a LayerNorm
//! `dx` forms `rstd * (dxhat - mean - xhat * mean2)` in `f32` and carries a few
//! ulp of cancellation; GELU's derivative carries the forward fit's 1.2e-7.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;
use ojas_oracle::encoder_golden::{check, Tolerances};

/// A sum of products (`dw`, `db`, the broadcast gradients, a softmax dot's
/// consumers): one rounding of a nearly exact sum.
const SUM_TOL: f64 = 1e-6;
/// A LayerNorm `dx` and a softmax `dx`: a few `f32` roundings of terms that
/// cancel.
const DX_TOL: f64 = 5e-6;
/// GELU's derivative: the forward fit's relative error plus its roundings.
const GELU_TOL: f64 = 1e-6;
/// A GEMM output, summed over `k` = 1024 or 4096 in different orders.
const GEMM_TOL: f64 = 2e-5;
/// Against torch float64: the CPU holds 1e-6; Metal's `f32` combination adds
/// the fit and `rstd`'s two roundings.
const TORCH_F64_TOL: f64 = 3e-6;

fn torch_f32_tol(case: &str) -> f64 {
    match case {
        // Torch's own f32 statistics lose digits to the row mean; Metal's
        // double-single ones agree with float64 torch.
        "ln_mean_offset" => 2e-4,
        "cln_nlc" => 1e-4,
        _ => 3e-6,
    }
}

fn cpu() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(8 * GIB), 4)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

fn err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, w| m.max(f64::from(w.abs())));
    let worst = got.iter().zip(want).fold(0.0f64, |m, (g, w)| {
        m.max((f64::from(*g) - f64::from(*w)).abs())
    });
    if scale == 0.0 {
        worst
    } else {
        worst / scale
    }
}

/// Metal's `got` against the CPU's `want` (a host tensor), within `tol`.
fn near(what: &str, got: &Tensor, want: &Tensor, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    assert_eq!(
        got.device(),
        Some(ojas_core::BackendId::Metal),
        "{what}: residency"
    );
    let (g, w) = (down(got), ok(what, want.to_f32_vec()));
    assert!(g.iter().all(|v| v.is_finite()), "{what}: non-finite");
    let e = err(&g, &w);
    eprintln!("ERR {what}: {e:.2e} (tol {tol:.0e})");
    assert!(e <= tol, "{what}: {e:e} > {tol:e}");
}

fn near_affine(
    what: &str,
    got: &ojas_core::AffineGrad,
    want: &ojas_core::AffineGrad,
    tols: [f64; 3],
) {
    near(&format!("{what} dx"), &got.input, &want.input, tols[0]);
    near(&format!("{what} dw"), &got.weight, &want.weight, tols[1]);
    near(&format!("{what} db"), &got.bias, &want.bias, tols[2]);
}

fn near_one(shape: &[usize], seed: u64) -> Tensor {
    let v: Vec<f32> = values(shape.iter().product(), seed, 0.4)
        .iter()
        .map(|x| 1.0 + x)
        .collect();
    host(&v, shape)
}

#[allow(dead_code)]
fn pending(m: &MetalBackend) -> Option<&'static str> {
    match m.sync() {
        Ok(()) => None,
        Err(OjasError::NonFinite { op }) => Some(op),
        Err(e) => panic!("sync: {e:?}"),
    }
}

// ---- torch ----

#[test]
fn the_metal_backwards_match_torch_autograd_on_the_golden_cases() {
    let m = metal();
    let before = KERNELS.map(|k| m.kernel_dispatches(k));
    let report = check(&m).expect("every case runs");
    report.print("metal");
    report.assert_within(&Tolerances {
        vs_f64: TORCH_F64_TOL,
        vs_f32: torch_f32_tol,
    });
    // The goldens ran on the new kernels, not on a fallback. The broadcast
    // gradient's second stage runs only for a reduction past 32 terms, which
    // the small cases do not make.
    for (k, b) in KERNELS.iter().zip(before).take(8) {
        assert!(
            m.kernel_dispatches(k) > b,
            "no dispatch of {k}: the golden cases did not reach it"
        );
    }
}

/// The kernels a golden run must reach (`cln_*` and `cln_nlc` have spatial
/// extents; `ln_*` rows; `bmul` and `badd` reductions).
const KERNELS: [&str; 9] = [
    "ojas_ln_bwd_rows",
    "ojas_ln_bwd_cols_part",
    "ojas_cols_fin4",
    "ojas_cln_bwd_pos",
    "ojas_cln_bwd_ch_part",
    "ojas_gelu_erf_bwd",
    "ojas_softmax_bwd_rows",
    "ojas_bcast_grad_part",
    "ojas_bcast_grad_fin",
];

// ---- the CPU, at SAM ViT shapes ----

#[test]
fn layer_norm_backward_at_the_vit_shape_matches_the_cpu() {
    let (m, c) = (metal(), cpu());
    let (x, gy) = (rand(&[5184, 1024], 1, 3.0), rand(&[5184, 1024], 2, 1.0));
    let (w, b) = (near_one(&[1024], 3), rand(&[1024], 4, 0.3));
    let want = ok("cpu", c.layer_norm_backward(&x, &w, &b, &gy, 1e-6));
    let got = ok(
        "metal",
        m.layer_norm_backward(&up(&m, &x), &up(&m, &w), &up(&m, &b), &up(&m, &gy), 1e-6),
    );
    near_affine("ln 5184x1024", &got, &want, [DX_TOL, SUM_TOL, SUM_TOL]);
}

#[test]
fn channel_layer_norm_backward_at_the_neck_shape_matches_the_cpu() {
    let (m, c) = (metal(), cpu());
    for shape in [[2usize, 256, 48, 48], [1, 256, 72, 72], [3, 64, 7, 5]] {
        let (x, gy) = (rand(&shape, 5, 2.0), rand(&shape, 6, 1.0));
        let (w, b) = (near_one(&[shape[1]], 7), rand(&[shape[1]], 8, 0.3));
        let want = ok("cpu", c.channel_layer_norm_backward(&x, &w, &b, &gy, 1e-6));
        let got = ok(
            "metal",
            m.channel_layer_norm_backward(
                &up(&m, &x),
                &up(&m, &w),
                &up(&m, &b),
                &up(&m, &gy),
                1e-6,
            ),
        );
        near_affine(
            &format!("cln {shape:?}"),
            &got,
            &want,
            [DX_TOL, SUM_TOL, SUM_TOL],
        );
    }
}

#[test]
fn channel_layer_norm_backward_of_flat_channels_takes_the_row_path() {
    let (m, c) = (metal(), cpu());
    // [N, C] and [N, C, 1, 1]: one position per row, so the plain row kernel.
    for shape in [vec![9usize, 96], vec![9, 96, 1, 1]] {
        let (x, gy) = (rand(&shape, 11, 2.0), rand(&shape, 12, 1.0));
        let (w, b) = (near_one(&[96], 13), rand(&[96], 14, 0.3));
        let rows = m.kernel_dispatches("ojas_ln_bwd_rows");
        let pos = m.kernel_dispatches("ojas_cln_bwd_pos");
        let want = ok("cpu", c.channel_layer_norm_backward(&x, &w, &b, &gy, 1e-5));
        let got = ok(
            "metal",
            m.channel_layer_norm_backward(
                &up(&m, &x),
                &up(&m, &w),
                &up(&m, &b),
                &up(&m, &gy),
                1e-5,
            ),
        );
        near_affine(
            &format!("cln {shape:?}"),
            &got,
            &want,
            [DX_TOL, SUM_TOL, SUM_TOL],
        );
        ok("sync", m.sync());
        assert_eq!(
            m.kernel_dispatches("ojas_ln_bwd_rows"),
            rows + 1,
            "{shape:?}"
        );
        assert_eq!(m.kernel_dispatches("ojas_cln_bwd_pos"), pos, "{shape:?}");
    }
}

#[test]
fn gelu_erf_backward_at_the_mlp_shape_matches_the_cpu() {
    let (m, c) = (metal(), cpu());
    let (x, gy) = (rand(&[5184, 4096], 21, 4.0), rand(&[5184, 4096], 22, 1.0));
    let want = ok("cpu", c.gelu_erf_backward(&x, &gy));
    let got = ok("metal", m.gelu_erf_backward(&up(&m, &x), &up(&m, &gy)));
    near("gelu 5184x4096", &got, &want, GELU_TOL);
}

#[test]
fn gelu_erf_backward_covers_the_tails_and_the_saturation_edge() {
    let (m, c) = (metal(), cpu());
    let mut xs: Vec<f32> = (-120..=120).map(|i| i as f32 * 0.1).collect();
    xs.extend([
        1e-20, -1e-20, 0.0, -0.0, 8.99, 9.01, -8.99, -9.01, 14.0, -14.0, 18.5, -18.5, 25.0, -25.0,
        1e5, -1e5, 1e30, -1e30,
    ]);
    let n = xs.len();
    let x = host(&xs, &[n]);
    // A gradient near the f32 maximum keeps the density term the forward's
    // exp would have underflowed (|x| up to 18.5).
    for scale in [1.0f32, 1e30] {
        let gy = host(&values(n, 23, scale), &[n]);
        let want = ok("cpu", c.gelu_erf_backward(&x, &gy));
        let got = ok("metal", m.gelu_erf_backward(&up(&m, &x), &up(&m, &gy)));
        near(
            &format!("gelu tails, gradient {scale:e}"),
            &got,
            &want,
            GELU_TOL,
        );
    }
}

#[test]
fn softmax_backward_matches_the_cpu_on_every_row_length_path() {
    let (m, c) = (metal(), cpu());
    // Row lengths pick 32, 64, 128 and 256 threads and a strided tail.
    for (rows, dim) in [
        (5184usize, 1024usize),
        (4608, 576),
        (7, 1),
        (7, 2),
        (7, 31),
        (7, 32),
        (7, 33),
        (7, 255),
        (7, 257),
        (3, 4097),
    ] {
        let logits = rand(&[rows, dim], 31, 4.0);
        let y = ok("softmax", c.softmax_forward(&logits));
        let gy = rand(&[rows, dim], 32, 1.0);
        let want = ok("cpu", c.softmax_backward(&y, &gy));
        let got = ok("metal", m.softmax_backward(&up(&m, &y), &up(&m, &gy)));
        near(&format!("softmax {rows}x{dim}"), &got, &want, DX_TOL);
    }
}

#[test]
fn the_broadcast_gradients_match_the_cpu_over_every_axis_pattern() {
    let (m, cp) = (metal(), cpu());
    let cases: [(&[usize], &[usize]); 11] = [
        (&[5184, 1024], &[1024]),
        (&[1024], &[5184, 1024]),
        (&[8, 16, 196, 196], &[1, 16, 196, 196]),
        (&[4, 1, 64, 1], &[1, 8, 1, 64]),
        (&[6, 5], &[6, 5]),
        (&[6, 5], &[1]),
        (&[3, 1, 5, 1, 7], &[2, 1, 4, 1]),
        (&[2, 3, 1, 5], &[3, 4, 1]),
        (&[1], &[300_000]),
        (&[2, 2, 2, 2, 2, 2, 2, 3], &[2, 1, 2, 1, 2, 1, 2, 1]),
        (&[1000, 1, 33], &[1, 77, 1]),
    ];
    for (i, (sa, sb)) in cases.into_iter().enumerate() {
        let (a, b) = (rand(sa, 40 + i as u64, 1.5), rand(sb, 60 + i as u64, 1.5));
        let out = ok("shape", ojas_core::broadcast_shapes("test", sa, sb)).out_shape;
        let gy = rand(&out, 80 + i as u64, 1.0);
        let (ua, ub, ug) = (up(&m, &a), up(&m, &b), up(&m, &gy));
        let (wa, wb) = ok("cpu", cp.broadcast_add_backward(&a, &b, &gy));
        let (ga, gb) = ok("metal", m.broadcast_add_backward(&ua, &ub, &ug));
        near(&format!("badd {sa:?} {sb:?} da"), &ga, &wa, SUM_TOL);
        near(&format!("badd {sa:?} {sb:?} db"), &gb, &wb, SUM_TOL);
        let (wa, wb) = ok("cpu", cp.broadcast_mul_backward(&a, &b, &gy));
        let (ga, gb) = ok("metal", m.broadcast_mul_backward(&ua, &ub, &ug));
        near(&format!("bmul {sa:?} {sb:?} da"), &ga, &wa, SUM_TOL);
        near(&format!("bmul {sa:?} {sb:?} db"), &gb, &wb, SUM_TOL);
    }
}

#[test]
fn linear_bias_backward_at_the_mlp_shape_matches_the_cpu() {
    let (m, c) = (metal(), cpu());
    let (x, w) = (rand(&[5184, 1024], 91, 1.0), rand(&[4096, 1024], 92, 0.05));
    let (b, gy) = (rand(&[4096], 93, 0.3), rand(&[5184, 4096], 94, 1.0));
    let want = ok("cpu", c.linear_bias_backward(&x, &w, &b, &gy));
    let got = ok(
        "metal",
        m.linear_bias_backward(&up(&m, &x), &up(&m, &w), &up(&m, &b), &up(&m, &gy)),
    );
    near_affine(
        "linear_bias 5184x1024x4096",
        &got,
        &want,
        [GEMM_TOL, GEMM_TOL, SUM_TOL],
    );
    // The input and weight gradients are linear_backward's, bit for bit.
    let (gx, gw) = ok(
        "linear_backward",
        m.linear_backward(&up(&m, &x), &up(&m, &w), &up(&m, &gy)),
    );
    same_bits(
        "linear_bias dx is linear_backward's",
        &got.input,
        &down_host(&gx),
    );
    same_bits(
        "linear_bias dw is linear_backward's",
        &got.weight,
        &down_host(&gw),
    );
}

/// A device tensor's bits as a host tensor, to compare with `same_bits`.
fn down_host(t: &Tensor) -> Tensor {
    host(&down(t), t.shape())
}

#[test]
fn linear_bias_backward_of_leading_axes_sums_all_the_rows() {
    let (m, c) = (metal(), cpu());
    let (x, w) = (rand(&[3, 5, 7, 16], 95, 1.0), rand(&[11, 16], 96, 0.3));
    let (b, gy) = (rand(&[11], 97, 0.3), rand(&[3, 5, 7, 11], 98, 1.0));
    let want = ok("cpu", c.linear_bias_backward(&x, &w, &b, &gy));
    let got = ok(
        "metal",
        m.linear_bias_backward(&up(&m, &x), &up(&m, &w), &up(&m, &b), &up(&m, &gy)),
    );
    near_affine(
        "linear_bias [3,5,7,16]",
        &got,
        &want,
        [GEMM_TOL, GEMM_TOL, SUM_TOL],
    );
}

#[test]
fn layer_norm_backward_covers_every_row_length_path() {
    let (m, c) = (metal(), cpu());
    for (rows, dim) in [
        (1usize, 1usize),
        (4, 2),
        (9, 31),
        (9, 32),
        (9, 33),
        (9, 255),
        (9, 257),
        (130, 1000),
        (2, 20_000),
    ] {
        let (x, gy) = (rand(&[rows, dim], 101, 2.0), rand(&[rows, dim], 102, 1.0));
        let (w, b) = (near_one(&[dim], 103), rand(&[dim], 104, 0.3));
        let want = ok("cpu", c.layer_norm_backward(&x, &w, &b, &gy, 1e-5));
        let got = ok(
            "metal",
            m.layer_norm_backward(&up(&m, &x), &up(&m, &w), &up(&m, &b), &up(&m, &gy), 1e-5),
        );
        if dim == 1 {
            // A single element is its own mean: dx is exactly 0.
            assert!(down(&got.input).iter().all(|v| *v == 0.0), "dim 1 dx");
        }
        let dx_tol = if dim == 2 { 1.5e-3 } else { DX_TOL };
        near_affine(
            &format!("ln {rows}x{dim}"),
            &got,
            &want,
            [dx_tol, SUM_TOL, SUM_TOL],
        );
    }
}

// ---- accuracy where a plain f32 sum fails ----

/// `dx`, `dw` and `db` of LayerNorm in `f64` for rows of `dim`.
fn layer_norm_f64(x: &[f32], w: &[f32], gy: &[f32], dim: usize, eps: f64) -> [Vec<f64>; 3] {
    let rows = x.len() / dim;
    let (mut dx, mut dw, mut db) = (vec![0.0; x.len()], vec![0.0; dim], vec![0.0; dim]);
    for r in 0..rows {
        let row: Vec<f64> = x[r * dim..(r + 1) * dim]
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let mean = row.iter().sum::<f64>() / dim as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / dim as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        let g: Vec<f64> = gy[r * dim..(r + 1) * dim]
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let dxhat: Vec<f64> = g.iter().zip(w).map(|(g, w)| g * f64::from(*w)).collect();
        let xhat: Vec<f64> = row.iter().map(|v| (v - mean) * rstd).collect();
        let m1 = dxhat.iter().sum::<f64>() / dim as f64;
        let m2 = dxhat.iter().zip(&xhat).map(|(a, b)| a * b).sum::<f64>() / dim as f64;
        for j in 0..dim {
            dx[r * dim + j] = rstd * (dxhat[j] - m1 - xhat[j] * m2);
            dw[j] += g[j] * xhat[j];
            db[j] += g[j];
        }
    }
    [dx, dw, db]
}

fn err64(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    let worst = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - w).abs()));
    worst / scale
}

#[test]
fn layer_norm_backward_keeps_its_accuracy_on_rows_with_a_large_mean() {
    let m = metal();
    let (rows, dim) = (64usize, 1024usize);
    // Mean 1000, deviation 1: a one-pass or f32-accumulated variance loses
    // every digit of it.
    let x: Vec<f32> = values(rows * dim, 111, 1.7)
        .iter()
        .map(|v| v + 1000.0)
        .collect();
    let (gy, w) = (values(rows * dim, 112, 1.0), values(dim, 113, 0.4));
    let w: Vec<f32> = w.iter().map(|v| 1.0 + v).collect();
    let b = vec![0.0f32; dim];
    let want = layer_norm_f64(&x, &w, &gy, dim, 1e-5);
    let got = ok(
        "metal",
        m.layer_norm_backward(
            &up(&m, &host(&x, &[rows, dim])),
            &up(&m, &host(&w, &[dim])),
            &up(&m, &host(&b, &[dim])),
            &up(&m, &host(&gy, &[rows, dim])),
            1e-5,
        ),
    );
    for (name, g, w) in [
        ("dx", &got.input, &want[0]),
        ("dw", &got.weight, &want[1]),
        ("db", &got.bias, &want[2]),
    ] {
        let e = err64(&down(g), w);
        eprintln!("ERR large-mean {name} vs f64: {e:.2e}");
        assert!(e <= 3e-6, "{name}: {e:e}");
    }
}

#[test]
fn the_broadcast_and_bias_sums_are_correctly_rounded_over_a_long_reduction() {
    let m = metal();
    // 2^20 terms of 0.1 and a few large cancelling ones: an f32 running sum
    // is off by 1e-3 relative; the double-single sum is the f64 sum rounded.
    let n = 1 << 20;
    let mut g = vec![0.1f32; n];
    g[0] = 1.0e7;
    g[n - 1] = -1.0e7;
    let want = g.iter().map(|v| f64::from(*v)).sum::<f64>();
    let a = up(&m, &host(&[0.0], &[1]));
    let b = up(&m, &host(&vec![0.0; n], &[n]));
    let (ga, _) = ok(
        "metal",
        m.broadcast_add_backward(&a, &b, &up(&m, &host(&g, &[n]))),
    );
    let got = f64::from(down(&ga)[0]);
    assert!(
        (got - want).abs() <= want.abs() * 1.3e-7,
        "{got} vs the f64 sum {want}"
    );
}

#[test]
fn adversarial_double_single_precision_n5184_d2_and_error_free_transforms() {
    let m = metal();
    // 1. Error-Free Transformation mathematical properties:
    let two_sum = |a: f32, b: f32| -> (f32, f32) {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        (s, e)
    };
    let two_prod = |a: f32, b: f32| -> (f32, f32) {
        let p = a * b;
        let e = a.mul_add(b, -p);
        (p, e)
    };
    for (a, b) in [
        (1.0f32, 1e-7f32),
        (1e7, -1e7 + 0.1),
        (std::f32::consts::PI, std::f32::consts::E),
        (1e-12, 1e-12), // product 1e-24 > 2^-102 (residual error is normal float32)
        (1e20, 1e10),   // product 1e30 < f32::MAX (3.4e38)
    ] {
        let (s, e) = two_sum(a, b);
        let exact_sum = f64::from(a) + f64::from(b);
        let recovered_sum = f64::from(s) + f64::from(e);
        assert_eq!(recovered_sum, exact_sum, "eb_two_sum failed on ({a}, {b})");

        let (p, pe) = two_prod(a, b);
        let exact_prod = f64::from(a) * f64::from(b);
        let recovered_prod = f64::from(p) + f64::from(pe);
        assert_eq!(
            recovered_prod, exact_prod,
            "eb_two_prod failed on ({a}, {b})"
        );
    }

    // Residual subnormal boundary verification:
    // When |a * b| < 2^-102 (~2e-31), the residual error e = fl(a*b - p) enters the
    // subnormal exponent range (< 2^-126) and loses bits through gradual underflow.
    let sub_a = 1e-18f32;
    let sub_b = 1e-18f32;
    let (sub_p, sub_pe) = two_prod(sub_a, sub_b);
    let sub_exact = f64::from(sub_a) * f64::from(sub_b);
    let sub_rec = f64::from(sub_p) + f64::from(sub_pe);
    assert!(
        (sub_rec - sub_exact).abs() < 1e-44,
        "subnormal residual is bounded by f32 subnormal precision"
    );

    // 2. ViT sequence length N=5184, small head dimension D=2:
    let (rows, dim) = (5184usize, 2usize);
    let x: Vec<f32> = values(rows * dim, 991, 1.5)
        .iter()
        .map(|v| v + 1000.0)
        .collect();
    let gy: Vec<f32> = values(rows * dim, 992, 1.0);
    let w = vec![1.2f32, 0.8f32];
    let b = vec![0.1f32, -0.1f32];

    let want = layer_norm_f64(&x, &w, &gy, dim, 1e-5);
    let got = ok(
        "metal ln n5184 d2",
        m.layer_norm_backward(
            &up(&m, &host(&x, &[rows, dim])),
            &up(&m, &host(&w, &[dim])),
            &up(&m, &host(&b, &[dim])),
            &up(&m, &host(&gy, &[rows, dim])),
            1e-5,
        ),
    );
    for (name, g, w) in [
        ("dx", &got.input, &want[0]),
        ("dw", &got.weight, &want[1]),
        ("db", &got.bias, &want[2]),
    ] {
        let e = err64(&down(g), w);
        eprintln!("ERR adversarial N=5184 D=2 {name} vs f64: {e:.2e}");
        let tol = if name == "dx" { 1.5e-3 } else { 3e-6 };
        assert!(e <= tol, "{name}: {e:e} > {tol:e}");
    }
}

// ---- determinism and dispatch ----

#[test]
fn every_backward_repeats_its_bits() {
    let m = metal();
    let (x, gy) = (
        up(&m, &rand(&[5184, 1024], 1, 3.0)),
        up(&m, &rand(&[5184, 1024], 2, 1.0)),
    );
    let (w, b) = (
        up(&m, &near_one(&[1024], 3)),
        up(&m, &rand(&[1024], 4, 0.3)),
    );
    let img = up(&m, &rand(&[2, 64, 40, 40], 5, 2.0));
    let img_g = up(&m, &rand(&[2, 64, 40, 40], 6, 1.0));
    let (cw, cb) = (up(&m, &near_one(&[64], 7)), up(&m, &rand(&[64], 8, 0.3)));
    let big = up(&m, &rand(&[5184, 1024], 9, 1.0));
    let row = up(&m, &rand(&[1024], 10, 1.0));
    let run = || -> Vec<Vec<u32>> {
        let ln = ok("ln", m.layer_norm_backward(&x, &w, &b, &gy, 1e-6));
        let cln = ok(
            "cln",
            m.channel_layer_norm_backward(&img, &cw, &cb, &img_g, 1e-6),
        );
        let sm = ok("softmax", m.softmax_backward(&x, &gy));
        let ge = ok("gelu", m.gelu_erf_backward(&x, &gy));
        let (ba, bb) = ok("badd", m.broadcast_add_backward(&big, &row, &gy));
        let (ma, mb) = ok("bmul", m.broadcast_mul_backward(&big, &row, &gy));
        [
            ln.input, ln.weight, ln.bias, cln.input, cln.weight, cln.bias, sm, ge, ba, bb, ma, mb,
        ]
        .iter()
        .map(|t| bits(&down(t)))
        .collect()
    };
    assert!(run() == run(), "a backward did not repeat its bits");
}

#[test]
fn each_backward_dispatches_its_own_kernels() {
    let m = metal();
    let count = |k: &str| m.kernel_dispatches(k);
    let (x, gy) = (
        up(&m, &rand(&[64, 128], 1, 1.0)),
        up(&m, &rand(&[64, 128], 2, 1.0)),
    );
    let (w, b) = (up(&m, &near_one(&[128], 3)), up(&m, &rand(&[128], 4, 0.3)));

    let before = KERNELS.map(count);
    ok("ln", m.layer_norm_backward(&x, &w, &b, &gy, 1e-5));
    ok("sync", m.sync());
    let after = KERNELS.map(count);
    let delta: Vec<u64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
    // rows, cols_part, fin4; nothing else.
    assert_eq!(delta, [1, 1, 1, 0, 0, 0, 0, 0, 0], "layer_norm_backward");

    let before = KERNELS.map(count);
    ok("gelu", m.gelu_erf_backward(&x, &gy));
    ok("softmax", m.softmax_backward(&x, &gy));
    ok("sync", m.sync());
    let delta: Vec<u64> = KERNELS
        .map(count)
        .iter()
        .zip(before)
        .map(|(a, b)| a - b)
        .collect();
    assert_eq!(delta, [0, 0, 0, 0, 0, 1, 1, 0, 0], "gelu and softmax");

    // Equal shapes: one thread per element, no reduction, no second stage.
    let before = KERNELS.map(count);
    ok("badd", m.broadcast_add_backward(&x, &gy, &gy));
    ok("sync", m.sync());
    let delta: Vec<u64> = KERNELS
        .map(count)
        .iter()
        .zip(before)
        .map(|(a, b)| a - b)
        .collect();
    assert_eq!(
        delta,
        [0, 0, 0, 0, 0, 0, 0, 2, 0],
        "broadcast_add_backward, equal shapes"
    );

    // A row broadcast: the stretched operand's gradient sums 64 rows in splits.
    let before = KERNELS.map(count);
    ok("badd", m.broadcast_add_backward(&x, &w, &gy));
    ok("sync", m.sync());
    let delta: Vec<u64> = KERNELS
        .map(count)
        .iter()
        .zip(before)
        .map(|(a, b)| a - b)
        .collect();
    assert_eq!(
        delta,
        [0, 0, 0, 0, 0, 0, 0, 2, 1],
        "broadcast_add_backward, row bias"
    );

    let img = up(&m, &rand(&[2, 8, 5, 5], 5, 1.0));
    let (cw, cb) = (up(&m, &near_one(&[8], 6)), up(&m, &rand(&[8], 7, 0.3)));
    let before = KERNELS.map(count);
    ok(
        "cln",
        m.channel_layer_norm_backward(&img, &cw, &cb, &img, 1e-5),
    );
    ok("sync", m.sync());
    let delta: Vec<u64> = KERNELS
        .map(count)
        .iter()
        .zip(before)
        .map(|(a, b)| a - b)
        .collect();
    assert_eq!(
        delta,
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        "channel_layer_norm_backward"
    );
}

// ---- faults ----

/// `x` with its last element replaced by `v`, as a device tensor.
fn poisoned(m: &MetalBackend, shape: &[usize], seed: u64, v: f32) -> Tensor {
    let n = shape.iter().product::<usize>();
    let mut d = values(n, seed, 1.0);
    d[n - 1] = v;
    up(m, &host(&d, shape))
}

#[test]
fn a_nan_or_infinity_in_any_operand_is_reported_naming_the_backward() {
    let m = metal();
    let clean = |shape: &[usize], seed: u64| up(&m, &rand(shape, seed, 1.0));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for at in 0..4 {
            let mut ops = [
                poisoned(&m, &[5, 8], 1, 1.0),
                clean(&[8], 2),
                clean(&[8], 3),
                clean(&[5, 8], 4),
            ];
            ops[at] = poisoned(&m, ops[at].shape(), 20 + at as u64, bad);
            let r = m.layer_norm_backward(&ops[0], &ops[1], &ops[2], &ops[3], 1e-5);
            deferred(
                &m,
                &format!("layer_norm {bad} at {at}"),
                r,
                "layer_norm_backward",
            );
        }
        for at in 0..4 {
            let mut ops = [
                clean(&[2, 6, 3, 3], 1),
                clean(&[6], 2),
                clean(&[6], 3),
                clean(&[2, 6, 3, 3], 4),
            ];
            ops[at] = poisoned(&m, ops[at].shape(), 30 + at as u64, bad);
            let r = m.channel_layer_norm_backward(&ops[0], &ops[1], &ops[2], &ops[3], 1e-5);
            deferred(
                &m,
                &format!("channel_layer_norm {bad} at {at}"),
                r,
                "channel_layer_norm_backward",
            );
        }
        for at in 0..2 {
            let mut ops = [clean(&[40], 1), clean(&[40], 2)];
            ops[at] = poisoned(&m, &[40], 40 + at as u64, bad);
            deferred(
                &m,
                &format!("gelu {bad} at {at}"),
                m.gelu_erf_backward(&ops[0], &ops[1]),
                "gelu_erf_backward",
            );
            deferred(
                &m,
                &format!("softmax {bad} at {at}"),
                m.softmax_backward(&ops[0], &ops[1]),
                "softmax_backward",
            );
        }
        // Every operand of a broadcast, even the ones a kernel reads only for
        // a multiply's factor.
        for at in 0..3 {
            let mut ops = [clean(&[4, 6], 1), clean(&[6], 2), clean(&[4, 6], 3)];
            ops[at] = poisoned(&m, ops[at].shape(), 50 + at as u64, bad);
            deferred(
                &m,
                &format!("badd {bad} at {at}"),
                m.broadcast_add_backward(&ops[0], &ops[1], &ops[2]),
                "broadcast_add_backward",
            );
            deferred(
                &m,
                &format!("bmul {bad} at {at}"),
                m.broadcast_mul_backward(&ops[0], &ops[1], &ops[2]),
                "broadcast_mul_backward",
            );
        }
        // The bias is read only for its check.
        let r = m.linear_bias_backward(
            &clean(&[4, 8], 1),
            &clean(&[5, 8], 2),
            &poisoned(&m, &[5], 3, bad),
            &clean(&[4, 5], 4),
        );
        deferred(
            &m,
            &format!("linear_bias {bad} in the bias"),
            r,
            "linear_bias_backward",
        );
    }
}

#[test]
fn an_overflowing_gradient_is_reported_not_returned_as_infinity() {
    let m = metal();
    let (x, gy) = (
        up(&m, &host(&[1.5, 0.5], &[2])),
        up(&m, &host(&[f32::MAX, 1.0], &[2])),
    );
    deferred(
        &m,
        "gelu overflow",
        m.gelu_erf_backward(&x, &gy),
        "gelu_erf_backward",
    );
    let (xl, wl, bl) = (
        up(&m, &rand(&[2, 8], 1, 1.0)),
        up(&m, &host(&[3.0e38; 8], &[8])),
        up(&m, &host(&[0.0; 8], &[8])),
    );
    let g = up(&m, &host(&[3.0e38; 16], &[2, 8]));
    deferred(
        &m,
        "layer_norm overflow",
        m.layer_norm_backward(&xl, &wl, &bl, &g, 1e-5),
        "layer_norm_backward",
    );
    // A sum whose total passes f32 even though every term is finite.
    let (a, b) = (up(&m, &host(&[0.0], &[1])), up(&m, &host(&[0.0; 4], &[4])));
    let gsum = up(&m, &host(&[3.0e38; 4], &[4]));
    deferred(
        &m,
        "broadcast sum overflow",
        m.broadcast_add_backward(&a, &b, &gsum),
        "broadcast_add_backward",
    );
}

#[test]
fn a_variance_that_is_not_positive_is_reported_as_on_the_cpu() {
    let (m, c) = (metal(), cpu());
    // A constant row with eps 0: rstd is infinite.
    let x = host(&[2.0; 8], &[1, 8]);
    let (w, b, gy) = (
        host(&[1.0; 8], &[8]),
        host(&[0.0; 8], &[8]),
        rand(&[1, 8], 1, 1.0),
    );
    let cpu_err = c.layer_norm_backward(&x, &w, &b, &gy, 0.0).unwrap_err();
    assert!(
        matches!(
            cpu_err,
            OjasError::NonFinite {
                op: "layer_norm_backward"
            }
        ),
        "{cpu_err:?}"
    );
    deferred(
        &m,
        "constant row, eps 0",
        m.layer_norm_backward(&up(&m, &x), &up(&m, &w), &up(&m, &b), &up(&m, &gy), 0.0),
        "layer_norm_backward",
    );
    // With an eps the same row is fine: dx = rstd * (dxhat - mean(dxhat)).
    let want = ok("cpu", c.layer_norm_backward(&x, &w, &b, &gy, 1e-5));
    let got = ok(
        "metal",
        m.layer_norm_backward(&up(&m, &x), &up(&m, &w), &up(&m, &b), &up(&m, &gy), 1e-5),
    );
    near_affine("constant row", &got, &want, [DX_TOL, SUM_TOL, SUM_TOL]);
}

// ---- refusals ----

#[test]
fn a_malformed_call_is_refused_with_the_cpus_error() {
    let (m, c) = (metal(), cpu());
    let r = |shape: &[usize]| rand(shape, 1, 1.0);
    let u = host_u32(&[1, 2, 3, 4], &[4]);
    // Shape and dtype are decided before placement, so host tensors reach the
    // validators on both backends.
    type Call = Box<dyn Fn(&dyn Backend) -> Option<String>>;
    let calls: Vec<(&str, Call)> = vec![
        (
            "ln gy shape",
            Box::new({
                let (x, w, b, g) = (r(&[4, 8]), r(&[8]), r(&[8]), r(&[4, 7]));
                move |be| {
                    be.layer_norm_backward(&x, &w, &b, &g, 1e-5)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "ln weight shape",
            Box::new({
                let (x, w, b, g) = (r(&[4, 8]), r(&[7]), r(&[8]), r(&[4, 8]));
                move |be| {
                    be.layer_norm_backward(&x, &w, &b, &g, 1e-5)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "ln eps nan",
            Box::new({
                let (x, w, b, g) = (r(&[4, 8]), r(&[8]), r(&[8]), r(&[4, 8]));
                move |be| {
                    be.layer_norm_backward(&x, &w, &b, &g, f32::NAN)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "ln dtype",
            Box::new({
                let (w, b) = (r(&[4]), r(&[4]));
                let u = u.clone();
                move |be| {
                    be.layer_norm_backward(&u, &w, &b, &u, 1e-5)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "cln rank 1",
            Box::new({
                let (x, w, b, g) = (r(&[8]), r(&[8]), r(&[8]), r(&[8]));
                move |be| {
                    be.channel_layer_norm_backward(&x, &w, &b, &g, 1e-5)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "cln channels",
            Box::new({
                let (x, w, b, g) = (r(&[2, 6, 3]), r(&[5]), r(&[6]), r(&[2, 6, 3]));
                move |be| {
                    be.channel_layer_norm_backward(&x, &w, &b, &g, 1e-5)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "gelu shape",
            Box::new({
                let (x, g) = (r(&[4, 8]), r(&[8, 4]));
                move |be| be.gelu_erf_backward(&x, &g).err().map(|e| format!("{e:?}"))
            }),
        ),
        (
            "softmax rank 0",
            Box::new({
                let (x, g) = (host(&[1.0], &[]), host(&[1.0], &[]));
                move |be| be.softmax_backward(&x, &g).err().map(|e| format!("{e:?}"))
            }),
        ),
        (
            "softmax shape",
            Box::new({
                let (x, g) = (r(&[4, 8]), r(&[4, 9]));
                move |be| be.softmax_backward(&x, &g).err().map(|e| format!("{e:?}"))
            }),
        ),
        (
            "badd incompatible",
            Box::new({
                let (a, b, g) = (r(&[3, 4]), r(&[5]), r(&[3, 4]));
                move |be| {
                    be.broadcast_add_backward(&a, &b, &g)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "bmul grad shape",
            Box::new({
                let (a, b, g) = (r(&[3, 4]), r(&[4]), r(&[4, 3]));
                move |be| {
                    be.broadcast_mul_backward(&a, &b, &g)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "bmul rank 9",
            Box::new({
                let (a, b, g) = (
                    r(&[1, 1, 1, 1, 1, 1, 1, 1, 2]),
                    r(&[2]),
                    r(&[1, 1, 1, 1, 1, 1, 1, 1, 2]),
                );
                move |be| {
                    be.broadcast_mul_backward(&a, &b, &g)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "linear_bias grad shape",
            Box::new({
                let (x, w, b, g) = (r(&[4, 8]), r(&[5, 8]), r(&[5]), r(&[4, 6]));
                move |be| {
                    be.linear_bias_backward(&x, &w, &b, &g)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
        (
            "linear_bias bias shape",
            Box::new({
                let (x, w, b, g) = (r(&[4, 8]), r(&[5, 8]), r(&[4]), r(&[4, 5]));
                move |be| {
                    be.linear_bias_backward(&x, &w, &b, &g)
                        .err()
                        .map(|e| format!("{e:?}"))
                }
            }),
        ),
    ];
    for (what, call) in &calls {
        let (want, got) = (call(&c), call(&m));
        assert!(want.is_some(), "{what}: the CPU accepted the call");
        assert_eq!(got, want, "{what}: Metal's refusal differs from the CPU's");
    }
}

#[test]
fn a_host_tensor_is_a_placement_error_after_the_shape_checks_pass() {
    let m = metal();
    let (x, w, b, g) = (
        rand(&[4, 8], 1, 1.0),
        rand(&[8], 2, 1.0),
        rand(&[8], 3, 1.0),
        rand(&[4, 8], 4, 1.0),
    );
    for r in [
        m.layer_norm_backward(&x, &w, &b, &g, 1e-5).map(|_| ()),
        m.gelu_erf_backward(&x, &g).map(|_| ()),
        m.softmax_backward(&x, &g).map(|_| ()),
        m.broadcast_add_backward(&x, &w, &g).map(|_| ()),
    ] {
        assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    }
}

#[test]
fn an_over_budget_backward_is_refused_and_leaves_no_charge() {
    let operands = |m: &MetalBackend| {
        (
            up(m, &rand(&[512, 256], 1, 1.0)),
            up(m, &near_one(&[256], 2)),
            up(m, &rand(&[256], 3, 0.3)),
            up(m, &rand(&[512, 256], 4, 1.0)),
        )
    };
    let probe = MetalBackend::new(Budget::new(1 << 30)).expect("device");
    let (x, w, b, g) = operands(&probe);
    let live = ok("live", probe.budget().live_bytes());
    probe.budget().reset_peak();
    drop(ok("ln", probe.layer_norm_backward(&x, &w, &b, &g, 1e-5)));
    ok("sync", probe.sync());
    let needed = probe.budget().peak_bytes() - live;
    assert!(
        needed > 4 * 512 * 256,
        "the call charged its outputs and scratch: {needed}"
    );

    // Room for the operands and the input gradient only.
    let m = MetalBackend::new(Budget::new(live + 4 * 512 * 256 + 1024)).expect("device");
    let (x, w, b, g) = operands(&m);
    let before = ok("live", m.budget().live_bytes());
    let e = m.layer_norm_backward(&x, &w, &b, &g, 1e-5).unwrap_err();
    assert!(matches!(e, OjasError::CapacityExceeded { .. }), "{e:?}");
    assert_eq!(
        ok("live", m.budget().live_bytes()),
        before,
        "a refusal left a charge"
    );
}

#[test]
fn a_broadcast_gradients_partial_sums_are_charged_to_the_budget() {
    // `a` has the output's shape, so its gradient is a copy with no partials.
    // `b`'s gradient sums 512 rows of 4096 in 16 splits: 16 * 4096 * 2 f32 of
    // partials (512 KiB), past the 100 KiB of room left over.
    let (rows, cols) = (512usize, 4096usize);
    let operand_bytes = 4 * (2 * rows * cols + cols) as u64;
    let outputs = 4 * (rows * cols + cols) as u64;
    let m = MetalBackend::new(Budget::new(operand_bytes + outputs + (100 << 10))).expect("device");
    let (a, b) = (
        up(&m, &rand(&[rows, cols], 3, 1.0)),
        up(&m, &rand(&[cols], 1, 1.0)),
    );
    let g = up(&m, &rand(&[rows, cols], 2, 1.0));
    let before = ok("live", m.budget().live_bytes());
    let e = m.broadcast_add_backward(&a, &b, &g).unwrap_err();
    assert!(matches!(e, OjasError::CapacityExceeded { .. }), "{e:?}");
    assert_eq!(
        ok("live", m.budget().live_bytes()),
        before,
        "a refusal left a charge"
    );
    // With room for the partials the same call is exact to the CPU.
    let roomy = MetalBackend::new(Budget::new(1 << 30)).expect("device");
    let (a, b) = (
        up(&roomy, &rand(&[rows, cols], 3, 1.0)),
        up(&roomy, &rand(&[cols], 1, 1.0)),
    );
    let g = up(&roomy, &rand(&[rows, cols], 2, 1.0));
    let (ga, gb) = ok("roomy", roomy.broadcast_add_backward(&a, &b, &g));
    let want = ok(
        "cpu",
        cpu().broadcast_add_backward(
            &rand(&[rows, cols], 3, 1.0),
            &rand(&[cols], 1, 1.0),
            &rand(&[rows, cols], 2, 1.0),
        ),
    );
    near("badd da", &ga, &want.0, SUM_TOL);
    near("badd db", &gb, &want.1, SUM_TOL);
}
