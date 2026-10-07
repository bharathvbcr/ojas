//! **K7: Qwen3.5's two RMSNorms**, forward and backward, f64.
//!
//! - `Qwen3_5RMSNorm` (`modeling_qwen3_5.py:736-751`): `y = x * rstd * (1 + w)`,
//!   `rstd = 1 / sqrt(mean(x^2) + eps)`, the zero-centred `w` as stored.
//! - `Qwen3_5RMSNormGated` (`:187-202`): `y = w * (x * rstd) * silu(z)` per
//!   `d`-wide head.
//!
//! Rows are `d` wide and dense. `dw` reduces over rows in ascending order.
//! Ports of tessl's in-test references (`tests/qwen35_bwd.rs:83-110,195-224`).
//!
//! # Validation
//!
//! - **golden**: `tests/fixtures/goldens/rms_norm_*`, `gated_rms_norm_*` (torch
//!   float64 forward and autograd, `d` up to 2048).
//! - **derivative**: central differences.

use super::conv1d::{silu, silu_grad};
use super::gdn_published::exact_f64;

fn check(what: &str, x: &[f64], w: &[f64], d: usize) {
    assert!(d > 0, "{what}: d must be non-zero");
    assert_eq!(
        x.len() % d,
        0,
        "{what}: x length {} is not a multiple of d {d}",
        x.len()
    );
    assert_eq!(w.len(), d, "{what}: w must be [d]");
    for (name, xs) in [("x", x), ("w", w)] {
        if let Some(i) = xs.iter().position(|v| !v.is_finite()) {
            panic!("{what}: {name}[{i}] is not finite");
        }
    }
}

fn rstd(row: &[f64], d: usize, eps: f64) -> f64 {
    1.0 / (row.iter().map(|v| v * v).sum::<f64>() / exact_f64(d) + eps).sqrt()
}

/// `Qwen3_5RMSNorm`: `y [rows, d]`.
pub fn rms_norm_fwd_f64(x: &[f64], w: &[f64], d: usize, eps: f64) -> Vec<f64> {
    check("rms_norm", x, w, d);
    x.chunks(d)
        .flat_map(|r| {
            let rs = rstd(r, d, eps);
            r.iter()
                .zip(w)
                .map(move |(v, wv)| v * rs * (1.0 + wv))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `(dx [rows, d], dw [d])` of [`rms_norm_fwd_f64`] for the upstream `dy`.
pub fn rms_norm_bwd_f64(
    x: &[f64],
    w: &[f64],
    dy: &[f64],
    d: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    check("rms_norm", x, w, d);
    assert_eq!(dy.len(), x.len(), "rms_norm: dy length");
    let dd = exact_f64(d);
    let mut dx = vec![0.0; x.len()];
    let mut dw = vec![0.0; d];
    for (r, (xr, gr)) in x.chunks(d).zip(dy.chunks(d)).enumerate() {
        let rs = rstd(xr, d, eps);
        let dot: f64 = (0..d).map(|j| gr[j] * (1.0 + w[j]) * xr[j]).sum();
        for j in 0..d {
            dx[r * d + j] = rs * gr[j] * (1.0 + w[j]) - xr[j] * rs.powi(3) * dot / dd;
            dw[j] += gr[j] * xr[j] * rs;
        }
    }
    (dx, dw)
}

/// `Qwen3_5RMSNormGated`: `y [units, d]`.
pub fn gated_rms_norm_fwd_f64(x: &[f64], z: &[f64], w: &[f64], d: usize, eps: f64) -> Vec<f64> {
    check("gated_rms_norm", x, w, d);
    assert_eq!(z.len(), x.len(), "gated_rms_norm: z length");
    x.chunks(d)
        .zip(z.chunks(d))
        .flat_map(|(xr, zr)| {
            let rs = rstd(xr, d, eps);
            (0..d)
                .map(move |j| w[j] * (xr[j] * rs) * silu(zr[j]))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `(dx, dz [units, d], dw [d])` of [`gated_rms_norm_fwd_f64`] for `dy`.
pub fn gated_rms_norm_bwd_f64(
    x: &[f64],
    z: &[f64],
    w: &[f64],
    dy: &[f64],
    d: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    check("gated_rms_norm", x, w, d);
    assert_eq!(z.len(), x.len(), "gated_rms_norm: z length");
    assert_eq!(dy.len(), x.len(), "gated_rms_norm: dy length");
    let dd = exact_f64(d);
    let (mut dx, mut dz, mut dw) = (vec![0.0; x.len()], vec![0.0; x.len()], vec![0.0; d]);
    for u in 0..x.len() / d {
        let (xr, zr, gr) = (
            &x[u * d..(u + 1) * d],
            &z[u * d..(u + 1) * d],
            &dy[u * d..(u + 1) * d],
        );
        let rs = rstd(xr, d, eps);
        let xn: Vec<f64> = xr.iter().map(|v| v * rs).collect();
        let dxn: Vec<f64> = (0..d).map(|j| gr[j] * w[j] * silu(zr[j])).collect();
        let m: f64 = (0..d).map(|j| dxn[j] * xn[j]).sum::<f64>() / dd;
        for j in 0..d {
            dx[u * d + j] = rs * (dxn[j] - xn[j] * m);
            dz[u * d + j] = gr[j] * w[j] * xn[j] * silu_grad(zr[j]);
            dw[j] += gr[j] * xn[j] * silu(zr[j]);
        }
    }
    (dx, dz, dw)
}
