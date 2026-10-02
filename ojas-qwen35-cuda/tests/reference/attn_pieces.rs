//! **K6: the attention pieces around the core**, forward and backward, f64.
//!
//! 1. **Q/K norm + partial RoPE**, one head row at a time: `Qwen3_5RMSNorm`
//!    (`(1 + w)`, `modeling_qwen3_5.py:736-751`) then transformers'
//!    `apply_rotary_pos_emb` (`:570-605`) on the first `rotary_dim` dims,
//!    pairing `p` with `p + rotary_dim / 2`. At Qwen3.5-2B: head_dim 256,
//!    rotary_dim 64, theta 1e7.
//! 2. **The attention output gate** (`:714`): `y = o * sigmoid(gate)`.
//!
//! **The RoPE angle is formed in f32**, as transformers forms it (`inv_freq`
//! and `inv_freq * position` are float32 tensors, `:142-145,149-167`); only its
//! cos and sin are taken in f64. That is tessl's convention
//! (`tests/common/qwen35.rs:187-197`, `tests/qwen35_bwd.rs:779-785`): the f32
//! angle is the model's own arithmetic, and at position 20000 an f64 angle
//! differs from it by ~1e-3 rad. [`rope_angle_f32`] reproduces the formula
//! with Rust's f32 `powf`, which can differ from torch's f32 `pow` by an ulp,
//! so the norm/RoPE function takes the per-pair `(cos, sin)` as input and the
//! angle is a separate, separately checked step.
//!
//! The norm/RoPE backward is a port of tessl's in-test reference
//! (`tests/qwen35_bwd.rs:787-827`); the q/gate de-interleave of `q_proj` and
//! the fused projection windows are the kernel's layout, not this reference's.
//!
//! # Validation
//!
//! - **golden** (norm + RoPE given the angle, and its backward):
//!   `tests/fixtures/goldens/qk_norm_rope_*`, torch float64 with transformers'
//!   `apply_rotary_pos_emb` and angles from transformers' own
//!   `Qwen3_5TextRotaryEmbedding` buffer at the 2B config.
//! - **golden** (the whole forward, angle included, at positions 20000+):
//!   tessl's transformers fixture `qwen35_rope_*` within tessl's `5e-6`
//!   (`tests/qwen35_kernels.rs:192-212`).
//! - **golden** (output gate): `tests/fixtures/goldens/attn_output_gate_*`.
//! - **derivative**: central differences of both.

use super::gates_published::sigmoid;
use super::gdn_published::exact_f64;

/// `n` as f32, checked exact (positions and dims are far below 2^24).
fn exact_f32(n: usize) -> f32 {
    let wide = exact_f64(n);
    let narrow = wide as f32;
    assert!(f64::from(narrow) == wide, "{n} is not exact in f32");
    narrow
}

/// transformers' angle for pair `p` at `pos`, in f32:
/// `(1 / theta^(2p / rotary_dim)) * pos` (`modeling_qwen3_5.py:142-145,161`).
pub fn rope_angle_f32(p: usize, rotary_dim: usize, pos: usize, theta: f64) -> f32 {
    let theta32 = theta as f32;
    assert!(
        f64::from(theta32) == theta,
        "theta {theta} is not exact in f32"
    );
    let inv_freq = 1.0f32 / theta32.powf(exact_f32(2 * p) / exact_f32(rotary_dim));
    exact_f32(pos) * inv_freq
}

/// `(cos, sin)` in f64 of each pair's f32 angle at `pos`.
pub fn rope_cos_sin(rotary_dim: usize, pos: usize, theta: f64) -> Vec<(f64, f64)> {
    assert!(
        rotary_dim.is_multiple_of(2),
        "rotary_dim must be even, got {rotary_dim}"
    );
    (0..rotary_dim / 2)
        .map(|p| {
            let a = f64::from(rope_angle_f32(p, rotary_dim, pos, theta));
            (a.cos(), a.sin())
        })
        .collect()
}

fn check_row(x: &[f64], w: &[f64], cs: &[(f64, f64)]) {
    let d = x.len();
    assert!(d > 0, "norm_rope: empty row");
    assert_eq!(w.len(), d, "norm_rope: w must be [head_dim]");
    assert!(
        2 * cs.len() <= d,
        "norm_rope: rotary_dim {} exceeds head_dim {d}",
        2 * cs.len()
    );
    for (name, xs) in [("x", x), ("w", w)] {
        if let Some(i) = xs.iter().position(|v| !v.is_finite()) {
            panic!("norm_rope: {name}[{i}] is not finite");
        }
    }
}

/// One head row through the `(1 + w)` norm and partial RoPE. `cs` holds
/// `rotary_dim / 2` pairs `(cos, sin)`.
pub fn norm_rope_fwd_f64(x: &[f64], w: &[f64], cs: &[(f64, f64)], eps: f64) -> Vec<f64> {
    check_row(x, w, cs);
    let d = x.len();
    let rstd = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / exact_f64(d) + eps).sqrt();
    let mut n: Vec<f64> = x
        .iter()
        .zip(w)
        .map(|(v, wi)| v * rstd * (1.0 + wi))
        .collect();
    let half = cs.len();
    for (p, &(c, s)) in cs.iter().enumerate() {
        let (n0, n1) = (n[p], n[p + half]);
        n[p] = n0 * c - n1 * s;
        n[p + half] = n1 * c + n0 * s;
    }
    n
}

/// `(dx, dw)` of [`norm_rope_fwd_f64`] for the output gradient `g`. `dw` is this
/// row's contribution; a caller sums rows.
pub fn norm_rope_bwd_f64(
    x: &[f64],
    w: &[f64],
    g: &[f64],
    cs: &[(f64, f64)],
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    check_row(x, w, cs);
    let d = x.len();
    assert_eq!(g.len(), d, "norm_rope: g length");
    let dd = exact_f64(d);
    let rstd = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / dd + eps).sqrt();
    let mut dn = g.to_vec();
    let half = cs.len();
    for (p, &(c, s)) in cs.iter().enumerate() {
        dn[p] = g[p] * c + g[p + half] * s;
        dn[p + half] = g[p + half] * c - g[p] * s;
    }
    let xn: Vec<f64> = x.iter().map(|v| v * rstd).collect();
    let dxn: Vec<f64> = dn.iter().zip(w).map(|(a, wi)| a * (1.0 + wi)).collect();
    let m = dxn.iter().zip(&xn).map(|(a, b)| a * b).sum::<f64>() / dd;
    let dx = dxn
        .iter()
        .zip(&xn)
        .map(|(a, b)| rstd * (a - b * m))
        .collect();
    let dw = dn.iter().zip(&xn).map(|(a, b)| a * b).collect();
    (dx, dw)
}

/// `y = o * sigmoid(gate)`, elementwise.
pub fn output_gate_fwd_f64(o: &[f64], gate: &[f64]) -> Vec<f64> {
    assert_eq!(
        o.len(),
        gate.len(),
        "output_gate: o and gate lengths differ"
    );
    o.iter().zip(gate).map(|(a, g)| a * sigmoid(*g)).collect()
}

/// `(do, dgate)` of [`output_gate_fwd_f64`] for `dy`.
pub fn output_gate_bwd_f64(o: &[f64], gate: &[f64], dy: &[f64]) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        o.len(),
        gate.len(),
        "output_gate: o and gate lengths differ"
    );
    assert_eq!(dy.len(), o.len(), "output_gate: dy length");
    let mut d_o = Vec::with_capacity(o.len());
    let mut dgate = Vec::with_capacity(o.len());
    for ((a, g), d) in o.iter().zip(gate).zip(dy) {
        let s = sigmoid(*g);
        d_o.push(d * s);
        dgate.push(d * a * s * (1.0 - s));
    }
    (d_o, dgate)
}
