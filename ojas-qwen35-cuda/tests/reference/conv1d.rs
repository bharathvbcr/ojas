//! **K4: causal depthwise conv1d + SiLU**, forward and backward, f64, with a
//! zero initial state (the training forward). transformers
//! (`modeling_qwen3_5.py:390-397,497`): `nn.Conv1d(C, C, K, groups=C,
//! padding=K-1, bias=False)` truncated to the first `T` outputs, then SiLU.
//!
//! Layouts: `x`, `y`, `dx`, `dy` `[B, T, C]`; `w`, `dw` `[C, K]` (the conv
//! weight `[C, 1, K]` squeezed). `y[b, t, c] = silu(sum_j w[c, j] x[b, t + j - (K - 1), c])`
//! with out-of-range taps zero. A port of tessl's in-test reference
//! (`tests/qwen35_bwd.rs:553-592`); `dw` accumulates in ascending
//! `(b, t)` order.
//!
//! # Validation
//!
//! - **golden**: `tests/fixtures/goldens/conv1d_silu_*` (torch float64
//!   `F.conv1d` + `F.silu` and autograd), including `T < K - 1` across batch rows.
//! - **derivative**: central differences.

use super::gates_published::sigmoid;

pub fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

pub fn silu_grad(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

#[derive(Clone, Copy, Debug)]
pub struct ConvShape {
    pub b: usize,
    pub t: usize,
    pub c: usize,
    pub k: usize,
}

impl ConvShape {
    fn check(&self, x: &[f64], w: &[f64]) {
        assert!(self.k >= 1, "conv1d: kernel width must be at least 1");
        assert_eq!(
            x.len(),
            self.b * self.t * self.c,
            "conv1d: x length for {self:?}"
        );
        assert_eq!(w.len(), self.c * self.k, "conv1d: w length for {self:?}");
        for (name, xs) in [("x", x), ("w", w)] {
            if let Some(i) = xs.iter().position(|v| !v.is_finite()) {
                panic!("conv1d: {name}[{i}] is not finite");
            }
        }
    }

    /// The taps `j` that read a real token for output `t`, and that token's row.
    fn taps(&self, bi: usize, ti: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let hist = self.k - 1;
        (0..self.k)
            .filter(move |&j| ti + j >= hist)
            .map(move |j| (j, bi * self.t + ti + j - hist))
    }
}

/// `y [B, T, C]`.
pub fn conv1d_silu_fwd_f64(s: ConvShape, x: &[f64], w: &[f64]) -> Vec<f64> {
    s.check(x, w);
    let mut y = vec![0.0; s.b * s.t * s.c];
    for bi in 0..s.b {
        for ti in 0..s.t {
            for ci in 0..s.c {
                let pre: f64 = s
                    .taps(bi, ti)
                    .map(|(j, row)| w[ci * s.k + j] * x[row * s.c + ci])
                    .sum();
                y[(bi * s.t + ti) * s.c + ci] = silu(pre);
            }
        }
    }
    y
}

/// `(dx [B, T, C], dw [C, K])` for the upstream `dy`.
pub fn conv1d_silu_bwd_f64(s: ConvShape, x: &[f64], w: &[f64], dy: &[f64]) -> (Vec<f64>, Vec<f64>) {
    s.check(x, w);
    assert_eq!(dy.len(), x.len(), "conv1d: dy length");
    let (mut dx, mut dw) = (vec![0.0; x.len()], vec![0.0; w.len()]);
    for bi in 0..s.b {
        for ti in 0..s.t {
            for ci in 0..s.c {
                let pre: f64 = s
                    .taps(bi, ti)
                    .map(|(j, row)| w[ci * s.k + j] * x[row * s.c + ci])
                    .sum();
                let dpre = dy[(bi * s.t + ti) * s.c + ci] * silu_grad(pre);
                for (j, row) in s.taps(bi, ti) {
                    dx[row * s.c + ci] += dpre * w[ci * s.k + j];
                    dw[ci * s.k + j] += dpre * x[row * s.c + ci];
                }
            }
        }
    }
    (dx, dw)
}
