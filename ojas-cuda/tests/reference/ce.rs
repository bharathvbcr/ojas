//! **K10: cross-entropy over the supervised rows of a tied head**, loss and
//! gradients, f64. For supplied `(row, target)` pairs over hidden states
//! `h [T, H]` and the head `W [V, H]`:
//!
//! ```text
//! logits_i = W h[rows[i]]                     (the full [n, V] matrix: the reference can afford it)
//! loss_i   = softplus(-(max_{j≠t} z_j - z_t)) = ln(1 + sum_{j≠t} exp(z_j - z_t))
//! loss     = mean_i loss_i  |  sum_i loss_i
//! dlogit_j = exp(z_j - z_t) / (1 + q) * scale / (n | 1)    (j ≠ t, q = sum_{j≠t} exp(z_j - z_t))
//! dlogit_t = -sum_{j≠t} dlogit_j
//! dh_i     = dlogit_i W        (per supplied row: a repeated row gets two)
//! dW       = sum_i dlogit_i^T h[rows[i]]
//! ```
//!
//! which is `F.cross_entropy` over those rows, and `dh`, `dW` are the gradients
//! of `scale * loss`. A port of tessl's in-test reference
//! (`tests/cross_entropy.rs:120-157`), dense hidden rows.
//!
//! # Validation
//!
//! - **golden**: `tests/fixtures/goldens/ce_rows_*` (torch float64
//!   `F.cross_entropy` and autograd; mean and sum, a duplicated row, targets
//!   at both vocabulary ends, a planted dominant logit).
//! - **derivative**: central differences.

use super::gdn_published::exact_f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Mean,
    Sum,
}

pub struct CeRows {
    pub per_row: Vec<f64>,
    pub loss: f64,
    /// `[n, H]`, one row per supplied row.
    pub dh: Vec<f64>,
    /// `[V, H]`.
    pub dw: Vec<f64>,
}

/// `h [t_total, hidden]`, `w [vocab, hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn cross_entropy_rows_f64(
    h: &[f64],
    w: &[f64],
    rows: &[usize],
    targets: &[usize],
    hidden: usize,
    vocab: usize,
    reduction: Reduction,
    scale: f64,
) -> CeRows {
    assert!(
        hidden > 0 && vocab > 0,
        "ce: hidden and vocab must be non-zero"
    );
    assert_eq!(h.len() % hidden, 0, "ce: h must be [T, hidden]");
    assert_eq!(w.len(), vocab * hidden, "ce: w must be [vocab, hidden]");
    assert_eq!(rows.len(), targets.len(), "ce: one target per row");
    assert!(!rows.is_empty(), "ce: no supervised rows");
    let t_total = h.len() / hidden;
    if let Some(i) = rows.iter().position(|&r| r >= t_total) {
        panic!("ce: rows[{i}] = {} is not below T {t_total}", rows[i]);
    }
    if let Some(i) = targets.iter().position(|&t| t >= vocab) {
        panic!(
            "ce: targets[{i}] = {} is not below vocab {vocab}",
            targets[i]
        );
    }
    let n = rows.len();
    let factor = scale
        / match reduction {
            Reduction::Mean => exact_f64(n),
            Reduction::Sum => 1.0,
        };
    let mut per_row = Vec::with_capacity(n);
    let mut dh = vec![0.0; n * hidden];
    let mut dw = vec![0.0; vocab * hidden];
    for i in 0..n {
        let hr = &h[rows[i] * hidden..(rows[i] + 1) * hidden];
        let logits: Vec<f64> = (0..vocab)
            .map(|j| {
                hr.iter()
                    .zip(&w[j * hidden..(j + 1) * hidden])
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        let t = targets[i];
        let mut max_o = f64::NEG_INFINITY;
        for (j, &l) in logits.iter().enumerate() {
            if j != t && l > max_o {
                max_o = l;
            }
        }
        let mut sum_o = 0.0;
        if max_o.is_finite() {
            for (j, &l) in logits.iter().enumerate() {
                if j != t {
                    sum_o += (l - max_o).exp();
                }
            }
        }
        // Same two branches as the f32 kernels, in f64: every exp argument
        // is non-positive, and an empty other-set is loss 0 / gradient 0.
        let (loss_i, scale) = if sum_o == 0.0 {
            (0.0, 0.0)
        } else {
            let delta = max_o - logits[t];
            if delta <= 0.0 {
                let e = delta.exp();
                let q = sum_o * e;
                (q.ln_1p(), e / (1.0 + q))
            } else {
                let e = (-delta).exp();
                (delta + (sum_o + e).ln(), 1.0 / (e + sum_o))
            }
        };
        per_row.push(loss_i);
        for (j, &l) in logits.iter().enumerate() {
            let g = if sum_o == 0.0 {
                0.0
            } else if j == t {
                -sum_o * scale * factor
            } else {
                (l - max_o).exp() * scale * factor
            };
            let wr = &w[j * hidden..(j + 1) * hidden];
            for k in 0..hidden {
                dh[i * hidden + k] += g * wr[k];
                dw[j * hidden + k] += g * hr[k];
            }
        }
    }
    let total: f64 = per_row.iter().sum();
    let loss = match reduction {
        Reduction::Mean => total / exact_f64(n),
        Reduction::Sum => total,
    };
    CeRows {
        per_row,
        loss,
        dh,
        dw,
    }
}
