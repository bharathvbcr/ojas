//! **K10: cross-entropy over the supervised rows of a tied head**, loss and
//! gradients, f64. For supplied `(row, target)` pairs over hidden states
//! `h [T, H]` and the head `W [V, H]`:
//!
//! ```text
//! logits_i = W h[rows[i]]                     (the full [n, V] matrix: the reference can afford it)
//! loss_i   = logsumexp(logits_i) - logits_i[targets[i]]
//! loss     = mean_i loss_i  |  sum_i loss_i
//! dlogit   = (softmax(logits_i) - onehot) * scale / (n | 1)
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
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
        let t = targets[i];
        per_row.push(lse - logits[t]);
        for (j, &l) in logits.iter().enumerate() {
            let onehot = if j == t { 1.0 } else { 0.0 };
            let g = ((l - lse).exp() - onehot) * factor;
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
