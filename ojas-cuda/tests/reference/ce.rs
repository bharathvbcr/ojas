//! **K10: cross-entropy over the supervised rows of a tied head**, loss and
//! gradients, f64. For supplied `(row, target)` pairs over hidden states
//! `h [T, H]` and the head `W [V, H]`:
//!
//! ```text
//! logits_i = W h[rows[i]]                     (the full [n, V] matrix: the reference can afford it)
//! S_i      = sum_{j != t} exp(logits_i[j] - logits_i[t])      (t = targets[i])
//! loss_i   = ln(1 + S_i)  =  logsumexp(logits_i) - logits_i[t]
//! loss     = mean_i loss_i  |  sum_i loss_i
//! dlogit_j = exp(logits_i[j] - logits_i[t]) / (1 + S_i) * scale / (n | 1)     (j != t)
//! dlogit_t = -S_i / (1 + S_i) * scale / (n | 1)
//! dh_i     = dlogit_i W        (per supplied row: a repeated row gets two)
//! dW       = sum_i dlogit_i^T h[rows[i]]
//! ```
//!
//! which is `F.cross_entropy` over those rows, and `dh`, `dW` are the gradients
//! of `scale * loss`. A port of tessl's in-test reference
//! (`tests/cross_entropy.rs:120-157`), dense hidden rows. The target is left
//! out of the sum, so no `softmax - onehot` subtraction is formed and the
//! target gradient is exactly the negation of the others' in f64, however far
//! the target leads (the textbook form's `p_t - 1` is 0 in f64 past a margin
//! of about 36). When another class leads the target by `d > 0` the same
//! expressions are formed with `exp(-d)` factored out, so nothing overflows.
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

/// One row's loss and its logit gradient before the `scale / n` factor, from
/// the target-excluded sum (see the module docs): `d` is how far the largest
/// other logit leads the target (0 when none does), `e_j = exp(l_j - l_t - d)`
/// for `j != t` and `S' = sum e_j`, so `S = exp(d) S'`, the loss is
/// `d + ln(exp(-d) + S')`, a non-target gradient `e_j / (exp(-d) + S')` and
/// the target's `-S' / (exp(-d) + S')`. A row with no other class is loss 0
/// and gradient 0.
fn excluded_row(logits: &[f64], t: usize) -> (f64, Vec<f64>) {
    let others = || logits.iter().enumerate().filter(|(j, _)| *j != t);
    let Some(lead) = others().map(|(_, &l)| l - logits[t]).reduce(f64::max) else {
        return (0.0, vec![0.0; logits.len()]);
    };
    let d = lead.max(0.0);
    let s_scaled: f64 = others().map(|(_, &l)| (l - logits[t] - d).exp()).sum();
    let denom = (-d).exp() + s_scaled;
    // With no lead the loss is ln(1 + S): `log1p` keeps it exact for a tiny S,
    // where `1 + S` has already lost S's low bits.
    let loss = if d == 0.0 {
        s_scaled.ln_1p()
    } else {
        d + denom.ln()
    };
    let probs = logits
        .iter()
        .enumerate()
        .map(|(j, &l)| {
            if j == t {
                -s_scaled / denom
            } else {
                (l - logits[t] - d).exp() / denom
            }
        })
        .collect();
    (loss, probs)
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
        let (loss, probs) = excluded_row(&logits, t);
        per_row.push(loss);
        for (j, &p) in probs.iter().enumerate() {
            let g = p * factor;
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
