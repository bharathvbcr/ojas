//! **K10: cross-entropy over the supervised rows of a tied head**
//! (`cuda-backend-scoping.md` §3 K10), with its gradients, **without ever
//! forming a `[rows, vocab]` matrix**: plans, CUDA-C source and the host
//! mirror of the whole walk. The device orchestration is
//! [`crate::ce_rows_cuda`].
//!
//! The walk is tessl's `cross_entropy_rows` (`tessl/src/cross_entropy.rs:1-36`,
//! `tessl/kernels/cross_entropy.metal`), at Qwen3.5's 248,320-row vocabulary
//! in [`CE_CHUNK`] = 8,192-column chunks (`tessl/src/qwen35_train.rs:56`). Like
//! tessl's, it leaves the target out of the running log-sum-exp, so neither
//! collapses past the f32 cliff. The arithmetic is this crate's own, not
//! "as tessl does": the scale is `exp(m - target) / (1 + s·exp(m - target))`
//! from the excluded max `m` and sum `s` ([`excluded_scale_f32`]), where
//! tessl forms `L = m + log s` in f32 and then `sigmoid(target - L)`. The two
//! are the same function and differ in rounding, so this walk is not
//! bit-identical to tessl's:
//!
//! 1. gather the `n` supervised rows of the hidden states into `[n, H]` f32
//!    (K0's `ce_gather_rows`);
//! 2. **first walk**, per chunk `[v0, v0 + w)`: `logits = h @ W[v0..v0+w]ᵀ`
//!    (K1 GEMM `nt`), then `qd_ce_lse_update_f32` folds the chunk into each
//!    row's running max `m` and running `s = sum(exp(logit - m))` with an
//!    **online log-sum-exp** (`m_new = max(m_old, chunk max)`,
//!    `s = s_old * exp(m_old - m_new) + sum(exp(logit - m_new))`), and picks up
//!    the target's logit when its chunk passes. The target column is left out
//!    of `m` and of `s`. The per-row loss is softplus of the negated margin,
//!    `ln(1 + s · exp(m - logit[target]))`, formed in f64 on the host from the
//!    three f32s. `s == 0` (one class, or a target-only chunk so far) is loss 0;
//! 3. **second walk** (with gradients): the chunk's logits are recomputed and
//!    `qd_ce_softmax_grad_f32` turns them, in place, into the other-class
//!    gradient `exp(logit - m) * scale'` and the target column `-s * scale'`
//!    (`scale' = scale / n` for the mean), so a row sums to 0 in f32; then
//!    `dh (+)= dlogits @ W_c` (K1 `nn`; the first chunk overwrites, the rest
//!    add) and `dW[v0..v0+w] = dlogitsᵀ @ h` (K1 `tn`). `dW` is overwritten;
//!    with the tied embedding, K9's backward adds onto it.
//!
//! Scratch is bounded by the chunk: `[n, chunk]` logits and `[n]` row state.
//! K1's GEMMs take buffer views, so the head chunk is read from `W` and the
//! `dW` rows are written in place, never copied (before, two `[chunk, H]`
//! scratch chunks and three chunk copies per vocabulary pass:
//! [`chunk_copy_bytes`]). A vocabulary that is not a multiple of the chunk
//! needs a second logits block at the tail width ([`CeRowsPlan::widths`]).
//!
//! **Numerics.** Every `exp` has a non-positive argument by construction:
//! the online update subtracts the new max, a non-target logit is at most
//! `m`, and [`excluded_scale_f32`] exponentiates `m - target` only when that
//! difference is not positive. They are `k8_act`'s `qd_exp_nonpos`, which has
//! a bit-identical host twin. `s` is the sum over the other classes, so a
//! one-class row has `s == 0` and is not treated as `ln(0)`. Every reduction
//! here has a fixed shape ([`crate::small_common`]), and the GEMMs' FFMA
//! engine is bit-identical to `host_ref::gemm_ffma_f32`. So [`ce_rows_mirror`]
//! is a **bitwise** mirror of the device walk on the FFMA engine (cuBLAS's
//! bf16 engine sums in its own order). A NaN or infinity anywhere makes a
//! row's loss non-finite, which the host refuses. An empty other-set does not.

use crate::error::CudaError;
use crate::gemm_plan::{GemmLayout, GemmShape};
use crate::host_ref::{gemm_ffma_f32, Operands};
use crate::k8_act::exp_nonpos_f32;
use crate::kernels::KernelModule;
use crate::small_common::{
    add, exact_len, mul, nonzero, row_max, row_sum, Window, NEG_FLT_MAX, ROW_THREADS,
};

/// tessl's vocabulary chunk for training (`tessl/src/qwen35_train.rs:56`).
pub const CE_CHUNK: u64 = 8192;

/// Entry: fold one chunk into the running log-sum-exp.
pub const CE_LSE_UPDATE: &str = "qd_ce_lse_update_f32";
/// Entry: one chunk's logits into `dlogits`, in place.
pub const CE_SOFTMAX_GRAD: &str = "qd_ce_softmax_grad_f32";

/// The K10 NVRTC module.
pub const MODULE: KernelModule = KernelModule {
    name: "k10_ce_rows",
    source: SOURCE,
    entries: &[CE_LSE_UPDATE, CE_SOFTMAX_GRAD],
};

const SOURCE: &str = concat!(
    crate::device_prelude!(),
    crate::act_prelude!(),
    crate::small_common::small_prelude!(),
    r#"
// One block per row n of a [n_rows, width] chunk (row stride ld) of the
// vocabulary columns [v0, v0 + width). first != 0 on the chunk at v0 = 0.
// The target column is left out of the max and the sum and is still stored
// in tlogit. The skip stays inside the element loops, before either block
// reduction, so every thread reaches the barriers.
extern "C" __global__ void qd_ce_lse_update_f32(
    const float* logits, float* m, float* s, float* tlogit, const unsigned int* targets,
    unsigned long long n_rows, unsigned long long width, unsigned long long ld,
    unsigned long long v0, int first)
{
    __shared__ float scratch[32];
    const unsigned long long n = blockIdx.x;
    if (n >= n_rows) {
        return;  // uniform per block
    }
    const float* row = logits + n * ld;
    const unsigned long long t = (unsigned long long)targets[n];
    float mx = QD_NEG_FLT_MAX;
    for (unsigned long long c = threadIdx.x; c < width; c += blockDim.x) {
        if (!(t >= v0 && t - v0 == c)) {
            mx = fmaxf(mx, row[c]);
        }
    }
    mx = qd_block_max(mx, scratch);
    const float m_old = first ? mx : m[n];
    const float m_new = fmaxf(m_old, mx);
    float acc = 0.0f;
    for (unsigned long long c = threadIdx.x; c < width; c += blockDim.x) {
        if (t >= v0 && t - v0 == c) {
            continue;
        }
        acc = acc + qd_exp_nonpos(row[c] - m_new);
    }
    acc = qd_block_sum(acc, scratch);
    if (threadIdx.x == 0) {
        const float carried = first ? 0.0f : s[n] * qd_exp_nonpos(m_old - m_new);
        s[n] = carried + acc;
        m[n] = m_new;
        if (t >= v0 && t - v0 < width) {
            tlogit[n] = row[t - v0];
        }
    }
}

// Unnormalized scale. Other classes are exp(logit - max_o) * scale and the
// target column is -sum_o * scale. Every exp argument is non-positive.
// sum_o == 0 (one class) returns 0; the caller then writes 0 and does not
// exponentiate the empty-max sentinel.
__device__ float qd_ce_excluded_scale(float max_o, float sum_o, float target)
{
    if (!(sum_o > 0.0f)) {
        return 0.0f;
    }
    const float delta = max_o - target;
    if (delta <= 0.0f) {
        const float e = qd_exp_nonpos(delta);
        const float q = sum_o * e;
        return e / (1.0f + q);
    }
    const float e = qd_exp_nonpos(-delta);
    return 1.0f / (e + sum_o);
}

// logits[n, c] becomes the target-excluded gradient times scale. tlogit is
// the target logit from the first walk, which this chunk may not hold.
extern "C" __global__ void qd_ce_softmax_grad_f32(
    float* logits, const float* m, const float* s, const float* tlogit,
    const unsigned int* targets,
    unsigned long long n_rows, unsigned long long width, unsigned long long ld,
    unsigned long long v0, float scale)
{
    const unsigned long long total = n_rows * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long n = i / width;
        const unsigned long long c = i - n * width;
        const unsigned long long at = n * ld + c;
        const float sum_o = s[n];
        if (!(sum_o > 0.0f)) {
            logits[at] = 0.0f;
        } else {
            const float sc = qd_ce_excluded_scale(m[n], sum_o, tlogit[n]) * scale;
            if (v0 + c == (unsigned long long)targets[n]) {
                logits[at] = -sum_o * sc;
            } else {
                logits[at] = qd_exp_nonpos(logits[at] - m[n]) * sc;
            }
        }
    }
}
"#
);

/// How the per-row losses combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Mean over the supplied rows (`reduction='mean'`).
    Mean,
    /// Sum over the supplied rows (`reduction='sum'`).
    Sum,
}

/// The cross-entropy of `n` supplied `(row, target)` pairs over hidden states
/// `h` (`t_rows` rows, hidden columns `[h.off, h.off + hidden)` of stride
/// `h.ld`) and a tied `[vocab, hidden]` head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeRowsPlan {
    /// Supplied rows.
    pub n: u64,
    /// Hidden size (2048 at Qwen3.5-2B).
    pub hidden: u64,
    /// Vocabulary (248,320 at Qwen3.5-2B).
    pub vocab: u64,
    /// Vocabulary columns per chunk ([`CE_CHUNK`] in training).
    pub chunk: u64,
    /// Rows of the hidden-state matrix.
    pub t_rows: u64,
    /// The hidden columns' window.
    pub h: Window,
    /// Mean or sum.
    pub reduction: Reduction,
}

impl CeRowsPlan {
    /// Validate the shape: every GEMM the walk runs is a valid
    /// [`GemmShape`], row indices and targets fit `u32`.
    pub fn new(
        (n, hidden, vocab): (u64, u64, u64),
        chunk: u64,
        (t_rows, h): (u64, Window),
        reduction: Reduction,
    ) -> Result<Self, CudaError> {
        const OP: &str = "cross_entropy_rows";
        nonzero(
            OP,
            "n (no supervised rows: an empty selection has no mean)",
            n,
        )?;
        nonzero(OP, "hidden", hidden)?;
        nonzero(OP, "vocab", vocab)?;
        nonzero(OP, "chunk", chunk)?;
        nonzero(OP, "t_rows", t_rows)?;
        if vocab > 1u64 << 32 || t_rows > 1u64 << 32 {
            return Err(CudaError::invalid(
                OP,
                "vocab and t_rows must fit u32 indices",
            ));
        }
        if add(h.off, hidden, OP)? > h.ld {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "hidden window {}+{hidden} is wider than its stride {}",
                    h.off, h.ld
                ),
            ));
        }
        mul(vocab, hidden, OP)?;
        let plan = CeRowsPlan {
            n,
            hidden,
            vocab,
            chunk,
            t_rows,
            h,
            reduction,
        };
        let us = |v: u64| {
            usize::try_from(v)
                .map_err(|_| CudaError::invalid(OP, format!("{v} does not fit usize")))
        };
        let (nn, hh) = (us(n)?, us(hidden)?);
        for w in plan.widths().into_iter().flatten() {
            let w = us(w)?;
            GemmShape::new(nn, w, hh)?;
            GemmShape::new(nn, hh, w)?;
            GemmShape::new(w, hh, nn)?;
        }
        Ok(plan)
    }

    /// `(v0, w)` of every chunk, ascending.
    pub fn chunks(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        (0..self.vocab)
            .step_by(usize::try_from(self.chunk).unwrap_or(usize::MAX))
            .map(|v0| (v0, self.chunk.min(self.vocab - v0)))
    }

    /// The (at most two) chunk widths: the full width, and the tail's when the
    /// vocabulary is not a multiple of the chunk.
    pub fn widths(&self) -> [Option<u64>; 2] {
        let full = self.chunk.min(self.vocab);
        let tail = (self.vocab > self.chunk && !self.vocab.is_multiple_of(self.chunk))
            .then_some(self.vocab % self.chunk);
        [Some(full), tail]
    }

    /// The gradient factor the softmax kernel multiplies by: `scale / n` (in
    /// f32, as tessl forms it, `cross_entropy.rs:325-328`) for the mean,
    /// `scale` for the sum. `scale` must be finite.
    pub fn grad_factor(&self, scale: f32) -> Result<f32, CudaError> {
        if !scale.is_finite() {
            return Err(CudaError::invalid(
                "cross_entropy_rows",
                "scale must be finite",
            ));
        }
        Ok(match self.reduction {
            Reduction::Mean => scale / self.n as f32,
            Reduction::Sum => scale,
        })
    }

    /// Check the supplied rows and targets.
    pub fn check_rows(&self, rows: &[u32], targets: &[u32]) -> Result<(), CudaError> {
        const OP: &str = "cross_entropy_rows";
        exact_len(OP, "rows", rows.len(), self.n)?;
        exact_len(OP, "targets", targets.len(), self.n)?;
        if let Some((i, &r)) = rows
            .iter()
            .enumerate()
            .find(|(_, &r)| u64::from(r) >= self.t_rows)
        {
            return Err(CudaError::invalid(
                OP,
                format!("rows[{i}] = {r} is past the {} hidden rows", self.t_rows),
            ));
        }
        if let Some((i, &t)) = targets
            .iter()
            .enumerate()
            .find(|(_, &t)| u64::from(t) >= self.vocab)
        {
            return Err(CudaError::invalid(
                OP,
                format!("targets[{i}] = {t} is past the vocabulary {}", self.vocab),
            ));
        }
        Ok(())
    }

    /// Check the hidden states and the head.
    pub fn check_inputs(&self, h: usize, w: usize) -> Result<(), CudaError> {
        const OP: &str = "cross_entropy_rows";
        self.h.check(OP, "h", self.t_rows, self.hidden, h)?;
        exact_len(OP, "weight", w, self.vocab * self.hidden)
    }

    /// Check the gradient outputs.
    pub fn check_grads(&self, dh: usize, dw: usize) -> Result<(), CudaError> {
        const OP: &str = "cross_entropy_rows";
        exact_len(OP, "dh", dh, self.n * self.hidden)?;
        exact_len(OP, "dw", dw, self.vocab * self.hidden)
    }
}

/// What a cross-entropy call returns.
#[derive(Clone, Debug, PartialEq)]
pub struct CeOutput {
    /// The reduced loss.
    pub loss: f64,
    /// Softplus of the negated margin for each supplied row. An empty
    /// other-set is 0.
    pub per_row: Vec<f64>,
}

/// `ln(1 + s · exp(m - t))` in f64, stable on both sides of the margin.
/// `s == 0` is 0 even when `m` is the empty-max sentinel. A non-finite input
/// or a negative sum is NaN.
fn excluded_loss_f64(max_o: f32, sum_o: f32, target: f32) -> f64 {
    if !max_o.is_finite() || !sum_o.is_finite() || !target.is_finite() || sum_o < 0.0 {
        return f64::NAN;
    }
    if sum_o == 0.0 {
        return 0.0;
    }
    let delta = f64::from(max_o) - f64::from(target);
    if delta <= 0.0 {
        (f64::from(sum_o) * delta.exp()).ln_1p()
    } else {
        delta + (f64::from(sum_o) + (-delta).exp()).ln()
    }
}

/// Host twin of `qd_ce_excluded_scale`. `sum_o == 0` is 0 and does not
/// exponentiate. No extra finite checks: a NaN target with a positive sum
/// falls through to `exp_nonpos_f32` and comes back NaN, matching the device.
fn excluded_scale_f32(max_o: f32, sum_o: f32, target: f32) -> f32 {
    if !(sum_o > 0.0) {
        return 0.0;
    }
    let delta = max_o - target;
    if delta <= 0.0 {
        let e = exp_nonpos_f32(delta);
        let q = sum_o * e;
        e / (1.0 + q)
    } else {
        let e = exp_nonpos_f32(-delta);
        1.0 / (e + sum_o)
    }
}

/// The per-row losses and their reduction from the walk's `(m, s, tlogit)`,
/// in f64. A non-finite row is refused, naming it. An empty other-set is 0.
pub fn losses(plan: &CeRowsPlan, m: &[f32], s: &[f32], t: &[f32]) -> Result<CeOutput, CudaError> {
    let per_row: Vec<f64> = m
        .iter()
        .zip(s)
        .zip(t)
        .map(|((&m, &s), &t)| excluded_loss_f64(m, s, t))
        .collect();
    if let Some(i) = per_row.iter().position(|l| !l.is_finite()) {
        return Err(CudaError::invalid(
            "cross_entropy_rows",
            format!("row {i}'s loss is not finite (non-finite hidden states or weights?)"),
        ));
    }
    let total: f64 = per_row.iter().sum();
    let loss = match plan.reduction {
        Reduction::Mean => total / per_row.len() as f64,
        Reduction::Sum => total,
    };
    Ok(CeOutput { loss, per_row })
}

/// `qd_ce_lse_update_f32` on the host for one chunk.
pub fn lse_update_mirror(
    logits: &[f32],
    (n_rows, width): (usize, usize),
    (m, s, tlogit): (&mut [f32], &mut [f32], &mut [f32]),
    targets: &[u32],
    (v0, first): (usize, bool),
) {
    let threads = ROW_THREADS as usize;
    for n in 0..n_rows {
        let row = &logits[n * width..(n + 1) * width];
        let t = targets[n] as usize;
        let skipped = |c: usize| t >= v0 && c == t - v0;
        // A skipped column contributes the running-max identity, which is
        // also `row_max`'s start value, so an all-skipped chunk stays there.
        let mx = row_max(threads, width, |c| {
            if skipped(c) {
                NEG_FLT_MAX
            } else {
                row[c]
            }
        });
        let m_old = if first { mx } else { m[n] };
        let m_new = m_old.max(mx);
        let acc = row_sum(threads, width, |c| {
            if skipped(c) {
                0.0
            } else {
                exp_nonpos_f32(row[c] - m_new)
            }
        });
        let carried = if first {
            0.0
        } else {
            s[n] * exp_nonpos_f32(m_old - m_new)
        };
        s[n] = carried + acc;
        m[n] = m_new;
        if t >= v0 && t - v0 < width {
            tlogit[n] = row[t - v0];
        }
    }
}

/// `qd_ce_softmax_grad_f32` on the host for one chunk, in place. `tlogit`
/// is the target logit recorded by the first walk.
pub fn softmax_grad_mirror(
    logits: &mut [f32],
    (n_rows, width): (usize, usize),
    (m, s, tlogit): (&[f32], &[f32], &[f32]),
    targets: &[u32],
    v0: usize,
    factor: f32,
) {
    for n in 0..n_rows {
        let sum_o = s[n];
        for c in 0..width {
            let at = n * width + c;
            if !(sum_o > 0.0) {
                logits[at] = 0.0;
                continue;
            }
            let sc = excluded_scale_f32(m[n], sum_o, tlogit[n]) * factor;
            if v0 + c == targets[n] as usize {
                logits[at] = -sum_o * sc;
            } else {
                logits[at] = exp_nonpos_f32(logits[at] - m[n]) * sc;
            }
        }
    }
}

/// What [`ce_rows_mirror`] returns.
#[derive(Clone, Debug, PartialEq)]
pub struct CeMirror {
    /// The loss and per-row losses.
    pub out: CeOutput,
    /// `[n, H]` (with gradients).
    pub dh: Option<Vec<f32>>,
    /// `[V, H]` (with gradients).
    pub dw: Option<Vec<f32>>,
}

/// The whole device walk on the host, chunk by chunk, with the GEMMs on
/// `host_ref::gemm_ffma_f32` (the FFMA engine's exact sequence).
/// `grad_scale` is the upstream gradient (`None`: forward only).
pub fn ce_rows_mirror(
    plan: &CeRowsPlan,
    h: &[f32],
    w: &[f32],
    (rows, targets): (&[u32], &[u32]),
    operands: Operands,
    grad_scale: Option<f32>,
) -> Result<CeMirror, CudaError> {
    plan.check_rows(rows, targets)?;
    plan.check_inputs(h.len(), w.len())?;
    let u = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
    let (n, hid) = (u(plan.n), u(plan.hidden));
    // 1. Gather.
    let mut hr = vec![0.0f32; n * hid];
    for (i, &r) in rows.iter().enumerate() {
        for c in 0..hid {
            hr[i * hid + c] = h[plan.h.at(r as usize, c)];
        }
    }
    let (mut m, mut s, mut t) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let logits_of = |v0: usize, wd: usize| -> Result<(Vec<f32>, &[f32]), CudaError> {
        let wc = &w[v0 * hid..(v0 + wd) * hid];
        let shape = GemmShape::new(n, wd, hid)?;
        Ok((
            gemm_ffma_f32(GemmLayout::Nt, shape, operands, &hr, wc, None),
            wc,
        ))
    };
    // 2. First walk.
    for (v0, wd) in plan.chunks() {
        let (v0, wd) = (u(v0), u(wd));
        let (logits, _) = logits_of(v0, wd)?;
        lse_update_mirror(
            &logits,
            (n, wd),
            (&mut m, &mut s, &mut t),
            targets,
            (v0, v0 == 0),
        );
    }
    let out = losses(plan, &m, &s, &t)?;
    let Some(scale) = grad_scale else {
        return Ok(CeMirror {
            out,
            dh: None,
            dw: None,
        });
    };
    // 3. Second walk.
    let factor = plan.grad_factor(scale)?;
    let mut dh = vec![0.0f32; n * hid];
    let mut dw = vec![0.0f32; u(plan.vocab) * hid];
    for (v0, wd) in plan.chunks() {
        let (v0, wd) = (u(v0), u(wd));
        let (mut logits, wc) = logits_of(v0, wd)?;
        softmax_grad_mirror(&mut logits, (n, wd), (&m, &s, &t), targets, v0, factor);
        let prev = (v0 != 0).then_some(dh.as_slice());
        dh = gemm_ffma_f32(
            GemmLayout::Nn,
            GemmShape::new(n, hid, wd)?,
            operands,
            &logits,
            wc,
            prev,
        );
        let dwc = gemm_ffma_f32(
            GemmLayout::Tn,
            GemmShape::new(wd, hid, n)?,
            operands,
            &logits,
            &hr,
            None,
        );
        dw[v0 * hid..(v0 + wd) * hid].copy_from_slice(&dwc);
    }
    Ok(CeMirror {
        out,
        dh: Some(dh),
        dw: Some(dw),
    })
}

/// Elements of device scratch one call holds for `plan`, the bound the
/// workspace and the per-call buffers stay within: logits at each width, the
/// gathered rows and the row state.
pub fn scratch_elements(plan: &CeRowsPlan) -> u64 {
    let widths: u64 = plan.widths().into_iter().flatten().sum();
    widths * plan.n + plan.n * plan.hidden + 3 * plan.n
}

/// Device bytes the K10 walk used to move through K0's `deliver` before its
/// GEMMs took views (now zero): with gradients, `W` was copied into chunk
/// scratch on both walks and `dW` copied out once, each copy one read and one
/// write of `vocab * hidden` f32s; without, `W` once. Arithmetic on the plan,
/// not a measurement.
pub fn chunk_copy_bytes(plan: &CeRowsPlan, with_grads: bool) -> u64 {
    let copies: u64 = if with_grads { 3 } else { 1 };
    copies * 2 * plan.vocab * plan.hidden * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::splitmix_f32;
    use crate::small_common::NEG_FLT_MAX;

    /// What the views removed, at Qwen3.5-2B's head (V = 248,320, H = 2048):
    /// three `[V, H]` f32 copies with gradients, ~12.2 GB of device traffic
    /// per call (arithmetic; the sm_90 measurement is the device bench's), and
    /// two `[8192, H]` scratch chunks per width from the workspace.
    #[test]
    fn views_remove_the_chunk_copies_and_their_scratch() {
        let p = plan(4, 2048, 248_320, CE_CHUNK);
        assert_eq!(chunk_copy_bytes(&p, true), 12_205_424_640);
        assert_eq!(chunk_copy_bytes(&p, false), 4_068_474_880);
        let widths: u64 = p.widths().into_iter().flatten().sum();
        assert_eq!(widths, 8192 + 2560);
        assert_eq!(
            scratch_elements(&p),
            widths * 4 + 4 * 2048 + 3 * 4,
            "no [chunk, H] weight or dW scratch"
        );
    }

    fn plan(n: u64, hidden: u64, vocab: u64, chunk: u64) -> CeRowsPlan {
        CeRowsPlan::new(
            (n, hidden, vocab),
            chunk,
            (6, Window::dense(hidden)),
            Reduction::Mean,
        )
        .unwrap()
    }

    #[test]
    fn chunks_cover_the_vocabulary_once_with_at_most_two_widths() {
        let p = plan(2, 8, 248_320, CE_CHUNK);
        let chunks: Vec<(u64, u64)> = p.chunks().collect();
        assert_eq!(chunks.len(), 31);
        assert_eq!(chunks[30], (245_760, 2560));
        assert_eq!(chunks.iter().map(|c| c.1).sum::<u64>(), 248_320);
        assert_eq!(p.widths(), [Some(8192), Some(2560)]);
        assert_eq!(plan(2, 8, 16_384, CE_CHUNK).widths(), [Some(8192), None]);
        assert_eq!(plan(2, 8, 300, CE_CHUNK).widths(), [Some(300), None]);
        assert_eq!(plan(2, 8, 300, 64).widths(), [Some(64), Some(44)]);
    }

    #[test]
    fn the_plan_refuses_empty_selections_bad_rows_targets_and_scales() {
        assert!(CeRowsPlan::new((0, 8, 10), 4, (6, Window::dense(8)), Reduction::Mean).is_err());
        assert!(CeRowsPlan::new((1, 8, 10), 0, (6, Window::dense(8)), Reduction::Mean).is_err());
        assert!(CeRowsPlan::new(
            (1, 8, 10),
            4,
            (6, Window { ld: 8, off: 1 }),
            Reduction::Mean
        )
        .is_err());
        let p = plan(2, 8, 10, 4);
        assert!(p.check_rows(&[0, 5], &[9, 0]).is_ok());
        assert!(p
            .check_rows(&[0, 6], &[9, 0])
            .unwrap_err()
            .to_string()
            .contains("rows[1] = 6"));
        assert!(p
            .check_rows(&[0, 5], &[10, 0])
            .unwrap_err()
            .to_string()
            .contains("targets[0] = 10"));
        assert!(p.check_rows(&[0], &[1]).is_err());
        assert!(p.grad_factor(f32::NAN).is_err());
        assert_eq!(p.grad_factor(1.0).unwrap(), 0.5);
    }

    #[test]
    fn the_online_lse_equals_one_pass_whatever_the_chunk() {
        // A dominant logit late in the vocabulary forces the carried sum to be
        // rescaled at a chunk boundary.
        let (n, hid, vocab) = (3u64, 16u64, 300u64);
        let h = splitmix_f32(11, 6 * 16, 1.0);
        let mut w = splitmix_f32(12, 300 * 16, 0.25);
        for k in 0..16 {
            w[299 * 16 + k] = 4.0 * h[2 * 16 + k];
        }
        let (rows, targets) = ([2u32, 0, 5], [299u32, 0, 150]);
        let one = ce_rows_mirror(
            &plan(n, hid, vocab, 300),
            &h,
            &w,
            (&rows, &targets),
            Operands::ExactF32,
            None,
        )
        .unwrap();
        for chunk in [7u64, 64, 299] {
            let c = ce_rows_mirror(
                &plan(n, hid, vocab, chunk),
                &h,
                &w,
                (&rows, &targets),
                Operands::ExactF32,
                None,
            )
            .unwrap();
            for (a, b) in c.out.per_row.iter().zip(&one.out.per_row) {
                assert!(
                    (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                    "chunk {chunk}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn non_finite_losses_are_refused_by_row() {
        let p = plan(2, 8, 10, 4);
        let e = losses(&p, &[1.0, f32::NAN], &[1.0, 1.0], &[0.0, 0.0]).unwrap_err();
        assert!(e.to_string().contains("row 1"), "{e}");
        // An empty excluded sum is the one-class identity, not ln(0).
        let empty = losses(&p, &[1.0, NEG_FLT_MAX], &[0.0, 1.0], &[4.0, 0.0]).unwrap();
        assert_eq!(empty.per_row[0], 0.0);
        // A NaN target is refused even when the other-set is empty.
        let nan_target = losses(&p, &[NEG_FLT_MAX, 1.0], &[0.0, 1.0], &[f32::NAN, 0.0]);
        assert!(nan_target.is_err(), "{nan_target:?}");
    }

    #[test]
    fn the_cuda_source_excludes_the_target_and_uses_the_crates_exp() {
        assert!(SOURCE
            .contains("const float carried = first ? 0.0f : s[n] * qd_exp_nonpos(m_old - m_new);"));
        assert!(SOURCE.contains("acc = acc + qd_exp_nonpos(row[c] - m_new);"));
        assert!(SOURCE.contains("qd_ce_excluded_scale"));
        assert!(SOURCE.contains("const float* tlogit"));
        assert!(
            !SOURCE.contains("const float lse = m[n] + qd_log(s[n]);"),
            "the max term of 1 is still inside the sum"
        );
        assert!(
            !SOURCE.contains("expf("),
            "libdevice expf is not host-reproducible"
        );
        assert!(
            !SOURCE.contains("logf("),
            "libdevice logf is not host-reproducible"
        );
    }

    /// `|got - want|` in units of the f32 spacing at `want`.
    fn ulps(got: f32, want: f64) -> f64 {
        let w = want as f32;
        let spacing = if w == 0.0 || !w.is_normal() {
            f64::from(f32::from_bits(1))
        } else {
            let next = f32::from_bits(w.abs().to_bits() + 1);
            f64::from(next) - f64::from(w.abs())
        };
        (f64::from(got) - want).abs() / spacing
    }

    /// Paper shape `[9, -9, -9]`, target 0. The margin is 18, past the f32
    /// cliff (~16.6), where `1 + exp(-m)` rounds to 1 and a `p - 1` target
    /// gradient is 0.
    fn cliff_oracle() -> (f64, [f64; 3]) {
        let z = [9.0f64, -9.0, -9.0];
        let max_o = z[1].max(z[2]);
        let sum_o = (z[1] - max_o).exp() + (z[2] - max_o).exp();
        let delta = max_o - z[0];
        let q = sum_o * delta.exp();
        let loss = q.ln_1p();
        let scale = delta.exp() / (1.0 + q);
        let g1 = (z[1] - max_o).exp() * scale;
        (loss, [-sum_o * scale, g1, g1])
    }

    fn assert_cliff(loss: f64, grad: &[f32]) {
        let (want_loss, want_g) = cliff_oracle();
        assert!(grad[0] != 0.0, "target gradient collapsed: {grad:?}");
        let sum: f64 = grad.iter().map(|g| f64::from(*g)).sum();
        let peak = grad.iter().fold(0.0f32, |m, g| m.max(g.abs()));
        let spacing = {
            let next = f32::from_bits(peak.to_bits() + 1);
            f64::from(next) - f64::from(peak)
        };
        assert!(
            sum.abs() <= 8.0 * spacing,
            "row sum {sum} of {grad:?} is more than 8 ulps of the peak {peak}"
        );
        assert!(
            ulps(loss as f32, want_loss) <= 16.0,
            "loss {loss} vs {want_loss}"
        );
        for (j, (&got, &want)) in grad.iter().zip(&want_g).enumerate() {
            let u = ulps(got, want);
            assert!(u <= 16.0, "grad[{j}] {got} vs {want} ({u} ulps)");
        }
    }

    #[test]
    fn a_row_past_the_fp32_cliff_keeps_a_nonzero_target_gradient() {
        let logits = [9.0f32, -9.0, -9.0];
        let targets = [0u32];
        let p = plan(1, 1, 3, 3);

        let mut m = [0.0f32; 1];
        let mut s = [0.0f32; 1];
        let mut t = [0.0f32; 1];
        lse_update_mirror(
            &logits,
            (1, 3),
            (&mut m, &mut s, &mut t),
            &targets,
            (0, true),
        );
        let wide = losses(&p, &m, &s, &t).unwrap();
        let mut grad = logits;
        softmax_grad_mirror(&mut grad, (1, 3), (&m, &s, &t), &targets, 0, 1.0);
        assert_cliff(wide.per_row[0], &grad);

        let mut m = [0.0f32; 1];
        let mut s = [0.0f32; 1];
        let mut t = [0.0f32; 1];
        for (i, &logit) in logits.iter().enumerate() {
            lse_update_mirror(
                &[logit],
                (1, 1),
                (&mut m, &mut s, &mut t),
                &targets,
                (i, i == 0),
            );
        }
        let chunked = losses(&p, &m, &s, &t).unwrap();
        let mut parts = [0.0f32; 3];
        for (i, &logit) in logits.iter().enumerate() {
            let mut chunk = [logit];
            softmax_grad_mirror(&mut chunk, (1, 1), (&m, &s, &t), &targets, i, 1.0);
            parts[i] = chunk[0];
        }
        assert_cliff(chunked.per_row[0], &parts);

        // One class: loss 0, gradient 0, and no exp against the empty max.
        let mut m = [0.0f32; 1];
        let mut s = [0.0f32; 1];
        let mut t = [0.0f32; 1];
        lse_update_mirror(&[4.0], (1, 1), (&mut m, &mut s, &mut t), &[0], (0, true));
        let one = losses(&plan(1, 1, 1, 1), &m, &s, &t).unwrap();
        assert_eq!(one.per_row[0], 0.0);
        let mut g = [4.0f32];
        softmax_grad_mirror(&mut g, (1, 1), (&m, &s, &t), &[0], 0, 1.0);
        assert_eq!(g[0], 0.0);
    }
}
