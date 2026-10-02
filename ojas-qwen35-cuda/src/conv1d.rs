//! **K4: causal depthwise conv1d + SiLU**, forward and backward, from a zero
//! initial state (the training forward; `cuda-backend-scoping.md` §3 K4):
//! plans, CUDA-C source and host mirrors. The device launches are
//! [`crate::conv1d_cuda`].
//!
//! transformers (`modeling_qwen3_5.py:390-397,497`): `nn.Conv1d(C, C, K,
//! groups=C, padding=K-1, bias=False)` truncated to the first `T` outputs,
//! then SiLU. Per batch row `b`, channel `c`, token `t`:
//!
//! ```text
//! pre[t] = sum over j < K with t + j >= K - 1 of w[c, j] * x[b, t + j - (K - 1), c]
//! y[t]   = silu(pre[t])
//! ```
//!
//! Backward (tessl `qwen35_conv1d_silu_bwd_{dx,dw}_f32`,
//! `tessl/kernels/qwen35_bwd.metal:297-417`), with `pre` recomputed from `x`
//! (SiLU is not invertible):
//!
//! ```text
//! dpre[t] = dy[t] * silu'(pre[t])
//! dx[s]   = sum over j with s + K - 1 - j < T of w[c, j] * dy[t] * silu'(pre[t]),  t = s + K - 1 - j
//! dw[c,j] = sum over rows of dpre[t] * x[t + j - (K - 1)]
//! ```
//!
//! `dx` is one thread per element. `dw` is per block of
//! [`CONV_ROWS_PER_BLOCK`] flattened `B * T` rows into `part[blk, c * K + j]`,
//! then [`crate::small_common::COL_SUM`] over `C * K` columns leaves `dw` in
//! the weight's own `[C, K]` layout. No atomics.
//!
//! `x` is a [`Window`] (in training, the first `C` columns of the fused GDN
//! projection), `dy` and `dx` are windows (`dx` lands in the projection
//! gradient); `y` is dense `[B * T, C]`. One `qd_conv_pre` serves the forward
//! and both backward kernels, so all three see the same `pre` bits.
//!
//! SiLU and its derivative are L-cuda-M1's `k8_act` (the crate's one copy),
//! whose host emulations are bit-identical; the rest is `+` and `*`. So the
//! host mirrors here are **bitwise** mirrors.

use crate::error::CudaError;
use crate::k8_act::{silu_f32, silu_grad_f32};
use crate::kernels::KernelModule;
use crate::small_common::{blocks_for, col_sum_blocks, exact_len, mul, nonzero, Window};

/// Flattened rows per weight-gradient block (tessl `CONV_ROWS_PER_BLOCK`,
/// `tessl/src/qwen35_bwd.rs:352`).
pub const CONV_ROWS_PER_BLOCK: u64 = 256;

/// Longest conv kernel (tessl `CONV_MAX_KW`, `tessl/src/qwen35_bwd.rs:354`);
/// the backward keeps that many per-thread accumulators.
pub const CONV_MAX_KW: u32 = 8;

/// Shortest conv kernel the plan takes (tessl's forward range 2..=8).
pub const CONV_MIN_KW: u32 = 2;

/// Entry: the forward.
pub const CONV_FWD: &str = "qd_conv1d_silu_f32";
/// Entry: the input gradient.
pub const CONV_BWD_DX: &str = "qd_conv1d_silu_bwd_dx_f32";
/// Entry: the weight gradient's per-block partials.
pub const CONV_BWD_DW: &str = "qd_conv1d_silu_bwd_dw_f32";

/// The K4 NVRTC module.
pub const MODULE: KernelModule = KernelModule {
    name: "k4_conv1d",
    source: SOURCE,
    entries: &[CONV_FWD, CONV_BWD_DX, CONV_BWD_DW],
};

const SOURCE: &str = concat!(
    crate::device_prelude!(),
    crate::act_prelude!(),
    crate::small_common::small_prelude!(),
    r#"
#define QD_CONV_MAX_KW 8

// pre[t] for one channel of one batch row; xc points at that channel of the
// row's first token. Taps before the sequence start read the zero state and
// are skipped.
__device__ __forceinline__ float qd_conv_pre(
    const float* xc, unsigned long long ld_x, const float* wc, unsigned int kw,
    unsigned long long t)
{
    const unsigned long long hist = kw - 1u;
    float acc = 0.0f;
    for (unsigned int j = 0; j < kw; ++j) {
        const unsigned long long e = t + j;
        if (e >= hist) {
            acc = acc + wc[j] * xc[(e - hist) * ld_x];
        }
    }
    return acc;
}

// y[b*T + t, c] = silu(pre[t]), y dense [B*T, C].
extern "C" __global__ void qd_conv1d_silu_f32(
    const float* x, const float* w, float* y,
    unsigned long long batch, unsigned long long seq, unsigned long long channels,
    unsigned int kw, unsigned long long ld_x, unsigned long long x_off)
{
    const unsigned long long total = batch * seq * channels;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long row = i / channels;
        const unsigned long long c = i - row * channels;
        const unsigned long long b = row / seq;
        const unsigned long long t = row - b * seq;
        const float* xc = x + b * seq * ld_x + x_off + c;
        y[i] = qd_silu(qd_conv_pre(xc, ld_x, w + c * kw, kw, t));
    }
}

// dx[s] = sum over taps j of w[j] * dy[t] * silu'(pre[t]), t = s + K-1 - j < T.
extern "C" __global__ void qd_conv1d_silu_bwd_dx_f32(
    const float* x, const float* w, const float* dy, float* dx,
    unsigned long long batch, unsigned long long seq, unsigned long long channels,
    unsigned int kw,
    unsigned long long ld_x, unsigned long long x_off,
    unsigned long long ld_dy, unsigned long long dy_off,
    unsigned long long ld_dx, unsigned long long dx_off)
{
    const unsigned long long total = batch * seq * channels;
    const unsigned long long hist = kw - 1u;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long row = i / channels;
        const unsigned long long c = i - row * channels;
        const unsigned long long b = row / seq;
        const unsigned long long s = row - b * seq;
        const float* xc = x + b * seq * ld_x + x_off + c;
        const float* dyc = dy + b * seq * ld_dy + dy_off + c;
        const float* wc = w + c * kw;
        float acc = 0.0f;
        for (unsigned int j = 0; j < kw; ++j) {
            const unsigned long long t = s + hist - j;  // j <= hist, so no wrap
            if (t < seq) {
                const float pre = qd_conv_pre(xc, ld_x, wc, kw, t);
                acc = acc + wc[j] * dyc[t * ld_dy] * qd_silu_grad(pre);
            }
        }
        dx[row * ld_dx + dx_off + c] = acc;
    }
}

// part[blk, c*K + j] = sum over the block's rows of dpre[t] * x_ext[t + j].
extern "C" __global__ void qd_conv1d_silu_bwd_dw_f32(
    const float* x, const float* w, const float* dy, float* part,
    unsigned long long batch, unsigned long long seq, unsigned long long channels,
    unsigned int kw,
    unsigned long long ld_x, unsigned long long x_off,
    unsigned long long ld_dy, unsigned long long dy_off,
    unsigned long long rows_per_block)
{
    const unsigned long long rows = batch * seq;
    const unsigned long long nblocks = (rows + rows_per_block - 1) / rows_per_block;
    const unsigned long long total = nblocks * channels;
    const unsigned long long hist = kw - 1u;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long blk = i / channels;
        const unsigned long long c = i - blk * channels;
        const unsigned long long r0 = blk * rows_per_block;
        const unsigned long long r1 = rows - r0 < rows_per_block ? rows : r0 + rows_per_block;
        const float* wc = w + c * kw;
        float acc[QD_CONV_MAX_KW];
        for (int j = 0; j < QD_CONV_MAX_KW; ++j) {
            acc[j] = 0.0f;
        }
        for (unsigned long long row = r0; row < r1; ++row) {
            const unsigned long long b = row / seq;
            const unsigned long long t = row - b * seq;
            const float* xc = x + b * seq * ld_x + x_off + c;
            const float pre = qd_conv_pre(xc, ld_x, wc, kw, t);
            const float dpre = dy[row * ld_dy + dy_off + c] * qd_silu_grad(pre);
            for (unsigned int j = 0; j < kw; ++j) {
                const unsigned long long e = t + j;
                if (e >= hist) {
                    acc[j] = acc[j] + dpre * xc[(e - hist) * ld_x];
                }
            }
        }
        float* out = part + blk * (channels * kw) + c * kw;
        for (unsigned int j = 0; j < kw; ++j) {
            out[j] = acc[j];
        }
    }
}
"#
);

/// A causal conv over `batch` rows of `seq` tokens and `channels` channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dPlan {
    /// Batch rows (1 in tessl's training step: one sequence at a time).
    pub batch: u64,
    /// Tokens per batch row.
    pub seq: u64,
    /// Channels (`conv_dim`: 6144 at Qwen3.5-2B, q + k + v).
    pub channels: u64,
    /// Kernel width (4 at Qwen3.5-2B).
    pub kw: u32,
    /// Where `x` lives: `batch * seq` rows of `channels` columns.
    pub x: Window,
}

impl Conv1dPlan {
    /// Validate the shape and `x`'s stride.
    pub fn new(batch: u64, seq: u64, channels: u64, kw: u32, x: Window) -> Result<Self, CudaError> {
        const OP: &str = "conv1d_silu";
        nonzero(OP, "batch", batch)?;
        nonzero(OP, "seq", seq)?;
        nonzero(OP, "channels", channels)?;
        if !(CONV_MIN_KW..=CONV_MAX_KW).contains(&kw) {
            return Err(CudaError::invalid(
                OP,
                format!("kernel width {kw} is outside {CONV_MIN_KW}..={CONV_MAX_KW}"),
            ));
        }
        let rows = mul(batch, seq, OP)?;
        mul(rows, channels, OP)?;
        mul(channels, u64::from(kw), OP)?;
        if x.off.checked_add(channels).is_none_or(|end| end > x.ld) {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "x window {}+{channels} is wider than its stride {}",
                    x.off, x.ld
                ),
            ));
        }
        Ok(Conv1dPlan {
            batch,
            seq,
            channels,
            kw,
            x,
        })
    }

    /// Flattened rows, `batch * seq`.
    pub fn rows(&self) -> u64 {
        self.batch * self.seq
    }

    /// Elements of the dense `y`.
    pub fn y_len(&self) -> u64 {
        self.rows() * self.channels
    }

    /// Elements of `w` and `dw`, `[channels, kw]`.
    pub fn w_len(&self) -> u64 {
        self.channels * u64::from(self.kw)
    }

    /// Weight-gradient blocks.
    pub fn bwd_blocks(&self) -> u64 {
        blocks_for(self.rows(), CONV_ROWS_PER_BLOCK)
    }

    /// Elements of the `dw` partials.
    pub fn part_len(&self) -> u64 {
        self.bwd_blocks() * self.w_len()
    }

    /// Check the forward's buffers.
    pub fn check_fwd(&self, x: usize, w: usize, y: usize) -> Result<(), CudaError> {
        const OP: &str = "conv1d_silu";
        self.x.check(OP, "x", self.rows(), self.channels, x)?;
        exact_len(OP, "w", w, self.w_len())?;
        exact_len(OP, "y", y, self.y_len())
    }

    /// Check the backward's buffers; `dy` and `dx` are windows.
    pub fn check_bwd(
        &self,
        (x, w): (usize, usize),
        (dy, dx): ((Window, usize), (Window, usize)),
        (part, dw): (usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "conv1d_silu_bwd";
        self.x.check(OP, "x", self.rows(), self.channels, x)?;
        exact_len(OP, "w", w, self.w_len())?;
        dy.0.check(OP, "dy", self.rows(), self.channels, dy.1)?;
        dx.0.check(OP, "dx", self.rows(), self.channels, dx.1)?;
        exact_len(OP, "part", part, self.part_len())?;
        exact_len(OP, "dw", dw, self.w_len())
    }
}

struct Dims {
    seq: usize,
    channels: usize,
    kw: usize,
    rows: usize,
}

fn dims(plan: &Conv1dPlan) -> Dims {
    let u = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
    Dims {
        seq: u(plan.seq),
        channels: u(plan.channels),
        kw: plan.kw as usize,
        rows: u(plan.rows()),
    }
}

/// `qd_conv_pre` on the host: `pre` for flattened row `row`, channel `c`.
fn conv_pre(plan: &Conv1dPlan, d: &Dims, x: &[f32], w: &[f32], row: usize, c: usize) -> f32 {
    let (b, t) = (row / d.seq, row % d.seq);
    let hist = d.kw - 1;
    let mut acc = 0.0f32;
    for j in 0..d.kw {
        let e = t + j;
        if e >= hist {
            acc += w[c * d.kw + j] * x[plan.x.at(b * d.seq + e - hist, c)];
        }
    }
    acc
}

/// `qd_conv1d_silu_f32` on the host: the dense `y`.
pub fn conv1d_silu_fwd_mirror(
    plan: &Conv1dPlan,
    x: &[f32],
    w: &[f32],
) -> Result<Vec<f32>, CudaError> {
    let d = dims(plan);
    plan.check_fwd(x.len(), w.len(), d.rows * d.channels)?;
    let mut y = vec![0.0f32; d.rows * d.channels];
    for row in 0..d.rows {
        for c in 0..d.channels {
            y[row * d.channels + c] = silu_f32(conv_pre(plan, &d, x, w, row, c));
        }
    }
    Ok(y)
}

/// The backward's two kernels and the column sum on the host: writes `dx`
/// into its window and returns `dw` `[channels, kw]`.
pub fn conv1d_silu_bwd_mirror(
    plan: &Conv1dPlan,
    (x, w): (&[f32], &[f32]),
    (dy_win, dy): (Window, &[f32]),
    (dx_win, dx): (Window, &mut [f32]),
) -> Result<Vec<f32>, CudaError> {
    let d = dims(plan);
    let part_len = usize::try_from(plan.part_len()).unwrap_or(usize::MAX);
    plan.check_bwd(
        (x.len(), w.len()),
        ((dy_win, dy.len()), (dx_win, dx.len())),
        (part_len, w.len()),
    )?;
    let hist = d.kw - 1;
    let dy_at = |row: usize, c: usize| dy[dy_win.at(row, c)];
    for row in 0..d.rows {
        let (b, s) = (row / d.seq, row % d.seq);
        for c in 0..d.channels {
            let mut acc = 0.0f32;
            for j in 0..d.kw {
                let t = s + hist - j;
                if t < d.seq {
                    let pre = conv_pre(plan, &d, x, w, b * d.seq + t, c);
                    acc += w[c * d.kw + j] * dy_at(b * d.seq + t, c) * silu_grad_f32(pre);
                }
            }
            dx[dx_win.at(row, c)] = acc;
        }
    }
    let rpb = usize::try_from(CONV_ROWS_PER_BLOCK).unwrap_or(usize::MAX);
    let nblocks = d.rows.div_ceil(rpb);
    let width = d.channels * d.kw;
    let mut part = vec![0.0f32; part_len];
    for blk in 0..nblocks {
        for c in 0..d.channels {
            let mut acc = [0.0f32; CONV_MAX_KW as usize];
            for row in blk * rpb..d.rows.min(blk * rpb + rpb) {
                let (b, t) = (row / d.seq, row % d.seq);
                let pre = conv_pre(plan, &d, x, w, row, c);
                let dpre = dy_at(row, c) * silu_grad_f32(pre);
                for (j, a) in acc.iter_mut().enumerate().take(d.kw) {
                    let e = t + j;
                    if e >= hist {
                        *a += dpre * x[plan.x.at(b * d.seq + e - hist, c)];
                    }
                }
            }
            part[blk * width + c * d.kw..blk * width + (c + 1) * d.kw]
                .copy_from_slice(&acc[..d.kw]);
        }
    }
    Ok(col_sum_blocks(&part, 0, nblocks, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::splitmix_f32;

    #[test]
    fn the_plan_refuses_bad_widths_and_windows() {
        assert!(Conv1dPlan::new(1, 9, 6, 1, Window::dense(6)).is_err());
        assert!(Conv1dPlan::new(1, 9, 6, 9, Window::dense(6)).is_err());
        assert!(Conv1dPlan::new(1, 9, 6, 4, Window { ld: 6, off: 1 }).is_err());
        assert!(Conv1dPlan::new(0, 9, 6, 4, Window::dense(6)).is_err());
        let p = Conv1dPlan::new(2, 300, 3, 4, Window { ld: 10, off: 2 }).unwrap();
        assert_eq!((p.rows(), p.bwd_blocks(), p.part_len()), (600, 3, 36));
        // x holds (600 - 1) * 10 + 2 + 3 = 5995.
        assert!(p.check_fwd(5995, 12, 1800).is_ok());
        assert!(p.check_fwd(5994, 12, 1800).is_err());
    }

    #[test]
    fn a_batch_rows_gradient_does_not_reach_its_neighbour() {
        // T = 2 < K - 1: each batch row sees only its own two tokens.
        let plan = Conv1dPlan::new(2, 2, 1, 4, Window::dense(1)).unwrap();
        let x = splitmix_f32(1, 4, 1.0);
        let w = splitmix_f32(2, 4, 1.0);
        let mut dy = vec![0.0f32; 4];
        dy[3] = 1.0; // only batch row 1, token 1
        let mut dx = vec![9.0f32; 4];
        conv1d_silu_bwd_mirror(
            &plan,
            (&x, &w),
            (Window::dense(1), &dy),
            (Window::dense(1), &mut dx),
        )
        .unwrap();
        assert_eq!(&dx[..2], &[0.0, 0.0], "batch row 0 got gradient from row 1");
        assert!(dx[2] != 0.0 && dx[3] != 0.0);
    }

    #[test]
    fn the_cuda_source_skips_the_zero_state_taps_and_recomputes_pre() {
        assert!(SOURCE
            .contains("if (e >= hist) {\n            acc = acc + wc[j] * xc[(e - hist) * ld_x];"));
        assert!(SOURCE.contains("acc = acc + wc[j] * dyc[t * ld_dy] * qd_silu_grad(pre);"));
        assert_eq!(SOURCE.matches("qd_conv_pre(xc, ld_x").count(), 3);
    }
}
