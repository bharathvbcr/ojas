//! **K6: the attention pieces around the core** (`cuda-backend-scoping.md`
//! §3 K6): the q/k norm with partial RoPE, and the output gate, forward and
//! backward. Plans, CUDA-C source and host mirrors; the device launches are
//! [`crate::qk_norm_rope_cuda`].
//!
//! # The fused projection layout (Qwen3.5's, tessl `qwen35_attn.metal:1-30`)
//!
//! Row `r = b*T + t` of the attention projection `p` (row stride `ld_p`) holds:
//! - query head `h` at columns `q_off + h*2D .. + D`, its **gate** right after
//!   at `q_off + h*2D + D .. + D` (`q_proj` is twice as wide as the output);
//! - key head `h` at `k_off + h*D`, value head `h` at `v_off + h*D`.
//!
//! # q/k norm + partial RoPE
//!
//! Per head row: `n = x * rstd * (1 + w)` (`Qwen3_5RMSNorm`), then
//! transformers' `apply_rotary_pos_emb` on the first `rotary_dim` dims, pairing
//! `p` with `p + rotary_dim/2` (`modeling_qwen3_5.py:570-605`):
//! `out[p] = n0 c - n1 s`, `out[p + half] = n1 c + n0 s`. Token `t` of each
//! batch row is at position `t` (training). The forward writes dense
//! `q [B*T, Hq, D]`, `k`, `v [B*T, Hkv, D]` (v copied). The backward (tessl
//! `qwen35_attn_qk_norm_rope_bwd_f32`, `qwen35_bwd.metal:419-582`) writes `dq`,
//! `dk`, `dv` back into the q, k, v columns of the projection gradient `dp`
//! (the gate columns are the output gate's), and the two norm weights'
//! gradients through per-block partials ([`QK_ROWS_PER_BLOCK`] tokens) and
//! [`crate::small_common::COL_SUM`].
//!
//! **The angle is taken from the host** (decided before any run; lead's
//! approval 2026-10-01). [`rope_inv_freq`] computes transformers' `inv_freq`
//! (`1 / theta^(2p / rotary_dim)` in f32, `modeling_qwen3_5.py:142-145`) on
//! the host, exactly as L-cuda-oracle's `rope_angle_f32` does; the kernel's
//! only angle operation is `angle = (float)pos * inv_freq[p]`, one correctly
//! rounded f32 multiply, so the device angle is **bit-identical** to the
//! oracle's f32 angle. A device `powf` (4 ulp, CUDA Programming Guide 13.4.2
//! §5.5) would be a third angle implementation that moves the angle at
//! position 20000 by ~1e-3 rad (`GAP-L-CUDA-ORACLE-ROPE-DEVICE-POW-2026-10-01`);
//! with the host table, tessl's 4e-3 bound is not needed and the forward is
//! held to tessl's `2e-5` absolute (`tessl/tests/qwen35_kernels.rs:1556,1562`).
//! `cosf` and `sinf` stay libdevice (2 ulp each, same table), the only
//! libdevice transcendentals left in this lane's kernels, so the q/k mirrors
//! are not bitwise.
//!
//! # Output gate
//!
//! `y = o * sigmoid(gate)` (`modeling_qwen3_5.py:714`), gate read in place
//! from `p`; backward `d_o = dy * s`, `d_gate = dy * o * s * (1 - s)` into the
//! gate columns of `dp` (tessl `qwen35_attn_gate_bwd_f32`,
//! `qwen35_bwd.metal:262-295`). Sigmoid is `k8_act`'s, so the gate mirrors
//! are bitwise.

use crate::error::CudaError;
use crate::geometry::MAX_GRID_X;
use crate::k8_act::sigmoid_f32;
use crate::kernels::KernelModule;
use crate::small_common::{
    add, blocks_for, col_sum_blocks, cols_overlap, exact_in_f32, exact_len, mul, nonzero,
    positive_eps, unit_sum, Window, MAX_COLS, UNIT_WARPS, WARP,
};

/// Tokens per block of the q/k backward (tessl `QK_ROWS_PER_BLOCK`,
/// `tessl/src/qwen35_bwd.rs:456`).
pub const QK_ROWS_PER_BLOCK: u64 = 16;

/// Widest head the warp-per-unit q/k backward takes.
pub const QK_BWD_MAX_DIM: u64 = WARP as u64 * MAX_COLS;

/// Entry: q/k norm + RoPE forward.
pub const QK_FWD: &str = "qd_attn_qk_norm_rope_f32";
/// Entry: q/k norm + RoPE backward.
pub const QK_BWD: &str = "qd_attn_qk_norm_rope_bwd_f32";
/// Entry: output gate forward.
pub const GATE_FWD: &str = "qd_attn_output_gate_f32";
/// Entry: output gate backward.
pub const GATE_BWD: &str = "qd_attn_output_gate_bwd_f32";

/// The K6 NVRTC module.
pub const MODULE: KernelModule = KernelModule {
    name: "k6_qk_norm_rope",
    source: SOURCE,
    entries: &[QK_FWD, QK_BWD, GATE_FWD, GATE_BWD],
};

const SOURCE: &str = concat!(
    crate::device_prelude!(),
    crate::act_prelude!(),
    crate::small_common::small_prelude!(),
    r#"
#define QD_K6_MAX_COLS 16

// One head row: n = x rstd (1 + w), then RoPE on the first `rot` dims. The
// angle is pos * inv_freq[p], inv_freq from the host.
__device__ __forceinline__ void qd_qk_norm_rope_row(
    const float* src, const float* weight, const float* inv_freq, float* dst,
    unsigned long long dim, unsigned long long rot, float pos, float eps, unsigned int lane)
{
    float ss = 0.0f;
    for (unsigned long long d = lane; d < dim; d += 32) {
        ss = ss + src[d] * src[d];
    }
    ss = qd_warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    const unsigned long long half = rot / 2;
    for (unsigned long long p = lane; p < half; p += 32) {
        const float x0 = src[p] * inv * (1.0f + weight[p]);
        const float x1 = src[p + half] * inv * (1.0f + weight[p + half]);
        const float angle = pos * inv_freq[p];
        const float c = cosf(angle);
        const float s = sinf(angle);
        dst[p] = x0 * c - x1 * s;
        dst[p + half] = x1 * c + x0 * s;
    }
    for (unsigned long long d = rot + lane; d < dim; d += 32) {
        dst[d] = src[d] * inv * (1.0f + weight[d]);
    }
}

// One warp per (token, head) over Hq query, Hkv key and Hkv value heads.
extern "C" __global__ void qd_attn_qk_norm_rope_f32(
    const float* p, const float* q_norm_w, const float* k_norm_w, const float* inv_freq,
    float* q_out, float* k_out, float* v_out,
    unsigned long long batch, unsigned long long seq, unsigned long long hq,
    unsigned long long hkv, unsigned long long dim, unsigned long long rot,
    unsigned long long ld_p, unsigned long long q_off, unsigned long long k_off,
    unsigned long long v_off, float eps)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned long long heads = hq + 2 * hkv;
    const unsigned long long unit =
        (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (unit >= batch * seq * heads) {
        return;  // uniform per warp; this kernel has no block barrier
    }
    const unsigned long long r = unit / heads;
    const unsigned long long j = unit - r * heads;
    const float pos = (float)(r % seq);
    const float* row = p + r * ld_p;
    if (j < hq) {
        qd_qk_norm_rope_row(row + q_off + j * 2 * dim, q_norm_w, inv_freq,
                            q_out + (r * hq + j) * dim, dim, rot, pos, eps, lane);
    } else if (j < hq + hkv) {
        const unsigned long long h = j - hq;
        qd_qk_norm_rope_row(row + k_off + h * dim, k_norm_w, inv_freq,
                            k_out + (r * hkv + h) * dim, dim, rot, pos, eps, lane);
    } else {
        const unsigned long long h = j - hq - hkv;
        const float* src = row + v_off + h * dim;
        float* dst = v_out + (r * hkv + h) * dim;
        for (unsigned long long d = lane; d < dim; d += 32) {
            dst[d] = src[d];
        }
    }
}

// Backward of one head row: g is d(out). Writes d(src) to dst and adds this
// row's (1 + w) gradient into acc (lane's columns lane, lane + 32, ...).
//   dn = R(pos)^T g on the rotary dims, g elsewhere
//   dx = rstd (dn (1 + w) - xn mean(dn (1 + w) xn)),  xn = x rstd;  dw += dn xn
__device__ __forceinline__ void qd_qk_norm_rope_row_bwd(
    const float* src, const float* weight, const float* inv_freq, const float* g,
    float* dst, unsigned long long dim, unsigned long long rot, float pos, float eps,
    unsigned int lane, float* acc)
{
    float ss = 0.0f;
    for (unsigned long long d = lane; d < dim; d += 32) {
        ss = ss + src[d] * src[d];
    }
    const float rstd = 1.0f / sqrtf(qd_warp_sum(ss) / (float)dim + eps);
    const unsigned long long half = rot / 2;
    float dn[QD_K6_MAX_COLS];
    float dot = 0.0f;
    int k = 0;
    for (unsigned long long d = lane; d < dim; d += 32, ++k) {
        float v;
        if (d < rot) {
            const unsigned long long pp = d < half ? d : d - half;
            const float angle = pos * inv_freq[pp];
            const float c = cosf(angle);
            const float s = sinf(angle);
            v = d < half ? g[pp] * c + g[pp + half] * s : g[d] * c - g[pp] * s;
        } else {
            v = g[d];
        }
        dn[k] = v;
        dot = dot + v * (1.0f + weight[d]) * src[d] * rstd;
    }
    const float m = qd_warp_sum(dot) / (float)dim;
    k = 0;
    for (unsigned long long d = lane; d < dim; d += 32, ++k) {
        const float xn = src[d] * rstd;
        dst[d] = rstd * (dn[k] * (1.0f + weight[d]) - xn * m);
        acc[k] = acc[k] + dn[k] * xn;
    }
}

// Blocks of rows_per_block tokens, 4 warps striding through the block's
// (token, head) units. part[blk, :] is the q norm's block sum,
// part[nblocks + blk, :] the k norm's, warps combined in order.
extern "C" __global__ void qd_attn_qk_norm_rope_bwd_f32(
    const float* p, const float* q_norm_w, const float* k_norm_w, const float* inv_freq,
    const float* dq, const float* dk, const float* dv, float* dp, float* part,
    unsigned long long batch, unsigned long long seq, unsigned long long hq,
    unsigned long long hkv, unsigned long long dim, unsigned long long rot,
    unsigned long long ld_p, unsigned long long q_off, unsigned long long k_off,
    unsigned long long v_off, float eps, unsigned long long rows_per_block)
{
    __shared__ float warp_acc[2 * 4 * QD_K6_MAX_COLS * 32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int nwarps = blockDim.x >> 5;
    const unsigned long long rows = batch * seq;
    const unsigned long long nblocks = (rows + rows_per_block - 1) / rows_per_block;
    const unsigned long long blk = blockIdx.x;
    const unsigned long long r0 = blk * rows_per_block;
    if (r0 >= rows) {
        return;  // uniform per block
    }
    const unsigned long long r1 = rows - r0 < rows_per_block ? rows : r0 + rows_per_block;
    const unsigned long long heads = hq + 2 * hkv;
    float acc_q[QD_K6_MAX_COLS];
    float acc_k[QD_K6_MAX_COLS];
    for (int k = 0; k < QD_K6_MAX_COLS; ++k) {
        acc_q[k] = 0.0f;
        acc_k[k] = 0.0f;
    }
    for (unsigned long long u = r0 * heads + warp; u < r1 * heads; u += nwarps) {
        const unsigned long long r = u / heads;
        const unsigned long long j = u - r * heads;
        const float pos = (float)(r % seq);
        const float* row = p + r * ld_p;
        float* drow = dp + r * ld_p;
        if (j < hq) {
            const unsigned long long col = q_off + j * 2 * dim;
            qd_qk_norm_rope_row_bwd(row + col, q_norm_w, inv_freq, dq + (r * hq + j) * dim,
                                    drow + col, dim, rot, pos, eps, lane, acc_q);
        } else if (j < hq + hkv) {
            const unsigned long long h = j - hq;
            const unsigned long long col = k_off + h * dim;
            qd_qk_norm_rope_row_bwd(row + col, k_norm_w, inv_freq, dk + (r * hkv + h) * dim,
                                    drow + col, dim, rot, pos, eps, lane, acc_k);
        } else {
            const unsigned long long h = j - hq - hkv;
            const float* src = dv + (r * hkv + h) * dim;
            float* out = drow + v_off + h * dim;
            for (unsigned long long d = lane; d < dim; d += 32) {
                out[d] = src[d];
            }
        }
    }
    const unsigned long long per = (dim + 31) / 32;
    const unsigned long long half_acc = 4 * QD_K6_MAX_COLS * 32;
    for (unsigned long long k = 0; k < per; ++k) {
        warp_acc[(warp * per + k) * 32 + lane] = acc_q[k];
        warp_acc[half_acc + (warp * per + k) * 32 + lane] = acc_k[k];
    }
    __syncthreads();
    if (warp == 0u) {
        for (unsigned long long k = 0; k < per; ++k) {
            const unsigned long long d = lane + 32 * k;
            if (d < dim) {
                float sq = 0.0f;
                float sk = 0.0f;
                for (unsigned int g = 0; g < nwarps; ++g) {
                    sq = sq + warp_acc[(g * per + k) * 32 + lane];
                    sk = sk + warp_acc[half_acc + (g * per + k) * 32 + lane];
                }
                part[blk * dim + d] = sq;
                part[(nblocks + blk) * dim + d] = sk;
            }
        }
    }
}

// out[r, out_off + col] = attn[r, col] * sigmoid(gate), the gate of head
// h = col / D at p[r, q_off + h*2D + D + col % D].
extern "C" __global__ void qd_attn_output_gate_f32(
    const float* attn, const float* p, float* out,
    unsigned long long rows, unsigned long long hq, unsigned long long dim,
    unsigned long long ld_p, unsigned long long q_off,
    unsigned long long ld_out, unsigned long long out_off)
{
    const unsigned long long width = hq * dim;
    const unsigned long long total = rows * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / width;
        const unsigned long long col = i - r * width;
        const unsigned long long h = col / dim;
        const unsigned long long d = col - h * dim;
        const float g = p[r * ld_p + q_off + h * 2 * dim + dim + d];
        out[r * ld_out + out_off + col] = attn[i] * qd_sigmoid(g);
    }
}

// d_attn = dy s; dp[gate] = dy attn s (1 - s).
extern "C" __global__ void qd_attn_output_gate_bwd_f32(
    const float* attn, const float* p, const float* dy, float* d_attn, float* dp,
    unsigned long long rows, unsigned long long hq, unsigned long long dim,
    unsigned long long ld_p, unsigned long long q_off,
    unsigned long long ld_dy, unsigned long long dy_off)
{
    const unsigned long long width = hq * dim;
    const unsigned long long total = rows * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / width;
        const unsigned long long col = i - r * width;
        const unsigned long long h = col / dim;
        const unsigned long long d = col - h * dim;
        const unsigned long long gi = r * ld_p + q_off + h * 2 * dim + dim + d;
        const float s = qd_sigmoid(p[gi]);
        const float a = attn[i];
        const float g = dy[r * ld_dy + dy_off + col];
        d_attn[i] = g * s;
        dp[gi] = g * a * s * (1.0f - s);
    }
}
"#
);

/// transformers' `inv_freq` on the host, in f32: `1 / theta^(2p / rotary_dim)`
/// for `p < rotary_dim / 2` (`modeling_qwen3_5.py:142-145`), the same
/// arithmetic as L-cuda-oracle's `rope_angle_f32`. `theta` must be exact in
/// f32 (1e7 at Qwen3.5-2B is).
pub fn rope_inv_freq(rotary_dim: u64, theta: f64) -> Result<Vec<f32>, CudaError> {
    const OP: &str = "rope_inv_freq";
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(CudaError::invalid(
            OP,
            format!("rotary_dim must be even and non-zero, got {rotary_dim}"),
        ));
    }
    exact_in_f32(OP, "rotary_dim", rotary_dim)?;
    let theta32 = theta as f32;
    if !(theta.is_finite() && theta > 0.0 && f64::from(theta32) == theta) {
        return Err(CudaError::invalid(
            OP,
            format!("theta {theta} must be positive, finite and exact in f32"),
        ));
    }
    let rot = rotary_dim as f32;
    Ok((0..rotary_dim / 2)
        .map(|p| 1.0f32 / theta32.powf((2 * p) as f32 / rot))
        .collect())
}

/// The q/k norm + RoPE over one batch of the fused projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QkNormRopePlan {
    /// Batch rows (1 in tessl's training step).
    pub batch: u64,
    /// Tokens per batch row; token `t` is at position `t`.
    pub seq: u64,
    /// Query heads (8 at Qwen3.5-2B).
    pub hq: u64,
    /// Key/value heads (2 at Qwen3.5-2B).
    pub hkv: u64,
    /// Head dim (256 at Qwen3.5-2B).
    pub dim: u64,
    /// Rotated dims (64 = 0.25 * 256 at Qwen3.5-2B).
    pub rot: u64,
    /// `rms_norm_eps`.
    pub eps: f32,
    /// Row stride of `p` and `dp`.
    pub ld_p: u64,
    /// First column of query head 0 (each head is `2D` wide: query, then gate).
    pub q_off: u64,
    /// First column of key head 0.
    pub k_off: u64,
    /// First column of value head 0.
    pub v_off: u64,
}

impl QkNormRopePlan {
    /// Validate the shape and the projection layout: the q (with gates), k
    /// and v regions lie inside the row and share no column.
    pub fn new(
        (batch, seq, hq, hkv, dim): (u64, u64, u64, u64, u64),
        (rot, eps): (u64, f32),
        (ld_p, q_off, k_off, v_off): (u64, u64, u64, u64),
    ) -> Result<Self, CudaError> {
        const OP: &str = "attn_qk_norm_rope";
        for (what, v) in [
            ("batch", batch),
            ("seq", seq),
            ("hq", hq),
            ("hkv", hkv),
            ("dim", dim),
        ] {
            nonzero(OP, what, v)?;
        }
        exact_in_f32(OP, "dim", dim)?;
        // Positions are 0..seq, formed as (float)t.
        exact_in_f32(OP, "seq", seq)?;
        positive_eps(OP, eps)?;
        if rot == 0 || rot % 2 != 0 || rot > dim {
            return Err(CudaError::invalid(
                OP,
                format!("rotary_dim {rot} must be even, non-zero and at most dim {dim}"),
            ));
        }
        let q_w = mul(mul(2, hq, OP)?, dim, OP)?;
        let kv_w = mul(hkv, dim, OP)?;
        for (name, off, w) in [("q", q_off, q_w), ("k", k_off, kv_w), ("v", v_off, kv_w)] {
            if add(off, w, OP)? > ld_p {
                return Err(CudaError::invalid(
                    OP,
                    format!(
                        "{name} columns {off}..{} exceed the row stride {ld_p}",
                        off + w
                    ),
                ));
            }
        }
        for ((an, a, aw), (bn, b, bw)) in [
            (("q", q_off, q_w), ("k", k_off, kv_w)),
            (("q", q_off, q_w), ("v", v_off, kv_w)),
            (("k", k_off, kv_w), ("v", v_off, kv_w)),
        ] {
            if cols_overlap((a, aw), (b, bw)) {
                return Err(CudaError::invalid(
                    OP,
                    format!("{an} and {bn} columns overlap"),
                ));
            }
        }
        let rows = mul(batch, seq, OP)?;
        let units = mul(rows, add(hq, mul(2, hkv, OP)?, OP)?, OP)?;
        if blocks_for(units, u64::from(UNIT_WARPS)) > u64::from(MAX_GRID_X) {
            return Err(CudaError::invalid(
                OP,
                format!("{units} units exceed the grid"),
            ));
        }
        Ok(QkNormRopePlan {
            batch,
            seq,
            hq,
            hkv,
            dim,
            rot,
            eps,
            ld_p,
            q_off,
            k_off,
            v_off,
        })
    }

    /// Tokens, `batch * seq`.
    pub fn rows(&self) -> u64 {
        self.batch * self.seq
    }

    /// (token, head) units over `hq + 2 hkv` heads.
    pub fn units(&self) -> u64 {
        self.rows() * (self.hq + 2 * self.hkv)
    }

    /// Blocks of the forward (one warp per unit).
    pub fn fwd_blocks(&self) -> u64 {
        blocks_for(self.units(), u64::from(UNIT_WARPS))
    }

    /// Blocks of the backward.
    pub fn bwd_blocks(&self) -> u64 {
        blocks_for(self.rows(), QK_ROWS_PER_BLOCK)
    }

    /// Elements of the backward's `part`: two halves of `[nblocks, dim]`.
    pub fn part_len(&self) -> u64 {
        2 * self.bwd_blocks() * self.dim
    }

    /// Elements of the dense `q` (and `dq`).
    pub fn q_len(&self) -> u64 {
        self.rows() * self.hq * self.dim
    }

    /// Elements of the dense `k`, `v` (and `dk`, `dv`).
    pub fn kv_len(&self) -> u64 {
        self.rows() * self.hkv * self.dim
    }

    fn check_p(&self, op: &str, name: &str, len: usize) -> Result<(), CudaError> {
        // The row's last used column bounds every region.
        let end = [
            self.q_off + 2 * self.hq * self.dim,
            self.k_off + self.hkv * self.dim,
            self.v_off + self.hkv * self.dim,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Window {
            ld: self.ld_p,
            off: 0,
        }
        .check(op, name, self.rows(), end, len)
    }

    /// Check the forward's buffers.
    pub fn check_fwd(
        &self,
        (p, q_norm_w, k_norm_w, inv_freq): (usize, usize, usize, usize),
        (q, k, v): (usize, usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "attn_qk_norm_rope";
        self.check_p(OP, "p", p)?;
        exact_len(OP, "q_norm_w", q_norm_w, self.dim)?;
        exact_len(OP, "k_norm_w", k_norm_w, self.dim)?;
        exact_len(OP, "inv_freq", inv_freq, self.rot / 2)?;
        exact_len(OP, "q", q, self.q_len())?;
        exact_len(OP, "k", k, self.kv_len())?;
        exact_len(OP, "v", v, self.kv_len())
    }

    /// Check the backward's buffers (`dp` has `p`'s layout).
    pub fn check_bwd(
        &self,
        (p, q_norm_w, k_norm_w, inv_freq): (usize, usize, usize, usize),
        (dq, dk, dv, dp): (usize, usize, usize, usize),
        (part, dq_norm_w, dk_norm_w): (usize, usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "attn_qk_norm_rope_bwd";
        if self.dim > QK_BWD_MAX_DIM {
            return Err(CudaError::invalid(
                OP,
                format!("dim {} exceeds the backward's {QK_BWD_MAX_DIM}", self.dim),
            ));
        }
        self.check_fwd((p, q_norm_w, k_norm_w, inv_freq), (dq, dk, dv))?;
        self.check_p(OP, "dp", dp)?;
        exact_len(OP, "part", part, self.part_len())?;
        exact_len(OP, "dq_norm_w", dq_norm_w, self.dim)?;
        exact_len(OP, "dk_norm_w", dk_norm_w, self.dim)
    }
}

/// The output gate over `rows` tokens of `hq` heads of `dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputGatePlan {
    /// Tokens.
    pub rows: u64,
    /// Query heads.
    pub hq: u64,
    /// Head dim.
    pub dim: u64,
    /// Row stride of `p` and `dp`.
    pub ld_p: u64,
    /// First column of query head 0 in `p` (gates at `+ h*2D + D`).
    pub q_off: u64,
}

impl OutputGatePlan {
    /// Validate the shape and the gate columns' extent.
    pub fn new(rows: u64, hq: u64, dim: u64, (ld_p, q_off): (u64, u64)) -> Result<Self, CudaError> {
        const OP: &str = "attn_output_gate";
        for (what, v) in [("rows", rows), ("hq", hq), ("dim", dim)] {
            nonzero(OP, what, v)?;
        }
        let q_w = mul(mul(2, hq, OP)?, dim, OP)?;
        if add(q_off, q_w, OP)? > ld_p {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "q/gate columns {q_off}..{} exceed the row stride {ld_p}",
                    q_off + q_w
                ),
            ));
        }
        mul(rows, q_w, OP)?;
        Ok(OutputGatePlan {
            rows,
            hq,
            dim,
            ld_p,
            q_off,
        })
    }

    /// Columns of the attention output, `hq * dim`.
    pub fn width(&self) -> u64 {
        self.hq * self.dim
    }

    fn p_window(&self) -> Window {
        Window {
            ld: self.ld_p,
            off: self.q_off,
        }
    }

    /// Check the forward's buffers; `out` is a window.
    pub fn check_fwd(&self, attn: usize, p: usize, out: (Window, usize)) -> Result<(), CudaError> {
        const OP: &str = "attn_output_gate";
        exact_len(OP, "attn", attn, self.rows * self.width())?;
        self.p_window()
            .check(OP, "p", self.rows, 2 * self.width(), p)?;
        out.0.check(OP, "out", self.rows, self.width(), out.1)
    }

    /// Check the backward's buffers; `dy` is a window, `dp` has `p`'s layout.
    pub fn check_bwd(
        &self,
        (attn, p): (usize, usize),
        dy: (Window, usize),
        (d_attn, dp): (usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "attn_output_gate_bwd";
        exact_len(OP, "attn", attn, self.rows * self.width())?;
        self.p_window()
            .check(OP, "p", self.rows, 2 * self.width(), p)?;
        dy.0.check(OP, "dy", self.rows, self.width(), dy.1)?;
        exact_len(OP, "d_attn", d_attn, self.rows * self.width())?;
        self.p_window()
            .check(OP, "dp", self.rows, 2 * self.width(), dp)
    }
}

// ----------------------------------------------------------------- mirrors ---

fn u(v: u64) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

/// One head row forward on the host (Rust's `f32::cos`/`sin` stand in for
/// libdevice's, so this is not a bitwise mirror).
fn norm_rope_row(src: &[f32], w: &[f32], inv_freq: &[f32], pos: f32, eps: f32, dst: &mut [f32]) {
    let dim = src.len();
    let ss = unit_sum(dim, |d| src[d] * src[d]);
    let inv = 1.0f32 / (ss / dim as f32 + eps).sqrt();
    let half = inv_freq.len();
    for p in 0..half {
        let x0 = src[p] * inv * (1.0 + w[p]);
        let x1 = src[p + half] * inv * (1.0 + w[p + half]);
        let angle = pos * inv_freq[p];
        let (c, s) = (angle.cos(), angle.sin());
        dst[p] = x0 * c - x1 * s;
        dst[p + half] = x1 * c + x0 * s;
    }
    for d in 2 * half..dim {
        dst[d] = src[d] * inv * (1.0 + w[d]);
    }
}

/// One head row backward on the host; adds the row's `dw` into `acc`.
#[allow(clippy::too_many_arguments)]
fn norm_rope_row_bwd(
    src: &[f32],
    w: &[f32],
    inv_freq: &[f32],
    g: &[f32],
    pos: f32,
    eps: f32,
    dst: &mut [f32],
    acc: &mut [f32],
) {
    let dim = src.len();
    let ss = unit_sum(dim, |d| src[d] * src[d]);
    let rstd = 1.0f32 / (ss / dim as f32 + eps).sqrt();
    let half = inv_freq.len();
    let rot = 2 * half;
    let dn: Vec<f32> = (0..dim)
        .map(|d| {
            if d < rot {
                let pp = if d < half { d } else { d - half };
                let angle = pos * inv_freq[pp];
                let (c, s) = (angle.cos(), angle.sin());
                if d < half {
                    g[pp] * c + g[pp + half] * s
                } else {
                    g[d] * c - g[pp] * s
                }
            } else {
                g[d]
            }
        })
        .collect();
    let dot = unit_sum(dim, |d| dn[d] * (1.0 + w[d]) * src[d] * rstd);
    let m = dot / dim as f32;
    for d in 0..dim {
        let xn = src[d] * rstd;
        dst[d] = rstd * (dn[d] * (1.0 + w[d]) - xn * m);
        acc[d] += dn[d] * xn;
    }
}

/// Dense `(q, k, v)`: `[rows, hq, dim]`, `[rows, hkv, dim]`, `[rows, hkv, dim]`.
pub type Qkv = (Vec<f32>, Vec<f32>, Vec<f32>);

/// The forward on the host: dense `(q, k, v)`.
pub fn qk_norm_rope_fwd_mirror(
    plan: &QkNormRopePlan,
    p: &[f32],
    (q_norm_w, k_norm_w): (&[f32], &[f32]),
    inv_freq: &[f32],
) -> Result<Qkv, CudaError> {
    let (ql, kvl) = (u(plan.q_len()), u(plan.kv_len()));
    plan.check_fwd(
        (p.len(), q_norm_w.len(), k_norm_w.len(), inv_freq.len()),
        (ql, kvl, kvl),
    )?;
    let (seq, hq, hkv, dim, ld) = (
        u(plan.seq),
        u(plan.hq),
        u(plan.hkv),
        u(plan.dim),
        u(plan.ld_p),
    );
    let (q_off, k_off, v_off) = (u(plan.q_off), u(plan.k_off), u(plan.v_off));
    let (mut q, mut k, mut v) = (vec![0.0f32; ql], vec![0.0f32; kvl], vec![0.0f32; kvl]);
    for r in 0..u(plan.rows()) {
        let pos = (r % seq) as f32;
        let row = &p[r * ld..];
        for h in 0..hq {
            let src = &row[q_off + h * 2 * dim..][..dim];
            norm_rope_row(
                src,
                q_norm_w,
                inv_freq,
                pos,
                plan.eps,
                &mut q[(r * hq + h) * dim..][..dim],
            );
        }
        for h in 0..hkv {
            let src = &row[k_off + h * dim..][..dim];
            norm_rope_row(
                src,
                k_norm_w,
                inv_freq,
                pos,
                plan.eps,
                &mut k[(r * hkv + h) * dim..][..dim],
            );
            v[(r * hkv + h) * dim..][..dim].copy_from_slice(&row[v_off + h * dim..][..dim]);
        }
    }
    Ok((q, k, v))
}

/// The backward on the host: writes `dq`, `dk`, `dv` into `dp`'s q, k, v
/// columns and returns `(dq_norm_w, dk_norm_w)`, block partials summed in the
/// kernels' order (each block's warps strided over its units, then warps in
/// order, then blocks in order).
pub fn qk_norm_rope_bwd_mirror(
    plan: &QkNormRopePlan,
    p: &[f32],
    (q_norm_w, k_norm_w): (&[f32], &[f32]),
    inv_freq: &[f32],
    (dq, dk, dv): (&[f32], &[f32], &[f32]),
    dp: &mut [f32],
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let part_len = u(plan.part_len());
    plan.check_bwd(
        (p.len(), q_norm_w.len(), k_norm_w.len(), inv_freq.len()),
        (dq.len(), dk.len(), dv.len(), dp.len()),
        (part_len, q_norm_w.len(), k_norm_w.len()),
    )?;
    let (seq, hq, hkv, dim, ld) = (
        u(plan.seq),
        u(plan.hq),
        u(plan.hkv),
        u(plan.dim),
        u(plan.ld_p),
    );
    let (q_off, k_off, v_off) = (u(plan.q_off), u(plan.k_off), u(plan.v_off));
    let (rows, heads) = (u(plan.rows()), hq + 2 * hkv);
    let rpb = u(QK_ROWS_PER_BLOCK);
    let nwarps = UNIT_WARPS as usize;
    let nblocks = u(plan.bwd_blocks());
    let mut part = vec![0.0f32; part_len];
    for blk in 0..nblocks {
        let (r0, r1) = (blk * rpb, rows.min(blk * rpb + rpb));
        let mut acc_q = vec![vec![0.0f32; dim]; nwarps];
        let mut acc_k = vec![vec![0.0f32; dim]; nwarps];
        for warp in 0..nwarps {
            for unit in (r0 * heads + warp..r1 * heads).step_by(nwarps) {
                let (r, j) = (unit / heads, unit % heads);
                let pos = (r % seq) as f32;
                if j < hq {
                    let col = r * ld + q_off + j * 2 * dim;
                    let src = p[col..col + dim].to_vec();
                    let g = &dq[(r * hq + j) * dim..][..dim];
                    norm_rope_row_bwd(
                        &src,
                        q_norm_w,
                        inv_freq,
                        g,
                        pos,
                        plan.eps,
                        &mut dp[col..col + dim],
                        &mut acc_q[warp],
                    );
                } else if j < hq + hkv {
                    let h = j - hq;
                    let col = r * ld + k_off + h * dim;
                    let src = p[col..col + dim].to_vec();
                    let g = &dk[(r * hkv + h) * dim..][..dim];
                    norm_rope_row_bwd(
                        &src,
                        k_norm_w,
                        inv_freq,
                        g,
                        pos,
                        plan.eps,
                        &mut dp[col..col + dim],
                        &mut acc_k[warp],
                    );
                } else {
                    let h = j - hq - hkv;
                    let col = r * ld + v_off + h * dim;
                    dp[col..col + dim].copy_from_slice(&dv[(r * hkv + h) * dim..][..dim]);
                }
            }
        }
        for d in 0..dim {
            part[blk * dim + d] = acc_q.iter().fold(0.0f32, |s, a| s + a[d]);
            part[(nblocks + blk) * dim + d] = acc_k.iter().fold(0.0f32, |s, a| s + a[d]);
        }
    }
    Ok((
        col_sum_blocks(&part, 0, nblocks, dim),
        col_sum_blocks(&part, nblocks * dim, nblocks, dim),
    ))
}

/// The output gate forward on the host (bitwise), into `out`'s window.
pub fn output_gate_fwd_mirror(
    plan: &OutputGatePlan,
    attn: &[f32],
    p: &[f32],
    out_win: Window,
    out: &mut [f32],
) -> Result<(), CudaError> {
    plan.check_fwd(attn.len(), p.len(), (out_win, out.len()))?;
    let (width, dim, ld, q_off) = (u(plan.width()), u(plan.dim), u(plan.ld_p), u(plan.q_off));
    for r in 0..u(plan.rows) {
        for col in 0..width {
            let (h, d) = (col / dim, col % dim);
            let g = p[r * ld + q_off + h * 2 * dim + dim + d];
            out[out_win.at(r, col)] = attn[r * width + col] * sigmoid_f32(g);
        }
    }
    Ok(())
}

/// The output gate backward on the host (bitwise): returns `d_attn` and
/// writes `d_gate` into `dp`'s gate columns.
pub fn output_gate_bwd_mirror(
    plan: &OutputGatePlan,
    (attn, p): (&[f32], &[f32]),
    (dy_win, dy): (Window, &[f32]),
    dp: &mut [f32],
) -> Result<Vec<f32>, CudaError> {
    let n = u(plan.rows * plan.width());
    plan.check_bwd((attn.len(), p.len()), (dy_win, dy.len()), (n, dp.len()))?;
    let (width, dim, ld, q_off) = (u(plan.width()), u(plan.dim), u(plan.ld_p), u(plan.q_off));
    let mut d_attn = vec![0.0f32; n];
    for r in 0..u(plan.rows) {
        for col in 0..width {
            let (h, d) = (col / dim, col % dim);
            let gi = r * ld + q_off + h * 2 * dim + dim + d;
            let s = sigmoid_f32(p[gi]);
            let a = attn[r * width + col];
            let g = dy[dy_win.at(r, col)];
            d_attn[r * width + col] = g * s;
            dp[gi] = g * a * s * (1.0 - s);
        }
    }
    Ok(d_attn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_2b(seq: u64) -> QkNormRopePlan {
        // Qwen3.5-2B: 8 query heads (with gates) then 2 key, 2 value heads of 256.
        let (q_w, kv_w) = (2 * 8 * 256, 2 * 256);
        QkNormRopePlan::new(
            (1, seq, 8, 2, 256),
            (64, 1e-6),
            (q_w + 2 * kv_w, 0, q_w, q_w + kv_w),
        )
        .unwrap()
    }

    #[test]
    fn the_2b_layout_is_accepted_and_overlaps_are_refused() {
        let p = plan_2b(10);
        assert_eq!(
            (p.units(), p.fwd_blocks(), p.bwd_blocks(), p.part_len()),
            (120, 30, 1, 512)
        );
        // k overlapping the last query head's gate.
        assert!(QkNormRopePlan::new((1, 4, 8, 2, 256), (64, 1e-6), (5120, 0, 4000, 4608)).is_err());
        // Odd and oversized rotary dims.
        assert!(QkNormRopePlan::new((1, 4, 1, 1, 8), (3, 1e-6), (32, 0, 16, 24)).is_err());
        assert!(QkNormRopePlan::new((1, 4, 1, 1, 8), (10, 1e-6), (32, 0, 16, 24)).is_err());
        assert!(QkNormRopePlan::new((1, 4, 1, 1, 8), (0, 1e-6), (32, 0, 16, 24)).is_err());
        // Positions past 2^24 are not exact as f32.
        assert!(
            QkNormRopePlan::new((1, (1 << 24) + 1, 1, 1, 8), (8, 1e-6), (32, 0, 16, 24)).is_err()
        );
        let wide =
            QkNormRopePlan::new((1, 1, 1, 1, 520), (8, 1e-6), (2080, 0, 1040, 1560)).unwrap();
        let e = wide
            .check_bwd((2080, 520, 520, 4), (520, 520, 520, 2080), (1040, 520, 520))
            .unwrap_err();
        assert!(e.to_string().contains("exceeds the backward"), "{e}");
    }

    #[test]
    fn inv_freq_is_transformers_and_refuses_bad_inputs() {
        let f = rope_inv_freq(64, 1e7).unwrap();
        assert_eq!(f.len(), 32);
        assert_eq!(f[0], 1.0);
        assert!((f64::from(f[31]) - 1e7f64.powf(-62.0 / 64.0)).abs() < 1e-12);
        assert!(rope_inv_freq(63, 1e7).is_err());
        assert!(rope_inv_freq(0, 1e7).is_err());
        assert!(rope_inv_freq(64, 0.1).is_err(), "0.1 is not exact in f32");
        assert!(rope_inv_freq(64, f64::NAN).is_err());
    }

    #[test]
    fn the_cuda_source_takes_the_angle_from_the_host_table_and_forms_one_plus_w() {
        assert_eq!(
            SOURCE
                .matches("const float angle = pos * inv_freq[")
                .count(),
            2
        );
        assert!(
            !SOURCE.contains("powf"),
            "the device must not form inv_freq itself"
        );
        assert_eq!(SOURCE.matches("(1.0f + weight[").count(), 5);
        assert!(
            SOURCE.contains("v = d < half ? g[pp] * c + g[pp + half] * s : g[d] * c - g[pp] * s;")
        );
        assert!(SOURCE.contains("dp[gi] = g * a * s * (1.0f - s);"));
    }

    #[test]
    fn the_gate_mirrors_write_only_the_gate_columns() {
        let plan = OutputGatePlan::new(2, 2, 3, (14, 1)).unwrap();
        let p: Vec<f32> = (0..28u8).map(|v| f32::from(v) * 0.1 - 1.0).collect();
        let attn = vec![1.0f32; 12];
        let dy = vec![0.5f32; 12];
        let mut dp = vec![-7.0f32; 28];
        let d_attn =
            output_gate_bwd_mirror(&plan, (&attn, &p), (Window::dense(6), &dy), &mut dp).unwrap();
        assert_eq!(d_attn.len(), 12);
        for r in 0..2 {
            for c in 0..14 {
                // head 0's gate is columns 1 + 3 .. 1 + 6, head 1's 1 + 9 .. 1 + 12.
                let gate = (4..7).contains(&c) || (10..13).contains(&c);
                assert_eq!(dp[r * 14 + c] == -7.0, !gate, "r{r} c{c}");
            }
        }
        assert!(OutputGatePlan::new(2, 2, 3, (12, 1)).is_err());
    }
}
