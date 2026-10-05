//! **K7: Qwen3.5's two RMSNorms**, forward and backward (`cuda-backend-scoping.md`
//! §3 K7): the plans, the CUDA-C source and the host mirrors. The device
//! launches are [`crate::rmsnorm_cuda`].
//!
//! - **`rms_norm`**, transformers' `Qwen3_5RMSNorm` (`modeling_qwen3_5.py:736-751`):
//!   `y = x * rstd * (1 + w)`, `rstd = 1 / sqrt(mean(x^2) + eps)`, with `w`
//!   stored as the checkpoint holds it and `1 + w` formed in the kernel, as
//!   tessl's `qwen35_rms_norm_f32` does (`tessl/kernels/qwen35_mlp.metal:73-112`).
//!   Passing these weights to a plain `* w` norm zeroes the output at init;
//!   that is the mutation this lane's fail-first plants.
//! - **`gated_rms_norm`**, transformers' `Qwen3_5RMSNormGated` (`:187-202`):
//!   `y = w * (x * rstd) * silu(z)` per `dim`-wide head (the weight applied as
//!   `w`, not `1 + w`; tessl `qwen35_gdn.metal:870-946`).
//!
//! # Algorithm (tessl's, `tessl/kernels/qwen35_bwd.metal:37-224`)
//!
//! - `rms_norm` forward: one block of [`ROW_THREADS`] threads per row.
//! - `rms_norm` backward: blocks of [`RMS_ROWS_PER_BLOCK`] rows. Per row, the
//!   sum of squares and `dot = sum(dy (1 + w) x)` are block sums, then
//!   `dx = rstd dy (1 + w) - x rstd^3 dot / dim` (added to `dx` with
//!   `accumulate`). Each thread owns columns `t, t + 256, ...` (at most
//!   [`MAX_COLS`]) and sums `dy x rstd` over its block's rows into the block's
//!   row of `part`; [`crate::small_common::COL_SUM`] then sums the blocks in
//!   order into `dw`.
//! - `gated_rms_norm`: one warp per (row, head) unit, [`UNIT_WARPS`] warps per
//!   block. The backward takes [`GATED_UNITS_PER_BLOCK`] units per block, its
//!   warps striding through them; each lane owns columns `lane, lane + 32, ...`
//!   of `dw`, and the warps' column sums are combined in warp order into the
//!   block's row of `part`.
//!
//! `rstd` is `1.0f / sqrtf(...)`: both correctly rounded under
//! [`crate::kernels::STRICT_SM90`] (`--prec-div=true --prec-sqrt=true`), where
//! `rsqrtf` is approximate. tessl uses `rsqrt`; the difference is within an ulp
//! of `rstd`, far inside every bound.
//!
//! # Host mirrors
//!
//! [`rms_norm_fwd_mirror`] and [`rms_norm_bwd_mirror`] run the kernels'
//! exact operation sequence in f32 (same per-thread partials, same reduction
//! tree, same operation order), so a device result can be compared with them
//! **bitwise**: the plain norm uses only `+ - * /` and `sqrt`, each one IEEE
//! rounding on both sides. The gated mirrors call `crate::k8_act`'s host
//! emulations of SiLU, which that module states are bit-identical to its
//! device functions.

use crate::error::CudaError;
use crate::geometry::MAX_GRID_X;
use crate::kernels::KernelModule;
use crate::small_common::{
    add, blocks_for, cols_overlap, exact_in_f32, exact_len, mul, nonzero, positive_eps, row_sum,
    unit_sum, Window, MAX_COLS, ROW_THREADS, UNIT_WARPS, WARP,
};

/// Rows per block of the `rms_norm` backward (tessl `RMS_ROWS_PER_BLOCK`,
/// `tessl/src/qwen35_bwd.rs:31`).
pub const RMS_ROWS_PER_BLOCK: u64 = 32;

/// Widest row the `rms_norm` backward takes: [`MAX_COLS`] columns per thread.
pub const RMS_BWD_MAX_DIM: u64 = ROW_THREADS as u64 * MAX_COLS;

/// (row, head) units per block of the `gated_rms_norm` backward (tessl
/// `GATED_UNITS_PER_BLOCK`, `tessl/src/qwen35_bwd.rs:34`).
pub const GATED_UNITS_PER_BLOCK: u64 = 64;

/// Widest head the warp-per-unit `gated_rms_norm` backward takes.
pub const GATED_BWD_MAX_DIM: u64 = WARP as u64 * MAX_COLS;

/// Entry: `rms_norm` forward.
pub const RMS_FWD: &str = "qd_rms_norm_f32";
/// Entry: `rms_norm` backward (`dx`, per-block `dw` partials).
pub const RMS_BWD: &str = "qd_rms_norm_bwd_f32";
/// Entry: `gated_rms_norm` forward.
pub const GATED_FWD: &str = "qd_gated_rms_norm_f32";
/// Entry: `gated_rms_norm` backward (`dx`, `dz`, per-block `dw` partials).
pub const GATED_BWD: &str = "qd_gated_rms_norm_bwd_f32";

/// The K7 NVRTC module.
pub const MODULE: KernelModule = KernelModule {
    name: "k7_rmsnorm",
    source: SOURCE,
    entries: &[RMS_FWD, RMS_BWD, GATED_FWD, GATED_BWD],
};

const SOURCE: &str = concat!(
    crate::act_prelude!(),
    crate::small_common::small_prelude!(),
    r#"
#define QD_K7_MAX_COLS 16

// Qwen3_5RMSNorm: y = x * rstd * (1 + w). One block per row.
extern "C" __global__ void qd_rms_norm_f32(
    const float* x, const float* w, float* out,
    unsigned long long rows, unsigned long long dim, float eps)
{
    __shared__ float scratch[32];
    const unsigned long long r = blockIdx.x;
    if (r >= rows) {
        return;  // uniform per block
    }
    const float* xr = x + r * dim;
    float* yr = out + r * dim;
    float ss = 0.0f;
    for (unsigned long long d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = xr[d];
        ss = ss + v * v;
    }
    ss = qd_block_sum(ss, scratch);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (unsigned long long d = threadIdx.x; d < dim; d += blockDim.x) {
        yr[d] = xr[d] * inv * (1.0f + w[d]);
    }
}

// dx = rstd dy (1 + w) - x rstd^3 mean(dy (1 + w) x)  (added with accumulate)
// part[blk, d] = sum over the block's rows of dy x rstd.
extern "C" __global__ void qd_rms_norm_bwd_f32(
    const float* x, const float* w, const float* dy, float* dx, float* part,
    unsigned long long rows, unsigned long long dim, float eps,
    unsigned long long rows_per_block, int accumulate)
{
    __shared__ float scratch[32];
    const unsigned long long blk = blockIdx.x;
    const unsigned long long r0 = blk * rows_per_block;
    if (r0 >= rows) {
        return;  // uniform per block
    }
    const unsigned long long r1 = rows - r0 < rows_per_block ? rows : r0 + rows_per_block;
    const float fdim = (float)dim;
    float acc[QD_K7_MAX_COLS];
    for (int k = 0; k < QD_K7_MAX_COLS; ++k) {
        acc[k] = 0.0f;
    }
    for (unsigned long long r = r0; r < r1; ++r) {
        const float* xr = x + r * dim;
        const float* gr = dy + r * dim;
        float ss = 0.0f;
        float dot = 0.0f;
        for (unsigned long long d = threadIdx.x; d < dim; d += blockDim.x) {
            const float xv = xr[d];
            ss = ss + xv * xv;
            dot = dot + gr[d] * (1.0f + w[d]) * xv;
        }
        ss = qd_block_sum(ss, scratch);
        dot = qd_block_sum(dot, scratch);
        const float rstd = 1.0f / sqrtf(ss / fdim + eps);
        const float c = rstd * rstd * rstd * dot / fdim;
        float* xo = dx + r * dim;
        int k = 0;
        for (unsigned long long d = threadIdx.x; d < dim; d += blockDim.x, ++k) {
            const float v = rstd * gr[d] * (1.0f + w[d]) - xr[d] * c;
            xo[d] = accumulate ? xo[d] + v : v;
            acc[k] = acc[k] + gr[d] * xr[d] * rstd;
        }
    }
    int k = 0;
    for (unsigned long long d = threadIdx.x; d < dim; d += blockDim.x, ++k) {
        part[blk * dim + d] = acc[k];
    }
}

// Qwen3_5RMSNormGated: y = w * (x * rstd) * silu(z), one warp per (row, head).
extern "C" __global__ void qd_gated_rms_norm_f32(
    const float* x, const float* z, const float* w, float* out,
    unsigned long long rows, unsigned long long heads, unsigned long long dim,
    unsigned long long ld_x, unsigned long long x_off,
    unsigned long long ld_z, unsigned long long z_off,
    unsigned long long ld_out, unsigned long long out_off, float eps)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned long long unit =
        (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (unit >= rows * heads) {
        return;  // uniform per warp; this kernel has no block barrier
    }
    const unsigned long long r = unit / heads;
    const unsigned long long h = unit - r * heads;
    const float* xr = x + r * ld_x + x_off + h * dim;
    const float* zr = z + r * ld_z + z_off + h * dim;
    float* yr = out + r * ld_out + out_off + h * dim;
    float ss = 0.0f;
    for (unsigned long long d = lane; d < dim; d += 32) {
        ss = ss + xr[d] * xr[d];
    }
    ss = qd_warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    for (unsigned long long d = lane; d < dim; d += 32) {
        yr[d] = w[d] * (xr[d] * inv) * qd_silu(zr[d]);
    }
}

// Per unit, xn = x rstd, s = silu(z):
//   dz = dy w xn silu'(z);  dx = rstd (dy w s - xn mean(dy w s xn))
//   part[blk, d] = sum over the block's units of dy xn s (warps in order).
extern "C" __global__ void qd_gated_rms_norm_bwd_f32(
    const float* x, const float* z, const float* w, const float* dy,
    float* dx, float* dz, float* part,
    unsigned long long rows, unsigned long long heads, unsigned long long dim,
    unsigned long long ld_x, unsigned long long x_off,
    unsigned long long ld_z, unsigned long long z_off,
    unsigned long long ld_dy, unsigned long long dy_off,
    unsigned long long ld_dx, unsigned long long dx_off,
    unsigned long long ld_dz, unsigned long long dz_off,
    float eps, unsigned long long units_per_block)
{
    __shared__ float warp_acc[4 * QD_K7_MAX_COLS * 32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int nwarps = blockDim.x >> 5;
    const unsigned long long units = rows * heads;
    const unsigned long long blk = blockIdx.x;
    const unsigned long long u0 = blk * units_per_block;
    if (u0 >= units) {
        return;  // uniform per block
    }
    const unsigned long long u1 = units - u0 < units_per_block ? units : u0 + units_per_block;
    const float fdim = (float)dim;
    float acc[QD_K7_MAX_COLS];
    for (int k = 0; k < QD_K7_MAX_COLS; ++k) {
        acc[k] = 0.0f;
    }
    for (unsigned long long u = u0 + warp; u < u1; u += nwarps) {
        const unsigned long long r = u / heads;
        const unsigned long long h = u - r * heads;
        const float* xr = x + r * ld_x + x_off + h * dim;
        const float* zr = z + r * ld_z + z_off + h * dim;
        const float* gr = dy + r * ld_dy + dy_off + h * dim;
        float* xo = dx + r * ld_dx + dx_off + h * dim;
        float* zo = dz + r * ld_dz + dz_off + h * dim;
        float ss = 0.0f;
        for (unsigned long long d = lane; d < dim; d += 32) {
            ss = ss + xr[d] * xr[d];
        }
        const float rstd = 1.0f / sqrtf(qd_warp_sum(ss) / fdim + eps);
        float dot = 0.0f;
        for (unsigned long long d = lane; d < dim; d += 32) {
            const float xn = xr[d] * rstd;
            dot = dot + gr[d] * w[d] * qd_silu(zr[d]) * xn;
        }
        const float m = qd_warp_sum(dot) / fdim;
        int k = 0;
        for (unsigned long long d = lane; d < dim; d += 32, ++k) {
            const float xn = xr[d] * rstd;
            const float zv = zr[d];
            const float s = qd_silu(zv);
            const float dxn = gr[d] * w[d] * s;
            xo[d] = rstd * (dxn - xn * m);
            zo[d] = gr[d] * w[d] * xn * qd_silu_grad(zv);
            acc[k] = acc[k] + gr[d] * xn * s;
        }
    }
    const unsigned long long per = (dim + 31) / 32;
    for (unsigned long long k = 0; k < per; ++k) {
        warp_acc[(warp * per + k) * 32 + lane] = acc[k];
    }
    __syncthreads();
    if (warp == 0u) {
        for (unsigned long long k = 0; k < per; ++k) {
            const unsigned long long d = lane + 32 * k;
            if (d < dim) {
                float s = 0.0f;
                for (unsigned int g = 0; g < nwarps; ++g) {
                    s = s + warp_acc[(g * per + k) * 32 + lane];
                }
                part[blk * dim + d] = s;
            }
        }
    }
}
"#
);

// ------------------------------------------------------------------- plans ---

/// A dense `[rows, dim]` `rms_norm`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormPlan {
    /// Rows.
    pub rows: u64,
    /// Row width (the hidden size, 2048 at Qwen3.5-2B).
    pub dim: u64,
    /// `rms_norm_eps` (1e-6 at Qwen3.5-2B).
    pub eps: f32,
}

impl RmsNormPlan {
    /// Validate the shape. Zero sizes are refused (skip the call instead).
    pub fn new(rows: u64, dim: u64, eps: f32) -> Result<Self, CudaError> {
        const OP: &str = "rms_norm";
        nonzero(OP, "rows", rows)?;
        nonzero(OP, "dim", dim)?;
        exact_in_f32(OP, "dim", dim)?;
        positive_eps(OP, eps)?;
        mul(rows, dim, OP)?;
        if rows > u64::from(MAX_GRID_X) {
            return Err(CudaError::invalid(
                OP,
                format!("{rows} rows exceed the one-block-per-row grid's {MAX_GRID_X}"),
            ));
        }
        Ok(RmsNormPlan { rows, dim, eps })
    }

    /// Elements of `x`, `y`, `dy` and `dx`.
    pub fn len(&self) -> u64 {
        self.rows * self.dim
    }

    /// Never true: a plan with no elements is refused.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Blocks of the backward.
    pub fn bwd_blocks(&self) -> u64 {
        blocks_for(self.rows, RMS_ROWS_PER_BLOCK)
    }

    /// Elements of the backward's `part` scratch.
    pub fn part_len(&self) -> u64 {
        self.bwd_blocks() * self.dim
    }

    /// Check the forward's buffers.
    pub fn check_fwd(&self, x: usize, w: usize, out: usize) -> Result<(), CudaError> {
        const OP: &str = "rms_norm";
        exact_len(OP, "x", x, self.len())?;
        exact_len(OP, "w", w, self.dim)?;
        exact_len(OP, "out", out, self.len())
    }

    /// Check the backward's buffers and its column cap.
    pub fn check_bwd(
        &self,
        (x, w, dy): (usize, usize, usize),
        (dx, part, dw): (usize, usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "rms_norm_bwd";
        if self.dim > RMS_BWD_MAX_DIM {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "dim {} exceeds the backward's {RMS_BWD_MAX_DIM} ({ROW_THREADS} threads x {MAX_COLS} columns)",
                    self.dim
                ),
            ));
        }
        exact_len(OP, "x", x, self.len())?;
        exact_len(OP, "w", w, self.dim)?;
        exact_len(OP, "dy", dy, self.len())?;
        exact_len(OP, "dx", dx, self.len())?;
        exact_len(OP, "part", part, self.part_len())?;
        exact_len(OP, "dw", dw, self.dim)
    }
}

/// `gated_rms_norm` over `heads` heads of `dim` per row, each operand a
/// [`Window`] of `heads * dim` columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedRmsNormPlan {
    /// Rows (tokens).
    pub rows: u64,
    /// Heads per row (16 value heads at Qwen3.5-2B).
    pub heads: u64,
    /// Head width (128 at Qwen3.5-2B).
    pub dim: u64,
    /// `rms_norm_eps`.
    pub eps: f32,
    /// Where `x` (the GDN output) lives.
    pub x: Window,
    /// Where `z` (the gate, a window of the fused projection) lives.
    pub z: Window,
}

impl GatedRmsNormPlan {
    /// Validate the shape and the two input windows' strides.
    pub fn new(
        (rows, heads, dim): (u64, u64, u64),
        eps: f32,
        x: Window,
        z: Window,
    ) -> Result<Self, CudaError> {
        const OP: &str = "gated_rms_norm";
        nonzero(OP, "rows", rows)?;
        nonzero(OP, "heads", heads)?;
        nonzero(OP, "dim", dim)?;
        exact_in_f32(OP, "dim", dim)?;
        positive_eps(OP, eps)?;
        let width = mul(heads, dim, OP)?;
        let units = mul(rows, heads, OP)?;
        let blocks = blocks_for(units, u64::from(UNIT_WARPS));
        if blocks > u64::from(MAX_GRID_X) {
            return Err(CudaError::invalid(
                OP,
                format!("{units} units need {blocks} blocks, past {MAX_GRID_X}"),
            ));
        }
        for (name, win) in [("x", x), ("z", z)] {
            if add(win.off, width, OP)? > win.ld {
                return Err(CudaError::invalid(
                    OP,
                    format!(
                        "{name} window {}+{width} is wider than its stride {}",
                        win.off, win.ld
                    ),
                ));
            }
        }
        Ok(GatedRmsNormPlan {
            rows,
            heads,
            dim,
            eps,
            x,
            z,
        })
    }

    /// Columns per row of every operand: `heads * dim`.
    pub fn width(&self) -> u64 {
        self.heads * self.dim
    }

    /// (row, head) units.
    pub fn units(&self) -> u64 {
        self.rows * self.heads
    }

    /// Blocks of the forward (one warp per unit).
    pub fn fwd_blocks(&self) -> u64 {
        blocks_for(self.units(), u64::from(UNIT_WARPS))
    }

    /// Blocks of the backward.
    pub fn bwd_blocks(&self) -> u64 {
        blocks_for(self.units(), GATED_UNITS_PER_BLOCK)
    }

    /// Elements of the backward's `part` scratch.
    pub fn part_len(&self) -> u64 {
        self.bwd_blocks() * self.dim
    }

    fn check_inputs(&self, op: &str, x: usize, z: usize, w: usize) -> Result<(), CudaError> {
        self.x.check(op, "x", self.rows, self.width(), x)?;
        self.z.check(op, "z", self.rows, self.width(), z)?;
        exact_len(op, "w", w, self.dim)
    }

    /// Check the forward's buffers; `out` is a window of the output buffer.
    pub fn check_fwd(
        &self,
        (x, z, w): (usize, usize, usize),
        out: (Window, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "gated_rms_norm";
        self.check_inputs(OP, x, z, w)?;
        out.0.check(OP, "out", self.rows, self.width(), out.1)
    }

    /// Check the backward's buffers. `dy`, `dx`, `dz` are windows; `dx` and
    /// `dz` are distinct buffers (one exclusive borrow each on the device).
    pub fn check_bwd(
        &self,
        (x, z, w): (usize, usize, usize),
        (dy, dx, dz): ((Window, usize), (Window, usize), (Window, usize)),
        (part, dw): (usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "gated_rms_norm_bwd";
        if self.dim > GATED_BWD_MAX_DIM {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "dim {} exceeds the backward's {GATED_BWD_MAX_DIM} (32 lanes x {MAX_COLS} columns)",
                    self.dim
                ),
            ));
        }
        self.check_inputs(OP, x, z, w)?;
        dy.0.check(OP, "dy", self.rows, self.width(), dy.1)?;
        dx.0.check(OP, "dx", self.rows, self.width(), dx.1)?;
        dz.0.check(OP, "dz", self.rows, self.width(), dz.1)?;
        exact_len(OP, "part", part, self.part_len())?;
        exact_len(OP, "dw", dw, self.dim)
    }
}

// ----------------------------------------------------------------- mirrors ---

fn to_usize_dims(a: u64, b: u64) -> (usize, usize) {
    // Plans bound both by buffer lengths that exist on the host.
    (
        usize::try_from(a).unwrap_or(usize::MAX),
        usize::try_from(b).unwrap_or(usize::MAX),
    )
}

/// `qd_rms_norm_f32` on the host, operation for operation.
pub fn rms_norm_fwd_mirror(
    plan: &RmsNormPlan,
    x: &[f32],
    w: &[f32],
) -> Result<Vec<f32>, CudaError> {
    plan.check_fwd(x.len(), w.len(), x.len())?;
    let (rows, dim) = to_usize_dims(plan.rows, plan.dim);
    let threads = ROW_THREADS as usize;
    let fdim = plan.dim as f32;
    let mut y = vec![0.0f32; x.len()];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let ss = row_sum(threads, dim, |d| xr[d] * xr[d]);
        let inv = 1.0f32 / (ss / fdim + plan.eps).sqrt();
        for d in 0..dim {
            y[r * dim + d] = xr[d] * inv * (1.0f32 + w[d]);
        }
    }
    Ok(y)
}

/// `qd_rms_norm_bwd_f32` then the column sum on the host: `(dx, dw)`. With
/// `dx_prev`, `dx` is added to it (`accumulate`).
pub fn rms_norm_bwd_mirror(
    plan: &RmsNormPlan,
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    dx_prev: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let part_len = to_usize_dims(plan.part_len(), 0).0;
    plan.check_bwd((x.len(), w.len(), dy.len()), (x.len(), part_len, w.len()))?;
    if let Some(p) = dx_prev {
        exact_len("rms_norm_bwd", "dx_prev", p.len(), plan.len())?;
    }
    let (rows, dim) = to_usize_dims(plan.rows, plan.dim);
    let threads = ROW_THREADS as usize;
    let fdim = plan.dim as f32;
    let rpb = RMS_ROWS_PER_BLOCK as usize;
    let nblocks = part_len / dim;
    let mut dx = dx_prev.map_or_else(|| vec![0.0f32; x.len()], <[f32]>::to_vec);
    let mut part = vec![0.0f32; part_len];
    for blk in 0..nblocks {
        let (r0, r1) = (blk * rpb, rows.min(blk * rpb + rpb));
        let mut acc = vec![0.0f32; dim];
        for r in r0..r1 {
            let xr = &x[r * dim..(r + 1) * dim];
            let gr = &dy[r * dim..(r + 1) * dim];
            let ss = row_sum(threads, dim, |d| xr[d] * xr[d]);
            let dot = row_sum(threads, dim, |d| gr[d] * (1.0f32 + w[d]) * xr[d]);
            let rstd = 1.0f32 / (ss / fdim + plan.eps).sqrt();
            let c = rstd * rstd * rstd * dot / fdim;
            for d in 0..dim {
                let v = rstd * gr[d] * (1.0f32 + w[d]) - xr[d] * c;
                let o = r * dim + d;
                dx[o] = if dx_prev.is_some() { dx[o] + v } else { v };
                acc[d] += gr[d] * xr[d] * rstd;
            }
        }
        part[blk * dim..(blk + 1) * dim].copy_from_slice(&acc);
    }
    let dw = crate::small_common::col_sum_blocks(&part, 0, nblocks, dim);
    Ok((dx, dw))
}

/// `qd_gated_rms_norm_f32` on the host, written into `out`'s window.
pub fn gated_rms_norm_fwd_mirror(
    plan: &GatedRmsNormPlan,
    (x, z, w): (&[f32], &[f32], &[f32]),
    out_win: Window,
    out: &mut [f32],
) -> Result<(), CudaError> {
    plan.check_fwd((x.len(), z.len(), w.len()), (out_win, out.len()))?;
    let (rows, heads) = to_usize_dims(plan.rows, plan.heads);
    let dim = to_usize_dims(plan.dim, 0).0;
    let fdim = plan.dim as f32;
    for r in 0..rows {
        for h in 0..heads {
            let xi = |d: usize| plan.x.at(r, h * dim + d);
            let zi = |d: usize| plan.z.at(r, h * dim + d);
            let ss = unit_sum(dim, |d| x[xi(d)] * x[xi(d)]);
            let inv = 1.0f32 / (ss / fdim + plan.eps).sqrt();
            for d in 0..dim {
                out[out_win.at(r, h * dim + d)] =
                    w[d] * (x[xi(d)] * inv) * crate::k8_act::silu_f32(z[zi(d)]);
            }
        }
    }
    Ok(())
}

/// Where the gated backward writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatedGradWindows {
    /// The upstream gradient's window.
    pub dy: Window,
    /// `dx`'s window.
    pub dx: Window,
    /// `dz`'s window (in the fused projection gradient, in training).
    pub dz: Window,
}

/// `qd_gated_rms_norm_bwd_f32` then the column sum on the host: writes `dx`
/// and `dz` into their windows and returns `dw`.
pub fn gated_rms_norm_bwd_mirror(
    plan: &GatedRmsNormPlan,
    (x, z, w, dy): (&[f32], &[f32], &[f32], &[f32]),
    win: GatedGradWindows,
    dx: &mut [f32],
    dz: &mut [f32],
) -> Result<Vec<f32>, CudaError> {
    let part_len = to_usize_dims(plan.part_len(), 0).0;
    plan.check_bwd(
        (x.len(), z.len(), w.len()),
        ((win.dy, dy.len()), (win.dx, dx.len()), (win.dz, dz.len())),
        (part_len, w.len()),
    )?;
    let (units, heads) = to_usize_dims(plan.units(), plan.heads);
    let dim = to_usize_dims(plan.dim, 0).0;
    let fdim = plan.dim as f32;
    let upb = GATED_UNITS_PER_BLOCK as usize;
    let nwarps = UNIT_WARPS as usize;
    let nblocks = part_len / dim;
    let mut part = vec![0.0f32; part_len];
    let silu = crate::k8_act::silu_f32;
    let silu_grad = crate::k8_act::silu_grad_f32;
    for blk in 0..nblocks {
        let (u0, u1) = (blk * upb, units.min(blk * upb + upb));
        // Each warp's column sums, then combined in warp order.
        let mut warp_acc = vec![vec![0.0f32; dim]; nwarps];
        for (g, acc) in warp_acc.iter_mut().enumerate() {
            for u in (u0 + g..u1).step_by(nwarps) {
                let (r, h) = (u / heads, u % heads);
                let at = |win: Window, d: usize| win.at(r, h * dim + d);
                let ss = unit_sum(dim, |d| x[at(plan.x, d)] * x[at(plan.x, d)]);
                let rstd = 1.0f32 / (ss / fdim + plan.eps).sqrt();
                let dot = unit_sum(dim, |d| {
                    let xn = x[at(plan.x, d)] * rstd;
                    dy[at(win.dy, d)] * w[d] * silu(z[at(plan.z, d)]) * xn
                });
                let m = dot / fdim;
                for d in 0..dim {
                    let xn = x[at(plan.x, d)] * rstd;
                    let zv = z[at(plan.z, d)];
                    let s = silu(zv);
                    let g_d = dy[at(win.dy, d)];
                    let dxn = g_d * w[d] * s;
                    dx[at(win.dx, d)] = rstd * (dxn - xn * m);
                    dz[at(win.dz, d)] = g_d * w[d] * xn * silu_grad(zv);
                    acc[d] += g_d * xn * s;
                }
            }
        }
        for d in 0..dim {
            part[blk * dim + d] = warp_acc.iter().fold(0.0f32, |s, a| s + a[d]);
        }
    }
    Ok(crate::small_common::col_sum_blocks(&part, 0, nblocks, dim))
}

/// Whether two gradient windows of one buffer would write the same column
/// (for a caller placing `dx` and `dz` in one buffer on the host).
pub fn windows_collide(a: Window, b: Window, width: u64) -> bool {
    a.ld != b.ld || cols_overlap((a.off, width), (b.off, width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::splitmix_f32;

    #[test]
    fn plans_refuse_zero_sizes_bad_eps_and_wide_backward_rows() {
        assert!(RmsNormPlan::new(0, 8, 1e-6).is_err());
        assert!(RmsNormPlan::new(2, 0, 1e-6).is_err());
        assert!(RmsNormPlan::new(2, 8, 0.0).is_err());
        assert!(RmsNormPlan::new(2, 8, f32::INFINITY).is_err());
        assert!(RmsNormPlan::new(2, (1 << 24) + 1, 1e-6).is_err());
        let wide = RmsNormPlan::new(2, RMS_BWD_MAX_DIM + 1, 1e-6).unwrap();
        let n = usize::try_from(wide.len()).unwrap();
        let d = usize::try_from(wide.dim).unwrap();
        let p = usize::try_from(wide.part_len()).unwrap();
        let err = wide.check_bwd((n, d, n), (n, p, d)).unwrap_err();
        assert!(err.to_string().contains("exceeds the backward"), "{err}");
        let ok = RmsNormPlan::new(3, 2048, 1e-6).unwrap();
        assert_eq!((ok.bwd_blocks(), ok.part_len()), (1, 2048));
        assert!(ok.check_fwd(6144, 2048, 6143).is_err());
    }

    #[test]
    fn gated_plans_check_windows_and_the_dim_cap() {
        let x = Window::dense(32);
        let z = Window { ld: 80, off: 40 };
        let plan = GatedRmsNormPlan::new((3, 2, 16), 1e-6, x, z).unwrap();
        assert_eq!((plan.width(), plan.units(), plan.fwd_blocks()), (32, 6, 2));
        // z's last row ends at 2 * 80 + 40 + 32 = 232.
        assert!(plan
            .check_fwd((96, 232, 16), (Window::dense(32), 96))
            .is_ok());
        // One element short.
        assert!(plan
            .check_fwd((96, 231, 16), (Window::dense(32), 96))
            .is_err());
        // A z window wider than its stride.
        assert!(GatedRmsNormPlan::new((3, 2, 16), 1e-6, x, Window { ld: 70, off: 40 }).is_err());
        let wide = GatedRmsNormPlan::new((1, 1, 513), 1e-6, Window::dense(513), Window::dense(513))
            .unwrap();
        let e = wide
            .check_bwd(
                (513, 513, 513),
                (
                    (Window::dense(513), 513),
                    (Window::dense(513), 513),
                    (Window::dense(513), 513),
                ),
                (513, 513),
            )
            .unwrap_err();
        assert!(e.to_string().contains("exceeds the backward"), "{e}");
        assert!(windows_collide(
            Window { ld: 8, off: 0 },
            Window { ld: 8, off: 2 },
            4
        ));
        assert!(!windows_collide(
            Window { ld: 8, off: 0 },
            Window { ld: 8, off: 4 },
            4
        ));
    }

    #[test]
    fn the_rms_mirror_is_the_zero_centred_norm_and_close_to_f64() {
        let plan = RmsNormPlan::new(3, 300, 1e-6).unwrap();
        let x = splitmix_f32(7, 900, 2.0);
        let w = splitmix_f32(8, 300, 0.5);
        let y = rms_norm_fwd_mirror(&plan, &x, &w).unwrap();
        for r in 0..3 {
            let xr = &x[r * 300..(r + 1) * 300];
            let ms: f64 = xr.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / 300.0;
            let rs = 1.0 / (ms + 1e-6).sqrt();
            for d in 0..300 {
                let want = f64::from(xr[d]) * rs * (1.0 + f64::from(w[d]));
                let got = f64::from(y[r * 300 + d]);
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs() + 1e-6,
                    "r{r} d{d}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn the_backward_mirror_accumulates_into_dx_when_asked() {
        let plan = RmsNormPlan::new(40, 64, 1e-6).unwrap();
        let x = splitmix_f32(1, 2560, 1.0);
        let w = splitmix_f32(2, 64, 0.3);
        let dy = splitmix_f32(3, 2560, 1.0);
        let (dx, dw) = rms_norm_bwd_mirror(&plan, &x, &w, &dy, None).unwrap();
        let prev = vec![0.5f32; 2560];
        let (dx2, dw2) = rms_norm_bwd_mirror(&plan, &x, &w, &dy, Some(&prev)).unwrap();
        assert_eq!(dw, dw2);
        for (a, b) in dx.iter().zip(&dx2) {
            assert_eq!((0.5f32 + a).to_bits(), b.to_bits());
        }
        // Two blocks of 32 and 8 rows: the column sum covers both.
        assert_eq!(plan.bwd_blocks(), 2);
    }

    #[test]
    fn the_cuda_source_forms_one_plus_w_in_both_rms_norm_kernels_and_w_in_the_gated_one() {
        // The formula pins: a `* w[d]` rms_norm (the Qwen-vs-plain mutation)
        // fails here on any host, before a device ever runs it.
        let body = |entry: &str| {
            let start = SOURCE.find(&format!("void {entry}(")).expect(entry);
            let rest = &SOURCE[start..];
            let end = rest[1..].find("extern \"C\"").map_or(rest.len(), |i| i + 1);
            &rest[..end]
        };
        assert!(body(RMS_FWD).contains("yr[d] = xr[d] * inv * (1.0f + w[d]);"));
        assert!(body(RMS_BWD).contains("dot = dot + gr[d] * (1.0f + w[d]) * xv;"));
        assert!(body(RMS_BWD).contains("const float v = rstd * gr[d] * (1.0f + w[d]) - xr[d] * c;"));
        assert!(body(GATED_FWD).contains("yr[d] = w[d] * (xr[d] * inv) * qd_silu(zr[d]);"));
        assert!(!body(GATED_FWD).contains("1.0f + w"));
        assert!(!body(GATED_BWD).contains("1.0f + w"));
    }
}
