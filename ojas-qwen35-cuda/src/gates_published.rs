//! **K3: the GDN gates, for the GDN operator at the `published` rule**,
//! forward and backward (`cuda-backend-scoping.md` §3 K3): plans, CUDA-C
//! source and host mirrors. The device launches are
//! [`crate::gates_published_cuda`]. Under rule 9 every name here says
//! `published`: the gates are part of that operator (lead's ruling,
//! 2026-10-01, recorded in `HANDOFF/ojas-l-cuda-oracle-2026-10-01.md`).
//!
//! transformers (`modeling_qwen3_5.py:516,518`), over `[rows, heads]` gate
//! logits read from two column windows of the fused GDN projection:
//!
//! ```text
//! g    = -exp(A_log[h]) * softplus(a[r, h] + dt_bias[h])    torch softplus: linear above 20
//! beta = sigmoid(b[r, h])
//! ```
//!
//! Backward, from `dg`, `dbeta` (tessl `qwen35_gdn_gates_bwd_f32`,
//! `tessl/kernels/qwen35_bwd.metal:617-672`):
//!
//! ```text
//! da = dg * -exp(A_log) * softplus'(a + dt_bias)       softplus' = 1 above 20, sigmoid below
//! db = dbeta * beta * (1 - beta)
//! dA_log[h]   = sum over rows of dg * g                ddt_bias[h] = sum over rows of da
//! ```
//!
//! `da` and `db` go to the `a` and `b` columns of the projection's gradient
//! `dp` (same row stride as `p`). The per-head sums are per block of
//! [`GATES_ROWS_PER_BLOCK`] rows, ascending, into `part[blk, h]` (dA_log) and
//! `part[nblocks + blk, h]` (ddt_bias), then [`crate::small_common::COL_SUM`]
//! adds the blocks in order: no atomics, the same bits on every run.
//!
//! **Numerics.** `exp`, `softplus` and `sigmoid` are the crate's one copy,
//! L-cuda-M1's `k8_act` (lead's ruling, 2026-10-01), whose host emulations are
//! bit-identical to the device functions. Everything else is `+ - *`. So the
//! host mirrors here are **bitwise** mirrors of the kernels.

use crate::error::CudaError;
use crate::k8_act::{exp_f32, sigmoid_f32, softplus_f32};
use crate::kernels::KernelModule;
use crate::small_common::{
    add, blocks_for, col_sum_blocks, cols_overlap, exact_len, mul, nonzero, Window,
};

/// Rows per weight-gradient block of the backward (tessl
/// `GATES_ROWS_PER_BLOCK`, `tessl/src/qwen35_bwd.rs:711`).
pub const GATES_ROWS_PER_BLOCK: u64 = 256;

/// Entry: the forward, `g` and `beta`.
pub const GATES_PUBLISHED_FWD: &str = "qd_gdn_gates_published_f32";
/// Entry: the backward, `da`/`db` into `dp` and per-block partials.
pub const GATES_PUBLISHED_BWD: &str = "qd_gdn_gates_published_bwd_f32";

/// The K3 NVRTC module.
pub const MODULE: KernelModule = KernelModule {
    name: "k3_gates_published",
    source: SOURCE,
    entries: &[GATES_PUBLISHED_FWD, GATES_PUBLISHED_BWD],
};

const SOURCE: &str = concat!(
    crate::device_prelude!(),
    crate::act_prelude!(),
    crate::small_common::small_prelude!(),
    r#"
// g = -exp(A_log) * softplus(a + dt_bias): tessl qwen35_act.h qwen35_log_decay.
__device__ __forceinline__ float qd_gdn_log_decay_published(float a, float a_log, float dt_bias)
{
    return -qd_exp(a_log) * qd_softplus(a + dt_bias);
}

// One thread per (row, head): g and beta, dense [rows, heads].
extern "C" __global__ void qd_gdn_gates_published_f32(
    const float* p, const float* a_log, const float* dt_bias, float* g, float* beta,
    unsigned long long rows, unsigned long long heads, unsigned long long ld,
    unsigned long long a_off, unsigned long long b_off)
{
    const unsigned long long total = rows * heads;
    QD_GRID_STRIDE(o, total) {
        const unsigned long long r = o / heads;
        const unsigned long long h = o - r * heads;
        const float* row = p + r * ld;
        g[o] = qd_gdn_log_decay_published(row[a_off + h], a_log[h], dt_bias[h]);
        beta[o] = qd_sigmoid(row[b_off + h]);
    }
}

// One thread per (row block, head): da, db into dp's a and b columns, and the
// block's dA_log and ddt_bias partials.
extern "C" __global__ void qd_gdn_gates_published_bwd_f32(
    const float* p, const float* a_log, const float* dt_bias,
    const float* dg, const float* dbeta, float* dp, float* part,
    unsigned long long rows, unsigned long long heads, unsigned long long ld,
    unsigned long long a_off, unsigned long long b_off, unsigned long long rows_per_block)
{
    const unsigned long long nblocks = (rows + rows_per_block - 1) / rows_per_block;
    const unsigned long long total = nblocks * heads;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long blk = i / heads;
        const unsigned long long h = i - blk * heads;
        const unsigned long long r0 = blk * rows_per_block;
        const unsigned long long r1 = rows - r0 < rows_per_block ? rows : r0 + rows_per_block;
        const float neg_a = -qd_exp(a_log[h]);
        float acc_log = 0.0f;
        float acc_dt = 0.0f;
        for (unsigned long long r = r0; r < r1; ++r) {
            const float* row = p + r * ld;
            float* drow = dp + r * ld;
            const unsigned long long o = r * heads + h;
            const float a = row[a_off + h];
            const float xx = a + dt_bias[h];
            const float sp_grad = xx > 20.0f ? 1.0f : qd_sigmoid(xx);
            const float da = dg[o] * neg_a * sp_grad;
            const float s = qd_sigmoid(row[b_off + h]);
            drow[a_off + h] = da;
            drow[b_off + h] = dbeta[o] * s * (1.0f - s);
            acc_log = acc_log + dg[o] * qd_gdn_log_decay_published(a, a_log[h], dt_bias[h]);
            acc_dt = acc_dt + da;
        }
        part[blk * heads + h] = acc_log;
        part[(nblocks + blk) * heads + h] = acc_dt;
    }
}
"#
);

/// The gates over `rows` rows of a fused projection `p` with row stride `ld`:
/// `a` at columns `[a_off, a_off + heads)`, `b` at `[b_off, b_off + heads)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatesPublishedPlan {
    /// Rows (tokens).
    pub rows: u64,
    /// Value heads (16 at Qwen3.5-2B).
    pub heads: u64,
    /// Row stride of `p` and of its gradient `dp`.
    pub ld: u64,
    /// First `a` column.
    pub a_off: u64,
    /// First `b` column.
    pub b_off: u64,
}

impl GatesPublishedPlan {
    /// Validate the shape and both windows. The `a` and `b` windows must not
    /// share a column: the backward writes both into `dp`.
    pub fn new(
        rows: u64,
        heads: u64,
        ld: u64,
        (a_off, b_off): (u64, u64),
    ) -> Result<Self, CudaError> {
        const OP: &str = "gdn_gates_published";
        nonzero(OP, "rows", rows)?;
        nonzero(OP, "heads", heads)?;
        mul(rows, heads, OP)?;
        for (name, off) in [("a", a_off), ("b", b_off)] {
            if add(off, heads, OP)? > ld {
                return Err(CudaError::invalid(
                    OP,
                    format!(
                        "{name} columns {off}..{} exceed the row stride {ld}",
                        off + heads
                    ),
                ));
            }
        }
        if cols_overlap((a_off, heads), (b_off, heads)) {
            return Err(CudaError::invalid(
                OP,
                format!("a columns at {a_off} and b columns at {b_off} overlap over {heads} heads"),
            ));
        }
        Ok(GatesPublishedPlan {
            rows,
            heads,
            ld,
            a_off,
            b_off,
        })
    }

    /// Elements of `g`, `beta`, `dg`, `dbeta`.
    pub fn len(&self) -> u64 {
        self.rows * self.heads
    }

    /// Never true: a plan with no elements is refused.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Weight-gradient blocks of the backward.
    pub fn bwd_blocks(&self) -> u64 {
        blocks_for(self.rows, GATES_ROWS_PER_BLOCK)
    }

    /// Elements of the backward's `part` scratch: two halves of `[nblocks, heads]`.
    pub fn part_len(&self) -> u64 {
        2 * self.bwd_blocks() * self.heads
    }

    fn check_p(&self, op: &str, name: &str, len: usize) -> Result<(), CudaError> {
        // a and b are each a window of `heads` columns; the wider end bounds both.
        let off = self.a_off.max(self.b_off);
        Window { ld: self.ld, off }.check(op, name, self.rows, self.heads, len)
    }

    /// Check the forward's buffers.
    pub fn check_fwd(
        &self,
        (p, a_log, dt_bias): (usize, usize, usize),
        (g, beta): (usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "gdn_gates_published";
        self.check_p(OP, "p", p)?;
        exact_len(OP, "A_log", a_log, self.heads)?;
        exact_len(OP, "dt_bias", dt_bias, self.heads)?;
        exact_len(OP, "g", g, self.len())?;
        exact_len(OP, "beta", beta, self.len())
    }

    /// Check the backward's buffers.
    pub fn check_bwd(
        &self,
        (p, a_log, dt_bias): (usize, usize, usize),
        (dg, dbeta, dp): (usize, usize, usize),
        (part, da_log, ddt_bias): (usize, usize, usize),
    ) -> Result<(), CudaError> {
        const OP: &str = "gdn_gates_published_bwd";
        self.check_fwd((p, a_log, dt_bias), (dg, dbeta))?;
        self.check_p(OP, "dp", dp)?;
        exact_len(OP, "part", part, self.part_len())?;
        exact_len(OP, "dA_log", da_log, self.heads)?;
        exact_len(OP, "ddt_bias", ddt_bias, self.heads)
    }
}

fn dims(plan: &GatesPublishedPlan) -> (usize, usize, usize, usize, usize) {
    let u = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
    (
        u(plan.rows),
        u(plan.heads),
        u(plan.ld),
        u(plan.a_off),
        u(plan.b_off),
    )
}

/// `-exp(A_log) * softplus(a + dt_bias)` with `k8_act`'s host functions.
pub fn log_decay_published_f32(a: f32, a_log: f32, dt_bias: f32) -> f32 {
    -exp_f32(a_log) * softplus_f32(a + dt_bias)
}

/// `qd_gdn_gates_published_f32` on the host: `(g, beta)`.
pub fn gates_published_fwd_mirror(
    plan: &GatesPublishedPlan,
    p: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    let n = usize::try_from(plan.len()).unwrap_or(usize::MAX);
    plan.check_fwd((p.len(), a_log.len(), dt_bias.len()), (n, n))?;
    let (rows, heads, ld, a_off, b_off) = dims(plan);
    let mut g = vec![0.0f32; n];
    let mut beta = vec![0.0f32; n];
    for r in 0..rows {
        for h in 0..heads {
            let o = r * heads + h;
            g[o] = log_decay_published_f32(p[r * ld + a_off + h], a_log[h], dt_bias[h]);
            beta[o] = sigmoid_f32(p[r * ld + b_off + h]);
        }
    }
    Ok((g, beta))
}

/// What [`gates_published_bwd_mirror`] returns besides `dp`.
#[derive(Clone, Debug, PartialEq)]
pub struct GatesPublishedGrads {
    /// `[heads]`.
    pub da_log: Vec<f32>,
    /// `[heads]`.
    pub ddt_bias: Vec<f32>,
}

/// `qd_gdn_gates_published_bwd_f32` then the two column sums on the host.
/// Writes `da`, `db` into `dp`'s `a` and `b` columns; leaves the rest of `dp`.
pub fn gates_published_bwd_mirror(
    plan: &GatesPublishedPlan,
    (p, a_log, dt_bias): (&[f32], &[f32], &[f32]),
    (dg, dbeta): (&[f32], &[f32]),
    dp: &mut [f32],
) -> Result<GatesPublishedGrads, CudaError> {
    let u = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
    let part_len = u(plan.part_len());
    plan.check_bwd(
        (p.len(), a_log.len(), dt_bias.len()),
        (dg.len(), dbeta.len(), dp.len()),
        (part_len, a_log.len(), dt_bias.len()),
    )?;
    let (rows, heads, ld, a_off, b_off) = dims(plan);
    let rpb = u(GATES_ROWS_PER_BLOCK);
    let nblocks = u(plan.bwd_blocks());
    let mut part = vec![0.0f32; part_len];
    for blk in 0..nblocks {
        for h in 0..heads {
            let (r0, r1) = (blk * rpb, rows.min(blk * rpb + rpb));
            let neg_a = -exp_f32(a_log[h]);
            let (mut acc_log, mut acc_dt) = (0.0f32, 0.0f32);
            for r in r0..r1 {
                let o = r * heads + h;
                let a = p[r * ld + a_off + h];
                let xx = a + dt_bias[h];
                let sp_grad = if xx > 20.0 { 1.0 } else { sigmoid_f32(xx) };
                let da = dg[o] * neg_a * sp_grad;
                let s = sigmoid_f32(p[r * ld + b_off + h]);
                dp[r * ld + a_off + h] = da;
                dp[r * ld + b_off + h] = dbeta[o] * s * (1.0 - s);
                acc_log += dg[o] * log_decay_published_f32(a, a_log[h], dt_bias[h]);
                acc_dt += da;
            }
            part[blk * heads + h] = acc_log;
            part[(nblocks + blk) * heads + h] = acc_dt;
        }
    }
    Ok(GatesPublishedGrads {
        da_log: col_sum_blocks(&part, 0, nblocks, heads),
        ddt_bias: col_sum_blocks(&part, nblocks * heads, nblocks, heads),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::splitmix_f32;

    #[test]
    fn rule_9_every_k3_name_says_published() {
        assert!(MODULE.name.contains("published"));
        for e in MODULE.entries {
            assert!(e.contains("published"), "{e}");
        }
        for name in crate::small_common::defined_entries(MODULE.source) {
            assert!(name.contains("published"), "{name}");
        }
    }

    #[test]
    fn the_plan_refuses_overlapping_or_out_of_stride_windows() {
        assert!(GatesPublishedPlan::new(4, 16, 64, (32, 48)).is_ok());
        assert!(GatesPublishedPlan::new(4, 16, 64, (32, 40)).is_err());
        assert!(GatesPublishedPlan::new(4, 16, 63, (32, 48)).is_err());
        assert!(GatesPublishedPlan::new(0, 16, 64, (32, 48)).is_err());
        let plan = GatesPublishedPlan::new(300, 3, 10, (1, 5)).unwrap();
        assert_eq!((plan.bwd_blocks(), plan.part_len()), (2, 12));
        // p must hold (rows - 1) * ld + max(a_off, b_off) + heads = 2998.
        assert!(plan.check_fwd((2998, 3, 3), (900, 900)).is_ok());
        assert!(plan.check_fwd((2997, 3, 3), (900, 900)).is_err());
    }

    #[test]
    fn the_cuda_source_keeps_torchs_threshold_and_the_published_decay() {
        assert!(SOURCE.contains("return -qd_exp(a_log) * qd_softplus(a + dt_bias);"));
        assert!(SOURCE.contains("const float sp_grad = xx > 20.0f ? 1.0f : qd_sigmoid(xx);"));
        assert!(SOURCE.contains("drow[b_off + h] = dbeta[o] * s * (1.0f - s);"));
    }

    #[test]
    fn the_mirror_writes_only_the_gate_columns_of_dp() {
        let plan = GatesPublishedPlan::new(5, 2, 7, (1, 4)).unwrap();
        let p = splitmix_f32(3, 35, 3.0);
        let al = splitmix_f32(4, 2, 1.0);
        let dt = splitmix_f32(5, 2, 1.0);
        let dg = splitmix_f32(6, 10, 1.0);
        let db = splitmix_f32(7, 10, 1.0);
        let mut dp = vec![-9.0f32; 35];
        gates_published_bwd_mirror(&plan, (&p, &al, &dt), (&dg, &db), &mut dp).unwrap();
        for r in 0..5 {
            for c in 0..7 {
                let gate_col = (1..3).contains(&c) || (4..6).contains(&c);
                assert_eq!(dp[r * 7 + c] == -9.0, !gate_col, "row {r} col {c}");
            }
        }
    }
}
