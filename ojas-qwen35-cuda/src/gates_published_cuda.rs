//! K3 on the device: [`crate::gates_published`]'s kernels (rule 9: the
//! `published` GDN gates, `g = -exp(A_log) softplus(a + dt_bias)` and
//! `beta = sigmoid(b)`), each launch validated through its plan against the
//! actual buffers first. Nothing here waits.

use cudarc::driver::PushKernelArg;

use crate::buffer::CudaBuffer;
use crate::error::CudaError;
use crate::gates_published::{
    GatesPublishedPlan, GATES_PUBLISHED_BWD, GATES_PUBLISHED_FWD, GATES_ROWS_PER_BLOCK, MODULE,
};
use crate::geometry::grid_1d;
use crate::runtime::{driver_error, CudaRuntime};
use crate::small_common_cuda::{cfg, col_sum_blocks, function};

/// `g` and `beta`, dense `[rows, heads]`, from `p`'s `a` and `b` columns.
pub fn gates_published(
    rt: &CudaRuntime,
    plan: &GatesPublishedPlan,
    (p, a_log, dt_bias): (&CudaBuffer<f32>, &CudaBuffer<f32>, &CudaBuffer<f32>),
    g: &mut CudaBuffer<f32>,
    beta: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_fwd((p.len(), a_log.len(), dt_bias.len()), (g.len(), beta.len()))?;
    let Some(l) = grid_1d(plan.len()) else {
        return Err(CudaError::invalid(
            GATES_PUBLISHED_FWD,
            "no (row, head) pairs",
        ));
    };
    let f = function(rt, &MODULE, GATES_PUBLISHED_FWD)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(p.slice())
        .arg(a_log.slice())
        .arg(dt_bias.slice())
        .arg(g.slice_mut())
        .arg(beta.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.heads)
        .arg(&plan.ld)
        .arg(&plan.a_off)
        .arg(&plan.b_off);
    // SAFETY: (const float* p, const float* a_log, const float* dt_bias, float*
    // g, float* beta, ull rows, ull heads, ull ld, ull a_off, ull b_off);
    // check_fwd bounded both column windows of p by p's length and matched g,
    // beta to rows * heads and a_log, dt_bias to heads. The grid strides over
    // rows * heads with one writer per element; g and beta are distinct
    // allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(GATES_PUBLISHED_FWD, e))?;
    Ok(())
}

/// The backward: `da`, `db` written into `dp`'s `a` and `b` columns (its
/// other columns untouched), and `dA_log`, `ddt_bias` overwritten through
/// `part` (`plan.part_len()` elements).
#[allow(clippy::too_many_arguments)]
pub fn gates_published_bwd(
    rt: &CudaRuntime,
    plan: &GatesPublishedPlan,
    (p, a_log, dt_bias): (&CudaBuffer<f32>, &CudaBuffer<f32>, &CudaBuffer<f32>),
    (dg, dbeta): (&CudaBuffer<f32>, &CudaBuffer<f32>),
    dp: &mut CudaBuffer<f32>,
    part: &mut CudaBuffer<f32>,
    (da_log, ddt_bias): (&mut CudaBuffer<f32>, &mut CudaBuffer<f32>),
) -> Result<(), CudaError> {
    plan.check_bwd(
        (p.len(), a_log.len(), dt_bias.len()),
        (dg.len(), dbeta.len(), dp.len()),
        (part.len(), da_log.len(), ddt_bias.len()),
    )?;
    let nblocks = plan.bwd_blocks();
    let threads = nblocks * plan.heads;
    let Some(l) = grid_1d(threads) else {
        return Err(CudaError::invalid(
            GATES_PUBLISHED_BWD,
            "no (block, head) pairs",
        ));
    };
    let f = function(rt, &MODULE, GATES_PUBLISHED_BWD)?;
    let rpb = GATES_ROWS_PER_BLOCK;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(p.slice())
        .arg(a_log.slice())
        .arg(dt_bias.slice())
        .arg(dg.slice())
        .arg(dbeta.slice())
        .arg(dp.slice_mut())
        .arg(part.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.heads)
        .arg(&plan.ld)
        .arg(&plan.a_off)
        .arg(&plan.b_off)
        .arg(&rpb);
    // SAFETY: (const float* p, a_log, dt_bias, dg, dbeta, float* dp, float*
    // part, ull rows, ull heads, ull ld, ull a_off, ull b_off, ull
    // rows_per_block); check_bwd bounded p's and dp's windows, refused
    // overlapping a and b windows (GatesPublishedPlan::new), and matched part
    // to 2 * nblocks * heads. Thread (blk, h) writes dp's (r, a_off + h) and
    // (r, b_off + h) for its rows only and part's two (blk, h) slots; dp and
    // part are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(GATES_PUBLISHED_BWD, e))?;
    col_sum_blocks(rt, part, 0, (nblocks, plan.heads), da_log)?;
    col_sum_blocks(rt, part, threads, (nblocks, plan.heads), ddt_bias)
}
