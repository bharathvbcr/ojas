//! K7 on the device: [`crate::rmsnorm`]'s kernels, each launch validated
//! through its plan against the actual buffers first. Nothing here waits.
//! The weight gradients go through a caller-owned `part` scratch of the
//! plan's `part_len()` and [`col_sum_blocks`], so they are deterministic.

use cudarc::driver::PushKernelArg;

use crate::buffer::CudaBuffer;
use crate::error::CudaError;
use crate::rmsnorm::{
    GatedGradWindows, GatedRmsNormPlan, RmsNormPlan, GATED_BWD, GATED_FWD, GATED_UNITS_PER_BLOCK,
    MODULE, RMS_BWD, RMS_FWD, RMS_ROWS_PER_BLOCK,
};
use crate::runtime::{driver_error, CudaRuntime};
use crate::small_common::{block_grid, Window, ROW_THREADS, UNIT_THREADS};
use crate::small_common_cuda::{cfg, col_sum_blocks, function};

/// `out = x * rstd * (1 + w)` per row.
pub fn rms_norm(
    rt: &CudaRuntime,
    plan: &RmsNormPlan,
    x: &CudaBuffer<f32>,
    w: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_fwd(x.len(), w.len(), out.len())?;
    let l = block_grid(plan.rows, ROW_THREADS, RMS_FWD)?;
    let f = function(rt, &MODULE, RMS_FWD)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(x.slice())
        .arg(w.slice())
        .arg(out.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.dim)
        .arg(&plan.eps);
    // SAFETY: (const float* x, const float* w, float* out, ull rows, ull dim,
    // float eps); x and out hold rows * dim and w holds dim (check_fwd). One
    // block of ROW_THREADS (a multiple of 32) per row, as qd_block_sum needs;
    // block r touches row r only. x and out are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(RMS_FWD, e))?;
    Ok(())
}

/// The backward: `dx` (overwritten, or added to with `accumulate`) and `dw`
/// (overwritten), through `part` (`plan.part_len()` elements).
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_bwd(
    rt: &CudaRuntime,
    plan: &RmsNormPlan,
    (x, w, dy): (&CudaBuffer<f32>, &CudaBuffer<f32>, &CudaBuffer<f32>),
    dx: &mut CudaBuffer<f32>,
    accumulate: bool,
    part: &mut CudaBuffer<f32>,
    dw: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_bwd(
        (x.len(), w.len(), dy.len()),
        (dx.len(), part.len(), dw.len()),
    )?;
    let nblocks = plan.bwd_blocks();
    let l = block_grid(nblocks, ROW_THREADS, RMS_BWD)?;
    let f = function(rt, &MODULE, RMS_BWD)?;
    let acc: i32 = i32::from(accumulate);
    let rpb = RMS_ROWS_PER_BLOCK;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(x.slice())
        .arg(w.slice())
        .arg(dy.slice())
        .arg(dx.slice_mut())
        .arg(part.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.dim)
        .arg(&plan.eps)
        .arg(&rpb)
        .arg(&acc);
    // SAFETY: (const float* x, const float* w, const float* dy, float* dx,
    // float* part, ull rows, ull dim, float eps, ull rows_per_block, int
    // accumulate); check_bwd matched every length (part = nblocks * dim) and
    // capped dim at ROW_THREADS * 16 columns, the kernel's per-thread
    // accumulators. Block blk writes rows [blk*32, ...) of dx and row blk of
    // part only. The three outputs are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(RMS_BWD, e))?;
    col_sum_blocks(rt, part, 0, (nblocks, plan.dim), dw)
}

/// `out[window] = w * (x * rstd) * silu(z)` per (row, head).
pub fn gated_rms_norm(
    rt: &CudaRuntime,
    plan: &GatedRmsNormPlan,
    (x, z, w): (&CudaBuffer<f32>, &CudaBuffer<f32>, &CudaBuffer<f32>),
    out_win: Window,
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_fwd((x.len(), z.len(), w.len()), (out_win, out.len()))?;
    let l = block_grid(plan.fwd_blocks(), UNIT_THREADS, GATED_FWD)?;
    let f = function(rt, &MODULE, GATED_FWD)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(x.slice())
        .arg(z.slice())
        .arg(w.slice())
        .arg(out.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.heads)
        .arg(&plan.dim)
        .arg(&plan.x.ld)
        .arg(&plan.x.off)
        .arg(&plan.z.ld)
        .arg(&plan.z.off)
        .arg(&out_win.ld)
        .arg(&out_win.off)
        .arg(&plan.eps);
    // SAFETY: (const float* x, const float* z, const float* w, float* out,
    // ull rows, ull heads, ull dim, ull ld_x, ull x_off, ull ld_z, ull z_off,
    // ull ld_out, ull out_off, float eps); check_fwd bounded all three
    // windows by their buffers. One warp per (row, head) unit, UNIT_THREADS
    // per block; each unit writes its own dim columns of out.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(GATED_FWD, e))?;
    Ok(())
}

/// The gated backward: `dx` and `dz` into their windows, `dw` overwritten,
/// through `part` (`plan.part_len()` elements).
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_bwd(
    rt: &CudaRuntime,
    plan: &GatedRmsNormPlan,
    (x, z, w, dy): (
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
    ),
    win: GatedGradWindows,
    (dx, dz): (&mut CudaBuffer<f32>, &mut CudaBuffer<f32>),
    part: &mut CudaBuffer<f32>,
    dw: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_bwd(
        (x.len(), z.len(), w.len()),
        ((win.dy, dy.len()), (win.dx, dx.len()), (win.dz, dz.len())),
        (part.len(), dw.len()),
    )?;
    let nblocks = plan.bwd_blocks();
    let l = block_grid(nblocks, UNIT_THREADS, GATED_BWD)?;
    let f = function(rt, &MODULE, GATED_BWD)?;
    let upb = GATED_UNITS_PER_BLOCK;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(x.slice())
        .arg(z.slice())
        .arg(w.slice())
        .arg(dy.slice())
        .arg(dx.slice_mut())
        .arg(dz.slice_mut())
        .arg(part.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.heads)
        .arg(&plan.dim)
        .arg(&plan.x.ld)
        .arg(&plan.x.off)
        .arg(&plan.z.ld)
        .arg(&plan.z.off)
        .arg(&win.dy.ld)
        .arg(&win.dy.off)
        .arg(&win.dx.ld)
        .arg(&win.dx.off)
        .arg(&win.dz.ld)
        .arg(&win.dz.off)
        .arg(&plan.eps)
        .arg(&upb);
    // SAFETY: the kernel's 22 parameters in this order (4 inputs, 3 outputs,
    // 3 sizes, 5 (ld, off) pairs, eps, units_per_block); check_bwd bounded
    // every window by its buffer, matched part to nblocks * dim and capped dim
    // at 32 * 16 columns, the per-lane accumulators and the static shared
    // array. Each (row, head) unit is handled by one warp of one block, which
    // writes its own columns of dx and dz; block blk writes row blk of part.
    // dx, dz and part are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(GATED_BWD, e))?;
    col_sum_blocks(rt, part, 0, (nblocks, plan.dim), dw)
}
