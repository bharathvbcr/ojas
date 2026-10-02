//! K4 on the device: [`crate::conv1d`]'s causal depthwise conv + SiLU, each
//! launch validated through its plan against the actual buffers first.
//! Nothing here waits.

use cudarc::driver::PushKernelArg;

use crate::buffer::CudaBuffer;
use crate::conv1d::{Conv1dPlan, CONV_BWD_DW, CONV_BWD_DX, CONV_FWD, CONV_ROWS_PER_BLOCK, MODULE};
use crate::error::CudaError;
use crate::geometry::grid_1d;
use crate::runtime::{driver_error, CudaRuntime};
use crate::small_common::Window;
use crate::small_common_cuda::{cfg, col_sum_blocks, function};

/// `y = silu(causal_conv(x, w))`, `y` dense `[batch * seq, channels]`.
pub fn conv1d_silu(
    rt: &CudaRuntime,
    plan: &Conv1dPlan,
    x: &CudaBuffer<f32>,
    w: &CudaBuffer<f32>,
    y: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_fwd(x.len(), w.len(), y.len())?;
    let Some(l) = grid_1d(plan.y_len()) else {
        return Err(CudaError::invalid(CONV_FWD, "no output elements"));
    };
    let f = function(rt, &MODULE, CONV_FWD)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(x.slice())
        .arg(w.slice())
        .arg(y.slice_mut())
        .arg(&plan.batch)
        .arg(&plan.seq)
        .arg(&plan.channels)
        .arg(&plan.kw)
        .arg(&plan.x.ld)
        .arg(&plan.x.off);
    // SAFETY: (const float* x, const float* w, float* y, ull batch, ull seq,
    // ull channels, unsigned int kw, ull ld_x, ull x_off); check_fwd bounded
    // x's window by x's length and matched w to channels * kw (kw in 2..=8,
    // Conv1dPlan::new) and y to batch * seq * channels. The grid strides over
    // y with one writer per element; x and y are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(CONV_FWD, e))?;
    Ok(())
}

/// The backward: `dx` written into its window (overwritten) and `dw`
/// (overwritten) through `part` (`plan.part_len()` elements).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_silu_bwd(
    rt: &CudaRuntime,
    plan: &Conv1dPlan,
    (x, w): (&CudaBuffer<f32>, &CudaBuffer<f32>),
    (dy_win, dy): (Window, &CudaBuffer<f32>),
    (dx_win, dx): (Window, &mut CudaBuffer<f32>),
    part: &mut CudaBuffer<f32>,
    dw: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_bwd(
        (x.len(), w.len()),
        ((dy_win, dy.len()), (dx_win, dx.len())),
        (part.len(), dw.len()),
    )?;
    let Some(l_dx) = grid_1d(plan.y_len()) else {
        return Err(CudaError::invalid(CONV_BWD_DX, "no input elements"));
    };
    let f_dx = function(rt, &MODULE, CONV_BWD_DX)?;
    let mut b = rt.stream().launch_builder(&f_dx);
    b.arg(x.slice())
        .arg(w.slice())
        .arg(dy.slice())
        .arg(dx.slice_mut())
        .arg(&plan.batch)
        .arg(&plan.seq)
        .arg(&plan.channels)
        .arg(&plan.kw)
        .arg(&plan.x.ld)
        .arg(&plan.x.off)
        .arg(&dy_win.ld)
        .arg(&dy_win.off)
        .arg(&dx_win.ld)
        .arg(&dx_win.off);
    // SAFETY: (const float* x, w, dy, float* dx, ull batch, seq, channels,
    // unsigned int kw, ull ld_x, x_off, ld_dy, dy_off, ld_dx, dx_off);
    // check_bwd bounded the x, dy and dx windows by their buffers. The grid
    // strides over batch * seq * channels and element (row, c) writes dx's
    // (row, dx_off + c) once; dx is a distinct allocation from x and dy.
    unsafe { b.launch(cfg(l_dx)) }.map_err(|e| driver_error(CONV_BWD_DX, e))?;

    let nblocks = plan.bwd_blocks();
    let Some(l_dw) = grid_1d(nblocks * plan.channels) else {
        return Err(CudaError::invalid(CONV_BWD_DW, "no (block, channel) pairs"));
    };
    let f_dw = function(rt, &MODULE, CONV_BWD_DW)?;
    let rpb = CONV_ROWS_PER_BLOCK;
    let mut b = rt.stream().launch_builder(&f_dw);
    b.arg(x.slice())
        .arg(w.slice())
        .arg(dy.slice())
        .arg(part.slice_mut())
        .arg(&plan.batch)
        .arg(&plan.seq)
        .arg(&plan.channels)
        .arg(&plan.kw)
        .arg(&plan.x.ld)
        .arg(&plan.x.off)
        .arg(&dy_win.ld)
        .arg(&dy_win.off)
        .arg(&rpb);
    // SAFETY: (const float* x, w, dy, float* part, ull batch, seq, channels,
    // unsigned int kw, ull ld_x, x_off, ld_dy, dy_off, rows_per_block);
    // check_bwd matched part to nblocks * channels * kw. Thread (blk, c)
    // writes part's kw slots at blk * channels * kw + c * kw only, and kw <= 8
    // fits the kernel's per-thread accumulator array.
    unsafe { b.launch(cfg(l_dw)) }.map_err(|e| driver_error(CONV_BWD_DW, e))?;
    col_sum_blocks(rt, part, 0, (nblocks, plan.w_len()), dw)
}
