//! K6 on the device: [`crate::qk_norm_rope`]'s q/k norm + partial RoPE and the
//! output gate, each launch validated through its plan against the actual
//! buffers first. The RoPE table is the host's [`rope_inv_freq`], uploaded
//! once per model ([`upload_inv_freq`]). Nothing here waits.

use cudarc::driver::PushKernelArg;

use crate::buffer::CudaBuffer;
use crate::error::CudaError;
use crate::geometry::grid_1d;
use crate::qk_norm_rope::{
    rope_inv_freq, OutputGatePlan, QkNormRopePlan, GATE_BWD, GATE_FWD, MODULE, QK_BWD, QK_FWD,
    QK_ROWS_PER_BLOCK,
};
use crate::runtime::{driver_error, CudaRuntime};
use crate::small_common::{block_grid, Window, UNIT_THREADS};
use crate::small_common_cuda::{cfg, col_sum_blocks, function};

/// transformers' `inv_freq` (`rotary_dim / 2` entries), computed on the host
/// and uploaded: the only angle input the K6 kernels take.
pub fn upload_inv_freq(
    rt: &CudaRuntime,
    rotary_dim: u64,
    theta: f64,
) -> Result<CudaBuffer<f32>, CudaError> {
    let table = rope_inv_freq(rotary_dim, theta)?;
    rt.upload(&table, "rope inv_freq")
}

/// The q/k norm + RoPE forward: dense `q [rows, hq, dim]`, `k` and `v`
/// `[rows, hkv, dim]` from the fused projection `p`.
pub fn qk_norm_rope(
    rt: &CudaRuntime,
    plan: &QkNormRopePlan,
    (p, q_norm_w, k_norm_w, inv_freq): (
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
    ),
    (q, k, v): (
        &mut CudaBuffer<f32>,
        &mut CudaBuffer<f32>,
        &mut CudaBuffer<f32>,
    ),
) -> Result<(), CudaError> {
    plan.check_fwd(
        (p.len(), q_norm_w.len(), k_norm_w.len(), inv_freq.len()),
        (q.len(), k.len(), v.len()),
    )?;
    let l = block_grid(plan.fwd_blocks(), UNIT_THREADS, QK_FWD)?;
    let f = function(rt, &MODULE, QK_FWD)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(p.slice())
        .arg(q_norm_w.slice())
        .arg(k_norm_w.slice())
        .arg(inv_freq.slice())
        .arg(q.slice_mut())
        .arg(k.slice_mut())
        .arg(v.slice_mut())
        .arg(&plan.batch)
        .arg(&plan.seq)
        .arg(&plan.hq)
        .arg(&plan.hkv)
        .arg(&plan.dim)
        .arg(&plan.rot)
        .arg(&plan.ld_p)
        .arg(&plan.q_off)
        .arg(&plan.k_off)
        .arg(&plan.v_off)
        .arg(&plan.eps);
    // SAFETY: (const float* p, q_norm_w, k_norm_w, inv_freq, float* q_out,
    // k_out, v_out, ull batch, seq, hq, hkv, dim, rot, ld_p, q_off, k_off,
    // v_off, float eps); check_fwd bounded every region of p by p's length,
    // matched the norm weights to dim, inv_freq to rot / 2 and q, k, v to
    // their dense lengths. One warp per (token, head) unit writes that unit's
    // dim outputs only; q, k, v are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(QK_FWD, e))?;
    Ok(())
}

/// The q/k norm + RoPE backward: `dq`, `dk`, `dv` written into the q, k, v
/// columns of `dp` (its gate columns untouched), and the two norm weights'
/// gradients overwritten through `part` (`plan.part_len()` elements).
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_bwd(
    rt: &CudaRuntime,
    plan: &QkNormRopePlan,
    (p, q_norm_w, k_norm_w, inv_freq): (
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
        &CudaBuffer<f32>,
    ),
    (dq, dk, dv): (&CudaBuffer<f32>, &CudaBuffer<f32>, &CudaBuffer<f32>),
    dp: &mut CudaBuffer<f32>,
    part: &mut CudaBuffer<f32>,
    (dq_norm_w, dk_norm_w): (&mut CudaBuffer<f32>, &mut CudaBuffer<f32>),
) -> Result<(), CudaError> {
    plan.check_bwd(
        (p.len(), q_norm_w.len(), k_norm_w.len(), inv_freq.len()),
        (dq.len(), dk.len(), dv.len(), dp.len()),
        (part.len(), dq_norm_w.len(), dk_norm_w.len()),
    )?;
    let nblocks = plan.bwd_blocks();
    let l = block_grid(nblocks, UNIT_THREADS, QK_BWD)?;
    let f = function(rt, &MODULE, QK_BWD)?;
    let rpb = QK_ROWS_PER_BLOCK;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(p.slice())
        .arg(q_norm_w.slice())
        .arg(k_norm_w.slice())
        .arg(inv_freq.slice())
        .arg(dq.slice())
        .arg(dk.slice())
        .arg(dv.slice())
        .arg(dp.slice_mut())
        .arg(part.slice_mut())
        .arg(&plan.batch)
        .arg(&plan.seq)
        .arg(&plan.hq)
        .arg(&plan.hkv)
        .arg(&plan.dim)
        .arg(&plan.rot)
        .arg(&plan.ld_p)
        .arg(&plan.q_off)
        .arg(&plan.k_off)
        .arg(&plan.v_off)
        .arg(&plan.eps)
        .arg(&rpb);
    // SAFETY: the kernel's 21 parameters in this order (4 inputs, dq, dk, dv,
    // dp, part, 7 sizes, 4 layout values, eps, rows_per_block); check_bwd
    // bounded p's and dp's regions, capped dim at 32 lanes x 16 columns (the
    // per-lane arrays and the static shared array, sized for UNIT_THREADS = 4
    // warps, the block size here) and matched part to 2 * nblocks * dim. The
    // q, k, v regions share no column (QkNormRopePlan::new), so each dp
    // element has one writer; block blk writes part rows blk and nblocks + blk.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(QK_BWD, e))?;
    col_sum_blocks(rt, part, 0, (nblocks, plan.dim), dq_norm_w)?;
    col_sum_blocks(rt, part, nblocks * plan.dim, (nblocks, plan.dim), dk_norm_w)
}

/// `out[window] = attn * sigmoid(gate)`, the gate read in place from `p`.
pub fn output_gate(
    rt: &CudaRuntime,
    plan: &OutputGatePlan,
    (attn, p): (&CudaBuffer<f32>, &CudaBuffer<f32>),
    out_win: Window,
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_fwd(attn.len(), p.len(), (out_win, out.len()))?;
    let Some(l) = grid_1d(plan.rows * plan.width()) else {
        return Err(CudaError::invalid(GATE_FWD, "no output elements"));
    };
    let f = function(rt, &MODULE, GATE_FWD)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(attn.slice())
        .arg(p.slice())
        .arg(out.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.hq)
        .arg(&plan.dim)
        .arg(&plan.ld_p)
        .arg(&plan.q_off)
        .arg(&out_win.ld)
        .arg(&out_win.off);
    // SAFETY: (const float* attn, const float* p, float* out, ull rows, ull
    // hq, ull dim, ull ld_p, ull q_off, ull ld_out, ull out_off); check_fwd
    // matched attn to rows * hq * dim and bounded p's q/gate window and out's
    // window by their buffers. The grid strides over rows * hq * dim with one
    // writer per out element; out is a distinct allocation from attn and p.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(GATE_FWD, e))?;
    Ok(())
}

/// The output gate's backward: `d_attn` (dense, overwritten) and the gate
/// gradient written into the gate columns of `dp` (its other columns
/// untouched).
pub fn output_gate_bwd(
    rt: &CudaRuntime,
    plan: &OutputGatePlan,
    (attn, p): (&CudaBuffer<f32>, &CudaBuffer<f32>),
    (dy_win, dy): (Window, &CudaBuffer<f32>),
    d_attn: &mut CudaBuffer<f32>,
    dp: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_bwd(
        (attn.len(), p.len()),
        (dy_win, dy.len()),
        (d_attn.len(), dp.len()),
    )?;
    let Some(l) = grid_1d(plan.rows * plan.width()) else {
        return Err(CudaError::invalid(GATE_BWD, "no output elements"));
    };
    let f = function(rt, &MODULE, GATE_BWD)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(attn.slice())
        .arg(p.slice())
        .arg(dy.slice())
        .arg(d_attn.slice_mut())
        .arg(dp.slice_mut())
        .arg(&plan.rows)
        .arg(&plan.hq)
        .arg(&plan.dim)
        .arg(&plan.ld_p)
        .arg(&plan.q_off)
        .arg(&dy_win.ld)
        .arg(&dy_win.off);
    // SAFETY: (const float* attn, p, dy, float* d_attn, float* dp, ull rows,
    // hq, dim, ld_p, q_off, ld_dy, dy_off); check_bwd matched attn and d_attn
    // to rows * hq * dim and bounded p's, dp's and dy's windows. Element
    // (r, h, d) writes d_attn[i] and dp's (r, q_off + h*2D + D + d) once;
    // d_attn and dp are distinct allocations.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(GATE_BWD, e))?;
    Ok(())
}
