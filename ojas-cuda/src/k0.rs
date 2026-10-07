//! K0 plumbing kernels on the device. Each function validates through its
//! [`crate::k0_plan`] plan, then queues one launch on the runtime's stream;
//! nothing here waits. The host reference for each is in
//! [`crate::host_ref`], and must match bitwise.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::buffer::{BufView, BufViewMut, CudaBuffer};
use crate::error::CudaError;
use crate::geometry::{grid_1d, Launch};
use crate::k0_plan::{
    CopyColsPlan, DeliverMode, DeliverPlan, GatherRowsPlan, ScatterAddRowsPlan, ZeroPlan,
};
use crate::kernels::{K0, STRICT_SM90};
use crate::runtime::{driver_error, CudaRuntime};

fn cfg(l: Launch) -> LaunchConfig {
    LaunchConfig {
        grid_dim: l.grid,
        block_dim: l.block,
        shared_mem_bytes: 0,
    }
}

fn len_u64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

fn k0(rt: &CudaRuntime, entry: &str) -> Result<CudaFunction, CudaError> {
    rt.function(&K0, &STRICT_SM90, entry)
}

/// `dst = bf16(src)`, round-to-nearest-even.
pub fn cast_f32_to_bf16(
    rt: &CudaRuntime,
    src: BufView<'_, f32>,
    mut dst: BufViewMut<'_, u16>,
) -> Result<(), CudaError> {
    const E: &str = "qd_cast_f32_to_bf16";
    if src.len() != dst.len() {
        return Err(CudaError::invalid(
            E,
            format!("{} -> {} elements", src.len(), dst.len()),
        ));
    }
    let n = len_u64(src.len());
    let Some(l) = grid_1d(n) else { return Ok(()) };
    let f = k0(rt, E)?;
    let src_dev = src.device();
    let mut dst_dev = dst.device();
    let mut b = rt.stream().launch_builder(&f);
    b.arg(&src_dev).arg(&mut dst_dev).arg(&n);
    // SAFETY: the kernel takes (const float*, unsigned short*, unsigned long
    // long n); both views hold n elements inside their buffers (view_range)
    // and it touches indices < n only.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// `dst = f32(src)`, exact.
pub fn cast_bf16_to_f32(
    rt: &CudaRuntime,
    src: &CudaBuffer<u16>,
    dst: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_cast_bf16_to_f32";
    if src.len() != dst.len() {
        return Err(CudaError::invalid(
            E,
            format!("{} -> {} elements", src.len(), dst.len()),
        ));
    }
    let n = len_u64(src.len());
    let Some(l) = grid_1d(n) else { return Ok(()) };
    let f = k0(rt, E)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(src.slice()).arg(dst.slice_mut()).arg(&n);
    // SAFETY: (const unsigned short*, float*, unsigned long long n); both
    // buffers hold n elements.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// [`CopyColsPlan`] on the device. The plan must have been built against
/// these buffers' lengths; this re-checks that it was.
pub fn copy_cols(
    rt: &CudaRuntime,
    plan: &CopyColsPlan,
    src: &CudaBuffer<f32>,
    dst: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_copy_cols_f32";
    let checked = CopyColsPlan::new(
        plan.rows,
        plan.width,
        (plan.ld_src, plan.src_off, src.len()),
        (plan.ld_dst, plan.dst_off, dst.len()),
    )?;
    let Some(l) = grid_1d(checked.total) else {
        return Ok(());
    };
    let f = k0(rt, E)?;
    let p = checked;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(src.slice())
        .arg(dst.slice_mut())
        .arg(&p.rows)
        .arg(&p.width)
        .arg(&p.ld_src)
        .arg(&p.src_off)
        .arg(&p.ld_dst)
        .arg(&p.dst_off);
    // SAFETY: (const float*, float*, 6 x unsigned long long) in this order;
    // the plan bounds every index by both buffers' lengths.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// [`DeliverPlan`] on the device: copy or `+=`.
pub fn deliver(
    rt: &CudaRuntime,
    plan: &DeliverPlan,
    src: &CudaBuffer<f32>,
    dst: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    let p = DeliverPlan::new(
        (plan.src_off, src.len()),
        (plan.dst_off, dst.len()),
        plan.n,
        plan.mode,
    )?;
    let e = match p.mode {
        DeliverMode::Copy => "qd_deliver_copy_f32",
        DeliverMode::Add => "qd_deliver_add_f32",
    };
    let Some(l) = grid_1d(p.n) else { return Ok(()) };
    let f = k0(rt, e)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(src.slice())
        .arg(&p.src_off)
        .arg(dst.slice_mut())
        .arg(&p.dst_off)
        .arg(&p.n);
    // SAFETY: (const float*, ull, float*, ull, ull n); the plan bounds
    // off + n by each buffer's length. src and dst are distinct allocations
    // (one shared and one exclusive borrow).
    unsafe { b.launch(cfg(l)) }.map_err(|err| driver_error(e, err))?;
    Ok(())
}

/// [`ZeroPlan`] on the device.
pub fn zero(rt: &CudaRuntime, plan: &ZeroPlan, dst: &mut CudaBuffer<f32>) -> Result<(), CudaError> {
    const E: &str = "qd_zero_f32";
    let p = ZeroPlan::new(plan.off, plan.n, dst.len())?;
    let Some(l) = grid_1d(p.n) else { return Ok(()) };
    let f = k0(rt, E)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(dst.slice_mut()).arg(&p.off).arg(&p.n);
    // SAFETY: (float*, ull off, ull n); off + n <= len by the plan.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// `dst[pos[i]] += src[i]` row by row, `dst` being `[dst_rows, width]`.
/// Positions are validated (in range, distinct) on the host, then uploaded.
pub fn scatter_add_rows(
    rt: &CudaRuntime,
    src: &CudaBuffer<f32>,
    pos: &[u32],
    width: u64,
    dst_rows: u64,
    dst: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_scatter_add_rows_f32";
    let p = ScatterAddRowsPlan::new(pos, width, src.len(), dst_rows, dst.len())?;
    let Some(l) = grid_1d(p.total) else {
        return Ok(());
    };
    let f = k0(rt, E)?;
    // Freed in stream order after the launch (cudarc's free_async).
    let pos_dev = rt.upload(pos, "scatter_add_rows positions")?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(src.slice())
        .arg(pos_dev.slice())
        .arg(dst.slice_mut())
        .arg(&p.n)
        .arg(&p.width);
    // SAFETY: (const float*, const unsigned int*, float*, ull n, ull width);
    // src is [n, width], pos has n entries below dst_rows, dst is
    // [dst_rows, width], and distinct positions give each element one writer.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// Which element type a gather reads.
pub enum GatherSource<'a> {
    /// f32 hidden states.
    F32(&'a CudaBuffer<f32>),
    /// bf16 hidden states, widened to f32.
    Bf16(&'a CudaBuffer<u16>),
}

/// `out[i] = h[rows[i] * ld + off ..][..hidden]`, widened to f32.
pub fn ce_gather_rows(
    rt: &CudaRuntime,
    h: GatherSource<'_>,
    rows: &[u32],
    hidden: u64,
    (ld, off): (u64, u64),
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    let (e, h_len) = match &h {
        GatherSource::F32(b) => ("qd_ce_gather_rows_f32", b.len()),
        GatherSource::Bf16(b) => ("qd_ce_gather_rows_bf16", b.len()),
    };
    let p = GatherRowsPlan::new(rows, hidden, (ld, off, h_len), out.len())?;
    let Some(l) = grid_1d(p.total) else {
        return Ok(());
    };
    let f = k0(rt, e)?;
    let rows_dev = rt.upload(rows, "ce_gather_rows rows")?;
    let mut b = rt.stream().launch_builder(&f);
    match &h {
        GatherSource::F32(buf) => b.arg(buf.slice()),
        GatherSource::Bf16(buf) => b.arg(buf.slice()),
    };
    b.arg(rows_dev.slice())
        .arg(out.slice_mut())
        .arg(&p.n_rows)
        .arg(&p.hidden)
        .arg(&p.ld)
        .arg(&p.off);
    // SAFETY: (const T* h, const unsigned int* rows, float* out, ull n_rows,
    // ull hidden, ull ld, ull off), T matching the entry; the plan bounds
    // every row's read by h's length and out is [n_rows, hidden].
    unsafe { b.launch(cfg(l)) }.map_err(|err| driver_error(e, err))?;
    Ok(())
}
