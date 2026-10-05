//! K8 on the device: SwiGLU forward and backward, the residual add and the
//! activation sweep. Each function re-checks its [`crate::k8_plan`] plan
//! against the buffers it is given, then queues one launch; nothing waits.

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::error::CudaError;
use crate::geometry::{grid_1d, Launch};
use crate::k8_kernels::K8;
use crate::k8_plan::{BwdOut, ResidualAddPlan, SwigluBwdPlan, SwigluPlan, ACT_SWEEP};
use crate::kernels::STRICT_SM90;
use crate::runtime::{driver_error, CudaRuntime};

fn cfg(l: Launch) -> LaunchConfig {
    LaunchConfig {
        grid_dim: l.grid,
        block_dim: l.block,
        shared_mem_bytes: 0,
    }
}

fn k8(rt: &CudaRuntime, entry: &str) -> Result<CudaFunction, CudaError> {
    rt.function(&K8, &STRICT_SM90, entry)
}

/// `out = silu(gate) * up`, f32 out. `gate` and `up` may be one buffer.
pub fn swiglu_f32(
    rt: &CudaRuntime,
    plan: &SwigluPlan,
    gate: &CudaBuffer<f32>,
    up: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_swiglu_f32";
    plan.check(gate.len(), up.len(), out.len())?;
    let Some(l) = grid_1d(plan.total) else {
        return Ok(());
    };
    let f = k8(rt, E)?;
    let p = plan;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(gate.slice())
        .arg(up.slice())
        .arg(out.slice_mut())
        .arg(&p.rows)
        .arg(&p.width)
        .arg(&p.gate.ld)
        .arg(&p.gate.off)
        .arg(&p.up.ld)
        .arg(&p.up.off)
        .arg(&p.out.ld)
        .arg(&p.out.off);
    // SAFETY: (const float*, const float*, float*, 8 x unsigned long long) in
    // this order. The plan, re-checked against these lengths, bounds every
    // window index; `out` is a distinct `&mut` borrow, so it aliases no input.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// The same product rounded once to bf16, for the down projection.
pub fn swiglu_bf16(
    rt: &CudaRuntime,
    plan: &SwigluPlan,
    gate: &CudaBuffer<f32>,
    up: &CudaBuffer<f32>,
    out: &mut CudaBuffer<u16>,
) -> Result<(), CudaError> {
    const E: &str = "qd_swiglu_bf16";
    plan.check(gate.len(), up.len(), out.len())?;
    let Some(l) = grid_1d(plan.total) else {
        return Ok(());
    };
    let f = k8(rt, E)?;
    let p = plan;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(gate.slice())
        .arg(up.slice())
        .arg(out.slice_mut())
        .arg(&p.rows)
        .arg(&p.width)
        .arg(&p.gate.ld)
        .arg(&p.gate.off)
        .arg(&p.up.ld)
        .arg(&p.up.off)
        .arg(&p.out.ld)
        .arg(&p.out.off);
    // SAFETY: (const float*, const float*, unsigned short*, 8 x unsigned long
    // long); the plan bounds every index by these buffers' lengths, and `out`
    // is a distinct `&mut` borrow.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// The backward into two buffers ([`BwdOut::Separate`]).
pub fn swiglu_bwd_separate(
    rt: &CudaRuntime,
    plan: &SwigluBwdPlan,
    (gate, up, dy): (&CudaBuffer<f32>, &CudaBuffer<f32>, &CudaBuffer<f32>),
    dgate: &mut CudaBuffer<f32>,
    dup: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_swiglu_bwd_f32";
    if plan.out != BwdOut::Separate {
        return Err(CudaError::invalid(E, "the plan writes one shared buffer"));
    }
    let rebuilt = SwigluBwdPlan::new(
        plan.rows,
        plan.width,
        (plan.gate.ld, plan.gate.off, gate.len()),
        (plan.up.ld, plan.up.off, up.len()),
        (plan.dy.ld, plan.dy.off, dy.len()),
        (plan.dgate.ld, plan.dgate.off, dgate.len()),
        (plan.dup.ld, plan.dup.off, dup.len()),
        BwdOut::Separate,
    )?;
    let Some(l) = grid_1d(rebuilt.total) else {
        return Ok(());
    };
    let f = k8(rt, E)?;
    let p = rebuilt;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(gate.slice())
        .arg(up.slice())
        .arg(dy.slice())
        .arg(dgate.slice_mut())
        .arg(dup.slice_mut())
        .arg(&p.rows)
        .arg(&p.width)
        .arg(&p.gate.ld)
        .arg(&p.gate.off)
        .arg(&p.up.ld)
        .arg(&p.up.off)
        .arg(&p.dy.ld)
        .arg(&p.dy.off)
        .arg(&p.dgate.ld)
        .arg(&p.dgate.off)
        .arg(&p.dup.ld)
        .arg(&p.dup.off);
    // SAFETY: (3 x const float*, 2 x float*, 12 x unsigned long long) in this
    // order. The plan was rebuilt against these exact lengths, so every window
    // index is in bounds; dgate and dup are two distinct `&mut` borrows and
    // alias neither each other nor the inputs.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// The backward into two disjoint windows of one buffer ([`BwdOut::Shared`]).
pub fn swiglu_bwd_shared(
    rt: &CudaRuntime,
    plan: &SwigluBwdPlan,
    (gate, up, dy): (&CudaBuffer<f32>, &CudaBuffer<f32>, &CudaBuffer<f32>),
    dgu: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_swiglu_bwd_shared_f32";
    if plan.out != BwdOut::Shared {
        return Err(CudaError::invalid(E, "the plan writes two buffers"));
    }
    let rebuilt = SwigluBwdPlan::new(
        plan.rows,
        plan.width,
        (plan.gate.ld, plan.gate.off, gate.len()),
        (plan.up.ld, plan.up.off, up.len()),
        (plan.dy.ld, plan.dy.off, dy.len()),
        (plan.dgate.ld, plan.dgate.off, dgu.len()),
        (plan.dup.ld, plan.dup.off, dgu.len()),
        BwdOut::Shared,
    )?;
    let Some(l) = grid_1d(rebuilt.total) else {
        return Ok(());
    };
    let f = k8(rt, E)?;
    let p = rebuilt;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(gate.slice())
        .arg(up.slice())
        .arg(dy.slice())
        .arg(dgu.slice_mut())
        .arg(&p.rows)
        .arg(&p.width)
        .arg(&p.gate.ld)
        .arg(&p.gate.off)
        .arg(&p.up.ld)
        .arg(&p.up.off)
        .arg(&p.dy.ld)
        .arg(&p.dy.off)
        .arg(&p.dgate.ld)
        .arg(&p.dgate.off)
        .arg(&p.dup.ld)
        .arg(&p.dup.off);
    // SAFETY: (3 x const float*, float*, 12 x unsigned long long). The plan,
    // rebuilt against these lengths with BwdOut::Shared, proves the dgate and
    // dup windows disjoint (same stride, separate columns), so each element
    // of `dgu` has one writer; `dgu` is a distinct `&mut` borrow from the
    // inputs.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// `resid += y`, exact f32.
pub fn residual_add(
    rt: &CudaRuntime,
    plan: &ResidualAddPlan,
    y: &CudaBuffer<f32>,
    resid: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_residual_add_f32";
    let p = ResidualAddPlan::new(
        plan.rows,
        plan.width,
        (plan.y.ld, plan.y.off, y.len()),
        (plan.resid.ld, plan.resid.off, resid.len()),
    )?;
    let Some(l) = grid_1d(p.total) else {
        return Ok(());
    };
    let f = k8(rt, E)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(y.slice())
        .arg(resid.slice_mut())
        .arg(&p.rows)
        .arg(&p.width)
        .arg(&p.y.ld)
        .arg(&p.y.off)
        .arg(&p.resid.ld)
        .arg(&p.resid.off);
    // SAFETY: (const float*, float*, 6 x unsigned long long). The plan, built
    // against these lengths, bounds every index; each resid element is read
    // and written by one thread, and `resid` is a distinct `&mut` borrow from
    // `y`.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}

/// Every activation function over `x`, function-major, into `out` (which
/// holds `ACT_SWEEP.len() * x.len()`).
pub fn act_sweep(
    rt: &CudaRuntime,
    x: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const E: &str = "qd_act_sweep_f32";
    let want = x
        .len()
        .checked_mul(ACT_SWEEP.len())
        .ok_or_else(|| CudaError::invalid(E, "sweep length overflows"))?;
    if out.len() != want {
        return Err(CudaError::invalid(
            E,
            format!("out holds {}, the sweep writes {want}", out.len()),
        ));
    }
    let n = u64::try_from(x.len()).unwrap_or(u64::MAX);
    let Some(l) = grid_1d(n) else {
        return Ok(());
    };
    let f = k8(rt, E)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(x.slice()).arg(out.slice_mut()).arg(&n);
    // SAFETY: (const float*, float*, unsigned long long n). x holds n floats
    // and out 7n (checked), and the kernel writes out[f * n + i] for f < 7,
    // i < n only.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(E, e))?;
    Ok(())
}
