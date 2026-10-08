//! Metal glue for homogeneous SSM mixers (Mamba-2 prep/finish, activations).

use crate::dispatch::{dispatch_1d, dispatch_3d, set_f32, set_tensor, set_u32};
use crate::runtime::GpuRuntime;
use crate::tensor::Tensor;
use std::sync::Arc;

pub fn silu_fwd(rt: &Arc<GpuRuntime>, x: &Tensor, y: &Tensor) -> Result<(), String> {
    let n = x.numel();
    assert_eq!(n, y.numel());
    let p = rt.pipeline("silu_fwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, y, 1);
        set_u32(bnd, n as u32, 2);
    })
}

pub fn silu_bwd(rt: &Arc<GpuRuntime>, x: &Tensor, dy: &Tensor, dx: &Tensor) -> Result<(), String> {
    let n = x.numel();
    let p = rt.pipeline("silu_bwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, dy, 1);
        set_tensor(bnd, dx, 2);
        set_u32(bnd, n as u32, 3);
    })
}

pub fn silu_fwd_store(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    y: &Tensor,
    pre: &Tensor,
) -> Result<(), String> {
    let n = x.numel();
    let p = rt.pipeline("silu_fwd_store_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, y, 1);
        set_tensor(bnd, pre, 2);
        set_u32(bnd, n as u32, 3);
    })
}

pub fn silu_bwd_store(
    rt: &Arc<GpuRuntime>,
    pre: &Tensor,
    dy: &Tensor,
    dx: &Tensor,
) -> Result<(), String> {
    let n = pre.numel();
    let p = rt.pipeline("silu_bwd_store_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, pre, 0);
        set_tensor(bnd, dy, 1);
        set_tensor(bnd, dx, 2);
        set_u32(bnd, n as u32, 3);
    })
}

pub fn softplus_bias_fwd(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    bias: &Tensor,
    out: &Tensor,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), String> {
    let n = b * t * h;
    let p = rt.pipeline("softplus_bias_fwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, bias, 1);
        set_tensor(bnd, out, 2);
        set_u32(bnd, b as u32, 3);
        set_u32(bnd, t as u32, 4);
        set_u32(bnd, h as u32, 5);
    })
}

pub fn softplus_bias_bwd(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    bias: &Tensor,
    dy: &Tensor,
    dx: &Tensor,
    dbias: &Tensor,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), String> {
    dbias.buffer.zero();
    let n = b * t * h;
    let p = rt.pipeline("softplus_bias_bwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, bias, 1);
        set_tensor(bnd, dy, 2);
        set_tensor(bnd, dx, 3);
        set_tensor(bnd, dbias, 4);
        set_u32(bnd, b as u32, 5);
        set_u32(bnd, t as u32, 6);
        set_u32(bnd, h as u32, 7);
    })
}

pub fn mamba2_log_da(
    rt: &Arc<GpuRuntime>,
    dt: &Tensor,
    a_log: &Tensor,
    log_da: &Tensor,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), String> {
    let n = b * t * h;
    let p = rt.pipeline("mamba2_log_da_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, dt, 0);
        set_tensor(bnd, a_log, 1);
        set_tensor(bnd, log_da, 2);
        set_u32(bnd, b as u32, 3);
        set_u32(bnd, t as u32, 4);
        set_u32(bnd, h as u32, 5);
    })
}

pub fn mamba2_x_scaled(
    rt: &Arc<GpuRuntime>,
    xs: &Tensor,
    dt: &Tensor,
    x_scaled: &Tensor,
    b: usize,
    t: usize,
    h: usize,
    p: usize,
) -> Result<(), String> {
    let pipe = rt.pipeline("mamba2_x_scaled_rows_f32")?;
    dispatch_3d(rt, &pipe, p, h, b, |bnd| {
        set_tensor(bnd, xs, 0);
        set_tensor(bnd, dt, 1);
        set_tensor(bnd, x_scaled, 2);
        set_u32(bnd, b as u32, 3);
        set_u32(bnd, t as u32, 4);
        set_u32(bnd, h as u32, 5);
        set_u32(bnd, p as u32, 6);
    })
}

pub fn mamba2_d_skip_fwd(
    rt: &Arc<GpuRuntime>,
    y: &Tensor,
    xs: &Tensor,
    d_param: &Tensor,
    b: usize,
    t: usize,
    h: usize,
    p: usize,
) -> Result<(), String> {
    let pipe = rt.pipeline("mamba2_d_skip_fwd_f32")?;
    dispatch_3d(rt, &pipe, p, h, b, |bnd| {
        set_tensor(bnd, y, 0);
        set_tensor(bnd, xs, 1);
        set_tensor(bnd, d_param, 2);
        set_u32(bnd, b as u32, 3);
        set_u32(bnd, t as u32, 4);
        set_u32(bnd, h as u32, 5);
        set_u32(bnd, p as u32, 6);
    })
}

pub fn mamba2_d_skip_bwd(
    rt: &Arc<GpuRuntime>,
    dy: &Tensor,
    xs: &Tensor,
    d_param: &Tensor,
    dxs: &Tensor,
    dd: &Tensor,
    b: usize,
    t: usize,
    h: usize,
    p: usize,
) -> Result<(), String> {
    dd.buffer.zero();
    let pipe = rt.pipeline("mamba2_d_skip_bwd_f32")?;
    dispatch_3d(rt, &pipe, p, h, b, |bnd| {
        set_tensor(bnd, dy, 0);
        set_tensor(bnd, xs, 1);
        set_tensor(bnd, d_param, 2);
        set_tensor(bnd, dxs, 3);
        set_tensor(bnd, dd, 4);
        set_u32(bnd, b as u32, 5);
        set_u32(bnd, t as u32, 6);
        set_u32(bnd, h as u32, 7);
        set_u32(bnd, p as u32, 8);
    })
}

pub fn rms_norm_weight_fwd(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    weight: &Tensor,
    out: &Tensor,
    rows: usize,
    c: usize,
    eps: f32,
) -> Result<(), String> {
    let p = rt.pipeline("rms_norm_weight_fwd_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, weight, 1);
        set_tensor(bnd, out, 2);
        set_u32(bnd, rows as u32, 3);
        set_u32(bnd, c as u32, 4);
        set_f32(bnd, eps, 5);
    })
}

pub fn rms_norm_weight_bwd(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    weight: &Tensor,
    dy: &Tensor,
    dx: &Tensor,
    dweight: &Tensor,
    rows: usize,
    c: usize,
    eps: f32,
) -> Result<(), String> {
    dweight.buffer.zero();
    let p = rt.pipeline("rms_norm_weight_bwd_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, weight, 1);
        set_tensor(bnd, dy, 2);
        set_tensor(bnd, dx, 3);
        set_tensor(bnd, dweight, 4);
        set_u32(bnd, rows as u32, 5);
        set_u32(bnd, c as u32, 6);
        set_f32(bnd, eps, 7);
    })
}

pub fn mul_fwd(rt: &Arc<GpuRuntime>, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<(), String> {
    let n = a.numel();
    let p = rt.pipeline("mul_fwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, a, 0);
        set_tensor(bnd, b, 1);
        set_tensor(bnd, out, 2);
        set_u32(bnd, n as u32, 3);
    })
}

pub fn mul_bwd(
    rt: &Arc<GpuRuntime>,
    a: &Tensor,
    b: &Tensor,
    dy: &Tensor,
    da: &Tensor,
    db: &Tensor,
) -> Result<(), String> {
    let n = a.numel();
    let p = rt.pipeline("mul_bwd_a_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, a, 0);
        set_tensor(bnd, b, 1);
        set_tensor(bnd, dy, 2);
        set_tensor(bnd, da, 3);
        set_tensor(bnd, db, 4);
        set_u32(bnd, n as u32, 5);
    })
}

/// View `[B,T,C]` tensor column slice `[start..start+len)` along last dim.
pub fn slice_last_dim(t: &Tensor, start: usize, len: usize) -> Tensor {
    assert!(t.shape().len() >= 2);
    let c = *t.shape().last().unwrap();
    assert!(start + len <= c);
    let byte_offset = t.byte_offset() + start * 4;
    let mut shape = t.shape().to_vec();
    *shape.last_mut().unwrap() = len;
    Tensor::from_buffer(t.runtime(), t.buffer.clone(), &shape, t.dtype, byte_offset)
        .expect("these views are built over a buffer this crate just allocated")
}

pub fn accum_slice_grad(
    rt: &Arc<GpuRuntime>,
    dst: &Tensor,
    src: &Tensor,
    rows: usize,
    total_out: usize,
    len: usize,
    start: usize,
) -> Result<(), String> {
    let n = rows * len;
    let p = rt.pipeline("accum_slice_grad_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, dst, 0);
        set_tensor(bnd, src, 1);
        set_u32(bnd, rows as u32, 2);
        set_u32(bnd, total_out as u32, 3);
        set_u32(bnd, len as u32, 4);
        set_u32(bnd, start as u32, 5);
    })
}

pub fn mamba2_x_scaled_bwd(
    rt: &Arc<GpuRuntime>,
    grad_x_scaled: &Tensor,
    xs: &Tensor,
    dt: &Tensor,
    grad_xs: &Tensor,
    grad_dt: &Tensor,
    b: usize,
    t: usize,
    h: usize,
    p: usize,
) -> Result<(), String> {
    grad_dt.buffer.zero();
    let pipe = rt.pipeline("mamba2_x_scaled_bwd_f32")?;
    dispatch_3d(rt, &pipe, p, h, b, |bnd| {
        set_tensor(bnd, grad_x_scaled, 0);
        set_tensor(bnd, xs, 1);
        set_tensor(bnd, dt, 2);
        set_tensor(bnd, grad_xs, 3);
        set_tensor(bnd, grad_dt, 4);
        set_u32(bnd, b as u32, 5);
        set_u32(bnd, t as u32, 6);
        set_u32(bnd, h as u32, 7);
        set_u32(bnd, p as u32, 8);
    })
}

pub fn mamba2_log_da_bwd(
    rt: &Arc<GpuRuntime>,
    grad_log_da: &Tensor,
    dt: &Tensor,
    a_log: &Tensor,
    grad_dt: &Tensor,
    grad_a_log: &Tensor,
    b: usize,
    t: usize,
    h: usize,
) -> Result<(), String> {
    grad_a_log.buffer.zero();
    let n = b * t * h;
    let p = rt.pipeline("mamba2_log_da_bwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, grad_log_da, 0);
        set_tensor(bnd, dt, 1);
        set_tensor(bnd, a_log, 2);
        set_tensor(bnd, grad_dt, 3);
        set_tensor(bnd, grad_a_log, 4);
        set_u32(bnd, b as u32, 5);
        set_u32(bnd, t as u32, 6);
        set_u32(bnd, h as u32, 7);
    })
}

/// Reshape last dim split for `[B,T,C]` → `[B,T,H,P]` when C = H*P.
pub fn reshape_heads(t: &Tensor, h: usize, p: usize) -> Tensor {
    assert_eq!(*t.shape().last().unwrap(), h * p);
    let mut shape = t.shape().to_vec();
    shape.pop();
    shape.push(h);
    shape.push(p);
    Tensor::from_buffer(
        t.runtime(),
        t.buffer.clone(),
        &shape,
        t.dtype,
        t.byte_offset(),
    )
    .expect("these views are built over a buffer this crate just allocated")
}

pub fn flatten_heads(t: &Tensor, d_inner: usize) -> Tensor {
    let mut shape = t.shape().to_vec();
    shape.pop();
    shape.pop();
    shape.push(d_inner);
    Tensor::from_buffer(
        t.runtime(),
        t.buffer.clone(),
        &shape,
        t.dtype,
        t.byte_offset(),
    )
    .expect("these views are built over a buffer this crate just allocated")
}

pub fn unflatten_heads(t: &Tensor, h: usize, p: usize) -> Tensor {
    reshape_heads(t, h, p)
}
