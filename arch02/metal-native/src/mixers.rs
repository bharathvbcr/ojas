use crate::dispatch::{dispatch_1d, dispatch_2d, dispatch_3d, set_tensor, set_u32};
use crate::runtime::GpuRuntime;
use crate::tensor::{DType, Tensor};
use std::sync::Arc;

fn validate_mingru_fwd_inputs(
    z_shape: &[usize],
    z_dtype: DType,
    h_shape: &[usize],
    h_dtype: DType,
) -> Result<(usize, usize, usize), String> {
    // `mingru_fwd` binds every operand as `device float*`. Reject two-byte
    // storage before allocation, pipeline lookup, or dispatch rather than
    // reinterpreting BF16/F16 buffers as f32 and reading out of bounds.
    if z_dtype != DType::F32 || h_dtype != DType::F32 {
        return Err(format!(
            "mingru_fwd requires f32 inputs, got z={z_dtype:?}, h={h_dtype:?}"
        ));
    }
    let [b, t, h] = z_shape else {
        return Err(format!(
            "mingru_fwd requires rank-3 z input, got shape {z_shape:?}"
        ));
    };
    if h_shape != z_shape {
        return Err(format!(
            "mingru_fwd requires equal input shapes, got z={z_shape:?}, h={h_shape:?}"
        ));
    }
    Ok((*b, *t, *h))
}

pub fn mingru_fwd(rt: &Arc<GpuRuntime>, z_in: &Tensor, h_in: &Tensor) -> Result<Tensor, String> {
    let (b, t, h) = validate_mingru_fwd_inputs(&z_in.shape(), z_in.dtype, &h_in.shape(), h_in.dtype)?;
    let out = rt.alloc_tensor_f32(&[b, t, h])?;

    let pipe = rt.pipeline("mingru_fwd")?;
    dispatch_2d(rt, &pipe, h, b, |bnd| {
        set_tensor(bnd, z_in, 0);
        set_tensor(bnd, h_in, 1);
        set_tensor(bnd, &out, 2);
        set_u32(bnd, b as u32, 3);
        set_u32(bnd, t as u32, 4);
        set_u32(bnd, h as u32, 5);
    })?;

    Ok(out)
}

#[cfg(test)]
mod mingru_fwd_contract_tests {
    use super::*;

    #[test]
    fn rejects_every_non_f32_input_before_gpu_work() {
        assert_eq!(
            validate_mingru_fwd_inputs(&[2, 3, 4], DType::F32, &[2, 3, 4], DType::F32),
            Ok((2, 3, 4))
        );
        for (z, h) in [
            (DType::BF16, DType::F32),
            (DType::F16, DType::F32),
            (DType::F32, DType::BF16),
            (DType::F32, DType::F16),
            (DType::BF16, DType::BF16),
            (DType::F16, DType::F16),
        ] {
            let error = validate_mingru_fwd_inputs(&[2, 3, 4], z, &[2, 3, 4], h)
                .expect_err("two-byte inputs must be rejected");
            assert!(error.contains("requires f32 inputs"), "{error}");
        }
    }
}

pub fn mingru_bwd(
    rt: &Arc<GpuRuntime>,
    z_in: &Tensor,
    h_in: &Tensor,
    h_out: &Tensor,
    grad_h_out: &Tensor,
) -> Result<(Tensor, Tensor), String> {
    let b = z_in.shape()[0];
    let t = z_in.shape()[1];
    let h = z_in.shape()[2];

    let grad_z = rt.alloc_tensor_f32(&[b, t, h])?;
    let grad_h_in = rt.alloc_tensor_f32(&[b, t, h])?;

    let pipe = rt.pipeline("mingru_bwd")?;
    dispatch_2d(rt, &pipe, h, b, |bnd| {
        set_tensor(bnd, z_in, 0);
        set_tensor(bnd, h_in, 1);
        set_tensor(bnd, h_out, 2);
        set_tensor(bnd, grad_h_out, 3);
        set_tensor(bnd, &grad_z, 4);
        set_tensor(bnd, &grad_h_in, 5);
        set_u32(bnd, b as u32, 6);
        set_u32(bnd, t as u32, 7);
        set_u32(bnd, h as u32, 8);
    })?;

    Ok((grad_z, grad_h_in))
}

pub fn mingru_vr_blend_fwd(
    rt: &Arc<GpuRuntime>,
    h_raw: &Tensor,
    v0_up: &Tensor,
    vr_lambda: &Tensor,
    h_pre: &Tensor,
    use_v0: bool,
) -> Result<(), String> {
    let n = h_raw.numel();
    let p = rt.pipeline("mingru_vr_blend_fwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, h_raw, 0);
        set_tensor(bnd, v0_up, 1);
        set_tensor(bnd, vr_lambda, 2);
        set_tensor(bnd, h_pre, 3);
        set_u32(bnd, n as u32, 4);
        set_u32(bnd, use_v0 as u32, 5);
    })
}

pub fn mingru_vr_blend_bwd(
    rt: &Arc<GpuRuntime>,
    d_h_pre: &Tensor,
    h_raw: &Tensor,
    v0_up: &Tensor,
    vr_lambda: &Tensor,
    d_h_raw: &Tensor,
    d_v0_up: &Tensor,
    d_vr_lambda: &Tensor,
    use_v0: bool,
) -> Result<(), String> {
    let n = d_h_pre.numel();
    let p = rt.pipeline("mingru_vr_blend_bwd_f32")?;
    dispatch_1d(rt, &p, n, |bnd| {
        set_tensor(bnd, d_h_pre, 0);
        set_tensor(bnd, h_raw, 1);
        set_tensor(bnd, v0_up, 2);
        set_tensor(bnd, vr_lambda, 3);
        set_tensor(bnd, d_h_raw, 4);
        set_tensor(bnd, d_v0_up, 5);
        set_tensor(bnd, d_vr_lambda, 6);
        set_u32(bnd, n as u32, 7);
        set_u32(bnd, use_v0 as u32, 8);
    })
}

pub struct Mamba2FwdOutputs {
    pub y: Tensor,
    pub h_states: Tensor,
}

pub fn mamba2_fwd(
    rt: &Arc<GpuRuntime>,
    x_scaled: &Tensor,
    b_h: &Tensor,
    c_h: &Tensor,
    log_da: &Tensor,
) -> Result<Mamba2FwdOutputs, String> {
    // x_scaled: (B, T, H, P)
    // b_h: (B, T, N)
    // c_h: (B, T, N)
    // log_da: (B, T, H)
    let b_sz = x_scaled.shape()[0];
    let t_sz = x_scaled.shape()[1];
    let h_sz = x_scaled.shape()[2];
    let p_sz = x_scaled.shape()[3];
    let n_sz = b_h.shape()[2];

    let y = rt.alloc_tensor_f32(&[b_sz, t_sz, h_sz, p_sz])?;
    let h_states = rt.alloc_tensor_f32(&[b_sz, t_sz, h_sz, p_sz, n_sz])?;

    let pipe = rt.pipeline("mamba2_fwd")?;
    dispatch_3d(rt, &pipe, p_sz, h_sz, b_sz, |bnd| {
        set_tensor(bnd, x_scaled, 0);
        set_tensor(bnd, b_h, 1);
        set_tensor(bnd, c_h, 2);
        set_tensor(bnd, log_da, 3);
        set_tensor(bnd, &y, 4);
        set_tensor(bnd, &h_states, 5);
        set_u32(bnd, b_sz as u32, 6);
        set_u32(bnd, t_sz as u32, 7);
        set_u32(bnd, h_sz as u32, 8);
        set_u32(bnd, p_sz as u32, 9);
        set_u32(bnd, n_sz as u32, 10);
    })?;

    Ok(Mamba2FwdOutputs { y, h_states })
}

pub struct Mamba2BwdOutputs {
    pub grad_x_scaled: Tensor,
    pub grad_b_h: Tensor,
    pub grad_c_h: Tensor,
    pub grad_log_da: Tensor,
}

pub fn mamba2_bwd(
    rt: &Arc<GpuRuntime>,
    x_scaled: &Tensor,
    b_h: &Tensor,
    c_h: &Tensor,
    log_da: &Tensor,
    h_states: &Tensor,
    grad_y: &Tensor,
) -> Result<Mamba2BwdOutputs, String> {
    let b_sz = x_scaled.shape()[0];
    let t_sz = x_scaled.shape()[1];
    let h_sz = x_scaled.shape()[2];
    let p_sz = x_scaled.shape()[3];
    let n_sz = b_h.shape()[2];

    let grad_x_scaled = rt.alloc_tensor_f32(&[b_sz, t_sz, h_sz, p_sz])?;
    let grad_b_h = rt.alloc_tensor_f32(&[b_sz, t_sz, n_sz])?;
    let grad_c_h = rt.alloc_tensor_f32(&[b_sz, t_sz, n_sz])?;
    let grad_log_da = rt.alloc_tensor_f32(&[b_sz, t_sz, h_sz])?;

    // Zero-initialize atomic accumulated gradients
    grad_b_h.buffer.zero();
    grad_c_h.buffer.zero();
    grad_log_da.buffer.zero();

    let pipe = rt.pipeline("mamba2_bwd")?;
    dispatch_3d(rt, &pipe, p_sz, h_sz, b_sz, |bnd| {
        set_tensor(bnd, x_scaled, 0);
        set_tensor(bnd, b_h, 1);
        set_tensor(bnd, c_h, 2);
        set_tensor(bnd, log_da, 3);
        set_tensor(bnd, h_states, 4);
        set_tensor(bnd, grad_y, 5);
        set_tensor(bnd, &grad_x_scaled, 6);
        set_tensor(bnd, &grad_b_h, 7);
        set_tensor(bnd, &grad_c_h, 8);
        set_tensor(bnd, &grad_log_da, 9);
        set_u32(bnd, b_sz as u32, 10);
        set_u32(bnd, t_sz as u32, 11);
        set_u32(bnd, h_sz as u32, 12);
        set_u32(bnd, p_sz as u32, 13);
        set_u32(bnd, n_sz as u32, 14);
    })?;

    Ok(Mamba2BwdOutputs {
        grad_x_scaled,
        grad_b_h,
        grad_c_h,
        grad_log_da,
    })
}

pub fn mamba2_conv1d_fwd(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
) -> Result<Tensor, String> {
    let b = x.shape()[0];
    let t = x.shape()[1];
    let c = x.shape()[2];
    let k = w.shape()[1];
    let out = rt.alloc_tensor_f32(&[b, t, c])?;
    let pipe = rt.pipeline("mamba2_conv1d_fwd")?;
    dispatch_2d(rt, &pipe, c, b, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, w, 1);
        set_tensor(bnd, bias, 2);
        set_tensor(bnd, &out, 3);
        set_u32(bnd, b as u32, 4);
        set_u32(bnd, t as u32, 5);
        set_u32(bnd, c as u32, 6);
        set_u32(bnd, k as u32, 7);
    })?;
    Ok(out)
}

pub struct Mamba2Conv1dBwdOutputs {
    pub grad_x: Tensor,
    pub grad_w: Tensor,
    pub grad_bias: Tensor,
}

pub fn mamba2_conv1d_bwd(
    rt: &Arc<GpuRuntime>,
    x: &Tensor,
    w: &Tensor,
    grad_y: &Tensor,
) -> Result<Mamba2Conv1dBwdOutputs, String> {
    let b = x.shape()[0];
    let t = x.shape()[1];
    let c = x.shape()[2];
    let k = w.shape()[1];
    let grad_x = rt.alloc_tensor_f32(&[b, t, c])?;
    let grad_w = rt.alloc_tensor_f32(&w.shape())?;
    let grad_bias = rt.alloc_tensor_f32(&[c])?;
    grad_w.buffer.zero();
    grad_bias.buffer.zero();
    let pipe = rt.pipeline("mamba2_conv1d_bwd")?;
    dispatch_2d(rt, &pipe, c, b, |bnd| {
        set_tensor(bnd, x, 0);
        set_tensor(bnd, w, 1);
        set_tensor(bnd, grad_y, 2);
        set_tensor(bnd, &grad_x, 3);
        set_tensor(bnd, &grad_w, 4);
        set_tensor(bnd, &grad_bias, 5);
        set_u32(bnd, b as u32, 6);
        set_u32(bnd, t as u32, 7);
        set_u32(bnd, c as u32, 8);
        set_u32(bnd, k as u32, 9);
    })?;
    Ok(Mamba2Conv1dBwdOutputs {
        grad_x,
        grad_w,
        grad_bias,
    })
}
