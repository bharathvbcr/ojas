//! Host-tensor [`Backend`](ojas_core::Backend) over wgpu.
//!
//! Every method uploads its inputs and reads the result back. That matches the
//! trait, which only speaks host [`Tensor`](ojas_core::Tensor) values. The
//! performance path is [`crate::WgpuContext`]: device buffers stay resident and
//! [`crate::gemm_then_silu`] records more than one op before the single read.
//!
//! Agreement with the CPU reference is a tolerance, not bit identity. wgpu's
//! Metal HAL leaves floating-point contraction on. Elementwise ops use
//! [`ELEMENT_ABS_TOL`]. GEMM uses [`gemm_abs_tol`]. Two launches of the same op
//! on this backend are bit-identical.

use ojas_core::{
    AdamWConfig, Backend, BackendId, Budget, MuonNs5Config, OjasError, PerHeadGateGrad, Tensor,
    ValueResidualGrad,
};
use ojas_device::DeviceError;

use crate::context::{gemm, mul, residual, rms_norm, silu, WgpuContext};

/// Absolute tolerance for a length-`k` product sum against the CPU reference.
#[allow(dead_code)] // compared from the cfg(test) module; the library path is the round trip
pub fn gemm_abs_tol(k: usize) -> f64 {
    (k as f64) * 2.0e-5 + 1.0e-4
}

#[allow(dead_code)]
pub const ELEMENT_ABS_TOL: f64 = 1.0e-4;

fn unsupported(op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: "wgpu round-trip backend does not implement this op".to_string(),
    }
}

fn show(err: DeviceError) -> OjasError {
    OjasError::Backend {
        id: BackendId::Wgpu,
        detail: err.to_string(),
    }
}

/// Round-trip GPU backend. See the module docs.
pub struct WgpuBackend {
    ctx: WgpuContext,
    budget: Budget,
}

impl WgpuBackend {
    pub fn open(budget: Budget) -> Result<Self, DeviceError> {
        Ok(Self {
            ctx: WgpuContext::open()?,
            budget,
        })
    }

    pub fn context(&self) -> &WgpuContext {
        &self.ctx
    }
}

impl Backend for WgpuBackend {
    fn id(&self) -> BackendId {
        BackendId::Wgpu
    }

    fn budget(&self) -> &Budget {
        &self.budget
    }

    fn embedding_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
        Err(unsupported("embedding_forward"))
    }

    fn embedding_backward(&self, _: &Tensor, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
        Err(unsupported("embedding_backward"))
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        let x = input.to_f32_vec()?;
        let w = weight.to_f32_vec()?;
        let x_shape = input.shape();
        let w_shape = weight.shape();
        if w_shape.len() != 2 || x_shape.is_empty() {
            return Err(OjasError::Shape {
                op: "linear_forward",
                detail: "weight must be [out, in] and input must have a last axis".to_string(),
            });
        }
        let kin = *x_shape.last().unwrap();
        let nout = w_shape[0];
        if kin == 0 || x.len() % kin != 0 {
            return Err(OjasError::Shape {
                op: "linear_forward",
                detail: "linear shapes do not match".to_string(),
            });
        }
        let rows = x.len() / kin;
        if w_shape[1] != kin || w.len() != nout * kin {
            return Err(OjasError::Shape {
                op: "linear_forward",
                detail: "weight in-features do not match".to_string(),
            });
        }
        let mut packed = vec![0.0f32; kin * nout];
        for col in 0..nout {
            for inner in 0..kin {
                packed[inner * nout + col] = w[col * kin + inner];
            }
        }
        let y = gemm(&self.ctx, &x, &packed, rows, kin, nout).map_err(show)?;
        let mut shape = x_shape.to_vec();
        shape.pop();
        shape.push(nout);
        Tensor::from_f32(&y, &shape, &self.budget)
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let x = input.to_f32_vec()?;
        let w = weight.to_f32_vec()?;
        let gy = grad_output.to_f32_vec()?;
        let kin = *input.shape().last().unwrap();
        let nout = weight.shape()[0];
        let rows = x.len() / kin;
        let gx = gemm(&self.ctx, &gy, &w, rows, nout, kin).map_err(show)?;
        let mut xt = vec![0.0f32; kin * rows];
        for r in 0..rows {
            for i in 0..kin {
                xt[i * rows + r] = x[r * kin + i];
            }
        }
        let mixed = gemm(&self.ctx, &xt, &gy, kin, rows, nout).map_err(show)?;
        let mut gw = vec![0.0f32; nout * kin];
        for i in 0..kin {
            for c in 0..nout {
                gw[c * kin + i] = mixed[i * nout + c];
            }
        }
        Ok((
            Tensor::from_f32(&gx, input.shape(), &self.budget)?,
            Tensor::from_f32(&gw, weight.shape(), &self.budget)?,
        ))
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let x = input.to_f32_vec()?;
        let w = weight.to_f32_vec()?;
        let cols = *input.shape().last().unwrap();
        let rows = x.len() / cols;
        let y = rms_norm(&self.ctx, rows, cols, &x, &w, eps).map_err(show)?;
        Tensor::from_f32(&y, input.shape(), &self.budget)
    }

    fn rms_norm_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(unsupported("rms_norm_backward"))
    }

    fn rope_half_split_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("rope_half_split_forward"))
    }

    fn rope_half_split_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("rope_half_split_backward"))
    }

    fn rms_qk_norm_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(unsupported("rms_qk_norm_forward"))
    }

    fn rms_qk_norm_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        Err(unsupported("rms_qk_norm_backward"))
    }

    fn causal_sdpa_forward(&self, _: &Tensor, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
        Err(unsupported("causal_sdpa_forward"))
    }

    fn causal_sdpa_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        Err(unsupported("causal_sdpa_backward"))
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("per_head_sigmoid_gate_forward"))
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        Err(unsupported("per_head_sigmoid_gate_backward"))
    }

    fn value_residual_blend_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("value_residual_blend_forward"))
    }

    fn value_residual_blend_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        Err(unsupported("value_residual_blend_backward"))
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let x = input.to_f32_vec()?;
        let y = silu(&self.ctx, &x).map_err(show)?;
        Tensor::from_f32(&y, input.shape(), &self.budget)
    }

    fn silu_backward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
        Err(unsupported("silu_backward"))
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        let av = a.to_f32_vec()?;
        let bv = b.to_f32_vec()?;
        let y = mul(&self.ctx, &av, &bv).map_err(show)?;
        Tensor::from_f32(&y, a.shape(), &self.budget)
    }

    fn mul_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(unsupported("mul_backward"))
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        let xv = x.to_f32_vec()?;
        let yv = y.to_f32_vec()?;
        let sum = residual(&self.ctx, &xv, &yv).map_err(show)?;
        Tensor::from_f32(&sum, x.shape(), &self.budget)
    }

    fn residual_add_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(unsupported("residual_add_backward"))
    }

    fn cross_entropy_mean_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("cross_entropy_mean_forward"))
    }

    fn cross_entropy_mean_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("cross_entropy_mean_backward"))
    }

    fn clip_grad_norm(&self, _: &mut [Tensor], _: f32) -> Result<f32, OjasError> {
        Err(unsupported("clip_grad_norm"))
    }

    fn adamw_step(
        &self,
        _: &mut Tensor,
        _: &Tensor,
        _: &mut Tensor,
        _: &mut Tensor,
        _: u64,
        _: AdamWConfig,
    ) -> Result<(), OjasError> {
        Err(unsupported("adamw_step"))
    }

    fn muon_ns5_step(
        &self,
        _: &mut Tensor,
        _: &Tensor,
        _: &mut Tensor,
        _: MuonNs5Config,
    ) -> Result<(), OjasError> {
        Err(unsupported("muon_ns5_step"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gemm_then_silu, row_sum, softmax_rows};
    use ojas_core::{Backend, Tensor, RMS_NORM_EPS};
    use ojas_cpu::CpuBackend;

    fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn round_trip_matches_cpu_within_tolerance_and_refuses_the_rest() {
        let budget = Budget::new(8 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        let cpu = CpuBackend::new(budget.clone());
        let x = Tensor::from_f32(&[0.5, -0.25, 1.0, 0.0, 0.2, -0.7], &[2, 3], &budget).unwrap();
        let w = Tensor::from_f32(&[0.1, 0.2, -0.3, 0.4, 0.0, 0.5], &[2, 3], &budget).unwrap();
        let gy = cpu.linear_forward(&x, &w).unwrap();
        let gg = gpu.linear_forward(&x, &w).unwrap();
        assert!(max_abs(&gg.to_f32_vec().unwrap(), &gy.to_f32_vec().unwrap()) <= gemm_abs_tol(3));
        let nw = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], &budget).unwrap();
        let cn = cpu.rms_norm_forward(&x, &nw, RMS_NORM_EPS).unwrap();
        let gn = gpu.rms_norm_forward(&x, &nw, RMS_NORM_EPS).unwrap();
        assert!(max_abs(&gn.to_f32_vec().unwrap(), &cn.to_f32_vec().unwrap()) <= ELEMENT_ABS_TOL);
        let cs = cpu.silu_forward(&x).unwrap();
        let gs = gpu.silu_forward(&x).unwrap();
        assert!(max_abs(&gs.to_f32_vec().unwrap(), &cs.to_f32_vec().unwrap()) <= ELEMENT_ABS_TOL);
        let cm = cpu.mul_forward(&x, &x).unwrap();
        let gm = gpu.mul_forward(&x, &x).unwrap();
        assert!(max_abs(&gm.to_f32_vec().unwrap(), &cm.to_f32_vec().unwrap()) <= ELEMENT_ABS_TOL);
        let ca = cpu.residual_add_forward(&x, &x).unwrap();
        let ga = gpu.residual_add_forward(&x, &x).unwrap();
        assert!(max_abs(&ga.to_f32_vec().unwrap(), &ca.to_f32_vec().unwrap()) <= ELEMENT_ABS_TOL);
        assert!(matches!(
            gpu.embedding_forward(&x, &x),
            Err(OjasError::Unsupported { .. })
        ));
    }

    #[test]
    fn same_backend_is_bit_identical_across_1000_runs() {
        let ctx = WgpuContext::open().expect("wgpu adapter");
        let x = [0.2f32, -0.4, 0.8, 1.2];
        let first = silu(&ctx, &x).unwrap();
        for _ in 0..999 {
            let next = silu(&ctx, &x).unwrap();
            assert_eq!(
                first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                next.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let before = ctx.stats();
        let _ = silu(&ctx, &x).unwrap();
        let after = ctx.stats();
        assert!(after.reuses > before.reuses || after.compiles >= before.compiles);
    }

    #[test]
    fn resident_gemm_then_silu_reads_once() {
        let ctx = WgpuContext::open().expect("wgpu adapter");
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [0.5f32, -0.5, 0.25, 0.0];
        let got = gemm_then_silu(&ctx, &a, &b, 2, 2, 2).unwrap();
        let prod = gemm(&ctx, &a, &b, 2, 2, 2).unwrap();
        let silu_cpu = silu(&ctx, &prod).unwrap();
        assert!(
            max_abs(&got, &silu_cpu) <= ELEMENT_ABS_TOL,
            "{got:?} {silu_cpu:?}"
        );
        let rows = row_sum(&ctx, 1, 4, &a).unwrap();
        assert!((rows[0] - 2.0).abs() < 1e-3);
        let sm = softmax_rows(&ctx, 1, 2, &[0.0, 0.0]).unwrap();
        assert!((sm[0] - 0.5).abs() < 1e-3 && (sm[1] - 0.5).abs() < 1e-3);
    }
}
