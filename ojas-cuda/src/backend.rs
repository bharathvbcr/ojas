#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use ojas_core::DType;
use ojas_core::{
    AdamWConfig, Backend, BackendId, Budget, MuonNs5Config, Numerics, OjasError, PerHeadGateGrad,
    Tensor, ValueResidualGrad,
};
use ojas_device::{Device, DeviceError};

#[cfg(feature = "cuda")]
use crate::buffer::CudaDeviceBuffer;
#[cfg(feature = "cuda")]
use crate::runtime::{driver_error, CudaRuntime};

/// Device-resident [`Backend`] over NVIDIA CUDA.
#[derive(Clone)]
pub struct CudaBackend {
    pub(crate) budget: Budget,
    /// `Rc`, not `Arc`: the runtime owns a raw cuBLAS handle and is neither
    /// `Send` nor `Sync`, so a backend and its clones stay on one thread.
    #[cfg(feature = "cuda")]
    pub(crate) rt: std::rc::Rc<CudaRuntime>,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("id", &BackendId::Cuda)
            .finish()
    }
}

impl CudaBackend {
    /// Open CUDA device 0 with `budget`.
    pub fn open(budget: Budget) -> Result<Self, DeviceError> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = budget;
            Err(DeviceError::NotCompiled { kind: Device::Cuda })
        }
        #[cfg(feature = "cuda")]
        {
            crate::runtime::probe_libraries().map_err(|e| DeviceError::NoDevice {
                kind: Device::Cuda,
                detail: e.to_string(),
            })?;
            let cfg = crate::runtime::RuntimeConfig {
                budget_bytes: budget.cap_bytes(),
                ..crate::runtime::RuntimeConfig::default()
            };
            let rt = CudaRuntime::open(cfg).map_err(|e| DeviceError::NoDevice {
                kind: Device::Cuda,
                detail: e.to_string(),
            })?;
            Ok(Self {
                budget,
                rt: std::rc::Rc::new(rt),
            })
        }
    }

    /// Construct a [`CudaBackend`] around an existing runtime.
    #[cfg(feature = "cuda")]
    pub fn with_runtime(rt: std::rc::Rc<CudaRuntime>, budget: Budget) -> Self {
        Self { budget, rt }
    }

    /// Access the underlying [`CudaRuntime`].
    #[cfg(feature = "cuda")]
    pub fn runtime(&self) -> &std::rc::Rc<CudaRuntime> {
        &self.rt
    }
}

impl Backend for CudaBackend {
    fn id(&self) -> BackendId {
        BackendId::Cuda
    }

    fn budget(&self) -> &Budget {
        &self.budget
    }

    fn numerics(&self) -> Numerics {
        Numerics::Fast
    }

    fn sync(&self) -> Result<(), OjasError> {
        #[cfg(not(feature = "cuda"))]
        {
            Ok(())
        }
        #[cfg(feature = "cuda")]
        {
            self.rt.sync("CudaBackend::sync").map_err(OjasError::from)
        }
    }

    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "Backend::upload";
        if let Some(buf) = tensor.device_buffer() {
            if buf.backend() == BackendId::Cuda {
                #[cfg(feature = "cuda")]
                if let Some(cb) = buf.as_any().downcast_ref::<CudaDeviceBuffer>() {
                    if Arc::ptr_eq(&cb.stream, self.rt.stream()) {
                        return Ok(tensor.clone());
                    }
                }
            }
            return Err(OjasError::Placement {
                op: OP,
                expected: Some(BackendId::Cuda),
                found: Some(buf.backend()),
            });
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(OjasError::Unsupported {
                op: OP,
                detail: "cuda runtime not compiled into this build".to_string(),
            })
        }
        #[cfg(feature = "cuda")]
        {
            let bytes = tensor.to_ne_bytes()?;
            if bytes.is_empty() {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: "cannot upload an empty tensor".to_string(),
                });
            }
            let byte_len = bytes.len();
            let reservation = self.budget.try_reserve(byte_len as u64)?;
            let mut slice = self
                .rt
                .stream()
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| OjasError::from(driver_error("alloc_zeros", e)))?;
            self.rt
                .stream()
                .memcpy_htod(&bytes, &mut slice)
                .map_err(|e| OjasError::from(driver_error("memcpy_htod", e)))?;
            let shadow_u32 = if tensor.dtype() == DType::U32 {
                Some(Arc::<[u32]>::from(tensor.u32_slice()?))
            } else {
                None
            };
            let dev_buf = CudaDeviceBuffer {
                slice,
                len: byte_len,
                stream: Arc::clone(self.rt.stream()),
                shadow_u32,
            };
            Tensor::from_device_reserved(
                Arc::new(dev_buf),
                tensor.shape(),
                tensor.dtype(),
                reservation,
            )
        }
    }

    fn download(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        let host = tensor.to_host(self.budget())?;
        self.sync()?;
        Ok(host)
    }

    fn permute(&self, _input: &Tensor, _dims: &[usize]) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "permute",
            detail: "Cuda backend does not yet implement permute".to_string(),
        })
    }

    fn embedding_forward(&self, _table: &Tensor, _token_ids: &Tensor) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "embedding_forward",
            detail: "Cuda backend does not yet implement embedding_forward".to_string(),
        })
    }

    fn embedding_backward(
        &self,
        _table: &Tensor,
        _token_ids: &Tensor,
        _grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "embedding_backward",
            detail: "Cuda backend does not yet implement embedding_backward".to_string(),
        })
    }

    fn linear_forward(&self, _input: &Tensor, _weight: &Tensor) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "linear_forward",
            detail: "Cuda backend does not yet implement linear_forward".to_string(),
        })
    }

    fn linear_backward(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "linear_backward",
            detail: "Cuda backend does not yet implement linear_backward".to_string(),
        })
    }

    fn rms_norm_forward(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _eps: f32,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "rms_norm_forward",
            detail: "Cuda backend does not yet implement rms_norm_forward".to_string(),
        })
    }

    fn rms_norm_backward(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _grad_output: &Tensor,
        _eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "rms_norm_backward",
            detail: "Cuda backend does not yet implement rms_norm_backward".to_string(),
        })
    }

    fn rope_half_split_forward(
        &self,
        _x: &Tensor,
        _cos: &Tensor,
        _sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "rope_half_split_forward",
            detail: "Cuda backend does not yet implement rope_half_split_forward".to_string(),
        })
    }

    fn rope_half_split_backward(
        &self,
        _grad_output: &Tensor,
        _cos: &Tensor,
        _sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "rope_half_split_backward",
            detail: "Cuda backend does not yet implement rope_half_split_backward".to_string(),
        })
    }

    fn rms_qk_norm_forward(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _q_weight: &Tensor,
        _k_weight: &Tensor,
        _eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "rms_qk_norm_forward",
            detail: "Cuda backend does not yet implement rms_qk_norm_forward".to_string(),
        })
    }

    fn rms_qk_norm_backward(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _q_weight: &Tensor,
        _k_weight: &Tensor,
        _grad_q: &Tensor,
        _grad_k: &Tensor,
        _eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "rms_qk_norm_backward",
            detail: "Cuda backend does not yet implement rms_qk_norm_backward".to_string(),
        })
    }

    fn causal_sdpa_forward(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "causal_sdpa_forward",
            detail: "Cuda backend does not yet implement causal_sdpa_forward".to_string(),
        })
    }

    fn causal_sdpa_backward(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _output: &Tensor,
        _lse: &Tensor,
        _grad_output: &Tensor,
        _window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "causal_sdpa_backward",
            detail: "Cuda backend does not yet implement causal_sdpa_backward".to_string(),
        })
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _bias: &Tensor,
        _attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "per_head_sigmoid_gate_forward",
            detail: "Cuda backend does not yet implement per_head_sigmoid_gate_forward".to_string(),
        })
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _bias: &Tensor,
        _attn_out: &Tensor,
        _grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        Err(OjasError::Unsupported {
            op: "per_head_sigmoid_gate_backward",
            detail: "Cuda backend does not yet implement per_head_sigmoid_gate_backward"
                .to_string(),
        })
    }

    fn value_residual_blend_forward(
        &self,
        _value: &Tensor,
        _value0: &Tensor,
        _lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "value_residual_blend_forward",
            detail: "Cuda backend does not yet implement value_residual_blend_forward".to_string(),
        })
    }

    fn value_residual_blend_backward(
        &self,
        _value: &Tensor,
        _value0: &Tensor,
        _lambda: &Tensor,
        _grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        Err(OjasError::Unsupported {
            op: "value_residual_blend_backward",
            detail: "Cuda backend does not yet implement value_residual_blend_backward".to_string(),
        })
    }

    fn silu_forward(&self, _input: &Tensor) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "silu_forward",
            detail: "Cuda backend does not yet implement silu_forward".to_string(),
        })
    }

    fn silu_backward(&self, _input: &Tensor, _grad_output: &Tensor) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "silu_backward",
            detail: "Cuda backend does not yet implement silu_backward".to_string(),
        })
    }

    fn mul_forward(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "mul_forward",
            detail: "Cuda backend does not yet implement mul_forward".to_string(),
        })
    }

    fn mul_backward(
        &self,
        _a: &Tensor,
        _b: &Tensor,
        _grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "mul_backward",
            detail: "Cuda backend does not yet implement mul_backward".to_string(),
        })
    }

    fn residual_add_forward(&self, _x: &Tensor, _y: &Tensor) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "residual_add_forward",
            detail: "Cuda backend does not yet implement residual_add_forward".to_string(),
        })
    }

    fn residual_add_backward(
        &self,
        _x: &Tensor,
        _y: &Tensor,
        _grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(OjasError::Unsupported {
            op: "residual_add_backward",
            detail: "Cuda backend does not yet implement residual_add_backward".to_string(),
        })
    }

    fn cross_entropy_mean_forward(
        &self,
        _logits: &Tensor,
        _targets: &Tensor,
        _ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "cross_entropy_mean_forward",
            detail: "Cuda backend does not yet implement cross_entropy_mean_forward".to_string(),
        })
    }

    fn cross_entropy_mean_backward(
        &self,
        _logits: &Tensor,
        _targets: &Tensor,
        _ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        Err(OjasError::Unsupported {
            op: "cross_entropy_mean_backward",
            detail: "Cuda backend does not yet implement cross_entropy_mean_backward".to_string(),
        })
    }

    fn clip_grad_norm(&self, _grads: &mut [Tensor], _max_norm: f32) -> Result<f32, OjasError> {
        Err(OjasError::Unsupported {
            op: "clip_grad_norm",
            detail: "Cuda backend does not yet implement clip_grad_norm".to_string(),
        })
    }

    fn adamw_step(
        &self,
        _param: &mut Tensor,
        _grad: &Tensor,
        _moment1: &mut Tensor,
        _moment2: &mut Tensor,
        _step: u64,
        _config: AdamWConfig,
    ) -> Result<(), OjasError> {
        Err(OjasError::Unsupported {
            op: "adamw_step",
            detail: "Cuda backend does not yet implement adamw_step".to_string(),
        })
    }

    fn muon_ns5_step(
        &self,
        _param: &mut Tensor,
        _grad: &Tensor,
        _momentum: &mut Tensor,
        _config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        Err(OjasError::Unsupported {
            op: "muon_ns5_step",
            detail: "Cuda backend does not yet implement muon_ns5_step".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_open_without_cuda_feature() {
        #[cfg(not(feature = "cuda"))]
        {
            let budget = Budget::new(1024 * 1024);
            match CudaBackend::open(budget) {
                Err(DeviceError::NotCompiled { kind: Device::Cuda }) => {}
                Err(other) => panic!("open returned {other}"),
                Ok(_) => panic!("open succeeded without cuda feature"),
            }
        }
    }

    #[test]
    fn cuda_backend_metadata_defaults() {
        #[cfg(not(feature = "cuda"))]
        {
            let budget = Budget::new(1024 * 1024);
            let backend = CudaBackend {
                budget: budget.clone(),
            };
            assert_eq!(backend.id(), BackendId::Cuda);
            assert_eq!(backend.numerics(), Numerics::Fast);
            assert_eq!(backend.budget().cap_bytes(), 1024 * 1024);
            assert!(backend.sync().is_ok());
            assert_eq!(format!("{backend:?}"), "CudaBackend { id: Cuda }");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_open_handles_no_device() {
        let budget = Budget::new(1024 * 1024);
        match std::panic::catch_unwind(|| CudaBackend::open(budget)) {
            Ok(Ok(backend)) => {
                assert_eq!(backend.id(), BackendId::Cuda);
                assert_eq!(backend.numerics(), Numerics::Fast);
            }
            Ok(Err(DeviceError::NoDevice {
                kind: Device::Cuda, ..
            })) => {}
            Ok(Err(other)) => panic!("open returned {other}"),
            Err(_) => panic!("CudaBackend::open panicked instead of returning NoDevice"),
        }
    }
}
