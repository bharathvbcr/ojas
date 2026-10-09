#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use ojas_core::DType;
use ojas_core::{
    AdamWConfig, Backend, BackendId, Budget, MuonNs5Config, Numerics, OjasError, PerHeadGateGrad,
    Tensor, ValueResidualGrad,
};
use ojas_device::DeviceError;

#[cfg(feature = "cuda")]
use crate::buffer::CudaDeviceBuffer;
#[cfg(feature = "cuda")]
use crate::runtime::{CudaRuntime, RuntimeConfig};

/// Device-resident [`Backend`] over NVIDIA CUDA.
///
/// Its [`Backend::budget`] is its runtime's ([`CudaRuntime::budget`]): the
/// tensors it uploads, the runtime's kernel buffers and the cuBLAS workspace
/// are charged to one [`Budget`], so together they never pass its cap.
#[derive(Clone)]
pub struct CudaBackend {
    /// Without `cuda` there is no runtime to hold the budget.
    #[cfg(not(feature = "cuda"))]
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
    /// Open CUDA device 0, charging `budget` for every device allocation,
    /// the runtime's 32 MiB cuBLAS workspace included
    /// ([`CudaRuntime::open_with`]).
    ///
    /// A failure keeps its kind (`impl From<CudaError> for DeviceError`):
    /// missing libraries or no device are [`DeviceError::NoDevice`], a
    /// compute capability other than [`crate::kernels::REQUIRED_CC`] is
    /// [`DeviceError::Unsupported`], a budget too small for the workspace or
    /// a device out of memory is [`DeviceError::Capacity`], an NVRTC failure
    /// is [`DeviceError::Compile`], and a cuBLAS, stream or first-sync
    /// failure is [`DeviceError::Init`].
    pub fn open(budget: Budget) -> Result<Self, DeviceError> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = budget;
            Err(DeviceError::NotCompiled {
                kind: ojas_device::Device::Cuda,
            })
        }
        #[cfg(feature = "cuda")]
        {
            let rt = CudaRuntime::open_with(RuntimeConfig::default(), budget)?;
            Ok(Self {
                rt: std::rc::Rc::new(rt),
            })
        }
    }

    /// A [`CudaBackend`] on an existing runtime, charging the runtime's
    /// budget.
    #[cfg(feature = "cuda")]
    pub fn with_runtime(rt: std::rc::Rc<CudaRuntime>) -> Self {
        Self { rt }
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
        #[cfg(not(feature = "cuda"))]
        {
            &self.budget
        }
        #[cfg(feature = "cuda")]
        {
            self.rt.budget().budget()
        }
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
            // The host window is the copy's source: no staging copy, except
            // for F16, which has no borrowed accessor.
            let staged;
            let host: &[u8] = match tensor.dtype() {
                DType::F32 => ne_bytes(tensor.f32_slice()?),
                DType::U32 => ne_bytes(tensor.u32_slice()?),
                DType::Bf16 => ne_bytes(tensor.bf16_slice()?),
                DType::F16 => {
                    staged = tensor.to_ne_bytes()?;
                    &staged
                }
            };
            if host.is_empty() {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: "cannot upload an empty tensor".to_string(),
                });
            }
            let byte_len = host.len();
            // The runtime's budget, charged in `ojas_core` terms so a refusal
            // stays `CapacityExceeded`; then the device's free memory.
            let reservation = self.budget().try_reserve(byte_len as u64)?;
            self.rt.check_free(byte_len as u64, OP)?;
            let slice = self.rt.copy_in(host, OP)?;
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
                sync_timeout: self.rt.config().sync_timeout,
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

/// Element types whose every byte is initialised value bits (no padding), so
/// a slice of them may be read as bytes.
#[cfg(feature = "cuda")]
trait PlainBits: Copy {}
#[cfg(feature = "cuda")]
impl PlainBits for f32 {}
#[cfg(feature = "cuda")]
impl PlainBits for u32 {}
#[cfg(feature = "cuda")]
impl PlainBits for u16 {}

/// `data`'s bytes in native order, borrowed.
#[cfg(feature = "cuda")]
fn ne_bytes<T: PlainBits>(data: &[T]) -> &[u8] {
    // SAFETY: `T` is f32, u32 or u16 (`PlainBits`), so every byte of `data`
    // is initialised and none is padding; the result covers exactly
    // `size_of_val(data)` bytes of the same allocation, u8 needs alignment 1,
    // and the returned borrow keeps `data` borrowed (so alive and unmutated)
    // for as long as the bytes are.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_device::Device;

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
        let budget = Budget::new(1 << 30);
        match std::panic::catch_unwind(|| CudaBackend::open(budget.clone())) {
            Ok(Ok(backend)) => {
                assert_eq!(backend.id(), BackendId::Cuda);
                assert_eq!(backend.numerics(), Numerics::Fast);
                // The backend charges the caller's budget, workspace included.
                assert_eq!(
                    budget.live_bytes().unwrap(),
                    crate::runtime::CUBLAS_WORKSPACE_BYTES as u64
                );
            }
            Ok(Err(DeviceError::NoDevice {
                kind: Device::Cuda, ..
            })) => {}
            Ok(Err(other)) => panic!("open returned {other}"),
            Err(_) => panic!("CudaBackend::open panicked instead of returning NoDevice"),
        }
    }

    /// A budget smaller than the cuBLAS workspace cannot open a backend: on a
    /// device that is `Capacity` (`E_CAPACITY` through the C ABI), not
    /// `NoDevice`; without one the open stops at `NoDevice` first.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_budget_below_the_workspace_is_capacity_on_a_device() {
        let budget = Budget::new(1 << 20);
        match std::panic::catch_unwind(|| CudaBackend::open(budget.clone())) {
            Ok(Err(DeviceError::Capacity {
                kind: Device::Cuda, ..
            }))
            | Ok(Err(DeviceError::NoDevice {
                kind: Device::Cuda, ..
            })) => assert_eq!(budget.live_bytes().unwrap(), 0),
            Ok(Ok(_)) => panic!("a 1 MiB budget held the 32 MiB cuBLAS workspace"),
            Ok(Err(other)) => panic!("open returned {other}"),
            Err(_) => panic!("CudaBackend::open panicked"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn ne_bytes_is_the_native_encoding() {
        let x = [1.5f32, -2.0];
        assert_eq!(
            ne_bytes(&x),
            [x[0].to_ne_bytes(), x[1].to_ne_bytes()].concat()
        );
        let ids = [7u32, u32::MAX];
        assert_eq!(
            ne_bytes(&ids),
            [ids[0].to_ne_bytes(), ids[1].to_ne_bytes()].concat()
        );
        assert!(ne_bytes::<u16>(&[]).is_empty());
    }

    /// The borrowed view is byte-for-byte `Tensor::to_ne_bytes`'s encoding
    /// (the staged path it replaces) for every length 0..=64, random bit
    /// patterns (NaN payloads, subnormals, infinities included) and every
    /// dtype the view serves.
    #[cfg(feature = "cuda")]
    #[test]
    fn stress_ne_bytes_matches_the_staged_encoding() {
        let host = Budget::new(1 << 20);
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for len in 1..=64usize {
            let bits: Vec<u32> = (0..len).map(|_| next() as u32).collect();
            let f: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
            let half: Vec<u16> = bits.iter().map(|&b| b as u16).collect();
            let t = Tensor::from_f32(&f, &[len], &host).unwrap();
            assert_eq!(ne_bytes(t.f32_slice().unwrap()), t.to_ne_bytes().unwrap());
            let t = Tensor::from_u32(&bits, &[len], &host).unwrap();
            assert_eq!(ne_bytes(t.u32_slice().unwrap()), t.to_ne_bytes().unwrap());
            let t = Tensor::from_bf16_bits(&half, &[len], &host).unwrap();
            assert_eq!(ne_bytes(t.bf16_slice().unwrap()), t.to_ne_bytes().unwrap());
        }
        assert!(ne_bytes::<f32>(&[]).is_empty());
    }
}
