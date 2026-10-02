//! Shared test doubles and f64 references for the `ojas-autograd` tests.
//!
//! `Resident` runs the CPU kernels behind a backend id of the test's choice:
//! - `Resident::new()` reports `BackendId::Wgpu`, keeps every tensor in its
//!   own `DeviceBuffer`, and refuses any input that is not one of those with
//!   `Placement`. It exercises the tape's device path with CPU bits.
//! - `Resident::host()` reports `BackendId::Cpu` and keeps host tensors. It
//!   exercises the tape's host path.
//!
//! Both implement `linear_cross_entropy_mean` by composing the unfused CPU
//! ops (`linear_forward` → `cross_entropy_mean_forward`, and
//! `cross_entropy_mean_backward` → `linear_backward`). That is the
//! composition the trait contract names, so the fused result equals the
//! unfused tape path bit for bit. The double ignores `chunk` beyond
//! validating it: it is a reference for the tape, not a memory-bounded
//! kernel. `without_fused` makes the op return `Unsupported`, as the trait
//! default does.
//!
//! Each test binary compiles this module and uses part of it.
#![allow(dead_code)]

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_core::{
    linear_ce_dims, AdamWConfig, Backend, BackendId, Budget, CeChunk, DType, DeviceBuffer,
    LinearCe, MuonNs5Config, Numerics, OjasError, PerHeadGateGrad, Tensor, ValueResidualGrad,
};
use ojas_cpu::CpuBackend;

#[derive(Debug)]
pub struct Bytes {
    id: BackendId,
    bytes: Vec<u8>,
}

impl DeviceBuffer for Bytes {
    fn backend(&self) -> BackendId {
        self.id
    }
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(self.bytes[offset..offset + len].to_vec())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A copy of `host` in a fake `id` device buffer, charged to `budget`.
pub fn on(id: BackendId, host: &Tensor, budget: &Budget) -> Result<Tensor, OjasError> {
    let bytes = host.to_ne_bytes()?;
    Tensor::from_device(
        Arc::new(Bytes { id, bytes }),
        host.shape(),
        host.dtype(),
        budget,
    )
}

/// How `Resident::linear_cross_entropy_mean` answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fused {
    /// The unfused CPU composition.
    Composed,
    /// `Unsupported`, as the trait default.
    Unsupported,
    /// Breaks the contract: no gradients although `want_grad` was set.
    NoGrads,
    /// Breaks the contract: `grad_weight` shaped like the input.
    WrongShape,
}

pub struct Resident {
    cpu: CpuBackend,
    id: BackendId,
    fused: Fused,
    fused_calls: AtomicUsize,
}

impl Resident {
    /// Device double: reports `Wgpu`, keeps tensors in fake device buffers.
    pub fn new() -> Self {
        Self::with_id(BackendId::Wgpu)
    }

    /// Host double: reports `Cpu`, keeps host tensors.
    pub fn host() -> Self {
        Self::with_id(BackendId::Cpu)
    }

    fn with_id(id: BackendId) -> Self {
        Self {
            cpu: CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact),
            id,
            fused: Fused::Composed,
            fused_calls: AtomicUsize::new(0),
        }
    }

    /// `linear_cross_entropy_mean` refuses with `Unsupported`.
    pub fn without_fused(self) -> Self {
        self.with_fused(Fused::Unsupported)
    }

    pub fn with_fused(mut self, fused: Fused) -> Self {
        self.fused = fused;
        self
    }

    /// The CPU backend the kernels run on, sharing this double's budget.
    pub fn cpu(&self) -> &CpuBackend {
        &self.cpu
    }

    /// How many times `linear_cross_entropy_mean` has been called.
    pub fn fused_calls(&self) -> usize {
        self.fused_calls.load(Ordering::SeqCst)
    }

    /// Host copy of one of our tensors for the CPU kernel. Reads the fake
    /// buffer directly, so it is not counted as a readback.
    fn host_of(&self, op: &'static str, t: &Tensor) -> Result<Tensor, OjasError> {
        if self.id == BackendId::Cpu {
            return match t.device() {
                None => Ok(t.clone()),
                found => Err(OjasError::Placement {
                    op,
                    expected: None,
                    found,
                }),
            };
        }
        let placement = OjasError::Placement {
            op,
            expected: Some(self.id),
            found: t.device(),
        };
        if t.device() != Some(self.id) {
            return Err(placement);
        }
        let buf = t
            .device_buffer()
            .and_then(|b| b.as_any().downcast_ref::<Bytes>())
            .ok_or(placement)?;
        if !t.is_contiguous()? {
            return Err(OjasError::Shape {
                op,
                detail: "non-contiguous".to_string(),
            });
        }
        let start = t.byte_offset();
        let window = &buf.bytes[start..start + t.num_elements()? * 4];
        let words = window.as_chunks::<4>().0.iter().copied();
        match t.dtype() {
            DType::F32 => {
                let data: Vec<f32> = words.map(f32::from_ne_bytes).collect();
                Tensor::from_f32(&data, t.shape(), self.cpu.budget())
            }
            DType::U32 => {
                let data: Vec<u32> = words.map(u32::from_ne_bytes).collect();
                Tensor::from_u32(&data, t.shape(), self.cpu.budget())
            }
            other => Err(OjasError::Dtype {
                op,
                expected: DType::F32,
                got: other,
            }),
        }
    }

    fn dev(&self, t: Tensor) -> Result<Tensor, OjasError> {
        if self.id == BackendId::Cpu {
            Ok(t)
        } else {
            on(self.id, &t, self.cpu.budget())
        }
    }
}

impl Default for Resident {
    fn default() -> Self {
        Self::new()
    }
}

pub fn unsupported(op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: "test double".to_string(),
    }
}

impl Backend for Resident {
    fn id(&self) -> BackendId {
        self.id
    }
    fn budget(&self) -> &Budget {
        self.cpu.budget()
    }
    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        match tensor.device() {
            None => self.dev(tensor.clone()),
            Some(found) if found == self.id => Ok(tensor.clone()),
            found => Err(OjasError::Placement {
                op: "Resident::upload",
                expected: Some(self.id),
                found,
            }),
        }
    }
    fn embedding_forward(&self, table: &Tensor, ids: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_forward";
        let y = self
            .cpu
            .embedding_forward(&self.host_of(OP, table)?, &self.host_of(OP, ids)?)?;
        self.dev(y)
    }
    fn embedding_backward(
        &self,
        table: &Tensor,
        ids: &Tensor,
        grad: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        let g = self.cpu.embedding_backward(
            &self.host_of(OP, table)?,
            &self.host_of(OP, ids)?,
            &self.host_of(OP, grad)?,
        )?;
        self.dev(g)
    }
    fn linear_forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let y = self
            .cpu
            .linear_forward(&self.host_of(OP, x)?, &self.host_of(OP, w)?)?;
        self.dev(y)
    }
    fn linear_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        grad: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let (gx, gw) = self.cpu.linear_backward(
            &self.host_of(OP, x)?,
            &self.host_of(OP, w)?,
            &self.host_of(OP, grad)?,
        )?;
        Ok((self.dev(gx)?, self.dev(gw)?))
    }
    fn rms_norm_forward(&self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor, OjasError> {
        const OP: &str = "rms_norm_forward";
        let y = self
            .cpu
            .rms_norm_forward(&self.host_of(OP, x)?, &self.host_of(OP, w)?, eps)?;
        self.dev(y)
    }
    fn rms_norm_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        grad: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let (gx, gw) = self.cpu.rms_norm_backward(
            &self.host_of(OP, x)?,
            &self.host_of(OP, w)?,
            &self.host_of(OP, grad)?,
            eps,
        )?;
        Ok((self.dev(gx)?, self.dev(gw)?))
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
    fn silu_forward(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        let y = self.cpu.silu_forward(&self.host_of("silu_forward", x)?)?;
        self.dev(y)
    }
    fn silu_backward(&self, x: &Tensor, grad: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        let g = self
            .cpu
            .silu_backward(&self.host_of(OP, x)?, &self.host_of(OP, grad)?)?;
        self.dev(g)
    }
    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        let y = self
            .cpu
            .mul_forward(&self.host_of(OP, a)?, &self.host_of(OP, b)?)?;
        self.dev(y)
    }
    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        let (ga, gb) = self.cpu.mul_backward(
            &self.host_of(OP, a)?,
            &self.host_of(OP, b)?,
            &self.host_of(OP, grad)?,
        )?;
        Ok((self.dev(ga)?, self.dev(gb)?))
    }
    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        let z = self
            .cpu
            .residual_add_forward(&self.host_of(OP, x)?, &self.host_of(OP, y)?)?;
        self.dev(z)
    }
    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "residual_add_backward";
        let (gx, gy) = self.cpu.residual_add_backward(
            &self.host_of(OP, x)?,
            &self.host_of(OP, y)?,
            &self.host_of(OP, grad)?,
        )?;
        Ok((self.dev(gx)?, self.dev(gy)?))
    }
    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_forward";
        let y = self.cpu.cross_entropy_mean_forward(
            &self.host_of(OP, logits)?,
            &self.host_of(OP, targets)?,
            ignore,
        )?;
        self.dev(y)
    }
    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_backward";
        let g = self.cpu.cross_entropy_mean_backward(
            &self.host_of(OP, logits)?,
            &self.host_of(OP, targets)?,
            ignore,
        )?;
        self.dev(g)
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
    fn linear_cross_entropy_mean(
        &self,
        input: &Tensor,
        weight: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
        chunk: CeChunk,
        want_grad: bool,
    ) -> Result<LinearCe, OjasError> {
        const OP: &str = "linear_cross_entropy_mean";
        self.fused_calls.fetch_add(1, Ordering::SeqCst);
        if self.fused == Fused::Unsupported {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: "test double built without_fused".to_string(),
            });
        }
        linear_ce_dims(input, weight, targets, chunk)?;
        let x = self.host_of(OP, input)?;
        let w = self.host_of(OP, weight)?;
        let t = self.host_of(OP, targets)?;
        let logits = self.cpu.linear_forward(&x, &w)?;
        let loss = self
            .cpu
            .cross_entropy_mean_forward(&logits, &t, ignore_index)?;
        let (grad_input, grad_weight) = if want_grad {
            let gl = self
                .cpu
                .cross_entropy_mean_backward(&logits, &t, ignore_index)?;
            let (gx, gw) = self.cpu.linear_backward(&x, &w, &gl)?;
            match self.fused {
                Fused::NoGrads => (None, None),
                Fused::WrongShape => (Some(self.dev(gx.clone())?), Some(self.dev(gx)?)),
                _ => (Some(self.dev(gx)?), Some(self.dev(gw)?)),
            }
        } else {
            (None, None)
        };
        Ok(LinearCe {
            loss: self.dev(loss)?,
            grad_input,
            grad_weight,
        })
    }
}

/// Deterministic values in `[-1, 1)`, normal-range, from a hash of the index.
pub fn data(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed * 97)
                % 1000;
            v as f32 / 500.0 - 1.0
        })
        .collect()
}

/// The bits of an `F32` host tensor.
pub fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// f32-representable value in `[-scale, scale)`, returned as f64.
    pub fn unit(&mut self, scale: f32) -> f64 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        f64::from(scale * (2.0 * u - 1.0))
    }

    /// Bounded away from zero, so RMSNorm of a short row stays smooth.
    pub fn away(&mut self) -> f64 {
        let v = self.unit(1.0);
        f64::from((v.signum() * (0.25 + 0.75 * v.abs())) as f32)
    }

    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f64> {
        (0..n).map(|_| self.unit(scale)).collect()
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// f64 `x @ w^T`: `x` is `[rows, kin]`, `w` is `[nout, kin]`.
pub fn lin(x: &[f64], w: &[f64], kin: usize, nout: usize) -> Vec<f64> {
    let rows = x.len() / kin;
    let mut y = vec![0.0; rows * nout];
    for r in 0..rows {
        for c in 0..nout {
            y[r * nout + c] = (0..kin).map(|i| x[r * kin + i] * w[c * kin + i]).sum();
        }
    }
    y
}

/// f64 mean cross-entropy over the rows whose target is not `ignore`.
pub fn ce(logits: &[f64], targets: &[u32], vocab: usize, ignore: Option<u32>) -> f64 {
    let mut total = 0.0;
    let mut n = 0.0;
    for (row, &tg) in targets.iter().enumerate() {
        if ignore == Some(tg) {
            continue;
        }
        let l = &logits[row * vocab..(row + 1) * vocab];
        let m = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + l.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        total += lse - l[tg as usize];
        n += 1.0;
    }
    total / n
}
