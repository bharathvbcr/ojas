//! Shared test doubles for the ojas-model tests.
//!
//! - `Resident` runs every op on `CpuBackend` (Exact) behind a non-CPU
//!   backend id, keeps each tensor in its own fake `DeviceBuffer`, and
//!   refuses any other input with `Placement`. It reads its own buffers
//!   directly, so only `download` (`Tensor::to_host`) counts as a readback.
//!   The in-place ops (`accumulate_grad`, `clip_grad_norm`, `adamw_step`,
//!   `muon_ns5_step`, `kv_cache_write`) refuse a target whose device buffer
//!   is shared, as a real device backend must.
//! - `Probe<B>` forwards every method to `B`, can fail one chosen call of
//!   one chosen op, and can record the gradient and config of every
//!   optimizer call.
//!
//! Each test binary compiles this module and uses part of it.
#![allow(dead_code)]

use std::any::Any;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ojas_core::DataCursor;
use ojas_core::{
    AdamWConfig, AutocastGuard, AutocastMode, Backend, BackendId, Budget, CeChunk, DType,
    DeviceBuffer, LinearCe, MuonNs5Config, Numerics, OjasError, OptimizerKind, PerHeadGateGrad,
    Reservation, Tensor, ValueResidualGrad,
};
use ojas_cpu::CpuBackend;
use ojas_data::TokenBin;
use ojas_model::{MomentsRef, Trainer};

// ---------------------------------------------------------------- Resident

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

pub struct Resident {
    cpu: CpuBackend,
    id: BackendId,
}

impl Resident {
    pub fn new(cap: u64) -> Self {
        Self {
            cpu: CpuBackend::new(Budget::new(cap)).with_numerics(Numerics::Exact),
            id: BackendId::Wgpu,
        }
    }

    fn h(&self, op: &'static str, t: &Tensor) -> Result<Tensor, OjasError> {
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

    fn d(&self, t: Tensor) -> Result<Tensor, OjasError> {
        let bytes = t.to_ne_bytes()?;
        Tensor::from_device(
            Arc::new(Bytes { id: self.id, bytes }),
            t.shape(),
            t.dtype(),
            self.cpu.budget(),
        )
    }

    /// Refuse an in-place target that another handle shares.
    fn unique(op: &'static str, t: &mut Tensor) -> Result<(), OjasError> {
        t.device_buffer_mut()
            .map(|_| ())
            .map_err(|_| OjasError::Shape {
                op,
                detail: "in-place target is shared".to_string(),
            })
    }
}

impl Backend for Resident {
    fn id(&self) -> BackendId {
        self.id
    }
    fn budget(&self) -> &Budget {
        self.cpu.budget()
    }
    fn numerics(&self) -> Numerics {
        Numerics::Exact
    }
    fn upload(&self, t: &Tensor) -> Result<Tensor, OjasError> {
        match t.device() {
            None => self.d(t.clone()),
            Some(found) if found == self.id => Ok(t.clone()),
            found => Err(OjasError::Placement {
                op: "Resident::upload",
                expected: Some(self.id),
                found,
            }),
        }
    }
    fn permute(&self, x: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        self.d(self.cpu.permute(&self.h("permute", x)?, dims)?)
    }
    fn embedding_forward(&self, table: &Tensor, ids: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_forward";
        self.d(self
            .cpu
            .embedding_forward(&self.h(OP, table)?, &self.h(OP, ids)?)?)
    }
    fn embedding_backward(&self, t: &Tensor, i: &Tensor, g: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        self.d(self
            .cpu
            .embedding_backward(&self.h(OP, t)?, &self.h(OP, i)?, &self.h(OP, g)?)?)
    }
    fn linear_forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        self.d(self.cpu.linear_forward(&self.h(OP, x)?, &self.h(OP, w)?)?)
    }
    fn linear_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        g: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let (a, b) = self
            .cpu
            .linear_backward(&self.h(OP, x)?, &self.h(OP, w)?, &self.h(OP, g)?)?;
        Ok((self.d(a)?, self.d(b)?))
    }
    fn rms_norm_forward(&self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor, OjasError> {
        const OP: &str = "rms_norm_forward";
        self.d(self
            .cpu
            .rms_norm_forward(&self.h(OP, x)?, &self.h(OP, w)?, eps)?)
    }
    fn rms_norm_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        g: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let (a, b) =
            self.cpu
                .rms_norm_backward(&self.h(OP, x)?, &self.h(OP, w)?, &self.h(OP, g)?, eps)?;
        Ok((self.d(a)?, self.d(b)?))
    }
    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        c: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_forward";
        self.d(self.cpu.rope_half_split_forward(
            &self.h(OP, x)?,
            &self.h(OP, c)?,
            &self.h(OP, s)?,
        )?)
    }
    fn rope_half_split_backward(
        &self,
        g: &Tensor,
        c: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_backward";
        self.d(self.cpu.rope_half_split_backward(
            &self.h(OP, g)?,
            &self.h(OP, c)?,
            &self.h(OP, s)?,
        )?)
    }
    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        qw: &Tensor,
        kw: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_qk_norm_forward";
        let (a, b) = self.cpu.rms_qk_norm_forward(
            &self.h(OP, q)?,
            &self.h(OP, k)?,
            &self.h(OP, qw)?,
            &self.h(OP, kw)?,
            eps,
        )?;
        Ok((self.d(a)?, self.d(b)?))
    }
    fn rms_qk_norm_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        qw: &Tensor,
        kw: &Tensor,
        gq: &Tensor,
        gk: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "rms_qk_norm_backward";
        let (a, b, c, d) = self.cpu.rms_qk_norm_backward(
            &self.h(OP, q)?,
            &self.h(OP, k)?,
            &self.h(OP, qw)?,
            &self.h(OP, kw)?,
            &self.h(OP, gq)?,
            &self.h(OP, gk)?,
            eps,
        )?;
        Ok((self.d(a)?, self.d(b)?, self.d(c)?, self.d(d)?))
    }
    fn causal_sdpa_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_forward";
        let (y, lse) = self.cpu.causal_sdpa_forward(
            &self.h(OP, q)?,
            &self.h(OP, k)?,
            &self.h(OP, v)?,
            window,
        )?;
        Ok((self.d(y)?, self.d(lse)?))
    }
    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        lse: &Tensor,
        g: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_backward";
        let (a, b, c) = self.cpu.causal_sdpa_backward(
            &self.h(OP, q)?,
            &self.h(OP, k)?,
            &self.h(OP, v)?,
            &self.h(OP, o)?,
            &self.h(OP, lse)?,
            &self.h(OP, g)?,
            window,
        )?;
        Ok((self.d(a)?, self.d(b)?, self.d(c)?))
    }
    fn per_head_sigmoid_gate_forward(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        a: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_forward";
        self.d(self.cpu.per_head_sigmoid_gate_forward(
            &self.h(OP, x)?,
            &self.h(OP, w)?,
            &self.h(OP, b)?,
            &self.h(OP, a)?,
        )?)
    }
    fn per_head_sigmoid_gate_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        a: &Tensor,
        g: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_backward";
        let r = self.cpu.per_head_sigmoid_gate_backward(
            &self.h(OP, x)?,
            &self.h(OP, w)?,
            &self.h(OP, b)?,
            &self.h(OP, a)?,
            &self.h(OP, g)?,
        )?;
        Ok(PerHeadGateGrad {
            input: self.d(r.input)?,
            weight: self.d(r.weight)?,
            bias: self.d(r.bias)?,
            attn_out: self.d(r.attn_out)?,
        })
    }
    fn value_residual_blend_forward(
        &self,
        v: &Tensor,
        v0: &Tensor,
        l: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "value_residual_blend_forward";
        self.d(self.cpu.value_residual_blend_forward(
            &self.h(OP, v)?,
            &self.h(OP, v0)?,
            &self.h(OP, l)?,
        )?)
    }
    fn value_residual_blend_backward(
        &self,
        v: &Tensor,
        v0: &Tensor,
        l: &Tensor,
        g: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        const OP: &str = "value_residual_blend_backward";
        let r = self.cpu.value_residual_blend_backward(
            &self.h(OP, v)?,
            &self.h(OP, v0)?,
            &self.h(OP, l)?,
            &self.h(OP, g)?,
        )?;
        Ok(ValueResidualGrad {
            value: self.d(r.value)?,
            value0: self.d(r.value0)?,
            lambda: self.d(r.lambda)?,
        })
    }
    fn silu_forward(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        self.d(self.cpu.silu_forward(&self.h("silu_forward", x)?)?)
    }
    fn silu_backward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        self.d(self.cpu.silu_backward(&self.h(OP, x)?, &self.h(OP, g)?)?)
    }
    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        self.d(self.cpu.mul_forward(&self.h(OP, a)?, &self.h(OP, b)?)?)
    }
    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        g: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        let (x, y) = self
            .cpu
            .mul_backward(&self.h(OP, a)?, &self.h(OP, b)?, &self.h(OP, g)?)?;
        Ok((self.d(x)?, self.d(y)?))
    }
    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        self.d(self
            .cpu
            .residual_add_forward(&self.h(OP, x)?, &self.h(OP, y)?)?)
    }
    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        g: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "residual_add_backward";
        let (a, b) =
            self.cpu
                .residual_add_backward(&self.h(OP, x)?, &self.h(OP, y)?, &self.h(OP, g)?)?;
        Ok((self.d(a)?, self.d(b)?))
    }
    fn cross_entropy_mean_forward(
        &self,
        l: &Tensor,
        t: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_forward";
        self.d(self
            .cpu
            .cross_entropy_mean_forward(&self.h(OP, l)?, &self.h(OP, t)?, ignore)?)
    }
    fn cross_entropy_mean_backward(
        &self,
        l: &Tensor,
        t: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_backward";
        self.d(self
            .cpu
            .cross_entropy_mean_backward(&self.h(OP, l)?, &self.h(OP, t)?, ignore)?)
    }
    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        const OP: &str = "clip_grad_norm";
        let mut hosts = Vec::with_capacity(grads.len());
        for g in grads.iter_mut() {
            Self::unique(OP, g)?;
            hosts.push(self.h(OP, g)?);
        }
        let norm = self.cpu.clip_grad_norm(&mut hosts, max_norm)?;
        let fresh = hosts
            .into_iter()
            .map(|h| self.d(h))
            .collect::<Result<Vec<_>, _>>()?;
        for (g, f) in grads.iter_mut().zip(fresh) {
            *g = f;
        }
        Ok(norm)
    }
    fn adamw_step(
        &self,
        p: &mut Tensor,
        g: &Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError> {
        const OP: &str = "adamw_step";
        for t in [&mut *p, &mut *m, &mut *v] {
            Self::unique(OP, t)?;
        }
        let (mut hp, mut hm, mut hv) = (self.h(OP, p)?, self.h(OP, m)?, self.h(OP, v)?);
        self.cpu
            .adamw_step(&mut hp, &self.h(OP, g)?, &mut hm, &mut hv, step, config)?;
        let (np, nm, nv) = (self.d(hp)?, self.d(hm)?, self.d(hv)?);
        (*p, *m, *v) = (np, nm, nv);
        Ok(())
    }
    fn muon_ns5_step(
        &self,
        p: &mut Tensor,
        g: &Tensor,
        m: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        const OP: &str = "muon_ns5_step";
        for t in [&mut *p, &mut *m] {
            Self::unique(OP, t)?;
        }
        let (mut hp, mut hm) = (self.h(OP, p)?, self.h(OP, m)?);
        self.cpu
            .muon_ns5_step(&mut hp, &self.h(OP, g)?, &mut hm, config)?;
        let (np, nm) = (self.d(hp)?, self.d(hm)?);
        (*p, *m) = (np, nm);
        Ok(())
    }
    fn accumulate_grad(&self, acc: &mut Tensor, g: &Tensor) -> Result<(), OjasError> {
        const OP: &str = "accumulate_grad";
        Self::unique(OP, acc)?;
        let mut h = self.h(OP, acc)?;
        self.cpu.accumulate_grad(&mut h, &self.h(OP, g)?)?;
        *acc = self.d(h)?;
        Ok(())
    }
    fn linear_cross_entropy_mean(
        &self,
        x: &Tensor,
        w: &Tensor,
        t: &Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
        want_grad: bool,
    ) -> Result<LinearCe, OjasError> {
        const OP: &str = "linear_cross_entropy_mean";
        let r = self.cpu.linear_cross_entropy_mean(
            &self.h(OP, x)?,
            &self.h(OP, w)?,
            &self.h(OP, t)?,
            ignore,
            chunk,
            want_grad,
        )?;
        Ok(LinearCe {
            loss: self.d(r.loss)?,
            grad_input: r.grad_input.map(|g| self.d(g)).transpose()?,
            grad_weight: r.grad_weight.map(|g| self.d(g)).transpose()?,
        })
    }
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cached_attention_forward";
        self.d(self.cpu.cached_attention_forward(
            &self.h(OP, q)?,
            &self.h(OP, k)?,
            &self.h(OP, v)?,
            kv_len,
        )?)
    }
    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        const OP: &str = "kv_cache_write";
        Self::unique(OP, cache)?;
        let mut h = self.h(OP, cache)?;
        self.cpu.kv_cache_write(&mut h, &self.h(OP, src)?, at)?;
        *cache = self.d(h)?;
        Ok(())
    }
    fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "cast_bf16";
        // Read this double's own shadow. The trait default refuses a device tensor.
        self.d(self.cpu.cast_bf16(&self.h(OP, tensor)?)?)
    }
}

// ------------------------------------------------------------------- Probe

/// What a chosen call does instead of running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    NonFinite,
    Capacity,
    Backend,
    /// Run the op, then replace its output with NaN (`linear_forward` only).
    NanOutput,
}

impl Fault {
    fn error(self, op: &'static str) -> OjasError {
        match self {
            Fault::NonFinite | Fault::NanOutput => OjasError::NonFinite { op },
            Fault::Capacity => OjasError::CapacityExceeded {
                requested: 1,
                cap: 0,
                live: 0,
            },
            Fault::Backend => OjasError::Backend {
                id: BackendId::Cpu,
                detail: format!("injected fault in {op}"),
            },
        }
    }
}

/// One optimizer call, as the trainer made it.
#[derive(Clone, Debug)]
pub enum OptCall {
    Muon {
        grad: Vec<f32>,
        config: MuonNs5Config,
    },
    AdamW {
        grad: Vec<f32>,
        step: u64,
        config: AdamWConfig,
    },
}

pub struct Probe<B: Backend> {
    inner: B,
    counts: Mutex<BTreeMap<&'static str, usize>>,
    /// `(op, call index from 0, fault)`.
    fault: Mutex<Option<(&'static str, usize, Fault)>>,
    record: bool,
    calls: Mutex<Vec<OptCall>>,
    /// `budget().live_bytes()` just before each `download`.
    download_live: Mutex<Vec<u64>>,
    /// `(budget().live_bytes(), tensor bytes)` just before each `upload`.
    upload_live: Mutex<Vec<(u64, u64)>>,
    /// `(clip_grad_norm call index, room)`: after that call, hold budget so
    /// that exactly `room` bytes stay free (another holder taking memory
    /// between the gradients and the optimizer).
    squeeze: Mutex<Option<(usize, u64)>>,
    held: Mutex<Vec<Reservation>>,
}

impl<B: Backend> Probe<B> {
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            counts: Mutex::new(BTreeMap::new()),
            fault: Mutex::new(None),
            record: false,
            calls: Mutex::new(Vec::new()),
            download_live: Mutex::new(Vec::new()),
            upload_live: Mutex::new(Vec::new()),
            squeeze: Mutex::new(None),
            held: Mutex::new(Vec::new()),
        }
    }

    /// After call `nth` (from 0, counted from now on) of `clip_grad_norm`,
    /// leave only `room` bytes of the budget free until [`Self::release`].
    pub fn squeeze_after_clip(&self, nth: usize, room: u64) {
        let base = self.count("clip_grad_norm");
        *self.squeeze.lock().unwrap() = Some((base + nth, room));
    }

    /// Drop every hold of [`Self::squeeze_after_clip`] and disarm it.
    pub fn release(&self) {
        *self.squeeze.lock().unwrap() = None;
        self.held.lock().unwrap().clear();
    }

    /// Record every optimizer call's gradient (read on the host).
    pub fn recording(mut self) -> Self {
        self.record = true;
        self
    }

    /// Fail call `nth` (from 0, counted from now on) of `op`.
    pub fn fail(&self, op: &'static str, nth: usize, fault: Fault) {
        let base = self.count(op);
        *self.fault.lock().unwrap() = Some((op, base + nth, fault));
    }

    pub fn clear_fault(&self) {
        *self.fault.lock().unwrap() = None;
    }

    pub fn count(&self, op: &'static str) -> usize {
        self.counts.lock().unwrap().get(op).copied().unwrap_or(0)
    }

    pub fn take_calls(&self) -> Vec<OptCall> {
        std::mem::take(&mut *self.calls.lock().unwrap())
    }

    /// Live budget bytes seen at each `download` since the last call.
    pub fn take_download_live(&self) -> Vec<u64> {
        std::mem::take(&mut *self.download_live.lock().unwrap())
    }

    /// Live budget bytes and the tensor's bytes at each `upload` since the
    /// last call.
    pub fn take_upload_live(&self) -> Vec<(u64, u64)> {
        std::mem::take(&mut *self.upload_live.lock().unwrap())
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// Count the call; the injected fault if this is the chosen one.
    fn hit(&self, op: &'static str) -> Option<Fault> {
        let mut counts = self.counts.lock().unwrap();
        let n = counts.entry(op).or_insert(0);
        let index = *n;
        *n += 1;
        match *self.fault.lock().unwrap() {
            Some((fop, at, fault)) if fop == op && at == index => Some(fault),
            _ => None,
        }
    }

    fn gate(&self, op: &'static str) -> Result<(), OjasError> {
        match self.hit(op) {
            Some(fault) => Err(fault.error(op)),
            None => Ok(()),
        }
    }

    fn host_values(&self, t: &Tensor) -> Vec<f32> {
        let host = match t.device() {
            None => t.clone(),
            Some(_) => t.to_host(&Budget::new(u64::MAX)).unwrap(),
        };
        host.to_f32_vec().unwrap()
    }
}

impl<B: Backend> Backend for Probe<B> {
    fn id(&self) -> BackendId {
        self.inner.id()
    }
    fn budget(&self) -> &Budget {
        self.inner.budget()
    }
    fn numerics(&self) -> Numerics {
        self.inner.numerics()
    }
    fn upload(&self, t: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("upload")?;
        let live = self.inner.budget().live_bytes()?;
        let bytes = (t.num_elements()? * t.dtype().size()) as u64;
        self.upload_live.lock().unwrap().push((live, bytes));
        self.inner.upload(t)
    }
    fn download(&self, t: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("download")?;
        let live = self.inner.budget().live_bytes()?;
        self.download_live.lock().unwrap().push(live);
        self.inner.download(t)
    }
    fn permute(&self, x: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        self.gate("permute")?;
        self.inner.permute(x, dims)
    }
    fn embedding_forward(&self, t: &Tensor, i: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("embedding_forward")?;
        self.inner.embedding_forward(t, i)
    }
    fn embedding_backward(&self, t: &Tensor, i: &Tensor, g: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("embedding_backward")?;
        self.inner.embedding_backward(t, i, g)
    }
    fn linear_forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, OjasError> {
        match self.hit("linear_forward") {
            Some(Fault::NanOutput) => {
                let y = self.inner.linear_forward(x, w)?;
                let nan = vec![f32::NAN; y.num_elements()?];
                let host = Tensor::from_f32(&nan, y.shape(), self.inner.budget())?;
                if y.device().is_some() {
                    self.inner.upload(&host)
                } else {
                    Ok(host)
                }
            }
            Some(fault) => Err(fault.error("linear_forward")),
            None => self.inner.linear_forward(x, w),
        }
    }
    fn linear_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        g: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.gate("linear_backward")?;
        self.inner.linear_backward(x, w, g)
    }
    fn rms_norm_forward(&self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor, OjasError> {
        self.gate("rms_norm_forward")?;
        self.inner.rms_norm_forward(x, w, eps)
    }
    fn rms_norm_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        g: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.gate("rms_norm_backward")?;
        self.inner.rms_norm_backward(x, w, g, eps)
    }
    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        c: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.gate("rope_half_split_forward")?;
        self.inner.rope_half_split_forward(x, c, s)
    }
    fn rope_half_split_backward(
        &self,
        g: &Tensor,
        c: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.gate("rope_half_split_backward")?;
        self.inner.rope_half_split_backward(g, c, s)
    }
    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        qw: &Tensor,
        kw: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.gate("rms_qk_norm_forward")?;
        self.inner.rms_qk_norm_forward(q, k, qw, kw, eps)
    }
    fn rms_qk_norm_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        qw: &Tensor,
        kw: &Tensor,
        gq: &Tensor,
        gk: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        self.gate("rms_qk_norm_backward")?;
        self.inner.rms_qk_norm_backward(q, k, qw, kw, gq, gk, eps)
    }
    fn causal_sdpa_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.gate("causal_sdpa_forward")?;
        self.inner.causal_sdpa_forward(q, k, v, window)
    }
    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        lse: &Tensor,
        g: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        self.gate("causal_sdpa_backward")?;
        self.inner.causal_sdpa_backward(q, k, v, o, lse, g, window)
    }
    fn per_head_sigmoid_gate_forward(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        a: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.gate("per_head_sigmoid_gate_forward")?;
        self.inner.per_head_sigmoid_gate_forward(x, w, b, a)
    }
    fn per_head_sigmoid_gate_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        a: &Tensor,
        g: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        self.gate("per_head_sigmoid_gate_backward")?;
        self.inner.per_head_sigmoid_gate_backward(x, w, b, a, g)
    }
    fn value_residual_blend_forward(
        &self,
        v: &Tensor,
        v0: &Tensor,
        l: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.gate("value_residual_blend_forward")?;
        self.inner.value_residual_blend_forward(v, v0, l)
    }
    fn value_residual_blend_backward(
        &self,
        v: &Tensor,
        v0: &Tensor,
        l: &Tensor,
        g: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        self.gate("value_residual_blend_backward")?;
        self.inner.value_residual_blend_backward(v, v0, l, g)
    }
    fn silu_forward(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("silu_forward")?;
        self.inner.silu_forward(x)
    }
    fn silu_backward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("silu_backward")?;
        self.inner.silu_backward(x, g)
    }
    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("mul_forward")?;
        self.inner.mul_forward(a, b)
    }
    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        g: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.gate("mul_backward")?;
        self.inner.mul_backward(a, b, g)
    }
    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("residual_add_forward")?;
        self.inner.residual_add_forward(x, y)
    }
    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        g: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.gate("residual_add_backward")?;
        self.inner.residual_add_backward(x, y, g)
    }
    fn cross_entropy_mean_forward(
        &self,
        l: &Tensor,
        t: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        self.gate("cross_entropy_mean_forward")?;
        self.inner.cross_entropy_mean_forward(l, t, ignore)
    }
    fn cross_entropy_mean_backward(
        &self,
        l: &Tensor,
        t: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        self.gate("cross_entropy_mean_backward")?;
        self.inner.cross_entropy_mean_backward(l, t, ignore)
    }
    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        self.gate("clip_grad_norm")?;
        let norm = self.inner.clip_grad_norm(grads, max_norm)?;
        let index = self.count("clip_grad_norm") - 1;
        let squeeze = *self.squeeze.lock().unwrap();
        if let Some((at, room)) = squeeze {
            if at == index {
                let budget = self.inner.budget();
                let free = budget.cap_bytes() - budget.live_bytes()?;
                if free > room {
                    let held = budget.try_reserve(free - room)?;
                    self.held.lock().unwrap().push(held);
                }
            }
        }
        Ok(norm)
    }
    fn adamw_step(
        &self,
        p: &mut Tensor,
        g: &Tensor,
        m: &mut Tensor,
        v: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError> {
        self.gate("adamw_step")?;
        if self.record {
            let grad = self.host_values(g);
            self.calls
                .lock()
                .unwrap()
                .push(OptCall::AdamW { grad, step, config });
        }
        self.inner.adamw_step(p, g, m, v, step, config)
    }
    fn muon_ns5_step(
        &self,
        p: &mut Tensor,
        g: &Tensor,
        m: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        self.gate("muon_ns5_step")?;
        if self.record {
            let grad = self.host_values(g);
            self.calls
                .lock()
                .unwrap()
                .push(OptCall::Muon { grad, config });
        }
        self.inner.muon_ns5_step(p, g, m, config)
    }
    fn optimizer_scratch_bytes(
        &self,
        kind: OptimizerKind,
        rows: usize,
        cols: usize,
    ) -> Result<Option<u64>, OjasError> {
        self.inner.optimizer_scratch_bytes(kind, rows, cols)
    }
    fn sync(&self) -> Result<(), OjasError> {
        self.gate("sync")?;
        self.inner.sync()
    }
    fn accumulate_grad(&self, acc: &mut Tensor, g: &Tensor) -> Result<(), OjasError> {
        self.gate("accumulate_grad")?;
        self.inner.accumulate_grad(acc, g)
    }
    fn linear_cross_entropy_mean(
        &self,
        x: &Tensor,
        w: &Tensor,
        t: &Tensor,
        ignore: Option<u32>,
        chunk: CeChunk,
        want_grad: bool,
    ) -> Result<LinearCe, OjasError> {
        self.gate("linear_cross_entropy_mean")?;
        self.inner
            .linear_cross_entropy_mean(x, w, t, ignore, chunk, want_grad)
    }
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        self.gate("cached_attention_forward")?;
        self.inner.cached_attention_forward(q, k, v, kv_len)
    }
    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        self.gate("kv_cache_write")?;
        self.inner.kv_cache_write(cache, src, at)
    }
    fn argmax_rows(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("argmax_rows")?;
        self.inner.argmax_rows(x)
    }
    fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.gate("cast_bf16")?;
        self.inner.cast_bf16(tensor)
    }
    fn autocast_region(&self, mode: AutocastMode) -> Result<AutocastGuard, OjasError> {
        self.gate("autocast_region")?;
        self.inner.autocast_region(mode)
    }
}

// ----------------------------------------------------------------- helpers

/// A headerless u16 token file that is deleted on drop.
pub struct TempBin {
    pub path: PathBuf,
}

impl Drop for TempBin {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

static BIN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// `len` tokens below `vocab` with a learnable next-token rule
/// (`x' = (5x + 3) mod vocab`, restarted from a varying seed every 97
/// tokens), written to a temp file.
pub fn token_bin(len: usize, vocab: u32) -> (TempBin, TokenBin) {
    let (bytes, is_u32) = if vocab <= 65536 {
        let mut bytes = Vec::with_capacity(len * 2);
        let mut x = 1u32;
        for i in 0..len {
            if i % 97 == 0 {
                x = (i as u32 / 97 * 31 + 7) % vocab;
            } else {
                x = (5 * x + 3) % vocab;
            }
            bytes.extend_from_slice(&(x as u16).to_le_bytes());
        }
        (bytes, false)
    } else {
        let mut bytes = Vec::with_capacity(len * 4);
        let mut x = 1u32;
        for i in 0..len {
            if i % 97 == 0 {
                x = (i as u32 / 97 * 31 + 7) % vocab;
            } else {
                x = (5 * x + 3) % vocab;
            }
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        (bytes, true)
    };
    let n = BIN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("ojas-model-test-{}-{n}.bin", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let bin = if is_u32 {
        TokenBin::open_headerless_u32(&path).unwrap()
    } else {
        TokenBin::open_headerless(&path).unwrap()
    };
    (TempBin { path }, bin)
}

/// Bits of an `F32` tensor, read back through `backend` if it is resident.
pub fn bits<B: Backend + ?Sized>(backend: &B, t: &Tensor) -> Vec<u32> {
    backend
        .download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

pub fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Every parameter's and moment's bits, the step and the cursor.
#[derive(Debug, PartialEq)]
pub struct Snapshot {
    pub params: Vec<Vec<u32>>,
    pub moments: Vec<Vec<Vec<u32>>>,
    pub step: u64,
    pub cursor: DataCursor,
}

pub fn snapshot<B: Backend>(t: &Trainer<B>) -> Snapshot {
    let b = t.backend();
    let mut params = Vec::new();
    let mut moments = Vec::new();
    for (info, value) in t.params() {
        params.push(bits(b, value));
        moments.push(match t.moments(&info.name).unwrap() {
            MomentsRef::Muon { momentum } => vec![bits(b, momentum)],
            MomentsRef::AdamW { m, v } => vec![bits(b, m), bits(b, v)],
            MomentsRef::Frozen => vec![],
        });
    }
    Snapshot {
        params,
        moments,
        step: t.step_count(),
        cursor: t.cursor(),
    }
}

/// A fresh directory under the system temp dir, removed on drop.
pub struct TempDir {
    pub path: PathBuf,
}

static DIRS: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    pub fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("ojas-model-ckpt-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir(&path).unwrap();
        Self { path }
    }

    pub fn ckpt(&self) -> PathBuf {
        self.path.join("ckpt")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
