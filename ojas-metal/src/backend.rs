//! [`MetalBackend`]: `ojas_core::Backend` on device-resident Metal tensors.
//!
//! Every input must already live on this backend's device (see
//! [`Backend::upload`]); a host tensor or another device's tensor is
//! [`OjasError::Placement`]. Outputs are fresh device buffers wrapped with
//! [`Tensor::from_device_reserved`], so nothing comes back to the host until
//! [`Tensor::to_host`] or [`Backend::download`].
//!
//! Each op reserves its output and scratch bytes in the [`Budget`] before the
//! device allocates anything, and refuses with
//! [`OjasError::CapacityExceeded`] when they do not fit. Device memory is
//! tessl's pool, which rounds each allocation up to a power of two (at least
//! 256 bytes); the budget charges the logical byte count, so resident device
//! memory can be up to twice the budget's live bytes.
//!
//! Numerics are [`Numerics::Fast`]: GEMMs are tessl TensorOps with their own
//! reduction order, and reductions are threadgroup trees, not ascending f32.
//!
//! Faults are deferred (`docs/metal-deferred-faults.md`). Ops are recorded
//! into tessl's command buffer and return without waiting. A NaN or infinity
//! a kernel finds does not fail its op: the next [`Backend::sync`],
//! [`Backend::download`] or [`Backend::clip_grad_norm`] on this backend
//! returns [`OjasError::NonFinite`] naming the first faulting op in
//! recording order, once. Everything the host can decide stays immediate:
//! shape, dtype, placement, capacity, limits, token ids (checked against the
//! host's copy of every U32 upload) and an all-ignored cross-entropy batch.
//! So an input that is both misshapen and non-finite reports the shape
//! error, and an out-of-range target is refused even beside a non-finite
//! logit, as on wgpu.
//!
//! Shape and dtype come first (`docs/shape-contract.md`): every op calls its
//! `ojas_core::shapes` validator before it checks placement or contiguity,
//! range-checks ids, applies a device limit (the head-dim cap, the 32-bit
//! index caps) or charges the budget, so a malformed call reports what the
//! CPU and wgpu report, whatever the budget holds and wherever its operands
//! live.

use std::any::Any;
use std::fmt;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use ojas_core::{
    adamw_step_dims, cached_attention_dims, causal_sdpa_backward_dims, causal_sdpa_forward_dims,
    check_adamw, clip_grad_norm_dims, clip_scale, cross_entropy_mean_backward_dims,
    cross_entropy_mean_forward_dims, embedding_backward_dims, embedding_forward_dims,
    kv_cache_write_dims, linear_backward_dims, linear_ce_dims, linear_forward_dims,
    mul_backward_dims, mul_forward_dims, muon_ns5_step_dims, per_head_sigmoid_gate_backward_dims,
    per_head_sigmoid_gate_forward_dims, permute_output_shape, refuse_unsupported_metal_head_dim,
    residual_add_backward_dims, residual_add_forward_dims, rms_norm_backward_dims,
    rms_norm_forward_dims, rms_qk_norm_backward_dims, rms_qk_norm_forward_dims,
    rope_half_split_backward_dims, rope_half_split_forward_dims, silu_backward_dims,
    silu_forward_dims, value_residual_blend_backward_dims, value_residual_blend_forward_dims,
    AdamWConfig, Backend, BackendId, Budget, CeChunk, DType, DeviceBuffer, GateDims, LinearCe,
    MuonNs5Config, Numerics, OjasError, PerHeadGateGrad, Reservation, RmsDims, RopeDims,
    RopeLayout, SdpaDims, Tensor, ValueResidualGrad, MAX_PERMUTE_RANK,
};

use crate::link::{
    metal_err, rms_w_chunks, Arg, Cmd, LceGeom, Link, Reply, Res, RmsSide, RopeMode,
};

/// A Metal allocation owned by the device thread, named by id.
pub struct MetalBuffer {
    id: u64,
    len: usize,
    link: Arc<Link>,
    /// The host's copy of a U32 upload. Every U32 Metal tensor is an upload
    /// (no op produces one), so token ids are range-checked, and
    /// cross-entropy's valid rows counted, here rather than on the device.
    ids: Option<Arc<[u32]>>,
}

impl fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("id", &self.id)
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        self.link.post(Cmd::Free { id: self.id });
    }
}

impl DeviceBuffer for MetalBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }

    fn byte_len(&self) -> usize {
        self.len
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        match self.link.call(Cmd::Read {
            id: self.id,
            off: offset,
            len,
        })? {
            Reply::Bytes(bytes) => Ok(bytes),
            other => Err(metal_err(format!("read returned {other:?}"))),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Device-resident Metal backend. Clones share one device thread; any
/// number of threads may issue ops on one backend, and the device runs them
/// one at a time.
#[derive(Clone)]
pub struct MetalBackend {
    link: Arc<Link>,
    budget: Budget,
}

impl fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device", &self.link.device_name())
            .finish()
    }
}

impl MetalBackend {
    /// Open the default Metal device on its own thread. Off macOS, or without
    /// the `metal` feature, this is [`OjasError::Unsupported`].
    pub fn new(budget: Budget) -> Result<Self, OjasError> {
        let waits = Arc::new(AtomicU64::new(0));
        let (tx, name) = crate::device::spawn(Arc::clone(&waits), budget.cap_bytes())?;
        Ok(Self {
            link: Arc::new(Link::new(tx, name, waits)),
            budget,
        })
    }

    pub fn device_name(&self) -> &str {
        self.link.device_name()
    }

    /// Waited GPU commits this backend's device thread has made since it
    /// opened, shared by every clone. Telemetry for tests and benches, not a
    /// stable API.
    #[doc(hidden)]
    pub fn waits(&self) -> u64 {
        self.link.waits()
    }

    /// Another handle on this backend's device that charges `budget`
    /// instead: it accepts this backend's tensors and shares its device
    /// thread, faults and `sync`, as wgpu's `WgpuBackend::with_context`
    /// shares a context. The device's memory-cap trigger stays the one set
    /// from the opening backend's cap. Lets a test hold operands on one
    /// budget and run an op under another (`tests/shape_first.rs`); not a
    /// stable API.
    #[doc(hidden)]
    pub fn with_budget(&self, budget: Budget) -> Self {
        Self {
            link: Arc::clone(&self.link),
            budget,
        }
    }

    fn arg(&self, op: &'static str, t: &Tensor, dtype: DType) -> Res<Arg> {
        let buf = t.device_buffer().ok_or(OjasError::Placement {
            op,
            expected: Some(BackendId::Metal),
            found: None,
        })?;
        if buf.backend() != BackendId::Metal {
            return Err(OjasError::Placement {
                op,
                expected: Some(BackendId::Metal),
                found: Some(buf.backend()),
            });
        }
        let mb = buf
            .as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or_else(|| metal_err(format!("{op}: Metal tensor is not an ojas-metal buffer")))?;
        if !Arc::ptr_eq(&mb.link, &self.link) {
            return Err(metal_err(format!(
                "{op}: tensor belongs to a different MetalBackend device"
            )));
        }
        if t.dtype() != dtype {
            return Err(OjasError::Dtype {
                op,
                expected: dtype,
                got: t.dtype(),
            });
        }
        if !t.is_contiguous()? {
            return Err(shape(op, "non-contiguous view is not supported"));
        }
        let n = t.num_elements()?;
        if n == 0 || t.shape().contains(&0) {
            return Err(shape(op, "empty tensor"));
        }
        if n > u32::MAX as usize {
            return Err(OjasError::Unsupported {
                op,
                detail: format!("{n} elements exceed the kernels' 32-bit indexing"),
            });
        }
        Ok(Arg {
            id: mb.id,
            off: t.byte_offset(),
            n,
        })
    }

    fn f32(&self, op: &'static str, t: &Tensor) -> Res<Arg> {
        self.arg(op, t, DType::F32)
    }

    /// A U32 tensor, with its ids range-checked against `limit` on the host
    /// (`ignore` is skipped). Returns the count of ids neither ignored nor
    /// out of range. The first out-of-range position is
    /// [`OjasError::OutOfRange`], as the device check reported it.
    fn ids(
        &self,
        op: &'static str,
        t: &Tensor,
        limit: u32,
        ignore: Option<u32>,
    ) -> Res<(Arg, u32)> {
        let a = self.arg(op, t, DType::U32)?;
        let host = t
            .device_buffer()
            .and_then(|b| b.as_any().downcast_ref::<MetalBuffer>())
            .and_then(|mb| mb.ids.as_ref())
            .ok_or_else(|| metal_err(format!("{op}: U32 tensor has no host copy of its ids")))?;
        let start = a.off / 4;
        let window = host
            .get(start..start + a.n)
            .ok_or_else(|| metal_err(format!("{op}: id window is outside its buffer")))?;
        let mut valid = 0u32;
        for (i, &id) in window.iter().enumerate() {
            if ignore == Some(id) {
                continue;
            }
            if id >= limit {
                return Err(OjasError::OutOfRange {
                    op,
                    detail: format!("index at position {i} is out of range"),
                });
            }
            valid += 1;
        }
        Ok((a, valid))
    }

    fn reserve(&self, op: &'static str, elems: usize) -> Res<Reservation> {
        let bytes = elems.checked_mul(4).ok_or_else(|| overflow(op))?;
        self.budget.try_reserve(bytes as u64)
    }

    /// Reserve every output, hold `scratch` elements for the call, run `cmd`,
    /// and wrap the outputs in the order given.
    fn outputs(
        &self,
        op: &'static str,
        shapes: &[&[usize]],
        scratch: usize,
        cmd: Cmd,
    ) -> Res<Vec<Tensor>> {
        let mut reservations = Vec::with_capacity(shapes.len());
        for s in shapes {
            reservations.push(self.reserve(op, product(op, s)?)?);
        }
        let _scratch = self.reserve(op, scratch)?;
        self.call_wrap(op, shapes, reservations, cmd)
    }

    /// Run `cmd` and wrap the buffers it returns, in order, as tensors of
    /// `shapes` holding `reservations`.
    fn call_wrap(
        &self,
        op: &'static str,
        shapes: &[&[usize]],
        reservations: Vec<Reservation>,
        cmd: Cmd,
    ) -> Res<Vec<Tensor>> {
        let bufs = match self.link.call(cmd)? {
            Reply::Bufs(bufs) => bufs,
            other => return Err(metal_err(format!("{op}: device returned {other:?}"))),
        };
        let owned: Vec<Arc<MetalBuffer>> = bufs
            .into_iter()
            .map(|b| {
                Arc::new(MetalBuffer {
                    id: b.id,
                    len: b.bytes,
                    link: Arc::clone(&self.link),
                    ids: None,
                })
            })
            .collect();
        if owned.len() != shapes.len() {
            return Err(metal_err(format!(
                "{op}: device returned {} outputs, expected {}",
                owned.len(),
                shapes.len()
            )));
        }
        let mut out = Vec::with_capacity(owned.len());
        for ((buf, s), r) in owned.into_iter().zip(shapes).zip(reservations) {
            let dynbuf: Arc<dyn DeviceBuffer> = buf;
            out.push(Tensor::from_device_reserved(dynbuf, s, DType::F32, r)?);
        }
        Ok(out)
    }

    fn one(&self, op: &'static str, s: &[usize], scratch: usize, cmd: Cmd) -> Res<Tensor> {
        let mut v = self.outputs(op, &[s], scratch, cmd)?;
        v.pop().ok_or_else(|| metal_err(format!("{op}: no output")))
    }

    fn two(&self, op: &'static str, a: &[usize], b: &[usize], cmd: Cmd) -> Res<(Tensor, Tensor)> {
        let mut v = self.outputs(op, &[a, b], 0, cmd)?.into_iter();
        match (v.next(), v.next()) {
            (Some(x), Some(y)) => Ok((x, y)),
            _ => Err(metal_err(format!("{op}: missing outputs"))),
        }
    }

    /// One RMSNorm operand set whose shapes `dims` already validated: x, w
    /// and the gradient placed in that order, then the 32-bit caps.
    fn rms_side(
        &self,
        op: &'static str,
        dims: RmsDims,
        x: &Tensor,
        w: &Tensor,
        gy: Option<&Tensor>,
    ) -> Res<RmsSide> {
        let xa = self.f32(op, x)?;
        let wa = self.f32(op, w)?;
        let ga = gy.map(|g| self.f32(op, g)).transpose()?;
        Ok(RmsSide {
            x: xa,
            w: wa,
            gy: ga,
            rows: u32_dim(op, dims.rows)?,
            dim: u32_dim(op, dims.dim)?,
        })
    }

    /// QK-norm as one command (one dispatch sequence instead of two).
    ///
    /// Both pairs are validated (`rms_qk_norm_*_dims`), then q's operands
    /// and then k's are placed, before anything is reserved or recorded, so
    /// a refused call, q's side or k's, records nothing. q is then reserved
    /// as its own call would be. If k's outputs and scratch do not fit the
    /// budget beside q's, this returns `None` and the caller runs the
    /// composition, `rms_norm_*` on q and then on k, which holds only one
    /// side's scratch at a time and records q before k's capacity refusal.
    /// On the device each side has its own status slot, q's recorded first.
    /// Outputs are q's (y, or gx and gw) then k's.
    fn rms_pair(
        &self,
        op: &'static str,
        q: [&Tensor; 2],
        k: [&Tensor; 2],
        grads: Option<[&Tensor; 2]>,
        eps: f32,
    ) -> Res<Option<Vec<Tensor>>> {
        let (qd, kd) = match grads {
            None => rms_qk_norm_forward_dims(q[0], k[0], q[1], k[1], eps)?,
            Some(g) => rms_qk_norm_backward_dims(q[0], k[0], q[1], k[1], g[0], g[1], eps)?,
        };
        let q_side = self.rms_side(op, qd, q[0], q[1], grads.map(|g| g[0]))?;
        let k_side = self.rms_side(op, kd, k[0], k[1], grads.map(|g| g[1]))?;
        let shapes_of = |side: [&Tensor; 2]| -> Vec<Vec<usize>> {
            if grads.is_some() {
                vec![side[0].shape().to_vec(), side[1].shape().to_vec()]
            } else {
                vec![side[0].shape().to_vec()]
            }
        };
        let scratch_of = |s: &RmsSide| if grads.is_some() { rms_bwd_scratch(s) } else { 0 };
        let (q_shapes, k_shapes) = (shapes_of(q), shapes_of(k));
        let mut reservations = Vec::with_capacity(q_shapes.len() + k_shapes.len());
        for s in &q_shapes {
            reservations.push(self.reserve(op, product(op, s)?)?);
        }
        let _q_scratch = self.reserve(op, scratch_of(&q_side))?;
        for s in &k_shapes {
            match product(op, s).and_then(|n| self.reserve(op, n)) {
                Ok(r) => reservations.push(r),
                Err(_) => return Ok(None),
            }
        }
        let Ok(_k_scratch) = self.reserve(op, scratch_of(&k_side)) else {
            return Ok(None);
        };
        let shapes: Vec<&[usize]> = q_shapes.iter().chain(&k_shapes).map(Vec::as_slice).collect();
        let cmd = Cmd::Rms {
            sides: vec![q_side, k_side],
            eps,
        };
        self.call_wrap(op, &shapes, reservations, cmd).map(Some)
    }

    /// The device id of a tensor this call may write in place. The tensor
    /// must solely own its buffer, as [`Tensor::write_f32`] requires on the
    /// host; a shared one is [`OjasError::Shape`].
    fn unique(&self, op: &'static str, t: &mut Tensor) -> Res<()> {
        let buf = t.device_buffer_mut()?;
        if buf.as_any().downcast_ref::<MetalBuffer>().is_none() {
            return Err(metal_err(format!("{op}: not an ojas-metal buffer")));
        }
        Ok(())
    }
}

/// rstd and the weight gradient's partial sums of one backward side.
fn rms_bwd_scratch(side: &RmsSide) -> usize {
    side.rows as usize + rms_w_chunks(side.rows) as usize * side.dim as usize
}

fn shape(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op,
        detail: detail.into(),
    }
}

fn overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "size overflows".to_string(),
    }
}

fn product(op: &'static str, dims: &[usize]) -> Res<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| overflow(op))
}

/// The per-step f32 scalars of `ojas_adamw_*`, formed in f64 from the bias
/// corrections [`check_adamw`] returns (so `1 - beta^step` has the CPU
/// reference's bits), in `OjasAdamW` field order. A scalar that does not
/// survive the f32 rounding is [`OjasError::NonFinite`] before the device
/// sees it.
fn adamw_scalars(config: AdamWConfig, bc1: f64, bc2: f64) -> Res<[f32; 7]> {
    let s = [
        (1.0 - config.lr * config.weight_decay) as f32,
        (1.0 - config.beta1) as f32,
        config.beta2 as f32,
        (1.0 - config.beta2) as f32,
        (config.lr / bc1) as f32,
        bc2.sqrt() as f32,
        config.eps as f32,
    ];
    if s.iter().any(|v| !v.is_finite()) {
        return Err(OjasError::NonFinite { op: "adamw_step" });
    }
    Ok(s)
}

/// Row-major element strides of `shape`. The input has already passed
/// [`MetalBackend::arg`], so the product fits u32 and cannot overflow.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn u32_dim(op: &'static str, n: usize) -> Res<u32> {
    u32::try_from(n).map_err(|_| OjasError::Unsupported {
        op,
        detail: format!("dimension {n} exceeds the kernels' 32-bit indexing"),
    })
}

/// `accumulate_grad` has no validator of its own (docs/shape-contract.md):
/// it is [`residual_add_forward_dims`] reported under its own name.
fn accumulate_grad_dims(acc: &Tensor, grad: &Tensor) -> Res<usize> {
    const OP: &str = "accumulate_grad";
    residual_add_forward_dims(acc, grad).map_err(|err| match err {
        OjasError::Shape { detail, .. } => OjasError::Shape { op: OP, detail },
        OjasError::Dtype { expected, got, .. } => OjasError::Dtype {
            op: OP,
            expected,
            got,
        },
        OjasError::OutOfRange { detail, .. } => OjasError::OutOfRange { op: OP, detail },
        other => other,
    })
}

/// The rope kernel's `(rows, dim, mode)` of validated `dims`.
fn rope_mode(op: &'static str, dims: &RopeDims) -> Res<(u32, u32, RopeMode)> {
    let mode = match dims.layout {
        RopeLayout::Same => RopeMode::Same,
        RopeLayout::TimeDim { time, heads } => RopeMode::TimeDim {
            time: u32_dim(op, time)?,
            heads: u32_dim(op, heads)?,
        },
    };
    Ok((u32_dim(op, dims.rows)?, u32_dim(op, dims.dim)?, mode))
}

/// The attention kernels' `(bh, t, d)` of validated `dims`, with Metal's
/// head-dim cap: a device limit, so after the validator (D9).
fn sdpa_bhtd(op: &'static str, dims: &SdpaDims) -> Res<(u32, u32, u32)> {
    let d = u32_dim(op, dims.head_dim)?;
    refuse_unsupported_metal_head_dim(BackendId::Metal, d)?;
    let bh = product(op, &[dims.batch, dims.heads])?;
    Ok((u32_dim(op, bh)?, u32_dim(op, dims.seq)?, d))
}

/// The gate kernels' `(rows, din, heads, dh)` of validated `dims`.
fn gate_params(op: &'static str, g: &GateDims) -> Res<(u32, u32, u32, u32)> {
    Ok((
        u32_dim(op, g.rows)?,
        u32_dim(op, g.d_model)?,
        u32_dim(op, g.heads)?,
        u32_dim(op, g.head_dim)?,
    ))
}

impl Backend for MetalBackend {
    fn id(&self) -> BackendId {
        BackendId::Metal
    }

    fn budget(&self) -> &Budget {
        &self.budget
    }

    fn numerics(&self) -> Numerics {
        Numerics::Fast
    }

    /// Wait for every op recorded on this backend (by any clone, on any
    /// thread), then report the first non-finite value found since the last
    /// report as [`OjasError::NonFinite`] naming its op, once.
    fn sync(&self) -> Result<(), OjasError> {
        match self.link.call(Cmd::Sync)? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("sync: device returned {other:?}"))),
        }
    }

    /// A sync point: the read is charged to this backend's budget, then a
    /// pending fault is reported and the host copy dropped. The read's wait
    /// leaves nothing for the report to wait for.
    fn download(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        let host = tensor.to_host(self.budget())?;
        self.sync()?;
        Ok(host)
    }

    /// Host bytes are copied to a new device buffer. A tensor already on this
    /// device is a shared clone; another device's tensor is
    /// [`OjasError::Placement`].
    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "MetalBackend::upload";
        match tensor.device_buffer() {
            Some(buf) => {
                let ours = buf
                    .as_any()
                    .downcast_ref::<MetalBuffer>()
                    .is_some_and(|mb| Arc::ptr_eq(&mb.link, &self.link));
                if ours {
                    Ok(tensor.clone())
                } else {
                    Err(OjasError::Placement {
                        op: OP,
                        expected: Some(BackendId::Metal),
                        found: Some(buf.backend()),
                    })
                }
            }
            None => {
                let bytes = tensor.to_ne_bytes()?;
                if bytes.is_empty() {
                    return Err(shape(OP, "empty tensor"));
                }
                let reservation = self.budget.try_reserve(bytes.len() as u64)?;
                let ids = match tensor.dtype() {
                    DType::U32 => Some(Arc::<[u32]>::from(tensor.u32_slice()?)),
                    _ => None,
                };
                let nb = match self.link.call(Cmd::Upload { bytes })? {
                    Reply::Bufs(mut v) if v.len() == 1 => v.remove(0),
                    other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
                };
                let buf: Arc<dyn DeviceBuffer> = Arc::new(MetalBuffer {
                    id: nb.id,
                    len: nb.bytes,
                    link: Arc::clone(&self.link),
                    ids,
                });
                Tensor::from_device_reserved(buf, tensor.shape(), tensor.dtype(), reservation)
            }
        }
    }

    /// A bit-exact device copy. As on the CPU reference, a NaN or infinity
    /// in the input is [`OjasError::NonFinite`], at the next sync point as
    /// for every Metal op. Unlike the CPU, an input
    /// with a zero-length axis is [`OjasError::Shape`], as for every Metal
    /// op: a Metal tensor is never empty.
    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        const OP: &str = "permute";
        let out = permute_output_shape(OP, input.shape(), dims)?;
        let x = self.f32(OP, input)?;
        let in_strides = row_major_strides(input.shape());
        let mut oshape = [1u32; MAX_PERMUTE_RANK];
        let mut istride = [0u32; MAX_PERMUTE_RANK];
        for (a, &axis) in dims.iter().enumerate() {
            oshape[a] = u32_dim(OP, out[a])?;
            istride[a] = u32_dim(OP, in_strides[axis])?;
        }
        self.one(
            OP,
            &out,
            0,
            Cmd::Permute {
                x,
                rank: u32_dim(OP, dims.len())?,
                oshape,
                istride,
            },
        )
    }

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_forward";
        let dims = embedding_forward_dims(table, token_ids)?;
        let t = self.f32(OP, table)?;
        self.arg(OP, token_ids, DType::U32)?;
        let (vocab, dim) = (u32_dim(OP, dims.vocab)?, u32_dim(OP, dims.dim)?);
        u32_dim(OP, product(OP, &dims.out_shape)?)?;
        let (ids, _) = self.ids(OP, token_ids, vocab, None)?;
        self.one(
            OP,
            &dims.out_shape,
            0,
            Cmd::Embed {
                table: t,
                ids,
                vocab,
                dim,
            },
        )
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        let dims = embedding_backward_dims(table, token_ids, grad_output)?;
        let t = self.f32(OP, table)?;
        self.arg(OP, token_ids, DType::U32)?;
        let g = self.f32(OP, grad_output)?;
        let (vocab, dim) = (u32_dim(OP, dims.vocab)?, u32_dim(OP, dims.dim)?);
        let (ids, _) = self.ids(OP, token_ids, vocab, None)?;
        let scratch = 2 * vocab as usize + ids.n;
        self.one(
            OP,
            table.shape(),
            scratch,
            Cmd::EmbedBwd {
                table: t,
                ids,
                grad: g,
                vocab,
                dim,
            },
        )
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let dims = linear_forward_dims(input, weight)?;
        let x = self.f32(OP, input)?;
        let w = self.f32(OP, weight)?;
        u32_dim(OP, product(OP, &dims.out_shape)?)?;
        self.one(
            OP,
            &dims.out_shape,
            0,
            Cmd::Linear {
                x,
                w,
                rows: dims.rows,
                kin: dims.in_features,
                nout: dims.out_features,
            },
        )
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let dims = linear_backward_dims(input, weight, grad_output)?;
        let x = self.f32(OP, input)?;
        let w = self.f32(OP, weight)?;
        let gy = self.f32(OP, grad_output)?;
        self.two(
            OP,
            input.shape(),
            weight.shape(),
            Cmd::LinearBwd {
                x,
                w,
                gy,
                rows: dims.rows,
                kin: dims.in_features,
                nout: dims.out_features,
            },
        )
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rms_norm_forward";
        let dims = rms_norm_forward_dims(input, weight, eps)?;
        let side = self.rms_side(OP, dims, input, weight, None)?;
        self.one(
            OP,
            input.shape(),
            0,
            Cmd::Rms {
                sides: vec![side],
                eps,
            },
        )
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let dims = rms_norm_backward_dims(input, weight, grad_output, eps)?;
        let side = self.rms_side(OP, dims, input, weight, Some(grad_output))?;
        let mut v = self
            .outputs(
                OP,
                &[input.shape(), weight.shape()],
                rms_bwd_scratch(&side),
                Cmd::Rms {
                    sides: vec![side],
                    eps,
                },
            )?
            .into_iter();
        match (v.next(), v.next()) {
            (Some(a), Some(b)) => Ok((a, b)),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_forward";
        let dims = rope_half_split_forward_dims(x, cos, sin)?;
        rope(self, OP, &dims, x, cos, sin, false)
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_backward";
        let dims = rope_half_split_backward_dims(grad_output, cos, sin)?;
        rope(self, OP, &dims, grad_output, cos, sin, true)
    }

    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_forward";
        if let Some(out) = self.rms_pair(OP, [q, q_weight], [k, k_weight], None, eps)? {
            let mut out = out.into_iter();
            if let (Some(qn), Some(kn)) = (out.next(), out.next()) {
                return Ok((qn, kn));
            }
            return Err(metal_err(format!("{OP}: missing outputs")));
        }
        let qn = self.rms_norm_forward(q, q_weight, eps)?;
        let kn = self.rms_norm_forward(k, k_weight, eps)?;
        Ok((qn, kn))
    }

    fn rms_qk_norm_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        grad_q: &Tensor,
        grad_k: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let grads = Some([grad_q, grad_k]);
        if let Some(out) = self.rms_pair(OP, [q, q_weight], [k, k_weight], grads, eps)? {
            let mut out = out.into_iter();
            if let (Some(gq), Some(gqw), Some(gk), Some(gkw)) =
                (out.next(), out.next(), out.next(), out.next())
            {
                return Ok((gq, gk, gqw, gkw));
            }
            return Err(metal_err(format!("{OP}: missing outputs")));
        }
        let (gq, gqw) = self.rms_norm_backward(q, q_weight, grad_q, eps)?;
        let (gk, gkw) = self.rms_norm_backward(k, k_weight, grad_k, eps)?;
        Ok((gq, gk, gqw, gkw))
    }

    fn causal_sdpa_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "causal_sdpa_forward";
        let dims = causal_sdpa_forward_dims(q, k, v)?;
        let (qa, ka, va) = (self.f32(OP, q)?, self.f32(OP, k)?, self.f32(OP, v)?);
        let (bh, t, d) = sdpa_bhtd(OP, &dims)?;
        self.one(
            OP,
            q.shape(),
            0,
            Cmd::Sdpa {
                q: qa,
                k: ka,
                v: va,
                bh,
                t,
                d,
            },
        )
    }

    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_backward";
        let dims = causal_sdpa_backward_dims(q, k, v, grad_output)?;
        let (qa, ka, va) = (self.f32(OP, q)?, self.f32(OP, k)?, self.f32(OP, v)?);
        let gy = self.f32(OP, grad_output)?;
        let (bh, t, d) = sdpa_bhtd(OP, &dims)?;
        let s = q.shape();
        let mut out = self
            .outputs(
                OP,
                &[s, s, s],
                2 * bh as usize * t as usize,
                Cmd::SdpaBwd {
                    q: qa,
                    k: ka,
                    v: va,
                    gy,
                    bh,
                    t,
                    d,
                },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(a), Some(b), Some(c)) => Ok((a, b, c)),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_forward";
        let dims = per_head_sigmoid_gate_forward_dims(input, weight, bias, attn_out)?;
        let (x, w, b, a) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, attn_out)?,
        );
        let (rows, din, heads, dh) = gate_params(OP, &dims)?;
        self.one(
            OP,
            attn_out.shape(),
            rows as usize * heads as usize,
            Cmd::Gate {
                x,
                w,
                b,
                attn: a,
                rows,
                din,
                heads,
                dh,
            },
        )
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_backward";
        let dims =
            per_head_sigmoid_gate_backward_dims(input, weight, bias, attn_out, grad_output)?;
        let (x, w, b, a) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, attn_out)?,
        );
        let gy = self.f32(OP, grad_output)?;
        let (rows, din, heads, dh) = gate_params(OP, &dims)?;
        let mut out = self
            .outputs(
                OP,
                &[
                    input.shape(),
                    weight.shape(),
                    bias.shape(),
                    attn_out.shape(),
                ],
                2 * rows as usize * heads as usize,
                Cmd::GateBwd {
                    x,
                    w,
                    b,
                    attn: a,
                    gy,
                    rows,
                    din,
                    heads,
                    dh,
                },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next(), out.next()) {
            (Some(input), Some(weight), Some(bias), Some(attn_out)) => Ok(PerHeadGateGrad {
                input,
                weight,
                bias,
                attn_out,
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "value_residual_blend_forward";
        value_residual_blend_forward_dims(value, value0, lambda)?;
        let (v, v0, lam) = (
            self.f32(OP, value)?,
            self.f32(OP, value0)?,
            self.f32(OP, lambda)?,
        );
        self.one(OP, value.shape(), 0, Cmd::Vres { v, v0, lam })
    }

    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        const OP: &str = "value_residual_blend_backward";
        value_residual_blend_backward_dims(value, value0, lambda, grad_output)?;
        let (v, v0, lam) = (
            self.f32(OP, value)?,
            self.f32(OP, value0)?,
            self.f32(OP, lambda)?,
        );
        let gy = self.f32(OP, grad_output)?;
        let mut out = self
            .outputs(
                OP,
                &[value.shape(), value0.shape(), lambda.shape()],
                v.n + 2048,
                Cmd::VresBwd { v, v0, lam, gy },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(value), Some(value0), Some(lambda)) => Ok(ValueResidualGrad {
                value,
                value0,
                lambda,
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_forward";
        silu_forward_dims(input)?;
        let x = self.f32(OP, input)?;
        self.one(OP, input.shape(), 0, Cmd::Silu { x })
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        silu_backward_dims(input, grad_output)?;
        let x = self.f32(OP, input)?;
        let gy = self.f32(OP, grad_output)?;
        self.one(OP, input.shape(), 0, Cmd::SiluBwd { x, gy })
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        mul_forward_dims(a, b)?;
        let (aa, ba) = (self.f32(OP, a)?, self.f32(OP, b)?);
        self.one(OP, a.shape(), 0, Cmd::Mul { a: aa, b: ba })
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        mul_backward_dims(a, b, grad_output)?;
        let (aa, ba) = (self.f32(OP, a)?, self.f32(OP, b)?);
        let gy = self.f32(OP, grad_output)?;
        self.two(OP, a.shape(), b.shape(), Cmd::MulBwd { a: aa, b: ba, gy })
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        residual_add_forward_dims(x, y)?;
        let (xa, ya) = (self.f32(OP, x)?, self.f32(OP, y)?);
        self.one(OP, x.shape(), 0, Cmd::Add { x: xa, y: ya })
    }

    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "residual_add_backward";
        residual_add_backward_dims(x, y, grad_output)?;
        let (xa, ya) = (self.f32(OP, x)?, self.f32(OP, y)?);
        let gy = self.f32(OP, grad_output)?;
        self.two(OP, x.shape(), y.shape(), Cmd::AddBwd { x: xa, y: ya, gy })
    }

    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_forward";
        ce(self, OP, logits, targets, ignore_index, false)
    }

    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_backward";
        ce(self, OP, logits, targets, ignore_index, true)
    }

    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        const OP: &str = "clip_grad_norm";
        clip_grad_norm_dims(grads)?;
        let args = grads
            .iter()
            .map(|g| self.f32(OP, g))
            .collect::<Res<Vec<_>>>()?;
        let _stats = self.reserve(OP, 2 * args.len() + 2048)?;
        let norm = match self.link.call(Cmd::ClipNorm {
            grads: args.clone(),
        })? {
            Reply::Norm(n) => n,
            other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
        };
        if !norm.is_finite() {
            return Err(OjasError::NonFinite { op: OP });
        }
        let scale = clip_scale(max_norm, norm)?;
        if scale < 1.0 {
            for g in grads.iter_mut() {
                self.unique(OP, g)?;
            }
            match self.link.call(Cmd::Scale { grads: args, scale })? {
                Reply::Done => {}
                other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
            }
        }
        Ok(norm)
    }

    /// One command: tessl GEMMs make each `[rows, cols]` logits tile, and
    /// this crate's kernels turn tiles into the loss and gradients (see
    /// `Worker::linear_ce`). The budget is charged for the outputs and for
    /// [`LceGeom::scratch`], whose only logits buffer is one tile, so a call
    /// runs under a budget far below `N * V * 4` bytes.
    ///
    /// An out-of-range target is [`OjasError::OutOfRange`] and an
    /// all-ignored batch [`OjasError::NonFinite`] at the call, from the
    /// host's copy of the targets. A non-finite input, logit, gradient or
    /// loss is found on the device and reported at the next sync point.
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
        let dims = linear_ce_dims(input, weight, targets, chunk)?;
        let x = self.f32(OP, input)?;
        let w = self.f32(OP, weight)?;
        let (t, valid) = self.ids(OP, targets, u32_dim(OP, dims.vocab)?, ignore_index)?;
        if valid == 0 {
            return Err(OjasError::NonFinite { op: OP });
        }
        let geom = LceGeom::new(
            dims.rows,
            dims.model_dim,
            dims.vocab,
            chunk.rows,
            chunk.cols,
            x.off,
            w.off,
            want_grad,
        );
        let scratch = geom.scratch().ok_or_else(|| overflow(OP))?;
        let cmd = Cmd::LinearCe {
            x,
            w,
            t,
            geom,
            ignore: ignore_index,
            valid,
        };
        if !want_grad {
            let loss = self.one(OP, &[], scratch, cmd)?;
            return Ok(LinearCe {
                loss,
                grad_input: None,
                grad_weight: None,
            });
        }
        let mut out = self
            .outputs(OP, &[&[], input.shape(), weight.shape()], scratch, cmd)?
            .into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(loss), Some(gx), Some(gw)) => Ok(LinearCe {
                loss,
                grad_input: Some(gx),
                grad_weight: Some(gw),
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    /// Validated by [`cached_attention_dims`] and the Metal head-dim limit
    /// before anything reaches the device. A NaN or infinity in `q`, or in
    /// either cache at a position below `kv_len`, is [`OjasError::NonFinite`]
    /// at the next sync point; positions at or past `kv_len` are not read.
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cached_attention_forward";
        let dims = cached_attention_dims(q, k_cache, v_cache, kv_len)?;
        let d = u32_dim(OP, dims.head_dim)?;
        refuse_unsupported_metal_head_dim(BackendId::Metal, d)?;
        let (qa, ka, va) = (self.f32(OP, q)?, self.f32(OP, k_cache)?, self.f32(OP, v_cache)?);
        u32_dim(OP, product(OP, &[dims.new, dims.heads])?)?;
        self.one(
            OP,
            q.shape(),
            0,
            Cmd::CachedAttn {
                q: qa,
                k: ka,
                v: va,
                batch: u32_dim(OP, dims.batch)?,
                tq: u32_dim(OP, dims.new)?,
                heads: u32_dim(OP, dims.heads)?,
                kv_heads: u32_dim(OP, dims.kv_heads)?,
                d,
                cap: u32_dim(OP, dims.capacity)?,
                kv_len: u32_dim(OP, kv_len)?,
            },
        )
    }

    /// Validated by [`kv_cache_write_dims`] (so `at + Tn > Tcap` is refused
    /// before any dispatch), then written in place. A shared `cache` is
    /// [`OjasError::Shape`]; a non-finite `src` writes nothing and is
    /// [`OjasError::NonFinite`] at the next sync point.
    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        const OP: &str = "kv_cache_write";
        let dims = kv_cache_write_dims(cache, src, at)?;
        let (c, s) = (self.f32(OP, cache)?, self.f32(OP, src)?);
        self.unique(OP, cache)?;
        match self.link.call(Cmd::KvWrite {
            cache: c,
            src: s,
            batch: u32_dim(OP, dims.batch)?,
            tn: u32_dim(OP, dims.new)?,
            cap: u32_dim(OP, dims.capacity)?,
            row: u32_dim(OP, product(OP, &[dims.kv_heads, dims.head_dim])?)?,
            at: u32_dim(OP, at)?,
        })? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
        }
    }

    /// In place, with no new memory, when `acc` solely owns its buffer;
    /// otherwise the sum goes to a new buffer that replaces `acc`, and the
    /// other handles keep the old values. Either way the device decides
    /// whether every input and sum is finite before it writes anything: a
    /// non-finite one leaves `acc`'s values unchanged (a shared `acc` still
    /// gets its new buffer, holding them) and is [`OjasError::NonFinite`] at
    /// the next sync point.
    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        const OP: &str = "accumulate_grad";
        accumulate_grad_dims(acc, grad)?;
        let a = self.f32(OP, acc)?;
        let g = self.f32(OP, grad)?;
        if self.unique(OP, acc).is_ok() {
            return match self.link.call(Cmd::Accumulate {
                acc: a,
                g,
                in_place: true,
            })? {
                Reply::Done => Ok(()),
                other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
            };
        }
        let sum = self.one(
            OP,
            acc.shape(),
            0,
            Cmd::Accumulate {
                acc: a,
                g,
                in_place: false,
            },
        )?;
        *acc = sum;
        Ok(())
    }

    fn adamw_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        moment1: &mut Tensor,
        moment2: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError> {
        const OP: &str = "adamw_step";
        adamw_step_dims(param, grad, moment1, moment2)?;
        let (p, g, m, v) = (
            self.f32(OP, param)?,
            self.f32(OP, grad)?,
            self.f32(OP, moment1)?,
            self.f32(OP, moment2)?,
        );
        let (_, bc1, bc2) = check_adamw(config, step)?;
        let scalars = adamw_scalars(config, bc1, bc2)?;
        self.unique(OP, param)?;
        self.unique(OP, moment1)?;
        self.unique(OP, moment2)?;
        // In place with no scratch: the device checks the whole step, then
        // writes only if it is finite (see `ojas_adamw_check`).
        match self.link.call(Cmd::AdamW {
            p,
            g,
            m,
            v,
            scalars,
        })? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
        }
    }

    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        const OP: &str = "muon_ns5_step";
        let dims = muon_ns5_step_dims(param, grad, momentum)?;
        let (p, g, m) = (
            self.f32(OP, param)?,
            self.f32(OP, grad)?,
            self.f32(OP, momentum)?,
        );
        ojas_core::require_ns5(5)?;
        if !config.lr.is_finite()
            || !config.momentum.is_finite()
            || !config.weight_decay.is_finite()
        {
            return Err(OjasError::NonFinite { op: OP });
        }
        if config.lr < 0.0 || config.momentum < 0.0 || config.weight_decay < 0.0 {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "muon lr, momentum and weight_decay must be >= 0".to_string(),
            });
        }
        let (rows, cols) = (dims.rows, dims.cols);
        let r = rows.min(cols);
        let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
        let alpha = (-config.lr * scale) as f32;
        let decay = if config.weight_decay == 0.0 {
            1.0
        } else {
            (1.0 - config.lr * config.weight_decay) as f32
        };
        self.unique(OP, param)?;
        self.unique(OP, momentum)?;
        let scratch = 6 * p.n + 3 * r * r + 2048;
        let _scratch = self.reserve(OP, scratch)?;
        match self.link.call(Cmd::Muon {
            p,
            g,
            m,
            rows,
            cols,
            momentum: config.momentum as f32,
            nesterov: config.nesterov,
            decay,
            alpha,
        })? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
        }
    }
}

/// Rope of validated `dims`: the operands placed, then the 32-bit caps.
fn rope(
    be: &MetalBackend,
    op: &'static str,
    dims: &RopeDims,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    backward: bool,
) -> Res<Tensor> {
    let (xa, ca, sa) = (be.f32(op, x)?, be.f32(op, cos)?, be.f32(op, sin)?);
    let (rows, dim, mode) = rope_mode(op, dims)?;
    be.one(
        op,
        x.shape(),
        0,
        Cmd::Rope {
            x: xa,
            cos: ca,
            sin: sa,
            rows,
            dim,
            mode,
            backward,
        },
    )
}

fn ce(
    be: &MetalBackend,
    op: &'static str,
    logits: &Tensor,
    targets: &Tensor,
    ignore: Option<u32>,
    grad: bool,
) -> Res<Tensor> {
    let dims = if grad {
        cross_entropy_mean_backward_dims(logits, targets)?
    } else {
        cross_entropy_mean_forward_dims(logits, targets)?
    };
    let la = be.f32(op, logits)?;
    be.arg(op, targets, DType::U32)?;
    let (rows, vocab) = (u32_dim(op, dims.rows)?, u32_dim(op, dims.vocab)?);
    let (ta, valid) = be.ids(op, targets, vocab, ignore)?;
    if valid == 0 {
        return Err(OjasError::NonFinite { op });
    }
    let out: &[usize] = if grad { logits.shape() } else { &[] };
    be.one(
        op,
        out,
        rows as usize + 2048,
        Cmd::Ce {
            logits: la,
            targets: ta,
            rows,
            vocab,
            ignore,
            valid,
            grad,
        },
    )
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    /// Run `f` on its own thread; panic if it neither returns nor panics in
    /// time, so a hang is a failure rather than a stuck test run.
    fn watchdog<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = mpsc::channel();
        let h = thread::spawn(move || {
            let out = f();
            let _ = tx.send(());
            out
        });
        match rx.recv_timeout(Duration::from_secs(secs)) {
            Ok(()) => h.join().expect("worker panicked"),
            Err(mpsc::RecvTimeoutError::Disconnected) => match h.join() {
                Ok(_) => unreachable!("sender dropped without sending"),
                Err(p) => std::panic::resume_unwind(p),
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("watchdog: no result after {secs} s; a call hung")
            }
        }
    }

    /// A panic on the device thread in the middle of a stream of ops from
    /// several threads poisons the backend: the panicking call and every
    /// later call fail promptly with an error (none hangs, none returns a
    /// result), and dropping the tensors and the backend completes.
    #[test]
    fn a_device_thread_panic_mid_stream_fails_later_calls_cleanly() {
        watchdog(120, || {
            let host = Budget::new(1 << 30);
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let vals: Vec<f32> = (0..64 * 64).map(|i| (i % 17) as f32 / 17.0).collect();
            let x = m
                .upload(&Tensor::from_f32(&vals, &[64, 64], &host).expect("host"))
                .expect("upload");
            let done = Arc::new(AtomicUsize::new(0));
            let workers: Vec<_> = (0..6)
                .map(|i| {
                    let (m, x, done) = (m.clone(), x.clone(), Arc::clone(&done));
                    thread::spawn(move || {
                        let mut ok_before = 0usize;
                        loop {
                            let r = if i % 2 == 0 {
                                m.linear_forward(&x, &x).map(drop)
                            } else {
                                m.rms_norm_forward(&x, &x.view(&[64], &[1], 0).expect("w"), 1e-6)
                                    .map(drop)
                            };
                            match r {
                                Ok(()) => {
                                    ok_before += 1;
                                    done.fetch_add(1, Ordering::AcqRel);
                                }
                                Err(e) => return (ok_before, e),
                            }
                        }
                    })
                })
                .collect();
            while done.load(Ordering::Acquire) < 60 {
                thread::yield_now();
            }
            let injected = m.link.call(Cmd::InjectPanic);
            assert!(
                matches!(&injected, Err(OjasError::Backend { detail, .. }) if detail.contains("panicked")),
                "{injected:?}"
            );
            for (i, w) in workers.into_iter().enumerate() {
                let (ok_before, err) = w.join().expect("worker thread");
                assert!(ok_before > 0, "worker {i} never ran before the panic");
                assert!(matches!(err, OjasError::Poisoned), "worker {i}: {err:?}");
            }
            assert!(matches!(m.silu_forward(&x), Err(OjasError::Poisoned)));
            assert!(matches!(m.download(&x), Err(OjasError::Poisoned)));
            let fresh = Tensor::from_f32(&[1.0], &[1], &host).expect("host");
            assert!(matches!(m.upload(&fresh), Err(OjasError::Poisoned)));
            drop(x);
            drop(m);
        });
    }

    /// QK-norm takes the one-command path when both sides validate and fit,
    /// refuses a k that does not validate (the validator runs on both pairs
    /// first, so nothing is recorded), and declines (so the composition
    /// runs) only when the two sides' outputs and scratch do not fit beside
    /// each other.
    #[test]
    fn qk_norm_fuses_only_when_the_composition_could_not_differ() {
        let host = Budget::new(1 << 24);
        let t = |m: &MetalBackend, n: usize, shape: &[usize]| {
            m.upload(&Tensor::from_f32(&vec![0.25; n], shape, &host).expect("host"))
                .expect("upload")
        };
        let m = MetalBackend::new(Budget::new(1 << 24)).expect("Metal device");
        let (q, k, w) = (t(&m, 64 * 16, &[64, 16]), t(&m, 64 * 16, &[64, 16]), t(&m, 16, &[16]));
        let fused = m.rms_pair("rms_norm_forward", [&q, &w], [&k, &w], None, 1e-6);
        assert_eq!(fused.expect("fwd").map(|v| v.len()), Some(2));
        let fused = m.rms_pair("rms_norm_backward", [&q, &w], [&k, &w], Some([&q, &k]), 1e-6);
        assert_eq!(fused.expect("bwd").map(|v| v.len()), Some(4));
        let bad_w = t(&m, 17, &[17]);
        let refused = m.rms_pair("rms_norm_forward", [&q, &w], [&k, &bad_w], None, 1e-6);
        assert!(
            matches!(&refused, Err(OjasError::Shape { op: "rms_norm_forward", .. })),
            "{refused:?}"
        );
        let host_k = Tensor::from_f32(&vec![0.25; 64 * 16], &[64, 16], &host).expect("host");
        let refused = m.rms_pair("rms_norm_forward", [&q, &w], [&host_k, &w], None, 1e-6);
        assert!(matches!(&refused, Err(OjasError::Placement { .. })), "{refused:?}");
        assert_eq!(pending(&m), None);
        // Inputs, then room for q's output and k's but not both sides' scratch.
        let inputs = 4 * (3 * 64 * 16 + 16) as u64;
        let outs = 2 * 4 * (64 * 16 + 16) as u64;
        let scratch = 4 * (64 + 16) as u64;
        let tight = MetalBackend::new(Budget::new(inputs + outs + scratch)).expect("Metal");
        let (q, k, w) = (t(&tight, 64 * 16, &[64, 16]), t(&tight, 64 * 16, &[64, 16]), t(&tight, 16, &[16]));
        let g = t(&tight, 64 * 16, &[64, 16]);
        let declined = tight.rms_pair("rms_norm_backward", [&q, &w], [&k, &w], Some([&g, &g]), 1e-6);
        assert!(matches!(declined, Ok(None)), "{declined:?}");
        assert!(tight.rms_qk_norm_backward(&q, &k, &w, &w, &g, &g, 1e-6).is_ok());
    }

    fn put(m: &MetalBackend, vals: &[f32], shape: &[usize]) -> Tensor {
        let host = Tensor::from_f32(vals, shape, &Budget::new(1 << 30)).expect("host");
        m.upload(&host).expect("upload")
    }

    fn tune(
        m: &MetalBackend,
        overlap: Option<usize>,
        mem_cap: Option<u64>,
        ws_limit: Option<u64>,
        fail_allocs: (u32, u32),
    ) {
        let r = m.link.call(Cmd::Tune {
            overlap,
            mem_cap,
            ws_limit,
            fail_allocs,
        });
        assert!(matches!(r, Ok(Reply::Done)), "{r:?}");
    }

    /// What `sync` reports: `None` for `Ok`, the op for `NonFinite`.
    fn pending(m: &MetalBackend) -> Option<&'static str> {
        match m.sync() {
            Ok(()) => None,
            Err(OjasError::NonFinite { op }) => Some(op),
            Err(e) => panic!("sync: {e:?}"),
        }
    }

    fn adam_fault(m: &MetalBackend) -> (Tensor, Tensor, Tensor) {
        let mut g: Vec<f32> = (0..24).map(|i| i as f32 / 24.0).collect();
        g[23] = f32::NAN;
        let g = put(m, &g, &[4, 6]);
        let mut p = put(m, &[0.5; 24], &[4, 6]);
        let mut m1 = put(m, &[0.0; 24], &[4, 6]);
        let mut m2 = put(m, &[0.0; 24], &[4, 6]);
        m.adamw_step(&mut p, &g, &mut m1, &mut m2, 1, AdamWConfig::nanolab(1e-3, 0.1))
            .expect("adamw records");
        (p, m1, m2)
    }

    /// §7.4, F12: a read that fails after its wait keeps the fault for the
    /// next sync point.
    #[test]
    fn a_fault_survives_a_failed_read() {
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let (p, _m1, _m2) = adam_fault(&m);
        assert!(matches!(m.link.call(Cmd::FailNextRead), Ok(Reply::Done)));
        let r = p.to_host(&Budget::new(1 << 20));
        assert!(
            matches!(&r, Err(OjasError::Backend { detail, .. }) if detail.contains("injected read failure")),
            "{r:?}"
        );
        assert_eq!(pending(&m), Some("adamw_step"));
        assert_eq!(pending(&m), None);
        assert!(p.to_host(&Budget::new(1 << 20)).expect("read").to_f32_vec().expect("f32").iter().all(|&v| v == 0.5));
    }

    /// §7.5: memory-cap commits wait and scan but hold the fault; no call
    /// between the fault and the sync returns an error.
    #[test]
    fn a_fault_survives_memory_cap_commits() {
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let nan = put(&m, &[f32::NAN, 1.0], &[2]);
        let one = put(&m, &[0.5, 0.25], &[2]);
        tune(&m, None, Some(1), None, (0, 0));
        let w0 = m.waits();
        let _y = m.silu_forward(&nan).expect("silu records");
        for _ in 0..10 {
            m.silu_forward(&one).expect("clean op past a cap commit");
        }
        assert!(m.waits() - w0 >= 11, "every op should have hit the 1-byte cap: {}", m.waits() - w0);
        assert_eq!(pending(&m), Some("silu_forward"));
        assert_eq!(pending(&m), None);
    }

    /// The working-set trigger: with the device above its threshold, a
    /// backend waits once per 64 MiB it allocates, not on every op.
    #[test]
    fn the_working_set_trigger_waits_on_growth_not_on_every_op() {
        let m = MetalBackend::new(Budget::new(4 << 30)).expect("Metal device");
        let one = put(&m, &[0.5], &[1]);
        m.sync().expect("settle");
        tune(&m, None, Some(u64::MAX), Some(1), (0, 0));
        let w0 = m.waits();
        for _ in 0..200 {
            drop(m.silu_forward(&one).expect("tiny op"));
        }
        let tiny = m.waits() - w0;
        assert_eq!(tiny, 0, "tiny ops above the threshold waited {tiny} times");
        let big = put(&m, &vec![0.5; 8 << 20], &[8 << 20]);
        let w1 = m.waits();
        let kept: Vec<Tensor> = (0..6).map(|_| m.silu_forward(&big).expect("big op")).collect();
        let grew = m.waits() - w1;
        assert!(
            (1..=3).contains(&grew),
            "192 MiB allocated above the threshold waited {grew} times (one per 64 MiB expected)"
        );
        assert_eq!(pending(&m), None);
        drop(kept);
    }

    /// An allocation that fails while work is recorded is retried after a
    /// waited commit, and that commit keeps the status of the op being
    /// recorded: its fault still reaches the next sync.
    #[test]
    fn an_allocation_retry_keeps_the_current_ops_status() {
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let x = put(&m, &[1.0, f32::NAN, 2.0, 3.0], &[2, 2]);
        let w = put(&m, &[1.0, 1.0], &[2]);
        let one = put(&m, &[0.5], &[1]);
        for _ in 0..5 {
            m.silu_forward(&one).expect("clean");
        }
        tune(&m, None, None, None, (0, 1));
        let w0 = m.waits();
        let y = m.rms_norm_forward(&x, &w, 1e-6);
        assert!(y.is_ok(), "the retried allocation should succeed: {y:?}");
        assert!(m.waits() - w0 >= 1, "the retry should follow a waited commit");
        assert_eq!(pending(&m), Some("rms_norm_forward"));
        assert_eq!(pending(&m), None);
    }

    /// A call that fails after its kernels were recorded returns its own
    /// error and leaves nothing pending: its status slot is dropped with
    /// its outputs, as the synchronous path discarded both.
    #[test]
    fn a_call_that_fails_after_recording_leaves_no_fault() {
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let mut gv = vec![0.5f32; 24];
        gv[23] = f32::NAN;
        let g = put(&m, &gv, &[4, 6]);
        let mut p = put(&m, &[0.5; 24], &[4, 6]);
        let mut mo = put(&m, &[0.0; 24], &[4, 6]);
        // Muon checks p, g and m, then allocates: fail that allocation and
        // its retry, after the checks have run.
        tune(&m, None, None, None, (0, 2));
        let r = m.muon_ns5_step(&mut p, &g, &mut mo, MuonNs5Config::nanolab_default());
        assert!(matches!(r, Err(OjasError::CapacityExceeded { .. })), "{r:?}");
        assert_eq!(pending(&m), None);
    }

    /// QK-norm as one command: when k's side fails to encode after q's was
    /// recorded, the call returns k's error and q's fault still reaches the
    /// next sync, as in the composition, where q is an op of its own.
    #[test]
    fn a_fused_qk_norm_keeps_qs_fault_when_ks_side_fails() {
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let mut qv = vec![0.5f32; 64 * 16];
        qv[5] = f32::NAN;
        let q = put(&m, &qv, &[64, 16]);
        let k = put(&m, &[0.25; 64 * 16], &[64, 16]);
        let w = put(&m, &[1.0; 16], &[16]);
        // q's side allocates one output; fail k's output and its retry.
        tune(&m, None, None, None, (1, 2));
        let r = m.rms_qk_norm_forward(&q, &k, &w, &w, 1e-6);
        assert!(matches!(r, Err(OjasError::CapacityExceeded { .. })), "{r:?}");
        assert_eq!(pending(&m), Some("rms_norm_forward"));
        assert_eq!(pending(&m), None);
    }

    /// §7.9: a pending fault, then a device panic: every later call,
    /// `sync` included, is `Poisoned` promptly, and drop completes.
    #[test]
    fn a_pending_fault_then_a_device_panic_is_poisoned_without_hanging() {
        watchdog(60, || {
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let nan = put(&m, &[f32::NAN], &[1]);
            let y = m.silu_forward(&nan).expect("silu records");
            let injected = m.link.call(Cmd::InjectPanic);
            assert!(injected.is_err(), "{injected:?}");
            assert!(matches!(m.sync(), Err(OjasError::Poisoned)));
            assert!(matches!(m.silu_forward(&nan), Err(OjasError::Poisoned)));
            assert!(matches!(m.download(&y), Err(OjasError::Poisoned)));
            assert!(matches!(m.sync(), Err(OjasError::Poisoned)));
            drop((nan, y));
            drop(m);
        });
    }

    /// One thread's chain of tiny ops; returns every 50th output's bits and
    /// the last.
    fn chain(m: &MetalBackend, seed: u32, len: usize, start: &std::sync::Barrier) -> Vec<Vec<u32>> {
        let vals: Vec<f32> = (0..64).map(|i| ((i * 7 + seed as usize * 13) % 29) as f32 / 29.0 - 0.5).collect();
        let x = put(m, &vals, &[64]);
        let c = put(m, &vec![0.75 + seed as f32 / 64.0; 64], &[64]);
        start.wait();
        let mut keep = Vec::new();
        let mut y = x.clone();
        for i in 0..len {
            y = match i % 3 {
                0 => m.silu_forward(&y),
                1 => m.mul_forward(&y, &c),
                _ => m.residual_add_forward(&y, &x),
            }
            .expect("chain op");
            if i % 50 == 49 {
                keep.push(y.clone());
            }
        }
        keep.push(y);
        keep.iter()
            .map(|t| {
                let h = t.to_host(&Budget::new(1 << 20)).expect("read");
                h.to_f32_vec().expect("f32").iter().map(|v| v.to_bits()).collect()
            })
            .collect()
    }

    /// Overlap commits with both command allocators in flight: six threads
    /// record long chains of tiny ops on one backend with no wait between
    /// them, and every thread's bits equal a serial run with overlap off.
    #[test]
    fn overlap_commits_from_many_threads_match_a_serial_run_bit_for_bit() {
        const THREADS: u32 = 6;
        const LEN: usize = 1000;
        watchdog(300, || {
            let serial = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            tune(&serial, Some(usize::MAX), None, None, (0, 0));
            let one = std::sync::Barrier::new(1);
            let want: Vec<_> = (0..THREADS).map(|i| chain(&serial, i, LEN, &one)).collect();
            assert_eq!(pending(&serial), None);
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let start = Arc::new(std::sync::Barrier::new(THREADS as usize));
            let handles: Vec<_> = (0..THREADS)
                .map(|i| {
                    let (m, start) = (m.clone(), Arc::clone(&start));
                    thread::spawn(move || chain(&m, i, LEN, &start))
                })
                .collect();
            for (i, h) in handles.into_iter().enumerate() {
                let got = h.join().expect("chain thread");
                assert!(got == want[i], "thread {i}: bits differ from the serial run");
            }
            assert_eq!(pending(&m), None);
        });
    }

    /// `kv_cache_write`'s range refusal happens on the host, before any
    /// command: on a poisoned backend, where every command that reaches the
    /// device answers `Poisoned`, it is still `OutOfRange`. A write in range
    /// reaches the device and is `Poisoned`, which shows the probe can tell.
    #[test]
    fn a_kv_write_past_capacity_is_refused_before_any_dispatch() {
        watchdog(60, || {
            let host = Budget::new(1 << 20);
            let m = MetalBackend::new(Budget::new(1 << 20)).expect("Metal device");
            let up = |n: usize, shape: &[usize]| {
                m.upload(&Tensor::from_f32(&vec![0.5; n], shape, &host).expect("host"))
                    .expect("upload")
            };
            let mut cache = up(2 * 4 * 3, &[1, 4, 2, 3]);
            let src = up(2 * 2 * 3, &[1, 2, 2, 3]);
            let injected = m.link.call(Cmd::InjectPanic);
            assert!(injected.is_err(), "{injected:?}");
            for at in [3usize, 4, usize::MAX] {
                let r = m.kv_cache_write(&mut cache, &src, at);
                assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "at {at}: {r:?}");
            }
            let r = m.kv_cache_write(&mut cache, &src, 2);
            assert!(matches!(r, Err(OjasError::Poisoned)), "{r:?}");
        });
    }
}
