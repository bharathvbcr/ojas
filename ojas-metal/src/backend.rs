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
//! [`OjasError::CapacityExceeded`] when they do not fit. The budget charges
//! each tensor's logical bytes; the device holds a little more, and the gap
//! is bounded:
//!
//! - Rounding. Every buffer is a tessl `Cold` allocation, made at its pool
//!   bucket (tessl `GpuRuntime::allocated_bytes_for`): up to 1 MiB, the
//!   next power of two, at least 256 bytes, so under 512 KiB (and under the
//!   buffer's own size) uncharged per buffer; past 1 MiB, the next multiple
//!   of 16 KiB, Metal's own allocation granule, so under 16 KiB per buffer.
//!   `cold_rounding_stays_inside_the_documented_bound` pins this.
//! - The pool cache. Freed buffers tessl keeps for reuse are not charged.
//!   The cache is capped when the backend opens, at a quarter of its budget
//!   and at most 1 GiB (tessl's own default was 2 GiB whatever the budget),
//!   reported as [`crate::MetalMemory::pool_cache_cap`] and set aside from
//!   the device's room by `ojas_device::ResourcePlan`. [`MetalBackend::trim_pool`]
//!   releases it, and a failed allocation does so before it gives up.
//! - Buffers dropped and not yet recycled. A dropped tensor's charge comes
//!   off the budget at once, but tessl returns its buffer to the pool only
//!   once a commit after the drop completes (an unwaited commit recycles
//!   what the GPU has finished; a waited one, everything). The device
//!   thread counts the bytes freed since the last waited commit and waits
//!   before a command that would start past the memory cap
//!   ([`WaitCounts::freed`]). An op's temporaries (its charged scratch,
//!   dropped when it returns) count toward the bytes allocated since the
//!   last waited commit, which wait after the command that passes the cap
//!   ([`WaitCounts::mem_cap`]). The memory cap is a quarter of the budget
//!   the backend opened with, at most 1 GiB.
//!
//! So the device holds at most the budget's live bytes, plus the rounding
//! above, plus the pool cap, plus the backend's status slab (128 KiB), plus
//! dropped buffers awaiting a commit: under one memory cap of frees, and
//! under another of temporaries plus the last command's own.
//!
//! Numerics are [`Numerics::Fast`]: GEMMs are tessl TensorOps with their own
//! reduction order, and reductions are threadgroup trees, not ascending f32.
//! A linear GEMM whose two operands are both [`Tensor::bf16_exact`] (what
//! [`ojas_core::Autocast`] hands this backend in a bf16 region) runs on
//! tessl's bf16 lane, the 4-simdgroup cooperative kernels, instead of the
//! exact-f32 1-simdgroup tile, on a device with TensorOps
//! ([`MetalBackend::has_tensorops`]): the products are the same while they
//! stay in f32's normal range, and the f32 sums run in another order.
//! Untagged `F32` and `Bf16` operands keep the exact-f32 lane, so a `Bf16`
//! operand still gives its widened copy's bits.
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
use std::sync::Arc;

use ojas_core::{
    accumulate_grad_dims, adamw_step_dims, argmax_rows_dims, attention_forward_dims,
    attention_forward_saving_dims, broadcast_add_forward_dims, broadcast_mul_forward_dims,
    cached_attention_dims, causal_conv1d_silu_backward_dims, causal_conv1d_silu_forward_dims,
    causal_sdpa_backward_dims, causal_sdpa_forward_dims, channel_layer_norm_forward_dims,
    check_adamw, chunked_gdn_backward_dims, chunked_gdn_forward_dims, clip_grad_norm_dims,
    clip_scale, concat_dims, conv2d_forward_dims, conv_transpose2d_forward_dims,
    cross_entropy_mean_backward_dims, cross_entropy_mean_forward_dims, embedding_backward_dims,
    embedding_forward_dims, fill_dims, gated_rms_norm_backward_dims, gated_rms_norm_forward_dims,
    gdn_log_decay_backward_dims, gdn_log_decay_forward_dims, gelu_erf_forward_dims,
    group_norm_forward_dims, interpolate_bilinear_forward_dims, kv_cache_write_dims,
    layer_norm_forward_dims, linear_backward_dims, linear_bias_forward_dims, linear_ce_dims,
    linear_forward_dims, mul_backward_dims, mul_forward_dims, muon_ns5_step_dims,
    per_head_sigmoid_gate_backward_dims, per_head_sigmoid_gate_forward_dims, permute_dims,
    refuse_unsupported_metal_gdn, refuse_unsupported_metal_head_dim, relu_forward_dims,
    residual_add_backward_dims, residual_add_forward_dims, rms_norm_backward_dims,
    rms_norm_forward_dims, rms_qk_norm_backward_dims, rms_qk_norm_forward_dims,
    rope_half_split_backward_dims, rope_half_split_forward_dims, rope_interleaved_backward_dims,
    rope_interleaved_forward_dims, rope_partial_backward_dims, rope_partial_forward_dims,
    scale_forward_dims, sigmoid_backward_dims, sigmoid_forward_dims, silu_backward_dims,
    silu_forward_dims, slice_dims, softmax_forward_dims, topk_rows_dims,
    value_residual_blend_backward_dims, value_residual_blend_forward_dims, AdamWConfig,
    AttentionDims, AttentionOpts, Backend, BackendId, BroadcastDims, Budget, CeChunk, CeGrads,
    Conv1dDims, Conv2dDims, Conv2dParams, ConvTranspose2dDims, ConvTranspose2dParams, DType,
    DeviceBuffer, GateDims, GatedRmsGrad, GdnDecayGrad, GdnDims, GdnForward, GdnGrad, GdnInputs,
    InterleavedRopeDims, LinearCe, MuonNs5Config, Ns5Precision, Numerics, OjasError, OptimizerKind,
    PartialRopeDims, PerHeadGateGrad, Reservation, RmsDims, RopeDims, RopeLayout, SdpaDims, Tensor,
    ValueResidualGrad, MAX_PERMUTE_RANK,
};

use crate::link::{
    cached_attn_origin, cached_attn_part, metal_err, norm_chunks, rms_w_chunks, Arg, AttnGeom,
    BilinearGeom, Cmd, Conv1dGeom, ConvGeom, EmbedRows, EncAttnGeom, GdnArgs, GroupNormGeom, Lane,
    LceGeom, Link, ReadAs, Reply, Res, RmsSide, RopeMode, SdpaGeom, WaitCounts, Waits,
    SQ_OUT_WORDS,
};

/// A Metal allocation owned by the device thread, named by id.
pub struct MetalBuffer {
    id: u64,
    len: usize,
    link: Arc<Link>,
    /// What the host knows of a U32 buffer's values, so token ids are
    /// range-checked, and cross-entropy's valid rows counted, here rather
    /// than on the device. `None` for an F32 buffer.
    ids: Option<Ids>,
}

/// The host's knowledge of a U32 Metal buffer.
enum Ids {
    /// An upload: the values themselves.
    Host(Arc<[u32]>),
    /// Produced on the device (`argmax_rows`, `topk_rows`): every value is below this
    /// bound, by construction. Enough to range-check an embedding lookup;
    /// an op that needs the values (a valid-row count) refuses it.
    Below(u32),
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
        let waits = Arc::new(Waits::default());
        let (tx, device) = crate::device::spawn(Arc::clone(&waits), budget.cap_bytes())?;
        Ok(Self {
            link: Arc::new(Link::new(tx, device, waits)),
            budget,
        })
    }

    pub fn device_name(&self) -> &str {
        self.link.device_name()
    }

    /// Whether the device runs tessl's cooperative TensorOps kernels. Without
    /// them, bf16-exact operands take the exact-f32 lane like any other.
    pub fn has_tensorops(&self) -> bool {
        self.link.has_tensorops()
    }

    /// The device's recommended working set and what this process holds on
    /// it now. The result is an [`ojas_device::MemoryProbe`] for
    /// [`ojas_device::ResourcePlan::derive`]. Answered on a poisoned backend
    /// too; it records and commits nothing.
    pub fn memory(&self) -> Result<crate::MetalMemory, OjasError> {
        self.read_memory(false)
    }

    /// [`Self::memory`], then [`crate::MetalMemory::peak_allocated`]
    /// restarts from what is allocated now: the reading's peak covers the
    /// span since the backend opened or the last reset.
    pub fn memory_and_reset_peak(&self) -> Result<crate::MetalMemory, OjasError> {
        self.read_memory(true)
    }

    fn read_memory(&self, reset_peak: bool) -> Result<crate::MetalMemory, OjasError> {
        match self.link.call(Cmd::Memory { reset_peak })? {
            Reply::Memory(m) => Ok(m),
            other => Err(metal_err(format!("memory: device returned {other:?}"))),
        }
    }

    /// Release every freed buffer the device keeps cached for reuse, after
    /// a waited commit returns the ones freed since the last. Live tensors
    /// are untouched, and later frees cache as before. The cache is not
    /// charged to the budget (see the module docs); this hands its memory
    /// back, as wgpu's `WgpuContext::trim_pool` does. An allocation that
    /// fails for want of device memory does it on its own before giving up.
    pub fn trim_pool(&self) -> Result<(), OjasError> {
        match self.link.call(Cmd::TrimPool)? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("trim_pool: device returned {other:?}"))),
        }
    }

    /// Declare `tensor` a frozen parameter, once, at load: its values are
    /// scanned for NaN and infinity now (an `F32` or `Bf16` device tensor),
    /// and from then on no matmul, embedding gather or fused cross-entropy
    /// that reads it (or an exact `f32` widening of it) scans it again. A
    /// step over a frozen 2B base otherwise re-reads every weight once per
    /// use for nothing: the weights cannot change.
    ///
    /// The mark is on the device buffer, so every view of the tensor shares
    /// it, and it holds only while nothing writes that buffer: an
    /// `adamw_step`, `muon_ns5_step`, `accumulate_grad`, `scale_grad` or
    /// `kv_cache_write` that targets it is refused with
    /// [`OjasError::Backend`] before it records anything. The scan is
    /// recorded like any other: a NaN or infinity is [`OjasError::NonFinite`]
    /// at the next [`Backend::sync`], and the mark is then withdrawn, so a
    /// buffer that was not finite is scanned by every later op as before.
    /// Freed with its buffer. Freezing a window of a buffer marks only that
    /// window.
    pub fn freeze_checked(&self, tensor: &Tensor) -> Result<(), OjasError> {
        const OP: &str = "freeze_checked";
        let half = match tensor.dtype() {
            DType::F32 => false,
            DType::Bf16 => true,
            got => {
                return Err(OjasError::Dtype {
                    op: OP,
                    expected: DType::F32,
                    got,
                })
            }
        };
        let x = self.arg(OP, tensor, tensor.dtype())?;
        match self.link.call(Cmd::Freeze { x, half })? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
        }
    }

    /// Waited GPU commits this backend's device thread has made since it
    /// opened, shared by every clone. Telemetry for tests and benches, not a
    /// stable API.
    #[doc(hidden)]
    pub fn waits(&self) -> u64 {
        self.link.wait_counts().total()
    }

    /// Compute dispatches this backend's device thread has encoded since it
    /// opened, shared by every clone, counted when each command finishes.
    /// Telemetry for tests and benches, not a stable API.
    #[doc(hidden)]
    pub fn dispatches(&self) -> u64 {
        self.link.dispatches()
    }

    /// [`Self::waits`] by trigger: every waited commit counted once, under
    /// what made it wait (an upload while work was recorded, a read, a
    /// sync, `clip_grad_norm`, the memory cap, the working set, a full
    /// status slab, a recycle after a failed allocation). Telemetry for
    /// tests and benches, not a stable API.
    #[doc(hidden)]
    pub fn wait_counts(&self) -> WaitCounts {
        self.link.wait_counts()
    }

    /// Carry uploads of at most 64 KiB inline, and `U32` uploads of any size
    /// through a host-written fresh buffer, while work is recorded (the
    /// default); or turn both off so every such upload waits as it did
    /// before either existed. For before/after wait measurements; not a
    /// stable API.
    #[doc(hidden)]
    pub fn set_inline_uploads(&self, on: bool) -> Result<(), OjasError> {
        match self.link.call(Cmd::InlineUploads { on })? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!(
                "set_inline_uploads: device returned {other:?}"
            ))),
        }
    }

    /// Another handle on this backend's device that charges `budget`
    /// instead: it accepts this backend's tensors and shares its device
    /// thread, faults and `sync`, as wgpu's `WgpuBackend::with_context`
    /// shares a context. The device's memory cap and pool cache cap stay
    /// the ones set from the opening backend's budget.
    ///
    /// ojas-capi's load uses it on the production path: a session whose
    /// budget the device plan cuts below the caller's is rebound to the
    /// smaller budget (`model::open`). Tests use it to hold operands on one
    /// budget and run an op under another (`tests/shape_first.rs`).
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

    /// The host copy of `t`'s ids window `a` (an upload's values), or
    /// [`OjasError::Unsupported`] for ids produced on the device, whose
    /// values the host does not hold.
    fn host_ids<'t>(&self, op: &'static str, t: &'t Tensor, a: Arg) -> Res<&'t [u32]> {
        let known = t
            .device_buffer()
            .and_then(|b| b.as_any().downcast_ref::<MetalBuffer>())
            .and_then(|mb| mb.ids.as_ref());
        match known {
            Some(Ids::Host(host)) => host
                .get(a.off / 4..a.off / 4 + a.n)
                .ok_or_else(|| metal_err(format!("{op}: id window is outside its buffer"))),
            _ => Err(OjasError::Unsupported {
                op,
                detail: "the gradient's rows are grouped from the ids on the host; these were \
                         produced on the device"
                    .to_string(),
            }),
        }
    }

    fn f32(&self, op: &'static str, t: &Tensor) -> Res<Arg> {
        self.arg(op, t, DType::F32)
    }

    /// An operand of a matmul-class op ([`Backend::bf16_operands`]): a
    /// `Bf16` tensor widened exactly into a new `F32` one charged to the
    /// budget, which lives for the call; any other tensor as it is, for the
    /// op's own checks. The op then runs on `f32` values only, so a `Bf16`
    /// operand gives the bits its widened `F32` copy gives.
    fn widened(&self, t: &Tensor) -> Res<Tensor> {
        match t.dtype() {
            DType::Bf16 => self.to_f32(t),
            _ => Ok(t.clone()),
        }
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
        let known = t
            .device_buffer()
            .and_then(|b| b.as_any().downcast_ref::<MetalBuffer>())
            .and_then(|mb| mb.ids.as_ref())
            .ok_or_else(|| metal_err(format!("{op}: U32 tensor has no host copy of its ids")))?;
        let host = match known {
            Ids::Host(host) => host,
            Ids::Below(bound) => {
                if *bound > limit {
                    return Err(OjasError::OutOfRange {
                        op,
                        detail: format!(
                            "device-produced ids are only known to be below {bound}, past {limit}"
                        ),
                    });
                }
                if ignore.is_some_and(|i| i < *bound) {
                    return Err(OjasError::Unsupported {
                        op,
                        detail: "counting rows equal to the ignore index needs the ids on the \
                                 host; these were produced on the device"
                            .to_string(),
                    });
                }
                return Ok((a, u32_dim(op, a.n)?));
            }
        };
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

    /// `a + b` or `a * b` over a broadcast whose dims the caller validated.
    fn broadcast(
        &self,
        op: &'static str,
        mul: bool,
        a: &Tensor,
        b: &Tensor,
        dims: &BroadcastDims,
    ) -> Res<Tensor> {
        let (x, y) = (self.f32(op, a)?, self.f32(op, b)?);
        let mut oshape = [1u32; MAX_PERMUTE_RANK];
        let mut xstride = [0u32; MAX_PERMUTE_RANK];
        let mut ystride = [0u32; MAX_PERMUTE_RANK];
        for axis in 0..dims.out_shape.len() {
            oshape[axis] = u32_dim(op, dims.out_shape[axis])?;
            xstride[axis] = u32_dim(op, dims.a_strides[axis])?;
            ystride[axis] = u32_dim(op, dims.b_strides[axis])?;
        }
        u32_dim(op, product(op, &dims.out_shape)?)?;
        let cmd = Cmd::Bcast {
            x,
            y,
            mul,
            rank: u32_dim(op, dims.out_shape.len())?,
            oshape,
            xstride,
            ystride,
        };
        self.one(op, &dims.out_shape, 0, cmd)
    }

    fn two(&self, op: &'static str, a: &[usize], b: &[usize], cmd: Cmd) -> Res<(Tensor, Tensor)> {
        let mut v = self.outputs(op, &[a, b], 0, cmd)?.into_iter();
        match (v.next(), v.next()) {
            (Some(x), Some(y)) => Ok((x, y)),
            _ => Err(metal_err(format!("{op}: missing outputs"))),
        }
    }

    /// Causal SDPA forward after its validator: the output and the
    /// `[B, H, T]` row log-sum-exp. Grouped-query heads read their KV head
    /// in place, so nothing beyond the two outputs is charged.
    fn sdpa_forward(
        &self,
        op: &'static str,
        dims: &SdpaDims,
        window: Option<usize>,
        [q, k, v]: [&Tensor; 3],
    ) -> Res<(Tensor, Tensor)> {
        let (qa, ka, va) = (self.f32(op, q)?, self.f32(op, k)?, self.f32(op, v)?);
        let geom = sdpa_launch(op, dims, window)?;
        let shapes = [q.shape(), lse_shape(q.shape())];
        let mut out = self
            .outputs(
                op,
                &shapes,
                0,
                Cmd::Sdpa {
                    q: qa,
                    k: ka,
                    v: va,
                    geom,
                },
            )?
            .into_iter();
        match (out.next(), out.next()) {
            (Some(y), Some(lse)) => Ok((y, lse)),
            _ => Err(metal_err(format!("{op}: missing outputs"))),
        }
    }

    /// Attention of validated `dims` on tessl's tiled `encoder_attn`
    /// ([`routes_to_encoder_attn`] says which calls). tessl takes equal query
    /// and key lengths, so a shorter Q (`Tq < Tk`) is zero-padded to `Tk`
    /// rows and only the first `Tq` output rows are kept. Scratch: the
    /// permuted (and padded) Q, K, V and the kernel's `Tk`-row output, one
    /// length per batch row, and a copy of each operand that is a view.
    ///
    /// `None` when the budget has room for the output but not for that
    /// scratch: the caller runs the general kernel, which needs only the
    /// output and the log-sum-exp. Nothing has been charged or enqueued
    /// then: a refused reservation takes no room, the output's reservation
    /// is released as it drops, and the device has not been called.
    fn encoder_attn_forward(
        &self,
        op: &'static str,
        dims: &AttentionDims,
        [q, k, v]: [&Tensor; 3],
    ) -> Res<Option<Tensor>> {
        let (qa, ka, va) = (self.f32(op, q)?, self.f32(op, k)?, self.f32(op, v)?);
        let geom = EncAttnGeom {
            batch: u32_dim(op, dims.batch)?,
            heads: u32_dim(op, dims.heads)?,
            seq: u32_dim(op, dims.kv_len)?,
            q_len: u32_dim(op, dims.q_len)?,
            d: u32_dim(op, dims.head_dim)?,
        };
        let n = product(op, &[dims.batch, dims.heads, dims.q_len, dims.head_dim])?;
        let n_kv = product(op, &[dims.batch, dims.heads, dims.kv_len, dims.head_dim])?;
        u32_dim(op, n_kv)?;
        let scratch = plus(
            op,
            plus(op, product(op, &[4, n_kv])?, dims.batch)?,
            copied_elems(op, ReadAs::Whole, &[qa, ka, va])?,
        )?;
        let cmd = Cmd::EncoderAttn {
            op,
            q: qa,
            k: ka,
            v: va,
            geom,
        };
        let output = self.reserve(op, n)?;
        let _scratch = match self.reserve(op, scratch) {
            Ok(held) => held,
            Err(OjasError::CapacityExceeded { .. }) => return Ok(None),
            Err(e) => return Err(e),
        };
        let mut out = self.call_wrap(op, &[q.shape()], vec![output], cmd)?;
        out.pop()
            .map(Some)
            .ok_or_else(|| metal_err(format!("{op}: no output")))
    }

    /// General attention forward after its validator: the output and the
    /// `[B, H, Tq]` row log-sum-exp. A non-causal, unwindowed call at head dim
    /// 16 or 32 runs the split-K tiled kernels and is charged their partial
    /// results (`splits * B * H * Tq * (D + 2)` values) as scratch; when the
    /// budget cannot hold them it runs the scalar `ojas_attn_gen_fwd`, which
    /// is charged nothing beyond the two outputs, as is every other call.
    /// Grouped-query heads read their KV head in place.
    fn attn_forward(
        &self,
        op: &'static str,
        dims: &AttentionDims,
        [q, k, v]: [&Tensor; 3],
        bias: Option<&Tensor>,
        key_lens: Option<&Tensor>,
    ) -> Res<(Tensor, Tensor)> {
        let (qa, ka, va) = (self.f32(op, q)?, self.f32(op, k)?, self.f32(op, v)?);
        let bias_arg = bias.map(|b| self.f32(op, b)).transpose()?;
        let lens_arg = key_lens.map(|l| self.arg(op, l, DType::U32)).transpose()?;
        let geom = attn_launch(op, dims)?;
        let shapes = [q.shape(), lse_shape(q.shape())];
        let command = |geom: AttnGeom| Cmd::Attn {
            op,
            q: qa,
            k: ka,
            v: va,
            bias: bias_arg,
            lens: lens_arg,
            geom,
        };
        // The split-K tiled kernel where it covers the call, with its
        // partial results charged as scratch; a budget that holds only the
        // outputs runs the scalar kernel, which needs nothing more.
        let tiled = xattn_splits(&geom).and_then(|(splits, per_split)| {
            let words = product(
                op,
                &[
                    splits as usize,
                    geom.bh as usize,
                    geom.q_len as usize,
                    geom.d as usize + 2,
                ],
            )
            .ok()?;
            Some((
                AttnGeom {
                    splits,
                    per_split,
                    ..geom
                },
                words,
            ))
        });
        let run = |g: AttnGeom, scratch: usize| self.outputs(op, &shapes, scratch, command(g));
        let outputs = match tiled {
            Some((tiled_geom, words)) => match run(tiled_geom, words) {
                Err(OjasError::CapacityExceeded { .. }) => run(geom, 0),
                other => other,
            },
            None => run(geom, 0),
        };
        let mut out = outputs?.into_iter();
        match (out.next(), out.next()) {
            (Some(y), Some(lse)) => Ok((y, lse)),
            _ => Err(metal_err(format!("{op}: missing outputs"))),
        }
    }

    /// Causal SDPA backward after its validator, from the forward's output
    /// and log-sum-exp: `[q, k, v, output, lse, grad_output]`. The only
    /// scratch is the per-row `Dr = dO · O`, `B * H * T` floats; grouped-query
    /// gradients are summed in place onto their KV head.
    fn sdpa_backward(
        &self,
        op: &'static str,
        dims: &SdpaDims,
        window: Option<usize>,
        operands: [&Tensor; 6],
    ) -> Res<(Tensor, Tensor, Tensor)> {
        let [q, k, v, ..] = operands;
        let mut args = [Arg {
            id: 0,
            off: 0,
            n: 0,
        }; 6];
        for (slot, t) in args.iter_mut().zip(operands) {
            *slot = self.f32(op, t)?;
        }
        let geom = sdpa_launch(op, dims, window)?;
        let scratch = product(op, lse_shape(q.shape()))?;
        let mut out = self
            .outputs(
                op,
                &[q.shape(), k.shape(), v.shape()],
                scratch,
                Cmd::SdpaBwd { args, geom },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(a), Some(b), Some(c)) => Ok((a, b, c)),
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
        let scratch_of = |s: &RmsSide| {
            if grads.is_some() {
                rms_bwd_scratch(s)
            } else {
                0
            }
        };
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
        let shapes: Vec<&[usize]> = q_shapes
            .iter()
            .chain(&k_shapes)
            .map(Vec::as_slice)
            .collect();
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

/// f32 values of scratch `muon_ns5_step` reserves for a `[rows, cols]`
/// matrix: `6 n + 3 r² + 2048` with `n = rows * cols`, `r = min(rows, cols)`.
fn muon_scratch_elems(op: &'static str, rows: usize, cols: usize) -> Res<usize> {
    let n = rows.checked_mul(cols).ok_or_else(|| overflow(op))?;
    let r = rows.min(cols);
    n.checked_mul(6)
        .and_then(|a| r.checked_mul(r)?.checked_mul(3)?.checked_add(a))
        .and_then(|s| s.checked_add(2048))
        .ok_or_else(|| overflow(op))
}

/// Widths tessl's `qwen35_conv1d_silu*` kernels take
/// (`CONV_BWD_MAX_KW` in tessl's source).
const METAL_CONV1D_WIDTHS: std::ops::RangeInclusive<usize> = 2..=8;
/// Flattened rows per weight-gradient block of tessl's conv backward
/// (`CONV_ROWS_PER_BLOCK` in `qwen35_bwd.rs`).
const METAL_CONV1D_BWD_ROWS: usize = 256;
/// Widest row tessl's gated-norm backward takes (`32 * MAX_COLS`).
const METAL_GATED_RMS_BWD_MAX_DIM: usize = 512;
/// Rows per weight-gradient block of tessl's gated-norm backward
/// (`GATED_UNITS_PER_BLOCK`).
const METAL_GATED_RMS_BWD_ROWS: usize = 64;
/// Largest `k` `ojas_topk_rows` takes. The kernel walks the whole row once
/// per leader, so its cost grows as `k * cols`; sampling's top-k sits far
/// below this, and a larger `k` belongs to a sort, not `k` selection rounds.
pub const METAL_TOPK_MAX_K: usize = 1024;

/// The conv kernels' geometry of validated `d`: a width tessl is compiled
/// for, and `batch * seq` within 32-bit indexing.
fn conv1d_geom(op: &'static str, d: &Conv1dDims) -> Res<Conv1dGeom> {
    if !METAL_CONV1D_WIDTHS.contains(&d.width) {
        return Err(OjasError::Unsupported {
            op,
            detail: format!(
                "Metal's conv1d kernels take widths {METAL_CONV1D_WIDTHS:?}, got {}",
                d.width
            ),
        });
    }
    u32_dim(op, product(op, &[d.batch, d.time, d.channels])?)?;
    Ok(Conv1dGeom {
        batch: u32_dim(op, d.batch)?,
        seq: u32_dim(op, d.time)?,
        channels: u32_dim(op, d.channels)?,
        width: u32_dim(op, d.width)?,
    })
}

/// f32 values of weight-gradient scratch the conv backward needs; the device
/// checks this against tessl's `conv1d_silu_bwd_part_len`.
fn conv1d_part_elems(op: &'static str, d: &Conv1dDims) -> Res<usize> {
    let blocks = product(op, &[d.batch, d.time])?.div_ceil(METAL_CONV1D_BWD_ROWS);
    product(op, &[blocks, d.channels, d.width])
}

/// `(rows, dim)` of validated `d` within 32-bit indexing, every element too.
fn gated_rms_geom(op: &'static str, d: &RmsDims) -> Res<(u32, u32)> {
    u32_dim(op, product(op, &[d.rows, d.dim])?)?;
    Ok((u32_dim(op, d.rows)?, u32_dim(op, d.dim)?))
}

/// The partial rope kernel's `(rows, dim, rotary, mode)` of validated `d`.
fn rope_partial_geom(op: &'static str, d: &PartialRopeDims) -> Res<([u32; 3], RopeMode)> {
    u32_dim(op, product(op, &[d.rows, d.dim])?)?;
    let mode = RopeMode::TimeDim {
        time: u32_dim(op, d.time)?,
        heads: u32_dim(op, d.heads)?,
    };
    Ok((
        [
            u32_dim(op, d.rows)?,
            u32_dim(op, d.dim)?,
            u32_dim(op, d.rotary)?,
        ],
        mode,
    ))
}

/// f32 values the device copies for `args` read `how` ([`Arg::copied`]).
/// The copies live for the call, so the op charges them with its scratch.
fn copied_elems(op: &'static str, how: ReadAs, args: &[Arg]) -> Res<usize> {
    args.iter()
        .filter(|a| a.copied(how))
        .try_fold(0usize, |acc, a| acc.checked_add(a.n))
        .ok_or_else(|| overflow(op))
}

/// Scratch values of the bf16 copies a linear GEMM on `lane` makes of
/// operands of `a` and `b` elements ([`Lane::copies`]).
fn lane_copies(op: &'static str, lane: Lane, a: usize, b: usize) -> Res<usize> {
    lane.copies(a, b).ok_or_else(|| overflow(op))
}

/// `a + b` scratch values, or the op's overflow error.
fn plus(op: &'static str, a: usize, b: usize) -> Res<usize> {
    a.checked_add(b).ok_or_else(|| overflow(op))
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

/// Refuse a convolution Metal does not run: dilation above 1, and a grouped
/// convolution that is not depthwise (`groups == in == out`, one filter per
/// channel). Any stride and padding runs: `ojas_im2col` forms the window of
/// every output pixel, and the depthwise kernel does the same inline. The
/// validator has run, so every refusal here is a device limit, never a
/// malformed call.
fn refuse_unsupported_conv2d(op: &'static str, d: &Conv2dDims) -> Res<()> {
    let p = d.params;
    let depthwise = p.groups == d.in_channels && d.in_channels == d.out_channels;
    if p.dilation == [1, 1] && (p.groups == 1 || depthwise) {
        return Ok(());
    }
    Err(OjasError::Unsupported {
        op,
        detail: format!(
            "Metal runs dilation 1 with one group or one filter per channel (depthwise), at any \
             stride and padding; got kernel {:?}, stride {:?}, padding {:?}, dilation {:?}, \
             groups {} of {} in and {} out channels",
            d.kernel, p.stride, p.padding, p.dilation, p.groups, d.in_channels, d.out_channels
        ),
    })
}

/// One bilinear axis's source-index scale, in f32 with the CPU's own
/// expressions (`ojas-cpu/src/spatial.rs` `bilinear_axis`).
fn bilinear_scale(size: usize, out: usize, align_corners: bool) -> f32 {
    if align_corners {
        if out > 1 {
            (size - 1) as f32 / (out - 1) as f32
        } else {
            0.0
        }
    } else {
        size as f32 / out as f32
    }
}

/// Refuse a transposed convolution the GEMM reformulation does not cover:
/// only a kernel equal to its stride, without padding or output padding.
fn refuse_unsupported_conv_transpose2d(op: &'static str, d: &ConvTranspose2dDims) -> Res<()> {
    let p = d.params;
    if p.stride == d.kernel && p.padding == [0, 0] && p.output_padding == [0, 0] {
        return Ok(());
    }
    Err(OjasError::Unsupported {
        op,
        detail: format!(
            "Metal runs kernel == stride without padding or output padding; got kernel {:?}, \
             stride {:?}, padding {:?}, output_padding {:?}",
            d.kernel, p.stride, p.padding, p.output_padding
        ),
    })
}

/// Threadgroups the GroupNorm passes aim to launch, so a few large rows
/// (SAM's 8 rows of 0.66 to 2.65 million values) fill the GPU, and the fewest
/// values a threadgroup is given.
const GROUP_NORM_TARGET_GROUPS: usize = 512;
const GROUP_NORM_MIN_CHUNK: usize = 1024;
/// Threads of a GroupNorm threadgroup (`GN_THREADS`).
const GROUP_NORM_THREADS: usize = 256;

/// `(chunks, chunk_len)` of a GroupNorm row of `width` values among `rows`
/// rows: enough chunks to reach the target, but no more than one per
/// `GROUP_NORM_MIN_CHUNK` values of the row (rounded up, so a chunk can be
/// shorter than the minimum: a row of 1025 gives two of 513), and the ranges
/// cover the row.
fn group_norm_chunks(rows: usize, width: usize) -> (usize, usize) {
    let wanted = GROUP_NORM_TARGET_GROUPS.div_ceil(rows.max(1));
    let chunks = wanted.clamp(1, width.div_ceil(GROUP_NORM_MIN_CHUNK).max(1));
    let chunk_len = width.div_ceil(chunks);
    (width.div_ceil(chunk_len).max(1), chunk_len)
}

/// The convolution kernels' geometry of validated `dims`.
fn conv_geom(op: &'static str, d: &Conv2dDims) -> Res<ConvGeom> {
    let pair = |v: [usize; 2]| -> Res<[u32; 2]> { Ok([u32_dim(op, v[0])?, u32_dim(op, v[1])?]) };
    Ok(ConvGeom {
        batch: u32_dim(op, d.batch)?,
        c: u32_dim(op, d.in_channels)?,
        h: u32_dim(op, d.height)?,
        w: u32_dim(op, d.width)?,
        o: u32_dim(op, d.out_channels)?,
        kernel: pair(d.kernel)?,
        stride: pair(d.params.stride)?,
        padding: pair(d.params.padding)?,
        oh: u32_dim(op, d.out_height)?,
        ow: u32_dim(op, d.out_width)?,
    })
}

/// The head dims `encoder_attn` is routed at. 64 is the SAM vision attention,
/// where it measured 22.8 ms against 606 ms for `ojas_attn_gen_fwd` on
/// `[1, 16, 5184, 64]` and 4.7 ms against 77 ms on `[9, 16, 576, 64]`
/// (bench/results/2026-10-26-attention-vision). 32 is the same tiled kernel
/// family (`ENCODER_TILED_KERNEL`, one simdgroup and 16 queries a group) with
/// the same i32-plane and length semantics. The head dims tessl compiles
/// beyond these (256, 512) have no measurement here.
const ENCODER_ATTN_HEAD_DIMS: [usize; 2] = [32, 64];

/// The longest `Tk / Tq` a cross-attention is routed to `encoder_attn` at;
/// beyond it the split-K tiled kernel (`ojas_xattn_part_dD`) runs it.
///
/// The pad route pads Q to `Tk` rows, so it computes `Tk^2` scores where the
/// tiled kernel computes `Tq * Tk`: pad-q wins while the waste is small and
/// loses as `Tk / Tq` grows. Measured interleaved at q `[1, 8, 5184, 32]`
/// (`bench/results/2026-10-27-tiled-vs-padq/summary.md`, Lane C3, 50 paired
/// samples each): pad-q wins at ratios 1.003, 2.006 and 2.049 (m 5200, 10400,
/// 10624) and tiled wins at 3.012 (m 15616); a two-parameter fit puts the
/// crossover near 2.23. SpikeTrack's ratio is `F + P * 16 / 5184` for F frames
/// of memory and P prompt tokens, so F = 2 spans 2.003 to 2.049 and F = 3
/// starts at 3.0: any bound in that gap routes every SAM shape the same way,
/// and 2.25 is the fit's neighbour. It is exact in binary, so the comparison
/// has no rounding at the bound. Rerun that bench whenever either kernel
/// changes, and delete the loser once one kernel wins at every m.
const ENCODER_ATTN_MAX_PAD_RATIO: f64 = 2.25;

/// Whether `attention_forward` runs on tessl's tiled `encoder_attn` rather
/// than `ojas_attn_gen_fwd`: bidirectional attention over equal head counts
/// with no bias, key lengths or window, at one of [`ENCODER_ATTN_HEAD_DIMS`],
/// with `0 < Tq <= Tk <= ENCODER_ATTN_MAX_PAD_RATIO * Tq` (a shorter Q is
/// padded; Metal refuses an empty operand before routing, so the `Tq > 0` term
/// is defensive), on a device with TensorOps, with one head's `Tk`-row plane addressable in
/// `i32` as tessl's matrix-unit views require and the padded tensor within
/// the kernels' 32-bit element indexing (the permutes index it in `u32`).
/// Everything else keeps the general kernel, which refuses what it cannot
/// run. A budget that holds the output but not the routed path's scratch
/// also runs the general kernel ([`MetalBackend::encoder_attn_forward`]).
/// `attention_forward_saving` never routes: `encoder_attn` returns no
/// log-sum-exp.
fn routes_to_encoder_attn(dims: &AttentionDims, tensorops: bool) -> bool {
    let plane = dims
        .kv_len
        .checked_mul(dims.heads)
        .and_then(|p| p.checked_mul(dims.head_dim))
        .is_some_and(|p| i32::try_from(p).is_ok());
    let elements = plane
        .then(|| dims.batch.checked_mul(dims.kv_len))
        .flatten()
        .and_then(|bt| bt.checked_mul(dims.heads))
        .and_then(|bth| bth.checked_mul(dims.head_dim))
        .is_some_and(|n| u32::try_from(n).is_ok());
    let padded_ok = dims.q_len > 0
        && dims.q_len <= dims.kv_len
        && dims.kv_len as f64 <= ENCODER_ATTN_MAX_PAD_RATIO * dims.q_len as f64;
    tensorops
        && !dims.causal
        && dims.window.is_none()
        && dims.bias.is_none()
        && !dims.key_lens
        && padded_ok
        && dims.heads == dims.kv_heads
        && ENCODER_ATTN_HEAD_DIMS.contains(&dims.head_dim)
        && plane
        && elements
}

/// The attention kernels' geometry of validated `dims`, with Metal's
/// head-dim cap: a device limit, so after the validator (D9). `rep` is 1
/// when the head counts match, including both zero, so this never divides
/// by zero. `bh` counts query heads. A window of at least `T` sees every
/// earlier key and is sent as `0`.
fn sdpa_launch(op: &'static str, dims: &SdpaDims, window: Option<usize>) -> Res<SdpaGeom> {
    let d = u32_dim(op, dims.head_dim)?;
    refuse_unsupported_metal_head_dim(BackendId::Metal, d)?;
    let bh = product(op, &[dims.batch, dims.heads])?;
    let rep = if dims.kv_heads == dims.heads {
        1
    } else {
        dims.heads / dims.kv_heads
    };
    let window = match window {
        Some(0) => return Err(metal_err(format!("{op}: sdpa window must be at least 1"))),
        Some(w) if w < dims.seq => u32_dim(op, w)?,
        _ => 0,
    };
    Ok(SdpaGeom {
        bh: u32_dim(op, bh)?,
        t: u32_dim(op, dims.seq)?,
        d,
        rep: u32_dim(op, rep)?,
        window,
    })
}

/// The general attention kernel's geometry of validated `dims`, with
/// Metal's head-dim cap: a device limit, so after the validator (D9). The
/// kernel forms positions in `i32`, so both lengths stay below it. A window
/// of at least `Tk` arrives as `None` and is sent as `0`.
fn attn_launch(op: &'static str, dims: &AttentionDims) -> Res<AttnGeom> {
    let d = u32_dim(op, dims.head_dim)?;
    refuse_unsupported_metal_head_dim(BackendId::Metal, d)?;
    let bh = product(op, &[dims.batch, dims.heads])?;
    let positions = |n: usize| {
        i32::try_from(n)
            .ok()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| OjasError::Unsupported {
                op,
                detail: format!("sequence length {n} exceeds the kernel's 31-bit positions"),
            })
    };
    Ok(AttnGeom {
        bh: u32_dim(op, bh)?,
        heads: u32_dim(op, dims.heads)?,
        rep: u32_dim(op, dims.heads / dims.kv_heads)?,
        q_len: positions(dims.q_len)?,
        kv_len: positions(dims.kv_len)?,
        d,
        causal: dims.causal,
        window: dims.window.map_or(Ok(0), |w| u32_dim(op, w))?,
        bias: dims
            .bias
            .map(|b| {
                Ok::<_, OjasError>([
                    u32_dim(op, b.batch)?,
                    u32_dim(op, b.heads)?,
                    u32_dim(op, b.queries)?,
                ])
            })
            .transpose()?,
        has_lens: dims.key_lens,
        splits: 0,
        per_split: 0,
    })
}

/// Keys per tile of `ojas_xattn_part_dD` (`XA_TILE`) and its threadgroup's
/// query rows (`XA_ROWS`).
const XATTN_TILE: u32 = 64;
const XATTN_ROWS: u32 = 128;
/// Threadgroups the split-K kernel aims to launch, so a short Q over a long K
/// still fills the GPU.
const XATTN_TARGET_GROUPS: u64 = 384;
/// Most key ranges one row is cut into.
const XATTN_MAX_SPLITS: u32 = 64;

/// `(splits, per_split)` of the split-K tiled kernel for `g`, or `None` when
/// the scalar kernel must run it (causal or windowed, or a head dim other
/// than 16 and 32). Each range is a whole number of key tiles and none is
/// empty.
fn xattn_splits(g: &AttnGeom) -> Option<(u32, u32)> {
    if g.causal || g.window != 0 || !matches!(g.d, 16 | 32) || g.q_len == 0 || g.kv_len == 0 {
        return None;
    }
    let groups = u64::from(g.q_len.div_ceil(XATTN_ROWS)) * u64::from(g.bh);
    let wanted = XATTN_TARGET_GROUPS.div_ceil(groups).max(1);
    let by_keys = g.kv_len.div_ceil(XATTN_TILE);
    let splits = u32::try_from(wanted)
        .unwrap_or(XATTN_MAX_SPLITS)
        .clamp(1, XATTN_MAX_SPLITS)
        .min(by_keys);
    let per_split = g.kv_len.div_ceil(splits).div_ceil(XATTN_TILE) * XATTN_TILE;
    Some((g.kv_len.div_ceil(per_split), per_split))
}

/// `[B, H, T]`: one log-sum-exp per query row of a `[B, H, T, D]` query.
fn lse_shape(q: &[usize]) -> &[usize] {
    &q[..q.len().saturating_sub(1)]
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

impl MetalBackend {
    /// The gate forward, with the per-head sigmoid when `save`.
    fn gate_forward(
        &self,
        [input, weight, bias, attn_out]: [&Tensor; 4],
        save: bool,
    ) -> Res<(Tensor, Option<Tensor>)> {
        const OP: &str = "per_head_sigmoid_gate_forward";
        let dims = per_head_sigmoid_gate_forward_dims(input, weight, bias, attn_out)?;
        let (x, w, b, a) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, attn_out)?,
        );
        let (rows, din, heads, dh) = gate_params(OP, &dims)?;
        let units = rows as usize * heads as usize;
        let scale_shape = [dims.rows, dims.heads];
        let shapes: &[&[usize]] = if save {
            &[attn_out.shape(), &scale_shape]
        } else {
            &[attn_out.shape()]
        };
        let cmd = Cmd::Gate {
            x,
            w,
            b,
            attn: a,
            rows,
            din,
            heads,
            dh,
            save,
        };
        let scratch = plus(OP, units, copied_elems(OP, ReadAs::Matrix, &[x, w])?)?;
        let mut out = self.outputs(OP, shapes, scratch, cmd)?.into_iter();
        let y = out
            .next()
            .ok_or_else(|| metal_err(format!("{OP}: no output")))?;
        Ok((y, out.next()))
    }

    /// The gate backward, from the saving forward's sigmoid when `scales`.
    fn gate_backward(
        &self,
        [input, weight, bias, attn_out, grad_output]: [&Tensor; 5],
        scales: Option<&Tensor>,
    ) -> Res<PerHeadGateGrad> {
        const OP: &str = "per_head_sigmoid_gate_backward";
        let dims = per_head_sigmoid_gate_backward_dims(input, weight, bias, attn_out, grad_output)?;
        if let Some(s) = scales {
            if s.shape() != [dims.rows, dims.heads] {
                return Err(shape(
                    OP,
                    format!(
                        "saved scales {:?}, expected [{}, {}]",
                        s.shape(),
                        dims.rows,
                        dims.heads
                    ),
                ));
            }
        }
        let (x, w, b, a) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, attn_out)?,
        );
        let gy = self.f32(OP, grad_output)?;
        let scales = scales.map(|s| self.f32(OP, s)).transpose()?;
        let (rows, din, heads, dh) = gate_params(OP, &dims)?;
        let units = rows as usize * heads as usize;
        // d_pre, plus the logits when they are recomputed, plus the GEMM
        // operands' copies.
        let pre = if scales.is_some() { units } else { 2 * units };
        let scratch = plus(OP, pre, copied_elems(OP, ReadAs::Matrix, &[x, w])?)?;
        let mut out = self
            .outputs(
                OP,
                &[
                    input.shape(),
                    weight.shape(),
                    bias.shape(),
                    attn_out.shape(),
                ],
                scratch,
                Cmd::GateBwd {
                    x,
                    w,
                    b,
                    attn: a,
                    gy,
                    scales,
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
}

impl MetalBackend {
    /// The gated delta rule's operands placed in argument order, then
    /// tessl's compiled dims ([`refuse_unsupported_metal_gdn`]) and the
    /// 32-bit dims its kernels index with.
    fn gdn_args(&self, op: &'static str, inputs: GdnInputs<'_>, d: &GdnDims) -> Res<GdnArgs> {
        let q = self.f32(op, inputs.q)?;
        let k = self.f32(op, inputs.k)?;
        let v = self.f32(op, inputs.v)?;
        let g = self.f32(op, inputs.g)?;
        let beta = self.f32(op, inputs.beta)?;
        let s0 = inputs.initial_state.map(|t| self.f32(op, t)).transpose()?;
        refuse_unsupported_metal_gdn(BackendId::Metal, d.key_dim, d.value_dim)?;
        u32_dim(op, d.rows())?;
        let (batch, seq, heads, v_dim) = (
            u32_dim(op, d.batch)?,
            u32_dim(op, d.seq)?,
            u32_dim(op, d.heads)?,
            u32_dim(op, d.value_dim)?,
        );
        Ok(GdnArgs {
            q,
            k,
            v,
            g,
            beta,
            s0,
            batch,
            seq,
            heads,
            v_dim,
            ws_elems: crate::device::gdn_backward_workspace_elems(batch, seq, heads, v_dim)?,
        })
    }
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
                // The device thread encodes the window into the new buffer
                // itself: no host copy of it is made here, and the device
                // copy is charged before anything is allocated.
                if !tensor.is_contiguous()? {
                    return Err(shape(OP, "view is not contiguous"));
                }
                let len = tensor
                    .num_elements()?
                    .checked_mul(tensor.dtype().size())
                    .ok_or_else(|| overflow(OP))?;
                if len == 0 {
                    return Err(shape(OP, "empty tensor"));
                }
                let reservation = self.budget.try_reserve(len as u64)?;
                let ids = match tensor.dtype() {
                    DType::U32 => Some(Ids::Host(Arc::from(tensor.u32_slice()?))),
                    _ => None,
                };
                let nb = match self.link.call(Cmd::Upload {
                    src: tensor.clone(),
                })? {
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
        let out = permute_dims(input, dims)?;
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
        let half = table.dtype() == DType::Bf16;
        let t = self.arg(OP, table, table.dtype())?;
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
                half,
                ids,
                vocab,
                dim,
            },
        )
    }

    /// `ojas_argmax_rows`: one threadgroup per row. A non-finite input is a
    /// deferred fault, reported at the next sync. The result stays on the
    /// device and carries [`Ids::Below`]`(cols)`, so it can feed
    /// `embedding_forward` with no readback and no host copy.
    fn argmax_rows(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "argmax_rows";
        let (rows, cols) = argmax_rows_dims(x)?;
        let xa = self.f32(OP, x)?;
        let (r, c) = (u32_dim(OP, rows)?, u32_dim(OP, cols)?);
        let reservation = self.reserve(OP, rows)?;
        let bufs = match self.link.call(Cmd::Argmax {
            x: xa,
            rows: r,
            cols: c,
        })? {
            Reply::Bufs(bufs) => bufs,
            other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
        };
        // Wrapped before the count check, so a surplus buffer is freed.
        let mut owned: Vec<Arc<MetalBuffer>> = bufs
            .into_iter()
            .map(|b| {
                Arc::new(MetalBuffer {
                    id: b.id,
                    len: b.bytes,
                    link: Arc::clone(&self.link),
                    ids: Some(Ids::Below(c)),
                })
            })
            .collect();
        let (Some(buf), true) = (owned.pop(), owned.is_empty()) else {
            return Err(metal_err(format!("{OP}: device did not return one output")));
        };
        let buf: Arc<dyn DeviceBuffer> = buf;
        Tensor::from_device_reserved(buf, &[rows], DType::U32, reservation)
    }

    /// `ojas_topk_rows`: one threadgroup per row and `k` rounds over the
    /// row, so the kernel's cost is `k * cols / 256` reads per lane; `k` past
    /// [`METAL_TOPK_MAX_K`] is [`OjasError::Unsupported`]. A NaN or `+inf` input
    /// is a deferred fault, reported at the next sync. Both results stay on
    /// the device; the ids carry [`Ids::Below`]`(cols)`, so they can feed
    /// `embedding_forward` with no readback and no host copy.
    fn topk_rows(&self, x: &Tensor, k: usize) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "topk_rows";
        let (rows, cols) = topk_rows_dims(x, k)?;
        if k > METAL_TOPK_MAX_K {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: format!("k = {k} is past the Metal kernel's {METAL_TOPK_MAX_K}"),
            });
        }
        let xa = self.f32(OP, x)?;
        let (r, c, kk) = (u32_dim(OP, rows)?, u32_dim(OP, cols)?, u32_dim(OP, k)?);
        let n = rows.checked_mul(k).ok_or_else(|| overflow(OP))?;
        let value_charge = self.reserve(OP, n)?;
        let id_charge = self.reserve(OP, n)?;
        let bufs = match self.link.call(Cmd::Topk {
            x: xa,
            rows: r,
            cols: c,
            k: kk,
        })? {
            Reply::Bufs(bufs) => bufs,
            other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
        };
        // Wrapped before the count check, so a surplus buffer is freed. The
        // first is the values (F32), the second the ids.
        let mut owned: Vec<Arc<MetalBuffer>> = bufs
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                Arc::new(MetalBuffer {
                    id: b.id,
                    len: b.bytes,
                    link: Arc::clone(&self.link),
                    ids: (i == 1).then_some(Ids::Below(c)),
                })
            })
            .collect();
        let (Some(id_buf), Some(value_buf), true) = (owned.pop(), owned.pop(), owned.is_empty())
        else {
            return Err(metal_err(format!(
                "{OP}: device did not return two outputs"
            )));
        };
        let value_buf: Arc<dyn DeviceBuffer> = value_buf;
        let id_buf: Arc<dyn DeviceBuffer> = id_buf;
        let values = Tensor::from_device_reserved(value_buf, &[rows, k], DType::F32, value_charge)?;
        let ids = Tensor::from_device_reserved(id_buf, &[rows, k], DType::U32, id_charge)?;
        Ok((values, ids))
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        let dims = embedding_backward_dims(table, token_ids, grad_output)?;
        let half = table.dtype() == DType::Bf16;
        let t = self.arg(OP, table, table.dtype())?;
        self.arg(OP, token_ids, DType::U32)?;
        let g = self.f32(OP, grad_output)?;
        let (vocab, dim) = (u32_dim(OP, dims.vocab)?, u32_dim(OP, dims.dim)?);
        let (ids, _) = self.ids(OP, token_ids, vocab, None)?;
        // The rows are grouped on the host from the ids it holds, as wgpu
        // does; ids produced on the device have no host copy and are
        // refused, as on wgpu.
        let host = self.host_ids(OP, token_ids, ids)?;
        let rows = EmbedRows::new(host, dims.vocab);
        let scratch = 2 * vocab as usize + ids.n;
        self.one(
            OP,
            table.shape(),
            scratch,
            Cmd::EmbedBwd {
                table: t,
                half,
                grad: g,
                rows,
                dim,
            },
        )
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let dims = linear_forward_dims(input, weight)?;
        let lane = self.link.lane(input.bf16_exact(), weight.bf16_exact());
        let (input, weight) = (self.widened(input)?, self.widened(weight)?);
        let x = self.f32(OP, &input)?;
        let w = self.f32(OP, &weight)?;
        u32_dim(OP, product(OP, &dims.out_shape)?)?;
        let scratch = plus(
            OP,
            copied_elems(OP, ReadAs::Matrix, &[x, w])?,
            lane_copies(OP, lane, x.n, w.n)?,
        )?;
        self.one(
            OP,
            &dims.out_shape,
            scratch,
            Cmd::Linear {
                x,
                w,
                rows: dims.rows,
                kin: dims.in_features,
                nout: dims.out_features,
                lane,
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
        let gx_lane = self
            .link
            .lane(grad_output.bf16_exact(), weight.bf16_exact());
        let gw_lane = self.link.lane(grad_output.bf16_exact(), input.bf16_exact());
        let input = self.widened(input)?;
        let weight = self.widened(weight)?;
        let grad_output = self.widened(grad_output)?;
        let x = self.f32(OP, &input)?;
        let w = self.f32(OP, &weight)?;
        let gy = self.f32(OP, &grad_output)?;
        let cmd = Cmd::LinearBwd {
            x,
            w,
            gy,
            rows: dims.rows,
            kin: dims.in_features,
            nout: dims.out_features,
            gx_lane,
            gw_lane,
        };
        // Each GEMM's bf16 copies live until the command's commit, so both
        // are charged.
        let copies = plus(
            OP,
            copied_elems(OP, ReadAs::Matrix, &[x, w, gy])?,
            plus(
                OP,
                lane_copies(OP, gx_lane, gy.n, w.n)?,
                lane_copies(OP, gw_lane, gy.n, x.n)?,
            )?,
        )?;
        let mut out = self
            .outputs(OP, &[input.shape(), weight.shape()], copies, cmd)?
            .into_iter();
        match (out.next(), out.next()) {
            (Some(gx), Some(gw)) => Ok((gx, gw)),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn linear_backward_input(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_backward_input";
        let dims = linear_backward_dims(input, weight, grad_output)?;
        let lane = self
            .link
            .lane(grad_output.bf16_exact(), weight.bf16_exact());
        let input = self.widened(input)?;
        let weight = self.widened(weight)?;
        let grad_output = self.widened(grad_output)?;
        let x = self.f32(OP, &input)?;
        let w = self.f32(OP, &weight)?;
        let gy = self.f32(OP, &grad_output)?;
        let scratch = plus(
            OP,
            copied_elems(OP, ReadAs::Matrix, &[x, w, gy])?,
            lane_copies(OP, lane, gy.n, w.n)?,
        )?;
        self.one(
            OP,
            input.shape(),
            scratch,
            Cmd::LinearBwdInput {
                x,
                w,
                gy,
                rows: dims.rows,
                kin: dims.in_features,
                nout: dims.out_features,
                lane,
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

    fn causal_sdpa_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_forward";
        let dims = causal_sdpa_forward_dims(q, k, v, window)?;
        let (q, k, v) = (self.widened(q)?, self.widened(k)?, self.widened(v)?);
        self.sdpa_forward(OP, &dims, window, [&q, &k, &v])
    }

    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        output: &Tensor,
        lse: &Tensor,
        grad_output: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_backward";
        let dims = causal_sdpa_backward_dims(q, k, v, output, lse, grad_output, window)?;
        // The log-sum-exp stays F32 (the op's own check refuses a Bf16 one).
        let [q, k, v, output, grad_output] =
            [q, k, v, output, grad_output].map(|t| self.widened(t));
        let (q, k, v, output, grad_output) = (q?, k?, v?, output?, grad_output?);
        self.sdpa_backward(OP, &dims, window, [&q, &k, &v, &output, lse, &grad_output])
    }

    /// The inference form of [`Self::attention_forward_saving`]: it refuses
    /// a malformed call under its own name, as the CPU backend does, and a
    /// non-finite operand faults under `attention_forward`.
    ///
    /// Not bit-identical to the saving form for unmasked attention at head
    /// dim 32 or 64 on a TensorOps device, self- or cross-attention with
    /// `0 < Tq <= Tk <= 2.25 Tq` (`routes_to_encoder_attn`): that call runs
    /// tessl's tiled `encoder_attn`, 26x faster at `[1, 16, 5184, 64]` (22.8
    /// ms against 606 ms), whose sums add in another order. The two forms
    /// agree within the forward tolerance (1.5e-7 measured at the SAM
    /// shapes). Every other call runs the same kernel as the saving form and
    /// is bit-identical to it, with one caveat: which kernel a head dim 16 or
    /// 32 call runs, the split-K tiled one or the scalar one, depends on the
    /// budget (the tiled one needs room for its partials), and the two agree
    /// to rounding, not bit for bit.
    fn attention_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        opts: AttentionOpts<'_>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "attention_forward";
        let dims = attention_forward_dims(q, k, v, &opts)?;
        let (q, k, v) = (self.widened(q)?, self.widened(k)?, self.widened(v)?);
        if routes_to_encoder_attn(&dims, self.link.has_tensorops()) {
            if let Some(output) = self.encoder_attn_forward(OP, &dims, [&q, &k, &v])? {
                return Ok(output);
            }
        }
        let bias = opts.bias.map(|b| self.widened(b)).transpose()?;
        self.attn_forward(OP, &dims, [&q, &k, &v], bias.as_ref(), opts.key_lens)
            .map(|(output, _)| output)
    }

    /// Bidirectional and cross attention, a bias and key lengths: the split-K
    /// tiled kernels (`ojas_xattn_part_dD`) for a non-causal call at head dim
    /// 16 or 32, the scalar `ojas_attn_gen_fwd` for causal, windowed and other
    /// head dims, or when the budget cannot hold the tiled partials (so the
    /// bits of a head dim 16 or 32 call can depend on the budget; the values
    /// agree to rounding). There is no backward on Metal yet:
    /// `attention_backward` is `Unsupported`, as the default says.
    fn attention_forward_saving(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        opts: AttentionOpts<'_>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "attention_forward_saving";
        let dims = attention_forward_saving_dims(q, k, v, &opts)?;
        let (q, k, v) = (self.widened(q)?, self.widened(k)?, self.widened(v)?);
        let bias = opts.bias.map(|b| self.widened(b)).transpose()?;
        self.attn_forward(OP, &dims, [&q, &k, &v], bias.as_ref(), opts.key_lens)
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.gate_forward([input, weight, bias, attn_out], false)
            .map(|(y, _)| y)
    }

    /// The forward and the per-head sigmoid `[rows, heads]` it multiplied
    /// by, bit for bit the value the recomputing backward forms.
    fn per_head_sigmoid_gate_forward_saving(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>), OjasError> {
        self.gate_forward([input, weight, bias, attn_out], true)
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        self.gate_backward([input, weight, bias, attn_out, grad_output], None)
    }

    /// The backward with the saving forward's sigmoid: no logits GEMM and
    /// no bias read. `scales` must be this backend's `[rows, heads]` F32.
    fn per_head_sigmoid_gate_backward_saved(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
        scales: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        self.gate_backward([input, weight, bias, attn_out, grad_output], Some(scales))
    }

    fn chunked_gdn_forward(&self, inputs: GdnInputs<'_>) -> Result<GdnForward, OjasError> {
        const OP: &str = "chunked_gdn_forward";
        let dims = chunked_gdn_forward_dims(inputs)?;
        let x = self.gdn_args(OP, inputs, &dims)?;
        let shapes: [&[usize]; 3] = [
            inputs.v.shape(),
            &dims.state_shape(),
            &dims.checkpoint_shape(),
        ];
        // tessl's chunked forward allocates its prep buffers inside the call.
        let scratch = crate::device::gdn_forward_transient_elems(x.batch, x.seq, x.heads, x.v_dim)?;
        let mut out = self
            .outputs(OP, &shapes, scratch, Cmd::Gdn { x })?
            .into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(output), Some(final_state), Some(checkpoints)) => Ok(GdnForward {
                output,
                final_state,
                checkpoints,
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn chunked_gdn_backward(
        &self,
        inputs: GdnInputs<'_>,
        checkpoints: &Tensor,
        grad_output: &Tensor,
        grad_final_state: Option<&Tensor>,
    ) -> Result<GdnGrad, OjasError> {
        const OP: &str = "chunked_gdn_backward";
        let dims = chunked_gdn_backward_dims(inputs, checkpoints, grad_output, grad_final_state)?;
        let x = self.gdn_args(OP, inputs, &dims)?;
        let ckpt = self.f32(OP, checkpoints)?;
        let d_o = self.f32(OP, grad_output)?;
        let d_fin = grad_final_state.map(|t| self.f32(OP, t)).transpose()?;
        let (qs, vs, gs) = (inputs.q.shape(), inputs.v.shape(), inputs.g.shape());
        let state = dims.state_shape();
        let mut shapes: Vec<&[usize]> = vec![qs, qs, vs, gs, gs];
        if inputs.initial_state.is_some() {
            shapes.push(&state);
        }
        let scratch = x.ws_elems;
        let mut out = self
            .outputs(
                OP,
                &shapes,
                scratch,
                Cmd::GdnBwd {
                    x,
                    ckpt,
                    d_o,
                    d_fin,
                },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next(), out.next(), out.next()) {
            (Some(q), Some(k), Some(v), Some(g), Some(beta)) => {
                let initial_state = out.next();
                if inputs.initial_state.is_some() != initial_state.is_some() {
                    return Err(metal_err(format!("{OP}: initial-state gradient mismatch")));
                }
                Ok(GdnGrad {
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                })
            }
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn causal_conv1d_silu_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "causal_conv1d_silu_forward";
        let dims = causal_conv1d_silu_forward_dims(input, weight)?;
        let (x, w) = (self.f32(OP, input)?, self.f32(OP, weight)?);
        let geom = conv1d_geom(OP, &dims)?;
        let copies = copied_elems(OP, ReadAs::Whole, &[x, w])?;
        self.one(OP, input.shape(), copies, Cmd::Conv1d { x, w, geom })
    }

    fn causal_conv1d_silu_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "causal_conv1d_silu_backward";
        let dims = causal_conv1d_silu_backward_dims(input, weight, grad_output)?;
        let (x, w) = (self.f32(OP, input)?, self.f32(OP, weight)?);
        let gy = self.f32(OP, grad_output)?;
        let geom = conv1d_geom(OP, &dims)?;
        let part = conv1d_part_elems(OP, &dims)?;
        let scratch = plus(OP, part, copied_elems(OP, ReadAs::Whole, &[x, w, gy])?)?;
        let shapes = [input.shape(), weight.shape()];
        let cmd = Cmd::Conv1dBwd {
            x,
            w,
            gy,
            geom,
            part,
        };
        let mut out = self.outputs(OP, &shapes, scratch, cmd)?.into_iter();
        match (out.next(), out.next()) {
            (Some(dx), Some(dw)) => Ok((dx, dw)),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn gated_rms_norm_forward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "gated_rms_norm_forward";
        let dims = gated_rms_norm_forward_dims(input, gate, weight, eps)?;
        let (x, z, w) = (
            self.f32(OP, input)?,
            self.f32(OP, gate)?,
            self.f32(OP, weight)?,
        );
        let (rows, dim) = gated_rms_geom(OP, &dims)?;
        let copies = copied_elems(OP, ReadAs::Whole, &[x, z, w])?;
        let cmd = Cmd::GatedRms {
            x,
            z,
            w,
            rows,
            dim,
            eps,
        };
        self.one(OP, input.shape(), copies, cmd)
    }

    fn gated_rms_norm_backward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<GatedRmsGrad, OjasError> {
        const OP: &str = "gated_rms_norm_backward";
        let dims = gated_rms_norm_backward_dims(input, gate, weight, grad_output, eps)?;
        let (x, z, w) = (
            self.f32(OP, input)?,
            self.f32(OP, gate)?,
            self.f32(OP, weight)?,
        );
        let gy = self.f32(OP, grad_output)?;
        if dims.dim > METAL_GATED_RMS_BWD_MAX_DIM {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: format!(
                    "Metal's gated-norm backward takes rows of at most \
                     {METAL_GATED_RMS_BWD_MAX_DIM}, got {}",
                    dims.dim
                ),
            });
        }
        let (rows, dim) = gated_rms_geom(OP, &dims)?;
        let part = product(
            OP,
            &[dims.rows.div_ceil(METAL_GATED_RMS_BWD_ROWS), dims.dim],
        )?;
        let scratch = plus(OP, part, copied_elems(OP, ReadAs::Whole, &[x, z, w, gy])?)?;
        let shapes = [input.shape(), gate.shape(), weight.shape()];
        let cmd = Cmd::GatedRmsBwd {
            x,
            z,
            w,
            gy,
            rows,
            dim,
            eps,
            part,
        };
        let mut out = self.outputs(OP, &shapes, scratch, cmd)?.into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(input), Some(gate), Some(weight)) => Ok(GatedRmsGrad {
                input,
                gate,
                weight,
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn rope_partial_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_partial_forward";
        let dims = rope_partial_forward_dims(x, cos, sin)?;
        rope_partial(self, OP, &dims, [x, cos, sin], false)
    }

    fn rope_partial_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_partial_backward";
        let dims = rope_partial_backward_dims(grad_output, cos, sin)?;
        rope_partial(self, OP, &dims, [grad_output, cos, sin], true)
    }

    fn rope_interleaved_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_interleaved_forward";
        let dims = rope_interleaved_forward_dims(x, cos, sin)?;
        rope_interleaved(self, OP, &dims, [x, cos, sin], false)
    }

    fn rope_interleaved_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_interleaved_backward";
        let dims = rope_interleaved_backward_dims(grad_output, cos, sin)?;
        rope_interleaved(self, OP, &dims, [grad_output, cos, sin], true)
    }

    /// A convolution as im2col, a GEMM and a layout pass, at any stride and
    /// padding (the patch embedding, 1x1 and 3x3 convolutions, the strided
    /// downsampler); a depthwise one (`groups == in == out`) on its own
    /// direct kernel. Dilation above 1 and other grouped convolutions are
    /// [`OjasError::Unsupported`].
    fn conv2d_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        params: Conv2dParams,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "conv2d_forward";
        let dims = conv2d_forward_dims(input, weight, bias, params)?;
        refuse_unsupported_conv2d(OP, &dims)?;
        let (input, weight) = (self.widened(input)?, self.widened(weight)?);
        let bias = bias.map(|b| self.widened(b)).transpose()?;
        let (x, w) = (self.f32(OP, &input)?, self.f32(OP, &weight)?);
        let bias_arg = bias.as_ref().map(|b| self.f32(OP, b)).transpose()?;
        let geom = conv_geom(OP, &dims)?;
        if dims.params.groups > 1 {
            // Depthwise: one direct kernel, no im2col matrix and no scratch.
            u32_dim(OP, product(OP, &dims.out_shape())?)?;
            let cmd = Cmd::DepthwiseConv2d {
                op: OP,
                x,
                w,
                bias: bias_arg,
                geom,
            };
            return self.one(OP, &dims.out_shape(), 0, cmd);
        }
        let rows = product(OP, &[dims.batch, dims.out_height, dims.out_width])?;
        let width = product(OP, &[dims.in_channels, dims.kernel[0], dims.kernel[1]])?;
        let cols = product(OP, &[rows, width])?;
        let prod = product(OP, &[rows, dims.out_channels])?;
        u32_dim(OP, cols)?;
        let scratch = plus(
            OP,
            plus(OP, cols, prod)?,
            copied_elems(OP, ReadAs::Matrix, &[w])?,
        )?;
        let cmd = Cmd::Conv2d {
            op: OP,
            x,
            w,
            bias: bias_arg,
            geom,
        };
        self.one(OP, &dims.out_shape(), scratch, cmd)
    }

    /// A transposed convolution whose kernel equals its stride (the SAM
    /// neck's 2 x 2 stride 2) as a permute, a GEMM and a layout pass. Any
    /// other stride, padding or output padding is [`OjasError::Unsupported`].
    fn conv_transpose2d_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        params: ConvTranspose2dParams,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "conv_transpose2d_forward";
        let dims = conv_transpose2d_forward_dims(input, weight, bias, params)?;
        refuse_unsupported_conv_transpose2d(OP, &dims)?;
        let (input, weight) = (self.widened(input)?, self.widened(weight)?);
        let bias = bias.map(|b| self.widened(b)).transpose()?;
        let (x, w) = (self.f32(OP, &input)?, self.f32(OP, &weight)?);
        let bias_arg = bias.as_ref().map(|b| self.f32(OP, b)).transpose()?;
        let pair =
            |v: [usize; 2]| -> Res<[u32; 2]> { Ok([u32_dim(OP, v[0])?, u32_dim(OP, v[1])?]) };
        let geom = ConvGeom {
            batch: u32_dim(OP, dims.batch)?,
            c: u32_dim(OP, dims.in_channels)?,
            h: u32_dim(OP, dims.height)?,
            w: u32_dim(OP, dims.width)?,
            o: u32_dim(OP, dims.out_channels)?,
            kernel: pair(dims.kernel)?,
            stride: pair(dims.params.stride)?,
            padding: pair(dims.params.padding)?,
            oh: u32_dim(OP, dims.out_height)?,
            ow: u32_dim(OP, dims.out_width)?,
        };
        let rows = product(OP, &[dims.batch, dims.height, dims.width])?;
        let cols = product(OP, &[dims.out_channels, dims.kernel[0], dims.kernel[1]])?;
        let permuted = product(OP, &[rows, dims.in_channels])?;
        let prod = product(OP, &[rows, cols])?;
        u32_dim(OP, prod)?;
        let scratch = plus(
            OP,
            plus(OP, permuted, prod)?,
            copied_elems(OP, ReadAs::Matrix, &[w])?,
        )?;
        let cmd = Cmd::ConvTranspose2d {
            op: OP,
            x,
            w,
            bias: bias_arg,
            geom,
        };
        self.one(OP, &dims.out_shape(), scratch, cmd)
    }

    /// Torch's bilinear rule on `ojas_bilinear`: each thread forms its own
    /// taps from the host's f32 axis scales, so there are no axis tables to
    /// charge (the CPU charges its two tables to the budget) and no scratch.
    fn interpolate_bilinear_forward(
        &self,
        input: &Tensor,
        out_size: [usize; 2],
        align_corners: bool,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "interpolate_bilinear_forward";
        let dims = interpolate_bilinear_forward_dims(input, out_size)?;
        let x = self.f32(OP, input)?;
        let geom = BilinearGeom {
            planes: u32_dim(OP, product(OP, &[dims.batch, dims.channels])?)?,
            h: u32_dim(OP, dims.height)?,
            w: u32_dim(OP, dims.width)?,
            oh: u32_dim(OP, dims.out_height)?,
            ow: u32_dim(OP, dims.out_width)?,
            scale_y: bilinear_scale(dims.height, dims.out_height, align_corners),
            scale_x: bilinear_scale(dims.width, dims.out_width, align_corners),
            align_corners,
        };
        u32_dim(OP, product(OP, &dims.out_shape())?)?;
        self.one(OP, &dims.out_shape(), 0, Cmd::Bilinear { x, geom })
    }

    /// GroupNorm as three passes over many threadgroups per `(image, group)`
    /// row (see `ojas_gn_sum`): f32 two-pass statistics, so values agree with
    /// the CPU's `f64` ones to f32 summation error, not bit for bit (the mean
    /// is corrected by its residual, so a row far from zero keeps its
    /// accuracy). The per-chunk partial sums (three arrays of `rows * chunks`)
    /// are charged as scratch. A mean or variance that overflows f32 is a
    /// deferred fault where the CPU's `f64` ones finish.
    fn group_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        groups: usize,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "group_norm_forward";
        let dims = group_norm_forward_dims(input, weight, bias, groups, eps)?;
        let (x, w, b) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
        );
        let rows = product(OP, &[dims.batch, dims.groups])?;
        let per_group = dims.channels / dims.groups;
        let width = product(OP, &[per_group, dims.spatial])?;
        u32_dim(OP, product(OP, &[rows, width])?)?;
        // The kernels step a row's index by the threadgroup size in 32 bits.
        u32_dim(
            OP,
            width
                .checked_add(GROUP_NORM_THREADS)
                .ok_or_else(|| overflow(OP))?,
        )?;
        let (chunks, chunk_len) = group_norm_chunks(rows, width);
        let geom = GroupNormGeom {
            rows: u32_dim(OP, rows)?,
            width: u32_dim(OP, width)?,
            chunks: u32_dim(OP, chunks)?,
            chunk_len: u32_dim(OP, chunk_len)?,
            spatial: u32_dim(OP, dims.spatial)?,
            per_group: u32_dim(OP, per_group)?,
            groups: u32_dim(OP, dims.groups)?,
        };
        let scratch = product(OP, &[3, rows, chunks])?;
        self.one(
            OP,
            input.shape(),
            scratch,
            Cmd::GroupNorm { x, w, b, geom, eps },
        )
    }

    /// A bit copy of every input into place along `axis`, one dispatch per
    /// input; a non-finite input is a deferred fault under `concat`. The
    /// output is the only charge.
    fn concat(&self, inputs: &[&Tensor], axis: usize) -> Result<Tensor, OjasError> {
        const OP: &str = "concat";
        let dims = concat_dims(inputs, axis)?;
        let xs = inputs
            .iter()
            .map(|t| self.f32(OP, t))
            .collect::<Res<Vec<Arg>>>()?;
        u32_dim(OP, product(OP, &dims.out_shape)?)?;
        let cmd = Cmd::Concat {
            xs,
            outer: u32_dim(OP, dims.outer)?,
            inner: u32_dim(OP, dims.inner)?,
            axis_lens: dims
                .axis_lens
                .iter()
                .map(|&l| u32_dim(OP, l))
                .collect::<Res<Vec<u32>>>()?,
        };
        self.one(OP, &dims.out_shape, 0, cmd)
    }

    /// A bit copy of rows `start..start + len` of `axis`; a non-finite element
    /// anywhere in the input is a deferred fault under `slice`, as the CPU
    /// refuses it. The output is the only charge.
    fn slice(
        &self,
        input: &Tensor,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "slice";
        let dims = slice_dims(input, axis, start, len)?;
        let x = self.f32(OP, input)?;
        u32_dim(OP, product(OP, &dims.out_shape)?)?;
        let cmd = Cmd::Slice {
            x,
            outer: u32_dim(OP, dims.outer)?,
            inner: u32_dim(OP, dims.inner)?,
            axis_len: u32_dim(OP, dims.axis_len)?,
            start: u32_dim(OP, dims.start)?,
            len: u32_dim(OP, dims.len)?,
        };
        self.one(OP, &dims.out_shape, 0, cmd)
    }

    /// A buffer filled on the device (no host buffer is uploaded): `F32` and
    /// `U32` store the value, `Bf16` its pattern. The output is the only
    /// charge.
    fn fill(&self, shape: &[usize], dtype: DType, value: f32) -> Result<Tensor, OjasError> {
        const OP: &str = "fill";
        let n = fill_dims(shape, dtype, value)?;
        let bytes = n.checked_mul(dtype.size()).ok_or_else(|| overflow(OP))?;
        u32_dim(OP, bytes.div_ceil(4))?;
        let word = match dtype {
            DType::F32 => value.to_bits(),
            // The value is exactly a bf16: its high half is the pattern.
            DType::Bf16 => {
                let half = value.to_bits() >> 16;
                half | (half << 16)
            }
            // An integer in 0..2^32 (fill_dims), so the cast is exact.
            DType::U32 => value as u32,
            other => {
                return Err(OjasError::Dtype {
                    op: OP,
                    expected: DType::F32,
                    got: other,
                })
            }
        };
        let reservation = self.budget.try_reserve(bytes as u64)?;
        let nb = match self.link.call(Cmd::Fill { bytes, word })? {
            Reply::Bufs(mut v) if v.len() == 1 => v.remove(0),
            other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
        };
        let buf: Arc<dyn DeviceBuffer> = Arc::new(MetalBuffer {
            id: nb.id,
            len: nb.bytes,
            link: Arc::clone(&self.link),
            ids: None,
        });
        Tensor::from_device_reserved(buf, shape, dtype, reservation)
    }

    fn relu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "relu_forward";
        relu_forward_dims(input)?;
        let x = self.f32(OP, input)?;
        self.one(OP, input.shape(), 0, Cmd::Relu { x })
    }

    fn sigmoid_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "sigmoid_forward";
        sigmoid_forward_dims(input)?;
        let x = self.f32(OP, input)?;
        self.one(OP, input.shape(), 0, Cmd::Sigmoid { x })
    }

    fn sigmoid_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "sigmoid_backward";
        sigmoid_backward_dims(input, grad_output)?;
        let x = self.f32(OP, input)?;
        let gy = self.f32(OP, grad_output)?;
        self.one(OP, input.shape(), 0, Cmd::SigmoidBwd { x, gy })
    }

    fn gdn_log_decay_forward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "gdn_log_decay_forward";
        let dims = gdn_log_decay_forward_dims(a, a_log, dt_bias)?;
        let heads = u32_dim(OP, dims.heads)?;
        let cmd = Cmd::GdnDecay {
            a: self.f32(OP, a)?,
            a_log: self.f32(OP, a_log)?,
            dt_bias: self.f32(OP, dt_bias)?,
            heads,
        };
        self.one(OP, a.shape(), 0, cmd)
    }

    fn gdn_log_decay_backward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        grad_output: &Tensor,
    ) -> Result<GdnDecayGrad, OjasError> {
        const OP: &str = "gdn_log_decay_backward";
        let dims = gdn_log_decay_backward_dims(a, a_log, dt_bias, grad_output)?;
        let heads = u32_dim(OP, dims.heads)?;
        let cmd = Cmd::GdnDecayBwd {
            a: self.f32(OP, a)?,
            a_log: self.f32(OP, a_log)?,
            dt_bias: self.f32(OP, dt_bias)?,
            gy: self.f32(OP, grad_output)?,
            heads,
        };
        let shapes = [a.shape(), a_log.shape(), dt_bias.shape()];
        let mut out = self.outputs(OP, &shapes, 0, cmd)?.into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(input), Some(a_log), Some(dt_bias)) => Ok(GdnDecayGrad {
                input,
                a_log,
                dt_bias,
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

    fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "cast_bf16";
        let x = self.f32(OP, tensor)?;
        self.one(OP, tensor.shape(), 0, Cmd::RoundBf16 { x })
    }

    /// The matmul-class ops widen each `Bf16` operand on the device into
    /// an `F32` copy charged for the call, then run their `f32` kernels: a
    /// resident `Bf16` weight costs 2 bytes per element, and its widened
    /// copy exists only while one op runs. The tied table is read in place:
    /// the gather widens each picked value, and the fused head one weight
    /// tile at a time into its staged tile.
    fn bf16_operands(&self) -> bool {
        true
    }

    /// Rounded on the device. A host tensor is [`OjasError::Placement`], as
    /// for every Metal op; nothing is downloaded.
    fn to_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "to_bf16";
        match tensor.dtype() {
            DType::Bf16 => return Ok(tensor.clone()),
            DType::F32 => {}
            got => {
                return Err(OjasError::Dtype {
                    op: OP,
                    expected: DType::F32,
                    got,
                })
            }
        }
        let x = self.f32(OP, tensor)?;
        let bytes = x.n.checked_mul(2).ok_or_else(|| overflow(OP))?;
        let reservation = self.budget.try_reserve(bytes as u64)?;
        let nb = match self.link.call(Cmd::ToBf16 { x })? {
            Reply::Bufs(mut v) if v.len() == 1 => v.remove(0),
            other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
        };
        let buf: Arc<dyn DeviceBuffer> = Arc::new(MetalBuffer {
            id: nb.id,
            len: nb.bytes,
            link: Arc::clone(&self.link),
            ids: None,
        });
        Tensor::from_device_reserved(buf, tensor.shape(), DType::Bf16, reservation)
    }

    /// Widened exactly on the device. A host tensor is
    /// [`OjasError::Placement`], as for every Metal op.
    fn to_f32(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "to_f32";
        match tensor.dtype() {
            DType::F32 => return Ok(tensor.clone()),
            DType::Bf16 => {}
            got => {
                return Err(OjasError::Dtype {
                    op: OP,
                    expected: DType::Bf16,
                    got,
                })
            }
        }
        let x = self.arg(OP, tensor, DType::Bf16)?;
        self.one(OP, tensor.shape(), 0, Cmd::ToF32 { x })
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

    /// tessl's `bert::layer_norm`, which needs `eps > 0` (the CPU also takes
    /// 0 for a row with variance).
    fn layer_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "layer_norm_forward";
        let dims = layer_norm_forward_dims(input, weight, bias, eps)?;
        if eps <= 0.0 {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: format!("tessl's LayerNorm needs eps > 0, got {eps}"),
            });
        }
        let (x, w, b) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
        );
        let (rows, dim) = gated_rms_geom(OP, &dims)?;
        let copies = copied_elems(OP, ReadAs::Whole, &[x, w, b])?;
        let cmd = Cmd::LayerNorm {
            x,
            w,
            b,
            rows,
            dim,
            eps,
        };
        self.one(OP, input.shape(), copies, cmd)
    }

    /// Permute to channels-last, [`Self::layer_norm_forward`], and back; a
    /// rank-2 `[N, C]` input is already channels-last.
    fn channel_layer_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let dims = channel_layer_norm_forward_dims(input, weight, bias, eps)?;
        let rank = input.shape().len();
        if dims.spatial == 1 {
            let flat = input.reshape(&[dims.batch, dims.channels])?;
            let out = self.layer_norm_forward(&flat, weight, bias, eps)?;
            return out.reshape(input.shape());
        }
        let mut to_last = vec![0usize];
        to_last.extend(2..rank);
        to_last.push(1);
        let mut back = vec![0usize; rank];
        for (out_axis, &in_axis) in to_last.iter().enumerate() {
            back[in_axis] = out_axis;
        }
        let moved = self.permute(input, &to_last)?;
        let normed = self.layer_norm_forward(&moved, weight, bias, eps)?;
        self.permute(&normed, &back)
    }

    /// tessl's `bert::bias_gelu_erf` with a zero bias.
    fn gelu_erf_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "gelu_erf_forward";
        gelu_erf_forward_dims(input)?;
        let x = self.f32(OP, input)?;
        u32_dim(OP, x.n)?;
        // The one-element zero bias.
        self.one(OP, input.shape(), 1, Cmd::GeluErf { x })
    }

    /// [`Self::linear_forward`] and then [`Self::broadcast_add_forward`] of
    /// the bias: the CPU's definition, with the product held beside the sum.
    fn linear_bias_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, OjasError> {
        linear_bias_forward_dims(input, weight, bias)?;
        let product = self.linear_forward(input, weight)?;
        self.broadcast_add_forward(&product, bias)
    }

    fn broadcast_add_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "broadcast_add_forward";
        let dims = broadcast_add_forward_dims(a, b)?;
        self.broadcast(OP, false, a, b, &dims)
    }

    fn broadcast_mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "broadcast_mul_forward";
        let dims = broadcast_mul_forward_dims(a, b)?;
        self.broadcast(OP, true, a, b, &dims)
    }

    fn scale_forward(&self, input: &Tensor, scale: f32) -> Result<Tensor, OjasError> {
        const OP: &str = "scale_forward";
        scale_forward_dims(input, scale)?;
        let x = self.f32(OP, input)?;
        u32_dim(OP, x.n)?;
        self.one(OP, input.shape(), 0, Cmd::ScaleFwd { x, scale })
    }

    fn softmax_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "softmax_forward";
        let dims = softmax_forward_dims(input)?;
        let x = self.f32(OP, input)?;
        let (rows, dim) = gated_rms_geom(OP, &dims)?;
        self.one(OP, input.shape(), 0, Cmd::Softmax { x, rows, dim })
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
        // tessl's sq_norm scratch, reserved before the worker builds the plan:
        // a 10-word slot per gradient, the parts buffer (two words, sum and
        // shift, per piece: `SqNormPlan::parts_words`, which the worker
        // allocates by), and the six result words.
        let pieces = args.iter().map(|a| norm_chunks(a.n)).sum::<usize>();
        let _stats = self.reserve(OP, 10 * args.len() + 2 * pieces + SQ_OUT_WORDS)?;
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
            match self.link.call(Cmd::Scale {
                op: OP,
                grads: args,
                scale,
            })? {
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
    /// runs under a budget far below `N * V * 4` bytes. A `Bf16` weight stays
    /// at 2 bytes: each `[cols, d]` tile is widened on the device into the
    /// one staged tile that scratch charges, never the whole table.
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
        grads: CeGrads,
    ) -> Result<LinearCe, OjasError> {
        const OP: &str = "linear_cross_entropy_mean";
        let dims = linear_ce_dims(input, weight, targets, chunk)?;
        let x = self.f32(OP, input)?;
        let w_half = weight.dtype() == DType::Bf16;
        let w = self.arg(OP, weight, weight.dtype())?;
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
            grads,
            w_half,
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
        // The device returns the loss, then the input's gradient and the
        // weight's, each only when asked for (`Worker::linear_ce`).
        let mut shapes: Vec<&[usize]> = vec![&[]];
        if grads.input {
            shapes.push(input.shape());
        }
        if grads.weight {
            shapes.push(weight.shape());
        }
        let mut out = self.outputs(OP, &shapes, scratch, cmd)?.into_iter();
        let missing = || metal_err(format!("{OP}: missing outputs"));
        let loss = out.next().ok_or_else(missing)?;
        let grad_input = grads
            .input
            .then(|| out.next().ok_or_else(missing))
            .transpose()?;
        let grad_weight = grads
            .weight
            .then(|| out.next().ok_or_else(missing))
            .transpose()?;
        Ok(LinearCe {
            loss,
            grad_input,
            grad_weight,
        })
    }

    /// Validated by [`cached_attention_dims`] and the Metal head-dim limit
    /// before anything reaches the device. A NaN or infinity in `q`, or in
    /// either cache at a position some query reads (below `kv_len`, inside
    /// its window), is [`OjasError::NonFinite`] at the next sync point;
    /// other slots are not read.
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
        window: Option<usize>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cached_attention_forward";
        let dims = cached_attention_dims(q, k_cache, v_cache, kv_len, window)?;
        let d = u32_dim(OP, dims.head_dim)?;
        refuse_unsupported_metal_head_dim(BackendId::Metal, d)?;
        let (qa, ka, va) = (
            self.f32(OP, q)?,
            self.f32(OP, k_cache)?,
            self.f32(OP, v_cache)?,
        );
        u32_dim(OP, product(OP, &[dims.new, dims.heads])?)?;
        let (tq, kv_len) = (u32_dim(OP, dims.new)?, u32_dim(OP, kv_len)?);
        let window = match dims.window {
            Some(w) => u32_dim(OP, w)?,
            None => 0,
        };
        // A decode the device splits writes per-split partials for the merge.
        let span = kv_len - cached_attn_origin(tq, kv_len, window);
        let rows = product(OP, &[dims.batch, dims.new, dims.heads])?;
        let part =
            cached_attn_part(rows, span as usize, dims.head_dim).ok_or_else(|| overflow(OP))?;
        self.one(
            OP,
            q.shape(),
            part,
            Cmd::CachedAttn {
                q: qa,
                k: ka,
                v: va,
                batch: u32_dim(OP, dims.batch)?,
                tq,
                heads: u32_dim(OP, dims.heads)?,
                kv_heads: u32_dim(OP, dims.kv_heads)?,
                d,
                cap: u32_dim(OP, dims.capacity)?,
                kv_len,
                window,
            },
        )
    }

    /// Validated by [`kv_cache_write_dims`] (so `Tn > Tcap` is refused
    /// before any dispatch), then written in place, position `at + t` in
    /// slot `(at + t) % Tcap`. A shared `cache` is
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
            at: u32_dim(OP, at % dims.capacity)?,
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

    /// In place on the device when `grad` solely owns its buffer, so a
    /// frozen buffer is refused ([`Self::freeze_checked`]); a shared `grad`
    /// is replaced by the product, as the trait default does, and the buffer
    /// it shared is not written.
    fn scale_grad(&self, grad: &mut Tensor, scale: f32) -> Result<(), OjasError> {
        const OP: &str = "scale_grad";
        if !scale.is_finite() {
            return Err(OjasError::NonFinite { op: OP });
        }
        let g = self.f32(OP, grad)?;
        if self.unique(OP, grad).is_ok() {
            return match self.link.call(Cmd::Scale {
                op: OP,
                grads: vec![g],
                scale,
            })? {
                Reply::Done => Ok(()),
                other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
            };
        }
        let host = Tensor::from_f32(&[scale], &[1], self.budget())?;
        let scalar = self.upload(&host)?;
        let full = ojas_core::broadcast_scalar(self, &scalar, grad.shape())?;
        *grad = self.mul_forward(grad, &full)?;
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

    /// AdamW runs in place with no scratch; Muon reserves
    /// [`muon_scratch_elems`] f32 values, the figure its step charges.
    fn optimizer_scratch_bytes(
        &self,
        kind: OptimizerKind,
        rows: usize,
        cols: usize,
    ) -> Result<Option<u64>, OjasError> {
        const OP: &str = "optimizer_scratch_bytes";
        match kind {
            OptimizerKind::AdamW => Ok(Some(0)),
            OptimizerKind::MuonNs5 => muon_scratch_elems(OP, rows, cols)?
                .checked_mul(4)
                .map(|b| Some(b as u64))
                .ok_or_else(|| overflow(OP)),
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
        let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
        let alpha = (-config.lr * scale) as f32;
        let decay = if config.weight_decay == 0.0 {
            1.0
        } else {
            (1.0 - config.lr * config.weight_decay) as f32
        };
        self.unique(OP, param)?;
        self.unique(OP, momentum)?;
        let _scratch = self.reserve(OP, muon_scratch_elems(OP, rows, cols)?)?;
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
            bf16: config.ns5 == Ns5Precision::Bf16,
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
            op,
            x: xa,
            cos: ca,
            sin: sa,
            rows,
            dim,
            rotary: dim,
            mode,
            backward,
        },
    )
}

/// Partial rope of validated `dims`: the leading `rotary` of each head
/// rotate on `ojas_rope`, the rest are copied.
fn rope_partial(
    be: &MetalBackend,
    op: &'static str,
    dims: &PartialRopeDims,
    [x, cos, sin]: [&Tensor; 3],
    backward: bool,
) -> Res<Tensor> {
    let (xa, ca, sa) = (be.f32(op, x)?, be.f32(op, cos)?, be.f32(op, sin)?);
    let ([rows, dim, rotary], mode) = rope_partial_geom(op, dims)?;
    be.one(
        op,
        x.shape(),
        0,
        Cmd::Rope {
            op,
            x: xa,
            cos: ca,
            sin: sa,
            rows,
            dim,
            rotary,
            mode,
            backward,
        },
    )
}

/// Interleaved rope of validated `dims` on `ojas_rope` (mode 2): every pair
/// of adjacent columns rotates, nothing is copied.
fn rope_interleaved(
    be: &MetalBackend,
    op: &'static str,
    dims: &InterleavedRopeDims,
    [x, cos, sin]: [&Tensor; 3],
    backward: bool,
) -> Res<Tensor> {
    let (xa, ca, sa) = (be.f32(op, x)?, be.f32(op, cos)?, be.f32(op, sin)?);
    u32_dim(op, product(op, &[dims.rows, dims.dim])?)?;
    let mode = RopeMode::Interleaved {
        time: u32_dim(op, dims.time)?,
        heads: u32_dim(op, dims.heads)?,
    };
    let dim = u32_dim(op, dims.dim)?;
    be.one(
        op,
        x.shape(),
        0,
        Cmd::Rope {
            op,
            x: xa,
            cos: ca,
            sin: sa,
            rows: u32_dim(op, dims.rows)?,
            dim,
            rotary: dim,
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

    use ojas_core::METAL_GDN_KEY_DIM;

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

    /// The memory query sees an upload, survives a poisoned backend, and
    /// gives `ResourcePlan` a Metal room that, on Apple silicon, is never
    /// past the shared budget.
    #[test]
    fn memory_probe_tracks_residency_and_plans_one_shared_budget() {
        use ojas_device::{
            probe_system, Device, MemoryArchitecture, MemoryProbe, MemoryReport, ResourcePlan,
            ResourcePolicy,
        };
        watchdog(120, || {
            let host = Budget::new(1 << 30);
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let before = m.memory().expect("memory");
            assert!(before.recommended_working_set >= 1 << 30, "{before:?}");
            let vals = vec![1.0f32; 4 << 20];
            let x = m
                .upload(&Tensor::from_f32(&vals, &[4 << 20], &host).expect("host"))
                .expect("upload");
            let after = m.memory().expect("memory");
            assert_eq!(
                after.recommended_working_set,
                before.recommended_working_set
            );
            // `allocated` is the device's process-wide figure, and other
            // tests in this binary allocate and free beside this one, so a
            // before/after delta is not this test's to assert. A lower bound
            // is: while `x` lives, its 16 MiB are part of the figure.
            assert!(
                after.allocated >= 16 << 20,
                "a live 16 MiB upload is not counted: {after:?}"
            );
            // The peak covers the live upload, survives its free, and
            // restarts from what is allocated at a reset (other tests move
            // the process-wide figure too, so only bounds are asserted).
            assert!(after.peak_allocated >= after.allocated, "{after:?}");
            let y = m
                .upload(&Tensor::from_f32(&vals, &[4 << 20], &host).expect("host"))
                .expect("second upload");
            m.sync().expect("sync");
            drop(y);
            m.trim_pool().expect("trim");
            let held = m.memory_and_reset_peak().expect("memory and reset");
            assert!(held.peak_allocated >= 32 << 20, "{held:?}");
            let reset = m.memory().expect("memory after reset");
            assert!(reset.peak_allocated >= reset.allocated, "{reset:?}");
            // Concurrent queries from several threads all answer.
            thread::scope(|s| {
                for _ in 0..8 {
                    s.spawn(|| assert!(m.memory().is_ok()));
                }
            });

            let profile = probe_system();
            let mut policy = ResourcePolicy::new(8 << 30);
            policy.devices = vec![Device::Metal, Device::Cpu];
            let plan = ResourcePlan::derive(&policy, &profile, &[after]);
            assert_eq!(plan.device_memory[0], after.memory_bytes());
            // tessl runs on Apple silicon only, where the device reports
            // unified memory and the host profile never contradicts it. The
            // device's answer alone puts Metal inside the shared budget.
            assert!(after.has_unified_memory, "{after:?}");
            assert_eq!(after.architecture(), MemoryArchitecture::Unified);
            assert_ne!(profile.architecture, MemoryArchitecture::Discrete);
            assert!(plan.shared_budget);
            assert!(plan.device_shares_host[0]);
            match plan.device_room[0] {
                MemoryReport::Known(room) => assert!(room <= plan.budget_bytes),
                MemoryReport::Unknown => panic!("a shared device has a room"),
            }

            let injected = m.link.call(Cmd::InjectPanic);
            assert!(injected.is_err());
            assert!(matches!(m.silu_forward(&x), Err(OjasError::Poisoned)));
            let poisoned = m.memory().expect("memory after poison");
            assert_eq!(
                poisoned.recommended_working_set,
                before.recommended_working_set
            );
            drop(x);
            drop(m);
        });
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
        let (q, k, w) = (
            t(&m, 64 * 16, &[64, 16]),
            t(&m, 64 * 16, &[64, 16]),
            t(&m, 16, &[16]),
        );
        let fused = m.rms_pair("rms_norm_forward", [&q, &w], [&k, &w], None, 1e-6);
        assert_eq!(fused.expect("fwd").map(|v| v.len()), Some(2));
        let fused = m.rms_pair(
            "rms_norm_backward",
            [&q, &w],
            [&k, &w],
            Some([&q, &k]),
            1e-6,
        );
        assert_eq!(fused.expect("bwd").map(|v| v.len()), Some(4));
        let bad_w = t(&m, 17, &[17]);
        let refused = m.rms_pair("rms_norm_forward", [&q, &w], [&k, &bad_w], None, 1e-6);
        assert!(
            matches!(
                &refused,
                Err(OjasError::Shape {
                    op: "rms_norm_forward",
                    ..
                })
            ),
            "{refused:?}"
        );
        let host_k = Tensor::from_f32(&vec![0.25; 64 * 16], &[64, 16], &host).expect("host");
        let refused = m.rms_pair("rms_norm_forward", [&q, &w], [&host_k, &w], None, 1e-6);
        assert!(
            matches!(&refused, Err(OjasError::Placement { .. })),
            "{refused:?}"
        );
        assert_eq!(pending(&m), None);
        // Inputs, then room for q's output and k's but not both sides' scratch.
        let inputs = 4 * (3 * 64 * 16 + 16) as u64;
        let outs = 2 * 4 * (64 * 16 + 16) as u64;
        let scratch = 4 * (64 + 16) as u64;
        let tight = MetalBackend::new(Budget::new(inputs + outs + scratch)).expect("Metal");
        let (q, k, w) = (
            t(&tight, 64 * 16, &[64, 16]),
            t(&tight, 64 * 16, &[64, 16]),
            t(&tight, 16, &[16]),
        );
        let g = t(&tight, 64 * 16, &[64, 16]);
        let declined = tight.rms_pair(
            "rms_norm_backward",
            [&q, &w],
            [&k, &w],
            Some([&g, &g]),
            1e-6,
        );
        assert!(matches!(declined, Ok(None)), "{declined:?}");
        assert!(tight
            .rms_qk_norm_backward(&q, &k, &w, &w, &g, &g, 1e-6)
            .is_ok());
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
        m.adamw_step(
            &mut p,
            &g,
            &mut m1,
            &mut m2,
            1,
            AdamWConfig::nanolab(1e-3, 0.1),
        )
        .expect("adamw records");
        (p, m1, m2)
    }

    /// A tessl runtime poisoned by a failed or timed-out command buffer is
    /// the device's loss: every later op, sync, upload and read is the
    /// typed [`OjasError::DeviceLost`], not a `Backend` string.
    #[test]
    fn a_poisoned_runtime_surfaces_as_the_typed_device_lost_error() {
        let host = Budget::new(1 << 20);
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let x = m
            .upload(&Tensor::from_f32(&[1.0, 2.0], &[2], &host).expect("host"))
            .expect("upload");
        let _pending = m.silu_forward(&x).expect("records");
        assert!(matches!(m.link.call(Cmd::PoisonRuntime), Ok(Reply::Done)));
        let fresh = Tensor::from_f32(&[3.0], &[1], &host).expect("host");
        let results = [
            ("silu_forward", m.silu_forward(&x).map(drop)),
            ("sync", m.sync()),
            ("upload", m.upload(&fresh).map(drop)),
            ("download", m.download(&x).map(drop)),
        ];
        for (name, result) in results {
            assert!(
                matches!(
                    &result,
                    Err(OjasError::DeviceLost {
                        backend: BackendId::Metal,
                        ..
                    })
                ),
                "{name} on a poisoned runtime: {result:?}"
            );
        }
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
        assert!(p
            .to_host(&Budget::new(1 << 20))
            .expect("read")
            .to_f32_vec()
            .expect("f32")
            .iter()
            .all(|&v| v == 0.5));
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
        assert!(
            m.waits() - w0 >= 11,
            "every op should have hit the 1-byte cap: {}",
            m.waits() - w0
        );
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
        let kept: Vec<Tensor> = (0..6)
            .map(|_| m.silu_forward(&big).expect("big op"))
            .collect();
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
        assert!(
            m.waits() - w0 >= 1,
            "the retry should follow a waited commit"
        );
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
        // both its retries (after the recycle, after the trim), after the
        // checks have run.
        tune(&m, None, None, None, (0, 3));
        let r = m.muon_ns5_step(&mut p, &g, &mut mo, MuonNs5Config::nanolab_default());
        assert!(
            matches!(r, Err(OjasError::CapacityExceeded { .. })),
            "{r:?}"
        );
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
        // q's side allocates one output; fail k's output and both its
        // retries.
        tune(&m, None, None, None, (1, 3));
        let r = m.rms_qk_norm_forward(&q, &k, &w, &w, 1e-6);
        assert!(
            matches!(r, Err(OjasError::CapacityExceeded { .. })),
            "{r:?}"
        );
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
        let vals: Vec<f32> = (0..64)
            .map(|i| ((i * 7 + seed as usize * 13) % 29) as f32 / 29.0 - 0.5)
            .collect();
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
                h.to_f32_vec()
                    .expect("f32")
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
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
                assert!(
                    got == want[i],
                    "thread {i}: bits differ from the serial run"
                );
            }
            assert_eq!(pending(&m), None);
        });
    }

    /// The GDN forward charges tessl's per-call prep buffers and the backward
    /// tessl's workspace, both by tessl's own figures, so a tessl layout
    /// change cannot leave the budget short. Each op runs under a fresh
    /// budget whose peak is its outputs plus that scratch: T off the 64-token
    /// chunk, and more than one batch row and head.
    #[test]
    fn gdn_charges_follow_tessls_own_scratch_figures() {
        use tessl::gdn_train::{
            gdn_train_forward_transient_bytes, GdnTrainDims, GdnTrainWorkspace,
        };
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let bytes = |ts: &[&Tensor]| -> u64 {
            ts.iter()
                .map(|t| 4 * t.shape().iter().product::<usize>() as u64)
                .sum()
        };
        for (b, t, h, dv) in [(1usize, 130usize, 2usize, 32usize), (2, 70, 1, 16)] {
            let dk = METAL_GDN_KEY_DIM;
            let rows = b * t * h;
            let q = put(&m, &vec![0.1; rows * dk], &[b, t, h, dk]);
            let k = put(&m, &vec![0.2; rows * dk], &[b, t, h, dk]);
            let v = put(&m, &vec![0.3; rows * dv], &[b, t, h, dv]);
            let g = put(&m, &vec![-0.1; rows], &[b, t, h]);
            let beta = put(&m, &vec![0.5; rows], &[b, t, h]);
            let d_o = put(&m, &vec![0.1; rows * dv], &[b, t, h, dv]);
            let x = GdnInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
                initial_state: None,
            };
            let dims = GdnTrainDims {
                batch: b as u32,
                seq: t as u32,
                heads: h as u32,
                v_dim: dv as u32,
            };

            let fb = Budget::new(1 << 30);
            let f = m
                .with_budget(fb.clone())
                .chunked_gdn_forward(x)
                .expect("forward");
            let prep = gdn_train_forward_transient_bytes(dims, h as u32).div_ceil(4) * 4;
            assert!(prep > 0, "tessl's chunked forward allocates prep buffers");
            assert_eq!(
                fb.peak_bytes(),
                bytes(&[&f.output, &f.final_state, &f.checkpoints]) + prep,
                "forward {b}x{t}x{h}x{dv}: outputs plus tessl's transient bytes"
            );

            let bb = Budget::new(1 << 30);
            let gr = m
                .with_budget(bb.clone())
                .chunked_gdn_backward(x, &f.checkpoints, &d_o, None)
                .expect("backward");
            let ws = GdnTrainWorkspace::bytes_for(dims) as u64;
            assert_eq!(
                bb.peak_bytes(),
                bytes(&[&gr.q, &gr.k, &gr.v, &gr.g, &gr.beta]) + ws,
                "backward {b}x{t}x{h}x{dv}: outputs plus tessl's workspace"
            );
        }
    }

    /// One GDN forward and backward (key dim 128, two value slices, T 130,
    /// an initial state and a final-state gradient), downloaded as bits.
    fn gdn_round(m: &MetalBackend, seed: u32) -> Vec<Vec<u32>> {
        let (b, t, h, dk, dv) = (1usize, 130usize, 2usize, METAL_GDN_KEY_DIM, 32usize);
        let vals = |n: usize, k: u32, lo: f32, hi: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let x = (i as u32)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(seed ^ k)
                        >> 8;
                    lo + (hi - lo) * (x as f32 / (1u32 << 24) as f32)
                })
                .collect()
        };
        let rows = b * t * h;
        let q = put(m, &vals(rows * dk, 1, -1.0, 1.0), &[b, t, h, dk]);
        let k = put(m, &vals(rows * dk, 2, -1.0, 1.0), &[b, t, h, dk]);
        let v = put(m, &vals(rows * dv, 3, -1.0, 1.0), &[b, t, h, dv]);
        let g = put(m, &vals(rows, 4, -1.0, -0.05), &[b, t, h]);
        let beta = put(m, &vals(rows, 5, 0.1, 0.9), &[b, t, h]);
        let s0 = put(m, &vals(b * h * dk * dv, 6, -0.5, 0.5), &[b, h, dk, dv]);
        let d_o = put(m, &vals(rows * dv, 7, -1.0, 1.0), &[b, t, h, dv]);
        let d_fin = put(m, &vals(b * h * dk * dv, 8, -1.0, 1.0), &[b, h, dk, dv]);
        let x = GdnInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            initial_state: Some(&s0),
        };
        let f = m.chunked_gdn_forward(x).expect("gdn forward");
        let gr = m
            .chunked_gdn_backward(x, &f.checkpoints, &d_o, Some(&d_fin))
            .expect("gdn backward");
        let ds0 = gr.initial_state.expect("ds0");
        let bits = |t: &Tensor| -> Vec<u32> {
            let host = m.download(t).expect("download");
            host.to_f32_vec()
                .expect("f32")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        [
            &f.output,
            &f.final_state,
            &f.checkpoints,
            &gr.q,
            &gr.k,
            &gr.v,
            &gr.g,
            &gr.beta,
            &ds0,
        ]
        .into_iter()
        .map(bits)
        .collect()
    }

    /// The backward's tessl workspace is dropped when its command returns,
    /// before the GPU has run it. tessl recycles such a buffer only after a
    /// waited commit; this attacks that from the other side: an unwaited
    /// commit after every dispatch, a waited one whenever anything was
    /// allocated since the last, other ops allocating between rounds, and
    /// rounds on six threads at once. Every round must keep the bits of an
    /// unstressed run.
    #[test]
    fn gdn_bits_survive_forced_commits_recycling_and_threads() {
        watchdog(600, || {
            // One round: allocate and drop first, so a recycled workspace
            // buffer would be handed to someone else, then the GDN round.
            let body = |m: &MetalBackend| {
                let junk = put(m, &vec![0.25; 1 << 16], &[1 << 16]);
                drop(m.silu_forward(&junk).expect("silu"));
                gdn_round(m, 11)
            };
            let calm = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            tune(&calm, Some(usize::MAX), None, None, (0, 0));
            let before = calm.waits();
            let want = body(&calm);
            let calm_round = calm.waits() - before;
            assert_eq!(pending(&calm), None);

            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            tune(&m, Some(1), Some(1), None, (0, 0));
            let before = m.waits();
            for round in 0..12 {
                assert!(body(&m) == want, "round {round}: bits changed");
            }
            // The same body waits a fixed number of times untuned (each
            // download waits). Only the tune's 1-byte cap can add waits, so
            // equality would mean the stress never happened.
            let forced = m.waits() - before;
            assert!(
                forced > 12 * calm_round,
                "{forced} waited commits over 12 tuned rounds; one untuned round makes {calm_round}"
            );
            assert_eq!(pending(&m), None);
            let start = Arc::new(std::sync::Barrier::new(6));
            let handles: Vec<_> = (0..6)
                .map(|_| {
                    let (m, start) = (m.clone(), Arc::clone(&start));
                    thread::spawn(move || {
                        start.wait();
                        (0..3).map(|_| gdn_round(&m, 11)).collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in handles {
                for got in h.join().expect("gdn thread") {
                    assert!(got == want, "a concurrent round changed bits");
                }
            }
            assert_eq!(pending(&m), None);
        });
    }

    /// An allocation that fails as an exhausted device is retried after a
    /// waited commit, at every allocation the GDN round makes; a failure
    /// that persists is `CapacityExceeded`, leaves no fault pending and
    /// returns every budget charge.
    #[test]
    fn gdn_survives_or_cleanly_refuses_failed_allocations() {
        watchdog(600, || {
            let calm = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let want = gdn_round(&calm, 12);
            for skip in 0..24u32 {
                let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
                tune(&m, None, None, None, (skip, 1));
                assert!(
                    gdn_round(&m, 12) == want,
                    "skip {skip}: a retried allocation changed bits"
                );
                assert_eq!(pending(&m), None);
            }
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let host = Budget::new(1 << 24);
            let up = |n: usize, shape: &[usize]| {
                m.upload(&Tensor::from_f32(&vec![0.5; n], shape, &host).expect("host"))
                    .expect("upload")
            };
            let (t, dk, dv) = (70usize, METAL_GDN_KEY_DIM, 16usize);
            let q = up(t * dk, &[1, t, 1, dk]);
            let v = up(t * dv, &[1, t, 1, dv]);
            let g = up(t, &[1, t, 1]);
            let base = m.budget().live_bytes().expect("live");
            tune(&m, None, None, None, (0, u32::MAX));
            let r = m.chunked_gdn_forward(GdnInputs {
                q: &q,
                k: &q,
                v: &v,
                g: &g,
                beta: &g,
                initial_state: None,
            });
            assert!(
                matches!(r, Err(OjasError::CapacityExceeded { .. })),
                "{r:?}"
            );
            tune(&m, None, None, None, (0, 0));
            assert_eq!(pending(&m), None);
            assert_eq!(m.budget().live_bytes().expect("live"), base);
        });
    }

    /// The pool cache is capped from the budget when the backend opens (a
    /// quarter, at most 1 GiB; tessl's default is 2 GiB whatever the
    /// budget), the probe reports that cap, and a trim recycles first,
    /// keeps the cap, and leaves the backend computing.
    #[test]
    fn the_pool_cache_is_capped_from_the_budget_and_trims() {
        use ojas_device::{MemoryProbe, MemoryReport};
        watchdog(120, || {
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let mem = m.memory().expect("memory");
            assert_eq!(mem.pool_cache_cap, 1 << 28, "{mem:?}");
            assert_eq!(mem.pool_cache_bytes(), MemoryReport::Known(1 << 28));
            let x = put(&m, &vec![0.5; 1 << 20], &[1 << 20]);
            drop(m.silu_forward(&x).expect("silu"));
            let before = m.wait_counts().recycle;
            m.trim_pool().expect("trim");
            assert_eq!(m.wait_counts().recycle, before + 1, "a trim recycles first");
            assert_eq!(m.memory().expect("memory").pool_cache_cap, 1 << 28);
            // A session planned down (capi rebinds with `with_budget`) keeps
            // the cap its device enforces, which is the cap the plan set
            // aside when it chose the smaller budget.
            let planned = m.with_budget(Budget::new(1 << 20));
            assert_eq!(planned.memory().expect("memory").pool_cache_cap, 1 << 28);
            let y = m.silu_forward(&x).expect("silu after a trim");
            assert_eq!(y.shape(), x.shape());
            assert_eq!(pending(&m), None);
            let big = MetalBackend::new(Budget::new(16 << 30)).expect("Metal device");
            assert_eq!(big.memory().expect("memory").pool_cache_cap, 1 << 30);
        });
    }

    /// The cost of the pool cap a backend now opens with (a quarter of its
    /// budget, at most 1 GiB) against tessl's 2 GiB default, at the case it
    /// can matter: each step frees more temporaries than the cap holds, so
    /// the capped pool misses and allocates afresh where the default reuses.
    /// A 1 GiB budget (cap 256 MiB); each step makes eight 64 MiB outputs
    /// (512 MiB) and drops them before the next. Interleaved A/B rounds,
    /// min of the steps in each. A measurement, not a pass/fail: run with
    /// `--ignored --nocapture` under the machine lock.
    #[test]
    #[ignore]
    fn bench_the_pool_cap_against_the_tessl_default() {
        use std::time::{Duration, Instant};
        const N: usize = 16 << 20;
        let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
        let opened = m.memory().expect("memory").pool_cache_cap as usize;
        let x = put(&m, &vec![0.5; N], &[N]);
        let tiny = put(&m, &[0.5; 64], &[64]);
        let step = |m: &MetalBackend| {
            let t = Instant::now();
            let outs: Vec<Tensor> = (0..8).map(|_| m.silu_forward(&x).expect("silu")).collect();
            m.sync().expect("sync");
            let took = t.elapsed();
            // Freed buffers reach tessl's pool only at a waited commit, so
            // one more (untimed) puts these back before the next step asks.
            drop(outs);
            drop(m.silu_forward(&tiny).expect("silu"));
            m.sync().expect("sync");
            took
        };
        let set = |bytes: usize| {
            let r = m.link.call(Cmd::SetPoolCap { bytes });
            assert!(matches!(r, Ok(Reply::Done)), "{r:?}");
        };
        let (mut capped, mut default) = (Duration::MAX, Duration::MAX);
        for round in 0..6 {
            for (cap, best) in [(opened, &mut capped), (2 << 30, &mut default)] {
                // Each side starts from an empty pool and warms it once.
                set(0);
                set(cap);
                step(&m);
                for _ in 0..5 {
                    *best = (*best).min(step(&m));
                }
            }
            eprintln!(
                "round {round}: cap {} MiB {:.2} ms, tessl default 2048 MiB {:.2} ms",
                opened >> 20,
                capped.as_secs_f64() * 1e3,
                default.as_secs_f64() * 1e3
            );
        }
        eprintln!(
            "pool cap {} MiB vs 2048 MiB, min step: {:.2} ms vs {:.2} ms ({:.2}x)",
            opened >> 20,
            capped.as_secs_f64() * 1e3,
            default.as_secs_f64() * 1e3,
            capped.as_secs_f64() / default.as_secs_f64()
        );
        assert_eq!(pending(&m), None);
    }

    /// An allocation still refused after the recycling commit is tried once
    /// more with the pool cache released. A refusal past that reports the
    /// device's working set as the cap and what the device holds as live:
    /// before, `live` was always 0.
    #[test]
    fn a_failed_allocation_retries_after_a_trim_and_reports_live_bytes() {
        watchdog(120, || {
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let x = put(&m, &vec![0.5; 1 << 16], &[1 << 16]);
            let before = m.wait_counts().recycle;
            tune(&m, None, None, None, (0, 2));
            let y = m.silu_forward(&x).expect("the third try allocates");
            assert_eq!(m.wait_counts().recycle, before + 1);
            drop(y);
            tune(&m, None, None, None, (0, u32::MAX));
            let ws = m.memory().expect("memory").recommended_working_set;
            let live_before = m.budget().live_bytes().expect("live");
            match m.silu_forward(&x) {
                Err(OjasError::CapacityExceeded {
                    requested,
                    cap,
                    live,
                }) => {
                    assert_eq!(requested, 4 << 16);
                    assert_eq!(cap, ws);
                    assert!(live >= 4 << 16, "live {live}: x is held on the device");
                }
                other => panic!("{other:?}"),
            }
            tune(&m, None, None, None, (0, 0));
            assert_eq!(pending(&m), None);
            assert_eq!(m.budget().live_bytes().expect("live"), live_before);
        });
    }

    /// A `U32` upload behind recorded work allocates through the same
    /// retries: a refusal is retried after a recycling commit, never as an
    /// upload wait, and a refusal that persists is `CapacityExceeded` that
    /// returns the budget charge.
    #[test]
    fn an_id_upload_behind_recorded_work_retries_and_refuses_like_any_allocation() {
        watchdog(120, || {
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let x = put(&m, &[0.5; 1024], &[1024]);
            let want: Vec<u32> = (0..1u32 << 18)
                .map(|i| i.wrapping_mul(2_654_435_761))
                .collect();
            let ids = Tensor::from_u32(&want, &[want.len()], &Budget::new(1 << 30)).expect("host");
            let _recorded = m.silu_forward(&x).expect("recorded");
            let before = m.wait_counts();
            tune(&m, None, None, None, (0, 1));
            let d = m.upload(&ids).expect("the retry allocates");
            let after = m.wait_counts();
            assert_eq!(after.recycle, before.recycle + 1);
            assert_eq!(after.upload, before.upload);
            let got = m.download(&d).expect("download");
            assert_eq!(got.u32_slice().expect("u32"), want.as_slice());

            let _recorded = m.silu_forward(&x).expect("recorded");
            tune(&m, None, None, None, (0, u32::MAX));
            let live_before = m.budget().live_bytes().expect("live");
            let before = m.wait_counts();
            match m.upload(&ids) {
                Err(OjasError::CapacityExceeded { requested, .. }) => {
                    assert_eq!(requested, 4 << 18);
                }
                other => panic!("{other:?}"),
            }
            assert_eq!(m.wait_counts().upload, before.upload);
            tune(&m, None, None, None, (0, 0));
            assert_eq!(pending(&m), None);
            assert_eq!(m.budget().live_bytes().expect("live"), live_before);
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
            let too_long = up(5 * 2 * 3, &[1, 5, 2, 3]);
            let injected = m.link.call(Cmd::InjectPanic);
            assert!(injected.is_err(), "{injected:?}");
            // The cache is a ring: only more positions than slots, or an
            // `at + Tn` that overflows, is refused.
            for (what, src, at) in [
                ("Tn > Tcap", &too_long, 0usize),
                ("at + Tn overflows", &src, usize::MAX),
            ] {
                let r = m.kv_cache_write(&mut cache, src, at);
                assert!(
                    matches!(r, Err(OjasError::OutOfRange { .. })),
                    "{what}: {r:?}"
                );
            }
            // In range, wrapping or not, reaches the poisoned device.
            for at in [2usize, 3] {
                let r = m.kv_cache_write(&mut cache, &src, at);
                assert!(matches!(r, Err(OjasError::Poisoned)), "at {at}: {r:?}");
            }
        });
    }

    fn vision_dims() -> AttentionDims {
        AttentionDims {
            batch: 1,
            heads: 16,
            kv_heads: 16,
            q_len: 5184,
            kv_len: 5184,
            head_dim: 64,
            causal: false,
            window: None,
            bias: None,
            key_lens: false,
        }
    }

    #[test]
    fn only_plain_attention_at_head_dim_32_or_64_with_tq_at_most_tk_routes_to_encoder_attn() {
        let base = vision_dims();
        assert!(routes_to_encoder_attn(&base, true));
        assert!(
            routes_to_encoder_attn(
                &AttentionDims {
                    head_dim: 32,
                    heads: 8,
                    kv_heads: 8,
                    ..base
                },
                true
            ),
            "head dim 32 routes"
        );
        // Cross-attention pads Q: the memory encoder's one-frame and
        // two-frame lengths route (F = 2 spans ratios 2.003 to 2.049), up to a
        // Tk of 2.25 Tq; three frames (F = 3, ratio 3.01) do not.
        for kv_len in [5200usize, 10400, 10624] {
            assert!(
                routes_to_encoder_attn(
                    &AttentionDims {
                        head_dim: 32,
                        heads: 8,
                        kv_heads: 8,
                        kv_len,
                        ..base
                    },
                    true
                ),
                "Tq 5184 over Tk {kv_len} routes"
            );
        }
        // The bound itself (2.25 Tq) routes; one key more does not.
        for (kv_len, routes) in [(144usize, true), (145, false)] {
            assert_eq!(
                routes_to_encoder_attn(
                    &AttentionDims {
                        q_len: 64,
                        kv_len,
                        ..base
                    },
                    true
                ),
                routes,
                "Tq 64 over Tk {kv_len}"
            );
        }
        assert!(!routes_to_encoder_attn(&base, false), "no TensorOps");
        let bias = ojas_core::AttentionBias {
            batch: 1,
            heads: 1,
            queries: 1,
        };
        let stays = [
            (
                "causal",
                AttentionDims {
                    causal: true,
                    ..base
                },
            ),
            (
                "window",
                AttentionDims {
                    window: Some(8),
                    ..base
                },
            ),
            (
                "bias",
                AttentionDims {
                    bias: Some(bias),
                    ..base
                },
            ),
            (
                "key lengths",
                AttentionDims {
                    key_lens: true,
                    ..base
                },
            ),
            ("Tq > Tk", AttentionDims { kv_len: 77, ..base }),
            (
                "an empty query",
                AttentionDims {
                    q_len: 0,
                    kv_len: 0,
                    ..base
                },
            ),
            (
                "an empty query at head dim 32",
                AttentionDims {
                    q_len: 0,
                    kv_len: 0,
                    head_dim: 32,
                    heads: 8,
                    kv_heads: 8,
                    ..base
                },
            ),
            (
                "Tk past the pad bound (ratio 2.5)",
                AttentionDims {
                    q_len: 64,
                    kv_len: 160,
                    ..base
                },
            ),
            (
                "three memory frames (ratio 3.01)",
                AttentionDims {
                    head_dim: 32,
                    heads: 8,
                    kv_heads: 8,
                    kv_len: 15616,
                    ..base
                },
            ),
            (
                "grouped heads",
                AttentionDims {
                    kv_heads: 8,
                    ..base
                },
            ),
            (
                "head dim 16",
                AttentionDims {
                    head_dim: 16,
                    ..base
                },
            ),
            (
                "head dim 128",
                AttentionDims {
                    head_dim: 128,
                    ..base
                },
            ),
            (
                "head dim 256",
                AttentionDims {
                    head_dim: 256,
                    ..base
                },
            ),
            (
                "a plane past i32",
                AttentionDims {
                    q_len: 1 << 20,
                    kv_len: 1 << 20,
                    heads: 64,
                    kv_heads: 64,
                    ..base
                },
            ),
            (
                // Each head's plane fits i32, the whole tensor (5.4e9
                // elements) does not fit u32 indexing.
                "more than u32 elements",
                AttentionDims {
                    batch: 1024,
                    ..base
                },
            ),
        ];
        for (what, d) in stays {
            assert!(
                !routes_to_encoder_attn(&d, true),
                "{what} must keep the general kernel"
            );
        }
    }

    /// Every key range of the split-K kernel is a whole number of tiles, none
    /// is empty, and together they cover `Tk`: `attn_gen` refuses a launch
    /// whose ranges would drop keys, and this is the only producer of one.
    #[test]
    fn the_split_ranges_cover_every_key_with_none_empty() {
        for d in [16u32, 32] {
            for bh in [1u32, 2, 16, 64, 1000] {
                for q_len in [1u32, 4, 127, 128, 129, 201, 5184] {
                    for kv_len in [1u32, 63, 64, 65, 128, 1000, 4096, 5184, 20000, 36544] {
                        let g = AttnGeom {
                            bh,
                            heads: 1,
                            rep: 1,
                            q_len,
                            kv_len,
                            d,
                            causal: false,
                            window: 0,
                            bias: None,
                            has_lens: false,
                            splits: 0,
                            per_split: 0,
                        };
                        let (splits, per_split) = xattn_splits(&g).expect("tiled");
                        let at = format!("bh {bh} Tq {q_len} Tk {kv_len} d {d}");
                        assert!((1..=XATTN_MAX_SPLITS).contains(&splits), "{at}: {splits}");
                        assert_eq!(per_split % XATTN_TILE, 0, "{at}");
                        assert!(
                            u64::from(splits) * u64::from(per_split) >= u64::from(kv_len),
                            "{at}: the ranges drop keys"
                        );
                        assert!(
                            u64::from(splits - 1) * u64::from(per_split) < u64::from(kv_len),
                            "{at}: the last range is empty"
                        );
                    }
                }
            }
        }
        let causal = AttnGeom {
            bh: 1,
            heads: 1,
            rep: 1,
            q_len: 8,
            kv_len: 8,
            d: 32,
            causal: true,
            window: 0,
            bias: None,
            has_lens: false,
            splits: 0,
            per_split: 0,
        };
        assert!(xattn_splits(&causal).is_none(), "causal stays scalar");
        assert!(
            xattn_splits(&AttnGeom {
                causal: false,
                d: 64,
                ..causal
            })
            .is_none(),
            "head dim 64 stays scalar"
        );
    }
}
