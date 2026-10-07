//! Device-resident [`Backend`] over wgpu.
//!
//! Shape first (docs/shape-contract.md): every op calls its
//! `ojas_core::shapes` validator before anything else, so a malformed call
//! returns the validator's exact error (dtype, empty axis, rank, shape,
//! count overflow) whatever the placement, values or budget. Placement,
//! contiguity, token-id and target ranges, optimizer scalars and device
//! limits are this backend's and come after it.
//!
//! Upload policy: every op takes tensors that live on this backend's
//! [`WgpuContext`] and returns tensors that stay there. A well-formed host
//! tensor passed to an op is [`OjasError::Placement`] with `found: None`; a
//! tensor of another device or another wgpu context is `Placement` as well.
//! Data moves
//! only through [`Backend::upload`] (contiguous, non-empty) and
//! [`Backend::download`]. A contiguous view with a non-zero `byte_offset` is
//! compacted on the device; a non-contiguous view is [`OjasError::Shape`]. A
//! U32 upload keeps a host copy of its ids so token ids and targets are
//! range-checked, and the cross-entropy valid count formed, with no readback.
//!
//! Ops record into one shared encoder and do not wait. Nothing is read back
//! until `download`, `clip_grad_norm` (the trait returns the norm) or
//! [`WgpuBackend::sync`].
//!
//! # Non-finite values are reported at the next sync point (the contract)
//!
//! This backend's contract differs from the CPU's on purpose: an op whose
//! kernel produces a non-finite value still returns `Ok`. Detecting it
//! synchronously would cost a device round trip per op, which is the cost
//! the shared encoder exists to avoid. Instead:
//!
//! - the kernel raises its op's bit in a device fault word and keeps going;
//! - the very next [`WgpuBackend::sync`], [`Backend::download`] (of any
//!   tensor) or [`Backend::clip_grad_norm`] returns [`OjasError::NonFinite`]
//!   naming the first op, in recording order on this backend, that faulted
//!   since the previous report. A fault is never lost and is reported at
//!   least once: one caller reports it once, but two threads whose syncs
//!   overlap may both see it;
//! - `clip_grad_norm` reports a pending fault before it scales anything;
//! - `adamw_step` and `muon_ns5_step` decide on the device: if any value of
//!   *this* call is non-finite they write nothing, so a bad step is never
//!   applied in part, and the fault is reported as above.
//!
//! A caller that must not run past a fault calls `sync` after the steps it
//! cares about. The fault word belongs to the context, not to a caller: when
//! threads share one backend, a fault raised by one thread's op is reported
//! once, to whichever thread synchronizes next. Callers that need their
//! faults kept apart open one backend each.
//!
//! The optimizers update in place through [`Tensor::device_buffer_mut`], so
//! a parameter shared with another tensor is refused before anything is
//! written.
//!
//! [`Numerics::Fast`]: reductions are tree- and chunk-ordered, GEMM tiles
//! accumulate in registers, Adam runs in f32 (the CPU reference uses f64),
//! and the Metal HAL may contract multiply-adds. Results match the CPU at a
//! tolerance and are bit-identical across runs on one device.

use std::sync::Arc;

use ojas_core::{
    accumulate_grad_dims, adamw_step_dims, cached_attention_dims, causal_sdpa_backward_dims,
    causal_sdpa_forward_dims, check_adamw, clip_grad_norm_dims, clip_scale, cross_entropy_mean_backward_dims,
    cross_entropy_mean_forward_dims, embedding_backward_dims, embedding_forward_dims,
    kv_cache_write_dims, linear_backward_dims, linear_ce_dims, linear_forward_dims,
    mul_backward_dims, mul_forward_dims, muon_ns5_step_dims, per_head_sigmoid_gate_backward_dims,
    per_head_sigmoid_gate_forward_dims, permute_dims, refuse_bf16_operands, require_ns5,
    residual_add_backward_dims, residual_add_forward_dims, rms_norm_backward_dims,
    rms_norm_forward_dims, rms_qk_norm_backward_dims, rms_qk_norm_forward_dims,
    rope_half_split_backward_dims, rope_half_split_forward_dims, sdpa_scale, silu_backward_dims,
    silu_forward_dims, value_residual_blend_backward_dims, value_residual_blend_forward_dims,
    AdamWConfig, Backend, BackendId, Budget, CeChunk, DType, LinearCe, MuonNs5Config, Ns5Precision,
    Numerics, OjasError, OptimizerKind, PerHeadGateGrad, RmsDims, RopeDims, RopeLayout, SdpaDims,
    Tensor, ValueResidualGrad, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS,
};
use ojas_device::DeviceError;
use ojas_kernels::{
    attention_tiles, cached_attention_splits, fold_grid, gemm_grid, gemm_tile, WgslModule,
    ATTENTION_MAX_HEAD_DIM, GEMM_BIG_TILE,
};

use crate::context::{Job, Kernel, Slot, WgpuBuffer, WgpuContext, FAULT_OPS};

/// One fault bit per op, by index. The fault words hold a 64-bit mask
/// ([`FAULT_OPS`]); `OP_NAMES` may grow to that many, and the build refuses
/// more.
const OP_NAMES: [&str; 33] = [
    "embedding_forward",
    "embedding_backward",
    "linear_forward",
    "linear_backward",
    "rms_norm_forward",
    "rms_norm_backward",
    "rope_half_split_forward",
    "rope_half_split_backward",
    "rms_qk_norm_forward",
    "rms_qk_norm_backward",
    "causal_sdpa_forward",
    "causal_sdpa_backward",
    "per_head_sigmoid_gate_forward",
    "per_head_sigmoid_gate_backward",
    "value_residual_blend_forward",
    "value_residual_blend_backward",
    "silu_forward",
    "silu_backward",
    "mul_forward",
    "mul_backward",
    "residual_add_forward",
    "residual_add_backward",
    "cross_entropy_mean_forward",
    "cross_entropy_mean_backward",
    "clip_grad_norm",
    "adamw_step",
    "permute",
    "muon_ns5_step",
    "accumulate_grad",
    "linear_cross_entropy_mean",
    "cached_attention_forward",
    "kv_cache_write",
    "cast_bf16",
];

const _: () = assert!(
    OP_NAMES.len() as u32 <= FAULT_OPS,
    "more ops than the fault words hold"
);

#[derive(Clone, Copy)]
struct Op(u32);

impl Op {
    fn name(self) -> &'static str {
        OP_NAMES.get(self.0 as usize).copied().unwrap_or("wgpu")
    }

    /// The id its kernels raise: index + 1, so 0 stays "reports nothing".
    fn fault_id(self) -> u32 {
        self.0 + 1
    }
}

const EMBED_F: Op = Op(0);
const EMBED_B: Op = Op(1);
const LINEAR_F: Op = Op(2);
const LINEAR_B: Op = Op(3);
const RMS_F: Op = Op(4);
const RMS_B: Op = Op(5);
const ROPE_F: Op = Op(6);
const ROPE_B: Op = Op(7);
const SDPA_F: Op = Op(10);
const SDPA_B: Op = Op(11);
const GATE_F: Op = Op(12);
const GATE_B: Op = Op(13);
const VR_F: Op = Op(14);
const VR_B: Op = Op(15);
const SILU_F: Op = Op(16);
const SILU_B: Op = Op(17);
const MUL_F: Op = Op(18);
const MUL_B: Op = Op(19);
const ADD_F: Op = Op(20);
const ADD_B: Op = Op(21);
const CE_F: Op = Op(22);
const CE_B: Op = Op(23);
const CLIP: Op = Op(24);
const ADAM: Op = Op(25);
const PERMUTE: Op = Op(26);
const MUON: Op = Op(27);
const ACC: Op = Op(28);
const LCE: Op = Op(29);
const CATTN: Op = Op(30);
const KVW: Op = Op(31);
/// Past the low mask word: its bit is bit 0 of fault word 1.
const CAST: Op = Op(32);

const fn k(module: WgslModule, entry: &'static str, slots: &'static [Slot]) -> Kernel {
    Kernel {
        module,
        entry,
        slots,
    }
}

use Slot::{R, W};
use WgslModule::{CachedAttention, Gemm, Layout, Loss, Norm, Optim, Pointwise, Reduce};

const PERMUTE_K: Kernel = k(Layout, "permute", &[R(2), R(3), W(4)]);
const KV_WRITE: Kernel = k(Layout, "kv_write", &[R(2), W(4), R(5)]);
const CATTN_SPLIT: Kernel = k(CachedAttention, "cattn_split", &[R(2), R(3), R(4), W(5)]);
const CATTN_MERGE: Kernel = k(CachedAttention, "cattn_merge", &[W(6), R(7)]);

const GEMM_NT: Kernel = k(Gemm, "gemm_nt", &[R(2), R(3), W(4)]);
const GEMM_NN: Kernel = k(Gemm, "gemm_nn", &[R(2), R(3), W(4)]);
const GEMM_TN: Kernel = k(Gemm, "gemm_tn", &[R(2), R(3), W(4)]);
const GEMM_NT_BIG: Kernel = k(Gemm, "gemm_nt_big", &[R(2), R(3), W(4)]);
const GEMM_NN_BIG: Kernel = k(Gemm, "gemm_nn_big", &[R(2), R(3), W(4)]);
const GEMM_TN_BIG: Kernel = k(Gemm, "gemm_tn_big", &[R(2), R(3), W(4)]);
const ROUND_BF16: Kernel = k(Pointwise, "round_bf16", &[R(2), W(6)]);
const SILU_FWD: Kernel = k(Pointwise, "silu_fwd", &[R(2), W(6)]);
const SILU_BWD: Kernel = k(Pointwise, "silu_bwd", &[R(2), R(3), W(6)]);
const MUL_FWD: Kernel = k(Pointwise, "mul_fwd", &[R(2), R(3), W(6)]);
const MUL_BWD: Kernel = k(Pointwise, "mul_bwd", &[R(2), R(3), R(4), W(6), W(7)]);
const ADD_FWD: Kernel = k(Pointwise, "add_fwd", &[R(2), R(3), W(6)]);
const ADD_INPLACE: Kernel = k(Pointwise, "add_inplace", &[R(2), W(6)]);
const CHECK_FINITE: Kernel = k(Pointwise, "check_finite", &[R(2)]);
const SCALE_INPLACE: Kernel = k(Pointwise, "scale_inplace", &[W(6)]);
const VR_FWD: Kernel = k(Pointwise, "vr_fwd", &[R(2), R(3), R(4), W(6)]);
const VR_BWD: Kernel = k(Pointwise, "vr_bwd", &[R(2), R(4), W(6), W(7)]);
const GATE_FWD: Kernel = k(Pointwise, "gate_fwd", &[R(2), R(3), R(4), W(6)]);
const GATE_BWD: Kernel = k(Pointwise, "gate_bwd", &[R(2), R(3), R(4), R(5), W(6), W(7)]);
const GATE_FWD_SAVE: Kernel = k(Pointwise, "gate_fwd_save", &[R(2), R(3), R(4), W(6), W(7)]);
const GATE_BWD_SAVED: Kernel = k(Pointwise, "gate_bwd_saved", &[R(2), R(4), R(5), W(6), W(7)]);
const ROPE: Kernel = k(Pointwise, "rope", &[R(2), R(3), R(4), W(6)]);
const SUM_PARTIAL: Kernel = k(Reduce, "sum_partial", &[R(2), R(3), R(4), W(5)]);
const SUM_FINISH: Kernel = k(Reduce, "sum_finish", &[R(2), R(3), W(5)]);
const COL_PARTIAL: Kernel = k(Reduce, "col_partial", &[R(2), R(3), R(4), W(5)]);
const COL_FINISH: Kernel = k(Reduce, "col_finish", &[R(2), W(5)]);
const RMS_FWD: Kernel = k(Norm, "rms_fwd", &[R(2), R(3), W(5)]);
const RMS_BWD: Kernel = k(Norm, "rms_bwd", &[R(2), R(3), R(4), W(5), W(6)]);
const EMBED_FWD: Kernel = k(Loss, "embed_fwd", &[R(2), R(3), W(6)]);
const EMBED_BWD: Kernel = k(Loss, "embed_bwd", &[R(2), R(3), R(4), R(5), W(6)]);
const CE_FWD: Kernel = k(Loss, "ce_fwd", &[R(2), R(3), W(6)]);
const CE_BWD: Kernel = k(Loss, "ce_bwd", &[R(2), R(3), W(6)]);
const LCE_STATS: Kernel = k(Loss, "lce_stats", &[R(2), R(3), W(6)]);
const LCE_FINISH: Kernel = k(Loss, "lce_finish", &[R(2), R(3), W(6)]);
const LCE_GRAD: Kernel = k(Loss, "lce_grad", &[R(2), R(3), W(6)]);
const ABSMAX_PARTIAL: Kernel = k(Optim, "absmax_partial", &[R(2), W(5), W(8)]);
const ABSMAX_FINISH: Kernel = k(Optim, "absmax_finish", &[R(2), W(8)]);
const SUMSQ_PARTIAL: Kernel = k(Optim, "sumsq_partial", &[R(2), W(5), R(9)]);
const CLIP_FINISH: Kernel = k(Optim, "clip_finish", &[R(2), W(8)]);
const ADAM_UPDATE: Kernel = k(Optim, "adam_update", &[R(2), W(5), W(6), W(7), W(8)]);
const MUON_MOMENTUM: Kernel = k(Optim, "muon_momentum", &[R(2), R(3), W(5), W(6)]);
const MUON_NORMALIZE: Kernel = k(Optim, "muon_normalize", &[R(2), W(5), R(9)]);
const MUON_LINCOMB: Kernel = k(Optim, "muon_lincomb", &[R(2), R(3), W(5)]);
const MUON_APPLY: Kernel = k(Optim, "muon_apply", &[R(2), R(3), W(5)]);
const MUON_COMMIT: Kernel = k(Optim, "muon_commit", &[R(2), R(3), W(5), W(6), R(9)]);
const ADAM_COMMIT: Kernel = k(
    Optim,
    "adam_commit",
    &[R(2), R(3), R(4), W(5), W(6), W(7), R(9)],
);

fn attn_kernel(module: WgslModule, entry: &'static str) -> Kernel {
    let slots: &'static [Slot] = match entry {
        "attn_fwd" => &[R(2), R(3), R(4), W(7), W(8)],
        "attn_bwd_dr" => &[R(2), R(5), R(6), W(7)],
        "attn_bwd_dq" => &[R(2), R(3), R(4), R(5), R(6), W(7)],
        _ => &[R(2), R(3), R(4), R(5), R(6), W(7), W(8)],
    };
    k(module, entry, slots)
}

/// One SDPA call: its planes, the compiled module, and the rows one
/// workgroup owns in the forward/stats and in the dQ/dK/dV kernels.
struct SdpaPlan {
    bh: usize,
    time: usize,
    dim: usize,
    /// Query heads per KV head: query plane `bh` reads KV plane `bh / rep`.
    rep: usize,
    /// Keys a query sees, counting itself; `0` is every earlier key. Always
    /// below `time` when set.
    window: usize,
    scale: f32,
    module: WgslModule,
    fwd_rows: usize,
    bwd_rows: usize,
}

impl SdpaPlan {
    /// The attention kernels' words: time, scale bits, `bh * time`, D, rep,
    /// window.
    fn words(&self, op: Op) -> Result<[u32; 6], OjasError> {
        Ok([
            u(op, self.time)?,
            self.scale.to_bits(),
            u(op, product(op, &[self.bh, self.time])?)?,
            u(op, self.dim)?,
            u(op, self.rep)?,
            u(op, self.window)?,
        ])
    }
}

/// Elements one stage-one reduction group covers (`CHUNK` in the WGSL).
const CHUNK: usize = 4096;

fn shape(op: Op, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op: op.name(),
        detail: detail.into(),
    }
}

/// A count that does not fit: [`OjasError::OutOfRange`], the shape
/// contract's variant for every overflow (decision 1).
fn overflow(op: Op, detail: impl Into<String>) -> OjasError {
    OjasError::OutOfRange {
        op: op.name(),
        detail: detail.into(),
    }
}

fn product(op: Op, dims: &[usize]) -> Result<usize, OjasError> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| overflow(op, "shape product overflows"))
    })
}

fn u(op: Op, value: usize) -> Result<u32, OjasError> {
    u32::try_from(value).map_err(|_| OjasError::OutOfRange {
        op: op.name(),
        detail: format!("{value} does not fit in u32"),
    })
}

/// A validated input: a tensor of this context, contiguous and non-empty.
struct In<'t> {
    t: &'t Tensor,
    buf: &'t WgpuBuffer,
    elems: usize,
}

impl In<'_> {
    fn shape(&self) -> &[usize] {
        self.t.shape()
    }
}

/// The buffer a kernel binds for `a`, compacting a view with a non-zero offset.
fn bind(op: Op, job: &mut Job<'_>, a: &In<'_>) -> Result<wgpu::Buffer, OjasError> {
    let raw = a.buf.raw()?;
    let off = a.t.byte_offset();
    if off == 0 {
        return Ok(raw.clone());
    }
    if !off.is_multiple_of(4) {
        return Err(shape(
            op,
            format!("byte_offset {off} is not 4-byte aligned"),
        ));
    }
    let bytes = (a.elems as u64) * 4;
    let dst = job.scratch(bytes)?;
    job.copy(raw, off as u64, &dst, bytes);
    Ok(dst)
}

/// The global count of targets that are not `ignore`, each checked against
/// `vocab` on the host copy. No valid target is `NonFinite`, as in the CPU
/// reference: the mean would divide by zero.
fn valid_targets(op: Op, ids: &[u32], vocab: usize, ignore: Option<u32>) -> Result<u32, OjasError> {
    let mut valid: u32 = 0;
    for (n, &tgt) in ids.iter().enumerate() {
        if ignore == Some(tgt) {
            continue;
        }
        if (tgt as usize) >= vocab {
            return Err(OjasError::OutOfRange {
                op: op.name(),
                detail: format!("target {tgt} at {n} is outside vocab {vocab}"),
            });
        }
        valid = valid.checked_add(1).ok_or_else(|| OjasError::OutOfRange {
            op: op.name(),
            detail: "valid target count overflows".to_string(),
        })?;
    }
    if valid == 0 {
        return Err(OjasError::NonFinite { op: op.name() });
    }
    Ok(valid)
}

/// The raw buffer of a tensor this call may overwrite: sole owner, offset 0.
fn exclusive(op: Op, t: &mut Tensor) -> Result<wgpu::Buffer, OjasError> {
    if t.byte_offset() != 0 {
        return Err(shape(op, "in-place update needs a view starting at byte 0"));
    }
    let dev = t.device_buffer_mut()?;
    let buf = dev
        .as_any()
        .downcast_ref::<WgpuBuffer>()
        .ok_or(OjasError::Placement {
            op: op.name(),
            expected: Some(BackendId::Wgpu),
            found: Some(dev.backend()),
        })?;
    Ok(buf.raw()?.clone())
}

/// Lanes per RMSNorm row: the row length rounded up to a power of two, at
/// most 256, so `256 / width` rows share a workgroup (see `norm.wgsl`).
fn rms_width(dim: usize) -> usize {
    dim.next_power_of_two().min(256)
}

/// A validated, allocated `rms_norm_forward`, ready to record.
struct RmsFwd<'t> {
    xv: In<'t>,
    wv: In<'t>,
    rows: usize,
    dim: usize,
    y: Tensor,
    yb: wgpu::Buffer,
}

/// A validated, allocated `rms_norm_backward`, ready to record.
struct RmsBwd<'t> {
    xv: In<'t>,
    wv: In<'t>,
    gv: In<'t>,
    rows: usize,
    dim: usize,
    gx: Tensor,
    gxb: wgpu::Buffer,
    gw: Tensor,
    gwb: wgpu::Buffer,
}

/// Device-resident GPU backend. See the module docs for the upload policy.
pub struct WgpuBackend {
    ctx: WgpuContext,
    budget: Budget,
}

/// Where a GEMM's operands sit in their buffers, and whether C accumulates.
#[derive(Clone, Copy, Default)]
struct Place {
    offsets: [usize; 3],
    accumulate: bool,
}

#[derive(Clone, Copy)]
enum Mm {
    Nt,
    Nn,
    Tn,
}

impl WgpuBackend {
    pub fn open(budget: Budget) -> Result<Self, DeviceError> {
        Ok(Self::with_context(WgpuContext::open()?, budget))
    }

    /// Run on an existing context (for example one opened with capped limits).
    pub fn with_context(ctx: WgpuContext, budget: Budget) -> Self {
        Self { ctx, budget }
    }

    pub fn context(&self) -> &WgpuContext {
        &self.ctx
    }

    fn faults(&self) -> Result<(), OjasError> {
        let (bits, first) = self.ctx.take_faults();
        if bits == 0 && first == 0 {
            return Ok(());
        }
        // The first op in recording order; the lowest set bit only if the
        // first-op record is somehow absent.
        let index = match first.checked_sub(1) {
            Some(index) => index,
            None => bits.trailing_zeros(),
        };
        // An id the kernels raised that names no op is a host bug, not a
        // value fault; report it rather than a made-up op name.
        if index as usize >= OP_NAMES.len() {
            return Err(OjasError::Backend {
                id: BackendId::Wgpu,
                detail: format!("fault words name op index {index}, past the op table"),
            });
        }
        Err(OjasError::NonFinite {
            op: Op(index).name(),
        })
    }

    /// Placement and contiguity of an operand the op's `ojas_core::shapes`
    /// validator has already accepted, so its dtype, extents and counts are
    /// not checked again here: a tensor of this context, contiguous.
    fn placed<'t>(&self, op: Op, t: &'t Tensor) -> Result<In<'t>, OjasError> {
        let placement = |found| OjasError::Placement {
            op: op.name(),
            expected: Some(BackendId::Wgpu),
            found,
        };
        let dev = t.device_buffer().ok_or_else(|| placement(None))?;
        let buf = dev
            .as_any()
            .downcast_ref::<WgpuBuffer>()
            .ok_or_else(|| placement(Some(dev.backend())))?;
        if !self.ctx.same(buf.context()) {
            return Err(placement(Some(BackendId::Wgpu)));
        }
        if !t.is_contiguous()? {
            return Err(shape(op, "non-contiguous view is not supported"));
        }
        let elems = t.num_elements()?;
        Ok(In { t, buf, elems })
    }

    /// Host copy of a U32 tensor's values, kept since its upload.
    fn ids<'t>(&self, op: Op, a: &In<'t>) -> Result<&'t [u32], OjasError> {
        let shadow = a.buf.shadow().ok_or_else(|| OjasError::Unsupported {
            op: op.name(),
            detail: "U32 tensor was not created by WgpuBackend::upload; its ids cannot be \
                     range-checked without a readback"
                .to_string(),
        })?;
        let start = a.t.byte_offset() / 4;
        shadow
            .get(start..start + a.elems)
            .ok_or_else(|| shape(op, "U32 view lies outside its upload"))
    }

    /// A new device tensor. The budget is charged before the buffer exists.
    fn out(&self, op: Op, dims: &[usize]) -> Result<(Tensor, wgpu::Buffer), OjasError> {
        let elems = product(op, dims)?;
        let bytes = (elems as u64)
            .checked_mul(4)
            .ok_or_else(|| overflow(op, "byte length overflows"))?;
        self.ctx.check_bytes(bytes, &self.budget)?;
        let charge = self.budget.try_reserve(bytes)?;
        let wb = self.ctx.tensor_buffer(bytes)?;
        let raw = wb.raw()?.clone();
        let t = Tensor::from_device_reserved(Arc::new(wb), dims, DType::F32, charge)?;
        Ok((t, raw))
    }

    /// Refuse a scratch or output length before anything is recorded.
    fn fits(&self, op: Op, elems: usize) -> Result<(), OjasError> {
        let bytes = (elems as u64)
            .checked_mul(4)
            .ok_or_else(|| overflow(op, "byte length overflows"))?;
        self.ctx.check_bytes(bytes, &self.budget)
    }

    fn groups(&self, groups: usize) -> Result<(u32, u32, u32), OjasError> {
        let (x, y) = fold_grid(
            groups as u64,
            self.ctx.limits().max_compute_workgroups_per_dimension,
        )?;
        Ok((x, y, 1))
    }

    fn lanes(&self, n: usize) -> Result<(u32, u32, u32), OjasError> {
        self.groups(n.div_ceil(256))
    }

    fn job(&self, op: Op) -> Job<'_> {
        self.ctx.job(&self.budget, op.fault_id())
    }

    /// The gate forward: `z = x @ W^T` on the GEMM, then the gated product,
    /// and with `save` the per-head sigmoid `[rows, heads]` as well.
    fn gate_forward(
        &self,
        [input, weight, bias, attn_out]: [&Tensor; 4],
        save: bool,
    ) -> Result<(Tensor, Option<Tensor>), OjasError> {
        let op = GATE_F;
        let d = per_head_sigmoid_gate_forward_dims(input, weight, bias, attn_out)?;
        let (rows, din, heads, dh) = (d.rows, d.d_model, d.heads, d.head_dim);
        let xv = self.placed(op, input)?;
        let wv = self.placed(op, weight)?;
        let bv = self.placed(op, bias)?;
        let av = self.placed(op, attn_out)?;
        let zlen = product(op, &[rows, heads])?;
        self.fits(op, zlen)?;
        let grid = self.lanes(av.elems)?;
        let (y, yb) = self.out(op, av.shape())?;
        let saved = if save {
            Some(self.out(op, &[rows, heads])?)
        } else {
            None
        };
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let wb = bind(op, &mut job, &wv)?;
        let bb = bind(op, &mut job, &bv)?;
        let ab = bind(op, &mut job, &av)?;
        let z = job.scratch((zlen as u64) * 4)?;
        self.gemm(
            op,
            &mut job,
            Mm::Nt,
            &xb,
            &wb,
            &z,
            (rows, heads, din),
            [din, 1, 1, din],
        )?;
        let words = [u(op, av.elems)?, u(op, heads)?, u(op, dh)?];
        match &saved {
            Some((_, sb)) => {
                job.dispatch(&GATE_FWD_SAVE, &words, &[&z, &bb, &ab, &yb, sb], grid)?
            }
            None => job.dispatch(&GATE_FWD, &words, &[&z, &bb, &ab, &yb], grid)?,
        }
        job.commit()?;
        Ok((y, saved.map(|(s, _)| s)))
    }

    /// The gate backward. With `scales` (a saving forward's sigmoid) the
    /// logits GEMM is skipped and the bias is not read; it is still checked
    /// finite, as the recomputing kernel's read of it checks it.
    fn gate_backward(
        &self,
        [input, weight, bias, attn_out, grad_output]: [&Tensor; 5],
        scales: Option<&Tensor>,
    ) -> Result<PerHeadGateGrad, OjasError> {
        let op = GATE_B;
        let d = per_head_sigmoid_gate_backward_dims(input, weight, bias, attn_out, grad_output)?;
        let (rows, din, heads, dh) = (d.rows, d.d_model, d.heads, d.head_dim);
        if let Some(s) = scales {
            if s.shape() != [rows, heads] || s.dtype() != DType::F32 {
                return Err(shape(
                    op,
                    format!(
                        "saved scales {:?} {:?}, expected F32 [{rows}, {heads}]",
                        s.dtype(),
                        s.shape()
                    ),
                ));
            }
        }
        let xv = self.placed(op, input)?;
        let wv = self.placed(op, weight)?;
        let bv = self.placed(op, bias)?;
        let av = self.placed(op, attn_out)?;
        let gv = self.placed(op, grad_output)?;
        let sv = scales.map(|s| self.placed(op, s)).transpose()?;
        let zlen = product(op, &[rows, heads])?;
        self.fits(op, zlen)?;
        let grid = self.lanes(zlen)?;
        let (gx, gxb) = self.out(op, xv.shape())?;
        let (gw, gwb) = self.out(op, wv.shape())?;
        let (gbias, gbb) = self.out(op, bv.shape())?;
        let (ga, gab) = self.out(op, av.shape())?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let wb = bind(op, &mut job, &wv)?;
        let bb = bind(op, &mut job, &bv)?;
        let ab = bind(op, &mut job, &av)?;
        let gb = bind(op, &mut job, &gv)?;
        let gz = job.scratch((zlen as u64) * 4)?;
        let words = [u(op, zlen)?, u(op, heads)?, u(op, dh)?];
        match &sv {
            Some(sv) => {
                let sb = bind(op, &mut job, sv)?;
                self.check_finite(op, &mut job, &bb, heads)?;
                job.dispatch(&GATE_BWD_SAVED, &words, &[&sb, &ab, &gb, &gab, &gz], grid)?;
            }
            None => {
                let z = job.scratch((zlen as u64) * 4)?;
                self.gemm(
                    op,
                    &mut job,
                    Mm::Nt,
                    &xb,
                    &wb,
                    &z,
                    (rows, heads, din),
                    [din, 1, 1, din],
                )?;
                job.dispatch(&GATE_BWD, &words, &[&z, &bb, &ab, &gb, &gab, &gz], grid)?;
            }
        }
        self.gemm(
            op,
            &mut job,
            Mm::Nn,
            &gz,
            &wb,
            &gxb,
            (rows, din, heads),
            [heads, 1, din, 1],
        )?;
        self.gemm(
            op,
            &mut job,
            Mm::Tn,
            &gz,
            &xb,
            &gwb,
            (heads, din, rows),
            [1, heads, din, 1],
        )?;
        self.col_sum(op, &mut job, &gz, None, rows, heads, &gbb)?;
        job.commit()?;
        Ok(PerHeadGateGrad {
            input: gx,
            weight: gw,
            bias: gbias,
            attn_out: ga,
        })
    }

    /// `C[m, n] = A(m, k) B(k, n)` with element strides `[a_rs, a_cs, b_rs, b_cs]`.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        op: Op,
        job: &mut Job<'_>,
        kind: Mm,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        mnk: (usize, usize, usize),
        strides: [usize; 4],
    ) -> Result<(), OjasError> {
        self.gemm_at(op, job, kind, a, b, c, mnk, strides, Place::default())
    }

    /// [`WgpuBackend::gemm`] on sub-matrices: A, B and C start at element
    /// offsets `place.offsets` of their buffers (C's row stride stays `n`),
    /// and with `place.accumulate` the product is added to C.
    #[allow(clippy::too_many_arguments)]
    fn gemm_at(
        &self,
        op: Op,
        job: &mut Job<'_>,
        kind: Mm,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        (m, n, kk): (usize, usize, usize),
        strides: [usize; 4],
        place: Place,
    ) -> Result<(), OjasError> {
        let max = self.ctx.limits().max_compute_workgroups_per_dimension;
        let big = gemm_tile(m, n) == GEMM_BIG_TILE;
        let (gx, gy) = gemm_grid(m, n, max)?;
        let kernel = match (kind, big) {
            (Mm::Nt, false) => &GEMM_NT,
            (Mm::Nn, false) => &GEMM_NN,
            (Mm::Tn, false) => &GEMM_TN,
            (Mm::Nt, true) => &GEMM_NT_BIG,
            (Mm::Nn, true) => &GEMM_NN_BIG,
            (Mm::Tn, true) => &GEMM_TN_BIG,
        };
        let words = [
            u(op, m)?,
            u(op, n)?,
            u(op, kk)?,
            u(op, strides[0])?,
            u(op, strides[1])?,
            u(op, strides[2])?,
            u(op, strides[3])?,
            u(op, place.offsets[0])?,
            u(op, place.offsets[1])?,
            u(op, place.offsets[2])?,
            u32::from(place.accumulate),
        ];
        job.dispatch(kernel, &words, &[a, b, c], (gx, gy, 1))
    }

    /// `out[0] = finish(sum of terms)`, two stages with a fixed chunk.
    /// Mode 0 sums `x0` and divides by `divisor`; mode 1 is the value-residual
    /// lambda gradient and reads `lambda`.
    #[allow(clippy::too_many_arguments)]
    fn sum(
        &self,
        op: Op,
        job: &mut Job<'_>,
        n: usize,
        terms: [&wgpu::Buffer; 3],
        mode: u32,
        divisor: f32,
        lambda: Option<&wgpu::Buffer>,
        out: &wgpu::Buffer,
    ) -> Result<(), OjasError> {
        let groups = n.div_ceil(CHUNK);
        let grid = self.groups(groups)?;
        let partial = job.scratch((groups as u64) * 4)?;
        job.dispatch(
            &SUM_PARTIAL,
            &[u(op, n)?, mode],
            &[terms[0], terms[1], terms[2], &partial],
            grid,
        )?;
        job.dispatch(
            &SUM_FINISH,
            &[u(op, groups)?, mode, divisor.to_bits()],
            &[&partial, lambda.unwrap_or(&partial), out],
            (1, 1, 1),
        )
    }

    /// Column sums of `[rows, cols]`; with `extra` the term is `x0 * (x1 * x2[row])`.
    #[allow(clippy::too_many_arguments)]
    fn col_sum(
        &self,
        op: Op,
        job: &mut Job<'_>,
        src: &wgpu::Buffer,
        extra: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
        rows: usize,
        cols: usize,
        out: &wgpu::Buffer,
    ) -> Result<(), OjasError> {
        let per = rows.div_ceil(1024).max(64);
        let chunks = rows.div_ceil(per);
        let max = self.ctx.limits().max_compute_workgroups_per_dimension;
        let gx = u(op, cols.div_ceil(256))?;
        let gy = u(op, chunks)?;
        if gx > max || gy > max {
            return Err(OjasError::OutOfRange {
                op: op.name(),
                detail: format!("column sum grid ({gx}, {gy}) exceeds {max} per axis"),
            });
        }
        let partial_len = chunks
            .checked_mul(cols)
            .ok_or_else(|| overflow(op, "column partials overflow"))?;
        let partial = job.scratch((partial_len as u64) * 4)?;
        let (x1, x2, mode) = match extra {
            Some((a, b)) => (a, b, 1u32),
            None => (src, src, 0u32),
        };
        job.dispatch(
            &COL_PARTIAL,
            &[u(op, rows)?, u(op, cols)?, u(op, per)?, mode],
            &[src, x1, x2, &partial],
            (gx, gy, 1),
        )?;
        job.dispatch(
            &COL_FINISH,
            &[u(op, chunks)?, u(op, cols)?],
            &[&partial, out],
            self.lanes(cols)?,
        )
    }

    /// Record `dst = permute(src)`, where `src` is contiguous with `shape` and
    /// output axis `a` is input axis `dims[a]`. The caller has validated
    /// `dims` with [`permute_dims`]. Words move as `u32`, so the bits
    /// are unchanged; a non-finite one raises `op`'s fault bit, as any op's
    /// non-finite output does.
    #[allow(clippy::too_many_arguments)]
    fn permute_into(
        &self,
        op: Op,
        job: &mut Job<'_>,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        shape: &[usize],
        dims: &[usize],
    ) -> Result<(), OjasError> {
        let rank = shape.len();
        let n = product(op, shape)?;
        let grid = self.lanes(n)?;
        let mut strides = vec![1usize; rank];
        for a in (0..rank.saturating_sub(1)).rev() {
            strides[a] = strides[a + 1] * shape[a + 1];
        }
        let mut geom = Vec::with_capacity(2 * rank + 1);
        for &axis in dims {
            geom.push(u(op, shape[axis])?);
        }
        for &axis in dims {
            geom.push(u(op, strides[axis])?);
        }
        // A rank-0 move still needs a non-empty binding.
        if geom.is_empty() {
            geom.push(0);
        }
        let geom = job.upload_u32(&geom)?;
        job.dispatch(
            &PERMUTE_K,
            &[u(op, n)?, u(op, rank)?],
            &[src, &geom, dst],
            grid,
        )
    }

    /// Global L2 norm of `parts` (buffer, element count) into `status`:
    /// word 0 the norm bits, 1 a non-finite flag, 2 the abs-max bits. The
    /// norm is `amax * sqrt(sum((x / amax)^2))`, so the f32 sum of squares
    /// cannot overflow while the norm itself is finite.
    fn global_norm(
        &self,
        op: Op,
        job: &mut Job<'_>,
        parts: &[(&wgpu::Buffer, usize)],
        status: &wgpu::Buffer,
    ) -> Result<(), OjasError> {
        let mut total = 0usize;
        for &(_, n) in parts {
            total = total
                .checked_add(n.div_ceil(CHUNK))
                .ok_or_else(|| overflow(op, "norm partials overflow"))?;
        }
        self.fits(op, total)?;
        let partial = job.scratch((total as u64) * 4)?;
        job.clear(status);
        let mut base = 0usize;
        for &(b, n) in parts {
            job.dispatch(
                &ABSMAX_PARTIAL,
                &[u(op, n)?, u(op, base)?],
                &[b, &partial, status],
                self.groups(n.div_ceil(CHUNK))?,
            )?;
            base += n.div_ceil(CHUNK);
        }
        job.dispatch(
            &ABSMAX_FINISH,
            &[u(op, total)?],
            &[&partial, status],
            (1, 1, 1),
        )?;
        let mut base = 0usize;
        for &(b, n) in parts {
            job.dispatch(
                &SUMSQ_PARTIAL,
                &[u(op, n)?, u(op, base)?],
                &[b, &partial, status],
                self.groups(n.div_ceil(CHUNK))?,
            )?;
            base += n.div_ceil(CHUNK);
        }
        job.dispatch(
            &CLIP_FINISH,
            &[u(op, total)?],
            &[&partial, status],
            (1, 1, 1),
        )
    }

    /// One elementwise `y = s0 * x0 + s1 * x1` over `n` values.
    #[allow(clippy::too_many_arguments)]
    fn lincomb(
        &self,
        op: Op,
        job: &mut Job<'_>,
        n: usize,
        s0: f32,
        x0: &wgpu::Buffer,
        s1: f32,
        x1: &wgpu::Buffer,
        y: &wgpu::Buffer,
    ) -> Result<(), OjasError> {
        job.dispatch(
            &MUON_LINCOMB,
            &[u(op, n)?, s0.to_bits(), s1.to_bits()],
            &[x0, x1, y],
            self.lanes(n)?,
        )
    }

    /// Raise `op`'s fault bit if any of the first `n` values of `buf` is not
    /// finite. Used for inputs whose values need not reach an output.
    fn check_finite(
        &self,
        op: Op,
        job: &mut Job<'_>,
        buf: &wgpu::Buffer,
        n: usize,
    ) -> Result<(), OjasError> {
        job.dispatch(&CHECK_FINITE, &[u(op, n)?], &[buf], self.lanes(n)?)
    }

    /// One elementwise kernel over `ins`, writing `outs` tensors shaped like
    /// `ins[0]`. Word 0 is the element count; `extra` follows.
    fn pointwise(
        &self,
        op: Op,
        kernel: &Kernel,
        ins: &[&In<'_>],
        outs: usize,
        extra: &[u32],
    ) -> Result<Vec<Tensor>, OjasError> {
        let first = ins.first().ok_or_else(|| shape(op, "no operands"))?;
        let n = first.elems;
        let grid = self.lanes(n)?;
        let mut tensors = Vec::with_capacity(outs);
        let mut out_bufs = Vec::with_capacity(outs);
        for _ in 0..outs {
            let (t, b) = self.out(op, first.shape())?;
            tensors.push(t);
            out_bufs.push(b);
        }
        let mut job = self.job(op);
        let mut bound = Vec::with_capacity(ins.len());
        for a in ins {
            bound.push(bind(op, &mut job, a)?);
        }
        let mut words = vec![u(op, n)?];
        words.extend_from_slice(extra);
        let bufs: Vec<&wgpu::Buffer> = bound.iter().chain(out_bufs.iter()).collect();
        job.dispatch(kernel, &words, &bufs, grid)?;
        job.commit()?;
        Ok(tensors)
    }

    /// Workgroups for `rows` rows of `dim`, [`rms_width`] lanes per row.
    fn rms_grid(&self, rows: usize, dim: usize) -> Result<(u32, u32, u32), OjasError> {
        self.groups(rows.div_ceil(256 / rms_width(dim)))
    }

    /// Placement, device limits and outputs of an `rms_norm_forward` whose
    /// shapes [`rms_norm_forward_dims`] accepted, before recording.
    fn rms_fwd_plan<'t>(
        &self,
        d: RmsDims,
        x: &'t Tensor,
        w: &'t Tensor,
    ) -> Result<RmsFwd<'t>, OjasError> {
        let op = RMS_F;
        let xv = self.placed(op, x)?;
        let wv = self.placed(op, w)?;
        let RmsDims { rows, dim } = d;
        self.rms_grid(rows, dim)?;
        let (y, yb) = self.out(op, xv.shape())?;
        Ok(RmsFwd {
            xv,
            wv,
            rows,
            dim,
            y,
            yb,
        })
    }

    fn rms_fwd_record(&self, job: &mut Job<'_>, p: &RmsFwd<'_>, eps: f32) -> Result<(), OjasError> {
        let op = RMS_F;
        let xb = bind(op, job, &p.xv)?;
        let wb = bind(op, job, &p.wv)?;
        job.dispatch(
            &RMS_FWD,
            &[
                u(op, p.rows)?,
                u(op, p.dim)?,
                eps.to_bits(),
                u(op, rms_width(p.dim))?,
            ],
            &[&xb, &wb, &p.yb],
            self.rms_grid(p.rows, p.dim)?,
        )
    }

    /// Placement, device limits and outputs of an `rms_norm_backward` whose
    /// shapes [`rms_norm_backward_dims`] accepted, before recording.
    fn rms_bwd_plan<'t>(
        &self,
        d: RmsDims,
        x: &'t Tensor,
        w: &'t Tensor,
        g: &'t Tensor,
    ) -> Result<RmsBwd<'t>, OjasError> {
        let op = RMS_B;
        let xv = self.placed(op, x)?;
        let wv = self.placed(op, w)?;
        let gv = self.placed(op, g)?;
        let RmsDims { rows, dim } = d;
        self.rms_grid(rows, dim)?;
        self.fits(op, rows)?;
        let (gx, gxb) = self.out(op, xv.shape())?;
        let (gw, gwb) = self.out(op, wv.shape())?;
        Ok(RmsBwd {
            xv,
            wv,
            gv,
            rows,
            dim,
            gx,
            gxb,
            gw,
            gwb,
        })
    }

    fn rms_bwd_record(&self, job: &mut Job<'_>, p: &RmsBwd<'_>, eps: f32) -> Result<(), OjasError> {
        let op = RMS_B;
        let xb = bind(op, job, &p.xv)?;
        let wb = bind(op, job, &p.wv)?;
        let gb = bind(op, job, &p.gv)?;
        let rstd = job.scratch((p.rows as u64) * 4)?;
        job.dispatch(
            &RMS_BWD,
            &[
                u(op, p.rows)?,
                u(op, p.dim)?,
                eps.to_bits(),
                u(op, rms_width(p.dim))?,
            ],
            &[&xb, &wb, &gb, &p.gxb, &rstd],
            self.rms_grid(p.rows, p.dim)?,
        )?;
        self.col_sum(op, job, &gb, Some((&xb, &rstd)), p.rows, p.dim, &p.gwb)
    }

    /// RoPE in `direction` 0 (forward) or 1 (backward) for a call whose
    /// shapes the op's validator accepted as `d`. Kernel layout 0 is a table
    /// shaped like `x`; layout 1 is `x` `[B, T, H, D]` with `[T, D]` tables.
    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        op: Op,
        d: RopeDims,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        direction: u32,
    ) -> Result<Tensor, OjasError> {
        let xv = self.placed(op, x)?;
        let cv = self.placed(op, cos)?;
        let sv = self.placed(op, sin)?;
        let RopeDims { rows, dim, layout } = d;
        let (layout, heads, time) = match layout {
            RopeLayout::Same => (0u32, 1usize, 1usize),
            RopeLayout::TimeDim { time, heads } => (1, heads, time),
        };
        let half_lanes = rows
            .checked_mul(dim / 2)
            .ok_or_else(|| overflow(op, "rope lanes overflow"))?;
        let grid = self.lanes(half_lanes)?;
        let (y, yb) = self.out(op, xv.shape())?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let cb = bind(op, &mut job, &cv)?;
        let sb = bind(op, &mut job, &sv)?;
        job.dispatch(
            &ROPE,
            &[
                u(op, half_lanes)?,
                u(op, dim)?,
                layout,
                u(op, heads)?,
                u(op, time)?,
                direction,
            ],
            &[&xb, &cb, &sb, &yb],
            grid,
        )?;
        job.commit()?;
        Ok(y)
    }

    /// The plan of a call whose shapes the op's validator accepted as `d`:
    /// this device's head-dim and shared-memory limits. A window of 0 is
    /// refused; one of at least `T` sees every earlier key and is dropped.
    fn sdpa_plan(&self, op: Op, d: SdpaDims, window: Option<usize>) -> Result<SdpaPlan, OjasError> {
        if window == Some(0) {
            return Err(shape(op, "sdpa window must be at least 1"));
        }
        let dim = d.head_dim;
        let dim_u32 = u32::try_from(dim)
            .map_err(|_| overflow(op, format!("head dim {dim} does not fit in u32")))?;
        let scale = sdpa_scale(dim_u32)?;
        if dim_u32 > ATTENTION_MAX_HEAD_DIM {
            return Err(OjasError::UnsupportedHeadDim {
                head_dim: dim_u32,
                limit: ATTENTION_MAX_HEAD_DIM,
            });
        }
        let shared = self.ctx.limits().max_compute_workgroup_storage_size;
        let bh = product(op, &[d.batch, d.heads])?;
        // `1` when the counts match, including both zero, so this never
        // divides by zero. Any other grouping the validator missed is a
        // shape error here, before a launch.
        let rep = if d.kv_heads == d.heads {
            1
        } else if d.kv_heads == 0 || !d.heads.is_multiple_of(d.kv_heads) {
            return Err(shape(
                op,
                format!("sdpa query heads {} over kv heads {}", d.heads, d.kv_heads),
            ));
        } else {
            d.heads / d.kv_heads
        };
        let tiles = attention_tiles(dim_u32, shared)?;
        Ok(SdpaPlan {
            bh,
            time: d.seq,
            dim,
            rep,
            window: window.filter(|&w| w < d.seq).unwrap_or(0),
            scale,
            module: WgslModule::Attention(tiles),
            fwd_rows: tiles.fwd_rows as usize,
            bwd_rows: tiles.bwd_block as usize,
        })
    }

    /// Causal SDPA forward after its validator: the output and the
    /// `[B, H, T]` row log-sum-exp. Grouped-query heads read their KV head
    /// in place, so nothing beyond the two outputs is allocated.
    fn sdpa_forward(
        &self,
        op: Op,
        d: SdpaDims,
        window: Option<usize>,
        [q, k, v]: [&Tensor; 3],
    ) -> Result<(Tensor, Tensor), OjasError> {
        let qv = self.placed(op, q)?;
        let kv = self.placed(op, k)?;
        let vv = self.placed(op, v)?;
        let plan = self.sdpa_plan(op, d, window)?;
        let grid = self.attn_grid(op, plan.bh, plan.time, plan.fwd_rows)?;
        let (y, yb) = self.out(op, qv.shape())?;
        let (lse, lb) = self.out(op, &qv.shape()[..qv.shape().len().saturating_sub(1)])?;
        let mut job = self.job(op);
        let qb = bind(op, &mut job, &qv)?;
        let kb = bind(op, &mut job, &kv)?;
        let vb = bind(op, &mut job, &vv)?;
        job.dispatch(
            &attn_kernel(plan.module, "attn_fwd"),
            &plan.words(op)?,
            &[&qb, &kb, &vb, &yb, &lb],
            grid,
        )?;
        job.commit()?;
        Ok((y, lse))
    }

    /// Causal SDPA backward after its validator, from the forward's output
    /// and log-sum-exp: `[q, k, v, output, lse, grad_output]`. The only
    /// scratch is the `2 * B * H * T` row statistics; grouped-query
    /// gradients are summed onto their KV head inside `attn_bwd_dkv`.
    fn sdpa_backward(
        &self,
        op: Op,
        d: SdpaDims,
        window: Option<usize>,
        [q, k, v, o, lse, grad_output]: [&Tensor; 6],
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        let qv = self.placed(op, q)?;
        let kv = self.placed(op, k)?;
        let vv = self.placed(op, v)?;
        let ov = self.placed(op, o)?;
        let lv = self.placed(op, lse)?;
        let gv = self.placed(op, grad_output)?;
        let plan = self.sdpa_plan(op, d, window)?;
        let bht = product(op, &[plan.bh, plan.time])?;
        let stats_len = product(op, &[bht, 2])?;
        self.fits(op, stats_len)?;
        let q_grid = self.attn_grid(op, plan.bh, plan.time, plan.bwd_rows)?;
        let kv_grid = self.attn_grid(op, plan.bh / plan.rep, plan.time, plan.bwd_rows)?;
        let (gq, gqb) = self.out(op, qv.shape())?;
        let (gk, gkb) = self.out(op, kv.shape())?;
        let (gvv, gvb) = self.out(op, vv.shape())?;
        let mut job = self.job(op);
        let qb = bind(op, &mut job, &qv)?;
        let kb = bind(op, &mut job, &kv)?;
        let vb = bind(op, &mut job, &vv)?;
        let ob = bind(op, &mut job, &ov)?;
        let lb = bind(op, &mut job, &lv)?;
        let gb = bind(op, &mut job, &gv)?;
        let stats = job.scratch((stats_len as u64) * 4)?;
        let words = plan.words(op)?;
        job.dispatch(
            &attn_kernel(plan.module, "attn_bwd_dr"),
            &words,
            &[&ob, &gb, &lb, &stats],
            self.lanes(bht)?,
        )?;
        job.dispatch(
            &attn_kernel(plan.module, "attn_bwd_dq"),
            &words,
            &[&qb, &kb, &vb, &gb, &stats, &gqb],
            q_grid,
        )?;
        job.dispatch(
            &attn_kernel(plan.module, "attn_bwd_dkv"),
            &words,
            &[&qb, &kb, &vb, &gb, &stats, &gkb, &gvb],
            kv_grid,
        )?;
        job.commit()?;
        Ok((gq, gk, gvv))
    }

    /// `(ceil(time / rows), batch * heads)` workgroups.
    fn attn_grid(
        &self,
        op: Op,
        bh: usize,
        time: usize,
        rows: usize,
    ) -> Result<(u32, u32, u32), OjasError> {
        let gx = u(op, time.div_ceil(rows))?;
        let gy = u(op, bh)?;
        let max = self.ctx.limits().max_compute_workgroups_per_dimension;
        if gx > max || gy > max {
            return Err(OjasError::OutOfRange {
                op: op.name(),
                detail: format!("attention grid ({gx}, {gy}) exceeds {max} per axis"),
            });
        }
        Ok((gx, gy, 1))
    }

    /// `(ignore word, valid count, any ignored)` of targets the op's
    /// validator accepted against `rows` logits rows of `vocab`: each target
    /// range-checked on the upload's host copy, and the global valid count.
    fn ce_targets(
        &self,
        op: Op,
        targets: &In<'_>,
        vocab: usize,
        ignore: Option<u32>,
        rows: usize,
    ) -> Result<(u32, u32, bool), OjasError> {
        let ids = self.ids(op, targets)?;
        let valid = valid_targets(op, ids, vocab, ignore)?;
        let any_ignored = (valid as usize) < rows;
        Ok((ignore.unwrap_or(0), valid, any_ignored))
    }
}

/// Every token id below `vocab`, on the upload's host copy.
fn check_ids(op: Op, ids: &[u32], vocab: usize) -> Result<(), OjasError> {
    for (n, &id) in ids.iter().enumerate() {
        if (id as usize) >= vocab {
            return Err(OjasError::OutOfRange {
                op: op.name(),
                detail: format!("token id {id} at {n} is outside vocab {vocab}"),
            });
        }
    }
    Ok(())
}

/// The most budget bytes (`job.scratch` and `job.upload_u32`) one optimizer
/// step on a `[rows, cols]` parameter charges, with `n = rows * cols`,
/// `b = 4n` and `r = min(rows, cols)`. Both steps may first copy a gradient
/// view that sits at a byte offset (`bind`, `b`). Then:
/// - AdamW: old copies of the parameter and both moments, and a 16-byte
///   status: `3b + 16`.
/// - Muon: two 16-byte words; six `b`-byte planes (momentum buffer, update,
///   X, X', BX, new parameter); three `r × r` f32 matrices; the global
///   norm's partials, 4 bytes per [`CHUNK`] values; and for a tall matrix
///   (`rows > cols`) two `b`-byte transposes, each with a 16-byte geometry
///   upload: `32 + 6b + 12 r² + 4 ceil(n / CHUNK) [+ 2b + 32]`.
///
/// `adamw_step` and `muon_ns5_step` reserve exactly these pieces; the test
/// `reported_optimizer_scratch_bounds_the_measured_peak` compares them.
fn optimizer_scratch(kind: OptimizerKind, rows: usize, cols: usize) -> Result<u64, OjasError> {
    let overflow = || OjasError::OutOfRange {
        op: "optimizer_scratch_bytes",
        detail: format!("[{rows}, {cols}] scratch overflows u64"),
    };
    let n = (rows as u64)
        .checked_mul(cols as u64)
        .ok_or_else(overflow)?;
    let b = n.checked_mul(4).ok_or_else(overflow)?;
    let sum = |terms: &[Option<u64>]| {
        terms
            .iter()
            .try_fold(0u64, |acc, t| acc.checked_add((*t)?))
            .ok_or_else(overflow)
    };
    match kind {
        OptimizerKind::AdamW => sum(&[Some(b), b.checked_mul(3), Some(16)]),
        OptimizerKind::MuonNs5 => {
            let r = rows.min(cols) as u64;
            let partials = n.div_ceil(CHUNK as u64).checked_mul(4);
            let tall = if rows > cols {
                b.checked_mul(2).and_then(|t| t.checked_add(32))
            } else {
                Some(0)
            };
            sum(&[
                Some(b),
                Some(32),
                b.checked_mul(6),
                r.checked_mul(r).and_then(|rr| rr.checked_mul(12)),
                partials,
                tall,
            ])
        }
    }
}

impl Backend for WgpuBackend {
    fn id(&self) -> BackendId {
        BackendId::Wgpu
    }

    fn budget(&self) -> &Budget {
        &self.budget
    }

    fn numerics(&self) -> Numerics {
        Numerics::Fast
    }

    /// Submit everything recorded, wait for it, and report a deferred fault
    /// as [`OjasError::NonFinite`] or a wgpu error (a lost device included)
    /// as [`OjasError::Backend`]. The only `sync`: an inherent twin would
    /// leave `Arc<WgpuBackend>` and generic callers on the trait default.
    fn sync(&self) -> Result<(), OjasError> {
        self.ctx.sync()?;
        self.faults()
    }

    /// Host -> device copy. A tensor already on this context is a clone;
    /// another device's or context's tensor is [`OjasError::Placement`].
    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "WgpuBackend::upload";
        if let Some(dev) = tensor.device_buffer() {
            let ours = dev
                .as_any()
                .downcast_ref::<WgpuBuffer>()
                .is_some_and(|b| self.ctx.same(b.context()));
            if ours {
                return Ok(tensor.clone());
            }
            return Err(OjasError::Placement {
                op: OP,
                expected: Some(BackendId::Wgpu),
                found: Some(dev.backend()),
            });
        }
        if !tensor.is_contiguous()? {
            return Err(OjasError::Shape {
                op: OP,
                detail: "non-contiguous view is not supported".to_string(),
            });
        }
        if tensor.num_elements()? == 0 || tensor.shape().contains(&0) {
            return Err(OjasError::Shape {
                op: OP,
                detail: "empty tensor".to_string(),
            });
        }
        // Contiguous and non-empty (checked above), so the window is the
        // element count times the dtype size and fits the host storage.
        let len = tensor.num_elements()? * tensor.dtype().size();
        let padded = (len as u64).div_ceil(4) * 4;
        self.ctx.check_bytes(padded, &self.budget)?;
        let charge = self.budget.try_reserve(padded)?;
        let shadow: Option<Arc<[u32]>> = if tensor.dtype() == DType::U32 {
            Some(tensor.u32_slice()?.into())
        } else {
            None
        };
        let bytes = tensor.to_ne_bytes()?;
        let wb = self.ctx.upload_bytes(&bytes, shadow)?;
        Tensor::from_device_reserved(Arc::new(wb), tensor.shape(), tensor.dtype(), charge)
    }

    /// The only readback. A fault deferred by any earlier op is reported here.
    fn download(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        let host = tensor.to_host(&self.budget)?;
        if tensor.device().is_some() {
            self.faults()?;
        }
        Ok(host)
    }

    /// Device to device, bit-exact. F32 only: a U32 tensor's host copy of its
    /// ids could not follow the move without a readback. A non-finite input
    /// is moved unchanged and reported as [`OjasError::NonFinite`] naming
    /// `permute` at the next sync point (the module docs' contract).
    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        let op = PERMUTE;
        // dtype, zero axis and axes, before placement or contiguity (D17).
        let out_shape = permute_dims(input, dims)?;
        let x = self.placed(op, input)?;
        let (y, yb) = self.out(op, &out_shape)?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &x)?;
        self.permute_into(op, &mut job, &xb, &yb, x.shape(), dims)?;
        job.commit()?;
        Ok(y)
    }

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        let op = EMBED_F;
        let d = embedding_forward_dims(table, token_ids)?;
        let tv = self.placed(op, table)?;
        let iv = self.placed(op, token_ids)?;
        check_ids(op, self.ids(op, &iv)?, d.vocab)?;
        let n = product(op, &[d.tokens, d.dim])?;
        let dim = d.dim;
        let grid = self.lanes(n)?;
        let (y, yb) = self.out(op, &d.out_shape)?;
        let mut job = self.job(op);
        let tb = bind(op, &mut job, &tv)?;
        let ib = bind(op, &mut job, &iv)?;
        self.check_finite(op, &mut job, &tb, tv.elems)?;
        job.dispatch(
            &EMBED_FWD,
            &[u(op, n)?, u(op, dim)?],
            &[&tb, &ib, &yb],
            grid,
        )?;
        job.commit()?;
        Ok(y)
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let op = EMBED_B;
        let d = embedding_backward_dims(table, token_ids, grad_output)?;
        let (vocab, dim) = (d.vocab, d.dim);
        let tv = self.placed(op, table)?;
        let iv = self.placed(op, token_ids)?;
        let gv = self.placed(op, grad_output)?;
        let ids = self.ids(op, &iv)?;
        check_ids(op, ids, vocab)?;
        u(op, ids.len())?;
        let mut order: Vec<u32> = (0..ids.len() as u32).collect();
        order.sort_by_key(|&p| ids[p as usize]);
        let mut starts = Vec::new();
        let mut uniq = Vec::new();
        for (j, &p) in order.iter().enumerate() {
            let id = ids[p as usize];
            if uniq.last() != Some(&id) {
                uniq.push(id);
                starts.push(j as u32);
            }
        }
        starts.push(order.len() as u32);
        let lanes = product(op, &[uniq.len(), dim])?;
        let grid = self.lanes(lanes)?;
        let (y, yb) = self.out(op, &[vocab, dim])?;
        let mut job = self.job(op);
        let gb = bind(op, &mut job, &gv)?;
        let tb = bind(op, &mut job, &tv)?;
        self.check_finite(op, &mut job, &tb, tv.elems)?;
        let perm = job.upload_u32(&order)?;
        let seg = job.upload_u32(&starts)?;
        let ub = job.upload_u32(&uniq)?;
        job.clear(&yb);
        job.dispatch(
            &EMBED_BWD,
            &[u(op, lanes)?, u(op, dim)?],
            &[&gb, &perm, &seg, &ub, &yb],
            grid,
        )?;
        job.commit()?;
        Ok(y)
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        let op = LINEAR_F;
        let d = linear_forward_dims(input, weight)?;
        refuse_bf16_operands(op.name(), BackendId::Wgpu, &[input, weight])?;
        let (rows, kin, nout) = (d.rows, d.in_features, d.out_features);
        let xv = self.placed(op, input)?;
        let wv = self.placed(op, weight)?;
        let (y, yb) = self.out(op, &d.out_shape)?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let wb = bind(op, &mut job, &wv)?;
        self.gemm(
            op,
            &mut job,
            Mm::Nt,
            &xb,
            &wb,
            &yb,
            (rows, nout, kin),
            [kin, 1, 1, kin],
        )?;
        job.commit()?;
        Ok(y)
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let op = LINEAR_B;
        let d = linear_backward_dims(input, weight, grad_output)?;
        refuse_bf16_operands(op.name(), BackendId::Wgpu, &[input, weight, grad_output])?;
        let (rows, kin, nout) = (d.rows, d.in_features, d.out_features);
        let xv = self.placed(op, input)?;
        let wv = self.placed(op, weight)?;
        let gv = self.placed(op, grad_output)?;
        let (gx, gxb) = self.out(op, xv.shape())?;
        let (gw, gwb) = self.out(op, wv.shape())?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let wb = bind(op, &mut job, &wv)?;
        let gb = bind(op, &mut job, &gv)?;
        self.gemm(
            op,
            &mut job,
            Mm::Nn,
            &gb,
            &wb,
            &gxb,
            (rows, kin, nout),
            [nout, 1, kin, 1],
        )?;
        self.gemm(
            op,
            &mut job,
            Mm::Tn,
            &gb,
            &xb,
            &gwb,
            (nout, kin, rows),
            [1, nout, kin, 1],
        )?;
        job.commit()?;
        Ok((gx, gw))
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let d = rms_norm_forward_dims(input, weight, eps)?;
        let p = self.rms_fwd_plan(d, input, weight)?;
        let mut job = self.job(RMS_F);
        self.rms_fwd_record(&mut job, &p, eps)?;
        job.commit()?;
        Ok(p.y)
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let d = rms_norm_backward_dims(input, weight, grad_output, eps)?;
        let p = self.rms_bwd_plan(d, input, weight, grad_output)?;
        let mut job = self.job(RMS_B);
        self.rms_bwd_record(&mut job, &p, eps)?;
        job.commit()?;
        Ok((p.gx, p.gw))
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let d = rope_half_split_forward_dims(x, cos, sin)?;
        self.rope(ROPE_F, d, x, cos, sin, 0)
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let d = rope_half_split_backward_dims(grad_output, cos, sin)?;
        self.rope(ROPE_B, d, grad_output, cos, sin, 1)
    }

    /// One job for both. [`rms_qk_norm_forward_dims`] checks both pairs, then
    /// q and k are planned (placement, limits, outputs), and only then is
    /// anything recorded: a refused call, q's or k's, records nothing.
    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let (qd, kd) = rms_qk_norm_forward_dims(q, k, q_weight, k_weight, eps)?;
        let pq = self.rms_fwd_plan(qd, q, q_weight)?;
        let pk = self.rms_fwd_plan(kd, k, k_weight)?;
        let mut job = self.job(RMS_F);
        self.rms_fwd_record(&mut job, &pq, eps)?;
        self.rms_fwd_record(&mut job, &pk, eps)?;
        job.commit()?;
        Ok((pq.y, pk.y))
    }

    /// One job for both; as the forward, a refused call records nothing.
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
        let (qd, kd) = rms_qk_norm_backward_dims(q, k, q_weight, k_weight, grad_q, grad_k, eps)?;
        let pq = self.rms_bwd_plan(qd, q, q_weight, grad_q)?;
        let pk = self.rms_bwd_plan(kd, k, k_weight, grad_k)?;
        let mut job = self.job(RMS_B);
        self.rms_bwd_record(&mut job, &pq, eps)?;
        self.rms_bwd_record(&mut job, &pk, eps)?;
        job.commit()?;
        Ok((pq.gx, pk.gx, pq.gw, pk.gw))
    }

    fn causal_sdpa_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let d = causal_sdpa_forward_dims(q, k, v, window)?;
        refuse_bf16_operands(SDPA_F.name(), BackendId::Wgpu, &[q, k, v])?;
        self.sdpa_forward(SDPA_F, d, window, [q, k, v])
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
        let d = causal_sdpa_backward_dims(q, k, v, output, lse, grad_output, window)?;
        refuse_bf16_operands(
            SDPA_B.name(),
            BackendId::Wgpu,
            &[q, k, v, output, lse, grad_output],
        )?;
        self.sdpa_backward(SDPA_B, d, window, [q, k, v, output, lse, grad_output])
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
    /// no bias read (the bias is still checked finite). `scales` must be
    /// this context's `[rows, heads]` F32.
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

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let op = VR_F;
        value_residual_blend_forward_dims(value, value0, lambda)?;
        let v = self.placed(op, value)?;
        let v0 = self.placed(op, value0)?;
        let lam = self.placed(op, lambda)?;
        let mut out = self.pointwise(op, &VR_FWD, &[&v, &v0, &lam], 1, &[])?;
        out.pop().ok_or_else(|| shape(op, "no output"))
    }

    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        let op = VR_B;
        let n = value_residual_blend_backward_dims(value, value0, lambda, grad_output)?;
        let v = self.placed(op, value)?;
        let v0 = self.placed(op, value0)?;
        let lam = self.placed(op, lambda)?;
        let gy = self.placed(op, grad_output)?;
        let grid = self.lanes(n)?;
        self.fits(op, n.div_ceil(CHUNK))?;
        let (gv, gvb) = self.out(op, v.shape())?;
        let (gv0, gv0b) = self.out(op, v0.shape())?;
        let (gl, glb) = self.out(op, lam.shape())?;
        let mut job = self.job(op);
        let vb = bind(op, &mut job, &v)?;
        let v0b = bind(op, &mut job, &v0)?;
        let lb = bind(op, &mut job, &lam)?;
        let gb = bind(op, &mut job, &gy)?;
        job.dispatch(&VR_BWD, &[u(op, n)?], &[&gb, &lb, &gvb, &gv0b], grid)?;
        self.sum(op, &mut job, n, [&vb, &v0b, &gb], 1, 1.0, Some(&lb), &glb)?;
        job.commit()?;
        Ok(ValueResidualGrad {
            value: gv,
            value0: gv0,
            lambda: gl,
        })
    }

    /// Device round under its own fault bit ([`CAST`], in the high mask
    /// word). `round_bf16` never raises: a NaN stays a value, as the CPU's
    /// cast keeps it. A host tensor is [`OjasError::Placement`].
    fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "cast_bf16";
        let placement = |found| OjasError::Placement {
            op: OP,
            expected: Some(BackendId::Wgpu),
            found,
        };
        let dev = tensor.device_buffer().ok_or_else(|| placement(None))?;
        let buf = dev
            .as_any()
            .downcast_ref::<WgpuBuffer>()
            .ok_or_else(|| placement(Some(dev.backend())))?;
        if !self.ctx.same(buf.context()) {
            return Err(placement(Some(BackendId::Wgpu)));
        }
        if tensor.dtype() != DType::F32 {
            return Err(OjasError::Dtype {
                op: OP,
                expected: DType::F32,
                got: tensor.dtype(),
            });
        }
        if !tensor.is_contiguous()? {
            return Err(OjasError::Shape {
                op: OP,
                detail: "non-contiguous view is not supported".to_string(),
            });
        }
        let elems = tensor.num_elements()?;
        if elems == 0 || tensor.shape().contains(&0) {
            return Err(OjasError::Shape {
                op: OP,
                detail: "empty tensor".to_string(),
            });
        }
        let n = u32::try_from(elems).map_err(|_| OjasError::OutOfRange {
            op: OP,
            detail: format!("{elems} does not fit in u32"),
        })?;
        let bytes = (elems as u64)
            .checked_mul(4)
            .ok_or_else(|| OjasError::OutOfRange {
                op: OP,
                detail: "byte length overflows".to_string(),
            })?;
        self.ctx.check_bytes(bytes, &self.budget)?;
        let charge = self.budget.try_reserve(bytes)?;
        let wb = self.ctx.tensor_buffer(bytes)?;
        let raw_out = wb.raw()?.clone();
        let out = Tensor::from_device_reserved(Arc::new(wb), tensor.shape(), DType::F32, charge)?;
        let mut job = self.job(CAST);
        let src = {
            let raw = buf.raw()?;
            let off = tensor.byte_offset();
            if off == 0 {
                raw.clone()
            } else {
                if !off.is_multiple_of(4) {
                    return Err(OjasError::Shape {
                        op: OP,
                        detail: format!("byte_offset {off} is not 4-byte aligned"),
                    });
                }
                let dst = job.scratch(bytes)?;
                job.copy(raw, off as u64, &dst, bytes);
                dst
            }
        };
        let groups = elems.div_ceil(256);
        let (gx, gy) = fold_grid(
            groups as u64,
            self.ctx.limits().max_compute_workgroups_per_dimension,
        )
        .map_err(|err| match err {
            OjasError::OutOfRange { detail, .. } => OjasError::OutOfRange { op: OP, detail },
            other => other,
        })?;
        job.dispatch(&ROUND_BF16, &[n], &[&src, &raw_out], (gx, gy, 1))
            .map_err(|err| match err {
                OjasError::OutOfRange { detail, .. } => OjasError::OutOfRange { op: OP, detail },
                other => other,
            })?;
        job.commit()?;
        Ok(out)
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let op = SILU_F;
        silu_forward_dims(input)?;
        let x = self.placed(op, input)?;
        let mut out = self.pointwise(op, &SILU_FWD, &[&x], 1, &[])?;
        out.pop().ok_or_else(|| shape(op, "no output"))
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        let op = SILU_B;
        silu_backward_dims(input, grad_output)?;
        let x = self.placed(op, input)?;
        let gy = self.placed(op, grad_output)?;
        let mut out = self.pointwise(op, &SILU_BWD, &[&x, &gy], 1, &[])?;
        out.pop().ok_or_else(|| shape(op, "no output"))
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        let op = MUL_F;
        mul_forward_dims(a, b)?;
        let av = self.placed(op, a)?;
        let bv = self.placed(op, b)?;
        let mut out = self.pointwise(op, &MUL_FWD, &[&av, &bv], 1, &[])?;
        out.pop().ok_or_else(|| shape(op, "no output"))
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let op = MUL_B;
        mul_backward_dims(a, b, grad_output)?;
        let av = self.placed(op, a)?;
        let bv = self.placed(op, b)?;
        let gy = self.placed(op, grad_output)?;
        let mut out = self.pointwise(op, &MUL_BWD, &[&av, &bv, &gy], 2, &[])?;
        let gb = out.pop().ok_or_else(|| shape(op, "no output"))?;
        let ga = out.pop().ok_or_else(|| shape(op, "no output"))?;
        Ok((ga, gb))
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        let op = ADD_F;
        residual_add_forward_dims(x, y)?;
        let xv = self.placed(op, x)?;
        let yv = self.placed(op, y)?;
        let mut out = self.pointwise(op, &ADD_FWD, &[&xv, &yv], 1, &[])?;
        out.pop().ok_or_else(|| shape(op, "no output"))
    }

    /// Two fresh copies of `grad_output`, so each can be clipped in place.
    /// `x` and `y` are only scanned for non-finite values, as the CPU does.
    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let op = ADD_B;
        let n = residual_add_backward_dims(x, y, grad_output)?;
        let xv = self.placed(op, x)?;
        let yv = self.placed(op, y)?;
        let gy = self.placed(op, grad_output)?;
        self.lanes(n)?;
        let (gx, gxb) = self.out(op, xv.shape())?;
        let (gyy, gyb) = self.out(op, yv.shape())?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let yb = bind(op, &mut job, &yv)?;
        let gb = bind(op, &mut job, &gy)?;
        self.check_finite(op, &mut job, &xb, n)?;
        self.check_finite(op, &mut job, &yb, n)?;
        self.check_finite(op, &mut job, &gb, n)?;
        let bytes = (n as u64) * 4;
        job.copy(&gb, 0, &gxb, bytes);
        job.copy(&gb, 0, &gyb, bytes);
        job.commit()?;
        Ok((gx, gyy))
    }

    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        let op = CE_F;
        let d = cross_entropy_mean_forward_dims(logits, targets)?;
        let (rows, vocab) = (d.rows, d.vocab);
        let lv = self.placed(op, logits)?;
        let tv = self.placed(op, targets)?;
        let (ignore, valid, any_ignored) = self.ce_targets(op, &tv, vocab, ignore_index, rows)?;
        let grid = self.groups(rows)?;
        self.fits(op, rows)?;
        let (y, yb) = self.out(op, &[])?;
        let mut job = self.job(op);
        let lb = bind(op, &mut job, &lv)?;
        let tb = bind(op, &mut job, &tv)?;
        if any_ignored {
            self.check_finite(op, &mut job, &lb, lv.elems)?;
        }
        let row_loss = job.scratch((rows as u64) * 4)?;
        let denom = valid as f32;
        job.dispatch(
            &CE_FWD,
            &[
                u(op, rows)?,
                u(op, vocab)?,
                u32::from(ignore_index.is_some()),
                ignore,
                denom.to_bits(),
            ],
            &[&lb, &tb, &row_loss],
            grid,
        )?;
        self.sum(
            op,
            &mut job,
            rows,
            [&row_loss, &row_loss, &row_loss],
            0,
            denom,
            None,
            &yb,
        )?;
        job.commit()?;
        Ok(y)
    }

    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        let op = CE_B;
        let d = cross_entropy_mean_backward_dims(logits, targets)?;
        let (rows, vocab) = (d.rows, d.vocab);
        let lv = self.placed(op, logits)?;
        let tv = self.placed(op, targets)?;
        let (ignore, valid, any_ignored) = self.ce_targets(op, &tv, vocab, ignore_index, rows)?;
        let grid = self.groups(rows)?;
        let (y, yb) = self.out(op, lv.shape())?;
        let mut job = self.job(op);
        let lb = bind(op, &mut job, &lv)?;
        let tb = bind(op, &mut job, &tv)?;
        if any_ignored {
            self.check_finite(op, &mut job, &lb, lv.elems)?;
        }
        job.dispatch(
            &CE_BWD,
            &[
                u(op, rows)?,
                u(op, vocab)?,
                u32::from(ignore_index.is_some()),
                ignore,
                (valid as f32).to_bits(),
            ],
            &[&lb, &tb, &yb],
            grid,
        )?;
        job.commit()?;
        Ok(y)
    }

    /// Reads back four words (the norm and a non-finite flag), then scales
    /// every gradient in place. Any fault deferred by an earlier op is
    /// reported here, before anything is scaled. `max_norm` is checked
    /// after the norm, by [`clip_scale`], as the CPU and Metal do (shape
    /// contract D13): non-finite gradients outrank a bad `max_norm`.
    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        let op = CLIP;
        clip_grad_norm_dims(grads)?;
        let mut sizes = Vec::with_capacity(grads.len());
        let mut total = 0usize;
        for g in grads.iter() {
            let n = self.placed(op, g)?.elems;
            self.lanes(n)?;
            sizes.push(n);
            total = total
                .checked_add(n.div_ceil(CHUNK))
                .ok_or_else(|| overflow(op, "clip partials overflow"))?;
        }
        self.groups(total)?;
        self.fits(op, total)?;
        // The status words are read in a later submission. Job scratch goes
        // back to the pool at the next submit by any thread, where another
        // caller could take and overwrite it before that read; a tensor's
        // buffer is held until `_held` drops, after the read.
        let (_held, status) = self.out(op, &[4])?;
        {
            let mut job = self.job(op);
            let mut bound = Vec::with_capacity(grads.len());
            for g in grads.iter() {
                let gv = self.placed(op, g)?;
                bound.push(bind(op, &mut job, &gv)?);
            }
            let parts: Vec<(&wgpu::Buffer, usize)> =
                bound.iter().zip(sizes.iter().copied()).collect();
            self.global_norm(op, &mut job, &parts, &status)?;
            job.commit()?;
        }
        let words = self.ctx.read(&status, 0, 8)?;
        self.faults()?;
        let word = |i: usize| -> Result<u32, OjasError> {
            let bytes = words
                .get(i * 4..i * 4 + 4)
                .ok_or_else(|| shape(op, "short status read"))?;
            let mut raw = [0u8; 4];
            raw.copy_from_slice(bytes);
            Ok(u32::from_le_bytes(raw))
        };
        let norm = f32::from_bits(word(0)?);
        if word(1)? != 0 {
            return Err(OjasError::NonFinite { op: op.name() });
        }
        let scale = clip_scale(max_norm, norm)?;
        if scale < 1.0 {
            let mut raws = Vec::with_capacity(grads.len());
            for g in grads.iter_mut() {
                raws.push(exclusive(op, g)?);
            }
            let mut job = self.job(op);
            for (raw, &n) in raws.iter().zip(&sizes) {
                job.dispatch(
                    &SCALE_INPLACE,
                    &[u(op, n)?, scale.to_bits()],
                    &[raw],
                    self.lanes(n)?,
                )?;
            }
            job.commit()?;
        }
        Ok(norm)
    }

    /// In place on `param`, `moment1` and `moment2`, which must each be the
    /// sole owner of their device buffer. The old state is copied to scratch
    /// first and restored on the device if any lane saw a non-finite value;
    /// the call still returns `Ok(())` and the fault surfaces as
    /// [`OjasError::NonFinite`] at the next sync point, naming `adamw_step`
    /// unless an op recorded earlier faulted first (see the module docs).
    fn optimizer_scratch_bytes(
        &self,
        kind: OptimizerKind,
        rows: usize,
        cols: usize,
    ) -> Result<Option<u64>, OjasError> {
        optimizer_scratch(kind, rows, cols).map(Some)
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
        let op = ADAM;
        let n = adamw_step_dims(param, grad, moment1, moment2)?;
        self.placed(op, param)?;
        let gv = self.placed(op, grad)?;
        self.placed(op, moment1)?;
        self.placed(op, moment2)?;
        let (_, bc1, bc2) = check_adamw(config, step)?;
        let grid = self.lanes(n)?;
        let one_minus_b1 = 1.0 - config.beta1;
        let mut flags = 0u32;
        if config.weight_decay != 0.0 {
            flags |= 1;
        }
        if one_minus_b1 < 0.5 {
            flags |= 2;
        }
        let words = [
            u(op, n)?,
            ((config.lr / bc1) as f32).to_bits(),
            (bc2.sqrt() as f32).to_bits(),
            (config.eps as f32).to_bits(),
            (config.beta1 as f32).to_bits(),
            (one_minus_b1 as f32).to_bits(),
            (config.beta2 as f32).to_bits(),
            ((1.0 - config.beta2) as f32).to_bits(),
            ((1.0 - config.lr * config.weight_decay) as f32).to_bits(),
            flags,
        ];
        let pb = exclusive(op, param)?;
        let mb = exclusive(op, moment1)?;
        let vb = exclusive(op, moment2)?;
        let bytes = (n as u64) * 4;
        let mut job = self.job(op);
        let gb = bind(op, &mut job, &gv)?;
        let old_p = job.scratch(bytes)?;
        let old_m = job.scratch(bytes)?;
        let old_v = job.scratch(bytes)?;
        let status = job.scratch(16)?;
        job.copy(&pb, 0, &old_p, bytes);
        job.copy(&mb, 0, &old_m, bytes);
        job.copy(&vb, 0, &old_v, bytes);
        job.clear(&status);
        job.dispatch(&ADAM_UPDATE, &words, &[&gb, &pb, &mb, &vb, &status], grid)?;
        job.dispatch(
            &ADAM_COMMIT,
            &[u(op, n)?],
            &[&old_p, &old_m, &old_v, &pb, &mb, &vb, &status],
            grid,
        )?;
        job.commit()
    }

    /// f32 Newton-Schulz with the CPU reference's semantics: Nesterov
    /// momentum, Frobenius normalization with [`MUON_NS_EPS`] added to the
    /// norm, five steps with [`MUON_NS5_A`]/[`MUON_NS5_B`]/[`MUON_NS5_C`] on
    /// the wide orientation (a tall matrix is transposed first and back
    /// after), scale `max(1, rows/cols)^0.5`, and decoupled decay.
    /// [`Ns5Precision::Bf16`] is [`OjasError::Unsupported`], refused before
    /// anything is charged or written.
    ///
    /// In place on `param` and `momentum`, which must each be the sole owner
    /// of their device buffer. Everything is computed into scratch with a
    /// per-call fault word; a final kernel writes the new state only if that
    /// word is clear. Otherwise nothing changes and the fault surfaces as
    /// [`OjasError::NonFinite`] at the next sync point, as for AdamW.
    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        let op = MUON;
        let d = muon_ns5_step_dims(param, grad, momentum)?;
        let (rows, cols) = (d.rows, d.cols);
        require_ns5(5)?;
        if config.ns5 != Ns5Precision::F32 {
            return Err(OjasError::Unsupported {
                op: op.name(),
                detail: format!(
                    "wgpu runs Newton-Schulz in f32 only; {:?} is not implemented here",
                    config.ns5
                ),
            });
        }
        self.placed(op, param)?;
        let gv = self.placed(op, grad)?;
        self.placed(op, momentum)?;
        let scalars = [config.lr, config.momentum, config.weight_decay];
        if scalars.iter().any(|v| !v.is_finite()) {
            return Err(OjasError::NonFinite { op: op.name() });
        }
        for (name, value) in [
            ("lr", config.lr),
            ("momentum", config.momentum),
            ("weight_decay", config.weight_decay),
        ] {
            if value < 0.0 {
                return Err(OjasError::OutOfRange {
                    op: op.name(),
                    detail: format!("{name} {value} is negative"),
                });
            }
        }
        let n = product(op, &[rows, cols])?;
        // Newton-Schulz runs on the wide orientation [r, c], r <= c.
        let tall = rows > cols;
        let (r, c) = if tall { (cols, rows) } else { (rows, cols) };
        let rr = product(op, &[r, r])?;
        let grid = self.lanes(n)?;
        self.lanes(rr)?;
        self.fits(op, n)?;
        self.fits(op, rr)?;
        let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
        let alpha = (-config.lr * scale) as f32;
        let decays = config.weight_decay != 0.0;
        let decay = (1.0 - config.lr * config.weight_decay) as f32;
        let mom = config.momentum as f32;
        let (a, b, cc) = (MUON_NS5_A as f32, MUON_NS5_B as f32, MUON_NS5_C as f32);

        let pb = exclusive(op, param)?;
        let mb = exclusive(op, momentum)?;
        let bytes = (n as u64) * 4;
        let mut job = self.job(op);
        let gb = bind(op, &mut job, &gv)?;
        let local = job.scratch(16)?;
        let norm = job.scratch(16)?;
        let buf = job.scratch(bytes)?;
        let upd = job.scratch(bytes)?;
        let mut x = job.scratch(bytes)?;
        let mut x_next = job.scratch(bytes)?;
        let bx = job.scratch(bytes)?;
        let am = job.scratch((rr as u64) * 4)?;
        let a2 = job.scratch((rr as u64) * 4)?;
        let bm = job.scratch((rr as u64) * 4)?;

        job.local_fault(&local);
        job.dispatch(
            &MUON_MOMENTUM,
            &[u(op, n)?, mom.to_bits(), u32::from(config.nesterov)],
            &[&gb, &mb, &buf, &upd],
            grid,
        )?;
        let src = if tall {
            let t = job.scratch(bytes)?;
            self.permute_into(op, &mut job, &upd, &t, &[rows, cols], &[1, 0])?;
            t
        } else {
            upd.clone()
        };
        self.global_norm(op, &mut job, &[(&src, n)], &norm)?;
        job.dispatch(
            &MUON_NORMALIZE,
            &[u(op, n)?, (MUON_NS_EPS as f32).to_bits()],
            &[&src, &x, &norm],
            grid,
        )?;
        for _ in 0..5 {
            // A = X X^T, A2 = A A, B = b A + c A2, X' = a X + B X.
            self.gemm(op, &mut job, Mm::Nt, &x, &x, &am, (r, r, c), [c, 1, 1, c])?;
            self.gemm(op, &mut job, Mm::Nn, &am, &am, &a2, (r, r, r), [r, 1, r, 1])?;
            self.lincomb(op, &mut job, rr, b, &am, cc, &a2, &bm)?;
            self.gemm(op, &mut job, Mm::Nn, &bm, &x, &bx, (r, c, r), [r, 1, c, 1])?;
            self.lincomb(op, &mut job, n, a, &x, 1.0, &bx, &x_next)?;
            std::mem::swap(&mut x, &mut x_next);
        }
        let ortho = if tall {
            let t = job.scratch(bytes)?;
            self.permute_into(op, &mut job, &x, &t, &[r, c], &[1, 0])?;
            t
        } else {
            x
        };
        let new_p = job.scratch(bytes)?;
        job.dispatch(
            &MUON_APPLY,
            &[
                u(op, n)?,
                alpha.to_bits(),
                decay.to_bits(),
                u32::from(decays),
            ],
            &[&pb, &ortho, &new_p],
            grid,
        )?;
        job.global_fault();
        job.dispatch(
            &MUON_COMMIT,
            &[u(op, n)?],
            &[&new_p, &buf, &pb, &mb, &local],
            grid,
        )?;
        job.commit()
    }

    /// `acc += grad` in `acc`'s own buffer when `acc` solely owns it from
    /// byte 0; otherwise the sum goes to a new buffer that replaces `acc`
    /// (other handles keep the old values). A non-finite sum is deferred like
    /// every wgpu fault: `Ok` here, `NonFinite` naming this op at the next
    /// sync point, and `acc` is then invalid (the trait's rule for a
    /// deferring backend). Shape, dtype and placement are refused first.
    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        let op = ACC;
        let n = accumulate_grad_dims(acc, grad)?;
        self.placed(op, acc)?;
        let gv = self.placed(op, grad)?;
        let grid = self.lanes(n)?;
        match exclusive(op, acc) {
            Ok(ab) => {
                let mut job = self.job(op);
                let gb = bind(op, &mut job, &gv)?;
                job.dispatch(&ADD_INPLACE, &[u(op, n)?], &[&gb, &ab], grid)?;
                job.commit()
            }
            // Shared, or a view that does not start at byte 0.
            Err(OjasError::Shape { .. }) => {
                let av = self.placed(op, acc)?;
                let mut out = self.pointwise(op, &ADD_FWD, &[&av, &gv], 1, &[])?;
                *acc = out.pop().ok_or_else(|| shape(op, "no output"))?;
                Ok(())
            }
            Err(other) => Err(other),
        }
    }

    /// Copy `src` `[B, Tn, Hkv, D]` into `cache` `[B, Tcap, Hkv, D]` at time
    /// `at`. Everything [`kv_cache_write_dims`] checks, placement, and a
    /// cache shared with another handle are refused before anything is
    /// recorded. `src` is checked for non-finite values on the device into
    /// a per-call fault word, and the copy runs only if that word is clear,
    /// so the cache is unchanged on every error; a non-finite source
    /// surfaces as `NonFinite` naming this op at the next sync point. Words
    /// move as `u32`, so the cache holds `src`'s bits exactly.
    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        let op = KVW;
        let dims = kv_cache_write_dims(cache, src, at)?;
        self.placed(op, cache)?;
        let sv = self.placed(op, src)?;
        let row = product(op, &[dims.kv_heads, dims.head_dim])?;
        let per_src = product(op, &[dims.new, row])?;
        let per_dst = product(op, &[dims.capacity, row])?;
        u(op, product(op, &[dims.batch, per_dst])?)?;
        let offset = product(op, &[at, row])?;
        let n = sv.elems;
        let grid = self.lanes(n)?;
        let words = [u(op, n)?, u(op, per_src)?, u(op, per_dst)?, u(op, offset)?];
        let cb = exclusive(op, cache)?;
        let mut job = self.job(op);
        let sb = bind(op, &mut job, &sv)?;
        let status = job.scratch(16)?;
        job.local_fault(&status);
        self.check_finite(op, &mut job, &sb, n)?;
        job.global_fault();
        job.dispatch(&KV_WRITE, &words, &[&sb, &cb, &status], grid)?;
        job.commit()
    }

    /// Mean cross-entropy of `input @ weight^T` with at most `chunk.rows x
    /// chunk.cols` logits alive: one scratch tile, reused. For each block
    /// of rows, every vocabulary tile is computed by the GEMM and folded
    /// into the rows' running max and sum (an online softmax); with
    /// `want_grad`, each tile is then turned into its seed-1 gradient in
    /// place and fed to two accumulating GEMMs, `grad_input[rows] += G W`
    /// and `grad_weight[cols] += G^T x`. A row block whose vocabulary fits
    /// in one tile reuses that tile; otherwise its tiles are recomputed.
    /// Targets are checked on the host and the valid count is global; an
    /// all-ignored batch is `NonFinite`, synchronously. A non-finite logit,
    /// loss or gradient is reported at the next sync point.
    fn linear_cross_entropy_mean(
        &self,
        input: &Tensor,
        weight: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
        chunk: CeChunk,
        want_grad: bool,
    ) -> Result<LinearCe, OjasError> {
        let op = LCE;
        let dims = linear_ce_dims(input, weight, targets, chunk)?;
        let (n, d, vocab) = (dims.rows, dims.model_dim, dims.vocab);
        let xv = self.placed(op, input)?;
        let wv = self.placed(op, weight)?;
        let tv = self.placed(op, targets)?;
        let valid = valid_targets(op, self.ids(op, &tv)?, vocab, ignore_index)?;
        let (rc, cc) = (chunk.rows.min(n), chunk.cols.min(vocab));
        let tile_elems = product(op, &[rc, cc])?;
        u(op, product(op, &[n, d])?)?;
        u(op, product(op, &[vocab, d])?)?;
        u(op, product(op, &[3, n])?)?;
        self.fits(op, tile_elems)?;
        self.fits(op, 3 * n)?;
        self.groups(rc)?;
        self.lanes(tile_elems)?;
        let (has_ignore, ignore) = (u32::from(ignore_index.is_some()), ignore_index.unwrap_or(0));
        let denom = valid as f32;
        let (loss, loss_b) = self.out(op, &[])?;
        let grads = if want_grad {
            Some((self.out(op, &[n, d])?, self.out(op, &[vocab, d])?))
        } else {
            None
        };
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let wb = bind(op, &mut job, &wv)?;
        let tb = bind(op, &mut job, &tv)?;
        let tile = job.scratch((tile_elems as u64) * 4)?;
        let stats = job.scratch((3 * n as u64) * 4)?;
        let row_loss = job.scratch((n as u64) * 4)?;
        let single = cc == vocab;
        let logits = |job: &mut Job<'_>, r0: usize, rr: usize, c0: usize, cw: usize| {
            self.gemm_at(
                op,
                job,
                Mm::Nt,
                &xb,
                &wb,
                &tile,
                (rr, cw, d),
                [d, 1, 1, d],
                Place {
                    offsets: [r0 * d, c0 * d, 0],
                    accumulate: false,
                },
            )
        };
        for r0 in (0..n).step_by(rc) {
            let rr = rc.min(n - r0);
            for c0 in (0..vocab).step_by(cc) {
                let cw = cc.min(vocab - c0);
                logits(&mut job, r0, rr, c0, cw)?;
                job.dispatch(
                    &LCE_STATS,
                    &[
                        u(op, rr)?,
                        u(op, cw)?,
                        has_ignore,
                        ignore,
                        u(op, c0)?,
                        u(op, r0)?,
                        u32::from(c0 == 0),
                        u(op, n)?,
                    ],
                    &[&tile, &tb, &stats],
                    self.groups(rr)?,
                )?;
            }
            let Some(((_, gxb), (_, gwb))) = &grads else {
                continue;
            };
            for c0 in (0..vocab).step_by(cc) {
                let cw = cc.min(vocab - c0);
                if !single {
                    logits(&mut job, r0, rr, c0, cw)?;
                }
                let elems = rr * cw;
                job.dispatch(
                    &LCE_GRAD,
                    &[
                        u(op, elems)?,
                        u(op, cw)?,
                        has_ignore,
                        ignore,
                        u(op, c0)?,
                        u(op, r0)?,
                        denom.to_bits(),
                        u(op, n)?,
                    ],
                    &[&stats, &tb, &tile],
                    self.lanes(elems)?,
                )?;
                self.gemm_at(
                    op,
                    &mut job,
                    Mm::Nn,
                    &tile,
                    &wb,
                    gxb,
                    (rr, d, cw),
                    [cw, 1, d, 1],
                    Place {
                        offsets: [0, c0 * d, r0 * d],
                        accumulate: c0 > 0,
                    },
                )?;
                self.gemm_at(
                    op,
                    &mut job,
                    Mm::Tn,
                    &tile,
                    &xb,
                    gwb,
                    (cw, d, rr),
                    [1, cw, d, 1],
                    Place {
                        offsets: [0, r0 * d, c0 * d],
                        accumulate: r0 > 0,
                    },
                )?;
            }
        }
        job.dispatch(
            &LCE_FINISH,
            &[u(op, n)?, 0, has_ignore, ignore],
            &[&stats, &tb, &row_loss],
            self.lanes(n)?,
        )?;
        self.sum(
            op,
            &mut job,
            n,
            [&row_loss, &row_loss, &row_loss],
            0,
            denom,
            None,
            &loss_b,
        )?;
        job.commit()?;
        let (grad_input, grad_weight) = match grads {
            Some(((gx, _), (gw, _))) => (Some(gx), Some(gw)),
            None => (None, None),
        };
        Ok(LinearCe {
            loss,
            grad_input,
            grad_weight,
        })
    }

    /// Split-key attention against a time-major cache: a pass over key
    /// ranges (sized by [`cached_attention_splits`], so one decode query
    /// still fills the GPU) and a merge that combines the ranges in
    /// ascending order. Rows never read keys past their own position.
    /// Every score and output is checked; a non-finite one is reported at
    /// the next sync point.
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        let op = CATTN;
        let dims = cached_attention_dims(q, k_cache, v_cache, kv_len)?;
        let head_dim = u(op, dims.head_dim)?;
        if head_dim > ATTENTION_MAX_HEAD_DIM {
            return Err(OjasError::UnsupportedHeadDim {
                head_dim,
                limit: ATTENTION_MAX_HEAD_DIM,
            });
        }
        let scale = sdpa_scale(head_dim)?;
        let qv = self.placed(op, q)?;
        let kv = self.placed(op, k_cache)?;
        let vv = self.placed(op, v_cache)?;
        u(op, kv.elems)?;
        let row_heads = product(op, &[dims.new, dims.heads])?;
        let rows = product(op, &[dims.batch, row_heads])?;
        let (split, splits) = cached_attention_splits(rows, kv_len)?;
        let partial = product(op, &[rows, splits, dims.head_dim + 2])?;
        u(op, partial)?;
        self.fits(op, partial)?;
        let max = self.ctx.limits().max_compute_workgroups_per_dimension;
        let split_grid = (u(op, splits)?, u(op, row_heads)?, u(op, dims.batch)?);
        for g in [split_grid.0, split_grid.1, split_grid.2] {
            if g > max {
                return Err(OjasError::OutOfRange {
                    op: op.name(),
                    detail: format!("grid {split_grid:?} exceeds {max} per axis"),
                });
            }
        }
        let words = [
            u(op, dims.batch)?,
            u(op, dims.new)?,
            u(op, dims.heads)?,
            u(op, dims.kv_heads)?,
            head_dim,
            u(op, dims.capacity)?,
            u(op, kv_len)?,
            u(op, split)?,
            u(op, splits)?,
            scale.to_bits(),
        ];
        let (y, yb) = self.out(op, qv.shape())?;
        let mut job = self.job(op);
        let qb = bind(op, &mut job, &qv)?;
        let kb = bind(op, &mut job, &kv)?;
        let vb = bind(op, &mut job, &vv)?;
        let part = job.scratch((partial as u64) * 4)?;
        job.dispatch(&CATTN_SPLIT, &words, &[&qb, &kb, &vb, &part], split_grid)?;
        job.dispatch(
            &CATTN_MERGE,
            &words,
            &[&yb, &part],
            (split_grid.1, split_grid.2, 1),
        )?;
        job.commit()?;
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::RMS_NORM_EPS;
    use ojas_cpu::CpuBackend;

    const ELEMENT_ABS_TOL: f64 = 1.0e-4;

    /// The shared comparator: a non-finite value never measures as close.
    fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        ojas_kernels::max_abs(a, b).expect("equal lengths")
    }

    #[test]
    fn a_failed_read_keeps_the_fault_for_the_next_sync() {
        // Pre-fix, the read cleared the device fault word in the submission
        // that copied it, so a read that failed afterwards lost the fault and
        // the next sync reported Ok.
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        gpu.sync().unwrap();
        let x = gpu
            .upload(&Tensor::from_f32(&[f32::INFINITY, 1.0], &[2], &budget).unwrap())
            .unwrap();
        let _y = gpu.silu_forward(&x).unwrap();
        gpu.ctx.fail_next_read();
        assert!(matches!(gpu.sync(), Err(OjasError::Backend { .. })));
        match gpu.sync() {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "silu_forward"),
            other => panic!("the fault was lost with the failed read: {other:?}"),
        }
        gpu.sync().expect("reported once, then clear");
    }

    #[test]
    fn rms_backward_weight_matches_cpu() {
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        let cpu = CpuBackend::new(budget.clone()).with_numerics(Numerics::Exact);
        let x = Tensor::from_f32(
            &[0.5, -0.25, 1.0, 0.0, 0.2, -0.7, 0.3, -1.2],
            &[2, 4],
            &budget,
        )
        .unwrap();
        let w = Tensor::from_f32(&[1.0, 0.5, -0.5, 2.0], &[4], &budget).unwrap();
        let gy = Tensor::from_f32(
            &[0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0, 0.5],
            &[2, 4],
            &budget,
        )
        .unwrap();
        let (cpu_dx, cpu_dw) = cpu.rms_norm_backward(&x, &w, &gy, RMS_NORM_EPS).unwrap();
        let before = budget.device_readbacks();
        let (gpu_dx, gpu_dw) = gpu
            .rms_norm_backward(
                &gpu.upload(&x).unwrap(),
                &gpu.upload(&w).unwrap(),
                &gpu.upload(&gy).unwrap(),
                RMS_NORM_EPS,
            )
            .unwrap();
        assert_eq!(
            budget.device_readbacks(),
            before,
            "rms backward read a tensor back"
        );
        let dw = gpu.download(&gpu_dw).unwrap().to_f32_vec().unwrap();
        let dx = gpu.download(&gpu_dx).unwrap().to_f32_vec().unwrap();
        // Positive control: the two downloads are counted on this budget.
        assert_eq!(budget.device_readbacks().0, before.0 + 2);
        assert!(
            dw.iter().any(|value| value.abs() > 1e-6),
            "weight gradient is all zeros: {dw:?}"
        );
        let cpu_dw = cpu_dw.to_f32_vec().unwrap();
        let cpu_dx = cpu_dx.to_f32_vec().unwrap();
        assert!(
            max_abs(&dw, &cpu_dw) <= ELEMENT_ABS_TOL,
            "dW {dw:?} vs {cpu_dw:?}"
        );
        assert!(
            max_abs(&dx, &cpu_dx) <= ELEMENT_ABS_TOL,
            "dx {dx:?} vs {cpu_dx:?}"
        );
    }

    /// The raw buffer behind a tensor this backend uploaded.
    fn raw_of(t: &Tensor) -> wgpu::Buffer {
        t.device_buffer()
            .and_then(|d| d.as_any().downcast_ref::<WgpuBuffer>())
            .expect("a wgpu tensor")
            .raw()
            .expect("a live buffer")
            .clone()
    }

    /// Record `check_finite` over the first `n` floats of `buf` under fault
    /// id `id`, as an op of that index would raise.
    fn raise_as(gpu: &WgpuBackend, id: u32, buf: &wgpu::Buffer, n: usize) {
        let mut job = gpu.ctx.job(&gpu.budget, id);
        job.dispatch(&CHECK_FINITE, &[n as u32], &[buf], gpu.lanes(n).unwrap())
            .unwrap();
        job.commit().unwrap();
    }

    #[test]
    fn fault_ids_at_both_ends_of_both_mask_words_land_on_their_own_bit() {
        // Pre-fix, the fault word was one u32 mask holding exactly 32 ops:
        // an op of index 32 had no bit, so cast_bf16 ran with mask 0 and a
        // 33rd op could never report a fault.
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        gpu.sync().unwrap();
        let bad = gpu
            .upload(&Tensor::from_f32(&[1.0, f32::NAN], &[2], &budget).unwrap())
            .unwrap();
        let buf = raw_of(&bad);
        for id in [1, 31, 32, 33, 63, FAULT_OPS] {
            raise_as(&gpu, id, &buf, 2);
            gpu.ctx.sync().unwrap();
            let (bits, first) = gpu.ctx.take_faults();
            assert_eq!(bits, 1u64 << (id - 1), "id {id}: mask {bits:#x}");
            assert_eq!(first, id, "id {id}");
        }
        // Every read released exactly what it saw: nothing is held over.
        gpu.ctx.sync().unwrap();
        assert_eq!(gpu.ctx.take_faults(), (0, 0));
    }

    #[test]
    fn a_fault_id_past_the_mask_is_refused_before_anything_is_recorded() {
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        let bad = gpu
            .upload(&Tensor::from_f32(&[f32::NAN], &[1], &budget).unwrap())
            .unwrap();
        let mut job = gpu.ctx.job(&gpu.budget, FAULT_OPS + 1);
        let refused = job.dispatch(&CHECK_FINITE, &[1], &[&raw_of(&bad)], (1, 1, 1));
        assert!(
            matches!(refused, Err(OjasError::Backend { .. })),
            "{refused:?}"
        );
        drop(job);
        gpu.sync().expect("the refused dispatch recorded nothing");
    }

    #[test]
    fn first_fault_precedence_holds_across_the_two_mask_words() {
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        gpu.sync().unwrap();
        let bad = gpu
            .upload(&Tensor::from_f32(&[f32::INFINITY, 1.0], &[2], &budget).unwrap())
            .unwrap();
        let buf = raw_of(&bad);
        let first_op = |gpu: &WgpuBackend| match gpu.sync() {
            Err(OjasError::NonFinite { op }) => op,
            other => panic!("expected a deferred fault, got {other:?}"),
        };

        // Low word (silu_forward, index 16), then high word (cast_bf16, 32).
        let _y = gpu.silu_forward(&bad).unwrap();
        raise_as(&gpu, CAST.fault_id(), &buf, 2);
        assert_eq!(first_op(&gpu), "silu_forward");
        gpu.sync().expect("both bits were reported by one sync");

        // High word first, then low.
        raise_as(&gpu, CAST.fault_id(), &buf, 2);
        let _y = gpu.silu_forward(&bad).unwrap();
        assert_eq!(first_op(&gpu), "cast_bf16");
        gpu.sync().unwrap();

        // A read that fails after its copy keeps a high-word fault held.
        raise_as(&gpu, CAST.fault_id(), &buf, 2);
        gpu.ctx.fail_next_read();
        assert!(matches!(gpu.sync(), Err(OjasError::Backend { .. })));
        let _y = gpu.silu_forward(&bad).unwrap();
        assert_eq!(first_op(&gpu), "cast_bf16");
        gpu.sync().unwrap();
    }

    #[test]
    fn kv_write_refuses_a_bad_source_under_an_op_in_the_high_mask_word() {
        // Pre-fix, kv_write read only word 0 of its per-call fault word, so
        // under an op whose bit is in word 1 a non-finite source was written.
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        gpu.sync().unwrap();
        let src = gpu
            .upload(&Tensor::from_f32(&[f32::NAN, 2.0, 3.0, 4.0], &[4], &budget).unwrap())
            .unwrap();
        let cache = gpu
            .upload(&Tensor::from_f32(&[0.0; 8], &[8], &budget).unwrap())
            .unwrap();
        let mut job = gpu.ctx.job(&gpu.budget, CAST.fault_id());
        let status = job.scratch(16).unwrap();
        job.local_fault(&status);
        job.dispatch(&CHECK_FINITE, &[4], &[&raw_of(&src)], (1, 1, 1))
            .unwrap();
        job.global_fault();
        job.dispatch(
            &KV_WRITE,
            &[4, 4, 8, 0],
            &[&raw_of(&src), &raw_of(&cache), &status],
            (1, 1, 1),
        )
        .unwrap();
        job.commit().unwrap();
        match gpu.sync() {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "cast_bf16"),
            other => panic!("the refused write raised nothing: {other:?}"),
        }
        let after = gpu.download(&cache).unwrap().to_f32_vec().unwrap();
        assert_eq!(after, vec![0.0; 8], "a faulting source reached the cache");
    }

    #[test]
    fn muon_commit_refuses_a_bad_step_under_an_op_in_the_high_mask_word() {
        // Pre-fix, muon_commit read only word 0 of its per-call fault word.
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        gpu.sync().unwrap();
        let up = |v: &[f32]| {
            gpu.upload(&Tensor::from_f32(v, &[v.len()], &budget).unwrap())
                .unwrap()
        };
        let new_p = up(&[f32::NAN, 9.0]);
        let new_m = up(&[9.0, 9.0]);
        let p = up(&[1.0, 2.0]);
        let m = up(&[3.0, 4.0]);
        let mut job = gpu.ctx.job(&gpu.budget, CAST.fault_id());
        let status = job.scratch(16).unwrap();
        job.local_fault(&status);
        job.dispatch(&CHECK_FINITE, &[2], &[&raw_of(&new_p)], (1, 1, 1))
            .unwrap();
        job.global_fault();
        job.dispatch(
            &MUON_COMMIT,
            &[2],
            &[
                &raw_of(&new_p),
                &raw_of(&new_m),
                &raw_of(&p),
                &raw_of(&m),
                &status,
            ],
            (1, 1, 1),
        )
        .unwrap();
        job.commit().unwrap();
        match gpu.sync() {
            Err(OjasError::NonFinite { op }) => assert_eq!(op, "cast_bf16"),
            other => panic!("the refused commit raised nothing: {other:?}"),
        }
        assert_eq!(gpu.download(&p).unwrap().to_f32_vec().unwrap(), [1.0, 2.0]);
        assert_eq!(gpu.download(&m).unwrap().to_f32_vec().unwrap(), [3.0, 4.0]);
    }
}
