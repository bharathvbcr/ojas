//! Device-resident [`Backend`] over wgpu.
//!
//! Upload policy: every op takes tensors that live on this backend's
//! [`WgpuContext`] and returns tensors that stay there. A host tensor passed
//! to an op is [`OjasError::Placement`] with `found: None`; a tensor of
//! another device or another wgpu context is `Placement` as well. Data moves
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
    check_adamw, clip_scale, permute_output_shape, require_ns5, sdpa_scale, AdamWConfig, Backend,
    BackendId, Budget, DType, MuonNs5Config, Numerics, OjasError, PerHeadGateGrad, Tensor,
    ValueResidualGrad, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS,
};
use ojas_device::DeviceError;
use ojas_kernels::{
    attention_tiles, fold_grid, gemm_grid, gemm_tile, WgslModule, ATTENTION_MAX_HEAD_DIM,
    GEMM_BIG_TILE, GEMM_TILE,
};

use crate::context::{Job, Kernel, Slot, WgpuBuffer, WgpuContext};

const OP_NAMES: [&str; 28] = [
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
];

#[derive(Clone, Copy)]
struct Op(u32);

impl Op {
    fn name(self) -> &'static str {
        OP_NAMES.get(self.0 as usize).copied().unwrap_or("wgpu")
    }

    fn bit(self) -> u32 {
        1u32 << self.0
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

const fn k(module: WgslModule, entry: &'static str, slots: &'static [Slot]) -> Kernel {
    Kernel {
        module,
        entry,
        slots,
    }
}

use Slot::{R, W};
use WgslModule::{Gemm, Layout, Loss, Norm, Optim, Pointwise, Reduce};

const PERMUTE_K: Kernel = k(Layout, "permute", &[R(2), R(3), W(4)]);

const GEMM_NT: Kernel = k(Gemm, "gemm_nt", &[R(2), R(3), W(4)]);
const GEMM_NN: Kernel = k(Gemm, "gemm_nn", &[R(2), R(3), W(4)]);
const GEMM_TN: Kernel = k(Gemm, "gemm_tn", &[R(2), R(3), W(4)]);
const GEMM_NT_BIG: Kernel = k(Gemm, "gemm_nt_big", &[R(2), R(3), W(4)]);
const GEMM_NN_BIG: Kernel = k(Gemm, "gemm_nn_big", &[R(2), R(3), W(4)]);
const GEMM_TN_BIG: Kernel = k(Gemm, "gemm_tn_big", &[R(2), R(3), W(4)]);
const SILU_FWD: Kernel = k(Pointwise, "silu_fwd", &[R(2), W(6)]);
const SILU_BWD: Kernel = k(Pointwise, "silu_bwd", &[R(2), R(3), W(6)]);
const MUL_FWD: Kernel = k(Pointwise, "mul_fwd", &[R(2), R(3), W(6)]);
const MUL_BWD: Kernel = k(Pointwise, "mul_bwd", &[R(2), R(3), R(4), W(6), W(7)]);
const ADD_FWD: Kernel = k(Pointwise, "add_fwd", &[R(2), R(3), W(6)]);
const CHECK_FINITE: Kernel = k(Pointwise, "check_finite", &[R(2)]);
const SCALE_INPLACE: Kernel = k(Pointwise, "scale_inplace", &[W(6)]);
const VR_FWD: Kernel = k(Pointwise, "vr_fwd", &[R(2), R(3), R(4), W(6)]);
const VR_BWD: Kernel = k(Pointwise, "vr_bwd", &[R(2), R(4), W(6), W(7)]);
const GATE_FWD: Kernel = k(Pointwise, "gate_fwd", &[R(2), R(3), R(4), W(6)]);
const GATE_BWD: Kernel = k(Pointwise, "gate_bwd", &[R(2), R(3), R(4), R(5), W(6), W(7)]);
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
        "attn_fwd" => &[R(2), R(3), R(4), W(7)],
        "attn_bwd_prep" => &[R(2), R(3), R(4), R(5), W(7)],
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
    scale: f32,
    module: WgslModule,
    fwd_rows: usize,
    bwd_rows: usize,
}

/// Elements one stage-one reduction group covers (`CHUNK` in the WGSL).
const CHUNK: usize = 4096;

fn shape(op: Op, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op: op.name(),
        detail: detail.into(),
    }
}

fn same_shape(op: Op, a: &[usize], b: &[usize]) -> Result<(), OjasError> {
    if a == b {
        Ok(())
    } else {
        Err(shape(op, format!("shape {a:?} does not match {b:?}")))
    }
}

fn product(op: Op, dims: &[usize]) -> Result<usize, OjasError> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| OjasError::OutOfRange {
            op: op.name(),
            detail: "shape product overflows".to_string(),
        })
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

/// Device-resident GPU backend. See the module docs for the upload policy.
pub struct WgpuBackend {
    ctx: WgpuContext,
    budget: Budget,
    legacy_attention: std::sync::atomic::AtomicBool,
    legacy_gemm: std::sync::atomic::AtomicBool,
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
        Self {
            ctx,
            budget,
            legacy_attention: std::sync::atomic::AtomicBool::new(false),
            legacy_gemm: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Route SDPA through the one-row-per-lane kernel the tiled one replaced.
    /// For the interleaved before/after measurement only.
    #[doc(hidden)]
    pub fn set_legacy_attention(&self, on: bool) {
        self.legacy_attention
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Route every GEMM through the 64x64 tile. Measurement only.
    #[doc(hidden)]
    pub fn set_legacy_gemm(&self, on: bool) {
        self.legacy_gemm
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn context(&self) -> &WgpuContext {
        &self.ctx
    }

    /// Submit everything recorded, wait for it, and report a deferred fault
    /// as [`OjasError::NonFinite`] or a wgpu error as [`OjasError::Backend`].
    pub fn sync(&self) -> Result<(), OjasError> {
        self.ctx.sync()?;
        self.faults()
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
        Err(OjasError::NonFinite {
            op: Op(index).name(),
        })
    }

    fn input<'t>(&self, op: Op, t: &'t Tensor, dtype: DType) -> Result<In<'t>, OjasError> {
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
        if t.dtype() != dtype {
            return Err(OjasError::Dtype {
                op: op.name(),
                expected: dtype,
                got: t.dtype(),
            });
        }
        if !t.is_contiguous()? {
            return Err(shape(op, "non-contiguous view is not supported"));
        }
        let elems = t.num_elements()?;
        if elems == 0 || t.shape().contains(&0) {
            return Err(shape(op, "empty tensor"));
        }
        Ok(In { t, buf, elems })
    }

    fn f32_in<'t>(&self, op: Op, t: &'t Tensor) -> Result<In<'t>, OjasError> {
        self.input(op, t, DType::F32)
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
            .ok_or_else(|| shape(op, "byte length overflows"))?;
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
            .ok_or_else(|| shape(op, "byte length overflows"))?;
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
        self.ctx.job(&self.budget, op.bit())
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
        (m, n, kk): (usize, usize, usize),
        strides: [usize; 4],
    ) -> Result<(), OjasError> {
        let max = self.ctx.limits().max_compute_workgroups_per_dimension;
        let legacy = self.legacy_gemm.load(std::sync::atomic::Ordering::Relaxed);
        let big = !legacy && gemm_tile(m, n) == GEMM_BIG_TILE;
        let (gx, gy) = if big {
            gemm_grid(m, n, max)?
        } else {
            let (gx, gy) = (n.div_ceil(GEMM_TILE as usize), m.div_ceil(GEMM_TILE as usize));
            (u(op, gx)?, u(op, gy)?)
        };
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
            .ok_or_else(|| shape(op, "column partials overflow"))?;
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
    /// `dims` with [`permute_output_shape`]. Words move as `u32`, so the bits
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
                .ok_or_else(|| shape(op, "norm partials overflow"))?;
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

    fn rms_layout(
        &self,
        op: Op,
        x: &[usize],
        w: &[usize],
        eps: f32,
    ) -> Result<(usize, usize), OjasError> {
        if !eps.is_finite() {
            return Err(OjasError::NonFinite { op: op.name() });
        }
        let dim = *x.last().ok_or_else(|| shape(op, "rms_norm input rank 0"))?;
        if w.len() != 1 || w[0] != dim {
            return Err(shape(op, format!("rms weight {w:?} != last dim {dim}")));
        }
        let rows = product(op, &x[..x.len() - 1])?;
        if rows == 0 {
            return Err(shape(op, "empty tensor"));
        }
        Ok((rows, dim))
    }

    /// `(layout, heads, time, rows, dim)`. Layout 0 is a table shaped like
    /// `x`; layout 1 is `x` `[B, T, H, D]` with `[T, D]` tables.
    fn rope_layout(
        &self,
        op: Op,
        x: &[usize],
        cos: &[usize],
        sin: &[usize],
    ) -> Result<(u32, usize, usize, usize, usize), OjasError> {
        if cos != sin {
            return Err(shape(op, format!("cos shape {cos:?} != sin shape {sin:?}")));
        }
        let dim = *x.last().ok_or_else(|| shape(op, "rope input rank 0"))?;
        if dim % 2 != 0 {
            return Err(shape(op, format!("rope last dim {dim} is odd")));
        }
        let rows = product(op, &x[..x.len() - 1])?;
        if rows == 0 {
            return Err(shape(op, "empty tensor"));
        }
        if cos == x {
            return Ok((0, 1, 1, rows, dim));
        }
        if x.len() == 4 && cos.len() == 2 && cos[0] == x[1] && cos[1] == dim {
            return Ok((1, x[2], x[1], rows, dim));
        }
        Err(shape(
            op,
            format!("cos/sin shape {cos:?} does not broadcast onto {x:?}"),
        ))
    }

    fn rope(
        &self,
        op: Op,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        direction: u32,
    ) -> Result<Tensor, OjasError> {
        let xv = self.f32_in(op, x)?;
        let cv = self.f32_in(op, cos)?;
        let sv = self.f32_in(op, sin)?;
        let (layout, heads, time, rows, dim) =
            self.rope_layout(op, xv.shape(), cv.shape(), sv.shape())?;
        let half_lanes = rows
            .checked_mul(dim / 2)
            .ok_or_else(|| shape(op, "rope lanes overflow"))?;
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

    /// The call's plan after the CPU's shape rules and this device's
    /// head-dim and shared-memory limits.
    fn sdpa_dims(
        &self,
        op: Op,
        q: &[usize],
        kk: &[usize],
        v: &[usize],
    ) -> Result<SdpaPlan, OjasError> {
        if q.len() != 4 {
            return Err(shape(
                op,
                format!("sdpa query rank {} != 4 [B, H, T, D]", q.len()),
            ));
        }
        if kk != q || v != q {
            return Err(shape(
                op,
                format!("sdpa shapes q {q:?} k {kk:?} v {v:?} differ"),
            ));
        }
        let dim = q[3];
        let dim_u32 = u32::try_from(dim).map_err(|_| OjasError::OutOfRange {
            op: op.name(),
            detail: format!("head dim {dim} does not fit in u32"),
        })?;
        let scale = sdpa_scale(dim_u32)?;
        if dim_u32 > ATTENTION_MAX_HEAD_DIM {
            return Err(OjasError::UnsupportedHeadDim {
                head_dim: dim_u32,
                limit: ATTENTION_MAX_HEAD_DIM,
            });
        }
        let shared = self.ctx.limits().max_compute_workgroup_storage_size;
        let bh = product(op, &q[..2])?;
        let legacy = self
            .legacy_attention
            .load(std::sync::atomic::Ordering::Relaxed);
        let (module, fwd_rows, bwd_rows) = if legacy {
            let bytes = |tile: u32| tile * (2 * dim_u32 + 2) * 4;
            let tile = [64u32, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|&t| bytes(t) <= shared)
                .ok_or(OjasError::CapacityExceeded {
                    requested: u64::from(bytes(1)),
                    cap: u64::from(shared),
                    live: 0,
                })?;
            let module = WgslModule::AttentionLegacy {
                head_dim: dim_u32,
                key_tile: tile,
            };
            (module, 64, 64)
        } else {
            let tiles = attention_tiles(dim_u32, shared)?;
            (
                WgslModule::Attention(tiles),
                tiles.fwd_rows as usize,
                tiles.bwd_block as usize,
            )
        };
        Ok(SdpaPlan {
            bh,
            time: q[2],
            dim,
            scale,
            module,
            fwd_rows,
            bwd_rows,
        })
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

    fn linear_dims(
        &self,
        op: Op,
        x: &[usize],
        w: &[usize],
    ) -> Result<(usize, usize, usize, Vec<usize>), OjasError> {
        if w.len() != 2 {
            return Err(shape(
                op,
                format!("weight rank {} != 2 ([out, in])", w.len()),
            ));
        }
        let kin = *x
            .last()
            .ok_or_else(|| shape(op, "input rank 0 has no in-features"))?;
        if kin == 0 {
            return Err(shape(op, "in_dim is 0"));
        }
        let (nout, w_in) = (w[0], w[1]);
        if kin != w_in {
            return Err(shape(
                op,
                format!("input in-features {kin} != weight in-features {w_in}"),
            ));
        }
        let rows = product(op, &x[..x.len() - 1])?;
        if rows == 0 || nout == 0 {
            return Err(shape(op, "empty tensor"));
        }
        let mut y = x[..x.len() - 1].to_vec();
        y.push(nout);
        Ok((rows, kin, nout, y))
    }

    /// `(rows, din, heads, dh)` after the CPU's gate shape rules.
    fn gate_layout(
        &self,
        op: Op,
        x: &[usize],
        w: &[usize],
        b: &[usize],
        attn: &[usize],
    ) -> Result<(usize, usize, usize, usize), OjasError> {
        if w.len() != 2 {
            return Err(shape(op, "gate weight must be [n_head, d_model]"));
        }
        let din = *x.last().ok_or_else(|| shape(op, "gate input rank 0"))?;
        if w[1] != din {
            return Err(shape(
                op,
                format!("gate weight in {} != input dim {din}", w[1]),
            ));
        }
        let heads = w[0];
        if b.len() != 1 || b[0] != heads {
            return Err(shape(op, format!("gate bias {b:?} != [{heads}]")));
        }
        if attn.len() != x.len() + 1 {
            return Err(shape(op, "gate attn rank must be input rank + 1"));
        }
        let dh = *attn.last().ok_or_else(|| shape(op, "gate attn rank 0"))?;
        if attn[attn.len() - 2] != heads {
            return Err(shape(op, "gate attn head axis != weight rows"));
        }
        if attn[..attn.len() - 2] != x[..x.len() - 1] {
            return Err(shape(op, "gate attn prefix does not match input prefix"));
        }
        let rows = product(op, &x[..x.len() - 1])?;
        if rows == 0 {
            return Err(shape(op, "empty tensor"));
        }
        Ok((rows, din, heads, dh))
    }

    /// `(rows, vocab, ignore word, valid count, any ignored)`.
    fn ce_layout(
        &self,
        op: Op,
        logits: &In<'_>,
        targets: &In<'_>,
        ignore: Option<u32>,
    ) -> Result<(usize, usize, u32, u32, bool), OjasError> {
        let ls = logits.shape();
        let vocab = *ls
            .last()
            .ok_or_else(|| shape(op, "cross-entropy logits rank 0"))?;
        let prefix = &ls[..ls.len() - 1];
        if targets.shape() != prefix {
            return Err(shape(
                op,
                format!("targets {:?} != logits prefix {prefix:?}", targets.shape()),
            ));
        }
        let rows = product(op, prefix)?;
        let ids = self.ids(op, targets)?;
        if rows != ids.len() {
            return Err(shape(op, "target count does not match logits prefix"));
        }
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
        let any_ignored = (valid as usize) < rows;
        Ok((rows, vocab, ignore.unwrap_or(0), valid, any_ignored))
    }

    fn embed_layout(
        &self,
        op: Op,
        table: &In<'_>,
        ids: &[u32],
    ) -> Result<(usize, usize), OjasError> {
        let ts = table.shape();
        if ts.len() != 2 {
            return Err(shape(
                op,
                format!("embedding table rank {} != 2 [vocab, dim]", ts.len()),
            ));
        }
        let (vocab, dim) = (ts[0], ts[1]);
        for (n, &id) in ids.iter().enumerate() {
            if (id as usize) >= vocab {
                return Err(OjasError::OutOfRange {
                    op: op.name(),
                    detail: format!("token id {id} at {n} is outside vocab {vocab}"),
                });
            }
        }
        Ok((vocab, dim))
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
        let bytes = tensor.contiguous_bytes()?;
        let padded = (bytes.len() as u64).div_ceil(4) * 4;
        self.ctx.check_bytes(padded, &self.budget)?;
        let charge = self.budget.try_reserve(padded)?;
        let shadow: Option<Arc<[u32]>> = if tensor.dtype() == DType::U32 {
            Some(tensor.to_u32_vec()?.into())
        } else {
            None
        };
        let wb = self.ctx.upload_bytes(bytes, shadow)?;
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
        let x = self.f32_in(op, input)?;
        let out_shape = permute_output_shape(op.name(), x.shape(), dims)?;
        let (y, yb) = self.out(op, &out_shape)?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &x)?;
        self.permute_into(op, &mut job, &xb, &yb, x.shape(), dims)?;
        job.commit()?;
        Ok(y)
    }

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        let op = EMBED_F;
        let tv = self.f32_in(op, table)?;
        let iv = self.input(op, token_ids, DType::U32)?;
        let ids = self.ids(op, &iv)?;
        let (_, dim) = self.embed_layout(op, &tv, ids)?;
        let n = product(op, &[ids.len(), dim])?;
        let mut dims = iv.shape().to_vec();
        dims.push(dim);
        let grid = self.lanes(n)?;
        let (y, yb) = self.out(op, &dims)?;
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
        let tv = self.f32_in(op, table)?;
        let iv = self.input(op, token_ids, DType::U32)?;
        let gv = self.f32_in(op, grad_output)?;
        let ids = self.ids(op, &iv)?;
        let (vocab, dim) = self.embed_layout(op, &tv, ids)?;
        let mut expect = iv.shape().to_vec();
        expect.push(dim);
        if gv.shape() != expect.as_slice() {
            return Err(shape(
                op,
                format!("embedding grad shape {:?} != {expect:?}", gv.shape()),
            ));
        }
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
        let xv = self.f32_in(op, input)?;
        let wv = self.f32_in(op, weight)?;
        let (rows, kin, nout, dims) = self.linear_dims(op, xv.shape(), wv.shape())?;
        let (y, yb) = self.out(op, &dims)?;
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
        let xv = self.f32_in(op, input)?;
        let wv = self.f32_in(op, weight)?;
        let gv = self.f32_in(op, grad_output)?;
        let (rows, kin, nout, dims) = self.linear_dims(op, xv.shape(), wv.shape())?;
        if gv.shape() != dims.as_slice() {
            return Err(shape(
                op,
                format!("grad_output {:?} != output {dims:?}", gv.shape()),
            ));
        }
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
        let op = RMS_F;
        let xv = self.f32_in(op, input)?;
        let wv = self.f32_in(op, weight)?;
        let (rows, dim) = self.rms_layout(op, xv.shape(), wv.shape(), eps)?;
        let grid = self.groups(rows)?;
        let (y, yb) = self.out(op, xv.shape())?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let wb = bind(op, &mut job, &wv)?;
        job.dispatch(
            &RMS_FWD,
            &[u(op, rows)?, u(op, dim)?, eps.to_bits()],
            &[&xb, &wb, &yb],
            grid,
        )?;
        job.commit()?;
        Ok(y)
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let op = RMS_B;
        let xv = self.f32_in(op, input)?;
        let wv = self.f32_in(op, weight)?;
        let gv = self.f32_in(op, grad_output)?;
        let (rows, dim) = self.rms_layout(op, xv.shape(), wv.shape(), eps)?;
        same_shape(op, xv.shape(), gv.shape())?;
        let grid = self.groups(rows)?;
        self.fits(op, rows)?;
        let (gx, gxb) = self.out(op, xv.shape())?;
        let (gw, gwb) = self.out(op, wv.shape())?;
        let mut job = self.job(op);
        let xb = bind(op, &mut job, &xv)?;
        let wb = bind(op, &mut job, &wv)?;
        let gb = bind(op, &mut job, &gv)?;
        let rstd = job.scratch((rows as u64) * 4)?;
        job.dispatch(
            &RMS_BWD,
            &[u(op, rows)?, u(op, dim)?, eps.to_bits()],
            &[&xb, &wb, &gb, &gxb, &rstd],
            grid,
        )?;
        self.col_sum(op, &mut job, &gb, Some((&xb, &rstd)), rows, dim, &gwb)?;
        job.commit()?;
        Ok((gx, gw))
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.rope(ROPE_F, x, cos, sin, 0)
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.rope(ROPE_B, grad_output, cos, sin, 1)
    }

    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
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
        let (gq, gqw) = self.rms_norm_backward(q, q_weight, grad_q, eps)?;
        let (gk, gkw) = self.rms_norm_backward(k, k_weight, grad_k, eps)?;
        Ok((gq, gk, gqw, gkw))
    }

    fn causal_sdpa_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError> {
        let op = SDPA_F;
        let qv = self.f32_in(op, q)?;
        let kv = self.f32_in(op, k)?;
        let vv = self.f32_in(op, v)?;
        let plan = self.sdpa_dims(op, qv.shape(), kv.shape(), vv.shape())?;
        let bht = product(op, &[plan.bh, plan.time])?;
        let grid = self.attn_grid(op, plan.bh, plan.time, plan.fwd_rows)?;
        let (y, yb) = self.out(op, qv.shape())?;
        let mut job = self.job(op);
        let qb = bind(op, &mut job, &qv)?;
        let kb = bind(op, &mut job, &kv)?;
        let vb = bind(op, &mut job, &vv)?;
        job.dispatch(
            &attn_kernel(plan.module, "attn_fwd"),
            &[
                u(op, plan.time)?,
                plan.scale.to_bits(),
                u(op, bht)?,
                u(op, plan.dim)?,
            ],
            &[&qb, &kb, &vb, &yb],
            grid,
        )?;
        job.commit()?;
        Ok(y)
    }

    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        let op = SDPA_B;
        let qv = self.f32_in(op, q)?;
        let kv = self.f32_in(op, k)?;
        let vv = self.f32_in(op, v)?;
        let gv = self.f32_in(op, grad_output)?;
        if gv.shape() != qv.shape() {
            return Err(shape(
                op,
                format!("sdpa grad shape {:?} != query {:?}", gv.shape(), qv.shape()),
            ));
        }
        let plan = self.sdpa_dims(op, qv.shape(), kv.shape(), vv.shape())?;
        let bht = product(op, &[plan.bh, plan.time])?;
        let stats_len = product(op, &[bht, 2])?;
        self.fits(op, stats_len)?;
        let fwd_grid = self.attn_grid(op, plan.bh, plan.time, plan.fwd_rows)?;
        let bwd_grid = self.attn_grid(op, plan.bh, plan.time, plan.bwd_rows)?;
        let (gq, gqb) = self.out(op, qv.shape())?;
        let (gk, gkb) = self.out(op, kv.shape())?;
        let (gvv, gvb) = self.out(op, vv.shape())?;
        let mut job = self.job(op);
        let qb = bind(op, &mut job, &qv)?;
        let kb = bind(op, &mut job, &kv)?;
        let vb = bind(op, &mut job, &vv)?;
        let gb = bind(op, &mut job, &gv)?;
        let stats = job.scratch((stats_len as u64) * 4)?;
        let words = [
            u(op, plan.time)?,
            plan.scale.to_bits(),
            u(op, bht)?,
            u(op, plan.dim)?,
        ];
        job.dispatch(
            &attn_kernel(plan.module, "attn_bwd_prep"),
            &words,
            &[&qb, &kb, &vb, &gb, &stats],
            fwd_grid,
        )?;
        job.dispatch(
            &attn_kernel(plan.module, "attn_bwd_dq"),
            &words,
            &[&qb, &kb, &vb, &gb, &stats, &gqb],
            bwd_grid,
        )?;
        job.dispatch(
            &attn_kernel(plan.module, "attn_bwd_dkv"),
            &words,
            &[&qb, &kb, &vb, &gb, &stats, &gkb, &gvb],
            bwd_grid,
        )?;
        job.commit()?;
        Ok((gq, gk, gvv))
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let op = GATE_F;
        let xv = self.f32_in(op, input)?;
        let wv = self.f32_in(op, weight)?;
        let bv = self.f32_in(op, bias)?;
        let av = self.f32_in(op, attn_out)?;
        let (rows, din, heads, dh) =
            self.gate_layout(op, xv.shape(), wv.shape(), bv.shape(), av.shape())?;
        let zlen = product(op, &[rows, heads])?;
        self.fits(op, zlen)?;
        let grid = self.lanes(av.elems)?;
        let (y, yb) = self.out(op, av.shape())?;
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
        job.dispatch(
            &GATE_FWD,
            &[u(op, av.elems)?, u(op, heads)?, u(op, dh)?],
            &[&z, &bb, &ab, &yb],
            grid,
        )?;
        job.commit()?;
        Ok(y)
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        let op = GATE_B;
        let xv = self.f32_in(op, input)?;
        let wv = self.f32_in(op, weight)?;
        let bv = self.f32_in(op, bias)?;
        let av = self.f32_in(op, attn_out)?;
        let gv = self.f32_in(op, grad_output)?;
        let (rows, din, heads, dh) =
            self.gate_layout(op, xv.shape(), wv.shape(), bv.shape(), av.shape())?;
        same_shape(op, av.shape(), gv.shape())?;
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
        let z = job.scratch((zlen as u64) * 4)?;
        let gz = job.scratch((zlen as u64) * 4)?;
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
        job.dispatch(
            &GATE_BWD,
            &[u(op, zlen)?, u(op, heads)?, u(op, dh)?],
            &[&z, &bb, &ab, &gb, &gab, &gz],
            grid,
        )?;
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

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let op = VR_F;
        let v = self.f32_in(op, value)?;
        let v0 = self.f32_in(op, value0)?;
        let lam = self.f32_in(op, lambda)?;
        if lam.elems != 1 {
            return Err(shape(op, "expected a scalar tensor"));
        }
        same_shape(op, v.shape(), v0.shape())?;
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
        let v = self.f32_in(op, value)?;
        let v0 = self.f32_in(op, value0)?;
        let lam = self.f32_in(op, lambda)?;
        let gy = self.f32_in(op, grad_output)?;
        if lam.elems != 1 {
            return Err(shape(op, "value residual lambda must be a scalar"));
        }
        same_shape(op, v.shape(), v0.shape())?;
        same_shape(op, v.shape(), gy.shape())?;
        let n = v.elems;
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

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let op = SILU_F;
        let x = self.f32_in(op, input)?;
        let mut out = self.pointwise(op, &SILU_FWD, &[&x], 1, &[])?;
        out.pop().ok_or_else(|| shape(op, "no output"))
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        let op = SILU_B;
        let x = self.f32_in(op, input)?;
        let gy = self.f32_in(op, grad_output)?;
        same_shape(op, x.shape(), gy.shape())?;
        let mut out = self.pointwise(op, &SILU_BWD, &[&x, &gy], 1, &[])?;
        out.pop().ok_or_else(|| shape(op, "no output"))
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        let op = MUL_F;
        let av = self.f32_in(op, a)?;
        let bv = self.f32_in(op, b)?;
        same_shape(op, av.shape(), bv.shape())?;
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
        let av = self.f32_in(op, a)?;
        let bv = self.f32_in(op, b)?;
        let gy = self.f32_in(op, grad_output)?;
        same_shape(op, av.shape(), bv.shape())?;
        same_shape(op, av.shape(), gy.shape())?;
        let mut out = self.pointwise(op, &MUL_BWD, &[&av, &bv, &gy], 2, &[])?;
        let gb = out.pop().ok_or_else(|| shape(op, "no output"))?;
        let ga = out.pop().ok_or_else(|| shape(op, "no output"))?;
        Ok((ga, gb))
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        let op = ADD_F;
        let xv = self.f32_in(op, x)?;
        let yv = self.f32_in(op, y)?;
        same_shape(op, xv.shape(), yv.shape())?;
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
        let xv = self.f32_in(op, x)?;
        let yv = self.f32_in(op, y)?;
        let gy = self.f32_in(op, grad_output)?;
        same_shape(op, xv.shape(), yv.shape())?;
        same_shape(op, xv.shape(), gy.shape())?;
        let n = gy.elems;
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
        let lv = self.f32_in(op, logits)?;
        let tv = self.input(op, targets, DType::U32)?;
        let (rows, vocab, ignore, valid, any_ignored) =
            self.ce_layout(op, &lv, &tv, ignore_index)?;
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
        let lv = self.f32_in(op, logits)?;
        let tv = self.input(op, targets, DType::U32)?;
        let (rows, vocab, ignore, valid, any_ignored) =
            self.ce_layout(op, &lv, &tv, ignore_index)?;
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
    /// reported here, before anything is scaled.
    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        let op = CLIP;
        if grads.is_empty() {
            return Err(shape(op, "empty tensor"));
        }
        if !max_norm.is_finite() {
            return Err(OjasError::NonFinite { op: op.name() });
        }
        if max_norm < 0.0 {
            return Err(OjasError::OutOfRange {
                op: op.name(),
                detail: format!("max_norm {max_norm} is negative"),
            });
        }
        let mut sizes = Vec::with_capacity(grads.len());
        let mut total = 0usize;
        for g in grads.iter() {
            let n = self.f32_in(op, g)?.elems;
            self.lanes(n)?;
            sizes.push(n);
            total = total
                .checked_add(n.div_ceil(CHUNK))
                .ok_or_else(|| shape(op, "clip partials overflow"))?;
        }
        self.groups(total)?;
        self.fits(op, total)?;
        let status = {
            let mut job = self.job(op);
            let mut bound = Vec::with_capacity(grads.len());
            for g in grads.iter() {
                let gv = self.f32_in(op, g)?;
                bound.push(bind(op, &mut job, &gv)?);
            }
            let parts: Vec<(&wgpu::Buffer, usize)> =
                bound.iter().zip(sizes.iter().copied()).collect();
            let status = job.scratch(16)?;
            self.global_norm(op, &mut job, &parts, &status)?;
            job.commit()?;
            status
        };
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
        let gv = self.f32_in(op, grad)?;
        let n = {
            let pv = self.f32_in(op, param)?;
            let mv = self.f32_in(op, moment1)?;
            let vv = self.f32_in(op, moment2)?;
            same_shape(op, pv.shape(), gv.shape())?;
            same_shape(op, pv.shape(), mv.shape())?;
            same_shape(op, pv.shape(), vv.shape())?;
            pv.elems
        };
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
        require_ns5(5)?;
        let gv = self.f32_in(op, grad)?;
        let (rows, cols) = {
            let pv = self.f32_in(op, param)?;
            let mv = self.f32_in(op, momentum)?;
            if pv.shape().len() != 2 {
                return Err(shape(op, "muon parameter must be a matrix"));
            }
            same_shape(op, pv.shape(), gv.shape())?;
            same_shape(op, pv.shape(), mv.shape())?;
            (pv.shape()[0], pv.shape()[1])
        };
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
}
