//! macOS training step. Every GPU call goes through tessl, except the
//! per-head gate overlay compiled by this crate.

use std::sync::Arc;

use ojas_core::{
    next_step, refuse_unsupported_metal_head_dim, sdpa_scale, AdamWConfig, BackendId, OjasError,
    RMS_NORM_EPS,
};
use tessl::cross_entropy::{self, CeGrads, CeHidden, CeWorkspace, Reduction};
use tessl::dispatch::{self, set_gpu_buf, set_gpu_buf_offset, set_u32};
use tessl::gemm::GemmOperands;
use tessl::nn::{self, AttnDims};
use tessl::qwen35::{self, AttnShape, AttnTargets, Cols, OutCols};
use tessl::runtime::GpuRuntime;
use tessl::tensor::Tensor;
use tessl::DType;

pub const MAX_BATCH: u32 = 2;
pub const MAX_SEQ: u32 = 16;
pub const MAX_VOCAB: u32 = 128;
/// Two heads of [`HEAD_DIM`], packed `[T, H, D]`.
pub const D_MODEL: u32 = 128;
pub const N_HEAD: u32 = 2;
pub const HEAD_DIM: u32 = 64;
/// Nanolab `rope_base`.
pub const ROPE_THETA: f32 = 10_000.0;
/// Default micro-batch whose unchunked logit tape is refused, not allocated.
pub const LOGIT_TAPE_BATCH: u32 = 16;
pub const LOGIT_TAPE_SEQ: u32 = 1024;
pub const LOGIT_TAPE_VOCAB: u32 = 50_304;
/// Elements of padding in front of the `dh` view passed to cross-entropy.
pub const DH_PAD: usize = 16;
const CE_CHUNK: u32 = 16;

/// Absolute tolerances the tests use against local f32 formulas.
pub const STEP_ABS_TOL: f32 = 1e-3;
pub const GATE_ABS_TOL: f32 = 1e-5;
pub const ADAMW_ABS_TOL: f32 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TinyShape {
    pub batch: u32,
    pub seq: u32,
    pub d_model: u32,
    pub n_head: u32,
    pub head_dim: u32,
    pub vocab: u32,
}

impl TinyShape {
    pub fn rows(self) -> Result<u32, OjasError> {
        self.batch
            .checked_mul(self.seq)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "TinyShape::rows",
                detail: "batch * seq overflows u32".into(),
            })
    }
}

pub fn validate_tiny_shape(shape: &TinyShape) -> Result<(), OjasError> {
    refuse_unsupported_metal_head_dim(BackendId::Metal, shape.head_dim)?;
    if shape.head_dim != HEAD_DIM {
        return Err(OjasError::Shape {
            op: "validate_tiny_shape",
            detail: format!(
                "head_dim {} is not the lane B cap {HEAD_DIM}",
                shape.head_dim
            ),
        });
    }
    if shape.d_model != D_MODEL || shape.n_head != N_HEAD {
        return Err(OjasError::Shape {
            op: "validate_tiny_shape",
            detail: format!(
                "lane B step is d_model {D_MODEL}, n_head {N_HEAD}; got d_model {} n_head {}",
                shape.d_model, shape.n_head
            ),
        });
    }
    if shape.batch == 0 || shape.batch > MAX_BATCH {
        return Err(OjasError::Shape {
            op: "validate_tiny_shape",
            detail: format!("batch {} is outside 1..={MAX_BATCH}", shape.batch),
        });
    }
    if shape.seq == 0 || shape.seq > MAX_SEQ {
        return Err(OjasError::Shape {
            op: "validate_tiny_shape",
            detail: format!("seq {} is outside 1..={MAX_SEQ}", shape.seq),
        });
    }
    if shape.vocab == 0 || shape.vocab > MAX_VOCAB {
        return Err(OjasError::Shape {
            op: "validate_tiny_shape",
            detail: format!("vocab {} is outside 1..={MAX_VOCAB}", shape.vocab),
        });
    }
    let _ = shape.rows()?;
    Ok(())
}

fn ce_chunk(vocab: u32) -> u32 {
    CE_CHUNK.min(vocab).max(1)
}

/// Logical bytes `tiny_train_step` allocates. Not the Metal pool bucket.
pub fn scratch_bytes(shape: &TinyShape) -> Result<u64, OjasError> {
    validate_tiny_shape(shape)?;
    let rows = u64::from(shape.rows()?);
    let d = u64::from(shape.d_model);
    let vocab = u64::from(shape.vocab);
    // x, rms, gate, up, swiglu, down, residual, rotated Q, K, V, Q projection,
    // attention output, and the packed dQ, dK, dV.
    let acts = 15u64
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(d))
        .ok_or_else(|| overflow("scratch_bytes"))?;
    let dh = (DH_PAD as u64)
        .checked_add(
            rows.checked_mul(d)
                .ok_or_else(|| overflow("scratch_bytes"))?,
        )
        .ok_or_else(|| overflow("scratch_bytes"))?;
    let dw = vocab
        .checked_mul(d)
        .ok_or_else(|| overflow("scratch_bytes"))?;
    let dw_q = d.checked_mul(d).ok_or_else(|| overflow("scratch_bytes"))?;
    let seq = u64::from(shape.seq);
    let head = u64::from(shape.head_dim);
    // One head plane at a time: Q, K, V, dO, dQ, dK, dV, plus S, dP, P, dS.
    // `seq * seq` is allocated only after the tiny-step check (seq <= 16).
    let head_planes = 7u64
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(head))
        .ok_or_else(|| overflow("scratch_bytes"))?;
    let score_planes = 4u64
        .checked_mul(seq)
        .and_then(|n| n.checked_mul(seq))
        .ok_or_else(|| overflow("scratch_bytes"))?;
    let elems = acts
        .checked_add(dh)
        .and_then(|n| n.checked_add(dw))
        .and_then(|n| n.checked_add(dw_q))
        .and_then(|n| n.checked_add(head_planes))
        .and_then(|n| n.checked_add(score_planes))
        .ok_or_else(|| overflow("scratch_bytes"))?;
    let f32s = elems
        .checked_mul(4)
        .ok_or_else(|| overflow("scratch_bytes"))?;
    let scalars = 3u64 * 4;
    let ce = CeWorkspace::bytes_for(
        shape.rows()?,
        shape.d_model,
        ce_chunk(shape.vocab),
        DType::F32,
    ) as u64;
    f32s.checked_add(scalars)
        .and_then(|n| n.checked_add(ce))
        .ok_or_else(|| overflow("scratch_bytes"))
}

/// Parameter and Adam moments, in bytes. The Q projection and the LM head
/// each carry two moments.
pub fn param_bytes(shape: &TinyShape) -> Result<u64, OjasError> {
    validate_tiny_shape(shape)?;
    let d = u64::from(shape.d_model);
    let vocab = u64::from(shape.vocab);
    // rms weight, Q and K norm weights, three square MLP matrices,
    // Q projection plus two moments, LM head plus two moments.
    let qk = u64::from(shape.head_dim).checked_mul(2);
    let square = d.checked_mul(d).ok_or_else(|| overflow("param_bytes"))?;
    let elems = square
        .checked_mul(3)
        .and_then(|mlp| mlp.checked_add(d))
        .and_then(|n| n.checked_add(qk?))
        .and_then(|n| n.checked_add(square.checked_mul(3)?))
        .and_then(|n| n.checked_add(vocab.checked_mul(d)?.checked_mul(3)?))
        .ok_or_else(|| overflow("param_bytes"))?;
    elems.checked_mul(4).ok_or_else(|| overflow("param_bytes"))
}

/// Bytes of three unchunked `[batch * seq, vocab]` f32 logit tensors.
///
/// Nanolab's default micro-batch is batch 16, sequence 1024, vocab 50,304.
/// Chunked cross-entropy does not allocate this tape. [`refuse_unchunked_logit_tape`]
/// compares the estimate to a budget and returns before any allocation.
pub fn unchunked_logit_tape_bytes(batch: u32, seq: u32, vocab: u32) -> Result<u64, OjasError> {
    let rows = u64::from(batch)
        .checked_mul(u64::from(seq))
        .ok_or_else(|| overflow("unchunked_logit_tape_bytes"))?;
    let one = rows
        .checked_mul(u64::from(vocab))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| overflow("unchunked_logit_tape_bytes"))?;
    one.checked_mul(3)
        .ok_or_else(|| overflow("unchunked_logit_tape_bytes"))
}

/// Capacity error when the unchunked logit tape would exceed `byte_cap`.
/// Does not allocate.
pub fn refuse_unchunked_logit_tape(
    batch: u32,
    seq: u32,
    vocab: u32,
    byte_cap: u64,
) -> Result<(), OjasError> {
    let requested = unchunked_logit_tape_bytes(batch, seq, vocab)?;
    if requested > byte_cap {
        return Err(over_cap(requested, byte_cap));
    }
    Ok(())
}

fn overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "byte count overflows u64".into(),
    }
}

fn over_cap(requested: u64, cap: u64) -> OjasError {
    OjasError::CapacityExceeded {
        requested,
        cap,
        live: 0,
    }
}

fn metal(detail: impl Into<String>) -> OjasError {
    OjasError::Backend {
        id: BackendId::Metal,
        detail: detail.into(),
    }
}

fn finite_f32(op: &'static str, data: &[f32]) -> Result<(), OjasError> {
    if data.iter().any(|x| !x.is_finite()) {
        Err(OjasError::NonFinite { op })
    } else {
        Ok(())
    }
}

/// Open the default Metal device and load the gate overlay.
/// A missing device is an error.
pub struct Session {
    rt: Arc<GpuRuntime>,
}

impl Session {
    pub fn open() -> Result<Self, OjasError> {
        ojas_device::require_kind(ojas_device::Device::Metal, ojas_device::Device::Metal).map_err(
            |err| OjasError::Backend {
                id: BackendId::Metal,
                detail: err.to_string(),
            },
        )?;
        let rt = GpuRuntime::new().map_err(metal)?;
        rt.set_async_encode(true).map_err(metal)?;
        let gate = include_bytes!(concat!(env!("OUT_DIR"), "/ojas_per_head_gate.metallib"));
        if gate.is_empty() {
            return Err(metal("embedded gate metallib is empty"));
        }
        rt.add_metallib_bytes(gate).map_err(metal)?;
        for name in [
            "ojas_per_head_gate_fwd",
            "ojas_per_head_gate_bwd",
            "ojas_per_head_gate_dbias",
            "ojas_attn_pack_plane",
            "ojas_attn_unpack_plane",
            "ojas_causal_softmax_bwd",
        ] {
            rt.pipeline(name).map_err(metal)?;
        }
        Ok(Self { rt })
    }

    pub fn device_name(&self) -> String {
        self.rt.device_name()
    }
}

pub struct TinyState {
    rt: Arc<GpuRuntime>,
    shape: TinyShape,
    rms_w: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    w_lm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    w_q: Tensor,
    m_lm: Tensor,
    v_lm: Tensor,
    m_q: Tensor,
    v_q: Tensor,
    step: u64,
}

/// Host weights for [`TinyState::new`], row-major.
#[derive(Clone, Copy, Debug)]
pub struct TinyWeights<'a> {
    pub rms_w: &'a [f32],
    pub w_gate: &'a [f32],
    pub w_up: &'a [f32],
    pub w_down: &'a [f32],
    pub w_lm: &'a [f32],
    /// Learnable Q RMSNorm scale, length `head_dim`. Multiplied, not `1 + w`.
    pub q_norm: &'a [f32],
    /// Learnable K RMSNorm scale, length `head_dim`.
    pub k_norm: &'a [f32],
    /// Square Q projection, `[d_model, d_model]`, applied after RoPE.
    pub w_q: &'a [f32],
}

impl TinyState {
    pub fn new(
        session: &Session,
        shape: TinyShape,
        byte_cap: u64,
        weights: TinyWeights<'_>,
    ) -> Result<Self, OjasError> {
        let TinyWeights {
            rms_w,
            w_gate,
            w_up,
            w_down,
            w_lm,
            q_norm,
            k_norm,
            w_q,
        } = weights;
        let need = param_bytes(&shape)?;
        if need > byte_cap {
            return Err(over_cap(need, byte_cap));
        }
        let d = shape.d_model as usize;
        let mlp = d;
        let vocab = shape.vocab as usize;
        expect_len("rms_w", rms_w, d)?;
        expect_len("w_gate", w_gate, mlp * d)?;
        expect_len("w_up", w_up, mlp * d)?;
        expect_len("w_down", w_down, d * mlp)?;
        expect_len("w_lm", w_lm, vocab * d)?;
        let hd = shape.head_dim as usize;
        expect_len("q_norm", q_norm, hd)?;
        expect_len("k_norm", k_norm, hd)?;
        expect_len("w_q", w_q, d * d)?;
        for (name, data) in [
            ("rms_w", rms_w),
            ("w_gate", w_gate),
            ("w_up", w_up),
            ("w_down", w_down),
            ("w_lm", w_lm),
            ("q_norm", q_norm),
            ("k_norm", k_norm),
            ("w_q", w_q),
        ] {
            finite_f32(name, data)?;
        }
        let rt = Arc::clone(&session.rt);
        let rms_w_t = f32_tensor(&rt, &[d], rms_w)?;
        let w_gate_t = f32_tensor(&rt, &[mlp, d], w_gate)?;
        let w_up_t = f32_tensor(&rt, &[mlp, d], w_up)?;
        let w_down_t = f32_tensor(&rt, &[d, mlp], w_down)?;
        let w_lm_t = f32_tensor(&rt, &[vocab, d], w_lm)?;
        let q_norm_t = f32_tensor(&rt, &[hd], q_norm)?;
        let k_norm_t = f32_tensor(&rt, &[hd], k_norm)?;
        let w_q_t = f32_tensor(&rt, &[d, d], w_q)?;
        let m_lm = zeros(&rt, &[vocab, d])?;
        let v_lm = zeros(&rt, &[vocab, d])?;
        let m_q = zeros(&rt, &[d, d])?;
        let v_q = zeros(&rt, &[d, d])?;
        Ok(Self {
            rt,
            shape,
            rms_w: rms_w_t,
            w_gate: w_gate_t,
            w_up: w_up_t,
            w_down: w_down_t,
            w_lm: w_lm_t,
            q_norm: q_norm_t,
            k_norm: k_norm_t,
            w_q: w_q_t,
            m_lm,
            v_lm,
            m_q,
            v_q,
            step: 0,
        })
    }

    pub fn shape(&self) -> TinyShape {
        self.shape
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn lm_head(&self) -> Result<Vec<f32>, OjasError> {
        self.w_lm.read_f32().map_err(metal)
    }

    pub fn q_projection(&self) -> Result<Vec<f32>, OjasError> {
        self.w_q.read_f32().map_err(metal)
    }
}

fn expect_len(op: &'static str, data: &[f32], n: usize) -> Result<(), OjasError> {
    if data.len() == n {
        Ok(())
    } else {
        Err(OjasError::Shape {
            op,
            detail: format!("len {} , expected {n}", data.len()),
        })
    }
}

fn zeros(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Result<Tensor, OjasError> {
    rt.alloc_tensor_f32(shape).map_err(metal)
}

fn f32_tensor(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Result<Tensor, OjasError> {
    let t = zeros(rt, shape)?;
    t.write_f32(data).map_err(metal)?;
    Ok(t)
}

#[derive(Debug)]
pub struct TinyStep {
    pub loss: f64,
    /// AdamW step count after this call.
    pub adamw_step: u64,
    /// Causal SDPA at head dim 64 wrote the hidden state cross-entropy reads,
    /// after QK-norm, half-split RoPE, and the Q projection.
    pub attention_ran: bool,
}

/// One pre-norm SwiGLU block, QK-norm, half-split RoPE, a square Q projection,
/// causal attention at head dim 64, chunked cross-entropy, and one AdamW step
/// on the Q projection and the LM head.
///
/// Q and K are the residual after RMSNorm (`* weight`, eps `1e-6`) and
/// half-split RoPE. The Q projection is `q @ Wq^T` after that rotation. V is
/// the residual, not normalised. The attention output is what cross-entropy
/// sees. A failure from `nn::flash_attn_rows` or from the attention backward
/// is returned before any weight update. Weights move only after the loss and
/// both gradients are finite. Both AdamW updates use the same step count.
///
/// `qwen35_attn_bwd_*_h256` is FlashAttention-2 on a TensorOps tile whose N
/// dimension is head dim 256 (`matmul2d_descriptor` of `32 x 256`, threadgroup
/// `q32_k32_sg4`). Substituting 64 would be a different tile, so that kernel
/// is not instantiated. This step backpropagates with exact-f32 GEMM and a
/// causal softmax-gradient shader, for sequence at most 16 and head dim 64.
pub fn tiny_train_step(
    state: &mut TinyState,
    input: &[f32],
    targets: &[i32],
    byte_cap: u64,
    adam: AdamWConfig,
) -> Result<TinyStep, OjasError> {
    validate_tiny_shape(&state.shape)?;
    validate_adam(&adam)?;
    let rows_u = state.shape.rows()?;
    let rows = rows_u as usize;
    let d = state.shape.d_model as usize;
    expect_len("tiny_train_step input", input, rows * d)?;
    if targets.len() != rows {
        return Err(OjasError::Shape {
            op: "tiny_train_step",
            detail: format!("{} targets for {rows} rows", targets.len()),
        });
    }
    finite_f32("tiny_train_step", input)?;
    check_targets(state.shape.vocab, targets)?;
    let need = scratch_bytes(&state.shape)?;
    if need > byte_cap {
        return Err(over_cap(need, byte_cap));
    }

    let rt = Arc::clone(&state.rt);
    let x = f32_tensor(&rt, &[rows, d], input)?;
    let normed = zeros(&rt, &[rows, d])?;
    nn::rms_norm_f32(
        &rt,
        &x.buffer,
        &state.rms_w.buffer,
        &normed.buffer,
        rows_u,
        d as u32,
        RMS_NORM_EPS,
    )
    .map_err(metal)?;
    let mlp = d as u32;
    let gate = zeros(&rt, &[rows, d])?;
    let up = zeros(&rt, &[rows, d])?;
    linear_nt(&normed, &state.w_gate, &gate)?;
    linear_nt(&normed, &state.w_up, &up)?;
    let hidden = zeros(&rt, &[rows, d])?;
    qwen35::swiglu(
        &rt,
        Cols::dense(&gate.buffer, mlp),
        Cols::dense(&up.buffer, mlp),
        OutCols {
            cols: Cols::dense(&hidden.buffer, mlp),
            dtype: DType::F32,
        },
        rows_u,
        mlp,
    )
    .map_err(metal)?;
    let down = zeros(&rt, &[rows, d])?;
    linear_nt(&hidden, &state.w_down, &down)?;
    let resid = f32_tensor(&rt, &[rows, d], input)?;
    qwen35::residual_add(
        &rt,
        Cols::dense(&down.buffer, mlp),
        Cols::dense(&resid.buffer, mlp),
        rows_u,
        mlp,
    )
    .map_err(metal)?;

    let (q, k, v) = qk_norm_rope(&rt, &resid, state)?;
    let q_proj = zeros(&rt, &[rows, d])?;
    linear_nt(&q, &state.w_q, &q_proj)?;
    let attended = causal_attention(&rt, &q_proj, &k, &v, &state.shape)?;

    let ce = ce_at_offset(&rt, &attended, &state.w_lm, targets)?;
    let slab = ce.dh_slab.read_f32().map_err(metal)?;
    if slab[..DH_PAD].iter().any(|x| *x != 7.0) {
        return Err(metal(
            "cross-entropy wrote before the dh view (nonzero byte offset was ignored)",
        ));
    }
    if !ce.loss.is_finite() {
        return Err(OjasError::NonFinite {
            op: "tiny_train_step",
        });
    }
    let dh = ce.dh_slab.try_view(&[rows, d], DH_PAD).map_err(metal)?;
    let dq = zeros(&rt, &[rows, d])?;
    let dk = zeros(&rt, &[rows, d])?;
    let dv = zeros(&rt, &[rows, d])?;
    let attn = AttnBwdDims {
        batch: state.shape.batch,
        seq: state.shape.seq,
        heads: state.shape.n_head,
        head_dim: state.shape.head_dim,
    };
    causal_attn_backward(&rt, &attn, &q_proj, &k, &v, &dh, &dq, &dk, &dv)?;
    let dw_q = zeros(&rt, &[d, d])?;
    GemmOperands::ExactF32.tn(&dq, &q, &dw_q).map_err(metal)?;
    let dw_host = ce.dw.read_f32().map_err(metal)?;
    let dw_q_host = dw_q.read_f32().map_err(metal)?;
    // `ce.dw` stays a host read: tessl has `Qwen35Model::adamw_step` and no
    // generic `adamw_step(rt, param, grad, ...)` yet. This is the call site.
    finite_f32("tiny_train_step", &dw_host)?;
    finite_f32("tiny_train_step", &dw_q_host)?;
    let next = next_step(state.step)?;
    adamw_apply(
        &rt,
        &mut state.w_lm,
        &ce.dw,
        &mut state.m_lm,
        &mut state.v_lm,
        next,
        &adam,
    )?;
    adamw_apply(
        &rt,
        &mut state.w_q,
        &dw_q,
        &mut state.m_q,
        &mut state.v_q,
        next,
        &adam,
    )?;
    state.step = next;
    Ok(TinyStep {
        loss: ce.loss,
        adamw_step: state.step,
        attention_ran: true,
    })
}

fn check_targets(vocab: u32, targets: &[i32]) -> Result<(), OjasError> {
    for (i, &t) in targets.iter().enumerate() {
        if t < 0 {
            return Err(OjasError::OutOfRange {
                op: "tiny_train_step",
                detail: format!("token id {t} at row {i} is negative"),
            });
        }
        if t as u32 >= vocab {
            return Err(OjasError::OutOfRange {
                op: "tiny_train_step",
                detail: format!("token id {t} at row {i} is past vocab {vocab}"),
            });
        }
    }
    Ok(())
}

fn validate_adam(cfg: &AdamWConfig) -> Result<(), OjasError> {
    if !(cfg.lr.is_finite() && cfg.lr >= 0.0) {
        return Err(OjasError::OutOfRange {
            op: "adamw",
            detail: format!("lr {}", cfg.lr),
        });
    }
    for (name, b) in [("beta1", cfg.beta1), ("beta2", cfg.beta2)] {
        if !(0.0..1.0).contains(&b) {
            return Err(OjasError::OutOfRange {
                op: "adamw",
                detail: format!("{name} {b}"),
            });
        }
    }
    if !(cfg.eps.is_finite() && cfg.eps > 0.0) {
        return Err(OjasError::OutOfRange {
            op: "adamw",
            detail: format!("eps {}", cfg.eps),
        });
    }
    if !(cfg.weight_decay.is_finite() && cfg.weight_decay >= 0.0) {
        return Err(OjasError::OutOfRange {
            op: "adamw",
            detail: format!("weight_decay {}", cfg.weight_decay),
        });
    }
    Ok(())
}

fn linear_nt(x: &Tensor, w: &Tensor, y: &Tensor) -> Result<(), OjasError> {
    GemmOperands::ExactF32.nt(x, w, y).map_err(metal)
}

/// Causal SDPA over packed heads `[batch, seq, n_head, head_dim]`. Position
/// `t` attends to keys `<= t` inside its sequence. Scale is `1/sqrt(head_dim)`.
fn causal_attention(
    rt: &Arc<GpuRuntime>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    shape: &TinyShape,
) -> Result<Tensor, OjasError> {
    let rows = shape.rows()? as usize;
    let d = shape.d_model as usize;
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.shape() != [rows, d] {
            return Err(OjasError::Shape {
                op: "causal_attention",
                detail: format!("{name} shape {:?}, expected [{rows}, {d}]", t.shape()),
            });
        }
    }
    let o = zeros(rt, &[rows, d])?;
    let tkv = rt.alloc_buffer(4).map_err(metal)?;
    let qpos = rt.alloc_buffer(4).map_err(metal)?;
    let kvpos = rt.alloc_buffer(4).map_err(metal)?;
    tkv.write_u32(&[shape.seq]);
    qpos.write_u32(&[0]);
    kvpos.write_u32(&[0]);
    let dims = AttnDims {
        batch: shape.batch,
        tq: shape.seq,
        heads: shape.n_head,
        heads_kv: shape.n_head,
        window: 0,
        scale: sdpa_scale(shape.head_dim)?,
    };
    nn::flash_attn_rows(
        rt,
        &q.buffer,
        &k.buffer,
        &v.buffer,
        &o.buffer,
        &tkv,
        &qpos,
        &kvpos,
        dims,
        shape.head_dim,
        false,
    )
    .map_err(metal)?;
    Ok(o)
}

/// Sequence, batch, and head counts the tiny causal backward will run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttnBwdDims {
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub head_dim: u32,
}

/// Refuse a backward that is outside the tiny step before any `[seq, seq]`
/// allocation. Head dim above 64 is [`OjasError::UnsupportedHeadDim`]. Any
/// other miss (`seq > 16`, head dim other than 64, more than two heads) is
/// [`OjasError::Shape`].
fn attn_bwd_limits(dims: &AttnBwdDims) -> Result<(), OjasError> {
    refuse_unsupported_metal_head_dim(BackendId::Metal, dims.head_dim)?;
    if dims.head_dim != HEAD_DIM {
        return Err(OjasError::Shape {
            op: "causal_attn_backward",
            detail: format!("head_dim {} is not {HEAD_DIM}", dims.head_dim),
        });
    }
    if dims.seq == 0 || dims.seq > MAX_SEQ {
        return Err(OjasError::Shape {
            op: "causal_attn_backward",
            detail: format!("seq {} is outside 1..={MAX_SEQ}", dims.seq),
        });
    }
    if dims.heads == 0 || dims.heads > N_HEAD {
        return Err(OjasError::Shape {
            op: "causal_attn_backward",
            detail: format!("heads {} is outside 1..={N_HEAD}", dims.heads),
        });
    }
    if dims.batch == 0 || dims.batch > MAX_BATCH {
        return Err(OjasError::Shape {
            op: "causal_attn_backward",
            detail: format!("batch {} is outside 1..={MAX_BATCH}", dims.batch),
        });
    }
    Ok(())
}

/// Causal attention backward for one packed `[batch, seq, heads, 64]` plane.
///
/// `d_o` is the gradient of the attention output. `dq`, `dk`, and `dv` are
/// overwritten. A key `j > t` contributes nothing to query `t`, and query `t`
/// contributes nothing to that key. The products are `GemmOperands::ExactF32`;
/// `ojas_causal_softmax_bwd` applies the causal mask and the softmax gradient.
#[allow(clippy::too_many_arguments)]
fn causal_attn_backward(
    rt: &Arc<GpuRuntime>,
    dims: &AttnBwdDims,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    d_o: &Tensor,
    dq: &Tensor,
    dk: &Tensor,
    dv: &Tensor,
) -> Result<(), OjasError> {
    attn_bwd_limits(dims)?;
    let rows = dims.batch as usize * dims.seq as usize;
    let width = dims.heads as usize * dims.head_dim as usize;
    let plane = [rows, width];
    for (name, t) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("d_o", d_o),
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
    ] {
        if t.shape() != plane {
            return Err(OjasError::Shape {
                op: "causal_attn_backward",
                detail: format!("{name} shape {:?}, expected {plane:?}", t.shape()),
            });
        }
    }
    let seq = dims.seq as usize;
    let head_dim = dims.head_dim as usize;
    let scale = sdpa_scale(dims.head_dim)?;
    let q_h = zeros(rt, &[seq, head_dim])?;
    let k_h = zeros(rt, &[seq, head_dim])?;
    let v_h = zeros(rt, &[seq, head_dim])?;
    let do_h = zeros(rt, &[seq, head_dim])?;
    let dq_h = zeros(rt, &[seq, head_dim])?;
    let dk_h = zeros(rt, &[seq, head_dim])?;
    let dv_h = zeros(rt, &[seq, head_dim])?;
    let scores = zeros(rt, &[seq, seq])?;
    let d_p = zeros(rt, &[seq, seq])?;
    let probs = zeros(rt, &[seq, seq])?;
    let d_scores = zeros(rt, &[seq, seq])?;
    for b in 0..dims.batch {
        for h in 0..dims.heads {
            pack_plane(rt, q, &q_h, dims, b, h)?;
            pack_plane(rt, k, &k_h, dims, b, h)?;
            pack_plane(rt, v, &v_h, dims, b, h)?;
            pack_plane(rt, d_o, &do_h, dims, b, h)?;
            GemmOperands::ExactF32
                .nt(&q_h, &k_h, &scores)
                .map_err(metal)?;
            GemmOperands::ExactF32
                .nt(&do_h, &v_h, &d_p)
                .map_err(metal)?;
            softmax_bwd(rt, &scores, &d_p, &probs, &d_scores, dims.seq, scale)?;
            GemmOperands::ExactF32
                .nn(&d_scores, &k_h, &dq_h)
                .map_err(metal)?;
            GemmOperands::ExactF32
                .tn(&d_scores, &q_h, &dk_h)
                .map_err(metal)?;
            GemmOperands::ExactF32
                .tn(&probs, &do_h, &dv_h)
                .map_err(metal)?;
            unpack_plane(rt, &dq_h, dq, dims, b, h)?;
            unpack_plane(rt, &dk_h, dk, dims, b, h)?;
            unpack_plane(rt, &dv_h, dv, dims, b, h)?;
        }
    }
    Ok(())
}

fn elem_off(t: &Tensor) -> Result<u32, OjasError> {
    let bytes = t.byte_offset();
    if bytes % 4 != 0 {
        return Err(OjasError::Shape {
            op: "causal_attn_backward",
            detail: format!("byte offset {bytes} is not a multiple of 4"),
        });
    }
    u32::try_from(bytes / 4).map_err(|_| OjasError::OutOfRange {
        op: "causal_attn_backward",
        detail: format!("element offset {} does not fit u32", bytes / 4),
    })
}

fn pack_plane(
    rt: &Arc<GpuRuntime>,
    src: &Tensor,
    dst: &Tensor,
    dims: &AttnBwdDims,
    batch_index: u32,
    head_index: u32,
) -> Result<(), OjasError> {
    plane_copy(
        rt,
        "ojas_attn_pack_plane",
        src,
        dst,
        dims,
        batch_index,
        head_index,
    )
}

fn unpack_plane(
    rt: &Arc<GpuRuntime>,
    src: &Tensor,
    dst: &Tensor,
    dims: &AttnBwdDims,
    batch_index: u32,
    head_index: u32,
) -> Result<(), OjasError> {
    plane_copy(
        rt,
        "ojas_attn_unpack_plane",
        src,
        dst,
        dims,
        batch_index,
        head_index,
    )
}

fn plane_copy(
    rt: &Arc<GpuRuntime>,
    kernel: &str,
    src: &Tensor,
    dst: &Tensor,
    dims: &AttnBwdDims,
    batch_index: u32,
    head_index: u32,
) -> Result<(), OjasError> {
    let p = rt.pipeline(kernel).map_err(metal)?;
    let n = dims.seq as usize * dims.head_dim as usize;
    let off = if kernel == "ojas_attn_pack_plane" {
        elem_off(src)?
    } else {
        elem_off(dst)?
    };
    dispatch::dispatch_1d(rt, &p, n, |bnd| {
        set_gpu_buf(bnd, &src.buffer, 0);
        set_gpu_buf(bnd, &dst.buffer, 1);
        set_u32(bnd, dims.seq, 2);
        set_u32(bnd, dims.heads, 3);
        set_u32(bnd, dims.head_dim, 4);
        set_u32(bnd, batch_index, 5);
        set_u32(bnd, head_index, 6);
        set_u32(bnd, off, 7);
    })
    .map_err(metal)
}

fn softmax_bwd(
    rt: &Arc<GpuRuntime>,
    scores: &Tensor,
    d_p: &Tensor,
    probs: &Tensor,
    d_scores: &Tensor,
    seq: u32,
    scale: f32,
) -> Result<(), OjasError> {
    let p = rt.pipeline("ojas_causal_softmax_bwd").map_err(metal)?;
    dispatch::dispatch_1d(rt, &p, seq as usize, |bnd| {
        set_gpu_buf_offset(bnd, &scores.buffer, scores.byte_offset(), 0);
        set_gpu_buf_offset(bnd, &d_p.buffer, d_p.byte_offset(), 1);
        set_gpu_buf_offset(bnd, &probs.buffer, probs.byte_offset(), 2);
        set_gpu_buf_offset(bnd, &d_scores.buffer, d_scores.byte_offset(), 3);
        set_u32(bnd, seq, 4);
        dispatch::set_f32(bnd, scale, 5);
    })
    .map_err(metal)
}

/// Q and K: RMSNorm (`* weight`) and half-split RoPE at stride `head_dim`.
/// V is a copy of `resid`. Query head `j` is column `j * head_dim`, not
/// `j * 2 * head_dim`.
fn qk_norm_rope(
    rt: &Arc<GpuRuntime>,
    resid: &Tensor,
    state: &TinyState,
) -> Result<(Tensor, Tensor, Tensor), OjasError> {
    let shape = &state.shape;
    let rows = shape.rows()? as usize;
    let d = shape.d_model as usize;
    let q = zeros(rt, &[rows, d])?;
    let k = zeros(rt, &[rows, d])?;
    let v = zeros(rt, &[rows, d])?;
    let attn = AttnShape {
        batch: shape.batch,
        seq: shape.seq,
        q_heads: shape.n_head,
        kv_heads: shape.n_head,
        head_dim: shape.head_dim,
        rotary_dim: shape.head_dim,
    };
    qwen35::attn_qk_norm_rope_packed(
        rt,
        &attn,
        Cols::dense(&resid.buffer, d as u32),
        shape.head_dim,
        0.0,
        &state.q_norm.buffer,
        &state.k_norm.buffer,
        &AttnTargets {
            q_out: &q.buffer,
            k_cache: &k.buffer,
            v_cache: &v.buffer,
        },
        0,
        ROPE_THETA,
        RMS_NORM_EPS,
    )
    .map_err(metal)?;
    Ok((q, k, v))
}

struct CeAtOffset {
    loss: f64,
    dw: Tensor,
    dh_slab: Tensor,
}

fn ce_at_offset(
    rt: &Arc<GpuRuntime>,
    hidden: &Tensor,
    weight: &Tensor,
    targets_i: &[i32],
) -> Result<CeAtOffset, OjasError> {
    let rows = hidden.shape()[0];
    let hs = hidden.shape()[1];
    let vocab = weight.shape()[0];
    let targets: Vec<u32> = targets_i.iter().map(|&t| t as u32).collect();
    let row_ids: Vec<u32> = (0..rows as u32).collect();
    let chunk = ce_chunk(vocab as u32);
    let ws = CeWorkspace::new(rt, rows as u32, hs as u32, chunk, DType::F32).map_err(metal)?;
    let slab = zeros(rt, &[DH_PAD + rows * hs])?;
    let sentinel = vec![7.0f32; DH_PAD + rows * hs];
    slab.write_f32(&sentinel).map_err(metal)?;
    let dh = slab.try_view(&[rows, hs], DH_PAD).map_err(metal)?;
    let dw = zeros(rt, &[vocab, hs])?;
    let out = cross_entropy::cross_entropy_rows(
        rt,
        CeHidden {
            rows: hidden,
            off: 0,
        },
        weight,
        &row_ids,
        &targets,
        Reduction::Mean,
        GemmOperands::ExactF32,
        &ws,
        Some(CeGrads {
            dh: &dh,
            dw: &dw,
            scale: 1.0,
        }),
    )
    .map_err(|e| {
        if e.contains("not finite") {
            OjasError::NonFinite {
                op: "cross_entropy_rows",
            }
        } else {
            metal(e)
        }
    })?;
    Ok(CeAtOffset {
        loss: out.loss,
        dw,
        dh_slab: slab,
    })
}

#[cfg(test)]
fn adamw_tensor(
    rt: &Arc<GpuRuntime>,
    param: &mut Tensor,
    grad: &Tensor,
    moment1: &mut Tensor,
    moment2: &mut Tensor,
    step: &mut u64,
    cfg: &AdamWConfig,
) -> Result<(), OjasError> {
    let next = next_step(*step)?;
    adamw_apply(rt, param, grad, moment1, moment2, next, cfg)?;
    *step = next;
    Ok(())
}

fn adamw_apply(
    rt: &Arc<GpuRuntime>,
    param: &mut Tensor,
    grad: &Tensor,
    moment1: &mut Tensor,
    moment2: &mut Tensor,
    step_after: u64,
    cfg: &AdamWConfig,
) -> Result<(), OjasError> {
    let t = step_after as f64;
    let bc1 = 1.0 - cfg.beta1.powf(t);
    let bc2 = 1.0 - cfg.beta2.powf(t);
    if bc1 == 0.0 || bc2 < 0.0 {
        return Err(OjasError::OutOfRange {
            op: "adamw",
            detail: format!("bias correction bc1 {bc1} bc2 {bc2}"),
        });
    }
    let scalars = [
        (1.0 - cfg.lr * cfg.weight_decay) as f32,
        (1.0 - cfg.beta1) as f32,
        cfg.beta2 as f32,
        (1.0 - cfg.beta2) as f32,
        (cfg.lr / bc1) as f32,
        bc2.powf(0.5) as f32,
        cfg.eps as f32,
        1.0f32,
    ];
    if scalars.iter().any(|s| !s.is_finite()) {
        return Err(OjasError::NonFinite { op: "adamw" });
    }
    let &[rows, width] = param.shape() else {
        return Err(OjasError::Shape {
            op: "adamw",
            detail: format!("parameter shape {:?} is not rank 2", param.shape()),
        });
    };
    let (Ok(rows_u), Ok(width_u)) = (u32::try_from(rows), u32::try_from(width)) else {
        return Err(OjasError::OutOfRange {
            op: "adamw",
            detail: format!("[{rows}, {width}] does not fit u32 indexing"),
        });
    };
    if grad.shape() != param.shape()
        || moment1.shape() != param.shape()
        || moment2.shape() != param.shape()
    {
        return Err(OjasError::Shape {
            op: "adamw",
            detail: "moment or gradient shape does not match the parameter".into(),
        });
    }
    let bytes: Vec<u8> = scalars.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = rt.pipeline("qwen35_adamw_f32").map_err(metal)?;
    dispatch::dispatch_2d(rt, &p, width, rows, |bnd| {
        set_gpu_buf_offset(bnd, &param.buffer, param.byte_offset(), 0);
        set_gpu_buf_offset(bnd, &grad.buffer, grad.byte_offset(), 1);
        set_gpu_buf_offset(bnd, &moment1.buffer, moment1.byte_offset(), 2);
        set_gpu_buf_offset(bnd, &moment2.buffer, moment2.byte_offset(), 3);
        bnd.bind_bytes(&bytes, 4);
        set_u32(bnd, rows_u, 5);
        set_u32(bnd, width_u, 6);
        set_u32(bnd, width_u, 7);
        set_u32(bnd, 0, 8);
    })
    .map_err(metal)?;
    rt.commit(false).map_err(metal)?;
    Ok(())
}

pub struct GateGrad {
    pub d_input: Vec<f32>,
    pub d_weight: Vec<f32>,
    pub d_bias: Vec<f32>,
    pub d_attn: Vec<f32>,
}

/// Host inputs to the per-head gate, row-major. `x` is `[rows, d_model]`,
/// `weight` is `[n_head, d_model]`, `bias` is `[n_head]`, and `attn` is
/// `[rows, n_head, head_dim]`.
#[derive(Clone, Copy, Debug)]
pub struct GateInputs<'a> {
    pub x: &'a [f32],
    pub weight: &'a [f32],
    pub bias: &'a [f32],
    pub attn: &'a [f32],
    pub rows: u32,
}

/// `out = attn * sigmoid(x @ W^T + bias)`, per head.
/// `W` is `[n_head, d_model]`. The matmul is `GemmOperands::nt`.
pub fn per_head_gate_forward(
    session: &Session,
    inputs: GateInputs<'_>,
    byte_cap: u64,
) -> Result<Vec<f32>, OjasError> {
    let g = gate_prep(session, inputs, byte_cap, gate_fwd_bytes)?;
    let out = zeros(&g.rt, &[g.rows, g.n_head, g.head_dim])?;
    gate_fwd_kernel(&g, &out)?;
    out.read_f32().map_err(metal)
}

pub fn per_head_gate_backward(
    session: &Session,
    inputs: GateInputs<'_>,
    dy: &[f32],
    byte_cap: u64,
) -> Result<GateGrad, OjasError> {
    expect_len(
        "per_head_gate dy",
        dy,
        inputs.rows as usize * (N_HEAD * HEAD_DIM) as usize,
    )?;
    finite_f32("per_head_gate", dy)?;
    let g = gate_prep(session, inputs, byte_cap, gate_bwd_bytes)?;
    let dy_t = f32_tensor(&g.rt, &[g.rows, g.n_head, g.head_dim], dy)?;
    let d_attn = zeros(&g.rt, &[g.rows, g.n_head, g.head_dim])?;
    let d_pre = zeros(&g.rt, &[g.rows, g.n_head])?;
    let d_bias = zeros(&g.rt, &[g.n_head])?;
    let p = g.rt.pipeline("ojas_per_head_gate_bwd").map_err(metal)?;
    let units = g.rows * g.n_head;
    dispatch::dispatch_1d(&g.rt, &p, units, |bnd| {
        set_gpu_buf(bnd, &g.attn.buffer, 0);
        set_gpu_buf(bnd, &g.pre.buffer, 1);
        set_gpu_buf(bnd, &g.bias.buffer, 2);
        set_gpu_buf(bnd, &dy_t.buffer, 3);
        set_gpu_buf(bnd, &d_attn.buffer, 4);
        set_gpu_buf(bnd, &d_pre.buffer, 5);
        set_u32(bnd, g.rows as u32, 6);
        set_u32(bnd, g.n_head as u32, 7);
        set_u32(bnd, g.head_dim as u32, 8);
        let plane = (g.rows * g.n_head * g.head_dim) as u32;
        let pre_n = (g.rows * g.n_head) as u32;
        set_u32(bnd, plane, 9);
        set_u32(bnd, pre_n, 10);
        set_u32(bnd, g.n_head as u32, 11);
    })
    .map_err(metal)?;
    let pb = g.rt.pipeline("ojas_per_head_gate_dbias").map_err(metal)?;
    dispatch::dispatch_1d(&g.rt, &pb, g.n_head, |bnd| {
        set_gpu_buf(bnd, &d_pre.buffer, 0);
        set_gpu_buf(bnd, &d_bias.buffer, 1);
        set_u32(bnd, g.rows as u32, 2);
        set_u32(bnd, g.n_head as u32, 3);
        set_u32(bnd, (g.rows * g.n_head) as u32, 4);
        set_u32(bnd, g.n_head as u32, 5);
    })
    .map_err(metal)?;
    let d_weight = zeros(&g.rt, &[g.n_head, g.d_model])?;
    let d_input = zeros(&g.rt, &[g.rows, g.d_model])?;
    // dW = d_pre^T @ x, dX = d_pre @ W.
    GemmOperands::ExactF32
        .tn(&d_pre, &g.x, &d_weight)
        .map_err(metal)?;
    GemmOperands::ExactF32
        .nn(&d_pre, &g.weight, &d_input)
        .map_err(metal)?;
    Ok(GateGrad {
        d_input: d_input.read_f32().map_err(metal)?,
        d_weight: d_weight.read_f32().map_err(metal)?,
        d_bias: d_bias.read_f32().map_err(metal)?,
        d_attn: d_attn.read_f32().map_err(metal)?,
    })
}

struct GateBufs {
    rt: Arc<GpuRuntime>,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    attn: Tensor,
    pre: Tensor,
    rows: usize,
    n_head: usize,
    d_model: usize,
    head_dim: usize,
}

/// Element counts of the buffers both gate directions upload.
struct GateElems {
    x: usize,
    weight: usize,
    bias: usize,
    attn: usize,
    pre: usize,
}

fn gate_elems(
    op: &'static str,
    rows: usize,
    n_head: usize,
    d_model: usize,
    head_dim: usize,
) -> Result<GateElems, OjasError> {
    let x = rows.checked_mul(d_model).ok_or_else(|| overflow(op))?;
    let weight = n_head.checked_mul(d_model).ok_or_else(|| overflow(op))?;
    let pre = rows.checked_mul(n_head).ok_or_else(|| overflow(op))?;
    let attn = pre.checked_mul(head_dim).ok_or_else(|| overflow(op))?;
    Ok(GateElems {
        x,
        weight,
        bias: n_head,
        attn,
        pre,
    })
}

fn f32_bytes_of(op: &'static str, parts: &[usize]) -> Result<u64, OjasError> {
    parts
        .iter()
        .try_fold(0u64, |acc, &n| acc.checked_add(u64::try_from(n).ok()?))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| overflow(op))
}

/// Logical bytes `per_head_gate_forward` allocates.
pub fn gate_fwd_bytes(
    rows: usize,
    n_head: usize,
    d_model: usize,
    head_dim: usize,
) -> Result<u64, OjasError> {
    let e = gate_elems("gate_fwd_bytes", rows, n_head, d_model, head_dim)?;
    f32_bytes_of(
        "gate_fwd_bytes",
        &[e.x, e.weight, e.bias, e.attn, e.pre, e.attn],
    )
}

/// Logical bytes `per_head_gate_backward` allocates: the shared uploads plus
/// dy, d_attn, d_pre, d_bias, d_weight, and d_input.
pub fn gate_bwd_bytes(
    rows: usize,
    n_head: usize,
    d_model: usize,
    head_dim: usize,
) -> Result<u64, OjasError> {
    let e = gate_elems("gate_bwd_bytes", rows, n_head, d_model, head_dim)?;
    f32_bytes_of(
        "gate_bwd_bytes",
        &[
            e.x, e.weight, e.bias, e.attn, e.pre, e.attn, e.attn, e.pre, e.bias, e.weight, e.x,
        ],
    )
}

fn gate_prep(
    session: &Session,
    inputs: GateInputs<'_>,
    byte_cap: u64,
    bytes_for: fn(usize, usize, usize, usize) -> Result<u64, OjasError>,
) -> Result<GateBufs, OjasError> {
    let GateInputs {
        x,
        weight,
        bias,
        attn,
        rows: rows_u,
    } = inputs;
    if rows_u == 0 || rows_u > MAX_BATCH * MAX_SEQ {
        return Err(OjasError::Shape {
            op: "per_head_gate",
            detail: format!("rows {rows_u} outside 1..={}", MAX_BATCH * MAX_SEQ),
        });
    }
    refuse_unsupported_metal_head_dim(BackendId::Metal, HEAD_DIM)?;
    let rows = rows_u as usize;
    let n_head = N_HEAD as usize;
    let d_model = D_MODEL as usize;
    let head_dim = HEAD_DIM as usize;
    let need = bytes_for(rows, n_head, d_model, head_dim)?;
    if need > byte_cap {
        return Err(over_cap(need, byte_cap));
    }
    expect_len("per_head_gate x", x, rows * d_model)?;
    expect_len("per_head_gate weight", weight, n_head * d_model)?;
    expect_len("per_head_gate bias", bias, n_head)?;
    expect_len("per_head_gate attn", attn, rows * n_head * head_dim)?;
    finite_f32("per_head_gate", x)?;
    finite_f32("per_head_gate", weight)?;
    finite_f32("per_head_gate", bias)?;
    finite_f32("per_head_gate", attn)?;
    let rt = Arc::clone(&session.rt);
    let x_t = f32_tensor(&rt, &[rows, d_model], x)?;
    let w_t = f32_tensor(&rt, &[n_head, d_model], weight)?;
    let b_t = f32_tensor(&rt, &[n_head], bias)?;
    let attn_t = f32_tensor(&rt, &[rows, n_head, head_dim], attn)?;
    let pre = zeros(&rt, &[rows, n_head])?;
    GemmOperands::ExactF32.nt(&x_t, &w_t, &pre).map_err(metal)?;
    Ok(GateBufs {
        rt,
        x: x_t,
        weight: w_t,
        bias: b_t,
        attn: attn_t,
        pre,
        rows,
        n_head,
        d_model,
        head_dim,
    })
}

fn gate_fwd_kernel(g: &GateBufs, out: &Tensor) -> Result<(), OjasError> {
    let p = g.rt.pipeline("ojas_per_head_gate_fwd").map_err(metal)?;
    dispatch::dispatch_2d(&g.rt, &p, g.head_dim, g.rows * g.n_head, |bnd| {
        set_gpu_buf(bnd, &g.attn.buffer, 0);
        set_gpu_buf(bnd, &g.pre.buffer, 1);
        set_gpu_buf(bnd, &g.bias.buffer, 2);
        set_gpu_buf(bnd, &out.buffer, 3);
        set_u32(bnd, g.rows as u32, 4);
        set_u32(bnd, g.n_head as u32, 5);
        set_u32(bnd, g.head_dim as u32, 6);
        let plane = (g.rows * g.n_head * g.head_dim) as u32;
        set_u32(bnd, plane, 7);
        set_u32(bnd, (g.rows * g.n_head) as u32, 8);
        set_u32(bnd, g.n_head as u32, 9);
        set_u32(bnd, plane, 10);
    })
    .map_err(metal)?;
    g.rt.commit(false).map_err(metal)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        Session::open().expect("Metal device required; a missing device fails the test")
    }

    fn shape() -> TinyShape {
        TinyShape {
            batch: 1,
            seq: 4,
            d_model: D_MODEL,
            n_head: N_HEAD,
            head_dim: HEAD_DIM,
            vocab: 32,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tw<'a>(
        rms_w: &'a [f32],
        w_gate: &'a [f32],
        w_up: &'a [f32],
        w_down: &'a [f32],
        w_lm: &'a [f32],
        q_norm: &'a [f32],
        k_norm: &'a [f32],
        w_q: &'a [f32],
    ) -> TinyWeights<'a> {
        TinyWeights {
            rms_w,
            w_gate,
            w_up,
            w_down,
            w_lm,
            q_norm,
            k_norm,
            w_q,
        }
    }

    fn gi<'a>(
        x: &'a [f32],
        weight: &'a [f32],
        bias: &'a [f32],
        attn: &'a [f32],
        rows: usize,
    ) -> GateInputs<'a> {
        GateInputs {
            x,
            weight,
            bias,
            attn,
            rows: rows as u32,
        }
    }

    fn pattern(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(1103515245).wrapping_add(seed) % 1000;
                (x as f32) / 500.0 - 1.0
            })
            .collect()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn sigmoid(x: f32) -> f32 {
        let e = (-x.abs()).exp();
        let r = 1.0 / (1.0 + e);
        if x >= 0.0 {
            r
        } else {
            e * r
        }
    }

    fn gemm_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += a[i * k + t] * b[j * k + t];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn gemm_nn(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += a[i * k + t] * b[t * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn rms(x: &[f32], w: &[f32], rows: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let mut ss = 0.0f32;
            for v in row {
                ss += v * v;
            }
            let inv = (ss / dim as f32 + RMS_NORM_EPS).sqrt().recip();
            for c in 0..dim {
                out[r * dim + c] = row[c] * inv * w[c];
            }
        }
        out
    }

    fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(g, u)| g * sigmoid(*g) * u)
            .collect()
    }

    fn softmax_ce(
        hidden: &[f32],
        weight: &[f32],
        targets: &[i32],
        rows: usize,
        dim: usize,
        vocab: usize,
    ) -> (f64, Vec<f32>, Vec<f32>) {
        let logits = gemm_nt(hidden, weight, rows, vocab, dim);
        let mut loss = 0.0f64;
        let mut d_logits = vec![0.0f32; rows * vocab];
        let scale = 1.0f32 / rows as f32;
        for r in 0..rows {
            let row = &logits[r * vocab..(r + 1) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let mut exp = vec![0.0f32; vocab];
            for j in 0..vocab {
                exp[j] = (row[j] - max).exp();
                sum += exp[j];
            }
            let t = targets[r] as usize;
            loss += f64::from(max + sum.ln() - row[t]);
            for j in 0..vocab {
                let p = exp[j] / sum;
                let one = if j == t { 1.0 } else { 0.0 };
                d_logits[r * vocab + j] = (p - one) * scale;
            }
        }
        loss /= rows as f64;
        let mut h_t = vec![0.0f32; dim * rows];
        for r in 0..rows {
            for c in 0..dim {
                h_t[c * rows + r] = hidden[r * dim + c];
            }
        }
        let dw = gemm_nn(&transpose(&d_logits, rows, vocab), hidden, vocab, dim, rows);
        (loss, dw, d_logits)
    }

    fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut t = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = a[r * cols + c];
            }
        }
        t
    }

    fn adamw_cpu(
        p: &mut [f32],
        g: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        step_before: u64,
        cfg: &AdamWConfig,
    ) {
        let next = step_before + 1;
        let t = next as f64;
        let bc1 = 1.0 - cfg.beta1.powf(t);
        let bc2 = 1.0 - cfg.beta2.powf(t);
        let decay = (1.0 - cfg.lr * cfg.weight_decay) as f32;
        let lerp_w = (1.0 - cfg.beta1) as f32;
        let beta2 = cfg.beta2 as f32;
        let one_m = (1.0 - cfg.beta2) as f32;
        let step_size = (cfg.lr / bc1) as f32;
        let bc2_sqrt = bc2.powf(0.5) as f32;
        let eps = cfg.eps as f32;
        for i in 0..p.len() {
            let mut w = p[i] * decay;
            let gi = g[i];
            let mut mi = m[i];
            let diff = gi - mi;
            mi = if lerp_w < 0.5 {
                mi + lerp_w * diff
            } else {
                gi - diff * (1.0 - lerp_w)
            };
            let vi = v[i] * beta2 + (one_m * gi) * gi;
            let denom = vi.sqrt() / bc2_sqrt + eps;
            w += -step_size * (mi / denom);
            p[i] = w;
            m[i] = mi;
            v[i] = vi;
        }
    }

    #[test]
    fn metal_device_is_required() {
        let s = session();
        assert!(!s.device_name().is_empty(), "device name was empty");
    }

    #[test]
    fn head_dim_above_64_is_refused() {
        let _ = session();
        let mut s = shape();
        s.head_dim = 128;
        let err = validate_tiny_shape(&s).expect_err("head 128");
        match err {
            OjasError::UnsupportedHeadDim { head_dim, limit } => {
                assert_eq!(head_dim, 128);
                assert_eq!(limit, 64);
            }
            other => panic!("expected UnsupportedHeadDim, got {other}"),
        }
        s.head_dim = 65;
        assert!(matches!(
            validate_tiny_shape(&s),
            Err(OjasError::UnsupportedHeadDim {
                head_dim: 65,
                limit: 64
            })
        ));
    }

    #[test]
    fn b16_t1024_logit_tape_is_a_capacity_error_without_allocating() {
        let bytes = unchunked_logit_tape_bytes(LOGIT_TAPE_BATCH, LOGIT_TAPE_SEQ, LOGIT_TAPE_VOCAB)
            .expect("estimate");
        let rows = u64::from(LOGIT_TAPE_BATCH) * u64::from(LOGIT_TAPE_SEQ);
        let one = rows * u64::from(LOGIT_TAPE_VOCAB) * 4;
        assert_eq!(bytes, one * 3);
        assert!(bytes > 8 * (1 << 30), "the refused tape is {bytes} bytes");
        let budget = 4096u64;
        match refuse_unchunked_logit_tape(
            LOGIT_TAPE_BATCH,
            LOGIT_TAPE_SEQ,
            LOGIT_TAPE_VOCAB,
            budget,
        ) {
            Err(OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            }) => {
                assert_eq!(requested, bytes);
                assert_eq!(cap, budget);
                assert_eq!(live, 0);
            }
            other => panic!("expected CapacityExceeded, got {other:?}"),
        }
    }

    #[test]
    fn negative_token_does_not_change_weights() {
        let session = session();
        let shape = shape();
        let cap = param_bytes(&shape).unwrap();
        let d = D_MODEL as usize;
        let mut state = TinyState::new(
            &session,
            shape,
            cap,
            tw(
                &vec![1.0; d],
                &pattern(d * d, 2),
                &pattern(d * d, 3),
                &pattern(d * d, 4),
                &pattern(shape.vocab as usize * d, 5),
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; d * d],
            ),
        )
        .unwrap();
        let before = state.lm_head().unwrap();
        let before_q = state.q_projection().unwrap();
        let rows = (shape.batch * shape.seq) as usize;
        let mut targets = vec![1i32; rows];
        targets[0] = -1;
        let err = tiny_train_step(
            &mut state,
            &pattern(rows * d, 9),
            &targets,
            scratch_bytes(&shape).unwrap(),
            AdamWConfig::nanolab(1e-2, 0.0),
        )
        .expect_err("negative token");
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
        assert_eq!(state.step_count(), 0);
        assert_eq!(state.lm_head().unwrap(), before);
        assert_eq!(state.q_projection().unwrap(), before_q);
    }

    #[test]
    fn non_finite_input_does_not_write_weights() {
        let session = session();
        let shape = shape();
        let d = D_MODEL as usize;
        let mut state = TinyState::new(
            &session,
            shape,
            param_bytes(&shape).unwrap(),
            tw(
                &vec![1.0; d],
                &pattern(d * d, 2),
                &pattern(d * d, 3),
                &pattern(d * d, 4),
                &pattern(shape.vocab as usize * d, 5),
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; d * d],
            ),
        )
        .unwrap();
        let before = state.lm_head().unwrap();
        let before_q = state.q_projection().unwrap();
        let rows = (shape.batch * shape.seq) as usize;
        let mut input = pattern(rows * d, 9);
        input[3] = f32::NAN;
        let err = tiny_train_step(
            &mut state,
            &input,
            &vec![0i32; rows],
            scratch_bytes(&shape).unwrap(),
            AdamWConfig::nanolab(1e-2, 0.0),
        )
        .expect_err("nan");
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err}");
        assert_eq!(state.step_count(), 0);
        assert_eq!(state.lm_head().unwrap(), before);
        assert_eq!(state.q_projection().unwrap(), before_q);
    }

    #[test]
    fn scratch_cap_is_not_clamped() {
        let session = session();
        let shape = shape();
        let d = D_MODEL as usize;
        let mut state = TinyState::new(
            &session,
            shape,
            param_bytes(&shape).unwrap(),
            tw(
                &vec![1.0; d],
                &pattern(d * d, 2),
                &pattern(d * d, 3),
                &pattern(d * d, 4),
                &pattern(shape.vocab as usize * d, 5),
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; d * d],
            ),
        )
        .unwrap();
        let need = scratch_bytes(&shape).unwrap();
        let err = tiny_train_step(
            &mut state,
            &pattern((shape.batch * shape.seq) as usize * d, 1),
            &vec![0i32; (shape.batch * shape.seq) as usize],
            need - 1,
            AdamWConfig::nanolab(1e-2, 0.0),
        )
        .expect_err("cap");
        match err {
            OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            } => {
                assert_eq!(requested, need);
                assert_eq!(cap, need - 1);
                assert_eq!(live, 0);
            }
            other => panic!("expected CapacityExceeded, got {other}"),
        }
        assert_eq!(state.step_count(), 0);
    }

    #[test]
    fn ce_nonzero_dh_offset_keeps_prefix() {
        let session = session();
        let shape = shape();
        let rows = (shape.batch * shape.seq) as usize;
        let d = D_MODEL as usize;
        let vocab = shape.vocab as usize;
        let hidden = pattern(rows * d, 11);
        let weight = pattern(vocab * d, 12);
        let targets: Vec<i32> = (0..rows).map(|i| (i % vocab) as i32).collect();
        let h = f32_tensor(&session.rt, &[rows, d], &hidden).unwrap();
        let w = f32_tensor(&session.rt, &[vocab, d], &weight).unwrap();
        let ce = ce_at_offset(&session.rt, &h, &w, &targets).unwrap();
        let slab = ce.dh_slab.read_f32().unwrap();
        assert!(
            slab[..DH_PAD].iter().all(|&x| x == 7.0),
            "dh pad was overwritten"
        );
        let (loss, dw, _) = softmax_ce(&hidden, &weight, &targets, rows, d, vocab);
        assert!(
            (ce.loss - loss).abs() < f64::from(STEP_ABS_TOL),
            "loss {} vs {loss}",
            ce.loss
        );
        let got_dw = ce.dw.read_f32().unwrap();
        let err = max_abs(&got_dw, &dw);
        assert!(err < STEP_ABS_TOL, "dw err {err}");
        let _ = slab;
    }

    #[test]
    fn gate_forward_and_backward_match_cpu() {
        let session = session();
        let rows = 4usize;
        let d = D_MODEL as usize;
        let h = N_HEAD as usize;
        let hd = HEAD_DIM as usize;
        let x = pattern(rows * d, 21);
        let w = pattern(h * d, 22);
        let bias = pattern(h, 23);
        let attn = pattern(rows * h * hd, 24);
        let dy = pattern(rows * h * hd, 25);
        let bytes = gate_fwd_bytes(rows, h, d, hd).unwrap();
        let got = per_head_gate_forward(&session, gi(&x, &w, &bias, &attn, rows), bytes).unwrap();
        let pre = gemm_nt(&x, &w, rows, h, d);
        let mut expect = vec![0.0f32; rows * h * hd];
        for r in 0..rows {
            for head in 0..h {
                let g = sigmoid(pre[r * h + head] + bias[head]);
                for col in 0..hd {
                    let i = (r * h + head) * hd + col;
                    expect[i] = attn[i] * g;
                }
            }
        }
        let fwd_err = max_abs(&got, &expect);
        assert!(fwd_err < GATE_ABS_TOL, "gate forward err {fwd_err}");

        let grad = per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, bytes * 4)
            .unwrap();
        let mut d_pre = vec![0.0f32; rows * h];
        let mut d_attn = vec![0.0f32; rows * h * hd];
        let mut d_bias = vec![0.0f32; h];
        for r in 0..rows {
            for head in 0..h {
                let z = pre[r * h + head] + bias[head];
                let g = sigmoid(z);
                let mut acc = 0.0f32;
                for col in 0..hd {
                    let i = (r * h + head) * hd + col;
                    d_attn[i] = dy[i] * g;
                    acc += dy[i] * attn[i];
                }
                let dp = acc * g * (1.0 - g);
                d_pre[r * h + head] = dp;
                d_bias[head] += dp;
            }
        }
        let d_weight = gemm_nn(&transpose(&d_pre, rows, h), &x, h, d, rows);
        let d_input = gemm_nn(&d_pre, &w, rows, d, h);
        assert!(max_abs(&grad.d_attn, &d_attn) < GATE_ABS_TOL, "d_attn");
        assert!(max_abs(&grad.d_bias, &d_bias) < GATE_ABS_TOL, "d_bias");
        let w_err = max_abs(&grad.d_weight, &d_weight);
        let x_err = max_abs(&grad.d_input, &d_input);
        assert!(w_err < STEP_ABS_TOL, "d_weight err {w_err}");
        assert!(x_err < STEP_ABS_TOL, "d_input err {x_err}");
    }

    #[test]
    fn adamw_one_step_matches_torch_formula() {
        let session = session();
        let n = 64usize;
        let p0 = pattern(n, 31);
        let g = pattern(n, 32);
        let mut p = f32_tensor(&session.rt, &[1, n], &p0).unwrap();
        let grad = f32_tensor(&session.rt, &[1, n], &g).unwrap();
        let mut m = zeros(&session.rt, &[1, n]).unwrap();
        let mut v = zeros(&session.rt, &[1, n]).unwrap();
        let mut step = 0u64;
        let cfg = AdamWConfig::nanolab(1e-2, 0.01);
        adamw_tensor(&session.rt, &mut p, &grad, &mut m, &mut v, &mut step, &cfg).unwrap();
        assert_eq!(step, 1);
        let mut cpu_p = p0.clone();
        let mut cpu_m = vec![0.0f32; n];
        let mut cpu_v = vec![0.0f32; n];
        adamw_cpu(&mut cpu_p, &g, &mut cpu_m, &mut cpu_v, 0, &cfg);
        let err = max_abs(&p.read_f32().unwrap(), &cpu_p);
        assert!(err < ADAMW_ABS_TOL, "adamw err {err}");
    }

    #[test]
    fn adamw_rank_one_tensor_is_a_shape_error() {
        let session = session();
        let n = 63usize;
        let p0 = pattern(n, 31);
        let mut p = f32_tensor(&session.rt, &[n], &p0).unwrap();
        let grad = f32_tensor(&session.rt, &[n], &pattern(n, 32)).unwrap();
        let mut m = zeros(&session.rt, &[n]).unwrap();
        let mut v = zeros(&session.rt, &[n]).unwrap();
        let mut step = 0u64;
        let cfg = AdamWConfig::nanolab(1e-2, 0.01);
        let err = adamw_tensor(&session.rt, &mut p, &grad, &mut m, &mut v, &mut step, &cfg)
            .expect_err("rank 1");
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
        assert_eq!(step, 0);
        assert_eq!(p.read_f32().unwrap(), p0);
    }

    #[test]
    fn adamw_matches_cpu_at_odd_widths() {
        let session = session();
        let cfg = AdamWConfig::nanolab(1e-2, 0.01);
        for (rows, width) in [
            (1usize, 1usize),
            (1, 63),
            (3, 65),
            (1, 1023),
            (2, 1025),
            (7, 4099),
        ] {
            let n = rows * width;
            let p0 = pattern(n, 33);
            let g = pattern(n, 34);
            let mut p = f32_tensor(&session.rt, &[rows, width], &p0).unwrap();
            let grad = f32_tensor(&session.rt, &[rows, width], &g).unwrap();
            let mut m = zeros(&session.rt, &[rows, width]).unwrap();
            let mut v = zeros(&session.rt, &[rows, width]).unwrap();
            let mut step = 0u64;
            let mut cpu_p = p0.clone();
            let mut cpu_m = vec![0.0f32; n];
            let mut cpu_v = vec![0.0f32; n];
            for _ in 0..3 {
                adamw_cpu(&mut cpu_p, &g, &mut cpu_m, &mut cpu_v, step, &cfg);
                adamw_tensor(&session.rt, &mut p, &grad, &mut m, &mut v, &mut step, &cfg).unwrap();
            }
            let err = max_abs(&p.read_f32().unwrap(), &cpu_p);
            assert!(err < ADAMW_ABS_TOL, "[{rows}, {width}]: adamw err {err}");
        }
    }

    /// QK-norm and half-split RoPE. Packed `[B, T, H, D]`. V is the residual.
    fn rope_qk(
        resid: &[f32],
        q_w: &[f32],
        k_w: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let width = heads * head_dim;
        let mut q = vec![0.0f32; resid.len()];
        let mut k = vec![0.0f32; resid.len()];
        for row in 0..batch * seq {
            let t = row % seq;
            for h in 0..heads {
                let at = row * width + h * head_dim;
                let src = &resid[at..at + head_dim];
                q[at..at + head_dim].copy_from_slice(&half_split(src, q_w, t as u32));
                k[at..at + head_dim].copy_from_slice(&half_split(src, k_w, t as u32));
            }
        }
        (q, k)
    }

    /// Causal SDPA in f64. Position `t` attends to keys `0..=t` of the same head.
    fn causal_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0.0f32; q.len()];
        for b in 0..batch {
            for h in 0..heads {
                for t_q in 0..seq {
                    let q_base = ((b * seq + t_q) * heads + h) * head_dim;
                    let mut scores = vec![f64::NEG_INFINITY; seq];
                    for (t_k, score) in scores.iter_mut().enumerate().take(t_q + 1) {
                        let k_base = ((b * seq + t_k) * heads + h) * head_dim;
                        let mut dot = 0.0f64;
                        for c in 0..head_dim {
                            dot += f64::from(q[q_base + c]) * f64::from(k[k_base + c]);
                        }
                        *score = dot * scale;
                    }
                    let m = scores[..=t_q]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut l = 0.0f64;
                    let mut acc = vec![0.0f64; head_dim];
                    for (t_k, score) in scores.iter().enumerate().take(t_q + 1) {
                        let p = (*score - m).exp();
                        l += p;
                        let v_base = ((b * seq + t_k) * heads + h) * head_dim;
                        for c in 0..head_dim {
                            acc[c] += p * f64::from(v[v_base + c]);
                        }
                    }
                    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
                    for c in 0..head_dim {
                        out[q_base + c] = (acc[c] * inv) as f32;
                    }
                }
            }
        }
        out
    }

    /// f32 causal backward, same left-to-right order as `ojas_causal_softmax_bwd`.
    /// Keys `j > t` stay out of query `t`'s softmax and out of that query's `dK`.
    #[allow(clippy::too_many_arguments)]
    fn causal_bwd_f32(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_o: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n = q.len();
        let mut dq = vec![0.0f32; n];
        let mut dk = vec![0.0f32; n];
        let mut dv = vec![0.0f32; n];
        for b in 0..batch {
            for h in 0..heads {
                let at = |t: usize, c: usize| ((b * seq + t) * heads + h) * head_dim + c;
                let mut scores = vec![0.0f32; seq * seq];
                let mut dp = vec![0.0f32; seq * seq];
                for t in 0..seq {
                    for j in 0..seq {
                        let mut sq = 0.0f32;
                        let mut sd = 0.0f32;
                        for c in 0..head_dim {
                            sq += q[at(t, c)] * k[at(j, c)];
                            sd += d_o[at(t, c)] * v[at(j, c)];
                        }
                        scores[t * seq + j] = sq;
                        dp[t * seq + j] = sd;
                    }
                }
                let mut probs = vec![0.0f32; seq * seq];
                let mut ds = vec![0.0f32; seq * seq];
                for t in 0..seq {
                    let mut m = f32::NEG_INFINITY;
                    for j in 0..=t {
                        m = m.max(scores[t * seq + j] * scale);
                    }
                    let mut sum = 0.0f32;
                    for j in 0..=t {
                        let e = (scores[t * seq + j] * scale - m).exp();
                        probs[t * seq + j] = e;
                        sum += e;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    for j in 0..=t {
                        probs[t * seq + j] *= inv;
                    }
                    let mut dot = 0.0f32;
                    for j in 0..=t {
                        dot += probs[t * seq + j] * dp[t * seq + j];
                    }
                    for j in 0..=t {
                        let p = probs[t * seq + j];
                        ds[t * seq + j] = p * (dp[t * seq + j] - dot) * scale;
                    }
                }
                for t in 0..seq {
                    for c in 0..head_dim {
                        let mut gq = 0.0f32;
                        let mut gk = 0.0f32;
                        let mut gv = 0.0f32;
                        for j in 0..seq {
                            gq += ds[t * seq + j] * k[at(j, c)];
                            gk += ds[j * seq + t] * q[at(j, c)];
                            gv += probs[j * seq + t] * d_o[at(j, c)];
                        }
                        dq[at(t, c)] = gq;
                        dk[at(t, c)] = gk;
                        dv[at(t, c)] = gv;
                    }
                }
            }
        }
        (dq, dk, dv)
    }

    #[allow(clippy::too_many_arguments)]
    fn gpu_attn_bwd(
        session: &Session,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_o: &[f32],
        batch: u32,
        seq: u32,
        heads: u32,
        head_dim: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let rows = batch as usize * seq as usize;
        let width = heads as usize * head_dim as usize;
        let q_t = f32_tensor(&session.rt, &[rows, width], q).unwrap();
        let k_t = f32_tensor(&session.rt, &[rows, width], k).unwrap();
        let v_t = f32_tensor(&session.rt, &[rows, width], v).unwrap();
        let do_t = f32_tensor(&session.rt, &[rows, width], d_o).unwrap();
        let dq = zeros(&session.rt, &[rows, width]).unwrap();
        let dk = zeros(&session.rt, &[rows, width]).unwrap();
        let dv = zeros(&session.rt, &[rows, width]).unwrap();
        causal_attn_backward(
            &session.rt,
            &AttnBwdDims {
                batch,
                seq,
                heads,
                head_dim,
            },
            &q_t,
            &k_t,
            &v_t,
            &do_t,
            &dq,
            &dk,
            &dv,
        )
        .unwrap();
        (
            dq.read_f32().unwrap(),
            dk.read_f32().unwrap(),
            dv.read_f32().unwrap(),
        )
    }

    /// `* weight` RMSNorm, then `x * cos + cat(-x2, x1) * sin` on the full head.
    fn half_split(x: &[f32], w: &[f32], pos: u32) -> Vec<f32> {
        let dim = x.len();
        let mut ss = 0.0f32;
        for v in x {
            ss += v * v;
        }
        let inv = (ss / dim as f32 + RMS_NORM_EPS).sqrt().recip();
        let mut n = vec![0.0f32; dim];
        for i in 0..dim {
            n[i] = x[i] * inv * w[i];
        }
        let half = dim / 2;
        let mut y = vec![0.0f32; dim];
        for p in 0..half {
            let inv_freq = 1.0 / ROPE_THETA.powf((2 * p) as f32 / dim as f32);
            let angle = pos as f32 * inv_freq;
            let (s, c) = angle.sin_cos();
            let x1 = n[p];
            let x2 = n[p + half];
            y[p] = x1 * c - x2 * s;
            y[p + half] = x2 * c + x1 * s;
        }
        y
    }

    #[test]
    fn causal_backward_matches_cpu_at_t4_one_head() {
        let session = session();
        let (seq, heads, head_dim) = (4usize, 1usize, 64usize);
        let n = seq * heads * head_dim;
        let draw = |seed: u32| -> Vec<f32> { pattern(n, seed).iter().map(|x| x * 0.05).collect() };
        let q = draw(3);
        let k = draw(5);
        let v = draw(7);
        let d_o = draw(9);
        let (dq, dk, dv) = gpu_attn_bwd(&session, &q, &k, &v, &d_o, 1, 4, 1, 64);
        let (want_q, want_k, want_v) = causal_bwd_f32(&q, &k, &v, &d_o, 1, seq, heads, head_dim);
        let q_err = max_abs(&dq, &want_q);
        let k_err = max_abs(&dk, &want_k);
        let v_err = max_abs(&dv, &want_v);
        assert!(q_err < STEP_ABS_TOL, "dQ err {q_err}");
        assert!(k_err < STEP_ABS_TOL, "dK err {k_err}");
        assert!(v_err < STEP_ABS_TOL, "dV err {v_err}");
    }

    #[test]
    fn future_key_does_not_receive_position_zero_gradient() {
        let session = session();
        let head_dim = 64usize;
        let seq = 4usize;
        let n = seq * head_dim;
        let q = pattern(n, 11).iter().map(|x| x * 0.05).collect::<Vec<_>>();
        let v = pattern(n, 13).iter().map(|x| x * 0.05).collect::<Vec<_>>();
        let mut k_far = pattern(n, 17).iter().map(|x| x * 0.05).collect::<Vec<_>>();
        let mut k_quiet = k_far.clone();
        let future = 3 * head_dim;
        for c in 0..head_dim {
            k_far[future + c] = 1.0e6;
            k_quiet[future + c] = 0.0;
        }
        let mut d_o = vec![0.0f32; n];
        for slot in d_o.iter_mut().take(head_dim) {
            *slot = 0.25;
        }
        let shape = TinyShape {
            batch: 1,
            seq: 4,
            d_model: 64,
            n_head: 1,
            head_dim: 64,
            vocab: 1,
        };
        let fwd = |k: &[f32]| {
            let q_t = f32_tensor(&session.rt, &[seq, head_dim], &q).unwrap();
            let k_t = f32_tensor(&session.rt, &[seq, head_dim], k).unwrap();
            let v_t = f32_tensor(&session.rt, &[seq, head_dim], &v).unwrap();
            causal_attention(&session.rt, &q_t, &k_t, &v_t, &shape)
                .unwrap()
                .read_f32()
                .unwrap()
        };
        let o_far = fwd(&k_far);
        let o_quiet = fwd(&k_quiet);
        let row0 = max_abs(&o_far[..head_dim], &o_quiet[..head_dim]);
        assert!(
            row0 < STEP_ABS_TOL,
            "a key at position 3 changed position 0's forward output by {row0}"
        );
        let (dq_far, dk_far, _) = gpu_attn_bwd(&session, &q, &k_far, &v, &d_o, 1, 4, 1, 64);
        let (dq_quiet, dk_quiet, _) = gpu_attn_bwd(&session, &q, &k_quiet, &v, &d_o, 1, 4, 1, 64);
        let leak = dk_far[future..future + head_dim]
            .iter()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            leak < STEP_ABS_TOL,
            "position 0 produced a gradient of {leak} on a future key of 1e6"
        );
        assert!(dk_far[future..future + head_dim]
            .iter()
            .all(|x| x.is_finite()));
        let moved = max_abs(
            &dk_far[future..future + head_dim],
            &dk_quiet[future..future + head_dim],
        );
        assert!(
            moved < STEP_ABS_TOL,
            "the 1e6 future key changed its own dK from position 0 by {moved}"
        );
        let dq0 = max_abs(&dq_far[..head_dim], &dq_quiet[..head_dim]);
        assert!(
            dq0 < STEP_ABS_TOL,
            "the future key changed dQ at position 0 by {dq0}"
        );
    }

    #[test]
    fn causal_backward_refuses_large_seq_without_a_score_matrix() {
        let _ = session();
        for seq in [17u32, 1024, u32::MAX] {
            let err = attn_bwd_limits(&AttnBwdDims {
                batch: 1,
                seq,
                heads: 1,
                head_dim: 64,
            })
            .expect_err("large seq");
            assert!(
                matches!(
                    err,
                    OjasError::Shape {
                        op: "causal_attn_backward",
                        ..
                    }
                ),
                "seq {seq}: {err}"
            );
        }
        let err = attn_bwd_limits(&AttnBwdDims {
            batch: 1,
            seq: 4,
            heads: 1,
            head_dim: 32,
        })
        .expect_err("head 32");
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
        let err = attn_bwd_limits(&AttnBwdDims {
            batch: 1,
            seq: 4,
            heads: 1,
            head_dim: 128,
        })
        .expect_err("head 128");
        match err {
            OjasError::UnsupportedHeadDim { head_dim, limit } => {
                assert_eq!(head_dim, 128);
                assert_eq!(limit, 64);
            }
            other => panic!("expected UnsupportedHeadDim, got {other}"),
        }
        let session = session();
        let n = 17 * 64;
        let z = vec![0.0f32; n];
        let q = f32_tensor(&session.rt, &[17, 64], &z).unwrap();
        let err = causal_attn_backward(
            &session.rt,
            &AttnBwdDims {
                batch: 1,
                seq: 17,
                heads: 1,
                head_dim: 64,
            },
            &q,
            &q,
            &q,
            &q,
            &q,
            &q,
            &q,
        )
        .expect_err("seq 17");
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
    }

    #[test]
    fn train_step_matches_cpu_with_causal_attention() {
        let moved = assert_step_matches_cpu(&session(), shape(), 40);
        assert!(moved > 0.0, "q projection did not move");
    }

    #[test]
    fn train_step_matches_cpu_across_shapes() {
        let session = session();
        // (batch, seq, vocab, seed): single row, vocab 1, a vocab tail past the
        // 16-wide CE chunk, both batch rows, and the caps.
        for (batch, seq, vocab, seed) in [
            (1, 1, 1, 100),
            (1, 1, 2, 110),
            (1, 5, 17, 120),
            (2, 3, 31, 130),
            (2, 7, 33, 140),
            (1, 16, 64, 150),
            (2, 9, 100, 160),
            (2, 16, 128, 170),
            (2, 16, 127, 180),
        ] {
            let shape = TinyShape {
                batch,
                seq,
                vocab,
                ..shape()
            };
            assert_step_matches_cpu(&session, shape, seed);
        }
    }

    #[test]
    fn tiny_state_refuses_shapes_outside_the_envelope() {
        let session = session();
        let d = D_MODEL as usize;
        for head_dim in [65u32, 128] {
            let shape = TinyShape {
                head_dim,
                ..shape()
            };
            let err = TinyState::new(
                &session,
                shape,
                u64::MAX,
                tw(
                    &vec![1.0; d],
                    &pattern(d * d, 2),
                    &pattern(d * d, 3),
                    &pattern(d * d, 4),
                    &pattern(shape.vocab as usize * d, 5),
                    &vec![1.0; HEAD_DIM as usize],
                    &vec![1.0; HEAD_DIM as usize],
                    &vec![1.0; d * d],
                ),
            )
            .err()
            .expect("head_dim above 64 must be refused");
            assert!(
                matches!(err, OjasError::UnsupportedHeadDim { head_dim: h, limit: 64 } if h == head_dim),
                "{err}"
            );
            assert!(scratch_bytes(&shape).is_err());
        }
        for bad in [
            TinyShape {
                batch: 0,
                ..shape()
            },
            TinyShape {
                batch: 3,
                ..shape()
            },
            TinyShape { seq: 0, ..shape() },
            TinyShape { seq: 17, ..shape() },
            TinyShape {
                vocab: 0,
                ..shape()
            },
            TinyShape {
                vocab: 129,
                ..shape()
            },
            TinyShape {
                d_model: 63,
                ..shape()
            },
            TinyShape {
                d_model: 64,
                ..shape()
            },
            TinyShape {
                n_head: 1,
                ..shape()
            },
            TinyShape {
                n_head: 3,
                ..shape()
            },
            TinyShape {
                head_dim: 32,
                ..shape()
            },
        ] {
            assert!(
                matches!(validate_tiny_shape(&bad), Err(OjasError::Shape { .. })),
                "{bad:?} was accepted"
            );
        }
    }

    #[test]
    fn train_steps_are_deterministic_and_finite() {
        let session = session();
        let shape = TinyShape {
            batch: 2,
            seq: 7,
            vocab: 33,
            ..shape()
        };
        let rows = shape.rows().unwrap() as usize;
        let d = D_MODEL as usize;
        let vocab = shape.vocab as usize;
        let new_state = || {
            TinyState::new(
                &session,
                shape,
                param_bytes(&shape).unwrap(),
                tw(
                    &pattern(d, 51),
                    &pattern(d * d, 52),
                    &pattern(d * d, 53),
                    &pattern(d * d, 54),
                    &pattern(vocab * d, 55),
                    &pattern(HEAD_DIM as usize, 61),
                    &pattern(HEAD_DIM as usize, 62),
                    &pattern(d * d, 63),
                ),
            )
            .unwrap()
        };
        let input = pattern(rows * d, 56);
        let targets: Vec<i32> = (0..rows).map(|i| (i * 5 % vocab) as i32).collect();
        let cap = scratch_bytes(&shape).unwrap();
        let cfg = AdamWConfig::nanolab(1e-3, 0.01);
        let mut a = new_state();
        let mut b = new_state();
        for i in 0..1000u64 {
            let la = tiny_train_step(&mut a, &input, &targets, cap, cfg).unwrap();
            let lb = tiny_train_step(&mut b, &input, &targets, cap, cfg).unwrap();
            assert!(la.loss.is_finite(), "step {i}: loss {}", la.loss);
            assert_eq!(la.loss.to_bits(), lb.loss.to_bits(), "step {i}");
            assert_eq!(la.adamw_step, i + 1);
        }
        let wa = a.lm_head().unwrap();
        assert!(wa.iter().all(|x| x.is_finite()));
        let wb = b.lm_head().unwrap();
        assert!(wa.iter().zip(&wb).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    fn gate_inputs(rows: usize, seed: u32) -> [Vec<f32>; 5] {
        let d = D_MODEL as usize;
        let h = N_HEAD as usize;
        let hd = HEAD_DIM as usize;
        [
            pattern(rows * d, seed),
            pattern(h * d, seed + 1),
            pattern(h, seed + 2),
            pattern(rows * h * hd, seed + 3),
            pattern(rows * h * hd, seed + 4),
        ]
    }

    #[test]
    fn gate_matches_cpu_across_rows() {
        let session = session();
        let d = D_MODEL as usize;
        let h = N_HEAD as usize;
        let hd = HEAD_DIM as usize;
        for (rows, seed) in [(1usize, 60u32), (3, 70), (7, 80), (31, 90), (32, 100)] {
            let [x, w, bias, attn, dy] = gate_inputs(rows, seed);
            let got = per_head_gate_forward(
                &session,
                gi(&x, &w, &bias, &attn, rows),
                gate_fwd_bytes(rows, h, d, hd).unwrap(),
            )
            .unwrap();
            let pre = gemm_nt(&x, &w, rows, h, d);
            let mut expect = vec![0.0f32; rows * h * hd];
            let mut d_attn = vec![0.0f32; rows * h * hd];
            let mut d_pre = vec![0.0f32; rows * h];
            let mut d_bias = vec![0.0f32; h];
            for r in 0..rows {
                for head in 0..h {
                    let g = sigmoid(pre[r * h + head] + bias[head]);
                    let mut acc = 0.0f32;
                    for col in 0..hd {
                        let i = (r * h + head) * hd + col;
                        expect[i] = attn[i] * g;
                        d_attn[i] = dy[i] * g;
                        acc += dy[i] * attn[i];
                    }
                    let dp = acc * g * (1.0 - g);
                    d_pre[r * h + head] = dp;
                    d_bias[head] += dp;
                }
            }
            let fwd_err = max_abs(&got, &expect);
            assert!(fwd_err < GATE_ABS_TOL, "rows {rows}: forward err {fwd_err}");
            let grad = per_head_gate_backward(
                &session,
                gi(&x, &w, &bias, &attn, rows),
                &dy,
                gate_bwd_bytes(rows, h, d, hd).unwrap(),
            )
            .unwrap();
            let d_weight = gemm_nn(&transpose(&d_pre, rows, h), &x, h, d, rows);
            let d_input = gemm_nn(&d_pre, &w, rows, d, h);
            assert!(
                max_abs(&grad.d_attn, &d_attn) < GATE_ABS_TOL,
                "rows {rows}: d_attn"
            );
            assert!(
                max_abs(&grad.d_bias, &d_bias) < STEP_ABS_TOL,
                "rows {rows}: d_bias"
            );
            assert!(
                max_abs(&grad.d_weight, &d_weight) < STEP_ABS_TOL,
                "rows {rows}: d_weight"
            );
            assert!(
                max_abs(&grad.d_input, &d_input) < STEP_ABS_TOL,
                "rows {rows}: d_input"
            );
        }
    }

    #[test]
    fn gate_refuses_rows_outside_the_envelope() {
        let session = session();
        for rows in [0usize, 33, 1025] {
            let [x, w, bias, attn, dy] = gate_inputs(rows, 7);
            let fwd = per_head_gate_forward(&session, gi(&x, &w, &bias, &attn, rows), u64::MAX);
            assert!(matches!(fwd, Err(OjasError::Shape { .. })), "rows {rows}");
            let bwd =
                per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, u64::MAX);
            assert!(bwd.is_err(), "rows {rows}");
        }
    }

    #[test]
    fn gate_backward_cap_counts_backward_buffers() {
        let session = session();
        let rows = 4usize;
        let (h, d, hd) = (N_HEAD as usize, D_MODEL as usize, HEAD_DIM as usize);
        let [x, w, bias, attn, dy] = gate_inputs(rows, 3);
        let need = gate_bwd_bytes(rows, h, d, hd).unwrap();
        assert!(need > gate_fwd_bytes(rows, h, d, hd).unwrap());
        let err = per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, need - 1)
            .err()
            .expect("backward under its own byte count must be refused");
        match err {
            OjasError::CapacityExceeded { requested, cap, .. } => {
                assert_eq!(requested, need);
                assert_eq!(cap, need - 1);
            }
            other => panic!("expected CapacityExceeded, got {other}"),
        }
        per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, need).unwrap();
        let fwd_need = gate_fwd_bytes(rows, h, d, hd).unwrap();
        assert!(matches!(
            per_head_gate_forward(&session, gi(&x, &w, &bias, &attn, rows), fwd_need - 1),
            Err(OjasError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn gate_backward_checks_dy_before_dispatch() {
        let session = session();
        let rows = 2usize;
        let [x, w, bias, attn, mut dy] = gate_inputs(rows, 9);
        dy[0] = f32::INFINITY;
        let err = per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, u64::MAX)
            .err()
            .expect("inf dy");
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err}");
        dy.pop();
        let err = per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, u64::MAX)
            .err()
            .expect("short dy");
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
    }

    #[test]
    fn gate_forward_is_deterministic_over_1000_dispatches() {
        let session = session();
        let rows = 31usize;
        let [x, w, bias, attn, dy] = gate_inputs(rows, 41);
        let cap = u64::MAX;
        let first = per_head_gate_forward(&session, gi(&x, &w, &bias, &attn, rows), cap).unwrap();
        let first_bwd =
            per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, cap).unwrap();
        assert!(first.iter().all(|v| v.is_finite()));
        for i in 0..1000 {
            let got = per_head_gate_forward(&session, gi(&x, &w, &bias, &attn, rows), cap).unwrap();
            assert!(
                got.iter()
                    .zip(&first)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "forward dispatch {i} differs"
            );
            if i % 10 == 0 {
                let g = per_head_gate_backward(&session, gi(&x, &w, &bias, &attn, rows), &dy, cap)
                    .unwrap();
                assert_eq!(g.all_bits(), first_bwd.all_bits(), "backward dispatch {i}");
            }
        }
    }

    impl GateGrad {
        fn all_bits(&self) -> Vec<u32> {
            self.d_input
                .iter()
                .chain(&self.d_weight)
                .chain(&self.d_bias)
                .chain(&self.d_attn)
                .map(|v| v.to_bits())
                .collect()
        }
    }

    fn assert_step_matches_cpu(session: &Session, shape: TinyShape, seed: u32) -> f32 {
        let rows = (shape.batch * shape.seq) as usize;
        let d = D_MODEL as usize;
        let vocab = shape.vocab as usize;
        let rms_w = pattern(d, seed + 1);
        // The raw draw makes logits of several thousand. f32 flash attention
        // and the f64 reference then differ by about 1.2e-3 on the loss, past
        // the absolute bound. Scaling the draw keeps that bound at 1e-3.
        let scale = 0.25f32;
        let damp = |src: &[f32]| src.iter().map(|x| x * scale).collect::<Vec<_>>();
        let w_gate = damp(&pattern(d * d, seed + 2));
        let w_up = damp(&pattern(d * d, seed + 3));
        let w_down = damp(&pattern(d * d, seed + 4));
        let w_lm = damp(&pattern(vocab * d, seed + 5));
        let q_norm = damp(&pattern(HEAD_DIM as usize, seed + 7));
        let k_norm = damp(&pattern(HEAD_DIM as usize, seed + 8));
        let w_q = damp(&pattern(d * d, seed + 9));
        let input = damp(&pattern(rows * d, seed + 6));
        let targets: Vec<i32> = (0..rows).map(|i| (i * 3 % vocab) as i32).collect();
        let mut state = TinyState::new(
            session,
            shape,
            param_bytes(&shape).unwrap(),
            tw(
                &rms_w, &w_gate, &w_up, &w_down, &w_lm, &q_norm, &k_norm, &w_q,
            ),
        )
        .unwrap();
        let cfg = AdamWConfig::nanolab(1e-2, 0.0);
        let step = tiny_train_step(
            &mut state,
            &input,
            &targets,
            scratch_bytes(&shape).unwrap(),
            cfg,
        )
        .unwrap();
        assert_eq!(step.adamw_step, 1);
        assert!(step.attention_ran, "causal attention did not run");

        let normed = rms(&input, &rms_w, rows, d);
        let gate = gemm_nt(&normed, &w_gate, rows, d, d);
        let up = gemm_nt(&normed, &w_up, rows, d, d);
        let hidden = swiglu(&gate, &up);
        let down = gemm_nt(&hidden, &w_down, rows, d, d);
        let mut resid = input.clone();
        for (y, r) in down.iter().zip(resid.iter_mut()) {
            *r += y;
        }
        let (q_rope, k) = rope_qk(
            &resid,
            &q_norm,
            &k_norm,
            shape.batch as usize,
            shape.seq as usize,
            shape.n_head as usize,
            shape.head_dim as usize,
        );
        let q_proj = gemm_nt(&q_rope, &w_q, rows, d, d);
        let attended = causal_sdpa(
            &q_proj,
            &k,
            &resid,
            shape.batch as usize,
            shape.seq as usize,
            shape.n_head as usize,
            shape.head_dim as usize,
        );
        let (loss, dw, d_logits) = softmax_ce(&attended, &w_lm, &targets, rows, d, vocab);
        assert!(
            (step.loss - loss).abs() < f64::from(STEP_ABS_TOL),
            "{shape:?}: loss {} vs {loss}",
            step.loss
        );
        let dh = gemm_nn(&d_logits, &w_lm, rows, d, vocab);
        let (dq, _, _) = causal_bwd_f32(
            &q_proj,
            &k,
            &resid,
            &dh,
            shape.batch as usize,
            shape.seq as usize,
            shape.n_head as usize,
            shape.head_dim as usize,
        );
        let dw_q = gemm_nn(&transpose(&dq, rows, d), &q_rope, d, d, rows);
        let mut cpu_w = w_lm.clone();
        let mut m = vec![0.0f32; cpu_w.len()];
        let mut v = vec![0.0f32; cpu_w.len()];
        adamw_cpu(&mut cpu_w, &dw, &mut m, &mut v, 0, &cfg);
        let err = max_abs(&state.lm_head().unwrap(), &cpu_w);
        assert!(err < STEP_ABS_TOL, "{shape:?}: lm head err {err}");
        let mut cpu_q = w_q.clone();
        let mut mq = vec![0.0f32; cpu_q.len()];
        let mut vq = vec![0.0f32; cpu_q.len()];
        adamw_cpu(&mut cpu_q, &dw_q, &mut mq, &mut vq, 0, &cfg);
        let got_q = state.q_projection().unwrap();
        let mut stable = 0.0f32;
        let mut near_zero = 0.0f32;
        for ((gpu, cpu), grad) in got_q.iter().zip(&cpu_q).zip(&dw_q) {
            let delta = (gpu - cpu).abs();
            if grad.abs() >= 1.0e-4 {
                stable = stable.max(delta);
            } else {
                near_zero = near_zero.max(delta);
            }
        }
        assert!(
            stable < STEP_ABS_TOL,
            "{shape:?}: q projection err {stable}"
        );
        // A gradient around 1e-8 is next to AdamW's eps, so the first step
        // moves the weight by a fraction of the learning rate 1e-2.
        assert!(
            near_zero < 2.0e-3,
            "{shape:?}: q projection err {near_zero} where the gradient is near eps"
        );
        max_abs(&got_q, &w_q)
    }

    #[test]
    fn step_count_at_u64_max_is_refused() {
        let session = session();
        let shape = shape();
        let d = D_MODEL as usize;
        let mut state = TinyState::new(
            &session,
            shape,
            param_bytes(&shape).unwrap(),
            tw(
                &vec![1.0; d],
                &pattern(d * d, 2),
                &pattern(d * d, 3),
                &pattern(d * d, 4),
                &pattern(shape.vocab as usize * d, 5),
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; HEAD_DIM as usize],
                &vec![1.0; d * d],
            ),
        )
        .unwrap();
        state.step = u64::MAX;
        let before = state.lm_head().unwrap();
        let before_q = state.q_projection().unwrap();
        let rows = (shape.batch * shape.seq) as usize;
        let err = tiny_train_step(
            &mut state,
            &pattern(rows * d, 8),
            &vec![1i32; rows],
            scratch_bytes(&shape).unwrap(),
            AdamWConfig::nanolab(1e-3, 0.0),
        )
        .expect_err("step overflow");
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
        assert_eq!(state.lm_head().unwrap(), before);
        assert_eq!(state.q_projection().unwrap(), before_q);
    }
}
