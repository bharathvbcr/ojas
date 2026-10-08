//! Shape and dtype contract of the [`crate::Backend`] training ops.
//!
//! Each `*_dims` function is pure: it reads tensor metadata (dtype and
//! shape), never element values, placement, strides or a [`crate::Budget`],
//! and allocates at most a small output-shape `Vec`. A backend calls it
//! before it reserves budget or dispatches, so a malformed call is refused
//! with the same variant, op name and detail on every backend whatever the
//! budget holds.
//!
//! Every validator checks in this order, and stops at the first refusal:
//!
//! 1. Each operand in argument order: the dtype ([`OjasError::Dtype`]),
//!    then no zero extent (`"empty tensor"`), then that its element and
//!    byte counts fit `usize`.
//! 2. Scalar arguments that the backends check inside their layout code
//!    (`eps` for RMSNorm, [`OjasError::NonFinite`]).
//! 3. Ranks and cross-operand equalities, in the order the op documents.
//! 4. Derived sizes (an output's elements and bytes) fit `usize`.
//!
//! Every refusal of a shape is [`OjasError::Shape`]. An element or byte count
//! that overflows `usize` is [`OjasError::OutOfRange`], the variant every
//! backend and [`crate::shape_product`] already use. The `op` is the name the CPU backend
//! already reports; the composite `rms_qk_norm_*` ops report the
//! `rms_norm_*` name they are built from.
//!
//! Not checked here, and still each backend's: placement, contiguity
//! (non-contiguous metadata is legal; see [`Tensor`]), NaN and infinity,
//! token-id and target ranges, optimizer and clip scalars
//! ([`crate::check_adamw`], [`crate::clip_scale`]), and device limits such as
//! [`crate::METAL_MAX_HEAD_DIM`], 32-bit index caps and workgroup limits.

use crate::{CeChunk, DType, GdnInputs, OjasError, Tensor, GDN_CHECKPOINT_TOKENS};

const EMBEDDING_FORWARD: &str = "embedding_forward";
const EMBEDDING_BACKWARD: &str = "embedding_backward";
const LINEAR_FORWARD: &str = "linear_forward";
const LINEAR_BACKWARD: &str = "linear_backward";
const RMS_NORM_FORWARD: &str = "rms_norm_forward";
const RMS_NORM_BACKWARD: &str = "rms_norm_backward";
const ROPE_FORWARD: &str = "rope_half_split_forward";
const ROPE_BACKWARD: &str = "rope_half_split_backward";
const SDPA_FORWARD: &str = "causal_sdpa_forward";
const SDPA_BACKWARD: &str = "causal_sdpa_backward";
const GATE_FORWARD: &str = "per_head_sigmoid_gate_forward";
const GATE_BACKWARD: &str = "per_head_sigmoid_gate_backward";
const VALUE_RESIDUAL_FORWARD: &str = "value_residual_blend_forward";
const VALUE_RESIDUAL_BACKWARD: &str = "value_residual_blend_backward";
const GDN_FORWARD: &str = "chunked_gdn_forward";
const GDN_BACKWARD: &str = "chunked_gdn_backward";
const CONV1D_FORWARD: &str = "causal_conv1d_silu_forward";
const CONV1D_BACKWARD: &str = "causal_conv1d_silu_backward";
const GATED_RMS_FORWARD: &str = "gated_rms_norm_forward";
const GATED_RMS_BACKWARD: &str = "gated_rms_norm_backward";
const ROPE_PARTIAL_FORWARD: &str = "rope_partial_forward";
const ROPE_PARTIAL_BACKWARD: &str = "rope_partial_backward";
const SILU_FORWARD: &str = "silu_forward";
const SILU_BACKWARD: &str = "silu_backward";
const SIGMOID_FORWARD: &str = "sigmoid_forward";
const SIGMOID_BACKWARD: &str = "sigmoid_backward";
const GDN_DECAY_FORWARD: &str = "gdn_log_decay_forward";
const GDN_DECAY_BACKWARD: &str = "gdn_log_decay_backward";
const MUL_FORWARD: &str = "mul_forward";
const MUL_BACKWARD: &str = "mul_backward";
const ADD_FORWARD: &str = "residual_add_forward";
const ADD_BACKWARD: &str = "residual_add_backward";
const ACCUMULATE_GRAD: &str = "accumulate_grad";
const PERMUTE: &str = "permute";
const CE_FORWARD: &str = "cross_entropy_mean_forward";
const CE_BACKWARD: &str = "cross_entropy_mean_backward";
const CLIP: &str = "clip_grad_norm";
const ADAMW: &str = "adamw_step";
const MUON: &str = "muon_ns5_step";

/// Dimensions of an embedding lookup or its gradient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingDims {
    pub vocab: usize,
    pub dim: usize,
    /// Number of token ids (the product of the id tensor's shape).
    pub tokens: usize,
    /// Shape of the result: `ids.shape ++ [dim]` for the forward, the table's
    /// `[vocab, dim]` for the backward.
    pub out_shape: Vec<usize>,
}

/// Dimensions of `y = x @ W^T` with `x` `[..., in]` and `W` `[out, in]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearDims {
    /// Product of `x`'s leading axes.
    pub rows: usize,
    pub in_features: usize,
    pub out_features: usize,
    /// Forward output shape `x.shape[..-1] ++ [out]`. In the backward it is
    /// the shape `grad_output` was checked against; the gradients are shaped
    /// like `input` and `weight`.
    pub out_shape: Vec<usize>,
}

/// Dimensions of an RMSNorm over the last axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsDims {
    /// Product of the leading axes.
    pub rows: usize,
    pub dim: usize,
}

/// How the RoPE tables map onto the rotated tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    /// `cos` and `sin` have the rotated tensor's shape.
    Same,
    /// The rotated tensor is `[B, T, H, D]` and the tables are `[T, D]`.
    TimeDim { time: usize, heads: usize },
}

/// Dimensions of a half-split RoPE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeDims {
    /// Product of the rotated tensor's leading axes.
    pub rows: usize,
    /// Last axis, even.
    pub dim: usize,
    pub layout: RopeLayout,
}

/// Dimensions of causal attention. Query is `[B, H, T, D]`; K and V are
/// `[B, Hkv, T, D]`. `H` is a positive multiple of `Hkv` (multi-head when
/// `Hkv == H`, including both zero). The product of each tensor, in `f32`
/// bytes, fits `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdpaDims {
    pub batch: usize,
    /// Query heads.
    pub heads: usize,
    /// Key and value heads. Query head `h` reads KV head `h / (heads / kv_heads)`.
    pub kv_heads: usize,
    pub seq: usize,
    pub head_dim: usize,
}

/// Dimensions of the gated delta rule ([`crate::GdnInputs`]): `q`, `k`
/// `[B, T, H, Dk]`, `v` `[B, T, H, Dv]`, `g`, `beta` `[B, T, H]`, states
/// `[B, H, Dk, Dv]`, checkpoints `[B, H, checkpoints, Dk, Dv]`. Every one of
/// those tensors' `f32` byte counts fits `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnDims {
    pub batch: usize,
    pub seq: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// `ceil(seq / GDN_CHECKPOINT_TOKENS)`.
    pub checkpoints: usize,
}

impl GdnDims {
    /// `B * T * H`: the `(b, t, h)` rows of every per-token operand.
    pub fn rows(&self) -> usize {
        self.batch * self.seq * self.heads
    }

    /// `[B, H, Dk, Dv]`.
    pub fn state_shape(&self) -> [usize; 4] {
        [self.batch, self.heads, self.key_dim, self.value_dim]
    }

    /// `[B, H, checkpoints, Dk, Dv]`.
    pub fn checkpoint_shape(&self) -> [usize; 5] {
        [
            self.batch,
            self.heads,
            self.checkpoints,
            self.key_dim,
            self.value_dim,
        ]
    }
}

/// Dimensions of the per-head sigmoid gate: `input` `[..., d_model]`,
/// `weight` `[heads, d_model]`, `bias` `[heads]`, `attn_out`
/// `[..., heads, head_dim]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateDims {
    /// Product of `input`'s leading axes.
    pub rows: usize,
    pub d_model: usize,
    pub heads: usize,
    pub head_dim: usize,
}

/// Dimensions of a depthwise causal conv: input `[batch, time, channels]`,
/// weight `[channels, width]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dDims {
    pub batch: usize,
    pub time: usize,
    pub channels: usize,
    pub width: usize,
}

/// Dimensions of a partial RoPE over `[batch, time, heads, dim]` with
/// `[time, rotary]` tables: the leading `rotary` of each head's `dim` values
/// rotate, the rest pass through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialRopeDims {
    /// `batch * time * heads`.
    pub rows: usize,
    pub time: usize,
    pub heads: usize,
    pub dim: usize,
    pub rotary: usize,
}

/// Dimensions of the gated delta rule's log decay: `a` `[..., heads]`, and
/// `a_log`, `dt_bias` `[heads]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnDecayDims {
    /// Product of `a`'s leading axes.
    pub rows: usize,
    pub heads: usize,
}

/// Dimensions of mean cross-entropy over `logits` `[..., vocab]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeDims {
    /// Product of the logits' leading axes; also the number of targets.
    pub rows: usize,
    pub vocab: usize,
}

/// Dimensions of a Muon matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MuonDims {
    pub rows: usize,
    pub cols: usize,
}

fn refuse(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op,
        detail: detail.into(),
    }
}

/// A count that does not fit `usize`. This is [`OjasError::OutOfRange`], as
/// every backend and [`crate::shape_product`] already report it, so adopting
/// these validators does not change that variant.
fn too_large(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: detail.into(),
    }
}

/// Elements of `dims` whose `elem_size`-byte values also fit in `usize`
/// bytes. Overflow of either is [`OjasError::OutOfRange`], never a wrapped count.
fn sized_product(op: &'static str, dims: &[usize], elem_size: usize) -> Result<usize, OjasError> {
    let mut n = 1usize;
    for &d in dims {
        n = n
            .checked_mul(d)
            .ok_or_else(|| too_large(op, format!("shape product of {dims:?} overflows")))?;
    }
    n.checked_mul(elem_size)
        .ok_or_else(|| too_large(op, format!("byte length of {dims:?} overflows")))?;
    Ok(n)
}

/// Product of `dims` as a count of `f32` values.
fn f32_product(op: &'static str, dims: &[usize]) -> Result<usize, OjasError> {
    sized_product(op, dims, DType::F32.size())
}

/// Per-operand rule: the dtype, then no zero extent, then an element and
/// byte count that fit. Returns the element count.
fn operand(op: &'static str, tensor: &Tensor, dtype: DType) -> Result<usize, OjasError> {
    if tensor.dtype() != dtype {
        return Err(OjasError::Dtype {
            op,
            expected: dtype,
            got: tensor.dtype(),
        });
    }
    if tensor.shape().contains(&0) {
        return Err(refuse(op, "empty tensor"));
    }
    sized_product(op, tensor.shape(), dtype.size())
}

fn f32_operand(op: &'static str, tensor: &Tensor) -> Result<usize, OjasError> {
    operand(op, tensor, DType::F32)
}

/// A matmul-class operand ([`crate::Backend::bf16_operands`]): `F32`, or
/// `Bf16` storage that the kernel widens exactly into its `f32`
/// accumulation. Any other dtype is refused naming `F32`, the default.
fn matmul_operand(op: &'static str, tensor: &Tensor) -> Result<usize, OjasError> {
    let dtype = match tensor.dtype() {
        DType::Bf16 => DType::Bf16,
        _ => DType::F32,
    };
    operand(op, tensor, dtype)
}

/// [`matmul_operand`] over every tensor, in order.
fn matmul_operands(op: &'static str, tensors: &[&Tensor]) -> Result<(), OjasError> {
    for t in tensors {
        matmul_operand(op, t)?;
    }
    Ok(())
}

/// [`f32_operand`] over every tensor, in order.
fn f32_operands(op: &'static str, tensors: &[&Tensor]) -> Result<(), OjasError> {
    for t in tensors {
        f32_operand(op, t)?;
    }
    Ok(())
}

fn same(op: &'static str, a: &[usize], b: &[usize]) -> Result<(), OjasError> {
    if a == b {
        Ok(())
    } else {
        Err(refuse(op, format!("shape {a:?} does not match {b:?}")))
    }
}

/// All `f32`, every later tensor shaped like the first, in order. Returns
/// the element count.
fn same_shape(op: &'static str, tensors: &[&Tensor]) -> Result<usize, OjasError> {
    f32_operands(op, tensors)?;
    let Some((first, rest)) = tensors.split_first() else {
        return Err(refuse(op, "no operands"));
    };
    for t in rest {
        same(op, first.shape(), t.shape())?;
    }
    f32_product(op, first.shape())
}

/// `(leading axes, last axis)` of a rank-1-or-more shape, with `rank0` as
/// the refusal for rank 0.
fn split_last<'s>(
    op: &'static str,
    shape: &'s [usize],
    rank0: &str,
) -> Result<(&'s [usize], usize), OjasError> {
    match shape.split_last() {
        Some((&last, lead)) => Ok((lead, last)),
        None => Err(refuse(op, rank0)),
    }
}

fn table_dims(op: &'static str, table: &Tensor) -> Result<(usize, usize), OjasError> {
    match table.shape() {
        &[vocab, dim] => Ok((vocab, dim)),
        other => Err(refuse(
            op,
            format!("embedding table rank {} != 2 [vocab, dim]", other.len()),
        )),
    }
}

/// [`crate::Backend::argmax_rows`]: `x` `F32` `[rows, cols]`. Returns
/// `(rows, cols)`. Both must fit `u32`, so every result id does too.
pub fn argmax_rows_dims(x: &Tensor) -> Result<(usize, usize), OjasError> {
    const OP: &str = "argmax_rows";
    f32_operand(OP, x)?;
    let &[rows, cols] = x.shape() else {
        return Err(refuse(
            OP,
            format!("rank {} != 2 [rows, cols]", x.shape().len()),
        ));
    };
    if u32::try_from(rows).is_err() || u32::try_from(cols).is_err() {
        return Err(too_large(
            OP,
            format!("[{rows}, {cols}] does not fit u32 ids"),
        ));
    }
    Ok((rows, cols))
}

/// [`crate::Backend::embedding_forward`]: `table` `[vocab, dim]` `F32`,
/// `token_ids` `U32` of any rank (rank 0 included). The output is
/// `ids.shape ++ [dim]`. Id range is the backend's to check.
pub fn embedding_forward_dims(
    table: &Tensor,
    token_ids: &Tensor,
) -> Result<EmbeddingDims, OjasError> {
    const OP: &str = EMBEDDING_FORWARD;
    f32_operand(OP, table)?;
    let tokens = operand(OP, token_ids, DType::U32)?;
    let (vocab, dim) = table_dims(OP, table)?;
    let mut out_shape = token_ids.shape().to_vec();
    out_shape.push(dim);
    f32_product(OP, &out_shape)?;
    Ok(EmbeddingDims {
        vocab,
        dim,
        tokens,
        out_shape,
    })
}

/// [`crate::Backend::embedding_backward`]: as the forward, plus `grad_output`
/// `F32` shaped `ids.shape ++ [dim]`. The result is shaped like `table`.
pub fn embedding_backward_dims(
    table: &Tensor,
    token_ids: &Tensor,
    grad_output: &Tensor,
) -> Result<EmbeddingDims, OjasError> {
    const OP: &str = EMBEDDING_BACKWARD;
    f32_operand(OP, table)?;
    let tokens = operand(OP, token_ids, DType::U32)?;
    f32_operand(OP, grad_output)?;
    let (vocab, dim) = table_dims(OP, table)?;
    let mut expect = token_ids.shape().to_vec();
    expect.push(dim);
    if grad_output.shape() != expect.as_slice() {
        return Err(refuse(
            OP,
            format!(
                "embedding grad shape {:?} != {expect:?}",
                grad_output.shape()
            ),
        ));
    }
    Ok(EmbeddingDims {
        vocab,
        dim,
        tokens,
        out_shape: table.shape().to_vec(),
    })
}

fn linear_dims(op: &'static str, input: &Tensor, weight: &Tensor) -> Result<LinearDims, OjasError> {
    let (out_features, w_in) = match weight.shape() {
        &[out, w_in] => (out, w_in),
        other => {
            return Err(refuse(
                op,
                format!("weight rank {} != 2 ([out, in])", other.len()),
            ))
        }
    };
    let (lead, in_features) = split_last(op, input.shape(), "input rank 0 has no in-features")?;
    if in_features != w_in {
        return Err(refuse(
            op,
            format!("input in-features {in_features} != weight in-features {w_in}"),
        ));
    }
    let rows = f32_product(op, lead)?;
    let mut out_shape = lead.to_vec();
    out_shape.push(out_features);
    f32_product(op, &out_shape)?;
    Ok(LinearDims {
        rows,
        in_features,
        out_features,
        out_shape,
    })
}

/// [`crate::Backend::linear_forward`]: `input` `[..., in]` (rank 1 or more)
/// and `weight` `[out, in]`, each `F32` or `Bf16`. The output is `F32`.
pub fn linear_forward_dims(input: &Tensor, weight: &Tensor) -> Result<LinearDims, OjasError> {
    const OP: &str = LINEAR_FORWARD;
    matmul_operands(OP, &[input, weight])?;
    linear_dims(OP, input, weight)
}

/// [`crate::Backend::linear_backward`]: as the forward, plus `grad_output`
/// (`F32` or `Bf16`) shaped like the forward output. Both gradients are
/// `F32`.
pub fn linear_backward_dims(
    input: &Tensor,
    weight: &Tensor,
    grad_output: &Tensor,
) -> Result<LinearDims, OjasError> {
    const OP: &str = LINEAR_BACKWARD;
    matmul_operands(OP, &[input, weight, grad_output])?;
    let dims = linear_dims(OP, input, weight)?;
    if grad_output.shape() != dims.out_shape.as_slice() {
        return Err(refuse(
            OP,
            format!(
                "grad shape {:?} != forward shape {:?}",
                grad_output.shape(),
                dims.out_shape
            ),
        ));
    }
    Ok(dims)
}

fn rms_layout(
    op: &'static str,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<RmsDims, OjasError> {
    if !eps.is_finite() {
        return Err(OjasError::NonFinite { op });
    }
    let (lead, dim) = split_last(op, input.shape(), "rms_norm input rank 0")?;
    let w = weight.shape();
    if w != [dim] {
        return Err(refuse(op, format!("rms weight {w:?} != last dim {dim}")));
    }
    Ok(RmsDims {
        rows: f32_product(op, lead)?,
        dim,
    })
}

/// [`crate::Backend::rms_norm_forward`]: `input` `[..., dim]` (rank 1 or
/// more) and `weight` `[dim]`, both `F32`. A non-finite `eps` is
/// [`OjasError::NonFinite`], checked after the operands and before the
/// shapes, where every backend checks it today.
pub fn rms_norm_forward_dims(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<RmsDims, OjasError> {
    const OP: &str = RMS_NORM_FORWARD;
    f32_operands(OP, &[input, weight])?;
    rms_layout(OP, input, weight, eps)
}

/// [`crate::Backend::rms_norm_backward`]: `grad_output` shaped like `input`
/// (checked before `eps` and the weight), then as the forward.
pub fn rms_norm_backward_dims(
    input: &Tensor,
    weight: &Tensor,
    grad_output: &Tensor,
    eps: f32,
) -> Result<RmsDims, OjasError> {
    const OP: &str = RMS_NORM_BACKWARD;
    f32_operands(OP, &[input, weight, grad_output])?;
    same(OP, input.shape(), grad_output.shape())?;
    rms_layout(OP, input, weight, eps)
}

/// [`crate::Backend::rms_qk_norm_forward`]: [`rms_norm_forward_dims`] of
/// `(q, q_weight)` and then of `(k, k_weight)`, both before either norm runs.
/// Refusals name `rms_norm_forward`, as every backend's composition does.
pub fn rms_qk_norm_forward_dims(
    q: &Tensor,
    k: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    eps: f32,
) -> Result<(RmsDims, RmsDims), OjasError> {
    let qd = rms_norm_forward_dims(q, q_weight, eps)?;
    let kd = rms_norm_forward_dims(k, k_weight, eps)?;
    Ok((qd, kd))
}

/// [`crate::Backend::rms_qk_norm_backward`]: [`rms_norm_backward_dims`] of
/// `(q, q_weight, grad_q)` and then of `(k, k_weight, grad_k)`. Refusals name
/// `rms_norm_backward`.
#[allow(clippy::too_many_arguments)]
pub fn rms_qk_norm_backward_dims(
    q: &Tensor,
    k: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    grad_q: &Tensor,
    grad_k: &Tensor,
    eps: f32,
) -> Result<(RmsDims, RmsDims), OjasError> {
    let qd = rms_norm_backward_dims(q, q_weight, grad_q, eps)?;
    let kd = rms_norm_backward_dims(k, k_weight, grad_k, eps)?;
    Ok((qd, kd))
}

fn rope_dims(
    op: &'static str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<RopeDims, OjasError> {
    f32_operands(op, &[x, cos, sin])?;
    let (xs, c, s) = (x.shape(), cos.shape(), sin.shape());
    if c != s {
        return Err(refuse(op, format!("cos shape {c:?} != sin shape {s:?}")));
    }
    let (lead, dim) = split_last(op, xs, "rope input rank 0")?;
    if dim % 2 != 0 {
        return Err(refuse(op, format!("rope last dim {dim} is odd")));
    }
    let rows = f32_product(op, lead)?;
    let layout = match (xs, c) {
        _ if c == xs => RopeLayout::Same,
        (&[_, time, heads, _], &[c_time, c_dim]) if c_time == time && c_dim == dim => {
            RopeLayout::TimeDim { time, heads }
        }
        _ => {
            return Err(refuse(
                op,
                format!("cos/sin shape {c:?} does not broadcast onto {xs:?}"),
            ))
        }
    };
    Ok(RopeDims { rows, dim, layout })
}

/// [`crate::Backend::rope_half_split_forward`]: `F32` `x` `[..., dim]` (rank 1
/// or more, `dim` even), `cos` and `sin` of equal shape, either `x`'s shape
/// or `[T, dim]` against an `x` of `[B, T, H, dim]`.
pub fn rope_half_split_forward_dims(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<RopeDims, OjasError> {
    rope_dims(ROPE_FORWARD, x, cos, sin)
}

/// [`crate::Backend::rope_half_split_backward`]: the forward's rules with
/// `grad_output` in place of `x`.
pub fn rope_half_split_backward_dims(
    grad_output: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<RopeDims, OjasError> {
    rope_dims(ROPE_BACKWARD, grad_output, cos, sin)
}

fn sdpa_dims(op: &'static str, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<SdpaDims, OjasError> {
    let (qs, ks, vs) = (q.shape(), k.shape(), v.shape());
    let &[batch, heads, seq, head_dim] = qs else {
        return Err(refuse(
            op,
            format!("sdpa query rank {} != 4 [B, H, T, D]", qs.len()),
        ));
    };
    // K and V are one shape. A mismatch is a shape error even when each
    // would be a legal KV layout against Q on its own.
    if ks != vs {
        return Err(refuse(op, format!("sdpa k {ks:?} and v {vs:?} differ")));
    }
    let &[kb, kv_heads, kseq, kd] = ks else {
        return Err(refuse(
            op,
            format!("sdpa shapes q {qs:?} k {ks:?} v {vs:?} differ"),
        ));
    };
    // Equal shapes, including zero query heads and zero KV heads, stay
    // multi-head. Grouping is checked only when the head counts differ, so
    // `0 % 0` never runs.
    let grouped = kv_heads != heads;
    let bad_group =
        grouped && (kv_heads == 0 || heads == 0 || kv_heads > heads || heads % kv_heads != 0);
    let bad_axes = kb != batch || kseq != seq || kd != head_dim;
    if bad_axes || bad_group {
        return Err(refuse(
            op,
            format!("sdpa shapes q {qs:?} k {ks:?} v {vs:?} differ"),
        ));
    }
    Ok(SdpaDims {
        batch,
        heads,
        kv_heads,
        seq,
        head_dim,
    })
}

/// A sliding window keeps at least the query's own key.
fn sdpa_window(op: &'static str, window: Option<usize>) -> Result<(), OjasError> {
    if window == Some(0) {
        return Err(refuse(op, "sdpa window must be at least 1"));
    }
    Ok(())
}

/// [`crate::Backend::causal_sdpa_forward`]: `q`, `k` and `v` each `F32` or
/// `Bf16`; the output and log-sum-exp are `F32`. Query is
/// rank 4 `[B, H, T, D]`; K and V are `[B, Hkv, T, D]` with the same batch,
/// sequence and head dimension, and `H` a positive multiple of `Hkv`. A
/// window, when given, is at least 1. The head-dimension limit of a device
/// is the backend's.
pub fn causal_sdpa_forward_dims(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window: Option<usize>,
) -> Result<SdpaDims, OjasError> {
    const OP: &str = SDPA_FORWARD;
    matmul_operands(OP, &[q, k, v])?;
    sdpa_window(OP, window)?;
    sdpa_dims(OP, q, k, v)
}

/// `grad_output` is shaped like `q`: the first rule of
/// [`causal_sdpa_backward_dims`], also checked by
/// [`crate::Backend::causal_sdpa_backward_recompute`] before its forward.
pub(crate) fn causal_sdpa_grad_shape(q: &Tensor, grad_output: &Tensor) -> Result<(), OjasError> {
    if grad_output.shape() != q.shape() {
        return Err(refuse(
            SDPA_BACKWARD,
            format!(
                "sdpa grad shape {:?} != query {:?}",
                grad_output.shape(),
                q.shape()
            ),
        ));
    }
    Ok(())
}

/// [`crate::Backend::causal_sdpa_backward`]: `lse` `F32`, every other
/// operand `F32` or `Bf16`, checked in argument order; the gradients are
/// `F32`. `grad_output` shaped like `q` (checked first), then `output` shaped like
/// `q` and `lse` `[B, H, T]` (`q`'s shape without its last axis), then as the
/// forward.
pub fn causal_sdpa_backward_dims(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    lse: &Tensor,
    grad_output: &Tensor,
    window: Option<usize>,
) -> Result<SdpaDims, OjasError> {
    const OP: &str = SDPA_BACKWARD;
    matmul_operands(OP, &[q, k, v, output])?;
    f32_operand(OP, lse)?;
    matmul_operand(OP, grad_output)?;
    causal_sdpa_grad_shape(q, grad_output)?;
    if output.shape() != q.shape() {
        return Err(refuse(
            OP,
            format!(
                "sdpa output shape {:?} != query {:?}",
                output.shape(),
                q.shape()
            ),
        ));
    }
    let rows = &q.shape()[..q.shape().len().saturating_sub(1)];
    if lse.shape() != rows {
        return Err(refuse(
            OP,
            format!("sdpa lse shape {:?} != {rows:?}", lse.shape()),
        ));
    }
    sdpa_window(OP, window)?;
    sdpa_dims(OP, q, k, v)
}

fn gate_layout(
    op: &'static str,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    attn_out: &Tensor,
) -> Result<GateDims, OjasError> {
    let (x, b, a) = (input.shape(), bias.shape(), attn_out.shape());
    let &[heads, w_in] = weight.shape() else {
        return Err(refuse(op, "gate weight must be [n_head, d_model]"));
    };
    let (x_lead, d_model) = split_last(op, x, "gate input rank 0")?;
    if w_in != d_model {
        return Err(refuse(
            op,
            format!("gate weight in {w_in} != input dim {d_model}"),
        ));
    }
    if b != [heads] {
        return Err(refuse(op, format!("gate bias {b:?} != [{heads}]")));
    }
    // `x` has rank 1 or more here, so a matching `a` has rank 2 or more.
    if a.len() != x.len() + 1 {
        return Err(refuse(op, "gate attn rank must be input rank + 1"));
    }
    let (a_lead, a_tail) = a.split_at(a.len() - 2);
    let &[a_heads, head_dim] = a_tail else {
        return Err(refuse(op, "gate attn rank must be input rank + 1"));
    };
    if a_heads != heads {
        return Err(refuse(op, "gate attn head axis != weight rows"));
    }
    if a_lead != x_lead {
        return Err(refuse(op, "gate attn prefix does not match input prefix"));
    }
    Ok(GateDims {
        rows: f32_product(op, x_lead)?,
        d_model,
        heads,
        head_dim,
    })
}

/// [`crate::Backend::per_head_sigmoid_gate_forward`]: all `F32`; `weight`
/// `[heads, d_model]`, `input` `[..., d_model]` (rank 1 or more), `bias`
/// `[heads]`, `attn_out` `[..., heads, head_dim]` with `input`'s leading
/// axes. The output is shaped like `attn_out`.
pub fn per_head_sigmoid_gate_forward_dims(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    attn_out: &Tensor,
) -> Result<GateDims, OjasError> {
    const OP: &str = GATE_FORWARD;
    f32_operands(OP, &[input, weight, bias, attn_out])?;
    gate_layout(OP, input, weight, bias, attn_out)
}

/// [`crate::Backend::per_head_sigmoid_gate_backward`]: `grad_output` shaped
/// like `attn_out` (checked first), then as the forward.
pub fn per_head_sigmoid_gate_backward_dims(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    attn_out: &Tensor,
    grad_output: &Tensor,
) -> Result<GateDims, OjasError> {
    const OP: &str = GATE_BACKWARD;
    f32_operands(OP, &[input, weight, bias, attn_out, grad_output])?;
    same(OP, attn_out.shape(), grad_output.shape())?;
    gate_layout(OP, input, weight, bias, attn_out)
}

fn gdn_layout(op: &'static str, x: &GdnInputs<'_>) -> Result<GdnDims, OjasError> {
    let &[batch, seq, heads, key_dim] = x.q.shape() else {
        return Err(refuse(op, "gdn q must be rank 4 [B, T, H, Dk]"));
    };
    if x.k.shape() != x.q.shape() {
        return Err(refuse(
            op,
            format!("gdn k {:?} != q {:?}", x.k.shape(), x.q.shape()),
        ));
    }
    let &[vb, vt, vh, value_dim] = x.v.shape() else {
        return Err(refuse(op, "gdn v must be rank 4 [B, T, H, Dv]"));
    };
    if [vb, vt, vh] != [batch, seq, heads] {
        return Err(refuse(
            op,
            format!(
                "gdn v {:?} does not share q's [B, T, H] {:?}",
                x.v.shape(),
                [batch, seq, heads]
            ),
        ));
    }
    for (name, t) in [("g", x.g), ("beta", x.beta)] {
        if t.shape() != [batch, seq, heads] {
            return Err(refuse(
                op,
                format!(
                    "gdn {name} {:?} != [B, T, H] {:?}",
                    t.shape(),
                    [batch, seq, heads]
                ),
            ));
        }
    }
    let dims = GdnDims {
        batch,
        seq,
        heads,
        key_dim,
        value_dim,
        checkpoints: seq.div_ceil(GDN_CHECKPOINT_TOKENS),
    };
    if let Some(s0) = x.initial_state {
        if s0.shape() != dims.state_shape() {
            return Err(refuse(
                op,
                format!(
                    "gdn initial state {:?} != [B, H, Dk, Dv] {:?}",
                    s0.shape(),
                    dims.state_shape()
                ),
            ));
        }
    }
    f32_product(op, &dims.checkpoint_shape())?;
    Ok(dims)
}

fn gdn_operands<'a>(x: &GdnInputs<'a>) -> Vec<&'a Tensor> {
    let mut ts = vec![x.q, x.k, x.v, x.g, x.beta];
    ts.extend(x.initial_state);
    ts
}

/// [`crate::Backend::chunked_gdn_forward`]: every operand `F32`, in the order
/// `q, k, v, g, beta, initial_state`. Then `q` rank 4 `[B, T, H, Dk]`, `k`
/// shaped like `q`, `v` `[B, T, H, Dv]`, `g` and `beta` `[B, T, H]`, and
/// `initial_state` `[B, H, Dk, Dv]`. The checkpoint tensor's size fits.
/// The key and value dims a device's kernels take are the backend's.
pub fn chunked_gdn_forward_dims(inputs: GdnInputs<'_>) -> Result<GdnDims, OjasError> {
    const OP: &str = GDN_FORWARD;
    f32_operands(OP, &gdn_operands(&inputs))?;
    gdn_layout(OP, &inputs)
}

/// [`crate::Backend::chunked_gdn_backward`]: the forward's operands, then
/// `checkpoints`, `grad_output` and `grad_final_state`, all `F32`. Then the
/// forward's layout, `checkpoints` `[B, H, ceil(T / 64), Dk, Dv]`,
/// `grad_output` shaped like `v`, and `grad_final_state` `[B, H, Dk, Dv]`.
pub fn chunked_gdn_backward_dims(
    inputs: GdnInputs<'_>,
    checkpoints: &Tensor,
    grad_output: &Tensor,
    grad_final_state: Option<&Tensor>,
) -> Result<GdnDims, OjasError> {
    const OP: &str = GDN_BACKWARD;
    let mut ts = gdn_operands(&inputs);
    ts.push(checkpoints);
    ts.push(grad_output);
    ts.extend(grad_final_state);
    f32_operands(OP, &ts)?;
    let dims = gdn_layout(OP, &inputs)?;
    if checkpoints.shape() != dims.checkpoint_shape() {
        return Err(refuse(
            OP,
            format!(
                "gdn checkpoints {:?} != [B, H, NC, Dk, Dv] {:?}",
                checkpoints.shape(),
                dims.checkpoint_shape()
            ),
        ));
    }
    same(OP, inputs.v.shape(), grad_output.shape())?;
    if let Some(gf) = grad_final_state {
        if gf.shape() != dims.state_shape() {
            return Err(refuse(
                OP,
                format!(
                    "gdn final state grad {:?} != [B, H, Dk, Dv] {:?}",
                    gf.shape(),
                    dims.state_shape()
                ),
            ));
        }
    }
    Ok(dims)
}

/// One element of any rank: every axis is 1 (zero axes were refused).
fn is_scalar(t: &Tensor) -> bool {
    t.shape().iter().all(|&d| d == 1)
}

fn conv1d_layout(
    op: &'static str,
    input: &Tensor,
    weight: &Tensor,
) -> Result<Conv1dDims, OjasError> {
    let &[batch, time, channels] = input.shape() else {
        return Err(refuse(
            op,
            format!(
                "conv1d input rank {} != 3 [batch, time, channels]",
                input.shape().len()
            ),
        ));
    };
    let &[w_channels, width] = weight.shape() else {
        return Err(refuse(
            op,
            format!(
                "conv1d weight rank {} != 2 [channels, width]",
                weight.shape().len()
            ),
        ));
    };
    if w_channels != channels {
        return Err(refuse(
            op,
            format!("conv1d weight channels {w_channels} != input channels {channels}"),
        ));
    }
    Ok(Conv1dDims {
        batch,
        time,
        channels,
        width,
    })
}

/// [`crate::Backend::causal_conv1d_silu_forward`]: `input` `[batch, time,
/// channels]` and `weight` `[channels, width]`, both `F32`. The output is
/// shaped like `input`. Which widths a kernel is compiled for is the
/// device's to check.
pub fn causal_conv1d_silu_forward_dims(
    input: &Tensor,
    weight: &Tensor,
) -> Result<Conv1dDims, OjasError> {
    const OP: &str = CONV1D_FORWARD;
    f32_operands(OP, &[input, weight])?;
    conv1d_layout(OP, input, weight)
}

/// [`crate::Backend::causal_conv1d_silu_backward`]: the forward's operands,
/// then `grad_output` shaped like `input`.
pub fn causal_conv1d_silu_backward_dims(
    input: &Tensor,
    weight: &Tensor,
    grad_output: &Tensor,
) -> Result<Conv1dDims, OjasError> {
    const OP: &str = CONV1D_BACKWARD;
    f32_operands(OP, &[input, weight, grad_output])?;
    same(OP, input.shape(), grad_output.shape())?;
    conv1d_layout(OP, input, weight)
}

/// [`crate::Backend::gated_rms_norm_forward`]: `input` and `gate` of one
/// shape, rank 1 or more, `weight` `[dim]` for the last axis `dim`, all
/// `F32`; `eps` finite. Rows are every leading axis (a head is a row).
pub fn gated_rms_norm_forward_dims(
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<RmsDims, OjasError> {
    const OP: &str = GATED_RMS_FORWARD;
    f32_operands(OP, &[input, gate, weight])?;
    same(OP, input.shape(), gate.shape())?;
    rms_layout(OP, input, weight, eps)
}

/// [`crate::Backend::gated_rms_norm_backward`]: the forward's operands,
/// then `grad_output` shaped like `input`.
pub fn gated_rms_norm_backward_dims(
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    grad_output: &Tensor,
    eps: f32,
) -> Result<RmsDims, OjasError> {
    const OP: &str = GATED_RMS_BACKWARD;
    f32_operands(OP, &[input, gate, weight, grad_output])?;
    same(OP, input.shape(), gate.shape())?;
    same(OP, input.shape(), grad_output.shape())?;
    rms_layout(OP, input, weight, eps)
}

fn rope_partial_layout(
    op: &'static str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<PartialRopeDims, OjasError> {
    f32_operands(op, &[x, cos, sin])?;
    let (c, s) = (cos.shape(), sin.shape());
    if c != s {
        return Err(refuse(op, format!("cos shape {c:?} != sin shape {s:?}")));
    }
    let &[batch, time, heads, dim] = x.shape() else {
        return Err(refuse(
            op,
            format!(
                "partial rope input rank {} != 4 [batch, time, heads, head_dim]",
                x.shape().len()
            ),
        ));
    };
    let &[c_time, rotary] = c else {
        return Err(refuse(
            op,
            format!("cos/sin {c:?} are not [time, rotary_dim]"),
        ));
    };
    if c_time != time {
        return Err(refuse(
            op,
            format!("cos/sin hold {c_time} positions for {time}"),
        ));
    }
    if !rotary.is_multiple_of(2) || rotary > dim {
        return Err(refuse(
            op,
            format!("rotary_dim {rotary} must be even and at most head_dim {dim}"),
        ));
    }
    Ok(PartialRopeDims {
        rows: f32_product(op, &[batch, time, heads])?,
        time,
        heads,
        dim,
        rotary,
    })
}

/// [`crate::Backend::rope_partial_forward`]: `x` `[batch, time, heads,
/// head_dim]`, `cos` and `sin` `[time, rotary_dim]` with `rotary_dim` even
/// and at most `head_dim`, all `F32`.
pub fn rope_partial_forward_dims(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<PartialRopeDims, OjasError> {
    rope_partial_layout(ROPE_PARTIAL_FORWARD, x, cos, sin)
}

/// [`crate::Backend::rope_partial_backward`]: `grad_output` in place of `x`.
pub fn rope_partial_backward_dims(
    grad_output: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<PartialRopeDims, OjasError> {
    rope_partial_layout(ROPE_PARTIAL_BACKWARD, grad_output, cos, sin)
}

/// [`crate::Backend::value_residual_blend_forward`]: all `F32`, `lambda` one
/// element of any rank, `value0` shaped like `value`. Returns the element
/// count of `value`.
pub fn value_residual_blend_forward_dims(
    value: &Tensor,
    value0: &Tensor,
    lambda: &Tensor,
) -> Result<usize, OjasError> {
    const OP: &str = VALUE_RESIDUAL_FORWARD;
    f32_operands(OP, &[value, value0, lambda])?;
    if !is_scalar(lambda) {
        return Err(refuse(OP, "expected a scalar tensor"));
    }
    same(OP, value.shape(), value0.shape())?;
    f32_product(OP, value.shape())
}

/// [`crate::Backend::value_residual_blend_backward`]: as the forward, plus
/// `grad_output` shaped like `value`.
pub fn value_residual_blend_backward_dims(
    value: &Tensor,
    value0: &Tensor,
    lambda: &Tensor,
    grad_output: &Tensor,
) -> Result<usize, OjasError> {
    const OP: &str = VALUE_RESIDUAL_BACKWARD;
    f32_operands(OP, &[value, value0, lambda, grad_output])?;
    if !is_scalar(lambda) {
        return Err(refuse(OP, "value residual lambda must be a scalar"));
    }
    same(OP, value.shape(), value0.shape())?;
    same(OP, value.shape(), grad_output.shape())?;
    f32_product(OP, value.shape())
}

/// [`crate::Backend::silu_forward`]: one `F32` tensor of any rank. Returns
/// its element count.
pub fn silu_forward_dims(input: &Tensor) -> Result<usize, OjasError> {
    same_shape(SILU_FORWARD, &[input])
}

/// [`crate::Backend::silu_backward`]: `grad_output` shaped like `input`.
pub fn silu_backward_dims(input: &Tensor, grad_output: &Tensor) -> Result<usize, OjasError> {
    same_shape(SILU_BACKWARD, &[input, grad_output])
}

/// [`crate::Backend::sigmoid_forward`]: one `F32` tensor of any rank.
/// Returns its element count.
pub fn sigmoid_forward_dims(input: &Tensor) -> Result<usize, OjasError> {
    same_shape(SIGMOID_FORWARD, &[input])
}

/// [`crate::Backend::sigmoid_backward`]: `grad_output` shaped like `input`.
pub fn sigmoid_backward_dims(input: &Tensor, grad_output: &Tensor) -> Result<usize, OjasError> {
    same_shape(SIGMOID_BACKWARD, &[input, grad_output])
}

fn gdn_decay_layout(
    op: &'static str,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<GdnDecayDims, OjasError> {
    f32_operands(op, &[a, a_log, dt_bias])?;
    let (lead, heads) = split_last(op, a.shape(), "gdn log decay input must have rank >= 1")?;
    for (name, t) in [("a_log", a_log), ("dt_bias", dt_bias)] {
        if t.shape() != [heads] {
            return Err(refuse(
                op,
                format!("{name} shape {:?} is not [{heads}]", t.shape()),
            ));
        }
    }
    Ok(GdnDecayDims {
        rows: f32_product(op, lead)?,
        heads,
    })
}

/// [`crate::Backend::gdn_log_decay_forward`]: `a` `[..., heads]` of rank 1
/// or more, `a_log` and `dt_bias` `[heads]`, all `F32`.
pub fn gdn_log_decay_forward_dims(
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<GdnDecayDims, OjasError> {
    gdn_decay_layout(GDN_DECAY_FORWARD, a, a_log, dt_bias)
}

/// [`crate::Backend::gdn_log_decay_backward`]: the forward's operands, then
/// `grad_output` shaped like `a`.
pub fn gdn_log_decay_backward_dims(
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    grad_output: &Tensor,
) -> Result<GdnDecayDims, OjasError> {
    const OP: &str = GDN_DECAY_BACKWARD;
    f32_operand(OP, grad_output)?;
    let dims = gdn_decay_layout(OP, a, a_log, dt_bias)?;
    same(OP, a.shape(), grad_output.shape())?;
    Ok(dims)
}

/// [`crate::Backend::mul_forward`]: `a` and `b` of one shape, no broadcast.
pub fn mul_forward_dims(a: &Tensor, b: &Tensor) -> Result<usize, OjasError> {
    same_shape(MUL_FORWARD, &[a, b])
}

/// [`crate::Backend::mul_backward`]: `b`, then `grad_output`, shaped like `a`.
pub fn mul_backward_dims(a: &Tensor, b: &Tensor, grad_output: &Tensor) -> Result<usize, OjasError> {
    same_shape(MUL_BACKWARD, &[a, b, grad_output])
}

/// [`crate::Backend::residual_add_forward`]: `x` and `y` of one shape.
pub fn residual_add_forward_dims(x: &Tensor, y: &Tensor) -> Result<usize, OjasError> {
    same_shape(ADD_FORWARD, &[x, y])
}

/// [`crate::Backend::residual_add_backward`]: `y`, then `grad_output`,
/// shaped like `x`.
pub fn residual_add_backward_dims(
    x: &Tensor,
    y: &Tensor,
    grad_output: &Tensor,
) -> Result<usize, OjasError> {
    same_shape(ADD_BACKWARD, &[x, y, grad_output])
}

/// [`crate::Backend::accumulate_grad`]: `grad` shaped like `acc`, both `F32`
/// (the rules of [`residual_add_forward_dims`], reported as
/// `accumulate_grad`). Returns the element count.
pub fn accumulate_grad_dims(acc: &Tensor, grad: &Tensor) -> Result<usize, OjasError> {
    same_shape(ACCUMULATE_GRAD, &[acc, grad])
}

/// Largest rank [`crate::Backend::permute`] accepts.
pub const MAX_PERMUTE_RANK: usize = 8;

/// [`crate::Backend::permute`]: an `F32` `input` with no zero axis, and
/// `dims` naming each of its axes exactly once ([`permute_output_shape`]).
/// Returns the output shape. Rank 0 with `dims` `[]` is accepted: it is one
/// element. A zero axis is `"empty tensor"`, as for every other op (D17 in
/// `docs/shape-contract.md`).
pub fn permute_dims(input: &Tensor, dims: &[usize]) -> Result<Vec<usize>, OjasError> {
    f32_operand(PERMUTE, input)?;
    permute_output_shape(PERMUTE, input.shape(), dims)
}

/// Output shape of permuting `shape` by `dims`, after checking that `dims`
/// names every axis of `shape` exactly once. Reads shapes only, so a caller
/// that holds no tensor (the tape, before it records) can check axes with
/// it; a backend validates with [`permute_dims`].
///
/// Output axis `i` is input axis `dims[i]`, as in `torch.permute`. A rank
/// above [`MAX_PERMUTE_RANK`], a length mismatch, an axis out of range, or a
/// repeated axis is [`OjasError::Shape`]; nothing is clamped or reordered.
pub fn permute_output_shape(
    op: &'static str,
    shape: &[usize],
    dims: &[usize],
) -> Result<Vec<usize>, OjasError> {
    if shape.len() > MAX_PERMUTE_RANK {
        return Err(refuse(
            op,
            format!("rank {} exceeds {MAX_PERMUTE_RANK}", shape.len()),
        ));
    }
    if dims.len() != shape.len() {
        return Err(refuse(
            op,
            format!(
                "{} permutation axes for a rank-{} tensor",
                dims.len(),
                shape.len()
            ),
        ));
    }
    let mut seen = [false; MAX_PERMUTE_RANK];
    for &axis in dims {
        if axis >= shape.len() {
            return Err(refuse(
                op,
                format!("axis {axis} out of range for rank {}", shape.len()),
            ));
        }
        if seen[axis] {
            return Err(refuse(op, format!("axis {axis} repeated in {dims:?}")));
        }
        seen[axis] = true;
    }
    Ok(dims.iter().map(|&axis| shape[axis]).collect())
}

/// The permutation that undoes `dims`: permuting by `dims` and then by the
/// result is the identity. The gradient of a permute is the upstream gradient
/// permuted by this. `dims` must already have passed [`permute_output_shape`].
pub fn inverse_permutation(dims: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; dims.len()];
    for (out_axis, &in_axis) in dims.iter().enumerate() {
        inverse[in_axis] = out_axis;
    }
    inverse
}

fn ce_dims(op: &'static str, logits: &Tensor, targets: &Tensor) -> Result<CeDims, OjasError> {
    f32_operand(op, logits)?;
    operand(op, targets, DType::U32)?;
    let (prefix, vocab) = split_last(op, logits.shape(), "cross-entropy logits rank 0")?;
    let t = targets.shape();
    if t != prefix {
        return Err(refuse(
            op,
            format!("targets {t:?} != logits prefix {prefix:?}"),
        ));
    }
    Ok(CeDims {
        rows: f32_product(op, prefix)?,
        vocab,
    })
}

/// [`crate::Backend::cross_entropy_mean_forward`]: `logits` `F32` `[..., vocab]`
/// (rank 1 or more) and `targets` `U32` shaped like the logits' leading
/// axes (rank 0 for rank-1 logits). Target range and the all-ignored case
/// are the backend's.
pub fn cross_entropy_mean_forward_dims(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<CeDims, OjasError> {
    ce_dims(CE_FORWARD, logits, targets)
}

/// [`crate::Backend::cross_entropy_mean_backward`]: the forward's rules. The
/// gradient is shaped like `logits`.
pub fn cross_entropy_mean_backward_dims(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<CeDims, OjasError> {
    ce_dims(CE_BACKWARD, logits, targets)
}

/// [`crate::Backend::clip_grad_norm`]: at least one gradient, each `F32` of
/// any rank and any shape. Returns the total element count, whose `f32`
/// bytes fit `usize`. `max_norm` is checked by [`crate::clip_scale`].
pub fn clip_grad_norm_dims(grads: &[Tensor]) -> Result<usize, OjasError> {
    const OP: &str = CLIP;
    if grads.is_empty() {
        return Err(refuse(OP, "empty tensor"));
    }
    let mut total = 0usize;
    for g in grads {
        let n = f32_operand(OP, g)?;
        total = total
            .checked_add(n)
            .ok_or_else(|| too_large(OP, "total gradient length overflows"))?;
    }
    total
        .checked_mul(DType::F32.size())
        .ok_or_else(|| too_large(OP, "total gradient byte length overflows"))?;
    Ok(total)
}

/// [`crate::Backend::adamw_step`]: `param`, `grad`, `moment1` and `moment2`
/// `F32`, each later one shaped like `param`. Returns the element count.
/// The config and step are checked by [`crate::check_adamw`].
pub fn adamw_step_dims(
    param: &Tensor,
    grad: &Tensor,
    moment1: &Tensor,
    moment2: &Tensor,
) -> Result<usize, OjasError> {
    same_shape(ADAMW, &[param, grad, moment1, moment2])
}

/// [`crate::Backend::muon_ns5_step`]: `param` a rank-2 `F32` matrix, `grad`
/// and then `momentum` shaped like it. The config is the backend's.
pub fn muon_ns5_step_dims(
    param: &Tensor,
    grad: &Tensor,
    momentum: &Tensor,
) -> Result<MuonDims, OjasError> {
    const OP: &str = MUON;
    f32_operands(OP, &[param, grad, momentum])?;
    let &[rows, cols] = param.shape() else {
        return Err(refuse(OP, "muon parameter must be a matrix"));
    };
    same(OP, param.shape(), grad.shape())?;
    same(OP, param.shape(), momentum.shape())?;
    Ok(MuonDims { rows, cols })
}

/// Validated dimensions of a [`crate::Backend::linear_cross_entropy_mean`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearCeDims {
    pub rows: usize,
    pub model_dim: usize,
    pub vocab: usize,
}

/// Check the operands of [`crate::Backend::linear_cross_entropy_mean`]: `input`
/// `[N, d]` and `weight` `[V, d]` in `F32`, `targets` `[N]` in `U32`, `N`,
/// `d` and `V` non-zero, and both chunk dimensions non-zero. Target values
/// are the backend's to check, as in the unfused path.
pub fn linear_ce_dims(
    input: &Tensor,
    weight: &Tensor,
    targets: &Tensor,
    chunk: CeChunk,
) -> Result<LinearCeDims, OjasError> {
    const OP: &str = "linear_cross_entropy_mean";
    operand(OP, input, DType::F32)?;
    operand(OP, weight, DType::F32)?;
    operand(OP, targets, DType::U32)?;
    let (x, w, t) = (input.shape(), weight.shape(), targets.shape());
    let ([rows, model_dim], [vocab, w_dim], [t_rows]) = (x, w, t) else {
        return Err(refuse(
            OP,
            format!("expected input [N, d], weight [V, d], targets [N]; got {x:?}, {w:?}, {t:?}"),
        ));
    };
    if model_dim != w_dim || rows != t_rows {
        return Err(refuse(
            OP,
            format!("input {x:?}, weight {w:?} and targets {t:?} disagree"),
        ));
    }
    if *rows == 0 || *model_dim == 0 || *vocab == 0 {
        return Err(refuse(
            OP,
            format!("empty operand: input {x:?}, weight {w:?}"),
        ));
    }
    if chunk.rows == 0 || chunk.cols == 0 {
        return Err(refuse(OP, format!("chunk {chunk:?} has a zero dimension")));
    }
    Ok(LinearCeDims {
        rows: *rows,
        model_dim: *model_dim,
        vocab: *vocab,
    })
}

/// Validated dimensions of a KV-cache attention or write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvDims {
    pub batch: usize,
    /// New positions: `Tq` for attention, `Tn` for a write.
    pub new: usize,
    pub capacity: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

fn rank4(op: &'static str, name: &str, tensor: &Tensor) -> Result<[usize; 4], OjasError> {
    f32_operand(op, tensor)?;
    match tensor.shape() {
        [a, b, c, d] => Ok([*a, *b, *c, *d]),
        other => Err(refuse(op, format!("{name} must be rank 4, got {other:?}"))),
    }
}

/// Check the operands of [`crate::Backend::cached_attention_forward`]: `F32`
/// `q [B, Tq, H, D]`, `k_cache` and `v_cache [B, Tcap, Hkv, D]` of equal
/// shape, every dimension non-zero, `H % Hkv == 0`, and
/// `Tq <= kv_len <= Tcap`.
pub fn cached_attention_dims(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    kv_len: usize,
) -> Result<KvDims, OjasError> {
    const OP: &str = "cached_attention_forward";
    let [batch, tq, heads, head_dim] = rank4(OP, "q", q)?;
    let k = rank4(OP, "k_cache", k_cache)?;
    let v = rank4(OP, "v_cache", v_cache)?;
    if k != v {
        return Err(refuse(OP, format!("k_cache {k:?} != v_cache {v:?}")));
    }
    let [kb, capacity, kv_heads, kd] = k;
    if kb != batch || kd != head_dim {
        return Err(refuse(
            OP,
            format!("q {:?} and cache {k:?} disagree", q.shape()),
        ));
    }
    if [batch, tq, heads, head_dim, capacity, kv_heads].contains(&0) {
        return Err(refuse(
            OP,
            format!("zero dimension in q {:?} or cache {k:?}", q.shape()),
        ));
    }
    if heads % kv_heads != 0 {
        return Err(refuse(
            OP,
            format!("{heads} query heads over {kv_heads} kv heads"),
        ));
    }
    if kv_len < tq || kv_len > capacity {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: format!("kv_len {kv_len} outside {tq}..={capacity}"),
        });
    }
    Ok(KvDims {
        batch,
        new: tq,
        capacity,
        heads,
        kv_heads,
        head_dim,
    })
}

/// Check the operands of [`crate::Backend::kv_cache_write`]: `F32` `cache
/// [B, Tcap, Hkv, D]` and `src [B, Tn, Hkv, D]`, every dimension non-zero,
/// and `at + Tn <= Tcap` without overflow.
pub fn kv_cache_write_dims(cache: &Tensor, src: &Tensor, at: usize) -> Result<KvDims, OjasError> {
    const OP: &str = "kv_cache_write";
    let [batch, capacity, kv_heads, head_dim] = rank4(OP, "cache", cache)?;
    let [sb, tn, sh, sd] = rank4(OP, "src", src)?;
    if [sb, sh, sd] != [batch, kv_heads, head_dim] {
        return Err(refuse(
            OP,
            format!(
                "src {:?} does not fit cache {:?}",
                src.shape(),
                cache.shape()
            ),
        ));
    }
    if [batch, capacity, kv_heads, head_dim, tn].contains(&0) {
        return Err(refuse(
            OP,
            format!(
                "zero dimension in cache {:?} or src {:?}",
                cache.shape(),
                src.shape()
            ),
        ));
    }
    let end = at.checked_add(tn).ok_or_else(|| OjasError::OutOfRange {
        op: OP,
        detail: format!("at {at} + {tn} overflows"),
    })?;
    if end > capacity {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: format!("positions {at}..{end} exceed capacity {capacity}"),
        });
    }
    Ok(KvDims {
        batch,
        new: tn,
        capacity,
        heads: kv_heads,
        kv_heads,
        head_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendId, Budget, DeviceBuffer};
    use std::any::Any;
    use std::sync::Arc;

    fn budget() -> Budget {
        Budget::new(1 << 24)
    }

    fn f(shape: &[usize]) -> Tensor {
        Tensor::zeros(shape, DType::F32, &budget()).unwrap()
    }

    fn u(shape: &[usize]) -> Tensor {
        Tensor::zeros(shape, DType::U32, &budget()).unwrap()
    }

    fn of(dtype: DType, shape: &[usize]) -> Tensor {
        Tensor::zeros(shape, dtype, &budget()).unwrap()
    }

    /// A one-element tensor viewed with stride 0 on every axis: legal
    /// metadata of any shape, whatever its product, with no allocation.
    fn broadcast(dtype: DType, shape: &[usize]) -> Tensor {
        of(dtype, &[1])
            .view(shape, &vec![0; shape.len()], 0)
            .unwrap()
    }

    /// Device memory that reports `len` bytes and holds none, so a
    /// contiguous tensor of any size can exist as metadata.
    #[derive(Debug)]
    struct Phantom(usize);

    impl DeviceBuffer for Phantom {
        fn backend(&self) -> BackendId {
            BackendId::Metal
        }
        fn byte_len(&self) -> usize {
            self.0
        }
        fn read_bytes(&self, _: usize, _: usize) -> Result<Vec<u8>, OjasError> {
            Err(OjasError::Unsupported {
                op: "Phantom::read_bytes",
                detail: "metadata only".to_string(),
            })
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn phantom(dtype: DType, shape: &[usize]) -> Tensor {
        let bytes = shape.iter().product::<usize>() * dtype.size();
        Tensor::from_device(
            Arc::new(Phantom(bytes)),
            shape,
            dtype,
            &Budget::new(u64::MAX),
        )
        .unwrap()
    }

    #[track_caller]
    fn shape_err<T: std::fmt::Debug>(r: Result<T, OjasError>, op: &str, needle: &str) {
        match r {
            Err(OjasError::Shape { op: got, detail }) => {
                assert_eq!(got, op, "{detail}");
                assert!(detail.contains(needle), "{op}: {detail:?} lacks {needle:?}");
            }
            other => panic!("{op}: expected Shape containing {needle:?}, got {other:?}"),
        }
    }

    #[track_caller]
    fn range_err<T: std::fmt::Debug>(r: Result<T, OjasError>, op: &str, needle: &str) {
        match r {
            Err(OjasError::OutOfRange { op: got, detail }) => {
                assert_eq!(got, op, "{detail}");
                assert!(detail.contains(needle), "{op}: {detail:?} lacks {needle:?}");
            }
            other => panic!("{op}: expected OutOfRange containing {needle:?}, got {other:?}"),
        }
    }

    #[track_caller]
    fn dtype_err<T: std::fmt::Debug>(
        r: Result<T, OjasError>,
        op: &str,
        expected: DType,
        got: DType,
    ) {
        match r {
            Err(OjasError::Dtype {
                op: o,
                expected: e,
                got: g,
            }) => assert_eq!((o, e, g), (op, expected, got)),
            other => panic!("{op}: expected Dtype {expected:?}/{got:?}, got {other:?}"),
        }
    }

    // ---- per-operand rules, shared by every validator -------------------

    #[test]
    fn every_operand_refuses_a_zero_extent_and_a_wrong_dtype() {
        let op = "silu_backward";
        let ok = f(&[2, 3]);
        shape_err(silu_backward_dims(&f(&[2, 0]), &ok), op, "empty tensor");
        shape_err(silu_backward_dims(&ok, &f(&[0])), op, "empty tensor");
        dtype_err(
            silu_backward_dims(&u(&[2, 3]), &ok),
            op,
            DType::F32,
            DType::U32,
        );
        dtype_err(
            silu_backward_dims(&ok, &of(DType::Bf16, &[2, 3])),
            op,
            DType::F32,
            DType::Bf16,
        );
        // The first operand's refusal wins, and dtype precedes emptiness.
        dtype_err(
            silu_backward_dims(&u(&[0]), &f(&[0])),
            op,
            DType::F32,
            DType::U32,
        );
        shape_err(silu_backward_dims(&f(&[0]), &u(&[2])), op, "empty tensor");
    }

    #[test]
    fn an_operand_whose_element_or_byte_count_overflows_is_out_of_range() {
        let big = broadcast(DType::F32, &[1 << 40, 1 << 40]);
        range_err(silu_forward_dims(&big), "silu_forward", "shape product");
        // Elements fit, bytes do not.
        let bytes = broadcast(DType::F32, &[usize::MAX / 2]);
        range_err(silu_forward_dims(&bytes), "silu_forward", "byte length");
        // A zero extent is refused as empty before any product is formed.
        let zero = broadcast(DType::F32, &[0, usize::MAX, usize::MAX]);
        shape_err(silu_forward_dims(&zero), "silu_forward", "empty tensor");
    }

    // ---- embedding ------------------------------------------------------

    #[test]
    fn embedding_forward_rules() {
        let op = "embedding_forward";
        let table = f(&[10, 4]);
        let d = embedding_forward_dims(&table, &u(&[2, 3])).unwrap();
        assert_eq!(
            d,
            EmbeddingDims {
                vocab: 10,
                dim: 4,
                tokens: 6,
                out_shape: vec![2, 3, 4]
            }
        );
        // Rank-0 ids give a [dim] row.
        let d = embedding_forward_dims(&table, &u(&[])).unwrap();
        assert_eq!((d.tokens, d.out_shape), (1, vec![4]));
        for bad in [&[10][..], &[], &[2, 5, 4]] {
            shape_err(
                embedding_forward_dims(&f(bad), &u(&[3])),
                op,
                "embedding table rank",
            );
        }
        dtype_err(
            embedding_forward_dims(&u(&[10, 4]), &u(&[3])),
            op,
            DType::F32,
            DType::U32,
        );
        dtype_err(
            embedding_forward_dims(&table, &f(&[3])),
            op,
            DType::U32,
            DType::F32,
        );
        // Table before ids: both wrong reports the table.
        dtype_err(
            embedding_forward_dims(&u(&[10, 4]), &f(&[3])),
            op,
            DType::F32,
            DType::U32,
        );
        shape_err(
            embedding_forward_dims(&f(&[10, 0]), &u(&[3])),
            op,
            "empty tensor",
        );
        shape_err(embedding_forward_dims(&table, &u(&[0])), op, "empty tensor");
        // tokens * dim overflows although each operand fits.
        let wide = broadcast(DType::F32, &[2, 1 << 40]);
        let many = broadcast(DType::U32, &[1 << 40]);
        range_err(embedding_forward_dims(&wide, &many), op, "shape product");
    }

    #[test]
    fn embedding_backward_rules() {
        let op = "embedding_backward";
        let table = f(&[10, 4]);
        let ids = u(&[2, 3]);
        let d = embedding_backward_dims(&table, &ids, &f(&[2, 3, 4])).unwrap();
        assert_eq!(
            (d.vocab, d.dim, d.tokens, d.out_shape),
            (10, 4, 6, vec![10, 4])
        );
        for bad in [&[2, 3][..], &[2, 3, 5], &[6, 4], &[3, 2, 4], &[1, 2, 3, 4]] {
            shape_err(
                embedding_backward_dims(&table, &ids, &f(bad)),
                op,
                "embedding grad shape",
            );
        }
        // The table's rank is checked before the gradient's shape.
        shape_err(
            embedding_backward_dims(&f(&[40]), &ids, &f(&[9])),
            op,
            "embedding table rank",
        );
        dtype_err(
            embedding_backward_dims(&table, &ids, &u(&[2, 3, 4])),
            op,
            DType::F32,
            DType::U32,
        );
        dtype_err(
            embedding_backward_dims(&table, &f(&[2, 3]), &f(&[2, 3, 4])),
            op,
            DType::U32,
            DType::F32,
        );
        dtype_err(
            embedding_backward_dims(&u(&[10, 4]), &ids, &f(&[2, 3, 4])),
            op,
            DType::F32,
            DType::U32,
        );
        shape_err(
            embedding_backward_dims(&table, &ids, &f(&[2, 0, 4])),
            op,
            "empty tensor",
        );
    }

    // ---- linear ---------------------------------------------------------

    #[test]
    fn linear_forward_rules() {
        let op = "linear_forward";
        let w = f(&[5, 4]);
        let d = linear_forward_dims(&f(&[2, 3, 4]), &w).unwrap();
        assert_eq!(
            d,
            LinearDims {
                rows: 6,
                in_features: 4,
                out_features: 5,
                out_shape: vec![2, 3, 5]
            }
        );
        let d = linear_forward_dims(&f(&[4]), &w).unwrap();
        assert_eq!((d.rows, d.out_shape), (1, vec![5]));
        shape_err(
            linear_forward_dims(&f(&[]), &w),
            op,
            "input rank 0 has no in-features",
        );
        for bad in [&[20][..], &[], &[1, 5, 4]] {
            shape_err(linear_forward_dims(&f(&[2, 4]), &f(bad)), op, "weight rank");
        }
        shape_err(
            linear_forward_dims(&f(&[2, 3]), &w),
            op,
            "input in-features 3 != weight in-features 4",
        );
        // The weight rank is checked before the input rank.
        shape_err(linear_forward_dims(&f(&[]), &f(&[4])), op, "weight rank");
        dtype_err(
            linear_forward_dims(&f(&[2, 4]), &of(DType::F16, &[5, 4])),
            op,
            DType::F32,
            DType::F16,
        );
        shape_err(
            linear_forward_dims(&f(&[2, 4]), &f(&[0, 4])),
            op,
            "empty tensor",
        );
    }

    #[test]
    fn linear_backward_rules() {
        let op = "linear_backward";
        let (x, w) = (f(&[2, 3, 4]), f(&[5, 4]));
        let d = linear_backward_dims(&x, &w, &f(&[2, 3, 5])).unwrap();
        assert_eq!((d.rows, d.out_shape), (6, vec![2, 3, 5]));
        for bad in [&[2, 3, 4][..], &[6, 5], &[2, 3], &[3, 2, 5], &[1, 2, 3, 5]] {
            shape_err(linear_backward_dims(&x, &w, &f(bad)), op, "grad shape");
        }
        // Forward rules come before the gradient check.
        shape_err(
            linear_backward_dims(&f(&[2, 3]), &w, &f(&[9])),
            op,
            "in-features",
        );
        dtype_err(
            linear_backward_dims(&x, &w, &u(&[2, 3, 5])),
            op,
            DType::F32,
            DType::U32,
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn linear_output_overflow_is_out_of_range_for_contiguous_operands() {
        // rows * out = 2^64 elements; each operand is 2^34 bytes.
        let (x, w) = (
            phantom(DType::F32, &[1 << 32, 1]),
            phantom(DType::F32, &[1 << 32, 1]),
        );
        assert!(x.is_contiguous().unwrap());
        range_err(
            linear_forward_dims(&x, &w),
            "linear_forward",
            "shape product",
        );
        // rows * out = 2^62 elements fit, their bytes do not.
        let (x, w) = (
            phantom(DType::F32, &[1 << 31, 1]),
            phantom(DType::F32, &[1 << 31, 1]),
        );
        range_err(linear_forward_dims(&x, &w), "linear_forward", "byte length");
        // The same sizes one power lower pass.
        let (x, w) = (
            phantom(DType::F32, &[1 << 30, 1]),
            phantom(DType::F32, &[1 << 30, 1]),
        );
        assert_eq!(
            linear_forward_dims(&x, &w).unwrap().out_shape,
            vec![1 << 30, 1 << 30]
        );
    }

    // ---- rms_norm and rms_qk_norm ---------------------------------------

    #[test]
    fn rms_norm_forward_rules() {
        let op = "rms_norm_forward";
        let d = rms_norm_forward_dims(&f(&[2, 3, 8]), &f(&[8]), 1e-6).unwrap();
        assert_eq!(d, RmsDims { rows: 6, dim: 8 });
        assert_eq!(
            rms_norm_forward_dims(&f(&[8]), &f(&[8]), 0.0).unwrap(),
            RmsDims { rows: 1, dim: 8 }
        );
        shape_err(
            rms_norm_forward_dims(&f(&[]), &f(&[1]), 1e-6),
            op,
            "rms_norm input rank 0",
        );
        for bad in [&[7][..], &[9], &[1, 8], &[]] {
            shape_err(
                rms_norm_forward_dims(&f(&[2, 8]), &f(bad), 1e-6),
                op,
                "rms weight",
            );
        }
        for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                rms_norm_forward_dims(&f(&[2, 8]), &f(&[8]), eps),
                Err(OjasError::NonFinite {
                    op: "rms_norm_forward"
                })
            ));
            // eps is checked before the shapes, after the operands.
            assert!(matches!(
                rms_norm_forward_dims(&f(&[]), &f(&[3]), eps),
                Err(OjasError::NonFinite { .. })
            ));
            dtype_err(
                rms_norm_forward_dims(&u(&[2, 8]), &f(&[8]), eps),
                op,
                DType::F32,
                DType::U32,
            );
        }
        dtype_err(
            rms_norm_forward_dims(&f(&[2, 8]), &u(&[8]), 1e-6),
            op,
            DType::F32,
            DType::U32,
        );
    }

    #[test]
    fn rms_norm_backward_rules() {
        let op = "rms_norm_backward";
        let (x, w) = (f(&[2, 8]), f(&[8]));
        assert_eq!(
            rms_norm_backward_dims(&x, &w, &f(&[2, 8]), 1e-6).unwrap(),
            RmsDims { rows: 2, dim: 8 }
        );
        shape_err(
            rms_norm_backward_dims(&x, &w, &f(&[8, 2]), 1e-6),
            op,
            "does not match",
        );
        // grad_output is checked before eps and before the weight.
        shape_err(
            rms_norm_backward_dims(&x, &f(&[3]), &f(&[16]), f32::NAN),
            op,
            "does not match",
        );
        assert!(matches!(
            rms_norm_backward_dims(&x, &f(&[3]), &f(&[2, 8]), f32::NAN),
            Err(OjasError::NonFinite {
                op: "rms_norm_backward"
            })
        ));
        shape_err(
            rms_norm_backward_dims(&x, &f(&[3]), &f(&[2, 8]), 1e-6),
            op,
            "rms weight",
        );
        dtype_err(
            rms_norm_backward_dims(&x, &w, &u(&[2, 8]), 1e-6),
            op,
            DType::F32,
            DType::U32,
        );
    }

    #[test]
    fn rms_qk_norm_checks_both_pairs_under_the_rms_norm_name() {
        let (q, k, w) = (f(&[2, 3, 8]), f(&[2, 1, 8]), f(&[8]));
        let (qd, kd) = rms_qk_norm_forward_dims(&q, &k, &w, &w, 1e-6).unwrap();
        assert_eq!(
            (qd, kd),
            (RmsDims { rows: 6, dim: 8 }, RmsDims { rows: 2, dim: 8 })
        );
        // A malformed k is refused even though q is fine, before either runs.
        shape_err(
            rms_qk_norm_forward_dims(&q, &k, &w, &f(&[4]), 1e-6),
            "rms_norm_forward",
            "rms weight",
        );
        // q's pair is checked first.
        dtype_err(
            rms_qk_norm_forward_dims(&u(&[2]), &f(&[0]), &w, &w, 1e-6),
            "rms_norm_forward",
            DType::F32,
            DType::U32,
        );
        let (gq, gk) = (f(&[2, 3, 8]), f(&[2, 1, 8]));
        assert!(rms_qk_norm_backward_dims(&q, &k, &w, &w, &gq, &gk, 1e-6).is_ok());
        shape_err(
            rms_qk_norm_backward_dims(&q, &k, &w, &w, &gq, &gq, 1e-6),
            "rms_norm_backward",
            "does not match",
        );
    }

    // ---- rope -----------------------------------------------------------

    #[test]
    fn rope_rules() {
        type Rope = fn(&Tensor, &Tensor, &Tensor) -> Result<RopeDims, OjasError>;
        let checks: [(&str, Rope); 2] = [
            ("rope_half_split_forward", rope_half_split_forward_dims),
            ("rope_half_split_backward", rope_half_split_backward_dims),
        ];
        for (op, check) in checks {
            let x = f(&[2, 5, 3, 8]);
            let same_tables = f(&[2, 5, 3, 8]);
            let d = check(&x, &same_tables, &same_tables).unwrap();
            assert_eq!(
                d,
                RopeDims {
                    rows: 30,
                    dim: 8,
                    layout: RopeLayout::Same
                }
            );
            let td = f(&[5, 8]);
            let d = check(&x, &td, &td).unwrap();
            assert_eq!(d.layout, RopeLayout::TimeDim { time: 5, heads: 3 });
            // A rank-2 x with [T, D] tables is the Same layout.
            assert_eq!(check(&td, &td, &td).unwrap().layout, RopeLayout::Same);
            shape_err(check(&x, &td, &f(&[5, 6])), op, "cos shape");
            shape_err(check(&f(&[]), &f(&[]), &f(&[])), op, "rope input rank 0");
            let odd = f(&[3, 5]);
            shape_err(check(&odd, &odd, &odd), op, "rope last dim 5 is odd");
            for bad in [&[4, 8][..], &[5, 6], &[1, 5, 8], &[8], &[2, 5, 3, 6]] {
                let t = f(bad);
                shape_err(check(&x, &t, &t), op, "does not broadcast");
            }
            // [T, D] tables need a rank-4 x.
            let x3 = f(&[5, 3, 8]);
            shape_err(check(&x3, &td, &td), op, "does not broadcast");
            dtype_err(check(&x, &td, &u(&[5, 8])), op, DType::F32, DType::U32);
            shape_err(check(&x, &f(&[0, 8]), &td), op, "empty tensor");
        }
    }

    // ---- causal sdpa ----------------------------------------------------

    #[test]
    fn causal_sdpa_forward_rules() {
        let op = "causal_sdpa_forward";
        let q = f(&[2, 3, 5, 4]);
        let d = causal_sdpa_forward_dims(&q, &q, &q, None).unwrap();
        assert_eq!(
            d,
            SdpaDims {
                batch: 2,
                heads: 3,
                kv_heads: 3,
                seq: 5,
                head_dim: 4
            }
        );
        for rank in [&[][..], &[4], &[5, 4], &[3, 5, 4], &[1, 2, 3, 5, 4]] {
            let t = f(rank);
            shape_err(
                causal_sdpa_forward_dims(&t, &t, &t, None),
                op,
                "sdpa query rank",
            );
        }
        // A mismatch on each axis, in k and in v.
        for bad in [[1, 3, 5, 4], [2, 1, 5, 4], [2, 3, 6, 4], [2, 3, 5, 2]] {
            let t = f(&bad);
            shape_err(causal_sdpa_forward_dims(&q, &t, &q, None), op, "differ");
            shape_err(causal_sdpa_forward_dims(&q, &q, &t, None), op, "differ");
        }
        // Legal grouped-query: both K and V carry the smaller head count.
        let kv = f(&[2, 1, 5, 4]);
        let gqa = causal_sdpa_forward_dims(&q, &kv, &kv, None).unwrap();
        assert_eq!(gqa.heads, 3);
        assert_eq!(gqa.kv_heads, 1);
        // Not a divisor, and more KV heads than query heads. A zero KV-head
        // count is an empty tensor, refused before the grouping check.
        for bad in [[2, 2, 5, 4], [2, 6, 5, 4]] {
            let t = f(&bad);
            shape_err(causal_sdpa_forward_dims(&q, &t, &t, None), op, "differ");
        }
        dtype_err(
            causal_sdpa_forward_dims(&q, &q, &u(&[2, 3, 5, 4]), None),
            op,
            DType::F32,
            DType::U32,
        );
        // A wide head dimension is not a shape error; device limits are the backend's.
        let wide = f(&[1, 1, 1, 4096]);
        assert!(causal_sdpa_forward_dims(&wide, &wide, &wide, None).is_ok());
    }

    #[test]
    fn causal_sdpa_window_rules() {
        let q = f(&[2, 3, 5, 4]);
        for w in [Some(1), Some(4), Some(5), Some(usize::MAX), None] {
            assert!(causal_sdpa_forward_dims(&q, &q, &q, w).is_ok(), "{w:?}");
        }
        shape_err(
            causal_sdpa_forward_dims(&q, &q, &q, Some(0)),
            "causal_sdpa_forward",
            "sdpa window must be at least 1",
        );
        let lse = f(&[2, 3, 5]);
        shape_err(
            causal_sdpa_backward_dims(&q, &q, &q, &q, &lse, &q, Some(0)),
            "causal_sdpa_backward",
            "sdpa window must be at least 1",
        );
    }

    #[test]
    fn causal_sdpa_backward_rules() {
        let op = "causal_sdpa_backward";
        let q = f(&[2, 3, 5, 4]);
        let lse = f(&[2, 3, 5]);
        let call = |q: &Tensor, k: &Tensor, v: &Tensor, o: &Tensor, l: &Tensor, g: &Tensor| {
            causal_sdpa_backward_dims(q, k, v, o, l, g, None)
        };
        assert!(call(&q, &q, &q, &q, &lse, &f(&[2, 3, 5, 4])).is_ok());
        shape_err(
            call(&q, &q, &q, &q, &lse, &f(&[2, 3, 5, 2])),
            op,
            "sdpa grad shape",
        );
        // The gradient is checked before the rank, the output and the lse.
        let r3 = f(&[3, 5, 4]);
        shape_err(
            call(&r3, &r3, &r3, &f(&[1]), &f(&[1]), &f(&[4])),
            op,
            "sdpa grad shape",
        );
        shape_err(
            call(&r3, &r3, &r3, &r3, &f(&[3, 5]), &r3),
            op,
            "sdpa query rank",
        );
        // The output is the query's shape, the lse drops its last axis.
        shape_err(
            call(&q, &q, &q, &f(&[2, 3, 5, 2]), &lse, &q),
            op,
            "sdpa output shape",
        );
        for bad in [&[2, 3, 5, 1][..], &[2, 3, 4], &[2, 3], &[3, 5]] {
            shape_err(call(&q, &q, &q, &q, &f(bad), &q), op, "sdpa lse shape");
        }
        dtype_err(
            call(&q, &q, &q, &q, &u(&[2, 3, 5]), &q),
            op,
            DType::F32,
            DType::U32,
        );
        shape_err(call(&q, &f(&[2, 3, 6, 4]), &q, &q, &lse, &q), op, "differ");
        let kv = f(&[2, 1, 5, 4]);
        assert!(call(&q, &kv, &kv, &q, &lse, &q).is_ok());
        shape_err(
            call(&q, &f(&[2, 2, 5, 4]), &f(&[2, 2, 5, 4]), &q, &lse, &q),
            op,
            "differ",
        );
    }

    // ---- per-head gate --------------------------------------------------

    #[test]
    fn gate_forward_rules() {
        let op = "per_head_sigmoid_gate_forward";
        let (x, w, b, a) = (f(&[2, 5, 16]), f(&[4, 16]), f(&[4]), f(&[2, 5, 4, 8]));
        let d = per_head_sigmoid_gate_forward_dims(&x, &w, &b, &a).unwrap();
        assert_eq!(
            d,
            GateDims {
                rows: 10,
                d_model: 16,
                heads: 4,
                head_dim: 8
            }
        );
        let d = per_head_sigmoid_gate_forward_dims(&f(&[16]), &w, &b, &f(&[4, 8])).unwrap();
        assert_eq!(d.rows, 1);
        let call = |x: &Tensor, w: &Tensor, b: &Tensor, a: &Tensor| {
            per_head_sigmoid_gate_forward_dims(x, w, b, a)
        };
        for bad in [&[64][..], &[], &[1, 4, 16]] {
            shape_err(call(&x, &f(bad), &b, &a), op, "gate weight must be");
        }
        shape_err(call(&f(&[]), &w, &b, &f(&[4, 8])), op, "gate input rank 0");
        shape_err(
            call(&f(&[2, 5, 15]), &w, &b, &a),
            op,
            "gate weight in 16 != input dim 15",
        );
        for bad in [&[3][..], &[5], &[1, 4], &[]] {
            shape_err(call(&x, &w, &f(bad), &a), op, "gate bias");
        }
        for bad in [&[2, 5, 4][..], &[2, 5, 4, 8, 1], &[10, 4, 8]] {
            shape_err(call(&x, &w, &b, &f(bad)), op, "gate attn rank");
        }
        shape_err(
            call(&x, &w, &b, &f(&[2, 5, 3, 8])),
            op,
            "gate attn head axis",
        );
        shape_err(call(&x, &w, &b, &f(&[5, 2, 4, 8])), op, "gate attn prefix");
        shape_err(call(&x, &w, &b, &f(&[2, 6, 4, 8])), op, "gate attn prefix");
        dtype_err(call(&x, &w, &u(&[4]), &a), op, DType::F32, DType::U32);
    }

    #[test]
    fn gate_backward_rules() {
        let op = "per_head_sigmoid_gate_backward";
        let (x, w, b, a) = (f(&[2, 5, 16]), f(&[4, 16]), f(&[4]), f(&[2, 5, 4, 8]));
        assert!(per_head_sigmoid_gate_backward_dims(&x, &w, &b, &a, &f(&[2, 5, 4, 8])).is_ok());
        shape_err(
            per_head_sigmoid_gate_backward_dims(&x, &w, &b, &a, &f(&[2, 5, 4, 7])),
            op,
            "does not match",
        );
        // The gradient is checked before the layout.
        shape_err(
            per_head_sigmoid_gate_backward_dims(&x, &f(&[64]), &b, &a, &f(&[1])),
            op,
            "does not match",
        );
        shape_err(
            per_head_sigmoid_gate_backward_dims(&x, &f(&[64]), &b, &a, &a),
            op,
            "gate weight must be",
        );
    }

    // ---- value residual -------------------------------------------------

    #[test]
    fn value_residual_rules() {
        let (v, s) = (f(&[2, 3]), f(&[]));
        assert_eq!(value_residual_blend_forward_dims(&v, &v, &s).unwrap(), 6);
        for lam in [&[1][..], &[1, 1, 1]] {
            assert_eq!(
                value_residual_blend_forward_dims(&v, &v, &f(lam)).unwrap(),
                6
            );
        }
        let op = "value_residual_blend_forward";
        for lam in [&[2][..], &[1, 2], &[3, 1]] {
            shape_err(
                value_residual_blend_forward_dims(&v, &v, &f(lam)),
                op,
                "expected a scalar tensor",
            );
        }
        shape_err(
            value_residual_blend_forward_dims(&v, &f(&[3, 2]), &s),
            op,
            "does not match",
        );
        // lambda is checked before the value shapes.
        shape_err(
            value_residual_blend_forward_dims(&v, &f(&[6]), &f(&[2])),
            op,
            "expected a scalar",
        );
        dtype_err(
            value_residual_blend_forward_dims(&v, &v, &u(&[])),
            op,
            DType::F32,
            DType::U32,
        );
        shape_err(
            value_residual_blend_forward_dims(&v, &v, &f(&[0])),
            op,
            "empty tensor",
        );

        let op = "value_residual_blend_backward";
        assert_eq!(
            value_residual_blend_backward_dims(&v, &v, &s, &v).unwrap(),
            6
        );
        shape_err(
            value_residual_blend_backward_dims(&v, &v, &f(&[2]), &v),
            op,
            "lambda must be a scalar",
        );
        shape_err(
            value_residual_blend_backward_dims(&v, &f(&[6]), &s, &v),
            op,
            "does not match",
        );
        shape_err(
            value_residual_blend_backward_dims(&v, &v, &s, &f(&[2, 4])),
            op,
            "does not match",
        );
    }

    // ---- pointwise ------------------------------------------------------

    #[test]
    fn pointwise_rules() {
        type Two = fn(&Tensor, &Tensor) -> Result<usize, OjasError>;
        type Three = fn(&Tensor, &Tensor, &Tensor) -> Result<usize, OjasError>;
        let (a, s) = (f(&[2, 3]), f(&[]));
        assert_eq!(silu_forward_dims(&a).unwrap(), 6);
        assert_eq!(silu_forward_dims(&s).unwrap(), 1);
        let twos: [(&str, Two); 3] = [
            ("silu_backward", silu_backward_dims),
            ("mul_forward", mul_forward_dims),
            ("residual_add_forward", residual_add_forward_dims),
        ];
        for (op, check) in twos {
            assert_eq!(check(&a, &a).unwrap(), 6);
            assert_eq!(check(&s, &s).unwrap(), 1);
            for bad in [&[3, 2][..], &[6], &[2, 3, 1], &[]] {
                shape_err(check(&a, &f(bad)), op, "does not match");
            }
            dtype_err(check(&a, &u(&[2, 3])), op, DType::F32, DType::U32);
        }
        let threes: [(&str, Three); 2] = [
            ("mul_backward", mul_backward_dims),
            ("residual_add_backward", residual_add_backward_dims),
        ];
        for (op, check) in threes {
            assert_eq!(check(&a, &a, &a).unwrap(), 6);
            shape_err(check(&a, &f(&[3, 2]), &a), op, "does not match");
            shape_err(check(&a, &a, &f(&[6])), op, "does not match");
            // The second operand's mismatch is reported before the third's.
            shape_err(check(&a, &f(&[3, 2]), &f(&[6])), op, "[3, 2]");
            dtype_err(check(&a, &a, &u(&[2, 3])), op, DType::F32, DType::U32);
        }
    }

    // ---- cross-entropy --------------------------------------------------

    #[test]
    fn cross_entropy_rules() {
        type Ce = fn(&Tensor, &Tensor) -> Result<CeDims, OjasError>;
        let checks: [(&str, Ce); 2] = [
            (
                "cross_entropy_mean_forward",
                cross_entropy_mean_forward_dims,
            ),
            (
                "cross_entropy_mean_backward",
                cross_entropy_mean_backward_dims,
            ),
        ];
        for (op, check) in checks {
            assert_eq!(
                check(&f(&[2, 3, 7]), &u(&[2, 3])).unwrap(),
                CeDims { rows: 6, vocab: 7 }
            );
            // Rank-1 logits take rank-0 targets.
            assert_eq!(
                check(&f(&[7]), &u(&[])).unwrap(),
                CeDims { rows: 1, vocab: 7 }
            );
            shape_err(check(&f(&[]), &u(&[])), op, "cross-entropy logits rank 0");
            for bad in [&[6][..], &[3, 2], &[2, 3, 7], &[2], &[]] {
                shape_err(check(&f(&[2, 3, 7]), &u(bad)), op, "logits prefix");
            }
            dtype_err(check(&f(&[2, 7]), &f(&[2])), op, DType::U32, DType::F32);
            dtype_err(check(&u(&[2, 7]), &u(&[2])), op, DType::F32, DType::U32);
            // Logits before targets: both wrong reports the logits.
            dtype_err(check(&u(&[2, 7]), &f(&[2])), op, DType::F32, DType::U32);
            shape_err(check(&f(&[2, 7]), &u(&[0])), op, "empty tensor");
        }
    }

    // ---- optimizer and clip ---------------------------------------------

    #[test]
    fn clip_grad_norm_rules() {
        let op = "clip_grad_norm";
        assert_eq!(
            clip_grad_norm_dims(&[f(&[2, 3]), f(&[]), f(&[4])]).unwrap(),
            11
        );
        shape_err(clip_grad_norm_dims(&[]), op, "empty tensor");
        shape_err(clip_grad_norm_dims(&[f(&[2]), f(&[0])]), op, "empty tensor");
        dtype_err(
            clip_grad_norm_dims(&[f(&[2]), u(&[2])]),
            op,
            DType::F32,
            DType::U32,
        );
        // Each fits. Three eighths of usize::MAX elements fit, their bytes
        // do not; five quarters do not fit as elements.
        let eighth = broadcast(DType::F32, &[usize::MAX / 8]);
        let three = [eighth.clone(), eighth.clone(), eighth];
        range_err(
            clip_grad_norm_dims(&three),
            op,
            "total gradient byte length",
        );
        let quarter = broadcast(DType::F32, &[usize::MAX / 4]);
        let five: Vec<Tensor> = (0..5).map(|_| quarter.clone()).collect();
        range_err(clip_grad_norm_dims(&five), op, "total gradient length");
    }

    #[test]
    fn adamw_rules() {
        let op = "adamw_step";
        let p = f(&[3, 4]);
        assert_eq!(adamw_step_dims(&p, &p, &p, &p).unwrap(), 12);
        assert_eq!(
            adamw_step_dims(&f(&[]), &f(&[]), &f(&[]), &f(&[])).unwrap(),
            1
        );
        let bad = f(&[4, 3]);
        shape_err(adamw_step_dims(&p, &bad, &p, &p), op, "does not match");
        shape_err(adamw_step_dims(&p, &p, &bad, &p), op, "does not match");
        shape_err(adamw_step_dims(&p, &p, &p, &bad), op, "does not match");
        // Param before grad.
        dtype_err(
            adamw_step_dims(&u(&[3, 4]), &u(&[3, 4]), &p, &p),
            op,
            DType::F32,
            DType::U32,
        );
        dtype_err(
            adamw_step_dims(&p, &p, &p, &u(&[3, 4])),
            op,
            DType::F32,
            DType::U32,
        );
    }

    #[test]
    fn muon_rules() {
        let op = "muon_ns5_step";
        let p = f(&[3, 4]);
        assert_eq!(
            muon_ns5_step_dims(&p, &p, &p).unwrap(),
            MuonDims { rows: 3, cols: 4 }
        );
        for rank in [&[][..], &[12], &[1, 3, 4]] {
            let t = f(rank);
            shape_err(
                muon_ns5_step_dims(&t, &t, &t),
                op,
                "muon parameter must be a matrix",
            );
        }
        shape_err(
            muon_ns5_step_dims(&p, &f(&[4, 3]), &p),
            op,
            "does not match",
        );
        shape_err(
            muon_ns5_step_dims(&p, &p, &f(&[3, 5])),
            op,
            "does not match",
        );
        dtype_err(
            muon_ns5_step_dims(&p, &u(&[3, 4]), &p),
            op,
            DType::F32,
            DType::U32,
        );
    }

    // ---- op names -------------------------------------------------------

    /// Every validator names the trait method the CPU backend reports, so a
    /// backend's existing error text does not change when it adopts these.
    #[test]
    fn every_validator_names_its_op() {
        fn unit<T>(r: Result<T, OjasError>) -> Result<(), OjasError> {
            r.map(|_| ())
        }
        // Every op takes F32 in its first position, so a U32 tensor in every
        // position is a Dtype refusal naming the op.
        let x = &u(&[2]);
        let cases = [
            ("embedding_forward", unit(embedding_forward_dims(x, x))),
            ("embedding_backward", unit(embedding_backward_dims(x, x, x))),
            ("linear_forward", unit(linear_forward_dims(x, x))),
            ("linear_backward", unit(linear_backward_dims(x, x, x))),
            ("rms_norm_forward", unit(rms_norm_forward_dims(x, x, 1e-6))),
            (
                "rms_norm_backward",
                unit(rms_norm_backward_dims(x, x, x, 1e-6)),
            ),
            (
                "rms_norm_forward",
                unit(rms_qk_norm_forward_dims(x, x, x, x, 1e-6)),
            ),
            (
                "rms_norm_backward",
                unit(rms_qk_norm_backward_dims(x, x, x, x, x, x, 1e-6)),
            ),
            (
                "rope_half_split_forward",
                unit(rope_half_split_forward_dims(x, x, x)),
            ),
            (
                "rope_half_split_backward",
                unit(rope_half_split_backward_dims(x, x, x)),
            ),
            (
                "causal_sdpa_forward",
                unit(causal_sdpa_forward_dims(x, x, x, None)),
            ),
            (
                "causal_sdpa_backward",
                unit(causal_sdpa_backward_dims(x, x, x, x, x, x, None)),
            ),
            (
                "per_head_sigmoid_gate_forward",
                unit(per_head_sigmoid_gate_forward_dims(x, x, x, x)),
            ),
            (
                "per_head_sigmoid_gate_backward",
                unit(per_head_sigmoid_gate_backward_dims(x, x, x, x, x)),
            ),
            (
                "value_residual_blend_forward",
                unit(value_residual_blend_forward_dims(x, x, x)),
            ),
            (
                "value_residual_blend_backward",
                unit(value_residual_blend_backward_dims(x, x, x, x)),
            ),
            ("silu_forward", unit(silu_forward_dims(x))),
            ("silu_backward", unit(silu_backward_dims(x, x))),
            ("mul_forward", unit(mul_forward_dims(x, x))),
            ("mul_backward", unit(mul_backward_dims(x, x, x))),
            (
                "residual_add_forward",
                unit(residual_add_forward_dims(x, x)),
            ),
            (
                "residual_add_backward",
                unit(residual_add_backward_dims(x, x, x)),
            ),
            (
                "cross_entropy_mean_forward",
                unit(cross_entropy_mean_forward_dims(x, x)),
            ),
            (
                "cross_entropy_mean_backward",
                unit(cross_entropy_mean_backward_dims(x, x)),
            ),
            (
                "clip_grad_norm",
                unit(clip_grad_norm_dims(std::slice::from_ref(x))),
            ),
            ("adamw_step", unit(adamw_step_dims(x, x, x, x))),
            ("muon_ns5_step", unit(muon_ns5_step_dims(x, x, x))),
            (
                "chunked_gdn_forward",
                unit(chunked_gdn_forward_dims(gdn_all(x))),
            ),
            (
                "chunked_gdn_backward",
                unit(chunked_gdn_backward_dims(gdn_all(x), x, x, Some(x))),
            ),
        ];
        for (want, got) in cases {
            dtype_err(got, want, DType::F32, DType::U32);
        }
        // The empty-list refusal of clip carries the name too.
        shape_err(clip_grad_norm_dims(&[]), "clip_grad_norm", "empty tensor");
    }

    /// Every operand position of every validator, with the others valid:
    /// the other dtype is a Dtype refusal naming the op and that slot's
    /// dtype, and a zero extent is an `"empty tensor"` Shape refusal.
    #[test]
    fn every_operand_position_refuses_a_wrong_dtype_and_a_zero_extent() {
        type Call = fn(&[Tensor]) -> Result<(), OjasError>;
        fn unit<T>(r: Result<T, OjasError>) -> Result<(), OjasError> {
            r.map(|_| ())
        }
        let cases: Vec<(&str, Call, Vec<Tensor>)> = vec![
            (
                "embedding_forward",
                |t| unit(embedding_forward_dims(&t[0], &t[1])),
                vec![f(&[10, 4]), u(&[2, 3])],
            ),
            (
                "embedding_backward",
                |t| unit(embedding_backward_dims(&t[0], &t[1], &t[2])),
                vec![f(&[10, 4]), u(&[2, 3]), f(&[2, 3, 4])],
            ),
            (
                "linear_forward",
                |t| unit(linear_forward_dims(&t[0], &t[1])),
                vec![f(&[2, 4]), f(&[5, 4])],
            ),
            (
                "linear_backward",
                |t| unit(linear_backward_dims(&t[0], &t[1], &t[2])),
                vec![f(&[2, 4]), f(&[5, 4]), f(&[2, 5])],
            ),
            (
                "rms_norm_forward",
                |t| unit(rms_norm_forward_dims(&t[0], &t[1], 1e-6)),
                vec![f(&[2, 8]), f(&[8])],
            ),
            (
                "rms_norm_backward",
                |t| unit(rms_norm_backward_dims(&t[0], &t[1], &t[2], 1e-6)),
                vec![f(&[2, 8]), f(&[8]), f(&[2, 8])],
            ),
            (
                "rms_norm_forward",
                |t| unit(rms_qk_norm_forward_dims(&t[0], &t[1], &t[2], &t[3], 1e-6)),
                vec![f(&[2, 8]), f(&[3, 8]), f(&[8]), f(&[8])],
            ),
            (
                "rms_norm_backward",
                |t| {
                    unit(rms_qk_norm_backward_dims(
                        &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], 1e-6,
                    ))
                },
                vec![
                    f(&[2, 8]),
                    f(&[3, 8]),
                    f(&[8]),
                    f(&[8]),
                    f(&[2, 8]),
                    f(&[3, 8]),
                ],
            ),
            (
                "rope_half_split_forward",
                |t| unit(rope_half_split_forward_dims(&t[0], &t[1], &t[2])),
                vec![f(&[2, 8]), f(&[2, 8]), f(&[2, 8])],
            ),
            (
                "rope_half_split_backward",
                |t| unit(rope_half_split_backward_dims(&t[0], &t[1], &t[2])),
                vec![f(&[2, 8]), f(&[2, 8]), f(&[2, 8])],
            ),
            (
                "causal_sdpa_forward",
                |t| unit(causal_sdpa_forward_dims(&t[0], &t[1], &t[2], None)),
                vec![f(&[1, 2, 3, 4]); 3],
            ),
            (
                "causal_sdpa_backward",
                |t| {
                    unit(causal_sdpa_backward_dims(
                        &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], None,
                    ))
                },
                vec![
                    f(&[1, 2, 3, 4]),
                    f(&[1, 2, 3, 4]),
                    f(&[1, 2, 3, 4]),
                    f(&[1, 2, 3, 4]),
                    f(&[1, 2, 3]),
                    f(&[1, 2, 3, 4]),
                ],
            ),
            (
                "per_head_sigmoid_gate_forward",
                |t| {
                    unit(per_head_sigmoid_gate_forward_dims(
                        &t[0], &t[1], &t[2], &t[3],
                    ))
                },
                vec![f(&[2, 16]), f(&[4, 16]), f(&[4]), f(&[2, 4, 8])],
            ),
            (
                "per_head_sigmoid_gate_backward",
                |t| {
                    unit(per_head_sigmoid_gate_backward_dims(
                        &t[0], &t[1], &t[2], &t[3], &t[4],
                    ))
                },
                vec![
                    f(&[2, 16]),
                    f(&[4, 16]),
                    f(&[4]),
                    f(&[2, 4, 8]),
                    f(&[2, 4, 8]),
                ],
            ),
            (
                "value_residual_blend_forward",
                |t| unit(value_residual_blend_forward_dims(&t[0], &t[1], &t[2])),
                vec![f(&[2, 3]), f(&[2, 3]), f(&[])],
            ),
            (
                "value_residual_blend_backward",
                |t| {
                    unit(value_residual_blend_backward_dims(
                        &t[0], &t[1], &t[2], &t[3],
                    ))
                },
                vec![f(&[2, 3]), f(&[2, 3]), f(&[]), f(&[2, 3])],
            ),
            (
                "silu_forward",
                |t| unit(silu_forward_dims(&t[0])),
                vec![f(&[2, 3])],
            ),
            (
                "silu_backward",
                |t| unit(silu_backward_dims(&t[0], &t[1])),
                vec![f(&[2, 3]); 2],
            ),
            (
                "mul_forward",
                |t| unit(mul_forward_dims(&t[0], &t[1])),
                vec![f(&[2, 3]); 2],
            ),
            (
                "mul_backward",
                |t| unit(mul_backward_dims(&t[0], &t[1], &t[2])),
                vec![f(&[2, 3]); 3],
            ),
            (
                "residual_add_forward",
                |t| unit(residual_add_forward_dims(&t[0], &t[1])),
                vec![f(&[2, 3]); 2],
            ),
            (
                "residual_add_backward",
                |t| unit(residual_add_backward_dims(&t[0], &t[1], &t[2])),
                vec![f(&[2, 3]); 3],
            ),
            (
                "cross_entropy_mean_forward",
                |t| unit(cross_entropy_mean_forward_dims(&t[0], &t[1])),
                vec![f(&[2, 7]), u(&[2])],
            ),
            (
                "cross_entropy_mean_backward",
                |t| unit(cross_entropy_mean_backward_dims(&t[0], &t[1])),
                vec![f(&[2, 7]), u(&[2])],
            ),
            (
                "clip_grad_norm",
                |t| unit(clip_grad_norm_dims(t)),
                vec![f(&[2]), f(&[3, 1]), f(&[])],
            ),
            (
                "adamw_step",
                |t| unit(adamw_step_dims(&t[0], &t[1], &t[2], &t[3])),
                vec![f(&[3, 4]); 4],
            ),
            (
                "muon_ns5_step",
                |t| unit(muon_ns5_step_dims(&t[0], &t[1], &t[2])),
                vec![f(&[3, 4]); 3],
            ),
            (
                "chunked_gdn_forward",
                |t| unit(chunked_gdn_forward_dims(gdn_slots(t))),
                gdn_valid(false),
            ),
            (
                "chunked_gdn_backward",
                |t| {
                    unit(chunked_gdn_backward_dims(
                        gdn_slots(t),
                        &t[6],
                        &t[7],
                        Some(&t[8]),
                    ))
                },
                gdn_valid(true),
            ),
        ];
        let mut positions = 0;
        for (op, call, valid) in &cases {
            assert!(call(valid).is_ok(), "{op}: the valid operands are refused");
            for slot in 0..valid.len() {
                let expected = valid[slot].dtype();
                let other = if expected == DType::F32 {
                    DType::U32
                } else {
                    DType::F32
                };
                let mut operands = valid.clone();
                operands[slot] = of(other, valid[slot].shape());
                dtype_err(call(&operands), op, expected, other);
                operands[slot] = of(expected, &[2, 0]);
                shape_err(call(&operands), op, "empty tensor");
                positions += 1;
            }
        }
        assert_eq!(cases.len(), 29);
        assert_eq!(positions, 98);
    }

    /// Every slot of a GDN call is the same tensor.
    fn gdn_all(x: &Tensor) -> GdnInputs<'_> {
        GdnInputs {
            q: x,
            k: x,
            v: x,
            g: x,
            beta: x,
            initial_state: Some(x),
        }
    }

    /// `q, k, v, g, beta, initial_state` from the first six slots.
    fn gdn_slots(t: &[Tensor]) -> GdnInputs<'_> {
        GdnInputs {
            q: &t[0],
            k: &t[1],
            v: &t[2],
            g: &t[3],
            beta: &t[4],
            initial_state: Some(&t[5]),
        }
    }

    /// B 1, T 65 (two checkpoints), H 2, Dk 4, Dv 3, with an initial state;
    /// the backward adds checkpoints, `grad_output` and `grad_final_state`.
    fn gdn_valid(backward: bool) -> Vec<Tensor> {
        let mut ts = vec![
            f(&[1, 65, 2, 4]),
            f(&[1, 65, 2, 4]),
            f(&[1, 65, 2, 3]),
            f(&[1, 65, 2]),
            f(&[1, 65, 2]),
            f(&[1, 2, 4, 3]),
        ];
        if backward {
            ts.extend([f(&[1, 2, 2, 4, 3]), f(&[1, 65, 2, 3]), f(&[1, 2, 4, 3])]);
        }
        ts
    }

    #[test]
    fn gdn_dims_and_checkpoint_count() {
        let t = gdn_valid(true);
        let dims = chunked_gdn_forward_dims(gdn_slots(&t)).unwrap();
        assert_eq!(
            dims,
            GdnDims {
                batch: 1,
                seq: 65,
                heads: 2,
                key_dim: 4,
                value_dim: 3,
                checkpoints: 2,
            }
        );
        assert_eq!(dims.rows(), 130);
        assert_eq!(dims.checkpoint_shape(), [1, 2, 2, 4, 3]);
        assert_eq!(
            chunked_gdn_backward_dims(gdn_slots(&t), &t[6], &t[7], Some(&t[8])).unwrap(),
            dims
        );
        // No initial state and no final-state gradient.
        let bare = GdnInputs {
            initial_state: None,
            ..gdn_slots(&t)
        };
        assert_eq!(chunked_gdn_forward_dims(bare).unwrap(), dims);
        assert_eq!(
            chunked_gdn_backward_dims(bare, &t[6], &t[7], None).unwrap(),
            dims
        );
        for (seq, nc) in [(1, 1), (63, 1), (64, 1), (65, 2), (128, 2), (129, 3)] {
            let (q, v, gb) = (f(&[1, seq, 1, 2]), f(&[1, seq, 1, 2]), f(&[1, seq, 1]));
            let x = GdnInputs {
                q: &q,
                k: &q,
                v: &v,
                g: &gb,
                beta: &gb,
                initial_state: None,
            };
            assert_eq!(
                chunked_gdn_forward_dims(x).unwrap().checkpoints,
                nc,
                "T {seq}"
            );
        }
    }

    #[test]
    fn gdn_layout_refusals() {
        let op = "chunked_gdn_forward";
        let t = gdn_valid(false);
        let with = |slot: usize, bad: Tensor| {
            let mut ts = t.clone();
            ts[slot] = bad;
            chunked_gdn_forward_dims(gdn_slots(&ts)).map(|_| ())
        };
        shape_err(with(0, f(&[65, 2, 4])), op, "q must be rank 4");
        shape_err(with(1, f(&[1, 65, 2, 5])), op, "gdn k");
        shape_err(with(2, f(&[65, 2, 3])), op, "v must be rank 4");
        shape_err(with(2, f(&[1, 64, 2, 3])), op, "does not share");
        shape_err(with(3, f(&[1, 65, 1])), op, "gdn g");
        shape_err(with(4, f(&[1, 65, 2, 1])), op, "gdn beta");
        shape_err(with(5, f(&[1, 2, 3, 4])), op, "initial state");

        let op = "chunked_gdn_backward";
        let t = gdn_valid(true);
        let back = |slot: usize, bad: Tensor| {
            let mut ts = t.clone();
            ts[slot] = bad;
            chunked_gdn_backward_dims(gdn_slots(&ts), &ts[6], &ts[7], Some(&ts[8])).map(|_| ())
        };
        shape_err(back(6, f(&[1, 2, 1, 4, 3])), op, "checkpoints");
        shape_err(back(7, f(&[1, 65, 2, 4])), op, "does not match");
        shape_err(back(8, f(&[1, 2, 3, 4])), op, "final state grad");
        // The forward's layout is checked before the backward's own operands.
        shape_err(back(0, f(&[65, 2, 4])), op, "q must be rank 4");
    }

    /// The validators read metadata only: a device tensor (whose bytes are
    /// unreadable here) validates exactly as a host tensor of its shape.
    #[test]
    fn device_tensors_validate_by_metadata_alone() {
        let x = phantom(DType::F32, &[2, 3, 4]);
        let w = phantom(DType::F32, &[5, 4]);
        assert_eq!(
            linear_forward_dims(&x, &w).unwrap(),
            linear_forward_dims(&f(&[2, 3, 4]), &f(&[5, 4])).unwrap()
        );
        assert!(x.to_f32_vec().is_err());
    }

    #[test]
    fn linear_ce_dims_checks_operands_and_both_chunk_dimensions() {
        let budget = Budget::new(1 << 14);
        let f = |shape: &[usize]| {
            let n = shape.iter().product::<usize>();
            Tensor::from_f32(&vec![0.0; n], shape, &budget).unwrap()
        };
        let u = |n: usize| Tensor::from_u32(&vec![0; n], &[n], &budget).unwrap();
        let chunk = CeChunk { rows: 2, cols: 3 };
        assert_eq!(
            linear_ce_dims(&f(&[4, 8]), &f(&[11, 8]), &u(4), chunk).unwrap(),
            LinearCeDims {
                rows: 4,
                model_dim: 8,
                vocab: 11
            }
        );
        // Chunks larger than the problem are fine; the backend clamps.
        let big = CeChunk {
            rows: usize::MAX,
            cols: usize::MAX,
        };
        assert!(linear_ce_dims(&f(&[4, 8]), &f(&[11, 8]), &u(4), big).is_ok());
        for bad in [CeChunk { rows: 0, cols: 3 }, CeChunk { rows: 2, cols: 0 }] {
            assert!(
                matches!(
                    linear_ce_dims(&f(&[4, 8]), &f(&[11, 8]), &u(4), bad),
                    Err(OjasError::Shape { .. })
                ),
                "{bad:?}"
            );
        }
        let shape_refusals = [
            (f(&[4, 8]), f(&[11, 7]), u(4)),    // d mismatch
            (f(&[4, 8]), f(&[11, 8]), u(3)),    // N mismatch
            (f(&[2, 2, 8]), f(&[11, 8]), u(4)), // rank
            (f(&[4]), f(&[11, 8]), u(4)),       // rank
        ];
        for (x, w, t) in &shape_refusals {
            assert!(
                matches!(linear_ce_dims(x, w, t, chunk), Err(OjasError::Shape { .. })),
                "{:?} {:?} {:?}",
                x.shape(),
                w.shape(),
                t.shape()
            );
        }
        assert!(matches!(
            linear_ce_dims(&f(&[4, 8]), &f(&[11, 8]), &f(&[4]), chunk),
            Err(OjasError::Dtype { .. })
        ));
        assert!(matches!(
            linear_ce_dims(&u(4), &f(&[11, 8]), &u(4), chunk),
            Err(OjasError::Dtype { .. })
        ));
    }

    #[test]
    fn kv_dims_accept_gqa_and_refuse_every_out_of_range_position() {
        let budget = Budget::new(1 << 16);
        let f = |shape: &[usize]| {
            let n = shape.iter().product::<usize>();
            Tensor::from_f32(&vec![0.0; n], shape, &budget).unwrap()
        };
        // 6 query heads over 2 kv heads, 3 new queries, cache of 8.
        let q = f(&[2, 3, 6, 4]);
        let cache = f(&[2, 8, 2, 4]);
        let dims = cached_attention_dims(&q, &cache, &cache, 5).unwrap();
        assert_eq!(
            dims,
            KvDims {
                batch: 2,
                new: 3,
                capacity: 8,
                heads: 6,
                kv_heads: 2,
                head_dim: 4
            }
        );
        assert!(cached_attention_dims(&q, &cache, &cache, 3).is_ok());
        assert!(cached_attention_dims(&q, &cache, &cache, 8).is_ok());
        for kv_len in [0, 2, 9, usize::MAX] {
            assert!(
                matches!(
                    cached_attention_dims(&q, &cache, &cache, kv_len),
                    Err(OjasError::OutOfRange { .. })
                ),
                "kv_len {kv_len}"
            );
        }
        let shape_refusals = [
            (f(&[2, 3, 5, 4]), f(&[2, 8, 2, 4]), f(&[2, 8, 2, 4])), // 5 % 2
            (f(&[2, 3, 6, 4]), f(&[2, 8, 2, 4]), f(&[2, 8, 3, 4])), // k != v
            (f(&[1, 3, 6, 4]), f(&[2, 8, 2, 4]), f(&[2, 8, 2, 4])), // batch
            (f(&[2, 3, 6, 8]), f(&[2, 8, 2, 4]), f(&[2, 8, 2, 4])), // head dim
            (f(&[2, 3, 6]), f(&[2, 8, 2, 4]), f(&[2, 8, 2, 4])),    // rank
        ];
        for (q, k, v) in &shape_refusals {
            assert!(
                matches!(
                    cached_attention_dims(q, k, v, 3),
                    Err(OjasError::Shape { .. })
                ),
                "{:?} {:?} {:?}",
                q.shape(),
                k.shape(),
                v.shape()
            );
        }

        let src = f(&[2, 3, 2, 4]);
        assert_eq!(kv_cache_write_dims(&cache, &src, 0).unwrap().new, 3);
        assert!(kv_cache_write_dims(&cache, &src, 5).is_ok(), "exact fit");
        for at in [6, 8, usize::MAX - 1, usize::MAX] {
            assert!(
                matches!(
                    kv_cache_write_dims(&cache, &src, at),
                    Err(OjasError::OutOfRange { .. })
                ),
                "at {at}"
            );
        }
        for bad in [
            f(&[2, 3, 1, 4]),
            f(&[1, 3, 2, 4]),
            f(&[2, 3, 2, 5]),
            f(&[2, 3, 2]),
        ] {
            assert!(
                matches!(
                    kv_cache_write_dims(&cache, &bad, 0),
                    Err(OjasError::Shape { .. })
                ),
                "{:?}",
                bad.shape()
            );
        }
        let ids = Tensor::from_u32(&[0; 48], &[2, 3, 2, 4], &budget).unwrap();
        assert!(matches!(
            kv_cache_write_dims(&cache, &ids, 0),
            Err(OjasError::Dtype { .. })
        ));
    }

    #[test]
    fn accumulate_grad_has_residual_add_rules_under_its_own_name() {
        const OP: &str = "accumulate_grad";
        assert_eq!(accumulate_grad_dims(&f(&[3, 5]), &f(&[3, 5])).unwrap(), 15);
        assert_eq!(accumulate_grad_dims(&f(&[]), &f(&[])).unwrap(), 1);
        shape_err(
            accumulate_grad_dims(&f(&[3, 5]), &f(&[5, 3])),
            OP,
            "does not match",
        );
        shape_err(
            accumulate_grad_dims(&f(&[3, 5]), &f(&[15])),
            OP,
            "does not match",
        );
        // Each operand in argument order: acc's dtype outranks grad's.
        dtype_err(
            accumulate_grad_dims(&u(&[3]), &of(DType::Bf16, &[3])),
            OP,
            DType::F32,
            DType::U32,
        );
        dtype_err(
            accumulate_grad_dims(&f(&[3]), &u(&[3])),
            OP,
            DType::F32,
            DType::U32,
        );
        shape_err(
            accumulate_grad_dims(&f(&[0, 3]), &f(&[0, 3])),
            OP,
            "empty tensor",
        );
        shape_err(
            accumulate_grad_dims(&f(&[2, 3]), &f(&[2, 0])),
            OP,
            "empty tensor",
        );
        let huge = broadcast(DType::F32, &[usize::MAX, 2]);
        range_err(accumulate_grad_dims(&huge, &huge), OP, "overflows");
        // The same verdict as residual_add_forward_dims, renamed and nothing else.
        let cases = [
            (f(&[2, 3]), f(&[3, 2])),
            (u(&[2]), f(&[2])),
            (f(&[0]), f(&[0])),
            (huge.clone(), huge.clone()),
            (f(&[4]), f(&[4])),
        ];
        for (a, b) in &cases {
            let want = format!("{:?}", residual_add_forward_dims(a, b))
                .replace("residual_add_forward", OP);
            assert_eq!(format!("{:?}", accumulate_grad_dims(a, b)), want);
        }
    }

    #[test]
    fn permute_dims_checks_the_operand_then_the_axes() {
        const OP: &str = "permute";
        assert_eq!(
            permute_dims(&f(&[2, 5, 3, 4]), &[0, 2, 1, 3]).unwrap(),
            vec![2, 3, 5, 4]
        );
        // Rank 0 is one element and permutes by `[]`.
        assert_eq!(permute_dims(&f(&[]), &[]).unwrap(), Vec::<usize>::new());
        // D17: a zero axis is refused as every other op refuses one, before
        // the axes are looked at.
        shape_err(permute_dims(&f(&[2, 0, 3]), &[2, 0, 1]), OP, "empty tensor");
        shape_err(permute_dims(&f(&[0]), &[1]), OP, "empty tensor");
        // The dtype outranks both the zero axis and bad axes.
        dtype_err(
            permute_dims(&u(&[0, 3]), &[0, 0]),
            OP,
            DType::F32,
            DType::U32,
        );
        range_err(
            permute_dims(&broadcast(DType::F32, &[usize::MAX, 2]), &[1, 0]),
            OP,
            "overflows",
        );
        shape_err(permute_dims(&f(&[2, 3]), &[1, 1]), OP, "repeated");
        shape_err(
            permute_dims(&f(&[1; 9]), &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
            OP,
            "exceeds",
        );
        // Metadata only: a strided view is the backend's to refuse.
        let strided = f(&[3, 2]).view(&[2, 3], &[1, 2], 0).unwrap();
        assert_eq!(permute_dims(&strided, &[1, 0]).unwrap(), vec![3, 2]);
    }

    #[test]
    fn permute_shape_accepts_exactly_the_permutations() {
        // [B, T, H, D] -> [B, H, T, D], the RoPE-to-attention move.
        assert_eq!(
            permute_output_shape("t", &[2, 5, 3, 4], &[0, 2, 1, 3]).unwrap(),
            vec![2, 3, 5, 4]
        );
        assert_eq!(
            permute_output_shape("t", &[], &[]).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(permute_output_shape("t", &[7], &[0]).unwrap(), vec![7]);
        let refused: [(&[usize], &[usize]); 5] = [
            (&[2, 3], &[0]),                         // too few axes
            (&[2, 3], &[0, 1, 2]),                   // too many axes
            (&[2, 3], &[0, 2]),                      // axis out of range
            (&[2, 3], &[1, 1]),                      // repeated axis
            (&[1; 9], &[0, 1, 2, 3, 4, 5, 6, 7, 8]), // rank above the cap
        ];
        for (shape, dims) in refused {
            match permute_output_shape("t", shape, dims) {
                Err(OjasError::Shape { op, .. }) => assert_eq!(op, "t"),
                other => panic!("{shape:?} by {dims:?}: expected Shape, got {other:?}"),
            }
        }
    }

    #[test]
    fn inverse_permutation_round_trips_every_rank_four_order() {
        let shape = [2usize, 3, 5, 7];
        let mut count = 0;
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let dims = [a, b, c, d];
                        let Ok(out) = permute_output_shape("t", &shape, &dims) else {
                            continue;
                        };
                        count += 1;
                        let inverse = inverse_permutation(&dims);
                        let back = permute_output_shape("t", &out, &inverse).unwrap();
                        assert_eq!(back, shape, "{dims:?} then {inverse:?}");
                    }
                }
            }
        }
        assert_eq!(count, 24);
    }

    #[test]
    fn gdn_log_decay_takes_per_head_parameters_over_any_leading_axes() {
        let (fw, bw) = ("gdn_log_decay_forward", "gdn_log_decay_backward");
        let a = f(&[2, 5, 3]);
        let h = f(&[3]);
        assert_eq!(
            gdn_log_decay_forward_dims(&a, &h, &h).unwrap(),
            GdnDecayDims { rows: 10, heads: 3 }
        );
        assert_eq!(
            gdn_log_decay_forward_dims(&h, &h, &h).unwrap(),
            GdnDecayDims { rows: 1, heads: 3 }
        );
        assert_eq!(
            gdn_log_decay_backward_dims(&a, &h, &h, &a).unwrap(),
            GdnDecayDims { rows: 10, heads: 3 }
        );
        shape_err(
            gdn_log_decay_forward_dims(&a, &f(&[2]), &h),
            fw,
            "a_log shape [2] is not [3]",
        );
        shape_err(
            gdn_log_decay_forward_dims(&a, &h, &f(&[1, 3])),
            fw,
            "dt_bias shape [1, 3] is not [3]",
        );
        shape_err(
            gdn_log_decay_backward_dims(&a, &h, &h, &f(&[2, 5, 4])),
            bw,
            "does not match",
        );
        shape_err(gdn_log_decay_forward_dims(&f(&[2, 0]), &h, &h), fw, "empty");
        dtype_err(
            gdn_log_decay_backward_dims(&a, &h, &h, &u(&[2, 5, 3])),
            bw,
            DType::F32,
            DType::U32,
        );
    }

    #[test]
    fn sigmoid_takes_one_shape_of_any_rank() {
        assert_eq!(sigmoid_forward_dims(&f(&[2, 3, 4])).unwrap(), 24);
        assert_eq!(sigmoid_backward_dims(&f(&[7]), &f(&[7])).unwrap(), 7);
        shape_err(
            sigmoid_backward_dims(&f(&[2, 3]), &f(&[3, 2])),
            "sigmoid_backward",
            "does not match",
        );
        dtype_err(
            sigmoid_forward_dims(&u(&[2])),
            "sigmoid_forward",
            DType::F32,
            DType::U32,
        );
    }
}
