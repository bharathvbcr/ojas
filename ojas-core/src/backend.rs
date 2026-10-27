//! Nanolab-default forward and backward operations.
//!
//! The trait is the frozen contract. This crate does not implement it.
//! Later crates own the CPU reference and the Metal kernels. There is no
//! silent fallback from [`BackendId::Metal`] to [`BackendId::Cpu`].

use crate::autocast::{bf16_to_f32, f32_to_bf16, round_f32_to_bf16, AutocastGuard, AutocastMode};
use crate::budget::Budget;
use crate::dtype::DType;
use crate::spatial::{Conv2dParams, ConvTranspose2dParams};
use crate::tensor::Tensor;
use crate::OjasError;

/// Explicit device selection. Callers choose one. Nothing here picks for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendId {
    Cpu,
    Metal,
    Wgpu,
    Cuda,
    Hip,
}

/// Arithmetic contract a backend promises.
///
/// `Exact` is the reference: reductions run in ascending index order in
/// `f32`, there is no fused multiply-add, and the bits do not depend on the
/// thread count. On the CPU its `exp` is correctly rounded and the same on
/// every platform, but `ln` (cross-entropy's loss) still comes from the
/// platform libm, which does not round alike everywhere (glibc and Apple
/// differ), so a loss's bits match across thread counts, not across
/// platforms. `Fast` may
/// fuse multiply-adds, use SIMD intrinsics or a
/// vendor BLAS, and reorder within a reduction; the bits still must not
/// depend on the thread count, and results are compared to `Exact` at a
/// stated tolerance rather than bitwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Numerics {
    #[default]
    Exact,
    Fast,
}

/// RMSNorm epsilon for the nanolab default (`mixers.py` `RMSNorm`, `eps=1e-6`).
/// Do not substitute [`f32::EPSILON`].
pub const RMS_NORM_EPS: f32 = 1e-6;

/// Added to the total gradient norm in the denominator of `clip_grad_norm_`.
///
/// The coefficient is `max_norm / (total_norm + CLIP_GRAD_NORM_EPS)`.
/// Gradients are multiplied by `min(1, coefficient)`. The constant stays in
/// the denominator even when the coefficient is clamped to 1.
pub const CLIP_GRAD_NORM_EPS: f32 = 1e-6;

/// `MetalBackend` and `WgpuBackend` causal attention refuse a head dimension
/// above this.
///
/// The tiled forward and backward kernels are instantiated for widths 16, 32,
/// 64, 128 and 256, and a head dimension pads up to the next width. Nothing
/// is clamped: a larger head dimension is [`OjasError::UnsupportedHeadDim`].
/// 256 is the ceiling that still fits a WebGPU workgroup (256 invocations,
/// 16 KiB of shared memory at the smallest tiles). The separate tiny training
/// step in `ojas-metal/src/gpu.rs` accepts only 64 and refuses anything else
/// as a shape error.
pub const METAL_MAX_HEAD_DIM: u32 = 256;

/// Newton-Schulz coefficients from nanolab `zeropower_via_newtonschulz5`.
pub const MUON_NS5_A: f64 = 3.4445;
pub const MUON_NS5_B: f64 = -4.7750;
pub const MUON_NS5_C: f64 = 2.0315;

/// Divisor epsilon inside that Newton-Schulz normalization (`eps: float = 1e-7`).
pub const MUON_NS_EPS: f64 = 1e-7;

/// nanolab `Config.beta1`.
pub const ADAMW_BETA1: f64 = 0.9;
/// nanolab `Config.beta2` (0.95, not 0.999).
pub const ADAMW_BETA2: f64 = 0.95;
/// nanolab `Config.eps` for AdamW. This is not [`CLIP_GRAD_NORM_EPS`].
pub const ADAMW_EPS: f64 = 1e-8;

/// `1 / sqrt(head_dim)` in f32. `head_dim == 0` is [`OjasError::OutOfRange`].
pub fn sdpa_scale(head_dim: u32) -> Result<f32, OjasError> {
    if head_dim == 0 {
        return Err(OjasError::OutOfRange {
            op: "sdpa_scale",
            detail: "head_dim is 0".to_string(),
        });
    }
    Ok(1.0 / (head_dim as f32).sqrt())
}

/// The order logits are ranked in for decoding: column `a` of `row` against
/// column `b`, value descending, then column ascending. `+0.0` and `-0.0`
/// are one value, so a tie between them goes to the lower column, as
/// `torch.argmax` and [`Backend::argmax_rows`] take ties; `-inf` ranks after
/// every finite value. Distinct columns never compare equal. NaN is not
/// ranked: callers refuse it first. [`Backend::topk_rows`] and
/// `ojas_infer`'s sampler (`top_k`, and temperature 0) rank by it, so every
/// greedy pick in ojas is the lowest column of the largest value.
pub fn rank_logits(row: &[f32], a: usize, b: usize) -> std::cmp::Ordering {
    let key = |v: f32| if v == 0.0 { 0.0 } else { v };
    key(row[b]).total_cmp(&key(row[a])).then(a.cmp(&b))
}

/// Metal refuses `head_dim > METAL_MAX_HEAD_DIM`. CPU allows any non-zero
/// head dimension.
/// Neither path accepts 0. This does not clamp.
pub fn refuse_unsupported_metal_head_dim(
    backend: BackendId,
    head_dim: u32,
) -> Result<(), OjasError> {
    if head_dim == 0 {
        return Err(OjasError::OutOfRange {
            op: "refuse_unsupported_metal_head_dim",
            detail: "head_dim is 0".to_string(),
        });
    }
    if backend == BackendId::Metal && head_dim > METAL_MAX_HEAD_DIM {
        return Err(OjasError::UnsupportedHeadDim {
            head_dim,
            limit: METAL_MAX_HEAD_DIM,
        });
    }
    Ok(())
}

/// Next optimizer step. `u64::MAX` is refused so the count cannot wrap to 0.
pub fn next_step(step: u64) -> Result<u64, OjasError> {
    step.checked_add(1).ok_or_else(|| OjasError::OutOfRange {
        op: "next_step",
        detail: "step counter is u64::MAX".to_string(),
    })
}

/// The NS5 step runs five iterations. Any other count is an error.
pub fn require_ns5(steps: u32) -> Result<(), OjasError> {
    if steps == 5 {
        Ok(())
    } else {
        Err(OjasError::OutOfRange {
            op: "require_ns5",
            detail: format!("Newton-Schulz steps {steps} != 5"),
        })
    }
}

/// f64 binary exponentiation. A huge exponent underflows to 0 for a base in
/// `[0, 1)` instead of overflowing, which is what bias correction needs.
///
/// Every backend forms AdamW bias correction with this, so the f64 bits of
/// `1 - beta^step` are the same on CPU, Metal and wgpu.
pub fn pow_u64(base: f64, mut exp: u64) -> f64 {
    let mut result = 1.0f64;
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= b;
        }
        exp >>= 1;
        if exp > 0 {
            b *= b;
        }
    }
    result
}

/// Global-norm clip coefficient `min(1, max_norm / (total_norm + eps))`.
///
/// A non-finite `max_norm`, `total_norm` or coefficient is
/// [`OjasError::NonFinite`]; a negative `max_norm` is
/// [`OjasError::OutOfRange`]. The caller scales only on `Ok`.
///
/// A `max_norm` of 0 is accepted and gives 0, so the gradients are zeroed:
/// torch's `clip_grad_norm_(max_norm=0)` does the same, and the oracle
/// compares this op to torch. It is not "no clipping". ojas-model's
/// `TrainConfig::validate` refuses a `grad_clip` of 0 for that reason.
pub fn clip_scale(max_norm: f32, total_norm: f32) -> Result<f32, OjasError> {
    const OP: &str = "clip_grad_norm";
    if !max_norm.is_finite() || !total_norm.is_finite() {
        return Err(OjasError::NonFinite { op: OP });
    }
    if max_norm < 0.0 {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: format!("max_norm {max_norm} is negative"),
        });
    }
    let coef = max_norm / (total_norm + CLIP_GRAD_NORM_EPS);
    if !coef.is_finite() {
        return Err(OjasError::NonFinite { op: OP });
    }
    Ok(coef.min(1.0))
}

/// Validate an [`AdamWConfig`] and the step it will run, returning
/// `(next_step, 1 - beta1^next, 1 - beta2^next)`.
///
/// Refused, before any state is touched: a non-finite scalar
/// ([`OjasError::NonFinite`]); a beta outside `[0, 1)`, `eps <= 0`, a
/// negative `lr` or `weight_decay`, a step of `u64::MAX`, or a bias
/// correction that is not a positive finite value ([`OjasError::OutOfRange`]).
pub fn check_adamw(config: AdamWConfig, step: u64) -> Result<(u64, f64, f64), OjasError> {
    const OP: &str = "adamw_step";
    let scalars = [
        config.lr,
        config.beta1,
        config.beta2,
        config.eps,
        config.weight_decay,
    ];
    if scalars.iter().any(|value| !value.is_finite()) {
        return Err(OjasError::NonFinite { op: OP });
    }
    if !(config.beta1 >= 0.0 && config.beta1 < 1.0 && config.beta2 >= 0.0 && config.beta2 < 1.0) {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: "adamw beta must be in [0, 1)".to_string(),
        });
    }
    if config.eps <= 0.0 {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: "adamw eps must be > 0".to_string(),
        });
    }
    for (name, value) in [("lr", config.lr), ("weight_decay", config.weight_decay)] {
        if value < 0.0 {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("{name} {value} is negative"),
            });
        }
    }
    let next = next_step(step)?;
    let bc1 = 1.0 - pow_u64(config.beta1, next);
    let bc2 = 1.0 - pow_u64(config.beta2, next);
    if !(bc1.is_finite() && bc2.is_finite() && bc1 > 0.0 && bc2 > 0.0) {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: "adamw bias correction is not a positive finite value".to_string(),
        });
    }
    Ok((next, bc1, bc2))
}

/// Torch single-tensor AdamW scalars. The host forms them in f64.
///
/// Order, matching tessl `qwen35_adamw_f32`:
/// 1. Decoupled decay first: `w = p * (1 - lr * weight_decay)`.
/// 2. First moment by torch `lerp` (the form switches at weight 0.5), and
///    second moment `v = beta2 * v + (1 - beta2) * g * g`.
/// 3. Bias correction is outside the square root:
///    `denom = sqrt(v) / sqrt(1 - beta2^step) + eps`.
/// 4. `w += -step_size * m / denom` with `step_size = lr / (1 - beta1^step)`.
///
/// The step count is the post-increment value (`step + 1` before the update).
/// Use [`next_step`]. Do not wrap. `eps` is [`ADAMW_EPS`] unless the caller
/// passes another finite value. Weight decay on the nanolab Muon hybrid is 0
/// for both AdamW groups; Muon carries decay 0.1.
#[derive(Clone, Copy, Debug)]
pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl AdamWConfig {
    pub fn nanolab(lr: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1: ADAMW_BETA1,
            beta2: ADAMW_BETA2,
            eps: ADAMW_EPS,
            weight_decay,
        }
    }
}

/// Which optimizer call [`Backend::optimizer_scratch_bytes`] sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerKind {
    AdamW,
    MuonNs5,
}

/// Precision of the Newton-Schulz iterate in [`Backend::muon_ns5_step`].
///
/// The momentum buffer, the parameter and its update stay f32 under both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Ns5Precision {
    /// The iterate and every intermediate are f32. This is nanolab with
    /// `X = G.float()` in place of `X = G.bfloat16()`.
    #[default]
    F32,
    /// Stock nanolab and PyTorch Muon (`X = G.bfloat16()`, optim.py:46), as
    /// torch's eager bf16 ops run it: every intermediate is rounded to bf16
    /// (round to nearest even), and each GEMM takes bf16 operands and
    /// accumulates in f32 before its output is rounded. Per step:
    /// `A = r(X Xᵀ)`, `B = r(r(b A) + r(c r(A A)))`,
    /// `X = r(r(a X) + r(B X))`. Before the loop `X = r(G)`, the Frobenius
    /// norm is rounded, `+ eps` is a bf16 add, and the division is rounded.
    /// The result is widened back to f32 for the update.
    Bf16,
}

/// Muon NS5 as [`Backend::muon_ns5_step`] runs it.
///
/// Nesterov momentum (`buf = mom * buf + g`, update = `g + mom * buf` when
/// Nesterov is on). Orthogonalize with five Newton-Schulz steps at `ns5`
/// precision. Then scale by `max(1, rows/cols)^0.5`. Decoupled decay is
/// `p *= 1 - lr * weight_decay` before the update is added.
#[derive(Clone, Copy, Debug)]
pub struct MuonNs5Config {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub ns5: Ns5Precision,
}

impl MuonNs5Config {
    /// `lr` 0.025, momentum 0.99, weight decay 0.1, Nesterov on.
    /// Those are the nanolab defaults (`matrix_lr`, `muon_momentum`, `weight_decay`).
    /// Newton-Schulz is [`Ns5Precision::F32`]; stock nanolab's is
    /// [`Ns5Precision::Bf16`].
    pub fn nanolab_default() -> Self {
        Self {
            lr: 0.025,
            momentum: 0.99,
            weight_decay: 0.1,
            nesterov: true,
            ns5: Ns5Precision::F32,
        }
    }
}

/// Gradients of the per-head sigmoid output gate.
#[derive(Clone, Debug)]
pub struct PerHeadGateGrad {
    pub input: Tensor,
    pub weight: Tensor,
    pub bias: Tensor,
    pub attn_out: Tensor,
}

/// How [`Backend::attention_forward`] masks and biases the scores
/// `scale * q . k`, with `scale = `[`sdpa_scale`]`(D)`. The default is
/// bidirectional attention over every key, no bias.
///
/// A key is visible to query `i` of batch `b` when it passes every mask
/// that is set, and a row that sees no key outputs zeros (its
/// log-sum-exp is stored as 0; see [`Backend::attention_forward_saving`]):
///
/// - `causal`: query `i` sits at position `p = i + Tk - Tq` and sees keys
///   `j <= p`. With `Tq == Tk` that is the usual lower triangle; with
///   `Tq < Tk` the queries are the last `Tq` positions (a KV-cache prefill),
///   and with `Tq > Tk` the first `Tq - Tk` queries see nothing.
/// - `window` (only with `causal`, at least 1): also `j + window > p`. A
///   window of at least `Tk` is no window.
/// - `key_lens`: a `U32` tensor `[B]`; also `j < key_lens[b]`. A length
///   above `Tk` is `Tk`, and 0 hides every key of that batch row. This is
///   the mask of a padded text prompt: the padded keys are the tail.
///
/// `bias` is added to the scores before the softmax: `F32` (or `Bf16`)
/// `[Bb, Hb, Tqb, Tk]` with `Bb` in `{1, B}`, `Hb` in `{1, H}` and `Tqb` in
/// `{1, Tq}`; an extent of 1 broadcasts. A bias is finite, as every operand
/// is; a mask that hides a key is `key_lens`, `causal` or `window`, not a
/// large negative number.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttentionOpts<'a> {
    pub causal: bool,
    pub window: Option<usize>,
    pub bias: Option<&'a Tensor>,
    pub key_lens: Option<&'a Tensor>,
}

/// Gradients of [`Backend::attention_backward`].
#[derive(Clone, Debug)]
pub struct AttentionGrad {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    /// Shaped like the bias, summed over the extents it broadcast across.
    /// `None` unless the caller asked for it.
    pub bias: Option<Tensor>,
}

/// Gradients of the value-residual blend. `lambda` is the scalar parameter.
#[derive(Clone, Debug)]
pub struct ValueResidualGrad {
    pub value: Tensor,
    pub value0: Tensor,
    pub lambda: Tensor,
}

/// Tokens between the states [`Backend::chunked_gdn_forward`] saves. The
/// backward recomputes each chunk's states from its checkpoint, as tessl's
/// `gdn_train` kernels do (`GDN_TRAIN_CKPT`).
pub const GDN_CHECKPOINT_TOKENS: usize = 64;

/// The epsilon of the gated delta rule's in-kernel `l2norm` on `q` and `k`:
/// `x * rsqrt(|x|^2 + eps)` (transformers `modeling_qwen3_5.py`).
pub const GDN_L2NORM_EPS: f32 = 1e-6;

/// The key head dim tessl's Metal `gdn_train` kernels are compiled for.
pub const METAL_GDN_KEY_DIM: usize = 128;

/// Metal's `gdn_train` kernels split the value dim into blocks of this many
/// columns, so the value dim must be a multiple of it.
pub const METAL_GDN_VALUE_BLOCK: usize = 16;

/// Operands of the gated delta rule (Gated DeltaNet), at transformers'
/// training seam `torch_chunk_gated_delta_rule(q, k, v, g, beta,
/// initial_state, use_qk_l2norm_in_kernel=True)`, with the gates already
/// computed.
///
/// `q`, `k` are `[B, T, H, Dk]`, `v` is `[B, T, H, Dv]`, `g` (the log decay)
/// and `beta` are `[B, T, H]`, and `initial_state` is `[B, H, Dk, Dv]`
/// (zeros when `None`). All `F32`. There is no head grouping: value heads
/// equal key heads, and a grouped model repeats its key heads before the
/// call, as transformers does.
#[derive(Clone, Copy, Debug)]
pub struct GdnInputs<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub g: &'a Tensor,
    pub beta: &'a Tensor,
    pub initial_state: Option<&'a Tensor>,
}

/// What [`Backend::chunked_gdn_forward`] produces.
#[derive(Clone, Debug)]
pub struct GdnForward {
    /// `[B, T, H, Dv]`.
    pub output: Tensor,
    /// The state after the last token, `[B, H, Dk, Dv]`.
    pub final_state: Tensor,
    /// `[B, H, ceil(T / GDN_CHECKPOINT_TOKENS), Dk, Dv]`: checkpoint `c` is
    /// the state entering token `64 c`, before that token's decay. The
    /// backward consumes it.
    pub checkpoints: Tensor,
}

/// Gradients of the gated delta rule, each shaped like its input.
/// `initial_state` is `Some` exactly when the forward had one.
#[derive(Clone, Debug)]
pub struct GdnGrad {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub g: Tensor,
    pub beta: Tensor,
    pub initial_state: Option<Tensor>,
}

/// Gradients of [`Backend::gated_rms_norm_forward`], each shaped like its
/// input.
#[derive(Clone, Debug)]
pub struct GatedRmsGrad {
    pub input: Tensor,
    pub gate: Tensor,
    pub weight: Tensor,
}

/// Gradients of [`Backend::gdn_log_decay_forward`], each shaped like its
/// input.
#[derive(Clone, Debug)]
pub struct GdnDecayGrad {
    pub input: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
}

/// `host` where `backend` computes: as given on the CPU, uploaded otherwise.
fn place_host<B: Backend + ?Sized>(backend: &B, host: Tensor) -> Result<Tensor, OjasError> {
    if backend.id() == BackendId::Cpu {
        Ok(host)
    } else {
        backend.upload(&host)
    }
}

/// `scalar` (one element, where `backend` computes) repeated over `shape`,
/// built on the backend without reading `scalar` back: `ones[cols, 1] @
/// s[1, 1]^T` is a `[cols, 1]` column of `s`, and `ones[rows, 1] @
/// column^T` is `[rows, cols]`. Each output is one product with 1, so it is
/// `s` exactly. Holds `rows + cols` ones besides the result.
pub fn broadcast_scalar<B: Backend + ?Sized>(
    backend: &B,
    scalar: &Tensor,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    const OP: &str = "broadcast_scalar";
    let cols = *shape.last().ok_or_else(|| OjasError::Shape {
        op: OP,
        detail: "cannot broadcast a scalar over rank 0".to_string(),
    })?;
    let n = crate::shape_product(shape)?;
    if n == 0 {
        return Err(OjasError::Shape {
            op: OP,
            detail: "empty tensor".to_string(),
        });
    }
    let rows = n / cols;
    let ones = |len: usize| {
        let host = Tensor::from_f32(&vec![1.0; len], &[len, 1], backend.budget())?;
        place_host(backend, host)
    };
    let s = scalar.reshape(&[1, 1])?;
    let column = backend.linear_forward(&ones(cols)?, &s)?;
    let full = backend.linear_forward(&ones(rows)?, &column)?;
    full.reshape(shape)
}

/// The refusal of a Qwen3.5 hybrid-layer op (causal conv1d, gated RMSNorm,
/// partial RoPE, sigmoid, the GDN log decay) that backend `id` does not
/// implement. There is no host
/// fallback.
fn unsupported_hybrid(id: BackendId, op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: format!("{id:?} backend does not implement {op}"),
    }
}

/// The refusal of a forward-only inference op (LayerNorm, GELU, linear with
/// bias, broadcast add and mul, scale, softmax) that backend `id` does not
/// implement. There is no host fallback.
fn unsupported_inference(id: BackendId, op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: format!("{id:?} backend does not implement {op}"),
    }
}

/// The refusal of a spatial op (2D convolution, pooling, resampling,
/// interleaved RoPE) that backend `id` does not implement. There is no host
/// fallback.
fn unsupported_spatial(id: BackendId, op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: format!("{id:?} backend does not implement {op}"),
    }
}

/// Metal runs the gated delta rule on tessl's `gdn_train` kernels, which are
/// compiled for a key dim of [`METAL_GDN_KEY_DIM`] and a value dim that is a
/// multiple of [`METAL_GDN_VALUE_BLOCK`]. Any other shape on Metal is
/// [`OjasError::Unsupported`]; other backends take any positive dims. This
/// does not pad.
pub fn refuse_unsupported_metal_gdn(
    backend: BackendId,
    key_dim: usize,
    value_dim: usize,
) -> Result<(), OjasError> {
    if backend != BackendId::Metal {
        return Ok(());
    }
    if key_dim != METAL_GDN_KEY_DIM {
        return Err(OjasError::Unsupported {
            op: "chunked_gdn",
            detail: format!(
                "Metal gdn_train is compiled for key dim {METAL_GDN_KEY_DIM}, got {key_dim}"
            ),
        });
    }
    if !value_dim.is_multiple_of(METAL_GDN_VALUE_BLOCK) {
        return Err(OjasError::Unsupported {
            op: "chunked_gdn",
            detail: format!(
                "Metal gdn_train needs a value dim that is a multiple of \
                 {METAL_GDN_VALUE_BLOCK}, got {value_dim}"
            ),
        });
    }
    Ok(())
}

/// The host defaults of the dtype casts do not download: a device tensor is
/// [`OjasError::Unsupported`], and the device backend overrides the cast.
fn refuse_device_cast(op: &'static str, tensor: &Tensor) -> Result<(), OjasError> {
    match tensor.device() {
        None => Ok(()),
        Some(found) => Err(OjasError::Unsupported {
            op,
            detail: format!(
                "{found:?} tensor is not downloaded to cast it; there is no CPU fallback"
            ),
        }),
    }
}

/// Refuse a `Bf16` operand on a backend whose [`Backend::bf16_operands`] is
/// `false`, with [`OjasError::Unsupported`] naming `op` and `backend`.
///
/// Such a backend calls this right after the op's shape validator, which
/// accepts `Bf16` for the matmul-class operands, and before it reads,
/// charges or writes anything, so a bf16 operand is never read as `f32`
/// bytes.
pub fn refuse_bf16_operands(
    op: &'static str,
    backend: BackendId,
    operands: &[&Tensor],
) -> Result<(), OjasError> {
    match operands.iter().position(|t| t.dtype() == DType::Bf16) {
        None => Ok(()),
        Some(slot) => Err(OjasError::Unsupported {
            op,
            detail: format!(
                "{backend:?} backend does not take bf16 operands (operand {slot}); \
                 widen it with to_f32 first"
            ),
        }),
    }
}

/// Refuse a `Bf16` tied table on a backend that has no bf16 table lane
/// ([`Backend::bf16_operands`]): [`OjasError::Dtype`] naming `op`, expecting
/// `F32`, as the validators refused it before the lane existed.
///
/// Such a backend calls this right after the op's shape validator, which
/// accepts a `Bf16` table, and before it reads, charges or writes anything,
/// so the table's 2-byte elements are never read as `f32` bytes.
pub fn refuse_bf16_table(op: &'static str, table: &Tensor) -> Result<(), OjasError> {
    match table.dtype() {
        DType::Bf16 => Err(OjasError::Dtype {
            op,
            expected: DType::F32,
            got: DType::Bf16,
        }),
        _ => Ok(()),
    }
}

/// Nanolab-default training ops.
///
/// Every method returns [`Result`]. Implementations live in `ojas-cpu`
/// and `ojas-metal`. A method that this crate cannot implement has no
/// body here.
///
/// Layout notes shared by the methods:
/// - Linear weights are `[out, in]`, no bias, `y = x @ W^T`, matching
///   `nn.Linear(..., bias=False)`.
/// - Token ids are [`crate::DType::U32`]. An id outside the vocabulary is
///   [`OjasError::OutOfRange`], not a wrapped read.
/// - `ignore_index: Option<u32>` is the unsigned form of nanolab's `-1`.
///   DType has no signed integer. The data loader maps torch's `-1` onto the
///   sentinel before the call. Mean is over valid targets. If every row is
///   ignored, forward and backward return [`OjasError::NonFinite`]. The count
///   is not floored at 1, because that turned an empty reduction into a
///   finite loss of 0.
/// - RoPE is the half-split in `mixers.apply_rope`: last axis split in half,
///   rotate `(-x2, x1)`, then `x * cos + rot * sin`. Strides and `byte_offset`
///   come from the tensor. Do not hard-code a query stride of `2 * head_dim`.
/// - Causal attention scale is [`sdpa_scale`]. Metal calls
///   [`refuse_unsupported_metal_head_dim`] and does not truncate.
/// - The per-head gate is `sigmoid(x @ W^T + bias)` with `W` shaped
///   `[n_head, d_model]`, then multiplied onto the attention output.
/// - Value residual is `s = sigmoid(lambda); (1 - s) * v + s * v0`.
/// - Non-finite inputs and gradients are [`OjasError::NonFinite`].
///   Clipping and Adam must not apply them.
pub trait Backend {
    fn id(&self) -> BackendId;

    /// Budget this backend charges for outputs and scratch probes.
    fn budget(&self) -> &Budget;

    /// Arithmetic contract of every op on this backend.
    ///
    /// The default is [`Numerics::Exact`]. A GPU backend whose reductions
    /// are not ascending-index `f32` without fused multiply-add must
    /// override this to return [`Numerics::Fast`].
    fn numerics(&self) -> Numerics {
        Numerics::Exact
    }

    /// Make `tensor` resident where this backend computes.
    ///
    /// A tensor already there is returned as a shared clone. The default
    /// fits a host backend: a device tensor is copied back to the host and
    /// counted in [`crate::device_readbacks`]. On a backend whose id is not
    /// [`BackendId::Cpu`], the default refuses a host tensor with
    /// [`OjasError::Unsupported`] rather than pretend it is resident, and
    /// refuses another device's tensor with [`OjasError::Placement`]. A
    /// device backend overrides this to copy host bytes up.
    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        match tensor.device() {
            None if self.id() == BackendId::Cpu => Ok(tensor.clone()),
            None => Err(OjasError::Unsupported {
                op: "Backend::upload",
                detail: format!(
                    "{:?} backend does not implement upload of a host tensor; \
                     there is no CPU fallback",
                    self.id()
                ),
            }),
            Some(_) if self.id() == BackendId::Cpu => tensor.to_host(self.budget()),
            Some(found) if found == self.id() => Ok(tensor.clone()),
            Some(found) => Err(OjasError::Placement {
                op: "Backend::upload",
                expected: Some(self.id()),
                found: Some(found),
            }),
        }
    }

    /// Host copy of a tensor this backend produced.
    fn download(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        tensor.to_host(self.budget())
    }

    /// A contiguous copy of an f32 `input` with its axes reordered:
    /// output axis `i` is input axis `dims[i]`, as in
    /// `torch.permute(input, dims).contiguous()`.
    ///
    /// This is how `[B, T, H, D]` (RoPE, the per-head gate) reaches
    /// `[B, H, T, D]` (causal attention) and back. The gradient of a permute
    /// is the upstream gradient permuted by [`crate::inverse_permutation`], so
    /// there is no separate backward method. Implementations validate with
    /// [`crate::permute_dims`] before anything else, keep the result on this
    /// backend, and move
    /// values without arithmetic, so a finite output's bits equal the input's
    /// bits. A non-finite input is [`OjasError::NonFinite`], reported the way
    /// this backend reports it for every other op; it is not passed through.
    ///
    /// The default refuses with [`OjasError::Unsupported`]. A backend that
    /// has not implemented it says so; there is no host fallback.
    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        let _ = (input, dims);
        Err(OjasError::Unsupported {
            op: "permute",
            detail: format!("{:?} backend does not implement permute", self.id()),
        })
    }

    /// The rows of `table` (`[vocab, dim]`) that `token_ids` pick, as an
    /// `F32` tensor shaped `ids.shape ++ [dim]`. A `Bf16` table is a
    /// bf16-operand slot ([`Self::bf16_operands`]): each picked row is
    /// widened exactly as it is read, never through an `f32` copy of the
    /// table. Validate with [`crate::embedding_forward_dims`].
    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError>;

    /// The table gradient of [`Self::embedding_forward`], `F32` shaped like
    /// `table`. The table is read for its shape (and scanned as every other
    /// operand is), so a `Bf16` table gives the `F32` table's bits.
    /// Validate with [`crate::embedding_backward_dims`].
    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError>;

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError>;

    /// `(grad_input, grad_weight)`.
    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError>;

    /// `grad_input` of [`Backend::linear_backward`] alone, for a frozen
    /// weight: the same operands and the same bits, with no weight
    /// gradient. `input` is not read by the product, but it is checked
    /// (shape, dtype, finiteness) exactly as the full path checks it, so
    /// both return the same verdict.
    ///
    /// The default runs `linear_backward` and drops the weight gradient,
    /// so every backend answers it correctly; CPU and Metal override it
    /// with a body that never allocates, charges or computes that
    /// gradient.
    fn linear_backward_input(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.linear_backward(input, weight, grad_output)
            .map(|(grad_input, _)| grad_input)
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError>;

    /// `(grad_input, grad_weight)`.
    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError>;

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError>;

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError>;

    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError>;

    /// `(grad_q, grad_k, grad_q_weight, grad_k_weight)`.
    #[allow(clippy::too_many_arguments)]
    fn rms_qk_norm_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        grad_q: &Tensor,
        grad_k: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError>;

    /// Causal SDPA. `q` is `[B, H, T, D]`; `k` and `v` are `[B, Hkv, T, D]`.
    /// `H` is a positive multiple of `Hkv`. Query head `h` reads KV head
    /// `h / (H / Hkv)`. `T` is the same on every operand. With `window`
    /// `Some(W)` (`W >= 1`) query `t` attends to keys `t - W < j <= t`;
    /// with `None`, to `0..=t`. A window of at least `T` is `None`.
    ///
    /// Returns the output, shaped like `q`, and each query row's
    /// log-sum-exp `ln sum_j e^(s_tj)` over the keys it attends to, `[B, H,
    /// T]`, which [`Self::causal_sdpa_backward`] consumes. A device may
    /// refuse `D` above its own limit.
    fn causal_sdpa_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError>;

    /// `(grad_q, grad_k, grad_v)`, shaped like `q`, `k` and `v`, from the
    /// forward's `output` and `lse` for the same operands and window: the
    /// probabilities are `e^(s - lse)` and each row's `Dr = dO · O`, so no
    /// row maximum or sum is formed again. `grad_k` and `grad_v` sum the
    /// query heads of each KV head in increasing query-head order.
    #[allow(clippy::too_many_arguments)]
    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        output: &Tensor,
        lse: &Tensor,
        grad_output: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError>;

    /// [`Self::causal_sdpa_backward`] for a caller that did not keep the
    /// forward's output and log-sum-exp: the forward runs again first. The
    /// gradient's shape is checked before that forward runs.
    fn causal_sdpa_backward_recompute(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        crate::shapes::causal_sdpa_grad_shape(q, grad_output)?;
        let (output, lse) = self.causal_sdpa_forward(q, k, v, window)?;
        self.causal_sdpa_backward(q, k, v, &output, &lse, grad_output, window)
    }

    /// Scaled dot-product attention with the masks and bias of
    /// [`AttentionOpts`]: bidirectional by default, and the query and key
    /// lengths may differ (cross-attention).
    ///
    /// `q` is `[B, H, Tq, D]`; `k` and `v` are `[B, Hkv, Tk, D]`, with `H` a
    /// positive multiple of `Hkv` (query head `h` reads KV head `h / (H /
    /// Hkv)`). Returns the output, shaped like `q`. With `causal` set,
    /// `Tq == Tk`, no bias and no `key_lens` it equals
    /// [`Self::causal_sdpa_forward`]. A device may refuse `D` above its own
    /// limit. Validate with [`crate::attention_forward_dims`].
    ///
    /// The inference form: a backend that does not need the log-sum-exp
    /// can skip it. A backend without the op returns
    /// [`OjasError::Unsupported`] under this method's own name, not under
    /// [`Self::attention_forward_saving`]'s: the default does not delegate,
    /// so a backend that implements the saving form implements this one too
    /// (the CPU and Metal backends both do).
    fn attention_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        opts: AttentionOpts<'_>,
    ) -> Result<Tensor, OjasError> {
        let _ = (q, k, v, opts);
        Err(unsupported_inference(self.id(), "attention_forward"))
    }

    /// [`Self::attention_forward`] and each query row's log-sum-exp `ln
    /// sum_j e^(s_ij)` over the keys it sees, `[B, H, Tq]`, which
    /// [`Self::attention_backward`] consumes. A row that sees no key has
    /// output zeros and log-sum-exp 0 (the empty sum's `-inf` is not a
    /// finite value); the backward gives it no gradient.
    fn attention_forward_saving(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        opts: AttentionOpts<'_>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let _ = (q, k, v, opts);
        Err(OjasError::Unsupported {
            op: "attention_forward_saving",
            detail: format!(
                "{:?} backend does not implement attention_forward_saving",
                self.id()
            ),
        })
    }

    /// The gradients of [`Self::attention_forward_saving`] from its `output`
    /// and `lse` for the same operands and `opts`: the probabilities are
    /// `e^(s - lse)` and each row's `Dr = dO . O`. `bias_grad` asks for the
    /// bias gradient too, which needs a bias in `opts`; a constant bias
    /// (a mask) leaves it off and pays nothing for it. `grad_k` and `grad_v`
    /// sum the query heads of each KV head in increasing query-head order.
    #[allow(clippy::too_many_arguments)]
    fn attention_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        output: &Tensor,
        lse: &Tensor,
        grad_output: &Tensor,
        opts: AttentionOpts<'_>,
        bias_grad: bool,
    ) -> Result<AttentionGrad, OjasError> {
        let _ = (q, k, v, output, lse, grad_output, opts, bias_grad);
        Err(OjasError::Unsupported {
            op: "attention_backward",
            detail: format!(
                "{:?} backend does not implement attention_backward",
                self.id()
            ),
        })
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError>;

    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError>;

    /// Forward, and the per-head sigmoid `[rows, heads]` when this backend
    /// keeps it for backward. The default runs the forward and saves nothing.
    fn per_head_sigmoid_gate_forward_saving(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>), OjasError> {
        Ok((
            self.per_head_sigmoid_gate_forward(input, weight, bias, attn_out)?,
            None,
        ))
    }

    /// Backward using a scale from [`Self::per_head_sigmoid_gate_forward_saving`].
    /// The default ignores `scales` and recomputes.
    fn per_head_sigmoid_gate_backward_saved(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
        scales: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        let _ = scales;
        self.per_head_sigmoid_gate_backward(input, weight, bias, attn_out, grad_output)
    }

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError>;

    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError>;

    /// The gated delta rule (Gated DeltaNet's mixer), published form
    /// (`arXiv:2412.06464` eq. 8, transformers' recurrence). Per `(b, h)`,
    /// with the state `S` `[Dk, Dv]` starting at `initial_state` or zeros:
    ///
    /// ```text
    /// q^ = l2norm(q_t) / sqrt(Dk),  k^ = l2norm(k_t)   l2norm(x) = x / sqrt(|x|^2 + 1e-6)
    /// S^ = exp(g_t) S
    /// S  = S^ + k^ (beta_t (v_t - S^^T k^))^T
    /// o_t = S^T q^
    /// ```
    ///
    /// The correction reads the decayed state. Shapes are [`GdnInputs`];
    /// [`crate::chunked_gdn_forward_dims`] validates them. The forward also
    /// returns the state every [`GDN_CHECKPOINT_TOKENS`] tokens, which
    /// [`Self::chunked_gdn_backward`] consumes in place of the per-token
    /// states. A device may refuse dims its kernels are not compiled for
    /// ([`refuse_unsupported_metal_gdn`]).
    ///
    /// The default refuses with [`OjasError::Unsupported`]. There is no host
    /// fallback.
    fn chunked_gdn_forward(&self, inputs: GdnInputs<'_>) -> Result<GdnForward, OjasError> {
        let _ = inputs;
        Err(OjasError::Unsupported {
            op: "chunked_gdn_forward",
            detail: format!(
                "{:?} backend does not implement the gated delta rule",
                self.id()
            ),
        })
    }

    /// Gradients of `sum(grad_output * o) + sum(grad_final_state * S_T)`
    /// with respect to every input of [`Self::chunked_gdn_forward`], given
    /// the same inputs and the `checkpoints` that forward returned.
    /// `grad_final_state` is zeros when `None`.
    fn chunked_gdn_backward(
        &self,
        inputs: GdnInputs<'_>,
        checkpoints: &Tensor,
        grad_output: &Tensor,
        grad_final_state: Option<&Tensor>,
    ) -> Result<GdnGrad, OjasError> {
        let _ = (inputs, checkpoints, grad_output, grad_final_state);
        Err(OjasError::Unsupported {
            op: "chunked_gdn_backward",
            detail: format!(
                "{:?} backend does not implement the gated delta rule",
                self.id()
            ),
        })
    }

    /// Depthwise causal convolution over time, then SiLU: Qwen3.5's
    /// `conv1d` in front of the gated delta rule, from a zero state.
    ///
    /// ```text
    /// y[b, t, c] = silu(sum_{j < K} weight[c, j] * x[b, t + j - (K - 1), c])
    /// ```
    ///
    /// with `x` zero before `t = 0`. `input` is `[B, T, C]`, `weight`
    /// `[C, K]` (transformers' `conv1d.weight.squeeze(1)`); the output is
    /// shaped like `input`. [`crate::causal_conv1d_silu_forward_dims`]
    /// validates the shapes. The default refuses with
    /// [`OjasError::Unsupported`].
    fn causal_conv1d_silu_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, weight);
        Err(unsupported_hybrid(self.id(), "causal_conv1d_silu_forward"))
    }

    /// `(grad_input, grad_weight)` of [`Self::causal_conv1d_silu_forward`]
    /// for `grad_output` shaped like `input`.
    fn causal_conv1d_silu_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let _ = (input, weight, grad_output);
        Err(unsupported_hybrid(self.id(), "causal_conv1d_silu_backward"))
    }

    /// Gated RMSNorm (`Qwen3_5RMSNormGated`), elementwise in the gate:
    ///
    /// ```text
    /// out = x / sqrt(mean(x^2) + eps) * weight * silu(gate)
    /// ```
    ///
    /// over the last axis of `input`; `gate` is shaped like `input` and
    /// `weight` is `[dim]` (a plain scale, not `1 + w`). A head is a row.
    /// [`crate::gated_rms_norm_forward_dims`] validates the shapes. The
    /// default refuses with [`OjasError::Unsupported`].
    fn gated_rms_norm_forward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, gate, weight, eps);
        Err(unsupported_hybrid(self.id(), "gated_rms_norm_forward"))
    }

    /// Gradients of [`Self::gated_rms_norm_forward`] for `grad_output`
    /// shaped like `input`.
    fn gated_rms_norm_backward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<GatedRmsGrad, OjasError> {
        let _ = (input, gate, weight, grad_output, eps);
        Err(unsupported_hybrid(self.id(), "gated_rms_norm_backward"))
    }

    /// Partial half-split RoPE: the leading `R` values of each head rotate
    /// as [`Self::rope_half_split_forward`] rotates a whole head (pairs `p`,
    /// `p + R / 2`), and the other `D - R` pass through unchanged. `x` is
    /// `[B, T, H, D]`; `cos` and `sin` are `[T, R]` with `R` even and at
    /// most `D`. Qwen3.5 rotates 64 of 256.
    ///
    /// Qwen3.5's multimodal RoPE assigns each frequency to a time, height
    /// or width position stream; on text-only input every stream is the
    /// token position, so it collapses to this op with the tables
    /// [`crate::mrope_text_tables`] builds.
    /// [`crate::rope_partial_forward_dims`] validates the shapes. The
    /// default refuses with [`OjasError::Unsupported`].
    fn rope_partial_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let _ = (x, cos, sin);
        Err(unsupported_hybrid(self.id(), "rope_partial_forward"))
    }

    /// Gradient of [`Self::rope_partial_forward`]: the rotation by `-angle`
    /// on the leading `R` values, the rest passed through.
    fn rope_partial_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let _ = (grad_output, cos, sin);
        Err(unsupported_hybrid(self.id(), "rope_partial_backward"))
    }

    /// Elementwise logistic sigmoid `1 / (1 + e^-x)`: Qwen3.5's attention
    /// output gate (`attn * sigmoid(gate)`) and the delta rule's `beta`.
    /// [`crate::sigmoid_forward_dims`] validates the shape. The default
    /// refuses with [`OjasError::Unsupported`].
    fn sigmoid_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let _ = input;
        Err(unsupported_hybrid(self.id(), "sigmoid_forward"))
    }

    /// `grad_output * s * (1 - s)` with `s = sigmoid(input)`.
    fn sigmoid_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        let _ = (input, grad_output);
        Err(unsupported_hybrid(self.id(), "sigmoid_backward"))
    }

    /// The gated delta rule's log decay (transformers
    /// `Qwen3_5GatedDeltaNet`):
    ///
    /// ```text
    /// g[..., h] = -exp(a_log[h]) * softplus(a[..., h] + dt_bias[h])
    /// ```
    ///
    /// with torch's `softplus` at its defaults (linear above 20). `a` is
    /// `[..., H]` (`in_proj_a`'s output), `a_log` and `dt_bias` `[H]`; the
    /// output, the `g` of [`GdnInputs`], is shaped like `a`.
    /// [`crate::gdn_log_decay_forward_dims`] validates the shapes. The
    /// default refuses with [`OjasError::Unsupported`].
    fn gdn_log_decay_forward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let _ = (a, a_log, dt_bias);
        Err(unsupported_hybrid(self.id(), "gdn_log_decay_forward"))
    }

    /// Gradients of [`Self::gdn_log_decay_forward`] for `grad_output` shaped
    /// like `a`: `da = grad * -exp(a_log) * softplus'(a + dt_bias)`, `ddt_bias`
    /// the sum of `da` over rows, and `da_log` the sum of `grad * g`.
    fn gdn_log_decay_backward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        grad_output: &Tensor,
    ) -> Result<GdnDecayGrad, OjasError> {
        let _ = (a, a_log, dt_bias, grad_output);
        Err(unsupported_hybrid(self.id(), "gdn_log_decay_backward"))
    }

    /// 2D convolution over `NCHW` images: `input` `[N, C, H, W]`, `weight`
    /// `[O, C / groups, kh, kw]` (PyTorch's layout) and an optional `bias`
    /// `[O]`, giving `[N, O, Ho, Wo]`. A group is an im2col matrix times
    /// the group's weight, so patch embedding, 1x1 and strided 3x3
    /// convolutions are one GEMM per image; `groups == C` is the depthwise
    /// convolution. Each output sums its taps in `(channel, kh, kw)` order
    /// and adds the bias last.
    /// [`crate::conv2d_forward_dims`] validates the shapes. The default
    /// refuses with [`OjasError::Unsupported`].
    fn conv2d_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        params: Conv2dParams,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, weight, bias, params);
        Err(unsupported_spatial(self.id(), "conv2d_forward"))
    }

    /// Transposed 2D convolution (`torch.nn.ConvTranspose2d`, one group):
    /// `input` `[N, C, H, W]`, `weight` `[C, O, kh, kw]`, an optional `bias`
    /// `[O]`, giving `[N, O, Ho, Wo]`. Each input pixel scatters a
    /// weighted `kh x kw` patch; an output pixel sums its contributions in
    /// ascending `(kh, kw)` order, then adds the bias.
    /// [`crate::conv_transpose2d_forward_dims`] validates the shapes. The
    /// default refuses with [`OjasError::Unsupported`].
    fn conv_transpose2d_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        params: ConvTranspose2dParams,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, weight, bias, params);
        Err(unsupported_spatial(self.id(), "conv_transpose2d_forward"))
    }

    /// `torch.nn.functional.max_pool2d` without padding or dilation over
    /// `[N, C, H, W]`. [`crate::max_pool2d_forward_dims`] validates the
    /// shapes. The default refuses with [`OjasError::Unsupported`].
    fn max_pool2d_forward(
        &self,
        input: &Tensor,
        kernel: [usize; 2],
        stride: [usize; 2],
    ) -> Result<Tensor, OjasError> {
        let _ = (input, kernel, stride);
        Err(unsupported_spatial(self.id(), "max_pool2d_forward"))
    }

    /// `torch.nn.functional.interpolate(mode="bilinear")` of `[N, C, H, W]`
    /// to `out_size = [out_h, out_w]`, without antialiasing. With
    /// `align_corners` false, source coordinate `(dst + 0.5) * in / out -
    /// 0.5` clamped at 0; with it true, `dst * (in - 1) / (out - 1)`.
    /// [`crate::interpolate_bilinear_forward_dims`] validates the shapes.
    /// The default refuses with [`OjasError::Unsupported`].
    fn interpolate_bilinear_forward(
        &self,
        input: &Tensor,
        out_size: [usize; 2],
        align_corners: bool,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, out_size, align_corners);
        Err(unsupported_spatial(
            self.id(),
            "interpolate_bilinear_forward",
        ))
    }

    /// RoPE that rotates adjacent pairs `(x[2i], x[2i + 1])`, the complex
    /// multiplication of SAM2 and SAM3's `apply_rotary_enc`, with no norm:
    ///
    /// ```text
    /// out[2i]     = x[2i] * cos[t, i] - x[2i + 1] * sin[t, i]
    /// out[2i + 1] = x[2i] * sin[t, i] + x[2i + 1] * cos[t, i]
    /// ```
    ///
    /// `x` is `[B, T, H, D]` with `D` even and the tables `[T, D / 2]`.
    /// [`crate::axial_rope_2d_tables`] builds the 2D axial tables; any
    /// position scheme works. [`crate::rope_interleaved_forward_dims`]
    /// validates the shapes. The default refuses with
    /// [`OjasError::Unsupported`].
    fn rope_interleaved_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let _ = (x, cos, sin);
        Err(unsupported_spatial(self.id(), "rope_interleaved_forward"))
    }

    /// Gradient of [`Self::rope_interleaved_forward`]: the rotation by
    /// `-angle`.
    fn rope_interleaved_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let _ = (grad_output, cos, sin);
        Err(unsupported_spatial(self.id(), "rope_interleaved_backward"))
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError>;

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError>;

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError>;

    /// `(grad_a, grad_b)`.
    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError>;

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError>;

    /// `(grad_x, grad_y)` for an unbroadcast add.
    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError>;

    /// `LayerNorm(x) * weight + bias` over the last axis of `input`
    /// (`[..., dim]`, `weight` and `bias` `[dim]`): torch's
    /// `F.layer_norm`. Each row is centred on its mean, divided by
    /// `sqrt(var + eps)` with the biased variance (divisor `dim`), then
    /// scaled and shifted. The mean and variance are two-pass `f64` sums
    /// that ascend from index 0 (the deviations are taken from the `f64`
    /// mean), so a row whose mean is far from zero keeps its accuracy and the
    /// value does not depend on the thread count; a `var + eps` that is not
    /// finite and positive is [`OjasError::NonFinite`].
    /// [`crate::layer_norm_forward_dims`] validates. The default refuses with
    /// [`OjasError::Unsupported`].
    fn layer_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, weight, bias, eps);
        Err(unsupported_inference(self.id(), "layer_norm_forward"))
    }

    /// LayerNorm across the channel axis of an image-shaped tensor:
    /// `input` `[batch, channels, *spatial]` (`[N, C, H, W]`), `weight` and
    /// `bias` `[channels]`. Every position `(n, *spatial)` is normalized
    /// over its `channels` values as [`Self::layer_norm_forward`] does, then
    /// `weight[c]` and `bias[c]` scale and shift channel `c`. This is
    /// SAM's `LayerNorm2d`, and it equals permuting to channels-last,
    /// `layer_norm_forward`, and permuting back, without the two copies.
    /// [`crate::channel_layer_norm_forward_dims`] validates. The default
    /// refuses with [`OjasError::Unsupported`].
    fn channel_layer_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, weight, bias, eps);
        Err(unsupported_inference(
            self.id(),
            "channel_layer_norm_forward",
        ))
    }

    /// GroupNorm over `input` `[N, C, H, W]`, torch's `nn.GroupNorm`: the `C`
    /// channels split into `groups` runs of `C / groups`, and each run of
    /// every image is normalized over its channels and all `H * W`
    /// positions with the biased variance, `(x - mean) / sqrt(var + eps)`,
    /// then channel `c` is scaled and shifted by `weight[c]` and `bias[c]`
    /// (both `[C]`). `groups == C` is instance norm and `groups == 1`
    /// normalizes the whole image. `eps` must be finite and above 0.
    /// [`crate::group_norm_forward_dims`] validates. The default refuses
    /// with [`OjasError::Unsupported`].
    fn group_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        groups: usize,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, weight, bias, groups, eps);
        Err(unsupported_inference(self.id(), "group_norm_forward"))
    }

    /// The exact GELU `x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, torch's
    /// `F.gelu(x, approximate="none")`, not the tanh approximation. On the
    /// CPU it is [`crate::gelu_erf_exact`], the same bits on every platform.
    /// [`crate::gelu_erf_forward_dims`] validates. The default refuses with
    /// [`OjasError::Unsupported`].
    fn gelu_erf_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let _ = input;
        Err(unsupported_inference(self.id(), "gelu_erf_forward"))
    }

    /// `max(x, 0)`, torch's `F.relu`: `x` where `x >= 0` and `+0` where
    /// `x < 0`, so every output is an input value or zero and no rounding
    /// happens. [`crate::relu_forward_dims`] validates. The default refuses
    /// with [`OjasError::Unsupported`].
    fn relu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let _ = input;
        Err(unsupported_inference(self.id(), "relu_forward"))
    }

    /// Join `inputs` along `axis`, torch's `torch.cat(inputs, axis)`: every
    /// input has the first one's shape except on `axis`, and the output's
    /// extent there is their sum. Values move without arithmetic, so an
    /// output's bits are the inputs' bits; as for [`Self::permute`], a
    /// non-finite input is [`OjasError::NonFinite`], not passed through. The
    /// output stays on this backend. [`crate::concat_dims`] validates. The
    /// default refuses with [`OjasError::Unsupported`].
    fn concat(&self, inputs: &[&Tensor], axis: usize) -> Result<Tensor, OjasError> {
        let _ = (inputs, axis);
        Err(unsupported_inference(self.id(), "concat"))
    }

    /// The rows `start..start + len` of `input` along `axis`, torch's
    /// `input.narrow(axis, start, len).contiguous()`: a copy, `len` at least
    /// 1 and `start + len` within the axis. Values move without arithmetic,
    /// as for [`Self::concat`]. [`crate::slice_dims`] validates. The default
    /// refuses with [`OjasError::Unsupported`].
    fn slice(
        &self,
        input: &Tensor,
        axis: usize,
        start: usize,
        len: usize,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, axis, start, len);
        Err(unsupported_inference(self.id(), "slice"))
    }

    /// A new tensor of `shape` and `dtype` with every element `value`, torch's
    /// `torch.full(shape, value, dtype=dtype)`, made on this backend: no host
    /// buffer is uploaded. `F32` stores `value`'s bits; `Bf16` takes only a
    /// value a bf16 holds exactly and `U32` only an integer in `0..2^32`, and
    /// anything else is [`OjasError::OutOfRange`], never rounded; a NaN or an
    /// infinity is [`OjasError::NonFinite`], as no op takes or stores one.
    /// The output is the only charge. [`crate::fill_dims`] validates. The
    /// default refuses with [`OjasError::Unsupported`].
    fn fill(&self, shape: &[usize], dtype: DType, value: f32) -> Result<Tensor, OjasError> {
        let _ = (shape, dtype, value);
        Err(unsupported_inference(self.id(), "fill"))
    }

    /// `y = x @ W^T + b` for `x` `[..., in]`, `W` `[out, in]` and an `F32`
    /// `bias` `[out]`: [`Self::linear_forward`] and then one `f32` add of the
    /// bias, so the product is the `linear_forward` bits and the bias adds
    /// one rounding. `x` and `W` follow `linear_forward`'s `F32` or `Bf16`
    /// rules. [`crate::linear_bias_forward_dims`] validates. The default
    /// refuses with [`OjasError::Unsupported`].
    fn linear_bias_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, OjasError> {
        let _ = (input, weight, bias);
        Err(unsupported_inference(self.id(), "linear_bias_forward"))
    }

    /// `a + b` with numpy broadcasting ([`crate::broadcast_shapes`]): shapes
    /// aligned at the last axis, each pair of extents equal or one of them
    /// 1, rank at most [`crate::MAX_BROADCAST_RANK`]. One `f32` add per
    /// output element. Unlike [`Self::residual_add_forward`] the shapes may
    /// differ. The default refuses with [`OjasError::Unsupported`].
    fn broadcast_add_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        let _ = (a, b);
        Err(unsupported_inference(self.id(), "broadcast_add_forward"))
    }

    /// `a * b` with numpy broadcasting, as [`Self::broadcast_add_forward`]:
    /// one `f32` multiply per output element. The default refuses with
    /// [`OjasError::Unsupported`].
    fn broadcast_mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        let _ = (a, b);
        Err(unsupported_inference(self.id(), "broadcast_mul_forward"))
    }

    /// `input * scale` into a new tensor, for a finite `scale`
    /// ([`OjasError::NonFinite`] otherwise): one `f32` multiply per element.
    /// [`Self::scale_grad`] is the in-place gradient form. The default
    /// refuses with [`OjasError::Unsupported`].
    fn scale_forward(&self, input: &Tensor, scale: f32) -> Result<Tensor, OjasError> {
        let _ = (input, scale);
        Err(unsupported_inference(self.id(), "scale_forward"))
    }

    /// Softmax over the last axis of `input` (`[..., dim]`):
    /// `e^(x - max) / sum(e^(x - max))`. The row maximum is exact. On the
    /// CPU the exponential is [`crate::exp_exact`] under [`Numerics::Exact`]
    /// and the sum ascends from index 0 in `f32`, so a row of `dim` equal
    /// values gives `1 / dim` each; under [`Numerics::Fast`] it is the
    /// vector exponential with fixed lane sums. Neither depends on the
    /// thread count. To softmax another axis, [`Self::permute`] first. The
    /// default refuses with [`OjasError::Unsupported`].
    fn softmax_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        let _ = input;
        Err(unsupported_inference(self.id(), "softmax_forward"))
    }

    /// Scalar mean cross-entropy. Targets are `U32`.
    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError>;

    /// Gradient w.r.t. logits. The denominator is the valid-row count. An
    /// all-ignored batch returns [`OjasError::NonFinite`] and does not write.
    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError>;

    /// In-place global-norm clip of `grads`. Returns the total norm.
    ///
    /// Scale is `min(1, max_norm / (total_norm + CLIP_GRAD_NORM_EPS))`.
    /// A non-finite norm returns [`OjasError::NonFinite`] and does not
    /// scale. `max_norm` is nanolab's `grad_clip` (default 1.0).
    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError>;

    /// One parameter. `step` is the count before this update; the body calls
    /// [`next_step`]. Gradient is already clipped. This method does not clip again.
    fn adamw_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        moment1: &mut Tensor,
        moment2: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError>;

    /// One matrix. Newton-Schulz runs at `config.ns5` ([`Ns5Precision`]). A
    /// backend without a [`Ns5Precision::Bf16`] path refuses it with
    /// [`OjasError::Unsupported`] before it charges or writes anything. `step`
    /// uses [`next_step`]. Call [`require_ns5`] if a caller can pass a step
    /// count.
    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError>;

    /// The most bytes one [`Backend::adamw_step`] or [`Backend::muon_ns5_step`]
    /// on a `[rows, cols]` parameter charges to this backend's budget beyond
    /// its operands (a vector is `[1, len]`). `None`: this backend does not
    /// say.
    ///
    /// A training loop checks this room before its first optimizer call, so
    /// a refusal arrives while every parameter is still unchanged. It is an
    /// upper bound on what the step itself reserves, never below it; each
    /// backend derives both from one sizing rule, and a test per backend
    /// compares the figure with the budget's measured peak.
    fn optimizer_scratch_bytes(
        &self,
        kind: OptimizerKind,
        rows: usize,
        cols: usize,
    ) -> Result<Option<u64>, OjasError> {
        let _ = (kind, rows, cols);
        Ok(None)
    }

    /// Wait for every op this backend has submitted, then report any fault
    /// it deferred.
    ///
    /// A backend that reports every fault from the op that caused it (CPU)
    /// keeps this default. A backend that defers non-finite faults to a later
    /// call (Metal and wgpu; see `docs/metal-deferred-faults.md`) must override
    /// it so the fault surfaces here as [`OjasError::NonFinite`]. A training loop calls `sync` after
    /// its optimizer calls and before it advances its step counter.
    fn sync(&self) -> Result<(), OjasError> {
        Ok(())
    }

    /// `acc += grad`, elementwise, same shape and dtype.
    ///
    /// On success `acc` holds the sum and is uniquely owned. A non-finite
    /// sum is [`OjasError::NonFinite`], reported as [`Backend::sync`]
    /// describes: a backend that reports synchronously leaves `acc`
    /// unchanged; on a deferring backend, `acc` must be treated as invalid
    /// once `sync` reports the fault. Shapes are checked by
    /// [`crate::accumulate_grad_dims`] first, so a refusal names
    /// `accumulate_grad` on every backend. The default then composes
    /// [`Backend::residual_add_forward`]; a backend may add in place.
    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        crate::accumulate_grad_dims(acc, grad)?;
        *acc = self.residual_add_forward(acc, grad)?;
        Ok(())
    }

    /// `grad *= scale`, for a gradient its caller owns: the loss seed a
    /// tape applies to a loss's unit gradient when the walk was not seeded
    /// with exactly 1 (a trainer averaging K micro-batches seeds `1 / K`).
    ///
    /// Each value is one rounding of `value * scale`. A non-finite `scale`,
    /// input or product is [`OjasError::NonFinite`], reported as
    /// [`Backend::sync`] describes, and `grad`'s values are then
    /// unspecified: the caller discards it. A backend that overrides this
    /// writes a uniquely owned `grad` in place and replaces a shared one
    /// with a new tensor (the other handles keep the old values). The
    /// default replaces `grad` with [`Backend::mul_forward`] of `grad` and
    /// `scale` repeated over its shape ([`broadcast_scalar`]), built on the
    /// backend from one uploaded value, so nothing is read back.
    fn scale_grad(&self, grad: &mut Tensor, scale: f32) -> Result<(), OjasError> {
        if !scale.is_finite() {
            return Err(OjasError::NonFinite { op: "scale_grad" });
        }
        let scalar = place_host(self, Tensor::from_f32(&[scale], &[1], self.budget())?)?;
        let full = broadcast_scalar(self, &scalar, grad.shape())?;
        *grad = self.mul_forward(grad, &full)?;
        Ok(())
    }

    /// Mean cross-entropy of `input @ weight^T` without materializing the
    /// `[N, V]` logits, plus the gradients `grads` asks for: each of
    /// [`LinearCe`]'s gradients is present exactly when its flag is set, and
    /// one that is not asked for is neither computed nor charged (a frozen
    /// tied head takes no `[V, d]` gradient).
    ///
    /// `input` is `[N, d]` `F32`, `weight` is `[V, d]` `F32` or `Bf16` (the
    /// tied embedding, a bf16-operand slot of [`Self::bf16_operands`]),
    /// `targets` is `[N]` `U32`. A `Bf16` weight is widened exactly one
    /// vocabulary chunk at a time (`chunk.cols * d` values, charged as
    /// scratch), never as a whole-table `f32` copy, and every gradient is
    /// `F32`. Values equal
    /// [`Backend::linear_forward`] → [`Backend::cross_entropy_mean_forward`],
    /// and the gradients equal [`Backend::cross_entropy_mean_backward`] →
    /// [`Backend::linear_backward`] at seed 1, within the backend's
    /// [`Numerics`]. The valid-row count is global, not per chunk; an
    /// all-ignored batch is [`OjasError::NonFinite`]. Logit scratch must stay
    /// within `chunk.rows * chunk.cols` elements: an implementation that
    /// allocates `N * V` logits does not meet this contract. Validate with
    /// [`linear_ce_dims`].
    fn linear_cross_entropy_mean(
        &self,
        input: &Tensor,
        weight: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
        chunk: CeChunk,
        grads: CeGrads,
    ) -> Result<LinearCe, OjasError> {
        let _ = (input, weight, targets, ignore_index, chunk, grads);
        Err(OjasError::Unsupported {
            op: "linear_cross_entropy_mean",
            detail: format!(
                "{:?} backend does not implement linear_cross_entropy_mean",
                self.id()
            ),
        })
    }

    /// Causal attention of `Tq` new queries against the `kv_len` positions
    /// written to a KV cache, for decode and for prefill onto a non-empty
    /// cache.
    ///
    /// `q` is `[B, Tq, H, D]`; `k_cache` and `v_cache` are `[B, Tcap, Hkv, D]`,
    /// time-major like `ojas-infer`'s host cache, and a ring: position `j`
    /// is slot `j % Tcap` (as [`Backend::kv_cache_write`] puts it), so a
    /// cache never written past `Tcap` is a plain prefix. Query `i` sits at
    /// position `p = kv_len - Tq + i` and attends to positions `0..=p`, or
    /// with a `window` to `p + 1 - window..=p` (as
    /// [`Backend::causal_sdpa_forward`]'s window), visited oldest first.
    /// Head `h` reads KV head `h / (H / Hkv)` (grouped-query attention).
    /// Scale is [`sdpa_scale`]`(D)`. Returns `[B, Tq, H, D]`. With `kv_len
    /// == Tq` and `H == Hkv` this equals [`Backend::causal_sdpa_forward`]
    /// with the same window after the layout permute. Validate with
    /// [`cached_attention_dims`], which refuses a read the ring no longer
    /// holds; [`KvDims::visible`] gives each query's positions.
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
        window: Option<usize>,
    ) -> Result<Tensor, OjasError> {
        let _ = (q, k_cache, v_cache, kv_len, window);
        Err(OjasError::Unsupported {
            op: "cached_attention_forward",
            detail: format!(
                "{:?} backend does not implement cached_attention_forward",
                self.id()
            ),
        })
    }

    /// Write `src` (`[B, Tn, Hkv, D]`) into `cache` (`[B, Tcap, Hkv, D]`) at
    /// time positions `at..at + Tn`: position `j` goes to slot `j % Tcap`,
    /// so `Tn <= Tcap` and the cache is a ring.
    ///
    /// `cache` must be uniquely owned. On any error the cache is unchanged.
    /// Validate with [`kv_cache_write_dims`].
    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        let _ = (cache, src, at);
        Err(OjasError::Unsupported {
            op: "kv_cache_write",
            detail: format!("{:?} backend does not implement kv_cache_write", self.id()),
        })
    }

    /// The column of each row's largest value: `x` `F32` `[rows, cols]` to
    /// `U32` `[rows]`, ties to the lowest column (greedy decoding's pick).
    /// Any NaN or infinity is [`OjasError::NonFinite`]; a device backend may
    /// report it at the next [`Backend::sync`]. Every id is below `cols`, so
    /// a backend may let the result index an `embedding_forward` table of at
    /// least `cols` rows without reading the ids back. Validate with
    /// [`crate::argmax_rows_dims`].
    ///
    /// The default reads `x` to the host (for a device tensor, a readback
    /// and a [`Backend::sync`]), scans it there, and returns a host tensor
    /// charged to this backend's budget.
    fn argmax_rows(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "argmax_rows";
        let (rows, cols) = crate::argmax_rows_dims(x)?;
        let host = match x.device() {
            None => x.clone(),
            Some(_) => {
                let host = self.download(x)?;
                self.sync()?;
                host
            }
        };
        let values = host.to_f32_vec()?;
        let mut ids = Vec::with_capacity(rows);
        for row in values.chunks_exact(cols) {
            let mut best = 0usize;
            for (i, &v) in row.iter().enumerate() {
                if !v.is_finite() {
                    return Err(OjasError::NonFinite { op: OP });
                }
                if v > row[best] {
                    best = i;
                }
            }
            // `argmax_rows_dims` checked that `cols` fits u32.
            ids.push(best as u32);
        }
        Tensor::from_u32(&ids, &[rows], self.budget())
    }

    /// The `k` leaders of each row of `x` (`F32` `[rows, cols]`, `1 <= k <= cols`),
    /// ranked by [`rank_logits`]: value descending with `+0.0` and `-0.0` one
    /// value and `-inf` after every finite value, then column ascending (the
    /// order `ojas_infer::sample_token` ranks candidates in). Returns
    /// `(values, ids)`: `F32` `[rows, k]` and `U32` `[rows, k]`, every id
    /// `< cols`. Any NaN or `+inf` in a row is
    /// `OjasError::NonFinite` (a device backend may report it at the next sync);
    /// `-inf` is a value (a masked logit). Validate with `topk_rows_dims`. The default
    /// downloads `x` and selects on the host.
    ///
    /// As with [`Backend::argmax_rows`], a backend may let the ids index an
    /// `embedding_forward` table of at least `cols` rows without reading them
    /// back. A device kernel may refuse a large `k` as
    /// [`OjasError::Unsupported`] (Metal and wgpu take `k <= 1024`). The
    /// default's host scan is `O(cols + k log k)` per row (a
    /// selection, then a sort of the `k` leaders), and its two results are
    /// host tensors charged to this backend's budget.
    fn topk_rows(&self, x: &Tensor, k: usize) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "topk_rows";
        let (rows, cols) = crate::topk_rows_dims(x, k)?;
        let host = match x.device() {
            None => x.clone(),
            Some(_) => {
                let host = self.download(x)?;
                self.sync()?;
                host
            }
        };
        let values = host.to_f32_vec()?;
        // `rows * k <= rows * cols`, the element count of `x`.
        let mut top_values = Vec::with_capacity(rows * k);
        let mut top_ids = Vec::with_capacity(rows * k);
        // `topk_rows_dims` checked that `cols` fits u32.
        let mut order: Vec<u32> = Vec::with_capacity(cols);
        for row in values.chunks_exact(cols) {
            if row.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
                return Err(OjasError::NonFinite { op: OP });
            }
            let rank = |a: &u32, b: &u32| rank_logits(row, *a as usize, *b as usize);
            order.clear();
            order.extend(0..cols as u32);
            if k < cols {
                order.select_nth_unstable_by(k - 1, rank);
            }
            let leaders = &mut order[..k];
            leaders.sort_unstable_by(rank);
            top_values.extend(leaders.iter().map(|&i| row[i as usize]));
            top_ids.extend_from_slice(leaders);
        }
        let values = Tensor::from_f32(&top_values, &[rows, k], self.budget())?;
        let ids = Tensor::from_u32(&top_ids, &[rows, k], self.budget())?;
        Ok((values, ids))
    }

    /// Round a host f32 tensor to bf16 and widen it back to f32.
    ///
    /// A non-f32 tensor is [`OjasError::Dtype`]. A device tensor is
    /// [`OjasError::Unsupported`]: this default does not download. An empty
    /// host tensor stays empty. A non-contiguous host tensor fails with the
    /// [`Tensor::f32_slice`] shape error. [`crate::Autocast`] overrides this
    /// to skip a tensor whose compute tag is already bf16.
    fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "cast_bf16";
        if tensor.dtype() != DType::F32 {
            return Err(OjasError::Dtype {
                op: OP,
                expected: DType::F32,
                got: tensor.dtype(),
            });
        }
        refuse_device_cast(OP, tensor)?;
        let src = tensor.f32_slice()?;
        let mut out = Tensor::zeros(tensor.shape(), DType::F32, self.budget())?;
        {
            let dst = out.f32_slice_mut()?;
            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = round_f32_to_bf16(*src);
            }
        }
        Ok(out)
    }

    /// Whether the matmul-class ops take `Bf16` storage as operands.
    ///
    /// The matmul-class ops are [`Self::linear_forward`],
    /// [`Self::linear_backward`], [`Self::causal_sdpa_forward`] and
    /// [`Self::causal_sdpa_backward`] (except its `lse`, which stays `F32`).
    /// Their validators accept each operand as `F32` or `Bf16`, mixed freely.
    /// A bf16 operand is widened exactly to `f32` (`bits << 16`), every
    /// product and sum accumulates in `f32`, and every output is `F32`. So an
    /// operand whose `f32` values are bf16-exact gives the same bits whether
    /// it arrives as `F32` or as `Bf16`, under the same [`Self::numerics`].
    ///
    /// The tied table takes the same rule (decided 2026-10-14): the `table`
    /// of [`Self::embedding_forward`] and [`Self::embedding_backward`] and
    /// the `weight` of [`Self::linear_cross_entropy_mean`]. Their other
    /// operands stay `F32` (`U32` for ids and targets). The table lane reads
    /// bf16 rows in place, a gathered row or one vocabulary chunk at a time,
    /// so a frozen bf16 table never costs an `f32` copy of itself. The CPU
    /// and Metal take it; a backend without the lane refuses a `Bf16` table
    /// with [`OjasError::Dtype`] ([`refuse_bf16_table`]), whatever this flag
    /// says. Every other op refuses `Bf16` with [`OjasError::Dtype`].
    ///
    /// The default is `false`. Such a backend refuses a matmul-class `Bf16`
    /// operand with [`OjasError::Unsupported`] ([`refuse_bf16_operands`])
    /// before it charges or writes anything. [`crate::Autocast`] rounds its operand
    /// copies in `f32` storage on every backend; this flag decides only
    /// whether a `Bf16` tensor the caller holds is accepted.
    fn bf16_operands(&self) -> bool {
        false
    }

    /// A new `Bf16` tensor holding `tensor`'s `F32` values rounded to nearest
    /// even ([`f32_to_bf16`]), on the same backend.
    ///
    /// A `Bf16` input is returned as a shared clone. Any other dtype is
    /// [`OjasError::Dtype`]. A NaN stays a NaN and an infinity stays an
    /// infinity; a finite value above the bf16 range rounds to an infinity,
    /// which the matmul-class ops then refuse as [`OjasError::NonFinite`].
    /// The default converts a contiguous host tensor and refuses a device
    /// tensor with [`OjasError::Unsupported`]; it does not download.
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
        refuse_device_cast(OP, tensor)?;
        let src = tensor.f32_slice()?;
        let mut out = Tensor::zeros(tensor.shape(), DType::Bf16, self.budget())?;
        for (dst, src) in out.bf16_slice_mut()?.iter_mut().zip(src) {
            *dst = f32_to_bf16(*src);
        }
        Ok(out)
    }

    /// A new `F32` tensor holding `tensor`'s `Bf16` values widened exactly
    /// ([`bf16_to_f32`]), on the same backend.
    ///
    /// An `F32` input is returned as a shared clone. Any other dtype is
    /// [`OjasError::Dtype`]. The default converts a contiguous host tensor and
    /// refuses a device tensor with [`OjasError::Unsupported`].
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
        refuse_device_cast(OP, tensor)?;
        let src = tensor.bf16_slice()?;
        let mut out = Tensor::zeros(tensor.shape(), DType::F32, self.budget())?;
        for (dst, src) in out.f32_slice_mut()?.iter_mut().zip(src) {
            *dst = bf16_to_f32(*src);
        }
        Ok(out)
    }

    /// Enter an autocast region.
    ///
    /// [`AutocastMode::Off`] is a no-op guard. [`AutocastMode::Bf16`] is
    /// [`OjasError::Unsupported`] unless the backend is [`crate::Autocast`],
    /// which owns the region stack.
    fn autocast_region(&self, mode: AutocastMode) -> Result<AutocastGuard, OjasError> {
        match mode {
            AutocastMode::Off => Ok(AutocastGuard::noop()),
            AutocastMode::Bf16 => Err(OjasError::Unsupported {
                op: "autocast_region",
                detail: format!(
                    "{:?} backend has no bf16 autocast region; wrap it in Autocast",
                    self.id()
                ),
            }),
        }
    }
}

/// The one list of [`Backend`] methods. Every wrapper that must forward the
/// whole trait (`&B`, `Arc<B>`, `Gated`, the test `Probe`) is generated from
/// it, so a method a wrapper forgets cannot silently run the trait default
/// on the wrapper instead of the wrapped backend's override.
///
/// `with_backend_methods!(callback, ctx)` expands to `callback! { ctx
/// <entries> }`. Each entry is `kind fn name(&self, args..) -> ret;` where
/// `kind` is `op` (a computation, which an admission hook may gate) or `pure`
/// (an accessor or size query that is never gated).
///
/// `backend.rs`'s `trait_and_method_list_agree` test reads this trait's
/// source and fails when a method here is missing from the list, or the list
/// names a method the trait does not have.
#[doc(hidden)]
#[macro_export]
macro_rules! with_backend_methods {
    ($callback:ident, $ctx:tt) => {
        $callback! { $ctx
            pure fn id(&self) -> $crate::BackendId;
            pure fn budget(&self) -> &$crate::Budget;
            pure fn numerics(&self) -> $crate::Numerics;
            op fn upload(&self, tensor: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn download(&self, tensor: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn permute(&self, input: &$crate::Tensor, dims: &[usize]) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn embedding_forward(&self, table: &$crate::Tensor, token_ids: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn embedding_backward(&self, table: &$crate::Tensor, token_ids: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn linear_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn linear_backward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn linear_backward_input(&self, input: &$crate::Tensor, weight: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn rms_norm_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, eps: f32) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn rms_norm_backward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, grad_output: &$crate::Tensor, eps: f32) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn rope_half_split_forward(&self, x: &$crate::Tensor, cos: &$crate::Tensor, sin: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn rope_half_split_backward(&self, grad_output: &$crate::Tensor, cos: &$crate::Tensor, sin: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn rms_qk_norm_forward(&self, q: &$crate::Tensor, k: &$crate::Tensor, q_weight: &$crate::Tensor, k_weight: &$crate::Tensor, eps: f32) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn rms_qk_norm_backward(&self, q: &$crate::Tensor, k: &$crate::Tensor, q_weight: &$crate::Tensor, k_weight: &$crate::Tensor, grad_q: &$crate::Tensor, grad_k: &$crate::Tensor, eps: f32) -> Result<($crate::Tensor, $crate::Tensor, $crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn causal_sdpa_forward(&self, q: &$crate::Tensor, k: &$crate::Tensor, v: &$crate::Tensor, window: Option<usize>) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn causal_sdpa_backward(&self, q: &$crate::Tensor, k: &$crate::Tensor, v: &$crate::Tensor, output: &$crate::Tensor, lse: &$crate::Tensor, grad_output: &$crate::Tensor, window: Option<usize>) -> Result<($crate::Tensor, $crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn causal_sdpa_backward_recompute(&self, q: &$crate::Tensor, k: &$crate::Tensor, v: &$crate::Tensor, grad_output: &$crate::Tensor, window: Option<usize>) -> Result<($crate::Tensor, $crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn attention_forward(&self, q: &$crate::Tensor, k: &$crate::Tensor, v: &$crate::Tensor, opts: $crate::AttentionOpts<'_>) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn attention_forward_saving(&self, q: &$crate::Tensor, k: &$crate::Tensor, v: &$crate::Tensor, opts: $crate::AttentionOpts<'_>) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn attention_backward(&self, q: &$crate::Tensor, k: &$crate::Tensor, v: &$crate::Tensor, output: &$crate::Tensor, lse: &$crate::Tensor, grad_output: &$crate::Tensor, opts: $crate::AttentionOpts<'_>, bias_grad: bool) -> Result<$crate::AttentionGrad, $crate::OjasError>;
            op fn per_head_sigmoid_gate_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor, attn_out: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn per_head_sigmoid_gate_backward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor, attn_out: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<$crate::PerHeadGateGrad, $crate::OjasError>;
            op fn per_head_sigmoid_gate_forward_saving(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor, attn_out: &$crate::Tensor) -> Result<($crate::Tensor, Option<$crate::Tensor>), $crate::OjasError>;
            op fn per_head_sigmoid_gate_backward_saved(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor, attn_out: &$crate::Tensor, grad_output: &$crate::Tensor, scales: &$crate::Tensor) -> Result<$crate::PerHeadGateGrad, $crate::OjasError>;
            op fn value_residual_blend_forward(&self, value: &$crate::Tensor, value0: &$crate::Tensor, lambda: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn value_residual_blend_backward(&self, value: &$crate::Tensor, value0: &$crate::Tensor, lambda: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<$crate::ValueResidualGrad, $crate::OjasError>;
            op fn chunked_gdn_forward(&self, inputs: $crate::GdnInputs<'_>) -> Result<$crate::GdnForward, $crate::OjasError>;
            op fn chunked_gdn_backward(&self, inputs: $crate::GdnInputs<'_>, checkpoints: &$crate::Tensor, grad_output: &$crate::Tensor, grad_final_state: Option<&$crate::Tensor>) -> Result<$crate::GdnGrad, $crate::OjasError>;
            op fn causal_conv1d_silu_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn causal_conv1d_silu_backward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn gated_rms_norm_forward(&self, input: &$crate::Tensor, gate: &$crate::Tensor, weight: &$crate::Tensor, eps: f32) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn gated_rms_norm_backward(&self, input: &$crate::Tensor, gate: &$crate::Tensor, weight: &$crate::Tensor, grad_output: &$crate::Tensor, eps: f32) -> Result<$crate::GatedRmsGrad, $crate::OjasError>;
            op fn rope_partial_forward(&self, x: &$crate::Tensor, cos: &$crate::Tensor, sin: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn rope_partial_backward(&self, grad_output: &$crate::Tensor, cos: &$crate::Tensor, sin: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn sigmoid_forward(&self, input: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn sigmoid_backward(&self, input: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn gdn_log_decay_forward(&self, a: &$crate::Tensor, a_log: &$crate::Tensor, dt_bias: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn gdn_log_decay_backward(&self, a: &$crate::Tensor, a_log: &$crate::Tensor, dt_bias: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<$crate::GdnDecayGrad, $crate::OjasError>;
            op fn conv2d_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: Option<&$crate::Tensor>, params: $crate::Conv2dParams) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn conv_transpose2d_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: Option<&$crate::Tensor>, params: $crate::ConvTranspose2dParams) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn max_pool2d_forward(&self, input: &$crate::Tensor, kernel: [usize; 2], stride: [usize; 2]) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn interpolate_bilinear_forward(&self, input: &$crate::Tensor, out_size: [usize; 2], align_corners: bool) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn rope_interleaved_forward(&self, x: &$crate::Tensor, cos: &$crate::Tensor, sin: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn rope_interleaved_backward(&self, grad_output: &$crate::Tensor, cos: &$crate::Tensor, sin: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn silu_forward(&self, input: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn silu_backward(&self, input: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn mul_forward(&self, a: &$crate::Tensor, b: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn mul_backward(&self, a: &$crate::Tensor, b: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn residual_add_forward(&self, x: &$crate::Tensor, y: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn residual_add_backward(&self, x: &$crate::Tensor, y: &$crate::Tensor, grad_output: &$crate::Tensor) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn layer_norm_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor, eps: f32) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn channel_layer_norm_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor, eps: f32) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn group_norm_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor, groups: usize, eps: f32) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn gelu_erf_forward(&self, input: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn relu_forward(&self, input: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn concat(&self, inputs: &[&$crate::Tensor], axis: usize) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn slice(&self, input: &$crate::Tensor, axis: usize, start: usize, len: usize) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn fill(&self, shape: &[usize], dtype: $crate::DType, value: f32) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn linear_bias_forward(&self, input: &$crate::Tensor, weight: &$crate::Tensor, bias: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn broadcast_add_forward(&self, a: &$crate::Tensor, b: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn broadcast_mul_forward(&self, a: &$crate::Tensor, b: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn scale_forward(&self, input: &$crate::Tensor, scale: f32) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn softmax_forward(&self, input: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn cross_entropy_mean_forward(&self, logits: &$crate::Tensor, targets: &$crate::Tensor, ignore_index: Option<u32>) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn cross_entropy_mean_backward(&self, logits: &$crate::Tensor, targets: &$crate::Tensor, ignore_index: Option<u32>) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn clip_grad_norm(&self, grads: &mut [$crate::Tensor], max_norm: f32) -> Result<f32, $crate::OjasError>;
            op fn adamw_step(&self, param: &mut $crate::Tensor, grad: &$crate::Tensor, moment1: &mut $crate::Tensor, moment2: &mut $crate::Tensor, step: u64, config: $crate::AdamWConfig) -> Result<(), $crate::OjasError>;
            op fn muon_ns5_step(&self, param: &mut $crate::Tensor, grad: &$crate::Tensor, momentum: &mut $crate::Tensor, config: $crate::MuonNs5Config) -> Result<(), $crate::OjasError>;
            pure fn optimizer_scratch_bytes(&self, kind: $crate::OptimizerKind, rows: usize, cols: usize) -> Result<Option<u64>, $crate::OjasError>;
            op fn sync(&self) -> Result<(), $crate::OjasError>;
            op fn accumulate_grad(&self, acc: &mut $crate::Tensor, grad: &$crate::Tensor) -> Result<(), $crate::OjasError>;
            op fn scale_grad(&self, grad: &mut $crate::Tensor, scale: f32) -> Result<(), $crate::OjasError>;
            op fn linear_cross_entropy_mean(&self, input: &$crate::Tensor, weight: &$crate::Tensor, targets: &$crate::Tensor, ignore_index: Option<u32>, chunk: $crate::CeChunk, grads: $crate::CeGrads) -> Result<$crate::LinearCe, $crate::OjasError>;
            op fn cached_attention_forward(&self, q: &$crate::Tensor, k_cache: &$crate::Tensor, v_cache: &$crate::Tensor, kv_len: usize, window: Option<usize>) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn kv_cache_write(&self, cache: &mut $crate::Tensor, src: &$crate::Tensor, at: usize) -> Result<(), $crate::OjasError>;
            op fn argmax_rows(&self, x: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn topk_rows(&self, x: &$crate::Tensor, k: usize) -> Result<($crate::Tensor, $crate::Tensor), $crate::OjasError>;
            op fn cast_bf16(&self, tensor: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            pure fn bf16_operands(&self) -> bool;
            op fn to_bf16(&self, tensor: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn to_f32(&self, tensor: &$crate::Tensor) -> Result<$crate::Tensor, $crate::OjasError>;
            op fn autocast_region(&self, mode: $crate::AutocastMode) -> Result<$crate::AutocastGuard, $crate::OjasError>;
        }
    };
}

/// Every name in [`with_backend_methods!`], in trait order.
#[doc(hidden)]
pub use backend_method_names::BACKEND_METHODS;

mod backend_method_names {
    macro_rules! names {
        ({} $($kind:ident fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty;)*) => {
            pub const BACKEND_METHODS: &[&str] = &[$(stringify!($name)),*];
        };
    }
    with_backend_methods!(names, {});
}

/// Generates forwarding impls of every [`Backend`] method from
/// [`with_backend_methods!`]. The context is
/// `{ self_ident, inner_expr, admit, hand }`:
///
/// - `inner_expr` is the backend each method forwards to (`(**self)`,
///   `self.inner`);
/// - `admit` is `none`, or the name of a `&self` method taking the op name
///   and returning `Result<(), OjasError>`; it runs before every `op`
///   method and a refusal returns before the inner backend is touched;
/// - `hand` is a macro `(name, { item.. })` that emits the generated `item`
///   or, for the methods the wrapper writes itself, nothing.
///   [`forward_backend_keep`] emits them all.
#[doc(hidden)]
#[macro_export]
macro_rules! forward_backend_methods {
    ({ $s:ident, $inner:expr, $admit:tt, $hand:ident }
     $($kind:ident fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty;)*) => {
        $(
            $crate::forward_backend_methods!(
                @one $s, $inner, $admit, $hand, $kind, $name, [$($arg: $ty),*], $ret
            );
        )*
    };
    (@one $s:ident, $inner:expr, none, $hand:ident, $kind:ident, $name:ident,
     [$($arg:ident : $ty:ty),*], $ret:ty) => {
        $hand! { $name, {
            #[allow(clippy::too_many_arguments)]
            fn $name(&$s $(, $arg: $ty)*) -> $ret {
                $inner.$name($($arg),*)
            }
        } }
    };
    (@one $s:ident, $inner:expr, $admit:ident, $hand:ident, pure, $name:ident,
     [$($arg:ident : $ty:ty),*], $ret:ty) => {
        $hand! { $name, {
            #[allow(clippy::too_many_arguments)]
            fn $name(&$s $(, $arg: $ty)*) -> $ret {
                $inner.$name($($arg),*)
            }
        } }
    };
    (@one $s:ident, $inner:expr, $admit:ident, $hand:ident, op, $name:ident,
     [$($arg:ident : $ty:ty),*], $ret:ty) => {
        $hand! { $name, {
            #[allow(clippy::too_many_arguments)]
            fn $name(&$s $(, $arg: $ty)*) -> $ret {
                $s.$admit(stringify!($name))?;
                $inner.$name($($arg),*)
            }
        } }
    };
}

/// The `hand` argument of [`forward_backend_methods`] that keeps every
/// generated method.
#[doc(hidden)]
#[macro_export]
macro_rules! forward_backend_keep {
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

/// A shared reference to a backend is a backend, so one backend can serve a
/// `Tape` and a trainer at once.
impl<B: Backend + ?Sized> Backend for &B {
    with_backend_methods!(
        forward_backend_methods,
        { self, (**self), none, forward_backend_keep }
    );
}

/// An `Arc` of a backend is a backend, for backends that are not `Clone`
/// (for example `WgpuBackend`) shared across sessions.
impl<B: Backend + ?Sized> Backend for std::sync::Arc<B> {
    with_backend_methods!(
        forward_backend_methods,
        { self, (**self), none, forward_backend_keep }
    );
}

/// Tiling of [`Backend::linear_cross_entropy_mean`]: at most `rows` rows and
/// `cols` vocabulary columns of logits exist at once.
///
/// Both dimensions matter. At Qwen's vocabulary of 248,320, a row-only chunk
/// of 1,024 rows is 1 GB of `f32` logits; tessl bounds scratch by walking the
/// vocabulary in 8,192-column chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeChunk {
    pub rows: usize,
    pub cols: usize,
}

/// Which gradients [`Backend::linear_cross_entropy_mean`] returns, one flag
/// per operand: [`LinearCe::grad_input`] for `input`, [`LinearCe::grad_weight`]
/// for `weight`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeGrads {
    pub input: bool,
    pub weight: bool,
}

impl CeGrads {
    /// The loss alone.
    pub const NONE: Self = Self {
        input: false,
        weight: false,
    };
    /// Both gradients.
    pub const BOTH: Self = Self {
        input: true,
        weight: true,
    };
    /// Every combination, for a sweep over the op's contract.
    pub const EACH: [Self; 4] = [
        Self::NONE,
        Self {
            input: true,
            weight: false,
        },
        Self {
            input: false,
            weight: true,
        },
        Self::BOTH,
    ];

    /// Whether any gradient is asked for.
    pub fn any(self) -> bool {
        self.input || self.weight
    }
}

/// Loss and, when requested, gradients from [`Backend::linear_cross_entropy_mean`].
///
/// `loss` has the shape [`Backend::cross_entropy_mean_forward`] returns.
/// `grad_input` is `[N, d]` and `grad_weight` is `[V, d]`; each is `Some`
/// exactly when its [`CeGrads`] flag was set.
#[derive(Debug)]
pub struct LinearCe {
    pub loss: Tensor,
    pub grad_input: Option<Tensor>,
    pub grad_weight: Option<Tensor>,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn step_counter_refuses_wrap() {
        assert_eq!(next_step(0).unwrap(), 1);
        assert_eq!(next_step(u64::MAX - 1).unwrap(), u64::MAX);
        assert!(matches!(
            next_step(u64::MAX),
            Err(OjasError::OutOfRange { .. })
        ));
    }

    #[test]
    fn head_dim_edges() {
        assert!(sdpa_scale(0).is_err());
        assert_eq!(sdpa_scale(1).unwrap(), 1.0);
        assert_eq!(sdpa_scale(64).unwrap(), 0.125);
        let tiny = sdpa_scale(u32::MAX).unwrap();
        assert!(tiny.is_finite() && tiny > 0.0);
        for backend in [BackendId::Cpu, BackendId::Metal] {
            assert!(refuse_unsupported_metal_head_dim(backend, 0).is_err());
            assert!(refuse_unsupported_metal_head_dim(backend, 64).is_ok());
            assert!(refuse_unsupported_metal_head_dim(backend, 128).is_ok());
            assert!(refuse_unsupported_metal_head_dim(backend, 256).is_ok());
        }
        assert!(refuse_unsupported_metal_head_dim(BackendId::Cpu, u32::MAX).is_ok());
        assert!(matches!(
            refuse_unsupported_metal_head_dim(BackendId::Metal, 257),
            Err(OjasError::UnsupportedHeadDim {
                head_dim: 257,
                limit: 256
            })
        ));
    }

    use crate::{DType, DeviceBuffer};
    use std::any::Any;
    use std::sync::Arc;

    #[derive(Debug)]
    struct Resident(BackendId);

    impl DeviceBuffer for Resident {
        fn backend(&self) -> BackendId {
            self.0
        }
        fn byte_len(&self) -> usize {
            4
        }
        fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
            Ok(vec![0; len])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Implements only the required methods, so every defaulted method
    /// (`upload`, `permute`, `sync`, the framework ops) is the trait default.
    struct NoUpload {
        id: BackendId,
        budget: Budget,
    }

    fn none(op: &'static str) -> OjasError {
        OjasError::Unsupported {
            op,
            detail: "test double".to_string(),
        }
    }

    impl Backend for NoUpload {
        fn id(&self) -> BackendId {
            self.id
        }
        fn budget(&self) -> &Budget {
            &self.budget
        }
        fn embedding_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(none("embedding_forward"))
        }
        fn embedding_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(none("embedding_backward"))
        }
        fn linear_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(none("linear_forward"))
        }
        fn linear_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(none("linear_backward"))
        }
        fn rms_norm_forward(&self, _: &Tensor, _: &Tensor, _: f32) -> Result<Tensor, OjasError> {
            Err(none("rms_norm_forward"))
        }
        fn rms_norm_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: f32,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(none("rms_norm_backward"))
        }
        fn rope_half_split_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(none("rope_half_split_forward"))
        }
        fn rope_half_split_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(none("rope_half_split_backward"))
        }
        fn rms_qk_norm_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: f32,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(none("rms_qk_norm_forward"))
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
            Err(none("rms_qk_norm_backward"))
        }
        fn causal_sdpa_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: Option<usize>,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(none("causal_sdpa_forward"))
        }
        fn causal_sdpa_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: Option<usize>,
        ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
            Err(none("causal_sdpa_backward"))
        }
        fn per_head_sigmoid_gate_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(none("per_head_sigmoid_gate_forward"))
        }
        fn per_head_sigmoid_gate_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<PerHeadGateGrad, OjasError> {
            Err(none("per_head_sigmoid_gate_backward"))
        }
        fn value_residual_blend_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(none("value_residual_blend_forward"))
        }
        fn value_residual_blend_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<ValueResidualGrad, OjasError> {
            Err(none("value_residual_blend_backward"))
        }
        fn silu_forward(&self, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(none("silu_forward"))
        }
        fn silu_backward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(none("silu_backward"))
        }
        fn mul_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(none("mul_forward"))
        }
        fn mul_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(none("mul_backward"))
        }
        /// A host sum, so the default `accumulate_grad` can be checked end to end.
        fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
            if x.shape() != y.shape() {
                return Err(none("residual_add_forward"));
            }
            let (x_values, y_values) = (x.to_f32_vec()?, y.to_f32_vec()?);
            let sum: Vec<f32> = x_values.iter().zip(&y_values).map(|(a, b)| a + b).collect();
            Tensor::from_f32(&sum, x.shape(), &self.budget)
        }
        fn residual_add_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(none("residual_add_backward"))
        }
        fn cross_entropy_mean_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: Option<u32>,
        ) -> Result<Tensor, OjasError> {
            Err(none("cross_entropy_mean_forward"))
        }
        fn cross_entropy_mean_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: Option<u32>,
        ) -> Result<Tensor, OjasError> {
            Err(none("cross_entropy_mean_backward"))
        }
        fn clip_grad_norm(&self, _: &mut [Tensor], _: f32) -> Result<f32, OjasError> {
            Err(none("clip_grad_norm"))
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
            Err(none("adamw_step"))
        }
        fn muon_ns5_step(
            &self,
            _: &mut Tensor,
            _: &Tensor,
            _: &mut Tensor,
            _: MuonNs5Config,
        ) -> Result<(), OjasError> {
            Err(none("muon_ns5_step"))
        }
    }

    #[test]
    fn default_upload_never_pretends_a_host_tensor_is_on_a_device() {
        let budget = Budget::new(1 << 10);
        let host = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
        let on = |id: BackendId| {
            Tensor::from_device(Arc::new(Resident(id)), &[1], DType::F32, &budget).unwrap()
        };
        let wgpu = NoUpload {
            id: BackendId::Wgpu,
            budget: budget.clone(),
        };
        match wgpu.upload(&host) {
            Err(OjasError::Unsupported { op, detail }) => {
                assert_eq!(op, "Backend::upload");
                assert!(detail.contains("Wgpu"), "{detail}");
                assert!(detail.contains("does not implement upload"), "{detail}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert!(matches!(
            wgpu.upload(&on(BackendId::Metal)),
            Err(OjasError::Placement {
                expected: Some(BackendId::Wgpu),
                found: Some(BackendId::Metal),
                ..
            })
        ));
        let resident = wgpu.upload(&on(BackendId::Wgpu)).unwrap();
        assert_eq!(resident.device(), Some(BackendId::Wgpu));

        let cpu = NoUpload {
            id: BackendId::Cpu,
            budget: budget.clone(),
        };
        let live = budget.live_bytes().unwrap();
        let same = cpu.upload(&host).unwrap();
        assert_eq!(same.device(), None);
        assert_eq!(budget.live_bytes().unwrap(), live, "host on Cpu is a clone");
        assert_eq!(same.to_f32_vec().unwrap(), [1.0, 2.0]);
        assert_eq!(cpu.numerics(), Numerics::Exact);
    }

    #[test]
    fn only_five_newton_schulz_steps() {
        assert!(require_ns5(5).is_ok());
        for steps in [0u32, 4, 6, u32::MAX] {
            assert!(require_ns5(steps).is_err());
        }
    }

    #[test]
    fn default_permute_refuses_instead_of_falling_back() {
        let budget = Budget::new(1 << 10);
        let host = Tensor::from_f32(&[1.0, 2.0], &[1, 2], &budget).unwrap();
        let backend = NoUpload {
            id: BackendId::Metal,
            budget: budget.clone(),
        };
        match backend.permute(&host, &[1, 0]) {
            Err(OjasError::Unsupported { op, .. }) => assert_eq!(op, "permute"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn default_attention_forward_refuses_under_its_own_name() {
        let budget = Budget::new(1 << 10);
        let t = Tensor::from_f32(&[1.0], &[1, 1, 1, 1], &budget).unwrap();
        for id in [BackendId::Wgpu, BackendId::Cuda, BackendId::Hip] {
            let backend = NoUpload {
                id,
                budget: budget.clone(),
            };
            match backend.attention_forward(&t, &t, &t, AttentionOpts::default()) {
                Err(OjasError::Unsupported { op, detail }) => {
                    assert_eq!(op, "attention_forward", "{id:?}");
                    assert!(
                        detail.contains(&format!("{id:?}"))
                            && detail.contains("does not implement attention_forward"),
                        "{detail}"
                    );
                }
                other => panic!("{id:?}: expected Unsupported, got {other:?}"),
            }
            match backend.attention_forward_saving(&t, &t, &t, AttentionOpts::default()) {
                Err(OjasError::Unsupported { op, .. }) => {
                    assert_eq!(op, "attention_forward_saving", "{id:?}");
                }
                other => panic!("{id:?}: expected Unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn pow_u64_matches_repeated_multiplication_and_underflows() {
        // Squaring and repeated multiplication round differently, so compare
        // with a relative tolerance of 64 ulps rather than bits.
        for base in [0.0f64, 0.5, 0.9, 0.95, 0.999, 1.0] {
            let mut naive = 1.0f64;
            for exp in 0..64u64 {
                let fast = pow_u64(base, exp);
                assert!(
                    (fast - naive).abs() <= 64.0 * f64::EPSILON * naive.max(f64::MIN_POSITIVE),
                    "{base}^{exp}: {fast} vs {naive}"
                );
                naive *= base;
            }
        }
        assert_eq!(pow_u64(0.5, 3), 0.125);
        assert_eq!(pow_u64(0.999, u64::MAX), 0.0);
        assert_eq!(pow_u64(1.0, u64::MAX), 1.0);
    }

    #[test]
    fn clip_scale_refuses_nonfinite_and_negative() {
        assert_eq!(clip_scale(1.0, 0.5).unwrap(), 1.0);
        // torch's max_norm 0 zeroes the gradients; it is not "off".
        assert_eq!(clip_scale(0.0, 4.0).unwrap(), 0.0);
        let coef = clip_scale(1.0, 4.0).unwrap();
        assert!(coef < 0.25 && coef > 0.2499);
        for (max_norm, norm) in [
            (f32::NAN, 1.0),
            (1.0, f32::NAN),
            (f32::INFINITY, 1.0),
            (1.0, f32::INFINITY),
        ] {
            assert!(matches!(
                clip_scale(max_norm, norm),
                Err(OjasError::NonFinite { .. })
            ));
        }
        assert!(matches!(
            clip_scale(-1.0, 1.0),
            Err(OjasError::OutOfRange { .. })
        ));
    }

    #[test]
    fn check_adamw_refuses_before_any_update() {
        let ok = AdamWConfig::nanolab(1e-3, 0.1);
        let (next, bc1, bc2) = check_adamw(ok, 0).unwrap();
        assert_eq!(next, 1);
        assert_eq!(bc1, 1.0 - ok.beta1);
        assert_eq!(bc2, 1.0 - ok.beta2);
        assert!(matches!(
            check_adamw(ok, u64::MAX),
            Err(OjasError::OutOfRange { .. })
        ));
        let bad = [
            AdamWConfig { lr: f64::NAN, ..ok },
            AdamWConfig {
                eps: f64::INFINITY,
                ..ok
            },
        ];
        for config in bad {
            assert!(matches!(
                check_adamw(config, 0),
                Err(OjasError::NonFinite { .. })
            ));
        }
        let out_of_range = [
            AdamWConfig { beta1: 1.0, ..ok },
            AdamWConfig { beta2: -0.1, ..ok },
            AdamWConfig { eps: 0.0, ..ok },
            AdamWConfig { lr: -1e-3, ..ok },
            AdamWConfig {
                weight_decay: -0.1,
                ..ok
            },
        ];
        for config in out_of_range {
            assert!(
                matches!(check_adamw(config, 0), Err(OjasError::OutOfRange { .. })),
                "{config:?}"
            );
        }
    }

    /// Overrides every method, required and defaulted, with a distinct
    /// marker, so a wrapper that runs a trait default instead of forwarding
    /// is caught. `autocast.rs` reuses it for [`crate::Autocast`].
    pub(crate) struct Marker {
        pub(crate) budget: Budget,
    }

    fn mark(op: &str) -> OjasError {
        OjasError::Backend {
            id: BackendId::Wgpu,
            detail: format!("marker:{op}"),
        }
    }

    /// One `op` method of [`Marker`]; the `pure` ones are written by hand.
    macro_rules! marker_method {
        ($name:ident, pure, [$($ty:ty),*], $ret:ty) => {};
        ($name:ident, op, [$($ty:ty),*], $ret:ty) => {
            fn $name(&self $(, _: $ty)*) -> $ret {
                Err(mark(stringify!($name)))
            }
        };
    }

    macro_rules! marker_methods {
        ({} $($kind:ident fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty;)*) => {
            $( marker_method!($name, $kind, [$($ty),*], $ret); )*
        };
    }

    impl Backend for Marker {
        fn id(&self) -> BackendId {
            BackendId::Wgpu
        }
        fn budget(&self) -> &Budget {
            &self.budget
        }
        fn numerics(&self) -> Numerics {
            Numerics::Fast
        }
        fn bf16_operands(&self) -> bool {
            true
        }
        fn optimizer_scratch_bytes(
            &self,
            _: OptimizerKind,
            _: usize,
            _: usize,
        ) -> Result<Option<u64>, OjasError> {
            Err(mark("optimizer_scratch_bytes"))
        }
        with_backend_methods!(marker_methods, {});
    }

    /// Call every method through `backend` (a wrapper, dispatched
    /// statically through its own impl) and require each to reach `inner`.
    /// Returns the number of methods checked.
    fn every_call_reaches<B: Backend + ?Sized>(backend: &B, inner: &Marker) -> usize {
        every_call_reaches_except(backend, inner, &[])
    }

    /// [`every_call_reaches`], except that the methods named in `owned`
    /// are ones the wrapper answers itself by design; each must be a method
    /// the harness calls, and must not reach `inner`.
    pub(crate) fn every_call_reaches_except<B: Backend + ?Sized>(
        backend: &B,
        inner: &Marker,
        owned: &[&str],
    ) -> usize {
        let budget = &inner.budget;
        let t = Tensor::from_f32(&[1.0], &[1], budget).unwrap();
        let mut m = t.clone();
        let mut m2 = t.clone();
        let mut m3 = t.clone();
        let adamw = AdamWConfig::nanolab(1e-3, 0.0);
        let muon = MuonNs5Config::nanolab_default();
        let chunk = CeChunk { rows: 1, cols: 1 };
        let gdn = GdnInputs {
            q: &t,
            k: &t,
            v: &t,
            g: &t,
            beta: &t,
            initial_state: None,
        };
        /// The marker the call reached, or `"-"` when it did not.
        fn op<T: std::fmt::Debug>(result: Result<T, OjasError>) -> String {
            match result {
                Err(OjasError::Backend {
                    id: BackendId::Wgpu,
                    detail,
                }) if detail.starts_with("marker:") => detail["marker:".len()..].to_string(),
                _ => "-".to_string(),
            }
        }
        let reached = [
            op(backend.upload(&t)),
            op(backend.download(&t)),
            op(backend.permute(&t, &[0])),
            op(backend.embedding_forward(&t, &t)),
            op(backend.embedding_backward(&t, &t, &t)),
            op(backend.linear_forward(&t, &t)),
            op(backend.linear_backward(&t, &t, &t)),
            op(backend.linear_backward_input(&t, &t, &t)),
            op(backend.rms_norm_forward(&t, &t, 1e-6)),
            op(backend.rms_norm_backward(&t, &t, &t, 1e-6)),
            op(backend.rope_half_split_forward(&t, &t, &t)),
            op(backend.rope_half_split_backward(&t, &t, &t)),
            op(backend.rms_qk_norm_forward(&t, &t, &t, &t, 1e-6)),
            op(backend.rms_qk_norm_backward(&t, &t, &t, &t, &t, &t, 1e-6)),
            op(backend.causal_sdpa_forward(&t, &t, &t, None)),
            op(backend.causal_sdpa_backward(&t, &t, &t, &t, &t, &t, None)),
            op(backend.causal_sdpa_backward_recompute(&t, &t, &t, &t, None)),
            op(backend.attention_forward(&t, &t, &t, AttentionOpts::default())),
            op(backend
                .attention_forward_saving(&t, &t, &t, AttentionOpts::default())
                .map(drop)),
            op(backend
                .attention_backward(&t, &t, &t, &t, &t, &t, AttentionOpts::default(), false)
                .map(drop)),
            op(backend.per_head_sigmoid_gate_forward(&t, &t, &t, &t)),
            op(backend.per_head_sigmoid_gate_backward(&t, &t, &t, &t, &t)),
            op(backend
                .per_head_sigmoid_gate_forward_saving(&t, &t, &t, &t)
                .map(drop)),
            op(backend.per_head_sigmoid_gate_backward_saved(&t, &t, &t, &t, &t, &t)),
            op(backend.value_residual_blend_forward(&t, &t, &t)),
            op(backend.value_residual_blend_backward(&t, &t, &t, &t)),
            op(backend.chunked_gdn_forward(gdn).map(drop)),
            op(backend.chunked_gdn_backward(gdn, &t, &t, None).map(drop)),
            op(backend.causal_conv1d_silu_forward(&t, &t)),
            op(backend.causal_conv1d_silu_backward(&t, &t, &t)),
            op(backend.gated_rms_norm_forward(&t, &t, &t, 1e-6)),
            op(backend.gated_rms_norm_backward(&t, &t, &t, &t, 1e-6)),
            op(backend.rope_partial_forward(&t, &t, &t)),
            op(backend.rope_partial_backward(&t, &t, &t)),
            op(backend.sigmoid_forward(&t)),
            op(backend.sigmoid_backward(&t, &t)),
            op(backend.gdn_log_decay_forward(&t, &t, &t)),
            op(backend.gdn_log_decay_backward(&t, &t, &t, &t).map(drop)),
            op(backend.conv2d_forward(&t, &t, None, Conv2dParams::new())),
            op(backend.conv_transpose2d_forward(&t, &t, None, ConvTranspose2dParams::new())),
            op(backend.max_pool2d_forward(&t, [2, 2], [2, 2])),
            op(backend.interpolate_bilinear_forward(&t, [2, 2], false)),
            op(backend.rope_interleaved_forward(&t, &t, &t)),
            op(backend.rope_interleaved_backward(&t, &t, &t)),
            op(backend.silu_forward(&t)),
            op(backend.silu_backward(&t, &t)),
            op(backend.mul_forward(&t, &t)),
            op(backend.mul_backward(&t, &t, &t)),
            op(backend.residual_add_forward(&t, &t)),
            op(backend.residual_add_backward(&t, &t, &t)),
            op(backend.layer_norm_forward(&t, &t, &t, 1e-6)),
            op(backend.channel_layer_norm_forward(&t, &t, &t, 1e-6)),
            op(backend.group_norm_forward(&t, &t, &t, 1, 1e-5)),
            op(backend.gelu_erf_forward(&t)),
            op(backend.relu_forward(&t)),
            op(backend.concat(&[&t, &t], 0)),
            op(backend.slice(&t, 0, 0, 1)),
            op(backend.fill(&[1], DType::F32, 0.0)),
            op(backend.linear_bias_forward(&t, &t, &t)),
            op(backend.broadcast_add_forward(&t, &t)),
            op(backend.broadcast_mul_forward(&t, &t)),
            op(backend.scale_forward(&t, 0.5)),
            op(backend.softmax_forward(&t)),
            op(backend.cross_entropy_mean_forward(&t, &t, None)),
            op(backend.cross_entropy_mean_backward(&t, &t, None)),
            op(backend.clip_grad_norm(std::slice::from_mut(&mut m), 1.0)),
            op(backend.adamw_step(&mut m, &t, &mut m2, &mut m3, 0, adamw)),
            op(backend.muon_ns5_step(&mut m, &t, &mut m2, muon)),
            op(backend.optimizer_scratch_bytes(OptimizerKind::MuonNs5, 1, 1)),
            op(backend.sync()),
            op(backend.accumulate_grad(&mut m, &t)),
            op(backend.scale_grad(&mut m, 0.5)),
            op(backend.linear_cross_entropy_mean(&t, &t, &t, None, chunk, CeGrads::BOTH)),
            op(backend.cached_attention_forward(&t, &t, &t, 1, None)),
            op(backend.kv_cache_write(&mut m, &t, 0)),
            op(backend.argmax_rows(&t)),
            op(backend.topk_rows(&t, 1)),
            op(backend.cast_bf16(&t)),
            op(backend.to_bf16(&t)),
            op(backend.to_f32(&t)),
            op(backend.autocast_region(AutocastMode::Off).map(drop)),
        ];
        let expected = [
            "upload",
            "download",
            "permute",
            "embedding_forward",
            "embedding_backward",
            "linear_forward",
            "linear_backward",
            "linear_backward_input",
            "rms_norm_forward",
            "rms_norm_backward",
            "rope_half_split_forward",
            "rope_half_split_backward",
            "rms_qk_norm_forward",
            "rms_qk_norm_backward",
            "causal_sdpa_forward",
            "causal_sdpa_backward",
            "causal_sdpa_backward_recompute",
            "attention_forward",
            "attention_forward_saving",
            "attention_backward",
            "per_head_sigmoid_gate_forward",
            "per_head_sigmoid_gate_backward",
            "per_head_sigmoid_gate_forward_saving",
            "per_head_sigmoid_gate_backward_saved",
            "value_residual_blend_forward",
            "value_residual_blend_backward",
            "chunked_gdn_forward",
            "chunked_gdn_backward",
            "causal_conv1d_silu_forward",
            "causal_conv1d_silu_backward",
            "gated_rms_norm_forward",
            "gated_rms_norm_backward",
            "rope_partial_forward",
            "rope_partial_backward",
            "sigmoid_forward",
            "sigmoid_backward",
            "gdn_log_decay_forward",
            "gdn_log_decay_backward",
            "conv2d_forward",
            "conv_transpose2d_forward",
            "max_pool2d_forward",
            "interpolate_bilinear_forward",
            "rope_interleaved_forward",
            "rope_interleaved_backward",
            "silu_forward",
            "silu_backward",
            "mul_forward",
            "mul_backward",
            "residual_add_forward",
            "residual_add_backward",
            "layer_norm_forward",
            "channel_layer_norm_forward",
            "group_norm_forward",
            "gelu_erf_forward",
            "relu_forward",
            "concat",
            "slice",
            "fill",
            "linear_bias_forward",
            "broadcast_add_forward",
            "broadcast_mul_forward",
            "scale_forward",
            "softmax_forward",
            "cross_entropy_mean_forward",
            "cross_entropy_mean_backward",
            "clip_grad_norm",
            "adamw_step",
            "muon_ns5_step",
            "optimizer_scratch_bytes",
            "sync",
            "accumulate_grad",
            "scale_grad",
            "linear_cross_entropy_mean",
            "cached_attention_forward",
            "kv_cache_write",
            "argmax_rows",
            "topk_rows",
            "cast_bf16",
            "to_bf16",
            "to_f32",
            "autocast_region",
        ];
        // The harness calls every method in the generated list: a method
        // added to the trait and the list but not called here fails.
        let accessors = ["id", "budget", "numerics", "bf16_operands"];
        let mut called: Vec<&str> = expected.iter().chain(&accessors).copied().collect();
        let mut listed: Vec<&str> = crate::BACKEND_METHODS.to_vec();
        called.sort_unstable();
        listed.sort_unstable();
        assert_eq!(called, listed, "the harness and BACKEND_METHODS disagree");
        for name in owned {
            assert!(
                expected.contains(name),
                "{name} is not a method the harness calls"
            );
        }
        for (got, want) in reached.iter().zip(expected) {
            if owned.contains(&want) {
                assert_eq!(got, "-", "{want} is owned by the wrapper but reached inner");
            } else {
                assert_eq!(got, want, "{want} did not reach the inner backend");
            }
        }
        assert_eq!(reached.len(), expected.len());
        assert_eq!(backend.id(), BackendId::Wgpu);
        assert_eq!(backend.numerics(), Numerics::Fast);
        assert!(backend.bf16_operands(), "bf16_operands did not reach inner");
        assert!(std::ptr::eq(backend.budget(), budget));
        // The four accessors plus every op above.
        accessors.len() + reached.len()
    }

    /// The generated wrappers forward exactly the methods in
    /// `with_backend_methods!`; this reads the trait's own source so a method
    /// added to the trait but not to the list (which every wrapper then
    /// silently runs as the trait default) fails here, as does a stale name.
    #[test]
    fn the_method_list_names_every_trait_method() {
        let source = include_str!("backend.rs");
        let start = source.find("pub trait Backend {").unwrap();
        let end = start + source[start..].find("\n}\n").unwrap();
        let mut in_trait: Vec<&str> = source[start..end]
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("    fn ")?;
                Some(&rest[..rest.find(['(', '<'])?])
            })
            .collect();
        let mut listed: Vec<&str> = crate::BACKEND_METHODS.to_vec();
        in_trait.sort_unstable();
        listed.sort_unstable();
        assert!(
            in_trait.len() > 50,
            "the trait scan found {}",
            in_trait.len()
        );
        let unlisted: Vec<_> = in_trait.iter().filter(|n| !listed.contains(n)).collect();
        let stale: Vec<_> = listed.iter().filter(|n| !in_trait.contains(n)).collect();
        assert!(
            unlisted.is_empty() && stale.is_empty(),
            "not in with_backend_methods!: {unlisted:?}; not in the trait: {stale:?}"
        );
        let mut deduped = listed.clone();
        deduped.dedup();
        assert_eq!(deduped, listed, "a method is listed twice");
    }

    #[test]
    fn references_and_arcs_forward_every_method() {
        let inner = Arc::new(Marker {
            budget: Budget::new(1 << 10),
        });
        let checked = every_call_reaches(&&*inner, &inner);
        assert_eq!(every_call_reaches(&&&*inner, &inner), checked);
        assert_eq!(every_call_reaches(&inner, &inner), checked);
        let shared: Arc<dyn Backend> = inner.clone();
        assert_eq!(every_call_reaches(&shared, &inner), checked);
        assert_eq!(every_call_reaches(&&shared, &inner), checked);
        // A forwarding impl that misses a method fails above; this pins the
        // count so a new trait method is added to `every_call_reaches` too.
        assert_eq!(checked, crate::BACKEND_METHODS.len());
    }

    #[test]
    fn default_framework_ops_refuse_and_sync_succeeds() {
        let budget = Budget::new(1 << 12);
        let backend = NoUpload {
            id: BackendId::Metal,
            budget: budget.clone(),
        };
        backend.sync().unwrap();
        let x = Tensor::from_f32(&[1.0, 2.0], &[1, 2], &budget).unwrap();
        let ids = Tensor::from_u32(&[0], &[1], &budget).unwrap();
        let chunk = CeChunk { rows: 1, cols: 1 };
        let refused = |result: Result<(), OjasError>, want: &str| match result {
            Err(OjasError::Unsupported { op, detail }) => {
                assert_eq!(op, want);
                assert!(detail.contains("Metal"), "{detail}");
            }
            other => panic!("{want}: expected Unsupported, got {other:?}"),
        };
        refused(
            backend
                .linear_cross_entropy_mean(&x, &x, &ids, None, chunk, CeGrads::BOTH)
                .map(drop),
            "linear_cross_entropy_mean",
        );
        let q = Tensor::from_f32(&[1.0; 4], &[1, 1, 1, 4], &budget).unwrap();
        refused(
            backend
                .cached_attention_forward(&q, &q, &q, 1, None)
                .map(drop),
            "cached_attention_forward",
        );
        let mut cache = Tensor::from_f32(&[7.0; 8], &[1, 2, 1, 4], &budget).unwrap();
        refused(backend.kv_cache_write(&mut cache, &q, 0), "kv_cache_write");
        assert_eq!(
            cache.to_f32_vec().unwrap(),
            [7.0; 8],
            "refusal wrote the cache"
        );
        let gates = Tensor::from_f32(&[0.0], &[1, 1, 1], &budget).unwrap();
        let gdn = GdnInputs {
            q: &q,
            k: &q,
            v: &q,
            g: &gates,
            beta: &gates,
            initial_state: None,
        };
        refused(
            backend.chunked_gdn_forward(gdn).map(drop),
            "chunked_gdn_forward",
        );
        refused(
            backend.chunked_gdn_backward(gdn, &q, &q, None).map(drop),
            "chunked_gdn_backward",
        );
        refused(
            backend.causal_conv1d_silu_forward(&x, &x).map(drop),
            "causal_conv1d_silu_forward",
        );
        refused(
            backend.causal_conv1d_silu_backward(&x, &x, &x).map(drop),
            "causal_conv1d_silu_backward",
        );
        refused(
            backend.gated_rms_norm_forward(&x, &x, &x, 1e-6).map(drop),
            "gated_rms_norm_forward",
        );
        refused(
            backend
                .gated_rms_norm_backward(&x, &x, &x, &x, 1e-6)
                .map(drop),
            "gated_rms_norm_backward",
        );
        refused(
            backend.rope_partial_forward(&x, &x, &x).map(drop),
            "rope_partial_forward",
        );
        refused(
            backend.rope_partial_backward(&x, &x, &x).map(drop),
            "rope_partial_backward",
        );
        refused(backend.sigmoid_forward(&x).map(drop), "sigmoid_forward");
        refused(
            backend.sigmoid_backward(&x, &x).map(drop),
            "sigmoid_backward",
        );
        refused(
            backend.gdn_log_decay_forward(&x, &x, &x).map(drop),
            "gdn_log_decay_forward",
        );
        refused(
            backend.gdn_log_decay_backward(&x, &x, &x, &x).map(drop),
            "gdn_log_decay_backward",
        );
        refused(
            backend
                .conv2d_forward(&x, &x, None, Conv2dParams::new())
                .map(drop),
            "conv2d_forward",
        );
        refused(
            backend
                .conv_transpose2d_forward(&x, &x, None, ConvTranspose2dParams::new())
                .map(drop),
            "conv_transpose2d_forward",
        );
        refused(
            backend.max_pool2d_forward(&x, [2, 2], [2, 2]).map(drop),
            "max_pool2d_forward",
        );
        refused(
            backend
                .interpolate_bilinear_forward(&x, [2, 2], false)
                .map(drop),
            "interpolate_bilinear_forward",
        );
        refused(
            backend.rope_interleaved_forward(&x, &x, &x).map(drop),
            "rope_interleaved_forward",
        );
        refused(
            backend.rope_interleaved_backward(&x, &x, &x).map(drop),
            "rope_interleaved_backward",
        );
    }

    #[test]
    fn metal_gdn_dims_are_refused_not_padded() {
        refuse_unsupported_metal_gdn(BackendId::Metal, METAL_GDN_KEY_DIM, 16).unwrap();
        refuse_unsupported_metal_gdn(BackendId::Metal, METAL_GDN_KEY_DIM, 256).unwrap();
        for (dk, dv) in [
            (64, 16),
            (256, 16),
            (METAL_GDN_KEY_DIM, 8),
            (METAL_GDN_KEY_DIM, 24),
        ] {
            assert!(
                matches!(
                    refuse_unsupported_metal_gdn(BackendId::Metal, dk, dv),
                    Err(OjasError::Unsupported {
                        op: "chunked_gdn",
                        ..
                    })
                ),
                "dk {dk} dv {dv}"
            );
            refuse_unsupported_metal_gdn(BackendId::Cpu, dk, dv).unwrap();
            refuse_unsupported_metal_gdn(BackendId::Wgpu, dk, dv).unwrap();
        }
    }

    #[test]
    fn default_accumulate_grad_adds_and_leaves_acc_alone_on_error() {
        let budget = Budget::new(1 << 12);
        let backend = NoUpload {
            id: BackendId::Cpu,
            budget: budget.clone(),
        };
        let mut acc = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], &budget).unwrap();
        let grad = Tensor::from_f32(&[0.5, -2.0, 4.0], &[3], &budget).unwrap();
        backend.accumulate_grad(&mut acc, &grad).unwrap();
        assert_eq!(acc.to_f32_vec().unwrap(), [1.5, 0.0, 7.0]);
        backend.accumulate_grad(&mut acc, &grad).unwrap();
        assert_eq!(acc.to_f32_vec().unwrap(), [2.0, -2.0, 11.0]);

        let wrong = Tensor::from_f32(&[1.0, 1.0], &[2], &budget).unwrap();
        assert!(backend.accumulate_grad(&mut acc, &wrong).is_err());
        assert_eq!(acc.to_f32_vec().unwrap(), [2.0, -2.0, 11.0]);
    }

    #[derive(Debug)]
    struct CountingDev {
        reads: std::sync::atomic::AtomicU64,
    }

    impl DeviceBuffer for CountingDev {
        fn backend(&self) -> BackendId {
            BackendId::Metal
        }
        fn byte_len(&self) -> usize {
            4
        }
        fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
            self.reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(vec![0; len])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn default_cast_rounds_host_f32_and_refuses_the_rest() {
        use std::sync::atomic::Ordering;

        let budget = Budget::new(1 << 12);
        let backend = NoUpload {
            id: BackendId::Cpu,
            budget: budget.clone(),
        };
        let x = Tensor::from_f32(&[f32::from_bits(0x3f81_8000)], &[1], &budget).unwrap();
        let y = backend.cast_bf16(&x).unwrap();
        assert_eq!(y.to_f32_vec().unwrap()[0].to_bits(), 0x3f82_0000);
        assert_eq!(y.compute_tag(), 0);

        let empty_budget = Budget::new(0);
        let empty = Tensor::zeros(&[0], DType::F32, &empty_budget).unwrap();
        let empty_backend = NoUpload {
            id: BackendId::Cpu,
            budget: empty_budget,
        };
        let rounded_empty = empty_backend.cast_bf16(&empty).unwrap();
        assert_eq!(rounded_empty.shape(), &[0]);

        let ids = Tensor::from_u32(&[1], &[1], &budget).unwrap();
        assert!(matches!(
            backend.cast_bf16(&ids),
            Err(OjasError::Dtype {
                expected: DType::F32,
                got: DType::U32,
                ..
            })
        ));
        let half = Tensor::zeros(&[1], DType::F16, &budget).unwrap();
        assert!(matches!(
            backend.cast_bf16(&half),
            Err(OjasError::Dtype {
                expected: DType::F32,
                got: DType::F16,
                ..
            })
        ));

        let wide = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &budget).unwrap();
        let transposed = wide.view(&[2, 2], &[1, 2], 0).unwrap();
        assert!(!transposed.is_contiguous().unwrap());
        assert!(matches!(
            backend.cast_bf16(&transposed),
            Err(OjasError::Shape { .. })
        ));

        let device = Arc::new(CountingDev {
            reads: std::sync::atomic::AtomicU64::new(0),
        });
        let device_tensor = Tensor::from_device(device.clone(), &[1], DType::F32, &budget).unwrap();
        assert!(matches!(
            backend.cast_bf16(&device_tensor),
            Err(OjasError::Unsupported {
                op: "cast_bf16",
                ..
            })
        ));
        assert_eq!(device.reads.load(Ordering::Relaxed), 0);

        let tight = Budget::new(8);
        let tight_backend = NoUpload {
            id: BackendId::Cpu,
            budget: tight.clone(),
        };
        let held = Tensor::from_f32(&[1.0, 2.0], &[2], &tight).unwrap();
        assert_eq!(tight.live_bytes().unwrap(), 8);
        assert!(matches!(
            tight_backend.cast_bf16(&held),
            Err(OjasError::CapacityExceeded {
                requested: 8,
                cap: 8,
                live: 8
            })
        ));
        assert_eq!(tight.live_bytes().unwrap(), 8);

        let guard = backend.autocast_region(AutocastMode::Off).unwrap();
        drop(guard);
        assert!(matches!(
            backend.autocast_region(AutocastMode::Bf16),
            Err(OjasError::Unsupported {
                op: "autocast_region",
                ..
            })
        ));
    }
}
