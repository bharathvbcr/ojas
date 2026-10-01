//! Nanolab-default forward and backward operations.
//!
//! The trait is the frozen contract. This crate does not implement it.
//! Later crates own the CPU reference and the Metal kernels. There is no
//! silent fallback from [`BackendId::Metal`] to [`BackendId::Cpu`].

use crate::tensor::Tensor;
use crate::OjasError;

/// Explicit device selection. Callers choose one. Nothing here picks for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendId {
    Cpu,
    Metal,
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

/// Metal flash-attention path refuses a head dimension above this.
/// The kernel clamps with `min(D, 64)`; ojas returns an error instead.
pub const METAL_MAX_HEAD_DIM: u32 = 64;

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

/// Metal refuses `head_dim > 64`. CPU allows any non-zero head dimension.
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

/// Muon NS5 as nanolab's `Muon.step` runs it.
///
/// Nesterov momentum (`buf = mom * buf + g`, update = `g + mom * buf` when
/// Nesterov is on). Orthogonalize with five Newton-Schulz steps in bf16 on
/// the fused path (`X = G.bfloat16()`), then scale by `max(1, rows/cols)^0.5`.
/// Decoupled decay is `p *= 1 - lr * weight_decay` before the update is added.
/// metal-native's f32 Newton-Schulz is not the v1 dtype.
#[derive(Clone, Copy, Debug)]
pub struct MuonNs5Config {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
}

impl MuonNs5Config {
    /// `lr` 0.025, momentum 0.99, weight decay 0.1, Nesterov on.
    /// Those are the nanolab defaults (`matrix_lr`, `muon_momentum`, `weight_decay`).
    pub fn nanolab_default() -> Self {
        Self {
            lr: 0.025,
            momentum: 0.99,
            weight_decay: 0.1,
            nesterov: true,
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

/// Gradients of the value-residual blend. `lambda` is the scalar parameter.
#[derive(Clone, Debug)]
pub struct ValueResidualGrad {
    pub value: Tensor,
    pub value0: Tensor,
    pub lambda: Tensor,
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

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError>;

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

    fn causal_sdpa_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError>;

    /// `(grad_q, grad_k, grad_v)`.
    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError>;

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

    /// One matrix. The fused path runs Newton-Schulz in bf16. `step` uses
    /// [`next_step`]. Call [`require_ns5`] if a caller can pass a step count.
    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError>;
}

#[cfg(test)]
mod tests {
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
        }
        assert!(refuse_unsupported_metal_head_dim(BackendId::Cpu, u32::MAX).is_ok());
        assert!(matches!(
            refuse_unsupported_metal_head_dim(BackendId::Metal, 65),
            Err(OjasError::UnsupportedHeadDim {
                head_dim: 65,
                limit: 64
            })
        ));
    }

    #[test]
    fn only_five_newton_schulz_steps() {
        assert!(require_ns5(5).is_ok());
        for steps in [0u32, 4, 6, u32::MAX] {
            assert!(require_ns5(steps).is_err());
        }
    }
}
