//! Nanolab-default forward and backward operations.
//!
//! The trait is the frozen contract. This crate does not implement it.
//! Later crates own the CPU reference and the Metal kernels. There is no
//! silent fallback from [`BackendId::Metal`] to [`BackendId::Cpu`].

use crate::budget::Budget;
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
/// thread count. `Fast` may fuse multiply-adds, use SIMD intrinsics or a
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

/// `MetalBackend` causal attention refuses a head dimension above this.
///
/// The tiled forward and backward kernels are instantiated for widths 16, 32,
/// 64 and 128, and a head dimension pads up to the next width. Nothing is
/// clamped: a larger head dimension is [`OjasError::UnsupportedHeadDim`].
/// The separate tiny training step in `ojas-metal/src/gpu.rs` accepts only
/// 64 and refuses anything else as a shape error.
pub const METAL_MAX_HEAD_DIM: u32 = 128;

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

/// Largest rank [`Backend::permute`] accepts.
pub const MAX_PERMUTE_RANK: usize = 8;

/// Output shape of permuting `shape` by `dims`, after checking that `dims`
/// names every axis of `shape` exactly once.
///
/// Output axis `i` is input axis `dims[i]`, as in `torch.permute`. A rank
/// above [`MAX_PERMUTE_RANK`], a length mismatch, an axis out of range, or a
/// repeated axis is [`OjasError::Shape`]; nothing is clamped or reordered.
pub fn permute_output_shape(
    op: &'static str,
    shape: &[usize],
    dims: &[usize],
) -> Result<Vec<usize>, OjasError> {
    let refuse = |detail: String| OjasError::Shape { op, detail };
    if shape.len() > MAX_PERMUTE_RANK {
        return Err(refuse(format!(
            "rank {} exceeds {MAX_PERMUTE_RANK}",
            shape.len()
        )));
    }
    if dims.len() != shape.len() {
        return Err(refuse(format!(
            "{} permutation axes for a rank-{} tensor",
            dims.len(),
            shape.len()
        )));
    }
    let mut seen = [false; MAX_PERMUTE_RANK];
    for &axis in dims {
        if axis >= shape.len() {
            return Err(refuse(format!(
                "axis {axis} out of range for rank {}",
                shape.len()
            )));
        }
        if seen[axis] {
            return Err(refuse(format!("axis {axis} repeated in {dims:?}")));
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
    /// is the upstream gradient permuted by [`inverse_permutation`], so there
    /// is no separate backward method. Implementations validate with
    /// [`permute_output_shape`], keep the result on this backend, and move
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

    /// One matrix. Every backend runs Newton-Schulz in f32 (nanolab runs it in
    /// bf16; see `docs/op-coverage.md`). `step` uses [`next_step`]. Call
    /// [`require_ns5`] if a caller can pass a step count.
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
    /// once `sync` reports the fault. The default composes
    /// [`Backend::residual_add_forward`]; a backend may add in place.
    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        *acc = self.residual_add_forward(acc, grad)?;
        Ok(())
    }

    /// Mean cross-entropy of `input @ weight^T` without materializing the
    /// `[N, V]` logits, plus both gradients when `want_grad` is set.
    ///
    /// `input` is `[N, d]` `F32`, `weight` is `[V, d]` `F32` (the tied
    /// embedding), `targets` is `[N]` `U32`. Values equal
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
        want_grad: bool,
    ) -> Result<LinearCe, OjasError> {
        let _ = (input, weight, targets, ignore_index, chunk, want_grad);
        Err(OjasError::Unsupported {
            op: "linear_cross_entropy_mean",
            detail: format!(
                "{:?} backend does not implement linear_cross_entropy_mean",
                self.id()
            ),
        })
    }

    /// Causal attention of `Tq` new queries against the first `kv_len`
    /// positions of a KV cache, for decode and for prefill onto a non-empty
    /// cache.
    ///
    /// `q` is `[B, Tq, H, D]`; `k_cache` and `v_cache` are `[B, Tcap, Hkv, D]`,
    /// time-major like `ojas-infer`'s host cache. Query `i` sits at position
    /// `kv_len - Tq + i` and attends to keys `0..=` that position. Head `h`
    /// reads KV head `h / (H / Hkv)` (grouped-query attention). Scale is
    /// [`sdpa_scale`]`(D)`. Returns `[B, Tq, H, D]`. With `kv_len == Tq` and
    /// `H == Hkv` this equals [`Backend::causal_sdpa_forward`] after the
    /// layout permute. Validate with [`cached_attention_dims`].
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        let _ = (q, k_cache, v_cache, kv_len);
        Err(OjasError::Unsupported {
            op: "cached_attention_forward",
            detail: format!(
                "{:?} backend does not implement cached_attention_forward",
                self.id()
            ),
        })
    }

    /// Write `src` (`[B, Tn, Hkv, D]`) into `cache` (`[B, Tcap, Hkv, D]`) at
    /// time positions `at..at + Tn`.
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
}

/// Forward every [`Backend`] method, including the defaulted ones, to the
/// value behind `self`. A defaulted method that were not forwarded would run
/// the trait default on the wrapper, for example a `sync` that skips the
/// inner backend's deferred faults.
macro_rules! forward_backend {
    () => {
        fn id(&self) -> BackendId {
            (**self).id()
        }
        fn budget(&self) -> &Budget {
            (**self).budget()
        }
        fn numerics(&self) -> Numerics {
            (**self).numerics()
        }
        fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
            (**self).upload(tensor)
        }
        fn download(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
            (**self).download(tensor)
        }
        fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
            (**self).permute(input, dims)
        }
        fn embedding_forward(
            &self,
            table: &Tensor,
            token_ids: &Tensor,
        ) -> Result<Tensor, OjasError> {
            (**self).embedding_forward(table, token_ids)
        }
        fn embedding_backward(
            &self,
            table: &Tensor,
            token_ids: &Tensor,
            grad_output: &Tensor,
        ) -> Result<Tensor, OjasError> {
            (**self).embedding_backward(table, token_ids, grad_output)
        }
        fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
            (**self).linear_forward(input, weight)
        }
        fn linear_backward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            (**self).linear_backward(input, weight, grad_output)
        }
        fn rms_norm_forward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            eps: f32,
        ) -> Result<Tensor, OjasError> {
            (**self).rms_norm_forward(input, weight, eps)
        }
        fn rms_norm_backward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            grad_output: &Tensor,
            eps: f32,
        ) -> Result<(Tensor, Tensor), OjasError> {
            (**self).rms_norm_backward(input, weight, grad_output, eps)
        }
        fn rope_half_split_forward(
            &self,
            x: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
        ) -> Result<Tensor, OjasError> {
            (**self).rope_half_split_forward(x, cos, sin)
        }
        fn rope_half_split_backward(
            &self,
            grad_output: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
        ) -> Result<Tensor, OjasError> {
            (**self).rope_half_split_backward(grad_output, cos, sin)
        }
        fn rms_qk_norm_forward(
            &self,
            q: &Tensor,
            k: &Tensor,
            q_weight: &Tensor,
            k_weight: &Tensor,
            eps: f32,
        ) -> Result<(Tensor, Tensor), OjasError> {
            (**self).rms_qk_norm_forward(q, k, q_weight, k_weight, eps)
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
            (**self).rms_qk_norm_backward(q, k, q_weight, k_weight, grad_q, grad_k, eps)
        }
        fn causal_sdpa_forward(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
        ) -> Result<Tensor, OjasError> {
            (**self).causal_sdpa_forward(q, k, v)
        }
        fn causal_sdpa_backward(
            &self,
            q: &Tensor,
            k: &Tensor,
            v: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
            (**self).causal_sdpa_backward(q, k, v, grad_output)
        }
        fn per_head_sigmoid_gate_forward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
        ) -> Result<Tensor, OjasError> {
            (**self).per_head_sigmoid_gate_forward(input, weight, bias, attn_out)
        }
        fn per_head_sigmoid_gate_backward(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
            grad_output: &Tensor,
        ) -> Result<PerHeadGateGrad, OjasError> {
            (**self).per_head_sigmoid_gate_backward(input, weight, bias, attn_out, grad_output)
        }
        fn per_head_sigmoid_gate_forward_saving(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
        ) -> Result<(Tensor, Option<Tensor>), OjasError> {
            (**self).per_head_sigmoid_gate_forward_saving(input, weight, bias, attn_out)
        }
        fn per_head_sigmoid_gate_backward_saved(
            &self,
            input: &Tensor,
            weight: &Tensor,
            bias: &Tensor,
            attn_out: &Tensor,
            grad_output: &Tensor,
            scales: &Tensor,
        ) -> Result<PerHeadGateGrad, OjasError> {
            (**self).per_head_sigmoid_gate_backward_saved(
                input,
                weight,
                bias,
                attn_out,
                grad_output,
                scales,
            )
        }
        fn value_residual_blend_forward(
            &self,
            value: &Tensor,
            value0: &Tensor,
            lambda: &Tensor,
        ) -> Result<Tensor, OjasError> {
            (**self).value_residual_blend_forward(value, value0, lambda)
        }
        fn value_residual_blend_backward(
            &self,
            value: &Tensor,
            value0: &Tensor,
            lambda: &Tensor,
            grad_output: &Tensor,
        ) -> Result<ValueResidualGrad, OjasError> {
            (**self).value_residual_blend_backward(value, value0, lambda, grad_output)
        }
        fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
            (**self).silu_forward(input)
        }
        fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
            (**self).silu_backward(input, grad_output)
        }
        fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
            (**self).mul_forward(a, b)
        }
        fn mul_backward(
            &self,
            a: &Tensor,
            b: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            (**self).mul_backward(a, b, grad_output)
        }
        fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
            (**self).residual_add_forward(x, y)
        }
        fn residual_add_backward(
            &self,
            x: &Tensor,
            y: &Tensor,
            grad_output: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            (**self).residual_add_backward(x, y, grad_output)
        }
        fn cross_entropy_mean_forward(
            &self,
            logits: &Tensor,
            targets: &Tensor,
            ignore_index: Option<u32>,
        ) -> Result<Tensor, OjasError> {
            (**self).cross_entropy_mean_forward(logits, targets, ignore_index)
        }
        fn cross_entropy_mean_backward(
            &self,
            logits: &Tensor,
            targets: &Tensor,
            ignore_index: Option<u32>,
        ) -> Result<Tensor, OjasError> {
            (**self).cross_entropy_mean_backward(logits, targets, ignore_index)
        }
        fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
            (**self).clip_grad_norm(grads, max_norm)
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
            (**self).adamw_step(param, grad, moment1, moment2, step, config)
        }
        fn muon_ns5_step(
            &self,
            param: &mut Tensor,
            grad: &Tensor,
            momentum: &mut Tensor,
            config: MuonNs5Config,
        ) -> Result<(), OjasError> {
            (**self).muon_ns5_step(param, grad, momentum, config)
        }
        fn optimizer_scratch_bytes(
            &self,
            kind: OptimizerKind,
            rows: usize,
            cols: usize,
        ) -> Result<Option<u64>, OjasError> {
            (**self).optimizer_scratch_bytes(kind, rows, cols)
        }
        fn sync(&self) -> Result<(), OjasError> {
            (**self).sync()
        }
        fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
            (**self).accumulate_grad(acc, grad)
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
            (**self).linear_cross_entropy_mean(
                input,
                weight,
                targets,
                ignore_index,
                chunk,
                want_grad,
            )
        }
        fn cached_attention_forward(
            &self,
            q: &Tensor,
            k_cache: &Tensor,
            v_cache: &Tensor,
            kv_len: usize,
        ) -> Result<Tensor, OjasError> {
            (**self).cached_attention_forward(q, k_cache, v_cache, kv_len)
        }
        fn kv_cache_write(
            &self,
            cache: &mut Tensor,
            src: &Tensor,
            at: usize,
        ) -> Result<(), OjasError> {
            (**self).kv_cache_write(cache, src, at)
        }
    };
}

/// A shared reference to a backend is a backend, so one backend can serve a
/// `Tape` and a trainer at once.
impl<B: Backend + ?Sized> Backend for &B {
    forward_backend!();
}

/// An `Arc` of a backend is a backend, for backends that are not `Clone`
/// (for example `WgpuBackend`) shared across sessions.
impl<B: Backend + ?Sized> Backend for std::sync::Arc<B> {
    forward_backend!();
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

/// Loss and, when requested, gradients from [`Backend::linear_cross_entropy_mean`].
///
/// `loss` has the shape [`Backend::cross_entropy_mean_forward`] returns.
/// `grad_input` is `[N, d]` and `grad_weight` is `[V, d]`; both are `None`
/// exactly when `want_grad` was false.
#[derive(Debug)]
pub struct LinearCe {
    pub loss: Tensor,
    pub grad_input: Option<Tensor>,
    pub grad_weight: Option<Tensor>,
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
            assert!(refuse_unsupported_metal_head_dim(backend, 128).is_ok());
        }
        assert!(refuse_unsupported_metal_head_dim(BackendId::Cpu, u32::MAX).is_ok());
        assert!(matches!(
            refuse_unsupported_metal_head_dim(BackendId::Metal, 129),
            Err(OjasError::UnsupportedHeadDim {
                head_dim: 129,
                limit: 128
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
        ) -> Result<Tensor, OjasError> {
            Err(none("causal_sdpa_forward"))
        }
        fn causal_sdpa_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
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
    /// is caught.
    struct Marker {
        budget: Budget,
    }

    fn mark(op: &str) -> OjasError {
        OjasError::Backend {
            id: BackendId::Wgpu,
            detail: format!("marker:{op}"),
        }
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
        fn upload(&self, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("upload"))
        }
        fn download(&self, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("download"))
        }
        fn permute(&self, _: &Tensor, _: &[usize]) -> Result<Tensor, OjasError> {
            Err(mark("permute"))
        }
        fn embedding_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("embedding_forward"))
        }
        fn embedding_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(mark("embedding_backward"))
        }
        fn linear_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("linear_forward"))
        }
        fn linear_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(mark("linear_backward"))
        }
        fn rms_norm_forward(&self, _: &Tensor, _: &Tensor, _: f32) -> Result<Tensor, OjasError> {
            Err(mark("rms_norm_forward"))
        }
        fn rms_norm_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: f32,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(mark("rms_norm_backward"))
        }
        fn rope_half_split_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(mark("rope_half_split_forward"))
        }
        fn rope_half_split_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(mark("rope_half_split_backward"))
        }
        fn rms_qk_norm_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: f32,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(mark("rms_qk_norm_forward"))
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
            Err(mark("rms_qk_norm_backward"))
        }
        fn causal_sdpa_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(mark("causal_sdpa_forward"))
        }
        fn causal_sdpa_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
            Err(mark("causal_sdpa_backward"))
        }
        fn per_head_sigmoid_gate_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(mark("per_head_sigmoid_gate_forward"))
        }
        fn per_head_sigmoid_gate_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<PerHeadGateGrad, OjasError> {
            Err(mark("per_head_sigmoid_gate_backward"))
        }
        fn per_head_sigmoid_gate_forward_saving(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Option<Tensor>), OjasError> {
            Err(mark("per_head_sigmoid_gate_forward_saving"))
        }
        fn per_head_sigmoid_gate_backward_saved(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<PerHeadGateGrad, OjasError> {
            Err(mark("per_head_sigmoid_gate_backward_saved"))
        }
        fn value_residual_blend_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<Tensor, OjasError> {
            Err(mark("value_residual_blend_forward"))
        }
        fn value_residual_blend_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<ValueResidualGrad, OjasError> {
            Err(mark("value_residual_blend_backward"))
        }
        fn silu_forward(&self, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("silu_forward"))
        }
        fn silu_backward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("silu_backward"))
        }
        fn mul_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("mul_forward"))
        }
        fn mul_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(mark("mul_backward"))
        }
        fn residual_add_forward(&self, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
            Err(mark("residual_add_forward"))
        }
        fn residual_add_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
        ) -> Result<(Tensor, Tensor), OjasError> {
            Err(mark("residual_add_backward"))
        }
        fn cross_entropy_mean_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: Option<u32>,
        ) -> Result<Tensor, OjasError> {
            Err(mark("cross_entropy_mean_forward"))
        }
        fn cross_entropy_mean_backward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: Option<u32>,
        ) -> Result<Tensor, OjasError> {
            Err(mark("cross_entropy_mean_backward"))
        }
        fn clip_grad_norm(&self, _: &mut [Tensor], _: f32) -> Result<f32, OjasError> {
            Err(mark("clip_grad_norm"))
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
            Err(mark("adamw_step"))
        }
        fn muon_ns5_step(
            &self,
            _: &mut Tensor,
            _: &Tensor,
            _: &mut Tensor,
            _: MuonNs5Config,
        ) -> Result<(), OjasError> {
            Err(mark("muon_ns5_step"))
        }
        fn optimizer_scratch_bytes(
            &self,
            _: OptimizerKind,
            _: usize,
            _: usize,
        ) -> Result<Option<u64>, OjasError> {
            Err(mark("optimizer_scratch_bytes"))
        }
        fn sync(&self) -> Result<(), OjasError> {
            Err(mark("sync"))
        }
        fn accumulate_grad(&self, _: &mut Tensor, _: &Tensor) -> Result<(), OjasError> {
            Err(mark("accumulate_grad"))
        }
        fn linear_cross_entropy_mean(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: Option<u32>,
            _: CeChunk,
            _: bool,
        ) -> Result<LinearCe, OjasError> {
            Err(mark("linear_cross_entropy_mean"))
        }
        fn cached_attention_forward(
            &self,
            _: &Tensor,
            _: &Tensor,
            _: &Tensor,
            _: usize,
        ) -> Result<Tensor, OjasError> {
            Err(mark("cached_attention_forward"))
        }
        fn kv_cache_write(&self, _: &mut Tensor, _: &Tensor, _: usize) -> Result<(), OjasError> {
            Err(mark("kv_cache_write"))
        }
    }

    /// Call every method through `backend` (a wrapper, dispatched
    /// statically through its own impl) and require each to reach `inner`.
    /// Returns the number of methods checked.
    fn every_call_reaches<B: Backend + ?Sized>(backend: &B, inner: &Marker) -> usize {
        let budget = &inner.budget;
        let t = Tensor::from_f32(&[1.0], &[1], budget).unwrap();
        let mut m = t.clone();
        let mut m2 = t.clone();
        let mut m3 = t.clone();
        let adamw = AdamWConfig::nanolab(1e-3, 0.0);
        let muon = MuonNs5Config::nanolab_default();
        let chunk = CeChunk { rows: 1, cols: 1 };
        fn op<T: std::fmt::Debug>(result: Result<T, OjasError>) -> String {
            match result {
                Err(OjasError::Backend {
                    id: BackendId::Wgpu,
                    detail,
                }) => detail
                    .strip_prefix("marker:")
                    .unwrap_or(&detail)
                    .to_string(),
                other => panic!("call did not reach the inner backend: {other:?}"),
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
            op(backend.rms_norm_forward(&t, &t, 1e-6)),
            op(backend.rms_norm_backward(&t, &t, &t, 1e-6)),
            op(backend.rope_half_split_forward(&t, &t, &t)),
            op(backend.rope_half_split_backward(&t, &t, &t)),
            op(backend.rms_qk_norm_forward(&t, &t, &t, &t, 1e-6)),
            op(backend.rms_qk_norm_backward(&t, &t, &t, &t, &t, &t, 1e-6)),
            op(backend.causal_sdpa_forward(&t, &t, &t)),
            op(backend.causal_sdpa_backward(&t, &t, &t, &t)),
            op(backend.per_head_sigmoid_gate_forward(&t, &t, &t, &t)),
            op(backend.per_head_sigmoid_gate_backward(&t, &t, &t, &t, &t)),
            op(backend
                .per_head_sigmoid_gate_forward_saving(&t, &t, &t, &t)
                .map(drop)),
            op(backend.per_head_sigmoid_gate_backward_saved(&t, &t, &t, &t, &t, &t)),
            op(backend.value_residual_blend_forward(&t, &t, &t)),
            op(backend.value_residual_blend_backward(&t, &t, &t, &t)),
            op(backend.silu_forward(&t)),
            op(backend.silu_backward(&t, &t)),
            op(backend.mul_forward(&t, &t)),
            op(backend.mul_backward(&t, &t, &t)),
            op(backend.residual_add_forward(&t, &t)),
            op(backend.residual_add_backward(&t, &t, &t)),
            op(backend.cross_entropy_mean_forward(&t, &t, None)),
            op(backend.cross_entropy_mean_backward(&t, &t, None)),
            op(backend.clip_grad_norm(std::slice::from_mut(&mut m), 1.0)),
            op(backend.adamw_step(&mut m, &t, &mut m2, &mut m3, 0, adamw)),
            op(backend.muon_ns5_step(&mut m, &t, &mut m2, muon)),
            op(backend.optimizer_scratch_bytes(OptimizerKind::MuonNs5, 1, 1)),
            op(backend.sync()),
            op(backend.accumulate_grad(&mut m, &t)),
            op(backend.linear_cross_entropy_mean(&t, &t, &t, None, chunk, true)),
            op(backend.cached_attention_forward(&t, &t, &t, 1)),
            op(backend.kv_cache_write(&mut m, &t, 0)),
        ];
        let expected = [
            "upload",
            "download",
            "permute",
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
            "per_head_sigmoid_gate_forward_saving",
            "per_head_sigmoid_gate_backward_saved",
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
            "muon_ns5_step",
            "optimizer_scratch_bytes",
            "sync",
            "accumulate_grad",
            "linear_cross_entropy_mean",
            "cached_attention_forward",
            "kv_cache_write",
        ];
        assert_eq!(reached, expected);
        assert_eq!(backend.id(), BackendId::Wgpu);
        assert_eq!(backend.numerics(), Numerics::Fast);
        assert!(std::ptr::eq(backend.budget(), budget));
        // The three accessors plus every op above.
        3 + reached.len()
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
        assert_eq!(checked, 41);
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
                .linear_cross_entropy_mean(&x, &x, &ids, None, chunk, true)
                .map(drop),
            "linear_cross_entropy_mean",
        );
        let q = Tensor::from_f32(&[1.0; 4], &[1, 1, 1, 4], &budget).unwrap();
        refused(
            backend.cached_attention_forward(&q, &q, &q, 1).map(drop),
            "cached_attention_forward",
        );
        let mut cache = Tensor::from_f32(&[7.0; 8], &[1, 2, 1, 4], &budget).unwrap();
        refused(backend.kv_cache_write(&mut cache, &q, 0), "kv_cache_write");
        assert_eq!(
            cache.to_f32_vec().unwrap(),
            [7.0; 8],
            "refusal wrote the cache"
        );
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
}
