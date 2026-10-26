//! Adversarial helper (card ft-fe7dcee6): a backend that answers every `op`
//! with `Err("reached:<name>")` and records the name, plus a call table
//! that invokes any op by name on any backend. Used to ask, behaviourally,
//! whether a wrapper reaches the wrapped backend for every method, instead
//! of reading source text.

#![allow(dead_code)]

use std::sync::Mutex;

use ojas_core::{
    with_backend_methods, AdamWConfig, AttentionOpts, AutocastMode, Backend, BackendId, Budget,
    CeChunk, CeGrads, Conv2dParams, ConvTranspose2dParams, GdnInputs, MuonNs5Config, Numerics,
    OjasError, OptimizerKind, Tensor,
};

/// Every `Backend` op, answered with its own name.
pub struct Reach {
    budget: Budget,
    seen: Mutex<Vec<&'static str>>,
}

impl Reach {
    pub fn new() -> Self {
        Self {
            budget: Budget::new(1 << 20),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn reached(&self, name: &'static str) -> OjasError {
        self.seen.lock().unwrap().push(name);
        OjasError::Backend {
            id: BackendId::Cpu,
            detail: format!("reached:{name}"),
        }
    }

    /// How many times `name` reached this backend.
    pub fn count(&self, name: &str) -> usize {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .filter(|n| **n == name)
            .count()
    }

    pub fn total(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

macro_rules! reach_method {
    (pure, $s:ident, $name:ident, [$($arg:ident : $ty:ty),*], $ret:ty) => {};
    (op, $s:ident, $name:ident, [$($arg:ident : $ty:ty),*], $ret:ty) => {
        #[allow(unused_variables, clippy::too_many_arguments)]
        fn $name(&$s $(, $arg: $ty)*) -> $ret {
            Err($s.reached(stringify!($name)))
        }
    };
}

macro_rules! reach_methods {
    ({ $s:ident } $($kind:ident fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty;)*) => {
        $( reach_method!($kind, $s, $name, [$($arg : $ty),*], $ret); )*
    };
}

impl Backend for Reach {
    with_backend_methods!(reach_methods, { self });
    fn id(&self) -> BackendId {
        BackendId::Cpu
    }
    fn budget(&self) -> &Budget {
        &self.budget
    }
    fn numerics(&self) -> Numerics {
        Numerics::Exact
    }
    fn optimizer_scratch_bytes(
        &self,
        _: OptimizerKind,
        _: usize,
        _: usize,
    ) -> Result<Option<u64>, OjasError> {
        Ok(None)
    }
    fn bf16_operands(&self) -> bool {
        false
    }
}

/// Call op `name` on `b` with minimal operands. `None` when the table has no
/// such op (the caller asserts the table covers `ojas_core::BACKEND_METHODS`).
pub fn call<B: Backend + ?Sized>(b: &B, name: &str) -> Option<Result<(), OjasError>> {
    let budget = Budget::new(1 << 20);
    let t = Tensor::from_f32(&[1.0], &[1], &budget).unwrap();
    let mut m = t.clone();
    let mut m2 = t.clone();
    let mut m3 = t.clone();
    let mut gs = vec![t.clone()];
    let inputs = GdnInputs {
        q: &t,
        k: &t,
        v: &t,
        g: &t,
        beta: &t,
        initial_state: None,
    };
    let _ = (&mut m2, &mut m3, &mut gs, &inputs);
    Some(match name {
        "upload" => b.upload(&t).map(|_| ()),
        "download" => b.download(&t).map(|_| ()),
        "permute" => b.permute(&t, &[0usize]).map(|_| ()),
        "embedding_forward" => b.embedding_forward(&t, &t).map(|_| ()),
        "embedding_backward" => b.embedding_backward(&t, &t, &t).map(|_| ()),
        "linear_forward" => b.linear_forward(&t, &t).map(|_| ()),
        "linear_backward" => b.linear_backward(&t, &t, &t).map(|_| ()),
        "linear_backward_input" => b.linear_backward_input(&t, &t, &t).map(|_| ()),
        "rms_norm_forward" => b.rms_norm_forward(&t, &t, 1.0).map(|_| ()),
        "rms_norm_backward" => b.rms_norm_backward(&t, &t, &t, 1.0).map(|_| ()),
        "rope_half_split_forward" => b.rope_half_split_forward(&t, &t, &t).map(|_| ()),
        "rope_half_split_backward" => b.rope_half_split_backward(&t, &t, &t).map(|_| ()),
        "rms_qk_norm_forward" => b.rms_qk_norm_forward(&t, &t, &t, &t, 1.0).map(|_| ()),
        "rms_qk_norm_backward" => b
            .rms_qk_norm_backward(&t, &t, &t, &t, &t, &t, 1.0)
            .map(|_| ()),
        "causal_sdpa_forward" => b.causal_sdpa_forward(&t, &t, &t, None).map(|_| ()),
        "causal_sdpa_backward" => b
            .causal_sdpa_backward(&t, &t, &t, &t, &t, &t, None)
            .map(|_| ()),
        "causal_sdpa_backward_recompute" => b
            .causal_sdpa_backward_recompute(&t, &t, &t, &t, None)
            .map(|_| ()),
        "attention_forward" => b
            .attention_forward(&t, &t, &t, AttentionOpts::default())
            .map(|_| ()),
        "attention_forward_saving" => b
            .attention_forward_saving(&t, &t, &t, AttentionOpts::default())
            .map(|_| ()),
        "attention_backward" => b
            .attention_backward(&t, &t, &t, &t, &t, &t, AttentionOpts::default(), false)
            .map(|_| ()),
        "per_head_sigmoid_gate_forward" => {
            b.per_head_sigmoid_gate_forward(&t, &t, &t, &t).map(|_| ())
        }
        "per_head_sigmoid_gate_backward" => b
            .per_head_sigmoid_gate_backward(&t, &t, &t, &t, &t)
            .map(|_| ()),
        "per_head_sigmoid_gate_forward_saving" => b
            .per_head_sigmoid_gate_forward_saving(&t, &t, &t, &t)
            .map(|_| ()),
        "per_head_sigmoid_gate_backward_saved" => b
            .per_head_sigmoid_gate_backward_saved(&t, &t, &t, &t, &t, &t)
            .map(|_| ()),
        "value_residual_blend_forward" => b.value_residual_blend_forward(&t, &t, &t).map(|_| ()),
        "value_residual_blend_backward" => {
            b.value_residual_blend_backward(&t, &t, &t, &t).map(|_| ())
        }
        "chunked_gdn_forward" => b.chunked_gdn_forward(inputs).map(|_| ()),
        "chunked_gdn_backward" => b.chunked_gdn_backward(inputs, &t, &t, None).map(|_| ()),
        "causal_conv1d_silu_forward" => b.causal_conv1d_silu_forward(&t, &t).map(|_| ()),
        "causal_conv1d_silu_backward" => b.causal_conv1d_silu_backward(&t, &t, &t).map(|_| ()),
        "gated_rms_norm_forward" => b.gated_rms_norm_forward(&t, &t, &t, 1.0).map(|_| ()),
        "gated_rms_norm_backward" => b.gated_rms_norm_backward(&t, &t, &t, &t, 1.0).map(|_| ()),
        "rope_partial_forward" => b.rope_partial_forward(&t, &t, &t).map(|_| ()),
        "rope_partial_backward" => b.rope_partial_backward(&t, &t, &t).map(|_| ()),
        "sigmoid_forward" => b.sigmoid_forward(&t).map(|_| ()),
        "sigmoid_backward" => b.sigmoid_backward(&t, &t).map(|_| ()),
        "gdn_log_decay_forward" => b.gdn_log_decay_forward(&t, &t, &t).map(|_| ()),
        "gdn_log_decay_backward" => b.gdn_log_decay_backward(&t, &t, &t, &t).map(|_| ()),
        "conv2d_forward" => b
            .conv2d_forward(&t, &t, None, Conv2dParams::new())
            .map(|_| ()),
        "conv_transpose2d_forward" => b
            .conv_transpose2d_forward(&t, &t, None, ConvTranspose2dParams::new())
            .map(|_| ()),
        "max_pool2d_forward" => b.max_pool2d_forward(&t, [2, 2], [2, 2]).map(|_| ()),
        "interpolate_bilinear_forward" => b
            .interpolate_bilinear_forward(&t, [2, 2], false)
            .map(|_| ()),
        "rope_interleaved_forward" => b.rope_interleaved_forward(&t, &t, &t).map(|_| ()),
        "rope_interleaved_backward" => b.rope_interleaved_backward(&t, &t, &t).map(|_| ()),
        "silu_forward" => b.silu_forward(&t).map(|_| ()),
        "silu_backward" => b.silu_backward(&t, &t).map(|_| ()),
        "mul_forward" => b.mul_forward(&t, &t).map(|_| ()),
        "mul_backward" => b.mul_backward(&t, &t, &t).map(|_| ()),
        "residual_add_forward" => b.residual_add_forward(&t, &t).map(|_| ()),
        "residual_add_backward" => b.residual_add_backward(&t, &t, &t).map(|_| ()),
        "layer_norm_forward" => b.layer_norm_forward(&t, &t, &t, 1e-6).map(|_| ()),
        "channel_layer_norm_forward" => b.channel_layer_norm_forward(&t, &t, &t, 1e-6).map(|_| ()),
        "gelu_erf_forward" => b.gelu_erf_forward(&t).map(|_| ()),
        "linear_bias_forward" => b.linear_bias_forward(&t, &t, &t).map(|_| ()),
        "broadcast_add_forward" => b.broadcast_add_forward(&t, &t).map(|_| ()),
        "broadcast_mul_forward" => b.broadcast_mul_forward(&t, &t).map(|_| ()),
        "scale_forward" => b.scale_forward(&t, 0.5).map(|_| ()),
        "softmax_forward" => b.softmax_forward(&t).map(|_| ()),
        "cross_entropy_mean_forward" => b.cross_entropy_mean_forward(&t, &t, None).map(|_| ()),
        "cross_entropy_mean_backward" => b.cross_entropy_mean_backward(&t, &t, None).map(|_| ()),
        "clip_grad_norm" => b.clip_grad_norm(&mut gs[..], 1.0).map(|_| ()),
        "adamw_step" => b
            .adamw_step(
                &mut m,
                &t,
                &mut m2,
                &mut m3,
                1,
                AdamWConfig::nanolab(1e-3, 0.0),
            )
            .map(|_| ()),
        "muon_ns5_step" => b
            .muon_ns5_step(&mut m, &t, &mut m2, MuonNs5Config::nanolab_default())
            .map(|_| ()),
        "sync" => b.sync().map(|_| ()),
        "accumulate_grad" => b.accumulate_grad(&mut m, &t).map(|_| ()),
        "scale_grad" => b.scale_grad(&mut m, 1.0).map(|_| ()),
        "linear_cross_entropy_mean" => b
            .linear_cross_entropy_mean(
                &t,
                &t,
                &t,
                None,
                CeChunk { rows: 1, cols: 1 },
                CeGrads::BOTH,
            )
            .map(|_| ()),
        "cached_attention_forward" => b.cached_attention_forward(&t, &t, &t, 1, None).map(|_| ()),
        "kv_cache_write" => b.kv_cache_write(&mut m, &t, 1).map(|_| ()),
        "argmax_rows" => b.argmax_rows(&t).map(|_| ()),
        "topk_rows" => b.topk_rows(&t, 1).map(|_| ()),
        "cast_bf16" => b.cast_bf16(&t).map(|_| ()),
        "to_bf16" => b.to_bf16(&t).map(|_| ()),
        "to_f32" => b.to_f32(&t).map(|_| ()),
        "autocast_region" => b.autocast_region(AutocastMode::Bf16).map(|_| ()),
        _ => return None,
    })
}

/// The `op` names, in trait order.
pub fn op_names() -> Vec<&'static str> {
    macro_rules! ops {
        ({} $($kind:ident fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty;)*) => {
            vec![$( (stringify!($kind), stringify!($name)) ),*]
        };
    }
    let all: Vec<(&'static str, &'static str)> = with_backend_methods!(ops, {});
    all.into_iter()
        .filter(|(kind, _)| *kind == "op")
        .map(|(_, name)| name)
        .collect()
}
