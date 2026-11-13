//! The Qwen3.5 hybrid text tower (gated delta net and gated attention
//! layers) written once over [`crate::Graph`], so it trains on
//! `ojas_autograd::Tape` on any backend that implements the hybrid ops
//! (CPU, Metal) and runs eagerly on [`crate::Eval`].
//!
//! - [`Qwen35Spec`]: the dims. ojas-qwen35's `Qwen35TextConfig` parses a
//!   Hugging Face `config.json` and converts into it.
//! - [`load_hf`] reads a Hugging Face checkpoint into [`Qwen35Params`],
//!   splitting the fused tensors; [`fuse_grads`] is the inverse for
//!   gradients (`params`'s docs say how each fused tensor splits).
//!   [`load_hf_pinned`] is `load_hf` of a single file that must match an
//!   adapter checkpoint's [`adapters::BaseIdentity`] (name and sha256).
//! - [`FrozenWeight`] and [`load_hf_q8_pinned`]: a frozen base whose ten adapted linears stay
//!   packed (MLX affine q8) and resident, each dequantised per use by [`forward_loss_lora_base`]
//!   ([`bind_frozen_base`]); the dense entry points are the same code over
//!   [`Qwen35Params::frozen_view`].
//! - [`forward_loss`]: the tower to the fused tied-head cross-entropy, each
//!   layer optionally an activation-checkpointed segment.
//! - [`forward_hidden`]: the same tower to the final norm, for a host-side
//!   head; [`gather_rows`] reads one hidden row per example with a gradient,
//!   and [`RightPadded`] is the only padding it accepts.
//! - [`adapters`], [`bank`] and [`optim`]: the trainable-entry side of the
//!   LoRA track. [`bank::AdapterBank`] sums per-sequence gradients,
//!   [`optim::adamw_all_or_nothing`] steps every entry or none, and
//!   [`adapters`] writes and reads the adapter checkpoint
//!   (`docs/adapter-checkpoint.md`).
//! - [`step_provider::TapeStepProvider`] puts them together as an
//!   `ojas_train::lappi::step::StepProvider`: one tape per sequence, the
//!   bank, AdamW over the trainable entries and the adapter checkpoint.

pub mod adapters;
pub mod bank;
mod forward;
pub mod optim;
mod params;
mod spec;
pub mod step_provider;

pub use forward::{
    bind, bind_frozen, bind_frozen_base, forward_hidden, forward_hidden_lora,
    forward_hidden_lora_base, forward_loss, forward_loss_lora, forward_loss_lora_base, gather_rows,
    LoraForward, Qwen35Tables, RightPadded,
};
pub use params::{
    deinterleave_affine, fuse_grads, hf_tensors, load_hf, load_hf_pinned, load_hf_q8_pinned,
    lora_targets, split_rows_affine, AttnParams, FrozenWeight, GdnParams, HfKind, HfTensor,
    HfValues, LayerLora, LayerParams, LoraKind, LoraTarget, MixerLora, MixerParams, Qwen35Lora,
    Qwen35Params, LORA_KINDS,
};
pub use spec::{Qwen35Mixer, Qwen35Spec};
pub use step_provider::{DeviceHooks, LoraInit, OpenConfig, TapeStepProvider};
