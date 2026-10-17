//! The Qwen3.5 hybrid text tower (gated delta net and gated attention
//! layers) written once over [`crate::Graph`], so it trains on
//! `ojas_autograd::Tape` on any backend that implements the hybrid ops
//! (CPU, Metal) and runs eagerly on [`crate::Eval`].
//!
//! - [`Qwen35Spec`]: the dims. ojas-qwen35's `Qwen35TextConfig` parses a
//!   Hugging Face `config.json` and converts into it.
//! - [`load_hf`] reads a Hugging Face checkpoint into [`Qwen35Params`],
//!   splitting the fused tensors; [`fuse_grads`] is the inverse for
//!   gradients ([`params`]'s docs say how each fused tensor splits).
//!   [`load_hf_pinned`] is `load_hf` of a single file that must match an
//!   adapter checkpoint's [`adapters::BaseIdentity`] (name and sha256).
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

pub mod adapters;
pub mod bank;
mod forward;
pub mod optim;
mod params;
mod spec;

pub use forward::{
    bind, bind_frozen, forward_hidden, forward_hidden_lora, forward_loss, forward_loss_lora,
    gather_rows, LoraForward, Qwen35Tables, RightPadded,
};
pub use params::{
    fuse_grads, hf_tensors, load_hf, load_hf_pinned, lora_targets, AttnParams, GdnParams, HfKind, HfTensor,
    HfValues, LayerLora, LayerParams, LoraKind, LoraTarget, MixerLora, MixerParams, Qwen35Lora,
    Qwen35Params, LORA_KINDS,
};
pub use spec::{Qwen35Mixer, Qwen35Spec};
