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
//! - [`forward_loss`]: the tower to the fused tied-head cross-entropy, each
//!   layer optionally an activation-checkpointed segment.

mod forward;
mod params;
mod spec;

pub use forward::{bind, forward_loss, Qwen35Tables};
pub use params::{
    fuse_grads, hf_tensors, load_hf, AttnParams, GdnParams, HfTensor, HfValues, LayerParams,
    MixerParams, Qwen35Params,
};
pub use spec::{Qwen35Mixer, Qwen35Spec};
