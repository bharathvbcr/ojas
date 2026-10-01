//! Lane B. A tiny Metal training step on tessl.
//!
//! The step is one layer at `d_model` 128, two heads, head dimension 64,
//! vocabulary at most 128, sequence at most 16, and batch 1 or 2. It is not
//! the 124M model. Q and K take nanolab QK-norm and half-split RoPE, then a
//! square Q projection and causal `flash_attn_rows` at head dimension 64.
//! The loss updates that projection and the LM head. The shaders this crate
//! compiles are the per-head sigmoid gate and the tiny causal softmax backward.

#![forbid(unsafe_code)]

#[doc(hidden)]
pub use ojas_core::BackendId;

#[cfg(target_os = "macos")]
pub mod gpu;

#[cfg(not(target_os = "macos"))]
pub mod gpu {}
