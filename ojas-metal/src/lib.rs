//! Lane B. A tiny Metal training step on tessl.
//!
//! The step is one layer at `d_model` 128, two heads, head dimension 64,
//! vocabulary at most 128, sequence at most 16, and batch 1 or 2. It is not
//! the 124M model. Q and K take nanolab QK-norm and half-split RoPE, then a
//! square Q projection and causal `flash_attn_rows` at head dimension 64.
//! The loss updates that projection and the LM head. The step's own shaders
//! are the per-head sigmoid gate and the tiny causal softmax backward.

//!
//! [`MetalBackend`] is the general path: every `ojas_core::Backend` op on
//! device-resident tensors at any shape the kernels index, with head
//! dimension at most 64. Its kernels are in `kernels/ojas_backend.metal`,
//! including `permute` and the tiled causal attention backward.

#![forbid(unsafe_code)]
// Without a device thread nothing reads the command protocol.
#![cfg_attr(not(all(target_os = "macos", feature = "metal")), allow(dead_code))]

#[doc(hidden)]
pub use ojas_core::BackendId;

mod backend;
#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod gauge;
mod link;
mod memory;
mod permute;

pub use backend::{MetalBackend, MetalBuffer, METAL_TOPK_MAX_K};
pub use link::WaitCounts;
pub use memory::MetalMemory;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod device;

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod device {
    use std::sync::{mpsc, Arc};

    use ojas_core::OjasError;

    use crate::link::{DeviceInfo, Msg, Res, Waits};

    const NO_METAL: &str =
        "ojas-metal was built without Metal (needs macOS and the `metal` feature)";

    pub(crate) fn spawn(_waits: Arc<Waits>, _cap: u64) -> Res<(mpsc::Sender<Msg>, DeviceInfo)> {
        Err(OjasError::Unsupported {
            op: "MetalBackend::new",
            detail: NO_METAL.to_string(),
        })
    }

    /// tessl owns the GDN workspace sizes; without it there is no device to
    /// run the op on, so the backend never gets this far.
    pub(crate) fn gdn_backward_workspace_elems(_: u32, _: u32, _: u32, _: u32) -> Res<usize> {
        Err(OjasError::Unsupported {
            op: "chunked_gdn_backward",
            detail: NO_METAL.to_string(),
        })
    }

    pub(crate) fn gdn_forward_transient_elems(_: u32, _: u32, _: u32, _: u32) -> Res<usize> {
        Err(OjasError::Unsupported {
            op: "chunked_gdn_forward",
            detail: NO_METAL.to_string(),
        })
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod gpu;

#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub mod gpu {}
