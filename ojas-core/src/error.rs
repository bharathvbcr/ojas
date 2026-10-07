use crate::backend::BackendId;
use crate::dtype::DType;
use std::fmt;

/// Recoverable failure on a library path.
///
/// Library code returns this instead of panicking. [`OjasError::Poisoned`]
/// stays in the enum so existing matches keep compiling. [`crate::Budget`]
/// charges an atomic counter, so reservation does not take a lock and does
/// not return `Poisoned`.
#[derive(Debug)]
pub enum OjasError {
    Shape {
        op: &'static str,
        detail: String,
    },
    Dtype {
        op: &'static str,
        expected: DType,
        got: DType,
    },
    OutOfRange {
        op: &'static str,
        detail: String,
    },
    NonFinite {
        op: &'static str,
    },
    /// Head dimension the selected backend's kernel cannot run.
    /// Metal refuses values above [`crate::METAL_MAX_HEAD_DIM`].
    UnsupportedHeadDim {
        head_dim: u32,
        limit: u32,
    },
    /// The request would pass a configured byte cap, or the allocator refused it.
    /// The cap is not lowered to fit the request.
    CapacityExceeded {
        requested: u64,
        cap: u64,
        live: u64,
    },
    Poisoned,
    /// The backend's device can no longer run work: wgpu's device-lost
    /// callback fired, Metal's runtime was poisoned by a failed or timed-out
    /// command buffer, or CUDA reported a sticky context error. Set by the
    /// backend where it detects the loss; loss is permanent, so every later
    /// call on that backend fails too. Open a new backend to continue.
    DeviceLost {
        backend: BackendId,
        detail: String,
    },
    Backend {
        id: BackendId,
        detail: String,
    },
    /// The named backend cannot run this op. There is no CPU fallback.
    Unsupported {
        op: &'static str,
        detail: String,
    },
    /// The tensor lives on a different device than the call needs.
    /// `found` is `None` for host memory. Nothing is copied implicitly;
    /// call [`crate::Tensor::to_host`] or [`crate::Backend::upload`].
    Placement {
        op: &'static str,
        expected: Option<BackendId>,
        found: Option<BackendId>,
    },
}

impl fmt::Display for OjasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OjasError::Shape { op, detail } => write!(f, "{op}: shape: {detail}"),
            OjasError::Dtype { op, expected, got } => {
                write!(f, "{op}: dtype: expected {expected:?}, got {got:?}")
            }
            OjasError::OutOfRange { op, detail } => write!(f, "{op}: out of range: {detail}"),
            OjasError::NonFinite { op } => write!(f, "{op}: non-finite value"),
            OjasError::UnsupportedHeadDim { head_dim, limit } => {
                write!(f, "unsupported head dim {head_dim} (limit {limit})")
            }
            OjasError::CapacityExceeded {
                requested,
                cap,
                live,
            } => write!(
                f,
                "capacity exceeded: requested {requested} bytes, cap {cap}, live {live}"
            ),
            OjasError::Poisoned => write!(f, "poisoned"),
            OjasError::DeviceLost { backend, detail } => {
                write!(f, "backend {backend:?}: device lost: {detail}")
            }
            OjasError::Backend { id, detail } => write!(f, "backend {id:?}: {detail}"),
            OjasError::Unsupported { op, detail } => write!(f, "{op}: unsupported: {detail}"),
            OjasError::Placement {
                op,
                expected,
                found,
            } => {
                let place = |id: &Option<BackendId>| match id {
                    Some(id) => format!("{id:?} device"),
                    None => "host".to_string(),
                };
                write!(
                    f,
                    "{op}: tensor is on {}, expected {}",
                    place(found),
                    place(expected)
                )
            }
        }
    }
}

impl std::error::Error for OjasError {}
