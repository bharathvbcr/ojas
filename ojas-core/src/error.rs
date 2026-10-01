use crate::backend::BackendId;
use crate::dtype::DType;
use std::fmt;

/// Recoverable failure on a library path.
///
/// Library code returns this instead of panicking. A poisoned budget lock is
/// [`OjasError::Poisoned`].
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
    Backend {
        id: BackendId,
        detail: String,
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
            OjasError::Backend { id, detail } => write!(f, "backend {id:?}: {detail}"),
        }
    }
}

impl std::error::Error for OjasError {}
