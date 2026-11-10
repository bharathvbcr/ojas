use crate::backend::BackendId;
use crate::dtype::DType;
use std::fmt;

/// Recoverable failure on a library path.
///
/// Library code returns this instead of panicking. [`OjasError::Poisoned`]
/// stays in the enum so existing matches keep compiling. [`crate::Budget`]
/// charges an atomic counter, and its trim-handler list recovers a poisoned
/// lock instead of failing, so no budget call returns `Poisoned`.
///
/// `#[non_exhaustive]`: new variants may be added (the typed
/// [`OjasError::WaitTimeout`] was), so a `match` outside `ojas-core` needs a
/// wildcard arm. Make it fail closed: an error this build does not know is
/// never success, and never a weaker kind than the nearest known one.
#[derive(Debug)]
#[non_exhaustive]
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
    /// callback fired, Metal's runtime was poisoned (a command buffer faulted
    /// on the GPU, or tessl's third consecutive event-wait timeout,
    /// `CONSECUTIVE_EVENT_WAIT_TIMEOUTS` in `tessl/src/runtime.rs`), or CUDA
    /// reported a sticky context error. A single wait that merely ran out of
    /// budget is not a loss: it is [`OjasError::WaitTimeout`]. Set by the
    /// backend where it detects the loss; loss is permanent, so every later
    /// call on that backend fails too. Open a new backend to continue.
    DeviceLost {
        backend: BackendId,
        detail: String,
    },
    /// A wait for the device to finish recorded work ran out of its budget
    /// before the work did. The work is slow, not faulted: it is still
    /// queued, nothing was reset or recycled early, and the backend stays
    /// usable. **Retryable**: the next sync point waits again, and a wait
    /// that completes means everything recorded finished. A call that failed
    /// before it committed anything (an upload, a read) can be repeated. A
    /// call whose work was already committed must not be: for a provider run
    /// (`MetalBackend::with_runtime`) the closure ran and its work is
    /// committed, not complete, so call `sync` until it returns `Ok` and never
    /// run the closure again, which would apply it twice.
    ///
    /// An op that had already committed its work before the wait (an in-place
    /// optimizer step, a cache write, a gradient accumulation) returns `Ok`,
    /// and its effect lands exactly once when the device finishes. The
    /// timeout is held and reported at the next sync point (`sync`, a
    /// download, an upload, a clip-norm) unless a later wait completes first;
    /// a completed wait clears it and nothing else does. A sync point that
    /// times out again keeps the work pending. A backend that gives up after
    /// repeated timeouts reports [`OjasError::DeviceLost`] instead (Metal:
    /// the third consecutive one). The streak counts every wait on the
    /// device, including the backend's own waits while it encodes an op
    /// (command-buffer backpressure), not only the waits a caller asked for:
    /// the op that takes the third one gets `DeviceLost`.
    ///
    /// `waited_ms` is the budget that ran out and `detail` is the backend's
    /// own description. Shared by every backend that bounds a device wait
    /// (Metal today, CUDA's wait as it adopts it).
    WaitTimeout {
        backend: BackendId,
        waited_ms: u64,
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
    /// call [`crate::Tensor::to_host`] or [`crate::BackendCore::upload`].
    Placement {
        op: &'static str,
        expected: Option<BackendId>,
        found: Option<BackendId>,
    },
}

impl OjasError {
    /// The one way to build [`OjasError::WaitTimeout`]: a wait on `backend`
    /// ran out of its `waited_ms` budget, and `detail` is the backend's own
    /// description. Every backend that classifies a timed-out wait goes
    /// through here, so the variant is built in one place.
    pub fn wait_timeout(backend: BackendId, waited_ms: u64, detail: impl Into<String>) -> Self {
        OjasError::WaitTimeout {
            backend,
            waited_ms,
            detail: detail.into(),
        }
    }
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
            OjasError::WaitTimeout {
                backend,
                waited_ms,
                detail,
            } => write!(
                f,
                "backend {backend:?}: device wait timed out after {waited_ms} ms (retryable): {detail}"
            ),
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
