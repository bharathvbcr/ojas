//! Messages between [`crate::MetalBackend`] handles and the device thread.
//!
//! tessl's `GpuRuntime` is `!Send`: it holds Metal objects behind `Retained`
//! pointers and a single host-access lease, and is not documented as safe to
//! use from more than one thread. The backend never moves it. One device
//! thread creates the runtime and every buffer, and owns them for their
//! whole lives. Handles on any thread send it a [`Cmd`] over an
//! `mpsc::Sender` (which is `Send + Sync`) and wait for the reply. A device
//! buffer crosses threads only as a `u64` id, so no `unsafe impl Send` is
//! needed anywhere in the crate.
//!
//! Commands run one at a time, in the order the channel delivers them.

use std::sync::mpsc;

use ojas_core::{BackendId, OjasError, MAX_PERMUTE_RANK};

pub(crate) type Res<T> = Result<T, OjasError>;

/// A contiguous element window of a device buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Arg {
    pub id: u64,
    /// Byte offset of element 0.
    pub off: usize,
    /// Element count, at most `u32::MAX`.
    pub n: usize,
}

/// A buffer the device thread allocated for an op output.
#[derive(Debug)]
pub(crate) struct NewBuf {
    pub id: u64,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RopeMode {
    /// cos/sin have the shape of x.
    Same,
    /// x is `[batch, time, heads, dim]`, cos/sin are `[time, dim]`.
    TimeDim { time: u32, heads: u32 },
}

/// One op. Dimensions are validated by the caller; the device thread checks
/// buffer ids and windows again before it binds anything.
#[derive(Debug)]
pub(crate) enum Cmd {
    Upload {
        bytes: Vec<u8>,
    },
    Read {
        id: u64,
        off: usize,
        len: usize,
    },
    Free {
        id: u64,
    },
    /// Test hook: encode a dispatch, then panic on the device thread, as a
    /// bug mid-op would.
    #[cfg(test)]
    InjectPanic,
    /// Contiguous copy with axes reordered. `oshape` is the output shape and
    /// `istride[a]` the input stride (elements) of the axis output axis `a`
    /// came from; entries past `rank` are 1 and 0.
    Permute {
        x: Arg,
        rank: u32,
        oshape: [u32; MAX_PERMUTE_RANK],
        istride: [u32; MAX_PERMUTE_RANK],
    },
    Embed {
        table: Arg,
        ids: Arg,
        vocab: u32,
        dim: u32,
    },
    EmbedBwd {
        table: Arg,
        ids: Arg,
        grad: Arg,
        vocab: u32,
        dim: u32,
    },
    Linear {
        x: Arg,
        w: Arg,
        rows: usize,
        kin: usize,
        nout: usize,
    },
    LinearBwd {
        x: Arg,
        w: Arg,
        gy: Arg,
        rows: usize,
        kin: usize,
        nout: usize,
    },
    Rms {
        x: Arg,
        w: Arg,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    RmsBwd {
        x: Arg,
        w: Arg,
        gy: Arg,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    Rope {
        x: Arg,
        cos: Arg,
        sin: Arg,
        rows: u32,
        dim: u32,
        mode: RopeMode,
        backward: bool,
    },
    Sdpa {
        q: Arg,
        k: Arg,
        v: Arg,
        bh: u32,
        t: u32,
        d: u32,
    },
    SdpaBwd {
        q: Arg,
        k: Arg,
        v: Arg,
        gy: Arg,
        bh: u32,
        t: u32,
        d: u32,
    },
    Gate {
        x: Arg,
        w: Arg,
        b: Arg,
        attn: Arg,
        rows: u32,
        din: u32,
        heads: u32,
        dh: u32,
    },
    GateBwd {
        x: Arg,
        w: Arg,
        b: Arg,
        attn: Arg,
        gy: Arg,
        rows: u32,
        din: u32,
        heads: u32,
        dh: u32,
    },
    Vres {
        v: Arg,
        v0: Arg,
        lam: Arg,
    },
    VresBwd {
        v: Arg,
        v0: Arg,
        lam: Arg,
        gy: Arg,
    },
    Silu {
        x: Arg,
    },
    SiluBwd {
        x: Arg,
        gy: Arg,
    },
    Mul {
        a: Arg,
        b: Arg,
    },
    MulBwd {
        a: Arg,
        b: Arg,
        gy: Arg,
    },
    Add {
        x: Arg,
        y: Arg,
    },
    AddBwd {
        x: Arg,
        y: Arg,
        gy: Arg,
    },
    Ce {
        logits: Arg,
        targets: Arg,
        rows: u32,
        vocab: u32,
        ignore: Option<u32>,
        grad: bool,
    },
    /// Checks every gradient and returns the global L2 norm.
    ClipNorm {
        grads: Vec<Arg>,
    },
    /// Multiplies every gradient by `scale` in place.
    Scale {
        grads: Vec<Arg>,
        scale: f32,
    },
    /// Writes `p`, `m`, `v` only if the whole step is finite.
    AdamW {
        p: Arg,
        g: Arg,
        m: Arg,
        v: Arg,
        /// `ojas_adamw_*`'s `OjasAdamW`, in field order.
        scalars: [f32; 7],
    },
    /// Writes `p` and `m` only if the whole step is finite.
    Muon {
        p: Arg,
        g: Arg,
        m: Arg,
        rows: usize,
        cols: usize,
        momentum: f32,
        nesterov: bool,
        decay: f32,
        alpha: f32,
    },
}

#[derive(Debug)]
pub(crate) enum Reply {
    Bufs(Vec<NewBuf>),
    Bytes(Vec<u8>),
    Norm(f32),
    Done,
}

pub(crate) struct Msg {
    pub cmd: Cmd,
    pub reply: Option<mpsc::SyncSender<Res<Reply>>>,
}

/// The sending end. Shared by a backend, its clones, and every buffer it
/// made; the device thread exits when the last one drops.
pub(crate) struct Link {
    tx: mpsc::Sender<Msg>,
    device_name: String,
}

impl Link {
    pub(crate) fn new(tx: mpsc::Sender<Msg>, device_name: String) -> Self {
        Self { tx, device_name }
    }

    pub(crate) fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Run `cmd` on the device thread and wait for its result.
    pub(crate) fn call(&self, cmd: Cmd) -> Res<Reply> {
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx
            .send(Msg {
                cmd,
                reply: Some(reply),
            })
            .map_err(|_| gone())?;
        rx.recv().map_err(|_| gone())?
    }

    /// Queue `cmd` without waiting. Used for frees, which cannot fail in a
    /// way the dropping thread could act on.
    pub(crate) fn post(&self, cmd: Cmd) {
        // A closed channel means the device thread has exited, and its
        // buffer table, with this buffer in it, is already gone.
        let _ = self.tx.send(Msg { cmd, reply: None });
    }
}

fn gone() -> OjasError {
    OjasError::Backend {
        id: BackendId::Metal,
        detail: "Metal device thread has exited".to_string(),
    }
}

/// Rows per partial sum of the RMSNorm weight gradient (`RMS_W_CHUNK` in
/// the Metal source).
pub(crate) const RMS_W_CHUNK: u32 = 64;

/// Partial sums the RMSNorm weight gradient is reduced through. The backend
/// charges `rows + chunks * dim` scratch for them and the rstd vector.
pub(crate) fn rms_w_chunks(rows: u32) -> u32 {
    rows.div_ceil(RMS_W_CHUNK)
}

pub(crate) fn metal_err(detail: impl Into<String>) -> OjasError {
    OjasError::Backend {
        id: BackendId::Metal,
        detail: detail.into(),
    }
}
