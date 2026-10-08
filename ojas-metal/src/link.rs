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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

use ojas_core::{BackendId, OjasError, MAX_PERMUTE_RANK};

pub(crate) type Res<T> = Result<T, OjasError>;

/// Why the device thread made a waited commit: the trigger that called it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wait {
    /// A host upload while work was recorded (tessl maps a buffer for the
    /// host only after a waited commit).
    Upload,
    /// A raw read of a device buffer (`download`, `to_host`).
    Read,
    /// `Backend::sync`.
    Sync,
    /// `clip_grad_norm`, which returns the norm to the host.
    ClipNorm,
    /// The bytes allocated since the last wait passed the memory cap.
    MemCap,
    /// The device's working set passed its share of the recommended size.
    WorkingSet,
    /// Every status slot was taken.
    SlabFull,
    /// An allocation failed, and dropped buffers only recycle after a wait.
    Recycle,
}

const WAIT_KINDS: usize = 8;

/// Waited commits by [`Wait`] trigger, shared by the device thread and
/// every handle on it.
#[derive(Default)]
pub(crate) struct Waits([AtomicU64; WAIT_KINDS]);

impl Waits {
    pub(crate) fn count(&self, why: Wait) {
        self.0[why as usize].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> WaitCounts {
        let at = |why: Wait| self.0[why as usize].load(Ordering::Relaxed);
        WaitCounts {
            upload: at(Wait::Upload),
            read: at(Wait::Read),
            sync: at(Wait::Sync),
            clip_norm: at(Wait::ClipNorm),
            mem_cap: at(Wait::MemCap),
            working_set: at(Wait::WorkingSet),
            slab_full: at(Wait::SlabFull),
            recycle: at(Wait::Recycle),
        }
    }
}

/// Waited GPU commits a Metal backend's device thread has made since it
/// opened, counted by what triggered each one. Every commit that waits is
/// counted under exactly one trigger, so [`WaitCounts::total`] is the
/// number of waits. An upload, read or sync with nothing recorded does not
/// wait and is not counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaitCounts {
    pub upload: u64,
    pub read: u64,
    pub sync: u64,
    pub clip_norm: u64,
    pub mem_cap: u64,
    pub working_set: u64,
    pub slab_full: u64,
    pub recycle: u64,
}

impl WaitCounts {
    pub fn total(&self) -> u64 {
        self.upload
            + self.read
            + self.sync
            + self.clip_norm
            + self.mem_cap
            + self.working_set
            + self.slab_full
            + self.recycle
    }

    /// The waits made after `earlier` was taken, trigger by trigger.
    pub fn since(&self, earlier: &WaitCounts) -> WaitCounts {
        WaitCounts {
            upload: self.upload.saturating_sub(earlier.upload),
            read: self.read.saturating_sub(earlier.read),
            sync: self.sync.saturating_sub(earlier.sync),
            clip_norm: self.clip_norm.saturating_sub(earlier.clip_norm),
            mem_cap: self.mem_cap.saturating_sub(earlier.mem_cap),
            working_set: self.working_set.saturating_sub(earlier.working_set),
            slab_full: self.slab_full.saturating_sub(earlier.slab_full),
            recycle: self.recycle.saturating_sub(earlier.recycle),
        }
    }
}

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

/// One RMSNorm operand set: `x [rows, dim]`, `w [dim]`, and for the
/// backward the output gradient `gy`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RmsSide {
    pub x: Arg,
    pub w: Arg,
    pub gy: Option<Arg>,
    pub rows: u32,
    pub dim: u32,
}

/// The gated delta rule's operands (`ojas_core::GdnInputs`) and dims, for
/// tessl's `gdn_train` kernels: `q`, `k` `[B, T, H, 128]`, `v`
/// `[B, T, H, v_dim]`, `g`, `beta` `[B, T, H]`, `s0` `[B, H, 128, v_dim]`.
/// `ws_elems` is the f32 count of tessl's backward workspace the backend
/// charged; the device refuses a command whose figure disagrees with tessl's.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnArgs {
    pub q: Arg,
    pub k: Arg,
    pub v: Arg,
    pub g: Arg,
    pub beta: Arg,
    pub s0: Option<Arg>,
    pub batch: u32,
    pub seq: u32,
    pub heads: u32,
    pub v_dim: u32,
    pub ws_elems: usize,
}

/// One causal attention launch: `bh` query planes of `[t, d]`, `rep` query
/// planes per KV plane (`bh / rep` KV planes), and the sliding window, `0`
/// for every earlier key (the backend sends a window below `t`, or `0`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SdpaGeom {
    pub bh: u32,
    pub t: u32,
    pub d: u32,
    pub rep: u32,
    pub window: u32,
}

/// Causal conv1d geometry, validated and within the kernels' limits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Conv1dGeom {
    pub batch: u32,
    pub seq: u32,
    pub channels: u32,
    pub width: u32,
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
    /// Wait for everything recorded, then report the pending fault.
    Sync,
    /// Wait for everything recorded, then copy the bytes out. A fault found
    /// stays pending.
    Read {
        id: u64,
        off: usize,
        len: usize,
    },
    Free {
        id: u64,
    },
    /// Read the device's working set and current allocation. Answered even
    /// on a poisoned backend: it records nothing and commits nothing.
    Memory,
    /// Release every freed buffer tessl's pool keeps for reuse, after a
    /// waited commit returns the ones freed since the last. Live buffers
    /// are untouched, and the pool's cap stays what it was.
    TrimPool,
    /// Carry small uploads inline while work is recorded (the default), or
    /// make every such upload wait, for before/after wait measurements.
    InlineUploads {
        on: bool,
    },
    /// Test hook: encode a dispatch, then panic on the device thread, as a
    /// bug mid-op would.
    #[cfg(test)]
    InjectPanic,
    /// Test hook: the next `Read` fails after its wait, as a mapping failure
    /// would.
    #[cfg(test)]
    FailNextRead,
    /// Test hook: set tessl's runtime poison, as a command buffer that
    /// failed or timed out on the GPU does.
    #[cfg(test)]
    PoisonRuntime,
    /// Test hook: set tessl's pool cache cap, for an A/B of the cap a
    /// backend opens with against another.
    #[cfg(test)]
    SetPoolCap {
        bytes: usize,
    },
    /// Test hook: override the commit triggers, and after letting
    /// `fail_allocs.0` device allocations through, fail the next
    /// `fail_allocs.1` as an exhausted device would.
    #[cfg(test)]
    Tune {
        overlap: Option<usize>,
        mem_cap: Option<u64>,
        ws_limit: Option<u64>,
        fail_allocs: (u32, u32),
    },
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
    /// RMSNorm forward, or backward when the sides carry `gy`, of one side
    /// (`rms_norm_*`) or two (q then k, `rms_qk_norm_*`) in one command.
    Rms {
        sides: Vec<RmsSide>,
        eps: f32,
    },
    /// Half-split RoPE on the leading `rotary` values of each row of
    /// `dim`, the rest copied; `rotary == dim` rotates the whole row. `op`
    /// names a fault.
    Rope {
        op: &'static str,
        x: Arg,
        cos: Arg,
        sin: Arg,
        rows: u32,
        dim: u32,
        rotary: u32,
        mode: RopeMode,
        backward: bool,
    },
    /// Replies with the output and the `[bh, t]` row log-sum-exp.
    Sdpa {
        q: Arg,
        k: Arg,
        v: Arg,
        geom: SdpaGeom,
    },
    /// `[q, k, v, output, lse, grad_output]`; replies with dQ, dK, dV.
    SdpaBwd {
        args: [Arg; 6],
        geom: SdpaGeom,
    },
    /// With `save`, the reply carries the per-head sigmoid `[rows, heads]`
    /// after the output.
    Gate {
        x: Arg,
        w: Arg,
        b: Arg,
        attn: Arg,
        rows: u32,
        din: u32,
        heads: u32,
        dh: u32,
        save: bool,
    },
    /// With `scales` (a saving forward's sigmoid), the logits are not
    /// recomputed and `b` is not read.
    GateBwd {
        x: Arg,
        w: Arg,
        b: Arg,
        attn: Arg,
        gy: Arg,
        scales: Option<Arg>,
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
    /// The gated delta rule's forward: `o`, the final state, the
    /// checkpoints.
    Gdn {
        x: GdnArgs,
    },
    /// Its backward: `dq`, `dk`, `dv`, `dg`, `dbeta`, then `ds0` when the
    /// forward had an initial state.
    GdnBwd {
        x: GdnArgs,
        ckpt: Arg,
        d_o: Arg,
        d_fin: Option<Arg>,
    },
    /// Depthwise causal conv + SiLU over `[batch, seq, channels]` from a
    /// zero state, `w` `[channels, width]`.
    Conv1d {
        x: Arg,
        w: Arg,
        geom: Conv1dGeom,
    },
    /// Its backward: `dx`, then `dw`. `part` is the weight-gradient
    /// scratch the backend charged, in f32 values.
    Conv1dBwd {
        x: Arg,
        w: Arg,
        gy: Arg,
        geom: Conv1dGeom,
        part: usize,
    },
    /// Gated RMSNorm over rows of `dim`, `w` `[dim]`.
    GatedRms {
        x: Arg,
        z: Arg,
        w: Arg,
        rows: u32,
        dim: u32,
        eps: f32,
    },
    /// Its backward: `dx`, `dz`, then `dw`. `part` as for [`Cmd::Conv1dBwd`].
    GatedRmsBwd {
        x: Arg,
        z: Arg,
        w: Arg,
        gy: Arg,
        rows: u32,
        dim: u32,
        eps: f32,
        part: usize,
    },
    RoundBf16 {
        x: Arg,
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
        /// Targets neither ignored nor out of range, counted on the host.
        valid: u32,
        grad: bool,
    },
    /// Checks every gradient, waits, reports a pending fault (this
    /// command's included), and otherwise returns the global L2 norm.
    ClipNorm {
        grads: Vec<Arg>,
    },
    /// Multiplies every gradient by `scale` in place.
    Scale {
        grads: Vec<Arg>,
        scale: f32,
    },
    /// Mean cross-entropy of `x @ w^T` over `t`, one logits tile at a time;
    /// returns the loss and, with `geom.grad`, grad_input and grad_weight.
    LinearCe {
        x: Arg,
        w: Arg,
        t: Arg,
        geom: LceGeom,
        ignore: Option<u32>,
        /// Targets neither ignored nor out of range, counted on the host.
        valid: u32,
    },
    /// Grouped-query causal attention of `q [B, Tq, H, D]` against the first
    /// `kv_len` positions of `k`, `v [B, cap, Hkv, D]`.
    CachedAttn {
        q: Arg,
        k: Arg,
        v: Arg,
        batch: u32,
        tq: u32,
        heads: u32,
        kv_heads: u32,
        d: u32,
        cap: u32,
        kv_len: u32,
    },
    /// `cache[b, at..at + tn] = src[b]`, in place, only if `src` is finite.
    /// `row` is `Hkv * D`.
    KvWrite {
        cache: Arg,
        src: Arg,
        batch: u32,
        tn: u32,
        cap: u32,
        row: u32,
        at: u32,
    },
    /// `acc += g`, written only if every input and sum is finite. In place
    /// when `in_place`; otherwise into a new buffer that starts as a copy of
    /// `acc`, which is returned.
    Accumulate {
        acc: Arg,
        g: Arg,
        in_place: bool,
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
        /// Newton-Schulz at `Ns5Precision::Bf16`.
        bf16: bool,
    },
}

#[derive(Debug)]
pub(crate) enum Reply {
    Bufs(Vec<NewBuf>),
    Bytes(Vec<u8>),
    Norm(f32),
    Memory(crate::MetalMemory),
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
    waits: Arc<Waits>,
}

impl Link {
    pub(crate) fn new(tx: mpsc::Sender<Msg>, device_name: String, waits: Arc<Waits>) -> Self {
        Self {
            tx,
            device_name,
            waits,
        }
    }

    pub(crate) fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Waited GPU commits the device thread has made so far, by trigger.
    pub(crate) fn wait_counts(&self) -> WaitCounts {
        self.waits.snapshot()
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

/// Tiling of a linear cross-entropy call, computed once on the host so the
/// budget charge and the device's allocations come from the same numbers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LceGeom {
    pub n: usize,
    pub d: usize,
    pub v: usize,
    /// Rows and vocabulary columns of one logits tile.
    pub rt: usize,
    pub ct: usize,
    /// Whether some x (or weight) row tile starts at a byte offset tessl's
    /// GEMMs cannot take (not 16-aligned), so it is staged in a copy.
    pub stage_x: bool,
    pub stage_w: bool,
    pub grad: bool,
}

/// Whether a row tile of a `[rows, step]` matrix at byte offset `off`, cut
/// every `tile` rows, can start off 16-byte alignment.
fn needs_stage(off: usize, step: usize, tile: usize, rows: usize) -> bool {
    let tiles = rows.div_ceil(tile);
    let tile_bytes = tile.saturating_mul(step).saturating_mul(4);
    !(off.is_multiple_of(16) && (tiles == 1 || tile_bytes.is_multiple_of(16)))
}

/// Partial sums `ojas_reduce_partial` uses for `n` values.
pub(crate) fn reduce_groups(n: usize) -> usize {
    n.div_ceil(4096).clamp(1, 1024)
}

impl LceGeom {
    /// `rows` and `cols` are the caller's chunk, clamped to the problem and
    /// to the GEMMs' i32 element limit for one tile (a smaller tile only
    /// lowers scratch).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        n: usize,
        d: usize,
        v: usize,
        rows: usize,
        cols: usize,
        x_off: usize,
        w_off: usize,
        grad: bool,
    ) -> Self {
        let limit = i32::MAX as usize;
        let rt = rows.min(n).min(limit).max(1);
        let ct = cols.min(v).min(limit / rt).max(1);
        Self {
            n,
            d,
            v,
            rt,
            ct,
            stage_x: needs_stage(x_off, d, rt, n),
            stage_w: needs_stage(w_off, d, ct, v),
            grad,
        }
    }

    /// Elements of device scratch: the logits tile, per-row (max, sum,
    /// target logit), the loss rows and their reduction, and with gradients
    /// the two GEMM destinations; plus any staging copies.
    pub(crate) fn scratch(&self) -> Option<usize> {
        let mut total = self.rt.checked_mul(self.ct)?;
        let mut add = |x: usize| -> Option<()> {
            total = total.checked_add(x)?;
            Some(())
        };
        add(self.rt.checked_mul(3)?)?;
        add(self.n)?;
        add(1 + reduce_groups(self.n))?;
        let x_tile = self.rt.checked_mul(self.d)?;
        let w_tile = self.ct.checked_mul(self.d)?;
        if self.grad {
            add(x_tile)?;
            add(w_tile)?;
        }
        if self.stage_x {
            add(x_tile)?;
        }
        if self.stage_w {
            add(w_tile)?;
        }
        Some(total)
    }
}

pub(crate) fn metal_err(detail: impl Into<String>) -> OjasError {
    OjasError::Backend {
        id: BackendId::Metal,
        detail: detail.into(),
    }
}

/// The Metal device can no longer run work (tessl poisoned its runtime).
pub(crate) fn device_lost(detail: impl Into<String>) -> OjasError {
    OjasError::DeviceLost {
        backend: BackendId::Metal,
        detail: detail.into(),
    }
}
