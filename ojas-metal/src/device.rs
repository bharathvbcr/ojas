//! The device thread: macOS with the `metal` feature.
//!
//! It creates the tessl runtime, owns every buffer by id, and runs one
//! [`Cmd`] at a time. An op records its dispatches into tessl's open command
//! buffer and returns without waiting (`docs/metal-deferred-faults.md`). Its
//! finite checks write a slot of the status slab, which the worker reads,
//! in recording order, after each waited commit; the first faulting op
//! becomes the pending fault, which a sync point reports once.
//!
//! Commits: a sync point or read waits; every [`OVERLAP_DISPATCHES`] the
//! command buffer is committed without a wait so the GPU runs while the host
//! records; and a waited commit that only holds faults keeps memory bounded
//! (bytes allocated since the last wait, device working-set growth, a full
//! slab, an allocation that failed while dropped buffers awaited recycling).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::thread;

use ojas_core::{sdpa_scale, OjasError, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS};
use tessl::dispatch::{self, set_f32, set_gpu_buf_offset, set_u32, Binder};
use tessl::gdn_train::{
    gdn_train_backward, gdn_train_forward, gdn_train_forward_transient_bytes, GdnTrainDims,
    GdnTrainGrads, GdnTrainInputs, GdnTrainWorkspace, GDN_TRAIN_DK,
};
use tessl::gemm::{transpose_f32_into, GemmOperands};
use tessl::qwen35::{conv1d_silu, gated_rms_norm, Cols, OutCols, StateIn};
use tessl::qwen35_bwd::{
    conv1d_silu_bwd, conv1d_silu_bwd_part_len, gated_rms_norm_bwd, gated_rms_norm_bwd_part_len,
};
use tessl::runtime::GpuRuntime;
use tessl::sq_norm::{decode_sq_norm, SqNormPlan, SqWindow};
use tessl::tensor::{gpu_copy, GpuBuffer, Tensor as TT};
use tessl::DType;

use crate::gpu::gate_dbias_threads;
use crate::link::{
    cached_attn_origin, cached_attn_part, cached_attn_splits, copied_at, device_lost, metal_err,
    norm_chunks, reduce_groups, rms_w_chunks, Arg, AttnGeom, Cmd, Conv1dGeom, DeviceInfo,
    EmbedRows, GdnArgs, Lane, LceGeom, Msg, NewBuf, ReadAs, Reply, Res, RmsSide, RopeMode,
    SdpaGeom, Wait, Waits, SQ_OUT_WORDS, SQ_PIECE,
};

const ST_IN: u32 = 0;
const ST_OUT: u32 = 1;
const ST_WORDS: usize = 8;
/// Floats each `ojas_check_finite` thread checks. Measured with
/// `metal_bench kernels` over `[4096, 2048]`: 225 µs at 1, 136–147 µs at 2–32,
/// 152 µs at 64. 4 had the best minimum and keeps the most threads in flight
/// for mid-size tensors.
const CHECK_PER_THREAD: usize = 4;
const ST_BYTES: usize = ST_WORDS * 4;
/// Gradients one `ojas_scale_multi` dispatch binds (`NORM_SLOTS` in the
/// Metal source).
const NORM_SLOTS: usize = 24;

// link.rs builds without tessl, so it mirrors these; they must agree.
const _: () = assert!(SQ_PIECE == tessl::sq_norm::SQ_PIECE);
const _: () = assert!(SQ_OUT_WORDS == tessl::sq_norm::SQ_OUT_WORDS);

/// Status slots in the slab. A full slab forces a waited commit.
const SLAB_SLOTS: usize = 4096;
/// Dispatches between unwaited commits.
const OVERLAP_DISPATCHES: usize = 256;
/// Largest upload carried inline (in tessl's constant arena) while work is
/// recorded. Larger uploads wait and map.
const INLINE_UPLOAD_BYTES: usize = 64 << 10;
/// Inline upload bytes allowed between waited commits. tessl resets its
/// 16 MiB constant arena only at a waited commit, and an exhausted arena
/// poisons the runtime, so inline uploads keep to a sixteenth of it and the
/// rest stays for the dispatches' own constants.
const INLINE_ARENA_BYTES: usize = 1 << 20;

/// Whether `len` upload bytes may ride inline after `used` inline bytes
/// since the last wait: non-empty, whole words, and inside both caps.
fn inline_upload_fits(len: usize, used: usize) -> bool {
    len > 0
        && len.is_multiple_of(4)
        && len <= INLINE_UPLOAD_BYTES
        && used.saturating_add(len) <= INLINE_ARENA_BYTES
}

/// Ceiling of the bytes allocated between waited commits; the default cap is
/// the smaller of this and a quarter of the backend's budget.
const MEM_CAP_CEILING: u64 = 1 << 30;
/// Ceiling of the freed buffers tessl's pool keeps for reuse. tessl's own
/// default is 2 GiB whatever the backend's budget, outside it.
const POOL_CACHE_CEILING: u64 = 1 << 30;

/// The pool cache cap of a backend opened with `budget_cap`: a quarter of
/// the budget, at most [`POOL_CACHE_CEILING`]. The cached buffers are not
/// charged to the budget, so the cap bounds what this backend can hold on
/// the device past it, and [`crate::MetalMemory::pool_cache_cap`] reports
/// it to the plan, which sets it aside from the device's room.
pub(crate) fn pool_cache_cap(budget_cap: u64) -> u64 {
    POOL_CACHE_CEILING.min(budget_cap / 4)
}
/// Device memory above this share of the recommended working set forces a
/// waited commit once this backend has allocated more than [`WS_GROWTH`]
/// since its last one. The device figure is every allocation on the device,
/// other backends' included; the backend's own count keeps a steady state
/// (or another backend's growth) from making every op wait.
const WS_NUM: u64 = 3;
const WS_DEN: u64 = 4;
const WS_GROWTH: u64 = 64 << 20;
/// Tiled attention geometry, as `ATT_THREADS` (four simdgroups) and
/// `ATT_BQ` / `ATT_BK` (rows per threadgroup) in the Metal source.
/// The forward kernel's key tile is wider (`ATT_FWD_BK`); its grid is
/// still one threadgroup per query block.
const ATTN_THREADS: usize = 128;
const ATTN_ROWS: usize = 32;
/// Largest head dim the tiled attention kernels are compiled for.
const ATTN_MAX_HEAD_DIM: u32 = 256;
/// Threads per `ojas_cached_attn` threadgroup (`CA_THREADS`: 32
/// simdgroups, so a decode step's few threadgroups each walk 32 keys per
/// simdgroup at 1024 positions). tessl's dispatch refuses a threadgroup
/// larger than the pipeline allows, so a device that cannot run that many
/// fails the call rather than running it short.
const CA_THREADS: usize = 1024;
/// Every kernel `MetalBackend` dispatches from the crate's metallib. Start-up
/// fails if one is missing, not the first op that needs it.
const KERNELS: &[&str] = &[
    "ojas_check_finite",
    "ojas_upload_words",
    "ojas_adamw_check",
    "ojas_adamw_apply",
    "ojas_copy_if_clean",
    "ojas_acc_check",
    "ojas_acc_apply",
    "ojas_kv_write",
    "ojas_lce_stats",
    "ojas_lce_loss",
    "ojas_lce_grad",
    "ojas_add_into",
    "ojas_cached_attn",
    "ojas_cached_attn_merge",
    "ojas_permute",
    "ojas_bcast_binary",
    "ojas_scale_fwd",
    "ojas_softmax_rows",
    "ojas_round_bf16",
    "ojas_bf16_to_f32",
    "ojas_check_finite_bf16",
    "ojas_f32_to_bf16",
    "ojas_silu_fwd",
    "ojas_silu_bwd",
    "ojas_sigmoid_fwd",
    "ojas_sigmoid_bwd",
    "ojas_gdn_decay_fwd",
    "ojas_gdn_decay_bwd",
    "ojas_gdn_decay_bwd_sum",
    "ojas_mul_fwd",
    "ojas_mul_bwd",
    "ojas_add_fwd",
    "ojas_add_bwd",
    "ojas_axpby",
    "ojas_axpby_bf16",
    "ojas_axpby_fma",
    "ojas_div_scalar",
    "ojas_ns_denom",
    "ojas_ns_denom_bf16",
    "ojas_vres_fwd",
    "ojas_vres_bwd",
    "ojas_vres_lambda",
    "ojas_rope",
    "ojas_embed_fwd",
    "ojas_embed_fwd_bf16",
    "ojas_argmax_rows",
    "ojas_topk_rows",
    "ojas_embed_gather",
    "ojas_reduce_partial",
    "ojas_reduce_final",
    "ojas_scale_multi",
    "ojas_ce_fused",
    "ojas_ce_mean",
    "ojas_rms_fwd",
    "ojas_rms_bwd_rows",
    "ojas_rms_bwd_w_part",
    "ojas_rms_bwd_w_sum",
    "ojas_attn_fwd_d16",
    "ojas_attn_fwd_d32",
    "ojas_attn_fwd_d64",
    "ojas_attn_fwd_d128",
    "ojas_attn_bwd_dr",
    "ojas_attn_bwd_dq_d16",
    "ojas_attn_bwd_dq_d32",
    "ojas_attn_bwd_dq_d64",
    "ojas_attn_bwd_dkv_d16",
    "ojas_attn_bwd_dkv_d32",
    "ojas_attn_bwd_dkv_d64",
    "ojas_attn_bwd_dq_d128",
    "ojas_attn_bwd_dkv_d128",
    "ojas_attn_fwd_d256",
    "ojas_attn_gen_fwd",
    "ojas_attn_bwd_dq_d256",
    "ojas_attn_bwd_dkv_d256",
    "ojas_per_head_gate_fwd",
    "ojas_per_head_gate_bwd",
    "ojas_per_head_gate_scale",
    "ojas_per_head_gate_bwd_saved",
    "ojas_per_head_gate_dbias",
];

/// tessl's dims for a dense GDN call (one key head per value head).
fn gdn_dims(batch: u32, seq: u32, heads: u32, v_dim: u32) -> GdnTrainDims {
    GdnTrainDims {
        batch,
        seq,
        heads,
        v_dim,
    }
}

/// f32 values of tessl's backward workspace for these dims, from tessl's own
/// `GdnTrainWorkspace::bytes_for`, so the backend's charge follows whichever
/// kernel tessl routes production through.
pub(crate) fn gdn_backward_workspace_elems(
    batch: u32,
    seq: u32,
    heads: u32,
    v_dim: u32,
) -> Res<usize> {
    Ok(GdnTrainWorkspace::bytes_for(gdn_dims(batch, seq, heads, v_dim)) / 4)
}

/// f32 values the forward allocates and frees inside one call, from tessl's
/// `gdn_train_forward_transient_bytes`. That figure is pool-rounded, so the
/// charge is at least the logical bytes.
pub(crate) fn gdn_forward_transient_elems(
    batch: u32,
    seq: u32,
    heads: u32,
    v_dim: u32,
) -> Res<usize> {
    let bytes = gdn_train_forward_transient_bytes(gdn_dims(batch, seq, heads, v_dim), heads);
    usize::try_from(bytes.div_ceil(4)).map_err(|_| {
        metal_err(format!(
            "chunked_gdn_forward: {bytes} transient bytes overflow usize"
        ))
    })
}

/// Start the device thread and wait until its runtime is up.
pub(crate) fn spawn(waits: Arc<Waits>, budget_cap: u64) -> Res<(mpsc::Sender<Msg>, DeviceInfo)> {
    spawn_with(move || Worker::open(waits, budget_cap))
}

/// As [`spawn`], with the opener the tests can fail. A failed or panicked
/// opener is joined before this returns, so the owner thread does not stay
/// in the process. A successful opener is detached: it runs until its channel
/// closes.
fn spawn_with(
    open: impl FnOnce() -> Res<Worker> + Send + 'static,
) -> Res<(mpsc::Sender<Msg>, DeviceInfo)> {
    let (tx, rx) = mpsc::channel::<Msg>();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Res<DeviceInfo>>(1);
    let handle = thread::Builder::new()
        .name("ojas-metal-device".to_string())
        .spawn(move || {
            let opened = catch_unwind(AssertUnwindSafe(open))
                .unwrap_or_else(|_| Err(metal_err("Metal runtime start-up panicked")));
            let mut worker = match opened {
                Ok(worker) => worker,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(DeviceInfo {
                name: worker.rt.device_name(),
                tensorops: worker.rt.has_tensorops(),
            }));
            drop(ready_tx);
            worker.serve(rx);
        })
        .map_err(|e| metal_err(format!("could not spawn the Metal device thread: {e}")))?;
    match ready_rx.recv() {
        Ok(Ok(device)) => {
            drop(handle);
            Ok((tx, device))
        }
        Ok(Err(err)) => {
            let _ = handle.join();
            Err(err)
        }
        Err(_) => {
            let _ = handle.join();
            Err(metal_err("Metal device thread exited during start-up"))
        }
    }
}

/// A contiguous f32 or u32 window of a buffer.
#[derive(Clone)]
struct V {
    buf: GpuBuffer,
    off: usize,
    n: usize,
}

/// The weight of a linear cross-entropy: an `F32` view, or a bf16 window
/// widened one tile at a time.
enum LceWeight {
    F32(V),
    Half(Half),
}

impl LceWeight {
    fn n(&self) -> usize {
        match self {
            LceWeight::F32(v) => v.n,
            LceWeight::Half(h) => h.n,
        }
    }
}

/// A contiguous bf16 window: `n` 2-byte elements from element `first` of a
/// binding at byte `base`, the 4-byte boundary at or below the window's
/// start (the kernels' bind alignment).
#[derive(Clone)]
struct Half {
    buf: GpuBuffer,
    base: usize,
    first: usize,
    n: usize,
}

fn bind(b: &mut Binder<'_>, v: &V, index: usize) {
    set_gpu_buf_offset(b, &v.buf, v.off, index);
}

/// The attention kernels' matrix-unit tensors index one `[T, D]` plane with
/// i32 extents.
fn attn_plane_fits(op: &'static str, t: u32, d: u32) -> Res<()> {
    if u64::from(t) * u64::from(d) > i32::MAX as u64 {
        return Err(OjasError::Unsupported {
            op,
            detail: format!("a [T, D] = [{t}, {d}] plane exceeds i32 indexing"),
        });
    }
    Ok(())
}

/// RMSNorm geometry, as `RMS_ROWS_PER_TG` (one simdgroup per row) and
/// `RMS_W_CHUNK` in the Metal source.
const RMS_THREADS: usize = 256;
const RMS_ROWS_PER_TG: usize = 8;

fn rms_groups(rows: u32) -> usize {
    (rows as usize).div_ceil(RMS_ROWS_PER_TG)
}

/// The slot table of one `ojas_scale_multi` dispatch over `group` (at most
/// [`NORM_SLOTS`] views): per slot its element count and the index of its
/// first chunk in the dispatch, as native-endian bytes; and the dispatch's
/// chunk count, its threadgroups.
fn norm_table(group: &[V]) -> Res<(Vec<u8>, usize)> {
    let mut tbl = Vec::with_capacity(group.len() * 8);
    let mut at = 0usize;
    for v in group {
        tbl.extend_from_slice(&u32_of(v.n)?.to_ne_bytes());
        tbl.extend_from_slice(&u32_of(at)?.to_ne_bytes());
        at += norm_chunks(v.n);
    }
    u32_of(at)?;
    Ok((tbl, at))
}

/// Elements `start..start + n` of a view.
fn sub(v: &V, start: usize, n: usize) -> V {
    V {
        buf: v.buf.clone(),
        off: v.off + start * 4,
        n,
    }
}

fn u32_of(n: usize) -> Res<u32> {
    u32::try_from(n).map_err(|_| OjasError::Unsupported {
        op: "MetalBackend",
        detail: format!("{n} exceeds the kernels' 32-bit indexing"),
    })
}

/// One op's status slot, in recording order.
struct Entry {
    slot: usize,
    op: &'static str,
    /// The command that took it.
    cmd: u64,
    /// Kept if its command later fails (`rms_sides`: q's side is the
    /// composition's first op, which would have returned `Ok`).
    sealed: bool,
    /// The buffer whose [`Book::clean`] mark this slot's check earned
    /// (`Worker::freeze`). A fault in the slot withdraws the mark.
    clean_of: Option<u64>,
}

/// A buffer known finite and never written: a frozen parameter after
/// [`Worker::freeze`], or an exact widening of one. Ops skip their finite
/// scan of an operand window inside `windows`, and every in-place op refuses
/// the buffer, so the mark cannot go stale.
struct Clean {
    /// Byte ranges `[start, end)` the check covered.
    windows: Vec<(usize, usize)>,
    /// The frozen buffer this one was widened from. Withdrawing that mark
    /// withdraws this one too.
    from: Option<u64>,
}

/// Recording state between waited commits.
struct Book {
    /// Slots taken since the slab was last cleared.
    used: usize,
    pending: Vec<Entry>,
    /// The command being run.
    cmd: u64,
    /// The first fault found and not yet reported.
    fault: Option<&'static str>,
    /// Work has been encoded since the last waited commit.
    dirty: bool,
    since_overlap: usize,
    alloc_since_wait: u64,
    /// Bytes of buffers freed ([`Cmd::Free`]) since the last waited
    /// commit. tessl returns a dropped buffer to its pool only once a
    /// commit after the drop completes, so these are still on the device
    /// though their charges are off the budget.
    freed_since_wait: u64,
    /// Upload bytes carried in tessl's constant arena since the last
    /// waited commit, which is the only point tessl resets the arena.
    inline_since_wait: usize,
    /// Small uploads behind recorded work ride inline; off, they wait.
    inline_uploads: bool,
    /// Buffers whose finite scan an op may skip, by buffer id. Ids are never
    /// reused, and [`Worker::free`] drops the entry with the buffer.
    clean: HashMap<u64, Clean>,
    overlap: usize,
    mem_cap: u64,
    ws_limit: u64,
    #[cfg(test)]
    /// Allocations to let through, then allocations to fail.
    fail_allocs: (u32, u32),
    #[cfg(test)]
    fail_next_read: bool,
}

/// What one per-head gate command runs.
#[derive(Clone, Copy)]
enum GateMode {
    Forward,
    /// The forward, and the per-head sigmoid `[rows, heads]` after it.
    ForwardSaving,
    /// Gradient `gy`, recomputing the logits.
    Backward(Arg),
    /// Gradient `gy` with a saving forward's sigmoid.
    BackwardSaved(Arg, Arg),
}

/// Which pending slots a waited commit may scan.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scan {
    All,
    /// Not the current command's: its kernels may still be unencoded.
    BeforeCurrent,
}

/// A status slot to bind.
struct St {
    buf: GpuBuffer,
    off: usize,
}

fn bind_st(b: &mut Binder<'_>, st: &St, index: usize) {
    set_gpu_buf_offset(b, &st.buf, st.off, index);
}

/// How tessl reported a failed device allocation.
///
/// tessl's allocator returns its failures as `String`s, so this is the one
/// place ojas reads them: no other code in this crate decides "out of
/// memory" from a message. `contract_matches_tessls_source` pins the
/// fragments below to tessl's own source, and the oversize tests run them
/// against tessl's real refusals.
#[derive(Debug, PartialEq, Eq)]
enum AllocFailure {
    /// The request is past what the device can ever hold (tessl's
    /// "exceeds host/device allocation limit" and "rounded buffer size
    /// exceeds device limit"). Permanent: a waited commit does not cure it.
    Limit,
    /// The device refused the buffer (tessl's `newBuffer({key}) failed`).
    /// Buffers dropped since the last waited commit return to the pool only
    /// at the next one, so a waited commit might cure it.
    Exhausted,
    /// Not an allocation failure.
    Other,
}

fn classify_alloc(e: &str) -> AllocFailure {
    if e.contains("exceeds host/device allocation limit") || e.contains("exceeds device limit") {
        AllocFailure::Limit
    } else if is_new_buffer_failure(e) {
        AllocFailure::Exhausted
    } else {
        AllocFailure::Other
    }
}

/// tessl's `newBuffer({key}) failed`, with `key` the bucket size in decimal.
fn is_new_buffer_failure(e: &str) -> bool {
    e.strip_prefix("newBuffer(")
        .and_then(|rest| rest.strip_suffix(") failed"))
        .is_some_and(|key| !key.is_empty() && key.bytes().all(|b| b.is_ascii_digit()))
}

struct Worker {
    rt: Arc<GpuRuntime>,
    bufs: HashMap<u64, GpuBuffer>,
    next_id: u64,
    poisoned: bool,
    /// Waited commits, shared with every handle (`MetalBackend::waits`).
    waits: Arc<Waits>,
    slab: GpuBuffer,
    /// One word per `ojas_scale_multi` slot, at offset `4 * slot`, bound in
    /// the slots a short last group leaves unused. The kernel never touches
    /// them; binding these, not a repeat of a gradient, keeps every
    /// read-write binding of a dispatch on memory of its own.
    spare: GpuBuffer,
    book: RefCell<Book>,
}

impl Worker {
    fn open(waits: Arc<Waits>, budget_cap: u64) -> Res<Self> {
        let rt = GpuRuntime::new().map_err(metal_err)?;
        // Before the first allocation, so no buffer is ever cached past it.
        let pool_cap = usize::try_from(pool_cache_cap(budget_cap)).unwrap_or(usize::MAX);
        rt.set_pool_cache_cap_bytes(pool_cap);
        rt.set_async_encode(true).map_err(metal_err)?;
        let lib = include_bytes!(concat!(env!("OUT_DIR"), "/ojas_per_head_gate.metallib"));
        if lib.is_empty() {
            return Err(metal_err("embedded ojas-metal metallib is empty"));
        }
        rt.add_metallib_bytes(lib).map_err(metal_err)?;
        for name in KERNELS {
            rt.pipeline(name).map_err(metal_err)?;
        }
        let slab = rt.alloc_buffer(SLAB_SLOTS * ST_BYTES).map_err(metal_err)?;
        slab.try_contents_u32().map_err(metal_err)?.fill(0);
        let spare = rt.alloc_buffer(NORM_SLOTS * 4).map_err(metal_err)?;
        let ws = rt.memory_info().recommended_working_set;
        let book = Book {
            used: 0,
            pending: Vec::new(),
            cmd: 0,
            fault: None,
            dirty: false,
            since_overlap: 0,
            alloc_since_wait: 0,
            freed_since_wait: 0,
            inline_since_wait: 0,
            inline_uploads: true,
            clean: HashMap::new(),
            overlap: OVERLAP_DISPATCHES,
            mem_cap: MEM_CAP_CEILING.min(budget_cap / 4),
            ws_limit: ws / WS_DEN * WS_NUM,
            #[cfg(test)]
            fail_allocs: (0, 0),
            #[cfg(test)]
            fail_next_read: false,
        };
        Ok(Self {
            rt,
            bufs: HashMap::new(),
            next_id: 1,
            poisoned: false,
            waits,
            slab,
            spare,
            book: RefCell::new(book),
        })
    }

    fn serve(&mut self, rx: mpsc::Receiver<Msg>) {
        for msg in rx {
            if let Cmd::Free { id } = msg.cmd {
                self.free(id);
                continue;
            }
            if let Cmd::Memory { reset_peak } = msg.cmd {
                // A read of device properties: not a command, so it neither
                // counts toward nor fires the commit triggers, and a
                // poisoned backend still reports what it holds.
                let out = match catch_unwind(AssertUnwindSafe(|| {
                    let m = self.memory();
                    if reset_peak {
                        self.rt.reset_peak_allocated_bytes();
                    }
                    m
                })) {
                    Ok(m) => Ok(Reply::Memory(m)),
                    Err(_) => {
                        self.poisoned = true;
                        Err(metal_err(
                            "the Metal memory query panicked; this backend is poisoned",
                        ))
                    }
                };
                if let Some(reply) = msg.reply {
                    let _ = reply.send(out);
                }
                continue;
            }
            let out = if self.poisoned {
                Err(OjasError::Poisoned)
            } else if self.rt.is_poisoned() {
                Err(device_lost(
                    "the tessl runtime is poisoned by an earlier failed or timed-out command buffer",
                ))
            } else {
                match catch_unwind(AssertUnwindSafe(|| self.command(msg.cmd))) {
                    Ok(out) => out.map_err(|err| self.lost_if_poisoned(err)),
                    Err(_) => {
                        self.poisoned = true;
                        Err(metal_err(
                            "the Metal device thread panicked; this backend is poisoned",
                        ))
                    }
                }
            };
            if let Some(reply) = msg.reply {
                let _ = reply.send(out);
            }
        }
    }

    /// A command that failed on a runtime tessl has poisoned (a command
    /// buffer that failed or timed out) failed because the device is gone:
    /// its tessl error becomes [`OjasError::DeviceLost`]. Typed errors (a
    /// fault, a capacity refusal) keep their kind.
    fn lost_if_poisoned(&self, err: OjasError) -> OjasError {
        match err {
            OjasError::Backend { detail, .. } if self.rt.is_poisoned() => {
                device_lost(format!("tessl runtime poisoned: {detail}"))
            }
            other => other,
        }
    }

    fn memory(&self) -> crate::MetalMemory {
        let info = self.rt.memory_info();
        crate::MetalMemory {
            recommended_working_set: info.recommended_working_set,
            allocated: self.rt.current_allocated_bytes(),
            peak_allocated: self.rt.peak_allocated_bytes(),
            has_unified_memory: info.has_unified_memory,
            pool_cache_cap: info.pool_cache_cap as u64,
        }
    }

    /// Run one command, then commit if a trigger says so. A command that
    /// fails drops the status slots it took (its outputs never get ids),
    /// except sealed ones. A trigger that fails fails the command, and its
    /// outputs are released.
    fn command(&mut self, cmd: Cmd) -> Res<Reply> {
        self.recycle_freed()?;
        let id = {
            let mut book = self.book.borrow_mut();
            book.cmd += 1;
            book.cmd
        };
        let out = self.run(cmd);
        self.note_dispatches();
        if out.is_err() {
            self.book
                .borrow_mut()
                .pending
                .retain(|e| e.cmd != id || e.sealed);
            return out;
        }
        if let Err(err) = self.after_command() {
            if let Ok(Reply::Bufs(bufs)) = &out {
                for b in bufs {
                    self.bufs.remove(&b.id);
                }
            }
            return Err(err);
        }
        out
    }

    /// Fold tessl's dispatch count into the book.
    fn note_dispatches(&self) {
        let n = self.rt.take_dispatch_count();
        if n > 0 {
            self.waits.add_dispatches(n);
            let mut book = self.book.borrow_mut();
            book.dirty = true;
            book.since_overlap += n;
        }
    }

    /// Drop the buffer `id` names. Its bytes count toward [`Wait::Freed`]
    /// until the next waited commit.
    fn free(&mut self, id: u64) {
        self.book.borrow_mut().clean.remove(&id);
        if let Some(buf) = self.bufs.remove(&id) {
            self.book.borrow_mut().freed_since_wait += buf.nbytes() as u64;
        }
    }

    /// The trigger checked before each command: buffers freed since the
    /// last waited commit past the memory cap are recycled by one, so a
    /// command never allocates while more than the cap of dropped buffers
    /// still sits on the device. It waits even with nothing recorded:
    /// what was freed waits for a commit too.
    fn recycle_freed(&self) -> Res<()> {
        let (freed, cap) = {
            let b = self.book.borrow();
            (b.freed_since_wait, b.mem_cap)
        };
        if freed <= cap {
            return Ok(());
        }
        self.note_dispatches();
        self.book.borrow_mut().dirty = true;
        self.settle(Scan::All, Wait::Freed)
    }

    /// The commit triggers checked between commands: memory (bytes since
    /// the last wait, then the device's working set), then overlap.
    fn after_command(&self) -> Res<()> {
        let (dirty, alloc, cap, limit, since, overlap) = {
            let b = self.book.borrow();
            (
                b.dirty,
                b.alloc_since_wait,
                b.mem_cap,
                b.ws_limit,
                b.since_overlap,
                b.overlap,
            )
        };
        if !dirty {
            return Ok(());
        }
        if alloc > cap {
            return self.settle(Scan::All, Wait::MemCap);
        }
        if alloc > WS_GROWTH && self.rt.current_allocated_bytes() > limit {
            return self.settle(Scan::All, Wait::WorkingSet);
        }
        if since >= overlap {
            self.rt.commit(false).map_err(metal_err)?;
            self.book.borrow_mut().since_overlap = 0;
        }
        Ok(())
    }

    /// Wait for everything recorded, then read the status slots `scan`
    /// allows, in recording order. A fault found is held for the next sync
    /// point; a fault already held stays first. On a failed wait or mapping
    /// nothing is scanned and the pending list is kept. A wait is counted
    /// under `why`; with nothing recorded nothing waits and nothing counts.
    fn settle(&self, scan: Scan, why: Wait) -> Res<()> {
        self.note_dispatches();
        if self.book.borrow().dirty {
            self.waits.count(why);
            self.rt.synchronize().map_err(metal_err)?;
            let mut book = self.book.borrow_mut();
            book.dirty = false;
            book.since_overlap = 0;
            book.alloc_since_wait = 0;
            book.freed_since_wait = 0;
            book.inline_since_wait = 0;
        }
        self.scan(scan)
    }

    fn scan(&self, scan: Scan) -> Res<()> {
        let mut book = self.book.borrow_mut();
        if book.used == 0 {
            return Ok(());
        }
        let current = book.cmd;
        let mut words = self.slab.try_contents_u32().map_err(metal_err)?;
        let mut kept = Vec::new();
        let mut fault = book.fault;
        let mut withdrawn = Vec::new();
        for e in book.pending.drain(..) {
            if scan == Scan::BeforeCurrent && e.cmd == current {
                kept.push(e);
                continue;
            }
            let w = &words[e.slot * ST_WORDS..(e.slot + 1) * ST_WORDS];
            if w[ST_IN as usize] != 0 || w[ST_OUT as usize] != 0 {
                withdrawn.extend(e.clean_of);
                if fault.is_none() {
                    fault = Some(e.op);
                }
            }
        }
        // Every slot not still pending is cleared, including those of
        // commands that failed after taking one.
        let live: HashSet<usize> = kept.iter().map(|e| e.slot).collect();
        for slot in (0..book.used).filter(|s| !live.contains(s)) {
            words[slot * ST_WORDS..(slot + 1) * ST_WORDS].fill(0);
        }
        drop(words);
        book.fault = fault;
        if kept.is_empty() {
            book.used = 0;
        }
        book.pending = kept;
        // A frozen buffer whose check faulted is not known finite: its mark,
        // and its widened copies', go, so every later op scans it again.
        if !withdrawn.is_empty() {
            book.clean.retain(|id, c| {
                !withdrawn.contains(id) && !c.from.is_some_and(|from| withdrawn.contains(&from))
            });
        }
        Ok(())
    }

    /// Hand the pending fault to the caller, once.
    fn report(&self) -> Res<()> {
        match self.book.borrow_mut().fault.take() {
            Some(op) => Err(OjasError::NonFinite { op }),
            None => Ok(()),
        }
    }

    // ------------------------------------------------------------ buffers --

    fn view(&self, a: Arg) -> Res<V> {
        let buf = self
            .bufs
            .get(&a.id)
            .ok_or_else(|| metal_err(format!("unknown device buffer {}", a.id)))?;
        let end =
            a.n.checked_mul(4)
                .and_then(|b| b.checked_add(a.off))
                .ok_or_else(|| metal_err("device window overflows"))?;
        if !a.off.is_multiple_of(4) || end > buf.nbytes() {
            return Err(metal_err(format!(
                "window [{}, {end}) does not fit device buffer {} of {} bytes",
                a.off,
                a.id,
                buf.nbytes()
            )));
        }
        Ok(V {
            buf: buf.clone(),
            off: a.off,
            n: a.n,
        })
    }

    /// A bf16 window of `a` (`a.n` 2-byte elements from byte `a.off`).
    fn half_view(&self, a: Arg) -> Res<Half> {
        let buf = self
            .bufs
            .get(&a.id)
            .cloned()
            .ok_or_else(|| metal_err(format!("unknown device buffer {}", a.id)))?;
        let end =
            a.n.checked_mul(2)
                .and_then(|b| b.checked_add(a.off))
                .ok_or_else(|| metal_err("device window overflows"))?;
        if !a.off.is_multiple_of(2) || end > buf.nbytes() {
            return Err(metal_err(format!(
                "bf16 window [{}, {end}) does not fit device buffer {} of {} bytes",
                a.off,
                a.id,
                buf.nbytes()
            )));
        }
        // Bound at the 4-byte boundary below the window, with the odd
        // element, if any, as the kernels' first index.
        let base = a.off & !3;
        Ok(Half {
            buf,
            base,
            first: (a.off - base) / 2,
            n: a.n,
        })
    }

    /// A device allocation. See [`Self::retrying`].
    fn alloc(&self, bytes: usize) -> Res<GpuBuffer> {
        let buf = self.retrying(bytes, || self.try_alloc(bytes))?;
        self.book.borrow_mut().alloc_since_wait += bytes as u64;
        Ok(buf)
    }

    /// Run a device allocation of `bytes`. Buffers dropped since the last
    /// waited commit return to tessl's pool only at the next one, so one
    /// the device refused ([`AllocFailure::Exhausted`]) is retried after a
    /// waited commit that leaves the current command's status slots alone,
    /// and once more after the pool's cached buffers are released (a cached
    /// buffer of another size cannot serve the request, but its memory
    /// can). A request past the device's limit ([`AllocFailure::Limit`]) is
    /// refused at once. Either failing past that is `CapacityExceeded`
    /// against the device's working set, with what the device holds now as
    /// `live`; a failure that is not an allocation's stays `Backend`.
    fn retrying<T>(&self, bytes: usize, mut attempt: impl FnMut() -> Result<T, String>) -> Res<T> {
        // An attempt that makes several buffers and then fails drops the ones
        // it made. They return to the pool only at the next waited commit, so
        // they count as freed bytes toward the wait trigger, like any other
        // dropped buffer. The device-allocated figure is process-wide, so
        // another thread's allocations can only inflate the count, which
        // brings a wait forward and never delays one.
        let mut run = || {
            let before = self.rt.current_allocated_bytes();
            let r = attempt();
            if r.is_err() {
                let left = self.rt.current_allocated_bytes().saturating_sub(before);
                self.book.borrow_mut().freed_since_wait += left;
            }
            r
        };
        let mut failed = match run() {
            Ok(v) => return Ok(v),
            Err(e) => e,
        };
        for step in 0..2 {
            match classify_alloc(&failed) {
                AllocFailure::Other => return Err(metal_err(failed)),
                AllocFailure::Limit => break,
                AllocFailure::Exhausted => {}
            }
            if step == 0 {
                self.recycle()?;
            } else {
                self.drop_pool();
            }
            failed = match run() {
                Ok(v) => return Ok(v),
                Err(e) => e,
            };
        }
        if classify_alloc(&failed) == AllocFailure::Other {
            return Err(metal_err(failed));
        }
        Err(OjasError::CapacityExceeded {
            requested: bytes as u64,
            cap: self.rt.memory_info().recommended_working_set,
            live: self.rt.current_allocated_bytes(),
        })
    }

    /// gdn backward's scratch, retried like any device allocation.
    fn gdn_workspace(&self, dims: GdnTrainDims) -> Res<GdnTrainWorkspace> {
        let bytes = GdnTrainWorkspace::bytes_for(dims);
        let ws = self.retrying(bytes, || GdnTrainWorkspace::new(&self.rt, dims))?;
        self.book.borrow_mut().alloc_since_wait += bytes as u64;
        Ok(ws)
    }

    fn try_alloc(&self, bytes: usize) -> Result<GpuBuffer, String> {
        #[cfg(test)]
        self.injected_failure(bytes)?;
        self.rt.alloc_buffer(bytes)
    }

    /// [`Cmd::Tune`]'s `fail_allocs`, consulted by every device allocation
    /// attempt ([`Self::try_alloc`], [`Self::fresh_from`]).
    #[cfg(test)]
    fn injected_failure(&self, bytes: usize) -> Result<(), String> {
        let mut book = self.book.borrow_mut();
        let (skip, fail) = book.fail_allocs;
        if skip > 0 {
            book.fail_allocs.0 -= 1;
        } else if fail > 0 {
            book.fail_allocs.1 -= 1;
            return Err(format!("newBuffer({bytes}) failed"));
        }
        Ok(())
    }

    /// A waited commit for its recycling. It runs even with nothing
    /// recorded, because buffers freed since the last one wait for it too.
    fn recycle(&self) -> Res<()> {
        self.note_dispatches();
        self.book.borrow_mut().dirty = true;
        self.settle(Scan::BeforeCurrent, Wait::Recycle)
    }

    /// Release every buffer tessl's pool has cached: lowering the cap to 0
    /// drops them (tessl `set_max_cache_bytes` trims to the new cap), and
    /// the cap is then restored, so later frees cache as before.
    fn drop_pool(&self) {
        let cap = self.rt.memory_info().pool_cache_cap;
        self.rt.set_pool_cache_cap_bytes(0);
        self.rt.set_pool_cache_cap_bytes(cap);
    }

    /// [`Cmd::TrimPool`]: recycle what was freed since the last waited
    /// commit, then release the whole cache.
    fn trim_pool(&self) -> Res<Reply> {
        self.recycle()?;
        self.drop_pool();
        Ok(Reply::Done)
    }

    /// `n` four-byte elements, offset 0.
    fn fresh(&self, n: usize) -> Res<V> {
        let bytes = n
            .checked_mul(4)
            .ok_or_else(|| metal_err("allocation size overflows"))?;
        Ok(V {
            buf: self.alloc(bytes.max(4))?,
            off: 0,
            n,
        })
    }

    /// A new buffer holding `data`, written on the host without a wait: no
    /// recorded command can be using a buffer fresh from the pool (tessl
    /// `alloc_buffer_from_u32`). For host-built index data and `U32`
    /// uploads behind recorded work.
    fn fresh_from(&self, data: &[u32]) -> Res<V> {
        let bytes = data.len().max(1) * 4;
        let buf = self.retrying(bytes, || {
            #[cfg(test)]
            self.injected_failure(bytes)?;
            self.rt.alloc_buffer_from_u32(data)
        })?;
        self.book.borrow_mut().alloc_since_wait += bytes as u64;
        Ok(V {
            buf,
            off: 0,
            n: data.len(),
        })
    }

    /// Give `outs` ids. Called only after the op's checks passed.
    fn keep(&mut self, outs: Vec<V>) -> Reply {
        let mut ids = Vec::with_capacity(outs.len());
        for v in outs {
            let id = self.next_id;
            self.next_id += 1;
            ids.push(NewBuf { id, bytes: v.n * 4 });
            self.bufs.insert(id, v.buf);
        }
        Reply::Bufs(ids)
    }

    fn tt(&self, v: &V, shape: &[usize]) -> Res<TT> {
        TT::from_buffer(&self.rt, v.buf.clone(), shape, DType::F32, v.off).map_err(metal_err)
    }

    /// A GEMM operand: tessl needs a 16-byte aligned offset, so a view that is
    /// not aligned is copied first.
    /// `v` as an `[r, c]` matrix for tessl's GEMMs, copied first when its
    /// offset is off their alignment ([`ReadAs::Matrix`]), into memory the
    /// backend charged as scratch.
    fn mat(&self, v: &V, r: usize, c: usize) -> Res<TT> {
        if !copied_at(v.off, ReadAs::Matrix) {
            return self.tt(v, &[r, c]);
        }
        let dst = self.fresh(v.n)?;
        self.copy(v, &dst)?;
        self.tt(&dst, &[r, c])
    }

    fn copy(&self, src: &V, dst: &V) -> Res<()> {
        gpu_copy(&self.tt(src, &[src.n])?, &self.tt(dst, &[dst.n])?).map_err(metal_err)
    }

    // ----------------------------------------------------------- dispatch --

    fn k1(&self, name: &str, n: usize, f: impl FnOnce(&mut Binder<'_>)) -> Res<()> {
        let p = self.rt.pipeline(name).map_err(metal_err)?;
        dispatch::dispatch_1d(&self.rt, &p, n, f).map_err(metal_err)
    }

    fn k2(&self, name: &str, nx: usize, ny: usize, f: impl FnOnce(&mut Binder<'_>)) -> Res<()> {
        let p = self.rt.pipeline(name).map_err(metal_err)?;
        dispatch::dispatch_2d(&self.rt, &p, nx, ny, f).map_err(metal_err)
    }

    fn ktg(
        &self,
        name: &str,
        gx: usize,
        gy: usize,
        threads: usize,
        f: impl FnOnce(&mut Binder<'_>),
    ) -> Res<()> {
        let p = self.rt.pipeline(name).map_err(metal_err)?;
        dispatch::dispatch_2d_tg(&self.rt, &p, gx, gy, threads, f).map_err(metal_err)
    }

    /// The next status slot, recorded for `op` in recording order. Its
    /// words are zero (the host clears slots after each scan). A full slab
    /// forces a waited commit first; at this point every slot already taken
    /// belongs to fully encoded work.
    fn status(&self, op: &'static str) -> Res<St> {
        if self.book.borrow().used == SLAB_SLOTS {
            self.note_dispatches();
            self.settle(Scan::All, Wait::SlabFull)?;
        }
        let mut book = self.book.borrow_mut();
        if book.used == SLAB_SLOTS {
            return Err(metal_err("status slab still full after a waited commit"));
        }
        let slot = book.used;
        book.used += 1;
        let cmd = book.cmd;
        book.pending.push(Entry {
            slot,
            op,
            cmd,
            sealed: false,
            clean_of: None,
        });
        Ok(St {
            buf: self.slab.clone(),
            off: slot * ST_BYTES,
        })
    }

    /// Keep the current command's slots taken so far even if it fails later.
    fn seal(&self) {
        let mut book = self.book.borrow_mut();
        let cmd = book.cmd;
        for e in book.pending.iter_mut().filter(|e| e.cmd == cmd) {
            e.sealed = true;
        }
    }

    /// Flag word `word` if any element of `v` is NaN or infinite.
    fn check(&self, st: &St, v: &V, word: u32) -> Res<()> {
        let n = u32_of(v.n)?;
        self.k1("ojas_check_finite", v.n.div_ceil(CHECK_PER_THREAD), |b| {
            bind(b, v, 0);
            bind_st(b, st, 1);
            set_u32(b, n, 2);
            set_u32(b, word, 3);
        })
    }

    /// [`Self::check`] of a bf16 window, on its bits.
    fn check_half(&self, st: &St, h: &Half, word: u32) -> Res<()> {
        let (n, first) = (u32_of(h.n)?, u32_of(h.first)?);
        self.k1(
            "ojas_check_finite_bf16",
            h.n.div_ceil(CHECK_PER_THREAD),
            |b| {
                set_gpu_buf_offset(b, &h.buf, h.base, 0);
                bind_st(b, st, 1);
                set_u32(b, n, 2);
                set_u32(b, word, 3);
                set_u32(b, first, 4);
            },
        )
    }

    /// Whether window `a` of `elem`-byte elements lies inside a range of its
    /// buffer already checked finite and kept unwritten ([`Self::freeze`]).
    fn known_clean(&self, a: Arg, elem: usize) -> bool {
        let Some(end) = a.n.checked_mul(elem).and_then(|b| b.checked_add(a.off)) else {
            return false;
        };
        self.book.borrow().clean.get(&a.id).is_some_and(|c| {
            c.windows
                .iter()
                .any(|&(start, stop)| start <= a.off && end <= stop)
        })
    }

    /// [`Self::check`] of `v`, the window `a` names, unless a frozen mark
    /// already answers for it. An operand that is not a frozen parameter is
    /// scanned as ever.
    fn check_operand(&self, st: &St, a: Arg, v: &V, word: u32) -> Res<()> {
        if self.known_clean(a, 4) {
            return Ok(());
        }
        self.check(st, v, word)
    }

    /// [`Self::check_operand`] of a bf16 window.
    fn check_half_operand(&self, st: &St, a: Arg, h: &Half, word: u32) -> Res<()> {
        if self.known_clean(a, 2) {
            return Ok(());
        }
        self.check_half(st, h, word)
    }

    /// Refuse a write into a frozen parameter: the mark says the buffer is
    /// finite because it never changes, so no op may change it.
    ///
    /// A widened copy is the exception: it is a new buffer the caller owns
    /// (`to_f32` of a frozen bf16 weight is public), so a write to it is
    /// legitimate and withdraws its mark instead, and every later op scans it
    /// again. The withdrawal precedes the write in the command order, so the
    /// mark is never live over changed bytes.
    fn writable(&self, op: &'static str, a: Arg) -> Res<()> {
        let mut book = self.book.borrow_mut();
        let derived = match book.clean.get(&a.id) {
            None => return Ok(()),
            Some(c) => c.from.is_some(),
        };
        if derived {
            book.clean.remove(&a.id);
            return Ok(());
        }
        Err(metal_err(format!(
            "{op}: device buffer {} is a frozen, checked parameter and is not written",
            a.id
        )))
    }

    /// Whether window `a` of `elem`-byte elements is already inside a window
    /// a [`Self::freeze`] scanned (not a widened copy's inherited one).
    fn frozen_root(&self, a: Arg, elem: usize) -> bool {
        let Some(end) = a.n.checked_mul(elem).and_then(|b| b.checked_add(a.off)) else {
            return false;
        };
        self.book.borrow().clean.get(&a.id).is_some_and(|c| {
            c.from.is_none()
                && c.windows
                    .iter()
                    .any(|&(start, stop)| start <= a.off && end <= stop)
        })
    }

    /// Check window `x` (bf16 when `half`) for NaN and infinity once, and
    /// mark its buffer so later ops skip that scan ([`Self::check_operand`])
    /// and refuse to write it ([`Self::writable`]). The check is recorded
    /// like any other: a fault is reported at the next sync, and a faulted
    /// check withdraws the mark in [`Self::scan`], so a buffer that is not
    /// finite is never skipped.
    fn freeze(&mut self, x: Arg, half: bool) -> Res<Reply> {
        const OP: &str = "freeze_checked";
        // Freezing a window again scans nothing and records nothing: the
        // bytes cannot have changed, and a repeat must not grow the list
        // every op's lookup walks.
        if self.frozen_root(x, if half { 2 } else { 4 }) {
            return Ok(Reply::Done);
        }
        // The views validate the window (its end cannot overflow below).
        if half {
            let h = self.half_view(x)?;
            let st = self.status(OP)?;
            self.check_half(&st, &h, ST_IN)?;
        } else {
            let v = self.view(x)?;
            let st = self.status(OP)?;
            self.check(&st, &v, ST_IN)?;
        }
        let end = x.off + x.n * if half { 2 } else { 4 };
        let mut book = self.book.borrow_mut();
        if let Some(e) = book.pending.last_mut() {
            e.clean_of = Some(x.id);
        }
        let mark = book.clean.entry(x.id).or_insert_with(|| Clean {
            windows: Vec::new(),
            from: None,
        });
        // A widened copy frozen on purpose is a frozen parameter from now on.
        mark.from = None;
        mark.windows.push((x.off, end));
        Ok(Reply::Done)
    }

    /// Elements `start..start + dst.n` of `h` widened exactly into `dst`.
    fn widen_into(&self, h: &Half, start: usize, dst: &V) -> Res<()> {
        if start.checked_add(dst.n).is_none_or(|end| end > h.n) {
            return Err(metal_err(format!(
                "bf16 rows {start}..+{} are outside a window of {}",
                dst.n, h.n
            )));
        }
        let (n, first) = (u32_of(dst.n)?, u32_of(h.first + start)?);
        self.k1("ojas_bf16_to_f32", dst.n, |b| {
            set_gpu_buf_offset(b, &h.buf, h.base, 0);
            bind(b, dst, 1);
            set_u32(b, n, 2);
            set_u32(b, first, 3);
        })
    }

    fn copy_if_clean(&self, st: &St, src: &V, dst: &V) -> Res<()> {
        let n = u32_of(dst.n)?;
        self.k1("ojas_copy_if_clean", dst.n, |b| {
            bind(b, src, 0);
            bind(b, dst, 1);
            bind_st(b, st, 2);
            set_u32(b, n, 3);
        })
    }

    /// `out = alpha x + beta y` through one of the three Muon combination
    /// kernels: `ojas_axpby` (two products and a sum), `ojas_axpby_fma`
    /// (`fma(beta, y, alpha x)`, torch's fused `add(.., alpha=beta)`) or
    /// `ojas_axpby_bf16` (every op rounded to bf16).
    #[allow(clippy::too_many_arguments)]
    fn axpby(&self, kernel: &str, x: &V, y: &V, out: &V, alpha: f32, beta: f32) -> Res<()> {
        let n = u32_of(out.n)?;
        self.k1(kernel, out.n, |b| {
            bind(b, x, 0);
            bind(b, y, 1);
            bind(b, out, 2);
            set_u32(b, n, 3);
            set_f32(b, alpha, 4);
            set_f32(b, beta, 5);
        })
    }

    /// `dst = r(src)`, `r` rounding to bf16; `src` and `dst` may be one view.
    fn round_bf16_into(&self, src: &V, dst: &V) -> Res<()> {
        let n = u32_of(dst.n)?;
        self.k1("ojas_round_bf16", dst.n, |b| {
            bind(b, src, 0);
            bind(b, dst, 1);
            set_u32(b, n, 2);
        })
    }

    /// Reduce `src` into `out[slot]`. mode 0 sum, 1 max |x|, 2 sum of
    /// (x / scale)^2 with the scale read at `scale` (buffer, byte offset).
    fn reduce(
        &self,
        src: &V,
        mode: u32,
        scale: Option<(&GpuBuffer, usize)>,
        out: &GpuBuffer,
        slot: u32,
    ) -> Res<()> {
        let groups = reduce_groups(src.n);
        let chunk = src.n.div_ceil(groups);
        let part = self.fresh(groups)?;
        let n = u32_of(src.n)?;
        let chunk_u = u32_of(chunk)?;
        let groups_u = u32_of(groups)?;
        self.ktg("ojas_reduce_partial", groups, 1, 256, |b| {
            bind(b, src, 0);
            bind(b, &part, 1);
            set_u32(b, n, 2);
            set_u32(b, chunk_u, 3);
            set_u32(b, mode, 4);
            match scale {
                Some((buf, off)) => set_gpu_buf_offset(b, buf, off, 5),
                None => set_gpu_buf_offset(b, out, 0, 5),
            }
        })?;
        self.ktg("ojas_reduce_final", 1, 1, 256, |b| {
            bind(b, &part, 0);
            set_gpu_buf_offset(b, out, 0, 1);
            set_u32(b, groups_u, 2);
            set_u32(b, mode, 3);
            set_u32(b, slot, 4);
        })
    }

    fn run(&mut self, cmd: Cmd) -> Res<Reply> {
        match cmd {
            Cmd::Free { id } => {
                self.free(id);
                Ok(Reply::Done)
            }
            // `serve` answers it first; kept here so the match stays total.
            Cmd::Memory { .. } => Ok(Reply::Memory(self.memory())),
            Cmd::TrimPool => self.trim_pool(),
            Cmd::InlineUploads { on } => {
                self.book.borrow_mut().inline_uploads = on;
                Ok(Reply::Done)
            }
            Cmd::Upload { src } => self.upload(&src),
            #[cfg(test)]
            Cmd::InjectPanic => {
                let st = self.status("inject_panic")?;
                let open_work = self.fresh(1)?;
                self.check(&st, &open_work, ST_IN)?;
                panic!("injected device-thread panic");
            }
            #[cfg(test)]
            Cmd::FailNextRead => {
                self.book.borrow_mut().fail_next_read = true;
                Ok(Reply::Done)
            }
            #[cfg(test)]
            Cmd::SetPoolCap { bytes } => {
                self.rt.set_pool_cache_cap_bytes(bytes);
                Ok(Reply::Done)
            }
            #[cfg(test)]
            Cmd::PoisonRuntime => {
                self.rt.poison_as_shared_event_timeout_for_test();
                Ok(Reply::Done)
            }
            #[cfg(test)]
            Cmd::Tune {
                overlap,
                mem_cap,
                ws_limit,
                fail_allocs,
            } => {
                let mut book = self.book.borrow_mut();
                book.overlap = overlap.unwrap_or(book.overlap);
                book.mem_cap = mem_cap.unwrap_or(book.mem_cap);
                book.ws_limit = ws_limit.unwrap_or(book.ws_limit);
                book.fail_allocs = fail_allocs;
                Ok(Reply::Done)
            }
            Cmd::Sync => {
                self.settle(Scan::All, Wait::Sync)?;
                self.report()?;
                Ok(Reply::Done)
            }
            Cmd::Read { id, off, len } => self.read(id, off, len),
            Cmd::Permute {
                x,
                rank,
                oshape,
                istride,
            } => self.permute(x, rank, &oshape, &istride),
            Cmd::LayerNorm {
                x,
                w,
                b,
                rows,
                dim,
                eps,
            } => self.layer_norm([x, w, b], [rows, dim], eps),
            Cmd::GeluErf { x } => self.gelu_erf(x),
            Cmd::Bcast {
                x,
                y,
                mul,
                rank,
                oshape,
                xstride,
                ystride,
            } => self.bcast([x, y], mul, rank, &oshape, &xstride, &ystride),
            Cmd::ScaleFwd { x, scale } => self.scale_fwd(x, scale),
            Cmd::Softmax { x, rows, dim } => self.softmax(x, rows, dim),
            Cmd::Embed {
                table,
                half,
                ids,
                vocab,
                dim,
            } => self.embed(table, half, ids, vocab, dim),
            Cmd::Argmax { x, rows, cols } => self.argmax(x, rows, cols),
            Cmd::Topk { x, rows, cols, k } => self.topk(x, rows, cols, k),
            Cmd::EmbedBwd {
                table,
                half,
                grad,
                rows,
                dim,
            } => self.embed_bwd(table, half, grad, &rows, dim),
            Cmd::Linear {
                x,
                w,
                rows,
                kin,
                nout,
                lane,
            } => self.linear(x, w, rows, kin, nout, lane),
            Cmd::LinearBwd {
                x,
                w,
                gy,
                rows,
                kin,
                nout,
                gx_lane,
                gw_lane,
            } => self.linear_bwd(x, w, gy, rows, kin, nout, [gx_lane, gw_lane]),
            Cmd::LinearBwdInput {
                x,
                w,
                gy,
                rows,
                kin,
                nout,
                lane,
            } => self.linear_bwd_input(x, w, gy, rows, kin, nout, lane),
            Cmd::Rms { sides, eps } => self.rms_sides(&sides, eps),
            Cmd::Rope {
                op,
                x,
                cos,
                sin,
                rows,
                dim,
                rotary,
                mode,
                backward,
            } => self.rope(op, [x, cos, sin], [rows, dim, rotary], mode, backward),
            Cmd::Sdpa { q, k, v, geom } => self.sdpa(q, k, v, geom),
            Cmd::Attn {
                q,
                k,
                v,
                bias,
                lens,
                geom,
            } => self.attn_gen([q, k, v], bias, lens, geom),
            Cmd::SdpaBwd { args, geom } => self.sdpa_bwd(args, geom),
            Cmd::Gate {
                x,
                w,
                b,
                attn,
                rows,
                din,
                heads,
                dh,
                save,
            } => {
                let mode = if save {
                    GateMode::ForwardSaving
                } else {
                    GateMode::Forward
                };
                self.gate(x, w, b, attn, mode, [rows, din, heads, dh])
            }
            Cmd::GateBwd {
                x,
                w,
                b,
                attn,
                gy,
                scales,
                rows,
                din,
                heads,
                dh,
            } => {
                let mode = match scales {
                    Some(s) => GateMode::BackwardSaved(gy, s),
                    None => GateMode::Backward(gy),
                };
                self.gate(x, w, b, attn, mode, [rows, din, heads, dh])
            }
            Cmd::Vres { v, v0, lam } => self.vres(v, v0, lam, None),
            Cmd::VresBwd { v, v0, lam, gy } => self.vres(v, v0, lam, Some(gy)),
            Cmd::Gdn { x } => self.gdn(x),
            Cmd::GdnBwd {
                x,
                ckpt,
                d_o,
                d_fin,
            } => self.gdn_bwd(x, ckpt, d_o, d_fin),
            Cmd::Conv1d { x, w, geom } => self.conv1d(x, w, geom),
            Cmd::Conv1dBwd {
                x,
                w,
                gy,
                geom,
                part,
            } => self.conv1d_bwd([x, w, gy], geom, part),
            Cmd::GatedRms {
                x,
                z,
                w,
                rows,
                dim,
                eps,
            } => self.gated_rms([x, z, w], [rows, dim], eps),
            Cmd::GatedRmsBwd {
                x,
                z,
                w,
                gy,
                rows,
                dim,
                eps,
                part,
            } => self.gated_rms_bwd([x, z, w, gy], [rows, dim], eps, part),
            Cmd::RoundBf16 { x } => self.round_bf16(x),
            Cmd::ToF32 { x } => self.widen_bf16(x),
            Cmd::Freeze { x, half } => self.freeze(x, half),
            Cmd::ToBf16 { x } => self.narrow_bf16(x),
            Cmd::Silu { x } => self.silu(x, None),
            Cmd::SiluBwd { x, gy } => self.silu(x, Some(gy)),
            Cmd::Sigmoid { x } => self.sigmoid(x, None),
            Cmd::SigmoidBwd { x, gy } => self.sigmoid(x, Some(gy)),
            Cmd::GdnDecay {
                a,
                a_log,
                dt_bias,
                heads,
            } => self.gdn_decay([a, a_log, dt_bias], heads),
            Cmd::GdnDecayBwd {
                a,
                a_log,
                dt_bias,
                gy,
                heads,
            } => self.gdn_decay_bwd([a, a_log, dt_bias, gy], heads),
            Cmd::Mul { a, b } => self.mul(a, b, None),
            Cmd::MulBwd { a, b, gy } => self.mul(a, b, Some(gy)),
            Cmd::Add { x, y } => self.add(x, y),
            Cmd::AddBwd { x, y, gy } => self.add_bwd(x, y, gy),
            Cmd::Ce {
                logits,
                targets,
                rows,
                vocab,
                ignore,
                valid,
                grad,
            } => self.ce(logits, targets, rows, vocab, ignore, valid, grad),
            Cmd::ClipNorm { grads } => self.clip_norm(&grads),
            Cmd::Scale { op, grads, scale } => self.scale(op, &grads, scale),
            Cmd::Accumulate { acc, g, in_place } => self.accumulate(acc, g, in_place),
            Cmd::LinearCe {
                x,
                w,
                t,
                geom,
                ignore,
                valid,
            } => self.linear_ce(x, w, t, geom, ignore, valid),
            Cmd::CachedAttn {
                q,
                k,
                v,
                batch,
                tq,
                heads,
                kv_heads,
                d,
                cap,
                kv_len,
                window,
            } => self.cached_attn(
                q,
                k,
                v,
                [batch, tq, heads, kv_heads, d, cap, kv_len, window],
            ),
            Cmd::KvWrite {
                cache,
                src,
                batch,
                tn,
                cap,
                row,
                at,
            } => self.kv_write(cache, src, batch, tn, cap, row, at),
            Cmd::AdamW {
                p,
                g,
                m,
                v,
                scalars,
            } => self.adamw(p, g, m, v, &scalars),
            Cmd::Muon {
                p,
                g,
                m,
                rows,
                cols,
                momentum,
                nesterov,
                decay,
                alpha,
                bf16,
            } => self.muon(p, g, m, rows, cols, momentum, nesterov, decay, alpha, bf16),
        }
    }

    /// tessl maps a buffer for the host only after a waited commit. With
    /// nothing recorded that costs no wait, so the bytes are mapped in. With
    /// work recorded, a small upload rides in the command itself: its words
    /// go into tessl's constant arena and `ojas_upload_words` copies them,
    /// in order with the recorded work, so nothing waits. A `U32` upload
    /// (token ids, targets) behind recorded work, at any size, is written
    /// on the host into a buffer fresh from the pool ([`Self::fresh_from`]),
    /// which no recorded command can be using: no upload wait, no room in
    /// the constant arena, and no copy but the one into the buffer. Its
    /// bytes count toward the memory-cap trigger like any allocation's, so
    /// a large one can still bring a [`Wait::MemCap`] forward. Any other
    /// upload waits ([`Wait::Upload`]) and maps.
    /// A new buffer holding `src`'s contiguous window. Small uploads behind
    /// recorded work ride inline (encoded into at most
    /// [`INLINE_UPLOAD_BYTES`] of host memory); the rest wait and are
    /// encoded straight into the buffer's mapping, so the host never holds a
    /// second copy of the window.
    fn upload(&mut self, src: &ojas_core::Tensor) -> Res<Reply> {
        let len = src
            .num_elements()?
            .checked_mul(src.dtype().size())
            .ok_or_else(|| metal_err("upload size overflows"))?;
        self.note_dispatches();
        let behind = {
            let book = self.book.borrow();
            book.dirty && book.inline_uploads
        };
        if behind && src.dtype() == ojas_core::DType::U32 {
            let ids = self.fresh_from(src.u32_slice()?)?;
            return Ok(self.adopt(ids.buf, len));
        }
        let buf = self.alloc(len)?;
        self.note_dispatches();
        let inline = {
            let book = self.book.borrow();
            book.dirty && book.inline_uploads && inline_upload_fits(len, book.inline_since_wait)
        };
        if inline {
            let bytes = src.to_ne_bytes()?;
            let n = len / 4;
            let dst = V {
                buf: buf.clone(),
                off: 0,
                n,
            };
            self.k1("ojas_upload_words", n, |b| {
                bind(b, &dst, 0);
                set_u32(b, n as u32, 1);
                b.bind_bytes(&bytes, 2);
            })?;
            self.book.borrow_mut().inline_since_wait += len;
        } else {
            self.settle(Scan::All, Wait::Upload)?;
            let mut map = buf.try_contents_u8().map_err(metal_err)?;
            src.write_ne_bytes(&mut map[..len])?;
        }
        Ok(self.adopt(buf, len))
    }

    /// Give a new upload of `len` bytes its id.
    fn adopt(&mut self, buf: GpuBuffer, len: usize) -> Reply {
        let id = self.next_id;
        self.next_id += 1;
        self.bufs.insert(id, buf);
        Reply::Bufs(vec![NewBuf { id, bytes: len }])
    }

    /// A raw read waits and scans, holding any fault for the next sync point.
    fn read(&mut self, id: u64, off: usize, len: usize) -> Res<Reply> {
        let buf = self
            .bufs
            .get(&id)
            .cloned()
            .ok_or_else(|| metal_err(format!("unknown device buffer {id}")))?;
        let end = off
            .checked_add(len)
            .filter(|end| *end <= buf.nbytes())
            .ok_or_else(|| metal_err("read past the end of a device buffer"))?;
        self.settle(Scan::All, Wait::Read)?;
        #[cfg(test)]
        if std::mem::take(&mut self.book.borrow_mut().fail_next_read) {
            return Err(metal_err("injected read failure"));
        }
        let map = buf.try_contents_u8().map_err(metal_err)?;
        Ok(Reply::Bytes(map[off..end].to_vec()))
    }

    /// A bit copy. A NaN or infinity in the input is refused, as the CPU
    /// reference refuses one; the output holds only copied finite bits, so
    /// it needs no check of its own.
    fn permute(&mut self, x: Arg, rank: u32, oshape: &[u32], istride: &[u32]) -> Res<Reply> {
        const OP: &str = "permute";
        let xv = self.view(x)?;
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let shape_bytes: Vec<u8> = oshape.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let stride_bytes: Vec<u8> = istride.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        self.k1("ojas_permute", xv.n, |b| {
            bind(b, &xv, 0);
            bind(b, &y, 1);
            set_u32(b, n, 2);
            set_u32(b, rank, 3);
            b.bind_bytes(&shape_bytes, 4);
            b.bind_bytes(&stride_bytes, 5);
        })?;
        Ok(self.keep(vec![y]))
    }

    /// The gather; a `half` (bf16) table is scanned on its bits and each
    /// picked value widened in the gather (`ojas_embed_fwd_bf16`).
    fn embed(&mut self, table: Arg, half: bool, ids: Arg, vocab: u32, dim: u32) -> Res<Reply> {
        const OP: &str = "embedding_forward";
        let ids = self.view(ids)?;
        let n = u32_of(ids.n)?;
        let out = self.fresh(ids.n * dim as usize)?;
        let st = self.status(OP)?;
        if half {
            let t = self.half_view(table)?;
            let first = u32_of(t.first)?;
            self.check_half_operand(&st, table, &t, ST_IN)?;
            self.k2("ojas_embed_fwd_bf16", dim as usize, ids.n, |b| {
                set_gpu_buf_offset(b, &t.buf, t.base, 0);
                bind(b, &ids, 1);
                bind(b, &out, 2);
                set_u32(b, n, 3);
                set_u32(b, dim, 4);
                set_u32(b, vocab, 5);
                set_u32(b, first, 6);
            })?;
            self.check(&st, &out, ST_OUT)?;
            return Ok(self.keep(vec![out]));
        }
        let t = self.view(table)?;
        self.check_operand(&st, table, &t, ST_IN)?;
        self.k2("ojas_embed_fwd", dim as usize, ids.n, |b| {
            bind(b, &t, 0);
            bind(b, &ids, 1);
            bind(b, &out, 2);
            set_u32(b, n, 3);
            set_u32(b, dim, 4);
            set_u32(b, vocab, 5);
        })?;
        self.check(&st, &out, ST_OUT)?;
        Ok(self.keep(vec![out]))
    }

    /// Each row's argmax column through `ojas_argmax_rows`, one threadgroup
    /// per row. The kernel flags a non-finite input in word 0 itself, so no
    /// separate finite pass runs.
    fn argmax(&mut self, x: Arg, rows: u32, cols: u32) -> Res<Reply> {
        const OP: &str = "argmax_rows";
        let xv = self.view(x)?;
        let out = self.fresh(rows as usize)?;
        let st = self.status(OP)?;
        self.ktg("ojas_argmax_rows", rows as usize, 1, 256, |b| {
            bind(b, &xv, 0);
            bind(b, &out, 1);
            bind_st(b, &st, 2);
            set_u32(b, rows, 3);
            set_u32(b, cols, 4);
        })?;
        Ok(self.keep(vec![out]))
    }

    /// Each row's `k` leaders through `ojas_topk_rows`, one threadgroup per
    /// row and `k` rounds. The kernel flags a NaN or `+inf` input in word 0
    /// itself (`-inf` is a value), so no separate finite pass runs.
    fn topk(&mut self, x: Arg, rows: u32, cols: u32, k: u32) -> Res<Reply> {
        const OP: &str = "topk_rows";
        let xv = self.view(x)?;
        let n = (rows as usize)
            .checked_mul(k as usize)
            .ok_or_else(|| metal_err(format!("{OP}: [{rows}, {k}] overflows")))?;
        let values = self.fresh(n)?;
        let ids = self.fresh(n)?;
        let st = self.status(OP)?;
        self.ktg("ojas_topk_rows", rows as usize, 1, 256, |b| {
            bind(b, &xv, 0);
            bind(b, &values, 1);
            bind(b, &ids, 2);
            bind_st(b, &st, 3);
            set_u32(b, rows, 4);
            set_u32(b, cols, 5);
            set_u32(b, k, 6);
        })?;
        Ok(self.keep(vec![values, ids]))
    }

    /// Deterministic scatter-add: count rows per id, prefix-sum the counts,
    /// place each token in ascending order within its id, then gather. Each
    /// table row is a left-to-right f32 sum from 0, as on the CPU.
    /// Every vocabulary row's gradient in one gather, from the host's
    /// grouping of the token ids (`rows`, in fresh buffers written without a
    /// wait): row `v` sums its tokens' gradient rows in ascending token
    /// order, as the CPU reference adds them; a row with no token is 0.
    /// The table is only scanned (a `half` one on its bits).
    fn embed_bwd(
        &mut self,
        table: Arg,
        half: bool,
        grad: Arg,
        rows: &EmbedRows,
        dim: u32,
    ) -> Res<Reply> {
        const OP: &str = "embedding_backward";
        let g = self.view(grad)?;
        let v = rows.counts.len();
        let vocab = u32_of(v)?;
        let counts = self.fresh_from(&rows.counts)?;
        let starts = self.fresh_from(&rows.starts)?;
        let pos = self.fresh_from(&rows.pos)?;
        let out = self.fresh(v * dim as usize)?;
        let st = self.status(OP)?;
        if half {
            self.check_half(&st, &self.half_view(table)?, ST_IN)?;
        } else {
            self.check(&st, &self.view(table)?, ST_IN)?;
        }
        self.check(&st, &g, ST_IN)?;
        self.k2("ojas_embed_gather", dim as usize, v, |b| {
            bind(b, &g, 0);
            bind(b, &starts, 1);
            bind(b, &counts, 2);
            bind(b, &pos, 3);
            bind(b, &out, 4);
            set_u32(b, vocab, 5);
            set_u32(b, dim, 6);
        })?;
        self.check(&st, &out, ST_OUT)?;
        Ok(self.keep(vec![out]))
    }

    /// tessl's operands for `lane` on this device: [`Lane::Bf16`] needs
    /// TensorOps, and a device without them takes the exact products. The
    /// backend already picks linear lanes by the same test (`Link::lane`);
    /// this one also covers Muon's bf16 Newton-Schulz.
    fn gemm_lane(&self, lane: Lane) -> GemmOperands {
        match lane {
            Lane::Bf16 if self.rt.has_tensorops() => GemmOperands::Bf16,
            Lane::Bf16 | Lane::ExactF32 => GemmOperands::ExactF32,
        }
    }

    fn linear(
        &mut self,
        x: Arg,
        w: Arg,
        rows: usize,
        kin: usize,
        nout: usize,
        lane: Lane,
    ) -> Res<Reply> {
        const OP: &str = "linear_forward";
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let y = self.fresh(rows * nout)?;
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        self.check_operand(&st, w, &wv, ST_IN)?;
        self.gemm_lane(lane)
            .nt(
                &self.mat(&xv, rows, kin)?,
                &self.mat(&wv, nout, kin)?,
                &self.tt(&y, &[rows, nout])?,
            )
            .map_err(metal_err)?;
        self.check(&st, &y, ST_OUT)?;
        Ok(self.keep(vec![y]))
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_bwd(
        &mut self,
        x: Arg,
        w: Arg,
        gy: Arg,
        rows: usize,
        kin: usize,
        nout: usize,
        [gx_lane, gw_lane]: [Lane; 2],
    ) -> Res<Reply> {
        const OP: &str = "linear_backward";
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let gv = self.view(gy)?;
        let gx = self.fresh(rows * kin)?;
        let gw = self.fresh(nout * kin)?;
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        self.check_operand(&st, w, &wv, ST_IN)?;
        self.check(&st, &gv, ST_IN)?;
        let x_t = self.mat(&xv, rows, kin)?;
        let w_t = self.mat(&wv, nout, kin)?;
        let g_t = self.mat(&gv, rows, nout)?;
        self.gemm_lane(gx_lane)
            .nn(&g_t, &w_t, &self.tt(&gx, &[rows, kin])?)
            .map_err(metal_err)?;
        self.gemm_lane(gw_lane)
            .tn(&g_t, &x_t, &self.tt(&gw, &[nout, kin])?)
            .map_err(metal_err)?;
        self.check(&st, &gx, ST_OUT)?;
        self.check(&st, &gw, ST_OUT)?;
        Ok(self.keep(vec![gx, gw]))
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_bwd_input(
        &mut self,
        x: Arg,
        w: Arg,
        gy: Arg,
        rows: usize,
        kin: usize,
        nout: usize,
        lane: Lane,
    ) -> Res<Reply> {
        const OP: &str = "linear_backward_input";
        let xv = self.view(x)?;
        let wv = self.view(w)?;
        let gv = self.view(gy)?;
        let gx = self.fresh(rows * kin)?;
        let st = self.status(OP)?;
        // `x` is not an operand of the product; it is checked as the full
        // path checks it, so a poisoned activation is refused here too. The
        // weight is the one operand a frozen mark answers for.
        self.check(&st, &xv, ST_IN)?;
        self.check_operand(&st, w, &wv, ST_IN)?;
        self.check(&st, &gv, ST_IN)?;
        self.gemm_lane(lane)
            .nn(
                &self.mat(&gv, rows, nout)?,
                &self.mat(&wv, nout, kin)?,
                &self.tt(&gx, &[rows, kin])?,
            )
            .map_err(metal_err)?;
        self.check(&st, &gx, ST_OUT)?;
        Ok(self.keep(vec![gx]))
    }

    /// RMSNorm forward of one side into `st`.
    fn rms_encode(&self, st: &St, side: &RmsSide, eps: f32) -> Res<V> {
        let xv = self.view(side.x)?;
        let wv = self.view(side.w)?;
        let (rows, dim) = (side.rows, side.dim);
        let y = self.fresh(xv.n)?;
        self.check(st, &xv, ST_IN)?;
        self.check(st, &wv, ST_IN)?;
        self.ktg("ojas_rms_fwd", rms_groups(rows), 1, RMS_THREADS, |b| {
            bind(b, &xv, 0);
            bind(b, &wv, 1);
            bind(b, &y, 2);
            bind_st(b, st, 3);
            set_u32(b, rows, 4);
            set_u32(b, dim, 5);
            set_f32(b, eps, 6);
        })?;
        self.check(st, &y, ST_OUT)?;
        Ok(y)
    }

    /// RMSNorm backward of one side into `st`. Returns
    /// (gx, gw).
    fn rms_bwd_encode(&self, st: &St, side: &RmsSide, eps: f32) -> Res<(V, V)> {
        let xv = self.view(side.x)?;
        let wv = self.view(side.w)?;
        let gv = self.view(
            side.gy
                .ok_or_else(|| metal_err("rms_norm_backward without a gradient"))?,
        )?;
        let (rows, dim) = (side.rows, side.dim);
        let gx = self.fresh(xv.n)?;
        let gw = self.fresh(dim as usize)?;
        let rstd = self.fresh(rows as usize)?;
        let chunks = rms_w_chunks(rows);
        let part = self.fresh(chunks as usize * dim as usize)?;
        for v in [&xv, &wv, &gv] {
            self.check(st, v, ST_IN)?;
        }
        self.ktg("ojas_rms_bwd_rows", rms_groups(rows), 1, RMS_THREADS, |b| {
            bind(b, &xv, 0);
            bind(b, &wv, 1);
            bind(b, &gv, 2);
            bind(b, &rstd, 3);
            bind(b, &gx, 4);
            bind_st(b, st, 5);
            set_u32(b, rows, 6);
            set_u32(b, dim, 7);
            set_f32(b, eps, 8);
        })?;
        self.k2("ojas_rms_bwd_w_part", dim as usize, chunks as usize, |b| {
            bind(b, &xv, 0);
            bind(b, &gv, 1);
            bind(b, &rstd, 2);
            bind(b, &part, 3);
            set_u32(b, rows, 4);
            set_u32(b, dim, 5);
        })?;
        self.k1("ojas_rms_bwd_w_sum", dim as usize, |b| {
            bind(b, &part, 0);
            bind(b, &gw, 1);
            set_u32(b, chunks, 2);
            set_u32(b, dim, 3);
        })?;
        self.check(st, &gx, ST_OUT)?;
        self.check(st, &gw, ST_OUT)?;
        Ok((gx, gw))
    }

    /// RMSNorm forward (`gy` unset) or backward of each side, each with its
    /// own status slot, q's recorded first. Once a side is encoded its slot
    /// is sealed: if k's work then fails to encode, q's fault still reaches
    /// the next sync point, as when q runs as an op of its own before k.
    fn rms_sides(&mut self, sides: &[RmsSide], eps: f32) -> Res<Reply> {
        let backward = sides.first().is_some_and(|s| s.gy.is_some());
        let op = if backward {
            "rms_norm_backward"
        } else {
            "rms_norm_forward"
        };
        let mut outs: Vec<V> = Vec::with_capacity(2 * sides.len());
        for side in sides {
            let st = self.status(op)?;
            if backward {
                let (gx, gw) = self.rms_bwd_encode(&st, side, eps)?;
                outs.push(gx);
                outs.push(gw);
            } else {
                outs.push(self.rms_encode(&st, side, eps)?);
            }
            self.seal();
        }
        Ok(self.keep(outs))
    }

    fn rope(
        &mut self,
        op: &'static str,
        [x, cos, sin]: [Arg; 3],
        [rows, dim, rotary]: [u32; 3],
        mode: RopeMode,
        backward: bool,
    ) -> Res<Reply> {
        if rotary > dim || !rotary.is_multiple_of(2) {
            return Err(metal_err(format!("{op}: rotary {rotary} of dim {dim}")));
        }
        let xv = self.view(x)?;
        let cv = self.view(cos)?;
        let sv = self.view(sin)?;
        let y = self.fresh(xv.n)?;
        let (mode_u, time, heads) = match mode {
            RopeMode::Same => (0u32, 1u32, 1u32),
            RopeMode::TimeDim { time, heads } => (1, time, heads),
        };
        let st = self.status(op)?;
        for v in [&xv, &cv, &sv] {
            self.check(&st, v, ST_IN)?;
        }
        // Units: one per rotated pair, then one per copied column.
        let units = (rotary / 2 + (dim - rotary)) as usize;
        self.k2("ojas_rope", units, rows as usize, |b| {
            bind(b, &xv, 0);
            bind(b, &cv, 1);
            bind(b, &sv, 2);
            bind(b, &y, 3);
            set_u32(b, rows, 4);
            set_u32(b, dim, 5);
            set_u32(b, mode_u, 6);
            set_u32(b, time, 7);
            set_u32(b, heads, 8);
            set_u32(b, u32::from(backward), 9);
            set_u32(b, rotary, 10);
        })?;
        self.check(&st, &y, ST_OUT)?;
        Ok(self.keep(vec![y]))
    }

    /// Compiled width of the tiled attention kernels (`ojas_attn_*_d*`).
    /// Their threadgroup memory does not grow with the head dim, so they are
    /// built up to 256; `ojas_core::METAL_MAX_HEAD_DIM` may refuse less
    /// before a call gets here.
    fn attn_width(d: u32) -> Res<u32> {
        match d {
            1..=16 => Ok(16),
            17..=32 => Ok(32),
            33..=64 => Ok(64),
            65..=128 => Ok(128),
            129..=ATTN_MAX_HEAD_DIM => Ok(256),
            _ => Err(OjasError::UnsupportedHeadDim {
                head_dim: d,
                limit: ATTN_MAX_HEAD_DIM,
            }),
        }
    }

    /// The views of one attention launch hold exactly the planes `g`
    /// names: `bh` query planes and `bh / rep` KV planes of `t * d` each.
    /// The kernels read KV plane `bh / rep`, so a short K or V would be read
    /// past its end.
    fn sdpa_views(op: &'static str, g: &SdpaGeom, q: &[&V], kv: &[&V], rows: &[&V]) -> Res<()> {
        if g.rep == 0 || !g.bh.is_multiple_of(g.rep) {
            return Err(metal_err(format!(
                "{op}: {} query planes do not group by {}",
                g.bh, g.rep
            )));
        }
        let plane = g.t as usize * g.d as usize;
        let q_n = g.bh as usize * plane;
        let kv_n = (g.bh / g.rep) as usize * plane;
        let rows_n = g.bh as usize * g.t as usize;
        let fits = q.iter().all(|v| v.n == q_n)
            && kv.iter().all(|v| v.n == kv_n)
            && rows.iter().all(|v| v.n == rows_n);
        if !fits {
            return Err(metal_err(format!(
                "{op}: buffers do not hold {} query and {} kv planes of [{}, {}]",
                g.bh,
                g.bh / g.rep,
                g.t,
                g.d
            )));
        }
        Ok(())
    }

    /// The tiled forward in `ojas_backend.metal`: the output and each query
    /// row's log-sum-exp. Grouped-query heads read their KV plane in place.
    fn sdpa(&mut self, q: Arg, k: Arg, v: Arg, g: SdpaGeom) -> Res<Reply> {
        const OP: &str = "causal_sdpa_forward";
        let width = Self::attn_width(g.d)?;
        let scale = sdpa_scale(g.d)?;
        attn_plane_fits(OP, g.t, g.d)?;
        let (qv, kv, vv) = (self.view(q)?, self.view(k)?, self.view(v)?);
        Self::sdpa_views(OP, &g, &[&qv], &[&kv, &vv], &[])?;
        let o = self.fresh(qv.n)?;
        let lse = self.fresh(g.bh as usize * g.t as usize)?;
        let st = self.status(OP)?;
        // No separate pass over Q, K or V. A non-finite Q or K makes a live
        // score non-finite, which the kernel reports; a non-finite V shows
        // up in O, which the output check reports.
        if qv.n > 0 {
            let name = format!("ojas_attn_fwd_d{width}");
            let groups = (g.t as usize).div_ceil(ATTN_ROWS);
            self.ktg(&name, groups, g.bh as usize, ATTN_THREADS, |b| {
                bind(b, &qv, 0);
                bind(b, &kv, 1);
                bind(b, &vv, 2);
                bind(b, &o, 3);
                bind(b, &lse, 4);
                bind_st(b, &st, 5);
                set_u32(b, g.t, 6);
                set_u32(b, g.d, 7);
                set_u32(b, g.rep, 8);
                set_u32(b, g.window, 9);
                set_f32(b, scale, 10);
            })?;
        }
        self.check(&st, &o, ST_OUT)?;
        self.check(&st, &lse, ST_OUT)?;
        Ok(self.keep(vec![o, lse]))
    }

    /// General attention forward (`ojas_attn_gen_fwd`): bidirectional or
    /// causal, `Tq != Tk`, an additive bias and per-batch key lengths, one
    /// simdgroup per query row. Replies with the output and each query row's
    /// log-sum-exp. Every operand is scanned for NaN and infinity first, as
    /// the CPU refuses one wherever it sits; a score that overflows sets the
    /// output word.
    fn attn_gen(
        &mut self,
        [q, k, v]: [Arg; 3],
        bias: Option<Arg>,
        lens: Option<Arg>,
        g: AttnGeom,
    ) -> Res<Reply> {
        const OP: &str = "attention_forward_saving";
        // One simdgroup per query row, four to a threadgroup.
        const ROWS_PER_TG: usize = ATTN_THREADS / 32;
        let scale = sdpa_scale(g.d)?;
        if g.d > ATTN_MAX_HEAD_DIM || g.rep == 0 || !g.bh.is_multiple_of(g.rep) {
            return Err(metal_err(format!(
                "{OP}: head dim {} over {} planes of {} per KV plane",
                g.d, g.bh, g.rep
            )));
        }
        let (qv, kv, vv) = (self.view(q)?, self.view(k)?, self.view(v)?);
        let bv = bias.map(|a| self.view(a)).transpose()?;
        let lv = lens.map(|a| self.view(a)).transpose()?;
        let (q_len, kv_len, d) = (g.q_len as usize, g.kv_len as usize, g.d as usize);
        let planes = g.bh as usize;
        let q_n = planes * q_len * d;
        let kv_n = (planes / g.rep as usize) * kv_len * d;
        let bias_n = g
            .bias
            .map(|[b, h, t]| b as usize * h as usize * t as usize * kv_len);
        let batches = planes / g.heads as usize;
        let fits = qv.n == q_n
            && kv.n == kv_n
            && vv.n == kv_n
            && bv.as_ref().map(|b| b.n) == bias_n
            && lv.as_ref().map(|l| l.n) == g.has_lens.then_some(batches)
            && planes.is_multiple_of(g.heads as usize);
        if !fits {
            return Err(metal_err(format!(
                "{OP}: buffers do not hold {planes} query planes of [{q_len}, {d}],                  {} kv planes of [{kv_len}, {d}] and the bias and key lengths asked for",
                planes / g.rep as usize
            )));
        }
        let o = self.fresh(qv.n)?;
        let lse = self.fresh(planes * q_len)?;
        let st = self.status(OP)?;
        self.check_operand(&st, q, &qv, ST_IN)?;
        self.check_operand(&st, k, &kv, ST_IN)?;
        self.check_operand(&st, v, &vv, ST_IN)?;
        if let (Some(a), Some(b)) = (bias, bv.as_ref()) {
            self.check_operand(&st, a, b, ST_IN)?;
        }
        if qv.n > 0 {
            let flags = u32::from(g.causal)
                | (u32::from(g.bias.is_some()) << 1)
                | (u32::from(g.has_lens) << 2);
            let [bias_b, bias_h, bias_q] = g.bias.unwrap_or([1, 1, 1]);
            // A buffer the kernel does not read this call is still bound:
            // Metal's validation wants every declared argument set.
            let (bias_buf, lens_buf) = (bv.as_ref().unwrap_or(&qv), lv.as_ref().unwrap_or(&qv));
            self.ktg(
                "ojas_attn_gen_fwd",
                q_len.div_ceil(ROWS_PER_TG),
                planes,
                ATTN_THREADS,
                |b| {
                    bind(b, &qv, 0);
                    bind(b, &kv, 1);
                    bind(b, &vv, 2);
                    bind(b, &o, 3);
                    bind(b, &lse, 4);
                    bind(b, bias_buf, 5);
                    bind(b, lens_buf, 6);
                    bind_st(b, &st, 7);
                    set_u32(b, g.q_len, 8);
                    set_u32(b, g.kv_len, 9);
                    set_u32(b, g.d, 10);
                    set_u32(b, g.heads, 11);
                    set_u32(b, g.rep, 12);
                    set_u32(b, flags, 13);
                    set_u32(b, g.window, 14);
                    set_u32(b, bias_b, 15);
                    set_u32(b, bias_h, 16);
                    set_u32(b, bias_q, 17);
                    set_f32(b, scale, 18);
                },
            )?;
        }
        self.check(&st, &o, ST_OUT)?;
        self.check(&st, &lse, ST_OUT)?;
        Ok(self.keep(vec![o, lse]))
    }

    /// The tiled backward in `ojas_backend.metal`, from the forward's output
    /// and log-sum-exp: `dr` forms each query row's `Dr = dO · O`, `dq`
    /// walks key blocks per query block, and `dkv` walks query blocks (of
    /// every query plane sharing the KV plane) per key block. Three
    /// dispatches, no T x T buffer, no expanded K or V, no wait.
    fn sdpa_bwd(&mut self, args: [Arg; 6], g: SdpaGeom) -> Res<Reply> {
        const OP: &str = "causal_sdpa_backward";
        let width = Self::attn_width(g.d)?;
        let scale = sdpa_scale(g.d)?;
        attn_plane_fits(OP, g.t, g.d)?;
        let [q, k, v, o, lse, gy] = args;
        let (qv, kv, vv) = (self.view(q)?, self.view(k)?, self.view(v)?);
        let (ov, lv, gv) = (self.view(o)?, self.view(lse)?, self.view(gy)?);
        Self::sdpa_views(OP, &g, &[&qv, &ov, &gv], &[&kv, &vv], &[&lv])?;
        let rows = g.bh as usize * g.t as usize;
        let dvec = self.fresh(rows)?;
        let dq = self.fresh(qv.n)?;
        let dk = self.fresh(kv.n)?;
        let dv = self.fresh(vv.n)?;
        let st = self.status(OP)?;
        for x in [&qv, &kv, &vv, &ov, &lv, &gv] {
            self.check(&st, x, ST_IN)?;
        }
        if qv.n > 0 {
            let rows_u32 = u32_of(rows)?;
            self.ktg(
                "ojas_attn_bwd_dr",
                rows.div_ceil(ATTN_THREADS / 32),
                1,
                ATTN_THREADS,
                |b| {
                    bind(b, &gv, 0);
                    bind(b, &ov, 1);
                    bind(b, &dvec, 2);
                    set_u32(b, rows_u32, 3);
                    set_u32(b, g.d, 4);
                },
            )?;
            let groups = (g.t as usize).div_ceil(ATTN_ROWS);
            let inputs = |b: &mut Binder<'_>| {
                bind(b, &qv, 0);
                bind(b, &kv, 1);
                bind(b, &vv, 2);
                bind(b, &gv, 3);
                bind(b, &lv, 4);
                bind(b, &dvec, 5);
            };
            self.ktg(
                &format!("ojas_attn_bwd_dq_d{width}"),
                groups,
                g.bh as usize,
                ATTN_THREADS,
                |b| {
                    inputs(b);
                    bind(b, &dq, 6);
                    set_u32(b, g.t, 7);
                    set_u32(b, g.d, 8);
                    set_u32(b, g.rep, 9);
                    set_u32(b, g.window, 10);
                    set_f32(b, scale, 11);
                },
            )?;
            self.ktg(
                &format!("ojas_attn_bwd_dkv_d{width}"),
                groups,
                (g.bh / g.rep) as usize,
                ATTN_THREADS,
                |b| {
                    inputs(b);
                    bind(b, &dk, 6);
                    bind(b, &dv, 7);
                    set_u32(b, g.t, 8);
                    set_u32(b, g.d, 9);
                    set_u32(b, g.rep, 10);
                    set_u32(b, g.window, 11);
                    set_f32(b, scale, 12);
                },
            )?;
        }
        for x in [&dq, &dk, &dv] {
            self.check(&st, x, ST_OUT)?;
        }
        Ok(self.keep(vec![dq, dk, dv]))
    }

    /// `pre = x @ W^T` is a tessl GEMM; the sigmoid, its backward and the
    /// bias sum are this crate's gate kernels. A saving forward also writes
    /// the sigmoid `[rows, heads]`; a saved backward reads it in place of
    /// forming `pre` and reading `bias`. `dims` is rows, d_model, heads,
    /// head_dim.
    fn gate(
        &mut self,
        x: Arg,
        w: Arg,
        bias: Arg,
        attn: Arg,
        mode: GateMode,
        dims: [u32; 4],
    ) -> Res<Reply> {
        let [rows, din, heads, dh] = dims;
        let (gy, saved) = match mode {
            GateMode::Forward | GateMode::ForwardSaving => (None, None),
            GateMode::Backward(gy) => (Some(gy), None),
            GateMode::BackwardSaved(gy, scales) => (Some(gy), Some(scales)),
        };
        let op = if gy.is_some() {
            "per_head_sigmoid_gate_backward"
        } else {
            "per_head_sigmoid_gate_forward"
        };
        let (xv, wv, bv, av) = (
            self.view(x)?,
            self.view(w)?,
            self.view(bias)?,
            self.view(attn)?,
        );
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let sv = saved.map(|s| self.view(s)).transpose()?;
        let (r, di, h) = (rows as usize, din as usize, heads as usize);
        if sv.as_ref().is_some_and(|s| s.n != r * h) {
            return Err(metal_err(format!(
                "{op}: saved scales hold a different count than rows * heads"
            )));
        }
        let units = u32_of(r * h)?;
        let plane = u32_of(av.n)?;
        let st = self.status(op)?;
        // `x` and `w` are read only by tessl GEMMs, so they keep standalone
        // checks in both directions. The backward kernels check `bias`,
        // `attn`, `gy`, `pre` (or the saved scale), `d_attn` and `d_bias`
        // themselves (every row reads every head's bias, and an empty
        // operand was refused).
        self.check(&st, &xv, ST_IN)?;
        self.check(&st, &wv, ST_IN)?;
        if gv.is_none() {
            self.check(&st, &bv, ST_IN)?;
            self.check(&st, &av, ST_IN)?;
        }
        // The saved backward never reads `bias`, but a non-finite bias is
        // the same refusal it is for the recomputing one and the CPU.
        if sv.is_some() {
            self.check(&st, &bv, ST_IN)?;
        }
        let x_t = self.mat(&xv, r, di)?;
        let w_t = self.mat(&wv, h, di)?;
        let pre = if sv.is_some() {
            None
        } else {
            let pre = self.fresh(r * h)?;
            GemmOperands::ExactF32
                .nt(&x_t, &w_t, &self.tt(&pre, &[r, h])?)
                .map_err(metal_err)?;
            Some(pre)
        };
        let Some(g) = gv else {
            let pre = pre.ok_or_else(|| metal_err(format!("{op}: forward without logits")))?;
            self.check(&st, &pre, ST_OUT)?;
            let out = self.fresh(av.n)?;
            self.k2("ojas_per_head_gate_fwd", dh as usize, r * h, |b| {
                bind(b, &av, 0);
                bind(b, &pre, 1);
                bind(b, &bv, 2);
                bind(b, &out, 3);
                set_u32(b, rows, 4);
                set_u32(b, heads, 5);
                set_u32(b, dh, 6);
                set_u32(b, plane, 7);
                set_u32(b, units, 8);
                set_u32(b, heads, 9);
                set_u32(b, plane, 10);
            })?;
            self.check(&st, &out, ST_OUT)?;
            if !matches!(mode, GateMode::ForwardSaving) {
                return Ok(self.keep(vec![out]));
            }
            let scale = self.fresh(r * h)?;
            self.k1("ojas_per_head_gate_scale", r * h, |b| {
                bind(b, &pre, 0);
                bind(b, &bv, 1);
                bind(b, &scale, 2);
                set_u32(b, rows, 3);
                set_u32(b, heads, 4);
                set_u32(b, units, 5);
                set_u32(b, heads, 6);
            })?;
            return Ok(self.keep(vec![out, scale]));
        };
        let d_attn = self.fresh(av.n)?;
        let d_pre = self.fresh(r * h)?;
        let d_bias = self.fresh(h)?;
        let gx = self.fresh(r * di)?;
        let gw = self.fresh(h * di)?;
        match (&pre, &sv) {
            (_, Some(scale)) => self.k1("ojas_per_head_gate_bwd_saved", r * h, |b| {
                bind(b, &av, 0);
                bind(b, scale, 1);
                bind(b, &g, 2);
                bind(b, &d_attn, 3);
                bind(b, &d_pre, 4);
                set_u32(b, rows, 5);
                set_u32(b, heads, 6);
                set_u32(b, dh, 7);
                set_u32(b, plane, 8);
                set_u32(b, units, 9);
                bind_st(b, &st, 10);
            })?,
            (Some(pre), None) => self.k1("ojas_per_head_gate_bwd", r * h, |b| {
                bind(b, &av, 0);
                bind(b, pre, 1);
                bind(b, &bv, 2);
                bind(b, &g, 3);
                bind(b, &d_attn, 4);
                bind(b, &d_pre, 5);
                set_u32(b, rows, 6);
                set_u32(b, heads, 7);
                set_u32(b, dh, 8);
                set_u32(b, plane, 9);
                set_u32(b, units, 10);
                set_u32(b, heads, 11);
                bind_st(b, &st, 12);
            })?,
            (None, None) => return Err(metal_err(format!("{op}: backward without logits"))),
        }
        self.k1("ojas_per_head_gate_dbias", gate_dbias_threads(h)?, |b| {
            bind(b, &d_pre, 0);
            bind(b, &d_bias, 1);
            set_u32(b, rows, 2);
            set_u32(b, heads, 3);
            set_u32(b, units, 4);
            set_u32(b, heads, 5);
            bind_st(b, &st, 6);
        })?;
        let dp_t = self.tt(&d_pre, &[r, h])?;
        GemmOperands::ExactF32
            .nn(&dp_t, &w_t, &self.tt(&gx, &[r, di])?)
            .map_err(metal_err)?;
        GemmOperands::ExactF32
            .tn(&dp_t, &x_t, &self.tt(&gw, &[h, di])?)
            .map_err(metal_err)?;
        self.check(&st, &gx, ST_OUT)?;
        self.check(&st, &gw, ST_OUT)?;
        Ok(self.keep(vec![gx, gw, d_bias, d_attn]))
    }

    fn vres(&mut self, v: Arg, v0: Arg, lam: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "value_residual_blend_backward"
        } else {
            "value_residual_blend_forward"
        };
        let (vv, v0v, lv) = (self.view(v)?, self.view(v0)?, self.view(lam)?);
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let n = u32_of(vv.n)?;
        let st = self.status(op)?;
        // The kernels check `v`, `v0`, `gy` and their full-size outputs
        // themselves; `lambda` (one value) and its gradient keep their passes.
        self.check(&st, &lv, ST_IN)?;
        let Some(g) = gv else {
            let y = self.fresh(vv.n)?;
            self.k1("ojas_vres_fwd", vv.n, |b| {
                bind(b, &vv, 0);
                bind(b, &v0v, 1);
                bind(b, &lv, 2);
                bind(b, &y, 3);
                set_u32(b, n, 4);
                bind_st(b, &st, 5);
            })?;
            return Ok(self.keep(vec![y]));
        };
        let gvv = self.fresh(vv.n)?;
        let gv0 = self.fresh(vv.n)?;
        let terms = self.fresh(vv.n)?;
        let sum = self.fresh(1)?;
        let glam = self.fresh(1)?;
        self.k1("ojas_vres_bwd", vv.n, |b| {
            bind(b, &vv, 0);
            bind(b, &v0v, 1);
            bind(b, &lv, 2);
            bind(b, &g, 3);
            bind(b, &gvv, 4);
            bind(b, &gv0, 5);
            bind(b, &terms, 6);
            set_u32(b, n, 7);
            bind_st(b, &st, 8);
        })?;
        self.reduce(&terms, 0, None, &sum.buf, 0)?;
        self.k1("ojas_vres_lambda", 1, |b| {
            bind(b, &sum, 0);
            bind(b, &lv, 1);
            bind(b, &glam, 2);
        })?;
        self.check(&st, &glam, ST_OUT)?;
        Ok(self.keep(vec![gvv, gv0, glam]))
    }

    /// The gated delta rule's operand views and tessl dims. tessl's kernels
    /// scan nothing, so every operand is checked into `ST_IN` and every
    /// output into `ST_OUT`, as the GEMM ops are.
    fn gdn_views(&self, x: &GdnArgs) -> Res<(GdnTrainDims, [V; 5], Option<V>)> {
        let dims = GdnTrainDims {
            batch: x.batch,
            seq: x.seq,
            heads: x.heads,
            v_dim: x.v_dim,
        };
        let views = [
            self.view(x.q)?,
            self.view(x.k)?,
            self.view(x.v)?,
            self.view(x.g)?,
            self.view(x.beta)?,
        ];
        let s0 = x.s0.map(|a| self.view(a)).transpose()?;
        Ok((dims, views, s0))
    }

    /// `[B, T, H]`, `[B, T, H, 128]`, `[B, T, H, Dv]`, `[B, H, 128, Dv]`.
    fn gdn_shapes(d: &GdnTrainDims) -> ([usize; 3], [usize; 4], [usize; 4], [usize; 4]) {
        let (b, t, h, dv) = (
            d.batch as usize,
            d.seq as usize,
            d.heads as usize,
            d.v_dim as usize,
        );
        let dk = GDN_TRAIN_DK as usize;
        ([b, t, h], [b, t, h, dk], [b, t, h, dv], [b, h, dk, dv])
    }

    fn gdn(&mut self, x: GdnArgs) -> Res<Reply> {
        const OP: &str = "chunked_gdn_forward";
        let (dims, [q, k, v, g, beta], s0) = self.gdn_views(&x)?;
        let (tok, key, val, state) = Self::gdn_shapes(&dims);
        let o = self.fresh(val.iter().product())?;
        let fin = self.fresh(state.iter().product())?;
        let ckpt = self.fresh(dims.checkpoint_shape().iter().product())?;
        let st = self.status(OP)?;
        for input in [&q, &k, &v, &g, &beta].into_iter().chain(s0.as_ref()) {
            self.check(&st, input, ST_IN)?;
        }
        let s0_t = s0.as_ref().map(|s| self.tt(s, &state)).transpose()?;
        gdn_train_forward(
            &self.rt,
            dims,
            GdnTrainInputs {
                q: &self.tt(&q, &key)?,
                k: &self.tt(&k, &key)?,
                v: &self.tt(&v, &val)?,
                g: &self.tt(&g, &tok)?,
                beta: &self.tt(&beta, &tok)?,
                s0: s0_t.as_ref(),
            },
            &self.tt(&o, &val)?,
            Some(&self.tt(&fin, &state)?),
            &self.tt(&ckpt, &dims.checkpoint_shape())?,
        )
        .map_err(metal_err)?;
        for out in [&o, &fin, &ckpt] {
            self.check(&st, out, ST_OUT)?;
        }
        Ok(self.keep(vec![o, fin, ckpt]))
    }

    fn gdn_bwd(&mut self, x: GdnArgs, ckpt: Arg, d_o: Arg, d_fin: Option<Arg>) -> Res<Reply> {
        const OP: &str = "chunked_gdn_backward";
        let (dims, [q, k, v, g, beta], s0) = self.gdn_views(&x)?;
        let (ckv, dov) = (self.view(ckpt)?, self.view(d_o)?);
        let dfv = d_fin.map(|a| self.view(a)).transpose()?;
        let ws_bytes = GdnTrainWorkspace::bytes_for(dims);
        if x.ws_elems.checked_mul(4) != Some(ws_bytes) {
            return Err(metal_err(format!(
                "{OP}: the backend charged {} workspace values; tessl needs {ws_bytes} bytes",
                x.ws_elems
            )));
        }
        let (tok, key, val, state) = Self::gdn_shapes(&dims);
        let n = |s: &[usize]| s.iter().product::<usize>();
        let (dq, dk, dv) = (
            self.fresh(n(&key))?,
            self.fresh(n(&key))?,
            self.fresh(n(&val))?,
        );
        let (dg, dbeta) = (self.fresh(n(&tok))?, self.fresh(n(&tok))?);
        let ds0 = s0.as_ref().map(|_| self.fresh(n(&state))).transpose()?;
        let ws = self.gdn_workspace(dims)?;
        let st = self.status(OP)?;
        let inputs = [&q, &k, &v, &g, &beta, &ckv, &dov];
        for input in inputs.into_iter().chain(s0.as_ref()).chain(dfv.as_ref()) {
            self.check(&st, input, ST_IN)?;
        }
        let s0_t = s0.as_ref().map(|s| self.tt(s, &state)).transpose()?;
        let df_t = dfv.as_ref().map(|s| self.tt(s, &state)).transpose()?;
        let ds0_t = ds0.as_ref().map(|s| self.tt(s, &state)).transpose()?;
        gdn_train_backward(
            &self.rt,
            dims,
            GdnTrainInputs {
                q: &self.tt(&q, &key)?,
                k: &self.tt(&k, &key)?,
                v: &self.tt(&v, &val)?,
                g: &self.tt(&g, &tok)?,
                beta: &self.tt(&beta, &tok)?,
                s0: s0_t.as_ref(),
            },
            &self.tt(&ckv, &dims.checkpoint_shape())?,
            &self.tt(&dov, &val)?,
            df_t.as_ref(),
            &ws,
            GdnTrainGrads {
                dq: &self.tt(&dq, &key)?,
                dk: &self.tt(&dk, &key)?,
                dv: &self.tt(&dv, &val)?,
                dg: &self.tt(&dg, &tok)?,
                dbeta: &self.tt(&dbeta, &tok)?,
                ds0: ds0_t.as_ref(),
            },
        )
        .map_err(metal_err)?;
        let mut outs = vec![dq, dk, dv, dg, dbeta];
        outs.extend(ds0);
        for out in &outs {
            self.check(&st, out, ST_OUT)?;
        }
        Ok(self.keep(outs))
    }

    /// The views of `args`, each at offset 0 of its own buffer: tessl's
    /// row-local kernels take a buffer and a column window, not a byte
    /// offset, so a view that does not start its buffer ([`ReadAs::Whole`])
    /// is copied first, into memory the backend charged as scratch.
    fn wholes<const N: usize>(&self, args: [Arg; N]) -> Res<[V; N]> {
        let mut out = Vec::with_capacity(N);
        for a in args {
            let v = self.view(a)?;
            if a.copied(ReadAs::Whole) {
                let dst = self.fresh(v.n)?;
                self.copy(&v, &dst)?;
                out.push(dst);
            } else {
                out.push(v);
            }
        }
        out.try_into()
            .map_err(|_| metal_err("operand count changed"))
    }

    /// Weight-gradient scratch of `need` values, refused unless the backend
    /// charged exactly that many.
    fn part(&self, op: &'static str, charged: usize, need: usize) -> Res<V> {
        if charged != need {
            return Err(metal_err(format!(
                "{op}: the backend charged {charged} scratch values; tessl needs {need}"
            )));
        }
        self.fresh(need)
    }

    /// tessl's `qwen35::conv1d_silu` from a zero state. It scans nothing,
    /// so the operands and the output are checked here.
    fn conv1d(&mut self, x: Arg, w: Arg, g: Conv1dGeom) -> Res<Reply> {
        const OP: &str = "causal_conv1d_silu_forward";
        let [xv, wv] = self.wholes([x, w])?;
        let y = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        for v in [&xv, &wv] {
            self.check(&st, v, ST_IN)?;
        }
        conv1d_silu(
            &self.rt,
            Cols::dense(&xv.buf, g.channels),
            &wv.buf,
            g.width,
            StateIn::Zero,
            &y.buf,
            None,
            g.batch,
            g.seq,
            g.channels,
        )
        .map_err(metal_err)?;
        self.check(&st, &y, ST_OUT)?;
        Ok(self.keep(vec![y]))
    }

    fn conv1d_bwd(&mut self, args: [Arg; 3], g: Conv1dGeom, part: usize) -> Res<Reply> {
        const OP: &str = "causal_conv1d_silu_backward";
        let [xv, wv, gv] = self.wholes(args)?;
        let need = conv1d_silu_bwd_part_len(g.batch, g.seq, g.channels, g.width);
        let scratch = self.part(OP, part, need)?;
        let (dx, dw) = (self.fresh(xv.n)?, self.fresh(wv.n)?);
        let st = self.status(OP)?;
        for v in [&xv, &wv, &gv] {
            self.check(&st, v, ST_IN)?;
        }
        conv1d_silu_bwd(
            &self.rt,
            Cols::dense(&xv.buf, g.channels),
            &wv.buf,
            g.width,
            Cols::dense(&gv.buf, g.channels),
            Cols::dense(&dx.buf, g.channels),
            &dw.buf,
            &scratch.buf,
            g.batch,
            g.seq,
            g.channels,
        )
        .map_err(metal_err)?;
        for v in [&dx, &dw] {
            self.check(&st, v, ST_OUT)?;
        }
        Ok(self.keep(vec![dx, dw]))
    }

    /// tessl's `qwen35::gated_rms_norm` with one head per row: its `heads`
    /// is 1 and its `dim` the row.
    fn gated_rms(&mut self, args: [Arg; 3], [rows, dim]: [u32; 2], eps: f32) -> Res<Reply> {
        const OP: &str = "gated_rms_norm_forward";
        let [xv, zv, wv] = self.wholes(args)?;
        let y = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        for v in [&xv, &zv, &wv] {
            self.check(&st, v, ST_IN)?;
        }
        let out = OutCols {
            cols: Cols::dense(&y.buf, dim),
            dtype: DType::F32,
        };
        gated_rms_norm(
            &self.rt,
            Cols::dense(&xv.buf, dim),
            Cols::dense(&zv.buf, dim),
            &wv.buf,
            out,
            rows,
            1,
            dim,
            eps,
        )
        .map_err(metal_err)?;
        self.check(&st, &y, ST_OUT)?;
        Ok(self.keep(vec![y]))
    }

    fn gated_rms_bwd(
        &mut self,
        args: [Arg; 4],
        [rows, dim]: [u32; 2],
        eps: f32,
        part: usize,
    ) -> Res<Reply> {
        const OP: &str = "gated_rms_norm_backward";
        let [xv, zv, wv, gv] = self.wholes(args)?;
        let scratch = self.part(OP, part, gated_rms_norm_bwd_part_len(rows, 1, dim))?;
        let (dx, dz, dw) = (self.fresh(xv.n)?, self.fresh(zv.n)?, self.fresh(wv.n)?);
        let st = self.status(OP)?;
        for v in [&xv, &zv, &wv, &gv] {
            self.check(&st, v, ST_IN)?;
        }
        gated_rms_norm_bwd(
            &self.rt,
            Cols::dense(&xv.buf, dim),
            Cols::dense(&zv.buf, dim),
            &wv.buf,
            Cols::dense(&gv.buf, dim),
            Cols::dense(&dx.buf, dim),
            Cols::dense(&dz.buf, dim),
            &dw.buf,
            &scratch.buf,
            rows,
            1,
            dim,
            eps,
        )
        .map_err(metal_err)?;
        for v in [&dx, &dz, &dw] {
            self.check(&st, v, ST_OUT)?;
        }
        Ok(self.keep(vec![dx, dz, dw]))
    }

    /// No status slot and no finite scan: NaN rounds to a quiet NaN.
    fn round_bf16(&mut self, x: Arg) -> Res<Reply> {
        let xv = self.view(x)?;
        let y = self.fresh(xv.n)?;
        self.round_bf16_into(&xv, &y)?;
        Ok(self.keep(vec![y]))
    }

    /// `x` (a bf16 window: `n` 2-byte elements from byte offset `off`)
    /// widened exactly into a new f32 buffer. No status slot and no finite
    /// scan: a NaN or infinity is carried to the op that reads the copy.
    fn widen_bf16(&mut self, x: Arg) -> Res<Reply> {
        let h = self.half_view(x)?;
        let from_frozen = self.known_clean(x, 2);
        let y = self.fresh(h.n)?;
        self.widen_into(&h, 0, &y)?;
        let reply = self.keep(vec![y]);
        // Widening is exact, so the copy of a window known finite is known
        // finite, and nothing writes it: a matmul over a frozen bf16 weight
        // skips its scan of the copy as it does of the weight.
        if let (true, Reply::Bufs(bufs)) = (from_frozen, &reply) {
            let mut book = self.book.borrow_mut();
            for b in bufs {
                book.clean.insert(
                    b.id,
                    Clean {
                        windows: vec![(0, b.bytes)],
                        from: Some(x.id),
                    },
                );
            }
        }
        Ok(reply)
    }

    /// `x` (f32) rounded to bf16 into a new buffer of `n` 2-byte elements,
    /// its size rounded up to whole words. No status slot and no finite
    /// scan, as [`Self::round_bf16`].
    fn narrow_bf16(&mut self, x: Arg) -> Res<Reply> {
        let xv = self.view(x)?;
        let n = u32_of(xv.n)?;
        let bytes =
            xv.n.checked_mul(2)
                .ok_or_else(|| metal_err("allocation size overflows"))?;
        let buf = self.alloc(bytes.next_multiple_of(4))?;
        let y = V {
            buf,
            off: 0,
            n: xv.n,
        };
        self.k1("ojas_f32_to_bf16", xv.n, |b| {
            bind(b, &xv, 0);
            bind(b, &y, 1);
            set_u32(b, n, 2);
        })?;
        let id = self.next_id;
        self.next_id += 1;
        self.bufs.insert(id, y.buf);
        Ok(Reply::Bufs(vec![NewBuf { id, bytes }]))
    }

    fn silu(&mut self, x: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "silu_backward"
        } else {
            "silu_forward"
        };
        let xv = self.view(x)?;
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let st = self.status(op)?;
        // The kernels check their inputs and output themselves.
        match &gv {
            None => self.k1("ojas_silu_fwd", xv.n, |b| {
                bind(b, &xv, 0);
                bind(b, &y, 1);
                set_u32(b, n, 2);
                bind_st(b, &st, 3);
            })?,
            Some(g) => self.k1("ojas_silu_bwd", xv.n, |b| {
                bind(b, &xv, 0);
                bind(b, g, 1);
                bind(b, &y, 2);
                set_u32(b, n, 3);
                bind_st(b, &st, 4);
            })?,
        }
        Ok(self.keep(vec![y]))
    }

    fn sigmoid(&mut self, x: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "sigmoid_backward"
        } else {
            "sigmoid_forward"
        };
        let xv = self.view(x)?;
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let st = self.status(op)?;
        // The kernels check their inputs and output themselves.
        match &gv {
            None => self.k1("ojas_sigmoid_fwd", xv.n, |b| {
                bind(b, &xv, 0);
                bind(b, &y, 1);
                set_u32(b, n, 2);
                bind_st(b, &st, 3);
            })?,
            Some(g) => self.k1("ojas_sigmoid_bwd", xv.n, |b| {
                bind(b, &xv, 0);
                bind(b, g, 1);
                bind(b, &y, 2);
                set_u32(b, n, 3);
                bind_st(b, &st, 4);
            })?,
        }
        Ok(self.keep(vec![y]))
    }

    /// The gated delta rule's log decay over `a` `[rows, heads]`. The kernel
    /// checks its inputs and output itself.
    fn gdn_decay(&mut self, [a, a_log, dt_bias]: [Arg; 3], heads: u32) -> Res<Reply> {
        const OP: &str = "gdn_log_decay_forward";
        let (av, lv, dv) = (self.view(a)?, self.view(a_log)?, self.view(dt_bias)?);
        let n = u32_of(av.n)?;
        let g = self.fresh(av.n)?;
        let st = self.status(OP)?;
        self.k1("ojas_gdn_decay_fwd", av.n, |b| {
            bind(b, &av, 0);
            bind(b, &lv, 1);
            bind(b, &dv, 2);
            bind(b, &g, 3);
            set_u32(b, n, 4);
            set_u32(b, heads, 5);
            bind_st(b, &st, 6);
        })?;
        Ok(self.keep(vec![g]))
    }

    /// `da` per element, then `da_log` and `ddt_bias`, one thread per head
    /// summing the rows in ascending order. The kernels check their inputs
    /// and outputs themselves.
    fn gdn_decay_bwd(&mut self, [a, a_log, dt_bias, gy]: [Arg; 4], heads: u32) -> Res<Reply> {
        const OP: &str = "gdn_log_decay_backward";
        let (av, lv, dv) = (self.view(a)?, self.view(a_log)?, self.view(dt_bias)?);
        let gv = self.view(gy)?;
        let n = u32_of(av.n)?;
        let rows = n / heads.max(1);
        let (da, dlog, ddt) = (self.fresh(av.n)?, self.fresh(lv.n)?, self.fresh(dv.n)?);
        let st = self.status(OP)?;
        self.k1("ojas_gdn_decay_bwd", av.n, |b| {
            bind(b, &av, 0);
            bind(b, &lv, 1);
            bind(b, &dv, 2);
            bind(b, &gv, 3);
            bind(b, &da, 4);
            set_u32(b, n, 5);
            set_u32(b, heads, 6);
            bind_st(b, &st, 7);
        })?;
        self.k1("ojas_gdn_decay_bwd_sum", heads as usize, |b| {
            bind(b, &av, 0);
            bind(b, &lv, 1);
            bind(b, &dv, 2);
            bind(b, &gv, 3);
            bind(b, &da, 4);
            bind(b, &dlog, 5);
            bind(b, &ddt, 6);
            set_u32(b, rows, 7);
            set_u32(b, heads, 8);
            bind_st(b, &st, 9);
        })?;
        Ok(self.keep(vec![da, dlog, ddt]))
    }

    fn mul(&mut self, a: Arg, b_arg: Arg, gy: Option<Arg>) -> Res<Reply> {
        let op = if gy.is_some() {
            "mul_backward"
        } else {
            "mul_forward"
        };
        let (av, bv) = (self.view(a)?, self.view(b_arg)?);
        let gv = gy.map(|g| self.view(g)).transpose()?;
        let n = u32_of(av.n)?;
        let st = self.status(op)?;
        // The kernels check their inputs and outputs themselves.
        let Some(g) = gv else {
            let y = self.fresh(av.n)?;
            self.k1("ojas_mul_fwd", av.n, |b| {
                bind(b, &av, 0);
                bind(b, &bv, 1);
                bind(b, &y, 2);
                set_u32(b, n, 3);
                bind_st(b, &st, 4);
            })?;
            return Ok(self.keep(vec![y]));
        };
        let ga = self.fresh(av.n)?;
        let gb = self.fresh(av.n)?;
        self.k1("ojas_mul_bwd", av.n, |b| {
            bind(b, &av, 0);
            bind(b, &bv, 1);
            bind(b, &g, 2);
            bind(b, &ga, 3);
            bind(b, &gb, 4);
            set_u32(b, n, 5);
            bind_st(b, &st, 6);
        })?;
        Ok(self.keep(vec![ga, gb]))
    }

    /// `out = x + y` in one pass that also checks `x`, `y` and `out`.
    fn add(&mut self, x: Arg, y: Arg) -> Res<Reply> {
        const OP: &str = "residual_add_forward";
        let (xv, yv) = (self.view(x)?, self.view(y)?);
        let n = u32_of(xv.n)?;
        let out = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        self.k1("ojas_add_fwd", xv.n, |b| {
            bind(b, &xv, 0);
            bind(b, &yv, 1);
            bind(b, &out, 2);
            set_u32(b, n, 3);
            bind_st(b, &st, 4);
        })?;
        Ok(self.keep(vec![out]))
    }

    /// tessl's `bert::layer_norm`, which scans nothing: the operands and the
    /// output are checked here.
    fn layer_norm(&mut self, args: [Arg; 3], [rows, dim]: [u32; 2], eps: f32) -> Res<Reply> {
        const OP: &str = "layer_norm_forward";
        let [xv, wv, bv] = self.wholes(args)?;
        let y = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        for v in [&xv, &wv, &bv] {
            self.check(&st, v, ST_IN)?;
        }
        tessl::bert::layer_norm(&self.rt, &xv.buf, &wv.buf, &bv.buf, &y.buf, rows, dim, eps)
            .map_err(metal_err)?;
        self.check(&st, &y, ST_OUT)?;
        Ok(self.keep(vec![y]))
    }

    /// tessl's in-place `bert::bias_gelu_erf` over a copy of the operand, one
    /// column of zero bias (`x + 0.0` is `x`). It scans nothing, so the
    /// operand and the output are checked here.
    fn gelu_erf(&mut self, x: Arg) -> Res<Reply> {
        const OP: &str = "gelu_erf_forward";
        let xv = self.view(x)?;
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let zero = self.fresh_from(&[0u32])?;
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        self.copy(&xv, &y)?;
        tessl::bert::bias_gelu_erf(&self.rt, &y.buf, &zero.buf, n, 1).map_err(metal_err)?;
        self.check(&st, &y, ST_OUT)?;
        Ok(self.keep(vec![y]))
    }

    /// `x + y` or `x * y` over a numpy broadcast, one thread per output
    /// element; the kernel also checks both operands and the output.
    fn bcast(
        &mut self,
        [x, y]: [Arg; 2],
        mul: bool,
        rank: u32,
        oshape: &[u32],
        xstride: &[u32],
        ystride: &[u32],
    ) -> Res<Reply> {
        let op = if mul {
            "broadcast_mul_forward"
        } else {
            "broadcast_add_forward"
        };
        let (xv, yv) = (self.view(x)?, self.view(y)?);
        let out_n = oshape
            .iter()
            .take(rank as usize)
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| metal_err(format!("{op}: output size overflows")))?;
        let n = u32_of(out_n)?;
        let out = self.fresh(out_n)?;
        let words = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_ne_bytes()).collect() };
        let (shape_bytes, x_bytes, y_bytes) = (words(oshape), words(xstride), words(ystride));
        let st = self.status(op)?;
        self.k1("ojas_bcast_binary", out_n, |b| {
            bind(b, &xv, 0);
            bind(b, &yv, 1);
            bind(b, &out, 2);
            set_u32(b, n, 3);
            set_u32(b, rank, 4);
            set_u32(b, u32::from(mul), 5);
            b.bind_bytes(&shape_bytes, 6);
            b.bind_bytes(&x_bytes, 7);
            b.bind_bytes(&y_bytes, 8);
            bind_st(b, &st, 9);
        })?;
        Ok(self.keep(vec![out]))
    }

    /// `x * scale` in one pass that also checks `x` and the output.
    fn scale_fwd(&mut self, x: Arg, scale: f32) -> Res<Reply> {
        const OP: &str = "scale_forward";
        let xv = self.view(x)?;
        let n = u32_of(xv.n)?;
        let y = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        self.k1("ojas_scale_fwd", xv.n, |b| {
            bind(b, &xv, 0);
            bind(b, &y, 1);
            set_u32(b, n, 2);
            set_f32(b, scale, 3);
            bind_st(b, &st, 4);
        })?;
        Ok(self.keep(vec![y]))
    }

    /// Softmax of each row, one thread per row summing in ascending order;
    /// the kernel also checks the operand and the output.
    fn softmax(&mut self, x: Arg, rows: u32, dim: u32) -> Res<Reply> {
        const OP: &str = "softmax_forward";
        let xv = self.view(x)?;
        let y = self.fresh(xv.n)?;
        let st = self.status(OP)?;
        self.k1("ojas_softmax_rows", rows as usize, |b| {
            bind(b, &xv, 0);
            bind(b, &y, 1);
            set_u32(b, rows, 2);
            set_u32(b, dim, 3);
            bind_st(b, &st, 4);
        })?;
        Ok(self.keep(vec![y]))
    }

    /// Both gradients are copies of `gy`, written in one pass that also
    /// checks `x`, `y` and `gy`.
    fn add_bwd(&mut self, x: Arg, y: Arg, gy: Arg) -> Res<Reply> {
        const OP: &str = "residual_add_backward";
        let (xv, yv, gv) = (self.view(x)?, self.view(y)?, self.view(gy)?);
        let n = u32_of(gv.n)?;
        let gx = self.fresh(gv.n)?;
        let gyy = self.fresh(gv.n)?;
        let st = self.status(OP)?;
        self.k1("ojas_add_bwd", gv.n, |b| {
            bind(b, &xv, 0);
            bind(b, &yv, 1);
            bind(b, &gv, 2);
            bind(b, &gx, 3);
            bind(b, &gyy, 4);
            set_u32(b, n, 5);
            bind_st(b, &st, 6);
        })?;
        Ok(self.keep(vec![gx, gyy]))
    }

    #[allow(clippy::too_many_arguments)]
    fn ce(
        &mut self,
        logits: Arg,
        targets: Arg,
        rows: u32,
        vocab: u32,
        ignore: Option<u32>,
        valid: u32,
        grad: bool,
    ) -> Res<Reply> {
        let op = if grad {
            "cross_entropy_mean_backward"
        } else {
            "cross_entropy_mean_forward"
        };
        let lv = self.view(logits)?;
        let tv = self.view(targets)?;
        let loss_rows = self.fresh(rows as usize)?;
        let gout = if grad { Some(self.fresh(lv.n)?) } else { None };
        let st = self.status(op)?;
        // The logits' finite check and the gradient's are inside
        // `ojas_ce_fused` (every element of both is still checked). The
        // targets were range-checked and counted on the host.
        self.ktg("ojas_ce_fused", rows as usize, 1, 256, |b| {
            bind(b, &lv, 0);
            bind(b, &tv, 1);
            bind_st(b, &st, 2);
            bind(b, &loss_rows, 3);
            bind(b, gout.as_ref().unwrap_or(&loss_rows), 4);
            set_u32(b, rows, 5);
            set_u32(b, vocab, 6);
            set_u32(b, u32::from(ignore.is_some()), 7);
            set_u32(b, ignore.unwrap_or(0), 8);
            set_u32(b, u32::from(grad), 9);
            set_u32(b, valid, 10);
        })?;
        if let Some(g) = gout {
            return Ok(self.keep(vec![g]));
        }
        let sum = self.fresh(1)?;
        let loss = self.fresh(1)?;
        self.reduce(&loss_rows, 0, None, &sum.buf, 0)?;
        self.k1("ojas_ce_mean", 1, |b| {
            bind(b, &sum, 0);
            set_u32(b, valid, 1);
            bind(b, &loss, 2);
        })?;
        self.check(&st, &loss, ST_OUT)?;
        Ok(self.keep(vec![loss]))
    }

    /// The global L2 norm of `grads`: tessl's [`SqNormPlan`] over every
    /// non-empty gradient, two dispatches whatever their count (the squares
    /// in pieces through a table of GPU addresses, then an exact integer
    /// sum), so the norm is deterministic, finite for any finite gradients
    /// and rounded once to f64 before its square root. A sync point: it
    /// waits, then reports the first pending fault (an earlier op's) before
    /// the norm; a non-finite gradient makes the sum non-finite, reported as
    /// [`OjasError::NonFinite`] for this op. The table, pieces and result
    /// are this worker's scratch, charged by the caller.
    fn clip_norm(&mut self, grads: &[Arg]) -> Res<Reply> {
        const OP: &str = "clip_grad_norm";
        let views = grads
            .iter()
            .map(|a| self.view(*a))
            .collect::<Res<Vec<_>>>()?;
        // The backend refused empty gradients and an empty list
        // (`clip_grad_norm_dims`), and tessl refuses an empty window, so
        // there is at least one piece: `out` is always written.
        let windows: Vec<SqWindow<'_>> = views
            .iter()
            .map(|v| SqWindow::dense(&v.buf, v.off, DType::F32, v.n))
            .collect();
        let plan = SqNormPlan::new(&self.rt, &windows, OP).map_err(metal_err)?;
        if plan.pieces() == 0 {
            return Err(metal_err(format!("{OP}: no gradient values")));
        }
        let table = self.fresh_from(plan.table_words())?;
        let part = self.fresh(plan.parts_words())?;
        let out = self.fresh(SQ_OUT_WORDS)?;
        plan.encode(
            &self.rt,
            (&table.buf, table.off),
            (&part.buf, part.off),
            (&out.buf, out.off),
            OP,
        )
        .map_err(metal_err)?;
        self.settle(Scan::All, Wait::ClipNorm)?;
        self.report()?;
        let sq = {
            let map = out.buf.try_contents_u32().map_err(metal_err)?;
            let at = out.off / 4;
            decode_sq_norm(&map[at..at + SQ_OUT_WORDS]).map_err(metal_err)?
        };
        if !sq.is_finite() {
            return Err(OjasError::NonFinite { op: OP });
        }
        Ok(Reply::Norm(sq.sqrt() as f32))
    }

    /// Every gradient times `scale` in place, `NORM_SLOTS` gradients per
    /// dispatch; a non-finite product is flagged by the same kernel.
    fn scale(&mut self, op: &'static str, grads: &[Arg], scale: f32) -> Res<Reply> {
        let views = grads
            .iter()
            .map(|a| self.view(*a))
            .collect::<Res<Vec<_>>>()?;
        for a in grads {
            self.writable(op, *a)?;
        }
        let st = self.status(op)?;
        let spare = &self.spare;
        for group in views.chunks(NORM_SLOTS) {
            let (tbl, groups) = norm_table(group)?;
            let used = u32_of(group.len())?;
            self.ktg("ojas_scale_multi", groups, 1, 256, |b| {
                for slot in 0..NORM_SLOTS {
                    match group.get(slot) {
                        Some(v) => bind(b, v, slot),
                        None => set_gpu_buf_offset(b, spare, 4 * slot, slot),
                    }
                }
                bind_st(b, &st, 25);
                b.bind_bytes(&tbl, 26);
                set_u32(b, used, 27);
                set_f32(b, scale, 28);
            })?;
        }
        Ok(Reply::Done)
    }

    /// A GEMM operand for rows `row0..row0 + rows` of a `[_, cols]` matrix:
    /// the view itself when its start is 16-byte aligned, otherwise a copy in
    /// `stage`, which the caller allocated when [`LceGeom`] said some tile
    /// would need it.
    fn rows_operand(
        &self,
        m: &V,
        row0: usize,
        rows: usize,
        cols: usize,
        stage: Option<&V>,
    ) -> Res<TT> {
        let view = sub(m, row0 * cols, rows * cols);
        if view.off.is_multiple_of(16) {
            return self.tt(&view, &[rows, cols]);
        }
        let stage = stage.ok_or_else(|| metal_err("unstaged misaligned GEMM operand"))?;
        let dst = sub(stage, 0, rows * cols);
        self.copy(&view, &dst)?;
        self.tt(&dst, &[rows, cols])
    }

    fn add_into(&self, src: &V, dst: &V, assign: bool) -> Res<()> {
        let n = u32_of(dst.n)?;
        self.k1("ojas_add_into", dst.n, |b| {
            bind(b, src, 0);
            bind(b, dst, 1);
            set_u32(b, n, 2);
            set_u32(b, u32::from(assign), 3);
        })
    }

    /// Linear cross-entropy, one `[rt, ct]` logits tile at a time. Per row
    /// tile: pass 1 makes every logits tile (tessl GEMM) and merges its
    /// (max, sum of exp) into the rows' running pair; the loss rows follow.
    /// With gradients, pass 2 makes each tile again, turns it into the
    /// gradient of the logits in place (with one vocabulary tile, the tile
    /// pass 1 left is reused rather than remade), and adds `G @ W_tile` into
    /// grad_input's rows and `G^T @ x_tile` into grad_weight's rows, each
    /// only for a gradient `g.grads` asks for (a frozen head gets neither
    /// the `[V, d]` output nor its GEMMs). The
    /// targets were range-checked, and `valid` counted, on the host. A
    /// `geom.w_half` (bf16) weight is scanned on its bits, and each `[ct, d]`
    /// weight tile is widened into the stage buffer when it is used, so no
    /// f32 copy of the whole table exists.
    fn linear_ce(
        &mut self,
        x: Arg,
        w: Arg,
        t: Arg,
        g: LceGeom,
        ignore: Option<u32>,
        valid: u32,
    ) -> Res<Reply> {
        const OP: &str = "linear_cross_entropy_mean";
        let (xv, tv) = (self.view(x)?, self.view(t)?);
        let wv = if g.w_half {
            LceWeight::Half(self.half_view(w)?)
        } else {
            LceWeight::F32(self.view(w)?)
        };
        let (n, d, v) = (g.n, g.d, g.v);
        if xv.n != n * d || wv.n() != v * d || tv.n != n {
            return Err(metal_err(format!(
                "{OP}: windows do not match the dimensions"
            )));
        }
        let vocab = u32_of(v)?;
        let (has_ignore, ignore_id) = (u32::from(ignore.is_some()), ignore.unwrap_or(0));
        let loss = self.fresh(1)?;
        let gx = g.grads.input.then(|| self.fresh(n * d)).transpose()?;
        let gw = g.grads.weight.then(|| self.fresh(v * d)).transpose()?;
        let tile = self.fresh(g.rt * g.ct)?;
        let stats = self.fresh(3 * g.rt)?;
        let loss_rows = self.fresh(n)?;
        let sum = self.fresh(1)?;
        let gx_tmp = g.grads.input.then(|| self.fresh(g.rt * d)).transpose()?;
        let gw_tmp = g.grads.weight.then(|| self.fresh(g.ct * d)).transpose()?;
        let stage_x = if g.stage_x {
            Some(self.fresh(g.rt * d)?)
        } else {
            None
        };
        let stage_w = if g.stage_w {
            Some(self.fresh(g.ct * d)?)
        } else {
            None
        };
        let st = self.status(OP)?;
        self.check(&st, &xv, ST_IN)?;
        match &wv {
            LceWeight::F32(wv) => self.check_operand(&st, w, wv, ST_IN)?,
            LceWeight::Half(wh) => self.check_half_operand(&st, w, wh, ST_IN)?,
        }
        let one_col_tile = g.ct >= v;
        for row0 in (0..n).step_by(g.rt) {
            let rr = g.rt.min(n - row0);
            let rr_u = u32_of(rr)?;
            let x_t = self.rows_operand(&xv, row0, rr, d, stage_x.as_ref())?;
            let t_rows = sub(&tv, row0, rr);
            for pass in 0..if g.grads.any() { 2 } else { 1 } {
                for (ci, col0) in (0..v).step_by(g.ct).enumerate() {
                    let cc = g.ct.min(v - col0);
                    let (cc_u, col0_u) = (u32_of(cc)?, u32_of(col0)?);
                    let w_t = match &wv {
                        LceWeight::F32(wv) => {
                            self.rows_operand(wv, col0, cc, d, stage_w.as_ref())?
                        }
                        // Nothing else writes the stage, so with one
                        // vocabulary tile the first widening serves every
                        // row tile and both passes.
                        LceWeight::Half(wh) => {
                            let stage = stage_w
                                .as_ref()
                                .ok_or_else(|| metal_err("unstaged bf16 weight tile"))?;
                            let dst = sub(stage, 0, cc * d);
                            if !one_col_tile || (pass == 0 && row0 == 0) {
                                self.widen_into(wh, col0 * d, &dst)?;
                            }
                            self.tt(&dst, &[cc, d])?
                        }
                    };
                    let tile_t = self.tt(&tile, &[rr, cc])?;
                    // With one vocabulary tile, pass 1 left this row tile's
                    // logits in `tile` (the stats kernel only reads it), so
                    // pass 2 starts from them instead of a second GEMM.
                    if pass == 0 || !one_col_tile {
                        GemmOperands::ExactF32
                            .nt(&x_t, &w_t, &tile_t)
                            .map_err(metal_err)?;
                    }
                    if pass == 0 {
                        self.ktg("ojas_lce_stats", rr, 1, 256, |b| {
                            bind(b, &tile, 0);
                            bind(b, &t_rows, 1);
                            bind_st(b, &st, 2);
                            bind(b, &stats, 3);
                            set_u32(b, rr_u, 4);
                            set_u32(b, cc_u, 5);
                            set_u32(b, col0_u, 6);
                            set_u32(b, u32::from(ci == 0), 7);
                        })?;
                        continue;
                    }
                    self.k2("ojas_lce_grad", cc, rr, |b| {
                        bind(b, &tile, 0);
                        bind(b, &t_rows, 1);
                        bind_st(b, &st, 2);
                        bind(b, &stats, 3);
                        set_u32(b, rr_u, 4);
                        set_u32(b, cc_u, 5);
                        set_u32(b, col0_u, 6);
                        set_u32(b, vocab, 7);
                        set_u32(b, has_ignore, 8);
                        set_u32(b, ignore_id, 9);
                        set_u32(b, valid, 10);
                    })?;
                    if let (Some(gx), Some(gx_tmp)) = (&gx, &gx_tmp) {
                        let gx_rows = sub(gx_tmp, 0, rr * d);
                        GemmOperands::ExactF32
                            .nn(&tile_t, &w_t, &self.tt(&gx_rows, &[rr, d])?)
                            .map_err(metal_err)?;
                        self.add_into(&gx_rows, &sub(gx, row0 * d, rr * d), ci == 0)?;
                    }
                    if let (Some(gw), Some(gw_tmp)) = (&gw, &gw_tmp) {
                        let gw_rows = sub(gw_tmp, 0, cc * d);
                        GemmOperands::ExactF32
                            .tn(&tile_t, &x_t, &self.tt(&gw_rows, &[cc, d])?)
                            .map_err(metal_err)?;
                        self.add_into(&gw_rows, &sub(gw, col0 * d, cc * d), row0 == 0)?;
                    }
                }
                if pass == 0 {
                    let loss_out = sub(&loss_rows, row0, rr);
                    self.k1("ojas_lce_loss", rr, |b| {
                        bind(b, &stats, 0);
                        bind(b, &t_rows, 1);
                        bind(b, &loss_out, 2);
                        set_u32(b, rr_u, 3);
                        set_u32(b, vocab, 4);
                        set_u32(b, has_ignore, 5);
                        set_u32(b, ignore_id, 6);
                    })?;
                }
            }
        }
        self.reduce(&loss_rows, 0, None, &sum.buf, 0)?;
        self.k1("ojas_ce_mean", 1, |b| {
            bind(b, &sum, 0);
            set_u32(b, valid, 1);
            bind(b, &loss, 2);
        })?;
        self.check(&st, &loss, ST_OUT)?;
        // The order `MetalBackend::linear_cross_entropy_mean` unpacks: the
        // loss, then each gradient asked for, the input's first.
        let mut outs = vec![loss];
        for grad in [gx, gw].into_iter().flatten() {
            self.check(&st, &grad, ST_OUT)?;
            outs.push(grad);
        }
        Ok(self.keep(outs))
    }

    /// One threadgroup of [`CA_THREADS`] per (query, head) and batch; the
    /// finite checks are inside `ojas_cached_attn` (see its comment).
    fn cached_attn(&mut self, q: Arg, k: Arg, v: Arg, dims: [u32; 8]) -> Res<Reply> {
        const OP: &str = "cached_attention_forward";
        let [batch, tq, heads, kv_heads, d, cap, kv_len, window] = dims;
        if tq == 0 || tq > kv_len {
            return Err(metal_err(format!(
                "{OP}: {tq} queries over kv_len {kv_len}"
            )));
        }
        let origin = cached_attn_origin(tq, kv_len, window);
        let span = (kv_len - origin) as usize;
        if d > ATTN_MAX_HEAD_DIM {
            return Err(OjasError::UnsupportedHeadDim {
                head_dim: d,
                limit: ATTN_MAX_HEAD_DIM,
            });
        }
        let scale = sdpa_scale(d)?;
        let (qv, kv, vv) = (self.view(q)?, self.view(k)?, self.view(v)?);
        let out = self.fresh(qv.n)?;
        let st = self.status(OP)?;
        let groups = tq as usize * heads as usize;
        let rows = groups * batch as usize;
        let splits = cached_attn_splits(rows, span);
        let chunk = u32_of(span.div_ceil(splits))?;
        // Splits > 1 write (max, sum, output) per split here for the merge.
        let part = if splits > 1 {
            let n = cached_attn_part(rows, span, d as usize)
                .ok_or_else(|| metal_err(format!("{OP}: split scratch overflows")))?;
            self.fresh(n)?
        } else {
            out.clone()
        };
        let splits = u32_of(splits)?;
        self.ktg(
            "ojas_cached_attn",
            groups * splits as usize,
            batch as usize,
            CA_THREADS,
            |b| {
                bind(b, &qv, 0);
                bind(b, &kv, 1);
                bind(b, &vv, 2);
                bind(b, &out, 3);
                bind_st(b, &st, 4);
                set_u32(b, tq, 5);
                set_u32(b, heads, 6);
                set_u32(b, kv_heads, 7);
                set_u32(b, d, 8);
                set_u32(b, cap, 9);
                set_u32(b, kv_len, 10);
                set_f32(b, scale, 11);
                set_u32(b, splits, 12);
                set_u32(b, chunk, 13);
                bind(b, &part, 14);
                set_u32(b, window, 15);
                set_u32(b, origin, 16);
            },
        )?;
        if splits > 1 {
            self.ktg(
                "ojas_cached_attn_merge",
                groups,
                batch as usize,
                d as usize,
                |b| {
                    bind(b, &part, 0);
                    bind(b, &out, 1);
                    bind_st(b, &st, 2);
                    set_u32(b, d, 3);
                    set_u32(b, splits, 4);
                },
            )?;
        }
        Ok(self.keep(vec![out]))
    }

    /// Check `src`, then `ojas_kv_write` copies it in only if it was finite.
    #[allow(clippy::too_many_arguments)]
    fn kv_write(
        &mut self,
        cache: Arg,
        src: Arg,
        batch: u32,
        tn: u32,
        cap: u32,
        row: u32,
        at: u32,
    ) -> Res<Reply> {
        const OP: &str = "kv_cache_write";
        self.writable(OP, cache)?;
        let (cv, sv) = (self.view(cache)?, self.view(src)?);
        let span = u32_of(tn as usize * row as usize)?;
        let cap_span = u32_of(cap as usize * row as usize)?;
        if tn > cap || cap == 0 {
            return Err(metal_err(format!(
                "{OP}: {tn} positions over a {cap}-slot ring"
            )));
        }
        let at_slot = at % cap;
        if sv.n != batch as usize * span as usize || cv.n != batch as usize * cap_span as usize {
            return Err(metal_err(format!(
                "{OP}: windows do not match the dimensions"
            )));
        }
        let n = u32_of(sv.n)?;
        let st = self.status(OP)?;
        self.check(&st, &sv, ST_IN)?;
        self.k1("ojas_kv_write", sv.n, |b| {
            bind(b, &sv, 0);
            bind(b, &cv, 1);
            bind_st(b, &st, 2);
            set_u32(b, n, 3);
            set_u32(b, span, 4);
            set_u32(b, cap_span, 5);
            set_u32(b, at_slot, 6);
            set_u32(b, row, 7);
            set_u32(b, cap, 8);
        })?;
        Ok(Reply::Done)
    }

    /// `ojas_acc_check` decides (inputs and every sum finite), then
    /// `ojas_acc_apply` adds in place only if no status word is set. A shared
    /// accumulator is first copied into a new buffer, which takes the add.
    fn accumulate(&mut self, acc: Arg, g: Arg, in_place: bool) -> Res<Reply> {
        const OP: &str = "accumulate_grad";
        if in_place {
            self.writable(OP, acc)?;
        }
        let (av, gv) = (self.view(acc)?, self.view(g)?);
        let n = u32_of(av.n)?;
        let target = if in_place {
            av
        } else {
            let out = self.fresh(av.n)?;
            self.copy(&av, &out)?;
            out
        };
        let st = self.status(OP)?;
        for name in ["ojas_acc_check", "ojas_acc_apply"] {
            self.k1(name, target.n, |b| {
                bind(b, &target, 0);
                bind(b, &gv, 1);
                bind_st(b, &st, 2);
                set_u32(b, n, 3);
            })?;
        }
        Ok(if in_place {
            Reply::Done
        } else {
            self.keep(vec![target])
        })
    }

    /// `ojas_adamw_check` decides the whole step (inputs and every new p, m,
    /// v finite), then `ojas_adamw_apply`, in the same command buffer, writes
    /// in place only if this call's status words are clear. No wait, no copies.
    fn adamw(&mut self, p: Arg, g: Arg, m: Arg, v: Arg, scalars: &[f32; 7]) -> Res<Reply> {
        const OP: &str = "adamw_step";
        for written in [p, m, v] {
            self.writable(OP, written)?;
        }
        let (pv, gv, mv, vv) = (self.view(p)?, self.view(g)?, self.view(m)?, self.view(v)?);
        let n = pv.n;
        let n_u = u32_of(n)?;
        let bytes: Vec<u8> = scalars.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let st = self.status(OP)?;
        for name in ["ojas_adamw_check", "ojas_adamw_apply"] {
            self.k1(name, n, |b| {
                bind(b, &pv, 0);
                bind(b, &gv, 1);
                bind(b, &mv, 2);
                bind(b, &vv, 3);
                bind_st(b, &st, 4);
                set_u32(b, n_u, 5);
                b.bind_bytes(&bytes, 6);
            })?;
        }
        Ok(Reply::Done)
    }

    /// Momentum, Nesterov, five Newton-Schulz iterations with tessl GEMMs,
    /// then the update. Written back only if every stage is finite.
    ///
    /// With `bf16` (`Ns5Precision::Bf16`) the iterate holds bf16 values in
    /// f32 buffers and every stage rounds its output as the CPU reference
    /// does: the cast, the norm and `norm + eps`, the division, each GEMM
    /// output and each `axpby` (`ojas_axpby_bf16`). The GEMMs take tessl's
    /// bf16 operand lane (f32 accumulation) when the device has TensorOps,
    /// and exact f32 otherwise; on bf16 operands the two compute the same
    /// products.
    #[allow(clippy::too_many_arguments)]
    fn muon(
        &mut self,
        p: Arg,
        g: Arg,
        m: Arg,
        rows: usize,
        cols: usize,
        momentum: f32,
        nesterov: bool,
        decay: f32,
        alpha: f32,
        bf16: bool,
    ) -> Res<Reply> {
        const OP: &str = "muon_ns5_step";
        for written in [p, m] {
            self.writable(OP, written)?;
        }
        let (pv, gv, mv) = (self.view(p)?, self.view(g)?, self.view(m)?);
        let n = pv.n;
        let transposed = rows > cols;
        let (r, c) = if transposed {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let st = self.status(OP)?;
        for x in [&pv, &gv, &mv] {
            self.check(&st, x, ST_IN)?;
        }
        let buf = self.fresh(n)?;
        self.axpby("ojas_axpby", &mv, &gv, &buf, momentum, 1.0)?;
        self.check(&st, &buf, ST_OUT)?;
        let update = if nesterov {
            let u = self.fresh(n)?;
            self.axpby("ojas_axpby_fma", &gv, &buf, &u, 1.0, momentum)?;
            u
        } else {
            buf.clone()
        };
        let x = self.fresh(n)?;
        // Rounds a value to bf16 in place under bf16; nothing otherwise.
        let round = |v: &V| {
            if bf16 {
                self.round_bf16_into(v, v)
            } else {
                Ok(())
            }
        };
        if transposed {
            transpose_f32_into(&self.tt(&update, &[rows, cols])?, &self.tt(&x, &[r, c])?)
                .map_err(metal_err)?;
        } else {
            self.copy(&update, &x)?;
        }
        round(&x)?;
        let stats = self.fresh(3)?;
        self.reduce(&x, 1, None, &stats.buf, 0)?;
        self.reduce(&x, 2, Some((&stats.buf, 0)), &stats.buf, 1)?;
        let denom = if bf16 {
            "ojas_ns_denom_bf16"
        } else {
            "ojas_ns_denom"
        };
        self.k1(denom, 1, |b| {
            bind(b, &stats, 0);
            bind_st(b, &st, 1);
            set_f32(b, MUON_NS_EPS as f32, 2);
        })?;
        let n_u = u32_of(n)?;
        self.k1("ojas_div_scalar", n, |b| {
            bind(b, &x, 0);
            bind(b, &x, 1);
            set_u32(b, n_u, 2);
            bind(b, &stats, 3);
            set_u32(b, 2, 4);
        })?;
        round(&x)?;
        let lane = self.gemm_lane(if bf16 { Lane::Bf16 } else { Lane::ExactF32 });
        let a = self.fresh(r * r)?;
        let a2 = self.fresh(r * r)?;
        let bm = self.fresh(r * r)?;
        let bx = self.fresh(n)?;
        let x_t = self.tt(&x, &[r, c])?;
        let a_t = self.tt(&a, &[r, r])?;
        let a2_t = self.tt(&a2, &[r, r])?;
        let bm_t = self.tt(&bm, &[r, r])?;
        let bx_t = self.tt(&bx, &[r, c])?;
        // Each GEMM output is rounded under bf16 before the combination
        // reads it: B = r(r(b A) + r(c r(A A))), X = r(r(a X) + r(B X)); the
        // 1.0 multiple of a bf16 value is exact.
        let comb = if bf16 {
            "ojas_axpby_bf16"
        } else {
            "ojas_axpby"
        };
        for _ in 0..5 {
            lane.nt(&x_t, &x_t, &a_t).map_err(metal_err)?;
            round(&a)?;
            lane.nn(&a_t, &a_t, &a2_t).map_err(metal_err)?;
            round(&a2)?;
            self.axpby(comb, &a, &a2, &bm, MUON_NS5_B as f32, MUON_NS5_C as f32)?;
            lane.nn(&bm_t, &x_t, &bx_t).map_err(metal_err)?;
            round(&bx)?;
            self.axpby(comb, &x, &bx, &x, MUON_NS5_A as f32, 1.0)?;
            self.check(&st, &x, ST_OUT)?;
        }
        let ortho = if transposed {
            let o = self.fresh(n)?;
            transpose_f32_into(&x_t, &self.tt(&o, &[rows, cols])?).map_err(metal_err)?;
            o
        } else {
            x
        };
        let new_p = self.fresh(n)?;
        self.axpby("ojas_axpby_fma", &pv, &ortho, &new_p, decay, alpha)?;
        self.check(&st, &new_p, ST_OUT)?;
        // Committed on the device only if every check above passed, so the
        // step needs no wait, neither before the write-back nor
        // after it.
        self.copy_if_clean(&st, &new_p, &pv)?;
        self.copy_if_clean(&st, &buf, &mv)?;
        Ok(Reply::Done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ojas_core::{Backend, Budget, Tensor};

    /// Every public op indexes elements with 32 bits, so no op on the
    /// backend can ask for more than the device's buffer limit (about
    /// 36 GiB here); the worker's own allocation is the lowest layer that
    /// can, and nothing is resident when tessl refuses it.
    fn oversize_worker() -> (Worker, Arc<Waits>) {
        let waits = Arc::new(Waits::default());
        let w = Worker::open(Arc::clone(&waits), 1 << 44).expect("Metal device");
        (w, waits)
    }

    /// A request past the device's limit, through tessl's real allocator
    /// (nothing injected), is `CapacityExceeded` at once: no waited commit
    /// and no second try, whichever of tessl's permanent refusals it is.
    #[test]
    fn a_real_oversize_allocation_is_capacity_exceeded_without_a_retry() {
        let (w, waits) = oversize_worker();
        let before = waits.snapshot();
        for bytes in [1usize << 50, 1 << 46, isize::MAX as usize, usize::MAX] {
            let r = w.alloc(bytes);
            assert!(
                matches!(&r, Err(OjasError::CapacityExceeded { requested, .. }) if *requested == bytes as u64),
                "{bytes}: {:?}",
                r.map(|_| ())
            );
        }
        assert_eq!(
            waits.snapshot(),
            before,
            "a permanent refusal is not retried behind a commit"
        );
        assert_eq!(
            w.book.borrow().alloc_since_wait,
            0,
            "a refused allocation is not counted"
        );
    }

    /// gdn backward's scratch goes through the same classification: a
    /// workspace past the device limit (tessl sizes it from the dims alone,
    /// so no operand exists) is `CapacityExceeded`, and one that tessl
    /// rejects as malformed stays `Backend`.
    #[test]
    fn a_real_oversize_gdn_workspace_is_capacity_exceeded() {
        let (w, waits) = oversize_worker();
        let before = waits.snapshot();
        // heads 64 at this seq is ~516 GiB, under the 1 TiB bar, so the
        // classification below never ran. heads 256 is ~2 TiB and still
        // inside tessl's shape check.
        let huge = GdnTrainDims {
            batch: 1,
            seq: 1 << 20,
            heads: 256,
            v_dim: 128,
        };
        assert!(GdnTrainWorkspace::bytes_for(huge) > 1 << 40);
        let r = w.gdn_workspace(huge);
        assert!(
            matches!(&r, Err(OjasError::CapacityExceeded { .. })),
            "{:?}",
            r.map(|_| ())
        );
        assert_eq!(waits.snapshot(), before);
        let bad = GdnTrainDims {
            batch: 1,
            seq: 1,
            heads: 1,
            v_dim: 7,
        };
        let r = w.gdn_workspace(bad);
        assert!(
            matches!(&r, Err(OjasError::Backend { .. })),
            "{:?}",
            r.map(|_| ())
        );
    }

    /// The matcher against tessl's three real allocation failures and
    /// against ones that are not allocation failures.
    #[test]
    fn classify_alloc_reads_tessls_failures() {
        let rt = GpuRuntime::new().expect("Metal device");
        let real = rt
            .alloc_buffer(usize::MAX)
            .map(|_| ())
            .expect_err("past any device");
        assert_eq!(classify_alloc(&real), AllocFailure::Limit, "{real}");
        for e in [
            "rounded buffer size exceeds device limit",
            "buffer request 9 exceeds host/device allocation limit",
        ] {
            assert_eq!(classify_alloc(e), AllocFailure::Limit, "{e}");
        }
        assert_eq!(
            classify_alloc("newBuffer(4096) failed"),
            AllocFailure::Exhausted
        );
        for e in [
            "",
            "newBuffer failed",
            "pipeline 'x' not found",
            "poisoned: newBuffer(1) failed later",
        ] {
            assert_eq!(classify_alloc(e), AllocFailure::Other, "{e:?}");
        }
    }

    /// Where tessl cannot be made to fail the transient way without
    /// exhausting this machine, pin the exact text it builds: the source of
    /// the sibling checkout must still contain each fragment
    /// `classify_alloc` reads. A tessl change that moves one fails here
    /// until tessl exposes a typed error and this function is deleted.
    #[test]
    fn classify_alloc_matches_the_text_tessl_builds() {
        let src = include_str!("../../../tessl/src/runtime.rs");
        for fragment in [
            r#"format!("newBuffer({key}) failed")"#,
            r#""rounded buffer size exceeds device limit""#,
            "exceeds host/device allocation limit",
        ] {
            assert!(src.contains(fragment), "tessl no longer builds {fragment}");
        }
    }

    /// A device refusal is retried behind a waited commit, then after the
    /// pool is trimmed, and is `CapacityExceeded` only if both fail; one
    /// that is not an allocation failure is neither retried nor capacity.
    #[test]
    fn a_device_refusal_takes_both_retries_and_other_errors_none() {
        let (w, waits) = oversize_worker();
        let mut calls = 0;
        let r: Res<()> = w.retrying(8, || {
            calls += 1;
            Err("newBuffer(8) failed".to_string())
        });
        assert!(
            matches!(r, Err(OjasError::CapacityExceeded { requested: 8, .. })),
            "{r:?}"
        );
        assert_eq!(calls, 3);
        calls = 0;
        let r = w.retrying(8, || {
            calls += 1;
            if calls < 3 {
                Err("newBuffer(8) failed".to_string())
            } else {
                Ok(calls)
            }
        });
        assert!(matches!(r, Ok(3)), "{r:?}");
        calls = 0;
        let r: Res<()> = w.retrying(8, || {
            calls += 1;
            Err("pipeline 'x' not found".to_string())
        });
        assert!(matches!(r, Err(OjasError::Backend { .. })), "{r:?}");
        assert_eq!(calls, 1);
        let _ = waits;
    }

    /// The pool cache is a quarter of the budget, at most 1 GiB, never the
    /// 2 GiB tessl default whatever the budget.
    #[test]
    fn the_pool_cache_cap_follows_the_budget() {
        for (budget, want) in [
            (0u64, 0u64),
            (3, 0),
            (1 << 20, 1 << 18),
            (1 << 30, 1 << 28),
            (4 << 30, 1 << 30),
            (64 << 30, 1 << 30),
            (u64::MAX, 1 << 30),
        ] {
            assert_eq!(pool_cache_cap(budget), want, "budget {budget}");
        }
    }

    /// The bound `backend.rs` documents for what the budget does not
    /// charge: a `Cold` buffer of up to 1 MiB is made at most 512 KiB
    /// larger than asked, and never more than its own size larger past 256
    /// bytes; a larger one at most 16 KiB larger. Read from tessl's own
    /// rounding, so a tessl change that widens it fails here.
    #[test]
    fn cold_rounding_stays_inside_the_documented_bound() {
        use tessl::BufferKind;
        let mut sizes: Vec<usize> = (1..=4096).step_by(3).collect();
        for shift in 8..=34 {
            let p = 1usize << shift;
            sizes.extend([p - 4, p, p + 4, p + p / 2 + 4]);
        }
        for n in sizes {
            let made = GpuRuntime::allocated_bytes_for(n, BufferKind::Cold);
            let extra = made - n as u64;
            if n <= 1 << 20 {
                assert!(extra < 512 << 10, "{n} -> {made}");
                assert!(n <= 256 || extra < n as u64, "{n} -> {made}");
            } else {
                assert!(extra < 16 << 10, "{n} -> {made}");
            }
        }
    }

    /// One decode request (12 heads, 1024 keys) splits; a batch that already
    /// fills the GPU, a short cache, and degenerate inputs do not.
    #[test]
    fn cached_attn_splits_only_where_the_gpu_would_sit_idle() {
        for (rows, kv_len, want) in [
            (12usize, 1024usize, 8usize),
            (24, 1024, 4),
            (1, 4097, 64),
            (12, 129, 2),
            (12, 127, 1),
            (12, 63, 1),
            (96, 1024, 1),
            (192, 1024, 1),
            (0, 1024, 16),
            (12, 0, 1),
        ] {
            assert_eq!(
                cached_attn_splits(rows, kv_len),
                want,
                "rows {rows}, kv_len {kv_len}"
            );
        }
    }

    fn pattern(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    fn upload(w: &mut Worker, v: &[f32]) -> Arg {
        let src =
            ojas_core::Tensor::from_f32(v, &[v.len()], &ojas_core::Budget::new(1 << 30)).unwrap();
        match w.run(Cmd::Upload { src }) {
            Ok(Reply::Bufs(b)) if b.len() == 1 => Arg {
                id: b[0].id,
                off: 0,
                n: v.len(),
            },
            other => panic!("upload: {other:?}"),
        }
    }

    fn read(w: &mut Worker, id: u64, n: usize) -> Vec<f32> {
        match w.run(Cmd::Read {
            id,
            off: 0,
            len: n * 4,
        }) {
            Ok(Reply::Bytes(b)) => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes(*c))
                .collect(),
            other => panic!("read: {other:?}"),
        }
    }

    /// Planes past i32 extents are refused before anything is allocated
    /// (too large to build here, so the bound is tested directly).
    #[test]
    fn attention_planes_past_i32_extents_are_unsupported() {
        const OP: &str = "causal_sdpa_forward";
        assert!(attn_plane_fits(OP, 1 << 23, 128).is_ok());
        assert!(attn_plane_fits(OP, (1 << 24) - 1, 128).is_ok());
        assert!(attn_plane_fits(OP, i32::MAX as u32, 1).is_ok());
        for (t, d) in [
            (1u32 << 24, 128u32),
            (u32::MAX, 1),
            (1 << 31, 1),
            (u32::MAX, 128),
        ] {
            assert!(
                matches!(
                    attn_plane_fits(OP, t, d),
                    Err(OjasError::Unsupported { op: OP, .. })
                ),
                "t {t} d {d}"
            );
        }
    }

    /// The tiled backward at head dims through 256. This drives the device
    /// thread's command directly and compares with the CPU reference at the
    /// parity tests' tolerance. One past the compiled width is refused.
    #[test]
    fn tiled_backward_matches_cpu_at_head_dims_up_to_256() {
        let mut w = Worker::open(Arc::new(Waits::default()), 8 << 30).expect("Metal device");
        let cpu = ojas_cpu::CpuBackend::new(Budget::new(8 << 30))
            .with_numerics(ojas_core::Numerics::Exact);
        for (bh, t, d) in [
            (2usize, 67usize, 128usize),
            (1, 300, 128),
            (3, 33, 100),
            (1, 17, 256),
            (2, 9, 200),
        ] {
            let n = bh * t * d;
            let shape = [1, bh, t, d];
            let host: Vec<Vec<f32>> = (0..4).map(|i| pattern(n, (t * d) as u64 + i)).collect();
            let ht: Vec<Tensor> = host
                .iter()
                .map(|v| Tensor::from_f32(v, &shape, &Budget::new(1 << 30)).expect("host"))
                .collect();
            let (wq, wk, wv) = cpu
                .causal_sdpa_backward_recompute(&ht[0], &ht[1], &ht[2], &ht[3], None)
                .expect("cpu");
            let args: Vec<Arg> = host.iter().map(|v| upload(&mut w, v)).collect();
            let geom = SdpaGeom {
                bh: bh as u32,
                t: t as u32,
                d: d as u32,
                rep: 1,
                window: 0,
            };
            let saved = match w.run(Cmd::Sdpa {
                q: args[0],
                k: args[1],
                v: args[2],
                geom,
            }) {
                Ok(Reply::Bufs(b)) if b.len() == 2 => b,
                other => panic!("sdpa fwd d{d}: {other:?}"),
            };
            let whole = |id: u64, n: usize| Arg { id, off: 0, n };
            let bufs = match w.run(Cmd::SdpaBwd {
                args: [
                    args[0],
                    args[1],
                    args[2],
                    whole(saved[0].id, n),
                    whole(saved[1].id, bh * t),
                    args[3],
                ],
                geom,
            }) {
                Ok(Reply::Bufs(b)) if b.len() == 3 => b,
                other => panic!("sdpa bwd d{d}: {other:?}"),
            };
            let atol = 2e-5 * (t as f32).sqrt();
            for (buf, want, what) in [
                (&bufs[0], &wq, "dq"),
                (&bufs[1], &wk, "dk"),
                (&bufs[2], &wv, "dv"),
            ] {
                let got = read(&mut w, buf.id, n);
                let want = want.to_f32_vec().expect("cpu values");
                for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - e).abs() <= atol + 1e-3 * e.abs(),
                        "bh{bh} t{t} d{d} {what}[{i}]: got {g} want {e}"
                    );
                }
            }
        }
        // One past the compiled width, both directions refuse and name it.
        let over = ATTN_MAX_HEAD_DIM + 1;
        let x = upload(&mut w, &pattern(4 * over as usize, 1));
        let geom = SdpaGeom {
            bh: 1,
            t: 4,
            d: over,
            rep: 1,
            window: 0,
        };
        let fwd = w.run(Cmd::Sdpa {
            q: x,
            k: x,
            v: x,
            geom,
        });
        let bwd = w.run(Cmd::SdpaBwd { args: [x; 6], geom });
        for r in [fwd, bwd] {
            assert!(
                matches!(
                    r,
                    Err(OjasError::UnsupportedHeadDim { head_dim, limit })
                        if head_dim == over && limit == ATTN_MAX_HEAD_DIM
                ),
                "{r:?}"
            );
        }
    }

    /// The forward at head dims through 256, against the CPU reference at
    /// the parity tests' tolerance.
    #[test]
    fn forward_matches_cpu_at_head_dims_up_to_256() {
        let mut w = Worker::open(Arc::new(Waits::default()), 8 << 30).expect("Metal device");
        let cpu = ojas_cpu::CpuBackend::new(Budget::new(8 << 30))
            .with_numerics(ojas_core::Numerics::Exact);
        for (bh, t, d) in [
            (2usize, 67usize, 128usize),
            (1, 300, 128),
            (3, 33, 100),
            (1, 17, 256),
            (2, 9, 200),
        ] {
            let n = bh * t * d;
            let shape = [1, bh, t, d];
            let host: Vec<Vec<f32>> = (0..3).map(|i| pattern(n, (t * d) as u64 + 7 + i)).collect();
            let ht: Vec<Tensor> = host
                .iter()
                .map(|v| Tensor::from_f32(v, &shape, &Budget::new(1 << 30)).expect("host"))
                .collect();
            let want = cpu
                .causal_sdpa_forward(&ht[0], &ht[1], &ht[2], None)
                .map(|(y, _)| y)
                .expect("cpu")
                .to_f32_vec()
                .expect("cpu values");
            let args: Vec<Arg> = host.iter().map(|v| upload(&mut w, v)).collect();
            let out = match w.run(Cmd::Sdpa {
                q: args[0],
                k: args[1],
                v: args[2],
                geom: SdpaGeom {
                    bh: bh as u32,
                    t: t as u32,
                    d: d as u32,
                    rep: 1,
                    window: 0,
                },
            }) {
                Ok(Reply::Bufs(b)) if b.len() == 2 => b,
                other => panic!("sdpa fwd d{d}: {other:?}"),
            };
            let got = read(&mut w, out[0].id, n);
            let atol = 2e-5 * (t as f32).sqrt();
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - e).abs() <= atol + 1e-4 * e.abs(),
                    "bh{bh} t{t} d{d} o[{i}]: got {g} want {e}"
                );
            }
        }
    }

    const CHILD_ENV: &str = "OJAS_METAL_REFUSED_OPEN_CHILD";
    const SELF_NAME: &str = "device::tests::refused_opens_do_not_accumulate_owner_threads";

    /// Continuation lines from `ps -M` are threads beyond the process line.
    fn extra_threads() -> usize {
        let out = std::process::Command::new("/bin/ps")
            .args(["-M", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps -M");
        assert!(
            out.status.success(),
            "ps -M failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|line| line.starts_with(' ') || line.starts_with('\t'))
            .count()
    }

    /// A Metal load that gets as far as the device thread and then fails must
    /// not leave that thread behind. On this Mac a real load does not refuse
    /// after a probe, so this injects the open failure the owner thread sees.
    #[test]
    fn refused_opens_do_not_accumulate_owner_threads() {
        if std::env::var_os(CHILD_ENV).is_some() {
            let before = extra_threads();
            for i in 0..8 {
                let err = if i == 3 {
                    spawn_with(|| panic!("injected open panic")).expect_err("panic opener")
                } else {
                    spawn_with(|| Err(metal_err("injected open failure")))
                        .expect_err("failed opener")
                };
                let msg = err.to_string();
                assert!(
                    msg.contains("injected") || msg.contains("panicked"),
                    "{msg}"
                );
            }
            let after = extra_threads();
            assert!(
                after <= before,
                "owner threads grew across 8 refused opens: {before} extra -> {after} extra"
            );
            return;
        }
        let exe = std::env::current_exe().expect("test binary");
        let out = std::process::Command::new(exe)
            .args(["--exact", "--test-threads=1", SELF_NAME])
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn isolated child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "refused-open child failed:\n{stdout}\n{stderr}"
        );
        assert!(
            stdout.contains("1 passed") || stderr.contains("1 passed"),
            "child ran no test:\n{stdout}\n{stderr}"
        );
    }
}

#[cfg(test)]
#[path = "adversarial_oom.rs"]
mod adversarial_oom;
