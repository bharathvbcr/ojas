//! One adapter, one device, explicit pipeline layouts, a bounded buffer pool,
//! and a command recorder that batches dispatches until the host needs bytes.
//!
//! Nothing here waits on the GPU except [`WgpuContext::read`] and
//! [`WgpuContext::sync`]. A [`Job`] collects the dispatches, clears and copies
//! of one op and records them into the shared encoder only when the whole op
//! validated, so a refused op leaves nothing behind. The encoder is submitted
//! every [`FLUSH_AT`] dispatches and at every read.
//!
//! Kernels that produce a non-finite value OR their op's bit into a 64-bit
//! op mask held in two device words ([`FAULT_OPS`] ops) and keep going; the
//! first op to fault also records itself in a third word. Every read runs
//! `fault_hold`, which moves those words into a second, held set, and
//! copies the held set out in the same submission. Only
//! after the host has seen the copy does it record `fault_release`, which
//! clears exactly what it saw. A read that fails between the two (a poll
//! timeout, a failed map) therefore leaves the fault held for the next read
//! instead of losing it. The bits collect in [`WgpuContext::take_faults`].
//! Two reads on different threads can both copy the same held bits before
//! either release lands, so a fault may be reported twice; it is never lost.
//!
//! A lost device is recorded apart from the capped uncaptured-error queue,
//! every later error check reports it, and every error from a device call
//! names it.
//!
//! Uploads write through a buffer mapped at creation. A pooled buffer is
//! only ever written by commands recorded after the ones that read its
//! previous contents, so reuse needs no fence.
//!
//! A dispatch's sixteen parameter words go to a slot of the context's
//! parameter ring, bound at a dynamic offset. The host fills the slots of
//! the open encoder and writes them with one `Queue::write_buffer` just
//! before that encoder is submitted; the queue orders that write after
//! every earlier submission's reads of the ring and before this one's. A
//! job that would pass [`PARAM_SLOTS`] submits what is recorded first.
//! Bind groups are cached by kernel and buffer identity
//! ([`BIND_CACHE_CAP`]), and `MAP_READ` staging buffers are kept for the
//! next read ([`STAGING_CACHE_BYTES`]), so a steady-state dispatch creates
//! no wgpu object and pops no error scope. [`CacheStats`] counts what is
//! still created and every blocking error-scope pop.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ojas_core::{BackendId, Budget, DeviceBuffer, OjasError, Reservation};
use ojas_device::{Device, DeviceError};
use ojas_kernels::{wgsl_module, WgslModule};

use crate::{gpu_error, hal_backends, info_of};

/// Dispatches recorded before the encoder is submitted without a read.
pub const FLUSH_AT: usize = 64;

/// Bytes of one dispatch's parameter words: sixteen `u32`.
const PARAM_BYTES: u64 = 64;

/// Dispatches whose parameter words one submission's ring holds. A job that
/// would pass it submits what is already recorded and goes on in a new
/// encoder.
pub const PARAM_SLOTS: usize = 1024;

/// Bind groups the cache keeps. At the cap the cache is emptied and refills
/// from the next dispatches.
pub const BIND_CACHE_CAP: usize = 4096;

/// Bytes of idle `MAP_READ` staging buffers kept for later reads. A read
/// larger than this uses a buffer of its own and drops it.
pub const STAGING_CACHE_BYTES: u64 = 64 << 20;

/// Smallest staging buffer: reads of up to this many bytes share a size.
const STAGING_MIN_BYTES: u64 = 256;

/// Host bytes an upload encodes at a time into the device mapping.
const UPLOAD_PIECE_BYTES: usize = 64 << 10;

/// How long dropping the last handle on a context waits for its queue to go
/// idle before it lets a background thread finish the drop (see
/// [`WgpuContext::set_drop_wait`]).
pub const DROP_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// Bytes of freed buffers the pool keeps for reuse. Larger frees go back to
/// wgpu. Pooled memory is not live tensor memory and is not charged to a
/// [`Budget`]; [`WgpuContext::trim_pool`] releases it, and an allocation the
/// device refuses for want of memory releases it before its one retry. The
/// probe ([`WgpuContext::memory`]) reports this cap to the plan, which sets
/// it aside from the device's room.
pub const POOL_CAP_BYTES: u64 = 512 << 20;

/// Parked context drops allowed at once: drops that returned before the
/// GPU finished, so their thread still holds the queue, the device, that
/// device's memory and itself ([`Inner`]'s `Drop`). At this many,
/// [`WgpuContext::open`] refuses a new context rather than park another, so
/// what timed-out drops hold is bounded; [`drop_stats`] reports them. A drop
/// that finishes within its wait never parks and never counts.
pub const MAX_PARKED_DROPS: usize = 4;

/// Drops parked now: each is decremented when its thread has released
/// everything.
static DROPS_PARKED: AtomicUsize = AtomicUsize::new(0);
/// Drops that returned before the GPU finished, since the process started.
static DROPS_TIMED_OUT: AtomicU64 = AtomicU64::new(0);

/// A drop's thread is still releasing; its drop still waits.
const DROP_RUNNING: u8 = 0;
/// The drop returned first: the thread is parked and counted.
const DROP_PARKED: u8 = 1;
/// The thread has released everything.
const DROP_DONE: u8 = 2;

/// What context drops hold in this process ([`drop_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DropStats {
    /// Drops parked now, each holding a queue, a device, its memory and a
    /// thread until its GPU work finishes. Opens stop at
    /// [`MAX_PARKED_DROPS`]; contexts already open when that was reached can
    /// still park, so this is at most that plus the contexts alive then.
    pub parked: usize,
    /// Drops that returned before the GPU had finished, ever.
    pub timed_out: u64,
}

/// The process's parked context drops, and how many drops ever timed out.
pub fn drop_stats() -> DropStats {
    DropStats {
        parked: DROPS_PARKED.load(Ordering::Acquire),
        timed_out: DROPS_TIMED_OUT.load(Ordering::Relaxed),
    }
}

/// Whether a new context may open while `parked` drops still hold their
/// devices.
fn admit_open(parked: usize) -> Result<(), DeviceError> {
    if parked >= MAX_PARKED_DROPS {
        return Err(DeviceError::Capacity {
            kind: Device::Vulkan,
            detail: format!(
                "{parked} freed wgpu contexts are still waiting for their GPU work and \
                 each holds its device and memory (at most {MAX_PARKED_DROPS}); retry once \
                 that work finishes"
            ),
        });
    }
    Ok(())
}

/// The drop's side, once its wait ran out: count the drop as parked unless
/// its thread has already finished. The count goes up before the state
/// moves, so the thread's uncount ([`reap`]) can only follow it and the
/// count never wraps below zero; a drop whose thread won the race takes its
/// count back. Between the two steps the count may read one high, never low.
fn park(state: &AtomicU8, parked: &AtomicUsize) -> bool {
    parked.fetch_add(1, Ordering::AcqRel);
    let won = state
        .compare_exchange(
            DROP_RUNNING,
            DROP_PARKED,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok();
    if !won {
        parked.fetch_sub(1, Ordering::AcqRel);
    }
    won
}

/// The thread's side, once it has released everything: done, and uncounted
/// if its drop had parked it.
fn reap(state: &AtomicU8, parked: &AtomicUsize) {
    if state.swap(DROP_DONE, Ordering::AcqRel) == DROP_PARKED {
        parked.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Reaps a drop's thread when it ends, panicking or not.
struct Reaped(Arc<AtomicU8>);

impl Drop for Reaped {
    fn drop(&mut self) {
        reap(&self.0, &DROPS_PARKED);
    }
}

/// Distinct sizes the free list may remember. A byte cap alone still allows
/// one tiny buffer per size until the map holds millions of keys.
pub const POOL_MAX_KEYS: usize = 64;

/// Largest element count one binding may hold. Kernel index arithmetic is
/// `u32`, and the folded grid can overshoot the length by one row of groups.
pub const MAX_ELEMENTS: usize = (1 << 31) - 1;

/// Set to exactly `1` to let [`WgpuContext::open`] take a CPU adapter
/// (lavapipe, SwiftShader, WARP). For GPU-less CI only; unset, or any other
/// value, refuses one.
pub const ALLOW_CPU_ADAPTER_ENV: &str = "OJAS_WGPU_ALLOW_CPU_ADAPTER";

/// Storage usage of every tensor and scratch buffer.
pub(crate) fn storage_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

/// Ops the fault words can name: a 64-bit mask across words 0 and 1. A fault
/// id is an op index + 1, so ids run 1..=`FAULT_OPS`; 0 reports nothing.
pub(crate) const FAULT_OPS: u32 = 64;

const FAULT_HOLD: Kernel = Kernel {
    module: WgslModule::Fault,
    entry: "fault_hold",
    slots: &[Slot::W(2)],
};

const FAULT_RELEASE: Kernel = Kernel {
    module: WgslModule::Fault,
    entry: "fault_release",
    slots: &[Slot::W(2)],
};

/// One storage binding of a kernel, numbered from 2. `R` is read-only.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    R(u32),
    W(u32),
}

/// A WGSL entry point plus the storage bindings it uses, in bind order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Kernel {
    pub module: WgslModule,
    pub entry: &'static str,
    pub slots: &'static [Slot],
}

pub(crate) struct Pipe {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Counters since the context opened. `pool_hits` counts allocations served
/// from freed buffers; `alloc_retries` counts allocations the device
/// refused for want of memory and that were tried again after the pool was
/// released; `uploads` are host-to-device copies.
///
/// The fixed cost of a dispatch: `scope_pops` counts every blocking
/// error-scope pop (a buffer creation, a bind group the cache missed, a
/// shader or pipeline compile); `bind_groups` the bind groups created and
/// `bind_hits` the dispatches the cache served; `staging_creates` the
/// `MAP_READ` buffers created for reads; `offset_copies` the operands
/// compacted into scratch because their byte offset cannot be bound in
/// place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub compiles: u64,
    pub pool_hits: u64,
    pub alloc_retries: u64,
    pub submits: u64,
    pub dispatches: u64,
    pub uploads: u64,
    pub upload_bytes: u64,
    pub reads: u64,
    pub read_bytes: u64,
    pub scope_pops: u64,
    pub bind_groups: u64,
    pub bind_hits: u64,
    pub staging_creates: u64,
    pub offset_copies: u64,
}

#[derive(Default)]
struct Counters {
    compiles: AtomicU64,
    pool_hits: AtomicU64,
    alloc_retries: AtomicU64,
    submits: AtomicU64,
    dispatches: AtomicU64,
    uploads: AtomicU64,
    upload_bytes: AtomicU64,
    reads: AtomicU64,
    read_bytes: AtomicU64,
    scope_pops: AtomicU64,
    bind_groups: AtomicU64,
    bind_hits: AtomicU64,
    staging_creates: AtomicU64,
    offset_copies: AtomicU64,
}

#[derive(Default)]
struct Pool {
    free: HashMap<u64, Vec<wgpu::Buffer>>,
    bytes: u64,
}

/// Scratch memory held until its commands are submitted. The reservation
/// charges the budget for that long.
pub(crate) struct Scratch {
    buf: wgpu::Buffer,
    bytes: u64,
    _charge: Reservation,
}

#[derive(Default)]
struct Recorder {
    encoder: Option<wgpu::CommandEncoder>,
    scratch: Vec<Scratch>,
    dispatches: usize,
    /// Host copy of the parameter ring's slots for the open encoder, written
    /// to the device just before it is submitted.
    params: Vec<u8>,
    /// Slots of `params` the open encoder's dispatches use.
    slots: usize,
}

impl Recorder {
    /// The next parameter slot, holding `words`: its byte offset in the
    /// ring. The caller has checked that a slot is free.
    fn slot(&mut self, words: &[u32; 16], stride: u64) -> u32 {
        put_slot(&mut self.params, &mut self.slots, words, stride)
    }
}

/// Write `words` to slot `*slots` of the ring's host copy `params` and take
/// the slot: its byte offset. Free-standing so a caller can hold the
/// recorder's encoder while it fills slots.
fn put_slot(params: &mut Vec<u8>, slots: &mut usize, words: &[u32; 16], stride: u64) -> u32 {
    let at = *slots * stride as usize;
    if params.len() < at + PARAM_BYTES as usize {
        params.resize(PARAM_SLOTS * stride as usize, 0);
    }
    let (dst, _) = params[at..at + PARAM_BYTES as usize].as_chunks_mut::<4>();
    for (d, w) in dst.iter_mut().zip(words) {
        *d = w.to_le_bytes();
    }
    *slots += 1;
    at as u32
}

/// A storage binding: a whole buffer, or a range of one.
pub(crate) trait Binding {
    fn buffer(&self) -> &wgpu::Buffer;
    /// Byte offset and length of the bound range.
    fn range(&self) -> (u64, u64);
}

impl Binding for wgpu::Buffer {
    fn buffer(&self) -> &wgpu::Buffer {
        self
    }

    fn range(&self) -> (u64, u64) {
        (0, self.size())
    }
}

/// `size` bytes at byte `offset` of a buffer, bound in place: a tensor view
/// that does not start at byte 0.
#[derive(Clone, Debug)]
pub(crate) struct View {
    buf: wgpu::Buffer,
    offset: u64,
    size: u64,
}

impl View {
    /// `size` bytes at byte `offset` of `buf`; `offset` meets the device's
    /// storage-offset alignment.
    pub(crate) fn at(buf: &wgpu::Buffer, offset: u64, size: u64) -> Self {
        Self {
            buf: buf.clone(),
            offset,
            size,
        }
    }

    pub(crate) fn whole(buf: &wgpu::Buffer) -> Self {
        Self {
            buf: buf.clone(),
            offset: 0,
            size: buf.size(),
        }
    }
}

impl Binding for View {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    fn range(&self) -> (u64, u64) {
        (self.offset, self.size)
    }
}

/// What a cached bind group binds: the kernel, the fault words at binding
/// 1, and each storage slot's buffer and byte range. The parameter ring is
/// the same for every key and is bound at a dynamic offset.
#[derive(Clone, PartialEq, Eq, Hash)]
struct BindKey {
    kernel: (WgslModule, &'static str),
    fault: wgpu::Buffer,
    bufs: Vec<(wgpu::Buffer, u64, u64)>,
}

impl BindKey {
    fn buffers(&self) -> impl Iterator<Item = &wgpu::Buffer> {
        std::iter::once(&self.fault).chain(self.bufs.iter().map(|(b, _, _)| b))
    }
}

/// Bind groups by [`BindKey`], with every key that binds a buffer indexed by
/// that buffer. A cached group holds its buffers alive, so a buffer that
/// leaves the pool for good is forgotten here at once
/// ([`BindCache::forget`]); otherwise the cache would keep freed device
/// memory. The context's own fault words live as long as the context and
/// are not indexed.
#[derive(Default)]
struct BindCache {
    groups: HashMap<BindKey, wgpu::BindGroup>,
    by_buf: HashMap<wgpu::Buffer, Vec<BindKey>>,
}

impl BindCache {
    fn insert(&mut self, key: BindKey, group: wgpu::BindGroup, permanent: &[&wgpu::Buffer]) {
        if self.groups.len() >= BIND_CACHE_CAP {
            self.clear();
        }
        let mut seen: Vec<&wgpu::Buffer> = Vec::with_capacity(key.bufs.len() + 1);
        for b in key.buffers() {
            if permanent.contains(&b) || seen.contains(&b) {
                continue;
            }
            seen.push(b);
            self.by_buf.entry(b.clone()).or_default().push(key.clone());
        }
        self.groups.insert(key, group);
    }

    /// Drop every cached group that binds `buf`.
    fn forget(&mut self, buf: &wgpu::Buffer) {
        let Some(keys) = self.by_buf.remove(buf) else {
            return;
        };
        for key in keys {
            if self.groups.remove(&key).is_none() {
                continue;
            }
            for other in key.buffers() {
                if other == buf {
                    continue;
                }
                if let Some(list) = self.by_buf.get_mut(other) {
                    list.retain(|k| k != &key);
                    if list.is_empty() {
                        self.by_buf.remove(other);
                    }
                }
            }
        }
    }

    fn clear(&mut self) {
        self.groups.clear();
        self.by_buf.clear();
    }
}

/// Idle `MAP_READ` staging buffers, each a power-of-two size of at least
/// [`STAGING_MIN_BYTES`], at most [`STAGING_CACHE_BYTES`] in all.
#[derive(Default)]
struct StagingCache {
    free: Vec<wgpu::Buffer>,
    bytes: u64,
}

pub(crate) struct Inner {
    device: wgpu::Device,
    /// `Some` for the context's whole life: only [`Inner`]'s `Drop` takes it,
    /// to drop it off the caller's thread.
    queue: Option<wgpu::Queue>,
    /// [`DROP_WAIT`] unless a test changed it, in milliseconds.
    drop_wait_ms: AtomicU64,
    limits: wgpu::Limits,
    device_type: wgpu::DeviceType,
    adapter_name: String,
    hal: String,
    vendor: String,
    modules: Mutex<HashMap<WgslModule, wgpu::ShaderModule>>,
    pipelines: Mutex<HashMap<(WgslModule, &'static str), Arc<Pipe>>>,
    pool: Mutex<Pool>,
    rec: Mutex<Recorder>,
    /// The parameter ring: [`PARAM_SLOTS`] slots of `param_stride` bytes,
    /// bound at binding 0 of every dispatch at its slot's offset.
    params: wgpu::Buffer,
    param_stride: u64,
    binds: Mutex<BindCache>,
    staging: Mutex<StagingCache>,
    /// Dispatches recorded before a submit without a read: [`FLUSH_AT`]
    /// unless a benchmark changed it ([`WgpuContext::set_flush_at`]).
    flush_at: AtomicUsize,
    /// Live fault words, written by kernels: 0 and 1 the op mask (ops 0..31,
    /// then 32..63), 2 the first op + 1.
    fault: wgpu::Buffer,
    /// Faults moved out of `fault` by a read and not yet observed by the host.
    held: wgpu::Buffer,
    hold: Mutex<Option<(Arc<Pipe>, wgpu::BindGroup)>>,
    observed: AtomicU64,
    /// First op (+ 1) among the `observed` bits; 0 when none is recorded.
    first: AtomicU32,
    errors: Arc<Mutex<Vec<String>>>,
    /// Set by the device-lost callback and never cleared: loss is permanent.
    lost: Arc<Mutex<Option<String>>>,
    counters: Counters,
    /// Test seam: the next [`WgpuContext::read`] fails after its submission
    /// completes, before the host sees the copied bytes.
    #[cfg(test)]
    fail_next_read: std::sync::atomic::AtomicBool,
    /// Test seam: this many next buffer creations fail as out of memory.
    #[cfg(test)]
    fail_allocs: AtomicU32,
}

impl Inner {
    fn queue(&self) -> &wgpu::Queue {
        // Taken only in `Drop`, when no `&Inner` can exist any more.
        self.queue
            .as_ref()
            .expect("the queue is present until the context drops")
    }
}

/// wgpu's `Queue` drop waits for the queue to go idle with no timeout
/// (wgpu-core-30.0.1 `device/queue.rs:281`; on Metal a committed empty
/// command buffer and `waitUntilCompleted`, wgpu-hal-30.0.1
/// `src/metal/mod.rs:832-837`). With work in flight, or a GPU shared with
/// a long job of another process, that wait has no bound, and it runs on
/// whichever thread drops the last handle: for the C ABI, the caller of the
/// session free.
///
/// The queue is not the only drop that can wait. On Mesa's llvmpipe
/// (lavapipe), with 64 GEMMs in flight, the queue wait timed out at 300 ms
/// and then dropping the pipeline cache here did not finish within the
/// test's 1 s limit (`tests/drop.rs`): the pipelines those GEMMs use are
/// destroyed only once their work is done, and the driver waits for it.
///
/// So the queue and every other GPU object this context owns (pipelines,
/// shader modules, pooled buffers, the open encoder and its scratch, the
/// held bind group, the bind group and staging caches, and handles to the
/// device, the parameter ring and the fault buffers) move
/// to a thread of their own. That thread drops the queue first, which waits
/// for the GPU, then the rest. This drop waits for it at most
/// [`WgpuContext::set_drop_wait`] (default [`DROP_WAIT`]) and then drops
/// only the context's remaining handles to the device and the fault
/// buffers, which the thread still holds, so nothing is destroyed here. If
/// the GPU has not finished by then, the thread finishes the drop when the
/// GPU does: until then those objects, their GPU memory and one parked
/// thread stay alive, and if the GPU never finishes they are released only
/// at process exit. If the thread cannot be spawned, everything is dropped
/// here, unbounded, as before.
///
/// That hold is bounded, counted and reported: a drop that returns first
/// counts in [`DropStats::timed_out`] and, until its thread has released
/// everything, in [`DropStats::parked`]; [`WgpuContext::open`] refuses
/// while [`MAX_PARKED_DROPS`] are parked. The handoff is one atomic state
/// the drop and its thread race on ([`park`], [`reap`]): whichever moves it
/// off "running" first decides, so a thread that finishes at the deadline
/// stays uncounted and a parked one is uncounted exactly once. The count is
/// raised before the state moves, so while the two race it can read one
/// high (a refusal one drop early), never wrap.
impl Drop for Inner {
    fn drop(&mut self) {
        let Some(queue) = self.queue.take() else {
            return;
        };
        let rest = (
            (self.device.clone(), self.params.clone()),
            self.fault.clone(),
            self.held.clone(),
            std::mem::take(&mut *lock(&self.modules)),
            std::mem::take(&mut *lock(&self.pipelines)),
            std::mem::take(&mut *lock(&self.pool)),
            std::mem::take(&mut *lock(&self.rec)),
            lock(&self.hold).take(),
            (
                std::mem::take(&mut *lock(&self.binds)),
                std::mem::take(&mut *lock(&self.staging)),
            ),
        );
        let wait = std::time::Duration::from_millis(self.drop_wait_ms.load(Ordering::Relaxed));
        let (done, finished) = std::sync::mpsc::channel::<()>();
        let state = Arc::new(AtomicU8::new(DROP_RUNNING));
        let reaped = Reaped(Arc::clone(&state));
        let reaper = std::thread::Builder::new()
            .name("ojas-wgpu-queue-drop".to_string())
            .spawn(move || {
                drop(queue);
                drop(rest);
                // Released: done, and uncounted if it had parked.
                drop(reaped);
                // The receiver may have given up waiting; nothing to report.
                let _gone = done.send(());
            });
        // A spawn failure drops the closure and everything it held here,
        // and the state is done before anything could count it.
        if reaper.is_ok() && finished.recv_timeout(wait).is_err() && park(&state, &DROPS_PARKED) {
            // The drop is left to the thread, as documented above.
            DROPS_TIMED_OUT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Persistent portable GPU context. Clones share the device, caches and pool.
#[derive(Clone)]
pub struct WgpuContext {
    inner: Arc<Inner>,
}

/// Whether `size` bytes of a freed buffer may join the free list.
///
/// A new size is refused once [`POOL_MAX_KEYS`] distinct sizes are stored,
/// even when the byte cap would still allow it.
fn pool_accepts(existing_key: bool, key_count: usize, pooled_bytes: u64, size: u64) -> bool {
    if size == 0 || size > POOL_CAP_BYTES {
        return false;
    }
    if pooled_bytes.saturating_add(size) > POOL_CAP_BYTES {
        return false;
    }
    existing_key || key_count < POOL_MAX_KEYS
}

/// Whether an adapter of `kind` may be opened. A CPU adapter needs the
/// explicit opt-in value `1`; anything else, or no value, refuses it.
fn admits_adapter(kind: wgpu::DeviceType, opt_in: Option<&str>) -> bool {
    kind != wgpu::DeviceType::Cpu || opt_in == Some("1")
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn backend_err(detail: impl Into<String>) -> OjasError {
    OjasError::Backend {
        id: BackendId::Wgpu,
        detail: detail.into(),
    }
}

fn device_lost(detail: impl Into<String>) -> OjasError {
    OjasError::DeviceLost {
        backend: BackendId::Wgpu,
        detail: detail.into(),
    }
}

impl WgpuContext {
    /// Open the adapter with every limit it reports.
    pub fn open() -> Result<Self, DeviceError> {
        Self::open_inner(None)
    }

    /// Open with `cap` applied field by field as an upper bound on the
    /// adapter's limits. Used to run the refusal paths of a smaller device.
    pub fn open_capped(cap: &wgpu::Limits) -> Result<Self, DeviceError> {
        Self::open_inner(Some(cap))
    }

    fn open_inner(cap: Option<&wgpu::Limits>) -> Result<Self, DeviceError> {
        admit_open(DROPS_PARKED.load(Ordering::Acquire))?;
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = hal_backends();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|err| gpu_error(format!("request_adapter: {err}")))?;
        let info = adapter.get_info();
        let opt_in = std::env::var(ALLOW_CPU_ADAPTER_ENV).ok();
        if !admits_adapter(info.device_type, opt_in.as_deref()) {
            return Err(gpu_error(format!(
                "refusing CPU adapter {} ({:?}); set {ALLOW_CPU_ADAPTER_ENV}=1 to allow one \
                 (GPU-less CI only)",
                info.name, info.backend
            )));
        }
        let mut limits = adapter.limits();
        if let Some(cap) = cap {
            limits = cap_limits(&limits, cap);
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ojas-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|err| gpu_error(format!("request_device: {err}")))?;
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&errors);
        device.on_uncaptured_error(Arc::new(move |err: wgpu::Error| {
            let mut guard = lock(&sink);
            if guard.len() < 16 {
                guard.push(err.to_string());
            }
        }));
        // Blocking, and outside the error cap: a lost device must never be
        // dropped because the error queue was busy or full. Nothing holds
        // this lock across a wgpu call, so the callback cannot deadlock.
        let lost: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let lost_sink = Arc::clone(&lost);
        device.set_device_lost_callback(move |reason, message| {
            let mut guard = lock(&lost_sink);
            if guard.is_none() {
                *guard = Some(format!("device lost ({reason:?}): {message}"));
            }
        });
        let fault = zeroed_words(&device, "ojas-fault")?;
        let held = zeroed_words(&device, "ojas-fault-held")?;
        let param_stride = PARAM_BYTES
            .next_multiple_of(u64::from(limits.min_uniform_buffer_offset_alignment.max(1)));
        let params = param_ring(&device, param_stride)?;
        let (adapter_name, hal, vendor) = info_of(&info);
        Ok(Self {
            inner: Arc::new(Inner {
                device,
                queue: Some(queue),
                drop_wait_ms: AtomicU64::new(DROP_WAIT.as_millis() as u64),
                limits,
                device_type: info.device_type,
                adapter_name,
                hal,
                vendor,
                modules: Mutex::new(HashMap::new()),
                pipelines: Mutex::new(HashMap::new()),
                pool: Mutex::new(Pool::default()),
                rec: Mutex::new(Recorder::default()),
                params,
                param_stride,
                binds: Mutex::new(BindCache::default()),
                staging: Mutex::new(StagingCache::default()),
                flush_at: AtomicUsize::new(FLUSH_AT),
                fault,
                held,
                hold: Mutex::new(None),
                observed: AtomicU64::new(0),
                first: AtomicU32::new(0),
                errors,
                lost,
                counters: Counters::default(),
                #[cfg(test)]
                fail_next_read: std::sync::atomic::AtomicBool::new(false),
                #[cfg(test)]
                fail_allocs: AtomicU32::new(0),
            }),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }

    pub fn hal(&self) -> &str {
        &self.inner.hal
    }

    pub fn vendor(&self) -> &str {
        &self.inner.vendor
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.inner.queue()
    }

    /// How long dropping the last handle on this context (the backend and
    /// every tensor) waits for the GPU; [`DROP_WAIT`] by default. Tests set
    /// a short one.
    #[doc(hidden)]
    pub fn set_drop_wait(&self, wait: std::time::Duration) {
        let ms = u64::try_from(wait.as_millis()).unwrap_or(u64::MAX);
        self.inner.drop_wait_ms.store(ms, Ordering::Relaxed);
    }

    pub fn limits(&self) -> &wgpu::Limits {
        &self.inner.limits
    }

    pub fn stats(&self) -> CacheStats {
        let c = &self.inner.counters;
        let get = |a: &AtomicU64| a.load(Ordering::Relaxed);
        CacheStats {
            compiles: get(&c.compiles),
            pool_hits: get(&c.pool_hits),
            alloc_retries: get(&c.alloc_retries),
            submits: get(&c.submits),
            dispatches: get(&c.dispatches),
            uploads: get(&c.uploads),
            upload_bytes: get(&c.upload_bytes),
            reads: get(&c.reads),
            read_bytes: get(&c.read_bytes),
            scope_pops: get(&c.scope_pops),
            bind_groups: get(&c.bind_groups),
            bind_hits: get(&c.bind_hits),
            staging_creates: get(&c.staging_creates),
            offset_copies: get(&c.offset_copies),
        }
    }

    /// Dispatches recorded before the encoder is submitted without a read
    /// ([`FLUSH_AT`] by default). For the benchmark that chose that value;
    /// 0 is taken as 1.
    #[doc(hidden)]
    pub fn set_flush_at(&self, dispatches: usize) {
        self.inner
            .flush_at
            .store(dispatches.max(1), Ordering::Relaxed);
    }

    /// One more operand compacted into scratch for want of an aligned
    /// in-place binding.
    pub(crate) fn note_offset_copy(&self) {
        self.inner
            .counters
            .offset_copies
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Byte alignment a storage binding's offset needs on this device.
    pub(crate) fn storage_offset_alignment(&self) -> u64 {
        u64::from(self.inner.limits.min_storage_buffer_offset_alignment.max(4))
    }

    /// One blocking error-scope pop, counted.
    fn pop(&self, scope: wgpu::ErrorScopeGuard) -> Option<wgpu::Error> {
        self.inner
            .counters
            .scope_pops
            .fetch_add(1, Ordering::Relaxed);
        pollster::block_on(scope.pop())
    }

    /// Release every pooled buffer. Live tensors are not touched. An
    /// allocation the device refuses for want of memory does this before
    /// its one retry.
    pub fn trim_pool(&self) {
        let mut pool = lock(&self.inner.pool);
        pool.free.clear();
        pool.bytes = 0;
        drop(pool);
        // A cached group would keep a released buffer alive.
        lock(&self.inner.binds).clear();
        let mut staging = lock(&self.inner.staging);
        staging.free.clear();
        staging.bytes = 0;
    }

    /// Bytes of freed buffers the pool holds now, at most
    /// [`POOL_CAP_BYTES`].
    pub fn pooled_bytes(&self) -> u64 {
        lock(&self.inner.pool).bytes
    }

    /// What the device reports about its memory, as an
    /// [`ojas_device::MemoryProbe`] for [`ojas_device::ResourcePlan`]. wgpu
    /// has no device memory size, so the plan's room stays unknown
    /// ([`crate::WgpuMemory`]).
    pub fn memory(&self) -> crate::WgpuMemory {
        crate::WgpuMemory {
            device_type: self.inner.device_type,
            reserved_bytes: self
                .inner
                .device
                .generate_allocator_report()
                .map(|r| r.total_reserved_bytes),
            pool_cache_cap: POOL_CAP_BYTES,
        }
    }

    /// Make the next `n` buffer creations fail as out of memory.
    #[cfg(test)]
    pub(crate) fn fail_next_allocs(&self, n: u32) {
        self.inner.fail_allocs.store(n, Ordering::Relaxed);
    }

    /// One buffer creation under an out-of-memory error scope: the buffer,
    /// or wgpu's error text.
    fn create_scoped(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<wgpu::Buffer, String> {
        #[cfg(test)]
        {
            let left = &self.inner.fail_allocs;
            let mut n = left.load(Ordering::Relaxed);
            while n > 0 {
                match left.compare_exchange_weak(n, n - 1, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => return Err("injected out of memory".to_string()),
                    Err(now) => n = now,
                }
            }
        }
        let scope = self
            .inner
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buf = self.inner.device.create_buffer(desc);
        match self.pop(scope) {
            Some(err) => Err(err.to_string()),
            None => Ok(buf),
        }
    }

    /// A buffer of `desc`, as Metal allocates: a creation the device
    /// refuses for want of memory is tried once more after the pool's
    /// buffers are released and the device has had a bounded wait (5 s,
    /// as [`Self::failure`] waits) to retire what finished work freed. A
    /// second refusal is the error, naming both.
    fn new_buffer(
        &self,
        what: &str,
        desc: &wgpu::BufferDescriptor<'_>,
    ) -> Result<wgpu::Buffer, OjasError> {
        let first = match self.create_scoped(desc) {
            Ok(buf) => return Ok(buf),
            Err(first) => first,
        };
        self.inner
            .counters
            .alloc_retries
            .fetch_add(1, Ordering::Relaxed);
        self.trim_pool();
        // Its own result is not this call's: the retry below reports.
        let _settled = self.inner.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        self.create_scoped(desc).map_err(|again| {
            self.failure(format!(
                "{what} of {} bytes: {again} (retried after releasing the pool; first: {first})",
                desc.size
            ))
        })
    }

    pub(crate) fn same(&self, other: &Arc<Inner>) -> bool {
        Arc::ptr_eq(&self.inner, other)
    }

    /// Largest byte length one storage binding may have on this device.
    pub fn binding_cap(&self) -> u64 {
        let l = &self.inner.limits;
        let elems = (MAX_ELEMENTS as u64) * 4;
        l.max_storage_buffer_binding_size
            .min(l.max_buffer_size)
            .min(elems)
    }

    /// Refuse `bytes` past the binding cap before anything is allocated.
    pub(crate) fn check_bytes(&self, bytes: u64, budget: &Budget) -> Result<(), OjasError> {
        let cap = self.binding_cap();
        if bytes > cap {
            return Err(OjasError::CapacityExceeded {
                requested: bytes,
                cap,
                live: budget.live_bytes()?,
            });
        }
        Ok(())
    }

    fn alloc(&self, bytes: u64) -> Result<wgpu::Buffer, OjasError> {
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(backend_err(format!(
                "refusing a {bytes}-byte device buffer"
            )));
        }
        {
            let mut pool = lock(&self.inner.pool);
            let taken = pool.free.get_mut(&bytes).and_then(Vec::pop);
            if let Some(buf) = taken {
                pool.bytes = pool.bytes.saturating_sub(bytes);
                if pool.free.get(&bytes).is_some_and(|slot| slot.is_empty()) {
                    pool.free.remove(&bytes);
                }
                drop(pool);
                self.inner
                    .counters
                    .pool_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(buf);
            }
        }
        self.new_buffer(
            "allocating a tensor",
            &wgpu::BufferDescriptor {
                label: Some("ojas-tensor"),
                size: bytes,
                usage: storage_usage(),
                mapped_at_creation: false,
            },
        )
    }

    fn give_back(&self, buf: wgpu::Buffer, bytes: u64) {
        let mut pool = lock(&self.inner.pool);
        let exists = pool.free.contains_key(&bytes);
        if !pool_accepts(exists, pool.free.len(), pool.bytes, bytes) {
            drop(pool);
            // Gone for good: no cached bind group may keep it alive.
            lock(&self.inner.binds).forget(&buf);
            return;
        }
        pool.bytes = pool.bytes.saturating_add(bytes);
        pool.free.entry(bytes).or_default().push(buf);
    }

    /// A device buffer for a new tensor. The caller charges the budget by
    /// wrapping it in [`ojas_core::Tensor::from_device`].
    pub(crate) fn tensor_buffer(&self, bytes: u64) -> Result<WgpuBuffer, OjasError> {
        let buf = self.alloc(bytes)?;
        Ok(WgpuBuffer {
            buf: Some(buf),
            bytes,
            ctx: Arc::clone(&self.inner),
            shadow: None,
            below: None,
        })
    }

    /// Copy a host tensor's contiguous window to a new device buffer,
    /// encoded straight into the buffer's mapping a piece at a time
    /// ([`ojas_core::Tensor::for_each_ne_piece`]), so the host never holds a
    /// second full copy. `shadow` keeps the values of a U32 tensor so ids
    /// can be range-checked without reading the device.
    pub(crate) fn upload_tensor(
        &self,
        tensor: &ojas_core::Tensor,
        shadow: Option<Arc<[u32]>>,
    ) -> Result<WgpuBuffer, OjasError> {
        let len = (tensor.num_elements()? * tensor.dtype().size()) as u64;
        let bytes = len.div_ceil(4) * 4;
        let buf = self.mapped_with(bytes, storage_usage(), |view| {
            let mut at = 0usize;
            tensor.for_each_ne_piece(UPLOAD_PIECE_BYTES, |piece| {
                view.slice(at..at + piece.len()).copy_from_slice(piece);
                at += piece.len();
                Ok(())
            })?;
            Ok(at)
        })?;
        let c = &self.inner.counters;
        c.uploads.fetch_add(1, Ordering::Relaxed);
        c.upload_bytes.fetch_add(len, Ordering::Relaxed);
        Ok(WgpuBuffer {
            buf: Some(buf),
            bytes,
            ctx: Arc::clone(&self.inner),
            shadow,
            below: None,
        })
    }

    fn mapped(
        &self,
        bytes: u64,
        usage: wgpu::BufferUsages,
        data: &[u8],
    ) -> Result<wgpu::Buffer, OjasError> {
        self.mapped_with(bytes, usage, |view| {
            view.slice(..data.len()).copy_from_slice(data);
            Ok(data.len())
        })
    }

    /// A new buffer of `bytes`, mapped at creation: `fill` writes its
    /// leading bytes and returns how many, the rest is zeroed, and the
    /// buffer is unmapped.
    fn mapped_with(
        &self,
        bytes: u64,
        usage: wgpu::BufferUsages,
        fill: impl FnOnce(&mut wgpu::BufferViewMut) -> Result<usize, OjasError>,
    ) -> Result<wgpu::Buffer, OjasError> {
        if bytes == 0 {
            return Err(backend_err("refusing a zero-length upload"));
        }
        let buf = self.new_buffer(
            "allocating an upload",
            &wgpu::BufferDescriptor {
                label: Some("ojas-upload"),
                size: bytes,
                usage,
                mapped_at_creation: true,
            },
        )?;
        {
            let mut view = buf
                .slice(..)
                .get_mapped_range_mut()
                .map_err(|err| self.failure(format!("upload map: {err}")))?;
            let n = fill(&mut view)?;
            let mut rest = n;
            while rest < view.len() {
                let end = (rest + 64).min(view.len());
                let zeros = [0u8; 64];
                view.slice(rest..end).copy_from_slice(&zeros[..end - rest]);
                rest = end;
            }
        }
        buf.unmap();
        Ok(buf)
    }

    fn pipe(&self, kernel: &Kernel) -> Result<Arc<Pipe>, OjasError> {
        let key = (kernel.module, kernel.entry);
        if let Some(hit) = lock(&self.inner.pipelines).get(&key) {
            return Ok(Arc::clone(hit));
        }
        let storage = 1 + kernel.slots.len() as u32;
        let max = self.inner.limits.max_storage_buffers_per_shader_stage;
        if storage > max {
            return Err(OjasError::Unsupported {
                op: kernel.entry,
                detail: format!("needs {storage} storage bindings; the device allows {max}"),
            });
        }
        let device = &self.inner.device;
        let module = {
            let mut modules = lock(&self.inner.modules);
            match modules.get(&kernel.module) {
                Some(m) => m.clone(),
                None => {
                    let src = wgsl_module(kernel.module)?;
                    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
                    let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(kernel.entry),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                    if let Some(err) = self.pop(scope) {
                        return Err(self.failure(format!("compiling {:?}: {err}", kernel.module)));
                    }
                    modules.insert(kernel.module, m.clone());
                    m
                }
            }
        };
        let buffer = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Binding 0 is one slot of the parameter ring, at a dynamic offset.
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(PARAM_BYTES),
            },
            count: None,
        };
        let mut entries = vec![
            uniform,
            buffer(1, wgpu::BufferBindingType::Storage { read_only: false }),
        ];
        for slot in kernel.slots {
            entries.push(match *slot {
                Slot::R(b) => buffer(b, wgpu::BufferBindingType::Storage { read_only: true }),
                Slot::W(b) => buffer(b, wgpu::BufferBindingType::Storage { read_only: false }),
            });
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(kernel.entry),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(kernel.entry),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(kernel.entry),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(kernel.entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });
        if let Some(err) = self.pop(scope) {
            return Err(self.failure(format!("pipeline {}: {err}", kernel.entry)));
        }
        self.inner.counters.compiles.fetch_add(1, Ordering::Relaxed);
        let pipe = Arc::new(Pipe { pipeline, layout });
        lock(&self.inner.pipelines).insert(key, Arc::clone(&pipe));
        Ok(pipe)
    }

    /// A job whose kernels raise fault id `fault_id` (an op index + 1, or 0
    /// for a job that reports nothing).
    pub(crate) fn job<'a>(&'a self, budget: &'a Budget, fault_id: u32) -> Job<'a> {
        Job {
            ctx: self,
            budget,
            fault_id,
            local_fault: None,
            steps: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Pipeline and bind group for one dispatch of `kernel`: parameter
    /// words at binding 0 (a ring slot, given at the dynamic offset when the
    /// dispatch is recorded), `fault` at binding 1 and `bufs` at the
    /// kernel's slots. `params` words fill 0..15 and word 15 is `fault_id`.
    /// The group comes from the cache when an identical one was made before.
    fn bind(
        &self,
        kernel: &Kernel,
        params: usize,
        fault_id: u32,
        fault: &wgpu::Buffer,
        bufs: &[&dyn Binding],
    ) -> Result<(Arc<Pipe>, wgpu::BindGroup), OjasError> {
        if fault_id > FAULT_OPS {
            return Err(backend_err(format!(
                "{}: fault id {fault_id} is past the {FAULT_OPS} ops the fault words hold",
                kernel.entry
            )));
        }
        if bufs.len() != kernel.slots.len() {
            return Err(backend_err(format!(
                "{} takes {} buffers, given {}",
                kernel.entry,
                kernel.slots.len(),
                bufs.len()
            )));
        }
        if params > 15 {
            return Err(backend_err(format!(
                "{}: too many parameters",
                kernel.entry
            )));
        }
        let pipe = self.pipe(kernel)?;
        let key = BindKey {
            kernel: (kernel.module, kernel.entry),
            fault: fault.clone(),
            bufs: bufs
                .iter()
                .map(|b| {
                    let (offset, size) = b.range();
                    (b.buffer().clone(), offset, size)
                })
                .collect(),
        };
        let c = &self.inner.counters;
        if let Some(hit) = lock(&self.inner.binds).groups.get(&key) {
            c.bind_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((pipe, hit.clone()));
        }
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.inner.params,
                    offset: 0,
                    size: NonZeroU64::new(PARAM_BYTES),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fault.as_entire_binding(),
            },
        ];
        for (slot, (buf, offset, size)) in kernel.slots.iter().zip(&key.bufs) {
            let binding = match *slot {
                Slot::R(b) | Slot::W(b) => b,
            };
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: *offset,
                    size: NonZeroU64::new(*size),
                }),
            });
        }
        let scope = self
            .device()
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel.entry),
            layout: &pipe.layout,
            entries: &entries,
        });
        if let Some(err) = self.pop(scope) {
            return Err(self.failure(format!("bind group {}: {err}", kernel.entry)));
        }
        c.bind_groups.fetch_add(1, Ordering::Relaxed);
        lock(&self.inner.binds).insert(
            key,
            group.clone(),
            &[&self.inner.fault, &self.inner.held],
        );
        Ok((pipe, group))
    }

    /// The cached `fault_hold` dispatch over this context's words.
    fn hold(&self) -> Result<(Arc<Pipe>, wgpu::BindGroup), OjasError> {
        let mut cached = lock(&self.inner.hold);
        if let Some(hit) = cached.as_ref() {
            return Ok(hit.clone());
        }
        let made = self.bind(&FAULT_HOLD, 0, 0, &self.inner.fault, &[&self.inner.held])?;
        *cached = Some(made.clone());
        Ok(made)
    }

    fn submit_locked(&self, rec: &mut Recorder) -> Option<wgpu::SubmissionIndex> {
        let encoder = rec.encoder.take()?;
        if rec.slots > 0 {
            // Ordered after every earlier submission and before this one.
            let used = rec.slots * self.inner.param_stride as usize;
            self.inner
                .queue()
                .write_buffer(&self.inner.params, 0, &rec.params[..used]);
            rec.slots = 0;
        }
        let index = self.inner.queue().submit(std::iter::once(encoder.finish()));
        self.inner.counters.submits.fetch_add(1, Ordering::Relaxed);
        rec.dispatches = 0;
        for s in rec.scratch.drain(..) {
            self.give_back(s.buf, s.bytes);
        }
        Some(index)
    }

    fn wait(&self, index: wgpu::SubmissionIndex) -> Result<(), OjasError> {
        self.inner
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(std::time::Duration::from_secs(120)),
            })
            .map(|_| ())
            .map_err(|err| self.failure(format!("poll: {err}")))
    }

    /// Copy `len` bytes at `offset` of `buf` to the host after all recorded
    /// work. The fault word rides in the same copy and is cleared.
    pub(crate) fn read(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, OjasError> {
        let start = offset & !3;
        let end = (offset + len).div_ceil(4) * 4;
        let span = end - start;
        let stage_bytes = span + 16;
        let staging = self.staging(stage_bytes)?;
        let (hold_pipe, hold_group) = self.hold()?;
        let index = {
            let mut rec = lock(&self.inner.rec);
            if rec.slots == PARAM_SLOTS {
                self.submit_locked(&mut rec);
            }
            let at = rec.slot(&[0; 16], self.inner.param_stride);
            let encoder = rec.encoder.get_or_insert_with(|| self.encoder());
            if span > 0 {
                encoder.copy_buffer_to_buffer(buf, start, &staging, 0, span);
            }
            // Move the live fault words into `held` and copy `held` out. The
            // live word is cleared, but `held` is only cleared by a release
            // the host records after it has seen these bytes.
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ojas-fault-hold"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&hold_pipe.pipeline);
                pass.set_bind_group(0, &hold_group, &[at]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.inner.held, 0, &staging, span, 16);
            self.submit_locked(&mut rec)
        };
        let index = index.ok_or_else(|| backend_err("read recorded no commands"))?;
        let (sender, receiver) = std::sync::mpsc::channel();
        staging
            .slice(..stage_bytes)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.wait(index)?;
        #[cfg(test)]
        if self.inner.fail_next_read.swap(false, Ordering::Relaxed) {
            return Err(backend_err("injected read failure"));
        }
        // Another thread's poll may be the one that runs this callback, and it
        // can land just after our poll returns.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let mapped = loop {
            match receiver.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(result) => break result,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(
                        self.failure("map callback was dropped without running".to_string())
                    );
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(
                            self.failure("map callback did not run within 120 s".to_string())
                        );
                    }
                    self.inner
                        .device
                        .poll(wgpu::PollType::Poll)
                        .map_err(|err| self.failure(format!("poll: {err}")))?;
                }
            }
        };
        mapped.map_err(|err| self.failure(format!("map_async: {err}")))?;
        let (out, bits, first) = {
            let view = staging
                .slice(..stage_bytes)
                .get_mapped_range()
                .map_err(|err| self.failure(format!("get_mapped_range: {err}")))?;
            let s = span as usize;
            let word = |i: usize| {
                let mut w = [0u8; 4];
                w.copy_from_slice(&view[s + 4 * i..s + 4 * i + 4]);
                u32::from_le_bytes(w)
            };
            let lo = (offset - start) as usize;
            let bits = u64::from(word(0)) | (u64::from(word(1)) << 32);
            (view[lo..lo + len as usize].to_vec(), bits, word(2))
        };
        staging.unmap();
        // Unmapped and idle again: the next read may take it.
        self.staging_back(staging);
        if bits != 0 || first != 0 {
            self.inner.observed.fetch_or(bits, Ordering::Relaxed);
            if first != 0 {
                let _ = self.inner.first.compare_exchange(
                    0,
                    first,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
            }
            // Seen: now, and only now, clear exactly these from `held`.
            // The release allocates nothing, so it charges an empty budget.
            let none = Budget::new(0);
            let mut job = self.job(&none, 0);
            job.dispatch(
                &FAULT_RELEASE,
                &[bits as u32, (bits >> 32) as u32, first],
                &[&self.inner.held],
                (1, 1, 1),
            )?;
            job.commit()?;
        }
        let c = &self.inner.counters;
        c.reads.fetch_add(1, Ordering::Relaxed);
        c.read_bytes.fetch_add(len, Ordering::Relaxed);
        self.check_errors()?;
        Ok(out)
    }

    /// Submit everything recorded and wait for it, collecting the fault word.
    pub fn sync(&self) -> Result<(), OjasError> {
        self.read(&self.inner.fault, 0, 0).map(|_| ())
    }

    /// Make the next [`WgpuContext::read`] fail once its submission is done.
    #[cfg(test)]
    pub(crate) fn fail_next_read(&self) {
        self.inner.fail_next_read.store(true, Ordering::Relaxed);
    }

    /// `(bits, first)` seen by reads since the last call, then cleared:
    /// every faulting op's bit (bit `i` for op index `i`), and the fault id
    /// (index plus one) of the first of them in recording order, 0 when no
    /// fault was seen.
    pub(crate) fn take_faults(&self) -> (u64, u32) {
        let bits = self.inner.observed.swap(0, Ordering::Relaxed);
        let first = self.inner.first.swap(0, Ordering::Relaxed);
        (bits, first)
    }

    /// The error for a failed device call (a wait, a map, or a buffer,
    /// shader, pipeline or bind group it creates): [`OjasError::DeviceLost`]
    /// when the device-lost callback has fired, naming the loss, otherwise
    /// [`OjasError::Backend`].
    fn failure(&self, detail: String) -> OjasError {
        if lock(&self.inner.lost).is_none() {
            // wgpu runs the lost callback from a poll that finds the queue
            // empty, but a wait on a submission the destroyed device refused
            // fails before that point. Wait (bounded) for the last accepted
            // submission so a pending loss is delivered. Its own result is
            // not the error being reported, so it is deliberately dropped.
            let _delivered = self.inner.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });
        }
        match lock(&self.inner.lost).as_deref() {
            Some(lost) => device_lost(format!("{detail}; {lost}")),
            None => backend_err(detail),
        }
    }

    /// Uncaptured wgpu errors since the last call, and a lost device on every
    /// call after the loss: [`OjasError::DeviceLost`] once the device is
    /// lost, [`OjasError::Backend`] for errors alone.
    fn check_errors(&self) -> Result<(), OjasError> {
        let lost = lock(&self.inner.lost).clone();
        let mut errors = lock(&self.inner.errors);
        if errors.is_empty() && lost.is_none() {
            return Ok(());
        }
        let is_lost = lost.is_some();
        let mut parts: Vec<String> = lost.into_iter().collect();
        parts.append(&mut errors);
        let detail = format!("wgpu reported: {}", parts.join("; "));
        Err(if is_lost {
            device_lost(detail)
        } else {
            backend_err(detail)
        })
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ojas-batch"),
            })
    }

    /// An idle `MAP_READ` buffer of at least `bytes`: a kept one of the
    /// same power-of-two size, or a new one.
    fn staging(&self, bytes: u64) -> Result<wgpu::Buffer, OjasError> {
        let size = staging_size(bytes);
        {
            let mut cache = lock(&self.inner.staging);
            if let Some(i) = cache.free.iter().position(|b| b.size() == size) {
                let buf = cache.free.swap_remove(i);
                cache.bytes = cache.bytes.saturating_sub(size);
                return Ok(buf);
            }
        }
        self.inner
            .counters
            .staging_creates
            .fetch_add(1, Ordering::Relaxed);
        self.new_buffer(
            "a staging buffer",
            &wgpu::BufferDescriptor {
                label: Some("ojas-staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )
    }

    /// Keep an unmapped staging buffer for a later read, within
    /// [`STAGING_CACHE_BYTES`].
    fn staging_back(&self, buf: wgpu::Buffer) {
        let mut cache = lock(&self.inner.staging);
        let size = buf.size();
        if cache.bytes.saturating_add(size) <= STAGING_CACHE_BYTES {
            cache.bytes += size;
            cache.free.push(buf);
        }
    }
}

/// The staging size a read of `bytes` uses: the next power of two, at least
/// [`STAGING_MIN_BYTES`].
fn staging_size(bytes: u64) -> u64 {
    bytes.max(STAGING_MIN_BYTES).next_power_of_two()
}

/// The parameter ring: [`PARAM_SLOTS`] slots of `stride` bytes.
fn param_ring(device: &wgpu::Device, stride: u64) -> Result<wgpu::Buffer, DeviceError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ojas-params"),
        size: stride * PARAM_SLOTS as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(DeviceError::Capacity {
            kind: Device::Vulkan,
            detail: format!("ojas-params: {err}"),
        });
    }
    Ok(buf)
}

/// A 16-byte storage buffer of four zero words.
fn zeroed_words(device: &wgpu::Device, label: &str) -> Result<wgpu::Buffer, DeviceError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 16,
        usage: storage_usage(),
        mapped_at_creation: true,
    });
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(DeviceError::Capacity {
            kind: Device::Vulkan,
            detail: format!("{label}: {err}"),
        });
    }
    {
        let mut view = buf
            .slice(..)
            .get_mapped_range_mut()
            .map_err(|err| gpu_error(format!("{label} map: {err}")))?;
        view.copy_from_slice(&[0u8; 16]);
    }
    buf.unmap();
    Ok(buf)
}

/// Upper bound `want` by `cap`, one field at a time.
fn cap_limits(have: &wgpu::Limits, cap: &wgpu::Limits) -> wgpu::Limits {
    let mut out = have.clone();
    out.max_storage_buffer_binding_size = have
        .max_storage_buffer_binding_size
        .min(cap.max_storage_buffer_binding_size);
    out.max_buffer_size = have.max_buffer_size.min(cap.max_buffer_size);
    out.max_compute_workgroup_storage_size = have
        .max_compute_workgroup_storage_size
        .min(cap.max_compute_workgroup_storage_size);
    out.max_compute_workgroups_per_dimension = have
        .max_compute_workgroups_per_dimension
        .min(cap.max_compute_workgroups_per_dimension);
    out.max_storage_buffers_per_shader_stage = have
        .max_storage_buffers_per_shader_stage
        .min(cap.max_storage_buffers_per_shader_stage);
    out
}

enum Step {
    Dispatch {
        pipe: Arc<Pipe>,
        group: wgpu::BindGroup,
        words: [u32; 16],
        grid: (u32, u32, u32),
    },
    Clear(wgpu::Buffer),
    Copy {
        src: wgpu::Buffer,
        src_offset: u64,
        dst: wgpu::Buffer,
        size: u64,
    },
}

/// The commands of one op. Nothing reaches the encoder until [`Job::commit`].
pub(crate) struct Job<'a> {
    ctx: &'a WgpuContext,
    budget: &'a Budget,
    fault_id: u32,
    local_fault: Option<wgpu::Buffer>,
    steps: Vec<Step>,
    scratch: Vec<Scratch>,
}

impl Job<'_> {
    /// `params` fill words 0..15; word 15 is this op's fault id.
    pub fn dispatch(
        &mut self,
        kernel: &Kernel,
        params: &[u32],
        bufs: &[&dyn Binding],
        grid: (u32, u32, u32),
    ) -> Result<(), OjasError> {
        let max = self.ctx.limits().max_compute_workgroups_per_dimension;
        for dim in [grid.0, grid.1, grid.2] {
            if dim == 0 || dim > max {
                return Err(OjasError::OutOfRange {
                    op: kernel.entry,
                    detail: format!("grid {grid:?} is outside 1..={max} per axis"),
                });
            }
        }
        let fault = self.local_fault.as_ref().unwrap_or(&self.ctx.inner.fault);
        let (pipe, group) = self
            .ctx
            .bind(kernel, params.len(), self.fault_id, fault, bufs)?;
        let mut words = [0u32; 16];
        words[..params.len()].copy_from_slice(params);
        words[15] = self.fault_id;
        self.steps.push(Step::Dispatch {
            pipe,
            group,
            words,
            grid,
        });
        Ok(())
    }

    /// Bind `word` (16 bytes, cleared here) as binding 1 of every dispatch
    /// recorded after this call, in place of the context's fault word, until
    /// [`Job::global_fault`]. An op that must decide on the device whether
    /// *this* call faulted reads `word` back as an input; the context's word
    /// may still hold an unobserved bit from an earlier call.
    pub fn local_fault(&mut self, word: &wgpu::Buffer) {
        self.clear(word);
        self.local_fault = Some(word.clone());
    }

    /// Bind the context's fault word again.
    pub fn global_fault(&mut self) {
        self.local_fault = None;
    }

    pub fn clear(&mut self, buf: &wgpu::Buffer) {
        self.steps.push(Step::Clear(buf.clone()));
    }

    /// Copy `size` bytes from `src_offset` within `src`'s bound range to the
    /// start of `dst`.
    pub fn copy(&mut self, src: &dyn Binding, src_offset: u64, dst: &wgpu::Buffer, size: u64) {
        self.steps.push(Step::Copy {
            src: src.buffer().clone(),
            src_offset: src.range().0 + src_offset,
            dst: dst.clone(),
            size,
        });
    }

    /// Device scratch charged to the budget until its commands are submitted.
    ///
    /// Valid only for the commands this job records: at the next submit, by
    /// any thread, the buffer goes back to the pool and another caller may
    /// take and overwrite it. A value the host reads back in a later
    /// submission must live in a buffer the caller holds (a tensor from the
    /// backend's `out`), not in scratch.
    pub fn scratch(&mut self, bytes: u64) -> Result<wgpu::Buffer, OjasError> {
        self.ctx.check_bytes(bytes, self.budget)?;
        let charge = self.budget.try_reserve(bytes)?;
        let buf = self.ctx.alloc(bytes)?;
        self.scratch.push(Scratch {
            buf: buf.clone(),
            bytes,
            _charge: charge,
        });
        Ok(buf)
    }

    /// Byte alignment a storage binding's offset needs on this device.
    pub fn storage_offset_alignment(&self) -> u64 {
        self.ctx.storage_offset_alignment()
    }

    /// Count one operand compacted for want of an aligned binding.
    pub fn note_offset_copy(&self) {
        self.ctx.note_offset_copy();
    }

    /// Small host-built index data, uploaded through a mapped buffer.
    pub fn upload_u32(&mut self, data: &[u32]) -> Result<wgpu::Buffer, OjasError> {
        let bytes = (data.len() as u64) * 4;
        self.ctx.check_bytes(bytes, self.budget)?;
        let charge = self.budget.try_reserve(bytes)?;
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = self.ctx.mapped(bytes, storage_usage(), &raw)?;
        let c = &self.ctx.inner.counters;
        c.uploads.fetch_add(1, Ordering::Relaxed);
        c.upload_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.scratch.push(Scratch {
            buf: buf.clone(),
            bytes,
            _charge: charge,
        });
        Ok(buf)
    }

    /// Record every step into the shared encoder. Consecutive dispatches share
    /// one compute pass; wgpu orders dispatches that touch the same buffer.
    /// When the parameter ring is full, what is recorded is submitted and
    /// the job goes on in a new encoder; submissions run in order, so the
    /// split changes nothing the job computes.
    pub fn commit(self) -> Result<(), OjasError> {
        let ctx = self.ctx;
        let stride = ctx.inner.param_stride;
        let mut rec = lock(&ctx.inner.rec);
        let mut dispatched = 0usize;
        let mut i = 0;
        while i < self.steps.len() {
            if matches!(self.steps[i], Step::Dispatch { .. }) && rec.slots == PARAM_SLOTS {
                ctx.submit_locked(&mut rec);
            }
            let rec = &mut *rec;
            let encoder = rec.encoder.get_or_insert_with(|| ctx.encoder());
            match &self.steps[i] {
                Step::Dispatch { .. } => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ojas-op"),
                        timestamp_writes: None,
                    });
                    while let Some(Step::Dispatch {
                        pipe,
                        group,
                        words,
                        grid,
                    }) = self.steps.get(i)
                    {
                        if rec.slots == PARAM_SLOTS {
                            break;
                        }
                        let at = put_slot(&mut rec.params, &mut rec.slots, words, stride);
                        pass.set_pipeline(&pipe.pipeline);
                        pass.set_bind_group(0, group, &[at]);
                        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                        dispatched += 1;
                        i += 1;
                    }
                }
                Step::Clear(buf) => {
                    encoder.clear_buffer(buf, 0, None);
                    i += 1;
                }
                Step::Copy {
                    src,
                    src_offset,
                    dst,
                    size,
                } => {
                    encoder.copy_buffer_to_buffer(src, *src_offset, dst, 0, *size);
                    i += 1;
                }
            }
        }
        rec.scratch.extend(self.scratch);
        rec.dispatches += dispatched;
        ctx.inner
            .counters
            .dispatches
            .fetch_add(dispatched as u64, Ordering::Relaxed);
        if rec.dispatches >= ctx.inner.flush_at.load(Ordering::Relaxed) {
            ctx.submit_locked(&mut rec);
        }
        Ok(())
    }
}

/// Device memory behind a wgpu tensor. Dropping it returns the buffer to the
/// pool. A U32 upload keeps a host copy of its values (`shadow`) so token ids
/// and targets are validated without a device read. A U32 tensor a kernel
/// produced (`argmax_rows`) has no shadow; `below` bounds its values.
pub struct WgpuBuffer {
    buf: Option<wgpu::Buffer>,
    bytes: u64,
    ctx: Arc<Inner>,
    shadow: Option<Arc<[u32]>>,
    below: Option<u32>,
}

impl WgpuBuffer {
    pub(crate) fn raw(&self) -> Result<&wgpu::Buffer, OjasError> {
        self.buf
            .as_ref()
            .ok_or_else(|| backend_err("device buffer was already released"))
    }

    pub(crate) fn context(&self) -> &Arc<Inner> {
        &self.ctx
    }

    pub(crate) fn shadow(&self) -> Option<&Arc<[u32]>> {
        self.shadow.as_ref()
    }

    /// Every value of this U32 buffer is below the bound, by construction.
    pub(crate) fn below(&self) -> Option<u32> {
        self.below
    }

    /// The same buffer, marked as holding U32 values below `bound`.
    pub(crate) fn bounded(mut self, bound: u32) -> Self {
        self.below = Some(bound);
        self
    }
}

impl std::fmt::Debug for WgpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBuffer")
            .field("bytes", &self.bytes)
            .field("shadow", &self.shadow.as_ref().map(|s| s.len()))
            .finish()
    }
}

impl Drop for WgpuBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            WgpuContext {
                inner: Arc::clone(&self.ctx),
            }
            .give_back(buf, self.bytes);
        }
    }
}

impl DeviceBuffer for WgpuBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Wgpu
    }

    fn byte_len(&self) -> usize {
        self.bytes as usize
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        let ctx = WgpuContext {
            inner: Arc::clone(&self.ctx),
        };
        ctx.read(self.raw()?, offset as u64, len as u64)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_list_refuses_a_new_size_once_the_key_cap_is_full() {
        // Pre-fix, give_back inserted every distinct size until POOL_CAP_BYTES.
        // Sixteen-byte buffers could add millions of HashMap keys.
        let mut keys = 0usize;
        let mut bytes = 0u64;
        for n in 1..10_000u64 {
            let size = n.saturating_mul(4);
            if pool_accepts(false, keys, bytes, size) {
                keys += 1;
                bytes = bytes.saturating_add(size);
            }
        }
        assert!(keys <= POOL_MAX_KEYS, "free-list grew to {keys} keys");
        assert!(!pool_accepts(false, POOL_MAX_KEYS, 0, 16));
        assert!(pool_accepts(true, POOL_MAX_KEYS, 0, 16));
        assert!(!pool_accepts(true, 1, POOL_CAP_BYTES, 16));
    }

    #[test]
    fn a_cpu_adapter_needs_the_explicit_opt_in() {
        use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu, Other, VirtualGpu};
        for opt_in in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("yes"),
            Some(" 1"),
        ] {
            assert!(
                !admits_adapter(Cpu, opt_in),
                "{opt_in:?} admitted a CPU adapter"
            );
        }
        assert!(admits_adapter(Cpu, Some("1")));
        for kind in [DiscreteGpu, IntegratedGpu, VirtualGpu, Other] {
            assert!(admits_adapter(kind, None), "{kind:?}");
            assert!(admits_adapter(kind, Some("0")), "{kind:?}");
        }
    }

    /// An allocation the device refuses for want of memory is tried once
    /// more after the pool is released, for tensors, uploads and staging
    /// alike, as Metal recycles and retries. Before, the first refusal
    /// failed the call while up to `POOL_CAP_BYTES` of idle buffers stayed
    /// pooled. A second refusal is the error and names the retry.
    #[test]
    fn an_out_of_memory_allocation_is_retried_once_after_releasing_the_pool() {
        let ctx = WgpuContext::open().expect("wgpu adapter");
        drop(ctx.tensor_buffer(1 << 20).expect("alloc"));
        assert_eq!(ctx.pooled_bytes(), 1 << 20, "a freed buffer is pooled");
        ctx.fail_next_allocs(1);
        let b = ctx.tensor_buffer(2 << 20).expect("the retry allocates");
        assert_eq!(ctx.pooled_bytes(), 0, "the retry released the pool first");
        assert_eq!(ctx.stats().alloc_retries, 1);
        drop(b);
        ctx.fail_next_allocs(1);
        let up = ctx
            .upload_tensor(
                &ojas_core::Tensor::from_f32(&[1.0; 16], &[16], &Budget::new(1 << 20)).unwrap(),
                None,
            )
            .expect("an upload retries");
        assert_eq!(ctx.stats().alloc_retries, 2);
        drop(up);
        ctx.fail_next_allocs(1);
        ctx.sync().expect("a read's staging buffer retries");
        assert_eq!(ctx.stats().alloc_retries, 3);
        ctx.fail_next_allocs(2);
        let err = ctx.tensor_buffer(3 << 20).expect_err("two refusals fail");
        assert!(
            matches!(&err, OjasError::Backend { detail, .. }
                if detail.contains("retried after releasing the pool")),
            "{err:?}"
        );
        assert_eq!(ctx.stats().alloc_retries, 4);
        ctx.tensor_buffer(3 << 20)
            .expect("a later allocation is unaffected");
    }

    /// A cached bind group holds its buffers alive, so a buffer the pool
    /// will not keep must leave the cache with every group that binds it,
    /// and only those; trimming the pool empties the cache.
    #[test]
    fn a_buffer_the_pool_refuses_leaves_the_bind_group_cache() {
        let ctx = WgpuContext::open().expect("wgpu adapter");
        let budget = Budget::new(1 << 30);
        let kept = ctx.alloc(1 << 12).expect("alloc");
        let other = ctx.alloc(1 << 12).expect("alloc");
        let gone = ctx.alloc(1 << 12).expect("alloc");
        let kernel = Kernel {
            module: WgslModule::Pointwise,
            entry: "add_inplace",
            slots: &[Slot::R(2), Slot::W(6)],
        };
        for (a, b) in [(&kept, &gone), (&gone, &other), (&kept, &other)] {
            let mut job = ctx.job(&budget, 0);
            job.dispatch(&kernel, &[1], &[a, b], (1, 1, 1)).unwrap();
            job.commit().unwrap();
        }
        ctx.sync().unwrap();
        let groups = |ctx: &WgpuContext| lock(&ctx.inner.binds).groups.len();
        // Three op groups and the read's fault hold.
        assert_eq!(groups(&ctx), 4);
        // Past every cap the pool has: refused, so forgotten.
        lock(&ctx.inner.pool).bytes = POOL_CAP_BYTES;
        ctx.give_back(gone.clone(), 1 << 12);
        assert_eq!(groups(&ctx), 2, "only the two groups binding the freed buffer go");
        {
            let binds = lock(&ctx.inner.binds);
            assert!(!binds.by_buf.contains_key(&gone));
            assert!(binds
                .by_buf
                .values()
                .flatten()
                .all(|k| k.buffers().all(|b| b != &gone)));
        }
        lock(&ctx.inner.pool).bytes = 0;
        ctx.trim_pool();
        assert_eq!(groups(&ctx), 0);
    }

    /// The drop and its thread race on one state: whatever the order, the
    /// count is back to 0 once both have run, a parked drop is counted until
    /// its thread reaps it, and the count never reads past the drops that
    /// could be parked (it wrapped to `usize::MAX` when the uncount could
    /// run before the count, which refused every open).
    #[test]
    fn parking_and_reaping_race_without_wrapping_the_count() {
        const ROUNDS: usize = 5_000;
        let parked = Arc::new(AtomicUsize::new(0));
        let seen_max = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watcher = {
            let (parked, seen_max, stop) = (parked.clone(), seen_max.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    seen_max.fetch_max(parked.load(Ordering::Acquire), Ordering::Relaxed);
                }
            })
        };
        let mut parked_rounds = 0usize;
        for _ in 0..ROUNDS {
            let state = Arc::new(AtomicU8::new(DROP_RUNNING));
            let reaper = {
                let (state, parked) = (state.clone(), parked.clone());
                std::thread::spawn(move || reap(&state, &parked))
            };
            if park(&state, &parked) {
                parked_rounds += 1;
            }
            reaper.join().unwrap();
            assert_eq!(parked.load(Ordering::Acquire), 0, "the count leaked");
            assert_eq!(state.load(Ordering::Acquire), DROP_DONE);
        }
        stop.store(true, Ordering::Relaxed);
        watcher.join().unwrap();
        // One round at a time: at most the drop's own count, briefly.
        assert!(seen_max.load(Ordering::Relaxed) <= 1, "{seen_max:?}");
        eprintln!("{parked_rounds} of {ROUNDS} rounds parked before their thread reaped");
        // A thread that finished first is never counted.
        let state = AtomicU8::new(DROP_RUNNING);
        reap(&state, &parked);
        assert!(!park(&state, &parked));
        assert_eq!(parked.load(Ordering::Acquire), 0);
    }

    /// Drops still holding their devices bound how many more contexts may
    /// open: below the cap one opens, at it the open is a capacity error
    /// naming the count.
    #[test]
    fn opens_are_refused_while_too_many_drops_hold_their_devices() {
        for n in 0..MAX_PARKED_DROPS {
            assert!(admit_open(n).is_ok(), "{n}");
        }
        for n in [MAX_PARKED_DROPS, MAX_PARKED_DROPS + 1, usize::MAX] {
            match admit_open(n) {
                Err(DeviceError::Capacity { kind, detail }) => {
                    assert_eq!(kind, Device::Vulkan);
                    assert!(detail.contains(&n.to_string()), "{detail}");
                }
                other => panic!("{n}: {other:?}"),
            }
        }
    }

    /// The probe reports the adapter's type, its pool cap, and no device
    /// memory size: wgpu has none to give.
    #[test]
    fn the_memory_probe_reports_what_wgpu_knows() {
        use ojas_device::{MemoryProbe, MemoryReport};
        let ctx = WgpuContext::open().expect("wgpu adapter");
        let m = ctx.memory();
        assert_eq!(m.pool_cache_cap, POOL_CAP_BYTES);
        assert_eq!(m.memory_bytes(), MemoryReport::Unknown);
        assert_eq!(m.kind(), Device::Vulkan);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // wgpu's Metal HAL keeps no allocator report, and an Apple GPU
            // is integrated.
            assert_eq!(m.reserved_bytes, None, "{m:?}");
            assert_eq!(m.device_type, wgpu::DeviceType::IntegratedGpu, "{m:?}");
        }
    }

    #[test]
    fn device_lost_is_reported_even_past_the_error_cap() {
        // Pre-fix, the lost callback used try_lock and the same 16-message
        // cap as uncaptured errors, so a full queue dropped "device lost".
        let ctx = WgpuContext::open().expect("wgpu adapter");
        for _ in 0..20 {
            let _ = ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("empty-usage"),
                size: 4,
                usage: wgpu::BufferUsages::empty(),
                mapped_at_creation: false,
            });
        }
        ctx.device().destroy();
        let _ = ctx.device().poll(wgpu::PollType::Poll);
        let first = ctx.check_errors().expect_err("errors were queued");
        assert!(
            matches!(
                &first,
                OjasError::DeviceLost {
                    backend: BackendId::Wgpu,
                    ..
                }
            ),
            "device lost was dropped: {first:?}"
        );
        // Loss is permanent: it is reported again, not consumed.
        let again = ctx.check_errors().expect_err("a lost device stays lost");
        assert!(matches!(again, OjasError::DeviceLost { .. }), "{again:?}");
    }

    #[test]
    fn a_device_call_after_loss_is_the_typed_device_lost_error() {
        // Pre-fix, a lost device surfaced as OjasError::Backend with
        // "device lost" in its text, and callers matched the substring.
        let ctx = WgpuContext::open().expect("wgpu adapter");
        ctx.sync().expect("a live device syncs");
        ctx.device().destroy();
        let lost = ctx.sync().expect_err("a destroyed device cannot sync");
        assert!(
            matches!(
                &lost,
                OjasError::DeviceLost {
                    backend: BackendId::Wgpu,
                    ..
                }
            ),
            "{lost:?}"
        );
        // A plain uncaptured error on a live device stays Backend.
        let live = WgpuContext::open().expect("wgpu adapter");
        let _ = live.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("empty-usage"),
            size: 4,
            usage: wgpu::BufferUsages::empty(),
            mapped_at_creation: false,
        });
        let plain = live
            .check_errors()
            .expect_err("the bad buffer was reported");
        assert!(matches!(plain, OjasError::Backend { .. }), "{plain:?}");
    }
}
