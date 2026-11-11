//! Persistent-pool L2-tiled `ojas-simd` CPU `e-means` engine.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ojas_core::{
    BackendId, Budget, DType, HostLease, OjasError, Scratch, Tensor, CPU_THREAD_CEILING,
};

use crate::{
    choose_splits, epoch_permutation, prescaled_unit_normalize_f32, prescaled_unit_normalize_f64,
    resolve_recipe, sample_subset_sorted, scheduled_alpha, separate_collapsed_split_f32, CpuMath,
    EmeansConfig, EmeansFit, EpochStat, SplitMix64, UNASSIGNED_LABEL,
};

/// Maximum worker count permitted by [`ojas_core::CPU_THREAD_CEILING`].
const MAX_THREADS: usize = CPU_THREAD_CEILING as usize;
/// Explicit stack size (`8 MiB`) for persistent CPU worker threads.
const WORKER_STACK_BYTES: usize = 8 << 20;
/// Short spin window (`40 µs`) before a persistent worker parks on its condvar.
const WORKER_SPIN: Duration = Duration::from_micros(40);

/// Fixed row sub-tile height (`64` rows) independent of `pool.threads()`, ensuring `sgemm_tile`
/// and per-row reductions are 0-ULP bit-identical across `threads = 1, 2, 4, 8`.
const ROW_TILE: usize = 64;
/// L2-resident centroid panel width (`256` clusters -> `64 * 256 * 4 = 64 KiB` score tile).
const C_TILE: usize = 256;
/// Widened Accelerate row panel (`2048` rows) under `CpuMath::Fast` when `threads.is_none()`.
#[cfg(target_os = "macos")]
const ACCEL_ROW_PANEL: usize = 2_048;
/// Widened Accelerate centroid panel (`min(k, 4096)`) under `CpuMath::Fast` when `threads.is_none()`.
#[cfg(target_os = "macos")]
const ACCEL_COL_PANEL: usize = 4_096;

/// Row threshold at which CSR bucket histogramming runs in parallel across [`CpuPool`].
const PARALLEL_CSR_MIN_ROWS: usize = 4_096;
/// Fixed row block (`1024` rows) per parallel CSR histogram tile.
const CSR_HIST_TILE: usize = 1_024;
/// Disjoint cluster block size (`16` clusters) per parallel CSR sum-gather task.
const CLUSTER_GATHER_TILE: usize = 16;

/// Domain separator mixed with `seed` when sampling initial centroids (`init_centroids: None`).
const INIT_CENTROID_SEED_TAG: u64 = 0x494e_4954_4345_4e54;
/// Revival interval (`_CHECK_UNUSED_EVERY = 5` in `faiss/contrib/e_means/base.py`).
pub const CHECK_UNUSED_EVERY: usize = 5;

fn out_of_range(detail: impl Into<String>) -> OjasError {
    OjasError::OutOfRange {
        op: "emeans",
        detail: detail.into(),
    }
}

fn shape_err(detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op: "emeans",
        detail: detail.into(),
    }
}

const fn nonfinite_err() -> OjasError {
    OjasError::NonFinite { op: "emeans" }
}

fn checked_mul_len(a: usize, b: usize, what: &str) -> Result<usize, OjasError> {
    a.checked_mul(b)
        .ok_or_else(|| out_of_range(format!("{what} ({a} * {b}) overflows usize")))
}

/// Canonical revival trigger check (`(step + 1) % 5 == 0` when `unused_threshold > 0.0`).
#[inline]
pub fn should_check_revive(step: usize, unused_threshold: f32) -> bool {
    unused_threshold > 0.0 && (step + 1).is_multiple_of(CHECK_UNUSED_EVERY)
}

// =============================================================================
// Zero-copy 2D Tensor View (`Mapped` checked before `Host`) + `bf16` Helpers
// =============================================================================

/// Borrowed element representation of a validated 2D `F32` or `Bf16` [`Tensor`].
#[derive(Debug, Clone, Copy)]
pub enum MatrixData<'a> {
    /// Contiguous row-major `f32` elements (`rows * cols`).
    F32(&'a [f32]),
    /// Contiguous row-major host `bf16` words (`rows * cols`).
    Bf16(&'a [u16]),
    /// Contiguous row-major mapped `bf16` byte pairs (`rows * cols`) over a zero-copy [`HostLease`].
    Bf16Bytes(&'a [[u8; 2]]),
}

enum MatrixBacking<'a> {
    HostF32(&'a [f32]),
    HostBf16(&'a [u16]),
    MappedF32(Box<dyn HostLease>),
    MappedBf16(Box<dyn HostLease>),
}

/// Validated zero-copy 2D view over a `Host` or `Mapped` (`SharedBuf`) [`Tensor`].
pub struct MatrixView<'a> {
    rows: usize,
    cols: usize,
    backing: MatrixBacking<'a>,
}

impl<'a> MatrixView<'a> {
    /// Open a zero-copy 2D view over `tensor`.
    ///
    /// Checks `DType::{F32, Bf16}` (`OjasError::Dtype`), refuses pure device buffers
    /// (`OjasError::Placement`), validates 2D shape (`OjasError::Shape`), and checks
    /// `tensor.mapped_buffer().is_some()` **before** `f32_slice()`/`bf16_slice()` so `Mapped`
    /// tensors acquire a zero-copy `host_view_window()` lease without cloning host storage.
    pub fn open(tensor: &'a Tensor, op: &'static str) -> Result<Self, OjasError> {
        let dtype = tensor.dtype();
        if !matches!(dtype, DType::F32 | DType::Bf16) {
            return Err(OjasError::Dtype {
                op,
                expected: DType::F32,
                got: dtype,
            });
        }
        if let Some(dev_buf) = tensor.device_buffer() {
            return Err(OjasError::Placement {
                op,
                expected: None,
                found: Some(dev_buf.backend()),
            });
        }
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(OjasError::Shape {
                op,
                detail: format!("expected 2D tensor [rows, cols], got shape {shape:?}"),
            });
        }
        let rows = shape[0];
        let cols = shape[1];
        let expected_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: format!("matrix dimensions {rows} * {cols} overflow usize"),
            })?;

        let backing = if tensor.mapped_buffer().is_some() {
            let lease = tensor.host_view_window()?;
            match dtype {
                DType::F32 => {
                    let slice = lease.as_f32()?;
                    if slice.len() != expected_elems {
                        return Err(OjasError::Shape {
                            op,
                            detail: format!(
                                "mapped f32 lease len {} != expected {expected_elems}",
                                slice.len()
                            ),
                        });
                    }
                    MatrixBacking::MappedF32(lease)
                }
                DType::Bf16 => {
                    let bytes = lease.as_bytes();
                    let expected_bytes =
                        expected_elems
                            .checked_mul(2)
                            .ok_or_else(|| OjasError::OutOfRange {
                                op,
                                detail: "mapped bf16 byte length overflows usize".to_string(),
                            })?;
                    if bytes.len() != expected_bytes {
                        return Err(OjasError::Shape {
                            op,
                            detail: format!(
                                "mapped bf16 lease bytes {} != expected {expected_bytes}",
                                bytes.len()
                            ),
                        });
                    }
                    MatrixBacking::MappedBf16(lease)
                }
                _ => unreachable!("dtype validated above"),
            }
        } else {
            match dtype {
                DType::F32 => {
                    let slice = tensor.f32_slice()?;
                    if slice.len() != expected_elems {
                        return Err(OjasError::Shape {
                            op,
                            detail: format!(
                                "host f32 slice len {} != expected {expected_elems}",
                                slice.len()
                            ),
                        });
                    }
                    MatrixBacking::HostF32(slice)
                }
                DType::Bf16 => {
                    let slice = tensor.bf16_slice()?;
                    if slice.len() != expected_elems {
                        return Err(OjasError::Shape {
                            op,
                            detail: format!(
                                "host bf16 slice len {} != expected {expected_elems}",
                                slice.len()
                            ),
                        });
                    }
                    MatrixBacking::HostBf16(slice)
                }
                _ => unreachable!("dtype validated above"),
            }
        };

        Ok(Self {
            rows,
            cols,
            backing,
        })
    }

    /// Number of rows in the 2D tensor.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Feature dimension (`cols`) of each row.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Borrowed underlying element data.
    #[inline]
    pub fn data(&self) -> MatrixData<'_> {
        match &self.backing {
            MatrixBacking::HostF32(s) => MatrixData::F32(s),
            MatrixBacking::HostBf16(s) => MatrixData::Bf16(s),
            MatrixBacking::MappedF32(lease) => {
                MatrixData::F32(lease.as_f32().expect("validated in MatrixView::open"))
            }
            MatrixBacking::MappedBf16(lease) => {
                MatrixData::Bf16Bytes(lease.as_bytes().as_chunks::<2>().0)
            }
        }
    }

    /// Copy and validate row `row` into `dst` (`len == self.cols()`), refusing non-finite values.
    pub fn copy_row_f32_checked(&self, row: usize, dst: &mut [f32]) -> Result<(), OjasError> {
        if row >= self.rows {
            return Err(out_of_range(format!(
                "row index {row} out of range for {} rows",
                self.rows
            )));
        }
        if dst.len() != self.cols {
            return Err(shape_err(format!(
                "row buffer len {} != cols {}",
                dst.len(),
                self.cols
            )));
        }
        let start = row * self.cols;
        let end = start + self.cols;
        match self.data() {
            MatrixData::F32(src) => {
                let r = &src[start..end];
                for (&v, out) in r.iter().zip(dst.iter_mut()) {
                    if !v.is_finite() {
                        return Err(nonfinite_err());
                    }
                    *out = v;
                }
                Ok(())
            }
            MatrixData::Bf16(src) => decode_bf16_row_checked(&src[start..end], dst),
            MatrixData::Bf16Bytes(src) => decode_bf16_bytes_checked(&src[start..end], dst),
        }
    }

    /// Copy and validate a contiguous window of `row_count` rows starting at `row_start` into `dst`.
    pub fn copy_window_f32_checked(
        &self,
        row_start: usize,
        row_count: usize,
        dst: &mut [f32],
    ) -> Result<(), OjasError> {
        let row_end = row_start
            .checked_add(row_count)
            .ok_or_else(|| out_of_range("row window end overflows usize"))?;
        if row_end > self.rows {
            return Err(out_of_range(format!(
                "row window {row_start}..{row_end} exceeds {} rows",
                self.rows
            )));
        }
        let elems = checked_mul_len(row_count, self.cols, "window elements")?;
        if dst.len() != elems {
            return Err(shape_err(format!(
                "window dst len {} != expected {elems}",
                dst.len()
            )));
        }
        let start = row_start * self.cols;
        let end = start + elems;
        match self.data() {
            MatrixData::F32(src) => {
                for (&v, out) in src[start..end].iter().zip(dst.iter_mut()) {
                    if !v.is_finite() {
                        return Err(nonfinite_err());
                    }
                    *out = v;
                }
                Ok(())
            }
            MatrixData::Bf16(src) => decode_bf16_row_checked(&src[start..end], dst),
            MatrixData::Bf16Bytes(src) => decode_bf16_bytes_checked(&src[start..end], dst),
        }
    }

    /// Validate that every coordinate and squared row L2 norm across the effective dataset is finite.
    pub fn validate_rows_finite(&self, subset: Option<&[u32]>) -> Result<(), OjasError> {
        let d = self.cols;
        match subset {
            Some(sub) => {
                for &r_u32 in sub {
                    let r = r_u32 as usize;
                    if r >= self.rows {
                        return Err(out_of_range("subset row index exceeds matrix rows"));
                    }
                    self.validate_single_row(r, d)?;
                }
            }
            None => {
                for r in 0..self.rows {
                    self.validate_single_row(r, d)?;
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn validate_single_row(&self, r: usize, d: usize) -> Result<(), OjasError> {
        let start = r * d;
        let end = start + d;
        let mut rn = 0.0_f32;
        match self.data() {
            MatrixData::F32(s) => {
                for &v in &s[start..end] {
                    if !v.is_finite() {
                        return Err(nonfinite_err());
                    }
                    rn += v * v;
                }
            }
            MatrixData::Bf16(s) => {
                for &bits in &s[start..end] {
                    let v = bf16_word_to_f32_checked(bits)?;
                    rn += v * v;
                }
            }
            MatrixData::Bf16Bytes(s) => {
                for pair in &s[start..end] {
                    let v = bf16_word_to_f32_checked(u16::from_ne_bytes(*pair))?;
                    rn += v * v;
                }
            }
        }
        if !rn.is_finite() {
            return Err(nonfinite_err());
        }
        Ok(())
    }
}

#[inline]
fn bf16_word_to_f32_checked(bits: u16) -> Result<f32, OjasError> {
    if (bits & 0x7fff) >= 0x7f80 {
        return Err(nonfinite_err());
    }
    let clean = if (bits & 0x7f80) == 0 {
        bits & 0x8000
    } else {
        bits
    };
    Ok(f32::from_bits(u32::from(clean) << 16))
}

fn decode_bf16_bytes_checked(src: &[[u8; 2]], dst: &mut [f32]) -> Result<(), OjasError> {
    if src.len() != dst.len() {
        return Err(shape_err(format!(
            "bf16 byte-pair len {} != dst len {}",
            src.len(),
            dst.len()
        )));
    }
    for (pair, out) in src.iter().zip(dst.iter_mut()) {
        *out = bf16_word_to_f32_checked(u16::from_ne_bytes(*pair))?;
    }
    Ok(())
}

/// Convert raw `bf16` bits into FTZ-clean `f32` coordinates, refusing non-finite values.
pub fn decode_bf16_row_checked(src: &[u16], dst: &mut [f32]) -> Result<(), OjasError> {
    if src.len() != dst.len() {
        return Err(shape_err(format!(
            "bf16 slice len {} != dst len {}",
            src.len(),
            dst.len()
        )));
    }
    for (&bits, out) in src.iter().zip(dst.iter_mut()) {
        *out = bf16_word_to_f32_checked(bits)?;
    }
    Ok(())
}

/// Convert raw `bf16` bits into a `Budget`-charged `Scratch<f32>`, refusing non-finite values.
pub fn decode_bf16_to_scratch(src: &[u16], budget: &Budget) -> Result<Scratch<f32>, OjasError> {
    for &bits in src {
        let _ = bf16_word_to_f32_checked(bits)?;
    }
    Scratch::<f32>::try_extend(src.len(), budget, |dst| {
        for &bits in src {
            let clean = if (bits & 0x7f80) == 0 {
                bits & 0x8000
            } else {
                bits
            };
            dst.push(f32::from_bits(u32::from(clean) << 16));
        }
    })
}

/// Convert mapped native-endian `bf16` byte pairs into a `Budget`-charged `Scratch<f32>`,
/// refusing non-finite values.
pub fn decode_bf16_bytes_to_scratch(
    src: &[[u8; 2]],
    budget: &Budget,
) -> Result<Scratch<f32>, OjasError> {
    for pair in src {
        let _ = bf16_word_to_f32_checked(u16::from_ne_bytes(*pair))?;
    }
    Scratch::<f32>::try_extend(src.len(), budget, |dst| {
        for pair in src {
            let bits = u16::from_ne_bytes(*pair);
            let clean = if (bits & 0x7f80) == 0 {
                bits & 0x8000
            } else {
                bits
            };
            dst.push(f32::from_bits(u32::from(clean) << 16));
        }
    })
}

/// Validate explicit `init_centroids` or deterministically sample `k` initial rows from `x`.
pub fn resolve_init_centroids(
    x: &MatrixView<'_>,
    subset: Option<&[u32]>,
    init_centroids: Option<&Tensor>,
    k: usize,
    spherical: bool,
    seed: u64,
    budget: &Budget,
) -> Result<Scratch<f32>, OjasError> {
    let d = x.cols();
    let total_elems = checked_mul_len(k, d, "centroid elements")?;
    let mut out = Scratch::<f32>::try_alloc(total_elems, budget)?;

    if let Some(init_t) = init_centroids {
        let init_view = MatrixView::open(init_t, "emeans")?;
        if init_view.rows() != k || init_view.cols() != d {
            return Err(shape_err(format!(
                "init_centroids shape [{}, {}] != expected [{k}, {d}]",
                init_view.rows(),
                init_view.cols()
            )));
        }
        init_view.copy_window_f32_checked(0, k, out.as_mut_slice())?;
    } else {
        let effective_n = subset.map_or(x.rows(), <[u32]>::len);
        if effective_n < k {
            return Err(out_of_range(format!(
                "effective_n ({effective_n}) must be >= k ({k})"
            )));
        }
        if effective_n > (u32::MAX as usize) {
            return Err(out_of_range(format!(
                "effective_n ({effective_n}) exceeds u32::MAX"
            )));
        }
        if effective_n == k {
            for j in 0..k {
                let global_row = subset.map_or(j, |s| s[j] as usize);
                x.copy_row_f32_checked(global_row, &mut out.as_mut_slice()[j * d..(j + 1) * d])?;
            }
        } else {
            let mut idx_pool = Scratch::<u32>::try_extend(effective_n, budget, |dst| {
                dst.extend(0..(effective_n as u32));
            })?;
            let mut rng = SplitMix64::new(SplitMix64::mix(seed ^ INIT_CENTROID_SEED_TAG));
            let slice = idx_pool.as_mut_slice();
            for i in 0..k {
                let rem = (effective_n - i) as u64;
                let pick = i + (rng.next_bounded_u64(rem) as usize);
                slice.swap(i, pick);
            }
            for (j, &eff_idx) in slice[..k].iter().enumerate() {
                let global_row = subset.map_or(eff_idx as usize, |s| s[eff_idx as usize] as usize);
                x.copy_row_f32_checked(global_row, &mut out.as_mut_slice()[j * d..(j + 1) * d])?;
            }
        }
    }

    if spherical {
        let mut row_tmp = Scratch::<f32>::try_alloc(d, budget)?;
        for j in 0..k {
            let row_slice = &mut out.as_mut_slice()[j * d..(j + 1) * d];
            row_tmp.as_mut_slice().copy_from_slice(row_slice);
            prescaled_unit_normalize_f32(row_tmp.as_slice(), row_slice);
            if row_slice.iter().any(|v| !v.is_finite()) {
                return Err(nonfinite_err());
            }
        }
    }

    Ok(out)
}

// =============================================================================
// Persistent Worker Pool (`CpuPool`) Modeled on `ojas-cpu/src/pool.rs`
// =============================================================================

trait Work: Send + Sync {
    fn run_one(&self) -> bool;
}

type TaskFn<T> = Arc<dyn Fn(usize) -> T + Send + Sync>;

struct BatchState<T> {
    results: Vec<Option<T>>,
    finished: usize,
    panic: Option<(usize, String)>,
}

struct Batch<T> {
    next: AtomicUsize,
    count: usize,
    task: Mutex<Option<TaskFn<T>>>,
    state: Mutex<BatchState<T>>,
    done: Condvar,
}

impl<T: Send> Work for Batch<T> {
    fn run_one(&self) -> bool {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        if index >= self.count {
            return false;
        }
        let task = lock_unpoisoned(&self.task).clone();
        // Drop `t` BEFORE acquiring `self.state` and incrementing `finished`, guaranteeing
        // no worker still holds an `Arc` reference to captured `Scratch` buffers once `run` wakes.
        let outcome = match task {
            Some(t) => {
                let res = caught_unwind(|| t(index));
                drop(t);
                res
            }
            None => Err("batch closure taken before task ran".to_string()),
        };
        let mut state = lock_unpoisoned(&self.state);
        match outcome {
            Ok(value) => {
                if let Some(slot) = state.results.get_mut(index) {
                    *slot = Some(value);
                }
            }
            Err(text) => {
                if state.panic.as_ref().is_none_or(|(first, _)| index < *first) {
                    state.panic = Some((index, text));
                }
            }
        }
        state.finished += 1;
        if state.finished == self.count {
            self.done.notify_all();
        }
        true
    }
}

struct Queue {
    batches: VecDeque<Arc<dyn Work>>,
    shutdown: bool,
}

struct SharedPool {
    queue: Mutex<Queue>,
    wake: Condvar,
    posted: AtomicUsize,
}

struct Workers {
    handles: Vec<JoinHandle<()>>,
    failed: Option<String>,
}

/// Persistent worker pool keeping `ojas_simd::sgemm_tile`'s `2.125 MiB` `thread_local!`
/// packing buffer warm across `fit` and `assign` calls.
pub struct CpuPool {
    shared: Arc<SharedPool>,
    threads: usize,
    explicit_threads: bool,
    spawned: AtomicBool,
    workers: Mutex<Workers>,
}

impl std::fmt::Debug for CpuPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuPool")
            .field("threads", &self.threads)
            .field("explicit_threads", &self.explicit_threads)
            .finish()
    }
}

impl CpuPool {
    /// Construct a persistent worker pool.
    ///
    /// - `None` selects `available_parallelism().clamp(1, CPU_THREAD_CEILING)` with
    ///   `is_explicit_threads() == false`.
    /// - `Some(n)` selects `n` threads with `is_explicit_threads() == true`, refusing `0` and
    ///   values above [`ojas_core::CPU_THREAD_CEILING`] with [`OjasError::OutOfRange`].
    pub fn new(threads: Option<usize>) -> Result<Self, OjasError> {
        let (resolved, explicit_threads) = match threads {
            Some(0) => {
                return Err(out_of_range("CPU thread count must be >= 1, got 0"));
            }
            Some(t) if t > MAX_THREADS => {
                return Err(out_of_range(format!(
                    "CPU thread count ({t}) exceeds CPU_THREAD_CEILING ({MAX_THREADS})"
                )));
            }
            Some(t) => (t, true),
            None => {
                let avail = std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
                    .clamp(1, MAX_THREADS);
                (avail, false)
            }
        };

        Ok(Self {
            shared: Arc::new(SharedPool {
                queue: Mutex::new(Queue {
                    batches: VecDeque::new(),
                    shutdown: false,
                }),
                wake: Condvar::new(),
                posted: AtomicUsize::new(0),
            }),
            threads: resolved,
            explicit_threads,
            spawned: AtomicBool::new(resolved == 1),
            workers: Mutex::new(Workers {
                handles: Vec::new(),
                failed: None,
            }),
        })
    }

    /// Total configured worker count (including the calling thread).
    #[inline]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Whether the caller pinned an explicit `Some(n)` thread count.
    #[inline]
    pub fn is_explicit_threads(&self) -> bool {
        self.explicit_threads
    }

    fn ensure_workers(&self) -> Result<(), OjasError> {
        if self.spawned.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut workers = lock_unpoisoned(&self.workers);
        if let Some(text) = &workers.failed {
            return Err(OjasError::Backend {
                id: BackendId::Cpu,
                detail: text.clone(),
            });
        }
        while workers.handles.len() + 1 < self.threads {
            let idx = workers.handles.len() + 1;
            let shared = Arc::clone(&self.shared);
            match std::thread::Builder::new()
                .name(format!("ojas-cluster-w{idx}"))
                .stack_size(WORKER_STACK_BYTES)
                .spawn(move || worker_loop(shared))
            {
                Ok(handle) => workers.handles.push(handle),
                Err(err) => {
                    let msg = format!("spawning ojas-cluster worker {idx} failed: {err}");
                    workers.failed = Some(msg.clone());
                    return Err(OjasError::Backend {
                        id: BackendId::Cpu,
                        detail: msg,
                    });
                }
            }
        }
        self.spawned.store(true, Ordering::Release);
        Ok(())
    }

    /// Run `task(0..count)` across the persistent pool and return outputs in ascending index order.
    pub fn run<T, F>(&self, count: usize, task: F) -> Result<Vec<T>, OjasError>
    where
        T: Send + 'static,
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        if count == 0 {
            return Ok(Vec::new());
        }
        if count == 1 || self.threads <= 1 {
            return match caught_unwind(|| (0..count).map(task).collect()) {
                Ok(v) => Ok(v),
                Err(text) => Err(task_panic_error(&text)),
            };
        }

        let batch = Arc::new(Batch {
            next: AtomicUsize::new(0),
            count,
            task: Mutex::new(Some(Arc::new(task))),
            state: Mutex::new(BatchState {
                results: (0..count).map(|_| None).collect(),
                finished: 0,
                panic: None,
            }),
            done: Condvar::new(),
        });

        self.ensure_workers()?;
        {
            let work: Arc<dyn Work> = batch.clone();
            let mut queue = lock_unpoisoned(&self.shared.queue);
            queue.batches.push_back(work);
            self.shared.posted.fetch_add(1, Ordering::Release);
        }
        if count >= self.threads {
            self.shared.wake.notify_all();
        } else {
            for _ in 0..(count - 1) {
                self.shared.wake.notify_one();
            }
        }

        while batch.run_one() {}
        unpublish(&self.shared, Arc::as_ptr(&batch) as *const ());

        {
            let mut state = lock_unpoisoned(&batch.state);
            while state.finished < count {
                state = match batch.done.wait(state) {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
            }
        }

        // Drop the task closure outside locks so any captured `Arc<Scratch<T>>` releases its
        // `Budget` reservation deterministically before `run` returns.
        let taken_task = lock_unpoisoned(&batch.task).take();
        drop(taken_task);

        let mut state = lock_unpoisoned(&batch.state);
        if let Some((_, text)) = state.panic.take() {
            return Err(task_panic_error(&text));
        }

        let mut out = Vec::with_capacity(count);
        for slot in state.results.iter_mut() {
            match slot.take() {
                Some(v) => out.push(v),
                None => {
                    return Err(OjasError::Backend {
                        id: BackendId::Cpu,
                        detail: "worker task finished without a result".to_string(),
                    });
                }
            }
        }
        Ok(out)
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        {
            let mut queue = lock_unpoisoned(&self.shared.queue);
            queue.shutdown = true;
        }
        self.shared.posted.fetch_add(1, Ordering::Release);
        self.shared.wake.notify_all();
        let me = std::thread::current().id();
        let handles = std::mem::take(&mut lock_unpoisoned(&self.workers).handles);
        for handle in handles {
            if handle.thread().id() != me {
                let _ = handle.join();
            }
        }
    }
}

fn worker_loop(shared: Arc<SharedPool>) {
    loop {
        let seen = shared.posted.load(Ordering::Acquire);
        let job = {
            let queue = lock_unpoisoned(&shared.queue);
            if queue.shutdown {
                return;
            }
            queue.batches.front().cloned()
        };
        match job {
            Some(job) => {
                while job.run_one() {}
                unpublish(&shared, Arc::as_ptr(&job) as *const ());
            }
            None => {
                let start = Instant::now();
                let mut spins = 0_u32;
                while shared.posted.load(Ordering::Acquire) == seen {
                    std::hint::spin_loop();
                    spins = spins.wrapping_add(1);
                    if spins.is_multiple_of(64) && start.elapsed() >= WORKER_SPIN {
                        break;
                    }
                }
                let mut queue = lock_unpoisoned(&shared.queue);
                while queue.batches.is_empty() && !queue.shutdown {
                    queue = match shared.wake.wait(queue) {
                        Ok(guard) => guard,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                }
            }
        }
    }
}

fn unpublish(shared: &SharedPool, target: *const ()) {
    let mut removed = Vec::new();
    {
        let mut queue = lock_unpoisoned(&shared.queue);
        let mut kept = VecDeque::with_capacity(queue.batches.len());
        for item in queue.batches.drain(..) {
            if Arc::as_ptr(&item) as *const () == target {
                removed.push(item);
            } else {
                kept.push_back(item);
            }
        }
        queue.batches = kept;
    }
    drop(removed);
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn caught_unwind<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let text = panic_text(payload.as_ref());
        if let Err(second) = catch_unwind(AssertUnwindSafe(move || drop(payload))) {
            std::mem::forget(second);
        }
        text
    })
}

fn task_panic_error(text: &str) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail: format!("worker task panicked: {text}"),
    }
}

// =============================================================================
// Shared L2-Tiled Assignment & Deterministic Ascending CSR Gather Kernel
// =============================================================================

/// Prepare effective centroids (`unit_centroids` when `spherical`) and `norm2_c` into pre-allocated
/// slices, refusing non-finite coordinates or squared-norm overflow with [`OjasError::NonFinite`].
pub(crate) fn prepare_centroids_into(
    centroids: &[f32],
    k: usize,
    d: usize,
    spherical: bool,
    c_eff_out: &mut [f32],
    norm2_c_out: &mut [f32],
) -> Result<(), OjasError> {
    let expected = checked_mul_len(k, d, "centroids len")?;
    if centroids.len() != expected || c_eff_out.len() != expected || norm2_c_out.len() != k {
        return Err(shape_err("centroid buffer length mismatch"));
    }
    if centroids.iter().any(|v| !v.is_finite()) {
        return Err(nonfinite_err());
    }

    if spherical {
        for j in 0..k {
            let src = &centroids[j * d..(j + 1) * d];
            let dst = &mut c_eff_out[j * d..(j + 1) * d];
            prescaled_unit_normalize_f32(src, dst);
            if dst.iter().any(|v| !v.is_finite()) {
                return Err(nonfinite_err());
            }
            norm2_c_out[j] = 0.0;
        }
    } else {
        c_eff_out.copy_from_slice(centroids);
        for j in 0..k {
            let row = &c_eff_out[j * d..(j + 1) * d];
            let mut cn = 0.0_f32;
            for &v in row {
                cn += v * v;
            }
            if !cn.is_finite() {
                return Err(nonfinite_err());
            }
            norm2_c_out[j] = cn;
        }
    }
    Ok(())
}

#[inline]
fn finalize_row_dist_f32(
    x_row: &[f32],
    c_row: &[f32],
    rn: f32,
    best_s: f32,
    spherical: bool,
) -> f32 {
    if spherical {
        return (2.0_f32 + best_s).max(0.0_f32);
    }
    let raw = rn + best_s;
    let scale = rn.max(1.0e-30_f32);
    if raw.abs() <= 1_024.0_f32 * 1.192_092_9e-7_f32 * scale {
        let mut exact_d = 0.0_f32;
        for (&xv, &cv) in x_row.iter().zip(c_row.iter()) {
            let diff = xv - cv;
            exact_d += diff * diff;
        }
        return exact_d;
    }
    raw.max(0.0_f32)
}

/// Exact `f64` nearest-cluster fallback invoked only when an `f32` intermediate (`dot`,
/// `cn - 2 * dot`, or `rn + best_s`) overflows near `f32::MAX`. Preserves lowest-index tie
/// breaking and fails closed with [`OjasError::NonFinite`] if the true minimum distance exceeds
/// finite `f32` range.
#[cold]
fn assign_row_exact_f64(
    x_row: &[f32],
    c_eff: &[f32],
    k: usize,
    d: usize,
    spherical: bool,
) -> Result<(u32, f32), OjasError> {
    let mut best_l = 0_u32;
    if spherical {
        let mut best_s = f64::INFINITY;
        for j in 0..k {
            let c_row = &c_eff[j * d..(j + 1) * d];
            let mut dot = 0.0_f64;
            for col in 0..d {
                dot += f64::from(x_row[col]) * f64::from(c_row[col]);
            }
            if !dot.is_finite() {
                return Err(nonfinite_err());
            }
            let sc = -2.0_f64 * dot;
            if sc < best_s {
                best_s = sc;
                best_l = j as u32;
            }
        }
        let dist = ((2.0_f64 + best_s).max(0.0_f64)) as f32;
        if !dist.is_finite() {
            return Err(nonfinite_err());
        }
        Ok((best_l, dist))
    } else {
        let mut best_d = f64::INFINITY;
        for j in 0..k {
            let c_row = &c_eff[j * d..(j + 1) * d];
            let mut sum_sq = 0.0_f64;
            for col in 0..d {
                let diff = f64::from(x_row[col]) - f64::from(c_row[col]);
                sum_sq += diff * diff;
            }
            if !sum_sq.is_finite() {
                return Err(nonfinite_err());
            }
            if sum_sq < best_d {
                best_d = sum_sq;
                best_l = j as u32;
            }
        }
        let dist = (best_d.max(0.0_f64)) as f32;
        if !dist.is_finite() {
            return Err(nonfinite_err());
        }
        Ok((best_l, dist))
    }
}

fn compute_row_norms_checked(
    pool: &CpuPool,
    rows: &[f32],
    x_shared: Option<&Arc<Scratch<f32>>>,
    chunk_rows: usize,
    d: usize,
    budget: &Budget,
) -> Result<Scratch<f32>, OjasError> {
    let mut row_norm2 = Scratch::<f32>::try_alloc(chunk_rows, budget)?;
    if chunk_rows == 0 {
        return Ok(row_norm2);
    }

    let num_tiles = chunk_rows.div_ceil(ROW_TILE);
    if pool.threads() <= 1 || num_tiles <= 1 || x_shared.is_none() {
        for r in 0..chunk_rows {
            let x_row = &rows[r * d..(r + 1) * d];
            let mut rn = 0.0_f32;
            for &v in x_row {
                if !v.is_finite() {
                    return Err(nonfinite_err());
                }
                rn += v * v;
            }
            if !rn.is_finite() {
                return Err(nonfinite_err());
            }
            row_norm2.as_mut_slice()[r] = rn;
        }
        return Ok(row_norm2);
    }

    let x_arc = Arc::clone(x_shared.expect("checked is_some"));
    let tile_results = pool.run(num_tiles, move |t| -> Result<Vec<f32>, OjasError> {
        let r_start = t * ROW_TILE;
        let r_end = (r_start + ROW_TILE).min(chunk_rows);
        let mut out = Vec::with_capacity(r_end - r_start);
        let x_slice = x_arc.as_slice();
        for r in r_start..r_end {
            let x_row = &x_slice[r * d..(r + 1) * d];
            let mut rn = 0.0_f32;
            for &v in x_row {
                if !v.is_finite() {
                    return Err(nonfinite_err());
                }
                rn += v * v;
            }
            if !rn.is_finite() {
                return Err(nonfinite_err());
            }
            out.push(rn);
        }
        Ok(out)
    })?;

    let dst = row_norm2.as_mut_slice();
    let mut cursor = 0_usize;
    for tile in tile_results {
        let vals = tile?;
        dst[cursor..cursor + vals.len()].copy_from_slice(&vals);
        cursor += vals.len();
    }
    Ok(row_norm2)
}

struct ScoreSlotPool {
    slots: Mutex<Vec<Scratch<f32>>>,
}

struct ScoreSlotGuard<'a> {
    pool: &'a ScoreSlotPool,
    buf: Option<Scratch<f32>>,
}

impl ScoreSlotPool {
    fn new(slot_count: usize, slot_elems: usize, budget: &Budget) -> Result<Arc<Self>, OjasError> {
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(Scratch::<f32>::try_alloc(slot_elems, budget)?);
        }
        Ok(Arc::new(Self {
            slots: Mutex::new(slots),
        }))
    }

    fn checkout(&self) -> Result<ScoreSlotGuard<'_>, OjasError> {
        let buf = lock_unpoisoned(&self.slots)
            .pop()
            .ok_or_else(|| OjasError::Backend {
                id: BackendId::Cpu,
                detail: "score slot pool unexpectedly empty".to_string(),
            })?;
        Ok(ScoreSlotGuard {
            pool: self,
            buf: Some(buf),
        })
    }
}

impl Drop for ScoreSlotGuard<'_> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            lock_unpoisoned(&self.pool.slots).push(buf);
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn try_assign_accelerate_chunk(
    rows: &[f32],
    row_norm2: &[f32],
    chunk_rows: usize,
    d: usize,
    c_eff: &[f32],
    norm2_c: &[f32],
    k: usize,
    spherical: bool,
    labels_out: &mut [u32],
    dists_out: &mut [f32],
    budget: &Budget,
) -> Result<bool, OjasError> {
    let m_cap = chunk_rows.min(ACCEL_ROW_PANEL);
    let n_cap = k.min(ACCEL_COL_PANEL);
    let panel_elems = checked_mul_len(m_cap, n_cap, "Accelerate panel scores")?;
    let mut panel_scores = Scratch::<f32>::try_alloc(panel_elems, budget)?;
    let mut best_scores = Scratch::<f32>::try_extend(m_cap, budget, |v| {
        v.resize(m_cap, f32::INFINITY);
    })?;
    let mut best_labels = Scratch::<u32>::try_alloc(m_cap, budget)?;
    let mut row_had_nonfinite = Scratch::<u8>::try_alloc(m_cap, budget)?;

    let mut r0 = 0_usize;
    while r0 < chunk_rows {
        let m_cur = (chunk_rows - r0).min(ACCEL_ROW_PANEL);
        best_scores.as_mut_slice()[..m_cur].fill(f32::INFINITY);
        best_labels.as_mut_slice()[..m_cur].fill(0);
        row_had_nonfinite.as_mut_slice()[..m_cur].fill(0);

        let a_panel = &rows[r0 * d..(r0 + m_cur) * d];
        let mut c0 = 0_usize;
        while c0 < k {
            let n_cur = (k - c0).min(ACCEL_COL_PANEL);
            let b_panel = &c_eff[c0 * d..(c0 + n_cur) * d];
            let score_win = &mut panel_scores.as_mut_slice()[..m_cur * n_cur];

            if ojas_simd::sgemm_accelerate(
                m_cur, n_cur, d, a_panel, d, 1, b_panel, 1, d, score_win, n_cur, false,
            )
            .is_err()
            {
                return Ok(false);
            }

            let bs_slice = &mut best_scores.as_mut_slice()[..m_cur];
            let bl_slice = &mut best_labels.as_mut_slice()[..m_cur];
            let nf_slice = &mut row_had_nonfinite.as_mut_slice()[..m_cur];
            if spherical {
                for mr in 0..m_cur {
                    let row_dots = &score_win[mr * n_cur..(mr + 1) * n_cur];
                    let mut bs = bs_slice[mr];
                    let mut bl = bl_slice[mr];
                    let mut nf = nf_slice[mr];
                    for (nc, &dot) in row_dots.iter().enumerate() {
                        let sc = -2.0_f32 * dot;
                        if !dot.is_finite() || !sc.is_finite() {
                            nf = 1;
                            continue;
                        }
                        if sc < bs {
                            bs = sc;
                            bl = (c0 + nc) as u32;
                        }
                    }
                    bs_slice[mr] = bs;
                    bl_slice[mr] = bl;
                    nf_slice[mr] = nf;
                }
            } else {
                let cn_panel = &norm2_c[c0..c0 + n_cur];
                for mr in 0..m_cur {
                    let row_dots = &score_win[mr * n_cur..(mr + 1) * n_cur];
                    let mut bs = bs_slice[mr];
                    let mut bl = bl_slice[mr];
                    let mut nf = nf_slice[mr];
                    for (nc, (&dot, &cn)) in row_dots.iter().zip(cn_panel.iter()).enumerate() {
                        let sc = cn - 2.0_f32 * dot;
                        if !dot.is_finite() || !sc.is_finite() {
                            nf = 1;
                            continue;
                        }
                        if sc < bs {
                            bs = sc;
                            bl = (c0 + nc) as u32;
                        }
                    }
                    bs_slice[mr] = bs;
                    bl_slice[mr] = bl;
                    nf_slice[mr] = nf;
                }
            }

            c0 += n_cur;
        }

        let bs_slice = &best_scores.as_slice()[..m_cur];
        let bl_slice = &best_labels.as_slice()[..m_cur];
        let nf_slice = &row_had_nonfinite.as_slice()[..m_cur];
        for mr in 0..m_cur {
            let r = r0 + mr;
            let x_row = &rows[r * d..(r + 1) * d];
            if nf_slice[mr] != 0 {
                let (bl, dist) = assign_row_exact_f64(x_row, c_eff, k, d, spherical)?;
                labels_out[r] = bl;
                dists_out[r] = dist;
                continue;
            }
            let bl = bl_slice[mr];
            let dist = finalize_row_dist_f32(
                x_row,
                &c_eff[(bl as usize) * d..(bl as usize + 1) * d],
                row_norm2[r],
                bs_slice[mr],
                spherical,
            );
            if !dist.is_finite() {
                let (exact_bl, exact_dist) = assign_row_exact_f64(x_row, c_eff, k, d, spherical)?;
                labels_out[r] = exact_bl;
                dists_out[r] = exact_dist;
            } else {
                labels_out[r] = bl;
                dists_out[r] = dist;
            }
        }

        r0 += m_cur;
    }

    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn eval_row_tile(
    x_rows: &[f32],
    rn_tile: &[f32],
    m_tile: usize,
    d: usize,
    c_eff: &[f32],
    norm2_c: &[f32],
    k: usize,
    spherical: bool,
    use_fma_tile: bool,
    score_buf: &mut [f32],
) -> Result<Vec<(u32, f32)>, OjasError> {
    let mut best_s = [f32::INFINITY; ROW_TILE];
    let mut best_l = [0_u32; ROW_TILE];
    let mut row_had_nonfinite = [false; ROW_TILE];

    if use_fma_tile {
        let mut c0 = 0_usize;
        while c0 < k {
            let n_tile = (k - c0).min(C_TILE);
            let b_panel = &c_eff[c0 * d..(c0 + n_tile) * d];
            let scores = &mut score_buf[..m_tile * n_tile];
            ojas_simd::sgemm_tile(
                m_tile, n_tile, d, x_rows, d, 1, b_panel, 1, d, scores, n_tile, false,
            )
            .map_err(|err| OjasError::Backend {
                id: BackendId::Cpu,
                detail: format!("sgemm_tile failed in emeans assign: {err}"),
            })?;

            if spherical {
                for mr in 0..m_tile {
                    let row_dots = &scores[mr * n_tile..(mr + 1) * n_tile];
                    let mut bs = best_s[mr];
                    let mut bl = best_l[mr];
                    let mut nf = row_had_nonfinite[mr];
                    for (nc, &dot) in row_dots.iter().enumerate() {
                        let sc = -2.0_f32 * dot;
                        if !dot.is_finite() || !sc.is_finite() {
                            nf = true;
                            continue;
                        }
                        if sc < bs {
                            bs = sc;
                            bl = (c0 + nc) as u32;
                        }
                    }
                    best_s[mr] = bs;
                    best_l[mr] = bl;
                    row_had_nonfinite[mr] = nf;
                }
            } else {
                let cn_panel = &norm2_c[c0..c0 + n_tile];
                for mr in 0..m_tile {
                    let row_dots = &scores[mr * n_tile..(mr + 1) * n_tile];
                    let mut bs = best_s[mr];
                    let mut bl = best_l[mr];
                    let mut nf = row_had_nonfinite[mr];
                    for (nc, (&dot, &cn)) in row_dots.iter().zip(cn_panel.iter()).enumerate() {
                        let sc = cn - 2.0_f32 * dot;
                        if !dot.is_finite() || !sc.is_finite() {
                            nf = true;
                            continue;
                        }
                        if sc < bs {
                            bs = sc;
                            bl = (c0 + nc) as u32;
                        }
                    }
                    best_s[mr] = bs;
                    best_l[mr] = bl;
                    row_had_nonfinite[mr] = nf;
                }
            }
            c0 += n_tile;
        }
    } else {
        for mr in 0..m_tile {
            let x_row = &x_rows[mr * d..(mr + 1) * d];
            let mut bs = f32::INFINITY;
            let mut bl = 0_u32;
            let mut nf = false;
            for j in 0..k {
                let c_row = &c_eff[j * d..(j + 1) * d];
                let mut dot = 0.0_f32;
                for col in 0..d {
                    dot += x_row[col] * c_row[col];
                }
                let sc = if spherical {
                    -2.0_f32 * dot
                } else {
                    norm2_c[j] - 2.0_f32 * dot
                };
                if !dot.is_finite() || !sc.is_finite() {
                    nf = true;
                    continue;
                }
                if sc < bs {
                    bs = sc;
                    bl = j as u32;
                }
            }
            best_s[mr] = bs;
            best_l[mr] = bl;
            row_had_nonfinite[mr] = nf;
        }
    }

    let mut out = Vec::with_capacity(m_tile);
    for mr in 0..m_tile {
        let x_row = &x_rows[mr * d..(mr + 1) * d];
        if row_had_nonfinite[mr] {
            out.push(assign_row_exact_f64(x_row, c_eff, k, d, spherical)?);
            continue;
        }
        let bl = best_l[mr];
        let dist = finalize_row_dist_f32(
            x_row,
            &c_eff[(bl as usize) * d..(bl as usize + 1) * d],
            rn_tile[mr],
            best_s[mr],
            spherical,
        );
        if !dist.is_finite() {
            out.push(assign_row_exact_f64(x_row, c_eff, k, d, spherical)?);
        } else {
            out.push((bl, dist));
        }
    }
    Ok(out)
}

pub(crate) type AssignedChunk = (Scratch<u32>, Scratch<f32>, Option<Arc<Scratch<f32>>>);

/// Assign a validated chunk of `chunk_rows` rows (`rows.len() == chunk_rows * d`), returning
/// `Budget`-charged `(labels, row_dists)` and optional shared `Arc<Scratch<f32>>` of `rows` for
/// downstream parallel CSR sum-gathering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assign_chunk_validated(
    pool: &CpuPool,
    rows: &[f32],
    chunk_rows: usize,
    d: usize,
    c_eff: &Arc<Scratch<f32>>,
    norm2_c: &Arc<Scratch<f32>>,
    k: usize,
    math: CpuMath,
    spherical: bool,
    need_shared_rows_after: bool,
    budget: &Budget,
) -> Result<AssignedChunk, OjasError> {
    if k == 0 || k >= (UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "k ({k}) must be in 1..UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
        )));
    }
    if chunk_rows > (u32::MAX as usize) {
        return Err(out_of_range(format!(
            "chunk_rows ({chunk_rows}) exceeds u32::MAX"
        )));
    }
    let expected_len = checked_mul_len(chunk_rows, d, "chunk elements")?;
    if rows.len() != expected_len {
        return Err(shape_err(format!(
            "chunk slice len {} != chunk_rows ({chunk_rows}) * d ({d})",
            rows.len()
        )));
    }
    if chunk_rows == 0 {
        return Ok((
            Scratch::<u32>::try_alloc(0, budget)?,
            Scratch::<f32>::try_alloc(0, budget)?,
            None,
        ));
    }

    let num_row_tiles = chunk_rows.div_ceil(ROW_TILE);
    let want_parallel = pool.threads() > 1
        && (num_row_tiles > 1 || (need_shared_rows_after && k > CLUSTER_GATHER_TILE));
    let x_shared = if want_parallel {
        Some(Arc::new(Scratch::<f32>::try_from_slice(rows, budget)?))
    } else {
        None
    };

    let row_norm2 =
        compute_row_norms_checked(pool, rows, x_shared.as_ref(), chunk_rows, d, budget)?;
    let mut labels = Scratch::<u32>::try_alloc(chunk_rows, budget)?;
    let mut dists = Scratch::<f32>::try_alloc(chunk_rows, budget)?;

    #[cfg(target_os = "macos")]
    if math == CpuMath::Fast && !pool.is_explicit_threads() {
        let used_accel = try_assign_accelerate_chunk(
            rows,
            row_norm2.as_slice(),
            chunk_rows,
            d,
            c_eff.as_slice(),
            norm2_c.as_slice(),
            k,
            spherical,
            labels.as_mut_slice(),
            dists.as_mut_slice(),
            budget,
        )?;
        if used_accel {
            return Ok((labels, dists, x_shared));
        }
    }

    let use_fma_tile = matches!(math, CpuMath::Fast | CpuMath::DeterministicFma);
    let score_slot_elems = if use_fma_tile {
        ROW_TILE * k.min(C_TILE)
    } else {
        0
    };

    if pool.threads() <= 1 || num_row_tiles <= 1 {
        let mut single_score = Scratch::<f32>::try_alloc(score_slot_elems, budget)?;
        for t in 0..num_row_tiles {
            let r_start = t * ROW_TILE;
            let r_end = (r_start + ROW_TILE).min(chunk_rows);
            let m_tile = r_end - r_start;
            let pairs = eval_row_tile(
                &rows[r_start * d..r_end * d],
                &row_norm2.as_slice()[r_start..r_end],
                m_tile,
                d,
                c_eff.as_slice(),
                norm2_c.as_slice(),
                k,
                spherical,
                use_fma_tile,
                single_score.as_mut_slice(),
            )?;
            for (mr, (lbl, dist)) in pairs.into_iter().enumerate() {
                labels.as_mut_slice()[r_start + mr] = lbl;
                dists.as_mut_slice()[r_start + mr] = dist;
            }
        }
        return Ok((labels, dists, x_shared));
    }

    let slot_count = pool.threads().min(num_row_tiles);
    let slot_pool = ScoreSlotPool::new(slot_count, score_slot_elems, budget)?;
    let x_arc = Arc::clone(x_shared.as_ref().expect("allocated when parallel"));
    let rn_arc = Arc::new(row_norm2);
    let c_arc = Arc::clone(c_eff);
    let cn_arc = Arc::clone(norm2_c);

    let tile_outputs = pool.run(
        num_row_tiles,
        move |t| -> Result<Vec<(u32, f32)>, OjasError> {
            let r_start = t * ROW_TILE;
            let r_end = (r_start + ROW_TILE).min(chunk_rows);
            let m_tile = r_end - r_start;
            let mut slot = slot_pool.checkout()?;
            let score_slice = slot.buf.as_mut().expect("checked out").as_mut_slice();
            eval_row_tile(
                &x_arc.as_slice()[r_start * d..r_end * d],
                &rn_arc.as_slice()[r_start..r_end],
                m_tile,
                d,
                c_arc.as_slice(),
                cn_arc.as_slice(),
                k,
                spherical,
                use_fma_tile,
                score_slice,
            )
        },
    )?;

    for (t, tile_res) in tile_outputs.into_iter().enumerate() {
        let pairs = tile_res?;
        let r_start = t * ROW_TILE;
        for (mr, (lbl, dist)) in pairs.into_iter().enumerate() {
            labels.as_mut_slice()[r_start + mr] = lbl;
            dists.as_mut_slice()[r_start + mr] = dist;
        }
    }

    Ok((labels, dists, x_shared))
}

/// Build deterministic CSR `(offsets[0..=k], csr_rows[0..chunk_rows])` where every cluster's row
/// indices appear in strictly ascending `0..chunk_rows` order. Uses parallel per-tile histograms
/// reduced in fixed ascending tile order when `chunk_rows >= 4096`.
pub(crate) fn build_csr_ascending(
    pool: &CpuPool,
    labels: &Scratch<u32>,
    k: usize,
    budget: &Budget,
) -> Result<(Scratch<u32>, Scratch<u32>), OjasError> {
    if k == 0 || k >= (UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "k ({k}) must be in 1..UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
        )));
    }
    let chunk_rows = labels.len();
    if chunk_rows > (u32::MAX as usize) {
        return Err(out_of_range(format!(
            "chunk_rows ({chunk_rows}) exceeds u32::MAX"
        )));
    }
    let offsets_len = k
        .checked_add(1)
        .ok_or_else(|| out_of_range("k + 1 overflows usize"))?;
    let mut offsets = Scratch::<u32>::try_alloc(offsets_len, budget)?;
    let mut csr_rows = Scratch::<u32>::try_alloc(chunk_rows, budget)?;

    if chunk_rows == 0 {
        return Ok((offsets, csr_rows));
    }

    if chunk_rows >= PARALLEL_CSR_MIN_ROWS && pool.threads() > 1 {
        let num_tiles = chunk_rows.div_ceil(CSR_HIST_TILE);
        let hist_total = checked_mul_len(num_tiles, k, "CSR tile histograms")?;
        // Reserve the histogram footprint upfront in `budget` so tight budgets fail closed.
        let _hist_charge = Scratch::<u32>::try_alloc(hist_total, budget)?;
        let labels_arc = Arc::new(Scratch::<u32>::try_from_slice(labels.as_slice(), budget)?);
        let tile_hists = pool.run(num_tiles, move |t| -> Vec<u32> {
            let start = t * CSR_HIST_TILE;
            let end = (start + CSR_HIST_TILE).min(chunk_rows);
            let mut h = vec![0_u32; k];
            for &lbl in &labels_arc.as_slice()[start..end] {
                h[lbl as usize] += 1;
            }
            h
        })?;
        drop(_hist_charge);

        // Reduce per-tile histograms in fixed ascending tile order `0..num_tiles`.
        let off_slice = offsets.as_mut_slice();
        for h in &tile_hists {
            for (j, &cnt) in h.iter().enumerate() {
                off_slice[j + 1] += cnt;
            }
        }
        for j in 0..k {
            off_slice[j + 1] += off_slice[j];
        }
    } else {
        let off_slice = offsets.as_mut_slice();
        for &lbl in labels.as_slice() {
            off_slice[(lbl as usize) + 1] += 1;
        }
        for j in 0..k {
            off_slice[j + 1] += off_slice[j];
        }
    }

    let mut cursors = Scratch::<u32>::try_from_slice(&offsets.as_slice()[..k], budget)?;
    let cur_slice = cursors.as_mut_slice();
    let row_slice = csr_rows.as_mut_slice();
    for (r, &lbl) in labels.as_slice().iter().enumerate() {
        let pos = cur_slice[lbl as usize] as usize;
        row_slice[pos] = r as u32;
        cur_slice[lbl as usize] += 1;
    }

    Ok((offsets, csr_rows))
}

/// Accumulate `f64` cluster sums in strictly ascending row order across disjoint cluster blocks
/// on `pool` (reused by `lloyd_fit`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_csr_sums_f64(
    pool: &CpuPool,
    rows: &[f32],
    x_shared: Option<Arc<Scratch<f32>>>,
    offsets: Scratch<u32>,
    csr_rows: Scratch<u32>,
    k: usize,
    d: usize,
    pass_sums_f64: &mut [f64],
    pass_counts: &mut [u64],
    budget: &Budget,
) -> Result<(), OjasError> {
    let off = offsets.as_slice();
    for j in 0..k {
        let n_j = u64::from(off[j + 1] - off[j]);
        pass_counts[j] += n_j;
    }

    let num_c_tasks = k.div_ceil(CLUSTER_GATHER_TILE);
    if pool.threads() <= 1 || num_c_tasks <= 1 {
        let csr = csr_rows.as_slice();
        for j in 0..k {
            let start = off[j] as usize;
            let end = off[j + 1] as usize;
            if start == end {
                continue;
            }
            let dst = &mut pass_sums_f64[j * d..(j + 1) * d];
            for &r_u32 in &csr[start..end] {
                let x_row = &rows[(r_u32 as usize) * d..(r_u32 as usize + 1) * d];
                for (acc, &xv) in dst.iter_mut().zip(x_row.iter()) {
                    *acc += f64::from(xv);
                }
            }
        }
        return Ok(());
    }

    let x_arc = match x_shared {
        Some(arc) => arc,
        None => Arc::new(Scratch::<f32>::try_from_slice(rows, budget)?),
    };
    let off_arc = Arc::new(offsets);
    let csr_arc = Arc::new(csr_rows);
    let prev_arc = Arc::new(Scratch::<f64>::try_from_slice(pass_sums_f64, budget)?);

    let blocks = pool.run(num_c_tasks, move |task_idx| -> Vec<f64> {
        let j_start = task_idx * CLUSTER_GATHER_TILE;
        let j_end = (j_start + CLUSTER_GATHER_TILE).min(k);
        let mut block = prev_arc.as_slice()[j_start * d..j_end * d].to_vec();
        let off_s = off_arc.as_slice();
        let csr_s = csr_arc.as_slice();
        let x_s = x_arc.as_slice();

        for j in j_start..j_end {
            let start = off_s[j] as usize;
            let end = off_s[j + 1] as usize;
            if start == end {
                continue;
            }
            let local_dst = &mut block[(j - j_start) * d..(j - j_start + 1) * d];
            for &r_u32 in &csr_s[start..end] {
                let x_row = &x_s[(r_u32 as usize) * d..(r_u32 as usize + 1) * d];
                for (acc, &xv) in local_dst.iter_mut().zip(x_row.iter()) {
                    *acc += f64::from(xv);
                }
            }
        }
        block
    })?;

    for (task_idx, block) in blocks.into_iter().enumerate() {
        let j_start = task_idx * CLUSTER_GATHER_TILE;
        let j_end = (j_start + CLUSTER_GATHER_TILE).min(k);
        pass_sums_f64[j_start * d..j_end * d].copy_from_slice(&block);
    }
    Ok(())
}

// =============================================================================
// Reusable Streaming CPU `e-means` State Machine (`CpuSession`)
// =============================================================================

/// Reusable streaming CPU `e-means` state machine caching `c_eff` and `norm2_c` across chunks.
pub struct CpuSession {
    k: usize,
    d: usize,
    spherical: bool,
    unused_threshold: f32,
    split_epsilon: f32,
    seed: u64,
    final_epoch: bool,
    centroids: Scratch<f32>,
    c_eff: Arc<Scratch<f32>>,
    norm2_c: Arc<Scratch<f32>>,
    usage: Scratch<f32>,
    batch_sums: Scratch<f32>,
    batch_counts: Scratch<u64>,
    epoch_counts: Scratch<u64>,
    pass_sums: Scratch<f64>,
    pass_counts: Scratch<u64>,
    split_scratch: Scratch<f32>,
    closing_row_f64: Scratch<f64>,
    epoch_objective: f64,
}

impl CpuSession {
    /// Initialize a CPU `e-means` session and precompute effective centroids and squared norms.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: usize,
        d: usize,
        spherical: bool,
        unused_threshold: f32,
        split_epsilon: f32,
        init_usage: f32,
        seed: u64,
        init_centroids: &[f32],
        budget: &Budget,
    ) -> Result<Self, OjasError> {
        if k == 0 {
            return Err(out_of_range("k must be >= 1"));
        }
        if k >= (UNASSIGNED_LABEL as usize) {
            return Err(out_of_range(format!(
                "k ({k}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
            )));
        }
        if d == 0 {
            return Err(out_of_range("d must be >= 1"));
        }
        if !unused_threshold.is_finite() || unused_threshold < 0.0 {
            return Err(out_of_range("unused_threshold must be finite and >= 0.0"));
        }
        if !split_epsilon.is_finite() || split_epsilon < 0.0 {
            return Err(out_of_range("split_epsilon must be finite and >= 0.0"));
        }
        if !init_usage.is_finite() || init_usage < 0.0 {
            return Err(out_of_range("init_usage must be finite and >= 0.0"));
        }
        let kd = checked_mul_len(k, d, "k * d")?;
        if init_centroids.len() != kd {
            return Err(shape_err(format!(
                "init_centroids len {} != k * d ({kd})",
                init_centroids.len()
            )));
        }

        let mut centroids = Scratch::<f32>::try_from_slice(init_centroids, budget)?;
        if spherical {
            let mut row_tmp = Scratch::<f32>::try_alloc(d, budget)?;
            for j in 0..k {
                let row = &mut centroids.as_mut_slice()[j * d..(j + 1) * d];
                row_tmp.as_mut_slice().copy_from_slice(row);
                prescaled_unit_normalize_f32(row_tmp.as_slice(), row);
                if row.iter().any(|v| !v.is_finite()) {
                    return Err(nonfinite_err());
                }
            }
        }

        let mut c_eff_buf = Scratch::<f32>::try_alloc(kd, budget)?;
        let mut norm2_c_buf = Scratch::<f32>::try_alloc(k, budget)?;
        prepare_centroids_into(
            centroids.as_slice(),
            k,
            d,
            spherical,
            c_eff_buf.as_mut_slice(),
            norm2_c_buf.as_mut_slice(),
        )?;

        let usage = Scratch::<f32>::try_extend(k, budget, |v| {
            v.resize(k, init_usage);
        })?;
        let batch_sums = Scratch::<f32>::try_alloc(kd, budget)?;
        let batch_counts = Scratch::<u64>::try_alloc(k, budget)?;
        let epoch_counts = Scratch::<u64>::try_alloc(k, budget)?;
        let pass_sums = Scratch::<f64>::try_alloc(kd, budget)?;
        let pass_counts = Scratch::<u64>::try_alloc(k, budget)?;
        let split_scratch =
            Scratch::<f32>::try_alloc(checked_mul_len(2, d, "split scratch")?, budget)?;
        let closing_row_f64 = Scratch::<f64>::try_alloc(d, budget)?;

        Ok(Self {
            k,
            d,
            spherical,
            unused_threshold,
            split_epsilon,
            seed,
            final_epoch: false,
            centroids,
            c_eff: Arc::new(c_eff_buf),
            norm2_c: Arc::new(norm2_c_buf),
            usage,
            batch_sums,
            batch_counts,
            epoch_counts,
            pass_sums,
            pass_counts,
            split_scratch,
            closing_row_f64,
            epoch_objective: 0.0,
        })
    }

    /// Number of clusters `k`.
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Feature dimension `d`.
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Whether cosine/spherical projection is active.
    #[inline]
    pub fn spherical(&self) -> bool {
        self.spherical
    }

    fn refresh_prepared_centroids(&mut self) -> Result<(), OjasError> {
        let ceff_mut = Arc::get_mut(&mut self.c_eff).ok_or_else(|| OjasError::Backend {
            id: BackendId::Cpu,
            detail: "CpuSession::c_eff still shared during centroid refresh".to_string(),
        })?;
        let norm2_mut = Arc::get_mut(&mut self.norm2_c).ok_or_else(|| OjasError::Backend {
            id: BackendId::Cpu,
            detail: "CpuSession::norm2_c still shared during centroid refresh".to_string(),
        })?;
        prepare_centroids_into(
            self.centroids.as_slice(),
            self.k,
            self.d,
            self.spherical,
            ceff_mut.as_mut_slice(),
            norm2_mut.as_mut_slice(),
        )
    }

    /// Reset per-epoch objective and counts (and final-pass closing accumulators when `final_epoch`).
    pub fn reset_pass_accumulators(&mut self, final_epoch: bool) {
        self.final_epoch = final_epoch;
        self.epoch_objective = 0.0;
        self.batch_sums.as_mut_slice().fill(0.0);
        self.batch_counts.as_mut_slice().fill(0);
        self.epoch_counts.as_mut_slice().fill(0);
        if final_epoch {
            self.pass_sums.as_mut_slice().fill(0.0);
            self.pass_counts.as_mut_slice().fill(0);
        }
    }

    /// Assign and accumulate an `f32` row chunk into the current mini-batch and epoch totals.
    ///
    /// Validates every row coordinate and squared norm **before** mutating session accumulators.
    pub fn accumulate_f32_chunk(
        &mut self,
        pool: &CpuPool,
        rows: &[f32],
        chunk_rows: usize,
        math: CpuMath,
        labels_out: Option<&mut [u32]>,
        budget: &Budget,
    ) -> Result<(), OjasError> {
        if chunk_rows > (u32::MAX as usize) {
            return Err(out_of_range(format!(
                "chunk_rows ({chunk_rows}) exceeds u32::MAX"
            )));
        }
        if let Some(ref out) = labels_out {
            if out.len() != chunk_rows {
                return Err(shape_err(format!(
                    "labels_out len {} != chunk_rows {chunk_rows}",
                    out.len()
                )));
            }
        }
        if chunk_rows == 0 {
            return Ok(());
        }

        let (labels, dists, x_shared) = assign_chunk_validated(
            pool,
            rows,
            chunk_rows,
            self.d,
            &self.c_eff,
            &self.norm2_c,
            self.k,
            math,
            self.spherical,
            true,
            budget,
        )?;

        let (offsets, csr_rows) = build_csr_ascending(pool, &labels, self.k, budget)?;

        // Gather `batch_sums` (and `pass_sums` on `final_epoch`) in strictly ascending row order.
        let k = self.k;
        let d = self.d;
        let final_epoch = self.final_epoch;
        let num_c_tasks = k.div_ceil(CLUSTER_GATHER_TILE);

        if pool.threads() <= 1 || num_c_tasks <= 1 {
            let off_s = offsets.as_slice();
            let csr_s = csr_rows.as_slice();
            let b_sums = self.batch_sums.as_mut_slice();
            let p_sums = self.pass_sums.as_mut_slice();
            for j in 0..k {
                let start = off_s[j] as usize;
                let end = off_s[j + 1] as usize;
                if start == end {
                    continue;
                }
                let b_row = &mut b_sums[j * d..(j + 1) * d];
                if final_epoch {
                    let p_row = &mut p_sums[j * d..(j + 1) * d];
                    for &r_u32 in &csr_s[start..end] {
                        let x_row = &rows[(r_u32 as usize) * d..(r_u32 as usize + 1) * d];
                        for col in 0..d {
                            let xv = x_row[col];
                            b_row[col] += xv;
                            p_row[col] += f64::from(xv);
                        }
                    }
                } else {
                    for &r_u32 in &csr_s[start..end] {
                        let x_row = &rows[(r_u32 as usize) * d..(r_u32 as usize + 1) * d];
                        for col in 0..d {
                            b_row[col] += x_row[col];
                        }
                    }
                }
            }
        } else {
            let x_arc = match x_shared {
                Some(arc) => arc,
                None => Arc::new(Scratch::<f32>::try_from_slice(rows, budget)?),
            };
            let off_arc = Arc::new(offsets);
            let off_for_tasks = Arc::clone(&off_arc);
            let csr_arc = Arc::new(csr_rows);
            let prev_b = Arc::new(Scratch::<f32>::try_from_slice(
                self.batch_sums.as_slice(),
                budget,
            )?);
            let prev_p = if final_epoch {
                Some(Arc::new(Scratch::<f64>::try_from_slice(
                    self.pass_sums.as_slice(),
                    budget,
                )?))
            } else {
                None
            };

            let gathered = pool.run(num_c_tasks, move |task_idx| -> (Vec<f32>, Vec<f64>) {
                let j_start = task_idx * CLUSTER_GATHER_TILE;
                let j_end = (j_start + CLUSTER_GATHER_TILE).min(k);
                let mut b_block = prev_b.as_slice()[j_start * d..j_end * d].to_vec();
                let mut p_block = match &prev_p {
                    Some(p) => p.as_slice()[j_start * d..j_end * d].to_vec(),
                    None => Vec::new(),
                };
                let off_s = off_for_tasks.as_slice();
                let csr_s = csr_arc.as_slice();
                let x_s = x_arc.as_slice();

                for j in j_start..j_end {
                    let start = off_s[j] as usize;
                    let end = off_s[j + 1] as usize;
                    if start == end {
                        continue;
                    }
                    let loc = (j - j_start) * d;
                    let b_row = &mut b_block[loc..loc + d];
                    if final_epoch {
                        let p_row = &mut p_block[loc..loc + d];
                        for &r_u32 in &csr_s[start..end] {
                            let x_row = &x_s[(r_u32 as usize) * d..(r_u32 as usize + 1) * d];
                            for col in 0..d {
                                let xv = x_row[col];
                                b_row[col] += xv;
                                p_row[col] += f64::from(xv);
                            }
                        }
                    } else {
                        for &r_u32 in &csr_s[start..end] {
                            let x_row = &x_s[(r_u32 as usize) * d..(r_u32 as usize + 1) * d];
                            for col in 0..d {
                                b_row[col] += x_row[col];
                            }
                        }
                    }
                }
                (b_block, p_block)
            })?;

            // Update counts from `off_arc` before committing block sums.
            let off_s = off_arc.as_slice();
            for j in 0..k {
                let n_j = u64::from(off_s[j + 1] - off_s[j]);
                if n_j > 0 {
                    self.batch_counts.as_mut_slice()[j] += n_j;
                    self.epoch_counts.as_mut_slice()[j] += n_j;
                    if final_epoch {
                        self.pass_counts.as_mut_slice()[j] += n_j;
                    }
                }
            }
            for (task_idx, (b_block, p_block)) in gathered.into_iter().enumerate() {
                let j_start = task_idx * CLUSTER_GATHER_TILE;
                let j_end = (j_start + CLUSTER_GATHER_TILE).min(k);
                self.batch_sums.as_mut_slice()[j_start * d..j_end * d].copy_from_slice(&b_block);
                if final_epoch {
                    self.pass_sums.as_mut_slice()[j_start * d..j_end * d].copy_from_slice(&p_block);
                }
            }
            for &dist in dists.as_slice() {
                self.epoch_objective += f64::from(dist);
            }
            if let Some(out) = labels_out {
                out.copy_from_slice(labels.as_slice());
            }
            return Ok(());
        }

        // Commit counts, objective, and optional labels for the inline gather path.
        let off_s = offsets.as_slice();
        for j in 0..k {
            let n_j = u64::from(off_s[j + 1] - off_s[j]);
            if n_j > 0 {
                self.batch_counts.as_mut_slice()[j] += n_j;
                self.epoch_counts.as_mut_slice()[j] += n_j;
                if final_epoch {
                    self.pass_counts.as_mut_slice()[j] += n_j;
                }
            }
        }
        for &dist in dists.as_slice() {
            self.epoch_objective += f64::from(dist);
        }
        if let Some(out) = labels_out {
            out.copy_from_slice(labels.as_slice());
        }
        Ok(())
    }

    /// Assign and accumulate a raw `bf16` row chunk into the current mini-batch and epoch totals.
    pub fn accumulate_bf16_chunk(
        &mut self,
        pool: &CpuPool,
        rows_bf16: &[u16],
        chunk_rows: usize,
        math: CpuMath,
        labels_out: Option<&mut [u32]>,
        budget: &Budget,
    ) -> Result<(), OjasError> {
        if chunk_rows > (u32::MAX as usize) {
            return Err(out_of_range(format!(
                "chunk_rows ({chunk_rows}) exceeds u32::MAX"
            )));
        }
        let expected = checked_mul_len(chunk_rows, self.d, "bf16 chunk elements")?;
        if rows_bf16.len() != expected {
            return Err(shape_err(format!(
                "bf16 chunk len {} != chunk_rows ({chunk_rows}) * d ({})",
                rows_bf16.len(),
                self.d
            )));
        }
        let decoded = decode_bf16_to_scratch(rows_bf16, budget)?;
        self.accumulate_f32_chunk(
            pool,
            decoded.as_slice(),
            chunk_rows,
            math,
            labels_out,
            budget,
        )
    }

    /// Assign and accumulate a mapped native-endian `bf16` byte-pair row chunk into the current
    /// mini-batch and epoch totals.
    pub fn accumulate_bf16_bytes_chunk(
        &mut self,
        pool: &CpuPool,
        rows_bf16: &[[u8; 2]],
        chunk_rows: usize,
        math: CpuMath,
        labels_out: Option<&mut [u32]>,
        budget: &Budget,
    ) -> Result<(), OjasError> {
        if chunk_rows > (u32::MAX as usize) {
            return Err(out_of_range(format!(
                "chunk_rows ({chunk_rows}) exceeds u32::MAX"
            )));
        }
        let expected = checked_mul_len(chunk_rows, self.d, "mapped bf16 chunk elements")?;
        if rows_bf16.len() != expected {
            return Err(shape_err(format!(
                "mapped bf16 chunk len {} != chunk_rows ({chunk_rows}) * d ({})",
                rows_bf16.len(),
                self.d
            )));
        }
        let decoded = decode_bf16_bytes_to_scratch(rows_bf16, budget)?;
        self.accumulate_f32_chunk(
            pool,
            decoded.as_slice(),
            chunk_rows,
            math,
            labels_out,
            budget,
        )
    }

    /// Apply EMA centroid update (`weight == 1.0` exact override) and optional Gumbel split revival,
    /// then refresh cached `c_eff` and `norm2_c`.
    pub fn apply_batch_update(
        &mut self,
        alpha: f32,
        step: u64,
        check_revive: bool,
    ) -> Result<usize, OjasError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(out_of_range(format!(
                "alpha ({alpha}) must be finite and in [0.0, 1.0]"
            )));
        }

        let k = self.k;
        let d = self.d;
        let replace = alpha == 1.0;

        {
            let cents = self.centroids.as_mut_slice();
            let usage = self.usage.as_mut_slice();
            let b_sums = self.batch_sums.as_mut_slice();
            let b_counts = self.batch_counts.as_mut_slice();

            for j in 0..k {
                let n_j = b_counts[j];
                let n_old = usage[j];
                let c_row = &mut cents[j * d..(j + 1) * d];
                let s_row = &mut b_sums[j * d..(j + 1) * d];

                if n_j > 0 {
                    let nf = n_j as f32;
                    let n_new = if replace {
                        alpha * nf
                    } else {
                        (1.0 - alpha) * n_old + alpha * nf
                    };
                    let w = if n_new > 0.0 {
                        (alpha * nf) / n_new
                    } else {
                        0.0
                    };
                    let exact_mean = w == 1.0;
                    let keep_old = w == 0.0;

                    for col in 0..d {
                        let s_val = s_row[col];
                        s_row[col] = 0.0;
                        let mean = s_val / nf;
                        let c_old = c_row[col];
                        c_row[col] = if exact_mean {
                            mean
                        } else if keep_old {
                            c_old
                        } else {
                            c_old + w * (mean - c_old)
                        };
                    }
                    usage[j] = n_new;
                    b_counts[j] = 0;
                } else {
                    s_row.fill(0.0);
                    usage[j] = if replace { 0.0 } else { (1.0 - alpha) * n_old };
                    b_counts[j] = 0;
                }
            }
        }

        let mut revived = 0_usize;
        if check_revive && self.unused_threshold > 0.0 {
            let splits = choose_splits(
                self.usage.as_slice(),
                self.unused_threshold,
                self.seed,
                step,
            );
            revived = splits.len();
            if revived > 0 {
                let eps = self.split_epsilon;
                let (d_slice, c_slice) = self.split_scratch.as_mut_slice().split_at_mut(d);
                let cents = self.centroids.as_mut_slice();
                let usage = self.usage.as_mut_slice();

                for (donor_u32, target_u32) in splits {
                    let donor = donor_u32 as usize;
                    let target = target_u32 as usize;
                    if donor >= k || target >= k || donor == target {
                        continue;
                    }
                    for col in 0..d {
                        let c = cents[donor * d + col];
                        let off = if col % 2 == 0 { eps } else { -eps };
                        d_slice[col] = c * (1.0 - off);
                        c_slice[col] = c * (1.0 + off);
                    }
                    separate_collapsed_split_f32(d_slice, c_slice);
                    cents[donor * d..(donor + 1) * d].copy_from_slice(d_slice);
                    cents[target * d..(target + 1) * d].copy_from_slice(c_slice);

                    let half_n = 0.5 * usage[donor];
                    usage[donor] = half_n;
                    usage[target] = half_n;
                }
            }
        }

        self.refresh_prepared_centroids()?;
        Ok(revived)
    }

    /// Return `(epoch_objective_sum, epoch_cluster_counts)` at the end of an epoch.
    pub fn finish_epoch(&mut self) -> (f64, Vec<u64>) {
        (self.epoch_objective, self.epoch_counts.as_slice().to_vec())
    }

    /// Borrow raw internal `f32` EMA centroid positions `[k * d]`.
    #[inline]
    pub fn read_centroids(&self) -> &[f32] {
        self.centroids.as_slice()
    }

    /// Borrow effective `f32` centroids (`unit_centroids` when `spherical`, else raw positions).
    #[inline]
    pub fn effective_centroids(&self) -> &[f32] {
        self.c_eff.as_slice()
    }

    /// Borrow current `f32` EMA cluster usage masses `[k]`.
    #[inline]
    pub fn read_usage(&self) -> &[f32] {
        self.usage.as_slice()
    }

    /// Compute `f64` last-epoch closing means (falling back to `_centroids[j]` where
    /// `pass_counts[j] == 0`, and normalizing via `prescaled_unit_normalize_f64` when `spherical`),
    /// updating `self.centroids` and cached `c_eff`/`norm2_c`.
    pub fn finalize_closing_centroids(&mut self) -> Result<Vec<f32>, OjasError> {
        let k = self.k;
        let d = self.d;
        let mut out = vec![0.0_f32; k * d];

        {
            let sums = self.pass_sums.as_slice();
            let counts = self.pass_counts.as_slice();
            let cur_cents = self.centroids.as_slice();
            let rf = self.closing_row_f64.as_mut_slice();

            for j in 0..k {
                let cnt = counts[j];
                if cnt > 0 {
                    let inv = 1.0 / (cnt as f64);
                    for col in 0..d {
                        rf[col] = sums[j * d + col] * inv;
                    }
                } else {
                    for col in 0..d {
                        rf[col] = f64::from(cur_cents[j * d + col]);
                    }
                }

                let dst = &mut out[j * d..(j + 1) * d];
                if self.spherical {
                    prescaled_unit_normalize_f64(rf, dst);
                    if dst.iter().any(|v| !v.is_finite()) {
                        return Err(nonfinite_err());
                    }
                } else {
                    for col in 0..d {
                        let v = rf[col] as f32;
                        if !v.is_finite() {
                            return Err(nonfinite_err());
                        }
                        dst[col] = v;
                    }
                }
            }
        }

        self.centroids.as_mut_slice().copy_from_slice(&out);
        self.refresh_prepared_centroids()?;
        Ok(out)
    }
}

// =============================================================================
// Standalone CPU Entry Points (`assign_cpu_f32`, `assign_cpu_bf16`, `fit_cpu`)
// =============================================================================

/// Assign `n` `f32` rows `[n, d]` to nearest `centroids` `[k, d]`, returning `(labels, total_objective)`.
#[allow(clippy::too_many_arguments)]
pub fn assign_cpu_f32(
    pool: &CpuPool,
    rows: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    chunk_rows: usize,
    math: CpuMath,
    spherical: bool,
    budget: &Budget,
) -> Result<(Vec<u32>, f64), OjasError> {
    if k == 0 {
        return Err(out_of_range("k must be >= 1"));
    }
    if k >= (UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "k ({k}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
        )));
    }
    if d == 0 {
        return Err(out_of_range("d must be >= 1"));
    }
    if chunk_rows == 0 {
        return Err(out_of_range("chunk_rows must be >= 1"));
    }
    let expected_rows = checked_mul_len(n, d, "rows len")?;
    if rows.len() != expected_rows {
        return Err(shape_err(format!(
            "rows len {} != n ({n}) * d ({d})",
            rows.len()
        )));
    }
    let kd = checked_mul_len(k, d, "centroids len")?;
    if centroids.len() != kd {
        return Err(shape_err(format!(
            "centroids len {} != k ({k}) * d ({d})",
            centroids.len()
        )));
    }

    let mut c_eff_buf = Scratch::<f32>::try_alloc(kd, budget)?;
    let mut norm2_c_buf = Scratch::<f32>::try_alloc(k, budget)?;
    prepare_centroids_into(
        centroids,
        k,
        d,
        spherical,
        c_eff_buf.as_mut_slice(),
        norm2_c_buf.as_mut_slice(),
    )?;
    let c_eff = Arc::new(c_eff_buf);
    let norm2_c = Arc::new(norm2_c_buf);

    let mut all_labels = vec![0_u32; n];
    let mut total_obj = 0.0_f64;
    let mut r0 = 0_usize;
    while r0 < n {
        let m_cur = (n - r0).min(chunk_rows);
        let (lbl_chunk, dist_chunk, _) = assign_chunk_validated(
            pool,
            &rows[r0 * d..(r0 + m_cur) * d],
            m_cur,
            d,
            &c_eff,
            &norm2_c,
            k,
            math,
            spherical,
            false,
            budget,
        )?;
        all_labels[r0..r0 + m_cur].copy_from_slice(lbl_chunk.as_slice());
        for &dist in dist_chunk.as_slice() {
            total_obj += f64::from(dist);
        }
        r0 += m_cur;
    }

    Ok((all_labels, total_obj))
}

/// Assign `n` `bf16` rows `[n, d]` to nearest `centroids` `[k, d]`, returning `(labels, total_objective)`.
#[allow(clippy::too_many_arguments)]
pub fn assign_cpu_bf16(
    pool: &CpuPool,
    rows_bf16: &[u16],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    chunk_rows: usize,
    math: CpuMath,
    spherical: bool,
    budget: &Budget,
) -> Result<(Vec<u32>, f64), OjasError> {
    if k == 0 {
        return Err(out_of_range("k must be >= 1"));
    }
    if k >= (UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "k ({k}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
        )));
    }
    if d == 0 {
        return Err(out_of_range("d must be >= 1"));
    }
    if chunk_rows == 0 {
        return Err(out_of_range("chunk_rows must be >= 1"));
    }
    let expected_rows = checked_mul_len(n, d, "bf16 rows len")?;
    if rows_bf16.len() != expected_rows {
        return Err(shape_err(format!(
            "bf16 rows len {} != n ({n}) * d ({d})",
            rows_bf16.len()
        )));
    }
    let kd = checked_mul_len(k, d, "centroids len")?;
    if centroids.len() != kd {
        return Err(shape_err(format!(
            "centroids len {} != k ({k}) * d ({d})",
            centroids.len()
        )));
    }

    let mut c_eff_buf = Scratch::<f32>::try_alloc(kd, budget)?;
    let mut norm2_c_buf = Scratch::<f32>::try_alloc(k, budget)?;
    prepare_centroids_into(
        centroids,
        k,
        d,
        spherical,
        c_eff_buf.as_mut_slice(),
        norm2_c_buf.as_mut_slice(),
    )?;
    let c_eff = Arc::new(c_eff_buf);
    let norm2_c = Arc::new(norm2_c_buf);

    let chunk_cap = n.min(chunk_rows);
    let mut decoded_win =
        Scratch::<f32>::try_alloc(checked_mul_len(chunk_cap, d, "bf16 staging")?, budget)?;
    let mut all_labels = vec![0_u32; n];
    let mut total_obj = 0.0_f64;

    let mut r0 = 0_usize;
    while r0 < n {
        let m_cur = (n - r0).min(chunk_rows);
        let win_slice = &mut decoded_win.as_mut_slice()[..m_cur * d];
        decode_bf16_row_checked(&rows_bf16[r0 * d..(r0 + m_cur) * d], win_slice)?;
        let (lbl_chunk, dist_chunk, _) = assign_chunk_validated(
            pool, win_slice, m_cur, d, &c_eff, &norm2_c, k, math, spherical, false, budget,
        )?;
        all_labels[r0..r0 + m_cur].copy_from_slice(lbl_chunk.as_slice());
        for &dist in dist_chunk.as_slice() {
            total_obj += f64::from(dist);
        }
        r0 += m_cur;
    }

    Ok((all_labels, total_obj))
}

/// Assign any 2D `Host` or `Mapped` (`SharedBuf`) `F32`/`Bf16` tensor `x` `[n, d]` against
/// `centroids` `[k, d]`, returning `(labels, total_objective)`.
pub fn assign_cpu_tensor(
    pool: &CpuPool,
    x: &Tensor,
    centroids: &Tensor,
    chunk_rows: usize,
    math: CpuMath,
    spherical: bool,
    budget: &Budget,
) -> Result<(Vec<u32>, f64), OjasError> {
    if chunk_rows == 0 {
        return Err(out_of_range("chunk_rows must be >= 1"));
    }
    let x_view = MatrixView::open(x, "emeans")?;
    let c_view = MatrixView::open(centroids, "emeans")?;
    let n = x_view.rows();
    let d = x_view.cols();
    let k = c_view.rows();
    if k == 0 {
        return Err(out_of_range("k must be >= 1"));
    }
    if k >= (UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "k ({k}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
        )));
    }
    if d == 0 {
        return Err(out_of_range("d must be >= 1"));
    }
    if c_view.cols() != d {
        return Err(shape_err(format!(
            "centroids cols {} != x cols {d}",
            c_view.cols()
        )));
    }

    let kd = checked_mul_len(k, d, "centroids len")?;
    let mut c_f32 = Scratch::<f32>::try_alloc(kd, budget)?;
    c_view.copy_window_f32_checked(0, k, c_f32.as_mut_slice())?;
    drop(c_view);

    match x_view.data() {
        MatrixData::F32(rows) => assign_cpu_f32(
            pool,
            rows,
            n,
            d,
            c_f32.as_slice(),
            k,
            chunk_rows,
            math,
            spherical,
            budget,
        ),
        MatrixData::Bf16(rows_bf16) => assign_cpu_bf16(
            pool,
            rows_bf16,
            n,
            d,
            c_f32.as_slice(),
            k,
            chunk_rows,
            math,
            spherical,
            budget,
        ),
        MatrixData::Bf16Bytes(rows_bytes) => {
            let mut c_eff_buf = Scratch::<f32>::try_alloc(kd, budget)?;
            let mut norm2_c_buf = Scratch::<f32>::try_alloc(k, budget)?;
            prepare_centroids_into(
                c_f32.as_slice(),
                k,
                d,
                spherical,
                c_eff_buf.as_mut_slice(),
                norm2_c_buf.as_mut_slice(),
            )?;
            drop(c_f32);
            let c_eff = Arc::new(c_eff_buf);
            let norm2_c = Arc::new(norm2_c_buf);

            let chunk_cap = n.min(chunk_rows);
            let mut decoded_win = Scratch::<f32>::try_alloc(
                checked_mul_len(chunk_cap, d, "mapped bf16 staging")?,
                budget,
            )?;
            let mut all_labels = vec![0_u32; n];
            let mut total_obj = 0.0_f64;

            let mut r0 = 0_usize;
            while r0 < n {
                let m_cur = (n - r0).min(chunk_rows);
                let win_slice = &mut decoded_win.as_mut_slice()[..m_cur * d];
                decode_bf16_bytes_checked(&rows_bytes[r0 * d..(r0 + m_cur) * d], win_slice)?;
                let (lbl_chunk, dist_chunk, _) = assign_chunk_validated(
                    pool, win_slice, m_cur, d, &c_eff, &norm2_c, k, math, spherical, false, budget,
                )?;
                all_labels[r0..r0 + m_cur].copy_from_slice(lbl_chunk.as_slice());
                for &dist in dist_chunk.as_slice() {
                    total_obj += f64::from(dist);
                }
                r0 += m_cur;
            }
            Ok((all_labels, total_obj))
        }
    }
}

/// Run full `canonical_v1` `e-means` training on CPU using `pool`.
pub fn fit_cpu(
    pool: &CpuPool,
    x: &Tensor,
    init_centroids: Option<&Tensor>,
    cfg: &EmeansConfig,
    math: CpuMath,
    budget: &Budget,
) -> Result<EmeansFit, OjasError> {
    if cfg.k == 0 {
        return Err(out_of_range("k must be >= 1"));
    }
    if cfg.k >= (UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "k ({}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})",
            cfg.k
        )));
    }
    let x_view = MatrixView::open(x, "emeans")?;
    let global_n = x_view.rows();
    let d = x_view.cols();
    let recipe = resolve_recipe(global_n, d, cfg)?;

    let subset = if recipe.effective_n < global_n {
        Some(sample_subset_sorted(
            global_n,
            recipe.effective_n,
            recipe.seed,
            budget,
        )?)
    } else {
        None
    };
    let subset_slice = subset.as_ref().map(Scratch::as_slice);

    // Validate all rows in the active training set upfront so fractional passes (`nepochs < 1.0`)
    // never miss a non-finite or norm-overflowing row.
    x_view.validate_rows_finite(subset_slice)?;

    // Pass `spherical = false` so `CpuSession::new` is the single canonical owner of initial
    // spherical unit-normalization on the `e-means` path (avoiding 1-ULP double normalization).
    let init_c = resolve_init_centroids(
        &x_view,
        subset_slice,
        init_centroids,
        recipe.k,
        false,
        recipe.seed,
        budget,
    )?;

    let mut session = CpuSession::new(
        recipe.k,
        recipe.d,
        recipe.spherical,
        recipe.unused_threshold,
        recipe.split_epsilon,
        recipe.init_usage,
        recipe.seed,
        init_c.as_slice(),
        budget,
    )?;
    drop(init_c);

    let max_chunk = recipe.assignment_chunk_size.min(recipe.batch_size).max(1);
    let mut gathered_chunk = Scratch::<f32>::try_alloc(
        checked_mul_len(max_chunk, d, "chunk gather buffer")?,
        budget,
    )?;
    let mut chunk_labels_buf = if cfg.capture_assignments {
        Scratch::<u32>::try_alloc(max_chunk, budget)?
    } else {
        Scratch::<u32>::try_alloc(0, budget)?
    };

    let mut labels = if cfg.capture_assignments {
        vec![UNASSIGNED_LABEL; global_n]
    } else {
        Vec::new()
    };

    let num_epochs = recipe.epoch_rows.len();
    let mut epoch_stats = Vec::with_capacity(num_epochs);
    let mut final_counts = vec![0_u64; recipe.k];
    let mut global_step = 0_usize;

    for (epoch, &rows_this_epoch) in recipe.epoch_rows.iter().enumerate() {
        let final_epoch = epoch + 1 == num_epochs;
        session.reset_pass_accumulators(final_epoch);

        let order = epoch_permutation(recipe.effective_n, recipe.seed, epoch, budget)?;
        let order_slice = order.as_slice();

        let mut revived_this_epoch = 0_usize;
        let mut batch_updates = 0_usize;
        let mut pos = 0_usize;

        while pos < rows_this_epoch {
            let b_len = (rows_this_epoch - pos).min(recipe.batch_size);
            let mut sub_off = 0_usize;

            while sub_off < b_len {
                let c_len = (b_len - sub_off).min(recipe.assignment_chunk_size);
                let chunk_indices = &order_slice[pos + sub_off..pos + sub_off + c_len];
                let chunk_dst = &mut gathered_chunk.as_mut_slice()[..c_len * d];

                for (r_local, &eff_idx) in chunk_indices.iter().enumerate() {
                    let global_row =
                        subset_slice.map_or(eff_idx as usize, |s| s[eff_idx as usize] as usize);
                    x_view.copy_row_f32_checked(
                        global_row,
                        &mut chunk_dst[r_local * d..(r_local + 1) * d],
                    )?;
                }

                if final_epoch && cfg.capture_assignments {
                    let lbl_slice = &mut chunk_labels_buf.as_mut_slice()[..c_len];
                    session.accumulate_f32_chunk(
                        pool,
                        chunk_dst,
                        c_len,
                        math,
                        Some(lbl_slice),
                        budget,
                    )?;
                    for (r_local, &eff_idx) in chunk_indices.iter().enumerate() {
                        let global_row =
                            subset_slice.map_or(eff_idx as usize, |s| s[eff_idx as usize] as usize);
                        labels[global_row] = lbl_slice[r_local];
                    }
                } else {
                    session.accumulate_f32_chunk(pool, chunk_dst, c_len, math, None, budget)?;
                }

                sub_off += c_len;
            }

            let alpha = scheduled_alpha(recipe.alpha0, global_step, recipe.total_steps);
            let check_revive = should_check_revive(global_step, recipe.unused_threshold);
            revived_this_epoch +=
                session.apply_batch_update(alpha, global_step as u64, check_revive)?;
            global_step += 1;
            batch_updates += 1;
            pos += b_len;
        }

        drop(order);
        let (ep_obj, ep_counts) = session.finish_epoch();
        if final_epoch {
            final_counts = ep_counts;
        }
        epoch_stats.push(EpochStat {
            epoch,
            rows_seen: rows_this_epoch,
            batch_updates,
            mean_objective: ep_obj / (rows_this_epoch as f64),
            revived_clusters: revived_this_epoch,
        });
    }

    drop(chunk_labels_buf);
    drop(gathered_chunk);
    drop(subset);

    let closing_vec = session.finalize_closing_centroids()?;
    let usage = session.read_usage().to_vec();
    drop(session);

    let centroids = Tensor::from_f32(&closing_vec, &[recipe.k, recipe.d], budget)?;

    Ok(EmeansFit {
        centroids,
        labels,
        cluster_counts: final_counts,
        usage,
        epoch_stats,
        recipe,
    })
}
