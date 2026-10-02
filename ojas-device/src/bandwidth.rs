//! Memory copy bandwidth, measured only when a caller asks.
//!
//! The probe copies one buffer into another on 1 thread and on N threads,
//! takes the fastest of several repetitions, and reports bytes moved (read
//! plus write) per second. Allocation is fallible and bounded, wall time is
//! bounded, and nothing is measured at startup. A number taken while other
//! work runs is a lower bound; [`Bandwidth::load_avg`] records the load the
//! measurement saw where the OS reports one.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::host::MemoryReport;
use crate::system::SystemProfile;
use ojas_core::CPU_THREAD_CEILING;

/// Smallest buffer measured. Below this the copy fits in a cache and says
/// nothing about memory.
pub const BANDWIDTH_MIN_BYTES: usize = 1 << 20;
/// Largest buffer measured (two buffers of this size are allocated).
pub const BANDWIDTH_MAX_BYTES: usize = 1 << 30;
/// Default buffer: past any L2 or system cache shipped so far.
pub const BANDWIDTH_DEFAULT_BYTES: usize = 64 << 20;
/// Most repetitions per thread count.
pub const BANDWIDTH_MAX_REPS: u32 = 100;
/// Longest wall time a caller may allow per thread count.
pub const BANDWIDTH_MAX_TIME: Duration = Duration::from_secs(10);

/// What to measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BandwidthConfig {
    /// Bytes per buffer. Two are allocated.
    pub bytes: usize,
    /// Threads for the parallel measurement. 1 measures only one thread.
    pub threads: usize,
    /// Repetitions per thread count; the fastest is kept.
    pub reps: u32,
    /// Repetitions stop once this much wall time has passed (at least one
    /// always runs).
    pub max_time: Duration,
}

impl BandwidthConfig {
    /// Buffers of [`BANDWIDTH_DEFAULT_BYTES`], each cut to 1/16 of known
    /// available memory (and of a known cgroup room) and rounded down to a
    /// page, so the two together take at most 1/8; the usable CPU count;
    /// 8 repetitions; 250 ms.
    ///
    /// Refused when that leaves less than [`BANDWIDTH_MIN_BYTES`]: on a
    /// machine that tight the measurement would itself be the pressure.
    pub fn for_profile(profile: &SystemProfile) -> Result<Self, BandwidthError> {
        let mut bytes = BANDWIDTH_DEFAULT_BYTES as u64;
        if let MemoryReport::Known(avail) = profile.memory.available_bytes {
            bytes = bytes.min(avail / 16);
        }
        if let MemoryReport::Known(limit) = profile.memory.cgroup_limit_bytes {
            let room = match profile.memory.cgroup_current_bytes {
                MemoryReport::Known(cur) => limit.saturating_sub(cur),
                MemoryReport::Unknown => limit,
            };
            bytes = bytes.min(room / 16);
        }
        let page = match profile.cpu.page_bytes {
            MemoryReport::Known(p) if p.is_power_of_two() && p <= 1 << 21 => p,
            _ => 4096,
        };
        bytes -= bytes % page;
        let bytes = usize::try_from(bytes).map_err(|_| BandwidthError::Config("bytes".into()))?;
        if bytes < BANDWIDTH_MIN_BYTES {
            return Err(BandwidthError::Config(format!(
                "only {bytes} bytes fit in 1/16 of the memory this process can use; \
                 at least {BANDWIDTH_MIN_BYTES} are needed"
            )));
        }
        let threads = match profile.cpu.usable {
            MemoryReport::Known(n) => usize::try_from(n)
                .unwrap_or(usize::MAX)
                .clamp(1, CPU_THREAD_CEILING as usize),
            MemoryReport::Unknown => 1,
        };
        Ok(Self {
            bytes,
            threads,
            reps: 8,
            max_time: Duration::from_millis(250),
        })
    }

    fn validate(&self) -> Result<(), BandwidthError> {
        if !(BANDWIDTH_MIN_BYTES..=BANDWIDTH_MAX_BYTES).contains(&self.bytes) {
            return Err(BandwidthError::Config(format!(
                "bytes {} outside {BANDWIDTH_MIN_BYTES}..={BANDWIDTH_MAX_BYTES}",
                self.bytes
            )));
        }
        if self.threads == 0 || self.threads > CPU_THREAD_CEILING as usize {
            return Err(BandwidthError::Config(format!(
                "threads {} outside 1..={CPU_THREAD_CEILING}",
                self.threads
            )));
        }
        if self.reps == 0 || self.reps > BANDWIDTH_MAX_REPS {
            return Err(BandwidthError::Config(format!(
                "reps {} outside 1..={BANDWIDTH_MAX_REPS}",
                self.reps
            )));
        }
        if self.max_time.is_zero() || self.max_time > BANDWIDTH_MAX_TIME {
            return Err(BandwidthError::Config(format!(
                "max_time {:?} outside (0, {BANDWIDTH_MAX_TIME:?}]",
                self.max_time
            )));
        }
        Ok(())
    }
}

/// A measured copy bandwidth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bandwidth {
    /// Read plus write bytes per second on one thread.
    pub single_bytes_per_sec: u64,
    /// Read plus write bytes per second on `threads` threads. Equal to the
    /// single figure's measurement when `threads` is 1.
    pub multi_bytes_per_sec: u64,
    pub threads: usize,
    pub buffer_bytes: usize,
    /// Repetitions that ran for the single and the multi measurement.
    pub reps_run: (u32, u32),
    /// 1-minute load average when the measurement started, if the OS
    /// reports one. A high value means the figures are lower bounds.
    pub load_avg: Option<f64>,
}

impl Bandwidth {
    /// Threads that reach the multi-thread bandwidth if each adds the
    /// single-thread figure: `ceil(multi / single)`, at least 1 and at most
    /// the threads measured. An estimate from two points, not a sweep.
    pub fn saturating_threads(&self) -> usize {
        if self.single_bytes_per_sec == 0 {
            return self.threads.max(1);
        }
        let n = self.multi_bytes_per_sec.div_ceil(self.single_bytes_per_sec);
        usize::try_from(n)
            .unwrap_or(usize::MAX)
            .clamp(1, self.threads.max(1))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BandwidthError {
    /// The configuration was outside the bounds above.
    Config(String),
    /// A buffer could not be reserved. Nothing was measured.
    Alloc { bytes: usize },
    /// A measuring thread could not be spawned.
    Spawn(String),
    /// A measuring thread panicked.
    Panicked,
    /// A copy finished in less time than the clock resolves.
    Clock,
}

impl fmt::Display for BandwidthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BandwidthError::Config(d) => write!(f, "bandwidth config: {d}"),
            BandwidthError::Alloc { bytes } => {
                write!(f, "bandwidth: could not reserve {bytes} bytes")
            }
            BandwidthError::Spawn(d) => write!(f, "bandwidth: spawn failed: {d}"),
            BandwidthError::Panicked => write!(f, "bandwidth: a measuring thread panicked"),
            BandwidthError::Clock => write!(f, "bandwidth: a copy finished below clock resolution"),
        }
    }
}

impl std::error::Error for BandwidthError {}

/// Measure copy bandwidth with `config`. Blocks for at most about
/// `2 * max_time` plus one repetition, and allocates `2 * bytes`.
pub fn measure_bandwidth(config: &BandwidthConfig) -> Result<Bandwidth, BandwidthError> {
    config.validate()?;
    let load_avg = load_average();
    let mut src = try_buffer(config.bytes)?;
    let mut dst = try_buffer(config.bytes)?;
    // Touch every page of both, so the timed copies do not fault them in.
    for (i, b) in src.iter_mut().enumerate() {
        *b = i as u8;
    }
    dst.fill(0);
    let (single, single_reps) = timed_copies(&src, &mut dst, 1, config)?;
    let (multi, multi_reps) = if config.threads == 1 {
        (single, single_reps)
    } else {
        timed_copies(&src, &mut dst, config.threads, config)?
    };
    if dst[config.bytes - 1] != src[config.bytes - 1] || dst[0] != src[0] {
        return Err(BandwidthError::Config("copy did not land".into()));
    }
    let rate = |d: Duration| -> Result<u64, BandwidthError> {
        let nanos = d.as_nanos();
        if nanos == 0 {
            return Err(BandwidthError::Clock);
        }
        let moved = 2u128 * config.bytes as u128;
        Ok(u64::try_from(moved * 1_000_000_000 / nanos).unwrap_or(u64::MAX))
    };
    Ok(Bandwidth {
        single_bytes_per_sec: rate(single)?,
        multi_bytes_per_sec: rate(multi)?,
        threads: config.threads,
        buffer_bytes: config.bytes,
        reps_run: (single_reps, multi_reps),
        load_avg,
    })
}

/// [`measure_bandwidth`] with [`BandwidthConfig::for_profile`], kept for the
/// rest of the process once it succeeds. Bandwidth is a property of the
/// machine, so a later call returns the first success whatever profile it
/// passes. An error is not kept: a profile too tight to measure now (or a
/// failed spawn) does not stop a later call from measuring. Concurrent
/// callers wait for one measurement rather than running several at once.
pub fn cached_bandwidth(profile: &SystemProfile) -> Result<Bandwidth, BandwidthError> {
    static CACHE: Mutex<Option<Bandwidth>> = Mutex::new(None);
    let mut cached = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bw) = *cached {
        return Ok(bw);
    }
    let bw = BandwidthConfig::for_profile(profile).and_then(|c| measure_bandwidth(&c))?;
    *cached = Some(bw);
    Ok(bw)
}

fn try_buffer(bytes: usize) -> Result<Vec<u8>, BandwidthError> {
    let mut v: Vec<u8> = Vec::new();
    v.try_reserve_exact(bytes)
        .map_err(|_| BandwidthError::Alloc { bytes })?;
    v.resize(bytes, 0);
    Ok(v)
}

/// Fastest of up to `reps` copies of `src` into `dst` split across
/// `threads` workers that stay alive between repetitions.
fn timed_copies(
    src: &[u8],
    dst: &mut [u8],
    threads: usize,
    config: &BandwidthConfig,
) -> Result<(Duration, u32), BandwidthError> {
    let chunk = src.len().div_ceil(threads).max(1);
    let src_chunks: Vec<&[u8]> = src.chunks(chunk).collect();
    let dst_chunks: Vec<&mut [u8]> = dst.chunks_mut(chunk).collect();
    let workers = src_chunks.len();
    // Workers and the timer meet twice per repetition: start and end.
    let gate = Barrier::new(workers + 1);
    let stop = AtomicBool::new(false);
    // No worker touches `gate` until every worker exists: a spawn failure
    // would otherwise leave the spawned ones parked on a barrier that can
    // never fill. `latch` is `None` until the spawns end, then go / abort.
    let latch = (Mutex::new(None::<bool>), Condvar::new());
    let started = Instant::now();
    std::thread::scope(|scope| -> Result<(Duration, u32), BandwidthError> {
        let mut handles = Vec::with_capacity(workers);
        let mut spawn_error = None;
        for (s, d) in src_chunks.into_iter().zip(dst_chunks) {
            let (gate, stop, latch) = (&gate, &stop, &latch);
            let spawned = std::thread::Builder::new()
                .name("ojas-bandwidth".into())
                .spawn_scoped(scope, move || {
                    if !wait_latch(latch) {
                        return;
                    }
                    loop {
                        gate.wait();
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        d.copy_from_slice(s);
                        std::hint::black_box(&*d);
                        gate.wait();
                    }
                });
            match spawned {
                Ok(h) => handles.push(h),
                Err(e) => {
                    spawn_error = Some(e.to_string());
                    break;
                }
            }
        }
        set_latch(&latch, spawn_error.is_none());
        if let Some(e) = spawn_error {
            // The spawned workers saw abort and return; the scope joins them.
            return Err(BandwidthError::Spawn(e));
        }
        let mut best = Duration::MAX;
        let mut reps = 0;
        while reps < config.reps {
            gate.wait();
            let t0 = Instant::now();
            gate.wait();
            best = best.min(t0.elapsed());
            reps += 1;
            if started.elapsed() >= config.max_time {
                break;
            }
        }
        stop.store(true, Ordering::Release);
        gate.wait();
        let mut panicked = false;
        for h in handles {
            panicked |= h.join().is_err();
        }
        if panicked {
            return Err(BandwidthError::Panicked);
        }
        Ok((best, reps))
    })
}

type Latch = (Mutex<Option<bool>>, Condvar);

/// Block until the latch is set; `true` is go.
fn wait_latch(latch: &Latch) -> bool {
    let (lock, cv) = latch;
    let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(go) = *state {
            return go;
        }
        state = cv.wait(state).unwrap_or_else(|e| e.into_inner());
    }
}

fn set_latch(latch: &Latch, go: bool) {
    let (lock, cv) = latch;
    *lock.lock().unwrap_or_else(|e| e.into_inner()) = Some(go);
    cv.notify_all();
}

fn load_average() -> Option<f64> {
    #[cfg(unix)]
    {
        load_average_unix()
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn load_average_unix() -> Option<f64> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let mut avg = [0f64; 3];
        // SAFETY: `avg` is writable for 3 doubles and `getloadavg` writes at
        // most the count passed.
        let n = unsafe { libc::getloadavg(avg.as_mut_ptr(), 1) };
        (n >= 1 && avg[0].is_finite() && avg[0] >= 0.0).then_some(avg[0])
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::SystemProfile;
    use crate::topology::CpuTopology;
    use crate::HostMemory;

    fn small(threads: usize) -> BandwidthConfig {
        BandwidthConfig {
            bytes: BANDWIDTH_MIN_BYTES,
            threads,
            reps: 3,
            max_time: Duration::from_millis(50),
        }
    }

    #[test]
    fn out_of_bounds_configs_are_refused_before_allocating() {
        let ok = small(1);
        let cases = [
            BandwidthConfig { bytes: 0, ..ok },
            BandwidthConfig {
                bytes: BANDWIDTH_MIN_BYTES - 1,
                ..ok
            },
            BandwidthConfig {
                bytes: BANDWIDTH_MAX_BYTES + 1,
                ..ok
            },
            BandwidthConfig {
                bytes: usize::MAX,
                ..ok
            },
            BandwidthConfig { threads: 0, ..ok },
            BandwidthConfig {
                threads: CPU_THREAD_CEILING as usize + 1,
                ..ok
            },
            BandwidthConfig {
                threads: usize::MAX,
                ..ok
            },
            BandwidthConfig { reps: 0, ..ok },
            BandwidthConfig {
                reps: BANDWIDTH_MAX_REPS + 1,
                ..ok
            },
            BandwidthConfig {
                max_time: Duration::ZERO,
                ..ok
            },
            BandwidthConfig {
                max_time: BANDWIDTH_MAX_TIME + Duration::from_nanos(1),
                ..ok
            },
        ];
        for c in cases {
            assert!(
                matches!(measure_bandwidth(&c), Err(BandwidthError::Config(_))),
                "{c:?}"
            );
        }
    }

    #[test]
    fn a_small_measurement_is_positive_and_bounded() {
        for threads in [1, 2, 3, 7] {
            let t0 = Instant::now();
            let bw = measure_bandwidth(&small(threads)).unwrap();
            assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
            assert!(bw.single_bytes_per_sec > 0);
            assert!(bw.multi_bytes_per_sec > 0);
            assert_eq!(bw.threads, threads);
            assert!((1..=3).contains(&bw.reps_run.0));
            assert!((1..=3).contains(&bw.reps_run.1));
            assert!((1..=threads).contains(&bw.saturating_threads()));
            if threads == 1 {
                assert_eq!(bw.single_bytes_per_sec, bw.multi_bytes_per_sec);
            }
        }
    }

    #[test]
    fn more_threads_than_bytes_per_page_still_copies_everything() {
        // 1 MiB over 1024 workers: 1 KiB each, every byte checked.
        let c = BandwidthConfig {
            threads: CPU_THREAD_CEILING as usize,
            reps: 1,
            ..small(1)
        };
        let bw = measure_bandwidth(&c).unwrap();
        assert_eq!(bw.threads, CPU_THREAD_CEILING as usize);
    }

    #[test]
    fn saturating_threads_is_clamped() {
        let mut bw = Bandwidth {
            single_bytes_per_sec: 10,
            multi_bytes_per_sec: 35,
            threads: 8,
            buffer_bytes: 1,
            reps_run: (1, 1),
            load_avg: None,
        };
        assert_eq!(bw.saturating_threads(), 4);
        bw.multi_bytes_per_sec = 1_000;
        assert_eq!(
            bw.saturating_threads(),
            8,
            "never past the threads measured"
        );
        bw.multi_bytes_per_sec = 1;
        assert_eq!(bw.saturating_threads(), 1);
        bw.single_bytes_per_sec = 0;
        assert_eq!(bw.saturating_threads(), 8);
        bw.threads = 0;
        assert_eq!(bw.saturating_threads(), 1);
        bw.single_bytes_per_sec = 1;
        bw.multi_bytes_per_sec = u64::MAX;
        bw.threads = usize::MAX;
        assert_eq!(
            bw.saturating_threads(),
            usize::try_from(u64::MAX).unwrap_or(usize::MAX)
        );
    }

    #[test]
    fn for_profile_shrinks_to_memory_and_refuses_when_too_tight() {
        let mut memory = HostMemory::all_unknown();
        let mut p = SystemProfile::from_memory(memory);
        let c = BandwidthConfig::for_profile(&p).unwrap();
        assert_eq!(c.bytes, BANDWIDTH_DEFAULT_BYTES);
        assert_eq!(c.threads, 1, "unknown usable count measures one thread");
        memory.available_bytes = MemoryReport::Known(32 << 20);
        p = SystemProfile::from_memory(memory);
        assert_eq!(BandwidthConfig::for_profile(&p).unwrap().bytes, 2 << 20);
        memory.cgroup_limit_bytes = MemoryReport::Known(100 << 20);
        memory.cgroup_current_bytes = MemoryReport::Known(90 << 20);
        p = SystemProfile::from_memory(memory);
        assert!(matches!(
            BandwidthConfig::for_profile(&p),
            Err(BandwidthError::Config(_))
        ));
        memory.cgroup_limit_bytes = MemoryReport::Unknown;
        memory.available_bytes = MemoryReport::Known(0);
        p = SystemProfile::from_memory(memory);
        assert!(BandwidthConfig::for_profile(&p).is_err());
        // Page rounding: an odd figure rounds down to a page.
        memory.available_bytes = MemoryReport::Known((40 << 20) + 12_345);
        p = SystemProfile::from_memory(memory);
        p.cpu = CpuTopology {
            page_bytes: MemoryReport::Known(16384),
            usable: MemoryReport::Known(5),
            ..CpuTopology::all_unknown()
        };
        let c = BandwidthConfig::for_profile(&p).unwrap();
        assert_eq!(c.bytes % 16384, 0);
        assert!(c.bytes <= ((40 << 20) + 12_345) / 16);
        assert_eq!(c.threads, 5);
        // A hostile page size is not trusted.
        p.cpu.page_bytes = MemoryReport::Known(3);
        assert_eq!(BandwidthConfig::for_profile(&p).unwrap().bytes % 4096, 0);
        p.cpu.usable = MemoryReport::Known(u64::MAX);
        assert_eq!(
            BandwidthConfig::for_profile(&p).unwrap().threads,
            CPU_THREAD_CEILING as usize
        );
    }

    #[test]
    fn concurrent_measurements_do_not_deadlock_or_corrupt() {
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..4)
                .map(|i| s.spawn(move || measure_bandwidth(&small(1 + i))))
                .collect();
            for h in hs {
                assert!(h.join().unwrap().is_ok());
            }
        });
    }

    /// The only test in this binary that calls `cached_bandwidth`, so the
    /// process-wide cache starts empty here.
    #[test]
    fn a_refused_measurement_is_not_cached_and_a_success_is() {
        let mut tight = HostMemory::all_unknown();
        tight.available_bytes = MemoryReport::Known(0);
        let refused = cached_bandwidth(&SystemProfile::from_memory(tight));
        assert!(
            matches!(refused, Err(BandwidthError::Config(_))),
            "{refused:?}"
        );
        let mut roomy = HostMemory::all_unknown();
        roomy.available_bytes = MemoryReport::Known(64 << 20);
        let first = cached_bandwidth(&SystemProfile::from_memory(roomy))
            .expect("an earlier refusal must not stick");
        assert_eq!(first.buffer_bytes, 4 << 20, "1/16 of 64 MiB per buffer");
        // Kept: even a profile that could not measure now gets the success.
        assert_eq!(
            cached_bandwidth(&SystemProfile::from_memory(tight)),
            Ok(first)
        );
    }

    #[test]
    fn an_aborted_latch_releases_every_waiter_without_the_gate() {
        // Workers parked on the latch return when it is set to abort, so a
        // spawn failure part-way never leaves a thread on the barrier.
        let latch: Latch = (Mutex::new(None), Condvar::new());
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..8).map(|_| s.spawn(|| wait_latch(&latch))).collect();
            std::thread::sleep(Duration::from_millis(5));
            set_latch(&latch, false);
            for h in hs {
                assert!(!h.join().unwrap());
            }
        });
        // A latch set before anyone waits is seen too.
        let latch: Latch = (Mutex::new(None), Condvar::new());
        set_latch(&latch, true);
        assert!(wait_latch(&latch));
    }
}
