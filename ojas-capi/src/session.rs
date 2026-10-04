//! Session table. The lock is process-global, matching the engine hooks.
//!
//! The table lock covers the id map only. Each session's model sits behind
//! its own `Arc<Mutex<ModelState>>`, taken per call with `try_lock`
//! ([`Session::lock_state`]); a call never waits on another call's model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, TryLockError};

use ojas_core::Budget;

use crate::model::ModelState;

pub const SESSION_CAP: usize = 64;

/// The process-wide memory ceiling when nothing raised it: 1 GiB, or the
/// machine's hard limit ([`hard_memory_limit`]) when that is smaller.
///
/// Every session's budget is a child of one root [`Budget`] with this cap,
/// so all sessions together account at most this many bytes, whatever each
/// one's own budget says. [`set_memory_ceiling`] is the only way to change
/// it.
pub const DEFAULT_MEMORY_CEILING_BYTES: u64 = 1 << 30;

/// The root every session budget charges, and the token each session's
/// [`Lease`] clones. The lock covers both, so a lease is only ever taken
/// against the root it was counted with.
struct Ceiling {
    root: Budget,
    leases: Arc<()>,
}

static CEILING: LazyLock<Mutex<Ceiling>> = LazyLock::new(|| {
    // On a machine (or cgroup) smaller than the default, the default would
    // let sessions charge memory that does not exist; start at the limit.
    let cap = hard_memory_limit().map_or(DEFAULT_MEMORY_CEILING_BYTES, |limit| {
        DEFAULT_MEMORY_CEILING_BYTES.min(limit)
    });
    Mutex::new(Ceiling {
        root: Budget::new(cap),
        leases: Arc::new(()),
    })
});

/// The most memory this process can ever hold: physical RAM, or a tighter
/// cgroup limit. `None` when neither can be read (the ceiling is then not
/// checked against the machine).
pub fn hard_memory_limit() -> Option<u64> {
    let host = ojas_device::probe_host();
    [host.total_bytes, host.cgroup_limit_bytes]
        .into_iter()
        .filter_map(|report| match report {
            ojas_device::MemoryReport::Known(n) => Some(n),
            ojas_device::MemoryReport::Unknown => None,
        })
        .min()
}

fn ceiling() -> MutexGuard<'static, Ceiling> {
    // A panic under this lock leaves either the old ceiling or the new one,
    // never half of either: the only mutation is one assignment.
    CEILING.lock().unwrap_or_else(|e| e.into_inner())
}

/// Held by a session's [`ModelState`] from device open until the state is
/// dropped. While any lease lives the ceiling cannot change, because a
/// [`Budget`]'s cap is fixed and a new root would split the accounting
/// between sessions on the old root and sessions on the new one.
#[derive(Debug)]
pub struct Lease {
    _token: Arc<()>,
}

/// A session budget of `bytes`, drawn from the process ceiling, and the
/// lease that pins that ceiling while the session lives.
///
/// A charge on the returned budget charges the root first and is refused
/// (`CapacityExceeded`, kind `E_CAPACITY`) when all sessions together would
/// pass the ceiling. `bytes` above the ceiling can never be met and is
/// refused here with `E_CAPACITY`.
pub fn session_budget(bytes: u64) -> Result<(Budget, Lease), String> {
    let ceiling = ceiling();
    let cap = ceiling.root.cap_bytes();
    if bytes > cap {
        return Err(crate::kinded(
            crate::ErrorKind::Capacity,
            format!(
                "capacity exceeded: session budget of {bytes} bytes exceeds the process ceiling \
                 of {cap} bytes"
            ),
        ));
    }
    Ok((
        ceiling.root.child(bytes),
        Lease {
            _token: Arc::clone(&ceiling.leases),
        },
    ))
}

/// Replace the process-wide memory ceiling with `bytes`.
///
/// Refused when `bytes` is 0, above the machine's memory
/// ([`hard_memory_limit`]: physical RAM or a tighter cgroup limit; kind
/// `E_CAPACITY`), and while any model holds the ceiling: a session in the
/// table, one still being built, or one freed while a call was inside it.
/// Free every model first. Nothing is changed on a refusal. This is the
/// only setter; raising the ceiling is always this explicit call, never a
/// per-session budget.
pub fn set_memory_ceiling(bytes: u64) -> Result<(), String> {
    set_memory_ceiling_within(bytes, hard_memory_limit())
}

/// [`set_memory_ceiling`] against an explicit machine limit.
pub(crate) fn set_memory_ceiling_within(bytes: u64, limit: Option<u64>) -> Result<(), String> {
    if bytes == 0 {
        return Err("memory ceiling: 0 bytes".to_string());
    }
    if let Some(limit) = limit.filter(|&limit| bytes > limit) {
        // Every charge under such a ceiling could pass while the memory
        // behind it does not exist: the run would die of the OS's OOM kill
        // instead of a refusal.
        return Err(crate::kinded(
            crate::ErrorKind::Capacity,
            format!(
                "capacity exceeded: memory ceiling of {bytes} bytes exceeds this machine's \
                 {limit} bytes"
            ),
        ));
    }
    let mut ceiling = ceiling();
    let held = Arc::strong_count(&ceiling.leases) - 1;
    if held != 0 {
        return Err(format!(
            "memory ceiling: {held} models hold the current ceiling; free them first"
        ));
    }
    let live = ceiling
        .root
        .live_bytes()
        .map_err(|e| crate::ojas_error("memory ceiling", &e))?;
    if live != 0 {
        return Err(format!(
            "memory ceiling: {live} bytes are still charged to the current ceiling"
        ));
    }
    *ceiling = Ceiling {
        root: Budget::new(bytes),
        leases: Arc::new(()),
    };
    Ok(())
}

/// `(cap, live)` bytes of the process ceiling.
pub fn memory_ceiling() -> Result<(u64, u64), String> {
    let ceiling = ceiling();
    let live = ceiling
        .root
        .live_bytes()
        .map_err(|e| crate::ojas_error("memory ceiling", &e))?;
    Ok((ceiling.root.cap_bytes(), live))
}

/// Where a session computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// [`ojas_cpu::CpuBackend`]; `threads > 1` uses `CpuBackend::with_threads`.
    Cpu { threads: usize },
    /// The session's own `MetalBackend`, opened at load.
    Metal,
    /// The session's own `WgpuBackend`, opened at load. Its non-finite fault
    /// word is per backend, so one session never reports another's fault.
    Wgpu,
}

#[derive(Clone)]
pub struct Session {
    pub id: u64,
    /// The file or checkpoint directory the model came from; `None` for a
    /// fresh init.
    pub path: Option<PathBuf>,
    /// Parameter tensors in the model.
    pub tensors: u32,
    pub device: DeviceKind,
    state: Arc<Mutex<ModelState>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("path", &self.path)
            .field("tensors", &self.tensors)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// The model, for one call. Another call holding it is `ojas:E_BUSY:`;
    /// a model a panicking call left behind is `ojas:E_POISONED:`.
    pub fn lock_state(&self) -> Result<MutexGuard<'_, ModelState>, String> {
        match self.state.try_lock() {
            Ok(guard) => Ok(guard),
            Err(TryLockError::WouldBlock) => Err(crate::kinded(
                crate::ErrorKind::Busy,
                format!("model {} is busy: another call holds it", self.id),
            )),
            Err(TryLockError::Poisoned(_)) => Err(crate::kinded(
                crate::ErrorKind::Poisoned,
                format!("model {}: a call panicked while holding it", self.id),
            )),
        }
    }
}

struct Table {
    root: Option<PathBuf>,
    sessions: HashMap<u64, Session>,
    next_id: u64,
}

static TABLE: LazyLock<Mutex<Table>> = LazyLock::new(|| {
    Mutex::new(Table {
        root: None,
        sessions: HashMap::new(),
        next_id: 1,
    })
});

static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn lock() -> MutexGuard<'static, Table> {
    match TABLE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            // A panic while the table was held leaves whatever mutation
            // unwound halfway. Drop it rather than serving that state, once:
            // without clear_poison every later lock would drop new sessions.
            let mut guard = poisoned.into_inner();
            guard.sessions.clear();
            TABLE.clear_poison();
            guard
        }
    }
}

pub fn set_last_error(msg: impl Into<String>) {
    let mut slot = LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner());
    *slot = msg.into();
}

pub fn last_error() -> String {
    LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn clear_last_error() {
    let mut slot = LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner());
    slot.clear();
}

/// Copy the stored error into `dst` and return its full length.
///
/// The stored message is cleared only when `dst` holds all of it. A short
/// buffer leaves the text in place so the caller can take it again. An empty
/// message is not cleared into a newer one; there is nothing to remove.
pub fn take_last_error(dst: &mut [u8]) -> usize {
    let mut slot = LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner());
    let full = slot.len();
    if dst.is_empty() || full == 0 {
        return full;
    }
    let n = full.min(dst.len());
    dst[..n].copy_from_slice(&slot.as_bytes()[..n]);
    if dst.len() >= full {
        slot.clear();
    }
    full
}

pub fn set_model_root(path: &str) -> Result<(), String> {
    if path.is_empty() || path.contains('\0') {
        return Err("model root is empty or contains NUL".to_string());
    }
    let canon = Path::new(path)
        .canonicalize()
        .map_err(|err| format!("model root: {err}"))?;
    if !canon.is_dir() {
        return Err(format!(
            "model root is not a directory: {}",
            canon.display()
        ));
    }
    lock().root = Some(canon);
    clear_last_error();
    Ok(())
}

pub fn root() -> Result<PathBuf, String> {
    lock()
        .root
        .clone()
        .ok_or_else(|| "model root is not configured".to_string())
}

fn full(table: &Table) -> Result<(), String> {
    if table.sessions.len() >= SESSION_CAP {
        return Err(crate::kinded(
            crate::ErrorKind::Capacity,
            format!("capacity exceeded: session table holds {SESSION_CAP} models"),
        ));
    }
    if table.next_id == 0 {
        return Err(crate::kinded(
            crate::ErrorKind::Capacity,
            "capacity exceeded: model ids are exhausted",
        ));
    }
    Ok(())
}

/// Refuse early when the table is full, before a model is read. The insert
/// checks again; this check only saves the work.
pub fn check_room() -> Result<(), String> {
    full(&lock())
}

/// Add a built model. The table cap is checked here, under the lock.
pub fn insert(
    path: Option<PathBuf>,
    tensors: u32,
    device: DeviceKind,
    state: ModelState,
) -> Result<Session, String> {
    let mut table = lock();
    full(&table)?;
    let id = table.next_id;
    table.next_id = table.next_id.checked_add(1).unwrap_or(0);
    let session = Session {
        id,
        path,
        tensors,
        device,
        state: Arc::new(Mutex::new(state)),
    };
    table.sessions.insert(id, session.clone());
    Ok(session)
}

pub fn require(id: u64) -> Result<Session, String> {
    lock()
        .sessions
        .get(&id)
        .cloned()
        .ok_or_else(|| format!("unknown model id {id}"))
}

/// Drop `id` from the table. A call already inside it keeps its model until
/// it returns.
pub fn try_free(id: u64) -> Result<(), String> {
    let mut table = lock();
    if table.sessions.remove(&id).is_none() {
        return Err(format!("unknown model id {id}"));
    }
    Ok(())
}

pub fn session_count() -> usize {
    lock().sessions.len()
}

pub fn clear_sessions() {
    lock().sessions.clear();
}

pub fn reset_sessions() {
    clear_sessions();
}

#[cfg(test)]
pub(crate) fn poison_table() {
    let joined = std::thread::spawn(|| {
        let _held = TABLE.lock();
        panic!("ojas test: poison the session table");
    })
    .join();
    assert!(joined.is_err());
    assert!(TABLE.is_poisoned());
}

#[cfg(test)]
impl Session {
    /// Hold this session's model from another thread until `release` fires.
    pub(crate) fn hold(&self) -> (std::sync::mpsc::Sender<()>, std::thread::JoinHandle<()>) {
        let state = Arc::clone(&self.state);
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            let _guard = state.lock().unwrap_or_else(|e| e.into_inner());
            held_tx.send(()).unwrap();
            let _ = release_rx.recv();
        });
        held_rx.recv().unwrap();
        (release_tx, handle)
    }

    /// The next `op` on this session's backend fails with `make()`, once.
    pub(crate) fn inject(&self, op: &'static str, make: fn() -> ojas_core::OjasError) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.slot.inject(op, make);
    }

    /// Poison this session's model lock with a panic while it is held.
    pub(crate) fn poison_state(&self) {
        let state = Arc::clone(&self.state);
        let joined = std::thread::spawn(move || {
            let _guard = state.lock();
            panic!("ojas test: poison a model");
        })
        .join();
        assert!(joined.is_err());
    }
}
