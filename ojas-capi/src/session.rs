//! Session table. The lock is process-global, matching the engine hooks.
//!
//! The table lock covers the id map only. Each session's model sits behind
//! its own `Arc<Mutex<ModelState>>`, taken per call with `try_lock`
//! ([`Session::lock_state`]); a call never waits on another call's model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, TryLockError};

use crate::model::ModelState;

pub const SESSION_CAP: usize = 64;

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
