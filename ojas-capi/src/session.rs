//! Session table. The lock is process-global, matching the engine hooks.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

pub const SESSION_CAP: usize = 64;

/// Where a session's step and generate run.
#[derive(Clone)]
pub enum Compute {
    /// [`ojas_cpu::CpuBackend`]; `threads > 1` uses `CpuBackend::with_threads`.
    Cpu { threads: usize },
    /// The backend opened at load. Clones share its one device thread, which
    /// exits when the last clone drops.
    #[cfg(target_os = "macos")]
    Metal(ojas_metal::MetalBackend),
    /// The backend opened at load, shared by every clone of the session. Its
    /// non-finite fault word is per backend, so one session never reports
    /// another session's fault.
    Wgpu(Arc<ojas_wgpu::WgpuBackend>),
}

impl std::fmt::Debug for Compute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu { threads } => f.debug_struct("Cpu").field("threads", threads).finish(),
            #[cfg(target_os = "macos")]
            Self::Metal(metal) => f.debug_tuple("Metal").field(metal).finish(),
            Self::Wgpu(wgpu) => f
                .debug_tuple("Wgpu")
                .field(&wgpu.context().adapter_name())
                .finish(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Session {
    pub id: u64,
    pub path: PathBuf,
    pub tensors: u32,
    pub compute: Compute,
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

pub fn load_model(path: PathBuf, tensors: u32) -> Result<Session, String> {
    load_model_on(path, tensors, Compute::Cpu { threads: 1 })
}

pub fn load_model_on(path: PathBuf, tensors: u32, compute: Compute) -> Result<Session, String> {
    if matches!(compute, Compute::Cpu { threads: 0 }) {
        return Err("thread count is 0".to_string());
    }
    let mut table = lock();
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
    let id = table.next_id;
    table.next_id = table.next_id.checked_add(1).unwrap_or(0);
    let session = Session {
        id,
        path,
        tensors,
        compute,
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
