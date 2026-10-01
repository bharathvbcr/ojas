//! Session table. The lock is process-global, matching the engine hooks.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, MutexGuard};

pub const SESSION_CAP: usize = 64;

#[derive(Clone, Debug)]
pub struct Session {
    pub id: u64,
    pub path: PathBuf,
    pub tensors: u32,
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
    Ok(())
}

pub fn root() -> Result<PathBuf, String> {
    lock()
        .root
        .clone()
        .ok_or_else(|| "model root is not configured".to_string())
}

pub fn load_model(path: PathBuf, tensors: u32) -> Result<Session, String> {
    let mut table = lock();
    if table.sessions.len() >= SESSION_CAP {
        return Err(format!(
            "capacity exceeded: session table holds {SESSION_CAP} models"
        ));
    }
    if table.next_id == 0 {
        return Err("capacity exceeded: model ids are exhausted".to_string());
    }
    let id = table.next_id;
    table.next_id = table.next_id.checked_add(1).unwrap_or(0);
    let session = Session { id, path, tensors };
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
