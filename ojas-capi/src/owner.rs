//! A Metal session lives on one thread because tessl's `GpuRuntime` is not `Send`.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{LazyLock, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

#[allow(dead_code)] // the name query is the owner-thread probe; training still runs on the CPU
enum Msg {
    Name(Sender<Result<String, String>>),
    Stop,
}

pub struct MetalOwner {
    tx: Sender<Msg>,
    join: Option<JoinHandle<()>>,
}

#[allow(dead_code)]
impl MetalOwner {
    pub fn spawn() -> Result<Self, String> {
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let join = std::thread::Builder::new()
            .name("ojas-metal-owner".into())
            .spawn(move || owner_main(rx, ready_tx))
            .map_err(|err| format!("metal owner thread: {err}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx,
                join: Some(join),
            }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err("metal owner thread exited before it was ready".to_string()),
        }
    }

    pub fn device_name(&self) -> Result<String, String> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Msg::Name(tx))
            .map_err(|_| "metal owner is gone".to_string())?;
        loop {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(name) => return name,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("metal owner dropped the reply".to_string())
                }
            }
        }
    }

    /// Wait for a reply while `check` can observe cancellation.
    pub fn device_name_checked(
        &self,
        mut check: impl FnMut() -> Result<(), String>,
    ) -> Result<String, String> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(Msg::Name(tx))
            .map_err(|_| "metal owner is gone".to_string())?;
        loop {
            check()?;
            match rx.recv_timeout(Duration::from_millis(20)) {
                Ok(name) => return name,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("metal owner dropped the reply".to_string())
                }
            }
        }
    }
}

static METAL_OWNERS: LazyLock<Mutex<HashMap<u64, MetalOwner>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn retain(id: u64, owner: MetalOwner) -> Result<(), String> {
    let mut map = METAL_OWNERS
        .lock()
        .map_err(|_| "metal owner table is poisoned".to_string())?;
    map.insert(id, owner);
    Ok(())
}

pub fn release(id: u64) {
    if let Ok(mut map) = METAL_OWNERS.lock() {
        map.remove(&id);
    }
}

pub fn release_all() {
    if let Ok(mut map) = METAL_OWNERS.lock() {
        map.clear();
    }
}

impl Drop for MetalOwner {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Stop);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

#[cfg(target_os = "macos")]
fn owner_main(rx: Receiver<Msg>, ready: Sender<Result<(), String>>) {
    let opened = catch_unwind(AssertUnwindSafe(ojas_metal::gpu::Session::open));
    let session = match opened {
        Ok(Ok(session)) => session,
        Ok(Err(err)) => {
            let _ = ready.send(Err(format!("metal: {err}")));
            return;
        }
        Err(_) => {
            let _ = ready.send(Err("metal owner panicked while opening".to_string()));
            return;
        }
    };
    let _ = ready.send(Ok(()));
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Stop => break,
            Msg::Name(reply) => {
                let result = catch_unwind(AssertUnwindSafe(|| session.device_name()));
                let _ = reply.send(match result {
                    Ok(name) => Ok(name),
                    Err(_) => Err("metal owner panicked".to_string()),
                });
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn owner_main(rx: Receiver<Msg>, ready: Sender<Result<(), String>>) {
    let _ = rx;
    let _ = ready.send(Err("Metal sessions require macOS".to_string()));
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn owner_thread_reports_the_device_and_a_cancel_is_checked() {
        let owner = MetalOwner::spawn().expect("Metal device");
        let name = owner.device_name().unwrap();
        assert!(!name.is_empty(), "{name}");
        let mut checks = 0u32;
        let again = owner
            .device_name_checked(|| {
                checks += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(name, again);
        assert!(checks >= 1);
    }
}
