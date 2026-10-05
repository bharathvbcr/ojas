//! Device opens on a short-lived thread, so a cancelled load does not wait
//! for the device. The Metal thread calls [`ojas_metal::MetalBackend::new`],
//! whose own device thread is the only owner of tessl's `GpuRuntime` (which is
//! not `Send`); this crate never opens a second one. The wgpu thread calls
//! [`ojas_wgpu::WgpuBackend::open`].

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use ojas_core::Budget;

/// Open a [`ojas_metal::MetalBackend`] charging `budget`.
///
/// `check` is polled while the device opens, so a cancelled job returns
/// without waiting for it. A backend that finishes opening after that is
/// dropped by the abandoned thread, because its reply receiver is gone, and
/// its device thread exits with it.
#[cfg(target_os = "macos")]
pub fn open_metal(
    budget: Budget,
    check: impl FnMut() -> Result<(), String>,
) -> Result<ojas_metal::MetalBackend, String> {
    open_on_thread(
        "metal",
        move || {
            ojas_metal::MetalBackend::new(budget).map_err(|err| crate::ojas_error("metal", &err))
        },
        check,
    )
}

/// Without macOS there is no Metal backend to open.
#[cfg(not(target_os = "macos"))]
pub fn open_metal(mut check: impl FnMut() -> Result<(), String>) -> Result<(), String> {
    check()?;
    Err("metal: Metal requires macOS".to_string())
}

/// Open a [`ojas_wgpu::WgpuBackend`] charging `budget`, polling `check` as
/// [`open_metal`] does. No adapter is the error `wgpu: ...`.
pub fn open_wgpu(
    budget: Budget,
    check: impl FnMut() -> Result<(), String>,
) -> Result<ojas_wgpu::WgpuBackend, String> {
    open_on_thread(
        "wgpu",
        move || {
            ojas_wgpu::WgpuBackend::open(budget).map_err(|err| crate::device_error("wgpu", &err))
        },
        check,
    )
}

fn open_on_thread<T: Send + 'static>(
    device: &'static str,
    open: impl FnOnce() -> Result<T, String> + Send + 'static,
    mut check: impl FnMut() -> Result<(), String>,
) -> Result<T, String> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name(format!("ojas-{device}-open"))
        .spawn(move || {
            let answer = catch_unwind(AssertUnwindSafe(open))
                .unwrap_or_else(|_| Err(format!("{device} open panicked")));
            let _ = tx.send(answer);
        })
        .map_err(|err| format!("{device} open thread: {err}"))?;
    wait_for(device, &rx, &mut check)
}

fn wait_for<T>(
    device: &str,
    rx: &Receiver<Result<T, String>>,
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<T, String> {
    loop {
        check()?;
        match rx.recv_timeout(Duration::from_millis(20)) {
            Ok(answer) => return answer,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                return Err(format!("{device} open thread exited without an answer"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cancel_returns_while_the_open_is_pending() {
        let (_tx, rx) = mpsc::channel::<Result<String, String>>();
        let mut calls = 0u32;
        let err = wait_for("metal", &rx, &mut || {
            calls += 1;
            if calls > 2 {
                Err("cancelled: Explicit".to_string())
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(err, "cancelled: Explicit");
        assert_eq!(calls, 3);
    }

    #[test]
    fn an_open_thread_that_drops_its_reply_is_an_error() {
        let (tx, rx) = mpsc::channel::<Result<String, String>>();
        drop(tx);
        let err = wait_for("metal", &rx, &mut || Ok(())).unwrap_err();
        assert!(err.contains("without an answer"), "{err}");
    }

    #[test]
    fn a_panicking_open_is_an_error() {
        let err =
            open_on_thread::<()>("wgpu", || panic!("ojas test: open"), || Ok(())).unwrap_err();
        assert_eq!(err, "wgpu open panicked");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn open_reports_the_metal_device_name() {
        match open_metal(Budget::new(1 << 20), || Ok(())) {
            Ok(backend) => assert!(!backend.device_name().is_empty()),
            Err(err) => crate::tests::skip_or_fail("open_metal", &err),
        }
    }
}
