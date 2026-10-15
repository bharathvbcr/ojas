//! What the crates that link a GPU runtime (`ojas-cuda`, `ojas-hip`) share:
//! the one bounded wait and the one free-memory check. Neither opens a
//! device; each runtime calls them with its own readiness query and error.

use std::time::{Duration, Instant};

use crate::{Device, DeviceError};

/// Poll `ready` every `interval` until it reports `true`, it fails, or
/// `timeout` has passed, whichever comes first. `ready` is always called at
/// least once, so a zero timeout still answers an already-finished wait. A
/// timeout is `timed_out(waited)`. No sleep runs past the deadline.
///
/// Every device wait polls through this instead of blocking: an unbounded
/// stream synchronize on a hung or lost device would hang the process.
pub fn poll_until<E>(
    timeout: Duration,
    interval: Duration,
    mut ready: impl FnMut() -> Result<bool, E>,
    timed_out: impl FnOnce(Duration) -> E,
) -> Result<(), E> {
    let start = Instant::now();
    loop {
        if ready()? {
            return Ok(());
        }
        let waited = start.elapsed();
        if waited >= timeout {
            return Err(timed_out(waited));
        }
        std::thread::sleep(interval.min(timeout - waited));
    }
}

/// `need` device bytes do not fit in the `free` bytes the runtime reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreeShortfall {
    pub need: u64,
    pub free: u64,
}

impl FreeShortfall {
    /// The shortfall as [`DeviceError::Capacity`] for `kind`, naming `op`.
    pub fn capacity(self, kind: Device, op: &str) -> DeviceError {
        DeviceError::Capacity {
            kind,
            detail: format!(
                "{op}: {} device bytes needed, {} bytes free",
                self.need, self.free
            ),
        }
    }
}

/// Refuse `need` bytes more than the `free` bytes a device reported, before
/// the driver is asked to allocate them. The figures come back structured so
/// each runtime reports them in its own error type.
pub fn free_bytes_fit(free: u64, need: u64) -> Result<(), FreeShortfall> {
    if need > free {
        Err(FreeShortfall { need, free })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    enum E {
        Failed(u32),
        TimedOut,
    }

    #[test]
    fn a_ready_wait_returns_at_once_even_with_no_time_left() {
        let mut calls = 0;
        let got = poll_until(
            Duration::ZERO,
            Duration::from_secs(60),
            || {
                calls += 1;
                Ok::<_, E>(true)
            },
            |_| E::TimedOut,
        );
        assert_eq!(got, Ok(()));
        assert_eq!(calls, 1);
    }

    #[test]
    fn a_wait_that_never_finishes_times_out_within_its_bound() {
        let timeout = Duration::from_millis(30);
        let start = Instant::now();
        let mut calls = 0u32;
        let got = poll_until(
            timeout,
            // An interval far past the timeout: the sleep must be cut to the
            // deadline, not run for a minute.
            Duration::from_secs(60),
            || {
                calls += 1;
                Ok::<_, E>(false)
            },
            |waited| {
                assert!(waited >= timeout, "{waited:?}");
                E::TimedOut
            },
        );
        assert_eq!(got, Err(E::TimedOut));
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "{:?}",
            start.elapsed()
        );
        assert!(calls >= 2, "{calls}");
    }

    #[test]
    fn the_checks_error_stops_the_wait_and_is_returned() {
        let mut calls = 0u32;
        let got = poll_until(
            Duration::from_secs(60),
            Duration::from_millis(1),
            || {
                calls += 1;
                if calls == 3 {
                    Err(E::Failed(calls))
                } else {
                    Ok(false)
                }
            },
            |_| E::TimedOut,
        );
        assert_eq!(got, Err(E::Failed(3)));
    }

    /// Readiness that arrives after some polls ends the wait then, and a
    /// zero interval does not spin forever past the deadline.
    #[test]
    fn stress_readiness_at_every_poll_count_and_zero_intervals() {
        for ready_at in 1..=200u32 {
            let mut calls = 0u32;
            let got = poll_until(
                Duration::from_secs(10),
                Duration::ZERO,
                || {
                    calls += 1;
                    Ok::<_, E>(calls >= ready_at)
                },
                |_| E::TimedOut,
            );
            assert_eq!(got, Ok(()), "{ready_at}");
            assert_eq!(calls, ready_at);
        }
        let start = Instant::now();
        let got = poll_until(
            Duration::from_millis(5),
            Duration::ZERO,
            || Ok::<_, E>(false),
            |_| E::TimedOut,
        );
        assert_eq!(got, Err(E::TimedOut));
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_need_past_free_bytes_is_a_shortfall_and_equal_fits() {
        assert_eq!(free_bytes_fit(64, 64), Ok(()));
        assert_eq!(free_bytes_fit(64, 0), Ok(()));
        assert_eq!(free_bytes_fit(0, 0), Ok(()));
        assert_eq!(
            free_bytes_fit(32, 64),
            Err(FreeShortfall { need: 64, free: 32 })
        );
        assert_eq!(
            free_bytes_fit(u64::MAX - 1, u64::MAX),
            Err(FreeShortfall {
                need: u64::MAX,
                free: u64::MAX - 1,
            })
        );
    }

    #[test]
    fn a_shortfall_is_capacity_for_the_named_device_with_both_figures() {
        let err = FreeShortfall { need: 16, free: 8 }.capacity(Device::Hip, "hipMemGetInfo");
        match &err {
            DeviceError::Capacity {
                kind: Device::Hip,
                detail,
            } => {
                assert!(detail.contains("hipMemGetInfo"), "{detail}");
                assert!(detail.contains("16 device bytes"), "{detail}");
                assert!(detail.contains("8 bytes free"), "{detail}");
            }
            other => panic!("{other}"),
        }
    }
}
