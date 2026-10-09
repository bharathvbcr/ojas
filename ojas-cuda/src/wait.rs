//! The crate's one bounded wait.
//!
//! Every wait on the device polls a readiness check until a timeout and then
//! fails with the caller's error, never blocking without a bound: an
//! unbounded `cuStreamSynchronize` on a hung or lost device would hang the
//! process. [`crate::runtime::wait_stream`] and the device probe's
//! affine check both wait through [`poll_until`].

use std::time::{Duration, Instant};

/// Poll `ready` every `interval` until it reports `true`, it fails, or
/// `timeout` has passed, whichever comes first. `ready` is always called at
/// least once, so a zero timeout still answers an already-finished wait. A
/// timeout is `timed_out(waited)`. No sleep runs past the deadline.
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
}
