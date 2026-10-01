//! Many OS threads issuing a mix of ops at once, some on one shared
//! `MetalBackend` (one device thread, commands interleaved) and some on
//! backends of their own (separate device threads on one GPU), must each get
//! exactly the bits a serial run of the same sequence gets. A watchdog turns
//! a hang into a failure.

#![cfg(target_os = "macos")]

mod common;

use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use common::*;
use ojas_core::{AdamWConfig, Backend, Tensor};
use ojas_metal::MetalBackend;

const WATCHDOG: Duration = Duration::from_secs(300);

/// Run `f` on its own thread; panic if it neither returns nor panics in time.
fn watchdog<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    let h = thread::spawn(move || {
        let out = f();
        let _ = tx.send(());
        out
    });
    match rx.recv_timeout(WATCHDOG) {
        Ok(()) => h.join().expect("worker panicked"),
        Err(mpsc::RecvTimeoutError::Disconnected) => match h.join() {
            Ok(_) => unreachable!("sender dropped without sending"),
            Err(p) => std::panic::resume_unwind(p),
        },
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("watchdog: no result after {WATCHDOG:?}; the device thread hung")
        }
    }
}

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// One thread's mixed sequence: linear, RMSNorm, attention, permute,
/// cross-entropy, and an AdamW step on its own state, `iters` times. Returns
/// the bits of every output of the last iteration plus the final parameter.
fn sequence(m: &MetalBackend, seed: u64, iters: usize) -> Vec<Vec<u32>> {
    let (rows, d) = (48usize, 64usize);
    let x = up(m, &rand(&[rows, d], seed, 1.0));
    let w = up(m, &rand(&[d, d], seed + 1, 0.2));
    let rw = up(m, &rand(&[d], seed + 2, 1.0));
    let g = up(m, &rand(&[rows, d], seed + 3, 1.0));
    let q = up(m, &rand(&[1, 2, 40, 32], seed + 4, 1.0));
    let tgt = up(m, &host_u32(&ids(rows, seed + 5, d as u32), &[rows]));
    let mut p = up(m, &rand(&[d, d], seed + 6, 0.1));
    let mut m1 = up(m, &host(&vec![0.0; d * d], &[d, d]));
    let mut m2 = up(m, &host(&vec![0.0; d * d], &[d, d]));
    let mut out = Vec::new();
    for step in 0..iters as u64 {
        out.clear();
        let y = ok("linear", m.linear_forward(&x, &w));
        let (gx, gw) = ok("linear bwd", m.linear_backward(&x, &w, &g));
        let n = ok("rms", m.rms_norm_forward(&y, &rw, 1e-6));
        let (nx, nw) = ok("rms bwd", m.rms_norm_backward(&y, &rw, &g, 1e-6));
        let a = ok("sdpa", m.causal_sdpa_forward(&q, &q, &q));
        let (aq, ak, av) = ok("sdpa bwd", m.causal_sdpa_backward(&q, &q, &q, &a));
        let t = ok("permute", m.permute(&a, &[0, 2, 1, 3]));
        let ce = ok("ce bwd", m.cross_entropy_mean_backward(&n, &tgt, None));
        ok(
            "adamw",
            m.adamw_step(&mut p, &gw, &mut m1, &mut m2, step, AdamWConfig::nanolab(1e-3, 0.1)),
        );
        for o in [&y, &gx, &gw, &n, &nx, &nw, &a, &aq, &ak, &av, &t, &ce] {
            out.push(bits(o));
        }
    }
    out.push(bits(&p));
    out.push(bits(&m2));
    out
}

#[test]
fn concurrent_mixed_ops_match_the_serial_run_bit_for_bit() {
    const SHARED: u64 = 6;
    const OWN: u64 = 2;
    const ITERS: usize = 6;
    watchdog(|| {
        let shared = Arc::new(metal());
        // Serial reference, one sequence after another on the shared backend.
        let want: Vec<_> = (0..SHARED + OWN)
            .map(|i| sequence(&shared, 1000 * (i + 1), ITERS))
            .collect();
        let mut handles = Vec::new();
        for i in 0..SHARED + OWN {
            let shared = Arc::clone(&shared);
            handles.push(thread::spawn(move || {
                if i < SHARED {
                    sequence(&shared, 1000 * (i + 1), ITERS)
                } else {
                    sequence(&metal(), 1000 * (i + 1), ITERS)
                }
            }));
        }
        for (i, h) in handles.into_iter().enumerate() {
            let got = h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
            assert_eq!(got.len(), want[i].len(), "thread {i}: output count");
            for (k, (g, w)) in got.iter().zip(&want[i]).enumerate() {
                assert!(g == w, "thread {i}: output {k} differs from the serial run");
            }
        }
    });
}
