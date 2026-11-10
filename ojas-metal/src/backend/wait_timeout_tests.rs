//! ft-757cb6b2 E7: a wait on the device that runs out of its budget.
//!
//! tessl bounds every wait on the GPU timeline (`set_event_wait_ms`, 1 ms at
//! the least). One that runs out is not a fault: the work is queued and the
//! runtime stays usable. The third consecutive one poisons it
//! (`CONSECUTIVE_EVENT_WAIT_TIMEOUTS` in `tessl/src/runtime.rs`).
//!
//! A timeout has to be made to happen, and tessl's own door for staging one
//! is private, so these tests use real GPU backlog, in an order that leaves
//! nothing to chance:
//!
//! 1. record a chain of GEMMs at the default budget, with the memory cap
//!    raised so nothing waits while it is recorded. The chain is calibrated
//!    once ([`chain_len`]) to take at least [`MIN_CHAIN_MS`] of GPU time, 150
//!    times the 1 ms floor of the budget;
//! 2. lower the budget to 1 ms and perform exactly the waits under test (the
//!    hooks `SetEventWaitMs` and `ForceWaitNext` are served outside
//!    `command`, so they never wait or spend a timeout themselves, and the
//!    op's own encoding never meets the small budget);
//! 3. raise the budget before the backend drops (tessl's final wait runs
//!    under the current budget and leaks what it cannot finish).
//!
//! Every test asserts the timeout happened: the typed error, and
//! `Link::timeouts`, which counts classified timeouts.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::*;
use crate::link::Reply;
use ojas_core::{AdamWConfig, BackendFloat, MuonNs5Config};

/// GPU time a calibrated chain takes at the least: 150 times the 1 ms floor.
const MIN_CHAIN_MS: u128 = 150;
/// The budget a test restores before it ends, long enough for any chain.
const AMPLE_MS: u64 = 120_000;
/// The side of one chain GEMM.
const SIDE: usize = 4096;
/// Chain length ceiling: 64 outputs of 64 MiB.
const MAX_CHAIN: usize = 64;

fn host() -> Budget {
    Budget::new(1 << 30)
}

fn open() -> MetalBackend {
    let m = MetalBackend::new(Budget::new(1 << 36)).expect("Metal device");
    // Nothing waits while a chain is recorded: a wait would finish the work
    // the test needs to still be running.
    m.set_mem_cap(1 << 36).expect("mem cap");
    m
}

fn put(m: &MetalBackend, vals: &[f32], shape: &[usize]) -> Tensor {
    m.upload(&Tensor::from_f32(vals, shape, &host()).expect("host"))
        .expect("upload")
}

fn hook(m: &MetalBackend, cmd: Cmd) -> Reply {
    m.link.call(cmd).expect("test hook")
}

fn set_budget(m: &MetalBackend, ms: u64) {
    assert!(matches!(hook(m, Cmd::SetEventWaitMs { ms }), Reply::Done));
}

fn poisoned(m: &MetalBackend) -> bool {
    match hook(m, Cmd::RuntimePoisoned) {
        Reply::Flag(p) => p,
        other => panic!("{other:?}"),
    }
}

/// The operands of a chain: a square GEMM's input and weight.
struct Chain {
    x: Tensor,
    w: Tensor,
    n: usize,
}

impl Chain {
    fn new(m: &MetalBackend, n: usize) -> Self {
        let vals: Vec<f32> = (0..SIDE * SIDE).map(|i| (i % 97) as f32 / 97.0).collect();
        Self {
            x: put(m, &vals, &[SIDE, SIDE]),
            w: put(m, &vals, &[SIDE, SIDE]),
            n,
        }
    }

    /// Record the chain. Nothing waits (the memory cap is raised), so the
    /// work is committed by the first wait that follows.
    fn record(&self, m: &MetalBackend) -> Vec<Tensor> {
        (0..self.n)
            .map(|_| m.linear_forward(&self.x, &self.w).expect("chain GEMM"))
            .collect()
    }
}

/// Chain length whose GPU time is at least [`MIN_CHAIN_MS`], and the time it
/// took, found once per process by running chains to completion. A GPU too
/// fast for [`MAX_CHAIN`] GEMMs fails loudly: a chain shorter than the
/// budget would make every timeout test pass or fail by luck.
fn chain_len() -> (usize, Duration) {
    static LEN: OnceLock<(usize, Duration)> = OnceLock::new();
    *LEN.get_or_init(|| {
        let m = open();
        let mut n = 4;
        loop {
            let chain = Chain::new(&m, n);
            let outs = chain.record(&m);
            let started = Instant::now();
            m.sync().expect("calibration sync");
            let took = started.elapsed();
            drop(outs);
            if took.as_millis() >= MIN_CHAIN_MS {
                return (n, took);
            }
            assert!(
                n < MAX_CHAIN,
                "{MAX_CHAIN} GEMMs of {SIDE}^2 took only {took:?}: too short to outlast a 1 ms budget"
            );
            n *= 2;
        }
    })
}

fn is_timeout(r: &Result<(), OjasError>) -> bool {
    matches!(
        r,
        Err(OjasError::WaitTimeout {
            backend: BackendId::Metal,
            waited_ms: 1,
            ..
        })
    )
}

/// Criterion 1: a waited commit whose wait exceeds the budget is the typed
/// timeout, tessl's runtime is not poisoned, and once the budget is raised
/// the next `sync` succeeds.
#[test]
fn a_waited_commit_past_the_budget_is_the_typed_timeout_and_the_runtime_is_not_poisoned() {
    let (n, _) = chain_len();
    let m = open();
    let outs = Chain::new(&m, n).record(&m);
    set_budget(&m, 1);
    let first = m.sync();
    assert!(is_timeout(&first), "{first:?}");
    assert_eq!(m.link.timeouts(), 1, "the wait did not time out");
    assert!(!poisoned(&m), "one timeout poisoned the runtime");
    let text = first.unwrap_err().to_string();
    assert!(text.contains("retryable"), "{text}");

    set_budget(&m, AMPLE_MS);
    m.sync().expect("the sync after the budget was raised");
    assert!(!poisoned(&m));
    // The chain's results are real.
    let last = outs.last().expect("chain").to_host(&host()).expect("read");
    assert!(last
        .to_f32_vec()
        .expect("f32")
        .iter()
        .all(|v| v.is_finite()));
    assert_eq!(m.link.timeouts(), 1);
}

/// A read maps device bytes, so it needs a completed wait: at a timed-out
/// one it is the typed timeout, never bytes of unfinished work.
#[test]
fn a_read_whose_wait_times_out_returns_the_timeout_not_bytes() {
    let (n, _) = chain_len();
    let m = open();
    let outs = Chain::new(&m, n).record(&m);
    set_budget(&m, 1);
    let r = outs[0].to_host(&host()).map(drop);
    assert!(is_timeout(&r), "{r:?}");
    assert_eq!(m.link.timeouts(), 1);
    set_budget(&m, AMPLE_MS);
    let vals = outs[0]
        .to_host(&host())
        .expect("retry after the budget was raised");
    assert!(vals
        .to_f32_vec()
        .expect("f32")
        .iter()
        .all(|v| v.is_finite()));
}

/// Criterion 3: three consecutive timeouts are tessl's definition of a lost
/// device. The first two are the retryable timeout; the third call, and every
/// later one, is `DeviceLost`. The constant lives in tessl
/// (`CONSECUTIVE_EVENT_WAIT_TIMEOUTS`); ojas keeps no counter, so this pins
/// what tessl does through ojas's mapping.
#[test]
fn the_third_consecutive_timeout_and_every_call_after_it_is_device_lost() {
    let (n, took) = chain_len();
    let m = open();
    let outs = Chain::new(&m, n).record(&m);
    set_budget(&m, 1);
    for i in 1..=2 {
        let r = m.sync();
        assert!(is_timeout(&r), "timeout {i}: {r:?}");
        assert!(!poisoned(&m), "poisoned after {i} timeouts");
    }
    let third = m.sync();
    assert!(
        matches!(
            &third,
            Err(OjasError::DeviceLost {
                backend: BackendId::Metal,
                ..
            })
        ),
        "third: {third:?}"
    );
    assert!(poisoned(&m));
    assert_eq!(
        m.link.timeouts(),
        2,
        "the third timeout is tessl's poison, not a retryable one"
    );
    let fresh = Tensor::from_f32(&[3.0], &[1], &host()).expect("host");
    let later = [
        ("sync", m.sync()),
        ("silu_forward", m.silu_forward(&outs[0]).map(drop)),
        ("upload", m.upload(&fresh).map(drop)),
        ("download", outs[0].to_host(&host()).map(drop)),
    ];
    for (name, r) in later {
        assert!(
            matches!(&r, Err(OjasError::DeviceLost { .. })),
            "{name} after the third timeout: {r:?}"
        );
    }
    // tessl's final wait runs under the current budget: raise it, and let the
    // GPU finish the chain so the drop leaks nothing.
    set_budget(&m, AMPLE_MS);
    std::thread::sleep(took * 2);
}

/// A completed wait between timeouts restarts tessl's streak: two timeouts,
/// a completed wait, and two more leave the runtime usable.
#[test]
fn a_completed_wait_between_timeouts_restarts_the_streak() {
    let (n, _) = chain_len();
    let m = open();
    let chain = Chain::new(&m, n);
    for round in 0..2 {
        let outs = chain.record(&m);
        set_budget(&m, 1);
        for i in 1..=2 {
            let r = m.sync();
            assert!(is_timeout(&r), "round {round} timeout {i}: {r:?}");
        }
        assert!(!poisoned(&m), "round {round}");
        set_budget(&m, AMPLE_MS);
        m.sync().expect("a completed wait");
        drop(outs);
    }
    assert!(!poisoned(&m));
    assert_eq!(m.link.timeouts(), 4);
}

/// Everything one in-place op case needs: how to make its tensors and how to
/// apply the op once.
struct Case {
    name: &'static str,
    make: fn(&MetalBackend) -> Vec<Tensor>,
    apply: fn(&MetalBackend, &mut [Tensor]) -> Result<(), OjasError>,
    /// Whether applying the op twice would show in the bits (a cache write of
    /// the same rows twice does not).
    twice_shows: bool,
}

fn ramp(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32 * 0.37).sin() * scale).collect()
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "adamw_step",
            make: |m| {
                vec![
                    put(m, &[0.5; 24], &[4, 6]),
                    put(m, &ramp(24, 0.1), &[4, 6]),
                    put(m, &[0.01; 24], &[4, 6]),
                    put(m, &[0.02; 24], &[4, 6]),
                ]
            },
            apply: |m, t| {
                let [p, g, m1, m2] = t else { unreachable!() };
                m.adamw_step(p, g, m1, m2, 3, AdamWConfig::nanolab(1e-3, 0.1))
            },
            twice_shows: true,
        },
        Case {
            name: "muon_ns5_step",
            make: |m| {
                vec![
                    put(m, &ramp(8 * 16, 0.5), &[8, 16]),
                    put(m, &ramp(8 * 16, 0.2), &[8, 16]),
                    put(m, &ramp(8 * 16, 0.1), &[8, 16]),
                ]
            },
            apply: |m, t| {
                let [p, g, mo] = t else { unreachable!() };
                m.muon_ns5_step(p, g, mo, MuonNs5Config::nanolab_default())
            },
            twice_shows: true,
        },
        Case {
            name: "accumulate_grad",
            make: |m| vec![put(m, &ramp(64, 1.0), &[64]), put(m, &ramp(64, 0.3), &[64])],
            apply: |m, t| {
                let [acc, g] = t else { unreachable!() };
                m.accumulate_grad(acc, g)
            },
            twice_shows: true,
        },
        Case {
            name: "kv_cache_write",
            make: |m| {
                vec![
                    put(m, &[0.0; 8 * 2 * 4], &[1, 8, 2, 4]),
                    put(m, &ramp(2 * 2 * 4, 1.0), &[1, 2, 2, 4]),
                ]
            },
            apply: |m, t| {
                let [cache, src] = t else { unreachable!() };
                m.kv_cache_write(cache, src, 3)
            },
            twice_shows: false,
        },
    ]
}

fn bits(t: &[Tensor]) -> Vec<Vec<u32>> {
    t.iter()
        .map(|t| {
            t.to_host(&host())
                .expect("read")
                .to_f32_vec()
                .expect("f32")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

/// Criterion 2. The op's own trigger check waits and runs out of budget after
/// the op committed. Owner decision E7 (a): the call is `Ok`, the next sync
/// point reports the (retryable) timeout while the GPU is still busy, and
/// once a wait completes the tensors hold exactly one application of the op:
/// bit-identical to one application on a fresh backend.
#[test]
fn an_in_place_op_whose_wait_times_out_is_ok_and_lands_exactly_once() {
    let (n, _) = chain_len();
    for case in cases() {
        // The reference: one application, settled, on a fresh backend.
        let reference = {
            let m = open();
            let mut t = (case.make)(&m);
            (case.apply)(&m, &mut t).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
            m.sync().expect("reference sync");
            bits(&t)
        };
        // Two applications, to know the bits would show a double apply.
        let twice = {
            let m = open();
            let mut t = (case.make)(&m);
            (case.apply)(&m, &mut t).expect("first");
            (case.apply)(&m, &mut t).expect("second");
            m.sync().expect("twice sync");
            bits(&t)
        };
        assert_eq!(
            twice != reference,
            case.twice_shows,
            "{}: a double application {} in the bits",
            case.name,
            if case.twice_shows {
                "must show"
            } else {
                "cannot show"
            }
        );

        let m = open();
        let mut t = (case.make)(&m);
        let outs = Chain::new(&m, n).record(&m);
        assert!(matches!(
            hook(&m, Cmd::ForceWaitNext { budget_ms: 1 }),
            Reply::Done
        ));
        let r = (case.apply)(&m, &mut t);
        assert!(
            r.is_ok(),
            "{}: the op committed before its wait timed out, so it is Ok: {r:?}",
            case.name
        );
        assert_eq!(
            m.link.timeouts(),
            1,
            "{}: the op's own wait did not time out",
            case.name
        );
        assert!(!poisoned(&m), "{}", case.name);
        // The next sync point reports the held timeout: its wait is a fresh
        // one on the same unfinished work.
        let held = m.sync();
        assert!(is_timeout(&held), "{}: {held:?}", case.name);
        assert_eq!(m.link.timeouts(), 2, "{}", case.name);
        set_budget(&m, AMPLE_MS);
        m.sync()
            .unwrap_or_else(|e| panic!("{}: sync after raising the budget: {e:?}", case.name));
        assert_eq!(
            bits(&t),
            reference,
            "{}: the tensors are not one application of the op",
            case.name
        );
        drop(outs);
    }
}
