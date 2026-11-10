//! e7adv: adversarial tests of ft-757cb6b2 E7 (a Metal event-wait timeout is the
//! retryable `OjasError::WaitTimeout`, and ops land exactly once around it).
//!
//! The setup is `wait_timeout_tests`' (a calibrated GEMM backlog, then a 1 ms
//! budget at exactly the wait under test), plus four test-only hooks served
//! outside `command` (`Cmd::ForceWaitAt`, `FillSlab`, `SetFreed`,
//! `GpuOutstanding`) that reach waits the trigger hook cannot: a scratch
//! recycle, a full status slab, a freed-bytes recycle, an upload.
//!
//! A test that fails here is a finding against b3/e7. Every test gathers its
//! outcomes, restores the budget and lets the GPU drain before it asserts, so
//! a failing assertion never leaks a runtime that is still busy.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::*;
use crate::link::{Reply, Wait};
use ojas_core::{AdamWConfig, BackendFloat, MuonNs5Config};

const MIN_CHAIN_MS: u128 = 150;
const AMPLE_MS: u64 = 120_000;
const SIDE: usize = 4096;
const MAX_CHAIN: usize = 64;

fn host() -> Budget {
    Budget::new(1 << 30)
}

fn open() -> MetalBackend {
    let m = MetalBackend::new(Budget::new(1 << 36)).expect("Metal device");
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

fn flag(m: &MetalBackend, cmd: Cmd) -> bool {
    match hook(m, cmd) {
        Reply::Flag(p) => p,
        other => panic!("{other:?}"),
    }
}

fn poisoned(m: &MetalBackend) -> bool {
    flag(m, Cmd::RuntimePoisoned)
}

/// Whether the GPU still has submitted work tessl has not seen finish.
fn outstanding(m: &MetalBackend) -> bool {
    flag(m, Cmd::GpuOutstanding)
}

fn force_at(m: &MetalBackend, why: Wait, budget_ms: u64) {
    assert!(matches!(
        hook(m, Cmd::ForceWaitAt { why, budget_ms }),
        Reply::Done
    ));
}

fn force_next(m: &MetalBackend) {
    assert!(matches!(
        hook(m, Cmd::ForceWaitNext { budget_ms: 1 }),
        Reply::Done
    ));
}

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

    fn record(&self, m: &MetalBackend) -> Vec<Tensor> {
        (0..self.n)
            .map(|_| m.linear_forward(&self.x, &self.w).expect("chain GEMM"))
            .collect()
    }
}

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
            assert!(n < MAX_CHAIN, "{MAX_CHAIN} GEMMs took only {took:?}");
            n *= 2;
        }
    })
}

fn is_timeout<T>(r: &Result<T, OjasError>) -> bool {
    matches!(
        r,
        Err(OjasError::WaitTimeout {
            backend: BackendId::Metal,
            waited_ms: 1,
            ..
        })
    )
}

fn is_lost<T>(r: &Result<T, OjasError>) -> bool {
    matches!(
        r,
        Err(OjasError::DeviceLost {
            backend: BackendId::Metal,
            ..
        })
    )
}

fn show<T>(r: &Result<T, OjasError>) -> String {
    match r {
        Ok(_) => "Ok".to_string(),
        Err(e) => format!("{e:?}"),
    }
}

fn ramp(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32 * 0.37).sin() * scale).collect()
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

/// Let the GPU finish and leave the runtime's budget ample, whatever state a
/// test left it in.
fn drain(m: &MetalBackend, took: Duration) {
    set_budget(m, AMPLE_MS);
    std::thread::sleep(took * 2);
}

struct Case {
    name: &'static str,
    make: fn(&MetalBackend) -> Vec<Tensor>,
    apply: fn(&MetalBackend, &mut [Tensor]) -> Result<(), OjasError>,
    /// Whether the op allocates scratch on the device (a recycle can happen
    /// inside it).
    scratch: bool,
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
            scratch: false,
        },
        Case {
            name: "muon_ns5_step(8x16)",
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
            scratch: true,
        },
        Case {
            name: "muon_ns5_step(16x8 transposed)",
            make: |m| {
                vec![
                    put(m, &ramp(16 * 8, 0.5), &[16, 8]),
                    put(m, &ramp(16 * 8, 0.2), &[16, 8]),
                    put(m, &ramp(16 * 8, 0.1), &[16, 8]),
                ]
            },
            apply: |m, t| {
                let [p, g, mo] = t else { unreachable!() };
                m.muon_ns5_step(p, g, mo, MuonNs5Config::nanolab_default())
            },
            scratch: true,
        },
        Case {
            name: "accumulate_grad(in place)",
            make: |m| vec![put(m, &ramp(64, 1.0), &[64]), put(m, &ramp(64, 0.3), &[64])],
            apply: |m, t| {
                let [acc, g] = t else { unreachable!() };
                m.accumulate_grad(acc, g)
            },
            scratch: false,
        },
        Case {
            name: "accumulate_grad(shared accumulator)",
            make: |m| {
                let acc = put(m, &ramp(64, 1.0), &[64]);
                let keep = acc.clone();
                vec![acc, put(m, &ramp(64, 0.3), &[64]), keep]
            },
            apply: |m, t| {
                let [acc, g, _keep] = t else { unreachable!() };
                m.accumulate_grad(acc, g)
            },
            scratch: true,
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
            scratch: false,
        },
    ]
}

#[derive(Clone, Copy, Debug)]
enum Site {
    /// `recycle_freed`, before the op runs.
    Freed,
    /// `status`, a full slab: before the op's first dispatch.
    SlabFull,
    /// The `k`-th scratch allocation of the op fails once, so `retrying`
    /// takes a waited commit (`recycle`) in the middle of the op.
    Recycle(u32),
}

fn arm(m: &MetalBackend, site: Site) {
    match site {
        Site::Freed => {
            hook(m, Cmd::SetFreed { bytes: (1 << 36) + 1 });
            force_at(m, Wait::Freed, 1);
        }
        Site::SlabFull => {
            hook(m, Cmd::FillSlab);
            force_at(m, Wait::SlabFull, 1);
        }
        Site::Recycle(k) => {
            hook(
                m,
                Cmd::Tune {
                    overlap: None,
                    ws_limit: None,
                    fail_allocs: (k, 1),
                },
            );
            force_at(m, Wait::Recycle, 1);
        }
    }
}

enum Reach {
    /// The wait was taken and timed out; the op failed, nothing landed, and
    /// one retry landed exactly once.
    Hit,
    /// The site never fired: the op ran to completion and landed once.
    Missed,
}

/// One op, one wait site: a timeout there, then a retry after a completed
/// wait. The failure text names what went wrong.
fn run_site(
    case: &Case,
    site: Site,
    n: usize,
    initial: &[Vec<u32>],
    reference: &[Vec<u32>],
) -> Result<Reach, String> {
    let m = open();
    let mut t = (case.make)(&m);
    let outs = Chain::new(&m, n).record(&m);
    arm(&m, site);
    let first = (case.apply)(&m, &mut t);
    let timeouts = m.link.timeouts();
    let lost = poisoned(&m);
    // A completed wait, then read: the failed op's own recorded dispatches
    // have run by now, so anything they applied shows.
    set_budget(&m, AMPLE_MS);
    let settled = m.sync();
    let after_first = bits(&t);
    let out = (|| {
        if lost {
            return Err("the runtime was poisoned by one timeout".to_string());
        }
        settled.map_err(|e| format!("sync after raising the budget: {e:?}"))?;
        if !is_timeout(&first) {
            if first.is_err() {
                return Err(format!("the op failed with something else: {}", show(&first)));
            }
            if after_first != reference {
                return Err("the site never fired, and the op did not land exactly once".into());
            }
            return Ok(Reach::Missed);
        }
        if timeouts != 1 {
            return Err(format!("timeouts counted {timeouts}, expected 1"));
        }
        if after_first != initial {
            return Err(
                "the op returned the retryable WaitTimeout but its effect had already landed: \
                 a retry applies it twice"
                    .into(),
            );
        }
        let again = (case.apply)(&m, &mut t);
        again.map_err(|e| format!("the retry failed: {e:?}"))?;
        m.sync().map_err(|e| format!("sync after the retry: {e:?}"))?;
        if bits(&t) != reference {
            return Err("after the retry the tensors are not one application of the op".into());
        }
        Ok(Reach::Hit)
    })();
    drop(outs);
    out
}

/// Attack 1. Inject the timeout at every wait an op can take before its last
/// dispatch, then retry after a completed wait, and compare with one
/// application on a fresh backend.
#[test]
fn a_timeout_at_a_wait_inside_an_op_leaves_nothing_applied_and_a_retry_lands_once() {
    let (n, _) = chain_len();
    let mut failures = Vec::new();
    let mut coverage = Vec::new();
    for case in cases() {
        let fresh = open();
        let mut t = (case.make)(&fresh);
        let initial = bits(&t);
        (case.apply)(&fresh, &mut t).expect("reference apply");
        fresh.sync().expect("reference sync");
        let reference = bits(&t);
        let mut twice = (case.make)(&fresh);
        (case.apply)(&fresh, &mut twice).expect("first");
        (case.apply)(&fresh, &mut twice).expect("second");
        fresh.sync().expect("twice sync");
        if case.name != "kv_cache_write" {
            assert_ne!(
                bits(&twice),
                reference,
                "{}: a double application would not show in the bits",
                case.name
            );
        }
        drop(fresh);

        for site in [Site::Freed, Site::SlabFull] {
            match run_site(&case, site, n, &initial, &reference) {
                Ok(Reach::Hit) => coverage.push(format!("{} @ {site:?}: hit", case.name)),
                Ok(Reach::Missed) => {
                    failures.push(format!("{} @ {site:?}: the wait was never taken", case.name))
                }
                Err(e) => failures.push(format!("{} @ {site:?}: {e}", case.name)),
            }
        }
        let mut sites = 0;
        for k in 0..16 {
            match run_site(&case, Site::Recycle(k), n, &initial, &reference) {
                Ok(Reach::Hit) => sites += 1,
                Ok(Reach::Missed) => break,
                Err(e) => failures.push(format!("{} @ Recycle({k}): {e}", case.name)),
            }
        }
        coverage.push(format!("{} @ Recycle: {sites} scratch-allocation sites hit", case.name));
        if case.scratch && sites < 1 {
            failures.push(format!("{}: no scratch recycle site was reachable", case.name));
        }
        if !case.scratch && sites != 0 {
            failures.push(format!(
                "{}: expected no scratch allocation, hit {sites} recycle sites",
                case.name
            ));
        }
        if case.name.starts_with("muon") && sites < 8 {
            failures.push(format!("{}: only {sites} recycle sites (expected >= 8)", case.name));
        }
    }
    eprintln!("{}", coverage.join("\n"));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The op the held-timeout attacks use: `accumulate_grad` (the bits show a
/// double or a missing application).
fn held_op() -> Case {
    cases()
        .into_iter()
        .find(|c| c.name == "accumulate_grad(in place)")
        .expect("case")
}

/// Attack 2. The op returned `Ok` over a held timeout. Each action is a path
/// that could clear the held state without a completed wait. After it the
/// work must still be reported pending (a timeout), never `Ok`.
#[test]
fn a_held_timeout_is_not_cleared_by_anything_but_a_completed_wait() {
    let (n, took) = chain_len();
    let case = held_op();
    let reference = {
        let m = open();
        let mut t = (case.make)(&m);
        (case.apply)(&m, &mut t).expect("apply");
        m.sync().expect("sync");
        bits(&t)
    };
    struct Act {
        name: &'static str,
        /// Whether the action is itself a wait (its own timeout is the second).
        waits: bool,
        /// What the action returns when it is itself a wait: `Some(its result
        /// is a timeout)`; `None` for an action that does not wait, whose
        /// follow-up `sync` is the second timeout.
        run: fn(&MetalBackend, &mut Vec<Tensor>) -> Option<String>,
    }
    let acts = [
        Act {
            name: "overlap commit(false) from the next ops",
            waits: false,
            run: |m, t| {
                hook(m, Cmd::Tune { overlap: Some(1), ws_limit: None, fail_allocs: (0, 0) });
                let [acc, g] = &mut t[..] else { unreachable!() };
                m.accumulate_grad(acc, g).expect("next op");
                None
            },
        },
        Act {
            name: "free, memory query, mem cap, inline toggle, pool cap",
            waits: false,
            run: |m, _| {
                drop(put(m, &[1.0; 8], &[8]));
                m.memory().expect("memory");
                m.set_mem_cap(1 << 36).expect("mem cap");
                m.set_inline_uploads(true).expect("inline");
                assert!(matches!(hook(m, Cmd::SetPoolCap { bytes: 0 }), Reply::Done));
                assert!(matches!(hook(m, Cmd::SetPoolCap { bytes: 1 << 28 }), Reply::Done));
                None
            },
        },
        Act {
            name: "a later op that allocates",
            waits: false,
            run: |m, t| {
                let out = m.silu_forward(&t[1]).map(drop);
                assert!(out.is_ok(), "silu after a held timeout: {}", show(&out));
                None
            },
        },
        Act {
            name: "trim_pool",
            waits: true,
            run: |m, _| {
                let r = m.trim_pool();
                (!is_timeout(&r)).then(|| format!("trim_pool: {}", show(&r)))
            },
        },
        Act {
            name: "with_runtime (a provider run)",
            waits: true,
            run: |m, _| {
                let ran = Arc::new(AtomicUsize::new(0));
                let seen = Arc::clone(&ran);
                let r = m.with_runtime(move |_| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                });
                if !is_timeout(&r) {
                    return Some(format!("with_runtime: {}", show(&r)));
                }
                (ran.load(Ordering::SeqCst) != 0).then(|| "the provider ran behind a timed-out settle".into())
            },
        },
        Act {
            name: "clip_grad_norm",
            waits: true,
            run: |m, t| {
                let r = m.clip_grad_norm(&mut t[..1], 1.0).map(drop);
                (!is_timeout(&r)).then(|| format!("clip_grad_norm: {}", show(&r)))
            },
        },
        Act {
            name: "to_host (a download)",
            waits: true,
            run: |_, t| {
                let r = t[0].to_host(&host()).map(drop);
                (!is_timeout(&r)).then(|| format!("to_host over pending work: {}", show(&r)))
            },
        },
        Act {
            name: "download (to_host then sync)",
            waits: true,
            run: |m, t| {
                let r = m.download(&t[0]).map(drop);
                (!is_timeout(&r)).then(|| format!("download over pending work: {}", show(&r)))
            },
        },
    ];
    let mut failures = Vec::new();
    for act in acts {
        let m = open();
        let mut t = (case.make)(&m);
        let outs = Chain::new(&m, n).record(&m);
        force_next(&m);
        let held = (case.apply)(&m, &mut t);
        let after_held = (held.is_ok(), m.link.timeouts());
        let waited = (act.run)(&m, &mut t);
        let mut follow = None;
        if !act.waits {
            follow = Some(m.sync());
        }
        let pending = outstanding(&m);
        let lost = poisoned(&m);
        drain(&m, took);
        let settled = m.sync();
        let tmo = m.link.timeouts();
        let landed = if lost { Vec::new() } else { bits(&t[..2]) };
        drop(outs);
        if lost {
            failures.push(format!("{}: poisoned (classified timeouts {tmo}, held={after_held:?}, follow={:?})", act.name, follow.as_ref().map(show)));
            continue;
        }
        if after_held != (true, 1) {
            failures.push(format!("{}: the op did not return Ok over one timeout: {after_held:?}", act.name));
        }
        if let Some(w) = waited {
            failures.push(format!("{}: {w}", act.name));
        }
        let _ = act.waits;
        if let Some(f) = &follow {
            if !is_timeout(f) {
                failures.push(format!(
                    "{}: the next sync returned {} while the GPU still had work outstanding={pending}",
                    act.name,
                    show(f)
                ));
            }
        }
        if !pending {
            failures.push(format!("{}: the GPU finished before the check (setup too short)", act.name));
        }
        if lost {
            failures.push(format!("{}: poisoned", act.name));
        }
        if settled.is_err() {
            failures.push(format!("{}: the final sync: {}", act.name, show(&settled)));
        }
        let expect: Vec<Vec<u32>> = if act.name.starts_with("overlap") {
            // Two applications on purpose: the action applied the op again.
            let m2 = open();
            let mut t2 = (case.make)(&m2);
            (case.apply)(&m2, &mut t2).expect("a");
            (case.apply)(&m2, &mut t2).expect("b");
            m2.sync().expect("sync");
            bits(&t2)
        } else {
            reference.clone()
        };
        if landed != expect {
            failures.push(format!("{}: the tensors are not the expected applications", act.name));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Attack 3a. Timeouts from different sites (an op trigger, a read, an upload)
/// share tessl's one streak: a completed wait resets it, and the third
/// consecutive one, whatever the site, is `DeviceLost`, after which no call
/// returns `Ok` or `WaitTimeout`.
#[test]
fn timeouts_from_mixed_sites_share_one_streak_and_the_third_is_device_lost() {
    let (n, took) = chain_len();
    let m = open();
    m.set_inline_uploads(false).expect("inline off");
    let case = held_op();
    let mut t = (case.make)(&m);
    let chain = Chain::new(&m, n);
    let src = Tensor::from_f32(&ramp(64, 1.0), &[64], &host()).expect("host");
    let mut grads = vec![put(&m, &[1.0; 4], &[4])];

    let mut log = Vec::new();
    // Phase 1: op trigger, read, then a completed wait.
    let outs = chain.record(&m);
    force_next(&m);
    let held = (case.apply)(&m, &mut t);
    log.push(("phase1 op trigger", held.is_ok() && m.link.timeouts() == 1));
    let read = outs[0].to_host(&host()).map(drop);
    log.push(("phase1 read", is_timeout(&read)));
    log.push(("phase1 not poisoned", !poisoned(&m)));
    set_budget(&m, AMPLE_MS);
    let reset = m.sync();
    log.push(("phase1 completed wait", reset.is_ok()));
    drop(outs);

    // Phase 2: op trigger, upload, then the third.
    let outs = chain.record(&m);
    force_next(&m);
    let held2 = (case.apply)(&m, &mut t);
    log.push(("phase2 op trigger", held2.is_ok() && m.link.timeouts() == 3));
    let up = m.upload(&src).map(drop);
    log.push(("phase2 upload", is_timeout(&up)));
    log.push(("phase2 not poisoned", !poisoned(&m)));
    let third = m.sync();
    log.push(("phase2 third is DeviceLost", is_lost(&third)));
    log.push(("poisoned", poisoned(&m)));
    let counted = m.link.timeouts();

    let ran = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&ran);
    let later: Vec<(&str, bool, String)> = {
        let r: Vec<(&str, Result<(), OjasError>)> = vec![
            ("sync", m.sync()),
            ("trim_pool", m.trim_pool()),
            ("set_mem_cap", m.set_mem_cap(1 << 36)),
            ("accumulate_grad", {
                let [acc, g] = &mut t[..] else { unreachable!() };
                m.accumulate_grad(acc, g)
            }),
            ("clip_grad_norm", m.clip_grad_norm(&mut grads, 1.0).map(drop)),
            ("upload", m.upload(&src).map(drop)),
            ("download", m.download(&t[0]).map(drop)),
            ("to_host", t[0].to_host(&host()).map(drop)),
            (
                "with_runtime",
                m.with_runtime(move |_| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
            ),
        ];
        r.into_iter()
            .map(|(name, r)| (name, is_lost(&r), show(&r)))
            .collect()
    };
    drain(&m, took);
    drop(outs);
    for (name, ok) in &log {
        assert!(ok, "{name}: log {log:?}; third={}", show(&third));
    }
    assert_eq!(counted, 4, "the third timeout of the streak is tessl's poison, not a counted retry");
    for (name, lost, text) in &later {
        assert!(lost, "{name} after the third timeout: {text}");
    }
    assert_eq!(ran.load(Ordering::SeqCst), 0, "a provider ran on a poisoned runtime");
}

/// Attack 3b. Nothing ojas does resets tessl's streak without a completed
/// wait: after two timeouts, work that does not wait (a memory query, a free,
/// cap and trigger changes, an op that allocates) leaves the next wait the
/// third.
#[test]
fn nothing_that_does_not_wait_resets_the_streak() {
    let (n, took) = chain_len();
    let m = open();
    let case = held_op();
    let mut t = (case.make)(&m);
    let outs = Chain::new(&m, n).record(&m);
    force_next(&m);
    let held = (case.apply)(&m, &mut t);
    let first = m.link.timeouts();
    let second = m.sync();
    let second_ok = is_timeout(&second) && m.link.timeouts() == 2;

    m.memory().expect("memory");
    drop(put(&m, &[1.0; 8], &[8]));
    let cap = m.mem_cap().expect("cap");
    m.set_mem_cap(cap).expect("cap");
    m.set_overlap_dispatches(256).expect("overlap");
    m.set_inline_uploads(true).expect("inline");
    let silu = m.silu_forward(&t[1]).map(drop);
    let (acc, g) = {
        let [acc, g] = &mut t[..] else { unreachable!() };
        (acc, g)
    };
    let more = m.accumulate_grad(acc, g);
    let still_two = !poisoned(&m);
    let third = m.sync();
    let lost = is_lost(&third);
    let after = poisoned(&m);
    drain(&m, took);
    drop(outs);
    assert!(held.is_ok() && first == 1, "{}", show(&held));
    assert!(second_ok, "second: {}", show(&second));
    assert!(silu.is_ok() && more.is_ok(), "{}, {}", show(&silu), show(&more));
    assert!(still_two, "poisoned before the third timeout");
    assert!(lost, "third after non-waiting work: {}", show(&third));
    assert!(after);
}

/// Attack 3c (provider site) and attack 2 (a provider run). A provider whose
/// own final wait times out: the closure already ran to the end, so the call
/// reports `WaitTimeout` for work that is done and still running. Retrying the
/// call is the documented way to wait again, and the next `sync` is the
/// documented place the timeout is reported.
#[test]
fn a_provider_run_whose_final_wait_times_out_is_held_and_a_retry_does_not_rerun_it() {
    use tessl::gemm::GemmOperands;
    use tessl::tensor::Tensor as TT;
    use tessl::DType;

    let (n, took) = chain_len();
    let m = open();
    let runs = Arc::new(AtomicUsize::new(0));
    let provider = |runs: Arc<AtomicUsize>, budget: u64| {
        move |rt: &Arc<tessl::GpuRuntime>| -> Result<(), OjasError> {
            runs.fetch_add(1, Ordering::SeqCst);
            rt.set_async_encode(true).map_err(|e| metal_err(e))?;
            let bytes = SIDE * SIDE * 4;
            let side = |b| {
                TT::from_buffer(rt, b, &[SIDE, SIDE], DType::F32, 0).map_err(|e| metal_err(e))
            };
            let (x, w, o) = (
                side(rt.alloc_buffer(bytes).map_err(|e| metal_err(e))?)?,
                side(rt.alloc_buffer(bytes).map_err(|e| metal_err(e))?)?,
                side(rt.alloc_buffer(bytes).map_err(|e| metal_err(e))?)?,
            );
            for _ in 0..n * 2 {
                GemmOperands::ExactF32
                    .nn(&x, &w, &o)
                    .map_err(|e| metal_err(e))?;
            }
            rt.set_event_wait_ms(budget).map_err(|e| metal_err(e))?;
            Ok(())
        }
    };
    let first = m.with_runtime(provider(Arc::clone(&runs), 1));
    let pending = outstanding(&m);
    let sync = m.sync();
    let still = outstanding(&m);
    let poisoned_now = poisoned(&m);
    drain(&m, took);
    let settled = m.sync();
    let retry_runs = Arc::clone(&runs);
    let retry = m.with_runtime(provider(Arc::clone(&runs), AMPLE_MS));
    let total_runs = retry_runs.load(Ordering::SeqCst);
    drain(&m, took);

    assert!(is_timeout(&first), "the provider's final wait: {}", show(&first));
    assert!(pending, "the GPU had finished: the setup was too short to test anything");
    assert!(!poisoned_now);
    assert!(
        is_timeout(&sync),
        "the next sync point returned {} while the provider's GPU work was still outstanding={still}: \
         the timeout was not held",
        show(&sync)
    );
    assert!(settled.is_ok(), "{}", show(&settled));
    assert!(retry.is_ok(), "{}", show(&retry));
    assert_eq!(
        total_runs, 1,
        "retrying the call that returned the retryable WaitTimeout ran the provider again ({total_runs} runs)"
    );
}

/// Attack 4. An upload whose wait times out is never applied and leaks no
/// budget; the retry after a completed wait is applied once.
#[test]
fn an_upload_whose_wait_times_out_applies_nothing_and_the_retry_applies_once() {
    let (n, took) = chain_len();
    let m = open();
    m.set_inline_uploads(false).expect("inline off");
    let outs = Chain::new(&m, n).record(&m);
    let vals = ramp(256, 1.0);
    let src = Tensor::from_f32(&vals, &[256], &host()).expect("host");
    let before = m.budget().live_bytes().expect("live");
    let waits_before = m.wait_counts().upload;
    force_at(&m, Wait::Upload, 1);
    let first = m.upload(&src).map(drop);
    let after_fail = m.budget().live_bytes().expect("live");
    let counted = m.link.timeouts();
    set_budget(&m, AMPLE_MS);
    let again = m.upload(&src);
    let landed = again.as_ref().ok().map(|t| bits(std::slice::from_ref(t)));
    let after_ok = m.budget().live_bytes().expect("live");
    drain(&m, took);
    drop(outs);
    assert!(is_timeout(&first), "{}", show(&first));
    assert_eq!(counted, 1);
    assert_eq!(after_fail, before, "a failed upload leaked its budget reservation");
    assert!(again.is_ok(), "{}", show(&again));
    assert_eq!(landed.expect("bits")[0], vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert_eq!(after_ok, before + 256 * 4);
    assert_eq!(m.wait_counts().upload - waits_before, 2, "waits counted: the failed attempt and the retry");
}

/// Attack 6. One backend, two threads. Thread A's op returns `Ok` over a
/// timeout; thread B's `sync` is the next sync point, so B gets the held
/// timeout (never `Ok` over pending work), and once a wait completes the op
/// has landed exactly once.
#[test]
fn a_held_timeout_is_reported_to_the_thread_that_syncs_next() {
    let (n, took) = chain_len();
    let m = open();
    let case = held_op();
    let t = (case.make)(&m);
    let outs = Chain::new(&m, n).record(&m);
    let reference = {
        let r = open();
        let mut c = (case.make)(&r);
        (case.apply)(&r, &mut c).expect("apply");
        r.sync().expect("sync");
        bits(&c)
    };
    force_next(&m);
    let (tx, rx) = std::sync::mpsc::channel();
    let a_backend = m.clone();
    let a = std::thread::spawn(move || {
        let mut t = t;
        let r = (case.apply)(&a_backend, &mut t);
        tx.send(r.is_ok()).expect("send");
        t
    });
    let a_ok = rx.recv().expect("A finished its op");
    let b = m.clone();
    let b_sync = b.sync();
    let b_pending = outstanding(&b);
    let t = a.join().expect("A");
    drain(&m, took);
    let settled = m.sync();
    let landed = bits(&t);
    drop(outs);
    assert!(a_ok, "A's op committed before its wait timed out: it is Ok");
    assert!(
        is_timeout(&b_sync),
        "B's sync returned {} while the GPU had work outstanding={b_pending}",
        show(&b_sync)
    );
    assert!(settled.is_ok());
    assert_eq!(landed, reference, "the op did not land exactly once");
}

/// Attack 6b. Interleaved: thread A applies an op 40 times (all `Ok`, no wait
/// of its own) while thread B syncs twice under a 1 ms budget. The bits are
/// 40 applications whatever the interleaving, and B never sees `Ok` over the
/// pending backlog.
#[test]
fn interleaved_ops_and_timed_out_syncs_apply_every_op_exactly_once() {
    const OPS: usize = 40;
    let (n, took) = chain_len();
    let m = open();
    let case = held_op();
    let t = (case.make)(&m);
    let reference = {
        let r = open();
        let mut c = (case.make)(&r);
        for _ in 0..OPS {
            (case.apply)(&r, &mut c).expect("apply");
        }
        r.sync().expect("sync");
        bits(&c)
    };
    let outs = Chain::new(&m, n).record(&m);
    set_budget(&m, 1);
    let a_backend = m.clone();
    let a = std::thread::spawn(move || {
        let mut t = t;
        let rs: Vec<Result<(), OjasError>> =
            (0..OPS).map(|_| (case.apply)(&a_backend, &mut t)).collect();
        let oks = rs.iter().filter(|r| r.is_ok()).count();
        let first_err = rs.iter().find(|r| r.is_err()).map(show);
        eprintln!("A first error: {first_err:?}");
        (oks, t)
    });
    let syncs: Vec<Result<(), OjasError>> = (0..2).map(|_| m.sync()).collect();
    let (oks, t) = a.join().expect("A");
    let lost = poisoned(&m);
    drain(&m, took);
    let settled = m.sync();
    let tmo = m.link.timeouts();
    let syncs_shown: Vec<String> = syncs.iter().map(show).collect();
    assert!(!lost, "poisoned: classified timeouts {tmo}, A oks {oks}, B syncs {syncs_shown:?}");
    let landed = bits(&t);
    drop(outs);
    assert_eq!(oks, OPS, "an op without a wait of its own failed");
    assert!(!lost, "two timeouts poisoned the runtime");
    for (i, s) in syncs.iter().enumerate() {
        assert!(is_timeout(s), "B's sync {i}: {}", show(s));
    }
    assert!(settled.is_ok());
    assert_eq!(landed, reference, "the {OPS} ops did not land exactly once each");
}

/// Attack 3d. The serial form of the interleaving above: two timed-out syncs,
/// then 40 ops that carry no wait of their own. Every one must be `Ok` and
/// the runtime must still be usable, because nothing ojas did waited.
#[test]
fn ops_with_no_wait_of_their_own_after_two_timeouts_all_succeed() {
    const OPS: usize = 40;
    let (n, took) = chain_len();
    let m = open();
    let case = held_op();
    let mut t = (case.make)(&m);
    let outs = Chain::new(&m, n).record(&m);
    set_budget(&m, 1);
    let syncs = [m.sync(), m.sync()];
    let results: Vec<String> = (0..OPS)
        .map(|_| (case.apply)(&m, &mut t))
        .enumerate()
        .filter_map(|(i, r)| r.err().map(|e| format!("op {i}: {e:?}")))
        .collect();
    let lost = poisoned(&m);
    let tmo = m.link.timeouts();
    drain(&m, took);
    drop(outs);
    assert!(is_timeout(&syncs[0]) && is_timeout(&syncs[1]), "{}, {}", show(&syncs[0]), show(&syncs[1]));
    assert!(
        results.is_empty() && !lost,
        "{} of {OPS} ops failed after two timeouts (poisoned={lost}, classified timeouts {tmo}); first: {:?}",
        results.len(),
        results.first()
    );
}
