//! wgfadv: adversarial tests of the per-read fault hold slots (ft-84e2b74e,
//! branch b3/wgf). Each test is one attack from the brief; a failure is a
//! finding. Failures are injected through the per-thread hook of
//! `context/wgfadv.rs`, orphans are seeded with `adv_set_held`.
use super::*;
use crate::context::wgfadv::{clear_hook, set_hook, Act, Stage};
use crate::context::HOLD_SLOTS;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn raw_of(t: &Tensor) -> wgpu::Buffer {
    t.device_buffer()
        .and_then(|d| d.as_any().downcast_ref::<WgpuBuffer>())
        .expect("a wgpu tensor")
        .raw()
        .expect("a live buffer")
        .clone()
}

fn raise_as(gpu: &WgpuBackend, id: u32, buf: &wgpu::Buffer, n: usize) {
    let mut job = gpu.ctx.job(&gpu.budget, id);
    job.dispatch(&CHECK_FINITE, &[n as u32], &[buf], gpu.lanes(n).unwrap())
        .unwrap();
    job.commit().unwrap();
}

/// A buffer holding a NaN. The tensor is leaked: dropped, its buffer would
/// return to the pool and a later upload would overwrite the NaN.
fn bad_buffer(gpu: &WgpuBackend, budget: &Budget) -> wgpu::Buffer {
    let t = Box::leak(Box::new(
        gpu.upload(&Tensor::from_f32(&[1.0, f32::NAN], &[2], budget).unwrap())
            .unwrap(),
    ));
    raw_of(t)
}

fn open() -> (WgpuBackend, Budget, wgpu::Buffer) {
    let budget = Budget::new(1 << 24);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let buf = bad_buffer(&gpu, &budget);
    (gpu, budget, buf)
}

fn wait_until(what: &str, limit: Duration, cond: impl Fn() -> bool) {
    let end = Instant::now() + limit;
    while !cond() {
        assert!(Instant::now() < end, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn bit(id: u32) -> u64 {
    1u64 << (id - 1)
}

fn consistent(bits: u64, first: u32) -> bool {
    (1..=64).contains(&first) && bits & bit(first) != 0
}

/// Read until a read observes nothing, bounded. Every non-empty report.
fn drain(gpu: &WgpuBackend) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    for _ in 0..64 {
        let seen = gpu.ctx.sync_observed().unwrap();
        if seen == (0, 0) {
            return out;
        }
        out.push(seen);
    }
    panic!("64 reads and a fault is still reported: {out:x?}");
}

/// How often each op bit appears across `reports`; also asserts every
/// report is a consistent pair.
fn counts(reports: &[(u64, u32)]) -> [u32; 64] {
    let mut c = [0u32; 64];
    for &(bits, first) in reports {
        assert!(consistent(bits, first), "inconsistent ({bits:#x}, {first})");
        for (i, n) in c.iter_mut().enumerate() {
            *n += u32::from(bits >> i & 1 == 1);
        }
    }
    c
}

/// The report set must name exactly `want` (ids), each once.
fn assert_each_once(reports: &[(u64, u32)], want: &[u32], what: &str) {
    let c = counts(reports);
    for (i, &n) in c.iter().enumerate() {
        let id = i as u32 + 1;
        let w = u32::from(want.contains(&id));
        assert_eq!(
            n, w,
            "{what}: op {id} reported {n} times, raised {w}: {reports:x?}"
        );
    }
}

/// No lease outstanding and no orphan: every slot is free, and all 64 lease
/// at once with no wait.
fn assert_pool_whole(gpu: &WgpuBackend, what: &str) {
    let (free, orphans) = gpu.ctx.slot_census();
    assert_eq!(
        (free.len(), orphans.len()),
        (HOLD_SLOTS as usize, 0),
        "{what}: free {free:?} orphans {orphans:?}: a slot leaked"
    );
    let leases: Vec<_> = (0..HOLD_SLOTS)
        .map(|i| {
            gpu.ctx
                .lease_hold_slot(Duration::ZERO)
                .unwrap_or_else(|e| panic!("{what}: lease {i}: {e:?}"))
        })
        .collect();
    let mut slots: Vec<u32> = leases.iter().map(|l| l.slot).collect();
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(
        slots.len(),
        HOLD_SLOTS as usize,
        "{what}: a slot was shared"
    );
}

/// One read with the hook armed to `act` at `stage`; `Ok(None)` when it
/// panicked, otherwise the read's own result.
fn hooked_read(gpu: &WgpuBackend, stage: Stage, act: Act) -> Option<Result<(u64, u32), OjasError>> {
    set_hook(move |s| if s == stage { act } else { Act::Pass });
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| gpu.ctx.sync_observed()));
    clear_hook();
    r.ok()
}

// ---------------------------------------------------------------- attack 1

/// `n` reads, one per thread, each raising its own op (id `i + 1`, in thread
/// order) and then failing after its hold ran, so all `n` slots are orphans
/// at once. The reads drop their leases in the order `drop_rank` gives.
/// Returns what the reads after that report.
fn orphan_storm(n: usize, drop_rank: impl Fn(usize) -> usize) -> Vec<(u64, u32)> {
    let (gpu, _budget, buf) = open();
    let turn = std::sync::Arc::new(AtomicUsize::new(0));
    let dropped = std::sync::Arc::new(AtomicUsize::new(0));
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .map(|i| {
                let (gpu, buf) = (&gpu, &buf);
                let (turn, dropped) = (turn.clone(), dropped.clone());
                let rank = drop_rank(i);
                s.spawn(move || {
                    wait_until("my turn", Duration::from_secs(60), || {
                        turn.load(Ordering::SeqCst) == i
                    });
                    raise_as(gpu, i as u32 + 1, buf, 2);
                    let (hturn, hdropped) = (turn.clone(), dropped.clone());
                    set_hook(move |st| {
                        if st != Stage::BeforeWait {
                            return Act::Pass;
                        }
                        hturn.fetch_add(1, Ordering::SeqCst);
                        wait_until("all recorded", Duration::from_secs(60), || {
                            hturn.load(Ordering::SeqCst) == n
                        });
                        wait_until("my drop turn", Duration::from_secs(60), || {
                            hdropped.load(Ordering::SeqCst) == rank
                        });
                        Act::Fail
                    });
                    let r = gpu.ctx.sync_observed();
                    clear_hook();
                    dropped.fetch_add(1, Ordering::SeqCst);
                    assert!(r.is_err(), "the injected failure did not fail the read");
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    });
    let (free, orphans) = gpu.ctx.slot_census();
    assert_eq!((free.len(), orphans.len()), (HOLD_SLOTS as usize - n, n));
    let reports = drain(&gpu);
    assert_eq!(gpu.ctx.torn_pairs(), 0);
    assert_pool_whole(&gpu, "after the orphans drained");
    reports
}

fn lowest_id(bits: u64) -> u32 {
    bits.trailing_zeros() + 1
}

/// Ids ascend with age in these tests, so the oldest first op of a report is
/// its lowest bit.
fn assert_oldest_first(reports: &[(u64, u32)]) {
    for &(bits, first) in reports {
        assert_eq!(
            first,
            lowest_id(bits),
            "report ({bits:#x}, first {first}) names a newer op than the oldest in it; reports {reports:x?}"
        );
    }
}

/// More than 13 orphans left by concurrent failing reads whose leases drop in
/// the order the holds were recorded: all reported once, oldest first.
#[test]
fn wgfadv_30_concurrent_failing_reads_drop_in_record_order_reported_once_oldest_first() {
    let reports = orphan_storm(30, |i| i);
    let ids: Vec<u32> = (1..=30).collect();
    assert_each_once(&reports, &ids, "30 orphans, record order");
    assert_eq!(reports[0].1, 1, "the oldest orphan leads the first report");
    assert_oldest_first(&reports);
}

/// The same 30 orphans, but the reads fail in the reverse of the order their
/// holds ran: the orphan queue is in drop order, not record order, so a newer
/// pair is folded first and its first op wins over an older one.
#[test]
fn wgfadv_30_concurrent_failing_reads_drop_in_reverse_order_first_is_still_the_oldest() {
    let reports = orphan_storm(30, |i| 29 - i);
    let ids: Vec<u32> = (1..=30).collect();
    assert_each_once(&reports, &ids, "30 orphans, reverse drop order");
    assert_oldest_first(&reports);
}

/// 20 orphans are in the queue when a read folds the oldest 13 into its own
/// slot and then fails: that slot (holding the oldest pairs) goes to the BACK
/// of the queue, behind the 7 newer orphans.
#[test]
fn wgfadv_a_folding_read_that_fails_does_not_make_the_oldest_pairs_newest() {
    let (gpu, _b, _buf) = open();
    for id in 1..=20u32 {
        gpu.ctx.adv_set_held(bit(id), id);
    }
    let r = hooked_read(&gpu, Stage::BeforeWait, Act::Fail).expect("no panic");
    assert!(r.is_err());
    let reports = drain(&gpu);
    let ids: Vec<u32> = (1..=20).collect();
    assert_each_once(&reports, &ids, "20 orphans and a failed folding read");
    assert_oldest_first(&reports);
    assert_pool_whole(&gpu, "after");
}

/// Orphans are drained by fresh reads while faults keep arriving: with a free
/// slot every read, 40 orphans take ceil(40 / 13) reads, none starves.
#[test]
fn wgfadv_orphans_drain_while_fresh_faults_arrive() {
    let (gpu, _b, buf) = open();
    let mut seeded = 0usize;
    for id in 1..=40u32 {
        gpu.ctx.adv_set_held(bit(id), id);
        seeded += 1;
    }
    let mut reports = Vec::new();
    let mut left = seeded;
    for (k, fresh) in (41..=45u32).enumerate() {
        raise_as(&gpu, fresh, &buf, 2);
        let seen = gpu.ctx.sync_observed().unwrap();
        reports.push(seen);
        left = left.saturating_sub(13);
        let (_, orphans) = gpu.ctx.slot_census();
        assert_eq!(
            orphans.len(),
            left,
            "after read {}: orphans {orphans:?}",
            k + 1
        );
    }
    reports.extend(drain(&gpu));
    let ids: Vec<u32> = (1..=45).collect();
    assert_each_once(&reports, &ids, "40 orphans and 5 fresh faults");
    assert_pool_whole(&gpu, "after");
}

// ---------------------------------------------------------------- attack 2

struct Outcome {
    reports: Vec<(u64, u32)>,
    read: &'static str,
}

/// Raise `3`, deliver it (when `stale`: its slot keeps the delivered pair),
/// optionally seed an older orphan (op 20), raise `5`, then run one read that
/// is failed at `stage`. Then drain.
fn scenario(stage: Stage, act: Act, seed_orphan: bool, stale: bool) -> Outcome {
    let (gpu, _b, buf) = open();
    if stale {
        raise_as(&gpu, 3, &buf, 2);
        assert_eq!(gpu.ctx.sync_observed().unwrap(), (bit(3), 3));
    }
    if seed_orphan {
        gpu.ctx.adv_set_held(bit(20), 20);
    }
    raise_as(&gpu, 5, &buf, 2);
    let read = match hooked_read(&gpu, stage, act) {
        None => "panicked",
        Some(Err(_)) => "errored",
        Some(Ok(_)) => "returned ok",
    };
    let reports = drain(&gpu);
    // A read that returned Ok delivered its pair itself.
    assert_eq!(gpu.ctx.take_faults(), (0, 0), "nothing deferred");
    assert_eq!(gpu.ctx.torn_pairs(), 0);
    assert_pool_whole(&gpu, &format!("{stage:?} {act:?}"));
    Outcome { reports, read }
}

fn all_stages() -> [Stage; 5] {
    [
        Stage::AfterLease,
        Stage::AfterRecord,
        Stage::BeforeWait,
        Stage::AfterWait,
        Stage::AfterMap,
    ]
}

/// A read that fails at `stage` leaves its fault for the next read: the
/// raised op 5 (and the seeded orphan 20) are reported once in total.
fn assert_failure_keeps_every_fault(stage: Stage, act: Act, seed_orphan: bool) {
    let o = scenario(stage, act, seed_orphan, false);
    let want: Vec<u32> = if seed_orphan { vec![5, 20] } else { vec![5] };
    assert_each_once(
        &o.reports,
        &want,
        &format!(
            "{stage:?} {act:?} seeded={seed_orphan}, the read {}",
            o.read
        ),
    );
}

#[test]
fn wgfadv_failure_after_lease_before_record_keeps_every_fault() {
    for act in [Act::Fail, Act::Panic] {
        for seed in [false, true] {
            assert_failure_keeps_every_fault(Stage::AfterLease, act, seed);
        }
    }
}

#[test]
fn wgfadv_failure_after_submit_before_the_wait_keeps_every_fault() {
    for act in [Act::Fail, Act::Panic] {
        for seed in [false, true] {
            assert_failure_keeps_every_fault(Stage::BeforeWait, act, seed);
        }
    }
}

#[test]
fn wgfadv_failure_after_the_wait_keeps_every_fault() {
    for act in [Act::Fail, Act::Panic] {
        for seed in [false, true] {
            assert_failure_keeps_every_fault(Stage::AfterWait, act, seed);
        }
    }
}

#[test]
fn wgfadv_failure_during_the_map_keeps_every_fault() {
    for act in [Act::Fail, Act::Panic] {
        for seed in [false, true] {
            assert_failure_keeps_every_fault(Stage::AfterMap, act, seed);
        }
    }
}

/// The existing seam: a read failed after the wait, with a seeded orphan.
#[test]
fn wgfadv_the_existing_fail_next_read_seam_keeps_a_folded_orphan() {
    let (gpu, _b, buf) = open();
    gpu.ctx.adv_set_held(bit(20), 20);
    raise_as(&gpu, 5, &buf, 2);
    gpu.ctx.fail_next_read();
    gpu.ctx.sync().expect_err("the injected read failure");
    let reports = drain(&gpu);
    assert_each_once(&reports, &[5, 20], "fail_next_read");
    assert_pool_whole(&gpu, "after");
}

/// A read that fails between `record` and the encoder gaining the hold: the
/// orphan it folded was returned to the pool, but the hold that would have
/// moved its pair never ran, so the pair is lost when that slot is reused.
#[test]
fn wgfadv_a_failure_between_record_and_the_hold_does_not_lose_a_folded_orphan() {
    for act in [Act::Fail, Act::Panic] {
        let o = scenario(Stage::AfterRecord, act, true, false);
        assert_each_once(
            &o.reports,
            &[5, 20],
            &format!("AfterRecord {act:?}: the read {}", o.read),
        );
    }
}

/// The slot a read fails with before its hold ran still holds the pair its
/// last reader delivered; it is an orphan now, so the next hold folds it and
/// reports that old pair a second time.
#[test]
fn wgfadv_a_failure_between_record_and_the_hold_does_not_repeat_a_delivered_fault() {
    for act in [Act::Fail, Act::Panic] {
        let o = scenario(Stage::AfterRecord, act, false, true);
        assert_each_once(
            &o.reports,
            &[5],
            &format!("AfterRecord {act:?}: the read {}", o.read),
        );
    }
}

/// Clean (nothing seeded, nothing stale) failure between record and the hold.
#[test]
fn wgfadv_a_clean_failure_between_record_and_the_hold_keeps_the_live_fault() {
    for act in [Act::Fail, Act::Panic] {
        let o = scenario(Stage::AfterRecord, act, false, false);
        assert_each_once(
            &o.reports,
            &[5],
            &format!("AfterRecord {act:?}: the read {}", o.read),
        );
    }
}

/// Every stage x action once more with a delivered-then-reused slot, only the
/// stages after the hold was recorded: they must overwrite the stale pair.
#[test]
fn wgfadv_stale_slot_pairs_never_reappear_after_a_failure_past_the_hold() {
    for stage in all_stages() {
        if stage == Stage::AfterRecord {
            continue;
        }
        for act in [Act::Fail, Act::Panic] {
            let o = scenario(stage, act, false, true);
            assert_each_once(
                &o.reports,
                &[5],
                &format!("{stage:?} {act:?} stale: the read {}", o.read),
            );
        }
    }
}

// ---------------------------------------------------------------- attack 3

/// 64 readers hold their leases (blocked after their submit); 16 more readers
/// and 8 raisers arrive while no slot is free; 8 of the holders then fail.
/// Nothing may deadlock, duplicate, lose, or tear; the waiting readers only
/// proceed once a lease ends.
#[test]
fn wgfadv_80_readers_64_leases_raisers_and_failures_deliver_every_fault_once() {
    let (gpu, _budget, buf) = open();
    let gpu = std::sync::Arc::new(gpu);
    let hooked = std::sync::Arc::new(AtomicUsize::new(0));
    let gate = std::sync::Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Result<(u64, u32), String>>();
    let mut handles = Vec::new();
    for i in 0..64usize {
        let (gpu, buf, hooked, gate, tx) = (
            gpu.clone(),
            buf.clone(),
            hooked.clone(),
            gate.clone(),
            tx.clone(),
        );
        handles.push(std::thread::spawn(move || {
            if i < 40 {
                raise_as(&gpu, i as u32 + 1, &buf, 2);
            }
            let fail = i % 8 == 0;
            set_hook(move |st| {
                if st != Stage::BeforeWait {
                    return Act::Pass;
                }
                hooked.fetch_add(1, Ordering::SeqCst);
                wait_until("the gate", Duration::from_secs(60), || {
                    gate.load(Ordering::SeqCst)
                });
                if fail {
                    Act::Fail
                } else {
                    Act::Pass
                }
            });
            let r = gpu.ctx.sync_observed().map_err(|e| format!("{e:?}"));
            clear_hook();
            let _ = tx.send(r);
        }));
    }
    wait_until("64 leased reads", Duration::from_secs(60), || {
        hooked.load(Ordering::SeqCst) == 64
    });
    let (free, orphans) = gpu.ctx.slot_census();
    assert_eq!(
        (free.len(), orphans.len()),
        (0, 0),
        "all 64 slots are leased"
    );
    for _ in 0..16 {
        let (gpu, tx) = (gpu.clone(), tx.clone());
        handles.push(std::thread::spawn(move || {
            let r = gpu.ctx.sync_observed().map_err(|e| format!("{e:?}"));
            let _ = tx.send(r);
        }));
    }
    for k in 0..8u32 {
        let (gpu, buf) = (gpu.clone(), buf.clone());
        handles.push(std::thread::spawn(move || raise_as(&gpu, 41 + k, &buf, 2)));
    }
    assert!(
        rx.recv_timeout(Duration::from_millis(400)).is_err(),
        "a read finished while 64 leases were held and gated"
    );
    gate.store(true, Ordering::SeqCst);
    let mut reports = Vec::new();
    let mut errs = 0;
    for n in 0..80 {
        match rx.recv_timeout(Duration::from_secs(90)) {
            Ok(Ok(p)) if p != (0, 0) => reports.push(p),
            Ok(Ok(_)) => {}
            Ok(Err(_)) => errs += 1,
            Err(e) => panic!("read {n} of 80 never finished ({e:?}): deadlock or lost wakeup"),
        }
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(errs, 8, "the eight gated failures");
    reports.extend(drain(&gpu));
    let ids: Vec<u32> = (1..=48).collect();
    assert_each_once(&reports, &ids, "80 readers");
    assert_eq!(gpu.ctx.torn_pairs(), 0, "raw torn pairs");
    assert_eq!(gpu.ctx.take_faults(), (0, 0));
    assert_pool_whole(&gpu, "after the storm");
}

/// Every waiter on a full pool wakes: 64 leases are released all at once, or
/// one by one, to 70 waiters that each lease and drop at once (a chain of
/// `notify_one`).
#[test]
fn wgfadv_every_waiter_on_a_full_pool_wakes_without_waiting_out_the_timeout() {
    let (gpu, _b, _buf) = open();
    for round in 0..6 {
        let leases: Vec<_> = (0..HOLD_SLOTS)
            .map(|_| gpu.ctx.lease_hold_slot(Duration::ZERO).unwrap())
            .collect();
        let started = Instant::now();
        std::thread::scope(|s| {
            let waiters: Vec<_> = (0..70)
                .map(|_| {
                    s.spawn(|| {
                        let l = gpu.ctx.lease_hold_slot(Duration::from_secs(30));
                        l.map(|_| ()).map_err(|e| format!("{e:?}"))
                    })
                })
                .collect();
            std::thread::sleep(Duration::from_millis(100));
            if round % 2 == 0 {
                drop(leases);
            } else {
                for l in leases {
                    drop(l);
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            for (i, w) in waiters.into_iter().enumerate() {
                w.join()
                    .unwrap()
                    .unwrap_or_else(|e| panic!("round {round}: waiter {i} failed: {e}"));
            }
        });
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "round {round}: a waiter slept through a free slot ({:?})",
            started.elapsed()
        );
    }
}

// ---------------------------------------------------------------- attack 4

/// A fault reaches exactly one channel: one raised op per round, read at once
/// through every path that reads fault words (the backend `sync`, the shared-
/// pair `ctx.sync`, `download`, `to_host`, `clip_grad_norm`, `adamw_step`,
/// `take_faults`). Reports, over all of them and a final drain, number one.
#[test]
fn wgfadv_every_channel_together_delivers_each_fault_exactly_once() {
    let (gpu, budget, buf) = open();
    let cfg = AdamWConfig::nanolab(1e-2, 0.1);
    for round in 0..25u32 {
        // Inside the 38-name op table: a larger id is refused by the host.
        let id = 1 + (round * 7) % 38;
        raise_as(&gpu, id, &buf, 2);
        let delivered = AtomicUsize::new(0);
        let log = std::sync::Mutex::new(Vec::<String>::new());
        let other = AtomicUsize::new(0);
        let tally = |k: &str, r: Result<(), OjasError>| {
            match &r {
                Ok(()) => {}
                Err(OjasError::NonFinite { .. }) => {
                    delivered.fetch_add(1, Ordering::SeqCst);
                }
                Err(_) => {
                    other.fetch_add(1, Ordering::SeqCst);
                }
            }
            log.lock().unwrap().push(format!("{k}: {r:?}"));
        };
        let took = |k: &str, (bits, first): (u64, u32)| {
            delivered.fetch_add(bits.count_ones() as usize, Ordering::SeqCst);
            log.lock()
                .unwrap()
                .push(format!("{k}: take_faults ({bits:#x}, {first})"));
        };
        std::thread::scope(|s| {
            for k in 0..14usize {
                let (gpu, budget, tally, took, log) = (&gpu, &budget, &tally, &took, &log);
                s.spawn(move || {
                    let up = |v: &[f32]| {
                        gpu.upload(&Tensor::from_f32(v, &[v.len()], budget).unwrap())
                            .unwrap()
                    };
                    let name = format!("t{k}");
                    match k % 7 {
                        0 => tally(&format!("{name} sync"), gpu.sync()),
                        1 => {
                            gpu.ctx.sync().unwrap();
                            log.lock().unwrap().push(format!("{name} ctx.sync ok"));
                        }
                        2 => tally(
                            &format!("{name} download"),
                            gpu.download(&up(&[1.0; 8])).map(|_| ()),
                        ),
                        3 => {
                            let mut g = [up(&[0.5; 8])];
                            tally(
                                &format!("{name} clip"),
                                gpu.clip_grad_norm(&mut g, f32::MAX).map(|_| ()),
                            );
                        }
                        4 => {
                            up(&[1.0; 8]).to_host(budget).unwrap();
                            log.lock().unwrap().push(format!("{name} to_host ok"));
                        }
                        5 => took(&format!("{name} take"), gpu.ctx.take_faults()),
                        _ => {
                            let (mut p, g) = (up(&[1.0; 8]), up(&[0.5; 8]));
                            let (mut m, mut v) = (up(&[0.0; 8]), up(&[0.0; 8]));
                            tally(
                                &format!("{name} adamw"),
                                gpu.adamw_step(&mut p, &g, &mut m, &mut v, 1, cfg),
                            );
                        }
                    }
                });
            }
        });
        for i in 0..6 {
            tally(&format!("drain{i} sync"), gpu.sync());
            took(&format!("drain{i}"), gpu.ctx.take_faults());
        }
        let delivered = delivered.load(Ordering::SeqCst);
        let log = log.lock().unwrap();
        assert_eq!(other.load(Ordering::SeqCst), 0, "round {round}: {log:#?}");
        assert_eq!(
            delivered, 1,
            "round {round}: op {id} was delivered {delivered} times: {log:#?}"
        );
    }
    assert_eq!(gpu.ctx.torn_pairs(), 0);
    assert_pool_whole(&gpu, "after");
}

// ---------------------------------------------------------------- attack 5

/// Two faults around a failed read: the older op (7), orphaned, leads over the
/// newer op (3, a lower id) raised after it.
#[test]
fn wgfadv_an_orphans_first_op_beats_a_newer_live_one() {
    let (gpu, _b, buf) = open();
    raise_as(&gpu, 7, &buf, 2);
    gpu.ctx.fail_next_read();
    gpu.ctx.sync().expect_err("injected");
    raise_as(&gpu, 3, &buf, 2);
    assert_eq!(gpu.ctx.sync_observed().unwrap(), (bit(7) | bit(3), 7));
    assert_eq!(drain(&gpu), vec![]);
}

/// A chain: op 9 orphaned, then a read that folds it and fails too, then a
/// newer op 2: the pair that finally lands has first 9.
#[test]
fn wgfadv_a_fold_chain_of_failed_reads_keeps_the_oldest_first() {
    let (gpu, _b, buf) = open();
    raise_as(&gpu, 9, &buf, 2);
    gpu.ctx.fail_next_read();
    gpu.ctx.sync().expect_err("injected");
    raise_as(&gpu, 2, &buf, 2);
    gpu.ctx.fail_next_read();
    gpu.ctx.sync().expect_err("injected");
    raise_as(&gpu, 4, &buf, 2);
    assert_eq!(
        gpu.ctx.sync_observed().unwrap(),
        (bit(9) | bit(2) | bit(4), 9)
    );
    assert_eq!(drain(&gpu), vec![]);
    assert_pool_whole(&gpu, "after");
}

// ---------------------------------------------------------------- attack 6

/// A reader waiting for a slot holds neither `rec` nor anything a leased
/// reader needs: with every slot leased, raisers and uploads (which take
/// `rec`) all finish, and the waiting reader completes once a lease ends.
#[test]
fn wgfadv_a_reader_waiting_for_a_slot_does_not_block_the_recorder() {
    let (gpu, budget, buf) = open();
    let gpu = std::sync::Arc::new(gpu);
    let leases: Vec<_> = (0..HOLD_SLOTS)
        .map(|_| gpu.ctx.lease_hold_slot(Duration::ZERO).unwrap())
        .collect();
    let (tx, rx) = mpsc::channel();
    let reader = {
        let (gpu, tx) = (gpu.clone(), tx.clone());
        std::thread::spawn(move || {
            let _ = tx.send((
                "reader",
                gpu.ctx.sync_observed().map_err(|e| format!("{e:?}")),
            ));
        })
    };
    std::thread::sleep(Duration::from_millis(100));
    let busy = {
        let (gpu, buf, budget, tx) = (gpu.clone(), buf.clone(), budget.clone(), tx.clone());
        std::thread::spawn(move || {
            for id in 1..=6u32 {
                raise_as(&gpu, id, &buf, 2);
                let t = gpu
                    .upload(&Tensor::from_f32(&[1.0; 4], &[4], &budget).unwrap())
                    .unwrap();
                drop(t);
            }
            let _ = tx.send(("busy", Ok((0, 0))));
        })
    };
    let (who, _) = rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the recorder is blocked by a reader waiting for a slot");
    assert_eq!(who, "busy", "the waiting reader finished with no slot");
    drop(leases);
    let (who, r) = rx.recv_timeout(Duration::from_secs(30)).expect("reader");
    assert_eq!(who, "reader");
    let (bits, first) = r.unwrap();
    assert_eq!(bits, (1u64 << 6) - 1);
    assert!(consistent(bits, first));
    reader.join().unwrap();
    busy.join().unwrap();
    assert_eq!(drain(&gpu), vec![]);
    assert_pool_whole(&gpu, "after");
}
