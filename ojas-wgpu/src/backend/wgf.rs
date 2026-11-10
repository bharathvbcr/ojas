//! ft-84e2b74e: concurrent raisers and syncers over one context see a
//! consistent `(bits, first)` pair at the device boundary, and every raised
//! fault is reported exactly once.
//!
//! The pair is counted before the host repairs it
//! ([`WgpuContext::torn_pairs`]), so a device that still tears it cannot hide
//! behind `normalize_fault_words`.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn raw_of(t: &Tensor) -> wgpu::Buffer {
    t.device_buffer()
        .and_then(|d| d.as_any().downcast_ref::<WgpuBuffer>())
        .expect("a wgpu tensor")
        .raw()
        .expect("a live buffer")
        .clone()
}

/// Record `check_finite` over `n` floats of `buf` under fault id `id`.
fn raise_as(gpu: &WgpuBackend, id: u32, buf: &wgpu::Buffer, n: usize) {
    let mut job = gpu.ctx.job(&gpu.budget, id);
    job.dispatch(&CHECK_FINITE, &[n as u32], &[buf], gpu.lanes(n).unwrap())
        .unwrap();
    job.commit().unwrap();
}

/// `rounds` rounds of `raisers` threads that each raise their own op's bit
/// once and `syncers` threads that sync until every raiser has recorded.
/// Every report any thread received, by either channel, is collected; after
/// the round a final sync drains the rest. Each raised bit must then have
/// been reported exactly once in the round: a repeat is a fault reported
/// when nothing raised it since the previous report.
fn stress(raisers: usize, syncers: usize, rounds: usize) {
    let budget = Budget::new(1 << 20);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let bad = gpu
        .upload(&Tensor::from_f32(&[1.0, f32::NAN], &[2], &budget).unwrap())
        .unwrap();
    let buf = raw_of(&bad);
    // Ids alternate between the two mask words (1, 34, 3, 36, 5, 38),
    // distinct, within the 64 ops.
    let ids: Vec<u32> = (0..raisers as u32).map(|i| 1 + (i * 33) % 64).collect();
    let torn_before = gpu.ctx.torn_pairs();
    let mut total_reports = 0usize;
    let (mut inconsistent, mut miscounted) = (0u32, 0u32);
    let mut note: Option<String> = None;
    for round in 0..rounds {
        let recorded = AtomicUsize::new(0);
        let mut reports: Vec<(u64, u32)> = Vec::new();
        std::thread::scope(|s| {
            let watchers: Vec<_> = (0..syncers)
                .map(|_| {
                    s.spawn(|| {
                        let mut seen = Vec::new();
                        loop {
                            let last = recorded.load(Ordering::Acquire) == raisers;
                            let own = gpu.ctx.sync_observed().unwrap();
                            let taken = gpu.ctx.take_faults();
                            seen.extend([own, taken].into_iter().filter(|&p| p != (0, 0)));
                            if last {
                                break;
                            }
                        }
                        seen
                    })
                })
                .collect();
            for &id in &ids {
                let (gpu, buf, recorded) = (&gpu, &buf, &recorded);
                s.spawn(move || {
                    raise_as(gpu, id, buf, 2);
                    recorded.fetch_add(1, Ordering::Release);
                });
            }
            for w in watchers {
                reports.extend(w.join().unwrap());
            }
        });
        reports.extend(
            [gpu.ctx.sync_observed().unwrap(), gpu.ctx.take_faults()]
                .into_iter()
                .filter(|&p| p != (0, 0)),
        );
        total_reports += reports.len();
        let mut count = [0u32; 64];
        for &(bits, first) in &reports {
            let consistent = (1..=64).contains(&first) && bits & (1u64 << (first - 1)) != 0;
            if !consistent {
                inconsistent += 1;
                note.get_or_insert_with(|| {
                    format!("round {round}: report ({bits:#x}, {first}) is not a consistent pair")
                });
            }
            for (i, c) in count.iter_mut().enumerate() {
                *c += u32::from(bits >> i & 1 == 1);
            }
        }
        for (i, &c) in count.iter().enumerate() {
            let want = u32::from(ids.contains(&(i as u32 + 1)));
            if c != want {
                miscounted += 1;
                note.get_or_insert_with(|| {
                    format!(
                        "round {round}: op bit {i} reported {c} times, raised {want}: {reports:x?}"
                    )
                });
            }
        }
    }
    let torn = gpu.ctx.torn_pairs() - torn_before;
    eprintln!(
        "wgf {raisers}x{syncers}: {rounds} rounds, {total_reports} reports, \
         {torn} raw torn pairs, {inconsistent} inconsistent reports, \
         {miscounted} bits not reported exactly once"
    );
    assert_eq!(
        (torn, inconsistent, miscounted),
        (0, 0, 0),
        "{}",
        note.unwrap_or_default()
    );
    assert!(total_reports > 0, "no fault was ever reported");
    gpu.sync().expect("nothing is left to report");
}

#[test]
fn two_raisers_two_syncers_report_each_fault_once_with_a_consistent_pair() {
    stress(2, 2, 60);
}

#[test]
fn six_raisers_three_syncers_report_each_fault_once_with_a_consistent_pair() {
    stress(6, 3, 60);
}

/// The fault-free probe of the cost of a read's fault hand-off: AdamW steps
/// over eight 64k-float tensors, each followed by one sync (one read, so one
/// `fault_hold` and its copy), no fault ever raised. Run as
/// `cargo test --release -p ojas-wgpu --lib wgf_probe -- --ignored --nocapture`.
/// Prints the best (minimum) per-step time of 15 samples of 40 steps.
#[test]
#[ignore = "benchmark"]
fn wgf_probe_fault_free_adamw_step_and_sync() {
    let budget = Budget::new(1 << 30);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let n = 1 << 16;
    let up = |v: f32| {
        gpu.upload(&Tensor::from_f32(&vec![v; n], &[n], &budget).unwrap())
            .unwrap()
    };
    let mut state: Vec<_> = (0..8)
        .map(|_| (up(1.0), up(0.5), up(0.0), up(0.0)))
        .collect();
    let cfg = AdamWConfig::nanolab(1e-3, 0.1);
    let mut step = 0u64;
    let mut run = |iters: u32| {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            step += 1;
            for (p, g, m, v) in state.iter_mut() {
                gpu.adamw_step(p, g, m, v, step, cfg).unwrap();
            }
            gpu.sync().unwrap();
        }
        t.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
    };
    run(20);
    let samples: Vec<f64> = (0..15).map(|_| run(40)).collect();
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    println!("WGF-PROBE min_us={min:.2} samples_us={samples:.1?}");
}

use crate::context::HOLD_SLOTS;
use std::time::Duration;

fn bad_buffer(gpu: &WgpuBackend, budget: &Budget) -> wgpu::Buffer {
    raw_of(
        &gpu.upload(&Tensor::from_f32(&[1.0, f32::NAN], &[2], budget).unwrap())
            .unwrap(),
    )
}

/// With every hold slot leased, a read waits for a slot: it is never handed a
/// shared one, never overwrites an unseen pair, and fails closed (a backend
/// error naming the slots) when none comes free within the wait. Once one is
/// released the read completes and reports the fault raised meanwhile once.
#[test]
fn a_read_with_every_hold_slot_leased_waits_and_never_shares_or_drops() {
    let budget = Budget::new(1 << 20);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let buf = bad_buffer(&gpu, &budget);
    let mut leases: Vec<_> = (0..HOLD_SLOTS)
        .map(|_| gpu.ctx.lease_hold_slot(Duration::from_secs(5)).unwrap())
        .collect();
    let mut distinct: Vec<u32> = leases.iter().map(|l| l.slot).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), HOLD_SLOTS as usize, "a slot was shared");
    let err = gpu
        .ctx
        .lease_hold_slot(Duration::from_millis(50))
        .err()
        .expect("the 65th lease fails closed");
    assert!(format!("{err:?}").contains("no fault hold slot"), "{err:?}");

    let id = 5;
    raise_as(&gpu, id, &buf, 2);
    let seen = std::thread::scope(|s| {
        let reader = s.spawn(|| gpu.ctx.sync_observed());
        std::thread::sleep(Duration::from_millis(150));
        assert!(!reader.is_finished(), "the read ran without a slot");
        leases.pop();
        reader.join().unwrap().unwrap()
    });
    assert_eq!(seen, (1u64 << (id - 1), id));
    drop(leases);
    assert_eq!(gpu.ctx.sync_observed().unwrap(), (0, 0), "reported once");
    assert_eq!(gpu.ctx.take_faults(), (0, 0));
}

/// A read that fails before the host sees its copy leaves the fault in its
/// slot; the next read, on another thread, reports it exactly once.
#[test]
fn an_orphaned_fault_is_reported_once_by_the_next_read_on_another_thread() {
    let budget = Budget::new(1 << 20);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let buf = bad_buffer(&gpu, &budget);
    let id = 9;
    raise_as(&gpu, id, &buf, 2);
    gpu.ctx.fail_next_read();
    gpu.ctx.sync().expect_err("the injected read failure");
    assert_eq!(gpu.ctx.take_faults(), (0, 0), "the failed read saw nothing");
    let seen = std::thread::scope(|s| s.spawn(|| gpu.ctx.sync_observed()).join().unwrap()).unwrap();
    assert_eq!(seen, (1u64 << (id - 1), id));
    assert_eq!(gpu.ctx.sync_observed().unwrap(), (0, 0), "reported once");
    assert_eq!(gpu.ctx.take_faults(), (0, 0));
}

/// More orphans than one hold folds (13), and then every slot orphaned, so a
/// read finds none free and takes the oldest orphan as its own slot: every fault is reported once, none lost, the oldest
/// first op leads each report.
#[test]
fn many_orphans_are_folded_oldest_first_each_reported_once() {
    for orphans in [20u32, HOLD_SLOTS] {
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
        gpu.sync().unwrap();
        // Orphan slots a hold never folded (a failed read folds the earlier
        // ones into its own slot, so only a seam leaves this many).
        for id in 1..=orphans {
            gpu.ctx.adv_set_held(1u64 << (id - 1), id);
        }
        let mut union = 0u64;
        let mut firsts = Vec::new();
        loop {
            let (bits, first) = gpu.ctx.sync_observed().unwrap();
            if (bits, first) == (0, 0) {
                break;
            }
            assert_ne!(bits & (1u64 << (first - 1)), 0, "torn ({bits:#x}, {first})");
            assert_eq!(union & bits, 0, "{orphans}: bits {bits:#x} reported twice");
            union |= bits;
            firsts.push(first);
        }
        let want = if orphans == 64 {
            u64::MAX
        } else {
            (1u64 << orphans) - 1
        };
        assert_eq!(union, want, "{orphans} orphans: a fault was lost");
        assert_eq!(firsts[0], 1, "the oldest orphan leads");
        assert!(firsts.windows(2).all(|w| w[0] < w[1]), "{firsts:?}");
        assert_eq!(gpu.ctx.torn_pairs(), 0);
    }
}

/// Every hold slot is leasable at once: no wait, all distinct.
fn assert_every_hold_slot_is_free(gpu: &WgpuBackend, what: &str) {
    let leases: Vec<_> = (0..HOLD_SLOTS)
        .map(|i| {
            gpu.ctx
                .lease_hold_slot(Duration::ZERO)
                .unwrap_or_else(|e| panic!("{what}: lease {i} of {HOLD_SLOTS}: {e:?}"))
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

/// The slots a hold folds go back to the pool: after the orphans are all
/// reported, every slot can be leased again. 5 fits one fold, 20 is past
/// the 13 one hold folds, and 64 leaves no free slot, so the first read
/// folds its own carried slot too.
#[test]
fn folded_orphan_slots_return_to_the_pool() {
    for orphans in [5u32, 20, HOLD_SLOTS] {
        let budget = Budget::new(1 << 20);
        let gpu = WgpuBackend::open(budget).expect("wgpu adapter");
        gpu.sync().unwrap();
        for id in 1..=orphans {
            gpu.ctx.adv_set_held(1u64 << (id - 1), id);
        }
        let mut union = 0u64;
        loop {
            let (bits, first) = gpu.ctx.sync_observed().unwrap();
            if (bits, first) == (0, 0) {
                break;
            }
            assert_eq!(union & bits, 0, "{orphans}: bits {bits:#x} reported twice");
            union |= bits;
        }
        let want = if orphans == 64 {
            u64::MAX
        } else {
            (1u64 << orphans) - 1
        };
        assert_eq!(union, want, "{orphans} orphans: a fault was lost");
        assert_every_hold_slot_is_free(&gpu, &format!("{orphans} orphans"));
    }
}

/// Reads that fail before the host sees their copy fold the previous
/// orphan each: ten of them, then one good read, leave the pool whole and
/// the faults reported once.
#[test]
fn failed_reads_do_not_leak_hold_slots() {
    let budget = Budget::new(1 << 20);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let buf = bad_buffer(&gpu, &budget);
    let mut want = 0u64;
    for id in 1..=10u32 {
        raise_as(&gpu, id, &buf, 2);
        want |= 1u64 << (id - 1);
        gpu.ctx.fail_next_read();
        gpu.ctx.sync().expect_err("the injected read failure");
    }
    let (bits, first) = gpu.ctx.sync_observed().unwrap();
    assert_eq!((bits, first), (want, 1));
    assert_eq!(gpu.ctx.sync_observed().unwrap(), (0, 0), "reported once");
    assert_every_hold_slot_is_free(&gpu, "ten failed reads");
}
