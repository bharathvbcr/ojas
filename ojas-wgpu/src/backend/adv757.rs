//! Adversarial tests for ft-757cb6b2 (lane 757wadv). Not for merge.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

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

fn nan_tensor(gpu: &WgpuBackend, budget: &Budget) -> Tensor {
    gpu.upload(&Tensor::from_f32(&[1.0, f32::NAN], &[2], budget).unwrap())
        .unwrap()
}

/// A1: `Tensor::to_host` on a device tensor is a public read that never takes.
/// Its thread keeps an own copy; another thread reports the fault; later this
/// thread's unrelated `sync` reports the already-reported fault again.
#[test]
fn adv_a1_direct_read_leaves_a_stale_copy_for_a_later_sync() {
    let budget = Budget::new(1 << 20);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let b = WgpuBackend::with_context(a.ctx.clone(), budget.clone());
    a.sync().unwrap();
    let bad = nan_tensor(&a, &budget);
    raise_as(&a, CAST.fault_id(), &raw_of(&bad), 2);
    let host = bad.to_host(&budget).expect("a direct read");
    let _ = host;
    let other = std::thread::scope(|s| s.spawn(|| b.sync()).join().unwrap());
    eprintln!("ADV A1 other thread: {other:?}");
    assert!(matches!(other, Err(OjasError::NonFinite { .. })));
    // Nothing raised since the report above.
    let mine = a.sync();
    eprintln!("ADV A1 this thread's later sync: {mine:?}");
    assert!(mine.is_ok(), "BREAK A1: false Err, stale own copy: {mine:?}");
}

/// A1/A6: same stale copy makes adamw_step refuse after the fault was handled.
#[test]
fn adv_a1_stale_copy_refuses_an_adamw_step() {
    let budget = Budget::new(1 << 20);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let b = WgpuBackend::with_context(a.ctx.clone(), budget.clone());
    a.sync().unwrap();
    let up = |v: &[f32]| {
        a.upload(&Tensor::from_f32(v, &[v.len()], &budget).unwrap())
            .unwrap()
    };
    let (mut p, g, mut m, mut v) = (up(&[1.0; 8]), up(&[0.5; 8]), up(&[0.0; 8]), up(&[0.0; 8]));
    let bad = nan_tensor(&a, &budget);
    raise_as(&a, CAST.fault_id(), &raw_of(&bad), 2);
    let _ = bad.to_host(&budget).expect("a direct read");
    let other = std::thread::scope(|s| s.spawn(|| b.sync()).join().unwrap());
    assert!(matches!(other, Err(OjasError::NonFinite { .. })));
    let r = a.adamw_step(&mut p, &g, &mut m, &mut v, 1, AdamWConfig::nanolab(1e-2, 0.1));
    eprintln!("ADV A1b adamw_step after the fault was already reported elsewhere: {r:?}");
    assert!(r.is_ok(), "BREAK A1b: step refused by a stale fault: {r:?}");
}

/// A2/criterion 13: the faulting op sits in the shared encoder; another
/// clone's sync submits it, observes and takes the fault; this thread then
/// downloads the faulted values.
#[test]
fn adv_c13_a_read_after_another_threads_sync_took_the_fault() {
    let budget = Budget::new(1 << 20);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let b = WgpuBackend::with_context(a.ctx.clone(), budget.clone());
    a.sync().unwrap();
    let bad = nan_tensor(&a, &budget);
    raise_as(&a, CAST.fault_id(), &raw_of(&bad), 2);
    let other = std::thread::scope(|s| s.spawn(|| b.sync()).join().unwrap());
    assert!(matches!(other, Err(OjasError::NonFinite { .. })), "{other:?}");
    let mine = a.download(&bad);
    eprintln!("ADV C13 download of the faulted values after the other thread took: {mine:?}");
    assert!(
        mine.is_err(),
        "BREAK C13: download returned Ok with non-finite values: {:?}",
        mine.map(|t| t.to_f32_vec())
    );
}

/// A6: a refused adamw_step records nothing and charges nothing.
#[test]
fn adv_a6_refusal_records_nothing() {
    let budget = Budget::new(1 << 20);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let up = |v: &[f32]| {
        gpu.upload(&Tensor::from_f32(v, &[v.len()], &budget).unwrap())
            .unwrap()
    };
    let (mut p, g, mut m, mut v) = (up(&[1.0; 8]), up(&[0.5; 8]), up(&[0.0; 8]), up(&[0.0; 8]));
    gpu.ctx.push_uncaptured("pending");
    let (s0, live0) = (gpu.ctx.stats(), budget.live_bytes().unwrap());
    for _ in 0..3 {
        assert!(gpu
            .adamw_step(&mut p, &g, &mut m, &mut v, 1, AdamWConfig::nanolab(1e-2, 0.1))
            .is_err());
    }
    let (s1, live1) = (gpu.ctx.stats(), budget.live_bytes().unwrap());
    assert_eq!(s0.dispatches, s1.dispatches, "dispatches");
    assert_eq!(s0.submits, s1.submits, "submits");
    assert_eq!(live0, live1, "budget");
    assert_eq!(s0.reads, s1.reads, "no read");
    // Under a sticky error nothing can be read back (download refuses): record it.
    let d = gpu.download(&p);
    eprintln!("ADV A6 download under a sticky error: {:?}", d.as_ref().map(|_| "ok").map_err(|e| format!("{e:?}").len()));
}

/// A7: N threads x M clones; real GPU; fault injection.
fn stress(threads: usize, clones: usize, iters: usize) {
    const IDS: [u32; 8] = [1, 9, 17, 25, 33, 41, 49, 64];
    assert!(threads <= IDS.len());
    let budget = Budget::new(1 << 24);
    let base = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    base.sync().unwrap();
    let bad = nan_tensor(&base, &budget);
    let buf = raw_of(&bad);
    let raised: Vec<AtomicU64> = (0..64).map(|_| AtomicU64::new(0)).collect();
    let ops = AtomicU64::new(0);
    let own_missing = AtomicU64::new(0);
    let torn = AtomicU64::new(0);
    let first_not_in_bits = AtomicU64::new(0);
    let unraised = AtomicU64::new(0);
    let stale = AtomicU64::new(0);
    let reported: Vec<AtomicU64> = (0..64).map(|_| AtomicU64::new(0)).collect();
    std::thread::scope(|s| {
        for t in 0..threads {
            let (budget, buf, base) = (&budget, &buf, &base);
            let (raised, ops, own_missing, torn, fnb, unraised, stale, reported) = (
                &raised,
                &ops,
                &own_missing,
                &torn,
                &first_not_in_bits,
                &unraised,
                &stale,
                &reported,
            );
            s.spawn(move || {
                let id = IDS[t];
                let bk: Vec<WgpuBackend> = (0..clones)
                    .map(|_| WgpuBackend::with_context(base.ctx.clone(), budget.clone()))
                    .collect();
                // raised-count snapshot at the end of this thread's previous report
                let mut prev: Vec<u64> = vec![0; 64];
                for it in 0..iters {
                    let g = &bk[it % clones];
                    raised[(id - 1) as usize].fetch_add(1, Ordering::SeqCst);
                    raise_as(g, id, buf, 2);
                    // a read that copies the faulted values
                    let _vals = g.ctx.read(buf, 0, 8).unwrap();
                    let (bits, first) = g.ctx.take_faults();
                    ops.fetch_add(1, Ordering::Relaxed);
                    if (bits == 0) != (first == 0) {
                        torn.fetch_add(1, Ordering::Relaxed);
                    }
                    if first != 0 && bits & (1u64 << (first - 1)) == 0 {
                        fnb.fetch_add(1, Ordering::Relaxed);
                    }
                    if bits & (1u64 << (id - 1)) == 0 {
                        own_missing.fetch_add(1, Ordering::Relaxed);
                    }
                    for j in 0..64 {
                        if bits & (1u64 << j) != 0 {
                            reported[j].fetch_add(1, Ordering::Relaxed);
                            let now = raised[j].load(Ordering::SeqCst);
                            if now == 0 {
                                unraised.fetch_add(1, Ordering::Relaxed);
                            } else if now <= prev[j] {
                                stale.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    for j in 0..64 {
                        prev[j] = raised[j].load(Ordering::SeqCst);
                    }
                    // a clean sync interleaved: must not invent faults
                    if it % 7 == 0 {
                        g.ctx.sync().unwrap();
                    }
                }
            });
        }
    });
    base.ctx.sync().unwrap();
    let (rb, rf) = base.ctx.take_faults();
    let total_raised: u64 = raised.iter().map(|a| a.load(Ordering::SeqCst)).sum();
    let ops = ops.load(Ordering::Relaxed);
    eprintln!(
        "ADV A7 threads={threads} clones={clones} ops={ops} raised={total_raised} own_missing={} torn={} first_not_in_bits={} unraised={} stale_by_prev_snapshot={} leftover=({rb:#x},{rf})",
        own_missing.load(Ordering::Relaxed),
        torn.load(Ordering::Relaxed),
        first_not_in_bits.load(Ordering::Relaxed),
        unraised.load(Ordering::Relaxed),
        stale.load(Ordering::Relaxed),
    );
    assert_eq!(torn.load(Ordering::Relaxed), 0, "torn pairs");
    assert_eq!(first_not_in_bits.load(Ordering::Relaxed), 0, "first not in bits");
    assert_eq!(unraised.load(Ordering::Relaxed), 0, "a fault nobody raised");
    for t in 0..threads {
        assert!(
            reported[(IDS[t] - 1) as usize].load(Ordering::Relaxed) > 0,
            "id {} never reported to anyone",
            IDS[t]
        );
    }
    assert_eq!(
        own_missing.load(Ordering::Relaxed),
        0,
        "BREAK A7: {} of {ops} reads of faulted values returned without the reader's own raised fault",
        own_missing.load(Ordering::Relaxed)
    );
}

#[test]
fn adv_a7_stress_6x3() {
    stress(6, 3, 3500);
}

#[test]
fn adv_a7_stress_2x2() {
    stress(2, 2, 5000);
}
