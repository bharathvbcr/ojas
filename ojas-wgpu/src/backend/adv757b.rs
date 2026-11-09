//! Adversarial tests, second pass, for ft-757cb6b2 (lane 757wadv2). Not for merge.
//! Each test asserts the property the design claims; a failure is a BREAK.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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

/// [1, NaN, 1, 1, 1, 1, 1, 1] as shape [2, 4].
fn bad2d(gpu: &WgpuBackend, budget: &Budget) -> Tensor {
    let mut v = [1.0f32; 8];
    v[1] = f32::NAN;
    gpu.upload(&Tensor::from_f32(&v, &[2, 4], budget).unwrap())
        .unwrap()
}

fn is_nonfinite(r: &Result<(), OjasError>) -> bool {
    matches!(r, Err(OjasError::NonFinite { .. }))
}

struct Opt {
    p: Tensor,
    g: Tensor,
    m: Tensor,
    v: Tensor,
}

fn opt(gpu: &WgpuBackend, budget: &Budget) -> Opt {
    let up = |x: f32| {
        gpu.upload(&Tensor::from_f32(&[x; 8], &[8], budget).unwrap())
            .unwrap()
    };
    Opt {
        p: up(1.0),
        g: up(0.5),
        m: up(0.0),
        v: up(0.0),
    }
}

fn step(gpu: &WgpuBackend, o: &mut Opt) -> Result<(), OjasError> {
    gpu.adamw_step(
        &mut o.p,
        &o.g,
        &mut o.m,
        &mut o.v,
        1,
        AdamWConfig::nanolab(1e-2, 0.1),
    )
}

fn bits_of(gpu: &WgpuBackend, t: &Tensor) -> Vec<u32> {
    gpu.download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect()
}

// ---- attack 1: exactly once ----

#[test]
fn adv2_a1_view_download_reports_once_single_thread() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    let noncont = t.view(&[2, 2], &[4, 1], 0).unwrap();
    assert!(!noncont.is_contiguous().unwrap());
    raise_as(&a, CAST.fault_id(), &raw_of(&t), 2);
    let r = a.download(&noncont);
    assert!(
        matches!(r, Err(OjasError::NonFinite { op: "cast_bf16" })),
        "view download of faulted values: {:?}",
        r.map(|_| ())
    );
    assert!(
        a.sync().is_ok(),
        "delivered twice (view download, then sync)"
    );
    assert!(a.download(&noncont).is_ok());
}

#[test]
fn adv2_a1_to_host_then_one_sync_reports_once() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let b = WgpuBackend::with_context(a.ctx.clone(), budget.clone());
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    raise_as(&a, CAST.fault_id(), &raw_of(&t), 2);
    let _ = t.to_host(&budget).unwrap();
    assert!(is_nonfinite(&b.sync()), "deferred, first taker");
    assert!(a.sync().is_ok(), "second taker must not see it");
    assert!(b.sync().is_ok());
}

#[test]
fn adv2_a1_pub_ctx_sync_then_direct_download_once() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    let cont = t.view(&[2], &[1], 0).unwrap();
    raise_as(&a, CAST.fault_id(), &raw_of(&t), 2);
    a.ctx.sync().unwrap(); // defers
    let r = a.download(&cont);
    assert!(
        matches!(r, Err(OjasError::NonFinite { .. })),
        "{:?}",
        r.map(|_| ())
    );
    assert!(a.download(&cont).is_ok(), "twice");
    assert!(a.sync().is_ok(), "twice");
}

#[test]
fn adv2_a1_own_and_pending_merge_into_one_report() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    let cont = t.view(&[2], &[1], 0).unwrap();
    let x = CAST.fault_id();
    raise_as(&a, x, &raw_of(&t), 2);
    a.ctx.sync().unwrap(); // x deferred
    raise_as(&a, 9, &raw_of(&t), 2); // y observed by the download's own read
    let r = a.download(&cont);
    assert!(is_nonfinite(&r.map(|_| ())), "one merged report");
    assert!(a.sync().is_ok(), "merged: nothing left");
}

/// A download of a tensor of ANOTHER context: its read observes and defers on
/// that context's pair, but `faults()` takes THIS context's pair.
#[test]
fn adv2_a1_foreign_context_download_is_not_answered_ok() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let other_ctx = WgpuContext::open().expect("second context");
    let g = WgpuBackend::with_context(other_ctx, budget.clone());
    a.sync().unwrap();
    g.sync().unwrap();
    let t = bad2d(&g, &budget);
    raise_as(&g, CAST.fault_id(), &raw_of(&t), 2);
    let r = a.download(&t);
    eprintln!(
        "ADV2 foreign download: {:?}",
        r.as_ref().map(|_| "Ok").map_err(|e| format!("{e:?}"))
    );
    let later_on_owner = g.sync();
    eprintln!("ADV2 foreign owner's later sync: {later_on_owner:?}");
    assert!(
        r.is_err(),
        "foreign-context download copied NaN values and returned Ok; \
         the mask went to the owner's pair (owner's next sync: {later_on_owner:?})"
    );
}

/// Race: B takes only (no read of its own), as adamw_step's refusal does.
/// A's view download observes through `to_host` (deferred) and takes right
/// after. If B takes in between, A returns Ok with the faulted values.
#[test]
fn adv2_a1_view_download_vs_refuse_stealer() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let b = WgpuBackend::with_context(a.ctx.clone(), budget.clone());
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    let noncont = t.view(&[2, 2], &[4, 1], 0).unwrap();
    let buf = raw_of(&t);
    let done = AtomicBool::new(false);
    let stolen = AtomicU64::new(0);
    let a_ok = AtomicU64::new(0);
    let iters = 3000;
    std::thread::scope(|s| {
        let h = s.spawn(|| {
            while !done.load(Ordering::Acquire) {
                if b.refuse_if_pending().is_err() {
                    stolen.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        for _ in 0..iters {
            raise_as(&a, CAST.fault_id(), &buf, 2);
            if a.download(&noncont).is_ok() {
                a_ok.fetch_add(1, Ordering::Relaxed);
            }
            // clean up whatever remains for the next round
            let _ = a.sync();
        }
        done.store(true, Ordering::Release);
        h.join().unwrap();
    });
    eprintln!(
        "ADV2 A1 view-download iters={iters} a_ok_with_faulted_values={} stealer_took={}",
        a_ok.load(Ordering::Relaxed),
        stolen.load(Ordering::Relaxed)
    );
    assert_eq!(
        a_ok.load(Ordering::Relaxed),
        0,
        "a view download copied faulted values and returned Ok because another thread's \
         take (no read of its own) consumed the deferred mask"
    );
}

/// Same shape with the contiguous direct path: must hold (own mask).
#[test]
fn adv2_a1_direct_download_vs_refuse_stealer() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let b = WgpuBackend::with_context(a.ctx.clone(), budget.clone());
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    let cont = t.view(&[2], &[1], 0).unwrap();
    let buf = raw_of(&t);
    let done = AtomicBool::new(false);
    let a_ok = AtomicU64::new(0);
    std::thread::scope(|s| {
        let h = s.spawn(|| {
            while !done.load(Ordering::Acquire) {
                let _ = b.refuse_if_pending();
            }
        });
        for _ in 0..3000 {
            raise_as(&a, CAST.fault_id(), &buf, 2);
            if a.download(&cont).is_ok() {
                a_ok.fetch_add(1, Ordering::Relaxed);
            }
            let _ = a.sync();
        }
        done.store(true, Ordering::Release);
        h.join().unwrap();
    });
    assert_eq!(a_ok.load(Ordering::Relaxed), 0);
}

/// A later piece of a multi-piece download fails after an earlier one
/// observed: the Err goes to the caller, the mask is deferred, delivered once.
/// Deterministic: `fail_read_after(n)` fails the piece after `n` completed
/// ones, for every n that leaves the first (observing) piece done.
#[test]
fn adv2_a1_multi_piece_download_failing_late_defers_once() {
    const PIECES: u32 = 4;
    let budget = Budget::new(1 << 27);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    a.sync().unwrap();
    // `PIECES` pieces of `READBACK_CHUNK_BYTES`, the size `Tensor::to_host_via` reads in.
    let n = PIECES as usize * ojas_core::READBACK_CHUNK_BYTES / 4;
    let mut v = vec![1.0f32; n];
    v[1] = f32::NAN;
    let t = a
        .upload(&Tensor::from_f32(&v, &[n], &budget).unwrap())
        .unwrap();
    let buf = raw_of(&t);
    for after in 1..PIECES {
        raise_as(&a, CAST.fault_id(), &buf, 2);
        let reads0 = a.ctx.stats().reads;
        a.ctx.fail_read_after(after);
        let r = a.download(&t);
        match r {
            Err(OjasError::Backend { detail, .. }) if detail.contains("injected read failure") => {}
            other => panic!(
                "after {after}: the injection did not fail the download: {:?}",
                other.map(|_| ())
            ),
        }
        assert_eq!(
            a.ctx.stats().reads - reads0,
            u64::from(after),
            "after {after}: exactly {after} pieces completed before the failure"
        );
        // The first piece observed the fault; the Err carried no mask.
        let first = a.sync();
        assert!(
            is_nonfinite(&first),
            "after {after}: mask lost after a late failure: {first:?}"
        );
        assert!(a.sync().is_ok(), "after {after}: delivered twice");
    }
}

/// A failure BEFORE a read observes (injected after the wait) loses nothing:
/// sync, direct download and clip_grad_norm each.
#[test]
fn adv2_a1_failure_before_observation_loses_nothing() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    let cont = t.view(&[2], &[1], 0).unwrap();
    let buf = raw_of(&t);
    let mut grads = vec![a
        .upload(&Tensor::from_f32(&[1.0; 8], &[8], &budget).unwrap())
        .unwrap()];

    raise_as(&a, CAST.fault_id(), &buf, 2);
    a.ctx.fail_next_read();
    assert!(a.sync().is_err());
    assert!(is_nonfinite(&a.sync()), "sync lost it");
    assert!(a.sync().is_ok());

    raise_as(&a, CAST.fault_id(), &buf, 2);
    a.ctx.fail_next_read();
    assert!(a.download(&cont).is_err());
    assert!(
        is_nonfinite(&a.download(&cont).map(|_| ())),
        "download lost it"
    );
    assert!(a.download(&cont).is_ok());

    raise_as(&a, CAST.fault_id(), &buf, 2);
    a.ctx.fail_next_read();
    assert!(a.clip_grad_norm(&mut grads, 1e9).is_err());
    let r = a.clip_grad_norm(&mut grads, 1e9);
    assert!(
        matches!(r, Err(OjasError::NonFinite { .. })),
        "clip_grad_norm lost it: {r:?}"
    );
    assert!(a.clip_grad_norm(&mut grads, 1e9).is_ok());
}

/// clip_grad_norm's own status read observes the fault: it is told, once.
#[test]
fn adv2_a1_clip_grad_norm_reports_once_and_scales_nothing() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let b = WgpuBackend::with_context(a.ctx.clone(), budget.clone());
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    let mut grads = vec![a
        .upload(&Tensor::from_f32(&[3.0; 8], &[8], &budget).unwrap())
        .unwrap()];
    let before = bits_of(&a, &grads[0]);
    raise_as(&a, CAST.fault_id(), &raw_of(&t), 2);
    let r = b.clip_grad_norm(&mut grads, 0.001);
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
    assert_eq!(bits_of(&a, &grads[0]), before, "scaled despite the fault");
    assert!(a.sync().is_ok());
    assert!(b.clip_grad_norm(&mut grads, 1e9).is_ok());
}

/// A read that fails AFTER observing (sticky uncaptured error) defers the
/// mask once; it is not also returned to the caller as NonFinite.
#[test]
fn adv2_a1_read_failing_after_observation_defers_exactly_once() {
    let budget = Budget::new(1 << 22);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    a.sync().unwrap();
    let t = bad2d(&a, &budget);
    raise_as(&a, CAST.fault_id(), &raw_of(&t), 2);
    a.ctx.push_uncaptured("sticky");
    let r = a.sync();
    assert!(matches!(r, Err(OjasError::Backend { .. })), "{r:?}");
    let id = CAST.fault_id();
    assert_eq!(a.ctx.take_faults(), (1u64 << (id - 1), id), "deferred once");
    assert_eq!(a.ctx.take_faults(), (0, 0));
}

// ---- attack 4: adamw_step ----

#[test]
fn adv2_a4_refusal_consumes_exactly_the_entry_and_records_nothing() {
    let budget = Budget::new(1 << 22);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let mut o = opt(&gpu, &budget);
    let before = bits_of(&gpu, &o.p);
    let id = CAST.fault_id();

    gpu.ctx.defer_faults(1u64 << (id - 1), id);
    let (s0, live0, rb0) = (
        gpu.ctx.stats(),
        budget.live_bytes().unwrap(),
        budget.device_readbacks(),
    );
    let r = step(&gpu, &mut o);
    let (s1, live1, rb1) = (
        gpu.ctx.stats(),
        budget.live_bytes().unwrap(),
        budget.device_readbacks(),
    );
    assert!(
        matches!(r, Err(OjasError::NonFinite { op: "cast_bf16" })),
        "{r:?}"
    );
    assert_eq!(
        (s0.dispatches, s0.submits, s0.reads, live0, rb0),
        (s1.dispatches, s1.submits, s1.reads, live1, rb1),
        "a refusal recorded or charged something"
    );
    assert_eq!(
        gpu.ctx.take_faults(),
        (0, 0),
        "the refusal consumed the entry"
    );

    // The next step proceeds and equals one application on a fresh backend.
    step(&gpu, &mut o).unwrap();
    gpu.sync().unwrap();
    let applied = bits_of(&gpu, &o.p);
    assert_ne!(applied, before);
    let fresh = WgpuBackend::open(Budget::new(1 << 22)).expect("wgpu adapter");
    let fb = Budget::new(1 << 22);
    let mut f = opt(&fresh, &fb);
    step(&fresh, &mut f).unwrap();
    fresh.sync().unwrap();
    assert_eq!(applied, bits_of(&fresh, &f.p), "applied exactly once");
    for (a, b) in [(&o.m, &f.m), (&o.v, &f.v)] {
        assert_eq!(bits_of(&gpu, a), bits_of(&fresh, b));
    }
}

#[test]
fn adv2_a4_one_refusal_consumes_a_merged_entry_and_not_a_later_one() {
    let budget = Budget::new(1 << 22);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let mut o = opt(&gpu, &budget);
    gpu.ctx.defer_faults(1 << 2, 3);
    gpu.ctx.defer_faults(1 << 5, 6);
    assert!(is_nonfinite(&step(&gpu, &mut o)));
    assert!(step(&gpu, &mut o).is_ok(), "second step must proceed");
    gpu.ctx.defer_faults(1 << 7, 8);
    assert!(
        is_nonfinite(&step(&gpu, &mut o)),
        "a later entry refuses again"
    );
    assert!(step(&gpu, &mut o).is_ok());
}

/// Uncaptured error AND a deferred entry: the Backend error wins and the
/// deferred entry is not consumed (nothing else could report it).
#[test]
fn adv2_a4_sticky_error_leaves_the_deferred_entry() {
    let budget = Budget::new(1 << 22);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    gpu.sync().unwrap();
    let mut o = opt(&gpu, &budget);
    gpu.ctx.defer_faults(1 << 2, 3);
    gpu.ctx.push_uncaptured("sticky");
    assert!(matches!(step(&gpu, &mut o), Err(OjasError::Backend { .. })));
    assert_eq!(gpu.ctx.take_faults(), (1 << 2, 3));
}

// ---- attack 5: torn / garbage words through fault_error ----

#[test]
fn adv2_a5_fault_error_survives_garbage_pairs() {
    let budget = Budget::new(1 << 22);
    let gpu = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let pairs: &[(u64, u32)] = &[
        (0, 1),
        (0, 38),
        (0, 39),
        (0, 64),
        (0, 65),
        (0, u32::MAX),
        (1, 0),
        (u64::MAX, 0),
        (1 << 63, 0),
        (u64::MAX, 1),
        (u64::MAX, 65),
        (u64::MAX, u32::MAX),
        (1 << 2, 9),
    ];
    for &(bits, first) in pairs {
        let r = gpu.fault_error((bits, first));
        match &r {
            Err(OjasError::NonFinite { op }) => assert!(OP_NAMES.contains(op), "{op}"),
            Err(OjasError::Backend { detail, .. }) => {
                assert!(detail.contains("past the op table"), "{detail}")
            }
            other => panic!("({bits:#x},{first}) not failed closed: {other:?}"),
        }
        assert!(
            gpu.sync().is_ok(),
            "state left behind by ({bits:#x},{first})"
        );
    }
    // pending first wins over own first, including a garbage pending first.
    gpu.ctx.defer_faults(0, 65);
    let r = gpu.fault_error((1 << 3, 4));
    eprintln!("ADV2 A5 pending (0,65) + own (bit3,4): {r:?}");
    assert!(r.is_err());
    assert!(gpu.sync().is_ok());
}

// ---- attack 7: readback accounting ----

#[test]
fn adv2_a7_download_readbacks_match_to_host_in_every_shape() {
    let budget = Budget::new(1 << 24);
    let a = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let n = 256 * 1024;
    let big = a
        .upload(&Tensor::from_f32(&vec![2.0; n], &[n], &budget).unwrap())
        .unwrap();
    let small = a
        .upload(&Tensor::from_f32(&[1.0; 8], &[2, 4], &budget).unwrap())
        .unwrap();
    let cases: Vec<(&str, Tensor)> = vec![
        ("big", big.clone()),
        ("cont-view", small.view(&[4], &[1], 16).unwrap()),
        ("noncont-view", small.view(&[2, 2], &[4, 1], 0).unwrap()),
    ];
    for (name, t) in cases {
        let r0 = budget.device_readbacks();
        let _ = a.download(&t).unwrap();
        let r1 = budget.device_readbacks();
        let _ = t.to_host(&budget).unwrap();
        let r2 = budget.device_readbacks();
        assert_eq!(
            (r1.0 - r0.0, r1.1 - r0.1),
            (r2.0 - r1.0, r2.1 - r1.1),
            "{name}: download and to_host disagree on readbacks"
        );
        assert_eq!(r1.0 - r0.0, 1, "{name}: one readback per download");
    }
    // A failing direct download counts nothing, as a failing to_host.
    let r0 = budget.device_readbacks();
    a.ctx.fail_next_read();
    assert!(a.download(&big.view(&[4], &[1], 0).unwrap()).is_err());
    assert_eq!(budget.device_readbacks(), r0, "a failed read counted");
    // A download that reports a fault still counted its readback, like to_host.
    let t = bad2d(&a, &budget);
    raise_as(&a, CAST.fault_id(), &raw_of(&t), 2);
    let r0 = budget.device_readbacks();
    assert!(a.download(&t.view(&[2], &[1], 0).unwrap()).is_err());
    assert_eq!(budget.device_readbacks().0 - r0.0, 1);
    let _ = a.sync();
}

// ---- attack 3: concurrency stress with injected faults ----

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn stress(threads: usize, episodes: usize) {
    let budget = Budget::new(1 << 26);
    let base = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    base.sync().unwrap();
    let t2d = bad2d(&base, &budget);
    let cont = t2d.view(&[2], &[1], 0).unwrap();
    let noncont = t2d.view(&[2, 2], &[4, 1], 0).unwrap();
    let buf = raw_of(&t2d);
    let barrier = std::sync::Barrier::new(threads);
    let deliveries = AtomicU64::new(0);
    let lost = AtomicU64::new(0);
    let dup = AtomicU64::new(0);
    let stale = AtomicU64::new(0);
    let wrong_name = AtomicU64::new(0);
    let unexpected = AtomicU64::new(0);
    let own_ok = AtomicU64::new(0);
    let ep_deliveries: Vec<AtomicU64> = (0..episodes).map(|_| AtomicU64::new(0)).collect();
    let ep_drain: Vec<AtomicU64> = (0..episodes).map(|_| AtomicU64::new(0)).collect();
    std::thread::scope(|s| {
        for t in 0..threads {
            let (barrier, buf, base, budget) = (&barrier, &buf, &base, &budget);
            let (cont, noncont, t2d) = (&cont, &noncont, &t2d);
            let (deliveries, lost, dup, stale, wrong_name, unexpected, own_ok) = (
                &deliveries,
                &lost,
                &dup,
                &stale,
                &wrong_name,
                &unexpected,
                &own_ok,
            );
            let (ep_deliveries, ep_drain) = (&ep_deliveries, &ep_drain);
            s.spawn(move || {
                let gpu = WgpuBackend::with_context(base.ctx.clone(), budget.clone());
                let mut o = opt(&gpu, budget);
                let mut grads = vec![gpu
                    .upload(&Tensor::from_f32(&[1.0; 8], &[8], budget).unwrap())
                    .unwrap()];
                let mut seed = 0x9e3779b97f4a7c15u64 ^ (t as u64 * 7919);
                for ep in 0..episodes {
                    let id = 1 + (ep as u32 % 37);
                    let want = Op(id - 1).name();
                    barrier.wait();
                    if t == 0 {
                        raise_as(&gpu, id, buf, 2);
                    }
                    barrier.wait();
                    let kind = lcg(&mut seed) % 8;
                    let r: Result<(), OjasError> = match kind {
                        0 => gpu.download(cont).map(|_| ()),
                        1 => gpu.download(noncont).map(|_| ()),
                        2 => t2d.to_host(budget).map(|_| ()),
                        3 => gpu.sync(),
                        4 => gpu.clip_grad_norm(&mut grads, 1e9).map(|_| ()),
                        5 => step(&gpu, &mut o),
                        6 => gpu.ctx.sync(),
                        _ => Ok(()),
                    };
                    match &r {
                        Err(OjasError::NonFinite { op }) => {
                            ep_deliveries[ep].fetch_add(1, Ordering::Relaxed);
                            deliveries.fetch_add(1, Ordering::Relaxed);
                            if *op != want {
                                wrong_name.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(_) => {
                            unexpected.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(()) => {
                            if kind <= 1 || kind == 3 || kind == 4 {
                                own_ok.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    barrier.wait();
                    if t == 0 {
                        let d1 = gpu.sync();
                        if is_nonfinite(&d1) {
                            ep_deliveries[ep].fetch_add(1, Ordering::Relaxed);
                            deliveries.fetch_add(1, Ordering::Relaxed);
                        }
                        // everything is quiescent: nothing may remain.
                        if gpu.sync().is_err() {
                            stale.fetch_add(1, Ordering::Relaxed);
                            ep_drain[ep].store(1, Ordering::Relaxed);
                            let _ = gpu.sync();
                        }
                        match ep_deliveries[ep].load(Ordering::Relaxed) {
                            0 => {
                                lost.fetch_add(1, Ordering::Relaxed);
                            }
                            1 => {}
                            _ => {
                                dup.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            });
        }
    });
    eprintln!(
        "ADV2 A3 threads={threads} episodes={episodes} deliveries={} lost_episodes={} duplicate_episodes={} stale_after_drain={} wrong_name={} unexpected_err={} own_read_ok_in_episode(sequential-case suspects)={}",
        deliveries.load(Ordering::Relaxed),
        lost.load(Ordering::Relaxed),
        dup.load(Ordering::Relaxed),
        stale.load(Ordering::Relaxed),
        wrong_name.load(Ordering::Relaxed),
        unexpected.load(Ordering::Relaxed),
        own_ok.load(Ordering::Relaxed),
    );
    assert_eq!(
        unexpected.load(Ordering::Relaxed),
        0,
        "unexpected error kinds"
    );
    assert_eq!(
        wrong_name.load(Ordering::Relaxed),
        0,
        "a fault named the wrong op"
    );
    assert_eq!(
        lost.load(Ordering::Relaxed),
        0,
        "an injected fault was never delivered"
    );
    assert_eq!(
        stale.load(Ordering::Relaxed),
        0,
        "a delivery remained after quiescence"
    );
    // Ruled: overlapping readers may each be told (fail closed), so
    // duplicate deliveries are counted above and not asserted against. What
    // must hold is no lost observation, no wrong op name, nothing stale.
}

#[test]
fn adv2_a3_stress_4x300() {
    stress(4, 300);
}

#[test]
fn adv2_a3_stress_2x400() {
    stress(2, 400);
}

#[test]
fn adv2_a3_stress_6x200() {
    stress(6, 200);
}
