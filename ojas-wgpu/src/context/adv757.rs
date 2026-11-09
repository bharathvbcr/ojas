//! Adversarial tests for ft-757cb6b2 (lane 757wadv). Not for merge.
use super::*;

/// A3/A2: OWN_FAULTS stays bounded, and what eviction costs.
#[test]
fn adv_own_faults_is_bounded_over_many_short_lived_contexts() {
    for _ in 0..10_000 {
        let f = Faults::default();
        f.record(1, 1); // never taken, then dropped
    }
    let len = OWN_FAULTS.with(|o| o.borrow().len());
    eprintln!("ADV A3 OWN_FAULTS len after 10000 dropped contexts = {len} (cap {OWN_FAULTS_CAP})");
    assert!(len <= OWN_FAULTS_CAP);
}

#[test]
fn adv_eviction_loses_the_own_guarantee_past_the_cap() {
    OWN_FAULTS.with(|o| o.borrow_mut().clear());
    let fs: Vec<Arc<Faults>> = (0..OWN_FAULTS_CAP + 1)
        .map(|_| Arc::new(Faults::default()))
        .collect();
    for f in &fs {
        f.record(1 << 3, 4);
    }
    // fs[0]'s own entry was evicted. Another thread (another clone) takes the shared pair.
    let f0 = Arc::clone(&fs[0]);
    let other = std::thread::spawn(move || f0.take()).join().unwrap();
    let mine = fs[0].take();
    eprintln!("ADV A2 other={other:?} own-after-eviction={mine:?}");
    // The fault is reported once overall (to `other`), never lost, never twice...
    assert_eq!(other, (1 << 3, 4));
    // ...but the reader's own guarantee is gone: it sees nothing.
    assert_eq!(mine, (1 << 3, 4), "BREAK A2: evicted own copy, reader sees {mine:?}");
}

#[test]
fn adv_no_double_report_across_eviction_single_thread() {
    OWN_FAULTS.with(|o| o.borrow_mut().clear());
    let fs: Vec<Faults> = (0..OWN_FAULTS_CAP + 5).map(|_| Faults::default()).collect();
    for f in &fs {
        f.record(1, 1);
    }
    for f in &fs {
        let a = f.take();
        let b = f.take();
        assert_eq!(a, (1, 1));
        assert_eq!(b, (0, 0), "double report");
    }
}

/// A4: message text is unbounded (only the count is capped).
#[test]
fn adv_uncaptured_message_length_is_unbounded() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    let label = "L".repeat(256 * 1024);
    for _ in 0..20 {
        let _ = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&label),
            size: 4,
            usage: wgpu::BufferUsages::empty(),
            mapped_at_creation: false,
        });
    }
    let stored: usize = lock(&ctx.inner.errors).msgs.iter().map(String::len).sum();
    let err = ctx.check_errors().expect_err("sticky");
    let n = format!("{err:?}").len();
    eprintln!("ADV A4 stored text = {stored} bytes in 16 msgs; one check_errors() error = {n} bytes");
}

/// A4: the dropped count under concurrent callbacks.
#[test]
fn adv_dropped_count_exact_under_concurrent_callbacks() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    let (threads, per) = (8usize, 50usize);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                for _ in 0..per {
                    let _ = ctx.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("e"),
                        size: 4,
                        usage: wgpu::BufferUsages::empty(),
                        mapped_at_creation: false,
                    });
                }
            });
        }
    });
    let (kept, dropped) = {
        let q = lock(&ctx.inner.errors);
        (q.msgs.len(), q.dropped)
    };
    eprintln!("ADV A4 kept={kept} dropped={dropped} total={}", kept as u64 + dropped);
    assert_eq!(kept as u64 + dropped, (threads * per) as u64);
    // Unicode text round trip.
    let ctx2 = WgpuContext::open().expect("wgpu adapter");
    ctx2.push_uncaptured("é世界🙂");
    assert!(format!("{:?}", ctx2.check_errors().unwrap_err()).contains("世界"));
}

/// A4: nothing clears the sticky record: rebudget-like clone, many checks, reads.
#[test]
fn adv_nothing_clears_sticky_record() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    ctx.push_uncaptured("one");
    let clone = ctx.clone();
    for _ in 0..5 {
        assert!(ctx.check_errors().is_err());
        assert!(clone.sync().is_err());
        let _ = ctx.failure("x".to_string());
    }
    ctx.trim_pool();
    assert!(clone.check_errors().is_err());
}
