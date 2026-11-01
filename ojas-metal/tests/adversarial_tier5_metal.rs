//! Milestone 5 Tier 5: Adversarial Coverage Hardening for `ojas-metal` and `tessl`
//!
//! White-box adversarial test suite exercising:
//! 1. `SharedBuf` and `MappedBuffer` lease concurrency, mutual exclusion, and GPU stamping
//! 2. UMA zero-copy memory sharing, budget reservation, and fail-closed placement enforcement
//! 3. Timeline monotonic fences, uncommitted stamp handling, and poison propagation
//! 4. GPU status word checks (`ST_IN`, `ST_OUT`), non-finite fault detection, and clean cache withdrawal
//! 5. Buffer pool freelist cache trim ladder under memory pressure
//! 6. High-concurrency multi-threaded contention stress harness

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use common::*;
use ojas_core::{
    Backend, BackendId, Budget, ConvTranspose2dParams, DType, DeviceBuffer, FenceError,
    MappedBuffer, OjasError, Tensor,
};
use ojas_metal::BudgetLedger;
use tessl::shared_buf::{GpuRole, HostAccessError, SharedAlloc};
use tessl::{BufferKind, GpuRuntime, PoolLedger};

const WATCHDOG: Duration = Duration::from_secs(120);

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
            panic!("watchdog: no result after {WATCHDOG:?}; thread execution timed out or hung")
        }
    }
}

// =========================================================================
// 1. SharedBuf and MappedBuffer Fencing & Lease Concurrency Protocol
// =========================================================================

#[test]
fn test_shared_buf_concurrent_read_leases_and_write_exclusion() {
    watchdog(|| {
        let rt = GpuRuntime::new().expect("Metal device");
        let alloc = SharedAlloc::new(&rt, 4096).expect("4096-byte SharedAlloc");
        let buf = alloc.shared_buf();

        // 1. Concurrently take 16 read leases across 8 threads
        let read_count = Arc::new(AtomicUsize::new(0));
        let barrier_start = Arc::new(AtomicBool::new(false));
        let barrier_release = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let b = buf.clone();
            let count = Arc::clone(&read_count);
            let start = Arc::clone(&barrier_start);
            let release = Arc::clone(&barrier_release);

            handles.push(thread::spawn(move || {
                while !start.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                let lease1 = b.read().expect("read lease 1");
                let lease2 = b.read().expect("read lease 2");
                assert_eq!(lease1.len(), 4096);
                assert_eq!(lease2.len(), 4096);
                assert!(lease1.as_f32().is_some());
                assert!(lease2.as_u32().is_some());

                count.fetch_add(2, Ordering::SeqCst);

                while !release.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                drop(lease1);
                drop(lease2);
            }));
        }

        barrier_start.store(true, Ordering::Release);
        while read_count.load(Ordering::SeqCst) < 16 {
            thread::yield_now();
        }

        // 2. While read leases are active:
        // - check_gpu_access(Read) must succeed
        assert_eq!(buf.check_gpu_access(GpuRole::Read), Ok(()));
        // - check_gpu_access(Write) must fail with Busy
        assert!(matches!(
            buf.check_gpu_access(GpuRole::Write),
            Err(HostAccessError::Busy(_))
        ));
        // - write() must fail with Busy
        assert!(matches!(buf.write(), Err(HostAccessError::Busy(_))));

        // 3. Release read leases
        barrier_release.store(true, Ordering::Release);
        for h in handles {
            h.join().expect("reader thread join");
        }

        // 4. Once all read leases dropped: write() and check_gpu_access(Write) must succeed
        assert_eq!(buf.check_gpu_access(GpuRole::Write), Ok(()));
        let mut write_lease = buf.write().expect("exclusive write lease");
        assert_eq!(write_lease.len(), 4096);
        if let Some(f32s) = write_lease.as_f32_mut() {
            f32s[0] = 123.456;
        }
        drop(write_lease);

        // Verify written data visible in subsequent read lease
        let read_lease = buf.read().expect("read lease after write");
        assert_eq!(read_lease.as_f32().unwrap()[0], 123.456);
    });
}

#[test]
fn test_shared_buf_exclusive_write_lease_blocks_all_accesses() {
    watchdog(|| {
        let rt = GpuRuntime::new().expect("Metal device");
        let alloc = SharedAlloc::new(&rt, 2048).expect("SharedAlloc");
        let buf = alloc.shared_buf();

        let write_lease = buf.write().expect("first write lease");

        // While write lease is held:
        // - Second write fails
        assert!(matches!(buf.write(), Err(HostAccessError::Busy(_))));
        // - Read fails
        assert!(matches!(buf.read(), Err(HostAccessError::Busy(_))));
        // - check_gpu_access(Read) fails
        assert!(matches!(
            buf.check_gpu_access(GpuRole::Read),
            Err(HostAccessError::Busy(_))
        ));
        // - check_gpu_access(Write) fails
        assert!(matches!(
            buf.check_gpu_access(GpuRole::Write),
            Err(HostAccessError::Busy(_))
        ));

        drop(write_lease);

        // After dropping, both read and write become available
        assert_eq!(buf.check_gpu_access(GpuRole::Read), Ok(()));
        assert_eq!(buf.check_gpu_access(GpuRole::Write), Ok(()));
    });
}

#[test]
fn test_shared_buf_gpu_binding_and_fence_stamps() {
    let rt = GpuRuntime::new().expect("Metal device");
    let alloc = SharedAlloc::new(&rt, 1024).expect("SharedAlloc");

    assert_eq!(alloc.last_use(), 0);
    assert_eq!(alloc.last_write(), 0);

    // 1. Bind shared read-only using rt.with_binder
    let res = rt.with_binder(|bnd| {
        bnd.bind_shared_ro(&alloc, 0);
        Ok(())
    });
    assert!(res.is_ok(), "bind_shared_ro on idle buffer");
    assert!(alloc.last_use() > 0, "last_use was stamped");
    assert_eq!(alloc.last_write(), 0, "read-only bind does not stamp write");

    // 2. Bind shared read-write: must stamp both last_use and last_write
    let res_rw = rt.with_binder(|bnd| {
        bnd.bind_shared_rw(&alloc, 0);
        Ok(())
    });
    assert!(res_rw.is_ok(), "bind_shared_rw on idle buffer");
    assert!(alloc.last_use() > 0);
    assert!(alloc.last_write() > 0, "last_write was stamped");

    // 3. Pre-flight check_gpu_access with write lease
    let write_lease = alloc.write().expect("write lease");
    assert!(matches!(
        alloc.check_gpu_access(GpuRole::Read),
        Err(HostAccessError::Busy(_))
    ));
    assert!(matches!(
        alloc.check_gpu_access(GpuRole::Write),
        Err(HostAccessError::Busy(_))
    ));

    // 4. A conflicting GPU bind is a caller bug and poisons the runtime
    let res_conflict = rt.with_binder(|bnd| {
        bnd.bind_shared_ro(&alloc, 0);
        Ok(())
    });
    assert!(
        res_conflict.is_err(),
        "bind_shared_ro must fail and poison while write lease is live"
    );
    drop(write_lease);

    // Runtime is now poisoned as expected by Metal binder protocol
    let res_after_poison = rt.with_binder(|_bnd| Ok(()));
    assert!(
        res_after_poison.is_err(),
        "runtime must be poisoned after binder conflict"
    );
}

#[test]
fn test_shared_buf_subword_lengths_and_outofbounds() {
    let rt = GpuRuntime::new().expect("Metal device");

    // Allocate 7 bytes (not a multiple of 4)
    let alloc = SharedAlloc::new(&rt, 7).expect("7-byte SharedAlloc");
    let buf = alloc.shared_buf();
    assert_eq!(buf.len(), 7);
    assert!(!buf.is_empty());

    // HostRead: as_f32 / as_u32 must return None on unaligned / non-word length
    let read_lease = buf.read().expect("read lease");
    assert!(read_lease.as_f32().is_none());
    assert!(read_lease.as_u32().is_none());
    drop(read_lease);

    // MappedBuffer::host_read: as_f32 / as_u32 must return OjasError::Shape
    let mapped_read = buf.host_read().expect("mapped host_read");
    assert!(matches!(mapped_read.as_f32(), Err(OjasError::Shape { .. })));
    assert!(matches!(mapped_read.as_u32(), Err(OjasError::Shape { .. })));
    drop(mapped_read);

    // MappedBuffer::host_write: as_f32_mut / as_u32_mut must return OjasError::Shape
    let mut mapped_write = buf.host_write().expect("mapped host_write");
    assert!(matches!(
        mapped_write.as_f32_mut(),
        Err(OjasError::Shape { .. })
    ));
    assert!(matches!(
        mapped_write.as_u32_mut(),
        Err(OjasError::Shape { .. })
    ));
    drop(mapped_write);

    // DeviceBuffer::read_bytes bounds checks
    // Overflowing offset + len
    assert!(matches!(
        buf.read_bytes(usize::MAX, 1),
        Err(OjasError::OutOfRange { .. })
    ));
    // Range out of bounds
    assert!(matches!(
        buf.read_bytes(4, 5),
        Err(OjasError::OutOfRange { .. })
    ));
    // Valid read
    let bytes = buf.read_bytes(0, 7).expect("read 7 bytes");
    assert_eq!(bytes.len(), 7);

    // Zero-length allocation clamps to at least 1 byte
    let zero_alloc = SharedAlloc::new(&rt, 0).expect("zero-byte SharedAlloc");
    assert!(!zero_alloc.is_empty());
}

// =========================================================================
// 2. UMA Zero-Copy Memory Sharing and Fail-Closed Placement Enforcement
// =========================================================================

#[test]
fn test_uma_zero_copy_tensor_placement_and_budget_lifecycle() {
    let rt = GpuRuntime::new().expect("Metal device");
    let budget = Budget::new(64 * 1024);

    let alloc = SharedAlloc::new(&rt, 256).expect("256-byte SharedAlloc");
    let buf = alloc.shared_buf();

    // Populate buffer with known values via write lease
    {
        let mut wl = buf.write().expect("write lease");
        let floats = wl.as_f32_mut().expect("f32 slice");
        for (i, v) in floats.iter_mut().enumerate() {
            *v = (i as f32) * 1.5;
        }
    }

    // Wrap into Tensor::from_mapped
    let shared_arc: Arc<dyn MappedBuffer> = Arc::new(buf);
    let tensor =
        Tensor::from_mapped(shared_arc, &[8, 8], DType::F32, &budget).expect("Tensor::from_mapped");

    // Invariants check
    assert_eq!(tensor.device(), Some(BackendId::Metal));
    assert!(tensor.mapped_buffer().is_some());
    assert!(tensor.device_buffer().is_none());
    assert_eq!(budget.live_bytes().unwrap(), 256);

    // Fail-closed placement enforcement: Host and CPU ops must refuse!
    assert!(matches!(
        tensor.f32_slice(),
        Err(OjasError::Placement {
            expected: None,
            found: Some(BackendId::Metal),
            ..
        })
    ));

    let c = cpu();
    let cpu_op_err = c.linear_forward(&tensor, &tensor);
    assert!(
        matches!(
            cpu_op_err,
            Err(OjasError::Placement {
                expected: None,
                found: Some(BackendId::Metal),
                ..
            })
        ),
        "CPU op must refuse Metal mapped tensor without silent fallback: {cpu_op_err:?}"
    );

    // Zero-copy readback via host_view()
    let host_view = tensor.host_view().expect("tensor host_view");
    let f32s = host_view.as_f32().expect("view as f32");
    assert_eq!(f32s.len(), 64);
    for (i, &v) in f32s.iter().enumerate() {
        assert_eq!(v, (i as f32) * 1.5);
    }

    // to_host returns a timeline-fenced clone sharing the mapped storage
    let host_clone = tensor.to_host(&budget).expect("to_host clone");
    assert!(host_clone.mapped_buffer().is_some());

    drop(tensor);
    drop(host_clone);
    drop(host_view);
    // Budget must be completely released
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

#[test]
fn test_uma_tensor_reservation_mismatch_and_mutation_ownership() {
    let rt = GpuRuntime::new().expect("Metal device");
    let budget = Budget::new(64 * 1024);

    let alloc = SharedAlloc::new(&rt, 512).expect("512-byte SharedAlloc");
    let buf = alloc.shared_buf();
    let shared_arc: Arc<dyn MappedBuffer> = Arc::new(buf);

    // Reservation size mismatch: reserving 256 bytes for a 512-byte buffer
    let mismatched_res = budget.try_reserve(256).expect("reserve 256");
    let err =
        Tensor::from_mapped_reserved(Arc::clone(&shared_arc), &[64], DType::F32, mismatched_res);
    assert!(matches!(err, Err(OjasError::OutOfRange { .. })));

    // Correct reservation
    let correct_res = budget.try_reserve(512).expect("reserve 512");
    let mut tensor =
        Tensor::from_mapped_reserved(Arc::clone(&shared_arc), &[128], DType::F32, correct_res)
            .expect("from_mapped_reserved");

    // Mutation ownership: uniquely owned tensor can acquire device_buffer_mut
    drop(shared_arc); // Drop outside handle so tensor is sole Arc owner
    let dev_mut = tensor.device_buffer_mut();
    assert!(
        dev_mut.is_ok(),
        "uniquely owned mapped tensor device_buffer_mut"
    );

    // Cloned tensor shares storage -> device_buffer_mut must refuse!
    let tensor_clone = tensor.clone();
    let shared_err = tensor.device_buffer_mut();
    assert!(
        matches!(shared_err, Err(OjasError::Shape { .. })),
        "shared storage must refuse device_buffer_mut: {shared_err:?}"
    );
    drop(tensor_clone);
}

// =========================================================================
// 3. Timeline Monotonic Fences and Poison Propagation
// =========================================================================

#[test]
fn test_timeline_uncommitted_stamp_and_poison_propagation() {
    let rt = GpuRuntime::new().expect("Metal device");
    let alloc = SharedAlloc::new(&rt, 1024).expect("SharedAlloc");
    let timeline = alloc.timeline();

    let submitted = timeline.submitted().expect("fence submitted");
    let completed = timeline.completed().expect("fence completed");
    assert!(completed <= submitted);

    // Waiting on a stamp above submitted must answer FenceError::NotSubmitted immediately
    let uncommitted_stamp = submitted + 500;
    let wait_res = timeline.wait(uncommitted_stamp);
    assert!(
        matches!(
            wait_res,
            Err(FenceError::NotSubmitted { value, submitted: sub }) if value == uncommitted_stamp && sub == submitted
        ),
        "waiting on unsubmitted stamp must fail fast: {wait_res:?}"
    );

    // Timeline::wait_for convenience wrapper
    let wait_for_res = timeline.wait_for(uncommitted_stamp, None);
    assert!(wait_for_res.is_err());

    // Poison propagation on standalone runtime
    let rt_poisoned = GpuRuntime::new().expect("Metal device");
    let alloc_poisoned = SharedAlloc::new(&rt_poisoned, 1024).expect("SharedAlloc on rt2");
    let poison_timeline = alloc_poisoned.timeline();

    rt_poisoned.poison_as_shared_event_timeout_for_test();

    assert!(matches!(
        poison_timeline.submitted(),
        Err(FenceError::Poisoned)
    ));
    assert!(matches!(
        poison_timeline.completed(),
        Err(FenceError::Poisoned)
    ));
    assert!(matches!(poison_timeline.wait(0), Err(FenceError::Poisoned)));

    // SharedAlloc creation on poisoned runtime must fail
    assert!(SharedAlloc::new(&rt_poisoned, 1024).is_err());
}

// =========================================================================
// 4. GPU Status Word Checks (ST_IN, ST_OUT) and Fault Propagation
// =========================================================================

#[test]
fn test_status_word_checks_and_nonfinite_fault_detection() {
    let m = metal();

    // 1. gelu_tanh_forward with NaN input
    let nan_vec = vec![0.5f32, f32::NAN, 0.2f32, -1.0f32];
    let dev_nan = up(&m, &host(&nan_vec, &[4]));
    let _ = m.gelu_tanh_forward(&dev_nan);
    let sync_err = m.sync();
    assert!(
        matches!(sync_err, Err(OjasError::NonFinite { op }) if op == "gelu_tanh_forward"),
        "gelu_tanh_forward must detect NaN input via ST_IN: {sync_err:?}"
    );

    // 2. gelu_tanh_backward with NaN input
    let clean = up(&m, &host(&[0.5f32, 0.2f32, 0.1f32, -0.5f32], &[4]));
    let _ = m.gelu_tanh_backward(&dev_nan, &clean);
    let sync_err_bwd = m.sync();
    assert!(
        matches!(sync_err_bwd, Err(OjasError::NonFinite { op }) if op == "gelu_tanh_backward"),
        "gelu_tanh_backward must detect NaN input via ST_IN: {sync_err_bwd:?}"
    );

    // 3. max_pool2d_forward with NaN input
    let pool_nan = vec![1.0f32, 2.0, f32::NAN, 4.0];
    let dev_pool_nan = up(&m, &host(&pool_nan, &[1, 1, 2, 2]));
    let _ = m.max_pool2d_forward(&dev_pool_nan, [2, 2], [1, 1]);
    let sync_pool = m.sync();
    assert!(
        matches!(sync_pool, Err(OjasError::NonFinite { op }) if op == "max_pool2d_forward"),
        "max_pool2d_forward must detect NaN input via ST_IN: {sync_pool:?}"
    );

    // 4. conv_transpose2d_forward with NaN input
    let ct_nan = vec![f32::NAN];
    let dev_ct_nan = up(&m, &host(&ct_nan, &[1, 1, 1, 1]));
    let ct_w = up(&m, &host(&[1.0f32], &[1, 1, 1, 1]));
    let params = ConvTranspose2dParams::new();
    let _ = m.conv_transpose2d_forward(&dev_ct_nan, &ct_w, None, params);
    let sync_ct = m.sync();
    assert!(
        matches!(sync_ct, Err(OjasError::NonFinite { op }) if op == "conv_transpose2d_forward"),
        "conv_transpose2d_forward must detect NaN input via ST_IN: {sync_ct:?}"
    );

    // 5. seq_mean_forward with NaN input
    let dev_seq_nan = up(&m, &host(&[1.0f32, f32::NAN, 3.0, 4.0], &[1, 2, 2]));
    let key_lens = up(&m, &host_u32(&[2], &[1]));
    let _ = m.seq_mean_forward(&dev_seq_nan, &key_lens);
    let sync_seq = m.sync();
    assert!(
        matches!(sync_seq, Err(OjasError::NonFinite { op }) if op == "seq_mean_forward"),
        "seq_mean_forward must detect NaN input via ST_IN: {sync_seq:?}"
    );

    // 6. l2_normalize_forward with NaN input
    let dev_l2_nan = up(&m, &host(&[f32::NAN, 2.0, 3.0, 4.0], &[1, 4]));
    let _ = m.l2_normalize_forward(&dev_l2_nan, 1e-12);
    let sync_l2 = m.sync();
    assert!(
        matches!(sync_l2, Err(OjasError::NonFinite { op }) if op == "l2_normalize_forward"),
        "l2_normalize_forward must detect NaN input via ST_IN: {sync_l2:?}"
    );

    // 7. Verify subsequent clean op succeeds and is not corrupted by previous faults
    let clean_res = m.gelu_tanh_forward(&clean);
    assert!(clean_res.is_ok());
    assert!(matches!(m.sync(), Ok(())));
}

#[test]
fn test_empty_and_noncontiguous_tensor_rejections() {
    let m = metal();
    let budget = Budget::new(64 * 1024);

    // Empty tensor upload
    let empty_t = Tensor::zeros(&[0], DType::F32, &budget).expect("zeros [0]");
    let upload_empty_err = m.upload(&empty_t);
    assert!(
        matches!(
            upload_empty_err,
            Err(OjasError::Shape {
                op: "MetalBackend::upload",
                ..
            })
        ),
        "upload of empty tensor must refuse: {upload_empty_err:?}"
    );

    // Non-contiguous tensor upload
    let t_2d = host(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t_strided = t_2d.view(&[2, 2], &[1, 2], 0).expect("strided view");
    assert!(!t_strided.is_contiguous().expect("is_contiguous"));
    let upload_strided_err = m.upload(&t_strided);
    assert!(
        matches!(
            upload_strided_err,
            Err(OjasError::Shape {
                op: "MetalBackend::upload",
                ..
            })
        ),
        "upload of non-contiguous tensor must refuse: {upload_strided_err:?}"
    );

    // l2_normalize_forward unsupported epsilon
    let dev_x = up(&m, &host(&[1.0, 2.0, 3.0, 4.0], &[1, 4]));
    let eps_err = m.l2_normalize_forward(&dev_x, 1e-5);
    assert!(
        matches!(eps_err, Err(OjasError::Unsupported { .. })),
        "l2_normalize_forward with eps != 1e-12 must refuse: {eps_err:?}"
    );
}

// =========================================================================
// 5. Freelist Cache Trim Under Memory Pressure Ladder
// =========================================================================

#[test]
fn test_freelist_cache_trim_ladder_under_memory_pressure() {
    let root = Budget::new(256 * 1024);
    let rt = GpuRuntime::new().expect("Metal device");

    let ledger = BudgetLedger::new(&root, 128 * 1024);
    rt.set_pool_ledger(Some(ledger.as_pool_ledger()))
        .expect("set_pool_ledger");

    // Allocate two cold buffers that each take 64 KiB
    let b1 = rt
        .alloc_buffer_kind(64 * 1024, BufferKind::Cold)
        .expect("b1");
    let b2 = rt
        .alloc_buffer_kind(64 * 1024, BufferKind::Cold)
        .expect("b2");
    assert_eq!(ledger.live_bytes(), 128 * 1024);
    assert_eq!(ledger.cached_bytes(), 0);

    // Drop them so they become idle
    drop(b1);
    drop(b2);

    // Probe allocation to trigger idle retirement onto freelist
    let probe = rt.alloc_buffer_kind(1, BufferKind::Cold).expect("probe");
    assert_eq!(ledger.cached_bytes(), 128 * 1024);
    drop(probe);

    // Now request fresh allocation of 192 KiB.
    // Root cap is 256 KiB.
    // 192 KiB fresh + 128 KiB cached = 320 KiB > 256 KiB cap!
    // The freelist trim ladder must purge cached buffers and succeed!
    let big = rt
        .alloc_buffer_kind(192 * 1024, BufferKind::Cold)
        .expect("192 KiB alloc via freelist trim");
    assert_eq!(ledger.cached_bytes(), 0, "freelist was trimmed");
    assert!(ledger.live_bytes() >= 192 * 1024);
    drop(big);

    // Allocation past root budget cap must fail cleanly
    let impossible_err = rt.alloc_buffer_kind(512 * 1024, BufferKind::Cold);
    assert!(impossible_err.is_err(), "impossible allocation must refuse");
    assert_eq!(ledger.unmatched_releases(), 0);
}

// =========================================================================
// 6. High-Concurrency Multi-Threaded Contention Stress Harness
// =========================================================================

#[test]
fn test_multithreaded_shared_buffer_contention_and_stress() {
    watchdog(|| {
        let rt = GpuRuntime::new().expect("Metal device");
        let budget = Arc::new(Budget::new(8 * 1024 * 1024));

        // Create SharedBuf instances on runtime thread and distribute to workers
        let mut shared_buffers = Vec::new();
        let mut allocs = Vec::new();
        for _ in 0..8 {
            let alloc = SharedAlloc::new(&rt, 512).expect("alloc 512");
            shared_buffers.push(alloc.shared_buf());
            allocs.push(alloc);
        }

        let mut threads = Vec::new();
        for (thread_id, buf) in shared_buffers.into_iter().enumerate() {
            let budget = Arc::clone(&budget);

            threads.push(thread::spawn(move || {
                for iter in 0..50 {
                    // 1. Write phase
                    {
                        let mut wl = buf.write().expect("write lease");
                        if let Some(f32s) = wl.as_f32_mut() {
                            for (i, v) in f32s.iter_mut().enumerate() {
                                *v = (thread_id * 1000 + iter * 10 + i) as f32;
                            }
                        }
                    }

                    // 2. Read phase
                    {
                        let rl = buf.read().expect("read lease");
                        if let Some(f32s) = rl.as_f32() {
                            assert_eq!(f32s[0], (thread_id * 1000 + iter * 10) as f32);
                        }
                    }

                    // 3. Tensor wrapping and host_view
                    let shared_arc: Arc<dyn MappedBuffer> = Arc::new(buf.clone());
                    let tensor = Tensor::from_mapped(shared_arc, &[128], DType::F32, &budget)
                        .expect("Tensor::from_mapped");

                    let view = tensor.host_view().expect("host_view");
                    let f32s = view.as_f32().expect("view as f32");
                    assert_eq!(f32s[0], (thread_id * 1000 + iter * 10) as f32);

                    drop(view);
                    drop(tensor);
                }
            }));
        }

        for t in threads {
            t.join().expect("thread join");
        }

        drop(allocs);
        assert_eq!(
            budget.live_bytes().unwrap(),
            0,
            "all reservations must be freed"
        );
    });
}

// =========================================================================
// 7. Output Non-Finite Status Check (ST_OUT) and Strided UMA Views
// =========================================================================

#[test]
fn test_status_word_output_overflow_st_out() {
    let m = metal();

    // Both inputs are finite, but their product overflows to +inf in linear_forward.
    // This leaves ST_IN = 0, but triggers ST_OUT = 1 on GPU!
    let big = 1e30f32;
    assert!(big.is_finite());

    let dev_x = up(&m, &host(&[big, big, big, big], &[2, 2]));
    let dev_w = up(&m, &host(&[big, big, big, big], &[2, 2]));

    let _ = m.linear_forward(&dev_x, &dev_w);
    let sync_res = m.sync();
    assert!(
        matches!(sync_res, Err(OjasError::NonFinite { op }) if op == "linear_forward"),
        "finite inputs producing infinite output must be caught by ST_OUT: {sync_res:?}"
    );

    // Verify recovery after non-finite output fault
    let clean_x = up(&m, &host(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let clean_w = up(&m, &host(&[2.0, 0.0, 0.0, 2.0], &[2, 2]));
    let clean_out = m.linear_forward(&clean_x, &clean_w).expect("clean linear");
    assert!(m.sync().is_ok());
    let res_vec = down(&clean_out);
    assert_eq!(res_vec, vec![2.0, 0.0, 0.0, 2.0]);
}

#[test]
fn test_budget_ledger_unmatched_release_and_capacity_error_parsing() {
    let root = Budget::new(64 * 1024);
    let ledger = BudgetLedger::new(&root, 32 * 1024);

    // Charge 16 KiB live
    ledger
        .try_charge(tessl::LedgerClass::Live, 16 * 1024)
        .expect("charge 16 KiB");
    assert_eq!(ledger.live_bytes(), 16 * 1024);

    // Unmatched release for an uncharged size
    ledger.release(tessl::LedgerClass::Live, 8 * 1024);
    assert_eq!(ledger.unmatched_releases(), 1);
    assert_eq!(
        ledger.live_bytes(),
        16 * 1024,
        "unmatched release must not decrement live bytes"
    );

    // Release actual charge
    ledger.release(tessl::LedgerClass::Live, 16 * 1024);
    assert_eq!(ledger.live_bytes(), 0);
    assert_eq!(ledger.unmatched_releases(), 1);

    // Capacity error parsing test
    let sample_refusal = tessl::LedgerRefused {
        requested: 4096,
        cap: 16384,
        live: 16384,
    }
    .to_string();
    let parsed = BudgetLedger::capacity_error(&sample_refusal);
    assert!(
        matches!(
            parsed,
            Some(OjasError::CapacityExceeded {
                requested: 4096,
                cap: 16384,
                live: 16384
            })
        ),
        "capacity_error must parse LedgerRefused string into OjasError::CapacityExceeded: {parsed:?}"
    );

    // Unrelated text should return None
    assert!(BudgetLedger::capacity_error("unrelated driver error").is_none());
}

#[test]
fn test_uma_strided_mapped_tensor_view() {
    let rt = GpuRuntime::new().expect("Metal device");
    let budget = Budget::new(64 * 1024);

    // Allocate 16 floats (64 bytes)
    let alloc = SharedAlloc::new(&rt, 64).expect("64-byte SharedAlloc");
    let buf = alloc.shared_buf();

    // Populate data [0.0, 1.0, 2.0, ..., 15.0]
    {
        let mut wl = buf.write().expect("write lease");
        let f32s = wl.as_f32_mut().expect("f32 slice");
        for (i, v) in f32s.iter_mut().enumerate() {
            *v = i as f32;
        }
    }

    let shared_arc: Arc<dyn MappedBuffer> = Arc::new(buf);
    let tensor =
        Tensor::from_mapped(shared_arc, &[4, 4], DType::F32, &budget).expect("Tensor::from_mapped");

    // Take a strided/transposed view [4, 4] with strides [1, 4]
    let strided_view = tensor.view(&[4, 4], &[1, 4], 0).expect("strided view");
    assert!(!strided_view.is_contiguous().expect("is_contiguous"));

    // to_host on strided mapped tensor preserves mapped representation
    let host_mapped = strided_view.to_host(&budget).expect("to_host on mapped");
    assert!(host_mapped.mapped_buffer().is_some());

    // host_view on base tensor gives raw slice
    let host_v = tensor.host_view().expect("host_view");
    let raw_f32 = host_v.as_f32().expect("raw f32");
    assert_eq!(raw_f32[5], 5.0);
}
