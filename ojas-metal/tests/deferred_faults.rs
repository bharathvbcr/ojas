//! The deferred-fault contract (`docs/metal-deferred-faults.md` §7): a
//! device-detected non-finite value does not fail its op; the next
//! `sync`, `download` or `clip_grad_norm` on the backend reports the first
//! faulting op in recording order, once. Host-decided refusals stay
//! immediate, and in-place ops stay all-or-nothing.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::{Arc, Barrier};
use std::thread;

use common::*;
use ojas_core::{AdamWConfig, Backend, Budget, CeChunk, MuonNs5Config, OjasError, Tensor};
use ojas_metal::MetalBackend;

/// The status slab's slot count (`SLAB_SLOTS` in `device.rs`).
const SLAB_SLOTS: usize = 4096;

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// What `sync` reports: `None` for `Ok`, the op for `NonFinite`.
fn pending(m: &MetalBackend) -> Option<&'static str> {
    match m.sync() {
        Ok(()) => None,
        Err(OjasError::NonFinite { op }) => Some(op),
        Err(e) => panic!("sync: {e:?}"),
    }
}

fn nan_at_last(n: usize, seed: u64, shape: &[usize]) -> Tensor {
    let mut v = values(n, seed, 1.0);
    v[n - 1] = f32::NAN;
    host(&v, shape)
}

#[test]
fn the_first_faulting_op_is_named_and_reported_once() {
    let m = metal();
    let x = up(&m, &host(&[1.0, f32::NAN, 2.0], &[3]));
    let b = up(&m, &rand(&[3], 1, 1.0));
    let y = ok("silu records", m.silu_forward(&x));
    let z = ok("mul records", m.mul_forward(&y, &b));
    assert_eq!(pending(&m), Some("silu_forward"));
    assert_eq!(pending(&m), None, "a fault is reported once");
    // A later fault in another op is named after itself.
    ok("clean", m.silu_forward(&b));
    ok("mul records", m.mul_forward(&b, &x));
    assert_eq!(pending(&m), Some("mul_forward"));
    assert_eq!(pending(&m), None);
    drop(z);
}

#[test]
fn a_later_clean_op_does_not_clear_a_pending_fault() {
    let m = metal();
    let x = up(&m, &host(&[1.0, 2.0, f32::INFINITY, 3.0], &[2, 2]));
    let w = up(&m, &rand(&[2], 2, 1.0));
    let good = up(&m, &rand(&[2, 2], 3, 1.0));
    ok("rms records", m.rms_norm_forward(&x, &w, 1e-6));
    let mut y = good.clone();
    for _ in 0..50 {
        y = ok("clean", m.silu_forward(&y));
    }
    assert_eq!(pending(&m), Some("rms_norm_forward"));
    assert_eq!(pending(&m), None);
    // The clean chain's values are intact.
    assert!(down(&y).iter().all(|v| v.is_finite()));
}

/// p, m, v and a gradient with a NaN at its last element.
struct AdamState {
    p: Tensor,
    m1: Tensor,
    m2: Tensor,
    g: Tensor,
}

fn adam_state(m: &MetalBackend, seed: u64) -> AdamState {
    let shape = [5usize, 7];
    AdamState {
        p: up(m, &rand(&shape, seed, 1.0)),
        m1: up(m, &rand(&shape, seed + 1, 0.1)),
        m2: up(m, &host(&values(35, seed + 2, 0.01).iter().map(|v| v.abs()).collect::<Vec<_>>(), &shape)),
        g: up(m, &nan_at_last(35, seed + 3, &shape)),
    }
}

fn adam_fault(m: &MetalBackend, s: &mut AdamState) -> [Vec<u32>; 3] {
    let before = [bits(&s.p), bits(&s.m1), bits(&s.m2)];
    ok(
        "adamw records",
        m.adamw_step(&mut s.p, &s.g, &mut s.m1, &mut s.m2, 2, AdamWConfig::nanolab(1e-3, 0.1)),
    );
    before
}

#[test]
fn every_sync_point_reports_a_pending_adamw_fault() {
    let m = metal();
    let unrelated = up(&m, &rand(&[4], 9, 1.0));
    let unrelated_vals = down(&unrelated);

    // sync
    let mut s = adam_state(&m, 10);
    let before = adam_fault(&m, &mut s);
    assert_eq!(pending(&m), Some("adamw_step"));
    assert_eq!(pending(&m), None);
    assert_eq!([bits(&s.p), bits(&s.m1), bits(&s.m2)], before);

    // download of an unrelated tensor: reads, then reports.
    let mut s = adam_state(&m, 20);
    adam_fault(&m, &mut s);
    let r = m.download(&unrelated);
    assert!(matches!(r, Err(OjasError::NonFinite { op: "adamw_step" })), "{r:?}");
    assert_eq!(pending(&m), None);
    let clean = ok("clean download", m.download(&unrelated));
    assert_eq!(ok("vals", clean.to_f32_vec()), unrelated_vals);

    // clip_grad_norm: reports before it scales anything.
    let mut s = adam_state(&m, 30);
    adam_fault(&m, &mut s);
    let mut grads = vec![up(&m, &host(&[3.0, 4.0], &[2])), up(&m, &host(&[12.0], &[1]))];
    let gbits: Vec<_> = grads.iter().map(bits).collect();
    let r = m.clip_grad_norm(&mut grads, 0.1);
    assert!(matches!(r, Err(OjasError::NonFinite { op: "adamw_step" })), "{r:?}");
    assert_eq!(grads.iter().map(bits).collect::<Vec<_>>(), gbits, "clip scaled");
    assert_eq!(pending(&m), None);
    let norm = ok("clean clip", m.clip_grad_norm(&mut grads, 0.1));
    assert_eq!(norm, 13.0);

    // A raw read waits but leaves the fault for the next sync point.
    let mut s = adam_state(&m, 40);
    adam_fault(&m, &mut s);
    assert_eq!(down(&unrelated), unrelated_vals);
    assert_eq!(down(&s.p).len(), 35);
    assert_eq!(pending(&m), Some("adamw_step"));
    assert_eq!(pending(&m), None);
}

/// §6: a NaN produced by an earlier op is reported by clip, which then
/// scales nothing.
#[test]
fn clip_reports_an_earlier_ops_nan_and_scales_nothing() {
    let m = metal();
    let x = up(&m, &host(&[f32::NAN, 1.0], &[2]));
    let produced = ok("silu records", m.silu_forward(&x));
    let mut grads = vec![up(&m, &host(&[30.0, 40.0], &[2]))];
    let before = bits(&grads[0]);
    let r = m.clip_grad_norm(&mut grads, 1.0);
    assert!(matches!(r, Err(OjasError::NonFinite { op: "silu_forward" })), "{r:?}");
    assert_eq!(bits(&grads[0]), before);
    assert_eq!(pending(&m), None);
    drop(produced);
}

#[test]
fn a_fault_survives_a_slab_overflow_and_is_named_after_one() {
    let m = metal();
    let one = up(&m, &host(&[0.5], &[1]));
    let nan = up(&m, &host(&[f32::NAN], &[1]));
    // Fault first, then more ops than the slab has slots.
    ok("records", m.silu_forward(&nan));
    for _ in 0..SLAB_SLOTS + 100 {
        ok("clean", m.silu_forward(&one));
    }
    assert_eq!(pending(&m), Some("silu_forward"));
    assert_eq!(pending(&m), None);
    // More clean ops than slots, then the fault.
    for _ in 0..SLAB_SLOTS + 100 {
        ok("clean", m.silu_forward(&one));
    }
    ok("records", m.mul_forward(&one, &nan));
    for _ in 0..10 {
        ok("clean", m.silu_forward(&one));
    }
    assert_eq!(pending(&m), Some("mul_forward"));
    assert_eq!(pending(&m), None);
}

#[test]
fn in_place_ops_stay_all_or_nothing_under_batching() {
    let m = metal();
    // AdamW: NaN at the last gradient element.
    let mut s = adam_state(&m, 50);
    let before = adam_fault(&m, &mut s);
    assert_eq!(pending(&m), Some("adamw_step"));
    assert_eq!([bits(&s.p), bits(&s.m1), bits(&s.m2)], before);
    // A later clean call on the same tensors applies.
    let g = up(&m, &rand(&[5, 7], 55, 1.0));
    ok("clean adamw", m.adamw_step(&mut s.p, &g, &mut s.m1, &mut s.m2, 2, AdamWConfig::nanolab(1e-3, 0.1)));
    assert_eq!(pending(&m), None);
    assert_ne!(bits(&s.p), before[0], "the clean step must apply");

    // Muon.
    let shape = [6usize, 9];
    let mut p = up(&m, &rand(&shape, 60, 1.0));
    let mut mo = up(&m, &rand(&shape, 61, 0.1));
    let bad = up(&m, &nan_at_last(54, 62, &shape));
    let before = (bits(&p), bits(&mo));
    let cfg = MuonNs5Config::nanolab_default();
    ok("muon records", m.muon_ns5_step(&mut p, &bad, &mut mo, cfg));
    assert_eq!(pending(&m), Some("muon_ns5_step"));
    assert_eq!((bits(&p), bits(&mo)), before);
    ok("clean muon", m.muon_ns5_step(&mut p, &up(&m, &rand(&shape, 63, 1.0)), &mut mo, cfg));
    assert_eq!(pending(&m), None);
    assert_ne!(bits(&p), before.0);

    // accumulate_grad, unique and shared.
    let ptr = |t: &Tensor| match t.device_buffer() {
        Some(b) => Arc::as_ptr(b) as *const (),
        None => panic!("not a device tensor"),
    };
    for shared in [false, true] {
        let mut acc = up(&m, &rand(&[3, 11], 70, 1.0));
        let other = shared.then(|| acc.clone());
        let before = bits(&acc);
        let p0 = ptr(&acc);
        ok("accumulate records", m.accumulate_grad(&mut acc, &up(&m, &nan_at_last(33, 71, &[3, 11]))));
        assert_eq!(pending(&m), Some("accumulate_grad"), "shared {shared}");
        assert_eq!(bits(&acc), before, "shared {shared}: acc values changed");
        if shared {
            assert_ne!(ptr(&acc), p0, "a shared acc gets a new buffer");
            assert_eq!(bits(other.as_ref().expect("other")), before);
        } else {
            assert_eq!(ptr(&acc), p0, "a unique acc keeps its buffer");
        }
        let g = rand(&[3, 11], 72, 1.0);
        ok("clean accumulate", m.accumulate_grad(&mut acc, &up(&m, &g)));
        assert_eq!(pending(&m), None);
        let want: Vec<f32> = before
            .iter()
            .zip(ok("g", g.to_f32_vec()))
            .map(|(&a, g)| f32::from_bits(a) + g)
            .collect();
        assert_eq!(down(&acc), want, "shared {shared}: the clean add");
    }

    // kv_cache_write.
    let (b, cap, hkv, d) = (1usize, 6usize, 2usize, 4usize);
    let mut cache = up(&m, &rand(&[b, cap, hkv, d], 80, 1.0));
    let before = bits(&cache);
    let src = up(&m, &nan_at_last(2 * hkv * d, 81, &[b, 2, hkv, d]));
    ok("kv write records", m.kv_cache_write(&mut cache, &src, 1));
    assert_eq!(pending(&m), Some("kv_cache_write"));
    assert_eq!(bits(&cache), before);
    ok("clean kv write", m.kv_cache_write(&mut cache, &up(&m, &rand(&[b, 2, hkv, d], 82, 1.0)), 1));
    assert_eq!(pending(&m), None);
    assert_ne!(bits(&cache), before);
}

/// An earlier op's pending fault does not stop a later in-place op: each
/// call decides from its own status words.
#[test]
fn a_pending_fault_does_not_block_a_later_in_place_op() {
    let m = metal();
    let nan = up(&m, &host(&[f32::NAN], &[1]));
    ok("silu records", m.silu_forward(&nan));
    let mut s = adam_state(&m, 90);
    let g = up(&m, &rand(&[5, 7], 95, 1.0));
    let before = bits(&s.p);
    ok("clean adamw", m.adamw_step(&mut s.p, &g, &mut s.m1, &mut s.m2, 1, AdamWConfig::nanolab(1e-3, 0.0)));
    assert_eq!(pending(&m), Some("silu_forward"));
    assert_ne!(bits(&s.p), before, "the clean step must apply despite the earlier fault");
}

/// Host-decided refusals return at the call and record nothing. The
/// precedence rows flip: an out-of-range id is refused at once even when
/// the same call's values are non-finite.
#[test]
fn host_refusals_stay_immediate_and_leave_nothing_pending() {
    let m = metal();
    let nothing = |what: &str| assert_eq!(pending(&m), None, "{what} left a fault pending");
    let good = up(&m, &rand(&[2, 3], 1, 1.0));
    let r = m.mul_forward(&good, &up(&m, &rand(&[3, 2], 2, 1.0)));
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    nothing("shape");
    let ids = up(&m, &host_u32(&[0, 4], &[2]));
    let r = m.silu_forward(&ids);
    assert!(matches!(r, Err(OjasError::Dtype { .. })), "{r:?}");
    nothing("dtype");
    let r = m.silu_forward(&rand(&[2], 3, 1.0));
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    nothing("placement");
    let small = ok("small", MetalBackend::new(Budget::new(64)));
    let x = up(&small, &host(&[1.0; 16], &[16]));
    let r = small.silu_forward(&x);
    assert!(matches!(r, Err(OjasError::CapacityExceeded { .. })), "{r:?}");
    assert_eq!(pending(&small), None, "capacity left a fault pending");
    let over = ojas_core::METAL_MAX_HEAD_DIM as usize + 16;
    let q = up(&m, &rand(&[1, 1, 2, over], 4, 1.0));
    let r = m.causal_sdpa_forward(&q, &q, &q);
    assert!(matches!(r, Err(OjasError::UnsupportedHeadDim { .. })), "{r:?}");
    nothing("head dim");
    let (cap, hkv, d) = (4usize, 1usize, 8usize);
    let qd = up(&m, &rand(&[1, 1, 2, d], 5, 1.0));
    let kc = up(&m, &rand(&[1, cap, hkv, d], 6, 1.0));
    let r = m.cached_attention_forward(&qd, &kc, &kc, cap + 1);
    assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}");
    nothing("kv_len");
    let mut cache = up(&m, &rand(&[1, cap, hkv, d], 7, 1.0));
    let r = m.kv_cache_write(&mut cache, &up(&m, &rand(&[1, 2, hkv, d], 8, 1.0)), 3);
    assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}");
    nothing("at + Tn");

    // Ids: out of range, with and without non-finite values in the call.
    let table = up(&m, &rand(&[4, 3], 9, 1.0));
    let nan_table = up(&m, &nan_at_last(12, 10, &[4, 3]));
    for t in [&table, &nan_table] {
        let r = m.embedding_forward(t, &ids);
        assert!(matches!(r, Err(OjasError::OutOfRange { op: "embedding_forward", .. })), "{r:?}");
        nothing("embedding id");
        let r = m.embedding_backward(t, &ids, &up(&m, &rand(&[2, 3], 11, 1.0)));
        assert!(matches!(r, Err(OjasError::OutOfRange { op: "embedding_backward", .. })), "{r:?}");
        nothing("embedding_backward id");
    }
    let logits = up(&m, &rand(&[2, 4], 12, 1.0));
    let nan_logits = up(&m, &nan_at_last(8, 13, &[2, 4]));
    for l in [&logits, &nan_logits] {
        let r = m.cross_entropy_mean_forward(l, &ids, None);
        assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}");
        nothing("ce target");
        let r = m.cross_entropy_mean_backward(l, &ids, None);
        assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}");
        nothing("ce bwd target");
        let all = up(&m, &host_u32(&[3, 3], &[2]));
        let r = m.cross_entropy_mean_backward(l, &all, Some(3));
        assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
        nothing("ce all ignored");
    }
    let (x, w) = (up(&m, &rand(&[2, 3], 14, 1.0)), up(&m, &rand(&[4, 3], 15, 1.0)));
    let nan_w = up(&m, &nan_at_last(12, 16, &[4, 3]));
    let c = CeChunk { rows: 1, cols: 2 };
    for ww in [&w, &nan_w] {
        let r = m.linear_cross_entropy_mean(&x, ww, &ids, None, c, true);
        assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}");
        nothing("lce target");
        let all = up(&m, &host_u32(&[1, 1], &[2]));
        let r = m.linear_cross_entropy_mean(&x, ww, &all, Some(1), c, false);
        assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
        nothing("lce all ignored");
    }
    // A target range check sees only the tensor's own window.
    let window = up(&m, &host_u32(&[9, 1, 2, 9], &[4]));
    let inner = ok("view", window.view(&[2], &[1], 4));
    let ce = m.cross_entropy_mean_forward(&logits, &inner, None);
    assert!(ce.is_ok(), "{ce:?}");
    nothing("windowed targets");
}

/// One thread's clean sequence (the round-3 mix in miniature). Returns the
/// bits of every output.
fn clean_sequence(m: &MetalBackend, seed: u64, iters: usize) -> Vec<Vec<u32>> {
    let (rows, d) = (16usize, 32usize);
    let x = up(m, &rand(&[rows, d], seed, 1.0));
    let w = up(m, &rand(&[d, d], seed + 1, 0.2));
    let rw = up(m, &rand(&[d], seed + 2, 1.0));
    let tgt = up(m, &host_u32(&ids(rows, seed + 3, d as u32), &[rows]));
    let mut p = up(m, &rand(&[d, d], seed + 4, 0.1));
    let mut m1 = up(m, &host(&vec![0.0; d * d], &[d, d]));
    let mut m2 = up(m, &host(&vec![0.0; d * d], &[d, d]));
    let mut outs = Vec::new();
    for step in 0..iters as u64 {
        let y = ok("linear", m.linear_forward(&x, &w));
        let n = ok("rms", m.rms_norm_forward(&y, &rw, 1e-6));
        let ce = ok("ce", m.cross_entropy_mean_backward(&n, &tgt, None));
        let (_, gw) = ok("linear bwd", m.linear_backward(&x, &w, &y));
        ok("adamw", m.adamw_step(&mut p, &gw, &mut m1, &mut m2, step, AdamWConfig::nanolab(1e-3, 0.1)));
        outs.push(y);
        outs.push(n);
        outs.push(ce);
    }
    let mut b: Vec<Vec<u32>> = outs.iter().map(bits).collect();
    b.push(bits(&p));
    b
}

#[test]
fn concurrent_threads_share_one_pending_fault_reported_exactly_once() {
    const THREADS: u64 = 6;
    const ITERS: usize = 5;
    let shared = Arc::new(metal());
    let want: Vec<_> = (0..THREADS).map(|i| clean_sequence(&shared, 100 * (i + 1), ITERS)).collect();
    assert_eq!(pending(&shared), None);
    let other = metal();
    let start = Arc::new(Barrier::new(THREADS as usize + 1));
    let mut handles = Vec::new();
    for i in 0..THREADS {
        let (m, start) = (Arc::clone(&shared), Arc::clone(&start));
        handles.push(thread::spawn(move || {
            start.wait();
            let got = clean_sequence(&m, 100 * (i + 1), ITERS);
            if i == 2 {
                let nan = up(&m, &host(&[f32::NAN, 1.0], &[2]));
                ok("silu records", m.silu_forward(&nan));
            }
            let reported = match m.sync() {
                Ok(()) => 0,
                Err(OjasError::NonFinite { op: "silu_forward" }) => 1,
                Err(e) => panic!("thread {i}: {e:?}"),
            };
            (got, reported)
        }));
    }
    start.wait();
    let mut reports = 0;
    for (i, h) in handles.into_iter().enumerate() {
        let (got, reported) = h.join().unwrap_or_else(|_| panic!("thread {i} panicked"));
        reports += reported;
        assert!(got == want[i], "thread {i}: bits differ from the serial run");
    }
    if pending(&shared) == Some("silu_forward") {
        reports += 1;
    }
    assert_eq!(reports, 1, "the fault must be reported exactly once");
    assert_eq!(pending(&other), None, "a second backend sees nothing");
}

/// N recorded ops do no waited commit until a sync point, which does one.
#[test]
fn recorded_ops_wait_only_at_sync_points() {
    let m = metal();
    let x = up(&m, &rand(&[64], 1, 1.0));
    let w0 = m.waits();
    let mut y = x.clone();
    for _ in 0..20 {
        y = ok("silu", m.silu_forward(&y));
    }
    assert_eq!(m.waits() - w0, 0, "recording must not wait");
    ok("sync", m.sync());
    assert_eq!(m.waits() - w0, 1, "sync waits once");
    ok("sync", m.sync());
    assert_eq!(m.waits() - w0, 1, "a sync with nothing recorded does not wait");
    // 170 optimizer steps, as adamw_full: no wait until the sync.
    let mut states: Vec<AdamState> = (0..170).map(|i| adam_state(&m, 1000 + i)).collect();
    let g = up(&m, &rand(&[5, 7], 7, 1.0));
    let w1 = m.waits();
    for s in &mut states {
        ok("adamw", m.adamw_step(&mut s.p, &g, &mut s.m1, &mut s.m2, 1, AdamWConfig::nanolab(1e-3, 0.1)));
    }
    assert_eq!(m.waits() - w1, 0, "170 optimizer steps must not wait");
    ok("sync", m.sync());
    assert_eq!(m.waits() - w1, 1);
    // A download waits once for its read and reports with no second wait.
    let w2 = m.waits();
    ok("silu", m.silu_forward(&x));
    ok("download", m.download(&y));
    assert_eq!(m.waits() - w2, 1, "download waits once");
}
