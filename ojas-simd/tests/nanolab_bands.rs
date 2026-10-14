//! `with_nanolab_token_bands` commits only what both bands stored, and a band
//! that outlives the call stores nothing.
#![cfg(all(target_arch = "aarch64", target_feature = "neon"))]

use std::panic::{catch_unwind, AssertUnwindSafe};

use ojas_simd::{
    split_nanolab_head_pairs_append, with_nanolab_token_bands, NanolabWorkerBand, SimdError,
};

const N: usize = 1024 * 768;

fn source() -> Vec<f32> {
    (0..N)
        .map(|i| f32::from_bits((i as u32).wrapping_mul(2_654_435_761) | 1))
        .collect()
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|v| v.to_bits()).collect()
}

fn reference(src: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(N);
    split_nanolab_head_pairs_append(src, &mut out).unwrap();
    out
}

/// Both bands written, the worker on another thread.
fn split(src: &[f32], caller_tokens: usize) -> Vec<f32> {
    let mut dst = Vec::with_capacity(N);
    let done = with_nanolab_token_bands(src, &mut dst, caller_tokens, |caller, worker| {
        let handle = std::thread::spawn(move || worker.write());
        let written = caller.write();
        assert!(handle.join().unwrap());
        Ok::<_, ()>(Some(written))
    })
    .unwrap();
    assert_eq!(done, Ok(true));
    dst
}

#[test]
fn every_split_point_matches_the_one_call_path() {
    let src = source();
    let want = bits(&reference(&src));
    for caller_tokens in [1, 2, 640, 1022, 1023] {
        let got = split(&src, caller_tokens);
        assert_eq!(got.len(), N, "caller_tokens {caller_tokens}");
        assert_eq!(bits(&got), want, "caller_tokens {caller_tokens}");
    }
}

#[test]
fn zero_and_the_whole_batch_are_refused_before_body_runs() {
    let src = source();
    for caller_tokens in [0, 1024, 5000] {
        let mut dst = Vec::with_capacity(N);
        let mut ran = false;
        let err = with_nanolab_token_bands(&src, &mut dst, caller_tokens, |_, _| {
            ran = true;
            Ok::<_, ()>(None)
        })
        .unwrap_err();
        assert_eq!(err, SimdError::LengthTooLarge { len: caller_tokens });
        assert!(!ran);
        assert!(dst.is_empty());
    }
}

#[test]
fn declining_leaves_the_destination_alone() {
    let src = source();
    let mut dst = vec![7.0f32];
    dst.reserve_exact(N);
    let done = with_nanolab_token_bands(&src, &mut dst, 640, |_, _| Ok::<_, ()>(None)).unwrap();
    assert_eq!(done, Ok(false));
    assert_eq!(dst, [7.0]);
}

#[test]
fn an_error_from_body_is_returned_and_nothing_is_committed() {
    let src = source();
    let mut dst = Vec::with_capacity(N);
    let done = with_nanolab_token_bands(&src, &mut dst, 640, |caller, _worker| {
        caller.write();
        Err("no worker")
    })
    .unwrap();
    assert_eq!(done, Err("no worker"));
    assert!(dst.is_empty());
}

#[test]
fn a_forgotten_worker_band_is_not_committed_and_does_not_block() {
    let src = source();
    let mut dst = Vec::with_capacity(N);
    let err = with_nanolab_token_bands(&src, &mut dst, 640, |caller, worker| {
        std::mem::forget(worker);
        Ok::<_, ()>(Some(caller.write()))
    })
    .unwrap_err();
    assert_eq!(
        err,
        SimdError::Incomplete {
            done: 1,
            expected: 2
        }
    );
    assert!(dst.is_empty());
}

#[test]
fn a_dropped_worker_band_is_not_committed() {
    let src = source();
    let mut dst = Vec::with_capacity(N);
    let err = with_nanolab_token_bands(&src, &mut dst, 640, |caller, worker| {
        drop(worker);
        Ok::<_, ()>(Some(caller.write()))
    })
    .unwrap_err();
    assert!(matches!(err, SimdError::Incomplete { .. }));
    assert!(dst.is_empty());
}

#[test]
fn a_worker_band_kept_past_the_call_stores_nothing() {
    let src = source();
    let mut dst: Vec<f32> = Vec::with_capacity(N);
    let mut kept: Option<NanolabWorkerBand> = None;
    let done = with_nanolab_token_bands(&src, &mut dst, 640, |_caller, worker| {
        kept = Some(worker);
        Ok::<_, ()>(None)
    })
    .unwrap();
    assert_eq!(done, Ok(false));
    // The buffer is gone before the late write. If the band still stored
    // through its pointer this would be a use after free (Miri reports it).
    drop(dst);
    let late = kept.take().unwrap();
    assert!(!std::thread::spawn(move || late.write()).join().unwrap());
}

#[test]
fn a_late_write_does_not_touch_a_reallocated_destination() {
    let src = source();
    let mut dst: Vec<f32> = Vec::with_capacity(N);
    let mut kept = None;
    with_nanolab_token_bands(&src, &mut dst, 1023, |_c, worker| {
        kept = Some(worker);
        Ok::<_, ()>(None)
    })
    .unwrap()
    .unwrap();
    dst.extend(std::iter::repeat_n(3.0, N));
    dst.shrink_to_fit();
    dst.extend(std::iter::repeat_n(3.0, N));
    assert!(!kept.take().unwrap().write());
    assert!(dst.iter().all(|&v| v == 3.0));
}

#[test]
fn a_panic_in_body_after_spawning_the_worker_waits_and_commits_nothing() {
    let src = source();
    for _ in 0..8 {
        let mut dst: Vec<f32> = Vec::with_capacity(N);
        let mut handle = None;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            with_nanolab_token_bands(&src, &mut dst, 640, |_caller, worker| {
                handle = Some(std::thread::spawn(move || worker.write()));
                if handle.is_some() {
                    panic!("body fails after the handoff");
                }
                Ok::<_, ()>(None)
            })
        }));
        assert!(outcome.is_err());
        assert!(dst.is_empty());
        // The scope ended the band on unwind: the worker either finished
        // its store before the scope returned or never started.
        drop(dst);
        let _ran: bool = handle.take().unwrap().join().unwrap();
    }
}

#[test]
fn a_worker_write_racing_the_end_of_the_call_is_waited_for() {
    let src = source();
    for _ in 0..16 {
        let mut dst: Vec<f32> = Vec::with_capacity(N);
        let mut handle = None;
        let done = with_nanolab_token_bands(&src, &mut dst, 1, |caller, worker| {
            handle = Some(std::thread::spawn(move || worker.write()));
            Ok::<_, ()>(Some(caller.write()))
        });
        let ran = handle.take().unwrap().join().unwrap();
        match done {
            Ok(Ok(true)) => {
                assert!(ran);
                assert_eq!(bits(&dst), bits(&reference(&src)));
            }
            Err(SimdError::Incomplete { .. }) => {
                assert!(!ran);
                assert!(dst.is_empty());
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn a_short_destination_is_refused_before_body_runs() {
    let src = source();
    let mut dst = Vec::<f32>::with_capacity(16);
    let err = with_nanolab_token_bands(&src, &mut dst, 640, |_, _| Ok::<_, ()>(None)).unwrap_err();
    assert!(matches!(err, SimdError::OutputLength { .. }));
}
