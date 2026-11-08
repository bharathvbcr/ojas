//! A mapped view must never upload anything other than its own window.
//!
//! `Worker::upload` adopts a mapped `SharedBuf`'s whole GPU buffer and applies
//! no view offset, so `MetalBackend::upload` refuses a mapped view that is not
//! the entire allocation (a non-zero byte offset or a shorter length) instead
//! of uploading the base's bytes. These tests use a real tessl `SharedBuf`.

#![cfg(all(target_os = "macos", feature = "metal"))]

use ojas_core::{Backend, BackendId, Budget, DType, OjasError, Tensor};
use std::sync::Arc;
use tessl::shared_buf::{SharedAlloc, SharedBuf};

fn shared(backend: &ojas_metal::MetalBackend, nbytes: usize) -> SharedBuf {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    backend
        .with_runtime(move |rt| {
            let a = SharedAlloc::new(rt, nbytes).map_err(|e| OjasError::Backend {
                id: BackendId::Metal,
                detail: e,
            })?;
            tx.send(a.into_shared_buf()).unwrap();
            Ok(())
        })
        .expect("with_runtime");
    rx.recv().unwrap()
}

fn fill_u32(s: &SharedBuf, v: &[u32]) {
    let mut w = s.write().unwrap();
    w.as_u32_mut().unwrap().copy_from_slice(v);
}

fn fill_f32(s: &SharedBuf, v: &[f32]) {
    let mut w = s.write().unwrap();
    w.as_f32_mut().unwrap().copy_from_slice(v);
}

fn assert_refused(r: Result<Tensor, OjasError>, what: &str) {
    match r {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, "MetalBackend::upload", "{what}");
            assert!(detail.contains("mapped view"), "{what}: {detail}");
        }
        Err(other) => panic!("{what}: wrong error {other:?}"),
        Ok(_) => panic!("{what}: a mapped view was uploaded; the device reads the base buffer"),
    }
}

/// Ids [0..8) with a view on the second half: the old path uploaded 0..4.
#[test]
fn a_narrowed_mapped_ids_view_is_refused_not_uploaded_from_the_base() {
    let budget = Budget::new(64 << 20);
    let be = ojas_metal::MetalBackend::new(budget.clone()).expect("Metal");
    let s = shared(&be, 32);
    fill_u32(&s, &(0..8).collect::<Vec<u32>>());
    let t = Tensor::from_mapped(Arc::new(s), &[8], DType::U32, &budget).unwrap();
    let view = t.narrow(16, &[4], &[1]).unwrap();
    assert_refused(be.upload(&view), "narrowed ids view");
}

/// A row view of a [4, 8] f32 mapped tensor: the old path uploaded row 0.
#[test]
fn a_mapped_row_view_is_refused_not_uploaded_from_the_base() {
    let budget = Budget::new(64 << 20);
    let be = ojas_metal::MetalBackend::new(budget.clone()).expect("Metal");
    let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let s = shared(&be, 128);
    fill_f32(&s, &vals);
    let t = Tensor::from_mapped(Arc::new(s), &[4, 8], DType::F32, &budget).unwrap();
    let row = t.narrow(64, &[8], &[1]).unwrap();
    assert_refused(be.upload(&row), "row view");
    // A shorter view at offset 0 is the same hazard: only part of the buffer.
    let head = t.narrow(0, &[8], &[1]).unwrap();
    assert_refused(be.upload(&head), "short view at offset 0");
}

/// Ids [99, 99, 99, 99, 4, 5, 6, 7], view on the valid half. The host range
/// check reads the window (valid ids) while the device would gather from the
/// base (the 99s): the upload must be refused so the two cannot disagree.
#[test]
fn an_ids_view_over_the_valid_half_cannot_reach_the_out_of_range_half() {
    let budget = Budget::new(64 << 20);
    let be = ojas_metal::MetalBackend::new(budget.clone()).expect("Metal");
    let s = shared(&be, 32);
    fill_u32(&s, &[99, 99, 99, 99, 4, 5, 6, 7]);
    let t = Tensor::from_mapped(Arc::new(s), &[8], DType::U32, &budget).unwrap();
    let valid = t.narrow(16, &[4], &[1]).unwrap();
    assert_refused(be.upload(&valid), "valid-half ids view");
}

/// The whole allocation still uploads, byte for byte.
#[test]
fn a_whole_mapped_buffer_still_uploads_its_own_data() {
    let budget = Budget::new(64 << 20);
    let be = ojas_metal::MetalBackend::new(budget.clone()).expect("Metal");
    let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let s = shared(&be, 128);
    fill_f32(&s, &vals);
    let t = Tensor::from_mapped(Arc::new(s), &[4, 8], DType::F32, &budget).unwrap();
    let up = be.upload(&t).expect("whole mapped buffer");
    be.sync().unwrap();
    assert_eq!(be.download(&up).unwrap().to_f32_vec().unwrap(), vals);
}
