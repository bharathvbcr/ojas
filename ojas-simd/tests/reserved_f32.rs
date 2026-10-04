//! `ReservedF32` publishes a vector only after every lane of every chunk is stored.

use std::mem::MaybeUninit;

use ojas_simd::{ReservedF32, SimdError};

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|v| v.to_bits()).collect()
}

fn store_all(dst: &mut [MaybeUninit<f32>], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (slot, &v) in dst.iter_mut().zip(src) {
        slot.write(v);
    }
}

#[test]
fn a_one_element_tail_is_stored_and_negative_zero_keeps_its_sign() {
    let src = [1.0f32, -2.0, 0.0, -0.0, 3.5];
    let buf = ReservedF32::try_new(src.len(), &[(0, 4), (4, 1)]).unwrap();
    buf.write_chunk(0, |dst| store_all(dst, &src[..4])).unwrap();
    buf.write_chunk(1, |dst| {
        assert_eq!(dst.len(), 1);
        store_all(dst, &src[4..]);
    })
    .unwrap();
    let got = buf.into_vec().unwrap();
    assert_eq!(bits(&got), bits(&src));
    assert_eq!(got[3].to_bits(), (-0.0f32).to_bits());
    assert_eq!(got[4].to_bits(), 3.5f32.to_bits());
}

#[test]
fn a_missing_chunk_is_not_published() {
    let buf = ReservedF32::try_new(4, &[(0, 3), (3, 1)]).unwrap();
    buf.write_chunk(0, |dst| store_all(dst, &[1.0, 2.0, 3.0]))
        .unwrap();
    assert!(matches!(
        buf.into_vec(),
        Err(SimdError::Incomplete {
            done: 1,
            expected: 2
        })
    ));
}

#[test]
fn a_gap_in_the_chunks_refuses_before_a_write() {
    assert!(matches!(
        ReservedF32::try_new(4, &[(0, 2)]),
        Err(SimdError::OutputLength { .. })
    ));
    assert!(matches!(
        ReservedF32::try_new(4, &[(2, 2), (0, 2)]),
        Err(SimdError::OutputLength { .. })
    ));
}

#[test]
fn a_second_write_does_not_replace_the_first() {
    let buf = ReservedF32::try_new(2, &[(0, 2)]).unwrap();
    buf.write_chunk(0, |dst| store_all(dst, &[-0.0, 1.0]))
        .unwrap();
    let mut called = false;
    let err = buf
        .write_chunk(0, |_| {
            called = true;
        })
        .unwrap_err();
    assert_eq!(err, SimdError::OverlappingOutput);
    assert!(!called);
    assert_eq!(bits(&buf.into_vec().unwrap()), bits(&[-0.0, 1.0]));
}

#[test]
fn an_index_past_the_chunks_does_not_run_the_write() {
    let buf = ReservedF32::try_new(1, &[(0, 1)]).unwrap();
    let mut called = false;
    let err = buf
        .write_chunk(1, |_| {
            called = true;
        })
        .unwrap_err();
    assert!(matches!(err, SimdError::OutputLength { .. }), "{err:?}");
    assert!(!called);
    assert!(buf.into_vec().is_err());
}

#[test]
fn empty_is_an_empty_vector() {
    let buf = ReservedF32::try_new(0, &[]).unwrap();
    assert!(buf.into_vec().unwrap().is_empty());
    let buf = ReservedF32::try_new(0, &[(0, 0)]).unwrap();
    buf.write_chunk(0, |dst| assert!(dst.is_empty())).unwrap();
    assert!(buf.into_vec().unwrap().is_empty());
}

#[test]
fn dropping_before_publish_does_not_read_the_lanes() {
    let buf = ReservedF32::try_new(8, &[(0, 7), (7, 1)]).unwrap();
    buf.write_chunk(0, |dst| store_all(dst, &[1.0; 7])).unwrap();
    drop(buf);
}

#[test]
fn two_threads_store_disjoint_chunks() {
    let buf = ReservedF32::try_new(1001, &[(0, 1000), (1000, 1)]).unwrap();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            buf.write_chunk(0, |dst| {
                for (i, slot) in dst.iter_mut().enumerate() {
                    slot.write(i as f32);
                }
            })
            .unwrap();
        });
        scope.spawn(|| {
            buf.write_chunk(1, |dst| {
                assert_eq!(dst.len(), 1);
                dst[0].write(-0.0);
            })
            .unwrap();
        });
    });
    let got = buf.into_vec().unwrap();
    assert_eq!(got.len(), 1001);
    assert_eq!(got[999].to_bits(), 999.0f32.to_bits());
    assert_eq!(got[1000].to_bits(), (-0.0f32).to_bits());
}
