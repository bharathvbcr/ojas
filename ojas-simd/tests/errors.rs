//! Bad lengths and strides are refused with an error, C is left untouched, and
//! nothing panics.

mod common;

use common::available_backends;
use ojas_simd::{sgemm_tile_with, Operand, SimdError};

#[allow(clippy::too_many_arguments)]
fn call(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    c_len: usize,
    c_rs: usize,
) -> Result<(), SimdError> {
    let mut first = None;
    for bk in available_backends() {
        let mut c = vec![9.0f32; c_len];
        let r = sgemm_tile_with(
            bk, m, n, k, a, a_rs, a_cs, b, b_rs, b_cs, &mut c, c_rs, false,
        );
        if r.is_err() {
            assert!(
                c.iter().all(|&v| v == 9.0),
                "{bk:?} wrote C before refusing"
            );
        }
        match &first {
            None => first = Some(r),
            Some(f) => assert_eq!(f, &r, "{bk:?} disagrees"),
        }
    }
    first.expect("portable is always available")
}

#[test]
fn short_buffers() {
    let a = vec![1.0f32; 6];
    let b = vec![1.0f32; 12];
    // 2x4 = (2x3)(3x4), all row-major. Exact lengths pass.
    assert_eq!(call(2, 4, 3, &a, 3, 1, &b, 4, 1, 8, 4), Ok(()));
    assert_eq!(
        call(2, 4, 3, &a[..5], 3, 1, &b, 4, 1, 8, 4),
        Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: 6,
            len: 5
        })
    );
    assert_eq!(
        call(2, 4, 3, &a, 3, 1, &b[..11], 4, 1, 8, 4),
        Err(SimdError::BufferTooShort {
            operand: Operand::B,
            required: 12,
            len: 11
        })
    );
    assert_eq!(
        call(2, 4, 3, &a, 3, 1, &b, 4, 1, 7, 4),
        Err(SimdError::BufferTooShort {
            operand: Operand::C,
            required: 8,
            len: 7
        })
    );
    // A transposed stride that walks past the end.
    assert_eq!(
        call(2, 4, 3, &a, 1, 3, &b, 4, 1, 8, 4),
        Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: 8,
            len: 6
        })
    );
    // Empty slices with a nonempty problem.
    assert!(matches!(
        call(1, 1, 1, &[], 1, 1, &[1.0], 1, 1, 1, 1),
        Err(SimdError::BufferTooShort {
            operand: Operand::A,
            ..
        })
    ));
}

#[test]
fn overflowing_strides() {
    let a = vec![1.0f32; 16];
    let big = usize::MAX / 2 + 1;
    assert_eq!(
        call(2, 2, 2, &a, big, 1, &a, 2, 1, 4, 2),
        Err(SimdError::BufferTooShort {
            operand: Operand::A,
            required: big + 2,
            len: 16
        })
    );
    assert_eq!(
        call(3, 2, 2, &a, big, 1, &a, 2, 1, 4, 2),
        Err(SimdError::ExtentOverflow {
            operand: Operand::A
        })
    );
    assert_eq!(
        call(2, 2, 3, &a, 2, 1, &a, usize::MAX, 1, 4, 2),
        Err(SimdError::ExtentOverflow {
            operand: Operand::B
        })
    );
    assert_eq!(
        call(2, 2, 2, &a, 2, 1, &a, 2, usize::MAX, 4, 2),
        Err(SimdError::ExtentOverflow {
            operand: Operand::B
        })
    );
    assert_eq!(
        call(3, 2, 2, &a, 2, 1, &a, 2, 1, 4, usize::MAX),
        Err(SimdError::ExtentOverflow {
            operand: Operand::C
        })
    );
    assert_eq!(
        call(
            usize::MAX,
            usize::MAX,
            usize::MAX,
            &a,
            1,
            1,
            &a,
            1,
            1,
            4,
            usize::MAX
        ),
        Err(SimdError::ExtentOverflow {
            operand: Operand::A
        })
    );
}

#[test]
fn overlapping_output_rows() {
    let a = vec![1.0f32; 16];
    assert_eq!(
        call(2, 4, 2, &a, 2, 1, &a, 4, 1, 16, 3),
        Err(SimdError::OverlappingOutputRows { n: 4, c_rs: 3 })
    );
    assert_eq!(
        call(2, 4, 2, &a, 2, 1, &a, 4, 1, 16, 0),
        Err(SimdError::OverlappingOutputRows { n: 4, c_rs: 0 })
    );
    // One row: the row stride is never used.
    assert_eq!(call(1, 4, 2, &a, 0, 1, &a, 4, 1, 4, 0), Ok(()));
}

#[test]
fn errors_display() {
    let e = SimdError::BufferTooShort {
        operand: Operand::B,
        required: 3,
        len: 2,
    };
    assert_eq!(
        e.to_string(),
        "operand B needs 3 elements but the slice has 2"
    );
    let _: &dyn std::error::Error = &e;
}
