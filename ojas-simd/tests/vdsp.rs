//! Stride-1 `vDSP_vmul` / `vDSP_vadd` against one rounding of `a * b` and `a + b`.
//! Built only with `--features accelerate` on macOS.
#![cfg(all(feature = "accelerate", target_os = "macos"))]

use ojas_simd::{
    vdsp_mmov, vdsp_mmov_append, vdsp_vadd, vdsp_vadd_append, vdsp_vmul, vdsp_vmul_append, Operand,
    SimdError,
};

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|v| v.to_bits()).collect()
}

fn series(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let bits = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(salt);
            let mag = (bits >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0;
            mag * (1.0 + (i % 5) as f32)
        })
        .collect()
}

fn assert_products(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut c = vec![7.0; a.len()];
    vdsp_vmul(a, b, &mut c).unwrap();
    let want: Vec<u32> = a.iter().zip(b).map(|(x, y)| (x * y).to_bits()).collect();
    assert_eq!(bits(&c), want, "vmul len {}", a.len());

    let mut dst = Vec::with_capacity(a.len());
    vdsp_vmul_append(a, b, &mut dst).unwrap();
    assert_eq!(bits(&dst), want, "vmul append len {}", a.len());
}

fn assert_sums(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut c = vec![7.0; a.len()];
    vdsp_vadd(a, b, &mut c).unwrap();
    let want: Vec<u32> = a.iter().zip(b).map(|(x, y)| (x + y).to_bits()).collect();
    assert_eq!(bits(&c), want, "vadd len {}", a.len());

    let mut dst = vec![1.5];
    dst.reserve(a.len());
    vdsp_vadd_append(a, b, &mut dst).unwrap();
    assert_eq!(dst[0].to_bits(), 1.5f32.to_bits());
    assert_eq!(bits(&dst[1..]), want, "vadd append len {}", a.len());
}

#[test]
fn finite_values_match_one_rounding_including_the_tail() {
    let lengths = [
        1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 4095, 4096, 4097,
    ];
    for n in lengths {
        let a = series(n, 1);
        let b = series(n, 2);
        assert_products(&a, &b);
        assert_sums(&a, &b);
        // An unaligned tail: skip the first element.
        if n > 1 {
            assert_products(&a[1..], &b[1..]);
            assert_sums(&a[1..], &b[1..]);
        }
    }
}

#[test]
fn signed_zero_and_subnormals_match_the_scalar_formula() {
    let a = [
        -0.0,
        0.0,
        -0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        -2.5,
        4.0,
    ];
    let b = [-0.0, -0.0, 0.0, 1.0, -0.0, 1.0, 2.0, 2.0, 3.0, -0.5];
    assert_products(&a, &b);
    assert_sums(&a, &b);
}

#[test]
fn finite_overflow_matches_scalar_infinity() {
    let mul_a = [1.0e30f32, -1.0e30, 2.0e38];
    let mul_b = [1.0e30f32, 1.0e30, 2.0e38];
    assert_products(&mul_a, &mul_b);
    let add_a = [2.0e38f32, -2.0e38, 1.0];
    let add_b = [2.0e38f32, -2.0e38, -1.0];
    assert_sums(&add_a, &add_b);
}

#[test]
fn nonfinite_inputs_propagate() {
    let mut c = [0.0f32];
    vdsp_vadd(&[f32::NAN], &[1.0], &mut c).unwrap();
    assert!(c[0].is_nan());
    vdsp_vmul(&[1.0], &[f32::INFINITY], &mut c).unwrap();
    assert!(c[0].is_infinite() && c[0].is_sign_positive());
    vdsp_vadd(&[f32::NEG_INFINITY], &[0.0], &mut c).unwrap();
    assert!(c[0].is_infinite() && c[0].is_sign_negative());
}

#[test]
fn empty_does_not_write() {
    let mut c = [0.0f32; 0];
    vdsp_vmul(&[], &[], &mut c).unwrap();
    vdsp_vadd(&[], &[], &mut c).unwrap();
    let mut dst = vec![3.0];
    vdsp_vmul_append(&[], &[], &mut dst).unwrap();
    vdsp_vadd_append(&[], &[], &mut dst).unwrap();
    assert_eq!(dst, vec![3.0]);
}

#[test]
fn a_length_mismatch_or_short_output_does_not_write() {
    let mut c = [7.0f32, 7.0];
    let err = vdsp_vmul(&[1.0], &[2.0, 3.0], &mut c).unwrap_err();
    assert_eq!(err, SimdError::MismatchedLengths { a: 1, b: 2 });
    assert_eq!(c, [7.0, 7.0]);

    let err = vdsp_vadd(&[1.0, 2.0], &[3.0, 4.0], &mut c[..1]).unwrap_err();
    assert_eq!(
        err,
        SimdError::OutputLength {
            output: 1,
            expected: 2
        }
    );
    assert_eq!(c, [7.0, 7.0]);

    let mut dst = vec![9.0f32];
    let spare = dst.capacity() - dst.len();
    let n = spare + 1;
    let ones = vec![1.0f32; n];
    let err = vdsp_vadd_append(&ones, &ones, &mut dst).unwrap_err();
    assert_eq!(
        err,
        SimdError::OutputLength {
            output: spare,
            expected: n
        }
    );
    assert_eq!(dst, vec![9.0]);

    let err = vdsp_vmul_append(&[1.0], &[2.0, 3.0], &mut dst).unwrap_err();
    assert_eq!(err, SimdError::MismatchedLengths { a: 1, b: 2 });
    assert_eq!(dst, vec![9.0]);
}

#[test]
fn an_overlapping_output_is_refused_without_a_write() {
    let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ptr = buf.as_mut_ptr();
    // Alias on purpose: the wrapper must refuse before vDSP reads or writes.
    let err = unsafe {
        let a = std::slice::from_raw_parts(ptr, 4);
        let b = std::slice::from_raw_parts(ptr, 4);
        let c = std::slice::from_raw_parts_mut(ptr.add(2), 4);
        vdsp_vmul(a, b, c).unwrap_err()
    };
    assert_eq!(err, SimdError::OverlappingOutput);
    assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let err = unsafe {
        let a = std::slice::from_raw_parts(ptr, 4);
        let b = std::slice::from_raw_parts(ptr.add(1), 4);
        let c = std::slice::from_raw_parts_mut(ptr.add(1), 4);
        vdsp_vadd(a, b, c).unwrap_err()
    };
    assert_eq!(err, SimdError::OverlappingOutput);
    assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Scalar `C[n][m] = A[n][m]`. `__M` is columns and `__N` is rows.
fn mmov_ref(
    src: &[f32],
    rows: usize,
    cols: usize,
    src_stride: usize,
    dst_stride: usize,
) -> Vec<f32> {
    let span = if rows == 0 || cols == 0 {
        0
    } else {
        (rows - 1) * dst_stride + cols
    };
    let mut dst = vec![f32::from_bits(0x7fc0_0001); span];
    for n in 0..rows {
        for m in 0..cols {
            dst[n * dst_stride + m] = src[n * src_stride + m];
        }
    }
    dst
}

#[test]
fn mmov_moves_bits_including_signed_zero_and_the_column_order() {
    // 3 rows of 2 columns, pitch 5. Swapping `__M` and `__N` does not produce this.
    let mut src = [0.0f32; 12];
    for (i, slot) in src.iter_mut().enumerate() {
        *slot = i as f32;
    }
    src[0] = -0.0;
    src[5] = f32::from_bits(1);
    src[10] = -0.0;
    let mut dst = [7.0f32; 6];
    vdsp_mmov(&src, &mut dst, 3, 2, 5, 2).unwrap();
    assert_eq!(
        bits(&dst),
        bits(&mmov_ref(&src, 3, 2, 5, 2)),
        "column count is __M, row count is __N"
    );
    assert_eq!(dst[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(dst[2].to_bits(), 1);

    // Nanolab head: 4 rows of 64, source pitch 768, packed destination.
    let rows = 4usize;
    let cols = 64usize;
    let src_stride = 768usize;
    let span = (rows - 1) * src_stride + cols;
    let mut wide = vec![f32::from_bits(0x7fc0_0001); span];
    for n in 0..rows {
        for m in 0..cols {
            let bits = ((n * cols + m) as u32).wrapping_mul(0x9E37_79B1);
            wide[n * src_stride + m] = f32::from_bits(bits);
        }
    }
    wide[0] = -0.0;
    wide[src_stride + 3] = -0.0;
    wide[2 * src_stride + 63] = f32::from_bits(1);
    wide[3 * src_stride + 63] = f32::from_bits((-0.0f32).to_bits());
    // A gap column must not be copied.
    wide[64] = f32::from_bits(0x7f80_0001);
    let mut packed = vec![9.0f32; rows * cols];
    vdsp_mmov(&wide, &mut packed, rows, cols, src_stride, cols).unwrap();
    assert_eq!(
        bits(&packed),
        bits(&mmov_ref(&wide, rows, cols, src_stride, cols))
    );
    assert_eq!(packed[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(packed[cols + 3].to_bits(), (-0.0f32).to_bits());
    assert_eq!(packed[2 * cols + 63].to_bits(), 1);
    assert!(packed.iter().all(|v| v.to_bits() != 0x7f80_0001));

    let mut dst = vec![1.5f32];
    dst.reserve(rows * cols);
    vdsp_mmov_append(&wide, &mut dst, rows, cols, src_stride).unwrap();
    assert_eq!(dst[0].to_bits(), 1.5f32.to_bits());
    assert_eq!(bits(&dst[1..]), bits(&packed));
}

#[test]
fn mmov_empty_does_not_write_and_a_short_or_overlapping_buffer_is_refused() {
    let mut dst = [7.0f32; 4];
    vdsp_mmov(&[1.0, 2.0, 3.0, 4.0], &mut dst, 0, 2, 2, 2).unwrap();
    vdsp_mmov(&[1.0, 2.0, 3.0, 4.0], &mut dst, 2, 0, 2, 2).unwrap();
    assert_eq!(dst, [7.0, 7.0, 7.0, 7.0]);
    let mut acc = vec![3.0f32];
    vdsp_mmov_append(&[1.0, 2.0], &mut acc, 0, 2, 2).unwrap();
    assert_eq!(acc, vec![3.0]);

    let src = [0.0f32; 8];
    let err = vdsp_mmov(&src, &mut dst, 2, 4, 2, 4).unwrap_err();
    assert_eq!(err, SimdError::OverlappingOutputRows { n: 4, c_rs: 2 });
    assert_eq!(dst, [7.0, 7.0, 7.0, 7.0]);

    let err = vdsp_mmov(&src[..3], &mut [0.0f32; 4], 2, 2, 2, 2).unwrap_err();
    assert_eq!(
        err,
        SimdError::BufferTooShort {
            operand: Operand::A,
            required: 4,
            len: 3
        }
    );

    let mut buf = vec![1.0f32, 2.0, 3.0, 4.0];
    let err = unsafe {
        let ptr = buf.as_mut_ptr();
        let src = std::slice::from_raw_parts(ptr, 4);
        let dst = std::slice::from_raw_parts_mut(ptr, 4);
        vdsp_mmov(src, dst, 2, 2, 2, 2).unwrap_err()
    };
    assert_eq!(err, SimdError::OverlappingOutput);
    assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0]);
}
