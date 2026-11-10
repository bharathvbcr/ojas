//! `quantize_affine` / `dequantize_affine_bf16` against mlx 0.31.2 golden arrays (`mx.gpu`).
//!
//! The goldens (`tests/fixtures/mlx_affine/`, made by `ojas-oracle/python/mlx_affine_goldens.py`)
//! hold, per case, the bf16 input and mlx's packed codes, bf16 scales, bf16 biases and bf16
//! dequantised values for q8 and q4. Every array must match bit for bit. The cases cover the
//! in-feature widths of Qwen3.5-2B-Base's wrapped linears (2048 and 6144) with random weights
//! and a mix of every adversarial group kind (ties, all-negative, constant, zero, subnormals,
//! outliers, bf16-max magnitudes), and a width-64 case with one named group kind per row.

#[allow(dead_code)]
mod mlx_affine_common;

use mlx_affine_common::{load, Golden, CASES};
use ojas_core::{
    bf16_to_f32, dequantize_affine_bf16, quantize_affine, AffineQuantized, AffineSpec, Budget,
    Tensor,
};

fn budget() -> Budget {
    Budget::new(1 << 32)
}

fn spec_for(bits: u32) -> AffineSpec {
    AffineSpec::new(bits, 64).unwrap()
}

fn bf16_tensor(shape: &[usize], bits: &[u16], budget: &Budget) -> Tensor {
    Tensor::from_bf16_bits(bits, shape, budget).unwrap()
}

/// `(index of the first difference, count of differences)` between two equal-length slices.
fn diff<T: PartialEq + Copy>(got: &[T], want: &[T]) -> Option<(usize, usize)> {
    assert_eq!(got.len(), want.len(), "length");
    let mut first = None;
    let mut n = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            first.get_or_insert(i);
            n += 1;
        }
    }
    first.map(|i| (i, n))
}

fn expect_same<T: PartialEq + Copy + std::fmt::LowerHex>(what: &str, got: &[T], want: &[T]) {
    if let Some((i, n)) = diff(got, want) {
        panic!(
            "{what}: {n} of {} differ, first at {i}: got {:#x}, mlx {:#x}",
            got.len(),
            got[i],
            want[i]
        );
    }
}

fn check_against_golden(case: &str, golden: &Golden, bits: u32, quantized: &AffineQuantized) {
    let tag = format!("{case} q{bits}");
    let (_, want_packed) = golden.u32s(&format!("q{bits}.wq"));
    let (_, want_scales) = golden.bf16(&format!("q{bits}.scales"));
    let (_, want_biases) = golden.bf16(&format!("q{bits}.biases"));
    let (_, want_deq) = golden.bf16(&format!("q{bits}.deq"));
    expect_same(
        &format!("{tag} packed codes"),
        quantized.packed().u32_slice().unwrap(),
        want_packed,
    );
    expect_same(
        &format!("{tag} scales"),
        quantized.scales().bf16_slice().unwrap(),
        want_scales,
    );
    expect_same(
        &format!("{tag} biases"),
        quantized.biases().bf16_slice().unwrap(),
        want_biases,
    );
    let deq = dequantize_affine_bf16(quantized, &budget()).unwrap();
    assert_eq!(deq.shape(), &[quantized.rows(), quantized.cols()][..]);
    expect_same(
        &format!("{tag} dequantised"),
        deq.bf16_slice().unwrap(),
        want_deq,
    );
}

fn quantize_goldens(bits: u32) {
    let b = budget();
    for case in CASES {
        let golden = load(case);
        let (shape, w) = golden.bf16("w");
        let q = quantize_affine(&bf16_tensor(shape, w, &b), spec_for(bits), &b)
            .unwrap_or_else(|e| panic!("{case} q{bits}: {e}"));
        assert_eq!((q.rows(), q.cols()), (shape[0], shape[1]));
        check_against_golden(case, &golden, bits, &q);
    }
}

#[test]
fn q8_quantiser_and_dequantiser_match_mlx_gpu_bit_for_bit() {
    quantize_goldens(8);
}

#[test]
fn q4_quantiser_and_dequantiser_match_mlx_gpu_bit_for_bit() {
    quantize_goldens(4);
}

/// An `F32` weight whose values are all bf16-exact is the same input as the bf16 tensor.
#[test]
fn f32_input_that_is_bf16_exact_quantises_identically() {
    let b = budget();
    for case in ["kinds_w64", "mixed_w2048"] {
        let golden = load(case);
        let (shape, w) = golden.bf16("w");
        let wide: Vec<f32> = w.iter().map(|&x| bf16_to_f32(x)).collect();
        let f32_weight = Tensor::from_f32(&wide, shape, &b).unwrap();
        for bits in [8, 4] {
            let q = quantize_affine(&f32_weight, spec_for(bits), &b).unwrap();
            check_against_golden(case, &golden, bits, &q);
        }
    }
}

/// The dequantiser alone, on mlx's own packed codes, scales and biases (so a quantiser bug cannot
/// hide a dequantiser bug, or the other way round).
#[test]
fn dequantiser_reproduces_mlx_from_mlx_parts() {
    let b = budget();
    let mut cases = CASES.to_vec();
    cases.push("deq_params_w64");
    for case in cases {
        let golden = load(case);
        for bits in [8u32, 4] {
            let (wq_shape, wq) = golden.u32s(&format!("q{bits}.wq"));
            let (sb_shape, scales) = golden.bf16(&format!("q{bits}.scales"));
            let (_, biases) = golden.bf16(&format!("q{bits}.biases"));
            let (_, want) = golden.bf16(&format!("q{bits}.deq"));
            let cols = sb_shape[1] * 64;
            let parts = AffineQuantized::from_parts(
                spec_for(bits),
                wq_shape[0],
                cols,
                Tensor::from_u32(wq, wq_shape, &b).unwrap(),
                bf16_tensor(sb_shape, scales, &b),
                bf16_tensor(sb_shape, biases, &b),
            )
            .unwrap_or_else(|e| panic!("{case} q{bits}: {e}"));
            let deq = dequantize_affine_bf16(&parts, &b).unwrap();
            expect_same(
                &format!("{case} q{bits} deq"),
                deq.bf16_slice().unwrap(),
                want,
            );
        }
    }
}

/// The goldens must exercise what the quantiser's decisions are about, or a quantiser that
/// ignored them would still pass: the subnormal rows (flushed side test), the bf16-max constant
/// group (NaN code) are in the data. (The .5-code ties are present too, 380 in q8 `kinds_w64`, but
/// this test does not assert them.)
#[test]
fn goldens_exercise_the_flush_nan_and_tie_decisions() {
    let golden = load("kinds_w64");
    let (shape, w) = golden.bf16("w");
    assert_eq!(shape[1], 64);
    let rows = shape[0];
    let subnormal_rows = (0..rows)
        .filter(|&r| {
            w[r * 64..(r + 1) * 64]
                .iter()
                .any(|&x| x & 0x7FFF != 0 && x & 0x7F80 == 0)
        })
        .count();
    assert!(
        subnormal_rows >= 8,
        "{subnormal_rows} rows hold a bf16 subnormal"
    );
    // A group of zeros holding one negative subnormal: |w_min| == |w_max| == 0 once flushed, so the
    // stored scale is negative. Unflushed it would be positive.
    let (_, scales) = golden.bf16("q8.scales");
    let negative_scale_with_subnormal = (0..rows).any(|r| {
        let row = &w[r * 64..(r + 1) * 64];
        let neg_sub = row
            .iter()
            .any(|&x| x & 0x8000 != 0 && x & 0x7F80 == 0 && x & 0x7FFF != 0);
        let all_sub_or_zero = row.iter().all(|&x| x & 0x7F80 == 0);
        neg_sub && all_sub_or_zero && scales[r] & 0x8000 != 0
    });
    assert!(negative_scale_with_subnormal);
    let (_, wq) = golden.u32s("q8.wq");
    let all_255 = (0..rows).any(|r| wq[r * 16..(r + 1) * 16].iter().all(|&x| x == 0xFFFF_FFFF));
    assert!(
        all_255,
        "the bf16-max constant group must come out all 255 (NaN code)"
    );
}
