//! Tier 5: Adversarial Coverage Hardening (CPU Core & Oracle)
//!
//! White-box coverage audit suite exercising:
//! 1. Vision Spatial Ops (max_pool2d, conv2d, conv_transpose2d, interpolate_bilinear, rope_interleaved)
//! 2. Gemma Embedding Ops (gelu_tanh, seq_mean_forward, l2_normalize_forward, rms_qkv_norm_forward)
//! 3. GroupNorm, ChannelLayerNorm, Slice, Concat, and Symmetric Sliding Window Attention
//! 4. Subnormals, negative parameters, zero-length dimensions, non-finite values, and IEEE 754 edge cases
//! 5. Multi-threaded Exact numerics bit-identity contracts

#![allow(
    clippy::identity_op,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::redundant_pattern_matching
)]

use ojas_core::{
    axial_rope_2d_tables, channel_layer_norm_forward_dims, concat_dims, conv2d_backward_dims,
    conv2d_forward_dims, conv_transpose2d_backward_dims, conv_transpose2d_forward_dims,
    gelu_tanh_backward_exact, gelu_tanh_exact, group_norm_forward_dims,
    interpolate_bilinear_backward_dims, interpolate_bilinear_forward_dims,
    l2_normalize_forward_dims, max_pool2d_backward_dims, max_pool2d_forward_dims,
    rms_qkv_norm_forward_dims, rope_interleaved_forward_dims, seq_mean_forward_dims,
    slice_backward_dims, slice_dims, AttentionOpts, Backend, Budget, Conv2dParams,
    ConvTranspose2dParams, Numerics, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

fn cpu_exact(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(2 << 30), threads)
        .expect("CpuBackend::with_threads")
        .with_numerics(Numerics::Exact)
}

fn host(be: &CpuBackend, t: &Tensor) -> Vec<f32> {
    be.download(t)
        .expect("download")
        .to_f32_vec()
        .expect("to_f32_vec")
}

fn host_bits(be: &CpuBackend, t: &Tensor) -> Vec<u32> {
    host(be, t).into_iter().map(f32::to_bits).collect()
}

fn upload(be: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    let t = Tensor::from_f32(data, shape, be.budget()).expect("from_f32");
    be.upload(&t).expect("upload")
}

fn upload_u32(be: &CpuBackend, data: &[u32], shape: &[usize]) -> Tensor {
    let t = Tensor::from_u32(data, shape, be.budget()).expect("from_u32");
    be.upload(&t).expect("upload")
}

// =============================================================================
// SECTION 1: MAX POOLING (max_pool2d) EDGE CASES & TIE-BREAKING
// =============================================================================

#[test]
fn tier5_cpu_01_max_pool2d_negative_elements_and_first_tie_break() {
    let be = cpu_exact(1);
    // 1 image, 1 channel, 2x2 spatial. All negative elements with tie!
    // [-10.0, -5.0]
    // [-5.0, -20.0]
    // The maximum is -5.0. It occurs twice: at (0, 1) and (1, 0).
    // The tie-break contract requires the first in row-major order: index (0, 1) = flat index 1!
    let data = vec![-10.0f32, -5.0, -5.0, -20.0];
    let x = upload(&be, &data, &[1, 1, 2, 2]);

    let pooled = be.max_pool2d_forward(&x, [2, 2], [2, 2]).expect("forward");
    let y = host(&be, &pooled);
    assert_eq!(y.len(), 1);
    assert_eq!(
        y[0], -5.0,
        "max of negative numbers must be -5.0, not clamped to 0"
    );

    // Backward: seed gradient 1.0. Exactly the first maximum at index 1 must receive 1.0; index 2 must receive 0.0!
    let grad_out = upload(&be, &[1.0f32], &[1, 1, 1, 1]);
    let grad_in = be
        .max_pool2d_backward(&x, &grad_out, [2, 2], [2, 2])
        .expect("backward");
    let gx = host(&be, &grad_in);
    assert_eq!(
        gx,
        vec![0.0, 1.0, 0.0, 0.0],
        "first occurrence in row-major order must receive 100% of gradient"
    );
}

#[test]
fn tier5_cpu_02_max_pool2d_overlapping_windows_gradient_accumulation() {
    let be = cpu_exact(2);
    // 3x3 input with kernel [2, 2] and stride [1, 1] -> output is 2x2.
    // Place a dominant maximum 100.0 right in the center (1, 1).
    // The center is covered by ALL four 2x2 pooling windows:
    // Window (0, 0), Window (0, 1), Window (1, 0), Window (1, 1).
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0, 6.0, 7.0, 8.0, 9.0];
    let x = upload(&be, &data, &[1, 1, 3, 3]);
    let pooled = be.max_pool2d_forward(&x, [2, 2], [1, 1]).expect("forward");
    let y = host(&be, &pooled);
    assert_eq!(y, vec![100.0, 100.0, 100.0, 100.0]);

    // Backward with grads [1.0, 2.0, 3.0, 4.0]:
    // Center element must accumulate the sum of ALL four covering windows = 1 + 2 + 3 + 4 = 10.0!
    let grad_out = upload(&be, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let grad_in = be
        .max_pool2d_backward(&x, &grad_out, [2, 2], [1, 1])
        .expect("backward");
    let gx = host(&be, &grad_in);
    assert_eq!(
        gx[4], 10.0,
        "center element must accumulate gradients from all overlapping windows"
    );
    assert_eq!(
        gx.iter().filter(|&&v| v > 0.0).count(),
        1,
        "only center element should have gradient"
    );
}

#[test]
fn tier5_cpu_03_max_pool2d_validator_rejection_boundaries() {
    let be = cpu_exact(1);
    let x = upload(&be, &[1.0; 16], &[1, 1, 4, 4]);

    // 1. Kernel larger than image
    assert!(matches!(
        max_pool2d_forward_dims(&x, [5, 4], [1, 1]),
        Err(OjasError::Shape { .. })
    ));
    // 2. Zero kernel
    assert!(matches!(
        max_pool2d_forward_dims(&x, [0, 4], [1, 1]),
        Err(OjasError::Shape { .. })
    ));
    // 3. Zero stride
    assert!(matches!(
        max_pool2d_forward_dims(&x, [2, 2], [0, 1]),
        Err(OjasError::Shape { .. })
    ));
    // 4. Rank != 4
    let x_rank3 = upload(&be, &[1.0; 16], &[1, 4, 4]);
    assert!(matches!(
        max_pool2d_forward_dims(&x_rank3, [2, 2], [1, 1]),
        Err(OjasError::Shape { .. })
    ));
    // 5. Backward grad_output shape mismatch
    let bad_grad = upload(&be, &[1.0; 4], &[1, 1, 2, 2]);
    assert!(max_pool2d_backward_dims(&x, &bad_grad, [3, 3], [1, 1]).is_ok());
    let mismatch_grad = upload(&be, &[1.0; 9], &[1, 1, 3, 3]);
    assert!(matches!(
        max_pool2d_backward_dims(&x, &mismatch_grad, [2, 2], [2, 2]), // forward out is 2x2, mismatch_grad is 3x3
        Err(OjasError::Shape { .. })
    ));
}

// =============================================================================
// SECTION 2: 2D CONVOLUTIONS & TRANSPOSED CONVOLUTIONS
// =============================================================================

#[test]
fn tier5_cpu_04_conv2d_depthwise_and_asymmetric_stride_dilation() {
    let be = cpu_exact(2);
    // Depthwise conv: 4 channels, groups = 4 (per_group = 1), kernel [3, 1], stride [2, 1], dilation [1, 2]
    let params = Conv2dParams {
        stride: [2, 1],
        padding: [1, 1],
        dilation: [1, 2],
        groups: 4,
    };
    let input = upload(&be, &vec![1.0; 1 * 4 * 6 * 6], &[1, 4, 6, 6]);
    // Weight: [out_channels, in_channels / groups, kh, kw] = [4, 1, 3, 1]
    let weight = upload(&be, &vec![0.5; 4 * 1 * 3 * 1], &[4, 1, 3, 1]);
    let bias = upload(&be, &[0.1, 0.2, 0.3, 0.4], &[4]);

    let out = be
        .conv2d_forward(&input, &weight, Some(&bias), params)
        .expect("forward");
    assert_eq!(out.shape().len(), 4);
    assert_eq!(out.shape()[0], 1);
    assert_eq!(out.shape()[1], 4);

    // Backward pass: ensure input, weight, and bias gradients are all populated and finite
    let grad_out = upload(&be, &vec![1.0; out.shape().iter().product()], out.shape());
    let grads = be
        .conv2d_backward(&input, &weight, Some(&bias), &grad_out, params)
        .expect("backward");
    assert_eq!(grads.input.shape(), input.shape());
    assert_eq!(grads.weight.shape(), weight.shape());
    assert!(grads.bias.is_some());
    assert_eq!(grads.bias.as_ref().unwrap().shape(), bias.shape());
    assert!(host(&be, &grads.input).iter().all(|v| v.is_finite()));
    assert!(host(&be, &grads.weight).iter().all(|v| v.is_finite()));
}

#[test]
fn tier5_cpu_05_conv2d_validator_rejection_boundaries() {
    let be = cpu_exact(1);
    let x = upload(&be, &[1.0; 1 * 4 * 8 * 8], &[1, 4, 8, 8]);
    let w = upload(&be, &[1.0; 4 * 4 * 3 * 3], &[4, 4, 3, 3]);

    // 1. groups = 0
    let mut p = Conv2dParams::default();
    p.groups = 0;
    assert!(matches!(
        conv2d_forward_dims(&x, &w, None, p),
        Err(OjasError::Shape { .. })
    ));

    // 2. in_channels not divisible by groups
    let mut p2 = Conv2dParams::default();
    p2.groups = 3;
    assert!(matches!(
        conv2d_forward_dims(&x, &w, None, p2),
        Err(OjasError::Shape { .. })
    ));

    // 3. stride = 0
    let mut p3 = Conv2dParams::default();
    p3.stride = [0, 1];
    assert!(matches!(
        conv2d_forward_dims(&x, &w, None, p3),
        Err(OjasError::Shape { .. })
    ));

    // 4. dilation = 0
    let mut p4 = Conv2dParams::default();
    p4.dilation = [1, 0];
    assert!(matches!(
        conv2d_forward_dims(&x, &w, None, p4),
        Err(OjasError::Shape { .. })
    ));

    // 5. Dilated kernel exceeds padded extent
    let mut p5 = Conv2dParams::default();
    p5.dilation = [10, 1];
    assert!(matches!(
        conv2d_forward_dims(&x, &w, None, p5),
        Err(OjasError::Shape { .. })
    ));

    // 6. Bias length mismatch
    let bad_bias = upload(&be, &[1.0, 2.0], &[2]);
    assert!(matches!(
        conv2d_forward_dims(&x, &w, Some(&bad_bias), Conv2dParams::default()),
        Err(OjasError::Shape { .. })
    ));

    // 7. Backward grad_output shape mismatch
    let bad_grad = upload(&be, &[1.0; 16], &[1, 4, 2, 2]); // correct out shape for 8x8 with 3x3 no pad is 6x6
    assert!(matches!(
        conv2d_backward_dims(&x, &w, None, &bad_grad, Conv2dParams::default()),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn tier5_cpu_06_conv_transpose2d_odd_even_dimensions_and_output_padding() {
    let be = cpu_exact(2);
    // ConvTranspose2d with stride [2, 2] and output_padding [1, 0]
    let params = ConvTranspose2dParams {
        stride: [2, 2],
        padding: [1, 1],
        output_padding: [1, 0],
    };
    // input [1, 2, 5, 4], weight [in_channels=2, out_channels=3, 3, 3]
    let input = upload(&be, &vec![1.0; 1 * 2 * 5 * 4], &[1, 2, 5, 4]);
    let weight = upload(&be, &vec![0.25; 2 * 3 * 3 * 3], &[2, 3, 3, 3]);
    let bias = upload(&be, &[0.5, 1.0, 1.5], &[3]);

    let out = be
        .conv_transpose2d_forward(&input, &weight, Some(&bias), params)
        .expect("forward");
    // Ho = (5 - 1)*2 - 2*1 + 3 + 1 = 8 - 2 + 3 + 1 = 10
    // Wo = (4 - 1)*2 - 2*1 + 3 + 0 = 6 - 2 + 3 + 0 = 7
    assert_eq!(out.shape(), &[1, 3, 10, 7]);

    let grad_out = upload(&be, &vec![1.0; 1 * 3 * 10 * 7], &[1, 3, 10, 7]);
    let grads = be
        .conv_transpose2d_backward(&input, &weight, Some(&bias), &grad_out, params)
        .expect("backward");
    assert_eq!(grads.input.shape(), input.shape());
    assert_eq!(grads.weight.shape(), weight.shape());
    assert_eq!(grads.bias.as_ref().unwrap().shape(), bias.shape());
}

#[test]
fn tier5_cpu_07_conv_transpose2d_validator_rejection_boundaries() {
    let be = cpu_exact(1);
    let x = upload(&be, &[1.0; 1 * 2 * 4 * 4], &[1, 2, 4, 4]);
    let w = upload(&be, &[1.0; 2 * 2 * 3 * 3], &[2, 2, 3, 3]);

    // 1. output_padding >= stride
    let mut p1 = ConvTranspose2dParams::default();
    p1.stride = [2, 2];
    p1.output_padding = [2, 0];
    assert!(matches!(
        conv_transpose2d_forward_dims(&x, &w, None, p1),
        Err(OjasError::Shape { .. })
    ));

    // 2. zero stride
    let mut p2 = ConvTranspose2dParams::default();
    p2.stride = [1, 0];
    assert!(matches!(
        conv_transpose2d_forward_dims(&x, &w, None, p2),
        Err(OjasError::Shape { .. })
    ));

    // 3. weight in-channels mismatch
    let bad_w = upload(&be, &[1.0; 3 * 2 * 3 * 3], &[3, 2, 3, 3]); // w[0] is 3 != in_channels 2
    assert!(matches!(
        conv_transpose2d_forward_dims(&x, &bad_w, None, ConvTranspose2dParams::default()),
        Err(OjasError::Shape { .. })
    ));

    // 4. Excessive padding leaving empty output
    let mut p4 = ConvTranspose2dParams::default();
    p4.padding = [100, 0];
    assert!(matches!(
        conv_transpose2d_forward_dims(&x, &w, None, p4),
        Err(OjasError::Shape { .. })
    ));

    // 5. Backward grad_output shape mismatch
    let bad_grad = upload(&be, &[1.0; 8], &[1, 2, 2, 2]);
    assert!(matches!(
        conv_transpose2d_backward_dims(&x, &w, None, &bad_grad, ConvTranspose2dParams::default()),
        Err(OjasError::Shape { .. })
    ));
}

// =============================================================================
// SECTION 3: BILINEAR INTERPOLATION & ROPE TABLES
// =============================================================================

#[test]
fn tier5_cpu_08_bilinear_1x1_upscaling_and_gradient_conservation() {
    let be = cpu_exact(2);
    // 1x1 image upscaled to 4x4
    let input = upload(&be, &[42.0], &[1, 1, 1, 1]);

    // With align_corners: true (scale = 0.0), all output elements must equal the single pixel!
    let out_aligned = be
        .interpolate_bilinear_forward(&input, [4, 4], true)
        .expect("forward aligned");
    let y_aligned = host(&be, &out_aligned);
    assert_eq!(y_aligned, vec![42.0; 16]);

    // With align_corners: false, all output elements must still equal 42.0
    let out_unaligned = be
        .interpolate_bilinear_forward(&input, [4, 4], false)
        .expect("forward unaligned");
    let y_unaligned = host(&be, &out_unaligned);
    assert_eq!(y_unaligned, vec![42.0; 16]);

    // Gradient conservation: Bilinear interpolation is a linear partition of unity!
    // Sum of all output gradients must equal the sum of input gradients.
    let grad_out = upload(&be, &[1.0; 16], &[1, 1, 4, 4]);
    let grad_in = be
        .interpolate_bilinear_backward(&input, &grad_out, false)
        .expect("backward");
    let gx = host(&be, &grad_in);
    assert_eq!(gx.len(), 1);
    let sum_dy: f32 = 16.0;
    assert!(
        (gx[0] - sum_dy).abs() < 1e-5,
        "sum of input gradient ({}) must equal sum of output gradients ({sum_dy})",
        gx[0]
    );
}

#[test]
fn tier5_cpu_09_bilinear_validator_rejections() {
    let be = cpu_exact(1);
    let x = upload(&be, &[1.0; 16], &[1, 1, 4, 4]);

    // 1. Zero in out_size
    assert!(matches!(
        interpolate_bilinear_forward_dims(&x, [0, 4]),
        Err(OjasError::Shape { .. })
    ));
    assert!(matches!(
        interpolate_bilinear_forward_dims(&x, [4, 0]),
        Err(OjasError::Shape { .. })
    ));

    // 2. Rank != 4
    let x_rank2 = upload(&be, &[1.0; 16], &[4, 4]);
    assert!(matches!(
        interpolate_bilinear_forward_dims(&x_rank2, [8, 8]),
        Err(OjasError::Shape { .. })
    ));

    // 3. Backward grad_output shape mismatch
    let bad_grad = upload(&be, &[1.0; 32], &[1, 2, 4, 4]); // channels 2 != 1
    assert!(matches!(
        interpolate_bilinear_backward_dims(&x, &bad_grad),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn tier5_cpu_10_axial_rope_and_interleaved_inversion() {
    let be = cpu_exact(1);
    // Valid table generation
    let (cos, sin) = axial_rope_2d_tables(16, 4, 4, 10_000.0, 1.0, be.budget()).expect("tables");
    assert_eq!(cos.shape(), &[16, 8]);
    assert_eq!(sin.shape(), &[16, 8]);

    // Validator refusals
    assert!(matches!(
        axial_rope_2d_tables(15, 4, 4, 10_000.0, 1.0, be.budget()),
        Err(OjasError::OutOfRange { .. })
    ));
    assert!(matches!(
        axial_rope_2d_tables(16, 0, 4, 10_000.0, 1.0, be.budget()),
        Err(OjasError::Shape { .. })
    ));
    assert!(matches!(
        axial_rope_2d_tables(16, 4, 4, 0.5, 1.0, be.budget()),
        Err(OjasError::OutOfRange { .. })
    ));
    assert!(matches!(
        axial_rope_2d_tables(16, 4, 4, 10_000.0, -1.0, be.budget()),
        Err(OjasError::OutOfRange { .. })
    ));

    // Roundtrip rotation: interleaved forward then interleaved backward reconstructs the input!
    // Shape: [batch=1, time=16, heads=2, dim=16]
    let mut data = vec![0.0f32; 1 * 16 * 2 * 16];
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.1 - 0.8;
    }
    let x = upload(&be, &data, &[1, 16, 2, 16]);
    let rotated = be
        .rope_interleaved_forward(&x, &cos, &sin)
        .expect("forward");
    let unrotated = be
        .rope_interleaved_backward(&rotated, &cos, &sin)
        .expect("backward");
    let x_orig = host(&be, &x);
    let x_rec = host(&be, &unrotated);
    for (a, b) in x_orig.iter().zip(&x_rec) {
        assert!(
            (a - b).abs() < 1e-6,
            "rope roundtrip must preserve coordinates: {a} vs {b}"
        );
    }

    // Odd head_dim in rope_interleaved_forward_dims must be refused
    let odd_x = upload(&be, &[1.0; 15], &[1, 1, 1, 15]);
    let odd_cos = upload(&be, &[1.0; 7], &[1, 7]);
    assert!(matches!(
        rope_interleaved_forward_dims(&odd_x, &odd_cos, &odd_cos),
        Err(OjasError::Shape { .. })
    ));
}

// =============================================================================
// SECTION 4: GEMMA OPS NUMERICAL STABILITY & BOUNDARIES
// =============================================================================

#[test]
fn tier5_cpu_11_gelu_tanh_subnormals_and_analytical_diff() {
    let be = cpu_exact(1);
    // Subnormals (1e-40) near zero
    let x_sub = upload(&be, &[1e-40f32, -1e-40f32], &[2]);
    let y_sub = be.gelu_tanh_forward(&x_sub).expect("subnormal forward");
    let res = host(&be, &y_sub);
    assert!(res[0] > 0.0 && res[0].is_finite());
    assert!(res[1] <= 0.0 && res[1].is_finite());

    // Analytical derivative check against f64 central difference
    // gelu_tanh_backward_exact(x, dy) vs (f(x+h) - f(x-h)) / (2*h)
    let test_points = [-4.0f32, -2.5, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5, 4.0];
    let h = 1e-4f64;
    for &pt in &test_points {
        let analytic = gelu_tanh_backward_exact(pt, 1.0);
        let f_pos = gelu_tanh_exact((pt as f64 + h) as f32) as f64;
        let f_neg = gelu_tanh_exact((pt as f64 - h) as f32) as f64;
        let numeric = (f_pos - f_neg) / (2.0 * h);
        let diff = (f64::from(analytic) - numeric).abs();
        assert!(
            diff < 2e-3,
            "gelu_tanh analytical derivative at {pt}: analytic {analytic} vs numeric {numeric}"
        );
    }
}

#[test]
fn tier5_cpu_12_seq_mean_zero_length_and_dynamic_range_cancellation() {
    let be = cpu_exact(2);
    // Batch of 3 sequences:
    // Seq 0: key_lens = 0 (empty) -> must output all zeros!
    // Seq 1: key_lens = 100 (exceeds T=3) -> must clamp to T=3 without out-of-bounds read!
    // Seq 2: High dynamic range values: [1e7, -1e7, 3.0] -> sum in f64 gives 3.0, mean is 1.0!
    let data = vec![
        // Seq 0 (len 3, dim 2)
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Seq 1
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, // Seq 2 (high dynamic range)
        1e7, -1e7, -1e7, 1e7, 3.0, 6.0,
    ];
    let x = upload(&be, &data, &[3, 3, 2]);
    let lens = upload_u32(&be, &[0, 100, 3], &[3]);

    let mean = be.seq_mean_forward(&x, &lens).expect("seq_mean");
    let y = host(&be, &mean);
    assert_eq!(y.len(), 6); // [3, 2]

    // Seq 0 must be exactly [0.0, 0.0]
    assert_eq!(y[0], 0.0);
    assert_eq!(y[1], 0.0);

    // Seq 1 clamped to 3 tokens: mean of [10, 30, 50] = 30, mean of [20, 40, 60] = 40
    assert_eq!(y[2], 30.0);
    assert_eq!(y[3], 40.0);

    // Seq 2: (1e7 - 1e7 + 3.0)/3.0 = 1.0, (-1e7 + 1e7 + 6.0)/3.0 = 2.0
    assert_eq!(
        y[4], 1.0,
        "f64 sum must not cancel out small value in high dynamic range"
    );
    assert_eq!(y[5], 2.0);

    // Validator refusals
    assert!(matches!(
        seq_mean_forward_dims(&x, &x),
        Err(OjasError::Dtype { .. })
    )); // key_lens must be U32
    let bad_lens = upload_u32(&be, &[1, 2], &[2]); // lens len 2 != batch 3
    assert!(matches!(
        seq_mean_forward_dims(&x, &bad_lens),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn tier5_cpu_13_l2_normalize_extreme_vectors_and_subnormals() {
    let be = cpu_exact(1);
    // Row 0: All zeros -> magnitude 0, clamped by eps -> output is exact zeros
    // Row 1: Subnormals [1e-40, 2e-40]
    // Row 2: Huge floats [1e36, 1.0, -1e36] (sum of squares in f32 would overflow to inf, but in f64 fits!)
    // Row 3: Preserving negative zeros [-0.0, -0.0]
    let data = vec![
        0.0, 0.0, 0.0, 1e-40, 2e-40, 0.0, 1e36, 1.0, -1e36, -0.0, -0.0, 0.0,
    ];
    let x = upload(&be, &data, &[4, 3]);
    let norm = be.l2_normalize_forward(&x, 1e-12).expect("l2_normalize");
    let y = host(&be, &norm);

    // Row 0: all zeros
    assert_eq!(&y[0..3], &[0.0, 0.0, 0.0]);

    // Row 1: finite and non-NaN
    assert!(y[3].is_finite() && y[4].is_finite());

    // Row 2: Huge floats normalized without overflow: norm = sqrt(2 * 1e72) = 1.41421356e36
    let expected_scale = 1.0 / 2.0f32.sqrt();
    assert!(
        (y[6] - expected_scale).abs() < 1e-5,
        "huge float must be normalized: {}",
        y[6]
    );
    assert!((y[8] - (-expected_scale)).abs() < 1e-5);

    // Row 3: Negative zero signs preserved
    assert!(y[9].is_sign_negative());
    assert!(y[10].is_sign_negative());

    // Validator refusals
    assert!(matches!(
        l2_normalize_forward_dims(&x, 0.0),
        Err(OjasError::OutOfRange { .. })
    ));
    assert!(matches!(
        l2_normalize_forward_dims(&x, -1e-5),
        Err(OjasError::OutOfRange { .. })
    ));
    assert!(matches!(
        l2_normalize_forward_dims(&x, f32::NAN),
        Err(OjasError::NonFinite { .. })
    ));
    let x_rank1 = upload(&be, &[1.0, 2.0], &[2]);
    assert!(matches!(
        l2_normalize_forward_dims(&x_rank1, 1e-5),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn tier5_cpu_14_rms_qkv_norm_weightless_v_and_eps_refusal() {
    let be = cpu_exact(2);
    // [B=1, T=2, H=2, D=4]
    let q = upload(&be, &vec![2.0; 16], &[1, 2, 2, 4]);
    let k = upload(&be, &vec![4.0; 16], &[1, 2, 2, 4]);
    let v = upload(&be, &vec![8.0; 16], &[1, 2, 2, 4]);
    let q_w = upload(&be, &[0.5, 0.5, 0.5, 0.5], &[4]);

    // Forward: Q has weight, K is weightless, V is strictly weightless
    let (qn, kn, vn) = be
        .rms_qkv_norm_forward(&q, &k, &v, Some(&q_w), None, 1e-6)
        .expect("rms_qkv_norm");
    let y_q = host(&be, &qn);
    let y_k = host(&be, &kn);
    let y_v = host(&be, &vn);

    // Constant rows normalized by RMS:
    // Q is constant 2.0: RMS = 2.0. q / RMS = 1.0. Times weight 0.5 = 0.5.
    assert!(y_q.iter().all(|&val| (val - 0.5).abs() < 1e-5));
    // K is constant 4.0: RMS = 4.0. k / RMS = 1.0 (weightless).
    assert!(y_k.iter().all(|&val| (val - 1.0).abs() < 1e-5));
    // V is constant 8.0: RMS = 8.0. v / RMS = 1.0 (weightless).
    assert!(y_v.iter().all(|&val| (val - 1.0).abs() < 1e-5));

    // Validator refusals
    assert!(matches!(
        rms_qkv_norm_forward_dims(&q, &k, &v, None, None, f32::NAN),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(matches!(
        rms_qkv_norm_forward_dims(&q, &k, &v, None, None, f32::INFINITY),
        Err(OjasError::NonFinite { .. })
    ));
    let bad_weight = upload(&be, &[1.0; 5], &[5]); // head_dim is 4 != 5
    assert!(matches!(
        rms_qkv_norm_forward_dims(&q, &k, &v, Some(&bad_weight), None, 1e-6),
        Err(OjasError::Shape { .. })
    ));
}

// =============================================================================
// SECTION 5: GROUPNORM, CHANNEL LAYERNORM, SLICE, AND CONCAT
// =============================================================================

#[test]
fn tier5_cpu_15_group_norm_and_channel_layernorm_boundaries() {
    let be = cpu_exact(2);
    // Input: [N=1, C=4, H=2, W=2]
    let x = upload(&be, &vec![2.0; 16], &[1, 4, 2, 2]);
    let w = upload(&be, &[1.0, 1.0, 1.0, 1.0], &[4]);
    let b = upload(&be, &[0.0, 0.0, 0.0, 0.0], &[4]);

    // 1. GroupNorm with groups = 2 (divisible)
    let gn = be
        .group_norm_forward(&x, &w, &b, 2, 1e-5)
        .expect("gn forward");
    assert_eq!(gn.shape(), &[1, 4, 2, 2]);

    // 2. GroupNorm validator refusals
    assert!(matches!(
        group_norm_forward_dims(&x, &w, &b, 0, 1e-5),
        Err(OjasError::OutOfRange { .. })
    ));
    assert!(matches!(
        group_norm_forward_dims(&x, &w, &b, 3, 1e-5),
        Err(OjasError::Shape { .. })
    )); // 4 not divisible by 3
    assert!(matches!(
        group_norm_forward_dims(&x, &w, &b, 2, 0.0),
        Err(OjasError::OutOfRange { .. })
    ));
    assert!(matches!(
        group_norm_forward_dims(&x, &w, &b, 2, f32::NAN),
        Err(OjasError::NonFinite { .. })
    ));

    // 3. ChannelLayerNorm
    let cln = be
        .channel_layer_norm_forward(&x, &w, &b, 1e-5)
        .expect("cln forward");
    assert_eq!(cln.shape(), &[1, 4, 2, 2]);

    // 4. ChannelLayerNorm validator refusals
    let x_rank1 = upload(&be, &[1.0; 4], &[4]);
    assert!(matches!(
        channel_layer_norm_forward_dims(&x_rank1, &w, &b, 1e-5),
        Err(OjasError::Shape { .. })
    ));
    assert!(matches!(
        channel_layer_norm_forward_dims(&x, &w, &b, f32::INFINITY),
        Err(OjasError::NonFinite { .. })
    ));
}

#[test]
fn tier5_cpu_16_slice_and_concat_adversarial_boundaries() {
    let be = cpu_exact(1);
    let x = upload(
        &be,
        &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
    );

    // Slice valid
    let s = be.slice(&x, 1, 1, 2).expect("slice");
    assert_eq!(s.shape(), &[2, 2, 4]);

    // Slice backward places grad_output at [2, 1..3, 4]
    let grad_out = upload(&be, &[1.0; 16], &[2, 2, 4]);
    let s_bwd = be
        .slice_backward(&grad_out, &[2, 3, 4], 1, 1)
        .expect("slice_backward");
    assert_eq!(s_bwd.shape(), &[2, 3, 4]);
    let vals = host(&be, &s_bwd);
    // Axis 1 index 0 must be 0, index 1 and 2 must be 1.0
    for row in vals.chunks(12) {
        assert_eq!(&row[0..4], &[0.0; 4]);
        assert_eq!(&row[4..12], &[1.0; 8]);
    }

    // Slice refusals
    assert!(matches!(
        slice_dims(&x, 1, 0, 0),
        Err(OjasError::Shape { .. })
    )); // len 0
    assert!(matches!(
        slice_dims(&x, 1, 2, 2),
        Err(OjasError::OutOfRange { .. })
    )); // 2+2 > 3
    assert!(matches!(
        slice_dims(&x, 3, 0, 1),
        Err(OjasError::Shape { .. })
    )); // axis 3 >= rank 3

    // Slice backward dims check
    let dims_bwd = slice_backward_dims(&grad_out, &[2, 3, 4], 1, 1).expect("slice_backward_dims");
    assert_eq!(dims_bwd.start, 1);
    assert_eq!(dims_bwd.len, 2);
    assert!(matches!(
        slice_backward_dims(&grad_out, &[2, 2, 4], 1, 1),
        Err(OjasError::OutOfRange { .. })
    ));

    // Concat valid
    let c = be.concat(&[&s, &s], 1).expect("concat");
    assert_eq!(c.shape(), &[2, 4, 4]);

    // Concat refusals
    assert!(matches!(concat_dims(&[], 0), Err(OjasError::Shape { .. })));
    assert!(matches!(
        concat_dims(&[&s], 3),
        Err(OjasError::Shape { .. })
    )); // axis 3 >= rank 3
    assert!(matches!(
        concat_dims(&[&s, &x], 0),
        Err(OjasError::Shape { .. })
    )); // off-axis mismatch (s axis 1 has len 2, x axis 1 has len 3)
}

#[test]
fn tier5_cpu_17_attention_radius_sliding_window_refusals() {
    let be = cpu_exact(1);
    let q = upload(&be, &[1.0; 16], &[1, 2, 2, 4]);
    let k = upload(&be, &[1.0; 16], &[1, 2, 2, 4]);
    let v = upload(&be, &[1.0; 16], &[1, 2, 2, 4]);

    // 1. radius = 0 must be refused: "attention radius must be at least 1"
    let mut opts0 = AttentionOpts::default();
    opts0.radius = Some(0);
    assert!(be.attention_forward(&q, &k, &v, opts0).is_err());

    // 2. radius with causal = true must be refused: "attention radius is not causal"
    let mut opts_causal = AttentionOpts::default();
    opts_causal.radius = Some(1);
    opts_causal.causal = true;
    assert!(be.attention_forward(&q, &k, &v, opts_causal).is_err());

    // 3. radius with window.is_some() must be refused
    let mut opts_window = AttentionOpts::default();
    opts_window.radius = Some(1);
    opts_window.window = Some(1);
    assert!(be.attention_forward(&q, &k, &v, opts_window).is_err());

    // 4. radius with Tq != Tk must be refused
    let q_diff = upload(&be, &[1.0; 8], &[1, 1, 2, 4]);
    let mut opts_diff = AttentionOpts::default();
    opts_diff.radius = Some(1);
    assert!(be.attention_forward(&q_diff, &k, &v, opts_diff).is_err());

    // 5. Valid radius forward
    let mut opts_valid = AttentionOpts::default();
    opts_valid.radius = Some(1);
    let res = be.attention_forward(&q, &k, &v, opts_valid);
    assert!(res.is_ok());
}

// =============================================================================
// SECTION 6: MULTI-THREADED EXACT NUMERICS BIT-IDENTITY
// =============================================================================

#[test]
fn tier5_cpu_18_multi_threaded_exact_numerics_determinism() {
    // Under Numerics::Exact, 1-thread and 4-thread execution MUST agree bit for bit!
    let be1 = cpu_exact(1);
    let be4 = cpu_exact(4);

    // 1. MaxPool2D
    let mut pool_data = vec![0.0f32; 2 * 4 * 16 * 16];
    for (i, v) in pool_data.iter_mut().enumerate() {
        *v = ((i % 29) as f32) * 0.1 - 1.4;
    }
    let p_in1 = upload(&be1, &pool_data, &[2, 4, 16, 16]);
    let p_in4 = upload(&be4, &pool_data, &[2, 4, 16, 16]);
    let p_out1 = be1.max_pool2d_forward(&p_in1, [3, 3], [2, 2]).expect("p1");
    let p_out4 = be4.max_pool2d_forward(&p_in4, [3, 3], [2, 2]).expect("p4");
    assert_eq!(
        host_bits(&be1, &p_out1),
        host_bits(&be4, &p_out4),
        "max_pool2d bit-identity violated"
    );

    // 2. Bilinear Interpolation
    let b_out1 = be1
        .interpolate_bilinear_forward(&p_in1, [24, 24], false)
        .expect("b1");
    let b_out4 = be4
        .interpolate_bilinear_forward(&p_in4, [24, 24], false)
        .expect("b4");
    assert_eq!(
        host_bits(&be1, &b_out1),
        host_bits(&be4, &b_out4),
        "bilinear bit-identity violated"
    );

    // 3. Gelu Tanh
    let g_out1 = be1.gelu_tanh_forward(&p_in1).expect("g1");
    let g_out4 = be4.gelu_tanh_forward(&p_in4).expect("g4");
    assert_eq!(
        host_bits(&be1, &g_out1),
        host_bits(&be4, &g_out4),
        "gelu_tanh bit-identity violated"
    );

    // 4. L2 Normalize
    let flat1 = upload(&be1, &pool_data, &[32, 64]);
    let flat4 = upload(&be4, &pool_data, &[32, 64]);
    let l_out1 = be1.l2_normalize_forward(&flat1, 1e-12).expect("l1");
    let l_out4 = be4.l2_normalize_forward(&flat4, 1e-12).expect("l4");
    assert_eq!(
        host_bits(&be1, &l_out1),
        host_bits(&be4, &l_out4),
        "l2_normalize bit-identity violated"
    );

    // 5. Sequence Mean
    let s_in1 = upload(&be1, &pool_data[0..2 * 8 * 16], &[2, 8, 16]);
    let s_in4 = upload(&be4, &pool_data[0..2 * 8 * 16], &[2, 8, 16]);
    let lens1 = upload_u32(&be1, &[5, 8], &[2]);
    let lens4 = upload_u32(&be4, &[5, 8], &[2]);
    let s_out1 = be1.seq_mean_forward(&s_in1, &lens1).expect("s1");
    let s_out4 = be4.seq_mean_forward(&s_in4, &lens4).expect("s4");
    assert_eq!(
        host_bits(&be1, &s_out1),
        host_bits(&be4, &s_out4),
        "seq_mean bit-identity violated"
    );
}

#[test]
fn tier5_cpu_19_non_finite_rejection_across_all_cpu_ops() {
    let be = cpu_exact(1);
    let nan_data = vec![1.0, f32::NAN, 3.0, 4.0];
    let inf_data = vec![1.0, f32::INFINITY, 3.0, 4.0];

    // Upload with NaN and Inf
    let nan_t = Tensor::from_f32(&nan_data, &[1, 1, 2, 2], be.budget()).unwrap();
    let inf_t = Tensor::from_f32(&inf_data, &[1, 1, 2, 2], be.budget()).unwrap();

    let nan_up = be.upload(&nan_t).unwrap();
    let inf_up = be.upload(&inf_t).unwrap();

    // 1. max_pool2d
    assert!(matches!(
        be.max_pool2d_forward(&nan_up, [2, 2], [1, 1]),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(matches!(
        be.max_pool2d_forward(&inf_up, [2, 2], [1, 1]),
        Err(OjasError::NonFinite { .. })
    ));

    // 2. interpolate_bilinear
    assert!(matches!(
        be.interpolate_bilinear_forward(&nan_up, [4, 4], false),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(matches!(
        be.interpolate_bilinear_forward(&inf_up, [4, 4], false),
        Err(OjasError::NonFinite { .. })
    ));

    // 3. gelu_tanh
    assert!(matches!(
        be.gelu_tanh_forward(&nan_up),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(matches!(
        be.gelu_tanh_forward(&inf_up),
        Err(OjasError::NonFinite { .. })
    ));

    // 4. l2_normalize
    let nan_2d = Tensor::from_f32(&nan_data, &[2, 2], be.budget()).unwrap();
    let nan_2d_up = be.upload(&nan_2d).unwrap();
    assert!(matches!(
        be.l2_normalize_forward(&nan_2d_up, 1e-12),
        Err(OjasError::NonFinite { .. })
    ));
}
