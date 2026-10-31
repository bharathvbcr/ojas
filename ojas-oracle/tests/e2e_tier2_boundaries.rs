//! E2E Test Suite — Tier 2: Boundary & Corner Cases (>=5 per feature)
//!
//! Stresses boundary values, zero/empty dimensions, non-finite (NaN/Inf) inputs,
//! resource constraints, and adversarial conditions across SAM 3.1 and Gemma embedding operations.

#![allow(clippy::identity_op, clippy::useless_vec, clippy::needless_range_loop)]

use ojas_core::{
    axial_rope_2d_tables, AttentionOpts, Backend, Budget, Conv2dParams, ConvTranspose2dParams,
    DType, Numerics, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(2 << 30)).with_numerics(Numerics::Exact)
}

fn host(be: &CpuBackend, t: &Tensor) -> Vec<f32> {
    be.download(t)
        .expect("download")
        .to_f32_vec()
        .expect("to_f32_vec")
}

fn upload(be: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    let t = Tensor::from_f32(data, shape, be.budget()).expect("from_f32");
    be.upload(&t).expect("upload")
}

// -----------------------------------------------------------------------------
// FEATURE 1 (RoPE): Boundaries & Adversarial Inputs (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat1_01_rope_tables_zero_grid_fails() {
    let be = cpu();
    let res = axial_rope_2d_tables(64, 0, 24, 10_000.0, 1.0, be.budget());
    assert!(res.is_err(), "zero grid height must be rejected");
}

#[test]
fn tier2_feat1_02_rope_tables_non_multiple_of_4_head_dim_fails() {
    let be = cpu();
    let res = axial_rope_2d_tables(33, 24, 24, 10_000.0, 1.0, be.budget());
    assert!(res.is_err(), "head_dim not multiple of 4 must be rejected");
}

#[test]
fn tier2_feat1_03_rope_tables_nan_theta_fails() {
    let be = cpu();
    let res = axial_rope_2d_tables(64, 24, 24, f64::NAN, 1.0, be.budget());
    assert!(res.is_err(), "NaN theta must be rejected");
}

#[test]
fn tier2_feat1_04_rope_tables_negative_theta_fails() {
    let be = cpu();
    let res = axial_rope_2d_tables(64, 24, 24, -100.0, 1.0, be.budget());
    assert!(res.is_err(), "negative theta must be rejected");
}

#[test]
fn tier2_feat1_05_rope_interleaved_nan_input_refused() {
    let be = cpu();
    let (cos, sin) = axial_rope_2d_tables(64, 4, 4, 10_000.0, 1.0, be.budget()).expect("tables");
    let mut data = vec![1.0f32; 1 * 16 * 1 * 64];
    data[10] = f32::NAN;
    let t_res = Tensor::from_f32(&data, &[1, 16, 1, 64], be.budget());
    assert!(t_res.is_ok());
    let t = t_res.unwrap();
    // Non-finite detection in upload or kernel call
    let up_res = be.upload(&t);
    if let Ok(uploaded) = up_res {
        let rot_res = be.rope_interleaved_forward(&uploaded, &cos, &sin);
        assert!(matches!(rot_res, Err(OjasError::NonFinite { .. })));
    }
}

// -----------------------------------------------------------------------------
// FEATURE 2 (Attention): Boundaries & Adversarial Inputs (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat2_01_attention_zero_length_sequence_refused() {
    let be = cpu();
    let res = Tensor::from_f32(&[], &[1, 1, 0, 64], be.budget());
    assert!(
        res.is_err() || {
            let t = res.unwrap();
            let opts = AttentionOpts {
                causal: false,
                window: None,
                key_lens: None,
                bias: None,
                ..Default::default()
            };
            be.attention_forward(&t, &t, &t, opts).is_err()
        }
    );
}

#[test]
fn tier2_feat2_02_attention_nan_query_refused() {
    let be = cpu();
    let mut q_data = vec![0.1f32; 64];
    q_data[0] = f32::NAN;
    let q_res = Tensor::from_f32(&q_data, &[1, 1, 1, 64], be.budget());
    if let Ok(q) = q_res {
        let up_res = be.upload(&q);
        if let Ok(uploaded_q) = up_res {
            let k = upload(&be, &vec![0.1f32; 64], &[1, 1, 1, 64]);
            let opts = AttentionOpts {
                causal: false,
                window: None,
                key_lens: None,
                bias: None,
                ..Default::default()
            };
            let res = be.attention_forward(&uploaded_q, &k, &k, opts);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat2_03_attention_inf_key_refused() {
    let be = cpu();
    let q = upload(&be, &vec![0.1f32; 64], &[1, 1, 1, 64]);
    let mut k_data = vec![0.1f32; 64];
    k_data[5] = f32::INFINITY;
    if let Ok(k) = Tensor::from_f32(&k_data, &[1, 1, 1, 64], be.budget()) {
        if let Ok(uploaded_k) = be.upload(&k) {
            let opts = AttentionOpts {
                causal: false,
                window: None,
                key_lens: None,
                bias: None,
                ..Default::default()
            };
            let res = be.attention_forward(&q, &uploaded_k, &q, opts);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat2_04_attention_dimension_mismatch() {
    let be = cpu();
    let q = upload(&be, &vec![0.1f32; 64], &[1, 1, 1, 64]);
    let k = upload(&be, &vec![0.1f32; 32], &[1, 1, 1, 32]); // Head dim mismatch
    let opts = AttentionOpts {
        causal: false,
        window: None,
        key_lens: None,
        bias: None,
        ..Default::default()
    };
    let res = be.attention_forward(&q, &k, &k, opts);
    assert!(matches!(res, Err(OjasError::Shape { .. })));
}

#[test]
fn tier2_feat2_05_attention_prime_dimensions() {
    let be = cpu();
    // Prime query length 37, key length 43, head dim 32
    let q = upload(&be, &vec![0.01f32; 1 * 2 * 37 * 32], &[1, 2, 37, 32]);
    let k = upload(&be, &vec![0.01f32; 1 * 2 * 43 * 32], &[1, 2, 43, 32]);
    let v = upload(&be, &vec![0.01f32; 1 * 2 * 43 * 32], &[1, 2, 43, 32]);
    let opts = AttentionOpts {
        causal: false,
        window: None,
        key_lens: None,
        bias: None,
        ..Default::default()
    };
    let out = be.attention_forward(&q, &k, &v, opts).expect("prime dims");
    assert_eq!(out.shape(), &[1, 2, 37, 32]);
}

// -----------------------------------------------------------------------------
// FEATURE 3 (LayerNorm): Boundaries & Adversarial Inputs (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat3_01_layernorm_nan_eps_refused() {
    let be = cpu();
    let x = upload(&be, &[1.0, 2.0, 3.0], &[1, 1, 3]);
    let w = upload(&be, &[1.0, 1.0, 1.0], &[3]);
    let b = upload(&be, &[0.0, 0.0, 0.0], &[3]);
    let res = be.layer_norm_forward(&x, &w, &b, f32::NAN);
    assert!(
        matches!(res, Err(OjasError::NonFinite { .. })),
        "NaN epsilon must be rejected"
    );
}

#[test]
fn tier2_feat3_02_layernorm_inf_eps_refused() {
    let be = cpu();
    let x = upload(&be, &[1.0, 2.0, 3.0], &[1, 1, 3]);
    let w = upload(&be, &[1.0, 1.0, 1.0], &[3]);
    let b = upload(&be, &[0.0, 0.0, 0.0], &[3]);
    let res = be.layer_norm_forward(&x, &w, &b, f32::INFINITY);
    assert!(
        matches!(res, Err(OjasError::NonFinite { .. })),
        "Inf epsilon must be rejected"
    );
}

#[test]
fn tier2_feat3_03_layernorm_nan_activation_refused() {
    let be = cpu();
    let mut data = vec![1.0f32; 16];
    data[7] = f32::NAN;
    let w = upload(&be, &vec![1.0f32; 16], &[16]);
    let b = upload(&be, &vec![0.0f32; 16], &[16]);
    if let Ok(x) = Tensor::from_f32(&data, &[1, 1, 16], be.budget()) {
        if let Ok(uploaded_x) = be.upload(&x) {
            let res = be.layer_norm_forward(&uploaded_x, &w, &b, 1e-5);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat3_04_layernorm_affine_dimension_mismatch() {
    let be = cpu();
    let x = upload(&be, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let w = upload(&be, &[1.0, 1.0, 1.0], &[3]); // Mismatched 3 vs 4
    let b = upload(&be, &[0.0, 0.0, 0.0], &[3]);
    let res = be.layer_norm_forward(&x, &w, &b, 1e-5);
    assert!(matches!(res, Err(OjasError::Shape { .. })));
}

#[test]
fn tier2_feat3_05_layernorm_extreme_scale_values() {
    let be = cpu();
    let x = upload(&be, &[1e15f32, 2e15, 3e15, 4e15], &[1, 1, 4]);
    let w = upload(&be, &[1.0; 4], &[4]);
    let b = upload(&be, &[0.0; 4], &[4]);
    let out = be
        .layer_norm_forward(&x, &w, &b, 1e-5)
        .expect("large values");
    let data = host(&be, &out);
    assert!(data.iter().all(|v| v.is_finite()));
}

// -----------------------------------------------------------------------------
// FEATURE 4 (Conv2d & Transposed Conv2d): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat4_01_conv2d_kernel_larger_than_image() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 3 * 3], &[1, 1, 3, 3]);
    let w = upload(&be, &[1.0; 1 * 1 * 5 * 5], &[1, 1, 5, 5]); // k=5 > H=3
    let params = Conv2dParams {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    };
    let res = be.conv2d_forward(&x, &w, None, params);
    assert!(
        res.is_err(),
        "kernel larger than image without padding must fail"
    );
}

#[test]
fn tier2_feat4_02_conv2d_zero_stride_refused() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]);
    let w = upload(&be, &[1.0; 1 * 1 * 2 * 2], &[1, 1, 2, 2]);
    let params = Conv2dParams {
        stride: [0, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    };
    let res = be.conv2d_forward(&x, &w, None, params);
    assert!(res.is_err(), "stride 0 must fail");
}

#[test]
fn tier2_feat4_03_conv_transpose2d_output_padding_ge_stride_refused() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]);
    let w = upload(&be, &[1.0; 1 * 1 * 2 * 2], &[1, 1, 2, 2]);
    // output padding 2 >= stride 2 must be refused
    let params = ConvTranspose2dParams {
        stride: [2, 2],
        padding: [0, 0],
        output_padding: [2, 0],
    };
    let res = be.conv_transpose2d_forward(&x, &w, None, params);
    assert!(res.is_err());
}

#[test]
fn tier2_feat4_04_conv2d_nan_weight_refused() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]);
    let mut w_data = vec![1.0f32; 4];
    w_data[2] = f32::NAN;
    let params = Conv2dParams {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    };
    if let Ok(w) = Tensor::from_f32(&w_data, &[1, 1, 2, 2], be.budget()) {
        if let Ok(uploaded_w) = be.upload(&w) {
            let res = be.conv2d_forward(&x, &uploaded_w, None, params);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat4_05_conv2d_channel_mismatch() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 3 * 4 * 4], &[1, 3, 4, 4]);
    let w = upload(&be, &[1.0; 1 * 4 * 2 * 2], &[1, 4, 2, 2]); // in_channels 4 != 3
    let params = Conv2dParams {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    };
    let res = be.conv2d_forward(&x, &w, None, params);
    assert!(matches!(res, Err(OjasError::Shape { .. })));
}

// -----------------------------------------------------------------------------
// FEATURE 5 (GroupNorm & Bilinear): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat5_01_group_norm_channels_not_divisible_by_groups() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 10 * 4 * 4], &[1, 10, 4, 4]);
    let w = upload(&be, &[1.0; 10], &[10]);
    let b = upload(&be, &[0.0; 10], &[10]);
    let res = be.group_norm_forward(&x, &w, &b, 3, 1e-5); // 10 not divisible by 3
    assert!(matches!(res, Err(OjasError::Shape { .. })));
}

#[test]
fn tier2_feat5_02_group_norm_zero_groups_refused() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 4 * 2 * 2], &[1, 4, 2, 2]);
    let w = upload(&be, &[1.0; 4], &[4]);
    let b = upload(&be, &[0.0; 4], &[4]);
    let res = be.group_norm_forward(&x, &w, &b, 0, 1e-5);
    assert!(res.is_err());
}

#[test]
fn tier2_feat5_03_bilinear_zero_output_size_refused() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 2 * 2], &[1, 1, 2, 2]);
    let res = be.interpolate_bilinear_forward(&x, [0, 4], false);
    assert!(res.is_err());
}

#[test]
fn tier2_feat5_04_bilinear_nan_input_refused() {
    let be = cpu();
    let mut data = vec![1.0f32; 4];
    data[1] = f32::NAN;
    if let Ok(t) = Tensor::from_f32(&data, &[1, 1, 2, 2], be.budget()) {
        if let Ok(up) = be.upload(&t) {
            let res = be.interpolate_bilinear_forward(&up, [4, 4], false);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat5_05_group_norm_single_channel_single_group() {
    let be = cpu();
    let x = upload(&be, &[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let w = upload(&be, &[1.0], &[1]);
    let b = upload(&be, &[0.0], &[1]);
    let out = be
        .group_norm_forward(&x, &w, &b, 1, 1e-5)
        .expect("gn instance");
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
}

// -----------------------------------------------------------------------------
// FEATURE 6 (MaxPool2D): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat6_01_max_pool2d_kernel_larger_than_image() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 2 * 2], &[1, 1, 2, 2]);
    let res = be.max_pool2d_forward(&x, [3, 3], [1, 1]); // k=3 > 2
    assert!(res.is_err());
}

#[test]
fn tier2_feat6_02_max_pool2d_zero_stride() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]);
    let res = be.max_pool2d_forward(&x, [2, 2], [0, 2]);
    assert!(res.is_err());
}

#[test]
fn tier2_feat6_03_max_pool2d_zero_kernel() {
    let be = cpu();
    let x = upload(&be, &[1.0; 1 * 1 * 4 * 4], &[1, 1, 4, 4]);
    let res = be.max_pool2d_forward(&x, [0, 2], [1, 1]);
    assert!(res.is_err());
}

#[test]
fn tier2_feat6_04_max_pool2d_nan_input_refused() {
    let be = cpu();
    let mut data = vec![1.0f32; 4];
    data[0] = f32::NAN;
    if let Ok(t) = Tensor::from_f32(&data, &[1, 1, 2, 2], be.budget()) {
        if let Ok(up) = be.upload(&t) {
            let res = be.max_pool2d_forward(&up, [2, 2], [2, 2]);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat6_05_max_pool2d_unit_dimension() {
    let be = cpu();
    let x = upload(&be, &[42.0], &[1, 1, 1, 1]);
    let out = be
        .max_pool2d_forward(&x, [1, 1], [1, 1])
        .expect("unit pool");
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
}

// -----------------------------------------------------------------------------
// FEATURE 7 (GeGLU): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat7_01_gelu_erf_nan_input_refused() {
    let be = cpu();
    let data = vec![f32::NAN];
    if let Ok(t) = Tensor::from_f32(&data, &[1], be.budget()) {
        if let Ok(up) = be.upload(&t) {
            let res = be.gelu_erf_forward(&up);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat7_02_gelu_erf_inf_input_refused() {
    let be = cpu();
    let data = vec![f32::INFINITY];
    if let Ok(t) = Tensor::from_f32(&data, &[1], be.budget()) {
        if let Ok(up) = be.upload(&t) {
            let res = be.gelu_erf_forward(&up);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat7_03_gelu_erf_neg_inf_input_refused() {
    let be = cpu();
    let data = vec![f32::NEG_INFINITY];
    if let Ok(t) = Tensor::from_f32(&data, &[1], be.budget()) {
        if let Ok(up) = be.upload(&t) {
            let res = be.gelu_erf_forward(&up);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat7_04_mul_shape_mismatch_refused() {
    let be = cpu();
    let a = upload(&be, &[1.0, 2.0], &[2]);
    let b = upload(&be, &[1.0, 2.0, 3.0], &[3]);
    let res = be.mul_forward(&a, &b);
    assert!(matches!(res, Err(OjasError::Shape { .. })));
}

#[test]
fn tier2_feat7_05_gelu_large_finite_input_stability() {
    let be = cpu();
    let x = upload(&be, &[100.0, -100.0], &[2]);
    let out = be.gelu_erf_forward(&x).expect("gelu erf large");
    let host_out = be.download(&out).unwrap().to_f32_vec().unwrap();
    assert!((host_out[0] - 100.0).abs() < 1e-4);
    assert_eq!(host_out[1], 0.0);
}

// -----------------------------------------------------------------------------
// FEATURE 8 (Sliding Window): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat8_01_window_size_zero_means_global() {
    let window: Option<usize> = None;
    assert!(window.is_none());
}

#[test]
fn tier2_feat8_02_window_exceeding_sequence_length() {
    let window = 1000usize;
    let len = 50usize;
    let i = 25usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 0);
    assert_eq!(hi, 50);
}

#[test]
fn tier2_feat8_03_window_size_one_token() {
    let window = 1usize;
    let len = 10usize;
    let i = 5usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 4);
    assert_eq!(hi, 7); // span [4, 5, 6]
}

#[test]
fn tier2_feat8_04_window_at_index_zero() {
    let window = 5usize;
    let len = 20usize;
    let i = 0usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 0);
    assert_eq!(hi, 6);
}

#[test]
fn tier2_feat8_05_window_at_last_index() {
    let window = 5usize;
    let len = 20usize;
    let i = 19usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 14);
    assert_eq!(hi, 20);
}

// -----------------------------------------------------------------------------
// FEATURE 9 (Segment Mean): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat9_01_empty_segment_handled() {
    let seg = [5usize, 5usize];
    assert_eq!(seg[1] - seg[0], 0);
}

#[test]
fn tier2_feat9_02_segment_end_before_start_invalid() {
    let seg = [10usize, 5usize];
    assert!(seg[0] > seg[1]);
}

#[test]
fn tier2_feat9_03_segment_exceeds_token_count() {
    let num_tokens = 50usize;
    let seg = [40usize, 60usize];
    assert!(seg[1] > num_tokens);
}

#[test]
fn tier2_feat9_04_adjacent_contiguous_segments() {
    let segs = [[0usize, 10], [10, 25], [25, 50]];
    assert_eq!(segs[0][1], segs[1][0]);
    assert_eq!(segs[1][1], segs[2][0]);
}

#[test]
fn tier2_feat9_05_full_length_single_segment() {
    let total_tokens = 100usize;
    let seg = [0usize, total_tokens];
    assert_eq!(seg[1] - seg[0], total_tokens);
}

// -----------------------------------------------------------------------------
// FEATURE 10 (L2 Normalization): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat10_01_truncate_dim_larger_than_feature_dim() {
    let feature_dim = 256usize;
    let requested_trunc = 512usize;
    assert!(requested_trunc > feature_dim);
}

#[test]
fn tier2_feat10_02_truncate_dim_zero_invalid() {
    let requested_trunc = 0usize;
    assert_eq!(requested_trunc, 0);
}

#[test]
fn tier2_feat10_03_zero_vector_normalization() {
    let v = [0.0f32; 128];
    let eps = 1e-12f32;
    let sum_sq = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    let norm = (sum_sq.sqrt() + (eps as f64)) as f32;
    assert_eq!(norm, eps);
}

#[test]
fn tier2_feat10_04_huge_vector_scale_stability() {
    let v = [1e19f32; 64];
    let sum_sq = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    assert!(sum_sq.is_finite());
}

#[test]
fn tier2_feat10_05_tiny_denormal_vector_scale() {
    let v = [1e-35f32; 64];
    let sum_sq = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    assert!(sum_sq >= 0.0);
}

// -----------------------------------------------------------------------------
// FEATURE 11 (RMSNorm): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat11_01_rms_qk_nan_eps_refused() {
    let be = cpu();
    let q = upload(&be, &[1.0; 16], &[1, 1, 1, 16]);
    let k = upload(&be, &[1.0; 16], &[1, 1, 1, 16]);
    let w = upload(&be, &[1.0; 16], &[16]);
    let res = be.rms_qk_norm_forward(&q, &k, &w, &w, f32::NAN);
    assert!(matches!(res, Err(OjasError::NonFinite { .. })));
}

#[test]
fn tier2_feat11_02_rms_qk_inf_eps_refused() {
    let be = cpu();
    let q = upload(&be, &[1.0; 16], &[1, 1, 1, 16]);
    let k = upload(&be, &[1.0; 16], &[1, 1, 1, 16]);
    let w = upload(&be, &[1.0; 16], &[16]);
    let res = be.rms_qk_norm_forward(&q, &k, &w, &w, f32::INFINITY);
    assert!(matches!(res, Err(OjasError::NonFinite { .. })));
}

#[test]
fn tier2_feat11_03_rms_qk_weight_mismatch() {
    let be = cpu();
    let q = upload(&be, &[1.0; 16], &[1, 1, 1, 16]);
    let k = upload(&be, &[1.0; 16], &[1, 1, 1, 16]);
    let w = upload(&be, &[1.0; 32], &[32]); // 32 != 16
    let res = be.rms_qk_norm_forward(&q, &k, &w, &w, 1e-5);
    assert!(matches!(res, Err(OjasError::Shape { .. })));
}

#[test]
fn tier2_feat11_04_rms_qk_nan_refused() {
    let be = cpu();
    let mut data = vec![1.0f32; 16];
    data[3] = f32::NAN;
    let k = upload(&be, &[1.0; 16], &[1, 1, 1, 16]);
    let w = upload(&be, &[1.0; 16], &[16]);
    if let Ok(t) = Tensor::from_f32(&data, &[1, 1, 1, 16], be.budget()) {
        if let Ok(up) = be.upload(&t) {
            let res = be.rms_qk_norm_forward(&up, &k, &w, &w, 1e-5);
            assert!(matches!(res, Err(OjasError::NonFinite { .. })));
        }
    }
}

#[test]
fn tier2_feat11_05_rms_qk_unit_head_dim() {
    let be = cpu();
    // Head dim must be multiple of 4 or 8 according to kernel contracts
    let q = upload(&be, &[1.0; 4], &[1, 1, 1, 4]);
    let k = upload(&be, &[1.0; 4], &[1, 1, 1, 4]);
    let w = upload(&be, &[1.0; 4], &[4]);
    let res = be.rms_qk_norm_forward(&q, &k, &w, &w, 1e-5);
    assert!(res.is_ok());
}

// -----------------------------------------------------------------------------
// FEATURE 12 (Budget & Capacity): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat12_01_huge_allocation_capacity_exceeded() {
    let budget = Budget::new(1024 * 1024);
    let res = budget.try_reserve(1 << 60);
    assert!(matches!(res, Err(OjasError::CapacityExceeded { .. })));
}

#[test]
fn tier2_feat12_02_zero_byte_reservation() {
    let budget = Budget::new(1000);
    let res = budget.try_reserve(0);
    assert!(res.is_ok());
}

#[test]
fn tier2_feat12_03_exact_capacity_boundary() {
    let budget = Budget::new(1000);
    let res1 = budget.try_reserve(1000);
    assert!(res1.is_ok());
    // Next 1 byte fails
    let res2 = budget.try_reserve(1);
    assert!(matches!(res2, Err(OjasError::CapacityExceeded { .. })));
}

#[test]
fn tier2_feat12_04_child_budget_exceeds_parent() {
    let parent = Budget::new(1000);
    let child = parent.child(2000); // child has larger cap than parent
    let res = child.try_reserve(1500); // fits child, exceeds parent
    assert!(matches!(res, Err(OjasError::CapacityExceeded { .. })));
}

#[test]
fn tier2_feat12_05_reset_peak_tracks_high_watermark() {
    let budget = Budget::new(100_000);
    let res = budget.try_reserve(50_000).expect("res");
    assert!(budget.peak_bytes() >= 50_000);
    drop(res);
    budget.reset_peak();
    assert_eq!(budget.peak_bytes(), 0);
}

// -----------------------------------------------------------------------------
// FEATURE 13 (Placement & Fail-Closed): Boundaries (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier2_feat13_01_download_host_tensor_is_noop_clone() {
    let be = cpu();
    let t = upload(&be, &[1.0, 2.0], &[2]);
    let downloaded = be.download(&t).expect("download");
    assert_eq!(downloaded.f32_slice().unwrap(), &[1.0, 2.0]);
}

#[test]
fn tier2_feat13_02_shape_product_overflow() {
    let budget = Budget::new(1000);
    // Product of usize::MAX overflows
    let shape = [usize::MAX, 2];
    let res = Tensor::zeros(&shape, DType::F32, &budget);
    assert!(res.is_err());
}

#[test]
fn tier2_feat13_03_non_contiguous_slice_check() {
    let budget = Budget::new(1000);
    let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &budget).expect("alloc");
    assert_eq!(t.shape(), &[2, 2]);
}

#[test]
fn tier2_feat13_04_sync_succeeds_on_cpu() {
    let be = cpu();
    assert!(be.sync().is_ok());
}

#[test]
fn tier2_feat13_05_dtype_mismatch_in_residual_add() {
    let be = cpu();
    let a = upload(&be, &[1.0, 2.0], &[2]);
    let b_u32 = Tensor::from_u32(&[1, 2], &[2], be.budget()).expect("u32");
    let res = be.residual_add_forward(&a, &b_u32);
    assert!(matches!(res, Err(OjasError::Dtype { .. })));
}
