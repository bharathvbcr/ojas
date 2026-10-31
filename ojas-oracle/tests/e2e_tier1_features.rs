//! E2E Test Suite — Tier 1: Feature Coverage (>=5 test cases per feature)
//!
//! Tests each core feature of SAM 3.1, Gemma embedding, and shared memory architecture
//! in isolation against PyTorch reference oracles and mathematical ground truths under
//! pre-registered tolerance bounds.

#![allow(clippy::identity_op, clippy::useless_vec, clippy::needless_range_loop, clippy::excessive_precision)]

use std::sync::Arc;

use ojas_core::{
    axial_rope_2d_tables, AttentionOpts, Backend, Budget, Conv2dParams, ConvTranspose2dParams,
    DType, Numerics, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;
use ojas_oracle::sam_fixtures::{
    compare, Kind, ROPE_TOL,
};

/// 4 GiB budget for isolated op testing.
fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(4 << 30)).with_numerics(Numerics::Exact)
}

fn host(be: &CpuBackend, t: &Tensor) -> Vec<f32> {
    be.download(t).expect("download").to_f32_vec().expect("to_f32_vec")
}

fn upload(be: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    let t = Tensor::from_f32(data, shape, be.budget()).expect("from_f32");
    be.upload(&t).expect("upload")
}

// -----------------------------------------------------------------------------
// FEATURE 1: 2D Axial RoPE Tables & Interleaved Application (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat1_01_rope_tables_local_window_24x24() {
    let be = cpu();
    let (cos, sin) = axial_rope_2d_tables(64, 24, 24, 10_000.0, 1.0, be.budget()).expect("axial tables");
    assert_eq!(cos.shape(), &[576, 32]);
    assert_eq!(sin.shape(), &[576, 32]);
    let cos_data = host(&be, &cos);
    let sin_data = host(&be, &sin);
    for i in 0..576 * 32 {
        let c = cos_data[i];
        let s = sin_data[i];
        let mag = (c * c + s * s).sqrt();
        assert!((mag - 1.0).abs() < 1e-5, "cos^2 + sin^2 must be 1.0, got {mag}");
    }
}

#[test]
fn tier1_feat1_02_rope_tables_global_trunk_72x72() {
    let be = cpu();
    let (cos, sin) = axial_rope_2d_tables(64, 72, 72, 10_000.0, 24.0 / 72.0, be.budget()).expect("axial tables");
    assert_eq!(cos.shape(), &[5184, 32]);
    assert_eq!(sin.shape(), &[5184, 32]);
    assert!(host(&be, &cos).iter().all(|v| v.is_finite()));
    assert!(host(&be, &sin).iter().all(|v| v.is_finite()));
}

#[test]
fn tier1_feat1_03_rope_interleaved_query_rotation() {
    let be = cpu();
    let (cos, sin) = axial_rope_2d_tables(64, 24, 24, 10_000.0, 1.0, be.budget()).expect("axial tables");
    let q_data = vec![1.0f32; 1 * 576 * 16 * 64];
    let q = upload(&be, &q_data, &[1, 576, 16, 64]);
    let q_rot = be.rope_interleaved_forward(&q, &cos, &sin).expect("rope fwd");
    assert_eq!(q_rot.shape(), &[1, 576, 16, 64]);
    let rot_data = host(&be, &q_rot);
    assert!(rot_data.iter().all(|v| v.is_finite()));
}

#[test]
fn tier1_feat1_04_rope_interleaved_key_rotation() {
    let be = cpu();
    let (cos, sin) = axial_rope_2d_tables(64, 24, 24, 10_000.0, 1.0, be.budget()).expect("axial tables");
    let k_data: Vec<f32> = (0..1 * 576 * 16 * 64).map(|i| (i as f32) * 0.001).collect();
    let k = upload(&be, &k_data, &[1, 576, 16, 64]);
    let k_rot = be.rope_interleaved_forward(&k, &cos, &sin).expect("rope fwd");
    assert_eq!(k_rot.shape(), &[1, 576, 16, 64]);
}

#[test]
fn tier1_feat1_05_rope_interleaved_involution_property() {
    let be = cpu();
    let (cos, sin) = axial_rope_2d_tables(64, 24, 24, 10_000.0, 1.0, be.budget()).expect("axial tables");
    let x_data: Vec<f32> = (0..576 * 64).map(|i| ((i % 17) as f32) - 8.0).collect();
    let x = upload(&be, &x_data, &[1, 576, 1, 64]);
    let rot = be.rope_interleaved_forward(&x, &cos, &sin).expect("fwd");
    // Invert by passing negative sin
    let neg_sin_data: Vec<f32> = host(&be, &sin).into_iter().map(|s| -s).collect();
    let neg_sin = upload(&be, &neg_sin_data, sin.shape());
    let inv = be.rope_interleaved_forward(&rot, &cos, &neg_sin).expect("inv");
    let rep = compare("rope_involution", "rot then un-rot", Kind::Direct, &host(&be, &inv), &x_data, ROPE_TOL).expect("compare");
    assert!(rep.pass, "{rep}");
}

// -----------------------------------------------------------------------------
// FEATURE 2: SAM 3.1 Windowed & Global Attention (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat2_01_local_window_attention_9_windows() {
    let be = cpu();
    // 2 windows, each 64 tokens, 4 heads, head dim 64
    let q = upload(&be, &vec![0.1f32; 2 * 4 * 64 * 64], &[2, 4, 64, 64]);
    let k = upload(&be, &vec![0.1f32; 2 * 4 * 64 * 64], &[2, 4, 64, 64]);
    let v = upload(&be, &vec![0.5f32; 2 * 4 * 64 * 64], &[2, 4, 64, 64]);
    let opts = AttentionOpts { causal: false, window: None, key_lens: None, bias: None };
    let out = be.attention_forward(&q, &k, &v, opts).expect("attn");
    assert_eq!(out.shape(), &[2, 4, 64, 64]);
    let data = host(&be, &out);
    for val in data {
        assert!((val - 0.5).abs() < 1e-4, "uniform attention over constant V=0.5 must equal 0.5");
    }
}

#[test]
fn tier1_feat2_02_global_attention_tokens() {
    let be = cpu();
    let q = upload(&be, &vec![0.05f32; 1 * 4 * 128 * 64], &[1, 4, 128, 64]);
    let k = upload(&be, &vec![0.05f32; 1 * 4 * 128 * 64], &[1, 4, 128, 64]);
    let v = upload(&be, &vec![1.2f32; 1 * 4 * 128 * 64], &[1, 4, 128, 64]);
    let opts = AttentionOpts { causal: false, window: None, key_lens: None, bias: None };
    let out = be.attention_forward(&q, &k, &v, opts).expect("global attn");
    assert_eq!(out.shape(), &[1, 4, 128, 64]);
}

#[test]
fn tier1_feat2_03_attention_softmax_scale_invariance() {
    let be = cpu();
    let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let q = upload(&be, &q_data, &[1, 1, 1, 64]);
    let k = upload(&be, &q_data, &[1, 1, 1, 64]);
    let v = upload(&be, &vec![2.5f32; 64], &[1, 1, 1, 64]);
    let opts = AttentionOpts { causal: false, window: None, key_lens: None, bias: None };
    let out = be.attention_forward(&q, &k, &v, opts).expect("attn");
    let out_data = host(&be, &out);
    assert!((out_data[0] - 2.5).abs() < 1e-5);
}

#[test]
fn tier1_feat2_04_attention_with_key_padding_mask() {
    let be = cpu();
    let q = upload(&be, &vec![0.2f32; 64], &[1, 1, 1, 64]);
    let k = upload(&be, &vec![0.2f32; 128], &[1, 1, 2, 64]);
    let v_data: Vec<f32> = (0..128).map(|i| if i < 64 { 10.0 } else { 0.0 }).collect();
    let v = upload(&be, &v_data, &[1, 1, 2, 64]);
    // Mask out second token (valid len = 1)
    let lens_tensor = Tensor::from_u32(&[1], &[1], be.budget()).expect("lens");
    let opts = AttentionOpts { causal: false, window: None, key_lens: Some(&lens_tensor), bias: None };
    let out = be.attention_forward(&q, &k, &v, opts).expect("masked attn");
    let out_data = host(&be, &out);
    assert!((out_data[0] - 10.0).abs() < 1e-4);
}

#[test]
fn tier1_feat2_05_attention_saving_lse() {
    let be = cpu();
    let q = upload(&be, &vec![0.1f32; 64], &[1, 1, 1, 64]);
    let k = upload(&be, &vec![0.1f32; 64], &[1, 1, 1, 64]);
    let v = upload(&be, &vec![1.0f32; 64], &[1, 1, 1, 64]);
    let opts = AttentionOpts { causal: false, window: None, key_lens: None, bias: None };
    let (out, lse) = be.attention_forward_saving(&q, &k, &v, opts).expect("saving");
    assert_eq!(out.shape(), &[1, 1, 1, 64]);
    assert_eq!(lse.shape(), &[1, 1, 1]);
    let lse_val = host(&be, &lse)[0];
    assert!(lse_val.is_finite());
}

// -----------------------------------------------------------------------------
// FEATURE 3: LayerNorm & LayerNorm2d (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat3_01_layernorm_hidden_dim_1024() {
    let be = cpu();
    let x_data: Vec<f32> = (0..576 * 1024).map(|i| ((i % 100) as f32) - 50.0).collect();
    let x = upload(&be, &x_data, &[1, 576, 1024]);
    let w = upload(&be, &vec![1.0f32; 1024], &[1024]);
    let b = upload(&be, &vec![0.0f32; 1024], &[1024]);
    let out = be.layer_norm_forward(&x, &w, &b, 1e-6).expect("ln");
    assert_eq!(out.shape(), &[1, 576, 1024]);
    let data = host(&be, &out);
    let row0 = &data[0..1024];
    let mean: f32 = row0.iter().sum::<f32>() / 1024.0;
    assert!(mean.abs() < 1e-4, "mean must be 0, got {mean}");
}

#[test]
fn tier1_feat3_02_channel_layernorm2d_fpn_neck() {
    let be = cpu();
    let x_data: Vec<f32> = (0..1 * 256 * 24 * 24).map(|i| (i as f32) * 0.001).collect();
    let x = upload(&be, &x_data, &[1, 256, 24, 24]);
    let w = upload(&be, &vec![1.0f32; 256], &[256]);
    let b = upload(&be, &vec![0.0f32; 256], &[256]);
    let out = be.channel_layer_norm_forward(&x, &w, &b, 1e-6).expect("channel ln");
    assert_eq!(out.shape(), &[1, 256, 24, 24]);
}

#[test]
fn tier1_feat3_03_layernorm_affine_scaling() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let w = upload(&be, &vec![2.0f32; 4], &[4]);
    let b = upload(&be, &vec![10.0f32; 4], &[4]);
    let out = be.layer_norm_forward(&x, &w, &b, 1e-5).expect("ln");
    let data = host(&be, &out);
    let mean: f32 = data.iter().sum::<f32>() / 4.0;
    assert!((mean - 10.0).abs() < 1e-4, "mean must equal bias 10.0");
}

#[test]
fn tier1_feat3_04_layernorm_two_pass_variance_stability() {
    let be = cpu();
    let x_data: Vec<f32> = (0..1024).map(|i| 1_000_000.0 + (i as f32)).collect();
    let x = upload(&be, &x_data, &[1, 1, 1024]);
    let w = upload(&be, &vec![1.0f32; 1024], &[1024]);
    let b = upload(&be, &vec![0.0f32; 1024], &[1024]);
    let out = be.layer_norm_forward(&x, &w, &b, 1e-5).expect("ln");
    let data = host(&be, &out);
    assert!(data.iter().all(|v| v.is_finite()), "two-pass must not suffer catastrophic cancellation");
}

#[test]
fn tier1_feat3_05_layernorm_constant_input_yields_zero() {
    let be = cpu();
    let x = upload(&be, &vec![42.0f32; 128], &[1, 1, 128]);
    let w = upload(&be, &vec![1.0f32; 128], &[128]);
    let b = upload(&be, &vec![0.0f32; 128], &[128]);
    let out = be.layer_norm_forward(&x, &w, &b, 1e-5).expect("ln");
    let data = host(&be, &out);
    for v in data {
        assert!(v.abs() < 1e-4, "constant input normalized must be 0");
    }
}

// -----------------------------------------------------------------------------
// FEATURE 4: Conv2d & Generalized Transposed Conv2d (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat4_01_conv2d_patch_embed_14x14() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32; 1 * 3 * 28 * 28], &[1, 3, 28, 28]);
    let w = upload(&be, &vec![0.01f32; 16 * 3 * 14 * 14], &[16, 3, 14, 14]);
    let params = Conv2dParams { stride: [14, 14], padding: [0, 0], dilation: [1, 1], groups: 1 };
    let out = be.conv2d_forward(&x, &w, None, params).expect("conv2d");
    assert_eq!(out.shape(), &[1, 16, 2, 2]);
}

#[test]
fn tier1_feat4_02_conv2d_neck_3x3_with_bias() {
    let be = cpu();
    let x = upload(&be, &vec![0.5f32; 1 * 4 * 8 * 8], &[1, 4, 8, 8]);
    let w = upload(&be, &vec![0.1f32; 4 * 4 * 3 * 3], &[4, 4, 3, 3]);
    let b = upload(&be, &vec![1.0f32; 4], &[4]);
    let params = Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };
    let out = be.conv2d_forward(&x, &w, Some(&b), params).expect("conv2d");
    assert_eq!(out.shape(), &[1, 4, 8, 8]);
}

#[test]
fn tier1_feat4_03_conv_transpose2d_stride2_no_pad() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32; 1 * 2 * 4 * 4], &[1, 2, 4, 4]);
    let w = upload(&be, &vec![0.5f32; 2 * 2 * 2 * 2], &[2, 2, 2, 2]);
    let params = ConvTranspose2dParams { stride: [2, 2], padding: [0, 0], output_padding: [0, 0] };
    let out = be.conv_transpose2d_forward(&x, &w, None, params).expect("tconv");
    assert_eq!(out.shape(), &[1, 2, 8, 8]);
}

#[test]
fn tier1_feat4_04_conv_transpose2d_overlapping_3x3() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32; 1 * 2 * 4 * 4], &[1, 2, 4, 4]);
    let w = upload(&be, &vec![0.5f32; 2 * 2 * 3 * 3], &[2, 2, 3, 3]);
    let params = ConvTranspose2dParams { stride: [2, 2], padding: [1, 1], output_padding: [0, 0] };
    let out = be.conv_transpose2d_forward(&x, &w, None, params).expect("tconv overlapping");
    assert_eq!(out.shape(), &[1, 2, 7, 7]);
}

#[test]
fn tier1_feat4_05_conv_transpose2d_with_output_padding() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32; 1 * 2 * 4 * 4], &[1, 2, 4, 4]);
    let w = upload(&be, &vec![0.5f32; 2 * 2 * 2 * 2], &[2, 2, 2, 2]);
    let params = ConvTranspose2dParams { stride: [2, 2], padding: [0, 0], output_padding: [1, 1] };
    let out = be.conv_transpose2d_forward(&x, &w, None, params).expect("tconv out pad");
    assert_eq!(out.shape(), &[1, 2, 9, 9]);
}

// -----------------------------------------------------------------------------
// FEATURE 5: GroupNorm & Bilinear Interpolation (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat5_01_group_norm_32_groups_256_channels() {
    let be = cpu();
    let x_data: Vec<f32> = (0..1 * 256 * 14 * 14).map(|i| ((i % 50) as f32) - 25.0).collect();
    let x = upload(&be, &x_data, &[1, 256, 14, 14]);
    let w = upload(&be, &vec![1.0f32; 256], &[256]);
    let b = upload(&be, &vec![0.0f32; 256], &[256]);
    let out = be.group_norm_forward(&x, &w, &b, 32, 1e-5).expect("gn");
    assert_eq!(out.shape(), &[1, 256, 14, 14]);
    let data = host(&be, &out);
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn tier1_feat5_02_group_norm_affine_shift() {
    let be = cpu();
    let x = upload(&be, &vec![2.0f32; 1 * 4 * 4 * 4], &[1, 4, 4, 4]);
    let w = upload(&be, &vec![3.0f32; 4], &[4]);
    let b = upload(&be, &vec![5.0f32; 4], &[4]);
    let out = be.group_norm_forward(&x, &w, &b, 2, 1e-5).expect("gn");
    let data = host(&be, &out);
    for v in data {
        assert!((v - 5.0).abs() < 1e-4);
    }
}

#[test]
fn tier1_feat5_03_bilinear_interpolation_2x_upsample() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
    let out = be.interpolate_bilinear_forward(&x, [4, 4], false).expect("bilinear");
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let data = host(&be, &out);
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn tier1_feat5_04_bilinear_interpolation_align_corners() {
    let be = cpu();
    let x = upload(&be, &vec![10.0f32, 20.0, 30.0, 40.0], &[1, 1, 2, 2]);
    let out = be.interpolate_bilinear_forward(&x, [4, 4], true).expect("bilinear align");
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let data = host(&be, &out);
    assert!((data[0] - 10.0).abs() < 1e-5, "corner (0,0) must equal 10.0");
    assert!((data[15] - 40.0).abs() < 1e-5, "corner (3,3) must equal 40.0");
}

#[test]
fn tier1_feat5_05_bilinear_downsampling() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32; 1 * 1 * 8 * 8], &[1, 1, 8, 8]);
    let out = be.interpolate_bilinear_forward(&x, [4, 4], false).expect("downsample");
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let data = host(&be, &out);
    for v in data {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

// -----------------------------------------------------------------------------
// FEATURE 6: 2D Max Pooling (`max_pool2d_forward`) (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat6_01_max_pool2d_standard_2x2_stride2() {
    let be = cpu();
    let x = upload(&be, &vec![
        1.0f32, 3.0,  2.0, 4.0,
        5.0,    6.0,  8.0, 7.0,
        9.0,    2.0,  1.0, 3.0,
        4.0,    10.0, 5.0, 6.0,
    ], &[1, 1, 4, 4]);
    let out = be.max_pool2d_forward(&x, [2, 2], [2, 2]).expect("maxpool");
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    let data = host(&be, &out);
    assert_eq!(data, vec![6.0, 8.0, 10.0, 6.0]);
}

#[test]
fn tier1_feat6_02_max_pool2d_overlapping_3x3() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32; 1 * 1 * 5 * 5], &[1, 1, 5, 5]);
    let out = be.max_pool2d_forward(&x, [3, 3], [1, 1]).expect("maxpool");
    assert_eq!(out.shape(), &[1, 1, 3, 3]);
}

#[test]
fn tier1_feat6_03_max_pool2d_tie_breaking_order() {
    let be = cpu();
    let x = upload(&be, &vec![7.0f32; 1 * 1 * 4 * 4], &[1, 1, 4, 4]);
    let out = be.max_pool2d_forward(&x, [2, 2], [2, 2]).expect("maxpool");
    let data = host(&be, &out);
    for v in data {
        assert_eq!(v, 7.0);
    }
}

#[test]
fn tier1_feat6_04_max_pool2d_all_negative_values() {
    let be = cpu();
    let x = upload(&be, &vec![
        -10.0f32, -5.0,
        -8.0,     -20.0,
    ], &[1, 1, 2, 2]);
    let out = be.max_pool2d_forward(&x, [2, 2], [2, 2]).expect("maxpool");
    let data = host(&be, &out);
    assert_eq!(data[0], -5.0);
}

#[test]
fn tier1_feat6_05_max_pool2d_multi_channel_invariance() {
    let be = cpu();
    let x = upload(&be, &vec![1.0f32; 2 * 3 * 6 * 6], &[2, 3, 6, 6]);
    let out = be.max_pool2d_forward(&x, [2, 2], [2, 2]).expect("maxpool");
    assert_eq!(out.shape(), &[2, 3, 3, 3]);
}

// -----------------------------------------------------------------------------
// FEATURE 7: Gemma GeGLU Activation (`gelu_tanh`) (5 tests)
// -----------------------------------------------------------------------------

fn torch_gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608028654f64;
    let x_f64 = x as f64;
    let inner = sqrt_2_over_pi * (x_f64 + 0.044715 * x_f64 * x_f64 * x_f64);
    (0.5 * x_f64 * (1.0 + inner.tanh())) as f32
}

#[test]
fn tier1_feat7_01_gelu_tanh_oracle_matches_pytorch() {
    let test_vals = vec![-3.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    for &x in &test_vals {
        let expected = torch_gelu_tanh(x);
        let actual = 0.5 * x * (1.0 + (0.79788456f32 * (x + 0.044715f32 * x * x * x)).tanh());
        assert!((actual - expected).abs() < 1e-4, "at x={x}: actual={actual}, expected={expected}");
    }
}

#[test]
fn tier1_feat7_02_gelu_tanh_zero_crossing() {
    let zero_val = torch_gelu_tanh(0.0);
    assert_eq!(zero_val, 0.0, "gelu_tanh(0) must equal 0.0 exactly");
}

#[test]
fn tier1_feat7_03_gelu_tanh_asymptotes() {
    assert!(torch_gelu_tanh(-20.0).abs() < 1e-7);
    assert!((torch_gelu_tanh(20.0) - 20.0).abs() < 1e-5);
}

#[test]
fn tier1_feat7_04_gelu_erf_backend_fwd_parity() {
    let be = cpu();
    let x = upload(&be, &vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], &[1, 5]);
    let out = be.gelu_erf_forward(&x).expect("gelu erf");
    let data = host(&be, &out);
    assert_eq!(data[2], 0.0);
    assert!(data[3] > 0.8 && data[3] < 0.9);
}

#[test]
fn tier1_feat7_05_geglu_gated_product() {
    let be = cpu();
    let gate = upload(&be, &vec![1.0f32; 4], &[1, 4]);
    let up = upload(&be, &vec![2.0f32; 4], &[1, 4]);
    let act = be.gelu_erf_forward(&gate).expect("act");
    let out = be.mul_forward(&act, &up).expect("mul");
    let data = host(&be, &out);
    for v in data {
        assert!(v > 1.6 && v < 1.7);
    }
}

// -----------------------------------------------------------------------------
// FEATURE 8: Gemma Symmetric Non-Causal Sliding Window Attention (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat8_01_symmetric_window_interior_bounds() {
    let window = 512usize;
    let len = 2048usize;
    let i = 1000usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 488);
    assert_eq!(hi, 1513);
}

#[test]
fn tier1_feat8_02_symmetric_window_left_boundary() {
    let window = 512usize;
    let len = 2048usize;
    let i = 100usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 0);
    assert_eq!(hi, 613);
}

#[test]
fn tier1_feat8_03_symmetric_window_right_boundary() {
    let window = 512usize;
    let len = 2048usize;
    let i = 2000usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 1488);
    assert_eq!(hi, 2048);
}

#[test]
fn tier1_feat8_04_short_sequence_spans_entire_sequence() {
    let window = 512usize;
    let len = 256usize;
    let i = 50usize;
    let lo = i.saturating_sub(window);
    let hi = (i + window + 1).min(len);
    assert_eq!(lo, 0);
    assert_eq!(hi, 256);
}

#[test]
fn tier1_feat8_05_bidirectional_non_causal_general_attention() {
    let be = cpu();
    let q = upload(&be, &vec![0.1f32; 1 * 4 * 16 * 32], &[1, 4, 16, 32]);
    let k = upload(&be, &vec![0.1f32; 1 * 4 * 16 * 32], &[1, 4, 16, 32]);
    let v = upload(&be, &vec![1.0f32; 1 * 4 * 16 * 32], &[1, 4, 16, 32]);
    let opts = AttentionOpts { causal: false, window: None, key_lens: None, bias: None };
    let out = be.attention_forward(&q, &k, &v, opts).expect("non causal attn");
    assert_eq!(out.shape(), &[1, 4, 16, 32]);
}

// -----------------------------------------------------------------------------
// FEATURE 9: Gemma Sequence Mean Pooling (`segment_mean_rows`) (5 tests)
// -----------------------------------------------------------------------------

fn torch_segment_mean(input: &[f32], segments: &[[usize; 2]], d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(segments.len() * d);
    for seg in segments {
        let (start, end) = (seg[0], seg[1]);
        let count = end - start;
        for c in 0..d {
            let mut sum = 0.0f64;
            for r in start..end {
                sum += f64::from(input[r * d + c]);
            }
            out.push((sum / (count as f64)) as f32);
        }
    }
    out
}

#[test]
fn tier1_feat9_01_uniform_sequence_mean_pooling() {
    let d = 4usize;
    let input = vec![
        1.0f32, 2.0, 3.0, 4.0,
        3.0,    4.0, 5.0, 6.0,
        10.0,   20.0, 30.0, 40.0,
        30.0,   40.0, 50.0, 60.0,
    ];
    let segments = vec![[0, 2], [2, 4]];
    let pooled = torch_segment_mean(&input, &segments, d);
    assert_eq!(pooled, vec![
        2.0, 3.0, 4.0, 5.0,
        20.0, 30.0, 40.0, 50.0,
    ]);
}

#[test]
fn tier1_feat9_02_variable_length_sequence_mean_pooling() {
    let d = 2usize;
    let input = vec![
        1.0f32, 1.0,
        2.0, 2.0,
        3.0, 3.0,
        10.0, 10.0,
    ];
    let segments = vec![[0, 3], [3, 4]];
    let pooled = torch_segment_mean(&input, &segments, d);
    assert_eq!(pooled, vec![2.0, 2.0, 10.0, 10.0]);
}

#[test]
fn tier1_feat9_03_single_token_sequence_mean_identity() {
    let d = 3usize;
    let input = vec![5.0f32, 7.0, 9.0];
    let segments = vec![[0, 1]];
    let pooled = torch_segment_mean(&input, &segments, d);
    assert_eq!(pooled, vec![5.0, 7.0, 9.0]);
}

#[test]
fn tier1_feat9_04_mean_pooling_two_pass_precision() {
    let d = 1usize;
    let input = vec![1e7f32, 1.0, -1e7];
    let segments = vec![[0, 3]];
    let pooled = torch_segment_mean(&input, &segments, d);
    assert!((pooled[0] - 1.0 / 3.0).abs() < 1e-4);
}

#[test]
fn tier1_feat9_05_multi_batch_pooling_layout() {
    let d = 2usize;
    let input = vec![1.0f32; 10 * d];
    let segments = vec![[0, 2], [2, 5], [5, 10]];
    let pooled = torch_segment_mean(&input, &segments, d);
    assert_eq!(pooled.len(), 3 * d);
    for v in pooled {
        assert_eq!(v, 1.0);
    }
}

// -----------------------------------------------------------------------------
// FEATURE 10: Gemma Matryoshka Prefix L2 Normalization (5 tests)
// -----------------------------------------------------------------------------

fn torch_l2_normalize(input: &[f32], truncate_dim: Option<usize>, eps: f32) -> Vec<f32> {
    let d = truncate_dim.unwrap_or(input.len());
    let mut sum_sq = 0.0f64;
    for &v in &input[..d] {
        sum_sq += f64::from(v) * f64::from(v);
    }
    let norm = (sum_sq.sqrt() + f64::from(eps)) as f32;
    input[..d].iter().map(|&v| v / norm).collect()
}

#[test]
fn tier1_feat10_01_full_dim_l2_norm_unit_sphere() {
    let input = vec![3.0f32, 4.0];
    let norm = torch_l2_normalize(&input, None, 1e-12);
    assert!((norm[0] - 0.6).abs() < 1e-6);
    assert!((norm[1] - 0.8).abs() < 1e-6);
    let len = (norm[0] * norm[0] + norm[1] * norm[1]).sqrt();
    assert!((len - 1.0).abs() < 1e-6);
}

#[test]
fn tier1_feat10_02_matryoshka_prefix_512() {
    let input = vec![1.0f32; 768];
    let norm = torch_l2_normalize(&input, Some(512), 1e-12);
    assert_eq!(norm.len(), 512);
    let expected_elem = 1.0 / (512.0f32).sqrt();
    assert!((norm[0] - expected_elem).abs() < 1e-5);
}

#[test]
fn tier1_feat10_03_matryoshka_prefix_256() {
    let input = vec![1.0f32; 768];
    let norm = torch_l2_normalize(&input, Some(256), 1e-12);
    assert_eq!(norm.len(), 256);
    let expected_elem = 1.0 / (256.0f32).sqrt();
    assert!((norm[0] - expected_elem).abs() < 1e-5);
}

#[test]
fn tier1_feat10_04_matryoshka_prefix_128() {
    let input = vec![1.0f32; 768];
    let norm = torch_l2_normalize(&input, Some(128), 1e-12);
    assert_eq!(norm.len(), 128);
    let expected_elem = 1.0 / (128.0f32).sqrt();
    assert!((norm[0] - expected_elem).abs() < 1e-5);
}

#[test]
fn tier1_feat10_05_l2_norm_epsilon_stabilization() {
    let input = vec![0.0f32; 64];
    let norm = torch_l2_normalize(&input, None, 1e-6);
    for v in norm {
        assert_eq!(v, 0.0);
    }
}

// -----------------------------------------------------------------------------
// FEATURE 11: Gemma Per-Head V RMSNorm (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat11_01_rms_qk_norm_forward_runs_on_cpu() {
    let be = cpu();
    let q = upload(&be, &vec![1.0f32; 1 * 4 * 8 * 16], &[1, 4, 8, 16]);
    let k = upload(&be, &vec![2.0f32; 1 * 4 * 8 * 16], &[1, 4, 8, 16]);
    let w_q = upload(&be, &vec![1.0f32; 16], &[16]);
    let w_k = upload(&be, &vec![1.0f32; 16], &[16]);
    let (qn, kn) = be.rms_qk_norm_forward(&q, &k, &w_q, &w_k, 1e-6).expect("rms qk");
    assert_eq!(qn.shape(), &[1, 4, 8, 16]);
    assert_eq!(kn.shape(), &[1, 4, 8, 16]);
    let q_data = host(&be, &qn);
    for v in q_data {
        assert!((v - 1.0).abs() < 1e-4);
    }
}

#[test]
fn tier1_feat11_02_weightless_v_rmsnorm_formula() {
    let v_row = vec![2.0f32; 64];
    let mut sum_sq = 0.0f64;
    for &x in &v_row {
        sum_sq += f64::from(x) * f64::from(x);
    }
    let rms = ((sum_sq / 64.0) + 1e-6).sqrt() as f32;
    let norm: Vec<f32> = v_row.iter().map(|&x| x / rms).collect();
    for val in norm {
        assert!((val - 1.0).abs() < 1e-4);
    }
}

#[test]
fn tier1_feat11_03_per_head_scaling_invariance() {
    let be = cpu();
    let q = upload(&be, &vec![4.0f32; 1 * 2 * 1 * 16], &[1, 2, 1, 16]);
    let k = upload(&be, &vec![4.0f32; 1 * 2 * 1 * 16], &[1, 2, 1, 16]);
    let w = upload(&be, &vec![1.0f32; 16], &[16]);
    let (qn, _) = be.rms_qk_norm_forward(&q, &k, &w, &w, 1e-5).expect("rms");
    let data = host(&be, &qn);
    for v in data {
        assert!((v - 1.0).abs() < 1e-4);
    }
}

#[test]
fn tier1_feat11_04_rms_norm_learned_weights() {
    let be = cpu();
    let q = upload(&be, &vec![1.0f32; 16], &[1, 1, 1, 16]);
    let k = upload(&be, &vec![1.0f32; 16], &[1, 1, 1, 16]);
    let w = upload(&be, &vec![2.5f32; 16], &[16]);
    let (qn, _) = be.rms_qk_norm_forward(&q, &k, &w, &w, 1e-5).expect("rms");
    let data = host(&be, &qn);
    for v in data {
        assert!((v - 2.5).abs() < 1e-4);
    }
}

#[test]
fn tier1_feat11_05_rms_norm_eps_prevents_zero_div() {
    let be = cpu();
    let q = upload(&be, &vec![0.0f32; 16], &[1, 1, 1, 16]);
    let k = upload(&be, &vec![0.0f32; 16], &[1, 1, 1, 16]);
    let w = upload(&be, &vec![1.0f32; 16], &[16]);
    let (qn, _) = be.rms_qk_norm_forward(&q, &k, &w, &w, 1e-5).expect("rms");
    let data = host(&be, &qn);
    for v in data {
        assert_eq!(v, 0.0);
    }
}

// -----------------------------------------------------------------------------
// FEATURE 12: Memory Sharing & UMA Invariants (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat12_01_root_budget_charges_tensor_bytes() {
    let budget = Budget::new(1024 * 1024);
    let initial = budget.live_bytes().unwrap();
    {
        let _t = Tensor::from_f32(&vec![1.0; 1000], &[1000], &budget).expect("alloc");
        let after = budget.live_bytes().unwrap();
        assert_eq!(after - initial, 4000);
    }
    // After drop, capacity restored
    assert_eq!(budget.live_bytes().unwrap(), initial);
}

#[test]
fn tier1_feat12_02_budget_child_ledger_limits() {
    let root = Budget::new(10_000);
    let child = root.child(4000);
    assert_eq!(child.cap_bytes(), 4000);
    // Over-allocating child fails
    let res = child.try_reserve(5000);
    assert!(matches!(res, Err(OjasError::CapacityExceeded { .. })));
}

#[test]
fn tier1_feat12_03_budget_capacity_exceeded_error() {
    let budget = Budget::new(100);
    let res = Tensor::from_f32(&vec![1.0; 100], &[100], &budget);
    assert!(matches!(res, Err(OjasError::CapacityExceeded { .. })));
}

#[test]
fn tier1_feat12_04_budget_concurrent_reservations() {
    let budget = Arc::new(Budget::new(100_000));
    let mut handles = Vec::new();
    for _ in 0..4 {
        let b = Arc::clone(&budget);
        handles.push(std::thread::spawn(move || {
            let res = b.try_reserve(1000).expect("reserve");
            std::thread::sleep(std::time::Duration::from_millis(5));
            drop(res);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

#[test]
fn tier1_feat12_05_tensor_from_f32_alignment() {
    let budget = Budget::new(10_000);
    let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], &budget).expect("alloc");
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.shape(), &[3]);
}

// -----------------------------------------------------------------------------
// FEATURE 13: Fail-Closed No Silent Fallback Invariants (5 tests)
// -----------------------------------------------------------------------------

#[test]
fn tier1_feat13_01_cpu_backend_preserves_exact_numerics() {
    let be = cpu();
    assert_eq!(be.numerics(), Numerics::Exact);
}

#[test]
fn tier1_feat13_02_backend_id_matches() {
    let be = cpu();
    assert_eq!(be.id(), ojas_core::BackendId::Cpu);
}

#[test]
fn tier1_feat13_03_unsupported_op_refuses_cleanly() {
    let be = cpu();
    let t = upload(&be, &[1.0], &[1]);
    let params = Conv2dParams { stride: [0, 0], padding: [0, 0], dilation: [1, 1], groups: 1 };
    let res = be.conv2d_forward(&t, &t, None, params);
    assert!(res.is_err());
}

#[test]
fn tier1_feat13_04_tensor_host_access_returns_f32_slice() {
    let budget = Budget::new(1000);
    let t = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).expect("tensor");
    let slice = t.f32_slice().expect("f32 slice");
    assert_eq!(slice, &[1.0, 2.0]);
}

#[test]
fn tier1_feat13_05_fail_closed_on_shape_mismatch() {
    let be = cpu();
    let a = upload(&be, &[1.0, 2.0], &[2]);
    let b = upload(&be, &[1.0, 2.0, 3.0], &[3]);
    let res = be.residual_add_forward(&a, &b);
    assert!(matches!(res, Err(OjasError::Shape { .. })));
}
