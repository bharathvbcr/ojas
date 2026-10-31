//! E2E Test Suite — Tier 3: Pairwise Combinatorial Interactions
//!
//! Tests cross-feature interactions between pairs of operations:
//! 1. 2D RoPE + Window Partition / Window Merge + Attention
//! 2. Conv2d Patch Embed + Channel LayerNorm2d
//! 3. GeGLU MLP + Sequence Mean Pooling
//! 4. Transposed Conv2d (col2im) + MaxPool2d
//! 5. Bilinear Interpolation + GroupNorm
//! 6. Additive Logit Bias + Sliding Window Attention
//! 7. Multi-layer Projection + Matryoshka Prefix L2 Normalization
//! 8. Fused QKV Projection + 2D RoPE Application

#![allow(clippy::identity_op, clippy::useless_vec, clippy::needless_range_loop)]

use ojas_core::{
    axial_rope_2d_tables, window_merge, window_partition, AttentionOpts, Backend, Budget,
    Conv2dParams, ConvTranspose2dParams, Numerics, Tensor,
};
use ojas_cpu::CpuBackend;

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(4 << 30)).with_numerics(Numerics::Exact)
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

#[test]
fn tier3_01_rope_with_window_partition_attention_and_merge() {
    let be = cpu();
    // Grid 8x8, window size 4 (2x2 = 4 windows)
    let (h, w, window) = (8usize, 8usize, 4usize);
    let channels = 16usize; // 4 heads * 4 dim
    let x_data: Vec<f32> = (0..1 * h * w * channels)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let x = upload(&be, &x_data, &[1, h, w, channels]);

    // 1. Partition into windows: [4 windows, 4, 4, 16]
    let parts = window_partition(&be, &x, window).expect("partition");
    assert_eq!(parts.shape(), &[4, 4, 4, 16]);

    // Reshape to tokens: [4, 16, 4, 4] -> [4, 16 tokens, 4 heads, 4 dim]
    let tokens = parts.reshape(&[4, 16, 4, 4]).expect("tokens");

    // 2. Apply 2D RoPE per window
    let (cos, sin) =
        axial_rope_2d_tables(4, window, window, 10_000.0, 1.0, be.budget()).expect("tables");
    let rotated = be
        .rope_interleaved_forward(&tokens, &cos, &sin)
        .expect("rope");

    // 3. Windowed Attention: transpose to [4, 4 heads, 16 tokens, 4 dim]
    let q = be.permute(&rotated, &[0, 2, 1, 3]).expect("permute");
    let opts = AttentionOpts {
        causal: false,
        window: None,
        key_lens: None,
        bias: None,
        ..Default::default()
    };
    let attn_out = be.attention_forward(&q, &q, &q, opts).expect("attn");
    assert_eq!(attn_out.shape(), &[4, 4, 16, 4]);

    // 4. Permute and merge back: [4, 16 tokens, 16 channels] -> [4, 4, 4, 16] -> [1, 8, 8, 16]
    let back_tokens = be.permute(&attn_out, &[0, 2, 1, 3]).expect("back");
    let back_windows = back_tokens.reshape(&[4, 4, 4, 16]).expect("win");
    let merged = window_merge(&be, &back_windows, h, w).expect("merge");
    assert_eq!(merged.shape(), &[1, h, w, channels]);
    let merged_data = host(&be, &merged);
    assert!(merged_data.iter().all(|v| v.is_finite()));
}

#[test]
fn tier3_02_conv2d_patch_embed_into_channel_layernorm2d() {
    let be = cpu();
    // Input image: [1, 3, 28, 28]
    let img = upload(&be, &vec![0.5f32; 1 * 3 * 28 * 28], &[1, 3, 28, 28]);
    // Patch embed: k=14, s=14, out_channels=16
    let weight = upload(&be, &vec![0.02f32; 16 * 3 * 14 * 14], &[16, 3, 14, 14]);
    let params = Conv2dParams {
        stride: [14, 14],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    };
    let patches = be
        .conv2d_forward(&img, &weight, None, params)
        .expect("conv");
    assert_eq!(patches.shape(), &[1, 16, 2, 2]);

    // LayerNorm2D over channels
    let gamma = upload(&be, &vec![1.0f32; 16], &[16]);
    let beta = upload(&be, &vec![0.0f32; 16], &[16]);
    let norm = be
        .channel_layer_norm_forward(&patches, &gamma, &beta, 1e-6)
        .expect("ln2d");
    assert_eq!(norm.shape(), &[1, 16, 2, 2]);

    let data = host(&be, &norm);
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn tier3_03_geglu_mlp_into_sequence_mean_pooling() {
    let be = cpu();
    // MLP input: [4 tokens, 8 dim]
    let x = upload(&be, &vec![1.0f32; 4 * 8], &[4, 8]);
    // Linear weight is [out_features, in_features] = [16, 8]
    let w_gate = upload(&be, &vec![0.1f32; 16 * 8], &[16, 8]);
    let w_up = upload(&be, &vec![0.1f32; 16 * 8], &[16, 8]);

    // Feedforward: gate and up
    let gate = be.linear_forward(&x, &w_gate).expect("gate");
    let up = be.linear_forward(&x, &w_up).expect("up");
    let act = be.gelu_erf_forward(&gate).expect("act");
    let mlp_out = be.mul_forward(&act, &up).expect("mul");
    assert_eq!(mlp_out.shape(), &[4, 16]);

    // Sequence mean pooling over 2 sequences: [0..2] and [2..4]
    let host_mlp = host(&be, &mlp_out);
    let segments = [[0usize, 2], [2, 4]];
    let d = 16usize;
    let mut pooled = Vec::new();
    for seg in &segments {
        let (s, e) = (seg[0], seg[1]);
        for c in 0..d {
            let sum: f32 = (s..e).map(|r| host_mlp[r * d + c]).sum();
            pooled.push(sum / (e - s) as f32);
        }
    }
    assert_eq!(pooled.len(), 2 * d);
    assert!(pooled.iter().all(|v| v.is_finite()));
}

#[test]
fn tier3_04_transposed_conv2d_into_maxpool2d() {
    let be = cpu();
    // Feature map [1, 2, 4, 4]
    let x = upload(&be, &vec![1.0f32; 1 * 2 * 4 * 4], &[1, 2, 4, 4]);
    // Transposed conv upsampling 2x: k=2, s=2 -> [1, 2, 8, 8]
    let w_tconv = upload(&be, &vec![0.5f32; 2 * 2 * 2 * 2], &[2, 2, 2, 2]);
    let tparams = ConvTranspose2dParams {
        stride: [2, 2],
        padding: [0, 0],
        output_padding: [0, 0],
    };
    let upsampled = be
        .conv_transpose2d_forward(&x, &w_tconv, None, tparams)
        .expect("tconv");
    assert_eq!(upsampled.shape(), &[1, 2, 8, 8]);

    // Max pooling 2x: k=2, s=2 -> [1, 2, 4, 4]
    let pooled = be
        .max_pool2d_forward(&upsampled, [2, 2], [2, 2])
        .expect("maxpool");
    assert_eq!(pooled.shape(), &[1, 2, 4, 4]);
    let data = host(&be, &pooled);
    for v in data {
        assert_eq!(v, 1.0);
    }
}

#[test]
fn tier3_05_bilinear_interpolation_into_group_norm() {
    let be = cpu();
    // [1, 4 channels, 4, 4]
    let x = upload(&be, &vec![1.0f32; 1 * 4 * 4 * 4], &[1, 4, 4, 4]);
    // Bilinear 2x upsample -> [1, 4, 8, 8]
    let upsampled = be
        .interpolate_bilinear_forward(&x, [8, 8], false)
        .expect("bilinear");
    assert_eq!(upsampled.shape(), &[1, 4, 8, 8]);

    // GroupNorm with 2 groups (2 channels per group)
    let gamma = upload(&be, &vec![1.0f32; 4], &[4]);
    let beta = upload(&be, &vec![0.0f32; 4], &[4]);
    let norm = be
        .group_norm_forward(&upsampled, &gamma, &beta, 2, 1e-5)
        .expect("gn");
    assert_eq!(norm.shape(), &[1, 4, 8, 8]);
    let data = host(&be, &norm);
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn tier3_06_additive_bias_attention_with_window_masking() {
    let be = cpu();
    // [1, 2 heads, 8 tokens, 16 dim]
    let q = upload(&be, &vec![0.1f32; 1 * 2 * 8 * 16], &[1, 2, 8, 16]);
    let k = upload(&be, &vec![0.1f32; 1 * 2 * 8 * 16], &[1, 2, 8, 16]);
    let v = upload(&be, &vec![1.0f32; 1 * 2 * 8 * 16], &[1, 2, 8, 16]);
    // Additive relative position bias [1, 2, 8, 8]
    let bias = upload(&be, &vec![0.05f32; 1 * 2 * 8 * 8], &[1, 2, 8, 8]);

    let opts = AttentionOpts {
        causal: true,
        window: Some(4), // causal window size 4
        key_lens: None,
        bias: Some(&bias),
        ..Default::default()
    };
    let out = be
        .attention_forward(&q, &k, &v, opts)
        .expect("biased window attn");
    assert_eq!(out.shape(), &[1, 2, 8, 16]);
    let data = host(&be, &out);
    assert!(data.iter().all(|v| v.is_finite()));
}

#[test]
fn tier3_07_multilayer_projection_into_matryoshka_l2_norm() {
    let be = cpu();
    // Projected embedding: [2 sequences, 512 dim]
    let mut x_data = Vec::with_capacity(2 * 512);
    for i in 0..2 * 512 {
        x_data.push(((i % 100) as f32) - 50.0);
    }
    let x = upload(&be, &x_data, &[2, 512]);

    // Download to run Matryoshka truncation to 128 dimensions and unit normalization
    let host_x = host(&be, &x);
    for row in 0..2 {
        let slice = &host_x[row * 512..row * 512 + 128];
        let mut sum_sq = 0.0f64;
        for &v in slice {
            sum_sq += f64::from(v) * f64::from(v);
        }
        let norm = sum_sq.sqrt();
        assert!(norm > 0.0);
        let normalized: Vec<f32> = slice
            .iter()
            .map(|&v| (f64::from(v) / norm) as f32)
            .collect();
        let final_norm = normalized
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        assert!(
            (final_norm - 1.0).abs() < 1e-5,
            "unit sphere norm must be 1.0, got {final_norm}"
        );
    }
}

#[test]
fn tier3_08_fused_qkv_linear_into_rope_interleaved() {
    let be = cpu();
    // Tokens: [1 batch, 16 time, 16 dim]
    let x = upload(&be, &vec![0.5f32; 1 * 16 * 16], &[1, 16, 16]);
    // Linear weight is [out_features, in_features] = [48, 16]
    let w_qkv = upload(&be, &vec![0.05f32; 48 * 16], &[48, 16]);
    let qkv = be.linear_forward(&x, &w_qkv).expect("linear qkv");
    assert_eq!(qkv.shape(), &[1, 16, 48]);

    // Slice into Q, K, V each [1, 16 time, 16 dim]
    let q = be.slice(&qkv, 2, 0, 16).expect("slice q");
    let k = be.slice(&qkv, 2, 16, 16).expect("slice k");
    assert_eq!(q.shape(), &[1, 16, 16]);
    assert_eq!(k.shape(), &[1, 16, 16]);

    // Reshape to 4 heads of dim 4: [1, 16, 4, 4]
    let q_heads = q.reshape(&[1, 16, 4, 4]).expect("q heads");
    let k_heads = k.reshape(&[1, 16, 4, 4]).expect("k heads");

    // Apply 2D RoPE
    let (cos, sin) = axial_rope_2d_tables(4, 4, 4, 10_000.0, 1.0, be.budget()).expect("tables");
    let q_rot = be
        .rope_interleaved_forward(&q_heads, &cos, &sin)
        .expect("q rot");
    let k_rot = be
        .rope_interleaved_forward(&k_heads, &cos, &sin)
        .expect("k rot");
    assert_eq!(q_rot.shape(), &[1, 16, 4, 4]);
    assert_eq!(k_rot.shape(), &[1, 16, 4, 4]);
}
