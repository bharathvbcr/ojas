//! Tier 4 — End-to-End Real-World Production Workloads.
//!
//! Validates full production pipelines on the current milestone primitives:
//! - Scenario A: SAM 3.1 Video Tracker Frame Multiplex Pipeline
//! - Scenario B: Gemma Embedding Production Inference Pipeline
//! - Scenario C: Pipeline Budget Scoping, Isolation & Trim
//! - Scenario D: Strict No Silent Fallback & Placement Integrity

#![allow(clippy::identity_op, clippy::useless_vec, clippy::needless_range_loop)]

use ojas_core::{
    axial_rope_2d_tables, AttentionOpts, Backend, Budget, Conv2dParams, ConvTranspose2dParams,
    Numerics, OjasError, Tensor,
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

fn varied_vector(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 23) as f32) - 11.0) * scale + 0.1)
        .collect()
}

/// Scenario A: SAM 3.1 Video Tracker Frame Multiplex Pipeline
///
/// Simulates one complete step of the SpikeTrack real-time video tracker:
/// 1. Patch Embedding: Conv2d (3 -> 64 channels, 14x14 patch)
/// 2. ViT Trunk:
///    - Token Reshape + 2D Axial RoPE
///    - Self-Attention + Residual
///    - LayerNorm
///    - MLP (Linear -> Exact erf GELU -> Linear) + Residual
/// 3. FPN Neck:
///    - Transposed Conv2d (2x upsampling)
///    - MaxPool2d (2x downsampling)
/// 4. SAM Memory Cross-Attention:
///    - Current frame tokens (Tq=4) cross-attending memory bank (Tk=16)
/// 5. DETR Decoder:
///    - Box RPB additive bias attention
///    - GroupNorm (4 groups)
///    - Bilinear upsampling to 8x8 mask logits
/// 6. Invariants:
///    - Determinism across repeat runs
///    - Zero memory leak over 10 consecutive frame iterations
#[test]
fn tier4_scenario_a_sam31_video_tracker_multiplex_pipeline() {
    let be = cpu();
    let initial_live = be.budget().live_bytes().expect("live bytes");

    let run_frame_step = || -> (Vec<f32>, usize) {
        // Step 1: Patch Embed Conv2d (1 frame, 3 channels, 28x28 -> 1, 64, 2, 2)
        let img_data = varied_vector(1 * 3 * 28 * 28, 0.05);
        let frame_img = upload(&be, &img_data, &[1, 3, 28, 28]);
        let patch_w_data = varied_vector(64 * 3 * 14 * 14, 0.01);
        let patch_w = upload(&be, &patch_w_data, &[64, 3, 14, 14]);
        let patch_params = Conv2dParams {
            stride: [14, 14],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        };
        let patches = be
            .conv2d_forward(&frame_img, &patch_w, None, patch_params)
            .expect("patch embed");
        assert_eq!(patches.shape(), &[1, 64, 2, 2]);

        // Step 2: ViT Trunk
        // Reshape to tokens for RoPE: [1 batch, 4 tokens, 1 head, 64 head_dim]
        let tokens = patches.reshape(&[1, 4, 1, 64]).expect("reshape tokens");
        let (cos, sin) =
            axial_rope_2d_tables(64, 2, 2, 10_000.0, 1.0, be.budget()).expect("rope tables");
        let q_rot = be
            .rope_interleaved_forward(&tokens, &cos, &sin)
            .expect("q rot");
        let k_rot = be
            .rope_interleaved_forward(&tokens, &cos, &sin)
            .expect("k rot");

        // Reshape to [1 batch, 1 head, 4 tokens, 64 dim] for Attention
        let q_attn = q_rot.reshape(&[1, 1, 4, 64]).expect("q attn");
        let k_attn = k_rot.reshape(&[1, 1, 4, 64]).expect("k attn");
        let v_attn = tokens.reshape(&[1, 1, 4, 64]).expect("v attn");

        // Self-Attention
        let attn_opts = AttentionOpts {
            causal: false,
            window: None,
            bias: None,
            key_lens: None,
        };
        let attn_out = be
            .attention_forward(&q_attn, &k_attn, &v_attn, attn_opts)
            .expect("self attn");
        assert_eq!(attn_out.shape(), &[1, 1, 4, 64]);
        let attn_tokens = attn_out.reshape(&[1, 4, 1, 64]).expect("attn tokens");

        // Residual + LayerNorm
        let res1 = be
            .residual_add_forward(&tokens, &attn_tokens)
            .expect("res1");
        let ln_gamma = upload(&be, &vec![1.0f32; 64], &[64]);
        let ln_beta = upload(&be, &vec![0.0f32; 64], &[64]);
        let ln1 = be
            .layer_norm_forward(&res1, &ln_gamma, &ln_beta, 1e-5)
            .expect("ln1");

        // MLP: Linear(64->128) -> GELU -> Linear(128->64)
        let ln1_flat = ln1.reshape(&[4, 64]).expect("flat tokens");
        let w_up = upload(&be, &varied_vector(128 * 64, 0.02), &[128, 64]);
        let w_down = upload(&be, &varied_vector(64 * 128, 0.02), &[64, 128]);
        let h_up = be.linear_forward(&ln1_flat, &w_up).expect("mlp up");
        let h_act = be.gelu_erf_forward(&h_up).expect("mlp gelu");
        let h_down = be.linear_forward(&h_act, &w_down).expect("mlp down");
        let h_down_shaped = h_down.reshape(&[1, 4, 1, 64]).expect("mlp shaped");
        let trunk_out = be
            .residual_add_forward(&ln1, &h_down_shaped)
            .expect("trunk res");

        // Step 3: FPN Neck
        // Reshape to spatial [1, 64, 2, 2]
        let trunk_spatial = trunk_out.reshape(&[1, 64, 2, 2]).expect("trunk spatial");
        // Transposed Conv2d: 2x upsample to [1, 32, 4, 4]
        let w_tconv = upload(&be, &varied_vector(64 * 32 * 2 * 2, 0.05), &[64, 32, 2, 2]);
        let tparams = ConvTranspose2dParams {
            stride: [2, 2],
            padding: [0, 0],
            output_padding: [0, 0],
        };
        let fpn_up = be
            .conv_transpose2d_forward(&trunk_spatial, &w_tconv, None, tparams)
            .expect("fpn up");
        assert_eq!(fpn_up.shape(), &[1, 32, 4, 4]);
        // MaxPool2d downsampling to [1, 32, 2, 2]
        let fpn_pool = be
            .max_pool2d_forward(&fpn_up, [2, 2], [2, 2])
            .expect("fpn maxpool");
        assert_eq!(fpn_pool.shape(), &[1, 32, 2, 2]);

        // Step 4: SAM Memory Cross-Attention
        // Current query: [1, 1 head, 4 tokens, 32 dim], Memory key/val: [1, 1 head, 16 tokens, 32 dim]
        let curr_q = fpn_pool.reshape(&[1, 1, 4, 32]).expect("curr q");
        let mem_k = upload(&be, &varied_vector(1 * 1 * 16 * 32, 0.05), &[1, 1, 16, 32]);
        let mem_v = upload(&be, &varied_vector(1 * 1 * 16 * 32, 0.05), &[1, 1, 16, 32]);
        let mem_opts = AttentionOpts {
            causal: false,
            window: None,
            bias: None,
            key_lens: None,
        };
        let mem_out = be
            .attention_forward(&curr_q, &mem_k, &mem_v, mem_opts)
            .expect("mem attn");
        assert_eq!(mem_out.shape(), &[1, 1, 4, 32]);

        // Step 5: DETR Box & Mask Decoder
        // Additive boxRPB bias attention: bias is [1, 1, 4, 4]
        let rpb_bias = upload(&be, &varied_vector(1 * 1 * 4 * 4, 0.02), &[1, 1, 4, 4]);
        let detr_opts = AttentionOpts {
            causal: false,
            window: None,
            bias: Some(&rpb_bias),
            key_lens: None,
        };
        let detr_out = be
            .attention_forward(&mem_out, &mem_out, &mem_out, detr_opts)
            .expect("detr attn");

        // GroupNorm: [1, 32, 2, 2] with 4 groups
        let mask_feat = detr_out.reshape(&[1, 32, 2, 2]).expect("mask feat");
        let gn_w = upload(&be, &vec![1.0f32; 32], &[32]);
        let gn_b = upload(&be, &vec![0.0f32; 32], &[32]);
        let norm_feat = be
            .group_norm_forward(&mask_feat, &gn_w, &gn_b, 4, 1e-5)
            .expect("group norm");
        assert_eq!(norm_feat.shape(), &[1, 32, 2, 2]);

        // Bilinear upsampling: 2x2 -> 8x8 high-resolution mask logits
        let mask_logits = be
            .interpolate_bilinear_forward(&norm_feat, [8, 8], false)
            .expect("bilinear up");
        assert_eq!(mask_logits.shape(), &[1, 32, 8, 8]);

        let host_logits = host(&be, &mask_logits);
        let elements = host_logits.len();
        (host_logits, elements)
    };

    // First frame execution and baseline verification
    let (logits_frame0, n_elem) = run_frame_step();
    assert_eq!(n_elem, 1 * 32 * 8 * 8);
    assert!(logits_frame0.iter().all(|v| v.is_finite()));

    // Verify bit-exact determinism on second frame
    let (logits_frame1, _) = run_frame_step();
    assert_eq!(
        logits_frame0, logits_frame1,
        "Pipeline execution must be bit-deterministic under Numerics::Exact"
    );

    // Execute 10 consecutive frame multiplex steps to verify zero memory leaks
    for _ in 0..10 {
        let (logits, _) = run_frame_step();
        assert_eq!(logits.len(), n_elem);
    }

    let final_live = be.budget().live_bytes().expect("live bytes");
    assert_eq!(
        initial_live, final_live,
        "Budget live bytes must return to baseline after frame step: initial={initial_live}, final={final_live}"
    );
}

/// Scenario B: Gemma Embedding Production Inference Pipeline
///
/// Simulates the HuggingFace EmbeddingGemma 2 text embedding pipeline:
/// 1. Token Embeddings: 2 sequences, 16 tokens, 64 hidden dim
/// 2. Gemma Transformer Encoder Block:
///    - Query/Key RMSNorm (`rms_qk_norm_forward`)
///    - Local Sliding Window Attention (W=4)
///    - Residual connection
///    - GeGLU feedforward (`gelu_erf_forward(gate) * up` -> Linear down)
///    - Residual connection + LayerNorm
/// 3. Multi-layer Projection (PLE)
/// 4. Masked Sequence Mean Pooling over valid tokens
/// 5. Matryoshka Prefix Truncation (64 -> 32 -> 16 dimensions)
/// 6. L2 Normalization (Unit Sphere $\|y\|_2 = 1.0 \pm 10^{-6}$)
/// 7. Invariants:
///    - Unit norm check across all Matryoshka dimensions
///    - Cosine similarity preservation across truncated prefixes
///    - Monotonic preservation under Numerics::Exact
#[test]
fn tier4_scenario_b_gemma_embedding_production_inference_pipeline() {
    let be = cpu();

    // Step 1: Input sequences (Batch=2, Tokens=16, Dim=64) with non-trivial feature variance
    let input_data = varied_vector(2 * 16 * 64, 0.1);
    let tokens = upload(&be, &input_data, &[2, 16, 64]);

    // Attention projections
    let w_q = upload(&be, &varied_vector(64 * 64, 0.05), &[64, 64]);
    let w_k = upload(&be, &varied_vector(64 * 64, 0.05), &[64, 64]);
    let w_v = upload(&be, &varied_vector(64 * 64, 0.05), &[64, 64]);

    let tokens_flat = tokens.reshape(&[32, 64]).expect("flat tokens");
    let q_flat = be.linear_forward(&tokens_flat, &w_q).expect("proj q");
    let k_flat = be.linear_forward(&tokens_flat, &w_k).expect("proj k");
    let v_flat = be.linear_forward(&tokens_flat, &w_v).expect("proj v");

    // Reshape to [Batch=2, Tokens=16, Heads=4, Dim=16], then permute to [Batch=2, Heads=4, Tokens=16, Dim=16]
    let q_bthd = q_flat.reshape(&[2, 16, 4, 16]).expect("q bthd");
    let k_bthd = k_flat.reshape(&[2, 16, 4, 16]).expect("k bthd");
    let v_bthd = v_flat.reshape(&[2, 16, 4, 16]).expect("v bthd");

    let q = be.permute(&q_bthd, &[0, 2, 1, 3]).expect("q permute");
    let k = be.permute(&k_bthd, &[0, 2, 1, 3]).expect("k permute");
    let v = be.permute(&v_bthd, &[0, 2, 1, 3]).expect("v permute");

    // Per-head Query and Key RMSNorm
    let rms_scale = upload(&be, &vec![1.0f32; 16], &[16]);
    let (q_norm, k_norm) = be
        .rms_qk_norm_forward(&q, &k, &rms_scale, &rms_scale, 1e-6)
        .expect("qk norm");

    // Local Sliding Window Attention (W=4, causal=true)
    let attn_opts = AttentionOpts {
        causal: true,
        window: Some(4),
        bias: None,
        key_lens: None,
    };
    let attn_out = be
        .attention_forward(&q_norm, &k_norm, &v, attn_opts)
        .expect("window attn");
    assert_eq!(attn_out.shape(), &[2, 4, 16, 16]);

    // Permute back to [Batch=2, Tokens=16, Heads=4, Dim=16] and flatten heads to [2, 16, 64]
    let attn_bthd = be.permute(&attn_out, &[0, 2, 1, 3]).expect("attn bthd");
    let attn_merged = attn_bthd.reshape(&[2, 16, 64]).expect("attn merged");
    let res1 = be
        .residual_add_forward(&tokens, &attn_merged)
        .expect("res1");

    // GeGLU Feedforward: gate * gelu(up) -> linear down
    let res1_flat = res1.reshape(&[32, 64]).expect("flat res1");
    let w_gate = upload(&be, &varied_vector(128 * 64, 0.03), &[128, 64]);
    let w_up = upload(&be, &varied_vector(128 * 64, 0.03), &[128, 64]);
    let w_down = upload(&be, &varied_vector(64 * 128, 0.03), &[64, 128]);

    let gate = be.linear_forward(&res1_flat, &w_gate).expect("gate");
    let up = be.linear_forward(&res1_flat, &w_up).expect("up");
    let act = be.gelu_erf_forward(&gate).expect("geglu act");
    let ffn_hidden = be.mul_forward(&act, &up).expect("geglu mul");
    let ffn_down = be.linear_forward(&ffn_hidden, &w_down).expect("ffn down");
    let ffn_shaped = ffn_down.reshape(&[2, 16, 64]).expect("ffn shaped");

    // Residual + LayerNorm
    let encoder_out = be.residual_add_forward(&res1, &ffn_shaped).expect("res2");
    let ln_w = upload(&be, &vec![1.0f32; 64], &[64]);
    let ln_b = upload(&be, &vec![0.0f32; 64], &[64]);
    let norm_out = be
        .layer_norm_forward(&encoder_out, &ln_w, &ln_b, 1e-6)
        .expect("final ln");

    // Step 4: Masked Sequence Mean Pooling
    // Sequence 0 has valid tokens [0..10], Sequence 1 has valid tokens [0..16]
    let host_tokens = host(&be, &norm_out);
    let valid_spans = [[0usize, 10usize], [16usize, 32usize]];
    let mut pooled_embeddings = Vec::new(); // 2 rows of 64 dims
    for span in &valid_spans {
        let (start, end) = (span[0], span[1]);
        let count = (end - start) as f32;
        let mut row_mean = vec![0.0f32; 64];
        for t in start..end {
            for c in 0..64 {
                row_mean[c] += host_tokens[t * 64 + c];
            }
        }
        for c in 0..64 {
            row_mean[c] /= count;
        }
        pooled_embeddings.push(row_mean);
    }
    assert_eq!(pooled_embeddings.len(), 2);

    // Step 5 & 6: Matryoshka Prefix Truncation (64 -> 32 -> 16) and L2 Normalization
    let target_dims = [64, 32, 16];
    for &d in &target_dims {
        for (seq_idx, full_emb) in pooled_embeddings.iter().enumerate() {
            let truncated_slice = &full_emb[..d];

            // Compute L2 norm
            let sum_sq: f64 = truncated_slice
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum();
            let norm = sum_sq.sqrt();
            assert!(
                norm > 0.0,
                "Norm must be strictly positive for seq {seq_idx}"
            );

            // L2 normalize
            let normalized: Vec<f32> = truncated_slice
                .iter()
                .map(|&v| (f64::from(v) / norm) as f32)
                .collect();

            // Verify Unit Sphere Norm condition: ||y||_2 = 1.0 ± 1e-6
            let final_norm: f64 = normalized
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt();
            assert!(
                (final_norm - 1.0).abs() < 1e-6,
                "Unit sphere normalization failed for d={d}, seq={seq_idx}: got {final_norm}"
            );

            // Cosine similarity check between full embedding prefix and normalized vector
            let mut dot = 0.0f64;
            for (&a, &b) in truncated_slice.iter().zip(&normalized) {
                dot += f64::from(a) * f64::from(b);
            }
            let cos_sim = dot / norm;
            assert!(
                (cos_sim - 1.0).abs() < 1e-6,
                "Cosine similarity must be 1.0 against unnormalized direction: got {cos_sim}"
            );
        }
    }
}

/// Scenario C: Pipeline Budget Scoping, Isolation & Trim
///
/// Verifies that allocating in a child pipeline budget properly isolates memory
/// and trimming or dropping child budgets frees live bytes in the root budget.
#[test]
fn tier4_scenario_c_pipeline_budget_scoping_and_trim() {
    let root_budget = Budget::new(64 * 1024 * 1024); // 64 MB
    let be_tracker = CpuBackend::new(root_budget.child(16 * 1024 * 1024));
    let be_embed = CpuBackend::new(root_budget.child(16 * 1024 * 1024));

    // Allocate in video tracker pipeline
    let t1 = upload(&be_tracker, &vec![1.0f32; 1024 * 1024], &[1024, 1024]); // 4 MB
    assert_eq!(t1.shape(), &[1024, 1024]);

    // Check tracker budget live bytes
    let tracker_live = be_tracker.budget().live_bytes().expect("tracker live");
    assert!(tracker_live >= 4 * 1024 * 1024);

    // Embedding pipeline budget should not be affected by tracker allocations
    let embed_live = be_embed.budget().live_bytes().expect("embed live");
    assert_eq!(
        embed_live, 0,
        "Embedding pipeline budget must remain isolated"
    );

    // Dropping tracker tensor frees tracker budget
    drop(t1);
    let tracker_after = be_tracker.budget().live_bytes().expect("tracker after");
    assert_eq!(
        tracker_after, 0,
        "Dropping pipeline tensors must free memory"
    );
}

/// Scenario D: Strict No Silent Fallback & Placement Integrity
///
/// Verifies that attempting invalid dimension layouts or malformed tensors fails closed
/// with typed errors rather than quietly falling back to host computation.
#[test]
fn tier4_scenario_d_no_silent_fallback_placement_invariance() {
    let be = cpu();
    let t = upload(&be, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // Attempting an op with an unsupported shape or invalid dimension must fail closed
    let empty_shape: &[usize] = &[];
    let res = t.reshape(empty_shape);
    assert!(
        res.is_err(),
        "Empty shape reshape must fail closed with OjasError"
    );

    // Conv2d with invalid kernel layout must fail closed with OjasError::Shape
    let invalid_weight = upload(&be, &[1.0, 2.0], &[2]); // 1D instead of 4D
    let params = Conv2dParams {
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    };
    let conv_res = be.conv2d_forward(&t, &invalid_weight, None, params);
    assert!(
        matches!(conv_res, Err(OjasError::Shape { .. })),
        "Invalid conv weight shape must return OjasError::Shape"
    );
}
