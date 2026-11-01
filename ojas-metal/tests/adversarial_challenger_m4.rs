//! Adversarial Challenger M4 Stress & Parity Verification Suite
//!
//! Empirically challenges:
//! 1. Metal `gelu_tanh` numerical stability on GPU, clamp boundaries (|x| <= 20),
//!    and non-finite status word flagging.
//! 2. Metal sliding window attention (|i - j| <= W) dual routing (TensorOps tiled vs scalar)
//!    parity across head dimensions D in {32, 64, 256, 512}.
//! 3. Metal `seq_mean_forward` zero-readback segment generation under multi-batch ragged scenarios.
//! 4. Metal `l2_normalize_forward` fail-closed epsilon validation (eps != 1e-12 returns Unsupported).
//! 5. Parity against `fixtures/gemma_ops/golden.txt` under mac_heavy.sh lock:
//!    max abs error <= 1.8e-6 and cosine similarity >= 1.0 - 1e-9.
//! 6. Full Tier 4 Scenario B production pipeline executed on Metal GPU with zero stubs.
//! 7. Concurrency stress: multi-threaded allocations, ops, and GPU timeline fences.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::thread;

use common::*;
use ojas_core::{AttentionOpts, Backend, Budget, OjasError};
use ojas_oracle::gemma_golden::check;

// =========================================================================
// 1. Metal gelu_tanh numerical stability, clamp boundaries, and non-finite
// =========================================================================

#[test]
fn test_gelu_tanh_stability_clamp_and_nonfinite_status_words() {
    let m = metal();
    let c = cpu();

    // Dense sweep through critical regions:
    // - extreme negative tails x in [-100, -20.01] (y -> 0, dy/dx -> 0)
    // - boundary x in [-20.05, -19.95]
    // - near zero x in [-2.0, 2.0]
    // - boundary x in [19.95, 20.05]
    // - extreme positive tails x in [20.01, 100.0] (y -> x, dy/dx -> 1)
    let mut xs = Vec::new();
    let extremes_neg = [-100.0f32, -80.0, -50.0, -30.0, -25.0, -21.0, -20.5];
    let boundary_neg = [-20.05, -20.01, -20.0, -19.99, -19.95, -19.5, -19.0];
    let mid_neg = [-15.0, -12.0, -10.0, -8.0, -6.0, -5.0, -4.0, -3.0];
    let near_zero: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.05).collect();
    let mid_pos = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0];
    let boundary_pos = [19.0, 19.5, 19.95, 19.99, 20.0, 20.01, 20.05];
    let extremes_pos = [20.5, 21.0, 25.0, 30.0, 50.0, 80.0, 100.0];

    xs.extend_from_slice(&extremes_neg);
    xs.extend_from_slice(&boundary_neg);
    xs.extend_from_slice(&mid_neg);
    xs.extend(near_zero);
    xs.extend_from_slice(&mid_pos);
    xs.extend_from_slice(&boundary_pos);
    xs.extend_from_slice(&extremes_pos);

    let n = xs.len();
    let host_x = host(&xs, &[n]);
    let dev_x = up(&m, &host_x);

    // Forward check
    let got_fwd = ok("metal fwd", m.gelu_tanh_forward(&dev_x));
    let want_fwd = ok("cpu fwd", c.gelu_tanh_forward(&host_x));
    same_tensor(
        "gelu_tanh_forward stability",
        &got_fwd,
        &want_fwd,
        2e-6,
        1e-5,
    );

    let fwd_vals = down(&got_fwd);

    // Verify limit invariants on Metal GPU
    for (i, &x) in xs.iter().enumerate() {
        let y = fwd_vals[i];
        assert!(
            y.is_finite(),
            "GELU forward output at x={x} is non-finite: {y}"
        );
        if x <= -20.0 {
            assert!(
                y.abs() <= 1e-7,
                "GELU forward at extreme negative x={x} must be 0: got {y}"
            );
        } else if x >= 20.0 {
            assert!(
                (y - x).abs() <= 1e-4,
                "GELU forward at extreme positive x={x} must approach x: got {y}"
            );
        }
    }

    // Continuity across boundary |x| = 20
    let idx_20_neg_left = xs
        .iter()
        .position(|&x| (x - (-20.01)).abs() < 1e-4)
        .unwrap();
    let idx_20_neg_right = xs
        .iter()
        .position(|&x| (x - (-19.99)).abs() < 1e-4)
        .unwrap();
    let jump_neg = (fwd_vals[idx_20_neg_right] - fwd_vals[idx_20_neg_left]).abs();
    assert!(jump_neg < 1e-5, "Discontinuity at x = -20: jump {jump_neg}");

    let idx_20_pos_left = xs.iter().position(|&x| (x - 19.99).abs() < 1e-4).unwrap();
    let idx_20_pos_right = xs.iter().position(|&x| (x - 20.01).abs() < 1e-4).unwrap();
    let jump_pos = (fwd_vals[idx_20_pos_right] - fwd_vals[idx_20_pos_left] - 0.02).abs();
    assert!(
        jump_pos < 1e-4,
        "Discontinuity at x = +20: delta {jump_pos}"
    );

    // Backward check
    let gy = rand(&[n], 777, 1.0);
    let dev_gy = up(&m, &gy);
    let got_bwd = ok("metal bwd", m.gelu_tanh_backward(&dev_x, &dev_gy));
    let want_bwd = ok("cpu bwd", c.gelu_tanh_backward(&host_x, &gy));
    same_tensor(
        "gelu_tanh_backward stability",
        &got_bwd,
        &want_bwd,
        3e-6,
        1e-5,
    );

    let bwd_vals = down(&got_bwd);
    let gy_vals = down(&gy);
    for (i, &x) in xs.iter().enumerate() {
        let dx = bwd_vals[i];
        let g = gy_vals[i];
        assert!(
            dx.is_finite(),
            "GELU backward output at x={x} is non-finite: {dx}"
        );
        if x <= -20.0 {
            assert!(
                dx.abs() <= 1e-6,
                "GELU backward at extreme negative x={x} must be 0: got {dx}"
            );
        } else if x >= 20.0 {
            assert!(
                (dx - g).abs() <= 1e-4,
                "GELU backward at extreme positive x={x} must equal gy={g}: got {dx}"
            );
        }
    }

    // Non-finite input status word flagging on Metal
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bad_vec = vec![1.0f32, bad, 2.0f32];
        let bad_host = host(&bad_vec, &[3]);
        let bad_dev = up(&m, &bad_host);

        // Forward with non-finite input
        let r = m.gelu_tanh_forward(&bad_dev);
        // Deferred fault contract: sync reports NonFinite
        let sync_r = m.sync();
        let fault = match r {
            Err(e) => Err(e),
            Ok(t) => {
                let _ = t;
                sync_r
            }
        };
        assert!(
            matches!(
                fault,
                Err(OjasError::NonFinite {
                    op: "gelu_tanh_forward"
                })
            ),
            "Expected NonFinite for {bad} in gelu_tanh_forward, got {fault:?}"
        );

        // Backward with non-finite input x
        let clean_gy = up(&m, &host(&[1.0, 1.0, 1.0], &[3]));
        let r_bwd = m.gelu_tanh_backward(&bad_dev, &clean_gy);
        let sync_bwd = m.sync();
        let fault_bwd = match r_bwd {
            Err(e) => Err(e),
            Ok(t) => {
                let _ = t;
                sync_bwd
            }
        };
        assert!(
            matches!(
                fault_bwd,
                Err(OjasError::NonFinite {
                    op: "gelu_tanh_backward"
                })
            ),
            "Expected NonFinite for {bad} in gelu_tanh_backward x, got {fault_bwd:?}"
        );

        // Backward with non-finite input gy
        let clean_x = up(&m, &host(&[1.0, 2.0, 3.0], &[3]));
        let bad_gy = up(&m, &bad_host);
        let r_bwd2 = m.gelu_tanh_backward(&clean_x, &bad_gy);
        let sync_bwd2 = m.sync();
        let fault_bwd2 = match r_bwd2 {
            Err(e) => Err(e),
            Ok(t) => {
                let _ = t;
                sync_bwd2
            }
        };
        assert!(
            matches!(
                fault_bwd2,
                Err(OjasError::NonFinite {
                    op: "gelu_tanh_backward"
                })
            ),
            "Expected NonFinite for {bad} in gelu_tanh_backward gy, got {fault_bwd2:?}"
        );
    }
}

// =========================================================================
// 2. Sliding window attention dual routing across D in {32, 64, 256, 512}
// =========================================================================

#[test]
fn test_sliding_window_attention_dual_routing_across_head_dims() {
    let m = metal();
    let c = cpu();

    let head_dims = [32usize, 64, 256, 512];
    let (b, h, t) = (2usize, 2usize, 32usize);
    let radius = 6usize; // Sliding window W = 6, symmetric non-causal

    for &d in &head_dims {
        let seed = (d as u64) * 100;
        let q = rand(&[b, h, t, d], seed, 1.0);
        let k = rand(&[b, h, t, d], seed + 1, 1.0);
        let v = rand(&[b, h, t, d], seed + 2, 1.0);

        let opts = AttentionOpts {
            causal: false,
            radius: Some(radius),
            scale: None,
            bias: None,
            key_lens: None,
            ..Default::default()
        };

        // 1. CPU reference
        let want_cpu = ok("cpu swa", c.attention_forward(&q, &k, &v, opts));

        let dq = up(&m, &q);
        let dk = up(&m, &k);
        let dv = up(&m, &v);

        // Count dispatches for routing check
        ok("sync", m.sync());
        let before_routed = m.dispatches();
        let got_routed = ok("metal swa routed", m.attention_forward(&dq, &dk, &dv, opts));
        ok("sync", m.sync());
        let routed_dispatches = m.dispatches() - before_routed;

        if d <= 256 {
            // 2. Metal scalar kernel via saving form (never routes to tiled encoder_attn)
            let before_scalar = m.dispatches();
            let (got_scalar, _lse) = ok(
                "metal swa scalar",
                m.attention_forward_saving(&dq, &dk, &dv, opts),
            );
            ok("sync", m.sync());
            let scalar_dispatches = m.dispatches() - before_scalar;

            if m.has_tensorops() {
                // Tiled routing encodes permutes + kernel + permutes > scalar single dispatch
                assert!(
                    routed_dispatches > scalar_dispatches,
                    "D={d}: expected routed tiled dispatches ({routed_dispatches}) > scalar ({scalar_dispatches})"
                );
            }

            // Assert dual routing parity: Tiled vs Scalar on Metal
            let tol = 5e-4 * (t as f32).sqrt();
            let host_routed = down(&got_routed);
            let host_scalar = down(&got_scalar);
            close(
                &format!("SWA D={d} tiled vs scalar", d = d),
                &host_routed,
                &host_scalar,
                tol,
                1e-4,
            );

            // Assert Parity: Scalar vs CPU
            same_tensor(
                &format!("SWA D={d} scalar vs cpu", d = d),
                &got_scalar,
                &want_cpu,
                tol,
                1e-4,
            );
        } else {
            // D = 512 exceeds METAL_MAX_HEAD_DIM (256): scalar kernel must refuse with UnsupportedHeadDim
            let scalar_res = m.attention_forward_saving(&dq, &dk, &dv, opts);
            assert!(
                matches!(
                    scalar_res,
                    Err(OjasError::UnsupportedHeadDim {
                        head_dim: 512,
                        limit: 256
                    })
                ),
                "D=512 on saving form must return UnsupportedHeadDim: got {scalar_res:?}"
            );
        }

        // Assert Parity: Tiled vs CPU (supported for all D in {32, 64, 256, 512})
        let tol = 5e-4 * (t as f32).sqrt();
        same_tensor(
            &format!("SWA D={d} tiled vs cpu", d = d),
            &got_routed,
            &want_cpu,
            tol,
            1e-4,
        );
    }

    // Edge cases for sliding window
    let d = 64;
    let q = up(&m, &rand(&[1, 1, 16, d], 1, 1.0));
    let k = up(&m, &rand(&[1, 1, 16, d], 2, 1.0));
    let v = up(&m, &rand(&[1, 1, 16, d], 3, 1.0));

    // W = 0 must fail closed
    let zero_res = m.attention_forward(
        &q,
        &k,
        &v,
        AttentionOpts {
            radius: Some(0),
            ..Default::default()
        },
    );
    assert!(
        matches!(zero_res, Err(OjasError::Shape { .. })),
        "W=0 must return OjasError::Shape: got {zero_res:?}"
    );

    // Causal with radius must fail closed
    let causal_res = m.attention_forward(
        &q,
        &k,
        &v,
        AttentionOpts {
            causal: true,
            radius: Some(4),
            ..Default::default()
        },
    );
    assert!(
        matches!(causal_res, Err(OjasError::Shape { .. })),
        "Causal with radius must return OjasError::Shape: got {causal_res:?}"
    );

    // Tq != Tk with radius must fail closed
    let q_short = up(&m, &rand(&[1, 1, 8, d], 4, 1.0));
    let tq_diff_res = m.attention_forward(
        &q_short,
        &k,
        &v,
        AttentionOpts {
            radius: Some(4),
            ..Default::default()
        },
    );
    assert!(
        matches!(tq_diff_res, Err(OjasError::Shape { .. })),
        "Tq != Tk with radius must return OjasError::Shape: got {tq_diff_res:?}"
    );
}

// =========================================================================
// 3. Metal seq_mean_forward zero-readback segment generation (ragged)
// =========================================================================

#[test]
fn test_seq_mean_forward_zero_readback_ragged() {
    let m = metal();
    let c = cpu();

    let b = 4;
    let t = 32;
    let d = 64;

    // Ragged lengths: full, ragged, minimal, zero
    let lens_data = vec![32u32, 17u32, 1u32, 0u32];
    let host_lens = host_u32(&lens_data, &[b]);
    let dev_lens = up(&m, &host_lens);

    let host_x = rand(&[b, t, d], 42, 1.0);
    let dev_x = up(&m, &host_x);

    let want = ok("cpu seq_mean", c.seq_mean_forward(&host_x, &host_lens));
    let got = ok("metal seq_mean", m.seq_mean_forward(&dev_x, &dev_lens));
    same_tensor("seq_mean ragged", &got, &want, 1e-5, 1e-5);

    // Verify Row 3 (len = 0) is bit-exact zero row
    let got_vals = down(&got);
    let row3 = &got_vals[3 * d..4 * d];
    assert!(
        row3.iter().all(|&v| v == 0.0),
        "Row 3 (length 0) must be all zeros: got {:?}",
        &row3[..8]
    );

    // Adversarial padding corruption: populate padding tokens (rows >= len_b) with 1e6
    let mut corrupted_x = down(&host_x);
    for row_idx in 0..b {
        let valid_len = lens_data[row_idx] as usize;
        for token_idx in valid_len..t {
            for col in 0..d {
                corrupted_x[(row_idx * t + token_idx) * d + col] = 1.0e6;
            }
        }
    }
    let host_corrupted = host(&corrupted_x, &[b, t, d]);
    let dev_corrupted = up(&m, &host_corrupted);

    let got_corrupted = ok(
        "metal seq_mean corrupted",
        m.seq_mean_forward(&dev_corrupted, &dev_lens),
    );
    // The mean pooling must ONLY average active tokens; padding must not affect result
    let host_corrupted = down(&got_corrupted);
    let host_clean = down(&got);
    close(
        "seq_mean padding invariance",
        &host_corrupted,
        &host_clean,
        1e-6,
        1e-6,
    );

    // All-zero lengths batch: B=3, all length 0
    let all_zeros_lens = host_u32(&[0u32, 0u32, 0u32], &[3]);
    let all_zeros_x = rand(&[3, 16, 32], 99, 1.0);
    let got_all_zeros = ok(
        "metal seq_mean all zeros",
        m.seq_mean_forward(&up(&m, &all_zeros_x), &up(&m, &all_zeros_lens)),
    );
    let all_zeros_vals = down(&got_all_zeros);
    assert!(
        all_zeros_vals.iter().all(|&v| v == 0.0),
        "All zero lengths must produce all zeros output"
    );
}

// =========================================================================
// 4. Metal l2_normalize_forward fail-closed epsilon validation
// =========================================================================

#[test]
fn test_l2_normalize_forward_fail_closed_eps() {
    let m = metal();
    let c = cpu();

    let rows = 4;
    let dim = 768; // Gemma embedding dimension
    let host_x = rand(&[rows, dim], 12345, 1.0);
    let dev_x = up(&m, &host_x);

    // Supported epsilon: 1e-12
    let eps_valid = 1e-12f32;
    let want = ok("cpu l2_norm", c.l2_normalize_forward(&host_x, eps_valid));
    let got = ok("metal l2_norm", m.l2_normalize_forward(&dev_x, eps_valid));
    same_tensor("l2_normalize valid", &got, &want, 1e-6, 1e-6);

    // Verify unit norm condition on each row
    let norm_vals = down(&got);
    for r in 0..rows {
        let row = &norm_vals[r * dim..(r + 1) * dim];
        let norm: f64 = row
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() <= 1e-6,
            "Row {r} must have unit L2 norm: got {norm}"
        );
    }

    // Fail-closed validation for non-1e-12 epsilons
    for bad_eps in [1e-5f32, 1e-6, 1e-11, 1e-13, 1.0, 0.5] {
        let res = m.l2_normalize_forward(&dev_x, bad_eps);
        assert!(
            matches!(
                res,
                Err(OjasError::Unsupported {
                    op: "l2_normalize_forward",
                    ..
                })
            ),
            "Expected Unsupported for eps {bad_eps}, got {res:?}"
        );
    }

    // Non-positive or non-finite epsilons fail at boundary validator
    for bad_range_eps in [0.0f32, -1e-12, -1.0] {
        let res = m.l2_normalize_forward(&dev_x, bad_range_eps);
        assert!(
            matches!(
                res,
                Err(OjasError::OutOfRange {
                    op: "l2_normalize_forward",
                    ..
                })
            ),
            "Expected OutOfRange for eps {bad_range_eps}, got {res:?}"
        );
    }
    for non_finite_eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let res = m.l2_normalize_forward(&dev_x, non_finite_eps);
        assert!(
            matches!(
                res,
                Err(OjasError::NonFinite {
                    op: "l2_normalize_forward"
                })
            ),
            "Expected NonFinite for eps {non_finite_eps}, got {res:?}"
        );
    }

    // Zero-row handling: input row of zeros produces output row of zeros without NaN
    let zero_data = vec![0.0f32; dim];
    let zero_host = host(&zero_data, &[1, dim]);
    let zero_dev = up(&m, &zero_host);
    let zero_norm = ok(
        "metal zero norm",
        m.l2_normalize_forward(&zero_dev, eps_valid),
    );
    let zero_vals = down(&zero_norm);
    assert!(
        zero_vals.iter().all(|&v| v == 0.0),
        "Zero row must normalize to zero row without NaN"
    );

    // Matryoshka prefix slicing: 768 -> 512, 256, 128, 64
    for target_dim in [512, 256, 128, 64] {
        let prefix = ok("slice", m.slice(&dev_x, 1, 0, target_dim));
        let norm_prefix = ok("norm prefix", m.l2_normalize_forward(&prefix, eps_valid));
        let vals = down(&norm_prefix);
        for r in 0..rows {
            let row = &vals[r * target_dim..(r + 1) * target_dim];
            let norm: f64 = row
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() <= 1e-6,
                "Matryoshka prefix dim {target_dim} row {r} norm must be 1.0: got {norm}"
            );
        }
    }
}

// =========================================================================
// 5. Parity against fixtures/gemma_ops/golden.txt: max_abs <= 1.8e-6, cos >= 1.0 - 1e-9
// =========================================================================

#[test]
fn test_gemma_golden_fixtures_parity_strict() {
    let m = metal();
    let report = check(&m).expect("gemma golden cases run on metal");
    assert!(
        report.records.len() >= 25,
        "Expected at least 25 evaluated ops, got {}",
        report.records.len()
    );

    let mut worst_abs = 0.0f64;
    let mut worst_cos_deg = 0.0f64;

    for r in &report.records {
        if r.max_abs > worst_abs {
            worst_abs = r.max_abs;
        }
        let cos_deg = 1.0 - r.cosine;
        if cos_deg > worst_cos_deg {
            worst_cos_deg = cos_deg;
        }

        // Strict assertion required by mission:
        // max abs error <= 1.8e-6 and cosine similarity >= 1.0 - 1e-9
        assert!(
            r.max_abs <= 1.8e-6,
            "Golden parity violation: op {} case {} max_abs {:.3e} > 1.8e-6",
            r.op,
            r.case,
            r.max_abs
        );
        assert!(
            r.cosine >= 1.0 - 1.0e-9,
            "Golden parity violation: op {} case {} cosine {:.12} < 1.0 - 1e-9",
            r.op,
            r.case,
            r.cosine
        );
    }

    eprintln!(
        "CHALLENGER_VERIFIED: all {} ops in golden.txt satisfied: worst_max_abs={:.3e} <= 1.8e-6, worst_cosine_deg={:.3e} <= 1e-9",
        report.records.len(),
        worst_abs,
        worst_cos_deg
    );
}

// =========================================================================
// 6. Tier 4 Scenario B production inference pipeline on Metal GPU
// =========================================================================

#[test]
fn test_tier4_scenario_b_production_pipeline_on_metal() {
    let be = metal();

    // Step 1: Input sequences (Batch=2, Tokens=16, Dim=64)
    let input_data = values(2 * 16 * 64, 42, 0.1);
    let host_tokens = host(&input_data, &[2, 16, 64]);
    let tokens = up(&be, &host_tokens);

    // Attention projections
    let host_wq = host(&values(64 * 64, 43, 0.05), &[64, 64]);
    let host_wk = host(&values(64 * 64, 44, 0.05), &[64, 64]);
    let host_wv = host(&values(64 * 64, 45, 0.05), &[64, 64]);
    let w_q = up(&be, &host_wq);
    let w_k = up(&be, &host_wk);
    let w_v = up(&be, &host_wv);

    let tokens_flat = tokens.reshape(&[32, 64]).expect("flat tokens");
    let q_flat = be.linear_forward(&tokens_flat, &w_q).expect("proj q");
    let k_flat = be.linear_forward(&tokens_flat, &w_k).expect("proj k");
    let v_flat = be.linear_forward(&tokens_flat, &w_v).expect("proj v");

    let q_bthd = q_flat.reshape(&[2, 16, 4, 16]).expect("q bthd");
    let k_bthd = k_flat.reshape(&[2, 16, 4, 16]).expect("k bthd");
    let v_bthd = v_flat.reshape(&[2, 16, 4, 16]).expect("v bthd");

    let q = be.permute(&q_bthd, &[0, 2, 1, 3]).expect("q permute");
    let k = be.permute(&k_bthd, &[0, 2, 1, 3]).expect("k permute");
    let v = be.permute(&v_bthd, &[0, 2, 1, 3]).expect("v permute");

    // Per-head Query, Key, and Value RMSNorm
    let host_rms = host(&[1.0f32; 16], &[16]);
    let rms_scale = up(&be, &host_rms);
    let (q_norm, k_norm, v_norm) = be
        .rms_qkv_norm_forward(&q, &k, &v, Some(&rms_scale), Some(&rms_scale), 1e-6)
        .expect("qkv norm");

    let host_lens = host_u32(&[10, 16], &[2]);
    let key_lens = up(&be, &host_lens);

    // Local Sliding Window Attention (W=4, symmetric non-causal, scale=1.0)
    let attn_opts = AttentionOpts {
        causal: false,
        radius: Some(4),
        scale: Some(1.0),
        bias: None,
        key_lens: Some(&key_lens),
        ..Default::default()
    };
    let attn_out = be
        .attention_forward(&q_norm, &k_norm, &v_norm, attn_opts)
        .expect("window attn");
    assert_eq!(attn_out.shape(), &[2, 4, 16, 16]);

    let attn_bthd = be.permute(&attn_out, &[0, 2, 1, 3]).expect("attn bthd");
    let attn_merged = attn_bthd.reshape(&[2, 16, 64]).expect("attn merged");
    let res1 = be
        .residual_add_forward(&tokens, &attn_merged)
        .expect("res1");

    // GeGLU Feedforward: gate * gelu_tanh(up) -> linear down
    let res1_flat = res1.reshape(&[32, 64]).expect("flat res1");
    let w_gate = up(&be, &host(&values(128 * 64, 46, 0.03), &[128, 64]));
    let w_up = up(&be, &host(&values(128 * 64, 47, 0.03), &[128, 64]));
    let w_down = up(&be, &host(&values(64 * 128, 48, 0.03), &[64, 128]));

    let gate = be.linear_forward(&res1_flat, &w_gate).expect("gate");
    let up_t = be.linear_forward(&res1_flat, &w_up).expect("up");
    let act = be.gelu_tanh_forward(&gate).expect("geglu act");
    let ffn_hidden = be.mul_forward(&act, &up_t).expect("geglu mul");
    let ffn_down = be.linear_forward(&ffn_hidden, &w_down).expect("ffn down");
    let ffn_shaped = ffn_down.reshape(&[2, 16, 64]).expect("ffn shaped");

    // Residual + LayerNorm
    let encoder_out = be.residual_add_forward(&res1, &ffn_shaped).expect("res2");
    let ln_w = up(&be, &host(&[1.0f32; 64], &[64]));
    let ln_b = up(&be, &host(&[0.0f32; 64], &[64]));
    let norm_out = be
        .layer_norm_forward(&encoder_out, &ln_w, &ln_b, 1e-6)
        .expect("final ln");

    // Masked Sequence Mean Pooling
    let pooled_embeddings = be.seq_mean_forward(&norm_out, &key_lens).expect("seq mean");
    assert_eq!(pooled_embeddings.shape(), &[2, 64]);

    // Matryoshka Prefix Truncation (64 -> 32 -> 16) and L2 Normalization
    for &d in &[64usize, 32, 16] {
        let prefix = be.slice(&pooled_embeddings, 1, 0, d).expect("slice prefix");
        assert_eq!(prefix.shape(), &[2, d]);

        let normalized = be
            .l2_normalize_forward(&prefix, 1e-12)
            .expect("l2 normalize");
        assert_eq!(normalized.shape(), &[2, d]);

        let prefix_host = down(&prefix);
        let norm_host = down(&normalized);

        for seq_idx in 0..2 {
            let row_norm = &norm_host[seq_idx * d..(seq_idx + 1) * d];
            let row_prefix = &prefix_host[seq_idx * d..(seq_idx + 1) * d];

            let unit_norm: f64 = row_norm
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt();
            assert!(
                (unit_norm - 1.0).abs() < 1e-6,
                "Metal Scenario B: Unit norm failed for d={d}, seq={seq_idx}: got {unit_norm}"
            );

            let mut dot = 0.0f64;
            let mut prefix_sq = 0.0f64;
            for (&a, &b) in row_prefix.iter().zip(row_norm) {
                dot += f64::from(a) * f64::from(b);
                prefix_sq += f64::from(a) * f64::from(a);
            }
            let cos_sim = dot / (prefix_sq.sqrt() * unit_norm);
            assert!(
                (cos_sim - 1.0).abs() < 1e-6,
                "Metal Scenario B: Cosine similarity degraded for d={d}, seq={seq_idx}: got {cos_sim}"
            );
        }
    }
}

// =========================================================================
// 7. Concurrency stress: multi-threaded allocations and GPU timeline fences
// =========================================================================

#[test]
fn test_concurrency_stress_multithreaded_allocations_and_fences() {
    let m = metal();
    let initial_live = m.budget().live_bytes().expect("initial live");

    let num_threads = 8;
    let iterations_per_thread = 20;

    let mut handles = Vec::new();
    for thread_idx in 0..num_threads {
        let be = m.clone();
        let handle = thread::spawn(move || {
            for iter in 0..iterations_per_thread {
                let seed = ((thread_idx * 1000) + iter) as u64;
                let rows = 4;
                let dim = 32;

                // Allocate, upload
                let h_x = host(&values(rows * dim, seed, 1.0), &[rows, dim]);
                let dev_x = match be.upload(&h_x) {
                    Ok(t) => t,
                    Err(e) => panic!("Thread {thread_idx} upload failed: {e:?}"),
                };

                // GELU tanh forward
                let dev_gelu = match be.gelu_tanh_forward(&dev_x) {
                    Ok(t) => t,
                    Err(e) => panic!("Thread {thread_idx} gelu failed: {e:?}"),
                };

                // L2 normalize
                let dev_norm = match be.l2_normalize_forward(&dev_gelu, 1e-12) {
                    Ok(t) => t,
                    Err(e) => panic!("Thread {thread_idx} l2_norm failed: {e:?}"),
                };

                // Zero-copy UMA readback via to_host() exercising Timeline fences
                let host_read = match dev_norm.to_host(&Budget::new(GIB)) {
                    Ok(h) => h,
                    Err(e) => panic!("Thread {thread_idx} to_host failed: {e:?}"),
                };
                let vals = host_read.to_f32_vec().expect("to_f32_vec");

                // Verify unit norm
                for r in 0..rows {
                    let row = &vals[r * dim..(r + 1) * dim];
                    let norm: f64 = row
                        .iter()
                        .map(|&v| (v as f64) * (v as f64))
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        (norm - 1.0).abs() <= 1e-5,
                        "Thread {thread_idx} iter {iter} row {r} not unit norm: got {norm}"
                    );
                }
            }
        });
        handles.push(handle);
    }

    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .unwrap_or_else(|e| panic!("Thread {i} panicked: {e:?}"));
    }

    ok("sync", m.sync());
    let final_live = m.budget().live_bytes().expect("final live");
    assert_eq!(
        initial_live, final_live,
        "Concurrency stress memory leak: initial={initial_live}, final={final_live}"
    );
}
