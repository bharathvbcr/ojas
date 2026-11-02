//! Numerical stability and algebraic invariant verification for `ojas-cpu` and `Autocast`:
//! - T1 & T2: High-margin cross-entropy softmax-collapse immunity (`other_sum` + `log1p_exact`)
//!   and row-wise zero-sum gradient conservation under `F32` and `AutocastMode::Bf16`.
//! - T3 & T4: Attention backward key-translation gauge invariance (`K <- K + 1 * c^T`)
//!   and `bf16` output-rounding immunity in blocked `flash::backward`.
//! - T5: Muon 5-step Newton-Schulz (`muon_ns5_step`) singular value spectrum contraction.
//! - T6: Chunked Gated DeltaNet (`chunked_gdn_forward` / `chunked_gdn_backward`) FP32
//!   checkpoint and decay-gradient preservation under `AutocastMode::Bf16`.

mod common;

use common::{f32t, u32t, SplitMix64};
use ojas_core::{
    round_f32_to_bf16, AttentionOpts, Autocast, AutocastMode, Backend, Budget, GdnInputs,
    MuonNs5Config, Numerics, OjasError, Tensor, GDN_CHECKPOINT_TOKENS,
};
use ojas_cpu::{
    causal_sdpa_backward_gauge_projected, check_i8_accumulator_bound,
    cross_entropy_mean_backward_exact_margin, CpuBackend, QuantizedWeightI8, Sparse24Weight,
    I8_ACCUMULATOR_MAX_K,
};

fn cpu(numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 28), 4)
        .unwrap()
        .with_numerics(numerics)
}

#[test]
fn t1_t2_cross_entropy_softmax_collapse_and_zero_sum_gradient_conservation() {
    let vocab = 64usize;
    let margins = [
        0.5f32, 1.0, 2.0, 4.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 20.0, 21.0, 22.0, 24.0, 40.0,
        80.0,
    ];
    let rows = margins.len();
    let mut logits = vec![0.0f32; rows * vocab];
    let mut targets = vec![0u32; rows];
    for (r, &m) in margins.iter().enumerate() {
        targets[r] = (r % vocab) as u32;
        for c in 0..vocab {
            logits[r * vocab + c] = if c == (r % vocab) {
                round_f32_to_bf16(m)
            } else {
                0.0
            };
        }
    }

    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = cpu(numerics);
        let ac = Autocast::new(cpu(numerics));
        let logits_t = f32t(&be, &logits, &[rows, vocab]);
        let targets_t = u32t(&be, &targets, &[rows]);

        let grad_f32 = be
            .cross_entropy_mean_backward(&logits_t, &targets_t, None)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let grad_bf16 = {
            let _guard = ac.autocast_region(AutocastMode::Bf16).unwrap();
            ac.cross_entropy_mean_backward(&logits_t, &targets_t, None)
                .unwrap()
                .to_f32_vec()
                .unwrap()
        };

        for (r, &m) in margins.iter().enumerate() {
            let m_bf16 = f64::from(round_f32_to_bf16(m));
            let target_col = r % vocab;
            let row_f32 = &grad_f32[r * vocab..(r + 1) * vocab];
            let row_bf16 = &grad_bf16[r * vocab..(r + 1) * vocab];

            // T2: Row-wise zero-sum gradient conservation across all margins up to 80.0.
            let sum_f32: f64 = row_f32.iter().map(|&g| f64::from(g)).sum();
            let sum_bf16: f64 = row_bf16.iter().map(|&g| f64::from(g)).sum();
            assert!(
                sum_f32.abs() < 1e-6,
                "{numerics:?} margin {m}: F32 row grad sum {sum_f32:e} != 0"
            );
            assert!(
                sum_bf16.abs() < 1e-6,
                "{numerics:?} margin {m}: Bf16 autocast row grad sum {sum_bf16:e} != 0"
            );

            // T1: At all margins up to 80.0 (past both the bf16 softmax collapse cliff
            // m > ln(63 * 2^8) ≈ 9.69 and the f32 softmax collapse cliff m > ln(63 * 2^24) ≈ 20.78),
            // target gradients never vanish (`got < 0.0`). Once `sum == 1.0` triggers
            // `collapses(picked, max, sum)` (m >= 21.0 for both Exact and Fast), `other_sum`
            // restores full f32 precision (`rel_err < 1e-5`) and exact row-sum cancellation.
            let s_f64 = ((vocab - 1) as f64) * (-m_bf16).exp();
            let want_target_grad = -(s_f64 / (1.0 + s_f64)) / (rows as f64);
            let got_f32 = f64::from(row_f32[target_col]);
            let got_bf16 = f64::from(row_bf16[target_col]);
            assert!(
                got_f32 < 0.0 && got_bf16 < 0.0,
                "{numerics:?} margin {m}: target gradient collapsed to zero"
            );
            let rel_err = (got_bf16 - want_target_grad).abs() / want_target_grad.abs();
            if m <= 5.5 || (21.0..=80.0).contains(&m) {
                assert!(
                    rel_err < 2e-5,
                    "{numerics:?} margin {m}: target grad {got_bf16:e} vs f64 {want_target_grad:e} (rel {rel_err:e})"
                );
            } else {
                // Pre-collapse standard form `1/sum - 1` adds (vocab - 1) terms to 1.0f32,
                // giving worst-case absorption bound <= vocab * f32::EPSILON / S.
                let absorption_bound = ((vocab as f64) * f64::from(f32::EPSILON) / s_f64).max(2e-5);
                assert!(
                    rel_err <= absorption_bound,
                    "{numerics:?} margin {m}: rel_err {rel_err:e} exceeded standard-form absorption bound {absorption_bound:e}"
                );
            }
        }
    }
}

#[test]
fn t3_t4_attention_backward_key_translation_gauge_invariance_and_row_sum() {
    let be_exact = cpu(Numerics::Exact);
    let be_fast = cpu(Numerics::Fast);

    // Part A: Key-translation gauge invariance K <- K + 1 * c^T across causal_sdpa and attention_backward.
    for &(t, numerics) in &[(32usize, Numerics::Exact), (260usize, Numerics::Fast)] {
        let be = cpu(numerics);
        let (b, h, d) = (1usize, 2usize, 16usize);
        let shape = [b, h, t, d];
        let n = b * h * t * d;
        let mut rng = SplitMix64(0xA77E_0001 + t as u64);
        let q_vec = rng.vec(n, 0.5);
        let k_vec = rng.vec(n, 0.5);
        let v_vec = rng.vec(n, 0.5);
        let gy_vec = rng.vec(n, 0.5);

        // Large constant key shift c_d in [-25, 25] per head dimension
        let shift: Vec<f32> = (0..d).map(|i| (i as f32 - 8.0) * 3.0).collect();
        let mut k_shifted = k_vec.clone();
        for row in k_shifted.chunks_exact_mut(d) {
            for (slot, &c) in row.iter_mut().zip(&shift) {
                *slot += c;
            }
        }

        let q = f32t(&be, &q_vec, &shape);
        let k0 = f32t(&be, &k_vec, &shape);
        let k1 = f32t(&be, &k_shifted, &shape);
        let v = f32t(&be, &v_vec, &shape);
        let gy = f32t(&be, &gy_vec, &shape);

        let (gq0, gk0, gv0) = be
            .causal_sdpa_backward_recompute(&q, &k0, &v, &gy, None)
            .unwrap();
        let (gq1, gk1, gv1) = be
            .causal_sdpa_backward_recompute(&q, &k1, &v, &gy, None)
            .unwrap();

        for (label, a, b_ten) in [("dq", &gq0, &gq1), ("dk", &gk0, &gk1), ("dv", &gv0, &gv1)] {
            let av = a.to_f32_vec().unwrap();
            let bv = b_ten.to_f32_vec().unwrap();
            let max_ref = av.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
            let max_diff = av
                .iter()
                .zip(&bv)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                max_diff / max_ref < 5e-5,
                "T={t} {numerics:?} {label}: key-shift gauge drift {:.2e} (rel {:.2e})",
                max_diff,
                max_diff / max_ref
            );
        }
    }

    // Part B: Under AutocastMode::Bf16 at T = 260 (blocked flash kernel), `flash::backward`
    // computes row-wise `delta = dot(P_row, dP_row)` directly from P and dP rather than
    // `dot(dO, O_bf16)`, so rounding `output` to bf16 produces zero degradation in dQ/dK/dV.
    let shape = [1usize, 2, 260, 32];
    let n: usize = shape.iter().product();
    let mut rng = SplitMix64(0xF1A5_2026);
    let q_bf16: Vec<f32> = rng.vec(n, 0.8).into_iter().map(round_f32_to_bf16).collect();
    let k_bf16: Vec<f32> = rng.vec(n, 0.8).into_iter().map(round_f32_to_bf16).collect();
    let v_bf16: Vec<f32> = rng.vec(n, 0.8).into_iter().map(round_f32_to_bf16).collect();
    let gy_bf16: Vec<f32> = rng.vec(n, 0.8).into_iter().map(round_f32_to_bf16).collect();

    let q = f32t(&be_fast, &q_bf16, &shape);
    let k = f32t(&be_fast, &k_bf16, &shape);
    let v = f32t(&be_fast, &v_bf16, &shape);
    let gy = f32t(&be_fast, &gy_bf16, &shape);

    let (out_f32, lse) = be_fast.causal_sdpa_forward(&q, &k, &v, None).unwrap();
    let out_bf16_vec: Vec<f32> = out_f32
        .to_f32_vec()
        .unwrap()
        .into_iter()
        .map(round_f32_to_bf16)
        .collect();
    let out_bf16 = f32t(&be_fast, &out_bf16_vec, &shape);

    let (gq_f32_out, gk_f32_out, gv_f32_out) = be_fast
        .causal_sdpa_backward(&q, &k, &v, &out_f32, &lse, &gy, None)
        .unwrap();
    let (gq_bf16_out, gk_bf16_out, gv_bf16_out) = be_fast
        .causal_sdpa_backward(&q, &k, &v, &out_bf16, &lse, &gy, None)
        .unwrap();

    assert_eq!(
        gq_f32_out.to_f32_vec().unwrap(),
        gq_bf16_out.to_f32_vec().unwrap(),
        "flash::backward dQ must be invariant to bf16 rounding of saved output O"
    );
    assert_eq!(
        gk_f32_out.to_f32_vec().unwrap(),
        gk_bf16_out.to_f32_vec().unwrap(),
        "flash::backward dK must be invariant to bf16 rounding of saved output O"
    );
    assert_eq!(
        gv_f32_out.to_f32_vec().unwrap(),
        gv_bf16_out.to_f32_vec().unwrap(),
        "flash::backward dV must be invariant to bf16 rounding of saved output O"
    );

    // Part C: General attention backward key-translation gauge invariance.
    let g_shape = [1usize, 2, 16, 16];
    let gn: usize = g_shape.iter().product();
    let qg = f32t(&be_exact, &rng.vec(gn, 0.5), &g_shape);
    let kg0_vec = rng.vec(gn, 0.5);
    let mut kg1_vec = kg0_vec.clone();
    for row in kg1_vec.chunks_exact_mut(16) {
        for (d_idx, slot) in row.iter_mut().enumerate() {
            *slot += (d_idx as f32 - 7.5) * 4.0;
        }
    }
    let kg0 = f32t(&be_exact, &kg0_vec, &g_shape);
    let kg1 = f32t(&be_exact, &kg1_vec, &g_shape);
    let vg = f32t(&be_exact, &rng.vec(gn, 0.5), &g_shape);
    let gyg = f32t(&be_exact, &rng.vec(gn, 0.5), &g_shape);
    let opts = AttentionOpts {
        causal: true,
        ..Default::default()
    };
    let (yg0, lseg0) = be_exact
        .attention_forward_saving(&qg, &kg0, &vg, opts)
        .unwrap();
    let (yg1, lseg1) = be_exact
        .attention_forward_saving(&qg, &kg1, &vg, opts)
        .unwrap();
    let grads0 = be_exact
        .attention_backward(&qg, &kg0, &vg, &yg0, &lseg0, &gyg, opts, false)
        .unwrap();
    let grads1 = be_exact
        .attention_backward(&qg, &kg1, &vg, &yg1, &lseg1, &gyg, opts, false)
        .unwrap();
    let dq0 = grads0.q.to_f32_vec().unwrap();
    let dq1 = grads1.q.to_f32_vec().unwrap();
    let max_dq = dq0.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    let diff_dq = dq0
        .iter()
        .zip(&dq1)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        diff_dq / max_dq < 5e-5,
        "general attention_backward dQ key-shift drift: {:.2e}",
        diff_dq / max_dq
    );
}

#[test]
fn t5_muon_ns5_singular_value_contraction() {
    let be = cpu(Numerics::Exact);
    let n = 16usize;
    // Construct a 2D gradient matrix with ill-conditioned singular spectrum:
    // G = diag(10.0, 5.0, 2.0, 1.0, ..., 0.02) in a Hadamard-like orthogonal basis.
    let mut g_data = vec![0.0f32; n * n];
    for i in 0..n {
        let sigma = 10.0f32 * (0.002f32).powf(i as f32 / (n - 1) as f32);
        g_data[i * n + i] = sigma;
    }
    let mut param = f32t(&be, &vec![0.0f32; n * n], &[n, n]);
    let grad = f32t(&be, &g_data, &[n, n]);
    let mut momentum = f32t(&be, &vec![0.0f32; n * n], &[n, n]);
    let cfg = MuonNs5Config {
        lr: 1.0,
        momentum: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        ..MuonNs5Config::nanolab_default()
    };
    be.muon_ns5_step(&mut param, &grad, &mut momentum, cfg)
        .unwrap();
    let p_after = param.to_f32_vec().unwrap();

    // Before NS5: max/min singular value ratio is 10.0 / 0.02 = 500.0.
    // After 5-step Newton-Schulz: diagonal singular values are contracted into [0.65, 1.25].
    let mut min_sv = f32::INFINITY;
    let mut max_sv = 0.0f32;
    for i in 0..n {
        let sv = (-p_after[i * n + i]).abs();
        min_sv = min_sv.min(sv);
        max_sv = max_sv.max(sv);
    }
    let cond_after = max_sv / min_sv;
    assert!(
        min_sv > 0.60 && max_sv < 1.30 && cond_after < 2.0,
        "Muon NS5 failed to contract singular spectrum: min_sv={min_sv}, max_sv={max_sv}, cond={cond_after}"
    );
}

#[test]
fn t6_chunked_gdn_fp32_scan_state_stability_under_bf16_autocast() {
    let (b, t, h, dk, dv) = (1usize, GDN_CHECKPOINT_TOKENS * 3, 2usize, 16usize, 16usize);
    let mut rng = SplitMix64(0x6D4_2026);
    let q_v = rng.vec(b * t * h * dk, 0.5);
    let k_v = rng.vec(b * t * h * dk, 0.5);
    let v_v = rng.vec(b * t * h * dv, 0.5);
    // Negative log-decay g in [-0.25, -0.01] for stable recurrent contraction
    let g_v: Vec<f32> = (0..b * t * h)
        .map(|_| -0.01 - 0.24 * rng.unit().abs())
        .collect();
    let beta_v: Vec<f32> = (0..b * t * h)
        .map(|_| 0.1 + 0.8 * rng.unit().abs())
        .collect();
    let s0_v = rng.vec(b * h * dk * dv, 0.2);
    let go_v = rng.vec(b * t * h * dv, 0.5);

    let be_f32 = cpu(Numerics::Exact);
    let ac_bf16 = Autocast::new(cpu(Numerics::Exact));

    let q = Tensor::from_f32(&q_v, &[b, t, h, dk], be_f32.budget()).unwrap();
    let k = Tensor::from_f32(&k_v, &[b, t, h, dk], be_f32.budget()).unwrap();
    let v = Tensor::from_f32(&v_v, &[b, t, h, dv], be_f32.budget()).unwrap();
    let g = Tensor::from_f32(&g_v, &[b, t, h], be_f32.budget()).unwrap();
    let beta = Tensor::from_f32(&beta_v, &[b, t, h], be_f32.budget()).unwrap();
    let s0 = Tensor::from_f32(&s0_v, &[b, h, dk, dv], be_f32.budget()).unwrap();
    let go = Tensor::from_f32(&go_v, &[b, t, h, dv], be_f32.budget()).unwrap();

    let fwd_f32 = be_f32
        .chunked_gdn_forward(GdnInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            initial_state: Some(&s0),
        })
        .unwrap();

    let (fwd_bf16, bwd_bf16) = {
        let _guard = ac_bf16.autocast_region(AutocastMode::Bf16).unwrap();
        let inputs = GdnInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            initial_state: Some(&s0),
        };
        let fwd = ac_bf16.chunked_gdn_forward(inputs).unwrap();
        let bwd = ac_bf16
            .chunked_gdn_backward(inputs, &fwd.checkpoints, &go, None)
            .unwrap();
        (fwd, bwd)
    };

    // Checkpoints and g gradient must remain F32 (not rounded to bf16), preventing scan drift.
    let ckpt_f32 = fwd_f32.checkpoints.to_f32_vec().unwrap();
    let ckpt_bf16 = fwd_bf16.checkpoints.to_f32_vec().unwrap();
    assert!(ckpt_bf16.iter().all(|x| x.is_finite()));
    // Verify at least one checkpoint value has low-16 mantissa bits preserved (unrounded F32).
    assert!(
        ckpt_bf16
            .iter()
            .any(|&x| x.to_bits() != round_f32_to_bf16(x).to_bits()),
        "GDN state checkpoints were unexpectedly rounded to bf16"
    );
    let dg_bf16 = bwd_bf16.g.to_f32_vec().unwrap();
    assert!(
        dg_bf16
            .iter()
            .any(|&x| x.to_bits() != round_f32_to_bf16(x).to_bits()),
        "GDN log-decay gradient dg was unexpectedly rounded to bf16"
    );

    // State norm across all 3 chunk checkpoints remains bounded and within 2% of exact F32.
    let max_ckpt = ckpt_f32
        .iter()
        .fold(0.0f32, |m, x| m.max(x.abs()))
        .max(1e-6);
    let max_ckpt_err = ckpt_f32
        .iter()
        .zip(&ckpt_bf16)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_ckpt_err / max_ckpt < 2e-2,
        "GDN multi-chunk checkpoint drift under Bf16 autocast too large: {:.2e}",
        max_ckpt_err / max_ckpt
    );
}

#[test]
fn t7_exact_margin_cross_entropy_and_gauge_projected_sdpa_backward() {
    let budget = Budget::new(1 << 28);
    let vocab = 64usize;
    let margins = [
        0.5f32, 1.0, 2.0, 4.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 18.0, 20.0, 21.0, 24.0,
    ];
    let rows = margins.len();
    let mut logits = vec![0.0f32; rows * vocab];
    let mut targets = vec![0u32; rows];
    for (r, &m) in margins.iter().enumerate() {
        targets[r] = (r % vocab) as u32;
        logits[r * vocab + (r % vocab)] = m;
    }
    let logits_t = Tensor::from_f32(&logits, &[rows, vocab], &budget).unwrap();
    let targets_t = Tensor::from_u32(&targets, &[rows], &budget).unwrap();

    let (loss, grad_t) =
        cross_entropy_mean_backward_exact_margin(&logits_t, &targets_t, None, &budget).unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    let grad = grad_t.to_f32_vec().unwrap();

    // Across ALL margins (including the [6.0, 20.78] pre-collapse transition window),
    // target-excluded summation eliminates 1.0f32 mantissa absorption (< 1e-5 rel err).
    for (r, &m) in margins.iter().enumerate() {
        let target_col = r % vocab;
        let row_g = &grad[r * vocab..(r + 1) * vocab];
        let row_sum: f64 = row_g.iter().map(|&g| f64::from(g)).sum();
        assert!(row_sum.abs() < 1e-7, "margin {m}: row_sum {row_sum:e}");
        let s_f64 = ((vocab - 1) as f64) * (-f64::from(m)).exp();
        let want_target = -(s_f64 / (1.0 + s_f64)) / (rows as f64);
        let got_target = f64::from(row_g[target_col]);
        let rel_err = (got_target - want_target).abs() / want_target.abs();
        assert!(
            rel_err < 1e-5,
            "margin {m}: got {got_target:e} vs want {want_target:e} (rel {rel_err:e})"
        );
    }

    // Gauge-projected causal SDPA backward at T = 24 (unblocked regime T <= 256):
    // Verify key-shift gauge invariance K <- K + 1 * c^T and immunity to output rounding.
    let be = cpu(Numerics::Exact);
    let (b, h, t, d) = (1usize, 2usize, 24usize, 16usize);
    let shape = [b, h, t, d];
    let mut rng = SplitMix64(0x77_2026);
    let q_v = rng.vec(b * h * t * d, 1.0);
    let k_v = rng.vec(b * h * t * d, 1.0);
    let v_v = rng.vec(b * h * t * d, 1.0);
    let gy_v = rng.vec(b * h * t * d, 1.0);

    let shift_c = rng.vec(b * h * d, 3.5);
    let mut k_shift_v = k_v.clone();
    for bh in 0..(b * h) {
        for tok in 0..t {
            for dim_i in 0..d {
                k_shift_v[(bh * t + tok) * d + dim_i] += shift_c[bh * d + dim_i];
            }
        }
    }

    let q = f32t(&be, &q_v, &shape);
    let k = f32t(&be, &k_v, &shape);
    let k_shift = f32t(&be, &k_shift_v, &shape);
    let v = f32t(&be, &v_v, &shape);
    let gy = f32t(&be, &gy_v, &shape);

    let (_, lse0) = be.causal_sdpa_forward(&q, &k, &v, None).unwrap();
    let (_, lse1) = be.causal_sdpa_forward(&q, &k_shift, &v, None).unwrap();

    let (gq0, gk0, gv0) =
        causal_sdpa_backward_gauge_projected(&q, &k, &v, &lse0, &gy, None, &budget).unwrap();
    let (gq1, gk1, gv1) =
        causal_sdpa_backward_gauge_projected(&q, &k_shift, &v, &lse1, &gy, None, &budget).unwrap();

    for (name, a, b_ten) in [("gq", gq0, gq1), ("gk", gk0, gk1), ("gv", gv0, gv1)] {
        let av = a.to_f32_vec().unwrap();
        let bv = b_ten.to_f32_vec().unwrap();
        let max_ref = av.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let max_diff = av
            .iter()
            .zip(&bv)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            max_diff / max_ref < 5e-6,
            "gauge-projected {name} key-shift drift {:.2e}",
            max_diff / max_ref
        );
    }
}

#[test]
fn t8_quantized_i8_accumulator_bound_and_24_wanda_sparse_slow_weights() {
    let budget = Budget::new(1 << 28);
    let be = cpu(Numerics::Exact);

    // 1. Verify Theorem 5 INT8 -> INT32 accumulator overflow boundary:
    assert!(check_i8_accumulator_bound("test", I8_ACCUMULATOR_MAX_K).is_ok());
    let overflow_err = check_i8_accumulator_bound("test", I8_ACCUMULATOR_MAX_K + 1).unwrap_err();
    assert!(matches!(overflow_err, OjasError::OutOfRange { .. }));

    // 2. Verify QuantizedWeightI8 linear_i8_forward accuracy against dense F32 linear_forward:
    let (rows, k_in, n_out) = (8usize, 64usize, 16usize);
    let mut rng = SplitMix64(0x18_24);
    let x_v = rng.vec(rows * k_in, 1.2);
    let w_v = rng.vec(n_out * k_in, 0.8);
    let x_t = Tensor::from_f32(&x_v, &[rows, k_in], &budget).unwrap();
    let w_t = Tensor::from_f32(&w_v, &[n_out, k_in], &budget).unwrap();

    let ref_y = be.linear_forward(&x_t, &w_t).unwrap().to_f32_vec().unwrap();
    let q_w = QuantizedWeightI8::quantize(&w_t, &budget).unwrap();
    let got_y = q_w
        .linear_i8_forward(&x_t, &budget)
        .unwrap()
        .to_f32_vec()
        .unwrap();

    let max_ref = ref_y.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    let max_err = ref_y
        .iter()
        .zip(&got_y)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_err / max_ref < 2.5e-2,
        "INT8 linear relative error {:.4} exceeded 2.5%",
        max_err / max_ref
    );

    // 3. Verify Wanda 2:4 structured sparsity + prune-then-quantize INT8 pipeline:
    let act_norms: Vec<f32> = (0..k_in).map(|j| 0.5 + (j as f32) * 0.05).collect();
    let sparse_w = Sparse24Weight::prune_24_wanda(&w_t, Some(&act_norms)).unwrap();
    assert!(
        sparse_w.verify_24_invariant(),
        "2:4 sparsity invariant violated"
    );
    let zero_count = sparse_w.pruned_f32.iter().filter(|&&v| v == 0.0).count();
    assert!(
        zero_count * 2 >= sparse_w.pruned_f32.len(),
        "2:4 sparsity must zero at least 50% of weights"
    );

    let y_24 = sparse_w
        .linear_24_forward(&x_t, &budget)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let pq_w = sparse_w.quantize_i8(&budget).unwrap();
    let y_24_i8 = pq_w
        .linear_i8_forward(&x_t, &budget)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let max_24 = y_24.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    let pq_err = y_24
        .iter()
        .zip(&y_24_i8)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        pq_err / max_24 < 2.5e-2,
        "Prune-then-quantize 2:4 + INT8 error {:.4} exceeded 2.5%",
        pq_err / max_24
    );
}
