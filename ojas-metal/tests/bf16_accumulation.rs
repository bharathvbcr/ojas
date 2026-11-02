//! True bf16 accumulation on the Metal GPU backend (`MetalBackend::with_bf16_accumulation(true)`):
//! every matrix multiply, convolution, attention score/value/gradient reduction,
//! per-head gate, and gradient/bias sum accumulates step by step in bf16.
#![cfg(target_os = "macos")]

mod common;

use common::{down, host, host_u32, metal, ok, rand, up, values};
use ojas_core::{
    bf16_to_f32, f32_to_bf16, AttentionOpts, Autocast, AutocastMode, Backend, Budget, CeChunk,
    CeGrads, Conv2dParams, ConvTranspose2dParams, MuonNs5Config, Ns5Precision, OjasError, Tensor,
};

fn r(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

fn bf16_add(a: f32, b: f32) -> f32 {
    r(r(a) + r(b))
}

fn bf16_mac(acc: f32, a: f32, b: f32) -> f32 {
    r(acc + r(r(a) * r(b)))
}

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

fn f32_bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|v| v.to_bits()).collect()
}

fn host_gemm_nt_bf16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc = bf16_mac(acc, a[row * k + i], b[col * k + i]);
            }
            c[row * n + col] = acc;
        }
    }
    c
}

fn host_gemm_nn_bf16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc = bf16_mac(acc, a[row * k + i], b[i * n + col]);
            }
            c[row * n + col] = acc;
        }
    }
    c
}

fn host_gemm_tn_bf16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc = bf16_mac(acc, a[i * m + row], b[i * n + col]);
            }
            c[row * n + col] = acc;
        }
    }
    c
}

/// Vector `[1.0, eps, eps, ...]` where `r(1.0 + r(eps)) == 1.0` in bf16
/// step-by-step accumulation, whereas f32 accumulation sums to `1.0 + (k-1)*r(eps)`.
fn absorbing_row(k: usize) -> Vec<f32> {
    let eps = r(0.002);
    assert_eq!(bf16_add(1.0, eps), 1.0);
    let mut v = vec![eps; k];
    v[0] = 1.0;
    v
}

#[test]
fn bf16_accumulation_toggle_and_budget_inheritance() {
    let m = metal();
    assert!(!m.bf16_accumulation(), "default is f32 accumulation");
    let mb = m.with_bf16_accumulation(true);
    assert!(mb.bf16_accumulation());
    assert!(m.bf16_accumulation());
    let sub = mb.with_budget(Budget::new(64 * 1024 * 1024));
    assert!(sub.bf16_accumulation());
    m.set_bf16_accumulation(false);
    assert!(!m.bf16_accumulation());
}

#[test]
fn linear_forward_and_backward_accumulate_in_bf16() {
    let m = metal().with_bf16_accumulation(false);
    let k = 32usize;
    let row = absorbing_row(k);
    let ones = vec![1.0f32; k];
    let x = up(&m, &host(&row, &[1, k]));
    let w = up(&m, &host(&ones, &[1, k]));
    let b = up(&m, &host(&[r(0.002)], &[1]));

    // Under default f32 accumulation, 1.0 + 31 * r(0.002) > 1.05.
    let y_f32 = down(&ok("linear f32", m.linear_forward(&x, &w)))[0];
    assert!(y_f32 > 1.05, "f32 sum was {y_f32}");

    // Under bf16 accumulation, every step rounds to bf16 so the sum is 1.0.
    m.set_bf16_accumulation(true);
    let y_bf16 = down(&ok("linear bf16", m.linear_forward(&x, &w)))[0];
    assert_eq!(y_bf16.to_bits(), 1.0f32.to_bits());
    let yb_bf16 = down(&ok("linear_bias bf16", m.linear_bias_forward(&x, &w, &b)))[0];
    assert_eq!(yb_bf16.to_bits(), 1.0f32.to_bits());

    // Multi-row, multi-col bit-exactness against host bf16 GEMM on random data,
    // with F32 operands, Bf16 operands, Autocast(Bf16), and unaligned views.
    for &(rows, kin, nout) in &[(3usize, 17usize, 5usize), (19, 64, 37)] {
        let xd = values(rows * kin, 101, 1.0);
        let wd = values(nout * kin, 102, 1.0);
        let bd = values(nout, 103, 0.5);
        let gd = values(rows * nout, 104, 1.0);

        let xt = up(&m, &host(&xd, &[rows, kin]));
        let wt = up(&m, &host(&wd, &[nout, kin]));
        let bt = up(&m, &host(&bd, &[nout]));
        let gt = up(&m, &host(&gd, &[rows, nout]));

        let want_y = host_gemm_nt_bf16(&xd, &wd, rows, kin, nout);
        let want_yb: Vec<f32> = want_y
            .iter()
            .enumerate()
            .map(|(i, &v)| bf16_add(v, bd[i % nout]))
            .collect();
        let want_gx = host_gemm_nn_bf16(&gd, &wd, rows, nout, kin);
        let want_gw = host_gemm_tn_bf16(&gd, &xd, nout, rows, kin);
        let mut want_gb = vec![0.0f32; nout];
        for col in 0..nout {
            let mut acc = 0.0f32;
            for row in 0..rows {
                acc = bf16_add(acc, gd[row * nout + col]);
            }
            want_gb[col] = acc;
        }

        let got_y = ok("y", m.linear_forward(&xt, &wt));
        let got_yb = ok("yb", m.linear_bias_forward(&xt, &wt, &bt));
        let (got_gx, got_gw) = ok("bwd", m.linear_backward(&xt, &wt, &gt));
        let got_lb = ok("lb_bwd", m.linear_bias_backward(&xt, &wt, &bt, &gt));
        let got_gxi = ok("bwd_in", m.linear_backward_input(&xt, &wt, &gt));
        ok("sync", m.sync());

        assert_eq!(bits(&got_y), f32_bits(&want_y));
        assert_eq!(bits(&got_yb), f32_bits(&want_yb));
        assert_eq!(bits(&got_gx), f32_bits(&want_gx));
        assert_eq!(bits(&got_gw), f32_bits(&want_gw));
        assert_eq!(bits(&got_lb.input), f32_bits(&want_gx));
        assert_eq!(bits(&got_lb.weight), f32_bits(&want_gw));
        assert_eq!(bits(&got_lb.bias), f32_bits(&want_gb));
        assert_eq!(bits(&got_gxi), f32_bits(&want_gx));

        // Bf16 operands produce identical bits.
        let xh = ok("to_bf16 x", m.to_bf16(&xt));
        let wh = ok("to_bf16 w", m.to_bf16(&wt));
        let gh = ok("to_bf16 g", m.to_bf16(&gt));
        let got_yh = ok("yh", m.linear_forward(&xh, &wh));
        let got_ybh = ok("ybh", m.linear_bias_forward(&xh, &wh, &bt));
        let (got_gxh, got_gwh) = ok("bwdh", m.linear_backward(&xh, &wh, &gh));
        let got_lbh = ok("lb_bwdh", m.linear_bias_backward(&xh, &wh, &bt, &gh));
        ok("sync half", m.sync());
        assert_eq!(bits(&got_yh), f32_bits(&want_y));
        assert_eq!(bits(&got_ybh), f32_bits(&want_yb));
        assert_eq!(bits(&got_gxh), f32_bits(&want_gx));
        assert_eq!(bits(&got_gwh), f32_bits(&want_gw));
        assert_eq!(bits(&got_lbh.bias), f32_bits(&want_gb));

        // Autocast(Bf16) region produces identical bits.
        let ac = Autocast::new(m.clone());
        let _guard = ok("autocast", ac.autocast_region(AutocastMode::Bf16));
        let got_yac = ok("yac", ac.linear_forward(&xt, &wt));
        let got_ybac = ok("ybac", ac.linear_bias_forward(&xt, &wt, &bt));
        let (got_gxac, got_gwac) = ok("bwdac", ac.linear_backward(&xt, &wt, &gt));
        let got_lbac = ok("lb_bwdac", ac.linear_bias_backward(&xt, &wt, &bt, &gt));
        ok("sync ac", ac.sync());
        assert_eq!(bits(&got_yac), f32_bits(&want_y));
        assert_eq!(bits(&got_ybac), f32_bits(&want_yb));
        assert_eq!(bits(&got_gxac), f32_bits(&want_gx));
        assert_eq!(bits(&got_gwac), f32_bits(&want_gw));
        assert_eq!(bits(&got_lbac.bias), f32_bits(&want_gb));
    }

    // Unaligned view (offset of 1 f32 element = 4 bytes, not 16-byte aligned).
    let mut padded = vec![0.0f32; 1 + 4 * 8];
    let xd = values(4 * 8, 105, 1.0);
    padded[1..].copy_from_slice(&xd);
    let base = up(&m, &host(&padded, &[1 + 4 * 8]));
    let x_unaligned = ok("unaligned view", base.view(&[4, 8], &[8, 1], 4));
    let wd = values(3 * 8, 106, 1.0);
    let wt = up(&m, &host(&wd, &[3, 8]));
    let got_u = ok("linear unaligned", m.linear_forward(&x_unaligned, &wt));
    ok("sync u", m.sync());
    assert_eq!(
        bits(&got_u),
        f32_bits(&host_gemm_nt_bf16(&xd, &wd, 4, 8, 3))
    );

    m.set_bf16_accumulation(false);
}

#[test]
fn conv2d_im2col_pointwise_depthwise_and_transpose_accumulate_in_bf16() {
    let m = metal().with_bf16_accumulation(false);
    let c = 32usize;
    let row = absorbing_row(c);
    let ones = vec![1.0f32; c];

    // 1x1 pointwise conv2d: [1, c, 1, 1] * [1, c, 1, 1]
    let x_pw = up(&m, &host(&row, &[1, c, 1, 1]));
    let w_pw = up(&m, &host(&ones, &[1, c, 1, 1]));
    let b_pw = up(&m, &host(&[r(0.002)], &[1]));
    let p_pw = Conv2dParams::new();
    let y_pw_f32 = down(&ok(
        "pw f32",
        m.conv2d_forward(&x_pw, &w_pw, Some(&b_pw), p_pw),
    ))[0];
    assert!(y_pw_f32 > 1.05);

    m.set_bf16_accumulation(true);
    let y_pw_bf16 = down(&ok(
        "pw bf16",
        m.conv2d_forward(&x_pw, &w_pw, Some(&b_pw), p_pw),
    ))[0];
    assert_eq!(y_pw_bf16.to_bits(), 1.0f32.to_bits());

    // 3x3 im2col conv2d: [1, 4, 3, 3] * [1, 4, 3, 3] (k = 36)
    let k36 = 36usize;
    let x_im = up(&m, &host(&absorbing_row(k36), &[1, 4, 3, 3]));
    let w_im = up(&m, &host(&vec![1.0f32; k36], &[1, 4, 3, 3]));
    let y_im_bf16 = down(&ok(
        "im2col bf16",
        m.conv2d_forward(&x_im, &w_im, Some(&b_pw), p_pw),
    ))[0];
    assert_eq!(y_im_bf16.to_bits(), 1.0f32.to_bits());

    // Depthwise conv2d: [1, 2, 4, 4] * [2, 1, 4, 4] (kh*kw = 16, groups = 2)
    let mut dw_x = absorbing_row(16);
    dw_x.extend(absorbing_row(16));
    let x_dw = up(&m, &host(&dw_x, &[1, 2, 4, 4]));
    let w_dw = up(&m, &host(&[1.0f32; 32], &[2, 1, 4, 4]));
    let b_dw = up(&m, &host(&[r(0.002), r(0.002)], &[2]));
    let p_dw = Conv2dParams {
        groups: 2,
        ..Conv2dParams::new()
    };
    let y_dw = down(&ok(
        "dw bf16",
        m.conv2d_forward(&x_dw, &w_dw, Some(&b_dw), p_dw),
    ));
    assert_eq!(y_dw, vec![1.0f32, 1.0f32]);

    // Depthwise conv2d backward: sum over spatial positions in bf16.
    let gy_dw = up(&m, &host(&absorbing_row(16).repeat(2), &[1, 2, 4, 4]));
    let x_dw1 = up(&m, &host(&[1.0f32; 32], &[1, 2, 4, 4]));
    let w_dw1 = up(&m, &host(&[1.0f32; 2], &[2, 1, 1, 1]));
    let g_dw = ok(
        "dw bwd bf16",
        m.conv2d_backward(&x_dw1, &w_dw1, Some(&b_dw), &gy_dw, p_dw),
    );
    assert_eq!(down(&g_dw.weight), vec![1.0f32, 1.0f32]);
    assert_eq!(down(&g_dw.bias.unwrap()), vec![1.0f32, 1.0f32]);

    // Conv2d backward (groups == 1) with multi-split (batch * oh * ow >= 512).
    let gy_split_data = absorbing_row(512);
    let gy_split = up(&m, &host(&gy_split_data, &[2, 1, 16, 16]));
    let x_split = up(&m, &host(&[1.0f32; 512], &[2, 1, 16, 16]));
    let w_split = up(&m, &host(&[1.0f32], &[1, 1, 1, 1]));
    let g_s = ok(
        "conv2d bwd split bf16",
        m.conv2d_backward(
            &x_split,
            &w_split,
            Some(&b_pw),
            &gy_split,
            Conv2dParams::new(),
        ),
    );
    m.set_bf16_accumulation(false);
    let g_sf = ok(
        "conv2d bwd split f32",
        m.conv2d_backward(
            &x_split,
            &w_split,
            Some(&b_pw),
            &gy_split,
            Conv2dParams::new(),
        ),
    );
    assert_ne!(bits(&g_s.weight), bits(&g_sf.weight));
    assert_ne!(
        bits(g_s.bias.as_ref().unwrap()),
        bits(g_sf.bias.as_ref().unwrap())
    );

    // ConvTranspose2d forward & backward in bf16 accumulation.
    m.set_bf16_accumulation(true);
    let p_ct = ConvTranspose2dParams {
        stride: [2, 2],
        ..ConvTranspose2dParams::new()
    };
    let x_ct = up(&m, &host(&absorbing_row(c), &[1, c, 1, 1]));
    let w_ct = up(&m, &host(&vec![1.0f32; c * 4], &[c, 1, 2, 2]));
    let b_ct = up(&m, &host(&[r(0.002)], &[1]));
    let y_ct = down(&ok(
        "convt fwd bf16",
        m.conv_transpose2d_forward(&x_ct, &w_ct, Some(&b_ct), p_ct),
    ));
    assert_eq!(y_ct, vec![1.0f32; 4]);

    let gy_ct = up(&m, &host(&absorbing_row(16), &[1, 1, 4, 4]));
    let x_ct_bwd = up(&m, &host(&[1.0f32; 4], &[1, 1, 2, 2]));
    let w_ct_bwd = up(&m, &host(&[1.0f32; 4], &[1, 1, 2, 2]));
    let g_ct = ok(
        "convt bwd bf16",
        m.conv_transpose2d_backward(&x_ct_bwd, &w_ct_bwd, Some(&b_ct), &gy_ct, p_ct),
    );
    assert_eq!(down(&g_ct.bias.unwrap()), vec![1.0f32]);

    m.set_bf16_accumulation(false);
}

#[test]
fn causal_sdpa_and_general_attention_accumulate_in_bf16() {
    let m = metal().with_bf16_accumulation(false);
    for &d in &[16usize, 32, 64, 128, 256] {
        // Construct q = absorbing_row(d), k = [1, 1, ...], so q . k differs between
        // bf16 accumulation (1.0) and f32 accumulation (> 1.03).
        let (b, h, kvh, t) = (1usize, 2usize, 1usize, 4usize);
        let qd = absorbing_row(d).repeat(b * h * t);
        let kd = values(b * kvh * t * d, 200 + d as u64, 1.0);
        let vd = values(b * kvh * t * d, 300 + d as u64, 1.0);
        let gd = values(b * h * t * d, 400 + d as u64, 1.0);

        let q = up(&m, &host(&qd, &[b, h, t, d]));
        let k = up(&m, &host(&kd, &[b, kvh, t, d]));
        let v = up(&m, &host(&vd, &[b, kvh, t, d]));
        let gy = up(&m, &host(&gd, &[b, h, t, d]));

        m.set_bf16_accumulation(false);
        let (y_f32, lse_f32) = ok("sdpa f32", m.causal_sdpa_forward(&q, &k, &v, None));
        let (dq_f32, dk_f32, dv_f32) = ok(
            "sdpa bwd f32",
            m.causal_sdpa_backward(&q, &k, &v, &y_f32, &lse_f32, &gy, None),
        );

        m.set_bf16_accumulation(true);
        let (y_bf16, lse_bf16) = ok("sdpa bf16", m.causal_sdpa_forward(&q, &k, &v, None));
        let (dq_bf16, dk_bf16, dv_bf16) = ok(
            "sdpa bwd bf16",
            m.causal_sdpa_backward(&q, &k, &v, &y_bf16, &lse_bf16, &gy, None),
        );
        ok("sync sdpa", m.sync());

        assert_ne!(bits(&y_bf16), bits(&y_f32), "sdpa fwd d={d}");
        assert_ne!(bits(&dq_bf16), bits(&dq_f32), "sdpa dq d={d}");
        assert_ne!(bits(&dk_bf16), bits(&dk_f32), "sdpa dk d={d}");
        assert_ne!(bits(&dv_bf16), bits(&dv_f32), "sdpa dv d={d}");

        // General attention forward/saving/backward (including d=64 encoder-eligible shape).
        let scale = Some(1.0 / (d as f32).sqrt());
        for opts in [
            AttentionOpts {
                scale,
                ..Default::default()
            },
            AttentionOpts {
                causal: true,
                scale,
                ..Default::default()
            },
            AttentionOpts {
                radius: Some(2),
                scale,
                ..Default::default()
            },
        ] {
            m.set_bf16_accumulation(false);
            let (ay_f32, alse_f32) =
                ok("attn fwd f32", m.attention_forward_saving(&q, &k, &v, opts));
            let ag_f32 = ok(
                "attn bwd f32",
                m.attention_backward(&q, &k, &v, &ay_f32, &alse_f32, &gy, opts, false),
            );

            m.set_bf16_accumulation(true);
            let ay_only = ok("attn fwd only bf16", m.attention_forward(&q, &k, &v, opts));
            let (ay_bf16, alse_bf16) = ok(
                "attn fwd bf16",
                m.attention_forward_saving(&q, &k, &v, opts),
            );
            let ag_bf16 = ok(
                "attn bwd bf16",
                m.attention_backward(&q, &k, &v, &ay_bf16, &alse_bf16, &gy, opts, false),
            );
            ok("sync attn", m.sync());
            assert_eq!(bits(&ay_only), bits(&ay_bf16), "d={d} opts={opts:?}");
            assert_ne!(
                bits(&ay_bf16),
                bits(&ay_f32),
                "attn fwd d={d} opts={opts:?}"
            );
            assert_ne!(
                bits(&ag_bf16.q),
                bits(&ag_f32.q),
                "attn dq d={d} opts={opts:?}"
            );
            assert_ne!(
                bits(&ag_bf16.k),
                bits(&ag_f32.k),
                "attn dk d={d} opts={opts:?}"
            );
            assert_ne!(
                bits(&ag_bf16.v),
                bits(&ag_f32.v),
                "attn dv d={d} opts={opts:?}"
            );
        }
    }

    // Also test attention_forward/backward on a non-tile head dim (d=24 -> scalar ojas_attn_gen)
    // and with additive bias + bias_grad + key_lens.
    let (b, h, kvh, tq, tk, d) = (1usize, 2usize, 2usize, 2usize, 64usize, 24usize);
    let q = up(&m, &rand(&[b, h, tq, d], 501, 1.0));
    let k = up(&m, &rand(&[b, kvh, tk, d], 502, 1.0));
    let v = up(&m, &rand(&[b, kvh, tk, d], 503, 1.0));
    let bias = up(&m, &rand(&[1, h, tq, tk], 504, 0.25));
    let gy = up(&m, &rand(&[b, h, tq, d], 505, 1.0));
    let lens = up(&m, &host_u32(&[48], &[1]));
    let opts = AttentionOpts {
        bias: Some(&bias),
        key_lens: Some(&lens),
        scale: Some(0.2),
        ..Default::default()
    };

    m.set_bf16_accumulation(false);
    let (y_f32, lse_f32) = ok(
        "scalar attn f32",
        m.attention_forward_saving(&q, &k, &v, opts),
    );
    let g_f32 = ok(
        "scalar attn bwd f32",
        m.attention_backward(&q, &k, &v, &y_f32, &lse_f32, &gy, opts, true),
    );

    m.set_bf16_accumulation(true);
    let (y_bf16, lse_bf16) = ok(
        "scalar attn bf16",
        m.attention_forward_saving(&q, &k, &v, opts),
    );
    let g_bf16 = ok(
        "scalar attn bwd bf16",
        m.attention_backward(&q, &k, &v, &y_bf16, &lse_bf16, &gy, opts, true),
    );
    ok("sync scalar attn", m.sync());
    assert_ne!(bits(&y_bf16), bits(&y_f32));
    assert_ne!(bits(&g_bf16.q), bits(&g_f32.q));
    assert!(g_bf16.bias.is_some() && g_f32.bias.is_some());

    // Explicit flash, split-K, scalar, and padq routes at d=32 with tk=64 (2 splits).
    let (b32, h32, tq32, tk32, d32) = (1usize, 2usize, 4usize, 64usize, 32usize);
    let q32 = up(
        &m,
        &host(
            &absorbing_row(d32).repeat(b32 * h32 * tq32),
            &[b32, h32, tq32, d32],
        ),
    );
    let k32 = up(&m, &rand(&[b32, h32, tk32, d32], 511, 1.0));
    let v32 = up(&m, &rand(&[b32, h32, tk32, d32], 512, 1.0));
    let opts32 = AttentionOpts::default();

    m.set_bf16_accumulation(false);
    let (yf_f32, _) = ok(
        "flash f32",
        m.attention_forward_saving_flash(&q32, &k32, &v32, opts32),
    );
    let (ys_f32, _) = ok(
        "splitk f32",
        m.attention_forward_saving_splitk(&q32, &k32, &v32, opts32),
    );
    let (ysc_f32, _) = ok(
        "scalar32 f32",
        m.attention_forward_saving_scalar(&q32, &k32, &v32, opts32),
    );
    let yp_f32 = ok(
        "padq f32",
        m.attention_forward_padq(&q32, &k32, &v32, opts32),
    );

    m.set_bf16_accumulation(true);
    let (yf_bf16, _) = ok(
        "flash bf16",
        m.attention_forward_saving_flash(&q32, &k32, &v32, opts32),
    );
    let (ys_bf16, _) = ok(
        "splitk bf16",
        m.attention_forward_saving_splitk(&q32, &k32, &v32, opts32),
    );
    let (ysc_bf16, _) = ok(
        "scalar32 bf16",
        m.attention_forward_saving_scalar(&q32, &k32, &v32, opts32),
    );
    let yp_bf16 = ok(
        "padq bf16",
        m.attention_forward_padq(&q32, &k32, &v32, opts32),
    );
    ok("sync 32 routes", m.sync());
    assert_ne!(bits(&yf_bf16), bits(&yf_f32));
    assert_ne!(bits(&ys_bf16), bits(&ys_f32));
    assert_ne!(bits(&ysc_bf16), bits(&ysc_f32));
    assert_ne!(bits(&yp_bf16), bits(&yp_f32));

    m.set_bf16_accumulation(false);
}

#[test]
fn cached_attention_and_per_head_gate_accumulate_in_bf16() {
    let m = metal().with_bf16_accumulation(false);

    // Cached attention: single split (kv_len=8) and multi-split (kv_len=128).
    for &kv_len in &[8usize, 128usize] {
        let (b, tq, h, kvh, d, cap) = (1usize, 1usize, 2usize, 1usize, 32usize, 128usize);
        let q = up(
            &m,
            &host(&absorbing_row(d).repeat(b * tq * h), &[b, tq, h, d]),
        );
        let k = up(&m, &rand(&[b, cap, kvh, d], 600 + kv_len as u64, 1.0));
        let v = up(&m, &rand(&[b, cap, kvh, d], 610 + kv_len as u64, 1.0));

        m.set_bf16_accumulation(false);
        let y_f32 = ok(
            "cached_attn f32",
            m.cached_attention_forward(&q, &k, &v, kv_len, None),
        );
        m.set_bf16_accumulation(true);
        let y_bf16 = ok(
            "cached_attn bf16",
            m.cached_attention_forward(&q, &k, &v, kv_len, None),
        );
        ok("sync cached_attn", m.sync());
        assert_ne!(bits(&y_bf16), bits(&y_f32), "cached_attn kv_len={kv_len}");
    }

    // Per-head sigmoid gate forward, saving, backward, and backward_saved.
    let (b, t, h, dh, din) = (2usize, 8usize, 2usize, 16usize, 32usize);
    let x = up(&m, &rand(&[b, t, din], 701, 1.0));
    let w = up(&m, &rand(&[h, din], 702, 1.0));
    let bias = up(&m, &rand(&[h], 703, 0.5));
    let attn = up(&m, &rand(&[b, t, h, dh], 704, 1.0));
    let gy = up(&m, &rand(&[b, t, h, dh], 705, 1.0));

    m.set_bf16_accumulation(false);
    let (y_f32, scale_f32) = ok(
        "gate fwd f32",
        m.per_head_sigmoid_gate_forward_saving(&x, &w, &bias, &attn),
    );
    let scale_f32 = scale_f32.unwrap();
    let g_f32 = ok(
        "gate bwd f32",
        m.per_head_sigmoid_gate_backward(&x, &w, &bias, &attn, &gy),
    );
    let gs_f32 = ok(
        "gate bwd saved f32",
        m.per_head_sigmoid_gate_backward_saved(&x, &w, &bias, &attn, &gy, &scale_f32),
    );

    m.set_bf16_accumulation(true);
    let y_only = ok(
        "gate fwd only bf16",
        m.per_head_sigmoid_gate_forward(&x, &w, &bias, &attn),
    );
    let (y_bf16, scale_bf16) = ok(
        "gate fwd bf16",
        m.per_head_sigmoid_gate_forward_saving(&x, &w, &bias, &attn),
    );
    let scale_bf16 = scale_bf16.unwrap();
    let g_bf16 = ok(
        "gate bwd bf16",
        m.per_head_sigmoid_gate_backward(&x, &w, &bias, &attn, &gy),
    );
    let gs_bf16 = ok(
        "gate bwd saved bf16",
        m.per_head_sigmoid_gate_backward_saved(&x, &w, &bias, &attn, &gy, &scale_bf16),
    );
    ok("sync gate", m.sync());

    assert_eq!(bits(&y_only), bits(&y_bf16));
    assert_ne!(bits(&y_bf16), bits(&y_f32));
    assert_ne!(bits(&g_bf16.input), bits(&g_f32.input));
    assert_ne!(bits(&g_bf16.weight), bits(&g_f32.weight));
    assert_ne!(bits(&g_bf16.bias), bits(&g_f32.bias));
    assert_eq!(bits(&g_bf16.input), bits(&gs_bf16.input));
    assert_eq!(bits(&g_bf16.weight), bits(&gs_bf16.weight));
    assert_eq!(bits(&g_bf16.bias), bits(&gs_bf16.bias));
    let _ = (g_f32.attn_out, gs_f32);

    m.set_bf16_accumulation(false);
}

#[test]
fn reductions_embedding_accumulate_grad_linear_ce_and_muon_accumulate_in_bf16() {
    let m = metal().with_bf16_accumulation(true);

    // 1. embedding_backward: 32 identical token ids summing absorbing_row(32).
    let k = 32usize;
    let ids = up(&m, &host_u32(&vec![0u32; k], &[1, k]));
    let w_emb = up(&m, &host(&[0.0f32, 0.0f32], &[2, 1]));
    let gy_emb = up(&m, &host(&absorbing_row(k), &[1, k, 1]));
    let gw_emb = down(&ok(
        "emb bwd bf16",
        m.embedding_backward(&w_emb, &ids, &gy_emb),
    ));
    assert_eq!(gw_emb, vec![1.0f32, 0.0f32]);

    // 2. broadcast_add_backward & broadcast_mul_backward (1 split and multi-split).
    for &rows in &[32usize, 512usize] {
        let x_b = up(&m, &host(&vec![1.0f32; rows], &[rows, 1]));
        let y_b = up(&m, &host(&[1.0f32], &[1, 1]));
        let gy_b = up(&m, &host(&absorbing_row(rows), &[rows, 1]));
        let (_gx_a, gy_a) = ok(
            "bcast_add_bwd bf16",
            m.broadcast_add_backward(&x_b, &y_b, &gy_b),
        );
        let (_gx_m, gy_m) = ok(
            "bcast_mul_bwd bf16",
            m.broadcast_mul_backward(&x_b, &y_b, &gy_b),
        );
        if rows == 32 {
            assert_eq!(down(&gy_a), vec![1.0f32]);
            assert_eq!(down(&gy_m), vec![1.0f32]);
        } else {
            m.set_bf16_accumulation(false);
            let (_, gy_af) = ok(
                "bcast_add_bwd f32",
                m.broadcast_add_backward(&x_b, &y_b, &gy_b),
            );
            m.set_bf16_accumulation(true);
            assert_ne!(bits(&gy_a), bits(&gy_af));
        }
    }

    // 3. accumulate_grad (shared and in-place): 1.0 + r(0.002) rounds to 1.0 in bf16.
    let mut acc = up(&m, &host(&[1.0f32, 2.0f32], &[2]));
    let g_small = up(&m, &host(&[r(0.002), r(0.002)], &[2]));
    let shared_clone = acc.clone();
    ok("acc shared", m.accumulate_grad(&mut acc, &g_small));
    assert_eq!(down(&acc), vec![1.0f32, 2.0f32]);
    drop(shared_clone);
    ok("acc in_place", m.accumulate_grad(&mut acc, &g_small));
    assert_eq!(down(&acc), vec![1.0f32, 2.0f32]);

    // 4. linear_cross_entropy_mean (with F32 and Bf16 weights, multi-tile).
    let (n, d, v) = (8usize, 32usize, 16usize);
    let x_lce = up(&m, &rand(&[n, d], 801, 1.0));
    let w_lce = up(&m, &rand(&[v, d], 802, 1.0));
    let wh_lce = ok("to_bf16 lce", m.to_bf16(&w_lce));
    let wf_lce = ok("to_f32 lce", m.to_f32(&wh_lce));
    let t_lce = up(&m, &host_u32(&[0, 1, 2, 3, 4, 5, 6, 7], &[n]));
    let chunk = CeChunk { rows: 4, cols: 8 };

    m.set_bf16_accumulation(false);
    let lce_f32 = ok(
        "lce f32",
        m.linear_cross_entropy_mean(&x_lce, &wf_lce, &t_lce, None, chunk, CeGrads::BOTH),
    );
    m.set_bf16_accumulation(true);
    let lce_bf16 = ok(
        "lce bf16",
        m.linear_cross_entropy_mean(&x_lce, &wf_lce, &t_lce, None, chunk, CeGrads::BOTH),
    );
    let lce_h = ok(
        "lce half bf16",
        m.linear_cross_entropy_mean(&x_lce, &wh_lce, &t_lce, None, chunk, CeGrads::BOTH),
    );
    ok("sync lce", m.sync());
    assert_ne!(
        bits(lce_bf16.grad_input.as_ref().unwrap()),
        bits(lce_f32.grad_input.as_ref().unwrap())
    );
    assert_ne!(
        bits(lce_bf16.grad_weight.as_ref().unwrap()),
        bits(lce_f32.grad_weight.as_ref().unwrap())
    );
    assert_eq!(bits(&lce_bf16.loss), bits(&lce_h.loss));
    assert_eq!(
        bits(&lce_bf16.grad_input.unwrap()),
        bits(&lce_h.grad_input.unwrap())
    );
    assert_eq!(
        bits(&lce_bf16.grad_weight.unwrap()),
        bits(&lce_h.grad_weight.unwrap())
    );

    // 5. muon_ns5_step: Newton-Schulz GEMMs accumulate in bf16 when enabled.
    for prec in [Ns5Precision::F32, Ns5Precision::Bf16] {
        let mut p1 = up(&m, &rand(&[16, 32], 901, 1.0));
        let mut m1 = up(&m, &rand(&[16, 32], 902, 0.1));
        let mut p2 = up(&m, &host(&down(&p1), &[16, 32]));
        let mut m2 = up(&m, &host(&down(&m1), &[16, 32]));
        let g = up(&m, &rand(&[16, 32], 903, 0.5));
        let cfg = MuonNs5Config {
            lr: 0.02,
            momentum: 0.95,
            weight_decay: 0.01,
            nesterov: true,
            ns5: prec,
        };

        m.set_bf16_accumulation(false);
        ok("muon f32", m.muon_ns5_step(&mut p1, &g, &mut m1, cfg));
        m.set_bf16_accumulation(true);
        ok("muon bf16", m.muon_ns5_step(&mut p2, &g, &mut m2, cfg));
        ok("sync muon", m.sync());
        assert_ne!(bits(&p1), bits(&p2), "muon prec={prec:?}");
    }

    // 6. Non-finite deferred fault check and tight budget check under bf16 accumulation.
    let bad = up(&m, &host(&[1.0, f32::NAN, 2.0, 3.0], &[2, 2]));
    let good = up(&m, &host(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let _ = ok("enqueue bad linear", m.linear_forward(&bad, &good));
    let err = m.sync().unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::NonFinite {
                op: "linear_forward"
            }
        ),
        "{err:?}"
    );

    let tight = m.with_budget(Budget::new(8));
    let err = tight.linear_forward(&good, &good).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");

    m.set_bf16_accumulation(false);
}

#[test]
fn pool_norm_gdn_and_vres_backward_accumulate_in_bf16() {
    let m = metal();
    let s = r(0.002);

    // 1. max_pool2d_backward: center element (1, 1) of a 3x3 input is the unique
    // maximum of all four 2x2 windows (stride 1). Window grads [1.0, s, s, s]
    // sum to 1.0 in bf16 accumulation vs >1.0 in f32 accumulation.
    let mut x_mp = [0.0f32; 9];
    x_mp[4] = 10.0;
    let x_mp = up(&m, &host(&x_mp, &[1, 1, 3, 3]));
    let dy_mp = up(&m, &host(&[1.0, s, s, s], &[1, 1, 2, 2]));
    m.set_bf16_accumulation(false);
    let dx_mp_f32 = down(&ok(
        "max_pool2d_bwd f32",
        m.max_pool2d_backward(&x_mp, &dy_mp, [2, 2], [1, 1]),
    ));
    m.set_bf16_accumulation(true);
    let dx_mp_bf16 = down(&ok(
        "max_pool2d_bwd bf16",
        m.max_pool2d_backward(&x_mp, &dy_mp, [2, 2], [1, 1]),
    ));
    assert_eq!(dx_mp_bf16[4].to_bits(), 1.0f32.to_bits());
    assert!(dx_mp_f32[4] > 1.005, "dx_mp_f32={}", dx_mp_f32[4]);

    // 2. rms_norm_backward: gw sums gy[r, c] * (x[r, c] * rstd[r]) over rows.
    // With x = 1.0 everywhere and eps = 0.0, rstd = 1.0 and x * rstd = 1.0,
    // so gw[0] sums gy[r, 0] = [1.0, s, s, s].
    let x_rms = up(&m, &host(&[1.0; 16], &[4, 4]));
    let w_rms = up(&m, &host(&[1.0; 4], &[4]));
    let mut gy_rms_v = [0.0f32; 16];
    gy_rms_v[0] = 1.0;
    gy_rms_v[4] = s;
    gy_rms_v[8] = s;
    gy_rms_v[12] = s;
    let gy_rms = up(&m, &host(&gy_rms_v, &[4, 4]));
    m.set_bf16_accumulation(false);
    let (_, gw_rms_f32) = ok(
        "rms_bwd f32",
        m.rms_norm_backward(&x_rms, &w_rms, &gy_rms, 0.0),
    );
    m.set_bf16_accumulation(true);
    let (_, gw_rms_bf16) = ok(
        "rms_bwd bf16",
        m.rms_norm_backward(&x_rms, &w_rms, &gy_rms, 0.0),
    );
    assert_eq!(down(&gw_rms_bf16)[0].to_bits(), 1.0f32.to_bits());
    assert!(down(&gw_rms_f32)[0] > 1.005);

    // 3. layer_norm_backward: dw and db sum over rows (20 rows => 2 splits of 10 rows).
    // Split 0 sums [1.0, s, ..., s] -> 1.0 in Stage 1; Split 1 has [s, 0, ..., 0] -> s,
    // and Stage 2 sums Split 0 + Split 1 = 1.0 + s -> 1.0 in bf16 vs 1.0 + 10*s in f32.
    let rows_ln = 20usize;
    let mut x_ln_v = Vec::with_capacity(rows_ln * 4);
    let mut gy_ln_v = vec![0.0f32; rows_ln * 4];
    for r_idx in 0..rows_ln {
        x_ln_v.extend_from_slice(&[1.0, -1.0, 1.0, -1.0]);
        gy_ln_v[r_idx * 4] = if r_idx == 0 {
            1.0
        } else if r_idx <= 10 {
            s
        } else {
            0.0
        };
    }
    let x_ln = up(&m, &host(&x_ln_v, &[rows_ln, 4]));
    let w_ln = up(&m, &host(&[1.0; 4], &[4]));
    let b_ln = up(&m, &host(&[0.0; 4], &[4]));
    let gy_ln = up(&m, &host(&gy_ln_v, &[rows_ln, 4]));
    m.set_bf16_accumulation(false);
    let ln_f32 = ok(
        "ln_bwd f32",
        m.layer_norm_backward(&x_ln, &w_ln, &b_ln, &gy_ln, 0.0),
    );
    m.set_bf16_accumulation(true);
    let ln_bf16 = ok(
        "ln_bwd bf16",
        m.layer_norm_backward(&x_ln, &w_ln, &b_ln, &gy_ln, 0.0),
    );
    assert_eq!(down(&ln_bf16.weight)[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(down(&ln_bf16.bias)[0].to_bits(), 1.0f32.to_bits());
    assert!(down(&ln_f32.weight)[0] > 1.018);
    assert!(down(&ln_f32.bias)[0] > 1.018);

    // 4. channel_layer_norm_backward: dw and db sum over positions (batch * spatial).
    let pos_cln = 8usize;
    let mut x_cln_v = vec![0.0f32; 2 * pos_cln];
    let mut gy_cln_v = vec![0.0f32; 2 * pos_cln];
    for p in 0..pos_cln {
        x_cln_v[p] = 1.0;
        x_cln_v[pos_cln + p] = -1.0;
        gy_cln_v[p] = if p == 0 { 1.0 } else { s };
    }
    let x_cln = up(&m, &host(&x_cln_v, &[1, 2, pos_cln]));
    let w_cln = up(&m, &host(&[1.0, 1.0], &[2]));
    let b_cln = up(&m, &host(&[0.0, 0.0], &[2]));
    let gy_cln = up(&m, &host(&gy_cln_v, &[1, 2, pos_cln]));
    m.set_bf16_accumulation(false);
    let cln_f32 = ok(
        "cln_bwd f32",
        m.channel_layer_norm_backward(&x_cln, &w_cln, &b_cln, &gy_cln, 0.0),
    );
    m.set_bf16_accumulation(true);
    let cln_bf16 = ok(
        "cln_bwd bf16",
        m.channel_layer_norm_backward(&x_cln, &w_cln, &b_cln, &gy_cln, 0.0),
    );
    assert_eq!(down(&cln_bf16.weight)[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(down(&cln_bf16.bias)[0].to_bits(), 1.0f32.to_bits());
    assert!(down(&cln_f32.weight)[0] > 1.01);
    assert!(down(&cln_f32.bias)[0] > 1.01);

    // 5. gdn_log_decay_backward: da_log and ddt_bias sum over rows.
    let a_gdn = up(&m, &host(&[30.0; 8], &[8, 1]));
    let alog_gdn = up(&m, &host(&[0.0], &[1]));
    let dt_gdn = up(&m, &host(&[0.0], &[1]));
    let mut gy_gdn_v = [s; 8];
    gy_gdn_v[0] = 1.0;
    let gy_gdn = up(&m, &host(&gy_gdn_v, &[8, 1]));
    m.set_bf16_accumulation(false);
    let gdn_f32 = ok(
        "gdn_decay_bwd f32",
        m.gdn_log_decay_backward(&a_gdn, &alog_gdn, &dt_gdn, &gy_gdn),
    );
    m.set_bf16_accumulation(true);
    let gdn_bf16 = ok(
        "gdn_decay_bwd bf16",
        m.gdn_log_decay_backward(&a_gdn, &alog_gdn, &dt_gdn, &gy_gdn),
    );
    assert_eq!(down(&gdn_bf16.dt_bias)[0].to_bits(), (-1.0f32).to_bits());
    assert!(down(&gdn_f32.dt_bias)[0] < -1.01);
    assert_ne!(bits(&gdn_bf16.a_log), bits(&gdn_f32.a_log));

    // 6. value_residual_blend_backward: d_lambda sums over elements via reduce.
    let v_vr = up(&m, &host(&[0.0; 8], &[8]));
    let v0_vr = up(&m, &host(&[1.0; 8], &[8]));
    let lam_vr = up(&m, &host(&[0.0], &[1]));
    let mut gy_vr_v = [s; 8];
    gy_vr_v[0] = 1.0;
    let gy_vr = up(&m, &host(&gy_vr_v, &[8]));
    m.set_bf16_accumulation(false);
    let vr_f32 = ok(
        "vres_bwd f32",
        m.value_residual_blend_backward(&v_vr, &v0_vr, &lam_vr, &gy_vr),
    );
    m.set_bf16_accumulation(true);
    let vr_bf16 = ok(
        "vres_bwd bf16",
        m.value_residual_blend_backward(&v_vr, &v0_vr, &lam_vr, &gy_vr),
    );
    assert_eq!(down(&vr_bf16.lambda)[0].to_bits(), 0.25f32.to_bits());
    assert!(down(&vr_f32.lambda)[0] > 0.252);

    m.set_bf16_accumulation(false);
}

#[test]
fn seq_mean_and_l2_normalize_accumulate_in_bf16() {
    let m = metal();
    let s = r(0.002);
    assert_eq!(bf16_add(1.0, s), 1.0);

    // 1. seq_mean_forward: segment sum over 8 time steps in [B=1, T=8, D=4].
    // Row 0 is 1.0, rows 1..8 are s. In bf16 accumulation, sum = 1.0, mean = 0.125.
    // In f32 accumulation, sum = 1.0 + 7*s > 1.013, mean > 0.126.
    let mut x_sm_v = vec![s; 8 * 4];
    x_sm_v[..4].fill(1.0);
    let x_sm = up(&m, &host(&x_sm_v, &[1, 8, 4]));
    let key_lens = up(&m, &host_u32(&[8], &[1]));
    m.set_bf16_accumulation(false);
    let sm_f32 = ok("seq_mean f32", m.seq_mean_forward(&x_sm, &key_lens));
    m.set_bf16_accumulation(true);
    let sm_bf16 = ok("seq_mean bf16", m.seq_mean_forward(&x_sm, &key_lens));
    assert_eq!(down(&sm_bf16)[0].to_bits(), r(1.0 / 8.0).to_bits());
    assert!(down(&sm_f32)[0] > 0.126);

    // 2. l2_normalize_forward: row sum_sq over 64 columns.
    // Col 0 is 1.0, cols 1..64 are sqrt(s) (~0.04472) whose bf16 square is s (~0.002).
    // In bf16 accumulation, sum_sq = 1.0, inv = 1.0, y[0] = 1.0.
    // In f32 accumulation, sum_sq = 1.0 + 63*s > 1.12, inv < 0.945, y[0] < 0.945.
    let cols = 64usize;
    let sq_s = r(s.sqrt());
    let sq_s_sq = r(sq_s * sq_s);
    assert!(sq_s_sq > 0.0 && bf16_add(1.0, sq_s_sq) == 1.0);
    let mut x_l2_v = vec![sq_s; cols];
    x_l2_v[0] = 1.0;
    let x_l2 = up(&m, &host(&x_l2_v, &[1, cols]));
    m.set_bf16_accumulation(false);
    let l2_f32 = ok("l2_norm f32", m.l2_normalize_forward(&x_l2, 1e-12));
    m.set_bf16_accumulation(true);
    let l2_bf16 = ok("l2_norm bf16", m.l2_normalize_forward(&x_l2, 1e-12));
    assert_eq!(down(&l2_bf16)[0].to_bits(), 1.0f32.to_bits());
    assert!(down(&l2_f32)[0] < 0.95);

    // 3. l2_normalize_backward: both sum_sq and dot(gy, x) accumulate in bf16.
    // With gy = [1.0, 0, 0, ...], dot = 1.0 in both, but sum_sq = 1.0 in bf16 vs >1.12 in f32!
    let mut gy_l2_v = vec![0.0f32; cols];
    gy_l2_v[0] = 1.0;
    let gy_l2 = up(&m, &host(&gy_l2_v, &[1, cols]));
    m.set_bf16_accumulation(false);
    let l2b_f32 = ok("l2_bwd f32", m.l2_normalize_backward(&x_l2, &gy_l2, 1e-12));
    m.set_bf16_accumulation(true);
    let l2b_bf16 = ok("l2_bwd bf16", m.l2_normalize_backward(&x_l2, &gy_l2, 1e-12));
    assert_eq!(down(&l2b_bf16)[0].to_bits(), 0.0f32.to_bits());
    assert!(down(&l2b_f32)[0] > 0.08);

    m.set_bf16_accumulation(false);
}

#[test]
fn softmax_and_cross_entropy_online_reductions_accumulate_in_bf16() {
    let m = metal();

    // 1. softmax_forward: row of length 64 where col 0 has logit 0.0 (exp = 1.0)
    // and cols 1..64 have logit -6.25 (exp(-6.25) ~ 0.00193, which rounds in bf16 to ~0.00193787
    // and is absorbed into 1.0 under bf16 addition: r(1.0 + r(exp(-6.25))) == 1.0).
    let cols = 64usize;
    let neg_logit = -6.25f32;
    let exp_neg = r(neg_logit.exp());
    assert!(exp_neg > 0.0 && bf16_add(1.0, exp_neg) == 1.0);
    let mut logits_v = vec![neg_logit; cols];
    logits_v[0] = 0.0;
    let logits = up(&m, &host(&logits_v, &[1, cols]));

    m.set_bf16_accumulation(false);
    let sm_f32 = ok("softmax f32", m.softmax_forward(&logits));
    m.set_bf16_accumulation(true);
    let sm_bf16 = ok("softmax bf16", m.softmax_forward(&logits));
    // Under bf16 accumulation, denom == 1.0 so p[0] == 1.0; under f32, denom ~ 1.12 so p[0] < 0.90.
    assert_eq!(down(&sm_bf16)[0].to_bits(), 1.0f32.to_bits());
    assert!(down(&sm_f32)[0] < 0.90);

    // 2. softmax_backward: dot = sum_j(gy_j * y_j) accumulates in bf16.
    let s = r(0.002);
    let mut y_v = vec![1.0f32; cols];
    let mut gy_v = vec![s; cols];
    y_v[0] = 1.0;
    gy_v[0] = 1.0;
    let y_sm = up(&m, &host(&y_v, &[1, cols]));
    let gy_sm = up(&m, &host(&gy_v, &[1, cols]));
    m.set_bf16_accumulation(false);
    let smb_f32 = ok("softmax_bwd f32", m.softmax_backward(&y_sm, &gy_sm));
    m.set_bf16_accumulation(true);
    let smb_bf16 = ok("softmax_bwd bf16", m.softmax_backward(&y_sm, &gy_sm));
    // At col 0: y_0 * (gy_0 - dot) = 1.0 * (1.0 - 1.0) = 0.0 in bf16, vs < -0.12 in f32.
    assert_eq!(down(&smb_bf16)[0].to_bits(), 0.0f32.to_bits());
    assert!(down(&smb_f32)[0] < -0.10);

    // 3. cross_entropy_mean_forward & cross_entropy_mean_backward: online softmax denom in bf16.
    // Target is class 0 (logit 0.0), classes 1..64 have logit -6.25.
    // In bf16 accumulation, denom = 1.0 -> log(denom) = 0.0 -> loss = 0.0, dlogits[0] = 0.0.
    // In f32 accumulation, denom ~ 1.121 -> loss = log(1.121) > 0.11, dlogits[0] = p[0] - 1 < -0.10.
    let targets = up(&m, &host_u32(&[0], &[1]));
    m.set_bf16_accumulation(false);
    let ce_f32 = ok(
        "ce f32",
        m.cross_entropy_mean_forward(&logits, &targets, None),
    );
    let ceb_f32 = ok(
        "ce_bwd f32",
        m.cross_entropy_mean_backward(&logits, &targets, None),
    );
    m.set_bf16_accumulation(true);
    let ce_bf16 = ok(
        "ce bf16",
        m.cross_entropy_mean_forward(&logits, &targets, None),
    );
    let ceb_bf16 = ok(
        "ce_bwd bf16",
        m.cross_entropy_mean_backward(&logits, &targets, None),
    );
    assert_eq!(down(&ce_bf16)[0].to_bits(), 0.0f32.to_bits());
    assert!(down(&ce_f32)[0] > 0.10);
    assert_eq!(down(&ceb_bf16)[0].to_bits(), 0.0f32.to_bits());
    assert!(down(&ceb_f32)[0] < -0.10);

    // 4. linear_cross_entropy_mean: logits formed via bf16 GEMM + online stats in bf16.
    // Hidden h = [1.0] (bt=1, d=1), weight w has row 0 = [0.0] and rows 1..64 = [-6.25].
    let h_lce = up(&m, &host(&[1.0], &[1, 1]));
    let w_lce = up(&m, &host(&logits_v, &[cols, 1]));
    let chunk = CeChunk { rows: 1, cols };
    m.set_bf16_accumulation(false);
    let lce_f32 = ok(
        "lce f32",
        m.linear_cross_entropy_mean(&h_lce, &w_lce, &targets, None, chunk, CeGrads::BOTH),
    );
    m.set_bf16_accumulation(true);
    let lce_bf16 = ok(
        "lce bf16",
        m.linear_cross_entropy_mean(&h_lce, &w_lce, &targets, None, chunk, CeGrads::BOTH),
    );
    assert_eq!(down(&lce_bf16.loss)[0].to_bits(), 0.0f32.to_bits());
    assert!(down(&lce_f32.loss)[0] > 0.10);
    assert_ne!(
        bits(lce_bf16.grad_weight.as_ref().unwrap()),
        bits(lce_f32.grad_weight.as_ref().unwrap())
    );

    m.set_bf16_accumulation(false);
}

#[test]
fn norms_forward_and_row_backward_accumulate_in_bf16() {
    let m = metal();
    let cols = 64usize;
    let s = r(0.002);
    let sq_s = r(s.sqrt());
    assert_eq!(bf16_add(1.0, r(sq_s * sq_s)), 1.0);

    // 1. rms_norm_forward: x[0] = 8.0 (sq = 64.0),
    // x[1..64] = 8.0 * sq_s (sq = 64.0 * s = 0.128, r(64.0 + 0.128) == 64.0 in bf16!)
    let v0 = 8.0f32;
    let vs = r(8.0 * sq_s);
    assert_eq!(bf16_add(64.0, r(vs * vs)), 64.0);
    let mut x_rms_v = vec![vs; cols];
    x_rms_v[0] = v0;
    let x_rms = up(&m, &host(&x_rms_v, &[1, cols]));
    let w_ones = up(&m, &host(&vec![1.0f32; cols], &[cols]));
    m.set_bf16_accumulation(false);
    let rms_f32 = ok("rms f32", m.rms_norm_forward(&x_rms, &w_ones, 0.0));
    m.set_bf16_accumulation(true);
    let rms_bf16 = ok("rms bf16", m.rms_norm_forward(&x_rms, &w_ones, 0.0));
    // In bf16: sum_sq = 64.0, mean_sq = 1.0, rstd = 1.0, y[0] = 8.0.
    // In f32: sum_sq = 64.0 + 63 * vs^2 ~ 72.0, mean_sq ~ 1.125, y[0] < 7.55.
    assert_eq!(down(&rms_bf16)[0].to_bits(), 8.0f32.to_bits());
    assert!(down(&rms_f32)[0] < 7.6);

    // 2. rms_norm_backward dx: row sum_sq and dot(gy*w, x) accumulate in bf16.
    let mut gy_rms_v = vec![0.0f32; cols];
    gy_rms_v[0] = 1.0;
    let gy_rms = up(&m, &host(&gy_rms_v, &[1, cols]));
    m.set_bf16_accumulation(false);
    let rmsb_f32 = ok(
        "rms_bwd f32",
        m.rms_norm_backward(&x_rms, &w_ones, &gy_rms, 0.0),
    );
    m.set_bf16_accumulation(true);
    let rmsb_bf16 = ok(
        "rms_bwd bf16",
        m.rms_norm_backward(&x_rms, &w_ones, &gy_rms, 0.0),
    );
    // In bf16: mean_sq = 1.0, rstd = 1.0, dot = 8.0, c = 8.0 / 64 = 0.125,
    // dx[0] = 1.0 * (1.0 - 8.0 * 0.125) = 0.0.
    assert_eq!(down(&rmsb_bf16.0)[0].to_bits(), 0.0f32.to_bits());
    assert!(down(&rmsb_f32.0)[0] > 0.08);

    // 3. layer_norm_forward & layer_norm_backward dx (row stats in bf16).
    let mut x_ln_v = vec![s; cols];
    x_ln_v[0] = 1.0;
    let x_ln = up(&m, &host(&x_ln_v, &[1, cols]));
    let b_zeros = up(&m, &host(&vec![0.0f32; cols], &[cols]));
    m.set_bf16_accumulation(false);
    let ln_f32 = ok(
        "ln f32",
        m.layer_norm_forward(&x_ln, &w_ones, &b_zeros, 1e-5),
    );
    let lnb_f32 = ok(
        "ln_bwd f32",
        m.layer_norm_backward(&x_ln, &w_ones, &b_zeros, &gy_rms, 1e-5),
    );
    m.set_bf16_accumulation(true);
    let ln_bf16 = ok(
        "ln bf16",
        m.layer_norm_forward(&x_ln, &w_ones, &b_zeros, 1e-5),
    );
    let lnb_bf16 = ok(
        "ln_bwd bf16",
        m.layer_norm_backward(&x_ln, &w_ones, &b_zeros, &gy_rms, 1e-5),
    );
    assert_ne!(bits(&ln_bf16), bits(&ln_f32));
    assert_ne!(bits(&lnb_bf16.input), bits(&lnb_f32.input));

    // 4. channel_layer_norm_forward & channel_layer_norm_backward dx (channel stats in bf16).
    let x_cln = up(&m, &host(&x_ln_v, &[1, cols, 1]));
    let gy_cln = up(&m, &host(&gy_rms_v, &[1, cols, 1]));
    m.set_bf16_accumulation(false);
    let cln_f32 = ok(
        "cln f32",
        m.channel_layer_norm_forward(&x_cln, &w_ones, &b_zeros, 1e-5),
    );
    let clnb_f32 = ok(
        "cln_bwd f32",
        m.channel_layer_norm_backward(&x_cln, &w_ones, &b_zeros, &gy_cln, 1e-5),
    );
    m.set_bf16_accumulation(true);
    let cln_bf16 = ok(
        "cln bf16",
        m.channel_layer_norm_forward(&x_cln, &w_ones, &b_zeros, 1e-5),
    );
    let clnb_bf16 = ok(
        "cln_bwd bf16",
        m.channel_layer_norm_backward(&x_cln, &w_ones, &b_zeros, &gy_cln, 1e-5),
    );
    assert_ne!(bits(&cln_bf16), bits(&cln_f32));
    assert_ne!(bits(&clnb_bf16.input), bits(&clnb_f32.input));

    // 5. group_norm_forward: group sum & variance accumulate in bf16.
    let x_gn = up(&m, &host(&x_ln_v, &[1, 1, 8, 8]));
    let w_gn = up(&m, &host(&[1.0], &[1]));
    let b_gn = up(&m, &host(&[0.0], &[1]));
    m.set_bf16_accumulation(false);
    let gn_f32 = ok("gn f32", m.group_norm_forward(&x_gn, &w_gn, &b_gn, 1, 1e-5));
    m.set_bf16_accumulation(true);
    let gn_bf16 = ok(
        "gn bf16",
        m.group_norm_forward(&x_gn, &w_gn, &b_gn, 1, 1e-5),
    );
    assert_ne!(bits(&gn_bf16), bits(&gn_f32));

    // 6. gated_rms_norm_forward & gated_rms_norm_backward: per-head sum_sq and dw sum over rows accumulate in bf16.
    // 8 rows, 1 head of dim 64: row 0..8 have x_rms_v, gy has row 0 = 1.0, rows 1..8 = s.
    let mut x_grms_v = Vec::with_capacity(8 * cols);
    let mut gy_grms_v = Vec::with_capacity(8 * cols);
    for r_idx in 0..8 {
        x_grms_v.extend_from_slice(&x_rms_v);
        let g_val = if r_idx == 0 { 1.0 } else { s };
        gy_grms_v.extend(std::iter::repeat_n(g_val, cols));
    }
    let x_grms = up(&m, &host(&x_grms_v, &[8, cols]));
    let z_grms = up(&m, &host(&vec![2.0f32; 8 * cols], &[8, cols]));
    let gy_grms = up(&m, &host(&gy_grms_v, &[8, cols]));
    m.set_bf16_accumulation(false);
    let grms_f32 = ok(
        "grms f32",
        m.gated_rms_norm_forward(&x_grms, &z_grms, &w_ones, 1e-6),
    );
    let grmsb_f32 = ok(
        "grms_bwd f32",
        m.gated_rms_norm_backward(&x_grms, &z_grms, &w_ones, &gy_grms, 1e-6),
    );
    m.set_bf16_accumulation(true);
    let grms_bf16 = ok(
        "grms bf16",
        m.gated_rms_norm_forward(&x_grms, &z_grms, &w_ones, 1e-6),
    );
    let grmsb_bf16 = ok(
        "grms_bwd bf16",
        m.gated_rms_norm_backward(&x_grms, &z_grms, &w_ones, &gy_grms, 1e-6),
    );
    assert_ne!(bits(&grms_bf16), bits(&grms_f32));
    assert_ne!(bits(&grmsb_bf16.input), bits(&grmsb_f32.input));
    assert_ne!(bits(&grmsb_bf16.weight), bits(&grmsb_f32.weight));

    m.set_bf16_accumulation(false);
}

#[test]
fn causal_conv1d_silu_bilinear_and_adds_accumulate_in_bf16() {
    let m = metal();
    let s = r(0.002);
    assert_eq!(bf16_add(1.0, s), 1.0);

    // 1. causal_conv1d_silu_forward & causal_conv1d_silu_backward: kernel tap sum and dw batch*time sum in bf16.
    // b=1, seq=8, d=1, k=4: at t=3, tap=0 reads x[0]=1.0, taps 1..3 read s!
    let mut x_c1d_v = vec![s; 8];
    x_c1d_v[0] = 1.0;
    let x_c1d = up(&m, &host(&x_c1d_v, &[1, 8, 1]));
    let w_c1d = up(&m, &host(&[1.0f32; 4], &[1, 4]));
    let mut gy_c1d_v = vec![s; 8];
    gy_c1d_v[0] = 1.0;
    let gy_c1d = up(&m, &host(&gy_c1d_v, &[1, 8, 1]));

    m.set_bf16_accumulation(false);
    let c1d_f32 = ok("conv1d f32", m.causal_conv1d_silu_forward(&x_c1d, &w_c1d));
    let c1db_f32 = ok(
        "conv1d_bwd f32",
        m.causal_conv1d_silu_backward(&x_c1d, &w_c1d, &gy_c1d),
    );
    m.set_bf16_accumulation(true);
    let c1d_bf16 = ok("conv1d bf16", m.causal_conv1d_silu_forward(&x_c1d, &w_c1d));
    let c1db_bf16 = ok(
        "conv1d_bwd bf16",
        m.causal_conv1d_silu_backward(&x_c1d, &w_c1d, &gy_c1d),
    );
    // At t=3: pre-activation u = 1.0 in bf16 vs 1.0 + 3*s in f32.
    let silu_1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
    assert!((down(&c1d_bf16)[3] - silu_1).abs() < 1e-6);
    assert!(down(&c1d_f32)[3] > silu_1 + 0.003);
    assert_ne!(bits(&c1db_bf16.0), bits(&c1db_f32.0));
    assert_ne!(bits(&c1db_bf16.1), bits(&c1db_f32.1));

    // 2. interpolate_bilinear_forward: 2x2 -> 1x1 at fractional center (fy=0.5, fx=0.5, wy=0.5, wx=0.5).
    // Weights are 0.25 each; with [4.0, 4*s, 0.0, 0.0], top = 0.5*4.0 + 0.5*(4*s) = 2.0 in bf16 vs 2.0+2*s in f32,
    // so bf16 accumulation gives 1.0, while f32 accumulation gives 1.0 + s > 1.0015.
    let s4 = r(4.0 * s);
    let x_bil = up(&m, &host(&[4.0, s4, 0.0, 0.0], &[1, 1, 2, 2]));
    m.set_bf16_accumulation(false);
    let bil_f32 = ok(
        "bilinear f32",
        m.interpolate_bilinear_forward(&x_bil, [1, 1], false),
    );
    m.set_bf16_accumulation(true);
    let bil_bf16 = ok(
        "bilinear bf16",
        m.interpolate_bilinear_forward(&x_bil, [1, 1], false),
    );
    assert_eq!(down(&bil_bf16)[0].to_bits(), 1.0f32.to_bits());
    assert!(down(&bil_f32)[0] > 1.0015);

    // 3. broadcast_add_forward and residual_add_forward: a = 1.0, b = s -> 1.0 in bf16 vs 1.0 + s in f32.
    let a_add = up(&m, &host(&[1.0, 1.0], &[1, 2]));
    let b_bcast = up(&m, &host(&[s, s], &[2]));
    let b_resid = up(&m, &host(&[s, s], &[1, 2]));
    m.set_bf16_accumulation(false);
    let bc_f32 = ok("bcast_add f32", m.broadcast_add_forward(&a_add, &b_bcast));
    let ra_f32 = ok("resid_add f32", m.residual_add_forward(&a_add, &b_resid));
    m.set_bf16_accumulation(true);
    let bc_bf16 = ok("bcast_add bf16", m.broadcast_add_forward(&a_add, &b_bcast));
    let ra_bf16 = ok("resid_add bf16", m.residual_add_forward(&a_add, &b_resid));
    assert_eq!(down(&bc_bf16)[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(down(&ra_bf16)[0].to_bits(), 1.0f32.to_bits());
    assert!(down(&bc_f32)[0] > 1.001);
    assert!(down(&ra_f32)[0] > 1.001);

    m.set_bf16_accumulation(false);
}

#[test]
fn edge_cases_multi_chunk_and_non_finite_under_bf16_accumulation() {
    let m = metal();
    let s = r(0.002);

    // 1. seq_mean_forward with key_lens == 0 produces exact zeros under bf16 accumulation.
    let x_sm = up(&m, &host(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]));
    let zero_len = up(&m, &host_u32(&[0], &[1]));
    m.set_bf16_accumulation(true);
    let sm_zero = ok("seq_mean zero len", m.seq_mean_forward(&x_sm, &zero_len));
    assert_eq!(down(&sm_zero), vec![0.0, 0.0]);

    // 2. Multi-tile linear_cross_entropy_mean (vocab = 64, chunk.cols = 16 -> 4 tiles).
    let neg_logit = -6.25f32;
    let mut logits_v = vec![neg_logit; 64];
    logits_v[0] = 0.0;
    let h_lce = up(&m, &host(&[1.0], &[1, 1]));
    let w_lce = up(&m, &host(&logits_v, &[64, 1]));
    let targets = up(&m, &host_u32(&[0], &[1]));
    let tiled_chunk = CeChunk { rows: 1, cols: 16 };
    m.set_bf16_accumulation(false);
    let lce_f32 = ok(
        "lce tiled f32",
        m.linear_cross_entropy_mean(&h_lce, &w_lce, &targets, None, tiled_chunk, CeGrads::BOTH),
    );
    m.set_bf16_accumulation(true);
    let lce_bf16 = ok(
        "lce tiled bf16",
        m.linear_cross_entropy_mean(&h_lce, &w_lce, &targets, None, tiled_chunk, CeGrads::BOTH),
    );
    assert_eq!(down(&lce_bf16.loss)[0].to_bits(), 0.0f32.to_bits());
    assert!(down(&lce_f32.loss)[0] > 0.10);

    // 3. Multi-chunk causal_conv1d_silu_backward (rows = 300 > CONV1D_BWD_ROWS = 256,
    // channels = 64, width = 4 -> scratch is 2 * 256 = 512 floats = 2048 bytes).
    // Row 0 has gy = 1.0 and rows 64..256 have gy = s: because CONV1D_BWD_ROWS = 256,
    // rows 64..256 belong to chunk 0 (0..256) and are absorbed into row 0's contribution
    // under bf16 accumulation, whereas a 64-row chunking bug would accumulate rows 64..256
    // into separate chunks starting from 0.0 and corrupt dw (and overrun scratch by 4x).
    let rows = 300usize;
    let ch = 64usize;
    let kw = 4usize;
    let x_c1d_v = vec![1.0f32; rows * ch];
    let mut gy_c1d_v = vec![0.0f32; rows * ch];
    for c in 0..ch {
        gy_c1d_v[c] = 1.0;
        for r_idx in 64..256 {
            gy_c1d_v[r_idx * ch + c] = s;
        }
    }
    let mut w_c1d_v = vec![0.0f32; ch * kw];
    for c in 0..ch {
        w_c1d_v[c * kw + (kw - 1)] = 1.0;
    }
    let x_c1d = up(&m, &host(&x_c1d_v, &[rows, 1, ch]));
    let w_c1d = up(&m, &host(&w_c1d_v, &[ch, kw]));
    let gy_c1d = up(&m, &host(&gy_c1d_v, &[rows, 1, ch]));
    // Reference with only row 0 active (rows 64..256 = 0):
    let mut gy_row0_v = vec![0.0f32; rows * ch];
    gy_row0_v[..ch].fill(1.0);
    let gy_row0 = up(&m, &host(&gy_row0_v, &[rows, 1, ch]));
    m.set_bf16_accumulation(false);
    let (_, dw_c1d_f32) = ok(
        "conv1d_bwd 300 f32",
        m.causal_conv1d_silu_backward(&x_c1d, &w_c1d, &gy_c1d),
    );
    m.set_bf16_accumulation(true);
    let (_, dw_c1d_bf16) = ok(
        "conv1d_bwd 300 bf16",
        m.causal_conv1d_silu_backward(&x_c1d, &w_c1d, &gy_c1d),
    );
    let (_, dw_row0_bf16) = ok(
        "conv1d_bwd row0 bf16",
        m.causal_conv1d_silu_backward(&x_c1d, &w_c1d, &gy_row0),
    );
    // In bf16 accumulation, rows 64..256 (each ~ s * dsilu(1.0) * 1.0) are in chunk 0 (0..256)
    // and are completely absorbed into row 0's dsilu(1.0) = 0.8807971, so dw_c1d_bf16 == dw_row0_bf16
    // on tap k = kw - 1 for every channel.
    let dw_bf16_host = down(&dw_c1d_bf16);
    let dw_row0_host = down(&dw_row0_bf16);
    let dw_f32_host = down(&dw_c1d_f32);
    for c in 0..ch {
        assert_eq!(
            dw_bf16_host[c * kw + (kw - 1)].to_bits(),
            dw_row0_host[c * kw + (kw - 1)].to_bits(),
            "channel {c} tap {}: rows 64..256 must be absorbed in chunk 0 (0..256)",
            kw - 1
        );
        assert!(dw_f32_host[c * kw + (kw - 1)] > dw_bf16_host[c * kw + (kw - 1)] + 0.1);
    }

    // Multi-chunk gated_rms_norm_backward (rows = 150 > GATED_RMS_BWD_ROWS = 64).
    let g_rows = 150usize;
    let g_dim = 16usize;
    let x_gr = up(&m, &host(&vec![1.0f32; g_rows * g_dim], &[g_rows, g_dim]));
    let z_gr = up(&m, &host(&vec![1.0f32; g_rows * g_dim], &[g_rows, g_dim]));
    let w_gr = up(&m, &host(&vec![1.0f32; g_dim], &[g_dim]));
    let mut gy_gr_v = vec![0.0f32; g_rows * g_dim];
    for c in 0..g_dim {
        gy_gr_v[c] = 1.0;
        for r_idx in 1..64 {
            gy_gr_v[r_idx * g_dim + c] = s;
        }
    }
    let gy_gr = up(&m, &host(&gy_gr_v, &[g_rows, g_dim]));
    m.set_bf16_accumulation(false);
    let gr_f32 = ok(
        "gated_rms_bwd 150 f32",
        m.gated_rms_norm_backward(&x_gr, &z_gr, &w_gr, &gy_gr, 1e-6),
    );
    m.set_bf16_accumulation(true);
    let gr_bf16 = ok(
        "gated_rms_bwd 150 bf16",
        m.gated_rms_norm_backward(&x_gr, &z_gr, &w_gr, &gy_gr, 1e-6),
    );
    assert!(down(&gr_f32.weight)[0] > down(&gr_bf16.weight)[0] + 0.05);

    // 4. NonFinite deferred fault detection still triggers on bf16 accumulation kernels.
    let bad = up(&m, &host(&[f32::NAN, 1.0, 1.0, 1.0], &[1, 4]));
    let w4 = up(&m, &host(&[1.0; 4], &[4]));
    let b4 = up(&m, &host(&[0.0; 4], &[4]));
    let _ = ok(
        "ln_fwd nonfinite record",
        m.layer_norm_forward(&bad, &w4, &b4, 1e-5),
    );
    assert!(matches!(m.sync(), Err(OjasError::NonFinite { .. })));

    let _ = ok(
        "l2_fwd nonfinite record",
        m.l2_normalize_forward(&bad, 1e-12),
    );
    assert!(matches!(m.sync(), Err(OjasError::NonFinite { .. })));

    m.set_bf16_accumulation(false);
}
