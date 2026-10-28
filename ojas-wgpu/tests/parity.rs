//! `WgpuBackend` against `CpuBackend`, op by op, on the shapes that break
//! tiled kernels: 1x1, primes, tile edges 63/64/65 and 127/128/129, and
//! attention lengths 1, 17 and 257.
//!
//! The backend reports [`Numerics::Fast`]. Every comparison uses
//! `|gpu - cpu| <= TOL * max(1, max |cpu|)` over the whole output tensor.

mod common;

use std::sync::Arc;

use common::*;
use ojas_core::{
    AdamWConfig, Backend, BackendId, Budget, DType, DeviceBuffer, MuonNs5Config, Numerics,
    OjasError, Tensor, RMS_NORM_EPS,
};
use ojas_wgpu::{WgpuBackend, WgpuContext};

#[test]
fn numerics_is_fast() {
    assert_eq!(gpu().numerics(), Numerics::Fast);
}

const GEMM_SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (73, 11, 13),
    (127, 67, 131),
    (63, 64, 65),
    (64, 65, 63),
    (65, 63, 64),
    (127, 128, 129),
    (128, 129, 127),
    (129, 127, 128),
    (64, 64, 64),
    (128, 128, 128),
];

#[test]
fn linear_forward_and_backward_match_cpu() {
    let c = cpu();
    for (i, &(m, n, k)) in GEMM_SHAPES.iter().enumerate() {
        let s = i as u64 * 3;
        let x = host(s, &[m, k]);
        let w = host(s + 1, &[n, k]);
        let gy = host(s + 2, &[m, n]);
        let tag = format!("linear m{m} n{n} k{k}");
        let (gx, gw) = (up(&x), up(&w));
        close(
            &format!("{tag} y"),
            &gpu().linear_forward(&gx, &gw).unwrap(),
            &c.linear_forward(&x, &w).unwrap(),
        );
        let (dx, dw) = gpu().linear_backward(&gx, &gw, &up(&gy)).unwrap();
        let (cx, cw) = c.linear_backward(&x, &w, &gy).unwrap();
        close(&format!("{tag} dx"), &dx, &cx);
        close(&format!("{tag} dw"), &dw, &cw);
    }
    // Rank-3 input flattens its prefix.
    let x = host(90, &[3, 17, 13]);
    let w = host(91, &[11, 13]);
    let gy = host(92, &[3, 17, 11]);
    close(
        "linear rank3 y",
        &gpu().linear_forward(&up(&x), &up(&w)).unwrap(),
        &c.linear_forward(&x, &w).unwrap(),
    );
    let (dx, dw) = gpu().linear_backward(&up(&x), &up(&w), &up(&gy)).unwrap();
    let (cx, cw) = c.linear_backward(&x, &w, &gy).unwrap();
    close("linear rank3 dx", &dx, &cx);
    close("linear rank3 dw", &dw, &cw);
}

/// The shared harness in ojas-kernels drives this backend through its own
/// upload and download, so it compares the device kernel and not a host
/// fallback (before, it passed host tensors and could only be refused).
#[test]
fn shared_parity_harness_runs_the_device_kernel() {
    let c = cpu();
    let g = own();
    for (i, &(rows, kin, nout)) in GEMM_SHAPES.iter().enumerate() {
        let before = readbacks(&g).0;
        // Inputs lie in [-0.5, 0.5), so |y| <= kin / 4.
        let tol = TOL * (kin as f64 / 4.0).max(1.0);
        ojas_kernels::linear_close(&c, &g, rows, kin, nout, 1900 + i as u64, tol)
            .unwrap_or_else(|e| panic!("{rows}x{kin}x{nout}: {e}"));
        assert_eq!(
            readbacks(&g).0,
            before + 1,
            "the harness must read the device result exactly once"
        );
    }
}

#[test]
fn rms_norm_and_qk_norm_match_cpu() {
    let c = cpu();
    for (i, &(rows, dim)) in [
        (1, 1),
        (73, 13),
        (1, 64),
        (65, 63),
        (127, 128),
        (129, 129),
        (2049, 3),
    ]
    .iter()
    .enumerate()
    {
        let s = 100 + i as u64 * 3;
        let x = host(s, &[rows, dim]);
        let w = host(s + 1, &[dim]);
        let gy = host(s + 2, &[rows, dim]);
        let tag = format!("rms {rows}x{dim}");
        close(
            &format!("{tag} y"),
            &gpu()
                .rms_norm_forward(&up(&x), &up(&w), RMS_NORM_EPS)
                .unwrap(),
            &c.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap(),
        );
        let (dx, dw) = gpu()
            .rms_norm_backward(&up(&x), &up(&w), &up(&gy), RMS_NORM_EPS)
            .unwrap();
        let (cx, cw) = c.rms_norm_backward(&x, &w, &gy, RMS_NORM_EPS).unwrap();
        close(&format!("{tag} dx"), &dx, &cx);
        close(&format!("{tag} dw"), &dw, &cw);
    }
    for &t in &[1usize, 17, 257] {
        let (q, k) = (host(200, &[2, t, 3, 64]), host(201, &[2, t, 3, 64]));
        let (qw, kw) = (host(202, &[64]), host(203, &[64]));
        let (gq, gk) = (host(204, &[2, t, 3, 64]), host(205, &[2, t, 3, 64]));
        let (yq, yk) = gpu()
            .rms_qk_norm_forward(&up(&q), &up(&k), &up(&qw), &up(&kw), RMS_NORM_EPS)
            .unwrap();
        let (cq, ck) = c
            .rms_qk_norm_forward(&q, &k, &qw, &kw, RMS_NORM_EPS)
            .unwrap();
        close(&format!("qk T{t} q"), &yq, &cq);
        close(&format!("qk T{t} k"), &yk, &ck);
        let got = gpu()
            .rms_qk_norm_backward(
                &up(&q),
                &up(&k),
                &up(&qw),
                &up(&kw),
                &up(&gq),
                &up(&gk),
                RMS_NORM_EPS,
            )
            .unwrap();
        let want = c
            .rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &gk, RMS_NORM_EPS)
            .unwrap();
        close(&format!("qk T{t} dq"), &got.0, &want.0);
        close(&format!("qk T{t} dk"), &got.1, &want.1);
        close(&format!("qk T{t} dqw"), &got.2, &want.2);
        close(&format!("qk T{t} dkw"), &got.3, &want.3);
    }
}

fn rope_tables(t: usize, d: usize) -> (Tensor, Tensor) {
    let half = d / 2;
    let mut cos = Vec::with_capacity(t * d);
    let mut sin = Vec::with_capacity(t * d);
    for pos in 0..t {
        for j in 0..d {
            let f = 10000f64.powf(-((j % half) as f64) / half as f64);
            let a = pos as f64 * f;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    (
        Tensor::from_f32(&cos, &[t, d], host_budget()).unwrap(),
        Tensor::from_f32(&sin, &[t, d], host_budget()).unwrap(),
    )
}

#[test]
fn rope_matches_cpu() {
    let c = cpu();
    for &t in &[1usize, 17, 257] {
        for &d in &[2usize, 64, 128] {
            let x = host(300, &[2, t, 3, d]);
            let gy = host(301, &[2, t, 3, d]);
            let (cos, sin) = rope_tables(t, d);
            let tag = format!("rope T{t} D{d}");
            close(
                &format!("{tag} y"),
                &gpu()
                    .rope_half_split_forward(&up(&x), &up(&cos), &up(&sin))
                    .unwrap(),
                &c.rope_half_split_forward(&x, &cos, &sin).unwrap(),
            );
            close(
                &format!("{tag} dx"),
                &gpu()
                    .rope_half_split_backward(&up(&gy), &up(&cos), &up(&sin))
                    .unwrap(),
                &c.rope_half_split_backward(&gy, &cos, &sin).unwrap(),
            );
        }
    }
    // Tables with the same shape as x.
    let x = host(310, &[5, 8]);
    let (cos, sin) = (host(311, &[5, 8]), host(312, &[5, 8]));
    close(
        "rope same-shape y",
        &gpu()
            .rope_half_split_forward(&up(&x), &up(&cos), &up(&sin))
            .unwrap(),
        &c.rope_half_split_forward(&x, &cos, &sin).unwrap(),
    );
    close(
        "rope same-shape dx",
        &gpu()
            .rope_half_split_backward(&up(&x), &up(&cos), &up(&sin))
            .unwrap(),
        &c.rope_half_split_backward(&x, &cos, &sin).unwrap(),
    );
}

#[test]
fn causal_attention_matches_cpu() {
    let c = cpu();
    let cases: &[(usize, usize, usize, usize)] = &[
        (1, 1, 1, 1),
        (2, 3, 1, 64),
        (2, 3, 17, 64),
        (2, 3, 257, 64),
        (1, 2, 17, 13),
        (1, 2, 65, 128),
        (1, 1, 129, 32),
        (1, 1, 257, 16),
    ];
    for (i, &(b, h, t, d)) in cases.iter().enumerate() {
        let s = 400 + i as u64 * 4;
        let shape = [b, h, t, d];
        let (q, k, v, gy) = (
            host(s, &shape),
            host(s + 1, &shape),
            host(s + 2, &shape),
            host(s + 3, &shape),
        );
        let tag = format!("sdpa B{b} H{h} T{t} D{d}");
        let (gq, gk, gv) = (up(&q), up(&k), up(&v));
        close(
            &format!("{tag} y"),
            &gpu()
                .causal_sdpa_forward(&gq, &gk, &gv, None)
                .map(|(y, _)| y)
                .unwrap(),
            &c.causal_sdpa_forward(&q, &k, &v, None)
                .map(|(y, _)| y)
                .unwrap(),
        );
        let got = gpu()
            .causal_sdpa_backward_recompute(&gq, &gk, &gv, &up(&gy), None)
            .unwrap();
        let want = c
            .causal_sdpa_backward_recompute(&q, &k, &v, &gy, None)
            .unwrap();
        close(&format!("{tag} dq"), &got.0, &want.0);
        close(&format!("{tag} dk"), &got.1, &want.1);
        close(&format!("{tag} dv"), &got.2, &want.2);
    }
}

#[test]
fn attention_head_dim_past_the_kernel_limit_is_refused() {
    let q = up(&host(1, &[1, 1, 4, 257]));
    match gpu().causal_sdpa_forward(&q, &q, &q, None).map(|(y, _)| y) {
        Err(OjasError::UnsupportedHeadDim {
            head_dim: 257,
            limit: 256,
        }) => {}
        other => panic!("expected UnsupportedHeadDim, got {other:?}"),
    }
}

#[test]
fn grouped_query_causal_attention_matches_cpu() {
    let c = cpu();
    let cases: &[(usize, usize, usize, usize, usize)] = &[
        (1, 4, 2, 8, 32),
        (2, 6, 3, 5, 13),
        (1, 2, 1, 17, 192),
        (1, 4, 1, 9, 256),
        (1, 6, 2, 7, 129),
    ];
    for (i, &(b, h, hkv, t, d)) in cases.iter().enumerate() {
        let s = 700 + i as u64 * 4;
        let qs = [b, h, t, d];
        let ks = [b, hkv, t, d];
        let (q, k, v, gy) = (
            host(s, &qs),
            host(s + 1, &ks),
            host(s + 2, &ks),
            host(s + 3, &qs),
        );
        let tag = format!("gqa B{b} H{h}/{hkv} T{t} D{d}");
        let (gq, gk, gv) = (up(&q), up(&k), up(&v));
        close(
            &format!("{tag} y"),
            &gpu()
                .causal_sdpa_forward(&gq, &gk, &gv, None)
                .map(|(y, _)| y)
                .unwrap(),
            &c.causal_sdpa_forward(&q, &k, &v, None)
                .map(|(y, _)| y)
                .unwrap(),
        );
        let got = gpu()
            .causal_sdpa_backward_recompute(&gq, &gk, &gv, &up(&gy), None)
            .unwrap();
        let want = c
            .causal_sdpa_backward_recompute(&q, &k, &v, &gy, None)
            .unwrap();
        close(&format!("{tag} dq"), &got.0, &want.0);
        close(&format!("{tag} dk"), &got.1, &want.1);
        close(&format!("{tag} dv"), &got.2, &want.2);
    }
}

#[test]
fn future_keys_do_not_leak_into_earlier_rows() {
    let (b, h, t, d) = (1, 2, 17, 64);
    let shape = [b, h, t, d];
    let q = host(500, &shape);
    let k = host(501, &shape);
    let v = host(502, &shape);
    let cut = 9;
    let mut k2 = k.to_f32_vec().unwrap();
    let mut v2 = v.to_f32_vec().unwrap();
    for head in 0..h {
        for pos in cut + 1..t {
            for j in 0..d {
                let idx = (head * t + pos) * d + j;
                k2[idx] = 1.0e3 * (j as f32 + 1.0);
                v2[idx] = -7.5e2;
            }
        }
    }
    let k2 = Tensor::from_f32(&k2, &shape, host_budget()).unwrap();
    let v2 = Tensor::from_f32(&v2, &shape, host_budget()).unwrap();
    let a = down(
        &gpu()
            .causal_sdpa_forward(&up(&q), &up(&k), &up(&v), None)
            .map(|(y, _)| y)
            .unwrap(),
    );
    let z = down(
        &gpu()
            .causal_sdpa_forward(&up(&q), &up(&k2), &up(&v2), None)
            .map(|(y, _)| y)
            .unwrap(),
    );
    for head in 0..h {
        for pos in 0..=cut {
            for j in 0..d {
                let idx = (head * t + pos) * d + j;
                assert_eq!(
                    a[idx].to_bits(),
                    z[idx].to_bits(),
                    "head {head} row {pos} saw a future key"
                );
            }
        }
    }
    // dQ of an earlier row must not see future keys either.
    let gy = host(503, &shape);
    let ga = down(
        &gpu()
            .causal_sdpa_backward_recompute(&up(&q), &up(&k), &up(&v), &up(&gy), None)
            .unwrap()
            .0,
    );
    let gz = down(
        &gpu()
            .causal_sdpa_backward_recompute(&up(&q), &up(&k2), &up(&v2), &up(&gy), None)
            .unwrap()
            .0,
    );
    for head in 0..h {
        for pos in 0..=cut {
            for j in 0..d {
                let idx = (head * t + pos) * d + j;
                assert_eq!(
                    ga[idx].to_bits(),
                    gz[idx].to_bits(),
                    "dq head {head} row {pos} saw a future key"
                );
            }
        }
    }
}

#[test]
fn gate_and_value_residual_match_cpu() {
    let c = cpu();
    for &(bt, din, heads, hd) in &[
        (1usize, 1usize, 1usize, 1usize),
        (17, 13, 3, 5),
        (257, 64, 8, 64),
        (65, 129, 2, 7),
    ] {
        let x = host(600, &[bt, din]);
        let w = host(601, &[heads, din]);
        let bias = host(602, &[heads]);
        let attn = host(603, &[bt, heads, hd]);
        let gy = host(604, &[bt, heads, hd]);
        let tag = format!("gate {bt}x{din} h{heads} d{hd}");
        close(
            &format!("{tag} y"),
            &gpu()
                .per_head_sigmoid_gate_forward(&up(&x), &up(&w), &up(&bias), &up(&attn))
                .unwrap(),
            &c.per_head_sigmoid_gate_forward(&x, &w, &bias, &attn)
                .unwrap(),
        );
        let got = gpu()
            .per_head_sigmoid_gate_backward(&up(&x), &up(&w), &up(&bias), &up(&attn), &up(&gy))
            .unwrap();
        let want = c
            .per_head_sigmoid_gate_backward(&x, &w, &bias, &attn, &gy)
            .unwrap();
        close(&format!("{tag} dx"), &got.input, &want.input);
        close(&format!("{tag} dw"), &got.weight, &want.weight);
        close(&format!("{tag} db"), &got.bias, &want.bias);
        close(&format!("{tag} dattn"), &got.attn_out, &want.attn_out);
    }
    for &n in &[1usize, 73 * 11, 65_537] {
        let v = host(610, &[n]);
        let v0 = host(611, &[n]);
        let lam = Tensor::from_f32(&[0.375], &[1], host_budget()).unwrap();
        let gy = host(612, &[n]);
        close(
            &format!("vr {n} y"),
            &gpu()
                .value_residual_blend_forward(&up(&v), &up(&v0), &up(&lam))
                .unwrap(),
            &c.value_residual_blend_forward(&v, &v0, &lam).unwrap(),
        );
        let got = gpu()
            .value_residual_blend_backward(&up(&v), &up(&v0), &up(&lam), &up(&gy))
            .unwrap();
        let want = c.value_residual_blend_backward(&v, &v0, &lam, &gy).unwrap();
        close(&format!("vr {n} dv"), &got.value, &want.value);
        close(&format!("vr {n} dv0"), &got.value0, &want.value0);
        close(&format!("vr {n} dlam"), &got.lambda, &want.lambda);
    }
}

#[test]
fn pointwise_ops_match_cpu() {
    let c = cpu();
    for &n in &[1usize, 63, 64, 65, 73 * 11, 65_537, 1 << 20] {
        let a = host(700, &[n]);
        let b = host(701, &[n]);
        let gy = host(702, &[n]);
        let (ga, gb, ggy) = (up(&a), up(&b), up(&gy));
        close(
            &format!("silu {n}"),
            &gpu().silu_forward(&ga).unwrap(),
            &c.silu_forward(&a).unwrap(),
        );
        close(
            &format!("silu' {n}"),
            &gpu().silu_backward(&ga, &ggy).unwrap(),
            &c.silu_backward(&a, &gy).unwrap(),
        );
        close(
            &format!("mul {n}"),
            &gpu().mul_forward(&ga, &gb).unwrap(),
            &c.mul_forward(&a, &b).unwrap(),
        );
        let (da, db) = gpu().mul_backward(&ga, &gb, &ggy).unwrap();
        let (ca, cb) = c.mul_backward(&a, &b, &gy).unwrap();
        close(&format!("mul' a {n}"), &da, &ca);
        close(&format!("mul' b {n}"), &db, &cb);
        close(
            &format!("add {n}"),
            &gpu().residual_add_forward(&ga, &gb).unwrap(),
            &c.residual_add_forward(&a, &b).unwrap(),
        );
        let (dx, dy) = gpu().residual_add_backward(&ga, &gb, &ggy).unwrap();
        let (cx, cy) = c.residual_add_backward(&a, &b, &gy).unwrap();
        close(&format!("add' x {n}"), &dx, &cx);
        close(&format!("add' y {n}"), &dy, &cy);
    }
}

#[test]
fn embedding_matches_cpu_and_is_deterministic() {
    let c = cpu();
    for &(vocab, dim, n) in &[
        (1usize, 1usize, 1usize),
        (73, 13, 257),
        (131, 64, 17),
        (7, 129, 1000),
    ] {
        let table = host(800, &[vocab, dim]);
        let ids: Vec<u32> = (0..n).map(|i| ((i * 31 + i / 3) % vocab) as u32).collect();
        let ids = host_u32(&ids, &[n]);
        let gy = host(801, &[n, dim]);
        let tag = format!("embed V{vocab} D{dim} N{n}");
        let (gt, gi) = (up(&table), up(&ids));
        close(
            &format!("{tag} y"),
            &gpu().embedding_forward(&gt, &gi).unwrap(),
            &c.embedding_forward(&table, &ids).unwrap(),
        );
        let d1 = gpu().embedding_backward(&gt, &gi, &up(&gy)).unwrap();
        close(
            &format!("{tag} dtable"),
            &d1,
            &c.embedding_backward(&table, &ids, &gy).unwrap(),
        );
        let d2 = gpu().embedding_backward(&gt, &gi, &up(&gy)).unwrap();
        let (a, b) = (down(&d1), down(&d2));
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{tag}: backward is not deterministic"
        );
    }
    let table = up(&host(802, &[5, 3]));
    let bad = up(&host_u32(&[0, 5], &[2]));
    assert!(
        gpu().embedding_forward(&table, &bad).is_err(),
        "out-of-range id accepted"
    );
}

#[test]
fn cross_entropy_matches_cpu() {
    let c = cpu();
    for &(rows, vocab) in &[(1usize, 1usize), (17, 11), (257, 131), (64, 1025)] {
        let logits = host(900, &[rows, vocab]);
        let targets: Vec<u32> = (0..rows).map(|i| ((i * 7) % vocab) as u32).collect();
        let targets = host_u32(&targets, &[rows]);
        for ignore in [None, Some(0u32)] {
            let tag = format!("ce {rows}x{vocab} ignore {ignore:?}");
            let (gl, gt) = (up(&logits), up(&targets));
            let want = c.cross_entropy_mean_forward(&logits, &targets, ignore);
            let got = gpu().cross_entropy_mean_forward(&gl, &gt, ignore);
            match (got, want) {
                (Ok(g), Ok(w)) => {
                    close(&format!("{tag} loss"), &g, &w);
                    close(
                        &format!("{tag} dlogits"),
                        &gpu().cross_entropy_mean_backward(&gl, &gt, ignore).unwrap(),
                        &c.cross_entropy_mean_backward(&logits, &targets, ignore)
                            .unwrap(),
                    );
                }
                (Err(_), Err(_)) => {}
                (g, w) => panic!("{tag}: gpu {g:?} vs cpu {w:?}"),
            }
        }
    }
    // A logit that cannot reach the loss still poisons it, as on the CPU.
    let mut l = data(901, 12);
    l[5] = f32::NAN;
    let logits = Tensor::from_f32(&l, &[3, 4], host_budget()).unwrap();
    let targets = host_u32(&[0, 2, 3], &[3]);
    assert!(c
        .cross_entropy_mean_forward(&logits, &targets, Some(2))
        .is_err());
    let own = fresh();
    let r = own
        .cross_entropy_mean_forward(
            &own.upload(&logits).unwrap(),
            &own.upload(&targets).unwrap(),
            Some(2),
        )
        .and_then(|t| own.download(&t));
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "got {r:?}");
}

#[test]
fn clip_and_adam_refuse_shared_or_host_targets_without_writing() {
    let g = gpu();
    let a = up(&host(1100, &[64]));
    let alias = a.clone();
    let mut grads = vec![a];
    let r = g.clip_grad_norm(&mut grads, 1.0e-3);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    close_vec(
        "aliased grad untouched",
        &down(&alias),
        &host(1100, &[64]).to_f32_vec().unwrap(),
    );

    let mut hostp = host(1101, &[8]);
    let grad = up(&host(1102, &[8]));
    let mut m = up(&Tensor::zeros(&[8], DType::F32, host_budget()).unwrap());
    let mut v = up(&Tensor::zeros(&[8], DType::F32, host_budget()).unwrap());
    assert!(matches!(
        g.adamw_step(
            &mut hostp,
            &grad,
            &mut m,
            &mut v,
            1,
            AdamWConfig::nanolab(1e-2, 0.1)
        ),
        Err(OjasError::Placement { .. })
    ));
    assert!(down(&m).iter().all(|x| *x == 0.0));
}

#[test]
fn adamw_matches_cpu_over_steps() {
    let c = cpu();
    for &(n, wd) in &[(1usize, 0.0f64), (73 * 11, 0.1), (65_537, 0.01)] {
        let cfg = AdamWConfig::nanolab(3e-3, wd);
        let mut hp = host(1200, &[n]);
        let mut hm = Tensor::zeros(&[n], DType::F32, host_budget()).unwrap();
        let mut hv = Tensor::zeros(&[n], DType::F32, host_budget()).unwrap();
        let (mut dp, mut dm, mut dv) = (up(&hp), up(&hm), up(&hv));
        for step in 1..=6u64 {
            let grad = host(1300 + step, &[n]);
            gpu()
                .adamw_step(&mut dp, &up(&grad), &mut dm, &mut dv, step, cfg)
                .unwrap();
            c.adamw_step(&mut hp, &grad, &mut hm, &mut hv, step, cfg)
                .unwrap();
        }
        close(&format!("adam {n} p"), &dp, &hp);
        close(&format!("adam {n} m"), &dm, &hm);
        close(&format!("adam {n} v"), &dv, &hv);
    }
}

#[test]
fn non_finite_values_fail_closed() {
    let g = &fresh();
    let up = |t: &Tensor| g.upload(t).unwrap();
    let down = |t: &Tensor| g.download(t).unwrap().to_f32_vec().unwrap();
    g.sync().unwrap();
    // Forward op: deferred to the next synchronization, named after the op.
    let mut x = data(1400, 12);
    x[3] = f32::NAN;
    let x = up(&Tensor::from_f32(&x, &[3, 4], host_budget()).unwrap());
    let w = up(&host(1401, &[2, 4]));
    let y = g.linear_forward(&x, &w).unwrap();
    match g.download(&y) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "linear_forward"),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    g.sync().unwrap();
    let inf = up(&Tensor::from_f32(&[f32::INFINITY, 1.0], &[2], host_budget()).unwrap());
    let _ = g.silu_forward(&inf).unwrap();
    assert!(matches!(g.sync(), Err(OjasError::NonFinite { .. })));
    g.sync().unwrap();

    // Clip with a NaN gradient refuses and leaves every gradient as it was.
    let clean = host(1402, &[16]);
    let mut grads = vec![up(&clean), inf.clone()];
    drop(inf);
    assert!(matches!(
        g.clip_grad_norm(&mut grads, 1.0e-3),
        Err(OjasError::NonFinite { .. })
    ));
    close_vec(
        "clip untouched",
        &down(&grads[0]),
        &clean.to_f32_vec().unwrap(),
    );

    // AdamW with a NaN gradient refuses on the device: p, m and v unchanged.
    let p0 = host(1403, &[256]);
    let mut p = up(&p0);
    let mut m = up(&Tensor::zeros(&[256], DType::F32, host_budget()).unwrap());
    let mut v = up(&Tensor::zeros(&[256], DType::F32, host_budget()).unwrap());
    let mut bad = data(1404, 256);
    bad[200] = f32::NAN;
    let bad = up(&Tensor::from_f32(&bad, &[256], host_budget()).unwrap());
    let r = g.adamw_step(
        &mut p,
        &bad,
        &mut m,
        &mut v,
        1,
        AdamWConfig::nanolab(1e-2, 0.1),
    );
    let err = r.and_then(|()| g.sync());
    assert!(
        matches!(err, Err(OjasError::NonFinite { .. })),
        "got {err:?}"
    );
    g.sync().unwrap();
    close_vec("adam p untouched", &down(&p), &p0.to_f32_vec().unwrap());
    assert!(down(&m).iter().chain(down(&v).iter()).all(|x| *x == 0.0));
}

#[derive(Debug)]
struct Foreign;

impl DeviceBuffer for Foreign {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        16
    }
    fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(vec![0; len])
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn upload_policy_and_placement() {
    let g = &fresh();
    let h = host(1600, &[4]);
    let d = g.upload(&h).unwrap();
    let uploads = g.context().stats().uploads;
    let same = g.upload(&d).unwrap();
    assert_eq!(
        g.context().stats().uploads,
        uploads,
        "same-device upload copied"
    );
    assert_eq!(
        g.download(&same).unwrap().to_f32_vec().unwrap(),
        h.to_f32_vec().unwrap()
    );

    let foreign = Tensor::from_device(Arc::new(Foreign), &[4], DType::F32, host_budget()).unwrap();
    assert!(matches!(
        g.upload(&foreign),
        Err(OjasError::Placement { .. })
    ));
    assert!(matches!(
        g.silu_forward(&foreign),
        Err(OjasError::Placement { .. })
    ));
    assert!(
        matches!(g.silu_forward(&h), Err(OjasError::Placement { .. })),
        "host input accepted"
    );

    let other = WgpuBackend::open(Budget::new(1 << 20)).unwrap();
    let theirs = other.upload(&h).unwrap();
    assert!(
        matches!(g.silu_forward(&theirs), Err(OjasError::Placement { .. })),
        "other context accepted"
    );
    assert!(matches!(
        g.upload(&theirs),
        Err(OjasError::Placement { .. })
    ));
}

#[test]
fn device_tensors_charge_the_budget() {
    let budget = Budget::new(1 << 20);
    let g = WgpuBackend::open(budget.clone()).unwrap();
    let start = budget.live_bytes().unwrap();
    let d = g.upload(&host(1700, &[1000])).unwrap();
    assert_eq!(budget.live_bytes().unwrap(), start + 4000);
    let y = g.silu_forward(&d).unwrap();
    assert_eq!(budget.live_bytes().unwrap(), start + 8000);
    drop((d, y));
    assert_eq!(budget.live_bytes().unwrap(), start);
    // Past the budget: refused before any allocation.
    let big = host(1701, &[300_000]);
    match g.upload(&big) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }
    assert_eq!(budget.live_bytes().unwrap(), start);
}

#[test]
fn adapter_limits_are_refused_before_allocation() {
    let cap = wgpu::Limits {
        max_storage_buffer_binding_size: 1 << 16,
        max_buffer_size: 1 << 16,
        ..wgpu::Limits::default()
    };
    let ctx = WgpuContext::open_capped(&cap).unwrap();
    let budget = Budget::new(1 << 30);
    let g = WgpuBackend::with_context(ctx, budget.clone());
    let small = g.upload(&host(1800, &[1024])).unwrap();
    let _ = g.download(&g.silu_forward(&small).unwrap()).unwrap();
    let live = budget.live_bytes().unwrap();
    match g.upload(&host(1801, &[1 << 15])) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("upload past the binding limit: {other:?}"),
    }
    // An op whose output would pass the limit, from inputs that fit.
    let x = g.upload(&host(1802, &[128, 64])).unwrap();
    let w = g.upload(&host(1803, &[256, 64])).unwrap();
    let live2 = budget.live_bytes().unwrap();
    match g.linear_forward(&x, &w) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("linear output past the binding limit: {other:?}"),
    }
    assert_eq!(budget.live_bytes().unwrap(), live2);
    assert!(live2 > live);
}

#[test]
fn malformed_inputs_are_errors() {
    let g = gpu();
    let a = up(&host(1900, &[4, 3]));
    let b = up(&host(1901, &[3, 4]));
    let v = up(&host(1902, &[5]));
    let ids = up(&host_u32(&[0, 1], &[2]));
    let q = up(&host(1903, &[1, 1, 4, 8]));
    let q3 = up(&host(1904, &[1, 4, 8]));
    let results: Vec<(&str, Result<(), OjasError>)> = vec![
        ("linear k mismatch", g.linear_forward(&a, &b).map(drop)),
        ("silu' shape", g.silu_backward(&a, &b).map(drop)),
        ("mul shape", g.mul_forward(&a, &b).map(drop)),
        ("add shape", g.residual_add_forward(&a, &v).map(drop)),
        (
            "rms weight",
            g.rms_norm_forward(&a, &v, RMS_NORM_EPS).map(drop),
        ),
        (
            "sdpa rank 3",
            g.causal_sdpa_forward(&q3, &q3, &q3, None)
                .map(|(y, _)| y)
                .map(drop),
        ),
        (
            "sdpa kv shape",
            g.causal_sdpa_forward(&q, &q3, &q3, None)
                .map(|(y, _)| y)
                .map(drop),
        ),
        ("embed f32 ids", g.embedding_forward(&a, &a).map(drop)),
        (
            "embed f32 table dtype",
            g.embedding_forward(&ids, &ids).map(drop),
        ),
        (
            "ce targets",
            g.cross_entropy_mean_forward(&a, &ids, None).map(drop),
        ),
        (
            "vr lambda",
            g.value_residual_blend_forward(&v, &v, &v).map(drop),
        ),
        ("rope odd", g.rope_half_split_forward(&v, &v, &v).map(drop)),
        ("clip empty", g.clip_grad_norm(&mut [], 1.0).map(drop)),
    ];
    for (name, r) in results {
        match r {
            Err(
                OjasError::Shape { .. } | OjasError::Dtype { .. } | OjasError::OutOfRange { .. },
            ) => {}
            other => panic!("{name}: expected a shape/dtype error, got {other:?}"),
        }
    }
    let mut p = up(&host(1905, &[4]));
    let mut m = up(&host(1906, &[4]));
    // Muon is implemented (tests/muon.rs); a rank-1 parameter is refused.
    assert!(matches!(
        g.muon_ns5_step(&mut p, &v, &mut m, MuonNs5Config::nanolab_default()),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn concurrent_callers_get_their_own_results() {
    let g = &fresh();
    let c = cpu();
    let inputs: Vec<(Tensor, Tensor)> = (0..6)
        .map(|i| (host(2000 + i, &[65, 33]), host(2100 + i, &[17, 33])))
        .collect();
    let wants: Vec<Tensor> = inputs
        .iter()
        .map(|(x, w)| c.linear_forward(x, w).unwrap())
        .collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = inputs
            .iter()
            .map(|(x, w)| {
                s.spawn(move || {
                    let mut last = Vec::new();
                    for _ in 0..20 {
                        let y = g
                            .linear_forward(&g.upload(x).unwrap(), &g.upload(w).unwrap())
                            .unwrap();
                        last = g.download(&y).unwrap().to_f32_vec().unwrap();
                    }
                    last
                })
            })
            .collect();
        for (h, want) in handles.into_iter().zip(&wants) {
            close_vec(
                "concurrent linear",
                &h.join().unwrap(),
                &want.to_f32_vec().unwrap(),
            );
        }
    });
}
