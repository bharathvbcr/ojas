//! `MetalBackend` against `ojas_cpu::CpuBackend`, op by op.
//!
//! Metal is `Numerics::Fast`: GEMMs and reductions use a different order
//! (and FMA), so outputs are compared with `|got - want| <= atol + rtol*|want|`.
//! The tolerances here are the contract the crate documents.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{Backend, Numerics};

/// GEMM-based ops: K-length dot products of values in [-1, 1].
const GEMM_ATOL: f32 = 2e-5;
const GEMM_RTOL: f32 = 2e-4;
/// Pointwise and per-row ops.
const PW_ATOL: f32 = 1e-6;
const PW_RTOL: f32 = 1e-5;

const LINEAR: [(usize, usize, usize); 3] = [(1, 1, 1), (73, 11, 13), (127, 67, 131)];

#[test]
#[should_panic(expected = "index 1")]
fn comparator_rejects_a_single_perturbed_element() {
    close(
        "self-check",
        &[1.0, 2.001, 3.0],
        &[1.0, 2.0, 3.0],
        1e-6,
        1e-6,
    );
}

#[test]
#[should_panic(expected = "linear y")]
fn parity_harness_detects_a_wrong_device_result() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[4, 3], 1, 1.0);
    let w = rand(&[5, 3], 2, 1.0);
    let w2 = rand(&[5, 3], 3, 1.0);
    let want = ok("cpu", c.linear_forward(&x, &w));
    let got = ok("metal", m.linear_forward(&up(&m, &x), &up(&m, &w2)));
    same_tensor("linear y", &got, &want, GEMM_ATOL, GEMM_RTOL);
}

#[test]
fn adamw_matches_cpu_over_several_steps() {
    let (m, c) = (metal(), cpu());
    for (i, shape) in [vec![1usize], vec![73, 11, 13], vec![127, 67, 131]]
        .into_iter()
        .enumerate()
    {
        let n: usize = shape.iter().product();
        let cfg = ojas_core::AdamWConfig::nanolab(3e-3, 0.1);
        let mut p = rand(&shape, 700 + i as u64, 1.0);
        let mut m1 = host(&vec![0.0; n], &shape);
        let mut m2 = host(&vec![0.0; n], &shape);
        let mut dp = up(&m, &p);
        let mut dm1 = up(&m, &m1);
        let mut dm2 = up(&m, &m2);
        for step in 0..4u64 {
            let g = rand(&shape, 800 + step, 0.5);
            let dg = up(&m, &g);
            ok(
                "cpu adamw",
                c.adamw_step(&mut p, &g, &mut m1, &mut m2, step, cfg),
            );
            ok(
                "metal adamw",
                m.adamw_step(&mut dp, &dg, &mut dm1, &mut dm2, step, cfg),
            );
        }
        same_tensor("adamw p", &dp, &p, 1e-6, 1e-5);
        same_tensor("adamw m", &dm1, &m1, 1e-7, 1e-5);
        same_tensor("adamw v", &dm2, &m2, 1e-8, 1e-4);
    }
}

#[test]
fn muon_matches_cpu_for_wide_tall_and_square() {
    let (m, c) = (metal(), cpu());
    for &(rows, cols) in &[(1usize, 1usize), (11, 73), (131, 67), (64, 64)] {
        for nesterov in [true, false] {
            let cfg = ojas_core::MuonNs5Config {
                nesterov,
                ..ojas_core::MuonNs5Config::nanolab_default()
            };
            let shape = [rows, cols];
            let mut p = rand(&shape, rows as u64, 0.5);
            let mut mo = host(&vec![0.0; rows * cols], &shape);
            let (mut dp, mut dmo) = (up(&m, &p), up(&m, &mo));
            for step in 0..2u64 {
                let g = rand(&shape, 900 + step, 0.5);
                let dg = up(&m, &g);
                ok("cpu muon", c.muon_ns5_step(&mut p, &g, &mut mo, cfg));
                ok("metal muon", m.muon_ns5_step(&mut dp, &dg, &mut dmo, cfg));
            }
            let tag = format!("muon {rows}x{cols} nesterov={nesterov}");
            same_tensor(&format!("{tag} p"), &dp, &p, 2e-4, 1e-3);
            same_tensor(&format!("{tag} m"), &dmo, &mo, 1e-6, 1e-5);
        }
    }
}

/// `‖a − b‖ / ‖b‖`, normwise.
fn normwise_rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&a, &b) in a.iter().zip(b) {
        let d = f64::from(a) - f64::from(b);
        num += d * d;
        den += f64::from(b) * f64::from(b);
    }
    (num / den).sqrt()
}

/// bf16 Newton-Schulz on Metal against the CPU's, which the ojas-oracle
/// `muon_step_bf16` fixture holds to stock nanolab bit for bit.
///
/// The parameter starts at zero, `lr` is 1 and there is no decay, so on a
/// square or wide matrix the new parameter is exactly `-NS5(update)`: every
/// value must be bf16 (low 16 bits zero), which an f32 iteration never gives.
/// That is the test that tells the precisions apart.
///
/// Closeness to the CPU is bounded by [`NS5_BF16_REL`] only. Metal sums each
/// GEMM in another order, so a bf16 rounding of a GEMM output can land one
/// bf16 step apart, and the bf16 iteration is discontinuous in its input:
/// in torch alone, a 1e-6 relative change to a 64x192 input moves stock
/// NS5 by 1.9e-2 (`ojas-oracle/src/parity.rs`,
/// `TRACE_BF16_PARAM_NORMWISE_REL_TOL`). Measured here: 0 at 64x64, 6.4e-3 at
/// 192x64, 1.2e-2 at 64x192; the f32 iteration is about 3e-2 from either.
#[test]
fn muon_bf16_ns5_matches_cpu_bf16() {
    let (m, c) = (metal(), cpu());
    let cfg = ojas_core::MuonNs5Config {
        lr: 1.0,
        momentum: 0.99,
        weight_decay: 0.0,
        nesterov: true,
        ns5: ojas_core::Ns5Precision::Bf16,
    };
    for &(rows, cols) in &[
        (64usize, 64usize),
        (64, 192),
        (192, 64),
        (131, 67),
        (768, 768),
    ] {
        let shape = [rows, cols];
        let g = rand(&shape, 41 + cols as u64, 1e-3);
        let mo0 = rand(&shape, 42, 1e-3);
        let zeros = vec![0.0; rows * cols];
        // Fresh, uniquely owned copies: the step writes in place.
        let (mut p, mut mo) = (host(&zeros, &shape), host(&down(&mo0), &shape));
        ok("cpu muon", c.muon_ns5_step(&mut p, &g, &mut mo, cfg));
        let (mut dp, mut dmo) = (up(&m, &host(&zeros, &shape)), up(&m, &mo0));
        ok(
            "metal muon",
            m.muon_ns5_step(&mut dp, &up(&m, &g), &mut dmo, cfg),
        );
        let (want, got) = (down(&p), down(&dp));
        let rel = normwise_rel(&got, &want);
        eprintln!("muon bf16 {rows}x{cols}: metal vs cpu normwise {rel:e}");
        assert!(
            rel <= NS5_BF16_REL,
            "{rows}x{cols}: metal bf16 NS5 is {rel:e} from the CPU's"
        );
        if rows <= cols {
            for (i, v) in got.iter().enumerate() {
                assert_eq!(
                    v.to_bits() & 0xffff,
                    0,
                    "{rows}x{cols}: value {i} ({v:e}) is not bf16"
                );
            }
        }
        same_tensor(&format!("bf16 muon {rows}x{cols} m"), &dmo, &mo, 1e-6, 1e-5);
    }
}

/// Normwise bound on Metal's bf16-NS5 result against the CPU's.
const NS5_BF16_REL: f64 = 5e-2;

#[test]
fn clip_grad_norm_matches_cpu_and_scales_only_when_needed() {
    let (m, c) = (metal(), cpu());
    let shapes = [vec![1usize], vec![73, 11, 13], vec![127, 67]];
    for max_norm in [0.5f32, 1e6] {
        let mut hg: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| rand(s, 1000 + i as u64, 1.0))
            .collect();
        let mut dg: Vec<_> = hg.iter().map(|g| up(&m, g)).collect();
        let want = ok("cpu clip", c.clip_grad_norm(&mut hg, max_norm));
        let got = ok("metal clip", m.clip_grad_norm(&mut dg, max_norm));
        close("clip norm", &[got], &[want], 0.0, 1e-5);
        for (i, (d, h)) in dg.iter().zip(&hg).enumerate() {
            same_tensor(
                &format!("clipped grad {i} (max {max_norm})"),
                d,
                h,
                1e-7,
                1e-5,
            );
        }
    }
}

#[test]
fn metal_reports_fast_numerics() {
    assert_eq!(metal().numerics(), Numerics::Fast);
}

#[test]
fn linear_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for (i, &(rows, kin, nout)) in LINEAR.iter().enumerate() {
        let s = i as u64 * 10;
        let x = rand(&[rows, kin], s + 1, 1.0);
        let w = rand(&[nout, kin], s + 2, 1.0);
        let gy = rand(&[rows, nout], s + 3, 1.0);
        let (dx, dw, dgy) = (up(&m, &x), up(&m, &w), up(&m, &gy));
        let tol = GEMM_ATOL * kin.max(rows) as f32;
        let y = ok("cpu linear", c.linear_forward(&x, &w));
        let my = ok("metal linear", m.linear_forward(&dx, &dw));
        same_tensor("linear y", &my, &y, tol, GEMM_RTOL);
        let (gx, gw) = ok("cpu linear bwd", c.linear_backward(&x, &w, &gy));
        let (mgx, mgw) = ok("metal linear bwd", m.linear_backward(&dx, &dw, &dgy));
        same_tensor("linear dx", &mgx, &gx, GEMM_ATOL * nout as f32, GEMM_RTOL);
        same_tensor("linear dw", &mgw, &gw, GEMM_ATOL * rows as f32, GEMM_RTOL);
    }
}

#[test]
fn linear_accepts_leading_batch_dims() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[3, 5, 7], 41, 1.0);
    let w = rand(&[9, 7], 42, 1.0);
    let y = ok("cpu", c.linear_forward(&x, &w));
    let my = ok("metal", m.linear_forward(&up(&m, &x), &up(&m, &w)));
    same_tensor("batched linear", &my, &y, GEMM_ATOL * 7.0, GEMM_RTOL);
}

#[test]
fn rms_norm_and_qk_norm_match_cpu() {
    let (m, c) = (metal(), cpu());
    let eps = 1e-6;
    for (i, shape) in [vec![1usize, 1], vec![73, 11, 13], vec![127, 67, 131]]
        .into_iter()
        .enumerate()
    {
        let s = 200 + i as u64 * 10;
        let dim = *shape.last().unwrap_or(&1);
        let x = rand(&shape, s, 2.0);
        let w = rand(&[dim], s + 1, 1.5);
        let g = rand(&shape, s + 2, 1.0);
        let (dx, dw, dg) = (up(&m, &x), up(&m, &w), up(&m, &g));
        let tol = PW_ATOL * dim as f32;
        let want = ok("rms", c.rms_norm_forward(&x, &w, eps));
        same_tensor(
            "rms",
            &ok("m rms", m.rms_norm_forward(&dx, &dw, eps)),
            &want,
            tol,
            1e-4,
        );
        let (wx, ww) = ok("rms bwd", c.rms_norm_backward(&x, &w, &g, eps));
        let (gx, gw) = ok("m rms bwd", m.rms_norm_backward(&dx, &dw, &dg, eps));
        same_tensor("rms dx", &gx, &wx, tol, 1e-4);
        let rows: usize = shape[..shape.len() - 1].iter().product();
        same_tensor("rms dw", &gw, &ww, PW_ATOL * rows as f32, 1e-4);

        let (wq, wk) = ok("qk", c.rms_qk_norm_forward(&x, &g, &w, &w, eps));
        let (gq, gk) = ok("m qk", m.rms_qk_norm_forward(&dx, &dg, &dw, &dw, eps));
        same_tensor("qk q", &gq, &wq, tol, 1e-4);
        same_tensor("qk k", &gk, &wk, tol, 1e-4);
    }
}

#[test]
fn rope_matches_cpu_in_both_layouts() {
    let (m, c) = (metal(), cpu());
    // Same layout.
    for (i, shape) in [vec![1usize, 2], vec![73, 11, 14], vec![127, 67, 132]]
        .into_iter()
        .enumerate()
    {
        let s = 300 + i as u64 * 10;
        let x = rand(&shape, s, 2.0);
        let cs = rand(&shape, s + 1, 1.0);
        let sn = rand(&shape, s + 2, 1.0);
        let (dx, dc, ds) = (up(&m, &x), up(&m, &cs), up(&m, &sn));
        let want = ok("rope", c.rope_half_split_forward(&x, &cs, &sn));
        same_tensor(
            "rope",
            &ok("m", m.rope_half_split_forward(&dx, &dc, &ds)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
        let want = ok("rope bwd", c.rope_half_split_backward(&x, &cs, &sn));
        same_tensor(
            "rope bwd",
            &ok("m", m.rope_half_split_backward(&dx, &dc, &ds)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
    }
    // [B, T, H, D] with [T, D] tables.
    for &(t, h, d) in &[(1usize, 1usize, 16usize), (17, 3, 64), (257, 8, 16)] {
        let x = rand(&[2, t, h, d], 7 + t as u64, 2.0);
        let cs = rand(&[t, d], 8, 1.0);
        let sn = rand(&[t, d], 9, 1.0);
        let (dx, dc, ds) = (up(&m, &x), up(&m, &cs), up(&m, &sn));
        let want = ok("rope td", c.rope_half_split_forward(&x, &cs, &sn));
        same_tensor(
            "rope td",
            &ok("m", m.rope_half_split_forward(&dx, &dc, &ds)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
        let want = ok("rope td bwd", c.rope_half_split_backward(&x, &cs, &sn));
        same_tensor(
            "rope td bwd",
            &ok("m", m.rope_half_split_backward(&dx, &dc, &ds)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
    }
}

#[test]
fn embedding_matches_cpu_including_repeated_ids() {
    let (m, c) = (metal(), cpu());
    for &(vocab, dim, n) in &[
        (1usize, 1usize, 1usize),
        (73, 11, 13 * 5),
        (127, 67, 131 * 3),
    ] {
        let table = rand(&[vocab, dim], vocab as u64, 1.0);
        let tok = host_u32(&ids(n, n as u64, vocab as u32), &[n]);
        let g = rand(&[n, dim], 5, 1.0);
        let (dt, di, dg) = (up(&m, &table), up(&m, &tok), up(&m, &g));
        let want = ok("embed", c.embedding_forward(&table, &tok));
        same_tensor(
            "embed",
            &ok("m", m.embedding_forward(&dt, &di)),
            &want,
            0.0,
            0.0,
        );
        let want = ok("embed bwd", c.embedding_backward(&table, &tok, &g));
        // Both sum each row's gradients in ascending token order.
        same_tensor(
            "embed bwd",
            &ok("m", m.embedding_backward(&dt, &di, &dg)),
            &want,
            0.0,
            0.0,
        );
    }
}

#[test]
fn cross_entropy_matches_cpu_with_and_without_ignore() {
    let (m, c) = (metal(), cpu());
    for &(rows, vocab) in &[(1usize, 1usize), (73, 11), (127, 131), (17, 50_304)] {
        let logits = rand(&[rows, vocab], rows as u64, 4.0);
        let mut t = ids(rows, vocab as u64, vocab as u32);
        let (dl, dt) = (up(&m, &logits), up(&m, &host_u32(&t, &[rows])));
        let want = ok(
            "ce",
            c.cross_entropy_mean_forward(&logits, &host_u32(&t, &[rows]), None),
        );
        same_tensor(
            "ce",
            &ok("m ce", m.cross_entropy_mean_forward(&dl, &dt, None)),
            &want,
            1e-5,
            1e-5,
        );
        let want = ok(
            "ce bwd",
            c.cross_entropy_mean_backward(&logits, &host_u32(&t, &[rows]), None),
        );
        same_tensor(
            "ce bwd",
            &ok("m", m.cross_entropy_mean_backward(&dl, &dt, None)),
            &want,
            1e-7,
            1e-4,
        );

        if rows > 1 {
            let ignore = u32::MAX;
            t[0] = ignore;
            let ht = host_u32(&t, &[rows]);
            let dt = up(&m, &ht);
            let want = ok(
                "ce ign",
                c.cross_entropy_mean_forward(&logits, &ht, Some(ignore)),
            );
            same_tensor(
                "ce ign",
                &ok("m", m.cross_entropy_mean_forward(&dl, &dt, Some(ignore))),
                &want,
                1e-5,
                1e-5,
            );
            let want = ok(
                "ce ign bwd",
                c.cross_entropy_mean_backward(&logits, &ht, Some(ignore)),
            );
            same_tensor(
                "ce ign bwd",
                &ok("m", m.cross_entropy_mean_backward(&dl, &dt, Some(ignore))),
                &want,
                1e-7,
                1e-4,
            );
        }
    }
}

#[test]
fn per_head_gate_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for &(b, t, din, h, dh) in &[
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (2, 17, 11, 3, 16),
        (1, 127, 67, 8, 64),
    ] {
        let x = rand(&[b, t, din], 1, 1.0);
        let w = rand(&[h, din], 2, 1.0);
        let bias = rand(&[h], 3, 1.0);
        let a = rand(&[b, t, h, dh], 4, 1.0);
        let g = rand(&[b, t, h, dh], 5, 1.0);
        let d = [
            up(&m, &x),
            up(&m, &w),
            up(&m, &bias),
            up(&m, &a),
            up(&m, &g),
        ];
        let tol = GEMM_ATOL * din.max(dh * b * t) as f32;
        let want = ok("gate", c.per_head_sigmoid_gate_forward(&x, &w, &bias, &a));
        same_tensor(
            "gate",
            &ok(
                "m",
                m.per_head_sigmoid_gate_forward(&d[0], &d[1], &d[2], &d[3]),
            ),
            &want,
            tol,
            GEMM_RTOL,
        );
        let want = ok(
            "gate bwd",
            c.per_head_sigmoid_gate_backward(&x, &w, &bias, &a, &g),
        );
        let got = ok(
            "m gate bwd",
            m.per_head_sigmoid_gate_backward(&d[0], &d[1], &d[2], &d[3], &d[4]),
        );
        same_tensor("gate dx", &got.input, &want.input, tol, GEMM_RTOL);
        same_tensor("gate dw", &got.weight, &want.weight, tol, GEMM_RTOL);
        same_tensor("gate db", &got.bias, &want.bias, tol, GEMM_RTOL);
        same_tensor(
            "gate dattn",
            &got.attn_out,
            &want.attn_out,
            PW_ATOL,
            PW_RTOL,
        );
    }
}

#[test]
fn value_residual_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for (i, shape) in [vec![1usize], vec![73, 11, 13], vec![127, 67, 131]]
        .into_iter()
        .enumerate()
    {
        let n: usize = shape.iter().product();
        let v = rand(&shape, 600 + i as u64, 1.0);
        let v0 = rand(&shape, 610 + i as u64, 1.0);
        let lam = host(&[0.3], &[1]);
        let g = rand(&shape, 620 + i as u64, 1.0);
        let d = [up(&m, &v), up(&m, &v0), up(&m, &lam), up(&m, &g)];
        let want = ok("vres", c.value_residual_blend_forward(&v, &v0, &lam));
        same_tensor(
            "vres",
            &ok("m", m.value_residual_blend_forward(&d[0], &d[1], &d[2])),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
        let want = ok(
            "vres bwd",
            c.value_residual_blend_backward(&v, &v0, &lam, &g),
        );
        let got = ok(
            "m vres bwd",
            m.value_residual_blend_backward(&d[0], &d[1], &d[2], &d[3]),
        );
        same_tensor("vres dv", &got.value, &want.value, PW_ATOL, PW_RTOL);
        same_tensor("vres dv0", &got.value0, &want.value0, PW_ATOL, PW_RTOL);
        same_tensor(
            "vres dlam",
            &got.lambda,
            &want.lambda,
            1e-6 * n as f32,
            1e-4,
        );
    }
}

#[test]
fn causal_attention_matches_cpu_across_t_heads_and_head_dim() {
    let (m, c) = (metal(), cpu());
    let mut cases = 0;
    for &t in &[1usize, 17, 257] {
        for &h in &[1usize, 3, 8] {
            for &d in &[16usize, 64] {
                let b = if t == 257 { 1 } else { 2 };
                let shape = [b, h, t, d];
                let s = (t * 100 + h * 10 + d) as u64;
                let q = rand(&shape, s, 1.0);
                let k = rand(&shape, s + 1, 1.0);
                let v = rand(&shape, s + 2, 1.0);
                let g = rand(&shape, s + 3, 1.0);
                let dd = [up(&m, &q), up(&m, &k), up(&m, &v), up(&m, &g)];
                let tag = format!("sdpa t{t} h{h} d{d}");
                let want = ok(
                    &tag,
                    c.causal_sdpa_forward(&q, &k, &v, None).map(|(y, _)| y),
                );
                let got = ok(
                    &tag,
                    m.causal_sdpa_forward(&dd[0], &dd[1], &dd[2], None)
                        .map(|(y, _)| y),
                );
                same_tensor(&tag, &got, &want, 2e-5, 1e-4);
                let (wq, wk, wv) = ok(&tag, c.causal_sdpa_backward_recompute(&q, &k, &v, &g, None));
                let (gq, gk, gv) = ok(
                    &tag,
                    m.causal_sdpa_backward_recompute(&dd[0], &dd[1], &dd[2], &dd[3], None),
                );
                let tol = 2e-5 * (t as f32).sqrt().max(1.0);
                same_tensor(&format!("{tag} dq"), &gq, &wq, tol, 1e-3);
                same_tensor(&format!("{tag} dk"), &gk, &wk, tol, 1e-3);
                same_tensor(&format!("{tag} dv"), &gv, &wv, tol, 1e-3);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 18);
}

#[test]
fn pointwise_ops_match_cpu() {
    let (m, c) = (metal(), cpu());
    for (i, shape) in [vec![1usize], vec![73, 11, 13], vec![127, 67, 131]]
        .into_iter()
        .enumerate()
    {
        let s = 100 + i as u64 * 10;
        let a = rand(&shape, s, 3.0);
        let b = rand(&shape, s + 1, 3.0);
        let g = rand(&shape, s + 2, 1.0);
        let (da, db, dg) = (up(&m, &a), up(&m, &b), up(&m, &g));

        let want = ok("silu", c.silu_forward(&a));
        same_tensor(
            "silu",
            &ok("m silu", m.silu_forward(&da)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
        let want = ok("silu bwd", c.silu_backward(&a, &g));
        same_tensor(
            "silu bwd",
            &ok("m silu bwd", m.silu_backward(&da, &dg)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );

        let want = ok("mul", c.mul_forward(&a, &b));
        same_tensor(
            "mul",
            &ok("m mul", m.mul_forward(&da, &db)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
        let (wa, wb) = ok("mul bwd", c.mul_backward(&a, &b, &g));
        let (ga, gb) = ok("m mul bwd", m.mul_backward(&da, &db, &dg));
        same_tensor("mul da", &ga, &wa, PW_ATOL, PW_RTOL);
        same_tensor("mul db", &gb, &wb, PW_ATOL, PW_RTOL);

        let want = ok("add", c.residual_add_forward(&a, &b));
        same_tensor(
            "add",
            &ok("m add", m.residual_add_forward(&da, &db)),
            &want,
            PW_ATOL,
            PW_RTOL,
        );
        let (wa, wb) = ok("add bwd", c.residual_add_backward(&a, &b, &g));
        let (ga, gb) = ok("m add bwd", m.residual_add_backward(&da, &db, &dg));
        same_tensor("add dx", &ga, &wa, 0.0, 0.0);
        same_tensor("add dy", &gb, &wb, 0.0, 0.0);
    }
}
