//! The folded-grid kernels beyond the norm and sum stages, under a device
//! that reports two workgroups per dimension (`common::capped`). A 2-D grid
//! of at most four groups covers 1,024 lanes or four rows, so each case sits
//! on a group boundary (255, 256, 257, 512, 513, ...) where spare
//! workgroups exist and must stay inert. Every op is held to the CPU.
//!
//! Guard-removal audit (each kernel's early return deleted in turn, then
//! restored): these tests, the `folded_grid.rs` ones and the sum/norm tests
//! in `ojas-wgpu/src/backend.rs` turn red for `permute`, `kv_write`,
//! `widen_bf16`, `embed_fwd`, `ce_fwd`, `ce_bwd`, `rope`, `gate_fwd`,
//! `gate_fwd_save`, `gate_bwd`, `gate_bwd_saved`, `lce_stats`, `lce_finish`,
//! `lce_grad`, `argmax_rows`, `adam_apply`, `attn_bwd_dr`, `sum_partial` and
//! `norm_partial`. For the other folded kernels, run here under the cap
//! with poisoned input padding (`common::padded`), removing the guard is not
//! observable on this device: a spare lane's write past the end of its
//! output is dropped or lands in allocation slack, and its clamped read is
//! the value the owning lane read. They are `round_bf16`, `silu_*`, `mul_*`,
//! `add_fwd`, `add_inplace`, `vr_*`, `embed_bwd`, `narrow_bf16`, `rms_*`,
//! `adam_check`, `muon_*`, `topk_rows` and `scale_multi`. Their guards are
//! held by the static audit, `ojas_kernels::KERNEL_GRIDS`, whose unit test
//! fails on a kernel that is missing or whose guard follows a buffer access
//! (it went red for every removal above). `add_bwd` and `scale`, dispatched
//! since that audit (ft-2c183b71), have not been through the removal run;
//! the static audit and the capped cases here hold their guards.
//!
//! Kernels with no capped run:
//! - `col_finish`: one lane per column, `ceil(cols / 256)` groups, the same
//!   count as the x extent of the column-sum grid before it; that grid
//!   refuses more than the cap per axis, so `col_finish` never folds.
//! - `check_finite`: dispatched by the fault scans; no op here observably
//!   depends on it.

mod common;

use common::*;
use ojas_core::{AdamWConfig, Backend, CeChunk, CeGrads, DType, MuonNs5Config, Tensor};

/// Element counts around each group boundary of a cap-2 grid (at most 1,024).
const LANES: [usize; 11] = [1, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1024];

fn dev(g: &impl Backend, t: &Tensor) -> Tensor {
    padded(g, t)
}

/// `t` on the device in a buffer of exactly its size: the target of an
/// in-place op, whose spare-lane writes clamp onto its last element.
fn exact(g: &impl Backend, t: &Tensor) -> Tensor {
    g.upload(t).unwrap()
}

fn back(g: &impl Backend, t: &Tensor) -> Vec<f32> {
    g.download(t).unwrap().to_f32_vec().unwrap()
}

/// A host tensor with its own allocation: the CPU writes in place only to a
/// uniquely owned one, which a `clone` is not.
fn fresh_copy(t: &Tensor) -> Tensor {
    Tensor::from_f32(&t.to_f32_vec().unwrap(), t.shape(), host_budget()).unwrap()
}

fn same(name: &str, got: &[f32], want: &[f32]) {
    close_vec(name, got, want);
}

#[test]
fn permute_and_layout_conversions_leave_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    for n in LANES {
        let x = host(11, &[1, n]);
        let t = g.permute(&dev(&g, &x), &[1, 0]).unwrap();
        same(
            &format!("permute {n}"),
            &back(&g, &t),
            &c.permute(&x, &[1, 0]).unwrap().to_f32_vec().unwrap(),
        );
        let r = g.cast_bf16(&dev(&g, &x)).unwrap();
        same(
            &format!("cast_bf16 {n}"),
            &back(&g, &r),
            &c.cast_bf16(&x).unwrap().to_f32_vec().unwrap(),
        );
        let narrowed = g.to_bf16(&dev(&g, &x)).unwrap();
        let widened = g.to_f32(&narrowed).unwrap();
        let want = c.to_f32(&c.to_bf16(&x).unwrap()).unwrap();
        same(
            &format!("to_bf16/to_f32 {n}"),
            &back(&g, &widened),
            &want.to_f32_vec().unwrap(),
        );
    }
}

#[test]
fn accumulate_grad_leaves_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    for n in LANES {
        let (a, b) = (host(21, &[n]), host(22, &[n]));
        let mut da = exact(&g, &a);
        g.accumulate_grad(&mut da, &dev(&g, &b)).unwrap();
        let mut ha = fresh_copy(&a);
        c.accumulate_grad(&mut ha, &b).unwrap();
        same(
            &format!("accumulate {n}"),
            &back(&g, &da),
            &ha.to_f32_vec().unwrap(),
        );
    }
}

#[test]
fn adamw_leaves_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    let cfg = AdamWConfig::nanolab(1e-2, 0.1);
    for n in LANES {
        let (p, gr) = (host(31, &[n]), host(32, &[n]));
        let hv0: Vec<f32> = data(34, n).iter().map(|x| x.abs()).collect();
        let mut hv = Tensor::from_f32(&hv0, &[n], host_budget()).unwrap();
        let (mut hp, mut hm) = (fresh_copy(&p), host(33, &[n]));
        let (mut dp, mut dm, mut dv) = (exact(&g, &p), exact(&g, &hm), exact(&g, &hv));
        g.adamw_step(&mut dp, &dev(&g, &gr), &mut dm, &mut dv, 3, cfg)
            .unwrap();
        c.adamw_step(&mut hp, &gr, &mut hm, &mut hv, 3, cfg)
            .unwrap();
        same(
            &format!("adamw p {n}"),
            &back(&g, &dp),
            &hp.to_f32_vec().unwrap(),
        );
        same(
            &format!("adamw m {n}"),
            &back(&g, &dm),
            &hm.to_f32_vec().unwrap(),
        );
        same(
            &format!("adamw v {n}"),
            &back(&g, &dv),
            &hv.to_f32_vec().unwrap(),
        );
    }
}

#[test]
fn embedding_and_cross_entropy_leave_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    let (vocab, dim) = (9usize, 4usize);
    for tokens in [63usize, 64, 65, 128, 129, 192, 193, 255, 256] {
        let table = host(41, &[vocab, dim]);
        let ids: Vec<u32> = (0..tokens).map(|i| ((i * 5) % vocab) as u32).collect();
        let ids = host_u32(&ids, &[tokens]);
        let gy = host(42, &[tokens, dim]);
        let (dt, di) = (dev(&g, &table), dev(&g, &ids));
        same(
            &format!("embed fwd {tokens}"),
            &back(&g, &g.embedding_forward(&dt, &di).unwrap()),
            &c.embedding_forward(&table, &ids)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
        same(
            &format!("embed bwd {tokens}"),
            &back(&g, &g.embedding_backward(&dt, &di, &dev(&g, &gy)).unwrap()),
            &c.embedding_backward(&table, &ids, &gy)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
    }
    for (rows, classes) in [(1usize, 7usize), (2, 300), (3, 11), (4, 257)] {
        let logits = host(43, &[rows, classes]);
        let targets: Vec<u32> = (0..rows).map(|i| ((i * 3) % classes) as u32).collect();
        let targets = host_u32(&targets, &[rows]);
        let (dl, dt) = (dev(&g, &logits), dev(&g, &targets));
        same(
            &format!("ce fwd {rows}x{classes}"),
            &back(&g, &g.cross_entropy_mean_forward(&dl, &dt, None).unwrap()),
            &c.cross_entropy_mean_forward(&logits, &targets, None)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
        same(
            &format!("ce bwd {rows}x{classes}"),
            &back(&g, &g.cross_entropy_mean_backward(&dl, &dt, None).unwrap()),
            &c.cross_entropy_mean_backward(&logits, &targets, None)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
    }
}

#[test]
fn row_selection_and_cache_writes_leave_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    for (rows, cols) in [(1usize, 9usize), (2, 300), (3, 17), (4, 257)] {
        let x = host(51, &[rows, cols]);
        let dx = dev(&g, &x);
        let got = g.argmax_rows(&dx).unwrap();
        assert_eq!(
            g.download(&got).unwrap().to_u32_vec().unwrap(),
            c.argmax_rows(&x).unwrap().to_u32_vec().unwrap(),
            "argmax {rows}x{cols}"
        );
        let (gv, gi) = g.topk_rows(&dx, 3).unwrap();
        let (cv, ci) = c.topk_rows(&x, 3).unwrap();
        same(
            &format!("topk values {rows}x{cols}"),
            &back(&g, &gv),
            &cv.to_f32_vec().unwrap(),
        );
        assert_eq!(
            g.download(&gi).unwrap().to_u32_vec().unwrap(),
            ci.to_u32_vec().unwrap(),
            "topk ids {rows}x{cols}"
        );
    }
    // Source elements b * tn * hkv * d sit on and around 256-lane boundaries.
    let (b, cap, hkv, d) = (1usize, 40usize, 4usize, 8usize);
    for (tn, at) in [(7usize, 0usize), (8, 3), (9, 5), (31, 2), (32, 8)] {
        let base = host(61, &[b, cap, hkv, d]);
        let src = host(62, &[b, tn, hkv, d]);
        let mut dc = exact(&g, &base);
        g.kv_cache_write(&mut dc, &dev(&g, &src), at).unwrap();
        let mut hc = fresh_copy(&base);
        c.kv_cache_write(&mut hc, &src, at).unwrap();
        g.sync().unwrap();
        same(
            &format!("kv_write tn {tn} at {at}"),
            &back(&g, &dc),
            &hc.to_f32_vec().unwrap(),
        );
    }
}

/// RoPE tables for `t` positions of head dim `d`, as `parity.rs` builds them.
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
fn rope_leaves_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    // d = 2 rotates one pair per position, so `t` pairs are `t` lanes.
    for t in [255usize, 256, 257, 511, 512, 513, 768, 769, 1024] {
        let x = host(71, &[1, t, 1, 2]);
        let gy = host(72, &[1, t, 1, 2]);
        let (cos, sin) = rope_tables(t, 2);
        let (dc, ds) = (dev(&g, &cos), dev(&g, &sin));
        same(
            &format!("rope fwd {t}"),
            &back(
                &g,
                &g.rope_half_split_forward(&dev(&g, &x), &dc, &ds).unwrap(),
            ),
            &c.rope_half_split_forward(&x, &cos, &sin)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
        same(
            &format!("rope bwd {t}"),
            &back(
                &g,
                &g.rope_half_split_backward(&dev(&g, &gy), &dc, &ds).unwrap(),
            ),
            &c.rope_half_split_backward(&gy, &cos, &sin)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
    }
}

#[test]
fn gates_and_value_residual_leave_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    // (rows, din, heads, head dim): `rows * heads * hd` lanes forward and
    // `rows * heads` lanes in the backward. The gate's projection is a gemm
    // of `rows x heads` outputs, which a cap of 2 limits to 32 each, so the
    // forward folds through the head dim and the backward through heads.
    for (rows, din, heads, hd) in [
        (2usize, 3usize, 2usize, 63usize),
        (2, 3, 2, 64),
        (2, 3, 2, 65),
        (2, 3, 2, 129),
        (2, 3, 2, 256),
        (32, 3, 17, 1),
        (32, 3, 24, 1),
        (32, 3, 32, 1),
    ] {
        let tag = format!("gate {rows}x{din} h{heads} d{hd}");
        let x = host(81, &[rows, din]);
        let w = host(82, &[heads, din]);
        let bias = host(83, &[heads]);
        let attn = host(84, &[rows, heads, hd]);
        let gy = host(85, &[rows, heads, hd]);
        let d = [&x, &w, &bias, &attn, &gy].map(|t| dev(&g, t));
        same(
            &format!("{tag} y"),
            &back(
                &g,
                &g.per_head_sigmoid_gate_forward(&d[0], &d[1], &d[2], &d[3])
                    .unwrap(),
            ),
            &c.per_head_sigmoid_gate_forward(&x, &w, &bias, &attn)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
        let want = c
            .per_head_sigmoid_gate_backward(&x, &w, &bias, &attn, &gy)
            .unwrap();
        let got = g
            .per_head_sigmoid_gate_backward(&d[0], &d[1], &d[2], &d[3], &d[4])
            .unwrap();
        let (saved_y, scales) = g
            .per_head_sigmoid_gate_forward_saving(&d[0], &d[1], &d[2], &d[3])
            .unwrap();
        same(
            &format!("{tag} saving y"),
            &back(&g, &saved_y),
            &c.per_head_sigmoid_gate_forward(&x, &w, &bias, &attn)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
        let scales = scales.expect("the saving forward keeps its scales");
        let saved = g
            .per_head_sigmoid_gate_backward_saved(&d[0], &d[1], &d[2], &d[3], &d[4], &scales)
            .unwrap();
        for (name, a, b) in [
            ("dx", &got.input, &want.input),
            ("dw", &got.weight, &want.weight),
            ("db", &got.bias, &want.bias),
            ("dattn", &got.attn_out, &want.attn_out),
            ("saved dx", &saved.input, &want.input),
            ("saved dw", &saved.weight, &want.weight),
            ("saved db", &saved.bias, &want.bias),
            ("saved dattn", &saved.attn_out, &want.attn_out),
        ] {
            same(
                &format!("{tag} {name}"),
                &back(&g, a),
                &b.to_f32_vec().unwrap(),
            );
        }
    }
    for n in LANES {
        let (v, v0) = (host(91, &[n]), host(92, &[n]));
        let lam = Tensor::from_f32(&[0.375], &[1], host_budget()).unwrap();
        let gy = host(93, &[n]);
        let (dv, dv0, dl, dgy) = (dev(&g, &v), dev(&g, &v0), dev(&g, &lam), dev(&g, &gy));
        same(
            &format!("vr fwd {n}"),
            &back(&g, &g.value_residual_blend_forward(&dv, &dv0, &dl).unwrap()),
            &c.value_residual_blend_forward(&v, &v0, &lam)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        );
        let got = g
            .value_residual_blend_backward(&dv, &dv0, &dl, &dgy)
            .unwrap();
        let want = c.value_residual_blend_backward(&v, &v0, &lam, &gy).unwrap();
        same(
            &format!("vr dv {n}"),
            &back(&g, &got.value),
            &want.value.to_f32_vec().unwrap(),
        );
        same(
            &format!("vr dv0 {n}"),
            &back(&g, &got.value0),
            &want.value0.to_f32_vec().unwrap(),
        );
        same(
            &format!("vr dlam {n}"),
            &back(&g, &got.lambda),
            &want.lambda.to_f32_vec().unwrap(),
        );
    }
}

#[test]
fn muon_leaves_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    let cfg = MuonNs5Config::nanolab_default();
    // 17 x 32 and 24 x 32 are 544 and 768 lanes (three groups on a (2, 2)
    // grid); the Newton-Schulz products stay within a 2 x 2 gemm grid.
    for (rows, cols) in [(17usize, 32usize), (24, 32), (32, 17)] {
        let mut hp = host(101, &[rows, cols]);
        let mut hm = Tensor::zeros(&[rows, cols], DType::F32, host_budget()).unwrap();
        let (mut dp, mut dm) = (exact(&g, &hp), exact(&g, &hm));
        let grad = host(102, &[rows, cols]);
        g.muon_ns5_step(&mut dp, &dev(&g, &grad), &mut dm, cfg)
            .unwrap_or_else(|e| panic!("muon {rows}x{cols}: {e}"));
        c.muon_ns5_step(&mut hp, &grad, &mut hm, cfg).unwrap();
        same(
            &format!("muon p {rows}x{cols}"),
            &back(&g, &dp),
            &hp.to_f32_vec().unwrap(),
        );
        same(
            &format!("muon m {rows}x{cols}"),
            &back(&g, &dm),
            &hm.to_f32_vec().unwrap(),
        );
    }
}

#[test]
fn fused_linear_cross_entropy_leaves_spare_workgroups_inert() {
    let c = cpu();
    // Fused linear + cross entropy: `rows` workgroups of statistics, then
    // one lane per tile element. (rows, vocab, cap): the second case needs a
    // larger cap to fold the element grid past the gemm's own tile limit.
    for (cap, rows, d, vocab) in [
        (2u32, 3usize, 8usize, 17usize),
        (2, 4, 8, 17),
        (5, 21, 8, 80),
    ] {
        let g = capped(cap);
        let x = host(121, &[rows, d]);
        let w = host(122, &[vocab, d]);
        let ids: Vec<u32> = (0..rows).map(|i| ((i * 7) % vocab) as u32).collect();
        let t = host_u32(&ids, &[rows]);
        let chunk = CeChunk { rows, cols: vocab };
        let got = g
            .linear_cross_entropy_mean(
                &dev(&g, &x),
                &dev(&g, &w),
                &dev(&g, &t),
                None,
                chunk,
                CeGrads::BOTH,
            )
            .unwrap_or_else(|e| panic!("lce cap {cap} {rows}x{vocab}: {e}"));
        let want = c
            .linear_cross_entropy_mean(&x, &w, &t, None, chunk, CeGrads::BOTH)
            .unwrap();
        let tag = format!("lce cap {cap} {rows}x{vocab}");
        same(
            &format!("{tag} loss"),
            &back(&g, &got.loss),
            &want.loss.to_f32_vec().unwrap(),
        );
        same(
            &format!("{tag} dx"),
            &back(&g, &got.grad_input.unwrap()),
            &want.grad_input.unwrap().to_f32_vec().unwrap(),
        );
        same(
            &format!("{tag} dw"),
            &back(&g, &got.grad_weight.unwrap()),
            &want.grad_weight.unwrap().to_f32_vec().unwrap(),
        );
    }
}

/// `attn_bwd_dr` runs one lane per query row, `groups(batch * heads * time)`,
/// while the attention kernels' own grids are `(ceil(time / 32), batch *
/// heads)` and never fold. Under a cap of 2 those grids bound the rows at
/// 2 * 32 * 2 = 128, under the 512 that folds; a cap of 9 with time 288 and
/// nine heads gives 2,592 rows, 11 groups, folded to (9, 2).
#[test]
fn the_attention_backward_row_pass_leaves_spare_workgroups_inert() {
    let g = capped(9);
    let c = cpu();
    let (b, h, t, d) = (1usize, 9usize, 288usize, 8usize);
    let (q, k, v, gy) = (
        host(131, &[b, h, t, d]),
        host(132, &[b, h, t, d]),
        host(133, &[b, h, t, d]),
        host(134, &[b, h, t, d]),
    );
    let (dq, dk, dv, dgy) = (dev(&g, &q), dev(&g, &k), dev(&g, &v), dev(&g, &gy));
    let (out, lse) = g.causal_sdpa_forward(&dq, &dk, &dv, None).unwrap();
    let (gq, gk, gv) = g
        .causal_sdpa_backward(&dq, &dk, &dv, &out, &lse, &dgy, None)
        .unwrap_or_else(|e| panic!("attention backward at cap 9: {e}"));
    let (cout, clse) = c.causal_sdpa_forward(&q, &k, &v, None).unwrap();
    let (cq, ck, cv) = c
        .causal_sdpa_backward(&q, &k, &v, &cout, &clse, &gy, None)
        .unwrap();
    same("attn dq", &back(&g, &gq), &cq.to_f32_vec().unwrap());
    same("attn dk", &back(&g, &gk), &ck.to_f32_vec().unwrap());
    same("attn dv", &back(&g, &gv), &cv.to_f32_vec().unwrap());
}

/// `scale_grad` in place (`scale_multi`) on a sole owner, and through the
/// pointwise `scale` kernel on a shared gradient (the other handle keeps
/// its values), at every group boundary of the capped grid.
#[test]
fn scale_grad_leaves_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    let want = |x: &Tensor| {
        let mut w = fresh_copy(x);
        c.scale_grad(&mut w, 0.375).unwrap();
        w.to_f32_vec().unwrap()
    };
    // In place, one workgroup per 4,096-value chunk: three chunks give the
    // (2, 2) grid with a spare.
    for n in LANES
        .iter()
        .copied()
        .chain([4096 + 1, 2 * 4096 + 1, 3 * 4096 + 5])
    {
        let x = host(151, &[n]);
        let mut sole = exact(&g, &x);
        g.scale_grad(&mut sole, 0.375).unwrap();
        same(&format!("in place {n}"), &back(&g, &sole), &want(&x));
    }
    for n in LANES {
        let x = host(152, &[n]);
        let mut shared = dev(&g, &x);
        let keep = shared.clone();
        g.scale_grad(&mut shared, 0.375).unwrap();
        same(&format!("shared {n}"), &back(&g, &shared), &want(&x));
        same(
            &format!("kept {n}"),
            &back(&g, &keep),
            &x.to_f32_vec().unwrap(),
        );
    }
}

/// `scale_multi` runs only when the norm exceeds `max_norm`: clip three
/// chunks (a spare on a (2, 2) grid) and compare every element with the CPU.
#[test]
fn clip_scaling_leaves_spare_workgroups_inert() {
    let g = capped(2);
    let c = cpu();
    for n in [4096usize + 1, 2 * 4096 + 1, 3 * 4096, 3 * 4096 + 5] {
        let x = host(141, &[n]);
        let mut dev_grads = [exact(&g, &x)];
        let mut cpu_grads = [fresh_copy(&x)];
        let got = g.clip_grad_norm(&mut dev_grads, 1.0).unwrap();
        let want = c.clip_grad_norm(&mut cpu_grads, 1.0).unwrap();
        assert!(got > 1.0, "the clip must act: norm {got}");
        same(
            &format!("clip scaled {n}"),
            &back(&g, &dev_grads[0]),
            &cpu_grads[0].to_f32_vec().unwrap(),
        );
        assert!((got - want).abs() <= 1e-4 * want, "norm {got} vs {want}");
    }
}
