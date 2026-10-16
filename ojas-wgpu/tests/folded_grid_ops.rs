//! The folded-grid kernels beyond the norm and sum stages, under a device
//! that reports two workgroups per dimension (`common::capped`). A 2-D grid
//! of at most four groups covers 1,024 lanes or four rows, so each case sits
//! on a group boundary (255, 256, 257, 512, 513, ...) where spare
//! workgroups exist and must stay inert. Every op is held to the CPU.
//!
//! Entry points these ops dispatch (read from the backend's kernel
//! constants): `permute`, `embed_fwd`, `embed_bwd`, `ce_fwd`, `ce_bwd`,
//! `add_inplace` (accumulate_grad), `round_bf16` (cast_bf16), `widen_bf16`,
//! `narrow_bf16` (to_f32 / to_bf16), `kv_write`, `argmax_rows`, `topk_rows`,
//! `adam_check`, `adam_apply`. `scale_grad` is not one: wgpu composes it as
//! a broadcast gemm, which a cap of 2 refuses. The other folded kernels are
//! classified with their guards in `ojas_kernels::KERNEL_GRIDS`, whose unit
//! test fails on one that is missing or whose guard follows a buffer access.

mod common;

use common::*;
use ojas_core::{AdamWConfig, Backend, Tensor};

/// Element counts around each group boundary of a cap-2 grid (at most 1,024).
const LANES: [usize; 11] = [1, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1024];

fn dev(g: &impl Backend, t: &Tensor) -> Tensor {
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
        let mut da = dev(&g, &a);
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
        let (mut dp, mut dm, mut dv) = (dev(&g, &p), dev(&g, &hm), dev(&g, &hv));
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
        let mut dc = dev(&g, &base);
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
