//! Residency: ops leave their outputs on the device, and the only host reads
//! are the ones the caller asks for. Readbacks are counted on each test's own
//! budget (`Budget::device_readbacks`), so parallel tests cannot disturb
//! them. The lock remains because `CacheStats::reads` is per context and
//! these tests share the device.

mod common;

use std::sync::Mutex;

use common::*;
use ojas_core::{AdamWConfig, Backend, BackendId, DType, Tensor, RMS_NORM_EPS};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn clip_grad_norm_matches_cpu_in_place() {
    let _serial = serial();
    let c = cpu();
    for &max_norm in &[0.5f32, 1.0e6] {
        let shapes: [&[usize]; 3] = [&[1], &[73, 11], &[129, 65]];
        let mut hg: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| host(1000 + i as u64, s))
            .collect();
        let mut dg: Vec<Tensor> = hg.iter().map(up).collect();
        let g = own();
        let before = readbacks(&g);
        let gn = g.clip_grad_norm(&mut dg, max_norm).unwrap();
        assert_eq!(readbacks(&g), before, "clip downloaded a tensor");
        // Positive control: a download through `g` is counted.
        g.download(&dg[0]).unwrap();
        assert_eq!(readbacks(&g).0, before.0 + 1);
        let cn = c.clip_grad_norm(&mut hg, max_norm).unwrap();
        assert!(
            (f64::from(gn) - f64::from(cn)).abs() <= TOL * f64::from(cn.max(1.0)),
            "norm {gn} vs {cn}"
        );
        for (i, (g, h)) in dg.iter().zip(&hg).enumerate() {
            close(&format!("clip {max_norm} grad {i}"), g, h);
        }
    }
}

#[test]
fn outputs_stay_resident_through_a_full_forward_and_backward() {
    let _serial = serial();
    let g = &own();
    let x = up(&host(1500, &[2, 17, 32]));
    let w = up(&host(1501, &[32, 32]));
    let nw = up(&host(1502, &[32]));
    let before = readbacks(g);
    let h = g.rms_norm_forward(&x, &nw, RMS_NORM_EPS).unwrap();
    let y = g.linear_forward(&h, &w).unwrap();
    let s = g.silu_forward(&y).unwrap();
    let m = g.mul_forward(&s, &y).unwrap();
    let r = g.residual_add_forward(&m, &x).unwrap();
    let (dm, _) = g.residual_add_backward(&m, &x, &r).unwrap();
    let (ds, _) = g.mul_backward(&s, &y, &dm).unwrap();
    let dy = g.silu_backward(&y, &ds).unwrap();
    let (dh, dw) = g.linear_backward(&h, &w, &dy).unwrap();
    let (dx, _) = g.rms_norm_backward(&x, &nw, &dh, RMS_NORM_EPS).unwrap();
    for t in [&h, &y, &s, &m, &r, &dm, &ds, &dy, &dh, &dw, &dx] {
        assert_eq!(t.device(), Some(BackendId::Wgpu));
    }
    assert_eq!(readbacks(g), before, "an op read a tensor back");
    let _ = g.download(&dx).unwrap();
    assert_eq!(readbacks(g).0, before.0 + 1, "download is the one readback");
}

struct Model {
    table: Tensor,
    norm: Tensor,
    up: Tensor,
    out: Tensor,
}

/// Embedding, RMSNorm, SiLU MLP with a residual, output projection,
/// cross-entropy with ignore_index; clip then AdamW. Returns the loss.
fn train_step<B: Backend>(
    b: &B,
    p: &mut Model,
    opt: &mut [(Tensor, Tensor)],
    ids: &Tensor,
    tgt: &Tensor,
    step: u64,
) -> Tensor {
    let e = b.embedding_forward(&p.table, ids).unwrap();
    let h = b.rms_norm_forward(&e, &p.norm, RMS_NORM_EPS).unwrap();
    let u = b.linear_forward(&h, &p.up).unwrap();
    let s = b.silu_forward(&u).unwrap();
    let r = b.residual_add_forward(&e, &s).unwrap();
    let logits = b.linear_forward(&r, &p.out).unwrap();
    let loss = b.cross_entropy_mean_forward(&logits, tgt, Some(0)).unwrap();
    let gl = b
        .cross_entropy_mean_backward(&logits, tgt, Some(0))
        .unwrap();
    let (gr, g_out) = b.linear_backward(&r, &p.out, &gl).unwrap();
    let (ge1, gs) = b.residual_add_backward(&e, &s, &gr).unwrap();
    let gu = b.silu_backward(&u, &gs).unwrap();
    let (gh, g_up) = b.linear_backward(&h, &p.up, &gu).unwrap();
    let (ge2, g_norm) = b.rms_norm_backward(&e, &p.norm, &gh, RMS_NORM_EPS).unwrap();
    let ge = b.residual_add_forward(&ge1, &ge2).unwrap();
    let g_table = b.embedding_backward(&p.table, ids, &ge).unwrap();
    let mut grads = vec![g_table, g_norm, g_up, g_out];
    b.clip_grad_norm(&mut grads, 1.0).unwrap();
    let cfg = AdamWConfig::nanolab(1e-2, 0.01);
    let params = [&mut p.table, &mut p.norm, &mut p.up, &mut p.out];
    for ((param, grad), (m, v)) in params.into_iter().zip(&grads).zip(opt.iter_mut()) {
        b.adamw_step(param, grad, m, v, step, cfg).unwrap();
    }
    loss
}

#[test]
fn training_loop_matches_cpu_and_reads_back_only_the_loss() {
    let _serial = serial();
    let (vocab, dim, hidden_seq) = (31usize, 16usize, 17usize);
    let c = cpu();
    let g = &own();
    let mk = |s: u64| Model {
        table: host(s, &[vocab, dim]),
        norm: Tensor::from_f32(&vec![1.0; dim], &[dim], host_budget()).unwrap(),
        up: host(s + 1, &[dim, dim]),
        out: host(s + 2, &[vocab, dim]),
    };
    let zeros = |shape: &[usize]| Tensor::zeros(shape, DType::F32, host_budget()).unwrap();
    let shapes: [&[usize]; 4] = [&[vocab, dim], &[dim], &[dim, dim], &[vocab, dim]];
    let mut hp = mk(3000);
    let mut ho: Vec<(Tensor, Tensor)> = shapes.iter().map(|s| (zeros(s), zeros(s))).collect();
    let mut dp = Model {
        table: up(&hp.table),
        norm: up(&hp.norm),
        up: up(&hp.up),
        out: up(&hp.out),
    };
    let mut dopt: Vec<(Tensor, Tensor)> = ho.iter().map(|(m, v)| (up(m), up(v))).collect();
    let ids: Vec<u32> = (0..2 * hidden_seq)
        .map(|i| ((i * 5 + 1) % vocab) as u32)
        .collect();
    let tgt: Vec<u32> = (0..2 * hidden_seq)
        .map(|i| ((i * 3) % vocab) as u32)
        .collect();
    let (hids, htgt) = (
        host_u32(&ids, &[2, hidden_seq]),
        host_u32(&tgt, &[2, hidden_seq]),
    );
    let (dids, dtgt) = (up(&hids), up(&htgt));

    let steps = 8u64;
    let before = readbacks(g);
    let reads = g.context().stats().reads;
    let mut losses = Vec::new();
    for step in 1..=steps {
        let cl = train_step(&c, &mut hp, &mut ho, &hids, &htgt, step)
            .to_f32_vec()
            .unwrap()[0];
        let gl = g
            .download(&train_step(g, &mut dp, &mut dopt, &dids, &dtgt, step))
            .unwrap()
            .to_f32_vec()
            .unwrap()[0];
        losses.push((gl, cl));
    }
    let (count, _) = readbacks(g);
    assert_eq!(count - before.0, steps, "readbacks besides the loss");
    // The clip norm is the one other host read per step: clip returns an f32.
    assert_eq!(g.context().stats().reads - reads, 2 * steps);
    for (i, (gl, cl)) in losses.iter().enumerate() {
        assert!(
            (f64::from(*gl) - f64::from(*cl)).abs() <= TOL * f64::from(cl.abs().max(1.0)),
            "step {}: gpu {gl} cpu {cl}",
            i + 1
        );
    }
    assert!(
        losses[steps as usize - 1].1 < losses[0].1,
        "the model did not train: {losses:?}"
    );
    close("trained table", &dp.table, &hp.table);
    close("trained norm", &dp.norm, &hp.norm);
    close("trained up", &dp.up, &hp.up);
    close("trained out", &dp.out, &hp.out);
}
