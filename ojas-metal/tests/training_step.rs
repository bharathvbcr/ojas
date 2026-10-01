//! Several full training steps (forward, backward, clip, AdamW, Muon) on
//! Metal against the CPU reference, reading back only the loss.
//!
//! Readbacks are counted on the Metal backend's own budget
//! (`Budget::device_readbacks`), which only this backend's downloads charge,
//! so tests on other threads and backends cannot move the count.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AdamWConfig, Backend, BackendId, MuonNs5Config, OjasError, Tensor};

const V: usize = 64;
const D: usize = 64;
const B: usize = 2;
const T: usize = 17;
const STEPS: usize = 5;
/// Per-step loss tolerance, relative: Fast numerics compound through
/// backward and five optimizer updates.
const LOSS_RTOL: f32 = 1e-4;

struct Params {
    emb: Tensor,
    rw: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    gw: Tensor,
    gb: Tensor,
    wo: Tensor,
    head: Tensor,
}

impl Params {
    fn host() -> Self {
        let s = 0.15;
        Self {
            emb: rand(&[V, D], 1, 1.0),
            rw: host(&vec![1.0; D], &[D]),
            wq: rand(&[D, D], 2, s),
            wk: rand(&[D, D], 3, s),
            wv: rand(&[D, D], 4, s),
            gw: rand(&[1, D], 5, s),
            gb: host(&[0.0], &[1]),
            wo: rand(&[D, D], 6, s),
            head: rand(&[V, D], 7, s),
        }
    }

    fn map(&self, f: impl Fn(&Tensor) -> Tensor) -> Self {
        Self {
            emb: f(&self.emb),
            rw: f(&self.rw),
            wq: f(&self.wq),
            wk: f(&self.wk),
            wv: f(&self.wv),
            gw: f(&self.gw),
            gb: f(&self.gb),
            wo: f(&self.wo),
            head: f(&self.head),
        }
    }

    fn all_mut(&mut self) -> [&mut Tensor; 9] {
        [
            &mut self.emb,
            &mut self.rw,
            &mut self.wq,
            &mut self.wk,
            &mut self.wv,
            &mut self.gw,
            &mut self.gb,
            &mut self.wo,
            &mut self.head,
        ]
    }
}

fn reshape(t: &Tensor, shape: &[usize]) -> Result<Tensor, OjasError> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    t.view(shape, &strides, t.byte_offset())
}

fn resident(b: &impl Backend, ts: &[&Tensor]) {
    if b.id() == BackendId::Metal {
        for t in ts {
            assert_eq!(
                t.device(),
                Some(BackendId::Metal),
                "an intermediate left the device"
            );
        }
    }
}

/// One step; returns the loss tensor (on the backend's device) and the
/// pre-clip gradient norm.
fn step<Bk: Backend>(
    b: &Bk,
    p: &mut Params,
    opt: &mut [(Tensor, Tensor); 9],
    tok: &Tensor,
    tgt: &Tensor,
    n: u64,
) -> Result<(Tensor, f32), OjasError> {
    let x = b.embedding_forward(&p.emb, tok)?;
    let h = b.rms_norm_forward(&x, &p.rw, 1e-6)?;
    let q = b.linear_forward(&h, &p.wq)?;
    let k = b.linear_forward(&h, &p.wk)?;
    let v = b.linear_forward(&h, &p.wv)?;
    let (q4, k4, v4) = (
        reshape(&q, &[B, 1, T, D])?,
        reshape(&k, &[B, 1, T, D])?,
        reshape(&v, &[B, 1, T, D])?,
    );
    let a = b.causal_sdpa_forward(&q4, &k4, &v4)?;
    let a4 = reshape(&a, &[B, T, 1, D])?;
    let ga = b.per_head_sigmoid_gate_forward(&h, &p.gw, &p.gb, &a4)?;
    let ga3 = reshape(&ga, &[B, T, D])?;
    let o = b.linear_forward(&ga3, &p.wo)?;
    let r = b.residual_add_forward(&x, &o)?;
    let s = b.silu_forward(&r)?;
    let logits = b.linear_forward(&s, &p.head)?;
    let l2 = reshape(&logits, &[B * T, V])?;
    let loss = b.cross_entropy_mean_forward(&l2, tgt, None)?;
    resident(b, &[&x, &h, &q, &a, &ga, &o, &r, &s, &logits, &loss]);

    let gl = b.cross_entropy_mean_backward(&l2, tgt, None)?;
    let gl3 = reshape(&gl, &[B, T, V])?;
    let (gs, ghead) = b.linear_backward(&s, &p.head, &gl3)?;
    let gr = b.silu_backward(&r, &gs)?;
    let (gx1, go) = b.residual_add_backward(&x, &o, &gr)?;
    let (gga, gwo) = b.linear_backward(&ga3, &p.wo, &go)?;
    let gga4 = reshape(&gga, &[B, T, 1, D])?;
    let gg = b.per_head_sigmoid_gate_backward(&h, &p.gw, &p.gb, &a4, &gga4)?;
    let gattn = reshape(&gg.attn_out, &[B, 1, T, D])?;
    let (gq, gk, gv) = b.causal_sdpa_backward(&q4, &k4, &v4, &gattn)?;
    let (gh2, gwq) = b.linear_backward(&h, &p.wq, &reshape(&gq, &[B, T, D])?)?;
    let (gh3, gwk) = b.linear_backward(&h, &p.wk, &reshape(&gk, &[B, T, D])?)?;
    let (gh4, gwv) = b.linear_backward(&h, &p.wv, &reshape(&gv, &[B, T, D])?)?;
    let gh = b.residual_add_forward(&gg.input, &gh2)?;
    let gh = b.residual_add_forward(&gh, &gh3)?;
    let gh = b.residual_add_forward(&gh, &gh4)?;
    let (gx2, grw) = b.rms_norm_backward(&x, &p.rw, &gh, 1e-6)?;
    let gx = b.residual_add_forward(&gx1, &gx2)?;
    let gemb = b.embedding_backward(&p.emb, tok, &gx)?;
    resident(b, &[&gl, &gs, &gr, &gga, &gq, &gk, &gv, &gh, &gx, &gemb]);

    // Order matches Params::all_mut.
    let mut grads = [gemb, grw, gwq, gwk, gwv, gg.weight, gg.bias, gwo, ghead];
    drop((gx, gx1, gx2, gh, gh2, gh3, gh4, gr, gs, gl3));
    let norm = b.clip_grad_norm(&mut grads, 1.0)?;
    let cfg = AdamWConfig::nanolab(3e-3, 0.0);
    let muon = MuonNs5Config::nanolab_default();
    for (i, (param, g)) in p.all_mut().into_iter().zip(&grads).enumerate() {
        let (m1, m2) = &mut opt[i];
        if i == 7 {
            b.muon_ns5_step(param, g, m1, muon)?;
        } else {
            b.adamw_step(param, g, m1, m2, n, cfg)?;
        }
    }
    resident(b, &[&p.emb, &p.wo, &opt[0].0]);
    Ok((loss, norm))
}

fn zeros_like(t: &Tensor) -> Tensor {
    host(&vec![0.0; t.shape().iter().product()], t.shape())
}

fn moments(p: &Params, mk: impl Fn(Tensor) -> Tensor) -> [(Tensor, Tensor); 9] {
    let all = [
        &p.emb, &p.rw, &p.wq, &p.wk, &p.wv, &p.gw, &p.gb, &p.wo, &p.head,
    ];
    all.map(|t| (mk(zeros_like(t)), mk(zeros_like(t))))
}

#[test]
fn metal_training_steps_track_cpu_loss_and_read_back_only_the_loss() {
    let (m, c) = (metal(), cpu());
    let tok_v = ids(B * T, 11, V as u32);
    let tgt_v: Vec<u32> = tok_v.iter().map(|&t| (t * 7 + 3) % V as u32).collect();
    let (tok, tgt) = (host_u32(&tok_v, &[B, T]), host_u32(&tgt_v, &[B * T]));

    let mut hp = Params::host();
    let mut hopt = moments(&hp, |t| t);
    let mut dp = hp.map(|t| up(&m, t));
    let mut dopt = moments(&hp, |t| up(&m, &t));
    let (dtok, dtgt) = (up(&m, &tok), up(&m, &tgt));

    let mut cpu_losses = Vec::new();
    let mut metal_losses = Vec::new();
    for n in 0..STEPS as u64 {
        let (cl, cn) = ok("cpu step", step(&c, &mut hp, &mut hopt, &tok, &tgt, n));
        cpu_losses.push(ok("cpu loss", cl.to_f32_vec())[0]);

        let counted = || m.budget().device_readbacks();
        let before = counted();
        let (ml, mn) = ok("metal step", step(&m, &mut dp, &mut dopt, &dtok, &dtgt, n));
        assert_eq!(counted(), before, "step {n}: a readback before the loss download");
        // `download` charges the backend's budget, the one counted here.
        let loss = ok("download", m.download(&ml));
        metal_losses.push(ok("loss", loss.to_f32_vec())[0]);
        let after = counted();
        assert_eq!(
            after.0,
            before.0 + 1,
            "step {n}: exactly one readback (the loss)"
        );
        assert_eq!(after.1, before.1 + 4, "step {n}: the loss is 4 bytes");
        close(&format!("step {n} grad norm"), &[mn], &[cn], 1e-5, 1e-4);
    }
    for (n, (&g, &w)) in metal_losses.iter().zip(&cpu_losses).enumerate() {
        close(&format!("step {n} loss"), &[g], &[w], 0.0, LOSS_RTOL);
    }
    assert!(
        cpu_losses[STEPS - 1] < cpu_losses[0],
        "the model does not learn: {cpu_losses:?}"
    );
    eprintln!("cpu   losses {cpu_losses:?}\nmetal losses {metal_losses:?}");
}
