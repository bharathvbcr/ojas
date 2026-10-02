//! Release benchmarks for `MetalBackend`.
//!
//! `cargo run -p ojas-metal --release --example metal_bench [iters] [only]`
//!
//! `only` runs one group: `overhead`, `linear`, `attn`, `attn2048` or `step`.
//! Rows report the min and the median of `iters` timed runs after two
//! warm-ups.
//!
//! Ops are recorded and return before the device runs them
//! (`docs/metal-deferred-faults.md`), so every timed run ends with
//! `Backend::sync` and is device-complete time. Each row reports:
//! - resident: inputs already on the device, outputs left there;
//! - with transfer: host inputs uploaded, then the op, then every output
//!   downloaded. For the full step the parameters and optimizer state stay
//!   resident (as in training); the transfer is the batch up and the loss down.
//!
//! The full step runs with heads of 64. q/k/v `[B, T, d]` are reshaped (not
//! permuted) to `[B, H, T, 64]`, as `benches/torch_mps.py` does, so the two
//! time the same graph; the FLOPs and bytes match a standard head split, the
//! token-head mixing does not, which does not change timing.

use std::time::Instant;

use ojas_core::{AdamWConfig, Backend, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

type R<T> = Result<T, OjasError>;

const GIB: u64 = 1 << 30;

fn host_budget() -> Budget {
    Budget::new(16 * GIB)
}

fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn rand(shape: &[usize], seed: u64, scale: f32) -> R<Tensor> {
    Tensor::from_f32(
        &values(shape.iter().product(), seed, scale),
        shape,
        &host_budget(),
    )
}

fn zeros(shape: &[usize]) -> R<Tensor> {
    Tensor::zeros(shape, ojas_core::DType::F32, &host_budget())
}

fn reshape(t: &Tensor, shape: &[usize]) -> R<Tensor> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    t.view(shape, &strides, t.byte_offset())
}

/// `(min, median)` milliseconds of `iters` runs after two warm-ups. Each
/// run ends with `sync`, so it is device-complete time.
fn time(m: &MetalBackend, iters: usize, mut f: impl FnMut() -> R<()>) -> R<(f64, f64)> {
    for _ in 0..2 {
        f()?;
        m.sync()?;
    }
    let mut ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f()?;
        m.sync()?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    Ok((ms[0], ms[ms.len() / 2]))
}

fn row(name: &str, resident: (f64, f64), transfer: (f64, f64), flops: f64) {
    let tflops = flops / (resident.1 * 1e-3) / 1e12;
    println!(
        "| {name} | {:.3} | {:.3} | {:.3} | {:.3} | {tflops:.2} |",
        resident.0, resident.1, transfer.0, transfer.1
    );
}

fn ups(m: &MetalBackend, hs: &[&Tensor]) -> R<Vec<Tensor>> {
    hs.iter().map(|h| m.upload(h)).collect()
}

fn downs(ts: &[&Tensor]) -> R<()> {
    for t in ts {
        t.to_host(&host_budget())?;
    }
    Ok(())
}

fn linear(m: &MetalBackend, iters: usize, rows: usize, kin: usize, nout: usize) -> R<()> {
    let (x, w, g) = (
        rand(&[rows, kin], 1, 1.0)?,
        rand(&[nout, kin], 2, 1.0)?,
        rand(&[rows, nout], 3, 1.0)?,
    );
    let d = ups(m, &[&x, &w, &g])?;
    let flops = 2.0 * (rows * kin * nout) as f64;
    let name = format!("linear fwd {rows}x{kin}x{nout}");
    let res = time(m, iters, || m.linear_forward(&d[0], &d[1]).map(drop))?;
    let tr = time(m, iters, || {
        let u = ups(m, &[&x, &w])?;
        let y = m.linear_forward(&u[0], &u[1])?;
        downs(&[&y])
    })?;
    row(&name, res, tr, flops);
    let name = format!("linear bwd {rows}x{kin}x{nout}");
    let res = time(m, iters, || m.linear_backward(&d[0], &d[1], &d[2]).map(drop))?;
    let tr = time(m, iters, || {
        let u = ups(m, &[&x, &w, &g])?;
        let (a, b) = m.linear_backward(&u[0], &u[1], &u[2])?;
        downs(&[&a, &b])
    })?;
    row(&name, res, tr, 2.0 * flops);
    Ok(())
}

fn attention(m: &MetalBackend, iters: usize, b: usize, h: usize, t: usize, d: usize) -> R<()> {
    let shape = [b, h, t, d];
    let hs = [
        rand(&shape, 1, 1.0)?,
        rand(&shape, 2, 1.0)?,
        rand(&shape, 3, 1.0)?,
        rand(&shape, 4, 1.0)?,
    ];
    let dv = ups(m, &[&hs[0], &hs[1], &hs[2], &hs[3]])?;
    // Causal: half the T x T scores, 2 GEMM-like products of D each.
    let fwd = 2.0 * 2.0 * (b * h * d) as f64 * (t * t) as f64 / 2.0;
    let res = time(m, iters, || {
        m.causal_sdpa_forward(&dv[0], &dv[1], &dv[2]).map(drop)
    })?;
    let tr = time(m, iters, || {
        let u = ups(m, &[&hs[0], &hs[1], &hs[2]])?;
        let o = m.causal_sdpa_forward(&u[0], &u[1], &u[2])?;
        downs(&[&o])
    })?;
    row(&format!("attn fwd B{b} H{h} T{t} D{d}"), res, tr, fwd);
    let res = time(m, iters, || {
        m.causal_sdpa_backward(&dv[0], &dv[1], &dv[2], &dv[3])
            .map(drop)
    })?;
    let tr = time(m, iters, || {
        let u = ups(m, &[&hs[0], &hs[1], &hs[2], &hs[3]])?;
        let (a, bb, c) = m.causal_sdpa_backward(&u[0], &u[1], &u[2], &u[3])?;
        downs(&[&a, &bb, &c])
    })?;
    row(&format!("attn bwd B{b} H{h} T{t} D{d}"), res, tr, 2.5 * fwd);
    Ok(())
}

struct Model {
    p: Vec<Tensor>,
    m1: Vec<Tensor>,
    m2: Vec<Tensor>,
    b: usize,
    t: usize,
    d: usize,
    v: usize,
}

// Parameter order.
const EMB: usize = 0;
const RW: usize = 1;
const WQ: usize = 2;
const WK: usize = 3;
const WV: usize = 4;
const GW: usize = 5;
const GB: usize = 6;
const WO: usize = 7;
const HEAD: usize = 8;

impl Model {
    fn new(m: &MetalBackend, b: usize, t: usize, d: usize, v: usize) -> R<Self> {
        let heads = d / 64;
        let shapes: [Vec<usize>; 9] = [
            vec![v, d],
            vec![d],
            vec![d, d],
            vec![d, d],
            vec![d, d],
            vec![heads, d],
            vec![heads],
            vec![d, d],
            vec![v, d],
        ];
        let mut p = Vec::new();
        let (mut m1, mut m2) = (Vec::new(), Vec::new());
        for (i, s) in shapes.iter().enumerate() {
            p.push(m.upload(&rand(s, 10 + i as u64, 0.05)?)?);
            m1.push(m.upload(&zeros(s)?)?);
            m2.push(m.upload(&zeros(s)?)?);
        }
        Ok(Self {
            p,
            m1,
            m2,
            b,
            t,
            d,
            v,
        })
    }

    fn step(&mut self, be: &MetalBackend, tok: &Tensor, tgt: &Tensor, n: u64) -> R<Tensor> {
        let (b, t, d, v) = (self.b, self.t, self.d, self.v);
        let h = d / 64;
        let p = &self.p;
        let x = be.embedding_forward(&p[EMB], tok)?;
        let hn = be.rms_norm_forward(&x, &p[RW], 1e-6)?;
        let q = be.linear_forward(&hn, &p[WQ])?;
        let k = be.linear_forward(&hn, &p[WK])?;
        let vv = be.linear_forward(&hn, &p[WV])?;
        let s4 = [b, h, t, 64];
        let (q4, k4, v4) = (reshape(&q, &s4)?, reshape(&k, &s4)?, reshape(&vv, &s4)?);
        let a = be.causal_sdpa_forward(&q4, &k4, &v4)?;
        let a4 = reshape(&a, &[b, t, h, 64])?;
        let ga = be.per_head_sigmoid_gate_forward(&hn, &p[GW], &p[GB], &a4)?;
        let ga3 = reshape(&ga, &[b, t, d])?;
        let o = be.linear_forward(&ga3, &p[WO])?;
        let r = be.residual_add_forward(&x, &o)?;
        let s = be.silu_forward(&r)?;
        let logits = be.linear_forward(&s, &p[HEAD])?;
        let l2 = reshape(&logits, &[b * t, v])?;
        let loss = be.cross_entropy_mean_forward(&l2, tgt, None)?;

        let gl = be.cross_entropy_mean_backward(&l2, tgt, None)?;
        let (gs, ghead) = be.linear_backward(&s, &p[HEAD], &reshape(&gl, &[b, t, v])?)?;
        let gr = be.silu_backward(&r, &gs)?;
        let (gx1, go) = be.residual_add_backward(&x, &o, &gr)?;
        let (gga, gwo) = be.linear_backward(&ga3, &p[WO], &go)?;
        let gg = be.per_head_sigmoid_gate_backward(
            &hn,
            &p[GW],
            &p[GB],
            &a4,
            &reshape(&gga, &[b, t, h, 64])?,
        )?;
        let (gq, gk, gv) = be.causal_sdpa_backward(&q4, &k4, &v4, &reshape(&gg.attn_out, &s4)?)?;
        let (gh2, gwq) = be.linear_backward(&hn, &p[WQ], &reshape(&gq, &[b, t, d])?)?;
        let (gh3, gwk) = be.linear_backward(&hn, &p[WK], &reshape(&gk, &[b, t, d])?)?;
        let (gh4, gwv) = be.linear_backward(&hn, &p[WV], &reshape(&gv, &[b, t, d])?)?;
        let gh = be.residual_add_forward(&gg.input, &gh2)?;
        let gh = be.residual_add_forward(&gh, &gh3)?;
        let gh = be.residual_add_forward(&gh, &gh4)?;
        let (gx2, grw) = be.rms_norm_backward(&x, &p[RW], &gh, 1e-6)?;
        let gx = be.residual_add_forward(&gx1, &gx2)?;
        let gemb = be.embedding_backward(&p[EMB], tok, &gx)?;
        let mut grads = vec![gemb, grw, gwq, gwk, gwv, gg.weight, gg.bias, gwo, ghead];
        be.clip_grad_norm(&mut grads, 1.0)?;
        let cfg = AdamWConfig::nanolab(3e-4, 0.0);
        for (i, g) in grads.iter().enumerate() {
            be.adamw_step(&mut self.p[i], g, &mut self.m1[i], &mut self.m2[i], n, cfg)?;
        }
        Ok(loss)
    }
}

fn full_step(be: &MetalBackend, iters: usize, b: usize, t: usize, d: usize, v: usize) -> R<()> {
    let mut model = Model::new(be, b, t, d, v)?;
    let tok_v: Vec<u32> = (0..b * t).map(|i| (i * 2654435761 % v) as u32).collect();
    let tgt_v: Vec<u32> = tok_v.iter().map(|&x| (x + 1) % v as u32).collect();
    let tok_h = Tensor::from_u32(&tok_v, &[b, t], &host_budget())?;
    let tgt_h = Tensor::from_u32(&tgt_v, &[b * t], &host_budget())?;
    let (tok, tgt) = (be.upload(&tok_h)?, be.upload(&tgt_h)?);
    let mut n = 0u64;
    let res = time(be, iters, || {
        model.step(be, &tok, &tgt, n)?;
        n += 1;
        Ok(())
    })?;
    let tr = time(be, iters, || {
        let (a, bb) = (be.upload(&tok_h)?, be.upload(&tgt_h)?);
        let loss = model.step(be, &a, &bb, n)?;
        n += 1;
        downs(&[&loss])
    })?;
    // 6 * tokens * matmul params, plus attention.
    let mm = (4 * d * d + v * d) as f64;
    let attn = 2.0 * 2.0 * (b * d) as f64 * (t * t) as f64 / 2.0 * 3.5;
    let flops = 6.0 * (b * t) as f64 * mm + attn;
    row(&format!("full step B{b} T{t} d{d} V{v}"), res, tr, flops);
    Ok(())
}

fn main() -> R<()> {
    let iters = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20usize);
    let only = std::env::args().nth(2).unwrap_or_default();
    let run = |group: &str| only.is_empty() || only == group;
    let be = MetalBackend::new(Budget::new(24 * GIB))?;
    println!("device: {}, {iters} timed runs", be.device_name());
    println!(
        "| op | resident min ms | resident median ms | transfer min ms | transfer median ms | TFLOP/s (resident median) |"
    );
    println!("|---|---:|---:|---:|---:|---:|");
    if run("overhead") {
        // Fixed cost of one op round trip with its own sync: checks, one wait.
        let one = rand(&[1], 1, 1.0)?;
        let d1 = be.upload(&one)?;
        let res = time(&be, iters * 5, || be.silu_forward(&d1).map(drop))?;
        let tr = time(&be, iters * 5, || downs(&[&be.silu_forward(&be.upload(&one)?)?]))?;
        row("silu 1 element (per-op overhead)", res, tr, 0.0);
    }
    if run("linear") {
        linear(&be, iters, 512, 768, 768)?;
        linear(&be, iters, 2048, 2048, 2048)?;
    }
    if run("attn") {
        for t in [128, 512, 2048] {
            attention(&be, iters, 4, 8, t, 64)?;
        }
    }
    if only == "attn2048" {
        attention(&be, iters, 4, 8, 2048, 64)?;
    }
    if run("step") {
        for d in [512, 768] {
            full_step(&be, iters.min(10), 4, 128, d, 50_304)?;
        }
    }
    Ok(())
}
