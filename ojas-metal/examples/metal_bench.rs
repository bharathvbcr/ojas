//! Release benchmarks for `MetalBackend`.
//!
//! `cargo run -p ojas-metal --release --example metal_bench [iters] [only]`
//!
//! `only` runs one group: `overhead`, `linear`, `attn`, `attn2048` or `step`.
//! Rows report the min and the median of `iters` timed runs after two
//! warm-ups.
//!
//! `floor` (never part of the default run; macOS only) splits the per-op
//! fixed cost into phases, `25 × iters` runs per scenario:
//! - `MetalBackend`: the op call, `Backend::sync`, tessl's event wait inside
//!   it, and the commits, residency flushes and cold allocations each run
//!   made, from `tessl::infer_trace`. Scenarios: sync alone, one op and a
//!   sync, the same after an idle gap of 1, 5 or 20 ms, and 8 ops then one
//!   sync.
//! - tessl directly, with no device thread or channel: one in-place
//!   one-element `scale_f32_inplace` and `synchronize`, plus the GPU span
//!   between the command buffer's two timestamps. Also run after a 5 ms gap,
//!   and with a fresh 4-byte buffer allocated per run, as every
//!   `MetalBackend` op allocates its output.
//!
//! A run is "slow" when op plus sync exceed 1 ms. Each scenario prints the
//! mean of every phase over the fast and the slow runs separately, so the
//! phase that grows is the one holding the extra time. The GPU span is
//! converted with a ns-per-tick ratio measured against wall time over a
//! long command buffer. Wall time includes the host overhead, so the ratio
//! overstates by roughly that overhead's share; it is printed.
//!
//! `kernels` (never part of the default run; macOS only) loads this crate's
//! kernel library into a tessl runtime of its own and times each elementwise
//! kernel `MetalBackend` uses (each checks its own inputs and outputs), and
//! the standalone `ojas_check_finite` pass at several floats per thread.
//! Every command buffer holds just that, and the
//! GPU span between its timestamps is the time (min and median of
//! `max(iters, 10)` buffers). The shapes are the paired benchmark's:
//! `[4096, 2048]`, with the finite check also at `[4096, 768]`. Bandwidth is
//! the f32 passes the sequence must make over the data, divided by the
//! minimum. tessl's `scale_f32_inplace` (one read, one write) is the
//! reference for what a plain pass reaches.
//!
//! `percall` (never part of the default run; macOS only) splits what 16
//! decode requests (H 12, D 64, 1024 cached positions) cost as 16
//! `cached_attention_forward` calls against one batched call, `25 × iters`
//! runs per scenario:
//! - `MetalBackend`: host time of the calls, `Backend::sync`, tessl's event
//!   wait, and the commits, residency flushes, cold allocations, dispatches
//!   and barriers per run.
//! - tessl directly with this crate's `ojas_cached_attn`: the GPU span of one
//!   request, of the batched dispatch, of 16 dispatches with the barrier
//!   tessl records after each (as `MetalBackend` gets them), and of 16 with
//!   no barrier between them. Then the split cache walk (the kernel in
//!   parts, then `ojas_cached_attn_merge`) at 1, 2, 4 and 16 requests and 1
//!   to 32 splits.
//! - tessl directly: host time to record the 16 dispatches into one reused
//!   output, or each into a freshly allocated one as `MetalBackend` does.
//!
//! `gate` (macOS only) times, the same way, each dispatch of
//! `per_head_sigmoid_gate_backward` at the paired benchmark's shape (4096
//! rows, 12 heads of 64, d_model 768): the `pre` GEMM, the gate and bias-sum
//! kernels, the two gradient GEMMs, the standalone finite checks, and all of
//! them in one command buffer.
//!
//! `tn` (macOS only) times tessl's exact-f32 TN for a C of few tiles over a
//! long K: the single dispatch, the routed path, and the parallel split-K
//! (`gemm_tn_splitk_par_f32`) at several partition widths.
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
    let res = time(m, iters, || {
        m.linear_backward(&d[0], &d[1], &d[2]).map(drop)
    })?;
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
        m.causal_sdpa_forward(&dv[0], &dv[1], &dv[2], None)
            .map(|(y, _)| y)
            .map(drop)
    })?;
    let tr = time(m, iters, || {
        let u = ups(m, &[&hs[0], &hs[1], &hs[2]])?;
        let o = m
            .causal_sdpa_forward(&u[0], &u[1], &u[2], None)
            .map(|(y, _)| y)?;
        downs(&[&o])
    })?;
    row(&format!("attn fwd B{b} H{h} T{t} D{d}"), res, tr, fwd);
    // The backward from the forward's saved output and lse, as a tape runs it.
    let (o, lse) = m.causal_sdpa_forward(&dv[0], &dv[1], &dv[2], None)?;
    let res = time(m, iters, || {
        m.causal_sdpa_backward(&dv[0], &dv[1], &dv[2], &o, &lse, &dv[3], None)
            .map(drop)
    })?;
    let (ho, hl) = (m.download(&o)?, m.download(&lse)?);
    let tr = time(m, iters, || {
        let u = ups(m, &[&hs[0], &hs[1], &hs[2], &ho, &hl, &hs[3]])?;
        let (a, bb, c) = m.causal_sdpa_backward(&u[0], &u[1], &u[2], &u[3], &u[4], &u[5], None)?;
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
        let (a, a_lse) = be.causal_sdpa_forward(&q4, &k4, &v4, None)?;
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
        let gattn = reshape(&gg.attn_out, &s4)?;
        let (gq, gk, gv) = be.causal_sdpa_backward(&q4, &k4, &v4, &a, &a_lse, &gattn, None)?;
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
    if only == "floor" {
        return probe::floor(&be, iters * 25);
    }
    if only == "kernels" {
        return probe::kernels(iters);
    }
    if only == "percall" {
        return probe::percall(&be, iters);
    }
    if only == "gemm" {
        return probe::gemm(iters);
    }
    if only == "gate" {
        return probe::gate(iters);
    }
    if only == "tn" {
        return probe::tn(iters);
    }
    println!(
        "| op | resident min ms | resident median ms | transfer min ms | transfer median ms | TFLOP/s (resident median) |"
    );
    println!("|---|---:|---:|---:|---:|---:|");
    if run("overhead") {
        // Fixed cost of one op round trip with its own sync: checks, one wait.
        let one = rand(&[1], 1, 1.0)?;
        let d1 = be.upload(&one)?;
        let res = time(&be, iters * 5, || be.silu_forward(&d1).map(drop))?;
        let tr = time(&be, iters * 5, || {
            downs(&[&be.silu_forward(&be.upload(&one)?)?])
        })?;
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
    // Published paired-bench shape (`sdpa_b4h12t1024d64`).
    if only == "attnlab" {
        attention(&be, iters, 4, 12, 1024, 64)?;
    }
    if run("step") {
        for d in [512, 768] {
            full_step(&be, iters.min(10), 4, 128, d, 50_304)?;
        }
    }
    Ok(())
}

/// The `floor` and `kernels` groups: where one small op's fixed cost goes, and
/// what each elementwise kernel costs on the GPU. See the module docs.
#[cfg(target_os = "macos")]
mod probe {
    use std::sync::Arc;
    use std::thread::sleep;
    use std::time::{Duration, Instant};

    use ojas_core::{Backend, BackendId, OjasError, Tensor};
    use ojas_metal::MetalBackend;
    use tessl::dispatch::{dispatch_1d, set_f32, set_gpu_buf, set_gpu_buf_offset, set_u32, Binder};
    use tessl::gemm::{gemm_tn_splitk_par_f32, GemmOperands};
    use tessl::infer_trace::{self, Snapshot};
    use tessl::nn::scale_f32_inplace;
    use tessl::runtime::{mtl_size, GpuRuntime};
    use tessl::tensor::{GpuBuffer, Tensor as TT};
    use tessl::DType;

    use super::{rand, R};

    /// Op plus sync above this is a slow run.
    const SLOW_MS: f64 = 1.0;
    /// Floats each `ojas_check_finite` thread checks: the value `CHECK_PER_THREAD`
    /// has in `ojas-metal/src/device.rs`. The sequences below run the check as
    /// the backend does; the sweep shows the other counts.
    const CHECK_PER_THREAD: usize = 4;
    const WARMUP: usize = 20;

    fn tessl_err(detail: String) -> OjasError {
        OjasError::Backend {
            id: BackendId::Metal,
            detail,
        }
    }

    /// One timed run: wall milliseconds per phase and tessl counter deltas.
    #[derive(Clone, Copy, Default)]
    struct Sample {
        op: f64,
        sync: f64,
        wait: f64,
        gpu: Option<f64>,
        commits: u64,
        flushes: u64,
        cold: u64,
    }

    impl Sample {
        fn total(&self) -> f64 {
            self.op + self.sync
        }

        fn counters(mut self, a: Snapshot, b: Snapshot) -> Self {
            self.wait = b.sync_wait_us.saturating_sub(a.sync_wait_us) as f64 / 1e3;
            self.commits = b.commits.saturating_sub(a.commits);
            self.flushes = b.residency_flushes.saturating_sub(a.residency_flushes);
            self.cold = b.cold_allocs.saturating_sub(a.cold_allocs);
            self
        }
    }

    /// `"Device Utilization %"` from `ioreg`, as text; the probe's failure
    /// is printed rather than hidden.
    fn gpu_util() -> String {
        match std::process::Command::new("ioreg")
            .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
            .output()
        {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout);
                let tag = "\"Device Utilization %\"=";
                text.find(tag)
                    .and_then(|i| {
                        text[i + tag.len()..]
                            .split(|c: char| !c.is_ascii_digit())
                            .next()
                            .map(|d| format!("{d}%"))
                    })
                    .unwrap_or_else(|| "not reported".to_string())
            }
            Err(e) => format!("ioreg failed: {e}"),
        }
    }

    fn load() -> String {
        match std::process::Command::new("uptime").output() {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout).into_owned();
                s.split("load average")
                    .nth(1)
                    .map(|t| t.trim_start_matches(['s', ':', ' ']).trim().to_string())
                    .unwrap_or(s)
            }
            Err(e) => format!("uptime failed: {e}"),
        }
    }

    fn pct(sorted: &[f64], p: f64) -> f64 {
        sorted[((sorted.len() - 1) as f64 * p).round() as usize]
    }

    fn means(s: &[&Sample]) -> String {
        if s.is_empty() {
            return "-".to_string();
        }
        let n = s.len() as f64;
        let m = |f: &dyn Fn(&Sample) -> f64| s.iter().map(|x| f(x)).sum::<f64>() / n;
        let gpu: Vec<f64> = s.iter().filter_map(|x| x.gpu).collect();
        let gpu = if gpu.is_empty() {
            "-".to_string()
        } else {
            format!("{:.3}", gpu.iter().sum::<f64>() / gpu.len() as f64)
        };
        format!(
            "{:.3} / {:.3} / {:.3} / {gpu}",
            m(&|x| x.op),
            m(&|x| x.sync),
            m(&|x| x.wait)
        )
    }

    fn report(name: &str, s: &[Sample]) {
        let mut t: Vec<f64> = s.iter().map(Sample::total).collect();
        t.sort_by(f64::total_cmp);
        let slow: Vec<&Sample> = s.iter().filter(|x| x.total() > SLOW_MS).collect();
        let fast: Vec<&Sample> = s.iter().filter(|x| x.total() <= SLOW_MS).collect();
        let per = |f: &dyn Fn(&Sample) -> u64| {
            s.iter().map(|x| f(x) as f64).sum::<f64>() / s.len() as f64
        };
        println!(
            "| {name} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {}/{} | {} | {} | {:.2} / {:.2} / {:.2} |",
            s.len(),
            pct(&t, 0.0),
            pct(&t, 0.1),
            pct(&t, 0.5),
            pct(&t, 0.9),
            slow.len(),
            s.len(),
            means(&fast),
            means(&slow),
            per(&|x| x.commits),
            per(&|x| x.flushes),
            per(&|x| x.cold),
        );
    }

    /// `ops` one-element silus (0: sync alone), then one sync, after an idle
    /// `gap`. The outputs live until the sync returns.
    fn backend(be: &MetalBackend, x: &Tensor, n: usize, gap: u64, ops: usize) -> R<Vec<Sample>> {
        let mut out = Vec::with_capacity(n);
        for i in 0..WARMUP + n {
            if gap > 0 {
                sleep(Duration::from_millis(gap));
            }
            let a = infer_trace::snapshot();
            let t0 = Instant::now();
            let mut kept = Vec::with_capacity(ops);
            for _ in 0..ops {
                kept.push(be.silu_forward(x)?);
            }
            let t1 = Instant::now();
            be.sync()?;
            let t2 = Instant::now();
            let b = infer_trace::snapshot();
            drop(kept);
            if i >= WARMUP {
                out.push(
                    Sample {
                        op: (t1 - t0).as_secs_f64() * 1e3,
                        sync: (t2 - t1).as_secs_f64() * 1e3,
                        ..Sample::default()
                    }
                    .counters(a, b),
                );
            }
        }
        Ok(out)
    }

    /// One in-place `scale_f32_inplace` on one element and `synchronize`,
    /// on this thread, after an idle `gap`; `fresh` scales a newly
    /// allocated 4-byte buffer instead.
    fn direct(
        rt: &Arc<GpuRuntime>,
        n: usize,
        gap: u64,
        fresh: bool,
        ns_per_tick: f64,
    ) -> Result<Vec<Sample>, String> {
        let kept = rt.alloc_buffer(4)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..WARMUP + n {
            if gap > 0 {
                sleep(Duration::from_millis(gap));
            }
            let a = infer_trace::snapshot();
            let t0 = Instant::now();
            let new = if fresh {
                Some(rt.alloc_buffer(4)?)
            } else {
                None
            };
            scale_f32_inplace(rt, new.as_ref().unwrap_or(&kept), 1.0, 1)?;
            let t1 = Instant::now();
            rt.synchronize()?;
            let t2 = Instant::now();
            let b = infer_trace::snapshot();
            let gpu = rt
                .take_metal4_stamps()
                .map(|(s, e)| e.saturating_sub(s) as f64 * ns_per_tick / 1e6);
            drop(new);
            if i >= WARMUP {
                out.push(
                    Sample {
                        op: (t1 - t0).as_secs_f64() * 1e3,
                        sync: (t2 - t1).as_secs_f64() * 1e3,
                        gpu,
                        ..Sample::default()
                    }
                    .counters(a, b),
                );
            }
        }
        Ok(out)
    }

    /// Nanoseconds per GPU timestamp tick: the smallest wall-to-span ratio
    /// of five command buffers, each 16 passes over 256 MiB.
    fn calibrate(rt: &Arc<GpuRuntime>) -> Result<f64, String> {
        const N: u32 = 1 << 26;
        let big = rt.alloc_buffer(4 * N as usize)?;
        scale_f32_inplace(rt, &big, 1.0, N)?;
        rt.synchronize()?;
        rt.take_metal4_stamps();
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..16 {
                scale_f32_inplace(rt, &big, 1.0, N)?;
            }
            rt.synchronize()?;
            let wall = t0.elapsed().as_secs_f64() * 1e9;
            let (s, e) = rt
                .take_metal4_stamps()
                .ok_or("no Metal 4 timestamps: the counter heap is unavailable")?;
            if e <= s {
                return Err(format!("timestamps did not advance: {s} -> {e}"));
            }
            best = best.min(wall / (e - s) as f64);
        }
        Ok(best)
    }

    pub fn floor(be: &MetalBackend, n: usize) -> R<()> {
        println!(
            "start: GPU {}, load {}; slow = op + sync > {SLOW_MS} ms",
            gpu_util(),
            load()
        );
        infer_trace::set_enabled(true);
        let rt = GpuRuntime::new().map_err(tessl_err)?;
        rt.set_async_encode(true).map_err(tessl_err)?;
        let ns_per_tick = calibrate(&rt).map_err(tessl_err)?;
        println!("GPU timestamp: {ns_per_tick:.4} ns per tick (upper bound, see docs)");
        println!(
            "| scenario | runs | min ms | p10 | p50 | p90 | slow | fast: op / sync / wait / gpu ms | slow: op / sync / wait / gpu ms | per run: commits / residency flushes / cold allocs |"
        );
        println!("|---|---:|---:|---:|---:|---:|---:|---|---|---|");
        for (name, gap, fresh) in [
            ("tessl: scale 1 + synchronize", 0, false),
            ("tessl: same, 5 ms idle before", 5, false),
            ("tessl: fresh 4 B buffer + scale + synchronize", 0, true),
        ] {
            let s = direct(&rt, n, gap, fresh, ns_per_tick).map_err(tessl_err)?;
            report(name, &s);
        }
        drop(rt);
        let x = be.upload(&rand(&[1], 1, 1.0)?)?;
        for (name, gap, ops) in [
            ("backend: sync alone", 0, 0),
            ("backend: silu 1 + sync", 0, 1),
            ("backend: same, 1 ms idle before", 1, 1),
            ("backend: same, 5 ms idle before", 5, 1),
            ("backend: same, 20 ms idle before", 20, 1),
            ("backend: 8 x silu 1 + sync", 0, 8),
        ] {
            report(name, &backend(be, &x, n, gap, ops)?);
        }
        infer_trace::set_enabled(false);
        println!("end: GPU {}, load {}", gpu_util(), load());
        Ok(())
    }

    /// The `percall` decode shape: `sweep_decode_*16` in `bench/ojas_rows.rs`.
    const PC_B: usize = 16;
    const PC_H: usize = 12;
    const PC_D: usize = 64;
    const PC_T: usize = 1024;
    /// `CA_THREADS` in `ojas-metal/src/device.rs`.
    const PC_THREADS: usize = 1024;
    /// Most splits the split-walk table tries.
    const PC_MAX_SPLITS: usize = 32;

    /// `f` then one sync, `n` times: the calls' host time, the sync, tessl's
    /// event wait inside it, and the counters per run.
    fn calls(
        be: &MetalBackend,
        n: usize,
        mut f: impl FnMut() -> R<Vec<Tensor>>,
    ) -> R<Vec<(Sample, Snapshot)>> {
        let mut out = Vec::with_capacity(n);
        for i in 0..WARMUP + n {
            let a = infer_trace::snapshot();
            let t0 = Instant::now();
            let kept = f()?;
            let t1 = Instant::now();
            be.sync()?;
            let t2 = Instant::now();
            let b = infer_trace::snapshot();
            drop(kept);
            if i >= WARMUP {
                let s = Sample {
                    op: (t1 - t0).as_secs_f64() * 1e3,
                    sync: (t2 - t1).as_secs_f64() * 1e3,
                    ..Sample::default()
                }
                .counters(a, b);
                let d = Snapshot {
                    dispatches: b.dispatches.saturating_sub(a.dispatches),
                    barriers: b.barriers.saturating_sub(a.barriers),
                    ..Snapshot::default()
                };
                out.push((s, d));
            }
        }
        Ok(out)
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    fn report_calls(name: &str, ncalls: usize, s: &[(Sample, Snapshot)]) {
        let col = |f: &dyn Fn(&Sample) -> f64| median(s.iter().map(|(x, _)| f(x)).collect());
        let per = |f: &dyn Fn(&(Sample, Snapshot)) -> u64| {
            s.iter().map(|x| f(x) as f64).sum::<f64>() / s.len() as f64
        };
        let mut t: Vec<f64> = s.iter().map(|(x, _)| x.total()).collect();
        t.sort_by(f64::total_cmp);
        println!(
            "| {name} | {} | {:.3} | {:.3} | {:.3} | {:.1} | {:.3} | {:.3} | {:.2} / {:.2} / {:.2} / {:.2} / {:.2} |",
            s.len(),
            pct(&t, 0.0),
            pct(&t, 0.5),
            col(&|x| x.op),
            col(&|x| x.op) * 1e3 / ncalls as f64,
            col(&|x| x.sync),
            col(&|x| x.wait),
            per(&|x| x.0.commits),
            per(&|x| x.0.flushes),
            per(&|x| x.0.cold),
            per(&|x| x.1.dispatches),
            per(&|x| x.1.barriers),
        );
    }

    /// The `percall` group: where the extra cost of sending 16 decode
    /// requests as 16 calls instead of one batched call goes.
    pub fn percall(be: &MetalBackend, iters: usize) -> R<()> {
        let n = iters * 25;
        let (q_len, c_len) = (PC_H * PC_D, PC_T * PC_H * PC_D);
        let host = [
            super::values(PC_B * q_len, 171, 1.0),
            super::values(PC_B * c_len, 172, 1.0),
            super::values(PC_B * c_len, 173, 1.0),
        ];
        println!("start: GPU {}, load {}", gpu_util(), load());
        infer_trace::set_enabled(true);

        // 1. MetalBackend: host time of the calls vs the sync.
        let up =
            |v: &[f32], sh: &[usize]| be.upload(&Tensor::from_f32(v, sh, &super::host_budget())?);
        let qb = up(&host[0], &[PC_B, 1, PC_H, PC_D])?;
        let kb = up(&host[1], &[PC_B, PC_T, PC_H, PC_D])?;
        let vb = up(&host[2], &[PC_B, PC_T, PC_H, PC_D])?;
        let mut reqs = Vec::with_capacity(PC_B);
        for i in 0..PC_B {
            reqs.push((
                up(&host[0][i * q_len..(i + 1) * q_len], &[1, 1, PC_H, PC_D])?,
                up(&host[1][i * c_len..(i + 1) * c_len], &[1, PC_T, PC_H, PC_D])?,
                up(&host[2][i * c_len..(i + 1) * c_len], &[1, PC_T, PC_H, PC_D])?,
            ));
        }
        println!("\n`MetalBackend`, decode at H 12, D 64, 1024 cached positions; medians over {n} runs.\n");
        println!(
            "| scenario | runs | min ms | p50 ms | calls ms | per call µs | sync ms | event wait ms | per run: commits / residency flushes / cold allocs / dispatches / barriers |"
        );
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---|");
        let (q0, k0, v0) = &reqs[0];
        let s = calls(be, n, || {
            Ok(vec![be.cached_attention_forward(q0, k0, v0, PC_T, None)?])
        })?;
        report_calls("1 request, 1 call", 1, &s);
        let s = calls(be, n, || {
            Ok(vec![be.cached_attention_forward(&qb, &kb, &vb, PC_T, None)?])
        })?;
        report_calls("16 requests, 1 batched call", 1, &s);
        let s = calls(be, n, || {
            reqs.iter()
                .map(|(q, k, v)| be.cached_attention_forward(q, k, v, PC_T, None))
                .collect()
        })?;
        report_calls("16 requests, 16 calls", PC_B, &s);

        // 2. tessl directly with this crate's kernel: GPU span only.
        let rt = GpuRuntime::new().map_err(tessl_err)?;
        rt.set_async_encode(true).map_err(tessl_err)?;
        rt.add_metallib_bytes(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/ojas_per_head_gate.metallib"
        )))
        .map_err(tessl_err)?;
        let ns_per_tick = calibrate(&rt).map_err(tessl_err)?;
        let reps = n;
        let run = || -> Result<(), String> {
            let buf = |v: &[f32]| -> Result<GpuBuffer, String> {
                let b = rt.alloc_buffer(4 * v.len())?;
                b.write_f32_prefix(v);
                Ok(b)
            };
            let (q, k, v) = (buf(&host[0])?, buf(&host[1])?, buf(&host[2])?);
            let out = rt.alloc_buffer(4 * PC_B * q_len)?;
            let st = rt.alloc_buffer(4 * 8)?;
            let p = rt.pipeline("ojas_cached_attn")?;
            // Requests `first..first + batch` of the batched tensors, into
            // `dst` at the same rows.
            let one = |bd: &mut Binder<'_>,
                       first: usize,
                       batch: usize,
                       dst: &GpuBuffer,
                       dst_off: usize| {
                bd.set_pipeline(&p);
                set_gpu_buf_offset(bd, &q, 4 * first * q_len, 0);
                set_gpu_buf_offset(bd, &k, 4 * first * c_len, 1);
                set_gpu_buf_offset(bd, &v, 4 * first * c_len, 2);
                set_gpu_buf_offset(bd, dst, dst_off, 3);
                set_gpu_buf(bd, &st, 4);
                set_u32(bd, 1, 5);
                set_u32(bd, PC_H as u32, 6);
                set_u32(bd, PC_H as u32, 7);
                set_u32(bd, PC_D as u32, 8);
                set_u32(bd, PC_T as u32, 9);
                set_u32(bd, PC_T as u32, 10);
                set_f32(bd, 1.0 / (PC_D as f32).sqrt(), 11);
                set_u32(bd, 1, 12);
                set_u32(bd, PC_T as u32, 13);
                set_gpu_buf_offset(bd, dst, dst_off, 14);
                bd.dispatch(mtl_size(PC_H, batch, 1), mtl_size(PC_THREADS, 1, 1));
            };
            // Requests 0..batch with the cache walk in `splits` parts, then
            // the merge: the two dispatches `MetalBackend` records.
            let part = rt.alloc_buffer(4 * PC_B * PC_H * PC_MAX_SPLITS * (PC_D + 2))?;
            let merge = rt.pipeline("ojas_cached_attn_merge")?;
            let split = |batch: usize, splits: usize| -> Result<(), String> {
                rt.with_binder(|bd| {
                    bd.set_pipeline(&p);
                    set_gpu_buf(bd, &q, 0);
                    set_gpu_buf(bd, &k, 1);
                    set_gpu_buf(bd, &v, 2);
                    set_gpu_buf(bd, &out, 3);
                    set_gpu_buf(bd, &st, 4);
                    set_u32(bd, 1, 5);
                    set_u32(bd, PC_H as u32, 6);
                    set_u32(bd, PC_H as u32, 7);
                    set_u32(bd, PC_D as u32, 8);
                    set_u32(bd, PC_T as u32, 9);
                    set_u32(bd, PC_T as u32, 10);
                    set_f32(bd, 1.0 / (PC_D as f32).sqrt(), 11);
                    set_u32(bd, splits as u32, 12);
                    set_u32(bd, PC_T.div_ceil(splits) as u32, 13);
                    set_gpu_buf(bd, &part, 14);
                    bd.dispatch(
                        mtl_size(PC_H * splits, batch, 1),
                        mtl_size(PC_THREADS, 1, 1),
                    );
                    Ok(())
                })?;
                if splits > 1 {
                    rt.with_binder(|bd| {
                        bd.set_pipeline(&merge);
                        set_gpu_buf(bd, &part, 0);
                        set_gpu_buf(bd, &out, 1);
                        set_gpu_buf(bd, &st, 2);
                        set_u32(bd, PC_D as u32, 3);
                        set_u32(bd, splits as u32, 4);
                        bd.dispatch(mtl_size(PC_H, batch, 1), mtl_size(PC_D, 1, 1));
                        Ok(())
                    })?;
                }
                Ok(())
            };
            println!("\nGPU span of one command buffer, tessl directly, {reps} buffers.\n");
            println!("| command buffer holds | min µs | median µs |");
            println!("|---|---:|---:|");
            let row = |name: &str, t: (f64, f64)| println!("| {name} | {:.1} | {:.1} |", t.0, t.1);
            row(
                "1 request, 1 dispatch",
                gpu_span(&rt, reps, ns_per_tick, || {
                    rt.with_binder(|bd| {
                        one(bd, 0, 1, &out, 0);
                        Ok(())
                    })
                })?,
            );
            row(
                "16 requests, 1 batched dispatch",
                gpu_span(&rt, reps, ns_per_tick, || {
                    rt.with_binder(|bd| {
                        one(bd, 0, PC_B, &out, 0);
                        Ok(())
                    })
                })?,
            );
            row(
                "16 dispatches, a barrier after each (as `MetalBackend` records them)",
                gpu_span(&rt, reps, ns_per_tick, || {
                    for i in 0..PC_B {
                        rt.with_binder(|bd| {
                            one(bd, i, 1, &out, 4 * i * q_len);
                            Ok(())
                        })?;
                    }
                    Ok(())
                })?,
            );
            row(
                "16 dispatches, no barrier between them",
                gpu_span(&rt, reps, ns_per_tick, || {
                    rt.with_binder_barriers(Some(true), |bd| {
                        for i in 0..PC_B {
                            one(bd, i, 1, &out, 4 * i * q_len);
                        }
                        Ok(())
                    })
                })?,
            );

            // 2b. The split cache walk: GPU span by request count and splits.
            println!("\nSplit cache walk (`ojas_cached_attn` in parts, then `ojas_cached_attn_merge`), GPU span, {reps} buffers.\n");
            println!("| requests | splits | threadgroups | min µs | median µs | GB/s at min |");
            println!("|---:|---:|---:|---:|---:|---:|");
            for batch in [1usize, 2, 4, 16] {
                for splits in [1usize, 2, 4, 8, 16, PC_MAX_SPLITS] {
                    let t = gpu_span(&rt, reps, ns_per_tick, || split(batch, splits))?;
                    let bytes = (batch * 2 * c_len * 4) as f64;
                    println!(
                        "| {batch} | {splits} | {} | {:.1} | {:.1} | {:.0} |",
                        PC_H * splits * batch,
                        t.0,
                        t.1,
                        bytes / (t.0 * 1e3)
                    );
                }
            }

            // 3. Host cost of recording 16 dispatches: into one reused output
            // buffer, or each into a fresh one as `MetalBackend` allocates.
            println!("\nHost time to record 16 dispatches (no wait inside), tessl directly; medians over {reps} runs.\n");
            println!("| recording | µs per dispatch | per run: residency flushes / cold allocs |");
            println!("|---|---:|---|");
            for fresh in [false, true] {
                let mut us = Vec::with_capacity(reps);
                let (mut fl, mut co) = (0u64, 0u64);
                for i in 0..WARMUP + reps {
                    let a = infer_trace::snapshot();
                    let t0 = Instant::now();
                    let mut kept = Vec::with_capacity(PC_B);
                    for r in 0..PC_B {
                        if fresh {
                            let dst = rt.alloc_buffer(4 * q_len)?;
                            rt.with_binder(|bd| {
                                one(bd, r, 1, &dst, 0);
                                Ok(())
                            })?;
                            kept.push(dst);
                        } else {
                            rt.with_binder(|bd| {
                                one(bd, r, 1, &out, 4 * r * q_len);
                                Ok(())
                            })?;
                        }
                    }
                    let dt = t0.elapsed().as_secs_f64() * 1e6;
                    let b = infer_trace::snapshot();
                    rt.synchronize()?;
                    rt.take_metal4_stamps();
                    drop(kept);
                    if i >= WARMUP {
                        us.push(dt / PC_B as f64);
                        fl += b.residency_flushes.saturating_sub(a.residency_flushes);
                        co += b.cold_allocs.saturating_sub(a.cold_allocs);
                    }
                }
                println!(
                    "| {} | {:.1} | {:.2} / {:.2} |",
                    if fresh {
                        "each into a fresh output (as `MetalBackend`)"
                    } else {
                        "all into one reused output"
                    },
                    median(us),
                    fl as f64 / reps as f64,
                    co as f64 / reps as f64
                );
            }
            Ok(())
        };
        run().map_err(tessl_err)?;
        infer_trace::set_enabled(false);
        println!("\nend: GPU {}, load {}", gpu_util(), load());
        Ok(())
    }

    /// Encode one ojas kernel over `n` threads, as `MetalBackend` does.
    fn k(
        rt: &Arc<GpuRuntime>,
        name: &str,
        n: usize,
        f: impl FnOnce(&mut Binder<'_>),
    ) -> Result<(), String> {
        let p = rt.pipeline(name)?;
        dispatch_1d(rt, &p, n, f)
    }

    /// `(min, median)` GPU span in µs of `reps` command buffers, each
    /// holding what `encode` records and nothing else.
    fn gpu_span(
        rt: &Arc<GpuRuntime>,
        reps: usize,
        ns_per_tick: f64,
        mut encode: impl FnMut() -> Result<(), String>,
    ) -> Result<(f64, f64), String> {
        let mut us = Vec::with_capacity(reps);
        for i in 0..reps + 3 {
            encode()?;
            rt.synchronize()?;
            let (s, e) = rt
                .take_metal4_stamps()
                .ok_or("no Metal 4 timestamps: the counter heap is unavailable")?;
            if i >= 3 {
                us.push(e.saturating_sub(s) as f64 * ns_per_tick / 1e3);
            }
        }
        us.sort_by(f64::total_cmp);
        Ok((us[0], us[us.len() / 2]))
    }

    /// The `kernels` group: GPU time of each elementwise kernel `MetalBackend`
    /// runs, alone and as the op's full sequence (finite checks included),
    /// at the paired benchmark's shapes, against a plain read-write pass.
    pub fn kernels(iters: usize) -> R<()> {
        const N_FF: usize = 4096 * 2048;
        const N_DM: usize = 4096 * 768;
        println!("start: GPU {}, load {}", gpu_util(), load());
        let rt = GpuRuntime::new().map_err(tessl_err)?;
        rt.set_async_encode(true).map_err(tessl_err)?;
        rt.add_metallib_bytes(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/ojas_per_head_gate.metallib"
        )))
        .map_err(tessl_err)?;
        let ns_per_tick = calibrate(&rt).map_err(tessl_err)?;
        println!("GPU timestamp: {ns_per_tick:.4} ns per tick (upper bound, see docs)");
        let reps = iters.max(10);
        let run = || -> Result<(), String> {
            let buf = |n: usize| rt.alloc_buffer(4 * n);
            let (a, b, g, y, y2) = (buf(N_FF)?, buf(N_FF)?, buf(N_FF)?, buf(N_FF)?, buf(N_FF)?);
            let st = rt.alloc_buffer(4 * 8)?;
            let n = N_FF as u32;
            // Floats per thread: `per`, as `fn check` in `device.rs` dispatches it.
            let check_per = |v: &GpuBuffer, n: usize, per: usize| {
                k(&rt, "ojas_check_finite", n.div_ceil(per), |bd| {
                    set_gpu_buf(bd, v, 0);
                    set_gpu_buf(bd, &st, 1);
                    set_u32(bd, n as u32, 2);
                    set_u32(bd, 0, 3);
                })
            };
            let check = |v: &GpuBuffer, n: usize| check_per(v, n, CHECK_PER_THREAD);
            let silu = || {
                k(&rt, "ojas_silu_fwd", N_FF, |bd| {
                    set_gpu_buf(bd, &a, 0);
                    set_gpu_buf(bd, &y, 1);
                    set_u32(bd, n, 2);
                    set_gpu_buf(bd, &st, 3);
                })
            };
            let silu_bwd = || {
                k(&rt, "ojas_silu_bwd", N_FF, |bd| {
                    set_gpu_buf(bd, &a, 0);
                    set_gpu_buf(bd, &g, 1);
                    set_gpu_buf(bd, &y, 2);
                    set_u32(bd, n, 3);
                    set_gpu_buf(bd, &st, 4);
                })
            };
            let mul = || {
                k(&rt, "ojas_mul_fwd", N_FF, |bd| {
                    set_gpu_buf(bd, &a, 0);
                    set_gpu_buf(bd, &b, 1);
                    set_gpu_buf(bd, &y, 2);
                    set_u32(bd, n, 3);
                    set_gpu_buf(bd, &st, 4);
                })
            };
            let mul_bwd = || {
                k(&rt, "ojas_mul_bwd", N_FF, |bd| {
                    set_gpu_buf(bd, &a, 0);
                    set_gpu_buf(bd, &b, 1);
                    set_gpu_buf(bd, &g, 2);
                    set_gpu_buf(bd, &y, 3);
                    set_gpu_buf(bd, &y2, 4);
                    set_u32(bd, n, 5);
                    set_gpu_buf(bd, &st, 6);
                })
            };
            println!(
                "| kernel or sequence | elements | f32 passes | min µs | median µs | GB/s at min |"
            );
            println!("|---|---:|---:|---:|---:|---:|");
            let row = |name: &str, elems: usize, passes: f64, t: (f64, f64)| {
                let bytes = elems as f64 * 4.0 * passes;
                println!(
                    "| {name} | {elems} | {passes} | {:.1} | {:.1} | {:.0} |",
                    t.0,
                    t.1,
                    bytes / (t.0 * 1e3)
                );
            };
            row(
                "tessl scale_f32_inplace (reference read + write)",
                N_FF,
                2.0,
                gpu_span(&rt, reps, ns_per_tick, || {
                    scale_f32_inplace(&rt, &a, 1.0, n)
                })?,
            );
            for per in [1usize, 2, 4, 8, 16, 32, 64] {
                let name = format!("ojas_check_finite, {per} per thread");
                row(
                    &name,
                    N_FF,
                    1.0,
                    gpu_span(&rt, reps, ns_per_tick, || check_per(&a, N_FF, per))?,
                );
            }
            row(
                "ojas_check_finite [4096, 768]",
                N_DM,
                1.0,
                gpu_span(&rt, reps, ns_per_tick, || check(&a, N_DM))?,
            );
            // Each op is one kernel that checks its own inputs and outputs; the
            // passes column is what it must read and write.
            let add = || {
                k(&rt, "ojas_add_fwd", N_FF, |bd| {
                    set_gpu_buf(bd, &a, 0);
                    set_gpu_buf(bd, &b, 1);
                    set_gpu_buf(bd, &y, 2);
                    set_u32(bd, n, 3);
                    set_gpu_buf(bd, &st, 4);
                })
            };
            let add_bwd = || {
                k(&rt, "ojas_add_bwd", N_FF, |bd| {
                    set_gpu_buf(bd, &a, 0);
                    set_gpu_buf(bd, &b, 1);
                    set_gpu_buf(bd, &g, 2);
                    set_gpu_buf(bd, &y, 3);
                    set_gpu_buf(bd, &y2, 4);
                    set_u32(bd, n, 5);
                    set_gpu_buf(bd, &st, 6);
                })
            };
            row(
                "ojas_silu_fwd (checks inside)",
                N_FF,
                2.0,
                gpu_span(&rt, reps, ns_per_tick, silu)?,
            );
            row(
                "ojas_silu_bwd (checks inside)",
                N_FF,
                3.0,
                gpu_span(&rt, reps, ns_per_tick, silu_bwd)?,
            );
            row(
                "ojas_mul_fwd (checks inside)",
                N_FF,
                3.0,
                gpu_span(&rt, reps, ns_per_tick, mul)?,
            );
            row(
                "ojas_mul_bwd (checks inside)",
                N_FF,
                5.0,
                gpu_span(&rt, reps, ns_per_tick, mul_bwd)?,
            );
            row(
                "ojas_add_fwd (checks inside)",
                N_FF,
                3.0,
                gpu_span(&rt, reps, ns_per_tick, add)?,
            );
            row(
                "ojas_add_bwd (checks inside)",
                N_FF,
                5.0,
                gpu_span(&rt, reps, ns_per_tick, add_bwd)?,
            );
            Ok(())
        };
        run().map_err(tessl_err)?;
        println!("end: GPU {}, load {}", gpu_util(), load());
        Ok(())
    }

    /// The `gate` group: GPU time of each dispatch `MetalBackend` records for
    /// `per_head_sigmoid_gate_backward` (`fn gate` in `device.rs`), at the
    /// paired benchmark's shape (4096 rows, 12 heads of 64, d_model 768):
    /// the `pre` GEMM, the gate kernel, the bias sum, the two gradient GEMMs
    /// and the standalone finite checks, each alone and then all in one
    /// command buffer, in the backend's order.
    pub fn gate(iters: usize) -> R<()> {
        const R_: usize = 4096;
        const H: usize = 12;
        const D: usize = 64;
        const DM: usize = 768;
        const PLANE: usize = R_ * H * D;
        println!("start: GPU {}, load {}", gpu_util(), load());
        let rt = GpuRuntime::new().map_err(tessl_err)?;
        rt.set_async_encode(true).map_err(tessl_err)?;
        rt.add_metallib_bytes(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/ojas_per_head_gate.metallib"
        )))
        .map_err(tessl_err)?;
        let ns_per_tick = calibrate(&rt).map_err(tessl_err)?;
        println!("GPU timestamp: {ns_per_tick:.4} ns per tick (upper bound, see docs)");
        let reps = iters.max(10);
        let run = || -> Result<(), String> {
            let buf = |n: usize| rt.alloc_buffer(4 * n);
            let mat = |b: &GpuBuffer, rows: usize, cols: usize| {
                TT::from_buffer(&rt, b.clone(), &[rows, cols], DType::F32, 0)
            };
            let (x, w, bias) = (buf(R_ * DM)?, buf(H * DM)?, buf(H)?);
            let (attn, dy, pre) = (buf(PLANE)?, buf(PLANE)?, buf(R_ * H)?);
            let (d_attn, d_pre, d_bias) = (buf(PLANE)?, buf(R_ * H)?, buf(H)?);
            let (gx, gw) = (buf(R_ * DM)?, buf(H * DM)?);
            let st = rt.alloc_buffer(4 * 8)?;
            let (x_t, w_t) = (mat(&x, R_, DM)?, mat(&w, H, DM)?);
            let (pre_t, dp_t) = (mat(&pre, R_, H)?, mat(&d_pre, R_, H)?);
            let (gx_t, gw_t) = (mat(&gx, R_, DM)?, mat(&gw, H, DM)?);
            let (rows, heads, dh) = (R_ as u32, H as u32, D as u32);
            let (units, plane) = ((R_ * H) as u32, PLANE as u32);
            let check = |v: &GpuBuffer, n: usize| {
                k(
                    &rt,
                    "ojas_check_finite",
                    n.div_ceil(CHECK_PER_THREAD),
                    |bd| {
                        set_gpu_buf(bd, v, 0);
                        set_gpu_buf(bd, &st, 1);
                        set_u32(bd, n as u32, 2);
                        set_u32(bd, 0, 3);
                    },
                )
            };
            let pre_gemm = || GemmOperands::ExactF32.nt(&x_t, &w_t, &pre_t);
            let gate_bwd = || {
                k(&rt, "ojas_per_head_gate_bwd", R_ * H, |bd| {
                    set_gpu_buf(bd, &attn, 0);
                    set_gpu_buf(bd, &pre, 1);
                    set_gpu_buf(bd, &bias, 2);
                    set_gpu_buf(bd, &dy, 3);
                    set_gpu_buf(bd, &d_attn, 4);
                    set_gpu_buf(bd, &d_pre, 5);
                    set_u32(bd, rows, 6);
                    set_u32(bd, heads, 7);
                    set_u32(bd, dh, 8);
                    set_u32(bd, plane, 9);
                    set_u32(bd, units, 10);
                    set_u32(bd, heads, 11);
                    set_gpu_buf(bd, &st, 12);
                })
            };
            // One SIMD-group per head, as `gate_dbias_threads` dispatches it.
            let dbias = || {
                k(&rt, "ojas_per_head_gate_dbias", H * 32, |bd| {
                    set_gpu_buf(bd, &d_pre, 0);
                    set_gpu_buf(bd, &d_bias, 1);
                    set_u32(bd, rows, 2);
                    set_u32(bd, heads, 3);
                    set_u32(bd, units, 4);
                    set_u32(bd, heads, 5);
                    set_gpu_buf(bd, &st, 6);
                })
            };
            let gx_gemm = || GemmOperands::ExactF32.nn(&dp_t, &w_t, &gx_t);
            let gw_gemm = || GemmOperands::ExactF32.tn(&dp_t, &x_t, &gw_t);
            // The backend's standalone checks: `x` and `w` in, `gx` and `gw`
            // out (only tessl GEMMs touch them). The gate kernels check the
            // rest themselves.
            let checks = || {
                check(&x, R_ * DM)?;
                check(&w, H * DM)?;
                check(&gx, R_ * DM)?;
                check(&gw, H * DM)
            };
            println!("| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |");
            println!("|---|---:|---:|---:|---:|");
            let mb = |elems: usize| elems as f64 * 4.0 / 1e6;
            let row = |name: &str, moved: f64, t: (f64, f64)| {
                println!(
                    "| {name} | {moved:.1} | {:.1} | {:.1} | {:.0} |",
                    t.0,
                    t.1,
                    moved * 1e6 / (t.0 * 1e3)
                );
            };
            let span = |f: &dyn Fn() -> Result<(), String>| gpu_span(&rt, reps, ns_per_tick, f);
            row(
                "tessl scale_f32_inplace over attn (reference read + write)",
                mb(2 * PLANE),
                span(&|| scale_f32_inplace(&rt, &attn, 1.0, plane))?,
            );
            row(
                "nt pre = x · w^T (4096, 12, 768)",
                mb(R_ * DM + H * DM + R_ * H),
                span(&pre_gemm)?,
            );
            row(
                "ojas_per_head_gate_bwd",
                mb(3 * PLANE + 2 * R_ * H),
                span(&gate_bwd)?,
            );
            row("ojas_per_head_gate_dbias", mb(R_ * H + H), span(&dbias)?);
            row(
                "nn gx = d_pre · w (4096, 768, 12)",
                mb(R_ * H + H * DM + R_ * DM),
                span(&gx_gemm)?,
            );
            row(
                "tn gw = d_pre^T · x (12, 768, 4096)",
                mb(R_ * H + R_ * DM + H * DM),
                span(&gw_gemm)?,
            );
            row(
                "four standalone ojas_check_finite passes",
                mb(2 * R_ * DM + 2 * H * DM),
                span(&checks)?,
            );
            let all = || {
                pre_gemm()?;
                gate_bwd()?;
                dbias()?;
                gx_gemm()?;
                gw_gemm()?;
                checks()
            };
            row(
                "whole backward, one command buffer",
                mb(3 * R_ * DM + 4 * PLANE),
                span(&all)?,
            );
            Ok(())
        };
        run().map_err(tessl_err)?;
        println!("end: GPU {}, load {}", gpu_util(), load());
        Ok(())
    }

    /// The `tn` group: tessl's exact-f32 TN (`C [M, N] = A [K, M]^T · B [K,
    /// N]`) for a C of few tiles over a long K, the weight-gradient shape
    /// whose single dispatch has a few threadgroups each walking all of K.
    /// Per shape: that single dispatch (`matmul2d_tensorops_tn_f32`, encoded
    /// here as `gemm_tn_f32` does), what `GemmOperands::ExactF32.tn` routes
    /// to, and `gemm_tn_splitk_par_f32` at several partition widths. One GEMM
    /// per command buffer.
    pub fn tn(iters: usize) -> R<()> {
        println!("start: GPU {}, load {}", gpu_util(), load());
        let rt = GpuRuntime::new().map_err(tessl_err)?;
        rt.set_async_encode(true).map_err(tessl_err)?;
        let ns_per_tick = calibrate(&rt).map_err(tessl_err)?;
        println!("GPU timestamp: {ns_per_tick:.4} ns per tick (upper bound, see docs)");
        let reps = iters.max(10);
        println!("| shape (M, N, K) | tiles | path | min µs | median µs | TFLOP/s at min |");
        println!("|---|---:|---|---:|---:|---:|");
        let run = || -> Result<(), String> {
            let mat = |rows: usize, cols: usize| -> Result<TT, String> {
                TT::from_buffer(
                    &rt,
                    rt.alloc_buffer(4 * rows * cols)?,
                    &[rows, cols],
                    DType::F32,
                    0,
                )
            };
            // The last five are shapes the sequential TN split-K lane took
            // (`prefer_tn_splitk`) before the parallel lane replaced it for
            // f32; three are its own tests' shapes.
            let shapes: [(usize, usize, usize); 12] = [
                (12, 768, 4096),
                (12, 768, 1024),
                (12, 768, 16384),
                (40, 520, 9000),
                (64, 1024, 4096),
                (128, 768, 4096),
                (96, 2048, 4096),
                (64, 64, 4096),
                (256, 256, 4096),
                (128, 128, 4096),
                (128, 384, 4096),
                (128, 128, 2048),
            ];
            let single = rt.pipeline("matmul2d_tensorops_tn_f32")?;
            for (m, n, k) in shapes {
                let (a, b, c) = (mat(k, m)?, mat(k, n)?, mat(m, n)?);
                let (tiles_n, tiles_m) = (n.div_ceil(32), m.div_ceil(32));
                let tiles = tiles_n * tiles_m;
                let flops = 2.0 * (m * n * k) as f64;
                let report = |path: &str, t: (f64, f64)| {
                    println!(
                        "| ({m}, {n}, {k}) | {tiles} | {path} | {:.1} | {:.1} | {:.2} |",
                        t.0,
                        t.1,
                        flops / (t.0 * 1e-6) / 1e12
                    );
                };
                // One simdgroup per threadgroup: `dispatch_1d` gives each
                // threadgroup `threadExecutionWidth` threads.
                report(
                    "single dispatch",
                    gpu_span(&rt, reps, ns_per_tick, || {
                        dispatch_1d(&rt, &single, tiles * 32, |bd| {
                            set_gpu_buf(bd, &a.buffer, 0);
                            set_gpu_buf(bd, &b.buffer, 1);
                            set_gpu_buf(bd, &c.buffer, 2);
                            set_u32(bd, m as u32, 3);
                            set_u32(bd, n as u32, 4);
                            set_u32(bd, k as u32, 5);
                            set_u32(bd, tiles_n as u32, 6);
                            set_u32(bd, tiles_m as u32, 7);
                            set_u32(bd, 0, 8);
                        })
                    })?,
                );
                report(
                    "routed",
                    gpu_span(&rt, reps, ns_per_tick, || {
                        GemmOperands::ExactF32.tn(&a, &b, &c)
                    })?,
                );
                for k_tile in [128usize, 256, 512, 1024, 2048] {
                    if k_tile >= k {
                        continue;
                    }
                    // tessl refuses a width whose partial sums exceed its
                    // scratch cap (`TN_PAR_MAX_SCRATCH`, 1 << 22 floats).
                    let parts = k.div_ceil(k_tile);
                    if parts * (m * n).div_ceil(4) * 4 > 1 << 22 {
                        println!(
                            "| ({m}, {n}, {k}) | {tiles} | parallel, k_tile {k_tile} ({parts} parts) | refused: scratch over the cap | | |"
                        );
                        continue;
                    }
                    report(
                        &format!("parallel, k_tile {k_tile} ({} parts)", k.div_ceil(k_tile)),
                        gpu_span(&rt, reps, ns_per_tick, || {
                            gemm_tn_splitk_par_f32(&a, &b, &c, k_tile)
                        })?,
                    );
                }
            }
            Ok(())
        };
        run().map_err(tessl_err)?;
        println!("end: GPU {}, load {}", gpu_util(), load());
        Ok(())
    }

    /// The `gemm` group: tessl's exact-f32 GEMM, as `MetalBackend` calls it,
    /// at the paired benchmark's linear and LM-head shapes, one GEMM per
    /// command buffer. `nt` is a linear forward (`x [M, K] · w [N, K]^T`),
    /// `nn` the input gradient (`g [M, N] · w [N, K]`), `tn` the weight
    /// gradient (`g [M, N]^T · x [M, K]`); each row prints its own GEMM's
    /// (M, N, K). The LM head is also timed split into vocabulary chunks,
    /// summed, to see whether its shape or its size sets the rate, and at
    /// full width with as few as 32 rows (one row of 32x32 tiles, which
    /// reads `w` exactly once).
    pub fn gemm(iters: usize) -> R<()> {
        println!("start: GPU {}, load {}", gpu_util(), load());
        let rt = GpuRuntime::new().map_err(tessl_err)?;
        rt.set_async_encode(true).map_err(tessl_err)?;
        let ns_per_tick = calibrate(&rt).map_err(tessl_err)?;
        println!("GPU timestamp: {ns_per_tick:.4} ns per tick (upper bound, see docs)");
        let reps = iters.max(5);
        println!("| kernel | M | N | K | min ms | median ms | TFLOP/s at min |");
        println!("|---|---:|---:|---:|---:|---:|---:|");
        let run = || -> Result<(), String> {
            let mat = |rows: usize, cols: usize| -> Result<TT, String> {
                TT::from_buffer(
                    &rt,
                    rt.alloc_buffer(4 * rows * cols)?,
                    &[rows, cols],
                    DType::F32,
                    0,
                )
            };
            // (label, M, N, K) for C [M, N].
            let shapes: [(&str, usize, usize, usize); 6] = [
                ("qkv / up width", 4096, 2304, 768),
                ("mlp up", 4096, 2048, 768),
                ("mlp down", 4096, 768, 2048),
                ("vocab chunk 8192", 4096, 8192, 768),
                ("vocab chunk 16384", 4096, 16384, 768),
                ("lm head", 4096, 50304, 768),
            ];
            for (label, m, n, k) in shapes {
                let flops = 2.0 * (m * n * k) as f64;
                // Each row gives the GEMM's own (M, N, K) for its C: the
                // input gradient's C is [m, k] with n reduced, the weight
                // gradient's is [n, k] with m reduced.
                let report = |kind: &str, (gm, gn, gk): (usize, usize, usize), t: (f64, f64)| {
                    println!(
                        "| {kind} {label} | {gm} | {gn} | {gk} | {:.3} | {:.3} | {:.2} |",
                        t.0 / 1e3,
                        t.1 / 1e3,
                        flops / (t.0 * 1e-6) / 1e12
                    );
                };
                let (x, w, c) = (mat(m, k)?, mat(n, k)?, mat(m, n)?);
                report(
                    "nt",
                    (m, n, k),
                    gpu_span(&rt, reps, ns_per_tick, || {
                        GemmOperands::ExactF32.nt(&x, &w, &c)
                    })?,
                );
                let gx = mat(m, k)?;
                report(
                    "nn",
                    (m, k, n),
                    gpu_span(&rt, reps, ns_per_tick, || {
                        GemmOperands::ExactF32.nn(&c, &w, &gx)
                    })?,
                );
                let gw = mat(n, k)?;
                report(
                    "tn",
                    (n, k, m),
                    gpu_span(&rt, reps, ns_per_tick, || {
                        GemmOperands::ExactF32.tn(&c, &x, &gw)
                    })?,
                );
            }
            // Whether the LM head's rate is set by its width: the same
            // `x · w^T` over 50304 columns, done as column chunks (each a
            // contiguous slice of `w`'s rows into its own C), all in one
            // command buffer; and at full width with fewer rows.
            let (m, n, k) = (4096usize, 50304usize, 768usize);
            let x = mat(m, k)?;
            let wbuf = rt.alloc_buffer(4 * n * k)?;
            for chunk in [1024usize, 2048, 4096] {
                let pieces: Vec<(TT, TT)> = (0..n)
                    .step_by(chunk)
                    .map(|c0| {
                        let cc = chunk.min(n - c0);
                        let w =
                            TT::from_buffer(&rt, wbuf.clone(), &[cc, k], DType::F32, 4 * c0 * k)?;
                        Ok((w, mat(m, cc)?))
                    })
                    .collect::<Result<_, String>>()?;
                let t = gpu_span(&rt, reps, ns_per_tick, || {
                    for (w, c) in &pieces {
                        GemmOperands::ExactF32.nt(&x, w, c)?;
                    }
                    Ok(())
                })?;
                println!(
                    "| nt lm head in {chunk}-column chunks ({} GEMMs) | {m} | {n} | {k} | {:.3} | {:.3} | {:.2} |",
                    pieces.len(),
                    t.0 / 1e3,
                    t.1 / 1e3,
                    2.0 * (m * n * k) as f64 / (t.0 * 1e-6) / 1e12
                );
            }
            let w = TT::from_buffer(&rt, wbuf.clone(), &[n, k], DType::F32, 0)?;
            for rows in [32usize, 128, 512, 1024, 2048] {
                let (xr, c) = (mat(rows, k)?, mat(rows, n)?);
                let t = gpu_span(&rt, reps, ns_per_tick, || {
                    GemmOperands::ExactF32.nt(&xr, &w, &c)
                })?;
                println!(
                    "| nt lm head, {rows} rows | {rows} | {n} | {k} | {:.3} | {:.3} | {:.2} |",
                    t.0 / 1e3,
                    t.1 / 1e3,
                    2.0 * (rows * n * k) as f64 / (t.0 * 1e-6) / 1e12
                );
            }
            Ok(())
        };
        run().map_err(tessl_err)?;
        println!("end: GPU {}, load {}", gpu_util(), load());
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
mod probe {
    use ojas_core::OjasError;
    use ojas_metal::MetalBackend;

    fn unsupported() -> OjasError {
        OjasError::Unsupported {
            op: "metal_bench probe",
            detail: "the floor, kernels, gemm, gate and tn groups read tessl, which is macOS only"
                .to_string(),
        }
    }

    pub fn floor(_be: &MetalBackend, _n: usize) -> super::R<()> {
        Err(unsupported())
    }

    pub fn kernels(_iters: usize) -> super::R<()> {
        Err(unsupported())
    }

    pub fn percall(_be: &MetalBackend, _iters: usize) -> super::R<()> {
        Err(unsupported())
    }

    pub fn gemm(_iters: usize) -> super::R<()> {
        Err(unsupported())
    }

    pub fn gate(_iters: usize) -> super::R<()> {
        Err(unsupported())
    }

    pub fn tn(_iters: usize) -> super::R<()> {
        Err(unsupported())
    }
}
