//! Interleaved measurements behind the GPU step-throughput decisions
//! (task ft-2c183b7103c95088ae62ea0da43aebc3). Generic over
//! `ojas_core::Backend`, compiled with `ojas_rows.rs` into
//! `ojas-metal/examples/step_probe.rs` and `ojas-wgpu/examples/step_probe.rs`.
//!
//! Every probe runs its variants in the same process, `rounds` rounds of
//! `iters` timed iterations after `warmup`, alternating which variant goes
//! first each round. Each iteration ends with `Backend::sync`, so a time is
//! device-complete. One JSON line per (probe, variant, round) goes to
//! `OJAS_PROBE_OUT` with that round's min and median;
//! `bench/results/2026-10-08-gpu-step-ab/scripts/probe_agg.py`
//! reports, per variant, the min of the round minimums and the spread
//! (max / min of the round minimums).
//!
//! Probes (`OJAS_PROBES`, comma-separated prefixes; default all):
//! - `adamw`: the 170 nanolab tensors as 170 `adamw_step` calls, against
//!   one tensor of the same 123,699,612 values: what any multi-tensor
//!   batching of the calls could save at most.
//! - `muon`: 48 `[768, 768]` Muon steps recorded before one sync, and the
//!   NS5 GEMM work alone, as 48 per-matrix GEMMs against one stacked GEMM
//!   of the same FLOPs: the headroom for batched Muon GEMMs.
//! - `muon_split`: one Muon step at `[768, 768]` and `[2048, 768]` against
//!   its fifteen NS5 GEMMs alone: the GEMM share of a step.
//! - `flush`: the nanolab block forward + backward (53 ops, one sync) at
//!   each flush threshold the backend accepts ([`Knobs::flush_values`]).
//! - `acc`: `accumulate_grad` into a sole-owner accumulator (in place) and
//!   into a shared one (a new tensor), at `[4096, 768]` and `[50304, 768]`.
//! - `parts`: the open rows of `docs/bench-gpu-vs-torch.md` against their
//!   parts and against a bandwidth floor: `gate_fwd` / `gate_bwd` against
//!   their GEMMs alone; `cross_entropy_bwd` `[4096, 50304]` and
//!   `accumulate_grad` `[50304, 768]` against `mul_forward` over the same
//!   shape (two reads and a write, the traffic an add needs); the causal
//!   SDPA forward at d64 alone. (`block_fwd`'s parts are the paired
//!   bench's own rows at the block's shapes.)
//! - `link`: the Metal channel: an empty round trip, a one-value op, and a
//!   decode-sized `cached_attention_forward`, each recorded 256 times
//!   before one sync (host time per call), and the decode call with a sync
//!   after each.

#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::time::Instant;

use ojas_core::{AdamWConfig, Backend, Budget, MuonNs5Config, OjasError, Tensor};

use crate::ojas_rows::{
    block_backward, block_forward, block_setup, gen, gen_targets, param_shapes, B, D, DM, H, N, T,
    V,
};

/// What the backends differ in.
pub trait Knobs {
    /// Flush thresholds to sweep (wgpu's `set_flush_at`); empty if fixed.
    fn flush_values(&self) -> Vec<usize>;
    fn set_flush(&self, dispatches: usize);
    /// One channel round trip that records no GPU work (Metal `memory()`).
    fn round_trip(&self) -> Option<Result<(), OjasError>>;
}

pub struct Probe<'a, Bk: Backend + Knobs> {
    pub be: &'a Bk,
    pub host: Budget,
    pub rounds: usize,
    pub warmup: usize,
    pub iters: usize,
    pub runtime: String,
    pub filter: Vec<String>,
    pub out: fs::File,
}

type R<T> = Result<T, OjasError>;

/// One timed iteration of a probe variant ([`Probe::interleave`]).
type Step<'a, Bk, S> = dyn FnMut(&Bk, &mut S) -> R<()> + 'a;

impl<Bk: Backend + Knobs> Probe<'_, Bk> {
    fn want(&self, name: &str) -> bool {
        self.filter.is_empty() || self.filter.iter().any(|f| name.starts_with(f.as_str()))
    }

    fn emit(&mut self, line: String) {
        writeln!(self.out, "{line}").expect("write OJAS_PROBE_OUT");
        self.out.flush().expect("flush OJAS_PROBE_OUT");
        eprintln!("{line}");
    }

    fn dev(&self, shape: &[usize], seed: u64, e: i32) -> R<Tensor> {
        let n = shape.iter().product();
        self.be
            .upload(&Tensor::from_f32(&gen(n, seed, e), shape, &self.host)?)
    }

    fn zeros(&self, shape: &[usize]) -> R<Tensor> {
        self.be
            .upload(&Tensor::zeros(shape, ojas_core::DType::F32, &self.host)?)
    }

    /// Run `variants` (name, one timed iteration) interleaved. An iteration
    /// is timed through its sync. A failing variant is recorded and the
    /// probe stops.
    fn interleave<S>(
        &mut self,
        probe: &str,
        state: &mut S,
        variants: &mut [(&str, &mut Step<'_, Bk, S>)],
    ) {
        if !self.want(probe) {
            return;
        }
        for round in 0..self.rounds {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..variants.len()).collect()
            } else {
                (0..variants.len()).rev().collect()
            };
            for &vi in &order {
                let (name, f) = &mut variants[vi];
                let mut ms = Vec::with_capacity(self.iters);
                for i in 0..self.warmup + self.iters {
                    let t0 = Instant::now();
                    let res = f(self.be, state).and_then(|()| self.be.sync());
                    let dt = t0.elapsed().as_secs_f64() * 1e3;
                    if let Err(e) = res {
                        let line = format!(
                            "{{\"runtime\":\"{}\",\"probe\":\"{probe}\",\"variant\":\"{name}\",\"status\":\"error\",\"detail\":\"{}\"}}",
                            self.runtime,
                            format!("{e}").replace('"', "'")
                        );
                        self.emit(line);
                        return;
                    }
                    if i >= self.warmup {
                        ms.push(dt);
                    }
                }
                ms.sort_by(f64::total_cmp);
                let line = format!(
                    "{{\"runtime\":\"{}\",\"probe\":\"{probe}\",\"variant\":\"{name}\",\"round\":{round},\"status\":\"ok\",\"min_ms\":{:.6},\"median_ms\":{:.6},\"iters\":{}}}",
                    self.runtime,
                    ms[0],
                    ms[ms.len() / 2],
                    ms.len()
                );
                self.emit(line);
            }
        }
    }
}

struct Adam {
    p: Vec<Tensor>,
    g: Vec<Tensor>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    step: u64,
}

impl Adam {
    fn step<Bk: Backend>(&mut self, be: &Bk) -> R<()> {
        let cfg = AdamWConfig::nanolab(1e-3, 0.1);
        for i in 0..self.p.len() {
            be.adamw_step(
                &mut self.p[i],
                &self.g[i],
                &mut self.m[i],
                &mut self.v[i],
                self.step,
                cfg,
            )?;
        }
        self.step += 1;
        Ok(())
    }
}

fn adamw<Bk: Backend + Knobs>(p: &mut Probe<'_, Bk>) -> R<()> {
    if !p.want("adamw") {
        return Ok(());
    }
    let shapes = param_shapes();
    let total: usize = shapes.iter().map(|s| s.iter().product::<usize>()).sum();
    let mut many = Adam {
        p: Vec::new(),
        g: Vec::new(),
        m: Vec::new(),
        v: Vec::new(),
        step: 0,
    };
    for (i, s) in shapes.iter().enumerate() {
        many.p.push(p.dev(s, 2000 + i as u64, -5)?);
        many.g.push(p.dev(s, 3000 + i as u64, -6)?);
        many.m.push(p.zeros(s)?);
        many.v.push(p.zeros(s)?);
    }
    let mut one = Adam {
        p: vec![p.dev(&[total], 2000, -5)?],
        g: vec![p.dev(&[total], 3000, -6)?],
        m: vec![p.zeros(&[total])?],
        v: vec![p.zeros(&[total])?],
        step: 0,
    };
    let mut st = (&mut many, &mut one);
    p.interleave(
        "adamw",
        &mut st,
        &mut [
            ("per_tensor_170", &mut |be, st| st.0.step(be)),
            ("one_tensor_same_values", &mut |be, st| st.1.step(be)),
        ],
    );
    Ok(())
}

fn muon<Bk: Backend + Knobs>(p: &mut Probe<'_, Bk>) -> R<()> {
    if !p.want("muon") {
        return Ok(());
    }
    let k = 48usize;
    let s = [768usize, 768];
    let mut mats = Vec::with_capacity(k);
    for i in 0..k {
        mats.push((
            p.dev(&s, 4001 + i as u64, -5)?,
            p.dev(&s, 4101 + i as u64, -6)?,
            p.zeros(&s)?,
        ));
    }
    let x = p.dev(&s, 4201, -5)?;
    let stacked = p.dev(&[k * 768, 768], 4202, -5)?;
    let cfg = MuonNs5Config::nanolab_default();
    let mut st = (mats, x, stacked);
    if p.want("muon_batch") {
        p.interleave(
            "muon_batch",
            &mut st,
            &mut [
                ("muon_48x768sq_one_sync", &mut |be, st| {
                    for (pm, g, m) in st.0.iter_mut() {
                        be.muon_ns5_step(pm, g, m, cfg)?;
                    }
                    Ok(())
                }),
                // NS5's GEMMs, 15 per matrix: per-matrix against stacked rows
                // of the same total FLOPs (X X^T for every matrix at once).
                ("gemm_720_per_matrix", &mut |be, st| {
                    let mut keep = Vec::with_capacity(15 * 48);
                    for _ in 0..15 * 48 {
                        keep.push(be.linear_forward(&st.1, &st.1)?);
                    }
                    Ok(())
                }),
                ("gemm_15_stacked_48", &mut |be, st| {
                    let mut keep = Vec::with_capacity(15);
                    for _ in 0..15 {
                        keep.push(be.linear_forward(&st.2, &st.1)?);
                    }
                    Ok(())
                }),
            ],
        );
    }
    if p.want("muon_split") {
        for (rows, cols) in [(768usize, 768usize), (2048, 768)] {
            let sh = [rows, cols];
            let (r, c) = (rows.min(cols), rows.max(cols));
            let pm = p.dev(&sh, 4301, -5)?;
            let g = p.dev(&sh, 4302, -6)?;
            let m = p.zeros(&sh)?;
            // The wide orientation NS5 runs on: X [r, c], A [r, r], X^T [c, r].
            let xw = p.dev(&[r, c], 4303, -5)?;
            let a = p.dev(&[r, r], 4304, -5)?;
            let xt = p.dev(&[c, r], 4305, -5)?;
            let mut st2 = (pm, g, m, xw, a, xt);
            let name = format!("muon_split_{rows}x{cols}");
            p.interleave(
                &name,
                &mut st2,
                &mut [
                    ("muon_step", &mut |be, st| {
                        be.muon_ns5_step(&mut st.0, &st.1, &mut st.2, cfg)
                    }),
                    ("ns5_gemms_only", &mut |be, st| {
                        let mut keep = Vec::with_capacity(15);
                        for _ in 0..5 {
                            keep.push(be.linear_forward(&st.3, &st.3)?); // X X^T
                            keep.push(be.linear_forward(&st.4, &st.4)?); // A A (A symmetric)
                            keep.push(be.linear_forward(&st.4, &st.5)?); // B X
                        }
                        Ok(())
                    }),
                ],
            );
        }
    }
    Ok(())
}

fn flush<Bk: Backend + Knobs>(p: &mut Probe<'_, Bk>) -> R<()> {
    let values = p.be.flush_values();
    if !p.want("flush") || values.is_empty() {
        return Ok(());
    }
    let (bp, x, cos, sin, v0, gy) = block_setup(p.be, &p.host)?;
    let mut st = ();
    type Variant<'v, Bk> = Box<dyn FnMut(&Bk, &mut ()) -> R<()> + 'v>;
    let mut fns: Vec<(String, Variant<'_, Bk>)> = Vec::new();
    for &f in &values {
        let (bp, x, cos, sin, v0, gy) = (&bp, &x, &cos, &sin, &v0, &gy);
        fns.push((
            format!("flush_at_{f}"),
            Box::new(move |be: &Bk, _: &mut ()| {
                be.set_flush(f);
                let fw = block_forward(be, bp, x, cos, sin, v0)?;
                let grads = block_backward(be, bp, x, cos, sin, v0, &fw, gy)?;
                drop((fw, grads));
                Ok(())
            }),
        ));
    }
    let mut variants: Vec<(&str, &mut Step<'_, Bk, ()>)> = fns
        .iter_mut()
        .map(|(n, f)| (n.as_str(), f.as_mut() as &mut Step<'_, Bk, ()>))
        .collect();
    p.interleave("flush_block_fwd_bwd", &mut st, &mut variants);
    p.be.set_flush(ojas_default_flush(&values));
    Ok(())
}

/// The value the sweep leaves set: the first, which the examples put first
/// as the backend's default.
fn ojas_default_flush(values: &[usize]) -> usize {
    values[0]
}

fn acc<Bk: Backend + Knobs>(p: &mut Probe<'_, Bk>) -> R<()> {
    if !p.want("acc") {
        return Ok(());
    }
    for (rows, cols) in [(4096usize, 768usize), (50304, 768)] {
        let s = [rows, cols];
        let g = p.dev(&s, 152, 0)?;
        let mut st = (p.dev(&s, 151, -4)?, p.dev(&s, 151, -4)?, g);
        let name = format!("acc_{rows}x{cols}");
        p.interleave(
            &name,
            &mut st,
            &mut [
                ("sole_owner_in_place", &mut |be, st| {
                    be.accumulate_grad(&mut st.0, &st.2)
                }),
                ("shared_new_tensor", &mut |be, st| {
                    // A second handle makes the accumulator shared, as a
                    // gradient a residual add handed to both inputs is.
                    let other = st.1.clone();
                    be.accumulate_grad(&mut st.1, &st.2)?;
                    drop(other);
                    Ok(())
                }),
            ],
        );
    }
    Ok(())
}

fn parts<Bk: Backend + Knobs>(p: &mut Probe<'_, Bk>) -> R<()> {
    if !p.want("parts") {
        return Ok(());
    }
    if p.want("parts_gate") {
        let (xs, ws, bs, a4) = ([B, T, DM], [H, DM], [H], [B, T, H, D]);
        let x = p.dev(&xs, 91, 0)?;
        let w = p.dev(&ws, 92, -5)?;
        let b = p.dev(&bs, 93, 0)?;
        let a = p.dev(&a4, 94, 0)?;
        let g = p.dev(&a4, 95, 0)?;
        let gz = p.dev(&[B, T, H], 96, 0)?;
        let mut st = (x, w, b, a, g, gz);
        p.interleave(
            "parts_gate",
            &mut st,
            &mut [
                ("gate_fwd", &mut |be, st| {
                    drop(be.per_head_sigmoid_gate_forward(&st.0, &st.1, &st.2, &st.3)?);
                    Ok(())
                }),
                ("gemm_x_wt_4096x12x768", &mut |be, st| {
                    drop(be.linear_forward(&st.0, &st.1)?);
                    Ok(())
                }),
                ("gate_bwd", &mut |be, st| {
                    drop(be.per_head_sigmoid_gate_backward(&st.0, &st.1, &st.2, &st.3, &st.4)?);
                    Ok(())
                }),
                ("gemm_bwd_pair_4096x12x768", &mut |be, st| {
                    drop(be.linear_backward(&st.0, &st.1, &st.5)?);
                    Ok(())
                }),
            ],
        );
    }
    if p.want("parts_ce") {
        let ls = [N, V];
        let l = p.dev(&ls, 121, 2)?;
        let other = p.dev(&ls, 122, 0)?;
        let t = p.be.upload(&Tensor::from_u32(&gen_targets(N, 122), &[N], &p.host)?)?;
        let mut st = (l, other, t);
        p.interleave(
            "parts_ce_bwd",
            &mut st,
            &mut [
                ("cross_entropy_bwd", &mut |be, st| {
                    drop(be.cross_entropy_mean_backward(&st.0, &st.2, None)?);
                    Ok(())
                }),
                ("mul_floor_same_shape", &mut |be, st| {
                    drop(be.mul_forward(&st.0, &st.1)?);
                    Ok(())
                }),
            ],
        );
    }
    if p.want("parts_acc") {
        let s = [V, DM];
        let mut st = (p.dev(&s, 151, -4)?, p.dev(&s, 152, 0)?);
        p.interleave(
            "parts_acc",
            &mut st,
            &mut [
                ("accumulate_grad_in_place", &mut |be, st| {
                    be.accumulate_grad(&mut st.0, &st.1)
                }),
                ("mul_floor_same_shape", &mut |be, st| {
                    drop(be.mul_forward(&st.0, &st.1)?);
                    Ok(())
                }),
            ],
        );
    }
    if p.want("parts_sdpa") {
        let s = [4usize, 12, 1024, 64];
        let mut st = (p.dev(&s, 21, 0)?, p.dev(&s, 22, 0)?, p.dev(&s, 23, 0)?);
        p.interleave(
            "parts_sdpa_d64",
            &mut st,
            &mut [("sdpa_fwd_b4h12t1024d64", &mut |be, st| {
                drop(be.causal_sdpa_forward(&st.0, &st.1, &st.2, None)?);
                Ok(())
            })],
        );
    }
    Ok(())
}

fn link<Bk: Backend + Knobs>(p: &mut Probe<'_, Bk>) -> R<()> {
    if !p.want("link") || p.be.round_trip().is_none() {
        return Ok(());
    }
    let one = p.dev(&[1], 1, 0)?;
    let (qs, cs) = ([1usize, 1, H, D], [1usize, T, H, D]);
    let q = p.dev(&qs, 141, 0)?;
    let k = p.dev(&cs, 142, 0)?;
    let v = p.dev(&cs, 143, 0)?;
    let mut st = (one, q, k, v);
    const CALLS: usize = 256;
    p.interleave(
        "link_256_calls",
        &mut st,
        &mut [
            ("empty_round_trip", &mut |be, _| {
                for _ in 0..CALLS {
                    be.round_trip().expect("Metal")?;
                }
                Ok(())
            }),
            ("silu_1_value", &mut |be, st| {
                let mut keep = Vec::with_capacity(CALLS);
                for _ in 0..CALLS {
                    keep.push(be.silu_forward(&st.0)?);
                }
                Ok(())
            }),
            ("decode_attn_one_sync", &mut |be, st| {
                let mut keep = Vec::with_capacity(CALLS);
                for _ in 0..CALLS {
                    keep.push(be.cached_attention_forward(&st.1, &st.2, &st.3, T, None)?);
                }
                Ok(())
            }),
            ("decode_attn_sync_each", &mut |be, st| {
                for _ in 0..CALLS {
                    let y = be.cached_attention_forward(&st.1, &st.2, &st.3, T, None)?;
                    be.sync()?;
                    drop(y);
                }
                Ok(())
            }),
        ],
    );
    Ok(())
}

/// Read the environment and run every wanted probe.
pub fn probe_main<Bk: Backend + Knobs>(be: &Bk, runtime: &str) -> Result<(), String> {
    let env = |k: &str| std::env::var(k).ok();
    let num = |k: &str, d: usize| -> Result<usize, String> {
        env(k).map_or(Ok(d), |s| s.parse().map_err(|e| format!("{k}: {e}")))
    };
    let out_path = env("OJAS_PROBE_OUT").ok_or("OJAS_PROBE_OUT is not set")?;
    let out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .map_err(|e| format!("{out_path}: {e}"))?;
    let mut p = Probe {
        be,
        host: Budget::new(48 << 30),
        rounds: num("PROBE_ROUNDS", 7)?,
        warmup: num("PROBE_WARMUP", 3)?,
        iters: num("PROBE_ITERS", 10)?,
        runtime: runtime.to_string(),
        filter: env("OJAS_PROBES")
            .map(|s| s.split(',').filter(|x| !x.is_empty()).map(str::to_string).collect())
            .unwrap_or_default(),
        out,
    };
    let run = |r: R<()>| r.map_err(|e| format!("{e}"));
    run(adamw(&mut p))?;
    run(muon(&mut p))?;
    run(flush(&mut p))?;
    run(acc(&mut p))?;
    run(parts(&mut p))?;
    run(link(&mut p))?;
    Ok(())
}
