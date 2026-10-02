//! The ojas side of the paired GPU-vs-torch benchmark (`bench/README.md`).
//!
//! One generic runner over `ojas_core::Backend`. It is compiled into two
//! examples, `ojas-metal/examples/metal_vs_torch.rs` and
//! `ojas-wgpu/examples/wgpu_vs_torch.rs`, through `#[path]`, so both GPU
//! backends run the identical rows. `bench/torch_rows.py` holds the torch
//! twin of every row; the row names and the `spec` strings must match.
//!
//! Inputs come from `gen`, a 32-bit integer hash that `torch_rows.py`
//! reproduces bit for bit (every scale is a power of two, so the float
//! conversion is exact on both sides). Before any row, the binary checks the
//! first values of one stream against `generator.f32`, which the torch `ref`
//! pass wrote.
//!
//! Each row:
//! 1. builds its inputs once and uploads them;
//! 2. runs the op once, downloads every output, samples it (`sample`) and
//!    compares with the torch reference in `<ref>/<row>.f32`. A missing,
//!    stale (`spec` differs) or failing reference is reported and the row is
//!    NOT timed;
//! 3. warms up `warmup` times and times `iters` iterations, each one the op
//!    followed by `Backend::sync`, which waits for the recorded work and
//!    reports any deferred fault on both Metal and wgpu.
//!    Outputs are held until after the synchronize, then dropped.
//!
//! One JSON object per row goes to the file named by `OJAS_BENCH_OUT`.

use std::cell::RefCell;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use ojas_core::{AdamWConfig, Backend, Budget, CeChunk, MuonNs5Config, OjasError, Tensor};

pub type R<T> = Result<T, OjasError>;

pub const B: usize = 4;
pub const T: usize = 1024;
pub const H: usize = 12;
pub const D: usize = 64;
pub const DM: usize = 768;
pub const FF: usize = 2048;
pub const V: usize = 50304;
pub const N: usize = B * T;
pub const EPS: f32 = 1e-6;
/// Largest number of values one output contributes to the parity file.
pub const SAMPLE_MAX: usize = 1 << 22;
/// Default parity gate: max |ojas - torch| / max |torch|, per output.
pub const TOL: f64 = 1e-3;

const M32: u64 = 0xFFFF_FFFF;

/// The shared 32-bit hash. `torch_rows.py::_hash` is the same arithmetic on
/// int64 tensors; every product stays below 2^63 there.
pub fn hash32(i: u64, seed: u64) -> u64 {
    let mut x = (i.wrapping_mul(0x7FEB_352D)).wrapping_add(seed.wrapping_mul(0x2C1B_3C6D)) & M32;
    x ^= x >> 15;
    x = x.wrapping_mul(0x297A_2D39) & M32;
    x ^= x >> 12;
    x = x.wrapping_mul(0x2C1B_3C6D) & M32;
    x ^= x >> 15;
    x
}

/// `n` values uniform in `[-1, 1) * 2^scale_log2`, exactly as torch makes them.
pub fn gen(n: usize, seed: u64, scale_log2: i32) -> Vec<f32> {
    let scale = 2.0f32.powi(scale_log2);
    (0..n as u64)
        .map(|i| ((hash32(i, seed) >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0) * scale)
        .collect()
}

/// Class ids in `[0, V)`.
pub fn gen_targets(n: usize, seed: u64) -> Vec<u32> {
    (0..n as u64)
        .map(|i| (hash32(i, seed) % V as u64) as u32)
        .collect()
}

/// The values one output contributes: all of it up to `SAMPLE_MAX`, else
/// every `n / SAMPLE_MAX`-th value from index 0. `torch_rows.py::sample`
/// takes the same indices.
pub fn sample(v: &[f32]) -> Vec<f32> {
    if v.len() <= SAMPLE_MAX {
        return v.to_vec();
    }
    let stride = v.len() / SAMPLE_MAX;
    (0..SAMPLE_MAX).map(|k| v[k * stride]).collect()
}

fn dims(s: &[usize]) -> String {
    s.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

/// Canonical input description, `name:shape:seed:log2scale`. The torch twin
/// formats the identical string; a mismatch means a stale reference.
pub fn spec(parts: &[(&str, &[usize], u64, i32)]) -> String {
    parts
        .iter()
        .map(|(n, s, seed, e)| format!("{n}:{}:{seed}:{e}", dims(s)))
        .collect::<Vec<_>>()
        .join(";")
}

pub fn view(t: &Tensor, shape: &[usize]) -> R<Tensor> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    t.view(shape, &strides, t.byte_offset())
}

/// The swap `[B, T, H, D] <-> [B, H, T, D]` is its own inverse.
pub const SWAP12: [usize; 4] = [0, 2, 1, 3];

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn status_of(e: &OjasError) -> &'static str {
    match e {
        OjasError::Unsupported { .. } | OjasError::CapacityExceeded { .. } => "refused",
        _ => "error",
    }
}

pub struct Runner<'a, Bk: Backend> {
    pub be: &'a Bk,
    pub runtime: String,
    pub host: Budget,
    pub refdir: PathBuf,
    pub warmup: usize,
    pub iters: usize,
    pub filter: Vec<String>,
    pub out: fs::File,
}

/// Parity outcome of one row.
struct Parity {
    max_abs: f64,
    worst_rel: f64,
}

impl<'a, Bk: Backend> Runner<'a, Bk> {
    pub fn want(&self, name: &str) -> bool {
        self.filter.is_empty() || self.filter.iter().any(|f| name.starts_with(f.as_str()))
    }

    pub fn emit(&mut self, line: String) {
        // A row that cannot be written is lost evidence: fail loudly.
        writeln!(self.out, "{line}").expect("write OJAS_BENCH_OUT");
        self.out.flush().expect("flush OJAS_BENCH_OUT");
        eprintln!("{line}");
    }

    fn record(&mut self, name: &str, status: &str, detail: &str) {
        let line = format!(
            "{{\"runtime\":{},\"row\":{},\"status\":{},\"detail\":{}}}",
            json_str(&self.runtime),
            json_str(name),
            json_str(status),
            json_str(detail)
        );
        self.emit(line);
    }

    /// Run a row group; an error escaping it (setup, upload) is recorded
    /// against every row of the group that was wanted.
    pub fn group(&mut self, names: &[&str], f: impl FnOnce(&mut Self) -> R<()>) {
        if !names.iter().any(|n| self.want(n)) {
            return;
        }
        if let Err(e) = f(self) {
            let detail = format!("{e}");
            for n in names {
                if self.want(n) {
                    self.record(n, status_of(&e), &format!("setup: {detail}"));
                }
            }
        }
    }

    pub fn dev(&self, shape: &[usize], seed: u64, scale_log2: i32) -> R<Tensor> {
        let n = shape.iter().product();
        self.be.upload(&Tensor::from_f32(
            &gen(n, seed, scale_log2),
            shape,
            &self.host,
        )?)
    }

    pub fn zeros(&self, shape: &[usize]) -> R<Tensor> {
        self.be
            .upload(&Tensor::zeros(shape, ojas_core::DType::F32, &self.host)?)
    }

    pub fn targets(&self, n: usize, seed: u64) -> R<Tensor> {
        self.be
            .upload(&Tensor::from_u32(&gen_targets(n, seed), &[n], &self.host)?)
    }

    /// Download each tensor and return one sample per output.
    pub fn samples(&self, ts: &[&Tensor]) -> R<Vec<Vec<f32>>> {
        self.be.sync()?;
        ts.iter()
            .map(|t| Ok(sample(&self.be.download(t)?.to_f32_vec()?)))
            .collect()
    }

    fn load_ref(&self, name: &str, spec: &str) -> Result<Vec<Vec<f32>>, String> {
        let meta_path = self.refdir.join(format!("{name}.spec"));
        let meta = fs::read_to_string(&meta_path)
            .map_err(|e| format!("no reference {}: {e}", meta_path.display()))?;
        let mut lines = meta.lines();
        let ref_spec = lines.next().unwrap_or("");
        if ref_spec != spec {
            return Err(format!(
                "stale reference: spec {ref_spec:?} != this binary's {spec:?}"
            ));
        }
        let counts: Vec<usize> = lines
            .next()
            .and_then(|l| l.strip_prefix("counts="))
            .ok_or("reference has no counts line")?
            .split(',')
            .map(|c| {
                c.parse::<usize>()
                    .map_err(|e| format!("bad count {c:?}: {e}"))
            })
            .collect::<Result<_, _>>()?;
        let bytes = fs::read(self.refdir.join(format!("{name}.f32"))).map_err(|e| e.to_string())?;
        if bytes.len() != counts.iter().sum::<usize>() * 4 {
            return Err(format!(
                "reference holds {} bytes, counts say {}",
                bytes.len(),
                counts.iter().sum::<usize>() * 4
            ));
        }
        let all: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut out = Vec::with_capacity(counts.len());
        let mut at = 0;
        for c in counts {
            out.push(all[at..at + c].to_vec());
            at += c;
        }
        Ok(out)
    }

    fn compare(mine: &[Vec<f32>], theirs: &[Vec<f32>]) -> Result<Parity, String> {
        if mine.len() != theirs.len() {
            return Err(format!(
                "{} outputs here, {} in the reference",
                mine.len(),
                theirs.len()
            ));
        }
        let mut p = Parity {
            max_abs: 0.0,
            worst_rel: 0.0,
        };
        for (k, (a, b)) in mine.iter().zip(theirs).enumerate() {
            if a.len() != b.len() {
                return Err(format!(
                    "output {k}: {} sampled values here, {} in the reference",
                    a.len(),
                    b.len()
                ));
            }
            let mut err = 0f64;
            let mut mag = 0f64;
            for (&x, &y) in a.iter().zip(b) {
                if !x.is_finite() {
                    return Err(format!("output {k}: non-finite ojas value {x}"));
                }
                err = err.max((x as f64 - y as f64).abs());
                mag = mag.max((y as f64).abs());
            }
            p.max_abs = p.max_abs.max(err);
            let rel = if mag > 0.0 {
                err / mag
            } else if err == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            p.worst_rel = p.worst_rel.max(rel);
        }
        Ok(p)
    }

    /// `uptime` load and the GPU's "Device Utilization %" (an instantaneous
    /// `ioreg` sample), recorded before and after every row.
    fn load(&mut self, row: &str, when: &str) {
        let run = |cmd: &str, args: &[&str]| {
            std::process::Command::new(cmd)
                .args(args)
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_else(|e| format!("{cmd} failed: {e}"))
        };
        let up = run("uptime", &[]);
        let up = up.split("load average").nth(1).unwrap_or(up.trim());
        let up = up.trim_start_matches(['s', ':', ' ']).trim();
        let io = run(
            "ioreg",
            &["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
        );
        let tag = "\"Device Utilization %\"=";
        let gpu = io
            .find(tag)
            .and_then(|i| {
                io[i + tag.len()..]
                    .split(|c: char| !c.is_ascii_digit())
                    .next()
                    .and_then(|d| d.parse::<u32>().ok())
            })
            .map_or("null".to_string(), |v| v.to_string());
        let line = format!(
            "{{\"runtime\":{},\"row\":\"_load\",\"status\":\"info\",\"for\":{},\"when\":{},\"load\":{},\"gpu_util_pct\":{gpu}}}",
            json_str(&self.runtime),
            json_str(row),
            json_str(when),
            json_str(up)
        );
        self.emit(line);
    }

    /// Parity first, then timing. `parity` runs the op once from the row's
    /// initial state and returns the sampled outputs; `step` is one timed
    /// iteration and returns what must stay alive until the synchronize.
    /// Machine load is recorded before and after the row.
    pub fn run<S>(
        &mut self,
        name: &str,
        spec: &str,
        tol: f64,
        state: &mut S,
        parity: impl FnOnce(&Self, &mut S) -> R<Vec<Vec<f32>>>,
        step: impl FnMut(&Self, &mut S) -> R<Vec<Tensor>>,
    ) {
        if !self.want(name) {
            return;
        }
        self.load(name, "before");
        self.run_row(name, spec, tol, state, parity, step);
        self.load(name, "after");
    }

    fn run_row<S>(
        &mut self,
        name: &str,
        spec: &str,
        tol: f64,
        state: &mut S,
        parity: impl FnOnce(&Self, &mut S) -> R<Vec<Vec<f32>>>,
        mut step: impl FnMut(&Self, &mut S) -> R<Vec<Tensor>>,
    ) {
        let mine = match parity(self, state) {
            Ok(v) => v,
            Err(e) => {
                let st = status_of(&e);
                self.record(name, st, &format!("{e}"));
                return;
            }
        };
        let theirs = match self.load_ref(name, spec) {
            Ok(v) => v,
            Err(e) => {
                self.record(name, "no_ref", &e);
                return;
            }
        };
        let p = match Self::compare(&mine, &theirs) {
            Ok(p) => p,
            Err(e) => {
                self.record(name, "parity_fail", &e);
                return;
            }
        };
        drop((mine, theirs));
        let parity_json = format!(
            "\"parity_max_abs\":{:e},\"parity_rel\":{:e},\"tol\":{:e}",
            p.max_abs, p.worst_rel, tol
        );
        if p.worst_rel.is_nan() || p.worst_rel > tol {
            let line = format!(
                "{{\"runtime\":{},\"row\":{},\"status\":\"parity_fail\",{parity_json}}}",
                json_str(&self.runtime),
                json_str(name)
            );
            self.emit(line);
            return;
        }
        let mut ms = Vec::with_capacity(self.iters);
        let mut failure = None;
        for i in 0..self.warmup + self.iters {
            let t0 = Instant::now();
            let kept = step(self, state).and_then(|k| self.be.sync().map(|()| k));
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            match kept {
                Ok(k) => drop(k),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
            if i >= self.warmup {
                ms.push(dt);
            }
        }
        if let Some(e) = failure {
            let st = status_of(&e);
            self.record(name, st, &format!("during timing: {e}"));
            return;
        }
        ms.sort_by(f64::total_cmp);
        let med = if ms.len() % 2 == 1 {
            ms[ms.len() / 2]
        } else {
            0.5 * (ms[ms.len() / 2 - 1] + ms[ms.len() / 2])
        };
        let line = format!(
            "{{\"runtime\":{},\"row\":{},\"status\":\"ok\",\"min_ms\":{:.6},\"median_ms\":{:.6},\"iters\":{},\"warmup\":{},{parity_json}}}",
            json_str(&self.runtime),
            json_str(name),
            ms[0],
            med,
            ms.len(),
            self.warmup
        );
        self.emit(line);
    }

    /// A stateless op: `f` returns its outputs; parity samples them.
    pub fn op(&mut self, name: &str, spec: &str, tol: f64, f: impl Fn(&Self) -> R<Vec<Tensor>>) {
        let f = &f;
        self.run(
            name,
            spec,
            tol,
            &mut (),
            |r, _| {
                let outs = f(r)?;
                r.samples(&outs.iter().collect::<Vec<_>>())
            },
            |r, _| f(r),
        );
    }
}

/// Every row, in a fixed order.
pub fn run_all<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    floor(r);
    linear(r, "qkv", N, DM, DM);
    linear(r, "up", N, DM, FF);
    linear(r, "down", N, FF, DM);
    linear(r, "lmhead", N, DM, V);
    sdpa(r, "b4h12t1024d64", [4, 12, 1024, 64]);
    sdpa(r, "b4h8t2048d64", [4, 8, 2048, 64]);
    sdpa(r, "b2h8t1024d128", [2, 8, 1024, 128]);
    rms_norm(r);
    qk_norm(r);
    rope(r);
    silu(r);
    mul(r);
    add(r);
    gate(r);
    vres(r);
    permute(r);
    cross_entropy(r);
    linear_ce(r, 1024, 8192);
    linear_ce(r, N, V);
    clip(r);
    adamw(r);
    for (rows, cols) in [(DM, DM), (FF, DM), (DM, FF)] {
        muon(r, rows, cols);
    }
    block(r);
    decode_attention(r);
    accumulate_grad(r);
}

fn floor<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    let names = ["floor_silu_1"];
    r.group(&names, |r| {
        let x = r.dev(&[1], 1, 0)?;
        let s = spec(&[("x", &[1], 1, 0)]);
        r.op("floor_silu_1", &s, TOL, |r| {
            Ok(vec![r.be.silu_forward(&x)?])
        });
        Ok(())
    });
}

fn linear<Bk: Backend>(r: &mut Runner<'_, Bk>, tag: &str, rows: usize, kin: usize, nout: usize) {
    let fwd = format!("linear_{tag}_fwd");
    let bwd = format!("linear_{tag}_bwd");
    r.group(&[fwd.as_str(), bwd.as_str()], |r| {
        let (xs, ws, gs) = ([rows, kin], [nout, kin], [rows, nout]);
        let x = r.dev(&xs, 11, 0)?;
        let w = r.dev(&ws, 12, -5)?;
        let sf = spec(&[("x", &xs, 11, 0), ("w", &ws, 12, -5)]);
        r.op(&fwd, &sf, TOL, |r| Ok(vec![r.be.linear_forward(&x, &w)?]));
        if r.want(&bwd) {
            let g = r.dev(&gs, 13, 0)?;
            let sb = spec(&[("x", &xs, 11, 0), ("w", &ws, 12, -5), ("gy", &gs, 13, 0)]);
            r.op(&bwd, &sb, TOL, |r| {
                let (gx, gw) = r.be.linear_backward(&x, &w, &g)?;
                Ok(vec![gx, gw])
            });
        }
        Ok(())
    });
}

fn sdpa<Bk: Backend>(r: &mut Runner<'_, Bk>, tag: &str, s: [usize; 4]) {
    let fwd = format!("sdpa_{tag}_fwd");
    let bwd = format!("sdpa_{tag}_bwd");
    r.group(&[fwd.as_str(), bwd.as_str()], |r| {
        let q = r.dev(&s, 21, 0)?;
        let k = r.dev(&s, 22, 0)?;
        let v = r.dev(&s, 23, 0)?;
        let sf = spec(&[("q", &s, 21, 0), ("k", &s, 22, 0), ("v", &s, 23, 0)]);
        r.op(&fwd, &sf, TOL, |r| {
            Ok(vec![r.be.causal_sdpa_forward(&q, &k, &v)?])
        });
        if r.want(&bwd) {
            let g = r.dev(&s, 24, 0)?;
            let sb = format!("{sf};{}", spec(&[("gy", &s, 24, 0)]));
            r.op(&bwd, &sb, TOL, |r| {
                let (a, b, c) = r.be.causal_sdpa_backward(&q, &k, &v, &g)?;
                Ok(vec![a, b, c])
            });
        }
        Ok(())
    });
}

fn rms_norm<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["rms_norm_fwd", "rms_norm_bwd"], |r| {
        let (xs, ws) = ([N, DM], [DM]);
        let x = r.dev(&xs, 31, 0)?;
        let w = r.dev(&ws, 32, 0)?;
        let g = r.dev(&xs, 33, 0)?;
        let sf = spec(&[("x", &xs, 31, 0), ("w", &ws, 32, 0)]);
        r.op("rms_norm_fwd", &sf, TOL, |r| {
            Ok(vec![r.be.rms_norm_forward(&x, &w, EPS)?])
        });
        let sb = format!("{sf};{}", spec(&[("gy", &xs, 33, 0)]));
        r.op("rms_norm_bwd", &sb, TOL, |r| {
            let (gx, gw) = r.be.rms_norm_backward(&x, &w, &g, EPS)?;
            Ok(vec![gx, gw])
        });
        Ok(())
    });
}

fn qk_norm<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["rms_qk_norm_fwd", "rms_qk_norm_bwd"], |r| {
        let (s, ws) = ([B, T, H, D], [D]);
        let q = r.dev(&s, 41, 0)?;
        let k = r.dev(&s, 42, 0)?;
        let qw = r.dev(&ws, 43, 0)?;
        let kw = r.dev(&ws, 44, 0)?;
        let sf = spec(&[
            ("q", &s, 41, 0),
            ("k", &s, 42, 0),
            ("qw", &ws, 43, 0),
            ("kw", &ws, 44, 0),
        ]);
        r.op("rms_qk_norm_fwd", &sf, TOL, |r| {
            let (a, b) = r.be.rms_qk_norm_forward(&q, &k, &qw, &kw, EPS)?;
            Ok(vec![a, b])
        });
        if r.want("rms_qk_norm_bwd") {
            let gq = r.dev(&s, 45, 0)?;
            let gk = r.dev(&s, 46, 0)?;
            let sb = format!("{sf};{}", spec(&[("gq", &s, 45, 0), ("gk", &s, 46, 0)]));
            r.op("rms_qk_norm_bwd", &sb, TOL, |r| {
                let (a, b, c, d) = r.be.rms_qk_norm_backward(&q, &k, &qw, &kw, &gq, &gk, EPS)?;
                Ok(vec![a, b, c, d])
            });
        }
        Ok(())
    });
}

fn rope<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["rope_fwd", "rope_bwd"], |r| {
        let (s, cs) = ([B, T, H, D], [T, D]);
        let x = r.dev(&s, 51, 0)?;
        let c = r.dev(&cs, 52, 0)?;
        let sn = r.dev(&cs, 53, 0)?;
        let sf = spec(&[("x", &s, 51, 0), ("cos", &cs, 52, 0), ("sin", &cs, 53, 0)]);
        r.op("rope_fwd", &sf, TOL, |r| {
            Ok(vec![r.be.rope_half_split_forward(&x, &c, &sn)?])
        });
        // The backward reads only the incoming gradient; seed 51 plays it.
        let sb = spec(&[("gy", &s, 51, 0), ("cos", &cs, 52, 0), ("sin", &cs, 53, 0)]);
        r.op("rope_bwd", &sb, TOL, |r| {
            Ok(vec![r.be.rope_half_split_backward(&x, &c, &sn)?])
        });
        Ok(())
    });
}

fn silu<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["silu_fwd", "silu_bwd"], |r| {
        let s = [N, FF];
        let x = r.dev(&s, 61, 2)?;
        let g = r.dev(&s, 62, 0)?;
        let sf = spec(&[("x", &s, 61, 2)]);
        r.op("silu_fwd", &sf, TOL, |r| Ok(vec![r.be.silu_forward(&x)?]));
        let sb = format!("{sf};{}", spec(&[("gy", &s, 62, 0)]));
        r.op("silu_bwd", &sb, TOL, |r| {
            Ok(vec![r.be.silu_backward(&x, &g)?])
        });
        Ok(())
    });
}

fn mul<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["mul_fwd", "mul_bwd"], |r| {
        let s = [N, FF];
        let a = r.dev(&s, 71, 0)?;
        let b = r.dev(&s, 72, 0)?;
        let g = r.dev(&s, 73, 0)?;
        let sf = spec(&[("a", &s, 71, 0), ("b", &s, 72, 0)]);
        r.op("mul_fwd", &sf, TOL, |r| {
            Ok(vec![r.be.mul_forward(&a, &b)?])
        });
        let sb = format!("{sf};{}", spec(&[("gy", &s, 73, 0)]));
        r.op("mul_bwd", &sb, TOL, |r| {
            let (x, y) = r.be.mul_backward(&a, &b, &g)?;
            Ok(vec![x, y])
        });
        Ok(())
    });
}

fn add<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["residual_add_fwd", "residual_add_bwd"], |r| {
        let s = [N, DM];
        let a = r.dev(&s, 81, 0)?;
        let b = r.dev(&s, 82, 0)?;
        let g = r.dev(&s, 83, 0)?;
        let sf = spec(&[("x", &s, 81, 0), ("y", &s, 82, 0)]);
        r.op("residual_add_fwd", &sf, TOL, |r| {
            Ok(vec![r.be.residual_add_forward(&a, &b)?])
        });
        let sb = format!("{sf};{}", spec(&[("gy", &s, 83, 0)]));
        r.op("residual_add_bwd", &sb, TOL, |r| {
            let (x, y) = r.be.residual_add_backward(&a, &b, &g)?;
            Ok(vec![x, y])
        });
        Ok(())
    });
}

fn gate<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["gate_fwd", "gate_bwd"], |r| {
        let (xs, ws, bs, as_) = ([B, T, DM], [H, DM], [H], [B, T, H, D]);
        let x = r.dev(&xs, 91, 0)?;
        let w = r.dev(&ws, 92, -5)?;
        let b = r.dev(&bs, 93, 0)?;
        let a = r.dev(&as_, 94, 0)?;
        let g = r.dev(&as_, 95, 0)?;
        let sf = spec(&[
            ("x", &xs, 91, 0),
            ("w", &ws, 92, -5),
            ("b", &bs, 93, 0),
            ("attn", &as_, 94, 0),
        ]);
        r.op("gate_fwd", &sf, TOL, |r| {
            Ok(vec![r.be.per_head_sigmoid_gate_forward(&x, &w, &b, &a)?])
        });
        let sb = format!("{sf};{}", spec(&[("gy", &as_, 95, 0)]));
        r.op("gate_bwd", &sb, TOL, |r| {
            let gg = r.be.per_head_sigmoid_gate_backward(&x, &w, &b, &a, &g)?;
            Ok(vec![gg.input, gg.weight, gg.bias, gg.attn_out])
        });
        Ok(())
    });
}

fn vres<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["vres_fwd", "vres_bwd"], |r| {
        let s = [B, T, H, D];
        let v = r.dev(&s, 101, 0)?;
        let v0 = r.dev(&s, 102, 0)?;
        let l = r.dev(&[1], 103, 0)?;
        let g = r.dev(&s, 104, 0)?;
        let sf = spec(&[("v", &s, 101, 0), ("v0", &s, 102, 0), ("lam", &[1], 103, 0)]);
        r.op("vres_fwd", &sf, TOL, |r| {
            Ok(vec![r.be.value_residual_blend_forward(&v, &v0, &l)?])
        });
        let sb = format!("{sf};{}", spec(&[("gy", &s, 104, 0)]));
        r.op("vres_bwd", &sb, TOL, |r| {
            let gg = r.be.value_residual_blend_backward(&v, &v0, &l, &g)?;
            Ok(vec![gg.value, gg.value0, gg.lambda])
        });
        Ok(())
    });
}

fn permute<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["permute_bthd_bhtd"], |r| {
        let s = [B, T, H, D];
        let x = r.dev(&s, 111, 0)?;
        let sf = spec(&[("x", &s, 111, 0)]);
        r.op("permute_bthd_bhtd", &sf, 0.0, |r| {
            Ok(vec![r.be.permute(&x, &SWAP12)?])
        });
        Ok(())
    });
}

fn cross_entropy<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["cross_entropy_fwd", "cross_entropy_bwd"], |r| {
        let ls = [N, V];
        let l = r.dev(&ls, 121, 2)?;
        let t = r.targets(N, 122)?;
        let s = spec(&[("logits", &ls, 121, 2), ("targets", &[N], 122, 0)]);
        r.op("cross_entropy_fwd", &s, TOL, |r| {
            Ok(vec![r.be.cross_entropy_mean_forward(&l, &t, None)?])
        });
        r.op("cross_entropy_bwd", &s, TOL, |r| {
            Ok(vec![r.be.cross_entropy_mean_backward(&l, &t, None)?])
        });
        Ok(())
    });
}

/// Fused linear + mean cross-entropy, loss and both gradients, with logit
/// chunks of `rows x cols`. Inputs: x [N, d] at 2^0, the tied weight
/// [V, d] at 2^-5, targets seed 132.
fn linear_ce<Bk: Backend>(r: &mut Runner<'_, Bk>, rows: usize, cols: usize) {
    let name = format!("linear_ce_c{rows}x{cols}");
    r.group(&[name.as_str()], |r| {
        let (xs, ws) = ([N, DM], [V, DM]);
        let x = r.dev(&xs, 131, 0)?;
        let w = r.dev(&ws, 133, -5)?;
        let t = r.targets(N, 132)?;
        let s = spec(&[
            ("x", &xs, 131, 0),
            ("w", &ws, 133, -5),
            ("targets", &[N], 132, 0),
        ]);
        let chunk = CeChunk { rows, cols };
        r.op(&name, &s, TOL, |r| {
            let out =
                r.be.linear_cross_entropy_mean(&x, &w, &t, None, chunk, true)?;
            let missing = || OjasError::Unsupported {
                op: "linear_cross_entropy_mean",
                detail: "want_grad was set but a gradient is missing".to_string(),
            };
            let gx = out.grad_input.ok_or_else(missing)?;
            let gw = out.grad_weight.ok_or_else(missing)?;
            Ok(vec![out.loss, gx, gw])
        });
        Ok(())
    });
}

/// Decode attention: one new query per head against a 1024-position KV
/// cache, B 1, H = Hkv = 12, D 64. The caches are `[B, Tcap, Hkv, D]`.
fn decode_attention<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    let name = "decode_attn_kv1024";
    r.group(&[name], |r| {
        let (qs, cs) = ([1, 1, H, D], [1, T, H, D]);
        let q = r.dev(&qs, 141, 0)?;
        let k = r.dev(&cs, 142, 0)?;
        let v = r.dev(&cs, 143, 0)?;
        let s = spec(&[
            ("q", &qs, 141, 0),
            ("k_cache", &cs, 142, 0),
            ("v_cache", &cs, 143, 0),
        ]);
        r.op(name, &format!("{s};kv_len:{T}"), TOL, |r| {
            Ok(vec![r.be.cached_attention_forward(&q, &k, &v, T)?])
        });
        Ok(())
    });
}

/// `acc += g` in place on a [V, d] tensor (the tied embedding gradient).
fn accumulate_grad<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    let name = "accumulate_grad_50304x768";
    r.group(&[name], |r| {
        let sh = [V, DM];
        let mut st = (r.dev(&sh, 151, 0)?, r.dev(&sh, 152, 0)?);
        let s = spec(&[("acc", &sh, 151, 0), ("g", &sh, 152, 0)]);
        r.run(
            name,
            &s,
            TOL,
            &mut st,
            |r, st| {
                r.be.accumulate_grad(&mut st.0, &st.1)?;
                r.samples(&[&st.0])
            },
            |r, st| {
                r.be.accumulate_grad(&mut st.0, &st.1)?;
                Ok(Vec::new())
            },
        );
        Ok(())
    });
}

/// nanolab default parameter shapes (123,699,612 values in 170 tensors),
/// in the order `torch_rows.py::param_shapes` uses; it asserts the multiset
/// equals the real `GPT(Config())` parameters.
pub fn param_shapes() -> Vec<Vec<usize>> {
    let mut v = vec![vec![V, DM]];
    for _ in 0..12 {
        v.push(vec![DM]); // norm1
        v.push(vec![1]); // vr_lambda
        for _ in 0..4 {
            v.push(vec![DM, DM]); // q, k, v, o
        }
        v.push(vec![D]); // q_norm
        v.push(vec![D]); // k_norm
        v.push(vec![H, DM]); // gate.weight
        v.push(vec![H]); // gate.bias
        v.push(vec![DM]); // norm2
        v.push(vec![FF, DM]); // ffn.gate
        v.push(vec![FF, DM]); // ffn.up
        v.push(vec![DM, FF]); // ffn.down
    }
    v.push(vec![DM]); // norm_f
    v
}

fn clip<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["clip_grad_norm_full"], |r| {
        let shapes = param_shapes();
        let mut grads = Vec::with_capacity(shapes.len());
        for (i, s) in shapes.iter().enumerate() {
            grads.push(r.dev(s, 1000 + i as u64, -6)?);
        }
        let last = grads.len() - 1;
        // Seeds 1000.. at 2^-6; total norm about 100, so max_norm 1.0
        // scales. Every later call uses 0.9 * the previous norm, so every
        // timed call also scales (ojas skips the multiply when it would not).
        let s = "clip:params:1000:-6:max1.0".to_string();
        let mut st = (grads, 1.0f32);
        r.run(
            "clip_grad_norm_full",
            &s,
            TOL,
            &mut st,
            |r, st| {
                let norm = r.be.clip_grad_norm(&mut st.0, 1.0)?;
                st.1 = 0.9 * norm;
                let mut out = vec![vec![norm]];
                out.extend(r.samples(&[&st.0[0], &st.0[last]])?);
                Ok(out)
            },
            |r, st| {
                let norm = r.be.clip_grad_norm(&mut st.0, st.1)?;
                st.1 = 0.9 * norm;
                Ok(Vec::new())
            },
        );
        Ok(())
    });
}

struct Adam {
    p: Vec<Tensor>,
    g: Vec<Tensor>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    step: u64,
}

fn adamw_all<Bk: Backend>(r: &Runner<'_, Bk>, st: &mut Adam) -> R<()> {
    let cfg = AdamWConfig::nanolab(1e-3, 0.1);
    for i in 0..st.p.len() {
        r.be.adamw_step(
            &mut st.p[i],
            &st.g[i],
            &mut st.m[i],
            &mut st.v[i],
            st.step,
            cfg,
        )?;
    }
    st.step += 1;
    Ok(())
}

fn adamw<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["adamw_full"], |r| {
        let shapes = param_shapes();
        let mut st = Adam {
            p: Vec::new(),
            g: Vec::new(),
            m: Vec::new(),
            v: Vec::new(),
            step: 0,
        };
        for (i, s) in shapes.iter().enumerate() {
            st.p.push(r.dev(s, 2000 + i as u64, -5)?);
            st.g.push(r.dev(s, 3000 + i as u64, -6)?);
            st.m.push(r.zeros(s)?);
            st.v.push(r.zeros(s)?);
        }
        let (mid, last) = (shapes.len() / 2, shapes.len() - 1);
        let s = "adamw:p2000:-5:g3000:-6:lr1e-3:b0.9,0.95:eps1e-8:wd0.1".to_string();
        r.run(
            "adamw_full",
            &s,
            TOL,
            &mut st,
            |r, st| {
                adamw_all(r, st)?;
                r.samples(&[&st.p[0], &st.p[mid], &st.p[last], &st.v[mid]])
            },
            |r, st| {
                adamw_all(r, st)?;
                Ok(Vec::new())
            },
        );
        Ok(())
    });
}

fn muon<Bk: Backend>(r: &mut Runner<'_, Bk>, rows: usize, cols: usize) {
    let name = format!("muon_{rows}x{cols}");
    r.group(&[name.as_str()], |r| {
        let s = [rows, cols];
        let mut st = (r.dev(&s, 4001, -5)?, r.dev(&s, 4002, -6)?, r.zeros(&s)?);
        let sp = spec(&[("p", &s, 4001, -5), ("g", &s, 4002, -6)]);
        let cfg = MuonNs5Config::nanolab_default();
        // Five NS5 iterations compound rounding; the gate is looser here.
        r.run(
            &name,
            &sp,
            1e-2,
            &mut st,
            |r, st| {
                r.be.muon_ns5_step(&mut st.0, &st.1, &mut st.2, cfg)?;
                r.samples(&[&st.0, &st.2])
            },
            |r, st| {
                r.be.muon_ns5_step(&mut st.0, &st.1, &mut st.2, cfg)?;
                Ok(Vec::new())
            },
        );
        Ok(())
    });
}

/// One nanolab attention + SwiGLU block's parameters (`Block.state_dict`).
struct BlockParams {
    n1: Tensor,
    lam: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    qn: Tensor,
    kn: Tensor,
    gw: Tensor,
    gb: Tensor,
    n2: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
}

/// `(name, shape, seed, log2 scale)` in `state_dict` order; `torch_rows.py`
/// loads the same tensors into nanolab's real `Block`.
pub fn block_param_specs() -> Vec<(&'static str, Vec<usize>, u64, i32)> {
    vec![
        ("norm1.weight", vec![DM], 501, 0),
        ("mixer.vr_lambda", vec![1], 502, 0),
        ("mixer.q_proj.weight", vec![DM, DM], 503, -5),
        ("mixer.k_proj.weight", vec![DM, DM], 504, -5),
        ("mixer.v_proj.weight", vec![DM, DM], 505, -5),
        ("mixer.o_proj.weight", vec![DM, DM], 506, -5),
        ("mixer.q_norm.weight", vec![D], 507, 0),
        ("mixer.k_norm.weight", vec![D], 508, 0),
        ("mixer.gate.weight", vec![H, DM], 509, -5),
        ("mixer.gate.bias", vec![H], 510, 0),
        ("norm2.weight", vec![DM], 511, 0),
        ("ffn.gate.weight", vec![FF, DM], 512, -5),
        ("ffn.up.weight", vec![FF, DM], 513, -5),
        ("ffn.down.weight", vec![DM, FF], 514, -5),
    ]
}

struct Fwd {
    h1: Tensor,
    q4: Tensor,
    k4: Tensor,
    v4: Tensor,
    qp: Tensor,
    kp: Tensor,
    vp: Tensor,
    at: Tensor,
    ga: Tensor,
    x1: Tensor,
    h2: Tensor,
    g1: Tensor,
    u: Tensor,
    s: Tensor,
    m: Tensor,
    y: Tensor,
}

/// nanolab `Block.forward` (attention mixer, layer > 0 so `v0` blends in),
/// through Backend ops. Returns every activation the backward needs.
fn block_forward<Bk: Backend>(
    be: &Bk,
    p: &BlockParams,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    v0: &Tensor,
) -> R<Fwd> {
    let s4 = [B, T, H, D];
    let h1 = be.rms_norm_forward(x, &p.n1, EPS)?;
    let q4 = view(&be.linear_forward(&h1, &p.wq)?, &s4)?;
    let k4 = view(&be.linear_forward(&h1, &p.wk)?, &s4)?;
    let v4 = view(&be.linear_forward(&h1, &p.wv)?, &s4)?;
    let (qn, kn) = be.rms_qk_norm_forward(&q4, &k4, &p.qn, &p.kn, EPS)?;
    let qr = be.rope_half_split_forward(&qn, cos, sin)?;
    let kr = be.rope_half_split_forward(&kn, cos, sin)?;
    let vb = be.value_residual_blend_forward(&v4, v0, &p.lam)?;
    let qp = be.permute(&qr, &SWAP12)?;
    let kp = be.permute(&kr, &SWAP12)?;
    let vp = be.permute(&vb, &SWAP12)?;
    let a = be.causal_sdpa_forward(&qp, &kp, &vp)?;
    let at = be.permute(&a, &SWAP12)?;
    let ga = be.per_head_sigmoid_gate_forward(&h1, &p.gw, &p.gb, &at)?;
    let o = be.linear_forward(&view(&ga, &[B, T, DM])?, &p.wo)?;
    let x1 = be.residual_add_forward(x, &o)?;
    let h2 = be.rms_norm_forward(&x1, &p.n2, EPS)?;
    let g1 = be.linear_forward(&h2, &p.wg)?;
    let u = be.linear_forward(&h2, &p.wu)?;
    let s = be.silu_forward(&g1)?;
    let m = be.mul_forward(&s, &u)?;
    let dn = be.linear_forward(&m, &p.wd)?;
    let y = be.residual_add_forward(&x1, &dn)?;
    Ok(Fwd {
        h1,
        q4,
        k4,
        v4,
        qp,
        kp,
        vp,
        at,
        ga,
        x1,
        h2,
        g1,
        u,
        s,
        m,
        y,
    })
}

/// Hand-composed backward of `block_forward` (no Tape: neither GPU crate
/// depends on ojas-autograd). A residual add's gradient passes through
/// unchanged, as autograd does, so `residual_add_backward` is not called.
/// Returns `[grad_x, grad_q_proj, grad_ffn_down, grad_gate_w, grad_vr_lambda,
/// grad_norm1, grad_v0]`, the order `torch_rows.py` writes.
#[allow(clippy::too_many_arguments)]
fn block_backward<Bk: Backend>(
    be: &Bk,
    p: &BlockParams,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    v0: &Tensor,
    f: &Fwd,
    gy: &Tensor,
) -> R<Vec<Tensor>> {
    let s4 = [B, T, H, D];
    let s3 = [B, T, DM];
    let (gm, gwd) = be.linear_backward(&f.m, &p.wd, gy)?;
    let (gs, gu) = be.mul_backward(&f.s, &f.u, &gm)?;
    let gg1 = be.silu_backward(&f.g1, &gs)?;
    let (gh2a, _gwg) = be.linear_backward(&f.h2, &p.wg, &gg1)?;
    let (gh2b, _gwu) = be.linear_backward(&f.h2, &p.wu, &gu)?;
    let gh2 = be.residual_add_forward(&gh2a, &gh2b)?;
    let (gx1b, _gn2) = be.rms_norm_backward(&f.x1, &p.n2, &gh2, EPS)?;
    let gx1 = be.residual_add_forward(gy, &gx1b)?;
    let (gga, _gwo) = be.linear_backward(&view(&f.ga, &s3)?, &p.wo, &gx1)?;
    let gg = be.per_head_sigmoid_gate_backward(&f.h1, &p.gw, &p.gb, &f.at, &view(&gga, &s4)?)?;
    let gap = be.permute(&gg.attn_out, &SWAP12)?;
    let (gqp, gkp, gvp) = be.causal_sdpa_backward(&f.qp, &f.kp, &f.vp, &gap)?;
    let gqr = be.permute(&gqp, &SWAP12)?;
    let gkr = be.permute(&gkp, &SWAP12)?;
    let gvb = be.permute(&gvp, &SWAP12)?;
    let gqn = be.rope_half_split_backward(&gqr, cos, sin)?;
    let gkn = be.rope_half_split_backward(&gkr, cos, sin)?;
    let (gq4, gk4, _gqw, _gkw) =
        be.rms_qk_norm_backward(&f.q4, &f.k4, &p.qn, &p.kn, &gqn, &gkn, EPS)?;
    let vr = be.value_residual_blend_backward(&f.v4, v0, &p.lam, &gvb)?;
    let (gh1q, gwq) = be.linear_backward(&f.h1, &p.wq, &view(&gq4, &s3)?)?;
    let (gh1k, _gwk) = be.linear_backward(&f.h1, &p.wk, &view(&gk4, &s3)?)?;
    let (gh1v, _gwv) = be.linear_backward(&f.h1, &p.wv, &view(&vr.value, &s3)?)?;
    let gh1 = be.residual_add_forward(&gg.input, &gh1q)?;
    let gh1 = be.residual_add_forward(&gh1, &gh1k)?;
    let gh1 = be.residual_add_forward(&gh1, &gh1v)?;
    let (gx0, gn1) = be.rms_norm_backward(x, &p.n1, &gh1, EPS)?;
    let gx = be.residual_add_forward(&gx1, &gx0)?;
    Ok(vec![gx, gwq, gwd, gg.weight, vr.lambda, gn1, vr.value0])
}

fn block<Bk: Backend>(r: &mut Runner<'_, Bk>) {
    r.group(&["block_fwd", "block_fwd_bwd"], |r| {
        let specs = block_param_specs();
        let mut ts = Vec::with_capacity(specs.len());
        for (_, s, seed, e) in &specs {
            ts.push(r.dev(s, *seed, *e)?);
        }
        let mut it = ts.into_iter();
        let mut nx = || it.next().expect("block parameter count");
        let p = BlockParams {
            n1: nx(),
            lam: nx(),
            wq: nx(),
            wk: nx(),
            wv: nx(),
            wo: nx(),
            qn: nx(),
            kn: nx(),
            gw: nx(),
            gb: nx(),
            n2: nx(),
            wg: nx(),
            wu: nx(),
            wd: nx(),
        };
        let (xs, cs, vs) = ([B, T, DM], [T, D], [B, T, H, D]);
        let x = r.dev(&xs, 520, 0)?;
        let cos = r.dev(&cs, 521, 0)?;
        let sin = r.dev(&cs, 522, 0)?;
        let v0 = r.dev(&vs, 523, 0)?;
        let gy = r.dev(&xs, 524, 0)?;
        let mut parts: Vec<(&str, &[usize], u64, i32)> = specs
            .iter()
            .map(|(n, s, seed, e)| (*n, s.as_slice(), *seed, *e))
            .collect();
        parts.extend_from_slice(&[
            ("x", &xs[..], 520, 0),
            ("cos", &cs[..], 521, 0),
            ("sin", &cs[..], 522, 0),
            ("v0", &vs[..], 523, 0),
        ]);
        let sf = spec(&parts);
        let sb = format!("{sf};{}", spec(&[("gy", &xs, 524, 0)]));
        let held = RefCell::new(None::<Fwd>);
        r.op("block_fwd", &sf, TOL, |r| {
            let f = block_forward(r.be, &p, &x, &cos, &sin, &v0)?;
            let y = f.y.clone();
            *held.borrow_mut() = Some(f);
            Ok(vec![y])
        });
        held.borrow_mut().take();
        r.op("block_fwd_bwd", &sb, TOL, |r| {
            let f = block_forward(r.be, &p, &x, &cos, &sin, &v0)?;
            let mut out = vec![f.y.clone()];
            out.extend(block_backward(r.be, &p, &x, &cos, &sin, &v0, &f, &gy)?);
            Ok(out)
        });
        Ok(())
    });
}

/// Fail before any row if this binary's generator disagrees with torch's.
pub fn check_generator(refdir: &std::path::Path) -> Result<(), String> {
    let path = refdir.join("generator.f32");
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let theirs: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    // Stream seed 7 at 2^0: the first 64 values, then 64 from index 2^27.
    let mut mine: Vec<u32> = gen(64, 7, 0).iter().map(|v| v.to_bits()).collect();
    let far = 1usize << 27;
    for i in far..far + 64 {
        mine.push(
            (((hash32(i as u64, 7) >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0) * 1.0).to_bits(),
        );
    }
    mine.extend(gen_targets(64, 7));
    if mine != theirs {
        return Err(format!(
            "generator mismatch against {}: the torch and ojas inputs would differ",
            path.display()
        ));
    }
    Ok(())
}

/// Shared `main` body: read the environment, check the generator, run rows.
pub fn main_with<Bk: Backend>(be: &Bk, runtime: &str, device_json: &str) -> Result<(), String> {
    let env = |k: &str| std::env::var(k).ok();
    let refdir = PathBuf::from(env("OJAS_BENCH_REF").ok_or("OJAS_BENCH_REF is not set")?);
    let out_path = env("OJAS_BENCH_OUT").ok_or("OJAS_BENCH_OUT is not set")?;
    let warmup: usize = env("BENCH_WARMUP")
        .map_or(Ok(5), |s| s.parse())
        .map_err(|e| format!("BENCH_WARMUP: {e}"))?;
    let iters: usize = env("BENCH_ITERS")
        .map_or(Ok(20), |s| s.parse())
        .map_err(|e| format!("BENCH_ITERS: {e}"))?;
    if warmup < 5 || iters < 20 {
        return Err(format!(
            "warmup {warmup} / iters {iters}: the protocol needs at least 5 and 20"
        ));
    }
    let filter: Vec<String> = env("BENCH_ROWS")
        .map(|s| {
            s.split(',')
                .filter(|x| !x.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    check_generator(&refdir)?;
    let out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .map_err(|e| format!("{out_path}: {e}"))?;
    let mut r = Runner {
        be,
        runtime: runtime.to_string(),
        host: Budget::new(48 << 30),
        refdir,
        warmup,
        iters,
        filter,
        out,
    };
    r.emit(format!(
        "{{\"runtime\":{},\"row\":\"_device\",\"status\":\"info\",\"device\":{device_json}}}",
        json_str(runtime)
    ));
    run_all(&mut r);
    Ok(())
}

pub fn json_string(s: &str) -> String {
    json_str(s)
}
