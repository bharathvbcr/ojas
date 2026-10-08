//! Deep, structured logging for optimization work — core-level phase timing,
//! per-group gradient norms, optimizer internals, throughput, and memory.
//!
//! Two log surfaces:
//!  1. Console: one compact human line per `log_every` step.
//!  2. `out/metrics.jsonl`: one JSON object per step (append-only) for offline
//!     analysis / plotting.
//!
//! GPU work under Burn is asynchronous: kernels are queued and only forced by a
//! readback or an explicit `sync`. Wall-clock around a phase therefore measures
//! *queueing*, not execution, unless we sync. So phase timing is gated behind
//! `Profiler::synced`: when true (the `--profile` cadence) we `B::sync` at each
//! boundary to attribute real GPU time; when false the timers are inert and the
//! step runs full-async at production speed.
//!
//! Kernel-level timing (per-MSL-kernel, autotune decisions, generated shaders)
//! is handled one level down by CubeCL via `cubecl.toml` / `CUBECL_DEBUG_*`
//! env vars — see the crate README. This module covers the model/optimizer
//! layer that CubeCL cannot see.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use burn::tensor::backend::Backend;

/// Phases of one optimizer step, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    DataPrep,
    Upload,
    Forward,
    Backward,
    GradSplit,
    Clip,
    AdamW,
    Muon,
    Ema,
}

impl Phase {
    pub const ALL: [Phase; 9] = [
        Phase::DataPrep,
        Phase::Upload,
        Phase::Forward,
        Phase::Backward,
        Phase::GradSplit,
        Phase::Clip,
        Phase::AdamW,
        Phase::Muon,
        Phase::Ema,
    ];
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::DataPrep => "data_prep",
            Phase::Upload => "upload",
            Phase::Forward => "forward",
            Phase::Backward => "backward",
            Phase::GradSplit => "grad_split",
            Phase::Clip => "clip",
            Phase::AdamW => "adamw",
            Phase::Muon => "muon",
            Phase::Ema => "ema",
        }
    }
}

/// Accumulates per-phase milliseconds for a single step. When `synced` is set,
/// `enter` forces a device sync so the elapsed time reflects real GPU work.
pub struct Profiler<B: Backend> {
    synced: bool,
    device: B::Device,
    last: Instant,
    current: Option<Phase>,
    ms: [f64; 9],
}

impl<B: Backend> Profiler<B> {
    pub fn new(device: B::Device, synced: bool) -> Self {
        Self {
            synced,
            device,
            last: Instant::now(),
            current: None,
            ms: [0.0; 9],
        }
    }

    #[inline]
    fn idx(p: Phase) -> usize {
        Phase::ALL.iter().position(|q| *q == p).unwrap()
    }

    /// Close the current phase (syncing if enabled) and open a new one.
    pub fn enter(&mut self, phase: Phase) {
        if self.synced {
            let _ = B::sync(&self.device);
        }
        let now = Instant::now();
        if let Some(cur) = self.current {
            self.ms[Self::idx(cur)] += now.duration_since(self.last).as_secs_f64() * 1e3;
        }
        self.current = Some(phase);
        self.last = now;
    }

    /// Close the final phase (syncing if enabled).
    pub fn finish(&mut self) {
        if self.synced {
            let _ = B::sync(&self.device);
        }
        if let Some(cur) = self.current.take() {
            self.ms[Self::idx(cur)] +=
                Instant::now().duration_since(self.last).as_secs_f64() * 1e3;
        }
    }

    pub fn is_synced(&self) -> bool {
        self.synced
    }

    pub fn ms(&self, phase: Phase) -> f64 {
        self.ms[Self::idx(phase)]
    }

    /// `("forward", 12.3)` pairs for every phase with non-zero time.
    pub fn pairs(&self) -> Vec<(&'static str, f64)> {
        Phase::ALL
            .iter()
            .map(|p| (p.as_str(), self.ms[Self::idx(*p)]))
            .filter(|(_, v)| *v > 0.0)
            .collect()
    }
}

/// One step's worth of metrics. Everything the console line and JSONL share.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f64,
    pub grad_norm_global: f64,
    pub grad_norm_muon: f64,
    pub grad_norm_embed: f64,
    pub grad_norm_scalar: f64,
    pub clip_factor: f64,
    pub lr_mul: f64,
    pub momentum: f64,
    pub tokens_per_s: f64,
    pub step_ms: f64,
    pub profiled: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub phase_ms: Vec<(&'static str, f64)>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub muon_ns5_ms: Vec<(&'static str, f64)>,
    pub rss_mb: f64,
}

/// Appends `StepMetrics` to `out/metrics.jsonl` and prints a compact console
/// line at the configured cadence.
pub struct MetricsLogger {
    writer: Option<BufWriter<File>>,
}

impl MetricsLogger {
    pub fn new(out_dir: &Path) -> Self {
        let path = out_dir.join("metrics.jsonl");
        let writer = File::create(&path)
            .map(BufWriter::new)
            .map_err(|e| eprintln!("warn: cannot open {path:?} for metrics: {e}"))
            .ok();
        Self { writer }
    }

    pub fn log(&mut self, m: &StepMetrics) {
        if let Some(w) = self.writer.as_mut() {
            if let Ok(line) = serde_json::to_string(m) {
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
            }
        }
    }

    /// Human console line. Includes the phase breakdown only on profiled steps.
    pub fn console(&self, m: &StepMetrics) {
        let base = format!(
            "step {:>5} | loss {:.4} | gnorm {:.3} (clip x{:.3}) | lr {:.3} mom {:.3} | {:.0} tok/s | {:.1} ms/step | rss {:.0} MB",
            m.step, m.loss, m.grad_norm_global, m.clip_factor, m.lr_mul, m.momentum,
            m.tokens_per_s, m.step_ms, m.rss_mb,
        );
        if m.profiled && !m.phase_ms.is_empty() {
            let phases = m
                .phase_ms
                .iter()
                .map(|(k, v)| format!("{k} {v:.1}"))
                .collect::<Vec<_>>()
                .join(" · ");
            let ns5 = if m.muon_ns5_ms.is_empty() {
                String::new()
            } else {
                let s = m
                    .muon_ns5_ms
                    .iter()
                    .map(|(k, v)| format!("{k} {v:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("\n         ns5[{s}]")
            };
            println!("{base}\n         phases[{phases}]{ns5}");
        } else {
            println!("{base}");
        }
    }
}

/// Resident set size of this process in bytes (macOS: ru_maxrss is already in
/// bytes; Linux would be KiB, but this crate targets the M5).
pub fn mem_rss_bytes() -> u64 {
    // SAFETY: getrusage with a zeroed rusage is always sound.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as u64
        } else {
            0
        }
    }
}

/// Startup device / backend report. Printed once so every run's log records
/// exactly what hardware and backend produced it.
pub fn device_report<B: Backend>(device: &B::Device) -> String {
    format!("backend={} | device={:?}", B::name(device), device)
}
