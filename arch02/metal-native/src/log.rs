//! JSONL metrics logger (burn-port `log.rs` shape, no Burn dependency).
//!
//! # Profiling (Instruments)
//!
//! [`Profiler`] emits lightweight `os_signpost` intervals (subsystem
//! `com.parameter-golf.metal-native`, category `train`) for each
//! [`Phase`] when profiling is enabled. In Instruments:
//! - **Time Profiler / Points of Interest** — CPU phase boundaries
//!   (`data_prep`, `upload`, `forward`, `backward`, `clip`, `optim`).
//! - **Metal System Trace** — enable the **Neural Accelerators** (NAX)
//!   utilization counter to see whether TensorOps GEMMs feed the M5
//!   accelerators (ground truth for bf16 / tf32 experiments).
//! - Do **not** trust legacy `MTLCounterSampleBuffer` on macOS 26 (returns
//!   zeros). In-app GPU timestamps use `MTL4CounterHeap` on the Metal 4-only
//!   training encode path (DECISIONS M3: classic M3 CBs removed).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::checkpoint::DivergenceNorms;
use crate::research::ResearchTelemetry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    DataPrep,
    Upload,
    Forward,
    Backward,
    Clip,
    Optim,
}

impl Phase {
    pub const ALL: [Phase; 6] = [
        Phase::DataPrep,
        Phase::Upload,
        Phase::Forward,
        Phase::Backward,
        Phase::Clip,
        Phase::Optim,
    ];
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::DataPrep => "data_prep",
            Phase::Upload => "upload",
            Phase::Forward => "forward",
            Phase::Backward => "backward",
            Phase::Clip => "clip",
            Phase::Optim => "optim",
        }
    }
}

pub struct Profiler {
    synced: bool,
    last: Instant,
    current: Option<Phase>,
    ms: [f64; 6],
    sync_fn: Option<Box<dyn FnMut()>>,
}

impl Profiler {
    pub fn new(synced: bool) -> Self {
        Self {
            synced,
            last: Instant::now(),
            current: None,
            ms: [0.0; 6],
            sync_fn: None,
        }
    }

    pub fn with_sync<F: FnMut() + 'static>(mut self, f: F) -> Self {
        self.sync_fn = Some(Box::new(f));
        self
    }

    fn idx(p: Phase) -> usize {
        Phase::ALL.iter().position(|q| *q == p).unwrap()
    }

    pub fn enter(&mut self, phase: Phase) {
        // Instruments: host phase boundary (label = Phase::as_str). With
        // `synced`, synchronize before the stamp so wall time ≈ GPU phase end.
        // See `signpost` module + README "Profiling" for NAX counter notes.
        let _ = crate::signpost::instruments_label(phase.as_str());
        if self.synced {
            if let Some(f) = self.sync_fn.as_mut() {
                f();
            }
        }
        let now = Instant::now();
        if let Some(cur) = self.current {
            self.ms[Self::idx(cur)] += now.duration_since(self.last).as_secs_f64() * 1e3;
        }
        self.current = Some(phase);
        self.last = now;
    }

    pub fn finish(&mut self) {
        if self.synced {
            if let Some(f) = self.sync_fn.as_mut() {
                f();
            }
        }
        if let Some(cur) = self.current.take() {
            self.ms[Self::idx(cur)] += Instant::now().duration_since(self.last).as_secs_f64() * 1e3;
        }
    }

    pub fn pairs(&self) -> Vec<(&'static str, f64)> {
        Phase::ALL
            .iter()
            .map(|p| (p.as_str(), self.ms[Self::idx(*p)]))
            .filter(|(_, v)| *v > 0.0)
            .collect()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f64,
    pub grad_norm_global: f64,
    pub clip_factor: f64,
    pub lr_mul: f64,
    pub momentum: f64,
    pub tokens_per_s: f64,
    pub step_ms: f64,
    pub optimizer_ms: f64,
    pub dispatches: usize,
    pub profiled: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub phase_ms: Vec<(&'static str, f64)>,
    pub rss_mb: f64,
    pub current_physical_mb: f64,
    pub precision: String,
    pub tok_mult: usize,
    /// Scalar / bank / Muon-momentum L2 norms (Phase 0 divergence bisect).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub divergence: Option<DivergenceNorms>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub research: Option<ResearchTelemetry>,
}

pub struct MetricsLogger {
    writer: Option<BufWriter<File>>,
}

impl MetricsLogger {
    pub fn new(out_dir: &Path) -> Self {
        let _ = std::fs::create_dir_all(out_dir);
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

    pub fn console(&self, m: &StepMetrics) {
        let base = format!(
            "step {:>5} | loss {:.4} | gnorm {:.3} (clip x{:.3}) | lr {:.3} mom {:.3} | {:.0} tok/s | {:.1} ms/step (optim {:.1}) | disp {} | phys {:.0} MB",
            m.step,
            m.loss,
            m.grad_norm_global,
            m.clip_factor,
            m.lr_mul,
            m.momentum,
            m.tokens_per_s,
            m.step_ms,
            m.optimizer_ms,
            m.dispatches,
            m.current_physical_mb,
        );
        if m.profiled && !m.phase_ms.is_empty() {
            let phases = m
                .phase_ms
                .iter()
                .map(|(k, v)| format!("{k} {v:.1}"))
                .collect::<Vec<_>>()
                .join(" · ");
            println!("{base}\n         phases[{phases}]");
        } else {
            println!("{base}");
        }
        if let Some(ref d) = m.divergence {
            let fmt = |xs: &[f64]| {
                xs.iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            println!(
                "         div | resid_mix[{}] vr_λ[{}] attn_scale[{}] smear {:.4} | banks qo/kv/up/dn {:.4}/{:.4}/{:.4}/{:.4} | mom {:.4}/{:.4}/{:.4}/{:.4}",
                fmt(&d.resid_mix),
                fmt(&d.vr_lambda),
                fmt(&d.attn_scale),
                d.smear_gate,
                d.bank_qo,
                d.bank_kv,
                d.bank_mlp_up,
                d.bank_mlp_down,
                d.mom_qo,
                d.mom_kv,
                d.mom_mlp_up,
                d.mom_mlp_down,
            );
        }
    }
}

pub fn mem_rss_mb() -> f64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct RusageInfoV2 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
}

#[cfg(target_os = "macos")]
#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut libc::c_void) -> i32;
}

/// Current Darwin physical footprint, unlike `ru_maxrss` which is a high-water mark.
pub fn mem_current_physical_mb() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut info: RusageInfoV2 = std::mem::zeroed();
        if proc_pid_rusage(
            libc::getpid(),
            2, // RUSAGE_INFO_V2
            (&mut info as *mut RusageInfoV2).cast(),
        ) == 0
        {
            return info.phys_footprint as f64 / (1024.0 * 1024.0);
        }
    }
    0.0
}
