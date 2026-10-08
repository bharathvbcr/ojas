//! Structured diagnostic logging for download / load / Hot upload / decode triage.
//!
//! **Always on by default** so failed loads surface a clear trail. Silence with
//! `GEMMA_METAL_LOG=0` (also accepts `off` / `false` / `no`).
//!
//! Prefixes: `[gemma-metal:weights]`, `[gemma-metal:gpu]`, `[gemma-metal:kernels]`,
//! `[gemma-metal:bench]`, `[gemma-metal:serve]`, `[gemma-metal:mtp]`,
//! `[gemma-metal:quant]`, `[gemma-metal:ple]`, `[gemma-metal:cache]`,
//! `[gemma-metal:infer]`.
//!
//! Fine-grained *decode* per-op µs rollups remain under [`crate::trace`]
//! (`GEMMA_METAL_TRACE`). Line-level inference enter/exit for every host dispatch
//! uses [`infer_enabled`] / [`InferScope`] (`GEMMA_METAL_INFER_LOG`).

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(true);
static INIT: OnceLock<()> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();

static INFER_ENABLED: AtomicBool = AtomicBool::new(false);
static INFER_INIT: OnceLock<()> = OnceLock::new();

fn parse_enabled(s: &str) -> bool {
    match s.trim().to_ascii_lowercase().as_str() {
        "0" | "off" | "false" | "no" | "quiet" => false,
        _ => true,
    }
}

fn ensure_init() {
    INIT.get_or_init(|| {
        let on = std::env::var("GEMMA_METAL_LOG")
            .ok()
            .map(|s| parse_enabled(&s))
            .unwrap_or(true);
        ENABLED.store(on, Ordering::Relaxed);
        let _ = START.get_or_init(Instant::now);
        if on {
            let elapsed = 0.0f64;
            eprintln!(
                "[gemma-metal:diag] Elapsed={elapsed:.3}s logging ON \
                 (default; set GEMMA_METAL_LOG=0 to silence; \
                 GEMMA_METAL_INFER_LOG=0 to silence line-level decode; \
                 GEMMA_METAL_TRACE=1 for per-op rollup)"
            );
        }
    });
}

fn ensure_infer_init() {
    INFER_INIT.get_or_init(|| {
        let _ = START.get_or_init(Instant::now);
        let on = match std::env::var("GEMMA_METAL_INFER_LOG") {
            Ok(s) => parse_enabled(&s),
            // Tests drown in per-op lines unless explicitly opted in.
            Err(_) if cfg!(test) => false,
            // Bins / library use outside tests: ON for debugging (silence with =0).
            Err(_) => true,
        };
        INFER_ENABLED.store(on, Ordering::Relaxed);
        if on {
            eprintln!(
                "[gemma-metal:infer] Elapsed={:.3}s line-level inference log ON \
                 (GEMMA_METAL_INFER_LOG; set =0 to silence)",
                elapsed_s()
            );
        }
    });
}

/// Force-init (call from `bench` / `serve` `main` so the banner appears early).
pub fn init() {
    ensure_init();
    ensure_infer_init();
}

pub fn enabled() -> bool {
    ensure_init();
    ENABLED.load(Ordering::Relaxed)
}

/// Line-level inference logging (`GEMMA_METAL_INFER_LOG`).
///
/// Default **ON** for bins / non-test; **OFF** under `cfg!(test)` unless the
/// env var is explicitly set. Set `GEMMA_METAL_INFER_LOG=0` to silence.
pub fn infer_enabled() -> bool {
    ensure_infer_init();
    INFER_ENABLED.load(Ordering::Relaxed)
}

/// Process-relative elapsed seconds since first diag touch.
pub fn elapsed_s() -> f64 {
    ensure_init();
    START
        .get()
        .map(|t| t.elapsed().as_secs_f64())
        .unwrap_or(0.0)
}

/// Human-readable byte count (`1.5 GiB`, `256 MiB`, …).
pub fn fmt_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let x = n as f64;
    if x >= GIB {
        format!("{:.2} GiB", x / GIB)
    } else if x >= MIB {
        format!("{:.1} MiB", x / MIB)
    } else if x >= KIB {
        format!("{:.1} KiB", x / KIB)
    } else {
        format!("{n} B")
    }
}

/// Best-effort self RSS in MiB (macOS `ps`; None if unavailable).
pub fn rss_mib() -> Option<f64> {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let kb: f64 = s.trim().parse().ok()?;
    Some(kb / 1024.0)
}

/// Log a line: `[gemma-metal:{component}] Elapsed=…s …`.
pub fn log(component: &str, args: fmt::Arguments<'_>) {
    if !enabled() {
        return;
    }
    eprintln!(
        "[gemma-metal:{component}] Elapsed={:.3}s {args}",
        elapsed_s()
    );
}

/// Log under `[gemma-metal:infer]` when [`infer_enabled`].
pub fn infer_log(args: fmt::Arguments<'_>) {
    if !infer_enabled() {
        return;
    }
    eprintln!(
        "[gemma-metal:infer] Elapsed={:.3}s {args}",
        elapsed_s()
    );
}

/// CPU wait / Device sync — always labeled as a potential stall when infer-log is on.
pub fn infer_stall(context: &str) {
    infer_log(format_args!("⚠ STALL {context} (CPU wait on GPU)"));
}

/// Log an error with Display (and Debug when `T: Debug`).
pub fn err(component: &str, context: &str, e: &impl fmt::Debug) {
    if !enabled() {
        return;
    }
    eprintln!(
        "[gemma-metal:{component}] Elapsed={:.3}s ERROR {context}: {e:?}",
        elapsed_s()
    );
    if infer_enabled() {
        eprintln!(
            "[gemma-metal:infer] Elapsed={:.3}s ERROR {context}: {e:?}",
            elapsed_s()
        );
    }
}

/// Log an error from a Display-only type.
pub fn err_msg(component: &str, context: &str, e: &impl fmt::Display) {
    if !enabled() {
        return;
    }
    eprintln!(
        "[gemma-metal:{component}] Elapsed={:.3}s ERROR {context}: {e}",
        elapsed_s()
    );
    if infer_enabled() {
        eprintln!(
            "[gemma-metal:infer] Elapsed={:.3}s ERROR {context}: {e}",
            elapsed_s()
        );
    }
}

/// Timed stage helper — logs start + end with elapsed for the stage.
pub struct Stage {
    component: &'static str,
    name: String,
    t0: Instant,
}

impl Stage {
    pub fn begin(component: &'static str, name: impl Into<String>) -> Self {
        let name = name.into();
        log(component, format_args!("▶ {name}"));
        Self {
            component,
            name,
            t0: Instant::now(),
        }
    }

    pub fn ok(self) {
        let ms = self.t0.elapsed().as_secs_f64() * 1e3;
        log(
            self.component,
            format_args!("✔ {} done in {ms:.1} ms", self.name),
        );
    }

    pub fn fail(self, e: &impl fmt::Display) {
        let ms = self.t0.elapsed().as_secs_f64() * 1e3;
        err_msg(
            self.component,
            &format!("{} FAILED after {ms:.1} ms", self.name),
            e,
        );
    }
}

/// RAII timer for one inference op. Logs `▶` on construct and `◀` on drop when
/// [`infer_enabled`]. Zero format/IO cost when off (only an atomic load).
pub struct InferScope {
    op: String,
    active: bool,
    t0: Instant,
}

impl InferScope {
    /// Start a named op. `detail` is formatted immediately only when logging is on.
    /// When inactive, skips `op.into()` allocation.
    pub fn begin(op: impl Into<String>, detail: impl fmt::Display) -> Self {
        if !infer_enabled() {
            return Self {
                op: String::new(),
                active: false,
                t0: Instant::now(),
            };
        }
        let op = op.into();
        infer_log(format_args!("▶ {op} | {detail}"));
        Self {
            op,
            active: true,
            t0: Instant::now(),
        }
    }

    /// Mark success with an extra note (still logs duration on drop if not consumed).
    pub fn note(&self, detail: impl fmt::Display) {
        if self.active {
            infer_log(format_args!("· {} | {detail}", self.op));
        }
    }
}

impl Drop for InferScope {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let us = self.t0.elapsed().as_micros();
        infer_log(format_args!("◀ {} done {us}µs", self.op));
    }
}

/// Run `f` under an [`InferScope`] (start + end lines when infer-log is on).
#[inline]
pub fn infer_op<R>(op: &str, detail: impl fmt::Display, f: impl FnOnce() -> R) -> R {
    let _scope = InferScope::begin(op, detail);
    f()
}

/// Convenience macro: `trace_op!("name", "detail…", { … })`.
///
/// Always evaluates the body; `$detail` (often `format!(…)`) only when
/// [`infer_enabled`].
#[macro_export]
macro_rules! trace_op {
    ($name:expr, $detail:expr, $body:block) => {{
        let _infer_scope = if $crate::diag::infer_enabled() {
            Some($crate::diag::InferScope::begin($name, $detail))
        } else {
            None
        };
        $body
    }};
}
