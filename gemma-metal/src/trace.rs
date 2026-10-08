//! Env-gated decode trace (Phase 0). Zero cost when off.
//!
//! | Knob | Behavior |
//! |------|----------|
//! | `GEMMA_METAL_TRACE=1` | Per-op host µs + stderr rollup |
//! | `GEMMA_METAL_TRACE=json` | JSONL per token → `bench/results/trace_*.jsonl` |
//! | `GEMMA_METAL_TRACE=sync` | `synchronize()` after each named stage (diagnostic) |
//! | Off (default) | No Instant / no format / no IO |
//!
//! Gate claims must re-run with trace **off**.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use tessl::infer_trace;

/// 0=off, 1=host, 2=json, 3=sync
static MODE: AtomicU8 = AtomicU8::new(0);
static CLI_OVERRIDE: AtomicU8 = AtomicU8::new(255); // 255 = unset

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceMode {
    Off = 0,
    Host = 1,
    Json = 2,
    Sync = 3,
}

impl TraceMode {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Host,
            2 => Self::Json,
            3 => Self::Sync,
            _ => Self::Off,
        }
    }

    fn as_u8(self) -> u8 {
        self as u8
    }
}

fn parse_mode(s: &str) -> TraceMode {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "host" => TraceMode::Host,
        "json" | "jsonl" => TraceMode::Json,
        "sync" | "gpu" => TraceMode::Sync,
        _ => TraceMode::Off,
    }
}

/// Apply CLI override (`bench --trace` / `--trace-json` / `--trace-sync`). Call once at process start.
pub fn set_cli_mode(mode: TraceMode) {
    CLI_OVERRIDE.store(mode.as_u8(), Ordering::Relaxed);
    MODE.store(mode.as_u8(), Ordering::Relaxed);
    infer_trace::set_enabled(mode != TraceMode::Off);
}

fn init_mode() -> TraceMode {
    let cli = CLI_OVERRIDE.load(Ordering::Relaxed);
    if cli != 255 {
        let m = TraceMode::from_u8(cli);
        infer_trace::set_enabled(m != TraceMode::Off);
        return m;
    }
    let m = std::env::var("GEMMA_METAL_TRACE")
        .ok()
        .map(|s| parse_mode(&s))
        .unwrap_or(TraceMode::Off);
    MODE.store(m.as_u8(), Ordering::Relaxed);
    infer_trace::set_enabled(m != TraceMode::Off);
    m
}

pub fn mode() -> TraceMode {
    static INIT: OnceLock<TraceMode> = OnceLock::new();
    *INIT.get_or_init(init_mode)
}

pub fn enabled() -> bool {
    mode() != TraceMode::Off
}

#[derive(Clone, Debug)]
pub struct TraceOp {
    pub name: String,
    pub host_us: u64,
    pub bytes: u64,
    pub rows: u32,
    pub cols: u32,
}

#[derive(Clone, Debug, Default)]
pub struct TraceTokenRollup {
    pub tok: u64,
    pub host_encode_us: u64,
    pub gpu_wait_us: u64,
    pub sync_us: u64,
    pub dispatches: u64,
    pub barriers: u64,
    pub commits: u64,
    pub cold_allocs: u64,
    pub bytes_est: u64,
    pub top_ops: Vec<TraceOp>,
    pub flags: TraceFlags,
}

#[derive(Clone, Debug, Default)]
pub struct TraceFlags {
    pub hazard_barriers_auto: bool,
    pub ple: bool,
    pub async_encode: bool,
}

/// Active token span bookkeeping.
pub struct TraceSession {
    mode: TraceMode,
    token_idx: u64,
    token_t0: Option<Instant>,
    ops: Vec<TraceOp>,
    bytes_est: u64,
    sync_us: u64,
    jsonl: Option<Mutex<File>>,
    jsonl_path: Option<PathBuf>,
    /// Aggregate host_us by op name across all tokens (for final table).
    agg: HashMap<String, (u64, u64)>,
    tokens_logged: u64,
}

impl TraceSession {
    pub fn new() -> Self {
        let mode = mode();
        let (jsonl, jsonl_path) = if mode == TraceMode::Json {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/results");
            let _ = fs::create_dir_all(&dir);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let path = dir.join(format!("trace_{ts}.jsonl"));
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok();
            (
                f.map(Mutex::new),
                Some(path),
            )
        } else {
            (None, None)
        };
        Self {
            mode,
            token_idx: 0,
            token_t0: None,
            ops: Vec::new(),
            bytes_est: 0,
            sync_us: 0,
            jsonl,
            jsonl_path,
            agg: HashMap::new(),
            tokens_logged: 0,
        }
    }

    pub fn mode(&self) -> TraceMode {
        self.mode
    }

    pub fn jsonl_path(&self) -> Option<&PathBuf> {
        self.jsonl_path.as_ref()
    }

    pub fn begin_token(&mut self) {
        if self.mode == TraceMode::Off {
            return;
        }
        infer_trace::reset_token_counters();
        self.ops.clear();
        self.bytes_est = 0;
        self.sync_us = 0;
        self.token_t0 = Some(Instant::now());
    }

    /// Named stage. `bytes` = estimated traffic; optional `rows`/`cols` for GEMV.
    pub fn span<R>(
        &mut self,
        name: &str,
        bytes: u64,
        rows: u32,
        cols: u32,
        sync_gpu: Option<&dyn Fn() -> Result<(), String>>,
        f: impl FnOnce() -> R,
    ) -> R {
        if self.mode == TraceMode::Off {
            return f();
        }
        let t0 = Instant::now();
        let out = f();
        let mut host_us = t0.elapsed().as_micros() as u64;
        if self.mode == TraceMode::Sync {
            if let Some(sync) = sync_gpu {
                let s0 = Instant::now();
                let _ = sync();
                let su = s0.elapsed().as_micros() as u64;
                self.sync_us += su;
                host_us += su;
            }
        }
        self.record_op(name, host_us, bytes, rows, cols);
        out
    }

    /// After encoding a bucket of dispatches, optionally synchronize and attribute GPU wait.
    /// Used by `GEMMA_METAL_TRACE=sync` to split layer / lm_head / softcap time.
    pub fn flush_gpu_bucket(
        &mut self,
        name: &str,
        bytes: u64,
        sync_gpu: Option<&dyn Fn() -> Result<(), String>>,
    ) {
        if self.mode == TraceMode::Off {
            return;
        }
        let mut host_us = 0u64;
        if self.mode == TraceMode::Sync {
            if let Some(sync) = sync_gpu {
                let s0 = Instant::now();
                let _ = sync();
                let su = s0.elapsed().as_micros() as u64;
                self.sync_us += su;
                host_us = su;
            }
        }
        self.record_op(name, host_us, bytes, 0, 0);
    }

    fn record_op(&mut self, name: &str, host_us: u64, bytes: u64, rows: u32, cols: u32) {
        self.bytes_est += bytes;
        self.ops.push(TraceOp {
            name: name.to_string(),
            host_us,
            bytes,
            rows,
            cols,
        });
        let e = self.agg.entry(name.to_string()).or_insert((0, 0));
        e.0 += host_us;
        e.1 += bytes;
    }

    pub fn end_token(&mut self, flags: TraceFlags) -> Option<TraceTokenRollup> {
        if self.mode == TraceMode::Off {
            return None;
        }
        let host_encode_us = self
            .token_t0
            .map(|t| t.elapsed().as_micros() as u64)
            .unwrap_or(0);
        let snap = infer_trace::snapshot();
        let mut top = self.ops.clone();
        top.sort_by(|a, b| b.host_us.cmp(&a.host_us));
        top.truncate(12);
        let roll = TraceTokenRollup {
            tok: self.token_idx,
            host_encode_us,
            gpu_wait_us: snap.sync_wait_us,
            sync_us: self.sync_us,
            dispatches: snap.dispatches,
            barriers: snap.barriers,
            commits: snap.commits,
            cold_allocs: snap.cold_allocs,
            bytes_est: self.bytes_est,
            top_ops: top,
            flags,
        };
        self.token_idx += 1;
        self.tokens_logged += 1;

        if self.mode == TraceMode::Host || self.mode == TraceMode::Sync {
            eprintln!(
                "[trace] tok={} encode={}µs wait={}µs sync={}µs disp={} bar={} commit={} cold={} bytes≈{:.2}GiB",
                roll.tok,
                roll.host_encode_us,
                roll.gpu_wait_us,
                roll.sync_us,
                roll.dispatches,
                roll.barriers,
                roll.commits,
                roll.cold_allocs,
                roll.bytes_est as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            for op in roll.top_ops.iter().take(5) {
                eprintln!(
                    "  {:>18} {:>6}µs  ~{:.1} MiB",
                    op.name,
                    op.host_us,
                    op.bytes as f64 / (1024.0 * 1024.0)
                );
            }
        }
        if let Some(ref f) = self.jsonl {
            if let Ok(mut g) = f.lock() {
                let _ = writeln!(g, "{}", rollup_json(&roll));
            }
        }
        Some(roll)
    }

    /// Sorted aggregate ms table after N tokens.
    pub fn print_summary_table(&self) {
        if self.mode == TraceMode::Off || self.tokens_logged == 0 {
            return;
        }
        let mut rows: Vec<_> = self.agg.iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        eprintln!();
        eprintln!(
            "[trace] aggregate over {} tokens (host encode µs / est bytes):",
            self.tokens_logged
        );
        eprintln!("  {:>18} {:>10} {:>12}", "op", "ms_total", "MiB_est");
        for (name, (us, bytes)) in rows.iter().take(20) {
            eprintln!(
                "  {:>18} {:>10.2} {:>12.1}",
                name,
                *us as f64 / 1000.0,
                *bytes as f64 / (1024.0 * 1024.0)
            );
        }
    }
}

impl Default for TraceSession {
    fn default() -> Self {
        Self::new()
    }
}

fn rollup_json(r: &TraceTokenRollup) -> String {
    let tops: Vec<String> = r
        .top_ops
        .iter()
        .map(|o| {
            format!(
                "{{\"name\":\"{}\",\"host_us\":{},\"bytes\":{}}}",
                o.name, o.host_us, o.bytes
            )
        })
        .collect();
    format!(
        "{{\"tok\":{},\"host_encode_us\":{},\"gpu_wait_us\":{},\"sync_us\":{},\"dispatches\":{},\"barriers\":{},\"commits\":{},\"cold_allocs\":{},\"bytes_est\":{},\"top_ops\":[{}],\"flags\":{{\"hazard_barriers_auto\":{},\"ple\":{},\"async\":{}}}}}",
        r.tok,
        r.host_encode_us,
        r.gpu_wait_us,
        r.sync_us,
        r.dispatches,
        r.barriers,
        r.commits,
        r.cold_allocs,
        r.bytes_est,
        tops.join(","),
        r.flags.hazard_barriers_auto,
        r.flags.ple,
        r.flags.async_encode
    )
}

/// Estimated bytes for a Q4 GEMV (packed nibbles + f32 scales/biases + x + y).
pub fn gemv_bytes_est(rows: u32, cols: u32, group_size: u32) -> u64 {
    let packed = (rows as u64) * (cols as u64) / 2;
    let groups = if group_size == 0 {
        0
    } else {
        (cols / group_size) as u64
    };
    let scales = (rows as u64) * groups * 4 * 2; // scale + bias
    let x = cols as u64 * 4;
    let y = rows as u64 * 4;
    packed + scales + x + y
}
