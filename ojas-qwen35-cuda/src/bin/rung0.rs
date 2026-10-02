//! Rung 0: the toolchain smoke test on the GH200 (`cuda-backend-scoping.md`
//! §5.3 row 0; milestone M0, §6.1).
//!
//! In order, each step recorded in one JSON report at `<out>/rung0-report.json`:
//! 1. probe `libcuda`, `libnvrtc`, `libcublas` (refuse by name if any is
//!    missing), open the device, refuse anything but sm_90, and record the
//!    driver, NVRTC and cuBLAS versions, SM count and cuBLAS modes;
//! 2. compile every kernel module through NVRTC (and one at `compute_90a`);
//! 3. run every K0 kernel against its host reference, bitwise, twice;
//! 4. probe `cublasGemmEx` bf16 x bf16 -> f32 (the falsifier for
//!    `GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01`);
//! 5. run K1 (FFMA ExactF32, FFMA bf16, cuBLAS bf16) against the float64
//!    host on tessl's ragged shapes, plus repeat runs.
//!
//! A watchdog enforces the wall-clock cap (default 280 s, at most 300):
//! at the cap it writes the report with what finished and exits 3. Exit 0
//! means every check ran and passed; 1 a check failed or panicked; 2 a
//! refusal before any check; 3 the cap.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ojas_qwen35_cuda::check::{overall, panic_text, Check, Status};
use ojas_qwen35_cuda::gemm::{Accumulate, Bf16Engine, GemmSpec, Operands};
use ojas_qwen35_cuda::gemm_plan::{GemmLayout, GemmShape};
use ojas_qwen35_cuda::json::{Json, JsonObj};
use ojas_qwen35_cuda::libprobe;
use ojas_qwen35_cuda::rung0_cli::{
    parse, prepare_out, write_atomic, Args, EXIT_FAIL, EXIT_PASS, EXIT_REFUSED, EXIT_TIMEOUT, USAGE,
};
use ojas_qwen35_cuda::runtime::{CudaRuntime, RuntimeConfig};
use ojas_qwen35_cuda::smoke::{self, GemmCase};
use ojas_qwen35_cuda::CudaError;

/// Repetitions of the rung-0 determinism runs (the device tests run 25).
const RUNG0_DETERMINISM_REPS: usize = 5;

struct Report {
    header: JsonObj,
    device: Option<Json>,
    refusal: Option<Json>,
    checks: Vec<Check>,
    extra: JsonObj,
    phase: String,
    written: bool,
}

impl Report {
    fn render(&self, status: &str, exit_code: i32, elapsed: Duration) -> String {
        let count = |s: Status| self.checks.iter().filter(|c| c.status == s).count();
        let summary = JsonObj::new()
            .with("total", self.checks.len())
            .with("pass", count(Status::Pass))
            .with("fail", count(Status::Fail))
            .with("panicked", count(Status::Panicked))
            .with("not_run", count(Status::NotRun));
        let mut obj = self.header.clone();
        obj.push("device", self.device.clone());
        obj.push("refusal", self.refusal.clone());
        obj.push("extra", self.extra.clone());
        obj.push("phase_at_end", self.phase.as_str());
        obj.push(
            "checks",
            Json::Arr(self.checks.iter().map(Check::to_json).collect()),
        );
        obj.push("summary", summary);
        obj.push("status", status);
        obj.push("exit_code", exit_code);
        obj.push("elapsed_s", elapsed.as_secs_f64());
        let mut text = Json::from(obj).render();
        text.push('\n');
        text
    }
}

fn lock(state: &Mutex<Report>) -> MutexGuard<'_, Report> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn header(args: &Args) -> JsonObj {
    let env = |k: &str| std::env::var(k).ok();
    let exe = std::env::current_exe().ok();
    let exe_bytes = exe
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len());
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    JsonObj::new()
        .with("kind", "ojas-qwen35-cuda.rung0")
        .with("schema", 1u32)
        .with("quick", true)
        .with(
            "quick_reason",
            "rung-0 toolchain smoke: one process, small shapes, no training step",
        )
        .with("crate_version", env!("CARGO_PKG_VERSION"))
        .with(
            "binary",
            JsonObj::new()
                .with("path", exe.map(|p| p.display().to_string()))
                .with("bytes", exe_bytes),
        )
        .with("args", std::env::args().skip(1).collect::<Vec<String>>())
        .with("out", args.out.display().to_string())
        .with("cap_s", args.cap.as_secs())
        .with("started_unix_s", started)
        .with(
            "env",
            JsonObj::new()
                .with("LD_LIBRARY_PATH", env("LD_LIBRARY_PATH"))
                .with("NVIDIA_TF32_OVERRIDE", env("NVIDIA_TF32_OVERRIDE"))
                .with("CUDA_VISIBLE_DEVICES", env("CUDA_VISIBLE_DEVICES"))
                .with(
                    "hostname",
                    std::fs::read_to_string("/etc/hostname")
                        .ok()
                        .map(|h| h.trim().to_string()),
                ),
        )
}

/// How a report write went.
#[derive(Debug, PartialEq, Eq)]
enum Written {
    /// This call wrote it.
    Now,
    /// It was already written (by main or the watchdog).
    Already,
    /// The write failed; the reason is on stderr.
    Failed,
}

/// Write the report once; whoever gets here first (main or watchdog) writes.
fn write_once(
    state: &mut Report,
    path: &Path,
    status: &str,
    code: i32,
    elapsed: Duration,
) -> Written {
    if state.written {
        return Written::Already;
    }
    let text = state.render(status, code, elapsed);
    match write_atomic(path, &text) {
        Ok(()) => {
            state.written = true;
            Written::Now
        }
        Err(e) => {
            eprintln!("rung0: could not write the report: {e}");
            Written::Failed
        }
    }
}

/// What the watchdog does when the cap expires: record the cap as a failed
/// check and write the report, unless main already wrote it. Returns the exit
/// code, or `None` when main finished first. A failed write still returns
/// the timeout code: the cap holds whether or not the report landed. If the
/// report lock stays held for `lock_wait` (main never holds it across a
/// device call, so this means something is wrong), a minimal report is
/// written without it.
fn fire(
    state: &Mutex<Report>,
    path: &Path,
    cap: Duration,
    started: Instant,
    lock_wait: Duration,
) -> Option<i32> {
    let give_up = Instant::now() + lock_wait;
    loop {
        let guard = match state.try_lock() {
            Ok(g) => Some(g),
            Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        };
        if let Some(mut s) = guard {
            if s.written {
                return None;
            }
            let phase = s.phase.clone();
            s.checks.push(Check::fail(
                "rung0.wall_clock_cap",
                format!("the {} s cap expired during {phase}", cap.as_secs()),
            ));
            eprintln!(
                "rung0: TIMEOUT after {} s during {phase}; report {}",
                cap.as_secs(),
                path.display()
            );
            write_once(&mut s, path, "timeout", EXIT_TIMEOUT, started.elapsed());
            return Some(EXIT_TIMEOUT);
        }
        if Instant::now() >= give_up {
            let text = format!(
                "{{\"kind\":\"ojas-qwen35-cuda.rung0\",\"status\":\"timeout\",\"exit_code\":{EXIT_TIMEOUT},\"detail\":\"report lock unavailable at the cap\"}}\n"
            );
            if let Err(e) = write_atomic(path, &text) {
                eprintln!("rung0: could not write the timeout report: {e}");
            }
            return Some(EXIT_TIMEOUT);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn spawn_watchdog(state: Arc<Mutex<Report>>, path: PathBuf, cap: Duration, started: Instant) {
    std::thread::spawn(move || {
        std::thread::sleep(cap.saturating_sub(started.elapsed()));
        if let Some(code) = fire(&state, &path, cap, started, Duration::from_secs(2)) {
            std::process::exit(code);
        }
    });
}

fn record(state: &Mutex<Report>, checks: Vec<Check>) {
    for c in &checks {
        println!(
            "{:<8} {}  {}",
            c.status.name().to_uppercase(),
            c.name,
            c.detail
        );
    }
    lock(state).checks.extend(checks);
}

fn set_phase(state: &Mutex<Report>, phase: &str) {
    println!("== {phase}");
    lock(state).phase = phase.to_string();
}

/// Everything after argument parsing; returns the exit code.
fn run(state: &Mutex<Report>, args: &Args) -> i32 {
    set_phase(state, "open");
    let config = RuntimeConfig {
        ordinal: args.ordinal,
        budget_bytes: 2 << 30,
        sync_timeout: args.cap.min(Duration::from_secs(60)),
        ..RuntimeConfig::default()
    };
    let opened = catch_unwind(AssertUnwindSafe(|| CudaRuntime::open(config)));
    let rt = match opened {
        Ok(Ok(rt)) => rt,
        Ok(Err(e)) => {
            let refused = matches!(e, CudaError::LibraryMissing { .. } | CudaError::Arch { .. });
            eprintln!("rung0: {e}");
            let mut s = lock(state);
            s.refusal = Some(
                JsonObj::new()
                    .with("kind", e.kind())
                    .with("detail", e.to_string())
                    .into(),
            );
            s.checks.push(Check::from_error("runtime.open", &e));
            return if refused { EXIT_REFUSED } else { EXIT_FAIL };
        }
        Err(payload) => {
            let text = panic_text(payload.as_ref());
            eprintln!("rung0: CudaRuntime::open panicked: {text}");
            lock(state)
                .checks
                .push(Check::panicked("runtime.open", text));
            return EXIT_FAIL;
        }
    };
    {
        let info = rt.info();
        println!(
            "device {:?} cc {}.{} sms {} driver {} nvrtc {}.{} cublas {}.{}.{}",
            info.name,
            info.compute_capability.0,
            info.compute_capability.1,
            info.sm_count,
            info.driver_version,
            info.nvrtc_version.0,
            info.nvrtc_version.1,
            info.cublas_version.0,
            info.cublas_version.1,
            info.cublas_version.2
        );
        let mut s = lock(state);
        s.device = Some(info.to_json());
        s.checks.push(
            Check::pass(
                "runtime.open",
                "libraries probed, sm_90 device opened, cuBLAS handle ready",
            )
            .with("cublas_math_mode_is_default", info.cublas_math_mode == 0)
            .with("cublas_atomics_not_allowed", info.cublas_atomics_mode == 0),
        );
        if info.cublas_math_mode != 0 || info.cublas_atomics_mode != 0 {
            s.checks.push(Check::fail(
                "runtime.cublas_modes",
                format!(
                    "math mode {} (want 0, CUBLAS_DEFAULT_MATH), atomics mode {} (want 0, NOT_ALLOWED)",
                    info.cublas_math_mode, info.cublas_atomics_mode
                ),
            ));
        }
    }

    set_phase(state, "nvrtc");
    record(state, smoke::compile_checks(&rt));

    set_phase(state, "k0");
    record(state, smoke::k0_checks(&rt));

    set_phase(state, "cublas_probe");
    record(state, smoke::cublas_probe(&rt));

    set_phase(state, "k1");
    match smoke::gemm_cases() {
        Ok(cases) => {
            for case in &cases {
                record(state, smoke::gemm_case_checks(&rt, case));
            }
        }
        Err(e) => record(state, vec![Check::from_error("k1.cases", &e)]),
    }
    for layout in GemmLayout::ALL {
        record(
            state,
            smoke::bf16_engines_agree(&rt, layout, (130, 70, 260)),
        );
    }

    set_phase(state, "k1_determinism");
    match GemmShape::new(130, 70, 260) {
        Ok(shape) => {
            for (operands, engine) in [
                (Operands::ExactF32, Bf16Engine::Ffma),
                (Operands::Bf16, Bf16Engine::Ffma),
                (Operands::Bf16, Bf16Engine::Cublas),
            ] {
                let case = GemmCase {
                    spec: GemmSpec {
                        operands,
                        layout: GemmLayout::Nn,
                        shape,
                        acc: Accumulate::Add,
                    },
                    engine,
                };
                record(
                    state,
                    smoke::gemm_determinism(&rt, &case, RUNG0_DETERMINISM_REPS),
                );
            }
        }
        Err(e) => record(state, vec![Check::from_error("k1.determinism", &e)]),
    }

    set_phase(state, "libraries");
    let loaded = match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => Json::Arr(
            libprobe::REQUIRED
                .iter()
                .flat_map(|spec| libprobe::mapped_paths(&maps, spec.mapped_prefixes))
                .map(Json::from)
                .collect(),
        ),
        Err(e) => Json::from(format!("/proc/self/maps unreadable: {e}")),
    };
    let stats = rt.cache_stats();
    {
        let mut s = lock(state);
        s.extra.push("loaded_libraries", loaded);
        s.extra.push(
            "nvrtc_cache",
            JsonObj::new()
                .with("hits", stats.hits)
                .with("misses", stats.misses)
                .with("evictions", stats.evictions),
        );
        s.extra.push(
            "alloc_budget",
            JsonObj::new()
                .with("cap_bytes", rt.budget().cap())
                .with("reserved_at_end_bytes", rt.budget().used()),
        );
    }
    let status = overall(&lock(state).checks);
    if status == Status::Pass {
        EXIT_PASS
    } else {
        EXIT_FAIL
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    let args = match parse(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("rung0: {e}\n{USAGE}");
            std::process::exit(EXIT_REFUSED);
        }
    };
    let path = match prepare_out(&args.out) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("rung0: {e}");
            std::process::exit(EXIT_REFUSED);
        }
    };
    let started = Instant::now();
    let state = Arc::new(Mutex::new(Report {
        header: header(&args),
        device: None,
        refusal: None,
        checks: Vec::new(),
        extra: JsonObj::new(),
        phase: "start".to_string(),
        written: false,
    }));
    spawn_watchdog(Arc::clone(&state), path.clone(), args.cap, started);

    let code = run(&state, &args);
    let status = match code {
        EXIT_PASS => "pass",
        EXIT_REFUSED => "refused",
        _ => match overall(&lock(&state).checks) {
            Status::Panicked => "panicked",
            _ => "fail",
        },
    };
    let mut s = lock(&state);
    s.phase = "done".to_string();
    match write_once(&mut s, &path, status, code, started.elapsed()) {
        Written::Now => {}
        // The watchdog wrote first: the cap expired, whatever main concluded.
        Written::Already => {
            drop(s);
            std::process::exit(EXIT_TIMEOUT);
        }
        // No report means no evidence: never exit 0.
        Written::Failed => {
            drop(s);
            std::process::exit(if code == EXIT_PASS { EXIT_FAIL } else { code });
        }
    }
    let pass = s.checks.iter().filter(|c| c.status == Status::Pass).count();
    println!(
        "rung0: {status} ({pass}/{} checks pass) in {:.1} s; report {}",
        s.checks.len(),
        started.elapsed().as_secs_f64(),
        path.display()
    );
    drop(s);
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Mutex<Report> {
        Mutex::new(Report {
            header: JsonObj::new().with("kind", "test"),
            device: None,
            refusal: None,
            checks: vec![Check::pass("k0.zero.vs_host", "ok")],
            extra: JsonObj::new(),
            phase: "k1".to_string(),
            written: false,
        })
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("qd-rung0-{name}-{}", std::process::id()));
        prepare_out(&dir).unwrap()
    }

    fn cleanup(path: &Path) {
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn the_cap_writes_a_timeout_report_naming_the_phase_and_exits_3() {
        let state = report();
        let path = scratch("cap");
        let code = fire(
            &state,
            &path,
            Duration::from_secs(280),
            Instant::now(),
            Duration::from_millis(50),
        );
        assert_eq!(code, Some(EXIT_TIMEOUT));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"status\":\"timeout\""), "{text}");
        assert!(text.contains("\"exit_code\":3"), "{text}");
        assert!(text.contains("the 280 s cap expired during k1"), "{text}");
        // The finished check stays in the report beside the cap.
        assert!(text.contains("k0.zero.vs_host"), "{text}");
        // Main arriving after the watchdog finds the report written.
        let mut s = lock(&state);
        assert_eq!(
            write_once(&mut s, &path, "pass", EXIT_PASS, Duration::ZERO),
            Written::Already
        );
        drop(s);
        cleanup(&path);
    }

    #[test]
    fn the_watchdog_stands_down_when_main_already_wrote() {
        let state = report();
        let path = scratch("done");
        {
            let mut s = lock(&state);
            assert_eq!(
                write_once(&mut s, &path, "pass", EXIT_PASS, Duration::ZERO),
                Written::Now
            );
        }
        assert_eq!(
            fire(
                &state,
                &path,
                Duration::from_secs(1),
                Instant::now(),
                Duration::from_millis(50)
            ),
            None
        );
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"status\":\"pass\""), "{text}");
        cleanup(&path);
    }

    #[test]
    fn a_held_report_lock_still_ends_the_run_at_the_cap() {
        let state = report();
        let path = scratch("held");
        let held = lock(&state);
        let code = fire(
            &state,
            &path,
            Duration::from_secs(1),
            Instant::now(),
            Duration::from_millis(30),
        );
        drop(held);
        assert_eq!(code, Some(EXIT_TIMEOUT));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("report lock unavailable at the cap"),
            "{text}"
        );
        cleanup(&path);
    }

    #[test]
    fn a_failed_report_write_still_ends_the_run_at_the_cap() {
        let state = report();
        // A directory where the report file should be: the rename fails.
        let dir = std::env::temp_dir().join(format!("qd-rung0-fail-{}", std::process::id()));
        let path = dir.join("rung0-report.json");
        std::fs::create_dir_all(&path).unwrap();
        let code = fire(
            &state,
            &path,
            Duration::from_secs(1),
            Instant::now(),
            Duration::from_millis(30),
        );
        assert_eq!(code, Some(EXIT_TIMEOUT));
        assert!(!lock(&state).written);
        std::fs::remove_dir(&path).unwrap();
        // The failed rename took its temporary file with it; this fails if not.
        std::fs::remove_dir(&dir).unwrap();
    }
}
