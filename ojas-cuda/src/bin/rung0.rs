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
//! Steps 2-5 are [`smoke::m0_phases`], the sequence `runga` also runs. The
//! report, the watchdog and the end of `main` are
//! [`ojas_cuda::report_cli`]'s, shared with `runga`.
//!
//! A watchdog enforces the wall-clock cap (default 280 s, at most 300):
//! at the cap it writes the report with what finished and exits 3. Exit 0
//! means every check ran and passed; 1 a check failed or panicked; 2 a
//! refusal before any check; 3 the cap.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ojas_cuda::check::{overall, Status};
use ojas_cuda::report_cli::{finish, header, lock, record, set_phase, spawn_watchdog, Report};
use ojas_cuda::rung0_cli::{parse, prepare_out, Args, EXIT_FAIL, EXIT_PASS, EXIT_REFUSED, USAGE};
use ojas_cuda::runtime::RuntimeConfig;
use ojas_cuda::smoke;

const PROG: &str = "rung0";
/// Repetitions of the rung-0 determinism runs (the device tests run 25).
const RUNG0_DETERMINISM_REPS: usize = 5;

/// Everything after argument parsing; returns the exit code.
fn run(state: &Mutex<Report>, args: &Args) -> i32 {
    set_phase(state, "open");
    let config = RuntimeConfig {
        ordinal: args.ordinal,
        budget_bytes: 2 << 30,
        sync_timeout: args.cap.min(Duration::from_secs(60)),
        ..RuntimeConfig::default()
    };
    let rt = match smoke::open_recorded(state, PROG, config) {
        Ok(rt) => rt,
        Err(code) => return code,
    };
    smoke::record_device(state, &rt);

    smoke::m0_phases(
        &rt,
        RUNG0_DETERMINISM_REPS,
        &mut |p| set_phase(state, p),
        &mut |c| record(state, c),
    );

    set_phase(state, "libraries");
    let (loaded, cache) = smoke::loaded_libraries(&rt);
    {
        let mut s = lock(state);
        s.extra.push("loaded_libraries", loaded);
        s.extra.push("nvrtc_cache", cache);
        s.extra.push("alloc_budget", smoke::alloc_budget(&rt));
    }
    if overall(&lock(state).checks) == Status::Pass {
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
            eprintln!("{PROG}: {e}\n{USAGE}");
            std::process::exit(EXIT_REFUSED);
        }
    };
    let path = match prepare_out(&args.out) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{PROG}: {e}");
            std::process::exit(EXIT_REFUSED);
        }
    };
    let started = Instant::now();
    let head = header(
        PROG,
        "rung-0 toolchain smoke: one process, small shapes, no training step",
        &args,
        &raw,
    );
    let state = Arc::new(Mutex::new(Report::new(PROG, head)));
    spawn_watchdog(Arc::clone(&state), PROG, path.clone(), args.cap, started);

    let code = run(&state, &args);
    finish(&state, &path, code, started);
}
