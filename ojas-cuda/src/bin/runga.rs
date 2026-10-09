//! Rung (a)'s device run on the GH200: every device check of milestones M0
//! and M1 in one process, one JSON report at `<out>/runga-report.json`
//! (`cuda-backend-scoping.md` §6.1, M1: K0, K1, K8, K11 and the tiny-fixture
//! loader; plus K2(i)'s published-GDN checks, L-cuda-gdn's hook).
//!
//! In order:
//! 1. open the device as rung 0 does (probe the libraries, refuse anything
//!    but sm_90) under a 4 GiB budget, and record it;
//! 2. M0: NVRTC, K0, the cuBLAS probe, K1, K1 determinism
//!    ([`smoke::m0_phases`]);
//! 3. K8: SwiGLU forward and backward, the residual add, the activation
//!    sweep, bit for bit against the host;
//! 4. K11: AdamW and the squared norm, bit for bit against the f32
//!    emulation, within L-oracle's 1e-6 of torch on the decay-sensitive
//!    golden, and within the sanity bounds of the adamw_f float64 golden;
//! 5. the tiny-fixture loader, uploaded and read back bit for bit;
//! 6. L-cuda-small's K3, K4, K6, K7, K9 and K10 (`small_smoke`'s hooks, one
//!    phase each, after its own NVRTC check);
//! 7. K2(i), GDN at the published rule (`gdn_smoke::gdn_published_checks`);
//! 8. **report-only**: the GDN timing L-cuda-gdn's hook provides, at 4 x 8192,
//!    H = 16, Dv = 128, on its own runtime with a budget of at least 16 GiB
//!    (the lead's ruling), after the first runtime is dropped. It is never a
//!    check. When it cannot run (`--no-gdn-timing`, too little of the cap
//!    left, too little device memory, an allocation refused) the section
//!    says `not_run` and why.
//!
//! The same watchdog and exit codes as rung 0: 0 every check ran and passed,
//! 1 a check failed or panicked, 2 refused before any check, 3 the cap. The
//! report header names the sha256 pins of the goldens compiled in.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ojas_cuda::check::{overall, panic_text, Check, Status};
use ojas_cuda::gdn_smoke::{gdn_published_checks, gdn_published_timing};
use ojas_cuda::json::{Json, JsonObj};
use ojas_cuda::k11_golden::{AdamwF, DecaySensitive, DECAY_MANIFEST_SHA256};
use ojas_cuda::k11_smoke::k11_checks;
use ojas_cuda::k8_smoke::k8_checks;
use ojas_cuda::report_cli::{
    finish, header, lock, parse_runga, prepare_report, record, set_phase, spawn_watchdog, Report,
    RungaArgs, RUNGA_REPORT_FILE, RUNGA_USAGE,
};
use ojas_cuda::rung0_cli::{EXIT_FAIL, EXIT_PASS, EXIT_REFUSED};
use ojas_cuda::runtime::{CudaRuntime, RuntimeConfig};
use ojas_cuda::tiny_fixture_published::{files as tiny_files, loader_checks};
use ojas_cuda::CudaError;
use ojas_cuda::{small_common_cuda, small_smoke, smoke};

const PROG: &str = "runga";
/// Repetitions of the K1 determinism runs (rung 0's).
const DETERMINISM_REPS: usize = 5;
/// The pass/fail runtime's allocation budget.
const CHECKS_BUDGET_BYTES: u64 = 4 << 30;
/// The GDN timing section runs only with at least this much of the cap left.
const GDN_TIMING_MIN_REMAINING_S: u64 = 60;
/// The timing shape (L-cuda-gdn's hook): four sequences of 8192 tokens,
/// 16 value heads of dimension 128, 3 timed repetitions.
const GDN_TIMING_LENS: [usize; 4] = [8192; 4];
const GDN_TIMING_HEADS: usize = 16;
const GDN_TIMING_V_DIM: usize = 128;
const GDN_TIMING_REPS: usize = 3;

/// The sha256 pins of the goldens compiled into this binary.
fn pins() -> Json {
    let pairs = |v: Vec<(String, String)>| -> Json {
        let mut o = JsonObj::new();
        for (f, sha) in v {
            o.push(&f, sha);
        }
        o.into()
    };
    let decay = match DecaySensitive::embedded() {
        Ok(d) => {
            let mut v = d.pins;
            v.push((
                "manifest.json".to_string(),
                DECAY_MANIFEST_SHA256.to_string(),
            ));
            pairs(v)
        }
        Err(e) => Json::from(format!("unreadable: {e}")),
    };
    let adamw_f = match AdamwF::embedded() {
        Ok(f) => pairs(f.pins),
        Err(e) => Json::from(format!("unreadable: {e}")),
    };
    let tiny = pairs(
        tiny_files::SHA256SUMS
            .lines()
            .filter_map(|l| l.split_once("  "))
            .map(|(sha, f)| (f.to_string(), sha.to_string()))
            .collect(),
    );
    JsonObj::new()
        .with("adamw_decay_sensitive", decay)
        .with("adamw_f", adamw_f)
        .with("qwen35_train_published", tiny)
        .into()
}

/// The report-only GDN timing section. Never a check.
fn gdn_timing(args: &RungaArgs, started: Instant) -> Json {
    let budget = args.gdn_timing_budget_gib << 30;
    let remaining = args.base.cap.saturating_sub(started.elapsed()).as_secs();
    let base = JsonObj::new()
        .with("report_only", true)
        .with("budget_bytes", budget)
        .with("min_remaining_s", GDN_TIMING_MIN_REMAINING_S)
        .with("remaining_s_at_start", remaining)
        .with(
            "shape",
            JsonObj::new()
                .with("lens", GDN_TIMING_LENS.to_vec())
                .with("heads", GDN_TIMING_HEADS)
                .with("v_dim", GDN_TIMING_V_DIM)
                .with("reps", GDN_TIMING_REPS),
        );
    fn not_run(base: JsonObj, reason: String) -> Json {
        println!("NOT_RUN  gdn_published_timing  {reason}");
        base.with("status", "not_run").with("reason", reason).into()
    }
    if !args.gdn_timing {
        return not_run(base, "--no-gdn-timing".to_string());
    }
    if remaining < GDN_TIMING_MIN_REMAINING_S {
        return not_run(
            base,
            format!(
                "{remaining} s of the {} s cap left; the section needs {GDN_TIMING_MIN_REMAINING_S} s",
                args.base.cap.as_secs()
            ),
        );
    }
    let config = RuntimeConfig {
        ordinal: args.base.ordinal,
        budget_bytes: budget,
        sync_timeout: Duration::from_secs(remaining.min(60)),
        ..RuntimeConfig::default()
    };
    let rt = match catch_unwind(AssertUnwindSafe(|| CudaRuntime::open(config))) {
        Ok(Ok(rt)) => rt,
        Ok(Err(e)) => return not_run(base, format!("its runtime did not open: {e}")),
        Err(p) => {
            return not_run(
                base,
                format!("its runtime open panicked: {}", panic_text(p.as_ref())),
            )
        }
    };
    let total_mem = rt.info().total_mem;
    let base = base.with("device_total_mem_bytes", total_mem);
    if (total_mem as u64) < budget {
        return not_run(
            base,
            format!("the device has {total_mem} bytes, under the {budget}-byte budget"),
        );
    }
    let used_before = match rt.budget().used() {
        Ok(used) => used,
        Err(e) => return not_run(base, format!("its budget could not be read: {e}")),
    };
    let base = base.with("budget_used_before_bytes", used_before);
    let ran = catch_unwind(AssertUnwindSafe(|| {
        gdn_published_timing(
            &rt,
            &GDN_TIMING_LENS,
            GDN_TIMING_HEADS,
            GDN_TIMING_V_DIM,
            GDN_TIMING_REPS,
        )
    }));
    let base = match rt.budget().used() {
        Ok(used_after) => base.with("budget_used_after_bytes", used_after),
        Err(e) => base.with("budget_used_after_error", e.to_string()),
    };
    match ran {
        Ok(Ok(t)) => {
            println!("REPORT   gdn_published_timing  {}", t.to_json().render());
            base.with("status", "ran")
                .with("timing", t.to_json())
                .into()
        }
        Ok(Err(e @ CudaError::Capacity { .. })) => base
            .with("status", "not_run")
            .with("reason", format!("an allocation was refused: {e}"))
            .into(),
        Ok(Err(e)) => base
            .with("status", "error")
            .with("reason", e.to_string())
            .into(),
        Err(p) => base
            .with("status", "panicked")
            .with("reason", panic_text(p.as_ref()))
            .into(),
    }
}

/// Everything after argument parsing; returns the exit code.
fn run(state: &Mutex<Report>, args: &RungaArgs, started: Instant) -> i32 {
    set_phase(state, "open");
    let config = RuntimeConfig {
        ordinal: args.base.ordinal,
        budget_bytes: CHECKS_BUDGET_BYTES,
        sync_timeout: args.base.cap.min(Duration::from_secs(60)),
        ..RuntimeConfig::default()
    };
    let rt = match smoke::open_recorded(state, PROG, config) {
        Ok(rt) => rt,
        Err(code) => return code,
    };
    smoke::record_device(state, &rt);

    smoke::m0_phases(
        &rt,
        DETERMINISM_REPS,
        &mut |p| set_phase(state, p),
        &mut |c| record(state, c),
    );

    set_phase(state, "k8");
    record(state, k8_checks(&rt));

    set_phase(state, "k11");
    record(state, k11_checks(&rt));

    set_phase(state, "loader");
    record(state, smoke::guarded("loader", || loader_checks(&rt)));

    // L-cuda-small's hooks (src/small_smoke.rs), one phase each so a cap
    // names the kernel that was running.
    type Hook = fn(&CudaRuntime) -> Vec<Check>;
    let small: [(&str, Hook); 7] = [
        ("small.nvrtc", small_common_cuda::compile_checks),
        ("k3", small_smoke::k3_checks),
        ("k4", small_smoke::k4_checks),
        ("k6", small_smoke::k6_checks),
        ("k7", small_smoke::k7_checks),
        ("k9", small_smoke::k9_checks),
        ("k10", small_smoke::k10_checks),
    ];
    for (phase, hook) in small {
        set_phase(state, phase);
        record(state, smoke::guarded(phase, || hook(&rt)));
    }

    set_phase(state, "gdn_published");
    record(state, gdn_published_checks(&rt));

    set_phase(state, "libraries");
    let (loaded, cache) = smoke::loaded_libraries(&rt);
    let workspace = rt.config().cublas_workspace_bytes as u64;
    {
        let mut s = lock(state);
        s.extra.push("loaded_libraries", loaded);
        s.extra.push("nvrtc_cache", cache);
        s.extra.push(
            "alloc_budget",
            smoke::alloc_budget(&rt).with("cublas_workspace_bytes", workspace),
        );
        // Every check frees its buffers: only the cuBLAS workspace may remain.
        const NO_LEAKS: &str = "runtime.no_leaked_buffers";
        s.checks.push(match rt.budget().used() {
            Ok(used) if used == workspace => Check::pass(
                NO_LEAKS,
                format!("{used} bytes reserved at the end: the cuBLAS workspace only"),
            ),
            Ok(used) => Check::fail(
                NO_LEAKS,
                format!("{used} bytes reserved at the end; the workspace is {workspace}"),
            ),
            Err(e) => Check::fail(NO_LEAKS, format!("the budget could not be read: {e}")),
        });
    }
    let status = overall(&lock(state).checks);
    drop(rt);

    set_phase(state, "gdn_published_timing (report-only)");
    let timing = gdn_timing(args, started);
    lock(state).extra.push("gdn_published_timing", timing);

    if status == Status::Pass {
        EXIT_PASS
    } else {
        EXIT_FAIL
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{RUNGA_USAGE}");
        return;
    }
    let args = match parse_runga(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{PROG}: {e}\n{RUNGA_USAGE}");
            std::process::exit(EXIT_REFUSED);
        }
    };
    let path = match prepare_report(&args.base.out, RUNGA_REPORT_FILE) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{PROG}: {e}");
            std::process::exit(EXIT_REFUSED);
        }
    };
    let started = Instant::now();
    let head = header(
        PROG,
        "rung (a) device checks: one process, small shapes, no training step",
        &args.base,
        &raw,
    )
    .with("gdn_timing", args.gdn_timing)
    .with("gdn_timing_budget_gib", args.gdn_timing_budget_gib)
    .with("golden_pins", pins());
    let state = Arc::new(Mutex::new(Report::new(PROG, head)));
    spawn_watchdog(
        Arc::clone(&state),
        PROG,
        path.clone(),
        args.base.cap,
        started,
    );

    let code = run(&state, &args, started);
    finish(&state, &path, code, started);
}
