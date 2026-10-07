//! The rung binaries' report and watchdog end to end, on the host, in child
//! processes (this test binary re-run filtered to one test, with [`CHILD_ENV`]
//! naming the report directory):
//! - a run whose phase blocks past the wall-clock cap is ended by the
//!   watchdog thread with exit 3 and a timeout report naming that phase,
//!   while main is still blocked (`spawn_watchdog`, `fire`, `process::exit`);
//! - a run that finishes before the cap ends through `report_cli::finish`
//!   with its own exit code and a report it wrote itself.
//!
//! `report_cli`'s unit tests call `fire` directly; neither path can run
//! in-process, because both end the process.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ojas_cuda::check::Check;
use ojas_cuda::json::JsonObj;
use ojas_cuda::report_cli::{finish, prepare_report, record, set_phase, spawn_watchdog, Report};
use ojas_cuda::rung0_cli::{EXIT_PASS, EXIT_TIMEOUT};

/// Set in the child: the directory its report goes to.
const CHILD_ENV: &str = "OJAS_CUDA_WATCHDOG_CHILD_DIR";
const REPORT: &str = "watchdog-report.json";
const CAP: Duration = Duration::from_secs(1);
/// How long the child's blocked phase would last without the watchdog.
const BLOCKED_FOR: Duration = Duration::from_secs(60);
/// The parent's own bound on the child.
const PARENT_BOUND: Duration = Duration::from_secs(30);

/// A child's report and watchdog: one finished check, then `phase`.
fn child_state(dir: &Path, cap: Duration) -> (Arc<Mutex<Report>>, PathBuf, Instant) {
    let path = prepare_report(dir, REPORT).expect("child report path");
    let started = Instant::now();
    let state = Arc::new(Mutex::new(Report::new(
        "wdtest",
        JsonObj::new().with("kind", "ojas-cuda.wdtest"),
    )));
    spawn_watchdog(Arc::clone(&state), "wdtest", path.clone(), cap, started);
    set_phase(&state, "warmup");
    record(
        &state,
        vec![Check::pass("warmup.done", "finished before the cap")],
    );
    (state, path, started)
}

/// Run this binary's test `name` as a child with [`CHILD_ENV`] = a fresh
/// directory, bounded by [`PARENT_BOUND`]. Returns the status, the elapsed
/// time and the child's report text.
fn run_child(name: &str) -> (ExitStatus, Duration, String, PathBuf) {
    let dir =
        std::env::temp_dir().join(format!("ojas-cuda-watchdog-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let started = Instant::now();
    let mut proc = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, &dir)
        .spawn()
        .expect("spawn the child");
    let status = loop {
        if let Some(status) = proc.try_wait().expect("wait on the child") {
            break status;
        }
        if started.elapsed() > PARENT_BOUND {
            let _ = proc.kill();
            let _ = proc.wait();
            panic!("the child outlived {PARENT_BOUND:?}");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let path = dir.join(REPORT);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("no report at {}: {e}", path.display()));
    (status, started.elapsed(), text, dir)
}

#[test]
fn a_phase_blocked_past_the_cap_ends_the_process_with_exit_3_and_a_report_naming_it() {
    if let Some(dir) = std::env::var_os(CHILD_ENV) {
        let (state, _, _) = child_state(Path::new(&dir), CAP);
        set_phase(&state, "blocked_phase");
        std::thread::sleep(BLOCKED_FOR);
        // Reaching here means the watchdog did not end the process.
        std::process::exit(0);
    }
    let (status, elapsed, text, dir) = run_child(
        "a_phase_blocked_past_the_cap_ends_the_process_with_exit_3_and_a_report_naming_it",
    );
    assert_eq!(status.code(), Some(EXIT_TIMEOUT), "child status {status}");
    assert!(
        elapsed >= CAP && elapsed < BLOCKED_FOR,
        "the child ended after {elapsed:?}, not at the {CAP:?} cap"
    );
    assert!(text.contains("\"status\":\"timeout\""), "{text}");
    assert!(text.contains("\"exit_code\":3"), "{text}");
    assert!(
        text.contains("\"phase_at_end\":\"blocked_phase\""),
        "{text}"
    );
    assert!(
        text.contains("the 1 s cap expired during blocked_phase"),
        "{text}"
    );
    assert!(text.contains("wdtest.wall_clock_cap"), "{text}");
    assert!(text.contains("warmup.done"), "{text}");
    std::fs::remove_dir_all(&dir).expect("remove the child's report dir");
}

#[test]
fn a_run_that_finishes_before_the_cap_exits_with_its_own_code_through_finish() {
    if let Some(dir) = std::env::var_os(CHILD_ENV) {
        let (state, path, started) = child_state(Path::new(&dir), BLOCKED_FOR);
        finish(&state, &path, EXIT_PASS, started);
    }
    let (status, elapsed, text, dir) =
        run_child("a_run_that_finishes_before_the_cap_exits_with_its_own_code_through_finish");
    assert_eq!(status.code(), Some(EXIT_PASS), "child status {status}");
    assert!(elapsed < BLOCKED_FOR, "{elapsed:?}");
    assert!(text.contains("\"status\":\"pass\""), "{text}");
    assert!(text.contains("\"exit_code\":0"), "{text}");
    assert!(text.contains("\"phase_at_end\":\"done\""), "{text}");
    assert!(!text.contains("wall_clock_cap"), "{text}");
    std::fs::remove_dir_all(&dir).expect("remove the child's report dir");
}
