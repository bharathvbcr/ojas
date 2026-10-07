//! What a rung binary shares: the one-JSON-object report, written once by
//! whoever gets there first (main or the wall-clock watchdog), the watchdog
//! itself, and `runga`'s arguments. Host-side, so it is tested on the Mac.
//!
//! Both rung binaries (`rung0`, `runga`) run on this module: neither carries
//! a private copy of the report, the watchdog or the end of `main`
//! ([`finish`]). The watchdog's end-to-end path (a phase blocked past the cap
//! ends the process with exit 3) is `tests/watchdog_exit.rs`.
//!
//! Exit codes and the cap are [`crate::rung0_cli`]'s: 0 every check ran and
//! passed, 1 a check failed or panicked, 2 refused before any check, 3 the
//! wall-clock cap.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::check::{overall, Check, Status};
use crate::json::{Json, JsonObj};
use crate::rung0_cli::{
    self, write_atomic, Args, EXIT_FAIL, EXIT_PASS, EXIT_REFUSED, EXIT_TIMEOUT,
};

/// One run's report, filled in as checks finish.
pub struct Report {
    /// The binary's name (`runga`): the report kind is `ojas-cuda.<prog>`.
    pub prog: &'static str,
    pub header: JsonObj,
    pub device: Option<Json>,
    pub refusal: Option<Json>,
    pub checks: Vec<Check>,
    /// Report-only sections (timings, pins, library paths): never in `overall`.
    pub extra: JsonObj,
    pub phase: String,
    pub written: bool,
}

impl Report {
    pub fn new(prog: &'static str, header: JsonObj) -> Self {
        Report {
            prog,
            header,
            device: None,
            refusal: None,
            checks: Vec::new(),
            extra: JsonObj::new(),
            phase: "start".to_string(),
            written: false,
        }
    }

    pub fn render(&self, status: &str, exit_code: i32, elapsed: Duration) -> String {
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

/// The report, whatever a panicking holder left it as.
pub fn lock(state: &Mutex<Report>) -> MutexGuard<'_, Report> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The header every rung report starts with: kind, binary, arguments,
/// environment, and `quick: true` (rule 8: a smoke run promotes nothing).
pub fn header(prog: &str, quick_reason: &str, args: &Args, argv: &[String]) -> JsonObj {
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
        .with("kind", format!("ojas-cuda.{prog}"))
        .with("schema", 1u32)
        .with("quick", true)
        .with("quick_reason", quick_reason)
        .with("crate_version", env!("CARGO_PKG_VERSION"))
        .with(
            "binary",
            JsonObj::new()
                .with("path", exe.map(|p| p.display().to_string()))
                .with("bytes", exe_bytes),
        )
        .with("args", argv.to_vec())
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
pub enum Written {
    /// This call wrote it.
    Now,
    /// It was already written (by main or the watchdog).
    Already,
    /// The write failed; the reason is on stderr.
    Failed,
}

/// Write the report once; whoever gets here first (main or watchdog) writes.
pub fn write_once(
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
            eprintln!("{}: could not write the report: {e}", state.prog);
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
pub fn fire(
    state: &Mutex<Report>,
    prog: &str,
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
                &format!("{prog}.wall_clock_cap"),
                format!("the {} s cap expired during {phase}", cap.as_secs()),
            ));
            eprintln!(
                "{prog}: TIMEOUT after {} s during {phase}; report {}",
                cap.as_secs(),
                path.display()
            );
            write_once(&mut s, path, "timeout", EXIT_TIMEOUT, started.elapsed());
            return Some(EXIT_TIMEOUT);
        }
        if Instant::now() >= give_up {
            let text = format!(
                "{{\"kind\":\"ojas-cuda.{prog}\",\"status\":\"timeout\",\"exit_code\":{EXIT_TIMEOUT},\"detail\":\"report lock unavailable at the cap\"}}\n"
            );
            if let Err(e) = write_atomic(path, &text) {
                eprintln!("{prog}: could not write the timeout report: {e}");
            }
            return Some(EXIT_TIMEOUT);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Start the watchdog: at the cap it fires and, if it wrote, exits.
pub fn spawn_watchdog(
    state: Arc<Mutex<Report>>,
    prog: &'static str,
    path: PathBuf,
    cap: Duration,
    started: Instant,
) {
    std::thread::spawn(move || {
        std::thread::sleep(cap.saturating_sub(started.elapsed()));
        if let Some(code) = fire(&state, prog, &path, cap, started, Duration::from_secs(2)) {
            std::process::exit(code);
        }
    });
}

/// How main ends a run that got past argument parsing: name the phase
/// `done`, write the report (unless the watchdog already did), print the
/// summary, and exit. A report the watchdog wrote first means the cap
/// expired, whatever `code` says (exit 3); a report that could not be written
/// is no evidence, so the run never exits 0.
pub fn finish(state: &Mutex<Report>, path: &Path, code: i32, started: Instant) -> ! {
    let status = match code {
        EXIT_PASS => "pass",
        EXIT_REFUSED => "refused",
        _ => match overall(&lock(state).checks) {
            Status::Panicked => "panicked",
            _ => "fail",
        },
    };
    let mut s = lock(state);
    s.phase = "done".to_string();
    let written = write_once(&mut s, path, status, code, started.elapsed());
    let exit = match written {
        Written::Now => code,
        Written::Already => EXIT_TIMEOUT,
        Written::Failed if code == EXIT_PASS => EXIT_FAIL,
        Written::Failed => code,
    };
    if written == Written::Now {
        let pass = s.checks.iter().filter(|c| c.status == Status::Pass).count();
        println!(
            "{}: {status} ({pass}/{} checks pass) in {:.1} s; report {}",
            s.prog,
            s.checks.len(),
            started.elapsed().as_secs_f64(),
            path.display()
        );
    }
    drop(s);
    std::process::exit(exit);
}

/// Print and append checks.
pub fn record(state: &Mutex<Report>, checks: Vec<Check>) {
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

/// Name the phase now running (the watchdog's report names it).
pub fn set_phase(state: &Mutex<Report>, phase: &str) {
    println!("== {phase}");
    lock(state).phase = phase.to_string();
}

/// Create `dir` if needed and return the report path `<dir>/<file>`; refuse
/// if a report is already there.
pub fn prepare_report(dir: &Path, file: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create --out {}: {e}", dir.display()))?;
    let report = dir.join(file);
    if report.exists() {
        return Err(format!(
            "{} already exists; refusing to overwrite a report",
            report.display()
        ));
    }
    Ok(report)
}

/// `runga`'s report file name inside `--out`.
pub const RUNGA_REPORT_FILE: &str = "runga-report.json";

/// The smallest budget the GDN timing section may run with: its 4 x 8192,
/// H = 16, Dv = 128 shape needs about 15 GiB (L-cuda-gdn's hook; the lead's
/// ruling: its own budget of at least 16 GiB).
pub const MIN_GDN_TIMING_BUDGET_GIB: u64 = 16;
/// The default: the minimum plus headroom over "about 15 GiB", which is
/// L-cuda-gdn's estimate, not a measurement, and the 32 MiB cuBLAS workspace
/// the budget also holds.
pub const DEFAULT_GDN_TIMING_BUDGET_GIB: u64 = 24;
/// The largest budget `--gdn-timing-budget-gib` accepts: the GH200's HBM.
pub const MAX_GDN_TIMING_BUDGET_GIB: u64 = 96;

/// `runga` usage text.
pub const RUNGA_USAGE: &str = "usage: runga --out <dir> [--cap-secs <1..=300, default 280>] [--ordinal <n>] [--no-gdn-timing] [--gdn-timing-budget-gib <16..=96, default 24>]";

/// `runga`'s arguments: rung 0's, plus the GDN timing section's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RungaArgs {
    pub base: Args,
    /// Run the report-only GDN timing section.
    pub gdn_timing: bool,
    /// Its allocation budget, in GiB.
    pub gdn_timing_budget_gib: u64,
}

/// Parse `runga`'s arguments (without the program name): its own two flags
/// are taken out, and everything else goes to [`rung0_cli::parse`], which
/// refuses what it does not know.
pub fn parse_runga(args: &[String]) -> Result<RungaArgs, String> {
    let mut rest = Vec::with_capacity(args.len());
    let mut gdn_timing = true;
    let mut budget = DEFAULT_GDN_TIMING_BUDGET_GIB;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--no-gdn-timing" => gdn_timing = false,
            "--gdn-timing-budget-gib" => {
                let v = it.next().ok_or("--gdn-timing-budget-gib needs a value")?;
                budget = v
                    .parse()
                    .map_err(|_| format!("--gdn-timing-budget-gib {v:?} is not a whole number"))?;
                if !(MIN_GDN_TIMING_BUDGET_GIB..=MAX_GDN_TIMING_BUDGET_GIB).contains(&budget) {
                    return Err(format!(
                        "--gdn-timing-budget-gib {budget} is outside {MIN_GDN_TIMING_BUDGET_GIB}..={MAX_GDN_TIMING_BUDGET_GIB}"
                    ));
                }
            }
            _ => rest.push(a.clone()),
        }
    }
    Ok(RungaArgs {
        base: rung0_cli::parse(&rest)?,
        gdn_timing,
        gdn_timing_budget_gib: budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> Mutex<Report> {
        let mut r = Report::new("runga", JsonObj::new().with("kind", "test"));
        r.checks.push(Check::pass("k0.zero.vs_host", "ok"));
        r.phase = "k1".to_string();
        Mutex::new(r)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("qd-report-cli-{name}-{}", std::process::id()));
        prepare_report(&dir, RUNGA_REPORT_FILE).unwrap()
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
            "runga",
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
        assert!(text.contains("runga.wall_clock_cap"), "{text}");
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
                "runga",
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
            "runga",
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
        assert!(text.contains("ojas-cuda.runga"), "{text}");
        cleanup(&path);
    }

    #[test]
    fn a_failed_report_write_still_ends_the_run_at_the_cap() {
        let state = report();
        // A directory where the report file should be: the rename fails.
        let dir = std::env::temp_dir().join(format!("qd-report-cli-fail-{}", std::process::id()));
        let path = dir.join(RUNGA_REPORT_FILE);
        std::fs::create_dir_all(&path).unwrap();
        let code = fire(
            &state,
            "runga",
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

    fn s(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn runga_takes_its_own_flags_and_hands_the_rest_to_rung0s_parser() {
        let a = parse_runga(&s(&["--out", "d"])).unwrap();
        assert!(a.gdn_timing);
        assert_eq!(a.gdn_timing_budget_gib, 24);
        let m = parse_runga(&s(&["--out", "d", "--gdn-timing-budget-gib", "16"])).unwrap();
        assert_eq!(
            m.gdn_timing_budget_gib, 16,
            "the ruling's minimum is accepted"
        );
        assert_eq!(a.base.cap, Duration::from_secs(280));
        let b = parse_runga(&s(&[
            "--no-gdn-timing",
            "--out",
            "d",
            "--gdn-timing-budget-gib",
            "40",
            "--cap-secs",
            "200",
        ]))
        .unwrap();
        assert!(!b.gdn_timing);
        assert_eq!((b.gdn_timing_budget_gib, b.base.cap.as_secs()), (40, 200));
    }

    #[test]
    fn runga_refuses_a_small_budget_a_missing_value_and_unknown_flags() {
        assert!(parse_runga(&s(&["--out", "d", "--gdn-timing-budget-gib", "15"])).is_err());
        assert!(parse_runga(&s(&["--out", "d", "--gdn-timing-budget-gib", "97"])).is_err());
        assert!(parse_runga(&s(&["--out", "d", "--gdn-timing-budget-gib"])).is_err());
        assert!(parse_runga(&s(&["--out", "d", "--gdn-timing-budget-gib", "x"])).is_err());
        // Unknown flags still reach rung 0's parser, which refuses them.
        let err = parse_runga(&s(&["--out", "d", "--fast"])).unwrap_err();
        assert!(err.contains("unknown argument"), "{err}");
        assert!(
            parse_runga(&s(&["--no-gdn-timing"])).is_err(),
            "--out is still required"
        );
        assert!(parse_runga(&s(&["--out", "d", "--cap-secs", "301"])).is_err());
    }

    #[test]
    fn an_existing_report_is_refused() {
        let dir = std::env::temp_dir().join(format!("qd-report-cli-exists-{}", std::process::id()));
        let path = prepare_report(&dir, RUNGA_REPORT_FILE).unwrap();
        write_atomic(&path, "{}\n").unwrap();
        let err = prepare_report(&dir, RUNGA_REPORT_FILE).unwrap_err();
        assert!(err.contains("refusing to overwrite"), "{err}");
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
