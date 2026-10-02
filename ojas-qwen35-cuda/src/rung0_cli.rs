//! The rung-0 binary's arguments and report file, host-side so they are
//! tested on the Mac.
//!
//! - `--out <dir>` (required): the report is `<dir>/rung0-report.json`. An
//!   existing report is refused, never overwritten, and the report is written
//!   to a temporary name and renamed, so a reader never sees half a file.
//! - `--cap-secs <n>`: wall-clock cap, 1..=300, default 280. The binary
//!   enforces it itself with a watchdog thread; the default leaves 20 s under
//!   the box command's outer `timeout 300` for the report to land.
//! - `--ordinal <n>`: CUDA device ordinal, default 0.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// The report's file name inside `--out`.
pub const REPORT_FILE: &str = "rung0-report.json";
/// The default wall-clock cap.
pub const DEFAULT_CAP_SECS: u64 = 280;
/// The hard ceiling on the cap.
pub const MAX_CAP_SECS: u64 = 300;

/// Exit status: every check ran and passed.
pub const EXIT_PASS: i32 = 0;
/// Exit status: a check failed or panicked.
pub const EXIT_FAIL: i32 = 1;
/// Exit status: refused before any check (bad arguments, a missing library,
/// the wrong architecture, an existing report).
pub const EXIT_REFUSED: i32 = 2;
/// Exit status: the wall-clock cap expired.
pub const EXIT_TIMEOUT: i32 = 3;

/// Usage text.
pub const USAGE: &str =
    "usage: rung0 --out <dir> [--cap-secs <1..=300, default 280>] [--ordinal <n>]";

/// Parsed arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    /// Output directory.
    pub out: PathBuf,
    /// Wall-clock cap.
    pub cap: Duration,
    /// CUDA device ordinal.
    pub ordinal: usize,
}

/// Parse arguments (without the program name).
pub fn parse(args: &[String]) -> Result<Args, String> {
    let mut out = None;
    let mut cap_secs = DEFAULT_CAP_SECS;
    let mut ordinal = 0usize;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--cap-secs" => {
                let v = value("--cap-secs")?;
                cap_secs = v
                    .parse()
                    .map_err(|_| format!("--cap-secs {v:?} is not a whole number of seconds"))?;
                if cap_secs == 0 || cap_secs > MAX_CAP_SECS {
                    return Err(format!(
                        "--cap-secs {cap_secs} is outside 1..={MAX_CAP_SECS}"
                    ));
                }
            }
            "--ordinal" => {
                let v = value("--ordinal")?;
                ordinal = v
                    .parse()
                    .map_err(|_| format!("--ordinal {v:?} is not a device index"))?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let out = out.ok_or_else(|| "--out <dir> is required".to_string())?;
    Ok(Args {
        out,
        cap: Duration::from_secs(cap_secs),
        ordinal,
    })
}

/// Create `dir` if needed and return the report path; refuse if a report is
/// already there. [`crate::report_cli::prepare_report`] for rung 0's file.
pub fn prepare_out(dir: &Path) -> Result<PathBuf, String> {
    crate::report_cli::prepare_report(dir, REPORT_FILE)
}

/// Write `text` to `path` through a temporary sibling and a rename. A failed
/// rename removes the temporary file, and says so if that fails too.
pub fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let tmp = path.with_extension(format!("json.tmp-{}", std::process::id()));
    std::fs::write(&tmp, text).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let cleanup = match std::fs::remove_file(&tmp) {
            Ok(()) => String::new(),
            Err(re) => format!("; the temporary {} is left: {re}", tmp.display()),
        };
        format!(
            "rename {} -> {}: {e}{cleanup}",
            tmp.display(),
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn defaults_and_explicit_values_parse() {
        let a = parse(&s(&["--out", "/tmp/x"])).unwrap();
        assert_eq!(a.cap, Duration::from_secs(280));
        assert_eq!(a.ordinal, 0);
        let b = parse(&s(&["--cap-secs", "300", "--ordinal", "1", "--out", "d"])).unwrap();
        assert_eq!((b.cap.as_secs(), b.ordinal), (300, 1));
    }

    #[test]
    fn the_cap_cannot_exceed_300_seconds_or_be_zero() {
        assert!(parse(&s(&["--out", "d", "--cap-secs", "301"])).is_err());
        assert!(parse(&s(&["--out", "d", "--cap-secs", "0"])).is_err());
        assert!(parse(&s(&["--out", "d", "--cap-secs", "x"])).is_err());
    }

    #[test]
    fn missing_or_unknown_arguments_are_refused() {
        assert!(parse(&s(&[])).is_err());
        assert!(parse(&s(&["--out"])).is_err());
        assert!(parse(&s(&["--out", "d", "--fast"])).is_err());
    }

    #[test]
    fn an_existing_report_is_refused_and_writes_are_atomic() {
        let dir = std::env::temp_dir().join(format!("qd-rung0-cli-{}", std::process::id()));
        let path = prepare_out(&dir).unwrap();
        write_atomic(&path, "{}\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{}\n");
        let err = prepare_out(&dir).unwrap_err();
        assert!(err.contains("refusing to overwrite"), "{err}");
        let leftovers: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
        assert_eq!(leftovers.len(), 1, "a temporary file was left behind");
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
}
