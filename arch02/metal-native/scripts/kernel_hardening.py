#!/usr/bin/env python3
"""Run bounded, serial kernel checks; preserve failures, skips, and source drift.

Use --suite to select crates. This checks local test coverage, not publication
readiness or every device/shape. No packages are installed and no Git files are
restored. Gemma's existing tests write bench/results; use a disposable checkout
when those reports must be preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
CRATES = {
    "runtime": ROOT / "crates/metal-runtime",
    "native": ROOT / "arch_02_value_resid/metal-native",
    "gemma": ROOT / "gemma-metal",
}
SUMMARY = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;")
SKIP = re.compile(r"\bskip(?:ping|ped)?(?::|\s)", re.IGNORECASE)


def snapshot():
    files = set()
    for root in CRATES.values():
        for directory, extension in [("src", "*.rs"), ("kernels", "*.metal")]:
            files.update((root / directory).rglob(extension))
        files.update(root / name for name in ["Cargo.toml", "Cargo.lock", "build.rs"])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=CRATES, action="append")
    parser.add_argument("--repeats", type=int, default=5, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per invocation (30..1800)")
    parser.add_argument("--shader-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if not 30 <= args.timeout <= 1800:
        parser.error("--timeout must be between 30 and 1800 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    suites = args.suite or list(CRATES)
    env = os.environ.copy()
    if args.shader_validation:
        env.update(MTL_SHADER_VALIDATION="1", MTL_SHADER_VALIDATION_REPORT_TO_STDERR="1",
                   MTL_SHADER_VALIDATION_ABORT_ON_FAULT="1")
    report = {"source_before": snapshot(), "runs": [], "scope": "bounded local crate tests",
              "shader_validation_requested": args.shader_validation}
    for repeat in range(1, args.repeats + 1):
        for suite in suites:
            log = args.output / f"{repeat:02d}-{suite}.log"
            command = ["cargo", "test", "--release", "--manifest-path",
                       str(CRATES[suite] / "Cargo.toml"), "--lib", "--",
                       "--test-threads=1", "--nocapture"]
            before = snapshot()
            started = time.monotonic()
            timed_out = False
            with log.open("wb") as output:
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                try:
                    code = child.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
            content = log.read_text(errors="replace")
            summaries = SUMMARY.findall(content)
            skipped = [line for line in content.splitlines() if SKIP.search(line)]
            after = snapshot()
            changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
            validation = "Metal GPU Validation Enabled" in content
            successful = (code == 0 and not timed_out and bool(summaries)
                          and all(status == "ok" and int(failed) == 0 and int(passed) > 0
                                  for status, passed, failed, _ in summaries))
            status = "failed"
            if successful:
                status = "passed"
                if skipped or changed or (args.shader_validation and not validation):
                    status = "incomplete"
            run = {"suite": suite, "repeat": repeat, "command": command, "log": log.name,
                   "status": status, "exit_code": code, "timed_out": timed_out,
                   "seconds": round(time.monotonic() - started, 3),
                   "summaries": [{"passed": int(p), "failed": int(f), "ignored": int(i)}
                                 for _, p, f, i in summaries],
                   "skip_messages": skipped, "changed_sources": changed,
                   "shader_validation_observed": validation}
            report["runs"].append(run)
            report["source_after"] = after
            report["status"] = ("passed" if all(r["status"] == "passed" for r in report["runs"])
                                else "incomplete_or_failed")
            (args.output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{repeat}/{args.repeats} {suite}: {status} ({run['seconds']}s)", flush=True)
            if not successful:
                return 1
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
