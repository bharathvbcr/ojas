#!/usr/bin/env python3
"""Shared scorer for same-shape 128M CUDA logs + Metal metrics.jsonl.

Parses:
  - CUDA tee logs from cuda_ref_128m.sh (step:* val_bpb / step_avg)
  - Metal-native metrics.jsonl (final_ema_sliding_bpb, step_ms)

Does NOT invent a Mac-vs-CUDA speed claim. Prints side-by-side facts only when
both sides are present, and always surfaces GRAD_ACCUM_STEPS.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

STEP_RE = re.compile(
    r"step:(\d+)/(\d+)\s+"
    r"(?:train_loss:([0-9.eE+-]+)\s+)?"
    r"(?:val_loss:([0-9.eE+-]+)\s+val_bpb:([0-9.eE+-]+)\s+)?"
    r"(?:train_time:([0-9.]+)\s*ms\s+)?"
    r"(?:step_avg:([0-9.]+)\s*ms)?"
)
ACCUM_RE = re.compile(r"GRAD_ACCUM_STEPS[=:\s]+(\d+)|accum[=:\s]+(\d+)|DISCLOSE:\s*GRAD_ACCUM_STEPS=(\d+)", re.I)


def parse_cuda_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    steps = []
    vals = []
    accum = None
    for m in ACCUM_RE.finditer(text):
        accum = int(next(g for g in m.groups() if g is not None))
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        step_avg = float(m.group(7)) if m.group(7) else None
        val_bpb = float(m.group(5)) if m.group(5) else None
        if step_avg is not None and val_bpb is None:
            steps.append({"step": step, "step_avg_ms": step_avg})
        if val_bpb is not None:
            vals.append({"step": step, "val_bpb": val_bpb, "step_avg_ms": step_avg})
    steady = steps[-max(1, min(20, len(steps))):] if steps else []
    mean_ms = statistics.mean(s["step_avg_ms"] for s in steady) if steady else None
    last_val = vals[-1]["val_bpb"] if vals else None
    return {
        "source": str(path),
        "kind": "cuda_log",
        "grad_accum_steps": accum,
        "n_train_log_lines": len(steps),
        "steady_mean_step_avg_ms": mean_ms,
        "last_val_bpb": last_val,
        "val_points": vals[-5:],
    }


def parse_metal_metrics(path: Path) -> dict:
    steps, final = [], None
    ref = None
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "final_ema_sliding_bpb" in r:
                final = r["final_ema_sliding_bpb"]
                ref = r.get("reference", r.get("reference_3070ti_sota_ladder"))
            elif "step_ms" in r:
                steps.append(r)
    tail = steps[-8:] if steps else []
    mean_ms = statistics.mean(s["step_ms"] for s in tail) if tail else None
    return {
        "source": str(path),
        "kind": "metal_metrics",
        "n_steps": len(steps),
        "steady_mean_step_ms": mean_ms,
        "final_ema_sliding_bpb": final,
        "reference_field": ref,
    }


def collect(root: Path) -> dict:
    out = {
        "root": str(root),
        "cuda": [],
        "metal": [],
        "accum_file": None,
        "note": (
            "CUDA val_bpb and Metal final_ema_sliding_bpb use different eval "
            "paths; do not treat a raw delta as bit-identical parity. "
            "No Mac-vs-CUDA speed claim without same-shape CUDA numbers here."
        ),
    }
    accum_path = root / ".accum"
    if accum_path.is_file():
        out["accum_file"] = accum_path.read_text().strip()
    for p in sorted(root.rglob("*.log")):
        out["cuda"].append(parse_cuda_log(p))
    for p in sorted(root.rglob("metrics.jsonl")):
        out["metal"].append(parse_metal_metrics(p))
    # also accept flat summary.json dropped by ingest
    summary = root / "summary.json"
    if summary.is_file():
        try:
            out["existing_summary"] = json.loads(summary.read_text())
        except json.JSONDecodeError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "root",
        nargs="?",
        default="logs/cuda128m",
        help="directory with CUDA logs and/or Metal metrics.jsonl",
    )
    ap.add_argument("--write-summary", action="store_true", help="write summary.json under root")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"missing directory: {root}")
    report = collect(root)
    print(json.dumps(report, indent=2))
    if args.write_summary:
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
