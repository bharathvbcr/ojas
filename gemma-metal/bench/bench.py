#!/usr/bin/env python3
"""Phase 0 baseline ladder for Gemma 4 Metal inference gates.

Backends: ollama, mlx-lm, litert-lm (probe), basert (probe).
Pins KV/context and max generation tokens for comparable decode tok/s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_OLLAMA_URL.startswith("http"):
    DEFAULT_OLLAMA_URL = f"http://{DEFAULT_OLLAMA_URL}"

# Honest-lane pins (documented in gates.md)
DEFAULT_NUM_CTX = 4096
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMP = 0.0

# Historical floor from prior Benchmark runs on this host
HISTORICAL_OLLAMA_31B_TPS = 11.0


@dataclass
class RunMetrics:
    backend: str
    model: str
    prompt_id: str
    prompt: str
    lane: str = "honest"  # honest | mtp | frontier
    content: str = ""
    ttft_ms: float | None = None
    total_ms: float = 0.0
    load_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    decode_tok_s: float | None = None
    prefill_tok_s: float | None = None
    rss_mb: float | None = None
    num_ctx: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    mtp: bool = False
    notes: str = ""
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchReport:
    created_at: str
    host: dict[str, Any]
    pins: dict[str, Any]
    availability: dict[str, Any]
    runs: list[RunMetrics] = field(default_factory=list)


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": sys.platform}
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        mem = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        info["cpu"] = brand
        info["mem_gb"] = round(mem / (1024**3), 1)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        info["sysctl_error"] = str(exc)
    return info


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def probe_availability() -> dict[str, Any]:
    avail: dict[str, Any] = {
        "ollama": {"installed": False, "models": [], "mtp": "unknown"},
        "mlx_lm": {"installed": False, "cli": None, "mtp": "draft-model flag"},
        "litert_lm": {"installed": False, "notes": "not found on PATH"},
        "basert": {"installed": False, "notes": "not found on PATH / no libbaseRT"},
    }

    if which("ollama"):
        avail["ollama"]["installed"] = True
        avail["ollama"]["path"] = which("ollama")
        try:
            req = urllib.request.Request(f"{DEFAULT_OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            avail["ollama"]["models"] = [m["name"] for m in data.get("models", [])]
            avail["ollama"]["reachable"] = True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            avail["ollama"]["reachable"] = False
            avail["ollama"]["error"] = str(exc)
        # Ollama exposes no public MTP/speculative flag for gemma4:31b-mlx as of this probe
        avail["ollama"]["mtp"] = (
            "not exposed via API for gemma4:31b-mlx (no draft/speculative options)"
        )

    mlx_cli = which("mlx_lm") or which("mlx_lm.benchmark")
    if mlx_cli or which("mlx_lm.generate"):
        avail["mlx_lm"]["installed"] = True
        avail["mlx_lm"]["cli"] = which("mlx_lm") or which("mlx_lm.generate")
        avail["mlx_lm"]["benchmark"] = which("mlx_lm.benchmark")
        avail["mlx_lm"]["mtp"] = (
            "mlx_lm.generate --draft-model / --num-draft-tokens "
            "(stock mlx-lm; Gemma4 MTP stronger in mlx-vlm/Ollama when available)"
        )

    for name, keys in (
        ("litert_lm", ("litert-lm", "litert_lm", "litertlm")),
        ("basert", ("basert", "baseRT", "BaseRT")),
    ):
        for key in keys:
            path = which(key)
            if path:
                avail[name]["installed"] = True
                avail[name]["path"] = path
                break

    return avail


def process_rss_mb(pattern: str) -> float | None:
    """Best-effort RSS sum for processes matching pattern (macOS ps)."""
    try:
        out = subprocess.check_output(["ps", "-axo", "rss,comm"], text=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    total_kb = 0
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        rss_s, comm = parts
        if pattern.lower() in comm.lower():
            try:
                total_kb += int(rss_s)
            except ValueError:
                continue
    if total_kb <= 0:
        return None
    return round(total_kb / 1024.0, 1)


def load_prompts(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def ollama_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    prompt_id: str,
    num_ctx: int,
    max_tokens: int,
    temperature: float,
    think: bool = False,
) -> RunMetrics:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": think,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "seed": 0,
        },
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    metrics = RunMetrics(
        backend="ollama",
        model=model,
        prompt_id=prompt_id,
        prompt=prompt,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        temperature=temperature,
        mtp=False,
        notes="think=false; pinned num_ctx" if not think else "think=true",
    )
    start = time.perf_counter()
    ttft_recorded = False
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                content = msg.get("content") or ""
                thinking = msg.get("thinking") or ""
                if content:
                    metrics.content += content
                if not ttft_recorded and (content or thinking):
                    metrics.ttft_ms = (time.perf_counter() - start) * 1000
                    ttft_recorded = True
                if chunk.get("done"):
                    metrics.total_ms = (chunk.get("total_duration") or 0) / 1_000_000
                    metrics.load_ms = (chunk.get("load_duration") or 0) / 1_000_000
                    metrics.prompt_tokens = chunk.get("prompt_eval_count")
                    metrics.output_tokens = chunk.get("eval_count")
                    eval_ns = chunk.get("eval_duration") or 0
                    prompt_ns = chunk.get("prompt_eval_duration") or 0
                    if eval_ns > 0 and metrics.output_tokens:
                        metrics.decode_tok_s = metrics.output_tokens / (eval_ns / 1e9)
                    if prompt_ns > 0 and metrics.prompt_tokens:
                        metrics.prefill_tok_s = metrics.prompt_tokens / (
                            prompt_ns / 1e9
                        )
                    metrics.raw = {
                        k: chunk.get(k)
                        for k in (
                            "total_duration",
                            "load_duration",
                            "prompt_eval_count",
                            "prompt_eval_duration",
                            "eval_count",
                            "eval_duration",
                            "done_reason",
                        )
                    }
                    break
    except urllib.error.HTTPError as exc:
        metrics.error = exc.read().decode(errors="replace") or str(exc.reason)
        metrics.total_ms = (time.perf_counter() - start) * 1000
    except urllib.error.URLError as exc:
        metrics.error = str(getattr(exc, "reason", exc))
        metrics.total_ms = (time.perf_counter() - start) * 1000

    metrics.rss_mb = process_rss_mb("ollama")
    return metrics


def run_ollama(args: argparse.Namespace) -> BenchReport:
    avail = probe_availability()
    prompts = load_prompts(ROOT / "prompts.json")
    if args.quick:
        prompts = [p for p in prompts if p["id"] == "decode_pad"] or prompts[:1]

    report = BenchReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        host=host_info(),
        pins={
            "num_ctx": args.num_ctx,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "think": False,
            "lane": "honest",
        },
        availability=avail,
    )

    print(f"Ollama bench model={args.model} ctx={args.num_ctx} max={args.max_tokens}")
    # Warmup (discard)
    print("  warmup...", flush=True)
    _ = ollama_chat(
        base_url=args.ollama_url,
        model=args.model,
        prompt="Say hi in one word.",
        prompt_id="warmup",
        num_ctx=args.num_ctx,
        max_tokens=8,
        temperature=0.0,
        think=False,
    )

    for spec in prompts:
        print(f"  • {spec['id']}...", end=" ", flush=True)
        run = ollama_chat(
            base_url=args.ollama_url,
            model=args.model,
            prompt=spec["prompt"],
            prompt_id=spec["id"],
            num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            think=False,
        )
        report.runs.append(run)
        if run.error:
            print(f"ERROR {run.error[:120]}")
        else:
            print(
                f"decode={run.decode_tok_s and round(run.decode_tok_s, 2)} tok/s "
                f"ttft={run.ttft_ms and round(run.ttft_ms)} ms "
                f"out={run.output_tokens} rss={run.rss_mb} MB"
            )
    return report


def parse_mlx_benchmark_output(text: str) -> dict[str, Any]:
    """Parse mlx_lm.benchmark / generate stdout for tok/s style numbers."""
    out: dict[str, Any] = {"raw_tail": text[-2500:]}
    # mlx_lm.benchmark averages line:
    # Averages: prompt_tps=1966.868, generation_tps=75.718, peak_memory=4.458
    m = re.search(
        r"Averages:\s*prompt_tps=([0-9.]+),\s*generation_tps=([0-9.]+),"
        r"\s*peak_memory=([0-9.]+)",
        text,
    )
    if m:
        out["prompt_tps"] = float(m.group(1))
        out["generation_tps"] = float(m.group(2))
        out["peak_memory_gb"] = float(m.group(3))
    # Per-trial lines (take last if no averages)
    trials = list(
        re.finditer(
            r"Trial\s+\d+:\s*prompt_tps=([0-9.]+),\s*generation_tps=([0-9.]+),"
            r"\s*peak_memory=([0-9.]+)",
            text,
        )
    )
    if trials and "generation_tps" not in out:
        out["prompt_tps"] = float(trials[-1].group(1))
        out["generation_tps"] = float(trials[-1].group(2))
        out["peak_memory_gb"] = float(trials[-1].group(3))
    # mlx_lm.generate verbose:
    # Prompt: 32 tokens, 282.924 tokens-per-sec
    # Generation: 64 tokens, 76.085 tokens-per-sec
    # Peak memory: 4.316 GB
    for label, pattern in (
        ("prompt_tps", r"Prompt:\s*[0-9.]+\s*tokens?,\s*([0-9.]+)\s*tokens?-?per-?sec"),
        (
            "generation_tps",
            r"Generation:\s*[0-9.]+\s*tokens?,\s*([0-9.]+)\s*tokens?-?per-?sec",
        ),
        ("peak_memory_gb", r"Peak memory:\s*([0-9.]+)\s*GB"),
        ("ttft_ms", r"TTFT[:\s]+([0-9.]+)\s*ms"),
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m and label not in out:
            out[label] = float(m.group(1))
    return out


def run_mlx(args: argparse.Namespace) -> BenchReport:
    avail = probe_availability()
    bench_cli = which("mlx_lm.benchmark")
    if not bench_cli:
        # try python -m
        bench_cli = None

    report = BenchReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        host=host_info(),
        pins={
            "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "max_kv_size": args.num_ctx,
            "lane": "honest",
            "mtp": bool(args.draft_model),
        },
        availability=avail,
    )

    if not avail["mlx_lm"]["installed"] and not bench_cli:
        run = RunMetrics(
            backend="mlx_lm",
            model=args.model,
            prompt_id="mlx_benchmark",
            prompt="",
            error="mlx_lm not installed",
            notes="Install mlx-lm to run this backend",
        )
        report.runs.append(run)
        return report

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "benchmark",
        "--model",
        args.model,
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--generation-tokens",
        str(args.generation_tokens),
        "--num-trials",
        str(args.num_trials),
    ]
    print("Running:", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        run = RunMetrics(
            backend="mlx_lm",
            model=args.model,
            prompt_id="mlx_benchmark",
            prompt="",
            error=f"timeout after {args.timeout}s",
            total_ms=(time.perf_counter() - t0) * 1000,
        )
        report.runs.append(run)
        return report

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_mlx_benchmark_output(text)
    run = RunMetrics(
        backend="mlx_lm",
        model=args.model,
        prompt_id="mlx_benchmark",
        prompt=f"synthetic p={args.prompt_tokens} g={args.generation_tokens}",
        lane="mtp" if args.draft_model else "honest",
        mtp=bool(args.draft_model),
        total_ms=(time.perf_counter() - t0) * 1000,
        decode_tok_s=parsed.get("generation_tps"),
        prefill_tok_s=parsed.get("prompt_tps"),
        ttft_ms=parsed.get("ttft_ms"),
        output_tokens=args.generation_tokens,
        prompt_tokens=args.prompt_tokens,
        num_ctx=args.num_ctx,
        max_tokens=args.generation_tokens,
        rss_mb=process_rss_mb("python"),
        notes=f"mlx_lm.benchmark trials={args.num_trials}",
        raw={"parsed": parsed, "returncode": proc.returncode},
    )
    if proc.returncode != 0 and run.decode_tok_s is None:
        run.error = text[-1500:] or f"exit {proc.returncode}"
    report.runs.append(run)
    print(
        f"  decode={run.decode_tok_s} tok/s prefill={run.prefill_tok_s} "
        f"err={run.error and run.error[:80]}"
    )

    # Optional generate path for TTFT on a real prompt (post-load)
    if args.also_generate and not run.error:
        gen_cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            args.model,
            "--prompt",
            "What is 17 × 23? Respond with only the integer.",
            "--max-tokens",
            str(min(64, args.generation_tokens)),
            "--temp",
            "0",
            "--max-kv-size",
            str(args.num_ctx),
            "--verbose",
            "True",
        ]
        if args.draft_model:
            gen_cmd.extend(
                [
                    "--draft-model",
                    args.draft_model,
                    "--num-draft-tokens",
                    str(args.num_draft_tokens),
                ]
            )
        print("Running generate TTFT probe:", " ".join(gen_cmd[-8:]), flush=True)
        g0 = time.perf_counter()
        gproc = subprocess.run(
            gen_cmd, capture_output=True, text=True, timeout=args.timeout, check=False
        )
        gtext = (gproc.stdout or "") + "\n" + (gproc.stderr or "")
        # mlx_lm.generate verbose prints Prompt / Generation tokens-per-sec
        gparsed = parse_mlx_benchmark_output(gtext)
        # Wall TTFT approximation: time until first printable chunk is hard via
        # subprocess; use reported prompt time if present, else wall.
        gen_run = RunMetrics(
            backend="mlx_lm",
            model=args.model,
            prompt_id="math_easy_generate",
            prompt="What is 17 × 23? Respond with only the integer.",
            lane="mtp" if args.draft_model else "honest",
            mtp=bool(args.draft_model),
            content=gproc.stdout[-500:] if gproc.stdout else "",
            total_ms=(time.perf_counter() - g0) * 1000,
            decode_tok_s=gparsed.get("generation_tps"),
            prefill_tok_s=gparsed.get("prompt_tps"),
            num_ctx=args.num_ctx,
            max_tokens=min(64, args.generation_tokens),
            temperature=0.0,
            rss_mb=process_rss_mb("python"),
            notes="mlx_lm.generate verbose timing",
            raw={"parsed": gparsed, "returncode": gproc.returncode},
            error=None if gproc.returncode == 0 else gtext[-1200:],
        )
        report.runs.append(gen_run)
        print(
            f"  generate decode={gen_run.decode_tok_s} prefill={gen_run.prefill_tok_s}"
        )

    return report


def save_report(report: BenchReport, tag: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"run_{stamp}_{tag}.json"
    payload = {
        "created_at": report.created_at,
        "host": report.host,
        "pins": report.pins,
        "availability": report.availability,
        "runs": [asdict(r) for r in report.runs],
    }
    path.write_text(json.dumps(payload, indent=2))
    latest = RESULTS / "latest.json"
    latest.write_text(path.read_text())
    print(f"Wrote {path}")
    print(f"Wrote {latest}")
    return path


def cmd_probe(_: argparse.Namespace) -> None:
    avail = probe_availability()
    host = host_info()
    print(json.dumps({"host": host, "availability": avail}, indent=2))


def cmd_summarize(_: argparse.Namespace) -> None:
    latest = RESULTS / "latest.json"
    if not latest.exists():
        print("No results/latest.json yet", file=sys.stderr)
        sys.exit(1)
    data = json.loads(latest.read_text())
    print("=== Latest Phase 0 results ===")
    print(f"created_at: {data.get('created_at')}")
    print(f"host: {data.get('host')}")
    print(f"pins: {data.get('pins')}")
    for run in data.get("runs", []):
        print(
            f"- {run['backend']} {run['model']} [{run['prompt_id']}] "
            f"decode={run.get('decode_tok_s')} ttft_ms={run.get('ttft_ms')} "
            f"rss_mb={run.get('rss_mb')} err={run.get('error')}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gemma 4 Phase 0 baseline ladder")
    sub = p.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe", help="Detect installed runtimes")
    probe.set_defaults(func=cmd_probe)

    ollama = sub.add_parser("ollama", help="Bench Ollama model")
    ollama.add_argument("--model", default="gemma4:31b-mlx")
    ollama.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ollama.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    ollama.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ollama.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    ollama.add_argument("--quick", action="store_true")
    ollama.set_defaults(func=lambda a: save_report(run_ollama(a), "ollama"))

    mlx = sub.add_parser("mlx", help="Bench mlx-lm model")
    mlx.add_argument(
        "--model",
        default="mlx-community/gemma-4-e4b-it-4bit",
        help="HF repo or local path",
    )
    mlx.add_argument("--prompt-tokens", type=int, default=128)
    mlx.add_argument("--generation-tokens", type=int, default=128)
    mlx.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    mlx.add_argument("--num-trials", type=int, default=3)
    mlx.add_argument("--timeout", type=int, default=3600)
    mlx.add_argument("--also-generate", action="store_true", default=True)
    mlx.add_argument("--no-also-generate", action="store_false", dest="also_generate")
    mlx.add_argument("--draft-model", default=None, help="Optional MTP draft model")
    mlx.add_argument("--num-draft-tokens", type=int, default=5)
    mlx.set_defaults(func=lambda a: save_report(run_mlx(a), "mlx"))

    summarize = sub.add_parser("summarize", help="Print latest.json summary")
    summarize.set_defaults(func=cmd_summarize)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
