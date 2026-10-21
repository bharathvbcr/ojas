---
id: "gp-clip-paired-torch215-norm-only-and-bw-ceiling"
title: "Paired clip_grad_norm_full vs torch 2.15, first run of clip_norm_only, and a same-session bandwidth ceiling"
status: ready
priority: 1
severity: medium
type: bench
owner: "unassigned"
due: "none"
labels:
  - "bench"
  - "metal"
  - "torch-2.15"
  - "clip"
  - "research-2026-10-20"
repositories:
  - "ojas"
  - "tessl"
planned_files:
  - "bench/ojas_rows.rs"
  - "bench/torch_rows.py"
  - "bench/run_paired.sh"
  - "docs/bench-gpu-vs-torch.md"
acceptance_criteria:
  - "`clip_norm_only` compiles and passes parity on both lanes (ojas against torch.nn.utils.get_total_norm); the first call's check that the multiply is skipped is shown passing"
  - "torch 2.15 (RC now, GA after 2026-10-28) installed in its own venv; env.txt records torch version, git sha, macOS, ojas and tessl commits"
  - "`PYTHON=<venv>/bin/python BENCH_ROWS=clip_,adamw_full BENCH_ITERS=40 BENCH_WARMUP=10 BENCH_LANES='ojas-metal torch-mps' bench/run_paired.sh 7` run on a quiet machine (1-minute load < 6 at start, AC power, high power mode), committed under bench/results/<date>-clip-paired-torch215/ with min, median and per-round ratios per lane plus load per round"
  - "Scale-pass cost reported as clip_grad_norm_full − clip_norm_only for both ojas and torch, with achieved GB/s for each pass"
  - "A GPU read+write ceiling is measured in the same session with tessl's plain read-and-write reference pass (the 283 GB/s figure), and the clip row is reported as a % of it"
  - "docs/bench-gpu-vs-torch.md clip section quotes the paired ratio, replacing the unpaired ~1.13x; if paired ojas is slower than torch it is recorded as a red row, not dropped"
  - "The wgpu clip row re-timed after 0e3254f (the docs say it was never re-run) and the docs updated"
---

# Task brief v1

## Title
Paired clip_grad_norm_full vs torch 2.15, first run of clip_norm_only, and a same-session bandwidth ceiling

Task: gp-clip-paired-torch215-norm-only-and-bw-ceiling
Type: bench
Status: ready
Priority: 1
Source: research session 2026-10-20 (LinkedIn article audit, Qwen3.8-27B inference and megakernel research)

## Description
The ~1.13x (shipped `sq_norm`, 7.45 ms median) and ~1.2x (replaced kernel, 7.07 ms) figures against torch 2.15 are unpaired: the torch side (8.45 ms median, 2.15.0.dev20261004) was measured on 2026-10-05 under different load (`bench/results/2026-10-05-torch215-clip`). The LinkedIn article (MLSystemsLab/personal, v3) labels them "unpaired" and carries a `[PAIRED]` placeholder for this run.

`clip_norm_only` (befe116) was committed without being compiled or run. It times the norm pass alone (`max_norm` 1e30 so `clip_scale` is exactly 1), so `clip_grad_norm_full − clip_norm_only` approximates the in-place scale pass that the byte count says dominates (~0.49 GB norm read vs ~0.99 GB scale read+write at 123.7M f32 values).

PyTorch 2.15: branch cut 2026-10-05, final RC week of 2026-10-19, GA 2026-10-28. Run against the RC now, and again at GA.

Scope note: the full quiet GPU-vs-torch scorecard re-run belongs to `gp-long-runs-and-quiet-benches`. Add "against torch 2.15 GA" to that task rather than here.

## Acceptance criteria
- [ ] `clip_norm_only` compiles and passes parity on both lanes (ojas against torch.nn.utils.get_total_norm); the first call's check that the multiply is skipped is shown passing
- [ ] torch 2.15 (RC now, GA after 2026-10-28) installed in its own venv; env.txt records torch version, git sha, macOS, ojas and tessl commits
- [ ] `PYTHON=<venv>/bin/python BENCH_ROWS=clip_,adamw_full BENCH_ITERS=40 BENCH_WARMUP=10 BENCH_LANES="ojas-metal torch-mps" bench/run_paired.sh 7` run on a quiet machine (1-minute load < 6 at start, AC power, high power mode), committed under bench/results/<date>-clip-paired-torch215/ with min, median and per-round ratios per lane plus load per round
- [ ] Scale-pass cost reported as clip_grad_norm_full − clip_norm_only for both ojas and torch, with achieved GB/s for each pass
- [ ] A GPU read+write ceiling is measured in the same session with tessl's plain read-and-write reference pass (the 283 GB/s figure), and the clip row is reported as a % of it
- [ ] docs/bench-gpu-vs-torch.md clip section quotes the paired ratio, replacing the unpaired ~1.13x; if paired ojas is slower than torch it is recorded as a red row, not dropped
- [ ] The wgpu clip row re-timed after 0e3254f (the docs say it was never re-run) and the docs updated
