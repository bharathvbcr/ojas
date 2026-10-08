# Benchmark Plots, Every Result Folder

Each chart is drawn from the text files kept under `bench/results/<folder>`, by `bench/plot_all.py`. Regenerate with `python3 -I bench/plot_all.py` from the repository root. Ratio charts are times, so **lower is faster** and 1x is parity. Green bars are faster by more than 10%, red are slower by more than 10%, grey is inside the 10% band, which is the noise this machine showed. A hollow bar means the run itself flagged the row noisy. Whiskers are the per-round range and circles are medians where the source gives them.

Two cautions on reading them. The paired GPU summaries print ratios to two decimals, so where the printed ratio is under 0.05x (ojas more than 20x slower) the chart uses the ratio of the two printed medians instead and draws no whisker. That is a different statistic from the median of per-round ratios, and on rows the run flagged noisy the two can differ by 2x or more. Dot charts of absolute times use a log axis, so distance between dots is a ratio, not a difference.

Read each folder's own README or summary before quoting a number: several runs were taken under heavy load, and the charts show direction, not a verdict. The older prose analysis is in [bench-gpu-vs-torch.md](bench-gpu-vs-torch.md) and [bench-cpu-vs-torch.md](bench-cpu-vs-torch.md).

## 2026-10-01

### 2026-10-01/runA: ojas-metal vs torch-mps

![2026-10-01/runA: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01--runA--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01/runA: ojas-wgpu vs torch-mps

![2026-10-01/runA: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01--runA--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01/runB: ojas-metal vs torch-mps

![2026-10-01/runB: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01--runB--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01/runB: ojas-wgpu vs torch-mps

![2026-10-01/runB: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01--runB--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01/runC: ojas-metal vs torch-mps

![2026-10-01/runC: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01--runC--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01/runC: ojas-wgpu vs torch-mps

![2026-10-01/runC: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01--runC--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 44 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

Source: `bench/results/2026-10-01/`.

## 2026-10-01-ab

### 2026-10-01-ab: new / old

![2026-10-01-ab: new / old: Old binary against new binary, 10 rounds, 5 rows.](assets/plots/bench/2026-10-01-ab--ab.svg)

*Old binary against new binary, 10 rounds, 5 rows.*

Source: `bench/results/2026-10-01-ab/`.

## 2026-10-01-floor

### 2026-10-01-floor: submit-and-wait floor, before and after the condvar wait

![2026-10-01-floor: submit-and-wait floor, before and after the condvar wait: Median and 90th-percentile wall time of tiny GPU calls. The first run was at load 45 with the GPU 100% busy.](assets/plots/bench/2026-10-01-floor--floor.svg)

*Median and 90th-percentile wall time of tiny GPU calls. The first run was at load 45 with the GPU 100% busy.*

### 2026-10-01-floor/ab: submit-and-wait floor, old against new

![2026-10-01-floor/ab: submit-and-wait floor, old against new: 600 runs per side. The old side had 388 to 507 slow (>1 ms) runs of 600 per scenario; the new side had 0 to 6.](assets/plots/bench/2026-10-01-floor--ab.svg)

*600 runs per side. The old side had 388 to 507 slow (>1 ms) runs of 600 per scenario; the new side had 0 to 6.*

Source: `bench/results/2026-10-01-floor/`.

## 2026-10-01-r2

### 2026-10-01-r2: ojas-metal vs torch-mps

![2026-10-01-r2: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r2--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01-r2: ojas-wgpu vs torch-mps

![2026-10-01-r2: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r2--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

Source: `bench/results/2026-10-01-r2/`.

## 2026-10-01-r3

### 2026-10-01-r3: ojas-metal vs torch-mps

![2026-10-01-r3: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 48 rows charted, 2 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r3--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 48 rows charted, 2 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01-r3: ojas-wgpu vs torch-mps

![2026-10-01-r3: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 48 rows charted, 2 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r3--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 48 rows charted, 2 paired rounds. Hollow bars were flagged noisy by the run.*

Source: `bench/results/2026-10-01-r3/`.

## 2026-10-01-r3b

### 2026-10-01-r3b: ojas-metal vs torch-mps

![2026-10-01-r3b: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r3b--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01-r3b: ojas-wgpu vs torch-mps

![2026-10-01-r3b: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r3b--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

Source: `bench/results/2026-10-01-r3b/`.

## 2026-10-01-r4-smallops

### 2026-10-01-r4-smallops: ojas-metal vs torch-mps

![2026-10-01-r4-smallops: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 19 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r4-smallops--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 19 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-01-r4-smallops: ojas-wgpu vs torch-mps

![2026-10-01-r4-smallops: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 19 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-01-r4-smallops--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 19 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

Source: `bench/results/2026-10-01-r4-smallops/`.

## 2026-10-02-gate

### 2026-10-02-gate/paired-gate: ojas-metal vs torch-mps

![2026-10-02-gate/paired-gate: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-02-gate--paired-gate--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-02-gate/paired-gate: ojas-wgpu vs torch-mps

![2026-10-02-gate/paired-gate: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-02-gate--paired-gate--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-02-gate/paired-gate-quiet: ojas-metal vs torch-mps

![2026-10-02-gate/paired-gate-quiet: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-02-gate--paired-gate-quiet--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-02-gate/paired-gate-quiet: ojas-wgpu vs torch-mps

![2026-10-02-gate/paired-gate-quiet: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-02-gate--paired-gate-quiet--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 2 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-02-gate/ab-dbias: new / old

![2026-10-02-gate/ab-dbias: new / old: Interleaved A/B, 4 rounds per side, 8 rows.](assets/plots/bench/2026-10-02-gate--ab-dbias.svg)

*Interleaved A/B, 4 rounds per side, 8 rows.*

### 2026-10-02-gate/ab-fold: new / old

![2026-10-02-gate/ab-fold: new / old: Interleaved A/B, 4 rounds per side, 7 rows.](assets/plots/bench/2026-10-02-gate--ab-fold.svg)

*Interleaved A/B, 4 rounds per side, 7 rows.*

### 2026-10-02-gate/ab-tn: new / old

![2026-10-02-gate/ab-tn: new / old: Interleaved A/B, 4 rounds per side, 8 rows.](assets/plots/bench/2026-10-02-gate--ab-tn.svg)

*Interleaved A/B, 4 rounds per side, 8 rows.*

### 2026-10-02-gate/ab-stage: new / old

![2026-10-02-gate/ab-stage: new / old: Interleaved A/B, 4 rounds per side, 8 rows.](assets/plots/bench/2026-10-02-gate--ab-stage.svg)

*Interleaved A/B, 4 rounds per side, 8 rows.*

### 2026-10-02-gate/tn-sweep-1: TN GEMM, single dispatch against split-K

![2026-10-02-gate/tn-sweep-1: TN GEMM, single dispatch against split-K: Contended (other jobs running; the readme says -2 supersedes it). Split-K at width 128 to 2048 against the single dispatch and what tessl routes to.](assets/plots/bench/2026-10-02-gate--tn-sweep-1.svg)

*Contended (other jobs running; the readme says -2 supersedes it). Split-K at width 128 to 2048 against the single dispatch and what tessl routes to.*

### 2026-10-02-gate/tn-sweep-2: TN GEMM, single dispatch against split-K

![2026-10-02-gate/tn-sweep-2: TN GEMM, single dispatch against split-K: Quiet slot. Split-K at width 128 to 2048 against the single dispatch and what tessl routes to.](assets/plots/bench/2026-10-02-gate--tn-sweep-2.svg)

*Quiet slot. Split-K at width 128 to 2048 against the single dispatch and what tessl routes to.*

### 2026-10-02-gate: gate_bwd dispatch profile before the fix

![2026-10-02-gate: gate_bwd dispatch profile before the fix: ojas_per_head_gate_dbias (the serial bias sum) is the 800 µs outlier the fix removes.](assets/plots/bench/2026-10-02-gate--gate-probe.svg)

*ojas_per_head_gate_dbias (the serial bias sum) is the 800 µs outlier the fix removes.*

Source: `bench/results/2026-10-02-gate/`.

## 2026-10-02-gemm

### 2026-10-02-gemm/ab-grid-gate: new / old

![2026-10-02-gemm/ab-grid-gate: new / old: Interleaved A/B, 4 rounds per side, 26 rows.](assets/plots/bench/2026-10-02-gemm--ab-grid-gate.svg)

*Interleaved A/B, 4 rounds per side, 26 rows.*

### 2026-10-02-gemm/ab-footprint-gate: new / old

![2026-10-02-gemm/ab-footprint-gate: new / old: Interleaved A/B, 4 rounds per side, 26 rows.](assets/plots/bench/2026-10-02-gemm--ab-footprint-gate.svg)

*Interleaved A/B, 4 rounds per side, 26 rows.*

### 2026-10-02-gemm/ab-nn-splitk: new / old

![2026-10-02-gemm/ab-nn-splitk: new / old: Interleaved A/B, 4 rounds per side, 26 rows.](assets/plots/bench/2026-10-02-gemm--ab-nn-splitk.svg)

*Interleaved A/B, 4 rounds per side, 26 rows.*

### 2026-10-02-gemm/ab-rows: new / old

![2026-10-02-gemm/ab-rows: new / old: Old binary against new binary, 6 rounds, 3 rows.](assets/plots/bench/2026-10-02-gemm--ab-rows.svg)

*Old binary against new binary, 6 rounds, 3 rows.*

### 2026-10-02-gemm/ab-rows-ce: new / old

![2026-10-02-gemm/ab-rows-ce: new / old: Old binary against new binary, 6 rounds, 3 rows.](assets/plots/bench/2026-10-02-gemm--ab-rows-ce.svg)

*Old binary against new binary, 6 rounds, 3 rows.*

### 2026-10-02-gemm: min time per row

![2026-10-02-gemm: min time per row: Direction only. These rounds ran at load around 350.](assets/plots/bench/2026-10-02-gemm--rows.svg)

*Direction only. These rounds ran at load around 350.*

### 2026-10-02-gemm/gemm.md: exact-f32 GEMM throughput

![2026-10-02-gemm/gemm.md: exact-f32 GEMM throughput: Before the panel-walk fix. The LM-head nt and nn rows drop to about 2 TFLOP/s while tn stays near 5.5.](assets/plots/bench/2026-10-02-gemm--gemm-md.svg)

*Before the panel-walk fix. The LM-head nt and nn rows drop to about 2 TFLOP/s while tn stays near 5.5.*

### 2026-10-02-gemm/gemm-relabelled.md: exact-f32 GEMM throughput

![2026-10-02-gemm/gemm-relabelled.md: exact-f32 GEMM throughput: Before the panel-walk fix. The LM-head nt and nn rows drop to about 2 TFLOP/s while tn stays near 5.5.](assets/plots/bench/2026-10-02-gemm--gemm-relabelled-md.svg)

*Before the panel-walk fix. The LM-head nt and nn rows drop to about 2 TFLOP/s while tn stays near 5.5.*

Source: `bench/results/2026-10-02-gemm/`.

## 2026-10-02-kernels

### 2026-10-02-kernels/ab-check: new / old

![2026-10-02-kernels/ab-check: new / old: Old binary against new binary, 8 rounds, 17 rows.](assets/plots/bench/2026-10-02-kernels--ab-check.svg)

*Old binary against new binary, 8 rounds, 17 rows.*

### 2026-10-02-kernels/ab-fold: new / old

![2026-10-02-kernels/ab-fold: new / old: Old binary against new binary, 8 rounds, 13 rows.](assets/plots/bench/2026-10-02-kernels--ab-fold.svg)

*Old binary against new binary, 8 rounds, 13 rows.*

### 2026-10-02-kernels/kernels.md

![2026-10-02-kernels/kernels.md: Per-dispatch GPU time at the minimum. The sequence rows are the old separate finite-check passes.](assets/plots/bench/2026-10-02-kernels--kernels-md.svg)

*Per-dispatch GPU time at the minimum. The sequence rows are the old separate finite-check passes.*

### 2026-10-02-kernels/kernels-after-fold.md

![2026-10-02-kernels/kernels-after-fold.md: Per-dispatch GPU time at the minimum. The sequence rows are the old separate finite-check passes.](assets/plots/bench/2026-10-02-kernels--kernels-after-fold-md.svg)

*Per-dispatch GPU time at the minimum. The sequence rows are the old separate finite-check passes.*

### 2026-10-02-kernels: achieved bandwidth, before the fold

![2026-10-02-kernels: achieved bandwidth, before the fold: Memory bandwidth reached by each kernel before the finite checks were folded in.](assets/plots/bench/2026-10-02-kernels--bandwidth.svg)

*Memory bandwidth reached by each kernel before the finite checks were folded in.*

Source: `bench/results/2026-10-02-kernels/`.

## 2026-10-02-r5

### 2026-10-02-r5: ojas-metal vs torch-mps

![2026-10-02-r5: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-02-r5--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-02-r5: ojas-wgpu vs torch-mps

![2026-10-02-r5: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-02-r5--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 48 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

Source: `bench/results/2026-10-02-r5/`.

## 2026-10-03-lappi-inference

### 2026-10-03-lappi-inference: Qwen3.5-2B prefill

![2026-10-03-lappi-inference: Qwen3.5-2B prefill: tessl against PyTorch 2.12.1 MPS bf16. ojas has no Qwen3.5 inference path, so it is not in this comparison. The README tables were built from the raw logs in this folder; the charts read the tables.](assets/plots/bench/2026-10-03-lappi-inference--prefill.svg)

*tessl against PyTorch 2.12.1 MPS bf16. ojas has no Qwen3.5 inference path, so it is not in this comparison. The README tables were built from the raw logs in this folder; the charts read the tables.*

### 2026-10-03-lappi-inference: tessl full decision time

![2026-10-03-lappi-inference: tessl full decision time: Base and Lappi weights run at the same speed, as expected for equal shapes.](assets/plots/bench/2026-10-03-lappi-inference--decision.svg)

*Base and Lappi weights run at the same speed, as expected for equal shapes.*

Source: `bench/results/2026-10-03-lappi-inference/`.

## 2026-10-03-lappi-inference-rerun

### 2026-10-03-lappi-inference-rerun: Qwen3.5-2B prefill

![2026-10-03-lappi-inference-rerun: Qwen3.5-2B prefill: tessl against PyTorch 2.12.1 MPS bf16. ojas has no Qwen3.5 inference path, so it is not in this comparison. The README tables were built from the raw logs in this folder; the charts read the tables.](assets/plots/bench/2026-10-03-lappi-inference-rerun--prefill.svg)

*tessl against PyTorch 2.12.1 MPS bf16. ojas has no Qwen3.5 inference path, so it is not in this comparison. The README tables were built from the raw logs in this folder; the charts read the tables.*

### 2026-10-03-lappi-inference-rerun: tessl full decision time

![2026-10-03-lappi-inference-rerun: tessl full decision time: Base and Lappi weights run at the same speed, as expected for equal shapes.](assets/plots/bench/2026-10-03-lappi-inference-rerun--decision.svg)

*Base and Lappi weights run at the same speed, as expected for equal shapes.*

Source: `bench/results/2026-10-03-lappi-inference-rerun/`.

## 2026-10-04-percall

### 2026-10-04-percall: Metal decode call cost

![2026-10-04-percall: Metal decode call cost: One request, one batched call, and sixteen separate calls.](assets/plots/bench/2026-10-04-percall--percall.svg)

*One request, one batched call, and sixteen separate calls.*

### 2026-10-04-percall: GPU span of one command buffer

![2026-10-04-percall: GPU span of one command buffer: Batched dispatch against sixteen separate dispatches.](assets/plots/bench/2026-10-04-percall--gpu-span.svg)

*Batched dispatch against sixteen separate dispatches.*

Source: `bench/results/2026-10-04-percall/`.

## 2026-10-04-split

### 2026-10-04-split/ab: new / old

![2026-10-04-split/ab: new / old: Old binary against new binary, 8 rounds, 9 rows.](assets/plots/bench/2026-10-04-split--ab.svg)

*Old binary against new binary, 8 rounds, 9 rows.*

### 2026-10-04-split: Metal decode call cost

![2026-10-04-split: Metal decode call cost: One request, one batched call, and sixteen separate calls.](assets/plots/bench/2026-10-04-split--percall.svg)

*One request, one batched call, and sixteen separate calls.*

### 2026-10-04-split: GPU span of one command buffer

![2026-10-04-split: GPU span of one command buffer: Batched dispatch against sixteen separate dispatches.](assets/plots/bench/2026-10-04-split--gpu-span.svg)

*Batched dispatch against sixteen separate dispatches.*

### 2026-10-04-split: split count for the decode attention kernel

![2026-10-04-split: split count for the decode attention kernel: Time bottoms out near eight splits.](assets/plots/bench/2026-10-04-split--splits.svg)

*Time bottoms out near eight splits.*

Source: `bench/results/2026-10-04-split/`.

## 2026-10-04-sweep

### 2026-10-04-sweep: ojas-metal vs torch-mps

![2026-10-04-sweep: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 17 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-04-sweep--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 17 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-04-sweep: ojas-wgpu vs torch-mps

![2026-10-04-sweep: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 17 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-04-sweep--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 17 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-04-sweep: silu_forward time against size

![2026-10-04-sweep: silu_forward time against size: Below roughly 4M elements every runtime sits on its floor.](assets/plots/bench/2026-10-04-sweep--silu-over-n.svg)

*Below roughly 4M elements every runtime sits on its floor.*

### 2026-10-04-sweep: decode cost per request

![2026-10-04-sweep: decode cost per request: Batching amortises the per-call cost; the gap between split and batched is the extra per call.](assets/plots/bench/2026-10-04-sweep--decode-batching.svg)

*Batching amortises the per-call cost; the gap between split and batched is the extra per call.*

Source: `bench/results/2026-10-04-sweep/`.

## 2026-10-05-torch215-clip

### 2026-10-05-torch215-clip: clip_grad_norm on MPS

![2026-10-05-torch215-clip: clip_grad_norm on MPS: The folder's README attributes the 16x drop to a change in torch's MPS norm reduction and says a paired run with torch 2.15 is still needed before a new ratio is quoted.](assets/plots/bench/2026-10-05-torch215-clip--clip.svg)

*The folder's README attributes the 16x drop to a change in torch's MPS norm reduction and says a paired run with torch 2.15 is still needed before a new ratio is quoted.*

Source: `bench/results/2026-10-05-torch215-clip/`.

## 2026-10-06-gemm-bf16

### 2026-10-06-gemm-bf16: column-panel tile walk, new / old

![2026-10-06-gemm-bf16: column-panel tile walk, new / old: The largest-B cases fall to 0.5 to 0.8 of the old time; the Morton-order and small-B controls stay near 1.0.](assets/plots/bench/2026-10-06-gemm-bf16--ab2.svg)

*The largest-B cases fall to 0.5 to 0.8 of the old time; the Morton-order and small-B controls stay near 1.0.*

### 2026-10-06-gemm-bf16: panel-height sweep

![2026-10-06-gemm-bf16: panel-height sweep: Panel heights 4, 8 and 16 against the shipped kernel across six B sizes.](assets/plots/bench/2026-10-06-gemm-bf16--sweep.svg)

*Panel heights 4, 8 and 16 against the shipped kernel across six B sizes.*

### 2026-10-06-gemm-bf16: tune1 raw sweep

![2026-10-06-gemm-bf16: tune1 raw sweep: Raw times behind sweep/ratios.txt. Variants whose name ends _phN are column-panel heights.](assets/plots/bench/2026-10-06-gemm-bf16--tune1-raw.svg)

*Raw times behind sweep/ratios.txt. Variants whose name ends _phN are column-panel heights.*

### 2026-10-06-gemm-bf16: tune2 raw sweep

![2026-10-06-gemm-bf16: tune2 raw sweep: Raw times behind sweep/ratios.txt. Variants whose name ends _phN are column-panel heights.](assets/plots/bench/2026-10-06-gemm-bf16--tune2-raw.svg)

*Raw times behind sweep/ratios.txt. Variants whose name ends _phN are column-panel heights.*

Source: `bench/results/2026-10-06-gemm-bf16/`.

## 2026-10-07-gate-saved

### 2026-10-07-gate-saved: saved-sigmoid gate pair against recompute

![2026-10-07-gate-saved: saved-sigmoid gate pair against recompute: The saved backward is 7% faster on Metal and 15% faster on wgpu; the forward costs at most about 2%.](assets/plots/bench/2026-10-07-gate-saved--gate-saved.svg)

*The saved backward is 7% faster on Metal and 15% faster on wgpu; the forward costs at most about 2%.*

Source: `bench/results/2026-10-07-gate-saved/`.

## 2026-10-07-muon-bf16

### 2026-10-07-muon-bf16: ojas-metal vs torch-mps

![2026-10-07-muon-bf16: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 9 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-07-muon-bf16--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 9 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-07-muon-bf16: ojas-wgpu vs torch-mps

![2026-10-07-muon-bf16: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 6 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run. 3 further row(s) have no ratio and are not charted: `muon_768x768_bf16` (r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here); `muon_2048x768_bf16` (r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here); `muon_768x2048_bf16` (r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here).](assets/plots/bench/2026-10-07-muon-bf16--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 6 rows charted, 5 paired rounds. Hollow bars were flagged noisy by the run. 3 further row(s) have no ratio and are not charted: `muon_768x768_bf16` (r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here); `muon_2048x768_bf16` (r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here); `muon_768x2048_bf16` (r1 ojas refused: muon_ns5_step: unsupported: wgpu runs Newton-Schulz in f32 only; Bf16 is not implemented here).*

Source: `bench/results/2026-10-07-muon-bf16/`.

## 2026-10-08-attn-lse-ab

### 2026-10-08-attn-lse-ab: attention backward with saved log-sum-exp

![2026-10-08-attn-lse-ab: attention backward with saved log-sum-exp: Backward is 0.76 to 0.81 of the old time on every shape and both backends. Forward is within noise on the multi-head shapes; the grouped-query shape's Metal forward is 0.89.](assets/plots/bench/2026-10-08-attn-lse-ab--new-over-old.svg)

*Backward is 0.76 to 0.81 of the old time on every shape and both backends. Forward is within noise on the multi-head shapes; the grouped-query shape's Metal forward is 0.89.*

### 2026-10-08-attn-lse-ab: sliding window of 256 against full attention

![2026-10-08-attn-lse-ab: sliding window of 256 against full attention: The 256-wide window cuts backward time by roughly 3x at 2048 tokens.](assets/plots/bench/2026-10-08-attn-lse-ab--window.svg)

*The 256-wide window cuts backward time by roughly 3x at 2048 tokens.*

### 2026-10-08-attn-lse-ab: scratch charged at the Qwen3.5 shape

![2026-10-08-attn-lse-ab: scratch charged at the Qwen3.5 shape: Reading each KV head in place cuts the charged scratch to about a third of the expanded-heads path (forward) and 27% (backward).](assets/plots/bench/2026-10-08-attn-lse-ab--gqa-scratch.svg)

*Reading each KV head in place cuts the charged scratch to about a third of the expanded-heads path (forward) and 27% (backward).*

Source: `bench/results/2026-10-08-attn-lse-ab/`.

## 2026-10-08-cpu-hot-paths

### 2026-10-08-cpu-hot-paths: interleaved A/B, base against after

![2026-10-08-cpu-hot-paths: interleaved A/B, base against after: 30 rows. Ratios between 0.9 and 1.1 are noise on this machine.](assets/plots/bench/2026-10-08-cpu-hot-paths--ab-vs-base.svg)

*30 rows. Ratios between 0.9 and 1.1 are noise on this machine.*

### 2026-10-08-cpu-hot-paths: ojas / torch 2.13 CPU

![2026-10-08-cpu-hot-paths: ojas / torch 2.13 CPU: 11 rows with a torch counterpart.](assets/plots/bench/2026-10-08-cpu-hot-paths--ab-vs-torch.svg)

*11 rows with a torch counterpart.*

### 2026-10-08-cpu-hot-paths: tape_bench re-run (ab-tape/)

![2026-10-08-cpu-hot-paths: tape_bench re-run (ab-tape/): 3 rows. Ratios between 0.9 and 1.1 are noise on this machine.](assets/plots/bench/2026-10-08-cpu-hot-paths--ab-tape-vs-base.svg)

*3 rows. Ratios between 0.9 and 1.1 are noise on this machine.*

### 2026-10-08-cpu-hot-paths: Follow-up edits against a9af0a0 (ab-v2/)

![2026-10-08-cpu-hot-paths: Follow-up edits against a9af0a0 (ab-v2/): 10 rows. Ratios between 0.9 and 1.1 are noise on this machine.](assets/plots/bench/2026-10-08-cpu-hot-paths--ab-v2-vs-base.svg)

*10 rows. Ratios between 0.9 and 1.1 are noise on this machine.*

### 2026-10-08-cpu-hot-paths: Muon no-transpose view gate

![2026-10-08-cpu-hot-paths: Muon no-transpose view gate: Opening the view to every tall matrix (tall, tallbands) reads 1.3 to 2.9x slower than shipped, so the gate stays. Gate off reads 0.78 to 1.21 and is noise-bound at load 20-35 (the README calls 2048x768 on versus off within noise).](assets/plots/bench/2026-10-08-cpu-hot-paths--muon-gate.svg)

*Opening the view to every tall matrix (tall, tallbands) reads 1.3 to 2.9x slower than shipped, so the gate stays. Gate off reads 0.78 to 1.21 and is noise-bound at load 20-35 (the README calls 2048x768 on versus off within noise).*

### 2026-10-08-cpu-hot-paths: X times X-transpose, three ways

![2026-10-08-cpu-hot-paths: X times X-transpose, three ways: The ssyrk product matched sgemm bit for bit at every probed shape (bits_syrk_eq_gemm).](assets/plots/bench/2026-10-08-cpu-hot-paths--syrk.svg)

*The ssyrk product matched sgemm bit for bit at every probed shape (bits_syrk_eq_gemm).*

### 2026-10-08-cpu-hot-paths: ssyrk lower-from-upper mirror

![2026-10-08-cpu-hot-paths: ssyrk lower-from-upper mirror: 16 x 16 tiles shipped. At n = 2048 the 32 and 64 tile sizes are slower than the plain column walk.](assets/plots/bench/2026-10-08-cpu-hot-paths--mirror.svg)

*16 x 16 tiles shipped. At n = 2048 the 32 and 64 tile sizes are slower than the plain column walk.*

### 2026-10-08-cpu-hot-paths: cost of one std::thread::scope

![2026-10-08-cpu-hot-paths: cost of one std::thread::scope: About 9 µs for one thread and 34-38 µs for five at the minimum.](assets/plots/bench/2026-10-08-cpu-hot-paths--spawn.svg)

*About 9 µs for one thread and 34-38 µs for five at the minimum.*

### 2026-10-08-cpu-hot-paths: tape walk peak charge

![2026-10-08-cpu-hot-paths: tape walk peak charge: Fan-in peaks at 18,874,368 then 12,582,912 bytes. The seed one-quarter fused-CE walk falls from 154,533,892 to 4 bytes.](assets/plots/bench/2026-10-08-cpu-hot-paths--tape-peak-bytes.svg)

*Fan-in peaks at 18,874,368 then 12,582,912 bytes. The seed one-quarter fused-CE walk falls from 154,533,892 to 4 bytes.*

### 2026-10-08-cpu-hot-paths: tape walk time in the peak run

![2026-10-08-cpu-hot-paths: tape walk time in the peak run: One run per side under load; use the interleaved ab-tape table for time, not this chart.](assets/plots/bench/2026-10-08-cpu-hot-paths--tape-peak-time.svg)

*One run per side under load; use the interleaved ab-tape table for time, not this chart.*

Source: `bench/results/2026-10-08-cpu-hot-paths/`.

## 2026-10-08-decode-before

### 2026-10-08-decode-before: ojas-metal vs torch-mps

![2026-10-08-decode-before: ojas-metal vs torch-mps: ojas-metal vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-08-decode-before--ojas-metal-vs-torch-mps.svg)

*ojas-metal vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-08-decode-before: ojas-wgpu vs torch-mps

![2026-10-08-decode-before: ojas-wgpu vs torch-mps: ojas-wgpu vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-08-decode-before--ojas-wgpu-vs-torch-mps.svg)

*ojas-wgpu vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-08-decode-before: ojas-cpu vs torch-mps

![2026-10-08-decode-before: ojas-cpu vs torch-mps: ojas-cpu vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-08-decode-before--ojas-cpu-vs-torch-mps.svg)

*ojas-cpu vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-08-decode-before: ojas-cpu-host vs torch-mps

![2026-10-08-decode-before: ojas-cpu-host vs torch-mps: ojas-cpu-host vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.](assets/plots/bench/2026-10-08-decode-before--ojas-cpu-host-vs-torch-mps.svg)

*ojas-cpu-host vs torch-mps, 2 rows charted, 3 paired rounds. Hollow bars were flagged noisy by the run.*

### 2026-10-08-decode-before: generation, 32-token prompt and 32 new tokens

![2026-10-08-decode-before: generation, 32-token prompt and 32 new tokens: Every ojas lane produced the same 32 greedy ids as torch. ojas-cpu decodes at 5.09 ms per token against torch-mps at 5.88; the GPU lanes are slower.](assets/plots/bench/2026-10-08-decode-before--decode-tokens.svg)

*Every ojas lane produced the same 32 greedy ids as torch. ojas-cpu decodes at 5.09 ms per token against torch-mps at 5.88; the GPU lanes are slower.*

Source: `bench/results/2026-10-08-decode-before/`.

## 2026-10-08-gpu-step-ab

No chart: the folder holds only scripts, logs or text that is not a measurement table.
