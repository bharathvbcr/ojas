# bench: ojas GPU backends against PyTorch MPS

A paired A/B benchmark of `MetalBackend` (`ojas-metal`) and `WgpuBackend`
(`ojas-wgpu`, which runs on wgpu's Metal HAL on macOS) against PyTorch MPS, at
nanolab's default shapes. Results and their reading are in
[`docs/bench-gpu-vs-torch.md`](../docs/bench-gpu-vs-torch.md).

## Reproduce

```bash
# Every row, 5 alternating rounds, 5 warm-ups + 20 timed iterations per row.
bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 5

# A subset (row-name prefixes), more iterations, a chosen output directory.
BENCH_ROWS=sdpa,block BENCH_ITERS=30 OUT_DIR=/tmp/gpu-bench \
    bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 7
```

The runner prints the output directory; `summary.md` in it holds the tables.
Requirements: Apple silicon, the `tessl` checkout beside `ojas` (ojas-metal's
path dependency), Python 3.14 with torch 2.13 (MPS), and the nanolab checkout
at `/Users/bharath/Code/research/MLSystemsLab` (the torch side imports its real
`Block`, `GPT`, `Config`, `apply_rope` and `zeropower_via_newtonschulz5`;
bytecode writing is disabled so that checkout is not touched).

## Files

| file | what it does |
| :-- | :-- |
| `run_paired.sh` | Builds the two release examples first (a stale binary looks like a valid run), records the toolchain (`env.txt`, `env_torch.json`), writes the torch references, then runs N rounds. Odd rounds run Metal, wgpu, torch; even rounds torch, wgpu, Metal. `uptime` load and `ioreg` "Device Utilization %" are recorded before and after every runtime block (`load.jsonl`). A crashed lane is recorded as such and the run continues. Ends with `aggregate.py`. |
| `ojas_rows.rs` | The ojas side, generic over `ojas_core::Backend`: the shared input generator, the parity gate and the timing loop, and every row. Compiled into both examples by `#[path]`. |
| `../ojas-metal/examples/metal_vs_torch.rs` | Opens `MetalBackend` and runs the rows. Metal ops synchronize before they return, so the per-iteration sync hook is a no-op. |
| `../ojas-wgpu/examples/wgpu_vs_torch.rs` | Opens `WgpuBackend` and runs the rows with `WgpuBackend::sync` after every iteration. Its first output line records the adapter and the HAL. |
| `torch_rows.py` | The torch twin of every row. `ref` writes the reference outputs, `time` times the rows, `env` records versions and checks the optimizer parameter list against `GPT(Config())`. |
| `aggregate.py` | Per-round ratios, spread flags, the ranked list and the load table, as markdown. |

## Protocol

- **Same inputs, bit for bit.** Both sides generate every input from one
  32-bit integer hash (`hash32` / `_hash`) with power-of-two scales, so the
  float conversion is exact on both. Each ojas binary first compares 192 values
  against `generator.f32`, which the torch `ref` pass wrote, and refuses to run
  on a mismatch.
- **Parity before timing.** Each ojas row runs once, downloads its outputs and
  compares them with the torch reference for the same row. The reference file
  carries the row's input `spec` string; a different spec (a stale reference)
  is `no_ref`, not a pass. Outputs above 2^22 values are compared on every
  `n / 2^22`-th value (both sides take the same indices). The gate is
  `max |ojas - torch| / max |torch| <= 1e-3` per output (1e-2 for Muon, 0 for
  permute). A row that fails, is refused, or has no reference is reported with
  that status and is not timed.
- **Timing.** Inputs are allocated once per row. 5 warm-ups (minimum), then 20
  timed iterations (minimum); each iteration is the op plus the device
  synchronize (`torch.mps.synchronize()`; nothing extra for Metal;
  `WgpuBackend::sync` for wgpu). Each run reports min and median.
- **Pairing.** Per row, `ratio = torch median / ojas median` within a round
  (> 1: ojas faster). The summary reports the median ratio over rounds with its
  min and max, and each side's spread (max / min of its per-round medians). A
  row with either spread above 10% is flagged "noisy - not quoted".

## Row semantics

- `*_fwd`: torch runs under `torch.no_grad()`.
- `*_bwd`: torch builds the graph once outside the timer and times
  `torch.autograd.grad(..., retain_graph=True)`, so it reuses its saved
  tensors (SDPA's attention weights or logsumexp, RMSNorm's rstd, CE's
  log-softmax). ojas backward methods take the forward inputs and recompute
  what they need. Both are each framework's real backward cost; the
  `block_fwd_bwd` row is the combined training number.
- `clip_grad_norm_full`: all 170 nanolab parameter shapes (123,699,612 values).
  The first call uses `max_norm = 1.0` (total norm is about 100); each later
  call uses 0.9 times the previous norm, computed outside the timer, so every
  timed call scales on both sides (ojas skips the multiply when it would not
  scale; torch always multiplies).
- `adamw_full`: the same 170 tensors; ojas makes one `adamw_step` call per
  tensor, torch one `torch.optim.AdamW.step()` with its default implementation
  selection. Hyperparameters are ojas's (betas 0.9/0.95, eps 1e-8), lr 1e-3,
  weight decay 0.1, passed explicitly to torch.
- `muon_RxC`: one matrix of nanolab's `Muon.step` (momentum 0.99, Nesterov,
  NS5, lr 0.025, decay 0.1). ojas runs NS5 with f32 GEMMs; its reference is
  nanolab's NS5 with the bf16 cast replaced by f32. `muon_RxC vs muon_RxC_bf16`
  rows compare the same ojas time with nanolab's unchanged bf16 NS5 (timing only).
- `block_fwd`, `block_fwd_bwd`: nanolab's real `Block` (attention mixer with
  QK-norm, RoPE, value residual, per-head gate; SwiGLU) at B=4, T=1024. Weights
  come from the generator (not nanolab's zero init, so every gradient is
  non-trivial). The ojas side composes Backend ops by hand, forward and
  backward (neither GPU crate depends on `ojas-autograd`, so `Tape` cannot
  drive it from these examples). Parity checks the output and seven gradients.
  The forward is 24 device ops and forward + backward 53 (a residual add's
  gradient passes through, so `residual_add_backward` is not called). On wgpu
  the block records all of them and synchronizes once per iteration, while
  each per-kernel row synchronizes after its one op, so the block is not the
  sum of the kernel rows. On Metal every op synchronizes before it returns, in
  both.

## Results kept in the tree

`bench/results/<run>/` holds a quoted run's `summary.md`, `env.txt`,
`env_torch.json`, `load.jsonl` and every round's JSONL (not the ~200 MB
`ref/` directory, which `run_paired.sh` regenerates).
