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
| `../ojas-metal/examples/metal_vs_torch.rs` | Opens `MetalBackend` and runs the rows. Metal ops are recorded and return before the device runs them (`docs/metal-deferred-faults.md`); `Backend::sync` after every iteration waits for them and reports any deferred fault. |
| `../ojas-wgpu/examples/wgpu_vs_torch.rs` | Opens `WgpuBackend` and runs the rows with `Backend::sync` (submit and wait) after every iteration. Its first output line records the adapter and the HAL. |
| `decode_rows.rs` | Whole-model decode rows (see "Decode rows"), generic over `ojas_core::Backend`, compiled with `ojas_rows.rs` into `ojas-infer/examples/decode_vs_torch.rs`. |
| `gate_saved_ab.rs` | Per-head gate A/B: the plain forward and backward against the saved-sigmoid pair (`forward_saving`, `backward_saved`) at nanolab's shape (4096 rows, d_model 768, 12 heads of 64). Generic over `ojas_core::Backend`. It is compiled by `#[path]` into the ignored `bench_gate_saved_against_recomputed` tests of `ojas-metal/tests/gate_saved.rs` and `ojas-wgpu/tests/gate_saved.rs`. Each round times every variant once per iteration, alternating which goes first, with `Backend::sync` per call, and reports the min and median. Results are in `results/2026-10-07-gate-saved/`. |
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
- **Load.** Besides the before and after snapshot of each runtime block
  (`load.jsonl`), every lane writes a `_load` record (uptime load and an
  `ioreg` GPU sample) before and after every row. `summary.md` ends with
  their ranges.
- **Timing.** Inputs are allocated once per row. 5 warm-ups (minimum), then 20
  timed iterations (minimum); each iteration is the op plus the device
  synchronize (`torch.mps.synchronize()`; `Backend::sync`, submit and wait,
  for Metal and wgpu). Each run reports min and median.
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
- `muon_RxC_bf16` (ojas side): the same step with `Ns5Precision::Bf16`, every
  NS5 intermediate rounded to bf16 as nanolab's is, against torch's
  `muon_RxC_bf16` reference. wgpu refuses it (`Unsupported`), so its row
  records that.
- `linear_ce_c{rows}x{cols}`: `linear_cross_entropy_mean`, N 4096, d 768,
  V 50304, loss plus both gradients, with logit chunks of `rows x cols`
  (1024x8192 and 4096x50304). The torch twin is nanolab's own
  `FusedLinearCrossEntropy` (forward + `backward()`). It chunks rows only
  (n_chunks = N / rows) over the full vocabulary, so its 1024x8192 twin uses
  1024x50304 chunks.
- `decode_attn_kv1024`: `cached_attention_forward`, one query per head
  (B 1, H = Hkv = 12, D 64) against a 1024-position time-major cache. torch
  runs `F.scaled_dot_product_attention` with no mask on the cached K/V, kept
  in its own [B, H, T, D] layout (made contiguous once, outside the timer).
- `accumulate_grad_50304x768`: `accumulate_grad` in place; torch `acc.add_(g)`.
- `block_fwd`, `block_fwd_bwd`: nanolab's real `Block` (attention mixer with
  QK-norm, RoPE, value residual, per-head gate; SwiGLU) at B=4, T=1024. Weights
  come from the generator (not nanolab's zero init, so every gradient is
  non-trivial). The ojas side composes Backend ops by hand, forward and
  backward (neither GPU crate depends on `ojas-autograd`, so `Tape` cannot
  drive it from these examples). Parity checks the output and seven gradients.
  The forward is 24 device ops and forward + backward 53 (a residual add's
  gradient passes through, so `residual_add_backward` is not called). On both
  backends the block records all of them and synchronizes once per iteration
  (the Metal worker commits early only when its 4096-slot status slab or
  device memory fills), while each per-kernel row synchronizes after its one op, so the block is not
  the sum of the kernel rows.
- `sweep_silu_n{N}`: one `silu_forward` over N values, N = 1, 2^12, 2^16,
  2^18, 2^20, 2^21, 2^22, 2^23. It splits each runtime's time into a fixed
  per-op part and a per-value part.
- `sweep_decode_b{B}`: B decode requests (one query per head, H = Hkv = 12,
  D 64, a 1024-position cache each) as one batched `cached_attention_forward`,
  B = 1, 2, 4, 8, 16. torch runs one SDPA over `[B, H, T, D]`.
- `sweep_decode_x{B}` (B > 1): the same B requests over the same bytes
  (request i is row i of the batched inputs), as B batch-1 calls recorded
  before one synchronize; torch runs B SDPAs on `[1, H, T, D]` slices. `x`
  minus `b` is what dispatching the requests one at a time costs.

## Decode rows

`decode_rows.rs`, compiled into `ojas-infer/examples/decode_vs_torch.rs`
(argument `cpu`, `cpu-host`, `metal` or `wgpu`), times whole-model decode:
nanolab's default GPT (124M, vocab 50304) with weights from the shared
generator (seed per parameter `fnv1a32(name) % 2^20 + 1000`, scales in the
file's docs), a 32-token prompt from `gen_targets(32, 1701)`.

- `gen_prefill_p32`: the prompt forward from an empty cache.
- `gen_greedy_p32_n32`: the prompt forward and 31 single-token forwards, each
  fed the previous argmax (32 ids; the last is not forwarded).

ojas runs `DeviceDecoder` on `CpuBackend` (`ojas-cpu`), Metal and wgpu, and
`CpuGpt` (`ojas-cpu-host`). torch runs nanolab's own KV-cached path,
`GPT.forward_hidden_window(..., commit=True, causal=True)` plus `lm_head` on
the last position, under `torch.no_grad()`. Both rows are gated on the
prompt's last-position logits (1e-3). Greedy ids are not a gate: a
`_gen_ids` record counts how many equal torch's. A `_gen_traffic` record
holds `DeviceDecoder::traffic()` per decode step (host tensors uploaded and
bytes, bytes read back). `aggregate.py` ends with a "Decode" table: per-token
ms = (greedy - prefill) / 31, tokens/s, and those records.

The metal and wgpu lanes run their kernel rows and then the decode rows into
the same lane file. The CPU lanes are opt-in: `BENCH_LANES="ojas-metal
ojas-wgpu ojas-cpu ojas-cpu-host torch-mps"`. Decode rows only:
`BENCH_ROWS=gen_`.

## Tokenizer throughput

`bpe_throughput.rs` (compiled into `ojas-data/examples/bpe_throughput.rs`)
times GPT-2 `encode_ordinary` and `decode_ordinary` on the real GPT-2 rank
table over the complete documents of nanolab's FineWeb-Edu `val.bin`
(11.4 MB of English), after checking every document's re-encode against the
tiktoken ids in the bin. It is CPU-only and does not use the torch side.
Runs, the runner and the reading are in
`results/2026-10-08-bpe-throughput/`.

## Results kept in the tree

`bench/results/<run>/` holds a quoted run's `summary.md`, `env.txt`,
`env_torch.json`, `load.jsonl` and every round's JSONL (not the ~200 MB
`ref/` directory, which `run_paired.sh` regenerates).
