# Decode rows after the decode-path changes (2026-10-10)

Task `gp-inference-decode-path`. The after-change runs of `bench/decode_rows.rs`
beside `../2026-10-08-decode-before/` (taken at `8967457`, before `7f0a9f3`,
`e12d347` and `84ab777`, which no run had measured since).

- **`base/`** is `de5efaf`: the decode rows of this task (`gen_topk50_p32_n32`,
  the GQA rows, one traffic record per row) on the decoder as it was, where
  `generate` with a `top_k` reads the whole `[vocab]` row back every token.
- **`after/`** is `2b68d11`: `DeviceDecoder::forward_top`, which selects the
  leaders on the device (`Backend::topk_rows`) and reads back only them.
- **`ab-k1/`** is `cefc864`, the outcome: `forward_top` selects on the
  device only for one leader (`k` = 1) and reads the row back for `k` > 1,
  because `ab/` and `probes/` below show the device selection slower at
  every `k` > 1. See "`ab-k1/`" below.

Both are `bench/run_paired.sh 3` with `BENCH_ROWS=gen_` and all five lanes
(`ojas-metal ojas-wgpu ojas-cpu ojas-cpu-host torch-mps`), built from a clean
worktree of each commit (`dirty files: 0` in `env.txt`). Apple M5 Pro, macOS
27.0.1, rustc 1.99.0, torch 2.13.0. Other sessions shared the machine: the
1-minute load was 6–13 during `base/` and 5–29 during `after/` (15–29 on the
CPU and torch lanes), per `load.jsonl` and each `summary.md`'s load tables.
Read every timing against that.

## What changed: bytes per decode step (`_gen_traffic`)

From `DeviceDecoder::traffic()`, per decode step of `gen_topk50_p32_n32`
(uploads / bytes up / bytes read back). These counters do not depend on load.

| lane | base | after |
| :-- | :-- | :-- |
| ojas-metal | 1 / 4 B / 201,216 B (1 readback) | 1 / 4 B / 400 B (2 readbacks: 50 values, 50 ids) |
| ojas-wgpu | 1 / 4 B / 201,216 B (1 readback) | 1 / 4 B / 400 B (2 readbacks) |
| ojas-cpu | 1 / 4 B / 201,216 B | 1 / 4 B / 0 B (no device tensor) |

The greedy rows read back the 4-byte id and upload nothing per step on both
(`7f0a9f3`); `2026-10-08-decode-before` had 3 uploads / 516 B / 201,216 B
there. The one upload per sampled step is the drawn id: with `k` > 1 the
host draws it.

## Timing

Each `summary.md` holds the per-row tables against torch-mps and the Decode
table (per-token ms = (decode - prefill) / 31).

These two runs do not resolve `forward_top`'s effect on time. Per token,
`gen_topk50` went 10.47 to 9.56 ms on Metal and 23.55 to 24.49 ms on wgpu,
but the greedy row, whose code is the same in both commits, moved by as
much (Metal 9.79 to 8.72, wgpu 23.93 to 23.11), and most rows are flagged
noisy (spread over 10%). `ab/` answers it.

### `ab/`: interleaved A/B of the two decode binaries

`ab_decode.sh`: the `decode_vs_torch` binaries of `de5efaf` (before) and
`2b68d11` (after), 6 rounds, each round running Metal then wgpu, before and
after back to back, in alternating order (odd rounds before first), rows
`gen_prefill_p32`, `gen_greedy_p32_n32` and `gen_topk50_p32_n32` against
`after/`'s torch references (every row passed its gate). The greedy row is
the control: its code is the same in both binaries. Medians were contended
(the control's after/before ratio of medians ranged 0.81 to 4.9 across the two lanes), so the table quotes
each row's minimum (`benchreport decode-ab ab min_ms`; `summary.md` has both).

| lane | row | before ms / token | after ms / token | after - before per round, median [min-max] | row after / before (min_ms), median of rounds |
| :-- | :-- | --: | --: | :-- | :-- |
| Metal | greedy (control) | 10.04 | 10.09 | +0.10 [-0.54, +0.51] | 1.004 |
| Metal | top-50 | 10.41 | 10.78 | **+0.50** [-0.07, +0.99] | **1.046** [0.993-1.099] |
| wgpu | greedy (control) | 26.63 | 26.64 | +0.37 [-4.50, +2.83] | 1.013 |
| wgpu | top-50 | 27.42 | 27.73 | +0.74 [-3.77, +2.94] | 1.030 [0.866-1.051] |

**At `k` = 50 the device path is slower, not faster**: about +0.5 ms per
token (+4.6%) on Metal; on wgpu +3%, inside that lane's noise. It trades a
201 KB readback (about 10 µs on Metal, `_gen_readback` below) and a host
partial select for the `topk_rows` kernel, which makes `k` passes over the
50,304 columns (`ojas_topk_rows`, `reduce.wgsl` `topk_rows`), so its cost
grows with `k`.

### `probes/`: no `k` where the device selection is faster

`probes/probe_topk.rs` (a throwaway example, run from a scratch worktree at
`2b68d11`; not built by the workspace) times one resident `[1, 50304]` row,
interleaved, min of 50, for each `k`: (a) `topk_rows(k)`, read back values
and ids, `sync`; (b) read back the row, `sync`, `sample_token` at
temperature 0.7 with that `top_k` on the host. Two runs at 1-minute load
2.8–3.0, `probes/probe_topk.txt`. Run 1 (run 2 agrees within 15%):

| `k` | Metal device µs | Metal host µs | wgpu device µs | wgpu host µs |
| --: | --: | --: | --: | --: |
| 1 | 196 | 121 | 495 | 444 |
| 8 | 312 | 136 | 648 | 448 |
| 50 | 1,012 | 141 | 1,457 | 441 |
| 128 | 2,288 | 138 | 3,020 | 462 |
| 1024 | 17,566 | 149 | 21,275 | 479 |

The device path is slower at every `k` from 1 to 1024 on both backends;
there is no crossover. Each pass costs about 17 µs on Metal and 20 µs on wgpu
(one threadgroup per row walks all 50,304 columns `k` times), while the host
path stays flat (121–149 µs on Metal): on this unified-memory machine the
201 KB readback and a host partial select cost less than the device path
even at `k` = 1. Each
probe call also pays its own command-buffer round trip, which a decode step
shares with the forward, so the absolute gap here (+0.87 ms at `k` = 50 on
Metal) is larger than the A/B's +0.5 ms; the ordering is the same.

The probe times selection and readback only. It leaves out the drawn id's
upload, which the device path skips when one leader remains (`k` = 1:
temperature 0 or `top_k = 1`), so at `k` = 1 the decode-step comparison is
unmeasured.

Against `2026-10-08-decode-before` (greedy, ms per token, median over 3
rounds): Metal 12.42 then, 9.79 in `base/`; wgpu 29.52, 23.93; ojas-cpu
5.09, 4.17; ojas-cpu-host (`CpuGpt`) 50.74, 4.16. These span every commit
from `8967457` to `de5efaf` (device argmax and upload-free steps, the shared
decode step for `CpuGpt`, in-place KV, the CPU hot paths), so no one change
owns them.

### `ab-k1/`: device selection only at `k` = 1

`cefc864` keeps `topk_rows` for one leader only: temperature 0, as Go's
`GenerateIDs` asks, or `top_k = 1`. There the id stays on the device for the
next forward with no upload, as `forward_greedy`'s does. Every other `k`
reads the row back and ranks it on the host. `ab-k1/` times that against the
decoder before `forward_top` existed. **before** is `de5efaf` with
`867c921`'s bench files, so its temperature-0 and top-50 steps both read
back the row and upload the id. **after** is `cefc864`. Both use the same
runner as `ab/` (6 rounds, Metal then wgpu, order alternating), with the new
row `gen_temp0_p32_n32` (`generate` at temperature 0; its torch twin is the
greedy row) and torch references from `867c921`'s `torch_rows.py`. The
greedy row is a control, and so is top-50, which reads the row on both
sides.

Traffic per decode step (`_gen_traffic`, measured; independent of load):

| row | before | after |
| :-- | :-- | :-- |
| `gen_temp0_p32_n32` | 1 upload / 4 B / 201,216 B (1 readback) | 0 uploads / 0 B / 8 B (2 readbacks) |
| `gen_topk50_p32_n32` | 1 / 4 B / 201,216 B | 1 / 4 B / 201,216 B |

**Time: unresolved.** The 1-minute load ran from 3 to 46 (rounds 2–4 at
20–46), and even the controls moved by about ±1 ms per token from round to
round. The table gives after minus before for temperature 0 minus the same
difference for greedy (min_ms, ms per token), in the quiet rounds only:

| lane | round (load) | temp0 - greedy control |
| :-- | :-- | --: |
| Metal | 1 (6) | -0.06 |
| Metal | 6 (4) | -0.21 |
| wgpu | 1 (9-15) | +0.09 |
| wgpu | 5 (5-10) | +0.10 |
| wgpu | 6 (3-4) | +0.39 |

The probe bounds the loss at about +0.08 ms per token on Metal (196 against
121 µs at `k` = 1), and the probe does not count the id upload that the
device path skips. That is under 1% of a 10 ms step, so no speedup is
claimed for `forward_top` at any `k`. `k` = 1 stays on the device because it
reads back 8 B instead of 201 KB, uploads nothing, and costs no measurable
time. `summary.md` has every row's tables.

## GQA decode row

`gen_gqa4_prefill_p32` and `gen_gqa4_greedy_p32_n32` run nanolab's default
GPT with 4 KV heads for its 12 query heads (`Config(n_kv_head=4)` on the
torch side), the GQA decode row `gp-docs-and-bench-parity` assigned here.
Every lane passes the prompt-logit gate (max abs 8.2e-05 to 9.4e-05) and
emits torch's 32 greedy ids. Per token (`base/`, the quieter run): Metal 8.98
ms, wgpu 23.80, ojas-cpu 3.85, ojas-cpu-host 3.66, torch-mps 5.39.

## Readback into a caller's buffer: deferral re-recorded

`_gen_readback` times reading one `[1, 50304]` logit row (201,216 B), min of
50: `Tensor::to_host` (read, then decode the bytes into a typed host tensor)
against `DeviceBuffer::read_bytes` alone. The difference is what a
`read_f32_into` on the trait would save.

| lane | to_host µs | read_bytes µs | decode pass µs (one record per round, 6 rounds over both runs) |
| :-- | --: | --: | :-- |
| ojas-metal | 11.5–14.6 | 4.0–6.9 | 7.2, 7.6, 7.7, 9.1, 9.1, 9.4 |
| ojas-wgpu | 153–252 | 150–230 | −31.8 to 22.4: within the noise of `read_bytes` itself |

Against a decode step of 8.7–10.5 ms on Metal, 9 µs is 0.1%, and since
`7f0a9f3` and `forward_top` no decode step reads the row back except
`forward` itself, a sampled step without `top_k`, and the `k > 1024`
fallback. **Decision: `read_f32_into` stays deferred** (the
`docs/typed-storage-plan.md` decision of 2026-10-01, now with this number).

## CPU decode bytes moved per step, before and after in-place KV

`e12d347` made the CPU KV path in place (`ojas-cpu/src/kv.rs`, now
`kv_cache_write` at :265 and `cached_attention_forward`'s slot runs).
Byte counts below are **derived from the code** of `e12d347^` and of this
tree (inferred, not instrumented), for this bench's model (12 layers, 12 KV
heads x 64, so a position is 3,072 B per K or V tensor), its 64-slot cache
(196,608 B per tensor) and one decode token:

| per layer, per step | before `e12d347` | after |
| :-- | --: | --: |
| `kv_cache_write`, K and V: the old code copied the whole cache out (`to_f32_vec`) and back (`write_f32`) | 4 x 196,608 = 786,432 B | 2 x 3,072 = 6,144 B (the new row, in place) |
| `cached_attention_forward`: the old code repacked the `kv_len` prefix of K and V on one thread | 2 x `kv_len` x 3,072 B; 294,912 B at the mean `kv_len` of 48 | 0 B (read in place) |
| **per step, 12 layers** | **~12.98 MB** | **73,728 B** |

Measured time on the `ojas-cpu` lane (`DeviceDecoder<CpuBackend>`, greedy):
5.09 ms per token at `8967457` (`2026-10-08-decode-before`) and 4.17 ms in
`base/`. That difference also holds the CPU hot-path commits between them,
so it is not the in-place change alone.

## Files

| path | what |
| :-- | :-- |
| `base/`, `after/` | `env.txt`, `env_torch.json`, `load.jsonl`, `summary.md` and every round's JSONL, as `run_paired.sh` wrote them (not `ref/`), except that `summary.md`'s first line names the output directory relative to the scratch directory |
| `ab/` | `ab_decode.sh` (the runner), `load.jsonl`, `summary.md` and every round's JSONL |
| `probes/` | `probe_topk.rs` (the device-against-host selection probe) and `probe_topk.txt` (its two runs) |
| `ab-k1/` | the `k` = 1 A/B: `ab_decode.sh` (`ab/`'s runner, with `BENCH_ROWS` overridable), `load.jsonl`, `summary.md` and every round's JSONL. Both `summary.md` files are `go run scripts/benchreport/main.go decode-ab DIR min_ms` then `... median_ms` (it was a `summarize.py` in each folder; the Go port prints both byte for byte) |

## tessl revision

`base/` and `after/` record tessl `de4747a` (`env.txt`), as did the probe and `ab-k1/`'s two binaries.
tessl `d0b1792` (merged 2026-10-10) breaks `ojas-metal`'s build:
`CeWorkspace::new` gained a sixth argument (`GemmOperands`) that
`ojas-metal/src/gpu.rs:1139` does not pass (E0061). That break is upstream
and not part of this task; rerunning these rows against a newer tessl needs
that call site updated first.
