# nanolab training-step config A/B (2026-11-02)

A research sweep on 2026-11-02 (`research/docs/2026-11-02-research-sweep.md`
§1, §5) ranked three ojas training-step levers without measuring them. This
directory measures them. Each one is reached through `TrainConfig` alone, with
no change to the trainer or a backend:

| arm | lever | `TrainConfig` field | baseline | variant |
| :-- | :-- | :-- | :-- | :-- |
| (a) | Muon Newton-Schulz precision | `muon_ns5` | `Ns5Precision::F32` (default) | `Ns5Precision::Bf16` (stock nanolab/torch) |
| (b) | Fused LM-head CE vocabulary tile | `chunk` | `CeChunk { rows: 1024, cols: 8192 }` (`DEFAULT_CE_CHUNK`) | `cols: 50304` (one vocabulary tile) |
| (c) | Activation checkpointing (B=8 only) | `activations` | `ActivationCheckpoint::Off` | `ActivationCheckpoint::Blocks` |

Model: nanolab 124M (`ModelSpec::nanolab_124m`: d 768, 12 layers, vocab
50304, SwiGLU, tied embeddings), T=1024, B in {1, 8}, K=1, Metal backend,
autocast off in every arm. The token stream is the synthetic one
`bf16_memory_rows` uses.

## Pre-stated refutations (written before any run)

- **(a)** If the Bf16 NS5 saving at B=1 is under **0.04 s** per step, the
  sweep's reconstruction that "the optimizer is about 0.35 s of a 0.61 s
  step" is wrong.
- **(c)** If `Blocks` saves less than **2x** peak charged bytes at B=8, or
  costs more than **40%** wall time per step, it is not the memory lever.
- **(b)** No threshold was stated by the sweep. The code already has the
  3-GEMM path: `ojas-metal/src/device.rs` `linear_ce` skips pass 2's logits
  GEMM when one tile covers the vocabulary (`one_col_tile`), so the sweep's
  "4 vocab GEMMs where 3 would do" describes the *default* tile, and the lever
  exists behind `CeChunk`. This A/B measures whether it matters. Expected
  cost: about `1024 x 50304 x 4` bytes (206 MB) of logits tile plus a
  `[50304, 768]` weight-gradient tile (155 MB) more peak per call.

Evidence that each arm actually ran a different path:

- (a) loss trajectories must differ between F32 and Bf16 from step 2 on
  (identical losses would mean the flag never reached the device);
- (b) per-step peak charged bytes must rise by roughly the tile sizes above;
- (c) peak must drop, and the losses must be bit-identical to `Off`
  (`ActivationCheckpoint` promises the same values).

## Protocol

- Harness: `ojas-model/examples/nanolab_config_ab.rs`. One process per row,
  10 steps; the first 2 are warm-up and are dropped in the summary (the rows
  keep them). Every step is synced before its clock stops.
- Driver: `run.sh` here, run once under the machine lock. 5 reps; each rep
  runs every arm as its own process in an order rotated by the rep index,
  and the B=1 and B=8 groups alternate which goes first.
- An A/A control at B=1 (`b1-aa`, the baseline config under a second tag)
  gives the noise floor every B=1 delta is read against.
- Reported per arm: min-of-N and median of the per-process step time
  (mean of steps 3-10), the spread across processes, the per-rep paired
  delta against the baseline of the same rep and its sign consistency, the
  peak charged bytes over steps 3-10, the process's peak memory footprint
  (`/usr/bin/time -l`), and the 1/5/15-minute load before and after.
- Phase profile: the trainer has no per-phase timer, so `phase` mode times
  the two ops in isolation on the same device: one Muon sweep over every
  nanolab Muon-group matrix (`f32` vs `bf16` NS5), and one fused CE with both
  gradients over `B x 1024` rows (default tile vs one tile). Arms alternate
  within the process, each call synced, 8 reps per process, one process per
  batch size per rep.

## Baselines

No torch-MPS whole-step baseline exists for nanolab, and none was built: it
needs the nanolab model plus torch's Muon on MPS at matched precision, which
is not cheap. The strongest baselines available are op-level:
`bench/results/2026-10-07-muon-bf16/summary.md` (ojas-metal vs torch-mps on
the three Muon shapes). Every comparison here is ojas against ojas.

## Results (run 2026-11-02 16:30-16:47Z, under the machine lock)

Files: `rows.jsonl` (one JSON row per process, all 10 steps), `runs.tsv`
(per process: exit status, wall time, `/usr/bin/time -l` peak footprint,
load before and after), `env.txt`, `stderr/`. 50 processes, 0 failed. The
tables below were computed from `rows.jsonl` and `runs.tsv` by a throwaway
stdlib script that is not committed. Every number is from this machine
(M5 Pro, 64 GB), with the synthetic token stream, at T=1024.

### Verdicts against the pre-stated refutations

| arm | refutation | measured | verdict |
| :-- | :-- | :-- | :-- |
| (a) Bf16 NS5, B=1 | saving < 0.04 s refutes "optimizer is about 0.35 s of 0.61 s" | paired saving median **0.121 s** (range 0.088-0.140, 5/5 faster); A/A control median 0.002 s (range -0.024 to +0.008) | **Not refuted.** The isolated F32 Muon sweep is 0.334 s (median) of a 0.566 s B=1 step, and bf16 saves 0.123 s isolated against 0.121 s in the step |
| (c) Blocks, B=8 | < 2x peak saving or > 40% wall time | peak charged **10.40 to 2.79 GiB (3.73x)**; step time **+17.3%** (median paired ratio 1.173, min-of-N ratio 1.180, 0/5 faster; per-rep ratios 1.135, 1.048, 1.180, 1.173, 1.374) | **It is the memory lever.** The worst rep (+37%) is under the 40% line but close; the verdict rests on the median and min-of-N. Losses are bit-identical to `Off` in all 5 reps |
| (b) one vocab tile | none stated | B=1: -0.011 s (-1.9%, 5/5 faster; about the A/A spread). B=8: median -0.238 s (ratio 0.905, 5/5 faster); the isolated op saves 0.134 s | Real but small: about 0.13 s per 8192-row CE call; no change in step peak |

### Step time and memory per arm (mean of steps 3-10 per process; n = 5 processes)

| arm | min s/step | median | max | spread max/min | in-process step spread (median) | peak charged GiB | peak footprint GiB (median) | 1-min load range |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| b1-base | 0.5485 | 0.5659 | 0.5945 | 8.4% | 1.3% | 2.756 | 3.82 | 5.2..14.1 |
| b1-aa (A/A) | 0.5422 | 0.5578 | 0.5823 | 7.4% | 4.0% | 2.756 | 3.82 | 5.2..13.6 |
| b1-ns5bf16 | 0.4255 | 0.4429 | 0.4732 | 11.2% | 2.7% | 2.756 | 3.91 | 5.5..15.3 |
| b1-ce1tile | 0.5392 | 0.5470 | 0.5741 | 6.5% | 2.5% | 2.756 | 3.96 | 5.8..15.3 |
| b8-base | 2.1701 | 2.3196 | 2.5623 | 18.1% | 21.9% | 10.398 | 12.10 | 5.9..13.4 |
| b8-ns5bf16 | 2.0296 | 2.1041 | 2.4190 | 19.2% | 18.8% | 10.398 | 12.25 | 5.5..14.6 |
| b8-ce1tile | 1.9949 | 2.0500 | 2.2888 | 14.7% | 4.5% | 10.398 | 12.18 | 5.5..12.9 |
| b8-ckpt | 2.5604 | 2.6334 | 3.4769 | 35.8% | 3.1% | 2.788 | 4.30 | 5.4..12.9 |

"Peak charged" is the budget's peak over steps 3-10, which is the same in
every process of an arm. It includes the 1.07 GiB (1,144,738,632 bytes) of
parameters and optimizer state resident before the first step.

### Paired deltas against the same rep's baseline (variant - base, s/step)

| variant | n pairs | median delta | min | max | faster/slower | median ratio | min-of-N ratio |
| :-- | --: | --: | --: | --: | :-- | --: | --: |
| b1-aa | 5 | -0.0023 | -0.0237 | +0.0079 | 4/1 | 0.996 | 0.988 |
| b1-ns5bf16 | 5 | -0.1213 | -0.1404 | -0.0884 | 5/0 | 0.784 | 0.776 |
| b1-ce1tile | 5 | -0.0107 | -0.0203 | -0.0021 | 5/0 | 0.981 | 0.983 |
| b8-ns5bf16 | 5 | -0.2031 | -0.4258 | +0.0994 | 4/1 | 0.910 | 0.935 |
| b8-ce1tile | 5 | -0.2378 | -0.5123 | -0.1180 | 5/0 | 0.905 | 0.919 |
| b8-ckpt | 5 | +0.3854 | +0.1229 | +0.9470 | 0/5 | 1.173 | 1.180 |

The B=8 baseline is noisy: 21.9% spread between steps inside one process,
against 1-4% for B=1. The B=8 NS5 and CE medians (-0.20 s, -0.24 s) exceed
the isolated op savings (-0.127 s, -0.134 s) by more than the op could save,
so read the B=8 step deltas as "faster, by about the op saving", not as the
medians. B=1 is clean: the A/A control sits within ±0.024 s.

### Phase profile (ops in isolation on the same device)

Per process, the median of reps 2-8 (rep 1 is warm-up), then across the 5
processes. Every call is synced. The Muon sweep covers all 96 Muon-group
matrices in nanolab: per layer, q/k/v/o at 768x768, gate at 12x768, and
SwiGLU at 2048x768, 2048x768 and 768x2048.

| batch | op | arm | min s | median s | max s | paired delta (median) |
| --: | :-- | :-- | --: | --: | --: | --: |
| 1 | Muon sweep | f32 | 0.3172 | 0.3345 | 0.3625 | |
| 1 | Muon sweep | bf16 | 0.1999 | 0.2096 | 0.2228 | -0.1225 |
| 8 | Muon sweep | f32 | 0.3290 | 0.3323 | 0.3749 | |
| 8 | Muon sweep | bf16 | 0.2038 | 0.2052 | 0.2300 | -0.1271 |
| 1 | fused CE fwd+bwd, 1024 rows | 8192 cols | 0.0557 | 0.0606 | 0.0644 | |
| 1 | fused CE fwd+bwd, 1024 rows | one tile | 0.0447 | 0.0479 | 0.0501 | -0.0129 |
| 8 | fused CE fwd+bwd, 8192 rows | 8192 cols | 0.4525 | 0.4639 | 0.5306 | |
| 8 | fused CE fwd+bwd, 8192 rows | one tile | 0.3165 | 0.3313 | 0.3726 | -0.1342 |

The CE op's own peak charged bytes (operands included) rise from 1.30 to
1.58 GiB at B=1 and from 1.34 to 1.62 GiB at B=8. That is +0.28 GiB, the
predicted logits tile (206 MB) plus the weight-gradient tile (155 MB), less
the default tile's share.

### Evidence that each arm ran its own path

- (a) Losses differ from the baseline from step 2 on, in every rep
  (|dloss| at most 0.0058 at B=1 and 0.018 at B=8 over 10 steps). Step 1's
  loss is equal because it comes before the first update.
- (b) Losses differ by about 1e-6 (B=1 from step 1, B=8 from step 2): the
  tile changes the CE reduction order, so the chunk reached the device. The
  step's peak charged bytes are *bit-identical* to the baseline's
  (2,959,592,888 at B=1; 11,164,888,968 at B=8), so the CE's extra
  0.28 GiB never sits at the step's peak. Inferred: the peak falls later
  in the backward. The process's peak footprint rises by 0.08-0.14 GiB.
- (c) Peak drops 3.73x, and losses are bit-identical to `Off` in all 5 reps.
- A/A: identical losses in every rep, as expected for one config.

### What this changes about the sweep's ranking

- The "Muon NS5 about 0.35 s of 0.61 s" reconstruction holds when measured:
  the F32 sweep alone is 0.33 s (measured, in isolation), about 59% of the
  0.57 s B=1 step. That in-step share is inferred: the bf16 saving in the step
  (0.121 s) matches the isolated one (0.123 s), so the sweep runs serially in
  the step rather than overlapped. Switching to
  bf16 NS5 (stock nanolab) saves 0.12 s per step (21%) at any batch, since
  the optimizer's cost does not depend on B. It is a numerics change (losses
  move by about 1e-2 over 10 steps), so adopting it needs the parity
  decision `Ns5Precision` documents, not just this timing.
- The one-tile CE saves about 0.13 s per 8192-row call (29% of that op) and
  about 0.01 s at B=1. It costs about 0.28 GiB of scratch, which does not
  raise the step's peak here. It is reachable today through `TrainConfig::chunk`;
  `DEFAULT_CE_CHUNK` stays at 8192 columns for vocabularies like Qwen's
  248k, where one tile would be 1 GB.
- Block checkpointing is the memory lever at nanolab scale: 3.7x lower peak
  for 17% more step time (n=5). The earlier Qwen3.5-0.8B figure (18.08 to
  3.96 GB, +26%) was n=1 and is not re-measured here.

### Caveats

- No torch-MPS whole-step baseline exists (see above). These are
  ojas-against-ojas ratios, not a claim about ojas against torch.
- Load: other sessions shared the machine. The 1-minute load ranged 5.2-15.3
  across the run, logged per process in `runs.tsv`. The lock keeps other
  *locked* jobs off the GPU, not unlocked ones.
- A smoke run just before this one (16:27Z, not kept, 15-minute load about 31)
  saw B=1 steps of 1.8-2.3 s and a 0.80 s F32 Muon sweep, 3x this run's
  values. The cause was not identified. Repeat runs on a quieter machine
  before quoting absolute times.
- 10 steps per process from a fixed init: the loss deltas show which path
  ran, not convergence quality.
- Build: `target/release/examples/nanolab_config_ab`, built 16:22Z from the
  shared working tree. `env.txt` was written at run time (16:30Z), when HEAD
  was 777542b plus 61 dirty files. Peers committed during the session, so the
  build's HEAD is not recorded separately. Between the
  build and the run, the example's source was rustfmt-ed by another process
  (formatting only; `rustfmt --check` passes). `env.txt` records the
  formatted file's sha1. Peer change in the build: the
  `ojas-core/src/autocast.rs` attention-backward override (stable before the
  build started). It does not touch these arms, because autocast is off
  in every one.
- Lint: clippy on this example reported nothing. Clippy `-D warnings` on the
  workspace currently fails on peers' files (`ojas-cpu/src/quant_sparse.rs`
  `chunks_exact_to_as_chunks`, `ojas-metal/src/device.rs`
  `too_many_arguments`), not on this example.
