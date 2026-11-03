# Rules: flash forward, bf16 entry split (B) and 64-row f32 tile (C), 2026-11-03

**Written before any timed row and before any build of this folder's binaries.** The job log
prints this file's `stat` mtime, the worktree HEAD, `git status --short` and `uptime` as its first
lines.

## Why

SpikeTrack's tracker step at 16 objects spends ~195 ms per full-depth frame in four memory
cross-attention calls, q `[1,8,5184,32]` x k,v `[1,8,~36400,32]`, f32, no bias
(`SpikeTrack/bench/results/ab-coreml-tracker/README.md`, `run2-ops`). The dispatcher already picks
the fastest kernel ojas has for that call, the TensorOps flash forward `ojas_attn_fwd_d32`
(`../2026-11-02-sam-xattn-q2`: 5-8x faster than split-K). It runs at ~3.3-3.7 TFLOP/s. This run
asks whether two changes inside that kernel make it faster **without moving an output bit**.

- **B (`c80bda3`):** the bf16-accumulation forward moves out of `attn_fwd_tiled_body` into its own
  entry, `ojas_attn_fwd_bf16`. The f32 entries keep their names and arithmetic and lose the
  runtime branch to the scalar bf16 loops. The lead: the same kind of runtime `bf16_acc` branch
  made split-K's part kernel 3.2x slower at `d92a2e7` (`../2026-11-02-splitk-regression`). Flash
  moved ~+5 ms at that commit, within that probe's noise.
- **C (`33cddcd`, on top of B):** the f32 forward takes 64 query rows per threadgroup on eight
  simdgroups (was 32 on four), so each K/V block a threadgroup reads serves twice the rows (the
  call reads all of K and V once per query block: 162 blocks x 9.3 MB per head at 36,400 keys).
  Four threads still own a row and the key tile stays 64 wide, so each row's softmax, its sums and
  its rescales are the same operations in the same order. Whether the matrix units give the same
  bits at M = 64 on eight simdgroups as at M = 32 on four is **not known before the run**; the
  hashes decide.
- **A (`3406d65`):** `286a444` (the commit SpikeTrack pins) plus only the probe example.
- **A′:** A's binary, run as a separate process in every round: the noise floor.

Compiled with the crate's own flags (`-O2 -ffp-contract=off -fmetal-math-mode=safe`, `build.rs`).
Before the run, every pipeline of A, B and C was built from its metallib on this device
(`target-kern/pipes.sh`, compile only): all report 1024 max threads; static threadgroup memory
8576 B for A and B, 17152 B for C's f32 entries.

## Method (one tree, one lock window)

- Worktree `/Users/bharath/Code/kern1103/ojas` (branch `claude/flash-fwd-specialize-1103`), tessl
  worktree `/Users/bharath/Code/kern1103/tessl` at `52bdd19` (the pinned tessl, clean). One
  `CARGO_TARGET_DIR`. The job checks out A, B, C in turn (detached, `Cargo.lock` restored before
  each), builds `ojas-metal --example flash_fwd_probe --release` at `-j 2`, copies each binary out
  and records its sha256. All under `SpikeTrack/tools/locked.sh` (the mac_heavy lock).
- **Hashes:** `flash_fwd_probe hash` on A, B and C. Eleven cases, FNV-1a 64 over the f32 bits of O
  then LSE: SpikeTrack's call (`attention_forward`, and `attention_forward_saving`) at Tk 36,400;
  forced flash at d 32/64/128 with ragged query tails (q 100, 65, 40 and 5184 x 15600); a biased
  call; causal d 32 and causal windowed d 64; and the same under bf16 accumulation (cross, biased
  cross, causal).
- **Timing:** `flash_fwd_probe time 3 15`, 8 rounds. Each round runs A, B, C and A′ as four
  processes; round r starts at label r mod 4 and odd rounds walk the order in reverse. Each process
  times three arms with a device sync on both sides of every call, odd in-process rounds reversed:
  `x36400` (SpikeTrack's call), `x15600` (the same at 3 memory frames) and `ctl`
  (`attention_forward` at q,k,v `[1,8,5184,32]`, tessl's `encoder_attn`, untouched by B and C).
- The per-process statistic is the arm's median over its 15 timed calls (min, p10 and p90 are
  reported too).

## Gates, fixed now

1. **Bit identity (decides alone).** A variant whose hash differs from A's on **any** of the
   eleven lines is rejected, whatever its speed: SpikeTrack's parity gates compare outputs byte for
   byte.
2. **Kernel identity.** In every process, each `x36400` and `x15600` call encodes exactly one
   `ojas_attn_fwd_d32` and no `ojas_attn_fwd_bf16`; `ctl` encodes neither. A binary that breaks
   this is not read.
3. **Control.** Per round, `ctl` median of B, C and A′ over A's. If the median over rounds of any
   of these ratios is outside 1.00 ± 0.05, the window is `repeat` and the times are CONTENDED
   (identity results still stand).
4. **Noise floor.** N = the largest |A′/A − 1| over the 8 rounds on `x36400`'s median.
5. **Verdict per variant on `x36400`** (and reported the same way on `x15600`), from the per-round
   ratio V/A:
   - **faster:** every round's ratio < 1 − N;
   - **slower:** every round's ratio > 1 + N;
   - otherwise **not measured** (inside the noise). No gain inside the noise is claimed.
   The size is reported as the median ratio with its min-max over rounds.

## Predictions, fixed now

- B: **not measured** (inside the noise). The bf16 block is a top-level early return, not a branch
  inside the key loop as in split-K, so the compiler need not de-unroll the f32 loops.
- C: **faster**, by 5-20% at `x36400`; bit-identical hashes on all eleven lines. The K/V re-reads
  are likely served largely from cache already (one head's 162 blocks run together), so not 2x.
- Neither moves `ctl`.

## If C (or B) passes gates 1 and 5

A second job (rules appended below before it runs) rebuilds SpikeTrack against A and the winner
in one tree and runs the tracker A/B end to end: predictions byte-identical between the two
binaries, and the `tracker propagate ... memory attention (device)` stage median per steady frame.

## End to end (job 2), fixed before job 1 reported anything

Written while job 1 waited for the lock, so before any hash or time of this folder existed.

- **Tree:** `git archive 38e0a35` of SpikeTrack (`Cargo.toml`, `Cargo.lock`, `crates`,
  `tools/reference`) at `/Users/bharath/Code/kern1103/SpikeTrack`, so `../ojas` is this worktree
  and `../tessl` the pinned tessl. The submission checkout is not read or built. Two SpikeTrack
  binaries from that one tree and one target dir: `st-A` (ojas `3406d65`) and `st-<winner>`.
  The winner is the variant that passed gates 1 and 5 above; C if both did.
- **Runs** (`job2.sh`, the CoreML A/B's settings): VISEM 82, 34 frames, `--boxes detector`, r16
  LoRA, `--max-objects 16`. An unprofiled winner control, three rounds A, W / W, A / A, W with
  `--profile stages`, and an unprofiled A control last.
- **Gate E1, parity (decides alone):** every run's `82.json` byte-identical to the winner control,
  A's runs included. If not, the change moves SpikeTrack's output and is rejected.
- **Gate E2, speed** (`e2e_summary.py`): per round, the steady-frame (>= 8) median of the stage
  `tracker propagate, state 0: memory attention (device)` and of the tracker step
  (`tracker propagate*` + `memory encode*`). Noise = the larger of the two arms' round-to-round
  spread (max/min - 1). **faster** if every round's winner/A ratio < 1 - noise, **slower** if every
  ratio > 1 + noise, otherwise **not measured**.
- **Prediction:** E1 holds. On memory attention, faster in proportion to job 1's `x36400` ratio
  applied to the ~195 of ~285 ms that cross-attention takes; the tracker step moves by the same
  milliseconds, so its ratio is smaller and may fall inside the noise.
