# Flash forward: bf16 entry split (B) and 64-row f32 tile (C), 2026-11-03

Rules: `rules.md`, written before any build or timed row of this folder. Its mtime, 09:30:35
local, is printed at the head of `job1.log`. The job ran at 16:34:49Z, after 66 minutes waiting
for the mac_heavy lock. Job script: `job1.sh`. Analysis: `summarize1.py` → `summary1.txt`.

## Outcome in one line

**Neither change moves SpikeTrack's cross-attention call.** Both are bit-identical to the pinned
kernel on all eleven hash cases, but at q `[1,8,5184,32]` x k,v `[1,8,36400,32]` the ratios are:

| variant | median ratio to A | range over 8 rounds | noise floor N |
|---|---|---|---|
| B | 1.019 | 0.982-1.106 | 0.066 |
| C | 1.004 | 0.911-1.112 | 0.066 |

Both verdicts are **not measured** (inside the noise). As predicted, B is not faster. C was
predicted 5-20% faster, and that prediction failed. So the d = 32 flash forward is not bound
by re-reading K and V once per 32 query rows, and the runtime bf16 branch was not costing it.

## Gates

| gate | result |
|---|---|
| 1. Bit identity | **held**. `hash-B.txt` and `hash-C.txt` equal `hash-A.txt` on every case (`hash-diff-*.txt` empty): SpikeTrack's call via `attention_forward` and `attention_forward_saving` at 36,400 keys; forced flash at d 32 (q 100 x k 333, q 5184 x k 15,600), d 64 (b 2, h 3, q 65 x k 129) and d 128 (q 40 x k 70); a biased call; causal d 32; causal windowed d 64; the bf16-accumulation cross, biased cross and causal calls. |
| 2. Kernel identity | **held**. Every `x36400` / `x15600` process encoded 18 `ojas_attn_fwd_d32` and 0 `ojas_attn_fwd_bf16` (3 warm-ups + 15 timed calls); `ctl` encoded neither. In B and C the bf16 hash cases encoded `ojas_attn_fwd_bf16` (`[6,0] -> [6,1]`), so the new entry did run. |
| 3. Control | **held** (window clean). `ctl` (tessl `encoder_attn`, untouched) median ratio over A: B 1.010, C 1.003, A′ 0.997. |
| 4. Noise floor | N = 0.066 on `x36400` (A′/A from 0.944 to 1.066), 0.077 on `x15600`. |
| 5. Speed | not measured for B and C on both arms; see the table above and `summary1.txt`. |

Times (A, `x36400` medians per round): 47.9-52.7 ms. That is ~3.7 TFLOP/s on the useful
4·8·5184·36400·32 FLOP, matching `../2026-11-02-sam-xattn-q2`. One-minute load was 11-17 during
the rounds, above that folder's 6.0 bar. Gate 3 held, so the ratios are read. The absolute
milliseconds belong to this window only. Binaries: `bins.tsv` (A `a2d037e1…`, B `92d70148…`,
C `a0748016…`). Builds from one worktree and target dir: `build-*.log`.

## What this establishes, and what it does not

- **Verified:** with a cooperative destination, the flash forward's `matmul2d` gives the same bits
  at M = 64 on eight simdgroups as at M = 32 on four, for the Q·Kᵀ (reduction 32, 64, 128) and P·V
  (reduction 64) products, on this M5 Pro and at these shapes. The per-row softmax was unchanged by
  construction.
- **Not established:** anything about tessl's exact-f32 GEMM. That kernel writes a device-tensor
  destination with `mode::multiply` into a zeroed C, and tessl keeps it at a 32x32 tile on one
  simdgroup as "golden-safe" (`kernels/matmul_tensorops.metal:30`). This run says nothing about its
  bits under another tile.
- **Inferred, not measured:** at d = 32 each score carries only 128 FLOP of matrix work, against a
  scale, a finiteness check, a max, a `precise::exp`, a sum and two threadgroup-memory round trips.
  So the kernel is likely limited by per-score work, not K/V traffic. At d = 256, tessl's own
  prefill reaches 6.4 TFLOP/s. Every change to that per-score work (a faster exp, a register
  softmax with another summation order) moves output bits. That is a precision decision, not a
  bit-identical kernel tweak.

## Disposition (for the owner)

- **C (`33cddcd`): not for merge.** Bit-identical, no measured gain, and it doubles the f32
  entries' static threadgroup memory (8576 → 17152 B). It is reverted on the branch.
- **B (`c80bda3`): neutral.** Bit-identical with no speed claim. It moves the scalar bf16 loops
  out of the f32 body into one entry for every head dim. Merge it only as a cleanup, if you want
  that.
- **E2E (job 2) did not run.** By the rules it runs only for a variant that passed gates 1 and 5.
