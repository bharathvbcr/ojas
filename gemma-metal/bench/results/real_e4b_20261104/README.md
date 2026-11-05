# Real E4B weights: the decode was wrong, now it matches mlx_lm (2026-11-04)

Snapshot: `mlx-community/gemma-4-e4b-it-4bit` (`model.safetensors` sha256 `932b8271fc3fe65a…`,
4.8 GB), downloaded to `~/Code/research/models/gemma-4-e4b-it-4bit`. Every earlier E4B speed figure
in this crate (`../dispatch_profile_20261104`, `../coreml_*`) was on random weights at E4B shapes.

## The defect [V]

On real weights, greedy decode of BOS + "The capital of France is" was not language, under both
`GEMMA_METAL_E4B_DUAL_NORM=0` and `=1`, from the first token (`text/before-*.log`). Against
mlx_lm's per-layer outputs (`oracle/dump_hidden.py`, `oracle/compare.py`), layer 0 was already
off by rel L2 0.41 and the error grew past 1.0 (`text/parity-before-dualnorm1-vs-mlx_lm.txt`).

Cause: Gemma4's per-layer input (PLE) was a stand-in. The loader skipped `per_layer_input_gate`,
`per_layer_projection` and `per_layer_projection_norm` (and loaded but never used
`per_layer_model_projection`). The decode added `embed_tokens_per_layer[tok] * 16 / sqrt(2)`
into the first 256 entries of the residual after attention. mlx_lm (`gemma4_text.py`) instead:
- per token: `ple = (rmsnorm_256(per_layer_model_projection(h_emb) * H^-0.5) +
  embed_tokens_per_layer[tok] * sqrt(256)) * 2^-0.5`, one 256-row per layer;
- per layer, after the FFN residual: `x += post_per_layer_input_norm(per_layer_projection(
  gelu_tanh(per_layer_input_gate(x)) * ple[l]))`, then `x *= layer_scalar`.
The default E4B graph also used the legacy pre-LN algebra instead of Gemma4's dual norms.

## The fix [V]

`weights.rs` loads the four tensors. `gpu_model.rs`: `PleBlock` per layer and `ple_model_proj`
on the model, all-or-none (a partial set is refused). `project_per_layer_inputs` runs after the
embed (one GEMV, one 42-row RMSNorm, one whole-row table lookup, one add); `per_layer_input_block`
runs after the FFN residual (gate GEMV, `gelu * ple[l]` read at an offset, projection GEMV with
the post-norm, residual add and `layer_scalar` in one fused dispatch). E4B takes the dual-norm
path whenever the block is loaded. The two `2^-0.5`/`H^-0.5` scales are dropped because an RMSNorm
follows each. Decode (T = 1) only: the M-row verify path still has the stand-in.

After (`text/after-*.log`, `text/parity-v2-vs-mlx_lm.txt`): every layer at every prompt position
within rel L2 0.041 of mlx_lm (layer 0: 0.005), the same next token (9079), and greedy text
" Paris.\nParis is the capital of France.<turn|>..." and a correct `reverse_string` function.
Regression test: `tests/real_e4b_text.rs` (`--ignored`, needs `GEMMA_E4B_DIR`).

## Speed on real weights

`dispatch_profile --model <snapshot> --pipelined-only --pipelined-steps 64`, M5 Pro, load 4.8-6.5:
- first correct build (`profile_ple_v0`): 976 dispatches/token; 70.6-72.2 tok/s (9 runs).
- PLE block fused (`profile_ple_v1`): 768 dispatches/token, stamped non-GEMV 4.98 -> 3.74 ms.
- copy removed (v2), paired A/B against v1 (`ab_v1_v2.log`, 3 interleaved rounds x 3 runs):
  v1 73.8-76.7, v2 73.5-76.1 tok/s, no measurable difference. v2 is kept as the simpler form.
- v0 -> v1 was not run as a paired A/B (v0's binary was overwritten), so no speed-up is claimed
  for the fusion beyond the dispatch count.

- Gate GEMV with the GELU-multiply in its epilogue (v3: a `Q4Epi::GeluMul` tessl kernel,
  one dispatch and barrier fewer per layer), paired A/B against v2 (`ab_v2_v3.log`): v2
  76.96-78.13, v3 75.32-78.09 tok/s, v3 below v2 in every round. Rejected and removed.
- What the block costs: v2 against a throwaway build that skips it (wrong output, timing only),
  3 interleaved rounds x 3 runs at load 3.3-3.7: 12.88-13.12 ms against 12.04-12.33 ms, so
  about 0.7 ms per token. Its weights are ~32 MB (~0.12 ms of bandwidth); the rest is the
  serial latency of its small GEMVs (256 rows; 2560 rows over K = 256), not dispatch count
  (v1 -> v2 and v2 -> v3 each removed 42 dispatches for no gain).

**Against mlx_lm** (`ab_mlx.sh`, `ab_mlx_64.log`, `ab_mlx_128.log`): same snapshot and prompt,
arms interleaved per round, fresh process each, 3 rounds x 3 runs, load 3.9-5.6. mlx_lm is
`stream_generate` with EOS off, its own generation tok/s; gemma-metal is the pipelined decode.
Not like-for-like: mlx_lm hands each token to Python as it goes.

| tokens | gemma-metal v2 tok/s | mlx_lm tok/s | ratio of medians |
|---|---|---|---|
| 64 | 75.92-77.60 (median 76.92) | 81.75-83.92 (82.34) | 0.93 |
| 128 | 73.87-76.20 (75.39) | 81.09-83.09 (81.69) | 0.92 |

The ~0.9 ms gap is about the per-layer input block's 0.7 ms plus ~0.2 ms elsewhere.

**Where the block's 0.7 ms goes** (`ple_attribution.log`: throwaway builds that skip one part,
wrong output, timing only; 3 interleaved rounds x 3 runs, ms/token): full 12.84-13.27, no
cast+gate 12.38-12.76, no projection+norm-add 12.50-12.89, no GELU 12.74-13.19, no block
12.12-12.78. By medians: cast + gate GEMV ~0.45 ms (~10.7 us per layer for 0.37 MB, about
35 GB/s: 256 rows fill only 32 threadgroups), projection + norm-add ~0.33 ms, GELU ~0.05 ms.
The next lever is a split-K gate GEMV (partials reduced inside the GELU-multiply dispatch).
Not built.

**Also tried, after the attribution:** a split-K gate GEMV (v4: 160 threadgroups instead of 32,
partials summed in the GELU-multiply dispatch; parity unchanged, max rel L2 0.040). Paired A/B
against v2 (`ab_v2_v4.log`): round medians v2 76.52/76.29/76.37, v4 76.13/76.30/77.45 tok/s, not
sign-consistent, so no gain. Removed. Fewer dispatches (v2, v3), an epilogue fold (v3) and a wider
grid (v4) all leave the block's time where it was, which points at the latency of each dependent
step (a cold 0.37 MB weight read after a barrier), not at occupancy or launch count.

**Also tried:** warming the block's weights by running its gate and projection GEMVs into a junk
buffer in the same barrier interval as the down projection (throwaway env-gated build, same
binary both arms, `prefetch_ab.log`): 72.1-74.0 tok/s on against 75.4-77.1 off, slower in every
round. Not kept. So the block is not waiting on cold weights that a concurrent read could hide.

## Speculative decoding: measured, not viable today (`spec_economics.log`)

With the assistant (`google/gemma-4-E4B-it-assistant`, 191 MB, downloaded with approval) and
`tests/real_e4b_spec_economics.rs` (`--ignored`), before porting the block to the batched path:
- **verify(M) does not amortize the weights:** 13.3 ms at M = 1, 29.8 at 2 (2.25x), 53.3 at 4,
  67.0 at 5 (5.05x), 302.8 at 8 (22.8x). Checking M tokens costs more than M sequential steps, so
  no accept rate can pay for it. The batched path also still has the per-layer input stand-in.
- **The draft is unusable:** `generate_mtp_smoke` on three prompts, 64 tokens each: 0 of 108
  drafts accepted, one token per draft, 6.9 ms per host-side draft call, and its output differs
  from plain greedy (a defect in that loop or the assistant forward, not investigated).
- What it would take: a correct assistant forward, an M-row verify that reads each weight once,
  and the block ported to that path. Not started.

Found on the way and fixed: `step_verify` never reset the per-step scalar arena, so the
batched GEMM path ran out of slots on its second call ("icb_scalars u32 arena exhausted").
`tests/verify_scalar_arena.rs` (E4B shapes, random weights) fails without the reset and passes
with it.
