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

## Two more dispatch-chain cuts (2026-11-05): +3.9%, now 0.95-0.96x mlx_lm

From the real-weight stamps (`profile_ple_v1`), outside the GEMVs (which move ~86% of peak
bandwidth): every sliding layer copied its KV ring into dense scratch (`kv_ring_densify`, K and
V behind separate barriers, ~16.6 us stamped per layer), and each layer ran three small serial
norm/cast dispatches after its residual tails.
- **KV ring read in place** (`fa_buffers`): while the live window starts at physical row 0
  (every context up to the 512 window) the ring already is the dense prefix the copy produced,
  in the same order, so FA reads it directly. Decode ICB replay (opt-in) keeps the copy so its
  tape has one shape across a wrap. Test: `sliding_fa_buffers_read_in_place_until_the_window_
  leaves_row_zero` (fails with the copy forced on). Paired A/B, same tree, only that line
  differing (`ab_densify2.log`, 5 rounds x 3 runs): round medians 76.43-76.80 -> 77.89-78.27
  tok/s, B ahead in 5 of 5 rounds, +1.6% (~0.21 ms/token). (`ab_densify.log`, the first 3-round
  try, had one B round at ~50 tok/s from outside interference, so it was rerun.)
- **Residual tail emits the next bf16 input** (`kernels/residual_norm.metal`,
  `rms_norm_residual_add_emit_bf16`): the o_proj tail also writes the pre-FFN normed input, the
  FFN tail writes bf16(x) for the per-layer gate, and the per-layer block's tail writes the next
  layer's normed input: 3 dispatches and barriers fewer per layer. Same lanes and order as the
  separate kernels, bit-identical (`tests/residual_emit_bf16.rs`, 6 shapes/scales/modes).
  `GEMMA_METAL_EMIT_BF16_TAIL=0` restores the separate dispatches. Same binary both arms
  (`ab_emit_tail.log`, 5 x 3): round medians 78.44-79.70 -> 80.15-80.77 tok/s, 5 of 5, +1.3%.
- **Both paths ran** (`profile_v5_tail_on/`, `profile_v5_tail_off/`, stamped `dispatch_profile`):
  726 dispatches/token after v2 -> 686 with the ring read in place (no `kv_ring_densify` while
  unwrapped) -> 561 with the fused tail (125 `rms_norm_residual_add_emit_bf16`, 2
  `rms_norm_bf16`, 1 `cast_f32_to_bf16` left per token; the last layer emits nothing).
- **Together against aff9179** (`ab_total_v5.log`, 5 x 3): 77.33-77.97 -> 80.15-81.01 tok/s
  (medians 77.53 -> 80.53), 5 of 5 rounds, +3.9%. `real_e4b_text` passes; FA inputs and the
  fused tail are bit-identical, so the mlx_lm parity above is unchanged.

- **Greedy argmax across the GPU** (`kernels/argmax_two_pass.metal`): the one-pass softcap
  argmax scanned all 262,144 logits on one threadgroup (~72 us stamped). Now 256 threadgroups
  each reduce a span and one folds the 256 partials, with the same tie and non-finite rules
  (`tests/argmax_two_pass.rs`: ties within and across spans, NaN/inf, softcap saturation, odd
  lengths, all-NaN; identical tokens to the one-pass kernel; `--generate 32` hashes equal).
  `GEMMA_METAL_ARGMAX_TWO_PASS=0` restores the one-pass kernel. Same binary both arms
  (`ab_argmax2.log`, `ab_argmax2b.log`, 10 rounds x 3): B ahead in 9 of 10 rounds, round
  medians 80.40-80.95 -> 80.93-81.30 tok/s, about +0.5%.
- **Open: the output head looks slower with the fused tail on.** Stamped, whichever dispatch
  follows the 262,144-row head GEMV (one-pass argmax, multipass argmax, or the two-pass
  partials) takes 300-450 us with the tail on and ~72 us with it off, in 15 of 16 tokens over
  two interleaved profile pairs; tokens are identical. The extra time belongs to the head
  GEMV's tail, not the next kernel. Unexplained; the unstamped A/Bs above are the measurement
  of record.

- **Decode attention in one pass up to 128 keys** (`kernels/flash_attn_decode_single.metal`,
  generated from tessl's partial-pass macro): when every live key fits tessl's 128-key chunk,
  the partial pass writes O itself; over one chunk the reduce pass computes acc * (1/l), so
  this is bit-identical (`tests/attn_decode_single.rs`: D=256/512, 1-128 keys, window
  masking, fully masked, kv offsets, f32/bf16; declines at 129 keys). 561 -> 520
  dispatches/token. `GEMMA_METAL_ATTN_SINGLE_PASS=0` keeps partial + reduce. Same binary both
  arms (`ab_attn_single.log`, 5 x 3): round medians 80.96-81.18 -> 81.45-81.93 tok/s, 5 of 5,
  about +0.8%. Its stamped per-dispatch time looks worse (15.6 us vs 6.6 + 2.6) while the
  stamped token span drops 14.65 -> 14.26 ms: stamps misattribute time at this dispatch count.
- **The head-GEMV "slowdown" is a stamping artifact.** Knocking out any one of the three fused
  tail sites (throwaway env build, source restored) removes the 300+ us that the dispatch after
  the head shows, and the stamped span stays the same; unstamped, all three sites beat two
  (`ab_knock_end.log`, 5 of 5 rounds, 80.28-80.77 -> 80.68-81.22 tok/s).
- **All four against aff9179** (`ab_total_v7.log`, 5 x 3): round medians 77.16-78.44 ->
  81.12-82.14 (77.78 -> 81.71), 5 of 5, +5.1%. Against mlx_lm (`ab_mlx_64_v7.log`, 5 rounds,
  64 tokens): 81.72 vs 84.03 tok/s median, 0.97x.
- **Tried and removed: GELU folded into the per-layer projection's input load.** A row-major
  Q4 GEMV that built `bf16(gelu(gate) * ple)` as it loaded x (one dispatch and barrier fewer
  per layer, 520 -> 478/token; outputs within 4.8e-7 of the pair it replaced, greedy tokens
  identical on two prompts). Same binary both arms (`ab_gelu_fold.log`, 5 x 3): 81.51-81.96 ->
  79.63-80.44 tok/s round medians, slower in 5 of 5 (~-2.3%): every simdgroup recomputes the
  GELUs (~327k tanh instead of 256) inside a latency-bound GEMV. Same lesson as v3's epilogue
  fold; the separate GELU dispatch stays.
- **Re-measured on the real weights: the megakernel, whole-step replay, fp16 and Core ML.**
  - The bounded-TG persistent gate->down kernel (D18, `GEMMA_METAL_FUSE_GATE_DOWN=1`) does not
    engage on the real graph by design (`!(use_gemma4_dual_norm && has_post_ff)`): it writes
    `x += down(..)` without Gemma4's post-FFN RMSNorm. (A first check of this reported "same
    dispatch count with the flag set", but its shell passed the variables as one word, so the
    flag was never set; corrected here.) With the flag set, the guard keeps it off and the text
    is unchanged. **Then forced on and timed** (`megakernel_experiment/patch_mk.py`, a scratch
    copy of this tree: the kernel adds into a zeroed `down` scratch, then the normal post-FFN
    norm/residual tail; `MK_NOSYNC` also drops its per-layer host sync and resets the barrier
    counters on the GPU): greedy text becomes garbage in both variants (D18's stale `mid`: the
    in-kernel DOWN reads the PRODUCE results through relaxed-only device memory), and it runs at
    8.20-8.22 tok/s with the per-layer sync and 9.24-9.26 without, against 79.5-81.0 for the
    default (`ab_megakernel_sync.log`, `ab_megakernel_nosync.log`, 3 x 3 each): ~0.11x. Its
    8-threadgroup grid cannot stream a 26 MB GEMV near bandwidth either.
  - Whole-step command replay (`GEMMA_METAL_ENCODE_ONCE=1` + `TESSL_DECODE_ICB=1`; the log shows
    the DecodeIcb layer graph attached, 547 commands, skip-nop replay) keeps the tokens but runs
    at 30.8-31.5 tok/s against 81.5-82.4 (`ab_decode_icb.log`, 5 x 3, 5 of 5): 0.38x.
  - fp16 weights and Core ML: `../coreml_gemv_probe_20261104` re-run on real gate/down
    matrices: fp16 3.7-3.9x slower than tessl's Q4; Core ML int4 ~25% slower and off by rel L2
    0.61 (symmetric int4 cannot hold MLX's affine bias). gemma-metal's decode has no bf16 weight
    bank (`upload_quant_hot` takes Q4/Q8), so a whole-model fp16 decode cannot be run; at 4x the
    weight bytes it would be bandwidth-bound near 3.5x slower [inferred]. **Then run in full**
    (`f16_weights_experiment/`: a scratch copy of this tree with `gemv_f16w.metal`, a dense
    fp16-weight GEMV, and `patch_f16.py`, which under `F16W=1` dequantizes every decoder
    projection from MLX Q4 to fp16 at upload; the two vocab-sized tables stay Q4 because the
    output head is tied to the embedding lookup; the product tree is unchanged): same greedy
    text as Q4 on two prompts (logits within ~0.06), and 29.1-29.7 tok/s against 80.4-81.3
    for Q4 on the same binary (`ab_f16_weights.log`, 3 x 3, 3 of 3): 0.36x. ~8.1 GB of
    weights per token in 33.9 ms is ~239 GB/s, so the fp16 GEMV is near bandwidth too; it
    just reads 3.6x the bytes.
- **Checked, not pursued: precise RoPE trig.** The fused QK-norm/RoPE/KV-store dispatch costs
  ~8 us stamped per layer for little work. A throwaway build with `fast::` cos/sin (tessl
  restored right after; stamped profile only) moved it 0.204 -> 0.187 ms/token (store) and
  0.160 -> 0.147 (consumer layers), ~8%: the kernel is fixed latency, not trig.

**Against mlx_lm again** (`ab_mlx_64_v5.log`, `ab_mlx_128_v5.log`; same method as the table
above, quieter machine, load 2.0-3.8, which also lifted mlx_lm):

| tokens | gemma-metal tok/s | mlx_lm tok/s | ratio of medians |
|---|---|---|---|
| 64 (5 rounds) | 80.13-80.97 (median 80.73) | 82.75-84.73 (84.07) | 0.96 |
| 128 (3 rounds) | 78.70-79.13 (79.00) | 82.74-83.55 (83.06) | 0.95 |

The remaining ~0.5 ms is about the per-layer input block's latency (0.7 ms, above) less what
mlx_lm spends elsewhere. Next candidates, not built: skipping the FA reduce pass when the
context fits one 128-key chunk (a tessl kernel change; ~2.6 us stamped per layer).

## Speculative decoding, first measurement (`spec_economics.log`; superseded below)

With the assistant (`google/gemma-4-E4B-it-assistant`, 191 MB, downloaded with approval) and
`tests/real_e4b_spec_economics.rs` (`--ignored`), before porting the block to the batched path:
- **verify(M) does not amortize the weights:** 13.3 ms at M = 1, 29.8 at 2 (2.25x), 53.3 at 4,
  67.0 at 5 (5.05x), 302.8 at 8 (22.8x). Checking M tokens costs more than M sequential steps, so
  no accept rate can pay for it. The batched path also still has the per-layer input stand-in.
- ~~**The draft is unusable**~~ (first measurement: 0 of 108 drafts accepted, output differing
  from greedy). **Cause, found later the same day:** the drafter was not the HF recipe. Its input
  was `[hidden, mean of the last K row]` instead of `[target_embed(last token), target final-normed
  hidden]`, its draft loop blended the state 0.7/0.3 with an embedding row instead of feeding
  `post_projection` back, its layers skipped the post-attention / post-FFN norms before the
  residual adds and used theta 10000 on every layer, and its vocab clusters were `rank % 2048`
  instead of `token_ordering.view(2048, 128)`. `generate_mtp_smoke` also pushed every rejected
  round's target token twice, which is why its output differed from greedy.
- **Fixed** (`MtpSession::assistant_step` / `draft`, read from transformers'
  `Gemma4AssistantForCausalLM` and `Gemma4AssistantCandidateGenerator`). Checked against that
  reference: `tests/real_e4b_mtp_oracle.rs` dumps six real-state steps and
  `oracle/mtp_assistant_oracle.py` reruns them in float32: 6 of 6 drafted tokens equal,
  projected hidden within 3.6e-6 relative L2 (`mtp_oracle.log`).
- **Accept rate on real E4B** (three prompts x 64 tokens, T = 1 verification, output asserted
  equal to greedy; `spec_economics_v2.log`): first draft position accepted 70-78%; tokens per
  round 1.78 / 2.21 / 2.56 / 2.74 at draft length 1 / 2 / 3 / 4. The host f32 drafter costs
  5.07 ms per drafted token, and verify(4) still costs 52.9 ms against 12.4 ms for one step, so
  neither is usable yet: at draft length 3 a round must cost under 2.56 x 12.2 = 31 ms for
  speculation to win.

## Speculative decoding works: 1.18x over the pipelined decode (2026-11-05, `spec_e2e_v4.log`)

Four fixes, in order, each with its check:

1. **The drafter** (above): rebuilt from transformers' recipe, oracle-matched (6 of 6 tokens,
   1.5e-6 rel L2 after the speedup below, `mtp_oracle.log`). What the oracle proves is the
   assistant *forward* given its inputs; the input choice (which target hidden, which KV range,
   which position) is read from `Gemma4AssistantCandidateGenerator` and is supported only by
   the accept rate. One known difference: after a rejection HF's loop hands the drafter KV
   through the bonus position, ours stops at the last verified token.
2. **The verify path was wrong on real E4B**: it still ran the per-layer-input lookup stand-in
   and single norms. Ported the real block for M rows (`project_per_layer_inputs_m`,
   `per_layer_input_block_m`) and Gemma4's dual norms. `tests/real_e4b_verify_exact.rs`: 36 of
   36 verify blocks (M = 2..5, three prompts, three depths) predict the greedy continuation;
   0 of 36 with the port switched off.
3. **The verify GEMM did not amortize the weights**: tessl's `q4_simd_gemm` takes M at run time,
   so its x tile and accumulators (sized for 8) leave registers and every row re-unpacks the
   nibbles. `kernels/gemm_q4_mlx_mrow.metal`: M fixed per entry (2..5; 6..8 split in two, because a single
   entry is no faster than tessl's there, which we read as register spill: inferred from timing),
   nibbles unpacked once per weight row for all M. Same harness, gate 10240x2560 (`gemm_mrow.log`;
   the ratios there are to a single GEMV *call* whose host overhead dominates at these sizes, so
   compare the two GEMM columns with each other): M=2 0.33x vs 0.71x, M=4 0.50x vs 1.39x,
   M=8 0.93x vs 9.3x. Numerics: neither GEMM is bit-identical to the GEMV (about 2% of outputs
   differ in the last bits, the compiler contracts the block sums differently); worst
   |mrow - gemv| / max|gemv| is 9.6e-8.
4. **Verify attention**: tessl's multi-query flash kernel cost ~124 us per call for a few dozen
   keys. `flash_attn_verify_single_*` is the single-pass decode kernel with grid row = query row
   at position `pos0 + row` over one K/V buffer: bit-identical to decode row by row
   (`tests/attn_decode_single.rs`).

verify(M) on real E4B (`spec_economics_v2.log` before, final numbers from the last run):

| M | first | + PLE port + GEMM | + attention |
|---|---|---|---|
| 1 | 13.3 ms | 12.6 | 12.5 |
| 2 | 29.8 | 20.1 | 15.0 |
| 4 | 53.3 | 26.6 | 20.7 |
| 8 | 302.8 | 47.5 | 39.4 |

**End to end** (`tests/real_e4b_speculative.rs`, `GpuDecodeSession::generate_speculative`: draft
k, verify `[next, drafts]` in one block, keep the matching prefix plus the target's token, trim
the rest). Against the pipelined decode of record (`bench_decode_tok_s`: GPU-chained argmax, no
per-token readback), decode only on both sides (after the first generated token), eight prompts
x 64 new tokens, 5 repetitions interleaved in one process; median [min, max]
(`spec_e2e_v4.log`):

| draft k | speculative | pipelined decode | paired ratio | tokens/round by prompt |
|---|---|---|---|---|
| 1 | 95.2 [95.0, 95.7] tok/s | 80.9 [80.6, 81.1] | **1.177 [1.172, 1.184]** | 1.50, 1.91, 1.70, 1.50, 1.97, 1.70, 1.54, 1.47 |
| 2 | 91.5 [91.4, 91.8] | 80.1 [79.6, 81.0] | 1.142 [1.133, 1.148] | 1.75, 2.42, 1.97, 1.70, 2.86, 2.33, 1.97, 1.70 |
| 3 | 86.4 [85.5, 86.7] | 80.4 [80.1, 80.7] | 1.070 [1.063, 1.083] | 1.91, 3.15, 2.25, 1.75, 3.32, 2.62, 2.17, 1.75 |
| 4 | 75.7 [75.5, 75.8] | 81.0 [80.5, 81.1] | 0.935 [0.931, 0.942] | 1.91, 3.32, 2.33, 1.85, 3.71, 3.00, 2.25, 1.75 |

The first version of this table (`spec_e2e_v3.log`, 1.283x at k = 1) used the first three
prompts only. They happen to accept well, and one exact tie on prompt 0 moved its accept rate
from 1.80 to 1.50 tokens per round between builds, swinging the ratio by several points; the
five added prompts (explanation, list, translation, recipe, history) accept 1.47 to 1.97 at
k = 1. Every repetition is faster at k = 1 and 2. mlx_lm on the same weights decodes at 84.0 tok/s
(`ab_mlx_64_v7.log`, not in this paired run). Tokens per round count tokens after the first.
Longer drafts lose because the host drafter costs 1.46 ms per drafted token (it was 5.07: a
single-accumulator f32 loop; now eight partial sums, host-dequantized embedding rows instead of
a GPU round trip, and a KV sync that copies live rows instead of the whole capacity). Prompt 0
runs on its continuation after the tie below, which drafts less well; most prompts gain with k
in tokens per round, but not enough to pay for the extra drafted tokens past k = 2. An earlier run against `generate` (which reads every token back, 70.7 tok/s,
prefill included) gave 1.25x at k = 1 (`spec_e2e_v2.log`).

**Tried and reverted (2026-11-05).** (1) A four-rows-at-a-time host GEMV and a selection-based
clustered head for the drafter: bit-identical drafts, but no speedup in a paired A/B (1.25 vs
1.26 ms per drafted token), so the drafter is not bound by one row's add chain. (2) Pointer-walked
addressing in `gemm_q4_mlx_mrow` plus a fused gate∥up→gelu entry: verify(M = 2) 14.8 -> 13.7 ms
and verify(4) 20.1 -> 17.9 ms over four paired rounds, but the compiler contracts the sums
differently, and on the eight prompts the speculative output left greedy at a 0.112 logit gap
(prompt 6, k = 3, token 21; `spec_e2e_v5.log`), above the 0.05 tie limit the old kernel passes.

**Scope.** These are 64-token runs: every context fits one 128-key chunk, where decode and
verify attention both take the single-pass kernels. Past one chunk both fall back to tessl's
kernels, and the verify one was the slow part (~124 us per call); not measured there.

**Exactness.** The output equals greedy except at exact ties: on the eight prompts, one per
repetition at k = 1 and 2 and two at k = 3 and 4, all below the 0.05 limit. The one examined,
on prompt 0: the batched verify picks 5508 where T = 1 decode picks 22169; their logits after that
prefix differ by 0.013 here and are both 27.0 in mlx_lm, which also picks 5508. The test allows
a divergence only below a 0.05 logit gap and counts it (`tests/real_e4b_spec_divergence.rs`
replays the rounds and shows the same block verified from a fresh state gives the same answer,
so it is the arithmetic, not the loop's KV state).

Found on the way and fixed: `step_verify` never reset the per-step scalar arena, so the
batched GEMM path ran out of slots on its second call ("icb_scalars u32 arena exhausted").
`tests/verify_scalar_arena.rs` (E4B shapes, random weights) fails without the reset and passes
with it.
