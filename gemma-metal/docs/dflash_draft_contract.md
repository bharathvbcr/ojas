# DFlash draft — MLX reference contract (for the native accept≈0 bug)

Authoritative per-step spec from `dflash/model_mlx.py` (`DFlashAttention`/`DFlashDraftModel`)
cross-checked against `mlx_lm/models/gemma4_text.py`, for `z-lab/gemma-4-31B-it-DFlash`.
The native `mean_accept≈0` (vs MLX ~3.0) is somewhere below. Ordered by suspicion. Use this to
diff `src/dflash.rs` step-by-step; validate each intermediate against the MLX golden dumper
(`bench/golden_parity.py` — extend to dump the intermediates named here).

**Ruled out:** RMS convention. This Gemma-4 (MoE variant) uses **plain-weight** `mx.fast.rms_norm(x, w, eps)`,
NOT Gemma-2/3 `(1+w)`. Native plain-weight `rms_norm`/`rms_qkv_rope` is correct for BOTH target and qwen3 draft.

## ★1 — Capture tap point — ✅ VERIFIED CORRECT (2026-07-14, by inspection)
MLX `_LayerHook` captures `DecoderLayer.__call__`'s return, whose **last op is `h = h * layer_scalar`**
(gemma4_text.py:388) — so the captured tensor is the **post-layer-scalar residual** at
`target_layer_ids=[1,12,23,35,46,57]`, concatenated last-axis → `[T, 6·5376]`.
Native `gpu_model.rs:2060` applies `scale_f32_inplace(self.x, layer_scalar)` then captures `self.x`
at :2103 — **same tensor (post-scalar residual). Tap point matches.** (The `layer_scalar` *value* was a
prior bug — "left scalars at 1.0 poisoned captures" gpu_model.rs:492 — now loading real weights.)
**Not the bug.** Remaining live suspects are ★1b (which forward's hidden feeds which block) + ★2 + ★3.

## ★1b — Capture FEED: which target forward's hidden conditions which block (now top suspect)
MLX `stream_generate`: the `hidden` used to draft block N comes from the **verify forward of block N−1**,
then **trimmed to `[:, :accepted+1, :]`**. h_ctx grows by exactly the accepted rows each block; the very
first block uses the prefill hidden.
- **Check:** native conditioner must be fed the **verify-forward** hidden rows (M positions), **trimmed to
  the accepted count**, not a single position and not the pre-trim rows. An off-by-one or wrong-forward feed
  gives a plausible-magnitude but wrong `h_ctx` → the exact accept≈0 symptom, with target still healthy.
- **Check:** `DFlashGpuConditioner::project_row` must consume the same rows, in order, that MLX's trimmed
  `hidden` holds after each verify.

## ✅ GOLDEN REFERENCE (MLX, produced 2026-07-14) — compare native against these
Input `[2,105,4368,1246]`, block 1 (bs=5 → S=4 ctx rows). Full: `bench/results/golden_intermediates_31b.json`.
Run native `--dflash-31b` with an intermediate dump on the SAME input; diff via `bench/compare_intermediates.py`.

| Quantity | MLX golden value |
|---|---|
| embed_scale | 73.32121 (=√5376) |
| target_next_argmax (anchor) | **531** |
| **draft proposed block tokens** | **[14359, 532, 107, 563]** ← native draft must reproduce these |
| target_hidden absmean @ L[1,12,23,35,46,57] | [0.168, 0.372, 0.404, 1.267, 1.255, 0.905] |
| fc_out absmean (pre hidden_norm) | 38.931 |
| h_ctx absmean (post hidden_norm) | 0.0699 |
| draft RoPE offsets (all 5 layers) | q=4, ctx=0, S=4 |

**Inspection verdict (2026-07-14):** native `forward_layer` structure + offsets are CORRECT —
block Q/K RoPE uses `caches[li].offset` after `+= ctx_t` (= `cache.offset + S` = MLX), ctx uses
pre-append offset. So the bug is **numerical**, localized by which absmean above the native fails:
- `target_hidden` absmean wrong → capture magnitude / `layer_scalar` value / residual-stream scale.
- `fc_out` wrong but target_hidden right → `fc` weights / Q4 quant of fc / concat order.
- `h_ctx` wrong but fc_out right → `hidden_norm` weights/eps.
- all right but proposed tokens wrong → draft attention (FA h128 scale/mask) or lm_head/softcap.

## ★2 — Block vs context RoPE offsets  (✅ verified correct in native — see golden ref above)
Per block, `S = ctx rows fed this block`. MLX:
- `queries` (block)   RoPE offset = `cache.offset + S`
- `ctx_keys`  (h_ctx) RoPE offset = `cache.offset`
- `prop_keys` (block) RoPE offset = `cache.offset + S`
- **Check:** native block Q/K must use `cache_off + ctx_t`, ctx K must use `cache_off`. If the block reuses
  `cache_off` (not `+ctx_t`), block and ctx collide in position space → attention garbage.

## ★3 — Sliding-window context skip (first 4 layers, window=2048)
MLX, for `is_sliding` layers, BEFORE projecting: `keep = window-1; if S>keep: skip=S-keep; x_ctx=x_ctx[skip:]; cache.offset += skip`.
The `cache.offset += skip` then feeds the RoPE offsets in ★2. Full layer (5th) keeps all ctx.
- **Check:** native clamps `ctx_t` to `keep` and adjusts `caches[li].offset` — verify the offset adjustment
  happens **before** the ctx-K RoPE and that block RoPE uses the *adjusted* offset + ctx_t.

## ★4 — Exact projections / scales (lower suspicion — mostly confirmed in native)
| Item | MLX value | 31B |
|---|---|---|
| embed | `embed_tokens(block) * embed_scale`, reuse target embed | `embed_scale = √hidden = √5376 ≈ 73.32` |
| h_ctx | `hidden_norm(fc(concat))`, fc `Linear(6·5376→5376)` no bias, hidden_norm plain-RMS eps 1e-6 | — |
| q/k norm | plain-weight RMS over head_dim=128, eps 1e-6 (applied to q and to ctx_keys+prop_keys) | — |
| v | **no v_norm** (raw v_proj) | — |
| FA scale | `head_dim**-0.5` ≈ 0.0884 (NOT target's post-QKnorm 1.0) | — |
| MLP | qwen3 SwiGLU: `down(silu(gate(x)) * up(x))` (SiLU, not gelu) | — |
| lm_head | target tied embed as_linear; softcap `tanh(z/30)*30` | cap=30 |
| sample | greedy argmax per masked position | — |

## Fast localization procedure
1. Fix input `[2,105,4368,1246]` (already used in native bench). Dump from MLX golden:
   `target_hidden[6][:,-1,:]` sample+checksum, `h_ctx[-1]`, per-layer draft `queries/ctx_keys` after RoPE,
   and the draft's proposed block tokens for block 1.
2. Dump the same tensors from native (`read_h_ctx` exists; add draft q/k reads).
3. First tensor that diverges = the bug's layer. Given the symptom (accept≈0, not NaN, target healthy),
   expect ★1 (capture tap) or ★2/★3 (offsets) — a *plausible but wrong* h_ctx, not a blow-up.

**Note:** spec-decode is exact, so once accept>0 the token stream must still equal
`bench/results/golden_tokens_31b.json`. The bonus token diverging even at accept=0 (native doc "open")
points at the **verify** path (target argmax after the block), not the draft — check the verify seed/offset
is the committed prefix's next position, independent of the draft.
