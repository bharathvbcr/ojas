# Speed frontier: what breaks the current ceiling (2026-07-13 research sweep)

The MLX DFlash path is at ~31 tok/s median (2.5×), **draft-acceptance-bound**: linear-block
DFlash commits ~accept+1 tokens/verify (accept 3.5 on code, **2.2 on creative prose** → the
1.6× weak spot). Block-size tuning is exhausted (fixed=5 near-optimal; adaptive <4%, see
`bench/results/adaptive_block_finding.json`). Two frontier levers raise the ceiling:

## Lever A — Draft trees (DDTree / TAPS) — the acceptance lever ★

**Idea:** DFlash's single draft forward already emits per-position marginal distributions
`q_i(v)`. Instead of taking argmax per position (one linear candidate), build a **tree** of
the top-K tokens per position within a node budget B, and verify **all branches in ONE target
forward** using a tree-attention mask (each node attends only to root→ancestors→itself). Accept
the best matching path. More candidates where the draft is unsure (prose) → higher acceptance
at **no extra draft cost** (distributions already computed).

- **Reported:** +30–40% mean acceptance length; DFlash 5.56×→7.50× (Qwen3-8B MATH-500),
  4.81×→6.81× (Qwen3-4B HumanEval). Block-diffusion draft trees up to **3.6× on Gemma3**.
- **Papers:** DDTree (arxiv 2604.12989, code `github.com/liranringel/ddtree`), TAPS (2606.00487),
  Cost-Aware Diffusion Draft Trees (2606.01813). All Transformers/CUDA — **no MLX impl yet**.
- **Expected here:** lifts the 2.2-accept prose case most; plausible median 31 → ~38–42 tok/s
  and prose 19 → ~26. Biggest remaining MLX win.

### MLX implementation sketch (fork of `dflash.model_mlx.stream_generate`)
1. Draft: keep `draft_logits` (already have them); take `top_k` per position (K≈4, B≈16–24).
2. Tree build: best-first max-heap over prefix log-prob (DDTree Alg. 1) → list of nodes with
   parent pointers + per-node position depth.
3. Verify: flatten tree to a token sequence; build an `[N,N]` additive attention mask where
   node j attends to i iff i is an ancestor of j (or shared KV context). MLX
   `mx.fast.scaled_dot_product_attention(..., mask=tree_mask)` supports a custom additive mask.
   Also need per-node position ids (= depth) for RoPE, and per-node KV-cache offsets.
4. Accept: walk the tree root→leaf picking children whose token == target argmax at the parent;
   commit the longest accepted path + 1 bonus; trim target/draft KV by (tree_len − path_len).

**Risk/cost:** tree-attention mask + per-node position ids in MLX is the hard part (the linear
path uses a plain causal SDPA). Medium effort; validate against `golden_tokens_31b.json`
(still must be lossless — target verifies every accepted token).

## Lever B — Overlap scheduling (Spec V2 concept) — the overhead lever

SGLang's Spec V2 (lmsys blog 2026-06-15) got +33% by **overlapping host cleanup / KV alloc with
GPU compute** (not a model change; CUDA). The analog on the **custom gemma-metal engine** is the
same per-token-overhead fight already underway there (fuse dispatches, encode/GPU overlap; see
corrected `bottleneck.md` — 77% of the token is overhead). Not an MLX-path lever (mlx-lm already
overlaps via lazy eval), but confirms the native-engine direction is right. No new draft model
(Spec V2 released only a Qwen3.5 draft).

## Not levers (checked, ruled out)
- **Lower-bit/mxfp4 target:** decode is M=1 bandwidth-bound (same bytes for 4-bit); 3-bit = quality
  burn (against honest-lane doctrine). mxfp4 might speed M=block verify on NAX but marginal + 17 GB dl.
- **Wired memory:** no effect (weights already resident).
- **Adaptive block size:** <4%, single policy can't fit prose(3) and code(5) optima.
- **DFlash V2 draft:** none for Gemma-4 (only Qwen3.5-397B released).

## Lever A — ROI verdict (2026-07-13): BREAK-EVEN on 31B, DEFER ★→⏸

Built + unit-tested the algorithmic core (`bench/ddtree_core.py`, `bench/test_ddtree_core.py`,
19/19 CPU checks: best-first tree construction w/ prefix sharing, tree-attention mask, depth
positions, accept walk incl. zero/partial/full accept + bonus). Then modeled throughput from the
**measured** mlx-0.32 verify-cost curve (mlp `[10752x5376]` q4g64): M=1→274µs, M=6→403,
M=8→452, **M=10→588 (cliff)**, M=16→594.

Throughput(N) = accept(N)/(T_draft + T_verify(N)) with accept saturating at +35% (paper cap):

| tree budget N | accept | rel. throughput vs linear(M=6) |
|---|---|---|
| linear (M=6) | 3.56 | 1.00 |
| 8 | 3.84 | 0.98–1.01 |
| 16 | 4.45 | 0.90–0.98 |
| 24 | 4.67 | 0.94–1.02 |

**Best tree budget gives 0.98–1.02× on 31B — no win.** The M=10 verify cliff + saturating
acceptance means the extra tree-verify tokens cost ~what they earn. The DDTree paper's 1.35×
was on **small** models (Qwen3-4B/8B) where verify is cheap vs draft; at 31B, verify dominates.

**Tree drafting unlocks only when verify(M) gets cheaper:**
1. **Native engine true M>1 Q4 GEMM verify** (vs today's M×GEMV) — flatter verify(M) curve →
   trees pay off. The core here drops into the `.wip` `step_verify` (swap causal→tree mask +
   depth positions). **This is the right home for Lever A**, not the MLX fork.
2. **E4B (smaller target)** — verify cheaper relative to draft; re-run the model there.

## Recommendation (updated)
MLX path is **fully exhausted** — every lever now has a measured/analyzed verdict:
DFlash ✓ (2.5×), q4 draft ✓ (+6%), mlx 0.32 NAX ✓ (1.49×), block=5 ✓, adaptive ✗ (<4%),
wired mem ✗, lower-bit ✗ (doctrine/marginal), **draft trees ✗ on 31B (break-even; deferred to
native M>1 verify or E4B).** Remaining real speed is on the native engine (efficient batched
verify → then trees), owned by the concurrent session. `bench/ddtree_core.py` is ready for it.
