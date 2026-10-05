# Kernel-level speedups still open for a Mac decision (Qwen3.5-2B bf16, tessl via qd-metal, M5 Pro)

2026-10-03. Labels:
- **[V]** I read it in source at today's tree.
- **[R]** Reported in a document I read (tessl `docs/qwen35.md`, a paper, a PR), not remeasured here.
- **[I]** Inferred, from arithmetic over measured or reported numbers.
- **[U]** Unverified.

## Verified results

The banner that used to sit here, and item 2 below, are superseded. The ranking after this section is the original research, kept as history. Do not redo item 2.

Labels in this section: **[V]** measured. No inferred figures.

Product / full model (Apple M5 Pro, Qwen3.5-2B). Not all interleaved.

- Evening, before the option-pass batch: decision ~100 ms at T=200 (prefill ~45, two passes ~50, digests ~6). **[V]** From this folder's README. Not remeasured this session.
- After the equal-length batch, before GEMM tiles: decision T=200 median 94.22 ms, min 90.55, prefill 54.38, decode 34.03, digest 4.65, passes 61+61, bit-identical 5/5. **[V]**
- After GEMM tiles, unpaired: decision T=200 median 100.18 ms, min 99.44, prefill 56.44, decode 39.29, digest 4.59, bit-identical 5/5. Not faster than 94.22. **[V]**
- Prefill-only T=8192 after tiles: median 1844.80 ms, min 1828.47, 4441 tok/s. Earlier tiled prefill band 1931–2011 ms. Evening scalar prefill ~2180–2235 ms. **[V]** Not interleaved.
- 24-layer `bench_qwen35_layers` forward: 25.29 ms at T=61, 56.54 ms at T=200. GEMMs ~38 ms of the T=200 stage sum. GDN chunk scan 0.245 ms/layer. MLP SiLU+cast 0.089 ms/layer. **[V]**

Kernel paired, one process.

- One attention layer: `attn_prefill` median 0.1155 ms vs `flash_attn_rows` 0.2324 ms at T=200; 55.255 vs 176.212 at T=8192. Ranges did not overlap. The selector now always uses the tiled kernel. **[V]**
- Epilogue GEMM, Qwen out-proj, M=61: 64×64 0.0511 ms vs 128×64 0.0826 ms, bit-identical (worst abs diff 0). M=200 stays 128×64. **[V]**
- Plain GEMM, GDN in-proj N=8224: M=61 64×64 median 0.195 ms vs 128×64 0.268 ms, bit-identical. M=200 128×64 was faster (0.506 vs 0.725), so M>=128 stays wide. **[V]**
- Hazard-barrier skip dropped 283 barriers to 282. T=200 medians 53.8 vs 52.3 ms, ranges overlapped. Default unchanged. **[V]**
- Prefix attention `attn_prefix_rows`, suffix 61: prefix 200 → 0.151 ms/layer (0.91 ms ×6); prefix 8192 → 6.323 ms/layer (37.9 ms ×6). No existing tiled kernel accepts a shared prefix. No new kernel. **[V]**
- `GdnScanSlice` defaults to `Cols16`: paired batch 1 had Cols16 faster than Cols32 at T=200 and T=8192 and bit-identical, and paired batch 2 (one GDN scan layer, ABBA, 9 rounds, Apple M5 Pro) had Cols32 / Cols16 medians 0.1005 / 0.0781 ms at T=61 (ratio 0.777) and 0.3839 / 0.3210 ms at T=200 (ratio 0.836), with 0 mismatches in output and final state. Batch 4 was not remeasured. **[V]**

After `GdnScanSlice`'s default became `Cols16`, qd-metal's prefill uses that default (`GdnWorkspace::new`, no `with_scan_slice` in Lappi) and binds `qwen35_gdn_chunk_scan_bv16`. **[V, source]** An unpaired full-model rerun on Apple M5 Pro, same weight hash `92f6bd1c`, tessl HEAD `07f69906` with the Cols16 default: decision T=200, passes 61+61, bit-identical 5/5, median 103.90 ms, min 102.39, prefill 58.61, decode 41.46, digest 5.09. The prior unpaired run before this default, after GEMM tiles, was median 100.18, min 99.44, prefill 56.44, decode 39.29, digest 4.59. Delta +3.7 ms, the same scale as the ~1 ms layer-only Cols16 hint, not a separated product win. **[V]** Prefill-only T=8192: median 1981.55 ms, min 1950.87, 4134 tok/s. Prior unpaired: median 1844.80, min 1828.47, 4441 tok/s. Delta +137 ms, far outside the ~22 ms layer-only hint. The runs were not interleaved, so the cause is not the scan change and it is not grounds to revert Cols16. **[V measurements, I conclusion]**

Fixes, tests passed, source re-checked.

- The epilogue rejects a bias that aliases the output before the tile choice.
- `decode_batch` and `run_decision` both call `batch_score_rows` (checked `2*seq-1`) before the batched forward.
- The choice slot calls `decode_slots` once. The default still loops `decode_slot` twice.

Starting point (this folder's `README.md` and `raw/summary.txt`):
- **T=200 decision, ~100 ms:** prefill ~45 ms, two read-only 61-token option passes ~50 ms, three digests ~6 ms.
- **T=8192 prefill:** ~2,180 ms.

This list honours the prior rulings:
- No int8 weight-only.
- No new head-dim-256 attention kernel. Item 2 routes to a kernel tessl already ships.
- No ojas inference engine.

## Tessl vs PyTorch 2.12.1 MPS, 2026-10-04

Apple M5 Pro, one GPU-lock hold. Same work on both sides: real Qwen3.5-2B snapshot `b1485b2f`, prefix state kept, 17 answer-letter rows at the last position, no full-vocabulary head. Lengths interleaved. Load average 4.61–5.68.

| T | Tessl median/min ms | PyTorch median/min ms | PyTorch÷Tessl median |
|---|---|---|---|
| 200 | 55.98 / 53.70 | 119.94 / 116.81 | 2.14× |
| 2048 | 420.49 / 401.55 | 891.08 / 835.50 | 2.12× |
| 8192 | 1969.24 / 1882.59 | 4371.41 / 4317.57 | 2.22× |

Tessl bench line: embed gather + 24 layers (state kept) + final norm + 17 answer rows; median of 7 after 2 warm-up. PyTorch: `torch_task_speed.last_only`, bf16 sdpa MPS, median of 10 after 3 warm-up. Load time about 2 s Tessl vs 19–22 s PyTorch because each length reloaded. Token streams differ: Tessl repeats tokenized `model.rs` to length T; PyTorch uses a val prompt of that length. Shapes match. **[V]**

PyTorch gated-delta still used `torch_chunk_gated_delta_rule`. **[V]**

flash-linear-attention 0.5.2 was pip-installed into `/Users/bharath/.venvs/ml` (import `fla` works; also pulled fla-core 0.5.2). `causal-conv1d` was not installed. Qwen3.5 from the same snapshot still printed: "The fast path is not available because one of the required library is not installed. Falling back to torch implementation." Transformers 5.12.1 only treats `fla` and `causal_conv1d` as available when `torch.cuda` is available; CUDA is false and MPS is true on this machine. Direct import of `chunk_gated_delta_rule` fails with `ModuleNotFoundError: triton`. One T=200 forward after a 1-token forward took 351.2 ms; that is one run, not the 119.9 ms median above. **[V]**

## Two things found in the source that shape the ranking

1. **qd-metal's prefill uses the scalar attention kernel — superseded.** The selector now always uses the tiled kernel (verified results). The notes under this item are what the source said when the list was written.
   - `Lappi-decision/crates/qd-metal/src/model.rs:812` calls `nn::flash_attn_rows`. tessl's own header calls it "scalar f32: one simdgroup per query row" (`tessl/kernels/qwen35_attn_tiled.metal:3-7`).
   - tessl also ships `qwen35::attn_prefill` (`tessl/src/qwen35.rs:2155`), which is FlashAttention-2 with both products on `mpp::tensor_ops::matmul2d`. tessl's layer bench uses it by default. Only `--attn-rows` selects the scalar kernel (`tessl/src/bin/bench_qwen35_layers.rs:7,13-15`).
   - The 6,213 tok/s at 8K came from `attn_prefill`. The same bench with `--attn-rows` measured 4,189 tok/s (`tessl/docs/qwen35.md:932-936`) [R], close to qd-metal's ~3.8k. **This most likely explains the "unmeasured discrepancy" [I].**
2. **qd-metal's prefill time steps up at every multiple of 128 tokens [V numbers, I cause].**
   - Median prefill from `raw/summary.txt`, against the number of 128-row M tiles:

     | Prefix tokens | 128-row M tiles | Prefill (ms) |
     |---|---|---|
     | 115 | 1 | 28.2 |
     | 150 | 2 | 41.7 |
     | 165 | 2 | 42.2 |
     | 193 | 2 | 44.9 |
     | 271 | 3 | 61.9 |
     | 310 | 3 | 65.0 |
     | 399 | 4 | 93.0 |
     | 762 | 6 | 152.6 |

   - That is ~21–25 ms per 128-row tile throughout.
   - tessl's NN GEMM picks its tile from N only. `nn_coop_kernel(_m, n, _k, …)` returns 128×64 for every N > 512 (`tessl/src/gemm.rs:817-836`) [V].
   - The GDN chunk is 64 rows, and it fits the data worse: 150 → 193 adds a chunk but only 3 ms.

## Ranked list (top 8)

Historical research, as drafted. Item 2 is superseded. Do not redo it.

The rank is by expected milliseconds saved on product-length decisions (T≈130–800), with 8K as the tie-break. Cheapness counts too: items 1, 2, 4 and 5 reuse code that already exists.

### 1. One batched option pass instead of two sequential ones

- **What it changes.**
  - Both 61-token suffixes run as batch 2 over the same read-only snapshot: one GEMM with M=122 instead of two with M=61.
  - The weights are read once instead of twice, and launches are halved (one forward instead of two). Arithmetic is unchanged.
  - qd-metal already reads GDN/conv state at batch stride 0 and prefix K/V through `attn_prefix_rows` (model.rs:34-39 [V]). This is Fable ruling 2, item A, still unbuilt.
- **Why it should pay [I].** M=61 and M=122 both fit in one 128-row tile, and one tile costs ~21–28 ms (finding 2). So the batched pass should cost about what one pass costs now: **~-22 to -25 ms per decision at every T**, about 25% of a T=200 decision.
- **bf16 numerics.** Weights and dtypes are unchanged. It is not bit-identical: `gpu.rs` bounds batch versus batch-1 at 1e-4 (ruling 1) [R]. The state digest is unaffected (read-only). Logprob parity needs a re-run.
- **Cheapest experiment.** Add the `decision.rs` arm `passes=sequential|batched`, then one bench under the lock at T=131,409,770.
  - Confirm: Δ ≤ -15 ms at T=131 and max |Δlogprob| within the parity bound.
  - Kill: batched ≥ 0.8 × sequential.
- **Blockers (from the ruling).** Wording in `docs/schema-api.md:364-368` needs the human's approval. The product wiring waits on the owner of `backend.rs`.
- **Source.** Juravsky et al., *Hydragen: High-Throughput LLM Inference with Shared Prefixes*, arXiv 2402.05099, **2024**: batch the queries that share a prefix so one matrix-matrix product replaces many matrix-vector reads of the same data. Here the shared data is the weights plus the prefix state.

### 2. Route qd-metal's prefill attention to tessl's existing tensor-op `attn_prefill` — superseded

- **Superseded.** Prefill attention already goes through the tiled kernel, and the selector always uses it. The ~580 ms at 8K below is the original inference. Measured times are in "Verified results". Do not redo this.
- **What it changes.** (original claim, kept as history)
  - `nn::flash_attn_rows` is replaced by `qwen35::attn_prefill`: Q·Kᵀ and P·V go onto the M5 Neural Accelerators through `matmul2d`, with an f32 online softmax between them. Launch count is the same.
  - The arithmetic is identical in exact math. Per-layer attention time at Qwen3.5-2B shapes, `attn_prefill` vs `flash_attn_rows` [R, `tessl/docs/qwen35.md:938-940`]:

    | T | attn_prefill (ms) | flash_attn_rows (ms) |
    |---|---|---|
    | 1024 | 0.91 | 2.28 |
    | 2048 | 2.89 | 10.12 |
    | 8192 | 43.1 | 140.1 |

  - Over 6 attention layers that is **~-580 ms at 8K, ~-43 ms at 2K, ~-5 ms at 770, ~0 at T=200 [I]**. Expected 8K prefill: ~2.18 s → ~1.6 s [I].
- **bf16 numerics.** Weights stay bf16. The accumulation order changes, so the result "agree[s] to rounding, not bit for bit" (`qwen35_attn_tiled.metal:25-26`) [V].
  - Downstream activations and state change slightly, so the v1 digest pins and the `qd-metal-parity` run must be redone.
  - The 2K "saving not shown" note (`docs/qwen35.md:946-949`) is unresolved [R].
- **Cheapest experiment.**
  - First: `bench_qwen35_layers 8192` with and without `--attn-rows` on today's tessl tree, in one window. This also closes `GAP-QDM-8K-PREFILL-3797-VS-REPORTED-6213`.
    - Confirm: ~2.9× attention stage ratio and ~1.3–1.5 s forward.
    - Kill: the two arms within 10%.
  - Then a one-line A/B in qd-metal's prefill at T=2048 and 8192 for ms and max |Δlogprob|.
- **Source.**
  - Dao, *FlashAttention-2*, arXiv 2307.08691 (2023; ICLR **2024**).
  - Waschkowski et al., *BaseRT … with Apple M5 Neural Accelerators*, arXiv 2607.19438, **2026**: prefill attention with "both the QKᵀ score product and the PV output product through matmul2d" on an M5 Pro.
  - Apple, *Metal Performance Primitives Programming Guide* and WWDC25 session 262, **2025**.

### 3. M-aware GEMM tiling for 60–800-row prefills (tile quantization)

- **What it changes.**
  - Pick the M tile from M, as well as N, so the last M tile isn't mostly padding. For example, use the existing 64×64 `matmul2d_tensorops_bf16_f32_64x64_sg4` or a narrower tail tile for the remainder rows. Split-K is the other option.
  - Launch count is the same. Arithmetic on padding rows and wasted matrix-unit issue are removed.
  - If time is proportional to tiles (finding 2), the upper bound is ~21 ms × T/128: **T=150: 41.7 → ~25 ms; T=193: 44.9 → ~32 ms; T=271: 61.9 → ~44 ms [I]**.
  - An alternative explanation would need a different fix: each M tile re-streams its weight panel, which calls for larger M tiles and rasterization, not smaller tiles [I]. The experiment separates the two.
- **bf16 numerics.**
  - A different M-tile size keeps the K-loop order per output element. Whether `matmul2d`'s internal reduction is bit-stable across descriptor sizes is **[U]**, so it needs a bit-compare.
  - Split-K changes summation order, so it is not bit-identical. MLX found a split-K variant "rounds partial sums to the input dtype" (issue #4613) — avoid that with f32 partials.
- **Cheapest experiment.**
  - (a) Prefill-only sweep, T = 100…420 in steps of 8, in the existing bench. A staircase at 128/256/384 confirms the tile model; one at every 64 points to GDN.
  - (b) `bench_gemm_tile_tune` at M ∈ {61, 122, 150, 193, 271} for the model's (N, K) shapes, 128×64 vs 64×64. Bit-compare the outputs.
  - Kill: ≤5% gain at M=150/193.
- **Source.**
  - NVIDIA, *Matrix Multiplication Background User's Guide*, §3 "Dimension Quantization Effects" (current docs, accessed 2026-10-03): tile and wave quantization.
  - MLX PR #3120, *Add split-K for quantized matmul (small M)*, merged **2026-03-21**: small M "severely underutilizes the GPU"; split-K targets ~512 threadgroups.
  - MLX's shape-dependent steel-GEMM tile choice. I saw it only through a reproduction (candle `mlx_gemm.rs`), so MLX's exact current rule is [U].

### 4. Use tessl's fused `qwen35::swiglu` in qd-metal (and later, a GLU epilogue on the gate/up GEMM)

- **What it changes.**
  - qd-metal runs `nn::mlp_silu` (f32 → f32) and then `cast_f32_to_bf16_into` (model.rs:874-875) [V]. `qwen35::swiglu` does both in one kernel, writing bf16 directly (`tessl/src/qwen35.rs:1952`) [V].
  - That is one fewer launch and barrier per layer, and one fewer f32 write plus re-read of the T×6144 intermediate.
  - Reported: "half the time of `mlp_silu` + cast (0.28 vs 0.56 ms at 1024, 2.43 vs 4.05 at 8192)" per layer [R, `docs/qwen35.md:956-958`]. Over 24 layers: **~-1.5 ms at T=200, ~-7 ms at 1K, ~-39 ms at 8K [I]**.
  - The follow-up is BaseRT's fused gate/up GEMM with SiLU applied on the accumulators. That removes the f32 `m_gate`/`m_up` round trip too: ~4 × T × 6144 × 4 B of traffic per layer [I]. tessl's `Activation` enum has no two-operand GLU epilogue today (`gemm.rs:618-630`) [V], so that part is new kernel work.
- **bf16 numerics.** It is the same f32 SiLU and product followed by one bf16 rounding. Bit identity with `mlp_silu` + cast is **[U]** and needs a check: tessl notes `mlp_silu` and the epilogue `Silu` "match", but I did not compare the two kernel bodies.
- **Cheapest experiment.** `bench_qwen35_layers --mlp-unfused` vs default at T=200, 1024 and 8192 on today's tree. Then swap the call in qd-metal and bit-compare logits and the state digest at T=131.
- **Source.** BaseRT, arXiv 2607.19438, **2026** (fused gate/up with SiLU on tensor-core accumulators). llama.cpp PR #16220, *metal : fuse NORM + MUL + ADD*, merged **2025-09-25** (the same fusion pattern for norms).

### 5. GDN chunk scan with 16-column slices (`GdnScanSlice::Cols16`)

- **What it changes.**
  - The sequential inter-chunk scan launches `v_dim/16` threadgroups per head instead of `v_dim/32`: twice the threadgroups, for occupancy at batch 1. Total arithmetic is the same.
  - qd-metal builds `GdnWorkspace::new(...)` with the default `Cols32` (model.rs:609) [V].
  - Reported: the scan is ~75% of the GDN chunked rule, and that rule costs 18 layers × 1.4 ms at 1K and 18 × 8.4 ms at 8K [R, `docs/qwen35.md:954-963`]. So the ceiling is roughly 10–20% of ~25 ms (1K) or ~150 ms (8K) [I]. The actual gain is unmeasured.
- **bf16 numerics.** tessl states the two are **bit-identical**: "each output element runs the same arithmetic either way" (`qwen35.rs:801-803`) [V as a claim; not tested by me].
- **Cheapest experiment.** `bench_qwen35_layers --gdn-scan16` vs default at T=200, 1024 and 8192, then the qd-metal digest bit-compare.
  - Kill: <3% of GDN stage time.
- **Source.**
  - Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv 2412.06464 (ICLR **2025**): chunkwise WY form, with inter-chunk recurrence still sequential.
  - flash-linear-attention's chunk kernels, which use the same structure (`fla/ops/gated_delta_rule`; via DeepWiki, so secondary).
  - Wave quantization: the NVIDIA guide above.

### 6. Let independent dispatches overlap (stop the barrier after every dispatch)

- **What it changes.**
  - tessl's `Binder::dispatch` puts a Dispatch→Dispatch device barrier after **every** dispatch unless hazard mode is on (`tessl/src/dispatch.rs:371-377`; default off, `ab_flags.rs:123-127`, env `TESSL_HAZARD_BARRIERS`) [V]. qd-metal never sets it [V, rg].
  - With barriers only on real RAW/WAR edges, independent kernels overlap: MLP gate and up GEMMs, Q/K/V-side elementwise ops, the GDN gate computation next to the conv. Bytes and arithmetic are unchanged.
  - This matters most where single kernels underfill the GPU: the option passes and short prefills.
  - Reported llama.cpp gains on Apple GPUs: pp512 +1–6%, decode +3–17% (PR #15929); a further +0–11% from reordering (PR #19555) [R]. **Here, guess ~2–5 ms per decision [I].**
- **bf16 numerics.** Bit-identical if every RAW edge keeps its barrier. Same kernels and same order per output. The risk is correctness, not numerics: tessl warns "Do not enable as default without RAW-edge barriers".
- **Cheapest experiment.** Run qd-metal's GPU bit-identity tests (`gpu_digest_paths_are_bit_identical`, logits) with `TESSL_HAZARD_BARRIERS=1`.
  - Any mismatch kills it as-is (missing explicit barriers).
  - If identical, run one decision bench at T=131 and 409.
  - The env var's coverage of qd-metal's call sites is **[U]**.
- **Source.** llama.cpp PR #15929, *metal : allow ops to run concurrently*, merged **2025-09-13** (memory-range tracking, so a node runs concurrently unless it reads or writes a range an in-flight node writes), and PR #19555, *metal : improve concurrency*, merged **2026-02-13**. Apple WWDC25 *Discover Metal 4* (session 205, **2025**) for the MTL4 explicit-barrier model tessl already uses.

### 7. Remove host-side waits and encode tax from the decision path

- **What it changes.**
  - (a) Run the worker on `GpuRuntime::new_inference`, which has no `CounterHeap` timestamps. tessl calls the timestamps a "host encode tax".
  - (b) Encode the option pass(es) while the GPU is still finishing the prefill, and sync only where the digest contract needs host-visible bytes.
  - (c) If CPU encoding is on the critical path, replay the fixed-shape option-pass dispatch list from an indirect command buffer. tessl has a decode ICB tape (`crate::decode_icb`, referenced in `dispatch.rs:368-382`) [V], but nothing in qd-metal uses it [V, rg].
  - Bytes and arithmetic are unchanged. Launch overhead is reduced. Estimated at 0–5 ms per decision (ruling 2, item E) [I].
- **bf16 numerics.** Bit-identical.
- **Constraint.** Do not overlap a digest with a GPU pass. The digest is the host-observed read-only check (ruling 2 §1), so the sync before each digest stays.
- **Cheapest experiment.**
  - The `decision.rs` `RuntimeKind::Inference` arm already exists (decision.rs:130-176) [V]. Run it against `new` at T=131.
  - One Instruments Metal System Trace of a T=200 decision will show GPU idle gaps between command buffers.
  - Kill: total gap time <2 ms.
- **Source.** llama.cpp PR #15906, *metal : make the backend async v2* (**2025**, merged September 2025). Apple WWDC25 *Discover Metal 4* (MTL4 command allocators and argument tables, **2025**). Apple developer docs, `MTLIndirectCommandBuffer`, current.

### 8. Fold the remaining small ops into neighbouring kernels (RMSNorm, residual)

- **What it changes.**
  - Per layer, qd-metal still issues `rms_norm_bf16` twice as standalone dispatches (model.rs:689, 869) [V], each with its own barrier (item 6). Residual adds are already GEMM epilogues (`residual_add()`), and QK-norm plus RoPE and conv+SiLU are already fused.
  - Folding the post-norm into the preceding o-proj/down epilogue, or into the next GEMM's prologue, removes 48 launches and barriers per forward and one bf16 write/read of T×2048.
  - Byte savings are trivial at T=200 (~0.8 MB per norm); the win is launches. **Expected ≤1–2 ms per decision [I].** It is ranked last because items 1–7 are larger or cheaper.
- **bf16 numerics.** A prologue fold that keeps "normalise in f32, round to bf16, then GEMM" is bit-identical. A fold that applies the 1/rms scale after the GEMM is algebraically equal but rounds differently, so it is **not** bit-identical.
- **Cheapest experiment.** Before any kernel work, count dispatches and barriers per forward from tessl's `infer_trace` counters (`dispatch.rs:366,378`). Then compare the per-launch overhead (~µs each, [U] on M5) against the T=200 budget.
  - Kill: dispatch overhead below 2% of the forward.
- **Source.** llama.cpp PR #16220 (**2025**, NORM+MUL+ADD fusion) and PR #16102, *metal : fuse non-sequential nodes* (merged **2025-09-28**).

## Expected end state (all [I], to be replaced by measurements)

Original inference. Item 2 is superseded; its 8K line is not a remaining saving.

- **T=200 decision:** ~100 ms now. Item 1 brings it to ~75–78 ms, item 3 to ~65 ms, and items 4–8 to ~60 ms.
  - The floor at this length is one prefill plus one batched pass plus digests.
  - Each pass is bounded below by one weight read: ~16 ms at ~240 GB/s, as reported, not remeasured.
- **T=8192 prefill:** ~2.18 s now. Item 2 brings it to ~1.6 s, and items 4 and 5 to ~1.5 s.
  - Reference points: tessl's own bench reported 1.32 s at 8K with random weights [R].
  - On the same M5 Pro, BaseRT reports Qwen3.5-2B at 7,270 tok/s (Q4) or 7,268 tok/s (Q8) at pp2048. mlx-lm 0.31.3 reports 7,003 / 6,734 (BaseRT Tables 2–3 [R]).
  - tessl's bf16 at 2K is ~4.8k tok/s [V, summary.txt]. Those are quantized engines, so compare with care.

## Considered and not ranked

- **Restricting the LM head to the answer rows:** already done. `score_answer_rows` reads only the answer embeddings (`tessl/kernels/qwen35_score.metal:1-14`) [V].
- **Speculative or multi-token decoding:** there is no generation in a decision.
- **bf16 K/V cache or bf16 attention inputs:** this would halve K/V bytes, but it changes the stored state bytes, so the digest and contract change. Human's call.
- **GPU-side digest:** closed by ruling 2.
- **Re-tuning the attention tile:** tessl's sweep found the four tiles "within the noise of each other at 8192" [R].
