# Research sweep: Fast Gemma winners · MTP · DFlash · custom-kernel frontier (2026-07-18)

Companion to [`audit_deep_2026-07-18.md`](audit_deep_2026-07-18.md) (F1–F6: evidence integrity,
dispatch tax, hazard default, host tax, E4B graph fidelity, DFlash economics; quantified
**~37 µs/dispatch**, ~460 dispatches/token), [`speed_research_frontier.md`](speed_research_frontier.md)
(2026-07-13 sweep), and experiment notes
[`36-native-dflash-parity-accept`](../../../MLSystemsLab/experiment-notes/gemma-metal/36-native-dflash-parity-accept.md) /
[`37-golden-token-parity`](../../../MLSystemsLab/experiment-notes/gemma-metal/37-golden-token-parity.md)
(historical accept~1 / greet16 0/16 / `target_next` 929≠531 — **superseded** under
capture-on: greet16 16/16, `target_next=531`; see
[`lane_a_todo_status_2026-07-19.md`](../bench/results/lane_a_todo_status_2026-07-19.md)).
Speculative residual is **draft-accept**, not Hot `target_next`. Everything below is
mapped to gemma-metal on M5 Pro; sources at the end.

---

## 1. Fast Gemma Challenge — what actually won (and what it means here)

Final numbers (Jun 26 – Jul 2, single A10G 24 GB, E4B): frontier **491.8 TPS** (quality burn),
best **lossless 315 TPS**, ~5× over baseline. Anatomy of the 5× (pebblous teardown + vLLM
issues):

| Lever | Gain | Applies to gemma-metal? |
|---|---|---|
| **Kernel-path normalization** — vLLM's FlashAttention doesn't support head_dim 256/512, fell to a Triton detour (~9 tok/s on 4090, vLLM #38887). Restoring a proper attention path was the single biggest lever (×many) | 9 → 60–100 tok/s | **Already done natively.** Our dual FA (`flash_attn_swa_h256` / `_global_h512`) is exactly the fix the challenge (and FA4 on Hopper) needed. Decode FA is ~5% of our step — no further win here |
| PagedAttention + continuous batching | ×3–5 | N/A (bs=1 local) |
| FP8 / NVFP4 quantization | ~×2 | Against honest-lane doctrine at M=1 (same bytes for Q4; quality risk — note the challenge's FP8 **logit-saturation bug** #39407, silent quality loss) |
| **MTP drafters** | ×1.7–2.66, accept ~96.5% first-position | Directly relevant — §2 |
| **torch.compile** — trims kernel-call overhead | finishing move | **This is our F2.** CUDA-graph/compile is the NVIDIA analog of killing our 37 µs/dispatch tax — §4 |

**Takeaways:** (1) the challenge's lossless stack is *structurally identical* to our lane
(right attention kernels + Q4 + speculative + dispatch-overhead removal) — nothing exotic was
left on the table; (2) the meta-lesson from the agent swarm was that the **verify loop and
persistence** won, and that plausible-looking patches with silent quality loss (FP8 saturation)
were the main hazard — which is precisely our parity-gate discipline; keep it.

---

## 2. MTP: Google's official drafters change our 31B calculus

Google released **paired MTP assistant checkpoints for every Gemma 4 size** (2026-05-05,
Apache 2.0): a **4-layer transformer that cross-attends to the target's KVs**, fed by the
target's last-layer activations + token embeddings, with its own embedder. Up to 3× claimed;
supported day-one in **LiteRT-LM, MLX, HF Transformers, vLLM**.

Independent H100 head-to-head (JarvisLabs, SPEED-Bench, vLLM, c=1, greedy):

| Target | Baseline | MTP (k=8) | DFlash (k=15) |
|---|---|---|---|
| **31B dense** | 40.3 tok/s | **125.3 (3.11×)** ← wins | 122.1 (3.03×) |
| 26B-A4B MoE | 177.1 | 264.2 (1.49×) | **306.4 (1.73×)** ← wins |

Details that matter for us:

- **MTP has *higher per-position acceptance* than DFlash on both targets**; DFlash wins on MoE
  only because its draft cost is flatter and the MoE target step is cheap. Our target (31B
  dense on M5) is the regime where **MTP won**.
- Acceptance **collapses after the first few positions** for both — consistent with our
  block=5 optimum and the M=10 verify cliff; nobody should chase big blocks on 31B dense.
- Category spread (code 3.8× ↔ roleplay 1.4–1.6×) matches our measured 19–36 tok/s prompt
  spread almost exactly.
- TTFT degrades under concurrency with spec-decode — irrelevant at bs=1 local.

**Action for gemma-metal:** we bet on DFlash-only for 31B (D14) partly because clustered
per-token MTP measured poorly. But `google/gemma-4-31B-it-assistant` + **MLX MTP support**
now exists — a 30-minute A/B on the MLX side (MTP drafter vs our DFlash serve at bs=5) decides
whether the *native* port should verify MTP blocks instead of (or as fallback to) DFlash.
The engine work is shared: both need exactly our `step_verify` M>1 path. Our `mtp.rs`
E4B-assistant experiment (75% accept) is architecturally the same 4-layer cross-KV design —
it was never tried on 31B with block verify.

---

## 3. DFlash (ICML 2026) — details we haven't exploited

From the paper/site (arXiv 2602.06036): up to **6× lossless** on Qwen3-8B, ~2.5× over EAGLE-3.
Mechanics worth checking against our port:

1. **Feature fusion**: hidden features from layers *uniformly sampled across the target*,
   fused by a light projection (our capture layer_ids + conditioner FC — matches).
2. **KV injection into *every* draft layer** — the fused context enters each draft layer's
   K/V projections and *stays in the draft KV cache*. This is the load-bearing difference vs
   EAGLE-3 (input-only conditioning); acceptance *scales with draft depth* because of it.
   **ALREADY DONE** in `DFlashGpuDraft` (`propose_block` → `forward_layer` for every draft
   layer with shared `h_ctx`). Do not re-chase. Hot `target_next` 929≠531 / greet16 0/16
   ([exp 37](../../../MLSystemsLab/experiment-notes/gemma-metal/37-golden-token-parity.md)) is
   **superseded** under capture-on (greet16 16/16, `target_next=531` —
   [Lane A status](../bench/results/lane_a_todo_status_2026-07-19.md)). Remaining
   speculative gap is **draft-accept** (GPU Q4Mlx proposals vs host-dense/MLX;
   product mean_accept≈2.43 vs host-dense 3.0) plus second-order draft fidelity
   ([exp 36](../../../MLSystemsLab/experiment-notes/gemma-metal/36-native-dflash-parity-accept.md);
   audit [F6](audit_deep_2026-07-18.md#f6-dflash-31b-verify-graph-fixed-economics-still-upside-down)).
3. Draft reuses target embed/LM head; single denoising step; block 16 variants (`-b16`) exist
   for Qwen — draft cost is flat in block size, so *verify* cost is the only reason to stay
   at 5.
4. Reference verify/accept implementations now live in **SGLang (PR 16818)** and **vLLM
   (PR #41703)** — better ground truth for `accept_block` semantics than reverse-engineering
   the z-lab repo.

Frontier variants (from the 07-13 sweep, unchanged): DDTree/TAPS draft trees are break-even on
31B **until verify(M) flattens** — see §4/§5; `bench/ddtree_core.py` is ready and waiting.

---

## 4. Custom-kernel frontier: megakernels — the literature agrees with our F2 number

The low-latency world converged on the same diagnosis we measured (~37 µs/dispatch, GPU mostly
idle between small kernels):

| Work | Result | Transferable idea |
|---|---|---|
| **Hazy Research "No Bubbles" megakernel** (Llama-1B, H100/B200) | Entire forward = **1 kernel**; **78% of HBM BW** at bs=1; 1.5–2.5× over vLLM/SGLang; <1 ms forward | On-GPU **instruction interpreter** (each SM executes a pre-scheduled instruction list); **paged shared memory** so the next op's weight loads start while the previous op drains; **global atomic counters** for dependencies instead of kernel boundaries; chunked MLP so down-proj starts per-chunk |
| **MPK / Mirage megakernel compiler** (Zhihao Jia) | Compiles an LLM into one kernel automatically; 1.2–6.7× latency | You don't hand-write everything; a scheduler + op templates suffice — our `KernelId` enum + static decode graph is exactly the input such a scheduler needs |
| **Ada-MK** (arXiv 2605.11581) | DAG-search over fusion boundaries; +23.6% over TensorRT-LLM | Fusion boundary choice is searchable — A/B per-layer fusion granularity instead of guessing |
| **KOG single-kernel engine** (MI300X) | Whole engine in one kernel on AMD | Portability proof: the pattern isn't CUDA-specific |
| Their measured launch costs | 2.1 µs/launch CUDA stream, 1.3 µs CUDA-graph | **Ours is ~37 µs** — Metal 4 dispatch + argument table + auto device barrier. We have ~20× more per-launch overhead to win back than CUDA does |

### Mapping to Metal (what a "Metal megakernel" realistically is)

Metal has no grid-wide forward-progress guarantee, so a literal single persistent kernel is
risky. The practical ladder on Apple GPUs, in increasing ambition:

1. **Layer-block fusion (safe, biggest step):** one kernel per phase — `rms+qkv+rope+kv_store`
   fused, FA as-is, `o_proj+postnorm+residual` (exists), `rms+gate_up+gelu+down+postnorm+
   residual+layer_scalar` in one — → **~4–5 dispatches/layer ≈ 180/token** (E4B ~2× on the
   launch tax). Intra-kernel RAW replaces inter-dispatch barriers via `threadgroup_barrier` +
   grid sizing that keeps each dependency inside one threadgroup where possible.
2. **Encode-once, replay per token:** the decode graph is static; only `pos`/seed change.
   Move per-token scalars into a small GPU buffer that kernels *read* (instead of bound
   constants), then re-commit the **same pre-encoded command buffer / indirect command
   buffer** every token. Kills the ~2.5 ms host encode *and* most of the argument-table
   traffic — the Metal analog of CUDA graphs. (Verify MTL4 CB re-commit / MTLIndirectCommandBuffer
   compute support on macOS 26; fall back to two pre-encoded ping-pong CBs.)
3. **Persistent interpreter kernel (Hazy-style, experimental):** one dispatch of
   `#SM`-matched threadgroups looping over an instruction stream with atomic-counter deps.
   Works on paper with Metal atomics; no forward-progress guarantee — prototype on the mini
   graph only.

Steps 1+2 together attack the full 17 ms/token launch tax with no scheduling heroics, and are
the concrete version of audit plan items #3/#7.

### Metal/Apple-specific kernel resources

- **WWDC26 session 330 "Optimize custom ML ops with Metal tensors"** + MPP TensorOps: the
  M5 GPU **Neural Accelerator** path we already use for prefill GEMM. The native analog of
  MLX 0.32's 1.5–2× NAX verify win is moving `step_verify_gemm`'s M∈[2,8] GEMMs from
  simdgroup kernels to **TensorOps/MPP** — this both speeds block verify *and* flattens the
  verify(M) curve, which is the stated unlock condition for DDTree draft trees
  (`speed_research_frontier.md` Lever A verdict).
- **Apple MLX-on-M5 research note** (machinelearning.apple.com): confirms NAX is
  matmul-shaped (prefill/M>1) — spec-decode block verify is how decode borrows it.
- **Open-TQ-Metal** (arXiv 2604.16957): fused compressed-domain attention for long context on
  Apple Silicon — relevant later for KV-compression, not for our short-ctx gates.
- **mlx-metal-kernels** (community repo): fast attention/decode/KV-cache Metal primitives —
  worth diffing their GEMV/attention against ours for the 2560² occupancy hole (93–113 GB/s).

---

## 5. Updated unified plan (research → engine)

Ordered by expected tok/s per unit effort, merged with the 07-18 audit plan:

1. **MLX A/B: official 31B MTP drafter vs DFlash serve** (½ day, no engine work). If MTP ≥
   DFlash on M5 like on H100-dense, the ship lane improves for free and the native verify
   target may switch drafters.
2. **Layer-block fusion** (§4.1) → ~180 dispatches/token: E4B ~24 → ~40+; 31B ~8.5 → ~11.
3. **Encode-once/replay** (§4.2): +2–3 ms/token back, near-zero host encode; prerequisite
   groundwork for any interpreter experiment.
4. **Verify-GEMM on TensorOps/NAX** (§4 resources): flattens verify(M); re-run the DDTree
   break-even model — trees may flip to a win; also directly lifts DFlash/MTP block verify.
5. **Draft-accept parity (§3.2 retargeted):** Hot `target_next`/greet16 under capture-on
   is DONE (Lane A). Close GPU draft proposals vs host-dense/MLX Q4Mlx so product
   mean_accept moves 2.43→~3. Per-layer `h_ctx` KV injection is already landed — not
   on this checklist.
6. **Persistent-interpreter prototype on mini** (§4.3) only after 2–4 land.

Success math: fused greedy 31B ~11 × (1+accept≈3)/(verify≈1.3) ≈ **25–30 tok/s native** —
both gates, honestly.

---

## Sources

- Fast Gemma Challenge: [dashboard](https://huggingface.co/spaces/gemma-challenge/gemma-dashboard) · [results tweet](https://x.com/googlegemma/status/2075611948985835877) · [Pebblous teardown](https://blog.pebblous.ai/report/multi-agent-vllm-gemma4/en/) (vLLM #38887 fallback anatomy, FP8 #39407, lever stack)
- MTP: [Google blog — MTP drafters](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/) · [MarkTechPost summary](https://www.marktechpost.com/2026/05/06/google-ai-releases-multi-token-prediction-mtp-drafters-for-gemma-4-delivering-up-to-3x-faster-inference-without-quality-loss/) · [JarvisLabs MTP-vs-DFlash H100 benchmark](https://jarvislabs.ai/blog/gemma-4-mtp-vs-dflash-benchmark)
- DFlash: [arXiv 2602.06036](https://arxiv.org/abs/2602.06036) · [project page](https://z-lab.ai/projects/dflash/) · [repo](https://github.com/z-lab/dflash) · [vLLM speculators docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash/)
- Megakernels: [Hazy Research — No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) · [MPK — Compiling LLMs into a MegaKernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) · [Ada-MK](https://arxiv.org/abs/2605.11581) · [KOG MI300X single-kernel engine](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/) · [Deep Kernel Fusion](https://arxiv.org/pdf/2602.11808)
- Metal/Apple: [WWDC26 #330 Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/) · [Apple ML Research — MLX on M5 NAX](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) · [Open-TQ-Metal](https://arxiv.org/abs/2604.16957) · [mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels)
