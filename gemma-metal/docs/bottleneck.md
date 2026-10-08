# gemma-metal bottleneck report (E4B Q4)

**Date:** 2026-07-13 · **Host:** Apple M5 Pro (20 GPU / 64 GB)  
**Artifacts:** `bench/results/bottleneck_e4b.log` (`GEMMA_METAL_TRACE=sync`), quiet product run `bottleneck_e4b_quiet.txt`  
**Quiet decode (TRACE/INFER off):** **~23.9 tok/s** · TTFT ~142 ms  
**vs Phase-0:** mlx-lm **~75.7** · Ollama **~55.8** · gate ≥48–60 · MTP e2e **~10–12** @ 75% accept

Honest lane is **GPU-bound**: host encode of ~779 dispatches is ~2–3 ms when not sync-tracing; nearly all wall time is waiting on the Metal CB at the token boundary. Simdgroup Q4 GEMV (`gemv_q4_mlx_simd`, bfloat2 sb + qdot) is the default path (BlockedBn / Interleaved4 opt-in).

**Occupancy knobs (2026-07-20 quiet A/B):** shipping **SIMD_ROWS=4 × SG=2** already matches MLX `qmv_fast`; simd kernels use **no TG mem**. Retuned `rows=2` → exactness FAIL + 0.99×; `SG=4` → exact PASS but **+1.3%** (noise). Prior `rows=8` slower. **Park** — no product default change. Artifact `bench/results/simd_occ_ab_e4b_20260720T021250Z.json`.

**Interleaved4 Hot (2026-07-20 quiet A/B):** `GEMV_INTERLEAVE=1` exactness **PASS** @ HAZARD=0 but **0.953×** vs row-major (17.86 vs 18.75 tok/s paired). Keep default **OFF**. Artifact `bench/results/gemv_interleave_ab_e4b_20260720T021529Z.json`.

**Lane pause:** local layout/occupancy/fusion/encode-once levers exhausted for tok/s; honest bottleneck remains **MLP GEMV bandwidth (~75%)**. Resume only with a new BW hypothesis (≫10% token-time), not another dispatch-glue A/B.

---

## ⚠️ CORRECTION (2026-07-13, isolated-kernel audit): GEMV is NOT the 4× — per-token overhead is

An upload-once Hot-bank microbench (`bench/results/kernel_roofline_finding.json`; 200 iters,
dispatch+sync only) shows the Q4 GEMV kernels — old one-thread-per-row **and** new simd —
already run at **62–100% of the ~273 GB/s peak** (e.g. old kernel: 31B attn 277 GB/s, mlp_down
213 GB/s; simd gains only 1.03–1.5×). The "~20% of peak / ~4× in GEMV" figure below came from
`gemv_quant_host`, which **re-uploads ~4.9 MB of weights every call** — it measured host→device
upload bandwidth, not kernel bandwidth.

Corrected roofline: at ~21.5 tok/s (46.5 ms/tok), weight streaming at ~270 GB/s costs only
~10.6 ms — **~77% of the token is per-token overhead** (dispatch encode, barriers, bf16 casts,
argmax, 42-layer serialization over ~780 dispatches). mlx ~76 tok/s = same ~10.6 ms streaming
+ only ~2.6 ms overhead. **Remaining GEMV upside ≤ ~1.3×; the 3.5× gap is overhead.**
Effective levers, in order: (1) fewer dispatches/token (fusion → megakernel / persistent
decode); (2) block speculative verify (M=K forward amortizes overhead over K tokens — measured
on mlx: DFlash 31B ~12.7 → 27.8 tok/s with mlx 0.32 M=8 NAX GEMM; see
`bench/results/mlx032_nax_ab_31b.json`); (3) encode/GPU overlap.

The section below is retained for history but its BW-utilization numbers are superseded.

---

## Time breakdown (% of decode step)

Measured with `GEMMA_METAL_TRACE=sync` bucket flushes (embed → each layer → `lm_head` → `softcap_argmax`). Mid-step sync **slows absolute tok/s** (~15 tok/s during the sync run) but preserves relative GPU work share. Numbers below are averages over full `head=true` E4B decode tokens from that log (~65 ms/tok under sync). Quiet product ≈ **52 ms/tok** at 19.4 tok/s — same mix, less sync tax.

| Bucket | Measured / derived | ≈ % of step | Notes |
|--------|-------------------|-------------|--------|
| **MLP GEMVs** (gate / up / down × 42) | ~51 ms (est.) | **~78%** | Byte-split of measured `layer` (~63 ms); ~2.07 GB Q4 traffic/tok |
| **Attn Q/K/V/O GEMVs** | ~9 ms (est.) | **~14%** | Same split; producers only for K/V; ~0.37 GB |
| FA + RMS / RoPE / residuals | ~3 ms (est.) | **~5%** | Decode FA is short-KV; microbench FA SWA ~20 µs |
| **`lm_head` GEMV** `[262144×2560]` | **~1.9 ms** | **~3%** | Measured; ~401 MiB; near BW for that mat alone |
| softcap + argmax | **~0.26 ms** | **~0.4%** | Microbench ~173 µs |
| embed | **~0.16 ms** | **~0.2%** | Hot quant lookup |
| Host encode (quiet, no mid-sync) | ~2 ms | ~3–4% of quiet step | TRACE=1 baseline: ~695–779 disp, 0 mid-commits when `MID_COMMIT=∞` |

**Roofline:** ~2.86 GB Q4 weight traffic / token → at 19.4 tok/s ≈ **55 GB/s** ≈ **~20%** of ~273 GB/s unified peak (prior notes ~8–16% depending on traffic accounting and GPU contention). mlx ~76 tok/s on the same traffic implies **~60–80%** of peak if BW-bound — we leave **~4×** on the table in GEMV efficiency, not in FA/argmax/CPU sync.

**Top 3 bottlenecks (product quiet step):**

1. **MLP Q4 GEMVs — ~75–78%** of step  
2. **Attention projection GEMVs — ~14%**  
3. **`lm_head` tall GEMV — ~3%** (still large abs. work; ~near-roofline already)

CPU STALL sync on the **honest** decode bench path is one wait per N packed steps (`bench_decode_tok_s`); API `step()` / generate still stall once per token for readback — secondary vs GEMV BW.

---

## Why slower than MLX (~4×)

Code-level gaps vs `mlx-lm` / Metal matmul:

1. **Naive / still-short Q4 GEMV** (`kernels/gemv_q4_mlx.metal`): default is now
   `gemv_q4_mlx_simd` (32-lane K-split, 8 rows/SG, coalesced weight uints) — better than
   1-thread/row, but still far from MLX qmv bandwidth (E4B ~21 vs ~76 tok/s). BlockedBn
   coop + fused gate∥up→gelu exist but are opt-in. Tall/wide mats under-occupy vs MLX.

2. **Dispatch fragmentation:** ~**780 launches / token** (per-proj GEMV + FA + norms + barriers) vs fused MLX graphs. Packing into one CB helps host (~2 ms), not kernel occupancy. Mandatory `barrier` before `mlp_gelu` after gate∥up (and many optional hazard barriers) further serialize the DAG.

3. **No fused MLP / QKV:** separate `gemv_gate`, `gemv_up`, `mlp_gelu`, `gemv_down` (and Q/K/V) with full weight reloads of `x`/activations each time. Down-proj TG cache is **cols=10240 → 40 KiB** TG mem — expensive vs MLX fused epilogues.

4. **Decode still M=1 hand GEMV** by design (see `docs/architecture.md`) — correct for bandwidth vs TensorOps 32×32, but **our** GEMV is far from MLX’s tuned INT4 path. Square microbench `gemv_q4 [2560×2560]` ≈ **350 µs** is informative; scaling across 42×3 MLP mats dominates.

5. **Secondary (not the 4×):** PLE Hot still skipped; prior host-KV densify is gone on the happy path; soft mid-commit correctly disabled (premature commit previously ~10 tok/s). Host sync is small once encode is packed.

---

## Why MTP is slower than baseline (~10 vs ~15–19 tok/s)

Accept rate is fine (**75%**); the schedule is not speculative throughput.

| Mechanism | Code | Effect |
|-----------|------|--------|
| **Full backbone verify every draft token** | `GpuDecodeSession::generate_mtp_smoke` loops `self.step(tok)?` per draft | Each accepted token still costs a **full 42-layer** decode — no batch/tree verify; work ≥ greedy |
| Early-reject only stops **after** mismatch | same loop | Saves *rejected tail* only; accepted prefix still paid in full |
| **Host KV bridge every round** | `sync_mtp_cross_kv`: `synchronize` + `read_f32` shared sliding/global → densify into MTP | Extra stall + bandwidth vs GPU-resident draft |
| **Host draft** | `normed.read_f32()` + `MtpSession::draft_from_hidden` | CPU assistant forward on critical path |
| Draft overhead with no free backbone tokens | e2e measure | **~10 tok/s** < quiet baseline |

Until verify is a **single parallel/batch forward** over K draft tokens (classic speculative decoding) and draft+KV stay device-side, MTP cannot beat the backbone.

---

## Ranked next fixes (expected impact)

| Rank | Fix | Expected impact | Effort |
|------|-----|-----------------|--------|
| **1** | **Fewer dispatches / token** — fuse phases toward megakernel / persistent decode (ALU peel / bfloat2 sb landed; tok/s flat) | **~2–3×** if overhead falls toward mlx's ~2.6 ms | High |
| **2** | **Speculative verify** — one batched backbone step for K drafts; GPU draft + shared KV | MTP **≥1.5–1.7×** when accept stays ~70%+ | Med |
| **3** | **Fuse MLP** (gate∥up→gelu→down) deepen + QKV; cut launches + re-reads of `x` | **~1.2–1.5×** on top | Med–high |
| 4 | Trim RAW barriers / overlap where hazards allow; keep single CB / token | Small **(~5–10%)** unless fused | Low |
| 5 | Prefill TensorOps GEMM (already directed); does not move decode tok/s much | TTFT mainly | Med |

**Do not:** soft mid-commit until overlap is proven (known serialize to ~10 tok/s); claim gate clearance until GEMV BW utilization is re-measured near Phase-0 mlx.

---

## Re-measure

```bash
cd gemma-metal
# Quiet product
GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 cargo run --release --bin bench -- --e4b

# Sync bucket profile (relative %; slower absolute)
GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 GEMMA_METAL_TRACE=sync \
  cargo run --release --bin bench -- --e4b 2>bench/results/bottleneck_e4b.log
```

Instrumentation note: `TraceSession::flush_gpu_bucket` attributes GPU wait under `TRACE=sync` after each layer / `lm_head` / softcap. `TRACE=1` without sync still mostly measures host enqueue (~2 ms) + final wait.
