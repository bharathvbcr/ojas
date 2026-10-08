# DFlash block-verify port plan (gemma-metal native MTP v2)

**Date:** 2026-07-14 · Status: **E4B greedy green**; **GEMM verify dual-norm landing**; 31B accept still open

## Progress (2026-07-14)

| Step | Status | Notes |
|---|---|---|
| 1. `step_verify` | **Done** | M× decode GEMV fallback; Hot `M>1` → Q4 GEMM + FA(Tq=M) when `cols>256`. |
| 1b. Dual seed vs argmax | **Done** | `seed_tok` / `argmax_tok` / verify slots. |
| 1c. GEMM ≡ `step_inner` | **Done** | Dual-norm + `layer_scalar` ×M. Diag: GEMM≡M×GEMV≡seq under always-on (`GEMMA_METAL_31B_VERIFY_DIAG=1`). |
| 2. Hidden-capture + fc/hidden_norm | **Done** | Device `copy_f32` / `copy_f32_range` for M=1 and GEMM verify (no mid-layer host sync). Conditioner FC deferred post-softcap. Draft/fc re-quant **plain Q4 g64**. |
| 3. Native draft (D=128 attn) | **Done** | `DFlashGpuDraft` Hot Q4 + `flash_attn_swa_h128` + `mlp_silu`. |
| 4. GPU-draft generate loop | **Done (bring-up)** | `generate_with_dflash` → GPU draft. |
| 4b. MLX scale alignment | **Landed** | Draft FA **`1/√d`**; target FA **1.0**; **`embed_scale=√H`**. |
| 4c. Greedy finite logits | **Restored** | E4B + 31B finite; unique≫1. |
| 4d. Metal gelu + throughput | **Restored** | `precise::tanh` + clamps; host gelu removed; fuse MLP + hazard default **ON**. |
| 4e. 31B D-Flash remeasure | **Honest baseline** | Pre-dual-norm: mean_accept≈0, exact FAIL. Rebench after 1c. |
| 5. Tune block size | Pending | Need accept ≫0 first. |
| 6. True M>1 verify | **Wired (Hot)** | Live `kernels/gemm_q4_mlx.metal` + act scratch ×`VERIFY_MAX_M`. Mini stays M×GEMV (`cols>256` gate). `.wip` twins are historical leftovers. |

**Smoke:**
```bash
cd gemma-metal
CARGO_TARGET_DIR=target GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 cargo run --release --bin diag_tok -- e4b
CARGO_TARGET_DIR=target GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 cargo run --release --bin bench -- --e4b
CARGO_TARGET_DIR=target GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 cargo run --release --bin bench -- --dflash-31b
```

## Greedy health (do not regress)

| Setting | Result |
|---|---|
| Metal gelu | **`precise::tanh`** + clamp inner ±10 + clamp x ±20 (was `air.fast_tanh` NaNs) |
| `diag_tok e4b` | finite logits, unique≫1, wall ~19–20 tok/s incl. prefill |
| `bench --e4b` quiet | **~22.8–23.9 tok/s** (16 steps) |
| 31B greedy (short pad) | **~5.5–6.8 tok/s**, unique≫1, finite (not all-0) |

**Do not revert:** draft FA `1/√d`, target FA `1.0`, `embed_scale=√H`, gelu `precise::tanh`, GEMM dual-norm/`layer_scalar` parity with `step_inner`, plain-Q4 draft GEMV on f32 (no bf16×plain-Q4 poison).

## Hazard / barrier lane notes

| Lane | Barriers | When |
|---|---|---|
| Product / 31B exactness | hazard skip-auto **ON** (`METAL_RUNTIME_HAZARD_BARRIERS` default) | Never force always-on on HF 31B capture (collapsed streams historically) |
| Mini exactness | always-on Dispatch barriers | Synthetic mini only; restores ambient hazard after lane |
| Shared GPU draft↔target | draft `synchronize` before target reuses CB | Documented invariant — keep |

lm_head→softcap RAW must stay edged (skip-auto previously dropped it → exactness fail).

## Honest 31B D-Flash (healthy target) — pre dual-norm fix

Artifact: `bench/results/run_dflash_parity_gates_1784001103.json`. Prompt `[2,105,4368,1246]`, max_new=24.

| Metric | gemma-metal | MLX golden |
|---|---|---|
| Greedy decode | **~5.8–6.8 tok/s** | — |
| DFlash best | **~1.17 tok/s** @ bs=3/5 | **~28–37 tok/s** @ bs=5 |
| mean_accept @ bs=5 | **≈0** | **≈3.0** |
| Exact vs capture+cond greedy | **FAIL** (GEMM ≠ dual-norm) | **PASS** |

### Stream / accept bugs

| Issue | Fix / status |
|---|---|
| Conditioner FC before softcap sync | **Fixed** — project FC after argmax readback |
| Always-on barriers for HF 31B | **Fixed** — always-on + MASK-steer **synthetic mini only** |
| `step_verify_gemm` legacy residual / dropped `layer_scalar` | **Landing** — dual-norm + scale ×M |
| Draft Q4 g32 vs MLX g64 | **Open** (second-order after verify faithful) |
| mean_accept≈0 on real draft | **Open** — verify graph fixed; draft proposals still ≪ MLX (dump vs `golden_intermediates_31b.json`) |
| Always-on 31B exact PASS | **Vacuous** if greedy unique≤4 — do not claim; keep hazard for honest 31B |

## Mini gates (synthetic)

| Item | Status |
|---|---|
| Exactness under always-on | **PASS** |
| mean_accept | Full via **MASK→anchor steer** (mini only; **not** for HF drafts) |
| tok/s | DFlash can meet/beat hazard greedy when steered; M×GEMV verify |

## Scale contract (MLX-aligned)

| Path | Scale |
|---|---|
| Target Gemma FA (after QK-norm) | **1.0** |
| DFlash / Qwen3 draft FA | **`head_dim**-0.5`** (~0.0884 @ 128) |
| Target + draft embed | **`√hidden`** (31B ≈ 73.32; E4B ≈ 50.60) |

## Order of work (next)

1. Quiet rebench 31B after GEMM dual-norm — exactness @ accept=0 + mean_accept.
2. If exact PASS and accept still ~0: dump block-1 vs `golden_intermediates_31b.json`.
3. Draft+fc quant parity with MLX (Q4Mlx g64) if proposals diverge.
4. Device-side capture staging (drop mid-verify host sync); retune block size; chase ≥ greedy / ≥25.

See [`audit_deep_2026-07-14.md`](audit_deep_2026-07-14.md).
