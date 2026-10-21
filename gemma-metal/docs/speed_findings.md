# Inference-speed findings & artifacts (2026-07-13, MLX-side audit)

Index of the MLX-path speed work that runs alongside the native `gemma-metal` engine.
Companion to [dflash_port.md](dflash_port.md) (native port, owned by the engine work),
[bottleneck.md](bottleneck.md) (roofline correction), [gates.md](gates.md).

## Headline

**Gemma-4-31B on M5 Pro: plain 12.5 → DFlash ~31 tok/s median (2.5×)**, exact (lossless) vs greedy.
Honest range across 8 prompt types: **19–36 tok/s** — structured output (code/math/json/
factual/translate) **2.4–3.1×** (~30–36 tok/s), open-ended creative prose weakest at **1.6×**
(~19 tok/s, draft accept ~2.2 — unpredictable text is hard to draft). **Every prompt type
still beats plain and clears the ≥15 gate; all but pure prose clear the ≥25 stretch gate.**
Stack: DFlash block spec-decode × 4-bit draft × mlx 0.32 (M5 neural accelerators) × block=5.

> **Audit 2026-10-20 (evidence labels):** the ~31 median / 2.5× / 19–36 range come from
> `diversity_sweep_result.json`: 8 prompts × 1 run each, max_tokens 160, `dflash_median` 31.7,
> `speedup_median` 2.49, `dflash_min` 18.94, `dflash_max` 36.12; that file has no exactness
> field. "Exact / lossless" is supported by a different artifact, `golden_tokens_31b.json`
> (5 prompts, max_tokens 64, block 5, `dflash_exact_vs_greedy: true`, per-prompt
> `dflash_matches_greedy: true`), not by the 8 sweep prompts. The "37" figure is a single-prompt
> peak (`block_finesweep32_result.json`, `best_median` 37.17 at block 5, `all` = 3 repeats of one
> prompt), not a prompt-type result (sweep max is 36.12, json). `mlx032_nax_ab_31b.json` (1.49×,
> 27.77 vs 18.64) is one prompt, block 8, max_tokens 128.
MLX reference arm: `bench/dflash_fast_31b.py` (the comparator native DFlash is measured against, not a shipped CLI). Local serving: `bench/serve_dflash.py` (OpenAI-compatible :8788; Python under a recorded waiver until native DFlash reaches parity, see its docstring).
Runtime: `~/.venvs/dflash32` (mlx 0.32.0 + dflash; dflash's `[mlx]` extra pins 0.31.2 — install separately).

## Levers, measured (all interleaved / drift-checked under GPU contention)

| Lever | Effect | Note |
|---|---|---|
| DFlash spec-decode (`z-lab/gemma-4-31B-it-DFlash`) | 12.7 → ~17.6 | block verify amortizes per-token overhead |
| 4-bit draft (`nn.quantize` g64) | +6%, −2.2 GB | exact verify ⇒ zero quality impact |
| **mlx 0.31.2 → 0.32.0** | **1.49×** | M=block verify GEMM 1.5–2× on M5 NAX (macOS 26.2+, auto); M=1 decode unchanged |
| block_size (fine-swept) | **5 optimal** | non-monotonic: 5→37, 8→28, ≥12→~17 cliff; re-tune per mlx/kernel change |
| wired memory (`set_wired_limit`) | none | weights already resident |
| lower-bit/mxfp4 target | not pursued | decode is M=1 bandwidth-bound (same bytes); 3-bit = quality burn (against doctrine) |

## Key corrections to prior project assumptions

1. **GEMV was never the 4× bottleneck.** Old & new Q4 GEMV kernels already run at 62–100%
   of ~273 GB/s. The "~6–20% / 4×" figure came from a microbench (`gemv_quant_host`) that
   re-uploaded weights each call. ~77% of a decode token is per-token overhead. → the real
   levers are dispatch-count reduction and block-verify, not faster GEMV. (`kernel_roofline_finding.json`)
   > **Audit 2026-10-20:** the "~77% per-token overhead" is a residual against an isolated,
   > contended microbench rate, not a measured host/dispatch cost, and the dispatch-count
   > lever was later measured flat on E4B (−90 dispatches: 23.58 → 23.26 tok/s; host encode
   > 170 µs; layer stack ~90% GPU, MLP ~75%). See the audit note in [bottleneck.md](bottleneck.md),
   > `bench/results/decode_profile_e4b_20260720T015429Z.json`. Block-verify (DFlash) remains
   > measured on the MLX path only.
2. **M5 NAX helps decode indirectly.** Apple's research says NAX is prefill-only; but
   block-verify is a mini-prefill (M=block), so spec-decode converts a prefill-only HW
   feature into a decode win.

## Artifacts (`bench/results/`)

| File | What |
|---|---|
| `mlx032_nax_ab_31b.json` | 4-arm A/B: 0.31.2 vs 0.32.0 vs wired |
| `block_finesweep32_result.json` | block-size fine sweep on 0.32 |
| `dflash_q4draft_interleaved_31b.json` | 4-bit vs bf16 draft (interleaved) |
| `kernel_roofline_finding.json` | isolated GEMV kernel BW (roofline correction) |
| `golden_tokens_31b.json` | **greedy==DFlash golden token streams for native-port parity tests** |
| `diversity_sweep_result.json` | tok/s across prompt types (honest range) |

## Native port (the remaining big lever)

The custom engine's path to ≥25 tok/s is the DFlash block-verify port — design + progress in
[dflash_port.md](dflash_port.md). Parity target: reproduce `golden_tokens_31b.json` exactly.
