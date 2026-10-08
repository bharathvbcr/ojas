# Lane A todo status — 2026-07-19

## evidence-hygiene — DONE
- F1: `bench` writes live `layers`/`hidden`/`vocab` + snapshot hash into `latest_{e4b,31b}_gemma_metal.json` (no hardcoded E4B labels on 31B).
- R8: local 31B snap `696d436c…` (cached 2026-07-13) matches golden vintage; do not re-pull mid-parity.
- Fresh E4B/31B/DFlash rebenches under `bench/results/`.

## mlx-mtp-ab — DONE
- Doc: `bench/results/mlx_mtp_vs_dflash_ab_2026-07-19.md`
- MLX greedy ~13.3 tok/s; DFlash block=5 ~23.6; MTP blocked (`NotImplementedError` on shared KV).
- D14 not overturned.

## target-mlx-parity — DONE (with capture-on)
- Root cause #1: interleaved RoPE → NeoX proportional RoPE (`kernels/rms_qkv_rope.metal`, `forward.rs`).
- Root cause #2: 31B free-decode **requires hidden capture** (even `GEMMA_METAL_CAPTURE_NOP=1`); capture-off + always-on → 240017/236773. `GemmaGpu::new` no longer clobbers an explicit barrier choice.
- `target_next=531` on `[2,105,4368,1246]` (MLX agrees; second token 237076 is MLX-correct, not a native bug).
- **greet16 = 16/16** with capture-on (`bench/results/greet16_cap.txt`).
- golden_parity enables capture by default (`GEMMA_METAL_NO_CAPTURE=1` to A/B).

## draft-accept-parity — PARTIAL
- Host-dense dump: `h_ctx` absmean ≈ MLX; host_dense proposals ≈ MLX dense (one token off); GPU draft still drifts from MLX q4g64.
- Tried draft Q4Mlx g64 (was plain Q4); short-prompt mean_accept still ~1 (prompt mode-locks).
- **Greet prompt measure** (meaningful): bs=5 **mean_accept≈2.43**, exact PASS, greedy matches gold prefix (`rebench_dflash_31b_greet.log`). Target ≈3 not yet met; draft proposal fidelity remains the lever.
- 31B DFlash default: always-on + capture; `GEMMA_METAL_31B_HAZARD=1` for product skip-auto; measure prompt=greet unless `GEMMA_METAL_31B_SHORT_PROMPT=1`.
