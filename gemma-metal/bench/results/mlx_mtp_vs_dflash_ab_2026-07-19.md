# MLX MTP vs DFlash A/B (2026-07-19, M5 Pro)

Prompt: `Explain speculative decoding in 3 sentences.` · max_tokens=64 · block=5

| Path | tok/s | notes |
|------|------:|-------|
| MLX greedy 31B Q4 | **13.26** | `mlx_lm` 0.31.3 + PR#1276 tree |
| MLX DFlash block=5 | **23.59** | `dflash_fast_31b.py` / z-lab draft Q4g64 |
| MLX MTP (assistant) | **blocked** | see below |

## MTP status

- Downloaded `mlx-community/gemma-4-31B-it-assistant-bf16` (snapshot `28e92270…`).
- Installed mlx-lm PR [#1276](https://github.com/ml-explore/mlx-lm/pull/1276) so `gemma4_assistant` **loads**.
- `stream_generate(..., draft_model=assistant)` fails:

```
NotImplementedError: Gemma 4 assistant has no KV cache of its own;
pass the target's per-layer-type (K, V) tensors in `shared_kv_states` to `__call__`.
```

PR#1276 lands the model class only; speculative-decode wiring (`shared_kv_states`) is a follow-up. Stock `mlx_lm.generate --draft-model` cannot run 31B MTP on M5 today.

## D14

**Not overturned.** DFlash wins the measurable A/B (23.59 vs 13.26 tok/s). MTP could not be measured; no native MTP port.

## Artifacts

- `bench/results/mlx_ab_final.txt`
- `bench/results/mlx_mtp_ab_attempt.txt`
- `bench/results/download_mtp_31b_assistant.log` (bf16 assistant via `token=False`)
