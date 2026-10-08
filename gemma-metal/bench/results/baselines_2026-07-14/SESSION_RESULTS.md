# Inference speed optimization — session results (2026-07-14)

Machine was noisy (Cursor helpers; load ~4–5). Quiet historical refs:
E4B **23.92**, 31B **6.83**. Session floor ~4.5 on 31B.

## Native track

| Step | Status | E4B | 31B decode | Notes |
|------|--------|-----|------------|-------|
| Preflight baseline | done | 23.67 | 4.55 | Ownership quiet; mid-mmin hot files settled |
| 1 dual-norm fuse | **landed** | 23.76 | ~4.56 | `rms_norm_residual_add_f32`; `GEMMA_METAL_FUSE_DUAL_NORM=0` rollback |
| 2a FromArgmax | **landed** | — | — | `generate()` chained; `bench_decode_tok_s` already used FromArgmax |
| 2b mid-commit | **flat** | 23.9–24.0 | 4.54→4.59 | Sweep 0/128/256; leave default **0** |
| 3 bf16 cast fuse | **landed** | **24.90** | **5.41** | Producers emit bf16; A/B off: 24.45 / 5.57 (noise). HAZARD=0 bit-match vs casts |
| 4 K+V + layer_scalar | **landed** | 23.99 | **5.30** | `kv_store_timestep_pair`; scale folded into post-ff residual |
| 5 one-pass argmax | **landed** | 23.99 | **5.26** | `softcap_argmax_one_pass`; `GEMMA_METAL_ARGMAX_MULTIPASS=1` rollback |

Standing gates: E4B ≥23 ✓ · 31B greedy finite ✓ · E4B path untouched ✓

### Native #3 details
- Metal: `rms_norm_bf16`, FA `out_bf16`, `mlp_gelu_tanh_bf16`, gate_up `mid_as_bf16`
- Wire: skip `prepare_act_bf16` when fused; mid bf16 aliases `self.mid` (avoids clobbering act scratch x)
- Rollback: `GEMMA_METAL_FUSE_BF16=0` (debug slices: `rms` / `fa` / `mlp`)
- HAZARD=0: fuse-on ≡ fuse-off token streams (E4B). Default hazard remains nondeterministic (pre-existing).

### Golden vs `golden_tokens_31b.json`
`cargo run --release --bin golden_parity -- greet 16` with `METAL_RUNTIME_HAZARD_BARRIERS=0`:
**match_prefix=0/16** — native collapsed vs MLX greedy (open port gap; not a #3 regression).

## MLX track

| Step | Status | Notes |
|------|--------|-------|
| M1 prompt-cache | **landed** | sticky cache; `cached_tokens` grows across turns |
| M2 SSE overlap | **landed** | writer thread |
| 3-turn TTFT | **measured** | server TTFT≈370/366/363 ms; cached=0/23/45; decode≈36 tok/s. Short-turn prefill still ~15–23 toks so TTFT flat |

## A/B env flags
- `GEMMA_METAL_FUSE_DUAL_NORM=0`
- `GEMMA_METAL_FUSE_BF16=0|rms|fa|mlp`
- `METAL_RUNTIME_MID_COMMIT=0|128|256` (default 0)
- `GEMMA_METAL_ARGMAX_MULTIPASS=1`
- `METAL_RUNTIME_HAZARD_BARRIERS=0` — golden always-on Device barriers
