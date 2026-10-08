# Architecture — Gemma 4 Metal inference

Custom decode/prefill on Metal 4. Training graph / tape stays in `metal-native`.

## E4B vs 31B (do not parameterize-and-pray)

| Spec | E4B | 31B |
|------|-----|-----|
| Params | ~4.5B effective / ~8B w/ PLE | ~30.7B dense |
| hidden / MLP | 2560 / 10240 | 5376 / 21504 |
| Layers | 42 (35 SWA + 7 global) | 60 (50 SWA + 10 global) |
| Window / ctx | 512 / 128K | 1024 / 256K |
| Local Q/KV @ d | 8/2 @ **256** | 32/16 @ **256** |
| Global Q/KV @ d | 8/2 @ **512** | 32/**4** @ **512** |
| `attention_k_eq_v` | **false** | **true** (global: no `v_proj`) |
| `num_kv_shared_layers` | **18** → `first_kv_shared=24` | **0** |
| PLE | **yes** (~2.82B) | **no** |
| MTP drafter | ~77M, clustered LM head | ~500M, dense vocab |

Presets: `Gemma4TextConfig::e4b_preset()` / `b31_preset()` in `src/config.rs`.

**Critical kernel breaks vs Llama / arch_02 FA:** hetero head dims 256/512 (arch_02 FA caps D≤64); attention scale **1.0** after QK-Norm; p-RoPE on global (`partial_rotary_factor=0.25`); `gelu_pytorch_tanh` (not SiLU); logit softcap 30; E4B PLE + KV-share; 31B global K=V.

## Dual FlashAttention

| Path | Head dim | Use |
|------|----------|-----|
| `flash_attn_swa_h256` | 256 | Sliding-window layers |
| `flash_attn_global_h512` | 512 | Full-attention layers (+ p-RoPE) |

Both paths are **real FA-2 tiled kernels** (GPU-tested vs CPU causal refs; see [`dev.md`](dev.md)). They take dense contiguous `[B,T,H,D]` / `[B,T,Hkv,D]` — **not** yet bound to `KvLayout` sliding rings / shared producer-consumer buffers. WWDC single-tile TensorOps flash remains probe-only; do not use it for decode.

## Prefill GEMM vs decode GEMV

| Phase | Op | Where |
|-------|-----|-------|
| Prefill (M≫1) | TensorOps / simdgroup **GEMM** (± quant MTLTensor later) | `tessl` |
| Decode (M=1) | Dedicated **GEMV + inline dequant** | `gemma-metal` kernels |

Do **not** drive decode through M=1 TensorOps 32×32 tiles — bandwidth-bound occupancy waste (BaseRT lesson). Prefill may try native quant MTLTensor (WWDC26-330); decode stays hand GEMV until proven otherwise.

## PLE — Metal 4 GiB buffer split

HF packs `embed_tokens_per_layer` as `[vocab, L * ple_dim]`. E4B bf16 packed ≈ **5.6 GiB** → exceeds Metal single-buffer limit (**4 GiB**).

**Mandatory:** split into **per-layer** Hot banks `[vocab, ple_dim]` at load (`src/ple.rs`). Even if a future pack fits, load path always splits for uniformity. Validate with `PleBanks::validate_metal_limit()` / `HostWeightBanks::validate_metal_limits()`.

31B: `hidden_size_per_layer_input == 0` → no PLE.

## KV layout

Implemented in `src/kv.rs` as `KvLayout`:

```
first_kv_shared = num_hidden_layers - num_kv_shared_layers
E4B: 42 - 18 = 24
```

| Layer range | Role |
|-------------|------|
| `[0, first_kv_shared)` | **Producers** — own K/V weights + cache slots |
| `[first_kv_shared, L)` | **Consumers** — no K/V weights; read shared state |

| Slot kind | Shape idea |
|-----------|------------|
| `SlidingRing` | Width = `sliding_window`, head_dim = local (256) |
| `GlobalFull` | Seq = `max_seq`, head_dim = global (512) |
| `SharedKvId::SlidingFull` | **Full-length** shared sliding source for consumers — **not** truncated SWA ring |
| `SharedKvId::GlobalFull` | Full-length shared global |

E4B producers among first 24 layers: **20** sliding rings + **4** global (pattern 5 SWA + 1 full). 31B: all producers; no shared buffers.

Scratch / sizing: `ScratchPlan` (`src/scratch.rs`) — zero alloc after load (CPU plan).
Phase 4 `GpuDecodeSession` keeps KV on GPU (store + live FA bind; densify only on ring wrap) with packed async Metal 4 encode.

## MTP (Phase 5 — synthetic E2E)

`MtpSession` + `GpuDecodeSession::generate_mtp_smoke` run draft→verify on the decode
path (`src/mtp.rs`: presets, cross-KV, activation bridge, E4B centroids head, adaptive
draft, greedy verify). No real assistant weights on disk; no measured MTP tok/s yet.
Primary Mac speed lever after honest INT4 decode.

## Encode substrate

`tessl`: Metal 4-only CB/encoder, Hot/Cold residency, packed binder, const arena, GEMM, the promoted NN kernel library (RMSNorm, MLP activations, flash attention, quantized GEMV/GEMM, KV stores, sampling), MTLTensor prep hooks. `gemma-metal/build.rs` compiles only the Gemma-specific overlay (PLE, persistent interpreter).

```mermaid
flowchart LR
  rt[tessl encode plus GEMM]
  banks[Q4 Hot banks plus PLE split]
  kv[Dual KV rings]
  gemv[GEMV decode]
  fa[Dual FA 256 / 512]
  rt --> banks
  banks --> gemv
  kv --> fa
  gemv --> graph[E4B forward]
  fa --> graph
```
