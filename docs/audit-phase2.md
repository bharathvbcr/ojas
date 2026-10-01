# Phase 2 audit

Recorded 2026-10-01. Read-only. No `devmap build`, no cargo tests. Dead-code rows are listed only when confidence is high (about 0.9) and a second `devmap search` plus a source read agree. Rust call resolution is about a quarter of sites, and every `devmap dead` answer sets `walk_incomplete`, so an empty high-confidence list is a lower bound.

## 1. Index freshness

| Root | Store | Fresh | Generation | Rust gross resolution | `devmap dead` |
| --- | --- | --- | --- | --- | --- |
| ojas | `.devcouncil/codeintel/devmap.sqlite` | yes (`is_fresh`, `source_freshness`, `pending_count` 0) | 305 | 273 / 1000 (3044 resolved, 8096 unresolved, 2244 explained) | 8 rows, all confidence 0.40; `truncated` false; `walk_incomplete` 6259 of 9054 |
| tessl | `.devmap/codeintel/devmap.sqlite` | no | 779 | 236 / 1000 (7676 resolved, 24749 unresolved, 8348 explained) | 7 rows; `truncated` false; `walk_incomplete` 19295 of 36326 |
| gusset | `.devcouncil/codeintel/devmap.sqlite` | yes (`is_fresh`, `source_freshness`, `pending_count` 0) | 804 | 219 / 1000 (1471 resolved, 5243 unresolved, 2265 explained) | 4 rows, all confidence 0.40; `truncated` false; `walk_incomplete` 6131 of 16157 |

Ojas has an index. This audit is index-backed for liveness and source-checked for the gaps below. Schema is current (version 23) on all three. `devmap dead` itself reports whole-tree freshness as unchecked; the freshness column is from `devmap status`.

Tessl is the stale index. `degraded_reason` is “source tree differs from the indexed generation”. The delta is two changed paths: `src/nn.rs` and `tests/flash_attn_rows_h64.rs`. `query_ready` is still true. Answers about `flash_attn_rows` and that test file are from the previous generation.

## 2. Dead symbols (tessl and gusset)

No high-confidence symbol survives a second search as dead code on a training or inference path. Test helpers are omitted. The 0.40 rows are resolver misses (`exemption_reason`: an unresolved call site names the symbol), not dead findings, and `walk_incomplete` means further callers may be missing.

**tessl.** One row at confidence 0.90: `qwen35_residual_add_f32` in `kernels/qwen35_mlp.metal`. `devmap search` returned that definition only (`total` 1, `truncated` false). `devmap explore` returned 0 callers, with `walk_incomplete` 19295 of 36326 and no indexed traversal start for the Metal kernel (indexed as C++). A source read shows the host launches it by name:

- `src/qwen35.rs:1912` — `rt.pipeline("qwen35_residual_add_f32")`
- `tests/qwen35_kernels.rs:1840` — the same entry name in the kernel contract test

That kernel is the residual add on the Qwen3.5 path. It is live. The other six tessl rows are confidence 0.40 (`Acts.new`, `GdnWrapped.inputs`, `TesslQwen35.no_pending`, `Window.check`, `LayerGrads.parts`, and a self-call cluster on `TesslQwen35.bank`). `LayerGrads.parts` sits on the training file `src/qwen35_train.rs` and is not counted as dead.

**gusset.** Zero rows above confidence 0.40. The four names are pool methods with unresolved callers: `Handle.adopt_output`, `Handle.complete`, `JobQueue.close`, `JobQueue.push`. They stay off the dead list.

## 3. Product gaps

Each label is from a source read in this session.

### QK-norm stride `2 * D` (tessl qwen35) — verified-by-reading-source

`kernels/qwen35_attn.metal:110` reads query head `j` at `q_off + j * 2 * D`. The kernel header (`kernels/qwen35_attn.metal:4-5` and `:131-133`) states the fused projection is `[W_q | W_k | W_v]` with `W_q` `2 * Hq * D` wide: D query, then D gate, per head. Key and value use stride `D` (`:118`, `:121`). There is no stride argument on `qk_norm_rope_unit` (`:79-90`). The backward writes the same columns: `kernels/qwen35_bwd.metal:544` uses `q_off + j * 2 * D` and says the gate columns belong to `qwen35_attn_gate_bwd_f32` (`:485-487`).

A packed query of width `D` cannot go through this entry. `ojas-metal/src/gpu.rs:19-24` records that and sets `QK_ROPE_SKIP`. `tiny_train_step` does not call `attn_qk_norm_rope` (`ojas-metal/src/gpu.rs:329-331`, `:409`).

### Flash attention backward head dimensions — verified-by-reading-source

Two separate limits.

**tessl training backward is head dim 256 only.** `kernels/qwen35_attn_bwd.metal:2` and `:31` set `ATTN_BWD_D = 256`. The only launches are `qwen35_attn_bwd_dq_h256_q32_k32_sg4`, `qwen35_attn_bwd_dk_h256_q32_k32_sg4`, and `qwen35_attn_bwd_dv_h256_q32_k32_sg4` (`:267-269`, dispatched from `src/attn_train.rs:298-300`). `ATTN_TRAIN_HEAD_DIM` is `PREFIX_ATTN_HEAD_DIM` (`src/attn_train.rs:32-33`), which is `256` (`src/qwen35.rs:2165`). Forward `flash_attn_rows` has a separate head-dim 64 entry (`src/nn.rs:1684-1687`). The training backward does not.

**metal-native backward clamps to 64.** `Rust_MLKit/arch_02_value_resid/metal-native/kernels/flash_attn_bwd.metal:28` sets `d_lim = min(D, 64u)` inside `flash_attn_bwd_delta_f32` and loops only to `d_lim` (`:31-32`). The same clamp is repeated at `:65`, `:171`, `:250`, `:368`, and `:422` in the dQ and dK/dV kernels. Dimensions above 64 are dropped with no error.

### ojas-metal tiny step shape — verified-by-reading-source

`validate_tiny_shape` (`ojas-metal/src/gpu.rs:62-101`) accepts only `d_model == 64` and `n_head == 1` (`D_MODEL`, `N_HEAD` at `:29-31`), plus `head_dim == 64`, batch `1..=2`, seq `1..=16`, vocab `1..=128`. `tiny_train_step` calls that check first (`:347`). Attention is one head: `heads` and `heads_kv` are both `n_head`, and `head_dim` is passed to `nn::flash_attn_rows` (`:529-548`). Q, K, and V are the same residual, so `head_dim == d_model` (`:115`, `:506-508`).

### `TIKTOKEN` constant in ojas-data — verified-by-reading-source

`ojas-data/src/bpe.rs:20` is `pub const TIKTOKEN_GPT2_BYTE_IDENTITY: &str = "UNVERIFIED"`. The module comment (`:3-5`) says the fixture does not use tiktoken ranks and those ranks are not in the repository. The test `fixture_round_trips_and_is_not_a_tiktoken_claim` asserts the constant is still `"UNVERIFIED"` (`:242-243`).

### CUDA and HIP feature defaults — verified-by-reading-source

Both crates set `default = []`.

- `ojas-cuda/Cargo.toml:11-13`: feature `cuda` is optional (`dep:cudarc`). With it off, `CudaDevice::open` returns `DeviceError::NotCompiled { kind: Device::Cuda }` (`ojas-cuda/src/lib.rs:31-36`). `affine_f32` returns the same error (`:68-73`).
- `ojas-hip/Cargo.toml:11-13`: feature `hip` is optional (`dep:hip-runtime-sys`). With it off, `HipDevice::open` returns `DeviceError::NotCompiled { kind: Device::Hip }` (`ojas-hip/src/lib.rs:24-29`). `copy_roundtrip` returns the same error (`:45-49`).

## Files read for the gaps

`ojas-metal/src/gpu.rs`, `ojas-data/src/bpe.rs`, `ojas-cuda/Cargo.toml`, `ojas-cuda/src/lib.rs`, `ojas-hip/Cargo.toml`, `ojas-hip/src/lib.rs`, `tessl/kernels/qwen35_attn.metal`, `tessl/kernels/qwen35_bwd.metal`, `tessl/kernels/qwen35_attn_bwd.metal`, `tessl/kernels/qwen35_mlp.metal` (via search span), `tessl/src/qwen35.rs`, `tessl/src/attn_train.rs`, `tessl/src/nn.rs` (rows entry), `MLSystemsLab/Rust_MLKit/arch_02_value_resid/metal-native/kernels/flash_attn_bwd.metal`.

## Top issues

1. Tessl’s index is stale (generation 779; `src/nn.rs` and `tests/flash_attn_rows_h64.rs`).
2. Rust attribution is 219–273 per 1000 sites, and every dead-code walk is incomplete, so liveness lists are lower bounds.
3. The only high-confidence “dead” symbol, `qwen35_residual_add_f32`, is launched from `src/qwen35.rs:1912`.
4. Gusset has no confirmed dead symbol; `Handle.complete` and `JobQueue.push` are 0.40 resolver misses on the live pool.
5. Qwen3.5 QK-norm indexes each query head at stride `2 * D` and takes no stride argument (`kernels/qwen35_attn.metal:110`).
6. The ojas tiny step skips QK-norm and RoPE because of that stride (`QK_ROPE_SKIP`, `ojas-metal/src/gpu.rs:24`).
7. Tessl’s training attention backward exists only at head dim 256 (`ATTN_BWD_D`, `kernels/qwen35_attn_bwd.metal:31`).
8. metal-native `flash_attn_bwd_delta_f32` clamps the head with `min(D, 64u)` (`flash_attn_bwd.metal:28`).
9. `tiny_train_step` accepts only `d_model` 64 and `n_head` 1 (`ojas-metal/src/gpu.rs:73-80`).
10. `TIKTOKEN_GPT2_BYTE_IDENTITY` is `"UNVERIFIED"` (`ojas-data/src/bpe.rs:20`); CUDA and HIP default to off and return `NotCompiled`.
