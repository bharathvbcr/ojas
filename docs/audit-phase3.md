# Phase 3 audit

Recorded 2026-10-01. Read-only. No `devmap build`, no `cargo test`, no commit. Labels are from this session. `devmap dead` reports whole-tree freshness as unchecked; freshness below is from `devmap status`. Rust call resolution is about a quarter of sites on every tree, and every dead-code walk sets `walk_incomplete`, so a list with no high-confidence row is a lower bound.

## 1. Index freshness

| Root | Store | This session | Generation | Rust gross resolution | `devmap dead` |
| --- | --- | --- | --- | --- | --- |
| ojas | `.devcouncil/codeintel/devmap.sqlite` | stale on `gpu.rs` at the first sample and at generation 492; generation 496 then reported fresh | 479, then 492, then 496 | 237/1000 at gen 479 (3805 resolved, 12244 unresolved, 2750 explained); 236/1000 at gen 490 | 10 rows, all confidence 0.40; `truncated` false. First walk: 9664 of 13022. Later walk, same ten names: 9769 of 13140 |
| tessl | `.devmap/codeintel/devmap.sqlite` | status says fresh | 801 | 236/1000 (7709 resolved, 24841 unresolved, 8366 explained) | 7 rows; `truncated` false; walk 19371 of 36434 |
| gusset | `.devcouncil/codeintel/devmap.sqlite` | status says fresh | 804 | 219/1000 (1471 resolved, 5243 unresolved, 2265 explained) | 4 rows, all confidence 0.40; `truncated` false; walk 6131 of 16157 |

**ojas is behind the source.** The first `devmap status` was not fresh: `degraded_reason` “source tree differs from the indexed generation”, delta `ojas-metal/src/gpu.rs`, generation 479. A watcher then advanced the store (485 fresh, 489 stale on `README.md`, 490 fresh, 492). During that churn an MCP `devmap_search` returned `fts5: corruption found reading blob 10 from table "nodes_fts"`. A later CLI search and `PRAGMA integrity_check` returned `ok`. A status sample after the source lines below were read is generation 492, not fresh, delta `ojas-metal/src/gpu.rs` (mtime 2026-10-01 09:39:10). A later sample, still without a rebuild from this audit, is generation 496, `is_fresh` true, `delta` null. The citations are the source read, not the span the index stored while the file was moving.

**tessl status says fresh, which is a change from the stale generation 779 recorded in phase 2.** Two status samples this session are generation 801, `is_fresh` and `source_freshness` true, `delta` null. The dirty working tree (including `src/nn.rs` and `tests/flash_attn_rows_h64.rs`) has no file newer than the sqlite at 2026-10-01 09:11:11. This audit did not rebuild. A save after that timestamp would put the store behind; it had not done so at these samples.

**gusset is fresh at generation 804**, the same generation as phase 2. Dirty files are not newer than the sqlite at 2026-10-01 00:13:18.

## 2. Dead symbols that survived a second look

None. No confidence-0.90 symbol is dead after a second search plus a source read.

Ojas and gusset have no row above 0.40. Those rows carry `exemption_reason` that an unresolved call site names the symbol, so “nothing calls this” is a statement about the resolver. `walk_incomplete` means further callers may be missing. The one ojas cluster (`Parser.array`, `Parser.object`, `Parser.value` in `ojas-io/src/json.rs`) is the same 0.40 tier.

Tessl’s single 0.90 row is live. See section 4.

## 3. Open product gaps

### Attention backward — verified-by-reading-source

Two different facts. `docs/status.md` still says the tiny step has no attention backward at head dim 64. The source of `ojas-metal/src/gpu.rs` disagrees, and the index is behind that file.

**Tessl still has no head-dim-64 training backward.** `kernels/qwen35_attn_bwd.metal:31` sets `ATTN_BWD_D = 256`. The only backward entries are `qwen35_attn_bwd_dq_h256_q32_k32_sg4`, `qwen35_attn_bwd_dk_h256_q32_k32_sg4`, and `qwen35_attn_bwd_dv_h256_q32_k32_sg4` (`:267-269`), launched from `src/attn_train.rs:298-300`. `ATTN_TRAIN_HEAD_DIM` is `PREFIX_ATTN_HEAD_DIM` (`src/attn_train.rs:33`), which is `256` (`src/qwen35.rs:2271`). A search of `kernels/` and `src/` for an h64 backward name returned no match. Forward `flash_attn_rows` does have a head-dim-64 entry: `kernels/flash_attn_rows.metal:235` (`flash_attn_rows_h64_r8_g8`) and `src/nn.rs:1684-1685`.

**The ojas tiny step now backpropagates head dim 64, on a different kernel.** `tiny_train_step` calls `causal_attn_backward` (`ojas-metal/src/gpu.rs:549`) with `head_dim` from the tiny shape (`:543-548`). That function (`:734`) is exact-f32 GEMM plus `ojas_causal_softmax_bwd` (`:899`; kernel `ojas-metal/kernels/causal_attn_bwd.metal:52`). Limits are head dim 64, seq at most 16, at most two heads (`:699-724`, `HEAD_DIM` at `:25`, `MAX_SEQ` at `:20`). The comment at `:448-452` says `qwen35_attn_bwd_*_h256` is a `32 x 256` tile and is not instantiated at 64. `METAL_MAX_HEAD_DIM` is 64 (`ojas-core/src/backend.rs:30`). A `devmap search` for `causal_attn_backward` in ojas returned `total` 1, `truncated` false, span `[0, 0]`, while status still had `gpu.rs` dirty. The source lines above are the ones to use.

Whether that backward matches a reference was not re-run (no `cargo test`).

### 12-layer / 768 — verified-by-reading-source

`ojas-nn/src/lib.rs:1-4` says the crate is the nanolab-default GPT module and “does not implement a model yet.” The executing tiny step accepts only `d_model` 128 and `n_head` 2 (`ojas-metal/src/gpu.rs:23-24`, `:73-80`), one pre-norm block, vocab at most 128, seq at most 16 (`:20-21`, `:82-98`). The CPU torch fixture is batch 1, seq 4, width 16 (`ojas-cpu/tests/torch_ref.rs:59`). The 12-layer, width-768 diagram in `ojas-nn/README.md` is not an implementation. `docs/op-coverage.md` points at `nanolab/config.py` for that shape; this session did not open that file (**unverified**).

### CUDA and HIP not executed — verified-by-reading-source for the refusal; execution unverified

Both crates set `default = []` (`ojas-cuda/Cargo.toml:11-13`, `ojas-hip/Cargo.toml:11-13`). With the feature off, `CudaDevice::open` returns `DeviceError::NotCompiled { kind: Device::Cuda }` (`ojas-cuda/src/lib.rs:32-36`) and `affine_f32` returns the same (`:68-73`). `HipDevice::open` returns `DeviceError::NotCompiled { kind: Device::Hip }` (`ojas-hip/src/lib.rs:25-29`) and `copy_roundtrip` returns the same (`:45-49`). This session did not enable either feature and did not run a device. Kernel execution on NVIDIA or AMD is **unverified**.

### Tokenizer is twenty strings — verified-by-reading-source

`TIKTOKEN_GPT2_BYTE_IDENTITY` is `"verified-20-strings"` (`ojas-data/src/bpe.rs:32`). The module comment says that is twenty strings against tiktoken 0.12.0 `encode_ordinary`, not a million-line check (`:8-9`, `:26-31`). `twenty_gpt2_ids` is a 20-element array (`:870-892`), and the test asserts `rows.len() == 20` (`:941-945`). If neither rank-table directory exists, `verified_20_strings_match_tiktoken_encode_ordinary` returns without an assertion (`:910-912`).

### One torch step, not fifty — verified-by-reading-source

`ojas-cpu/tests/torch_ref.rs:1` and `:52` (`one_step_matches_torch_2_13_float32`) run one forward, one backward, and one AdamW update (`:137-156`). The cosine schedule’s horizon is 20 steps; the test reads the multiplier at step 8 (`:55-57`). There is no 50-step comparison in that file.

## 4. What the dead-code list got wrong

**`qwen35_residual_add_f32` at confidence 0.90 is live.** `devmap search` on tessl returned that definition only (`total` 1, `truncated` false, `kernels/qwen35_mlp.metal`). The host launches it by name:

- `src/qwen35.rs:2018` — `rt.pipeline("qwen35_residual_add_f32")`
- `tests/qwen35_kernels.rs:1840` — the same entry name

The kernel is the residual add. A Metal entry launched by string is invisible to the Rust call graph, which is why the row is high confidence and wrong. The other six tessl rows stay at 0.40 (`Acts.new`, `GdnWrapped.inputs`, `TesslQwen35.no_pending`, `Window.check`, `LayerGrads.parts`, and the self-call cluster on `TesslQwen35.bank`).

**`GateGrad.all_bits` is not a dead function.** It is a 0.40 row on `ojas-metal/src/gpu.rs`. The method is defined at `:2609` and called at `:2603` (`g.all_bits()`). Same class of resolver miss as the other 0.40 names.

## Top issues

1. The ojas index was stale on `ojas-metal/src/gpu.rs` at generation 479 and again at 492. A search hit for `causal_attn_backward` has span `[0, 0]`. A later sample reached generation 496 and reported fresh. Trust the source lines, not that span.
2. Tessl’s index reports fresh at generation 801. Phase 2’s stale generation 779 is not the current status. Rust resolution is still 236/1000, and the dead walk is incomplete (19371 of 36434).
3. No high-confidence symbol survived a second look as dead code. `qwen35_residual_add_f32` is launched from `src/qwen35.rs:2018`.
4. Tessl’s training attention backward is head dim 256 only (`kernels/qwen35_attn_bwd.metal:31`). There is no D=64 backward kernel.
5. The ojas tiny step does backpropagate head dim 64, with exact GEMM and `ojas_causal_softmax_bwd`, seq at most 16 (`ojas-metal/src/gpu.rs:448-452`, `:549`, `:734`). That is not the h256 flash backward. `docs/status.md` still says the step has no head-dim-64 attention backward.
6. Twelve layers at width 768 are not implemented (`ojas-nn/src/lib.rs:4`). The tiny step is `d_model` 128, two heads (`ojas-metal/src/gpu.rs:23-24`).
7. CUDA and HIP default to off and return `NotCompiled`. This session did not execute either device.
8. The tokenizer check is twenty strings (`ojas-data/src/bpe.rs:32`, `:870`), and the CPU torch match is one step (`ojas-cpu/tests/torch_ref.rs:52`).
