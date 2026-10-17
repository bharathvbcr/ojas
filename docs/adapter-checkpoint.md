# Adapter checkpoint, layout version 1

The checkpoint of the Qwen3.5 Tape LoRA track: the trainable entries by name, their AdamW moments, and what a resume needs to continue the same run or refuse. Lappi's converter (ft-58e316a9) reads it by this contract.

Writer and reader: `ojas-model/src/qwen35/adapters.rs` (`write_adapter_checkpoint`, `read_adapter_checkpoint`). Tests: `ojas-model/tests/qwen35_adapters.rs`. Where this page and the code disagree, the code wins and this page is the bug.

This is not Checkpoint v1 (`docs/checkpoint-v1.md`, the nanolab trainer's format). It shares the file conventions of the nanolab checkpoint directory (safetensors, `adam_m.<name>` / `adam_v.<name>`) and its atomic replace.

## Directory

A checkpoint is a directory with exactly two files. It is written to a staging sibling and swapped in whole (`ojas_io::replace_dir_with_if`), so a reader sees the old directory or the new one. An existing target is replaced only if it is absent, empty, or an earlier adapter checkpoint (`.DS_Store` excepted); anything else is refused with nothing touched.

| File | Content |
| :--- | :--- |
| `adapters.safetensors` | One `F32` tensor per entry under its own name. All scalars of the checkpoint are its safetensors metadata (below). |
| `moments.safetensors` | `adam_m.<name>` (first moment) and `adam_v.<name>` (second moment) per entry, `F32`, each the shape of its entry. |

Tensors are row-major, little-endian, written in ascending byte-wise name order. Every value and every moment is finite, and every second moment is `>= 0`.

## Entry names

A LoRA adapter is two entries, the PEFT convention:

| Name | Shape | |
| :--- | :--- | :--- |
| `<module path>.lora_A.weight` | `[r, in]` | `A` |
| `<module path>.lora_B.weight` | `[out, r]` | `B`, fused |

`<module path>` is the Hugging Face tensor name without `.weight`, for example `model.language_model.layers.3.self_attn.q_proj`. A fused Hugging Face tensor (`in_proj_qkv`, `q_proj`) has one fused `B` in its own row order, as PEFT shapes it, not the model's parts. The scaling is `alpha / r` and is not folded into either tensor.

The host entries have fixed names, the ones Lappi's own checkpoint uses (`qd-export/src/layout.rs`: `d17.` + `delta`, `AnswerRowDelta.delta`; and `span_head.` + the torch state-dict keys of `qd-train/src/span_head.rs` `PARAMETER_NAMES`). Layout 1 holds no others:

| Name | Shape (Lappi's) |
| :--- | :--- |
| `d17.delta` | `[17, H]` |
| `span_head.start_proj.weight` | `[H, H]`, `[out, in]` |
| `span_head.end_proj.weight` | `[H, H]`, `[out, in]` |
| `span_head.abstain_start` | `[H]` |
| `span_head.abstain_end` | `[H]` |

Every entry name must be one of these or a `<module>.lora_A.weight` / `<module>.lora_B.weight` with a non-empty module; any other name is refused on save and on read (`UnknownEntryName`). The shapes are the caller's expectation, checked on resume, not fixed by layout 1. Host entries carry the same moment names (`adam_m.<name>`).

## Metadata

All values are strings. Keys of `adapters.safetensors`:

| Key | Value |
| :--- | :--- |
| `ojas.adapter.layout` | `1`. Also in `moments.safetensors`. |
| `ojas.adapter.step` | Completed optimiser steps, decimal u64. This is the schedule step and the `step` argument of the next `Backend::adamw_step` (the count before that update). Also in `moments.safetensors`, and the two must agree. |
| `ojas.adapter.payload_fnv1a64` | 16 lowercase hex digits: FNV-1a 64 (offset `cbf29ce484222325`, prime `100000001b3`) over (1) the little-endian bytes of the file's tensors in ascending name order, then (2) every metadata pair of the file except this key, sorted by key, each pair as `len(key)` (u64 LE), key bytes, `len(value)` (u64 LE), value bytes. So a changed rng state, digest, step, base, quantizer, rank or alpha is detected even when it is still well-formed. Also in `moments.safetensors` (over that file's own tensors and metadata). An integrity check against truncation and bit rot, not an authenticity one. |
| `ojas.adapter.pair` | `moments.safetensors` only: the `ojas.adapter.payload_fnv1a64` of the `adapters.safetensors` it was written beside. Binds the two files, so moments from another save or run at the same step are refused. |
| `ojas.adapter.rng_state` | Data/batch-order RNG state, lowercase hex of the caller's bytes (1 to 4096 bytes). Opaque to ojas. |
| `ojas.adapter.consumed_digest` | The consumed-data digest, the caller's lowercase hex string (even length, at most 4096). Opaque to ojas. |
| `ojas.adapter.base_sha256` | sha256 of the frozen base file, 64 lowercase hex digits. |
| `ojas.adapter.base_file` | The name of that file (one plain file name, no directory). Layout 1 covers a single-file base only. A one-of-one name (`model.safetensors-00001-of-00001.safetensors`, as the local Qwen3.5 2B-Base and 0.8B-Base snapshots are named) is a single file and is accepted; a shard of several (`...-00001-of-00004.safetensors`) or an index (`...safetensors.index.json`) is refused at save with `ShardedBase`, since one sha256 would name an ambiguous set of files. |
| `ojas.adapter.lora_rank` | `r`, decimal. |
| `ojas.adapter.lora_alpha` | `alpha`, the f32 in Rust's shortest round-trip decimal (`16` for 16.0). A reader parses it as an f32 and compares the value (bit for bit), not the string. |
| `ojas.adapter.quant.format`, `ojas.adapter.quant.group_size`, `ojas.adapter.quant.mlx_version` | The quantizer record of a fake-quant base, copied from the arm's pre-registration. All three present or all three absent. Strings, compared exactly. |

There is no dropout key: the recipe's LoRA dropout is 0 and a non-zero dropout refuses to checkpoint (`AdapterError::Dropout`), so a resume never depends on a dropout stream.

The RNG state is the data-order RNG only. The frozen base is named by the name and sha256 of the single base file the run loaded (for the quantised arm, the fake-quant bf16 file) plus the quantizer record; the block-quantised storage dtype is not part of layout 1.

## Refusals

Save refuses, before touching the target: a dropout other than 0, a sharded or non-plain base file name, an entry name that is neither a LoRA A/B nor a fixed host entry, a malformed base sha256 / digest / RNG state / quantizer string, no entries, a duplicate or empty or reserved (`__metadata__`) name, a shape whose element count is 0 or overflows, a value or moment count that is not the shape's, a non-finite value or moment, a negative second moment.

Read, given what the resume expects (base identity, LoRA rank and alpha, entry names and shapes), refuses in this order: an unreadable or truncated file; a layout version other than `1` (before anything else is interpreted, in both files); missing or malformed metadata; a different base sha256 or base file name; a different or one-sided quantizer record; a different rank or alpha; files written at different steps; a missing or extra entry in either file; a dtype other than `F32` or a different shape; any non-finite value; a payload hash that does not match the recorded one (either file); moments whose `ojas.adapter.pair` is not the adapters file's hash (`PairMismatch`); a negative second moment.

## Versioning

A change to a name, shape, dtype, metadata key or its meaning, or to the hash, is a new `ojas.adapter.layout`. Readers refuse a version they do not know; they do not guess or migrate.
