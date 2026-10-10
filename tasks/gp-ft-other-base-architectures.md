---
id: "gp-ft-other-base-architectures"
title: "Other bases, out of the LoRA-2B track: a trainable Llama-3.2 block on the Tape with HF loading and parity; Gemma-4-E4B requirements recorded and deferred"
status: backlog
priority: 3
severity: low
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "llama"
  - "gemma"
  - "model"
repositories:
  - "ojas"
planned_files:
  - "ojas-model/src/spec.rs"
  - "ojas-model/src/llama/"
  - "ojas-data/src/bpe.rs"
acceptance_criteria:
  - "A Llama-3.2 block (RMSNorm, SwiGLU, GQA, llama3 RoPE frequency scaling, tied embeddings) trains on the Tape with an HF safetensors loader; forward and backward on a tiny config match transformers within a stated bound. ModelSpec forces nanolab-only features today (ojas-model/src/spec.rs:38-43) [A], so it is generalised or a separate spec is added"
  - "Llama's tokenizer is supported (a tokenizer.json reader; out of the LoRA-2B track)"
  - "Gemma-4-E4B is deferred, with its requirements recorded: gelu, logit softcapping, head_dim above 256, per-layer embeddings, shared KV. gemma-metal is inference-only and stays outside the workspace until gp-imported-research-trees decides its future"
---

# Task brief v1

## Title
Other bases, out of the LoRA-2B track: a trainable Llama-3.2 block on the Tape with HF loading and parity; Gemma-4-E4B requirements recorded and deferred

Task: gp-ft-other-base-architectures
Type: feature
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: fine-tune, llama, gemma, model

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949). Fable D4 (user-approved 2026-10-09): Llama-3.2 and Gemma-4 are out of scope for Lappi, whose serving is Qwen3.5-only. Kept as a low-priority record of what a second architecture would cost: Llama is medium (every op exists; a block, loader, RoPE table and tokenizer), Gemma-4 large.

Depends on: gp-ft-frozen-params-and-lora, gp-imported-research-trees. Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] A Llama-3.2 block (RMSNorm, SwiGLU, GQA, llama3 RoPE frequency scaling, tied embeddings) trains on the Tape with an HF safetensors loader; forward and backward on a tiny config match transformers within a stated bound. ModelSpec forces nanolab-only features today (ojas-model/src/spec.rs:38-43) [A], so it is generalised or a separate spec is added
- [ ] Llama's tokenizer is supported (a tokenizer.json reader; out of the LoRA-2B track)
- [ ] Gemma-4-E4B is deferred, with its requirements recorded: gelu, logit softcapping, head_dim above 256, per-layer embeddings, shared KV. gemma-metal is inference-only and stays outside the workspace until gp-imported-research-trees decides its future

## Planned files
- ojas-model/src/spec.rs
- ojas-model/src/llama/
- ojas-data/src/bpe.rs
