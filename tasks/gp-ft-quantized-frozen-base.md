---
id: "gp-ft-quantized-frozen-base"
title: "Quantised LoRA arm (owner 2026-10-17): mlx quality gate first; the first quick runs train LoRA over a fake-quant q8 base file; resident block-quantised storage and dequant GEMM deferred to stage 2"
status: ready
priority: 1
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "quantization"
  - "metal"
  - "gated"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/dtype.rs"
  - "ojas-core/src/quant.rs"
  - "ojas-metal/src/backend.rs"
  - "gemma-metal/src/quant.rs"
acceptance_criteria:
  - "Gate, a Lappi-side measurement written before it runs: quantize the bf16 Qwen3.5-2B-Base with mlx.core.quantize (mlx 0.31.2 installed) to affine q8 g64 and q4 g64, and score at least 500 Lappi validation prompts (never held-out; rule 3) with the 17-row letter log-softmax in bf16 vs q8 vs q4. A format is admitted only if argmax agreement >= 99.0% and mean |delta log-prob| on the gold letter <= 0.02 nats (Fable D3)"
  - "A format is built only if it is admitted and a demonstrated memory need exists: a pre-registered shape that does not fit in bf16 at micro-batch 1. 4-bit additionally needs the user to override Unsloth's advice in writing"
  - "If built: a block-quantized weight type in ojas-core (MLX affine q8/q4, group 64, bf16 scale and bias), dequantization bit-exact against mlx.core.dequantize, and a dX-only dequant GEMM on CPU (reference) and Metal (tiled for the run's row counts; kernels in canonical tessl per Lappi rule 6). Today tessl's quantized kernels are inference-only GEMV or M<=8 with no transposed variant [A]"
  - "If built: one owner for host quantization; gemma-metal's quantizer is moved behind ojas-core or replaced, fixing its bf16 rounding (ties up, NaN to Inf; gemma-metal/src/quant.rs:572-576), silent NaN/Inf groups (quant.rs:452-461) and panics on group size 0 (kernels.rs:377) [A]"
  - "8-bit AdamW is not needed (optimizer state covers LoRA, D17 and the span head only)"
---

# Task brief v1

## Title
Quantised LoRA arm (owner 2026-10-17): mlx quality gate first; the first quick runs train LoRA over a fake-quant q8 base file; resident block-quantised storage and dequant GEMM deferred to stage 2

Task: gp-ft-quantized-frozen-base
Type: feature
Status: ready
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: fine-tune, quantization, metal, gated

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); gated by Fable's ruling D3 the same day. The user asked for a 16-, 8- or 4-bit base; 16-bit is the decided precision (gp-ft-bf16-frozen-base). Unsloth recommends against QLoRA on every Qwen3.5 model ('higher than normal quantization differences' [V fetched]) and memory does not force quantization on this Mac (2B and even 4B fit in bf16 [I]). The quantized path's real cost is weeks of tessl kernel work. The safetensors whole-file dtype refusal, which used to sit here, is now a P0 fix in gp-ft-training-path-correctness.

Depends on: the gate above; gp-imported-research-trees (gemma-metal's future). Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] Gate, a Lappi-side measurement written before it runs: quantize the bf16 Qwen3.5-2B-Base with mlx.core.quantize (mlx 0.31.2 installed) to affine q8 g64 and q4 g64, and score at least 500 Lappi validation prompts (never held-out; rule 3) with the 17-row letter log-softmax in bf16 vs q8 vs q4. A format is admitted only if argmax agreement >= 99.0% and mean |delta log-prob| on the gold letter <= 0.02 nats (Fable D3)
- [ ] A format is built only if it is admitted and a demonstrated memory need exists: a pre-registered shape that does not fit in bf16 at micro-batch 1. 4-bit additionally needs the user to override Unsloth's advice in writing
- [ ] If built: a block-quantized weight type in ojas-core (MLX affine q8/q4, group 64, bf16 scale and bias), dequantization bit-exact against mlx.core.dequantize, and a dX-only dequant GEMM on CPU (reference) and Metal (tiled for the run's row counts; kernels in canonical tessl per Lappi rule 6). Today tessl's quantized kernels are inference-only GEMV or M<=8 with no transposed variant [A]
- [ ] If built: one owner for host quantization; gemma-metal's quantizer is moved behind ojas-core or replaced, fixing its bf16 rounding (ties up, NaN to Inf; gemma-metal/src/quant.rs:572-576), silent NaN/Inf groups (quant.rs:452-461) and panics on group size 0 (kernels.rs:377) [A]
- [ ] 8-bit AdamW is not needed (optimizer state covers LoRA, D17 and the span head only)

## Planned files
- ojas-core/src/dtype.rs
- ojas-core/src/quant.rs
- ojas-metal/src/backend.rs
- gemma-metal/src/quant.rs

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: backlog, gated, accurate; stays unbuilt [A].
- The gate measurement is not run or written: Lappi has no docs/fine-tune-plan.md and no quantize script outside AUDIT and ledgers [A]. The memory-need condition looks unmet: the bf16 full-gradient row is 18.97 GB at T = 9,638 [V], an upper bound for LoRA.
- Stale: gemma-metal kernels.rs:377 cite moved (divisions at :301, :342, guarded by callers at :272, :285; the panic is not shown reachable); mlx_lm's interpreter has mlx 0.32.3, not 0.31.2 [A].

### Fable ruling (2026-10-17)

Synced from the GitPulse board by the 2026-10-17 brief sync; the board is canonical (Fable's final ruling of 2026-10-17, section 5), this section is the long-form record. Board card ft-a1a4ba4d06d18ef307fa7776545e9a52, revision 4: status ready, priority 1, severity medium. Criterion numbers are the board's, counted from 0 in the order of the card's acceptance criteria.

Ruling set: status ready (unchanged), priority 1 (unchanged), severity high -> medium, title -> Quantised LoRA arm (owner 2026-10-17): mlx quality gate first; the first quick runs train LoRA over a fake-quant q8 base file; resident block-quantised storage and dequant GEMM deferred to stage 2

Marks (the text of the existing criterion is otherwise unchanged on the board; nothing deleted, no threshold moved):

- ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 1: [SUPERSEDED 2026-10-17 in its memory-need clause by criterion 5 (owner waiver); the 4-bit written-override clause stands and has not been met]
- ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 2: [DEFERRED 2026-10-17 to stage 2 (new card): not on the first quantised run's path]
- ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 3: [DEFERRED 2026-10-17 to stage 2 (new card); gemma-metal's quantiser is not MLX-exact (ojas gemma-metal/src/quant.rs:346-448: scale (hi-lo)/15, bias lo, no edge/at_zero snapping; q8 signed int8 with f32 scale/zero) and is not reused]

Criteria appended by the ruling, verbatim:

- [ ] ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 9: (Fable 2026-10-17) Scope for this track: the quantised arm's smoke and reduced run train the bf16 LoRA path over a fake-quant base file (dequantize(quantize(W)) for the 150 wrapped linears from mlx q8 g64 with bf16 scale and bias, written by the Lappi gate tool, pinned by sha256); ojas needs no quantised dtype, kernel or host quantiser for them. The memory arithmetic is recorded here: 2.745 GB bf16 wrapped linears vs ~1.46 GB q8 g64, a 1.29 GB saving of an estimated 12-14 GB step; no pre-registered shape needs it.
- [ ] ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 10: (Fable 2026-10-17) Criterion 8's fixture: the tiny Qwen3.5 fixture's wrapped linears replaced by dequantize(quantize(W)) from mlx (q8 g64, bf16 scale and bias; mlx version and the quantiser call in the fixture manifest), torch fp32 one-step and 20-step references regenerated by the fixture generator, and the existing (a'), (b'), (c') LoRA tests run on it unchanged on CPU and Metal.
- [ ] ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 11: (Fable 2026-10-17) The owner's waiver (criterion 5) is valid as the owner's decision but is recorded only on this board; it is written into Lappi in the owner's words before the sibling pre-registration (ft-da2c4ab1). q4 is measured in the gate and not built. Stage 2 (resident q8 storage, dequant-to-bf16-scratch kernel in canonical tessl, bit-identical loss to the fake-quant run) and stage 3 (fused dequant GEMM) start only on the owner's yes after the fake-quant reduced 2B run is scored.

Board criteria and marks already on the card before the ruling (2026-10-17 owner, gap and fold entries, and earlier audit marks), not previously in this brief, verbatim:

- [ ] ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 5: (owner 2026-10-17, quantised arm) The owner wants a quantised LoRA arm trained alongside the bf16 run. This waives the 'demonstrated memory need' condition of criterion 2 (that condition was the owner's to set, per Fable D3). The quality gate of criterion 1 is unchanged (gates are read-only) and still decides which format may be built: no quantised kernel work starts until the mlx q8/q4 measurement is recorded and a format is admitted.
- [ ] ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 6: (owner 2026-10-17, quantised arm) q8 (MLX affine, group 64) is the format to build if admitted. q4 is built only if admitted and the owner overrides Unsloth's 'no QLoRA on Qwen3.5' advice in writing; that override has not been given.
- [ ] ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 7: (owner 2026-10-17, quantised arm) The quantised set is written into the arm's pre-registration before any quantised result is read: which linears are quantised (attention, linear_attn and MLP projections). The tied embedding and head stay bf16 unless the record says otherwise, because D17 reads embed[answer ids] from it by name and the report-only full-vocab CE uses it. Norms, conv1d, LoRA, D17 and the span head stay f32.
- [ ] ft-a1a4ba4d06d18ef307fa7776545e9a52 criterion 8: (owner 2026-10-17, quantised arm) Training through the Tape with quantised frozen linears: the LoRA linear's forward and its dX-only backward run over the dequantised weight and take no weight gradient. On the tiny fixture, one-step loss and the adapter gradients match a torch fp32 oracle run on dequantize(quantize(W)) weights (fake-quant from mlx.core.dequantize; no bitsandbytes, Fable D7) within the bounds of the bf16 LoRA parity tests, on CPU and Metal.
