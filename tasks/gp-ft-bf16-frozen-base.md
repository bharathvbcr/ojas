---
id: "gp-ft-bf16-frozen-base"
title: "bf16 frozen base on Metal: four bf16 operand lanes (linear fwd, linear dX, SDPA fwd/bwd), HF bf16 load without widening, Precision::Bf16 through the provider, measured memory at T=2048 and T=9,638"
status: ready
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "bf16"
  - "metal"
  - "memory"
  - "qwen35"
repositories:
  - "ojas"
planned_files:
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-model/src/qwen35/params.rs"
  - "ojas-io/src/safetensors.rs"
  - "ojas-qwen35/src/step.rs"
  - "bench/results/"
acceptance_criteria:
  - "Metal linear_forward accepts a bf16 weight (2 bytes per element resident) with f32 input and f32 accumulation; today Metal refuses bf16 operands (ojas-metal/src/backend.rs:1257, 1283) [V by Fable]"
  - "Metal linear_backward_input (from gp-ft-frozen-params-and-lora) accepts the same bf16 weight"
  - "Metal SDPA forward and backward accept bf16 operands with f32 accumulation (backend.rs:1427, 1443 refuse today) [V by Fable]"
  - "HF bf16 safetensors load into bf16 frozen weights without widening to f32 (today ojas-model/src/qwen35/params.rs:304 reads through read_f32_widened [A]); a test asserts resident bytes for the frozen base"
  - "LoRA weights, D17, the span head, norm weights, the residual stream and optimizer state stay f32"
  - "Qwen35Step::open takes a precision and passes Precision::Bf16 to tessl's Qwen35Model::load; today it hard-codes Precision::F32 (ojas-qwen35/src/step.rs:361) [V] although tessl trains a bf16-stored model with f32 masters (qwen35_model.rs:389-392) [V by Fable]. A 2B memory row f32 vs bf16 is committed"
  - "Peak device bytes for one bf16-base LoRA step with checkpointing on, for 0.8B-Base and 2B-Base, at T=2048 and at T=9,638, micro-batch 1, committed under bench/results/ on the M5 Pro. If T=9,638 does not fit, the row says so and the pre-registered fallback applies (buckets capped at 8,192, longer rows dropped and counted; Fable D6)"
  - "A Metal out-of-memory at these shapes surfaces as the typed capacity error (gp-oom-error-class)"
---

# Task brief v1

## Title
bf16 frozen base on Metal: four bf16 operand lanes (linear fwd, linear dX, SDPA fwd/bwd), HF bf16 load without widening, Precision::Bf16 through the provider, measured memory at T=2048 and T=9,638

Task: gp-ft-bf16-frozen-base
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: fine-tune, bf16, metal, memory, qwen35

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); narrowed by Fable's ruling the same day (D3 and correction 5). Precision is decided: bf16 frozen base, f32 LoRA/head/norms; 8- and 4-bit are not built (Unsloth: 'It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense' [V fetched]; Lappi AUDIT/training-audit-2026-10-06.md:469 says the same). Memory does not force quantization: a 2B bf16 base is about 3.8 GB [I].

This brief carries only the four Metal bf16 lanes the fine-tune needs, so it does not wait on the whole gp-bf16-compute-tier (related, not a dependency: its wgpu and every-op items are outside this track).

Widths come from Lappi's pre-registered data shape (rows up to 9,638 tokens; Fable D6, approved by the user 2026-10-09). T=2048 is kept for the smoke and the D1 throughput comparison.

Depends on: gp-ft-frozen-params-and-lora (linear_backward_input), gp-backend-wrapper-forwarding, gp-device-memory-probes, gp-oom-error-class. Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] Metal linear_forward accepts a bf16 weight (2 bytes per element resident) with f32 input and f32 accumulation; today Metal refuses bf16 operands (ojas-metal/src/backend.rs:1257, 1283) [V by Fable]
- [ ] Metal linear_backward_input (from gp-ft-frozen-params-and-lora) accepts the same bf16 weight
- [ ] Metal SDPA forward and backward accept bf16 operands with f32 accumulation (backend.rs:1427, 1443 refuse today) [V by Fable]
- [ ] HF bf16 safetensors load into bf16 frozen weights without widening to f32 (today ojas-model/src/qwen35/params.rs:304 reads through read_f32_widened [A]); a test asserts resident bytes for the frozen base
- [ ] LoRA weights, D17, the span head, norm weights, the residual stream and optimizer state stay f32
- [ ] Qwen35Step::open takes a precision and passes Precision::Bf16 to tessl's Qwen35Model::load; today it hard-codes Precision::F32 (ojas-qwen35/src/step.rs:361) [V] although tessl trains a bf16-stored model with f32 masters (qwen35_model.rs:389-392) [V by Fable]. A 2B memory row f32 vs bf16 is committed
- [ ] Peak device bytes for one bf16-base LoRA step with checkpointing on, for 0.8B-Base and 2B-Base, at T=2048 and at T=9,638, micro-batch 1, committed under bench/results/ on the M5 Pro. If T=9,638 does not fit, the row says so and the pre-registered fallback applies (buckets capped at 8,192, longer rows dropped and counted; Fable D6)
- [ ] A Metal out-of-memory at these shapes surfaces as the typed capacity error (gp-oom-error-class)

## Planned files
- ojas-metal/src/backend.rs
- ojas-metal/src/device.rs
- ojas-model/src/qwen35/params.rs
- ojas-io/src/safetensors.rs
- ojas-qwen35/src/step.rs
- bench/results/
