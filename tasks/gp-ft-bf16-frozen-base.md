---
id: "gp-ft-bf16-frozen-base"
title: "bf16 frozen base on Metal: four bf16 operand lanes (linear fwd, linear dX, SDPA fwd/bwd), HF bf16 load without widening, Precision::Bf16 through the provider, measured memory at T=2048 and T=9,638"
status: review
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
Status: review
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
- [x] Metal linear_forward accepts a bf16 weight (2 bytes per element resident) with f32 input and f32 accumulation; today Metal refuses bf16 operands (ojas-metal/src/backend.rs:1257, 1283) [V by Fable]
- [ ] Metal linear_backward_input (from gp-ft-frozen-params-and-lora) accepts the same bf16 weight
- [x] Metal SDPA forward and backward accept bf16 operands with f32 accumulation (backend.rs:1427, 1443 refuse today) [V by Fable]
- [x] HF bf16 safetensors load into bf16 frozen weights without widening to f32 (today ojas-model/src/qwen35/params.rs:304 reads through read_f32_widened [A]); a test asserts resident bytes for the frozen base
- [ ] LoRA weights, D17, the span head, norm weights, the residual stream and optimizer state stay f32
- [x] Qwen35Step::open takes a precision and passes Precision::Bf16 to tessl's Qwen35Model::load; today it hard-codes Precision::F32 (ojas-qwen35/src/step.rs:361) [V] although tessl trains a bf16-stored model with f32 masters (qwen35_model.rs:389-392) [V by Fable]. A 2B memory row f32 vs bf16 is committed
- [ ] Peak device bytes for one bf16-base LoRA step with checkpointing on, for 0.8B-Base and 2B-Base, at T=2048 and at T=9,638, micro-batch 1, committed under bench/results/ on the M5 Pro. If T=9,638 does not fit, the row says so and the pre-registered fallback applies (buckets capped at 8,192, longer rows dropped and counted; Fable D6)
- [ ] A Metal out-of-memory at these shapes surfaces as the typed capacity error (gp-oom-error-class)

## Planned files
- ojas-metal/src/backend.rs
- ojas-metal/src/device.rs
- ojas-model/src/qwen35/params.rs
- ojas-io/src/safetensors.rs
- ojas-qwen35/src/step.rs
- bench/results/

### Landed for review (2026-10-11, worktree bf16-frozen-base-on-metal-four-bf16-oper-b35da9b5)
- **Criteria 1 and 3: met** (14cb81b, 2de4421).
  - Metal's linear_forward, linear_backward (dX and dW) and causal SDPA forward and backward take Bf16 operands. Each
    is widened exactly on the device into an f32 copy charged for the call, then runs the existing exact-f32 command.
  - bf16_operands() is true. ojas-oracle's bf16_operand_contract passes on Metal (192 mixed calls, 15 refusals,
    26 non-finite cases), and tests/bf16_operands.rs shows a 2-byte residency with the copy released.
  - No TensorOps bf16 lane is used: it would round the f32 input.
- **Criterion 2: blocked** on gp-ft-frozen-params-and-lora, since linear_backward_input does not exist yet.
  linear_backward's dX already takes the bf16 weight, so the new op adopts that lane.
- **Criterion 4: met** (c32dccb).
  - load_hf(…, DType::Bf16) keeps a BF16 checkpoint's linears as stored (ojas-io read_bf16_bits).
  - A fixture test asserts the resident bytes: 2 per linear element, 4 for the rest.
  - The Tape gives the f32 load's loss and gradient bits on the CPU and on Metal.
  - The tied embedding stays f32 here (Fable). The follow-up ft-69744e54 landed the bf16 table lane on CPU and Metal
    (1ea6c06, 3a9c55c; 2B rows in bench/results/2026-10-14-bf16-tied-table).
- **Criterion 5: met for what exists.** Norms, conv, A_log, dt_bias, the residual stream and the optimizer
  state are f32. LoRA, D17 and the span head are not in the tree yet.
- **Criterion 6: met with one deviation** (d65879c).
  - open(snapshot, numerics, precision, budget). Bf16 uses tessl's load_tower rather than load, because
    tessl's AdamW refuses load's packed head copy for a training model.
  - AdamW runs with UpdateRule::F32Master and MomentStorage::F32. Bf16 with ExactF32 is refused, and save_state and
    load_state refuse a bf16 provider.
  - The 2B row is committed: resident 30.13 GB at both precisions (16 bytes per parameter). Steps past
    T=2048 f32 were stopped by the machine's 32 GiB memguard.
  - Lappi's two open() call sites need the new argument after merge.
- **Criterion 7: blocked** on frozen leaves and LoRA.
  - Interim full-gradient Tape rows at 2B are committed (bench/results/2026-10-11-bf16-frozen-base). They are
    labelled as not the LoRA step: bf16 peaks are 16.37 GB at T=2048 and 18.97 GB at T=9,638.
  - T=9,638 fits, so the fallback is not triggered.
  - They were measured on Qwen3.5-2B (identical text_config). 2B-Base and 0.8B-Base weights are not on the machine.
- **Criterion 8: blocked** on gp-oom-error-class. No device OOM was reached: the peak was 33.9 GB of 51.5 GB.

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: merged at 03412af [V]; review is right. The follow-up ft-69744e54 (bf16 tied table) is on branch gitpulse/bf16-tied-embedding-table-embedding-forw-eab146a9 at 5afa1fb, not on main.
- Criteria: 1, 3, 4 met [A]; 2 met, the note "blocked" is stale: Metal linear_backward_input takes the bf16 weight (backend.rs:1363-1389), pinned by ojas-metal/tests/frozen_adversarial.rs:62 and qwen35_lora_metal.rs:68 [A]; 5 partial: LoRA A/B f32, D17 and the span head do not exist yet [A]; 6 met with the recorded deviation; 7 open: gpu_memory_rows.rs has no LoRA arm and no 0.8B run [V]; 8 landed on main (classify_alloc, retrying) but only the Limit path is exercised by a real refusal (device.rs:3335); the Exhausted path only by injected strings [A].
- Stale [V]: "2B-Base and 0.8B-Base weights are not on this machine" (here and in bench/results/2026-10-11-bf16-frozen-base/README.md): both were fetched 2026-10-12 (~/.cache/huggingface/hub, 4.5 GB and 1.7 GB shards). "Lappi's two open() call sites need the new argument": provider.rs:146 already passes Precision::F32. gpu_memory_rows.rs:19-20 "the Tape has no frozen leaves yet" is false (fixed on the tied-table branch).
- Owed for criterion 7 [V]: Lappi's pre-registration requires the T = 9,638 memory row before the Mac run (campaign/lora-2b-preregistered.json:64). Inferred estimate for a LoRA step at T = 9,638: 13-14 GB (the 18.97 GB full-gradient row minus ~5.5 GB of linear weight gradients) [I]; the binding limit is the 32 GiB memguard, not the 51.5 GB working set.

### Brought up to main (2026-10-15)
- Merge commit 36f1994 takes main b47cc33 (the real-2B Tape gate, scaling-1 LoRA parity, fine-tune audit). One conflict, `ojas-qwen35/tests/gpu_tape_2b.rs`: main's gate is kept, with this branch's table dtype (`DType::F32`) added to its `load_hf` call. `gpu_memory_rows.rs` merged clean onto the shared `tests/common` row.
- Run (M5 Pro, under mac_heavy, 2026-10-15T18:32Z): `cargo clippy --workspace --all-targets --release -- -D warnings` clean; `cargo test --release --no-fail-fast -p ojas-model -p ojas-cpu -p ojas-qwen35 -p ojas-metal -p ojas-core -p ojas-autograd`: 1,065 passed, 0 failed, 44 ignored over 133 test binaries; `gpu_parity -- --ignored gpu_tiny`: 8 passed.
