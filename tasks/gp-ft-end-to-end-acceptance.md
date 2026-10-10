---
id: "gp-ft-end-to-end-acceptance"
title: "LoRA-2B on the M5 Pro with ojas: the gating 2B Tape run, the pre-registration, the plain-torch reference arm, a 0.8B-Base smoke within Fable's bound (d'), then a reduced quick 2B-Base run scored by Lappi"
status: ready
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "lappi"
  - "parity"
  - "acceptance"
  - "metal"
repositories:
  - "ojas"
  - "Lappi-decision"
planned_files:
  - "bench/results/"
  - "ojas-oracle/"
acceptance_criteria:
  - "The decisions are recorded: Fable's ruling and the user's answers (2026-10-09) are in Lappi AUDIT/lora-2b-2026-10-09/ (fable-ft-decisions.md, human-decisions.md). Tape path (D1), 17-row readout with D17 (D2), bf16 base with 8/4-bit gated (D3), 2B-Base with a 0.8B-Base smoke and 4B out (D4), merged bf16 release (D5), Lappi's data shape and LRSchedule (D6), plain-torch reference arm (D7), quick Mac rows and the trainer moved into ojas-train (D8)"
  - "The gating run gp-qwen35-2b-metal-tape-run passes on the local Qwen3.5-2B instruct snapshot before any LoRA code is relied on (D1)"
  - "Lappi's campaign/lora-2b-preregistered.json is written before any smoke result is read, with Fable's D6 table: v5 width (max_seq_len 10240, widest ~9,638) and token budget 35,403 with grad_accum 1; r=16 alpha=16 dropout 0 on all linears including linear_attn; lr 2e-4 (D17 and span head 1e-4); wd 0.01; betas (0.9, 0.999), eps 1e-8; LRSchedule warmup steps//20, cosine to 0; no 0.1x layer split; optimizer_recipe 'lora'; init digests; 1 seed and quick on the Mac"
  - "The plain-torch LoRA reference arm exists in Lappi's python/qd_train with the same shards, supervision, init, schedule and batch order (no peft or bitsandbytes)"
  - "Smoke on 0.8B-Base (download approved 2026-10-09), 60 steps, run for both objectives (17-way D17 and full-vocab; D2's reversal test): ojas vs torch MPS bf16 within |delta loss| <= 0.02 nats for steps 1-50 and EMA(0.9) <= 0.05 after, with the fp32 torch CPU arm as the tight pin (bound d'). Peak bytes and tokens/s committed under bench/results/ and written as a quick ledger row"
  - "Reduced 2B-Base run (download approved): a truncated schedule of roughly 200-400 steps inside the approved 3-6 h Mac window, survives a kill -9 and resumes, exports through gp-ft-export, and is scored by Lappi's own scorer as a quick row that cannot promote. Promotion, if the rows justify it, is a separate box campaign on the torch arm (the user's spending decision)"
  - "Every gp-ft-* brief and every dependency below is done or explicitly waived in this brief's log"
---

# Task brief v1

## Title
LoRA-2B on the M5 Pro with ojas: the gating 2B Tape run, the pre-registration, the plain-torch reference arm, a 0.8B-Base smoke within Fable's bound (d'), then a reduced quick 2B-Base run scored by Lappi

Task: gp-ft-end-to-end-acceptance
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: fine-tune, lappi, parity, acceptance, metal

## Repositories
- ojas
- Lappi-decision

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); rewritten the same day from Fable's ruling and the user's answers. Umbrella for the fine-tune track.

Framing (Fable Q0, approved): LoRA-2B is a new, separately pre-registered recipe family compared with v5 at the gates only, never row for row. The parity target ojas is held to is a plain-torch arm of the same LoRA recipe, not v5's full-FT trainer. The Mac produces quick rows that decide whether box hours are spent, never a promotion. Unsloth's bf16 LoRA settings are adopted; its 4-bit base, seq 2048 and Clef head are not, on the evidence in the ruling.

Starting point: on 2026-10-09 the whole ojas workspace test suite passed on this Mac's GPU at d431949 (1825 passed, 0 failed, 41 ignored, Metal suites included; reported by the ojas-ae session's gate run, not re-run here). Lappi v0.1 was a full fine-tune in PyTorch on 2x H100; no Mac training run of Lappi has completed.

Critical path: gp-ft-training-path-correctness and gp-qwen35-2b-metal-tape-run (P0) -> gp-ft-frozen-params-and-lora -> gp-ft-bf16-frozen-base and gp-ft-decision-head-and-data-path -> gp-ft-trainer-loop (ojas-train move) -> gp-ft-step-memory-and-throughput -> pre-registration + reference arm -> 0.8B smoke -> reduced 2B run -> gp-ft-export.

Also depends on: gp-oom-error-class, gp-device-memory-probes, gp-backend-wrapper-forwarding, gp-oracle-and-hardening-coverage, gp-qwen35-host-neutral-crate. Lappi side: gp-ft-mac-trainer-blockers and gp-ft-lora-2b-lane on Lappi's board. Gated or out of track: gp-ft-quantized-frozen-base, gp-ft-step-throughput-followups, gp-ft-other-base-architectures, gp-qwen35-above-2b.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] The decisions are recorded: Fable's ruling and the user's answers (2026-10-09) are in Lappi AUDIT/lora-2b-2026-10-09/ (fable-ft-decisions.md, human-decisions.md). Tape path (D1), 17-row readout with D17 (D2), bf16 base with 8/4-bit gated (D3), 2B-Base with a 0.8B-Base smoke and 4B out (D4), merged bf16 release (D5), Lappi's data shape and LRSchedule (D6), plain-torch reference arm (D7), quick Mac rows and the trainer moved into ojas-train (D8)
- [ ] The gating run gp-qwen35-2b-metal-tape-run passes on the local Qwen3.5-2B instruct snapshot before any LoRA code is relied on (D1)
- [ ] Lappi's campaign/lora-2b-preregistered.json is written before any smoke result is read, with Fable's D6 table: v5 width (max_seq_len 10240, widest ~9,638) and token budget 35,403 with grad_accum 1; r=16 alpha=16 dropout 0 on all linears including linear_attn; lr 2e-4 (D17 and span head 1e-4); wd 0.01; betas (0.9, 0.999), eps 1e-8; LRSchedule warmup steps//20, cosine to 0; no 0.1x layer split; optimizer_recipe 'lora'; init digests; 1 seed and quick on the Mac
- [ ] The plain-torch LoRA reference arm exists in Lappi's python/qd_train with the same shards, supervision, init, schedule and batch order (no peft or bitsandbytes)
- [ ] Smoke on 0.8B-Base (download approved 2026-10-09), 60 steps, run for both objectives (17-way D17 and full-vocab; D2's reversal test): ojas vs torch MPS bf16 within |delta loss| <= 0.02 nats for steps 1-50 and EMA(0.9) <= 0.05 after, with the fp32 torch CPU arm as the tight pin (bound d'). Peak bytes and tokens/s committed under bench/results/ and written as a quick ledger row
- [ ] Reduced 2B-Base run (download approved): a truncated schedule of roughly 200-400 steps inside the approved 3-6 h Mac window, survives a kill -9 and resumes, exports through gp-ft-export, and is scored by Lappi's own scorer as a quick row that cannot promote. Promotion, if the rows justify it, is a separate box campaign on the torch arm (the user's spending decision)
- [ ] Every gp-ft-* brief and every dependency below is done or explicitly waived in this brief's log

## Planned files
- bench/results/
- ojas-oracle/
