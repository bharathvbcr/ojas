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

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: ready; no LoRA-2B run exists on any stack (Lappi campaign/lora-2b-preregistered.json:3) [A].
- Criteria: 1 decisions recorded, met (Lappi AUDIT/lora-2b-2026-10-09/fable-ft-decisions.md, human-decisions.md) [A]; 2 gating run open (gp-qwen35-2b-metal-tape-run); 3 pre-registration met (Lappi 5b6d6cb, amended c61584b before any result) [A]; 4 plain-torch reference partial: python/qd_train/lora.py has the module, D17, schedule and groups with 19 tests, but no driver runs shards, supervision, batch order and per-step logging [A]; 5 0.8B smoke open (weights local; no Tape trainer) [V weights]; 6 reduced 2B-Base run open; 7 dependencies open.
- Stale [V]: download approval is shown pending; 2B-Base and 0.8B-Base weights are local since 2026-10-12 (sha256 in Lappi AUDIT/lora-2b-2026-10-09/downloads.md). The pre-registration and reference arm are described as to-build; both mostly exist.
- New, P1 [V]: a one-seed D17 export has no route to the scorer. The Mac row is one seed (pre-registration :4, :8); the served model includes D17 (recipe.head.d17, "served as embed[answer ids] + D17"); Lappi's Metal export contract holds only `tower.*` and `span_head.*` (crates/qd-train/src/export.rs:1-20); `ckpt_average` refuses fewer than 2 inputs (tools/ckpt_average.py:1585-1586). Home: gp-ft-export.
- New, P1 [A]: Lappi's StepProvider contract cannot express the 17-way readout (RowTargets is full-vocab CE; accumulate returns one loss). Decide where D17 lives (host-side like the span head is an option) before gp-ft-trainer-loop's Tape provider. Home: gp-ft-decision-head-and-data-path.

Critical path to this run, ordered (merge state at 88be893):
1. gp-qwen35-2b-metal-tape-run: make the provider reference fit under the 32 GiB memguard (save the T = 128 reference to disk, or read gradients per entry), then run the T = 128 gate on 2B. Needs a user call (redesign vs a memguard window). Not started.
2. gp-ft-step-memory-and-throughput: per-gradient want flags on the fused CE (frozen head). ~~Not started.~~ (corrected 2026-10-17 at 5b67cdd: the flags are on main, CeGrads ojas-core/src/backend.rs:1748, Tape::fused_ce tape.rs:968; the rest of the card is open.)
3. gp-ft-bf16-frozen-base criterion 7: a LoRA arm in gpu_memory_rows, 0.8B-Base and 2B-Base at T = 2,048 and 9,638, with tokens/s; the T = 9,638 row is required before the Mac run. Not started; merge the tied-table branch first (1.02 GB less resident).
4. gp-ft-frozen-params-and-lora: a parity run at alpha == r; ratify or amend (b'). Merged at efa507f.
5. gp-ft-decision-head-and-data-path: hidden-state forward, gather at answer positions, D17 and its place in the provider contract. Not started.
6. gp-ft-trainer-loop: ojas-train per Lappi's 2026-10-14 Fable ruling, the Tape StepProvider, adapter checkpoint with sha256. Not started; ~~Lappi pins ojas at bc498c4~~ (corrected 2026-10-17 at 5b67cdd: the Lappi-decision working tree's upstream.lock.json reads ojas 132b46ea63e87868f468c016e02fe99acf2bb164, read 2026-10-17; whether that is committed on Lappi's main is not verified here) and a pin bump needs the user.
7. gp-ft-export: merged-bf16 tower export and the single-seed D17 route. Not started.
8. This card: the reference-arm driver, the 0.8B 60-step smoke (GPU, ~2 h), the reduced 2B-Base run with kill -9 resume (GPU, 3-6 h).

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
Revised order. Start now on main:
1. Run the two tiny GPU pins (gpu_save_state_bank.rs, gpu_parity.rs:654) and record them.
2. gpu_tape_2b redesign against transformers' reference (gp-qwen35-2b-metal-tape-run); queue tessl's fixture generator first.
3. The scaling-1 LoRA parity test (gp-ft-frozen-params-and-lora).

After the tied-table branch (ft-69744e54) merges, the owner's call:
4. Per-gradient want flags on the fused CE (ft-2bbced18); it conflicts with the branch's Metal linear_cross_entropy_mean rewrite. Not on the run's correctness path (D17 is host-side), but it decides the D1 tokens/s reading and the 0.8B full-vocab arm's cost, so it lands before the smoke.
5. The LoRA arm in gpu_memory_rows in the run's configuration (bf16 linears and table, frozen base, r = 16, alpha = 16), 0.8B-Base and 2B-Base at T = 2,048 and 9,638 with tokens/s.

Then the decision head with host-side D17, ojas-train with hand-rolled sha256, export with the one-seed D17 route, and the runs. The Lappi pin is bumped to 132b46e (Lappi 21fb7ad, branch ojas-pin-88be893); the bump to the ojas-train sha comes later.

### Fable ruling (2026-10-17)

Synced from the GitPulse board by the 2026-10-17 brief sync; the board is canonical (Fable's final ruling of 2026-10-17, section 5), this section is the long-form record. Board card ft-da2c4ab1677189ec57271befbd0040c1, revision 4: status ready, priority 1, severity high. Criterion numbers are the board's, counted from 0 in the order of the card's acceptance criteria.

Ruling set: status ready, priority 1, severity high (unchanged).

Marks (the text of the existing criterion is otherwise unchanged on the board; nothing deleted, no threshold moved):

- ft-da2c4ab1677189ec57271befbd0040c1 criterion 1: [MET WITH DEVIATION 2026-10-17: the gate ran on Qwen3.5-2B-Base against transformers' fp32 reference, not the instruct snapshot against the provider (Fable 2026-10-15 redesign); bench/results/2026-10-15-qwen35-2b-tape-gate]
- ft-da2c4ab1677189ec57271befbd0040c1 criterion 2: [MET 2026-10-17: Lappi 5b6d6cb, amended c61584b before any result]
- ft-da2c4ab1677189ec57271befbd0040c1 criterion 5: [SUPERSEDED 2026-10-17 in its "roughly 200-400 steps inside the approved 3-6 h window" clause by amendment 1 (next criteria): at the measured full-gradient Tape rate (203-281 tok/s, bench/results/2026-10-11-bf16-frozen-base and 2026-10-14-bf16-tied-table) a 35,403-token step is 126-174 s, so 400 steps cannot fit 6 h and 200 is marginal; the kill -9, export and scoring clauses stand]

Criteria appended by the ruling, verbatim:

- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 10: (Fable 2026-10-17) One step is one token-budget batch of 35,403 padded tokens (rows of one bucket width, widths up to 9,638, grad_accum 1), fed to the provider one sequence per launch with gradients summed (Lappi crates/qd-train/src/trainer.rs:745-780); "micro-batch 1" in the memory rows names that per-launch width. The step's device peak is the T=9,638 row's; its wall is ~35,403 tokens at the LoRA rate.
- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 11: (Fable 2026-10-17) Amendment 1 to Lappi campaign/lora-2b-preregistered.json, committed before any LoRA-2B result is read and after the owner's yes: (i) the step definition above in words; (ii) the window binds: the reduced 2B run's step count N is fixed from ft-2bbced18 criterion 7's measured LoRA tokens/s row as the largest N <= 400 whose N x 35,403 tokens fit 80% of the approved window, with warmup max(1, N//20) and cosine over N; N and the row id are written into the amendment before step 0 and into the ledger row. The owner may instead extend the window (owner item).
- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 12: (Fable 2026-10-17) Order of the quick runs: LoRA memory and tokens/s rows (ft-2bbced18 criterion 7) -> 0.8B bf16 smoke -> amendment 1 -> reduced 2B bf16 run -> quantised 0.8B smoke -> reduced quantised 2B run. No quantised run starts before the bf16 reduced run's ledger row exists.
- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 13: (Fable 2026-10-17) The first quantised quick runs are the bf16 trainer over a fake-quant base: a safetensors file whose 150 wrapped linears are dequantize(quantize(W)) from mlx (q8, group 64, bf16 scale and bias; mlx version recorded) and whose other tensors are the originals, written by the Lappi gate tool for 2B-Base and 0.8B-Base, pinned by sha256 in the quantised arm's sibling pre-registration and in the checkpoint's base name. No quantised dtype, kernel or host quantiser is on this run's path; a later resident-q8 (stage 2) run must reproduce this run's tiny-fixture loss bit for bit.
- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 14: (Fable 2026-10-17) Criterion 7's D3 addendum is a Lappi owner-decision file (AUDIT/lora-2b-quantised-2026-10-17/owner-decision.md or an addendum to AUDIT/lora-2b-2026-10-09/human-decisions.md) in the owner's words: the memory-need waiver, q8 only, no 4-bit override, the memory arithmetic (bf16 wrapped linears 2.745 GB vs ~1.46 GB q8 g64, 1.29 GB of a 12-14 GB estimated step, no shape needs it). The bf16 pre-registration is not edited for the quantised arm.
- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 15: (Fable 2026-10-17) The quantised rows never pool with the bf16 rows: family id lora-2b-q8 with base.format and the base-file sha256 in every row; the label rule (keep optimizer_recipe "lora" and key readers on the family id, or add "lora-q8" to OPTIMIZER_RECIPES with python/tests/test_real_ft_score_checkpoint.py:207 updated in the same commit) is decided by the Lappi lane from its ledger readers and written into the sibling pre-registration before the first quantised row.

Board criteria and marks already on the card before the ruling (2026-10-17 owner, gap and fold entries, and earlier audit marks), not previously in this brief, verbatim:

- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 7: (owner 2026-10-17, quantised arm) The D3 record gains the owner's 2026-10-17 decision: a quantised arm is trained in addition to the bf16 arm, gated by ft-a1a4ba4d's mlx measurement. The bf16 arm's pre-registration is not edited. The quantised arm gets its own pre-registration (format, quantised set, quantizer config and the quantised tensors' sha256, same recipe as D6 otherwise), written before any quantised result is read.
- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 8: (owner 2026-10-17, quantised arm) The quantised arm's torch reference is the plain-torch LoRA arm run over fake-quant weights, dequantize(quantize(W)) from mlx (no bitsandbytes or peft, D7). The 0.8B-Base smoke repeats for the admitted format, with the same 60 steps, objectives and bound (d') as the bf16 smoke.
- [ ] ft-da2c4ab1677189ec57271befbd0040c1 criterion 9: (owner 2026-10-17, quantised arm) A reduced quantised 2B-Base run, with the same truncated schedule, kill -9 resume and export path as the bf16 run, is scored by Lappi's scorer as a quick row. It is reported beside the bf16 quick row (loss trajectory, Lappi score, peak bytes, tokens/s) and cannot promote.
