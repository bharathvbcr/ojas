---
id: "gp-ft-frozen-params-and-lora"
title: "Frozen parameters and LoRA on the Tape: frozen leaves with no gradient, a dX-only linear backward, PEFT-initialised LoRA on every Qwen3.5 linear (attention, GDN linear_attn, MLP), merge/unmerge, Fable's parity bounds (a')-(c')"
status: ready
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "lora"
  - "autograd"
  - "qwen35"
  - "metal"
  - "parity"
repositories:
  - "ojas"
planned_files:
  - "ojas-autograd/src/tape.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-model/src/lora.rs"
  - "ojas-model/src/qwen35/params.rs"
  - "ojas-model/src/qwen35/forward.rs"
  - "ojas-qwen35/src/step.rs"
  - "ojas-oracle/"
acceptance_criteria:
  - "The Tape can mark a leaf frozen: no gradient buffer, excluded from grads(), optimizer state, the clip norm and per-step finite checks. Today Tape::leaf records every leaf as Rec::Leaf (ojas-autograd/src/tape.rs:362-365) and the walk keeps a gradient for every leaf [V]"
  - "Backend gains linear_backward_input (dX only) with CPU and Metal bodies, and a frozen linear never computes dW. Today linear_backward always returns (grad_input, grad_weight) (ojas-core/src/backend.rs:653-659) [V]. Autocast/Gated wrappers forward it (gp-backend-wrapper-forwarding). A test asserts dX equals the full path bit for bit and that no dW buffer is allocated (budget delta 0); an A/B benchmark at the 2B layer shapes reports the backward-time saving"
  - "A LoRA linear computes y = x W^T + (alpha/r) * (x A^T) B^T with r, alpha and dropout configurable; init as PEFT does: kaiming_uniform_(A, a=sqrt(5)), B = 0, scaling = alpha/r [V by Fable from peft/tuners/lora/layer.py]. Generic over Backend, not Qwen3.5-specific"
  - "Targets are chosen by HF tensor name from the shared name table (gp-qwen35-host-neutral-crate) and the pre-registered set covers every linear: attention q/k/v/o, GDN linear_attn in_proj_qkv / in_proj_z / out_proj, MLP gate/up/down (18 of 24 layers are GDN). Names are read from the checkpoint's tensor index, not from memory"
  - "Bound (a'), CPU f32 on the widened tiny fixture: LoRA forward loss <= 1e-5 rel against plain torch; dX, dA, dB within 1e-5 of each tensor's max; frozen leaves have no gradient (absent, not zero)"
  - "Bound (b'): a 20-step tiny trajectory, torch fp32 CPU vs ojas CPU Exact: per-step loss <= 1e-5 rel for steps 0-5 and <= 1e-4 to step 20; final A, B within 1e-5 of each tensor's max; two ojas runs bit-identical; Metal vs CPU <= 2^-8 on loss with bf16 operands"
  - "Bound (c'): Tape-with-LoRA at B = 0 gives a loss bit-equal to Tape-without-LoRA on CPU (identity at init)"
  - "merge folds (alpha/r) B A into W in f32 and reproduces unmerged logits within the stated bound; unmerge restores the f32 W bit for bit"
  - "Provider freeze semantics are fixed or documented: today lr_scale 0 still feeds grad_sq_norm (ojas-qwen35/src/step.rs:572-578) and updates moments (step.rs:584-586) [V]. Under Fable D1 the provider stays full-FT, so the minimum is a doc and a test pinning the behaviour; the fix (exclude scale-0 entries from the norm and the moments) is preferred"
---

# Task brief v1

## Title
Frozen parameters and LoRA on the Tape: frozen leaves with no gradient, a dX-only linear backward, PEFT-initialised LoRA on every Qwen3.5 linear (attention, GDN linear_attn, MLP), merge/unmerge, Fable's parity bounds (a')-(c')

Task: gp-ft-frozen-params-and-lora
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: fine-tune, lora, autograd, qwen35, metal, parity

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); updated by Fable's ruling the same day (Lappi AUDIT/lora-2b-2026-10-09/fable-ft-decisions.md, D1, D6, D7). There is no LoRA code in ojas, tessl or Lappi (rg -uu) [V], and no way to freeze a parameter on the Tape.

Decided (Fable D1, 2026-10-09): LoRA lives on the Tape (ojas-model qwen35 over ojas-core Backend, CPU + Metal), not in tessl's fused step. On the Tape LoRA is composition (two small f32 leaves and existing linear/add ops) plus one Backend method; rule 6 holds because ojas-metal's hybrid ops call tessl kernels. Reversal: if gp-qwen35-2b-metal-tape-run fails its parity bound on real weights, or the Tape's tokens/s at T=2048 stays below half the provider's after gp-ft-step-memory-and-throughput, LoRA moves into tessl's fused step.

The parity oracle is the plain-torch LoRA arm in Lappi's python/qd_train (Fable D7), not peft and not mlx-lm (mlx-lm uses a flat scale 20 and uniform A init).

Depends on: gp-ft-training-path-correctness (widened fixture, label mask), gp-qwen35-host-neutral-crate (name table), gp-backend-wrapper-forwarding. Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] The Tape can mark a leaf frozen: no gradient buffer, excluded from grads(), optimizer state, the clip norm and per-step finite checks. Today Tape::leaf records every leaf as Rec::Leaf (ojas-autograd/src/tape.rs:362-365) and the walk keeps a gradient for every leaf [V]
- [ ] Backend gains linear_backward_input (dX only) with CPU and Metal bodies, and a frozen linear never computes dW. Today linear_backward always returns (grad_input, grad_weight) (ojas-core/src/backend.rs:653-659) [V]. Autocast/Gated wrappers forward it (gp-backend-wrapper-forwarding). A test asserts dX equals the full path bit for bit and that no dW buffer is allocated (budget delta 0); an A/B benchmark at the 2B layer shapes reports the backward-time saving
- [ ] A LoRA linear computes y = x W^T + (alpha/r) * (x A^T) B^T with r, alpha and dropout configurable; init as PEFT does: kaiming_uniform_(A, a=sqrt(5)), B = 0, scaling = alpha/r [V by Fable from peft/tuners/lora/layer.py]. Generic over Backend, not Qwen3.5-specific
- [ ] Targets are chosen by HF tensor name from the shared name table (gp-qwen35-host-neutral-crate) and the pre-registered set covers every linear: attention q/k/v/o, GDN linear_attn in_proj_qkv / in_proj_z / out_proj, MLP gate/up/down (18 of 24 layers are GDN). Names are read from the checkpoint's tensor index, not from memory
- [ ] Bound (a'), CPU f32 on the widened tiny fixture: LoRA forward loss <= 1e-5 rel against plain torch; dX, dA, dB within 1e-5 of each tensor's max; frozen leaves have no gradient (absent, not zero)
- [ ] Bound (b'): a 20-step tiny trajectory, torch fp32 CPU vs ojas CPU Exact: per-step loss <= 1e-5 rel for steps 0-5 and <= 1e-4 to step 20; final A, B within 1e-5 of each tensor's max; two ojas runs bit-identical; Metal vs CPU <= 2^-8 on loss with bf16 operands
- [ ] Bound (c'): Tape-with-LoRA at B = 0 gives a loss bit-equal to Tape-without-LoRA on CPU (identity at init)
- [ ] merge folds (alpha/r) B A into W in f32 and reproduces unmerged logits within the stated bound; unmerge restores the f32 W bit for bit
- [ ] Provider freeze semantics are fixed or documented: today lr_scale 0 still feeds grad_sq_norm (ojas-qwen35/src/step.rs:572-578) and updates moments (step.rs:584-586) [V]. Under Fable D1 the provider stays full-FT, so the minimum is a doc and a test pinning the behaviour; the fix (exclude scale-0 entries from the norm and the moments) is preferred

## Planned files
- ojas-autograd/src/tape.rs
- ojas-core/src/backend.rs
- ojas-cpu/src/backend.rs
- ojas-metal/src/backend.rs
- ojas-model/src/lora.rs
- ojas-model/src/qwen35/params.rs
- ojas-model/src/qwen35/forward.rs
- ojas-qwen35/src/step.rs
- ojas-oracle/

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: merged at efa507f [V]. Header and body say ready, the board review; every criterion is still [ ] and there is no progress section. Propose: board review stays until (b') and the gaps below are ruled on.
- Criteria: 1 partial: frozen leaves take no gradient (tape.rs:384, :1460-1462; tests frozen.rs:123-228) [A], but the optimizer-state, clip-norm and finite-check clauses have no consumer because no Qwen3.5 Tape trainer exists (gp-ft-trainer-loop) [A]; 2 partial: dX-only exists on the trait, CPU, Metal, Autocast and Gated with bit-parity and no-dW-charge tests, but the A/B bench (frozen_linear_bench.rs) is ignored and no result is committed [A]; 3, 4, 5, 7 met [A]; 6 partial: losses and bit-identical rerun pinned, but the final-adapter bound was relaxed to max(1e-5, 3 x torch's own fp32 noise floor) (qwen35_lora.rs, `b_prime_final_adapters_are_within_the_oracles_own_noise_floor`) and the literal 1e-5 test was dropped in 60efedc [V]; the brief needs amending or the relaxation ratified; 8 partial: LoraLinear merge/unmerge tested, the model-level fold exists only inside a test [A]; 9 met as documented, its pin gpu_parity.rs:654 is #[ignore] [A].
- New, P2 [V]: no model-level LoRA test runs at scaling 1. Every (a'), (b'), (c') run uses RANK 4, ALPHA 8 (qwen35_common/lora.rs:21-22); the (b') sensitivity mutants vary alpha only by +1% and +0.1%. The pre-registered recipe is r = 16, alpha = 16, where `scaled_b` (lora.rs:308-316) returns B unscaled and LoraForward records no consts: the path the run takes is the one path no parity test exercises.
- New, P2 [V]: no adapter read or write in src (`rg -uu lora_A|lora_B` over ojas-model, ojas-qwen35, ojas-io src finds only a doc comment, lora.rs:289); `fixture_sites` is a test helper. Shared-init parity and the kill -9 resume need it. Home: gp-ft-trainer-loop (checkpoint) or gp-ft-export (PEFT format).
- Known, P1 [V]: a frozen tied head still gets a weight gradient: `Tape::fused_ce` always calls with want_grad = true and requires both gradients (tape.rs:955-990). Tracked by ft-2bbced18 (BREAK test frozen_adversarial.rs:280).
- Stale: line 59 "no LoRA code ... no way to freeze a parameter" and the "Today ..." clauses of criteria 1, 2 and 9 are false; cited lines moved (tape.rs:362-365 -> :365-386, backend.rs:653-659 -> :659-690, step.rs:572-586 -> :655-686) [A].

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
- **Criterion 6 amended, relaxation ratified:** the (b') final-adapter bound is max(1e-5, 3 x torch's own fp32-vs-fp64 floor), as 60efedc made it; a bound below the oracle's own noise is not a bound on ojas, and the test adds the stricter clause that ojas is closer to fp64 than torch on a majority of tensors (ojas-model/tests/qwen35_lora.rs:308-347). The printed floor column from a real run goes into this log.
- **A scaling-1 parity test is required before any LoRA number is read as evidence** (before the 0.8B smoke). A second oracle fixture dir at r = 4, alpha = 4 (the scaling-2 fixture is not edited) and a test file pinning (a') one-step loss and dA/dB within 1e-5, (b') the 20-step losses and a bit-identical rerun, (c') identity at B = 0, the Metal-vs-CPU adapter-gradient clause, and a structural pin that no scale constant exists at scaling 1 (`scaled_b`, lora.rs:308-316).
- **Board: stays in review** until the criterion 6 floor column, the scaling-1 test and a recorded run of gpu_parity.rs:654 are logged. Waived here, owned elsewhere: criterion 2's A/B bench (gp-ft-step-memory-and-throughput's interleaved rows) and criterion 8's model-level fold (gp-ft-export).
