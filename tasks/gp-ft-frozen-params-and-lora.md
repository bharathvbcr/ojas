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
