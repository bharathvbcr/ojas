---
id: "gp-ft-frozen-params-and-lora"
title: "Frozen parameters and LoRA on the Tape: frozen leaves with no gradient, a dX-only linear backward, PEFT-initialised LoRA on every Qwen3.5 linear (attention, GDN linear_attn, MLP), merge/unmerge, Fable's parity bounds (a')-(c')"
status: review
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
Status: review
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: fine-tune, lora, autograd, qwen35, metal, parity

## Repositories
- ojas

## Description
Filed by the 2026-10-09 fine-tune audit (at d431949); updated by Fable's ruling the same day (Lappi AUDIT/lora-2b-2026-10-09/fable-ft-decisions.md, D1, D6, D7). ~~There is no LoRA code in ojas, tessl or Lappi (rg -uu) [V], and no way to freeze a parameter on the Tape.~~ (corrected 2026-10-17 at 5b67cdd: ojas has ojas-model/src/lora.rs and forward_loss_lora forward.rs:275, Tape::frozen_leaf is tape.rs:390, and Lappi has python/qd_train/lora.py.)

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
- Known, P1 [V]: ~~a frozen tied head still gets a weight gradient: `Tape::fused_ce` always calls with want_grad = true and requires both gradients (tape.rs:955-990).~~ (corrected 2026-10-17 at 5b67cdd: fixed on main; Tape::fused_ce is tape.rs:968, its gradients are optional per operand, CeGrads is ojas-core/src/backend.rs:1748, and the frozen-head test is not ignored, frozen_adversarial.rs:280.) Tracked by ft-2bbced18 (BREAK test frozen_adversarial.rs:280).
- Stale: line 59 "no LoRA code ... no way to freeze a parameter" and the "Today ..." clauses of criteria 1, 2 and 9 are false; cited lines moved (tape.rs:362-365 -> :365-386, backend.rs:653-659 -> :659-690, step.rs:572-586 -> :655-686) [A].

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
- **Criterion 6 amended, relaxation ratified:** the (b') final-adapter bound is max(1e-5, 3 x torch's own fp32-vs-fp64 floor), as 60efedc made it; a bound below the oracle's own noise is not a bound on ojas, and the test adds the stricter clause that ojas is closer to fp64 than torch on a majority of tensors (ojas-model/tests/qwen35_lora.rs:308-347). The printed floor column from a real run goes into this log.
- **A scaling-1 parity test is required before any LoRA number is read as evidence** (before the 0.8B smoke). A second oracle fixture dir at r = 4, alpha = 4 (the scaling-2 fixture is not edited) and a test file pinning (a') one-step loss and dA/dB within 1e-5, (b') the 20-step losses and a bit-identical rerun, (c') identity at B = 0, the Metal-vs-CPU adapter-gradient clause, and a structural pin that no scale constant exists at scaling 1 (`scaled_b`, lora.rs:308-316).
- **Board: stays in review** until the criterion 6 floor column, the scaling-1 test and a recorded run of gpu_parity.rs:654 are logged. Waived here, owned elsewhere: criterion 2's A/B bench (gp-ft-step-memory-and-throughput's interleaved rows) and criterion 8's model-level fold (gp-ft-export).
- **Recorded run (2026-10-15, main 132b46e, M5 Pro, under mac_heavy, `cargo test --release -p ojas-qwen35 --test gpu_save_state_bank --test gpu_parity -- --ignored --exact ... --test-threads=1`):** `gpu_save_state_refuses_a_half_filled_gradient_bank ... ok` (1 passed); `gpu_tiny_scale_zero_entry_still_feeds_the_norm ... ok` (1 passed, 9 filtered out).
- **Scaling-1 parity landed for review (2026-10-15, branch gitpulse/lora-scaling-one-parity, bc08eeb on main 132b46e):** fixture `qwen35_tiny_lora_s1` (`qwen35_lora_fixture.py --alpha 4`: rank 4, alpha 4, same seed as the scaling-2 fixture, so the same A, B and batch); `ojas-model/tests/qwen35_lora_scale1.rs` holds (a'), (b') losses and bit-identical rerun, (b') finals, (c') and the LoRA-linear clause at scaling 1, plus a structural pin (LoraForward charges no scale constant at scaling 1 and does at 2; `scaled_b(1, None)` is B). The clauses now live once in `qwen35_common/lora.rs` per fixture, and the adversarial mutants are scored with the same trajectory and bound.
  - Run (M5 Pro, under mac_heavy, `cargo test --release -p ojas-model --test qwen35_lora --test qwen35_lora_scale1 --test qwen35_lora_adversarial --test qwen35_lora_metal -- --test-threads=1`; clippy -D warnings clean): qwen35_lora 15 passed, qwen35_lora_scale1 7 passed, qwen35_lora_adversarial 22 passed, qwen35_lora_metal 4 passed (Metal vs CPU at scaling 1: loss rel 2.62e-7).
  - Scaling 1: a' loss rel 1.75e-7, worst adapter gradient 1.83e-6 of peak; worst final adapter vs torch 5.37e-4 of peak, 0.67 of its bound; ojas closer to fp64 than torch fp32 on 21 of 26 tensors. Scaling 2: worst final 2.45e-4, 0.76 of bound, closer on 24 of 26.
  - The structural pin bites: with LoraForward building constants at scaling 1, or with scaled_b's short-circuit removed, `scaling_one_records_no_scale_constant` fails (qwen35_lora_scale1.rs:59 and :66); restored, it passes.
- **Criterion 6, the printed floor column** (the run above; ratified bound max(1e-5, 3 x floor)). The literal 1e-5 against torch would reject 20 of 26 tensors at scaling 2 and 26 of 26 at scaling 1, every one within three times torch fp32's own distance from fp64:
  - `b_prime_final_adapters_are_within_the_oracles_own_noise_floor`, qwen35_tiny_lora (scaling 2): tensor: ojas vs torch / torch vs fp64 (floor) / ojas vs fp64, each of the tensor's peak
    - l0.linear_attn.in_proj_qkv.lora_A.weight: 2.15e-5 / 1.32e-5 / 3.02e-5
    - l0.linear_attn.in_proj_qkv.lora_B.weight: 4.48e-5 / 4.16e-5 / 6.15e-5
    - l0.linear_attn.in_proj_z.lora_A.weight: 3.49e-6 / 8.26e-6 / 6.75e-6
    - l0.linear_attn.in_proj_z.lora_B.weight: 1.21e-5 / 2.46e-5 / 2.05e-5
    - l0.linear_attn.out_proj.lora_A.weight: 1.22e-5 / 1.95e-5 / 1.82e-5
    - l0.linear_attn.out_proj.lora_B.weight: 1.65e-5 / 4.11e-5 / 2.46e-5
    - l0.mlp.gate_proj.lora_A.weight: 6.91e-6 / 8.55e-6 / 6.79e-6
    - l0.mlp.gate_proj.lora_B.weight: 1.07e-5 / 3.79e-5 / 3.12e-5
    - l0.mlp.up_proj.lora_A.weight: 1.85e-5 / 1.51e-5 / 7.79e-6
    - l0.mlp.up_proj.lora_B.weight: 2.22e-5 / 2.89e-5 / 2.43e-5
    - l0.mlp.down_proj.lora_A.weight: 1.58e-5 / 2.99e-5 / 1.41e-5
    - l0.mlp.down_proj.lora_B.weight: 7.46e-6 / 1.79e-5 / 1.15e-5
    - l1.self_attn.q_proj.lora_A.weight: 1.44e-5 / 2.08e-5 / 8.60e-6
    - l1.self_attn.q_proj.lora_B.weight: 6.72e-5 / 4.74e-4 / 4.51e-4
    - l1.self_attn.k_proj.lora_A.weight: 3.13e-6 / 6.67e-6 / 6.46e-6
    - l1.self_attn.k_proj.lora_B.weight: 1.28e-5 / 4.22e-5 / 3.82e-5
    - l1.self_attn.v_proj.lora_A.weight: 4.93e-6 / 5.50e-6 / 4.70e-6
    - l1.self_attn.v_proj.lora_B.weight: 1.35e-5 / 2.15e-5 / 1.93e-5
    - l1.self_attn.o_proj.lora_A.weight: 2.45e-4 / 5.02e-4 / 2.57e-4
    - l1.self_attn.o_proj.lora_B.weight: 2.20e-5 / 4.70e-5 / 2.51e-5
    - l1.mlp.gate_proj.lora_A.weight: 1.35e-4 / 3.37e-4 / 2.01e-4
    - l1.mlp.gate_proj.lora_B.weight: 9.29e-5 / 2.31e-4 / 1.38e-4
    - l1.mlp.up_proj.lora_A.weight: 4.29e-6 / 1.86e-5 / 1.49e-5
    - l1.mlp.up_proj.lora_B.weight: 1.12e-4 / 2.76e-4 / 1.65e-4
    - l1.mlp.down_proj.lora_A.weight: 1.80e-5 / 4.26e-5 / 2.46e-5
    - l1.mlp.down_proj.lora_B.weight: 1.76e-5 / 2.00e-5 / 1.58e-5
  - `b_prime_final_adapters_are_within_the_oracles_noise_floor_at_scaling_one`, qwen35_tiny_lora_s1 (scaling 1): tensor: ojas vs torch / torch vs fp64 (floor) / ojas vs fp64, each of the tensor's peak
    - l0.linear_attn.in_proj_qkv.lora_A.weight: 1.67e-5 / 1.07e-5 / 2.15e-5
    - l0.linear_attn.in_proj_qkv.lora_B.weight: 4.10e-5 / 5.18e-5 / 5.76e-5
    - l0.linear_attn.in_proj_z.lora_A.weight: 1.01e-5 / 2.49e-5 / 2.47e-5
    - l0.linear_attn.in_proj_z.lora_B.weight: 3.92e-5 / 5.38e-5 / 2.86e-5
    - l0.linear_attn.out_proj.lora_A.weight: 5.37e-4 / 5.70e-4 / 6.06e-5
    - l0.linear_attn.out_proj.lora_B.weight: 1.70e-4 / 1.73e-4 / 1.97e-5
    - l0.mlp.gate_proj.lora_A.weight: 3.79e-5 / 2.74e-5 / 1.41e-5
    - l0.mlp.gate_proj.lora_B.weight: 4.53e-5 / 8.60e-5 / 4.07e-5
    - l0.mlp.up_proj.lora_A.weight: 1.75e-5 / 1.60e-5 / 9.68e-6
    - l0.mlp.up_proj.lora_B.weight: 3.52e-5 / 3.43e-5 / 2.07e-5
    - l0.mlp.down_proj.lora_A.weight: 2.79e-5 / 2.15e-5 / 1.20e-5
    - l0.mlp.down_proj.lora_B.weight: 2.05e-5 / 2.17e-5 / 1.18e-5
    - l1.self_attn.q_proj.lora_A.weight: 1.49e-5 / 2.32e-5 / 2.34e-5
    - l1.self_attn.q_proj.lora_B.weight: 1.20e-4 / 7.51e-4 / 7.04e-4
    - l1.self_attn.k_proj.lora_A.weight: 1.78e-5 / 1.76e-5 / 7.98e-6
    - l1.self_attn.k_proj.lora_B.weight: 2.41e-5 / 3.12e-5 / 3.36e-5
    - l1.self_attn.v_proj.lora_A.weight: 2.07e-5 / 2.15e-5 / 8.80e-6
    - l1.self_attn.v_proj.lora_B.weight: 3.48e-5 / 3.64e-5 / 1.80e-5
    - l1.self_attn.o_proj.lora_A.weight: 4.35e-5 / 5.08e-5 / 6.63e-5
    - l1.self_attn.o_proj.lora_B.weight: 1.81e-5 / 1.71e-5 / 1.64e-5
    - l1.mlp.gate_proj.lora_A.weight: 5.48e-5 / 1.52e-4 / 9.72e-5
    - l1.mlp.gate_proj.lora_B.weight: 8.62e-5 / 2.93e-4 / 2.07e-4
    - l1.mlp.up_proj.lora_A.weight: 1.21e-5 / 3.44e-5 / 3.18e-5
    - l1.mlp.up_proj.lora_B.weight: 4.39e-5 / 3.36e-5 / 2.79e-5
    - l1.mlp.down_proj.lora_A.weight: 4.38e-5 / 4.43e-5 / 2.15e-5
    - l1.mlp.down_proj.lora_B.weight: 7.48e-5 / 1.44e-4 / 6.91e-5

### Per-gradient flags on the fused cross-entropy (ft-2bbced18, 2026-10-15)
Landed for review on branch gitpulse/fused-ce-want-flags (off the tied-table branch's merge of main, 36f1994), closing the Known P1 above.
- `Backend::linear_cross_entropy_mean` takes `CeGrads { input, weight }` (ojas-core, beside `CeChunk`) instead of one `want_grad: bool`; each of `LinearCe`'s gradients is present exactly when its flag is set, and one not asked for is neither computed nor charged. CPU (`fused_ce.rs`: optional `grad_x`/`grad_w` slices, scratch per requested GEMM), Metal (`LceGeom.grads`; device and host agree on the output order loss, input's, weight's), wgpu, Autocast and the capi gate follow. The C ABI never carried the flag, so tessl_torch is unaffected.
- The Tape asks for the gradient of each operand that is not frozen (`Tape::fused_ce`), so a frozen tied head takes no `[V, d]` gradient; it refuses a backend that returns a gradient not asked for, or omits one asked for.
- Proof: `frozen_adversarial.rs` `a_frozen_fused_head_allocates_no_head_sized_gradient` is no longer ignored. On the unmodified tree (36f1994) it fails: "frozen head peak 2655272 B is not below one head-sized gradient 2097152 B"; on this branch it passes.
- New and widened tests: Metal `each_gradient_alone_has_the_bits_of_both_and_charges_only_itself` (each combination's slots carry the both-gradient call's bits; the charge differs from the loss-only call by exactly the requested output and its tile); the Tape contract test adds a frozen head with an unasked weight gradient and with none returned; the CPU heap/budget/bf16-table sweeps, the wgpu composition test and the oracle's bf16 contract run all four combinations (`CeGrads::EACH`), so the oracle's table-lane count is 2 x (1 + 1 + 8) = 20 mixed calls.
- Not reproduced here: BUILD_REPORT.md's note that `ojas-model/src/trainer.rs` binds frozen moment slots with `g.param` (no such line in this tree; the file is untracked in the main checkout, not in this branch).
- Run (M5 Pro, under mac_heavy, 2026-10-15T21:42Z, cw-check2): `cargo clippy --workspace --all-targets --release -- -D warnings` clean; `cargo test --release --no-fail-fast -p ojas-core -p ojas-autograd -p ojas-cpu -p ojas-metal -p ojas-wgpu -p ojas-model -p ojas-capi -p ojas-oracle -p ojas-qwen35`: 1,385 passed, 2 failed, 50 ignored over 173 test binaries; the 2 were the oracle table-lane count pins (204 vs 212 mixed calls, the four-combination sweep), updated and rerun below; `gpu_parity -- --ignored gpu_tiny`: 8 passed.
- The count pins (`ojas-oracle/tests/bf16_operands.rs:22`, `ojas-metal/tests/bf16_operands.rs:78`) now read `2 * (2 * 10 + 2 * 38) + 2 * 10`. Their rerun (clippy on ojas-oracle and ojas-metal, both `bf16_operands` tests) was queued for the heavy-job lock when this was committed; its result is the next line.

### Fable ruling (2026-10-17)

Synced from the GitPulse board by the 2026-10-17 brief sync; the board is canonical (Fable's final ruling of 2026-10-17, section 5), this section is the long-form record. Board card ft-6fb212ce8744aef3bdf4e5c6ff1fdf44, revision 8: status review, priority 1, severity high. Criterion numbers are the board's, counted from 0 in the order of the card's acceptance criteria.

Ruling set: status review (unchanged; owner verify to done), priority 1.

Marks (the text of the existing criterion is otherwise unchanged on the board; nothing deleted, no threshold moved):

- ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 3: [MET 2026-10-17: LoraKind::suffix (ojas-model/src/qwen35/params.rs:697-708) equals Lappi LORA_TARGET_SUFFIXES (python/qd_train/lora.py:89-100) ten for ten; 150 linears at 2B by construction; names read from the tensor index (lora_targets :771)]
- ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 9: [HOMED 2026-10-17 at ft-2bbced18 criteria 3 and 4 (the end-to-end dW-skip share and the interleaved rows); criterion 2's benchmark clause is waived here]

Board criteria and marks already on the card before the ruling (2026-10-17 owner, gap and fold entries, and earlier audit marks), not previously in this brief, verbatim:

- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 0: [MET — verified in source] The Tape can mark a leaf frozen: no gradient buffer, excluded from grads(), optimizer state, the clip norm and per-step finite checks. Today Tape::leaf records every leaf as Rec::Leaf (ojas-autograd/src/tape.rs:362-365) and the walk keeps a gradient for every leaf [V]
- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 1: [MET — verified in source; A/B record NOT committed, see criterion 10] Backend gains linear_backward_input (dX only) with CPU and Metal bodies, and a frozen linear never computes dW. Today linear_backward always returns (grad_input, grad_weight) (ojas-core/src/backend.rs:653-659) [V]. Autocast/Gated wrappers forward it (gp-backend-wrapper-forwarding). A test asserts dX equals the full path bit for bit and that no dW buffer is allocated (budget delta 0); an A/B benchmark at the 2B layer shapes reports the backward-time saving
- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 2: [MET — verified in source] A LoRA linear computes y = x W^T + (alpha/r) * (x A^T) B^T with r, alpha and dropout configurable; init as PEFT does: kaiming_uniform_(A, a=sqrt(5)), B = 0, scaling = alpha/r [V by Fable from peft/tuners/lora/layer.py]. Generic over Backend, not Qwen3.5-specific
- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 4: [MET — verified in source] Bound (a'), CPU f32 on the widened tiny fixture: LoRA forward loss <= 1e-5 rel against plain torch; dX, dA, dB within 1e-5 of each tensor's max; frozen leaves have no gradient (absent, not zero)
- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 5: [MET — verified in source, under the 2026-10-14 amendment] Bound (b'): a 20-step tiny trajectory, torch fp32 CPU vs ojas CPU Exact: per-step loss <= 1e-5 rel for steps 0-5 and <= 1e-4 to step 20; final A, B: each tensor within max(1e-5, 3x torch's own fp32-vs-fp64 gap) of each tensor's max, against torch fp32 and against fp64, and ojas closer to fp64 than torch fp32 on a majority of tensors; two ojas runs bit-identical; Metal vs CPU <= 2^-8 on loss with bf16 operands
- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 6: [MET — verified in source] Bound (c'): Tape-with-LoRA at B = 0 gives a loss bit-equal to Tape-without-LoRA on CPU (identity at init)
- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 7: [MET — verified in source] merge folds (alpha/r) B A into W in f32 and reproduces unmerged logits within the stated bound; unmerge restores the f32 W bit for bit
- [ ] ft-6fb212ce8744aef3bdf4e5c6ff1fdf44 criterion 8: [MET — verified in source, as doc + pin] Provider freeze semantics are fixed or documented: today lr_scale 0 still feeds grad_sq_norm (ojas-qwen35/src/step.rs:572-578) and updates moments (step.rs:584-586) [V]. Under Fable D1 the provider stays full-FT, so the minimum is a doc and a test pinning the behaviour; the fix (exclude scale-0 entries from the norm and the moments) is preferred
