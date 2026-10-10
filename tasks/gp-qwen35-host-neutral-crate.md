---
id: "gp-qwen35-host-neutral-crate"
title: "Share the platform-neutral Qwen3.5 provider half between Metal and CUDA (config, names, groups, state format, validators); bind Pending to its provider"
status: backlog
priority: 1
severity: medium
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "qwen35"
  - "cuda"
  - "metal"
  - "refactor"
  - "validation"
  - "linux"
repositories:
  - "ojas"
planned_files:
  - "ojas-qwen35/Cargo.toml"
  - "ojas-qwen35/src/step.rs"
  - "ojas-qwen35/src/config.rs"
  - "ojas-qwen35/src/names.rs"
  - "ojas-cuda/src/step.rs"
  - "ojas-cuda/src/tiny_fixture_published.rs"
  - "ojas-model/src/qwen35/params.rs"
acceptance_criteria:
  - "The tessl-independent half (config parse and refusals, name map, optimizer groups, state header/sharding format, sequence validators) lives in a crate that builds on every target; the tessl cross-check stays macOS-only"
  - "Metal and CUDA providers call one copy of the validators; a test fails if the CUDA path accepts an over-length or vision-token sequence (fails against today's CUDA validate)"
  - "On Linux a CUDA provider opens the real 2B config and safetensors index and refuses the same configs Metal refuses"
  - "ft-7162's planned path ojas-qwen35/src/cuda.rs is replaced by the cross-platform location (update that task)"
  - "A Pending is bound to the provider that made it (a per-provider identity checked in backward, not only weights_version, which starts at 0 everywhere, step.rs:402, :499); a failing test first passes one provider's Pending to another and must be refused before any device work, on Metal and on CUDA once its step lands"
  - "backward validates the external gradient before consuming the Pending (or hands the Pending back on a refusal), so a malformed dh no longer forces a full forward rerun; a test proves the same Pending runs backward after a refused bad gradient"
  - "One HF tensor-name table serves ojas-model's Tape tower (params.rs:167 hf_tensors) and the providers (names.rs tower_tensors), with a comparison test that runs on Linux CI (today only the macOS-only cpu_tape_spec.rs:7 ties them); CUDA's config re-parse (tiny_fixture_published.rs:188-312) is deleted"
  - "config.json and the index file are read with a size bound before parsing (config.rs:314 and names.rs:254 read_to_string unbounded; the 16 MiB cap applies after), num_hidden_layers is bounded before the layer_types vector is built (config.rs:259-265, :463-479 can collect ~4 GiB), and rms_norm_eps is validated as f32 (config.rs:662, :825; 1e-50 becomes 0)"
---

# Task brief v1

## Title
Share the platform-neutral Qwen3.5 provider half between Metal and CUDA (config, names, groups, state format, validators); bind Pending to its provider

Task: gp-qwen35-host-neutral-crate
Type: refactor
Status: backlog
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: qwen35, cuda, metal, refactor, validation, linux

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949).

- **Root cause [A]:** ojas-qwen35/Cargo.toml:14-23 makes tessl and ojas-model macOS-only dependencies, and the crate compiles to nothing elsewhere. The config parser, `Sequence`, `validate_external_grad` and the HF tensor names are unreachable from Linux.
- **Drift already happened [V]:** ojas-cuda/src/step.rs:52-140 is a DevMap exact clone of ojas-qwen35/src/step.rs:63-160 (`Sequence::validate`), minus the refusal of reserved vision-token ids ('vision tokens switch transformers to multimodal positions', ojas-qwen35/src/step.rs:110). It also returns OjasError::Shape/OutOfRange where the original returns Qwen35Error. Once the CUDA step runs real kernels it will accept ids the Metal path refuses.
- **Second copy [A]:** CUDA re-parses config.json in ojas-cuda/src/tiny_fixture_published.rs:188-312.
- **Two independent HF name tables [A]:** ojas-model/src/qwen35/params.rs:167 (hf_tensors) and ojas-qwen35/src/names.rs (tower_tensors). The only test tying them (ojas-qwen35/tests/cpu_tape_spec.rs:7) is macOS-only and reads ../../tessl fixtures, so Linux CI never compares them.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- This brief now carries GitPulse board card ft-e55afcb5d52094ecb9098f46b557a269 ('Share the platform-neutral Qwen3.5 provider half...', backlog, p1). It is the same work. The card's six criteria are carried verbatim, and the HF-name-table and bounded-config-read criteria from this audit are added, so the two reconcile as one item. The card predates this brief.
- Severity lowered from high to medium [A]. The CUDA validate copy is not reachable from any non-test path today: Qwen35Step::forward refuses before it (step.rs:324-366) and no crate depends on ojas-cuda. The copy is also not exact; letter_scale handling differs (cuda step.rs:127-128 vs qwen35 step.rs:141-146). The vision-token gap is CONFIRMED [V].
- New: unbounded config reads and a layer-count allocation before validation [A].

## Acceptance criteria
- [ ] The tessl-independent half (config parse and refusals, name map, optimizer groups, state header/sharding format, sequence validators) lives in a crate that builds on every target; the tessl cross-check stays macOS-only
- [ ] Metal and CUDA providers call one copy of the validators; a test fails if the CUDA path accepts an over-length or vision-token sequence (fails against today's CUDA validate)
- [ ] On Linux a CUDA provider opens the real 2B config and safetensors index and refuses the same configs Metal refuses
- [ ] ft-7162's planned path ojas-qwen35/src/cuda.rs is replaced by the cross-platform location (update that task)
- [ ] A Pending is bound to the provider that made it (a per-provider identity checked in backward, not only weights_version, which starts at 0 everywhere, step.rs:402, :499); a failing test first passes one provider's Pending to another and must be refused before any device work, on Metal and on CUDA once its step lands
- [ ] backward validates the external gradient before consuming the Pending (or hands the Pending back on a refusal), so a malformed dh no longer forces a full forward rerun; a test proves the same Pending runs backward after a refused bad gradient
- [ ] One HF tensor-name table serves ojas-model's Tape tower (params.rs:167 hf_tensors) and the providers (names.rs tower_tensors), with a comparison test that runs on Linux CI (today only the macOS-only cpu_tape_spec.rs:7 ties them); CUDA's config re-parse (tiny_fixture_published.rs:188-312) is deleted
- [ ] config.json and the index file are read with a size bound before parsing (config.rs:314 and names.rs:254 read_to_string unbounded; the 16 MiB cap applies after), num_hidden_layers is bounded before the layer_types vector is built (config.rs:259-265, :463-479 can collect ~4 GiB), and rms_norm_eps is validated as f32 (config.rs:662, :825; 1e-50 becomes 0)

## Planned files
- ojas-qwen35/Cargo.toml
- ojas-qwen35/src/step.rs
- ojas-qwen35/src/config.rs
- ojas-qwen35/src/names.rs
- ojas-cuda/src/step.rs
- ojas-cuda/src/tiny_fixture_published.rs
- ojas-model/src/qwen35/params.rs
