---
id: "gp-imported-research-trees"
title: "Decide the imported research trees arch02/ and gemma-metal/: 932 tracked files and 18-20 MB outside the workspace, never built in CI, absolute local paths, a Python HTTP server, .wip files"
status: backlog
priority: 2
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "housekeeping"
  - "ci"
  - "python-policy"
  - "needs-your-words"
repositories:
  - "ojas"
planned_files:
  - "arch02/"
  - "gemma-metal/"
  - "Cargo.toml"
  - ".github/workflows/test.yml"
  - "scripts/sitegen/main.go"
acceptance_criteria:
  - "A recorded decision per tree: its own repository (history preserved, link left in README) or kept here as a workspace member/exclude with a CI build job"
  - "Whatever stays builds on a fresh clone: no absolute /Users paths in any tracked Cargo.toml"
  - "Whatever stays follows the language policy: the Python HTTP server and analysis scripts are removed, ported, or confined to throwaway status with the reason; duplicate scripts and .wip files are gone"
---

# Task brief v1

## Title
Decide the imported research trees arch02/ and gemma-metal/: 932 tracked files and 18-20 MB outside the workspace, never built in CI, absolute local paths, a Python HTTP server, .wip files

Task: gp-imported-research-trees
Type: chore
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: housekeeping, ci, python-policy, needs-your-words

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949). Both research trees came in with 1fc6a03 [A, C]:
- **Size:** 932 tracked files and 108,608 lines (about 16M plus 3.8M on disk).
- **Outside the build:** each has its own `[workspace]`, CI never builds it, and `rg -uu` finds no ojas code that references it.
- **Machine-specific:** dependency paths are absolute (/Users/bharath/Code/research/tessl at arch02/metal-native/Cargo.toml:29 and gemma-metal/Cargo.toml:15), so neither builds on any other machine.
- **Unaudited dependencies:** arch02/burn-port pulls burn 0.21, candle-core, serde and rand, which the root lockfile and any dependency audit never see.
- **Python policy:** there are 27 Python files. gemma-metal/bench/serve_dflash.py is an HTTP server, a shipped-service shape the language policy forbids. gemma-metal/bench/ddtree_core.py is duplicated under bench/results/.
- **Leftovers:** two tracked .wip files (gemma-metal/src/verify_batch_impl.rs.wip, kernels/gemm_q4_mlx.metal.wip).
- **Mislabelled on the site:** sitegen publishes gemma-metal/README.md as an ojas crate page.

The decision is the user's: move each to its own repository, or make it a workspace member or exclude with a CI job.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- The crate-page criterion was owned twice; gp-docs-pipeline-single-source owns it now.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P2. Note for the decision: gemma-metal holds the only existing quantizer, which the gated gp-ft-quantized-frozen-base and gp-ft-other-base-architectures briefs depend on.
- Fine-tune briefs that depend on this one: gp-ft-quantized-frozen-base, gp-ft-other-base-architectures.

### Python inventory (2026-10-16, branch claude/trusting-ritchie-64a3c4)
Every tracked `.py` in both trees, classified against the language policy. The move-or-exclude decision above is still open, so nothing was deleted from arch02/. Labels: [V] re-read, [C] command output.
- **Deleted.** `gemma-metal/bench/results/ddtree_core.py`: byte-identical to `bench/ddtree_core.py` (`diff -q` silent [C]), and every reference names the bench/ copy [C, `rg -uu`]. `dflash_glimmer.py` and `muse_glimmer_dflash.py` were referenced only by each other [C]; the owner chose deletion over keeping them as an MLX comparator. Commit 919b1fd.
- **Ported.** `gemma-metal/bench/compare_intermediates.py` moves to `src/intermediates.rs`, run as `golden_parity --compare-intermediates GOLDEN NATIVE`. It is held to the Python's text on six fixture pairs (`tests/fixtures/compare_intermediates/`). The Python is deleted once the gemma-metal build and test pass under the machine lock.
- **Waived: `gemma-metal/bench/serve_dflash.py`.** An OpenAI-compatible HTTP server on MLX and dflash, which is a shipped-service shape the policy forbids. It stays because nothing native can replace it: `src/bin/serve.rs` is a stub, and native DFlash is not at parity. Per `docs/dflash_port.md` (MLX golden table) [V], mean_accept at bs=5 is about 0 against MLX's about 3.0, and the exact check FAILs; `docs/gates.md` ("31B DFlash (this pass)") [V] says the same. Unblock condition: native DFlash passes the 31B exact check with an MLX-like mean_accept, and serve.rs serves it. Then port the HTTP surface to Rust and delete the Python in the same change. The waiver is also in the script's docstring.
- **Stays: comparators and oracles that must run MLX, torch or ollama.**
  - `gemma-metal/bench/dflash_fast_31b.py` is the MLX DFlash reference arm, used as the comparator in `results/mlx_mtp_vs_dflash_ab_2026-07-19.md`. It is not a shipped CLI; the owner confirmed this, and `docs/speed_findings.md:15` is corrected.
  - `gemma-metal/bench/bench.py` is the ollama / mlx-lm baseline ladder.
  - `golden_intermediates.py` and `dflash_parity_golden.py` generate the MLX goldens.
  - The arch02 torch and MLX references: `mlx-baseline/*`, `export_goldens.py`, `export_step_grads.py`, `decode_reference.py`, `decode_kv_reference.py`, `arch02_kv.py`, `burn-port/scripts/export_token_bytes.py` (a fixture generator for the Rust BPB evaluator) and `bisect_continue_mps.py` (a cross-stack MPS bisect).
- **Parked, not removed.** `gemma-metal/bench/ddtree_core.py` and `test_ddtree_core.py` are the algorithmic core of DDTree, which is paused: `src/bin/bench.rs` prints "DDTree stays parked" unless verify(M) flattens. They are the reference a native port would be checked against.
- **Broken and caller-less; gated on the decision above.** Each has no reference anywhere in the tree [C, `rg -uu` on the base name]. Delete them if the tree stays; they go with it if it moves.
  - `arch02/metal-native/scripts/kernel_hardening.py`: ROOT is the ojas root, and `crates/metal-runtime` and `arch_02_value_resid/metal-native` do not exist [C].
  - `dump_golden_mixers.py` inserts `../../../../nanolab` into its import path, which resolves to the missing `research/nanolab`; nanolab lives under MLSystemsLab [C].
  - `test_kernel_builds.py` expects `crates/tessl/kernels` and `arch_02_value_resid/metal-native` [C].
  - `run_metal_crossover.py` is not broken: `src/bin/train.rs` accepts every flag it passes [V]. It is only caller-less.
- **Undecided with the tree.**
  - `arch02/metal-native/scripts/score_cuda128m.py` is called by `cuda_ref_128m.sh:167`, `ingest_cuda128m.sh:42` and `blog_results.sh:277` [C]. It is a stdlib result scorer, so port or delete it with the tree.
  - `export_coreml.py` and `arch02_kv_flat_export.py` are coremltools export probes, and only the move-or-keep decision says whether they ship.
- DevMap gap: `devmap_dead_symbols` caps Python at confidence 0.4 with walk_incomplete set (generation 7018), and it does not model a script as an entry point. The second deletion signal is therefore `rg -uu` on each file name across the tree, CI yml, scripts and docs.

## Acceptance criteria
- [ ] A recorded decision per tree: its own repository (history preserved, link left in README) or kept here as a workspace member/exclude with a CI build job
- [ ] Whatever stays builds on a fresh clone: no absolute /Users paths in any tracked Cargo.toml
- [ ] Whatever stays follows the language policy: the Python HTTP server and analysis scripts are removed, ported, or confined to throwaway status with the reason; duplicate scripts and .wip files are gone

## Planned files
- arch02/
- gemma-metal/
- Cargo.toml
- .github/workflows/test.yml
- scripts/sitegen/main.go
