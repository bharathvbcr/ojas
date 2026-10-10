---
id: "gp-docs-pipeline-single-source"
title: "Docs pipeline: rescue evidence scripts from target-* lane dirs, one site tree, relative links, gated Pages deploy, generated test counts"
status: ready
priority: 1
severity: high
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "docs"
  - "evidence"
  - "site"
  - "tooling"
  - "ci"
repositories:
  - "ojas"
planned_files:
  - "scripts/sitegen/main.go"
  - ".github/workflows/deploy-pages.yml"
  - "docs/checkpoint-v1.md"
  - "docs/bench-cpu-vs-torch.md"
  - "docs/bench-gpu-vs-torch.md"
  - "docs/status.md"
  - "docs/reference/"
  - "docs/brand/render.sh"
  - "site/index.html"
  - "README.md"
  - "bench/results/"
acceptance_criteria:
  - "First, before any lane dir is pruned: every script or oracle the docs cite inside a target-* directory is copied into a tracked home or rewritten. Urgent: target-robust/sampler_oracle.py (cited at docs/checkpoint-v1.md:122) is the last such evidence on disk; target-mathsrc and target-gpuhard are already gone and whether they were rescued is unrecorded"
  - "Citations into lane dirs that no longer exist are repointed or removed: 43 occurrences of 36 missing paths (target-matmul 16, target-baseline 15, target-attn-ab/ab and others); the closed gp-attention-kernels.md:81 also cites target-attn/attn-verify.sh, which does not exist"
  - "docs/ stops carrying a generated mirror of site/ (scripts/sitegen/main.go:133-147,170-173); local preview serves site/ directly; plot SVGs that bench/plot_all.py writes to docs/assets/plots/bench (87 files, absent from site/) are served from one place; docs/brand/render.sh:36 copies directories correctly (`cp \"$out\"/*` without -r aborts under set -eu now that a plots/ dir exists)"
  - "Markdown sources use repo-relative links: 69 file:// and 114 /Users/bharath references in tracked Markdown (README.md has 40) are gone, absRoot (main.go:31,713) is deleted, and sitegen -check fails on file:// links instead of rewriting them"
  - "deploy-pages.yml runs sitegen -check (or needs the test workflow) before publishing: run 37859886248 deployed edeaf73 while sitegen -check failed on the same SHA"
  - "Test counts and limits shown in README.md, docs/status.md, docs/backends.md and site/index.html come from one recorded run file that a Go tool (sitegen or a sibling) turns into the count blocks; hand-copied counts are removed; status.md keeps one current section and older runs move to a dated archive"
  - "site/index.html, which sitegen never generates, gets its numbers and caps from the same data, or sitegen -check validates them"
  - "Benchmark drivers that docs rely on are tracked and cited both ways: ojas-model/examples/attn_bwd_bench.rs is tracked (6a0a926) but no doc cites it back, and its silent early exits (:114-128) fail loudly"
  - "sitegen renders Markdown images: `![alt](src)` today becomes a literal '!' plus a GitHub blob link, so the 85 images in docs/bench-plots.md (83) and docs/bench-cpu-vs-torch.md (:318, :322) never render; a test pins the image output"
  - "sitegen stops publishing every top-level */README.md as a crate page (main.go:256-257): bench/README.md becomes 'crate-bench' and gemma-metal/README.md 'crate-gemma-metal'; only workspace members are crate pages"
---

# Task brief v1

## Title
Docs pipeline: rescue evidence scripts from target-* lane dirs, one site tree, relative links, gated Pages deploy, generated test counts

Task: gp-docs-pipeline-single-source
Type: refactor
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: docs, evidence, site, tooling, ci

## Repositories
- ojas

## Description
Gap audit 2026-10-07 (rg + Read + sitegen source). `gp-docs-and-bench-parity` fixes the wrong *content*; this card fixes the *pipeline* that keeps letting it drift. V = verified, I = inferred.

- **Evidence lives in gitignored lane dirs [V].** The docs cite target-* paths ~40 times. `target-robust/sampler_oracle.py` exists today but is ignored; `target-baseline`, `target-matmul`, `target-lane-measure`, `target-lane-cpu` are already gone. Pruning the lane dirs (23 GB) without this rescue destroys cited evidence. **This is the most time-sensitive item: do it before anyone cleans the target dirs.**
- **Two copies of the site [V].** `scripts/sitegen/main.go` builds `site/` from `docs/*.md` and crate READMEs, then mirrors everything into `docs/` (main.go:133-147, `staticMirror` 170-173). `diff -rq docs/reference site/reference` is clean. Every regen produces paired diffs (visible in git status). The Pages workflow publishes only `./site`.
- **Absolute paths [V].** sitegen hard-codes `absRoot = "/Users/bharath/Code/research/ojas/"` (main.go:31, :713) to rewrite links that are broken on GitHub.
- **Ungated deploy [V].** `.github/workflows/deploy-pages.yml` triggers on `docs/**` and `site/**` and uploads `./site` without running `sitegen -check`; that check runs only in test.yml's go job.
- **Hand-copied counts drift [V].** README.md:33 says 1510 passed while its per-crate numbers sum to 1347; site/index.html says 1280+ and 1286; status.md is an append-only stack ("where they disagree with this table, this table is current", :193).

Tools stay Go (sitegen is stdlib Go) per the language policy.

**Added by the second gap audit 2026-10-07** ([V] re-read by the auditor; [A] read by an audit subagent, not re-read):
- **An uncited, untracked benchmark driver [V untracked, A the rest]:** `git status` lists `ojas-model/examples/` as untracked; the file's header and its exit paths are [A]. It belongs to the attention lane's in-flight work (gp-attention-kernels), so coordinate before moving it.

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- All criteria open [A]. Criterion 1 is urgent: target-robust/ is the only remaining lane dir holding cited evidence.
- New: image syntax, stray crate pages, inverted docs/ to site/ plot assets, render.sh `cp` [A; render.sh abort is inferred, not run].
- A peer's uncommitted working tree edits scripts/sitegen/main.go, docs/ and site/; land or coordinate with that change before editing sitegen.

## Acceptance criteria
- [ ] First, before any lane dir is pruned: every script or oracle the docs cite inside a target-* directory is copied into a tracked home or rewritten. Urgent: target-robust/sampler_oracle.py (cited at docs/checkpoint-v1.md:122) is the last such evidence on disk; target-mathsrc and target-gpuhard are already gone and whether they were rescued is unrecorded
- [ ] Citations into lane dirs that no longer exist are repointed or removed: 43 occurrences of 36 missing paths (target-matmul 16, target-baseline 15, target-attn-ab/ab and others); the closed gp-attention-kernels.md:81 also cites target-attn/attn-verify.sh, which does not exist
- [ ] docs/ stops carrying a generated mirror of site/ (scripts/sitegen/main.go:133-147,170-173); local preview serves site/ directly; plot SVGs that bench/plot_all.py writes to docs/assets/plots/bench (87 files, absent from site/) are served from one place; docs/brand/render.sh:36 copies directories correctly (`cp "$out"/*` without -r aborts under set -eu now that a plots/ dir exists)
- [ ] Markdown sources use repo-relative links: 69 file:// and 114 /Users/bharath references in tracked Markdown (README.md has 40) are gone, absRoot (main.go:31,713) is deleted, and sitegen -check fails on file:// links instead of rewriting them
- [ ] deploy-pages.yml runs sitegen -check (or needs the test workflow) before publishing: run 37859886248 deployed edeaf73 while sitegen -check failed on the same SHA
- [ ] Test counts and limits shown in README.md, docs/status.md, docs/backends.md and site/index.html come from one recorded run file that a Go tool (sitegen or a sibling) turns into the count blocks; hand-copied counts are removed; status.md keeps one current section and older runs move to a dated archive
- [ ] site/index.html, which sitegen never generates, gets its numbers and caps from the same data, or sitegen -check validates them
- [ ] Benchmark drivers that docs rely on are tracked and cited both ways: ojas-model/examples/attn_bwd_bench.rs is tracked (6a0a926) but no doc cites it back, and its silent early exits (:114-128) fail loudly
- [ ] sitegen renders Markdown images: `![alt](src)` today becomes a literal '!' plus a GitHub blob link, so the 85 images in docs/bench-plots.md (83) and docs/bench-cpu-vs-torch.md (:318, :322) never render; a test pins the image output
- [ ] sitegen stops publishing every top-level */README.md as a crate page (main.go:256-257): bench/README.md becomes 'crate-bench' and gemma-metal/README.md 'crate-gemma-metal'; only workspace members are crate pages

## Planned files
- scripts/sitegen/main.go
- .github/workflows/deploy-pages.yml
- docs/checkpoint-v1.md
- docs/bench-cpu-vs-torch.md
- docs/bench-gpu-vs-torch.md
- docs/status.md
- docs/reference/
- docs/brand/render.sh
- site/index.html
- README.md
- bench/results/
