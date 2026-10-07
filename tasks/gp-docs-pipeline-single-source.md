---
id: "gp-docs-pipeline-single-source"
title: "Docs pipeline: rescue evidence scripts from target-* lane dirs, one site tree, relative links, gated Pages deploy, generated test counts"
status: ready
priority: 1
severity: medium
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
  - "First, before any lane dir is pruned: every script or oracle the docs cite inside a target-* directory (e.g. target-robust/sampler_oracle.py named by docs/checkpoint-v1.md:122; live drivers like target-mathsrc/ab_*.sh, target-gpuhard/bench.sh) is copied under bench/results/<run>/scripts/ or scripts/ and the citation updated"
  - "Citations into lane dirs that no longer exist (target-baseline, target-matmul, target-lane-measure, target-lane-cpu: 43 occurrences, rg 2026-10-07) are each repointed to a tracked copy or marked 'evidence lost' with the date"
  - "docs/ stops carrying a generated mirror of site/ (reference HTML, assets, index.html, CNAME, sitemap, webmanifest); local preview serves site/ directly; docs/brand/render.sh writes to one place"
  - "Markdown sources use repo-relative links: every file:///Users/bharath or /Users/bharath reference is gone (110 across tracked and ignored Markdown, rg --no-ignore, 2026-10-07), sitegen's hard-coded absRoot rewrite is deleted, and sitegen -check fails on any file:// or absolute home path"
  - "deploy-pages.yml runs sitegen -check (or needs the test workflow) before publishing, so a doc edit without a regenerate cannot publish a stale site"
  - "Test counts and limits shown in README.md, docs/status.md, docs/backends.md and site/index.html come from one recorded run file that a Go tool (sitegen or a sibling) turns into the count blocks; hand-copied counts are removed; status.md keeps one current section and older runs move to a dated archive"
  - "site/index.html, which sitegen never generates, gets its numbers and caps from the same data, or sitegen -check validates them"
  - "Benchmark drivers that docs rely on are tracked and cited both ways: ojas-model/examples/attn_bwd_bench.rs (untracked; its header calls it 'the A/B in docs/pytorch-parity-plan.md', but no doc, task or script names it) is committed and cited, or deleted; its silent exits are fixed (on non-macOS the metal argument runs nothing and exits 0, an unknown backend name does the same, an unparseable iteration count quietly becomes 20; lines 114-128)"
---

# Task brief v1

## Title
Docs pipeline: rescue evidence scripts from target-* lane dirs, one site tree, relative links, gated Pages deploy, generated test counts

Task: gp-docs-pipeline-single-source
Type: refactor
Status: ready
Priority: 1 (High)
Severity: medium
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
