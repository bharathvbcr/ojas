---
id: "gp-cpu-gdn-and-pool-followups"
title: "CPU follow-ups: GDN forward drops wasted norm vectors and head-major round trips, the 6-thread-only pointwise cut is generalised or justified, std-only spawn mitigations are measured"
status: backlog
priority: 3
severity: low
type: perf
owner: "unassigned"
due: "none"
labels:
  - "cpu"
  - "performance"
  - "gdn"
  - "qwen35"
repositories:
  - "ojas"
planned_files:
  - "ojas-cpu/src/gdn.rs"
  - "ojas-cpu/src/pointwise.rs"
  - "ojas-cpu/src/pool/scoped.rs"
  - "docs/bench-cpu-vs-torch.md"
acceptance_criteria:
  - "The CPU GDN forward neither allocates nor computes norm vectors it discards, and writes token-major directly where the kernel allows; Exact-tier digests unchanged; an interleaved A/B is committed"
  - "The pointwise piece-count rule is expressed in terms of thread count and size generally, or the 6-thread special case is justified with an A/B on at least two thread counts"
  - "The two std-only spawn mitigations are measured and adopted or rejected with numbers in docs/bench-cpu-vs-torch.md"
---

# Task brief v1

## Title
CPU follow-ups: GDN forward drops wasted norm vectors and head-major round trips, the 6-thread-only pointwise cut is generalised or justified, std-only spawn mitigations are measured

Task: gp-cpu-gdn-and-pool-followups
Type: perf
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: cpu, performance, gdn, qwen35

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949), with leftovers from the closed gp-cpu-autograd-hot-paths [A]:
- **Wasted GDN work:** the CPU GDN forward computes and throws away the f64 norm vectors that l2norm_rows returns (ojas-cpu/src/gdn.rs:205,207), one per row for both q and k. The forward and backward write head-major scratch and then transpose (:208 to_token_major; :292-296). That is one extra full-size copy and two wasted allocations per call.
- **Machine-specific cut:** pointwise.rs:508 forces 4 pieces only when threads() == 6 and n is in [1M, 2M]. It is tuned to one machine and invisible elsewhere. The bits are unaffected.
- **Spawn cost:** the two std-only mitigations for the ~37 µs scoped spawn are recorded as open and unmeasured at docs/bench-cpu-vs-torch.md:340. Adopting a dependency or unsafe stays the user's decision.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] The CPU GDN forward neither allocates nor computes norm vectors it discards, and writes token-major directly where the kernel allows; Exact-tier digests unchanged; an interleaved A/B is committed
- [ ] The pointwise piece-count rule is expressed in terms of thread count and size generally, or the 6-thread special case is justified with an A/B on at least two thread counts
- [ ] The two std-only spawn mitigations are measured and adopted or rejected with numbers in docs/bench-cpu-vs-torch.md

## Planned files
- ojas-cpu/src/gdn.rs
- ojas-cpu/src/pointwise.rs
- ojas-cpu/src/pool/scoped.rs
- docs/bench-cpu-vs-torch.md
