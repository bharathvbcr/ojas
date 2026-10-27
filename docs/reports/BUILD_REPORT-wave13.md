# BUILD_REPORT — wave 13 (scratch lifetime)

Date: 2026-10-26. Branch `gp/w13-scratch` at `85ffe0d`, uncommitted. Primary checkout moved to `d2922fb` during the wave and was not edited. This file is the wave record. `docs/reports/BUILD_REPORT.md` remains the wave-12 report.

Roles available in this session were Grok subagents, not Claude Opus or Sonnet. The implementation and the final review were done in this session. The adversarial pass was a read-only `code-reader` subagent. Its one return (the linear-forward budget) was fixed and re-run.

## 1. Tasks moved to review

| Task | Revision asked | Result |
|---|---|---|
| ft-81e49280d7fc01bedc9bc3e52e510f51 wgpu scratch lifetime | 5 → review, landed at revision 6 | Targeted tests passed. `verification` is null. Not moved to done. |

None moved to done. Done needs a settled passed `work.verify.task`. `.devcouncil/config.yaml` is not in the tree, so `dev verify` was not run.

### What changed

`ojas-wgpu/src/context.rs` and `ojas-wgpu/src/lib.rs`, plus tests in `ojas-wgpu/tests/accumulate_grad.rs` and `ojas-wgpu/tests/dispatch_cost.rs`.

`Recorder` tracks `scratch_bytes`. `Job::scratch`, on `CapacityExceeded` while the recorder's scratch vec is non-empty, submits once and reserves once. `SCRATCH_FLUSH_AT` defaults to `u64::MAX`, so the common path does not submit early. `set_scratch_flush_at` lowers it. `submit_locked` releases scratch even when no encoder is open, and does not wait on that path. `new_buffer`'s device-OOM retry flushes that scratch, trims the pool, then polls for up to 5 seconds, then tries once more.

`ojas-wgpu/src/backend.rs`, `ojas-metal/tests/scratch_stacking.rs`, and `ojas-core/src/backend.rs` were not edited. The contract cross-reference is on `Job::scratch`. The contract text at `ojas-core/src/backend.rs` is unchanged because that file was dirty in the primary merge when the wave started.

Local absolute `gusset` and `tessl` paths in this worktree's `Cargo.toml` files, and the one-line `Cargo.lock` path update, are a build aid so the nested worktree can resolve siblings. They overlap `gp/78-bench` on `Cargo.lock` and `ojas-metal/Cargo.toml`. They are not part of the fix.

### Review

Approved against the brief, with two recorded differences:

- The budget-refusal flush does not GPU-wait. Queue order is the safety argument. The device-OOM path still waits up to 5 seconds. Evidence: eight tight Muon steps match a roomy run that syncs after every step, bit for bit.
- `dispatch_cost.rs` lines 32–40 are the steady-dispatch pool test, not the scratch test. The line numbers in the card are stale. The test the card's own log names, `pending_scratch_that_outgrows_the_budget_triggers_a_submit`, was updated and passed. The steady-dispatch test was not run.

DevMap impact on `Job.scratch` (generation 7127, primary checkout, depth 2) is a lower bound: `walk_incomplete`, and the cached freshness describes generation 7125. Direct callers found: `operand`, `sum`, `global_norm`, `col_sum`, `rms_bwd_record`, `view_at`, `Job.commit`, `Job.upload_u32`. Depth 2 includes `muon_ns5_step` and `linear_forward`. `adamw_step` names `scratch` and is an unresolved namesake, not a bound edge. The index does not contain this worktree's uncommitted edit. 32 unresolved `scratch` namesakes include CUDA GDN, which is a different function.

### Tests run

Private `CARGO_TARGET_DIR` under the worktree, `CARGO_BUILD_JOBS=1`, `--test-threads=1`. Verified from `target/w13-test.log`:

| Test | Result |
|---|---|
| `an_out_of_memory_allocation_is_retried_once_after_releasing_the_pool` | ok, 1.05 s. The test's `alloc_retries` asserts (1, then 2, then 3, then 4) passed with it. |
| `consecutive_muon_steps_stay_inside_one_steps_scratch` (wgpu, 2048×6144) | ok. Rerun 30.22 s. `three muon steps: 2 submits`. |
| `a_step_whose_own_scratch_cannot_fit_leaves_the_parameter` | ok, 0.09 s, after the readback fix below. |
| `eight_tight_steps_match_across_two_backends` | ok, 0.66 s. |
| `eight_tight_steps_match_a_synced_roomy_run` | ok, 0.38 s. |
| `scratch_flush_at_submits_before_the_dispatch_cap` | ok, 0.34 s. |
| `pending_scratch_that_outgrows_the_budget_triggers_a_submit` | ok, 0.50 s. `8 bf16 linear_forward: 7 submits, 24 dispatches`. |
| Metal `scratch_stacking` `consecutive_muon_steps_stay_inside_one_steps_scratch` | ok, 7.22 s. |

The first hopeless-step run failed in `download`: `CapacityExceeded { requested: 4096, cap: 12352, live: 12288 }`. The step itself had already returned `CapacityExceeded` and had not submitted. The three tensors filled the cap, so the readback could not reserve a host copy. The test now drops the gradient and the momentum first. The rerun passed, and the parameter matched the upload.

The adversarial pass predicted the original linear-forward cap would die on call 4 of 8 inside `out()`, because an extra `widen / 4` let two scratches stay resident and the next output is not scratch. The cap is now the operands, the eight kept outputs, and exactly one widen. The rerun got 7 submits and 24 dispatches, under `FLUSH_AT` (64).

Not run: the full suite, `dev verify`, and a benchmark binary. The full suite was skipped because load stayed above the heavy-job bar (the 40-minute wait ended at load1 42 / load5 54, with other compilers still running).

## 2. Deferred or blocked

| Task | Why |
|---|---|
| ft-71629756 CUDA Qwen provider | in_progress, revision 10. Not touched. |
| ft-a1a4ba4d quantised LoRA | in_progress, revision 9. Not touched. |
| ft-2bbced18 LoRA step cost | in_progress, revision 15. Not touched. |
| ft-b5e64557 ojas-train | in_progress, revision 21. Not touched. |
| ft-53d639de bidirectional attention | Moved to in_progress (revision 2) during this wave. Not touched. |
| ft-96baeabc LayerNorm / GELU | Moved to in_progress (revision 2) during this wave. Not touched. |
| ft-92c5e9c6 conv2d and spatial ops | Moved to in_progress (revision 2) during this wave. Not touched. |
| G-LIBM ft-562b0add and the rest of Fable's wave 13 | Planned files intersected the primary merge or the softmax worktree when this wave was chosen (`ojas-cpu/src/backend.rs`, `tape.rs`, Metal kernels). Left for a later wave. |
| ft-f40eeed8 softmax collapse | A live worktree holds 16 dirty files, including `ojas-cpu/src/exp.rs` and `ojas-wgpu/tests/parity.rs`. |
| Folded-grid tails | Kernel guards and tests already exist in the tree. The card's planned files include `ojas-wgpu/src/backend.rs` and `ojas-wgpu/tests/`, which overlap this card. Not started, and not marked done. |
| ft-2c183b71 GPU step throughput, including the `FLUSH_AT` sweep | Sequenced after this card. Not started. |
| ft-30dd709e LoRA-A draw order | `ojas-train` is held by in_progress ft-b5e64557. |

Claude pid 44032 was busy in the primary checkout for the whole wave. `context.rs` itself was clean in every other worktree (`collision_risk` ok, 0 overlapping).

## 3. Benchmark deltas

No benchmark binary was run.

Written justification for the 5% bar: the default threshold is `u64::MAX`, so a call that fits does not take the recorder lock and does not submit. The new work runs on the refusal path, which previously returned `CapacityExceeded`. Submit counts from the tests: 3 Muon steps → 2 submits; 8 bf16 `linear_forward` → 7 submits and 24 dispatches. The `FLUSH_AT` sweep stays on ft-2c183b71 criterion 6.

## 4. New tasks filed

None. The two test bugs (linear-forward cap, hopeless-step readback) were fixed on this card.

## 5. Fable's decisions (this wave)

1. One card. The free correctness bug whose files were not held was ft-81e49280. Wave-13 libm, the softmax lane, and the in_progress trainer cards stay where they are.
2. No GPU wait on the budget-refusal flush. A waited poll would serialize every tight-budget step. wgpu queue order makes pooled reuse safe. The roomy-versus-tight Muon test is the check, and it passed.
3. Do not edit `ojas-core/src/backend.rs` while the primary checkout holds it. The cross-reference lives on `Job::scratch`.
4. Default scratch threshold `u64::MAX`. Early flush stays opt-in. The `FLUSH_AT` sweep is not this card.
5. Land this card before any later edit of `context.rs` (G-GPUFAULT, ft-cab0fdf1).
6. After the load gate expired at load1 42 / load5 54, run the targeted tests with one cargo job and a private target. Do not share the primary `target/`, and do not call `mac_heavy.sh`.
7. Do not merge to main, do not push, and do not commit the sibling-path `Cargo.toml` edits.

## 6. Owner decisions still pending

Carried from wave 12. Not re-decided here.

- E9 (`--locked` / SHA pins)
- E11b / OD10 (push)
- E12 (ft-69ef4ce1 back to ready)
- OD2 (C11 digest), OD3, OD6, OD7, OD12
- `dev verify`: no committed `.devcouncil/config.yaml`

## 7. Pending before done

- Manvi `work.verify.task` for this head. There is no such record, and the work is uncommitted on `gp/w13-scratch`, 15 commits behind `d2922fb`.
- A benchmark on the touched path, if the written justification above is not enough.
- The full suite, when load1 is under 12 and load5 is under 18.
