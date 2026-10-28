# Wave 20

Task: `ft-4e673113142ff0adf1b6b726fe8198ac`.
Moved to **review** at revision 3 (sequence 27482, `request_id` `w20-review-4e673113`, `expected_revision` 2). `previous_status` was `in_progress`. `held_for_review` is false. `verification` is null. Done was not requested. Manvi gates done.

Worktree: `/Users/bharath/Code/research/ojas-w20`
Branch: `gp/w20-gated-rstd`
HEAD: `170b22dc577716ddab96cb12c3011ba0242a2968` (not rebased).
Uncommitted. This session does not commit unless asked.

Primary `main` moved to `6abdb317b9211bd7ee93687de771edb5db34ff04` while this wave ran. This worktree was left on `170b22d`.

## What changed

`gated_rms_backward` (`ojas-cpu/src/hybrid.rs`) used to collect one `Vec` of `rstd` per piece and `parts.concat()` them. Both lived at once. `room_for` charged one `f32` per row only.

The row pass now allocates that one buffer (`vec![0.0f32; rows]`) and passes it as a third output of width 1. `chunks_into_n` writes disjoint piece slices. The second copy is gone. The charge is unchanged. The reason is in the comment above the allocation.

`dw` is still reduced by column, each column piece walking rows in ascending order.

`ojas-cpu/tests/redteam_ops_heap.rs` no longer ignores `gated_rms_norm_backward_many_rows_heap_peak_stays_within_its_charged_peak`.

`ojas-cpu/tests/hybrid.rs` adds `gated_rms_backward_short_last_piece_matches_the_serial_loop_and_releases`. That file was not on the card. Collision on it was empty before the edit.

`norm.rs` still concatenates its own `rstd` parts and charges `rows * 3`. Left alone. No new task filed for it.

## Acceptance criteria

1. Heap peak equals the Budget charge at 32768 rows and the other `redteam_ops_heap` sizes, and the BREAK test is un-ignored and passes. **Met**, by that test's own rule: peak may exceed the charge by at most `SLACK` (`64 << 10`, 65536 bytes, `redteam_ops_heap.rs:104`). The 32768-row peak is not byte-identical to the charge. Measured overage is 384 to 5904 bytes, inside that slack. The other sizes passed `every_op_heap_peak_stays_within_its_charged_peak` and `every_op_budget_boundary_is_sharp_and_balanced`. Their per-op deltas were not printed.

2. Remove the duplicate `rstd` buffer, or charge it, and say why in a comment. **Met.** The duplicate was removed. Comment at the allocation in `hybrid.rs`. The charge stayed 8519808 bytes.

## Measurements

Debug profile (`unoptimized + debuginfo`). Counting allocator. No timed A/B. No speedup is claimed.

The card records the pre-fix peak as 8651168 bytes against a charge of 8519808 (gap 131360). That run was not repeated on unmodified code this wave.

Post-fix, `gated_rms_norm_backward_many_rows_heap_peak_stays_within_its_charged_peak` with `--nocapture` (one test, 42.62s, exit 0, no recompile):

| Backend | Heap peak | Charged peak | Over |
| --- | ---: | ---: | ---: |
| threads 1 Exact | 8520192 | 8519808 | 384 |
| threads 1 Fast | 8520192 | 8519808 | 384 |
| threads 7 Exact | 8525712 | 8519808 | 5904 |
| threads 7 Fast | 8525488 | 8519808 | 5680 |

The harness's backends are threads 1 and 7 only. What the leftover bytes are was not traced. They are far under one extra `f32` per row (131072 bytes).

## Tests

`nice -n 19 cargo test -j 1 -p ojas-cpu --test hybrid --test redteam_ops_heap -- --test-threads=1`

Online, unlocked. Worktree `target/`. Not shared with primary.

- `hybrid`: 13 passed, 0 failed, 0 ignored, 2.76s. Includes the new test and `split_backward_passes_match_the_serial_loops_bit_for_bit`.
- `redteam_ops_heap`: 3 passed, 0 failed, 0 ignored, 485.19s.
- Process exit 0.

The new test checks `1×7` and `10007×7` at threads 1, 3, 7, and 18, Exact and Fast, bit-for-bit against `gated_bwd_serial`. A finite `1e20` row (first row and last row) returns `OjasError::NonFinite` after scratch exists and leaves `live_bytes` at 0. A NaN gradient is refused by the operand scan and also leaves `live_bytes` at 0. Inputs sit on a separate `Budget::new(u64::MAX)`.

`rustfmt --edition 2021 --check` on the three edited paths: exit 0.

## Not run

- `dev verify`. `.devcouncil/config.yaml` does not exist. Adding it is an owner decision. Gates are not claimed.
- `ojas-autograd` `tape_gradients_are_the_backend_backward_chain` (`hybrid_tape.rs`). DevMap named it as a caller. The index walk was incomplete. The test was not run. The CPU tests matched the serial oracle.
- Full workspace suite.
- A release-profile or timed benchmark. The card does not ask for one.

## Adversarial pass

A code-reader subagent (`01a17d4d-178e-7c40-92ac-fef9aaef2c4f`) stopped with HTTP 402 (usage balance exhausted) and returned no verdict. That is not a pass and not a fail. The attack that did run is the new hybrid test plus the un-ignored heap test.

## Cargo.lock

Unlocked cargo rewrote the worktree lock (`gusset` 0.0.2 to 0.0.3). That file was restored. `Cargo.lock` is clean on this branch and `gusset` is `0.0.2`. Primary's lock was not touched.

## Deferred this wave

One card only. These were not started. Held files, review cards, and owner decisions stay where they are.

- `ft-9aaa6349` (S4-02 serial cutoff). In review. Moving it back to ready is an owner decision. Largest measured CPU anti-scaling from wave 19. Not edited.
- `ft-845929b1` (S4-06 / S4-08 / S4-09). S4-09 is OD6. `pointwise.rs` is dirty in the softmax worktree. Criteria want a committed A/B.
- `ft-fb1b9798` (panic `catch_unwind`). `pointwise.rs` is a planned file and is held.
- `ft-562b0add` criterion 7 (Fast fused CE). `fused_ce.rs`, `pointwise.rs`, and `exp_exact.rs` are dirty in the softmax worktree. Criterion 6 is OD8.
- `ft-ee2a3753` cannot close: criterion 3 needs metal `device.rs` (held by `ojas-78a`), criterion 6 waits on trainer work owned by in-progress `ft-b5e64557`.
- GPU, Metal, Qwen, LoRA, and softmax-collapse cards stay deferred. Their files are held or the work is not one Mac-finishable concern.
- In-progress cards were not touched: `ft-71629756`, `ft-a1a4ba4d`, `ft-2bbced18`, `ft-b5e64557`, `ft-96baeabc`, `ft-92c5e9c6`, `ft-8d3aa337`.

## Fable

- Remove the duplicate buffer. Do not charge it twice.
- Do not reopen `ft-9aaa6349`.
- Do not edit `norm.rs` on this card.
- One card. Do not leave it `in_progress`.

## Owner decisions still pending

Not re-decided here. E9 (`--locked` / SHA pins). E11b / OD10 push. E12 (`ft-69ef4ce1` back to ready). OD2 (C11 digest). OD3. OD6 (S4-09). OD7 (S5-08 generate probe). OD8 (S4-01 runtime FMA). OD12. Committing `.devcouncil/config.yaml`. Committing the wave-19 results directory. Lappi pin past `4123dfb` (wave 18 is still uncommitted). Audit item 11 (whether to move `ft-9aaa6349` from review back to ready).

## Verification

`gitpulse_complete_task` returned `ok: true`, `outcome: updated`, `status: review`, `revision: 3`, `sequence: 27482`, `summary_recorded: true`, `held_for_review: false`, `verification: null`. Null is not a pass.
