# Build report: ojas-4b, GPU step throughput (ft-2c183b71 rev 8)

**Run:** 2026-10-24 to 2026-10-25, session ojas-4b, in the worktree `backups/x/wt-4b` on branch
`gp/4b-gpu-step`. This file is separate so that no peer report (`BUILD_REPORT.md`, `-7a`, `-a6`, `-w11`) is
overwritten.

**Base:** `main` at `42d5a9f`. **Result:** four code commits on the branch (listed under "Commits"). This
report came in `7aba6b3` and was brought up to date in the branch's last commit. **Not merged, not pushed.** The merge happens in the main checkout (see "Merge").

**Builds:** the owner authorised a detached in-repo launcher (`backups/4b-tools/`, gitignored), with every
cargo step inside `Lappi-decision/tools/mac_heavy.sh` and its own `CARGO_TARGET_DIR` (`target-4b`). The
worktree's `Cargo.toml` and `Cargo.lock` drop the gusset members locally so the workspace loads. That change is
never committed (`backups/4b-tools/git.sh` refuses to stage either file).

**No timing.** The machine sat at load ~50–300 with both cargo slots contended for the whole run, and the owner
ruled out timing benches. No speedup is claimed. Every row that needs a number is on a card (see "Cards").

**Labels:**
- **[V]** verified this run by a command or a read.
- **[I]** inferred.
- **[U]** unverified.

## Headline

- wgpu `residual_add_backward` is one `add_bwd` dispatch, down from 3. It checks all three inputs, as Metal's
  `ojas_add_bwd` does [V].
- wgpu `scale_grad` scales a sole owner in place. Its only transient is the 8-byte slot table, down from about
  2× the gradient [V].
- A short last group of a multi-tensor scale no longer binds one gradient read-write in several slots. On wgpu it
  dispatches the Clip module with exactly its count of slots; on Metal the unused slots bind a worker-owned
  spare [V in source; see crit 21 for why no test can fail first on this device].
- A Metal `U32` upload behind recorded work takes no upload wait, at any size. It is written on the host into a
  buffer fresh from tessl's pool (`Worker::fresh_from`, `alloc_buffer_from_u32`) [V].
  - **Waits that can still happen:** a memory-cap wait (its bytes count toward the trigger like any allocation's),
    and a recycle when the device refuses the allocation.
  - **"Never waits", used in `e575db5`'s message, is wrong** for that reason. The second review caught it.
  - **Small uploads take this path too.** The criterion asked only for uploads above the 64 KiB inline cap. This
    path is cheaper than inline: it uses no constant-arena room, dispatches no `ojas_upload_words`, and makes no
    encode copy on the host [V in source]. Its speed is not measured (card 1).
- The wgpu upload cost (1 fresh buffer and 1 blocking scope pop per upload) is pinned by a test and accepted by a
  recorded ruling [V].

## Commits

| sha | what |
|---|---|
| `def246a` | wgpu: one-dispatch residual_add_backward, in-place scale_grad, shared multi-tensor scale |
| `6cbd7db` | wgpu: short clip/scale groups bind exactly their slots; upload cost pinned |
| `e575db5` | metal: U32 uploads behind recorded work never wait; scale binds a spare |
| `740df7a` | metal: id uploads share fresh_from and its fault injection; pool-reuse and retry tests |

## Criteria (brief order, 1–23)

| # | criterion | status |
|---|---|---|
| 1 | wgpu AdamW check/apply | Code in base (0e3254f) [V, brief annotation]. The adamw_full re-bench is owed → card 1. |
| 2 | Fused clip_grad_norm, dispatch counts recorded | Code in base. Counts recorded below [V for after, I for before] → MET. The timing row → card 1. |
| 3 | residual_add_backward: clones vs copies | The dispatch half is done here (3 → 1) [V]. The decision is owed → card 2. |
| 4 | wgpu attention re-bench | Not started (timing) → card 1. |
| 5 | Open Metal/wgpu rows profiled | Not started (timing) → card 1. |
| 6 | FLUSH_AT A/B | Not started (timing) → card 1. |
| 7 | Metal Link::call round-trip | Not started (timing) → card 1. |
| 8 | Every speedup has a committed A/B | No speedup is claimed. The A/Bs owed → card 1. |
| 9 | wgpu dispatch fixed cost counted | Code in base. Before/after counts below [V after, I before] → MET for counts. The timing row → card 1. |
| 10 | wgpu read() staging reuse | Code in base. Counts below [V after, I before]. The timing row → card 1. |
| 11 | wgpu offset views bound in place | Code in base. Copy count below [V after, I before] → MET for counts. |
| 12 | Uploads make no uncharged host copy | wgpu met in base. Metal large uploads met in base. Metal `U32` uploads now make no staging copy on the device thread [V in source]. The backend's `Ids::Host` copy of the ids predates this. Metal's inline F32/BF16 path still copies ≤64 KiB uncharged → card 7. |
| 13 | Multi-tensor optimizers | Not started → card 4. |
| 14 | Metal LCE tile recompute | Not started → card 5. |
| 15 | Metal embed rank O(N²) | Code in base. The embed_bwd row is owed → card 1. |
| 16 | wgpu Muon f32 row | Not started (timing) → card 1. |
| 17 | Gate finite checks folded if the profile says so | Conditional on a profile → card 1. |
| 18 | GPU scale_grad in place | Metal in base (aaed8b3). wgpu done in `def246a` [V] → MET. |
| 19 | clip_grad_norm mid-step sync | Needs owner sign-off → card 6. |
| 20 | Upload cost covered and pooled or accepted | Accepted by ruling B, pinned by a test [V] → MET. Pooled A/B → card 3. |
| 21 | Unused slots stop aliasing group[0] read-write | wgpu and Metal both done [V in source]; regression guards on both → MET (see below). |
| 22 | Metal U32 uploads above the inline cap don't wait | Done: no upload wait, the test fails first [V]. The retry/refusal path and the pool-reuse safety are tested too → MET, except its A/B → card 1. |
| 23 | Metal clip_grad_norm within 1.2× torch | Timing → card 1. |

### Counts recorded (crits 2, 9, 10, 11)

**Crit 2, clip_grad_norm, 170 gradients:**
- **Metal now [V]:** the norm is one tessl binder scope for any count, and the scale is `1 + ceil(170/24)` = 9
  (`ojas-metal/tests/clip_multi.rs`, `clip_norm_is_one_scope_and_scale_goes_by_slot_groups`).
- **Metal before 0e3254f [I]:** 5 dispatches per gradient to check and reduce, plus 2 to scale, about 1190 (from
  that test file's header and the 0e3254f^ source).
- **wgpu now [V]:** the norm is `ceil(170/k) + 1`, and with the scale `2·ceil(170/k) + 1`. Here `k` comes from
  the device's storage-binding limit (`clip_dispatches_scale_with_slot_groups_not_tensors`).
- **wgpu before [I]:** an abs-max and a sum-of-squares dispatch per gradient, plus two finishes, plus one scale per
  gradient. That is 342 for the norm and 512 with the scale.

**Crits 9 and 10, a steady loop of 50 `silu_forward` + `sync`
(`dispatch_cost.rs`, `a_steady_dispatch_creates_nothing_and_pops_no_error_scope`):**
- **Before 0e3254f [I, from source; the counters did not exist then]:**
  - each dispatch made a 64 B mapped uniform and a bind group, each under a blocking error-scope pop;
  - each read made a fresh `MAP_READ` staging buffer under another pop.
  - For the loop that is at least 150 blocking pops, 50 bind groups and 50 staging creates.
- **Now [V]:** 0 pops, 0 bind groups and 0 staging creates. There are 100 bind-group hits and 50 pool hits.

**Crit 11, a fused-QKV buffer's three aligned row blocks
(`offset_views_bind_in_place_when_the_device_allows_it`):**
- **Before [I]:** the two blocks at a non-zero offset were copied into scratch on every op, 2 copies.
- **Now [V]:** 0 copies.
- A view one word in is not a storage-offset multiple. It is compacted (1 copy), before and after.

### Crit 20: the ruling

Fable ruled (B): uploads keep the mapped-at-creation path. Its decision sentence, quoted:

> "Uploads keep the mapped-at-creation path: `tests/dispatch_cost.rs` pins exactly 1 fresh buffer and 1 OOM-scope
> pop per upload (2K per step for K micro-batches), counts accepted; the timing A/B is owed to the bench card on a
> quiet machine, and a pooled upload is admissible only if `give_back` tags the buffer with the submit epoch so a
> hit never needs a forced submit."

**Why a pooled upload is not simply better [V]:**
- `WgpuBuffer::drop` gives a buffer back to the pool at once (`ojas-wgpu/src/context.rs`, `impl Drop for
  WgpuBuffer` → `give_back`).
- The pool's invariant (context.rs header) is that a pooled buffer is written only by commands recorded after the
  ones that read its old contents.
- A `queue.write_buffer` lands at the next submit, before such recorded work, so it would need a forced submit per
  upload.

**Reverses if** a quiet-machine A/B shows create+pop costing more than a submit beyond the noise floor (→ card 3).

The test is `an_upload_is_one_fresh_buffer_and_one_scope_pop`. It is a characterization test, and passes on the
base by design.

### Crit 21: why no test fails first

- **wgpu:** wgpu-core 30.0.1 accepts one buffer bound `STORAGE_READ_WRITE` in several slots of a bind group. It ORs
  the uses and has no overlap check. This is the independent reviewer's reading of `track/mod.rs:333-336` and
  `track/buffer.rs:761-763` [I: I did not re-read it]. The base's tests binding `group[0]` repeatedly passed here.
- **Metal:** binds buffers by pointer with no aliasing check [I]. The base already repeated `group[0]` there,
  and its tests passed.

So the aliasing could not be shown failing here. Only a stricter WebGPU implementation would refuse it [I]. The
fix removes the aliasing structurally:
- **wgpu `scale_multi`:** a short last group gets `WgslModule::Clip(group.len())`, whose slot list is
  `CLIP_SCALE_SLOTS[..1 + group.len()]`, so no slot is unused.
- **wgpu `global_norm`:** still repeats its first gradient in unused slots. Those slots are read-only (legal
  everywhere), and its charge stays exactly `norm_scratch`, which `optimizer_scratch_bytes` sums.
- **Metal `scale`:** binds `self.spare` at offset `4·slot` for each unused slot, so no two read-write bindings
  share memory.

Regression guards that pass before and after:
- wgpu `folded_grid_ops` `scale_grad_leaves_spare_workgroups_inert`, now including 2·4096+1, three chunks;
- Metal `clip_multi` `every_last_group_size_scales_like_the_cpu`, with counts 1, 23, 24, 25 and 47.

## Verification

All runs are under `backups/4b-tools/logs/<job>/` (run.out, STEPS.txt, one log per stage). tessl was
`/Users/bharath/Code/research/tessl` at `8bc0468`, clean.

**verify3** checked `def246a` plus the global_norm revert:
- wgpu: all ok.
- clippy: ok.
- **Pre-fix red on 42d5a9f sources [V]:**
  - `residual_add_backward_is_one_dispatch…`: 3 vs 1.
  - `scale_grad_of_a_sole_owner…`: peak +512004 B vs 8.
  - `scale_grad_leaves_spare_workgroups_inert`: `OutOfRange gemm_grid` (the trait default's `mul_forward` under the
    capped grid [I]).

**verify6** checked the full uncommitted follow-up. rustfmt `--check` exit 0.

| stage | result |
|---|---|
| `cargo test -p ojas-wgpu` | 28 targets, 205 passed, 0 failed, 7 ignored [V] |
| `cargo test -p ojas-metal` | 287 passed, 2 failed (both resolved in verify7, below) |
| `cargo test -p ojas-model --test device --test metal_waits` | 5 passed, 0 failed, 1 ignored [V] |
| `cargo clippy -p ojas-core -p ojas-kernels -p ojas-wgpu -p ojas-metal --all-targets -D warnings` | exit 0 [V] |
| new wgpu tests on 42d5a9f's wgpu/core/kernels sources | red as expected: residual_add 3 vs 1; scale_grad +512004 vs 8; folded scale_grad `OutOfRange`. Green as expected: the upload characterization, the shared-gradient guard, the offset sole-owner guard [V] |
| Metal `wait_triggers` on 42d5a9f's `device.rs` | red as expected: `a_large_id_upload_while_work_is_recorded_does_not_wait` got `WaitCounts { upload: 1, .. }` vs none [V] |

The sources were restored after the pre-fix stages. The shasums before and after match (run.out).

**verify6's two Metal reds:**
- **`clip_multi` `every_last_group_size_scales_like_the_cpu`, red on both trees.** The cause was a bug in my new
  test: it gave the CPU oracle `max_norm = f32::MAX`, whose coefficient overflows, and `clip_scale` refuses that
  (`NonFinite`). Now `1.0e9`, as the sibling tests use.
- **`adversarial_oom` `partial_allocations_are_released_after_the_final_refusal`:** leaked 16990208 B under a
  parallel run. The test reads the process-wide allocated figure, and its doc says to run it alone
  (`--test-threads=1`).

**verify7** re-ran with the oracle fix. tessl had moved to `5d6f786` (a peer's commit, clean) between the two runs.

| stage | result |
|---|---|
| rustfmt `--check` on `clip_multi.rs` | exit 0 [V] |
| `cargo test -p ojas-metal --test clip_multi` | 8 passed, 0 failed [V] |
| `cargo test -p ojas-metal --lib -- --test-threads=1` | 81 passed, 0 failed, 1 ignored, `adversarial_oom` included [V] |
| `clip_multi` on 42d5a9f's `device.rs` | 8 passed: `every_last_group…` is a guard, green before the fix as stated [V] |

Every other ojas-metal target passed in verify6. `6cbd7db` and `e575db5` differ from the verified trees only in
doc comments: the `upload_ids` doc, plus the faults-doc wording about the `Ids::Host` copy, corrected after
verify7. rustfmt `--check` on `device.rs` exit 0 afterwards [V].

**verify8** ran after the second review, on the `740df7a` tree: the U32 upload through `fresh_from`, the hook in
`injected_failure`, and the two new tests. tessl was `5d6f786`, clean. The committed files' shasums match the
tested ones (run.out) [V].

| stage | result |
|---|---|
| rustfmt `--check` on `device.rs`, `backend.rs`, `wait_triggers.rs` | exit 0 [V] |
| `cargo test -p ojas-metal -- --test-threads=1`, every target | 36 targets, 291 passed, 0 failed, 2 ignored [V] |
| `cargo test -p ojas-model --test device --test metal_waits` | 5 passed, 0 failed, 1 ignored [V] |
| `cargo clippy -p ojas-metal --all-targets -D warnings` | exit 0 [V] |
| the retry test on `e575db5`'s `device.rs` | red as expected: recycle count 0 vs 1, because the hook never reached the U32 path (`backend.rs:3483`) [V] |
| `wait_triggers` on `42d5a9f`'s `device.rs` | red as expected: `a_large_id_upload…` (`upload: 1`) and `an_id_upload_never_takes…` (upload waits 4 vs 0). The other 4 passed [V] |

**verify9** tested the combined tree as it will be merged. It made a detached trial merge in `wt-4b` of main
`216f3ed` (ft-e15ee76a merged, which also changes ojas-core, ojas-metal and ojas-model) into `740df7a`. No branch or
ref moved. `git merge-tree` gave a clean tree `f03986f`; the merge auto-merged `ojas-metal/src/backend.rs` and
`device.rs` and made combined HEAD `96d3383` with that same tree. tessl was `5d6f786`, clean. Afterwards the worktree
went back to `gp/4b-gpu-step` at `740df7a` (run.out) [V].

| stage | result |
|---|---|
| `cargo test -p ojas-metal -- --test-threads=1`, every target | 37 targets, 294 passed, 0 failed, 2 ignored (with main's new `autocast_lane`) [V] |
| `cargo test -p ojas-model --test device --test metal_waits --test trainer` | 3 targets, 24 passed, 0 failed, 1 ignored [V] |
| `cargo test -p ojas-core --lib` | 155 passed, 0 failed, 3 ignored [V] |
| `cargo test -p ojas-kernels` | 2 targets, 18 passed, 0 failed [V] |
| `cargo test -p ojas-wgpu`, every target | 28 targets, 205 passed, 0 failed, 7 ignored [V] |
| `cargo clippy -p ojas-core -p ojas-cpu -p ojas-kernels -p ojas-wgpu -p ojas-metal -p ojas-model --all-targets -D warnings` | exit 0, no warnings [V] |

This branch changes no ojas-cpu file. main's `ojas-cpu/src/optim.rs` change passed clippy here, but its tests were
not run in this tree.

**verify10** ran after main moved on to `464547a` and then `6bfa82e`. Of the two, only `4c1640a` changes code: it
rewrites `ojas-metal/tests/autocast_lane.rs` (a TensorOps canary for Apple M5, exact-bit backward checks). verify10
made a detached trial merge of main `6bfa82e` into `740df7a` the same way. It was clean, giving combined HEAD
`f104a21`, tree `6aa9bc3`, and the worktree went back to `gp/4b-gpu-step` at `740df7a` (run.out) [V].

| stage | result |
|---|---|
| `cargo test -p ojas-metal --test autocast_lane -- --test-threads=1` | 4 passed, 0 failed, 0 ignored, `an_apple_m5_reports_tensorops` included [V] |

## Independent review

An Opus reviewer read `def246a` plus the first follow-up against `42d5a9f`. It read code only. It found nothing
blocking. Its three low/info findings, each fixed in `6cbd7db`:
1. **The spare buffer in `scale_multi` was itself bound writable in every unused slot, and charged 4 uncounted
   bytes.** Fixed: `spare_slot` is deleted, and a short group dispatches `Clip(group.len())`.
2. **The in-place half of the folded scale_grad test never produced a spare workgroup.** Fixed: added 2·4096+1,
   and the file header now notes `add_bwd`/`scale` have not been through the guard-removal run. The in-place half
   still uploads `exact` (no padding), so a missing `i < n` guard in `scale_multi`'s table walk would not show
   there [I]. The shared half, which is padded, exercises spares.
3. **Only two new tests failed on base.** No test covered a sole-owner view at an offset. Fixed:
   `scale_grad_of_a_sole_owner_at_an_offset_replaces_it`. The guards are labelled as guards here.

An earlier in-run failure also led to a fix. `accumulate_grad`'s exact peak test saw 88 B vs 84 B when
`global_norm` bound an uncharged spare. Reverted: `global_norm` repeats its first read-only gradient.

### Second review: `e575db5`

A second Opus reviewer read only `e575db5`, the Metal commit, which the first review predates. It read code only,
in ojas and tessl. It found nothing blocking.

**Safety argument (its claim 1): holds [V by its read].**
- A dropped buffer goes back to tessl's pool at `release_point = event_value + 1` (tessl `runtime.rs:1397`), so only
  after the commit holding the commands recorded against it completes.
- `recycle_if_idle` (`runtime.rs:1531`) does nothing while a command buffer is open.

**Its other checks were clean:**
- `retrying` around `alloc_buffer_from_u32`;
- the `note_dispatches` move;
- the off switch;
- the spare: the kernel's `norm_slot` loops `k < used` (`ojas_backend.metal:1066`), and offset 92 of 96 bytes
  passes tessl's offset check.

**Its findings, each acted on in `740df7a`:**
1. **"Never waits" overstated it.** The upload's bytes feed the memory-cap trigger, and a refusal recycles. Fixed in
   the `Worker::upload` doc, the faults doc §9.1 and this report. `e575db5`'s message keeps the old wording.
2. **The test hook `fail_allocs` did not reach the U32 path**, so its retry and refusal were untested.
   - Fixed: the hook moved into `Worker::injected_failure`, which both `try_alloc` and `fresh_from` consult.
   - The U32 upload now calls `fresh_from`, so `upload_ids`, a near-copy of it, is deleted.
   - New test: `backend.rs` `an_id_upload_behind_recorded_work_retries_and_refuses_like_any_allocation`. It checks
     one injected refusal (a recycle, no upload wait, identical readback) and a persistent one (`CapacityExceeded`
     for 1 MiB, the budget charge returned).
   - No existing `fail_allocs` test reaches `fresh_from`; its only other callers are `embed_bwd` and `clip_norm`.
3. **Nothing tested pool reuse.** New test: `wait_triggers.rs`
   `an_id_upload_never_takes_a_buffer_that_recorded_work_still_reads`. Four rounds each free the 1 MiB of ids a
   pending `embedding_forward` reads, then upload as many new ids. Each gather must match its own ids.
   - This is a guard of tessl's release rule, not a regression test. `e575db5` makes the same tessl call, so it
     should pass there too [I: not run on `e575db5`]. On `42d5a9f`'s `device.rs` it fails on the upload-wait
     count (4 vs 0) before reaching the values (verify8).
   - It would fail if tessl returned a dropped buffer before its readers' commit completed [I].

The reviewer also noted that `dispatches()` drops by one for each small U32 upload behind recorded work, since no
`ojas_upload_words` runs. It found no test that counts dispatches across such an upload.

## Cards

The board refuses writes: the installed `gitpulse-mcp` knows schema 9, and the store is 10. Eight
`gitpulse_add_task` payloads are drafted in `backups/gp/wrap-4b.json`, ready to file once the reader is rebuilt:
1. The bench remainder: crits 1, 4–8, 15–17, 23, timing rows for 2 and 9–11, and the re-bench of residual_add and
   scale_grad.
2. The residual_add_backward clones-vs-copies decision (crit 3).
3. The wgpu pooled-upload A/B with an epoch-tagged pool (crit 20 follow-up).
4. Multi-tensor optimizers (crit 13).
5. The Metal LCE tile recompute (crit 14).
6. The clip_grad_norm mid-step sync, which needs owner sign-off (crit 19).
7. Metal's inline F32/BF16 upload host copy, ≤64 KiB and uncharged (crit 12 remainder).
8. `clip_scale(f32::MAX, norm)` refuses as `NonFinite` when the norm is below about 1. The coefficient overflows
   before the `min(1, ·)`, so "`max_norm = f32::MAX` means don't clip" fails. This is documented, but it cost this
   run a red test.

## Merge

From the main checkout, after its own work is committed:

```
git merge --no-ff gp/4b-gpu-step
go run scripts/sitegen/main.go
go run scripts/sitegen/main.go -check
```

- `git merge-tree --write-tree` against main is clean at every main this run saw: `216f3ed` gives tree `f03986f`,
  `464547a` gives `836d5db`, and `6bfa82e` gives `6aa9bc3` [V]. `216f3ed..6bfa82e` changes only
  `ojas-metal/tests/autocast_lane.rs` and two reports (`git diff --stat`) [V], so verify9 covers all of main's
  source and verify10 runs the one changed test against this branch.
- The branch carries no `Cargo.toml` or `Cargo.lock` change.
- sitegen was not run in the worktree: its member drop would remove the capi/gusset crate pages. The regeneration
  belongs in main after the merge. Commit the regenerated `site/`, or the next session's `-check` fails.
- `adversarial_oom::partial_allocations_are_released_after_the_final_refusal` passes only single-threaded. Its own
  doc says so, and it saw a 16 MiB delta from sibling tests under a parallel run here. A parallel post-merge failure
  of that test alone is that known gap, not this branch. I did not re-run it in parallel on the base, so "known" is
  by the test's doc, not by measurement [I].
- verify6 ran against tessl `8bc0468`; verify7 through verify10 ran against `5d6f786`.

## Left behind

- `target-4b` (this run's target dir) in the main checkout. Delete it if disk is needed.
- The gitignored launcher and logs under `backups/4b-tools/`.
- The trial merge commits from verify9 and verify10 (`96d3383` and `f104a21`). No ref names them,
  so git's gc will drop them.
