# Build report: ojas-4b, GPU step throughput (ft-2c183b71 rev 8)

**Run:** 2026-10-24 to 2026-10-25, session ojas-4b, in the worktree `backups/x/wt-4b` on branch
`gp/4b-gpu-step`. This file is separate so that no peer report (`BUILD_REPORT.md`, `-7a`, `-a6`, `-w11`) is
overwritten.

**Base:** `main` at `42d5a9f`. **Result:** three code commits on the branch (listed under "Commits"), with this
report in a fourth. **Not merged, not pushed.** The merge happens in the main checkout (see "Merge").

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
- A Metal `U32` upload behind recorded work never waits, at any size. It goes through tessl's
  `alloc_buffer_from_u32` [V].
- The wgpu upload cost (1 fresh buffer and 1 blocking scope pop per upload) is pinned by a test and accepted by a
  recorded ruling [V].

## Commits

| sha | what |
|---|---|
| `def246a` | wgpu: one-dispatch residual_add_backward, in-place scale_grad, shared multi-tensor scale |
| `6cbd7db` | wgpu: short clip/scale groups bind exactly their slots; upload cost pinned |
| `e575db5` | metal: U32 uploads behind recorded work never wait; scale binds a spare |

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
| 22 | Metal U32 uploads above the inline cap don't wait | Done; the test fails first [V] → MET, except its A/B → card 1. |
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
  the uses and has no overlap check (the independent review read `track/mod.rs:333-336`, `track/buffer.rs:761-763`).
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

Every other ojas-metal target passed in verify6. The two code commits differ from the verified trees only in doc
comments: the `upload_ids` doc, plus the faults-doc wording about the `Ids::Host` copy, corrected after
verify7. rustfmt `--check` on `device.rs` exit 0 afterwards [V].

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

## Cards

The board refuses writes: the installed `gitpulse-mcp` knows schema 9, and the store is 10. Seven
`gitpulse_add_task` payloads are drafted in `backups/gp/wrap-4b.json`, ready to file once the reader is rebuilt:
1. The bench remainder: crits 1, 4–8, 15–17, 23, timing rows for 2 and 9–11, and the re-bench of residual_add and
   scale_grad.
2. The residual_add_backward clones-vs-copies decision (crit 3).
3. The wgpu pooled-upload A/B with an epoch-tagged pool (crit 20 follow-up).
4. Multi-tensor optimizers (crit 13).
5. The Metal LCE tile recompute (crit 14).
6. The clip_grad_norm mid-step sync, which needs owner sign-off (crit 19).
7. Metal's inline F32/BF16 upload host copy, ≤64 KiB and uncharged (crit 12 remainder).

## Merge

From the main checkout, after its own work is committed:

```
git merge --no-ff gp/4b-gpu-step
go run scripts/sitegen/main.go
go run scripts/sitegen/main.go -check
```

- The branch carries no `Cargo.toml` or `Cargo.lock` change.
- sitegen was not run in the worktree: its member drop would remove the capi/gusset crate pages. The regeneration
  belongs in main after the merge.

## Left behind

- `target-4b` (this run's target dir) in the main checkout. Delete it if disk is needed.
- The gitignored launcher and logs under `backups/4b-tools/`.
