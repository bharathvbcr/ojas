# Resource governor audit

Recorded 2026-10-01 after the resource-governor lanes. A row is **verified** when this session ran the suite that contains the named test, or read the assertion and that suite exited 0. It is **reported** when a prior lane ran it and this session did not. It is **unverified** when no run on this machine covers the claim. Reading source is labeled as source, not as a passing test.

This file does not add behavior. It records what the current tree locks, and what it does not.

## What this session ran

**Verified.** From `/Users/bharath/Code/research/ojas`:

```bash
cargo test --workspace --release -- --test-threads=1
```

Exit 0 in 57s. 449 passed, 0 failed, 4 ignored. Doc-tests ran and contained no tests. The ignored tests are two `ojas-cpu` benches, one exact-golden case, and the wgpu bench. Features `cuda` and `hip` were not enabled. Per-crate counts are in `docs/status.md`.

**Verified.** `cargo build -p ojas-gusset-engine`, then:

```bash
cd go && PKG_CONFIG_PATH="$PWD" go test -tags gusset_pkgconfig -count=1 -timeout 15m ./...
```

Exit 0, test time 1.443s. The output did not show a Go compile after the archive rebuild, so that run may have reused a cached test binary. A second run with `-a` (required by the comment in `go/api.go`, because the cache does not track `libgusset.a`) also exited 0, test time 3.417s. `go test -tags gusset_pkgconfig -list '^Test' ./...` lists 27 tests, including `TestCloseRacingAnInFlightStepLeavesOnePool`, `TestConcurrentSessionStress`, and `TestPoisonDropsSession`.

**Reported.** A prior lane on 2026-10-01 ran tessl `shared_event_timeout_poison_rejects_further_encode` in release: 1 passed, 110 filtered. This session did not run tessl's suite.

## Crash

| ID | Status | Lock | What changed |
| :--- | :--- | :--- | :--- |
| C1 | Verified that the refusal path passes. The abort path is **not fixed**. | `ojas-core` `scratch_holds_one_charge_until_drop`, `scratch_reservation_failure_does_not_leave_a_charge`. `ojas-capi` `step_returns_capacity_when_the_cross_entropy_scratch_does_not_fit` (its comment says the test does not catch `abort`). | Fallible allocation is `Budget::try_reserve` and `Scratch`, which drop the `Vec` before releasing the charge. `ojas-gusset-engine` installs gusset's `Counting` allocator and its own comment says `vec!`, `format!`, and channels still abort. There is no allocation-error hook. |
| C2 | Verified for the ceilings. A forced OS spawn failure is unverified. | `with_threads_refuses_zero_and_above_the_ceiling_without_clamping`, `with_threads_refuses_u32_max_without_spawning` (`ojas-cpu/tests/governor.rs`). `zero_and_oversized_thread_counts_are_refused` (`pool.rs`). `device_header_selects_cpu_parallel_and_bounds_its_thread_count` (`ojas-capi`). `CPU_THREAD_CEILING` is 1024. `MAX_CPU_THREADS` is 256. | `CpuBackend::with_threads` refuses 0 and anything above 1024. Load refuses a parallel thread count above 256. Workers are `thread::Builder::spawn`, and a spawn `Err` is stored and returned. `spawn_scoped` is used by the pool's caller test `many_callers_share_one_pool`, not as the worker factory. No test injects a failed spawn. |
| C3 | Verified | `TestCloseRacingAnInFlightStepLeavesOnePool`, `TestCloseKeepsTheGateShutWhenWorkersRemain`. `TestConcurrentSessionStress` sets pool size 1, starts 64 goroutines, and fails if they are still running after 4 minutes. | `Close` returns the gusset error when workers are still running after the join budget, and `ensureHandle` then refuses. It does not open a second pool on top of those workers. |
| C4 | Partially verified. Adam moments are not in the snapshot. A real GPU hang is unverified. | `poisoned_runtime_host_weights_rebuild_on_a_new_session` (`ojas-metal`, in the 33 library tests that passed). Tessl `is_poisoned` and `poison_as_shared_event_timeout_for_test`. Reported: tessl `shared_event_timeout_poison_rejects_further_encode`. | The Metal test poisons with the shared-event timeout bit, checks that a further alloc mentions poison, and rebuilds weights from `host_weights` on a new session. The snapshot tuple is the eight weight slices. Moments are not copied. The test comment says a GPU hang is not produced. |
| C5 | Verified | `uncaptured_error_does_not_panic` (`ojas-wgpu`, in the 20 library tests that passed). | Creating a buffer with empty usage is inside `catch_unwind` and must not panic. That is one uncaptured error, not a device-lost recovery of a running model. |
| C6 | Host poll helpers verified. Device features unverified. | `stream_wait_returns_when_the_deadline_has_passed` (`ojas-cuda`). `copy_wait_returns_when_the_deadline_has_passed` (`ojas-hip`). Both are in the 8 default tests each crate passed. | A deadline that has already passed returns. Those tests call `poll_until` with a stub predicate. `--features cuda` and `--features hip` were not built, and this Mac has no NVIDIA GPU and no `/opt/rocm`. |

## Memory

| ID | Status | Lock | What changed |
| :--- | :--- | :--- | :--- |
| M1 | Verified | `STEP_PROCESS_BUDGET_BYTES` is `1 << 30` (`ojas-capi/src/step.rs`). `token_output_is_charged_once`. | Steps share one process child of 1 GiB, capped by the caller budget. The token-mode test binary-searches the smallest budget that succeeds and requires it to be below the old double charge (`formula + 88`). |
| M2 | Verified | `causal_sdpa_refuses_scratch_when_the_output_tensor_would_fit`. | The budget holds Q, K, V, and the output. The score scratch misses, the call is `CapacityExceeded`, and live bytes stay at the inputs. |
| M3 | Verified for the tests below. One discarded check remains. | `scratch_holds_one_charge_until_drop`. `room_for_keeps_the_charge_until_the_guard_drops`. `linear_forward_refuses_budget_that_fits_output_but_not_packed_weight`. `cross_entropy_refuses_row_scratch_when_the_scalar_loss_would_fit`. `clip_and_adam_write_rules` refuses `adamw_step` on a tight budget and checks the parameter and moments are unchanged. `headroom` is called at the start of `adamw_step` and `clip_grad_norm`. | Scratch and optimizer working memory are charged before the compute, and a miss leaves the stored values alone. `zero_extent_inf_offset_and_budget_do_not_corrupt_inputs` still does `let _ = before_prefix_budget`, so that particular live-byte comparison is still discarded. |
| M4 | Verified for the refusal tests. A full successful 1 GiB checkpoint and a full successful 32 MiB BPE encode were not run. | `node_cap_allows_the_exact_count_and_refuses_one_more`. `dense_array_is_refused_before_gigabyte_amplification`. `encoded_length_matches_the_bytes_and_the_writer_shares_the_read_cap`. `checkpoint_cap_allows_the_exact_byte_and_refuses_one_past`. `encode_refuses_input_past_the_file_cap` (`HF_TEXT_CAP` is 32 MiB). `tape_clear_drops_recorded_values_and_gradients`. | JSON refuses one node past the injected cap. Checkpoint encode and write share `MAX_CHECKPOINT_BYTES` (1 GiB). The io suite passed in 1.23s, and `encoded_length_matches_the_bytes_and_the_writer_shares_the_read_cap` allocates a payload one past that cap and expects both encode and write to refuse without creating the file. The exact-byte test skips building a 1 GiB success. BPE encode of `HF_TEXT_CAP + 1` bytes is refused. `Tape::clear` drops the recorded value and gradient. |
| M5 | Verified | `TestDefaultBufferBudgetIsSharedAndBelowOneMaxBuffer` expects `64 << 20`. `TestMemoryGovernorIsOptIn` rejects `WithMemoryGovernor(0)` and a negative total. | `bufferBudget` starts at 64 MiB and is passed as `WithBufferBudget` on open. `WithMemoryGovernor` is not called until the user calls it. |
| M6 | Verified | `param_bytes_rounds_each_buffer_up_to_a_power_of_two_bucket`. `session_bounds_the_pool_cache_below_the_tessl_default` (`SESSION_POOL_CACHE_BYTES` is 64 MiB). `free_list_refuses_a_new_size_once_the_key_cap_is_full` (`POOL_MAX_KEYS` is 64). | Metal charges tessl's power-of-two bucket, not the logical byte length, and the session pool-cache cap is 64 MiB. The wgpu free-list test refuses a new size once 64 keys are stored. Its byte cap (`POOL_CAP_BYTES`) is 512 MiB, separate from the 64 MiB Metal pool cache. |

## Silent behavior

| ID | Status | Lock | What changed |
| :--- | :--- | :--- | :--- |
| S1 | Partially verified. Command-buffer status is absent on Metal 4. Argmax nonfinite refusal was not run in this session. | Tessl `GpuRuntime` counts `MTL4CommitFeedback`. The comment states the command-buffer protocol has no status. Reported: `shared_event_timeout_poison_rejects_further_encode`. Source: tessl `argmax_f32_pass_refuses_a_nonfinite_row` in `tests/nn_wiring.rs`. | Commit feedback is the poison signal. This session did not run the argmax test. A real GPU fault was not induced. |
| S2 | Verified as the cap, not as a 1 GiB successful save. | Same checkpoint tests as M4. Read and write both call `exceeds_checkpoint_cap` with `MAX_CHECKPOINT_BYTES`. | A write past the shared cap is an error and does not leave the destination file. |
| S3 | Verified | `TestInBandErrorPrefixes`. `nonfinite_and_capacity_prefixes_survive_the_gusset_boundary`. `host_error` formats `ojas:E_CAPACITY:`, `ojas:E_DEVICE_LOST:`, `ojas:E_BUSY:`, and `ojas:E_NONFINITE:`. | Go `errors.Is` matches those prefixes, including when a gusset wrapper is in front. `go/api.go` still says “Rust is not emitting these prefixes yet”; that sentence is stale. This docs pass did not edit the Go package. |
| S4 | Source plus an identity test. The `O_NOFOLLOW` flag itself has no direct test. | `open_model` opens the file once. On macOS it sets custom flag `0x0000_0100`. `a_symlink_swapped_in_is_not_followed` uses `File::open` and `confirm_open_identity`, not `open_model`. | The loader does not reopen by path after the first open. A swapped final-component symlink fails the identity check. The test does not assert that `open_model` returns an error on a symlink. |
| S5 | Documented in source. No second-call test was found. | `ojas_engine_init` and `install_engine`: if `INSTALLED` is already true, the function returns success and does not call `clear_engine_handlers`. | A second init is a no-op. The first call still clears handlers that were already registered in the process. |
| S6 | Verified for the tests named here. Greedy still ignores the loaded weights. | `tape_clear_drops_recorded_values_and_gradients`. `a_successful_root_clears_last_error_and_a_full_take_does_not_leave_it`. `greedy_honors_cancel_after_the_budget_charge`. `step_cancel_runs_between_cross_entropy_forward_backward_and_clip`. `TestPoisonDropsSession` frees the id and expects `unknown model` before `Close`. | `LAST_ERROR` is cleared on success and after a full take. Generate cancel is checked during the greedy path. `OP_PANIC` stays registered because that Go test calls it through the staticlib, which is not built with `cfg(test)`. `generate.rs` and `GenerateGreedy` say the greedy path is a fixed two-token demonstration and does not use the loaded file. |
| S7 | Not fixed. Verified by reading the three parsers. | `ojas-io/src/json.rs` accepts surrogate pairs. `ojas-data/src/bpe.rs` rejects a surrogate in a vocab key and says it does not use the io parser. `ojas-oracle/src/lib.rs` refuses every escape and parses f64. | The grammars differ on purpose. They were not unified. |
| S8 | Unverified on a machine without Xcode. The paths are still absolute or sibling. | `ojas-metal` depends on `tessl` at path `../../tessl`. `ojas-capi` and `ojas-gusset-engine` depend on gusset at `../../../devtools/gusset/crates/gusset`. `ojas-metal/build.rs` runs `xcrun` and contains `unsafe { env::set_var(...) }`. | This Mac built Metal, so the toolchain is present. A checkout that cannot resolve those paths, or a Mac without the Metal toolchain, was not tried. |

## Tests

| ID | Status | Lock | What changed |
| :--- | :--- | :--- | :--- |
| T1 | Verified | `linear_above_two_task_macs_matches_one_thread_and_several` (32×256×256). `linear_one_row_above_two_task_macs_matches_one_thread_and_several` (1×2048×1024). `row_split_above_the_row_grain_matches_one_thread_and_several` (1024 rows, dim 64). All three are in `governor.rs` (11 passed). | Those shapes are at or above the parallel threshold, including a one-row linear and a row split. The comment says `map_rows` is gone. |
| T2 | Verified | `TestPoisonDropsSession` calls `engineReset`, then `Free` on the id, and requires `unknown model`, before `Close`. | A Free that only ran after Close would no longer hide a session that reset left behind. |
| T3 | Partially verified | `linear_forward_refuses_budget_that_fits_output_but_not_packed_weight` asserts `CapacityExceeded`. `clip_and_adam_write_rules` and the capacity case in `zero_extent_inf_offset_and_budget_do_not_corrupt_inputs` assert capacity on a tight AdamW budget and unchanged moments. | `budget_scratch.rs` now refuses when the packed weight does not fit. The infinity-gradient case is still `NonFinite`, which is the right error for that input. The discarded `let _ = before_prefix_budget` in the same redteam test was not removed. |
| T6 | Verified | `TestConcurrentSessionStress`: pool size 1, 64 goroutines, `time.After(4 * time.Minute)`. | The passing Go run finished inside that watchdog. The watchdog path itself (the test failing at 4 minutes) was not the path this run took. |

T4 and T5 from the original plan are not restated here. This pass did not re-audit every `is_err()` site.

## Docs

| ID | Status | Where | What changed |
| :--- | :--- | :--- | :--- |
| D1 | Corrected in `README.md` | `#![forbid(unsafe_code)]` is on the library crates, including `ojas-metal/src/lib.rs`. It is not true for `ojas-cuda` and `ojas-hip` when their features are on (`cfg_attr(not(feature = ...), forbid(unsafe_code))`), for `ojas-gusset-engine` (`unsafe extern "C"`), or for `ojas-metal/build.rs` (`unsafe { env::set_var }`). | The mermaid node no longer says the attribute holds by default for every crate. |
| D2 | Corrected in `README.md` and `docs/architecture.md` | `catch_unwind` catches a panic on a worker and poisons the handle. It does not catch an abort, a failed infallible allocation, or a stack overflow. | The sequence notes say that. |
| D3 | Corrected in `README.md` | `Cargo.toml` sets `overflow-checks = true` on release, and that remains true. `next_id` in `ojas-core/src/tensor.rs` uses `wrapping_add`. Budget release uses `saturating_sub`. | The mermaid no longer says checked arithmetic is the only arithmetic. |
| D4 | Already true in `Cargo.toml`. A code comment is still wrong, and this pass did not edit it. | Workspace `rust-version` is `"1.97"` because gusset declares that. `README.md`, `docs/status.md`, and `docs/architecture.md` do not say 1.82. `ojas-simd/src/arch.rs` still says the workspace MSRV is 1.82. | No rewrite of that comment in this docs pass. |
| D5 | Corrected | The old “227” figure is gone from `docs/status.md`, `README.md`, `site/index.html`, and `docs/index.html`. | Counts are the 449 / 4 ignored run above, and 27 Go tests. |
| D6 | Corrected in `site/index.html` and `docs/index.html` | `Budget::new` returns `Budget`. `CpuBackend::new` takes that value. `Tensor::from_f32` is the constructor. `Tape::new` takes the backend. | The C example no longer includes `ojas.h`. There is no such header in the tree. The host API is the Go package. The C exports are `ojas_engine_init`, `ojas_engine_reset`, `ojas_set_model_root`, and the last-error getters. |

## Dispatcher

**Verified** in the `ojas-capi` suite (41 passed).

`ResourcePolicy::allow_split` defaults to false (`allow_split_defaults_false_and_is_not_inferred`). `step_with_policy` with that default returns `ojas:E_CAPACITY:` when the caller budget cannot hold the request. With `allow_split` set, `split_1000_rows_into_three_unequal_parts` asserts parts `[334, 333, 333]` and the same bits on a second run. A policy whose device list names Metal, Vulkan, or CUDA is refused with a message that contains `refusing CPU fallback`. The plan does not infer `k` from the host probe.

## Unverified

These were not established by a run on this machine:

- CUDA with `--features cuda`, and any NVIDIA kernel.
- HIP with `--features hip`, and any ROCm device. There is no `/opt/rocm`.
- A Linux cgroup limit. No container was used.
- A real GPU hang or a device fault. The Metal poison test sets the shared-event timeout bit.
- 64 live Metal sessions.
- A successful BPE encode of a full 32 MiB input. The data suite refuses `HF_TEXT_CAP + 1` bytes.
- A successful checkpoint of a full 1 GiB. The io suite's over-cap test allocates near 1 GiB and expects refusal. The exact-byte test does not build a 1 GiB success.
- A machine that does not have Xcode, and a tree that cannot see `../../tessl` or the sibling gusset checkout.
- A forced failure of `thread::Builder::spawn`.
- Tessl `argmax_f32_pass_refuses_a_nonfinite_row` (present in source; not run here).

## DevMap

Both `devmap build` commands exited 0. Neither reported FTS corruption, so nothing was added to `.devcouncil/codeintel/sessions/gaps.jsonl`.

The first, before these doc edits, finished in 129ms and said generation `#1616` was still current (179 files). The second, after the doc edits, finished in 103ms and said “No source changes; generation `#1646` still current (340 files).” That is the index’s own report. It does not list `docs/audit-resources.md` as a new source. This session does not claim the new doc was indexed.

## Line delta

Taken after the doc edits in this pass. `git diff --stat` is the unstaged tracked diff. It includes other sessions’ uncommitted work. It does not include untracked files.

| Repo | Tracked diff |
| :--- | :--- |
| ojas | 80 files changed, 11021 insertions, 2623 deletions |
| tessl | 17 files changed, 702 insertions, 142 deletions |
| gusset | 53 files changed, 822 insertions, 139 deletions |

Untracked, so absent from those counts: `docs/audit-resources.md` (this file), `docs/index.html`, and `site/index.html` in ojas; `tests/flash_attn_rows_h64.rs` and `tests/qk_norm_half_rope.rs` in tessl (plus a `target-redteam/` build directory); `submit_permit_internal_test.go` and the `language-policy` skill copies in gusset.

## Commit order

Do not commit from this pass.

Commit **tessl** first, then **gusset**, then **ojas**. ojas `HEAD` already depends on uncommitted tessl APIs (`is_poisoned`, commit feedback, the argmax refusal path) and uncommitted gusset APIs (`register_engine` returning `Result`, `Close` bounded so workers that remain are reported). Committing ojas alone will not build against the published tips of those two trees.
