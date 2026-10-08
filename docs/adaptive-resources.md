# Adaptive resources audit

Recorded 2026-10-02 by the "Ojas system adaptive resources and hardening" session. Labels: **verified** (a run on this machine covers it, or a test asserts it and its suite exited 0), **inferred** (read from source, not run), **unverified** (no run covers it).

Machine: M5 Pro, 64 GiB, 18 CPUs in two performance levels (6 "Super" + 12 "Performance"), macOS (Darwin 27.0). Load average 10–25 during every run below, from peer sessions, so any timing here is a lower bound.

## Audit findings

| ID | Finding | Evidence | Status |
| :--- | :--- | :--- | :--- |
| A1 | The host probe and the plan were unwired. `probe_host` and `ResourcePlan::derive` had only test callers; no backend, capi op or Go call read them. Nothing in ojas adapted to the machine. | DevMap `devmap_neighbors` (callers: tests only), confirmed with `rg -uu` over the tree including `go/` | verified, fixed (see C1–C4) |
| A2 | The probe knew RAM, cgroup memory and `available_parallelism` only: no physical cores, no P/E split, no caches, no page size, no memory architecture, no pressure, no bandwidth. | `ojas-device/src/host.rs` before this change | verified, fixed |
| A3 | The cgroup memory reader looked at the leaf cgroup only. cgroup limits are hierarchical, so a container whose own `memory.max` is `max` inside a pod with a 4 GiB limit reported no limit. | Old `probe_cgroup` read `/sys/fs/cgroup/<leaf>/memory.max` only (source). `a_tighter_parent_cgroup_limit_is_not_missed` now passes; it was not run against the old reader, which had no fixture root to point at | gap inferred from source; fix verified |
| A4 | Probe files were read with `read_to_string` and no size bound. | `read_text_refuses_past_the_cap_and_accepts_at_it` | verified, fixed (64 KiB cap) |
| A5 | `ResourcePlan` had no notion of unified memory: a GPU's memory figure was reported beside the host budget as if separate. On Apple silicon both come out of the same RAM, so a caller budgeting each would double-count it. | `ojas-device/src/plan.rs` before this change | verified, fixed (C3) |
| A6 | macOS `available_bytes` is free + inactive pages. On this machine that read about 28 GB, while the kernel's `kern.memorystatus_level` reported 82% (about 52 GB) available. The figure is conservative, not wrong in the unsafe direction. | `vm_stat`, `sysctl kern.memorystatus_level` at 10:20 | verified; **left as is** (see Decisions) |
| A7 | tessl's `probe_system_memory_size` shells out to `sysctl -n hw.memsize` and returns 0 on any failure. | `../tessl/src/runtime.rs:1970` | inferred from source; out of this lane (tessl repo), not changed |
| A8 | Fixed sizes that do not scale with the machine: capi `DEFAULT_MEMORY_CEILING_BYTES` and `DEFAULT_BUDGET_BYTES` (1 GiB), Go `defaultBufferBudget` (64 MiB), wgpu `POOL_CAP_BYTES` (512 MiB), Metal `SESSION_POOL_CACHE_BYTES` (64 MiB), GEMM blocking (`FAST` 512/144/512, `EXACT` 256/72/256). | source | inferred; defaults not changed (see Decisions) |
| A9 | DevMap dead-symbol pass: nothing in `ojas-device` flagged (test callers count, which is why A1 never appeared as dead code). High-confidence rows are in other lanes: `wireSegmented`/`wireTabs` in `site/assets/ojas.js` and `docs/assets/ojas.js`, `GENERATED_DIR` in `ojas-oracle/python/common.py`, `CeCase` in `ojas-qwen35-cuda/src/small_smoke.rs`. | `devmap_dead_symbols`; the answer carried `walk_incomplete` (50,682 unresolved sites repo-wide) | recorded, not acted on |

## Changes

| ID | Change | Where |
| :--- | :--- | :--- |
| C1 | `probe_system()` → `SystemProfile { memory, cpu, architecture, pressure }`. `CpuTopology`: logical, physical, usable CPUs; core clusters fastest first with L1d, L2 and CPUs per L2; L3, cache line, page size; cgroup CPU quota. macOS reads `hw.perflevelN.*`; Linux reads `/sys/devices/system/cpu` (Intel `cpu_core`/`cpu_atom`, arm64 `cpu_capacity`). Every field fails on its own to `Unknown`. | `ojas-device/src/{system,topology,sysctl}.rs` |
| C2 | `measure_bandwidth` / `cached_bandwidth`: opt-in copy bandwidth on 1 and N threads, min of repetitions, fallible bounded allocation (1–64 MiB per buffer for every caller; two buffers shared by the threads; `for_profile` may shrink each to 1/16 of available memory), bounded wall time, load average recorded. A spawn failure part-way releases the spawned workers through a latch rather than leaving them on a barrier. | `ojas-device/src/bandwidth.rs` |
| C3 | `ResourcePlan::derive(policy, &SystemProfile, probes)`: adds `fast_threads`, `memory_bound_threads` (from `with_bandwidth`), `device_room` (device memory less what the process already holds), `device_shares_host`, `shared_budget`, `architecture`, `pressure`, `cpu_quota_millis` and `cache`. A device that shares host memory has its room capped at `budget_bytes`, and `shared_budget` tells the caller to charge CPU and GPU to one `Budget`. `MemoryProbe` gains `resident_bytes` and `architecture` with `Unknown` defaults. | `ojas-device/src/plan.rs` |
| C4 | `GemmBlocks::derive(&CacheBudget, mr, nr, elem_bytes)`: kc from the smallest L1, mc from the smallest L2 share, nc from the L3 (simplified from Low et al. 2016); nc is `None` when no L3 is reported (G1). Recommendation only. On this machine: kc 368, mc 474, nc none. | `ojas-device/src/tuning.rs` |
| C5 | cgroup memory and CPU quota take the tightest of the cgroup and its ancestors (A3); probe files are read with a 64 KiB cap (A4). | `ojas-device/src/host.rs`, `topology.rs` |
| C6 | `MetalBackend::memory()` → `MetalMemory { recommended_working_set, allocated }`, an `ojas_device::MemoryProbe`. Answered by the device thread outside the command path (no commit trigger), and still answered on a poisoned backend. | `ojas-metal/src/{memory,backend,device,link}.rs` |
| C7 | capi opcode 16 `SYSTEM_PROFILE` and Go `SystemProfile(ctx, budget)`: a versioned record of the profile and plan with a known flag per field. Reads only. The Go decoder refuses a wrong version, short count, length mismatch, bad flag, or an unknown field that carries a value; a fuzz target checks it never panics. | `ojas-capi/src/profile.rs`, `go/profile.go` |

## Decisions

- **Defaults are unchanged.** `CpuBackend::new` stays serial, capi's ceiling stays 1 GiB, thread count 0 is still refused. ojas is explicit-over-inferred (`allow_split` is not inferred; `TestCPUParallelThreadCountIsBounded` locks the zero-thread refusal), and raising a memory default from a probe would widen what the process may use without the caller asking. Adaptivity is opt-in: a Go host calls `SystemProfile`, then `SetMemoryCeiling` and `LoadOptions.Threads` with numbers it chose.
- **Unified memory: one logical budget, not a physical guarantee.** capi already charges every session, CPU and Metal, to children of one root ceiling (`session_budget` → `ceiling.root.child`), which is the shared-budget rule. The plan states it (`shared_budget`, capped `device_room`) for callers outside capi. That accounting is in logical bytes for the general `MetalBackend` (`reserve` charges `elems * 4`, `ojas-metal/src/backend.rs:271-273`; only the tiny `gpu.rs` step charges power-of-two buckets). `ojas-metal/src/backend.rs:9-14` states that tessl rounds each buffer up to a power of two, so resident device memory can be up to twice the charged bytes, and tessl's pool cache (capped at 64 MiB per session) is outside the budget. On unified memory that is RAM the host also needs, so a host must leave headroom (documented on Go `SetMemoryCeiling` and `Profile.BudgetBytes`). Charging the physical bucket is a tessl/ojas-metal change, not made here.
- **A6 kept conservative; review round 3 concurs.** `available_bytes` feeds the budget directly (`budget = tighten(budget, host.available_bytes)`, `ojas-device/src/plan.rs:111`), so switching to the kernel's figure would raise every derived budget by about 24 GB on this machine: a widening. What `kern.memorystatus_level` counts was not verified (no XNU source read), and on a shared Mac it plausibly includes pages peer processes are using. A second "kernel available" field was rejected too: no wire slot and no reader would use it. Widening stays the user's call; no code changed.
- **GEMM blocks are not adopted.** On macOS, Fast products at or above 2^13 multiply-adds go to Accelerate, so the packed blocks matter only for small products and Exact. The CPU lane (session 2f9dc4) owns `ojas-cpu` and will adopt them only after an interleaved A/B benchmark. `ojas-cpu` gives each thread its own `(rows, cols)` tile and traversal, so if it adopts `nc` it must divide it by the threads sharing the L3.
- **No Go helper that sets the ceiling from the profile.** "Never raise a lower existing ceiling" cannot be implemented in Go (no opcode reads the ceiling back), and the headroom fraction is a policy choice a helper would make silently. Without a default fraction the helper is one line. Instead, the `go/README.md` example now derives the ceiling from `SystemProfile` with an explicit fraction, checks for 0, and loads with that budget.
- **No `CpuBackend::from_plan`.** It would add an `ojas-device` dependency to `ojas-cpu` to wrap `with_threads(Budget::new(plan.budget_bytes), n)`.

## Verification

### Gate 1 (10:25, lock held)

`CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=target-adaptive`:

- `cargo test -p ojas-device --release`: **58 passed**, 0 failed (verified). Was 18.
- `cargo clippy -p ojas-device --all-targets -- -D warnings` on aarch64-apple-darwin, x86_64-unknown-linux-gnu and x86_64-pc-windows-msvc: clean after two fixes (an unused helper, a no-op `min` in a test) (verified).
- `cargo check -p ojas-device --all-targets` for x86_64 and aarch64 Linux and x86_64 Windows: exit 0 (verified compile only; **not run** on Linux or Windows).
- `cargo test -p ojas-metal --release --lib memory`: 3 passed, including `memory_probe_tracks_residency_and_plans_one_shared_budget` (a 16 MiB upload shows in `allocated`; eight concurrent queries answer; on Apple silicon the plan sets `shared_budget` and caps the Metal room; the probe still answers after an injected device-thread panic) (verified).

What the tests attack: hostile `/sys` and `/proc` text (reversed and 2^64 CPU ranges, empty tokens, oversize files, bad units, overflowing sizes, `..` and NUL paths, v1 `-1` quotas, `max`), each missing file blanking only its own field, a tighter parent cgroup, 1024 bandwidth threads over 1 MiB, every out-of-range bandwidth config refused before allocating, concurrent probing from 16 threads, an exhaustive sweep of host × GPU architecture × memory × residency × available × caller budget asserting a shared device's room never passes the budget, thread advice never passing the ceiling, and GEMM blocks satisfying the model's capacity constraints across L1/L2/LLC/tile/element sweeps.

### Gate 2 (10:41–11:00, lock held)

- `cargo test -p ojas-capi --release`: **70 passed**, 0 failed, including `system_profile_routes_through_dispatch_and_reads_only` (opcode 16 through `engine::dispatch`; the memory ceiling is unchanged afterwards) and the profile unit tests (field order equals wire order; payloads of 1/7/9/16 bytes and a 0 budget are refused) (verified).
- `cargo clippy -p ojas-capi -p ojas-device -p ojas-metal --all-targets -- -D warnings`: clean (verified). The first run failed on this lane's code (`Field` variants never constructed, `chunks_exact` with a constant); the record is now built by an exhaustive `match` over `Field::ALL`.
- `cargo build -p ojas-gusset-engine` into `target-adaptive`, linked by a lane-local `gusset.pc` so the shared `target/debug/libgusset.a` was not overwritten (verified).
- `go vet` and `go test -tags gusset_pkgconfig` (after one `-a` build against the new archive): **ok in 17.8 s**, including `TestSystemProfileFromTheEngine`, which loads a `DeviceCPUParallel` model with the profile's thread ceiling and runs a step (verified).
- `go test -fuzz FuzzDecodeProfile -fuzztime 15s`: 3,336,694 executions, no failure (verified).
- `cargo run -p ojas-device --example profile -- 8589934592 --bandwidth`: the profile below (verified).

**Incident, this lane's fault.** The first Go run was killed by the OS after 491 s, with load at about 34. A test case built its "count overflows" record by generating 4,294,967,295 entries (about 38 GB) and then slicing it. The case now overwrites the count field of a 16-entry record. The rerun finished in 17.8 s. A machine audit later found a JetsamEvent at 10:55 for a `go.test` process at 84–120 GiB resident; by timing this was very likely that run (inferred, not matched by PID). The "Find the 120 GiB go.test memory blowup" session later re-ran the old fixture under a 1 GiB cgroup, where it was OOM-killed (`memory.events` `oom_kill 1`). It also measured the current full `go test -a ./...` at a peak test-binary RSS of 212 MiB (its report, not re-run here). The current profile tests are bounded by construction: the decoder allocates a fixed 20 entries whatever count a record claims, and the one bandwidth measurement uses at most two 64 MiB buffers.

**Flake found by a peer, this lane's fault.** The full `cargo test -p ojas-metal --release` run by the "Rust/Go ML packages" session (about 11:05) failed `memory_probe_tracks_residency_and_plans_one_shared_budget`: a 16 MiB upload grew `allocated` by only 2.25 MiB. `MTLDevice.currentAllocatedSize` is process-wide, and the other lib tests free buffers in parallel between the two reads; my `--lib memory` run had filtered them out. The test now asserts a lower bound that other tests cannot break (`allocated >= 16 MiB` while the tensor lives), and `MetalMemory::allocated` is documented as the whole process's figure. The rerun result is under "Gate 3".

### Gate 3 (11:04, lock held)

`cargo test -p ojas-metal --release --lib`, the full lib suite, three runs: **53 passed** each time, 0 failed, with the fixed test running in parallel with the rest (verified). `cargo clippy -p ojas-metal --all-targets -- -D warnings`: clean (verified).

### DevMap

- `devmap_status` at the end: generation 3731, 8,321 nodes, 40,979 edges, `is_fresh: true`, no parse failures. HTML and CSS are pattern-recovered only (no grammar), and Rust net resolution is 38%, so edge answers are lower bounds.
- `devmap_clones` (min 40 nodes): 287 groups, 41 shown, `truncated: true`. None involve this lane's files. The largest are the `docs/assets/ojas.js` and `site/assets/ojas.js` exact copies, the `swiglu_f32`/`swiglu_bf16` pair in `ojas-qwen35-cuda/src/k8.rs`, and the per-backend `shape_first.rs` test helpers. Recorded, not acted on: other lanes.

### What ojas sees on this machine

| | |
| :--- | :--- |
| RAM | 64 GiB total, 43.3 GB available (free + inactive) at 10:55 |
| CPUs | 18 logical = 18 physical = 18 usable |
| Clusters | "Super": 6 cores, L1d 128 KiB, L2 16 MiB per 6. "Performance": 12 cores, L1d 64 KiB, L2 8 MiB per 6 |
| L3 | not reported (Apple's system cache is not exposed) |
| Line / page | 128 B / 16 KiB |
| Architecture / pressure | Unified / Normal |
| Copy bandwidth (load 4.7) | 120.8 GB/s on 1 thread; 409.9 GB/s on 18 (read + write bytes) |
| Plan for an 8 GiB budget | budget 8 GiB; thread ceiling 18; fast threads 6; memory-bound threads 4 (ceil(409.9 / 120.8)) |
| Cache budget → GEMM blocks | L1d 64 KiB, L2 share 1.33 MiB, L3 not reported → kc 368, mc 474, nc none (until G1 the L2 stood in for L3 and gave nc 2848) |

## Review round (Fable)

An independent review (claude-fable-5-1, read-only) found these; each was checked against the code before acting.

| ID | Finding | Action |
| :--- | :--- | :--- |
| R1 | cgroup `memory.current` counts page cache, so a container that has just read a large model sits near its limit and the plan's budget collapses toward 0. | Fixed: the room now uses the working set, `current − inactive_file` (v1 `total_inactive_file`) from `memory.stat`, as the kubelet does; an unreadable `memory.stat` keeps the raw figure. Tests: `page_cache_does_not_count_against_the_cgroup_room` (by the old formula, limit − current, the room was 0.1 GiB; it is now 3.1 GiB. Not run against the old code), `an_unreadable_memory_stat_keeps_the_raw_usage`. This raises the room relative to the previous code, but only by pages the kernel reclaims before an OOM kill. |
| R2 | `cached_bandwidth` cached an error forever. | Fixed: only a success is kept. Test `a_refused_measurement_is_not_cached_and_a_success_is` (its first call is a `Config` error, which the `OnceLock` version would have returned forever; inferred, not run against it). |
| R3 | The one-ceiling rule is logical bytes; Metal resident memory can be twice that plus the pool cache. | Verified (`MetalBackend::reserve` charges `elems * 4`). Documented in Decisions, Go `SetMemoryCeiling` and `Profile.BudgetBytes`. Not changed in code. |
| R4 | `derive` substituted the budget for a shared device that reported no memory. | Fixed: that room stays `Unknown` and `shared_budget` carries the bound. The sweep test now fails if a room is invented for an unreported device. |
| R5 | Docs said bandwidth takes 1/16 of available memory; two buffers take 1/8. | Fixed in code docs, README and here. |
| R6 | Opcode 16 never returned bandwidth, so `memory_bound_threads` was Rust-only. | Fixed: payload `budget: u64, flags: u32` with `FLAG_BANDWIDTH`; four appended fields; Go `SystemProfile(ctx, budget, measureBandwidth)`. Unknown flag bits are refused. |
| R7 | `MetalMemory::architecture()` is always `Unknown`; on an Intel Mac with an integrated GPU the plan would treat it as separate memory. | Not fixable in round 1: tessl did not expose `MTLDevice.hasUnifiedMemory`. Fixed in round 2 (C19), once tessl added the flag. |
| R8 | `fast_threads` is "Apple perflevel0", which on an M5 Pro is 6 of 18 performance-class cores, not "the P-cores". | Go doc reworded. |
| R9 | `GemmBlocks` sizes `nc` as if one B panel is shared by every core on an L2. | Superseded by G1: `nc` now comes from the L3 only. The shared-panel assumption is documented on `GemmBlocks::nc`; the per-thread division was sent to the CPU lane. |

### Review round 2 (Fable)

The same reviewer re-read the fixes. Verdict: B1–B5 and R6 closed. It found that the R1, R2, R4 and R6 tests fail against the pre-fix code (read, not run). Three follow-ups:

| ID | Finding | Action |
| :--- | :--- | :--- |
| N1 | A `memory.stat` claiming more `inactive_file` than the usage file held gave a working set of 0, so the whole limit was room: fail-open on contradictory input. | Fixed: the raw usage is kept when `inactive_file > current`. The test row now expects 900 (it expected 0 before). An exact-usage row was added. |
| N2 | The opcode-16 bandwidth measurement is not cancellable once started and holds a process-wide lock. | Documented in `ojas-capi/src/profile.rs` and Go `SystemProfile`: bounded to about half a second; concurrent callers wait. |
| N3 | A new Go decoder against an old engine fails with "16 fields, want at least 20". | Accepted: fail-closed, and the engine and Go package ship together. |
| N4 | `TestSystemProfileFromTheEngine` hard-failed when the engine refused to measure on a host with under 16 MiB available. | The test now skips in that case only, and still fails on any roomier host that does not measure. |

### Gate 5 (12:41, after review round 2, lock held)

All verified on this machine: `ojas-device` **61 passed** (including the flipped N1 row); clippy `-D warnings` clean on macOS and Linux x86_64; `ojas-capi` profile tests 5 passed; `ojas-gusset-engine` built; `go vet` clean; `go test -a -run Profile` passed `TestDecodeProfileRefusesWhatItCannotReadWhole`, `TestSystemProfileFromTheEngine` (it measured and did not skip) and the `FuzzDecodeProfile` seeds.

### Gate 4 (12:11, after the review fixes, lock held)

All verified on this machine: `ojas-device` **61 passed**; clippy `-D warnings` clean for macOS, Linux x86_64 and Windows; `ojas-capi` **71 passed** (including the bandwidth-flag and unknown-flag tests); clippy clean on capi, device and metal; `ojas-metal --lib` **53 passed**; `ojas-gusset-engine` built; `go vet` clean; `go test -a` **ok in 18.6 s** (including `SystemProfile(ctx, 0, true)` returning a memory-bound thread count within the ceiling); `FuzzDecodeProfile` 3,367,234 executions in 15 s, no failure.

### Review round 3 (Fable): the open decisions

The same reviewer (read-only; no Bash or DevMap in that run) was asked for a recommendation on each open decision. A6 and the Go helper are under Decisions. One finding was a bug:

| ID | Finding | Action |
| :--- | :--- | :--- |
| G1 | With no L3 reported, `CacheBudget` put the L2 in as the last-level cache, so `mc` (half of a core's L2 share) and `nc` (half of the "last level") were sized from the same cache. On this machine that is six 0.67 MiB A blocks plus a 4 MiB B panel in one 8 MiB L2, and 28 MiB if each thread packs its own panel, as `ojas-cpu` does. The published `nc 2848` was that figure. Neither test caught it: one asserted 2848, and the sweep checked `mc` and `nc` independently. | Fixed: `CacheBudget` carries `l3` (only when the OS reports one) in place of `l2` and `last_level`, and `GemmBlocks::nc` is `Option<usize>`, `None` without an L3; `kc` and `mc` are unchanged. The wire field `LastLevelBytes` became `L3Bytes` (Go `Profile.L3Bytes`), so Go and Rust report the same fact. Tests: `m5_pro_numbers_give_blocks_that_fit` now expects `nc: None`, and `a_reported_l3_sizes_the_b_panel_and_nothing_else_does` checks that a machine without its L3 keeps `kc` and loses `nc` however large its L2 is. Against the pre-fix code these fail by not compiling (`nc` was a `usize`); the old code returned 2848 for the same input, so there was also a behavioral failure. |

### Gate 6 (about 18:21–18:25 UTC, after review round 3, lock held)

All verified on this machine:
- `ojas-device`: **62 passed** (the new L3 test plus the reworked `nc` assertions). Clippy with `-D warnings` is clean on macOS, Linux x86_64 and Windows.
- The `profile` example prints `GemmBlocks { kc: 368, mc: 474, nc: None }`.
- `go vet` is clean, and the `go/README.md` example, extracted to a scratch module, passes `go vet` against the live package.
- `FuzzDecodeProfile` passed its 15 s run.
- The first `ojas-capi` and engine steps failed to compile inside `ojas-cpu`, which the CPU lane was editing at the time. Its first Go run linked the previous engine archive, so it does not count. After the CPU lane reported `ojas-cpu` compiling, the rerun gave: `ojas-capi` **71 passed**; clippy clean on capi, device and metal; engine built; `go test -a` **ok in 20.6 s** against the new archive.

### Linux run (about 18:22 UTC, Podman VM, lock held)

A real Linux kernel: the local `podman-machine-default` VM (applehv, 9 CPUs, 12 GiB, Fedora kernel 7.1.3, cgroup v2), using only the image already on the machine (`python:3.11-trixie`, `--pull=never`). The binaries are static x86_64 musl builds linked on the Mac with `rust-lld` and run through the VM's qemu-x86_64 binfmt handler. `/sys` and the cgroup files are the VM kernel's own. All verified:

- `ojas-device` unit suite: **57 passed** with no limits, and 57 passed inside `--memory=1g --cpus=2`. The five macOS-only tests are the three `sysctl::tests` (the `sysctl` module is macOS-only), `macos_reports_perf_levels_and_caches` and `macos_total_ram_is_a_sysctl_value`; 62 − 5 = 57 matches the Linux count. The names come from the source cfgs and the macOS `--list`; the Linux list was counted, not diffed.
- In a 1 GiB / 2-CPU container, the probe read `cgroup_limit_bytes` 1,073,741,824 and `cpu_quota_millis` 2000; `usable` was 2 of 9, and the plan's thread ceiling was 2. For an 8 GiB request, `budget_bytes` was 1,066,573,824 (the limit less the 7 MB working set).
- **Page cache (R1) on a real kernel.** After writing and reading 700 MiB in the same container, `memory.current` was 737,943,552, of which 734,117,888 was `inactive_file`. The plan's budget stayed at 1,061,515,264. The old formula (limit − current) would have given about 336 MB.
- With no limits: total 12.5 GB, available 11.8 GB, 9 CPUs, no cgroup limit, budget 8 GiB as asked.
- Bandwidth was measured inside the limited container on 2 threads (`memory_bound_threads` known). The figures, about 3.0 and 6.0 GB/s, are of qemu-emulated x86 code and say nothing about the hardware.
- **Caches were all `Unknown`, which is correct for this VM.** Its kernel publishes `cache/index*/level`, `type` and `shared_cpu_list` but no `size` or `coherency_line_size` (checked with `ls` and `cat`), and `getconf` reports sizes of 0. The readers failed closed; `GemmBlocks::derive` returned `None`. A Linux box that publishes cache sizes has still not been run.

## Unverified

- Windows at run time. Linux on bare metal, an x86 Linux host, and a kernel that publishes cache sizes (the VM above does not).
- An Intel Mac (`hw.nperflevels` absent; architecture `Unknown`).
- A discrete GPU through `MemoryProbe::architecture() == Discrete` (only stubbed in tests).
- Bandwidth figures as performance numbers: every measurement ran with load 4–25.
- Any speedup from `GemmBlocks`; none is claimed.
- The full `ojas-metal` integration suite (`tests/`) on this lane's own run. It compiled under `clippy --all-targets`; a peer ran it at about 11:05 and reported 162 passed beside the flake above.
- `ojas-cuda` and `ojas-hip` were not built. Their only imports from `ojas-device` are `require_kind`, `Device` and `DeviceError` (`ojas-cuda/src/lib.rs:11`, `ojas-hip/src/lib.rs:13`), all unchanged. `ojas-wgpu` (which also imports `DeviceInfo`) compiled as a capi dependency.

## Limitations

- **Nothing adapts on its own.** That is the design (see Decisions): the probe and plan measure and recommend, and the caller adopts.
- **(Superseded 2026-10-07: `DeviceProfile`, below.)** **The Go `Profile` has no GPU figures.** Opcode 16 opens no device, so it carries the unified-memory flag but no Metal working set, residency or room. Those are reachable only in Rust, through `MetalBackend::memory()` and `ResourcePlan::derive`.
- `docs/status.md` lists `ojas-device` at 18 tests only inside dated run records (2026-10-01), which are correct for those runs; the current count belongs in the next integration run's table. This lane changed only the component diagram line (line 243); the file's other uncommitted hunk (line 124) belongs to the "Rust/Go ML packages" session.

## Size

Tracked diff of this lane: about +760 / −80 lines across `ojas-device` (host, plan, lib, manifest, README), `ojas-metal/src` (backend, device, link, lib; the `gpu.rs` hunks there are another session's), `ojas-capi` and `go/ffi.go`. 22 of `go/ffi.go`'s 42 changed lines are gofmt realigning one constant block. New files: about 1,340 non-test Rust lines (`bandwidth`, `topology`, `tuning`, `system`, `sysctl`, capi `profile`, metal `memory`) and about 890 test lines, 141 Go plus 162 Go test lines, and a 57-line example. Most of the growth is the Linux `/sys` readers and their fixture tests. Nothing was deleted: the unwired probe was wired, not replaced.

# Round 2: fail before the work, not part-way

Recorded 2026-10-02 (evening) by the "Ojas system adaptive hardening" session, which took over this lane. The goal: a run that cannot fit is refused before compute is spent, never killed or poisoned part-way, and the plan acts on what it measures. Same labels as above.

## Audit findings

Four read-only audits ran: training/inference preflight, budget coverage across backends, the probe and plan, and thread adaptivity. The audit subagents had no DevMap tools, so their caller lists come from Grep over tracked source. DevMap itself then failed mid-session (see Gaps). Each finding below was checked against the code before acting.

| ID | Finding | Evidence | Status |
| :--- | :--- | :--- | :--- |
| P1 | An optimizer refusal poisoned the trainer. Muon reserves its scratch inside `apply`, after AdamW has already updated earlier slots, so a `CapacityExceeded` there left parameters partly updated, and a poisoned trainer cannot Save. Forward, backward and clip were already transactional (they change nothing the trainer owns). | `trainer.rs` `run`: only `apply`/`finish` set `Poisoned`; CPU `backend.rs` `headroom(muon_scratch)`, Metal `6n+3r²+2048`, wgpu `job.scratch` | verified; fixed (C8) |
| P2 | Nothing compared a run's memory to the budget before the work: `Trainer::new`, resume and load charged as they allocated. | source | verified; fixed (C8, C9) |
| P3 | `load_params` decoded every tensor into uncharged `Vec<f32>`, so the peak was about 2× the parameters while the budget saw 1×, and it charged only at the end. It also read every earlier tensor before discovering a missing one. Resume already used the charged chunked reader. | `load.rs` before; `checkpoint.rs` resume path | verified; fixed (C9) |
| P4 | `n_layer` had no upper bound. `param_table` builds about 14 named rows per layer before any budget is consulted, reachable from file metadata and from NEW over the wire: a process abort from input alone. | `spec.rs` `validate`, `names.rs` `param_table` | verified (path); fixed (C10) |
| P5 | The RoPE tables were built in uncharged `vec!`s and copied into charged tensors (2× the tables, uncharged). | `block.rs` `Rope::rows` | verified; fixed (C10) |
| P6 | `set_memory_ceiling` accepted any non-zero `u64`. Above RAM, every charge passes and the process dies of the OS's OOM kill instead of a refusal. | `session.rs` | verified; fixed (C11) |
| P7 | A Metal `head_dim > 128` was refused only at the first attention op, after the full upload. | `refuse_unsupported_metal_head_dim` callers | verified; fixed (C11) |
| P8 | CPU pool workers spawn lazily, on the first op large enough to split. A spawn failure surfaced inside a step, possibly inside `apply`. | `pool.rs` `ensure_workers` | verified; fixed (C12) |
| D1 | Memory pressure was reported and read by nothing: Critical gave the same budget as Normal. | `plan.rs` copied `profile.pressure` | verified; fixed (C13) |
| D2 | `thread_ceiling` ignored the known cgroup CPU quota and relied on std having applied it. | `plan.rs` | verified; fixed (C13) |
| D3 | An unreadable `memory.current` with a known limit counted as zero usage, contradicting its own doc comment. | `host.rs` `cgroup_room` | verified; fixed (C14) |
| D4 | cgroup v2 `memory.high` (systemd `MemoryHigh=`) was never read. | `rg` over `ojas-device` | verified; fixed (C14) |
| D5 | Opcode 16 with an empty payload on a host where nothing is readable sent `BudgetBytes` known at `u64::MAX`, and the Go doc said "Always known". | `profile.rs` | verified (path); fixed (C15) |
| D6 | A 4-byte sysctl holding `-1` was zero-extended to 4294967295 and passed `n > 0` checks. | `sysctl.rs` | verified; fixed (C14) |
| D7 | Found by Gate C on Linux. The bandwidth heap tests read process-wide allocator counters, while sibling tests ran `measure_bandwidth` in parallel, so a sibling's buffers counted against the pair under test. L1 failed `many_threads_share_two_buffers`. On macOS, `bandwidth::` at `--test-threads=2` failed 1 of 10 rounds (190 MiB against a 128 MiB pair), and 0 of 10 alone. The probe itself was never wrong. | `bandwidth.rs` tests, since 557b85c | verified; fixed (C17) |
| D8 | Found by Gate C on Linux, and caused by this round's own C13: the ceiling rounded the quota up, but std's `available_parallelism` rounds it down. A 1.5-CPU container therefore gave ceiling 1 when std read the quota, and 2 when only the probe did. The doc and the unit test claimed 2. | `plan.rs` `thread_ceiling` | verified; fixed (C17) |

## Changes

| ID | Change | Where |
| :--- | :--- | :--- |
| C7b | `Budget::peak_bytes` / `reset_peak`: a high-water mark that records each charge's exact value, so concurrent releases cannot hide a peak. It can over-report by a parent charge that a child rolled back, never under-report. `Budget::check_room(bytes)`: a point-in-time "would this fit the whole chain" query that charges nothing. | `ojas-core/src/budget.rs` |
| C7c | `Backend::optimizer_scratch_bytes(kind, rows, cols) -> Option<u64>`, an upper bound on what one AdamW or Muon call reserves beyond its operands. Default `None`; forwarded by `&B`, `Arc<B>`, capi `Gated` and the test `Probe`. CPU and Metal compute the figure from the same helper their step uses. wgpu has a closed form whose pieces are listed on `optimizer_scratch`. | `ojas-core/src/backend.rs`; `ojas-cpu`, `ojas-metal`, `ojas-wgpu` `backend.rs` |
| C8 | Trainer preflight, in four places. (1) `Trainer::new` refuses before its first allocation when the state cannot fit: values, moments and RoPE (`state_bytes`). (2) Before any compute, each step checks room for the optimizer scratch, or for the measured peak of an earlier step no smaller in rows or micro-batches, whichever is larger. (3) After clip, a last `check_room(optimizer_scratch)` runs while every parameter is still unchanged. (4) Each completed step records its peak. Capacity therefore never poisons a trainer. | `ojas-model/src/trainer.rs` |
| C9 | `load_params` checks the header first (every name, dtype and shape, the tie head's dtype and shape, and the total bytes against the budget) before reading any data. It then decodes each tensor in bounded chunks into storage charged before it is allocated, and compares the tied head against the embedding in 256 KiB pieces. Resume runs the same `state_bytes` check before its first read. Save checks room for its largest device download before creating the file. | `ojas-model/src/load.rs`, `checkpoint.rs` |
| C10 | `MAX_LAYERS = 4096` in `ModelSpec::validate`. `Rope::rows` fills charged tensors in place. | `ojas-model/src/spec.rs`, `block.rs` |
| C11 | capi: a ceiling above `hard_memory_limit()` (physical RAM, or a tighter cgroup limit) is refused with `E_CAPACITY`, and the 1 GiB default starts at that limit on a smaller machine. That is a tightening, never a raise. `preflight` runs before the device opens: a Metal `head_dim` the kernels cannot run, and parameters larger than the session budget. | `ojas-capi/src/session.rs`, `load.rs` |
| C12 | `CpuBackend::start_workers()`; capi calls it when a parallel CPU session opens, so a spawn failure refuses the load. | `ojas-cpu/src/pool.rs`, `backend.rs`; `ojas-capi/src/model.rs` |
| C13 | `ResourcePlan::derive` sets the budget to 0 under Critical pressure; Warning is report-only. The thread ceiling is cut to `quota_millis / 1000` rounded down, at least 1 (see C17). | `ojas-device/src/plan.rs` |
| C14 | When cgroup usage is unreadable, this process's `VmRSS` serves as a floor (a strict lower bound on the cgroup's usage). The limit is the tighter of `memory.max` and `memory.high` on every level. 4- and 8-byte sysctls are decoded as signed, and a negative value is `None`. | `ojas-device/src/host.rs`, `sysctl.rs` |
| C15 | Opcode 16 sends an unbounded budget as unknown (decoded as 0 in Go). The Go doc now lists the three cases that give 0. | `ojas-capi/src/profile.rs`, `go/profile.go` |
| C16 | Adaptive, opt-in or fail-closed only. **Admission under pressure:** the engine refuses LOAD, NEW, TRAIN_OPEN, TRAIN_STEP, RESUME, SAMPLE and GENERATE with `E_PRESSURE` (C18; first drafted as `E_CAPACITY`) while the kernel reports Critical pressure (one sysctl per call); FREE, SAVE and the queries still run, so a host can always checkpoint and release. **`DEVICE_CPU_AUTO` (4)** / Go `DeviceCPUAuto`: a parallel CPU session sized from the plan's thread ceiling, quota included, at most 256. It refuses rather than guesses when the CPU count is unreadable. | `ojas-capi/src/engine.rs`, `load.rs`; `go/api.go`, `go/ffi.go` |
| C17 | Each test that can allocate bandwidth buffers holds one test-only lock (`heap_limit::measuring()`, which tolerates poisoning). `heap_growth` takes the guard as an argument, so a heap check cannot compile without it. The quota cut rounds down, as std does, so both paths agree and a pool never outruns its quota. | `ojas-device/src/bandwidth.rs` tests, `plan.rs` |
| C18 | `ErrorKind::Pressure` / `ojas:E_PRESSURE:` / Go `ErrPressure`, used only by `engine::admit`; its message now begins "memory pressure:". Go decodes every kind from one table, `inBandKinds` in `ffi.go`, which replaces a five-way switch and three hand-copied sentinel lists in the tests. Two guards: a Rust test makes every `ErrorKind` list its prefix (an exhaustive match), checks the prefixes are distinct and checks each one appears in `go/ffi.go`; `TestInBandKindsTable` pins the Go table and checks its sentinels and prefixes are distinct. Docs: `go/api.go`, `go/README.md`, `ojas-capi/README.md`, and both copies of the site page. | `ojas-capi/src/lib.rs`, `engine.rs`, `tests.rs`; `go/ffi.go`, `api.go`, `governor_test.go`, `model_test.go` |
| C19 | R7 adopted at the user's request. `MetalMemory` carries `has_unified_memory` from tessl's `DeviceMemoryInfo` (one lock read, not two), and `architecture()` reports `Unified` or `Discrete` from the device itself instead of `Unknown`. The plan therefore puts Metal inside the shared host budget on the device's own word, even where the host profile is `Unknown`. A unit test covers both flags against all three host answers; the live Metal test asserts this machine reports unified memory and that the plan shares the budget. | `ojas-metal/src/memory.rs`, `device.rs`, `backend.rs` tests |

## Decisions

- **Nothing raises a budget, ceiling or thread count by default.** The prior ruling stands: every change here refuses earlier or tightens, and the one adaptive sizing (`DEVICE_CPU_AUTO`) is a new value a caller must choose.
- **Admission under Critical pressure is on by default.** It only refuses, and only work that would allocate, while the kernel is already killing to stay up. A refused call leaves the session and trainer exactly as they were. On Linux, pressure is `Unknown` (PSI is not read), so this is macOS-only today. (Superseded 2026-10-07: Linux reads PSI, below.) Fable confirmed keeping it on: with `E_PRESSURE` distinct (C18), a refused caller can back off and retry the same call.
- **Resolved (C18, on Fable's advice, at the user's request): `E_PRESSURE` is its own kind.** The pressure refusal first reused `E_CAPACITY`, which gave that kind two contracts: "this will not fit; stop" and "the machine is short right now; wait". `errors.Is(err, ErrCapacity)` could not tell them apart, so a Go loop that treats `ErrCapacity` as fatal would have aborted a run that could have resumed. `E_BUSY` was not reused either: it is per-model and retries at once, while pressure is machine-wide and wants a back-off. This is a public API addition. An older Go package against a newer engine receives the pressure refusal as a plain error with no sentinel; it never misreads it as `ErrCapacity`.- **Prepaid session mode: adopted as an opt-in mode by the user (2026-10-08); not yet implemented ([`tasks/gp-prepaid-session-mode.md`](../tasks/gp-prepaid-session-mode.md)).** The default stays a point-in-time check, so no load that succeeds today is refused by default. First recorded as deferred to the user (Fable concurred). The narrow case it would close: two sessions sharing the ceiling, where the other session takes room between this session's last `check_room` after clip and `apply`. A refusal inside `apply` poisons that trainer. Earlier refusals (open, the pre-compute check, forward, backward, clip, the pre-apply check) never poison. Closing the window means reserving room up front, which changes which loads succeed.
- **Non-finite weights are not refused at load.** An earlier draft scanned every loaded value. It was reverted because the engine's own tests load NaN weights on purpose to drive the Metal and wgpu deferred-fault paths through TRAIN_STEP, SAMPLE and SAVE. Without the scan, the first op that reads a NaN still refuses it (every backend checks its inputs), at a cost of at most one forward pass.
- **Batch rows are not bounded by `cfg.batch`.** `step_tokens` documents that rows may differ. The measured-peak check covers a larger step instead: it is refused before compute when an earlier, smaller step's peak no longer fits.
- **Preflight is a check at one moment, not a reservation.** Sessions draw from one process ceiling and may together oversubscribe it, so another session can take the room between `check_room` and the work. Within one session there is no such race. A prepaid session mode (each session reserving its cap from the ceiling at LOAD) would close it, but that changes which loads succeed. The user adopted it as an opt-in mode (2026-10-08): a caller that chooses it trades refused loads for a session that can never be poisoned by another session's allocation; the default keeps the narrow race described above.

## Verification

All runs were on this machine under the Lappi lead's lock, at `-j 2` and `--test-threads=2`, with load 4–7.

- **Gate A** (about 21:19–21:27 local): clippy `-D warnings`, all targets, on core, cpu, model, capi, metal and wgpu: clean. `ojas-core` lib 94 passed. `ojas-cpu --test budget_scratch` 2 passed (`reported_optimizer_scratch_is_the_measured_peak_and_the_exact_room`: reported equals the measured peak; one byte short is refused with both targets unchanged; exact room runs; across 1/4 threads, Exact/Fast and 7 shapes). `ojas-model` lib 15, checkpoint 20, forward 15 and trainer 14 passed (counts read from the log's `Running` lines; they match each file's `#[test]` count). `ojas-capi` lib 73 passed. Metal `reported_optimizer_scratch_is_the_measured_peak` and wgpu `reported_optimizer_scratch_bounds_the_measured_peak` (offset-0 gradient = reported − b, offset view = reported, tall and wide): passed. Two first runs failed on this lane's own bugs; both were fixed before the passing run. The wgpu closed form had missed the new-parameter plane and the norm partials, and the wgpu AdamW fixture had used a negative second moment.
- **Mutation check of P1:** with the pre-apply `check_room` removed, `optimizer_scratch_taken_after_the_gradients_refuses_before_any_update` fails ("state changed after CapacityExceeded": AdamW had updated `tok_emb`). Restored, it passes. The capacity sweep (`a_capacity_sweep_refuses_up_front_and_never_poisons`, 41 caps) passes both with and without that line: for the tiny model the activations always exceed the optimizer scratch. It is a regression guard (refusals at open, refusals at a step, and full runs all occur; nothing poisons), not proof of P1.
- **Gate B** (about 21:33–21:34): clippy `-D warnings` on device, cpu, capi and model, plus `ojas-device` for Linux x86_64 and Windows (check only): clean. `ojas-device` 72 passed (was 62). `ojas-cpu` `starting_workers_early_is_idempotent_and_changes_no_bits` passed. `ojas-capi` lib 76 passed: the ceiling-vs-machine test, `preflight`, admission at every pressure level, `DEVICE_CPU_AUTO` and `auto_threads`. `ojas-gusset-engine` built into `target-adaptive`. `go vet` clean. `go test -a -p 1`: ok in 19.6 s; the largest process in the `go` tree was 553 MiB (that is likely the linker, so not comparable with the 212 MiB test-binary baseline).
- **Formatting:** every changed file was rustfmt-clean at HEAD (checked through stdin, which follows no `mod` children). This lane's hunks were then formatted file by file through stdin. `gofmt -l go/`: clean. The closing gate (Gate C, below) is the build after that pass.
- **Gate C** (about 21:46–21:47 local, hold "ojasadapt3"):
  - **Build stage:** clippy `-D warnings`, all targets, on the seven crates: clean. `ojas-model` forward 15 and trainer 14 passed, the same counts as Gate A. `ojas-capi` lib 76 passed. `ojas-gusset-engine` built. `go vet` clean. `go test -a -p 1`: ok in 18.8 s, peak RSS 553 MiB, including the `DeviceCPUAuto` train-step block.
  - **Linux stage:** a static musl build in the local Podman VM (`--pull=never`). L1, the `ojas-device` suite with no limits, **failed 65/66**: `many_threads_share_two_buffers` (D7). L2, in a 1 GiB / 2-CPU container, passed 66. L3 profile, in a 1 GiB / 1.5-CPU container: `cgroup_limit_bytes` 1,073,741,824 and `cpu_quota_millis` 1500. The thread ceiling was **1**, where the script expected 2 (D8).
- **Gate D** (about 21:50, the same hold, after C17): `bandwidth.rs` was formatted through stdin (only these hunks moved), and clippy on `ojas-device` was clean. macOS: `ojas-device` 72 passed. The race loop ran 30 rounds alone and 30 rounds of the whole `bandwidth::` module at `--test-threads=2`, with 0 failures in each; before the fix that loop failed 1 in 10. Linux: L1 passed 66/66 three times in a row, and L2 passed 66. L3, at a 1.5-CPU quota, gave ceiling 1 (matching std). L4, at a 2.5-CPU quota, gave ceiling 2. `podman machine stop` ran. After Gate D only a comment on `heap_growth` changed.
- **Gate E** (about 22:10–22:11, hold "ojasadapt4", after C18): `tests.rs` was formatted through stdin (it was clean at HEAD, so only these hunks moved), and `gofmt -l` was clean. clippy `-D warnings`, all targets, on all seven crates was clean on the final tree, which closes the Gate D caveat. `ojas-capi` lib passed 77, including `every_error_kind_has_a_distinct_prefix_that_go_decodes`, the flipped `critical_pressure_refuses_allocating_calls_and_keeps_save_and_free`, and `a_user_path_never_selects_an_error_kind` with the new `E_PRESSURE` path. **Mutation check:** with the `E_PRESSURE` row removed from `go/ffi.go`, the guard failed ("go/ffi.go does not decode \"ojas:E_PRESSURE: \""); the file was then restored and compared. The engine built, and `go vet` was clean. `go test -a -p 1`: ok in 19.4 s, 555 MiB peak; `TestInBandKindsTable`, `TestInBandErrorPrefixes`, `TestAUserPathNeverSelectsAnErrorKind` and `TestInBandKindsMustLeadTheEngineMessage` passed.
- **Gate F** (about 22:36, hold "ojasadapt5", after C19), built against tessl HEAD `cf65d9d` plus 30 uncommitted files (working-tree diff sha256 `be52cad41bacfb0b…`; `has_unified_memory` at `src/runtime.rs` 87 and 713). `memory.rs` was formatted through stdin; it was clean at HEAD. clippy `-D warnings`, all targets, on ojas-metal, both with tessl and as the `--no-default-features` stub: clean. The `memory` filter passed 4: `the_device_flag_decides_whether_metal_shares_the_host_budget`, and the live `memory_probe_tracks_residency_and_plans_one_shared_budget`, which asserts this M-series GPU reports unified memory and that the plan shares the budget. The full ojas-metal lib suite passed 56, and `--test training_step` passed 2. Not separately mutation-checked: before C19 the new unit test could not compile, because the field did not exist. Under the old `Unknown` answer, with a host profile of `Unknown`, its `shares == unified` assertion would fail. That is inferred, not run.

## Gaps

- **DevMap failed mid-session.** At session start it was healthy (generation 3895, fresh). Later, `devmap_status` and `devmap_affected_tests` both returned "database disk image is malformed". It was not repaired from this lane (the store is shared), and the remaining caller checks used `rg -uu`. The four audit subagents had no DevMap tools at all.
- **No fuzzing or fork-heavy stress.** The machine lock forbids them after two kernel panics on 2026-10-02. Stress here means deterministic sweeps and fault injection (budget squeeze, one byte short, a capacity sweep over 41 caps); `FuzzDecodeProfile` was not run this round.
- **Linux coverage is partial.** On a real kernel (Gates C and D), the `ojas-device` suite and the profile ran under a memory limit and fractional CPU quotas. The RSS floor (unreadable `memory.current`) and `memory.high` ran against `/proc` and `/sys` fixtures only, because Podman's `--memory` sets `memory.max` and leaves `memory.current` readable. The Rust suites other than `ojas-device`, and the Go tests, ran on macOS only.
- **The C17 lock covers only the bandwidth tests.** Other tests in the `ojas-device` lib binary still allocate while a heap test runs. The largest fixtures (`host.rs` 724 and 738, `topology.rs` 739) are about 120 KiB and 64 KiB, and at `--test-threads=2` at most one runs beside a heap test. That fits inside the 1 MiB slack, so four clean full-suite runs on Linux and one on macOS stand as evidence. A future fixture over 1 MiB would bring the flake back.
- **L3 and L4 cannot separate the two quota paths.** In the container std already rounds the quota down, so the plan's own cut is a no-op there. The probe-only path, where std cannot see cgroupfs, is covered by `a_known_cpu_quota_caps_the_thread_ceiling` alone.
- **Disk space at Save is not checked.** std has no portable free-space call, and the atomic directory swap still keeps the previous checkpoint whole. Linux PSI is not read (superseded 2026-10-07: it is, below).
- **(Superseded 2026-10-07: bounded and pinned, below.)** **Metal physical bytes** remain as stated in round 1: the budget charges logical bytes, and tessl rounds buffers up.
- Under `Resident` (the model tests' device double) `optimizer_scratch_bytes` is `None`, so the pre-apply check does not cover that double.
- **ojas does not build against committed tessl, and that predates C19.** ojas builds tessl from the sibling checkout (`path = "../../tessl"`). ojas HEAD already calls `Runtime::is_poisoned` (`ojas-metal/src/gpu.rs`, `ojas-qwen35/src/step.rs`) and `Runtime::current_allocated_bytes` (`ojas-metal/src/device.rs`, `ojas-qwen35/src/state.rs`), and neither exists at tessl HEAD `cf65d9d`. Both live only in tessl's uncommitted `src/runtime.rs`, about 340 lines from another session. C19 adds a third such symbol, `DeviceMemoryInfo::has_unified_memory`. Each was checked with `git show HEAD:src/runtime.rs` against the working tree. A reset or stash in tessl breaks the ojas Metal and qwen35 builds. The fix is to commit tessl's `runtime.rs` (the user's call), then pin a tessl revision in ojas.
- **Only the `Unified` branch runs on a real device here.** tessl is Apple-silicon-only, so `Discrete` is covered only by the unit test's constructed readings.

# GPU memory accounting (2026-10-07)

Task `ft-748ed8b545133b8290a2c4714916a5cf`, worktree `wire-gpu-memory-probes-into-resourceplan-ce0a01d4`. A budget that never consulted the device could admit a plan the device cannot hold; this round wires the probes into the plan and makes the uncharged GPU memory bounded, reported or trimmable.

## Findings (re-read before the change)

| ID | Finding | Evidence |
| :--- | :--- | :--- |
| G1 | Every `ResourcePlan::derive` in capi passed `&[] as &[NoProbe]`; GPU sessions ran on the caller's budget (or `DEFAULT_BUDGET_BYTES`) whatever the device held. | `load.rs:282`, `profile.rs:129`; failing-first run below |
| G2 | wgpu had no `MemoryProbe`. wgpu 30.0.1 reports no device memory size (`AdapterInfo`, `Limits`); `Device::generate_allocator_report` is `Some` on the Vulkan and DX12 HALs only (`wgpu-hal-30.0.1` `vulkan/device.rs:2954`, `dx12/device.rs:2643`; default `None`, `lib.rs:1210`). | vendored source |
| G3 | Linux pressure and architecture were always `Unknown`. | `system.rs` before |
| G4 | `MetalBackend`'s worker never set tessl's pool cap, so up to tessl's 2 GiB default of freed buffers sat outside the budget; Metal had no trim. | `device.rs` `Worker::open`; tessl `runtime.rs:44` |
| G5 | Metal's device-OOM error said `live: 0`. | `device.rs` `alloc` |
| G6 | wgpu failed an allocation OOM at once while up to `POOL_CAP_BYTES` of idle buffers stayed pooled; `trim_pool` had no library caller. | `context.rs` `alloc`, `mapped`, `read` |
| G7 | A wgpu context drop that timed out held its queue, device and a thread with nothing counting or bounding them. | `context.rs` `Inner::drop` |

## Changes

| ID | Change | Where |
| :--- | :--- | :--- |
| H1 | `MemoryProbe::pool_cache_bytes` (default `Unknown`). `ResourcePlan` gains `caller_budget_bytes`, `device_pool_cache`, and `device_budget(device)`: the caller's number cut by the device's room (memory less residency less pool cap), starting from `budget_bytes` for a device that shares host memory and from the caller's number for a discrete one. Never widened. | `ojas-device/src/plan.rs` |
| H2 | capi `model::open` probes a GPU session's device after it opens (`DeviceProbe`: Metal's `MetalMemory`, wgpu's `WgpuMemory`), plans with `device_plan`, and when the planned budget is below the caller's, takes a smaller child of the ceiling and rebinds the backend to it (`MetalBackend::with_budget`, `WgpuBackend::with_context`). Planned to 0 is `E_CAPACITY`. `create` refuses parameters past the planned budget before anything uploads. **Plan-down, not refuse:** `DEFAULT_BUDGET_BYTES` is a number the caller never chose. `NoProbe` is gone; the CPU-auto path plans with an empty `DeviceProbe` list. | `ojas-capi/src/model.rs`, `load.rs` |
| H3 | `WgpuMemory`/`WgpuContext::memory()`: architecture from the adapter type (integrated or CPU unified, discrete discrete, virtual or other unknown), residency from the allocator report where the HAL keeps one, pool cap `POOL_CAP_BYTES`, device memory always `Unknown`. The plan's room stays unknown and is visible as unknown. | `ojas-wgpu/src/memory.rs`, `context.rs` |
| H4 | Opcode 16 flags `FLAG_DEVICE_METAL` (2) and `FLAG_DEVICE_WGPU` (4) open that device on a short-lived thread, plan against its probe exactly as a load does, and append ten fields (version stays 1): probe device, device memory, residency, pool cap, room, shares-host, shared budget, the session budget, wgpu parked drops, wgpu timed-out drops. Go: `DeviceProfile(ctx, budget, device)` and the matching `Profile` fields. | `ojas-capi/src/profile.rs`, `engine.rs`, `go/profile.go` |
| H5 | Linux pressure from PSI: the worst of `/proc/pressure/memory` and every cgroup v2 ancestor's `memory.pressure`. Thresholds are this crate's own: Warning at `some avg10` 10%, Critical at `full avg10` 10% or `some avg10` 50%. So `E_PRESSURE` admission can fire on Linux. | `ojas-device/src/system.rs` |
| H6 | Linux architecture from `/sys/class/drm/card<N>`: no card, or only platform (no PCI vendor), virtio or bus-0 Intel cards, is `Unified`; only NVIDIA cards is `Discrete`; a mix, AMD, or off-bus-0 Intel is `Unknown`. With no `/sys/class/drm` at all it is `Unified` only when `/sys/class` itself is readable (sysfs mounted, no GPU driver loaded); with no sysfs nothing was learned and it is `Unknown`. **Semantic call:** a host with no GPU is reported `Unified` (one pool, the host's), so Go's `UnifiedMemory` reads 1 there. | `ojas-device/src/system.rs` |
| H7 | Metal pool cap at open: a quarter of the budget, at most 1 GiB (`pool_cache_cap`), reported as `MetalMemory::pool_cache_cap` and set aside from the room. `MetalBackend::trim_pool` (`Cmd::TrimPool`: recycle, then drop the cache). A failed allocation now retries after the recycle and again after dropping the cache; a final refusal reports `live` as `currentAllocatedSize`, in the same unit as `cap` (the working set). | `ojas-metal/src/{device,backend,link,memory}.rs` |
| H8 | Metal's logical-byte charge: **bounded and documented**, not changed. `Tensor::from_device_reserved` requires the reservation to equal the buffer's logical length. tessl's `Cold` rounding is under 512 KiB per buffer up to 1 MiB and under 16 KiB past it; `cold_rounding_stays_inside_the_documented_bound` pins it against tessl's own `allocated_bytes_for`. | `ojas-metal/src/backend.rs` module docs |
| H9 | wgpu `new_buffer`: on OOM, `trim_pool`, a bounded 5 s poll, one retry, for tensors, uploads and staging; `CacheStats::alloc_retries`. A `#[cfg(test)]` seam injects OOMs. | `ojas-wgpu/src/context.rs` |
| H10 | wgpu drops: a drop that returns before its thread finishes is counted as parked (one atomic state the drop and its thread race on; the drop counts itself in before it tries to park and backs out if the thread already finished, so the count never wraps below zero, a thread that finishes at the deadline never stays counted, and a parked one is uncounted once) and timed out; `WgpuContext::open` refuses (`DeviceError::Capacity`) at `MAX_PARKED_DROPS` (4). `drop_stats()`, and the profile's two fields. | `ojas-wgpu/src/context.rs` |

| H11 | The Go heap-ceiling abort (below, found pre-existing): `Tensor::to_f32_vec`/`to_u32_vec` copy through `try_reserve_exact`, so a heap refusal is `CapacityExceeded` (cap and live 0, as `to_ne_bytes` already reported), and `trainer::fresh` builds the CPU copy straight from the host window, one fallible allocation instead of two. | `ojas-core/src/tensor.rs`, `ojas-model/src/trainer.rs` |

## Decisions (Fable review, 2026-10-08)

- **The wgpu park race** (Fable): the first H10 draft parked by compare-and-swap and then incremented, so a thread finishing between the two decremented first and the count wrapped. Fixed as H10 describes; `parking_and_reaping_race_without_wrapping_the_count` runs 5000 park/reap races with a watcher that fails on any count past 1.
- **The host cut stays for a unified device.** `device_budget` starts a shares-host device from `budget_bytes` (the host plan's cut), not the caller's number: the Metal working set and host RAM are one pool, so a budget the host cannot back is not one the device can.
- **No retune after plan-down.** A planned-down Metal session keeps the commit trigger (`mem_cap`) and pool cap its device thread opened with, derived from the caller's budget. Budget charges are logical and released when a tensor drops, whatever the commit cadence, so the trigger never decides whether a charge succeeds. The device side is bounded three ways: the pool cap is already taken out of the room the plan used, the working-set trigger commits past the device's share, and a refused allocation recycles and drops the cache before failing (H7). Retuning would only add commits.
- **wgpu device memory stays `Unknown`.** The only way to a heap size is through `as_hal`, which is `unsafe`, and `ojas-wgpu` is `#![forbid(unsafe_code)]`. The plan records the room as unknown rather than guessing; an unclassified adapter on a unified host still gets the host cut (the `WgpuMemory` test with a 1 GiB cgroup).
- **The profile's wgpu flag opens and drops a context**, so on a host already at `MAX_PARKED_DROPS` the profile itself is refused, and its drop can take a parked slot if the device hangs. That is the bound every open honours, and the fields report it.
- **The Go decoder** refuses a device block cut part-way (`profileFields < n < allProfileFields`) and flag or probe values out of range, instead of reading zeros as measurements.

## Verification (macOS M5 Pro and the local Podman VM, under `mac_heavy.sh`, `-j 2`)

- **Failing first** (window 1, `git archive HEAD` plus only the new test): `a_metal_budget_past_the_device_working_set_is_planned_down` failed: "the session budget 52613349376 is past the device's working set 51539607552 (caller 52613349376)". ojas-device baseline: 74 passed.
- **After** (window 4): clippy `-D warnings`, all targets, on ojas-device, ojas-metal, ojas-wgpu and ojas-capi: clean. ojas-device 78 passed; ojas-wgpu lib 32 and `tests/drop.rs` 1; ojas-capi lib 83 (the failing-first test now passes, plus the Metal and wgpu profile-flag tests). ojas-metal lib 64 passed in window 3, after the two persistent-failure tests were moved from 2 to 3 injected failures (there are now three attempts; both still fail every one).
- **Bug found by the run:** window 3's `concurrent_calls_from_threads_match_cpu` failed with "4 freed wgpu contexts are still waiting": the first draft counted every running drop thread, so ordinary concurrent frees tripped the cap. H10 counts only parked drops; window 4 passed.
- **Linux (Fedora 7.1.3 aarch64 VM kernel, native, `rust:1-bookworm` image, `--pull=never`):** no musl target is installed any more, so the build ran inside the image from a two-crate copy with `libc` as a path dep, `--offline`. `RUSTFLAGS='-D warnings' cargo check --all-targets`: clean (the image has no clippy). Suite: 72 passed unbounded and 72 in `--memory=1g --cpus=2`. Probe there: architecture `Unified` (`/sys/class/drm` holds only `version`), pressure `Normal`, read from `some avg10=0.00`/`full avg10=0.00` in both the system and cgroup files. An 8 GiB request planned to 1,070,845,952 bytes.
- **Go** (windows 5 and 6, against this tree's `libgusset.a`): `go vet` clean (window 2); `go test -a -p 1 -skip TestHeapCeilingIsACleanError ./...` ok in 19.9 s, peak RSS 538 MiB, including `TestDeviceProfileShowsTheMetalProbeAndItsRoom` (Apple silicon: shares host, shared budget, room at most the host budget) and `TestDeviceProfileShowsTheWgpuProbe` (room unknown). Window 5 caught one stale expectation of mine: the appended-fields decoder case used 23 fields, which now reach the device fields; it now appends past all 30.
- **Pre-existing, not this change:** `TestHeapCeilingIsACleanError` aborts the Go test binary ("memory allocation of 8388608 bytes failed", an infallible `Tensor::to_f32_vec` in `ojas_model::trainer::fresh`). It aborts identically on the unmodified tree (window 4, `ojas-wire-base`), and it kills every Go test after it in the binary.

## Gaps

- Pressure under real stall was not induced on Linux (only `Normal` was read live); the thresholds are covered by fixture tests.
- `Discrete` stays unit-tested only: no discrete GPU here.
- wgpu's OOM retry ran through the injection seam; Metal through wgpu rarely refuses an allocation for real.
- The Metal pool cap (budget/4, at most 1 GiB) can lower pool hits for a model whose per-step temporaries pass it. No benchmark was run; no speed claim is made either way.
- The Qwen35Step pool cap is task ft-c57c98e0, not this one.
