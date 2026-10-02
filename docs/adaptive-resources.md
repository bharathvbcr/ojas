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

**Incident, this lane's fault.** The first Go run was killed by the OS after 491 s, with load at about 34. A test case built its "count overflows" record by generating 4,294,967,295 entries (about 38 GB) and then slicing it. The case now overwrites the count field of a 16-entry record. The rerun finished in 17.8 s. A machine audit later found a JetsamEvent at 10:55 for a `go.test` process at 84–120 GiB resident; by timing this was very likely that run (inferred, not matched by PID). The current profile tests are bounded by construction: the decoder allocates a fixed 20 entries whatever count a record claims, and the one bandwidth measurement uses at most two 64 MiB buffers.

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
| R7 | `MetalMemory::architecture()` is always `Unknown`; on an Intel Mac with an integrated GPU the plan would treat it as separate memory. | Not fixable here: tessl does not expose `MTLDevice.hasUnifiedMemory`. Recorded as a tessl request. |
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
- **The Go `Profile` has no GPU figures.** Opcode 16 opens no device, so it carries the unified-memory flag but no Metal working set, residency or room. Those are reachable only in Rust, through `MetalBackend::memory()` and `ResourcePlan::derive`.
- `docs/status.md` lists `ojas-device` at 18 tests only inside dated run records (2026-10-01), which are correct for those runs; the current count belongs in the next integration run's table. This lane changed only the component diagram line (line 243); the file's other uncommitted hunk (line 124) belongs to the "Rust/Go ML packages" session.

## Size

Tracked diff of this lane: about +760 / −80 lines across `ojas-device` (host, plan, lib, manifest, README), `ojas-metal/src` (backend, device, link, lib; the `gpu.rs` hunks there are another session's), `ojas-capi` and `go/ffi.go`. 22 of `go/ffi.go`'s 42 changed lines are gofmt realigning one constant block. New files: about 1,340 non-test Rust lines (`bandwidth`, `topology`, `tuning`, `system`, `sysctl`, capi `profile`, metal `memory`) and about 890 test lines, 141 Go plus 162 Go test lines, and a 57-line example. Most of the growth is the Linux `/sys` readers and their fixture tests. Nothing was deleted: the unwired probe was wired, not replaced.
