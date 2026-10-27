Decision rules, fixed before the first measurement (committed ahead of the results). No timing from `ojas-metal/examples/alloc_bench.rs` had been seen when this file was written. Card ft-580e6c40.

Question. SpikeTrack's frame profile at Ojas cbd5565 (`SpikeTrack/bench/results/profile-cbd5565/frame-metal.log`, one run, load 16 to 24, every op synced) prices four large-output rows above their quiet single-call medians: conv2d `[1,256,288,288]x[256,256,3,3]x[256]` 51.0 ms against 25.3 (2.0x; 46.0, 51.8, 51.0 at F = 1, 3, 7 and 46.8 per call in the detector run), conv_transpose2d `[1,512,144,144]x[512,256,2,2]x[256]` 13.2 against 5.7 (2.3x), interpolate_bilinear `[1,32,1008,1008]` 9.6 against 1.9 (5.0x, 8.0 and 8.7 at F = 1 and 3), the memory linear `[1,36400,256]x[256,256]x[256]` 3.17 against 1.72 (1.85x). The reconcile (f5ec7e8) put these in one cluster and proposed, as an untested hypothesis, that these ops pay allocation or first-touch cost in a frame that the bench, which reuses pooled buffers, does not. Does allocation churn reproduce the gap? Measurement only: no kernel, pool or policy change.

What the code says, read before measuring (inferred from reading, not yet tested; file:line from this checkout, 11f5091 for ojas-metal and /Users/bharath/Code/research/tessl for the pool):
- ojas-metal/src/device.rs:88 `MEM_CAP_CEILING` = 1 GiB and :91 `POOL_CACHE_CEILING` = 1 GiB; the caps of a backend are `min(1 GiB, budget / 4)` (`pool_cache_cap`, device.rs:98, and the `mem_cap` initializer, :587), so Budget(24 GiB) and Budget(28 GiB) give the same caps. Buffers freed since the last waited commit past the memory cap force a waited commit before the next command (`recycle_freed`, device.rs:739, counted as `freed`); bytes allocated since the last wait past the cap wait after the command (`mem_cap`); the working-set rule (device.rs:106-108 and the check at :772) waits when this process's `currentAllocatedSize` is above three quarters of the recommended working set and more than 64 MiB was allocated since the last wait.
- tessl `src/runtime.rs` `BufferPool` (`bucket` :264, `recycle` :307): the freelist is keyed by the exact page-rounded size; `recycle` parks a freed buffer, and one that does not fit under the cap evicts the LARGEST cached buffer that is larger than itself, and is released if none is larger. So a freed 85 MB output stays cached behind a cached 764 MB scratch, but with the cache full of buffers SMALLER than the output it is released and the next call allocates fresh. Churn with buffers larger than the op's output therefore cannot evict it; arm B is the coordinator's design and arm B2 is the one the pool code says can.
- The Budget SpikeTrack opens its MetalBackend with: 24 GiB in the profile (`crates/st-model/tests/frame_profile.rs:163`, SpikeTrack 5a77151), 28 GiB in the CLI (`crates/spiketrack/src/image_cmd.rs:95` BUDGET_BYTES, used at :225) and in the VOS gate (`crates/st-model/tests/vos_gate.rs:62`). The bench's own backend is 24 GiB (as the profile's), so arm B is the profile's budget and arm C28 is the production one.

Rows (the profile's shapes, f32, device-resident, uniform [-1, 1] activations and [-0.05, 0.05] weights): the four above; conv stride 1 padding 1, conv_transpose stride 2, interpolate to 1152 x 1152 with align_corners false (the call at memory_encoder.rs:90).

Arms, each run twice per round (the repeat is A'), odd rounds reversed; 5 warm-up rounds, 25 timed; before every timed sample the same arm runs once untimed, so the pool holds that arm's own state and not the previous arm's. The timed region is: the arm's untimed `prep`, then counters, clock, the op, sync, clock, counters. Per sample the example records the deltas of `MetalBackend::waits()`, `dispatches()` and `wait_counts()` (by trigger) around the timed region only, and prints the mean per call, the maximum waits and the nonzero triggers. A synced call counts one `sync` wait by itself: the baseline of arm A is 1 wait per call.
- A steady state: the op alone, as `frame_ops_bench` times it.
- B big churn, 24 GiB: `prep` uploads three tensors of 168, 172 and 176 MB and one of 760 MB (distinct sizes from every buffer the op touches, including the 764 MB im2col scratch) and drops them: more than the 1 GiB pool cap and memory cap in one go.
- B2 small churn, 24 GiB: `prep` uploads and drops 1.25 GiB in pieces of half the op's output size, each a distinct 16 KiB bucket, so the pool cache is full of buffers smaller than the output.
- C28 big churn on a second MetalBackend opened with Budget(28 GiB) and its own operands.
- E fresh: `trim_pool` in `prep`, so the output and scratch are new allocations every call with no other churn: the direct test of first-touch cost.
- Dsync and Dnosync: eight consecutive calls with a sync after each versus one sync after all eight, per call, to bound the profile's per-op sync. (The profile's own sync is wall time around call-plus-sync, as A.)
- Control: the f32 fc1 linear `[5184,1024]x[4736,1024]` in every row's run, with the same untimed pass before each sample; it must repeat 6.3 TFLOP/s within 3% (6.11 to 6.49).

Statistics and cleanliness: as frame_ops (min, p10, median, p90; 1-minute load recorded at start and end of every attempt; a row attempt is clean when load is at most 6.0 at both ends, the control is in its band and no GATING pair is NOISY; up to 3 attempts; a row with no clean attempt gets no number). One deviation, stated here because it is a change: only the A/A' pair and the control's pair gate, since the churn arms are irregular by design (their A/A' gap measures their own variability, which is part of the result); every pair's verdict is still printed, marked `noisy (not gating)` where it fails.

Evidence in hand before measuring (tessl-45's in-situ run, `SpikeTrack/bench/results/profile-insitu/frame-metal.log`, budget 24 GiB; relayed by the coordinator; I checked the per-op rows below against the log and did NOT re-check the pass-level claims or the 4422-waits total): the three unprofiled passes started at load 7.35, 8.53 and 9.81 and the profiled pass at 15.31 (log lines 6 to 9); the rows show 1 wait and 7 dispatches per conv call: 3x3 288 squared conv 61.2, 65.8 and 61.2 ms at F = 1, 3, 7 and 58.6 per call in the detector run (log lines 43, 62, 67, 68), downsampler conv [1,32,1152,1152] 27.6, 27.5 and 22.3 ms (lines 98, 99, 117), the [1,36400,256] linear 3.33 ms (line 85), bilinear [1,32,1008,1008] 16.0, 11.7 and 10.7 ms with 2 dispatches (lines 135, 153, 160). Per the coordinator, pass 1 is not slower than passes 2 and 3 (refuting pass-level warm-up) and no op has more waits than calls, so every wait is the profiler's own sync and the MEM_CAP_CEILING forced-wait mechanism is refuted in situ. The conv sits at 46 to 52 ms in the cbd5565 profile (load 16 to 24 before it) and at 59 to 66 ms here (load 15 before the profiled pass), against 25.3 quiet: it varies with the run, which fits either explanation. The 1-minute load average does not measure GPU use by other processes.
The two explanations left, and the primary reading: (1) per-op first-touch from pool misses, tested directly by arms B2 and E (primary); (2) GPU contention from other sessions, tested by the control and by the quiet-round filter below. Arm C is a control and the waits columns of B and C are a secondary check of the already-refuted forced-wait mechanism. Arm D bounds the per-op sync cost.

The control within the same rounds. The fc1 control runs in every round of every row, interleaved with the arms (both passes, reversed on odd rounds), so a contended round shows up in it. Per attempt the example prints, per arm, the median over all rounds and the median over the QUIET rounds, a round being quiet when the control, in both of its passes of that round, is within 3% of the control's median over the attempt (`# quiet_rounds` lines, with the count of quiet rounds). The clean-row rule is unchanged and still gates on the control's band (6.11 to 6.49 TFLOP/s).

Reading, fixed now. The reference ratio is each arm's median over arm A's median at the same row, judged against A's own A/A' gap and spread, on a CLEAN attempt (control in its band).
- (1) is CONFIRMED for a row if, on a clean attempt, arm E (or B2) has median at least 1.8x A (the low end of the profile's 1.8x to 2.7x band for these ops) beyond noise, and the ratio holds on the quiet rounds too. Waits per call are then read as a secondary column: a rise in `freed`, `mem_cap`, `working_set`, `slab_full` or `recycle` waits names the waited-commit route; no rise with a slowdown names the pool-miss and first-touch route.
- (1) is PARTIAL if the best of E and B2 is between 1.2x and 1.8x of A: first-touch contributes and does not explain the profile's ratio. B and C28 are reported beside it, as controls.
- (1) is REFUTED if, on a clean attempt with the control in its band, E and B2 are both under 1.2x A on the 3x3 conv row and the other rows agree (the quiet-round medians agree with the all-round medians). Then, in the order tessl-45 named: contention becomes the leading explanation, with the untested remainder, namely in-frame state this bench does not model (resident weights of hundreds of MB to GB, a different live-set size, kernels cold per shape) and host-side cost between ops.
- If the control is NOT in its band in any attempt of a row (a contended machine) the row has no number and the reading waits for a clean run: a contended run cannot confirm or refute (1).
- D bounds the per-op sync cost: if `Dsync` minus `Dnosync` per call is under 2 ms on the 25 ms conv row, per-op sync cannot account for the 25.7 ms gap on that row.
- Interpolate is judged against its own profile ratio (5.0x in the cbd5565 profile, 8 to 16 ms in the later profiles) as well as the 1.8x floor.

Not tested here, and why: the working-set trigger (it needs this process to hold more than 3/4 of the recommended working set; the example prints `recommended_working_set` and `allocated` for both backends, and a bench cannot hold that much without leaving the 24 GiB budget); other processes' GPU use (the profile's load of 16 to 24 is the other sessions' work and cannot be reproduced in a quiet run).

The load gate: the job waits, outside the mac_heavy lock, for a 1-minute load of at most 5.0 before it takes the lock, and does not run while the queued interpolate bench holds it (the lock serializes them).

Addendum (2026-10-27, Fable ruling fable-alloc.md, committed before any timing from alloc_bench had been seen).

A clean attempt now also requires no memory pressure: the vm_stat Pageouts and Swapins deltas
across the attempt (or across the bench step, where only the job script records them) are both
zero. An attempt that fails this is "pressure", not clean, whatever its load and control. Reason:
Metal shared buffers are pageable; eviction of pooled buffers shrinks E minus A and reclaim before
first touch inflates it, so pressure can fake either verdict.

The reading of section "Reading, fixed now" is unchanged and applies to clean attempts only. No
ratio of arms measured under contention stands in for it, now or at the deadline: a constant
contention term compresses E/A and B2/A while inflated allocation syscalls expand them, so the
sign of the error is unknown. The wave-5 attention stand-in does not transfer because there the
ratio was observed flat across loads before use and contention was not a hypothesis under test.

A contended attempt (load above 6.0 at either end, or control out of band, or pressure) is written
to alloc-output-contended.txt and yields only the readings below. Each is labelled "contended" in
the report and none is a verdict on (1).

(i) Counters. waits_mean, waits_max and the triggers by name are deterministic in the pool state
and are read at any load. A rise in freed, mem_cap, working_set, slab_full or recycle waits in B,
B2, C28 or E over A names the waited-commit route and is a finding; no rise confirms the in-situ
refutation of that route. This reading says nothing about first-touch cost.

(ii) One-sided "small" readings, from differences in ms, never ratios, taken within the same
attempt. Dsync minus Dnosync is read from the printed medians. E minus A and B2 minus A are
paired per round and pass from # sample lines; if the pass was run without them, those two
readings are UNAVAILABLE and are not reconstructed from medians.
  - Dsync minus Dnosync per call: contention can only inflate sync wake-up latency, so a value
    under 2 ms on the conv row refutes per-op sync as the gap's cause on a quiet machine too.
  - E minus A and B2 minus A: valid one-sided ONLY on a pressure-free contended attempt (Pageouts
    and Swapins deltas zero), because contention without pressure can only inflate allocation and
    first-touch cost. Then, if the larger of the two is under half the row's profile gap, the
    gap taken at the LOW end of the profile band so the reading is the stricter one (conv
    46.0 - 25.3 = 20.7 ms, half 10.4; convT 13.2 - 5.7 = 7.5, half 3.7; bilinear 8.0 - 1.9 =
    6.1, half 3.0; linear 3.17 - 1.72 = 1.45, half 0.7), and its p90 over rounds is under the
    full gap, (1) is PROVISIONALLY REFUTED for that row: first-touch under
    contention costs less than half the gap, and a quiet machine cannot make it cost more. A
    clean attempt converts this to REFUTED; it is not a verdict on its own. A large difference
    confirms nothing (contention may have inflated it) and reads "consistent with (1),
    unconfirmed". Under pressure this reading is void.

(iii) Arm A in the profile's load regime: 1-minute load between 6 and 30 at both ends of the
attempt (the cbd5565 profile ran at 16 to 24, the in-situ pass after 15). This tests whether the
environment alone, with no churn and with pool hits, reproduces the frame's price; it tests
"environment at this load", not "GPU contention" specifically, since pressure and CPU load are
not separated. Per row, A's all-round median:
  - conv: at or above 46.0 ms, environment SUFFICIENT; at or below 30.4 ms (1.2 x 25.3),
    INSUFFICIENT; between, inconclusive.
  - convT: 13.2 / 6.8.   bilinear: 8.0 / 2.3.   linear: 3.17 / 2.06.
SUFFICIENT means the frame's price for the row is reproduced without allocation churn, so (1) is
not needed to explain it; it does not exclude (1), which only (ii) on a pressure-free attempt or a
clean run can do. The SUFFICIENT thresholds are the cbd5565 profile's LOW end (F = 1); matching
the in-situ pass (conv 59 to 66 ms, bilinear 10.7 to 16.0) needs A at or above 59 ms and 10.7 ms
on those rows, and a reading between the two is "reproduces the low end only", not "explains
everything". INSUFFICIENT means the environment at the profile's load does not reproduce the
gap, which leaves (1) and the untested in-frame state as the candidates and raises the priority
of the clean run.

The quiet-round filter on a contended attempt selects rounds at the attempt's modal contention
level, not quiet rounds. It stabilises a difference; it certifies nothing. The precedent already
recorded fc1 in band while 30-200 ms arms ran 16-34% slow.

Deadline: if no clean attempt exists for a row by 2026-10-28 12:00 local, the row is UNREAD. An
unread card proposes no pool or policy change. The report then carries the contended readings
with their labels, the minimum 1-minute load observed while waiting, and the swap state.
