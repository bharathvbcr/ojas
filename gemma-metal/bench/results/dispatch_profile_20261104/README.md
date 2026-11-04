# Per-dispatch GPU timestamp profile of one decode token (2026-11-04)

Step 1 of the decode speed-up plan: where does the ~42.4 ms of an E4B token go, given that Q4 GEMV
alone should take ~13 ms (`../gemv_fraction_of_peak_20261104/README.md`)?

**Status:** the instrumentation is built and verified. A real-weight E4B token has NOT been
profiled (the weights are not on disk; see "Pending: real E4B token"). An E4B-*shaped* token has: the
same tensor set, shapes, kernels and dispatch graph, with random MLX Q4 weights
(`--synthetic-e4b`, below). It found two defects, both fixed, and decode went from ~100 to 12.0 ms per
token at those shapes.

## E4B shapes, random weights (`--synthetic-e4b`) [V, `synthetic_e4b/`, `scalar_pool_ab*.log`]

`weights::synthetic_mlx_q4_banks` builds what `load_mlx_affine_q4_shards` builds from the MLX
snapshot: E4B preset config (42 layers, hidden 2560, vocab 262144, PLE 256, KV sharing from layer
24), MLX affine Q4 projections, embeddings and PLE table at group 64, bf16 norms, no k/v on consumer
layers. Kernel time does not depend on weight values, so the timing is the real graph's; the tokens
are meaningless. Load average 3.3-4.9 throughout.

**1. Every dispatch committed and waited (599 commits per token).** A sample of the process put
the host in `IcbScalarPool::push_u32` -> `GpuBuffer::contents_u32` -> `GpuRuntime::host_access` ->
`commit_m4(wait)`: tessl commits the open command buffer and waits before any host mapping, and the
pool wrote a host-mapped buffer for every scalar of every dispatch. Host encode was 96.8 ms per token
for 17.5 ms of GPU kernels. Fix (`kernels.rs`, `IcbScalarPool`): outside DecodeIcb / encode-once
the pool keeps a host shadow and binds each scalar through the binder's constant arena; `pos_buf`
and the host seed token are written by a one-thread `copy_u32` dispatch instead of a host write; the
model-constant softcap is rewritten only when it changes. The stable GPU pool is still used where a
replay re-reads it, and `GEMMA_METAL_STABLE_SCALARS=1` forces it (the A/B reference arm).
Result: 1 commit per token, host encode 0.47 ms.

**2. Decode attention ran on one lane per head.** With commits gone, attention was 4.5 ms of an
18.4 ms token at position ~12 and grew with context (pipelined: 18.0 / 21.9 / 29.8 / 45.5 ms per
token at 16 / 32 / 64 / 128 steps). The tiled FA-2 kernels at Tq = 1 run 8 threadgroups (one per
head) with one live lane walking every key. Fix: one-query calls go to tessl's existing split-KV
decode (`flash_attn_decode_*`, tuned per head dim), through a new
`tessl::nn::flash_attn_decode_values` that binds `Tkv` and the positions as constants (the buffer
form would need a host write). Same masking rules; tessl's `tests/attention.rs` checks the values
path bit for bit against the buffer path, and gemma-metal's GPU-vs-host parity tests run on it.
Attention is now 0.67 ms per token, and the per-token time is flat in context: 11.9-12.1 / 12.0-12.2
/ 12.3-12.5 ms at 16 / 64 / 128 steps.

**Paired A/B** (`scalar_pool_ab.sh`, pipelined decode, 64 steps, 3 rounds interleaved, 3 runs each,
fresh process per arm; reference = `GEMMA_METAL_STABLE_SCALARS=1`, which also keeps the tiled
attention):

| arm | tok/s (all 9 runs) | ms/token |
|---|---|---|
| reference (old path) | 8.63-9.01 | 111-116 |
| fix 1 only (`scalar_pool_ab_tiled_attention.log`) | 33.41-33.65 | 29.7-29.9 |
| fix 1 + 2 (`scalar_pool_ab.log`) | 80.73-83.39 | 12.0-12.4 |

The 64-step fix-1 row is slower than its 16-step 18 ms because of the attention growth above.

**One token now** (`synthetic_e4b/`, stamps on, median of 8): 643 dispatches, 1 command buffer;
GPU span 14.7 ms, of which GEMV 10.9 ms and everything else 3.8 ms; stamps-off wall 12.0 ms. GEMV
is 74% of the token and streams near the bandwidth ceiling (`../gemv_fraction_of_peak_20261104/`),
so the remaining non-GEMV share is about 1-3.8 ms (the stamps serialise, so 3.8 is an upper bound).

Tests: `tests/scalar_pool_commits.rs`: arena vs stable pool give identical tokens; the arena
token commits at most twice (it was 1 on this graph); the stable arm, which is the pre-fix path,
must still commit more than once per two dispatches. It was not run against pre-fix code: it ran
against the half-fixed state (scalars only, 3 commits) and failed there, before the `pos_buf` and
softcap writes moved off the host. Also `tests/dispatch_stamps_mini.rs` and the 198 lib tests (the
GPU-vs-host parity tests now take the split-KV decode), all also run with
`METAL_RUNTIME_HAZARD_BARRIERS=0` for the two integration tests; tessl `--lib` (312),
`tests/attention.rs` (29), `host_path_allocs`, `nn_decode_hazards`, `qwen35_kernels`,
`docs_name_real_tools`.

**Not verified:**
- Real E4B weights: tokens, and whether the real `config.json` matches `e4b_preset` (layer
  pattern, KV sharing, window).
- The cause of July's 42.4 ms real-weight figure against today's ~100 ms synthetic baseline. That
  tessl's host fence (mappings commit and wait) came between them is inferred, not shown.
- The ordering of the new GPU stores of `pos_buf` and the seed token against the previous token's
  reads (a write-after-read edge the host write used to cover by waiting). It rests on tessl's
  queue barrier at each encoder open and on identical tokens on the 3-layer mini graph.
- DecodeIcb / encode-once speed: they keep the old path by design.

**Opt-in fusions re-measured** (`fusion_ab.sh` -> `fusion_ab.log`, 64 steps, 3 rounds, arm order
rotated, 3 runs per arm): base 80.43-82.08 tok/s; `GEMMA_METAL_FUSE_LAYER=1` 82.16-83.29, ahead
in every round (+1.2% median); `GEMMA_METAL_FUSE_GATE_DOWN=1` 8.27-8.36 with or without
`FUSE_LAYER`, a 10x regression. Traced with stamps: the persistent gate->down kernel alone is
2.4 ms per layer against ~0.16 ms for the two kernels it replaces, because its grid barrier caps
it at 8 threadgroups (Metal has no co-residency guarantee for a larger grid), and each call
synchronizes and reads a failure flag back. Greedy streams of 24 tokens
(`dispatch_profile --generate 24`) are identical with `FUSE_LAYER` 0/1 crossed with
`STABLE_SCALARS` 0/1; the stable arm also runs the old tiled attention, so this also checks the
split-KV decode across all 42 layers. **`FUSE_LAYER` is now default ON** (DECISIONS D21);
`FUSE_GATE_DOWN` stays off.

**What further fusion can buy.** GEMV is 10.9 of the 12.0 ms and streams at 83-89% of the read
ceiling alone, so folding norms, RoPE and residuals or a persistent megakernel can recover at most
about the 1.1 ms of the 12.0 ms unstamped token that is not GEMV (about 10% more tok/s; the
stamped 3.8 ms is inflated by stamp serialisation), plus whatever GEMV gains from fewer launches
between projections.

## Method

### What was built

| Piece | Where | What it does |
|---|---|---|
| `TESSL_DISPATCH_STAMPS` flag | `tessl/src/ab_flags.rs` (`dispatch_stamps()`, `set_dispatch_stamps()`) | Off by default. Same `TriFlag` as the other ab_flags; pinned by the canonical-name test. |
| Stamp heap | `tessl/src/runtime.rs` (`Metal4EncodePackage::stamp_heap`) | A separate 1,024-slot `MTL4CounterHeap`, created only when the runtime is built with the flag on. The existing 64-slot `counter_heap` and `new_inference()` semantics are untouched. |
| Per-dispatch stamp | `tessl/src/dispatch.rs` (`Binder::dispatch`, `execute_icb_ex`) and `tessl/src/dispatch_stamps.rs` | After every dispatch, `writeTimestampWithGranularity:Precise intoHeap:atIndex:`. Each slot is tagged with the pipeline, whether a barrier was emitted since the previous dispatch, and a layer index. Slot 0 is an origin stamp written when the encoder opens. |
| N-entry resolve | `tessl/src/runtime.rs` (`resolve_timestamps`, `decode_timestamps`) | `resolve_two_timestamps` is now a call to the N-entry path; the unaligned-read contract lives in one decoder. |
| API | `GpuRuntime::take_dispatch_stamps()` -> `DispatchStampBatch { freq_hz, t0_ns, stamps: [ {name, layer, barrier_before, t_end_ns, dt_ns} ], overflowed, dropped_batches, command_buffers }` | Names are resolved from pipeline addresses at resolve time. Overflow, dropped batches and unwritten slots are reported, never hidden. |
| Layer tag | `gemma-metal/src/gpu_model.rs` (`step_inner`) | `tessl::dispatch_stamps::set_stamp_layer(li)` per layer, `-1` before the layers, `n_layers` for the tail. One relaxed load when stamps are off. |
| Session API | `GpuDecodeSession::profile_prime`, `profile_decode_token(stamps)` -> `TokenProfile { host_encode_ms, sync_ms, dispatches, barriers, commits, stamps }` | One token per command buffer: encode, then `synchronize`. |
| Profile binary | `gemma-metal/src/bin/dispatch_profile.rs` | Warm-up, ABBA stamps-off/on overhead arm, K profiled tokens, TSV + summary, pipelined `bench_decode_tok_s` before and after. |
| Driver | `run.sh` | Builds into its own target dir (`-j 2`), runs under `mac_heavy.sh`, logs load average before and after each run. |

Timestamp unit: `device.queryTimestampFrequency()` reads **24,000,000 Hz** on this Mac (one tick is
41.7 ns). Every `*_ns` field is converted from ticks with that frequency; the resolution of a
single `dt_ns` is therefore +-42 ns.

### What a stamp does and does not measure

Apple's header for `writeTimestamp`: it "ensures that any prior work finishes, but doesn't delay any
subsequent work". Consequences, all of which shape how to read the numbers:

1. **`dt_ns` of a dispatch = launch + run + drain + the barrier before it + the stamp itself.** A
   stamp cannot report when a kernel started, so **kernel time and inter-dispatch gap are not
   separable with this API.** A before/after pair around one dispatch gives the same value twice.
   The README for the real run therefore reports GPU time per dispatch *including* launch and
   drain, never "kernel time" and "gap" as two columns. The practical floor for a dispatch is the
   median `dt` of a near-empty kernel (mini smoke: 1.8 us).
2. **Stamps serialise.** A stamp after every dispatch waits for prior work, so dispatches that
   hazard mode would leave unordered can no longer overlap. Profiled GPU time is an upper bound on
   unprofiled GPU time. The stamps-off/on arm (interleaved ABBA, same session) measures how much.
3. **The command-buffer-level begin stamp is not usable as an origin.** Measured here, it landed
   *after* the first dispatch's stamp (125 ticks, ~5 us). The origin is an encoder-level stamp.
4. **Back-to-back stamps with no dispatch between them resolve to 0** (unwritten). An attempt at a
   "stamp-only floor calibration" was removed for that reason; the code reports an unwritten slot
   as an error instead of a zero-length interval.
5. A token must be one command buffer for the span to mean "GPU time of the token". `commits/token`
   and `command_buffers/token` are printed; the binary warns when it is more than one.

### Verified properties (tests)

| Claim | Test | Result |
|---|---|---|
| Flag name is canonical (`TESSL_*` first) | `tessl ab_flags::tests::every_flag_reads_its_canonical_tessl_name_first` | pass |
| N-entry decode: order, unaligned buffer, short buffer refused, size overflow refused | `tessl runtime::timestamp_decode_tests::*` (3 tests) | pass |
| Flag off: no heap, nothing armed, nothing recorded; on: heap exists, **stamp count == dispatch count**, names correct, layer tags as set, **timestamps monotone**, `dt` consistent with `t_end`, heap overflow reported (not truncated silently), flag flipped off after construction records nothing | `tessl dispatch_stamps::tests::stamps_off_record_nothing_and_on_match_the_dispatch_count` | pass |
| Existing behaviour unchanged with the flag off | full tessl lib suite, `cargo test --lib -- --test-threads=1`: **312 passed, 0 failed, 2 ignored** | pass |
| Same on gemma-metal's decode path: generated tokens identical flag off vs on; stamp count == `infer_trace` dispatch count; dispatch and commit counts unchanged by stamps; every layer 0..2 and the tail tagged; layer tags non-decreasing; monotone ns; GPU span <= host wall; no stamp unnamed | `gemma-metal tests/dispatch_stamps_mini.rs` | pass |
| Existing gemma-metal decode tests | `cargo test --release --lib gpu_model`: **29 passed, 0 failed** | pass |

Not verified: that stamps-on and stamps-off have the same *timing* on E4B (that is the overhead arm
of the real run), and that an E4B token fits one command buffer (expected from the 2026-07-20
profile: `host_encode` 0.17 ms with trace on, `async_encode` on; but the mini graph does not, below).

## Mini smoke (synthetic graph, 3 layers; NOT representative of E4B)

`bash run.sh --mini` (PROBES=1), outputs in `mini_smoke/`. Load average 3.7-4.1 (1-minute) at
start and end of the profile run. Median of 8 tokens:

| quantity | value |
|---|---|
| dispatches / barriers / commits per token | 76 / 78 / 76 |
| GPU span from stamps | 10.52 ms |
| host encode | 10.52 ms |
| host `synchronize` wait | 0.12 ms |
| gemv family / everything else (sum of `dt`) | 0.11 ms / 0.65 ms |
| tiny kernels (non gemv, non attention) median `dt` | 1.83 us (424 dispatches) |
| stamp overhead, wall ms/token (16 tokens per arm) | off 10.49, on 10.42 (within noise; the run is host-bound) |
| pipelined `bench_decode_tok_s` before / after | 93.2 / 93.2 tok/s |

[V] On the mini graph **each dispatch is its own command buffer** (76 commits for 76 dispatches, with
stamps off as well; the commit count is the same on and off), so the "GPU span" here is mostly host encode and
commit gaps, and the barrier flag is 0% (the first dispatch of every command buffer has none). This
is mini-specific host-side scalar writes, not something stamps cause. It shows the pipeline works;
it says nothing about E4B's split. The barrier/commit probes (`mini_smoke/probe/`, hazard 0,
coarse 0, mid-commit 100) all land at 91-94 tok/s on mini, i.e. no signal, as expected for a
host-bound graph.

## Pending: real E4B token (weights not on disk)

The snapshot `models--mlx-community--gemma-4-e4b-it-4bit` is **not on this Mac**. Only an empty
`~/.cache/huggingface/hub/.locks/models--mlx-community--gemma-4-e4b-it-4bit/` (dated Jul 13) exists;
no `snapshots/` directory, and no other Gemma E4B safetensors were found under `/Users/bharath` or
`/Volumes`. Nothing was downloaded.

Once the snapshot is in `~/.cache/huggingface/hub` (or `HF_HOME` / `HUGGINGFACE_HUB_CACHE` points at it):

```bash
bash /Users/bharath/Code/research/ojas/gemma-metal/bench/results/dispatch_profile_20261104/run.sh --e4b
# or from an explicit directory:
bash /Users/bharath/Code/research/ojas/gemma-metal/bench/results/dispatch_profile_20261104/run.sh --model /path/to/snapshot
# add the wall-clock barrier / mid-commit probes (no stamps, tok/s only):
PROBES=1 bash /Users/bharath/Code/research/ojas/gemma-metal/bench/results/dispatch_profile_20261104/run.sh --e4b
```

It writes `e4b/{run.log,dispatch_stamps.tsv,kernel_summary.tsv,family_summary.tsv,overhead.tsv}`.
Equivalent direct invocation (what `run.sh` runs under `mac_heavy.sh`):

```bash
CARGO_TARGET_DIR=/Users/bharath/Code/research/ojas/gemma-metal/target-stamps \
GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 \
/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh dispatch-profile \
  cargo run --manifest-path /Users/bharath/Code/research/ojas/gemma-metal/Cargo.toml --release -j 2 \
  --bin dispatch_profile -- --e4b --tokens 8 --rounds 8 --out <dir>
```

Reading the output when it exists:
- `GPU span` vs `host encode` vs `host sync wait`: with mid-commit off (`TESSL_MID_COMMIT` unset ->
  0), the command buffer commits at `synchronize`, so in the 16-step pipelined bench the GPU does not
  start until all 16 tokens are encoded. Host encode time therefore adds to GPU time in the 42.4 ms.
  The 2026-07-20 profile put host encode at 0.17 ms with tracing on; the real run confirms or kills
  that. If `host encode` is small, the remaining gap is in the GPU span.
- `gemv` family vs `everything else` in `family_summary.tsv`, exact names in `kernel_summary.tsv`.
  The family grouping is by kernel name (`family()` in the binary); the exact-name table is authoritative.
- `overhead.tsv`: stamps-off vs stamps-on tokens, interleaved ABBA. If stamps-on wall is more than
  ~1.5x stamps-off, the per-dispatch numbers are distorted by serialisation; fall back to stamping
  only after GEMVs and layer boundaries.
- `dispatches preceded by a barrier`: in hazard mode (default on, `kernels.rs:834-840`) a barrier is
  emitted lazily before every dispatch, so expect ~100%; this counts barriers, it does not price them.
- No "no barrier" flag exists in tessl (`forgo_pending_barrier` is crate-private), so the barrier-cost
  probe was not built as a removal. `PROBES=1` varies only what flags exist (hazard always-on vs lazy,
  coarse barriers, mid-commit) and reports wall tok/s. [I] Do not read those as the cost of barriers.
- Gaps: not separable (see "What a stamp does and does not measure"). The fraction of the ~29 ms that is
  non-GEMV kernel time vs launch/drain/barrier cannot be split by this tool. The median `dt` of
  near-empty kernels (norms, rope, KV append, copies) is the per-dispatch fixed cost; multiply by the
  non-GEMV dispatch count for an estimate of the fixed part, and label it [I]nferred.
