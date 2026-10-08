# Layer fusion v1 — native Metal dispatch-count reduction (2026-07-18)

Implements the rank-1 lever from [`audit_deep_2026-07-18.md`](audit_deep_2026-07-18.md) §F2
and [`research_2026-07-18_inference_frontier.md`](research_2026-07-18_inference_frontier.md) §4.

**Status: built + measured on M5 (Lane B 2026-07-19; E4B re-measure 2026-07-20).**
Exactness under `METAL_RUNTIME_HAZARD_BARRIERS=0`: **31B QKV PASS**; **E4B full
arms (baseline/qkv/ple/both) PASS vs bank-split `gemv_kv` (`FUSE_KV=1`)**
(`fusion_ab_e4b_20260720T012511Z`, TRACE=1 — overturns prior FAIL at
`fusion_ab_e4b_20260720T001652Z`). Root cause was dual-accumulate ULP in the
old `gemv_kv` (solo≡fused already; solo≠old `gemv_kv` at `k[41]`). Fix:
bank-partition K∥V like QKV (solo reduce per TG). Hot dump after fix:
**solo ≡ gemv_kv ≡ fused** bit-exact (`qkv_ab_dump_e4b_banksplit_kv.txt`).
Dispatch drops still engage (E4B 599.9 → both 509.9 Δ=90; **31B QKV 727.9 →
667.9 Δ=−60**). Tok/s ~flat — keep opt-in; **do not default-on `FUSE_QKV`/
`FUSE_LAYER`**. Do not copy into `gates.md` as default-on.

**E4B full arms vs bank-split `gemv_kv` (2026-07-20, TRACE=1):**
[`bench/results/fusion_ab_e4b_20260720T012511Z.json`](../bench/results/fusion_ab_e4b_20260720T012511Z.json)
([`fusion_ab_e4b_qkv_ple_latest.json`](../bench/results/fusion_ab_e4b_qkv_ple_latest.json))

**E4B QKV-only light gate (TRACE off):**
[`bench/results/fusion_ab_e4b_20260720T011736Z.json`](../bench/results/fusion_ab_e4b_20260720T011736Z.json)
([`fusion_ab_e4b_qkv_fuse_kv1_latest.json`](../bench/results/fusion_ab_e4b_qkv_fuse_kv1_latest.json))

**Prior FAIL artifact (dual-accumulate era):**
[`bench/results/fusion_ab_e4b_20260720T001652Z.json`](../bench/results/fusion_ab_e4b_20260720T001652Z.json)

**E4B QKV vs solo K/V artifact (2026-07-20):**
[`bench/results/fusion_ab_e4b_20260720T010239Z.json`](../bench/results/fusion_ab_e4b_20260720T010239Z.json)
([`fusion_ab_e4b_qkv_solo_kv_latest.json`](../bench/results/fusion_ab_e4b_qkv_solo_kv_latest.json))

**rope+kv_store:** written + host-wired + float unit parity
(`rms_qkv_rope_kv_store_matches_unfused`); E4B A/B exactness+dispatch PASS (Δ=24 =
producers). **Not re-measured on 31B** (jetsam risk; one-at-a-time policy — expect
Δ≈−60 on producers when `GEMMA_METAL_FUSE_ROPE_KV=1`).

**encode-once / CB replay (v0.5.9):** mini DecodeIcb **wired** (tape_execute
default; cmds≈71; parity PASS). **A2 prebuilt per-cmd arg-tables:** execute
`setArgumentTable` × cmds; mini last_setAddress **0/427**. **E4B Hot**
(cmds=594; last_setAddress **0/5003**). **Product tok/s regress fixed
(v0.5.6):** capture `barrier_after` + skip-auto replay → ~flat vs shipping.
**Freeze + range-batch + coarse-ranges (v0.5.7–0.5.9):** opt-in; E4B
execute_icb **594→361**; product **coarse_elided=0**; quiet still not a win.
**PARK true-ICB** — prefer prebuilt tape. Flags stay OFF. **31B** still
`IcbDecodeGraphNotMigrated`. Artifacts:
[`encode_once_mini_ab_latest.json`](../bench/results/encode_once_mini_ab_latest.json),
[`encode_once_e4b_hot_ab_latest.json`](../bench/results/encode_once_e4b_hot_ab_latest.json),
[`encode_once_e4b_toks_latest.json`](../bench/results/encode_once_e4b_toks_latest.json).
ICB smoke: `GEMMA_METAL_ICB_SMOKE=1` (default OFF).

**Interleaved4 QKV twin:** `gemv_q4_mlx_simd_qkv_i4` landed; `can_fuse_qkv` accepts matched
I4 banks when `GEMMA_METAL_FUSE_QKV`/`FUSE_LAYER` is on (layout selected by Hot pack).
Float unit (isolated) **PASS** for RowMajor / E4B-dim / I4 / tied-kv. Hot `fusion_ab`
pins `GEMV_INTERLEAVE=0` (RowMajor). **No 31B A/B** this cut.

**Persistent interpreter v0 (mini):** `persistent_interp_gate_down` + sibling
`persistent_interp_fa_o_proj` — instruction stream + atomic grid barrier for both
doctrine edges (gate→down, FA→o_proj mock). Opt-in `GEMMA_METAL_PERSISTENT_INTERP=1`
(default OFF). **Mini decode hooked:** `step_inner` dispatches both stand-ins once
per layer on dense scratch when flag on + synthetic mini (shipping Q4 path
unchanged; Hot/E4B/31B no-op). Metal forward-progress caveat remains (D17).

**Hot bounded-TG gate→down (D18):** opt-in `GEMMA_METAL_FUSE_GATE_DOWN=1`
(default OFF; separate from `PERSISTENT_INTERP`). Engagement measured (Δ=−42)
but **hard-blocked on exactness**: Q4 peel is bit-exact with encoder sync;
single-dispatch relaxed grid barrier does **not** publish `mid` to in-kernel
DOWN at E4B dims (host `mid_err=0`, fused resid ~1e9). Test lock
`persistent_interp_gate_down_q4_e4b_dims_visibility_blocker`. Two-dispatch
split has no dispatch win vs shipping. **Keep default OFF.**
Artifact: [`fusion_ab_e4b_gate_down_latest.json`](../bench/results/fusion_ab_e4b_gate_down_latest.json).

**31B measured artifact:**
[`bench/results/fusion_ab_31b_20260719T080223Z.json`](../bench/results/fusion_ab_31b_20260719T080223Z.json)
(quiet run; later Hot-upload peaks jetsam-OOM'd — do not re-run until harness mitigations
below). Policy: **one Metal job at a time** (no concurrent `bench` / `diag_tok` /
`golden_parity` / `fusion_ab`; never co-run e4b with 31b).

---

## Why dispatch count

Measured: **~37 µs fixed cost per dispatch** (mini: 746 tok/s ÷ ~36 dispatches; barrier mode
moved it ±1%, so the cost is the launch, not the barrier). At ~460 dispatches/token that is
**~17 ms of a ~42 ms E4B token**. Kernels themselves already run at 62–100% of the ~273 GB/s
roofline, so more GEMV peel is worth ≤1.3×; dispatch count is worth ~2×.

For reference, CUDA pays 1.3–2.1 µs/launch ([Hazy Research megakernel][hazy]) — Metal 4
dispatch + argument-table set is roughly **20× more expensive per launch**, which is why this
lever is bigger here than the equivalent `torch.compile` step was in the Fast Gemma Challenge.

## What v1 fuses (and what it deliberately does not)

A fusion is safe only when the consumer's dependency is **local** — same threadgroup, or
element-wise. Anything needing *all* of a producer's output (e.g. `gate/up → down`, or
`FA → o_proj`) requires a grid-wide sync, which Metal does not guarantee. Those are deferred to
a persistent-interpreter prototype (research doc §4.3), not attempted here.

| Fusion | Replaces | Saves | Why it's safe |
|---|---|---|---|
| **`gemv_q4_mlx_simd_qkv`** (+ `_i4`) | `gemv_q` + `gemv_kv` | **1 dispatch / producer layer** | Q, K, V all read the *same* `x_bf16`; only the weight bank and output differ. Threadgroups are assigned whole banks (`tg_q` / `tg_k` boundaries computed host-side), so no TG straddles a bank and no simdgroup splits before `simd_sum`. I4 Hot selects `_qkv_i4` (same partitioning, interleaved pointer walk). |
| **`gemv_q4_mlx_simd_kv`** (+ `_i4`) | solo `k` + `v` gemv | **1 dispatch / producer** (product default-on) | Bank-split like QKV (`[0,tg_k)→K`, `[tg_k,…)→V`); math ≡ solo `gemv`. Replaced dual-accumulate (ULP drift vs solo). |
| **`ple_lookup_q4_mlx_residual`** | `ple_lookup_q4` + `ple_residual_add` | **1 dispatch / PLE layer** (E4B: every layer) | `out[gid]` feeds only `dst[gid]` — strictly element-local, so no cross-thread visibility is needed. |
| **`rms_qkv_rope_kv_store`** | `rms_qkv_rope` + `kv_store_timestep_pair` | **1 dispatch / producer layer** | After K/V norm(+RoPE) into scratch, also write the timestep into the cache slot — element-local copy; shared-KV append stays separate when needed. |

Both kernels are **appended** to existing `.metal` files; no existing kernel was modified, so
with the flags off the compiled default path is byte-identical to before.

### Predicted vs measured

| Model | Dispatches/token now | After v1 | Launch tax | Predicted decode | Measured (HAZARD=0 exact; tok/s shipping) |
|---|---|---|---|---|---|
| E4B (42 layers, all PLE, ~24 producers) | ~600 | ~510 (both) | — | ~24 → **~26–27 tok/s** | **599.9 → 509.9 (both, Δ=90)**; tok/s ~flat; **full arms PASS vs bank-split `gemv_kv`** |
| 31B (60 layers, no PLE) | ~660 | ~630 | 24 → 23 ms | ~8.5 → **~8.8 tok/s** | **727.9 → 667.9 (qkv, Δ=−60)**; **5.64 → 5.71 tok/s** (~flat) |

Honest read: **v1 is a down payment, not the 2×.** Dispatch drops engage; E4B
full-arms exactness vs product `FUSE_KV=1` is now **PASS** after bank-split
`gemv_kv`. **Keep `FUSE_QKV` / `FUSE_LAYER` OFF** until a clear tok/s win
(still ~flat). PLE alone is exact but tok/s flat. Reaching ~180
dispatches/token still needs encode-once/replay.

**E4B VERDICT vs bank-split `gemv_kv`** (`fusion_ab_e4b_20260720T012511Z`,
HAZARD=0; `FUSION_AB_FUSE_KV=1`; arms baseline/qkv/ple/both; TRACE=1):

| arm | exactness | dispatch | notes |
|---|---|---:|---|
| baseline (`FUSE_KV=1`) | — | 599.9 | bank-split `gemv_kv` reference |
| qkv | **PASS** | 575.9 (Δ=24) | stream identical |
| ple | **PASS** | 557.9 (Δ=42) | stream identical |
| both | **PASS** | 509.9 (Δ=90) | stream identical; tok/s ~flat |

**Prior FAIL** (`fusion_ab_e4b_20260720T001652Z`, dual-accumulate `gemv_kv`):
qkv/both FAIL @ tok5; ple PASS (Δ=42). Historical only. Light qkv-only PASS
at `fusion_ab_e4b_20260720T011736Z` (TRACE off).

**E4B VERDICT vs solo K/V** (`fusion_ab_e4b_20260720T010239Z`, HAZARD=0;
`FUSION_AB_FUSE_KV=0`; arms baseline/qkv; TRACE off):

| arm | exactness | notes |
|---|---|---|
| baseline (`FUSE_QKV=0`, solo K/V) | — | reference |
| qkv (`FUSE_QKV=1`) | **PASS** | 16-token stream identical |

**Do not default-on `FUSE_QKV`** (tok/s still ~flat). Exactness gate vs product
`FUSE_KV=1` is valid again. PLE remains opt-in candidate only (no tok/s win yet).

**31B VERDICT** (`fusion_ab_31b_20260719T080223Z`):

| arm | tok/s | dispatch | speedup |
|---|---:|---:|---:|
| baseline | 5.64 | 727.9 | — |
| qkv | 5.71 | 667.9 | 1.012× |

Exactness qkv **PASS** (stream identical). Dispatches qkv **PASS** (Δ=60.0). Engagement:
Δ matches 60 producers (1 fused QKV dispatch replacing Q+KV per layer).

## Flags (all default OFF)

```bash
GEMMA_METAL_FUSE_LAYER=1   # master: enable every fusion below
GEMMA_METAL_FUSE_QKV=0|1   # fused producer Q∥K∥V GEMV      (overrides master)
GEMMA_METAL_FUSE_PLE=0|1   # fused PLE lookup + residual    (overrides master)
GEMMA_METAL_FUSE_ROPE_KV=0|1  # fused rope+norms + kv_store (overrides master)
GEMMA_METAL_FUSE_GATE_DOWN=0|1  # Hot bounded-TG gate→down Q4 (default OFF; D18; not in FUSE_LAYER master)
GEMMA_METAL_ENCODE_ONCE=0|1   # ping-pong ledger + mini/E4B DecodeIcb path (default OFF)
GEMMA_METAL_DECODE_ICB=0|1    # mini+E4B Hot layer-graph ICB capture/skip (default OFF; needs ENCODE_ONCE; not 31B)
GEMMA_METAL_ICB_FREEZE_BINDS=0|1  # classic setKernelBuffer+tg_mem freeze → setArgTable=0 + execute_icb (default OFF; v0.5.7; PARKED)
GEMMA_METAL_ICB_RANGE_BATCH=0|1   # freeze-only: coalesce executeCommandsInBuffer ranges between barrier_after (default OFF; v0.5.8; PARKED)
GEMMA_METAL_ICB_COARSE_RANGES=0|1  # with range-batch: elide non-interfering barrier_after (default follows RANGE_BATCH; v0.5.9; PARKED)
GEMMA_METAL_PERSISTENT_INTERP=0|1  # mini gate→down + FA→o_proj interpreter (default OFF; mini step_inner hook)
```

Gating conditions (`HotQuantBanks::can_fuse_qkv`): Q4Mlx scheme, **RowMajor or
Interleaved4** (same layout on Q/K/V; twin `gemv_q4_mlx_simd_qkv_i4` when I4), matching
`cols` / `group_size`, `cols ≥ 256`, `cols % 16 == 0`, simd GEMV enabled. Anything else
silently falls back to the existing path. **E4B: keep `FUSE_QKV`/`FUSE_LAYER` OFF**
(tok/s ~flat); exactness vs `FUSE_KV=1` is PASS after bank-split `gemv_kv`.

Hot-path env flags (`FUSE_*` incl. `FUSE_GATE_DOWN`, `ENCODE_ONCE`, `DECODE_ICB`,
`PERSISTENT_INTERP`, `GEMV_SIMD`, probe/capture toggles, etc.) are cached
(`OnceLock` / `AtomicI8`) so decode does not re-read `environ` every layer/token
(F4 host-tax).

**Test-suite note (2026-07-19, full suite green 118/118):** the `FUSE_*` flags use the
*settable* `AtomicI8` pattern (`set_fuse_layer/qkv/ple_residual/rope_kv`), not `OnceLock` —
a OnceLock freezes at first call, so tests that `env::set_var` after any earlier test touched
the decode path read a stale `false` (five 2026-07-19 failures, all test-infra: three
`can_fuse_qkv` asserts from the frozen cache, plus two parallel-test races on
`set_persistent_interp` / the hazard lane). Tests now use the setters and the crate pins
`RUST_TEST_THREADS=1` via `.cargo/config.toml` (one-Metal-job doctrine applied to tests;
`encode_once_mini_parity` passes serially — the [175,71…]≠[175,211…] flip was hazard-lane
contamination from a concurrent test, not an encode-once bug).

## How to verify (on the M5)

```bash
cd gemma-metal
# One Metal job at a time — wait until no bench/diag_tok/golden_parity/fusion_ab is alive.
bench/fusion_ab.sh e4b     # then (separately): bench/fusion_ab.sh 31b
# QKV exactness vs solo K/V (not gemv_kv) — expect PASS; keep FUSE_QKV OFF:
FUSION_AB_ARMS=baseline,qkv FUSION_AB_FUSE_KV=0 bench/fusion_ab.sh e4b
# Light D18 gate→down only (expect Δ≈−42; exactness currently FAIL — keep OFF):
FUSION_AB_ARMS=baseline,gate_down FUSION_AB_TRACE=1 bench/fusion_ab.sh e4b
```

Runs arms (`baseline` / `qkv` on 31B; + `ple` / `both` on E4B) × measurements:

1. **Exactness** — `diag_tok` fixed prompt `[2,150,2307]`, 16 greedy tokens. Fusion changes
   dispatch *count*, never kernel math, so the fused stream **must** be identical. A stream
   diff is a bug (bad bank partitioning, missing RAW edge, wrong PLE combine order), not a
   tradeoff. This gate outranks tok/s — a speedup with a diff is the FP8-logit-saturation
   failure mode from the challenge.
2. **Dispatches/token** (`GEMMA_METAL_TRACE=1`) — must actually drop, else the gate predicate
   is rejecting and the fusion never engaged. Do not quote tok/s from the TRACE arm.
3. **tok/s** — the payoff, TRACE off.

Writes `bench/results/fusion_ab_<model>_<ts>.json`; exits non-zero if any arm breaks exactness.

**Harness mitigations:** Hot upload + TRACE can jetsam a 64 GB M5. `fusion_ab.sh`:
TRACE default **OFF** for e4b and 31b (`FUSION_AB_TRACE=1` to opt in for
dispatch-drop hard-fail), `FUSION_AB_FUSE_KV=0|1` pin (default 1; use 0 to gate
QKV vs solo K/V), aborts if another `bench`/`diag_tok`/`fusion_ab` is
alive (`FUSION_AB_ALLOW_BUSY=1` to override), real-model scrape for tok/s +
disp, exclusive GPU. **Host banks dropped:** `GpuSynthModel::from_host_banks`
takes banks by value, uploads by ref (no unused `host_q` twin), and
`drop(banks)` before return so Hot+session KV no longer overlaps full host
weight residency.

### Bit-exactness notes (deliberate choices)

- The QKV kernel reuses `load_x16_qdot` + `qdot16` and the identical pointer-walk loop from
  `gemv_q4_mlx_simd` (RowMajor) or `gemv_q4_mlx_simd_i4` (Interleaved4) — same file, same
  helpers, so no accumulation-order drift. Float unit (run isolated / `--test-threads=1`):
  RowMajor + E4B-dim + I4 **PASS**. Hot E4B stream exactness is a separate in-graph issue.
- The PLE kernel keeps the two-pass arithmetic order: `v = (s·nibble + b) · scale`, then
  `dst += combine_scale · v`. **Do not** algebraically fold `scale · combine_scale` — that
  changes rounding.
- The out-of-vocab branch falls through (`v = 0`) rather than early-returning, so `dst += 0`
  still executes and signed-zero behaviour matches the two-pass path exactly.
- `ple_out` is still written even though fused, so parity dumps keep working (~`dim` floats).

### If exactness fails — triage order

1. **`qkv` arm only** → bank/threadgroup partitioning. Check `tg_q`/`tg_k` vs actual
   `rows_q`/`rows_kv` (`GEMMA_METAL_TRACE_GEMV=1` prints them); confirm `hkv·head_dim` matches
   `k.rows`. A partial last TG in a bank is expected and handled by `row0 >= rows`.
2. **`ple` arm only** → barrier placement. The fused lane must emit the RAW barrier *before*
   the kernel (`x` is written by the preceding o_proj residual); the old barrier moved, it did
   not disappear.
3. **`both` fails, singles pass** → interaction, most likely a missing edge between the fused
   QKV write of `self.k`/`self.v` and `rms_qkv_rope`. `barrier_qkv` is unchanged and should
   still cover it; verify with `METAL_RUNTIME_HAZARD_BARRIERS=0` (always-on) — if always-on
   passes and hazard fails, it is a missing explicit RAW edge, not a math bug.

Also worth knowing (F3, measured 2026-07-19):

```bash
METAL_RUNTIME_HAZARD_BARRIERS=0|1 GEMMA_METAL_LOG=0 cargo run --release --bin bench -- --e4b
```

**Shipping default = hazard skip-auto ON** (`GemmaGpu::new_inference` calls
`set_hazard_barriers(true)` unless env=`0`). E4B A/B: always-on **17.72** vs skip-auto
**22.54–23.24 tok/s**. Exactness / fusion gates **must** pin `HAZARD_BARRIERS=0` —
under skip-auto, even the unfused baseline stream is non-deterministic across runs.
`fusion_ab.sh` does this automatically for the diag_tok arm.

## Files touched

| File | Change |
|---|---|
| `kernels/gemv_q4_mlx.metal` | **+** `gemv_q4_mlx_simd_qkv`, `gemv_q4_mlx_simd_qkv_i4` (append only) |
| `kernels/ple_lookup.metal` | **+** `ple_lookup_q4_mlx_residual` (append only) |
| `kernels/rms_qkv_rope.metal` | **+** `rms_qkv_rope_posbuf`, `rms_qkv_rope_kv_store` (append only) |
| `kernels/persistent_interp.metal` | **+** `persistent_interp_gate_down`, `persistent_interp_fa_o_proj` |
| `src/kernels.rs` | fusion + encode-once + persistent-interp flags; float parity |
| `src/gpu_model.rs` | fused lanes; `GEMMA_METAL_ENCODE_ONCE` → `try_replay_ready` + `mark_live_step`; mini parity + encode-µs A/B; light E4B Hot encode A/B (sequential); `from_host_banks` by-value + drop host banks / no `host_q` twin; mini `PERSISTENT_INTERP` `step_inner` hook |
| `crates/metal-runtime/src/cb_replay.rs` | `PingPongCbReplay` + `survey_cb_replay_api_gaps` + `IcbReplayStub` / `ArgTableSlotPlan` |
| `crates/metal-runtime/src/icb_smoke.rs` | Mini ConcurrentDispatch ICB + `executeCommandsInBuffer` (`copy_f32`; inherit+arg-table) |
| `bench/fusion_ab.sh` | A/B + exactness gate; `FUSION_AB_FUSE_KV` pin; TRACE-off default; GPU-busy preflight |

`build.rs` compiles `kernels/*.metal` automatically — no build change needed.

## Next in this lane

1. ~~**Compute ICB single-dispatch smoke on mini**~~ → **landed**
   (`MTLIndirectCommandBuffer` feature; `icb_mini_copy_smoke`; inherit+arg-table
   bridge). Classic `setKernelBuffer` freeze dead on MTL4 pipelines; ICB freezes
   dispatch only — binds still host-side. **True MTL4 CB object reuse is
   impossible** (`beginCommandBufferWithAllocator:` always re-records) — encode-once
   path is **compute ICB** (`executeCommandsInBuffer`), not CB reuse.
2. ~~Persistent-interpreter megakernel — mini graph only~~ → **v0 prototype landed**
   (gate→down + FA→o_proj stand-ins; opt-in `GEMMA_METAL_PERSISTENT_INTERP=1`).
3. ~~**Drop host banks** after Hot upload (31B OOM peak cut)~~ → **landed**
   (`from_host_banks(banks)` by value + no `host_q` twin; embed/lm_head host
   fallbacks retained for DFlash bind).
4. ~~Useful `GEMMA_METAL_PERSISTENT_INTERP` wiring on **mini decode**~~ → **landed**
   (`step_inner` hook; default OFF; Hot no-op).
5. ~~Prove encode-once NotWired + ICB stub + mini encode-µs A/B~~ → **landed**
   (`encode_once_mini_encode_ab`; `IcbReplayStub`; `survey_cb_replay_api_gaps`).
6. ~~**A0 scalars + mini DecodeIcb bridge**~~ → **landed** (`IcbScalarPool`
   `reset_step` + FA/kv/softcap GpuBuffers; `try_replay_icb` → `execute_icb`;
   `decode_icb_mini_replay` PASS; flags default OFF).
7. ~~**Full mini layer-graph DecodeIcb**~~ → **landed** (Binder tape →
   `from_commands`; binder-nop prep + `execute_icb` skips Metal layer encode;
   `encode_once_mini_encode_ab`; `cb_replay_wired:true`; cmds≈50;
   `live_encodes=1`; `icb_replays=9`; default OFF).
8. ~~**E4B bounded-TG gate→down**~~ → **landed opt-in; measured; hard-blocked**
   (D18: Δ=−42 engage; exactness FAIL = in-kernel mid visibility; flag OFF).
9. **Bind-tax + densify shape SHIPPED** — GEMV/MLP/PLE/RMS/`kv_store` →
   `IcbScalarPool` (`immediate=0`); always-densify + capacity grid (cursor 147).
10. **Mini token parity PASS (tape_execute)** — `decode_icb_mini_token_parity`
    (`live_out == icb_out`); Q4 `fuse_bf16` → `cast_bf16_to_f32` before
    `gemv_q4` (fixes ~cmd 19). Default frozen-tape dispatch; opt-out
    `GEMMA_METAL_ICB_TAPE_EXECUTE=0`. See D16.
11. **A2 bind-tax cut SHIPPED (v0.5)** — inherit still residual no-op (opt-in).
    Host wins: binder-nop PSO stand-in; packed tape `with_binder` + cached
    `gpu_addr`; atomic `IcbScalarPool` cursors. Ratio was ~0.77. See D16.
12. **A2 residual SHIPPED (v0.5.1)** — `IcbScalarWriteTape` + skip-nop layer
    loop on mini DecodeIcb replay (ops≈191). Ratio **~0.71–0.73**; parity PASS
    cmds=71. Opt-out `GEMMA_METAL_ICB_SKIP_NOP_LOOP=0`. See D16.
13. **A2 sticky setAddress SHIPPED (v0.5.2)** — skip unchanged arg-table slots
    on tape execute; latch `setArgumentTable` once per binder. Ratio **~0.58**;
    sticky 17/427; last_setAddress 410/427; parity PASS. See D16.
14. **E4B Hot DecodeIcb SHIPPED (v0.5.3)** — eligibility beyond
    `is_synthetic_mini()` (`is_hot_e4b` / `decode_icb_graph_eligible`); gated
    smoke `decode_icb_e4b_hot_smoke`. Flags default OFF. See D16.
15. **Light E4B encode A/B SHIPPED (v0.5.3; v0.5.5 Immediate-zero)** —
    `encode_once_e4b_hot_encode_ab`: encode_us **67195 → 54101** (ratio
    **~0.81**); tok/s **14.88 → 18.48**; cmds=594; prebuilt=594;
    elided=5003; last_setAddress **0/5003** (was 42). Artifact
    `encode_once_e4b_hot_ab_latest.json`. See D16.
16. **A2 prebuilt arg-tables SHIPPED (v0.5.4)** — per-cmd
    `MTL4ArgumentTable` freezes Buf binds; execute `setArgumentTable` × cmds.
    Mini: last_setAddress **0/427**; ratio **~0.44**; parity PASS; prebuilt=71.
    Opt-out `GEMMA_METAL_ICB_PREBUILT_TABLES=0`. See D16.
17. **Quiet E4B product tok/s A/B** — prior **20.69 → 16.11** regress
    **root-caused + fixed (v0.5.6):** tape execute forced always-on barriers
    (TRACE bar **364 → 599**). Capture `barrier_after` + replay under
    skip-auto → shipping **23.50** vs encode-once **23.38** (~flat); flags
    stay OFF. Artifact `encode_once_e4b_toks_latest.json`. See D16.
18. ~~**Diagnose `FUSE_GATE_DOWN` exactness FAIL**~~ → **root-caused** (D18:
    relaxed grid barrier mid visibility; two-dispatch peel exact; flag stays
    OFF). See `persistent_interp_gate_down_q4_e4b_dims_visibility_blocker`.
19. **A2 E4B Immediate residual cleared (v0.5.5)** — post_ff `copy_f32` +
    `set_u32` → `copy_f32_n` / IcbScalarPool; also pool-backed
    `copy_f32_to_offset` / `copy_f32_range` / `scale_f32_inplace`. Mini
    parity PASS; E4B last_setAddress **0/5003**. See D16.
20. **Freeze-binds SHIPPED (v0.5.7)** — `GEMMA_METAL_ICB_FREEZE_BINDS=1`:
    classic `setKernelBuffer`+tg_mem into ICB; execute setArgumentTable **0**;
    mini parity PASS; E4B product tok/s not a win (`execute_icb`×N tax).
    Sticky adopt + table fingerprint dedup on prebuilt path. See D16.
21. **Range-batch SHIPPED (v0.5.8)** — `GEMMA_METAL_ICB_RANGE_BATCH=1` (needs
    freeze): safe `executeCommandsInBuffer` ranges between `barrier_after`;
    E4B execute_icb **594→361**; mini parity PASS; quiet tok/s still not a win.
    See D16.
22. **Coarse-ranges SHIPPED + true-ICB PARKED (v0.5.9)** — large-Buf disjoint
    barrier elision with range-batch; unit+mini PASS; product E4B
    **coarse_elided=0** / execute_icb **361/594**; quiet **20.57→19.85**.
    Prefer prebuilt tape. See D16.
23. **E4B QKV+PLE `fusion_ab` re-measure (2026-07-20)** — PLE exactness+dispatch
    PASS (Δ=42); QKV/both exactness **FAIL** despite dispatch drops (Δ=24/90).
    Artifact `fusion_ab_e4b_20260720T001652Z.json`. Do not ship QKV/both.
24. **E4B QKV triage (2026-07-20):** unit float parity PASS (isolated); Hot stream
    FAIL @ tok5. Extra Device barrier / sync-after-fuse do **not** restore
    fused≡unfused.
25. **E4B Hot QKV dump (2026-07-20):** `GEMMA_METAL_QKV_AB_DUMP=1` (in-process
    fused↔unfused after first producer; HAZARD=0; FUSE_KV=1; INTERLEAVE=0).
    Artifacts `qkv_ab_dump_e4b_fuse1.txt` / `_setbytes`. **Diverge locus:**
    layer0 `k[41]` at **pos=1** (q bit-exact; k≈7.6e-6 / v≈1.9e-6); pos=0
    exact; dims `2048×2560`/`512×2560` tg `256/64/64` OK. setBytes dim binds
    **no help** + zeroed tokens → **reverted** to `IcbScalarPool`.
26. **E4B Hot QKV solo-K split (2026-07-20):** dump compares fused K vs solo
    `k_proj.gemv` vs `gemv_kv`. At pos=1: **solo ≡ fused** bit-exact; solo vs
    `gemv_kv` = `k[41]` ULP (7.6e-6). Artifacts `qkv_ab_dump_e4b_fuse1_solo_k.txt`
    / `qkv_ab_dump_e4b_k_solo.txt`. **CONFIRMED:** exactness FAIL = baseline
    `gemv_kv` dual-accumulate drift, not fused wiring. Keep `FUSE_QKV` OFF.
27. **E4B QKV vs solo K/V gate (2026-07-20):**
    `FUSION_AB_ARMS=baseline,qkv FUSION_AB_FUSE_KV=0` → exactness **PASS**
    (HAZARD=0). Artifact `fusion_ab_e4b_20260720T010239Z.json`.
28. **`gemv_kv` bank-split (2026-07-20):** `gemv_q4_mlx_simd_kv` (+`_i4`) now
    partitions TGs `[0,tg_k)→K` / `[tg_k,…)→V` (math ≡ solo `gemv`); host binds
    `tg_k` @ buffer(12), dispatches `tg_k+tg_v`. Hot dump
    `qkv_ab_dump_e4b_banksplit_kv.txt`: **solo≡gemv_kv≡fused** bit-exact
    (pos=0..7). Light `fusion_ab` `FUSE_KV=1` → exactness **qkv PASS**
    (`fusion_ab_e4b_20260720T011736Z.json`). **Keep `FUSE_QKV` OFF** (tok/s
    flat). Overturns prior FAIL vs dual-accumulate `gemv_kv`.
29. **E4B full arms re-measure under bank-split (2026-07-20):**
    `FUSION_AB_ARMS=baseline,qkv,ple,both FUSION_AB_TRACE=1 FUSION_AB_FUSE_KV=1`
    → exactness **qkv/ple/both PASS**; dispatches 599.9 → 575.9 / 557.9 /
    509.9 (Δ=24/42/90); tok/s 23.58 → 23.56 / 23.54 / 23.26 (~flat).
    Artifact `fusion_ab_e4b_20260720T012511Z.json`. **Do not default-on
    `FUSE_QKV`/`FUSE_LAYER`.**
30. **D16 quiet tok/s re-measure (2026-07-20):** shipping **23.56** vs prebuilt
    encode-once **23.46** (~flat). Artifact `encode_once_e4b_toks_20260720T013526Z`.
    Keep `ENCODE_ONCE`/`DECODE_ICB` OFF; true-ICB stays PARKED.
31. **Interleaved4 Hot A/B (2026-07-20T021529Z):** `GEMV_INTERLEAVE=1` exact
    **PASS** @ HAZARD=0; tok/s **0.953×** vs ship. **PARK / NO-WIN** — keep
    default OFF. Artifact `gemv_interleave_ab_e4b_20260720T021529Z.json`.
32. **Lane pause:** local levers exhausted (fusion flat, encode-once flat,
    BlockedBn PARK, simd occ PARK, Interleaved4 NO-WIN). Bottleneck ~75% MLP
    GEMV BW. **No next fusion-lane tick** until a new BW hypothesis (≫10%
    token-time). 31B rope only when free (serialize).

**Done in this lane:** QKV (RowMajor + **Interleaved4 twin**) + PLE + **rope+kv_store**
(element-local) + **encode-once v0.5.9** (pos_buf + A0 scalars + mini **and E4B
Hot** layer-graph DecodeIcb; A2 bind-tax + skip-nop + sticky + prebuilt
arg-tables; freeze/range/coarse measured + **true-ICB PARKED** vs prebuilt) +
**persistent-interpreter v0** (mini) + Hot `FUSE_GATE_DOWN` opt-in measured +
exactness **hard-blocked** (D18; keep OFF). Shared-KV append and consumer
Q-only rope stay unfused by design.

### Encode-once / CB replay v0.5.9 (mini + E4B Hot DecodeIcb)

**Hard blocker for CUDA-graph-style skip:** `MTL4CommandBuffer` cannot replay a
prior encoding. Default path is Binder-tape direct-dispatch + prebuilt
arg-tables (**preferred** for product). **True-ICB (opt-in, PARKED):** freeze +
range-batch + coarse → setArgTable=0; E4B execute_icb **361/594**;
coarse_elided=**0** on product hazard tape; quiet tok/s loses — keep OFF. Mini
layer-graph **wired** (cmds≈71; `live_encodes=1`; `icb_replays≥8`;
last_setAddress **0/427** via prebuilt). **E4B Hot** same path via
`decode_icb_graph_eligible` (not 31B); prebuilt light A/B **PASS**. Default
replay = **tape_execute** + **scalar-write skip-nop** + **prebuilt arg-tables**
(sticky fallback if opted out; mini token parity PASS; D16). Opt-in freeze/
range/coarse: `GEMMA_METAL_ICB_FREEZE_BINDS=1`, `RANGE_BATCH=1`,
`COARSE_RANGES` follows range (opt-out `=0`). Opt-out live encode:
`GEMMA_METAL_ICB_TAPE_EXECUTE=0`. Opt-out skip-nop:
`GEMMA_METAL_ICB_SKIP_NOP_LOOP=0`. Opt-out prebuilt:
`GEMMA_METAL_ICB_PREBUILT_TABLES=0`. **31B** still `IcbDecodeGraphNotMigrated`.

**Flags default OFF:** `GEMMA_METAL_ENCODE_ONCE`, `GEMMA_METAL_DECODE_ICB`,
`GEMMA_METAL_ICB_SMOKE`. Tape execute **default ON** when DecodeIcb is wired
(opt-out `GEMMA_METAL_ICB_TAPE_EXECUTE=0`). Skip-nop **default ON** when scalar
tape is present (opt-out `GEMMA_METAL_ICB_SKIP_NOP_LOOP=0`). Prebuilt tables
**default ON** (opt-out `GEMMA_METAL_ICB_PREBUILT_TABLES=0`).

**Verify (serialized — no fusion_ab / no 31B):**

```bash
cd gemma-metal
cargo test --lib decode_icb_mini_token_parity -- --test-threads=1 --nocapture
cargo test --lib encode_once_mini_encode_ab -- --test-threads=1 --nocapture
cargo test --lib decode_icb_e4b_hot_smoke -- --test-threads=1 --nocapture
cargo test --lib encode_once_e4b_hot_encode_ab -- --test-threads=1 --nocapture
```

Artifacts: `bench/results/decode_icb_mini_token_parity_latest.json`,
`bench/results/encode_once_mini_ab_latest.json`,
`bench/results/decode_icb_e4b_hot_smoke_latest.json` (smoke may skip),
`bench/results/encode_once_e4b_hot_ab_latest.json` (A/B may skip).

### Persistent interpreter v0 (mini only)

**Design:** Prove fused **grid-sync** edges as Hazy-style stand-ins, not a full
megakernel rewrite. Two sibling kernels share the barrier helper:

| Kernel | Program | Producer | Consumer |
|---|---|---|---|
| `persistent_interp_gate_down` | `PRODUCE_MID → BARRIER → DOWN_PROJ → HALT` | `mid=gelu(gate)*up` | dense down |
| `persistent_interp_fa_o_proj` | `PRODUCE_CTX → BARRIER → O_PROJ → HALT` | `ctx=tanh(q·k·scale)·v` (mock FA) | dense o_proj |

| Op | Work | Sync |
|---|---|---|
| `PRODUCE_*` | TG-partitioned element-local fill | element-local |
| `BARRIER` | sense-reversing device atomics (`deps[arrival,gen]`) | **grid-wide** |
| `*_PROJ` | `out[r] = Σ buf[i]·W[r,i]` dense f32 | needs *all* of buf |

**Why these edges:** Same dependency shape as shipping `gate/up → down` and `FA → o_proj`
(consumer needs the full producer buffer). Element-local fusions (QKV/PLE/rope+kv) stay
on the v1 path. FA stand-in is **not** softmax attention — only the sync edge.

**Metal forward-progress caveat:** Apple GPUs do not guarantee that all threadgroups
in a dispatch make progress. A consumer TG spinning on an atomic while a producer TG
is not resident can **deadlock**. Mitigations in v0: `n_tg ≤ 8` (mini residency),
`max_spin` + `fail` flag (timeout instead of hang), opt-in default OFF, **never**
wired into E4B/31B decode. Do not claim production readiness.

**Metal atomics API note:** device-scope atomics only accept `memory_order_relaxed`
(no acquire/release). Barrier signaling is relaxed+spin; buffer visibility after
the atomic edge relies on GPU coherence — fine for mini parity, not a production
memory-order proof.

**Flag:** `GEMMA_METAL_PERSISTENT_INTERP=1` (default OFF). Host rejects stand-alone
dispatch when off. Mini decode: `GpuSynthModel::is_synthetic_mini()` + flag →
`step_inner` runs both stand-ins once per layer on scratch (tokens unchanged).
Hot/E4B/31B: no-op. **Metal forward-progress caveat unchanged** — do not enable on
real models; spin timeout sets `fail` rather than hang.

**Verify (serialized, mini only — no fusion_ab / no 31B):**

```bash
cd gemma-metal
# One Metal job at a time.
cargo test --lib persistent_interp_flag_default_off -- --test-threads=1
cargo test --lib persistent_interp_gate_down_matches_unfused -- --test-threads=1
cargo test --lib persistent_interp_fa_o_proj_matches_unfused -- --test-threads=1
cargo test --lib persistent_interp_fa_o_proj_mini_dims -- --test-threads=1
cargo test --lib persistent_interp_fa_o_proj_flag_off_rejects -- --test-threads=1
cargo test --lib persistent_interp_mini_decode_hook -- --test-threads=1
```

**Files:** `kernels/persistent_interp.metal`; `src/kernels.rs`
(`KernelId::PersistentInterpGateDown` / `PersistentInterpFaOProj`, flag, dispatch,
float parity); `src/gpu_model.rs` (`is_synthetic_mini`, `PersistentInterpMiniHook`,
`step_inner` dispatches).

[hazy]: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
