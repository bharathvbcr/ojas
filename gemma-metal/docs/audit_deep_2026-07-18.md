# Deep technical audit #2 — gemma-metal (2026-07-18)

Successor to [`audit_deep_2026-07-14.md`](audit_deep_2026-07-14.md). That audit's rank-1 fix
(dual-norm GEMM verify) has **landed**; this audit re-reads the tree as of **2026-07-18 16:24**
(`gpu_model.rs`, `dflash.rs`, `kernels.rs`, `lib.rs` modified today) against artifacts that are
all **≤ 2026-07-14** — i.e. the current tree is largely **unmeasured**. New here: a quantified
per-dispatch cost model, host-side tax inventory, a barrier-lane doc/code discrepancy, an E4B
graph-fidelity finding, and an evidence-integrity bug in the bench writer.

**Host context:** Apple M5 Pro · 20 GPU cores · 64 GB · ~273 GB/s unified peak.

---

## Executive snapshot

| Finding | Severity | One-liner |
|---|---|---|
| **F1** `latest_e4b_gemma_metal.json` mislabeled | Evidence | Today's run wrote **31B weights_dir** under hardcoded E4B metadata (5.57 tok/s recorded as "E4B") |
| **F2** Fixed cost ≈ **35–40 µs per dispatch** (GPU-side) | Root cause | Derived from mini A/B; explains the full 3.5× gap vs MLX with no remaining mystery |
| **F3** Hazard barrier lane: docs say default ON, code default **OFF** | Discrepancy | `bench --e4b` measures with a Device barrier after *every* dispatch unless env set |
| **F4** `trace_op!` formats eagerly; `env::var` in hot loop | Host tax | ~500 heap-allocating `format!` + ~340 env syscalls per token, logging off or not |
| **F5** E4B layer graph is **not Gemma4-faithful** (legacy pre-LN residuals) | Correctness | post-attn/post-ff norms mis-placed vs MLX; explains Phase-3 "no real-weight parity" |
| **F6** DFlash 31B: exactness **PASS**, accept **0.77–1.3**, but DFlash **1.91 < greedy 4.57 tok/s** | Blocker | AO-while-capture tax + per-block host round-trips; dual-norm GEMM fix newer than every artifact |

---

## 0. Tree vs artifact freshness (measure before believing anything below)

| Item | Timestamp |
|---|---|
| `gpu_model.rs` / `dflash.rs` / `kernels.rs` / `lib.rs` | **2026-07-18 16:24** |
| Newest DFlash/31B artifact (`latest_dflash_parity_gates.json`) | 2026-07-14 10:19 |
| Newest E4B artifact (`latest_e4b_gemma_metal.json`) | 2026-07-18 17:16 — **but mislabeled, see F1** |
| All docs (`gates.md`, `bottleneck.md`, `dflash_port.md`) | ≤ 2026-07-14 |

Every accept/exactness/tok-s claim in the docs predates today's code. First action of any
follow-up session: **re-run `bench --e4b`, `bench --model <31b>`, `bench --dflash-31b`** and
re-pin gates.md before optimizing further.

---

## F1. Evidence integrity: the bench writer lies

`bench.rs::write_e4b_result` hardcodes `model`, `layers: 42`, `hidden: 2560`, `vocab`, and the
entire `notes` string; only `weights_dir`, tok/s, TTFT, steps are live. Today's
`latest_e4b_gemma_metal.json`:

```json
"model": "mlx-community/gemma-4-e4b-it-4bit",
"weights_dir": ".../gemma-4-31b-it-4bit/snapshots/696d436c...",
"layers": 42, "hidden": 2560,
"decode_tok_s": 5.5677, "ttft_ms": 686.76
```

That is a **31B run recorded as E4B** (5.57 tok/s / 687 ms TTFT are 31B-class numbers). Anyone
(human or agent) reading `latest_e4b_*.json` will conclude E4B regressed 24→5.6. The hardcoded
`notes` string ("~25 tok/s … interim≥30 unmet") is also stamped into every artifact regardless
of what ran.

**Fix (30 min):** write model dims/dir/notes from the loaded session, not literals; add the
weight-snapshot hash + download date (see risk R8). Also emit `run_31b_*.json` under its own
prefix so `latest_e4b` can never be clobbered by a 31B run.

---

## F2. Quantified: fixed per-dispatch cost ≈ 35–40 µs — the whole gap, no residual mystery

The mini synthetic graph is small enough that kernels are free, which isolates the launch
machinery:

- Mini greedy: **746 tok/s (always-on) / 739 tok/s (hazard)** → **1.34 ms/token**
  (`latest_dflash_parity_gates.json`).
- Mini dispatches/token: 3 layers × ~11 + head ≈ **36**.
- ⇒ **~37 µs per dispatch**, and — critically — **barrier mode made no difference (±1%)**:
  the cost is the *dispatch itself* (MTL4 encoder dispatch + argument-table set + barrier),
  not the barrier.

Cross-checks: FA decode microbench 19.7 µs/call for ~2 µs of work; softcap_argmax 173 µs for
a ~1 MB reduction. Fixed cost 20–40 µs per launch is consistent across all three surfaces.

Budget model, E4B real (42 layers, ~460 dispatches/token, ~2.86 GB weight traffic):

| Component | Est. ms/token |
|---|---|
| Host encode (GPU idle — single CB, commit at token end) | ~2.5 |
| Weight streaming @ ~250 GB/s effective | ~11–13 |
| **Dispatch fixed cost: 460 × ~37 µs** | **~17** |
| Attn-mat inefficiency (93–113 GB/s on 2560² vs ~270; ~14% of traffic) | ~2–3 |
| Syncs / argmax readback / probes | ~1–2 |
| **Total** | **~34–37** (measured: ~42 at 23.9 tok/s ✓) |

MLX at 76 tok/s = 13.2 ms/token ≈ the same streaming + ~2.6 ms of everything else — i.e. MLX
pays ~10 µs/dispatch over far fewer, fused dispatches.

**Implication (sharper than bottleneck.md):** the megakernel/fusion lever is not "one of
several" — it is the *only* lever that closes E4B to gate. Target arithmetic:

| Dispatches/token | Launch tax | Predicted E4B | Note |
|---|---|---|---|
| 460 (today) | ~17 ms | ~24 tok/s | measured |
| ~100 (1 fused attn + 1 fused MLP + glue per layer) | ~3.7 ms | **~48–55 tok/s** | gate band |
| ~45 (1 kernel/layer + head) | ~1.7 ms | **~60–65 tok/s** | mlx-class |

Same model explains 31B Hot (~60 layers × ~11 ≈ 660 dispatches ≈ 24 ms launch tax on a ~75 ms
token → ~30%): fusion alone takes 31B greedy ~8.5 → ~11–12; ≥15 needs fusion **and** working
DFlash.

**Where to fuse first (by dispatch count per layer):** rms_input + Q/KV GEMV (3→1), rope+kv_store
(2–3→1), FA+o_proj+post-attn-residual (3→1), rms_pre_ff+gate_up_gelu+down+residual (3–4→1).
All operands are already Hot-resident and the fused dual-norm epilogues
(`gemv_postnorm_add_into_*`) prove the pattern works.

---

## F3. Barrier-lane discrepancy: docs claim hazard default ON; code default is OFF

- `ab_flags::hazard_barriers()` env default: **false** → `Binder::dispatch` encodes a
  **Device-visibility barrier after every dispatch**.
- `set_hazard_barriers(true)` is called **only** inside `run_dflash_gates` (bench) and dflash
  lanes — **never** on the `--e4b` / `--model` real bench path, and not in `serve.rs` or
  session init.
- D12/D13/gates.md say "hazard barriers default **on** for decode".

So unless `METAL_RUNTIME_HAZARD_BARRIERS=1` was exported in the shell (the documented bench
commands don't), **every recent real-model number was measured in always-on mode**. The mini
A/B says the barrier itself is nearly free there, but mini's working set is KB-scale; on E4B
(4.3 GiB streamed) a Device-scope barrier can cost real cache traffic.

**Cheap experiment (1 command, do before any fusion work):**
```bash
METAL_RUNTIME_HAZARD_BARRIERS=1 GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 \
  cargo run --release --bin bench -- --e4b
```
If it moves ≥10%, make the session ctor set the lane explicitly (and update docs either way —
right now doc and code disagree about what the numbers even measure).

---

## F4. Host-side taxes in the hot loop (small, but pure waste and they gate latency)

1. **`trace_op!` formats eagerly.** The macro expands to
   `InferScope::begin($name, $detail)` and every call site passes `format!(...)` — the String
   is built **before** `infer_enabled()` is ever consulted (the comment in `diag.rs` claiming
   "formatted only when logging is on" is wrong for these call sites). ~460–600 heap-allocating
   `format!` + `op.into()` String + 2× `Instant::now()` per token, always.
   *Fix:* pass a closure / `format_args!` and gate on `infer_enabled()` inside `begin`.
2. **`std::env::var("GEMMA_METAL_LAYER_PROBE")`** is evaluated ~8× per layer inside
   `step_inner` → ~340 env lookups/token. Same for `GEMMA_METAL_CAPTURE_NOP` /
   `CAPTURE_BARRIER` per layer on capture paths. *Fix:* `OnceLock<bool>`.
3. **4 mutex acquisitions per dispatch** in `with_binder` (`active_m4` ×2, `const_cursor`,
   `dispatch_count`) → ~1,800 lock ops/token. Uncontended but not free; fold the counters into
   the first lock.
4. **String pipeline lookups** per call on some paths (`rt.pipeline("copy_f32")` in the E4B
   legacy post-ff branch — a per-layer HashMap<String> hit). Cache the pipeline handle.
5. **Token boundary is fully serial:** encode ~460 dispatches (~2.5 ms, GPU idle) → commit →
   wait → read argmax → repeat. The dual-allocator + free-slot mid-commit machinery
   (`ensure_m4_cb_open`) already exists but `MID_COMMIT` defaults 0. Committing every ~64
   dispatches would overlap host encode with GPU execution for ~2 ms/token back, without the
   historical wait-storm (free-slot pick fixed that).

Sum ≈ 3–5 ms/token ≈ 10% at current speeds — worth one cleanup pass, and items 1–2 also
pollute every TTFT measurement.

---

## F5. E4B graph is not Gemma4-faithful (deliberate, but now the quality story blocks the lane)

`step_inner` line ~1387: `use_gemma4_dual_norm = has_pre_ff && !has_ple` — **every E4B layer
(has PLE) takes the legacy path**, with comment "E4B keeps the prior fused Pre-LN+PLE path so
decode tok/s does not regress." The legacy path computes:

| Sub-graph | Legacy (E4B today) | MLX/HF Gemma4 |
|---|---|---|
| Attn residual | `x += o_proj(attn)` — **no post_attention_layernorm** | `x += post_attn_ln(o_proj(attn))` |
| MLP residual | `x += down; x = post_ff_ln(x)` — **norm applied to the whole residual stream** | `x += post_ff_ln(down)` |
| layer_scalar | not applied on legacy path | applied per layer |

If the E4B checkpoint ships `post_attention_layernorm` / `post_feedforward_layernorm` /
non-unit `layer_scalar` (the 31B does; loader reads them for E4B too), the E4B graph is
**architecturally wrong**, which is consistent with Phase 3's "no real-weight HF/MLX logit
parity." Consequences:

- All E4B tok/s numbers are for a *different network* — quality unknown, honest-lane doctrine
  ("no quality burn") is currently unverifiable on E4B.
- The E4B MTP accept measurements (75% clustered-assistant) ride on the same unverified graph.

**Fix:** flip E4B to the dual-norm path (the fused epilogues already exist — the 31B path shows
the cost is ~0 extra dispatches when `fuse_dual_norm` folds scalar+norm into the GEMV), then
run real-weight logit parity vs MLX before the next speed sprint. If parity was already green
on some private run, pin the artifact; nothing in `bench/results/` shows it.

---

## F6. DFlash 31B: verify graph fixed, economics still upside-down

State per the newest artifact (2026-07-14 10:19, **pre-dating today's code**):

| Metric | Value |
|---|---|
| Exact vs capture-on greedy (real 31B) | **PASS** (streams identical) |
| mean_accept | **0.77 @ bs=3 · 1.0 @ bs=5** (notes: ≈1.3 with Q8 conditioner sweep) — MLX ≈ 3.0 |
| DFlash e2e | **1.91 tok/s** vs greedy-with-capture **4.57** vs quiet greedy **~5.5–6.8** |

So the 07-14 audit's smoking gun (dual-norm GEMM verify + `layer_scalar` ×M) is resolved in
code (`step_verify_gemm` now mirrors `step_inner`, scalar folded, `m_u`-batched final norm) —
and the accept metric responded (0 → ~1). What still keeps DFlash *below* greedy:

1. **AO-while-capture tax.** `CaptureAlwaysOnGuard` forces always-on Device barriers for every
   captured step — i.e. the *entire 60-layer target forward* runs in the most serialized mode
   whenever capture is armed. Cost visible in the artifact: greedy 5.5–6.8 → 4.57 with capture.
   *Fix:* narrow the guard to the capture-copy RAW edges (explicit `barrier()` before each
   `copy_f32_range` + one before conditioner), keep the rest of the forward on the product lane.
2. **Per-capture-layer host `synchronize()` under hazard** is still in `step_verify_gemm`
   (guarded by `hazard_barriers()`); today it's avoided only because AO mode is forced — fixing
   (1) re-exposes it. Replace with device-side barrier + staged copy (the stage buffer already
   exists).
3. **Host round-trips per block:** `propose_block` builds draft input on host
   (`x.write_f32_prefix`) with an internal `synchronize`; conditioner assembles `h_ctx` via
   host `read_f32` + concat; verify ends with two more synchronizes. ≈ 6–10 full pipeline
   drains per block at ~0.5–1 ms each on a 60-layer-deep pipeline.
4. **Accept ceiling:** draft/fc re-quantized at plain Q4 g32 / Q8 (MLX: g64 4-bit via
   `nn.quantize`). MLX accept ~3.0 @ bs=5 vs native ~1.0–1.3. Every +1 accept ≈ +25% e2e at
   fixed verify cost.
5. **Verify amortization unmeasured:** MLX's win requires M=5 verify ≈ 1.1–1.3× the cost of
   M=1 (NAX GEMMs). Native `gemm_q4_mlx_simd` M=5 cost has **no microbench artifact**. If it's
   ~2× M=1, the whole speculative economics fail regardless of accept. Add
   `step_verify` (M=1..8) to the microbench section before more draft tuning.

Break-even sanity: tok/s ≈ greedy_cost/token × (1+accept) / (verify(M) + draft + sync). With
accept 3, verify(5) ≈ 1.3× greedy step, and drains cut to ~2/block, native DFlash ≈ 2.5–3×
greedy ⇒ ~15–20 tok/s on top of a fused ~11–12 greedy — that is the ≥15 (stretch ≥25) path.

---

## Ranked plan (expected value ÷ effort, updated)

| # | Action | Cost | Expected |
|---|---|---|---|
| 0 | **Re-measure current tree** (E4B, 31B, dflash-31b) + fix F1 writer; pin new gates rows | hours | Ground truth for everything below |
| 1 | **Hazard-lane A/B on real E4B** (env flag, F3) + make lane explicit in session init | minutes | 0–15%, settles doc/code split |
| 2 | **Host-tax pass** (F4: lazy trace_op, OnceLock envs, cached pipelines, fold locks) | ~1 day | ~2–4 ms/token + honest TTFT |
| 3 | **Layer fusion sprint** (F2: 460→~100 dispatches; fuse attn block + MLP block) | days–1 wk | E4B ~24→**~48–55**; 31B ~8.5→**~11–12** |
| 4 | **Capture de-AO + device staging** (F6.1–2) and kill per-block drains (F6.3) | ~1–2 days | DFlash from 0.4× → ~1.5× greedy |
| 5 | **Draft quant parity g64 + accept re-sweep**; add verify(M) microbench (F6.4–5) | 1–2 days | accept 1→~2.5–3; decides native-DFlash viability |
| 6 | **E4B dual-norm fidelity + real-weight parity gate** (F5) | 1–2 days | Unblocks honest-lane quality claims; ~0 speed cost with fused epilogues |
| 7 | Mid-commit overlap (encode ahead of GPU, free-slot allocators) | 1 day | ~2–3 ms/token, helps serve TTFT |

Ship lane unchanged: `serve_dflash.py` (MLX 0.32 + DFlash bs=5) still the only thing clearing
≥15/≥25 today.

---

## Risk register (delta from 07-14)

| ID | Risk | Note |
|---|---|---|
| R8 **(new)** | **Gemma 4 stealth weight update (2026-07-16)** — HF weights for all sizes changed with no version bump | `golden_tokens_31b.json` / `golden_intermediates_31b.json` and every cached snapshot predate it. **Do not re-pull mid-parity-work.** When re-pulling: re-generate goldens, re-quantize drafts, record snapshot hash + download date in every artifact (F1 fix enables this) |
| R9 (new) | latest_* artifacts clobbered cross-model (F1) | Prefix per model; write dims from session |
| R3 (carried) | Doc/code drift (hazard default, README, notes strings) | This audit supersedes; update D13 after the F3 A/B |
| R1/R2 (carried) | NaN & barrier-lane regressions | Unchanged; keep `precise::tanh`, keep exactness lanes |

---

## Code/artifact index (this audit)

| Path | Evidence for |
|---|---|
| `src/gpu_model.rs` 1247–2434 (`step_inner`) | F2 dispatch census, F4 probes, F5 legacy path (~1836, ~2160–2185) |
| `src/gpu_model.rs` 2588–3210 (`step_verify_gemm`) | F6 dual-norm landed, capture sync, end-of-verify drains |
| `src/dflash.rs` 1913–2048 (`generate_with_dflash_inner`), 670–750 (`propose_block`) | F6.3 host round-trips |
| `crates/metal-runtime/src/dispatch.rs` 112–143 | F3 per-dispatch auto-barrier |
| `crates/metal-runtime/src/ab_flags.rs` | F3 default = false (always-on) |
| `crates/metal-runtime/src/runtime.rs` 740–930 | F4.3 locks, mid-commit infra, single-CB token |
| `src/diag.rs` 241–254 + `trace_op!` 285 | F4.1 eager formatting |
| `src/bin/bench.rs` (`write_e4b_result`, real-model section) | F1, F3 (no hazard set on real path) |
| `bench/results/latest_dflash_parity_gates.json` | F2 mini-derived dispatch cost, F6 state |
| `bench/results/kernel_roofline_finding.json` | attn 2560² at 93–113 GB/s; MLP near peak |
| `bench/results/latest_e4b_gemma_metal.json` (2026-07-18) | F1 mislabel |
