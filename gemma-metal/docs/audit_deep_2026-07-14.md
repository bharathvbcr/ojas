# Deep technical audit — gemma-metal / DFlash (2026-07-14)

Independent read of tree + artifacts under `gemma-metal`.  
Prior summaries (`docs/bottleneck.md`, `docs/dflash_port.md`, session notes) are treated as claims; this report prioritizes **code + JSON evidence**.

**Host context:** Apple M5 Pro · 20 GPU · 64 GB · ~273 GB/s unified peak.

---

## Executive snapshot (measured)

| Surface | Artifact | Value |
|---------|----------|-------|
| E4B quiet greedy | `bench/results/latest_e4b_gemma_metal.json` | **23.92 tok/s**, TTFT 141 ms |
| 31B quiet greedy | `bench/results/latest_31b.json` | **6.83 tok/s**, TTFT 549 ms, Hot 17.87 GiB |
| Mini DFlash (steered) | `latest_dflash_parity_gates.json` | ≥ hazard greedy (**~733** vs **~729** tok/s); exact PASS; mean_accept full |
| Honest 31B DFlash | `run_dflash_parity_gates_1784001103.json` | greedy **~5.84**; DFlash **~1.17** @ bs=5; **mean_accept≈0**; exact **FAIL** |
| MLX product path | `serve_dflash.py` + `mlx032_nax_ab_31b.json` / gates | DFlash **~28–37 tok/s**, exact vs greedy **PASS** |
| Prior accept≈3.8 claim | `run_dflash_parity_gates_final.json` | **Discard** — NaN-collapsed target (doc + gates agree) |

`latest_dflash_parity_gates.json` has `"real_31b": null` — latest gate write is **mini-only** after dual-scratch work. Honest 31B numbers live in `run_dflash_parity_gates_1784001103.json` (and peers ~1784000775).

---

## Doc vs code reality (stale claims)

| Claim | Source | Reality |
|-------|--------|---------|
| `vm_act=1`, GEMM quarantined / `.wip` only | `docs/dflash_port.md` “Do not revert”, “True M>1 parked” | **Stale.** `GpuDecodeSession::new` allocates act scratch with `vm_act = VERIFY_MAX_M` (`gpu_model.rs` ~972–996). `kernels/gemm_q4_mlx.metal` is a live `.metal` source (build.rs compiles all `kernels/*.metal`). `step_verify` selects GEMM when `m>1 && gemm_verify_available() && lm_head.can_gemm_simd()`. |
| GEMM pipelines “not linked until dual scratch” | `kernels.rs` test comment ~2822 | **Stale comment.** Metallib includes GEMM; runtime gate is buffer size + pipeline resolve + `cols>256`. |
| “GEMM verify still required” for ≥25 | `docs/gates.md` distance table (was) | **Wrong framing.** GEMM verify is **wired and used on 31B Hot**, but the **layer graph is not Gemma4-dual-norm faithful** (below). |
| README Phase 4/6 | `README.md` | **Stale** (~15.9 E4B; 31B “download blocked”). Live artifacts: E4B ~23.9, 31B Hot loaded. |
| Roofline “~20% BW / 4× in GEMV” | `docs/bottleneck.md` body (superseded header) | **Superseded** by own CORRECTION + `kernel_roofline_finding.json`: kernels near peak; gap is per-token overhead. |
| architecture 31B MLP 21504 | `docs/architecture.md` | Suspect vs draft/config **10752** used in `DFlashConfig::gemma4_31b` and Hot load path — treat architecture row as unverified. |
| Mini mean_accept / ≥ greedy = product readiness | gate JSON “completed” tasks | Mini uses **MASK→anchor steer + short-circuit propose**; explicitly **not** for HF drafts (`dflash.rs` / bench notes). |

---

## Q1. Why `mean_accept≈0` on native 31B DFlash?

### Evidence (honest healthy target)

From `run_dflash_parity_gates_1784001103.json` `real_31b`:

- Greedy unique ≫ 1 (finite, not collapsed).
- Block 3 / 5: **accepts all zeros**, `mean_accept: 0.0`.
- Exact vs capture+conditioner greedy: **FAIL**.
- First diverging tokens (greedy vs DFlash new streams differ at index 0) → failure is **not** “later compounding only”.

MLX golden (same prompt `[2,105,4368,1246]`) in `golden_intermediates_31b.json`:

- `target_next_argmax = 531`
- `proposed_block_tokens = [14359, 532, 107, 563]`
- embed_scale √5376 ≈ 73.32; draft FA scale ≈ 0.0884 (native debug matches these scales).

Native debug in that JSON records scales correctly but **does not dump proposed tokens / h_ctx absmean** — localization still incomplete vs `docs/dflash_draft_contract.md` procedure.

### Code-level mismatch vs MLX / vs native greedy (ranked)

#### A. Smoking gun — `step_verify_gemm` is not Gemma4 dual-norm (+ drops `layer_scalar`)

31B **all 60** `layer_scalar` weights are non-unit (sampled from HF shard index): range **≈0.036–0.992**, mean ≈0.78; capture layers L1/L12/… are ~0.065–0.89.

| Op | `step_inner` (greedy / M=1) | `step_verify_gemm` (M>1 Hot path) |
|----|-----------------------------|-----------------------------------|
| Attn residual | dual: `o_proj → post_attn_ln → x +=` | **legacy** `o_proj.gemm_add_into_bf16_x` into `x` |
| MLP input norm | `pre_ff_norm` when present | **`post_attn_norm` only** (~2736) |
| MLP residual | dual: `down → post_ff_ln → x +=` | **legacy** `down.gemm_add_into` + optional overwrite-norm |
| `layer_scalar` | applied after residuals | bound as `_layer_scalar` and **never used** (~2478) |

Effect when DFlash uses GEMM verify (confirmed by 31B bench notes: “Q4 GEMM+FA(Tq=M) verify”):

1. `next_tokens[0]` (bonus) ≠ greedy `step()` → **exactness FAIL even at accept=0**.
2. Capture rows taken during verify are wrong-scale → conditioner `h_ctx` after block 1 is poisoned → draft proposals cannot track MLX.
3. Accept compares draft against **wrong** `next_tokens` → structural path to mean_accept≈0.

This dominates host/draft FA scale debates for the measured e2e exactness failure.

#### B. Draft / conditioner numeric ≠ MLX serve path

| Item | MLX (`serve_dflash.py`) | Native |
|------|-------------------------|--------|
| Draft linear quant | `nn.quantize(..., group_size=64, bits=4)` | `QuantScheme::q4_default()` = **plain Q4 g=32** (`dflash.rs` `from_draft`) |
| Conditioner `fc` | MLX module (bound to draft) | Same **Q4 g32** Hot upload |
| Draft FA scale / embed √H | aligned | aligned (contract + debug JSON) |
| RoPE ctx/block offsets (short prompt) | golden q=4, ctx=0 | Structure matches contract (inspection) |

Even with correct `h_ctx`, re-quantizing the draft at g32 vs MLX g64 will shift greedy draft argmax. That alone can hold accept near 0 **after** verify graph is fixed — second-order until A is repaired.

#### C. Capture feed / trim (★1b) — plausible but secondary on first token

`generate_with_dflash_inner` feeds `ctx_t = h_ctx_len - draft.cache_offset()`, verify appends M capture/conditioner rows, `commit_verify` trims `M-keep`. Shape matches MLX “trim to accepted+1”. First-block `h_ctx` comes from **correct** `step()` capture+project (post-softcap FC). So first-token exactness FAIL is **not** explained by ★1b; later blocks inherit verify-capture poison from A.

#### D. Ruled out / lower suspicion (this audit)

- Target FA scale 1.0 vs draft `1/√d` — documented and coded.
- Conditioner FC before softcap — fixed (deferred post-readback on `step`).
- Always-on barriers forced on HF 31B — fixed (mini only).
- MASK-steer on HF drafts — not applied (`is_synthetic_mini()` gate).
- `accept_block` math — unit-tested MLX shape; not the root for bonus≠greedy.

---

## Q2. Is M>1 GEMM verify used on 31B Hot, or dead / gated off?

**Used on the Hot path when M>1.**

Activation conditions (`gpu_model.rs` `step_verify` / `gemm_verify_available` / `HotQuantBanks::can_gemm_simd`):

- `m > 1`
- act + logits buffers ≥ `H×VERIFY_MAX_M` / `V×VERIFY_MAX_M` (true after dual scratch)
- metallib resolves `gemm_q4_mlx_simd` (+ i4)
- scheme `Q4Mlx`, layout RowMajor|Interleaved4, simd GEMV enabled, **`cols > 256`**, `cols % 16 == 0`

31B `hidden=5376` → `can_gemm_simd() == true`. Mini `H=256` → **`cols > 256` fails** → mini stays on **M×GEMV** (matches latest gate JSON `verify_path` text).

**Not dead code.** `.wip` twins (`gemm_q4_mlx.metal.wip`, `verify_batch_impl.rs.wip`) are historical quarantine leftovers; live `.metal` is linked.

**Caveat:** “landed” ≠ “correct.” GEMM path is a **different residual algebra** than M=1 decode (Q1.A). Throughput benefit is theoretical until fidelity matches `step_inner`.

---

## Q3. Capture-on exactness — what isn’t bit-stable; 31B risk

### Mini (green with caveats)

- Exactness definition: DFlash stream == **capture-on** greedy under **always-on** Dispatch barriers.
- PASS after: plain-Q4/bf16 poison fix, deferred conditioner, lm_head→softcap RAW, always-on lane.
- Still not bit-stable vs **capture-off** (`capture_off_vs_on_first_token` soft-skips / notes ultra-near ties).
- Speed lane drops capture after MASK-steer — measures amortization, not HF parity.

### 31B (red)

| Risk | Mechanism |
|------|-----------|
| **Critical** | GEMM verify ≠ dual-norm/`layer_scalar` → bonus and captures diverge from greedy |
| High | Mid-verify **host sync + `read_f32` of `x`** at every capture layer (~2799) — timing/hazard + BW tax |
| High | Capture+conditioner CB packing vs capture-off (bench comments: early tokens still flip) |
| Med | Hazard skip-auto: RAW edges across multiphase draft/verify; lm_head→softcap barrier was one known fail mode |
| Med | Conditioner/draft Q4 re-quant noise → even fixed verify may need g64/Q4Mlx to match MLX proposals |
| Low (short ctx) | Sliding-window ctx skip / offset (**★3**) — window 2048 ≫ S=4 on parity prompt |

**Risk for 31B product claims:** do not advertise “exact verify” for native DFlash until (1) GEMM layer graph matches `step_inner`, (2) accept=0 streams match capture+cond greedy, (3) dump vs `golden_intermediates_31b.json` for block-1 proposals.

---

## Q4. Roofline after bfloat2 + qmv_fast

### Kernel BW (isolated)

`bench/results/kernel_roofline_finding.json` (2026-07-13): Hot-resident Q4 GEMV already **~62–100% of ~273 GB/s**; simd gains **~1.03–1.5×**, not 4×. Old “~20% peak” came from upload microbench.

bfloat2 Hot sb + qmv_fast qdot/pointer-walk **did not move e2e tok/s** (gates: ~25.1 peak → **~23.9** this pass; flat/noise).

### Effective BW % (quiet e2e, E4B)

Using bottleneck weight traffic **~2.86 GB/token**:

| Metric | @ 23.86 tok/s |
|--------|----------------|
| ms/tok | ≈ **41.9** |
| Stream time @ 273 GB/s | ≈ **10.5 ms** |
| Residual (overhead) | ≈ **31.4 ms (~75%)** |
| Effective GB/s | ≈ **68** ≈ **~25%** of peak |

So: kernels can be near-roofline in isolation, while e2e still shows **~25% peak effective** because ~¾ of the token is **non-streaming overhead**.

### Biggest host / sync taxes (ordered)

1. **~780 dispatches / token** + phase barriers / bf16 casts (serialization).
2. **Token-boundary GPU wait** (honest lane packs encode; wall still dominated by CB complete).
3. DFlash extras: draft `synchronize` after propose; GEMM verify **per-capture-layer host readback**; conditioner FC GEMVs; verify softcap loop ×M.
4. API `step()` / generate readback stalls (secondary to 1–2 on quiet bench).

**Implication:** more ALU peel / pack tricks ≤ ~1.3×. Closing E4B 48–76 needs fusion/megakernel or speculative amortization (working DFlash), not another qdot tweak.

---

## Q5. Path to ≥ greedy then ≥15 / ≥25 — ranked fixes (honest impact)

Gates: 31B Q4 ≥**15**; 31B+MTP ≥**25**. MLX DFlash already clears both on this host.

| Rank | Fix | Expected impact | Effort / risk |
|------|-----|-----------------|---------------|
| **0 (ship)** | Productize `bench/serve_dflash.py` (mlx 0.32 + DFlash block=5) as the local 31B API | **Clears ≥15 and typically ≥25 today** (~28–37 tok/s) | Low — already works |
| **1** | Make `step_verify_gemm` **clone `step_inner` residual algebra**: dual-norm attn/MLP + **`layer_scalar` ×M** + `pre_ff_norm` | Unblocks exactness@accept=0; prerequisite for any native accept>0 | Med — careful port; NaN/hazard risk |
| **2** | Dump native block-1 vs `golden_intermediates_31b.json` (target_hidden absmean → fc_out → h_ctx → proposed tokens) | Localizes remaining draft gap after (1) | Low |
| **3** | Draft+fc quant parity with MLX (**Q4Mlx g64** or f16 fc) | Accept toward MLX ~3 @ bs=5 once verify is faithful | Med |
| **4** | Remove mid-verify host capture sync → device `copy_f32` staging like M=1 | Exactness + verify latency | Med |
| **5** | Only then chase native ≥ greedy / ≥25 via true M>1 verify amortization | With mean_accept~3 and working GEMM: **possibly ~1.5–2× greedy** (still need base ≥~12–15 to clear 25 without MLX-level kernels) | High |
| **6** | Decode fusion / megakernel (E4B→48, 31B base→15) | **~2–3×** if overhead→mlx’s ~2.6 ms/tok class | High |

**Do not expect:** GEMV peel alone to clear ≥15 on 31B (6.83→15 needs ~2.2×; kernels already near BW).  
**Do not claim:** mini steered ≥greedy as 31B readiness.

---

## Q6. Risk register

| ID | Risk | Evidence | Mitigation |
|----|------|----------|------------|
| R1 | **NaN / collapse regressions** | Historical `fast_tanh` gelu; always-on+capture HF collapse; shared H×M uninit tails | Keep `precise::tanh`; refuse DFlash measure when greedy unique≤1 (bench already); dual-tier or zero-init act tails |
| R2 | **Hazard barriers** | Skip-auto dropped lm_head→softcap RAW; mini needs always-on for exactness | Lane discipline: product hazard ON; exactness always-on **mini only**; never force AO on 31B capture without re-bench |
| R3 | **Dual agents / `gpu_model` conflict** | Docs say GEMM quarantined while code enables GEMM; `.wip` + live `.metal`; `_layer_scalar` looks like incomplete merge of dual-norm into GEMM | Single owner for verify path; delete or sync stale port doc; GATE: GEMM must call same residual helpers as `step_inner` |
| R4 | **False “accept high” metrics** | `run_dflash_parity_gates_final.json` mean_accept 3.8 on collapsed target | Unique-token health gate (present) — keep |
| R5 | **Steer contamination** | Mini MASK-steer short-circuit | `is_synthetic_mini` + comments; never call `steer_mask_positions_to` on HF |
| R6 | **Capture sync tax / races** | GEMM capture `synchronize+read_f32` per layer | Device-only staging |
| R7 | **Shared GPU draft↔target** | Draft must `synchronize` before target reuses CB; prior bf16×plain-Q4 poison | Keep plain-Q4 draft GEMV on f32; document shared-runtime invariant |

---

## Q7. Where should effort go? Recommendation

**Recommendation: productize `serve_dflash` for ship gates now; fix native verify graph before any further native accept chasing; park GEMV peel.**

Rationale:

1. **Gates ≥15 / ≥25 are already achievable** on this machine via MLX 0.32 + DFlash (`serve_dflash.py`). That is the only path with measured exact-vs-greedy PASS and tok/s in band.
2. **Native DFlash is blocked on correctness of M>1 verify**, not on missing GEMM entry points. Shipping more draft tweaks while `step_verify_gemm` omits dual-norm/`layer_scalar` will keep mean_accept≈0 and exactness FAIL.
3. **GEMV/qmv/bfloat2 is past diminishing returns** (e2e flat; kernels near peak). E4B 48 and 31B 15 on the custom stack need fusion or a working speculative path, not another peel.
4. Native work that still has ROI after serve productization: **(1) dual-norm GEMM verify**, **(2) golden intermediate parity**, **(3) draft quant g64** — then re-measure accept. Until accept ≫0 and exact PASS, native ≥25 is fiction.

---

## Artifact index (this audit)

| Path | Role |
|------|------|
| `bench/results/latest_e4b_gemma_metal.json` | E4B quiet |
| `bench/results/latest_31b.json` | 31B quiet |
| `bench/results/latest_dflash_parity_gates.json` | Mini gates (no real_31b) |
| `bench/results/run_dflash_parity_gates_1784001103.json` | Honest 31B DFlash |
| `bench/results/golden_intermediates_31b.json` | MLX block-1 tensors |
| `bench/results/golden_tokens_31b.json` | MLX exact streams |
| `bench/results/kernel_roofline_finding.json` | GEMV BW correction |
| `bench/results/mlx032_nax_ab_31b.json` | MLX 0.32 NAX verify speedup |
| `bench/serve_dflash.py` | Product MLX serve |
| `src/gpu_model.rs` | `step_inner` vs `step_verify_gemm` |
| `src/dflash.rs` | Draft / generate loop |
| `src/step_verify.rs` | `accept_block` |
| `docs/dflash_draft_contract.md` | MLX contract + localization checklist |

---

## Suggested next debug command (native)

After fixing GEMM dual-norm/`layer_scalar`, extend the 31B bench to dump first-block `target_hidden` absmean, `fc_out`, `h_ctx`, and `proposed_block_tokens` vs `golden_intermediates_31b.json` (harness notes already point at `bench/compare_intermediates.py`).
