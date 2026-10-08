# Decisions — gemma-metal

Inference-stack decisions for agents. Training decisions stay in
`arch_02_value_resid/metal-native/DECISIONS.md` and `arch02/DECISIONS.md`.

---

## D1. Duplicate extract `metal-runtime`, do not rewire metal-native yet

- **Decision:** Copy encode / tensor / GEMM / util metallib into
  `../MLSystemsLab/research/parameter-golf/crates/metal-runtime`. Leave `metal-native` intact.
- **Why:** Training drags ~9–10k LOC of bwd/Muon/XSA/VE. Inference needs ~25–35% of
  that surface (encode + prefill GEMM) and **0%** of the training tape.
- **Overturn if:** metal-native is frozen and a single shared crate is worth the
  migration risk (optional later dep of training on metal-runtime).

---

## D2. Honest product lane vs frontier challenge lane

- **Decision:** Ship **honest INT4 (+ MTP)**. Do not treat A10G ~510 TPS / PPL-budget
  burns (`w188` SWA shrink, layer/vocab amputation) as Mac targets.
- **Evidence:** Phase 0 on M5 Pro — E4B mlx-lm ~76 tok/s; 31B Ollama ~12 tok/s.
  Locked gates in [`docs/gates.md`](docs/gates.md).
- **Optional later:** SWA shrink only after parity green, as an explicit quality dial.

---

## D3. PLE always per-layer split at load

- **Decision:** Split packed PLE into per-layer Hot banks; enforce ≤4 GiB per buffer.
- **Why:** MLX/Metal — E4B packed bf16 ≈ 5.6 GiB aborts / fails single-buffer alloc.
- **Code:** `src/ple.rs`, validated in load (`weights.rs`).

---

## D4. Dual FA rewrite; do not reuse arch_02 FA

- **Decision:** Separate SWA@256 and global@512 tiled FA kernels in gemma-metal.
- **Why:** arch_02 FA hard-caps `head_dim ≤ 64`; TensorOps FA probe wants D==32.
- **Status:** Phase 2 — **real** FA-2 tiled kernels + GPU tests (`flash_attn_swa_h256`,
  `flash_attn_global_h512`). Still dense-buffer only; not wired to KV-ring / share graph.

---

## D5. Decode = GEMV; prefill = GEMM (± quant MTLTensor)

- **Decision:** Never decode through M=1 TensorOps matmul tiles. Prefill may use
  TensorOps GEMM / future quant MTLTensor. The host-side `mtl_tensor` module
  reports that path as unwired through `QUANT_PREFILL_GEMM_WIRED`; its old
  always-error `try_quant_tensorops_prefill_gemm` stub was removed.
- **Why:** Decode is bandwidth-bound; BaseRT / challenge transfer = fewer bytes/token + MTP.

---

## D6. Core AI / stateful KV off v1 critical path

- **Decision:** v1 = custom Metal decode loop. Optional Core AI / ANE export A/B in Phase 6+.
- **Why:** Gemma 4 needs multi-state KV (SWA 256 + global 512 over non-shared layers);
  Apple 2-state sequential engines are insufficient; convert already painful for dynamic slice.

---

## D7. MTP synthetic E2E in Phase 5; real accept later

- **Decision:** Ship `mtp` module + wire `GpuDecodeSession::generate_mtp_smoke`
  (draft→verify on decode path). Load real HF assistant when cached; full cross-KV
  4-layer draft forward still stand-in until Hot E2E accept is measured.
- **Why:** Biggest open Mac lever after INT4.
  - **Status:** `google/gemma-4-E4B-it-assistant` **loaded**; cross-KV from GPU shared
  buffers; accept **75% (6/8)**; e2e ~**12.1 tok/s** with early-reject.

---

## D8. MTLTensor device probes are experimental (objc2 SIGSEGV risk)

- **Decision:** Unit tests build Int8 **descriptors** only. Do not call
  `tensorSizeAndAlign` / `newTensor` in default CI without a Phase-2 smoke gate.
- **Evidence:** Comments in `metal-runtime/src/mtl_tensor.rs` — some objc2/SDK combos
  SIGSEGV on unsupported layouts.
- **Int4 / FP8 E8M0 host descriptors:** not in objc2-metal 0.3;
  `QuantDType::{Int4,Fp8E8M0}` maps to `Err`. This is separate from the raw-address
  TensorOps design. `tessl::nax_verify_readiness()` records that Q4 TensorOps is
  **not shipped** because its shader-side sub-byte tensor constructor and wired
  host dispatch are missing; verify therefore stays on hand simdgroup Q4.

---

## D9. Q4 group_size default 32

- **Decision:** `QuantScheme::q4_default()` uses group_size **32** (challenge often g128).
- **Overturn if:** A/B on M5 Pro shows g128 wins decode bandwidth without hurting parity.

---

## D10. Phase 4 Hot decode = GPU KV + packed encode + vectorized GEMV

- **Decision:** Keep KV on GPU; `async_encode`; Q4 GEMV = one thread/row + 8-wide
  nibble peel (tiled TG entry kept as `*_tiled` for A/B). GPU embed lookup + GPU
  argmax index propagate. Dynamic TG `x_cache` via `setThreadgroupMemoryLength(cols*4)`.
- **Evidence (2026-07-14 bfloat2 / qmv_fast):** E4B ~**23.9 tok/s** / TTFT ~142 ms
  (prior peak ~25.1). 31B Hot ~**6.83 tok/s**. Still below gate 48–60 / mlx ~76.
  Default decode = row-major `gemv_q4_mlx_simd` with **interleaved bfloat2** Hot sb +
  MLX **qdot**/pointer-walk; Interleaved4 weight pack opt-in (`GEMMA_METAL_GEMV_INTERLEAVE=1`,
  measured ~22.8 — left off). BlockedBn opt-in. Fuse MLP + producer K∥V on by default.
- **PLE:** Q4 Hot residual lookup on; per-layer gate/proj still skipped.

---

## D11. 31B custom Hot: shards landed; speed below gate

- **Decision:** HF `mlx-community/gemma-4-31b-it-4bit` **complete** (~17 GB / 4 shards).
  Ollama floor remains ~12.3 tok/s. Custom Hot measured **~6.83 tok/s** (TTFT ~0.55 s) —
  below gate ≥15. Serve `--preset 31b` Hot path confirmed; `attention_k_eq_v` shares K/V Hot.
- **Overturn if:** Same decode stack lifts E4B toward mlx (~76) and re-bench 31B ≥15.

---

## D12. Trace first; soft mid-commit off until overlap is proven

- **Decision:** Env-gated `GEMMA_METAL_TRACE` / `bench --trace*` before claiming tok/s.
  Gate numbers always TRACE **off**. Dual MTL4 allocators + **free-slot pick** for
  mid-commit; soft `commit(false)` enabled only via `METAL_RUNTIME_MID_COMMIT=N`
  (default 0 / off).
- **Evidence:** Blind mid-commit with alternating allocators caused wait-storm
  (~737 commits/token). Free-allocator pick waits only when both slots are busy.
- **Next:** A/B mid-commit thresholds once host encode ≥ GPU chunk time.

---

## D13. Prefill / parity wiring (speed push)

- **Decision:** `step_prefill` skips lm_head/argmax; tied embed/lm_head share one Hot bank;
  Hot argmax scratch + dual GPU-resident `seed_tok` / `argmax_tok`; consumer layers Q-only `rms_qkv_rope`;
  `post_feedforward_layernorm` on GPU when present. Inference runtime: no CounterHeap.
  Hazard barriers default **on** for decode via `GemmaGpu::new_inference` →
  `set_hazard_barriers(true)` unless `METAL_RUNTIME_HAZARD_BARRIERS=0` (always-on Device
  barriers for golden/parity). MLX **PLE Q4 Hot** residual via `ple_lookup_q4_mlx`;
  per-layer input gate/projection still skipped.
- **Evidence (F3 A/B, E4B Q4 Hot, 2026-07-19):** shipping skip-auto **22.54–23.24 tok/s**
  vs always-on **17.72 tok/s** (~1.27×). Artifact:
  `bench/results/hazard_ab_e4b_20260719T065759Z.json`. Numbers measure real E4B decode
  (16 steps after prefill T=4), TRACE off, FUSE_LAYER off — not mini-graph lines.
- **Caveat:** E4B token streams under skip-auto are **non-deterministic** across runs
  (incomplete RAW coverage). Exactness / fusion gates must pin `HAZARD_BARRIERS=0`.
- **Why:** TTFT / WS / parity hooks from plan P1/P3 without SWA amputation.

---

## D14. DFlash block-verify supersedes clustered MTP for 31B×path

- **Decision:** Port `z-lab` DFlash block-verify into gemma-metal ([`docs/dflash_port.md`](docs/dflash_port.md)).
  Step 1 = `step_verify(tokens[M≤8])` + `trim_kv` / `commit_verify` + MLX-aligned
  `step_verify::accept_block`. Dual `seed_tok`/`argmax_tok` (+ packed `verify_seeds`/
  `verify_outs`). **M>1 Hot Q4Mlx:** `step_verify_gemm` via `gemm_q4_mlx_simd*` + FA(Tq=M);
  act scratch ×`VERIFY_MAX_M`; plain-Q4 / non-simd falls back to M× GEMV. Steps 2–4:
  hidden capture + `DFlashGpuConditioner` + **`DFlashGpuDraft`** + `generate_with_dflash`.
  Clustered-assistant `mtp.rs` remains for E4B experiments but is not the 31B≥25 path.
- **Evidence:** MLX 0.32 + DFlash block=5 → ~37 tok/s on 31B. Engine after GEMM wire:
  mini ~387 vs greedy ~677; 31B ~2.02 vs greedy ~5.48 (e2e still capture/draft-bound).
- **Next:** device-side capture (no mid-layer sync); draft GEMM; batched softcap; finite
  31B greedy logits → exact-vs-greedy; product ≥25.

---

## D15. Layer fusion v1 = dispatch-count lever, opt-in, exactness-gated

- **Decision:** Attack per-token overhead by cutting **dispatch count**, not GEMV math.
  Opt-in flags (default **OFF**): `gemv_q4_mlx_simd_qkv`, `ple_lookup_q4_mlx_residual`,
  and `rms_qkv_rope_kv_store` (rope+kv_store; master `GEMMA_METAL_FUSE_LAYER=1`).
  Existing kernels untouched (append-only), so the default compiled path is byte-identical.
- **Why:** `docs/audit_deep_2026-07-18.md` F2 measured **~37 µs fixed cost per dispatch**
  → ~17 ms of a ~42 ms E4B token over ~460–560 dispatches.
- **Scope limit:** only fusions with **local** dependencies. Full attn/MLP block kernels
  (~4–5 disp/layer) need grid-wide sync / persistent interpreter — **not shipped**; next
  after encode-once A/B, not attempted as silent math changes.
- **Gate:** `bench/fusion_ab.sh` — exactness under `METAL_RUNTIME_HAZARD_BARRIERS=0`
  (shipping hazard makes E4B streams non-deterministic even unfused); dispatches/token
  must drop; tok/s under shipping hazard is informational.
- **Status (Lane B 2026-07-19 / re-measure 2026-07-20):** built + measured on E4B and 31B.
  - **E4B (2026-07-20T001652Z, arms baseline/qkv/ple/both, TRACE=1, `FUSE_KV=1`):**
    dispatches 599.9 → qkv 575.9 (Δ=24) / ple 557.9 (Δ=42) / both 509.9 (Δ=90).
    Exactness **PLE PASS**; **QKV FAIL vs dual-accumulate `gemv_kv`** (tok5) —
    historical; overturned by bank-split fix below. Artifact:
    `bench/results/fusion_ab_e4b_20260720T001652Z.json`.
  - **E4B QKV vs solo K/V (2026-07-20T010239Z, `FUSION_AB_FUSE_KV=0`):** exactness
    **qkv PASS**. Artifact `bench/results/fusion_ab_e4b_20260720T010239Z.json`.
    Confirmed prior FAIL was baseline dual-accumulate ULP, not fused wiring.
  - **`gemv_kv` bank-split (2026-07-20):** `gemv_q4_mlx_simd_kv` (+`_i4`)
    partitions `[0,tg_k)→K` / `[tg_k,…)→V` (solo reduce per TG; host
    `tg_k` @ buf 12). Hot dump **solo≡gemv_kv≡fused** bit-exact
    (`qkv_ab_dump_e4b_banksplit_kv.txt`). Light fusion_ab
    `FUSION_AB_ARMS=baseline,qkv FUSION_AB_FUSE_KV=1` → exactness **qkv PASS**
    (`fusion_ab_e4b_20260720T011736Z.json`). **Do not default-on `FUSE_QKV`**
    (tok/s still ~flat). Product `FUSE_KV=1` is again a valid exactness baseline.
  - **E4B full arms under bank-split (2026-07-20T012511Z, TRACE=1,
    `FUSE_KV=1`, arms baseline/qkv/ple/both):** exactness **qkv/ple/both
    PASS**; dispatches 599.9 → 575.9 / 557.9 / 509.9 (Δ=24/42/90); tok/s
    23.58 → 23.56 / 23.54 / 23.26 (~flat). Artifact
    `bench/results/fusion_ab_e4b_20260720T012511Z.json`. Overturns prior
    both FAIL at `20260720T001652Z`. **Keep `FUSE_QKV`/`FUSE_LAYER` OFF.**
  - **31B (QKV only; no PLE):** exactness **PASS**, dispatches **727.9 → 667.9
    (Δ=−60 = 1/producer)**, tok/s 5.64 → 5.71 (~flat; do not overclaim). Artifact:
    `bench/results/fusion_ab_31b_20260719T080223Z.json`. Engagement inferred from
    exact Δ=60 (TRACE does not print `gemv_qkv` without `TRACE_GEMV`). Later
    re-loads jetsam-OOM'd Hot upload (~36–55 GiB peak on 64 GB) — harness now
    skips TRACE by default on 31b + drops host banks after Hot upload; still
    **serialize Metal work** (one `bench`/`diag_tok`/`fusion_ab` at a time; never
    co-run e4b with 31b). Fusion remains opt-in until a clear tok/s win lands with
    encode-once. Do not enter into `gates.md` as default-on.
  - **rope+kv_store:** written + E4B exactness/dispatch PASS; float unit parity
    `rms_qkv_rope_kv_store_matches_unfused`. Not 31B-measured (expect −60 disp).
    Harness: `FUSION_AB_TRACE` defaults off (e4b+31b); GPU-busy preflight.
  - **encode-once (D16 v0.5.9):** opt-in `GEMMA_METAL_ENCODE_ONCE=1` +
    `GEMMA_METAL_DECODE_ICB=1` (default OFF). Mini layer-graph DecodeIcb
    **wired** (tape_execute default; cmds≈71; parity PASS). **A2 prebuilt
    per-cmd arg-tables:** mini last_setAddress **0/427**. **E4B Hot** light
    encode A/B **PASS** (prebuilt); product tok/s ~flat after `barrier_after`
    (v0.5.6). **Freeze + range-batch + coarse-ranges (v0.5.7–0.5.9):** opt-in
    freeze/range; coarse elides non-interfering `barrier_after` (product E4B
    **coarse_elided=0**, execute_icb still **361/594**). Mini FREEZE+RANGE
    parity **PASS**. Quiet E4B still not a win. **PARK true-ICB** — prefer
    prebuilt tape. Flags stay OFF. **31B** still `IcbDecodeGraphNotMigrated`.
    See D16.
  - **Interleaved4 QKV twin:** `gemv_q4_mlx_simd_qkv_i4` + host layout select;
    `can_fuse_qkv` accepts RowMajor|Interleaved4 (matched). Float unit
    (isolated) **PASS**. Hot A/B pins `GEMV_INTERLEAVE=0`. No 31B A/B.
  - **Harness:** `FUSION_AB_TRACE` default **OFF** (e4b + 31b; opt in for dispatch
    hard-fail); real-model scrape; GPU-busy preflight. No `FUSE_GATE_DOWN` in
    layer-fusion arms. `FUSION_AB_FUSE_KV=0|1` pin (default 1).
- **Overturn if:** A/B shows no dispatch drop or exactness fails under HAZARD=0.
- **QKV triage (2026-07-20):** Float unit (isolated) RowMajor/E4B-dim/I4 **PASS**.
  Hot E4B stream diverged @ `new_token[4]` vs dual-accumulate `gemv_kv`
  (deterministic). Forced Device `barrier_qkv` / `synchronize` after fused QKV
  did not restore fused≡unfused — confirmed not a missing RAW barrier.
- **QKV Hot dump (2026-07-20):** Opt-in `GEMMA_METAL_QKV_AB_DUMP=1`. Pre-fix:
  pos=0 exact; pos=1 `k[41]` ULP vs `gemv_kv` (q=0). setBytes dim binds
  reverted (zeroed stream). Artifacts `qkv_ab_dump_e4b_fuse1.txt` (+`_setbytes`).
- **QKV solo-K split (2026-07-20):** at pos=1, **`k_solo ≡ fused` bit-exact**;
  **`k_solo` vs dual-accumulate `gemv_kv` = ULP gap**. Hypothesis CONFIRMED.
- **QKV vs solo-K/V gate (2026-07-20):** `FUSION_AB_FUSE_KV=0` → exactness
  **qkv PASS**. Artifact `fusion_ab_e4b_20260720T010239Z.json`.
- **`gemv_kv` bank-split fix (2026-07-20):** shipped. Dump
  `qkv_ab_dump_e4b_banksplit_kv.txt` + fusion_ab
  `fusion_ab_e4b_20260720T011736Z.json` → **solo≡gemv_kv** and **FUSE_QKV vs
  FUSE_KV=1 exactness PASS**. Keep `FUSE_QKV` OFF for product (tok/s flat).
- **E4B full arms re-measure (2026-07-20):** TRACE=1 + `FUSE_KV=1` →
  **qkv/ple/both exactness PASS** (`fusion_ab_e4b_20260720T012511Z.json`).
- **D16 quiet re-measure (2026-07-20):** shipping **23.56** vs prebuilt
  encode-once **23.46** (~flat). Artifact `encode_once_e4b_toks_20260720T013526Z`.
  Keep `ENCODE_ONCE`/`DECODE_ICB` OFF.
- **E4B decode time profile (2026-07-20T015429Z, shipping, `TRACE=sync`):**
  Hot steady **disp≈600**; **host encode ~0.17 ms** (`TRACE=1`, not the gap).
  GPU buckets (sync; scale to quiet ~42 ms/tok @ 23.58 tok/s): **42-layer stack
  ~34–39 ms (~90%)**, **lm_head ~2–4 ms (~8%)**, **softcap/argmax <0.2 ms**.
  Inside the layer stack, byte traffic still matches `bottleneck.md`: **MLP
  gate/up/down ~75%**, **attn GEMVs ~14%**, **FA/RMS/RoPE ~5%**. **Why −90 disp
  flat:** launch tax **90×~37 µs ≈ 3.3 ms** on a **~42 ms** token (**~8% ceiling**);
  host encode already negligible; **~510 disp** + **~2.86 GiB/tok** GEMV remain;
  measured **23.58→23.26 tok/s** is noise. Artifact
  `bench/results/decode_profile_e4b_20260720T015429Z.json`. **Hypothesis (refined):**
  dispatch-count fusion alone cannot move product tok/s until **MLP GEMV bandwidth**
  or **layer-scale fusion** (not QKV/PLE glue) cuts bytes or launch tax by **≫10%**.
- **GEMV BlockedBn A/B (2026-07-20T015644Z):** opt-in
  **`GEMMA_METAL_GEMV_BLOCKED=1`** vs default row-major simd (`GEMV_SIMD=1`,
  fusion/encode-once OFF, quiet logs). **`diag_tok` @ HAZARD=0: FAIL** — blocked
  stream diverges @ token 0 (63508 vs 133533). Paired quiet bench same session:
  **7.32 → 6.55 tok/s** (**0.895×**; blocked slower). **Do not default-on
  `GEMV_BLOCKED`.** Artifact `bench/results/gemv_blocked_ab_e4b_20260720T015644Z.json`.
- **BlockedBn triage (2026-07-20T0202Z) — PARK:**
  1. **Root cause (catastrophic):** Hot default `FUSE_BF16` feeds **bf16** `x` into
     `gemv_q4_mlx_blocked*` / blocked `gate_up_gelu`, which peel **`float *x`**
     (reinterpret). Same trap as classic Q4 (2026-07-19). **Fix shipped:** bf16→f32
     expand in `HotQuantBanks::gemv_impl` (BlockedBn + RowMajor float-peel fallback)
     and `gemv_q4_mlx_gate_up_gelu_impl` (BlockedBn). Unit:
     `gemv_q4_mlx_blocked_{bf16_x,gate_up_gelu_bf16_x,e4b_shapes}` + opt-in
     `GEMMA_METAL_BLOCKED_HOT_PARITY=1` real E4B weights vs `gemv_q4_mlx_wide`
     (**max_err ≈ 1e-6..2e-6** on q/o/gate/up/down including wide down).
  2. **Irreducible for token-exact vs simd (and vs float peel @ `diag_tok`):**
     after the cast fix, `LAYER_PROBE` shows L0 qkv/FA/MLP **lockstep** with
     `GEMV_SIMD=0` float peel through ~L2, then **coop-reduce associativity drift**
     grows (L23 max 470 vs 507) → greedy tok0 still diverges. Not a layout/scale/zero
     mismatch. Prior quiet A/B also **slower** (0.895×). **Park `GEMV_BLOCKED`**
     (keep default OFF; kernels/repack retained for A/B). Artifact
     `bench/results/gemv_blocked_bf16fix_20260720T020210Z` + layer-probe notes in
     this entry.
- **Simd occupancy A/B (2026-07-20T021250Z) — PARK / NO-WIN:** quiet paired E4B
  (fusion/encode-once/`GEMV_BLOCKED` OFF; `GEMV_SIMD=1` row-major; `HAZARD=0`
  exactness). Shipping already matches MLX `qmv_fast` (**2 SG × 4 rows**, packs=2);
  simd path uses **no `tg_mem`** (classic `cols*4` only on float peel / BlockedBn).
  Arms: **ship 4×2** 20.54 tok/s (ref); **r2 2×2** 20.34 (**0.990×**, exactness
  **FAIL @ tok1**); **sg4 4×4** 20.80 (**1.013×**, exact PASS — noise, do not
  default-on). Prior **`SIMD_ROWS=8`** already slower (21.27 vs 25.10 historical).
  **Keep shipping 4×2.** Harness `bench/simd_occ_ab.sh`. Artifact
  `bench/results/simd_occ_ab_e4b_20260720T021250Z.json`.
- **Interleaved4 Hot A/B (2026-07-20T021529Z) — PARK / NO-WIN:** quiet paired E4B
  (`GEMV_SIMD=1`; fusion/encode-once/`GEMV_BLOCKED`/`FUSE_GATE_DOWN` OFF;
  `HAZARD=0` exactness). **ship `GEMV_INTERLEAVE=0`** 18.75 tok/s vs **i4 `=1`**
  17.86 (**0.953×**). Exactness **PASS** (identical greedy stream). Confirms
  2026-07-14 ~22.8 vs ~23.8 under current stack. **Do not default-on
  `GEMV_INTERLEAVE`.** Harness `bench/gemv_interleave_ab.sh`. Artifact
  `bench/results/gemv_interleave_ab_e4b_20260720T021529Z.json`.
- **Lane pause (2026-07-20):** local tok/s levers exhausted — fusion exact/flat
  (−90 disp ≪10% ceiling), encode-once flat, `GEMV_BLOCKED` PARK, simd occ
  PARK/NO-WIN, Interleaved4 NO-WIN. Profile still **~75% MLP GEMV BW**
  (`decode_profile_e4b_20260720T015429Z` / `docs/bottleneck.md`). **Stop** until
  a new BW hypothesis (layer-scale MLP / persistent megakernel / weight-traffic
  cut with expected ≫10% token-time). Park gate→down (D18); 31B rope when free
  (serialize; no 31B co-run). Do not default-on
  `FUSE_QKV`/`FUSE_LAYER`/`ENCODE_ONCE`/`GEMV_BLOCKED`/`GEMV_INTERLEAVE`.
  True-ICB PARKED (D16).

---

## D16. Inference-lane doctrine: ICB out (train); encode-once = compute ICB; MID_COMMIT ≠ encode-once

- **Decision:**
  1. **ICB stays out for training** (and is not the default inference path). Training
     encode remains live Metal 4 begin→encode→end→commit.
  2. **Encode-once “CB replay” for inference means compute ICB**
     (`MTLIndirectCommandBuffer` + `executeCommandsInBuffer`), **not** reusing an
     `MTL4CommandBuffer` object. True CB object reuse is **impossible** on this SDK
     (`beginCommandBufferWithAllocator:` always re-records). Inference ICB is OK
     only after a measured A/B vs live encode on mini → E4B. Prototype scaffolding
     lives in [`tessl::cb_replay`](../../tessl/src/cb_replay.rs)
     (`PingPongCbReplay`, `IcbReplayStub`, `ArgTableSlotPlan`,
     `survey_cb_replay_api_gaps`) and session hooks
     [`GpuDecodeSession::pos_buf`](src/gpu_model.rs) /
     `encode_once_scaffold`. Opt-in **`GEMMA_METAL_ENCODE_ONCE=1`** (default OFF)
     probes `try_replay_ready` + advances the ping-pong ledger via
     `mark_live_step`. **With `GEMMA_METAL_DECODE_ICB=1`**, mini **and E4B Hot**
     wire a layer DecodeIcb (`decode_icb_graph_eligible`); default replay is
     **frozen-tape direct-dispatch** (`DecodeIcb::execute` + scalar-write tape;
     skips binder-nop layer loop). Opt-out live encode:
     `GEMMA_METAL_ICB_TAPE_EXECUTE=0`. Opt-out skip-nop:
     `GEMMA_METAL_ICB_SKIP_NOP_LOOP=0` (falls back to binder-nop layer loop).
     Opt-out prebuilt tables: `GEMMA_METAL_ICB_PREBUILT_TABLES=0` (sticky
     setAddress fallback). Opt-in freeze+range-batch:
     `GEMMA_METAL_ICB_FREEZE_BINDS=1` + `GEMMA_METAL_ICB_RANGE_BATCH=1`.
     **31B** still live-encodes until its graph migrates.
  3. **`METAL_RUNTIME_MID_COMMIT` ≠ encode-once.** Mid-commit only overlaps host
     encode with GPU drain of the *previous* chunk (dual allocator). It does not
     remove per-token host encode (~2.5 ms) or argument-table traffic.
- **Evidence policy for the artifact names below:** `_latest.json` is a
  continuity slot, not proof that a result is current. Apply
  `docs/dev.md#benchmark-artifact-writes`; `diagnostic_only` and
  `historical_unverified` records are not current performance evidence.
- **What works today (v0.5.9 cut, 2026-07-19):**
  - GPU-resident `seed_tok` / `argmax_tok` (D13) and `pos_buf` (`u32×1`) written
    **once per step** (not every layer).
  - `rms_qkv_rope_posbuf` binds `pos_buf` instead of const-arena `set_u32` for RoPE;
    float unit parity `rms_qkv_rope_posbuf_matches_const`.
  - Ping-pong record/commit/reuse **state machine** + `mark_live_step` /
    `mark_replay_step` under `GEMMA_METAL_ENCODE_ONCE=1`; mini token parity
    `encode_once_mini_parity`; mini encode-µs A/B `encode_once_mini_encode_ab`.
  - **ICB one-kernel mini smoke SHIPPED:** `icb_mini_copy_smoke` PASS
    (`InheritArgTable`). Opt-in `GEMMA_METAL_ICB_SMOKE` / `METAL_RUNTIME_ICB_SMOKE`
    (default OFF).
  - **A0 stable scalars SHIPPED:** `IcbScalarPool` (`softcap` + `u32s` + `f32s`) with
    `reset_step` at decode `step_inner`; FA dims/scale/pos offsets, kv_store /
    densify / `rms_qkv_rope_kv_store` dst offset, and softcap binds use stable
    `GpuBuffer`s (off const-arena).
  - **Bind-tax cut SHIPPED (2026-07-19):** residual GEMV / MLP / PLE / RMS /
    `kv_store` / gate-up / QKV fused dims moved onto `IcbScalarPool`
    (`push_*` before `with_binder` so binder-nop still refreshes). Mini capture
    tape: **`immediate=0`**, `buf≈358`. Immediate Hot-materialization removed;
    any residual Immediate rebinds via `bind_bytes` at execute.
  - **Mini layer-graph DecodeIcb SHIPPED (wired):** Binder capture tape →
    `DecodeIcb::from_commands` on first **head** mini step (layers only; lm_head
    stays live). Measured: cmds≈71; `live_encodes=1`; `icb_replays≥8`. Opt-in
    **`GEMMA_METAL_DECODE_ICB=1`** + **`GEMMA_METAL_ENCODE_ONCE=1`** (default OFF).
    Artifacts: `encode_once_mini_ab_latest.json`,
    `decode_icb_mini_token_parity_latest.json`.
  - **E4B Hot DecodeIcb SHIPPED (v0.5.3):** eligibility lifted beyond
    `is_synthetic_mini()` via `is_hot_e4b` / `decode_icb_graph_eligible`
    (2560×42; not 31B). Same capture→from_commands+tape_execute path; flags
    still default OFF. Gated smoke `decode_icb_e4b_hot_smoke` **PASS**
    (cmds=594; `live_encodes=1`; `icb_replays=4`; scalar_ops=2319; skips if
    cache missing or free+purgeable RAM ≲ 8 GiB). Artifact:
    `decode_icb_e4b_hot_smoke_latest.json`.
  - **Light E4B encode A/B SHIPPED (v0.5.3 measure; v0.5.5 Immediate-zero):**
    `encode_once_e4b_hot_encode_ab` — sequential single Hot loads (no dual
    residency / no 31B / no fusion_ab TRACE); warmup=1 + iters=4; skips on
    missing cache, high `memory_pressure`, free+purgeable ≲ 8 GiB, or
    competing Metal jobs. **v0.5.5:** encode_us **67195 → 54101 µs/tok**
    (ratio **~0.81**); tok/s **14.88 → 18.48**; cmds=594; `live_encodes=1`;
    `icb_replays=4`; prebuilt=594; setArgTable=594; elided=5003;
    last_setAddress **0/5003** (was 42; post_ff `copy_f32` Immediate →
    `copy_f32_n` / IcbScalarPool Buf). Prior v0.5.4: ratio ~0.80;
    setAddress 42/5003. Artifact: `encode_once_e4b_hot_ab_latest.json`.
    Full E4B token-parity still not claimed (would need dual Hot or careful
    replay compare).
  - **A2 E4B Immediate residual cleared (2026-07-19, v0.5.5):** E4B
    post_ff_norm residual-replace used inline `copy_f32` + const-arena
    `set_u32(hidden)` → **1 Immediate bind/layer × 42**. Migrated to
    `copy_f32_n` (IcbScalarPool). Also pool-backed: `copy_f32_to_offset`,
    `copy_f32_range`, `scale_f32_inplace`. Mini parity PASS
    (`decode_icb_mini_token_parity`); E4B last_setAddress **0/5003**.
  - **Densify shape SHIPPED (2026-07-19):** sliding rings **always** densify;
    grid = `capacity * n_slot` (kernel clips via `filled`). Pool u32 cursor
    **stable at 147** across wrap (was 139→147).
  - **Q4 fuse_bf16→gemv_q4 cast SHIPPED (2026-07-19):** mini banks are
    `QuantScheme::Q4` (classic `gemv_q4` reads `float *x`). `fuse_bf16` producers
    wrote bf16 into mid/act; `gemv_bf16_x` ignored the flag and reinterpreted /
    over-read the bf16 slab as f32 — Binder-tape replay blew residual at ~cmd 19
    `add_inplace` after MLP down. Fix: `cast_bf16_to_f32` into `act_f32_scratch`
    before `gemv_q4` when `x_is_bf16`. Capture tape grows (~52→~71 cmds).
  - **Mini token parity PASS (tape_execute):** `live_out == icb_out`. Default
    replay = scalar-write tape + frozen-tape direct-dispatch (skip-nop). Opt-out:
    **`GEMMA_METAL_ICB_TAPE_EXECUTE=0`** → `note_layer_live_replay`;
    **`GEMMA_METAL_ICB_SKIP_NOP_LOOP=0`** → binder-nop layer loop.
  - **A2 bind-tax cut SHIPPED (2026-07-19, v0.5):** inherit still blocked
    (residual no-op; keep opt-in). Landed: packed tape `with_binder` + cached
    `gpu_addr`; atomic `IcbScalarPool` cursors. Ratio was ~0.77.
    - **Reverted 2026-08-31: the binder-nop PSO stand-in.** Under binder-nop,
      `GpuRuntime::pipeline` returned an arbitrary cached PSO — HashMap
      iteration order — instead of resolving the requested kernel, so
      `pipeline("a_kernel_that_does_not_exist")` returned `Ok`, and any caller
      reading `threadExecutionWidth` off the handle got another kernel's
      geometry. Suppressing *encoding* must not also suppress *name
      resolution*: a typo'd or deleted kernel has to fail. Removing it cost
      nothing measurable — a cache hit is one uncontended lock, which live
      encode already pays per dispatch.
  - **A2 residual SHIPPED (2026-07-19, v0.5.1):** `IcbScalarWriteTape` captured
    with the layer-graph (const pushes + `IcbDynSrc` + KV host commits). Mini
    DecodeIcb replay **skips the binder-nop layer loop** — applies the push
    program + commits, then `DecodeIcb::execute`. Measured
    `encode_once_mini_encode_ab`: ratio **~0.71–0.73** (prior ~0.77);
    `decode_icb_mini_token_parity` PASS cmds=71; scalar tape ops≈191.
  - **A2 sticky setAddress SHIPPED (2026-07-19, v0.5.2):** tape execute skips
    `MTL4ArgumentTable::setAddress` when the slot already holds the same
    `gpu_addr`; `setArgumentTable` latched once per binder scope. Measured
    (pre-v0.5.4): ratio **~0.58**; sticky skip **17/427**; last_exec
    setAddress **410/427**. Fallback when prebuilt tables opted out.
  - **A2 prebuilt arg-tables SHIPPED (2026-07-19, v0.5.4):** freeze Buf
    `gpu_addr`s into one `MTL4ArgumentTable` per tape command at capture;
    execute **`setArgumentTable` × cmds** (Immediate `setAddress` cleared on
    mini+E4B as of v0.5.5). Measured
    `encode_once_mini_encode_ab`: ratio **~0.44** (was ~0.58); prebuilt=71;
    setArgTable=71; elided=427; last_setAddress **0/427**;
    `decode_icb_mini_token_parity` PASS. Opt-out
    `GEMMA_METAL_ICB_PREBUILT_TABLES=0`. Artifact
    `encode_once_mini_ab_latest.json`.
  - **`execute_icb` inherit** remains opt-in `GEMMA_METAL_ICB_EXECUTE=1` (needs
    ICB-capable pipelines). Default tape path uses direct `dispatch` + prebuilt
    tables. **Freeze-binds (v0.5.7)** is the honest true-ICB path (not inherit).
  - **Freeze-binds SHIPPED (2026-07-19, v0.5.7):** opt-in
    `GEMMA_METAL_ICB_FREEZE_BINDS=1` (default OFF). ICB built with
    `inheritBuffers=false` + classic `setKernelBuffer` + **tg_mem freeze**
    (`setThreadgroupMemoryLength` on each cmd — without this, GEMV collapses).
    Execute: parent PSO + `execute_icb` × cmds (no arg-table latch —
    `setArgumentTable` overrides frozen kernel binds on MTL4),
    **setArgumentTable=0**, setAddress=0.
    Mini `decode_icb_mini_token_parity` **PASS**; copy-chain unit
    `decode_icb_freeze_binds_zero_arg_table` **PASS**. Light E4B Hot A/B:
    setArgTable=0 but ratio **~1.03** / tok/s **21.11→20.43** (no win).
    Quiet product: shipping **23.48** vs freeze **22.63** (~−3.6%). Root
    tradeoff: `execute_icb`×N costs more than `dispatch`+`setArgumentTable`×N
    on this SDK. Also: prebuilt-table fingerprint dedup + sticky adopt (skip
    redundant `setArgumentTable` when table pointer unchanged).
  - **Range-batch SHIPPED (2026-07-19, v0.5.8):** opt-in
    `GEMMA_METAL_ICB_RANGE_BATCH=1` (default OFF; requires freeze-binds).
    Coalesce consecutive tape cmds between captured `barrier_after` markers
    into one `executeCommandsInBuffer:withRange:`. Product skip-auto capture:
    E4B barriers**=360** → execute_icb **361/594** (was 594); mini
    barriers**=61** → execute_icb **62/71**. Unit
    `decode_icb_range_batch_merges_safe_spans` **PASS** (3→1). Mini token
    parity under FREEZE+RANGE_BATCH **PASS**. Quiet E4B (same session window):
    shipping **19.56** vs freeze+range **18.61** tok/s (still not a win —
    361 `execute_icb` ranges still cost more than dispatch+prebuilt). Note:
    tests that force always-on barriers mark every cmd (`barrier_after≈cmds`)
    so range-batch is a no-op there; product hazard skip-auto is the honest
    measure. Flags stay OFF.
  - **Coarse-ranges SHIPPED + measured (2026-07-19, v0.5.9):** with
    range-batch, elide non-interfering `barrier_after` (large-Buf disjoint;
    ≤64B scalars + ambient ≥80% pools ignored). Unit
    `decode_icb_coarse_ranges_elides_disjoint_keeps_raw` **PASS**. Mini
    FREEZE+RANGE parity **PASS**. Product E4B hazard tape:
    **coarse_elided=0** (every captured barrier touches a live RAW act/weight
    edge) → execute_icb still **361/594**. Quiet: shipping **20.57** vs
    freeze+range+coarse **19.85** (still not a win). Opt-out
    `GEMMA_METAL_ICB_COARSE_RANGES=0`.
  - Gap survey: `IcbDecodeGraphNotMigrated` = **31B**;
    `IcbClassicBindsVsMtl4ArgumentTable` — **PARK true-ICB** (freeze/range/
    coarse) vs prebuilt: parity OK, product tok/s loses; prefer prebuilt.
- **What still blocks full decode encode-once:**
  1. **`MTL4CommandBuffer` object reuse is impossible** — mechanism is ICB only.
  2. **Per-token host traffic (default)** — mini pays **setArgumentTable × cmds**
     (71); E4B pays **setArgumentTable × 594**. Freeze zeroes table switches;
     range-batch cuts execute_icb **594→361** but still loses vs prebuilt;
     coarse cannot cut further on product hazard tape (elided=0).
  3. **True-ICB PARKED for product** — mini parity PASS; E4B quiet loses;
     keep freeze/range/coarse default OFF; encode-once honesty path = prebuilt.
  4. **31B decode graph not migrated** (`IcbDecodeGraphNotMigrated`) — E4B Hot
     wiring landed; 31B stays live-encode (jetsam / graph size).
- **Overturn if:** a cheaper ICB execute wins ≥ encode-once target on E4B
  without parity loss (do not wait on a fictional MTL4 CB-reuse API), **or**
  shipping hazard barriers drop enough that range spans beat setArgTable×N.
- **Quiet E4B product tok/s A/B (2026-07-19):** prior regress shipping
  **20.69 → 16.11** tok/s under `ENCODE_ONCE`+`DECODE_ICB` (encode_us still
  won). **Root cause:** `DecodeIcb::execute` forced always-on Device barriers
  for every tape cmd (~594) while shipping hazard skip-auto uses ~364 selective
  RAW barriers — TRACE host rollup bar **364 → 599**, host encode **245 → 171**
  µs. **Fix (v0.5.6):** capture `barrier_after` from auto + explicit
  `Binder::barrier`; execute skip-auto and replay markers only. Re-measure:
  shipping **23.50** vs encode-once **23.38** tok/s (~flat); TRACE bar_med
  **365**; encode_us still lower. Flags stay default OFF (no clear product win
  to default-on). Artifact `encode_once_e4b_toks_latest.json`.
- **Freeze-binds product A/B (v0.5.7):** setArgTable **0**/594; mini parity
  PASS; quiet E4B **23.48 → 22.63** (not a win). Artifact
  `encode_once_e4b_toks_20260719T233418Z_*` + `encode_once_e4b_hot_ab_latest.json`
  (freeze arm).
- **Range-batch product A/B (v0.5.8):** execute_icb **361/594** (barriers=360);
  mini parity PASS; quiet E4B **19.56 → 18.61** (not a win; machine cooler than
  v0.5.7 window). Artifact `encode_once_e4b_toks_20260719T234357Z_*`.
- **Coarse-ranges + PARK (v0.5.9):** product E4B coarse_elided=**0**;
  execute_icb **361/594**; quiet **20.57 → 19.85**. **Recommendation:** park
  true-ICB (freeze/range/coarse) vs prebuilt; keep encode-once on prebuilt
  tape for ~flat tok/s; chase GPU dispatch count next. Artifact
  `encode_once_e4b_toks_20260720T001404Z_*`.
- **Quiet re-measure post bank-split (2026-07-20):** shipping **23.56** vs
  prebuilt `ENCODE_ONCE`+`DECODE_ICB` **23.46** (~flat). Still no product win.
  Artifact `encode_once_e4b_toks_20260720T013526Z`. Flags stay OFF.
- **Next in lane:** park encode-once default-on; 31B rope when free
  (serialize); or host dispatch-tax outside Metal fusion. Fusion exactness
  green (D15) but tok/s flat — keep `FUSE_QKV` OFF. No `FUSE_GATE_DOWN` (D18).

---

## D17. Persistent interpreter = mini prototype only (grid-sync stand-in)

- **Decision:** Prototype Hazy-style **instruction-stream + atomic-deps** kernels on
  the mini graph for edges that need grid-wide sync (`gate/up → down`, `FA → o_proj`).
  Opt-in `GEMMA_METAL_PERSISTENT_INTERP=1` (default **OFF**). Two sibling stand-ins
  (dense f32, not Q4 Hot):
  - `persistent_interp_gate_down`: `PRODUCE_MID → BARRIER → DOWN_PROJ → HALT`
  - `persistent_interp_fa_o_proj`: `PRODUCE_CTX → BARRIER → O_PROJ → HALT`
    (`ctx[i]=tanh(q·k·scale)·v` mock FA — not softmax; proves the sync edge only)
  **Mini decode wired:** when flag on and `GpuSynthModel::is_synthetic_mini()`,
  `step_inner` dispatches both stand-ins once per layer on dedicated dense scratch
  (shipping Q4 FA/MLP unchanged — stand-in math ≠ Hot). Hot/E4B/31B: hook **no-ops**
  even if the env flag is set. No E4B/31B enablement of real edge replacement.
- **Why:** Metal has **no grid-wide forward-progress guarantee** — a literal single
  persistent megakernel can deadlock if consumer TGs spin while producers are not
  resident. Device atomics also only expose `memory_order_relaxed` (no
  acquire/release) — another reason this stays mini-only. **Confirmed on Hot Q4
  E4B dims (D18):** two-dispatch PRODUCE|host-sync|DOWN is bit-exact vs shipping;
  single-dispatch `PRODUCE→atomic BARRIER→DOWN` leaves host-visible `mid` exact
  but in-kernel DOWN reads stale mid (resid blows up). Dense mini stand-in can
  still pass at the same shape — do not treat dense parity as Q4 Hot proof.
  v1 fusions stay element-local; this lane explores the sync pattern safely on
  tiny TG counts (`n_tg ≤ 8`) with spin timeout + `fail` flag.
- **Gate:** float parity vs unfused produce + dense GEMV
  (`persistent_interp_gate_down_matches_unfused`, `_mini_dims`;
  `persistent_interp_fa_o_proj_matches_unfused`, `_mini_dims`); flag-off rejects
  dispatch; mini session hook (`persistent_interp_mini_decode_hook`) — hits fire,
  barrier `fail==0`, tokens ≡ baseline. Serialize Metal tests (`--test-threads=1`).
  No `fusion_ab` / no 31B load.
- **Status (2026-07-19):** both doctrine edges on mini + float parity + mini
  `step_inner` hook. Production 31B readiness: **not claimed**.
- **Overturn if:** barrier spin timeouts on mini, or OS gains a cooperative-grid /
  forward-progress API that changes the Metal megakernel calculus.

---

## D18. Hot bounded-TG gate→down (E4B opt-in; overturns D17 “never E4B” for this path only)

- **Decision:** Ship **`persistent_interp_gate_down_q4`** — same instruction stream as
  the D17 dense mini stand-in (`PRODUCE_MID → BARRIER → DOWN_PROJ → HALT`), but
  PRODUCE peels **`gemv_q4_mlx_simd_gate_up_gelu`** math and DOWN peels
  **`gemv_q4_mlx_simd_add`** into `x`, with **outer row tiles** inside **`n_tg ≤ 8`**
  (not the full 1280-TG E4B GEMV grid). Opt-in **`GEMMA_METAL_FUSE_GATE_DOWN=1`**
  (default **OFF**), **separate** from `GEMMA_METAL_PERSISTENT_INTERP`.
- **Wiring:** when flag on + `HotQuantBanks::can_fuse_gate_down` (RowMajor simd Q4,
  `fuse_bf16_mlp`, gate/up/down layout match), `GpuDecodeSession::step_inner`
  replaces shipping **`gate_up_gelu` + `gemv_add_into`** with one bounded-TG dispatch.
  D17 mini dense scratch hook is unchanged. **`is_synthetic_mini()` no-op lifted only
  for this Hot replacement** — real E4B decode may opt in; 31B still off unless layouts
  match and ops accept Metal FP risk.
- **Fail semantics:** barrier spin sets device `fail`; decode **`step_inner` hard-errors**
  on `fail≠0`. Bench may fall back to unfused MLP when flag off or dispatch fails.
- **Gate:** `persistent_interp_gate_down_e4b_dims_stress` (dense stand-in 10240→2560,
  `n_tg=8`, `fail==0`, CPU parity); `persistent_interp_gate_down_q4_matches_mlp_fuse`
  (Q4 vs shipping fuse path); `fuse_gate_down_flag_default_off`. Serialize Metal
  (`--test-threads=1`). **`fusion_ab`:** expect Δ≈−42 dispatches/token vs unfused
  gate+down at HAZARD=0 (informational); skip heavy E4B bench if machine risk.
- **Status (2026-07-19):** kernel + Rust dispatch + opt-in decode wiring **landed**;
  dense E4B-dim stress + small Q4 parity **PASS**
  (`persistent_interp_gate_down_e4b_dims_stress`,
  `persistent_interp_gate_down_q4_matches_mlp_fuse`). Light E4B `fusion_ab`
  (`FUSION_AB_ARMS=baseline,gate_down`): dispatch **Δ=−42.0** (599.9→557.9)
  **PASS** engagement; **exactness FAIL** @ HAZARD=0; tok/s **20.87→6.37**
  (0.305×; also pays host `synchronize()` per layer). Artifact
  `fusion_ab_e4b_gate_down_latest.json`. **Not production-default.**
- **Exactness root cause (diagnosed):** not peel math / not residual combine /
  not mid dtype after dual-bind fix. At E4B Q4 dims (10240→2560, `n_tg=8`):
  - **Two-dispatch** PRODUCE|`synchronize`|DOWN: `split_err=0`, peel ≡ shipping.
  - **Single-dispatch** PRODUCE→BARRIER→DOWN: post-kernel `mid_err=0` and
    shipping-down-on-fused-mid `err=0`, but fused resid **~1e9** — in-kernel
    DOWN saw stale `mid` (D17 relaxed device atomics; no acquire/release).
  - Test lock: `persistent_interp_gate_down_q4_e4b_dims_visibility_blocker`.
  - Secondary fixes landed: single mid buffer bind (no float*/bfloat* alias UB);
    `can_fuse_gate_down` **RowMajor-only** (peel has no I4 twin).
- **Hard blocker:** one-dispatch Hot gate→down cannot be exact on Metal without
  a real cross-TG memory order for `mid[]`. Two-dispatch split has **no dispatch
  win** vs shipping `gate_up_gelu` + `gemv_add_into`. **Keep
  `GEMMA_METAL_FUSE_GATE_DOWN` default OFF**; do not product-default.
- **Overturn if:** Metal gains device acquire/release (or another coherent
  publish for non-atomic mid), or a one-dispatch design proves bit-exact under
  `persistent_interp_gate_down_q4_e4b_dims_visibility_blocker` + E4B `fusion_ab`.
- **Next:** leave flag OFF; lane can park gate→down or explore non-PI dispatch
  cuts. Optional: drop per-layer `synchronize()` if a future path needs fail
  checks without tok/s collapse.

---

## D19. Sustained-load gate: throughput decides the verdict, the OS decides the blame

- **Decision:** `src/thermal.rs` gates a continuous-decode run on **three**
  statistics and reports a **four-valued** verdict (`Pass`/`Fail`/`Skipped`/
  `Error`) whose `Skipped` and `Error` arms are never readable as a pass
  (`ThermalReport::passed()` is true for `Pass` only; `bench --thermal` exits `1`
  on FAIL and `2` on no-verdict).
  - `sustained_ratio` = median(last third)/median(first third) — **directional**.
  - `rolling_ratio` = min(2-window mean)/**median**(2-window mean) — **sag**.
  - `cv` = sd/mean over windows — **dispersion**.
  - Order is asymmetric on purpose: a depressed tail FAILs even on a noisy run
    (noise is not an alibi for a real decline); a *stable* tail on a noisy run is
    SKIPPED, never PASS.
- **Decision:** `NSProcessInfo.thermalState` / `isLowPowerModeEnabled`
  (`objc2-foundation`, now a direct dependency) supply **attribution only** —
  they never move the verdict. Throughput decides *whether* it slowed; the OS
  reading decides *what to blame*. `NSProcessInfo` is coarse and lags, so letting
  it veto a measured decline would trade a false alarm for a missed regression.
  `HostThermalState::Unknown` counts as pressure: a level Apple adds later is far
  more likely above `Critical` than below `Nominal`.
- **Why:** the first version gated on `tail/head` plus raw `worst/best` and
  failed the mini graph at 0.89. It was not throttling — a build and two editors
  were saturating the CPU, and the mini graph is dispatch-bound, so window rates
  wandered 24–50 tok/s with no trend. A gate that calls host contention a thermal
  failure gets ignored; one that passes a real throttle is worse.
- **Three defects the measurements exposed, in order:**
  1. Raw `worst/best` fails on a single stall → replaced by a **two-window
     rolling** mean.
  2. `min/max` of those means compounds two order statistics, so it drifts down
     as the window count rises: the 300 s run scored 0.691 against 0.905–0.927
     for 11-window runs of the same material. → divide by the **median**.
  3. Attribution derived from the gate predicates contradicted verdicts that
     short-circuited before those arms ran ("this decline is NOT thermal" printed
     under a SKIPPED). → attribution is derived from the **decided verdict** via
     a three-valued `DeclineFinding` (`Declined`/`Held`/`Inconclusive`).
- **Calibration, not guesswork:** `max_cv = 0.09` sits in the measured gap
  between quiet runs (cv 0.017–0.075) and contended runs (0.100–0.163);
  `min_rolling_ratio = 0.85` between clean runs (0.867–0.974) and genuine sags
  (≤ 0.730).
- **Why the work unit is `bench_decode_sustained`, not `bench_decode_tok_s`:**
  the burst bench resets and re-prefills on **every** call, so a short call is
  mostly prefill and the window rate swings with how many calls land inside a
  window (measured: 3× spread, windows of 72/48/48/32/40 tokens). Continuing an
  already-running session pays prefill once per `kv_capacity` tokens instead,
  which cut the spread to ~1.18×. A full global KV slot is a **capacity limit,
  not a fault** — the session recycles rather than surfacing
  `Error::Kv("GPU KV full")` mid-run.
- **Evidence:** `docs/gates.md` § Sustained-load gate; artifacts
  `bench/results/thermal_gate_{e4b,mini}_*.json`. **No leak** (13 224 tokens /
  ~220 recycles / 292 s moved RSS 0.06 MiB). **This host cannot certify a
  sustained run** — a persistent ~100 %-CPU process drifts on a multi-minute
  cycle, and six 60 s E4B runs ranged 0.86–1.08 sustained at within-run cv
  0.066–0.075. Every non-PASS is attributed non-thermal by the OS reading.
  Whether the M5 Pro throttles under sustained decode is **unestablished** and
  needs a quiet host.
- **Overturn if:** a per-process GPU-time signal lands that can attribute noise
  to a specific competitor, at which point the dispersion arm can become a
  precondition check instead of a verdict.
