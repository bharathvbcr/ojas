# gemma-metal living gates

Phase 0 baseline ladder + locked product targets for the custom Metal stack.
Update measurement tables when re-running [`../bench/`](../bench/). Sibling docs:
[architecture](architecture.md) · [dev](dev.md) · [research](research.md) ·
[DECISIONS](../DECISIONS.md) · [crate README](../README.md).

**Host:** Apple M5 Pro · 20 GPU cores · 64 GB unified · macOS 26.x  
**Lane doctrine:** honest INT4 (+ optional MTP). No SWA-shrink / quality burn until parity is green.

**Custom stack status (2026-07-13 evening):** Greedy decode **finite** on E4B. Metal
`mlp_gelu_tanh` fixed (`precise::tanh` — was `fast_tanh` NaNs on large gate); host gelu
removed; **`GEMMA_METAL_FUSE_MLP` default ON**; hazard barriers default ON. Quiet E4B
**~23.6 tok/s** (was ~14.6 with host gelu tax; prior peak ~25). Still below gates ≥48–60.
See [`dflash_port.md`](dflash_port.md).

---

## Locked honest-lane targets

Product gates from the plan (do not loosen). Phase-0 measurements calibrate the
*live floor*; ship claims must clear these targets **and** beat Phase-0 best where noted.

| Gate | Target | Notes |
|------|--------|-------|
| **E4B Q4 decode** | **~48–60 tok/s** | Honest lane planning band; **Phase-0 best on this host is higher (~76 mlx-lm)** — treat ≥ Phase-0 best as the practical bar |
| **E4B Q4 + MTP** | **~90–110 tok/s** (~1.7×) | LiteRT-class accept on structured prompts |
| **31B Q4 decode** | **≥ 15 tok/s** | Beat local Ollama floor (~12 tok/s pinned) |
| **31B Q4 + MTP** | **≥ 25 tok/s** | Coding/math accept |
| Quality | HF/MLX logit parity; **no** `w188`-class SWA shrink until parity green | Challenge *honest* lane |
| Buffer | No single Metal buffer **> 4 GB** (PLE per-layer split on E4B) | MLX Metal limit |

**Frontier (non-product):** challenge ~510 TPS on A10G is a PPL-budget burn — do not use as a Mac target.

### Phase-0 best (this host) — practical bar

| Model | Best measured decode | Source |
|-------|----------------------|-------|
| E4B Q4 | **~75.7 tok/s** | `mlx_lm.benchmark` · `mlx-community/gemma-4-e4b-it-4bit` |
| 31B (nvfp4 via Ollama) | **~12.3 tok/s** | Ollama `gemma4:31b-mlx` decode_pad @ 128 tok |
| 31B plain mlx-lm 4bit | **~12.7 tok/s** | `mlx-community/gemma-4-31b-it-4bit` (bench harness) |
| **31B + DFlash spec-decode** | **~31 tok/s median** (2.5×); range **19–36** across prompt types (37 peak on code/json, 19 on creative prose) | **mlx 0.32.0** + `z-lab/gemma-4-31B-it-DFlash` 4-bit draft, **block=5** · `bench/dflash_fast_31b.py` · serve `bench/serve_dflash.py` · via `~/.venvs/dflash32` · **clears ≥15 on all prompt types; ≥25 on all but creative prose** |

**DFlash note (2026-07-13):** exact verify → output identical to greedy 31B (no quality burn;
honest-lane compatible). mlx **0.32.0 is required for the speed** (M5 GPU Neural Accelerators
make the M=8 block-verify quantized GEMMs 1.5–2× faster; 0.31.2 gives only ~18.6). dflash's
`[mlx]` extra pins mlx==0.31.2 — install base dflash + mlx separately. Artifacts:
`bench/results/mlx032_nax_ab_31b.json`, `dflash_*.json`. This is the native block-verify
target for gemma-metal MTP (supersedes clustered-assistant per-token verify).

---

## Pins (all Phase-0 runs)

| Knob | Value |
|------|-------|
| KV / context (`num_ctx` / `max_kv_size`) | **4096** |
| Max generation tokens | **128** (decode pad / mlx benchmark) |
| Temperature | **0** (greedy) |
| Thinking / reasoning stream | **off** (`think:false` on Ollama) |
| Batch | **1** |

---

## Runtime availability (this machine, 2026-07-13)

| Runtime | Status | Notes |
|---------|--------|-------|
| **Ollama** `gemma4:31b-mlx` | **Present** | 31.3B, nvfp4, MLX via Ollama |
| **Ollama** `gemma4:e4b-it-q4_K_M` | **Present** (pulled for Phase 0) | ~9.6 GB Q4_K_M |
| **mlx-lm** | **Present** | CLI + `mlx_lm.benchmark`; E4B 4bit cached |
| **mlx-lm 31B** | **Cached** | `mlx-community/gemma-4-31b-it-4bit` ~17 GB / 4 shards complete (auth `hf download`) |
| **LiteRT-LM** | **Not installed** | `litert-lm` / `litert_lm` not on PATH; no support dir found |
| **BaseRT** | **Not installed** | No `basert` / `libbaseRT.dylib`; publishes E2B / 26B-A4B only (not E4B/31B) |
| Ollama MTP | **Not available** | No draft/speculative API knobs for these tags |
| mlx-lm MTP | **Superseded by DFlash** | `z-lab/gemma-4-31B-it-DFlash` cached (~3 GB, q4 at runtime); accept 3.8–8.5 tok/verify; 31B **~27.8 tok/s** on mlx 0.32 (`~/.venvs/dflash32`) |
| **gemma-metal** | **In tree** | E4B greedy **finite** ~**23.6 tok/s** quiet (gelu `precise::tanh` + fuse MLP + hazard ON; was ~14.6 with host gelu); MTP/D-Flash bring-up — see `dflash_port.md` |

---

## Phase-0 measurements

### Historical floor (prior Benchmark tool)

| Runtime | Model | Decode tok/s | TTFT | Source |
|---------|-------|--------------|------|--------|
| Ollama | `gemma4:31b-mlx` | **~10.8–13.0** (typ. **~11**) | ~6.7 s (cold-ish) | `/Users/bharath/Code/Benchmark/results/` 2026-07-07/08 |

### Fresh Phase 0 runs (pinned ctx=4096, max_gen=128, temp=0, think=off)

| Date (UTC) | Runtime | Model | Lane | Prefill tok/s | Decode tok/s | TTFT ms | Mem | Artifact |
|------------|---------|-------|------|---------------|--------------|---------|-----|----------|
| 2026-07-13 | Ollama | `gemma4:31b-mlx` | honest | ~216 (decode_pad) | **12.27** (decode_pad); 9.84 (math, n=3) | **426** (warm) | RSS ~5.3 GB (ollama procs; under-counts unified) | `bench/results/run_20260713_100106_ollama.json` |
| 2026-07-13 | Ollama | `gemma4:e4b-it-q4_K_M` | honest | ~377 (decode_pad) | **55.79** (decode_pad); 71.06 (math, n=4) | **331** | RSS ~10.2 GB (ollama procs) | `bench/results/run_20260713_100745_ollama.json` |
| 2026-07-13 | mlx-lm | `mlx-community/gemma-4-e4b-it-4bit` | honest | **1966.9** (bench avg) | **75.72** (bench avg); 76.09 / 76.25 (generate) | n/a (bench); wall generate ~5 s incl. load | **peak 4.46 GB** (mlx reported) | `bench/results/run_20260713_100411_mlx.json` + `mlx_e4b_benchmark_raw.txt` |
| — | LiteRT-LM | E4B ±MTP | — | — | **not run** | — | — | runtime missing |
| — | BaseRT | — | — | — | **not run** | — | — | runtime missing |
| — | any | E4B/31B + MTP | mtp | — | **not run** | — | — | no real draft weights |

**Headline takeaways**

- **31B floor:** ~**12.3 tok/s** warm decode (Ollama MLX nvfp4) — aligns with historical ~11; ship gate **≥15** is a real stretch vs this backend.
- **E4B honest lane:** Ollama Q4_K_M ~**56 tok/s** (in the 48–60 band); mlx-lm 4bit ~**76 tok/s** (Phase-0 best).
- **E4B +MTP / LiteRT:** cannot calibrate until LiteRT-LM or a draft model is installed.

---

## Phase 4 — gemma-metal custom stack (honest lane)

| Date (UTC) | What | Decode tok/s | TTFT | Notes |
|------------|------|--------------|------|-------|
| 2026-07-14 | **Real E4B Q4** (bfloat2 Hot sb + qmv_fast qdot/pointer-walk) | **23.86** | **142 ms** (T=4) | TRACE/INFER off; Interleaved4 A/B ~22.8 (default off). Artifact: `bench_e4b_bfloat2_final.txt` / `latest_e4b_gemma_metal.json` |
| 2026-07-13 | **Real E4B Q4** (gelu `precise::tanh`; fuse MLP + hazard ON; host gelu removed) | **23.61** | **141 ms** (T=4) | TRACE/INFER off. Was ~14.6 with host gelu sync. Root cause: `air.fast_tanh` NaNs on gelu inner≈301. Artifact: `run_e4b_gemma_metal_1783994131.json` era + follow-up quiet rebench |
| 2026-07-13 | **Real E4B Q4** (MLX qdot peel; mid-BW chase) | **24.45** (range ~24.5–25.4) | **136 ms** (T=4) | TRACE mid MLP ~~41 GB/s vs lm_head ~~240; TG x-cache no-win/reverted. Artifact: `bench_e4b_qdot_final.txt` |
| 2026-07-13 | **Real E4B Q4** (bf16 activations on simd GEMV) | **24.98** | **133 ms** (T=4) | TRACE/INFER off; cast once/phase; Artifact: `bench_e4b_act_bf16.txt` / `latest_e4b_gemma_metal.json` |
| 2026-07-13 | **Real E4B Q4** (fuse K∥V + phase-coarse barriers) | **25.05** | **132 ms** (T=4) | TRACE/INFER off; `GEMMA_METAL_FUSE_KV` default on; Artifact: `bench_e4b_qkv_fuse_coarse.txt` / `run_e4b_gemma_metal_1783981846.json` |
| 2026-07-13 | **Real E4B Q4** (bf16 scales + packs=2 qmv, SIMD_ROWS=4) | **25.10** | **133 ms** (T=4) | TRACE/INFER off; fuse MLP+KV; Artifact: `bench_e4b_rows4.txt` / `latest_e4b_gemma_metal.json` |
| 2026-07-13 | **Real E4B Q4** (simd fuse gate_up + resid add) | **24.23** | **139 ms** (T=4) | TRACE off; ~611 disp / ~321 barriers; Artifact: `run_e4b_gemma_metal_1783979793.json` |
| 2026-07-13 | **Real E4B Q4** (simdgroup Q4 GEMV, SIMD_ROWS=8) | **21.27** | **159 ms** (T=4) | TRACE off; BlockedBn off (simd path); PLE residual on. Artifact: `bench/results/bench_e4b_simd_final.txt` / `latest_e4b_gemma_metal.json` |
| 2026-07-13 | **Real E4B Q4** (uint peel + hazard + PLE Q4 Hot) | **19.38** | **179 ms** (T=4) | TRACE off; PLE residual on (gate/proj skip). Artifact: `bench/results/run_e4b_gemma_metal_1783975501.json` / `latest_e4b_gemma_metal.json` |
| 2026-07-13 | **Real E4B Q4** (TG x-cache GEMV + fused softcap argmax) | **15.17** | **243 ms** (T=4) | RSS ≈9.5 GB (`time -l`); softcap≈229 µs; GPU contended (train + Q4 metallib XPC). Artifact: `bench/results/bench_e4b_gemv2_final.txt` |
| 2026-07-13 | **Real E4B Q4** (vectorized GEMV + GPU embed) | **15.91** | **230 ms** (T=4) | 1-thread/row Q4 peel, GPU embed lookup, GPU argmax index propagate; PLE Hot skipped. Artifact: `bench/results/run_e4b_gemma_metal_1783969171.json` / `latest_e4b_gemma_metal.json` |
| 2026-07-13 | **Real E4B Q4** (post speed pass) | **13.90** | **265 ms** (T=4) | GPU-resident KV, packed async encode, tiled Q4 GEMV TG=128 |
| 2026-07-13 | **Real E4B Q4** (pre speed pass) | **4.78** | **790 ms** (T=4) | Host KV densify + per-dispatch sync |
| 2026-07-13 | **GPU Hot synthetic mini** | **~550–620** | ~9 ms (T=8) | vocab=512, hidden=256, 3 layers; **Not** product E4B |
| — | SWA shrink (`w188`) | **not run** | — | Deferred until HF/MLX parity green |

**Honest verdict:** ~**23.9 tok/s** this pass (prior peak **~25.1**) — still **~2.0× below** gate lower band (48) and **~3.2× below** Phase-0 mlx (~76). Landed interleaved **bfloat2** Hot scale+bias + true MLX **qdot**/pointer-walk; tok/s **flat**. Interleaved4 weight pack A/B **regressed** (~22.8) → default off. Interim ≥30 **not** cleared. Do **not** claim gate clearance.

**Speed work this pass**

1. Hot Q4Mlx scale+bias → interleaved **bfloat2** (one 4B load/group); embed/PLE/GEMV updated together
2. Simd path = MLX `qmv_fast` **qdot** (16^k-prescale x + ushort mask) + **pointer-walk** ws/sb/x across K-blocks
3. `*_simd_i4` + Interleaved4 upload landing; **GEMMA_METAL_GEMV_INTERLEAVE** default **OFF** (measured slower)
4. Quiet re-bench: E4B **23.86** / 31B **6.83** — no interim gate clearance
5. Next: dispatch fusion / megakernel overhead (bottleneck still ~780 launches/tok), not more ALU peel tweaks
---

## Phase 5 — MTP

| Item | Status |
|------|--------|
| Assistant presets (E4B clustered / 31B dense) | **In tree** |
| Cross-KV into target shared sliding/global | **Real** — `sync_mtp_cross_kv` densifies GPU shared → host bridge |
| Activation bridge + clustered LM head | **Real weights** (centroids/embeds/pre/post + 4 Q-consumer layers) |
| Adaptive draft + verify | **Wired** (`generate_mtp_smoke`) |
| Decode-loop draft→verify smoke | **Done** — synthetic + real-weight draft |
| Accept-rate / tok/s with real draft vs backbone | **Measured** — accept **75% (6/8)**; e2e **~10.0 tok/s** (draft+full verify; no early-reject skip). Artifact: `bench/results/mtp_e2e_accept.txt` / `latest_mtp.json` |
| E4B assistant on disk | **Present** — `google/gemma-4-E4B-it-assistant` (~160 MB) |

**Phase 5 criteria:** synthetic E2E **met**; real weights **loaded**; shared-KV draft forward **wired**; product MTP tok/s still below gate (~90–110) until verify early-exit + faster draft.

### DFlash parity gates (2026-07-14) — mini green; 31B dual-norm GEMM landing

| Item | Status |
|------|--------|
| MLX golden stream vs greedy | **PASS** (`dflash_parity_mlx_golden.json`; mean_accept≈3.0 @ bs=5) |
| Prior metal mean_accept≈3.8 | **Invalid** — was NaN-collapsed target (all 0) |
| gemma-metal mini exact / accept | **PASS** / full (MASK-steer + always-on; mini only) |
| Mini DFlash vs greedy tok/s | Steered hazard lane can meet/beat greedy; M×GEMV verify (`cols≤256`) |
| 31B greedy | **Finite** **~5.8–6.8 tok/s** unique≫1 |
| 31B DFlash (this pass) | **~1.2 tok/s**; mean_accept **≈0–0.15** (was locked 0); hazard exactness **FAIL**; AO exact PASS was vacuous (unique≤4) |
| M>1 GEMM verify | **Wired + dual-norm/`layer_scalar`** — diag `GEMMA_METAL_31B_VERIFY_DIAG=1` GEMM≡seq under always-on |
| Product ≥15 / ≥25 | **Unmet** on native; MLX `serve_dflash.py` clears |

**Root causes / fixes**

1. **Fixed** — plain-Q4 draft/`gemv_bf16_x` poison; lm_head→softcap RAW; conditioner FC deferred post-softcap.
2. **Fixed** — always-on Dispatch + MASK-steer gated to **synthetic mini only**.
3. **Fixed** — `step_verify_gemm` Gemma4 dual-norm + `layer_scalar` ×M; device capture stage (no mid-verify host sync).
4. **Fixed** — draft/conditioner re-quant **plain Q4 g64** (MLX group size; still f32 GEMV).
5. **Open** — hazard-lane 31B exactness still FAIL (near-ties); AO exact PASS was vacuous (unique=4 mode-lock — do not claim). Accept still ≪ MLX ~3 → golden intermediates next.

Artifacts: `latest_dflash_parity_gates.json`, [`audit_deep_2026-07-14.md`](audit_deep_2026-07-14.md). See [`dflash_port.md`](dflash_port.md).

---

## Phase 6 — 31B + serve

| Item | Status vs gates |
|------|-----------------|
| Config deltas (no PLE, no KV-share, `k_eq_v`, GQA 32/16 vs 32/4) | **Presets + JSON parse tests green** |
| Real 31B Q4 Hot load / decode | **Measured** — Hot upload ~17.87 GiB / 60 layers; decode **6.83 tok/s**; TTFT **548.8 ms**. Multi-shard + global `k_eq_v` V=K. See `bench/results/latest_31b.json` / `bench_31b_bfloat2_qmv.txt` |
| 31B decode measurement (gate doc) | Custom **6.83** vs Ollama **~12.27** vs gate **≥15** — **unmet** (~1.8× below Ollama, ~2.2× below gate) |
| 31B + MTP ≥25 tok/s | **Unmet** — pre-fix ~1.17 tok/s / accept≈0 / exact FAIL; GEMM dual-norm landing (rebench). MLX DFlash **~28–37** proves gate reachable — see `docs/audit_deep_2026-07-14.md` |
| `serve` e4b vs 31b | **Works** — `--preset 31b` Hot confirmed (`Hot Q4 loaded from mlx-community/gemma-4-31b-it-4bit`) |
| Weight sources on this host | Ollama present; mlx 31B **complete** (~17 GB) |

### 31B Hot bench (2026-07-13)

```bash
export HF_XET_HIGH_PERFORMANCE=1
hf download mlx-community/gemma-4-31b-it-4bit --max-workers 4   # done; resumes from HF cache
cd gemma-metal
cargo run --release --bin serve -- --preset 31b --port 8787
cargo run --release --bin bench -- --model "$HOME/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/$(cat ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/refs/main)"
```

| Metric | gemma-metal Hot 31B Q4 | Ollama `gemma4:31b-mlx` | Gate |
|--------|------------------------|-------------------------|------|
| Decode tok/s | **6.83** (16 steps) | **~12.27** | ≥15 |
| TTFT | **548.8 ms** (T=4) | ~426 ms (warm) | — |
| RSS / peak | footprint **~38 GiB** · peak **~55 GiB** (ps RSS under-counts) | ~5.3 GB (ollama procs) | fits 64 GB |
| Hot weights | **17.87 GiB** | nvfp4 via Ollama | — |

### 31B memory / serve plan

| Resource | Estimate | Notes |
|----------|----------|-------|
| 31B Q4 weights | **~17.9 GB** Hot | Measured; fits 64 GB unified |
| +32K KV fp16 | ~1.2 GB | Plan default |
| Hot upload peak | footprint **~38 GiB** / peak **~55 GiB** (pre-mitigation) | Host+Hot overlap during upload; host banks + unused `host_q` twin now dropped before session KV alloc |
| Serve path | `serve --preset 31b` | **Hot** when mlx cache complete |
| Gate documentation | Ollama ~12.3 · custom **6.83** | Custom ≥15 **unmet** |

**Phase 6 criteria:** document + serve + measure — **met** (download no longer blocking; speed gate still open).

---

## Sustained-load (thermal) gate

Every other number in this file is a **burst**: warm up, run 16–32 steps, divide.
That is the cold-clock rate. It says nothing about a session that runs for
minutes while the die heats and the OS walks the clocks down — which is the
shape a user actually sees, and the one a fanless or thermally-limited part
fails first. This gate closes that hole.

```bash
cd gemma-metal
cargo run --release --bin bench -- --e4b --thermal          # 60 s on real E4B
cargo run --release --bin bench -- --thermal-quick          # 24 s on the mini graph
```

**What it measures.** Continuous decode, chopped into fixed wall-clock windows.
The work unit is `GpuDecodeSession::bench_decode_sustained` — further steps on a
session that keeps running, recycling (reset + re-prefill) only when the global
KV slot would otherwise fill, so prefill is paid once per `kv_capacity` tokens
instead of once per call. Consecutive calls therefore do near-identical work, and
a change in window rate is attributable to the machine.

**Three statistics, deliberately separated.**

| Statistic | What it answers | Default |
|-----------|-----------------|---------|
| `sustained_ratio` = median(last third) / median(first third) | *Did it slow down?* Directional — scheduling noise perturbs both thirds equally | ≥ **0.90** |
| `rolling_ratio` = min(2-window mean) / **median**(2-window mean) | *Did it sag?* Two adjacent windows must dip together, so one stall cannot fail a run. Dividing by the median, not the best pair, keeps the statistic from drifting with run length | ≥ **0.85** |
| `cv` = sd / mean over windows | *Was the run clean enough to certify?* | ≤ **0.09** |
| `peak_thermal_state` — `NSProcessInfo.thermalState` | *Was it actually hot?* The one input not derived from throughput | reported |
| RSS growth, first → last window | Leak detector, free with the window series | ≤ **512 MiB** |

The `cv` ceiling is calibrated, not guessed: measured on this host, quiet runs
land at **0.017–0.075** and runs with visible host contention at **0.100–0.163**.
The default sits in that gap.

**The OS reading decides attribution, never the verdict.** Throughput decides
*whether* the part slowed; `NSProcessInfo.thermalState` (plus
`isLowPowerModeEnabled`, since Low Power Mode caps clocks and looks identical to
heat) decides *what to blame*. `NSProcessInfo` is coarse and lags, so letting it
veto a measured decline would trade a false alarm for a missed regression. Every
report carries an `attribution` line, e.g.:

> `OS reported no thermal pressure (peak fair) — this decline is NOT thermal;
> look at host contention or a code regression`

**Verdict is four-valued, and `SKIPPED` is not `PASS`.** Order matters and is
asymmetric on purpose: a depressed tail is a **FAIL** even on a noisy run (noise
is not an alibi for a real decline), but a *stable* tail on a noisy run is
**SKIPPED** — the run did not earn a clean bill of health. Exit codes:
`0` pass / not requested · `1` a gate ran and FAILED · `2` a gate was requested
but produced no verdict (no GPU, no model, too few windows, too noisy).

**Flags:** `GEMMA_METAL_BENCH_STEPS=N` sets the burst-bench step count (default
16) so the burst and sustained numbers can be compared at the same KV depth.
Gate flags: `--thermal-secs S`, `--thermal-window S`, `--thermal-warmup S`,
`--thermal-windows N`, `--thermal-ratio R`, `--thermal-rolling R`,
`--thermal-cv X` (≤0 disables), `--thermal-floor TOK_S`.
Artifacts: `bench/results/thermal_gate_{label}_{ts}.json` + `_latest.json`.

### Measured (2026-08-31, M5 Pro 20-core / 64 GB, macOS 27.0)

| Subject | Verdict | sustained | rolling | cv | OS peak | Note |
|---------|---------|-----------|---------|-----|---------|------|
| mini synth, 24 s, quiet | **PASS** | 1.022 | 0.974 | 0.017 | fair | `thermal_gate_mini_1788234882.json` |
| E4B Q4 Hot, 60 s | FAIL | 0.863 | 0.867 | 0.071 | fair | 3.7 → 3.2 tok/s; attributed **not thermal**; `thermal_gate_e4b_1788234851.json` |
| mini synth, 300 s endurance | FAIL | 0.888 | 0.570 | 0.163 | fair | 27 windows · 13 224 tokens · ~220 KV recycles · **RSS growth 0.06 MiB**; attributed **not thermal**; `thermal_gate_mini_1788235220.json` |

**This host cannot certify a sustained run, and the gate says so rather than
pretending otherwise.** A persistent ~100 %-CPU background process keeps the
1-minute load average near 15 and drifts on a multi-minute cycle. Across six 60 s
E4B runs in one session the sustained ratio ranged **0.86 – 1.08** at a
within-run `cv` of only 0.066–0.075: the *windows* are consistent, but the
run-level mean wanders with the background load, which a 60 s head-vs-tail pair
straddles. Every non-PASS above carries the same attribution line — the OS
reports no thermal pressure, so the cause is contention, not heat.

**What is established regardless of host noise:**

- **No leak.** 13 224 tokens and ~220 KV recycles over 292 s moved RSS by
  **0.06 MiB**. A second 300 s run moved it 0.016 MiB.
- **No stalls, no errors.** No window failed to produce work across any run.
- **Correct discrimination.** Injected GPU contention that outlives the run
  produces FAIL or SKIPPED with a non-thermal attribution; a quiet mini run
  passes reproducibly at cv 0.017–0.020.

**Not established:** whether the M5 Pro throttles under sustained decode. That
needs a quiet host. Re-run `--e4b --thermal` with the background load stopped
before reading the E4B row as a property of the silicon.

**Caveat on the absolute rate.** The ~3.5 tok/s above is *not* comparable to the
~23.9 tok/s quiet figure elsewhere in this file. Per
[`bottleneck.md`](bottleneck.md) roughly 77 % of a decode token is host-side
per-token overhead, so a contended CPU depresses decode directly. A matched burst
bench (`GEMMA_METAL_BENCH_STEPS=200`) measured 3.29 tok/s on the same host —
i.e. the gate agrees with the burst bench at the same KV depth, and the gap is
the host, not the gate.

**31B is wired but unrun:** `mlx-community/gemma-4-31b-it-4bit` is not in the HF
cache on this machine, so the `31b` gate label has never executed.

---

## Distance to gates (summary)

| Gate | Target | Now | Gap |
|------|--------|-----|-----|
| E4B Q4 | ≥48–60 (pract. ≥76) | **~23.9** gemma-metal | ~2.0–3.2× |
| E4B + MTP | ~90–110 | accept **75%**; e2e **~12.1 tok/s** (early-reject) | need base ≥48 + GPU draft |
| 31B Q4 | ≥15 | Ollama **~12.3**; custom Hot **~6.83** | ~2.2× vs gate; ~1.8× vs Ollama |
| 31B + MTP | ≥25 | native DFlash pre-fix **~1.17** / accept≈0; MLX **~28–37** | unmet; GEMM dual-norm/`layer_scalar` fix landing — rebench |

---

## How to refresh

```bash
cd gemma-metal/bench
python3 bench.py probe
python3 bench.py ollama --model gemma4:31b-mlx --num-ctx 4096 --max-tokens 128
python3 bench.py ollama --model gemma4:e4b-it-q4_K_M --num-ctx 4096 --max-tokens 128
python3 bench.py mlx --model mlx-community/gemma-4-e4b-it-4bit \
  --prompt-tokens 128 --generation-tokens 128

# Custom stack Phase 4 harness
cd gemma-metal && cargo run --release --bin bench
cargo run --release --bin bench -- --e4b   # real MLX E4B from HF cache
cargo run --release --bin bench -- --e4b --thermal   # + 60 s sustained-load gate
cargo run --release --bin serve -- --preset 31b --port 8787
```

Paste new rows into the measurement table. Keep **Locked honest-lane targets** unless product doctrine changes.
