# arch02 (formerly MLSystemsLab `Rust_MLKit`) — Decision Log (for later audit)

Date: 2026-07-11. Covers the arch_02 Burn/Metal port on the M5 Pro. Each entry
records the decision, the evidence behind it, and what would overturn it.
Companion docs: [arch_02_value_resid/burn-port/README.md](burn-port/README.md)
(parity ledger + build/run), `reference/ablation_results/` (3070 Ti evidence).

**From-scratch metal-native stack (Phases 0–4):** decisions live in
[arch_02_value_resid/metal-native/DECISIONS.md](metal-native/DECISIONS.md)
(TensorOps, Metal 4, megakernels, Core ML deploy-only, f32 goldens, flash deferred).
Results / how-to: [arch_02_value_resid/metal-native/README.md](metal-native/README.md).

---

## D1. Stack: Burn 0.21 + Metal (CubeCL MSL), not Mojo / Core ML / rustml / Candle

- **Burn** is the only linked stack that is both Rust and trainable on the
  Apple GPU (autodiff + Metal backend). Verified 2026-07: Burn 0.21 backend
  table lists Metal for Apple GPUs via CubeCL.
- **Mojo/Modular**: Apple-silicon GPU support is kernels-only (M1–M5 work for
  Mojo GPU functions; MAX cannot serve/train full models on Apple GPUs yet).
  Revisit if MAX ships Apple-GPU model execution.
- **Core ML**: deployment/inference runtime, no training, no custom optimizer —
  reserved for a later export phase (int6 artifact → Core ML/ANE).
- **rustml**: abandoned (~2016, CPU-only). Ruled out.
- **Candle**: mature Metal inference but weaker training/optimizer story;
  Muon would be hand-rolled either way. Not chosen.

### Compiler-level facts this rests on (verified in Burn 0.21 source)

- `burn` feature `metal` = `burn-wgpu/metal` = `cubecl/wgpu-msl`: kernels are
  compiled by CubeCL's **native MSL compiler**, not WGSL→naga translation.
  `burn::backend::Metal<F=f32, I=i32, B=u8>` is the type alias.
- `fusion` (kernel fusion) and `autotune` (per-shape kernel variant search)
  ship in burn-wgpu's default features; we enable them explicitly in
  Cargo.toml so a feature-set change can't silently drop them.
- `burn::tensor::module::attention()` exists (scale/softcap/is_causal opts),
  BUT: (a) under `Autodiff` it decomposes to the naive fallback — no fused
  flash backward in 0.21; (b) the CubeCL flash kernel falls back to naive for
  `scale`/`softcap`/`attn_bias`, and causal-on-Metal has a known
  incorrect-results issue upstream. **Consequence:** hand-rolled grouped-GQA
  SDPA is the training default; `--features flash-attn` exists for
  benchmarking/forward-compat only. Re-audit when Burn ships flash backward
  or fixes Metal causal.
- rustc side: release profile is opt-level 3, thin LTO, codegen-units=1.

## D2. Architecture: arch_02 first, arch_01 as a config-flag A/B (not a separate port)

Second audit of the 3070 Ti evidence (see `reference/ablation_results/`):

- Long stage is only 4 runs; per-seed the archs split 1–1 (seed 1337:
  arch_01 by 0.0112; seed 42: arch_02 by 0.0057). Mean gap 0.0027 ≈ ⅓ of
  within-arch seed spread → statistical tie; "champion" is not settled.
- `champion_followup.json` is all nulls — every mid/long follow-up run failed
  (rc=1). No additional evidence for arch_01 exists.
- The "gating costs +72% step time" signal is measurement noise: arch_02's
  own two seeds differed 644 vs 1030 ms on identical config; mid-stage shows
  ~6%; theory says <1% (`Linear(512→8)` + sigmoid).
- Code delta for arch_01 is ~15 lines and lives in the AdamW/fp16 path
  (control tensor), never touching Muon or int6.

**Decision:** validate arch_02 first (fewer confounders), carry
`gated_attention` as a flag, and re-run the 2-arch × 2-seed ladder on the M5
to re-test the champion question on clean hardware. (Flag not yet
implemented — deferred with the golden-parity phase.)

## D3. Keep `burn-port` crate; promote to `mlkit/` workspace later — do not rewrite now

User asked explicitly whether a fresh `mlkit` scaffold would be better even
if a full rewrite is acceptable. Position:

- The crate's value is **validated knowledge, not scaffolding**: the parity
  ledger (RoPE NTK effective base ≈ 22082 at T=2048 because rotary's internal
  train_seq_len=1024; QK-norm *before* RoPE; XSA "mixed" value source;
  bigram boundary index = 2047; eval window scoring rules) plus verified Burn
  API patterns (ID-preserving `Param::map` write-back, `from_inner` +
  `require_grad`, gradient splitting by ParamId). A rewrite regenerates none
  of this for free and re-exposes every subtle bug this code already dodged.
- What a fresh workspace legitimately buys is **structure**, and structure is
  obtainable by *mechanical promotion*: move `src/` → `mlkit/crates/mlkit-core`,
  split the binary → `mlkit-cli`, contents unchanged (~1 hour of plumbing).
  Do this when the kit must host more than arch_02 (arch_01 flag, arch_03's
  custom GDN kernels do not belong in a crate named `arch02-burn`).
- **Rewrite trigger:** only if the golden-tensor parity phase reveals
  *systemic* math divergence from PyTorch. Goldens flip the economics —
  module-by-module rewrite against 1e-5 tests is fast and safe. Rewriting
  *before* goldens exist is the maximum-risk ordering.

**Sequencing:** land current optimization work → golden-parity phase →
workspace promotion (+ arch_01 flag) → A/B ladder.

## D4. Departures from the original plan (and their status)

Original plan: fresh `mlkit` workspace; Phase 0 Python golden exporter gating
every phase; then eval parity; then Metal training; arch_01 flag from day 1.

| Departure | Direction | Status |
|---|---|---|
| Build on existing `burn-port` instead of fresh workspace | Improvement (see D3) | Done |
| Muon: banked/batched NS5 instead of per-matrix optimizer | Improvement — closer to the Python 3D bank design than the original plan; equality test batched≡per-matrix | Done |
| Compiler layer pinned to verified facts (MSL path, fusion/autotune, flash-attn fallback) | Improvement over plan's "verify at cargo-add time" | Done |
| Optimization + deep logging pulled ahead of golden parity | **Open risk** — attention/Muon refactors currently protected by unit tests + ledger only. Goldens are the next gate; do not trust cross-implementation BPB comparisons until it lands | Pending |
| arch_01 flag deferred (user: "arch_02 only for now") | Neutral, scope choice | Pending |
| Calibration sweep (calibrated_bpb) deferred | Neutral — compare against sliding_bpb 1.9902 until then (calibrated 1.9875 needs the temp/softcap sweep) | Pending |

## D5. Mac-specific optimizations applied (audit against crate git diff)

1. **Backend**: `wgpu`(WGSL) → `metal`(MSL) feature + explicit
   `fusion`/`autotune`; startup device report; `cpu-smoke` feature for
   GPU-less end-to-end testing.
2. **Sync-stall elimination**: loss accumulated on-device (readback once per
   log interval, was per-micro-step ×96); global grad clip norm computed
   on-device with ONE scalar readback (was ~200 per-tensor readbacks); clip
   factor applied as device tensor.
3. **Per-forward rebuild removal**: RoPE cos/sin + causal mask built once per
   seq_len (was CPU trig + upload per attention call ≈ 1000/step).
4. **Banked Muon**: 66 matrices → 4 shape-stacks ([2L,512,512], [2L,512,256],
   [L,512,1536], [L,1536,512]); NS5 as batched matmuls (~50× fewer
   dispatches); momentum buffers as 3D banks; ID-preserving write-back.
5. **Attention**: GQA `repeat_dim` K/V copy removed via grouped reshape
   (fold group into matmul rows, batch over B·kv); flash path behind flag
   (see D1 caveat).
6. **Eval**: sliding windows batched (16/forward, one readback per batch; was
   ~1000 sequential batch-1 round-trips).
7. **Input pipeline**: prefetch thread (span assembly + bigram hash) feeding a
   bounded channel, overlapping host prep with GPU compute.
8. **Init correctness fix**: quintic Muon NS5 (singular-value band, NOT
   convergent) was being used for orthogonal *init* — replaced with a
   convergent cubic Newton-Schulz (`orthogonalize`); Muon keeps the quintic.
   This was a real latent bug (orthogonality error 0.27 → test-verified).

## D6. Deep-logging surfaces (what exists, where to look)

- `out/metrics.jsonl` + console: per-step loss, per-group grad norms
  (muon/embed/scalar), pre-clip global norm + clip factor, lr/momentum,
  tok/s, step ms, RSS.
- Phase profiler (sync-gated, `--profile-every N` / first N steps):
  data_prep / upload / forward / backward / grad_split / clip / adamw / muon /
  ema in ms — async otherwise, so steady-state speed is unaffected.
- Muon internals: per-bank NS5 ms on profiled steps.
- Kernel level (CubeCL, `cubecl.toml` → `out/cubecl/`): compile log
  (generated MSL), autotune log (winning kernel variant per shape),
  per-kernel profiling; env overrides `CUBECL_DEBUG_OPTION=profile-full`,
  `CUBECL_AUTOTUNE_LEVEL=full`, `CUBECL_DEBUG_LOG=stdout`.
- System level: Xcode Metal GPU capture and `powermetrics --samplers gpu_power`
  for utilization (manual).

## D7. Open items to audit later

1. Golden-tensor parity phase (exporter + 1e-5 module tests) — the gate
   before trusting any BPB comparison. (Long run started without it — see D8
   for interim dynamics evidence; final BPB still needs goldens for rigor.)
2. Full sprint-scale `--bench` (micro_batch {4,8,16,32} at 384 seqs/step of
   T=2048) — toy-scale already swept (D8); sprint-scale still open.
3. arch_01 `gated_attention` flag + 2×2 A/B ladder rerun.
4. Calibration sweep port for calibrated_bpb comparability.
5. Workspace promotion to `../MLSystemsLab/research/parameter-golf/mlkit/` (mechanical, per D3).
6. bf16 experiment on the MSL path (closer to Python's bf16 autocast;
   measure numerics + throughput before adopting). Likely the next big
   throughput lever given D8's ~4× CUDA gap is mostly f32/dispatch.
7. int6/int8 export + Core ML conversion for ANE inference (deployment).
8. CubeCL autotune "Async barrier instructions are not available" panic on
   Metal tensor-core matmul variants — currently benign (contained to
   benchmark threads; training continues). Track upstream / silence if noisy.

## D8. Measured M5 Pro results (2026-07-11, after optimizations)

### Config recovery (critical for fair comparison)

The default Burn-port sprint config (11L/512d, T=2048, 786432 tok/step) was
**not** what produced the recorded 1.99 BPB ladder numbers. Recovered from
the original 3070 Ti run logs into `--preset sota`:

- 4 layers, dim 128, T=256, 4096 tokens/step
- `lr_mul ≡ 1` (no warmdown at this length)
- Muon momentum 0.92 → 0.95
- mid-run val capped at 16,384 tokens

Always compare M5 BPB against this preset, not against the sprint defaults.

### Throughput (same math, different micro-batch shape)

| Shape | Hardware | ms/step | tok/s |
|---|---|---|---|
| 2×8 (3070 Ti reference shape) | M5 Metal f32 | ~slow | — |
| **16×1 (same 16 seqs/step)** | **M5 Metal f32** | **~2.9 s** | **~1,400** |
| reference | 3070 Ti CUDA bf16 | ~700–840 ms | — |

At toy scale Metal is dispatch-overhead-bound: collapsing grad_accum 8→1
(identical tokens/step) gave ~6× on the M5. Burn/Metal f32 is still ~4×
slower than PyTorch/CUDA bf16 at this size; phase profiles put the remaining
cost in forward/backward dispatch (clip/Muon now minor after banking).

### Dynamics fidelity (strongest interim evidence the port is faithful)

Seed 1337, `--preset sota`, live log
`arch_02_value_resid/burn-port/out/sota_arch02_seed1337/metrics.jsonl`:

| Step | M5 Burn loss | 3070 Ti CUDA loss (ref log) |
|---|---|---|
| 0 | 6.93 | ~6.9 |
| 50 | 4.93 | 4.89 |
| 100 | 4.92 | 4.52 (ref) / tracking within smoothing |

A full reproduction run was left running after the optimization session
(seed 1337, sota preset). When it finishes, compare
`FINAL val BPB (EMA)` against recorded value-resid sliding BPB
(1.9944 seed 1337 / 1.9860 seed 42). Re-run with `--seed 42` for the other arm.

### Autotune nuisance (documented, non-blocking)

CubeCL's autotuner panics loudly ("Async barrier instructions are not
available") when benchmarking tensor-core matmul variants unsupported on
Metal. Contained to benchmark threads; training continues. See D7.8.
