# metal-native (Phase 0–5 + throughput rebuild A–G/I)

From-scratch Rust + hand-written MSL training stack for `arch_02`. The Burn port
stays untouched as the parity reference.

## Weight layout

| Kind | Layout | Matmul / access |
|------|--------|-----------------|
| **Embedding tables** (`tok_emb`, `bigram.embed`, `ve.embed`) | `[vocab, dim]` | row gather |
| **Linear / bank matrices** (metal-native / Burn) | `[in, out]` | `x @ W` |
| **Python / golden `.npy`** | `[out, in]` | `F.linear` ≡ `x @ W.T` — **transpose last-2 on load** |

Tied logits use a transposed copy `tok_emb_t: [C, V]` for the head GEMM.

## What is delivered

- **Phase 0:** Metal runtime (**Metal 4 encode only**), TensorOps + simdgroup
  GEMM, AOT metallib, tests
- **Phase 1 (fwd):** per-op kernels + golden parity ≤ 1e-5 f32
- **Phase 2 (bwd):** hand-written backward + grad parity ≤ 1e-4 (post clip)
- **Phase 3 (optim):** on-device global L2 clip (device-resident coef), whole-bank
  Muon NS5 (4 dispatches), fused AdamW+EMA, `bin/train.rs` harness
- **Phase 4 (bf16-tune + full-run):**
  - Async multi-kernel command buffers (sync only at log / eval / bench)
  - Zero-copy bank matrix views + GPU blit copies (no host round-trips)
  - Fused `rms_norm_scale` megakernel glue; per-op kernels remain as fallback
  - bf16 TensorOps GEMM (`matmul2d_tensorops_bf16_f32`) + cast/copy helpers;
    `PrecisionMode::Bf16` wires GEMMs + bf16 flash + persistent bf16 GEMM
    operands (RMSNorm/CE remain f32; no bf16 BPB claim)
  - `--tok-mult` throughput sweep; FineWeb loader + JSONL metrics + sliding BPB
- **Phase 5 (Core ML export):** EMA/weight `.npy` dump → coremltools `.mlpackage`
  (fp16 + int8/int6 palettized), single-forward prefill, ANE bench — see
  [`out/coreml_export/README.md`](../../../MLSystemsLab/research/evidence/metal-native/coreml_export/README.md)

### Throughput rebuild (plan Phases A–G / I, validated in J)

- **A — Flash:** simdgroup FA-2 tiled online-softmax fwd+bwd with **L** on the
  tape; TensorOps tile + experimental multi-block probes in metallib — **not**
  hot path (DECISIONS **M8** blockers). Online probe matches FA-2 numerically
  on smoke shapes but still TG-stages O/P and has no TensorOps bwd.
- **B–C — Tape / sync:** views instead of copy storms; deferred CE readback;
  GPU transpose for tied head; cross-step overlap
- **D — Embeddings:** gather + GEMM instead of atomic storms
- **E — GEMM:** live TN/NT TensorOps descriptors + split-K for tall dW
  (`USE_TN_NT_DESCRIPTORS = true`); multi-tile slice axes fixed (TN slices A's M
  on dim0; NT slices B's N on dim1)
- **F — Optim packing:** clip reduce/scale packed into few encoders; scalar/embed
  AdamW segment-packed; per-step bump arena + dual input ping-pong wired in train
- **G — Eval:** no-tape inference forward, short-tail window scoring, GPU EMA blit
- **I — MLX baseline:** [`../mlx-baseline/`](../mlx-baseline/) reference port

## Train / bench

```bash
cd arch02/metal-native
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

# Throughput bench (B=16 / 32 / 64) — f32 path; keep short
cargo run --release --bin train -- --bench --bench-steps 15 --tok-mult 1 \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --out out/bench_b16 --f32

# 3k sota toy (warmdown=0 default) — Soft EMA ~2.02 with FA_TILED
cargo run --release --bin train -- \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json \
  --out out/sota_f32_clipsoft_seed1337 --iters 3000 --seed 1337 --f32 \
  --log-every 50

# Default 20k Soft recipe (FA_TILED + warmdown 3500; GEMM_ACCUM off)
# Seed 1337 published arm: --golden-init. Seed 42: omit it (seeded init).
export METAL_NATIVE_FA_TILED=1
unset METAL_NATIVE_GEMM_ACCUM
cargo run --release --bin train -- \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json \
  --out out/sota_f32_clipsoft_seed1337_20k_fa_tiled_softfix_warmdown \
  --iters 20000 --warmdown 3500 --seed 1337 --golden-init --f32 --clip-soft \
  --log-every 50 --eval-every 500
```

Optional **100k Soft** long-horizon (Soft-split + FA_TILED; **WSD** schedule — not last-10%):
```bash
export METAL_NATIVE_FA_TILED=1
unset METAL_NATIVE_GEMM_ACCUM
export METAL_NATIVE_DATA_SEED=0   # FineWeb skip=0 (published golden / CUDA-like)
cargo run --release --bin train -- \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json \
  --out out/sota_f32_clipsoft_seed1337_100k_fa_tiled_softsplit_wsd \
  --iters 100000 \
  --warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000 \
  --seed 1337 --golden-init --f32 --clip-soft \
  --log-every 100 --eval-every 1000
```
**Why WSD:** `--warmdown 10000` (start@90k) failed — best live **1.9137 @15999**,
then gnorm→thousands + BPB ~1.96–1.97 by ~22k; warmdown never reached. Validated
WSD arm (`out/.../100k_..._wsd/`): FINAL EMA **1.8828**, best live **1.8819**,
~5.6 h on M5 Pro. Do **not** use last-10% warmdown alone on Soft 100k.

**Validated 2026-07-13** (`out/.../100k_..._wsd/`): FINAL EMA **1.8828**, best
live **1.8819 @96999**, ~5.6 h wall / ~60k tok/s (M5 Pro). Through the former
~21–50k cliff: live BPB kept improving (~1.91→~1.90) vs failed rebound to ~1.97;
hold mean gnorm ~919 (failed ~3.4k).

### ~16M Soft scale-up (`--preset 16m`)

Shape (Metal FA-friendly `head_dim=32`, GQA=2, mlp=3×): **L=12 C=384 H=12/6
hd=32 mlp=1152 V=1024** → **16,411,948** params (~16.41M). Default step
**B=16 × T=256 = 4096 tok/step** (same tokens/step as sota toy; A/B winner vs
B8/T512). VE @ layers 10–11, XSA last 4. No golden banks — use seeded init
(omit `--golden-init`). Soft-split + FA_TILED + GEMM_ACCUM off carry over from
the 0.78M Soft recipe.

**Optimised bench (2026-07-13, M5 Pro 64GB, FA_TILED Soft f32):**
`out/opt16m_ab/fa_on_b16_t256_reconfirm/` → **~954 ms/step** (~4290 tok/s), RSS
~5.5 GB. Prior B8/T512 was ~1259 ms (~3250 tok/s). **20k wall estimate ≈ 5.3 h**
(20000 × 0.954 s; no eval overhead). See DECISIONS M12 for full A/B table.

**1k Soft verify (same recipe, warmdown 0):**
`out/sota_f32_clipsoft_16m_seed1337_1k_opt/` → FINAL EMA BPB **2.1462**, live
2.200 @999; wall ~16 min; steady ~890–950 ms/step; one transient gnorm ~3.9 @900
(recovered). Ready for 20k with `--warmdown 3500`.

```bash
# Smoke / step-time (distinct out/)
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export METAL_NATIVE_FA_TILED=1
unset METAL_NATIVE_GEMM_ACCUM
cargo run --release --bin train -- \
  --preset 16m --bench --bench-steps 20 --f32 --clip-soft --seed 1337 \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --out out/bench_16m_fa_tiled_softsplit

# Soft 1k smoothness check (warmdown 0)
cargo run --release --bin train -- \
  --preset 16m --iters 1000 --warmdown 0 --f32 --clip-soft --seed 1337 \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json \
  --out out/sota_f32_clipsoft_16m_seed1337_1k_opt \
  --log-every 50 --eval-every 200

# Soft 20k @16M (after 1k clean):
#   --preset 16m --iters 20000 --warmdown 3500 --f32 --clip-soft \
#   --out out/medium16m_f32_clipsoft_seed1337_20k_fa_tiled_softsplit_warmdown
# Optional longer context: --seq-len 512 --batch 8 (slower; same 4096 tok/step).
```

Flags: `--preset sota|16m` (default `sota`), `--batch N` / `--seq-len T`
(override preset defaults), `--f32` (parity precision), `--sync` (disable async
encode), `--tok-mult N` (B = preset_batch·N), `--eval-every`, `--log-every`,
`--seed N` (default 1337): **seeded weight init** (orthogonal banks + normal
embeds) and **FineWeb token skip** derived from N. Pass `--golden-init` (or
`METAL_NATIVE_GOLDEN_INIT=1`) to load exported golden `weights_init` instead
(**sota preset only**). `METAL_NATIVE_DATA_SEED=0` forces
FineWeb skip=0 (CUDA-like sequential-from-0).
`--warmdown N` (linear LR over N steps; with no `--warmdown-start`, final N of
`--total-iters`; **default 0** so 3k Soft stays unchanged; use **3500** on 20k Soft),
`--warmdown-start S` (absolute step to begin main warmdown; length = `--warmdown`
or through end if warmdown=0),
`--lr-floor F` (hold after main warmdown; default 0),
`--final-warmdown M` (last M steps: held level → 0),
`--total-iters N` (schedule horizon; default `start_step + iters`),
`--dump-at N` (Python weight + optim_step3 dump after logged step N),
`--load-weights DIR` / `--start-step N` / `--load-optim DIR` (resume /
bisect; `--load-optim` restores Muon momentum banks),
`--no-div-telemetry` (skip scalar/bank/Muon norm JSONL fields).
Long Soft runs with `--iters`/`--total-iters` ≥ 20000 and `--warmdown 0` print a
banner recommending `--warmdown 3500` (no silent schedule change). Runs ≥100k
with late/no warmdown recommend the WSD recipe above.

**BPB weight source:** mid-run `--eval-every` BPB is on **live** weights; the
final reported figure is **EMA** (shadows blitted into live weights before
eval). Sliding eval scores short tail windows (right-padded to `T`) so the val
set end is not dropped.

MLX throughput reference (Phase I): [`../mlx-baseline/`](../mlx-baseline/) —
minimal sota-shaped Transformer + Muon/`mx.compile` (`python train.py --bench`).

## Profiling (Instruments / NAX)

CPU phase boundaries (`data_prep` / `upload` / `forward` / `backward` /
`clip` / `optim`) are stamped by `Profiler` in `bin/train.rs` (see also
`src/signpost.rs` + `src/log.rs`). With profiling sync on, wall times
approximate GPU phase ends.

For **neural-accelerator utilization** (whether TensorOps GEMMs feed the M5
NAX — ground truth for Phase H bf16 / tf32):

1. Open Instruments → **Metal System Trace** (or GPU trace template).
2. Enable the **Neural Accelerators** utilization counter.
3. Run a short `--bench` under the trace; inspect NAX % during forward/backward
   GEMMs.

Notes:
- Legacy `MTLCounterSampleBuffer` returns **zeros** on macOS 26 — do not use it.
- In-app GPU timestamps: `GpuRuntime` builds an `MTL4CounterHeap` with the Metal 4
  package when available. Training M4 CBs fold t0/t1 into the same command buffer
  as compute; `synchronize` SharedEvent-waits that queue and exposes stamps via
  `take_metal4_stamps`. Encode path is **Metal 4 only** (DECISIONS **M3**).
  Lib tests cover batched `with_binder` encode (smoke + const-arena + offset binds).

## Divergence bisect (Phase 0) — resolved 2026-07-12 / clip-match 2026-07-12

Late-run explosion started ~step 2100 (pre-fix) then ~2500 (Muon×clip only).
Instrumentation + continue tooling:

1. **Norm telemetry** (default on): every `--log-every` step writes `divergence`
   to `metrics.jsonl` (`resid_mix` / `vr_lambda` / `attn_scale` / banks / Muon mom).
2. **Mid-run dump** + **resume**: `--dump-at 2000`, then
   `--load-weights …/weights --load-optim …/optim --start-step 2000`.
3. **Python CPU continue**: `scripts/bisect_continue_mps.py` (AdamW on banks).

**Bisect verdict (async / Muon×α):** Python AdamW continue from `dump_step2000`
stayed **stable** (gnorm ~0.3–0.6 through +150). Metal Muon continue re-exploded
→ **not** fully weight-poisoned; fault was metal post-2000 dynamics (async
hazards + Muon NS5 ignoring clip).

**Post-2500 root cause (Muon×clip only):** telemetry on
`out/sota_f32_postfix_seed1337/` showed layer-1 `vr_lambda` / `attn_scale`
norms climbing ~1.2→15 / ~7→35 while Muon banks **shrank** (e.g. `bank_qo`
23→12). Metal almost always clips (`gnorm` typically > 0.3). Muon×`clip_coef`
throttled updates but left **full weight decay**; AdamW’s `m/√v` is ~invariant
to grad scaling once moments adapt, so scalars kept taking full-size steps.
Result: banks starve under WD, residual scales explode, gnorm → 50+, EMA BPB 3.47.

**Fixes applied:**
- Per-`with_binder` compute encoder (async hazard fix) + per-step sync
- Flash dQ uses taped LSE + Delta; GPU GEMM C zero under async
- Muon `alpha *= clip_coef` so NS5 cannot undo global grad clip
- **Clip-match (2026-07-12):** Muon WD also `*= clip_coef`; AdamW step + WD
  `*= clip_coef` (device coef) so Adam/Muon stay matched under chronic clipping

**Evidence (seed 1337, FineWeb, f32):**

| Run | max gnorm @2500–3000 | EMA sliding BPB |
|-----|----------------------|-----------------|
| Pre-fix bisect (3000) | hundreds (@~2100) | 2.6887 |
| From-scratch + fixes, no Muon×clip (2500) | n/a (stop) | 2.2741 |
| From-scratch + Muon×clip (**2500 stop**) | n/a | 2.1109 |
| Continue dump + Muon×clip + `--load-optim` (+500) | **2.06** (@2000–2500) | **2.0436** |
| From-scratch + Muon×clip only (**full 3000**) | **51.6** @2950 | **3.4695** |
| From-scratch + **clip-match** (**full 3000**) | **2.80** @2999 | **2.0649** |
| 3070 Ti CUDA ref | — | **1.9944** |

Clip-match artifact: `out/sota_f32_clipmatch_seed1337/` (disp 404, ~16.4 min).
Live sliding BPB improves through the full run (**2.2607** @1999 → **2.1918**
@2499 → **2.0925** @2999). Layer-1 `vr_λ` / `attn_scale` stay O(1) (~1.06 /
~6.15 @2999); banks grow healthily (`bank_qo` 22→48).

```bash
# Continue from dump (preferred metal bisect path)
cargo run --release --bin train -- --f32 --iters 500 --start-step 2000 \
  --load-weights out/sota_f32_clipmatch_seed1337/dump_step2000/weights \
  --load-optim out/sota_f32_clipmatch_seed1337/dump_step2000/optim \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json --out out/continue_clipmatch_from2000
```

Smoke (synthetic, no FineWeb):

```bash
cargo run --release --bin train -- --iters 1 --dump-at 0 --f32 \
  --out out/dump_smoke --log-every 1
# expect out/dump_smoke/dump_step0/{weights,optim}/ + div lines in metrics.jsonl
```

## Core ML export (Phase 5)

```bash
# Python 3.12 venv (coremltools needs native BlobWriter; 3.14 wheels lack it)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install 'coremltools>=8' 'torch<2.8' numpy scikit-learn

# Prefer train EMA dump if present; else golden init (sota_seed1337 did not persist EMA)
python scripts/export_coreml.py \
  --weights golden/weights_init \
  --out out/coreml_export --seq-len 256
```

Packages land in `out/coreml_export/`:
`arch02_sota_fp16.mlpackage`, `arch02_sota_int8_palettized.mlpackage`,
`arch02_sota_int6_palettized.mlpackage`.

**BPB gap (updated 2026-07-12 Soft-harden):** default Soft EMA **2.0502**
(`out/sota_f32_clipsoft_seed1337_harden/`, accum off); quality recipe
`METAL_NATIVE_FA_TILED=1` → **2.0369**. CUDA **1.9944** (Δ ≈ 0.04–0.06).
Audit6 + `multiply_accumulate` Soft **2.058** / late gnorm~9 was rejected as
default. Remaining gap is FA numerics (not late explosion on the harden path).

## Benchmark (M5 Pro — f32, FineWeb, 15 steps, warmup 3 excluded)

Re-verified **2026-07-12** Soft-harden: `--bench --bench-steps 20 --f32`
→ **~56.6 ms/step / ~72k tok/s / 250 binders** (default). Speed A/B
`METAL_NATIVE_GEMM_ACCUM=1` → ~211 binders (Soft regresses).

| Config | ms/step | tok/s | Notes |
|--------|---------|-------|-------|
| metal-native B=16 (**Metal 4**, Soft-harden) | **~56.6** | **~72k** | default accum off / 250 binders |
| metal-native B=16 + FA_TILED | **~69** | **~60k** | Soft EMA **2.037** quality recipe |
| burn-port sota (ref) | ~2900–3400 | ~1200–1400 | |
| 3070 Ti CUDA (ref) | ~650–840 | — | bf16 PyTorch |

### Full 3000-step sota (seed 1337, FineWeb) — f32 clip-soft 2026-07-12

Default Soft-harden command:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo run --release --bin train -- \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json \
  --out out/sota_f32_clipsoft_seed1337_harden \
  --iters 3000 --seed 1337 --f32 --clip-soft \
  --log-every 50 --eval-every 500
```

Quality Soft (best EMA @3k): prefix `METAL_NATIVE_FA_TILED=1`.

| Metric | Soft harden (old) | Soft-everywhere + FA_TILED | Soft-split + FA_TILED | 3070 Ti ref |
|--------|-------------------|----------------------------|----------------------|-------------|
| late gnorm @2999 | **~3.4** | **~4.0** | **~5.7** | — |
| Final EMA sliding BPB @3k | **2.0502** | **2.0369** | **2.0222** | **1.9944** |
| Soft-split continue →8k FINAL EMA | — | explode **2.25** | **1.9469** | — |

**Default clip = Soft (`Muon×√c`, `AdamW×c`).** Soft-everywhere (`√c` on AdamW too)
explodes ~3.5–4.5k on 20k FA_TILED (FINAL EMA **2.2575**). Soft-split continue from
a Soft 3k FA_TILED dump →8k: FINAL EMA **1.9469**, best live **1.9635 @6499**
(`out/bisect_continue_softfix_fatiled_3k_8k/`). `--clip-match` / `--clip-python` remain.

### Default 20k Soft recipe (seed 1337, FineWeb) — validated 2026-07-12

Long Soft toy path (ladder parity with 3070 Ti). **Do not** change CLI
`--warmdown` default (stays **0** for 3k); pass `--warmdown 3500` explicitly.

| Knob | Value |
|------|--------|
| Clip | Soft-split (default / `--clip-soft`): Muon×√c, AdamW×c |
| Flash | `METAL_NATIVE_FA_TILED=1` |
| GEMM accum | **off** (`unset METAL_NATIVE_GEMM_ACCUM`) |
| Schedule | `--iters 20000 --warmdown 3500` (linear 1→0 from step 16500) |
| Precision / data | `--f32` + FineWeb + `--token-bytes` |

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export METAL_NATIVE_FA_TILED=1
unset METAL_NATIVE_GEMM_ACCUM
cargo run --release --bin train -- \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json \
  --out out/sota_f32_clipsoft_seed1337_20k_fa_tiled_softfix_warmdown \
  --iters 20000 --warmdown 3500 --seed 1337 --golden-init --f32 --clip-soft \
  --log-every 50 --eval-every 500
```

| Metric | Soft-split 20k + warmdown | Soft-split 20k no-wd | 3070 Ti ref |
|--------|---------------------------|----------------------|-------------|
| FINAL EMA sliding BPB (seed 1337) | **1.8969** | 1.9178 | **1.9944** |
| FINAL EMA sliding BPB (seed 42, reseed) | **1.8876** | — | **~1.9860** |
| 2-seed Soft mean (1337 golden + 42 reseed) | **1.8922** | — | ~1.990 |
| gnorm @19999 (1337 / 42 reseed) | **216 / 60** | 636 / — | — |
| Wall (M5 Pro, FA_TILED) | **~24–27 min** | ~23 min | ~4–5 h |

Artifacts:
`out/sota_f32_clipsoft_seed1337_20k_fa_tiled_softsplit_warmdown/` (golden-init),
`out/sota_f32_clipsoft_seed42_20k_fa_tiled_softsplit_warmdown_reseed/` (true
`--seed 42` init + FineWeb skip). Prior unused-seed seed42 dir is re-run noise only.

### Default recipes (3k / 20k / 100k)

| Horizon | Schedule | Notes |
|---------|----------|-------|
| **3k** | `--warmdown 0` (default) | Soft FA_TILED EMA ~2.02 |
| **20k** | `--warmdown 3500` | linear 1→0 from 16500; EMA **1.8969** |
| **100k** | `--warmdown-start 16000 --warmdown 24000 --lr-floor 0.1 --final-warmdown 10000` | WSD; FINAL EMA **1.8828**; **not** last-10% alone |

| Metric | 100k WSD (validated) | 100k last-10% wd (FAILED) |
|--------|----------------------|---------------------------|
| FINAL EMA sliding BPB | **1.8828** | stopped ~53.7k (no FINAL) |
| Best live BPB | **1.8819 @96999** | **1.9137 @15999** → rebound ~1.97 |
| Hold / late mean gnorm | ~919 ([40k,90k)) | ~3440 ([40k,53.7k)) |
| Wall (M5 Pro) | **~5.6 h** (~60k tok/s) | — |

Artifact: `out/sota_f32_clipsoft_seed1337_100k_fa_tiled_softsplit_wsd/`.

`--seed` now controls **weight init** (default) and **FineWeb token skip** (default;
`METAL_NATIVE_DATA_SEED=0` → skip 0). Use `--golden-init` for exported seed-1337
golden banks / parity. CUDA ladder seeds init only (sequential FineWeb from 0).

True seed-42 Soft 20k reproduce:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export METAL_NATIVE_FA_TILED=1
unset METAL_NATIVE_GEMM_ACCUM
cargo run --release --bin train -- \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes ../burn-port/token_bytes.json \
  --out out/sota_f32_clipsoft_seed42_20k_fa_tiled_softsplit_warmdown_reseed \
  --iters 20000 --warmdown 3500 --seed 42 --f32 --clip-soft \
  --log-every 50 --eval-every 500
```

Smoke (300 steps) showed early loss divergence: step0 6.930 vs 6.927; step100
4.528 vs 4.434; FINAL@300 EMA 3.033 vs 3.042.


### Core ML inference (M5 Pro, int8 palettized, T=256)

| Compute units | ms/forward | tok/s |
|---------------|------------|-------|
| CPU_AND_NE | **~0.75** | **~3.4e5** |
| ALL | ~0.97 | ~2.6e5 |
| CPU_AND_GPU | ~1.00 | ~2.6e5 |
| CPU_ONLY | ~5.7 | ~4.5e4 |

## Remaining gaps / deferred

### Exact 128M engine status (2026-07-14)

`--preset arch02-128m` is the exact 128,367,988-parameter B16/T256 target.
The Rust-owned engine now has version-6 full checkpoints/resume (including the
actual bf16 shadow bits) plus stable C functions
(`include/arch02_engine.h`), a move-only C++ RAII wrapper, and a Swift owner
wrapper. The independent MLX oracle validates manual GEMM attention backward at
C=768/KV=384 (dQ/dK/dV max error <1.2e-6).

The exact-scale Muon path uses bank-batched Metal 4 TensorOps, not the old
one-threadgroup-per-matrix loop. Six-step bf16/FineWeb smoke: **2816 ms/step,
1454 tok/s, 1701 dispatches, 16.1 GB RSS**. The full checkpoint/replay gate now
passes at **13.16 GiB current physical**, **1707 dispatches**, zero swap, loss
delta **0**, gradient-norm delta **7.15e-7**, and sampled-weight max delta
**2.41e-6**. NPY payloads use bulk I/O; the former per-float writer was rejected
after measuring only ~2 MB/s at exact scale.

Fourteen optimizer arms are native parity-qualified: NS5, NS3, Polar Express,
NorMuon, Muown, MONA, AdamW, Lion, Cautious AdamW/Lion, momentum SGD, Sophia,
Schedule-Free AdamW, and Prodigy. MiMuon and SOAP remain explicit systems-gate
exclusions because their exact SVD/eigendecomposition would require a forced
CPU synchronization on the current Metal/MPS API surface. They never silently
fall back to Muon.

Funnel + champion status (2026-07-21): the funnel is **complete** through the
exact-128M two-seed 1000 stage; the manifest unlocked **muon_polar_adamw @
matrix-lr 0.05**. The current champion is
**`out/champion_128m_seed1337_audit8`** (WIN stack, 2,000 steps, seed 1337):

| | audit7 | **current (Audit 8 WIN)** |
|---|---|---|
| FINAL EMA sliding BPB | 2.015576 | **2.0107** (seed2026 **2.0404**; no same-shape CUDA ref) |
| ms/step (mid-run / bench) | 2005 | **~1580–1683** (same-session baseline→WIN 2720→1506) |
| long20k FINAL EMA | — | **1.8155** @ ~1582 ms (`out/long20k_128m_audit8`) |
| current physical | ~13510 MB | ~13510 MB |

> **Scale note.** The `1.9944` / `~650–840 ms` 3070 Ti references elsewhere in
> this README are for `--preset sota` (4L × 128d × mlp384). `arch02-128m` is
> 24 × 768 × 2304, ~100× the parameters. **Do not compare 128M results against
> them.** `train.rs` now suppresses those strings for non-sota shapes and writes
> `"reference": null`. Same-shape CUDA 128M is **null** until
> `logs/cuda128m` is ingested from the 3070 Ti
> (`./scripts/cuda_ref_128m.sh probe|bench|quality`, then
> `./scripts/ingest_cuda128m.sh`). Always disclose `GRAD_ACCUM_STEPS`.

Wins now **default ON** (DECISIONS M15/M16 + Audit 9A): FA row bwd fast + bf16
(`fa_dqdkv` 892 → 80 ms), `BWD_CAST_ONCE`, and **`FA_FWD_FAST`** (flash 264 →
39 ms). `GLUE_ROWBLOCK` is **not** a speed win (A/B ≈ noise) — keep off.
Opt-out: `METAL_NATIVE_FA_FAST=0` / `FA_BF16=0` / `BWD_CAST_ONCE=0` /
`FA_FWD_FAST=0`.

WIN synced budget: **bwd ~457 · fwd ~412 · optim ~413** (`muon_banks` 99.8% of
optim — primary leftover). Instruments: `METAL_NATIVE_{BWD,FWD,OPTIM}_PROFILE=1`;
harness `scripts/bench_128m_ab.sh` / `scripts/blog_results.sh`. Architecture
truth: `docs/optimization_map.md` Audit 9.

Funnel machinery (kept for reruns):

```bash
cd /Users/bharath/Code/research/MLSystemsLab
python3 -m nanolab.native_funnel --status
python3 -m nanolab.native_funnel --dry-run-next
python3 -m nanolab.native_funnel --run-next   # repeat until stage complete
python3 -m nanolab.native_funnel --advance    # materialize the next gated stage
```

The study, resumable job ledger, and guarded final command live in
`../MLSystemsLab/research/optimizer-study.json`, `../MLSystemsLab/research/native-optimizer-funnel.json`, and
`../MLSystemsLab/research/champion-run.json`. Funnel arms retain final EMA validation but pass
`--no-final-weight-save`; the champion keeps EMA weights and complete v6
checkpoints every 250 steps.

Funnel status: the 100-step LR sweep finished **68/70** finite and stable-500
finished **13/14** finite. Polar leads the 500-step stage at EMA BPB **2.4070**,
followed by NorMuon **2.4680**, NS5 **2.4804**, and NS3 **2.5092**. Prodigy was
excluded after numerical failure at step 86. The next stage contains **14**
1,000-step jobs: the top six plus mandatory AdamW, each on seeds 42 and 2026.
See the journey note for the full tables. The runner always performs an
incremental release build before launch so an existing stale binary cannot
enter the study.

Two-seed 1,000-step update: MONA leads at mean EMA BPB **2.1809**, Polar is
statistically tied at **2.1823**, then NorMuon **2.2322** and NS5 **2.2326**.
Those four advance to exact 128M/500. The mandatory AdamW control completed but
degraded to mean BPB **11.2789** at its short-sweep-selected LR and is excluded
from exact scale.

1. **BPB parity** — @3k Soft-split FA_TILED EMA **2.0222** vs CUDA **1.9944**
   (Δ ≈ 0.03). @20k Soft+warmdown seed 1337 EMA **1.8969** and @100k WSD
   **1.8828** **beat** CUDA **1.9944**. Residual @3k gap: f32 vs CUDA bf16 +
   FA-2 vs flash-attn-3. **No bf16 BPB claim** until f32 3k closer to ~1.99.
2. **bf16 residual stream** — **landed** megakernel twins
   (`resid_mix_rms_norm_scale_bf16`, `residual_scale_add_rms_norm_scale_bf16`)
   write bf16 norm outs under `PrecisionMode::Bf16`; tape mix/mid + CE stay f32.
3. **TN/NT TensorOps descriptors** — **landed**.
4. **Full TensorOps flash as default** — probe + `--flash-tensorops` fwd opt-in;
   sota-shape smoke vs FA-2. Still off default (DECISIONS **M8**: no TensorOps
   bwd+LSE). Production: simdgroup FA-2.
5. **Metal 4** — **only training encode** (argument-table `Binder`, ~1 MiB
   const arena, residency registry, CounterHeap stamps in the training CB).
   Classic M3 encode / `--metal3` removed (DECISIONS **M3**). Lib goldens
   green; FineWeb bench B=16: **91.6** ms/step. Short BPB smoke OK (2-step EMA
   sliding **4.05**). Note: async M4 log path can show loss/gnorm **0** until
   sync.
6. **Block megakernels** — stem `rms_norm_smear`, resid/attn fusions, bf16 twins,
   decoder `skip_resid_mix_rms` kernel available; ~**403** disp/step (f32).
7. **Stateful KV Core ML** — `decode_reference` + `decode_kv_reference` **PASS**;
   `--stateful-kv` writes `arch02_sota_decode_step.pt` + state schema. Core ML
   `StateType` `.mlpackage` convert still blocked (dynamic slice lowering).
8. **Clip modes** — default **Soft** (`Muon×√c`, `AdamW×c`); `--clip-match` / `--clip-python`.
   See DECISIONS **M11**.
```bash
cargo test --release --lib
```

Phase J gate (re-verified **2026-07-14**, Metal 4-only encode):
**57/57** `--release --lib` pass (incl. optimizer formula/parity fixtures,
checkpoint replay, device research telemetry, intentional NaN rejection,
`metal4_batched_multi_dispatch_const_arena`,
fwd/bwd/optim goldens, `flash_attn_lse_and_bwd_gate`).
bf16 path: `bf16_train_path_smoke` / `bf16_flash_lse_smoke` (finite; not golden-gated).

Known caveat: pure e2e param matching can drift because AdamW amplifies tiny
bwd noise — optim kernels themselves match goldens (`optim_step3_parity_vs_goldens`).

## Prerequisites

- macOS 26+ (TensorOps / Metal 4 toolchain)
- Xcode 26+ with **Metal Toolchain** component
- Do **not** use `xcrun -sdk macosx metal` — `build.rs` uses `xcrun metal` + `-isysroot`
- Core ML export: Python **3.12** + coremltools (see Phase 5 above)
