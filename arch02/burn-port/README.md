# arch02-burn — Rust/Burn port of the Value-Residual GPT

Full **training loop** port of `train_gpt_sprint_native.py`
(`VALUE_RESIDUAL=1 GATED_ATTENTION=0`, calibrated BPB 1.9875) targeting
Apple Silicon. Runs on **Metal via CubeCL's native MSL compiler**
(`burn/metal` feature) — on an M5 Pro the whole loop (forward, backward,
batched Muon NS5) executes on the 20-core GPU with zero CUDA dependencies.

Status: **compiles, 18/18 unit tests green, CPU + Metal smoke-tested; training
dynamics verified against the original 3070 Ti run logs** (sota preset, loss at
step 50: 4.94 on M5/f32 vs 4.89 on 3070 Ti/bf16 — within seed noise).

## Measured on the M5 Pro (20-core GPU, Burn 0.21 Metal/MSL, f32)

| Config | tok/step | ms/step | tok/s | notes |
|---|---|---|---|---|
| `--preset sota` (4L·128d·T256, ladder config) | 4,096 | ~2,900 | ~1,400 | 3070 Ti reference: ~700–840 ms/step (bf16, PyTorch SDPA-math) |
| `--preset sota`, original 2×8 accum shape | 4,096 | ~17,000 | ~240 | grad-accum on tiny batches is pure dispatch overhead on Metal — don't |
| sprint (11L·512d·T2048), mb 4 | 16,384 | ~15,100 | ~1,080 | forward 4.1 s · backward 8.0 s per phase profile |
| sprint, mb 8 | 32,768 | ~33,900 | ~970 | GPU saturated; scores tensor is memory-bound |

Takeaways so far (from `out/metrics.jsonl` + phase profiles):
- At toy scale the run is **kernel-dispatch-overhead bound** (wgpu/Metal
  command overhead), so fewer+larger dispatches win: `micro_batch 16 × accum 1`
  is ~6× faster than the 3070 Ti's `2 × 8` shape at identical math.
- At sprint scale the [B,8,T,T] attention scores dominate (memory-bound) —
  the fix is a fused flash kernel with backward support, which Burn/CubeCL
  does not have for training yet (see below).

## Why Burn (and not Core ML)

The milestone is *continuing training* on the Mac. Core ML is a deployment
runtime — no custom optimizers, no Muon, no training loop. Burn gives
autodiff + Metal execution; Core ML export remains the right *later* step for
shipping inference.

## Layout

```
src/
├── config.rs         hyperparameters (exact python defaults) + LR/momentum
│                     schedules + logging/profiling knobs
├── model/
│   ├── norm.rs       RMSNorm (weightless, F.rms_norm parity)
│   ├── rope.rs       partial RoPE (16/64 dims) + NTK base extension +
│   │                 RopeTables (device-cached cos/sin, built once/seq_len)
│   ├── attention.rs  GQA 8q/4kv + QK-norm + value residual + q-gain + XSA;
│   │                 copy-free grouped GQA (no repeat_interleave); optional
│   │                 fused-attention path behind --features flash-attn
│   ├── mlp.rs        squared leaky-ReLU (slope 0.5), 512→1536→512
│   ├── block.rs      resid_mix / attn_scale / mlp_scale, ln 1/√(l+1)
│   └── gpt.rs        tied embed, bigram hash embed, SmearGate, ValueEmbed,
│                     U-net skips, softcap-30 logit head
├── optim/
│   ├── muon.rs       BankedMuon: batched NS5 over 4 shape-grouped banks
│   │                 ([2L,512,512] / [2L,512,256] / [L,512,1536] /
│   │                 [L,1536,512]) — one batched-matmul NS5 per bank instead
│   │                 of 66 tiny per-matrix loops. Momentum warmup via atomic
│   │                 handle. (MuonWarm per-param variant kept for tests.)
│   ├── clip.rs       GLOBAL grad-norm clip 0.3, fully on-device — ONE scalar
│   │                 readback per step (was ~200)
│   └── init.rs       orthogonal init via *convergent* cubic Newton-Schulz
│                     (the Muon quintic does NOT converge to orthonormal;
│                     see module docs)
├── ema.rs            EMA decay 0.997 (final export = EMA, matches python)
├── data.rs           shard reader (magic 20240520) + PrefetchLoader
│                     (background thread overlapping host prep w/ GPU)
├── bpb.rs            BPB + sliding-window eval, windows batched [B,T]
├── log.rs            deep logging: JSONL step metrics, sync-gated phase
│                     profiler, RSS, device report
└── bin/train.rs      training loop + --bench throughput sweep
cubecl.toml           kernel-level logging (per-kernel profile, autotune
                      decisions, generated-MSL compile log) → out/cubecl/
scripts/
└── export_token_bytes.py   SentencePiece → token_bytes.json LUTs
```

## Build & run (on the M5 Pro)

```bash
# 1. token-byte LUTs for BPB (once; requires `pip install sentencepiece`)
python scripts/export_token_bytes.py \
  ../../../parameter-golf/data/tokenizers/fineweb_1024_bpe.model token_bytes.json

# 2. tests (CPU, ndarray backend)
cargo test

# 3. REPRODUCE THE LADDER NUMBER (recommended first full run):
#    the exact 3070 Ti long-stage config (4L/128d, 3000 steps, 4096 tok/step,
#    lr_mul=1, momentum 0.92→0.95, val = first 16384 tokens, stride 64).
#    Reference to beat/match: sliding BPB ≈ 1.99 (value_resid, seeds 1337/42).
cargo run --release --bin train -- --preset sota --seed 1337 \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --token-bytes token_bytes.json --out out/sota_arch02_seed1337
# ≈2.9 s/step on the M5 → ~2.5 h for 3000 steps.

# 4. throughput sweep for the full sprint model
cargo run --release --bin train -- \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 --bench

# CPU-only smoke (no GPU needed): add --no-default-features --features cpu-smoke
```

Flags: `--preset sprint|sota --iters --warmdown --micro-batch --grad-accum
--seq-len --eval-every --eval-batch --log-every --profile-every --seed --out
--bench --bench-seqs`.

### Known noise: autotune "Async barrier" panics

On first sight of each new matmul shape you may see scary
`Can't compile cpp kernel … Async barrier instructions are not available`
panics from `DSD-*` threads. These are CubeCL's **autotuner** benchmarking its
tensor-core-style "accelerated" matmul variants, which need async-barrier
instructions the Metal MSL path doesn't expose on this device. The panic is
contained to the benchmark thread; autotune falls back to the working variants
and training continues (exit code 0). `out/cubecl/autotune.log` records what
actually won. Track upstream cubecl for Metal accelerated-matmul support.

## Deep logging (core → kernel level)

Three layers of visibility, cheapest first:

1. **Step metrics** (always on): console line every `--log-every` steps and
   `out/metrics.jsonl` (one JSON object/step): loss, global + per-group grad
   norms (muon/embed/scalar), clip factor, lr_mul, momentum, tok/s, ms/step,
   RSS. Loss and grad-norm readbacks happen **only** on log steps — steady-state
   steps run fully async with zero GPU→CPU syncs.
2. **Phase profiler** (`--profile-every N`, plus first 3 steps): sync-gated
   timing attribution per phase — data_prep / upload / forward / backward /
   grad_split / clip / adamw / muon / ema — plus per-bank NS5 ms (qo/kv/up/dn).
   Sync gates make these numbers real GPU time, so keep N ≥ 50 for long runs.
3. **Kernel level** (`cubecl.toml` → `out/cubecl/`): per-kernel profile log,
   autotune decisions (which kernel variant won per op/shape on the M5), and
   kernel compile log. Env overrides:
   `CUBECL_DEBUG_LOG=stdout`, `CUBECL_DEBUG_OPTION=profile-full|debug-full`,
   `CUBECL_AUTOTUNE_LEVEL=minimal|balanced|extensive|full`.
   For GPU counters beyond CubeCL: Xcode → Debug → GPU Capture, or
   `sudo powermetrics --samplers gpu_power`.

## Mac-specific optimizations (vs first draft)

| Change | Why it matters on Metal |
|---|---|
| `metal` feature (CubeCL MSL) instead of WGSL | native MSL codegen path for Apple GPUs; f16/i8 dtypes available |
| On-device loss accumulation | was: `into_scalar()` × 96 micro-steps/step — each forces a full pipeline flush |
| On-device global grad clip over **bank stacks** | was: ~200 per-tensor readbacks/step → now 1 scalar readback; norm+scale touch 4 stacked tensors instead of 66 |
| RoPE tables + causal mask cached per seq_len | was: CPU trig + upload × 11 layers × 96 micro-steps ≈ 1000/step |
| BankedMuon batched NS5 | 66 small NS5 loops → 4 batched-matmul NS5 banks (~50× fewer dispatches); grads stacked once, shared with the clip |
| Grouped GQA (no `repeat_dim`) | K/V read once per KV head; fewer/larger batched matmuls |
| PrefetchLoader thread | shard IO + bigram hashing off the critical path |
| Batched sliding eval (`--eval-batch`) | ~1000 batch-1 forwards+readbacks → ~64 batched forwards (16 k-token eval: 11.6 s) |
| Micro-batch over grad-accum at toy scale | accumulation multiplies dispatch count; `16×1` ≈ 6× faster than `2×8` at identical math |
| `--bench` sweep | picks micro_batch empirically (64 GB unified memory allows ≫ the 3070 Ti's 4) |

Not done (documented): Burn 0.21's flash-attention kernel is inference-only in
practice here — `Autodiff` decomposes `attention()` to the naive fallback for
backward, and causal-mode flash on Metal has a known upstream issue — so
training uses the grouped SDPA. `--features flash-attn` exists for
benchmarking/forward-compat.

## Parity ledger

Matched exactly:
- Model math: value residual (per-block λ init [0.5, 0.5], v0 = layer-0 raw V),
  QK-RMSNorm **before** RoPE, per-head q_gain init 1.5, GQA grouped attention
  (bit-equal to interleaved expansion), XSA subtraction on layers 7–10 using
  the *mixed* v, squared leaky-ReLU(0.5) MLP, resid_mix/attn_scale/mlp_scale,
  ln scale 1/√(l+1), U-net LIFO skips with learned weights, SmearGate, bigram
  hash embedding, shared ValueEmbedding on layers 9–10, tied head softcap 30.
- **RoPE NTK extension**: training at seq_len 2048 with rotary
  `train_seq_len=1024` ⇒ effective base ≈ **22082**, not 10000. Easy to miss.
- Optimizers: Muon (nesterov, NS5 exact coefficients a=3.4445 b=-4.775
  c=2.0315, per-matrix `sqrt(max(1, out/in))` LR scale, wd 0.04 at unadjusted
  LR) on the 66 block matrices; AdamW lr 0.035 on the 3 embedding tables;
  AdamW lr 0.025 on scalars/controls; betas (0.9, 0.95), eps 1e-8, wd 0.04.
- Schedules: no warmup, linear warmdown over final 3500 of 20000 steps;
  Muon momentum 0.92→0.99 over 1500 steps; global grad clip 0.3;
  EMA 0.997 with EMA as final weights.
- Data & eval: sequential shard streaming, 786432 tokens/step,
  sliding-window BPB with stride 64 and SentencePiece byte attribution
  (+1 leading-space rule).

Deliberate deviations (documented in-code):
1. **f32 instead of bf16 autocast.** Also changes the dtype-dependent
   `F.rms_norm(eps=None)` epsilon (bf16 ≈ 7.8e-3 → f32 ≈ 1.2e-7). Expect
   *slightly better* numerics, not identical curves. (bf16 experiment is a
   candidate follow-up on the MSL path.)
2. **Orthogonal init via convergent Newton-Schulz** of a Gaussian rather than
   torch's QR `orthogonal_` (Haar-like, singular values → 1; verified to
   ‖MᵀM−I‖∞ < 0.05 in tests).
3. **Omitted**: 20-step compile-warmup phase (no weight effect), QAT/int6
   export packaging, DDP paths, temperature/softcap calibration sweep (the
   1.9875 reference is *calibrated* BPB; compare sliding_bpb 1.9902 until the
   calibration sweep is ported).
4. **Seeding**: Burn's RNG streams differ from torch's; distributional parity
   only.

## Roadmap after first full run

1. Overnight 3000-step run × seeds {1337, 42}; compare sliding BPB vs 1.9902
   (3070 Ti reference; step_avg 837 ms there — report M5 ms/step)
2. Port the temperature/softcap calibration sweep for calibrated-BPB parity
3. arch_01 upgrade: per-head gating module (one extra Linear + sigmoid)
4. bf16 float-element experiment on the MSL path
5. Deployment phase: int6 export + Core ML conversion for ANE inference
