# Framework layer design (approved; items 1–13 landed)

Written 2026-10-01 by a read-only design pass over the tree at that date. It is linked from [`pytorch-parity-plan.md`](pytorch-parity-plan.md) §7.

**Labels:** **V** means the cited line was read in that pass. **I** means inferred.

**Paths:** relative to the ojas root. `nanolab/` means `/Users/bharath/Code/research/MLSystemsLab/nanolab/`.

**Goal:** a Go or Rust caller can train and serve the nanolab GPT without Python. The model is 12 layers, d=768, 12×64 heads, SwiGLU 2048, V=50304, with a tied embedding, QK-norm, RoPE, a per-head gate and a value residual. It trains with Muon NS5 on 2D hidden weights plus AdamW, clip 1.0, and a warmup with cosine or WSD schedule.

**Status:** approved and implemented across items 1–13 of §9. The nanolab GPT is defined once in `ojas-model`, trained across CPU, Metal and wgpu, checkpointed to safetensors directories, and driven through the in-process Go API (`go/`). The `CpuBackend` default is `Numerics::Fast`.

The design depends on neither default: every bitwise test sets `Numerics::Exact` explicitly.

## 1. Crate boundary

**New crate: `ojas-model`.** It owns:
- the nanolab model definition, written once: spec, parameter names, init and the block;
- the device trainer;
- the training-checkpoint glue;
- the model half of the device decoder.

| Crate | Change | Why |
| :--- | :--- | :--- |
| ojas-core | Trait additions (§8); blanket `Backend` impls for `&B` and `Arc<B>` | Trait owner |
| ojas-autograd | `Tape::backward_seeded`, `take_grad`, `linear_cross_entropy` | Tape owner. It stays op-level, with no model code |
| ojas-cpu | `WsdSchedule` and `enum LrSchedule` beside `CosineSchedule` (`schedule.rs` V). Native kernels for the new trait ops | Already owns the schedules and `optim_group` (`train.rs:28-60` V). That policy is reused, so there is no second registry |
| ojas-io | Streaming `SafeTensorsWriter`, `SafeTensors::from_file(File)`, `replace_dir_with` | Today's writer takes borrowed slices of every tensor (`safetensors.rs` V), so the whole model must be on the host at once |
| ojas-model (new) | spec, names, init; the `Graph` trait; the block; `Eval<B>`; `Trainer<B>`; the checkpoint directory | Composition layer above autograd, io and data |
| ojas-infer | Depends on ojas-model. Keeps sampling and the fast CPU `forward_token`. `GptConfig` and `GptWeights` re-export ojas-model types. Adds `DeviceDecoder<B>` | One owner of the block. `forward_sequence` is replaced by `ojas_model::Eval` |
| ojas-capi | Adds ojas-model, ojas-io and ojas-data as dependencies. Drops its duplicate `MAX_HEADER_BYTES` (`load.rs:13` V; the original is in ojas-io) | Real Load, Train, Save and Generate |

The dependency graph is `core ← cpu ← autograd ← model → {io, data}`, then `model ← infer`, and `capi → {model, infer, io, data, cpu, wgpu, metal, device}`. There are no cycles, and the trainer's math stays on `Backend`.

## 2. One block, two executors; names and init

`ojas_model::Graph` is a trait with `type V: Clone`. It has one method per op: embedding, linear, rms_norm, rope, permute, sdpa, cached_attn, gate, vres, silu, mul, add, reshape and lin_ce. Two types implement it:
- `impl<B: Backend> Graph for Tape<B>` (`V = Var`) records for training.
- `Eval<B>` (`V = Tensor`) runs eagerly with no record.

`fn block<G: Graph>(…)` follows nanolab `Attention.forward` (`mixers.py:280-345` V):
1. norm1, then the q/k/v projections.
2. Per-head QK RMSNorm, then RoPE in `[B,T,H,D]`.
3. Value-residual blend in `[B,T,H,D]`.
4. Permute to `[B,H,T,D]`, causal SDPA, permute back.
5. Gate (it reads `h`), then o_proj and the residual.
6. norm2, SwiGLU, residual.

That is the same order as `ojas-autograd/tests/multihead.rs`. `CpuGpt` runs `DeviceDecoder`'s step on `CpuBackend` against a host `KvCache`, and a parity test binds it to `Eval<CpuBackend>`. Training v1 refuses GQA, because nanolab's default is MHA.

Safetensors names are nanolab `state_dict` keys, so torch exports load unchanged.

| Name (layer i) | Shape | Init (nanolab) | Group |
| :--- | :--- | :--- | :--- |
| `tok_emb.weight` (tied `lm_head`) | [50304,768] | N(0, 0.02) | AdamW, wd 0, lr 6e-4 |
| `blocks.i.norm1.weight`, `.norm2.weight`, `norm_f.weight` | [768] | ones | AdamW wd 0 |
| `blocks.i.mixer.{q,k,v}_proj.weight` | [768,768] | N(0, 0.02) | Muon |
| `blocks.i.mixer.o_proj.weight` | [768,768] | **zeros** (`model.py:239-240,263-271` V) | Muon |
| `blocks.i.mixer.{q,k}_norm.weight` | [64] | ones | AdamW |
| `blocks.i.mixer.gate.weight` | [12,768] | **N(0, 0.02)**: `self.apply(_init_weights)` runs after `mixers.py:245` zeroed it (`model.py:212,236` V) | Muon |
| `blocks.i.mixer.gate.bias` | [12] | zeros | AdamW |
| `blocks.i.mixer.vr_lambda` | [1] | zeros (sigmoid 0.5). Layer 0's has no grad and is skipped | AdamW |
| `blocks.i.ffn.{gate,up}.weight` | [2048,768] | N(0, 0.02) | Muon |
| `blocks.i.ffn.down.weight` | [768,2048] | **zeros** (`model.py:290-293` V) | Muon |

**Init notes:**
- nanolab has no scaled-residual init. It zero-inits `o_proj` and `down` instead.
- Writing stores only `tok_emb`. Loading accepts a `lm_head.weight` only when it is bit-equal to `tok_emb`.
- The `_orig_mod.` prefix is stripped.
- The spec goes in `__metadata__["ojas.spec"]`.
- Each parameter is drawn from `CounterRng` seeded with `seed ^ fnv1a(name)`, through Box-Muller, so the init does not depend on parameter order.
- torch's RNG stream is not reproduced. Parity runs load torch's init from safetensors instead.

## 3. Training step on any `Backend`, device-resident

`Trainer<B>` holds:
- uniquely owned params;
- Muon momentum or AdamW m and v, per parameter;
- `grad_acc`;
- `step`;
- `LrSchedule`;
- the sampler and its `DataCursor`;
- `state: Ready | Poisoned`.

One `step()`:
1. `mult = schedule.multiplier(step)`. This is the pre-increment step, as in `HybridOptimizer` and nanolab `train.py:305-344` (clip at 336, `opt.step` at 343).
2. Memory preflight: `budget.check_room(self.preflight_bytes(rows, k))` ensures room for optimizer scratch and previously measured step peak before any forward execution. `budget.reset_peak()` restarts the high-water mark for this step.
3. Per micro-batch (K of them):
   - `tape.clear()`, then a leaf per param. A resident tensor uploads as a shared clone.
   - Forward through 12 blocks, then the fused head CE (§5).
   - `backward_seeded(loss, 1/K)`, then `take_grad` → `accumulate_grad` into trainer-owned buffers.
   - Sum the loss on the device.

   This scales before summing, like nanolab. The CPU `GradAccumulator` divides after summing instead. The two give identical bits for power-of-two K (I).
4. `tape.clear()` before any optimizer call, because in-place updates need unique ownership.
5. `clip_grad_norm(&mut accs, 1.0)`. On wgpu this is a sync point that reports any pending fault before it scales. Any error up to here leaves params, moments, step and cursor untouched. Policy `Abort` keeps the cursor; `SkipBatch` advances it and returns `E_NONFINITE`.
6. Pre-apply check: `budget.check_room(self.optimizer_scratch)` checks room for optimizer allocations with gradients alive, preventing out-of-memory errors from poisoning parameter states during the update.
7. Each parameter steps with `muon_ns5_step` or `adamw_step`, using the `ojas_cpu::optim_group` constants.
8. `backend.sync()?` (T1). wgpu reports optimizer faults only at the next sync. An error from step 7 or 8 marks the trainer `Poisoned`: later Step and Save calls refuse until a resume from checkpoint.
9. Commit: `step = next_step(step)?`, advance the cursor, download the loss once, record the measured step peak (`record_peak`), and return the mean micro-loss.

**Other properties:**
- **Readbacks:** one tensor readback per step, the loss.
- **Preflight Allocation Bounds:** `Trainer` creation computes `state_bytes(&table, &spec, cfg.seq_len)` and verifies room via `check_room` before allocating slots.
- **Data:** comes from the Feistel `BatchSampler`, with no stored RNG.
- **Optimizer reference:** `HybridOptimizer::step` (Exact, serial) is the bitwise reference for the optimizer phase on CPU.

## 4. Checkpoint

The full state is about 1.144 GB (1.066 GiB):

| Part | Size |
| :--- | ---: |
| Weights | 494.6 MB |
| Muon momentum | 340.2 MB |
| AdamW m+v for the embedding | 309.1 MB |
| AdamW m+v for vectors | 0.2 MB |

That exceeds `MAX_CHECKPOINT_BYTES = 1 GiB`. `CheckpointV1` also holds every payload as a `Vec<u8>`.

**Decision:** a checkpoint directory. The 1 GiB cap stays as a sanity cap on the small state file.

| File | Content |
| :--- | :--- |
| `model.safetensors` | F32 weights with nanolab names. Metadata: `ojas.spec`, `ojas.step`, `ojas.run`. Torch and `Load` can read it directly |
| `optim.safetensors` | `muon.<name>`, `adam_m.<name>`, `adam_v.<name>` |
| `state.ojck` | Checkpoint v1 with empty tensor sections: config JSON, `tokenizer_hash`, `git_sha`, `step`, `data_cursor`, `rng_state` |

**Writing** streams one tensor at a time, writes into a sibling temp directory, fsyncs, then renames.

**Reading:**
1. Read `state.ojck`.
2. Open both safetensors files.
3. Check that the step and run agree across all three files.
4. Check that names, shapes and dtypes match exactly.
5. Stream each tensor and upload it.
6. Swap the trainer only after every check passes.

**`DataCursor`:** `shard` is the epoch and `token_index` the window ordinal. `docs/checkpoint-v1.md` item 6 needs that one-line fix.

**Resume-equivalence test (G9):**
- **Tiny spec, CPU Exact, K=2.** Run 40 steps straight, and separately 20, save, drop the trainer, resume, then 20 more. Loss bits, final params, moments and cursor must all be bitwise equal.
- **Metal and wgpu.** Same test on one device.
- **Negative cases:** a step mismatch, a seed mismatch, a truncated optim file, and a missing or extra tensor must each refuse with nothing applied.

## 5. Memory

**Full-vocab CE.** The trait's CE needs `[N,V]` logits plus an `[N,V]` gradient, about 3.30 GB each at N=16384. T3 `linear_cross_entropy_mean` fixes this:
- **Chunking:** it works on a `CeChunk {rows, cols}` tile at a time and computes the loss and both gradients in one pass, like nanolab `FusedLinearCrossEntropy` (`model.py:124-170` V), which chunks rows only. Scratch is at most `rows·cols·4`; rows only, at 1024 rows and V=50304, would be about 206 MB.
- **Backward:** the gradients are for seed 1. `Tape` records `Rec::LinearCe` and scales them by the seed.
- **Valid count:** `n_valid` is global. If every target is ignored the result is `NonFinite`; nanolab clamps the count to ≥1 instead.

**Activations.** About 1.13 MB per token for 12 layers (I).

| Micro-batch, T=1024 | Activations | Total incl. state and CE | Metal device worst case |
| :--- | ---: | ---: | ---: |
| B=4 | 4.6 GB | about 6.4 GB | about 7.5 GB |
| B=16 | 18.5 GB | about 20.3 GB | about 21.4 GB |

The Metal column is the total plus the uncharged pool cache, capped at 1 GiB once a session's budget reaches 4 GiB, plus tessl's rounding: under 16 KiB per buffer past 1 MiB, and under 512 KiB per buffer up to 1 MiB (`ojas-metal/src/backend.rs` module docs). It was 2× when tessl rounded every buffer to a power of two and cached up to 2 GiB.

v1 runs B=4 × K=16, the same 65,536 tokens per step as nanolab's 16×4, and needs no activation checkpointing. A single B=16 micro-batch needs per-block recompute, which is phase 2. The capi 1 GiB process step budget cannot hold the model state, so training sessions get a per-session `Budget` from the Load options.

## 6. Inference on GPU

`DeviceDecoder<B>` keeps a device K and V cache per layer, time-major `[B, Tcap, Hkv, D]`. That is the same order as the host `KvCache`, so a write needs no permute.

| Phase | Path |
| :--- | :--- |
| Prefill on an empty cache | `Eval` block with causal SDPA; the cache is written with T5 `kv_cache_write` |
| Decode, and prefill onto a non-empty cache | T4 `cached_attention_forward`: q `[B,Tq,H,D]`, GQA folded in, no permute |
| Logits | Last row only. The `[V]` row is read back and passed to `sample_token` |
| RoPE | `cos`/`sin` rows for `pos..pos+Tq` uploaded per call |

## 7. Go/C API

`Session` gains `state: Arc<Mutex<ModelState>>`, where `ModelState` is `{engine, trainer, tokenizer}`. `Engine` is `enum { Cpu, Metal, Wgpu(Arc<WgpuBackend>) }`; the wgpu arm needs T6. A `try_lock` failure gives `ojas:E_BUSY:`, the first producer of that error kind.

Opcodes 1–5 keep their numbers.

| Op | Go | Payload → result |
| :--- | :--- | :--- |
| 1 LOAD (real) | `LoadModel(ctx, path, LoadOptions{Device, Threads, BudgetBytes})` | O_NOFOLLOW open, then `SafeTensors::from_file`, validate the spec, upload → id |
| 6 NEW | `NewModel(ctx, spec, seed, opts)` | Fresh nanolab init |
| 7 TRAIN_OPEN | `OpenTrainer(ctx, id, TrainConfig{TokenBin, Batch, Seq, Accum, Seed, Schedule, MatrixLR, AdamLR, GradClip, OnNonFinite})` | `TokenBin` plus a `BatchSampler` |
| 8 TRAIN_STEP | `TrainStep(ctx, id)`, `TrainStepTokens(ctx, id, x, y)` | `{Loss, GradNorm, LR; Step, Tokens}` |
| 9 SAVE, 10 RESUME | `SaveCheckpoint(ctx, id, dir)`, `Resume(ctx, dir, opts)` | Checkpoint directory (§4) |
| 11 TOKENIZER, 12 TOKENIZE | `LoadTokenizer`, `Tokenize`, `Detokenize` | ojas-data GPT-2 BPE |
| 13 SAMPLE | `GenerateIDs(ctx, id, prompt, SampleOptions{Temperature, TopK, TopP, Seed, MaxNewTokens, Stop})` | ids |

**Existing tests:**
- **Header-only `Load` tests:** today's header-only behaviour moves to a new `Inspect`, and its tests move with the same assertions.
- **Two-token `GenerateGreedy` demo tests:** replaced by real-model tests on a checked-in tiny nanolab safetensors, about 50 KB. No test is deleted or weakened.

## 8. Trait and Tape additions

```rust
// ojas-core Backend
fn sync(&self) -> Result<(), OjasError> { Ok(()) }                                          // T1
fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError>;         // T2: acc += grad
fn linear_cross_entropy_mean(&self, input: &Tensor /*[N,d]*/, weight: &Tensor /*[V,d]*/,       // T3
    targets: &Tensor /*U32 [N]*/, ignore_index: Option<u32>, chunk: CeChunk /*{rows, cols}*/, want_grad: bool)
    -> Result<LinearCe, OjasError>; // LinearCe { loss, grad_input: Option<Tensor>, grad_weight: Option<Tensor> }
fn cached_attention_forward(&self, q: &Tensor /*[B,Tq,H,D]*/, k_cache: &Tensor /*[B,Tcap,Hkv,D]*/, // T4
    v_cache: &Tensor, kv_len: usize, window: Option<usize>) -> Result<Tensor, OjasError>;
    // the cache is a ring: position j is slot j % Tcap; query i sits at position p = kv_len-Tq+i
    // and reads p+1-window..=p (0..=p without a window); head h reads kv head h/(H/Hkv)
fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor /*[B,Tn,Hkv,D]*/, at: usize) -> Result<(), OjasError>; // T5
    // position at+t goes to slot (at+t) % Tcap; Tn <= Tcap
impl<B: Backend + ?Sized> Backend for &B / Arc<B>  // T6: forward every method, including the defaulted ones
// ojas-autograd Tape
pub fn backward_seeded(&mut self, var: Var, seed: f32) -> Result<(), OjasError>;               // A1
pub fn take_grad(&mut self, var: Var) -> Option<Tensor>;                                        // A2
pub fn linear_cross_entropy(&mut self, x: Var, w: Var, targets: Tensor,
    ignore: Option<u32>, chunk: CeChunk) -> Result<Var, OjasError>;                          // A3
```

T3–T5 default to `Unsupported`, as `permute` does. T3 chunks both rows and vocabulary columns: at a vocabulary of 248,320, 1,024 rows alone is 1 GB of f32 logits. Validators `linear_ce_dims`, `cached_attention_dims` and `kv_cache_write_dims` live beside the trait. The landed T6 test (`references_and_arcs_forward_every_method`) overrides all 38 methods of a marker double and calls each through `&B`, `&&B`, `Arc<B>` and `Arc<dyn Backend>`.

| Add | CPU | Metal | wgpu | Gate |
| :--- | :--- | :--- | :--- | :--- |
| T1 | default | default (synchronous per op today) | override (inherent `sync`) | G1: a deferred NaN surfaces at `sync` |
| T2 | native in place | native | native | G2: parity vs CPU; NaN leaves `acc` unchanged |
| T3 | native | native | native | G3: (a) equals the linear→CE→linear_backward composition at 1e-6 rel; (b) f64 gradcheck; (c) a `Budget` below `N·V·4` still runs; (d) device vs CPU at N=4096, V=50304, 1e-4 |
| T4, T5 | native (reads the ring in place) | native | native | G4: `kv_len == Tq` equals causal SDPA (with the same window, bit for bit under Exact); GQA and windowed rings vs an f64 reference; device vs CPU at 1e-5 |
| T6 | — | — | — | G5: `Arc<WgpuBackend>` keeps `Numerics::Fast` and `permute` |
| A1 | — | — | — | G6: seed 0.5 gives the same result on every device. The root-CE shortcut in `tape.rs` must not drop the seed |

**Model-level gates:**
- **G7:** `CpuGpt::forward_token` equals `Eval<CpuBackend>` and `DeviceDecoder` at ≤ 1e-5.
- **G8:** the `Trainer<CpuBackend Exact>` optimizer phase equals `HybridOptimizer::step` bitwise over 5 steps.
- **G9:** resume equivalence (§4).
- **G10:** exactly one readback per step.
- **G11:** init statistics are right, and the zero-inits are exactly zero.

## 9. Work breakdown

The coordinator serializes edits to the trait file, `Cargo.toml` and `Cargo.lock`.

| # | Item | Size | Lane | Depends on |
| :--- | :--- | :---: | :--- | :--- |
| 1 | T1, T2 default, T6, plus T3–T5 stubs | S | core | — |
| 2 | A1–A3, G6, gradcheck | M | autograd | 1 |
| 3 | `SafeTensorsWriter`, `from_file`, `replace_dir_with` | M | io | — |
| 4 | `WsdSchedule` and `LrSchedule` (coordinate with the ojas-cpu session) | S | cpu | — |
| 5 | T2–T5 native on CPU | M | cpu | 1 |
| 6 | T1–T5 on wgpu | L | wgpu | 1 |
| 7 | T2–T5 on Metal | L | metal | 1 |
| 8 | ojas-model spec, names, init, `Graph`, block, `Eval`, G11 | M | model | 1, 2 |
| 9 | `Trainer<B>`, G8, G10 | L | model | 2, 4, 8 |
| 10 | Checkpoint directory, G9 | M | model | 3, 9 |
| 11 | ojas-infer migration, `DeviceDecoder`, G7 | M | infer | 5, 8 |
| 12 | capi ops, Go API, test migration | L | capi, go | 9, 10, 11 |
| 13 | Torch oracle export, batch-start handoff, golden fixtures in ojas-oracle. **Landed:** see [`ojas-oracle/README.md`](../ojas-oracle/README.md) for the spec JSON, the fixtures and the `ParityModel` gates | M | oracle | 8 |
| 14 | Acceptance runs (§10) | M | bench | 6, 7, 12, 13 |

After item 1, items 2–7 and 13 can run in parallel. After items 2 and 8, items 9 and 11 can run in parallel.

**Metal deferred-fault contract (Resolved & Landed):**
- Metal adopted wgpu's deferred-fault contract in round 5 (documented in [`metal-deferred-faults.md`](metal-deferred-faults.md)), moving per-op waits to sync points (`Backend::sync`, `download`, `clip_grad_norm`).
- Reduced training step blocking waits from ~3,358 to 71–72 per step at K=4, dropping `adamw_full` from 183 ms to 22.6 ms.

## 10. Acceptance

**Metal, 124M against PyTorch nanolab.**

Setup:
- nanolab defaults at fp32 with compile off.
- B=4, K=4, T=1024; warmup 30, cosine over 300 steps.
- A FineWeb token bin.
- Both sides load the same `init.safetensors` exported from torch.
- Both sides use the same batch starts, which ojas's `BatchSampler` (seed 1337) dumps. nanolab samples with replacement, so without this the batches would differ.
- The torch oracle patches NS5 to fp32 (ojas's Muon is f32; finding F8) and reports the mean micro-loss. nanolab itself logs the last micro-loss × K (`train.py:348`). The oracle records both, and the gates use the mean.

Pass criteria:

| Check | Tolerance |
| :--- | :--- |
| Step 0 loss | \|Δ\| ≤ 1e-4 |
| Steps 1–100 | \|Δ\| ≤ 0.02 nats |
| Steps 101–300 | EMA(0.9) \|Δ\| ≤ 0.03 |
| Last-20 mean | within 1% |
| Grad norm, steps ≤ 50 | within 10% |
| Stock bf16-NS5 torch run | separate, looser row: ≤ 0.05 |

**CI on CPU only.**

Setup:
- A tiny spec: 2 layers, d=64, 4×16 heads, hidden 192, V=256, T=32, B=2, K=2.
- 40 steps on a synthetic bin, with `CpuBackend` Exact set explicitly.

Pass criteria:
- It matches an ojas-oracle torch golden curve at ≤ 2e-4 per step for steps 1–10 and ≤ 2e-3 after that.
- Loss drops by at least 1 nat from ln 256.
- G9 holds bitwise.
- It finishes in under 30 s in release.

**Critical files:**
- `ojas-core/src/backend.rs`
- `ojas-autograd/src/tape.rs`
- `ojas-infer/src/gpt.rs`
- `ojas-capi/src/session.rs`
- `ojas-io/src/safetensors.rs`
