# ojas-oracle

Golden numbers that ojas is checked against, with no Python or PyTorch at
test time. Two kinds:

- **`fixtures/rms_norm_f64.json`**: one hand-written fp64 RMSNorm row
  (`rms_norm_fixture`, `tests/load.rs`).
- **`fixtures/tiny/`**: nanolab's GPT at a tiny size, exported from torch on
  CPU in fp32, plus the ojas `BatchSampler` window starts the torch side
  replays. This is framework-design.md §9 item 13. The model parity gates in
  `src/parity.rs` run against these once ojas-model implements
  `ParityModel` (see "Enabling the model parity tests").

Python is used only to *generate* files (`python/`, a reference oracle);
nothing in the crate runs it.

## The tiny fixtures

Model: 2 layers, d=64, 4 heads × 16, SwiGLU hidden **192**, V=256, T=32,
tied embedding, QK-norm, RoPE base 10000, per-head gate, value residual.
Batches: B=2, K=2, seed 1337, from `tokens.bin`.

The hidden width is 192, not 128: nanolab derives it as
`_swiglu_hidden(d_model)` (config.py:364-368), which is 192 at d=64, and it is
not a `Config` field. 192 also matches framework-design.md §10.

| File | Bytes | Content |
| :--- | ---: | :--- |
| `tokens.bin` | 16,384 | 8,192 synthetic ids, headerless LE u16 (`golden.py token-bin`) |
| `batch_starts.json` | 20,047 | 40 steps × K=2 × B=2 window starts and their tokens, from `ojas_data::BatchSampler` |
| `init.safetensors` | 500,800 | nanolab init, seed 1337 (`export_init.py --tiny`) |
| `forward.safetensors` | 69,632 | x, y (step 0, micro 0), logits `[2,32,256]`; loss and checksums in metadata |
| `grads_init.safetensors` | 501,244 | d(loss/K)/dθ at the init, same batch |
| `grads_step5.safetensors` | 500,572 | the same at the f32 trace's step-5 parameters |
| `trace_ns5_f32.safetensors` | 505,280 | 40 steps with NS5 in f32: per-step losses, grad norms, LR multipliers; parameters after step 5 |
| `trace_ns5_bf16.safetensors` | 505,000 | the same with stock bf16 NS5 |
| `muon_step_bf16.safetensors` | 288,920 | one stock nanolab `Muon.step` (bf16 NS5) per case: square, tall, wide, and no decay or Nesterov; inputs and torch's results (`muon_step.py`) |
| `lr_schedules.json` | 2,502 | cosine and WSD multipliers, 20 steps, tiny and §10 acceptance settings |

Total about 3.4 MB. Every safetensors file carries
`__metadata__["ojas.oracle"]`: generator, torch and Python versions, the
nanolab commit, CPU / 1 thread / deterministic, the full nanolab `Config`,
and the fixture's own numbers.

**Why two gradient sets.** At the init `o_proj` and `ffn.down` are zero, so
every attention and FFN gradient upstream of them is exactly zero: only
`tok_emb`, both `o_proj`, both `down` and `norm_f` have gradients
(`tests/golden.rs` pins this). `grads_step5` is taken where every parameter
has moved, so the gradient gate exercises the whole block. Layer 0's
`vr_lambda` has no gradient in torch (no earlier values to blend) and is
absent from both files.

**Trace semantics.** Each step is nanolab's train.py:305-344 (schedule
multiplier from the pre-increment step, K micro-batches scaled by 1/K before
backward, clip 1.0, Muon + AdamW with nanolab's defaults, warmup 4, cosine
over 40). Three losses are recorded per step:
`mean_loss` (the mean micro-batch loss, which is what ojas reports and what
the gates use), `micro_loss`, and `nanolab_loss`, which is what
train.py:348 logs: the **last** micro-batch's loss, not the mean.

**The NS5 patch (pytorch-parity-plan F8).** ojas's Muon iterates
Newton-Schulz in f32 by default; nanolab's casts to bf16 (optim.py:46).
`Ns5Precision::Bf16` runs nanolab's bf16 iteration, and
`ojas-model/tests/oracle_parity.rs` gates it against `trace_ns5_bf16`. For
`trace_ns5_f32`, `nanolab.optim.zeropower_via_newtonschulz5` is replaced, for
the duration of the run, by a copy whose only change is `X = G.float()` in
place of `X = G.bfloat16()`. `Muon._orthogonalize` looks the function up as a
module global at call time (optim.py:84), so the copy is what runs; a call
counter (`ns5_calls` in the metadata) proves it. `trace_ns5_bf16` is stock
nanolab. Over the first 5 steps the two differ by up to 1.8e-4 nats, more
than the 1e-4 trace gate; over 40 steps by up to 1.1e-3.

**Cross-check against nanolab itself.** `golden.py` runs nanolab's real
`train()` for 5 steps through its `batchers=` seam (with the replayed
batches) for both NS5 variants, and refuses to write unless the weights are
bit-identical to its own step-5 snapshot and the logged losses and grad
norms are equal.

## The batch-start handoff

nanolab samples windows with replacement from torch's RNG; ojas samples
without replacement through a Feistel permutation (`ojas-data/src/sampler.rs`).
Neither reproduces the other, so ojas is the source and torch replays it:

1. `examples/dump_batch_starts.rs` opens the bin with ojas-data's `TokenBin`, draws
   (step, micro, row) starts with `BatchSampler`, and checks every
   micro-batch: the walked starts read through `TokenBin::read_into` must
   equal what `next_batch` returned. It writes `ojas-batch-starts-v1` JSON:
   `starts` (flat, `(step*K + micro)*B + row`), optionally `rows` (each
   window's T+1 tokens), the bin's length, header size and FNV-1a 64.
2. `python/batches.py` reads the bin exactly as nanolab's `Batcher` does
   (`np.memmap` LE u16, `data[s:s+T+1]`, int64, `x = seq[:, :-1]`,
   `y = seq[:, 1:]`) at those starts, after checking the bin's length and
   hash. `batches.py verify` compares every replayed window byte for byte
   with the dump's `rows`. `StartReplay` also implements nanolab's `Batcher`
   contract, so it plugs into `train(cfg, batchers=...)`.

For the §10 acceptance run (124M, FineWeb), pass `--format fineweb` to the
example; the dump records `bin_header_bytes: 1024` and the replay skips it.

The example uses ojas-data through ojas-oracle's dev-dependency. Its unit
tests run under `cargo test -p ojas-oracle` (`[[example]] test = true`).

## The `ojas.spec` JSON (`ojas-spec-v1`)

Stored in `__metadata__["ojas.spec"]` of every exported `model.safetensors`.
Field names follow ojas-infer's `GptConfig`. One flat object; unknown keys
are an error.

| Key | Type | Meaning (nanolab source) |
| :--- | :--- | :--- |
| `format` | string | `"ojas-spec-v1"` |
| `arch` | string | `"nanolab-gpt"` |
| `vocab` | int | `vocab_size` |
| `n_embd` | int | `d_model` |
| `n_layer` | int | `n_layer` |
| `n_head` | int | `n_head` |
| `n_kv_head` | int | `n_kv_head` (= `n_head` for MHA) |
| `head_dim` | int | `head_dim`, even |
| `hidden` | int | SwiGLU width, `_swiglu_hidden(d_model)` |
| `max_seq` | int | `block_size` |
| `rope_base` | float | `rope_base` (10000.0) |
| `rms_eps` | float | RMSNorm eps, 1e-6 (mixers.py:50; not a Config field) |
| `tie_embeddings` | bool | `lm_head` is `tok_emb` |
| `qk_norm` | bool | per-head RMSNorm on q and k |
| `gated_attention` | bool | per-head sigmoid gate on the attention output |
| `value_residual` | bool | blend layer 0's values with `sigmoid(vr_lambda)` |

The exporter refuses any nanolab config ojas does not implement (another
mixer, FFN, norm or position scheme, μP, loops, dropout, an attention-scale
ablation), so the booleans are always true today; they are explicit so a
reader can refuse rather than assume. Floats are JSON numbers (`1e-06`), so
a reader needs an f64-capable JSON parser: `ojas_io::json` keeps integers
exact and floats as f64. `ojas_oracle::spec::parse_spec` is a reference
reader.

Tensor names are nanolab `state_dict` keys (§2): `_orig_mod.` stripped,
`lm_head.weight` not stored (the exporter first checks it is bit-equal to
`tok_emb.weight`), all F32.

## Regenerating

Interpreter: `/opt/homebrew/opt/python@3.14/bin/python3.14` (Python 3.14.7,
torch 2.13.0, numpy 2.4.6, safetensors 0.8.0), the one `bench/` uses.
nanolab is read from `/Users/bharath/Code/research/MLSystemsLab`
(`OJAS_NANOLAB_ROOT` overrides) with bytecode writing off. CPU only.

```bash
# every tiny fixture, in order (about 7 s)
bash ojas-oracle/python/regen_tiny.sh
# the same twice, failing unless every file is byte-identical
bash ojas-oracle/python/check_determinism.sh 2

# one step at a time
PY=/opt/homebrew/opt/python@3.14/bin/python3.14
F=ojas-oracle/fixtures/tiny
$PY ojas-oracle/python/golden.py token-bin --out $F/tokens.bin
CARGO_TARGET_DIR=target-oracle cargo run --release -p ojas-oracle \
    --example dump_batch_starts -- \
    --bin $F/tokens.bin --seed 1337 --steps 40 --batch 2 --accum 2 --seq 32 \
    --out $F/batch_starts.json --rows
$PY ojas-oracle/python/batches.py verify --starts $F/batch_starts.json --bin $F/tokens.bin
$PY ojas-oracle/python/export_init.py --tiny --seed 1337 --out $F/init.safetensors
$PY ojas-oracle/python/golden.py fixtures --dir $F

# the 124M init (0.5 GB): a generated artifact, gitignored, never a fixture
$PY ojas-oracle/python/export_init.py --out ojas-oracle/generated/124m/model.safetensors --report
cargo test -p ojas-oracle --release --test golden -- --ignored   # checks it against §2
```

`export_init.py` checks every row of §2's init table before writing: exact
zeros and ones, N(0, 0.02) to 6 standard errors, and each parameter's
optimizer group against nanolab's own `_split_params`.

## Parity gates

Pinned in `src/parity.rs` and by `tests/parity_gates.rs`:

| Gate | Tolerance | Runner |
| :--- | :--- | :--- |
| Forward loss | 1e-5 relative | `forward_parity` |
| Forward logits | 1e-5 normwise relative | `forward_parity` |
| Gradients, per parameter | 1e-4 normwise relative; exact 0 where torch's is 0 | `grads_parity(GradsAt::Init / Step5)` |
| 5-step trace, mean loss per step | \|Δ\| ≤ 1e-4 nats | `trace_parity` |
| 5-step trace, parameters | 1e-4 normwise relative per tensor | `trace_parity` |
| §10 CI curve, 40 steps | \|Δ\| ≤ 2e-4 (steps 1–10), ≤ 2e-3 after; ends ≥ 1 nat below ln V | `curve_parity` |
| LR multipliers | 1e-12 relative | `tests/ojas_cpu_parity.rs` (runs today against ojas-cpu) |

## Enabling the model parity tests

ojas-oracle depends on no model crate. ojas-model (items 8–9) takes
`ojas-oracle` as a dev-dependency (no cycle: ojas-oracle depends on ojas-core
only) and implements `ojas_oracle::parity::ParityModel` in its tests:

```rust
// ojas-model/tests/oracle_parity.rs
use ojas_oracle::golden::{GradsAt, TensorSet, TokenBatch, TrainSetup};
use ojas_oracle::parity::{self, ParityModel};

struct Cpu; // wraps Eval<CpuBackend> / Trainer<CpuBackend>, Numerics::Exact

impl ParityModel for Cpu {
    fn forward(&mut self, params: &TensorSet, batch: &TokenBatch)
        -> Result<(f32, Vec<f32>), ojas_core::OjasError> { /* load params, Eval forward */ }
    fn grads(&mut self, params: &TensorSet, batch: &TokenBatch, seed: f32)
        -> Result<TensorSet, ojas_core::OjasError> { /* Tape, backward_seeded(loss, seed) */ }
    fn train(&mut self, params: &TensorSet, setup: &TrainSetup, steps: usize)
        -> Result<(Vec<f64>, TensorSet), ojas_core::OjasError> {
        /* Trainer with setup's optimizer/schedule; BatchSampler over setup.token_bin */
    }
}

#[test] fn forward() { parity::forward_parity(&mut Cpu).unwrap(); }
#[test] fn grads_init() { parity::grads_parity(&mut Cpu, GradsAt::Init).unwrap(); }
#[test] fn grads_step5() { parity::grads_parity(&mut Cpu, GradsAt::Step5).unwrap(); }
#[test] fn trace() { parity::trace_parity(&mut Cpu).unwrap(); }
#[test] fn curve() { parity::curve_parity(&mut Cpu).unwrap(); }
```

The runners choose the fixture, the batch and the tolerance; the adapter only
runs the model. `TrainSetup` carries the seed, B, K, T, schedule and every
optimizer constant from the trace's nanolab `Config`, and the path of
`tokens.bin`; the trainer draws its own batches with ojas-data's
`BatchSampler`, which is what `batch_starts.json` recorded.

## RMSNorm fixture

`fixtures/rms_norm_f64.json` computes, in f64,
`y_i = x_i / sqrt(mean(x²) + 1e-6) · w_i`. The JSON reader (`ojas_io::json`
with the fixture limits) refuses anything outside RFC 8259 (leading zeros,
`+`, `NaN`), numbers that overflow f64, duplicate keys, nesting deeper than
16, input over 4 MiB, and trees over a byte budget proportional to the
input; the fixture policy also refuses `null` and string escapes.
