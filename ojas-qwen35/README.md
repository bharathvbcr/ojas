# ojas-qwen35

A Qwen3.5 text-tower **training-step provider** for ojas, run on canonical
[tessl](../../tessl)'s Metal kernels.

- **Metal only.** tessl builds only on macOS on Apple silicon. On any other
  target this crate compiles to an empty library, and its tests compile to
  nothing.
- **Not an ojas `Backend`.** The whole step is one provider-level unit:
  - the forward and backward over GDN, head-dim-256 grouped-query attention,
    Qwen's `(1 + w)` RMSNorm and gated RMSNorm, and partial RoPE;
  - cross-entropy over the tied 248,320-row head, computed in vocabulary
    chunks;
  - hidden rows exposed for a loss outside the crate;
  - the gradient bank, its global norm, and AdamW.

  This crate's step does not go through `ojas_core::Backend`. The gated
  delta rule alone is now also a trait op (`Backend::chunked_gdn_forward` /
  `chunked_gdn_backward`), on the CPU (any dims) and Metal (tessl's
  `gdn_train`, key dim 128). wgpu, CUDA and HIP refuse it with
  `Unsupported` through the trait default, as they do `permute`.
  Head-dim-256 attention, gated RMSNorm, causal conv1d and the gates are
  not trait ops.

The crate exists so that "trained with ojas" can rest on ojas code. Lappi's
trainer (`crates/qd-train-metal`) depends on it by path and adapts this
concrete API to its own step-provider trait. This crate does not depend on
Lappi.

## The loop it serves

```text
let snap = Snapshot::from_dir(dir)?;                       // CPU: config + header pre-flight
let mut step = Qwen35Step::open(&snap, Numerics::Bf16Operands, budget)?;   // GPU
let plan = OptimizerPlan::build(step.parameter_table(), &group_spec)?;

for batch in batches {
    for seq in batch {                                      // one sequence at a time
        let pending = step.forward(&Sequence { ids, letter_rows, letter_targets,
                                               letter_scale: 1.0 / n_letter_rows_in_batch,
                                               span_positions })?;
        // pending.letter_ce_sum(), pending.hidden(): [n_span, hidden] f32 host tensor
        let dh = span_head_backward(pending.hidden());     // caller's, on the host
        step.backward(pending, Some(ExternalGrad { positions: &distinct, dh: &dh }))?;
    }
    let clip = clip_coefficient(1.0, step.grad_sq_norm()?, span_head_sq_norm)?;
    step.adamw_step(&AdamWHyper { lr, beta1, beta2, eps, grad_scale: clip }, &plan)?;
}
step.save_state(&dir)?;    // masters + exp_avg + exp_avg_sq + step count
```

The step works on one sequence at a time. tessl's pending step holds every
layer's `T x hidden` input between its forward and its backward, so the API
does not offer a batch-level step.

The first backward after an AdamW step overwrites the bank. Later backwards
add to it. `bank_state()` reports which case applies.

## Numerics

`Numerics::ExactF32` gives exact f32 GEMMs. `Numerics::Bf16Operands` rounds
GEMM operands to bf16 and accumulates in f32. In both modes the weights,
activations, gradients, GDN state, attention and optimizer stay f32. There
is no default, and the choice is made at `open`.

## Refused, never defaulted

Every refusal names the field and fires before any GPU work runs.

### Config refusals

- Any `config.json` key the crate does not know, in the root, in
  `text_config` or in `rope_parameters`. The tables in `src/config.rs` list
  every key and say whether it is read, checked, or has no effect on the
  text tower's step (with the reason).
- `head_dim` other than 256, the value tessl's attention training kernels
  are compiled for.
- A GDN key head dim other than 128.
- A GDN value dim that is not a multiple of 16.
- Query heads that are not a multiple of KV heads.
- GDN value heads that differ from key heads, because `gdn_train` has no head
  grouping.
- A conv width outside 2..=8.
- `rope_type` other than `default`, and any non-null `rope_scaling`.
- `mrope_section` unless its entries are positive and sum to
  `rotary_dim / 2`. Only in that case does text-only MRoPE equal the plain
  partial RoPE that tessl computes.
- `attention_dropout != 0`.
- `mamba_ssm_dtype` other than `float32`.
- An untied LM head, MoE, attention bias, or an ungated attention output.
- Any field on which tessl's own `Qwen35Config::from_config_json` reading
  disagrees with this crate's.

### Input refusals

- Token ids at or above the vocabulary size.
- The config's vision special tokens (`image_token_id` and the others). They
  switch transformers to multimodal positions.
- Letter rows that repeat or that fall outside the sequence.
- A scale that is not finite.
- External-gradient positions that repeat. Sum duplicate rows before passing
  them.
- A non-finite `dh`.
- A backward on a non-finite letter loss.
- A pending step whose weights have changed since its forward.

### Weights-file refusals

- A tower tensor missing from the file, or present with the wrong shape.
- A tensor under the tower prefix that is not a tower parameter.
- An `lm_head.weight`.

## Optimizer groups are data

`GroupSpec` holds two rule lists:

- learning-rate scale per parameter;
- weight decay per parameter.

Each list must partition the parameters. Every parameter matches exactly one
rule in each list, and every rule matches at least one parameter. Anything
else is refused.

No exclusion is built in. tessl's `default_weight_decay` is never called. The
tests express transformers' norm and `dt_bias` exclusions as data, and also
decay everything, to show this.

**Per-parameter learning rates are refused today.** tessl's AdamW takes a
single learning rate. `OptimizerPlan::check_tessl_lr` refuses any scale other
than exactly 1.0, naming the missing capability: a per-entry `lr_scale` in
tessl's `adamw_step`, in progress on tessl branch
`lappi-train-lrscale-mrope`. A uniform scale is also refused. Folding it into
the learning rate is the caller's decision.

## Loading, and why tessl's loader

`Snapshot::from_dir` runs on the CPU:

1. It parses `config.json`.
2. It resolves the weights file. With `model.safetensors.index.json`, every
   tower tensor must be in one file, because tessl's loader reads one file.
3. It checks that file's header against the tower map through ojas-io. Only
   the header is read.

On the 2B, the check finds 320 tower tensors. It reports the vision tower and
the MTP block as not loaded.

`Qwen35Step::open` then calls tessl's `Qwen35Model::load`. ojas-io cannot do
this bulk read, because `load` is the model's only constructor and its fields
are `pub(crate)`.

`open` compares tessl's live parameter table with the crate's map: names,
shapes and order. It refuses any difference. Per-entry vectors are always
built against the live table.

## State on disk

`save_state` and `load_state` store three kinds of f32 safetensors shards,
written and read with ojas-io:

- the masters;
- AdamW's `exp_avg`;
- AdamW's `exp_avg_sq`.

Tensors use transformers' names and layout, so a torch oracle can read them
directly. Each file's metadata records:

- the format;
- the kind and the shard;
- the step count;
- the config's canonical summary.

`load_state` checks every header before it writes anything. A failure after
the first device write poisons the provider.

A save goes through ojas-io's `replace_dir_with`. The files are written into
a staging directory, the tree is synced, and only then is it renamed into
place. On failure the stage is removed. Refused before anything is written:

- a target that exists in any form (a state directory is never overwritten);
- a target that is a symbolic link, dangling or not;
- a parent directory that is a symbolic link (refused by ojas-io).

Two limits come from ojas-io. Saves into one parent directory are serialized
by an `flock` with no timeout. A directory someone else creates at the target
between the check and the swap would be replaced.

A resume opens the base snapshot and then loads the state over it.

Memory on the 2B, from the shape of tessl's API:

- `open` holds four f32 copies of the 1.88 B parameters: weights, bank and
  two moments, about 30 GB.
- A table read (`read_entries`, `read_table`), and each kind of a save or
  load, also stages a whole table (7.53 GB requested) in device tensors,
  because tessl copies a whole table per call.
- On the host, a save holds one entry at a time: each streams into its shard
  through ojas-io's `SafeTensorsWriter` in 1 MiB pieces. The peak is the
  2 GB embedding, which also gets a shard of its own.

### Device working set

Every staging goes through one helper, `staging_tensors`.

1. It synchronizes first, so tessl recycles what the last step freed.
2. It allocates the table.
3. It refuses if the device is then over `recommendedMaxWorkingSetSize`,
   before any GPU work. The figure checked is the measured allocation, not
   a projection.

Past that size, Metal pages the resident set. On the 2B the next command
buffer timed out (`MTL4CommandQueueErrorTimeout`) and poisoned the runtime.

Measured on a 64 GB M5 Pro, where the working set is 51.54 GB:

| Point | Allocated |
| --- | --- |
| After a 2B step | 43.59 GB |
| Recycle done | 41.44 GB |
| With staging | 50.20 GB |

That leaves **1.34 GB of headroom**.

Without the recycle, the same staging reached 52.44 GB and faulted, 4 of 4
runs. These numbers are from the gradient read after one step at 128 tokens
with `Numerics::ExactF32`.

Not measured:

- a 2B `save_state` (one staging per kind, each recycled before the next);
- `load_state`;
- `Bf16Operands`;
- longer sequences.

Any of these that ends over the working set gets the refusal, not a
timeout. Tessl reads that take one entry at a time would remove the
staging, and the headroom question with it.

## Threading

`Qwen35Step` is not `Send`, because tessl's `GpuRuntime` is not. Keep a
provider on one thread.

## Tests

`cargo test -p ojas-qwen35` (from the ojas workspace root) runs only CPU
tests and never opens a GPU runtime. It covers:

- config parsing and every refusal;
- the tower map against the real 2B config;
- tessl's tiny fixture header;
- optimizer-plan construction and the learning-rate refusal;
- sequence, gradient and clip checks;
- state-format round trips and refusals through ojas-io;
- save-target refusals: anything present, and symbolic links.

These tests are `#[ignore]`d:

- `snapshot_header_matches_the_name_map` (CPU only) reads the real 2B
  snapshot's header from the Hugging Face cache.
- All tests in `tests/gpu_parity.rs` are named `gpu_*`. They open a Metal
  runtime and run serialized in an agreed GPU window:

```text
cargo test -p ojas-qwen35 --release --test gpu_parity -- --ignored --test-threads=1 gpu_tiny
cargo test -p ojas-qwen35 --release --test gpu_parity -- --ignored --test-threads=1 gpu_real_2b
```

## JSON

`config.json` and `model.safetensors.index.json` are read with ojas-io's
strict reader (`parse_json_with`). The limits are depth 32 and 16 MiB of
input. Duplicate keys are refused.

A dimension must be an integer literal; `2048.0` is refused. A float field
written as an integer must be exact in `f64`, so `9007199254740993` is
refused, not rounded. An integer literal beyond the `u64`/`i64` range is
refused anywhere in the file.
