# ojas-qwen35

The host-neutral half of ojas's Qwen3.5 training route: the config reader, the
tower's tensor names, the optimizer groups and the state-directory format.
**It has no training step.**

> **DO-NOT-MERGE-BEFORE-AB.** This branch deletes the tessl whole-step
> provider shell (`Qwen35Step`, its gradient bank and host AdamW) that used to
> live here, and with it the only in-tree comparator for the paired
> Tape-vs-tessl-provider A/B at Qwen3.5-0.8B-Base, T=2,048 (owner decision
> D5). Merge only after that A/B has passed.

## Where the step went

ojas owns Metal training, through the Tape. A caller that ran `Qwen35Step`
now builds the tower with `Qwen35TextConfig::tape_spec()` and steps it with
`ojas_model::qwen35::TapeStepProvider`, which implements `ojas-train`'s
`StepProvider` contract. tessl stays underneath as the Metal extension:
the training kernels and per-op launchers (`gdn_train`, `qwen35_bwd`,
`attn_train`, `cross_entropy`, `sq_norm`) are what `ojas-metal` dispatches.
The same tower on the Tape is measured in
`bench/results/2026-10-15-qwen35-2b-tape-gate/` and
`bench/results/2026-10-21-lora-stepcost-ab/`.

## What stays

- **`config`**: `Qwen35TextConfig` reads `config.json` and refuses, by name,
  every field the Tape tower does not implement. It cross-checks against
  tessl's own reader (`tessl::qwen35_model::Qwen35Config`) and tessl's kernel
  constants (`GDN_TRAIN_DK`, `GDN_TRAIN_BV`, `ATTN_TRAIN_HEAD_DIM`), which is
  why this crate still depends on tessl on macOS. `tape_spec()` feeds the
  Tape tower.
- **`names`**: `tower_tensors`, `Snapshot` and `check_header` list the tower's
  tensors and pre-flight a snapshot's header through ojas-io before any weight
  is read.
- **`groups`**: `OptimizerPlan` builds per-parameter learning-rate scales and
  weight decays from the caller's rules, with no built-in exclusions.
- **`state`**: the masters / `exp_avg` / `exp_avg_sq` shard format on disk
  (`plan_shards`, `write_kind`, `StateIndex`, `check_new_state_dir`). The
  provider that wrote and loaded it is gone; the format and its checked
  reader remain.

Off macOS the crate is empty: tessl builds only there.

## Tests

CPU tests (macOS, no GPU): `cpu_config`, `cpu_names`, `cpu_groups`,
`cpu_state_read`, `cpu_state_prevalidation`, `cpu_tape_spec` and
`cpu_surface` (the retirement pin: no source file names tessl's retired
training layer, and `src/step.rs` is absent). GPU tests: `gpu_tape_provider`
and `gpu_tape_2b` exercise the Tape provider on Metal;
`sharded_checkpoint_adversarial` covers the header pre-flight.
