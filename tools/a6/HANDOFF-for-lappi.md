# ojas → Lappi handoff (ojas-a6, 2026-10-17)

From ojas session ojas-a6 to the Lappi orchestrator, who is not reachable by message; the owner relays. Context: the owner's goal is quantised (fake-quant q8 first) LoRA training of the decision model on the Mac.

## What ojas now has (on lane branches; merging into local ojas main once the owner's verification run is green)

| Piece | Where | What Lappi does with it |
|---|---|---|
| Value-only fused CE | `Tape::linear_cross_entropy_value` (ojas-autograd) | Lappi's scale-0 `RowTargets` (report-only full-vocab CE) should call it: no `[V,d]`/`[N,d]` gradient, no tape node. Treat a NaN scale as an error and -0.0 as 0 on Lappi's side. |
| Adapter checkpoint layout v1 | `docs/adapter-checkpoint.md`, `ojas_model::qwen35::adapters` | The converter (ft-58e316a9 / Lappi ft-2ed2fd55) binds to it. Host entries use Lappi's own names: `d17.delta`, `span_head.{start_proj,end_proj}.weight`, `span_head.abstain_{start,end}`. The base is one file, named with its sha256; `model.safetensors-00001-of-00001.safetensors` is accepted, and a sharded base is refused. **Ask:** which fields beyond the doc's does the converter need? |
| Merged bf16 export | `Qwen35Lora::merge_bf16` (ojas-model) | Yields (HF name, bf16) with one RNE rounding of W + (alpha/r)·B·A. Over a fake-quant base this is dequant(Q(W)), so both arms export the same way. **Confirm:** that rounding contract matches what the converter expects. |
| Fake-quant tiny fixture | `ojas-oracle/python/qwen35_fakequant_base.py` | Uses `mx.dequantize(*mx.quantize(W, group_size=64, bits=8), group_size=64, bits=8)` on exactly `model.layers.<n>.<suffix>.weight`, from a bf16 W, with bf16 scale and bias. **Confirm:** the Lappi gate tool uses the identical call, so the fixture and the 2B/0.8B base files match. |
| AdamW vs torch, 100 steps | `ojas-model/tests/adamw_torch_100.rs` | Uses the recipe's betas (0.9, 0.999), eps 1e-8, wd 0.01, lr 2e-4/1e-4. Step convention: `Backend::adamw_step`'s `step` is the count before the update. **Confirm:** Lappi's loop passes it the same way. |
| Memory pre-flight | `ojas_model::preflight` | Phase A wires it before step 0. It is calibrated from ojas `bench/results/2026-10-17-lora-step/rows.jsonl`. |
| ojas-train crate shell | `ojas-train/` (edition 2024, hand-rolled sha256) | Phase A imports into `ojas-train/src/lappi/`. Replace Lappi's `sha2` call sites with `ojas_train::sha256` (Fable: no sha2 in ojas). |

## UPDATE 2026-10-18: phase A is imported. Please freeze now

ojas imported the moving set **with history at your main `fecfd5d`** onto ojas branch `gp/a6-phase-a-import` (`62e0ec2`; Lappi `lib.rs` blob `962be359f`; 13 Lappi commits visible via `git log --follow ojas-train/src/lappi/trainer.rs`). Fable waived the freeze acknowledgement after checking that no Lappi worktree held edits to the moving files.
- **Keep frozen until the pin bump:** `crates/qd-train/src/{step,trainer,schedule,recipe,adamw,mock,run_control,pyjson}.rs` and `tests/{adamw_decay_sensitive,adamw_oracle,schedule_oracle,pyjson_oracle}.rs`, `tests/common/`, and the fixtures `trainer-oracle.json` and `adamw-decay-sensitive/`. Any further change to them goes through ojas-train.
- **ojas is rewiring now** (no new dependencies):
  - `serde_json` → `ojas_io::JsonValue`;
  - `sha2` → `ojas_io::sha256`;
  - `recipe.rs`'s `lora_param_groups` stops naming `crate::d17::D17_ENTRY` and `span_head::PARAMETER_NAMES`, and takes the host-entry names from the caller, so Lappi passes its constants;
  - `trainer_loop.rs` stays in Lappi (it imports `objective`).
- **After the owner's pin-bump yes:** Lappi depends on `ojas-train`, deletes its moved copies, and repoints `d17.rs`, `lora_init.rs`, `ledger.rs` and `trainer_loop.rs` imports to `ojas_train::lappi::…`.

## Phase A (the qd-train move): what ojas needs from Lappi

1. **The freeze.** When wave-l1 (`qd-train/src/{lib,objective,d17}.rs`) and wave-l2 (`lora_init.rs`, which imports `run_control`) have landed on Lappi main, post that sha. Then freeze `crates/qd-train/src/{step,trainer,schedule,recipe,adamw,mock,run_control,pyjson,lib}.rs` until ojas reports the import.
2. **The import.** The owner then runs `backups/a6-tools/owner-phase-a-import.sh <sha> --execute`. It uses git-filter-repo on a fresh clone and imports with history onto a branch, never main. Two test-support files are flagged as unsure because they import non-moving modules: `tests/pyjson_oracle.rs` (`ledger::isoformat_utc`) and `tests/common/`. Say whether they move.
3. **The pin.** Fable ruled the bump 5b67cdd → 04277f8 already covered by the owner's "merge all". The next bump is to the a6 merge sha, then to the phase-A sha. **Note:** two Lappi agent worktrees (agent-a68d…, agent-ac584…) both modify `upstream.lock.json`. Make sure only one commits the bump.

## Blockers on the Lappi side (ojas cannot do these)

- **Amendment 3:** v5r is withdrawn, so training needs the v6 shard rebuild. N=127 may need re-deriving if the width distribution changes.
- **Torch driver and gate tool:** GAP-LORA2B-TORCH-DRIVER-MISSING; GAP-LORA2B-QUANT-GATE-TOOL-MISSING (the 2B/0.8B fake-quant base files).
- **The mechanics check:** it needs a Lappi dump of about 60 train-split rows (rule 3) as u32 TokenBin plus lengths plus label mask, with sha256s. ojas does not read Lappi shards (that would bypass rule 3).
- **Recording owner decisions:** they are delegated to Fable on 2026-10-17. Record them as "Owner, 2026-10-17: 'ask fable for decisions'. Fable ruled: …":
  - item 5 is pre-approved by criteria;
  - item 6 is approved, with the reduced quantised 2B run after the bf16 row;
  - item 9 is peft "not run";
  - for item 8, ojas makes two doc fixes, then the owner runs Manvi verify.
