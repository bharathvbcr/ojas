# Lappi work order: what blocks LoRA-2B training (bf16 and fake-quant q8) on the Mac

From ojas session ojas-a6, 2026-10-18. Give this to a session rooted in `/Users/bharath/Code/research/Lappi-decision`. ojas cannot do Lappi work: its gate forbids editing another repo, and the owner's rule is to hand such steps to a session rooted there. The longer context is in `backups/a6-tools/HANDOFF-for-lappi.md` (same folder).

The order follows Fable's 2026-10-17 run order: LoRA rows (done) → 0.8B bf16 smoke → amendment 1 → reduced 2B bf16 → quantised 0.8B smoke → reduced quantised 2B.

## L1. v6 shard rebuild (blocks every run)
- **Why:** amendment 3 (`campaign/lora-2b-preregistered.json:260-261`) withdrew the v5r train and val shards, because they contain MMLU test/dev and CLINC test items (GAP-DECISION-INDEX-PANEL-CLINC-AND-MMLU-TEST-ITEMS-ARE-LAPPI-TRAINING-DATA-2026-10-03), plus the dedupe keep-rule gap.
- **Do:** rebuild under the v6 data rules, keep rule 3 (no held-out or val rows in training), record shard hashes, and re-derive amendment 1's N if the width distribution changes. N = 127 came from ojas's 261.6 tok/s at T = 9,638 (`ojas bench/results/2026-10-17-lora-step/rows.jsonl`, row 4).
- **ojas contract:** none. ojas never reads Lappi shards; that would bypass rule 3.

## L2. Plain-torch LoRA reference arm driver (ft-2ed2fd55 crit 1; GAP-LORA2B-TORCH-DRIVER-MISSING)
- **Do:** the driver in `python/qd_train`, with the same shards, supervision, init, schedule and batch order as the ojas arm. No peft or bitsandbytes (D7).
- **ojas contract:**
  - AdamW betas (0.9, 0.999), eps 1e-8, decoupled wd 0.01; lr 2e-4 for adapters, 1e-4 for D17 and the span head.
  - `step` passed to the optimiser is the count *before* the update.
  - ojas pins this against torch over 100 steps (`ojas-model/tests/adamw_torch_100.rs`; measured fp32/fp64 gap 5.69e-6, bound 2.3e-5).
  - One token-budget batch per step, sequences fed one per launch, gradients summed (Overwrite, then Add).
  - The report-only full-vocab CE (scale 0) is value-only: ojas `Tape::linear_cross_entropy_value`. Treat a NaN scale as an error and -0.0 as 0.

## L3. Quantisation gate tool and fake-quant base files (ft-2ed2fd55 crit 5; GAP-LORA2B-QUANT-GATE-TOOL-MISSING)
- **Do:**
  - Measure mlx q8 and q4 against bf16 on 500 or more Lappi validation prompts (never held-out). Admit a format only at argmax agreement ≥ 99.0% and mean |Δ log-prob| ≤ 0.02 nats.
  - Then write the fake-quant 2B-Base and 0.8B-Base files: the 150 wrapped linears as `dequantize(quantize(W))`; tied table, norms, conv, A_log, dt_bias and in_proj_a/b unquantised.
- **ojas contract (must match exactly):**
  - The call is `mx.dequantize(*mx.quantize(W, group_size=64, bits=8), group_size=64, bits=8)`, from a bf16 W, with bf16 scales and biases, on exactly `model.layers.<n>.<suffix>.weight` (ojas `ojas-oracle/python/qwen35_fakequant_base.py`; mlx 0.31.2).
  - Write a manifest with the mlx version, the call and sha256 per file.
  - The base file name and sha256 go into the q8 sibling pre-registration and into every checkpoint. ojas's `load_hf_pinned` refuses a mismatch, and accepts a one-of-one name such as `model.safetensors-00001-of-00001.safetensors`.

## L4. ojas adapter checkpoint → `lora-merged` converter (GAP-LORA-CKPT-LAYOUT-NOT-OJAS-VERIFIED; ft-58e316a9)
- **ojas contract:** layout v1 in ojas `docs/adapter-checkpoint.md`.
  - Files: `adapters.safetensors` plus `moments.safetensors`. Entries are PEFT names `<module>.lora_A.weight` / `.lora_B.weight`.
  - Host entries use **Lappi's** names: `d17.delta`, `span_head.{start_proj,end_proj}.weight`, `span_head.abstain_{start,end}`.
  - Metadata: step, rng_state, consumed digest, base file name and sha256, quantizer record, rank and alpha.
  - Integrity: an FNV-1a 64 hash over the tensors and every metadata pair, and the moments file bound to its adapters file.
- **Merged weights:** ojas `Qwen35Lora::merge_bf16` yields (HF name, bf16) with W + (alpha/r)·B·A summed in f32 and rounded once (RNE). Over a fake-quant base this is dequant(Q(W)) automatically. Lappi writes the files.
- **Ask back:** any field the converter needs that layout v1 lacks.
- **Two-part checkpoint (Fable ruling, 2026-10-18):** the converter reads one `ckpt-<step>/` of the moved loop (format `qd-train-checkpoint-v2`):
  - `provider/adapters.safetensors` plus `moments.safetensors`: layout v1, **LoRA entries only**, with base identity and rank/alpha;
  - `trainer.json` `host_entries` plus `host.f32`: D17 and span head, with values, moments and per-entry AdamW step counts.

  The objective, not the Tape provider, owns D17 and the span head. This is Lappi's own moved format, and ojas documents its host part.

## L5. Move qd-train onto ojas-train (after the owner's pin-bump yes)
- **Status (updated 2026-10-18 14:45):** **merged and verified on ojas main `4123dfb`** (merge `fb057d4`), imported with history at Lappi `fecfd5d`.
  - ojas-train now exports `ojas_train::lappi::{step, trainer, schedule, recipe, adamw, mock, run_control, pyjson}`, plus `LoraHostNames`.
  - The ojas Tape `StepProvider` (`ojas_model::qwen35::TapeStepProvider`) is in place.
  - Verified by selected suites only (run 9, `docs/reports/BUILD_REPORT-a6.md:51-58`): qwen35 CPU 42/42, provider CPU 14/14, Metal provider 5/5, clippy clean, ojas-io + ojas-train 180/180, gpu_parity 9/10 (the one failure is environmental: a missing `tessl/target/qwen35_train_ref/ids.npy`). The done gate was not run (`BUILD_REPORT-a6.md:553`).
  - **Pin target: ojas `4123dfb`.** It is on ojas origin/main as of 2026-10-19, so the lock's url can serve it.
- **Keep frozen until then:** `crates/qd-train/src/{step,trainer,schedule,recipe,adamw,mock,run_control,pyjson}.rs`, `tests/{adamw_decay_sensitive,adamw_oracle,schedule_oracle,pyjson_oracle}.rs`, `tests/common/`, and the two fixtures.
- **Then:** bump `upstream.lock.json` to the ojas main sha that contains phase A (ojas will post it), depend on `ojas-train`, delete the moved copies, and repoint `d17.rs`, `lora_init.rs`, `ledger.rs` and `trainer_loop.rs` to `ojas_train::lappi::…`. `lora_param_groups` and `lora_optimizer_table` now take `&LoraHostNames { d17_entry: d17::D17_ENTRY, span_head: &span_head::PARAMETER_NAMES }`. Replace `sha2` and `serde_json` uses that moved.

## L0. Owner decisions, 2026-10-19 ("choose the best"; chosen by ojas-a6, following Fable's earlier rulings)

Record these in Lappi as "Owner, 2026-10-19: 'choose the best'. Chosen: …":
1. **Pin bump: YES.** Bump Lappi's `upstream.lock.json` ojas pin to **`4123dfb`** (verified, on ojas origin/main), via the usual delta report and `make pin`. Then do L5 (switch qd-train onto ojas-train).
2. **q8 sibling pre-registration:** approved by criteria (Fable's item 5). Draft it exactly to the agreed list and commit it before any quantised result is read. Anything outside the list comes back to the owner.
3. **GPU time:** approved in Fable's order. The 0.8B bf16 smoke runs first, as soon as L1 (v6 shards) and L2 (torch driver) exist. The quantised runs follow only after the reduced 2B bf16 ledger row exists. The owner launches each run.
4. **Stage 2 (resident q8 base):** not now. Revisit after the fake-quant reduced 2B run is scored. tessl's `nn::gemm_q4_mlx_tensorops` is noted as a candidate.

## L6. Owner decisions to record in Lappi (delegated to Fable 2026-10-17)
Record as "Owner, 2026-10-17: 'ask fable for decisions'. Fable ruled: …":
- item 5: the q8 sibling pre-registration is pre-approved by criteria;
- item 6: GPU time for the quantised arm is approved, with the reduced quantised 2B run only after the bf16 row exists;
- item 9: peft stays "not run";
- the ojas pin bump 5b67cdd → 04277f8 is covered by the owner's "merge all".
