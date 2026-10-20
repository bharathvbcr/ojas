"""File the gap cards from ojas-a6's wrap-up (2026-10-19). Prints each response; a related_tasks_exist
refusal is printed for review (resend with reviewed_related)."""
import json, subprocess, sys
D = "/Users/bharath/Code/research/ojas/backups/a6-tools"
OJAS = "/Users/bharath/Code/research/ojas"
LAPPI = "/Users/bharath/Code/research/Lappi-decision"
SRC = "Filed by ojas-a6 at wrap-up, 2026-10-19 (ojas main 4123dfb = origin/main). Evidence: ojas BUILD_REPORT-a6.md and backups/a6-tools/logs/."

cards = [
 dict(repo_path=OJAS, title="TapeStepProvider follow-ups: Full-mode scale-0 [V,d] zeros, the bank emptied before AdamW validation, double gradient download per step, dry_run that sees only charged allocations",
  kind="improvement", priority=2, severity="medium", status="backlog",
  labels=["fine-tune", "qwen35", "trainer", "follow-up"],
  description=SRC + " Follow-ups the phase-B lanes and adversarial passes recorded on ojas_model::qwen35::TapeStepProvider (merged fb057d4, verified 95/95 CPU and 5/5 Metal) but left out of scope.",
  acceptance_criteria=[
   "Full-mode accumulate at scale 0 allocates no [V,d] zero gradient: the bank takes a 'zero' marker or the entry is skipped, with a Budget-delta test (LoRA mode already allocates none).",
   "adamw_step validates gradients (finite, shapes) before take_grads moves them out of the bank, so a refused update keeps the sums; a test injects a refusal and checks the bank is intact.",
   "grad_sq_norm and the all-or-nothing pass 1 download each gradient once per step (cache or a fused check); an interleaved A/B, min-of-5 at 2B T=2,048 and 9,638 shows the step time does not regress (>5% needs a written justification).",
   "dry_run documents and tests that allocations outside the rebound Budget are invisible; the Metal preflight uses DeviceHooks.rebind so the measured peak is the real one (a test with rebind None refuses, which exists, and one with rebind Some measures more than 0).",
   "CheckpointContext's seed/epoch → rng_state hex encoding is documented in docs/adapter-checkpoint.md and pinned by a round-trip test."],
  planned_files=["ojas-model/src/qwen35/step_provider.rs", "ojas-model/src/qwen35/bank.rs", "ojas-model/tests/qwen35_step_provider.rs", "docs/adapter-checkpoint.md"]),
 dict(repo_path=OJAS, title="ojas-train phase-A residuals: move groups.rs from ojas-qwen35, dedup AdamWHyper/clip_coefficient, re-home trainer_loop.rs, reword card ft-b5e64557's stale criteria",
  kind="refactor", priority=2, severity="low", status="backlog",
  labels=["fine-tune", "trainer", "refactor", "follow-up"],
  description=SRC + " Phase A imported Lappi qd-train into ojas-train/src/lappi with history (Lappi fecfd5d) and rewired it without new dependencies. These pieces were deferred.",
  acceptance_criteria=[
   "ojas-qwen35/src/groups.rs (Select, LrRule, WdRule, GroupSpec, OptimizerPlan) moves into ojas-train as the one per-entry table, behind a small trait in place of TensorSpec/Qwen35Error; ojas-qwen35 re-exports it and its cpu_groups tests pass unchanged.",
   "ojas-qwen35's duplicate AdamWHyper and clip_coefficient (step.rs ~:259, :280) are deleted in favour of ojas_train::lappi::adamw, with callers rewired and tests green.",
   "Lappi's trainer_loop.rs test (excluded from the import because it imports qd_train::objective) is re-homed against the mock provider in ojas-train, or recorded as staying in Lappi with the reason.",
   "Card ft-b5e64557 criteria 4, 9, 14, 16 and 19 are reworded to Fable's 2026-10-18 ruling (the loop's Objective owns D17 and the span head; provider/ holds LoRA only) and criterion 3's 'accumulate may take several sequences' is struck (Fable: one sequence per call)."],
  planned_files=["ojas-train/src/", "ojas-qwen35/src/groups.rs", "ojas-qwen35/src/step.rs", "ojas-qwen35/src/lib.rs"]),
 dict(repo_path=OJAS, title="Committed, gate-aware verification harness: the untracked backups/a6-tools scripts become a tracked tool with disk, load and cargo-slot gates; re-run gpu_parity's 2B test once tessl's fixture is regenerated",
  kind="infra", priority=2, severity="medium", status="backlog",
  labels=["ci", "verification", "tooling"],
  description=SRC + " Every ojas verification in this run went through untracked scripts in backups/a6-tools (launch-rerun.sh, owner-verify-rerun.sh, phase-b-verify.sh, the commit helpers), because the MANVI gate refuses mac_heavy.sh from a session. They hit three bugs (see memory verify-script-pitfalls) and a full disk twice.",
  acceptance_criteria=[
   "A tracked tool (e.g. tools/verify/, Go per the language policy, or a bounded shell entry) runs a named step list under Lappi-decision/tools/mac_heavy.sh with an explicit MAC_HEAVY_MAX_LOAD, a per-run CARGO_TARGET_DIR, a perl or Go deadline per step, a pid file and DONE sentinel, and a SUMMARY with passed/failed counts per step.",
   "It refuses to start a build when `df` free space on the data volume is under a stated floor (50 GiB for a cold build), and records the reason, instead of risking ENOSPC.",
   "Its target-dir keep-alive touches only an existing directory and each step refuses a non-directory target path (regression test for the 0-byte-file bug of 2026-10-18).",
   "It reports exit 142 alarms that expired while waiting for the mac_heavy lock or a cargo slot as 'not run', separately from failures.",
   "gpu_parity's gpu_real_2b_single_step_matches_the_transformers_reference is re-run once tessl/target/qwen35_train_ref is regenerated (owner command under mac_heavy: tools/qwen35_ref/make_train_fixture.py 2b --tokens 128 in tessl), and passes on ojas main."],
  planned_files=["tools/verify/", "docs/"]),
 dict(repo_path=LAPPI, title="Switch qd-train onto ojas-train at ojas 4123dfb: bump the pin, depend on ojas-train, delete the moved copies, repoint d17/lora_init/ledger/trainer_loop (phase A's Lappi half)",
  kind="refactor", priority=1, severity="high", status="ready",
  labels=["fine-tune", "lora-2b", "ojas", "trainer"],
  description="Filed by ojas session ojas-a6, 2026-10-19. ojas imported qd-train's moving set {step, trainer, schedule, recipe, adamw, mock, run_control, pyjson} with history at Lappi fecfd5d into ojas-train/src/lappi, rewired without new dependencies, merged and verified on ojas main 4123dfb (on ojas origin/main). Owner decision 2026-10-19 ('choose the best'): pin bump YES. Contract details: ojas backups/a6-tools/LAPPI-WORK-ORDER.md L0 and L5.",
  acceptance_criteria=[
   "upstream.lock.json pins ojas 4123dfb with a delta report and `make pin` PASS; the owner's 2026-10-19 decision is recorded in Lappi in their words.",
   "crates/qd-train depends on ojas-train; the eight moved modules and their moved tests are deleted from Lappi in the same change.",
   "d17.rs, lora_init.rs, ledger.rs and trainer_loop.rs import from ojas_train::lappi::…; lora_param_groups and lora_optimizer_table are called with &LoraHostNames { d17_entry: d17::D17_ENTRY, span_head: &span_head::PARAMETER_NAMES }.",
   "Moved code's sha2 and serde_json uses are gone (ojas uses ojas_io::sha256 and ojas_io::JsonValue); Lappi's own remaining uses are unaffected.",
   "Lappi's qd-train, qd-train-metal and the door tests pass; the moving set stays frozen in Lappi until this lands."],
  planned_files=["upstream.lock.json", "Cargo.toml", "crates/qd-train/"]),
 dict(repo_path=LAPPI, title="LoRA-2B data: rebuild the train/val shard set under the v6 data rules (amendment 3 withdrew v5r) and re-derive amendment 1's N if widths change",
  kind="data", priority=1, severity="high", status="ready",
  labels=["fine-tune", "lora-2b", "data", "rule-3"],
  description="Filed by ojas session ojas-a6, 2026-10-19. campaign/lora-2b-preregistered.json amendment 3 (~:260-261) withdrew the v5r shards (MMLU test/dev and CLINC test items in train; the dedupe keep-rule gap). Every LoRA-2B run (the 0.8B smoke first) is blocked on this. ojas never reads Lappi shards (rule 3).",
  acceptance_criteria=[
   "A shard set rebuilt under the v6 data rules, with no held-out or val rows in train (rule 3), MMLU/CLINC test items removed, and recorded shard hashes, sequence and token counts and max width.",
   "The LoRA-2B pre-registration's data section points at it by hash, committed before any result is read.",
   "Amendment 1's N is re-derived from ojas's measured LoRA rate (261.6 tok/s at T=9,638, ojas bench/results/2026-10-17-lora-step) if the width distribution changes, and recorded before step 0.",
   "Coordinate with any existing v6-rebuild card (one cap table over every pool) rather than building a second pipeline."],
  planned_files=["campaign/", "data/"]),
]

only = {a for a in sys.argv[1:] if not a.startswith("--")}
for i, c in enumerate(cards):
    if only and str(i) not in only:
        continue
    open(f"{D}/card-args.json", "w").write(json.dumps(c))
    out = subprocess.run(["bash", f"{D}/gp.sh", "gitpulse_add_task", f"@{D}/card-args.json", "12"], capture_output=True, text=True).stdout
    if "related_tasks_exist" in out and "--show" in sys.argv:
        print(f"[{i}] RELATED (not filed):\n{out.strip()[:3000]}\n")
        continue
    if "related_tasks_exist" in out:
        import re
        ids = list(dict.fromkeys(re.findall(r"ft-[0-9a-f]{32}", out)))[:25]
        print(f"[{i}] reviewed {len(ids)} related: {', '.join(x[:11] for x in ids)}")
        c = dict(c, reviewed_related=ids)
        open(f"{D}/card-args.json", "w").write(json.dumps(c))
        out = subprocess.run(["bash", f"{D}/gp.sh", "gitpulse_add_task", f"@{D}/card-args.json", "12"], capture_output=True, text=True).stdout
    print(f"[{i}] {c['title'][:60]}\n    {out.strip()[:300]}\n")
