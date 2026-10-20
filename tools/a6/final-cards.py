"""Log the verified state on the three cards (status stays in_progress: their criteria are not all met)."""
import json, subprocess
D = "/Users/bharath/Code/research/ojas/backups/a6-tools"
base = ("ojas-a6, 2026-10-18 09:15: LOCAL main 94aa3af is VERIFIED (logs: backups/a6-tools/logs/, rerun1..rerun7). "
        "First run: build OK; ojas-io+ojas-train 134/134; autograd 33/33; full ojas-io/train/autograd/model 614 pass; Metal lib 81/81; "
        "qwen35 4/4; GPU LoRA step 1/1; sha256 bench 326-331 MiB/s (> Fable's 150 MB/s bar, so no sha2 dependency). "
        "It found 6 real test failures, fixed in 43d5e82 (preflight overflow mapped to a refusal), 0a17930 (three stale tests), aaed8b3 (Metal scale_grad in place) and 111097d. "
        "Re-runs: model 137/137, Metal 106/106, then 124/124 including clip_multi after merging gp/clip-sq-norm (73cfe02) on tessl a37a0c2; Linux check (aarch64) OK; "
        "clippy -D warnings --keep-going over ojas-io/train/autograd/model/metal/qwen35 --all-targets clean after d0fbe97, 0e95fe7, 81a81ca and 94aa3af "
        "(rust 1.99 lints: as_chunks, a boxed AdapterError quantizer record, is_multiple_of, a single-element loop, a useless vec!). Nothing pushed (main is 172 commits ahead of origin). ")
cards = {
    "ft-b5e64557f88bdb433700db571b7a78ec": base + "This card: criteria 7 (AdamW vs torch, 100 steps), 10 (preflight), 14/16/18/19 seeds and 17 (value-only CE) are now verified on main. Phase A: Fable GO at Lappi freeze fecfd5d; the owner runs backups/a6-tools/owner-phase-a-import.sh from worktree ../ojas-phase-a (gate branch gp/a6-train recreated at ffc6f2e).",
    "ft-a1a4ba4d06d18ef307fa7776545e9a52": base + "This card: the criterion-8 fake-quant q8 fixture and the fq8 LoRA tests (CPU and Metal) pass on main; dequant(Q(W)) export via merge_bf16 is verified.",
    "ft-2bbced18f11d70727f58e25696dcd5ed": base + "This card: criterion 2 (freeze_checked) is verified, including the scale_grad fix; criterion 3 was already met. Criteria 1 and 4 still need the GPU A/B rows: run backups/a6-tools/owner-bench-stepcost.sh with WT=/Users/bharath/Code/research/ojas and a fresh target dir.",
}
for tid, summary in cards.items():
    args = {"repo_path": "/Users/bharath/Code/research/ojas", "task_id": tid, "status": "in_progress", "summary": summary}
    open(f"{D}/card-args.json", "w").write(json.dumps(args))
    out = subprocess.run(["bash", f"{D}/gp.sh", "gitpulse_complete_task", f"@{D}/card-args.json", "10"], capture_output=True, text=True).stdout
    print(tid[:11], out.strip()[:120])
