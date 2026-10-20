"""Wrap-up logs on the three a6 cards (status stays in_progress: their criteria are not all met)."""
import json, subprocess
D = "/Users/bharath/Code/research/ojas/backups/a6-tools"
head = ("ojas-a6 wrap-up 2026-10-19: ojas main 4123dfb = origin/main (pushed by the owner), with phase A+B merged. "
        "Last verification, run 10 against tessl a8fc965: qwen35 CPU 42/42, gpu_parity 9/10 (only the 2B fixture missing after the disk cleanup; "
        "regenerate tessl/target/qwen35_train_ref under mac_heavy, then re-run), Tape provider CPU 14/14 and Metal 5/5, clippy clean. Report: BUILD_REPORT-a6.md. ")
cards = {
    "ft-b5e64557f88bdb433700db571b7a78ec": head + "Open on this card: criterion 5 (Tape vs Lappi's tessl provider, after Lappi switches qd-train to ojas-train pinned at 4123dfb), criterion 6 memory rows (the tape-provider arm in gpu_memory_rows; needs a GPU window), criteria 1/2 Lappi side (the work order's L5), criteria 11-13 (vision toy, schedules).",
    "ft-a1a4ba4d06d18ef307fa7776545e9a52": head + "Open on this card: the Lappi quant gate tool and the 2B/0.8B fake-quant base files (work order L3), the q8 sibling pre-registration (owner approval, pre-approved by criteria), and the quantised smoke and reduced 2B runs after the bf16 rows. Stage 2 (a resident q8 base; tessl's new nn::gemm_q4_mlx_tensorops is a candidate kernel family) waits for the owner's yes.",
    "ft-2bbced18f11d70727f58e25696dcd5ed": head + "Open on this card: criteria 1 and 4 need the GPU A/B rows (backups/a6-tools/owner-bench-stepcost.sh with WT = the ojas checkout and a fresh target dir; about 3-5 h of GPU time).",
}
for tid, summary in cards.items():
    open(f"{D}/card-args.json", "w").write(json.dumps({"repo_path": "/Users/bharath/Code/research/ojas", "task_id": tid, "status": "in_progress", "summary": summary}))
    out = subprocess.run(["bash", f"{D}/gp.sh", "gitpulse_complete_task", f"@{D}/card-args.json", "10"], capture_output=True, text=True).stdout
    print(tid[:11], out.strip()[:90])
