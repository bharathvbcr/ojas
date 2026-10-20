import json, subprocess
D = "/Users/bharath/Code/research/ojas/backups/a6-tools"
dups = {
    "ft-29f7e4b6": "ft-562b0add (G-LIBM), ft-15ebd6b1 (Tape hardening), ft-bd6373d2 (C ABI/Go micro-batches), ft-3902a4b3 (Miri/TSan, toolchain)",
    "ft-dc4bfe22": "ft-f66c67c1, ft-4f11ce5a, ft-96023b34, ft-7f4f2d87, ft-64601da5 (vision chain)",
    "ft-e15ee76a": "ft-2c183b71 (clip_grad_norm fold), ft-69ef4ce1 (bf16 tier); the exact-f32 epilogue is tessl's (Fable 2026-10-16 ruling 5)",
    "ft-ee2a3753": "ft-b301b6d6 (G-IOREAD), ft-54e8ce27 (G-BUDGET), ft-2bbced18 (UMA zero-copy fold)",
    "ft-a95d2145": "ft-b90b30ae (gemma-metal build.rs), ft-301ddd4c (CI/hygiene); .devcouncil/config.yaml is an owner decision (w11 ruling 36)",
}
tasks = json.load(open(f"{D}/tasks.json"))["tasks"]
for pre, canon in dups.items():
    tid = next(t["item_id"] for t in tasks if t["item_id"].startswith(pre))
    args = {
        "repo_path": "/Users/bharath/Code/research/ojas", "task_id": tid, "status": "backlog",
        "summary": f"DUPLICATE (Fable ruling 2026-10-17, ojas-a6): imported from an untracked tasks/*.md bundle that the 2026-10-16 ruling said not to import. Canonical cards: {canon}. Work it there, not here. Not deleted or archived: archiving is the owner's call. Label/priority unchanged to avoid overwriting the card.",
    }
    open(f"{D}/dup-args.json", "w").write(json.dumps(args))
    out = subprocess.run(["bash", f"{D}/gp.sh", "gitpulse_complete_task", f"@{D}/dup-args.json", "8"], capture_output=True, text=True).stdout
    print(pre, out.strip()[:220])
