"""Per-round new/old ratios of the attention A/B (throwaway analysis)."""
import re
import statistics
import sys
from collections import defaultdict

rows = defaultdict(dict)  # (backend, shape, window) -> {(round, tree): fields}
for line in open(sys.argv[1]):
    if not line.startswith("AB "):
        continue
    f = dict(re.findall(r"(\w+)=(\S+)", line))
    key = (f["backend"], f["shape"], f["window"])
    rows[key][(int(f["round"]), f["tree"])] = {k: float(f[k]) for k in f if k[:3] in ("fwd", "bwd", "ctl")}

print("| backend | shape | metric | old min ms (best) | new min ms (best) | new/old per round (median, range) | control new/old (median, range) |")
print("|:--|:--|:--|--:|--:|:--|:--|")
for (be, shape, w), d in sorted(rows.items()):
    if w != "full":
        continue
    rounds = sorted({r for r, _ in d})
    for metric in ("bwd_min", "fwd_min"):
        rat, ctl = [], []
        for r in rounds:
            if (r, "old") in d and (r, "new") in d:
                rat.append(d[(r, "new")][metric] / d[(r, "old")][metric])
                ctl.append(d[(r, "new")]["ctl_min"] / d[(r, "old")]["ctl_min"])
        old = min(d[(r, "old")][metric] for r in rounds if (r, "old") in d)
        new = min(d[(r, "new")][metric] for r in rounds if (r, "new") in d)
        print(
            f"| {be} | {shape} | {metric[:3]} | {old:.3f} | {new:.3f} | "
            f"{statistics.median(rat):.3f} ({min(rat):.3f}-{max(rat):.3f}) | "
            f"{statistics.median(ctl):.3f} ({min(ctl):.3f}-{max(ctl):.3f}) |"
        )

print()
print("| backend | shape | window | bwd min ms (best of rounds) | fwd min ms (best) |")
print("|:--|:--|:--|--:|--:|")
for (be, shape, w), d in sorted(rows.items()):
    news = [v for (r, t), v in d.items() if t == "new"]
    if not news:
        continue
    print(f"| {be} | {shape} | {w} | {min(v['bwd_min'] for v in news):.3f} | {min(v['fwd_min'] for v in news):.3f} |")
