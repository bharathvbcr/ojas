"""Min over rounds of each OJAS_OP cell, base against after (analysis only).

    python3 summarize.py OUT_DIR > summary.md
"""
import collections
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
best = collections.defaultdict(dict)
for path in sorted(out.glob("ops-*-*.txt")):
    lane = path.stem.rsplit("-", 1)[1]
    for line in path.read_text().splitlines():
        if not line.startswith("OJAS_OP "):
            continue
        f = dict(re.findall(r"(\w+)=(\S+)", line))
        key = (f["op"], f["dir"], int(f["threads"]))
        ms = float(f["min_ms"])
        best[key][lane] = min(best[key].get(lane, ms), ms)

print("| op | dir | threads | base ms | after ms | after/base |")
print("| :-- | :-- | --: | --: | --: | --: |")
for (op, d, t), lanes in sorted(best.items()):
    b, a = lanes.get("base"), lanes.get("after")
    ratio = f"{a / b:.2f}" if a and b else "-"
    print(f"| {op} | {d} | {t} | {b:.3f} | {a:.3f} | {ratio} |")
