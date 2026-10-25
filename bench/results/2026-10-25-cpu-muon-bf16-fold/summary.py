"""Summarise muonab ops.txt: per op/dir, base vs new min-of-mins, per-round new/base ratio of min_ms."""
import re
import statistics
import sys
from collections import defaultdict

rows = defaultdict(dict)  # (op, dir) -> {(side, round): (min, med)}
pat = re.compile(r"(new|base)-(\d+)\.log:OJAS_OP op=(\S+) dir=(\S+) .*min_ms=([\d.]+) med_ms=([\d.]+)")
for line in open(sys.argv[1]):
    m = pat.search(line)
    if m:
        side, rnd, op, d, mn, md = m.groups()
        rows[(op, d)][(side, int(rnd))] = (float(mn), float(md))

print("| op | dir | base min-of-mins ms | new min-of-mins ms | base round spread | new round spread "
      "| per-round new/base (min_ms) | median ratio | new faster in |")
print("| :-- | :-- | --: | --: | --: | --: | :-- | --: | :-- |")
for (op, d), v in sorted(rows.items()):
    rounds = sorted({r for (_, r) in v})
    base = [v[("base", r)][0] for r in rounds if ("base", r) in v]
    new = [v[("new", r)][0] for r in rounds if ("new", r) in v]
    ratios = [v[("new", r)][0] / v[("base", r)][0] for r in rounds if ("new", r) in v and ("base", r) in v]
    faster = sum(1 for x in ratios if x < 1)
    print(f"| {op} | {d} | {min(base):.3f} | {min(new):.3f} | {max(base) / min(base) - 1:+.1%} | "
          f"{max(new) / min(new) - 1:+.1%} | {' '.join(f'{x:.3f}' for x in ratios)} | "
          f"{statistics.median(ratios):.3f} | {faster}/{len(ratios)} |")
