"""Old/new/torch tables for the GPU step-throughput A/B (throwaway analysis).

Reads <out>/round*/<lane>.jsonl as run_ab.sh writes them. Per row and lane:
the min of the round minimums, the median of the round medians, and the
spread (max / min of the round minimums). Per row and backend: new/old per
round on the round minimums (median and range over rounds), and torch / new
on the round medians, the paired bench's ratio (> 1: ojas faster).
"""
import glob
import json
import os
import statistics
import sys

out = sys.argv[1]
data = {}  # (row, lane) -> {round: (min, median)}
status = {}  # (row, lane) -> set of non-ok statuses
for path in sorted(glob.glob(os.path.join(out, "round*", "*.jsonl"))):
    rnd = int(os.path.basename(os.path.dirname(path))[5:])
    lane = os.path.basename(path)[:-6]
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = r.get("row", "")
        if row.startswith("_"):
            continue
        if r.get("status") == "ok":
            data.setdefault((row, lane), {})[rnd] = (r["min_ms"], r["median_ms"])
        else:
            status.setdefault((row, lane), set()).add(r.get("status", "?"))

rows = []
for (row, _lane) in data:
    if row not in rows:
        rows.append(row)


def side(row, lane):
    d = data.get((row, lane))
    if not d:
        return None
    mins = [v[0] for v in d.values()]
    meds = [v[1] for v in d.values()]
    return min(mins), statistics.median(meds), max(mins) / min(mins), d


def ratio(a, b, k):
    """Per-round b[k] / a[k] over rounds both have: median, min, max."""
    if not a or not b:
        return None
    rs = [b[3][r][k] / a[3][r][k] for r in a[3] if r in b[3]]
    if not rs:
        return None
    return statistics.median(rs), min(rs), max(rs)


def f(x):
    return "-" if x is None else f"{x:.3f}"


def fr(x):
    return "-" if x is None else f"{x[0]:.3f} ({x[1]:.3f}-{x[2]:.3f})"


print("| backend | row | old min ms | new min ms | new/old per round (median, range) | old spread | new spread | torch median ms | torch/new (median, range) |")
print("|:--|:--|--:|--:|:--|--:|--:|--:|:--|")
for be in ("metal", "wgpu"):
    for row in rows:
        o, n, t = side(row, f"old-{be}"), side(row, f"new-{be}"), side(row, "torch-mps")
        if not o and not n:
            continue
        print(
            f"| {be} | {row} | {f(o and o[0])} | {f(n and n[0])} | {fr(ratio(o, n, 0))} | "
            f"{f(o and o[2])} | {f(n and n[2])} | {f(t and t[1])} | {fr(ratio(n, t, 1))} |"
        )
bad = {k: v for k, v in status.items()}
if bad:
    print()
    print("Not timed (status per row and lane):")
    for (row, lane), st in sorted(bad.items()):
        print(f"- {row} / {lane}: {', '.join(sorted(st))}")
