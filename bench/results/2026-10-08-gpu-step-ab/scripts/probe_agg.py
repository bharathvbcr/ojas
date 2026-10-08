"""Per-variant table of step_probe JSONL (throwaway analysis).

For each (runtime, probe, variant): the min of the round minimums, the
median of the round medians, the spread (max / min of the round minimums),
and the ratio to the probe's first variant per round (median, range).
"""
import json
import statistics
import sys

d = {}
order = []
for path in sys.argv[1:]:
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") != "ok":
            if "probe" in r:
                print(f"! {r.get('runtime')} {r['probe']} {r.get('variant')}: {r.get('status')} {r.get('detail', '')}")
            continue
        key = (r["runtime"], r["probe"])
        if key not in order:
            order.append(key)
        d.setdefault(key, {}).setdefault(r["variant"], {})[r["round"]] = (r["min_ms"], r["median_ms"])

print("| runtime | probe | variant | min ms | median ms | spread | / first variant (median, range) |")
print("|:--|:--|:--|--:|--:|--:|:--|")
for key in order:
    variants = d[key]
    first = next(iter(variants.values()))
    for name, rounds in variants.items():
        mins = [v[0] for v in rounds.values()]
        meds = [v[1] for v in rounds.values()]
        rs = [rounds[k][0] / first[k][0] for k in rounds if k in first]
        rel = f"{statistics.median(rs):.3f} ({min(rs):.3f}-{max(rs):.3f})" if rs else "-"
        print(f"| {key[0]} | {key[1]} | {name} | {min(mins):.3f} | {statistics.median(meds):.3f} | {max(mins) / min(mins):.3f} | {rel} |")
