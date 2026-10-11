"""Summarize ab_decode.sh: per lane and row, the median over rounds of each
side's median_ms, after/before per round (median [min-max]), and decode ms
per token = (decode row - prefill row) / 31 per side. Analysis only."""
import glob, json, os, statistics, sys
out = sys.argv[1]
KEY = sys.argv[2] if len(sys.argv) > 2 else "median_ms"
rows = {}
for p in glob.glob(os.path.join(out, "round*", "*.jsonl")):
    r = int(os.path.basename(os.path.dirname(p))[5:])
    lane, side = os.path.basename(p)[:-6].split("-")
    for line in open(p):
        d = json.loads(line)
        if d.get("status") == "ok" and "median_ms" in d:
            rows.setdefault((lane, d["row"]), {}).setdefault(r, {})[side] = d[KEY]
        elif d.get("status") not in ("info", "ok"):
            print("PROBLEM", lane, side, r, d)
print(f"| lane | row | before {KEY} | after {KEY} | after / before per round, median [min-max] | rounds |")
print("| :-- | :-- | --: | --: | :-- | --: |")
per_tok = {}
for (lane, row) in sorted(rows):
    rs = rows[(lane, row)]
    both = [x for x in rs.values() if "before" in x and "after" in x]
    b = statistics.median(x["before"] for x in both)
    a = statistics.median(x["after"] for x in both)
    ratios = [x["after"] / x["before"] for x in both]
    print(f"| {lane} | {row} | {b:.2f} | {a:.2f} | {statistics.median(ratios):.3f} [{min(ratios):.3f}-{max(ratios):.3f}] | {len(both)} |")
print()
print("| lane | decode row | before ms / token | after ms / token | after - before ms / token, per round median [min-max] |")
print("| :-- | :-- | --: | --: | :-- |")
for lane in ("metal", "wgpu"):
    pre = rows.get((lane, "gen_prefill_p32"), {})
    for row in sorted(r for (l, r) in rows if l == lane and r != "gen_prefill_p32"):
        rs = rows.get((lane, row), {})
        tok = {s: [] for s in ("before", "after")}
        diffs = []
        for r in sorted(rs):
            if r not in pre:
                continue
            t = {s: (rs[r][s] - pre[r][s]) / 31 for s in ("before", "after") if s in rs[r] and s in pre[r]}
            for s, v in t.items():
                tok[s].append(v)
            if len(t) == 2:
                diffs.append(t["after"] - t["before"])
        print(f"| {lane} | {row} | {statistics.median(tok['before']):.3f} | {statistics.median(tok['after']):.3f} "
              f"| {statistics.median(diffs):+.3f} [{min(diffs):+.3f}-{max(diffs):+.3f}] |")
