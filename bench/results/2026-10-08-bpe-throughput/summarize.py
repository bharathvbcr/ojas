"""Tables for run.sh output (analysis only, never shipped).

    python3 summarize.py OUT_DIR BASE_LABEL

Per lane and row: the minimum over rounds of each process's min, the MB/s
and Mtok/s at that minimum, and the median over rounds of
(lane min / base min) within the same round. Digests must agree across lanes.
"""
import pathlib
import re
import statistics
import sys

out, base = pathlib.Path(sys.argv[1]), sys.argv[2]
runs = {}
for f in sorted(out.glob("r*_*.txt")):
    m = re.match(r"r(\d+)_(.+)\.txt", f.name)
    rnd, label = int(m.group(1)), m.group(2)
    text = f.read_text()
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 5 and cells[0].startswith(("encode", "decode")):
            rows[cells[0]] = (float(cells[1]), float(cells[3]), float(cells[4]))
    digest = re.search(r"digest=(\w+)", text).group(1)
    parity = re.search(r"tiktoken_parity=(\S+) documents_matched=(\S+)", text)
    runs.setdefault(label, {})[rnd] = (rows, digest, parity.groups())

digests = {d for lane in runs.values() for (_, d, _) in lane.values()}
parities = {p for lane in runs.values() for (_, _, p) in lane.values()}
print(f"digests: {sorted(digests)}  parity: {sorted(parities)}")
rounds = sorted(runs[base])
row_names = list(runs[base][rounds[0]][0])
print()
print("| lane | row | min s | MB/s | Mtok/s | lane/base (min of mins) | median of round ratios |")
print("| :-- | :-- | --: | --: | --: | --: | --: |")
for label, lane in runs.items():
    for row in row_names:
        mins = {r: lane[r][0][row] for r in lane}
        best = min(mins.values(), key=lambda v: v[0])
        base_best = min(runs[base][r][0][row][0] for r in runs[base])
        ratios = [mins[r][0] / runs[base][r][0][row][0] for r in mins if r in runs[base]]
        print(
            f"| {label} | {row} | {best[0]:.4f} | {best[1]:.2f} | {best[2]:.3f} | "
            f"{best[0] / base_best:.3f} | {statistics.median(ratios):.3f} |"
        )
