"""Aggregate a bench/run_paired.sh output directory into markdown tables.

Analysis only. Usage: aggregate.py OUT_DIR  (prints markdown on stdout)

Per row and ojas backend, each round gives one ojas median and one torch
median. ratio = torch_median / ojas_median for that round, so > 1 means ojas
was faster. Reported: ojas and torch min (min over rounds of the per-round
minimum) and median (median of the per-round medians), the median of the
per-round ratios with its min and max, and the within-run spread of each
side (max / min of its per-round medians). A row whose spread exceeds 10% on
either side is flagged "noisy - not quoted". A row that is missing or not
"ok" on either side in any round is reported with its status, never dropped.
"""

import glob
import json
import os
import statistics
import sys

SPREAD_LIMIT = 0.10
OJAS = ("ojas-metal", "ojas-wgpu")
TORCH = "torch-mps"


def load_rounds(out):
    rounds = {}
    for path in sorted(glob.glob(os.path.join(out, "round*", "*.jsonl"))):
        r = int(os.path.basename(os.path.dirname(path))[5:])
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rounds.setdefault(r, {})[(d["runtime"], d["row"])] = d
    return rounds


def fmt(x):
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:.1f}"
    if x >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def row_names(rounds, rt):
    names = []
    for rd in rounds.values():
        for (r, n) in rd:
            if r == rt and not n.startswith("_") and n not in names:
                names.append(n)
    return names


def summarize(rounds, rt, row, torch_row):
    o_meds, t_meds, o_mins, t_mins, ratios, parity, rel, problems = [], [], [], [], [], [], [], []
    for r in sorted(rounds):
        rd = rounds[r]
        o, t = rd.get((rt, row)), rd.get((TORCH, torch_row))
        if o is None:
            crash = rd.get((rt, "_process"))
            problems.append(f"r{r} ojas: {'crash exit ' + str(crash['exit']) if crash else 'NO DATA'}")
            continue
        if o.get("parity_max_abs") is not None:
            parity.append(o["parity_max_abs"])
            rel.append(o["parity_rel"])
        if o["status"] != "ok":
            problems.append(f"r{r} ojas {o['status']}: {o.get('detail', '')}"[:300])
            continue
        if t is None or t["status"] != "ok":
            problems.append(f"r{r} torch: {'NO DATA' if t is None else t['status'] + ' ' + t.get('detail', '')}"[:300])
            continue
        o_meds.append(o["median_ms"])
        t_meds.append(t["median_ms"])
        o_mins.append(o["min_ms"])
        t_mins.append(t["min_ms"])
        ratios.append(t["median_ms"] / o["median_ms"])
    s = {"row": row, "torch_row": torch_row, "problems": problems,
         "parity": max(parity) if parity else None, "rel": max(rel) if rel else None,
         "n": len(ratios)}
    if ratios:
        s.update(o_min=min(o_mins), o_med=statistics.median(o_meds),
                 t_min=min(t_mins), t_med=statistics.median(t_meds),
                 ratio=statistics.median(ratios), rmin=min(ratios), rmax=max(ratios),
                 o_spread=max(o_meds) / min(o_meds) - 1, t_spread=max(t_meds) / min(t_meds) - 1)
        s["noisy"] = s["o_spread"] > SPREAD_LIMIT or s["t_spread"] > SPREAD_LIMIT
    return s


def table(rows, rt):
    out = [f"### {rt} vs {TORCH}", "",
           "| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |",
           "| :-- | --: | --: | :-- | :-- | :-- | :-- |"]
    for s in rows:
        par = "-" if s["parity"] is None else f"{s['parity']:.2e} ({s['rel']:.1e})"
        name = s["row"] if s["torch_row"] == s["row"] else f"{s['row']} vs {s['torch_row']}"
        if s["n"] == 0:
            out.append(f"| {name} | - | - | - | - | **{'; '.join(s['problems'])}** | {par} |")
            continue
        flag = "noisy - not quoted" if s["noisy"] else ""
        if s["problems"]:
            flag = (flag + "; " if flag else "") + f"partial ({s['n']} rounds): " + "; ".join(s["problems"])
        out.append(
            f"| {name} | {fmt(s['o_min'])} / {fmt(s['o_med'])} | {fmt(s['t_min'])} / {fmt(s['t_med'])} "
            f"| {s['ratio']:.2f}x [{s['rmin']:.2f}-{s['rmax']:.2f}] "
            f"| {100 * s['o_spread']:.0f}% / {100 * s['t_spread']:.0f}% | {flag} | {par} |")
    return out


def main():
    out = sys.argv[1]
    rounds = load_rounds(out)
    lines = [f"# Paired GPU-vs-torch summary: {out}", "",
             f"{len(rounds)} rounds. ratio = torch median / ojas median per round "
             "(> 1: ojas faster). Spread = max/min of that side's per-round medians.", ""]
    env = os.path.join(out, "env.txt")
    if os.path.exists(env):
        lines += ["```", open(env).read().rstrip(), "```", ""]
    torch_rows = row_names(rounds, TORCH)
    ranked = []
    for rt in OJAS:
        rows = []
        names = row_names(rounds, rt)
        for n in names:
            rows.append(summarize(rounds, rt, n, n))
            if n + "_bf16" in torch_rows:
                rows.append(summarize(rounds, rt, n, n + "_bf16"))
        for n in torch_rows:
            if n not in names and not n.endswith("_bf16"):
                rows.append({"row": n, "torch_row": n, "n": 0, "parity": None, "rel": None,
                             "problems": [f"NO DATA: row absent from {rt}"]})
        lines += table(rows, rt) + [""]
        for s in rows:
            if s["n"] and not s["row"].startswith("floor"):
                ranked.append((s["ratio"], rt, s))
    ranked.sort(key=lambda x: x[0])
    lines += ["### Ranked: where ojas is slowest relative to torch (lowest ratio first)", "",
              "| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |",
              "| --: | :-- | :-- | :-- | --: | --: | :-- |"]
    for i, (ratio, rt, s) in enumerate(ranked, 1):
        name = s["row"] if s["torch_row"] == s["row"] else f"{s['row']} vs {s['torch_row']}"
        lines.append(f"| {i} | {rt} | {name} | {ratio:.2f}x [{s['rmin']:.2f}-{s['rmax']:.2f}] "
                     f"| {fmt(s['o_med'])} | {fmt(s['t_med'])} | {'noisy' if s['noisy'] else ''} |")
    load = os.path.join(out, "load.jsonl")
    if os.path.exists(load):
        lines += ["", "### Machine load around each runtime block", "",
                  "| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |",
                  "| --: | :-- | :-- | :-- | :-- | --: |"]
        for line in open(load):
            d = json.loads(line)
            lines.append(f"| {d['round']} | {d['lane']} | {d['when']} | {d['time']} | {d['load']} | {d['gpu_util_pct']} |")
    lines += per_row_load(out)
    print("\n".join(lines))


def per_row_load(out):
    """Range of the per-row `_load` records (uptime 1-min load and the ioreg
    GPU sample, before and after every row) per round and lane."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "round*", "*.jsonl"))):
        r = int(os.path.basename(os.path.dirname(path))[5:])
        lane = os.path.basename(path)[:-6]
        loads, gpus = [], []
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("row") != "_load":
                continue
            try:
                loads.append(float(d["load"].replace(",", " ").split()[0]))
            except (ValueError, IndexError):
                pass
            if d.get("gpu_util_pct") is not None:
                gpus.append(d["gpu_util_pct"])
        if loads or gpus:
            rows.append((r, lane, len(loads), loads, gpus))
    if not rows:
        return []
    lines = ["", "### Load recorded before and after every row", "",
             "| round | lane | samples | 1-min load min-max | GPU util % min / median / max |",
             "| --: | :-- | --: | :-- | :-- |"]
    for r, lane, n, loads, gpus in rows:
        lr = f"{min(loads):.1f}-{max(loads):.1f}" if loads else "-"
        gr = (f"{min(gpus)} / {statistics.median(gpus):.0f} / {max(gpus)}" if gpus else "-")
        lines.append(f"| {r} | {lane} | {n} | {lr} | {gr} |")
    return lines


def cross_run(outs):
    """Several run directories: per row, each run's median ratio [min-max]
    and a verdict over every round of every run. "slower in all N" / "faster
    in all N" holds only when every per-round ratio is on one side of 1."""
    runs = [load_rounds(o) for o in outs]
    torch_rows = row_names(runs[0], TORCH)
    labels = [chr(ord("A") + i) for i in range(len(outs))]
    lines = ["# Cross-run direction check", ""]
    lines += [f"- run {l}: {o}" for l, o in zip(labels, outs)] + [""]
    for rt in OJAS:
        lines += [f"### {rt} vs {TORCH}", "",
                  "| row | " + " | ".join(f"run {l} median [min-max]" for l in labels)
                  + " | all rounds min-max | verdict |",
                  "| :-- | " + " | ".join(":--" for _ in labels) + " | :-- | :-- |"]
        for n in row_names(runs[0], rt):
            for tr in [n] + ([n + "_bf16"] if n + "_bf16" in torch_rows else []):
                cells, all_r, missing = [], [], 0
                for rd in runs:
                    s = summarize(rd, rt, n, tr)
                    if not s["n"]:
                        cells.append("NO DATA")
                        missing += 1
                        continue
                    cells.append(f"{s['ratio']:.2f}x [{s['rmin']:.2f}-{s['rmax']:.2f}]")
                    for r in sorted(rd):
                        o, t = rd[r].get((rt, n)), rd[r].get((TORCH, tr))
                        if o and t and o["status"] == "ok" and t["status"] == "ok":
                            all_r.append(t["median_ms"] / o["median_ms"])
                if all_r and max(all_r) < 1:
                    verdict = f"ojas slower in all {len(all_r)} rounds"
                elif all_r and min(all_r) > 1:
                    verdict = f"ojas faster in all {len(all_r)} rounds"
                else:
                    verdict = "mixed"
                if missing:
                    verdict += f" ({missing} run(s) without data)"
                span = f"{min(all_r):.2f}-{max(all_r):.2f}" if all_r else "-"
                name = n if tr == n else f"{n} vs {tr}"
                lines.append(f"| {name} | " + " | ".join(cells) + f" | {span} | {verdict} |")
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        cross_run(sys.argv[1:])
    else:
        main()
