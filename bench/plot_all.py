#!/usr/bin/env python3
"""Plot every result folder under bench/results/ and write docs/bench-plots.md.

Analysis only, stdlib only. Run from the repository root:

    python3 -I bench/plot_all.py

Each folder keeps its own text format, so each has a small reader below. A
reader that cannot find its data prints a SKIP line and the page lists the
folder as having no chart, so a missing chart is never silent.
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plotlib as P  # noqa: E402

ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
OUT = os.path.join(ROOT, "docs", "assets", "plots", "bench")
DOC = os.path.join(ROOT, "docs", "bench-plots.md")
PALETTE = [P.BLUE, P.ORANGE, P.GREEN, P.RED, "#7b5ea7", "#8c564b", "#17becf", "#bcbd22"]

manifest = []  # dict(folder, file, title, note)
skipped = []


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def add(folder, name, ok, title, note):
    if ok:
        manifest.append(dict(folder=folder, file=name, title=title, note=note))
    else:
        skipped.append((folder, title))
        print("SKIP", folder, title)


def num(s):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s.replace(",", "").replace("−", "-"))
    return float(m.group(0)) if m else None


def md_tables(path):
    tabs, cur = [], []
    for line in open(path):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) or c == "" for c in cells) and any(cells):
                continue
            cur.append(cells)
        elif cur:
            tabs.append(cur)
            cur = []
    if cur:
        tabs.append(cur)
    return [dict(header=t[0], rows=t[1:]) for t in tabs if len(t) > 1]


def svg(folder, name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, f"{folder.replace('/', '--')}--{name}.svg")


def rel(folder, name):
    return f"assets/plots/bench/{folder.replace('/', '--')}--{name}.svg"


def emit_ratio(folder, name, rows, title, subtitle, note, band=1.1, sort=True):
    if sort:
        rows = sorted(rows, key=lambda r: r["v"])
    ok = P.ratio_chart(rows, svg(folder, name), title, subtitle, band=band)
    add(folder, rel(folder, name), ok, title, note)


def emit_grouped(folder, name, groups, series, title, subtitle, unit, note):
    ok = P.grouped_chart(groups, series, svg(folder, name), title, subtitle, unit=unit)
    add(folder, rel(folder, name), ok, title, note)


def emit_lines(folder, name, series, title, subtitle, xl, yl, note, logx=True, logy=True):
    ok = P.line_chart(series, svg(folder, name), title, subtitle, xl, yl, logx=logx, logy=logy)
    add(folder, rel(folder, name), ok, title, note)


# ---------------------------------------------------------------- paired GPU vs torch
def paired(folder, path=None, text=None):
    d = os.path.join(RES, folder)
    path = path or os.path.join(d, "summary.md")
    missing = {}
    if text is not None:
        tmp = os.path.join(OUT, ".paired.tmp.md")
        os.makedirs(OUT, exist_ok=True)
        open(tmp, "w").write(text)
        lanes = P.parse_paired(tmp, missing)
        os.remove(tmp)
        head = text
    else:
        if not os.path.exists(path):
            return
        lanes = P.parse_paired(path, missing)
        head = open(path).read()
    m = re.search(r"^(\d+) rounds", head, re.M)
    n = m.group(1) if m else "?"
    for lane, rows in lanes.items():
        miss = missing.get(lane, [])
        extra = ""
        if miss:
            extra = (f" {len(miss)} further row(s) have no ratio and are not charted: "
                     + "; ".join(f"`{a}` ({b})" for a, b in miss[:4]) + ".")
        emit_ratio(folder, slug(lane), rows,
                   f"{folder}: {lane}",
                   f"ojas median / torch median, median of {n} per-round ratios",
                   f"{lane}, {len(rows)} rows charted, {n} paired rounds. Hollow bars were flagged noisy by the run." + extra)


# ---------------------------------------------------------------- old/new probe markdown pairs
def probe_ab(folder, sub, note_extra=""):
    d = os.path.join(RES, folder, sub)
    olds = {}
    for f in glob.glob(os.path.join(d, "*old-*.md")):
        m = re.match(r"(.*)old-(\d+)\.md$", os.path.basename(f))
        if m:
            olds[(m.group(1), int(m.group(2)))] = f
    per_round = {}  # label -> {round: (old, new)}
    for (pre, i), fo in olds.items():
        fn = os.path.join(d, f"{pre}new-{i}.md")
        if not os.path.exists(fn):
            continue
        vals = []
        for p in (fo, fn):
            cur = {}
            for t in md_tables(p):
                col = next((k for k, h in enumerate(t["header"]) if h.startswith("min ")), None)
                if col is None:
                    continue
                for r in t["rows"]:
                    v = num(r[col])
                    if v and v > 0:
                        cur[r[0]] = v
            vals.append(cur)
        for lab in vals[0]:
            if lab in vals[1]:
                per_round.setdefault(lab, {})[i] = (vals[0][lab], vals[1][lab])
    rows = []
    for lab, rs in per_round.items():
        o = min(a for a, _ in rs.values())
        nw = min(b for _, b in rs.values())
        ratios = [b / a for a, b in rs.values()]
        rows.append(dict(name=lab, v=nw / o, lo=min(ratios), hi=max(ratios), mid=statistics.median(ratios)))
    n = len(olds)
    sfx = sub.replace("/", "--")
    emit_ratio(folder, sfx, rows, f"{folder}/{sub}: new / old",
               f"min over {n} interleaved rounds per side; whisker = per-round new/old range, circle = median",
               f"Interleaved A/B, {n} rounds per side, {len(rows)} rows. {note_extra}".strip())


# ---------------------------------------------------------------- old/new jsonl dirs via summary.txt
SUM_RE = re.compile(r"^(\S+) (old|new) rounds=(\d+) min=(\S+) med_of_mins=(\S+) med_of_medians=(\S+)")


def ab_summary(folder, sub=""):
    p = os.path.join(RES, folder, sub, "summary.txt")
    if not os.path.exists(p):
        return
    d = {}
    for line in open(p):
        m = SUM_RE.match(line)
        if m:
            d.setdefault(m[1], {})[m[2]] = (float(m[4]), float(m[5]), float(m[6]), int(m[3]))
    rows, n = [], 0
    for row, s in d.items():
        if "old" in s and "new" in s and s["old"][0] > 0:
            n = s["new"][3]
            rows.append(dict(name=row, v=s["new"][0] / s["old"][0], mid=s["new"][1] / s["old"][1]))
    emit_ratio(folder, (sub or "ab").replace("/", "--"), rows, f"{folder}/{sub}".rstrip("/") + ": new / old",
               f"min-of-rounds ratio, circle = median-of-round-minimums ratio, {n} rounds",
               f"Old binary against new binary, {n} rounds, {len(rows)} rows.")


# ---------------------------------------------------------------- grouped absolute-time tables
def table_series(path, label_col, value_cols, names, table_idx=0, match=None):
    t = md_tables(path)
    if match:
        t = [x for x in t if match(x["header"])]
    t = t[table_idx]
    groups = []
    for r in t["rows"]:
        groups.append((r[label_col], [num(r[c]) for c in value_cols]))
    return groups


def kernel_tables():
    f = "2026-10-02-kernels"
    for fn in ("kernels.md", "kernels-after-fold.md"):
        p = os.path.join(RES, f, fn)
        g = table_series(p, 0, [3], ["min us"])  # header: kernel, elements, f32 passes, min µs, median µs, GB/s
        emit_grouped(f, slug(fn), g, [("min µs (lower is faster)", P.BLUE)],
                     f"{f}/{fn}", "Metal kernels, one dispatch, GPU timestamps", "µs",
                     "Per-dispatch GPU time at the minimum. The sequence rows are the old separate finite-check passes.")
    p = os.path.join(RES, f, "kernels.md")
    t = md_tables(p)[0]
    emit_grouped(f, "bandwidth", [(r[0], [num(r[5])]) for r in t["rows"]], [("GB/s at min (higher is faster)", P.GREEN)],
                 f"{f}: achieved bandwidth, before the fold", "GB/s at the minimum", "GB/s",
                 "Memory bandwidth reached by each kernel before the finite checks were folded in.")


def floor_tables():
    f = "2026-10-01-floor"
    a = md_tables(os.path.join(RES, f, "floor.md"))[0]
    b = md_tables(os.path.join(RES, f, "floor-after-condvar.md"))[0]
    hb = {r[0]: r for r in a["rows"]}
    hd = {r[0]: r for r in b["rows"]}
    groups = []
    for k in hb:
        if k in hd:
            groups.append((k, [num(hb[k][4]), num(hd[k][4]), num(hb[k][5]), num(hd[k][5])]))
    emit_grouped(f, "floor", groups, [("p50 before", P.BLUE), ("p50 after condvar", P.GREEN),
                                       ("p90 before", P.ORANGE), ("p90 after condvar", P.RED)],
                 f"{f}: submit-and-wait floor, before and after the condvar wait",
                 "Per-scenario wall time over 500 runs", "ms",
                 "Median and 90th-percentile wall time of tiny GPU calls. The first run was at load 45 with the GPU 100% busy.")


def gemm_tables():
    f = "2026-10-02-gemm"
    for fn in ("gemm.md", "gemm-relabelled.md"):
        t = md_tables(os.path.join(RES, f, fn))[0]
        by, order = {}, []
        for r in t["rows"]:
            m = re.match(r"(nt|nn|tn) (.*)", r[0])
            if not m:
                continue
            key = m[2]
            if key not in by:
                by[key] = {}
                order.append(key)
            by[key][m[1]] = num(r[6])
        groups = [(k, [by[k].get("nt"), by[k].get("nn"), by[k].get("tn")]) for k in order]
        emit_grouped(f, slug(fn), groups, [("nt", P.BLUE), ("nn", P.ORANGE), ("tn", P.GREEN)],
                     f"{f}/{fn}: exact-f32 GEMM throughput", "TFLOP/s at the minimum, higher is faster", "TFLOP/s",
                     "Before the panel-walk fix. The LM-head nt and nn rows drop to about 2 TFLOP/s while tn stays near 5.5.")
    for tn in ("tn-sweep-1", "tn-sweep-2"):
        tn_sweep(tn)
    g1 = table_series(os.path.join(RES, "2026-10-02-gate", "gate-probe-1.md"), 0, [2], ["min"])
    g2 = table_series(os.path.join(RES, "2026-10-02-gate", "gate-probe-2.md"), 0, [2], ["min"])
    d2 = dict(g2)
    emit_grouped("2026-10-02-gate", "gate-probe", [(k, [v[0], (d2.get(k) or [None])[0]]) for k, v in g1], [("probe 1", P.BLUE), ("probe 2", P.ORANGE)],
                 "2026-10-02-gate: gate_bwd dispatch profile before the fix", "GPU span of each dispatch, min µs", "µs",
                 "ojas_per_head_gate_dbias (the serial bias sum) is the 800 µs outlier the fix removes.")


def tn_sweep(tn):
    t = md_tables(os.path.join(RES, "2026-10-02-gate", "tn-sweep", tn + ".md"))[0]
    by, order = {}, []
    for r in t["rows"]:
        k = r[0]
        if k not in by:
            by[k] = {"single": None, "routed": None, "par": None}
            order.append(k)
        v = num(r[3])
        if r[2].startswith("single"):
            by[k]["single"] = v
        elif r[2].startswith("routed"):
            by[k]["routed"] = v
        elif r[2].startswith("parallel"):
            by[k]["par"] = v if by[k]["par"] is None else min(by[k]["par"], v)
    groups = [(k, [by[k]["single"], by[k]["routed"], by[k]["par"]]) for k in order]
    quiet = "quiet slot" if tn.endswith("2") else "contended (other jobs running; the README says -2 supersedes it)"
    emit_grouped("2026-10-02-gate", tn, groups,
                 [("single dispatch", P.RED), ("routed", P.BLUE), ("best parallel split-K", P.GREEN)],
                 f"2026-10-02-gate/{tn}: TN GEMM, single dispatch against split-K", "min µs per shape (M, N, K), lower is faster",
                 "µs", f"{quiet.capitalize()}. Split-K at width 128 to 2048 against the single dispatch and what tessl routes to.")


def lappi():
    for f in ("2026-10-03-lappi-inference", "2026-10-03-lappi-inference-rerun"):
        t = next((x for x in md_tables(os.path.join(RES, f, "README.md")) if x["header"] and x["header"][0] == "Task"), None)
        if not t:
            skipped.append((f, "lappi table"))
            continue
        pair = lambda s: [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))[:2]]
        groups, dgroups = [], []
        for r in t["rows"]:
            a, b = pair(r[3]), pair(r[4])
            groups.append((f"{r[0]} {r[1]}", [a[0], a[1], b[0], b[1]]))
            d = pair(r[6])
            dgroups.append((f"{r[0]} {r[1]}", [d[0], d[1]]))
        emit_grouped(f, "prefill", groups, [("tessl base", P.BLUE), ("tessl Lappi", P.GREEN),
                                             ("PyTorch MPS base", P.ORANGE), ("PyTorch MPS Lappi", P.RED)],
                     f"{f}: Qwen3.5-2B prefill", "Prefill time per prompt, original model (base) and Lappi fine-tune", "ms",
                     "tessl against PyTorch 2.12.1 MPS bf16. ojas has no Qwen3.5 inference path, so it is not in this comparison. "
                     "The README tables were built from the raw logs in this folder; the charts read the tables.")
        emit_grouped(f, "decision", dgroups, [("tessl base", P.BLUE), ("tessl Lappi", P.GREEN)],
                     f"{f}: tessl full decision time", "Prefill plus decision head, per prompt", "ms",
                     "Base and Lappi weights run at the same speed, as expected for equal shapes.")


def percall():
    for f, fn in (("2026-10-04-percall", "percall.md"), ("2026-10-04-split", "percall.md")):
        p = os.path.join(RES, f, fn)
        tabs = md_tables(p)
        a = next((t for t in tabs if t["header"][0] == "scenario"), None)
        if a:
            emit_grouped(f, "percall", [(r[0], [num(r[2]), num(r[3])]) for r in a["rows"]],
                         [("min ms", P.BLUE), ("p50 ms", P.ORANGE)], f"{f}: Metal decode call cost",
                         "H 12, D 64, 1024 cached positions, 500 runs", "ms",
                         "One request, one batched call, and sixteen separate calls.")
        b = next((t for t in tabs if t["header"][0].startswith("command buffer holds")), None)
        if b:
            emit_grouped(f, "gpu-span", [(r[0], [num(r[1]), num(r[2])]) for r in b["rows"]],
                         [("min µs", P.BLUE), ("median µs", P.ORANGE)], f"{f}: GPU span of one command buffer",
                         "tessl directly, 500 buffers", "µs", "Batched dispatch against sixteen separate dispatches.")


def sweep_fit():
    f = "2026-10-04-sweep"
    tabs = md_tables(os.path.join(RES, f, "fit.md"))
    t = next(x for x in tabs if x["header"][0] == "N")
    names = ["ojas-metal", "ojas-wgpu", "torch-mps"]
    ser = []
    for k, nm in enumerate(names):
        pts = [(num(r[0]), num(r[1 + k].split("/")[1])) for r in t["rows"]]
        ser.append((nm, PALETTE[k], pts))
    emit_lines(f, "silu-over-n", ser, f"{f}: silu_forward time against size",
               "Median of per-round medians; the floor is the fixed cost of one call", "elements N", "ms",
               "Below roughly 4M elements every runtime sits on its floor.")
    d = next(x for x in tabs if x["header"][0] == "B")
    ser, idx = [], 0
    for rt in names:
        for mode, col in (("batched", 3), ("split", 5)):
            pts = [(num(r[0]), num(r[col])) for r in d["rows"] if r[1] == rt and num(r[col]) is not None]
            ser.append((f"{rt} {mode}", PALETTE[idx % len(PALETTE)], pts))
            idx += 1
    emit_lines(f, "decode-batching", ser, f"{f}: decode cost per request", "B requests in one call against B calls then one sync",
               "requests B", "ms per request", "Batching amortises the per-call cost; the gap between split and batched is the extra per call.",
               logx=True, logy=True)


def split_readme():
    f = "2026-10-04-split"
    for t in md_tables(os.path.join(RES, f, "README.md")):
        if t["header"][0] == "splits":
            r = next(x for x in t["rows"] if x[0].startswith("median"))
            pts = [(num(h), num(v)) for h, v in zip(t["header"][1:], r[1:])]
            emit_lines(f, "splits", [("median µs", P.BLUE, pts)], f"{f}: split count for the decode attention kernel",
                       "median µs against number of key splits", "splits", "µs", "Time bottoms out near eight splits.", logy=False)
            return
    skipped.append((f, "splits table"))


def clip():
    f = "2026-10-05-torch215-clip"
    t = md_tables(os.path.join(RES, f, "README.md"))[0]
    g = []
    for r in t["rows"]:
        g.append((f"torch {r[0]}", [num(r[1])]))
    txt = open(os.path.join(RES, f, "README.md")).read()
    m = re.search(r"ojas at ([\d.]+) ms median on Metal", txt)
    if m:
        g.append(("ojas Metal (round 5, unpaired)", [float(m[1])]))
    emit_grouped(f, "clip", g, [("median ms", P.BLUE)], f"{f}: clip_grad_norm on MPS",
                 "Median of run 1; lower is faster", "ms",
                 "The folder's README attributes the 16x drop to a change in torch's MPS norm reduction and says a paired run with torch 2.15 is still needed before a new ratio is quoted.")


def gate_saved():
    f = "2026-10-07-gate-saved"
    t = next(x for x in md_tables(os.path.join(RES, f, "README.md")) if x["header"][0] == "Backend")
    rows = []
    for r in t["rows"]:
        for k, nm in ((2, "forward_saving / forward"), (3, "backward_saved / backward")):
            mn, md = [num(x) for x in r[k].split("/")]
            rows.append(dict(name=f"{r[0]} round {r[1]}: {nm}", v=md, mid=mn, hollow="noisy" in r[1]))
    emit_ratio(f, "gate-saved", rows, f"{f}: saved-sigmoid gate pair against recompute",
               "saved / plain, median per round; circle = min; hollow = noisy round (started at load 24)",
               "The saved backward is 7% faster on Metal and 15% faster on wgpu; the forward costs at most about 2%.", sort=False)


def attn_lse():
    f = "2026-10-08-attn-lse-ab"
    tabs = md_tables(os.path.join(RES, f, "tables.md"))
    t = tabs[0]
    rows = []
    for r in t["rows"]:
        m = re.match(r"([\d.]+) \(([\d.]+)-([\d.]+)\)", r[5])
        c = re.match(r"([\d.]+)", r[6])
        rows.append(dict(name=f"{r[0]} {r[1]} {r[2]}", v=float(m[1]), lo=float(m[2]), hi=float(m[3]),
                         mid=float(c[1])))
    emit_ratio(f, "new-over-old", rows, f"{f}: attention backward with saved log-sum-exp",
               "new / old per round, median and range; circle = control op (should be 1.0)",
               "Backward is 0.76 to 0.81 of the old time on every shape and both backends. Forward is within noise on the multi-head shapes; "
               "the grouped-query shape's Metal forward is 0.89.")
    w = tabs[1]
    by, order = {}, []
    for r in w["rows"]:
        k = f"{r[0]} {r[1]}"
        if k not in by:
            by[k] = {}
            order.append(k)
        by[k][r[2]] = (num(r[3]), num(r[4]))
    groups = [(k, [by[k]["full"][0], by[k]["w256"][0], by[k]["full"][1], by[k]["w256"][1]]) for k in order]
    emit_grouped(f, "window", groups, [("bwd full", P.BLUE), ("bwd window 256", P.ORANGE),
                                        ("fwd full", P.GREEN), ("fwd window 256", P.RED)],
                 f"{f}: sliding window of 256 against full attention", "best min ms over rounds", "ms",
                 "The 256-wide window cuts backward time by roughly 3x at 2048 tokens.")


def cpu_hot_paths():
    f = "2026-10-08-cpu-hot-paths"
    p = os.path.join(RES, f, "summary.md")
    title, cur, secs = "ab", None, {}
    for line in open(p):
        if line.startswith("## "):
            title = line[3:].strip()
            continue
        if not line.startswith("|") or line.startswith("| :") or line.startswith("| row"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 8:
            continue
        secs.setdefault(title, []).append(c)
    for k, (title, rows) in enumerate(secs.items()):
        tag = ("ab", "ab-tape", "ab-v2")[k] if k < 3 else slug(title)[:24]
        pre = "" if k else "step"
        rr = []
        for c in rows:
            nm = c[0] + (" " + c[1] if c[1] not in ("step",) else "")
            nm = nm.replace("tape ", "")
            if num(c[4]) is not None:
                rr.append(dict(name=nm, v=num(c[4]), mid=num(c[5])))
        title = "interleaved A/B, base against after" if title == "ab" else title
        emit_ratio(f, tag + "-vs-base", rr, f"{f}: {title}",
                   "after / base, min over rounds; circle = median of per-round ratios; load 17-82 on the first table",
                   f"{len(rr)} rows. Ratios between 0.9 and 1.1 are noise on this machine.")
        tr = [dict(name=c[0] + " " + c[1], v=num(c[7])) for c in rows if num(c[7]) is not None]
        if tr:
            emit_ratio(f, tag + "-vs-torch", tr, f"{f}: ojas / torch 2.13 CPU", "min over rounds, rows torch covers",
                       f"{len(tr)} rows with a torch counterpart.")


def floor_ab():
    f = "2026-10-01-floor"
    t = md_tables(os.path.join(RES, f, "ab", "summary.md"))[0]
    by, order = {}, []
    for r in t["rows"]:
        if r[0] not in by:
            by[r[0]] = {}
            order.append(r[0])
        by[r[0]][r[1]] = (num(r[4]), num(r[5]))
    groups = [(k, [by[k]["old"][0], by[k]["new"][0], by[k]["old"][1], by[k]["new"][1]]) for k in order if "old" in by[k] and "new" in by[k]]
    emit_grouped(f, "ab", groups, [("p50 old", P.BLUE), ("p50 new", P.GREEN), ("p90 old", P.ORANGE), ("p90 new", P.RED)],
                 f"{f}/ab: submit-and-wait floor, old against new", "median over rounds of the per-round p50 and p90", "ms",
                 "600 runs per side. The old side had 388 to 507 slow (>1 ms) runs of 600 per scenario; the new side had 0 to 6.")


def kv_lines(path, prefix):
    for line in open(path):
        if line.startswith(prefix):
            yield dict(re.findall(r"(\w+)=([^\s]+)", line)), line


def cpu_probes():
    f = "2026-10-08-cpu-hot-paths"
    d = os.path.join(RES, f)
    best = {}
    for kv, _ in kv_lines(os.path.join(d, "muon-gate.txt"), "OJAS_OP"):
        pass
    gate = None
    for line in open(os.path.join(d, "muon-gate.txt")):
        m = re.match(r"round=\d+ gate=(\S+)", line)
        if m:
            gate = m[1]
        elif line.startswith("OJAS_OP") and gate:
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            k = (kv["op"], gate)
            best[k] = min(best.get(k, 1e18), float(kv["min_ms"]))
    rows = []
    for (op, g), v in best.items():
        if g != "nanolab" and (op, "nanolab") in best:
            rows.append(dict(name=f"{op}: gate {g}", v=v / best[(op, "nanolab")]))
    emit_ratio(f, "muon-gate", rows, f"{f}: Muon no-transpose view gate",
               "min over 3 rounds, divided by the shipped gate (nanolab); above 1 is slower than shipped",
               "Opening the view to every tall matrix (tall, tallbands) reads 1.3 to 2.9x slower than shipped, so the gate stays. "
               "Gate off reads 0.78 to 1.21 and is noise-bound at load 20-35 (the README calls 2048x768 on versus off within noise).")
    runs = {}
    for kv, _ in kv_lines(os.path.join(d, "probes", "syrk.txt"), "XXt"):
        k = f"{kv['r']}x{kv['c']}"
        for v in ("gemm", "gemm_2band", "syrk_mirror"):
            runs.setdefault(k, {})[v] = min(runs.get(k, {}).get(v, 1e18), float(kv[v]))
    emit_grouped(f, "syrk", [(k, [v["gemm"], v["gemm_2band"], v["syrk_mirror"]]) for k, v in runs.items()],
                 [("one sgemm", P.BLUE), ("two-band split", P.ORANGE), ("ssyrk + mirror", P.GREEN)],
                 f"{f}: X times X-transpose, three ways", "min over probe runs, by matrix shape (rows x cols of X)", "ms",
                 "The ssyrk product matched sgemm bit for bit at every probed shape (bits_syrk_eq_gemm).")
    runs = {}
    for kv, _ in kv_lines(os.path.join(d, "probes", "mirror.txt"), "mirror"):
        for v in ("serial", "t8", "t16", "t32", "t64"):
            runs.setdefault(kv["n"], {})[v] = min(runs.get(kv["n"], {}).get(v, 1e18), float(kv[v]))
    emit_grouped(f, "mirror", [(f"n = {k}", [v[x] for x in ("serial", "t8", "t16", "t32", "t64")]) for k, v in runs.items()],
                 [("column walk", P.BLUE), ("8 tiles", P.ORANGE), ("16 tiles", P.GREEN), ("32 tiles", P.RED), ("64 tiles", "#7b5ea7")],
                 f"{f}: ssyrk lower-from-upper mirror", "min over probe runs", "ms", "16 x 16 tiles shipped. At n = 2048 the 32 and 64 tile sizes are slower than the plain column walk.")
    pts = {}
    for kv, _ in kv_lines(os.path.join(d, "spawn.txt"), "scope"):
        for k in ("min_us", "med_us"):
            x = int(kv["spawned"])
            pts.setdefault(k, {})[x] = min(pts.get(k, {}).get(x, 1e18), float(kv[k]))
    emit_lines(f, "spawn", [("min µs", P.BLUE, list(pts["min_us"].items())), ("median µs", P.ORANGE, list(pts["med_us"].items()))],
               f"{f}: cost of one std::thread::scope", "min over two runs of the best and median over 2000 spawns, load 26", "threads spawned", "µs",
               "About 9 µs for one thread and 34-38 µs for five at the minimum.", logx=False, logy=False)
    base, v2 = {}, {}
    for line in open(os.path.join(d, "tape-peak.txt")):
        side = base if line.startswith("base") else v2
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        side[kv["row"]] = (float(kv["min_ms"]), float(kv["walk_peak_bytes"]))
    ks = list(base)
    emit_grouped(f, "tape-peak-bytes", [(k, [base[k][1], v2[k][1]]) for k in ks], [("base", P.BLUE), ("final tree", P.GREEN)],
                 f"{f}: tape walk peak charge", "bytes, from the budget counter (does not depend on machine load)", "bytes",
                 "Fan-in peaks at 18,874,368 then 12,582,912 bytes. The seed one-quarter fused-CE walk falls from 154,533,892 to 4 bytes.")
    emit_grouped(f, "tape-peak-time", [(k, [base[k][0], v2[k][0]]) for k in ks], [("base", P.BLUE), ("final tree", P.GREEN)],
                 f"{f}: tape walk time in the peak run", "min ms of one run each, so noisy (the fan-in row reads slower here)", "ms",
                 "One run per side under load; use the interleaved ab-tape table for time, not this chart.")


def attn_scratch():
    f = "2026-10-08-attn-lse-ab"
    groups = []
    for line in open(os.path.join(RES, f, "gqa_scratch.txt")):
        m = re.search(r"(metal|wgpu) qwen35.*forward (\d+) B \(expand path (\d+) B\), backward (\d+) B \(expand path (\d+) B\)", line)
        if m:
            groups += [(f"{m[1]} forward", [float(m[2]), float(m[3])]), (f"{m[1]} backward", [float(m[4]), float(m[5])])]
    emit_grouped(f, "gqa-scratch", groups, [("native grouped-query", P.GREEN), ("expand path", P.RED)],
                 f"{f}: scratch charged at the Qwen3.5 shape", "bytes charged by the backend's budget, b1 h8/2 t2048 d256", "bytes",
                 "Reading each KV head in place cuts the charged scratch to about a third of the expanded-heads path (forward) and 27% (backward).")


def decode_tokens():
    f = "2026-10-08-decode-before"
    t = next(x for x in md_tables(os.path.join(RES, f, "summary.md")) if x["header"][0] == "runtime")
    emit_grouped(f, "decode-tokens", [(r[0], [num(r[1]), num(r[2])]) for r in t["rows"]],
                 [("prefill median ms (32 tokens)", P.BLUE), ("decode ms per token", P.ORANGE)],
                 f"{f}: generation, 32-token prompt and 32 new tokens", "3 rounds; torch-mps against each ojas lane", "ms",
                 "Every ojas lane produced the same 32 greedy ids as torch. ojas-cpu decodes at 5.09 ms per token against torch-mps at 5.88; the GPU lanes are slower.")


def gemm_bf16():
    f = "2026-10-06-gemm-bf16"
    rows = []
    for line in open(os.path.join(RES, f, "ab2", "summary.txt")):
        m = re.match(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            rows.append(dict(name=m[1], v=float(m[6]), mid=float(m[6]), lo=float(m[5]), hi=float(m[7])))
    emit_ratio(f, "ab2", rows, f"{f}: column-panel tile walk, new / old",
               "median of 6 back-to-back pairs per case; whisker = min..max", "The largest-B cases fall to 0.5 to 0.8 of the old time; the Morton-order and small-B controls stay near 1.0.")
    rows, case = [], None
    for line in open(os.path.join(RES, f, "sweep", "ratios.txt")):
        h = re.match(r"^(\S+)\s+M=", line)
        if h:
            case = h[1]
            continue
        m = re.match(r"^\s+(\S+)\s+n=\d+\s+min\s+([\d.]+)\s+med\s+([\d.]+)\s+max\s+([\d.]+)", line)
        if m and case:
            nm = m[1].replace("mm_bf16_nt_coop_128x64_sg4", "base")
            rows.append(dict(name=f"{case} / {nm}", v=float(m[3]), mid=float(m[3]), lo=float(m[2]), hi=float(m[4])))
    emit_ratio(f, "sweep", rows, f"{f}: panel-height sweep", "ratio to the same round's production kernel; whisker = min..max",
               "Panel heights 4, 8 and 16 against the shipped kernel across six B sizes.", sort=False)


def gemm_bf16_tune():
    f = "2026-10-06-gemm-bf16"
    for tune in ("tune1", "tune2"):
        times, cases, variants = {}, [], []
        for p in sorted(glob.glob(os.path.join(RES, f, "sweep", tune + "-r*.txt"))):
            case = None
            for line in open(p):
                h = re.match(r"^(\S+)\s+M=\d+", line)
                if h:
                    case = h[1]
                    if case not in cases:
                        cases.append(case)
                    continue
                m = re.match(r"^\s+(\S+)\s+([\d.]+)\s+\d+\s+[\d.]+×", line)
                if m and case:
                    v = m[1].replace("mm_bf16_", "")
                    if v not in variants:
                        variants.append(v)
                    times.setdefault((case, v), []).append(float(m[2]))
        groups = [(c, [statistics.median(times[(c, v)]) if (c, v) in times else None for v in variants]) for c in cases]
        emit_grouped(f, f"{tune}-raw", groups, [(v, PALETTE[i % len(PALETTE)]) for i, v in enumerate(variants)],
                     f"{f}: {tune} raw sweep", "median over 3 rounds of the in-process time per kernel variant, by case", "ms",
                     "Raw times behind sweep/ratios.txt. Variants whose name ends _phN are column-panel heights.")


def jsonl_unpaired(folder, sub=""):
    d = os.path.join(RES, folder, sub)
    best, rts = {}, []
    for p in sorted(glob.glob(os.path.join(d, "round*.jsonl"))):
        for line in open(p):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if j.get("status") == "ok" and "min_ms" in j:
                k = (j["row"], j["runtime"])
                best[k] = min(best.get(k, 1e18), j["min_ms"])
                if j["runtime"] not in rts:
                    rts.append(j["runtime"])
    rows = sorted({k[0] for k in best})
    groups = [(r, [best.get((r, rt)) for rt in rts]) for r in rows]
    emit_grouped(folder, "rows", groups, [(rt, PALETTE[i]) for i, rt in enumerate(rts)], f"{folder}: min time per row",
                 "min over rounds of each round's minimum (unpaired, machine under heavy load)", "ms",
                 "Direction only. These rounds ran at load around 350.")


def cross_run():
    f = "2026-10-01"
    for run in ("runA", "runB", "runC"):
        paired(f + "/" + run)


def r3():
    d = os.path.join(RES, "2026-10-01-r3")
    out = subprocess.run([sys.executable, "-I", os.path.join(HERE, "aggregate.py"), d],
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        skipped.append(("2026-10-01-r3", "aggregate.py failed: " + out.stderr[-200:]))
        return
    paired("2026-10-01-r3", text=out.stdout)


def main():
    for stale in glob.glob(os.path.join(OUT, "*.svg")):  # no orphans from renamed charts
        os.remove(stale)
    for folder in ("2026-10-01-r2", "2026-10-01-r3b", "2026-10-01-r4-smallops", "2026-10-02-r5",
                   "2026-10-04-sweep", "2026-10-07-muon-bf16", "2026-10-08-decode-before"):
        paired(folder)
    paired("2026-10-02-gate/paired-gate")
    paired("2026-10-02-gate/paired-gate-quiet")
    r3()
    cross_run()
    ab_summary("2026-10-01-ab")
    for sub in ("ab-dbias", "ab-fold", "ab-tn", "ab-stage"):
        probe_ab("2026-10-02-gate", sub)
    for sub in ("ab-grid-gate", "ab-footprint-gate", "ab-nn-splitk"):
        probe_ab("2026-10-02-gemm", sub)
    for sub in ("ab-rows", "ab-rows-ce"):
        ab_summary("2026-10-02-gemm", sub)
    jsonl_unpaired("2026-10-02-gemm", "splitk-rows")
    for sub in ("ab-check", "ab-fold"):
        ab_summary("2026-10-02-kernels", sub)
    ab_summary("2026-10-04-split", "ab")
    for fn in (kernel_tables, floor_tables, gemm_tables, lappi, percall, sweep_fit, split_readme, clip,
               gate_saved, attn_lse, cpu_hot_paths, gemm_bf16, gemm_bf16_tune, floor_ab, cpu_probes, attn_scratch, decode_tokens):
        try:
            fn()
        except Exception as e:  # keep going; report at the end
            skipped.append((fn.__name__, repr(e)))
            print("ERROR", fn.__name__, repr(e))
    write_doc()


def write_doc():
    by = {}
    for m in manifest:
        by.setdefault(m["folder"].split("/")[0], []).append(m)
    folders = sorted(d for d in os.listdir(RES) if os.path.isdir(os.path.join(RES, d)))
    L = ["# Benchmark Plots, Every Result Folder", "",
         "Each chart is drawn from the text files kept under `bench/results/<folder>`, by `bench/plot_all.py`. "
         "Regenerate with `python3 -I bench/plot_all.py` from the repository root. Ratio charts are times, so "
         "**lower is faster** and 1x is parity. Green bars are faster by more than 10%, red are slower by more than 10%, "
         "grey is inside the 10% band, which is the noise this machine showed. A hollow bar means the run itself "
         "flagged the row noisy. Whiskers are the per-round range and circles are medians where the source gives them.",
         "", "Two cautions on reading them. The paired GPU summaries print ratios to two decimals, so where the "
         "printed ratio is under 0.05x (ojas more than 20x slower) the chart uses the ratio of the two printed "
         "medians instead and draws no whisker. That is a different statistic from the median of per-round "
         "ratios, and on rows the run flagged noisy the two can differ by 2x or more. Dot charts of absolute "
         "times use a log axis, so distance between dots is a ratio, not a difference.",
         "", "Read each folder's own README or summary before quoting a number: several runs were taken under heavy "
         "load, and the charts show direction, not a verdict. The older prose analysis is in "
         "[bench-gpu-vs-torch.md](bench-gpu-vs-torch.md) and [bench-cpu-vs-torch.md](bench-cpu-vs-torch.md).", ""]
    for d in folders:
        L.append(f"## {d}")
        L.append("")
        items = by.get(d, [])
        if not items:
            L += ["No chart: the folder holds only scripts, logs or text that is not a measurement table.", ""]
            continue
        for m in items:
            L += [f"### {m['title']}", "", f"![{m['title']}: {m['note']}]({m['file']})", "", f"*{m['note']}*", ""]
        L.append(f"Source: `bench/results/{d}/`.")
        L.append("")
    open(DOC, "w").write("\n".join(L))
    print(f"wrote {len(manifest)} charts, {len(skipped)} skipped -> {DOC}")
    for s in skipped:
        print("  skipped:", s)


main()
