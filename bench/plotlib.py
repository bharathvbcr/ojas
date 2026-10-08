"""Stdlib-only SVG charts for bench results (analysis only, ships nothing).

Every chart shows a time ratio on a log axis where lower is faster.
"""
import math
import re

GREEN, GREY, RED, BLUE, ORANGE = "#2a9d8f", "#9aa0a6", "#d1495b", "#3d6fb6", "#e08e2b"
FONT = 'font-family="Helvetica,Arial,sans-serif" font-size="12"'


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def verdict_color(v, band=1.1):
    return GREEN if v < 1 / band else (RED if v > band else GREY)


def _ticks(lo, hi, mults=None):
    if mults is None:
        mults = (1, 2, 5) if math.log10(hi / lo) <= 3 else (1,)
    out, e = [], math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.001:
        for m in mults:
            t = m * 10 ** e
            if lo * 0.999 <= t <= hi * 1.001:
                out.append(t)
        e += 1
    return out


def ratio_chart(rows, out, title, subtitle, band=1.1, label_w=300, plot_w=560):
    """rows: dicts name, v (ratio, <1 faster), optional lo/hi whisker, mid (median
    marker), hollow (bool, noisy), color. Sorted by caller."""
    if not rows:
        return False
    vals = [r["v"] for r in rows] + [r.get("lo", r["v"]) for r in rows] + [r.get("hi", r["v"]) for r in rows]
    vals = [v for v in vals if v and v > 0]
    lo = min(min(vals) * 0.55, 0.8)
    hi = max(max(vals) * 1.25, 1.25)
    lo, hi = max(lo, 1e-3), min(hi, 1e3)
    rh, top, right = 20, 78, 60
    W, H = label_w + plot_w + right, top + rh * len(rows) + 50
    llo, lhi = math.log(lo), math.log(hi)
    x = lambda v: label_w + (math.log(min(max(v, lo), hi)) - llo) / (lhi - llo) * plot_w
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT}>',
         '<rect width="100%" height="100%" fill="#fff"/>',
         f'<text x="16" y="24" font-size="15" font-weight="bold" fill="#111">{esc(title)}</text>',
         f'<text x="16" y="44" fill="#555">{esc(subtitle)}</text>',
         f'<text x="16" y="62" fill="#555">Lower is faster. Grey band = within {int((band-1)*100)}%. '
         f'Hollow bar = run marked noisy. Whisker = per-round min..max.</text>']
    bot = top + rh * len(rows)
    s.append(f'<rect x="{x(1/band):.1f}" y="{top-6}" width="{x(band)-x(1/band):.1f}" height="{bot-top+6}" fill="#eee"/>')
    for t in _ticks(lo, hi):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top-6}" y2="{bot}" '
                 f'stroke="{"#222" if abs(t-1)<1e-9 else "#ddd"}" stroke-width="{1.5 if abs(t-1)<1e-9 else 1}"/>')
        s.append(f'<text x="{x(t):.1f}" y="{bot+16}" text-anchor="middle" fill="#444">{t:g}x</text>')
    for i, r in enumerate(rows):
        y = top + i * rh
        v = r["v"]
        c = r.get("color") or verdict_color(v, band)
        x0, x1 = sorted((x(1), x(v)))
        s.append(f'<text x="{label_w-8}" y="{y+14}" text-anchor="end" fill="#222">{esc(r["name"][:46])}</text>')
        if r.get("hollow"):
            s.append(f'<rect x="{x0:.1f}" y="{y+3}" width="{max(x1-x0,1):.1f}" height="{rh-7}" fill="#fff" stroke="{c}" stroke-width="1.4"/>')
        else:
            s.append(f'<rect x="{x0:.1f}" y="{y+3}" width="{max(x1-x0,1):.1f}" height="{rh-7}" fill="{c}"/>')
        if "lo" in r and "hi" in r:
            yc = y + rh / 2 - 0.5
            s.append(f'<line x1="{x(r["lo"]):.1f}" x2="{x(r["hi"]):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="#222" stroke-width="1"/>')
            for e in (r["lo"], r["hi"]):
                s.append(f'<line x1="{x(e):.1f}" x2="{x(e):.1f}" y1="{yc-3:.1f}" y2="{yc+3:.1f}" stroke="#222" stroke-width="1"/>')
        if r.get("mid") is not None:
            s.append(f'<circle cx="{x(r["mid"]):.1f}" cy="{y+rh/2-0.5:.1f}" r="3.2" fill="#fff" stroke="#222" stroke-width="1.2"/>')
        edge = max(x0, x1) if v >= 1 else min(x0, x1)
        far = max(x(r.get("hi", v)), x1) if v >= 1 else min(x(r.get("lo", v)), x0)
        tx = far + 5 if v >= 1 else far - 5
        s.append(f'<text x="{tx:.1f}" y="{y+14}" text-anchor="{"start" if v>=1 else "end"}" fill="#333">{v:.2f}</text>')
    s.append("</svg>")
    open(out, "w").write("\n".join(s))
    return True


def grouped_chart(groups, series, out, title, subtitle, unit="ms", label_w=300, plot_w=520):
    """Grouped horizontal bars on a log axis in absolute units.
    groups: [(label, [value-or-None per series])]; series: [(name, color)]."""
    vals = [v for _, vs in groups for v in vs if v]
    if not vals:
        return False
    lo, hi = min(vals) * 0.7, max(vals) * 1.6
    n = len(series)
    bh, gap, right = 11, 9, 70
    gh = bh * n + gap
    W = label_w + plot_w + right
    # legend: items wrap onto new rows instead of running off the canvas
    pos, lx, ly = [], 16, 0
    for name, _ in series:
        w = 24 + len(name) * 6.4 + 18
        if lx + w > W - 16 and lx > 16:
            lx, ly = 16, ly + 18
        pos.append((lx, ly))
        lx += w
    top = 62 + ly + 22
    H = top + gh * len(groups) + 50
    llo, lhi = math.log(lo), math.log(hi)
    x = lambda v: label_w + (math.log(v) - llo) / (lhi - llo) * plot_w
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT}>',
         '<rect width="100%" height="100%" fill="#fff"/>',
         f'<text x="16" y="24" font-size="15" font-weight="bold" fill="#111">{esc(title)}</text>',
         f'<text x="16" y="44" fill="#555">{esc(subtitle)}</text>']
    for (name, col), (px_, py_) in zip(series, pos):
        s.append(f'<circle cx="{px_+6}" cy="{60+py_}" r="5" fill="{col}"/>'
                 f'<text x="{px_+17}" y="{64+py_}" fill="#333">{esc(name)}</text>')
    bot = top + gh * len(groups)
    for t in _ticks(lo, hi):
        s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top-6}" y2="{bot}" stroke="#ddd"/>')
        s.append(f'<text x="{x(t):.1f}" y="{bot+16}" text-anchor="middle" fill="#444">{t:g}</text>')
    s.append(f'<text x="{label_w+plot_w/2}" y="{bot+34}" text-anchor="middle" fill="#444">{unit} (log axis)</text>')
    for i, (label, vs) in enumerate(groups):
        y = top + i * gh
        s.append(f'<text x="{label_w-8}" y="{y+bh*n/2+4}" text-anchor="end" fill="#222">{esc(label[:46])}</text>')
        for k, v in enumerate(vs):
            if not v:
                continue
            yy = y + k * bh
            # dots, not bars: a bar's length on a log axis depends on an arbitrary baseline
            s.append(f'<line x1="{label_w}" x2="{x(v):.1f}" y1="{yy+bh/2-0.5:.1f}" y2="{yy+bh/2-0.5:.1f}" stroke="{series[k][1]}" stroke-opacity="0.35"/>')
            s.append(f'<circle cx="{x(v):.1f}" cy="{yy+bh/2-0.5:.1f}" r="4.2" fill="{series[k][1]}"/>')
            s.append(f'<text x="{x(v)+8:.1f}" y="{yy+bh-2}" fill="#333" font-size="10">{v:.3g}</text>')
    s.append("</svg>")
    open(out, "w").write("\n".join(s))
    return True


def line_chart(series, out, title, subtitle, xlabel, ylabel, logx=True, logy=True, w=720, h=380):
    """series: [(name, color, [(x, y), ...])]."""
    pts = [p for _, _, ps in series for p in ps if p[0] > 0 and p[1] > 0]
    if not pts:
        return False
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys) * 0.8, max(ys) * 1.25
    L, R, T, B = 70, 150, 80, 60
    fx = (lambda v: math.log(v)) if logx else (lambda v: v)
    fy = (lambda v: math.log(v)) if logy else (lambda v: v)
    px = lambda v: L + (fx(v) - fx(xlo)) / ((fx(xhi) - fx(xlo)) or 1) * (w - L - R)
    py = lambda v: h - B - (fy(v) - fy(ylo)) / ((fy(yhi) - fy(ylo)) or 1) * (h - T - B)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" {FONT}>',
         '<rect width="100%" height="100%" fill="#fff"/>',
         f'<text x="16" y="24" font-size="15" font-weight="bold" fill="#111">{esc(title)}</text>',
         f'<text x="16" y="44" fill="#555">{esc(subtitle)}</text>']
    for t in _ticks(ylo, yhi):
        s.append(f'<line x1="{L}" x2="{w-R}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="#e5e5e5"/>'
                 f'<text x="{L-6}" y="{py(t)+4:.1f}" text-anchor="end" fill="#444">{t:g}</text>')
    for t in _ticks(xlo, xhi):
        s.append(f'<line x1="{px(t):.1f}" x2="{px(t):.1f}" y1="{T-6}" y2="{h-B}" stroke="#eee"/>'
                 f'<text x="{px(t):.1f}" y="{h-B+16}" text-anchor="middle" fill="#444">{t:g}</text>')
    s.append(f'<text x="{(L+w-R)/2}" y="{h-12}" text-anchor="middle" fill="#333">{esc(xlabel)}</text>')
    s.append(f'<text x="16" y="{T-14}" fill="#333">{esc(ylabel)}</text>')
    for k, (name, col, ps) in enumerate(series):
        ps = sorted(p for p in ps if p[0] > 0 and p[1] > 0)
        if not ps:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{px(a):.1f},{py(b):.1f}' for i, (a, b) in enumerate(ps))
        s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for a, b in ps:
            s.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.6" fill="{col}"/>')
        s.append(f'<rect x="{w-R+12}" y="{T+k*18}" width="10" height="10" fill="{col}"/>'
                 f'<text x="{w-R+27}" y="{T+k*18+9}" fill="#333">{esc(name)}</text>')
    s.append("</svg>")
    open(out, "w").write("\n".join(s))
    return True


PAIRED_ROW = re.compile(r"^\|\s*(?P<row>[^|]+?)\s*\|\s*(?P<o>[\d.]+)\s*/\s*(?P<om>[\d.]+)\s*\|\s*(?P<t>[\d.]+)\s*/\s*(?P<tm>[\d.]+)\s*\|\s*(?P<r>[\d.]+)x\s*\[(?P<rlo>[\d.]+)-(?P<rhi>[\d.]+)\]\s*\|(?P<rest>.*)$")


def parse_paired(path, missing=None):
    """Paired GPU-vs-torch summary.md -> {lane: [row dict]}. ratio = torch/ojas,
    so time ratio ojas/torch = 1/ratio (lower is faster). Rows of a "vs" lane
    table that carry no ratio (refused, failed) are appended to `missing`
    as {lane: [(row, reason)]} so the caller can say they were not charted."""
    lanes, cur = {}, None
    for line in open(path):
        if line.startswith("### "):
            cur = line[4:].strip()
            lanes[cur] = []
            continue
        m = PAIRED_ROW.match(line)
        if (m is None and missing is not None and cur is not None and " vs " in cur
                and line.startswith("| ") and not line.startswith(("| row", "| :"))):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            reason = max(cells[1:], key=len).lstrip("* ")
            missing.setdefault(cur, []).append((cells[0], reason[:110]))
        if m and cur is not None:
            r = float(m["r"])
            om, tm = float(m["om"]), float(m["tm"])
            row = dict(name=m["row"], hollow="noisy" in m["rest"], ojas_ms=om, torch_ms=tm)
            if r < 0.05 and om > 0 and tm > 0:
                # two printed decimals are too coarse here (0.01x spans 0.005x..0.015x):
                # use the ratio of the printed medians and drop the whisker
                row.update(v=om / tm, mid=om / tm)
            elif r > 0:
                row.update(v=1 / r, mid=1 / r,
                           lo=1 / float(m["rhi"]) if float(m["rhi"]) > 0 else 1 / r,
                           hi=1 / float(m["rlo"]) if float(m["rlo"]) > 0 else 1 / r)
            else:
                continue
            lanes[cur].append(row)
    return {k: v for k, v in lanes.items() if v}
