"""torch MPS side of the paired GPU-vs-torch benchmark (see bench/README.md).

Measurement tool and reference oracle only; nothing here ships.

Every row mirrors one row of bench/ojas_rows.rs: same name, same input
`spec` string, same inputs bit for bit (the shared integer hash `_hash`, with
power-of-two scales so the float conversion is exact on both sides).

Modes
  ref   --ref DIR            write DIR/<row>.f32 (sampled outputs) + DIR/<row>.spec,
                             and DIR/generator.f32 (the generator check the ojas
                             binaries run before any row)
  time  --ref DIR --out F    time every row (warmup >= 5, iters >= 20, each
                             iteration = op + torch.mps.synchronize()), one JSON
                             object per row appended to F
  env   --out F              versions and device as JSON

Row semantics
  *_fwd  forward under torch.no_grad()
  *_bwd  the graph is built once outside the timer; the timed region is
         torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=True),
         i.e. torch's backward kernels on its saved tensors
  muon_* nanolab Muon.step for one matrix; *_fp32 runs nanolab's
         zeropower_via_newtonschulz5 with the bf16 cast replaced by fp32 (the
         reference for ojas, whose NS5 is ExactF32 GEMMs); *_bf16 is nanolab's
         own function unchanged (timing only, labelled)
  block_* nanolab's real Block (attention mixer, SwiGLU), weights loaded from
         the shared generator, v0 given so the value-residual path runs
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write .pyc into the nanolab checkout

import argparse
import json
import os
import platform
import statistics
import time
from collections import Counter

import torch
import torch.nn.functional as F

NANOLAB_ROOT = "/Users/bharath/Code/research/MLSystemsLab"
sys.path.insert(0, NANOLAB_ROOT)

from nanolab.config import Config  # noqa: E402
from nanolab.mixers import apply_rope  # noqa: E402
from nanolab.model import GPT, Block  # noqa: E402
from nanolab.optim import zeropower_via_newtonschulz5  # noqa: E402

DEV = torch.device("mps")
B, T, H, D, DM, FF, V = 4, 1024, 12, 64, 768, 2048, 50304
N = B * T
EPS = 1e-6
SAMPLE_MAX = 1 << 22
M32 = 0xFFFFFFFF


def sync():
    torch.mps.synchronize()


# ---------------------------------------------------------------------------
# shared generator (bench/ojas_rows.rs::hash32 / gen / gen_targets)
# ---------------------------------------------------------------------------
def _hash(i, seed):
    x = (i * 0x7FEB352D + seed * 0x2C1B3C6D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x297A2D39) & M32
    x = x ^ (x >> 12)
    x = (x * 0x2C1B3C6D) & M32
    x = x ^ (x >> 15)
    return x


def gen(shape, seed, e):
    n = 1
    for s in shape:
        n *= s
    out = torch.empty(n, dtype=torch.float32, device=DEV)
    ch = 1 << 24
    for s in range(0, n, ch):
        i = torch.arange(s, min(n, s + ch), dtype=torch.int64, device=DEV)
        out[s:s + ch] = ((_hash(i, seed) >> 8).to(torch.float32) * (2.0 / 16777216.0) - 1.0) * (2.0 ** e)
    return out.view(*shape)


def gen_targets(n, seed):
    i = torch.arange(n, dtype=torch.int64, device=DEV)
    return _hash(i, seed) % V


def spec(parts):
    return ";".join(f"{n}:{'x'.join(str(d) for d in s)}:{seed}:{e}" for n, s, seed, e in parts)


def sample(t):
    flat = t.detach().reshape(-1).float()
    n = flat.numel()
    if n <= SAMPLE_MAX:
        return flat.cpu()
    stride = n // SAMPLE_MAX
    return flat[0:stride * SAMPLE_MAX:stride].cpu()


# ---------------------------------------------------------------------------
# rows. Each row: (name, spec, make) where make() returns a dict with
#   "outs": () -> list[Tensor]     (reference outputs, from the initial state)
#   "step": () -> object            (one timed iteration, before the sync)
# make() is called once per mode; stateful rows build fresh state each time.
# ---------------------------------------------------------------------------
ROWS = []


def row(name, sp):
    def deco(make):
        ROWS.append((name, sp, make))
        return make
    return deco


def grad_row(name, sp, build):
    """build() -> (outputs, inputs, grad_outputs) with a live graph."""
    def make():
        outs, ins, gys = build()

        def run():
            return torch.autograd.grad(outs, ins, gys, retain_graph=True)
        return {"outs": lambda: list(run()), "step": run}
    ROWS.append((name, sp, make))


def fwd_row(name, sp, fn):
    def make():
        def run():
            with torch.no_grad():
                r = fn()
            return r if isinstance(r, (list, tuple)) else [r]
        return {"outs": lambda: list(run()), "step": run}
    ROWS.append((name, sp, make))


def req(t):
    return t.detach().clone().requires_grad_(True)


# floor -----------------------------------------------------------------------
def _floor():
    x = gen([1], 1, 0)
    fwd_row("floor_silu_1", spec([("x", [1], 1, 0)]), lambda: F.silu(x))


# linear ----------------------------------------------------------------------
def _linear(tag, rows, kin, nout):
    xs, ws, gs = [rows, kin], [nout, kin], [rows, nout]
    sf = spec([("x", xs, 11, 0), ("w", ws, 12, -5)])
    sb = spec([("x", xs, 11, 0), ("w", ws, 12, -5), ("gy", gs, 13, 0)])
    inp = lambda: cached(f"lin{tag}", lambda: (gen(xs, 11, 0), gen(ws, 12, -5)))  # noqa: E731
    fwd_row(f"linear_{tag}_fwd", sf, lambda: F.linear(*inp()))

    def build():
        xi, wi = (req(t) for t in inp())
        return [F.linear(xi, wi)], [xi, wi], [gen(gs, 13, 0)]
    grad_row(f"linear_{tag}_bwd", sb, build)


# causal SDPA -------------------------------------------------------------------
def _sdpa(tag, s):
    sf = spec([("q", s, 21, 0), ("k", s, 22, 0), ("v", s, 23, 0)])
    sb = sf + ";" + spec([("gy", s, 24, 0)])

    def qkv():
        return gen(s, 21, 0), gen(s, 22, 0), gen(s, 23, 0)
    fwd_row(f"sdpa_{tag}_fwd", sf,
            lambda: F.scaled_dot_product_attention(*cached(tag, qkv), is_causal=True))

    def build():
        q, k, v = (req(t) for t in cached(tag, qkv))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return [o], [q, k, v], [gen(s, 24, 0)]
    grad_row(f"sdpa_{tag}_bwd", sb, build)


_CACHE = {}


def cached(key, mk):
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = mk()
    return _CACHE[key]


# small ops ---------------------------------------------------------------------
def _rms():
    xs, ws = [N, DM], [DM]
    sf = spec([("x", xs, 31, 0), ("w", ws, 32, 0)])
    sb = sf + ";" + spec([("gy", xs, 33, 0)])
    inp = lambda: cached("rms", lambda: (gen(xs, 31, 0), gen(ws, 32, 0)))  # noqa: E731
    fwd_row("rms_norm_fwd", sf, lambda: F.rms_norm(inp()[0], (DM,), inp()[1], EPS))

    def build():
        x, w = (req(t) for t in inp())
        return [F.rms_norm(x, (DM,), w, EPS)], [x, w], [gen(xs, 33, 0)]
    grad_row("rms_norm_bwd", sb, build)


def _qk():
    s, ws = [B, T, H, D], [D]
    sf = spec([("q", s, 41, 0), ("k", s, 42, 0), ("qw", ws, 43, 0), ("kw", ws, 44, 0)])
    sb = sf + ";" + spec([("gq", s, 45, 0), ("gk", s, 46, 0)])
    inp = lambda: cached("qk", lambda: (gen(s, 41, 0), gen(s, 42, 0), gen(ws, 43, 0), gen(ws, 44, 0)))  # noqa: E731

    def fwd():
        q, k, qw, kw = inp()
        return [F.rms_norm(q, (D,), qw, EPS), F.rms_norm(k, (D,), kw, EPS)]
    fwd_row("rms_qk_norm_fwd", sf, fwd)

    def build():
        q, k, qw, kw = (req(t) for t in inp())
        outs = [F.rms_norm(q, (D,), qw, EPS), F.rms_norm(k, (D,), kw, EPS)]
        return outs, [q, k, qw, kw], [gen(s, 45, 0), gen(s, 46, 0)]
    grad_row("rms_qk_norm_bwd", sb, build)


def _rope():
    s, cs = [B, T, H, D], [T, D]
    sf = spec([("x", s, 51, 0), ("cos", cs, 52, 0), ("sin", cs, 53, 0)])
    sb = spec([("gy", s, 51, 0), ("cos", cs, 52, 0), ("sin", cs, 53, 0)])
    inp = lambda: cached("rope", lambda: (gen(s, 51, 0), gen(cs, 52, 0), gen(cs, 53, 0)))  # noqa: E731
    fwd_row("rope_fwd", sf, lambda: apply_rope(*inp()))

    def build():
        x, c, sn = inp()
        xi = req(x)
        return [apply_rope(xi, c, sn)], [xi], [x]
    grad_row("rope_bwd", sb, build)


def _silu():
    s = [N, FF]
    sf = spec([("x", s, 61, 2)])
    sb = sf + ";" + spec([("gy", s, 62, 0)])
    inp = lambda: cached("silu", lambda: gen(s, 61, 2))  # noqa: E731
    fwd_row("silu_fwd", sf, lambda: F.silu(inp()))

    def build():
        x = req(inp())
        return [F.silu(x)], [x], [gen(s, 62, 0)]
    grad_row("silu_bwd", sb, build)


def _mul():
    s = [N, FF]
    sf = spec([("a", s, 71, 0), ("b", s, 72, 0)])
    sb = sf + ";" + spec([("gy", s, 73, 0)])
    inp = lambda: cached("mul", lambda: (gen(s, 71, 0), gen(s, 72, 0)))  # noqa: E731
    fwd_row("mul_fwd", sf, lambda: inp()[0] * inp()[1])

    def build():
        a, b = (req(t) for t in inp())
        return [a * b], [a, b], [gen(s, 73, 0)]
    grad_row("mul_bwd", sb, build)


def _add():
    s = [N, DM]
    sf = spec([("x", s, 81, 0), ("y", s, 82, 0)])
    sb = sf + ";" + spec([("gy", s, 83, 0)])
    inp = lambda: cached("add", lambda: (gen(s, 81, 0), gen(s, 82, 0)))  # noqa: E731
    fwd_row("residual_add_fwd", sf, lambda: inp()[0] + inp()[1])

    def build():
        a, b = (req(t) for t in inp())
        return [a + b], [a, b], [gen(s, 83, 0)]
    grad_row("residual_add_bwd", sb, build)


def _gate_fn(x, w, b, a):
    return a * torch.sigmoid(F.linear(x, w, b)).unsqueeze(-1)


def _gate():
    xs, ws, bs, as_ = [B, T, DM], [H, DM], [H], [B, T, H, D]
    sf = spec([("x", xs, 91, 0), ("w", ws, 92, -5), ("b", bs, 93, 0), ("attn", as_, 94, 0)])
    sb = sf + ";" + spec([("gy", as_, 95, 0)])
    inp = lambda: cached("gate", lambda: (gen(xs, 91, 0), gen(ws, 92, -5), gen(bs, 93, 0), gen(as_, 94, 0)))  # noqa: E731
    fwd_row("gate_fwd", sf, lambda: _gate_fn(*inp()))

    def build():
        ins = [req(t) for t in inp()]
        return [_gate_fn(*ins)], ins, [gen(as_, 95, 0)]
    grad_row("gate_bwd", sb, build)


def _vres_fn(v, v0, lam):
    s = torch.sigmoid(lam)
    return (1 - s) * v + s * v0


def _vres():
    s = [B, T, H, D]
    sf = spec([("v", s, 101, 0), ("v0", s, 102, 0), ("lam", [1], 103, 0)])
    sb = sf + ";" + spec([("gy", s, 104, 0)])
    inp = lambda: cached("vres", lambda: (gen(s, 101, 0), gen(s, 102, 0), gen([1], 103, 0)))  # noqa: E731
    fwd_row("vres_fwd", sf, lambda: _vres_fn(*inp()))

    def build():
        ins = [req(t) for t in inp()]
        return [_vres_fn(*ins)], ins, [gen(s, 104, 0)]
    grad_row("vres_bwd", sb, build)


def _permute():
    s = [B, T, H, D]
    inp = lambda: cached("perm", lambda: gen(s, 111, 0))  # noqa: E731
    fwd_row("permute_bthd_bhtd", spec([("x", s, 111, 0)]),
            lambda: inp().permute(0, 2, 1, 3).contiguous())


def _ce():
    ls = [N, V]
    sp = spec([("logits", ls, 121, 2), ("targets", [N], 122, 0)])
    inp = lambda: cached("ce", lambda: (gen(ls, 121, 2), gen_targets(N, 122)))  # noqa: E731
    fwd_row("cross_entropy_fwd", sp, lambda: F.cross_entropy(*inp()))

    def build():
        l, t = inp()
        li = req(l)
        return [F.cross_entropy(li, t)], [li], None
    grad_row("cross_entropy_bwd", sp, build)


# optimizer rows -------------------------------------------------------------------
def param_shapes():
    v = [[V, DM]]
    for _ in range(12):
        v += [[DM], [1], [DM, DM], [DM, DM], [DM, DM], [DM, DM], [D], [D], [H, DM], [H],
              [DM], [FF, DM], [FF, DM], [DM, FF]]
    v.append([DM])
    return v


def check_param_shapes():
    """The list must be the real nanolab default model's parameters."""
    real = Counter(tuple(p.shape) for p in GPT(Config()).parameters())
    mine = Counter(tuple(s) for s in param_shapes())
    if real != mine:
        raise SystemExit(f"param_shapes() != nanolab GPT(Config()) parameters: {real} vs {mine}")
    return sum(p.numel() for p in GPT(Config()).parameters()), len(param_shapes())


def _clip():
    shapes = param_shapes()
    sp = "clip:params:1000:-6:max1.0"

    def make():
        ps = []
        for i, s in enumerate(shapes):
            p = torch.zeros(s, device=DEV, requires_grad=True)
            p.grad = gen(s, 1000 + i, -6)
            ps.append(p)
        st = {"max": 1.0}

        def outs():
            n = torch.nn.utils.clip_grad_norm_(ps, 1.0)
            st["max"] = 0.9 * n.item()
            return [n.reshape(1), ps[0].grad, ps[-1].grad]

        def step():
            return torch.nn.utils.clip_grad_norm_(ps, st["max"])

        def after(n):
            st["max"] = 0.9 * n.item()
        return {"outs": outs, "step": step, "after": after}
    ROWS.append(("clip_grad_norm_full", sp, make))


def _adamw():
    shapes = param_shapes()
    sp = "adamw:p2000:-5:g3000:-6:lr1e-3:b0.9,0.95:eps1e-8:wd0.1"

    def make():
        ps = []
        for i, s in enumerate(shapes):
            p = gen(s, 2000 + i, -5).requires_grad_(True)
            p.grad = gen(s, 3000 + i, -6)
            ps.append(p)
        opt = torch.optim.AdamW(ps, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        mid, last = len(ps) // 2, len(ps) - 1

        def outs():
            opt.step()
            return [ps[0], ps[mid], ps[last], opt.state[ps[mid]]["exp_avg_sq"]]
        return {"outs": outs, "step": opt.step}
    ROWS.append(("adamw_full", sp, make))


def ns5_fp32(G, steps=5, eps=1e-7):
    """nanolab.optim.zeropower_via_newtonschulz5 with `G.bfloat16()` replaced
    by `G.float()`. Every other line is the original."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


@torch.no_grad()
def muon_step(p, g, buf, ns, lr=0.025, mom=0.99, wd=0.1, nesterov=True):
    """nanolab Muon.step for one matrix (the per-parameter body of its loop)."""
    m, n = p.shape
    buf.mul_(mom).add_(g)
    upd = g.add(buf, alpha=mom) if nesterov else buf
    X = ns(upd)
    scale = max(1.0, m / n) ** 0.5
    if wd:
        p.mul_(1 - lr * wd)
    p.add_(X, alpha=-lr * scale)


def _muon(rows, cols):
    s = [rows, cols]
    sp = spec([("p", s, 4001, -5), ("g", s, 4002, -6)])
    for label, ns in (("", ns5_fp32), ("_bf16", zeropower_via_newtonschulz5)):
        def make(ns=ns):
            p, g, buf = gen(s, 4001, -5), gen(s, 4002, -6), torch.zeros(s, device=DEV)

            def outs():
                muon_step(p, g, buf, ns)
                return [p, buf]
            return {"outs": outs, "step": lambda: muon_step(p, g, buf, ns)}
        ROWS.append((f"muon_{rows}x{cols}{label}", sp, make))


# composed block -----------------------------------------------------------------------
BLOCK_PARAMS = [
    ("norm1.weight", [DM], 501, 0),
    ("mixer.vr_lambda", [1], 502, 0),
    ("mixer.q_proj.weight", [DM, DM], 503, -5),
    ("mixer.k_proj.weight", [DM, DM], 504, -5),
    ("mixer.v_proj.weight", [DM, DM], 505, -5),
    ("mixer.o_proj.weight", [DM, DM], 506, -5),
    ("mixer.q_norm.weight", [D], 507, 0),
    ("mixer.k_norm.weight", [D], 508, 0),
    ("mixer.gate.weight", [H, DM], 509, -5),
    ("mixer.gate.bias", [H], 510, 0),
    ("norm2.weight", [DM], 511, 0),
    ("ffn.gate.weight", [FF, DM], 512, -5),
    ("ffn.up.weight", [FF, DM], 513, -5),
    ("ffn.down.weight", [DM, FF], 514, -5),
]


def _block():
    xs, cs, vs = [B, T, DM], [T, D], [B, T, H, D]
    parts = list(BLOCK_PARAMS) + [("x", xs, 520, 0), ("cos", cs, 521, 0), ("sin", cs, 522, 0),
                                  ("v0", vs, 523, 0)]
    sf = spec(parts)
    sb = sf + ";" + spec([("gy", xs, 524, 0)])

    def build_block():
        blk = Block(Config()).to(DEV)
        sd = blk.state_dict()
        if sorted(sd) != sorted(n for n, *_ in BLOCK_PARAMS):
            raise SystemExit(f"nanolab Block state_dict keys changed: {sorted(sd)}")
        blk.load_state_dict({n: gen(s, seed, e) for n, s, seed, e in BLOCK_PARAMS})
        return blk, gen(xs, 520, 0), gen(cs, 521, 0), gen(cs, 522, 0), gen(vs, 523, 0)

    def make_fwd():
        blk, x, c, s, v0 = build_block()

        def run():
            with torch.no_grad():
                return [blk(x, c, s, v0)[0]]
        return {"outs": run, "step": run}
    ROWS.append(("block_fwd", sf, make_fwd))

    def make_bwd():
        blk, x, c, s, v0 = build_block()
        gy = gen(xs, 524, 0)
        xi, v0i = req(x), req(v0)
        mx = blk.mixer

        def run():
            for p in blk.parameters():
                p.grad = None
            xi.grad = None
            v0i.grad = None
            y, _ = blk(xi, c, s, v0i)
            y.backward(gy)
            return y

        def outs():
            y = run()
            return [y, xi.grad, mx.q_proj.weight.grad, blk.ffn.down.weight.grad,
                    mx.gate.weight.grad, mx.vr_lambda.grad, blk.norm1.weight.grad, v0i.grad]
        return {"outs": outs, "step": run}
    ROWS.append(("block_fwd_bwd", sb, make_bwd))


def register():
    _floor()
    _linear("qkv", N, DM, DM)
    _linear("up", N, DM, FF)
    _linear("down", N, FF, DM)
    _linear("lmhead", N, DM, V)
    _sdpa("b4h12t1024d64", [4, 12, 1024, 64])
    _sdpa("b4h8t2048d64", [4, 8, 2048, 64])
    _sdpa("b2h8t1024d128", [2, 8, 1024, 128])
    _rms()
    _qk()
    _rope()
    _silu()
    _mul()
    _add()
    _gate()
    _vres()
    _permute()
    _ce()
    _clip()
    _adamw()
    for r, c in ((DM, DM), (FF, DM), (DM, FF)):
        _muon(r, c)
    _block()


# ---------------------------------------------------------------------------
def wanted(name, filt):
    return not filt or any(name.startswith(f) for f in filt)


def write_generator(ref):
    vals = gen([64], 7, 0).cpu()
    far = 1 << 27
    i = torch.arange(far, far + 64, dtype=torch.int64, device=DEV)
    far_vals = ((_hash(i, 7) >> 8).to(torch.float32) * (2.0 / 16777216.0) - 1.0).cpu()
    tg = gen_targets(64, 7).to(torch.int64).cpu()
    raw = vals.numpy().tobytes() + far_vals.numpy().tobytes() + tg.to(torch.int32).numpy().tobytes()
    with open(os.path.join(ref, "generator.f32"), "wb") as f:
        f.write(raw)


def do_ref(ref, filt):
    os.makedirs(ref, exist_ok=True)
    write_generator(ref)
    for name, sp, make in ROWS:
        if not wanted(name, filt) or name.endswith("_bf16"):
            continue
        st = make()
        outs = st["outs"]()
        sync()
        samples = [sample(o) for o in outs]
        with open(os.path.join(ref, name + ".f32"), "wb") as f:
            for s in samples:
                f.write(s.contiguous().numpy().astype("<f4").tobytes())
        with open(os.path.join(ref, name + ".spec"), "w") as f:
            f.write(sp + "\n")
            f.write("counts=" + ",".join(str(s.numel()) for s in samples) + "\n")
        print(f"ref {name}: {[tuple(o.shape) for o in outs]}", file=sys.stderr)
        del st, outs, samples
        _CACHE.clear()
        torch.mps.empty_cache()


def do_time(out, filt, warmup, iters):
    with open(out, "a") as fo:
        def emit(d):
            line = json.dumps(d)
            fo.write(line + "\n")
            fo.flush()
            print(line, file=sys.stderr)
        emit({"runtime": "torch-mps", "row": "_device", "status": "info",
              "device": {"torch": torch.__version__, "mps": True}})
        for name, sp, make in ROWS:
            if not wanted(name, filt):
                continue
            try:
                st = make()
                st["outs"]()  # the same first call the ojas side makes for parity
                sync()
                after = st.get("after")
                ms = []
                for k in range(warmup + iters):
                    t0 = time.perf_counter()
                    r = st["step"]()
                    sync()
                    dt = (time.perf_counter() - t0) * 1e3
                    if after is not None:
                        after(r)
                    del r
                    if k >= warmup:
                        ms.append(dt)
                emit({"runtime": "torch-mps", "row": name, "status": "ok",
                      "min_ms": min(ms), "median_ms": statistics.median(ms),
                      "iters": iters, "warmup": warmup})
            except Exception as e:  # recorded per row, not swallowed
                emit({"runtime": "torch-mps", "row": name, "status": "error", "detail": repr(e)})
            finally:
                st = None
                _CACHE.clear()
                torch.mps.empty_cache()


def do_env(out):
    total, count = check_param_shapes()
    d = {"python": platform.python_version(), "python_exe": sys.executable,
         "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
         "nanolab_params": total, "nanolab_param_tensors": count,
         "platform": platform.platform()}
    with open(out, "w") as f:
        json.dump(d, f, indent=1)
    print(json.dumps(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "time", "env"])
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP", 5)))
    ap.add_argument("--iters", type=int, default=int(os.environ.get("BENCH_ITERS", 20)))
    args = ap.parse_args()
    filt = [f for f in os.environ.get("BENCH_ROWS", "").split(",") if f]
    if args.warmup < 5 or args.iters < 20:
        raise SystemExit("the protocol needs warmup >= 5 and iters >= 20")
    torch.manual_seed(0)
    register()
    if args.mode == "env":
        do_env(args.out)
    elif args.mode == "ref":
        check_param_shapes()
        do_ref(args.ref, filt)
    else:
        do_time(args.out, filt, args.warmup, args.iters)


if __name__ == "__main__":
    main()
