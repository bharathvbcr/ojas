"""Gemma E4B decode step exported to Core ML (stateful KV cache), random weights: timing only.

Throwaway analysis, never shipped (house policy). One decode step: token embedding in, greedy token
id out. Per layer, at E4B shapes (gemma-metal src/config.rs e4b_preset): RMSNorm, q/k/v (8 query
heads over 2 KV heads, head dim 256, 512 on every sixth "global" layer), the KV cache as Core ML
state updated in place, softmax attention over the cache with a position mask, o projection, RMSNorm,
GELU-tanh gate/up 2560 -> 10240, down, and a per-layer-input gate/projection (2560 -> 256 -> 2560).
Then the final norm, the tied 262,144-row output head and an argmax.

Simplified against gemma-metal, all small in bytes: no RoPE, no q/k norms, no logit softcap, no
per-layer-input embedding table (one row per token), and every layer has its own k/v projection
(E4B shares the KV of its last 18 layers).

A whole-model export does not fit the mac_heavy footprint cap during conversion (a 6-layer model with
the head was killed), so three programs are exported and timed separately:
  block6 : 6 layers (5 sliding + 1 global, one period of E4B's pattern), embedding in, hidden out;
  head   : final norm, the 262,144-row output head and the argmax;
  noop   : one add, the per-predict overhead.
The head program holds a quarter of the rows (65,536) to stay under the cap; a GEMV of this shape is
bound by its row bytes, so the full head is taken as 4 x head - 3 x noop. A 42-layer token is then
estimated as 7 x block6 + 4 x head - 10 x noop (one predict per token, not eleven).
Arms: fp16 weights, and int4 (linear_quantize_weights, symmetric, per block of 64; MLX Q4 uses 64).

usage: python -I export_decode.py SCRATCH_DIR RESULTS_JSON
"""
import gc
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

D, F, VOCAB, H, HKV, S = 2560, 10240, 262_144, 8, 2, 256
HD_S, HD_G, PLE = 256, 512, 256
STEPS, WARM = 48, 8
scratch, results_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
f16 = torch.float16


def lin(i, o):
    m = nn.Linear(i, o, bias=False, dtype=f16)
    with torch.no_grad():
        m.weight.normal_(0.0, 0.02)
    return m


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


class Layer(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.hd = hd
        self.q, self.k, self.v, self.o = lin(D, H * hd), lin(D, HKV * hd), lin(D, HKV * hd), lin(H * hd, D)
        self.gate, self.up, self.down = lin(D, F), lin(D, F), lin(F, D)
        self.ple_gate, self.ple_proj = lin(D, PLE), lin(PLE, D)
        self.n1 = nn.Parameter(torch.ones(D, dtype=f16), requires_grad=False)
        self.n2 = nn.Parameter(torch.ones(D, dtype=f16), requires_grad=False)

    def forward(self, x, kc, vc, onehot, mask):
        hd = self.hd
        h = rms(x, self.n1)
        q = self.q(h).view(1, 1, H, hd).transpose(1, 2)
        k = self.k(h).view(1, 1, HKV, hd).transpose(1, 2)
        v = self.v(h).view(1, 1, HKV, hd).transpose(1, 2)
        kc = kc * (1 - onehot) + k * onehot
        vc = vc * (1 - onehot) + v * onehot
        kr, vr = kc.repeat_interleave(H // HKV, dim=1), vc.repeat_interleave(H // HKV, dim=1)
        att = torch.softmax(q @ kr.transpose(2, 3) * (hd ** -0.5) + mask, dim=-1) @ vr
        x = x + self.o(att.transpose(1, 2).reshape(1, 1, H * hd))
        h = rms(x, self.n2)
        x = x + self.down(nn.functional.gelu(self.gate(h), approximate="tanh") * self.up(h))
        x = x + self.ple_proj(nn.functional.gelu(self.ple_gate(x), approximate="tanh"))
        return x, kc, vc


class Decode(nn.Module):
    def __init__(self, layers, head=True):
        super().__init__()
        self.with_head = head
        self.kinds = ["g" if (i + 1) % 6 == 0 else "s" for i in range(layers)]
        self.layers = nn.ModuleList([Layer(HD_G if k == "g" else HD_S) for k in self.kinds])
        ns, ng = self.kinds.count("s"), self.kinds.count("g")
        self.register_buffer("ks", torch.zeros(ns, HKV, S, HD_S, dtype=f16))
        self.register_buffer("vs", torch.zeros(ns, HKV, S, HD_S, dtype=f16))
        self.register_buffer("kg", torch.zeros(ng, HKV, S, HD_G, dtype=f16))
        self.register_buffer("vg", torch.zeros(ng, HKV, S, HD_G, dtype=f16))
        self.norm = nn.Parameter(torch.ones(D, dtype=f16), requires_grad=False)
        self.head = lin(D, VOCAB) if head else None

    def forward(self, x, onehot, mask):
        si = gi = 0
        for kind, layer in zip(self.kinds, self.layers):
            if kind == "s":
                x, kc, vc = layer(x, self.ks[si:si + 1], self.vs[si:si + 1], onehot, mask)
                self.ks[si:si + 1] = kc
                self.vs[si:si + 1] = vc
                si += 1
            else:
                x, kc, vc = layer(x, self.kg[gi:gi + 1], self.vg[gi:gi + 1], onehot, mask)
                self.kg[gi:gi + 1] = kc
                self.vg[gi:gi + 1] = vc
                gi += 1
        if self.head is None:
            return x
        logits = self.head(rms(x, self.norm))
        return torch.argmax(logits, dim=-1).to(torch.int32)


def step_inputs(pos):
    onehot = np.zeros((1, 1, S, 1), np.float16)
    onehot[0, 0, pos, 0] = 1
    mask = np.full((1, 1, 1, S), -1e4, np.float16)
    mask[0, 0, 0, : pos + 1] = 0
    x = np.random.default_rng(pos).standard_normal((1, 1, D)).astype(np.float16)
    return {"x": x, "onehot": onehot, "mask": mask}


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.ones(D, dtype=f16), requires_grad=False)
        self.head = lin(D, VOCAB // 4)  # a quarter of the head, scaled x4 below (row-bandwidth bound)

    def forward(self, x):
        return torch.argmax(self.head(rms(x, self.norm)), dim=-1).to(torch.int32)


class Noop(nn.Module):
    def forward(self, x):
        return x + 1


def time_model(path, stateful):
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    state = m.make_state() if stateful else None
    ts = []
    for pos in range(WARM + STEPS):
        inp = step_inputs(pos)
        if not stateful:
            inp = {"x": inp["x"]}
        t0 = time.perf_counter()
        m.predict(inp, state=state) if stateful else m.predict(inp)
        if pos >= WARM:
            ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, statistics.median(ts) * 1e3


cfg = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=64))
xin = ct.TensorType(name="x", shape=(1, 1, D), dtype=np.float16)
rows = []
for name in ("noop", "head", "block6"):
    ex = step_inputs(0)
    if name == "block6":
        model = Decode(6, head=False).eval()
        traced = torch.jit.trace(model, tuple(torch.from_numpy(ex[k]) for k in ("x", "onehot", "mask")))
        states = [ct.StateType(wrapped_type=ct.TensorType(shape=tuple(getattr(model, n).shape), dtype=np.float16), name=n)
                  for n in ("ks", "vs", "kg", "vg")]
        inputs = [ct.TensorType(name=k, shape=ex[k].shape, dtype=np.float16) for k in ("x", "onehot", "mask")]
        outputs = [ct.TensorType(name="h", dtype=np.float16)]
    else:
        model = (Head() if name == "head" else Noop()).eval()
        traced = torch.jit.trace(model, (torch.from_numpy(ex["x"]),))
        states, inputs = None, [xin]
        outputs = [ct.TensorType(name="tok", dtype=np.int32)] if name == "head" else [ct.TensorType(name="y", dtype=np.float16)]
    n_weights = sum(p.numel() for n, p in model.named_parameters() if p.dim() == 2)
    del model
    gc.collect()
    kw = {"states": states} if states else {}
    fp = ct.convert(traced, inputs=inputs, outputs=outputs, minimum_deployment_target=ct.target.macOS15,
                    compute_precision=ct.precision.FLOAT16, convert_to="mlprogram", **kw)
    del traced
    gc.collect()
    arms = ["fp16"] if name == "noop" else ["fp16", "int4"]
    for arm in arms:
        mlm = fp if arm == "fp16" else ct.optimize.coreml.linear_quantize_weights(fp, cfg)
        path = scratch / f"{name}_{arm}.mlpackage"
        mlm.save(str(path))
        del mlm
        gc.collect()
        lo, med = time_model(path, stateful=name == "block6")
        nbytes = n_weights * 2 if arm == "fp16" else n_weights // 2 + (n_weights // 64) * 2
        row = {"program": name, "arm": arm, "ms_min": lo, "ms_median": med, "weight_GB": nbytes / 1e9,
               "GBps_at_median": nbytes / (med * 1e-3) / 1e9 if nbytes else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    del fp
    gc.collect()

get = lambda prog, arm: next(r["ms_median"] for r in rows if r["program"] == prog and r["arm"] == arm)
noop = get("noop", "fp16")
est = {arm: 7 * get("block6", arm) + 4 * get("head", arm) - 10 * noop for arm in ("fp16", "int4")}
out = {"steps": STEPS, "warm": WARM, "rows": rows, "estimated_42_layer_ms_per_token": est}
print(json.dumps(est))
results_path.write_text(json.dumps(out, indent=1))
