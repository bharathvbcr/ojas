"""probe.py on the real E4B weights: Core ML fp16 / int4 against tessl's Q4 GEMV.

Throwaway analysis (house policy: Python only as an oracle or analysis that never ships).
Same chain as probe.py (24 one-token linears alternating 2560 -> 10240 -> 2560 with a tanh
GELU), but the weights are layers 0..11's real `mlp.gate_proj` and `mlp.down_proj` from the
MLX 4-bit snapshot, dequantized (`scale * q + bias`, group 64) to fp16. An RMS normalisation
between linears keeps real activations inside fp16. Each arm's output is also compared with
an fp32 numpy run of the same chain, so this is a parity check as well as a timing.
The safetensors file is read with numpy only (header + memmap), never executed.

usage: python -I probe_real.py SNAPSHOT_DIR OUTDIR
"""
import json
import pathlib
import statistics
import struct
import sys
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

D, F, PAIRS, REPS, WARM, EPS = 2560, 10240, 12, 30, 5, 1e-6
snap = pathlib.Path(sys.argv[1]) / "model.safetensors"
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

with open(snap, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen))
blob = np.memmap(snap, dtype=np.uint8, mode="r", offset=8 + hlen)


def raw(name):
    meta = header[name]
    a, b = meta["data_offsets"]
    return blob[a:b], meta


def bf16(name):
    buf, meta = raw(name)
    bits = np.frombuffer(buf.tobytes(), dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(meta["shape"])


def dequant(prefix):
    buf, meta = raw(prefix + ".weight")
    packed = np.frombuffer(buf.tobytes(), dtype=np.uint32).reshape(meta["shape"])
    rows, words = packed.shape
    q = np.empty((rows, words * 8), dtype=np.float32)
    for i in range(8):  # nibble i of each word is element 8*w + i
        q[:, i::8] = (packed >> (4 * i)) & 0xF
    scales, biases = bf16(prefix + ".scales"), bf16(prefix + ".biases")
    g = q.shape[1] // scales.shape[1]
    return (q.reshape(rows, -1, g) * scales[..., None] + biases[..., None]).reshape(rows, -1)


weights32 = []
for layer in range(PAIRS):
    p = f"language_model.model.layers.{layer}.mlp."
    weights32.append(dequant(p + "gate_proj"))  # [10240, 2560]
    weights32.append(dequant(p + "down_proj"))  # [2560, 10240]
assert [w.shape for w in weights32[:2]] == [(F, D), (D, F)], [w.shape for w in weights32[:2]]
weights = [w.astype(np.float16) for w in weights32]


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def reference(x):
    x = x.astype(np.float32)
    for w in weights32:
        x = x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS))
        x = gelu_tanh(x @ w.T)
    return x.ravel()


@mb.program(input_specs=[mb.TensorSpec(shape=(1, D), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog(x):
    for w in weights:
        ms = mb.reduce_mean(x=mb.mul(x=x, y=x), axes=[-1], keep_dims=True)
        x = mb.mul(x=x, y=mb.rsqrt(x=ms, epsilon=EPS))
        x = mb.linear(x=x, weight=w)
        x = mb.gelu(x=x, mode="TANH_APPROXIMATION")
    return x


base = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                  compute_precision=ct.precision.FLOAT16)
cfg = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=64))
q4 = ct.optimize.coreml.linear_quantize_weights(base, cfg)
paths = {"fp16": out / "chain_real_fp16.mlpackage", "int4": out / "chain_real_int4.mlpackage"}
base.save(str(paths["fp16"]))
q4.save(str(paths["int4"]))

n_weights = sum(w.size for w in weights)
bytes_per = {"fp16": n_weights * 2, "int4": n_weights // 2 + (n_weights // 64) * 2}
rng = np.random.default_rng(0)
x16 = rng.standard_normal((1, D)).astype(np.float16)
ref = reference(x16)
rows = []
units = {"CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
         "ALL": ct.ComputeUnit.ALL}
for arm, path in paths.items():
    for uname, unit in units.items():
        m = ct.models.MLModel(str(path), compute_units=unit)
        y = None
        for _ in range(WARM):
            y = list(m.predict({"x": x16}).values())[0]
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            m.predict({"x": x16})
            ts.append(time.perf_counter() - t0)
        y = np.asarray(y, dtype=np.float32).ravel()
        rel = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
        row = {"arm": arm, "units": uname, "min_us_per_layer": min(ts) / len(weights) * 1e6,
               "median_us_per_layer": statistics.median(ts) / len(weights) * 1e6,
               "GBps_at_min": bytes_per[arm] / min(ts) / 1e9,
               "rel_l2_vs_fp32_reference": rel, "finite": bool(np.isfinite(y).all())}
        rows.append(row)
        print(json.dumps(row), flush=True)
(out / "results_real.json").write_text(json.dumps({"layers": len(weights), "reps": REPS, "rows": rows}, indent=1))
