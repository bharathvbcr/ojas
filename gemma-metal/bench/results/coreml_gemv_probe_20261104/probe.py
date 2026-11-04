"""Core ML at Gemma E4B's dominant decode shape: does a Core ML export beat tessl's Q4 GEMV?

Throwaway analysis (house policy: Python only as an oracle or analysis that never ships).
A chain of LAYERS one-token linears alternating 2560 -> 10240 -> 2560 (E4B's MLP gate/up and
down shapes) with a GELU between, random weights, so each predict streams every weight once and the
chain is far larger than the system level cache. Arms:
  fp16 : fp16 weights (4 bytes per 2 weights, the "fp16 path")
  int4 : coremltools linear_quantize_weights, int4 symmetric, per_block 64 along the input dim
         (MLX Q4 uses group 64)
each on CPU_AND_GPU, CPU_AND_NE and ALL. Reports wall per layer (min and median over REPS predicts)
and the weight bytes streamed per second. Compare: tessl gemv_q4_mlx_simd, 59.4-64.2 us per layer at
these shapes, GPU time (../gemv_fraction_of_peak_20261104/README.md).

usage: python -I probe.py OUTDIR
"""
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

D, F, LAYERS, REPS, WARM = 2560, 10240, 24, 30, 5
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
weights = []
for i in range(LAYERS):
    shape = (F, D) if i % 2 == 0 else (D, F)  # linear weight is [out, in]
    weights.append((rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(np.float16))


@mb.program(input_specs=[mb.TensorSpec(shape=(1, D), dtype=types.fp16)], opset_version=ct.target.iOS18)
def prog(x):
    for w in weights:
        x = mb.linear(x=x, weight=w)
        x = mb.gelu(x=x, mode="TANH_APPROXIMATION")
    return x


base = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                  compute_precision=ct.precision.FLOAT16)
cfg = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=64))
q4 = ct.optimize.coreml.linear_quantize_weights(base, cfg)
paths = {"fp16": out / "chain_fp16.mlpackage", "int4": out / "chain_int4.mlpackage"}
base.save(str(paths["fp16"]))
q4.save(str(paths["int4"]))

n_weights = sum(w.size for w in weights)
bytes_per = {"fp16": n_weights * 2, "int4": n_weights // 2 + (n_weights // 64) * 2}  # + fp16 scales
x = {"x": rng.standard_normal((1, D)).astype(np.float16)}
ref = None
rows = []
units = {"CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
         "ALL": ct.ComputeUnit.ALL}
for arm, path in paths.items():
    for uname, unit in units.items():
        m = ct.models.MLModel(str(path), compute_units=unit)
        y = None
        for _ in range(WARM):
            y = list(m.predict(x).values())[0]
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            m.predict(x)
            ts.append(time.perf_counter() - t0)
        y = np.asarray(y, dtype=np.float32).ravel()
        if ref is None:
            ref = y
        rel = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
        row = {"arm": arm, "units": uname, "min_us_per_layer": min(ts) / LAYERS * 1e6,
               "median_us_per_layer": statistics.median(ts) / LAYERS * 1e6,
               "GBps_at_min": bytes_per[arm] / min(ts) / 1e9, "rel_l2_vs_first_arm": rel}
        rows.append(row)
        print(json.dumps(row), flush=True)
(out / "results.json").write_text(json.dumps({"layers": LAYERS, "reps": REPS, "rows": rows}, indent=1))
