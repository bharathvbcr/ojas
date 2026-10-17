"""torch.optim.AdamW over 100 steps: the LoRA recipe's optimizer, as a golden.

    /opt/homebrew/opt/python@3.14/bin/python3.14 -I ojas-oracle/python/adamw_oracle.py

(`-I` also drops the user site, where this machine keeps torch. If the import
below fails, run the same command without `-I`.)

Writes ojas-model/tests/fixtures/adamw_100/:

- `adamw_100.safetensors`  what ojas-model/tests/adamw_torch_100.rs reads.
- `adamw_100_fp64.safetensors`  the same run in float64, never read by Rust: it
  is the evidence for the bound.
- `manifest.json`  torch version, the exact optimizer arguments, the bound's
  derivation, and the sha256 of both files.

The recipe is Lappi campaign/lora-2b-preregistered.json: betas (0.9, 0.999),
eps 1e-8, decoupled weight decay 0.01 on every trainable tensor, lr 2e-4 for
the adapters and 1e-4 for D17 and the span head. Five scaled-down tensors
stand in for it (see TENSORS); `norm_nodecay` has weight decay 0 so the
decay-free branch is exercised too. lr is constant (the real run's schedule
only changes the lr passed per step, not the update rule).

Inputs are closed-form, so Python and Rust build identical f32 bits with no
shared RNG. For salt s, step, tensor t, index i:

    h  = fmix32(s*0x27D4EB2F + step*0x9E3779B1 + t*0x165667B1
                + i*0xC2B2AE35 + 0x85EBCA6B)          (all mod 2^32)
    q  = (h >> 8) - 2^23                               (in [-2^23, 2^23))
    parameter  = q(salt 1, step 0) * 2^-27             (|p| < 2^-4)
    grad       = (q(salt 2, step) + (q(salt 3, step 0) >> 1)) * 2^-33

The integer sum is below 2^24 in magnitude and the scale is a power of two,
so every value is exact in f32. `q(salt 3)` is a per-element offset that does
not change with the step, so the first moment has a signal and not only noise.
Two tensors get an all-zero gradient for whole steps (ZERO_GRAD).

The bound. Error is peak-relative: max|x - ref| / max|ref| per tensor. The gap
is that error between torch float32 and torch float64 over every tensor at
every checkpoint (parameters at 1, 10, 50, 100; both moments at 100): how far
torch's own float32 answer is from the exact one. An implementation that
rounds differently from torch can sit that far from torch on either side of
the exact answer, so the bound is 4 x gap rounded up to two significant
digits (twice the triangle sum, 2x margin). A real error is far above it: the
Rust test shows a wrong weight decay or step count lands outside.

Reference oracle only (house policy): CPU, one thread, deterministic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

try:
    import numpy as np
    import torch
    from safetensors import __version__ as SAFETENSORS_VERSION
    from safetensors.torch import load_file, save_file
except ImportError as err:
    raise SystemExit(
        f"cannot import torch/numpy/safetensors ({err}); torch lives in the user "
        "site, so run this script without `-I`"
    ) from err

GENERATOR = "ojas-oracle/python/adamw_oracle.py"
OUT = Path(__file__).resolve().parents[2] / "ojas-model" / "tests" / "fixtures" / "adamw_100"

STEPS = 100
CHECKPOINTS = (1, 10, 50, 100)
BETAS = (0.9, 0.999)
EPS = 1e-8
LR_ADAPTER, LR_HEAD = 2e-4, 1e-4

# (name, shape, lr, weight_decay). The index is the tensor id in the hash.
TENSORS = (
    ("lora_A", (16, 64), LR_ADAPTER, 0.01),
    ("lora_B", (64, 16), LR_ADAPTER, 0.01),
    ("d17", (17, 64), LR_HEAD, 0.01),
    ("span_head", (2, 64), LR_HEAD, 0.01),
    ("norm_nodecay", (64,), LR_HEAD, 0.0),
)
# (tensor id, first step, last step), 1-based and inclusive: the gradient is
# exactly zero for the whole tensor on those steps. Tensor 4 is zero from the
# first step, so its first update has m = v = 0.
ZERO_GRAD = ((1, 30, 39), (4, 1, 3))
PROBE_STEPS = (1, 2, 30, 50, 100)

MASK = 0xFFFFFFFF
K_SALT, K_STEP, K_TENSOR, K_INDEX, K_BASE = (
    0x27D4EB2F, 0x9E3779B1, 0x165667B1, 0xC2B2AE35, 0x85EBCA6B)
F1, F2 = 0x85EBCA6B, 0xC2B2AE35
SALT_INIT, SALT_GRAD, SALT_OFFSET = 1, 2, 3
INIT_EXP, GRAD_EXP = -27, -33


def hash32_py(salt: int, step: int, tensor: int, index: int) -> int:
    x = (salt * K_SALT + step * K_STEP + tensor * K_TENSOR + index * K_INDEX + K_BASE) & MASK
    x ^= x >> 16
    x = (x * F1) & MASK
    x ^= x >> 13
    x = (x * F2) & MASK
    return x ^ (x >> 16)


def hash32(salt: int, step: int, tensor: int, n: int) -> np.ndarray:
    base = (salt * K_SALT + step * K_STEP + tensor * K_TENSOR + K_BASE) & MASK
    x = np.arange(n, dtype=np.uint32) * np.uint32(K_INDEX) + np.uint32(base)
    x ^= x >> np.uint32(16)
    x = x * np.uint32(F1)
    x ^= x >> np.uint32(13)
    x = x * np.uint32(F2)
    x ^= x >> np.uint32(16)
    return x


def signed24(h):
    return (h >> 8) - (1 << 23)


def numel(shape) -> int:
    return math.prod(shape)


def zero_grad(step: int, tensor: int) -> bool:
    return any(t == tensor and lo <= step <= hi for t, lo, hi in ZERO_GRAD)


def init_ints(tensor: int, n: int) -> np.ndarray:
    return signed24(hash32(SALT_INIT, 0, tensor, n).astype(np.int64))


def grad_ints(step: int, tensor: int, n: int) -> np.ndarray:
    if zero_grad(step, tensor):
        return np.zeros(n, dtype=np.int64)
    noise = signed24(hash32(SALT_GRAD, step, tensor, n).astype(np.int64))
    offset = signed24(hash32(SALT_OFFSET, 0, tensor, n).astype(np.int64))
    return noise + (offset >> 1)


def scaled(ints: np.ndarray, exp: int) -> np.ndarray:
    if int(np.abs(ints).max(initial=0)) >= 1 << 24:
        raise SystemExit("integer does not fit f32 exactly")
    out = np.ldexp(ints.astype(np.float32), exp)
    if out.dtype != np.float32:
        raise SystemExit(f"ldexp returned {out.dtype}")
    return out


def initial(tensor: int) -> np.ndarray:
    name, shape, _, _ = TENSORS[tensor]
    return scaled(init_ints(tensor, numel(shape)), INIT_EXP).reshape(shape)


def gradient(step: int, tensor: int) -> np.ndarray:
    name, shape, _, _ = TENSORS[tensor]
    return scaled(grad_ints(step, tensor, numel(shape)), GRAD_EXP).reshape(shape)


def make_optimizer(params):
    groups = [
        {"params": [params[t]], "lr": TENSORS[t][2], "weight_decay": TENSORS[t][3]}
        for t in range(len(TENSORS))
    ]
    return torch.optim.AdamW(groups, betas=BETAS, eps=EPS, amsgrad=False,
                             maximize=False, foreach=False, fused=False,
                             capturable=False, differentiable=False)


def run(dtype) -> dict[str, torch.Tensor]:
    """The 100-step run in `dtype`; the checkpoints keyed `p<step>.<name>`,
    `m100.<name>`, `v100.<name>`."""
    params = [torch.nn.Parameter(torch.from_numpy(initial(t)).to(dtype).clone())
              for t in range(len(TENSORS))]
    opt = make_optimizer(params)
    out: dict[str, torch.Tensor] = {}
    for step in range(1, STEPS + 1):
        for t, p in enumerate(params):
            p.grad = torch.from_numpy(gradient(step, t)).to(dtype).clone()
        opt.step()
        if step in CHECKPOINTS:
            for t, p in enumerate(params):
                out[f"p{step}.{TENSORS[t][0]}"] = p.detach().clone().contiguous()
    for t, p in enumerate(params):
        state = opt.state[p]
        if int(state["step"]) != STEPS:
            raise SystemExit(f"{TENSORS[t][0]}: optimizer step {state['step']} != {STEPS}")
        out[f"m100.{TENSORS[t][0]}"] = state["exp_avg"].detach().clone().contiguous()
        out[f"v100.{TENSORS[t][0]}"] = state["exp_avg_sq"].detach().clone().contiguous()
    for name, t in out.items():
        if not bool(torch.isfinite(t).all()):
            raise SystemExit(f"{name} ({dtype}) is not finite")
    return out


def peak_rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    a, b = x.double(), ref.double()
    peak = float(b.abs().max())
    if not peak > 0.0:
        raise SystemExit("reference tensor is all zero: peak-relative error undefined")
    return float(((a - b).abs().max())) / peak


def round_up_2sig(x: float) -> float:
    scale = 10.0 ** (math.floor(math.log10(x)) - 1)
    return math.ceil(x / scale) * scale


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probes() -> dict[str, torch.Tensor]:
    keys, ints, grads = [], [], []
    init_keys, inits = [], []
    for t, (_, shape, _, _) in enumerate(TENSORS):
        n = numel(shape)
        idx = sorted({0, 1, n // 2, n - 1})
        for i in idx:
            init_keys.append((t, i))
            inits.append(float(initial(t).reshape(-1)[i]))
            for step in PROBE_STEPS:
                keys.append((step, t, i))
                q = int(grad_ints(step, t, n)[i])
                if not zero_grad(step, t):
                    py = (signed24(hash32_py(SALT_GRAD, step, t, i))
                          + (signed24(hash32_py(SALT_OFFSET, 0, t, i)) >> 1))
                    if py != q:
                        raise SystemExit(f"numpy and python hashes disagree at {(step, t, i)}")
                ints.append(q)
                grads.append(float(gradient(step, t).reshape(-1)[i]))
        for i in idx:
            if signed24(hash32_py(SALT_INIT, 0, t, i)) != int(init_ints(t, n)[i]):
                raise SystemExit(f"numpy and python init hashes disagree at {(t, i)}")
    return {
        "probe.key": torch.tensor(keys, dtype=torch.int64),
        "probe.gi": torch.tensor(ints, dtype=torch.int64),
        "probe.grad": torch.tensor(grads, dtype=torch.float32),
        "probe.init_key": torch.tensor(init_keys, dtype=torch.int64),
        "probe.init": torch.tensor(inits, dtype=torch.float32),
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    fp32 = run(torch.float32)
    fp64 = run(torch.float64)

    gaps = {name: peak_rel(fp32[name], fp64[name]) for name in fp32}
    gap = max(gaps.values())
    if not 0.0 < gap < 1e-3:
        raise SystemExit(f"fp32-vs-fp64 gap {gap:.3e} is outside (0, 1e-3): "
                         "the run is degenerate or torch is broken")
    bound = round_up_2sig(4.0 * gap)
    bound32 = float(np.float32(bound))

    tensors = dict(fp32)
    tensors.update(probes())
    tensors["bound"] = torch.tensor([bound32], dtype=torch.float32)
    tensors["gap"] = torch.tensor([gap], dtype=torch.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"generator": GENERATOR, "torch": torch.__version__, "kind": "adamw_100"}
    main_path, fp64_path = OUT / "adamw_100.safetensors", OUT / "adamw_100_fp64.safetensors"
    for path, data, md in ((main_path, tensors, meta),
                           (fp64_path, fp64, {**meta, "kind": "adamw_100_fp64"})):
        tmp = Path(str(path) + ".tmp")
        save_file(data, str(tmp), metadata=md)
        os.replace(tmp, path)
    back = load_file(str(main_path))
    for name, t in tensors.items():
        if back[name].dtype != t.dtype or not torch.equal(back[name], t):
            raise SystemExit(f"{name} did not round-trip through safetensors")

    manifest = {
        "format": "ojas-adamw-100-v1",
        "generator": GENERATOR,
        "recipe": "Lappi campaign/lora-2b-preregistered.json (betas, eps, weight decay, lr 2e-4 / 1e-4)",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "safetensors": SAFETENSORS_VERSION,
        "python": platform.python_version(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "optimizer": {
            "class": "torch.optim.AdamW", "betas": list(BETAS), "eps": EPS,
            "amsgrad": False, "maximize": False, "foreach": False, "fused": False,
            "capturable": False, "differentiable": False,
            "groups": [{"name": n, "shape": list(s), "lr": lr, "weight_decay": wd}
                       for n, s, lr, wd in TENSORS],
        },
        "steps": STEPS,
        "checkpoints": {"params": list(CHECKPOINTS), "moments": [STEPS]},
        "zero_grad": [{"tensor": TENSORS[t][0], "first_step": lo, "last_step": hi}
                      for t, lo, hi in ZERO_GRAD],
        "inputs": {
            "hash": "fmix32(salt*0x27D4EB2F + step*0x9E3779B1 + tensor*0x165667B1 "
                    "+ index*0xC2B2AE35 + 0x85EBCA6B) mod 2^32",
            "salts": {"init": SALT_INIT, "grad_noise": SALT_GRAD, "grad_offset": SALT_OFFSET},
            "init_scale_exp2": INIT_EXP, "grad_scale_exp2": GRAD_EXP,
        },
        "error": "max|x - ref| / max|ref| per tensor",
        "gap_fp32_vs_fp64": gap,
        "gap_by_tensor": gaps,
        "bound": {"value": bound32, "rule": "4 x gap, rounded up to two significant digits"},
        "probes": {
            "key_step_tensor_index": tensors["probe.key"].tolist(),
            "grad_int": tensors["probe.gi"].tolist(),
            "grad_f32": tensors["probe.grad"].tolist(),
            "init_key_tensor_index": tensors["probe.init_key"].tolist(),
            "init_f32": tensors["probe.init"].tolist(),
        },
        "files": {main_path.name: sha256(main_path), fp64_path.name: sha256(fp64_path)},
    }
    tmp = OUT / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, OUT / "manifest.json")
    print(json.dumps({"out": str(OUT), "gap": gap, "bound": bound32, "torch": torch.__version__}))


main()
