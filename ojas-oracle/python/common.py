"""Shared helpers for the ojas-oracle torch generators.

Reference oracle only (house policy): nothing here ships. Everything runs on
CPU, single-threaded, with deterministic algorithms, so two runs write the
same bytes.

nanolab is imported from its checkout, read-only: bytecode writing is
disabled so the checkout is never touched.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write .pyc into the nanolab checkout

import json
import math
import os
import platform
import struct
from pathlib import Path

import numpy as np
import torch

NANOLAB_ROOT = Path(os.environ.get("OJAS_NANOLAB_ROOT",
                                   "/Users/bharath/Code/research/MLSystemsLab"))
sys.path.insert(0, str(NANOLAB_ROOT))

from nanolab.config import Config, _swiglu_hidden  # noqa: E402
from nanolab.model import build_model  # noqa: E402
from nanolab.optim import _split_params  # noqa: E402
from nanolab.utils import set_seed  # noqa: E402

ORACLE_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ORACLE_ROOT / "fixtures" / "tiny"
GENERATED_DIR = ORACLE_ROOT / "generated"

SPEC_FORMAT = "ojas-spec-v1"
ORACLE_FORMAT = "ojas-oracle-fixture-v1"

# The tiny fixture model. SwiGLU's hidden width is not a nanolab Config field:
# `_swiglu_hidden(64)` is 192, so that is the hidden width (not 128).
TINY = dict(n_layer=2, d_model=64, n_head=4, n_kv_head=4, head_dim=16,
            vocab_size=256, block_size=32)
# nanolab's own defaults (12 x 768, 12 x 64 heads, V=50304, T=1024).
BASE_124M: dict = {}

# Architecture flags ojas implements. The exporter refuses anything else, so a
# spec can never describe a model ojas would run differently.
REQUIRED_FLAGS = dict(
    mixer="attention", layer_mixers="", ffn="swiglu", norm="rmsnorm",
    pos="rope", qk_norm=True, gated_attention=True, value_residual=True,
    tie_embeddings=True, zero_init_proj=True, mup=False, n_loops=1,
    dropout=0.0, sp_inv_d_attn_scale=False, mup_sqrt_attn_scale=False,
)


def setup_determinism() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def make_config(tiny: bool, seed: int, **overrides) -> Config:
    """fp32, eager, CPU: the oracle precision (framework-design.md §10)."""
    base = dict(TINY if tiny else BASE_124M)
    base.update(seed=seed, dtype="fp32", compile=False, device="cpu")
    base.update(overrides)
    cfg = Config(**base)
    for key, want in REQUIRED_FLAGS.items():
        got = getattr(cfg, key)
        if got != want:
            raise SystemExit(f"config {key}={got!r}: ojas implements only {want!r}")
    return cfg


def spec_from_config(cfg: Config) -> dict:
    """The `ojas.spec` JSON (ojas-spec-v1). Field names follow ojas-infer's
    GptConfig; see ojas-oracle/README.md for the schema."""
    return {
        "format": SPEC_FORMAT,
        "arch": "nanolab-gpt",
        "vocab": cfg.vocab_size,
        "n_embd": cfg.d_model,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_kv_head": cfg.n_kv_head,
        "head_dim": cfg.head_dim,
        "hidden": _swiglu_hidden(cfg.d_model),
        "max_seq": cfg.block_size,
        "rope_base": float(cfg.rope_base),
        "rms_eps": 1e-6,  # nanolab RMSNorm default (mixers.py:50); not a Config field
        "tie_embeddings": cfg.tie_embeddings,
        "qk_norm": cfg.qk_norm,
        "gated_attention": cfg.gated_attention,
        "value_residual": cfg.value_residual,
    }


def expected_params(spec: dict) -> list[tuple[str, list[int], str, str]]:
    """framework-design.md §2 table: (name, shape, init, group), in a fixed order.

    `init` is one of normal(0.02), zeros, ones. `group` is muon or adamw.
    """
    d, h, hd, f, v = (spec["n_embd"], spec["n_head"], spec["head_dim"],
                      spec["hidden"], spec["vocab"])
    kv = spec["n_kv_head"]
    rows = [("tok_emb.weight", [v, d], "normal(0.02)", "adamw")]
    for i in range(spec["n_layer"]):
        p = f"blocks.{i}."
        rows += [
            (p + "norm1.weight", [d], "ones", "adamw"),
            (p + "mixer.q_proj.weight", [h * hd, d], "normal(0.02)", "muon"),
            (p + "mixer.k_proj.weight", [kv * hd, d], "normal(0.02)", "muon"),
            (p + "mixer.v_proj.weight", [kv * hd, d], "normal(0.02)", "muon"),
            (p + "mixer.o_proj.weight", [d, h * hd], "zeros", "muon"),
            (p + "mixer.q_norm.weight", [hd], "ones", "adamw"),
            (p + "mixer.k_norm.weight", [hd], "ones", "adamw"),
            (p + "mixer.gate.weight", [h, d], "normal(0.02)", "muon"),
            (p + "mixer.gate.bias", [h], "zeros", "adamw"),
            (p + "mixer.vr_lambda", [1], "zeros", "adamw"),
            (p + "norm2.weight", [d], "ones", "adamw"),
            (p + "ffn.gate.weight", [f, d], "normal(0.02)", "muon"),
            (p + "ffn.up.weight", [f, d], "normal(0.02)", "muon"),
            (p + "ffn.down.weight", [d, f], "zeros", "muon"),
        ]
    rows.append(("norm_f.weight", [d], "ones", "adamw"))
    return rows


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == torch.float32:
        return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))
    return torch.equal(a, b)


def export_state(model) -> dict[str, torch.Tensor]:
    """nanolab state_dict under the §2 naming rules.

    - `_orig_mod.` (torch.compile) is stripped.
    - `lm_head.weight` must be bit-equal to `tok_emb.weight` and is dropped:
      only the embedding is stored.
    """
    out: dict[str, torch.Tensor] = {}
    for name, t in model.state_dict().items():
        name = name.removeprefix("_orig_mod.")
        if name in out:
            raise SystemExit(f"duplicate tensor {name} after stripping _orig_mod.")
        # clone: state_dict() aliases the live parameters, and a snapshot taken
        # mid-training must not move with them.
        out[name] = t.detach().to("cpu").contiguous().clone()
    head = out.pop("lm_head.weight", None)
    if head is not None and not bits_equal(head, out["tok_emb.weight"]):
        raise SystemExit("lm_head.weight is not bit-equal to tok_emb.weight; refusing to tie")
    return out


def load_state(model, tensors: dict[str, torch.Tensor]) -> None:
    """Inverse of export_state: re-tie the head and load strictly."""
    state = dict(tensors)
    state["lm_head.weight"] = state["tok_emb.weight"]
    model.load_state_dict(state, strict=True)
    if model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr():
        raise SystemExit("lm_head is no longer tied to tok_emb after load")


# ---------------------------------------------------------------------------
# safetensors, written by hand so the bytes are deterministic: tensors sorted
# by (dtype width descending, name), header keys sorted, no whitespace, header
# padded with spaces to an 8-byte boundary. Every file is read back through
# the reference `safetensors` package before it is accepted.
# ---------------------------------------------------------------------------
_ST_DTYPES = {torch.float32: ("F32", 4), torch.int64: ("I64", 8)}


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor],
                      metadata: dict[str, str]) -> None:
    items = []
    for name, t in tensors.items():
        if t.dtype not in _ST_DTYPES:
            raise SystemExit(f"{name}: dtype {t.dtype} is not F32 or I64")
        tag, width = _ST_DTYPES[t.dtype]
        if t.dtype == torch.float32 and not bool(torch.isfinite(t).all()):
            raise SystemExit(f"{name}: non-finite values")
        items.append((-width, name, tag, t.contiguous()))
    items.sort(key=lambda it: (it[0], it[1]))
    header: dict = {"__metadata__": dict(metadata)}
    chunks = []
    offset = 0
    for _, name, tag, t in items:
        raw = t.numpy().astype(t.numpy().dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": tag, "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    head = canonical_json(header).encode("ascii")
    head += b" " * ((8 - len(head) % 8) % 8)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(head)))
        fh.write(head)
        for raw in chunks:
            fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _verify_with_reference_reader(path, tensors, metadata)


def _verify_with_reference_reader(path, tensors, metadata) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file

    loaded = load_file(str(path))
    if set(loaded) != set(tensors):
        raise SystemExit(f"{path}: reference reader sees {sorted(loaded)}")
    for name, t in tensors.items():
        if not bits_equal(loaded[name], t.contiguous()):
            raise SystemExit(f"{path}: {name} did not round-trip")
    with safe_open(str(path), framework="pt") as fh:
        if fh.metadata() != metadata:
            raise SystemExit(f"{path}: metadata did not round-trip")


def read_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as fh:
        meta = dict(fh.metadata() or {})
    return load_file(str(path)), meta


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def nanolab_head() -> str:
    """Commit the nanolab checkout's HEAD names, read from .git (no git call)."""
    git = NANOLAB_ROOT / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = git / ref
        if loose.exists():
            return loose.read_text().strip()
        packed = git / "packed-refs"
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError as e:
        raise SystemExit(f"cannot read nanolab HEAD: {e}") from e
    raise SystemExit("nanolab HEAD ref not found")


def provenance(generator: str, cfg: Config | None = None, **extra) -> dict:
    out = {
        "format": ORACLE_FORMAT,
        "generator": generator,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "nanolab_head": nanolab_head(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    if cfg is not None:
        out["config"] = {k: v for k, v in cfg.to_dict().items()
                         if isinstance(v, (int, float, str, bool))}
    out.update(extra)
    return out


def build_seeded_model(cfg: Config):
    """nanolab's own path: set_seed then build_model (train.py:136,177)."""
    set_seed(cfg.seed)
    return build_model(cfg)


def param_groups(model) -> dict[str, str]:
    """nanolab's Muon/AdamW split (optim.py `_split_params`), by name."""
    matrix, embed_head, scalar = _split_params(model)
    ids = {id(p): "muon" for p in matrix}
    ids.update({id(p): "adamw" for p in embed_head + scalar})
    return {n: ids[id(p)] for n, p in model.named_parameters()}


def finite_float(x: float) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise SystemExit(f"non-finite value {x}")
    return x
