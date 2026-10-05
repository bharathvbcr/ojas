"""Float64 torch goldens for the CUDA Qwen3.5 kernels' host references.

Role: a fixture generator (the house language policy's permitted Python role 1). The Rust
references in ``tests/reference/`` are compared against what this writes; nothing shipped
imports it. Approved by the lead for lane L-cuda-oracle on 2026-10-01 with these conditions:
float64 throughout, fixed seeds, CPU only, ``torch.use_deterministic_algorithms(True)``, and a
manifest recording versions, the command line, shapes, seeds and every output's sha256 (the
Rust tests check those hashes, so a regenerated file without a manifest update fails).

Each golden is torch's own float64 forward and autograd backward of the operator as
transformers 5.12.1 writes it in ``transformers/models/qwen3_5/modeling_qwen3_5.py``. Where
transformers casts to float32 inside the function (the GDN recurrence, both RMSNorms), the
function body is transcribed here at float64 and the file:line it transcribes is named; the
GDN transcription is additionally checked against transformers' own float32 function on the
same inputs, so a transcription error cannot hide behind the dtype change.

Rule 9 (Lappi repo): every GDN file name, the gates' included, carries ``published``. The recurrence below is the
published rule (the correction reads the state after this token's decay,
``modeling_qwen3_5.py:356-359``); the generator also computes the repo rule and records that it
diverges, so the distinction is measured rather than asserted.

Run (from anywhere; paths are absolute):

    /Users/bharath/.venvs/ml/bin/python \
        /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/tests/fixtures/gen_goldens.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

F64 = torch.float64
HERE = Path(__file__).resolve().parent
OUT = HERE / "goldens"

DEFAULT_LAPPI_PYTHON = "/Users/bharath/Code/research/Lappi-decision/python"
DEFAULT_LAPPI_TOOLS = "/Users/bharath/Code/research/Lappi-decision/tools"
DEFAULT_QWEN35_CONFIG = "/Users/bharath/Code/research/tessl/tests/fixtures/qwen35/config_qwen35_2b_base.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Writer:
    """Writes .npy files into OUT and remembers what it wrote."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.cases: dict[str, dict] = {}

    def save(self, name: str, x: torch.Tensor | np.ndarray) -> None:
        arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        if arr.dtype not in (np.float64, np.float32, np.int64):
            raise TypeError(f"{name}: refusing dtype {arr.dtype}; goldens are <f8, <f4 or <i8")
        path = OUT / f"{name}.npy"
        if path.name in self.files:
            raise ValueError(f"{name} written twice")
        np.save(path, np.ascontiguousarray(arr), allow_pickle=False)
        self.files[path.name] = sha256_file(path)

    def save_text(self, name: str, text: str) -> None:
        path = OUT / name
        path.write_text(text, encoding="utf-8")
        self.files[path.name] = sha256_file(path)

    def case(self, name: str, **meta: object) -> None:
        if name in self.cases:
            raise ValueError(f"case {name} recorded twice")
        self.cases[name] = meta


def uniform(gen: torch.Generator, *shape: int) -> torch.Tensor:
    """Uniform in [-1, 1), float64."""
    return torch.rand(*shape, generator=gen, dtype=F64) * 2.0 - 1.0


def leaf(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


# --------------------------------------------------------------------------- K2: GDN ---


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``modeling_qwen3_5.py:240-243``."""
    inv_norm = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv_norm


def gdn_recurrent_f64(q, k, v, g, beta, s0, *, published: bool):
    """``torch_recurrent_gated_delta_rule`` (``modeling_qwen3_5.py:327-368``) with
    ``use_qk_l2norm_in_kernel=True``, transcribed at float64 (the original casts to float32
    at :335-337). ``published=False`` is the repo rule: the correction reads the undecayed
    state. Inputs ``[B, T, H, D]``, gates ``[B, T, H]``, state ``[B, H, Dk, Dv]``."""
    q = l2norm(q)
    k = l2norm(k)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous() for x in (q, k, v, beta, g)]
    scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    state = s0
    outs = []
    for i in range(q.shape[2]):
        q_t, k_t, v_t = q[:, :, i], k[:, :, i], v[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        if published:
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        else:
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            state = state * g_t
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    o = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
    return o, state


def gdn_published_train(w: Writer) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    b, h, dk, dv = 1, 2, 128, 16
    for t in (1, 63, 64, 65, 130):
        seed = 5000 + t
        gen = torch.Generator().manual_seed(seed)
        q, k = uniform(gen, b, t, h, dk), uniform(gen, b, t, h, dk)
        v = uniform(gen, b, t, h, dv)
        g = -0.75 * (uniform(gen, b, t, h) + 1.0) - 1e-3  # log decays in [-1.5, 0)
        beta = 0.5 + 0.45 * uniform(gen, b, t, h)
        s0 = uniform(gen, b, h, dk, dv)
        d_o, dfin = uniform(gen, b, t, h, dv), uniform(gen, b, h, dk, dv)
        xs = [leaf(x) for x in (q, k, v, g, beta, s0)]
        o, fin = gdn_recurrent_f64(*xs, published=True)
        ((o * d_o).sum() + (fin * dfin).sum()).backward()

        # The transcription against transformers' own float32 function.
        with torch.no_grad():
            o32, fin32 = torch_recurrent_gated_delta_rule(
                q.float(), k.float(), v.float(), g.float(), beta.float(), s0.float(),
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            rel_o = ((o32.double() - o).abs().max() / o.abs().max()).item()
            rel_s = ((fin32.double() - fin).abs().max() / fin.abs().max()).item()
            if max(rel_o, rel_s) > 1e-4:
                raise AssertionError(f"T={t}: f64 transcription vs transformers f32: {rel_o:.3e} / {rel_s:.3e}")
            o_repo, _ = gdn_recurrent_f64(q, k, v, g, beta, s0, published=False)
            repo_gap = ((o_repo - o).abs().max() / o.abs().max()).item()
            if t > 1 and repo_gap <= 1e-3:
                raise AssertionError(f"T={t}: the repo rule came within {repo_gap:.3e} of published")

        name = f"gdn_published_train_T{t}"
        for suffix, x in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("s0", s0),
                          ("d_o", d_o), ("dfin", dfin), ("o", o), ("fin", fin)]:
            w.save(f"{name}_{suffix}", x)
        for suffix, x in zip(("dq", "dk", "dv", "dg", "dbeta", "ds0"), xs):
            w.save(f"{name}_{suffix}", x.grad)
        w.case(name, rule="published", B=b, T=t, H=h, Dk=dk, Dv=dv, seed=seed,
               transformers_f32_rel_err_o=rel_o, transformers_f32_rel_err_final=rel_s,
               repo_rule_rel_divergence=repo_gap,
               source="modeling_qwen3_5.py:327-368 torch_recurrent_gated_delta_rule, l2norm in kernel, float64")


# --------------------------------------------------------------------------- K3: gates ---


def gates_published(w: Writer) -> None:
    """``modeling_qwen3_5.py:516,518``: ``beta = b.sigmoid()``,
    ``g = -A_log.exp() * F.softplus(a + dt_bias)`` (torch softplus: linear above 20).

    The gates are part of the GDN operator whose state update is the published rule, so
    under rule 9 their files carry ``published`` too (lead's ruling, 2026-10-01)."""
    rows, heads, seed = 40, 3, 6001
    gen = torch.Generator().manual_seed(seed)
    a = 12.0 * uniform(gen, rows, heads)
    a[::3, 0] = 26.0    # a + dt_bias > 20: softplus' linear branch
    a[1::5, -1] = -24.0  # deep in the series
    a[-1, 0] = 120.0     # exp(a) overflows f32; must not be formed
    b_ = 6.0 * uniform(gen, rows, heads)
    a_log = 0.8 * uniform(gen, heads)
    dt_bias = -3.0 + 2.0 * uniform(gen, heads)  # [-5, -1), tessl run_gates
    dg, dbeta = uniform(gen, rows, heads), uniform(gen, rows, heads)
    xa, xb, xal, xdt = (leaf(x) for x in (a, b_, a_log, dt_bias))
    g = -xal.exp() * F.softplus(xa + xdt)
    beta = xb.sigmoid()
    ((g * dg).sum() + (beta * dbeta).sum()).backward()
    for n, x in [("a", a), ("b", b_), ("a_log", a_log), ("dt_bias", dt_bias), ("dg", dg), ("dbeta", dbeta),
                 ("g", g), ("beta", beta), ("da", xa.grad), ("db", xb.grad), ("da_log", xal.grad),
                 ("ddt_bias", xdt.grad)]:
        w.save(f"gates_published_{n}", x)
    w.case("gates_published", rule="published", rows=rows, heads=heads, seed=seed,
           source="modeling_qwen3_5.py:516,518")


# -------------------------------------------------------------------- K4: conv1d + SiLU ---


def conv1d_silu(w: Writer) -> None:
    """``modeling_qwen3_5.py:390-397,497``: depthwise causal ``nn.Conv1d(groups=C,
    padding=K-1, bias=False)`` truncated to T, then SiLU; zero initial state. ``x`` is
    ``[B, T, C]``, ``w`` is the weight squeezed to ``[C, K]``."""
    for i, (b, t, c, k) in enumerate([(2, 9, 6, 4), (3, 2, 5, 4), (1, 70, 8, 4)]):
        seed = 6100 + i
        gen = torch.Generator().manual_seed(seed)
        x, wt, dy = 2.0 * uniform(gen, b, t, c), uniform(gen, c, k), uniform(gen, b, t, c)
        xl, wl = leaf(x), leaf(wt)
        pre = F.conv1d(xl.transpose(1, 2), wl.unsqueeze(1), bias=None, padding=k - 1, groups=c)[:, :, :t]
        y = F.silu(pre).transpose(1, 2).contiguous()
        (y * dy).sum().backward()
        name = f"conv1d_silu_c{i}"
        for n, v in [("x", x), ("w", wt), ("dy", dy), ("y", y), ("dx", xl.grad), ("dw", wl.grad)]:
            w.save(f"{name}_{n}", v)
        w.case(name, B=b, T=t, C=c, K=k, seed=seed, source="modeling_qwen3_5.py:390-397,497")


# ----------------------------------------------------------------------- K7: the norms ---

#: transformers' norm modules upcast to float32, so the float64 transcriptions here are held to
#: the module's own float32 output within this relative bound (f32 unit roundoff 6e-8 times a
#: 2048-term mean, with margin). A transcription error is O(1).
F32_MODULE_BOUND = 1e-4


def f32_check(out32: torch.Tensor, y64: torch.Tensor) -> float:
    ref = y64.detach()
    rel = ((out32.double() - ref).abs().max() / ref.abs().max()).item()
    if rel > F32_MODULE_BOUND:
        raise AssertionError(f"float64 transcription vs transformers' float32 module: {rel:.3e}")
    return rel


def transformers_rms_norm_f32(x: torch.Tensor, wt: torch.Tensor, eps: float) -> torch.Tensor:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    m = Qwen3_5RMSNorm(x.shape[-1], eps=eps)
    with torch.no_grad():
        m.weight.data = wt.float()
        return m(x.float())


def transformers_gated_rms_norm_f32(x: torch.Tensor, z: torch.Tensor, wt: torch.Tensor, eps: float) -> torch.Tensor:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

    m = Qwen3_5RMSNormGated(x.shape[-1], eps=eps)
    with torch.no_grad():
        m.weight.data = wt.float()
        return m(x.float(), gate=z.float())


def norms(w: Writer) -> None:
    eps = 1e-6
    for i, (rows, d) in enumerate([(5, 64), (3, 2048)]):
        seed = 6200 + i
        gen = torch.Generator().manual_seed(seed)
        x, wt, dy = 3.0 * uniform(gen, rows, d), 0.5 * uniform(gen, d), uniform(gen, rows, d)
        xl, wl = leaf(x), leaf(wt)
        # Qwen3_5RMSNorm, modeling_qwen3_5.py:736-751, at float64 (the original upcasts to f32).
        y = xl * torch.rsqrt(xl.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + wl)
        (y * dy).sum().backward()
        rel32 = f32_check(transformers_rms_norm_f32(x, wt, eps), y)
        name = f"rms_norm_c{i}"
        for n, v in [("x", x), ("w", wt), ("dy", dy), ("y", y), ("dx", xl.grad), ("dw", wl.grad)]:
            w.save(f"{name}_{n}", v)
        w.case(name, rows=rows, d=d, eps=eps, seed=seed, transformers_f32_module_rel_err=rel32,
               source="modeling_qwen3_5.py:736-751 (1 + w)")
    for i, (units, d) in enumerate([(12, 128), (4, 16)]):
        seed = 6250 + i
        gen = torch.Generator().manual_seed(seed)
        x, z = uniform(gen, units, d), 3.0 * uniform(gen, units, d)
        wt, dy = uniform(gen, d), uniform(gen, units, d)
        xl, zl, wl = leaf(x), leaf(z), leaf(wt)
        # Qwen3_5RMSNormGated, modeling_qwen3_5.py:187-202, at float64.
        hs = xl * torch.rsqrt(xl.pow(2).mean(-1, keepdim=True) + eps)
        y = (wl * hs) * F.silu(zl)
        (y * dy).sum().backward()
        rel32 = f32_check(transformers_gated_rms_norm_f32(x, z, wt, eps), y)
        name = f"gated_rms_norm_c{i}"
        for n, v in [("x", x), ("z", z), ("w", wt), ("dy", dy), ("y", y), ("dx", xl.grad), ("dz", zl.grad),
                     ("dw", wl.grad)]:
            w.save(f"{name}_{n}", v)
        w.case(name, units=units, d=d, eps=eps, seed=seed, transformers_f32_module_rel_err=rel32,
               source="modeling_qwen3_5.py:187-202")


# ------------------------------------------------- K6: q/k norm + partial RoPE, out gate ---


def qk_norm_rope(w: Writer, config_path: str) -> dict:
    """Per head row: ``Qwen3_5RMSNorm`` ``(1 + w)`` (:736-751) then transformers'
    ``apply_rotary_pos_emb`` (:570-605, called here unmodified at float64) over the first
    ``rotary_dim`` dims. The angle is formed in float32 exactly as
    ``Qwen3_5TextRotaryEmbedding`` forms it (``inv_freq`` from the module's own buffer,
    ``inv_freq * position`` in f32, :149-167) and only its cos/sin are taken in float64
    (tessl's convention, ``tests/common/qwen35.rs:187-197``). The angles are saved too."""
    from transformers.models.qwen3_5 import configuration_qwen3_5 as cfgmod
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb

    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    cfg = cfgmod.Qwen3_5TextConfig(**raw["text_config"])
    rot_mod = Qwen3_5TextRotaryEmbedding(cfg)
    rot = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    theta = float(cfg.rope_parameters["rope_theta"])
    eps = 1e-6
    d = cfg.head_dim
    positions = torch.tensor([0, 1, 2, 7, 63, 64, 300, 4097, 20000, 20001], dtype=torch.int64)

    # MRoPE collapse for text-only input: the module's own f32 cos/sin at identical
    # t/h/w position streams vs cos/sin of the plain f32 angle.
    with torch.no_grad():
        dummy = torch.zeros(1, positions.numel(), d, dtype=torch.float32)
        cos_mod, sin_mod = rot_mod(dummy, positions[None, :])
        angle32 = rot_mod.inv_freq[None, :].float() * positions[:, None].float()  # [R, rot/2], f32
        emb32 = torch.cat((angle32, angle32), dim=-1)
        collapse = max((cos_mod[0] - emb32.cos()).abs().max().item(), (sin_mod[0] - emb32.sin()).abs().max().item())

    seed = 6300
    gen = torch.Generator().manual_seed(seed)
    rows = positions.numel()
    x, wt, dy = 3.0 * uniform(gen, rows, d), uniform(gen, d), uniform(gen, rows, d)
    xl, wl = leaf(x), leaf(wt)
    n = xl * torch.rsqrt(xl.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + wl)
    norm_rel32 = f32_check(transformers_rms_norm_f32(x, wt, eps), n)
    emb64 = emb32.double()
    cos, sin = emb64.cos(), emb64.sin()
    # apply_rotary_pos_emb takes (q, k); pass the rows as q and a throwaway k.
    y, _ = apply_rotary_pos_emb(n[:, None, :], n[:, None, :].detach(), cos, sin, unsqueeze_dim=1)
    y = y[:, 0, :]
    (y * dy).sum().backward()
    for nm, v in [("x", x), ("w", wt), ("dy", dy), ("positions", positions), ("angle_f32", angle32.double()),
                  ("y", y), ("dx", xl.grad), ("dw", wl.grad)]:
        w.save(f"qk_norm_rope_{nm}", v)
    w.case("qk_norm_rope", rows=rows, head_dim=d, rotary_dim=rot, theta=theta, eps=eps, seed=seed,
           mrope_section=list(cfg.rope_parameters.get("mrope_section", [])),
           mrope_text_only_collapse_max_abs=collapse, config=config_path, config_sha256=sha256_file(Path(config_path)),
           norm_transformers_f32_module_rel_err=norm_rel32,
           source="modeling_qwen3_5.py:142-145,149-167,570-605,736-751")

    seed = 6350
    gen = torch.Generator().manual_seed(seed)
    o, gate, dy = uniform(gen, 9, 96), 5.0 * uniform(gen, 9, 96), uniform(gen, 9, 96)
    ol, gl = leaf(o), leaf(gate)
    y = ol * torch.sigmoid(gl)  # modeling_qwen3_5.py:714
    (y * dy).sum().backward()
    for nm, v in [("o", o), ("gate", gate), ("dy", dy), ("y", y), ("do", ol.grad), ("dgate", gl.grad)]:
        w.save(f"attn_output_gate_{nm}", v)
    w.case("attn_output_gate", rows=9, width=96, seed=seed, source="modeling_qwen3_5.py:714")
    return {"mrope_text_only_collapse_max_abs": collapse}


# ------------------------------------------------------------------------- K9: embedding ---


def embedding(w: Writer) -> None:
    seed = 6400
    gen = torch.Generator().manual_seed(seed)
    vocab, hidden = 10, 8
    ids = torch.tensor([5, 0, 5, 7, 0, 5, 2, 7, 7, 1, 9, 5], dtype=torch.int64)
    table, dy = uniform(gen, vocab, hidden), uniform(gen, ids.numel(), hidden)
    tl = leaf(table)
    y = F.embedding(ids, tl)
    (y * dy).sum().backward()
    for nm, v in [("ids", ids), ("table", table), ("dy", dy), ("y", y), ("dtable", tl.grad)]:
        w.save(f"embed_{nm}", v)
    w.case("embed", vocab=vocab, hidden=hidden, rows=ids.numel(), seed=seed, source="torch.nn.functional.embedding")


# --------------------------------------------------------------- K10: CE over chosen rows ---


def ce_rows(w: Writer) -> None:
    """``F.cross_entropy`` over the supervised rows of a tied head: ``logits = h[rows] @ W^T``.
    ``dh`` is per supplied row (a duplicated row gets two gradients), ``dW`` is ``[V, H]``;
    both are gradients of ``scale * loss``."""
    cases = [("mean", 1.0, None), ("sum", 0.7, None), ("mean", 1.0, (299, 2, 60.0))]
    for i, (reduction, scale, plant) in enumerate(cases):
        seed = 6500 + i
        gen = torch.Generator().manual_seed(seed)
        t_total, hidden, vocab = 10, 16, 300
        h = uniform(gen, t_total, hidden)
        wt = 0.25 * uniform(gen, vocab, hidden)
        rows = torch.tensor([3, 0, 9, 3], dtype=torch.int64)
        targets = torch.tensor([0, 127, 128, 299], dtype=torch.int64)
        if plant is not None:
            vrow, idx, factor = plant  # a dominant logit, so the softmax saturates
            wt[vrow] = factor * h[rows[idx]]
        hg, wl = leaf(h[rows]), leaf(wt)
        logits = hg @ wl.T
        per_row = F.cross_entropy(logits, targets, reduction="none")
        loss = per_row.mean() if reduction == "mean" else per_row.sum()
        (scale * loss).backward()
        name = f"ce_rows_c{i}"
        for nm, v in [("h", h), ("w", wt), ("rows", rows), ("targets", targets),
                      ("scale", torch.tensor([scale], dtype=F64)), ("per_row", per_row), ("loss", loss.reshape(1)),
                      ("dh", hg.grad), ("dw", wl.grad)]:
            w.save(f"{name}_{nm}", v)
        w.case(name, reduction=reduction, scale=scale, t_total=t_total, hidden=hidden, vocab=vocab, seed=seed,
               planted=plant, source="torch.nn.functional.cross_entropy")


# --------------------------------------------------------------------- K11: AdamW (F) ---

ADAMW_NAMES = [
    ("model.embed_tokens.weight", (12, 4)),
    ("model.layers.0.input_layernorm.weight", (4,)),
    ("model.layers.0.linear_attn.A_log", (2,)),
    ("model.layers.3.self_attn.q_proj.weight", (6, 4)),
    ("model.layers.7.mlp.down_proj.weight", (4, 5)),
    ("model.layers.8.mlp.down_proj.weight", (4, 5)),
    ("model.layers.10.linear_attn.dt_bias", (2,)),
    ("model.layers.23.post_attention_layernorm.weight", (4,)),
    ("model.norm.weight", (4,)),
]


#: F's flags, as `campaign/f-v4-preregistered.json` (`recipe.flags`) records them; checked
#: against that file's text at generation time, so they cannot drift from F's record.
F_FLAGS = {"--optimizer": "master", "--lr": "1e-5", "--lower-layers-n": "8", "--lower-layers-lr-scale": "0.1"}


def f_builder_groups(lappi_python: str, lappi_tools: str, names_shapes, base_lr: float, steps: int):
    """F's optimizer, built by F's own construction, nothing retyped
    (GAP-OJAS-K11-GOLDEN-RETYPES-F-HYPER-2026-10-01): ``layerwise_param_groups`` over the tower
    names (``lower_layers_n`` from F's flags, ``lower_lr_scale`` = ``real_ft_run.RSI_LOWER_LR_SCALE``)
    -> ``build_optimizer(spec=real_ft_run.optimizer_spec("bf16", "master"), lr, total_steps,
    beta2=DEFAULT_BETA2, fused=False)`` -> ``MasterWeightAdamW`` -> ``apply_lr``, as
    ``QwenDecisionStep`` builds and drives it (``backbone.py:1027-1045``, ``:1190``). The tower is
    bf16 in F, so the builder is run on bf16 shadow parameters with the golden's names and
    shapes; eps, weight decay, betas and every other group key are then read off the built
    optimizer's own groups. Returns ``(groups_in, built, recipe)``."""
    for p in (lappi_python, lappi_tools):
        if p not in sys.path:
            sys.path.insert(0, p)
    from qd_train import optim as lappi_optim
    import real_ft_run

    recipe_path = Path(lappi_tools).parent / "campaign" / "f-v4-preregistered.json"
    flags_text = json.loads(recipe_path.read_text(encoding="utf-8"))["recipe"]["flags"]
    for flag, value in F_FLAGS.items():
        if f"{flag} {value}" not in flags_text:
            raise ValueError(f"F's recorded flags do not contain {flag} {value!r}: {flags_text!r}")
    if base_lr != float(F_FLAGS["--lr"]):
        raise ValueError(f"base_lr {base_lr} is not F's lr {F_FLAGS['--lr']}; the decay-sensitive "
                         "golden is L-oracle's (Lappi crates/qd-train/tests/fixtures/adamw-decay-sensitive)")
    lower_n = int(F_FLAGS["--lower-layers-n"])
    if real_ft_run.RSI_LOWER_LR_SCALE != float(F_FLAGS["--lower-layers-lr-scale"]):
        raise ValueError("real_ft_run.RSI_LOWER_LR_SCALE disagrees with F's recorded flag")
    shadows = [(n, torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))) for n, shape in names_shapes]
    groups_in = lappi_optim.layerwise_param_groups(
        shadows, lower_layers_n=lower_n, lower_lr_scale=real_ft_run.RSI_LOWER_LR_SCALE)
    built = lappi_optim.build_optimizer(
        groups_in, spec=real_ft_run.optimizer_spec("bf16", F_FLAGS["--optimizer"]), lr=base_lr,
        total_steps=steps, beta2=lappi_optim.DEFAULT_BETA2, fused=False)
    if type(built).__name__ != "MasterWeightAdamW":
        raise TypeError(f"F's builder returned {type(built).__name__}, not MasterWeightAdamW")
    lappi_optim.apply_lr(built, base_lr)
    if len(built.param_groups) != len(groups_in):
        raise ValueError("the builder dropped or added a group")
    shadow_index = {id(p): i for i, (_, p) in enumerate(shadows)}
    recipe = {
        "builder": "qd_train.optim.layerwise_param_groups -> build_optimizer(spec=real_ft_run."
                   "optimizer_spec('bf16', 'master'), beta2=DEFAULT_BETA2, fused=False) -> "
                   "MasterWeightAdamW -> apply_lr, on bf16 shadow parameters",
        "f_flags": F_FLAGS, "f_flags_source": str(recipe_path), "f_flags_text": flags_text,
        "lappi_optim": str(Path(lappi_python) / "qd_train" / "optim.py"),
        "lappi_optim_sha256": sha256_file(Path(lappi_python) / "qd_train" / "optim.py"),
        "real_ft_run": str(Path(lappi_tools) / "real_ft_run.py"),
        "real_ft_run_sha256": sha256_file(Path(lappi_tools) / "real_ft_run.py"),
        "built_class": type(built).__name__, "inner_class": type(built._inner).__name__,
        "groups": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k != "params"}
                   for g in built.param_groups],
    }
    return groups_in, built, shadow_index, recipe


def adamw(w: Writer, lappi_python: str, lappi_tools: str) -> dict:
    """F's optimizer through F's own builder ([`f_builder_groups`]), stepped in float64: the
    builder's groups, every key as the builder set it, over float64 parameters, in torch's
    single-tensor path (``foreach=False`` pinned; F on the CPU dispatches there too, f-optimizer-
    spec.md). Five steps, a fresh seeded gradient each step."""
    base_lr, steps, seed = float(F_FLAGS["--lr"]), 5, 6600
    gen = torch.Generator().manual_seed(seed)
    params = [torch.nn.Parameter(uniform(gen, *shape)) for _, shape in ADAMW_NAMES]
    named = list(zip([n for n, _ in ADAMW_NAMES], params))
    groups_in, built, shadow_index, recipe = f_builder_groups(
        lappi_python, lappi_tools, ADAMW_NAMES, base_lr, steps)
    from qd_train import optim as lappi_optim

    f64_groups = []
    for g_in, g_built in zip(groups_in, built.param_groups):
        if len(g_in["params"]) != len(g_built["params"]):
            raise ValueError("a built group's size differs from its input group")
        keys = {k: v for k, v in g_built.items() if k != "params"}
        keys["foreach"], keys["fused"] = False, False
        f64_groups.append({**keys, "params": [params[shadow_index[id(p)]] for p in g_in["params"]]})
    opt = torch.optim.AdamW(f64_groups)
    for g_f64, g_built in zip(opt.param_groups, built.param_groups):
        for k, v in g_built.items():
            if k not in ("params", "foreach", "fused") and g_f64[k] != v:
                raise ValueError(f"group key {k}: float64 optimizer {g_f64[k]!r}, builder {v!r}")
    scale_of = {}
    for g in opt.param_groups:
        for p in g["params"]:
            scale_of[id(p)] = (g[lappi_optim.LR_SCALE_KEY], g["lr"], g["weight_decay"], g["eps"], tuple(g["betas"]))
    hyper = {(v[3], v[4]) for v in scale_of.values()}
    if len(hyper) != 1:
        raise ValueError(f"the builder's groups disagree on eps or betas: {hyper}")
    (eps, betas), = hyper
    flat = lambda ts: torch.cat([t.detach().reshape(-1) for t in ts])  # noqa: E731
    p0 = flat(params)
    grads_all, after_all, sqn = [], [], []
    for _ in range(steps):
        grads = [uniform(gen, *p.shape) for p in params]
        for p, g in zip(params, grads):
            p.grad = g.clone()
        sqn.append(torch.stack([(g * g).sum() for g in grads]).sum())
        opt.step()
        grads_all.append(flat(grads))
        after_all.append(flat(params))
    lr_scale = torch.tensor([scale_of[id(p)][0] for p in params], dtype=F64)
    wd = torch.tensor([scale_of[id(p)][2] for p in params], dtype=F64)
    sizes = torch.tensor([p.numel() for p in params], dtype=torch.int64)
    w.save_text("adamw_f_names.txt", "".join(f"{n}\n" for n, _ in ADAMW_NAMES))
    for nm, v in [("sizes", sizes), ("lr_scale", lr_scale), ("weight_decay", wd), ("p0", p0),
                  ("grads", torch.stack(grads_all)), ("params", torch.stack(after_all)),
                  ("grad_sq_norm", torch.stack(sqn)),
                  ("hyper", torch.tensor([base_lr, betas[0], betas[1], eps], dtype=F64))]:
        w.save(f"adamw_f_{nm}", v)
    w.case("adamw_f", base_lr=base_lr, steps=steps, seed=seed,
           lower_layers_n=int(F_FLAGS["--lower-layers-n"]),
           lower_lr_scale=recipe["groups"][1][lappi_optim.LR_SCALE_KEY],
           betas=list(betas), eps=eps, weight_decay=sorted({v[2] for v in scale_of.values()}),
           hyper_layout="[base_lr, beta1, beta2, eps]",
           groups={str(n): scale_of[id(p)][0] for n, p in named},
           f_builder=recipe,
           source="F's builder's groups over float64 parameters; torch.optim.AdamW(foreach=False, "
                  "fused=False) single-tensor path, torch/optim/adam.py:416-545")
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lappi-python", default=DEFAULT_LAPPI_PYTHON)
    ap.add_argument("--lappi-tools", default=DEFAULT_LAPPI_TOOLS)
    ap.add_argument("--qwen35-config", default=DEFAULT_QWEN35_CONFIG)
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(F64)
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    for stale in OUT.iterdir():
        stale.unlink()

    w = Writer()
    gdn_published_train(w)
    gates_published(w)
    conv1d_silu(w)
    norms(w)
    extra = qk_norm_rope(w, args.qwen35_config)
    embedding(w)
    ce_rows(w)
    adamw(w, args.lappi_python, args.lappi_tools)

    import transformers

    manifest = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "dtype": "float64",
        "device": "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "rule": "published",
        "checks": extra,
        "cases": w.cases,
        "files": w.files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(w.files)} files and manifest.json to {OUT}")


if __name__ == "__main__":
    main()
