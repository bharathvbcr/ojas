"""Golden fixtures for the tiny nanolab GPT (framework item 13).

    python3.14 ojas-oracle/python/golden.py token-bin --out ojas-oracle/fixtures/tiny/tokens.bin
    python3.14 ojas-oracle/python/golden.py fixtures --dir ojas-oracle/fixtures/tiny

`token-bin` writes the synthetic token stream (headerless little-endian
uint16): a first-order Markov chain over 256 tokens, mostly a fixed successor
with rare random jumps (see `synthetic_tokens`), so a 40-step run falls more
than 1 nat below ln 256 (framework-design.md §10, CI on CPU).

`fixtures` reads `init.safetensors` (export_init.py --tiny), `tokens.bin`
and `batch_starts.json` (examples/dump_batch_starts.rs) from `--dir` and writes:

- `forward.safetensors`: x, y (micro-batch 0 of step 0), logits; loss and
  logits checksums in metadata.
- `grads_init.safetensors`: d(loss/K)/dθ at the init for that batch.
- `grads_step5.safetensors`: the same at the step-5 parameters of the f32
  trace. At the init, o_proj and ffn.down are zero, so every attention and FFN
  gradient upstream of them is exactly zero; this second set exercises them.
- `trace_ns5_f32.safetensors` / `trace_ns5_bf16.safetensors`: 40 steps of
  nanolab's training step (train.py:305-344) with its own optimizers, the
  per-step losses, grad norms and LR multipliers in metadata, and the
  parameters after step 5. `f32` replaces nanolab's NS5 with a copy that
  differs only in `G.float()` for `G.bfloat16()` (optim.py:46); `bf16` is
  stock nanolab.
- `lr_schedules.json`: nanolab's cosine and WSD multipliers for 20 steps.

Each trace is cross-checked against nanolab's real `train()` (through its
`batchers=` seam) for 5 steps: the final weights must be bit-identical and
the logged losses and grad norms equal. Any failed check aborts.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batches  # noqa: E402
import common  # noqa: E402

import nanolab.optim as nanolab_optim  # noqa: E402
from nanolab.model import build_model  # noqa: E402
from nanolab.optim import build_optimizers  # noqa: E402
from nanolab.schedules import _peak_lr, apply_lr, make_schedule  # noqa: E402

GENERATOR = "ojas-oracle/python/golden.py"
SEED = 1337
TRACE_STEPS = 40
PARAM_SNAPSHOT_STEP = 5
CROSSCHECK_STEPS = PARAM_SNAPSHOT_STEP
LR_STEPS = 20

# ---------------------------------------------------------------------------
# synthetic token bin
# ---------------------------------------------------------------------------
_M64 = (1 << 64) - 1
TOKENS = 8192
VOCAB = 256
SUCCESSORS = 4
JUMP_ODDS = 8
TOKEN_SEED = 0x0DA5_0001


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return state, z ^ (z >> 31)


def synthetic_tokens(n: int = TOKENS) -> np.ndarray:
    """t[i+1] = (5 t[i] + 17 + 64 j) mod 256, where j = 0 with probability
    7/8 and is uniform in 0..4 otherwise.

    With j always 0 this is a full-period LCG: the stream repeats every 256
    tokens, so windows 8 apart would hold identical tokens and a start that is
    off by 256 would replay undetected. The jumps make 249 of the 255 T=32
    windows distinct while keeping the stream learnable inside 40 tiny steps
    (entropy floor about 0.41 nats; the f32 trace ends about 1.19 nats below
    ln 256). Rarer jumps learn faster but repeat more windows.
    """
    out = np.empty(n, dtype="<u2")
    state, tok = TOKEN_SEED, 0
    for i in range(n):
        out[i] = tok
        state, r = _splitmix64(state)
        j = 0 if r % JUMP_ODDS else (r >> 8) % SUCCESSORS
        tok = (5 * tok + 17 + 64 * j) % VOCAB
    return out


def write_token_bin(path: Path) -> None:
    data = synthetic_tokens().tobytes()
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def trace_config():
    """Tiny model, B=2, K=2, T=32, fp32 CPU; nanolab's defaults otherwise
    (muon_ns5_adamw, matrix_lr 0.025, lr 6e-4, momentum 0.99, wd 0.1,
    clip 1.0, cosine to 0.1). Warmup 4 over a 40-step schedule, so the LR is
    near peak inside the first 5 steps."""
    return common.make_config(
        True, SEED, batch_size=2, grad_accum=2, max_steps=TRACE_STEPS,
        lr_max_steps=TRACE_STEPS, warmup_steps=4, schedule="cosine")


# ---------------------------------------------------------------------------
# NS5 precision patch (pytorch-parity-plan.md F8)
# ---------------------------------------------------------------------------
_STOCK_NS5 = nanolab_optim.zeropower_via_newtonschulz5
NS5_F32_PATCH = ("nanolab.optim.zeropower_via_newtonschulz5 is replaced, for the "
                 "duration of the run, by a copy whose only change is "
                 "`X = G.float()` in place of `X = G.bfloat16()` (optim.py:46). "
                 "Muon._orthogonalize looks the function up as a module global "
                 "at call time (optim.py:84), so the copy is what runs.")


@torch.no_grad()
def ns5_f32(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()  # nanolab: X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


@contextlib.contextmanager
def ns5(precision: str):
    """Install the NS5 variant and count its calls (proves it is the one hit)."""
    inner = {"f32": ns5_f32, "bf16": _STOCK_NS5}[precision]
    calls = [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return inner(*args, **kwargs)

    nanolab_optim.zeropower_via_newtonschulz5 = counted
    try:
        yield calls
    finally:
        nanolab_optim.zeropower_via_newtonschulz5 = _STOCK_NS5


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def fresh_model(cfg, init: dict):
    common.set_seed(cfg.seed)  # construction draws; the values are overwritten
    model = build_model(cfg)
    common.load_state(model, init)
    return model


def logits_checksums(logits: torch.Tensor) -> dict:
    flat = logits.detach().double().reshape(-1).tolist()
    return {
        "sum": math.fsum(flat),
        "sum_sq": math.fsum(v * v for v in flat),
        # Index-weighted, so a permutation of the logits changes it.
        "weighted": math.fsum(v * ((i % 1021) + 1) for i, v in enumerate(flat)),
        "count": len(flat),
    }


def f32_bits(x: float) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def grads_of(model, x, y, k: int) -> tuple[dict, float, list, list]:
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    (loss / k).backward()
    grads, no_grad, zero = {}, [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(name)
            continue
        g = p.grad.detach().clone().contiguous()
        grads[name] = g
        if not bool(g.any()):
            zero.append(name)
    model.zero_grad(set_to_none=True)
    return grads, float(loss.detach()), no_grad, zero


def run_trace(cfg, init: dict, replay, precision: str, steps: int):
    """nanolab train.py:305-344, minus logging, eval and checkpointing."""
    model = fresh_model(cfg, init)
    model.train()
    k = cfg.grad_accum
    rec = {"lr_mult": [], "micro_loss": [], "mean_loss": [], "nanolab_loss": [],
           "grad_norm": []}
    snapshot = None
    with ns5(precision) as calls:
        optimizers = build_optimizers(model, cfg)
        schedule = make_schedule(cfg)
        peak = max(_peak_lr(cfg), 1e-12)
        for step in range(steps):
            lr = schedule(step)
            apply_lr(optimizers, lr, cfg)
            micro = []
            for m in range(k):
                x, y = replay.micro(step, m)
                _, loss = model(x, y)
                micro.append(common.finite_float(loss.detach()))
                loss = loss / k
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            rec["lr_mult"].append(lr / peak)
            rec["micro_loss"].extend(micro)
            rec["mean_loss"].append(math.fsum(micro) / k)
            rec["nanolab_loss"].append(float(loss.detach()) * k)  # train.py:348
            rec["grad_norm"].append(common.finite_float(grad_norm))
            if step + 1 == PARAM_SNAPSHOT_STEP:
                snapshot = common.export_state(model)
        rec["ns5_calls"] = calls[0]
    if calls[0] == 0:
        raise SystemExit(f"NS5 {precision} was never called")
    return rec, snapshot, common.export_state(model)


def crosscheck_with_nanolab_train(cfg, init, starts, bin_path, precision, mine) -> None:
    """Run nanolab's own train() for CROSSCHECK_STEPS through its `batchers=`
    seam and require bit-identical weights and identical logged metrics."""
    from nanolab.train import train

    with tempfile.TemporaryDirectory(prefix="ojas-oracle-") as tmp:
        run_cfg = dataclasses.replace(
            cfg, max_steps=CROSSCHECK_STEPS, out_dir=tmp, run_name="crosscheck",
            log_interval=1, eval_interval=10**9, eval_iters=1,
            ckpt_interval=10**9, eval_train=False)
        replay_train = batches.StartReplay(starts, bin_path)
        replay_val = batches.StartReplay(starts, bin_path)
        # train() prints its banner and step lines; keep stdout clean.
        with ns5(precision) as calls, contextlib.redirect_stdout(sys.stderr):
            train(run_cfg, batchers=(replay_train, replay_val))
        if calls[0] == 0:
            raise SystemExit("nanolab train() never called NS5")
        run_dir = Path(tmp) / "crosscheck"
        blob = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
        model = build_model(cfg)
        model.load_state_dict(blob["model"])
        theirs = common.export_state(model)
        steps = [json.loads(line) for line in
                 (run_dir / "metrics.jsonl").read_text().splitlines()]
        steps = [r for r in steps if r.get("event") == "train"]
    if replay_train.pos != CROSSCHECK_STEPS * cfg.grad_accum:
        raise SystemExit(f"train() drew {replay_train.pos} micro-batches")
    for name, t in mine["params"].items():
        if not common.bits_equal(t, theirs[name]):
            raise SystemExit(f"{precision}: {name} differs from nanolab train() after "
                             f"{CROSSCHECK_STEPS} steps")
    if [r["loss"] for r in steps] != mine["nanolab_loss"][:CROSSCHECK_STEPS]:
        raise SystemExit(f"{precision}: logged losses differ from nanolab train()")
    if [r["grad_norm"] for r in steps] != mine["grad_norm"][:CROSSCHECK_STEPS]:
        raise SystemExit(f"{precision}: logged grad norms differ from nanolab train()")


def lr_schedules(cfg_base) -> dict:
    cases = {
        "tiny_cosine": dict(schedule="cosine", warmup_steps=5, max_steps=20),
        "tiny_wsd": dict(schedule="wsd", warmup_steps=5, max_steps=20),
        # framework-design.md §10 acceptance run: warmup 30, cosine over 300.
        "acceptance_cosine": dict(schedule="cosine", warmup_steps=30, max_steps=300),
        "acceptance_wsd": dict(schedule="wsd", warmup_steps=30, max_steps=300),
    }
    out = {}
    for name, over in cases.items():
        cfg = dataclasses.replace(cfg_base, lr_max_steps=0, **over)
        sched = make_schedule(cfg)
        peak = max(_peak_lr(cfg), 1e-12)
        out[name] = {
            "schedule": cfg.schedule, "warmup_steps": cfg.warmup_steps,
            "total_steps": cfg.max_steps, "lr_floor_frac": cfg.lr_floor_frac,
            "wsd_decay_frac": cfg.wsd_decay_frac, "peak": peak,
            # exactly apply_lr's `frac` (schedules.py:124)
            "multipliers": [sched(s) / peak for s in range(LR_STEPS)],
        }
    return out


def fixtures(d: Path) -> None:
    common.setup_determinism()
    cfg = trace_config()
    k = cfg.grad_accum
    init, init_meta = common.read_safetensors(d / "init.safetensors")
    spec = common.spec_from_config(cfg)
    if json.loads(init_meta["ojas.spec"]) != spec:
        raise SystemExit("init.safetensors spec does not match the trace config")
    starts = batches.load_starts(d / "batch_starts.json")
    for key, want in (("seed", SEED), ("seq_len", cfg.block_size),
                      ("batch", cfg.batch_size), ("accum", k)):
        if starts[key] != want:
            raise SystemExit(f"batch_starts.json {key}={starts[key]}, need {want}")
    if starts["steps"] < TRACE_STEPS:
        raise SystemExit(f"batch_starts.json has {starts['steps']} steps, need {TRACE_STEPS}")
    bin_path = d / "tokens.bin"
    replay = batches.StartReplay(starts, bin_path)
    x, y = replay.micro(0, 0)
    batch_id = {"step": 0, "micro": 0, "starts": starts["starts"][:cfg.batch_size]}

    def meta(kind: str, **extra) -> dict[str, str]:
        prov = common.provenance(GENERATOR, cfg, kind=kind, seed=SEED, **extra)
        return {"ojas.oracle": common.canonical_json(prov)}

    # (b) forward
    model = fresh_model(cfg, init)
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)
    loss_v = common.finite_float(loss)
    common.write_safetensors(
        d / "forward.safetensors",
        {"x": x, "y": y, "logits": logits.contiguous()},
        meta("forward", batch=batch_id, loss=loss_v, loss_f32_bits=f32_bits(loss_v),
             logits_checksums=logits_checksums(logits)))

    # (c) gradients at the init
    grads, gloss, no_grad, zero = grads_of(fresh_model(cfg, init), x, y, k)
    common.write_safetensors(
        d / "grads_init.safetensors", grads,
        meta("grads", batch=batch_id, params="init", loss=gloss, loss_seed=1.0 / k,
             no_grad=no_grad, zero_grad=zero))

    # (d) traces, cross-checked against nanolab train()
    traces = {}
    for precision in ("f32", "bf16"):
        rec, snap, _ = run_trace(cfg, init, replay, precision, TRACE_STEPS)
        # The snapshot is the state after CROSSCHECK_STEPS (== PARAM_SNAPSHOT_STEP)
        # steps of a 40-step schedule, which is what train() runs below.
        crosscheck_with_nanolab_train(cfg, init, starts, bin_path, precision,
                                      {**rec, "params": snap})
        traces[precision] = rec
        common.write_safetensors(
            d / f"trace_ns5_{precision}.safetensors", snap,
            meta("trace", ns5=precision,
                 ns5_patch=NS5_F32_PATCH if precision == "f32" else "none (stock nanolab)",
                 steps=TRACE_STEPS, params_after_step=PARAM_SNAPSHOT_STEP,
                 crosschecked_steps=CROSSCHECK_STEPS, **rec))
        if precision == "f32":
            step5 = snap

    # (c') gradients at the step-5 f32-trace parameters, same batch
    grads5, gloss5, no_grad5, zero5 = grads_of(fresh_model(cfg, step5), x, y, k)
    common.write_safetensors(
        d / "grads_step5.safetensors", grads5,
        meta("grads", batch=batch_id, params="trace_ns5_f32 after step 5", loss=gloss5,
             loss_seed=1.0 / k, no_grad=no_grad5, zero_grad=zero5))

    # (e) LR schedules
    doc = {"op": "lr_schedule", "steps": LR_STEPS, **lr_schedules(cfg),
           **common.provenance(GENERATOR)}
    tmp = d / "lr_schedules.json.tmp"
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=1, allow_nan=False) + "\n")
    tmp.replace(d / "lr_schedules.json")

    delta = [abs(a - b) for a, b in zip(traces["f32"]["mean_loss"], traces["bf16"]["mean_loss"])]
    print(f"forward loss {loss_v:.7f}; f32 trace {traces['f32']['mean_loss'][0]:.6f} -> "
          f"{traces['f32']['mean_loss'][-1]:.6f}; max |f32-bf16| mean loss "
          f"{max(delta[:PARAM_SNAPSHOT_STEP]):.3e} (steps 1-5), {max(delta):.3e} (1-40); "
          f"init zero grads {len(zero)}, no grad {no_grad}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tb = sub.add_parser("token-bin")
    tb.add_argument("--out", type=Path, required=True)
    fx = sub.add_parser("fixtures")
    fx.add_argument("--dir", type=Path, default=common.FIXTURE_DIR)
    args = ap.parse_args()
    if args.cmd == "token-bin":
        write_token_bin(args.out)
    else:
        fixtures(args.dir)


if __name__ == "__main__":
    main()
