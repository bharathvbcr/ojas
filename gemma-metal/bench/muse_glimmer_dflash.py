"""MLX port of Meta's MuseGlimmerAssistantModel — the DFlash drafter for Muse Glimmer 30B.

z-lab's `dflash` package cannot load this checkpoint: its `load_draft` reads
`config["dflash_config"]["target_layer_ids"]` and `config["rope_theta"]`, whereas Meta
puts `target_layer_ids`/`mask_token_id` at the top level and `rope_theta` under
`rope_parameters`, and omits `num_target_layers`/`vocab_size` entirely.

The architectures also differ in one way that matters for accept rate. z-lab masks the
speculative block causally on sliding layers; Meta's reference
(`transformers/models/muse_glimmer_assistant/modular_muse_glimmer_assistant.py`) masks it
*bidirectionally* — the block's queries all sit at positions above the cached context, so
they attend to each other in both directions. All five Muse Glimmer drafter layers are
sliding, so running it under z-lab's masking would apply causal attention everywhere the
reference applies bidirectional.

Module names here mirror the checkpoint's tensor names (`encoder.fc`,
`encoder.output_norm_enc`, `layers.N.*`, `norm`), so weights load with no key remapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.cache import RotatingKVCache, KVCache
from mlx_lm.models.rope_utils import initialize_rope

DRAFT_REPO = "meta-models/Muse-Glimmer-30B-assistant"


@dataclass
class AssistantConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    mask_token_id: int
    layer_types: Tuple[str, ...]
    sliding_window: Optional[int]


class Attention(nn.Module):
    """Queries come from the noise block; keys/values from context ++ block.

    Mirrors `MuseGlimmerAssistantAttention`: the context half is cached across blocks and
    the block half is transient, so `cache` only ever holds accepted-token context.
    """

    def __init__(self, config: AssistantConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def _mask(self, ctx_len: int, block_len: int, dtype) -> Optional[mx.array]:
        """Bidirectional within the block, sliding-window over the context."""
        total = ctx_len + block_len
        if self.sliding_window is None or total <= self.sliding_window:
            return None
        q_pos = mx.arange(ctx_len, total).reshape(-1, 1)
        k_pos = mx.arange(total).reshape(1, -1)
        # No upper bound on k_pos: block queries see the whole block, not just the past.
        allowed = k_pos > q_pos - self.sliding_window
        return mx.where(allowed, mx.array(0, dtype), mx.array(-mx.inf, dtype))

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache) -> mx.array:
        B, L, _ = x.shape
        S = x_ctx.shape[1]

        if self.is_sliding:
            keep = self.sliding_window - 1
            if S > keep:
                skip = S - keep
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip

        queries = self.q_proj(x)
        ctx_keys, ctx_values = self.k_proj(x_ctx), self.v_proj(x_ctx)
        blk_keys, blk_values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        blk_keys = self.k_norm(blk_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        blk_values = blk_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        blk_keys = rope(blk_keys, offset=cache.offset + S)

        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, blk_keys], axis=2)
        values = mx.concatenate([values, blk_values], axis=2)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=self._mask(ctx_len, L, queries.dtype)
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: AssistantConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, x_ctx: mx.array, rope, cache) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class ContextProjection(nn.Module):
    """`MuseGlimmerAssistantContextProjection`: concat of target hidden states -> hidden_size."""

    def __init__(self, config: AssistantConfig):
        super().__init__()
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False
        )
        self.output_norm_enc = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_norm_enc(self.fc(x))


class MuseGlimmerDraftModel(nn.Module):
    """Drafter with the same call surface as `dflash.model_mlx.DFlashDraftModel`."""

    def __init__(self, config: AssistantConfig):
        super().__init__()
        self.config = config
        self.encoder = ContextProjection(config)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = initialize_rope(
            dims=config.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config={"rope_type": "default", "rope_theta": config.rope_theta},
            max_position_embeddings=config.max_position_embeddings,
        )
        self.embed_tokens = None
        self.embed_norm = None
        self.lm_head = None
        self.output_multiplier = 1.0
        self.final_logit_softcapping = None

    def bind(self, target_model):
        """Share the target's embedding and head so drafts live in the target's token space.

        Muse Glimmer normalizes embeddings via a weightless `embed_norm` rather than a
        scalar `embed_scale`, and scales logits by `output_multiplier` before softcapping;
        all three must match the target exactly or verification rejects every block.
        """
        language_model = getattr(target_model, "language_model", target_model)
        text_model = language_model.model

        self.embed_tokens = text_model.embed_tokens
        self.embed_norm = text_model.embed_norm
        self.lm_head = language_model.lm_head
        self.output_multiplier = language_model.output_multiplier
        self.final_logit_softcapping = language_model.final_logit_softcapping
        return self

    def make_cache(self) -> List[Any]:
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0))
            else:
                caches.append(KVCache())
        return caches

    def __call__(
        self,
        inputs: mx.array,
        target_hidden: mx.array,
        cache: List[Any],
        logits_start: int = 0,
    ) -> mx.array:
        h = self.embed_norm(self.embed_tokens(inputs))
        h_ctx = self.encoder(target_hidden)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, layer_cache)
        if logits_start:
            h = h[:, logits_start:]
        logits = self.lm_head(self.norm(h)) * self.output_multiplier
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = mx.tanh(logits / cap) * cap
        return logits


def load_draft(draft_id: str = DRAFT_REPO, target_config: Optional[dict] = None):
    """Load Meta's drafter checkpoint.

    `num_hidden_layers` for the *target* is not recorded in the drafter config, so
    `target_layer_ids` is validated against `target_config` when one is supplied.
    """
    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    cfg = json.loads((path / "config.json").read_text())

    model_type = cfg.get("model_type")
    if model_type != "muse_glimmer_assistant":
        raise ValueError(
            f"{draft_id} has model_type={model_type!r}; this loader implements "
            "'muse_glimmer_assistant'. Use dflash.load_draft for z-lab DFlash drafters."
        )

    layer_types = tuple(
        cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
    )
    if len(layer_types) != cfg["num_hidden_layers"]:
        raise ValueError("layer_types length must match num_hidden_layers")
    unknown = set(layer_types) - {"full_attention", "sliding_attention"}
    if unknown:
        raise ValueError(f"Unsupported layer_types: {sorted(unknown)}")
    if "sliding_attention" in layer_types and cfg.get("sliding_window") is None:
        raise ValueError("sliding_attention layers require sliding_window")

    target_layer_ids = tuple(cfg["target_layer_ids"])
    if target_config is not None:
        depth = target_config["num_hidden_layers"]
        if max(target_layer_ids) >= depth:
            raise ValueError(
                f"target_layer_ids {target_layer_ids} exceed target depth {depth}"
            )

    config = AssistantConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=float(cfg["rope_parameters"]["rope_theta"]),
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=target_layer_ids,
        mask_token_id=cfg["mask_token_id"],
        layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
    )

    model = MuseGlimmerDraftModel(config)
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))

    expected = set(k for k, _ in _flatten(model.parameters()))
    missing = expected - weights.keys()
    unexpected = weights.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"drafter weight mismatch\n  missing: {sorted(missing)[:8]}\n"
            f"  unexpected: {sorted(unexpected)[:8]}"
        )

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


def _flatten(params, prefix=""):
    if isinstance(params, dict):
        for k, v in params.items():
            yield from _flatten(v, f"{prefix}{k}.")
    elif isinstance(params, list):
        for i, v in enumerate(params):
            yield from _flatten(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], params
