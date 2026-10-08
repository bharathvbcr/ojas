# ojas-infer

`ojas-infer` serves nanolab's default decoder: KV-cached incremental decoding on the CPU (`CpuGpt`) and on any `Backend` (`DeviceDecoder`), greedy decoding, and seeded temperature / top-k / top-p sampling.

The model is defined once, in `ojas-model`. `GptConfig` is `ojas_model::ModelSpec`, `GptWeights` is `ojas_model::ModelParams<Tensor>` and `BlockWeights` is `ojas_model::BlockParams<Tensor>`, all under nanolab's `state_dict` names. Both decoders check the weights against `ojas_model::param_table`.

---

## The block

Both decoders run nanolab's default attention block (`nanolab/mixers.py` `Attention.forward`, `nanolab/model.py` `Block` and `SwiGLU`):

| step | what | weights |
| :--- | :--- | :--- |
| 1 | `h = RMSNorm(x)`, eps `rms_eps` (`1e-6`) | `norm1 [d]` |
| 2 | `q, k, v` linears, no bias | `q_proj [H*Dh, d]`, `k_proj`/`v_proj [Hkv*Dh, d]` |
| 3 | RMS QK-norm over `Dh`, **then** half-split RoPE at the absolute position (base `rope_base`) | `q_norm`, `k_norm [Dh]` |
| 4 | value residual: layer 0 publishes its `v` as `v0`; later layers use `(1-s) v + s v0`, `s = sigmoid(lambda)`. The cache holds blended values | `vr_lambda [1]` (layer 0's is unused) |
| 5 | causal attention, scale `1/sqrt(Dh)`, GQA (query head `i` reads KV head `i / (H/Hkv)`) | |
| 6 | per-head gate `sigmoid(h W_g^T + b_g)` on each head's output (reads the normed `h`) | `gate_w [H, d]`, `gate_b [H]` |
| 7 | `x += y Wo^T`; `x += down(silu(h2 W_gate^T) * (h2 W_up^T))`, `h2 = RMSNorm(x)` | `o_proj [d, H*Dh]`, `norm2`, `ffn_gate`/`ffn_up [hidden, d]`, `ffn_down [d, hidden]` |

The head is the tied embedding (`tok_emb`) after a final RMSNorm (`norm_f`). `head_dim` is its own config field, so `H * Dh` need not equal `d`. RoPE rows come from `ojas_model::Rope`: angles computed in f64 and rounded once, as `ojas-autograd`'s `rope_cache` does; nanolab builds its table in f32, so the last bits differ and the gap grows with position.

## Three forwards, one model

- `DeviceDecoder<B>::forward(tokens)`: decode on any `Backend` with the KV cache resident there, one `[1, slots, Hkv, D]` key and value tensor per layer. Each layer is `ojas_model::block_with` on `Eval`, the same block the trainer uses. Only the attention step belongs to this crate (`src/cache.rs`): it writes the new positions' post-RoPE keys and blended values with `kv_cache_write`, then attends with `cached_attention_forward` at `kv_len = len + Tn`. One path serves prefill (several tokens, onto an empty or warm cache) and decode (one token), and it runs GQA. Only the last position runs the final norm and the head. A call past the capacity is `CapacityExceeded` before anything runs, and a failed call leaves `len` unchanged.
- `CpuGpt::forward_tokens(tokens, cache)`: the same step on an `ojas_cpu::CpuBackend` (one thread per core, `Numerics::Fast` by default) against a caller-owned host `KvCache`. `forward_token` is the one-token case. There is no second copy of the block.
- The full forward is `ojas_model::forward_logits` on `ojas_model::Eval`: causal SDPA, with the model's sliding window if it has one.

**Sliding window.** A `ModelSpec` with `window: Some(W)` attends to the last `W` positions on every layer (nanolab's `swa` mixer with `swa_sinks = 0`). Training (the tape's windowed SDPA) and both decoders read `ModelSpec::attention_window`, which is `None` once `W >= max_seq`. The caches are rings (position `j` in slot `j % slots`); a windowed model keeps `2W - 1` slots instead of the capacity. A call of more than `W` tokens runs as `W`-token pieces, each all or nothing: a failure leaves `len` after the last piece that completed. `truncate` refuses a prefix whose window the ring has overwritten.

`tests/parity.rs` holds cached decode, the full forward and an f64 transcription of nanolab to each other at every position: MHA; GQA with `H*Dh != d`; and a second prompt appended to a warm cache. The decoder's prefill of every prefix is held to the full forward too. G7 holds `forward_token`, `Eval` and `DeviceDecoder<CpuBackend>` to each other over a 4-token prefill and 12 decode steps, on MHA and GQA, under both numerics.

Measured on 2026-10-01 (Apple silicon, release):

- **`Numerics::Exact`:** all three are bit-identical, on MHA and GQA. On the CPU, the fused QK-norm runs the same row kernel as two `rms_norm` calls, and `cached_attention_forward` runs causal SDPA's per-row kernel.
- **`Numerics::Fast`:** cached decode vs the full forward is within 5.7e-7 relative (asserted at 2e-5; G7 at 1e-5). Cached decode is within 3.5e-7 of the f64 reference, and the full forward within 4.1e-7 (both asserted at 2e-4).

`tests/device.rs` runs `DeviceDecoder` on wgpu and on Metal against `CpuGpt` Exact. The run is an 8-token prefill, 36 decode steps, then a 3-token prefill onto the warm cache, on MHA and GQA. Measured worst relative error was 9.9e-7 to 1.25e-6 (asserted at 1e-4), with exactly one readback per call of `4 * vocab` bytes, counted with `Budget::device_readbacks`. A missing device fails unless `OJAS_ALLOW_NO_GPU=1`.

## Decoding

`greedy_decode` and `generate` share one loop (`src/decode.rs`) on both decoders. Both forward the prompt as one prefill. The prompt is forwarded from the cache's current length; every emitted token except the last is forwarded, so `N` new tokens need `prompt + N - 1` free positions. That is checked before any forward: a request that does not fit is `CapacityExceeded` and the cache is untouched. To continue, pass the last returned token as the first token of the next prompt.

`generate` samples with a `SplitMix64` seeded from `GenerateConfig::seed` (the same stream as `ojas_data::CounterRng`), stops after `max_new_tokens` or right after a stop token (included in the output).

`sample_token` order: divide by temperature, keep the top `k`, softmax (f64), keep the smallest most-probable prefix whose mass reaches `top_p`, draw. Differences from nanolab's `GPT.generate`, on purpose:

- temperature `0` is greedy (nanolab divides by `max(T, 1e-6)` and always draws);
- top-k is strict, ranked by (logit desc, index asc), so `top_k = 1` is greedy (nanolab keeps every logit tied with the k-th);
- top-p exists (nanolab has none).

Refused: temperature that is negative or not finite, `top_k == 0`, `top_p` outside `(0, 1]`, NaN or `+inf` logits, and a row with no finite logit. `-inf` is a mask. `argmax_token` stays strict and refuses `-inf` as well.

## Safety properties

- An all-NaN logit row is `NonFinite`, never token 0.
- The backend's ops refuse non-finite results, so returned logits are finite.
- `KvCache` is per-layer budget-charged tensors, refused as one request; appending past `max_len` is `CapacityExceeded`, never a clamp or overwrite. A cache shaped for another model (heads, layers, window), or longer than `max_seq`, is refused before any position is written.
- `reset` and `truncate` (on both `KvCache` and `DeviceDecoder`) roll back to a prefix; the last token `generate` returns is not in the cache.
