# ojas-infer

`ojas-infer` serves nanolab's default decoder on the CPU: KV-cached incremental decoding, greedy decoding, and seeded temperature / top-k / top-p sampling.

---

## The block

`CpuGpt` runs nanolab's default attention block (`nanolab/mixers.py` `Attention.forward`, `nanolab/model.py` `Block` and `SwiGLU`):

| step | what | weights |
| :--- | :--- | :--- |
| 1 | `h = RMSNorm(x)`, eps `1e-6` | `ln1 [d]` |
| 2 | `q, k, v` linears, no bias | `wq [H*Dh, d]`, `wk`/`wv [Hkv*Dh, d]` |
| 3 | RMS QK-norm over `Dh`, **then** half-split RoPE at the absolute position (base `rope_base`) | `q_norm`, `k_norm [Dh]` |
| 4 | value residual: layer 0 publishes its `v` as `v0`; later layers use `(1-s) v + s v0`, `s = sigmoid(lambda)`. The cache holds blended values | `vr_lambda [1]` (layer 0's is unused) |
| 5 | causal attention, scale `1/sqrt(Dh)`, GQA (query head `i` reads KV head `i / (H/Hkv)`) | |
| 6 | per-head gate `sigmoid(h W_g^T + b_g)` on each head's output (reads the normed `h`) | `gate_w [H, d]`, `gate_b [H]` |
| 7 | `x += y Wo^T`; `x += down(silu(h2 W_gate^T) * (h2 W_up^T))`, `h2 = RMSNorm(x)` | `wo [d, H*Dh]`, `ln2`, `w_gate`/`w_up [hidden, d]`, `w_down [d, hidden]` |

The head is the tied embedding after a final RMSNorm. `head_dim` is its own config field, so `H * Dh` need not equal `d`. RoPE angles are computed in f64 and rounded once, as `ojas-autograd`'s `rope_cache` does; nanolab builds its table in f32, so the last bits differ and the gap grows with position.

## Two forwards, one model

- `forward_token(token, cache)`: one token at position `cache.len()`. The linears and single-query attention are local (the trait linear repacks the whole weight per call). RMSNorm, QK-norm, RoPE, the gate, the value residual, SiLU and the product run on `ojas_cpu::CpuBackend`.
- `forward_sequence(tokens, budget)`: every position at once through the trainer's `Backend` ops, including `causal_sdpa_forward`. The `[1,T,H,D]` to `[1,H,T,D]` reorder and the GQA head repeat happen on the host; `CpuBackend::permute` could take over the reorder.

`tests/parity.rs` holds cached decode, the full forward, and an f64 transcription of nanolab to each other at every position (MHA; GQA with `H*Dh != d`; a second prompt appended to a warm cache). Measured: cached vs full forward is bit-identical, both within 3.5e-7 relative of the f64 reference (asserted at 2e-5 and 2e-4).

## Decoding

`greedy_decode` and `generate` share one loop. The prompt is forwarded from `cache.len()`; every emitted token except the last is forwarded, so `N` new tokens need `prompt + N - 1` free positions. That is checked before any forward: a request that does not fit is `CapacityExceeded` and the cache is untouched. To continue, pass the last returned token as the first token of the next prompt.

`generate` samples with a `SplitMix64` seeded from `GenerateConfig::seed` (the same stream as `ojas_data::CounterRng`), stops after `max_new_tokens` or right after a stop token (included in the output).

`sample_token` order: divide by temperature, keep the top `k`, softmax (f64), keep the smallest most-probable prefix whose mass reaches `top_p`, draw. Differences from nanolab's `GPT.generate`, on purpose:

- temperature `0` is greedy (nanolab divides by `max(T, 1e-6)` and always draws);
- top-k is strict, ranked by (logit desc, index asc), so `top_k = 1` is greedy (nanolab keeps every logit tied with the k-th);
- top-p exists (nanolab has none).

Refused: temperature that is negative or not finite, `top_k == 0`, `top_p` outside `(0, 1]`, NaN or `+inf` logits, and a row with no finite logit. `-inf` is a mask. `argmax_token` stays strict and refuses `-inf` as well.

## Safety properties

- An all-NaN logit row is `NonFinite`, never token 0.
- Linear outputs and residual sums are checked finite, so returned logits are finite.
- The KV cache is one budget-charged allocation; appending past `max_len` is `CapacityExceeded`, never a clamp or overwrite. A cache shaped for another model, or longer than `max_seq`, is refused before any position is written.
