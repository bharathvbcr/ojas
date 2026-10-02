# CPU hot path for one step

Read 2026-10-01. Counts are trip counts of the loops, not a profile. No `cargo test` was run.

> **Superseded in part (later on 2026-10-01).** File and line references below are from the tree as read then. Three things have changed since:
> - **Host tensors hold typed values** (`HostData`), not bytes. `contiguous_bytes` and `from_le_fill` are gone, kernels borrow `Tensor::f32_slice` / `u32_slice`, and in-place writes go through `write_f32` (`ojas-core/src/tensor.rs`, [typed-storage-plan.md](typed-storage-plan.md)).
> - **AdamW under the default Fast tier does its element arithmetic in f32.** Only Exact keeps f64 moments.
> - **On macOS a Fast GEMM of at least 2¹³ multiply-adds is one Accelerate call** (`ojas_cpu::FAST_WHOLE_CALL_MACS`).

## Index

`devmap status --json` reported `is_fresh: true`, `source_freshness: true`, `rebuild_required: false`. After the reads below, generation was 528 and no file under `ojas-cpu/src` was newer than `.devcouncil/codeintel/devmap.sqlite`. Search and explore envelopes do not re-check freshness; they say to use `status`.

| query | truncated | shown / total | walk_incomplete |
| --- | --- | --- | --- |
| `devmap search matmul` | false | 2 / 2 | not set (search) |
| `devmap search linear` | true | 10 / 17 (hidden 7) | not set |
| `devmap search attention` | false | 5 / 5 | not set. No hit in `ojas-cpu`. CPU attention is named `causal_sdpa_*`. |
| `devmap search causal` | true | 8 / 24 (hidden 16) | not set. Shown page includes `ojas-cpu/src/attn.rs` `causal_sdpa_forward` [12, 38], `causal_sdpa_backward` [44, 96], `causal_probs` [139, 181]. |
| `devmap search cross_entropy` | true | 8 / 13 (hidden 5) | not set. Shown page includes `ojas-cpu/src/pointwise.rs` `cross_entropy` [369, 457]. |
| `devmap explore matmul` | callers truncated false (1 / 1, `newton_schulz`); blast layers truncated false (3 / 3) | | yes: walk stopped at depth 1 on edges and depth 3 on blast radius; lower bound. Repo-wide unresolved sites are not specific to this symbol. |

The body below is from the sources, not from the truncated search pages.

## Which step

The shape B=2, T=32, d=64, vocab=128, heads=1 is `regression_graph` in `ojas-cpu/tests/torch_ref.rs:319` (`one_step_wall_time`). It is not `TinyTrain` (`ojas-autograd/src/tiny.rs:76-79` is d=16, T=4, vocab=32).

One call does, in order (`ojas-cpu/tests/torch_ref.rs:97-183`):

- RMSNorm, then `linear_forward` for Q, K, V (`[d, d]`) and for logits (`wo` is `[vocab, d]`)
- RoPE, one `causal_sdpa_forward` with shape `[B, 1, T, d]`
- `cross_entropy_mean_forward` and `cross_entropy_mean_backward` (both call `cross_entropy`)
- one `causal_sdpa_backward`
- `linear_backward` for `wo`, `wq`, `wk`, `wv`
- two AdamW steps (`wq` is d×d, the norm weight is length d)

`linear_forward` / `linear_backward` do not call `matmul`. `matmul` (`ojas-cpu/src/linalg.rs:11`) is called only from `newton_schulz` (`ojas-cpu/src/optim.rs:299`, `:300`, `:305`), three products per Newton-Schulz iteration, five iterations (`:297`). This step uses AdamW, so `matmul` does not run.

## Likely cost: `linear_backward`

`ojas-cpu/src/linalg.rs:124-142`. Two f32 loops, reduction index increasing (`linalg.rs:1-5`):

- `grad_x[row, inner]`: `rows * kin * nout`
- `grad_w[col, inner]`: `nout * kin * rows`

For a rank-3 input the row count is the product of every axis except the last (`linalg.rs:170`). Here that is B·T = 64, `kin` = 64.

| call | site | f32 multiply-adds |
| --- | --- | --- |
| `linear_backward` logits `wo` `[128, 64]` | `torch_ref.rs:126` | 2 · 64 · 128 · 64 = 1,048,576 |
| `linear_backward` Q, K, V `[64, 64]` | `torch_ref.rs:150-152` | 3 · 2 · 64³ = 1,572,864 |
| `linear_forward` Q, K, V | `torch_ref.rs:100-102`, loop `linalg.rs:93-101` | 3 · 64³ = 786,432 |
| `linear_forward` logits | `torch_ref.rs:117` | 64 · 128 · 64 = 524,288 |
| `causal_sdpa_backward` | `attn.rs:65-91`, called at `torch_ref.rs:128` | 337,920 of the D-loop products |
| `causal_sdpa_forward` | `attn.rs:23-34`, called at `torch_ref.rs:115` | 135,168 of the D-loop products |
| `cross_entropy`, twice | `pointwise.rs:420-451`, via `backend.rs:408` and `:429` | 2 · 3 · 64 · 128 = 49,152 exp passes |

Linear forward + backward is 3,932,160 multiply-adds. Attention forward + backward is 473,088. The logits backward alone is larger than all of attention.

Each linear product also calls `flat` (`validate.rs:90-102`) and `get` (`validate.rs:156-163`) twice, so the inner loop is checked indexing, not a bare multiply-add. That does not change which function owns the trips.

## O(T²) attention

`causal_sdpa_forward` (`attn.rs:23-34`) and `causal_sdpa_backward` (`attn.rs:65-91`) nest batch, head, query time `t`, then key `j` in `0..=t`, then head dim. With one head that is B · T · (T+1) / 2 · D per pass: 2 · 528 · 64 = 67,584. Forward does scores (`causal_probs`, `attn.rs:150-154`) and the value mix (`attn.rs:27-31`). Backward does `causal_probs` again, the `dprobs` dots (`attn.rs:70-76`), and three accumulations per `(j, d)` (`attn.rs:84-89`). Scale is `1/sqrt(head_dim)` (`ojas-core/src/backend.rs:55`, applied at `attn.rs:155`). Query `t` does not read keys past `t`.

## Cross-entropy is not O(vocab · hidden)

`cross_entropy` (`pointwise.rs:369-456`) walks the last axis only. For logits `[B, T, vocab]` that is B·T·vocab, three times per call (max, exp-sum, probability write) and the step calls it twice. The vocab·hidden work is the logits linear: `wo` is `[vocab, d]` so `linear_forward` / `linear_backward` are O(B·T·vocab·d).

## Hazards verified in the sources

**All-ignored cross-entropy is `NonFinite`.** If every target matches `ignore`, `n_valid == 0` returns `nonfinite` before the grad buffer is allocated (`pointwise.rs:414-416`). The comment at `pointwise.rs:366-368` says this is not a finite loss of 0. Both `cross_entropy_mean_forward` and `cross_entropy_mean_backward` call that function, and the backward contract says an all-ignored batch returns `NonFinite` and does not write (`ojas-core/src/backend.rs:368-369`). A non-finite exp or a non-finite loss or grad also returns `NonFinite` (`pointwise.rs:435-436`, `:453-454`).

**AdamW refuses a non-finite f32 store.** The moment update is in f64 (Exact only since the later change; Fast checks the same intermediates in f32). After `p += delta`, a finite `p` whose `as f32` is not finite returns `nonfinite` (`ojas-cpu/src/optim.rs:127-134`). The same check applies to the f32 moment stores (`optim.rs:137-140`). `adamw_step` writes param and moments only after `adamw` returns (`backend.rs:495-499`). Weight decay exactly 0 and a zero delta keep the original param bits (`optim.rs:124-125`). This step’s norm AdamW uses weight decay 0 (`torch_ref.rs:182`).

**Byte offsets.** (Superseded 2026-10-02: `f32_in` is deleted and no op copies an operand; every op reads the same window through `validate::f32_values`, `f32_operands` or `Shared`.) `f32_in` copied through `Tensor::to_f32_vec` (`validate.rs:36`). Since typed storage, that is a copy of the `f32_slice` window starting at element `byte_offset / 4` (`ojas-core/src/tensor.rs:333`, `:354`). Kernels that read in place borrow the same window through `validate::f32_values`. A non-contiguous view is refused before the kernel (`validate.rs:246`). In-place writes use `write_f32`, which goes through the private `writable_f32`, on the same window (`tensor.rs:425`, `:445`). `zero_extent_inf_offset_and_budget_do_not_corrupt_inputs` checks a `narrow(8)` window (`byte_offset == 8`) for silu and embedding, and an AdamW step on that view must leave the prefix bytes unchanged (`ojas-cpu/tests/redteam.rs:397-431`). Kernels that receive a copied `&[f32]` do not see the offset; skipping the copy and reading the allocation at byte 0 would.

**`torch_ref` 1e-4 bound.** `one_step_matches_torch_2_13_float32` requires the tiny graph (B=1, T=4, d=16, vocab=32) to stay within 1e-4 absolute of the frozen torch 2.13.0 f32 dump on the loss, the updated `wq`, the updated norm, and the worst of cosine multiplier, RMSNorm output, attention, logits, loss, both grads, and both updated weights (`torch_ref.rs:224-227`). `one_step_wall_time` checks the same loss bound before it times anything (`torch_ref.rs:300-305`). The larger shape is only required to be finite (`torch_ref.rs:338-341`). Reductions accumulate in f32 from index 0 (`linalg.rs:1-5`, `lib.rs:3-8`). The linear loops, causal softmax (subtract max, then exp, `attn.rs:149-179`), and the cross-entropy three-pass exp (`pointwise.rs:425-448`) are the arithmetic that bound covers.
