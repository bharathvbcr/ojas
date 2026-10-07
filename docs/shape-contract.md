# Shape contract: `ojas_core::shapes`

**Owner:** `ojas-core/src/shapes.rs`. One pure validator per `Backend` op, `*_dims(...)`. Each reads only dtype and shape metadata and returns a dims struct for the kernels.

**Rule:** CPU, Metal and wgpu call the validator **before** they reserve budget, copy inputs, check placement or dispatch. A malformed call then returns the same variant, op name and detail on every backend, whatever the budget holds.

**Status (2026-10-01):**
- **Written and tested in core.** `cargo test -p ojas-core` passes (75 unit tests), and clippy `-D warnings` is clean.
- **Backend adoption:**
  - **CPU (reported by the ojas-cpu session):** adopted for every op except seven that its heavy-ops lane is still rewriting: embedding forward and backward, CE forward and backward, clip, AdamW and Muon.
    - Gate: `ojas-cpu/tests/shape_first.rs`, 103 malformed cases across three conditions (Budget 0, a cap that fits the inputs only, NaN present), 244 runs in all. Each must match the validator's Debug text with 0 bytes charged. It failed 165 of 244 before adoption and passes after.
    - The local checks were deleted.
  - **wgpu (wgpu round 5, lane-reported; the tests were re-run by the lane after its final edit):** every op adopted.
    - Gate: `ojas-wgpu/tests/shape_first.rs` covers every op in four modes: Budget 0, a cap that fits the inputs only, a NaN operand, and host operands. It failed 130 of 453 runs before adoption and passes after.
    - The local checks were deleted (`backend.rs` +251/−440).
    - One test was re-pointed: `norm.rs` `qk_norm_refuses_in_the_composition_order`. A refused k now records nothing.
    - Residue: in `permute`, contiguity is now checked before dtype.
  - **Metal (Metal round 6; verified: ojas-metal 154/154 serial):** every op adopted.
    - Gate: `ojas-metal/tests/shape_first.rs`, in the same four modes. It failed 160 of 477 runs before adoption and passes 477 of 477 after.
    - The local shape helpers were deleted (`backend.rs` net −104 lines).
    - D9 (grad shape before the head-dim cap) and D16 (CPU's rank-0 text) now hold on Metal.
    - `rms_pair`'s k-validation fallback is removed: a refused k records nothing, as on wgpu. Four rows of `norms.rs` were re-pointed.
    - Unpinned order change: qk-norm with k on another `MetalBackend` returns that error with nothing recorded.
  - **All three backends now call the validators first.** `shape_errors_match_the_cpu_classification` holds by construction.
  - **CPU (complete, reported by the ojas-cpu session):** the seven heavy ops followed, with `ojas-cpu/tests/shape_first_heavy.rs` as their gate.
- **Evidence:** the per-backend line references below come from the shape-contract lane's snapshot. The backends were being edited at the time, so treat the line numbers as approximate.

## Check order inside every validator
1. **Each operand, in argument order:**
   - dtype, else `Dtype`;
   - no zero axis, else `Shape "empty tensor"`;
   - element and byte counts fit `usize`, else `OutOfRange`.
2. **Scalars the layout depends on:** only RMSNorm `eps`, else `NonFinite`.
3. **Ranks and cross-operand equalities**, in the op's documented order, else `Shape`.
4. **Derived output sizes** fit `usize`, else `OutOfRange`.

## Not in the validators (stay in each backend, after the validator)
- placement and contiguity;
- NaN and infinity scans;
- token-id and target ranges, and the all-ignored check;
- optimizer and clip scalars (`check_adamw`, the Muon config check, `clip_scale`);
- device limits: `METAL_MAX_HEAD_DIM`, 32-bit index caps, workgroup and shared-memory limits.

## Decisions
| # | Question | Decision | Why |
| :--- | :--- | :--- | :--- |
| 1 | Variant for count overflow | **`OutOfRange`** | All three backends and `shape_product` already return it. The lane first wrote `Shape`, as briefed; that was changed before adoption so no observable variant moves. |
| 2 | Validator before or after the placement check | **Before** | A call that is both malformed and on the wrong device then reports Shape or Dtype on every backend. |
| 3 | Op names for `rms_qk_norm_*` | **Keep `rms_norm_*`** | All three backends report it today, and `ojas-metal/tests/norms.rs` pins it. |
| 4 | Contiguity | **Stays in the backends**, before any budget charge | `ojas-cpu/tests/redteam_linear.rs` pins that order. When operand 0 is non-contiguous and operand 1 has the wrong dtype, the result changes from Shape to Dtype. |
| 5 | Optimizer config vs budget | **Config before budget** (applies to CPU, D14) | Metal and wgpu already do this. It is not a shape rule, so it is not in `shapes.rs`. |
| 6 | wgpu `clip_grad_norm` `max_norm` timing | **After the norm** (D13; wgpu moves it) | CPU and Metal agree. |

## Where the backends disagree today (resolved by adopting the validators)
| # | Rule | CPU | Metal | wgpu | Canonical |
| :--- | :--- | :--- | :--- | :--- | :--- |
| D2 | Budget charged before cross-operand shape checks | yes (input copies) | no | no | No charge before validation |
| D3 | NaN scan before cross-operand shape checks | yes | no | no | Shape first |
| D5 | Embedding operand order | ids first | table first | table first | Argument order (table first) |
| D6 | Cross-entropy operand order | targets first | logits first | logits first | Argument order (logits first) |
| D7 | AdamW and Muon operand order | param first | param first | grad first | Param first |
| D8 | rms_norm bwd: grad check vs eps and layout | grad first | grad first | layout first | Grad first |
| D9 | sdpa bwd: grad check vs head-dim cap | grad first | cap first | grad first | Grad first (Metal changes from UnsupportedHeadDim to Shape) |
| D10 | gate bwd: grad check vs layout | grad first | grad first | layout first | Grad first |
| D11 | embedding bwd: id range vs grad shape | id range first | shape first | id range first | Shape first (validator), then id range |
| D12 | wgpu id lookup before the table rank check | — | — | lookup first | Validator first |
| D15 | head_dim > u32::MAX | OutOfRange | Unsupported | OutOfRange | Device limit; stays in the backends |
| D16 | Rank-0 detail text | specific | `"rank 0 input"` | specific | CPU's text |
| D17 | `permute` with a zero axis | accepted | refused | refused | **Refused** as `"empty tensor"` by `permute_dims`, like every other op (Metal cannot hold an empty tensor). Rank 0 with `dims` `[]` is accepted everywhere. |

## Adoption gate (every backend)
No existing test asserts that a malformed call returns Shape under `Budget::new(0)`, but that is the invariant this contract exists for. Each adoption lane adds a per-op sweep for two cases:
- a malformed call under `Budget::new(0)`;
- a malformed call under a cap that fits the inputs but not the output.

Both must return the validator's variant. The sweep must fail before adoption wherever D2 or D3 applies.

## `accumulate_grad`, `permute` and the hybrid-layer ops (2026-10-07)
- `accumulate_grad_dims`: `residual_add_forward_dims`'s rules under the name `accumulate_grad`. CPU, Metal, wgpu and the trait default call it first; the three per-backend copies and the two test-local copies are deleted.
- `permute_dims`: the operand rule (F32, no zero axis, sizes fit) and then `permute_output_shape`'s axis rules. `permute_output_shape`, `inverse_permutation` and `MAX_PERMUTE_RANK` moved here from `backend.rs` (same names, re-exported); `Tape::permute` still calls `permute_output_shape` before it records. Every backend calls `permute_dims` before placement and contiguity, which also removes the wgpu residue where contiguity came before dtype.
- Gates: the `permute` and `accumulate_grad` rows of `ojas-{cpu,metal,wgpu}/tests/shape_first.rs` (zero-axis and u32 cases added), and `edge_cases_match_the_cpu_error_for_error_and_bit_for_bit` in `ojas-{metal,wgpu}/tests/permute.rs` (rank 0, unit extents, rank 8 and 9, a zero axis as a host tensor and as a device view, u32, bad axes).
- `causal_conv1d_silu_*_dims`, `gated_rms_norm_*_dims`, `rope_partial_*_dims` validate the Qwen3.5 hybrid-layer ops (`Conv1dDims`, `RmsDims`, `PartialRopeDims`).
