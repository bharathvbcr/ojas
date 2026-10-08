// Per-layer PLE embed lookup.
// - ple_lookup: bf16 table [vocab, dim] (synthetic / HF bf16 banks)
// - ple_lookup_q4_mlx: packed MLX affine Q4 [vocab, L*dim] with per-layer slice
#include <metal_stdlib>
using namespace metal;

kernel void ple_lookup(
    device const uint *token_ids [[buffer(0)]],
    device const bfloat *ple_table [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &vocab [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n * dim) return;
    const uint tok_i = gid / dim;
    const uint d = gid % dim;
    const uint tid = token_ids[tok_i];
    if (tid >= vocab) {
        out[gid] = 0.0f;
        return;
    }
    float v = float(ple_table[tid * dim + d]);
    out[gid] = v * scale;
}

kernel void ple_lookup_q4_mlx(
    device const uint *token_ids [[buffer(0)]],
    device const uint *packed [[buffer(1)]],
    device const bfloat2 *sb [[buffer(2)]],
    device const bfloat *biases_unused [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &vocab [[buffer(6)]],
    constant uint &n [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &layer [[buffer(9)]],
    constant uint &num_layers [[buffer(10)]],
    constant uint &group_size [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    (void)biases_unused;
    if (gid >= n * dim) return;
    const uint tok_i = gid / dim;
    const uint d = gid % dim;
    const uint tid = token_ids[tok_i];
    if (tid >= vocab) {
        out[gid] = 0.0f;
        return;
    }
    const uint cols = num_layers * dim;
    const uint col = layer * dim + d;
    const uint groups_per_row = cols / group_size;
    const uint g = col / group_size;
    const uint scale_i = tid * groups_per_row + g;
    const bfloat2 sbv = sb[scale_i];
    const float s = float(sbv.x);
    const float b = float(sbv.y);
    const uint words_per_row = cols / 8u;
    const uint word = packed[tid * words_per_row + (col / 8u)];
    const uint shift = (col % 8u) * 4u;
    const uint nibble = (word >> shift) & 0x0fu;
    out[gid] = (s * float(nibble) + b) * scale;
}

/// Residual combine: dst += combine_scale * src (f32).
kernel void ple_residual_add(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant float &combine_scale [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] += combine_scale * src[gid];
}

// ---------------------------------------------------------------------------
// Fused PLE Q4 lookup + residual combine (one dispatch instead of two).
//
// Layer-fusion v1 (opt-in: GEMMA_METAL_FUSE_PLE=1 / GEMMA_METAL_FUSE_LAYER=1).
// The two-pass form is `ple_lookup_q4_mlx` -> barrier -> `ple_residual_add`,
// but the dependency is strictly element-local (out[gid] feeds dst[gid] only),
// so no cross-thread visibility is required and the pair collapses safely.
//
// Bit-exactness: the arithmetic order below is deliberately identical to the
// two-pass path — v = (s * nibble + b) * scale, then dst += combine_scale * v.
// Do NOT algebraically fold `scale * combine_scale`: that changes rounding.
//
// `out` is still written so PLE intermediates stay inspectable for parity dumps
// (cost is `dim` floats/layer, ~0.1% of the layer's traffic).
//
// Hazard note: the caller must still emit the producer->consumer barrier BEFORE
// this kernel, because `dst` (the residual stream x) is written by the
// preceding o_proj residual. Fusing removes a dispatch, not that RAW edge.
// ---------------------------------------------------------------------------
kernel void ple_lookup_q4_mlx_residual(
    device const uint *token_ids [[buffer(0)]],
    device const uint *packed [[buffer(1)]],
    device const bfloat2 *sb [[buffer(2)]],
    device const bfloat *biases_unused [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &vocab [[buffer(6)]],
    constant uint &n [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &layer [[buffer(9)]],
    constant uint &num_layers [[buffer(10)]],
    constant uint &group_size [[buffer(11)]],
    device float *dst [[buffer(12)]],
    constant float &combine_scale [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    (void)biases_unused;
    if (gid >= n * dim) return;
    const uint tok_i = gid / dim;
    const uint d = gid % dim;
    const uint tid = token_ids[tok_i];
    float v;
    if (tid >= vocab) {
        // Out-of-range token: two-pass writes out = 0 then still executes
        // `dst += combine_scale * 0`. Fall through rather than early-return so
        // the fused path is bit-identical even for signed-zero dst.
        v = 0.0f;
    } else {
        const uint cols = num_layers * dim;
        const uint col = layer * dim + d;
        const uint groups_per_row = cols / group_size;
        const uint g = col / group_size;
        const uint scale_i = tid * groups_per_row + g;
        const bfloat2 sbv = sb[scale_i];
        const float s = float(sbv.x);
        const float b = float(sbv.y);
        const uint words_per_row = cols / 8u;
        const uint word = packed[tid * words_per_row + (col / 8u)];
        const uint shift = (col % 8u) * 4u;
        const uint nibble = (word >> shift) & 0x0fu;
        v = (s * float(nibble) + b) * scale;
    }
    out[gid] = v;
    dst[gid] += combine_scale * v;
}
