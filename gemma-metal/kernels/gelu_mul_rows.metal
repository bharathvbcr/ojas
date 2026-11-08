// tessl's `mlp_gelu_tanh_bf16` for the m rows of a verify block in one dispatch:
// out[r * n + i] = bf16(gelu_pytorch_tanh(gate[r * n + i]) * up[up_off + r * up_stride + i]).
// The E4B per-layer input block multiplies row r's gate by token r's per-layer input for this
// layer, which sits at a row stride of `layers * n` in the packed table; the decode path runs
// the same per-element expression (same `act.h` helper) through `mlp_gelu_tanh_bf16_up_at`.
#include <metal_stdlib>
#include "act.h"
using namespace metal;

kernel void gelu_mul_rows_bf16(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &up_off [[buffer(5)]],
    constant uint &up_stride [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong total = (ulong)n * rows;
    if ((ulong)gid >= total) return;
    const uint r = gid / n;
    const uint i = gid % n;
    out[gid] = bfloat(tessl_gelu_pytorch_tanh(gate[gid]) *
                      up[(ulong)up_off + (ulong)r * up_stride + i]);
}
