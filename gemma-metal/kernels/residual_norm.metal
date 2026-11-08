// Gemma4 residual tail fused with the next GEMV's bf16 input (decode, one row per threadgroup).
//
// `rms_norm_residual_add_f32` followed by `rms_norm_bf16` (or a bf16 cast) is two dependent
// dispatches with a barrier between them, once after attention, once after the FFN and once
// at the end of every layer. This is both in one dispatch, with the same arithmetic in the
// same order: lane `lid` updates `resid[lid], resid[lid + tptg], ...` and then rereads exactly
// those elements, so the second pass needs no device-memory barrier. The two reductions use
// separate scratch regions (`reduce_tree.h` contract).
#include <metal_stdlib>
#include "reduce_tree.h"
using namespace metal;

/// `resid = s * (resid + rms_norm(x) * weight)` (`s == 1`: `resid += rms_norm(x) * weight`),
/// then `out = bf16(rms_norm(resid) * next_weight)` when `normalize_next != 0`, else
/// `out = bf16(resid)`.
[[max_total_threads_per_threadgroup(1024)]]
kernel void rms_norm_residual_add_emit_bf16(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *resid [[buffer(2)]],
    device const float *next_weight [[buffer(3)]],
    device bfloat *out [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant float &layer_scale [[buffer(8)]],
    constant uint &normalize_next [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    if (row >= rows) return;
    threadgroup float scratch[2 * REDUCE_MAX_SIMDGROUPS];
    device const float *xin = x + (ulong)row * dim;
    device float *r = resid + (ulong)row * dim;
    device bfloat *o = out + (ulong)row * dim;

    float ss = 0.0f;
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        float v = xin[d];
        ss += v * v;
    }
    const float inv = rsqrt(reduce_row_add(ss, scratch, sgid, lane, tptg) / (float)dim + eps);
    const float s = layer_scale;
    float ss_next = 0.0f;
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        float y;
        if (s == 1.0f) {
            y = r[d] + xin[d] * inv * weight[d];
        } else {
            y = s * (r[d] + xin[d] * inv * weight[d]);
        }
        r[d] = y;
        ss_next += y * y;
    }
    if (normalize_next == 0u) {
        for (ulong d = lid; d < (ulong)dim; d += tptg) {
            o[d] = (bfloat)r[d];
        }
        return;
    }
    const float inv_next = rsqrt(
        reduce_row_add(ss_next, scratch + REDUCE_MAX_SIMDGROUPS, sgid, lane, tptg) / (float)dim + eps);
    for (ulong d = lid; d < (ulong)dim; d += tptg) {
        o[d] = (bfloat)(r[d] * inv_next * next_weight[d]);
    }
}
