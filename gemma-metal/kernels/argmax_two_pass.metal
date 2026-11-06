// Greedy softcap+argmax over the full vocabulary in two dispatches spread over the GPU.
//
// `softcap_argmax_one_pass` scans all 262,144 logits with one threadgroup (one GPU core).
// Here `argmax_two_pass_partials` gives each of G threadgroups a contiguous span and writes
// one (value, index) per threadgroup, and `argmax_two_pass_final` folds the G partials.
// Same rule as the one-pass kernel: the largest finite softcapped value, the smallest index
// among equal values, non-finite logits skipped, ARGMAX_NONE when none is finite.
#include <metal_stdlib>
#include "softcap.h"
using namespace metal;

constant uint ARGMAX2_NONE = 0xFFFFFFFFu;
constant uint ARGMAX2_TG = 256u;

inline void argmax2_fold(threadgroup float *val, threadgroup uint *idx, uint a, uint b) {
    const uint ib = idx[b];
    if (ib == ARGMAX2_NONE) return;
    const uint ia = idx[a];
    if (ia == ARGMAX2_NONE || val[b] > val[a] || (val[b] == val[a] && ib < ia)) {
        val[a] = val[b];
        idx[a] = ib;
    }
}

inline void argmax2_reduce(threadgroup float *val, threadgroup uint *idx, uint lid, uint tptg) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tptg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            argmax2_fold(val, idx, lid, lid + stride);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void argmax_two_pass_partials(
    device const float *logits [[buffer(0)]],
    device uint *part_idx [[buffer(1)]],
    device float *part_val [[buffer(2)]],
    device const float *softcap_ptr [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &span [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tg [[threadgroup_position_in_grid]])
{
    threadgroup float tg_val[ARGMAX2_TG];
    threadgroup uint tg_idx[ARGMAX2_TG];
    const float softcap = *softcap_ptr;
    const ulong lo = (ulong)tg * span;
    const ulong hi = min(lo + span, (ulong)n);

    float best = -FLT_MAX;
    uint best_i = ARGMAX2_NONE;
    // Ascending indices per lane, so an equal later value never replaces the one it has.
    for (ulong i = lo + lid; i < hi; i += tptg) {
        const float raw = logits[i];
        if (!isfinite(raw)) continue;
        const float v = softcap > 0.0f ? tessl_apply_softcap(raw, softcap) : raw;
        if (best_i == ARGMAX2_NONE || v > best) {
            best = v;
            best_i = (uint)i;
        }
    }
    tg_val[lid] = best;
    tg_idx[lid] = best_i;
    argmax2_reduce(tg_val, tg_idx, lid, tptg);
    if (lid == 0) {
        part_idx[tg] = tg_idx[0];
        part_val[tg] = tg_val[0];
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void argmax_two_pass_final(
    device const uint *part_idx [[buffer(0)]],
    device const float *part_val [[buffer(1)]],
    device uint *out_token [[buffer(2)]],
    constant uint &groups [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float tg_val[ARGMAX2_TG];
    threadgroup uint tg_idx[ARGMAX2_TG];
    float best = -FLT_MAX;
    uint best_i = ARGMAX2_NONE;
    for (uint g = lid; g < groups; g += tptg) {
        const uint i = part_idx[g];
        if (i == ARGMAX2_NONE) continue;
        const float v = part_val[g];
        if (best_i == ARGMAX2_NONE || v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    tg_val[lid] = best;
    tg_idx[lid] = best_i;
    argmax2_reduce(tg_val, tg_idx, lid, tptg);
    if (lid == 0) {
        out_token[0] = tg_idx[0];
    }
}
