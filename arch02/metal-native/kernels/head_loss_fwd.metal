// Head: softcap + fused CE mean loss. Logits GEMM is separate (tied embed).
#include <metal_stdlib>
using namespace metal;

/// Per-row CE contribution: -log_softmax(logits)[target]. Writes [rows] then host/device reduces.
kernel void ce_row_f32(
    device const float *logits [[buffer(0)]], // [rows, V]
    device const int *targets [[buffer(1)]],  // [rows]
    device float *row_loss [[buffer(2)]],     // [rows]
    constant uint &rows [[buffer(3)]],
    constant uint &V [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const ulong row_off = (ulong)gid * (ulong)V;
    const device float *row = logits + row_off;
    // max for stability
    float m = -INFINITY;
    for (uint v = 0; v < V; ++v) m = max(m, row[v]);
    float sum = 0.0f;
    for (uint v = 0; v < V; ++v) sum += exp(row[v] - m);
    const int tgt = targets[gid];
    if (tgt < 0 || (uint)tgt >= V) {
        row_loss[gid] = NAN;
        return;
    }
    const float log_prob = (row[(uint)tgt] - m) - log(sum);
    row_loss[gid] = -log_prob;
}

/// Mean of row_loss → single scalar (simd tree reduce).
kernel void mean_reduce_f32(
    device const float *row_loss [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    float local = 0.0f;
    for (uint i = gid; i < n; i += tpg) {
        local += row_loss[i];
    }
    shared[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) {
            shared[idx] += shared[idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        out[0] = shared[0] / (float)n;
    }
}

/// Fused softcap + per-row CE into row_loss and optional post logits.
kernel void softcap_ce_row_f32(
    device const float *pre [[buffer(0)]],   // [rows, V]
    device const int *targets [[buffer(1)]],
    device float *post [[buffer(2)]],        // [rows, V]
    device float *row_loss [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &V [[buffer(5)]],
    constant float &softcap [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const ulong row_off = (ulong)gid * (ulong)V;
    const device float *prow = pre + row_off;
    device float *orow = post + row_off;
    float m = -INFINITY;
    for (uint v = 0; v < V; ++v) {
        float z = prow[v] / softcap;
        float p = softcap * tanh(z);
        orow[v] = p;
        m = max(m, p);
    }
    float sum = 0.0f;
    for (uint v = 0; v < V; ++v) sum += exp(orow[v] - m);
    const int tgt = targets[gid];
    if (tgt < 0 || (uint)tgt >= V) {
        row_loss[gid] = NAN;
        return;
    }
    const float log_prob = (orow[(uint)tgt] - m) - log(sum);
    row_loss[gid] = -log_prob;
}

// Note: zero_f32, copy_f32, copy_bf16, add_inplace_f32, transpose2d_f32,
// cast_f32_to_bf16, cast_bf16_to_f32 and softcap_f32 are defined once, in
// tessl/kernels/utils.metal, and linked into this metallib by build.rs.
// They were byte-identical duplicates here; two definitions of one kernel
// is a metallib link error waiting to happen, not a redundancy.
