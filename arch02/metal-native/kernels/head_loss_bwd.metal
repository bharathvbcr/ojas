// Head loss backward: CE dlogits + softcap Jacobian fused into d(pre).
#include <metal_stdlib>
using namespace metal;

/// dlogits_post from mean CE, then softcap Jacobian → dlogits_pre.
/// Only needs post (pre unused); softcap jacobian via (post/sc).
kernel void ce_softcap_bwd_f32(
    device const float *logits_post [[buffer(0)]], // [rows, V]
    device const int *targets [[buffer(1)]],
    device float *d_pre [[buffer(2)]],             // [rows, V] out
    constant uint &rows [[buffer(3)]],
    constant uint &V [[buffer(4)]],
    constant float &softcap [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const ulong row_off = (ulong)gid * (ulong)V;
    const device float *row = logits_post + row_off;
    device float *dout = d_pre + row_off;
    const int tgt = targets[gid];
    if (tgt < 0 || (uint)tgt >= V) {
        for (uint v = 0; v < V; ++v) dout[v] = NAN;
        return;
    }

    float m = -INFINITY;
    for (uint v = 0; v < V; ++v) m = max(m, row[v]);
    float sum = 0.0f;
    for (uint v = 0; v < V; ++v) sum += exp(row[v] - m);
    const float inv_sum = 1.0f / sum;
    const float inv_n = 1.0f / (float)rows;
    for (uint v = 0; v < V; ++v) {
        float p = exp(row[v] - m) * inv_sum;
        float d_post = (p - ((int)v == tgt ? 1.0f : 0.0f)) * inv_n;
        float t = row[v] / softcap;
        dout[v] = d_post * (1.0f - t * t);
    }
}

/// Softcap-only Jacobian if d_post already known: d_pre = d_post * (1-(post/sc)^2)
kernel void softcap_bwd_f32(
    device const float *post [[buffer(0)]],
    device const float *d_post [[buffer(1)]],
    device float *d_pre [[buffer(2)]],
    constant float &softcap [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float t = post[gid] / softcap;
    d_pre[gid] = d_post[gid] * (1.0f - t * t);
}
