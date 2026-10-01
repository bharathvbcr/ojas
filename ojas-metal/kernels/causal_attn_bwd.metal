// Causal softmax backward for the tiny step.
//
// Tessl GEMM writes the unscaled products
//   scores[t, j] = dot(Q[t], K[j])
//   dP[t, j]     = dot(dO[t], V[j])
// This kernel writes the causal probabilities and the gradient of those
// products. A key j > t is left at 0 in both P and dS, so it contributes
// nothing to dQ, dK, or dV. One thread owns a query row; T is at most 16.
#include <metal_stdlib>
using namespace metal;

kernel void ojas_attn_pack_plane(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &seq [[buffer(2)]],
    constant uint &heads [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &batch_index [[buffer(5)]],
    constant uint &head_index [[buffer(6)]],
    constant uint &src_elem_off [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    const uint n = seq * head_dim;
    if (id >= n || head_dim == 0u) return;
    const uint t = id / head_dim;
    const uint d = id - t * head_dim;
    const ulong src_i = (((ulong)batch_index * seq + t) * heads + head_index) * head_dim + d;
    dst[id] = src[(ulong)src_elem_off + src_i];
}

kernel void ojas_attn_unpack_plane(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &seq [[buffer(2)]],
    constant uint &heads [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &batch_index [[buffer(5)]],
    constant uint &head_index [[buffer(6)]],
    constant uint &dst_elem_off [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    const uint n = seq * head_dim;
    if (id >= n || head_dim == 0u) return;
    const uint t = id / head_dim;
    const uint d = id - t * head_dim;
    const ulong dst_i = (((ulong)batch_index * seq + t) * heads + head_index) * head_dim + d;
    dst[(ulong)dst_elem_off + dst_i] = src[id];
}

/// P and dS, row-major `[seq, seq]`. `scale` is `head_dim^-0.5`.
/// Grid: one thread per query row.
kernel void ojas_causal_softmax_bwd(
    device const float *scores [[buffer(0)]],
    device const float *d_p [[buffer(1)]],
    device float *probs [[buffer(2)]],
    device float *d_scores [[buffer(3)]],
    constant uint &seq [[buffer(4)]],
    constant float &scale [[buffer(5)]],
    uint t [[thread_position_in_grid]])
{
    if (t >= seq) return;
    float m = -INFINITY;
    for (uint j = 0u; j <= t; ++j) {
        m = max(m, scores[(ulong)t * seq + j] * scale);
    }
    float sum = 0.0f;
    for (uint j = 0u; j <= t; ++j) {
        const float e = precise::exp(scores[(ulong)t * seq + j] * scale - m);
        probs[(ulong)t * seq + j] = e;
        sum += e;
    }
    const float inv = sum > 0.0f ? precise::divide(1.0f, sum) : 0.0f;
    for (uint j = 0u; j <= t; ++j) {
        probs[(ulong)t * seq + j] *= inv;
    }
    for (uint j = t + 1u; j < seq; ++j) {
        probs[(ulong)t * seq + j] = 0.0f;
    }
    float dot = 0.0f;
    for (uint j = 0u; j <= t; ++j) {
        dot += probs[(ulong)t * seq + j] * d_p[(ulong)t * seq + j];
    }
    for (uint j = 0u; j <= t; ++j) {
        const float p = probs[(ulong)t * seq + j];
        d_scores[(ulong)t * seq + j] = p * (d_p[(ulong)t * seq + j] - dot) * scale;
    }
    for (uint j = t + 1u; j < seq; ++j) {
        d_scores[(ulong)t * seq + j] = 0.0f;
    }
}
