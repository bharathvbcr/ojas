// Per-head sigmoid output gate.
//
// The linear map pre = x @ W^T is a tessl GEMM. This file only applies
// sigmoid(pre + bias) to the attention output, and the matching backward.
// One thread owns each (row, head) on the backward so the head_dim sum is
// the same left-to-right f32 loop the CPU reference uses.
//
// Compiled with -fmetal-math-mode=safe and -ffp-contract=off.
#include <metal_stdlib>
using namespace metal;

inline float ojas_sigmoid(float x)
{
    const float e = precise::exp(-fabs(x));
    const float r = precise::divide(1.0f, 1.0f + e);
    return x >= 0.0f ? r : e * r;
}

/// out[r, h, d] = attn[r, h, d] * sigmoid(pre[r, h] + bias[h])
/// Grid: x = head dim, y = row * n_head + head.
/// Lengths are element counts. A lying length makes the thread return.
kernel void ojas_per_head_gate_fwd(
    device const float *attn [[buffer(0)]],
    device const float *pre [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &n_head [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &attn_len [[buffer(7)]],
    constant uint &pre_len [[buffer(8)]],
    constant uint &bias_len [[buffer(9)]],
    constant uint &out_len [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint rh = gid.y;
    const uint units = rows * n_head;
    if (d >= head_dim || rh >= units || n_head == 0u) return;
    const uint h = rh % n_head;
    const uint r = rh / n_head;
    const ulong pre_i = (ulong)r * n_head + h;
    const ulong i = pre_i * head_dim + d;
    if (i >= attn_len || i >= out_len || pre_i >= pre_len || h >= bias_len) return;
    const float z = pre[pre_i] + bias[h];
    const float g = ojas_sigmoid(z);
    out[i] = attn[i] * g;
}

/// d_attn = dy * g
/// d_pre = g * (1 - g) * sum_d(dy * attn)
/// Grid: one thread per (row, head). Threads past `rows * n_head` exit.
kernel void ojas_per_head_gate_bwd(
    device const float *attn [[buffer(0)]],
    device const float *pre [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const float *dy [[buffer(3)]],
    device float *d_attn [[buffer(4)]],
    device float *d_pre [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &n_head [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &plane_len [[buffer(9)]],
    constant uint &pre_len [[buffer(10)]],
    constant uint &bias_len [[buffer(11)]],
    uint rh [[thread_position_in_grid]])
{
    const uint units = rows * n_head;
    if (rh >= units || n_head == 0u) return;
    const uint h = rh % n_head;
    const uint r = rh / n_head;
    const ulong pre_i = (ulong)r * n_head + h;
    if (pre_i >= pre_len || h >= bias_len) return;
    const float z = pre[pre_i] + bias[h];
    const float g = ojas_sigmoid(z);
    const ulong base = pre_i * head_dim;
    if (base + head_dim > plane_len) return;
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        const float term = dy[base + d] * attn[base + d];
        d_attn[base + d] = dy[base + d] * g;
        acc += term;
    }
    d_pre[pre_i] = acc * g * (1.0f - g);
}

/// d_bias[h] = sum over rows of d_pre[r, h]. One thread per head.
kernel void ojas_per_head_gate_dbias(
    device const float *d_pre [[buffer(0)]],
    device float *d_bias [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &n_head [[buffer(3)]],
    constant uint &pre_len [[buffer(4)]],
    constant uint &bias_len [[buffer(5)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= n_head || h >= bias_len) return;
    float s = 0.0f;
    for (uint r = 0; r < rows; r++) {
        const ulong i = (ulong)r * n_head + h;
        if (i >= pre_len) return;
        s += d_pre[i];
    }
    d_bias[h] = s;
}
