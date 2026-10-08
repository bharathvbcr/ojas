// Value-embedding gather + GEMM helpers (Phase D).
#include <metal_stdlib>
using namespace metal;

kernel void ve_gather_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *ve_emb [[buffer(1)]],  // [V, De]
    device float *rows [[buffer(2)]],          // [BT, De]
    constant uint &BT [[buffer(3)]],
    constant uint &De [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BT) return;
    const int tok = input_ids[gid];
    if (tok < 0) {
        device float *out = rows + gid * De;
        for (uint d = 0; d < De; ++d) out[d] = 0.0f;
        return;
    }
    const device float *erow = ve_emb + (uint)tok * De;
    device float *out = rows + gid * De;
    for (uint d = 0; d < De; ++d) out[d] = erow[d];
}

/// out = h * (ve_scale * layer_scale); also used to form scaled output after GEMM.
kernel void ve_scale_out_f32(
    device float *h [[buffer(0)]],                // [BT, kv] in/out
    device const float *ve_scale [[buffer(1)]],
    device const float *layer_scale [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    h[gid] *= (*ve_scale) * (*layer_scale);
}

/// d_h = d_out * s; accumulate d(ve_scale) and d(layer_scale) via h_pre · d_out.
kernel void ve_scale_bwd_f32(
    device const float *h_pre [[buffer(0)]],       // [BT, kv] pre-scale (emb @ proj)
    device const float *d_out [[buffer(1)]],
    device const float *ve_scale [[buffer(2)]],
    device const float *layer_scale [[buffer(3)]],
    device float *d_h [[buffer(4)]],
    device atomic_float *d_ve_scale [[buffer(5)]],
    device atomic_float *d_layer_scale [[buffer(6)]],
    constant uint &n [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    threadgroup float shared[256];
    float local = 0.0f;
    const float s = (*ve_scale) * (*layer_scale);
    if (gid < n) {
        float dout = d_out[gid];
        d_h[gid] = dout * s;
        local = h_pre[gid] * dout;
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
        float ds = shared[0];
        atomic_fetch_add_explicit(d_ve_scale, ds * (*layer_scale), memory_order_relaxed);
        atomic_fetch_add_explicit(d_layer_scale, ds * (*ve_scale), memory_order_relaxed);
    }
}

kernel void ve_scatter_emb_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *d_emb_dense [[buffer(1)]], // [BT, De]
    device atomic_float *d_emb [[buffer(2)]],
    constant uint &BT [[buffer(3)]],
    constant uint &De [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BT) return;
    const int tok = input_ids[gid];
    if (tok < 0) return;
    const uint base = gid * De;
    for (uint d = 0; d < De; ++d) {
        atomic_fetch_add_explicit(&d_emb[(uint)tok * De + d], d_emb_dense[base + d], memory_order_relaxed);
    }
}
