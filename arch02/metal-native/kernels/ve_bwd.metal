// Value-embedding backward.
#include <metal_stdlib>
using namespace metal;

/// d(ve_out) → d(embed), d(proj), d(scale), d(layer_scale).
/// ve_out = (emb @ proj) * ve_scale * layer_scale
kernel void ve_bwd_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *ve_emb [[buffer(1)]],
    device const float *ve_proj [[buffer(2)]], // [De, kv]
    device const float *ve_scale [[buffer(3)]],
    device const float *layer_scale [[buffer(4)]],
    device const float *d_out [[buffer(5)]],   // [BT, kv]
    device atomic_float *d_emb [[buffer(6)]],
    device atomic_float *d_proj [[buffer(7)]],
    device atomic_float *d_ve_scale [[buffer(8)]],
    device atomic_float *d_layer_scale [[buffer(9)]],
    constant uint &B [[buffer(10)]],
    constant uint &T [[buffer(11)]],
    constant uint &De [[buffer(12)]],
    constant uint &kv_dim [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const int tok = input_ids[gid];
    const float s = (*ve_scale) * (*layer_scale);
    const device float *erow = ve_emb + (uint)tok * De;
    const uint base = gid * kv_dim;

    // Reconstruct h = emb @ proj (pre scale)
    // out = h * s
    // d_h = d_out * s
    // d_s += sum(h * d_out)
    float ds = 0.0f;
    for (uint o = 0; o < kv_dim; ++o) {
        float dout = d_out[base + o];
        float h = 0.0f;
        for (uint d = 0; d < De; ++d) {
            h += erow[d] * ve_proj[d * kv_dim + o];
        }
        ds += h * dout;
        float dh = dout * s;
        for (uint d = 0; d < De; ++d) {
            float e = erow[d];
            float p = ve_proj[d * kv_dim + o];
            atomic_fetch_add_explicit(&d_proj[d * kv_dim + o], e * dh, memory_order_relaxed);
            atomic_fetch_add_explicit(&d_emb[(uint)tok * De + d], p * dh, memory_order_relaxed);
        }
    }
    // s = ve_scale * layer_scale
    // d(ve_scale) += ds * layer_scale; d(layer_scale) += ds * ve_scale
    atomic_fetch_add_explicit(d_ve_scale, ds * (*layer_scale), memory_order_relaxed);
    atomic_fetch_add_explicit(d_layer_scale, ds * (*ve_scale), memory_order_relaxed);
}
