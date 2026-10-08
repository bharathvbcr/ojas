// XSA: subtract unit-v projection (GQA-aware). Out-of-place (kills tape deep_copy).
#include <metal_stdlib>
using namespace metal;

kernel void xsa_fwd_f32(
    device const float *y_in [[buffer(0)]],    // [B,T,H,D] flash output
    device float *y_out [[buffer(1)]],         // [B,T,H,D] post-XSA
    device const float *v [[buffer(2)]],       // [B,T,Hkv,D] mixed V
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &Hkv [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    // One thread per (b, t, hkv) — updates all Q heads in the group.
    if (gid >= B * T * Hkv) return;
    const uint hkv = gid % Hkv;
    const uint tmp = gid / Hkv;
    const uint t = tmp % T;
    const uint b = tmp / T;
    const uint group = H / Hkv;

    const uint v_off = ((b * T + t) * Hkv + hkv) * D;
    float norm_sq = 0.0f;
    for (uint d = 0; d < D; ++d) {
        float vv = v[v_off + d];
        norm_sq += vv * vv;
    }
    const float inv_n = rsqrt(norm_sq + eps);
    thread float vn[64];
    const uint d_lim = min(D, 64u);
    for (uint d = 0; d < d_lim; ++d) {
        vn[d] = v[v_off + d] * inv_n;
    }

    for (uint g = 0; g < group; ++g) {
        const uint h = hkv * group + g;
        const uint y_off = ((b * T + t) * H + h) * D;
        float proj = 0.0f;
        for (uint d = 0; d < d_lim; ++d) {
            proj += y_in[y_off + d] * vn[d];
        }
        for (uint d = 0; d < d_lim; ++d) {
            y_out[y_off + d] = y_in[y_off + d] - proj * vn[d];
        }
    }
}
