// MinGRU value-residual: blend v0_up into h_pre before scan.
#include <metal_stdlib>
using namespace metal;

kernel void mingru_vr_blend_fwd_f32(
    device const float *h_raw [[buffer(0)]],
    device const float *v0_up [[buffer(1)]],
    device const float *vr_lambda [[buffer(2)]],
    device float *h_pre [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &use_v0 [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    if (use_v0 != 0u) {
        h_pre[gid] = vr_lambda[0] * v0_up[gid] + vr_lambda[1] * h_raw[gid];
    } else {
        h_pre[gid] = h_raw[gid];
    }
}

kernel void mingru_vr_blend_bwd_f32(
    device const float *d_h_pre [[buffer(0)]],
    device const float *h_raw [[buffer(1)]],
    device const float *v0_up [[buffer(2)]],
    device const float *vr_lambda [[buffer(3)]],
    device float *d_h_raw [[buffer(4)]],
    device float *d_v0_up [[buffer(5)]],
    device atomic_float *d_vr_lambda [[buffer(6)]],
    constant uint &n [[buffer(7)]],
    constant uint &use_v0 [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    const float dh = d_h_pre[gid];
    if (use_v0 != 0u) {
        const float hr = h_raw[gid];
        const float vu = v0_up[gid];
        const float l0 = vr_lambda[0];
        const float l1 = vr_lambda[1];
        d_h_raw[gid] = l1 * dh;
        d_v0_up[gid] = l0 * dh;
        atomic_fetch_add_explicit(&d_vr_lambda[0], dh * vu, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_vr_lambda[1], dh * hr, memory_order_relaxed);
    } else {
        d_h_raw[gid] = dh;
        d_v0_up[gid] = 0.0f;
    }
}
