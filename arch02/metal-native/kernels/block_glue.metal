// Block glue: resid_mix, ln_scale prep, residual adds, U-net skip, MLP act.
#include <metal_stdlib>
using namespace metal;

/// x_in = m0 * x + m1 * x0. One thread per (b,t).
kernel void resid_mix_f32(
    device const float *x [[buffer(0)]],
    device const float *x0 [[buffer(1)]],
    device const float *mix [[buffer(2)]], // [2, C]
    device float *out [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &C [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        out[base + c] = mix[c] * x[base + c] + mix[C + c] * x0[base + c];
    }
}

/// out = x * scale (scalar). For ln_scale_factor after RMSNorm.
kernel void scale_f32(
    device float *x [[buffer(0)]],
    constant float &scale [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] *= scale;
}

/// out = x + scale_vec * branch. One thread per row (no in-place copy-add).
kernel void residual_scale_add_f32(
    device const float *x [[buffer(0)]],
    device const float *branch [[buffer(1)]],
    device const float *scale [[buffer(2)]], // [C]
    device float *out [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &C [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        out[base + c] = x[base + c] + scale[c] * branch[base + c];
    }
}

/// out = x + skip_w * skip. One thread per row (no in-place copy-add).
kernel void skip_add_f32(
    device const float *x [[buffer(0)]],
    device const float *skip [[buffer(1)]],
    device const float *skip_w [[buffer(2)]], // [C]
    device float *out [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &C [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        out[base + c] = x[base + c] + skip_w[c] * skip[base + c];
    }
}

/// MLP activation: leaky_relu(x, 0.5) then square. Out-of-place.
kernel void mlp_act_sq_leaky_f32(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = x[gid];
    float a = (v >= 0.0f) ? v : (0.5f * v);
    y[gid] = a * a;
}

/// Audit 9C: same act math, write bf16 — kills the post-act `cast_f32_to_bf16`
/// before the bf16 down-proj GEMM (f32 pre-act stays on tape for bwd).
/// Speed: REJECT as KEEP (<5% step); kept as dispatch cleanup (same math).
kernel void mlp_act_sq_leaky_f32_to_bf16(
    device const float *x [[buffer(0)]],
    device bfloat *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = x[gid];
    float a = (v >= 0.0f) ? v : (0.5f * v);
    y[gid] = bfloat(a * a);
}

// Note: zero_f32, copy_f32, copy_bf16, add_inplace_f32, transpose2d_f32,
// cast_f32_to_bf16, cast_bf16_to_f32 and softcap_f32 are defined once, in
// tessl/kernels/utils.metal, and linked into this metallib by build.rs.
// They were byte-identical duplicates here; two definitions of one kernel
// is a metallib link error waiting to happen, not a redundancy.
