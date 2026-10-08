// Elementwise / glue backward kernels.
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_bwd_f32(
    device const float *x [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dx [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &C [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float mean_sq = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x[base + c];
        mean_sq += v * v;
    }
    mean_sq /= (float)C;
    float inv = rsqrt(mean_sq + eps);
    float inv3 = inv * inv * inv;
    float dot = 0.0f;
    for (uint c = 0; c < C; ++c) {
        dot += x[base + c] * dy[base + c];
    }
    float coeff = inv3 / (float)C * dot;
    for (uint c = 0; c < C; ++c) {
        dx[base + c] = inv * dy[base + c] - coeff * x[base + c];
    }
}

kernel void resid_mix_bwd_simple_f32(
    device const float *x [[buffer(0)]],
    device const float *x0 [[buffer(1)]],
    device const float *mix [[buffer(2)]],
    device const float *dout [[buffer(3)]],
    device float *dx [[buffer(4)]],
    device float *dx0 [[buffer(5)]],
    device atomic_float *dmix [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &C [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        float d = dout[base + c];
        dx[base + c] = mix[c] * d;
        dx0[base + c] += mix[C + c] * d;
        atomic_fetch_add_explicit(&dmix[c], d * x[base + c], memory_order_relaxed);
        atomic_fetch_add_explicit(&dmix[C + c], d * x0[base + c], memory_order_relaxed);
    }
}

kernel void residual_scale_add_bwd_f32(
    device const float *branch [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    device const float *dout [[buffer(2)]],
    device float *dx [[buffer(3)]],       // = dout (copy)
    device float *dbranch [[buffer(4)]],
    device atomic_float *dscale [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &C [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        float d = dout[base + c];
        dx[base + c] = d;
        dbranch[base + c] = scale[c] * d;
        atomic_fetch_add_explicit(&dscale[c], d * branch[base + c], memory_order_relaxed);
    }
}

kernel void skip_add_bwd_f32(
    device const float *skip [[buffer(0)]],
    device const float *skip_w [[buffer(1)]],
    device const float *dout [[buffer(2)]],
    device float *dx [[buffer(3)]],
    device float *dskip [[buffer(4)]],
    device atomic_float *dskip_w [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &C [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        float d = dout[base + c];
        dx[base + c] = d;
        dskip[base + c] = skip_w[c] * d;
        atomic_fetch_add_explicit(&dskip_w[c], d * skip[base + c], memory_order_relaxed);
    }
}

kernel void mlp_act_bwd_f32(
    device const float *pre [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dx [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float x = pre[gid];
    // y = a^2, a = leaky(x,0.5); dy/dx = 2x (x>=0) else 0.5*x
    float g = (x >= 0.0f) ? (2.0f * x) : (0.5f * x);
    dx[gid] = dy[gid] * g;
}

kernel void scale_bwd_f32(
    device const float *dy [[buffer(0)]],
    device float *dx [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dx[gid] = dy[gid] * scale;
}

/// Fuse `scale_bwd` (dy *= ln_scale) + `rms_norm_bwd` into one pass.
/// Optionally accumulates into `dx_accum` when `accum != 0` (saves a later add).
kernel void rms_norm_scale_bwd_f32(
    device const float *x [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dx [[buffer(2)]],
    device float *dx_accum [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &C [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    constant float &ln_scale [[buffer(7)]],
    constant uint &accum [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float mean_sq = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x[base + c];
        mean_sq += v * v;
    }
    mean_sq /= (float)C;
    float inv = rsqrt(mean_sq + eps);
    float inv3 = inv * inv * inv;
    float dot = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float dys = dy[base + c] * ln_scale;
        dot += x[base + c] * dys;
    }
    float coeff = inv3 / (float)C * dot;
    for (uint c = 0; c < C; ++c) {
        float dys = dy[base + c] * ln_scale;
        float dxi = inv * dys - coeff * x[base + c];
        if (accum) {
            dx_accum[base + c] += dxi;
        } else {
            dx[base + c] = dxi;
        }
    }
}
kernel void copy_scale_f32(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = src[gid] * scale;
}
/// Fuse rms_norm_scale_bwd (accum into residual dx) + residual_scale_add_bwd.
/// `dx_mid` arrives with the MLP-residual contribution; this adds d(mlp_in) via
/// RMS and then splits into dx_in + d_attn_out (+ dscale).
kernel void residual_scale_add_rms_norm_scale_bwd_f32(
    device const float *x_mid [[buffer(0)]],
    device const float *d_mlp_in [[buffer(1)]],
    device const float *branch [[buffer(2)]],      // attn_out
    device const float *scale [[buffer(3)]],       // attn_scale [C]
    device float *dx_mid [[buffer(4)]],            // in: mlp resid contrib; out unused
    device float *dx_in [[buffer(5)]],
    device float *d_branch [[buffer(6)]],
    device atomic_float *dscale [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &C [[buffer(9)]],
    constant float &eps [[buffer(10)]],
    constant float &ln_scale [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float mean_sq = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x_mid[base + c];
        mean_sq += v * v;
    }
    mean_sq /= (float)C;
    float inv = rsqrt(mean_sq + eps);
    float inv3 = inv * inv * inv;
    float dot = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float dys = d_mlp_in[base + c] * ln_scale;
        dot += x_mid[base + c] * dys;
    }
    float coeff = inv3 / (float)C * dot;
    for (uint c = 0; c < C; ++c) {
        float dys = d_mlp_in[base + c] * ln_scale;
        float dxi = inv * dys - coeff * x_mid[base + c];
        float d = dx_mid[base + c] + dxi;
        dx_in[base + c] = d;
        d_branch[base + c] = scale[c] * d;
        atomic_fetch_add_explicit(&dscale[c], d * branch[base + c], memory_order_relaxed);
    }
}

/// Fuse rms_norm_scale_bwd (accum into dx_in) + resid_mix_bwd_simple.
kernel void resid_mix_rms_norm_scale_bwd_f32(
    device const float *x_in [[buffer(0)]],
    device const float *d_attn_in [[buffer(1)]],
    device const float *x_stream [[buffer(2)]],
    device const float *x0 [[buffer(3)]],
    device const float *mix [[buffer(4)]],
    device float *dx_in [[buffer(5)]],             // in: residual contrib; out: total d(x_in)
    device float *dx_stream [[buffer(6)]],
    device float *dx0 [[buffer(7)]],
    device atomic_float *dmix [[buffer(8)]],
    constant uint &rows [[buffer(9)]],
    constant uint &C [[buffer(10)]],
    constant float &eps [[buffer(11)]],
    constant float &ln_scale [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float mean_sq = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x_in[base + c];
        mean_sq += v * v;
    }
    mean_sq /= (float)C;
    float inv = rsqrt(mean_sq + eps);
    float inv3 = inv * inv * inv;
    float dot = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float dys = d_attn_in[base + c] * ln_scale;
        dot += x_in[base + c] * dys;
    }
    float coeff = inv3 / (float)C * dot;
    for (uint c = 0; c < C; ++c) {
        float dys = d_attn_in[base + c] * ln_scale;
        float dxi = inv * dys - coeff * x_in[base + c];
        float d = dx_in[base + c] + dxi;
        dx_in[base + c] = d;
        dx_stream[base + c] = mix[c] * d;
        dx0[base + c] += mix[C + c] * d;
        atomic_fetch_add_explicit(&dmix[c], d * x_stream[base + c], memory_order_relaxed);
        atomic_fetch_add_explicit(&dmix[C + c], d * x0[base + c], memory_order_relaxed);
    }
}

// Note: zero_f32, copy_f32, copy_bf16, add_inplace_f32, transpose2d_f32,
// cast_f32_to_bf16, cast_bf16_to_f32 and softcap_f32 are defined once, in
// tessl/kernels/utils.metal, and linked into this metallib by build.rs.
// They were byte-identical duplicates here; two definitions of one kernel
// is a metallib link error waiting to happen, not a redundancy.
