// Phase polish: low-risk elementwise chain fusions (crush dispatch count).
// Per-op kernels in block_glue.metal / phase4_bf16.metal remain as fallback.
#include <metal_stdlib>
using namespace metal;

/// resid_mix then RMSNorm * scale.
/// Writes mix_out (= x_in for tape/bwd) and norm_out (= attn_in).
/// One thread per row.
kernel void resid_mix_rms_norm_scale_f32(
    device const float *x [[buffer(0)]],
    device const float *x0 [[buffer(1)]],
    device const float *mix [[buffer(2)]], // [2, C]
    device float *mix_out [[buffer(3)]],
    device float *norm_out [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = mix[c] * x[base + c] + mix[C + c] * x0[base + c];
        mix_out[base + c] = v;
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * scale;
    for (uint c = 0; c < C; ++c) {
        norm_out[base + c] = mix_out[base + c] * inv;
    }
}

/// bf16 residual-stream twin: f32 mix_out (tape) + bf16 norm_out (GEMM input).
kernel void resid_mix_rms_norm_scale_bf16(
    device const float *x [[buffer(0)]],
    device const float *x0 [[buffer(1)]],
    device const float *mix [[buffer(2)]],
    device float *mix_out [[buffer(3)]],
    device bfloat *norm_out [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = mix[c] * x[base + c] + mix[C + c] * x0[base + c];
        mix_out[base + c] = v;
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * scale;
    for (uint c = 0; c < C; ++c) {
        norm_out[base + c] = bfloat(mix_out[base + c] * inv);
    }
}

/// residual_scale_add then RMSNorm * scale.
/// Writes mid_out (= x after attn residual) and norm_out (= mlp_in).
/// One thread per row.
kernel void residual_scale_add_rms_norm_scale_f32(
    device const float *x [[buffer(0)]],
    device const float *branch [[buffer(1)]],
    device const float *res_scale [[buffer(2)]], // [C]
    device float *mid_out [[buffer(3)]],
    device float *norm_out [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant float &ln_scale [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x[base + c] + res_scale[c] * branch[base + c];
        mid_out[base + c] = v;
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * ln_scale;
    for (uint c = 0; c < C; ++c) {
        norm_out[base + c] = mid_out[base + c] * inv;
    }
}

/// bf16 twin: f32 mid_out + bf16 norm_out.
kernel void residual_scale_add_rms_norm_scale_bf16(
    device const float *x [[buffer(0)]],
    device const float *branch [[buffer(1)]],
    device const float *res_scale [[buffer(2)]],
    device float *mid_out [[buffer(3)]],
    device bfloat *norm_out [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    constant float &ln_scale [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x[base + c] + res_scale[c] * branch[base + c];
        mid_out[base + c] = v;
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * ln_scale;
    for (uint c = 0; c < C; ++c) {
        norm_out[base + c] = bfloat(mid_out[base + c] * inv);
    }
}

/// Decoder: skip_add then resid_mix then RMSNorm*scale (3→1 dispatch).
kernel void skip_resid_mix_rms_norm_scale_f32(
    device const float *x [[buffer(0)]],
    device const float *skip [[buffer(1)]],
    device const float *skip_w [[buffer(2)]], // [C]
    device const float *x0 [[buffer(3)]],
    device const float *mix [[buffer(4)]], // [2, C]
    device float *after_skip [[buffer(5)]],
    device float *mix_out [[buffer(6)]],
    device float *norm_out [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &C [[buffer(9)]],
    constant float &eps [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float xs = x[base + c] + skip_w[c] * skip[base + c];
        after_skip[base + c] = xs;
        float v = mix[c] * xs + mix[C + c] * x0[base + c];
        mix_out[base + c] = v;
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * scale;
    for (uint c = 0; c < C; ++c) {
        norm_out[base + c] = mix_out[base + c] * inv;
    }
}

/// bf16 twin: f32 after_skip + mix_out (tape) + bf16 norm_out (GEMM input).
kernel void skip_resid_mix_rms_norm_scale_bf16(
    device const float *x [[buffer(0)]],
    device const float *skip [[buffer(1)]],
    device const float *skip_w [[buffer(2)]],
    device const float *x0 [[buffer(3)]],
    device const float *mix [[buffer(4)]],
    device float *after_skip [[buffer(5)]],
    device float *mix_out [[buffer(6)]],
    device bfloat *norm_out [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &C [[buffer(9)]],
    constant float &eps [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float xs = x[base + c] + skip_w[c] * skip[base + c];
        after_skip[base + c] = xs;
        float v = mix[c] * xs + mix[C + c] * x0[base + c];
        mix_out[base + c] = v;
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * scale;
    for (uint c = 0; c < C; ++c) {
        norm_out[base + c] = bfloat(mix_out[base + c] * inv);
    }
}

/// Stem: per-row RMSNorm then causal smear. One thread per (b,t) row — smear
/// reads the prior token's normalized row recomputed from `x` (no cross-thread
/// dependency on `post_norm` writes).
kernel void rms_norm_smear_f32(
    device const float *x [[buffer(0)]],
    device const float *gate [[buffer(1)]], // [C]
    device float *post_norm [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &T [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const uint BT = B * T;
    if (gid >= BT) return;
    const uint t = gid % T;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x[base + c];
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps);
    for (uint c = 0; c < C; ++c) {
        post_norm[base + c] = x[base + c] * inv;
    }
    float pinv = 0.0f;
    const uint pbase = (t > 0) ? (gid - 1) * C : 0;
    if (t > 0) {
        float pss = 0.0f;
        for (uint cc = 0; cc < C; ++cc) {
            float pv = x[pbase + cc];
            pss += pv * pv;
        }
        pinv = rsqrt(pss / float(C) + eps);
    }
    for (uint c = 0; c < C; ++c) {
        float n = post_norm[base + c];
        float g = 1.0f / (1.0f + exp(-gate[c]));
        float prev = (t == 0) ? 0.0f : x[pbase + c] * pinv;
        out[base + c] = (1.0f - g) * n + g * prev;
    }
}
