// Audit 8: row-block reduction for the residual/RMS backward megakernels.
//
// Problem (measured 2026-07-20): `resid_glue` was 62 ms of the synced backward
// while moving only 3.93 GB = **63 GB/s** on a 400-800 GB/s part — 8x off
// bandwidth, so it is not bandwidth-bound. The generic kernels are
// one-thread-per-row and issue `atomic_fetch_add(&dscale[c], ...)` *inside* the
// C loop:
//
//     rows x C = 4096 x 768 = 3.1M device atomics per call
//     (the resid_mix twin does two per element -> 6.2M)
//     x 24 layers x 2 kernels ~= 150M atomics/step, on 768 addresses.
//
// Fix: split the work. The `_noatom` kernels are byte-for-byte the originals
// with the atomic removed — every elementwise output is unchanged, and the
// value being reduced (`d`) is already materialized in `dx_in`. A second pass
// then reduces over rows.
//
// The reduction assigns each thread one (channel, row_block) pair:
//   c  = gid % C          -> consecutive threads read consecutive addresses,
//                            so each row access is fully coalesced
//   rb = gid / C          -> strided row walk, ROW_BLOCKS-way parallel
// and issues **one** atomic per (c, rb): C x ROW_BLOCKS = 768 x 32 = 24576
// atomics instead of 3.1M — a **128x** cut — with no threadgroup memory, no
// barriers, and no scratch buffer.
//
// Numerics: the sum order changes (row-block partials, then atomic combine), so
// this is *not* bit-identical. f32 sums of the same values — expect ~1e-6
// relative drift. Gate at 1e-5, not equality.
#include <metal_stdlib>
using namespace metal;

/// `residual_scale_add_rms_norm_scale_bwd_f32` with the dscale atomic removed.
/// Buffer layout is identical (buffer 7 is left untouched) so the host can swap
/// kernels without re-binding.
kernel void residual_scale_add_rms_norm_scale_bwd_noatom_f32(
    device const float *x_mid [[buffer(0)]],
    device const float *d_mlp_in [[buffer(1)]],
    device const float *branch [[buffer(2)]],
    device const float *scale [[buffer(3)]],
    device float *dx_mid [[buffer(4)]],
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
        // dscale reduced by reduce_dscale_rowblock_f32 (d is now in dx_in).
    }
}

/// `resid_mix_rms_norm_scale_bwd_f32` with both dmix atomics removed.
kernel void resid_mix_rms_norm_scale_bwd_noatom_f32(
    device const float *x_in [[buffer(0)]],
    device const float *d_attn_in [[buffer(1)]],
    device const float *x_stream [[buffer(2)]],
    device const float *x0 [[buffer(3)]],
    device const float *mix [[buffer(4)]],
    device float *dx_in [[buffer(5)]],
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
        // dmix reduced by reduce_dmix_rowblock_f32 (d is now in dx_in).
    }
}

/// dscale[c] += sum_r dx_in[r,c] * branch[r,c], one atomic per (c, row_block).
/// Grid: C * row_blocks threads.
kernel void reduce_dscale_rowblock_f32(
    device const float *dx_in [[buffer(0)]],
    device const float *branch [[buffer(1)]],
    device atomic_float *dscale [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &C [[buffer(4)]],
    constant uint &row_blocks [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= C * row_blocks) return;
    const uint c = gid % C;
    const uint rb = gid / C;
    float acc = 0.0f;
    for (uint r = rb; r < rows; r += row_blocks) {
        const uint idx = r * C + c;
        acc += dx_in[idx] * branch[idx];
    }
    if (acc != 0.0f) {
        atomic_fetch_add_explicit(&dscale[c], acc, memory_order_relaxed);
    }
}

/// dmix[c]   += sum_r dx_in[r,c] * x_stream[r,c]
/// dmix[C+c] += sum_r dx_in[r,c] * x0[r,c]
/// Two atomics per (c, row_block). Grid: C * row_blocks threads.
kernel void reduce_dmix_rowblock_f32(
    device const float *dx_in [[buffer(0)]],
    device const float *x_stream [[buffer(1)]],
    device const float *x0 [[buffer(2)]],
    device atomic_float *dmix [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &C [[buffer(5)]],
    constant uint &row_blocks [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= C * row_blocks) return;
    const uint c = gid % C;
    const uint rb = gid / C;
    float acc_s = 0.0f;
    float acc_0 = 0.0f;
    for (uint r = rb; r < rows; r += row_blocks) {
        const uint idx = r * C + c;
        const float d = dx_in[idx];
        acc_s += d * x_stream[idx];
        acc_0 += d * x0[idx];
    }
    if (acc_s != 0.0f) {
        atomic_fetch_add_explicit(&dmix[c], acc_s, memory_order_relaxed);
    }
    if (acc_0 != 0.0f) {
        atomic_fetch_add_explicit(&dmix[C + c], acc_0, memory_order_relaxed);
    }
}
