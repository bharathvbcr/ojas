// Throwaway experiment (scratch tree only): dense fp16-weight GEMV, y = W x.
// W is [rows, cols] fp16 row-major; x is bf16 (x_bf16 != 0) or f32. 64 threads per
// threadgroup, two simdgroups of four rows; each lane reads eight consecutive weights
// (16 bytes) per row per step, so a simdgroup reads 512 contiguous bytes of a row.
#include <metal_stdlib>
using namespace metal;

kernel void gemv_f16w(
    device const half *w [[buffer(0)]],
    device const void *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &x_bf16 [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = (tgid * 2u + sgid) * 4u;
    if (row0 >= rows) return;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = lane * 8u; k + 8u <= cols; k += 256u) {
        float4 x0, x1;
        if (x_bf16 != 0u) {
            device const bfloat4 *xb = (device const bfloat4 *)((device const bfloat *)x + k);
            x0 = float4(xb[0]);
            x1 = float4(xb[1]);
        } else {
            device const float4 *xf = (device const float4 *)((device const float *)x + k);
            x0 = xf[0];
            x1 = xf[1];
        }
        for (uint r = 0u; r < 4u; ++r) {
            if (row0 + r >= rows) break;
            device const half4 *wp = (device const half4 *)(w + (ulong)(row0 + r) * cols + k);
            acc[r] += dot(float4(wp[0]), x0) + dot(float4(wp[1]), x1);
        }
    }
    for (uint r = 0u; r < 4u; ++r) {
        const float s = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < rows) y[row0 + r] = s;
    }
}
