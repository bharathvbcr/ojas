#include <metal_stdlib>
using namespace metal;

kernel void research_diff_sq_reduce_f32(
    device const float *before [[buffer(0)]], device const float *after [[buffer(1)]],
    device const float *grad [[buffer(2)]], device atomic_float *update_sq [[buffer(3)]],
    device atomic_float *grad_sq [[buffer(4)]], device atomic_float *nonfinite [[buffer(5)]],
    constant uint &n [[buffer(6)]], uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]])
{
    threadgroup float update_red[1024];
    threadgroup float grad_red[1024];
    threadgroup float bad_red[1024];
    float du = 0.0f, dg = 0.0f, bad = 0.0f;
    if (gid < n) {
        float a = after[gid], b = before[gid], g = grad[gid];
        float d = a - b;
        du = d * d;
        dg = g * g;
        bad = (isfinite(a) && isfinite(g)) ? 0.0f : 1.0f;
    }
    update_red[lid] = du; grad_red[lid] = dg; bad_red[lid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint i = lid * (stride << 1);
        if (i + stride < tpg) {
            update_red[i] += update_red[i + stride];
            grad_red[i] += grad_red[i + stride];
            bad_red[i] += bad_red[i + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        atomic_fetch_add_explicit(update_sq, update_red[0], memory_order_relaxed);
        atomic_fetch_add_explicit(grad_sq, grad_red[0], memory_order_relaxed);
        atomic_fetch_add_explicit(nonfinite, bad_red[0], memory_order_relaxed);
    }
}

kernel void research_matrix_drift_f32(
    device const float *before [[buffer(0)]], device const float *after [[buffer(1)]],
    device atomic_float *row_drift [[buffer(2)]],
    device atomic_float *spectral_proxy [[buffer(3)]],
    device atomic_float *orth_error [[buffer(4)]],
    constant uint &N [[buffer(5)]], constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]], constant float &eps [[buffer(8)]],
    uint mid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (mid >= N) return;
    uint base = mid * rows * cols;
    float row_local = 0.0f, orth_local = 0.0f, b2 = 0.0f, a2 = 0.0f;
    // Python [out,in] rows are native [in,out] columns.
    for (uint col = lid; col < cols; col += tpg) {
        float rb = 0.0f, ra = 0.0f;
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            rb += before[i] * before[i];
            ra += after[i] * after[i];
        }
        row_local += abs(log((sqrt(ra) + eps) / (sqrt(rb) + eps)));
        if (col + 1 < cols) {
            float dot = 0.0f, x2 = 0.0f, y2 = 0.0f;
            for (uint row = 0; row < rows; row++) {
                uint i = base + row * cols + col;
                float x = after[i] - before[i];
                float y = after[i + 1] - before[i + 1];
                dot += x * y; x2 += x * x; y2 += y * y;
            }
            orth_local += abs(dot) / (sqrt(x2 * y2) + eps);
        }
    }
    for (uint i = lid; i < rows * cols; i += tpg) {
        b2 += before[base + i] * before[base + i];
        a2 += after[base + i] * after[base + i];
    }
    threadgroup float rrow[1024], rorth[1024], rb[1024], ra[1024];
    rrow[lid] = row_local; rorth[lid] = orth_local; rb[lid] = b2; ra[lid] = a2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint i = lid * (stride << 1);
        if (i + stride < tpg) {
            rrow[i] += rrow[i + stride]; rorth[i] += rorth[i + stride];
            rb[i] += rb[i + stride]; ra[i] += ra[i + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        atomic_fetch_add_explicit(row_drift, rrow[0], memory_order_relaxed);
        atomic_fetch_add_explicit(orth_error, rorth[0], memory_order_relaxed);
        float drift = abs(log((sqrt(ra[0]) + eps) / (sqrt(rb[0]) + eps)));
        atomic_fetch_add_explicit(spectral_proxy, drift, memory_order_relaxed);
    }
}
