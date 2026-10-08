// DFlash verify thin Q4 GEMM (M≤8). Standalone so gemv_q4_mlx.metal edits do not drop it.
#include <metal_stdlib>
using namespace metal;

constant uint SIMD_SIZE = 32u;
constant uint SIMD_ROWS = 4u;
constant uint SIMD_SG_PER_TG = 2u;
constant uint SIMD_PACKS = 2u;
constant uint SIMD_VPT = 8u * SIMD_PACKS;     // 16
constant uint SIMD_BLOCK = SIMD_SIZE * SIMD_VPT; // 512
constant uint GEMM_MAX_M = 8u;

inline float load_x16_qdot(device const bfloat *x, thread float *xp)
{
    bfloat4 x0 = ((device const bfloat4 *)(x))[0];
    bfloat4 x1 = ((device const bfloat4 *)(x + 4u))[0];
    bfloat4 x2 = ((device const bfloat4 *)(x + 8u))[0];
    bfloat4 x3 = ((device const bfloat4 *)(x + 12u))[0];
    float a0 = float(x0.x), a1 = float(x0.y), a2 = float(x0.z), a3 = float(x0.w);
    float a4 = float(x1.x), a5 = float(x1.y), a6 = float(x1.z), a7 = float(x1.w);
    float a8 = float(x2.x), a9 = float(x2.y), a10 = float(x2.z), a11 = float(x2.w);
    float a12 = float(x3.x), a13 = float(x3.y), a14 = float(x3.z), a15 = float(x3.w);
    float sum = (a0 + a1 + a2 + a3) + (a4 + a5 + a6 + a7)
              + (a8 + a9 + a10 + a11) + (a12 + a13 + a14 + a15);
    xp[0] = a0;             xp[1] = a1 / 16.0f;   xp[2] = a2 / 256.0f;  xp[3] = a3 / 4096.0f;
    xp[4] = a4;             xp[5] = a5 / 16.0f;   xp[6] = a6 / 256.0f;  xp[7] = a7 / 4096.0f;
    xp[8] = a8;             xp[9] = a9 / 16.0f;   xp[10] = a10 / 256.0f; xp[11] = a11 / 4096.0f;
    xp[12] = a12;           xp[13] = a13 / 16.0f; xp[14] = a14 / 256.0f; xp[15] = a15 / 4096.0f;
    return sum;
}

inline float qdot16(
    device const uchar *w,
    thread const float *xp,
    float scale,
    float bias,
    float xsum)
{
    device const ushort *ws = (device const ushort *)w;
    float accum = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        const ushort ww = ws[i];
        accum += xp[4u * i] * float(ww & 0x000fu)
               + xp[4u * i + 1u] * float(ww & 0x00f0u)
               + xp[4u * i + 2u] * float(ww & 0x0f00u)
               + xp[4u * i + 3u] * float(ww & 0xf000u);
    }
    return scale * accum + xsum * bias;
}

kernel void gemm_q4_mlx_simd(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint gpr = cols / group_size;
    const uint row_bytes = cols >> 1;
    const uint lane_col0 = lane * SIMD_VPT;
    float acc[SIMD_ROWS][GEMM_MAX_M];
    for (uint r = 0u; r < SIMD_ROWS; ++r)
        for (uint m = 0u; m < GEMM_MAX_M; ++m) acc[r][m] = 0.0f;
    for (uint k0 = 0u; k0 < cols; k0 += SIMD_BLOCK) {
        const uint col = k0 + lane_col0;
        if (col + SIMD_VPT <= cols) {
            float xt[GEMM_MAX_M][16];
            float xsum[GEMM_MAX_M];
            for (uint m = 0u; m < m_cap; ++m)
                xsum[m] = load_x16_qdot(x + m * cols + col, xt[m]);
            const uint g = col / group_size;
            const uint byte_off = col >> 1;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                const uint row = row0 + r;
                if (row >= rows) break;
                const bfloat2 sbv = sb[row * gpr + g];
                device const uchar *wp = packed + row * row_bytes + byte_off;
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows) y[m * rows + row] = sum;
        }
    }
}

kernel void gemm_q4_mlx_simd_add(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    device const float *resid [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint gpr = cols / group_size;
    const uint row_bytes = cols >> 1;
    const uint lane_col0 = lane * SIMD_VPT;
    float acc[SIMD_ROWS][GEMM_MAX_M];
    for (uint r = 0u; r < SIMD_ROWS; ++r)
        for (uint m = 0u; m < GEMM_MAX_M; ++m) acc[r][m] = 0.0f;
    for (uint k0 = 0u; k0 < cols; k0 += SIMD_BLOCK) {
        const uint col = k0 + lane_col0;
        if (col + SIMD_VPT <= cols) {
            float xt[GEMM_MAX_M][16];
            float xsum[GEMM_MAX_M];
            for (uint m = 0u; m < m_cap; ++m)
                xsum[m] = load_x16_qdot(x + m * cols + col, xt[m]);
            const uint g = col / group_size;
            const uint byte_off = col >> 1;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                const uint row = row0 + r;
                if (row >= rows) break;
                const bfloat2 sbv = sb[row * gpr + g];
                device const uchar *wp = packed + row * row_bytes + byte_off;
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows)
                y[m * rows + row] = sum + resid[m * rows + row];
        }
    }
}

kernel void gemm_q4_mlx_simd_i4(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint tile = row0 / SIMD_ROWS;
    const uint gpr = cols / group_size;
    const uint packs_u2 = cols >> 4;
    const uint lane_col0 = lane * SIMD_VPT;
    float acc[SIMD_ROWS][GEMM_MAX_M];
    for (uint r = 0u; r < SIMD_ROWS; ++r)
        for (uint m = 0u; m < GEMM_MAX_M; ++m) acc[r][m] = 0.0f;
    for (uint k0 = 0u; k0 < cols; k0 += SIMD_BLOCK) {
        const uint col = k0 + lane_col0;
        if (col + SIMD_VPT <= cols) {
            float xt[GEMM_MAX_M][16];
            float xsum[GEMM_MAX_M];
            for (uint m = 0u; m < m_cap; ++m)
                xsum[m] = load_x16_qdot(x + m * cols + col, xt[m]);
            const uint g = col / group_size;
            const uint pack2 = col >> 4;
            device const uchar *wp0 = packed + ((tile * packs_u2 + pack2) * SIMD_ROWS) * 8u;
            const uint sb0 = (tile * gpr + g) * SIMD_ROWS;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                if (row0 + r >= rows) break;
                const bfloat2 sbv = sb[sb0 + r];
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp0 + r * 8u, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows) y[m * rows + row] = sum;
        }
    }
}

kernel void gemm_q4_mlx_simd_add_i4(
    device const uchar *packed [[buffer(0)]],
    device const bfloat2 *sb [[buffer(1)]],
    device const bfloat *biases_unused [[buffer(2)]],
    device const bfloat *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    constant uint &M [[buffer(8)]],
    device const float *resid [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    (void)biases_unused;
    const uint m_cap = min(M, GEMM_MAX_M);
    if (m_cap == 0u) return;
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint tile = row0 / SIMD_ROWS;
    const uint gpr = cols / group_size;
    const uint packs_u2 = cols >> 4;
    const uint lane_col0 = lane * SIMD_VPT;
    float acc[SIMD_ROWS][GEMM_MAX_M];
    for (uint r = 0u; r < SIMD_ROWS; ++r)
        for (uint m = 0u; m < GEMM_MAX_M; ++m) acc[r][m] = 0.0f;
    for (uint k0 = 0u; k0 < cols; k0 += SIMD_BLOCK) {
        const uint col = k0 + lane_col0;
        if (col + SIMD_VPT <= cols) {
            float xt[GEMM_MAX_M][16];
            float xsum[GEMM_MAX_M];
            for (uint m = 0u; m < m_cap; ++m)
                xsum[m] = load_x16_qdot(x + m * cols + col, xt[m]);
            const uint g = col / group_size;
            const uint pack2 = col >> 4;
            device const uchar *wp0 = packed + ((tile * packs_u2 + pack2) * SIMD_ROWS) * 8u;
            const uint sb0 = (tile * gpr + g) * SIMD_ROWS;
            for (uint r = 0u; r < SIMD_ROWS; ++r) {
                if (row0 + r >= rows) break;
                const bfloat2 sbv = sb[sb0 + r];
                for (uint m = 0u; m < m_cap; ++m)
                    acc[r][m] += qdot16(wp0 + r * 8u, xt[m], float(sbv.x), float(sbv.y), xsum[m]);
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < m_cap; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows)
                y[m * rows + row] = sum + resid[m * rows + row];
        }
    }
}
