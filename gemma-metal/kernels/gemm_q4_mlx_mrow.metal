// Thin Q4 (MLX affine, row-major) GEMM for a verify block: y[m * rows + row] = X[m] . W[row]
// for M = 2..5 activation rows, one entry per M (the host splits 6..8 into two dispatches: at
// M >= 6 one entry is no faster than tessl's, which we read as its x tile and accumulators
// spilling; inferred from timing, not from the compiler's register report).
//
// tessl's `q4_simd_gemm` (gemm_q4_mlx_simd.h) carries a runtime M: its x tile and
// accumulators are sized for GEMM_MAX_M = 8 and indexed by a runtime loop, so they do not
// stay in registers, and `qdot16` re-unpacks the same sixteen nibbles for every row of X.
// On E4B shapes that made M = 2 cost 1.6x M = 1 and M = 4 cost 3.2x (README, verify(M)).
// Here M is a template parameter, so every array is unrolled into registers, and each lane
// unpacks its sixteen nibbles once per weight row and reuses them for all M rows of X.
//
// Same geometry as the tessl kernel (SIMD_ROWS rows per simdgroup, SIMD_VPT nibbles per
// lane, simd_sum over lanes) and the same per-row arithmetic: `load_x16_qdot`'s prescaled x
// and `qdot16`'s products in `qdot16`'s order, `scale * accum + xsum * bias` per block.
#include <metal_stdlib>
#include "q4_mlx_dot.h"
using namespace metal;

template <uint M>
inline void q4_gemm_mrow(
    device const uchar *packed,
    device const bfloat2 *sb,
    device const bfloat *x,
    device float *y,
    uint rows,
    uint cols,
    uint group_size,
    uint tgid,
    uint sgid,
    uint lane)
{
    const uint row0 = (tgid * SIMD_SG_PER_TG + sgid) * SIMD_ROWS;
    if (row0 >= rows) return;
    const uint gpr = cols / group_size;
    const uint row_bytes = cols >> 1;
    const uint lane_col0 = lane * SIMD_VPT;
    float acc[SIMD_ROWS][M];
    for (uint r = 0u; r < SIMD_ROWS; ++r)
        for (uint m = 0u; m < M; ++m) acc[r][m] = 0.0f;
    for (ulong k0 = 0ul; k0 < (ulong)cols; k0 += SIMD_BLOCK) {
        const ulong col = k0 + lane_col0;
        if (col + SIMD_VPT > (ulong)cols) continue;
        float xt[M][16];
        float xsum[M];
        for (uint m = 0u; m < M; ++m)
            xsum[m] = load_x16_qdot(x + (ulong)m * cols + col, xt[m]);
        const ulong g = col / group_size;
        const ulong byte_off = col >> 1;
        for (uint r = 0u; r < SIMD_ROWS; ++r) {
            const uint row = row0 + r;
            if (row >= rows) break;
            const bfloat2 sbv = sb[(ulong)row * gpr + g];
            const float scale = float(sbv.x);
            const float bias = float(sbv.y);
            const uint2 w2 =
                ((device const uint2 *)(packed + (ulong)row * row_bytes + byte_off))[0];
            const uint ww0 = w2.x & 0xffffu;
            const uint ww1 = w2.x >> 16;
            const uint ww2 = w2.y & 0xffffu;
            const uint ww3 = w2.y >> 16;
            const float w[16] = {
                float(ww0 & 0x000fu), float(ww0 & 0x00f0u), float(ww0 & 0x0f00u), float(ww0 & 0xf000u),
                float(ww1 & 0x000fu), float(ww1 & 0x00f0u), float(ww1 & 0x0f00u), float(ww1 & 0xf000u),
                float(ww2 & 0x000fu), float(ww2 & 0x00f0u), float(ww2 & 0x0f00u), float(ww2 & 0xf000u),
                float(ww3 & 0x000fu), float(ww3 & 0x00f0u), float(ww3 & 0x0f00u), float(ww3 & 0xf000u),
            };
            for (uint m = 0u; m < M; ++m) {
                thread const float *xp = xt[m];
                float accum = 0.0f;
                accum += xp[0] * w[0] + xp[1] * w[1] + xp[2] * w[2] + xp[3] * w[3];
                accum += xp[4] * w[4] + xp[5] * w[5] + xp[6] * w[6] + xp[7] * w[7];
                accum += xp[8] * w[8] + xp[9] * w[9] + xp[10] * w[10] + xp[11] * w[11];
                accum += xp[12] * w[12] + xp[13] * w[13] + xp[14] * w[14] + xp[15] * w[15];
                acc[r][m] += scale * accum + xsum[m] * bias;
            }
        }
    }
    for (uint r = 0u; r < SIMD_ROWS; ++r) {
        const uint row = row0 + r;
        for (uint m = 0u; m < M; ++m) {
            const float sum = simd_sum(acc[r][m]);
            if (lane == 0u && row < rows) y[(ulong)m * rows + row] = sum;
        }
    }
}

#define GEMM_Q4_MROW_ENTRY(MM)                                                          \
    kernel void gemm_q4_mlx_m##MM(                                                      \
        device const uchar *packed [[buffer(0)]],                                       \
        device const bfloat2 *sb [[buffer(1)]],                                         \
        device const bfloat *biases_unused [[buffer(2)]],                               \
        device const bfloat *x [[buffer(3)]],                                           \
        device float *y [[buffer(4)]],                                                  \
        constant uint &rows [[buffer(5)]],                                              \
        constant uint &cols [[buffer(6)]],                                              \
        constant uint &group_size [[buffer(7)]],                                        \
        uint tgid [[threadgroup_position_in_grid]],                                     \
        uint sgid [[simdgroup_index_in_threadgroup]],                                   \
        uint lane [[thread_index_in_simdgroup]])                                        \
    {                                                                                   \
        (void)biases_unused;                                                            \
        q4_gemm_mrow<MM>(packed, sb, x, y, rows, cols, group_size, tgid, sgid, lane);   \
    }

GEMM_Q4_MROW_ENTRY(2)
GEMM_Q4_MROW_ENTRY(3)
GEMM_Q4_MROW_ENTRY(4)
GEMM_Q4_MROW_ENTRY(5)
