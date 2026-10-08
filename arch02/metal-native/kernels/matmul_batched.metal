// Bank-batched exact f32 TensorOps GEMMs for the arch_02 optimizer's matrix
// contractions (Newton-Schulz orthogonalisation in optim.rs). One flat grid
// covers [batch, output tiles], eliminating per-layer launches.
//
// These live here rather than in tessl because nothing outside arch_02's
// optimizer dispatches them, and tessl's metallib is loaded by every consumer.
// The shared GEMM kernels come from tessl/kernels/matmul_tensorops.metal, which
// this crate's build.rs compiles from tessl's source directory.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// Decode Morton/Z-order code -> (x, y) tile coordinates.
inline uint2 batched_morton_decode_2d(uint c) {
    uint x = 0, y = 0;
#pragma unroll
    for (uint i = 0; i < 16; ++i) {
        x |= ((c >> (2 * i)) & 1u) << i;
        y |= ((c >> (2 * i + 1)) & 1u) << i;
    }
    return uint2(x, y);
}

/// Mirrors `tile_from_linear` in tessl's matmul_tensorops.metal.
inline uint2 batched_tile_from_linear(uint linear, uint tiles_n, uint tiles_m) {
    if (tiles_n == tiles_m && tiles_n != 0u && (tiles_n & (tiles_n - 1u)) == 0u) {
        return batched_morton_decode_2d(linear);
    }
    return uint2(linear % tiles_n, linear / tiles_n);
}

kernel void matmul2d_tensorops_batched_f32(
    device float *A [[buffer(0)]], device float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],
    constant uint &batch [[buffer(8)]], uint flat_tg [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32, SN = 32;
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    uint tiles = tiles_n * tiles_m;
    uint mid = flat_tg / tiles;
    if (mid >= batch) return;
    uint2 tile = batched_tile_from_linear(flat_tg - mid * tiles, tiles_n, tiles_m);
    int tx = (int)tile.x * SN, ty = (int)tile.y * SM;
    device float *Ab = A + mid * M * K;
    device float *Bb = B + mid * K * N;
    device float *Cb = C + mid * M * N;
    auto mA = tensor(Ab, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto mB = tensor(Bb, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto mC = tensor(Cb, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
    auto tA = mA.slice(0, ty);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

kernel void matmul2d_tensorops_batched_tn_f32(
    device float *A [[buffer(0)]], device float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],
    constant uint &batch [[buffer(8)]], uint flat_tg [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32, SN = 32;
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, true, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    uint tiles = tiles_n * tiles_m;
    uint mid = flat_tg / tiles;
    if (mid >= batch) return;
    uint2 tile = batched_tile_from_linear(flat_tg - mid * tiles, tiles_n, tiles_m);
    int tx = (int)tile.x * SN, ty = (int)tile.y * SM;
    device float *Ab = A + mid * K * M; // physical [K,M]
    device float *Bb = B + mid * K * N;
    device float *Cb = C + mid * M * N;
    auto mA = tensor(Ab, dextents<int, 2>{(int)M, (int)K}, array<int, 2>{1, (int)M});
    auto mB = tensor(Bb, dextents<int, 2>{(int)N, (int)K}, array<int, 2>{1, (int)N});
    auto mC = tensor(Cb, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
    auto tA = mA.slice(ty, 0);
    auto tB = mB.slice(tx, 0);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

kernel void matmul2d_tensorops_batched_nt_f32(
    device float *A [[buffer(0)]], device float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &tiles_n [[buffer(6)]], constant uint &tiles_m [[buffer(7)]],
    constant uint &batch [[buffer(8)]], uint flat_tg [[threadgroup_position_in_grid]])
{
    constexpr int SM = 32, SN = 32;
    constexpr auto desc = matmul2d_descriptor(
        SM, SN, dynamic_length_v<int>, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    uint tiles = tiles_n * tiles_m;
    uint mid = flat_tg / tiles;
    if (mid >= batch) return;
    uint2 tile = batched_tile_from_linear(flat_tg - mid * tiles, tiles_n, tiles_m);
    int tx = (int)tile.x * SN, ty = (int)tile.y * SM;
    device float *Ab = A + mid * M * K;
    device float *Bb = B + mid * N * K; // physical [N,K]
    device float *Cb = C + mid * M * N;
    auto mA = tensor(Ab, dextents<int, 2>{(int)K, (int)M}, array<int, 2>{1, (int)K});
    auto mB = tensor(Bb, dextents<int, 2>{(int)K, (int)N}, array<int, 2>{1, (int)K});
    auto mC = tensor(Cb, dextents<int, 2>{(int)N, (int)M}, array<int, 2>{1, (int)N});
    auto tA = mA.slice(0, ty);
    auto tB = mB.slice(0, tx);
    auto tC = mC.slice(tx, ty);
    op.run(tA, tB, tC);
}

// =============================================================================
// f32 exact — execution_simdgroup, SM=SN=32 (golden-safe)
// =============================================================================

