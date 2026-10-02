// Per-head sigmoid output gate.
//
// The linear map pre = x @ W^T is a tessl GEMM. This file only applies
// sigmoid(pre + bias) to the attention output, and the matching backward.
// One thread owns each (row, head) on the backward so the head_dim sum is
// the same left-to-right f32 loop the CPU reference uses; the bias sum keeps
// the CPU's ascending row order too.
//
// Compiled with -fmetal-math-mode=safe and -ffp-contract=off.
#include <metal_stdlib>
using namespace metal;

// The backward kernels check what they read and write, into the op's status
// slot, with the words `ST_IN` and `ST_OUT` of `ojas-metal/src/device.rs` (and
// `ojas_backend.metal`): ST_IN when an input element a thread read is
// non-finite, ST_OUT when a value it wrote, or the `pre` the GEMM wrote, is.
// That is what the standalone `ojas_check_finite` passes over those buffers
// reported, without the extra reads (`metal_bench gate`).
#define GATE_ST_IN 0u
#define GATE_ST_OUT 1u

inline void ojas_gate_flag(device atomic_uint *st, uint word)
{
    atomic_store_explicit(&st[word], 1u, memory_order_relaxed);
}

inline float ojas_sigmoid(float x)
{
    const float e = precise::exp(-fabs(x));
    const float r = precise::divide(1.0f, 1.0f + e);
    return x >= 0.0f ? r : e * r;
}

/// out[r, h, d] = attn[r, h, d] * sigmoid(pre[r, h] + bias[h])
/// Grid: x = head dim, y = row * n_head + head.
/// Lengths are element counts. A lying length makes the thread return.
kernel void ojas_per_head_gate_fwd(
    device const float *attn [[buffer(0)]],
    device const float *pre [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &n_head [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &attn_len [[buffer(7)]],
    constant uint &pre_len [[buffer(8)]],
    constant uint &bias_len [[buffer(9)]],
    constant uint &out_len [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint rh = gid.y;
    const uint units = rows * n_head;
    if (d >= head_dim || rh >= units || n_head == 0u) return;
    const uint h = rh % n_head;
    const uint r = rh / n_head;
    const ulong pre_i = (ulong)r * n_head + h;
    const ulong i = pre_i * head_dim + d;
    if (i >= attn_len || i >= out_len || pre_i >= pre_len || h >= bias_len) return;
    const float z = pre[pre_i] + bias[h];
    const float g = ojas_sigmoid(z);
    out[i] = attn[i] * g;
}

/// d_attn = dy * g
/// d_pre = g * (1 - g) * sum_d(dy * attn)
/// Grid: one thread per (row, head). Threads past `rows * n_head` exit.
/// Checks: `attn`, `dy` and `bias` (ST_IN); `pre` and `d_attn` (ST_OUT).
/// `d_pre` is not checked: it feeds only `d_bias` and the two GEMMs, whose
/// outputs are.
kernel void ojas_per_head_gate_bwd(
    device const float *attn [[buffer(0)]],
    device const float *pre [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const float *dy [[buffer(3)]],
    device float *d_attn [[buffer(4)]],
    device float *d_pre [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &n_head [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant uint &plane_len [[buffer(9)]],
    constant uint &pre_len [[buffer(10)]],
    constant uint &bias_len [[buffer(11)]],
    device atomic_uint *st [[buffer(12)]],
    uint rh [[thread_position_in_grid]])
{
    const uint units = rows * n_head;
    if (rh >= units || n_head == 0u) return;
    const uint h = rh % n_head;
    const uint r = rh / n_head;
    const ulong pre_i = (ulong)r * n_head + h;
    if (pre_i >= pre_len || h >= bias_len) return;
    const float p = pre[pre_i];
    const float b = bias[h];
    const float z = p + b;
    const float g = ojas_sigmoid(z);
    const ulong base = pre_i * head_dim;
    if (base + head_dim > plane_len) return;
    bool bad_in = !isfinite(b);
    bool bad_out = !isfinite(p);
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        const float dv = dy[base + d];
        const float av = attn[base + d];
        const float term = dv * av;
        const float da = dv * g;
        d_attn[base + d] = da;
        acc += term;
        bad_in = bad_in || !isfinite(dv) || !isfinite(av);
        bad_out = bad_out || !isfinite(da);
    }
    d_pre[pre_i] = acc * g * (1.0f - g);
    if (bad_in) ojas_gate_flag(st, GATE_ST_IN);
    if (bad_out) ojas_gate_flag(st, GATE_ST_OUT);
}

/// Threadgroup floats the bias sum stages rows in, shared out evenly among a
/// threadgroup's SIMD-groups (8 KiB; all of it when there is one).
constexpr constant uint DBIAS_STAGE = 2048u;

/// d_bias[h] = sum over rows of d_pre[r, h], rows in ascending order from 0.
/// Grid: one SIMD-group per head (`gate_dbias_threads` in `gpu.rs`).
///
/// The SIMD-group's lanes load a block of rows into threadgroup memory
/// together, so the whole block's loads are in flight at once. Lane 0 then
/// adds the block in row order, so the additions are the CPU reference's
/// `((0 + d_0) + d_1) + ...`, bit for bit. The chain of adds is the
/// floor; one thread per head made 4096 dependent loads at the training
/// shape (~0.8 ms on an M5 Pro), and adding through `simd_shuffle` made each
/// link a shuffle and an add (~120 µs). Every branch below but lane 0's
/// depends only on the head, so a SIMD-group takes it whole. A sum that is
/// not finite sets ST_OUT.
kernel void ojas_per_head_gate_dbias(
    device const float *d_pre [[buffer(0)]],
    device float *d_bias [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &n_head [[buffer(3)]],
    constant uint &pre_len [[buffer(4)]],
    constant uint &bias_len [[buffer(5)]],
    device atomic_uint *st [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sw [[threads_per_simdgroup]])
{
    const uint h = tg * sgs + sg;
    if (h >= n_head || h >= bias_len) return;
    // A length too short for the last row's index leaves d_bias unwritten,
    // as the serial loop did when it reached that row.
    if (rows > 0u && (ulong)(rows - 1u) * n_head + h >= pre_len) return;
    threadgroup float stage[DBIAS_STAGE];
    const uint block = DBIAS_STAGE / sgs;
    threadgroup float *slot = stage + sg * block;
    const ulong n = rows;
    float s = 0.0f;
    for (ulong base = 0; base < n; base += block) {
        const uint live = (uint)min((ulong)block, n - base);
        for (uint i = lane; i < live; i += sw) {
            slot[i] = d_pre[(base + i) * n_head + h];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            // Eight loads ahead of their adds; the adds stay in row order.
            uint j = 0;
            for (; j + 8u <= live; j += 8u) {
                const float a0 = slot[j], a1 = slot[j + 1u], a2 = slot[j + 2u], a3 = slot[j + 3u];
                const float a4 = slot[j + 4u], a5 = slot[j + 5u], a6 = slot[j + 6u], a7 = slot[j + 7u];
                s += a0; s += a1; s += a2; s += a3;
                s += a4; s += a5; s += a6; s += a7;
            }
            for (; j < live; j++) {
                s += slot[j];
            }
        }
        // Lane 0 has read the block before any lane overwrites it.
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        d_bias[h] = s;
        if (!isfinite(s)) ojas_gate_flag(st, GATE_ST_OUT);
    }
}
