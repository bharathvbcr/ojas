// General tiled attention backward (FlashAttention-2 on the TensorOps matrix
// units), shared by the causal entry points in `ojas_backend.metal` and the
// general ones in `attn_gen_bwd.metal`.
//
// The contract is `Backend::attention_backward`: bidirectional or causal
// attention, a query length that differs from the key length, an additive
// score bias, per-batch key lengths, grouped-query heads, and the gradient of
// the bias. After tessl's `qwen35_attn_bwd.metal` (causal, head dim 256,
// [B, T, H, D]); tessl has no kernel for the rest, and ojas's planes are
// [B, H, T, D].
//
//   S   = scale * Q Kᵀ + bias
//   P   = exp(S - lse),  dP = dO Vᵀ,  Dr = rowsum(dO ∘ O)
//   g   = P ∘ (dP - Dr)                           the bias gradient, unscaled
//   dS  = scale * g
//   dQ  = dS K,  dK = dSᵀ Q,  dV = Pᵀ dO
//
// The forward's O and lse come in, so no pass rebuilds the row statistics.
// Every output element is written once, by the threadgroup that owns it, in
// a fixed order and with no atomics, so results repeat bit for bit.
//
//   dq    a threadgroup owns 32 query rows of a query plane and walks the key
//         blocks those rows see.
//   dkv   a threadgroup owns 32 key rows of a KV plane and walks, for each of
//         the `rep` query planes sharing it in increasing order, the query
//         blocks that see them.
//   dbias a threadgroup owns one 32 x 32 tile of a bias plane (or, for a bias
//         with a single query row, 32 columns of it). A bias that broadcasts
//         over the batch or the heads is summed over the planes it serves by
//         that one threadgroup, batch-major and head-minor, query rows in
//         increasing order: the order the CPU reference sums in.
//
// Which keys a query row sees ([lo, hi), contiguous, `agb_range`):
//   keys j < len        len = min(lens[b], Tk) when key lengths are bound
//   causal              query i sits at position i + Tk - Tq: keys <= position,
//                       and with a window W > 0 also keys > position - W; a
//                       position below 0 sees none
//   radius R > 0        keys with |i - j| <= R (not with causal)
//   dead rows           with flag bit 3 (Tq == Tk), a query i >= len sees none
// A query that sees no key has no gradient: its dQ row is 0, and its P is 0
// wherever it would contribute to dK, dV or the bias gradient.
//
// Flags: bit 0 causal, bit 1 a bias is bound, bit 2 key lengths are bound,
// bit 3 padded query rows are dead.
//
// Q, K, V, dO are MPP tensors over the plane with extents (D, T): the matrix
// units read nothing past row T or column D. The score-side block is what is
// zeroed, not those operands: P and dS of every entry a row does not see are
// set to exactly 0 before they multiply V, K, Q or dO.
//
// Compiled with -ffp-contract=off and -fmetal-math-mode=safe; the
// transcendental calls are `precise::`.
#pragma once

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef ATT_BQ
#define ATT_BQ 32
#define ATT_BK 32
#define ATT_NSG 4
#define ATT_THREADS (ATT_NSG * 32)
#endif

#define AGB_CAUSAL 1u
#define AGB_BIAS 2u
#define AGB_LENS 4u
#define AGB_DEAD_ROWS 8u

/// The mask of one launch: query and key lengths, the causal window, the
/// radius (0: none) and the flags above.
struct AgbMask {
    uint Tq;
    uint Tk;
    uint W;
    uint R;
    uint flags;
};

/// The bias extents `[batch, heads, queries]`; an extent of 1 broadcasts.
struct AgbBias {
    uint b;
    uint h;
    uint q;
};

/// Keys of batch row `b` that exist: `min(lens[b], Tk)`, or `Tk`.
inline uint agb_len(AgbMask m, device const uint *lens, uint b)
{
    return ((m.flags & AGB_LENS) != 0u) ? min(lens[b], m.Tk) : m.Tk;
}

/// The keys query row `i` sees, `[lo, hi)`, before an empty range is
/// normalized: `lo` and `hi` do not decrease with `i`, and `lo >= hi` means
/// the row sees none.
inline void agb_range_raw(AgbMask m, uint len, uint i, thread uint &lo, thread uint &hi)
{
    lo = 0u;
    hi = len;
    if ((m.flags & AGB_DEAD_ROWS) != 0u && i >= len) {
        hi = 0u;
    } else if ((m.flags & AGB_CAUSAL) != 0u) {
        const long pos = (long)i + (long)m.Tk - (long)m.Tq;
        if (pos < 0) {
            hi = 0u;
        } else {
            hi = min(hi, (uint)pos + 1u);
            if (m.W != 0u && (uint)pos + 1u > m.W) { lo = (uint)pos + 1u - m.W; }
        }
    } else if (m.R != 0u) {
        lo = (i > m.R) ? i - m.R : 0u;
        hi = min(hi, i + m.R + 1u);
    }
}

/// The keys query row `i` sees, `[lo, hi)`; `lo == hi == 0` when none.
inline void agb_range(AgbMask m, uint len, uint i, thread uint &lo, thread uint &hi)
{
    agb_range_raw(m, len, i, lo, hi);
    if (lo >= hi) { lo = 0u; hi = 0u; }
}

/// The keys any of query rows `[q0, q0 + nq)` sees, `[lo, hi)`: a superset
/// that is tight for every mask above. The ends of a row's range do not
/// decrease with the row, so the first row bounds `lo` and the last bounds
/// `hi`, both taken before an empty range is normalized (a late row with an
/// empty range must not hide the keys of the earlier rows). Rows past `len`
/// are dead, so the last live one bounds `hi`.
inline void agb_block_keys(
    AgbMask m, uint len, uint q0, uint nq, thread uint &lo, thread uint &hi)
{
    uint last = q0 + nq - 1u;
    if ((m.flags & AGB_DEAD_ROWS) != 0u) {
        if (q0 >= len) { lo = 0u; hi = 0u; return; }
        last = min(last, len - 1u);
    }
    uint l0, h0, l1, h1;
    agb_range_raw(m, len, q0, l0, h0);
    agb_range_raw(m, len, last, l1, h1);
    lo = l0;
    hi = h1;
    if (lo >= hi) { lo = 0u; hi = 0u; }
}

/// The query rows `[lo, hi)` that see any of key rows `[k0, k0 + nk)`: a
/// superset that is tight for every mask above.
inline void agb_block_queries(
    AgbMask m, uint len, uint k0, uint nk, thread uint &lo, thread uint &hi)
{
    lo = 0u;
    hi = m.Tq;
    if (k0 >= len) {
        hi = 0u;
    } else if ((m.flags & AGB_CAUSAL) != 0u) {
        const long off = (long)m.Tk - (long)m.Tq;
        // The first query whose position reaches k0, and (with a window) the
        // end of those whose window still holds a key of the block.
        const long none = 0;
        const long one = 1;
        lo = (uint)max(none, min((long)k0 - off, (long)m.Tq));
        if (m.W != 0u) {
            hi = (uint)max(none, min((long)k0 + (long)nk + (long)m.W - one - off, (long)m.Tq));
        }
    } else if (m.R != 0u) {
        lo = (k0 > m.R) ? k0 - m.R : 0u;
        hi = min(m.Tq, k0 + nk + m.R);
    }
    if ((m.flags & AGB_DEAD_ROWS) != 0u) { hi = min(hi, len); }
    if (lo >= hi) { lo = 0u; hi = 0u; }
}

/// Offset of the bias row of query `i` of batch row `b`, head `h`.
inline ulong agb_bias_row(AgbMask m, AgbBias bd, uint b, uint h, uint i)
{
    const ulong plane = (ulong)(bd.b == 1u ? 0u : b) * bd.h + (bd.h == 1u ? 0u : h);
    return (plane * bd.q + (ulong)(bd.q == 1u ? 0u : i)) * m.Tk;
}

/// dQ for query rows [q0, q0 + BQ) of query plane `bh`: walk the key blocks
/// of the rows' keys, rebuild dS per block in threadgroup memory, accumulate
/// dS K.
template <int DM>
inline void agb_dq_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec,
    device const float *bias, device const uint *lens, device float *dQ,
    AgbMask m, uint D, uint H, uint rep, AgbBias bd, float scale,
    uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *dP, threadgroup float *lse_row,
    threadgroup float *d_row)
{
    const uint q0 = tgpig.x * (uint)ATT_BQ;
    if (q0 >= m.Tq) { return; }
    const uint nq = min((uint)ATT_BQ, m.Tq - q0);
    const uint bh = tgpig.y;
    const uint b = bh / H;
    const uint h = bh - b * H;
    const ulong base = (ulong)bh * m.Tq * D;
    const ulong kv_base = (ulong)(bh / rep) * m.Tk * D;
    const uint len = agb_len(m, lens, b);
    const bool biased = (m.flags & AGB_BIAS) != 0u;
    uint k_lo, k_hi;
    agb_block_keys(m, len, q0, nq, k_lo, k_hi);

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BQ, ATT_BK, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        ATT_BQ, DM, ATT_BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<ATT_NSG>> acc_op;

    auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)m.Tq}, array<int, 2>{1, (int)D});
    auto mK = tensor(K + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
    auto mdO = tensor(dO + base, dextents<int, 2>{(int)D, (int)m.Tq}, array<int, 2>{1, (int)D});
    auto tQ = mQ.slice(0, (int)q0);
    auto tdO = mdO.slice(0, (int)q0);
    auto tS = tensor(S, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});
    auto tdP = tensor(dP, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});

    auto dq = acc_op.template get_destination_cooperative_tensor<
        decltype(tS), decltype(mK.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dq.get_capacity(); ++i) { dq[i] = 0.0f; }

    if (tid < (uint)ATT_BQ) {
        const bool live = tid < nq;
        lse_row[tid] = live ? lse[(ulong)bh * m.Tq + q0 + tid] : 0.0f;
        d_row[tid] = live ? dvec[(ulong)bh * m.Tq + q0 + tid] : 0.0f;
    }

    for (uint kb = (k_lo / (uint)ATT_BK) * (uint)ATT_BK; kb < k_hi; kb += (uint)ATT_BK) {
        auto tK = mK.slice(0, (int)kb);
        auto tV = mV.slice(0, (int)kb);
        auto sT = s_op.template get_destination_cooperative_tensor<
            decltype(tQ), decltype(tK), float>();
        s_op.run(tQ, tK, sT);
        sT.store(tS);
        auto pT = s_op.template get_destination_cooperative_tensor<
            decltype(tdO), decltype(tV), float>();
        s_op.run(tdO, tV, pT);
        pT.store(tdP);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < (uint)(ATT_BQ * ATT_BK); i += ATT_THREADS) {
            const uint rr = i / (uint)ATT_BK, c = i % (uint)ATT_BK;
            float ds = 0.0f;
            if (rr < nq) {
                uint lo, hi;
                agb_range(m, len, q0 + rr, lo, hi);
                const uint kj = kb + c;
                if (kj >= lo && kj < hi) {
                    float s = S[i] * scale;
                    if (biased) { s += bias[agb_bias_row(m, bd, b, h, q0 + rr) + kj]; }
                    const float p = precise::exp(s - lse_row[rr]);
                    ds = scale * (p * (dP[i] - d_row[rr]));
                }
            }
            S[i] = ds;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc_op.run(tS, tK, dq);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto mdQ = tensor(dQ + base, dextents<int, 2>{(int)D, (int)m.Tq}, array<int, 2>{1, (int)D});
    dq.store(mdQ.slice(0, (int)q0));
}

/// dK and dV for key rows [k0, k0 + BK) of KV plane `g`: for each of its
/// `rep` query planes in increasing order, walk the query blocks that see a
/// key of the block, rebuild Pᵀ and dSᵀ per block in threadgroup memory,
/// accumulate Pᵀ dO and dSᵀ Q.
template <int DM>
inline void agb_dkv_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec,
    device const float *bias, device const uint *lens, device float *dK, device float *dV,
    AgbMask m, uint D, uint H, uint rep, AgbBias bd, float scale,
    uint2 tgpig, uint tid,
    threadgroup float *Pt, threadgroup float *dSt, threadgroup float *lse_col,
    threadgroup float *d_col)
{
    const uint k0 = tgpig.x * (uint)ATT_BK;
    if (k0 >= m.Tk) { return; }
    const uint nk = min((uint)ATT_BK, m.Tk - k0);
    const uint g = tgpig.y;
    const ulong kv_base = (ulong)g * m.Tk * D;
    // The `rep` query planes of a KV plane share a batch row: `rep` divides H.
    const uint b = (g * rep) / H;
    const uint len = agb_len(m, lens, b);
    const bool biased = (m.flags & AGB_BIAS) != 0u;
    uint q_lo, q_hi;
    agb_block_queries(m, len, k0, nk, q_lo, q_hi);

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BK, ATT_BQ, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto acc_desc = matmul2d_descriptor(
        ATT_BK, DM, ATT_BQ, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;
    matmul2d<acc_desc, execution_simdgroups<ATT_NSG>> acc_op;

    auto mK = tensor(K + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
    auto mV = tensor(V + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
    auto tK = mK.slice(0, (int)k0);
    auto tV = mV.slice(0, (int)k0);
    auto tPt = tensor(Pt, dextents<int, 2>{ATT_BQ, ATT_BK}, array<int, 2>{1, ATT_BQ});
    auto tdSt = tensor(dSt, dextents<int, 2>{ATT_BQ, ATT_BK}, array<int, 2>{1, ATT_BQ});

    auto dk = acc_op.template get_destination_cooperative_tensor<
        decltype(tPt), decltype(mK.slice(0, 0)), float>();
    auto dv = acc_op.template get_destination_cooperative_tensor<
        decltype(tPt), decltype(mK.slice(0, 0)), float>();
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < dk.get_capacity(); ++i) { dk[i] = 0.0f; dv[i] = 0.0f; }

    for (uint r = 0u; r < rep; ++r) {
        const uint bh = g * rep + r;
        const uint h = bh - b * H;
        const ulong base = (ulong)bh * m.Tq * D;
        auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)m.Tq}, array<int, 2>{1, (int)D});
        auto mdO = tensor(dO + base, dextents<int, 2>{(int)D, (int)m.Tq}, array<int, 2>{1, (int)D});
        for (uint qb = (q_lo / (uint)ATT_BQ) * (uint)ATT_BQ; qb < q_hi; qb += (uint)ATT_BQ) {
            const uint nq = min((uint)ATT_BQ, m.Tq - qb);
            if (tid < (uint)ATT_BQ) {
                const bool live = tid < nq;
                lse_col[tid] = live ? lse[(ulong)bh * m.Tq + qb + tid] : 0.0f;
                d_col[tid] = live ? dvec[(ulong)bh * m.Tq + qb + tid] : 0.0f;
            }
            auto tQ = mQ.slice(0, (int)qb);
            auto tdO = mdO.slice(0, (int)qb);
            auto sT = s_op.template get_destination_cooperative_tensor<
                decltype(tK), decltype(tQ), float>();
            s_op.run(tK, tQ, sT);
            sT.store(tPt);
            auto pT = s_op.template get_destination_cooperative_tensor<
                decltype(tV), decltype(tdO), float>();
            s_op.run(tV, tdO, pT);
            pT.store(tdSt);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < (uint)(ATT_BK * ATT_BQ); i += ATT_THREADS) {
                const uint c = i / (uint)ATT_BQ, rr = i % (uint)ATT_BQ;
                float p = 0.0f;
                float ds = 0.0f;
                if (c < nk && rr < nq) {
                    uint lo, hi;
                    agb_range(m, len, qb + rr, lo, hi);
                    const uint kj = k0 + c;
                    if (kj >= lo && kj < hi) {
                        float s = Pt[i] * scale;
                        if (biased) { s += bias[agb_bias_row(m, bd, b, h, qb + rr) + kj]; }
                        p = precise::exp(s - lse_col[rr]);
                        ds = scale * (p * (dSt[i] - d_col[rr]));
                    }
                }
                Pt[i] = p;
                dSt[i] = ds;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc_op.run(tPt, tdO, dv);
            acc_op.run(tdSt, tQ, dk);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    auto mdK = tensor(dK + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
    auto mdV = tensor(dV + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
    dk.store(mdK.slice(0, (int)k0));
    dv.store(mdV.slice(0, (int)k0));
}

/// The bias gradient `g = P (dP - Dr)` of one tile, summed over every
/// `(batch, head)` the bias plane serves.
///
/// Grid x walks key blocks. Grid y is `planes * query blocks`, one tile per
/// threadgroup, when the bias has a row per query; when it has one row
/// (`bd.q == 1`, shared by every query) grid y is `planes` and the group sums
/// every query block, every row in increasing order, into 32 columns. The
/// planes a bias plane serves are walked batch-major then head, as the CPU
/// does, so the sum does not depend on the grid.
template <int DM>
inline void agb_dbias_body(
    device float *Q, device float *K, device float *V, device float *dO,
    device const float *lse, device const float *dvec,
    device const float *bias, device const uint *lens, device float *dBias,
    AgbMask m, uint D, uint B, uint H, uint rep, AgbBias bd, float scale,
    uint2 tgpig, uint tid,
    threadgroup float *S, threadgroup float *dP, threadgroup float *lse_row,
    threadgroup float *d_row)
{
    const uint k0 = tgpig.x * (uint)ATT_BK;
    if (k0 >= m.Tk) { return; }
    const uint nk = min((uint)ATT_BK, m.Tk - k0);
    const bool one_row = (bd.q == 1u);
    const uint q_blocks = (m.Tq + (uint)ATT_BQ - 1u) / (uint)ATT_BQ;
    const uint plane = one_row ? tgpig.y : tgpig.y / q_blocks;
    const uint tile_q0 = one_row ? 0u : (tgpig.y % q_blocks) * (uint)ATT_BQ;
    const uint tile_q_end = one_row ? m.Tq : min(m.Tq, tile_q0 + (uint)ATT_BQ);
    const uint bb = plane / bd.h;
    const uint hb = plane - bb * bd.h;
    const uint b_lo = (bd.b == 1u) ? 0u : bb;
    const uint b_hi = (bd.b == 1u) ? B : bb + 1u;
    const uint h_lo = (bd.h == 1u) ? 0u : hb;
    const uint h_hi = (bd.h == 1u) ? H : hb + 1u;

    constexpr auto s_desc = matmul2d_descriptor(
        ATT_BQ, ATT_BK, DM, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<s_desc, execution_simdgroups<ATT_NSG>> s_op;
    auto tS = tensor(S, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});
    auto tdP = tensor(dP, dextents<int, 2>{ATT_BK, ATT_BQ}, array<int, 2>{1, ATT_BK});

    constexpr uint PER = (uint)(ATT_BQ * ATT_BK) / (uint)ATT_THREADS;
    float acc[PER];
#pragma clang loop unroll(full)
    for (uint e = 0u; e < PER; ++e) { acc[e] = 0.0f; }
    float column = 0.0f;

    for (uint b = b_lo; b < b_hi; ++b) {
        const uint len = agb_len(m, lens, b);
        for (uint h = h_lo; h < h_hi; ++h) {
            const uint bh = b * H + h;
            const ulong base = (ulong)bh * m.Tq * D;
            const ulong kv_base = (ulong)(bh / rep) * m.Tk * D;
            auto mQ = tensor(Q + base, dextents<int, 2>{(int)D, (int)m.Tq}, array<int, 2>{1, (int)D});
            auto mdO = tensor(dO + base, dextents<int, 2>{(int)D, (int)m.Tq}, array<int, 2>{1, (int)D});
            auto mK = tensor(K + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
            auto mV = tensor(V + kv_base, dextents<int, 2>{(int)D, (int)m.Tk}, array<int, 2>{1, (int)D});
            auto tK = mK.slice(0, (int)k0);
            auto tV = mV.slice(0, (int)k0);
            for (uint qb = tile_q0; qb < tile_q_end; qb += (uint)ATT_BQ) {
                const uint nq = min((uint)ATT_BQ, m.Tq - qb);
                uint k_lo, k_hi;
                agb_block_keys(m, len, qb, nq, k_lo, k_hi);
                // No row of the block sees a key of this tile: its gradient
                // is 0 there. (Uniform across the threadgroup.)
                if (k_hi <= k0 || k_lo >= k0 + nk) { continue; }
                if (tid < (uint)ATT_BQ) {
                    const bool live = tid < nq;
                    lse_row[tid] = live ? lse[(ulong)bh * m.Tq + qb + tid] : 0.0f;
                    d_row[tid] = live ? dvec[(ulong)bh * m.Tq + qb + tid] : 0.0f;
                }
                auto tQ = mQ.slice(0, (int)qb);
                auto tdO = mdO.slice(0, (int)qb);
                auto sT = s_op.template get_destination_cooperative_tensor<
                    decltype(tQ), decltype(tK), float>();
                s_op.run(tQ, tK, sT);
                sT.store(tS);
                auto pT = s_op.template get_destination_cooperative_tensor<
                    decltype(tdO), decltype(tV), float>();
                s_op.run(tdO, tV, pT);
                pT.store(tdP);
                threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
                for (uint e = 0u; e < PER; ++e) {
                    const uint i = tid + e * (uint)ATT_THREADS;
                    const uint rr = i / (uint)ATT_BK, c = i % (uint)ATT_BK;
                    float g = 0.0f;
                    if (rr < nq) {
                        uint lo, hi;
                        agb_range(m, len, qb + rr, lo, hi);
                        const uint kj = k0 + c;
                        if (kj >= lo && kj < hi) {
                            float s = S[i] * scale;
                            s += bias[agb_bias_row(m, bd, b, h, qb + rr) + kj];
                            const float p = precise::exp(s - lse_row[rr]);
                            g = p * (dP[i] - d_row[rr]);
                        }
                    }
                    if (one_row) {
                        S[i] = g;
                    } else {
                        acc[e] += g;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (one_row && tid < nk) {
                    for (uint rr = 0u; rr < nq; ++rr) {
                        column += S[rr * (uint)ATT_BK + tid];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    if (one_row) {
        if (tid < nk) { dBias[(ulong)plane * m.Tk + k0 + tid] = column; }
    } else {
#pragma clang loop unroll(full)
        for (uint e = 0u; e < PER; ++e) {
            const uint i = tid + e * (uint)ATT_THREADS;
            const uint rr = i / (uint)ATT_BK, c = i % (uint)ATT_BK;
            if (tile_q0 + rr < m.Tq && c < nk) {
                dBias[((ulong)plane * m.Tq + tile_q0 + rr) * m.Tk + k0 + c] = acc[e];
            }
        }
    }
}
