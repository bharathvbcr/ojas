// Phase G / Soft FA quality: FA-2 blockwise online softmax (FA-3-class numerics).
//
// Production FA fwd (`flash_attn_fwd_f32` / `*_d32_*`) updates (m,l,O) **per key
// token** inside each BC tile. Exact arithmetic is equivalent to FA-2, but the
// float recurrence rescales O/l many times per block and accumulates error.
//
// True FA-2 / FA-3 block formulation (Dao et al.):
//   m' = max(m, rowmax(S_block))
//   α  = exp(m − m')          // one rescale
//   P̃  = exp(S − m')
//   l' = α·l + rowsum(P̃)
//   O' = α·O + P̃ V
//
// Opt-in via METAL_NATIVE_FA_BLOCKSOFT=1 (Soft quality A/B). Not a default —
// changes O/LSE vs the sequential path (not bit-identical). Uses precise::exp
// / precise::log and fma for QK / O accumulate. Buffer ABI matches
// flash_attn_fwd_f32.
#include <metal_stdlib>
using namespace metal;

constant uint BR = 32;
constant uint BC = 32;

/// Generic D≤64 blocksoft forward. TG per (b*h, q_block); threads = BR.
kernel void flash_attn_fwd_blocksoft_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    device float *L [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg_vec [[threads_per_threadgroup]])
{
    const uint lid = tpitg.x;
    const uint tptg = tptg_vec.x;
    const uint bh = tgpig.y;
    const uint q_block = tgpig.x;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = H / Hkv;
    const uint hkv = h / group;
    const uint d_lim = min(D, 64u);

    const uint t_q0 = q_block * BR;
    if (t_q0 >= T) return;

    const uint t_q = t_q0 + lid;
    const bool row_valid = (lid < BR) && (t_q < T);

    threadgroup float Ks[BC * 64];
    threadgroup float Vs[BC * 64];

    float m_i = -INFINITY;
    float l_i = 0.0f;
    thread float acc[64];
    thread float q_reg[64];
    for (uint d = 0; d < d_lim; ++d) {
        acc[d] = 0.0f;
        q_reg[d] = 0.0f;
    }
    if (row_valid) {
        const uint q_off = ((b * T + t_q) * H + h) * D;
        for (uint d = 0; d < d_lim; ++d) q_reg[d] = Q[q_off + d];
    }

    const uint t_q_max = min(t_q0 + BR, T) - 1u;
    const uint n_k_blocks = (t_q_max / BC) + 1u;

    for (uint kb = 0; kb < n_k_blocks; ++kb) {
        const uint t_k0 = kb * BC;
        const uint n_k = min(BC, T - t_k0);

        for (uint i = lid; i < n_k * d_lim; i += tptg) {
            const uint tk = i / d_lim;
            const uint d = i % d_lim;
            const uint k_off = ((b * T + (t_k0 + tk)) * Hkv + hkv) * D;
            Ks[tk * 64 + d] = K[k_off + d];
            Vs[tk * 64 + d] = V[k_off + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            // Pass 1: scores + block rowmax (causal mask → −inf).
            thread float scores[BC];
            float m_block = -INFINITY;
            for (uint tk = 0; tk < n_k; ++tk) {
                const uint t_k = t_k0 + tk;
                if (t_k > t_q) {
                    scores[tk] = -INFINITY;
                    continue;
                }
                float score = 0.0f;
                for (uint d = 0; d < d_lim; ++d) {
                    score = fma(q_reg[d], Ks[tk * 64 + d], score);
                }
                score *= scale;
                scores[tk] = score;
                m_block = max(m_block, score);
            }

            if (m_block > -INFINITY) {
                const float m_new = max(m_i, m_block);
                const float alpha = precise::exp(m_i - m_new);
                l_i *= alpha;
                for (uint d = 0; d < d_lim; ++d) acc[d] *= alpha;

                float l_add = 0.0f;
                for (uint tk = 0; tk < n_k; ++tk) {
                    const float s = scores[tk];
                    if (s == -INFINITY) continue;
                    const float p = precise::exp(s - m_new);
                    l_add += p;
                    for (uint d = 0; d < d_lim; ++d) {
                        acc[d] = fma(p, Vs[tk * 64 + d], acc[d]);
                    }
                }
                l_i += l_add;
                m_i = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_valid) {
        const float inv_l = 1.0f / l_i;
        const uint o_off = ((b * T + t_q) * H + h) * D;
        for (uint d = 0; d < d_lim; ++d) {
            O[o_off + d] = acc[d] * inv_l;
        }
        L[(b * H + h) * T + t_q] = m_i + precise::log(l_i);
    }
}

constexpr constant uint FBR = 32;
constexpr constant uint FBC = 32;
constexpr constant uint DH = 32;

/// DH=32 specialized blocksoft forward (Soft quality hot shape).
kernel void flash_attn_fwd_blocksoft_d32_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *O [[buffer(3)]],
    device float *L [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg_vec [[threads_per_threadgroup]])
{
    const uint lid = tpitg.x;
    const uint tptg = tptg_vec.x;
    const uint bh = tgpig.y;
    const uint q_block = tgpig.x;
    const uint h = bh % H;
    const uint b = bh / H;
    const uint group = H / Hkv;
    const uint hkv = h / group;

    const uint t_q0 = q_block * FBR;
    if (t_q0 >= T) return;
    const uint t_q = t_q0 + lid;
    const bool row_valid = (lid < FBR) && (t_q < T);

    threadgroup float Ks[FBC * DH];
    threadgroup float Vs[FBC * DH];

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[DH];
    float q_reg[DH];
#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        acc[d] = 0.0f;
        q_reg[d] = 0.0f;
    }
    if (row_valid) {
        const uint q_off = ((b * T + t_q) * H + h) * DH;
#pragma clang loop unroll(full)
        for (uint d = 0; d < DH; ++d) q_reg[d] = Q[q_off + d];
    }

    const uint t_q_max = min(t_q0 + FBR, T) - 1u;
    const uint n_k_blocks = (t_q_max / FBC) + 1u;

    for (uint kb = 0; kb < n_k_blocks; ++kb) {
        const uint t_k0 = kb * FBC;
        const uint n_k = min(FBC, T - t_k0);

        for (uint i = lid; i < n_k * DH; i += tptg) {
            const uint tk = i / DH;
            const uint d = i % DH;
            const uint k_off = ((b * T + (t_k0 + tk)) * Hkv + hkv) * DH;
            Ks[tk * DH + d] = K[k_off + d];
            Vs[tk * DH + d] = V[k_off + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_valid) {
            float scores[FBC];
            float m_block = -INFINITY;
            for (uint tk = 0; tk < n_k; ++tk) {
                const uint t_k = t_k0 + tk;
                if (t_k > t_q) {
                    scores[tk] = -INFINITY;
                    continue;
                }
                float score = 0.0f;
#pragma clang loop unroll(full)
                for (uint d = 0; d < DH; ++d) {
                    score = fma(q_reg[d], Ks[tk * DH + d], score);
                }
                score *= scale;
                scores[tk] = score;
                m_block = max(m_block, score);
            }

            if (m_block > -INFINITY) {
                const float m_new = max(m_i, m_block);
                const float alpha = precise::exp(m_i - m_new);
                l_i *= alpha;
#pragma clang loop unroll(full)
                for (uint d = 0; d < DH; ++d) acc[d] *= alpha;

                float l_add = 0.0f;
                for (uint tk = 0; tk < n_k; ++tk) {
                    const float s = scores[tk];
                    if (s == -INFINITY) continue;
                    const float p = precise::exp(s - m_new);
                    l_add += p;
#pragma clang loop unroll(full)
                    for (uint d = 0; d < DH; ++d) {
                        acc[d] = fma(p, Vs[tk * DH + d], acc[d]);
                    }
                }
                l_i += l_add;
                m_i = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_valid) {
        const uint q_off = ((b * T + t_q) * H + h) * DH;
        const float inv_l = 1.0f / l_i;
#pragma clang loop unroll(full)
        for (uint d = 0; d < DH; ++d) {
            O[q_off + d] = acc[d] * inv_l;
        }
        L[(b * H + h) * T + t_q] = m_i + precise::log(l_i);
    }
}
