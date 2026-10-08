// Causal GQA flash attention forward — FA-2 style tiled online softmax (f32).
// Q: [B,T,H,D], K/V: [B,T,Hkv,D], O: [B,T,H,D], L: [B,H,T] logsumexp.
//
// Production hot path (DECISIONS M8). TensorOps multi-block probe lives in
// flash_attn_tensorops.metal and is not the training default.
#include <metal_stdlib>
using namespace metal;

constant uint BR = 32;
constant uint BC = 32;

/// FA-2 forward: threadgroup per (b, h, q_block). Threads = BR (one per query row).
kernel void flash_attn_fwd_f32(
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
    for (uint d = 0; d < d_lim; ++d) acc[d] = 0.0f;

    thread float q_reg[64];
    if (row_valid) {
        const uint q_off = ((b * T + t_q) * H + h) * D;
        for (uint d = 0; d < d_lim; ++d) q_reg[d] = Q[q_off + d];
    } else {
        for (uint d = 0; d < d_lim; ++d) q_reg[d] = 0.0f;
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
            for (uint tk = 0; tk < n_k; ++tk) {
                const uint t_k = t_k0 + tk;
                if (t_k > t_q) break;

                float score = 0.0f;
                for (uint d = 0; d < d_lim; ++d) {
                    score += q_reg[d] * Ks[tk * 64 + d];
                }
                score *= scale;

                const float m_new = max(m_i, score);
                const float alpha = exp(m_i - m_new);
                const float p = exp(score - m_new);
                l_i = l_i * alpha + p;
                for (uint d = 0; d < d_lim; ++d) {
                    acc[d] = acc[d] * alpha + p * Vs[tk * 64 + d];
                }
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
        L[(b * H + h) * T + t_q] = m_i + log(l_i);
    }
}
