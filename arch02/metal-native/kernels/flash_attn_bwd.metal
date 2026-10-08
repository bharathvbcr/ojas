// FA-2 style flash attention backward (causal GQA). f32.
// Forward tapes logsumexp L [B,H,T]. Precompute Delta = rowsum(dO⊙O).
// Tiled BR=BC=32 (matches flash_attn_fwd_f32). Uses taped L + Delta — no
// online-softmax recompute on the production path.
#include <metal_stdlib>
using namespace metal;

constant uint BR = 32;
constant uint BC = 32;

/// Delta[b,h,t] = sum_d dO[b,t,h,d] * O[b,t,h,d]
kernel void flash_attn_bwd_delta_f32(
    device const float *O [[buffer(0)]],
    device const float *dO [[buffer(1)]],
    device float *Delta [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &D [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = B * H * T;
    if (gid >= total) return;
    const uint t = gid % T;
    const uint tmp = gid / T;
    const uint h = tmp % H;
    const uint b = tmp / H;
    const uint d_lim = min(D, 64u);
    const uint off = ((b * T + t) * H + h) * D;
    float delta = 0.0f;
    for (uint d = 0; d < d_lim; ++d) {
        delta += dO[off + d] * O[off + d];
    }
    Delta[(b * H + h) * T + t] = delta;
}

/// dQ: TG per (q_block, b*h). Threads = BR (one query row each).
/// Stages K/V key tiles; p = exp(score - Li) from taped L.
kernel void flash_attn_bwd_dq_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device const float *dO [[buffer(3)]],
    device const float *L [[buffer(4)]],
    device const float *Delta [[buffer(5)]],
    device float *dQ [[buffer(6)]],
    constant uint &B [[buffer(7)]],
    constant uint &T [[buffer(8)]],
    constant uint &H [[buffer(9)]],
    constant uint &Hkv [[buffer(10)]],
    constant uint &D [[buffer(11)]],
    constant float &scale [[buffer(12)]],
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

    thread float q_reg[64];
    thread float do_reg[64];
    thread float dq[64];
    float Li = 0.0f;
    float Di = 0.0f;
    for (uint d = 0; d < d_lim; ++d) {
        q_reg[d] = 0.0f;
        do_reg[d] = 0.0f;
        dq[d] = 0.0f;
    }

    if (row_valid) {
        const uint q_off = ((b * T + t_q) * H + h) * D;
        for (uint d = 0; d < d_lim; ++d) {
            q_reg[d] = Q[q_off + d];
            do_reg[d] = dO[q_off + d];
        }
        Li = L[(b * H + h) * T + t_q];
        Di = Delta[(b * H + h) * T + t_q];
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

                const float p = exp(score - Li);
                float dp = 0.0f;
                for (uint d = 0; d < d_lim; ++d) {
                    dp += do_reg[d] * Vs[tk * 64 + d];
                }
                const float dss = p * (dp - Di) * scale;
                for (uint d = 0; d < d_lim; ++d) {
                    dq[d] += dss * Ks[tk * 64 + d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_valid) {
        const uint q_off = ((b * T + t_q) * H + h) * D;
        for (uint d = 0; d < d_lim; ++d) {
            dQ[q_off + d] = dq[d];
        }
    }
}

/// Legacy O(T³) dKV kept as `flash_attn_bwd_dkv_recompute_f32` for A/B;
/// production kernel below uses L.
kernel void flash_attn_bwd_dkv_recompute_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device const float *O [[buffer(3)]],
    device const float *dO [[buffer(4)]],
    device float *dK [[buffer(5)]],
    device float *dV [[buffer(6)]],
    constant uint &B [[buffer(7)]],
    constant uint &T [[buffer(8)]],
    constant uint &H [[buffer(9)]],
    constant uint &Hkv [[buffer(10)]],
    constant uint &D [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = B * Hkv * T;
    if (gid >= total) return;
    const uint t_k = gid % T;
    const uint tmp = gid / T;
    const uint hkv = tmp % Hkv;
    const uint b = tmp / Hkv;
    const uint group = H / Hkv;
    const uint d_lim = min(D, 64u);
    const uint k_off = ((b * T + t_k) * Hkv + hkv) * D;

    thread float dk[64];
    thread float dv[64];
    for (uint d = 0; d < d_lim; ++d) { dk[d] = 0.0f; dv[d] = 0.0f; }

    for (uint g = 0; g < group; ++g) {
        const uint h = hkv * group + g;
        for (uint t_q = t_k; t_q < T; ++t_q) {
            const uint q_off = ((b * T + t_q) * H + h) * D;

            float m_i = -INFINITY;
            float l_i = 0.0f;
            for (uint tk2 = 0; tk2 <= t_q; ++tk2) {
                const uint ko = ((b * T + tk2) * Hkv + hkv) * D;
                float score = 0.0f;
                for (uint d = 0; d < d_lim; ++d) score += Q[q_off + d] * K[ko + d];
                score *= scale;
                float m_new = max(m_i, score);
                float alpha = exp(m_i - m_new);
                float pp = exp(score - m_new);
                l_i = l_i * alpha + pp;
                m_i = m_new;
            }
            float inv_l = 1.0f / l_i;
            float delta = 0.0f;
            for (uint d = 0; d < d_lim; ++d) delta += dO[q_off + d] * O[q_off + d];

            float score = 0.0f;
            for (uint d = 0; d < d_lim; ++d) score += Q[q_off + d] * K[k_off + d];
            score *= scale;
            float p = exp(score - m_i) * inv_l;
            float dp = 0.0f;
            for (uint d = 0; d < d_lim; ++d) {
                dp += dO[q_off + d] * V[k_off + d];
                dv[d] += p * dO[q_off + d];
            }
            float ds = p * (dp - delta);
            float dss = ds * scale;
            for (uint d = 0; d < d_lim; ++d) {
                dk[d] += dss * Q[q_off + d];
            }
        }
    }
    for (uint d = 0; d < d_lim; ++d) {
        dK[k_off + d] = dk[d];
        dV[k_off + d] = dv[d];
    }
}

/// dK/dV: TG per (k_block, b*hkv). Threads = BC (one key row each).
/// Stages Q/dO/L/Delta query tiles; loops GQA groups inside TG.
kernel void flash_attn_bwd_dkv_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device const float *dO [[buffer(3)]],
    device const float *L [[buffer(4)]],
    device const float *Delta [[buffer(5)]],
    device float *dK [[buffer(6)]],
    device float *dV [[buffer(7)]],
    constant uint &B [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &H [[buffer(10)]],
    constant uint &Hkv [[buffer(11)]],
    constant uint &D [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint2 tpitg [[thread_position_in_threadgroup]],
    uint2 tptg_vec [[threads_per_threadgroup]])
{
    const uint lid = tpitg.x;
    const uint tptg = tptg_vec.x;
    const uint bhkv = tgpig.y;
    const uint k_block = tgpig.x;
    const uint hkv = bhkv % Hkv;
    const uint b = bhkv / Hkv;
    const uint group = H / Hkv;
    const uint d_lim = min(D, 64u);

    const uint t_k0 = k_block * BC;
    if (t_k0 >= T) return;

    const uint t_k = t_k0 + lid;
    const bool key_valid = (lid < BC) && (t_k < T);

    threadgroup float Qs[BR * 64];
    threadgroup float dOs[BR * 64];
    threadgroup float Ls[BR];
    threadgroup float Ds[BR];

    thread float k_reg[64];
    thread float v_reg[64];
    thread float dk[64];
    thread float dv[64];
    for (uint d = 0; d < d_lim; ++d) {
        k_reg[d] = 0.0f;
        v_reg[d] = 0.0f;
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }

    if (key_valid) {
        const uint k_off = ((b * T + t_k) * Hkv + hkv) * D;
        for (uint d = 0; d < d_lim; ++d) {
            k_reg[d] = K[k_off + d];
            v_reg[d] = V[k_off + d];
        }
    }

    const uint n_q_blocks = (T + BR - 1u) / BR;
    const uint q_block_start = t_k0 / BR;

    for (uint qb = q_block_start; qb < n_q_blocks; ++qb) {
        const uint t_q0 = qb * BR;
        const uint n_q = min(BR, T - t_q0);

        for (uint g = 0; g < group; ++g) {
            const uint h = hkv * group + g;

            for (uint i = lid; i < n_q * d_lim; i += tptg) {
                const uint tq = i / d_lim;
                const uint d = i % d_lim;
                const uint q_off = ((b * T + (t_q0 + tq)) * H + h) * D;
                Qs[tq * 64 + d] = Q[q_off + d];
                dOs[tq * 64 + d] = dO[q_off + d];
            }
            for (uint i = lid; i < n_q; i += tptg) {
                const uint t_q = t_q0 + i;
                Ls[i] = L[(b * H + h) * T + t_q];
                Ds[i] = Delta[(b * H + h) * T + t_q];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (key_valid) {
                for (uint tq = 0; tq < n_q; ++tq) {
                    const uint t_q = t_q0 + tq;
                    if (t_q < t_k) continue;

                    float score = 0.0f;
                    for (uint d = 0; d < d_lim; ++d) {
                        score += Qs[tq * 64 + d] * k_reg[d];
                    }
                    score *= scale;

                    const float p = exp(score - Ls[tq]);
                    float dp = 0.0f;
                    for (uint d = 0; d < d_lim; ++d) {
                        dp += dOs[tq * 64 + d] * v_reg[d];
                    }
                    const float dss = p * (dp - Ds[tq]) * scale;
                    for (uint d = 0; d < d_lim; ++d) {
                        dk[d] += dss * Qs[tq * 64 + d];
                        dv[d] += p * dOs[tq * 64 + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (key_valid) {
        const uint k_off = ((b * T + t_k) * Hkv + hkv) * D;
        for (uint d = 0; d < d_lim; ++d) {
            dK[k_off + d] = dk[d];
            dV[k_off + d] = dv[d];
        }
    }
}

// Row-wise O(T²) FA bwd (pre-Phase2 default). Faster at T=256 on M5 Pro.
// Opt into tiled via METAL_NATIVE_FA_TILED=1 (uses flash_attn_bwd_{dq,dkv}_f32).
kernel void flash_attn_bwd_dq_row_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device const float *dO [[buffer(3)]],
    device const float *L [[buffer(4)]],
    device const float *Delta [[buffer(5)]],
    device float *dQ [[buffer(6)]],
    constant uint &B [[buffer(7)]],
    constant uint &T [[buffer(8)]],
    constant uint &H [[buffer(9)]],
    constant uint &Hkv [[buffer(10)]],
    constant uint &D [[buffer(11)]],
    constant float &scale [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = B * H * T;
    if (gid >= total) return;
    const uint t_q = gid % T;
    const uint tmp = gid / T;
    const uint h = tmp % H;
    const uint b = tmp / H;
    const uint group = H / Hkv;
    const uint hkv = h / group;
    const uint d_lim = min(D, 64u);

    const uint q_off = ((b * T + t_q) * H + h) * D;
    const float Li = L[(b * H + h) * T + t_q];
    const float Di = Delta[(b * H + h) * T + t_q];

    thread float dq[64];
    for (uint d = 0; d < d_lim; ++d) dq[d] = 0.0f;

    for (uint t_k = 0; t_k <= t_q; ++t_k) {
        const uint k_off = ((b * T + t_k) * Hkv + hkv) * D;
        float score = 0.0f;
        for (uint d = 0; d < d_lim; ++d) {
            score += Q[q_off + d] * K[k_off + d];
        }
        score *= scale;
        const float p = exp(score - Li);
        float dp = 0.0f;
        for (uint d = 0; d < d_lim; ++d) {
            dp += dO[q_off + d] * V[k_off + d];
        }
        const float dss = p * (dp - Di) * scale;
        for (uint d = 0; d < d_lim; ++d) {
            dq[d] += dss * K[k_off + d];
        }
    }
    for (uint d = 0; d < d_lim; ++d) {
        dQ[q_off + d] = dq[d];
    }
}
kernel void flash_attn_bwd_dkv_row_f32(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device const float *dO [[buffer(3)]],
    device const float *L [[buffer(4)]],
    device const float *Delta [[buffer(5)]],
    device float *dK [[buffer(6)]],
    device float *dV [[buffer(7)]],
    constant uint &B [[buffer(8)]],
    constant uint &T [[buffer(9)]],
    constant uint &H [[buffer(10)]],
    constant uint &Hkv [[buffer(11)]],
    constant uint &D [[buffer(12)]],
    constant float &scale [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = B * Hkv * T;
    if (gid >= total) return;
    const uint t_k = gid % T;
    const uint tmp = gid / T;
    const uint hkv = tmp % Hkv;
    const uint b = tmp / Hkv;
    const uint group = H / Hkv;
    const uint d_lim = min(D, 64u);
    const uint k_off = ((b * T + t_k) * Hkv + hkv) * D;

    thread float dk[64];
    thread float dv[64];
    for (uint d = 0; d < d_lim; ++d) { dk[d] = 0.0f; dv[d] = 0.0f; }

    for (uint g = 0; g < group; ++g) {
        const uint h = hkv * group + g;
        for (uint t_q = t_k; t_q < T; ++t_q) {
            const uint q_off = ((b * T + t_q) * H + h) * D;
            const float Li = L[(b * H + h) * T + t_q];
            const float Di = Delta[(b * H + h) * T + t_q];

            float score = 0.0f;
            for (uint d = 0; d < d_lim; ++d) score += Q[q_off + d] * K[k_off + d];
            score *= scale;
            float p = exp(score - Li);
            float dp = 0.0f;
            for (uint d = 0; d < d_lim; ++d) {
                dp += dO[q_off + d] * V[k_off + d];
                dv[d] += p * dO[q_off + d];
            }
            float dss = p * (dp - Di) * scale;
            for (uint d = 0; d < d_lim; ++d) {
                dk[d] += dss * Q[q_off + d];
            }
        }
    }
    for (uint d = 0; d < d_lim; ++d) {
        dK[k_off + d] = dk[d];
        dV[k_off + d] = dv[d];
    }
}