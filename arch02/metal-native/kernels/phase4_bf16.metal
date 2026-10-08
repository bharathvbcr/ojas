// Phase 4: cast helpers, bf16 copy, fused RMSNorm+scale, bf16 flash (LSE).
#include <metal_stdlib>
using namespace metal;
/// Fused RMSNorm + scalar scale (ln_scale_factor). One thread per row.
/// f32 I/O; reduction accumulates in f32 (precision policy).
kernel void rms_norm_scale_f32(
    device const float *x [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &C [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant float &scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x[base + c];
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * scale;
    for (uint c = 0; c < C; ++c) {
        out[base + c] = x[base + c] * inv;
    }
}

/// Same fused op with bf16 storage; f32 reduction/accum.
kernel void rms_norm_scale_bf16(
    device const bfloat *x [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &C [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant float &scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = float(x[base + c]);
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps) * scale;
    for (uint c = 0; c < C; ++c) {
        out[base + c] = bfloat(float(x[base + c]) * inv);
    }
}

constant uint BR = 32;
constant uint BC = 32;

/// Causal GQA FA-2: bf16 Q/K/V inputs, f32 online-softmax / O / LSE.
/// Training-capable (emits L). Matches flash_attn_fwd_f32 tiling.
kernel void flash_attn_fwd_bf16(
    device const bfloat *Q [[buffer(0)]],
    device const bfloat *K [[buffer(1)]],
    device const bfloat *V [[buffer(2)]],
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
        for (uint d = 0; d < d_lim; ++d) q_reg[d] = float(Q[q_off + d]);
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
            Ks[tk * 64 + d] = float(K[k_off + d]);
            Vs[tk * 64 + d] = float(V[k_off + d]);
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

/// Legacy one-thread-per-row bf16 flash without LSE (inference probe only).
kernel void flash_attn_fwd_bf16_nolse(
    device const bfloat *Q [[buffer(0)]],
    device const bfloat *K [[buffer(1)]],
    device const bfloat *V [[buffer(2)]],
    device bfloat *O [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &T [[buffer(5)]],
    constant uint &H [[buffer(6)]],
    constant uint &Hkv [[buffer(7)]],
    constant uint &D [[buffer(8)]],
    constant float &scale [[buffer(9)]],
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

    const uint q_off = ((b * T + t_q) * H + h) * D;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    thread float acc[64];
    const uint d_lim = min(D, 64u);
    for (uint d = 0; d < d_lim; ++d) acc[d] = 0.0f;

    for (uint t_k = 0; t_k <= t_q; ++t_k) {
        const uint k_off = ((b * T + t_k) * Hkv + hkv) * D;
        float score = 0.0f;
        for (uint d = 0; d < d_lim; ++d) {
            score += float(Q[q_off + d]) * float(K[k_off + d]);
        }
        score *= scale;

        const float m_new = max(m_i, score);
        const float alpha = exp(m_i - m_new);
        const float p = exp(score - m_new);
        l_i = l_i * alpha + p;
        for (uint d = 0; d < d_lim; ++d) {
            acc[d] = acc[d] * alpha + p * float(V[k_off + d]);
        }
        m_i = m_new;
    }

    const float inv_l = 1.0f / l_i;
    for (uint d = 0; d < d_lim; ++d) {
        O[q_off + d] = bfloat(acc[d] * inv_l);
    }
}

/// Delta = rowsum(dO ⊙ O) — f32 (O/dO stay f32 after bf16 flash fwd).
kernel void flash_attn_bwd_delta_bf16(
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

/// dQ with bf16 Q/K/V + taped L/Delta. Tiled BR=BC=32 (matches fwd bf16).
kernel void flash_attn_bwd_dq_bf16(
    device const bfloat *Q [[buffer(0)]],
    device const bfloat *K [[buffer(1)]],
    device const bfloat *V [[buffer(2)]],
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
            q_reg[d] = float(Q[q_off + d]);
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
            Ks[tk * 64 + d] = float(K[k_off + d]);
            Vs[tk * 64 + d] = float(V[k_off + d]);
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

/// dK/dV with bf16 Q/K/V + taped L/Delta. Tiled BR=BC=32.
kernel void flash_attn_bwd_dkv_bf16(
    device const bfloat *Q [[buffer(0)]],
    device const bfloat *K [[buffer(1)]],
    device const bfloat *V [[buffer(2)]],
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
            k_reg[d] = float(K[k_off + d]);
            v_reg[d] = float(V[k_off + d]);
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
                Qs[tq * 64 + d] = float(Q[q_off + d]);
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

// Note: zero_f32, copy_f32, copy_bf16, add_inplace_f32, transpose2d_f32,
// cast_f32_to_bf16, cast_bf16_to_f32 and softcap_f32 are defined once, in
// tessl/kernels/utils.metal, and linked into this metallib by build.rs.
// They were byte-identical duplicates here; two definitions of one kernel
// is a metallib link error waiting to happen, not a redundancy.
