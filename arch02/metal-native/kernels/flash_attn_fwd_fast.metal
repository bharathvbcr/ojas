// Audit 8: head-dim-specialized FA-2 forward (causal GQA, tiled online softmax).
//
// Mirrors `flash_attn_fwd_f32` exactly — same FA-2 online-softmax recurrence,
// same tile geometry (BR=BC=32), same taped LSE — with three structural fixes
// that paid off ~11x on the backward twin (Audit 7, DECISIONS M15):
//
//  1. **Compile-time head dim.** The generic kernel bounds its `d` loops by
//     `d_lim = min(D, 64u)`, a runtime value, so `thread float acc[64]` and
//     `q_reg[64]` are dynamically indexed and spill to private (device-backed)
//     memory instead of living in registers. DH is `constexpr` here and every
//     `d` loop is fully unrolled.
//  2. **Half the threadgroup memory.** Generic stages `Ks[BC*64]`/`Vs[BC*64]`
//     = 16 KB even at head_dim 32; this stages `[BC*DH]` = 8 KB, which raises
//     the occupancy ceiling (threadgroup memory is the limiter here).
//  3. **No wasted register slots** — arrays are DH, not 64.
//
// The bf16 twin reads Q/K/V as bfloat with f32 online softmax / O / LSE, exactly
// like `flash_attn_fwd_bf16`. Note `model_fwd::use_bf16_flash` is hard-coded
// false, so bf16 forward flash has never actually run; `METAL_NATIVE_FA_FWD_FAST`
// under `PrecisionMode::Bf16` is the first path that reaches it. Its stated
// rationale ("casting to bf16 just to re-enter flash is a pure round-trip tax")
// predates Audit 7, where bf16 flash was worth ~460 ms in the backward.
//
// Host guards head_dim == DH before dispatch. Buffer indices match the generic
// forward kernels exactly.
#include <metal_stdlib>
using namespace metal;

constexpr constant uint FBR = 32;
constexpr constant uint FBC = 32;
constexpr constant uint DH = 32;

/// FA-2 forward, f32. Threadgroup per (b*h, q_block); threads = FBR.
kernel void flash_attn_fwd_d32_f32(
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

    // Causal: only key blocks up to this query block's last row.
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
            for (uint tk = 0; tk < n_k; ++tk) {
                const uint t_k = t_k0 + tk;
                if (t_k > t_q) break;

                float score = 0.0f;
#pragma clang loop unroll(full)
                for (uint d = 0; d < DH; ++d) {
                    score += q_reg[d] * Ks[tk * DH + d];
                }
                score *= scale;

                const float m_new = max(m_i, score);
                const float alpha = exp(m_i - m_new);
                const float p = exp(score - m_new);
                l_i = l_i * alpha + p;
#pragma clang loop unroll(full)
                for (uint d = 0; d < DH; ++d) {
                    acc[d] = acc[d] * alpha + p * Vs[tk * DH + d];
                }
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
        L[(b * H + h) * T + t_q] = m_i + log(l_i);
    }
}

/// FA-2 forward with bf16 Q/K/V; f32 online softmax / O / LSE.
kernel void flash_attn_fwd_d32_bf16(
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

    const uint t_q0 = q_block * FBR;
    if (t_q0 >= T) return;
    const uint t_q = t_q0 + lid;
    const bool row_valid = (lid < FBR) && (t_q < T);

    // bf16 staging halves threadgroup memory again (8 KB -> 4 KB).
    threadgroup bfloat Ks[FBC * DH];
    threadgroup bfloat Vs[FBC * DH];

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
        for (uint d = 0; d < DH; ++d) q_reg[d] = float(Q[q_off + d]);
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
            for (uint tk = 0; tk < n_k; ++tk) {
                const uint t_k = t_k0 + tk;
                if (t_k > t_q) break;

                float score = 0.0f;
#pragma clang loop unroll(full)
                for (uint d = 0; d < DH; ++d) {
                    score += q_reg[d] * float(Ks[tk * DH + d]);
                }
                score *= scale;

                const float m_new = max(m_i, score);
                const float alpha = exp(m_i - m_new);
                const float p = exp(score - m_new);
                l_i = l_i * alpha + p;
#pragma clang loop unroll(full)
                for (uint d = 0; d < DH; ++d) {
                    acc[d] = acc[d] * alpha + p * float(Vs[tk * DH + d]);
                }
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
        L[(b * H + h) * T + t_q] = m_i + log(l_i);
    }
}
