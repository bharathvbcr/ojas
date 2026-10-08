// Audit 7: head-dim-specialized flash-attn backward (row-wise, causal GQA).
//
// Motivation (2026-07-19 profile, exact-128M): `fa_dqdkv` is ~70% of the whole
// backward (~1.1 s of ~1.56 s). Two structural costs in the generic row
// kernels, both fixed here:
//
//  1. **Loop-invariant reloads.** `flash_attn_bwd_dq_row_f32` reads
//     `Q[q_off+d]` / `dO[q_off+d]` *inside* the t_k loop, and the dKV twin
//     reads `K[k_off+d]` / `V[k_off+d]` inside the t_q loop. The compiler
//     cannot hoist them: `device const float *Q` may alias `device float *dQ`,
//     so every iteration re-reads DH floats from device memory. Here they are
//     loaded once into registers before the loop.
//
//  2. **Accumulators in scratch memory.** The generic kernels bound their `d`
//     loops by `d_lim = min(D, 64u)` — a *runtime* value — so `thread float
//     dq[64]` is dynamically indexed and spills to private (device-backed)
//     memory instead of living in registers. Here DH is a compile-time
//     constant and every `d` loop is fully unrolled, so the accumulators are
//     true registers and no space is wasted (exact-128M head_dim = 32 vs the
//     64-slot generic arrays).
//
// The f32 kernels are **numerically identical** to the generic ones (same
// operations, same order — only redundant loads and spills removed).
//
// The bf16 twins additionally read Q/K/V as bfloat with f32 scores/accum/grads.
// dO stays f32 everywhere (dV accumulates `p * dO` directly and is
// precision-critical).
//
// CORRECTION (do not repeat the earlier claim): this is a bandwidth
// approximation, **not** a fwd/bwd consistency fix. `model_fwd::use_bf16_flash`
// is hard-coded `false`, so the forward always runs f32 flash and tapes an f32
// LSE; bf16 bwd therefore *introduces* a small precision mismatch rather than
// removing one. It is justified empirically only — two-seed 500-step BPB and a
// full 2000-step champion rerun showed no quality cost — and is gated on
// `PrecisionMode::Bf16` so f32 golden runs keep the exact path.
//
// Dispatch is guarded host-side: these kernels are only used when
// head_dim == DH. Buffer indices match the generic row kernels exactly.
#include <metal_stdlib>
using namespace metal;

// Exact-128M / sota head_dim. Host guards head_dim == DH before dispatch.
constexpr constant uint DH = 32;

/// dQ, one thread per (b, h, t_q). f32 Q/K/V — bit-comparable to
/// `flash_attn_bwd_dq_row_f32`.
kernel void flash_attn_bwd_dq_row_d32_f32(
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

    const uint q_off = ((b * T + t_q) * H + h) * DH;
    const float Li = L[(b * H + h) * T + t_q];
    const float Di = Delta[(b * H + h) * T + t_q];

    // Hoisted: loop-invariant across t_k (the generic kernel re-reads these).
    float q_reg[DH];
    float do_reg[DH];
    float dq[DH];
#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        q_reg[d] = Q[q_off + d];
        do_reg[d] = dO[q_off + d];
        dq[d] = 0.0f;
    }

    for (uint t_k = 0; t_k <= t_q; ++t_k) {
        const uint k_off = ((b * T + t_k) * Hkv + hkv) * DH;
        float score = 0.0f;
        float dp = 0.0f;
#pragma clang loop unroll(full)
        for (uint d = 0; d < DH; ++d) {
            score += q_reg[d] * K[k_off + d];
            dp += do_reg[d] * V[k_off + d];
        }
        score *= scale;
        const float p = exp(score - Li);
        const float dss = p * (dp - Di) * scale;
#pragma clang loop unroll(full)
        for (uint d = 0; d < DH; ++d) {
            dq[d] += dss * K[k_off + d];
        }
    }

#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        dQ[q_off + d] = dq[d];
    }
}

/// dK/dV, one thread per (b, hkv, t_k). f32 Q/K/V — bit-comparable to
/// `flash_attn_bwd_dkv_row_f32`.
kernel void flash_attn_bwd_dkv_row_d32_f32(
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
    const uint k_off = ((b * T + t_k) * Hkv + hkv) * DH;

    // Hoisted: loop-invariant across g and t_q.
    float k_reg[DH];
    float v_reg[DH];
    float dk[DH];
    float dv[DH];
#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        k_reg[d] = K[k_off + d];
        v_reg[d] = V[k_off + d];
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }

    for (uint g = 0; g < group; ++g) {
        const uint h = hkv * group + g;
        for (uint t_q = t_k; t_q < T; ++t_q) {
            const uint q_off = ((b * T + t_q) * H + h) * DH;
            const float Li = L[(b * H + h) * T + t_q];
            const float Di = Delta[(b * H + h) * T + t_q];

            float score = 0.0f;
            float dp = 0.0f;
#pragma clang loop unroll(full)
            for (uint d = 0; d < DH; ++d) {
                score += Q[q_off + d] * k_reg[d];
                dp += dO[q_off + d] * v_reg[d];
            }
            score *= scale;
            const float p = exp(score - Li);
            const float dss = p * (dp - Di) * scale;
#pragma clang loop unroll(full)
            for (uint d = 0; d < DH; ++d) {
                dk[d] += dss * Q[q_off + d];
                dv[d] += p * dO[q_off + d];
            }
        }
    }

#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        dK[k_off + d] = dk[d];
        dV[k_off + d] = dv[d];
    }
}

/// dQ with bf16 Q/K/V (f32 dO / L / Delta / scores / accum / dQ).
kernel void flash_attn_bwd_dq_row_d32_bf16(
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

    const uint q_off = ((b * T + t_q) * H + h) * DH;
    const float Li = L[(b * H + h) * T + t_q];
    const float Di = Delta[(b * H + h) * T + t_q];

    float q_reg[DH];
    float do_reg[DH];
    float dq[DH];
#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        q_reg[d] = float(Q[q_off + d]);
        do_reg[d] = dO[q_off + d];
        dq[d] = 0.0f;
    }

    for (uint t_k = 0; t_k <= t_q; ++t_k) {
        const uint k_off = ((b * T + t_k) * Hkv + hkv) * DH;
        float score = 0.0f;
        float dp = 0.0f;
#pragma clang loop unroll(full)
        for (uint d = 0; d < DH; ++d) {
            score += q_reg[d] * float(K[k_off + d]);
            dp += do_reg[d] * float(V[k_off + d]);
        }
        score *= scale;
        const float p = exp(score - Li);
        const float dss = p * (dp - Di) * scale;
#pragma clang loop unroll(full)
        for (uint d = 0; d < DH; ++d) {
            dq[d] += dss * float(K[k_off + d]);
        }
    }

#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        dQ[q_off + d] = dq[d];
    }
}

/// dK/dV with bf16 Q/K/V (f32 dO / L / Delta / scores / accum / grads).
kernel void flash_attn_bwd_dkv_row_d32_bf16(
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
    uint gid [[thread_position_in_grid]])
{
    const uint total = B * Hkv * T;
    if (gid >= total) return;
    const uint t_k = gid % T;
    const uint tmp = gid / T;
    const uint hkv = tmp % Hkv;
    const uint b = tmp / Hkv;
    const uint group = H / Hkv;
    const uint k_off = ((b * T + t_k) * Hkv + hkv) * DH;

    float k_reg[DH];
    float v_reg[DH];
    float dk[DH];
    float dv[DH];
#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        k_reg[d] = float(K[k_off + d]);
        v_reg[d] = float(V[k_off + d]);
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }

    for (uint g = 0; g < group; ++g) {
        const uint h = hkv * group + g;
        for (uint t_q = t_k; t_q < T; ++t_q) {
            const uint q_off = ((b * T + t_q) * H + h) * DH;
            const float Li = L[(b * H + h) * T + t_q];
            const float Di = Delta[(b * H + h) * T + t_q];

            float score = 0.0f;
            float dp = 0.0f;
#pragma clang loop unroll(full)
            for (uint d = 0; d < DH; ++d) {
                score += float(Q[q_off + d]) * k_reg[d];
                dp += dO[q_off + d] * v_reg[d];
            }
            score *= scale;
            const float p = exp(score - Li);
            const float dss = p * (dp - Di) * scale;
#pragma clang loop unroll(full)
            for (uint d = 0; d < DH; ++d) {
                dk[d] += dss * float(Q[q_off + d]);
                dv[d] += p * dO[q_off + d];
            }
        }
    }

#pragma clang loop unroll(full)
    for (uint d = 0; d < DH; ++d) {
        dK[k_off + d] = dk[d];
        dV[k_off + d] = dv[d];
    }
}
