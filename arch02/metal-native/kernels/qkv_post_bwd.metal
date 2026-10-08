// qkv_post backward: reverse q_gain, RoPE, QK-RMSNorm, value residual, VE.
//
// One GPU thread owns one (token, query-head).  The previous token-wise kernel
// used fixed q[256]/k[128] arrays and silently truncated wider models.  Per-head
// scratch is bounded by the public D<=64 shape contract and therefore covers
// the 128M target (Q=768, KV=384) without dimension-specific limits.
#include <metal_stdlib>
using namespace metal;

kernel void qkv_post_bwd_f32(
    device const float *q_pre [[buffer(0)]],
    device const float *k_pre [[buffer(1)]],
    device const float *v_pre [[buffer(2)]],
    device const float *ve [[buffer(3)]],
    device const float *v0 [[buffer(4)]],
    device const float *raw_v [[buffer(5)]],
    device const float *vr_lambda [[buffer(6)]],
    device const float *q_gain [[buffer(7)]],
    device const float *rope_cos [[buffer(8)]],
    device const float *rope_sin [[buffer(9)]],
    device const float *dq_post [[buffer(10)]],
    device const float *dk_post [[buffer(11)]],
    device const float *dv_post [[buffer(12)]],
    device float *dq_pre [[buffer(13)]],
    device float *dk_pre [[buffer(14)]],
    device float *dv_pre [[buffer(15)]],
    device float *dve [[buffer(16)]],
    device float *dv0_acc [[buffer(17)]],
    device atomic_float *d_vr_lambda [[buffer(18)]],
    device atomic_float *d_q_gain [[buffer(19)]],
    constant uint &B [[buffer(20)]],
    constant uint &T [[buffer(21)]],
    constant uint &H [[buffer(22)]],
    constant uint &Hkv [[buffer(23)]],
    constant uint &D [[buffer(24)]],
    constant uint &rope_dims [[buffer(25)]],
    constant uint &use_ve [[buffer(26)]],
    constant uint &use_v0 [[buffer(27)]],
    constant float &eps [[buffer(28)]],
    uint gid [[thread_position_in_grid]])
{
    const uint bt_count = B * T;
    if (gid >= bt_count * H) return;

    const uint bt = gid / H;
    const uint qh = gid - bt * H;
    const uint t = bt % T;
    const uint q_dim = H * D;
    const uint kv_dim = Hkv * D;
    const uint rope_half = rope_dims / 2;
    const device float *cos_t = rope_cos + t * rope_half;
    const device float *sin_t = rope_sin + t * rope_half;

    // Q: RMSNorm -> partial RoPE -> per-head gain.
    thread float q[64];
    thread float q_norm[64];
    thread float dq[64];
    const uint qoff = bt * q_dim + qh * D;
    float q_ms = 0.0f;
    for (uint d = 0; d < D; ++d) {
        q[d] = q_pre[qoff + d];
        dq[d] = dq_post[qoff + d];
        q_ms += q[d] * q[d];
    }
    const float q_inv = rsqrt(q_ms / (float)D + eps);
    for (uint d = 0; d < D; ++d) q_norm[d] = q[d] * q_inv;

    if (rope_dims != 0 && rope_dims < D) {
        for (uint i = 0; i < rope_half; ++i) {
            const float x1 = q_norm[i];
            const float x2 = q_norm[rope_half + i];
            const float c = cos_t[i];
            const float s = sin_t[i];
            q_norm[i] = x1 * c + x2 * s;
            q_norm[rope_half + i] = -x1 * s + x2 * c;
        }
    }

    float dg = 0.0f;
    const float gain = q_gain[qh];
    for (uint d = 0; d < D; ++d) {
        dg += dq[d] * q_norm[d];
        dq[d] *= gain;
    }
    atomic_fetch_add_explicit(&d_q_gain[qh], dg, memory_order_relaxed);

    if (rope_dims != 0 && rope_dims < D) {
        for (uint i = 0; i < rope_half; ++i) {
            const float d1 = dq[i];
            const float d2 = dq[rope_half + i];
            const float c = cos_t[i];
            const float s = sin_t[i];
            dq[i] = d1 * c - d2 * s;
            dq[rope_half + i] = d1 * s + d2 * c;
        }
    }
    float q_dot = 0.0f;
    for (uint d = 0; d < D; ++d) q_dot += q[d] * dq[d];
    const float q_coeff = q_inv * q_inv * q_inv * q_dot / (float)D;
    for (uint d = 0; d < D; ++d) {
        dq_pre[qoff + d] = q_inv * dq[d] - q_coeff * q[d];
    }

    // H is always >= Hkv for the supported GQA configurations.  The first
    // Hkv query-head workers also own one K/V head, so every KV element is
    // produced exactly once while Q remains fully parallel.
    if (qh < Hkv) {
        thread float k[64];
        thread float dk[64];
        const uint koff = bt * kv_dim + qh * D;
        float k_ms = 0.0f;
        for (uint d = 0; d < D; ++d) {
            k[d] = k_pre[koff + d];
            dk[d] = dk_post[koff + d];
            k_ms += k[d] * k[d];
        }
        const float k_inv = rsqrt(k_ms / (float)D + eps);

        if (rope_dims != 0 && rope_dims < D) {
            // K forward RoPE is reconstructed only conceptually; backward is
            // the transpose rotation applied directly to dk_post.
            for (uint i = 0; i < rope_half; ++i) {
                const float d1 = dk[i];
                const float d2 = dk[rope_half + i];
                const float c = cos_t[i];
                const float s = sin_t[i];
                dk[i] = d1 * c - d2 * s;
                dk[rope_half + i] = d1 * s + d2 * c;
            }
        }
        float k_dot = 0.0f;
        for (uint d = 0; d < D; ++d) k_dot += k[d] * dk[d];
        const float k_coeff = k_inv * k_inv * k_inv * k_dot / (float)D;
        for (uint d = 0; d < D; ++d) {
            dk_pre[koff + d] = k_inv * dk[d] - k_coeff * k[d];
        }

        float dl0 = 0.0f;
        float dl1 = 0.0f;
        const float l0 = vr_lambda[0];
        const float l1 = vr_lambda[1];
        for (uint d = 0; d < D; ++d) {
            const uint i = koff + d;
            const float incoming = dv_post[i];
            float draw = incoming;
            if (use_v0 != 0) {
                dl0 += incoming * v0[i];
                dl1 += incoming * raw_v[i];
                atomic_fetch_add_explicit(
                    (device atomic_float *)&dv0_acc[i], l0 * incoming,
                    memory_order_relaxed);
                draw = l1 * incoming;
            }
            dv_pre[i] = draw;
            dve[i] = use_ve != 0 ? draw : 0.0f;
        }
        if (use_v0 != 0) {
            atomic_fetch_add_explicit(&d_vr_lambda[0], dl0, memory_order_relaxed);
            atomic_fetch_add_explicit(&d_vr_lambda[1], dl1, memory_order_relaxed);
        }
    }

    (void)v_pre;
    (void)ve;
}
