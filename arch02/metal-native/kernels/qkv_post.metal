// qkv_post: VE inject + value-residual mix + reshape + QK-RMSNorm + partial RoPE + q_gain.
// Out-of-place: reads GEMM pre buffers, writes post buffers (kills tape deep_copy).
#include <metal_stdlib>
using namespace metal;

kernel void qkv_post_f32(
    device const float *q_in [[buffer(0)]],        // [B,T,H*D] GEMM pre
    device const float *k_in [[buffer(1)]],
    device const float *v_in [[buffer(2)]],
    device float *q_out [[buffer(3)]],             // post (RMS+RoPE+gain)
    device float *k_out [[buffer(4)]],
    device float *v_out [[buffer(5)]],
    device const float *ve [[buffer(6)]],
    device const float *v0 [[buffer(7)]],
    device float *raw_v_out [[buffer(8)]],
    device const float *vr_lambda [[buffer(9)]],
    device const float *q_gain [[buffer(10)]],
    device const float *rope_cos [[buffer(11)]],
    device const float *rope_sin [[buffer(12)]],
    constant uint &B [[buffer(13)]],
    constant uint &T [[buffer(14)]],
    constant uint &H [[buffer(15)]],
    constant uint &Hkv [[buffer(16)]],
    constant uint &D [[buffer(17)]],
    constant uint &rope_dims [[buffer(18)]],
    constant uint &use_ve [[buffer(19)]],
    constant uint &use_v0 [[buffer(20)]],
    constant float &eps [[buffer(21)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const uint bt = gid;
    const uint t = gid % T;
    const uint kv_dim = Hkv * D;
    const uint q_dim = H * D;

    // V: VE inject → raw_v → optional λ-mix into v_out
    for (uint i = 0; i < kv_dim; ++i) {
        float vv = v_in[bt * kv_dim + i];
        if (use_ve != 0) {
            vv += ve[bt * kv_dim + i];
        }
        raw_v_out[bt * kv_dim + i] = vv;
        if (use_v0 != 0) {
            vv = vr_lambda[0] * v0[bt * kv_dim + i] + vr_lambda[1] * vv;
        }
        v_out[bt * kv_dim + i] = vv;
    }

    // Q/K: copy → RMSNorm → RoPE → (q only) gain
    for (uint i = 0; i < q_dim; ++i) {
        q_out[bt * q_dim + i] = q_in[bt * q_dim + i];
    }
    for (uint i = 0; i < kv_dim; ++i) {
        k_out[bt * kv_dim + i] = k_in[bt * kv_dim + i];
    }

    auto rms_head = [&](device float *base, uint n_heads) {
        for (uint h = 0; h < n_heads; ++h) {
            device float *row = base + h * D;
            float ms = 0.0f;
            for (uint d = 0; d < D; ++d) ms += row[d] * row[d];
            ms /= (float)D;
            float inv = rsqrt(ms + eps);
            for (uint d = 0; d < D; ++d) row[d] *= inv;
        }
    };
    rms_head(q_out + bt * q_dim, H);
    rms_head(k_out + bt * kv_dim, Hkv);

    const uint rope_half = rope_dims / 2;
    const device float *cos_t = rope_cos + t * rope_half;
    const device float *sin_t = rope_sin + t * rope_half;

    auto apply_rope = [&](device float *base, uint n_heads) {
        for (uint h = 0; h < n_heads; ++h) {
            device float *row = base + h * D;
            for (uint i = 0; i < rope_half; ++i) {
                float x1 = row[i];
                float x2 = row[rope_half + i];
                float c = cos_t[i];
                float s = sin_t[i];
                row[i] = x1 * c + x2 * s;
                row[rope_half + i] = x1 * (-s) + x2 * c;
            }
        }
    };
    if (rope_dims > 0 && rope_dims < D) {
        apply_rope(q_out + bt * q_dim, H);
        apply_rope(k_out + bt * kv_dim, Hkv);
    }

    for (uint h = 0; h < H; ++h) {
        float g = q_gain[h];
        device float *row = q_out + bt * q_dim + h * D;
        for (uint d = 0; d < D; ++d) row[d] *= g;
    }
}

/// Build VE: embed gather + proj + scale * layer_scale. One thread per (b,t).
kernel void ve_fwd_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *ve_emb [[buffer(1)]],      // [V, De]
    device const float *ve_proj [[buffer(2)]],     // [De, kv_dim] (in,out)
    device const float *ve_scale [[buffer(3)]],
    device const float *layer_scale [[buffer(4)]], // [1]
    device float *out [[buffer(5)]],               // [B,T,kv_dim]
    constant uint &B [[buffer(6)]],
    constant uint &T [[buffer(7)]],
    constant uint &V [[buffer(8)]],
    constant uint &De [[buffer(9)]],
    constant uint &kv_dim [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const int tok = input_ids[gid];
    if (tok < 0 || (uint)tok >= V) {
        for (uint o = 0; o < kv_dim; ++o) out[gid * kv_dim + o] = 0.0f;
        return;
    }
    const float s = (*ve_scale) * (*layer_scale);
    const device float *erow = ve_emb + (uint)tok * De;
    const uint base = gid * kv_dim;
    for (uint o = 0; o < kv_dim; ++o) {
        float acc = 0.0f;
        for (uint d = 0; d < De; ++d) {
            acc += erow[d] * ve_proj[d * kv_dim + o];
        }
        out[base + o] = acc * s;
    }
}
