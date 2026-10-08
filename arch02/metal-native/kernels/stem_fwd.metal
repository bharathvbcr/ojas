// Stem forward pieces (f32). Orchestrated as stem_fwd in Rust:
//   embed+bigram → rms_norm → smear  →  fwd/stem_after_smear
#include <metal_stdlib>
using namespace metal;

/// Token embed + bigram hash/embed/(h@proj)*scale. One thread per (b,t).
kernel void stem_embed_f32(
    device const int *input_ids [[buffer(0)]],
    device const float *tok_emb [[buffer(1)]],     // [V, C]
    device const float *bigram_emb [[buffer(2)]],  // [Vb, Db]
    device const float *bigram_proj [[buffer(3)]], // [Db, C] (in,out)
    device const float *bigram_scale [[buffer(4)]],
    device float *out [[buffer(5)]],               // [B, T, C]
    constant uint &B [[buffer(6)]],
    constant uint &T [[buffer(7)]],
    constant uint &C [[buffer(8)]],
    constant uint &Vb [[buffer(9)]],
    constant uint &Db [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const uint b = gid / T;
    const uint t = gid % T;
    const int tok = input_ids[gid];
    const float scale = *bigram_scale;
    const int mod = (int)Vb - 1;

    int hash_idx;
    if (t == 0) {
        hash_idx = mod;
    } else {
        const int t_cur = input_ids[gid];
        const int t_prev = input_ids[b * T + (t - 1)];
        const int xored = (36313 * t_cur) ^ (27191 * t_prev);
        int r = xored % mod;
        if (r < 0) r += mod;
        hash_idx = r;
    }

    const device float *emb_row = bigram_emb + (uint)hash_idx * Db;
    const uint base = gid * C;
    for (uint c = 0; c < C; ++c) {
        float x = tok_emb[(uint)tok * C + c];
        float hsum = 0.0f;
        for (uint d = 0; d < Db; ++d) {
            hsum += emb_row[d] * bigram_proj[d * C + c];
        }
        out[base + c] = x + hsum * scale;
    }
}

/// RMSNorm over last dim. One thread per row (leading dims flattened).
kernel void rms_norm_f32(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &C [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float mean_sq = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = in[base + c];
        mean_sq += v * v;
    }
    mean_sq /= (float)C;
    const float inv = rsqrt(mean_sq + eps);
    for (uint c = 0; c < C; ++c) {
        out[base + c] = in[base + c] * inv;
    }
}

/// In-place RMSNorm (same buffer).
kernel void rms_norm_inplace_f32(
    device float *x [[buffer(0)]],
    constant uint &rows [[buffer(1)]],
    constant uint &C [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float mean_sq = 0.0f;
    for (uint c = 0; c < C; ++c) {
        float v = x[base + c];
        mean_sq += v * v;
    }
    mean_sq /= (float)C;
    const float inv = rsqrt(mean_sq + eps);
    for (uint c = 0; c < C; ++c) {
        x[base + c] *= inv;
    }
}

/// Smear gate along time. One thread per (b,t). Uses post-norm x_{t-1} (zeros at t=0).
kernel void stem_smear_f32(
    device const float *x_norm [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &C [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * T) return;
    const uint t = gid % T;
    const uint b = gid / T;
    const uint base = gid * C;
    const uint prev_base = (b * T + (t - 1)) * C;
    for (uint c = 0; c < C; ++c) {
        const float g = 1.0f / (1.0f + exp(-gate[c]));
        const float xt = x_norm[base + c];
        const float xp = (t == 0) ? 0.0f : x_norm[prev_base + c];
        out[base + c] = (1.0f - g) * xt + g * xp;
    }
}
