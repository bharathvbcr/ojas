// SSM helper kernels: activations, Mamba-2 prep/finish, weighted RMSNorm.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Elementwise activations
// ---------------------------------------------------------------------------

kernel void silu_fwd_f32(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = x[gid];
    y[gid] = v / (1.0f + exp(-v));
}

kernel void silu_bwd_f32(
    device const float *x [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dx [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = x[gid];
    float sig = 1.0f / (1.0f + exp(-v));
    dx[gid] = dy[gid] * sig * (1.0f + v * (1.0f - sig));
}

/// out = softplus(x + bias[h]) for layout [B,T,H] with per-head bias [H].
kernel void softplus_bias_fwd_f32(
    device const float *x [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = B * T * H;
    if (gid >= total) return;
    uint h = gid % H;
    float v = x[gid] + bias[h];
    out[gid] = log(1.0f + exp(v));
}

kernel void softplus_bias_bwd_f32(
    device const float *x [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *dx [[buffer(3)]],
    device atomic<float> *dbias [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = B * T * H;
    if (gid >= total) return;
    uint h = gid % H;
    float v = x[gid] + bias[h];
    float sig = 1.0f / (1.0f + exp(-v));
    float g = dy[gid] * sig;
    dx[gid] = g;
    atomic_fetch_add_explicit(&dbias[h], g, memory_order_relaxed);
}

/// log_dA[t,h] = dt[t,h] * (-exp(A_log[h])) for [B,T,H].
kernel void mamba2_log_da_f32(
    device const float *dt [[buffer(0)]],
    device const float *a_log [[buffer(1)]],
    device float *log_da [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = B * T * H;
    if (gid >= total) return;
    uint h = gid % H;
    float neg_a = -exp(a_log[h]);
    log_da[gid] = dt[gid] * neg_a;
}

/// Backward for x_scaled = dt * xs.
kernel void mamba2_x_scaled_bwd_f32(
    device const float *grad_x_scaled [[buffer(0)]],
    device const float *xs [[buffer(1)]],
    device const float *dt [[buffer(2)]],
    device float *grad_xs [[buffer(3)]],
    device float *grad_dt [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &P [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint p = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (p >= P || h >= H || b >= B) return;
    for (uint t = 0; t < T; t++) {
        uint idx = b * T * H * P + t * H * P + h * P + p;
        float g = grad_x_scaled[idx];
        float dt_val = dt[b * T * H + t * H + h];
        grad_xs[idx] += g * dt_val;
        if (p == 0) {
            float acc = 0.0f;
            for (uint pp = 0; pp < P; pp++) {
                uint j = b * T * H * P + t * H * P + h * P + pp;
                acc += grad_x_scaled[j] * xs[j];
            }
            grad_dt[b * T * H + t * H + h] = acc;
        }
    }
}

/// Backward for log_da = dt * (-exp(a_log[h])).
kernel void mamba2_log_da_bwd_f32(
    device const float *grad_log_da [[buffer(0)]],
    device const float *dt [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device float *grad_dt [[buffer(3)]],
    device atomic<float> *grad_a_log [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = B * T * H;
    if (gid >= total) return;
    uint h = gid % H;
    float neg_a = -exp(a_log[h]);
    float g = grad_log_da[gid];
    grad_dt[gid] += g * neg_a;
    atomic_fetch_add_explicit(&grad_a_log[h], g * dt[gid], memory_order_relaxed);
}

/// dst[r, start:start+len] += src[r, 0:len]
kernel void accum_slice_grad_f32(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &total_out [[buffer(3)]],
    constant uint &len [[buffer(4)]],
    constant uint &start [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows * len) return;
    uint r = gid / len;
    uint c = gid % len;
    dst[r * total_out + start + c] += src[gid];
}

/// x_scaled = dt[t,h] * xs for [B,T,H,P] with dt [B,T,H].
kernel void mamba2_x_scaled_rows_f32(
    device const float *xs [[buffer(0)]],
    device const float *dt [[buffer(1)]],
    device float *x_scaled [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &P [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint p = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (p >= P || h >= H || b >= B) return;
    for (uint t = 0; t < T; t++) {
        uint idx = b * T * H * P + t * H * P + h * P + p;
        float dt_val = dt[b * T * H + t * H + h];
        x_scaled[idx] = dt_val * xs[idx];
    }
}

/// y += xs * D[h] for head layout [B,T,H,P].
kernel void mamba2_d_skip_fwd_f32(
    device float *y [[buffer(0)]],
    device const float *xs [[buffer(1)]],
    device const float *d_param [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &P [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint p = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (p >= P || h >= H || b >= B) return;
    float d = d_param[h];
    for (uint t = 0; t < T; t++) {
        uint idx = b * T * H * P + t * H * P + h * P + p;
        y[idx] += xs[idx] * d;
    }
}

kernel void mamba2_d_skip_bwd_f32(
    device const float *dy [[buffer(0)]],
    device const float *xs [[buffer(1)]],
    device const float *d_param [[buffer(2)]],
    device float *dxs [[buffer(3)]],
    device atomic<float> *dd [[buffer(4)]],
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &P [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint p = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (p >= P || h >= H || b >= B) return;
    float d = d_param[h];
    float dd_h = 0.0f;
    for (uint t = 0; t < T; t++) {
        uint idx = b * T * H * P + t * H * P + h * P + p;
        float g = dy[idx];
        dxs[idx] += g * d;
        dd_h += g * xs[idx];
    }
    atomic_fetch_add_explicit(&dd[h], dd_h, memory_order_relaxed);
}

/// RMSNorm with learnable weight: out = x / rms(x) * weight.
kernel void rms_norm_weight_fwd_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &C [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; c++) {
        float v = x[base + c];
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps);
    for (uint c = 0; c < C; c++) {
        out[base + c] = x[base + c] * inv * weight[c];
    }
}

kernel void rms_norm_weight_bwd_f32(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *dx [[buffer(3)]],
    device atomic<float> *dweight [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant float &eps [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    const uint base = gid * C;
    float ss = 0.0f;
    for (uint c = 0; c < C; c++) {
        float v = x[base + c];
        ss += v * v;
    }
    float inv = rsqrt(ss / float(C) + eps);
    float inv3 = inv * inv * inv;
    float dot = 0.0f;
    for (uint c = 0; c < C; c++) {
        dot += x[base + c] * dy[base + c] * weight[c];
    }
    float coeff = inv3 / float(C) * dot;
    for (uint c = 0; c < C; c++) {
        float normed = x[base + c] * inv;
        float dys = dy[base + c] * weight[c];
        dx[base + c] = inv * dys - coeff * x[base + c];
        atomic_fetch_add_explicit(&dweight[c], dy[base + c] * normed, memory_order_relaxed);
    }
}

/// out = a * b elementwise.
kernel void mul_fwd_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = a[gid] * b[gid];
}

kernel void mul_bwd_a_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device const float *dy [[buffer(2)]],
    device float *da [[buffer(3)]],
    device float *db [[buffer(4)]],
    constant uint &n [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    da[gid] = dy[gid] * b[gid];
    db[gid] = dy[gid] * a[gid];
}

/// silu on conv output fused with storing pre-silu (optional second buffer).
kernel void silu_fwd_store_f32(
    device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]],
    device float *pre [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = x[gid];
    pre[gid] = v;
    y[gid] = v / (1.0f + exp(-v));
}

kernel void silu_bwd_store_f32(
    device const float *pre [[buffer(0)]],
    device const float *dy [[buffer(1)]],
    device float *dx [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = pre[gid];
    float sig = 1.0f / (1.0f + exp(-v));
    dx[gid] = dy[gid] * sig * (1.0f + v * (1.0f - sig));
}
