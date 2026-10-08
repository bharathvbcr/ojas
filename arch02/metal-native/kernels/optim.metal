// On-device optimizer kernels (f32): global L2 clip, banked Muon NS5, AdamW+EMA.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Global L2 clip: partial sum-of-squares → clip coef in a 4-byte device buffer
// (no host into_scalar in steady state).
// ---------------------------------------------------------------------------

kernel void grad_sq_reduce_f32(
    device const float *g [[buffer(0)]],
    device atomic_float *total_sq [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    threadgroup float shared[1024];
    float local = 0.0f;
    if (gid < n) {
        float v = g[gid];
        local = v * v;
    }
    shared[lid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Tree reduce; handles non-power-of-2 tpg.
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) {
            shared[idx] += shared[idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        atomic_fetch_add_explicit(total_sq, shared[0], memory_order_relaxed);
    }
}

kernel void clip_coef_f32(
    device const float *total_sq [[buffer(0)]],
    device float *clip_coef [[buffer(1)]],
    device float *norm_out [[buffer(2)]],
    constant float &max_norm [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) return;
    float sq = *total_sq;
    float norm = sqrt(sq);
    *norm_out = norm;
    float coef = 1.0f;
    if (norm > max_norm && norm > 0.0f) {
        coef = max_norm / (norm + eps);
    }
    *clip_coef = coef;
}

/// Soft optim coef = sqrt(clip_coef) for ClipMode::Soft.
kernel void clip_soft_coef_f32(
    device const float *clip_coef [[buffer(0)]],
    device float *clip_soft [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid > 0) return;
    *clip_soft = sqrt(max(*clip_coef, 0.0f));
}

kernel void scale_by_clip_coef_f32(
    device float *g [[buffer(0)]],
    device const float *clip_coef [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    g[gid] *= *clip_coef;
}

kernel void zero_scalar_f32(
    device float *buf [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    buf[gid] = 0.0f;
}

// ---------------------------------------------------------------------------
// AdamW (decoupled WD) + optional EMA in one pass.
// ---------------------------------------------------------------------------

kernel void adamw_ema_f32(
    device float *param [[buffer(0)]],
    device const float *grad [[buffer(1)]],
    device float *exp_avg [[buffer(2)]],
    device float *exp_avg_sq [[buffer(3)]],
    device float *ema [[buffer(4)]],
    constant float &lr [[buffer(5)]],
    constant float &beta1 [[buffer(6)]],
    constant float &beta2 [[buffer(7)]],
    constant float &eps [[buffer(8)]],
    constant float &wd [[buffer(9)]],
    constant float &step_size [[buffer(10)]],
    constant float &bias2_sqrt_inv [[buffer(11)]],
    constant float &ema_decay [[buffer(12)]],
    constant uint &n [[buffer(13)]],
    constant uint &do_ema [[buffer(14)]],
    // Global clip coefficient (1 when unclipped). Adam's m/√v is approximately
    // invariant to grad scaling once moments adapt, so without this AdamW keeps
    // taking full-size steps while Muon×clip is throttled — residual scales
    // (vr_lambda / attn_scale) then explode under chronic clipping (~step 2500+).
    device const float *clip_coef [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = grad[gid];
    float m = exp_avg[gid];
    float v = exp_avg_sq[gid];
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    exp_avg[gid] = m;
    exp_avg_sq[gid] = v;
    float denom = sqrt(v) * bias2_sqrt_inv + eps;
    float c = *clip_coef;
    float p = param[gid];
    p = p * (1.0f - lr * wd * c) - (step_size * c) * (m / denom);
    param[gid] = p;
    if (do_ema != 0) {
        ema[gid] = ema_decay * ema[gid] + (1.0f - ema_decay) * p;
    }
}

// ---------------------------------------------------------------------------
// Full-parameter research controls. State1/state2 are interpreted per
// algorithm; all candidates share the same fused master/EMA write contract.
// ---------------------------------------------------------------------------

kernel void cautious_mask_count_f32(
    device const float *grad [[buffer(0)]], device const float *state1 [[buffer(1)]],
    device atomic_float *count [[buffer(2)]], constant float &beta1 [[buffer(3)]],
    constant uint &algorithm [[buffer(4)]], constant uint &n [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = grad[gid];
    float m = state1[gid];
    float proposed = beta1 * m + (1.0f - beta1) * g;
    if (algorithm == 2u) proposed = sign(proposed); // cautious Lion
    if (proposed * g > 0.0f) {
        atomic_fetch_add_explicit(count, 1.0f, memory_order_relaxed);
    }
}

kernel void research_optimizer_ema_f32(
    device float *param [[buffer(0)]], device const float *grad [[buffer(1)]],
    device float *state1 [[buffer(2)]], device float *state2 [[buffer(3)]],
    device float *ema [[buffer(4)]], device const float *clip_coef [[buffer(5)]],
    constant uint &algorithm [[buffer(6)]], constant float &lr [[buffer(7)]],
    constant float &beta1 [[buffer(8)]], constant float &beta2 [[buffer(9)]],
    constant float &eps [[buffer(10)]], constant float &wd [[buffer(11)]],
    constant float &bc1 [[buffer(12)]], constant float &bc2 [[buffer(13)]],
    constant float &ema_decay [[buffer(14)]], constant uint &n [[buffer(15)]],
    constant uint &do_ema [[buffer(16)]], device const float *mask_count [[buffer(17)]],
    constant float &rho [[buffer(18)]], constant uint &hessian_update [[buffer(19)]],
    constant float &ckp1 [[buffer(20)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = grad[gid];
    float p = param[gid];
    float s1 = state1[gid];
    float s2 = state2[gid];
    float clip = *clip_coef;

    if (algorithm == 0u || algorithm == 3u) { // AdamW / cautious AdamW
        s1 = beta1 * s1 + (1.0f - beta1) * g;
        s2 = beta2 * s2 + (1.0f - beta2) * g * g;
        float update = (s1 / bc1) / (sqrt(s2 / bc2) + eps);
        if (algorithm == 3u) {
            float mean = max(*mask_count / float(n), 1.0e-3f);
            update *= (s1 * g > 0.0f) ? (1.0f / mean) : 0.0f;
        }
        p = p * (1.0f - lr * wd * clip) - lr * clip * update;
    } else if (algorithm == 1u || algorithm == 2u) { // Lion / cautious Lion
        float proposed = beta1 * s1 + (1.0f - beta1) * g;
        float update = sign(proposed);
        if (algorithm == 2u) {
            float mean = max(*mask_count / float(n), 1.0e-3f);
            update *= (update * g > 0.0f) ? (1.0f / mean) : 0.0f;
        }
        p = p * (1.0f - lr * wd * clip) - lr * clip * update;
        s1 = beta2 * s1 + (1.0f - beta2) * g;
    } else if (algorithm == 4u) { // momentum SGD
        s1 = beta1 * s1 + g;
        p = p * (1.0f - lr * wd * clip) - lr * clip * s1;
    } else if (algorithm == 5u) { // Sophia, empirical-Fisher diagonal
        s1 = beta1 * s1 + (1.0f - beta1) * g;
        if (hessian_update != 0u) s2 = beta2 * s2 + (1.0f - beta2) * g * g;
        float update = clamp(s1 / max(rho * s2, eps), -1.0f, 1.0f);
        p = p * (1.0f - lr * wd * clip) - lr * clip * update;
    } else { // 6: Schedule-Free AdamW; state1=z, state2=v, param=y
        float gw = g + wd * p;
        s2 = beta2 * s2 + (1.0f - beta2) * gw * gw;
        s1 -= lr * clip * gw / (sqrt(s2) + eps);
        float x = (1.0f - ckp1) * p + ckp1 * s1;
        p = beta1 * x + (1.0f - beta1) * s1;
    }

    param[gid] = p;
    state1[gid] = s1;
    state2[gid] = s2;
    if (do_ema != 0u) ema[gid] = ema_decay * ema[gid] + (1.0f - ema_decay) * p;
}

kernel void prodigy_accumulate_f32(
    device float *param [[buffer(0)]], device const float *grad [[buffer(1)]],
    device float *m [[buffer(2)]], device float *v [[buffer(3)]],
    device float *s [[buffer(4)]], device const float *p0 [[buffer(5)]],
    device atomic_float *numer [[buffer(6)]], device atomic_float *denom [[buffer(7)]],
    device const float *clip_coef [[buffer(8)]], constant float &d [[buffer(9)]],
    constant float &d0 [[buffer(10)]], constant float &dlr [[buffer(11)]],
    constant float &beta1 [[buffer(12)]], constant float &beta2 [[buffer(13)]],
    constant float &beta3 [[buffer(14)]], constant float &wd [[buffer(15)]],
    constant uint &n [[buffer(16)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = grad[gid];
    float c = *clip_coef;
    float scaled_dlr = dlr * c;
    float ratio = d / d0;
    atomic_fetch_add_explicit(numer, ratio * scaled_dlr * g * (p0[gid] - param[gid]), memory_order_relaxed);
    float mm = beta1 * m[gid] + d * (1.0f - beta1) * g;
    float vv = beta2 * v[gid] + d * d * (1.0f - beta2) * g * g;
    float ss = beta3 * s[gid] + ratio * scaled_dlr * g;
    m[gid] = mm; v[gid] = vv; s[gid] = ss;
    atomic_fetch_add_explicit(denom, abs(ss), memory_order_relaxed);
    param[gid] *= 1.0f - scaled_dlr * wd;
}

kernel void prodigy_finalize_ema_f32(
    device float *param [[buffer(0)]], device const float *m [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *ema [[buffer(3)]],
    device const float *clip_coef [[buffer(4)]], constant float &d [[buffer(5)]],
    constant float &dlr [[buffer(6)]], constant float &eps [[buffer(7)]],
    constant float &ema_decay [[buffer(8)]], constant uint &do_ema [[buffer(9)]],
    constant uint &n [[buffer(10)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float p = param[gid] - dlr * (*clip_coef) * m[gid] / (sqrt(v[gid]) + d * eps);
    param[gid] = p;
    if (do_ema != 0u) ema[gid] = ema_decay * ema[gid] + (1.0f - ema_decay) * p;
}

kernel void ema_update_f32(
    device float *ema [[buffer(0)]],
    device const float *live [[buffer(1)]],
    constant float &decay [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    ema[gid] = decay * ema[gid] + (1.0f - decay) * live[gid];
}

// ---------------------------------------------------------------------------
// TensorOps Muon bank helpers. Matrix contractions live in the batched MPP
// kernels; these kernels fuse momentum/normalization, polynomial combines, and
// final master+EMA writes around them.
// ---------------------------------------------------------------------------

kernel void muon_tensorops_prepare_f32(
    device const float *grad [[buffer(0)]], device float *momentum [[buffer(1)]],
    device float *x [[buffer(2)]], constant uint &N [[buffer(3)]],
    constant uint &mat [[buffer(4)]], constant float &mom [[buffer(5)]],
    constant float &eps [[buffer(6)]], constant float &norm_scale [[buffer(7)]],
    device float *aux_state [[buffer(8)]], device float *prev_grad [[buffer(9)]],
    constant uint &pre_kind [[buffer(10)]], constant float &pre_beta [[buffer(11)]],
    constant float &pre_alpha [[buffer(12)]], constant uint &first_step [[buffer(13)]],
    uint mid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (mid >= N) return;
    uint base = mid * mat;
    threadgroup float red[1024];
    float partial = 0.0f;
    for (uint i = lid; i < mat; i += tpg) {
        float g = grad[base + i];
        float transformed = g;
        if (pre_kind == 1u) {
            float diff = first_step != 0u ? 0.0f : g - prev_grad[base + i];
            float acc = pre_beta * aux_state[base + i] + (1.0f - pre_beta) * diff;
            aux_state[base + i] = acc;
            prev_grad[base + i] = g;
            transformed = g + pre_alpha * acc;
        }
        float m = mom * momentum[base + i] + transformed;
        momentum[base + i] = m;
        float u = transformed + mom * m;
        x[base + i] = u;
        partial += u * u;
    }
    red[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) red[idx] += red[idx + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / (sqrt(red[0]) * norm_scale + eps);
    for (uint i = lid; i < mat; i += tpg) x[base + i] *= inv;
}

kernel void muown_prepare_f32(
    device const float *param [[buffer(0)]], device const float *grad [[buffer(1)]],
    device float *direction_mom [[buffer(2)]], device const float *direction [[buffer(3)]],
    device float *x [[buffer(4)]], constant uint &N [[buffer(5)]],
    constant uint &rows [[buffer(6)]], constant uint &cols [[buffer(7)]],
    constant float &momentum [[buffer(8)]], constant float &eps [[buffer(9)]],
    uint mid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]])
{
    if (mid >= N) return;
    uint base = mid * rows * cols;
    for (uint col = lid; col < cols; col += tpg) {
        float r2 = 0.0f, w2 = 0.0f, dot = 0.0f;
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            r2 += direction[i] * direction[i];
            w2 += param[i] * param[i];
        }
        float r = max(sqrt(r2), eps);
        float gmag = sqrt(w2);
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            dot += grad[i] * (direction[i] / r);
        }
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            float d = direction[i] / r;
            float grad_r = (gmag / r) * (grad[i] - d * dot);
            float md = momentum * direction_mom[i] + grad_r;
            direction_mom[i] = md;
            x[i] = grad_r + momentum * md;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup float red[1024];
    float partial = 0.0f;
    for (uint i = lid; i < rows * cols; i += tpg) {
        float v = x[base + i]; partial += v * v;
    }
    red[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) red[idx] += red[idx + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / (sqrt(red[0]) + eps);
    for (uint i = lid; i < rows * cols; i += tpg) x[base + i] *= inv;
}

kernel void muown_finalize_f32(
    device float *param [[buffer(0)]], device const float *grad [[buffer(1)]],
    device const float *x [[buffer(2)]], device float *direction [[buffer(3)]],
    device float *mag_m [[buffer(4)]], device float *mag_v [[buffer(5)]],
    device float *ema [[buffer(6)]], device const float *clip_coef [[buffer(7)]],
    constant uint &N [[buffer(8)]], constant uint &rows [[buffer(9)]],
    constant uint &cols [[buffer(10)]], constant float &lr [[buffer(11)]],
    constant float &beta1 [[buffer(12)]], constant float &beta2 [[buffer(13)]],
    constant float &eps [[buffer(14)]], constant float &direction_scale [[buffer(15)]],
    constant float &wd [[buffer(16)]], constant float &bc1 [[buffer(17)]],
    constant float &bc2 [[buffer(18)]], constant float &ema_decay [[buffer(19)]],
    constant uint &do_ema [[buffer(20)]], uint mid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]])
{
    if (mid >= N) return;
    uint base = mid * rows * cols;
    float step_lr = lr * (*clip_coef);
    float dir_alpha = step_lr * direction_scale * sqrt(float(max(rows, cols)));
    for (uint col = lid; col < cols; col += tpg) {
        float r2 = 0.0f, w2 = 0.0f, radial = 0.0f;
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            r2 += direction[i] * direction[i];
            w2 += param[i] * param[i];
        }
        float old_r = max(sqrt(r2), eps);
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            radial += grad[i] * (direction[i] / old_r);
            direction[i] -= dir_alpha * x[i];
        }
        uint si = base + col;
        float mm = beta1 * mag_m[si] + (1.0f - beta1) * radial;
        float vv = beta2 * mag_v[si] + (1.0f - beta2) * radial * radial;
        mag_m[si] = mm; mag_v[si] = vv;
        float magnitude = sqrt(w2) - step_lr * (mm / bc1) / (sqrt(vv / bc2) + eps);
        float new_r2 = 0.0f;
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            new_r2 += direction[i] * direction[i];
        }
        float ratio = magnitude / max(sqrt(new_r2), eps);
        float decay = 1.0f - step_lr * wd;
        for (uint row = 0; row < rows; row++) {
            uint i = base + row * cols + col;
            float p = ratio * direction[i] * decay;
            param[i] = p;
            if (do_ema != 0u) ema[i] = ema_decay * ema[i] + (1.0f - ema_decay) * p;
        }
    }
}

kernel void muon_tensorops_poly_combine_f32(
    device const float *a [[buffer(0)]], device float *a2 [[buffer(1)]],
    constant float &b [[buffer(2)]], constant float &c [[buffer(3)]],
    constant uint &n [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) a2[gid] = b * a[gid] + c * a2[gid];
}

kernel void muon_tensorops_x_combine_f32(
    device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
    constant float &a [[buffer(2)]], constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = a * x[gid] + y[gid];
}

kernel void normuon_row_post_f32(
    device float *x [[buffer(0)]], device float *row_second [[buffer(1)]],
    constant uint &N [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]], constant float &beta2 [[buffer(5)]],
    constant float &eps [[buffer(6)]], uint mid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]])
{
    if (mid >= N) return;
    const uint base = mid * rows * cols;
    threadgroup float red[1024];
    float before = 0.0f;
    for (uint i = lid; i < rows * cols; i += tpg) {
        float v = x[base + i];
        before += v * v;
    }
    red[lid] = before;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) red[idx] += red[idx + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm_before = sqrt(red[0]);
    // Native banks are [in,out], the transpose of Python optimizer layout.
    // A Python "row" is therefore a native column.
    for (uint col = lid; col < cols; col += tpg) {
        float row_sum = 0.0f;
        for (uint row = 0; row < rows; row++) {
            float v = x[base + row * cols + col];
            row_sum += v * v;
        }
        uint state_index = base + col;
        float second = beta2 * row_second[state_index]
                     + (1.0f - beta2) * (row_sum / float(rows));
        row_second[state_index] = second;
        float inv = 1.0f / (sqrt(second) + eps);
        for (uint row = 0; row < rows; row++) x[base + row * cols + col] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float after = 0.0f;
    for (uint i = lid; i < rows * cols; i += tpg) {
        float v = x[base + i];
        after += v * v;
    }
    red[lid] = after;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) red[idx] += red[idx + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float renorm = norm_before / (sqrt(red[0]) + eps);
    for (uint i = lid; i < rows * cols; i += tpg) x[base + i] *= renorm;
}

kernel void muon_tensorops_finalize_f32(
    device float *param [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *ema [[buffer(2)]], device const float *clip_coef [[buffer(3)]],
    constant float &lr [[buffer(4)]], constant float &wd [[buffer(5)]],
    constant float &scale [[buffer(6)]], constant float &ema_decay [[buffer(7)]],
    constant uint &n [[buffer(8)]], constant uint &do_ema [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float clip = *clip_coef;
    float p = param[gid] * (1.0f - lr * wd * clip)
            - lr * scale * clip * x[gid];
    param[gid] = p;
    if (do_ema) ema[gid] = ema_decay * ema[gid] + (1.0f - ema_decay) * p;
}

// ---------------------------------------------------------------------------
// Whole-bank Muon NS5 (one dispatch per bank).
// Bank layout [N, R, C] Burn [in, out]; scale = sqrt(max(1, C/R)).
// NS coeffs: a=3.4445, b=-4.7750, c=2.0315.
// Scratch per matrix: R*C + p*q + 2*p*p  (p=min(R,C), q=max(R,C)).
//
// P0b: inner A=XXᵀ / A² / B·X use simdgroup_matrix 8×8 when p,q % 8 == 0
// (sota banks). Awkward sizes keep the serial-k hand loop. Still one TG per
// matrix → 4 bank dispatches/step.
// ---------------------------------------------------------------------------

/// C[M,N] = A[M,K] @ B[K,N] (NN), 8×8 simdgroup tiles. M,N,K must be % 8 == 0.
inline void ns5_sg_gemm_nn(
    device const float *A,
    device const float *B,
    device float *C,
    uint M,
    uint N,
    uint K,
    uint sid,
    uint n_sg)
{
    constexpr uint TM = 8;
    const uint tiles_m = M / TM;
    const uint tiles_n = N / TM;
    const uint tiles = tiles_m * tiles_n;
    for (uint t = sid; t < tiles; t += n_sg) {
        const uint ti = t / tiles_n;
        const uint tj = t % tiles_n;
        const uint row0 = ti * TM;
        const uint col0 = tj * TM;
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, TM, TM>(0.0f);
        for (uint k0 = 0; k0 < K; k0 += TM) {
            simdgroup_float8x8 a_tile;
            simdgroup_float8x8 b_tile;
            simdgroup_load(a_tile, A + row0 * K + k0, K, ulong2(0, 0), false);
            simdgroup_load(b_tile, B + k0 * N + col0, N, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        simdgroup_store(acc, C + row0 * N + col0, N, ulong2(0, 0), false);
    }
}

/// A[p,p] = X[p,q] @ Xᵀ  (XXᵀ). p,q must be % 8 == 0.
inline void ns5_sg_xxt(
    device const float *X,
    device float *A,
    uint p,
    uint q,
    uint sid,
    uint n_sg)
{
    constexpr uint TM = 8;
    const uint n_tile = p / TM;
    const uint tiles = n_tile * n_tile;
    for (uint t = sid; t < tiles; t += n_sg) {
        const uint ti = t / n_tile;
        const uint tj = t % n_tile;
        const uint row0 = ti * TM;
        const uint col0 = tj * TM;
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, TM, TM>(0.0f);
        for (uint k0 = 0; k0 < q; k0 += TM) {
            simdgroup_float8x8 a_tile;
            simdgroup_float8x8 b_tile;
            simdgroup_load(a_tile, X + row0 * q + k0, q, ulong2(0, 0), false);
            simdgroup_load(b_tile, X + col0 * q + k0, q, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        simdgroup_store(acc, A + row0 * p + col0, p, ulong2(0, 0), false);
    }
}

/// One Newton–Schulz step into x_buf (uses a_buf/b_buf/go_buf as scratch).
inline void ns5_step_cooperative(
    device float *x_buf,
    device float *go_buf,
    device float *a_buf,
    device float *b_buf,
    uint p,
    uint q,
    float ns_a,
    float ns_b,
    float ns_c,
    uint lid,
    uint tpg,
    uint sid,
    uint n_sg,
    bool use_sg)
{
    if (use_sg) {
        ns5_sg_xxt(x_buf, a_buf, p, q, sid, n_sg);
        threadgroup_barrier(mem_flags::mem_device);
        // b_buf ← A², then b ← ns_b*A + ns_c*A²
        ns5_sg_gemm_nn(a_buf, a_buf, b_buf, p, p, p, sid, n_sg);
        threadgroup_barrier(mem_flags::mem_device);
        for (uint idx = lid; idx < p * p; idx += tpg) {
            b_buf[idx] = ns_b * a_buf[idx] + ns_c * b_buf[idx];
        }
        threadgroup_barrier(mem_flags::mem_device);
        // go ← B @ X ; then x ← ns_a * X + go
        ns5_sg_gemm_nn(b_buf, x_buf, go_buf, p, q, p, sid, n_sg);
        threadgroup_barrier(mem_flags::mem_device);
        for (uint idx = lid; idx < p * q; idx += tpg) {
            x_buf[idx] = ns_a * x_buf[idx] + go_buf[idx];
        }
        threadgroup_barrier(mem_flags::mem_device);
    } else {
        for (uint idx = lid; idx < p * p; idx += tpg) {
            uint i = idx / p;
            uint j = idx % p;
            float sum = 0.0f;
            for (uint k = 0; k < q; k++) {
                sum += x_buf[i * q + k] * x_buf[j * q + k];
            }
            a_buf[idx] = sum;
        }
        threadgroup_barrier(mem_flags::mem_device);

        for (uint idx = lid; idx < p * p; idx += tpg) {
            uint i = idx / p;
            uint j = idx % p;
            float aij = a_buf[idx];
            float a2 = 0.0f;
            for (uint k = 0; k < p; k++) {
                a2 += a_buf[i * p + k] * a_buf[k * p + j];
            }
            b_buf[idx] = ns_b * aij + ns_c * a2;
        }
        threadgroup_barrier(mem_flags::mem_device);

        for (uint idx = lid; idx < p * q; idx += tpg) {
            uint i = idx / q;
            uint j = idx % q;
            float bx = 0.0f;
            for (uint k = 0; k < p; k++) {
                bx += b_buf[i * p + k] * x_buf[k * q + j];
            }
            go_buf[idx] = ns_a * x_buf[idx] + bx;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint idx = lid; idx < p * q; idx += tpg) {
            x_buf[idx] = go_buf[idx];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

inline void muon_step_coefficients(
    uint step, uint orth_kind, float ns_a, float ns_b, float ns_c,
    thread float &a, thread float &b, thread float &c)
{
    a = ns_a; b = ns_b; c = ns_c;
    if (orth_kind == 0u) return;
    if (step == 0u) { a = 8.156554f; b = -22.483293f; c = 15.878770f; }
    else if (step == 1u) { a = 4.042930f; b = -2.808917f; c = 0.500018f; }
    else if (step == 2u) { a = 3.891668f; b = -2.772484f; c = 0.506065f; }
    else if (step == 3u) { a = 3.285754f; b = -2.368129f; c = 0.464490f; }
    else { a = 2.346541f; b = -1.709783f; c = 0.423236f; }
}

kernel void muon_bank_ns5_f32(
    device float *param [[buffer(0)]],
    device const float *grad [[buffer(1)]],
    device float *momentum [[buffer(2)]],
    device float *scratch [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &R [[buffer(5)]],
    constant uint &C [[buffer(6)]],
    constant float &lr [[buffer(7)]],
    constant float &mom [[buffer(8)]],
    constant float &wd [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    constant float &ns_eps [[buffer(11)]],
    constant uint &ns_steps [[buffer(12)]],
    constant float &ns_a [[buffer(13)]],
    constant float &ns_b [[buffer(14)]],
    constant float &ns_c [[buffer(15)]],
    device const float *clip_coef [[buffer(16)]],
    constant uint &enable_sg [[buffer(17)]],
    constant uint &orth_kind [[buffer(18)]],
    uint mid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]])
{
    if (mid >= N) return;

    const uint mat = R * C;
    const uint base = mid * mat;
    const uint p = (R > C) ? C : R;
    const uint q = (R > C) ? R : C;
    const uint scratch_stride = mat + p * q + 2u * p * p;
    device float *sbase = scratch + mid * scratch_stride;
    device float *go_buf = sbase;
    device float *x_buf = sbase + mat;
    device float *a_buf = x_buf + p * q;
    device float *b_buf = a_buf + p * p;
    const uint n_sg = max(1u, tpg / 32u);
    // Host METAL_NATIVE_MUON_SG=1 opts into simdgroup path (default off — latency).
    const bool use_sg = enable_sg
        && ((p & 7u) == 0u) && ((q & 7u) == 0u) && (p > 0) && (q > 0);

    for (uint i = lid; i < mat; i += tpg) {
        float g = grad[base + i];
        float buf = momentum[base + i];
        buf = mom * buf + g;
        momentum[base + i] = buf;
        go_buf[i] = g + mom * buf;
    }
    threadgroup_barrier(mem_flags::mem_device);

    const bool needs_t = R > C;

    for (uint i = lid; i < p * q; i += tpg) {
        uint rr = i / q;
        uint cc = i % q;
        float v = needs_t ? go_buf[cc * C + rr] : go_buf[rr * C + cc];
        x_buf[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float red[1024];
    float partial = 0.0f;
    for (uint i = lid; i < p * q; i += tpg) {
        float v = x_buf[i];
        partial += v * v;
    }
    red[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) {
            red[idx] += red[idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_norm = 1.0f / (sqrt(red[0]) * (orth_kind == 1u ? 1.02f : 1.0f) + ns_eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < p * q; i += tpg) {
        x_buf[i] *= inv_norm;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint step = 0; step < ns_steps; step++) {
        float a, b, c;
        muon_step_coefficients(step, orth_kind, ns_a, ns_b, ns_c, a, b, c);
        ns5_step_cooperative(
            x_buf, go_buf, a_buf, b_buf, p, q, a, b, c, lid, tpg, sid, n_sg,
            use_sg);
    }

    for (uint i = lid; i < mat; i += tpg) {
        uint rr = i / C;
        uint cc = i % C;
        float v = needs_t ? x_buf[cc * q + rr] : x_buf[rr * q + cc];
        go_buf[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Multiply update AND weight-decay by the on-device global clip coefficient.
    // NS5 unit-normalizes the update, so without α*=c Muon ignores grad clip.
    // Without also scaling WD, chronic clipping (metal gnorm usually > max_norm)
    // applies full WD while updates are throttled → banks shrink and Adam
    // residual scales explode (~step 2500+). When unclipped, coef==1 (no change).
    float c = *clip_coef;
    float alpha = lr * scale * c;
    float decay = 1.0f - lr * wd * c;
    for (uint i = lid; i < mat; i += tpg) {
        float pv = param[base + i];
        param[base + i] = pv * decay - alpha * go_buf[i];
    }
}

/// Muon NS5 + EMA fused into the same dispatch (update-then-EMA order).
kernel void muon_bank_ns5_ema_f32(
    device float *param [[buffer(0)]],
    device const float *grad [[buffer(1)]],
    device float *momentum [[buffer(2)]],
    device float *scratch [[buffer(3)]],
    device float *ema [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    constant uint &R [[buffer(6)]],
    constant uint &C [[buffer(7)]],
    constant float &lr [[buffer(8)]],
    constant float &mom [[buffer(9)]],
    constant float &wd [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    constant float &ns_eps [[buffer(12)]],
    constant uint &ns_steps [[buffer(13)]],
    constant float &ns_a [[buffer(14)]],
    constant float &ns_b [[buffer(15)]],
    constant float &ns_c [[buffer(16)]],
    constant float &ema_decay [[buffer(17)]],
    device const float *clip_coef [[buffer(18)]],
    constant uint &enable_sg [[buffer(19)]],
    constant uint &orth_kind [[buffer(20)]],
    uint mid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]])
{
    if (mid >= N) return;

    const uint mat = R * C;
    const uint base = mid * mat;
    const uint p = (R > C) ? C : R;
    const uint q = (R > C) ? R : C;
    const uint scratch_stride = mat + p * q + 2u * p * p;
    device float *sbase = scratch + mid * scratch_stride;
    device float *go_buf = sbase;
    device float *x_buf = sbase + mat;
    device float *a_buf = x_buf + p * q;
    device float *b_buf = a_buf + p * p;
    const uint n_sg = max(1u, tpg / 32u);
    // Host METAL_NATIVE_MUON_SG=1 opts into simdgroup path (default off — latency).
    const bool use_sg = enable_sg
        && ((p & 7u) == 0u) && ((q & 7u) == 0u) && (p > 0) && (q > 0);

    for (uint i = lid; i < mat; i += tpg) {
        float g = grad[base + i];
        float buf = momentum[base + i];
        buf = mom * buf + g;
        momentum[base + i] = buf;
        go_buf[i] = g + mom * buf;
    }
    threadgroup_barrier(mem_flags::mem_device);

    const bool needs_t = R > C;

    for (uint i = lid; i < p * q; i += tpg) {
        uint rr = i / q;
        uint cc = i % q;
        float v = needs_t ? go_buf[cc * C + rr] : go_buf[rr * C + cc];
        x_buf[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float red[1024];
    float partial = 0.0f;
    for (uint i = lid; i < p * q; i += tpg) {
        float v = x_buf[i];
        partial += v * v;
    }
    red[lid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < tpg; stride <<= 1) {
        uint idx = lid * (stride << 1);
        if (idx + stride < tpg) {
            red[idx] += red[idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_norm = 1.0f / (sqrt(red[0]) * (orth_kind == 1u ? 1.02f : 1.0f) + ns_eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < p * q; i += tpg) {
        x_buf[i] *= inv_norm;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint step = 0; step < ns_steps; step++) {
        float a, b, c;
        muon_step_coefficients(step, orth_kind, ns_a, ns_b, ns_c, a, b, c);
        ns5_step_cooperative(
            x_buf, go_buf, a_buf, b_buf, p, q, a, b, c, lid, tpg, sid, n_sg,
            use_sg);
    }

    for (uint i = lid; i < mat; i += tpg) {
        uint rr = i / C;
        uint cc = i % C;
        float v = needs_t ? x_buf[cc * q + rr] : x_buf[rr * q + cc];
        go_buf[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);

    float c = *clip_coef;
    float alpha = lr * scale * c;
    float decay = 1.0f - lr * wd * c;
    for (uint i = lid; i < mat; i += tpg) {
        float pv = param[base + i] * decay - alpha * go_buf[i];
        param[base + i] = pv;
        ema[base + i] = ema_decay * ema[base + i] + (1.0f - ema_decay) * pv;
    }
}
