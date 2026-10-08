#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Mamba-2 (SSD) Forward Pass
// ---------------------------------------------------------------------------
// Sequential O(L) scan. Assigns one thread per (B, H, P) and iterates over T.
// Maintains hidden state `h_t` of size N in thread-local memory.
kernel void mamba2_fwd(
    device const float* x_scaled [[buffer(0)]], // (B, T, H, P)
    device const float* B_h      [[buffer(1)]], // (B, T, N)
    device const float* C_h      [[buffer(2)]], // (B, T, N)
    device const float* log_dA   [[buffer(3)]], // (B, T, H)
    device float* y              [[buffer(4)]], // (B, T, H, P)
    device float* h_states       [[buffer(5)]], // (B, T, H, P, N) - saved for BPTT
    constant uint& B_sz          [[buffer(6)]],
    constant uint& T_sz          [[buffer(7)]],
    constant uint& H_sz          [[buffer(8)]],
    constant uint& P_sz          [[buffer(9)]],
    constant uint& N_sz          [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint p = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (p >= P_sz || h >= H_sz || b >= B_sz) return;
    
    // Mamba-2 typical max d_state is 128
    float h_local[128]; 
    for(uint n = 0; n < N_sz; n++) {
        h_local[n] = 0.0;
    }
    
    for (uint t = 0; t < T_sz; t++) {
        float a_t = exp(log_dA[b * T_sz * H_sz + t * H_sz + h]);
        float x_t = x_scaled[b * T_sz * H_sz * P_sz + t * H_sz * P_sz + h * P_sz + p];
        
        float y_t = 0.0;
        
        for(uint n = 0; n < N_sz; n++) {
            float b_t = B_h[b * T_sz * N_sz + t * N_sz + n];
            float c_t = C_h[b * T_sz * N_sz + t * N_sz + n];
            
            float h_val = a_t * h_local[n] + x_t * b_t;
            h_local[n] = h_val;
            
            h_states[b * T_sz * H_sz * P_sz * N_sz + t * H_sz * P_sz * N_sz + h * P_sz * N_sz + p * N_sz + n] = h_val;
            
            y_t += h_val * c_t;
        }
        
        y[b * T_sz * H_sz * P_sz + t * H_sz * P_sz + h * P_sz + p] = y_t;
    }
}

// ---------------------------------------------------------------------------
// Mamba-2 (SSD) Backward Pass
// ---------------------------------------------------------------------------
// Uses atomic additions for gradients shared across the P (and H) dimensions.
// Requires grad_B_h, grad_C_h, and grad_log_dA to be zero-initialized before launch.
kernel void mamba2_bwd(
    device const float* x_scaled   [[buffer(0)]],
    device const float* B_h        [[buffer(1)]],
    device const float* C_h        [[buffer(2)]],
    device const float* log_dA     [[buffer(3)]],
    device const float* h_states   [[buffer(4)]],
    device const float* grad_y     [[buffer(5)]],
    device float* grad_x_scaled    [[buffer(6)]],
    device atomic<float>* grad_B_h [[buffer(7)]],
    device atomic<float>* grad_C_h [[buffer(8)]],
    device atomic<float>* grad_log_dA [[buffer(9)]],
    constant uint& B_sz            [[buffer(10)]],
    constant uint& T_sz            [[buffer(11)]],
    constant uint& H_sz            [[buffer(12)]],
    constant uint& P_sz            [[buffer(13)]],
    constant uint& N_sz            [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint p = gid.x;
    uint h = gid.y;
    uint b = gid.z;
    if (p >= P_sz || h >= H_sz || b >= B_sz) return;
    
    float dh_next[128];
    for(uint n = 0; n < N_sz; n++) {
        dh_next[n] = 0.0;
    }
    
    for (int t = T_sz - 1; t >= 0; t--) {
        float a_t = exp(log_dA[b * T_sz * H_sz + t * H_sz + h]);
        float x_t = x_scaled[b * T_sz * H_sz * P_sz + t * H_sz * P_sz + h * P_sz + p];
        float dy_t = grad_y[b * T_sz * H_sz * P_sz + t * H_sz * P_sz + h * P_sz + p];
        
        float dx_t = 0.0;
        float da_t_part = 0.0;
        
        for(uint n = 0; n < N_sz; n++) {
            float b_t = B_h[b * T_sz * N_sz + t * N_sz + n];
            float c_t = C_h[b * T_sz * N_sz + t * N_sz + n];
            
            float dh_t = dy_t * c_t + dh_next[n];
            
            float h_prev = (t > 0) ? h_states[b * T_sz * H_sz * P_sz * N_sz + (t-1) * H_sz * P_sz * N_sz + h * P_sz * N_sz + p * N_sz + n] : 0.0;
            float h_curr = h_states[b * T_sz * H_sz * P_sz * N_sz + t * H_sz * P_sz * N_sz + h * P_sz * N_sz + p * N_sz + n];
            
            da_t_part += dh_t * h_prev;
            dx_t += dh_t * b_t;
            
            float db_t_part = dh_t * x_t;
            float dc_t_part = dy_t * h_curr;
            
            // Atomic reductions across H and P
            atomic_fetch_add_explicit(&grad_B_h[b * T_sz * N_sz + t * N_sz + n], db_t_part, memory_order_relaxed);
            atomic_fetch_add_explicit(&grad_C_h[b * T_sz * N_sz + t * N_sz + n], dc_t_part, memory_order_relaxed);
            
            dh_next[n] = dh_t * a_t;
        }
        
        grad_x_scaled[b * T_sz * H_sz * P_sz + t * H_sz * P_sz + h * P_sz + p] = dx_t;
        
        float dlog_dA_part = da_t_part * a_t;
        // Atomic reduction across P
        atomic_fetch_add_explicit(&grad_log_dA[b * T_sz * H_sz + t * H_sz + h], dlog_dA_part, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Mamba-2 Local 1D Convolution (Forward)
// ---------------------------------------------------------------------------
// Computes causal depthwise 1D convolution over sequence length.
// x, y: (B, T, C)
// w: (C, K)
// bias: (C)
// Vectorized depthwise causal Conv1D (K=4): one thread per (b, c_group) streaming T.
kernel void mamba2_conv1d_fwd(
    device const float* x     [[buffer(0)]],
    device const float* w     [[buffer(1)]],
    device const float* bias  [[buffer(2)]],
    device float* y           [[buffer(3)]],
    constant uint& B_sz       [[buffer(4)]],
    constant uint& T_sz       [[buffer(5)]],
    constant uint& C_sz       [[buffer(6)]],
    constant uint& K_sz       [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint b = gid.y;
    if (c >= C_sz || b >= B_sz) return;

    float4 wf = float4(w[c * K_sz + 0], w[c * K_sz + 1], w[c * K_sz + 2], w[c * K_sz + 3]);
    float val = bias[c];
    uint base = b * T_sz * C_sz + c;

    for (uint t = 0; t < T_sz; ++t) {
        float acc = val;
        if (t >= 0) acc += wf.x * x[base + (t - 0) * C_sz];
        if (t >= 1) acc += wf.y * x[base + (t - 1) * C_sz];
        if (t >= 2) acc += wf.z * x[base + (t - 2) * C_sz];
        if (t >= 3) acc += wf.w * x[base + (t - 3) * C_sz];
        y[base + t * C_sz] = acc;
    }
}

kernel void mamba2_conv1d_bwd(
    device const float* x         [[buffer(0)]],
    device const float* w         [[buffer(1)]],
    device const float* grad_y    [[buffer(2)]],
    device float* grad_x          [[buffer(3)]],
    device atomic<float>* grad_w  [[buffer(4)]],
    device atomic<float>* grad_bias [[buffer(5)]],
    constant uint& B_sz           [[buffer(6)]],
    constant uint& T_sz           [[buffer(7)]],
    constant uint& C_sz           [[buffer(8)]],
    constant uint& K_sz           [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint b = gid.y;
    if (c >= C_sz || b >= B_sz) return;

    float4 wf = float4(w[c * K_sz + 0], w[c * K_sz + 1], w[c * K_sz + 2], w[c * K_sz + 3]);
    float4 dw_acc = float4(0.0);
    float db_acc = 0.0;
    uint base = b * T_sz * C_sz + c;

    for (uint t = 0; t < T_sz; ++t) {
        float dy_t = grad_y[base + t * C_sz];
        db_acc += dy_t;

        float dx_val = 0.0;
        if (t + 0 < T_sz) dx_val += wf.x * grad_y[base + (t + 0) * C_sz];
        if (t + 1 < T_sz) dx_val += wf.y * grad_y[base + (t + 1) * C_sz];
        if (t + 2 < T_sz) dx_val += wf.z * grad_y[base + (t + 2) * C_sz];
        if (t + 3 < T_sz) dx_val += wf.w * grad_y[base + (t + 3) * C_sz];
        grad_x[base + t * C_sz] = dx_val;

        if (t >= 0) dw_acc.x += x[base + (t - 0) * C_sz] * dy_t;
        if (t >= 1) dw_acc.y += x[base + (t - 1) * C_sz] * dy_t;
        if (t >= 2) dw_acc.z += x[base + (t - 2) * C_sz] * dy_t;
        if (t >= 3) dw_acc.w += x[base + (t - 3) * C_sz] * dy_t;
    }

    atomic_fetch_add_explicit(&grad_bias[c], db_acc, memory_order_relaxed);
    for (uint k = 0; k < K_sz; ++k) {
        atomic_fetch_add_explicit(&grad_w[c * K_sz + k], dw_acc[k], memory_order_relaxed);
    }
}
