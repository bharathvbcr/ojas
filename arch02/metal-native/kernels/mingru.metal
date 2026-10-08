#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// minGRU Forward Pass
// ---------------------------------------------------------------------------
// A linear-time parallel scan equivalent.
// We assign one thread per (batch, hidden_dim) and loop sequentially over T.
// This perfectly avoids inter-threadgroup synchronization and takes advantage
// of the fact that the hidden dimensions are independent in the recurrence.
kernel void mingru_fwd(
    device const float* Z_in [[buffer(0)]],
    device const float* H_in [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& B [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant uint& H [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint h_idx = gid.x;
    if (b >= B || h_idx >= H) return;
    
    float h_prev = 0.0;
    
    for (uint t = 0; t < T; t++) {
        uint idx = b * T * H + t * H + h_idx;
        float z = Z_in[idx];
        float h_in = H_in[idx];
        
        // sigmoid(z)
        float sig_z = 1.0 / (1.0 + exp(-z));
        // a_t = 1 - sigmoid(z)
        float a_t = 1.0 - sig_z;
        
        // g(h_in)
        float g_h = h_in >= 0.0 ? h_in + 0.5 : 1.0 / (1.0 + exp(-h_in));
        
        // b_t = sigmoid(z) * g(h_in)
        float b_t = sig_z * g_h;
        
        // recurrence
        float h_t = a_t * h_prev + b_t;
        out[idx] = h_t;
        h_prev = h_t;
    }
}

// ---------------------------------------------------------------------------
// minGRU Backward Pass (BPTT)
// ---------------------------------------------------------------------------
kernel void mingru_bwd(
    device const float* Z_in [[buffer(0)]],
    device const float* H_in [[buffer(1)]],
    device const float* h_out [[buffer(2)]],
    device const float* grad_h_out [[buffer(3)]],
    device float* grad_Z_in [[buffer(4)]],
    device float* grad_H_in [[buffer(5)]],
    constant uint& B [[buffer(6)]],
    constant uint& T [[buffer(7)]],
    constant uint& H [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint h_idx = gid.x;
    if (b >= B || h_idx >= H) return;
    
    float dh_next = 0.0;
    
    // Backpropagation Through Time
    for (int t = T - 1; t >= 0; t--) {
        uint idx = b * T * H + t * H + h_idx;
        
        float z = Z_in[idx];
        float h_in = H_in[idx];
        
        float sig_z = 1.0 / (1.0 + exp(-z));
        float a_t = 1.0 - sig_z;
        
        // g(h_in)
        float sig_h = 1.0 / (1.0 + exp(-h_in));
        float g_h = h_in >= 0.0 ? h_in + 0.5 : sig_h;
        
        // Accumulate gradient from current output and next state
        float dh_t = grad_h_out[idx] + dh_next;
        
        float h_prev = (t > 0) ? h_out[idx - H] : 0.0;
        
        float da_t = dh_t * h_prev;
        float db_t = dh_t;
        
        // dL / dZ
        float dsig_z_total = db_t * g_h - da_t;
        float dz = dsig_z_total * sig_z * (1.0 - sig_z);
        grad_Z_in[idx] = dz;
        
        // dL / dH
        float dg_h = db_t * sig_z;
        float dg_dh_in = h_in >= 0.0 ? 1.0 : sig_h * (1.0 - sig_h);
        grad_H_in[idx] = dg_h * dg_dh_in;
        
        dh_next = dh_t * a_t;
    }
}
