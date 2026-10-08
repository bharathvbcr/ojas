// XSA backward (paper mode, mixed V).
#include <metal_stdlib>
using namespace metal;

kernel void xsa_bwd_f32(
    device const float *y_pre [[buffer(0)]],   // flash output [B,T,H,D]
    device const float *v [[buffer(1)]],       // mixed V [B,T,Hkv,D]
    device const float *dy [[buffer(2)]],      // d(post-xsa)
    device float *dy_pre [[buffer(3)]],        // d(flash y)
    device float *dv [[buffer(4)]],            // accumulate into dV
    constant uint &B [[buffer(5)]],
    constant uint &T [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &Hkv [[buffer(8)]],
    constant uint &D [[buffer(9)]],
    constant float &eps [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    // One thread per (b, t, hkv)
    if (gid >= B * T * Hkv) return;
    const uint hkv = gid % Hkv;
    const uint tmp = gid / Hkv;
    const uint t = tmp % T;
    const uint b = tmp / T;
    const uint group = H / Hkv;
    const uint d_lim = min(D, 64u);
    const uint v_off = ((b * T + t) * Hkv + hkv) * D;

    float norm_sq = 0.0f;
    for (uint d = 0; d < d_lim; ++d) {
        float vv = v[v_off + d];
        norm_sq += vv * vv;
    }
    float inv_n = rsqrt(norm_sq + eps);
    thread float vn[64];
    for (uint d = 0; d < d_lim; ++d) vn[d] = v[v_off + d] * inv_n;

    thread float dvn_acc[64];
    for (uint d = 0; d < d_lim; ++d) dvn_acc[d] = 0.0f;

    for (uint g = 0; g < group; ++g) {
        const uint h = hkv * group + g;
        const uint y_off = ((b * T + t) * H + h) * D;

        // Forward: proj = y·vn; y' = y - proj*vn
        float proj = 0.0f;
        for (uint d = 0; d < d_lim; ++d) proj += y_pre[y_off + d] * vn[d];

        // dy_pre = (I - vn vn^T) dy
        float dy_dot = 0.0f;
        for (uint d = 0; d < d_lim; ++d) dy_dot += dy[y_off + d] * vn[d];
        for (uint d = 0; d < d_lim; ++d) {
            dy_pre[y_off + d] = dy[y_off + d] - dy_dot * vn[d];
        }

        // Through u = proj * vn with du = -dy:
        // dp = du·vn; dvn += proj*du + y*dp; dy += dp*vn (already in projector path for y)
        // Combined with projector: extra grads into vn from removing proj*vn:
        // dL/dvn from -proj*vn: for fixed proj, -proj * dy; from proj=y·vn: -(y·dy? wait)
        // Let u = (y·vn) vn. du = -dy.
        // dp = du · vn = -dy·vn
        // dvn += proj * du + y * dp = -proj*dy + y*(-dy·vn)
        // dy_from_p += dp * vn  — already handled by projector on y side if we only use dy_pre formula
        // We only need dvn extras:
        for (uint d = 0; d < d_lim; ++d) {
            dvn_acc[d] += -proj * dy[y_off + d] + y_pre[y_off + d] * (-dy_dot);
        }
    }

    // vn = v * inv_n; inv_n = (||v||^2+eps)^(-0.5)
    // d(inv) from dvn·v; dv = inv * dvn + v * d_inv
    float d_inv = 0.0f;
    for (uint d = 0; d < d_lim; ++d) {
        d_inv += dvn_acc[d] * v[v_off + d];
    }
    // inv_n = (s)^{-0.5}, s = norm_sq+eps; d_inv_n / ds = -0.5 * s^{-1.5} = -0.5 * inv^3
    // d_inv above is ∂L/∂inv from vn=v*inv: ∂L/∂inv = dvn·v
    // Actually vn_d = v_d * inv; dv_d += inv * dvn_d; d_inv += dvn_d * v_d
    // Then ds from d_inv: d_inv * (-0.5 * inv^3) into s, and s = sum v^2 + eps → dv += 2v * that
    float ds = d_inv * (-0.5f * inv_n * inv_n * inv_n);
    // dL/dv lanes are unique per (b,t,hkv) thread — no atomics needed.
    for (uint d = 0; d < d_lim; ++d) {
        float g = inv_n * dvn_acc[d] + 2.0f * v[v_off + d] * ds;
        dv[v_off + d] += g;
    }
}
