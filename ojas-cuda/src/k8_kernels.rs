//! K8's CUDA-C module: SwiGLU forward (f32 and bf16 out) and backward, the
//! exact-f32 residual add, and the activation sweep the device checks run.
//! [`crate::k8_plan`] holds the bit-for-bit host references.
//!
//! Each kernel grid-strides over `rows * width` with a 64-bit index (the
//! crate's `QD_GRID_STRIDE`); element `i` is row `i / width`, column `i %
//! width`, and has one writer. Every float operation is an explicitly
//! rounded intrinsic, and SiLU is the crate's one copy
//! (`crate::act_prelude!()`). No output is canonicalised: a NaN is whatever
//! the hardware makes (Fable's NaN ruling; see [`crate::k8_plan`]).

use crate::kernels::KernelModule;

/// The module.
pub const K8: KernelModule = KernelModule {
    name: "k8_swiglu",
    source: K8_SOURCE,
    entries: &[
        "qd_swiglu_f32",
        "qd_swiglu_bf16",
        "qd_swiglu_bwd_f32",
        "qd_swiglu_bwd_shared_f32",
        "qd_residual_add_f32",
        "qd_act_sweep_f32",
    ],
};

const K8_SOURCE: &str = concat!(
    crate::device_prelude!(),
    crate::act_prelude!(),
    r#"
#define QD_WIN(ld, off, r, c) ((r) * (ld) + (off) + (c))

// out = silu(gate) * up (tessl/kernels/qwen35_mlp.metal:23-47).
extern "C" __global__ void qd_swiglu_f32(
    const float* gate, const float* up, float* out,
    unsigned long long rows, unsigned long long width,
    unsigned long long ld_gate, unsigned long long gate_off,
    unsigned long long ld_up, unsigned long long up_off,
    unsigned long long ld_out, unsigned long long out_off)
{
    const unsigned long long total = rows * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / width;
        const unsigned long long c = i - r * width;
        const float g = gate[QD_WIN(ld_gate, gate_off, r, c)];
        const float u = up[QD_WIN(ld_up, up_off, r, c)];
        out[QD_WIN(ld_out, out_off, r, c)] = __fmul_rn(qd_silu(g), u);
    }
}

// The same product, rounded once to bf16 (tessl's qwen35_swiglu_bf16).
extern "C" __global__ void qd_swiglu_bf16(
    const float* gate, const float* up, unsigned short* out,
    unsigned long long rows, unsigned long long width,
    unsigned long long ld_gate, unsigned long long gate_off,
    unsigned long long ld_up, unsigned long long up_off,
    unsigned long long ld_out, unsigned long long out_off)
{
    const unsigned long long total = rows * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / width;
        const unsigned long long c = i - r * width;
        const float g = gate[QD_WIN(ld_gate, gate_off, r, c)];
        const float u = up[QD_WIN(ld_up, up_off, r, c)];
        out[QD_WIN(ld_out, out_off, r, c)] =
            qd_f32_to_bf16_bits(__fmul_rn(qd_silu(g), u));
    }
}

// dgate = dy * up * silu'(gate), left to right; dup = dy * silu(gate)
// (tessl/kernels/qwen35_bwd.metal:232-259).
#define QD_SWIGLU_BWD_BODY(DGATE, DUP)                                            \
    const unsigned long long total = rows * width;                                \
    QD_GRID_STRIDE(i, total) {                                                    \
        const unsigned long long r = i / width;                                   \
        const unsigned long long c = i - r * width;                               \
        const float g = gate[QD_WIN(ld_gate, gate_off, r, c)];                    \
        const float u = up[QD_WIN(ld_up, up_off, r, c)];                          \
        const float d = dy[QD_WIN(ld_dy, dy_off, r, c)];                          \
        DGATE[QD_WIN(ld_dgate, dgate_off, r, c)] =                                \
            __fmul_rn(__fmul_rn(d, u), qd_silu_grad(g));                          \
        DUP[QD_WIN(ld_dup, dup_off, r, c)] = __fmul_rn(d, qd_silu(g));            \
    }

extern "C" __global__ void qd_swiglu_bwd_f32(
    const float* gate, const float* up, const float* dy, float* dgate, float* dup,
    unsigned long long rows, unsigned long long width,
    unsigned long long ld_gate, unsigned long long gate_off,
    unsigned long long ld_up, unsigned long long up_off,
    unsigned long long ld_dy, unsigned long long dy_off,
    unsigned long long ld_dgate, unsigned long long dgate_off,
    unsigned long long ld_dup, unsigned long long dup_off)
{
    QD_SWIGLU_BWD_BODY(dgate, dup)
}

// dgate and dup as two disjoint windows of one buffer (the host plan proves
// the windows disjoint).
extern "C" __global__ void qd_swiglu_bwd_shared_f32(
    const float* gate, const float* up, const float* dy, float* dgu,
    unsigned long long rows, unsigned long long width,
    unsigned long long ld_gate, unsigned long long gate_off,
    unsigned long long ld_up, unsigned long long up_off,
    unsigned long long ld_dy, unsigned long long dy_off,
    unsigned long long ld_dgate, unsigned long long dgate_off,
    unsigned long long ld_dup, unsigned long long dup_off)
{
    QD_SWIGLU_BWD_BODY(dgu, dgu)
}

// resid += y (tessl/kernels/qwen35_mlp.metal:56-70).
extern "C" __global__ void qd_residual_add_f32(
    const float* y, float* resid,
    unsigned long long rows, unsigned long long width,
    unsigned long long ld_y, unsigned long long y_off,
    unsigned long long ld_resid, unsigned long long resid_off)
{
    const unsigned long long total = rows * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / width;
        const unsigned long long c = i - r * width;
        const unsigned long long k = QD_WIN(ld_resid, resid_off, r, c);
        resid[k] = __fadd_rn(resid[k], y[QD_WIN(ld_y, y_off, r, c)]);
    }
}

// out[f * n + i] = f(x[i]) for the seven functions, in k8_plan::ACT_SWEEP's order.
extern "C" __global__ void qd_act_sweep_f32(
    const float* x, float* out, unsigned long long n)
{
    QD_GRID_STRIDE(i, n) {
        const float v = x[i];
        out[0 * n + i] = qd_exp(v);
        out[1 * n + i] = qd_exp_nonpos(v);
        out[2 * n + i] = qd_log(v);
        out[3 * n + i] = qd_softplus(v);
        out[4 * n + i] = qd_sigmoid(v);
        out[5 * n + i] = qd_silu(v);
        out[6 * n + i] = qd_silu_grad(v);
    }
}
"#
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k8_plan::ACT_SWEEP;

    #[test]
    fn the_module_defines_exactly_its_entries() {
        for e in K8.entries {
            assert_eq!(
                K8.source
                    .matches(&format!("extern \"C\" __global__ void {e}("))
                    .count(),
                1,
                "{e}"
            );
        }
        assert_eq!(K8.source.matches("__global__").count(), K8.entries.len());
        for bad in ["atomicAdd", "#include", "\0", "expf(", "logf(", "__expf"] {
            assert!(!K8.source.contains(bad), "{bad:?}");
        }
        // One copy of each prelude.
        assert_eq!(K8.source.matches("#define QD_ACT_PRELUDE").count(), 1);
        assert_eq!(K8.source.matches("#define QD_GRID_STRIDE").count(), 1);
    }

    #[test]
    fn the_sweep_kernel_writes_the_host_sweeps_order() {
        let at = K8.source.find("qd_act_sweep_f32(").unwrap();
        let body = &K8.source[at..];
        let mut last = 0;
        for (k, (name, _)) in ACT_SWEEP.iter().enumerate() {
            let line = format!("out[{k} * n + i] = qd_{name}(v);");
            let pos = body.find(&line).unwrap_or_else(|| panic!("{line}"));
            assert!(pos >= last, "{name} out of order");
            last = pos;
        }
    }

    #[test]
    fn the_backward_multiplies_left_to_right() {
        assert!(K8
            .source
            .contains("__fmul_rn(__fmul_rn(d, u), qd_silu_grad(g))"));
        assert!(K8.source.contains("__fmul_rn(d, qd_silu(g))"));
        assert!(K8.source.contains("__fmul_rn(qd_silu(g), u)"));
    }
}
