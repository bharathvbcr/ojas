//! K11's CUDA-C module: AdamW in place over one bank window, and the squared
//! norm's per-chunk partials. [`crate::k11_host`] is the bit-for-bit host
//! emulation of both; the order of operations here is that file's, operation
//! for operation.
//!
//! Every float operation is an explicitly rounded intrinsic (`__fmul_rn`,
//! `__fadd_rn`, `__fsub_rn`, `__fdiv_rn`, `__fsqrt_rn`) or an explicit `fmaf`,
//! so the result does not rest on `--fmad=false` alone. Each AdamW element has
//! one writer. The norm reduces each block with the crate's one fixed block
//! reduction, `qd_block_sum` (`crate::small_common`, whose host mirror
//! `block_sum` the emulation calls); there are no atomics, and the grid is a
//! function of the window length only.

use crate::kernels::KernelModule;

/// Threads per block of the squared-norm kernel ([`crate::k11_host::SQ_THREADS`]).
pub const SQ_BLOCK: u32 = 256;

/// The module.
pub const K11: KernelModule = KernelModule {
    name: "k11_adamw",
    source: K11_SOURCE,
    entries: &["qd_adamw_window_f32", "qd_sq_partials_f32"],
};

const K11_SOURCE: &str = concat!(
    crate::device_prelude!(),
    crate::small_common::small_prelude!(),
    r#"
// torch single-tensor AdamW on bank[off, off + n), tessl's kernel order
// (qwen35_adamw.metal:58-66). Scalars are formed on the host in f64 and passed
// as f32 (k11_host.rs, entry_scalars / step_scalars).
extern "C" __global__ void qd_adamw_window_f32(
    float* p, const float* g, float* m, float* v,
    unsigned long long off, unsigned long long n,
    float decay_mul, float step_size, float bc2_sqrt,
    float lerp_w, float beta2, float one_minus_beta2, float eps, float grad_scale)
{
    const float neg_step = -step_size;
    const float one_minus_lerp = __fsub_rn(1.0f, lerp_w);
    QD_GRID_STRIDE(j, n) {
        const unsigned long long i = off + j;
        const float w = __fmul_rn(p[i], decay_mul);
        const float gi = __fmul_rn(g[i], grad_scale);
        const float m0 = m[i];
        const float d = __fsub_rn(gi, m0);
        const float mi = (lerp_w < 0.5f)
            ? __fadd_rn(m0, __fmul_rn(lerp_w, d))
            : __fsub_rn(gi, __fmul_rn(d, one_minus_lerp));
        const float vi = __fadd_rn(__fmul_rn(v[i], beta2),
                                   __fmul_rn(__fmul_rn(one_minus_beta2, gi), gi));
        const float denom = __fadd_rn(__fdiv_rn(__fsqrt_rn(vi), bc2_sqrt), eps);
        p[i] = __fadd_rn(w, __fmul_rn(neg_step, __fdiv_rn(mi, denom)));
        m[i] = mi;
        v[i] = vi;
    }
}

// The squared norm's partials over g[off, off + n): one 256-thread block per
// 4096-element chunk (blocks stride past the grid; every thread of a block
// walks the same chunks, so qd_block_sum's barriers are uniform). Thread t
// sums x*x by fmaf over elements t, t + 256, ... of its chunk from +0; the
// block then folds the 256 values with the crate's one block reduction,
// qd_block_sum (src/small_common.rs: warp butterfly, then a butterfly over
// the 8 warp sums). partials[part_off + c] is chunk c's sum.
extern "C" __global__ void qd_sq_partials_f32(
    const float* g, unsigned long long off, unsigned long long n,
    float* partials, unsigned long long part_off)
{
    __shared__ float scratch[32];
    const unsigned long long chunks = (n + 4095ull) / 4096ull;
    for (unsigned long long c = blockIdx.x; c < chunks; c += gridDim.x) {
        const unsigned long long base = c * 4096ull;
        float s = 0.0f;
        for (unsigned int j = 0; j < 16u; ++j) {
            const unsigned long long k = base + (unsigned long long)j * 256ull + threadIdx.x;
            if (k < n) {
                const float x = g[off + k];
                s = fmaf(x, x, s);
            }
        }
        const float total = qd_block_sum(s, scratch);
        if (threadIdx.x == 0) {
            partials[part_off + c] = total;
        }
    }
}
"#
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k11_host::{SQ_CHUNK, SQ_PER_THREAD, SQ_THREADS};

    #[test]
    fn the_module_defines_its_entries_and_nothing_unsafe() {
        for e in K11.entries {
            assert_eq!(
                K11.source
                    .matches(&format!("extern \"C\" __global__ void {e}("))
                    .count(),
                1,
                "{e}"
            );
        }
        assert_eq!(K11.source.matches("__global__").count(), K11.entries.len());
        for bad in [
            "atomicAdd",
            "#include",
            "\0",
            "smCount",
            "multiProcessorCount",
        ] {
            assert!(!K11.source.contains(bad), "{bad:?}");
        }
    }

    #[test]
    fn the_kernel_constants_are_the_host_emulations() {
        assert_eq!(SQ_BLOCK as usize, SQ_THREADS);
        assert!(K11.source.contains(&format!("{SQ_CHUNK}ull")));
        assert!(K11.source.contains(&format!("j < {SQ_PER_THREAD}u")));
        // The block fold is small_common's, spliced once, not a private tree.
        assert!(K11.source.contains("qd_block_sum(s, scratch)"));
        assert_eq!(
            K11.source
                .matches("__device__ __forceinline__ float qd_block_sum(")
                .count(),
            1
        );
        assert!(
            !K11.source.contains("stride >>= 1") && !K11.source.contains("lane[256]"),
            "no private reduction tree"
        );
    }

    /// The AdamW body uses only rounded intrinsics: no bare `*`, `+`, `/`
    /// between floats that a compiler could contract or reassociate.
    #[test]
    fn the_adamw_body_uses_explicitly_rounded_arithmetic() {
        let start = K11.source.find("qd_adamw_window_f32(").unwrap();
        let end = K11.source.find("// The squared norm's partials").unwrap();
        let body = &K11.source[start..end];
        for op in [
            "__fmul_rn",
            "__fadd_rn",
            "__fsub_rn",
            "__fdiv_rn",
            "__fsqrt_rn",
        ] {
            assert!(body.contains(op), "{op}");
        }
        assert!(!body.contains("fmaf"), "AdamW has no fused step");
        assert!(!body.contains("sqrtf("), "sqrt must be __fsqrt_rn");
    }
}
