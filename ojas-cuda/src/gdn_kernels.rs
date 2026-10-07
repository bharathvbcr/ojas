//! K2(i): the CUDA-C source of the GDN training scan at the **published**
//! rule, forward and backward, compiled by NVRTC at run time.
//!
//! A port of tessl's Metal kernel (`tessl/kernels/gdn_train.metal`), term for
//! term. Per head, with the state `S` `[128, Dv]`:
//!
//! ```text
//! q^ = l2norm(q) / sqrt(128),  k^ = l2norm(k)        l2norm(x) = x / sqrt(|x|^2 + 1e-6)
//! S^ = exp(g_t) S_{t-1}
//! u  = v_t - S^^T k^          (the published read: the DECAYED state)
//! S_t = S^ + k^ (beta_t u)^T
//! o_t = S_t^T q^
//! ```
//!
//! **Structure, as tessl's:** one 128-thread block per (16-column value
//! slice, sequence × head); thread `i` owns row `i` of its state slice; the
//! forward saves the state entering every 64th token (the checkpoint) and
//! nothing else; the backward walks the chunks in reverse, recomputing each
//! chunk's states from its checkpoint into a per-block scratch, then runs the
//! reverse-mode recurrence over it; per-slice partials of `dq`, `dk`, `dg`,
//! `dbeta` are summed in slice order by a finish kernel, which also applies
//! the l2norm backward. **Added:** variable-length sequences in one launch,
//! through the `tok_off` / `ck_off` arrays of [`crate::gdn_plan`].
//!
//! **Determinism.** Every output element has one writer; no atomics. Each
//! column sum over the 128 rows is a fixed tree: a warp butterfly
//! (`__shfl_xor_sync` at offsets 16, 8, 4, 2, 1, every lane ending with the
//! same bits because IEEE addition is commutative), then lane 0 of each of
//! the four warps to shared memory, then `(w0 + w1) + (w2 + w3)`: the
//! crate's one copy, `qd_colsum_4warps` in [`crate::small_common`]'s prelude,
//! mirrored by [`crate::small_common::colsum_4warps`]. Metal's
//! `simd_sum` has no specified order, so the CUDA bits are not Metal's; the
//! comparison with Metal is a bound, not bit equality
//! (`cuda-backend-scoping.md` §5.1).
//!
//! **Every float operation in the scan is an explicitly rounded intrinsic**
//! (`__fadd_rn`, `__fsub_rn`, `__fmul_rn`, `__fdiv_rn`, `__fsqrt_rn`), which
//! NVRTC never contracts or reassociates whatever `--fmad` / `--prec-*` say.
//! The decay `exp(g)` is the crate's one device exponential, `qd_exp_nonpos`
//! from [`crate::k8_act`] (L-cuda-M1's prelude, spliced in front of this
//! source): IEEE operations only, bit-identical to
//! [`crate::k8_act::exp_nonpos_f32`] under [`crate::kernels::STRICT_SM90`]
//! (`--fmad=false --ftz=false`, which that prelude's plain `*` relies on),
//! measured by M1 at 0.918 ulp worst on `[-104, 0]` against `f64::exp`
//! (within the 2 ulp of CUDA's `expf` this lane pre-registered). So the
//! kernels' bits are a function of their inputs alone, and
//! [`crate::gdn_host`] reproduces them bit for bit on any IEEE host (the Mac
//! included): the host mirror is both an on-host test of this algorithm and
//! a bitwise oracle for the device. tessl used Metal's `precise::exp` /
//! `precise::rsqrt`; here `rsqrt(x)` is `1 / sqrt(x)`, both correctly rounded.
//!
//! **Precondition `g <= 0`** (a log decay; in the model `g = -exp(A_log)
//! softplus(.)`). A positive `g`, `+inf` included, makes `qd_exp_nonpos`
//! return NaN, so the outputs it reaches are NaN: loud, never silently wrong.
//! A NaN `g` likewise.

use crate::kernels::KernelModule;

/// The forward scan.
pub const GDN_PUBLISHED_FWD: &str = "qd_gdn_published_fwd";
/// The backward scan (per-slice partials, `dv`, `ds0`).
pub const GDN_PUBLISHED_BWD: &str = "qd_gdn_published_bwd";
/// The backward's slice reduction and l2norm backward.
pub const GDN_PUBLISHED_BWD_FINISH: &str = "qd_gdn_published_bwd_finish";

/// K2(i), published rule: one NVRTC module.
pub const GDN_PUBLISHED: KernelModule = KernelModule {
    name: "k2_gdn_published_scan",
    source: GDN_PUBLISHED_SOURCE,
    entries: &[
        GDN_PUBLISHED_FWD,
        GDN_PUBLISHED_BWD,
        GDN_PUBLISHED_BWD_FINISH,
    ],
};

/// Forward flag: `s0` holds the initial state (else zeros).
pub const FLAG_S0: u32 = 1;
/// Forward flag: write the final state to `s_fin`.
pub const FLAG_SFIN: u32 = 2;
/// Backward flag: the forward had `s0`, so `ds0` is written.
pub const FLAG_DS0: u32 = 1;
/// Backward flag: `d_fin` holds the final state's gradient (else zeros).
pub const FLAG_DFIN: u32 = 2;

/// The module's own CUDA-C, after the activation prelude. A macro, so
/// `concat!` can splice it behind `crate::act_prelude!()`.
macro_rules! gdn_published_body {
    () => {
        r#"
#define QD_GDN_DK 128
#define QD_GDN_BV 16
#define QD_GDN_CKPT 64
#define QD_GDN_WARPS 4
#define QD_GDN_L2_EPS 1e-6f

typedef unsigned long long qd_u64;

__device__ __forceinline__ float qd_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float qd_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float qd_mul(float a, float b) { return __fmul_rn(a, b); }

// 1 / sqrt(x + 1e-6): the l2norm factor, both operations correctly rounded.
__device__ __forceinline__ float qd_gdn_rnorm(float sumsq) {
    return __fdiv_rn(1.0f, __fsqrt_rn(qd_add(sumsq, QD_GDN_L2_EPS)));
}

// Column sums over the block's 128 threads are small_common's
// qd_colsum_4warps<N> (spliced in by small_prelude): the warp butterfly at
// offsets 16, 8, 4, 2, 1, then (w0 + w1) + (w2 + w3).

// Forward. Grid (Dv / 16, nseq * H), 128 threads. flags: 1 = s0 given (else
// zeros); 2 = write the final state to sfin. s0 / sfin are null when their
// flag is clear and are then never read or written.
extern "C" __global__ void qd_gdn_published_fwd(
    const float* q, const float* k, const float* v, const float* g, const float* beta,
    const float* s0, float* o, float* sfin, float* ckpt,
    const unsigned int* tok_off, const unsigned int* ck_off,
    unsigned int nseq, unsigned int H, unsigned int Dv, unsigned int flags)
{
    __shared__ float red[2][QD_GDN_WARPS * (QD_GDN_BV + 2)];
    const unsigned int bh = blockIdx.y;
    if (bh >= nseq * H || (blockIdx.x + 1u) * QD_GDN_BV > Dv) return;  // uniform per block
    const unsigned int i = threadIdx.x, warp = i >> 5, lane = i & 31u;
    const unsigned int b = bh / H, h = bh - b * H, j0 = blockIdx.x * QD_GDN_BV;
    const qd_u64 t_base = tok_off[b];
    const unsigned int T = tok_off[b + 1] - tok_off[b];
    const qd_u64 c_base = ck_off[b];
    const qd_u64 nc = ck_off[b + 1] - ck_off[b];
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)QD_GDN_DK));
    const qd_u64 srow = ((qd_u64)bh * QD_GDN_DK + i) * Dv + j0;

    float S[QD_GDN_BV];
    #pragma unroll
    for (int j = 0; j < QD_GDN_BV; ++j) {
        S[j] = (flags & 1u) != 0u ? s0[srow + j] : 0.0f;
    }
    for (unsigned int t = 0; t < T; ++t) {
        if (t % QD_GDN_CKPT == 0u) {
            // The state entering token t, before its decay.
            const qd_u64 c = ((c_base * H + (qd_u64)h * nc + t / QD_GDN_CKPT) * QD_GDN_DK + i) * Dv + j0;
            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                ckpt[c + j] = S[j];
            }
        }
        const qd_u64 r = (t_base + t) * H + h;
        const float qi = q[r * QD_GDN_DK + i];
        const float ki = k[r * QD_GDN_DK + i];
        const float a = qd_exp_nonpos(g[r]);
        const float bt = beta[r];

        float p[QD_GDN_BV + 2], tot[QD_GDN_BV + 2];
        #pragma unroll
        for (int j = 0; j < QD_GDN_BV; ++j) {
            S[j] = qd_mul(S[j], a);
            p[j] = qd_mul(S[j], ki);
        }
        p[QD_GDN_BV] = qd_mul(ki, ki);
        p[QD_GDN_BV + 1] = qd_mul(qi, qi);
        qd_colsum_4warps<QD_GDN_BV + 2>(p, red[0], warp, lane, tot);
        const float rk = qd_gdn_rnorm(tot[QD_GDN_BV]);
        const float rq = qd_gdn_rnorm(tot[QD_GDN_BV + 1]);
        const float kh = qd_mul(ki, rk);
        const float qh = qd_mul(qd_mul(qi, rq), scale);

        float po[QD_GDN_BV], out[QD_GDN_BV];
        #pragma unroll
        for (int j = 0; j < QD_GDN_BV; ++j) {
            const float u = qd_sub(v[r * Dv + j0 + j], qd_mul(rk, tot[j]));
            S[j] = qd_add(S[j], qd_mul(kh, qd_mul(bt, u)));
            po[j] = qd_mul(S[j], qh);
        }
        qd_colsum_4warps<QD_GDN_BV>(po, red[1], warp, lane, out);
        #pragma unroll
        for (int j = 0; j < QD_GDN_BV; ++j) {
            if (i == (unsigned int)j) {
                o[r * Dv + j0 + j] = out[j];
            }
        }
    }
    if ((flags & 2u) != 0u) {
        #pragma unroll
        for (int j = 0; j < QD_GDN_BV; ++j) {
            sfin[srow + j] = S[j];
        }
    }
}

// The reverse-mode recurrence. Grid and block as the forward. `scratch` holds
// 64 * 128 * 16 floats per block (one chunk's states S_{t-1}; row i written
// and read only by thread i). Writes dv in full and, per value slice s, the
// partials dq_part, dk_part [NS, rows, 128] and dg_part, dbeta_part [NS, rows];
// qd_gdn_published_bwd_finish sums them. flags: 1 = the forward had s0 (write
// ds0); 2 = dfin holds the final state's gradient (else zeros). dfin / ds0 are
// null when their flag is clear and are then never read or written.
extern "C" __global__ void qd_gdn_published_bwd(
    const float* q, const float* k, const float* v, const float* g, const float* beta,
    const float* d_o, const float* dfin, const float* ckpt, float* scratch,
    float* dv, float* dq_part, float* dk_part, float* dg_part, float* dbeta_part, float* ds0,
    const unsigned int* tok_off, const unsigned int* ck_off,
    unsigned int nseq, unsigned int H, unsigned int Dv, unsigned int flags, qd_u64 rows)
{
    __shared__ float red[2][QD_GDN_WARPS * (QD_GDN_BV + 2)];
    __shared__ float u_c[QD_GDN_CKPT * QD_GDN_BV];
    __shared__ float rk_c[QD_GDN_CKPT];
    __shared__ float rq_c[QD_GDN_CKPT];
    const unsigned int bh = blockIdx.y;
    if (bh >= nseq * H || (blockIdx.x + 1u) * QD_GDN_BV > Dv) return;  // uniform per block
    const unsigned int i = threadIdx.x, warp = i >> 5, lane = i & 31u;
    const unsigned int ns = Dv / QD_GDN_BV;
    const unsigned int b = bh / H, h = bh - b * H, j0 = blockIdx.x * QD_GDN_BV;
    const qd_u64 t_base = tok_off[b];
    const unsigned int T = tok_off[b + 1] - tok_off[b];
    const qd_u64 c_base = ck_off[b];
    const unsigned int nc = ck_off[b + 1] - ck_off[b];
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)QD_GDN_DK));
    const qd_u64 srow = ((qd_u64)bh * QD_GDN_DK + i) * Dv + j0;
    float* mine = scratch + ((qd_u64)bh * ns + blockIdx.x) * (qd_u64)(QD_GDN_CKPT * QD_GDN_DK * QD_GDN_BV);

    float dS[QD_GDN_BV];
    #pragma unroll
    for (int j = 0; j < QD_GDN_BV; ++j) {
        dS[j] = (flags & 2u) != 0u ? dfin[srow + j] : 0.0f;
    }
    for (unsigned int cc = nc; cc-- > 0u;) {
        const unsigned int t0 = cc * QD_GDN_CKPT;
        const unsigned int t1 = (t0 + QD_GDN_CKPT < T) ? t0 + QD_GDN_CKPT : T;
        // Every thread has finished reading the previous chunk's u_c, rk_c and
        // rq_c before they are rewritten.
        __syncthreads();

        // Recompute the chunk's states from its checkpoint.
        float S[QD_GDN_BV];
        const qd_u64 c = ((c_base * H + (qd_u64)h * nc + cc) * QD_GDN_DK + i) * Dv + j0;
        #pragma unroll
        for (int j = 0; j < QD_GDN_BV; ++j) {
            S[j] = ckpt[c + j];
        }
        for (unsigned int t = t0; t < t1; ++t) {
            const unsigned int lt = t - t0;
            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                mine[((qd_u64)lt * QD_GDN_DK + i) * QD_GDN_BV + j] = S[j];
            }
            const qd_u64 r = (t_base + t) * H + h;
            const float qi = q[r * QD_GDN_DK + i];
            const float ki = k[r * QD_GDN_DK + i];
            const float a = qd_exp_nonpos(g[r]);
            float p[QD_GDN_BV + 2], tot[QD_GDN_BV + 2];
            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                S[j] = qd_mul(S[j], a);
                p[j] = qd_mul(S[j], ki);
            }
            p[QD_GDN_BV] = qd_mul(ki, ki);
            p[QD_GDN_BV + 1] = qd_mul(qi, qi);
            qd_colsum_4warps<QD_GDN_BV + 2>(p, red[lt & 1u], warp, lane, tot);
            const float rk = qd_gdn_rnorm(tot[QD_GDN_BV]);
            const float kh = qd_mul(ki, rk);
            const float bt = beta[r];
            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                const float u = qd_sub(v[r * Dv + j0 + j], qd_mul(rk, tot[j]));
                S[j] = qd_add(S[j], qd_mul(kh, qd_mul(bt, u)));
                if (i == (unsigned int)j) {
                    u_c[lt * QD_GDN_BV + j] = u;
                }
            }
            if (i == 0u) {
                rk_c[lt] = rk;
                rq_c[lt] = qd_gdn_rnorm(tot[QD_GDN_BV + 1]);
            }
        }
        __syncthreads();

        // Reverse over the chunk.
        for (unsigned int t = t1; t-- > t0;) {
            const unsigned int lt = t - t0;
            const qd_u64 r = (t_base + t) * H + h;
            const float a = qd_exp_nonpos(g[r]);
            const float bt = beta[r];
            const float kh = qd_mul(k[r * QD_GDN_DK + i], rk_c[lt]);
            const float qh = qd_mul(qd_mul(q[r * QD_GDN_DK + i], rq_c[lt]), scale);

            float Sh[QD_GDN_BV], delta[QD_GDN_BV], dorow[QD_GDN_BV];
            float dqh = 0.0f;
            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                Sh[j] = qd_mul(a, mine[((qd_u64)lt * QD_GDN_DK + i) * QD_GDN_BV + j]);
                delta[j] = qd_mul(bt, u_c[lt * QD_GDN_BV + j]);
                dorow[j] = d_o[r * Dv + j0 + j];
                // S_t = S^ + k^ delta^T
                dqh = qd_add(dqh, qd_mul(qd_add(Sh[j], qd_mul(kh, delta[j])), dorow[j]));
            }
            float p[QD_GDN_BV], ddelta[QD_GDN_BV];
            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                dS[j] = qd_add(dS[j], qd_mul(qh, dorow[j]));
                p[j] = qd_mul(dS[j], kh);
            }
            qd_colsum_4warps<QD_GDN_BV>(p, red[0], warp, lane, ddelta);

            float dkh = 0.0f, dgp[1] = {0.0f}, dgt[1];
            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                dkh = qd_add(dkh, qd_sub(qd_mul(dS[j], delta[j]), qd_mul(qd_mul(bt, Sh[j]), ddelta[j])));
                const float dSh = qd_sub(dS[j], qd_mul(qd_mul(bt, kh), ddelta[j]));
                dgp[0] = qd_add(dgp[0], qd_mul(dSh, Sh[j]));
                dS[j] = qd_mul(a, dSh);
            }
            qd_colsum_4warps<1>(dgp, red[1], warp, lane, dgt);

            #pragma unroll
            for (int j = 0; j < QD_GDN_BV; ++j) {
                if (i == (unsigned int)j) {
                    dv[r * Dv + j0 + j] = qd_mul(bt, ddelta[j]);
                }
            }
            const qd_u64 part = (qd_u64)blockIdx.x * rows + r;
            dq_part[part * QD_GDN_DK + i] = dqh;
            dk_part[part * QD_GDN_DK + i] = dkh;
            if (i == 0u) {
                float db = 0.0f;
                #pragma unroll
                for (int j = 0; j < QD_GDN_BV; ++j) {
                    db = qd_add(db, qd_mul(ddelta[j], u_c[lt * QD_GDN_BV + j]));
                }
                dbeta_part[part] = db;
                dg_part[part] = dgt[0];
            }
        }
    }
    if ((flags & 1u) != 0u) {
        #pragma unroll
        for (int j = 0; j < QD_GDN_BV; ++j) {
            ds0[srow + j] = dS[j];
        }
    }
}

// Sum the per-slice partials in slice order and apply the l2norm backward:
// with y = x r, r = 1 / sqrt(|x|^2 + eps), dx = r (dy - y (y . dy)); for q
// the incoming gradient is also scaled by 1/sqrt(128). One 128-thread block
// per (token, head) row, striding over rows with a fixed grid.
extern "C" __global__ void qd_gdn_published_bwd_finish(
    const float* q, const float* k,
    const float* dq_part, const float* dk_part, const float* dg_part, const float* dbeta_part,
    float* dq, float* dk, float* dg, float* dbeta, qd_u64 rows, unsigned int ns)
{
    __shared__ float red[2][QD_GDN_WARPS * 2];
    const unsigned int i = threadIdx.x, warp = i >> 5, lane = i & 31u;
    const float scale = __fdiv_rn(1.0f, __fsqrt_rn((float)QD_GDN_DK));
    for (qd_u64 r = blockIdx.x; r < rows; r += gridDim.x) {  // uniform per block
        const qd_u64 at = r * QD_GDN_DK + i;
        float dqh = 0.0f, dkh = 0.0f;
        for (unsigned int s = 0; s < ns; ++s) {
            dqh = qd_add(dqh, dq_part[(qd_u64)s * rows * QD_GDN_DK + at]);
            dkh = qd_add(dkh, dk_part[(qd_u64)s * rows * QD_GDN_DK + at]);
        }
        const float qi = q[at], ki = k[at];
        float p[2] = {qd_mul(qi, qi), qd_mul(ki, ki)}, tot[2];
        qd_colsum_4warps<2>(p, red[0], warp, lane, tot);
        const float rq = qd_gdn_rnorm(tot[0]);
        const float rk = qd_gdn_rnorm(tot[1]);
        const float yq = qd_mul(qi, rq), yk = qd_mul(ki, rk);
        const float dyq = qd_mul(dqh, scale);
        float p2[2] = {qd_mul(yq, dyq), qd_mul(yk, dkh)}, dots[2];
        qd_colsum_4warps<2>(p2, red[1], warp, lane, dots);
        dq[at] = qd_mul(rq, qd_sub(dyq, qd_mul(yq, dots[0])));
        dk[at] = qd_mul(rk, qd_sub(dkh, qd_mul(yk, dots[1])));
        if (i == 0u) {
            float sa = 0.0f, sb = 0.0f;
            for (unsigned int s = 0; s < ns; ++s) {
                sa = qd_add(sa, dg_part[(qd_u64)s * rows + r]);
                sb = qd_add(sb, dbeta_part[(qd_u64)s * rows + r]);
            }
            dg[r] = sa;
            dbeta[r] = sb;
        }
    }
}
"#
    };
}

/// This module's own CUDA-C (everything after the activation prelude).
pub const GDN_PUBLISHED_BODY: &str = gdn_published_body!();

/// The CUDA-C NVRTC compiles: L-cuda-M1's activation prelude
/// (`crate::act_prelude!()`, for `qd_exp_nonpos`), the small-kernel prelude
/// (`small_prelude!()`, for the column sum `qd_colsum_4warps`), then
/// [`GDN_PUBLISHED_BODY`]. No `#include`: NVRTC has no default include path.
pub const GDN_PUBLISHED_SOURCE: &str = concat!(
    crate::act_prelude!(),
    crate::small_common::small_prelude!(),
    gdn_published_body!()
);

#[cfg(test)]
mod tests {
    use super::*;

    fn defined_entries(source: &str) -> Vec<String> {
        source
            .lines()
            .filter_map(|l| l.trim().strip_prefix("extern \"C\" __global__ void "))
            .filter_map(|rest| rest.split('(').next())
            .map(|n| n.trim().to_string())
            .collect()
    }

    #[test]
    fn the_published_module_defines_exactly_its_entries_and_every_name_says_published() {
        let mut defined = defined_entries(GDN_PUBLISHED.source);
        defined.sort();
        let mut listed: Vec<String> = GDN_PUBLISHED
            .entries
            .iter()
            .map(|s| s.to_string())
            .collect();
        listed.sort();
        assert_eq!(defined, listed);
        for e in GDN_PUBLISHED.entries {
            assert!(
                e.contains("published"),
                "rule 9: kernel {e} does not name its rule"
            );
        }
        assert!(GDN_PUBLISHED.name.contains("published"));
    }

    #[test]
    fn the_published_source_has_no_nul_no_include_and_no_atomics() {
        let s = GDN_PUBLISHED.source;
        assert!(!s.contains('\0'));
        assert!(!s.contains("#include"));
        assert!(!s.contains("atomic"));
    }

    /// Every multiply and add in the scan is an explicitly rounded intrinsic,
    /// so its bits do not depend on `--fmad`: no bare `*`/`+`/`-` between
    /// float operands survives outside index arithmetic. Checked by the forms
    /// the recurrence's float lines take; each float statement is a `qd_*`
    /// call.
    #[test]
    fn the_published_float_arithmetic_is_explicitly_rounded() {
        for needle in [
            "S[j] *=", "S[j] +=", "dqh +=", "dkh +=", "dS[j] +=", "a * ", "bt * ", "rk * ", "kh * ",
        ] {
            assert!(
                !GDN_PUBLISHED_BODY.contains(needle),
                "a float operation `{needle}` is not an explicitly rounded intrinsic"
            );
        }
        assert!(!GDN_PUBLISHED.source.contains("expf("));
        assert!(!GDN_PUBLISHED.source.contains("rsqrtf("));
    }

    /// One device exponential in the crate: the decay is k8_act's
    /// `qd_exp_nonpos`, spliced in by the prelude, and this module defines no
    /// exponential of its own.
    #[test]
    fn the_published_decay_uses_the_crates_one_exp() {
        assert!(GDN_PUBLISHED.source.starts_with(crate::k8_act::ACT_PRELUDE));
        assert!(GDN_PUBLISHED.source.ends_with(GDN_PUBLISHED_BODY));
        assert_eq!(GDN_PUBLISHED_BODY.matches("qd_exp_nonpos(g[r])").count(), 3);
        assert!(!GDN_PUBLISHED_BODY.contains("exp(float"));
        assert!(!GDN_PUBLISHED_BODY.contains("qd_exp("));
    }

    /// One column reduction in the crate: the scan sums its columns with
    /// small_common's `qd_colsum_4warps`, spliced in by its prelude, and
    /// defines no warp butterfly of its own.
    #[test]
    fn the_published_column_sum_is_small_commons_one_reduction() {
        let small = crate::small_common::small_prelude!();
        assert!(GDN_PUBLISHED
            .source
            .starts_with(&format!("{}{small}", crate::k8_act::ACT_PRELUDE)));
        assert!(small.contains("void qd_colsum_4warps("));
        assert!(GDN_PUBLISHED_BODY.contains("qd_colsum_4warps<"));
        assert!(!GDN_PUBLISHED_BODY.contains("__shfl"));
        assert!(!GDN_PUBLISHED.source.contains("qd_gdn_warp_sum"));
        assert!(!GDN_PUBLISHED.source.contains("qd_gdn_colsum"));
    }

    /// The span of `source` between the first `from` and the next `to`.
    fn between<'a>(source: &'a str, from: &str, to: &str) -> &'a str {
        let at = source
            .find(from)
            .unwrap_or_else(|| panic!("{from:?} not found"));
        let rest = &source[at..];
        let end = rest
            .find(to)
            .unwrap_or_else(|| panic!("{to:?} not found after {from:?}"));
        &rest[..end]
    }

    /// The published rule, in the source text: the decay `S *= exp(g)` comes
    /// before the read `S^T k` that forms `u`, in the forward and in the
    /// backward's recompute; the checkpoint is written before the decay. A
    /// source edited to the repo rule (read first, then decay) fails here.
    #[test]
    fn the_published_rule_decays_before_the_read_in_the_forward_and_the_recompute() {
        let src = GDN_PUBLISHED_BODY;
        let decay = "S[j] = qd_mul(S[j], a);";
        let read = "p[j] = qd_mul(S[j], ki);";
        let u = "const float u = qd_sub(v[r * Dv + j0 + j], qd_mul(rk, tot[j]));";
        for (kernel, end) in [
            ("void qd_gdn_published_fwd(", "void qd_gdn_published_bwd("),
            ("void qd_gdn_published_bwd(", "// Reverse over the chunk."),
        ] {
            let body = between(src, kernel, end);
            let (d, rd, uu) = (
                body.find(decay)
                    .unwrap_or_else(|| panic!("{kernel}: no decay")),
                body.find(read)
                    .unwrap_or_else(|| panic!("{kernel}: no read")),
                body.find(u)
                    .unwrap_or_else(|| panic!("{kernel}: no correction")),
            );
            assert!(
                d < rd && rd < uu,
                "{kernel}: the read of S precedes its decay (the repo rule)"
            );
            assert_eq!(
                body.matches(decay).count(),
                1,
                "{kernel}: one decay per token"
            );
        }
        let fwd = between(
            src,
            "void qd_gdn_published_fwd(",
            "void qd_gdn_published_bwd(",
        );
        let ck = fwd.find("ckpt[c + j] = S[j];").expect("checkpoint write");
        assert!(
            ck < fwd.find(decay).unwrap(),
            "the checkpoint must be the state before the decay"
        );
        // The backward's reverse loop decays the recomputed S_{t-1} the same way.
        let rev = between(
            src,
            "// Reverse over the chunk.",
            "qd_gdn_published_bwd_finish",
        );
        assert!(rev.contains("Sh[j] = qd_mul(a, mine["));
    }
}
