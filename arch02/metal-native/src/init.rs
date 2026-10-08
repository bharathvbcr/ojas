//! Seeded weight init matching Python / CUDA ladder semantics (sota toy).
//!
//! CUDA `run_toy_3070ti` / `train_gpt_sprint_native.py`:
//!   `torch.manual_seed(seed)` → `nn.init.orthogonal_` / `normal_` / fixed scalars.
//! FineWeb `TokenStream` is sequential from shard 0 (seed does **not** shuffle data).
//!
//! Metal-native uses SplitMix64 + Box–Muller + thin-QR (same *structure* as torch
//! orthogonal/normal; not bit-identical to Philox). Use `--golden-init` for the
//! exported seed-1337 golden banks (parity / published Soft arm).

use crate::npy::transpose_last2;
use crate::runtime::GpuRuntime;
use crate::tensor::Tensor;
use crate::weights::{make_rope, BlockWeights, MixerKind, ModelConfig, Weights};
use std::ffi::c_int;
use std::sync::Arc;

unsafe extern "C" {
    fn sgeqrf_(
        m: *const c_int,
        n: *const c_int,
        a: *mut f32,
        lda: *const c_int,
        tau: *mut f32,
        work: *mut f32,
        lwork: *const c_int,
        info: *mut c_int,
    );
    fn sorgqr_(
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        a: *mut f32,
        lda: *const c_int,
        tau: *const f32,
        work: *mut f32,
        lwork: *const c_int,
        info: *mut c_int,
    );
}

/// SplitMix64 — fast deterministic stream for init (not torch Philox).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u64() >> 11; // 53 bits → keep 24 for f32
        (u as f32) * (1.0 / ((1u64 << 53) as f32))
    }

    /// Standard normal via Box–Muller.
    pub fn next_normal(&mut self) -> f32 {
        loop {
            let u1 = self.next_f32().max(1e-10);
            let u2 = self.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z = r * theta.cos();
            if z.is_finite() {
                return z;
            }
        }
    }

    pub fn fill_normal(&mut self, out: &mut [f32], std: f32) {
        for v in out.iter_mut() {
            *v = self.next_normal() * std;
        }
    }
}

/// Accelerate/LAPACK thin QR → orthonormal columns of A[m,n], column-major.
/// `sgeqrf` and `sorgqr` replace the scalar O(m*n^2) MGS path that is
/// prohibitive for the 128M banks.
fn thin_qr_lapack(mut a: Vec<f32>, m: usize, n: usize) -> Vec<f32> {
    assert!(m >= n && n > 0);
    assert_eq!(a.len(), m * n);
    let m_i = c_int::try_from(m).expect("QR m exceeds LAPACK LP64");
    let n_i = c_int::try_from(n).expect("QR n exceeds LAPACK LP64");
    let lda = m_i;
    let mut tau = vec![0.0f32; n];
    let mut info = 0;

    // Workspace query followed by the real factorization.
    let query = -1;
    let mut work_query = 0.0f32;
    unsafe {
        sgeqrf_(
            &m_i,
            &n_i,
            a.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            &mut work_query,
            &query,
            &mut info,
        );
    }
    assert_eq!(info, 0, "LAPACK sgeqrf workspace query failed: {info}");
    let lwork = (work_query.ceil() as usize).max(n).max(1);
    let lwork_i = c_int::try_from(lwork).expect("QR workspace exceeds LAPACK LP64");
    let mut work = vec![0.0f32; lwork];
    unsafe {
        sgeqrf_(
            &m_i,
            &n_i,
            a.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork_i,
            &mut info,
        );
    }
    assert_eq!(info, 0, "LAPACK sgeqrf failed: {info}");
    let signs: Vec<f32> = (0..n)
        .map(|j| if a[j * m + j] < 0.0 { -1.0 } else { 1.0 })
        .collect();

    // SORGQR has its own workspace recommendation.
    let mut org_query = 0.0f32;
    unsafe {
        sorgqr_(
            &m_i,
            &n_i,
            &n_i,
            a.as_mut_ptr(),
            &lda,
            tau.as_ptr(),
            &mut org_query,
            &query,
            &mut info,
        );
    }
    assert_eq!(info, 0, "LAPACK sorgqr workspace query failed: {info}");
    let org_lwork = (org_query.ceil() as usize).max(n).max(1);
    let org_lwork_i = c_int::try_from(org_lwork).expect("Q workspace exceeds LAPACK LP64");
    work.resize(org_lwork, 0.0);
    unsafe {
        sorgqr_(
            &m_i,
            &n_i,
            &n_i,
            a.as_mut_ptr(),
            &lda,
            tau.as_ptr(),
            work.as_mut_ptr(),
            &org_lwork_i,
            &mut info,
        );
    }
    assert_eq!(info, 0, "LAPACK sorgqr failed: {info}");

    // Match torch.nn.init.orthogonal_: multiply Q by sign(diag(R)).
    for j in 0..n {
        let sign = signs[j];
        for i in 0..m {
            a[j * m + i] *= sign;
        }
    }
    a
}

/// `nn.init.orthogonal_(tensor[rows, cols], gain)` — row-major `out` of len rows*cols.
pub fn orthogonal_fill(rng: &mut SplitMix64, rows: usize, cols: usize, gain: f32, out: &mut [f32]) {
    assert_eq!(out.len(), rows * cols);
    let mut flat = vec![0.0f32; rows * cols];
    rng.fill_normal(&mut flat, 1.0);

    // Torch: if rows < cols, QR the transpose (work on [cols, rows] tall).
    let (q_rows, q_cols, transposed) = if rows < cols {
        (cols, rows, true)
    } else {
        (rows, cols, false)
    };

    // Build column-major A [q_rows, q_cols] from row-major flat [rows, cols]
    // (or its transpose when rows < cols).
    let mut a = vec![0.0f32; q_rows * q_cols];
    if transposed {
        // flat is [rows, cols] row-major; we need [cols, rows] = flat^T as input to QR.
        for r in 0..rows {
            for c in 0..cols {
                // column-major A[c, r] = flat[r, c]
                a[r * q_rows + c] = flat[r * cols + c];
            }
        }
    } else {
        for r in 0..rows {
            for c in 0..cols {
                a[c * q_rows + r] = flat[r * cols + c];
            }
        }
    }

    let q = thin_qr_lapack(a, q_rows, q_cols); // [q_rows, q_cols] col-major

    if transposed {
        // q is [cols, rows] col-major → write row-major [rows, cols] = q^T
        for r in 0..rows {
            for c in 0..cols {
                out[r * cols + c] = q[r * q_rows + c] * gain;
            }
        }
    } else {
        for r in 0..rows {
            for c in 0..cols {
                out[r * cols + c] = q[c * q_rows + r] * gain;
            }
        }
    }
}

fn upload(rt: &Arc<GpuRuntime>, shape: &[usize], data: &[f32]) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32_hot(shape)?;
    t.buffer.write_f32(data);
    Ok(t)
}

fn alloc_hot_zeroed(rt: &Arc<GpuRuntime>, shape: &[usize]) -> Result<Tensor, String> {
    let t = rt.alloc_tensor_f32_hot(shape)?;
    t.buffer.zero();
    Ok(t)
}

/// Copy a Python-layout matrix [out,in] into one Burn-layout bank slice
/// [in,out] without allocating or transposing an entire bank on the host.
fn write_burn_bank_matrix(
    bank: &Tensor,
    matrix_index: usize,
    in_dim: usize,
    out_dim: usize,
    py: &[f32],
) {
    assert_eq!(py.len(), in_dim * out_dim);
    let mut dst = bank.buffer.contents_f32();
    let base = matrix_index * in_dim * out_dim;
    for o in 0..out_dim {
        for i in 0..in_dim {
            dst[base + i * out_dim + o] = py[o * in_dim + i];
        }
    }
}

fn upload_py_linear_as_burn(
    rt: &Arc<GpuRuntime>,
    py_shape: &[usize],
    py_row_major: &[f32],
) -> Result<Tensor, String> {
    let mut data = py_row_major.to_vec();
    let mut shape = py_shape.to_vec();
    transpose_last2(&mut data, &mut shape)?;
    upload(rt, &shape, &data)
}

/// Seeded sota-toy init (Python bank layout → Burn `[in,out]` on upload).
pub fn init_weights_seeded(
    rt: &Arc<GpuRuntime>,
    cfg: ModelConfig,
    seed: u64,
) -> Result<Weights, String> {
    let mut rng = SplitMix64::new(seed);
    let n = cfg.num_layers;
    let c = cfg.model_dim;
    let kv = cfg.kv_dim();
    let mlp = cfg.mlp_dim;
    let v = cfg.vocab_size;

    // tok_emb [V, C] normal(0, 0.005)
    let mut tok = vec![0.0f32; v * c];
    rng.fill_normal(&mut tok, 0.005);
    let tok_emb = upload(rt, &[v, c], &tok)?;
    let mut tok_t = tok.clone();
    let mut tok_t_shape = vec![v, c];
    transpose_last2(&mut tok_t, &mut tok_t_shape)?;
    let tok_emb_t = upload(rt, &tok_t_shape, &tok_t)?;

    // bigram: zeros + scale 0.05
    let bigram_emb = upload(
        rt,
        &[cfg.bigram_vocab, cfg.bigram_dim],
        &vec![0.0; cfg.bigram_vocab * cfg.bigram_dim],
    )?;
    let bigram_proj = upload(rt, &[cfg.bigram_dim, c], &vec![0.0; cfg.bigram_dim * c])?;
    let bigram_scale = upload(rt, &[1], &[0.05])?;

    let smear_gate = upload(rt, &[c], &vec![0.0; c])?;

    // ve: embed normal(0, 0.01), proj zeros, scale 0.1, layer scales 1
    let mut ve_e = vec![0.0f32; v * cfg.ve_dim];
    rng.fill_normal(&mut ve_e, 0.01);
    let ve_emb = upload(rt, &[v, cfg.ve_dim], &ve_e)?;
    let ve_proj = upload(rt, &[cfg.ve_dim, kv], &vec![0.0; cfg.ve_dim * kv])?;
    let ve_scale = upload(rt, &[1], &[0.1])?;
    let mut ve_layer_scales = Vec::new();
    for _ in 0..cfg.ve_layers.len() {
        ve_layer_scales.push(upload(rt, &[1], &[1.0])?);
    }

    let n_skip = n / 2; // encoder layers for sota U-net
    let skip_weights = upload(rt, &[n_skip, c], &vec![1.0; n_skip * c])?;

    let n_attn = cfg.mixer_count(MixerKind::Attention);
    let n_mingru = cfg.mixer_count(MixerKind::MinGRU);
    let n_mamba = cfg.mixer_count(MixerKind::Mamba2);

    let (
        qo_bank,
        kv_bank,
        mingru_to_z,
        mingru_to_h,
        mingru_out,
        mingru_v_proj,
        mingru_v0_up,
        mamba_in_proj,
        mamba_conv1d_weight,
        mamba_conv1d_bias,
        mamba_out_proj,
        mamba_a_log,
        mamba_d,
        mamba_dt_bias,
        mamba_norm,
    ) = {
        let qo = if n_attn > 0 {
            let qo_bank = alloc_hot_zeroed(rt, &[2 * n_attn, c, c])?;
            let mut matrix = vec![0.0f32; c * c];
            for i in 0..n_attn {
                orthogonal_fill(&mut rng, c, c, 1.0, &mut matrix);
                write_burn_bank_matrix(&qo_bank, i, c, c, &matrix);
            }
            qo_bank
        } else {
            alloc_hot_zeroed(rt, &[0, c, c])?
        };
        let kv_bank = if n_attn > 0 {
            let kv_bank = alloc_hot_zeroed(rt, &[2 * n_attn, c, kv])?;
            let mut matrix = vec![0.0f32; kv * c];
            for i in 0..2 * n_attn {
                orthogonal_fill(&mut rng, kv, c, 1.0, &mut matrix);
                write_burn_bank_matrix(&kv_bank, i, c, kv, &matrix);
            }
            kv_bank
        } else {
            alloc_hot_zeroed(rt, &[0, c, kv])?
        };

        let (mingru_to_z, mingru_to_h, mingru_out, mingru_v_proj, mingru_v0_up) = if n_mingru > 0 {
            let hid = cfg.mingru_hidden();
            let to_z = alloc_hot_zeroed(rt, &[n_mingru, c, hid])?;
            let mut matrix = vec![0.0f32; hid * c];
            for i in 0..n_mingru {
                orthogonal_fill(&mut rng, hid, c, 1.0, &mut matrix);
                write_burn_bank_matrix(&to_z, i, c, hid, &matrix);
            }
            let to_h = alloc_hot_zeroed(rt, &[n_mingru, c, hid])?;
            for i in 0..n_mingru {
                orthogonal_fill(&mut rng, hid, c, 1.0, &mut matrix);
                write_burn_bank_matrix(&to_h, i, c, hid, &matrix);
            }
            let out = alloc_hot_zeroed(rt, &[n_mingru, hid, c])?;
            matrix.resize(c * hid, 0.0);
            for i in 0..n_mingru {
                orthogonal_fill(&mut rng, c, hid, 1.0, &mut matrix);
                write_burn_bank_matrix(&out, i, hid, c, &matrix);
            }
            let (v_proj, v0_up) = if cfg.value_residual {
                let v_proj = alloc_hot_zeroed(rt, &[n_mingru, c, kv])?;
                matrix.resize(kv * c, 0.0);
                for i in 0..n_mingru {
                    orthogonal_fill(&mut rng, kv, c, 1.0, &mut matrix);
                    write_burn_bank_matrix(&v_proj, i, c, kv, &matrix);
                }
                let v0_up = alloc_hot_zeroed(rt, &[n_mingru, kv, hid])?;
                matrix.resize(hid * kv, 0.0);
                for i in 0..n_mingru {
                    orthogonal_fill(&mut rng, hid, kv, 1.0, &mut matrix);
                    write_burn_bank_matrix(&v0_up, i, kv, hid, &matrix);
                }
                (Some(v_proj), Some(v0_up))
            } else {
                (None, None)
            };
            (Some(to_z), Some(to_h), Some(out), v_proj, v0_up)
        } else {
            (None, None, None, None, None)
        };

        let (
            mamba_in_proj,
            mamba_conv1d_weight,
            mamba_conv1d_bias,
            mamba_out_proj,
            mamba_a_log,
            mamba_d,
            mamba_dt_bias,
            mamba_norm,
        ) = if n_mamba > 0 {
            let d_inner = cfg.mamba_d_inner();
            let n_head = cfg.mamba_n_head();
            let conv_dim = cfg.mamba_conv_dim();
            let in_out = cfg.mamba_in_proj_out();
            let d_conv = cfg.d_conv;

            let in_proj = alloc_hot_zeroed(rt, &[n_mamba, c, in_out])?;
            let mut matrix = vec![0.0f32; in_out * c];
            for i in 0..n_mamba {
                orthogonal_fill(&mut rng, in_out, c, 1.0, &mut matrix);
                write_burn_bank_matrix(&in_proj, i, c, in_out, &matrix);
            }

            // `[n_mamba, conv_dim, d_conv]` matches mamba2_conv1d kernel w(C, K); no linear transpose.
            let conv_w = alloc_hot_zeroed(rt, &[n_mamba, conv_dim, d_conv])?;
            let mut conv_slice = vec![0.0f32; conv_dim * d_conv];
            let mut conv_dst = conv_w.buffer.contents_f32();
            for i in 0..n_mamba {
                rng.fill_normal(&mut conv_slice, 0.02);
                let base = i * conv_dim * d_conv;
                conv_dst[base..base + conv_dim * d_conv].copy_from_slice(&conv_slice);
            }
            drop(conv_dst);
            let conv_b = upload(rt, &[n_mamba, conv_dim], &vec![0.0; n_mamba * conv_dim])?;

            let out_proj = alloc_hot_zeroed(rt, &[n_mamba, d_inner, c])?;
            matrix.resize(c * d_inner, 0.0);
            for i in 0..n_mamba {
                orthogonal_fill(&mut rng, c, d_inner, 1.0, &mut matrix);
                write_burn_bank_matrix(&out_proj, i, d_inner, c, &matrix);
            }

            let mut a_data = vec![0.0f32; n_mamba * n_head];
            for layer in 0..n_mamba {
                for h in 0..n_head {
                    a_data[layer * n_head + h] = ((h + 1) as f32).ln();
                }
            }
            let a_log = upload(rt, &[n_mamba, n_head], &a_data)?;
            let d_param = upload(rt, &[n_mamba, n_head], &vec![1.0; n_mamba * n_head])?;
            let dt_bias = upload(rt, &[n_mamba, n_head], &vec![0.0; n_mamba * n_head])?;
            let norm = upload(rt, &[n_mamba, d_inner], &vec![1.0; n_mamba * d_inner])?;

            (
                Some(in_proj),
                Some(conv_w),
                Some(conv_b),
                Some(out_proj),
                Some(a_log),
                Some(d_param),
                Some(dt_bias),
                Some(norm),
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };

        (
            qo,
            kv_bank,
            mingru_to_z,
            mingru_to_h,
            mingru_out,
            mingru_v_proj,
            mingru_v0_up,
            mamba_in_proj,
            mamba_conv1d_weight,
            mamba_conv1d_bias,
            mamba_out_proj,
            mamba_a_log,
            mamba_d,
            mamba_dt_bias,
            mamba_norm,
        )
    };

    // mlp_up: [n,C,mlp] Burn orthogonal; mlp_down: [n,mlp,C] zeros.
    let mut matrix = vec![0.0f32; mlp * c];
    let mlp_up = alloc_hot_zeroed(rt, &[n, c, mlp])?;
    for i in 0..n {
        orthogonal_fill(&mut rng, mlp, c, 1.0, &mut matrix);
        write_burn_bank_matrix(&mlp_up, i, c, mlp, &matrix);
    }
    let mlp_down = alloc_hot_zeroed(rt, &[n, mlp, c])?;

    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let q_gain = upload(rt, &[cfg.num_heads], &vec![1.5; cfg.num_heads])?;
        let vr_lambda = upload(rt, &[2], &[0.5, 0.5])?;
        let attn_scale = upload(rt, &[c], &vec![1.0; c])?;
        let mlp_scale = upload(rt, &[c], &vec![1.0; c])?;
        // resid_mix [2, C] = stack(ones, zeros)
        let mut mix = vec![0.0f32; 2 * c];
        for j in 0..c {
            mix[j] = 1.0;
        }
        let resid_mix = upload(rt, &[2, c], &mix)?;
        blocks.push(BlockWeights {
            q_gain,
            vr_lambda,
            attn_scale,
            mlp_scale,
            resid_mix,
        });
    }

    let (rope_cos, rope_sin) = make_rope(rt, &cfg)?;

    Ok(Weights {
        cfg,
        tok_emb,
        tok_emb_t,
        bigram_emb,
        bigram_proj,
        bigram_scale,
        smear_gate,
        ve_emb,
        ve_proj,
        ve_scale,
        ve_layer_scales,
        skip_weights,
        qo_bank,
        kv_bank,
        mingru_to_z,
        mingru_to_h,
        mingru_out,
        mingru_v_proj,
        mingru_v0_up,
        mamba_in_proj,
        mamba_conv1d_weight,
        mamba_conv1d_bias,
        mamba_out_proj,
        mamba_a_log,
        mamba_d,
        mamba_dt_bias,
        mamba_norm,
        mlp_up,
        mlp_down,
        blocks,
        rope_cos,
        rope_sin,
        bf16_banks: None,
    })
}

/// Deterministic FineWeb token skip from seed.
///
/// CUDA `TokenStream` always starts at shard 0 / pos 0 (seed only affects init).
/// Metal-native **defaults to a seed-mixed skip** so `--seed` also diversifies
/// batch windows (important with a single FineWeb shard). Set
/// `METAL_NATIVE_DATA_SEED=0` for CUDA-identical data order (skip=0).
pub fn fineweb_token_skip(seed: u64) -> usize {
    match std::env::var("METAL_NATIVE_DATA_SEED").ok().as_deref() {
        Some("0") | Some("false") | Some("FALSE") | Some("no") => 0,
        _ => {
            let z = seed
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0x6A09E667F3BCC909);
            (z % 50_000_000) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_rows_unit_when_square() {
        let mut rng = SplitMix64::new(42);
        let n = 8;
        let mut m = vec![0.0f32; n * n];
        orthogonal_fill(&mut rng, n, n, 1.0, &mut m);
        // rows should be ~unit and ~orthogonal
        for i in 0..n {
            let mut ni = 0.0f32;
            for k in 0..n {
                ni += m[i * n + k] * m[i * n + k];
            }
            assert!((ni - 1.0).abs() < 1e-4, "row {i} norm {ni}");
            for j in (i + 1)..n {
                let mut dot = 0.0f32;
                for k in 0..n {
                    dot += m[i * n + k] * m[j * n + k];
                }
                assert!(dot.abs() < 1e-4, "row {i}·{j}={dot}");
            }
        }
    }

    #[test]
    fn seed_divergence() {
        let mut a = vec![0.0f32; 16];
        let mut b = vec![0.0f32; 16];
        orthogonal_fill(&mut SplitMix64::new(1337), 4, 4, 1.0, &mut a);
        orthogonal_fill(&mut SplitMix64::new(42), 4, 4, 1.0, &mut b);
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    #[test]
    fn fineweb_skip_seed_divergent_unless_disabled() {
        if std::env::var("METAL_NATIVE_DATA_SEED").ok().as_deref() == Some("0") {
            assert_eq!(fineweb_token_skip(42), 0);
        } else {
            assert_ne!(fineweb_token_skip(42), fineweb_token_skip(1337));
            assert!(fineweb_token_skip(42) < 50_000_000);
        }
    }
}
