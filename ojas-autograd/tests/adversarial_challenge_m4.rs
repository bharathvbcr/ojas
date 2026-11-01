//! Empirical Adversarial Verification Suite for Milestone 4 (Features 21–25)
//!
//! Evaluates:
//! 1. `gelu_tanh` numerical stability across extreme ranges ([-1000, 1000], near zero, negative tail cancellation resistance), analytical derivative vs f64 central differences.
//! 2. Symmetric sliding window attention (|i - j| <= W) boundary conditions, empty ranges, dead padding queries (i >= len_b), and gradient propagation through tape.
//! 3. Masked sequence mean pooling (`seq_mean_forward`) high dynamic range cancellation guard ([10^7, 1, -10^7]) and zero active length handling.
//! 4. L2 normalization unit sphere invariant | ||y||_2 - 1.0 | <= 1e-6 and Matryoshka prefix slicing angle conservation.
//! 5. Per-head V RMSNorm (`rms_qkv_norm_forward`) multi-head independence and learned weight scaling.
//! 6. Bit-identity of CPU reductions across thread counts and execution modes.
#![allow(clippy::erasing_op, clippy::identity_op, clippy::needless_range_loop)]

use ojas_autograd::Tape;
use ojas_core::{
    gelu_tanh_backward_exact, gelu_tanh_exact, AttentionOpts, Backend, Budget, Numerics, OjasError,
    Tensor,
};
use ojas_cpu::CpuBackend;

fn cpu_backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 30), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn f32t(cpu: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, cpu.budget()).unwrap()
}

fn u32t(cpu: &CpuBackend, data: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(data, shape, cpu.budget()).unwrap()
}

// ----------------------------------------------------------------------------
// CHALLENGE 1: gelu_tanh extreme ranges, negative tail cancellation, analytical vs central diff
// ----------------------------------------------------------------------------

#[test]
fn challenge_1_gelu_tanh_extreme_ranges_and_cancellation() {
    let cpu = cpu_backend(2, Numerics::Exact);

    // 1. Extreme range x in [-1000, 1000] and beyond
    let large_pos = [
        10.0f32,
        15.0,
        30.0,
        100.0,
        500.0,
        1000.0,
        1e5,
        1e10,
        1e20,
        1e38,
        f32::MAX,
    ];
    for &x in &large_pos {
        let y = gelu_tanh_exact(x);
        assert_eq!(y, x, "large positive x must be identity: x = {x}, y = {y}");
        let dy = gelu_tanh_backward_exact(x, 1.0);
        assert_eq!(
            dy, 1.0,
            "large positive slope must be 1.0: x = {x}, dy = {dy}"
        );
        let dy_scale = gelu_tanh_backward_exact(x, 2.5);
        assert_eq!(dy_scale, 2.5, "scaled dy: x = {x}");
    }

    let large_neg = [
        -12.0f32,
        -15.0,
        -20.0,
        -50.0,
        -100.0,
        -500.0,
        -1000.0,
        -1e5,
        -1e10,
        -1e20,
        -1e38,
        f32::MIN,
    ];
    for &x in &large_neg {
        let y = gelu_tanh_exact(x);
        assert_eq!(
            y.to_bits(),
            (-0.0f32).to_bits(),
            "large negative x must be -0.0: x = {x}, y = {y}"
        );
        let dy = gelu_tanh_backward_exact(x, 1.0);
        assert_eq!(
            dy, 0.0,
            "large negative slope must be 0.0: x = {x}, dy = {dy}"
        );
    }

    // 2. Negative tail cancellation resistance (x in [-10.8, -6.0])
    // Torch's 0.5 * x * (1 + tanh(u)) collapses at x <= -8.5 where tanh(u) = -1 in f64.
    // The exact identity x * sigmoid(u) maintains relative precision.
    let sqrt_2_over_pi = 0.797_884_560_802_865_4f64;
    let cubic = 0.044_715f64;
    let test_tail = [-6.0f32, -7.0, -8.0, -8.5, -9.0, -9.5, -10.0, -10.5];
    for &x in &test_tail {
        let xd = f64::from(x);
        let u = 2.0 * sqrt_2_over_pi * (xd + cubic * xd * xd * xd);
        // Exact f64 reference using exp alone
        let s = u.exp() / (1.0 + u.exp());
        let want = xd * s;
        let got = f64::from(gelu_tanh_exact(x));
        assert!(
            got != 0.0,
            "x = {x} must not prematurely collapse to zero (want {want:e})"
        );
        assert!(
            (got - want).abs() <= 1.5e-7 * want.abs() + 1e-43,
            "x = {x}: got {got:e} vs want {want:e}"
        );
    }

    // 3. Near zero and subnormals
    assert_eq!(gelu_tanh_exact(0.0).to_bits(), 0.0f32.to_bits());
    assert_eq!(gelu_tanh_exact(-0.0).to_bits(), (-0.0f32).to_bits());
    assert_eq!(gelu_tanh_backward_exact(0.0, 1.0), 0.5);
    assert_eq!(gelu_tanh_backward_exact(-0.0, 1.0), 0.5);

    // Derivative is negative on the left shoulder of the local minimum (e.g. at x = -1.5)
    let slope_neg = gelu_tanh_backward_exact(-1.5, 1.0);
    assert!(
        slope_neg < -0.12 && slope_neg > -0.14,
        "derivative at x = -1.5 must be negative (~ -0.128): got {slope_neg}"
    );

    // Odd symmetry: gelu(x) - gelu(-x) = x
    let mut x_sym = 1e-5f32;
    while x_sym <= 10.0 {
        let diff = f64::from(gelu_tanh_exact(x_sym)) - f64::from(gelu_tanh_exact(-x_sym));
        assert!(
            (diff - f64::from(x_sym)).abs() <= 2.5e-7 * f64::from(x_sym),
            "odd symmetry failed at x = {x_sym}: diff = {diff}"
        );
        x_sym *= 1.5;
    }

    // 4. Analytical derivative vs f64 central differences
    let h = 1e-5f64;
    let mut x_grid = -12.0f64;
    while x_grid <= 12.0 {
        let gelu_f64 = |val: f64| -> f64 {
            let u = 2.0 * sqrt_2_over_pi * (val + cubic * val * val * val);
            let s = if u >= 0.0 {
                1.0 / (1.0 + (-u).exp())
            } else {
                u.exp() / (1.0 + u.exp())
            };
            val * s
        };
        let want_slope = (gelu_f64(x_grid + h) - gelu_f64(x_grid - h)) / (2.0 * h);
        let got_slope = f64::from(gelu_tanh_backward_exact(x_grid as f32, 1.0));
        assert!(
            (got_slope - want_slope).abs() <= 2e-7 * want_slope.abs() + 2e-8,
            "derivative mismatch at x = {x_grid}: got {got_slope} vs want {want_slope}"
        );
        x_grid += 0.05;
    }

    // 5. Backend Tensor dispatch and NonFinite refusal
    let t_valid = f32t(&cpu, &[1.0, -2.0, 3.0], &[3]);
    let out = cpu.gelu_tanh_forward(&t_valid).unwrap();
    assert_eq!(out.shape(), &[3]);

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let t_bad = f32t(&cpu, &[1.0, bad, 3.0], &[3]);
        assert!(matches!(
            cpu.gelu_tanh_forward(&t_bad),
            Err(OjasError::NonFinite {
                op: "gelu_tanh_forward"
            })
        ));
        assert!(matches!(
            cpu.gelu_tanh_backward(&t_bad, &t_valid),
            Err(OjasError::NonFinite {
                op: "gelu_tanh_backward"
            })
        ));
        assert!(matches!(
            cpu.gelu_tanh_backward(&t_valid, &t_bad),
            Err(OjasError::NonFinite {
                op: "gelu_tanh_backward"
            })
        ));
    }
}

// ----------------------------------------------------------------------------
// CHALLENGE 2: Symmetric sliding window attention boundary conditions & tape
// ----------------------------------------------------------------------------

#[test]
fn challenge_2_sliding_window_attention_boundary_and_autograd() {
    let cpu = cpu_backend(2, Numerics::Exact);

    let (b, h, t, d) = (2usize, 2usize, 6usize, 4usize);
    let radius = 2usize; // |i - j| <= 2
    let lens_data = [3u32, 0u32]; // Batch 0 has 3 valid tokens, Batch 1 has 0 valid tokens
    let key_lens = u32t(&cpu, &lens_data, &[b]);

    let mut q_data = vec![0.0f32; b * h * t * d];
    let mut k_data = vec![0.0f32; b * h * t * d];
    let mut v_data = vec![0.0f32; b * h * t * d];
    for i in 0..q_data.len() {
        q_data[i] = ((i % 17) as f32 + 1.0) * 0.1;
        k_data[i] = ((i % 13) as f32 + 1.0) * 0.1;
        v_data[i] = ((i % 11) as f32 + 1.0) * 0.1;
    }

    let q = f32t(&cpu, &q_data, &[b, h, t, d]);
    let k = f32t(&cpu, &k_data, &[b, h, t, d]);
    let v = f32t(&cpu, &v_data, &[b, h, t, d]);

    let opts = AttentionOpts {
        causal: false,
        radius: Some(radius),
        scale: Some(1.0),
        key_lens: Some(&key_lens),
        ..Default::default()
    };

    let (out, lse) = cpu.attention_forward_saving(&q, &k, &v, opts).unwrap();
    let out_v = out.to_f32_vec().unwrap();
    let lse_v = lse.to_f32_vec().unwrap();

    // Verify dead padding queries produce exactly 0.0
    // Batch 0 has active tokens 0..3, padding tokens 3..6
    for head in 0..h {
        for row in 3..t {
            let offset = ((0 * h + head) * t + row) * d;
            for c in 0..d {
                assert_eq!(
                    out_v[offset + c],
                    0.0,
                    "Batch 0 padding query {row} col {c} must be 0.0"
                );
            }
            let lse_idx = (0 * h + head) * t + row;
            assert_eq!(
                lse_v[lse_idx], 0.0,
                "Batch 0 padding query {row} lse must be 0.0"
            );
        }
    }

    // Batch 1 has 0 active tokens: all queries 0..6 must produce exactly 0.0
    for head in 0..h {
        for row in 0..t {
            let offset = ((1 * h + head) * t + row) * d;
            for c in 0..d {
                assert_eq!(
                    out_v[offset + c],
                    0.0,
                    "Batch 1 query {row} col {c} must be 0.0"
                );
            }
            let lse_idx = (1 * h + head) * t + row;
            assert_eq!(lse_v[lse_idx], 0.0, "Batch 1 query {row} lse must be 0.0");
        }
    }

    // Autograd Tape verification: test gradient propagation through sliding window
    let mut tape = Tape::new(cpu.clone());
    let var_q = tape.leaf(q.clone()).unwrap();
    let var_k = tape.leaf(k.clone()).unwrap();
    let var_v = tape.leaf(v.clone()).unwrap();

    let var_attn = tape
        .attention_with(
            var_q,
            var_k,
            var_v,
            None,
            Some(key_lens.clone()),
            false,
            None,
            Some(1.0),
            Some(radius),
        )
        .unwrap();

    // Loss is sum of outputs
    let ones = f32t(&cpu, &vec![1.0; b * h * t * d], &[b, h, t, d]);
    let var_ones = tape.leaf(ones).unwrap();
    let var_prod = tape.mul(var_attn, var_ones).unwrap();

    tape.backward(var_prod).unwrap();

    let grad_q = tape.grad(var_q).unwrap().to_f32_vec().unwrap();
    let grad_k = tape.grad(var_k).unwrap().to_f32_vec().unwrap();
    let grad_v = tape.grad(var_v).unwrap().to_f32_vec().unwrap();

    // Check padding gradients are identically zero
    // Batch 0 padding queries (rows >= 3)
    for head in 0..h {
        for row in 3..t {
            let offset = ((0 * h + head) * t + row) * d;
            for c in 0..d {
                assert_eq!(
                    grad_q[offset + c],
                    0.0,
                    "grad_q must be 0 for padding query row {row}"
                );
                assert_eq!(
                    grad_k[offset + c],
                    0.0,
                    "grad_k must be 0 for padding key row {row}"
                );
                assert_eq!(
                    grad_v[offset + c],
                    0.0,
                    "grad_v must be 0 for padding val row {row}"
                );
            }
        }
    }

    // Batch 1 has 0 active tokens: ALL gradients must be identically zero
    for head in 0..h {
        for row in 0..t {
            let offset = ((1 * h + head) * t + row) * d;
            for c in 0..d {
                assert_eq!(grad_q[offset + c], 0.0, "grad_q must be 0 for batch 1");
                assert_eq!(grad_k[offset + c], 0.0, "grad_k must be 0 for batch 1");
                assert_eq!(grad_v[offset + c], 0.0, "grad_v must be 0 for batch 1");
            }
        }
    }

    // Check active tokens have non-zero gradients
    assert!(
        grad_q[0..3 * d].iter().any(|&g| g.abs() > 1e-4),
        "active queries must have non-zero gradient"
    );
    assert!(
        grad_k[0..3 * d].iter().any(|&g| g.abs() > 1e-4),
        "active keys must have non-zero gradient"
    );
    assert!(
        grad_v[0..3 * d].iter().any(|&g| g.abs() > 1e-4),
        "active vals must have non-zero gradient"
    );
}

// ----------------------------------------------------------------------------
// CHALLENGE 3: seq_mean_forward High Dynamic Range cancellation & zero active length
// ----------------------------------------------------------------------------

#[test]
fn challenge_3_seq_mean_hdr_cancellation_and_zero_active_len() {
    let cpu = cpu_backend(2, Numerics::Exact);

    let (b, t, d) = (4usize, 5usize, 4usize);
    // Sequences:
    // Seq 0: [1e7, 1.0, -1e7, 0, 0] with len = 3 -> mean must be 1.0 / 3 = 0.33333334
    // Seq 1: [1e8, 42.0, -1e8, 0, 0] with len = 3 -> mean must be 42.0 / 3 = 14.0
    // Seq 2: len = 0 -> mean must be 0.0 everywhere
    // Seq 3: len = 999 (clamp to 5) -> [1, 2, 3, 4, 5] repeated across d -> mean = 3.0
    let mut data = vec![0.0f32; b * t * d];

    // Seq 0
    data[0 * t * d + 0 * d..0 * t * d + 1 * d].fill(1e7);
    data[0 * t * d + 1 * d..0 * t * d + 2 * d].fill(1.0);
    data[0 * t * d + 2 * d..0 * t * d + 3 * d].fill(-1e7);

    // Seq 1
    data[1 * t * d + 0 * d..1 * t * d + 1 * d].fill(1e8);
    data[1 * t * d + 1 * d..1 * t * d + 2 * d].fill(42.0);
    data[1 * t * d + 2 * d..1 * t * d + 3 * d].fill(-1e8);

    // Seq 2 is len 0, fill with arbitrary data that must never be read
    data[2 * t * d..3 * t * d].fill(999.0);

    // Seq 3
    for step in 0..5 {
        data[3 * t * d + step * d..3 * t * d + (step + 1) * d].fill((step + 1) as f32);
    }

    let lens = [3u32, 3, 0, 999];
    let x = f32t(&cpu, &data, &[b, t, d]);
    let l = u32t(&cpu, &lens, &[b]);

    let res = cpu.seq_mean_forward(&x, &l).unwrap();
    assert_eq!(res.shape(), &[b, d]);
    let out = res.to_f32_vec().unwrap();

    // Verify Seq 0 HDR cancellation resistance
    for c in 0..d {
        let val = out[0 * d + c];
        assert!(
            (val - 1.0 / 3.0).abs() < 1e-6,
            "Seq 0 HDR cancellation: got {val}, want {}",
            1.0 / 3.0
        );
    }

    // Verify Seq 1 HDR cancellation resistance
    for c in 0..d {
        let val = out[1 * d + c];
        assert_eq!(val, 14.0, "Seq 1 HDR cancellation: got {val}, want 14.0");
    }

    // Verify Seq 2 zero length produces exact +0.0
    for c in 0..d {
        let val = out[2 * d + c];
        assert_eq!(
            val.to_bits(),
            0.0f32.to_bits(),
            "Seq 2 len=0 must produce +0.0: got {val}"
        );
    }

    // Verify Seq 3 clamped to t=5
    for c in 0..d {
        let val = out[3 * d + c];
        assert_eq!(val, 3.0, "Seq 3 len=999 clamped: got {val}, want 3.0");
    }

    // Additional HDR stress: alternating extreme values [-1e8, 1e8, -1e8, 1e8, -5.0]
    let alt_data = vec![-1e8f32, 1e8, -1e8, 1e8, -5.0];
    let alt_x = f32t(&cpu, &alt_data, &[1, 5, 1]);
    let alt_l = u32t(&cpu, &[5], &[1]);
    let alt_out = cpu
        .seq_mean_forward(&alt_x, &alt_l)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(
        alt_out[0], -1.0,
        "Alternating HDR mean must be exactly -1.0"
    );

    // Odd column dimensions crossing MEAN_COLUMNS (16) boundaries
    for &odd_d in &[1usize, 3, 15, 17, 31, 33, 65] {
        let x_odd = f32t(&cpu, &vec![2.0; 2 * 3 * odd_d], &[2, 3, odd_d]);
        let l_odd = u32t(&cpu, &[2, 0], &[2]);
        let out_odd = cpu
            .seq_mean_forward(&x_odd, &l_odd)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        // Batch 0: mean of 2.0 is 2.0
        assert!(out_odd[0..odd_d].iter().all(|&v| v == 2.0));
        // Batch 1: len=0 is +0.0
        assert!(out_odd[odd_d..2 * odd_d].iter().all(|&v| v.to_bits() == 0));
    }
}

// ----------------------------------------------------------------------------
// CHALLENGE 4: L2 Normalization unit sphere invariant & Matryoshka angle conservation
// ----------------------------------------------------------------------------

#[test]
fn challenge_4_l2_normalization_and_matryoshka_prefix() {
    let cpu = cpu_backend(2, Numerics::Exact);

    let dims = [1usize, 2, 7, 16, 64, 128, 256, 768, 1024];
    let eps = 1e-12f32;

    for &d in &dims {
        let rows = 5usize;
        let mut data = vec![0.0f32; rows * d];

        // Row 0: Constant vector [2.0; d]
        data[0 * d..1 * d].fill(2.0);

        // Row 1: High dynamic range [1e5, 1e-2, 1e-3, ...]
        data[1 * d] = 1e5;
        for c in 1..d {
            data[1 * d + c] = 0.01 * ((c % 7) as f32);
        }

        // Row 2: Tiny vector with norm just above eps (e.g. 2 * eps)
        let tiny_step = (2.0 * eps) / (d as f32).sqrt();
        data[2 * d..3 * d].fill(tiny_step);

        // Row 3: Zero row [0.0; d]
        data[3 * d..4 * d].fill(0.0);

        // Row 4: Negative zero row [-0.0; d]
        data[4 * d..5 * d].fill(-0.0);

        let x = f32t(&cpu, &data, &[rows, d]);
        let y = cpu.l2_normalize_forward(&x, eps).unwrap();
        let y_v = y.to_f32_vec().unwrap();

        // Rows 0, 1, 2 must satisfy unit sphere invariant ||y||_2 = 1.0 +- 1e-6
        for r in 0..3 {
            let row = &y_v[r * d..(r + 1) * d];
            let norm: f64 = row
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() <= 1e-6,
                "Unit sphere invariant failed for d={d}, row={r}: got {norm}"
            );
        }

        // Row 3 must be exact 0.0
        assert!(y_v[3 * d..4 * d].iter().all(|&v| v.to_bits() == 0));

        // Row 4 must preserve -0.0
        assert!(y_v[4 * d..5 * d]
            .iter()
            .all(|&v| v.to_bits() == (-0.0f32).to_bits()));
    }

    // Matryoshka prefix slicing angle conservation
    let total_dim = 768usize;
    let mut u_data = vec![0.0f32; total_dim];
    let mut v_data = vec![0.0f32; total_dim];
    for i in 0..total_dim {
        u_data[i] = ((i * 17) % 100) as f32 * 0.05 - 2.5;
        v_data[i] = ((i * 31) % 100) as f32 * 0.05 - 2.5;
    }
    let u_tensor = f32t(&cpu, &u_data, &[1, total_dim]);
    let v_tensor = f32t(&cpu, &v_data, &[1, total_dim]);

    for prefix in [512usize, 256, 128, 64, 32, 16] {
        let u_sub = cpu.slice(&u_tensor, 1, 0, prefix).unwrap();
        let v_sub = cpu.slice(&v_tensor, 1, 0, prefix).unwrap();

        let u_norm = cpu.l2_normalize_forward(&u_sub, eps).unwrap();
        let v_norm = cpu.l2_normalize_forward(&v_sub, eps).unwrap();

        let un = u_norm.to_f32_vec().unwrap();
        let vn = v_norm.to_f32_vec().unwrap();
        let us = u_sub.to_f32_vec().unwrap();
        let vs = v_sub.to_f32_vec().unwrap();

        // 1. Direction preservation: cos(x, norm(x)) = 1.0 +- 1e-6
        let mut dot_u = 0.0f64;
        let mut norm_us = 0.0f64;
        let mut norm_un = 0.0f64;
        for (&a, &b) in us.iter().zip(&un) {
            dot_u += f64::from(a) * f64::from(b);
            norm_us += f64::from(a) * f64::from(a);
            norm_un += f64::from(b) * f64::from(b);
        }
        let cos_u = dot_u / (norm_us.sqrt() * norm_un.sqrt());
        assert!(
            (cos_u - 1.0).abs() <= 1e-6,
            "Matryoshka direction preservation failed for prefix {prefix}: {cos_u}"
        );

        // 2. Cosine similarity angle conservation: cos(un, vn) == cos(us, vs)
        let mut dot_norm = 0.0f64;
        let mut dot_raw = 0.0f64;
        let mut norm_vs = 0.0f64;
        for i in 0..prefix {
            dot_norm += f64::from(un[i]) * f64::from(vn[i]);
            dot_raw += f64::from(us[i]) * f64::from(vs[i]);
            norm_vs += f64::from(vs[i]) * f64::from(vs[i]);
        }
        let cos_raw = dot_raw / (norm_us.sqrt() * norm_vs.sqrt());
        assert!(
            (dot_norm - cos_raw).abs() <= 1e-6,
            "Matryoshka angle conservation failed for prefix {prefix}: norm {dot_norm} vs raw {cos_raw}"
        );
    }
}

// ----------------------------------------------------------------------------
// CHALLENGE 5: Per-head V RMSNorm multi-head independence & learned weight scaling
// ----------------------------------------------------------------------------

#[test]
fn challenge_5_rms_qkv_norm_multi_head_independence_and_scaling() {
    let cpu = cpu_backend(2, Numerics::Exact);

    let (b, h_q, h_kv, t, d) = (2usize, 4usize, 2usize, 4usize, 8usize);
    let mut q_data = vec![0.0f32; b * h_q * t * d];
    let mut k_data = vec![0.0f32; b * h_kv * t * d];
    let mut v_data = vec![0.0f32; b * h_kv * t * d];

    for i in 0..q_data.len() {
        q_data[i] = ((i % 11) as f32 + 1.0) * 0.2;
    }
    for i in 0..k_data.len() {
        k_data[i] = ((i % 13) as f32 + 1.0) * 0.2;
        v_data[i] = ((i % 17) as f32 + 1.0) * 0.2;
    }

    let mut qw_data = vec![0.0f32; d];
    let mut kw_data = vec![0.0f32; d];
    for c in 0..d {
        qw_data[c] = 1.0 + 0.1 * (c as f32);
        kw_data[c] = 0.5 + 0.05 * (c as f32);
    }

    let q = f32t(&cpu, &q_data, &[b, h_q, t, d]);
    let k = f32t(&cpu, &k_data, &[b, h_kv, t, d]);
    let v = f32t(&cpu, &v_data, &[b, h_kv, t, d]);
    let qw = f32t(&cpu, &qw_data, &[d]);
    let kw = f32t(&cpu, &kw_data, &[d]);

    let (qn_0, kn_0, vn_0) = cpu
        .rms_qkv_norm_forward(&q, &k, &v, Some(&qw), Some(&kw), 1e-6)
        .unwrap();

    let qn0_v = qn_0.to_f32_vec().unwrap();
    let kn0_v = kn_0.to_f32_vec().unwrap();
    let vn0_v = vn_0.to_f32_vec().unwrap();

    // 1. Multi-head independence: mutate Head 0 in Q by 1000x
    let mut q_mutated = q_data.clone();
    for row in 0..t {
        for c in 0..d {
            q_mutated[(0 * h_q + 0) * t * d + row * d + c] *= 1000.0;
        }
    }
    let q_m = f32t(&cpu, &q_mutated, &[b, h_q, t, d]);
    let (qn_1, kn_1, vn_1) = cpu
        .rms_qkv_norm_forward(&q_m, &k, &v, Some(&qw), Some(&kw), 1e-6)
        .unwrap();

    let qn1_v = qn_1.to_f32_vec().unwrap();
    let kn1_v = kn_1.to_f32_vec().unwrap();
    let vn1_v = vn_1.to_f32_vec().unwrap();

    // Verify heads 1, 2, 3 of Q remain bit-identical
    for h in 1..h_q {
        for row in 0..t {
            let offset = ((0 * h_q + h) * t + row) * d;
            for c in 0..d {
                assert_eq!(
                    qn1_v[offset + c].to_bits(),
                    qn0_v[offset + c].to_bits(),
                    "Q head {h} corrupted by mutation of head 0"
                );
            }
        }
    }
    // Verify K and V remain completely unaffected
    assert_eq!(
        kn1_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        kn0_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "K corrupted by Q mutation"
    );
    assert_eq!(
        vn1_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vn0_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "V corrupted by Q mutation"
    );

    // 2. Learned weight scaling: verify V is strictly weightless (rms_norm with weight=1.0)
    let ones_v = f32t(&cpu, &vec![1.0; d], &[d]);
    let want_v = cpu.rms_norm_forward(&v, &ones_v, 1e-6).unwrap();
    assert_eq!(
        vn_0.to_f32_vec().unwrap(),
        want_v.to_f32_vec().unwrap(),
        "V must match weightless rms_norm"
    );

    // Verify Q and K match their learned weights
    let want_q = cpu.rms_norm_forward(&q, &qw, 1e-6).unwrap();
    let want_k = cpu.rms_norm_forward(&k, &kw, 1e-6).unwrap();
    assert_eq!(qn_0.to_f32_vec().unwrap(), want_q.to_f32_vec().unwrap());
    assert_eq!(kn_0.to_f32_vec().unwrap(), want_k.to_f32_vec().unwrap());

    // 3. Weightless Q and K when weights are None
    let (qn_none, kn_none, vn_none) = cpu
        .rms_qkv_norm_forward(&q, &k, &v, None, None, 1e-6)
        .unwrap();
    let want_q_none = cpu.rms_norm_forward(&q, &ones_v, 1e-6).unwrap();
    let want_k_none = cpu.rms_norm_forward(&k, &ones_v, 1e-6).unwrap();
    assert_eq!(
        qn_none.to_f32_vec().unwrap(),
        want_q_none.to_f32_vec().unwrap()
    );
    assert_eq!(
        kn_none.to_f32_vec().unwrap(),
        want_k_none.to_f32_vec().unwrap()
    );
    assert_eq!(vn_none.to_f32_vec().unwrap(), vn_0.to_f32_vec().unwrap());

    // 4. Refusals: weight shape mismatch and non-finite eps
    let bad_w = f32t(&cpu, &[1.0; 5], &[5]); // D is 8, weight is 5
    assert!(matches!(
        cpu.rms_qkv_norm_forward(&q, &k, &v, Some(&bad_w), None, 1e-6),
        Err(OjasError::Shape { .. })
    ));

    for bad_eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let res = cpu.rms_qkv_norm_forward(&q, &k, &v, None, None, bad_eps);
        assert!(
            matches!(res, Err(OjasError::NonFinite { .. })),
            "bad eps {bad_eps} should refuse: got {res:?}"
        );
    }

    // Zero input with eps = 0.0 leads to nonfinite denom and refuses
    let zero_q = f32t(&cpu, &vec![0.0; b * h_q * t * d], &[b, h_q, t, d]);
    let zero_k = f32t(&cpu, &vec![0.0; b * h_kv * t * d], &[b, h_kv, t, d]);
    let zero_v = f32t(&cpu, &vec![0.0; b * h_kv * t * d], &[b, h_kv, t, d]);
    let res_zero = cpu.rms_qkv_norm_forward(&zero_q, &zero_k, &zero_v, None, None, 0.0);
    assert!(
        matches!(res_zero, Err(OjasError::NonFinite { .. })),
        "zero tensor with eps=0.0 should refuse nonfinite: got {res_zero:?}"
    );
}

// ----------------------------------------------------------------------------
// CHALLENGE 6: Bit-identity of CPU reductions across thread counts & execution modes
// ----------------------------------------------------------------------------

#[test]
fn challenge_6_bit_identity_across_threads_and_numerics() {
    let thread_counts = [1usize, 2, 3, 5, 8];
    let numerics = [Numerics::Exact, Numerics::Fast];

    let (b, t, d) = (33usize, 25usize, 48usize);
    let mut data = vec![0.0f32; b * t * d];
    for i in 0..data.len() {
        data[i] = ((i * 19) % 257) as f32 * 0.05 - 6.0;
    }
    let lens_data: Vec<u32> = (0..b).map(|i| ((i * 7) % (t + 2)) as u32).collect();

    let mut first_gelu_fwd: Option<Vec<u32>> = None;
    let mut first_gelu_bwd: Option<Vec<u32>> = None;
    let mut first_seq_mean: Option<Vec<u32>> = None;
    let mut first_l2_norm: Option<Vec<u32>> = None;
    let mut first_rms_qkv: Option<(Vec<u32>, Vec<u32>, Vec<u32>)> = None;

    for &tc in &thread_counts {
        for &num in &numerics {
            let cpu = cpu_backend(tc, num);
            let x = f32t(&cpu, &data, &[b, t, d]);
            let lens = u32t(&cpu, &lens_data, &[b]);

            // 1. gelu_tanh_forward
            let y = cpu.gelu_tanh_forward(&x).unwrap();
            let y_bits = bits(&y);
            if let Some(ref want) = first_gelu_fwd {
                assert_eq!(
                    &y_bits, want,
                    "gelu_tanh_forward bits differed on tc={tc}, num={num:?}"
                );
            } else {
                first_gelu_fwd = Some(y_bits);
            }

            // 2. gelu_tanh_backward
            let dy = f32t(&cpu, &vec![1.5f32; b * t * d], &[b, t, d]);
            let dx = cpu.gelu_tanh_backward(&x, &dy).unwrap();
            let dx_bits = bits(&dx);
            if let Some(ref want) = first_gelu_bwd {
                assert_eq!(
                    &dx_bits, want,
                    "gelu_tanh_backward bits differed on tc={tc}, num={num:?}"
                );
            } else {
                first_gelu_bwd = Some(dx_bits);
            }

            // 3. seq_mean_forward
            let sm = cpu.seq_mean_forward(&x, &lens).unwrap();
            let sm_bits = bits(&sm);
            if let Some(ref want) = first_seq_mean {
                assert_eq!(
                    &sm_bits, want,
                    "seq_mean_forward bits differed on tc={tc}, num={num:?}"
                );
            } else {
                first_seq_mean = Some(sm_bits);
            }

            // 4. l2_normalize_forward
            let x_flat = x.reshape(&[b * t, d]).unwrap();
            let l2 = cpu.l2_normalize_forward(&x_flat, 1e-12).unwrap();
            let l2_bits = bits(&l2);
            if let Some(ref want) = first_l2_norm {
                assert_eq!(
                    &l2_bits, want,
                    "l2_normalize_forward bits differed on tc={tc}, num={num:?}"
                );
            } else {
                first_l2_norm = Some(l2_bits);
            }

            // 5. rms_qkv_norm_forward
            let q_sub = x.reshape(&[b, 1, t, d]).unwrap();
            let k_sub = q_sub.clone();
            let v_sub = q_sub.clone();
            let qw = f32t(&cpu, &vec![1.2f32; d], &[d]);
            let kw = f32t(&cpu, &vec![0.8f32; d], &[d]);
            let (qn, kn, vn) = cpu
                .rms_qkv_norm_forward(&q_sub, &k_sub, &v_sub, Some(&qw), Some(&kw), 1e-6)
                .unwrap();
            let q_bits = bits(&qn);
            let k_bits = bits(&kn);
            let v_bits = bits(&vn);
            if let Some(ref want) = first_rms_qkv {
                assert_eq!(
                    (&q_bits, &k_bits, &v_bits),
                    (&want.0, &want.1, &want.2),
                    "rms_qkv_norm_forward bits differed on tc={tc}, num={num:?}"
                );
            } else {
                first_rms_qkv = Some((q_bits, k_bits, v_bits));
            }
        }
    }
}
