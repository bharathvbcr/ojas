//! One forward, backward, and AdamW step against PyTorch.
//!
//! The numbers in `torch_ref/data.rs` came from torch 2.13.0 float32 on this
//! machine on 2026-10-01 (CPU, math SDPA, seed 0). `cargo test` does not need
//! torch. The graph is RMSNorm (eps 1e-6, multiply by w), bias-free linear
//! `y = x @ W.T`, half-split RoPE, causal SDPA with scale `1/sqrt(head_dim)`,
//! mean cross-entropy, then one AdamW update. The learning rate is the
//! nanolab AdamW peak `6e-4` times [`ojas_cpu::CosineSchedule`] at step 8
//! (warmup 4, total 20). Matrices use weight decay 0.1. The norm weight uses
//! weight decay 0, so that parameter is not scaled before the moment update.

use ojas_core::{AdamWConfig, Backend, Budget, DType, Tensor, RMS_NORM_EPS};
use ojas_cpu::{scaled_lr, CosineSchedule, CpuBackend};

#[path = "torch_ref/data.rs"]
mod data;

use data::{
    ATTN, COS, COSINE_MULT_STEP8, GWN, GWQ, H, LOGITS, LOSS, NORM_AFTER, NORM_W0, SIN, TARGETS,
    WK0, WO0, WQ0, WQ_AFTER, WV0, X,
};

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(16 << 20))
}

fn f32t(cpu: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, cpu.budget()).expect("tensor")
}

fn flat(tensor: &Tensor) -> Vec<f32> {
    tensor.to_f32_vec().expect("f32")
}

fn max_abs(got: &[f32], expect: &[f32]) -> f64 {
    assert_eq!(got.len(), expect.len());
    got.iter()
        .zip(expect)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0_f64, f64::max)
}

fn add3(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((x, y), z)| x + y + z)
        .collect()
}

struct StepRecord {
    mult: f64,
    h: Vec<f32>,
    attn: Vec<f32>,
    logits: Vec<f32>,
    loss: f32,
    gwq: Vec<f32>,
    gnorm: Vec<f32>,
    wq: Vec<f32>,
    norm: Vec<f32>,
}

/// The regression test's one step: RMSNorm, bias-free QKV and output
/// linears, half-split RoPE, causal SDPA (1 head), mean cross-entropy,
/// backward, then AdamW on `wq` (weight decay 0.1) and the norm weight
/// (weight decay 0). Learning rate is `6e-4` times the cosine multiplier
/// at step 8. `wk`, `wv`, and `wo` get weight gradients and are not stepped.
#[allow(clippy::too_many_arguments)]
fn regression_graph(
    cpu: &CpuBackend,
    b: usize,
    t: usize,
    d: usize,
    vocab: usize,
    x_data: &[f32],
    norm_data: &[f32],
    wq_data: &[f32],
    wk_data: &[f32],
    wv_data: &[f32],
    wo_data: &[f32],
    cos_data: &[f32],
    sin_data: &[f32],
    targets_data: &[u32],
) -> StepRecord {
    let schedule = CosineSchedule::new(4, 20).expect("schedule");
    let mult = schedule.multiplier(8).expect("multiplier");

    let x = f32t(cpu, x_data, &[b, t, d]);
    let mut norm_w = f32t(cpu, norm_data, &[d]);
    let mut wq = f32t(cpu, wq_data, &[d, d]);
    let wk = f32t(cpu, wk_data, &[d, d]);
    let wv = f32t(cpu, wv_data, &[d, d]);
    let wo = f32t(cpu, wo_data, &[vocab, d]);
    let cos = f32t(cpu, cos_data, &[t, d]);
    let sin = f32t(cpu, sin_data, &[t, d]);
    let targets = Tensor::from_u32(targets_data, &[b, t], cpu.budget()).expect("targets");

    let h = cpu
        .rms_norm_forward(&x, &norm_w, RMS_NORM_EPS)
        .expect("rms");
    let q = cpu.linear_forward(&h, &wq).expect("q");
    let k = cpu.linear_forward(&h, &wk).expect("k");
    let v = cpu.linear_forward(&h, &wv).expect("v");
    let q_bt = f32t(cpu, &flat(&q), &[b, t, 1, d]);
    let k_bt = f32t(cpu, &flat(&k), &[b, t, 1, d]);
    let q_r = cpu
        .rope_half_split_forward(&q_bt, &cos, &sin)
        .expect("rope q");
    let k_r = cpu
        .rope_half_split_forward(&k_bt, &cos, &sin)
        .expect("rope k");
    // One head: [B, T, 1, D] and [B, 1, T, D] are the same contiguous order.
    let q_a = f32t(cpu, &flat(&q_r), &[b, 1, t, d]);
    let k_a = f32t(cpu, &flat(&k_r), &[b, 1, t, d]);
    let v_a = f32t(cpu, &flat(&v), &[b, 1, t, d]);
    let attn = cpu.causal_sdpa_forward(&q_a, &k_a, &v_a).expect("sdpa");
    let y = f32t(cpu, &flat(&attn), &[b, t, d]);
    let logits = cpu.linear_forward(&y, &wo).expect("logits");
    let loss = flat(
        &cpu.cross_entropy_mean_forward(&logits, &targets, None)
            .expect("loss"),
    )[0];

    let g_logits = cpu
        .cross_entropy_mean_backward(&logits, &targets, None)
        .expect("dlogits");
    let (g_y, _) = cpu.linear_backward(&y, &wo, &g_logits).expect("dwo");
    let g_attn = f32t(cpu, &flat(&g_y), &[b, 1, t, d]);
    let (g_q, g_k, g_v) = cpu
        .causal_sdpa_backward(&q_a, &k_a, &v_a, &g_attn)
        .expect("dsdpa");
    let g_q_bt = f32t(cpu, &flat(&g_q), &[b, t, 1, d]);
    let g_k_bt = f32t(cpu, &flat(&g_k), &[b, t, 1, d]);
    let g_q_lin = f32t(
        cpu,
        &flat(
            &cpu.rope_half_split_backward(&g_q_bt, &cos, &sin)
                .expect("drope q"),
        ),
        &[b, t, d],
    );
    let g_k_lin = f32t(
        cpu,
        &flat(
            &cpu.rope_half_split_backward(&g_k_bt, &cos, &sin)
                .expect("drope k"),
        ),
        &[b, t, d],
    );
    let g_v_lin = f32t(cpu, &flat(&g_v), &[b, t, d]);
    let (gq_h, g_wq) = cpu.linear_backward(&h, &wq, &g_q_lin).expect("dwq");
    let (gk_h, _) = cpu.linear_backward(&h, &wk, &g_k_lin).expect("dwk");
    let (gv_h, _) = cpu.linear_backward(&h, &wv, &g_v_lin).expect("dwv");
    let g_h = f32t(
        cpu,
        &add3(&flat(&gq_h), &flat(&gk_h), &flat(&gv_h)),
        &[b, t, d],
    );
    let (_, g_norm) = cpu
        .rms_norm_backward(&x, &norm_w, &g_h, RMS_NORM_EPS)
        .expect("dnorm");

    let lr = scaled_lr(6e-4, mult).expect("lr");
    let mut m1 = Tensor::zeros(&[d, d], DType::F32, cpu.budget()).expect("m1");
    let mut m2 = Tensor::zeros(&[d, d], DType::F32, cpu.budget()).expect("m2");
    cpu.adamw_step(
        &mut wq,
        &g_wq,
        &mut m1,
        &mut m2,
        0,
        AdamWConfig::nanolab(lr, 0.1),
    )
    .expect("adamw wq");
    let mut nm1 = Tensor::zeros(&[d], DType::F32, cpu.budget()).expect("nm1");
    let mut nm2 = Tensor::zeros(&[d], DType::F32, cpu.budget()).expect("nm2");
    cpu.adamw_step(
        &mut norm_w,
        &g_norm,
        &mut nm1,
        &mut nm2,
        0,
        AdamWConfig::nanolab(lr, 0.0),
    )
    .expect("adamw norm");

    StepRecord {
        mult,
        h: flat(&h),
        attn: flat(&attn),
        logits: flat(&logits),
        loss,
        gwq: flat(&g_wq),
        gnorm: flat(&g_norm),
        wq: flat(&wq),
        norm: flat(&norm_w),
    }
}

fn tiny_step(cpu: &CpuBackend) -> StepRecord {
    regression_graph(
        cpu, 1, 4, 16, 32, &*X, &*NORM_W0, &*WQ0, &*WK0, &*WV0, &*WO0, &*COS, &*SIN, &TARGETS,
    )
}

#[test]
fn one_step_matches_torch_2_13_float32() {
    // Reference tensors: torch 2.13.0 float32, this machine, 2026-10-01.
    let cpu = cpu();
    let got = tiny_step(&cpu);
    let mult_err = (got.mult - *COSINE_MULT_STEP8).abs();
    let err_h = max_abs(&got.h, &*H);
    let err_attn = max_abs(&got.attn, &*ATTN);
    let err_logits = max_abs(&got.logits, &*LOGITS);
    let err_loss = (f64::from(got.loss) - f64::from(*LOSS)).abs();
    let err_gwq = max_abs(&got.gwq, &*GWQ);
    let err_gnorm = max_abs(&got.gnorm, &*GWN);
    let err_wq = max_abs(&got.wq, &*WQ_AFTER);
    let err_norm = max_abs(&got.norm, &*NORM_AFTER);
    let worst = [
        mult_err, err_h, err_attn, err_logits, err_loss, err_gwq, err_gnorm, err_wq, err_norm,
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    assert!(
        err_loss <= 1e-4 && err_wq <= 1e-4 && err_norm <= 1e-4 && worst <= 1e-4,
        "cosine {mult_err:.3e} rms {err_h:.3e} attn {err_attn:.3e} logits {err_logits:.3e} loss {err_loss:.3e} dwq {err_gwq:.3e} dnorm {err_gnorm:.3e} wq {err_wq:.3e} norm {err_norm:.3e}"
    );
}

fn median(mut samples: Vec<f64>) -> f64 {
    assert!(!samples.is_empty());
    samples.sort_by(|a, b| a.partial_cmp(b).expect("time is finite"));
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    }
}

fn time_calls(total: usize, drop_first: usize, mut step: impl FnMut()) -> (f64, usize) {
    assert!(total > drop_first);
    let mut samples = Vec::with_capacity(total - drop_first);
    for i in 0..total {
        let start = std::time::Instant::now();
        step();
        let secs = start.elapsed().as_secs_f64();
        if i >= drop_first {
            samples.push(secs);
        }
    }
    let n = samples.len();
    (median(samples), n)
}

/// Half-split RoPE cache: `cat(freqs, freqs)` with `10000^(-2i/d)`.
fn rope_cache(time: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; time * dim];
    let mut sin = vec![0.0f32; time * dim];
    for pos in 0..time {
        for i in 0..half {
            let inv = 10000.0f64.powf(-((2 * i) as f64) / (dim as f64));
            let freq = (pos as f64) * inv;
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos[pos * dim + i] = c;
            cos[pos * dim + half + i] = c;
            sin[pos * dim + i] = s;
            sin[pos * dim + half + i] = s;
        }
    }
    (cos, sin)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        2.0 * mantissa - 1.0
    }

    fn normal(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| scale * self.unit()).collect()
    }
}

#[test]
fn one_step_wall_time() {
    let cpu = CpuBackend::new(Budget::new(64 << 20));
    let check = tiny_step(&cpu);
    let err_loss = (f64::from(check.loss) - f64::from(*LOSS)).abs();
    assert!(
        err_loss <= 1e-4,
        "tiny loss {} err {err_loss:.3e} exceeds 1e-4 of frozen torch ref",
        check.loss
    );

    let (tiny_median, tiny_n) = time_calls(200, 20, || {
        let got = tiny_step(&cpu);
        std::hint::black_box(got.loss);
        std::hint::black_box(got.wq);
        std::hint::black_box(got.norm);
    });
    println!(
        "OJAS_BENCH shape=tiny B=1 T=4 d=16 heads=1 vocab=32 calls=200 drop=20 n={tiny_n} median_s={tiny_median:.9e} loss={} loss_err={err_loss:.3e}",
        check.loss
    );

    let (b, t, d, vocab) = (2usize, 32usize, 64usize, 128usize);
    let mut rng = SplitMix64(1);
    let x = rng.normal(b * t * d, 0.02);
    let norm = rng
        .normal(d, 0.02)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let wq = rng.normal(d * d, 0.02);
    let wk = rng.normal(d * d, 0.02);
    let wv = rng.normal(d * d, 0.02);
    let wo = rng.normal(vocab * d, 0.02);
    let (cos, sin) = rope_cache(t, d);
    let targets = (0..b * t)
        .map(|_| (rng.next_u64() % (vocab as u64)) as u32)
        .collect::<Vec<_>>();
    let larger = regression_graph(
        &cpu, b, t, d, vocab, &x, &norm, &wq, &wk, &wv, &wo, &cos, &sin, &targets,
    );
    assert!(
        larger.loss.is_finite(),
        "larger step loss is not finite: {}",
        larger.loss
    );
    let (larger_median, larger_n) = time_calls(20, 4, || {
        let got = regression_graph(
            &cpu, b, t, d, vocab, &x, &norm, &wq, &wk, &wv, &wo, &cos, &sin, &targets,
        );
        std::hint::black_box(got.loss);
        std::hint::black_box(got.wq);
        std::hint::black_box(got.norm);
    });
    println!(
        "OJAS_BENCH shape=larger B=2 T=32 d=64 heads=1 vocab=128 calls=20 drop=4 n={larger_n} median_s={larger_median:.9e} loss={}",
        larger.loss
    );
}
