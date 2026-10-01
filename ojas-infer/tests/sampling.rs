//! Temperature / top-k / top-p sampling and the shared decode loop.

use ojas_core::{Budget, OjasError, Tensor};
use ojas_infer::{
    argmax_token, sample_token, BlockWeights, CpuGpt, GenerateConfig, GptConfig, GptWeights,
    KvCache, SamplingConfig, SplitMix64,
};

fn cfg(temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> SamplingConfig {
    SamplingConfig {
        temperature,
        top_k,
        top_p,
    }
}

/// ojas-data's `CounterRng` reference vectors (ojas-data/src/rng.rs). The
/// two must stay the same stream.
#[test]
fn splitmix_matches_ojas_data_counter_rng() {
    let mut r = SplitMix64::new(0);
    assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
    assert_eq!(r.next_u64(), 0x6E789E6AA1B965F4);
    assert_eq!(r.next_u64(), 0x06C45D188009454F);
    let mut top = SplitMix64::new(u64::MAX);
    assert_eq!(top.next_u64(), 0xE4D971771B652C20);
    assert_eq!(top.state(), 0x9E3779B97F4A7C14);
    let u = SplitMix64::new(0).next_f64();
    assert!((0.0..1.0).contains(&u));
    assert_eq!(u, (0xE220A8397B1DCDAFu64 >> 11) as f64 / (1u64 << 53) as f64);
}

/// Counts over `n` draws.
fn counts(logits: &[f32], c: &SamplingConfig, seed: u64, n: usize) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    let mut out = vec![0usize; logits.len()];
    for _ in 0..n {
        out[sample_token(logits, c, &mut rng).unwrap() as usize] += 1;
    }
    out
}

/// Pearson chi-square of `observed` against `probs`, over cells with
/// nonzero probability. A zero-probability cell with any count fails.
fn chi_square(observed: &[usize], probs: &[f64]) -> f64 {
    let n: usize = observed.iter().sum();
    let mut stat = 0.0;
    for (&o, &p) in observed.iter().zip(probs) {
        if p == 0.0 {
            assert_eq!(o, 0, "drew a token with probability 0");
            continue;
        }
        let e = p * n as f64;
        stat += (o as f64 - e).powi(2) / e;
    }
    stat
}

/// 0.999 quantiles of chi-square with 1, 2 and 3 degrees of freedom.
const CHI2_999: [f64; 3] = [10.83, 13.82, 16.27];

#[test]
fn draws_follow_temperature_top_k_and_top_p() {
    let probs = [0.1f64, 0.2, 0.3, 0.4];
    let logits: Vec<f32> = probs.iter().map(|p| p.ln() as f32).collect();
    let n = 40_000;

    let got = counts(&logits, &cfg(1.0, None, None), 11, n);
    let stat = chi_square(&got, &probs);
    assert!(stat < CHI2_999[2], "T=1 chi2 {stat} counts {got:?}");

    // T = 2 flattens: p_i ∝ p_i^(1/2).
    let soft: Vec<f64> = probs.iter().map(|p| p.sqrt()).collect();
    let z: f64 = soft.iter().sum();
    let soft: Vec<f64> = soft.iter().map(|p| p / z).collect();
    let got = counts(&logits, &cfg(2.0, None, None), 12, n);
    let stat = chi_square(&got, &soft);
    assert!(stat < CHI2_999[2], "T=2 chi2 {stat} counts {got:?}");

    // top-k 2 keeps 0.3 and 0.4, renormalized.
    let got = counts(&logits, &cfg(1.0, Some(2), None), 13, n);
    let stat = chi_square(&got, &[0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0]);
    assert!(stat < CHI2_999[0], "top-k chi2 {stat} counts {got:?}");

    // top-p 0.5: 0.4 alone is short of 0.5, 0.4 + 0.3 reaches it.
    let got = counts(&logits, &cfg(1.0, None, Some(0.5)), 14, n);
    let stat = chi_square(&got, &[0.0, 0.0, 3.0 / 7.0, 4.0 / 7.0]);
    assert!(stat < CHI2_999[0], "top-p chi2 {stat} counts {got:?}");

    // top-p 0.75 needs 0.4 + 0.3 + 0.2 = 0.9 ≥ 0.75: three tokens.
    let got = counts(&logits, &cfg(1.0, None, Some(0.75)), 15, n);
    let stat = chi_square(&got, &[0.0, 2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]);
    assert!(stat < CHI2_999[1], "top-p 0.75 chi2 {stat} counts {got:?}");

    // top-p 1.0 keeps everything.
    let got = counts(&logits, &cfg(1.0, None, Some(1.0)), 16, n);
    let stat = chi_square(&got, &probs);
    assert!(stat < CHI2_999[2], "top-p 1 chi2 {stat} counts {got:?}");
}

#[test]
fn the_same_seed_gives_the_same_draws() {
    let logits = [0.3f32, -1.0, 2.0, 0.0, 1.5, -0.5];
    let c = cfg(0.8, Some(4), Some(0.9));
    let draw = |seed| {
        let mut rng = SplitMix64::new(seed);
        (0..200)
            .map(|_| sample_token(&logits, &c, &mut rng).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(draw(5), draw(5));
    assert_ne!(draw(5), draw(6));
}

#[test]
fn temperature_zero_and_top_k_one_are_greedy() {
    let mut rng = SplitMix64::new(99);
    let mut gen = SplitMix64::new(1);
    for _ in 0..500 {
        let logits: Vec<f32> = (0..17).map(|_| (gen.next_f64() * 8.0 - 4.0) as f32).collect();
        let want = argmax_token(&logits).unwrap();
        assert_eq!(sample_token(&logits, &cfg(0.0, None, None), &mut rng).unwrap(), want);
        assert_eq!(
            sample_token(&logits, &cfg(1.3, Some(1), None), &mut rng).unwrap(),
            want
        );
        assert_eq!(
            sample_token(&logits, &cfg(0.0, Some(3), Some(0.2)), &mut rng).unwrap(),
            want
        );
    }
    // Greedy draws nothing from the rng.
    let before = rng.state();
    sample_token(&[1.0, 2.0], &cfg(0.0, None, None), &mut rng).unwrap();
    assert_eq!(rng.state(), before);
}

#[test]
fn ties_and_masked_logits() {
    let mut rng = SplitMix64::new(3);
    // Ties: greedy and top-k 1 take the lowest index, as argmax_token does.
    let tied = [1.0f32, 3.0, 3.0, 0.0];
    assert_eq!(sample_token(&tied, &cfg(0.0, None, None), &mut rng).unwrap(), 1);
    for _ in 0..100 {
        assert_eq!(sample_token(&tied, &cfg(1.0, Some(1), None), &mut rng).unwrap(), 1);
    }
    // Two tied leaders share the mass about evenly.
    let got = counts(&[5.0, 5.0, -50.0], &cfg(1.0, None, None), 4, 20_000);
    let stat = chi_square(&got, &[0.5, 0.5, 0.0]);
    assert!(stat < CHI2_999[0], "{got:?}");

    // -inf is a mask: only the finite logit can be drawn, at any setting.
    let one = [f32::NEG_INFINITY, f32::NEG_INFINITY, -7.0, f32::NEG_INFINITY];
    for c in [
        cfg(0.0, None, None),
        cfg(1.0, None, None),
        cfg(50.0, Some(3), Some(0.01)),
        cfg(0.01, None, Some(1.0)),
    ] {
        for _ in 0..50 {
            assert_eq!(sample_token(&one, &c, &mut rng).unwrap(), 2);
        }
    }
    // top-k larger than the vocabulary keeps all of it.
    let got = counts(&[0.0, 0.0], &cfg(1.0, Some(10), None), 8, 10_000);
    assert!(got[0] > 4_000 && got[1] > 4_000, "{got:?}");
}

#[test]
fn bad_logits_and_settings_are_refused() {
    let mut rng = SplitMix64::new(0);
    let ok = [0.0f32, 1.0];
    let refuse = |logits: &[f32], c: SamplingConfig, rng: &mut SplitMix64| {
        let before = rng.state();
        let err = sample_token(logits, &c, rng).unwrap_err();
        assert_eq!(rng.state(), before, "a refused call advanced the rng");
        err
    };
    for logits in [
        &[f32::NAN, 1.0][..],
        &[f32::INFINITY, 1.0],
        &[f32::NEG_INFINITY, f32::NEG_INFINITY],
    ] {
        for c in [cfg(0.0, None, None), cfg(1.0, None, None)] {
            let err = refuse(logits, c, &mut rng);
            assert!(matches!(err, OjasError::NonFinite { .. }), "{logits:?}: {err}");
        }
    }
    assert!(matches!(
        refuse(&[], cfg(1.0, None, None), &mut rng),
        OjasError::Shape { .. }
    ));
    for c in [
        cfg(-0.5, None, None),
        cfg(f32::NAN, None, None),
        cfg(f32::INFINITY, None, None),
        cfg(1.0, Some(0), None),
        cfg(1.0, None, Some(0.0)),
        cfg(1.0, None, Some(-0.1)),
        cfg(1.0, None, Some(1.000_001)),
        cfg(1.0, None, Some(f32::NAN)),
        cfg(0.0, Some(0), None),
    ] {
        let err = refuse(&ok, c.clone(), &mut rng);
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{c:?}: {err}");
    }
    // The smallest positive temperature cannot overflow the f64 scaled
    // logits (|f32::MAX / 1e-45| < 1e84): it is a finite, near-greedy draw.
    let tiny = f32::from_bits(1);
    for _ in 0..20 {
        assert_eq!(
            sample_token(&[f32::MAX, f32::MIN, 0.0], &cfg(tiny, None, None), &mut rng).unwrap(),
            0
        );
    }
}

fn model(budget: &Budget) -> (CpuGpt, GptConfig) {
    let cfg = GptConfig {
        vocab: 23,
        n_embd: 16,
        n_head: 2,
        n_kv_head: 1,
        head_dim: 8,
        n_layer: 2,
        hidden: 24,
        max_seq: 32,
        rope_base: 10000.0,
    };
    let mut r = SplitMix64::new(0xABCD);
    let mut fill = |n: usize, scale: f64| -> Vec<f32> {
        (0..n)
            .map(|_| ((r.next_f64() * 2.0 - 1.0) * scale) as f32)
            .collect()
    };
    let t = |data: Vec<f32>, shape: &[usize]| Tensor::from_f32(&data, shape, budget).unwrap();
    let (d, h, dh) = (cfg.n_embd, cfg.hidden, cfg.head_dim);
    let (qd, kvd) = (cfg.n_head * dh, cfg.n_kv_head * dh);
    let w = GptWeights {
        tok_emb: t(fill(cfg.vocab * d, 1.0), &[cfg.vocab, d]),
        ln_f: t(vec![1.0; d], &[d]),
        blocks: (0..cfg.n_layer)
            .map(|_| BlockWeights {
                ln1: t(vec![1.0; d], &[d]),
                wq: t(fill(qd * d, 0.4), &[qd, d]),
                wk: t(fill(kvd * d, 0.4), &[kvd, d]),
                wv: t(fill(kvd * d, 0.4), &[kvd, d]),
                q_norm: t(vec![1.0; dh], &[dh]),
                k_norm: t(vec![1.0; dh], &[dh]),
                gate_w: t(fill(cfg.n_head * d, 0.4), &[cfg.n_head, d]),
                gate_b: t(fill(cfg.n_head, 0.5), &[cfg.n_head]),
                vr_lambda: t(fill(1, 1.0), &[1]),
                wo: t(fill(d * qd, 0.3), &[d, qd]),
                ln2: t(vec![1.0; d], &[d]),
                w_gate: t(fill(h * d, 0.4), &[h, d]),
                w_up: t(fill(h * d, 0.4), &[h, d]),
                w_down: t(fill(d * h, 0.3), &[d, h]),
            })
            .collect(),
    };
    (CpuGpt::new(&cfg, &w).unwrap(), cfg)
}

fn gen_cfg(sampling: SamplingConfig, seed: u64, n: usize, stop: &[u32]) -> GenerateConfig {
    GenerateConfig {
        sampling,
        seed,
        max_new_tokens: n,
        stop_tokens: stop.to_vec(),
    }
}

#[test]
fn generate_is_deterministic_by_seed_and_greedy_matches_greedy_decode() {
    let budget = Budget::new(1 << 24);
    let (m, c) = model(&budget);
    let prompt = [3u32, 9, 14];
    let run = |g: &GenerateConfig| {
        let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
        m.generate(&prompt, &mut cache, g).unwrap()
    };
    let hot = cfg(1.5, None, None);
    let a = run(&gen_cfg(hot.clone(), 77, 20, &[]));
    assert_eq!(a.len(), 20);
    assert_eq!(a, run(&gen_cfg(hot.clone(), 77, 20, &[])));
    assert_ne!(a, run(&gen_cfg(hot, 78, 20, &[])));

    let greedy = run(&gen_cfg(cfg(0.0, None, None), 1, 12, &[]));
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    assert_eq!(greedy, m.greedy_decode(&prompt, &mut cache, 12).unwrap());
    let top1 = run(&gen_cfg(cfg(0.7, Some(1), None), 5, 12, &[]));
    assert_eq!(greedy, top1);
}

#[test]
fn generate_stops_at_a_stop_token_and_never_forwards_the_last_token() {
    let budget = Budget::new(1 << 24);
    let (m, c) = model(&budget);
    let prompt = [3u32, 9, 14];
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    let free = m
        .generate(&prompt, &mut cache, &gen_cfg(cfg(1.0, None, None), 4, 12, &[]))
        .unwrap();
    assert_eq!(cache.len(), prompt.len() + free.len() - 1);
    let stop = free[4];
    let first = free.iter().position(|&t| t == stop).unwrap();
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    let cut = m
        .generate(&prompt, &mut cache, &gen_cfg(cfg(1.0, None, None), 4, 12, &[stop]))
        .unwrap();
    assert_eq!(cut, free[..=first], "stops at the first stop token, inclusive");
    assert_eq!(cache.len(), prompt.len() + cut.len() - 1);

    // max_new_tokens 0 primes the cache with the prompt and returns nothing.
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    let none = m
        .generate(&prompt, &mut cache, &gen_cfg(cfg(1.0, None, None), 4, 0, &[]))
        .unwrap();
    assert!(none.is_empty());
    assert_eq!(cache.len(), prompt.len());
}

#[test]
fn generate_refuses_up_front_when_the_cache_cannot_hold_the_request() {
    let budget = Budget::new(1 << 24);
    let (m, _) = model(&budget);
    let prompt = [3u32, 9, 14];
    let n = 5;
    // Exactly prompt + n - 1 slots: fits.
    let mut cache = KvCache::for_model(&m, prompt.len() + n - 1, &budget).unwrap();
    let g = gen_cfg(cfg(0.9, Some(5), None), 2, n, &[]);
    assert_eq!(m.generate(&prompt, &mut cache, &g).unwrap().len(), n);
    // One fewer: refused before any forward, cache untouched.
    let mut cache = KvCache::for_model(&m, prompt.len() + n - 2, &budget).unwrap();
    let err = m.generate(&prompt, &mut cache, &g).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
    assert_eq!(cache.len(), 0);
    // Bad settings and an empty prompt are refused before any forward too.
    let mut cache = KvCache::for_model(&m, 16, &budget).unwrap();
    assert!(m
        .generate(&prompt, &mut cache, &gen_cfg(cfg(-1.0, None, None), 0, 2, &[]))
        .is_err());
    assert!(m
        .generate(&[], &mut cache, &gen_cfg(cfg(1.0, None, None), 0, 2, &[]))
        .is_err());
    assert!(m
        .generate(&prompt, &mut cache, &gen_cfg(cfg(1.0, None, None), 0, 2, &[999]))
        .is_err());
    assert_eq!(cache.len(), 0);
}
