//! Temperature / top-k / top-p sampling and the shared decode loop.

use ojas_core::{exp_exact, Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_infer::{
    argmax_token, sample_token, BlockWeights, CpuGpt, DeviceDecoder, GenerateConfig, GptConfig,
    GptWeights, KvCache, SamplingConfig, SplitMix64,
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
    assert_eq!(
        u,
        (0xE220A8397B1DCDAFu64 >> 11) as f64 / (1u64 << 53) as f64
    );
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
        let logits: Vec<f32> = (0..17)
            .map(|_| (gen.next_f64() * 8.0 - 4.0) as f32)
            .collect();
        let want = argmax_token(&logits).unwrap();
        assert_eq!(
            sample_token(&logits, &cfg(0.0, None, None), &mut rng).unwrap(),
            want
        );
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
    assert_eq!(
        sample_token(&tied, &cfg(0.0, None, None), &mut rng).unwrap(),
        1
    );
    for _ in 0..100 {
        assert_eq!(
            sample_token(&tied, &cfg(1.0, Some(1), None), &mut rng).unwrap(),
            1
        );
    }
    // Two tied leaders share the mass about evenly.
    let got = counts(&[5.0, 5.0, -50.0], &cfg(1.0, None, None), 4, 20_000);
    let stat = chi_square(&got, &[0.5, 0.5, 0.0]);
    assert!(stat < CHI2_999[0], "{got:?}");

    // -inf is a mask: only the finite logit can be drawn, at any setting.
    let one = [
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        -7.0,
        f32::NEG_INFINITY,
    ];
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
            assert!(
                matches!(err, OjasError::NonFinite { .. }),
                "{logits:?}: {err}"
            );
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
    let (cfg, w) = weights(budget);
    (CpuGpt::new(&cfg, &w).unwrap(), cfg)
}

/// Two layers, GQA (2 query heads on 1 KV head).
fn weights(budget: &Budget) -> (GptConfig, GptWeights) {
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
        rms_eps: 1e-6,
        tie_embeddings: true,
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
        norm_f: t(vec![1.0; d], &[d]),
        blocks: (0..cfg.n_layer)
            .map(|_| BlockWeights {
                norm1: t(vec![1.0; d], &[d]),
                q_proj: t(fill(qd * d, 0.4), &[qd, d]),
                k_proj: t(fill(kvd * d, 0.4), &[kvd, d]),
                v_proj: t(fill(kvd * d, 0.4), &[kvd, d]),
                q_norm: t(vec![1.0; dh], &[dh]),
                k_norm: t(vec![1.0; dh], &[dh]),
                gate_w: t(fill(cfg.n_head * d, 0.4), &[cfg.n_head, d]),
                gate_b: t(fill(cfg.n_head, 0.5), &[cfg.n_head]),
                vr_lambda: t(fill(1, 1.0), &[1]),
                o_proj: t(fill(d * qd, 0.3), &[d, qd]),
                norm2: t(vec![1.0; d], &[d]),
                ffn_gate: t(fill(h * d, 0.4), &[h, d]),
                ffn_up: t(fill(h * d, 0.4), &[h, d]),
                ffn_down: t(fill(d * h, 0.3), &[d, h]),
            })
            .collect(),
    };
    (cfg, w)
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
        .generate(
            &prompt,
            &mut cache,
            &gen_cfg(cfg(1.0, None, None), 4, 12, &[]),
        )
        .unwrap();
    assert_eq!(cache.len(), prompt.len() + free.len() - 1);
    let stop = free[4];
    let first = free.iter().position(|&t| t == stop).unwrap();
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    let cut = m
        .generate(
            &prompt,
            &mut cache,
            &gen_cfg(cfg(1.0, None, None), 4, 12, &[stop]),
        )
        .unwrap();
    assert_eq!(
        cut,
        free[..=first],
        "stops at the first stop token, inclusive"
    );
    assert_eq!(cache.len(), prompt.len() + cut.len() - 1);

    // max_new_tokens 0 primes the cache with the prompt and returns nothing.
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    let none = m
        .generate(
            &prompt,
            &mut cache,
            &gen_cfg(cfg(1.0, None, None), 4, 0, &[]),
        )
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
        .generate(
            &prompt,
            &mut cache,
            &gen_cfg(cfg(-1.0, None, None), 0, 2, &[])
        )
        .is_err());
    assert!(m
        .generate(&[], &mut cache, &gen_cfg(cfg(1.0, None, None), 0, 2, &[]))
        .is_err());
    assert!(m
        .generate(
            &prompt,
            &mut cache,
            &gen_cfg(cfg(1.0, None, None), 0, 2, &[999])
        )
        .is_err());
    assert_eq!(cache.len(), 0);
}

/// `DeviceDecoder` drives the same loop over its own cache: the same ids as
/// `CpuGpt` for the same seed (GQA model), the same stop rule and
/// `prompt + N - 1` positions, and the same up-front refusal.
#[test]
fn device_decoder_follows_the_shared_decode_loop() {
    let budget = Budget::new(1 << 24);
    let (m, c) = model(&budget);
    let (_, w) = weights(&budget);
    let cpu = CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact);
    let m = m.with_numerics(Numerics::Exact);
    let prompt = [3u32, 9, 14];
    let mut dec = DeviceDecoder::new(cpu, &c, &w, c.max_seq).unwrap();
    for g in [
        gen_cfg(cfg(1.5, None, None), 77, 20, &[]),
        gen_cfg(cfg(0.9, Some(5), Some(0.9)), 3, 12, &[]),
        gen_cfg(cfg(0.0, None, None), 1, 12, &[]),
    ] {
        let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
        let want = m.generate(&prompt, &mut cache, &g).unwrap();
        dec.reset();
        assert_eq!(dec.generate(&prompt, &g).unwrap(), want);
        assert_eq!(dec.len(), cache.len());
    }
    dec.reset();
    let greedy = dec.greedy_decode(&prompt, 12).unwrap();
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    assert_eq!(greedy, m.greedy_decode(&prompt, &mut cache, 12).unwrap());

    // Stop token: emitted, ends the loop, never forwarded.
    dec.reset();
    let free = dec
        .generate(&prompt, &gen_cfg(cfg(1.0, None, None), 4, 12, &[]))
        .unwrap();
    let stop = free[4];
    let first = free.iter().position(|&t| t == stop).unwrap();
    dec.reset();
    let cut = dec
        .generate(&prompt, &gen_cfg(cfg(1.0, None, None), 4, 12, &[stop]))
        .unwrap();
    assert_eq!(cut, free[..=first]);
    assert_eq!(dec.len(), prompt.len() + cut.len() - 1);

    // A request one position too big is refused before any forward.
    let n = 5;
    let small = prompt.len() + n - 2;
    let cpu = CpuBackend::new(Budget::new(1 << 26));
    let mut dec = DeviceDecoder::new(cpu, &c, &w, small).unwrap();
    let g = gen_cfg(cfg(0.9, Some(5), None), 2, n, &[]);
    let err = dec.generate(&prompt, &g).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
    assert_eq!(dec.len(), 0);
    assert_eq!(
        dec.generate(&prompt, &gen_cfg(cfg(0.9, Some(5), None), 2, n - 1, &[]))
            .unwrap()
            .len(),
        n - 1
    );
    assert_eq!(dec.remaining(), 0);
}

/// `truncate` keeps a prefix: forwarding the dropped tokens again gives the
/// same logits as the first time (CPU `Exact`, so bit for bit), and a
/// length past the filled positions is refused without a change.
#[test]
fn device_decoder_truncate_and_reset_roll_back_to_a_prefix() {
    let budget = Budget::new(1 << 24);
    let (c, w) = weights(&budget);
    let cpu = CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact);
    let mut dec = DeviceDecoder::new(cpu, &c, &w, c.max_seq).unwrap();
    let prompt = [3u32, 9, 14, 2, 20];
    let full = dec.forward(&prompt).unwrap();
    dec.truncate(2).unwrap();
    assert_eq!(dec.len(), 2);
    assert_eq!(dec.forward(&prompt[2..]).unwrap(), full);
    let err = dec.truncate(prompt.len() + 1).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
    assert_eq!(dec.len(), prompt.len());
    dec.reset();
    assert_eq!(dec.forward(&prompt).unwrap(), full);
}

/// Qwen3.5's vocabulary, the largest this sampler is run on.
const QWEN_VOCAB: usize = 248_320;

/// A row with thousands of ties per value, `-inf` masks, and (with
/// `nonpositive`) a maximum of zero held as `+0.0` at even indices and
/// `-0.0` at odd ones.
fn tied_row(seed: u64, nonpositive: bool) -> Vec<f32> {
    let mut r = SplitMix64::new(seed);
    let mut row: Vec<f32> = (0..QWEN_VOCAB)
        .map(|i| {
            let x = r.next_u64();
            let m = (x >> 8) % 64;
            if x.is_multiple_of(97) {
                f32::NEG_INFINITY
            } else if !nonpositive {
                m as f32 * 0.25 - 10.0
            } else if m == 0 {
                if i % 2 == 0 {
                    0.0
                } else {
                    -0.0
                }
            } else {
                -(m as f32) * 0.25
            }
        })
        .collect();
    if nonpositive {
        // A `-0.0` ahead of every `+0.0`, whatever the seed, so greedy
        // must pass over a tied value at a lower index.
        row[0] = -0.0;
    }
    row
}

/// `sample_token` as it was before the partial select: every finite index
/// sorted by (logit desc via `total_cmp`, index asc), truncated to `top_k`,
/// then the same draw. The oracle the select must reproduce.
fn full_sort_oracle(logits: &[f32], c: &SamplingConfig, rng: &mut SplitMix64) -> u32 {
    let mut ranked: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    ranked.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    if c.temperature == 0.0 {
        return ranked[0] as u32;
    }
    if let Some(k) = c.top_k {
        ranked.truncate(k);
    }
    let t = f64::from(c.temperature);
    let scaled: Vec<f64> = ranked.iter().map(|&i| f64::from(logits[i]) / t).collect();
    let top = scaled[0];
    let mut probs: Vec<f64> = scaled
        .iter()
        .map(|z| f64::from(exp_exact((z - top) as f32)))
        .collect();
    let total: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= total;
    }
    let mut keep = probs.len();
    if let Some(top_p) = c.top_p {
        let mut mass = 0.0;
        for (i, p) in probs.iter().enumerate() {
            mass += p;
            if mass >= f64::from(top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    let kept = &probs[..keep];
    let u = rng.next_f64() * kept.iter().sum::<f64>();
    let mut acc = 0.0;
    for (slot, p) in kept.iter().enumerate() {
        acc += p;
        if u < acc {
            return ranked[slot] as u32;
        }
    }
    ranked[keep - 1] as u32
}

/// Greedy takes the lowest index of the leading value (`+0.0` before
/// `-0.0`), and every setting draws what the full sort draws, with the rng
/// left in the same state.
#[test]
fn partial_select_keeps_the_full_sort_order_at_qwen_vocab() {
    for (seed, nonpositive) in [(11, false), (12, true)] {
        let row = tied_row(seed, nonpositive);
        let best = row
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, |a, b| {
                if b.total_cmp(&a).is_gt() {
                    b
                } else {
                    a
                }
            });
        let first = row
            .iter()
            .position(|v| v.to_bits() == best.to_bits())
            .unwrap();
        let ties = row.iter().filter(|v| v.to_bits() == best.to_bits()).count();
        assert!(ties > 1000, "{ties} ties");
        if nonpositive {
            assert_eq!(best.to_bits(), 0.0f32.to_bits());
            assert_eq!(first % 2, 0);
            assert!(
                row[..first].contains(&-0.0),
                "no -0.0 before the first +0.0"
            );
        }
        let mut rng = SplitMix64::new(0);
        assert_eq!(
            sample_token(&row, &cfg(0.0, None, None), &mut rng).unwrap() as usize,
            first
        );
        for c in [
            cfg(0.0, None, None),
            cfg(0.0, Some(7), Some(0.3)),
            cfg(1.0, Some(1), None),
            cfg(0.7, Some(2), None),
            cfg(1.3, Some(50), Some(0.9)),
            cfg(2.0, Some(5000), None),
            cfg(0.5, None, Some(0.5)),
            cfg(1.0, Some(2 * QWEN_VOCAB), None),
        ] {
            for s in 0..6 {
                let mut a = SplitMix64::new(s);
                let mut b = a.clone();
                assert_eq!(
                    sample_token(&row, &c, &mut a).unwrap(),
                    full_sort_oracle(&row, &c, &mut b),
                    "{c:?} seed {s}"
                );
                assert_eq!(a, b, "{c:?} seed {s}: rng state");
            }
        }
    }
}

#[test]
fn non_finite_rows_are_refused_at_qwen_vocab() {
    let row = tied_row(13, false);
    let masked = vec![f32::NEG_INFINITY; QWEN_VOCAB];
    let mut bad_rows = vec![masked];
    for (at, bad) in [
        (QWEN_VOCAB - 1, f32::NAN),
        (QWEN_VOCAB - 1, f32::INFINITY),
        (0, f32::NAN),
    ] {
        let mut r = row.clone();
        r[at] = bad;
        bad_rows.push(r);
    }
    for r in &bad_rows {
        for c in [
            cfg(0.0, None, None),
            cfg(1.0, Some(40), None),
            cfg(1.0, None, Some(0.9)),
        ] {
            let mut rng = SplitMix64::new(5);
            let before = rng.state();
            let err = sample_token(r, &c, &mut rng).unwrap_err();
            assert!(matches!(err, OjasError::NonFinite { .. }), "{c:?}: {err}");
            assert_eq!(rng.state(), before);
        }
    }
}

/// Seeded draws, pinned. The ranking, the f64 arithmetic and `exp_exact`
/// give the same bits on every platform, so these ids must be drawn
/// everywhere; std `exp` (libm) made that a per-platform property.
#[test]
fn seeded_draws_are_pinned() {
    let mut g = SplitMix64::new(2026);
    let logits: Vec<f32> = (0..1000)
        .map(|_| (g.next_f64() * 16.0 - 8.0) as f32)
        .collect();
    let mut got = Vec::new();
    for c in [
        cfg(0.8, Some(40), Some(0.95)),
        cfg(1.0, None, None),
        cfg(1.7, None, Some(0.5)),
    ] {
        let mut rng = SplitMix64::new(7);
        got.push(
            (0..12)
                .map(|_| sample_token(&logits, &c, &mut rng).unwrap())
                .collect::<Vec<_>>(),
        );
    }
    // The full sort the partial select replaced draws the same ids.
    for (c, ids) in [
        cfg(0.8, Some(40), Some(0.95)),
        cfg(1.0, None, None),
        cfg(1.7, None, Some(0.5)),
    ]
    .iter()
    .zip(&got)
    {
        let mut rng = SplitMix64::new(7);
        let oracle: Vec<u32> = (0..12)
            .map(|_| full_sort_oracle(&logits, c, &mut rng))
            .collect();
        assert_eq!(&oracle, ids, "{c:?}");
    }
    let want = [
        [59, 261, 239, 607, 224, 84, 224, 646, 276, 59, 534, 227],
        [239, 714, 0, 301, 99, 607, 6, 27, 928, 591, 84, 784],
        [454, 261, 336, 217, 247, 773, 531, 813, 447, 27, 482, 761],
    ];
    assert_eq!(got, want);
}

/// The trait's host `argmax_rows` (CpuBackend keeps the default): ties to
/// the lowest column, `-0.0` and `+0.0` tied, a NaN or infinity refused,
/// rank 2 only. `DeviceDecoder`'s greedy path then feeds each id back as
/// given: one upload (the prompt), and the ids equal `CpuGpt`'s.
#[test]
fn argmax_rows_default_and_greedy_feedback_on_the_cpu() {
    let cpu = CpuBackend::new(Budget::new(1 << 26));
    let x = Tensor::from_f32(
        &[1.0, 3.0, 3.0, 0.0, -1.0, -0.0, 0.0, -2.0],
        &[2, 4],
        cpu.budget(),
    )
    .unwrap();
    let ids = cpu.argmax_rows(&x).unwrap();
    assert_eq!(ids.shape(), &[2]);
    assert_eq!(ids.u32_slice().unwrap(), &[1, 1]);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x = Tensor::from_f32(&[0.0, bad, 1.0], &[1, 3], cpu.budget()).unwrap();
        let err = cpu.argmax_rows(&x).unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{bad}: {err}");
    }
    let flat = Tensor::from_f32(&[0.0, 1.0], &[2], cpu.budget()).unwrap();
    assert!(matches!(
        cpu.argmax_rows(&flat),
        Err(OjasError::Shape { .. })
    ));

    let budget = Budget::new(1 << 24);
    let (m, c) = model(&budget);
    let (_, w) = weights(&budget);
    let m = m.with_numerics(Numerics::Exact);
    let cpu = CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact);
    let mut dec = DeviceDecoder::new(cpu, &c, &w, c.max_seq).unwrap();
    let prompt = [3u32, 9, 14];
    let mut cache = KvCache::for_model(&m, c.max_seq, &budget).unwrap();
    let want = m.greedy_decode(&prompt, &mut cache, 12).unwrap();
    let before = dec.traffic();
    assert_eq!(dec.greedy_decode(&prompt, 12).unwrap(), want);
    let after = dec.traffic();
    assert_eq!(after.uploads - before.uploads, 1);
    assert_eq!(
        after.upload_bytes - before.upload_bytes,
        4 * prompt.len() as u64
    );
    // The last id is not in the cache: forwarding it appends one position.
    let len = dec.len();
    dec.forward(&[*want.last().unwrap()]).unwrap();
    assert_eq!(dec.len(), len + 1);
}
