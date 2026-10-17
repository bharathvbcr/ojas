//! The host sampler (`sample_token`) against torch.
//!
//! `ojas-oracle/fixtures/sampler/golden.txt` (`python/sampler_golden.py`)
//! holds, for fixed logits and settings, what torch computes in float64: the
//! ranked candidates, their renormalised cumulative probabilities after
//! temperature, top-k and top-p, and the inverse-CDF choice for a fixed list
//! of uniforms. torch's generator is not ojas's, so nothing here is seeded.
//! Instead the sampler is handed the same uniforms: `SplitMix64` is placed at
//! the state whose next `next_f64` is exactly `u` (the generator's finalizer
//! is a bijection, so the state is its inverse of `u << 11`, less the
//! increment), and `sample_token` must return torch's id.
//!
//! The sampler returns ids, not probabilities, so the probabilities are
//! recovered from its choice function: the choice is monotone in `u`, and
//! the `u` at which it first leaves candidate `i` is the renormalised
//! cumulative probability of candidates `0..=i`. Bisecting over the 2^53
//! values `next_f64` can return finds each boundary exactly, and the boundary
//! must match torch's cumulative probability within [`CDF_TOL`].
//!
//! The in-file full-sort oracle and the chi-square tests of `sampling.rs`
//! stay: they check the sampler against ojas, this file against torch.

use ojas_infer::{sample_token, SamplingConfig, SplitMix64};

const GOLDEN: &str = include_str!("../../ojas-oracle/fixtures/sampler/golden.txt");

/// Largest allowed |recovered cdf - torch cdf|. The sampler takes `exp` of
/// the logit gap rounded to f32 (`exp_exact`), torch softmaxes in f64, so
/// each probability carries the f32 rounding of its exponent: relative error
/// up to about 6e-8 times the gap in nats, a gap of at most ~10 in these
/// rows, so up to ~6e-7 on a probability and the same on a cdf step. 2e-6
/// is that bound with room for the sum; the observed worst is printed.
const CDF_TOL: f64 = 2e-6;

const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const MIX_1: u64 = 0xbf58_476d_1ce4_e5b9;
const MIX_2: u64 = 0x94d0_49bb_1331_11eb;

/// `c^-1 mod 2^64` for odd `c`, by Newton's iteration.
fn inverse(c: u64) -> u64 {
    assert_eq!(c & 1, 1);
    let mut x = c;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(c.wrapping_mul(x)));
    }
    assert_eq!(c.wrapping_mul(x), 1);
    x
}

/// Undo `x ^ (x >> shift)`.
fn unxorshift(y: u64, shift: u32) -> u64 {
    let mut x = y;
    let mut s = shift;
    while s < 64 {
        x ^= y >> s;
        s += shift;
    }
    x
}

/// The seed for which `SplitMix64::new(seed).next_u64() == out`.
fn seed_for_output(out: u64) -> u64 {
    let z2 = unxorshift(out, 31);
    let z1 = unxorshift(z2.wrapping_mul(inverse(MIX_2)), 27);
    let state = unxorshift(z1.wrapping_mul(inverse(MIX_1)), 30);
    state.wrapping_sub(GAMMA)
}

/// A generator whose next `next_f64` is exactly `m / 2^53`.
fn rng_drawing(m: u64) -> SplitMix64 {
    assert!(m < 1 << 53);
    let rng = SplitMix64::new(seed_for_output(m << 11));
    let mut probe = rng.clone();
    assert_eq!(probe.next_f64(), m as f64 / (1u64 << 53) as f64);
    rng
}

fn uniform_to_m(u: f64) -> u64 {
    let scaled = u * (1u64 << 53) as f64;
    assert!(scaled.fract() == 0.0 && (0.0..(1u64 << 53) as f64).contains(&scaled));
    scaled as u64
}

struct Case {
    name: String,
    logits: Vec<f32>,
    cfg: SamplingConfig,
    rank_ids: Vec<u32>,
    cdf: Vec<f64>,
    uniforms: Vec<f64>,
    choices: Vec<u32>,
}

fn bits(token: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(token, 16).unwrap_or_else(|e| panic!("{token}: {e}")))
}

fn parse(text: &str) -> Vec<Case> {
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("ojas-sampler-golden-v1"));
    assert_eq!(lines.next(), Some("torch 2.13.0"));
    let per_case: usize = lines
        .next()
        .and_then(|l| l.strip_prefix("uniforms_per_case "))
        .and_then(|n| n.parse().ok())
        .expect("uniforms_per_case");
    let mut cases = Vec::new();
    while let Some(head) = lines.next() {
        let name = head.strip_prefix("case ").expect("case line").to_string();
        let mut field = |key: &str| -> Vec<String> {
            let line = lines.next().unwrap_or_else(|| panic!("{name}: no {key}"));
            let mut it = line.split_whitespace();
            assert_eq!(it.next(), Some(key), "{name}: expected {key}");
            it.map(str::to_string).collect()
        };
        let logits: Vec<f32> = field("logits").iter().map(|t| bits(t)).collect();
        let temperature = bits(&field("temperature")[0]);
        let top_k = match field("top_k")[0].as_str() {
            "none" => None,
            k => Some(k.parse().unwrap()),
        };
        let top_p = match field("top_p")[0].as_str() {
            "none" => None,
            p => Some(bits(p)),
        };
        let rank_ids = field("rank_ids")
            .iter()
            .map(|t| t.parse().unwrap())
            .collect();
        let cdf = field("cdf").iter().map(|t| t.parse().unwrap()).collect();
        let uniforms: Vec<f64> = field("uniforms")
            .iter()
            .map(|t| t.parse().unwrap())
            .collect();
        let choices: Vec<u32> = field("choices")
            .iter()
            .map(|t| t.parse().unwrap())
            .collect();
        assert_eq!(field("end").len(), 0);
        assert_eq!(
            (uniforms.len(), choices.len()),
            (per_case, per_case),
            "{name}"
        );
        cases.push(Case {
            name,
            logits,
            cfg: SamplingConfig {
                temperature,
                top_k,
                top_p,
            },
            rank_ids,
            cdf,
            uniforms,
            choices,
        });
    }
    cases
}

fn is_greedy(c: &Case) -> bool {
    c.cfg.temperature == 0.0
}

#[test]
fn the_golden_is_well_formed_and_covers_every_knob() {
    let cases = parse(GOLDEN);
    assert_eq!(cases.len(), 15);
    for c in &cases {
        c.cfg.validate().unwrap();
        assert_eq!(c.rank_ids.len(), c.cdf.len(), "{}", c.name);
        assert!(c.cdf.windows(2).all(|w| w[0] < w[1]), "{}", c.name);
        // torch's float64 cumsum of a renormalised vector ends within an ulp or two of 1.
        assert!((*c.cdf.last().unwrap() - 1.0).abs() <= 1e-12, "{}", c.name);
        let mut ids = c.rank_ids.clone();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            c.rank_ids.len(),
            "{}: repeated candidate",
            c.name
        );
        assert!(c.rank_ids.iter().all(|&i| (i as usize) < c.logits.len()));
        assert!(
            c.choices.iter().all(|i| c.rank_ids.contains(i)),
            "{}",
            c.name
        );
    }
    let has = |f: &dyn Fn(&Case) -> bool| cases.iter().any(f);
    assert!(has(&|c| is_greedy(c)));
    assert!(has(&|c| !is_greedy(c)
        && c.cfg.top_k.is_none()
        && c.cfg.top_p.is_none()));
    assert!(has(&|c| !is_greedy(c)
        && c.cfg.top_k.is_some()
        && c.cfg.top_p.is_none()));
    assert!(has(&|c| !is_greedy(c)
        && c.cfg.top_k.is_none()
        && c.cfg.top_p.is_some()));
    assert!(has(&|c| c.cfg.top_k.is_some() && c.cfg.top_p.is_some()));
    // A row with masked (-inf) logits, one with a tie the cut falls inside,
    // and a signed-zero tie.
    assert!(has(&|c| c.logits.contains(&f32::NEG_INFINITY)));
    assert!(has(&|c| c.name == "top_k_cuts_a_tie"));
    assert!(has(&|c| c
        .logits
        .iter()
        .any(|v| v.to_bits() == 0x8000_0000)));
}

#[test]
fn the_inverse_of_the_generator_places_it_on_any_uniform() {
    for m in [0, 1, 12345, (1 << 52) + 7, (1 << 53) - 1] {
        let mut rng = rng_drawing(m);
        assert_eq!(rng.next_f64(), m as f64 / (1u64 << 53) as f64);
    }
}

/// Greedy and every seeded draw return torch's id for torch's uniform, and
/// a draw consumes exactly one value from the generator (a greedy call none).
#[test]
fn choices_match_torch_exactly_for_every_uniform() {
    let mut checked = 0;
    for c in parse(GOLDEN) {
        for (&u, &want) in c.uniforms.iter().zip(&c.choices) {
            let mut rng = rng_drawing(uniform_to_m(u));
            let before = rng.state();
            let got = sample_token(&c.logits, &c.cfg, &mut rng).unwrap();
            assert_eq!(got, want, "{} at u = {u}", c.name);
            let moved = rng.state().wrapping_sub(before);
            assert_eq!(
                moved,
                if is_greedy(&c) { 0 } else { GAMMA },
                "{}: draws consumed",
                c.name
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 15 * 26);
}

/// The sampler's cumulative probabilities, recovered by bisecting its
/// choice function, against torch's within [`CDF_TOL`].
#[test]
fn recovered_probabilities_match_torch_within_the_stated_tolerance() {
    let mut worst = 0.0f64;
    for c in parse(GOLDEN).into_iter().filter(|c| !is_greedy(c)) {
        let position = |m: u64| -> usize {
            let mut rng = rng_drawing(m);
            let id = sample_token(&c.logits, &c.cfg, &mut rng).unwrap();
            c.rank_ids
                .iter()
                .position(|&r| r == id)
                .unwrap_or_else(|| panic!("{}: drew {id}, outside the kept candidates", c.name))
        };
        let top = (1u64 << 53) - 1;
        assert_eq!(position(0), 0, "{}", c.name);
        assert_eq!(position(top), c.rank_ids.len() - 1, "{}", c.name);
        // The choice never goes back to an earlier candidate as u grows.
        let samples = 2000u64;
        let walk: Vec<usize> = (0..=samples)
            .map(|i| position(i * (top / samples)))
            .collect();
        assert!(
            walk.windows(2).all(|w| w[0] <= w[1]),
            "{}: not monotone",
            c.name
        );
        for (i, &want) in c.cdf[..c.cdf.len() - 1].iter().enumerate() {
            // Smallest m whose choice is a candidate after i.
            let (mut lo, mut hi) = (0u64, top);
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                if position(mid) > i {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let got = hi as f64 / (1u64 << 53) as f64;
            let d = (got - want).abs();
            worst = worst.max(d);
            assert!(
                d <= CDF_TOL,
                "{} cdf[{i}]: {got} vs torch {want} (|d| {d:e})",
                c.name
            );
        }
    }
    eprintln!("sampler cdf vs torch: worst |d| {worst:e} (tolerance {CDF_TOL:e})");
}

/// The golden can tell the sampler from one that ignores a setting: with
/// top-k dropped, top-p dropped, or the temperature moved, some choice of
/// the case changes.
#[test]
fn every_setting_changes_some_choice() {
    let changes = |c: &Case, cfg: &SamplingConfig| -> usize {
        c.uniforms
            .iter()
            .zip(&c.choices)
            .filter(|&(&u, &want)| {
                let mut rng = rng_drawing(uniform_to_m(u));
                sample_token(&c.logits, cfg, &mut rng).unwrap() != want
            })
            .count()
    };
    let mut seen = (0, 0, 0);
    for c in parse(GOLDEN).iter().filter(|c| !is_greedy(c)) {
        if c.cfg.top_k.is_some_and(|k| k < c.logits.len() && k > 1) {
            let wide = SamplingConfig {
                top_k: None,
                ..c.cfg.clone()
            };
            if changes(c, &wide) > 0 {
                seen.0 += 1;
            }
        }
        if c.cfg.top_p.is_some() {
            let full = SamplingConfig {
                top_p: None,
                ..c.cfg.clone()
            };
            if changes(c, &full) > 0 {
                seen.1 += 1;
            }
        }
        let hotter = SamplingConfig {
            temperature: c.cfg.temperature * 1.5,
            ..c.cfg.clone()
        };
        if changes(c, &hotter) > 0 {
            seen.2 += 1;
        }
    }
    eprintln!("cases where dropping top-k / top-p / moving temperature changes a choice: {seen:?}");
    assert!(seen.0 >= 2 && seen.1 >= 2 && seen.2 >= 8, "{seen:?}");
}
