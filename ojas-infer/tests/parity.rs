//! Nanolab-block parity for `CpuGpt` and `DeviceDecoder<CpuBackend>`.
//!
//! Three forwards over the same random weights must agree:
//! - `CpuGpt::forward_token` one token at a time through the host KV cache,
//! - the full forward: `ojas_model::forward_logits` on `Eval<CpuBackend>`
//!   (the model's one block: RoPE, QK-norm, gate, value residual; causal
//!   SDPA for multi-head attention, `cached_attention_forward` at
//!   `kv_len == T` for grouped-query),
//! - an f64 scalar transcription of nanolab `model.py` / `mixers.py`
//!   (`Attention.forward`, `SwiGLU`, `RMSNorm`, `apply_rope`), in this file.
//!
//! `DeviceDecoder<CpuBackend>`'s prefill of every prefix is held to the
//! full forward as well.
//!
//! G7 (`docs/framework-design.md` §8) holds `forward_token`, `Eval` and
//! `DeviceDecoder<CpuBackend>` to each other over a multi-token decode.
//!
//! Tolerances, as `max |a - b| / max(1, max |b|)` over every logit of every
//! position:
//! - cached vs full forward: [`PATH_TOL`]. Both are f32; they differ only in
//!   summation order (one-row linear and single-query attention vs the GEMM
//!   and the causal SDPA kernel).
//! - G7: [`G7_TOL`].
//! - either f32 path vs the f64 reference: [`REF_TOL`].

use ojas_core::{Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_infer::{
    BlockWeights, CpuGpt, DeviceDecoder, GenerateConfig, GptConfig, GptWeights, KvCache,
    SamplingConfig,
};
use ojas_model::{bind, forward_logits, Eval, Rope};

const PATH_TOL: f64 = 2.0e-5;
const G7_TOL: f64 = 1.0e-5;
const REF_TOL: f64 = 2.0e-4;
const EPS: f64 = 1.0e-6;

struct Mix(u64);

impl Mix {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[-a, a)` with `a = scale * sqrt(3 / fan_in)`: unit
    /// variance output for unit-variance input when `scale == 1`.
    fn fill(&mut self, n: usize, fan_in: usize, scale: f64) -> Vec<f32> {
        let a = scale * (3.0 / fan_in as f64).sqrt();
        (0..n)
            .map(|_| ((self.next() * 2.0 - 1.0) * a) as f32)
            .collect()
    }

    fn around_one(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| (0.75 + 0.5 * self.next()) as f32).collect()
    }
}

/// Host copy of one block, for the reference.
struct RefBlock {
    ln1: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate_w: Vec<f32>,
    gate_b: Vec<f32>,
    vr_lambda: f32,
    wo: Vec<f32>,
    ln2: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

struct Fixture {
    cfg: GptConfig,
    model: CpuGpt,
    weights: GptWeights,
    emb: Vec<f32>,
    ln_f: Vec<f32>,
    blocks: Vec<RefBlock>,
}

fn t(data: &[f32], shape: &[usize], budget: &Budget) -> Tensor {
    Tensor::from_f32(data, shape, budget).unwrap()
}

fn fixture(cfg: GptConfig, seed: u64, budget: &Budget) -> Fixture {
    let mut r = Mix(seed);
    let (d, h, dh) = (cfg.n_embd, cfg.hidden, cfg.head_dim);
    let qd = cfg.n_head * dh;
    let kvd = cfg.n_kv_head * dh;
    let emb = r.fill(cfg.vocab * d, 3, 1.0);
    let ln_f = r.around_one(d);
    let blocks: Vec<RefBlock> = (0..cfg.n_layer)
        .map(|_| RefBlock {
            ln1: r.around_one(d),
            wq: r.fill(qd * d, d, 1.0),
            wk: r.fill(kvd * d, d, 1.0),
            wv: r.fill(kvd * d, d, 1.0),
            q_norm: r.around_one(dh),
            k_norm: r.around_one(dh),
            gate_w: r.fill(cfg.n_head * d, d, 1.0),
            gate_b: r.fill(cfg.n_head, 1, 0.5),
            vr_lambda: (r.next() * 2.0 - 1.0) as f32,
            wo: r.fill(d * qd, qd, 0.5),
            ln2: r.around_one(d),
            w_gate: r.fill(h * d, d, 1.0),
            w_up: r.fill(h * d, d, 1.0),
            w_down: r.fill(d * h, h, 0.5),
        })
        .collect();
    let weights = GptWeights {
        tok_emb: t(&emb, &[cfg.vocab, d], budget),
        norm_f: t(&ln_f, &[d], budget),
        blocks: blocks
            .iter()
            .map(|b| BlockWeights {
                norm1: t(&b.ln1, &[d], budget),
                q_proj: t(&b.wq, &[qd, d], budget),
                k_proj: t(&b.wk, &[kvd, d], budget),
                v_proj: t(&b.wv, &[kvd, d], budget),
                q_norm: t(&b.q_norm, &[dh], budget),
                k_norm: t(&b.k_norm, &[dh], budget),
                gate_w: t(&b.gate_w, &[cfg.n_head, d], budget),
                gate_b: t(&b.gate_b, &[cfg.n_head], budget),
                vr_lambda: t(&[b.vr_lambda], &[1], budget),
                o_proj: t(&b.wo, &[d, qd], budget),
                norm2: t(&b.ln2, &[d], budget),
                ffn_gate: t(&b.w_gate, &[h, d], budget),
                ffn_up: t(&b.w_up, &[h, d], budget),
                ffn_down: t(&b.w_down, &[d, h], budget),
            })
            .collect(),
    };
    let model = CpuGpt::new(&cfg, &weights).unwrap();
    Fixture {
        cfg,
        model,
        weights,
        emb,
        ln_f,
        blocks,
    }
}

fn mat_vec(w: &[f32], x: &[f64], out: usize) -> Vec<f64> {
    let kin = x.len();
    (0..out)
        .map(|o| (0..kin).map(|i| f64::from(w[o * kin + i]) * x[i]).sum())
        .collect()
}

/// `F.rms_norm(x, (n,), weight, eps)`.
fn rms(x: &[f64], w: &[f32]) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + EPS).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, g)| v * inv * f64::from(*g))
        .collect()
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// `apply_rope` on one head vector at absolute position `pos`, with
/// `build_rope_cache`'s `cat(freqs, freqs)` table.
fn rope(u: &[f64], pos: usize, base: f64) -> Vec<f64> {
    let dh = u.len();
    let half = dh / 2;
    let mut out = vec![0.0; dh];
    for i in 0..dh {
        let f = i % half;
        let angle = pos as f64 * base.powf(-((2 * f) as f64) / dh as f64);
        let rot = if i < half { -u[i + half] } else { u[i - half] };
        out[i] = u[i] * angle.cos() + rot * angle.sin();
    }
    out
}

/// Logits for every position, f64, transcribed from nanolab.
fn reference(fx: &Fixture, tokens: &[u32]) -> Vec<Vec<f64>> {
    let cfg = &fx.cfg;
    let (d, dh, nh, nkv) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_kv_head);
    let rep = nh / nkv;
    let tlen = tokens.len();
    let mut x: Vec<Vec<f64>> = tokens
        .iter()
        .map(|&id| {
            let row = id as usize * d;
            fx.emb[row..row + d].iter().map(|&v| f64::from(v)).collect()
        })
        .collect();
    let mut v0: Option<Vec<Vec<f64>>> = None;
    for b in &fx.blocks {
        let h: Vec<Vec<f64>> = x.iter().map(|row| rms(row, &b.ln1)).collect();
        let mut q = Vec::with_capacity(tlen);
        let mut k = Vec::with_capacity(tlen);
        let mut v = Vec::with_capacity(tlen);
        let mut raw_v = Vec::with_capacity(tlen);
        for (pos, hrow) in h.iter().enumerate() {
            let qf = mat_vec(&b.wq, hrow, nh * dh);
            let kf = mat_vec(&b.wk, hrow, nkv * dh);
            let vf = mat_vec(&b.wv, hrow, nkv * dh);
            let mut qr = Vec::with_capacity(nh * dh);
            for head in 0..nh {
                let n = rms(&qf[head * dh..(head + 1) * dh], &b.q_norm);
                qr.extend(rope(&n, pos, cfg.rope_base));
            }
            let mut kr = Vec::with_capacity(nkv * dh);
            for head in 0..nkv {
                let n = rms(&kf[head * dh..(head + 1) * dh], &b.k_norm);
                kr.extend(rope(&n, pos, cfg.rope_base));
            }
            raw_v.push(vf.clone());
            let blended = match &v0 {
                None => vf,
                Some(v0) => {
                    let s = sigmoid(f64::from(b.vr_lambda));
                    vf.iter()
                        .zip(&v0[pos])
                        .map(|(a, z)| (1.0 - s) * a + s * z)
                        .collect()
                }
            };
            q.push(qr);
            k.push(kr);
            v.push(blended);
        }
        if v0.is_none() {
            v0 = Some(raw_v);
        }
        let scale = 1.0 / (dh as f64).sqrt();
        for pos in 0..tlen {
            let mut y = vec![0.0f64; nh * dh];
            for head in 0..nh {
                let kvh = head / rep;
                let qh = &q[pos][head * dh..(head + 1) * dh];
                // Keys `pos - W < j <= pos` under a sliding window.
                let first = cfg.window.map_or(0, |w| (pos + 1).saturating_sub(w));
                let scores: Vec<f64> = (first..=pos)
                    .map(|j| {
                        let kj = &k[j][kvh * dh..(kvh + 1) * dh];
                        scale * qh.iter().zip(kj).map(|(a, c)| a * c).sum::<f64>()
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = ex.iter().sum();
                let gate = sigmoid(
                    (0..d)
                        .map(|i| f64::from(b.gate_w[head * d + i]) * h[pos][i])
                        .sum::<f64>()
                        + f64::from(b.gate_b[head]),
                );
                for (j, e) in (first..=pos).zip(&ex) {
                    let p = e / sum;
                    for c in 0..dh {
                        y[head * dh + c] += p * v[j][kvh * dh + c];
                    }
                }
                for c in 0..dh {
                    y[head * dh + c] *= gate;
                }
            }
            let proj = mat_vec(&b.wo, &y, d);
            for i in 0..d {
                x[pos][i] += proj[i];
            }
        }
        for row in x.iter_mut() {
            let h2 = rms(row, &b.ln2);
            let g = mat_vec(&b.w_gate, &h2, cfg.hidden);
            let u = mat_vec(&b.w_up, &h2, cfg.hidden);
            let hid: Vec<f64> = g.iter().zip(&u).map(|(g, u)| g * sigmoid(*g) * u).collect();
            let down = mat_vec(&b.w_down, &hid, d);
            for i in 0..d {
                row[i] += down[i];
            }
        }
    }
    x.iter()
        .map(|row| mat_vec(&fx.emb, &rms(row, &fx.ln_f), cfg.vocab))
        .collect()
}

/// Max absolute error over the larger of 1 and the reference's magnitude.
/// `f64::max` drops NaN, so a non-finite value on either side is measured
/// as infinite before the fold rather than vanishing from it.
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    if got.iter().any(|g| !g.is_finite()) || want.iter().any(|w| !w.is_finite()) {
        return f64::INFINITY;
    }
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs())
        .fold(0.0, f64::max)
        / scale
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn cfg_mha() -> GptConfig {
    GptConfig {
        vocab: 50,
        n_embd: 32,
        n_head: 4,
        n_kv_head: 4,
        head_dim: 8,
        n_layer: 2,
        hidden: 48,
        max_seq: 24,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
        window: None,
    }
}

/// GQA (2 query heads per KV head) and `n_head * head_dim != n_embd`.
fn cfg_gqa() -> GptConfig {
    GptConfig {
        vocab: 50,
        n_embd: 32,
        n_head: 4,
        n_kv_head: 2,
        head_dim: 16,
        n_layer: 3,
        hidden: 40,
        max_seq: 24,
        rope_base: 500.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
        window: None,
    }
}

fn cpu(numerics: Numerics) -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 28)).with_numerics(numerics)
}

/// Logits `[len, vocab]` of `ojas_model::forward_logits` on
/// `Eval<CpuBackend>`, positions from 0.
fn eval_logits(fx: &Fixture, tokens: &[u32], numerics: Numerics) -> Result<Vec<f32>, OjasError> {
    let mut g = Eval::new(cpu(numerics));
    let budget = g.backend().budget().clone();
    let params = bind(&mut g, &fx.cfg, &fx.weights.clone().into_flat())?;
    let rope = Rope::new(&fx.cfg, tokens.len(), &budget)?;
    let ids = Tensor::from_u32(tokens, &[1, tokens.len()], &budget)?;
    forward_logits(&mut g, &fx.cfg, &params, &ids, &rope)?.to_f32_vec()
}

fn decoder(fx: &Fixture, numerics: Numerics) -> DeviceDecoder<CpuBackend> {
    DeviceDecoder::new(cpu(numerics), &fx.cfg, &fx.weights, fx.cfg.max_seq).unwrap()
}

/// `DeviceDecoder<CpuBackend>` prefill of every prefix of `tokens` from an
/// empty cache, last rows stacked: `[len, vocab]`.
fn prefill_logits(fx: &Fixture, tokens: &[u32], numerics: Numerics) -> Vec<f32> {
    let mut dec = decoder(fx, numerics);
    let mut out = Vec::with_capacity(tokens.len() * fx.cfg.vocab);
    for end in 1..=tokens.len() {
        dec.reset();
        out.extend(dec.forward(&tokens[..end]).unwrap());
        assert_eq!(dec.len(), end);
    }
    out
}

/// The full forward of the module docs: `Eval`, MHA and GQA alike.
fn full_logits(fx: &Fixture, tokens: &[u32]) -> Vec<f32> {
    eval_logits(fx, tokens, fx.model.numerics()).unwrap()
}

const TOKENS: [u32; 16] = [3, 17, 0, 49, 9, 9, 22, 41, 5, 30, 12, 7, 48, 1, 26, 33];

fn check_three_way(cfg: GptConfig, seed: u64) {
    check_three_way_under(cfg, seed, None);
}

/// [`check_three_way`] with the model's `CpuBackend` ops under `numerics`
/// (`None` keeps the default). Returns the full-sequence logits.
fn check_three_way_under(cfg: GptConfig, seed: u64, numerics: Option<Numerics>) -> Vec<f32> {
    let budget = Budget::new(1 << 26);
    let mut fx = fixture(cfg, seed, &budget);
    if let Some(numerics) = numerics {
        fx.model = fx.model.with_numerics(numerics);
        assert_eq!(fx.model.numerics(), numerics);
    }
    let want = reference(&fx, &TOKENS);
    let full = full_logits(&fx, &TOKENS);
    let prefill = prefill_logits(&fx, &TOKENS, fx.model.numerics());
    let vocab = fx.cfg.vocab;
    assert_eq!(full.len(), TOKENS.len() * vocab);
    assert_eq!(prefill.len(), full.len());
    let mut cache = KvCache::for_model(&fx.model, fx.cfg.max_seq, &budget).unwrap();
    let mut worst = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (pos, &id) in TOKENS.iter().enumerate() {
        let step = fx.model.forward_token(id, &mut cache).unwrap();
        assert_eq!(cache.len(), pos + 1);
        let row = &full[pos * vocab..(pos + 1) * vocab];
        let pre = &prefill[pos * vocab..(pos + 1) * vocab];
        worst.0 = worst.0.max(rel_err(&step, &widen(row)));
        worst.1 = worst.1.max(rel_err(&step, &want[pos]));
        worst.2 = worst.2.max(rel_err(row, &want[pos]));
        worst.3 = worst.3.max(rel_err(pre, &widen(row)));
    }
    eprintln!(
        "n_kv_head {}: cached-vs-full {:e}, cached-vs-ref {:e}, full-vs-ref {:e}, \
         decoder-prefill-vs-full {:e}",
        fx.cfg.n_kv_head, worst.0, worst.1, worst.2, worst.3
    );
    assert!(worst.0 <= PATH_TOL, "cached vs full forward {:e}", worst.0);
    assert!(worst.1 <= REF_TOL, "cached vs f64 reference {:e}", worst.1);
    assert!(worst.2 <= REF_TOL, "full vs f64 reference {:e}", worst.2);
    assert!(worst.3 <= PATH_TOL, "decoder prefill vs full {:e}", worst.3);
    if numerics == Some(Numerics::Exact) {
        // Every path then sums every reduction in ascending order with
        // separate rounding, so the logits agree bit for bit, MHA and GQA.
        let mut cache = KvCache::for_model(&fx.model, fx.cfg.max_seq, &budget).unwrap();
        for (pos, &id) in TOKENS.iter().enumerate() {
            let step = fx.model.forward_token(id, &mut cache).unwrap();
            let row = &full[pos * vocab..(pos + 1) * vocab];
            let pre = &prefill[pos * vocab..(pos + 1) * vocab];
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&step),
                bits(row),
                "Exact cached vs full, position {pos}"
            );
            assert_eq!(
                bits(pre),
                bits(row),
                "Exact prefill vs full, position {pos}"
            );
        }
    }
    full
}

#[test]
fn cached_decode_matches_full_forward_and_reference_mha() {
    check_three_way(cfg_mha(), 0x5EED_0001);
}

/// The default follows `CpuBackend`'s (Fast), and both settings keep the
/// three-way parity on MHA and GQA.
#[test]
fn both_numerics_keep_three_way_parity() {
    let budget = Budget::new(1 << 26);
    assert_eq!(
        fixture(cfg_mha(), 1, &budget).model.numerics(),
        Numerics::Fast
    );
    for numerics in [Numerics::Exact, Numerics::Fast] {
        check_three_way_under(cfg_mha(), 0x5EED_0001, Some(numerics));
        check_three_way_under(cfg_gqa(), 0x5EED_0002, Some(numerics));
    }
}

/// A vocabulary large enough that the full forward's LM head is
/// `16 * 32 * 4096 = 2^21` multiply-adds, at or above the size at which a
/// Fast GEMM leaves the packed FMA kernel on every platform
/// (`ojas_cpu::FAST_WHOLE_CALL_MACS`: 2^21 off macOS, 2^13 on it, where
/// the call is Accelerate). Parity must hold there too, under both settings.
#[test]
fn three_way_parity_holds_where_fast_gemms_take_the_whole_call_path() {
    let cfg = GptConfig {
        vocab: 4096,
        ..cfg_mha()
    };
    assert_eq!(TOKENS.len() * cfg.n_embd * cfg.vocab, 1 << 21);
    const { assert!(1 << 21 >= ojas_cpu::FAST_WHOLE_CALL_MACS) };
    for numerics in [Numerics::Exact, Numerics::Fast] {
        check_three_way_under(cfg, 0x5EED_0003, Some(numerics));
    }
}

/// `with_numerics` reaches the full forward: Exact repeats bit for bit, and
/// where the target fuses multiply-adds, Fast gives different bits.
#[test]
fn with_numerics_selects_the_full_forward_arithmetic() {
    let exact = check_three_way_under(cfg_mha(), 0x5EED_0004, Some(Numerics::Exact));
    let again = check_three_way_under(cfg_mha(), 0x5EED_0004, Some(Numerics::Exact));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&exact), bits(&again));
    let fast = check_three_way_under(cfg_mha(), 0x5EED_0004, Some(Numerics::Fast));
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        assert_ne!(
            bits(&exact),
            bits(&fast),
            "Fast did not reach the full forward"
        );
    }
}

#[test]
fn cached_decode_matches_full_forward_and_reference_gqa_wide_heads() {
    check_three_way(cfg_gqa(), 0x5EED_0002);
}

#[test]
fn rel_err_measures_a_non_finite_value_as_infinite() {
    let want = [1.0f64, -2.0, 0.5];
    for got in [
        [f32::NAN; 3],
        [1.0, f32::NAN, 0.5],
        [1.0, -2.0, f32::INFINITY],
    ] {
        assert_eq!(rel_err(&got, &want), f64::INFINITY, "{got:?}");
    }
    assert_eq!(
        rel_err(&[1.0, -2.0, 0.5], &[1.0, f64::NAN, 0.5]),
        f64::INFINITY
    );
    assert_eq!(rel_err(&[1.0, -2.0, 0.5], &want), 0.0);
}

/// A second prompt appended to a warm cache sits at absolute positions
/// `len(A)..`, not at 0.
#[test]
fn continuing_a_warm_cache_equals_one_forward_over_the_concatenation() {
    let budget = Budget::new(1 << 26);
    let fx = fixture(cfg_mha(), 0x5EED_0003, &budget);
    let (a, b) = TOKENS.split_at(6);
    let mut cache = KvCache::for_model(&fx.model, fx.cfg.max_seq, &budget).unwrap();
    for &id in a {
        fx.model.forward_token(id, &mut cache).unwrap();
    }
    let full = full_logits(&fx, &TOKENS);
    let vocab = fx.cfg.vocab;
    for (i, &id) in b.iter().enumerate() {
        let pos = a.len() + i;
        let step = fx.model.forward_token(id, &mut cache).unwrap();
        let row = &full[pos * vocab..(pos + 1) * vocab];
        let err = rel_err(&step, &widen(row));
        assert!(err <= PATH_TOL, "position {pos}: {err:e}");
    }
    // A fresh cache fed only B puts B at position 0: different logits.
    let mut fresh = KvCache::for_model(&fx.model, fx.cfg.max_seq, &budget).unwrap();
    let first = fx.model.forward_token(b[0], &mut fresh).unwrap();
    let at = a.len() * vocab;
    let err = rel_err(&first, &widen(&full[at..at + vocab]));
    assert!(
        err.is_finite() && err > 1e-3,
        "fresh cache vs position {}: {err:e}",
        a.len()
    );

    // The device decoder, on MHA and GQA: A as one prefill, then B as one
    // prefill onto the warm cache (its last row), and B again one token at
    // a time after a reset and a fresh A.
    for (cfg, seed) in [(cfg_mha(), 0x5EED_0003), (cfg_gqa(), 0x5EED_0005)] {
        let fx = fixture(cfg, seed, &budget);
        let full = full_logits(&fx, &TOKENS);
        let vocab = fx.cfg.vocab;
        let row = |pos: usize| widen(&full[pos * vocab..(pos + 1) * vocab]);
        let mut dec = decoder(&fx, fx.model.numerics());
        let a_last = dec.forward(a).unwrap();
        assert!(rel_err(&a_last, &row(a.len() - 1)) <= PATH_TOL);
        let b_last = dec.forward(b).unwrap();
        assert_eq!(dec.len(), TOKENS.len());
        let err = rel_err(&b_last, &row(TOKENS.len() - 1));
        assert!(err <= PATH_TOL, "prefill onto a warm cache: {err:e}");
        dec.reset();
        dec.forward(a).unwrap();
        for (i, &id) in b.iter().enumerate() {
            let pos = a.len() + i;
            let err = rel_err(&dec.forward(&[id]).unwrap(), &row(pos));
            assert!(err <= PATH_TOL, "decoder position {pos}: {err:e}");
        }
        // B alone from an empty decoder sits at position 0: different.
        dec.reset();
        let err = rel_err(&dec.forward(&b[..1]).unwrap(), &row(a.len()));
        assert!(err.is_finite() && err > 1e-3, "decoder fresh B: {err:e}");
    }
}

/// Without positional encoding the last position cannot tell `[x, y, z]`
/// from `[y, x, z]`. With RoPE it can.
#[test]
fn swapping_two_earlier_tokens_changes_the_last_logits() {
    let budget = Budget::new(1 << 26);
    let fx = fixture(cfg_mha(), 0x5EED_0004, &budget);
    let last = |ids: &[u32]| {
        let mut cache = KvCache::for_model(&fx.model, fx.cfg.max_seq, &budget).unwrap();
        let mut out = Vec::new();
        for &id in ids {
            out = fx.model.forward_token(id, &mut cache).unwrap();
        }
        out
    };
    let ab = last(&[4, 11, 20]);
    let ba = last(&[11, 4, 20]);
    assert!(
        ab.iter().chain(&ba).all(|v| v.is_finite()),
        "non-finite logits"
    );
    let diff = ab
        .iter()
        .zip(&ba)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(diff > 1e-3, "order-blind: max diff {diff:e}");
}

/// Each bad spec is refused by both decoders. The weights are the valid
/// MHA fixture's, and the unchanged spec builds both, so each refusal is
/// the spec's. (Before the move to `ModelSpec` this test used `n_layer: 0`
/// with no blocks; `ModelSpec::validate` refuses `n_layer == 0` first, which
/// would have made every case pass for that reason alone.)
#[test]
fn shapes_the_block_cannot_run_are_refused() {
    let budget = Budget::new(1 << 26);
    let base = cfg_mha();
    let fx = fixture(base, 9, &budget);
    assert!(CpuGpt::new(&base, &fx.weights).is_ok());
    assert!(DeviceDecoder::new(cpu(Numerics::Exact), &base, &fx.weights, 4).is_ok());
    for (name, cfg) in [
        (
            "odd head_dim",
            GptConfig {
                head_dim: 7,
                ..base
            },
        ),
        (
            "n_head not a multiple of n_kv_head",
            GptConfig {
                n_kv_head: 3,
                ..base
            },
        ),
        (
            "zero kv heads",
            GptConfig {
                n_kv_head: 0,
                ..base
            },
        ),
        (
            "non-finite rope base",
            GptConfig {
                rope_base: f64::NAN,
                ..base
            },
        ),
        (
            "rope base 0",
            GptConfig {
                rope_base: 0.0,
                ..base
            },
        ),
        (
            "untied head",
            GptConfig {
                tie_embeddings: false,
                ..base
            },
        ),
    ] {
        let err = match CpuGpt::new(&cfg, &fx.weights) {
            Err(err) => err,
            Ok(_) => panic!("{name}: accepted"),
        };
        assert!(
            matches!(
                err,
                OjasError::Shape { .. }
                    | OjasError::OutOfRange { .. }
                    | OjasError::Unsupported { .. }
            ),
            "{name}: {err}"
        );
        assert!(
            DeviceDecoder::new(cpu(Numerics::Exact), &cfg, &fx.weights, 4).is_err(),
            "{name}: decoder accepted"
        );
    }
}

/// Seeded sampled continuation of the MHA fixture, pinned: a change to the
/// block, the kernels it calls, the sampler or the RNG moves it. The
/// fixture's logits are held to the f64 reference by the tests above. The
/// fixture's tied head is sharply peaked (greedy repeats one token), hence
/// temperature 6. Recorded 2026-10-01 from this implementation.
const GOLDEN_SAMPLED: [u32; 12] = [30, 30, 30, 30, 44, 24, 24, 11, 11, 20, 49, 18];

#[test]
fn greedy_follows_the_reference_argmax_and_sampling_matches_the_golden() {
    let budget = Budget::new(1 << 26);
    let fx = fixture(cfg_mha(), 0x5EED_0001, &budget);
    let prompt = &TOKENS[..5];
    let mut cache = KvCache::for_model(&fx.model, fx.cfg.max_seq, &budget).unwrap();
    let out = fx.model.greedy_decode(prompt, &mut cache, 12).unwrap();
    let mut seq = prompt.to_vec();
    seq.extend_from_slice(&out[..out.len() - 1]);
    let want = reference(&fx, &seq);
    for (i, &tok) in out.iter().enumerate() {
        let row = &want[prompt.len() - 1 + i];
        let best = (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)))
            .unwrap();
        assert_eq!(tok as usize, best, "step {i}: reference argmax differs");
    }

    let g = GenerateConfig {
        sampling: SamplingConfig {
            temperature: 6.0,
            top_k: Some(20),
            top_p: Some(0.95),
        },
        seed: 2024,
        max_new_tokens: 12,
        stop_tokens: Vec::new(),
    };
    let mut cache = KvCache::for_model(&fx.model, fx.cfg.max_seq, &budget).unwrap();
    let sampled = fx.model.generate(prompt, &mut cache, &g).unwrap();
    assert_eq!(sampled, GOLDEN_SAMPLED);

    // The device decoder runs the same loop and sampler over its own
    // forward (prompt as one prefill): the same ids.
    let mut dec = decoder(&fx, fx.model.numerics());
    assert_eq!(dec.greedy_decode(prompt, 12).unwrap(), out);
    assert_eq!(dec.len(), prompt.len() + 11);
    dec.reset();
    assert_eq!(dec.generate(prompt, &g).unwrap(), GOLDEN_SAMPLED);
}

/// Both multi-token entry points refuse an empty, too long or out-of-vocab
/// input: `Eval`'s `forward_logits` (the full forward) and
/// `DeviceDecoder::forward` (prefill). A refused prefill leaves the
/// decoder's length where it was.
#[test]
fn full_forward_refuses_empty_too_long_and_out_of_vocab_input() {
    let budget = Budget::new(1 << 26);
    let fx = fixture(cfg_mha(), 5, &budget);
    let long = vec![1u32; fx.cfg.max_seq + 1];
    assert!(eval_logits(&fx, &[], Numerics::Exact).is_err());
    assert!(eval_logits(&fx, &long, Numerics::Exact).is_err());
    assert!(eval_logits(&fx, &[1, 50], Numerics::Exact).is_err());

    let mut dec = decoder(&fx, Numerics::Exact);
    dec.forward(&[1, 2]).unwrap();
    assert!(matches!(dec.forward(&[]), Err(OjasError::Shape { .. })));
    assert!(matches!(
        dec.forward(&long[..fx.cfg.max_seq - 1]),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert!(matches!(
        dec.forward(&[1, 50]),
        Err(OjasError::OutOfRange { .. })
    ));
    assert_eq!(dec.len(), 2);
    // Exactly the free positions fit; one more is refused.
    dec.forward(&long[..fx.cfg.max_seq - 2]).unwrap();
    assert_eq!(dec.remaining(), 0);
    assert!(matches!(
        dec.forward(&[1]),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert_eq!(dec.len(), fx.cfg.max_seq);
    assert!(DeviceDecoder::new(cpu(Numerics::Exact), &fx.cfg, &fx.weights, 0).is_err());
    assert!(DeviceDecoder::new(
        cpu(Numerics::Exact),
        &fx.cfg,
        &fx.weights,
        fx.cfg.max_seq + 1
    )
    .is_err());
}

/// One position's logits under each path.
struct G7Row {
    token: Vec<f32>,
    eval: Vec<f32>,
    device: Vec<f32>,
}

/// Prompt `TOKENS[..PROMPT]` as one decoder prefill (and token by token on
/// `CpuGpt`), then every later token of `TOKENS` decoded one at a time,
/// fed the same id on every path. Rows from the last prompt position on.
fn g7_rows(fx: &Fixture, numerics: Numerics) -> Vec<G7Row> {
    const PROMPT: usize = 4;
    let model = CpuGpt::new(&fx.cfg, &fx.weights)
        .unwrap()
        .with_numerics(numerics);
    let budget = Budget::new(1 << 26);
    let mut cache = KvCache::for_model(&model, fx.cfg.max_seq, &budget).unwrap();
    let mut dec = decoder(fx, numerics);
    let readbacks = dec.backend().budget().device_readbacks();
    let full = eval_logits(fx, &TOKENS, numerics).unwrap();
    let vocab = fx.cfg.vocab;
    let eval_row = |pos: usize| full[pos * vocab..(pos + 1) * vocab].to_vec();
    let mut token = Vec::new();
    for &id in &TOKENS[..PROMPT] {
        token = model.forward_token(id, &mut cache).unwrap();
    }
    let mut rows = vec![G7Row {
        token,
        eval: eval_row(PROMPT - 1),
        device: dec.forward(&TOKENS[..PROMPT]).unwrap(),
    }];
    for (pos, &id) in TOKENS.iter().enumerate().skip(PROMPT) {
        rows.push(G7Row {
            token: model.forward_token(id, &mut cache).unwrap(),
            eval: eval_row(pos),
            device: dec.forward(&[id]).unwrap(),
        });
    }
    // A host backend reads nothing back.
    assert_eq!(dec.backend().budget().device_readbacks(), readbacks);
    assert_eq!(dec.len(), TOKENS.len());
    rows
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// G7: `CpuGpt::forward_token`, `Eval<CpuBackend>` (`forward_logits`) and
/// `DeviceDecoder<CpuBackend>` agree to
/// [`G7_TOL`] over a 4-token prefill and 12 decode steps, on MHA and GQA,
/// under both numerics. Under `Numerics::Exact` all three are bit-identical:
/// the CPU's fused `rms_qk_norm_forward` runs the same row kernel as two
/// `rms_norm_forward` calls, `cached_attention_forward` runs causal SDPA's
/// per-row kernel, and every Exact reduction sums in ascending order.
#[test]
fn g7_forward_token_eval_and_device_decoder_agree() {
    let budget = Budget::new(1 << 26);
    for (cfg, seed) in [(cfg_mha(), 0x5EED_0007), (cfg_gqa(), 0x5EED_0008)] {
        let fx = fixture(cfg, seed, &budget);
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let rows = g7_rows(&fx, numerics);
            let mut worst = (0.0f64, 0.0f64, 0.0f64);
            let mut bitwise = (true, true);
            for row in &rows {
                worst.0 = worst.0.max(rel_err(&row.device, &widen(&row.token)));
                bitwise.0 &= bits(&row.device) == bits(&row.token);
                worst.1 = worst.1.max(rel_err(&row.eval, &widen(&row.token)));
                worst.2 = worst.2.max(rel_err(&row.device, &widen(&row.eval)));
                bitwise.1 &= bits(&row.eval) == bits(&row.token);
            }
            eprintln!(
                "G7 n_head {} n_kv_head {} {numerics:?}: {} rows; device-vs-token {:e} \
                 (bitwise {}); eval-vs-token {:e} (bitwise {}); device-vs-eval {:e}",
                fx.cfg.n_head,
                fx.cfg.n_kv_head,
                rows.len(),
                worst.0,
                bitwise.0,
                worst.1,
                bitwise.1,
                worst.2
            );
            assert!(worst.0 <= G7_TOL, "device vs forward_token {:e}", worst.0);
            assert!(worst.1 <= G7_TOL, "Eval vs forward_token {:e}", worst.1);
            assert!(worst.2 <= G7_TOL, "device vs Eval {:e}", worst.2);
            if numerics == Numerics::Exact {
                assert!(
                    bitwise.0,
                    "Exact: device decoder vs forward_token not bitwise"
                );
                assert!(bitwise.1, "Exact: Eval vs forward_token not bitwise");
            }
        }
    }
}

/// A sliding window `W` with `2W - 1` below `max_seq`, so both caches are
/// rings and the 16 tokens wrap them. Token-by-token decode (`CpuGpt`),
/// prefill of every prefix (`DeviceDecoder`, `W`-token pieces), the full
/// windowed forward (`Eval`, `causal_sdpa_forward` with the window) and
/// the windowed f64 reference agree, bit for bit under Exact.
#[test]
fn sliding_window_decode_matches_full_forward_and_reference() {
    for (cfg, w, seed) in [(cfg_mha(), 3, 0x5EED_0101), (cfg_gqa(), 4, 0x5EED_0102)] {
        let cfg = GptConfig {
            window: Some(w),
            ..cfg
        };
        let budget = Budget::new(1 << 26);
        let fx = fixture(cfg, seed, &budget);
        let cache = KvCache::for_model(&fx.model, cfg.max_seq, &budget).unwrap();
        assert_eq!((cache.max_len(), cache.slots()), (cfg.max_seq, 2 * w - 1));
        assert_eq!(decoder(&fx, Numerics::Exact).slots(), 2 * w - 1);
        assert!(TOKENS.len() > 2 * (2 * w - 1), "the rings wrap twice");
        for numerics in [Numerics::Exact, Numerics::Fast] {
            check_three_way_under(cfg, seed, Some(numerics));
        }
        // The window changes the logits: past position W - 1 they are not
        // the full-causal ones.
        let windowed = full_logits(&fx, &TOKENS);
        let fx_full = fixture(
            GptConfig {
                window: None,
                ..cfg
            },
            seed,
            &budget,
        );
        let causal = full_logits(&fx_full, &TOKENS);
        let v = cfg.vocab;
        assert_eq!(windowed[..w * v], causal[..w * v]);
        assert_ne!(windowed[w * v..], causal[w * v..]);
    }
}

/// A window of at least `max_seq` is full causal attention, and the cache
/// is not a ring.
#[test]
fn a_window_covering_max_seq_is_full_causal() {
    let budget = Budget::new(1 << 26);
    let cfg = cfg_mha();
    let wide = GptConfig {
        window: Some(cfg.max_seq),
        ..cfg
    };
    assert_eq!(wide.attention_window(), None);
    let fx = fixture(wide, 0x5EED_0103, &budget);
    let cache = KvCache::for_model(&fx.model, cfg.max_seq, &budget).unwrap();
    assert_eq!(cache.slots(), cfg.max_seq);
    let fx_full = fixture(cfg, 0x5EED_0103, &budget);
    assert_eq!(full_logits(&fx, &TOKENS), full_logits(&fx_full, &TOKENS));
}

/// On a ring of `2W - 1` slots, rolling back keeps working while the
/// prefix's window is still held, and is refused, changing nothing, once
/// it has been overwritten. A cache shaped without the window is refused
/// by a windowed model (it would attend without one).
#[test]
fn a_windowed_ring_rolls_back_only_while_it_holds_the_prefix_window() {
    let w = 4;
    let cfg = GptConfig {
        window: Some(w),
        ..cfg_gqa()
    };
    let budget = Budget::new(1 << 26);
    let mut fx = fixture(cfg, 0x5EED_0104, &budget);
    fx.model = fx.model.with_numerics(Numerics::Exact);
    let model = &fx.model;
    let n = TOKENS.len();
    let mut cache = KvCache::for_model(model, cfg.max_seq, &budget).unwrap();
    let full = model.forward_tokens(&TOKENS, &mut cache).unwrap();
    assert_eq!(cache.len(), n);
    // The ring holds positions n - 7..n; regenerating from n - 3 needs
    // n - 6..n - 3.
    cache.truncate(n - 3).unwrap();
    assert_eq!(
        model.forward_tokens(&TOKENS[n - 3..], &mut cache).unwrap(),
        full
    );
    assert!(matches!(
        cache.truncate(n - 5),
        Err(OjasError::OutOfRange { .. })
    ));
    assert_eq!(cache.len(), n);
    cache.reset();
    assert_eq!(model.forward_tokens(&TOKENS, &mut cache).unwrap(), full);

    let mut dec = decoder(&fx, Numerics::Exact);
    let got = dec.forward(&TOKENS).unwrap();
    dec.truncate(n - 3).unwrap();
    assert_eq!(dec.forward(&TOKENS[n - 3..]).unwrap(), got);
    assert!(matches!(
        dec.truncate(n - 5),
        Err(OjasError::OutOfRange { .. })
    ));

    let mut plain = KvCache::new(
        cfg.n_layer,
        cfg.n_kv_head,
        cfg.head_dim,
        cfg.max_seq,
        &budget,
    )
    .unwrap();
    assert!(matches!(
        model.forward_token(TOKENS[0], &mut plain),
        Err(OjasError::Shape { .. })
    ));
}
