//! Adversarial tests for LoRA on the Tape (card ft-6fb212ce): wrapper
//! forwarding of the dX-only backward, config and init edges, merge/unmerge
//! over hostile weights, target discovery against a hostile index, and
//! whether bounds (a') and (b') can tell a wrong ojas from a right one.
//!
//! A test that exposes a real break is `#[ignore = "BREAK: ..."]`.

mod qwen35_common;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_autograd::Tape;
use ojas_core::{
    f32_to_bf16, forward_backend_methods, AdamWConfig, Autocast, AutocastMode, Backend, DType,
    OjasError, Tensor,
};
use ojas_cpu::CpuBackend;
use ojas_data::CounterRng;
use ojas_io::{encode_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::lora::{dropout_mask, init_a, init_b, LoraConfig, LoraError, LoraLinear};
use ojas_model::qwen35::{
    bind_frozen, forward_loss_lora, load_hf, lora_targets, LoraForward, LoraKind, Qwen35Lora,
    Qwen35Params, LORA_KINDS,
};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use qwen35_common::lora::*;
use qwen35_common::*;

// ---- wrapper forwarding -------------------------------------------------

macro_rules! all_but_linear_backward {
    (linear_backward_input, { $($item:tt)* }) => {};
    (linear_backward, { $($item:tt)* }) => {};
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

/// A CPU backend that counts which of the two linear backwards ran.
struct Spy {
    inner: CpuBackend,
    dx_only: Arc<AtomicUsize>,
    full: Arc<AtomicUsize>,
}

impl Backend for Spy {
    ojas_core::with_backend_methods!(
        forward_backend_methods,
        { self, self.inner, none, all_but_linear_backward }
    );
    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.full.fetch_add(1, Ordering::SeqCst);
        self.inner.linear_backward(input, weight, grad_output)
    }
    fn linear_backward_input(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.dx_only.fetch_add(1, Ordering::SeqCst);
        self.inner.linear_backward_input(input, weight, grad_output)
    }
}

fn spy() -> (Spy, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let dx_only = Arc::new(AtomicUsize::new(0));
    let full = Arc::new(AtomicUsize::new(0));
    let s = Spy {
        inner: CpuBackend::new(ojas_core::Budget::new(1 << 26)),
        dx_only: Arc::clone(&dx_only),
        full: Arc::clone(&full),
    };
    (s, dx_only, full)
}

fn tens(seed: u32, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| {
            (((i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed * 97))
                % 1000) as f32
                / 500.0
                - 1.0
        })
        .collect();
    Tensor::from_f32(&v, shape, &budget()).unwrap()
}

/// One frozen-weight linear through `Autocast<Spy>` reaches the inner
/// `linear_backward_input` and never the full `linear_backward`, inside and
/// outside a bf16 region; a trainable weight takes the full path.
#[test]
fn autocast_forwards_dx_only_to_the_inner_backend() {
    for region in [false, true] {
        for frozen in [true, false] {
            let (s, dx_only, full) = spy();
            let mut tape = Tape::new(Autocast::new(s));
            let _g = region.then(|| tape.backend().autocast_region(AutocastMode::Bf16).unwrap());
            let x = tape.leaf(tens(1, &[4, 8])).unwrap();
            let w = if frozen {
                tape.frozen_leaf(tens(2, &[6, 8])).unwrap()
            } else {
                tape.leaf(tens(2, &[6, 8])).unwrap()
            };
            let y = tape.linear(x, w).unwrap();
            tape.backward(y).unwrap();
            let (d, f) = (dx_only.load(Ordering::SeqCst), full.load(Ordering::SeqCst));
            if frozen {
                assert_eq!((d, f), (1, 0), "region={region}: dx_only={d} full={f}");
            } else {
                assert_eq!((d, f), (0, 1), "region={region}: dx_only={d} full={f}");
            }
        }
    }
}

/// Inside a bf16 Autocast region the frozen path's dX is bit-equal to the
/// trainable path's dX, including with a bf16 resident weight.
#[test]
fn autocast_dx_only_bits_equal_the_full_path_in_a_bf16_region() {
    let run = |frozen: bool, bf16_w: bool| {
        let mut tape = Tape::new(Autocast::new(CpuBackend::new(ojas_core::Budget::new(
            1 << 26,
        ))));
        let _g = tape.backend().autocast_region(AutocastMode::Bf16).unwrap();
        let x = tape.leaf(tens(1, &[5, 24])).unwrap();
        let mut wv = tens(2, &[7, 24]);
        if bf16_w {
            wv = tape.backend().to_bf16(&wv).unwrap();
        }
        let w = if frozen {
            tape.frozen_leaf(wv).unwrap()
        } else {
            tape.leaf(wv).unwrap()
        };
        let y = tape.linear(x, w).unwrap();
        tape.backward(y).unwrap();
        tape.grad(x)
            .unwrap()
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    for bf16_w in [false, true] {
        assert_eq!(
            run(true, bf16_w),
            run(false, bf16_w),
            "bf16 weight = {bf16_w}"
        );
    }
}

// ---- config, init and dropout edges --------------------------------------

#[test]
fn config_edges() {
    for p in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
        1.000_000_1,
        -1e-9,
        2.0,
    ] {
        assert!(
            matches!(LoraConfig::new(4, 8.0, p), Err(LoraError::BadDropout(_))),
            "dropout {p}"
        );
    }
    for p in [0.0, -0.0, 0.5, 0.999_999] {
        LoraConfig::new(4, 8.0, p).unwrap_or_else(|e| panic!("dropout {p}: {e}"));
    }
    for a in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, -1e-30] {
        assert!(
            matches!(LoraConfig::new(4, a, 0.0), Err(LoraError::BadAlpha(_))),
            "alpha {a}"
        );
    }
    // alpha = 0, -0.0 and a subnormal are finite and non-negative.
    for a in [0.0, -0.0, 1e-42, f32::MAX] {
        LoraConfig::new(4, a, 0.0).unwrap_or_else(|e| panic!("alpha {a}: {e}"));
    }
    assert!(matches!(
        LoraConfig::new(0, 8.0, 0.0),
        Err(LoraError::ZeroRank)
    ));
}

/// r above min(in, out) on every real target is a typed error and r equal
/// to min is accepted.
#[test]
fn rank_above_min_dimension_is_a_typed_error_on_every_target() {
    let (s, t) = (spec(), targets());
    let min_dim = t
        .iter()
        .map(|x| x.in_features.min(x.out_parts.iter().sum()))
        .min()
        .unwrap();
    let ok = LoraConfig::new(min_dim, 8.0, 0.0).unwrap();
    Qwen35Lora::init(&s, &t, &ok, 1, &budget()).unwrap();
    let too_big = LoraConfig::new(min_dim + 1, 8.0, 0.0).unwrap();
    let err = Qwen35Lora::init(&s, &t, &too_big, 1, &budget()).unwrap_err();
    assert!(err.to_string().contains("rank"), "{err}");
}

/// PEFT init: A ~ U(-1/sqrt(in), 1/sqrt(in)), the distribution
/// `kaiming_uniform_(a = sqrt(5))` draws on a `[r, in]` weight (fan_in =
/// in). Variance is bound^2 / 3 and the extremes reach the bound.
#[test]
fn a_matches_kaiming_uniform_sqrt5_moments() {
    let (r, inn) = (64usize, 4096usize);
    let c = LoraConfig::new(r, 64.0, 0.0).unwrap();
    let a = init_a(&c, inn, &mut CounterRng::new(9), &budget())
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let bound = 1.0 / (inn as f64).sqrt();
    let n = a.len() as f64;
    let mean = a.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = a
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n;
    let max = a.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    assert!(mean.abs() < 0.01 * bound, "mean {mean}");
    assert!(
        (var / (bound * bound / 3.0) - 1.0).abs() < 0.01,
        "var ratio {}",
        var / (bound * bound / 3.0)
    );
    assert!(
        max <= bound && max > 0.999 * bound,
        "max {max} vs bound {bound}"
    );
    assert!(init_b(&c, 32, &budget())
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .all(|x| x.to_bits() == 0));
}

/// Dropout mask: i.i.d. per element (not per row), keep rate 1 - p, values
/// 0 or 1/(1-p); an extreme p still gives finite values.
#[test]
fn dropout_mask_statistics() {
    for p in [0.1f32, 0.5, 0.9, 0.999] {
        let m = dropout_mask(p, &[200, 500], &mut CounterRng::new(5), &budget())
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let keep = 1.0 / (1.0 - p);
        assert!(m.iter().all(|&x| x == 0.0 || x == keep), "p={p}");
        let kept = m.iter().filter(|&&x| x != 0.0).count() as f64 / m.len() as f64;
        let want = 1.0 - f64::from(p);
        // 100k draws: 5 sigma.
        let sigma = (want * (1.0 - want) / m.len() as f64).sqrt();
        assert!(
            (kept - want).abs() < 5.0 * sigma + 1e-9,
            "p={p}: kept {kept} want {want}"
        );
        // Not row-constant: some row has both kept and dropped (p not extreme).
        if p <= 0.9 {
            let row = &m[..500];
            assert!(
                row.contains(&0.0) && row.iter().any(|&x| x != 0.0),
                "p={p}: row-constant mask"
            );
        }
        // Expectation preserved.
        let mean = m.iter().map(|&x| f64::from(x)).sum::<f64>() / m.len() as f64;
        assert!(
            (mean - 1.0).abs() < 0.2 + 5.0 * sigma * f64::from(keep),
            "p={p}: E[mask] {mean}"
        );
    }
}

/// The public draw helpers take a rank or shape the config/shape check
/// would refuse: they must return a typed error, not panic or wrap.
#[test]
fn public_draw_helpers_do_not_panic_on_absurd_dimensions() {
    let huge = LoraConfig::new(usize::MAX / 2, 8.0, 0.0).unwrap();
    let b = budget();
    let a = catch_unwind(AssertUnwindSafe(|| {
        init_a(&huge, 4, &mut CounterRng::new(1), &b)
    }));
    assert!(
        matches!(a, Ok(Err(_))),
        "init_a: {:?}",
        a.as_ref().map(|r| r.is_ok())
    );
    let bb = catch_unwind(AssertUnwindSafe(|| init_b(&huge, 4, &b)));
    assert!(matches!(bb, Ok(Err(_))), "init_b");
    let m = catch_unwind(AssertUnwindSafe(|| {
        dropout_mask(0.5, &[usize::MAX / 2, 4], &mut CounterRng::new(1), &b)
    }));
    assert!(matches!(m, Ok(Err(_))), "dropout_mask");
}

/// Dropout through the model is finite at a very high rate and does not
/// disturb frozen leaves (still no gradient: asserted in `lora_step`).
#[test]
fn high_dropout_gives_finite_loss_and_gradients() {
    let file = lora_file();
    let (d, host, sites) = (data(), host_params(), fixture_sites(&file, true));
    let o = Opts {
        config: LoraConfig::new(RANK, ALPHA, 0.99).unwrap(),
        dropout_seed: 3,
        ..Opts::new()
    };
    let run = lora_step(cpu(), &host, &sites, &d, &o);
    assert!(run.loss.is_finite());
    for (m, a, b) in &run.grads {
        assert!(a.iter().chain(b).all(|x| x.is_finite()), "{m}");
    }
}

/// alpha = -0.0 behaves as alpha = 0: the loss is the base loss.
#[test]
fn negative_zero_alpha_is_the_base_model() {
    let file = lora_file();
    let (d, host, sites) = (data(), host_params(), fixture_sites(&file, true));
    let o = Opts {
        config: LoraConfig::new(RANK, -0.0, 0.0).unwrap(),
        ..Opts::new()
    };
    let run = lora_step(cpu(), &host, &sites, &d, &o);
    let base = read(&file, "c.base_loss")[0];
    assert!(rel(run.loss, base) <= LORA_LOSS_BOUND);
}

/// A bf16 frozen base on CPU gives the f32 run's bits (the checkpoint is
/// bf16, so both hold the same values): the dX-only path works with a bf16
/// weight, not only on Metal.
#[test]
fn a_bf16_frozen_base_on_cpu_gives_the_f32_bits() {
    let (d, sites) = (data(), fixture_sites(&lora_file(), true));
    let f32_base = host_params();
    let bf16_base: Qwen35Params<Tensor> = load_hf(
        &spec(),
        &base_file(),
        PREFIX,
        DType::Bf16,
        DType::F32,
        &budget(),
    )
    .unwrap();
    let a = lora_step(cpu(), &f32_base, &sites, &d, &Opts::new());
    let b = lora_step(cpu(), &bf16_base, &sites, &d, &Opts::new());
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    for (x, y) in a.grads.iter().zip(&b.grads) {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            (bits(&x.1), bits(&x.2)),
            (bits(&y.1), bits(&y.2)),
            "{}",
            x.0
        );
    }
}

// ---- merge / unmerge ------------------------------------------------------

fn f32_pattern() -> Vec<(&'static str, f32)> {
    vec![
        ("subnormal", f32::from_bits(1)),
        ("subnormal-neg", f32::from_bits(0x8000_0003)),
        ("min-positive", f32::MIN_POSITIVE),
        ("+0", 0.0),
        ("-0", -0.0),
        ("large", 3.0e38),
        ("-large", -3.0e38),
        ("inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
        ("nan", f32::NAN),
        ("nan-payload", f32::from_bits(0x7fc0_1234)),
    ]
}

fn fresh_pair(out: usize, inn: usize, rank: usize, a_val: f32, b_val: f32) -> (Tensor, Tensor) {
    let bud = budget();
    (
        Tensor::from_f32(&vec![a_val; rank * inn], &[rank, inn], &bud).unwrap(),
        Tensor::from_f32(&vec![b_val; out * rank], &[out, rank], &bud).unwrap(),
    )
}

fn stored_bits(t: &Tensor) -> Vec<u32> {
    match t.dtype() {
        DType::Bf16 => t
            .bf16_slice()
            .unwrap()
            .iter()
            .map(|&b| u32::from(b))
            .collect(),
        _ => t
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect(),
    }
}

/// For every hostile weight value, in f32 and bf16: merge either succeeds
/// and unmerge restores the stored bits exactly, or merge returns a typed
/// error and leaves the adapter unmerged with the weight untouched.
#[test]
fn merge_unmerge_is_bit_exact_or_a_clean_typed_error_for_hostile_weights() {
    let (out, inn) = (6, 10);
    let backend = cpu();
    let mut merged_ok = 0usize;
    for bf16 in [false, true] {
        for (name, v) in f32_pattern() {
            for position in [0usize, out * inn - 1] {
                let mut w = vec![0.25f32; out * inn];
                w[position] = v;
                let weight = if bf16 {
                    let bits: Vec<u16> = w.iter().map(|&x| f32_to_bf16(x)).collect();
                    Tensor::from_bf16_bits(&bits, &[out, inn], &budget()).unwrap()
                } else {
                    Tensor::from_f32(&w, &[out, inn], &budget()).unwrap()
                };
                let before = stored_bits(&weight);
                let (a, b) = fresh_pair(out, inn, RANK, 0.5, 0.125);
                let mut lin = LoraLinear::new(config(), weight, a, b).unwrap();
                let tag = format!("{} {name} at {position}", if bf16 { "bf16" } else { "f32" });
                match lin.merge(&backend) {
                    Ok(()) => {
                        merged_ok += 1;
                        assert!(lin.is_merged(), "{tag}");
                        assert!(
                            lin.weight()
                                .to_f32_vec()
                                .unwrap()
                                .iter()
                                .all(|x| x.is_finite()),
                            "{tag}"
                        );
                        lin.unmerge().unwrap();
                        assert_eq!(stored_bits(lin.weight()), before, "{tag}");
                    }
                    Err(LoraError::Core(OjasError::NonFinite { .. })) => {
                        assert!(!lin.is_merged(), "{tag}");
                        assert_eq!(stored_bits(lin.weight()), before, "{tag}");
                        assert!(matches!(lin.unmerge(), Err(LoraError::NotMerged)), "{tag}");
                        // A refused merge is repeatable, not sticky.
                        assert!(matches!(
                            lin.merge(&backend),
                            Err(LoraError::Core(OjasError::NonFinite { .. }))
                        ));
                    }
                    Err(e) => panic!("{tag}: untyped refusal {e:?}"),
                }
            }
        }
    }
    // Finite patterns merge: 2 dtypes x 2 positions x 7 finite values.
    assert_eq!(merged_ok, 2 * 2 * 7);
}

/// A finite weight whose merge overflows to inf is a typed error with the
/// adapter intact.
#[test]
fn merge_overflow_is_typed_and_leaves_the_weight_alone() {
    let (out, inn, rank) = (4, 6, RANK);
    let w = Tensor::from_f32(&vec![3.3e38; out * inn], &[out, inn], &budget()).unwrap();
    let before = stored_bits(&w);
    let (a, b) = fresh_pair(out, inn, rank, 1.0, 1.0e37);
    let mut lin = LoraLinear::new(config(), w, a, b).unwrap();
    let r = lin.merge(&cpu());
    assert!(
        matches!(r, Err(LoraError::Core(OjasError::NonFinite { .. }))),
        "{r:?}"
    );
    assert!(!lin.is_merged());
    assert_eq!(stored_bits(lin.weight()), before);
}

/// merged = W + (alpha/r) B A against an f64 reference, elementwise within
/// 2 ulp of the f32 result, so a wrong scaling or transposed A cannot hide.
#[test]
fn merged_weight_matches_an_f64_reference_elementwise() {
    let (out, inn, rank) = (9, 13, 4);
    let bud = budget();
    let wv: Vec<f32> = (0..out * inn)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.13)
        .collect();
    let av: Vec<f32> = (0..rank * inn)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.07)
        .collect();
    let bv: Vec<f32> = (0..out * rank)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.21)
        .collect();
    let mut lin = LoraLinear::new(
        config(),
        Tensor::from_f32(&wv, &[out, inn], &bud).unwrap(),
        Tensor::from_f32(&av, &[rank, inn], &bud).unwrap(),
        Tensor::from_f32(&bv, &[out, rank], &bud).unwrap(),
    )
    .unwrap();
    lin.merge(&cpu()).unwrap();
    let got = lin.weight().to_f32_vec().unwrap();
    let s = f64::from(ALPHA) / RANK as f64;
    for o in 0..out {
        for i in 0..inn {
            let d: f64 = (0..rank)
                .map(|k| f64::from(bv[o * rank + k]) * f64::from(av[k * inn + i]))
                .sum();
            let want = f64::from(wv[o * inn + i]) + s * d;
            let g = f64::from(got[o * inn + i]);
            let tol = 4.0 * f64::from(f32::EPSILON) * want.abs().max(s * d.abs()).max(1e-3);
            assert!((g - want).abs() <= tol, "[{o},{i}]: {g} vs {want}");
        }
    }
}

// ---- target discovery ------------------------------------------------------

/// An in-memory checkpoint with every real target's name (with `prefix`
/// substituted for `model.`) and shape, dtype `dtype`, plus `rename`.
fn synthetic_index(
    prefix: &str,
    dtype: StDtype,
    rename: impl Fn(&str) -> String,
    shape_of: impl Fn(&str, &[u64]) -> Vec<u64>,
) -> Vec<u8> {
    let real = base_file();
    let width = match dtype {
        StDtype::F32 => 4,
        StDtype::BF16 => 2,
        _ => 8,
    };
    let mut owned: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    for t in targets() {
        let info = real.info(&t.name).unwrap();
        let shape = shape_of(&t.name, &info.shape);
        let n: u64 = shape.iter().product();
        let name = rename(&t.name.replacen("model.", prefix, 1));
        owned.push((name, shape, vec![0u8; (n * width) as usize]));
    }
    let items: Vec<TensorOut<'_>> = owned
        .iter()
        .map(|(name, shape, data)| TensorOut {
            name,
            dtype,
            shape,
            data,
        })
        .collect();
    encode_safetensors(&items, &[]).unwrap()
}

#[test]
fn a_unicode_prefix_resolves_and_a_lookalike_name_is_refused_by_name() {
    let bytes = synthetic_index("模型.", StDtype::F32, |n| n.to_string(), |_, s| s.to_vec());
    let file = SafeTensors::parse(&bytes).unwrap();
    let found = lora_targets(&spec(), &file, "模型.", &LORA_KINDS).unwrap();
    assert_eq!(found.len(), 13);
    assert!(found.iter().all(|t| t.name.starts_with("模型.layers.")));
    // The ASCII prefix finds none of them.
    let err = lora_targets(&spec(), &file, "model.", &LORA_KINDS)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not in the checkpoint"), "{err}");

    // One name with a Cyrillic 'а' (U+0430) in `self_attn`: the lookalike
    // is a different tensor, so the ASCII target is missing.
    let bytes = synthetic_index(
        "model.",
        StDtype::F32,
        |n| n.replace("self_attn.q_proj", "self_\u{0430}ttn.q_proj"),
        |_, s| s.to_vec(),
    );
    let file = SafeTensors::parse(&bytes).unwrap();
    let err = lora_targets(&spec(), &file, "model.", &LORA_KINDS)
        .unwrap_err()
        .to_string();
    assert!(err.contains("self_attn.q_proj"), "{err}");

    // The same name in NFD form (decomposed) is a different key too.
    let bytes = synthetic_index("é.", StDtype::F32, |n| n.to_string(), |_, s| s.to_vec());
    let file = SafeTensors::parse(&bytes).unwrap();
    assert!(lora_targets(&spec(), &file, "e\u{0301}.", &LORA_KINDS).is_err());
    assert!(lora_targets(&spec(), &file, "é.", &LORA_KINDS).is_ok());
}

#[test]
fn a_duplicate_tensor_name_never_reaches_target_discovery() {
    let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0u8; 4]);
    assert!(SafeTensors::parse(&bytes).is_err());
}

#[test]
fn a_transposed_or_wrong_rank_target_is_refused_not_panicked() {
    let transposed = synthetic_index(
        "model.",
        StDtype::F32,
        |n| n.to_string(),
        |name, s| {
            if name.contains("mlp.gate_proj") {
                vec![s[1], s[0]]
            } else {
                s.to_vec()
            }
        },
    );
    let file = SafeTensors::parse(&transposed).unwrap();
    let err = lora_targets(&spec(), &file, "model.", &LORA_KINDS)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("mlp.gate_proj") && err.contains("shape"),
        "{err}"
    );

    let flat = synthetic_index(
        "model.",
        StDtype::F32,
        |n| n.to_string(),
        |name, s| {
            if name.contains("self_attn.o_proj") {
                vec![s.iter().product()]
            } else {
                s.to_vec()
            }
        },
    );
    let file = SafeTensors::parse(&flat).unwrap();
    let err = lora_targets(&spec(), &file, "model.", &LORA_KINDS)
        .unwrap_err()
        .to_string();
    assert!(err.contains("self_attn.o_proj"), "{err}");
}

/// Repeated kinds do not repeat a target.
#[test]
fn repeated_kinds_give_each_target_once() {
    let t = lora_targets(
        &spec(),
        &base_file(),
        PREFIX,
        &[LoraKind::MlpGate, LoraKind::MlpGate, LoraKind::MlpUp],
    )
    .unwrap();
    assert_eq!(t.len(), 2 * 2);
    let mut names: Vec<&str> = t.iter().map(|x| x.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), t.len());
}

/// A target stored as an integer dtype is not a weight the loader can use;
/// discovery reads only the index, so it must still refuse a dtype the
/// load would refuse.
#[test]
fn a_target_stored_as_an_integer_dtype_is_refused() {
    let bytes = synthetic_index("model.", StDtype::I64, |n| n.to_string(), |_, s| s.to_vec());
    let file = SafeTensors::parse(&bytes).unwrap();
    assert!(lora_targets(&spec(), &file, "model.", &LORA_KINDS).is_err());
}

// ---- can the bounds tell a wrong ojas from a right one? --------------------

struct Traj {
    losses: Vec<f32>,
    finals: Vec<(String, Vec<f32>, Vec<f32>)>,
}

/// `trajectory()` of `qwen35_lora.rs` with the optimizer and scaling knobs
/// exposed, so a deliberately wrong run can be scored against the same
/// bounds.
fn trajectory(alpha: f32, cfg: AdamWConfig) -> Traj {
    use ojas_model::lora::LoraSite;
    use ojas_model::qwen35::MixerLora;
    let file = lora_file();
    let (s, tg, d, host) = (spec(), targets(), data(), host_params());
    let backend = cpu();
    let lcfg = LoraConfig::new(RANK, alpha, 0.0).unwrap();
    let mut sites = fixture_sites(&file, false);
    let zero = |t: &Tensor| {
        Tensor::from_f32(&vec![0.0; t.num_elements().unwrap()], t.shape(), &budget()).unwrap()
    };
    let mut moments: Vec<(Tensor, Tensor)> = sites
        .sites()
        .iter()
        .flat_map(|site| std::iter::once(&site.a).chain(&site.b))
        .map(|t| (zero(t), zero(t)))
        .collect();
    let mut losses = Vec::new();
    for step in 0..STEPS {
        let grads: Vec<Tensor> = {
            let mut tape = Tape::new(backend.clone());
            let params = bind_frozen(&mut tape, &host).unwrap();
            let bound = sites
                .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
                .unwrap();
            let mut rng = CounterRng::new(0);
            let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
            let fwd = LoraForward::new(&mut tape, bound, &lcfg, b, t, &mut rng, &budget()).unwrap();
            let loss = forward_loss_lora(
                &mut tape,
                &s,
                &params,
                Some(&fwd),
                &d.ids,
                &d.targets,
                &d.tables,
                Some(IGNORE),
                DEFAULT_CE_CHUNK,
                ActivationCheckpoint::Off,
            )
            .unwrap();
            tape.backward(loss).unwrap();
            losses.push(to_vec(tape.value(loss).unwrap())[0]);
            fwd.sites()
                .sites()
                .iter()
                .flat_map(|site| std::iter::once(&site.a).chain(&site.b))
                .map(|v| tape.take_grad(*v).unwrap())
                .collect()
        };
        let mut grads = grads.into_iter();
        let mut moment = moments.iter_mut();
        let mut update = |p: &mut Tensor| {
            let g = grads.next().unwrap();
            let (m1, m2) = moment.next().unwrap();
            backend.adamw_step(p, &g, m1, m2, step as u64, cfg).unwrap();
        };
        for l in &mut sites.layers {
            let mixer: Vec<&mut Option<LoraSite<Tensor>>> = match &mut l.mixer {
                MixerLora::GatedDeltaNet { qkv, z, out } => vec![qkv, z, out],
                MixerLora::Attention { q, k, v, o } => vec![q, k, v, o],
            };
            for site in mixer
                .into_iter()
                .chain([&mut l.gate, &mut l.up, &mut l.down])
                .flatten()
            {
                update(&mut site.a);
                for b in &mut site.b {
                    update(b);
                }
            }
        }
    }
    let finals = tg
        .iter()
        .zip(sites.sites())
        .map(|(t, site)| {
            let parts: Vec<Vec<f32>> = site.b.iter().map(to_vec).collect();
            (
                t.module().to_string(),
                to_vec(&site.a),
                t.fuse_b(&s, &parts, RANK),
            )
        })
        .collect();
    Traj { losses, finals }
}

fn good_cfg() -> AdamWConfig {
    AdamWConfig {
        lr: LR,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: WD,
    }
}

/// Does the (b') loss clause pass: steps 0-5 within 1e-5, to 20 within 1e-4.
fn loss_clause_passes(t: &Traj, file: &SafeTensors<'_>) -> bool {
    let want = read(file, "b.losses");
    t.losses
        .iter()
        .zip(&want)
        .enumerate()
        .all(|(i, (&g, &w))| rel(g, w) <= if i <= 5 { 1e-5 } else { 1e-4 })
}

/// The noise-floor test's final-adapter clause exactly as written in
/// `qwen35_lora.rs`: per tensor `max(1e-5, 3 * floor)` against torch fp32
/// and against fp64, where the floor is torch fp32's own distance to fp64.
/// Returns the worst `ratio / bound` over tensors (above 1 = rejected) and
/// the count of tensors rejected.
fn final_clause(t: &Traj, file: &SafeTensors<'_>) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut rejected = 0;
    for (module, a, b) in &t.finals {
        for (w, v) in [("A", a), ("B", b)] {
            let name = format!("{module}.lora_{w}.weight");
            let torch = read(file, &format!("b.final.{name}"));
            let exact = read(file, &format!("b.final64.{name}"));
            let floor = peak_ratio(&torch, &exact, &name);
            let bound = f64::from(1e-5f32.max(3.0 * floor));
            let vs_torch = f64::from(peak_ratio(v, &torch, &name));
            let vs_exact = f64::from(peak_ratio(v, &exact, &name));
            let ratio = (vs_torch / bound).max(vs_exact / bound);
            worst = worst.max(ratio);
            rejected += usize::from(ratio > 1.0);
        }
    }
    (worst, rejected)
}

/// The fixture is not vacuous: the reference moves, every reference
/// gradient has a peak, and the oracle's own noise floor is a small
/// fraction of 1 on most tensors.
#[test]
fn the_fixture_exercises_what_the_bounds_claim() {
    let file = lora_file();
    let losses = read(&file, "b.losses");
    assert_eq!(losses.len(), STEPS);
    assert!(
        rel(losses[STEPS - 1], losses[0]) > 1e-3,
        "the reference trajectory barely moves: {} -> {}",
        losses[0],
        losses[STEPS - 1]
    );
    assert!(
        rel(read(&file, "a.loss")[0], read(&file, "c.loss")[0]) > 1e-4,
        "B != 0 changes the loss"
    );
    for t in targets() {
        for w in ["A", "B"] {
            let g = read(&file, &format!("a.grad.{}.lora_{w}.weight", t.module()));
            let peak = g.iter().fold(0.0f32, |p, x| p.max(x.abs()));
            assert!(
                peak > 1e-9 && g.iter().all(|x| x.is_finite()),
                "{} {w}: peak {peak}",
                t.module()
            );
            let fin = read(&file, &format!("b.final.{}.lora_{w}.weight", t.module()));
            let init = if w == "A" {
                read(&file, &format!("init.{}.lora_A.weight", t.module()))
            } else {
                vec![0.0; fin.len()]
            };
            assert!(
                fin != init,
                "{} {w} did not train in the reference",
                t.module()
            );
        }
    }
}

/// (b') sensitivity: a deliberately wrong optimizer or scaling must be
/// rejected by the loss clause or the noise-floor final-adapter clause.
/// Prints each mutant's verdicts. A mutant neither clause rejects is a
/// bound that cannot tell a wrong ojas from a right one.
#[test]
fn the_b_prime_bounds_reject_a_wrong_optimizer_or_scaling() {
    let file = lora_file();
    let good = trajectory(ALPHA, good_cfg());
    let (gw, gr) = final_clause(&good, &file);
    eprintln!(
        "baseline: loss clause {}, final clause worst {gw:.2} rejected {gr}",
        loss_clause_passes(&good, &file)
    );
    assert!(loss_clause_passes(&good, &file) && gr == 0);
    let mutants: Vec<(&str, f32, AdamWConfig)> = vec![
        (
            "lr x1.01",
            ALPHA,
            AdamWConfig {
                lr: LR * 1.01,
                ..good_cfg()
            },
        ),
        (
            "lr x1.001",
            ALPHA,
            AdamWConfig {
                lr: LR * 1.001,
                ..good_cfg()
            },
        ),
        (
            "weight decay 0",
            ALPHA,
            AdamWConfig {
                weight_decay: 0.0,
                ..good_cfg()
            },
        ),
        (
            "weight decay x10",
            ALPHA,
            AdamWConfig {
                weight_decay: WD * 10.0,
                ..good_cfg()
            },
        ),
        (
            "beta2 0.99",
            ALPHA,
            AdamWConfig {
                beta2: 0.99,
                ..good_cfg()
            },
        ),
        (
            "beta1 0.89",
            ALPHA,
            AdamWConfig {
                beta1: 0.89,
                ..good_cfg()
            },
        ),
        (
            "eps 1e-6",
            ALPHA,
            AdamWConfig {
                eps: 1e-6,
                ..good_cfg()
            },
        ),
        ("alpha 8.08 (+1%)", ALPHA * 1.01, good_cfg()),
        ("alpha 8.008 (+0.1%)", ALPHA * 1.001, good_cfg()),
    ];
    let mut missed = Vec::new();
    for (name, alpha, cfg) in mutants {
        let m = trajectory(alpha, cfg);
        let loss_ok = loss_clause_passes(&m, &file);
        let (worst, rejected) = final_clause(&m, &file);
        eprintln!("{name}: loss clause passes={loss_ok}, final clause worst {worst:.2} of bound, rejected {rejected}/26");
        if loss_ok && rejected == 0 {
            missed.push(name);
        }
    }
    assert!(
        missed.is_empty(),
        "bounds accept these wrong runs: {missed:?}"
    );
}

/// (a') sensitivity: a wrong scaling or a missing adapter must be rejected
/// by the loss or gradient clause.
#[test]
fn the_a_prime_bounds_reject_a_wrong_scaling_or_a_missing_site() {
    let file = lora_file();
    let (d, host, sites) = (data(), host_params(), fixture_sites(&file, true));
    let verdict = |run: &LoraRun| -> (f64, f32) {
        let l = rel(run.loss, read(&file, "a.loss")[0]);
        let mut worst = 0.0f32;
        for (module, a, b) in &run.grads {
            let ra = peak_ratio(
                a,
                &read(&file, &format!("a.grad.{module}.lora_A.weight")),
                module,
            );
            let rb = peak_ratio(
                b,
                &read(&file, &format!("a.grad.{module}.lora_B.weight")),
                module,
            );
            worst = worst.max(ra).max(rb);
        }
        (l, worst)
    };
    let base = lora_step(cpu(), &host, &sites, &d, &Opts::new());
    let (l0, g0) = verdict(&base);
    assert!(
        l0 <= LORA_LOSS_BOUND && g0 <= LORA_GRAD_BOUND,
        "baseline {l0:e} {g0:e}"
    );
    let mut missed = Vec::new();
    for (name, alpha) in [
        ("alpha +1%", ALPHA * 1.01),
        ("alpha +0.1%", ALPHA * 1.001),
        ("alpha +0.01%", ALPHA * 1.0001),
    ] {
        let o = Opts {
            config: LoraConfig::new(RANK, alpha, 0.0).unwrap(),
            ..Opts::new()
        };
        let (l, g) = verdict(&lora_step(cpu(), &host, &sites, &d, &o));
        eprintln!("{name}: loss rel {l:.2e} (bound {LORA_LOSS_BOUND:.0e}), worst grad {g:.2e} (bound {LORA_GRAD_BOUND:.0e})");
        if l <= LORA_LOSS_BOUND && g <= LORA_GRAD_BOUND {
            missed.push(name);
        }
    }
    // Zeroing one adapter's B (a missing site) must show in the loss.
    let mut nob = fixture_sites(&file, true);
    for l in &mut nob.layers {
        if let Some(site) = l.down.as_mut() {
            for b in &mut site.b {
                *b = Tensor::from_f32(&vec![0.0; b.num_elements().unwrap()], b.shape(), &budget())
                    .unwrap();
            }
        }
    }
    let (l, _) = verdict(&lora_step(cpu(), &host, &nob, &d, &Opts::new()));
    eprintln!("mlp.down adapters off: loss rel {l:.2e}");
    if l <= LORA_LOSS_BOUND {
        missed.push("mlp.down adapters off");
    }
    assert!(missed.is_empty(), "(a') accepts: {missed:?}");
}

// ---- Metal merge -----------------------------------------------------------

#[cfg(target_os = "macos")]
#[test]
fn merge_on_metal_matches_the_cpu_and_unmerges_bit_exactly() {
    let Ok(m) = ojas_metal::MetalBackend::new(ojas_core::Budget::new(2 << 30)) else {
        if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() {
            return;
        }
        panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip)");
    };
    let (out, inn, rank) = (24, 40, 4);
    let bud = budget();
    let wv: Vec<f32> = (0..out * inn)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.13)
        .collect();
    let bits: Vec<u16> = wv.iter().map(|&x| f32_to_bf16(x)).collect();
    let av: Vec<f32> = (0..rank * inn)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.07)
        .collect();
    let bv: Vec<f32> = (0..out * rank)
        .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.21)
        .collect();
    for bf16 in [false, true] {
        let weight = if bf16 {
            Tensor::from_bf16_bits(&bits, &[out, inn], &bud).unwrap()
        } else {
            Tensor::from_f32(&wv, &[out, inn], &bud).unwrap()
        };
        let mk = || {
            LoraLinear::new(
                config(),
                weight.clone(),
                Tensor::from_f32(&av, &[rank, inn], &bud).unwrap(),
                Tensor::from_f32(&bv, &[out, rank], &bud).unwrap(),
            )
            .unwrap()
        };
        let (mut on_cpu, mut on_gpu) = (mk(), mk());
        let before = stored_bits(&weight);
        on_cpu.merge(&cpu()).unwrap();
        on_gpu.merge(&m).unwrap();
        let (c, g) = (
            on_cpu.weight().to_f32_vec().unwrap(),
            on_gpu.weight().to_f32_vec().unwrap(),
        );
        for (i, (x, y)) in c.iter().zip(&g).enumerate() {
            assert!(
                (x - y).abs() <= 1e-5 * (1.0 + x.abs()),
                "bf16={bf16} [{i}]: cpu {x} metal {y}"
            );
        }
        on_gpu.unmerge().unwrap();
        assert_eq!(stored_bits(on_gpu.weight()), before, "bf16={bf16}");
    }
}
