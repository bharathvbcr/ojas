//! The merged dense export of a Qwen3.5 LoRA: `W + (alpha/r) B A` per adapted
//! linear, rounded once to bf16, under the base's Hugging Face names
//! (`LoraTarget::merged_bf16`, `Qwen35Lora::merge_bf16`,
//! `lora::merged_weight_bf16`).
//!
//! Fixtures: `tests/fixtures/qwen35_tiny` (the base, f32) and
//! `tests/fixtures/qwen35_tiny_lora` (rank 4, `A` under `init.*`, a non-zero
//! `B` under `bnz.*`). The helpers below are copies of the minimal part of
//! `qwen35_common`, which this test does not touch.
//!
//! The independent reference (`host_merge`) forms `B A` as the CPU backend's
//! `Numerics::Exact` reduction does (one accumulator from `+0.0`, `k`
//! ascending, `acc + b * a` with two roundings; `ojas-cpu/src/gemm.rs`,
//! `gemv_into` and the packed kernel), adds `s * d` to the widened base in
//! f32 and rounds once with `f32_to_bf16`. It reads the fused `B` straight
//! from the fixture, so it shares nothing with the code under test but the
//! rounding function and the reduction order it must reproduce.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ojas_core::{bf16_to_f32, f32_to_bf16, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_data::CounterRng;
use ojas_io::{encode_f32_as, encode_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::lora::{merged_weight_bf16, LoraConfig, LoraError, LoraLinear, LoraSite};
use ojas_model::qwen35::{
    bind, forward_hidden_lora, hf_tensors, load_hf, lora_targets, LoraForward, LoraKind,
    LoraTarget, Qwen35Lora, Qwen35Mixer, Qwen35Params, Qwen35Spec, Qwen35Tables, LORA_KINDS,
};
use ojas_model::{ActivationCheckpoint, Eval};

const PREFIX: &str = "model.";
const RANK: usize = 4;
const ALPHA: f32 = 8.0;

fn spec() -> Qwen35Spec {
    Qwen35Spec {
        vocab: 8320,
        hidden: 64,
        intermediate: 128,
        layers: vec![Qwen35Mixer::GatedDeltaNet, Qwen35Mixer::Attention],
        q_heads: 2,
        kv_heads: 1,
        head_dim: 256,
        rotary_dim: 64,
        rope_theta: 1e4,
        gdn_heads: 2,
        gdn_key_dim: 128,
        gdn_value_dim: 128,
        conv_width: 4,
        eps: 1e-6,
    }
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_tiny")
}

fn budget() -> Budget {
    Budget::new(1 << 30)
}

fn cpu_on(budget: &Budget) -> CpuBackend {
    CpuBackend::new(budget.clone()).with_numerics(Numerics::Exact)
}

fn cpu() -> CpuBackend {
    cpu_on(&Budget::new(4 << 30))
}

fn base_file() -> SafeTensors<'static> {
    SafeTensors::open(&dir().join("model.safetensors")).unwrap()
}

fn lora_file() -> SafeTensors<'static> {
    let path = dir()
        .parent()
        .unwrap()
        .join("qwen35_tiny_lora")
        .join("lora.safetensors");
    SafeTensors::open(&path).unwrap()
}

fn config(alpha: f32) -> LoraConfig {
    LoraConfig::new(RANK, alpha, 0.0).unwrap()
}

fn targets() -> Vec<LoraTarget> {
    lora_targets(&spec(), &base_file(), PREFIX, &LORA_KINDS).unwrap()
}

fn out_rows(t: &LoraTarget) -> usize {
    t.out_parts.iter().sum()
}

/// `(A [r, in], fused B [out, r])` of `t` as the fixture stores them.
fn adapter(lora: &SafeTensors<'_>, t: &LoraTarget) -> (Vec<f32>, Vec<f32>) {
    let read = |name: String| lora.read_f32(&name).unwrap().1;
    (
        read(format!("init.{}.lora_A.weight", t.module())),
        read(format!("bnz.{}.lora_B.weight", t.module())),
    )
}

/// The fixture's adapters, `B` split into the model's parts.
fn sites(budget: &Budget) -> Qwen35Lora<LoraSite<Tensor>> {
    let (s, lora) = (spec(), lora_file());
    let mut set = Qwen35Lora::empty(&s);
    for t in &targets() {
        let (a, fused) = adapter(&lora, t);
        let a = Tensor::from_f32(&a, &[RANK, t.in_features], budget).unwrap();
        let b = t
            .split_b(&s, &fused, RANK)
            .into_iter()
            .zip(&t.out_parts)
            .map(|(v, &rows)| Tensor::from_f32(&v, &[rows, RANK], budget).unwrap())
            .collect();
        *set.slot_mut(t.layer, t.kind).unwrap() = Some(LoraSite { a, b });
    }
    set
}

/// Every adapter tensor's bits, in `sites()` order: `A`, then each `B` part.
fn adapter_bits(set: &Qwen35Lora<LoraSite<Tensor>>) -> Vec<Vec<u32>> {
    set.sites()
        .iter()
        .flat_map(|site| std::iter::once(&site.a).chain(&site.b))
        .map(|t| {
            t.to_f32_vec()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        })
        .collect()
}

/// The independent reference: see the module docs.
fn host_merge(w: &[f32], a: &[f32], b: &[f32], rank: usize, inn: usize, alpha: f32) -> Vec<u16> {
    let s = alpha / rank as f32;
    let mut bits = Vec::with_capacity(w.len());
    for i in 0..w.len() / inn {
        for j in 0..inn {
            let mut acc = 0.0f32;
            for k in 0..rank {
                acc += b[i * rank + k] * a[k * inn + j];
            }
            bits.push(f32_to_bf16(w[i * inn + j] + s * acc));
        }
    }
    bits
}

fn oracle(base: &SafeTensors<'_>, t: &LoraTarget, alpha: f32) -> Vec<u16> {
    let (a, b) = adapter(&lora_file(), t);
    let w = base.read_f32_widened(&t.name).unwrap().1;
    host_merge(&w, &a, &b, RANK, t.in_features, alpha)
}

/// `base` as a safetensors file with `edits` (name, dtype, little-endian
/// bytes) replacing the stored tensors of those names.
fn checkpoint(base: &SafeTensors<'_>, edits: &BTreeMap<String, (StDtype, Vec<u8>)>) -> Vec<u8> {
    let mut owned: Vec<(String, StDtype, Vec<u64>, Vec<u8>)> = Vec::new();
    for name in base.names() {
        let info = base.info(name).unwrap();
        let (dtype, data) = match edits.get(name) {
            Some((d, bytes)) => (*d, bytes.clone()),
            None => (info.dtype, base.read_bytes(name).unwrap()),
        };
        owned.push((name.to_string(), dtype, info.shape.clone(), data));
    }
    let items: Vec<TensorOut<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| TensorOut {
            name,
            dtype: *dtype,
            shape,
            data,
        })
        .collect();
    encode_safetensors(&items, &[]).unwrap()
}

/// The fixture base with every adapted linear stored as `dtype` (the fixture
/// itself is stored as bf16, so an F32 copy widens exactly and a BF16 copy
/// is the fixture's own bits).
fn base_bytes_as(dtype: StDtype) -> Vec<u8> {
    let base = base_file();
    let edits = targets()
        .iter()
        .map(|t| {
            let w = base.read_f32_widened(&t.name).unwrap().1;
            let bytes = encode_f32_as(dtype, &w).unwrap();
            (t.name.clone(), (dtype, bytes))
        })
        .collect();
    checkpoint(&base, &edits)
}

fn bf16_base_bytes() -> Vec<u8> {
    base_bytes_as(StDtype::BF16)
}

fn f32_base_bytes() -> Vec<u8> {
    base_bytes_as(StDtype::F32)
}

fn bits_of(t: &Tensor) -> Vec<u16> {
    t.bf16_slice().unwrap().to_vec()
}

fn export(
    set: &Qwen35Lora<LoraSite<Tensor>>,
    cfg: &LoraConfig,
    file: &SafeTensors<'_>,
    budget: &Budget,
) -> Result<Vec<(String, Tensor)>, OjasError> {
    let mut out = Vec::new();
    set.merge_bf16(
        &spec(),
        &targets(),
        cfg,
        file,
        &cpu_on(budget),
        &mut |name, t| {
            out.push((name.to_string(), t));
            Ok(())
        },
    )?;
    Ok(out)
}

// ---- names --------------------------------------------------------------

#[test]
fn the_export_is_the_adapted_linears_under_the_bases_hugging_face_names() {
    let (base, bud) = (base_file(), budget());
    let set = sites(&bud);
    let merged = export(&set, &config(ALPHA), &base, &bud).unwrap();
    let t = targets();
    // 2 layers: GDN qkv, z, out (3) + attention q, k, v, o (4) + 3 MLP each.
    assert_eq!(t.len(), 13);
    assert_eq!(
        merged.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        t.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(),
        "names and order are the targets'"
    );
    let table = hf_tensors(&spec(), PREFIX);
    for ((name, tensor), target) in merged.iter().zip(&t) {
        let info = base.info(name).expect("the base has this tensor");
        let want: Vec<usize> = info.shape.iter().map(|&n| n as usize).collect();
        assert_eq!(tensor.shape(), &want[..], "{name}");
        assert_eq!(tensor.shape(), &[out_rows(target), target.in_features]);
        assert_eq!(tensor.dtype(), DType::Bf16, "{name}");
        assert!(table.iter().any(|h| &h.name == name), "{name}");
    }
    // Names are unique.
    let mut names: Vec<&str> = merged.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), merged.len());
}

// ---- bits ---------------------------------------------------------------

/// f32 base and bf16 base, at scalings 2, 1, 1.5 and 0: every merged tensor is
/// the host recomputation's bits, and the adapters are not a no-op.
#[test]
fn merged_tensors_are_bit_identical_to_the_host_recomputation() {
    let f32_bytes = f32_base_bytes();
    let f32_base = SafeTensors::parse(&f32_bytes).unwrap();
    let bf16_bytes = bf16_base_bytes();
    let bf16_base = SafeTensors::parse(&bf16_bytes).unwrap();
    for (label, base) in [("f32", &f32_base), ("bf16", &bf16_base)] {
        let mut by_alpha: Vec<(f32, Vec<Vec<u16>>)> = Vec::new();
        for alpha in [8.0f32, 4.0, 6.0, 0.0] {
            let bud = budget();
            let merged = export(&sites(&bud), &config(alpha), base, &bud).unwrap();
            let mut all = Vec::new();
            for ((name, tensor), t) in merged.iter().zip(&targets()) {
                let got = bits_of(tensor);
                assert_eq!(got, oracle(base, t, alpha), "{label} alpha {alpha}: {name}");
                all.push(got);
            }
            by_alpha.push((alpha, all));
        }
        let of = |alpha: f32| &by_alpha.iter().find(|(a, _)| *a == alpha).unwrap().1;
        assert_ne!(of(8.0), of(4.0), "{label}: the scaling reaches the bits");
        assert_ne!(of(4.0), of(6.0), "{label}: the scaling reaches the bits");
        // alpha 0 adds nothing: a bf16 base comes back as stored, an f32 one
        // rounded once.
        for (got, t) in of(0.0).iter().zip(targets()) {
            let w = base.read_f32_widened(&t.name).unwrap().1;
            let rounded: Vec<u16> = w.iter().map(|&x| f32_to_bf16(x)).collect();
            assert_eq!(
                got, &rounded,
                "{label}: alpha 0 is the base, rounded once: {}",
                t.name
            );
            if label == "bf16" {
                assert_eq!(
                    got,
                    &base.read_bf16_bits(&t.name).unwrap().1,
                    "{}: alpha 0 on a bf16 base is the stored bits",
                    t.name
                );
            }
        }
    }
}

#[test]
fn a_non_dyadic_scaling_is_alpha_over_rank_exactly() {
    const OUT: usize = 7;
    const INN: usize = 9;
    const R: usize = 3;
    let fill = |n: usize, k: f32| -> Vec<f32> {
        (0..n).map(|i| ((i * 7 % 17) as f32 - 8.0) * k).collect()
    };
    let (w, a, b) = (
        fill(OUT * INN, 0.113),
        fill(R * INN, 0.071),
        fill(OUT * R, 0.37),
    );
    let bud = budget();
    let t = |v: &[f32], shape: &[usize]| Tensor::from_f32(v, shape, &bud).unwrap();
    for alpha in [5.0f32, 1.0, 0.3, 7.0] {
        let cfg = LoraConfig::new(R, alpha, 0.0).unwrap();
        let got = merged_weight_bf16(
            &cfg,
            &t(&w, &[OUT, INN]),
            &t(&a, &[R, INN]),
            &t(&b, &[OUT, R]),
            &cpu_on(&bud),
        )
        .unwrap();
        assert_eq!(
            bits_of(&got),
            host_merge(&w, &a, &b, R, INN, alpha),
            "alpha {alpha}"
        );
    }
}

// ---- adapters are untouched ----------------------------------------------

#[test]
fn merging_does_not_change_the_adapters_and_repeats_bit_for_bit() {
    let (base, bud) = (base_file(), budget());
    let set = sites(&bud);
    let before = adapter_bits(&set);
    let live = bud.live_bytes().unwrap();
    let first = export(&set, &config(ALPHA), &base, &bud).unwrap();
    let second = export(&set, &config(ALPHA), &base, &bud).unwrap();
    assert_eq!(adapter_bits(&set), before);
    for ((_, x), (_, y)) in first.iter().zip(&second) {
        assert_eq!(bits_of(x), bits_of(y));
    }
    drop((first, second));
    assert_eq!(bud.live_bytes().unwrap(), live, "nothing is left charged");
}

#[test]
fn a_lora_linear_exports_without_merging_and_from_its_stored_weight() {
    const OUT: usize = 6;
    const INN: usize = 5;
    const R: usize = 2;
    let fill =
        |n: usize, k: f32| -> Vec<f32> { (0..n).map(|i| ((i % 13) as f32 - 6.0) * k).collect() };
    let bud = budget();
    let bits: Vec<u16> = fill(OUT * INN, 0.173)
        .iter()
        .map(|&v| f32_to_bf16(v))
        .collect();
    let (a, b) = (fill(R * INN, 0.37), fill(OUT * R, 0.29));
    let cfg = LoraConfig::new(R, 6.0, 0.0).unwrap();
    let mut layer = LoraLinear::new(
        cfg,
        Tensor::from_bf16_bits(&bits, &[OUT, INN], &bud).unwrap(),
        Tensor::from_f32(&a, &[R, INN], &bud).unwrap(),
        Tensor::from_f32(&b, &[OUT, R], &bud).unwrap(),
    )
    .unwrap();
    let backend = cpu_on(&bud);
    let widened: Vec<f32> = bits.iter().map(|&x| bf16_to_f32(x)).collect();
    let want = host_merge(&widened, &a, &b, R, INN, 6.0);
    let live = bud.live_bytes().unwrap();
    let got = layer.merged_weight_bf16(&backend).unwrap();
    assert_eq!(bits_of(&got), want);
    assert!(!layer.is_merged());
    assert_eq!(layer.weight().bf16_slice().unwrap(), &bits[..]);
    drop(got);
    assert_eq!(bud.live_bytes().unwrap(), live);
    // After merge() the stored original is still the base.
    layer.merge(&backend).unwrap();
    assert_eq!(bits_of(&layer.merged_weight_bf16(&backend).unwrap()), want);
}

// ---- refusals ----------------------------------------------------------------

fn is_non_finite(e: &LoraError) -> bool {
    matches!(e, LoraError::Core(OjasError::NonFinite { .. }))
}

#[test]
fn bad_shapes_dtypes_ranks_and_non_finite_inputs_are_refused_and_charge_nothing() {
    const OUT: usize = 6;
    const INN: usize = 5;
    const R: usize = 2;
    let fill =
        |n: usize, k: f32| -> Vec<f32> { (0..n).map(|i| ((i % 13) as f32 - 6.0) * k).collect() };
    let bud = budget();
    let be = cpu_on(&bud);
    let cfg = LoraConfig::new(R, 4.0, 0.0).unwrap();
    let f = |v: &[f32], shape: &[usize]| Tensor::from_f32(v, shape, &bud).unwrap();
    let (w, a, b) = (
        fill(OUT * INN, 0.05),
        fill(R * INN, 0.1),
        fill(OUT * R, 0.1),
    );
    let run = |w: &Tensor, a: &Tensor, b: &Tensor| merged_weight_bf16(&cfg, w, a, b, &be);

    // The good call works, and rank 0 is refused where the config is made.
    assert!(run(&f(&w, &[OUT, INN]), &f(&a, &[R, INN]), &f(&b, &[OUT, R])).is_ok());
    assert!(matches!(
        LoraConfig::new(0, 8.0, 0.0),
        Err(LoraError::ZeroRank)
    ));

    // Shapes.
    let wt = f(&w, &[OUT, INN]);
    let bad_a = f(&fill(R * (INN + 1), 0.1), &[R, INN + 1]);
    let bad_b = f(&fill((OUT + 1) * R, 0.1), &[OUT + 1, R]);
    let rank3 = f(&fill(R * INN, 0.1), &[R, 1, INN]);
    assert!(matches!(
        run(&wt, &bad_a, &f(&b, &[OUT, R])),
        Err(LoraError::Mismatch(_))
    ));
    assert!(matches!(
        run(&wt, &f(&a, &[R, INN]), &bad_b),
        Err(LoraError::Mismatch(_))
    ));
    assert!(matches!(
        run(&rank3, &f(&a, &[R, INN]), &f(&b, &[OUT, R])),
        Err(LoraError::Mismatch(_))
    ));
    // A rank above min(in, out).
    let big = LoraConfig::new(INN + 1, 4.0, 0.0).unwrap();
    assert!(matches!(
        merged_weight_bf16(&big, &wt, &f(&a, &[R, INN]), &f(&b, &[OUT, R]), &be),
        Err(LoraError::RankTooLarge { .. })
    ));

    // Dtypes: a non-float weight, a bf16 adapter.
    let ints = Tensor::from_u32(&[1u32; OUT * INN], &[OUT, INN], &bud).unwrap();
    assert!(matches!(
        run(&ints, &f(&a, &[R, INN]), &f(&b, &[OUT, R])),
        Err(LoraError::Mismatch(_))
    ));
    let bf_a = Tensor::from_bf16_bits(&vec![0u16; R * INN], &[R, INN], &bud).unwrap();
    assert!(matches!(
        run(&wt, &bf_a, &f(&b, &[OUT, R])),
        Err(LoraError::Mismatch(_))
    ));

    // Non-finite weight (f32 and bf16), A and B.
    for poke in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut wv = w.clone();
        wv[7] = poke;
        let bits: Vec<u16> = wv.iter().map(|&v| f32_to_bf16(v)).collect();
        let bf = Tensor::from_bf16_bits(&bits, &[OUT, INN], &bud).unwrap();
        for weight in [f(&wv, &[OUT, INN]), bf] {
            let e = run(&weight, &f(&a, &[R, INN]), &f(&b, &[OUT, R])).unwrap_err();
            assert!(is_non_finite(&e), "{poke} weight: {e}");
        }
        let mut av = a.clone();
        av[3] = poke;
        let e = run(&wt, &f(&av, &[R, INN]), &f(&b, &[OUT, R])).unwrap_err();
        assert!(is_non_finite(&e), "{poke} in A: {e}");
        let mut bv = b.clone();
        bv[3] = poke;
        let e = run(&wt, &f(&a, &[R, INN]), &f(&bv, &[OUT, R])).unwrap_err();
        assert!(is_non_finite(&e), "{poke} in B: {e}");
    }
    // A NaN adapter under alpha 0 still refuses: 0 * NaN is NaN.
    let zero = LoraConfig::new(R, 0.0, 0.0).unwrap();
    let mut av = a.clone();
    av[0] = f32::NAN;
    let e = merged_weight_bf16(&zero, &wt, &f(&av, &[R, INN]), &f(&b, &[OUT, R]), &be).unwrap_err();
    assert!(is_non_finite(&e));

    // Every refusal above released what it charged.
    let (good_a, good_b) = (f(&a, &[R, INN]), f(&b, &[OUT, R]));
    let live = bud.live_bytes().unwrap();
    assert!(run(&ints, &good_a, &good_b).is_err());
    assert!(run(&wt, &bad_a, &good_b).is_err());
    assert!(merged_weight_bf16(&zero, &wt, &f(&av, &[R, INN]), &good_b, &be).is_err());
    assert_eq!(bud.live_bytes().unwrap(), live);
}

/// A finite f32 sum that rounds past the largest bf16 (`0x7F7F`) is refused,
/// not written as infinity.
#[test]
fn a_sum_that_rounds_to_bf16_infinity_is_refused() {
    let bud = budget();
    let be = cpu_on(&bud);
    let cfg = LoraConfig::new(1, 1.0, 0.0).unwrap();
    let t = |v: f32, shape: &[usize]| {
        Tensor::from_f32(&vec![v; shape.iter().product()], shape, &bud).unwrap()
    };
    // 3.0e38 + 0.4e38 = 3.4e38 < f32::MAX, above the bf16 midpoint to infinity.
    let sum = 3.0e38f32 + 0.4e38f32;
    assert!(sum.is_finite() && f32_to_bf16(sum) & 0x7F80 == 0x7F80);
    let e = merged_weight_bf16(
        &cfg,
        &t(3.0e38, &[2, 3]),
        &t(1.0, &[1, 3]),
        &t(0.4e38, &[2, 1]),
        &be,
    )
    .unwrap_err();
    assert!(is_non_finite(&e), "{e}");
    // The same call just below the edge works.
    let ok = merged_weight_bf16(
        &cfg,
        &t(3.0e38, &[2, 3]),
        &t(1.0, &[1, 3]),
        &t(0.1e38, &[2, 1]),
        &be,
    )
    .unwrap();
    assert!(ok
        .bf16_slice()
        .unwrap()
        .iter()
        .all(|&b| b & 0x7F80 != 0x7F80));
}

#[test]
fn a_set_that_does_not_match_its_targets_is_refused_before_anything_is_read() {
    let (s, base, bud) = (spec(), base_file(), budget());
    let be = cpu_on(&bud);
    let cfg = config(ALPHA);
    let set = sites(&bud);
    let t = targets();
    let run = |set: &Qwen35Lora<LoraSite<Tensor>>, targets: &[LoraTarget]| {
        let mut calls = 0usize;
        let r = set.merge_bf16(&s, targets, &cfg, &base, &be, &mut |_, _| {
            calls += 1;
            Ok(())
        });
        (r, calls)
    };
    let live = bud.live_bytes().unwrap();

    // An adapter no target names would be silently dropped from the export.
    let (r, calls) = run(&set, &t[..t.len() - 1]);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    assert_eq!(calls, 0);

    // A duplicate target.
    let mut twice = t.clone();
    twice.push(t[0].clone());
    let (r, calls) = run(&set, &twice);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    assert_eq!(calls, 0);

    // A target with no adapter.
    let mut empty = set.clone();
    *empty.slot_mut(0, LoraKind::GdnQkv).unwrap() = None;
    let (r, calls) = run(&empty, &t);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    assert_eq!(calls, 0);

    // B parts that are not the target's parts (qkv has three, here one).
    let mut parts = set.clone();
    let slot = parts.slot_mut(0, LoraKind::GdnQkv).unwrap();
    let site = slot.as_mut().unwrap();
    site.b.truncate(1);
    let (r, calls) = run(&parts, &t);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    assert_eq!(calls, 0);

    // A B part of the wrong rank.
    let mut wrong = set.clone();
    let site = wrong
        .slot_mut(0, LoraKind::GdnQkv)
        .unwrap()
        .as_mut()
        .unwrap();
    let rows = t[0].out_parts[0];
    site.b[0] = Tensor::from_f32(&vec![0.0; rows * (RANK + 1)], &[rows, RANK + 1], &bud).unwrap();
    let (r, _) = run(&wrong, &t);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");

    // A NaN in an adapter stops the run with nothing left charged.
    let mut nan = set.clone();
    let site = nan.slot_mut(0, LoraKind::GdnQkv).unwrap().as_mut().unwrap();
    let mut a = site.a.to_f32_vec().unwrap();
    a[0] = f32::NAN;
    site.a = Tensor::from_f32(&a, &[RANK, t[0].in_features], &bud).unwrap();
    let (r, calls) = run(&nan, &t);
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
    assert_eq!(calls, 0);

    // A sink error stops the run and is the run's error.
    let mut seen = 0usize;
    let r = set.merge_bf16(&s, &t, &cfg, &base, &be, &mut |_, _| {
        seen += 1;
        Err(OjasError::OutOfRange {
            op: "test::sink",
            detail: "stop".into(),
        })
    });
    assert!(
        matches!(
            r,
            Err(OjasError::OutOfRange {
                op: "test::sink",
                ..
            })
        ),
        "{r:?}"
    );
    assert_eq!(seen, 1);

    drop((empty, parts, wrong, nan));
    assert_eq!(bud.live_bytes().unwrap(), live);
}

// ---- the budget ----------------------------------------------------------------

/// One weight plus temporaries. Per target the export's transient over what
/// was already live is the base weight it reads (f32 here), the f32 product,
/// the bf16 result and `A^T` plus the backend's scratch: against
/// `LoraLinear::merge` (the f32 product and the f32 merged weight, with the
/// base already live) that is at most one more f32 weight, and a bf16 base
/// costs less than an f32 one. Across the set a sink that drops each tensor
/// never holds two: the set's peak is the worst single target's.
#[test]
fn the_budget_peak_is_one_weight_plus_temporaries() {
    let (s, base, bud) = (spec(), base_file(), budget());
    let be = cpu_on(&bud);
    let cfg = config(ALPHA);
    let set = sites(&bud);
    let t = targets();
    let live0 = bud.live_bytes().unwrap();

    // One target's transient over what was live when it started, and its
    // release.
    let single = |file: &SafeTensors<'_>, target: &LoraTarget| -> u64 {
        let site = set
            .slot(target.layer, target.kind)
            .unwrap()
            .as_ref()
            .unwrap();
        let live = bud.live_bytes().unwrap();
        bud.reset_peak();
        let merged = target.merged_bf16(&s, site, &cfg, file, &be).unwrap();
        let peak = bud.peak_bytes();
        drop(merged);
        assert_eq!(bud.live_bytes().unwrap(), live, "{}: leak", target.name);
        peak - live
    };
    let worst = t.iter().map(|x| single(&base, x)).max().unwrap();

    bud.reset_peak();
    let mut count = 0usize;
    set.merge_bf16(&s, &t, &cfg, &base, &be, &mut |_, merged| {
        count += 1;
        drop(merged);
        Ok(())
    })
    .unwrap();
    assert_eq!(count, t.len());
    assert!(
        bud.peak_bytes() - live0 <= worst,
        "set peak {} over one target's {worst}",
        bud.peak_bytes() - live0
    );
    assert_eq!(bud.live_bytes().unwrap(), live0);

    // Against merge() on the biggest target.
    let big = t
        .iter()
        .max_by_key(|x| out_rows(x) * x.in_features)
        .unwrap();
    let (out, inn) = (out_rows(big), big.in_features);
    let weight_f32 = (out * inn * 4) as u64;
    let (a, fused) = adapter(&lora_file(), big);
    let w = base.read_f32_widened(&big.name).unwrap().1;
    let mut layer = LoraLinear::new(
        cfg,
        Tensor::from_f32(&w, &[out, inn], &bud).unwrap(),
        Tensor::from_f32(&a, &[RANK, inn], &bud).unwrap(),
        Tensor::from_f32(&fused, &[out, RANK], &bud).unwrap(),
    )
    .unwrap();
    let live1 = bud.live_bytes().unwrap();
    bud.reset_peak();
    layer.merge(&be).unwrap();
    let merge_transient = bud.peak_bytes() - live1;
    let f32_bytes = f32_base_bytes();
    let f32_base = SafeTensors::parse(&f32_bytes).unwrap();
    let export_transient = single(&f32_base, big);
    assert!(
        export_transient <= merge_transient + weight_f32,
        "export {export_transient} over merge {merge_transient} + one weight {weight_f32}"
    );
    eprintln!(
        "{}: export transient {export_transient} B, merge transient {merge_transient} B, f32 weight {weight_f32} B",
        big.name
    );

    // A bf16 base reads half the bytes.
    let bytes = bf16_base_bytes();
    let bf16 = SafeTensors::parse(&bytes).unwrap();
    assert!(single(&bf16, big) < export_transient);
}

// ---- the forward -----------------------------------------------------------------

/// Both bounds are on `max |h - h_lora| / max |h_lora|` of the final hidden
/// state, `[2, 69, 64]`, computed by `Eval` in f32 on the exact CPU backend.
///
/// `F32_MERGE_BOUND`: a dense forward with `W' = W + s B A` kept in f32
/// (`LoraLinear::merge`) against the LoRA forward `x W^T + (x A^T)(s B)^T`.
/// The two are the same function; they differ only in where the f32 sums
/// round. Each linear's output moves by a few f32 ulps (`2^-24` relative to
/// the sum of the terms' magnitudes, times the depth of the reduction, 64 to
/// 256), the 13 linears and the norms keep that scale, so the ratio is
/// `1e-5` to `1e-4` at the very worst; the bound is `1e-4`, the repo's
/// gradient bound.
///
/// `BF16_BOUND`: the bf16 export. Each merged weight carries one rounding,
/// `|W'_bf16 - W'| <= 2^-8 |W'|` elementwise (bf16 keeps 8 significant bits;
/// half an ulp is `2^-9` of the binade's lower edge, `2^-8` of the value at
/// worst). A linear's output error is then `|dy_i| <= 2^-8 sum_j |x_j W'_ij|`.
/// For a dense random row that sum is about `0.64 sqrt(n)` times `|y_i|`'s
/// typical size (`n` = 64 or 128 here), so a single linear's worst case is
/// about `5 * 2^-8 = 2%` of its output scale, and independent signs give
/// about `2^-8` in practice. Over the 13 linears, whose errors add through
/// the residual stream and are renormalised by RMSNorm, the worst case is
/// below `13 * 2% = 25%` and the expected value near `1%` of the output's
/// standard deviation, which is under `0.3%` of the peak (the peak of 4096
/// values is above 3 standard deviations). The bound is `5e-2`, twice the
/// expected-value scale times a safety factor that the worst case would
/// still exceed: it holds a dropped or wrong scaling only coarsely, which
/// `merged_tensors_are_bit_identical_to_the_host_recomputation` and the f32
/// bound above hold sharply. The measured ratios are printed.
const F32_MERGE_BOUND: f64 = 1e-4;
const BF16_BOUND: f64 = 5e-2;

fn ids(t: usize) -> Tensor {
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let (shape, ids) = reference.read_i64("ids").unwrap();
    let (rows, full) = (shape[0] as usize, shape[1] as usize);
    let mut v = Vec::new();
    for r in 0..rows {
        for i in 0..t {
            v.push(u32::try_from(ids[r * full + i]).unwrap());
        }
    }
    Tensor::from_u32(&v, &[rows, t], &budget()).unwrap()
}

fn hidden(
    params: &Qwen35Params<Tensor>,
    adapters: Option<&Qwen35Lora<LoraSite<Tensor>>>,
    ids: &Tensor,
    tables: &Qwen35Tables,
) -> Vec<f32> {
    let s = spec();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, params).unwrap();
    let [b, t] = [ids.shape()[0], ids.shape()[1]];
    let fwd = adapters.map(|sites| {
        let mut rng = CounterRng::new(0);
        LoraForward::new(
            &mut eval,
            sites.clone(),
            &config(ALPHA),
            b,
            t,
            &mut rng,
            &budget(),
        )
        .unwrap()
    });
    let h = forward_hidden_lora(
        &mut eval,
        &s,
        &bound,
        fwd.as_ref(),
        ids,
        tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    h.to_f32_vec().unwrap()
}

fn ratio(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f32, |p, x| p.max(x.abs()));
    assert!(peak > 0.0);
    let worst = got.iter().zip(want).fold(0.0f32, |m, (a, b)| {
        assert!(a.is_finite());
        m.max((a - b).abs())
    });
    f64::from(worst) / f64::from(peak)
}

#[test]
fn the_merged_dense_forward_equals_the_lora_forward_within_the_derived_bounds() {
    let (s, base, lora, bud) = (spec(), base_file(), lora_file(), budget());
    let cfg = config(ALPHA);
    let set = sites(&bud);
    let ids = ids(69);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let load = |bytes: &[u8]| {
        let file = SafeTensors::parse(bytes).unwrap();
        load_hf(&s, &file, PREFIX, DType::F32, DType::F32, &budget()).unwrap()
    };

    let lora_h = hidden(&load_plain(&base), Some(&set), &ids, &tables);
    let base_h = hidden(&load_plain(&base), None, &ids, &tables);

    // f32 fold, through LoraLinear::merge.
    let be = cpu_on(&bud);
    let mut f32_edits = BTreeMap::new();
    for t in &targets() {
        let (out, inn) = (out_rows(t), t.in_features);
        let (a, fused) = adapter(&lora, t);
        let w = base.read_f32_widened(&t.name).unwrap().1;
        let mut layer = LoraLinear::new(
            cfg,
            Tensor::from_f32(&w, &[out, inn], &bud).unwrap(),
            Tensor::from_f32(&a, &[RANK, inn], &bud).unwrap(),
            Tensor::from_f32(&fused, &[out, RANK], &bud).unwrap(),
        )
        .unwrap();
        layer.merge(&be).unwrap();
        let bytes = encode_f32_as(StDtype::F32, layer.weight().f32_slice().unwrap()).unwrap();
        f32_edits.insert(t.name.clone(), (StDtype::F32, bytes));
    }
    let f32_h = hidden(&load(&checkpoint(&base, &f32_edits)), None, &ids, &tables);

    // The bf16 export.
    let mut bf16_edits = BTreeMap::new();
    for (name, tensor) in export(&set, &cfg, &base, &bud).unwrap() {
        let bytes: Vec<u8> = tensor
            .bf16_slice()
            .unwrap()
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        bf16_edits.insert(name, (StDtype::BF16, bytes));
    }
    let bf16_h = hidden(&load(&checkpoint(&base, &bf16_edits)), None, &ids, &tables);

    let r_f32 = ratio(&f32_h, &lora_h);
    let r_bf16 = ratio(&bf16_h, &lora_h);
    let r_base = ratio(&base_h, &lora_h);
    eprintln!(
        "hidden vs LoRA forward: f32 fold {r_f32:.2e} (bound {F32_MERGE_BOUND:e}), bf16 export {r_bf16:.2e} (bound {BF16_BOUND:e}), unadapted base {r_base:.2e}"
    );
    assert!(r_f32 <= F32_MERGE_BOUND, "f32 fold {r_f32:.3e}");
    assert!(r_bf16 <= BF16_BOUND, "bf16 export {r_bf16:.3e}");
    assert!(r_base > 0.0, "the adapters change the forward");
}

fn load_plain(base: &SafeTensors<'_>) -> Qwen35Params<Tensor> {
    load_hf(&spec(), base, PREFIX, DType::F32, DType::F32, &budget()).unwrap()
}

// ---- attack: f16 base, signed zeros and subnormals ---------------------------------

/// An f16 base is widened exactly and rounded once, so the export equals the
/// host recomputation over the f16 values widened to f32, bit for bit.
#[test]
fn an_f16_base_is_widened_exactly_and_rounded_once() {
    let base = base_file();
    let edits = targets()
        .iter()
        .map(|t| {
            let w = base.read_f32_widened(&t.name).unwrap().1;
            let bytes = encode_f32_as(StDtype::F16, &w).unwrap();
            (t.name.clone(), (StDtype::F16, bytes))
        })
        .collect();
    let bytes = checkpoint(&base, &edits);
    let f16_base = SafeTensors::parse(&bytes).unwrap();
    let bud = budget();
    let merged = export(&sites(&bud), &config(ALPHA), &f16_base, &bud).unwrap();
    assert_eq!(merged.len(), targets().len());
    for ((name, tensor), t) in merged.iter().zip(&targets()) {
        assert_eq!(bits_of(tensor), oracle(&f16_base, t, ALPHA), "{name}");
    }
}

/// With a zero adapter the sum is `w + 0`: a subnormal keeps its bits (no
/// flush), and `-0.0 + 0.0` is `+0.0` exactly as `LoraLinear::merge`'s f32
/// sum forms it, so the export is not bit-identical to the stored `-0.0`.
#[test]
fn subnormals_survive_and_negative_zero_follows_the_f32_sum() {
    let bud = budget();
    let be = cpu_on(&bud);
    let cfg = LoraConfig::new(1, 1.0, 0.0).unwrap();
    let stored: [u16; 6] = [0x0001, 0x8001, 0x007F, 0x8000, 0x0000, 0x3F80];
    let w = Tensor::from_bf16_bits(&stored, &[2, 3], &bud).unwrap();
    let a = Tensor::from_f32(&[0.0; 3], &[1, 3], &bud).unwrap();
    let b = Tensor::from_f32(&[0.0; 2], &[2, 1], &bud).unwrap();
    let got = merged_weight_bf16(&cfg, &w, &a, &b, &be).unwrap();
    assert_eq!(
        got.bf16_slice().unwrap(),
        [0x0001, 0x8001, 0x007F, 0x0000, 0x0000, 0x3F80]
    );
}
