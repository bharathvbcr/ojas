//! LoRA on the Qwen3.5 tape, CPU, against the plain-torch LoRA arm
//! (`ojas-oracle/python/qwen35_lora_fixture.py`, after Lappi's
//! `qd_train/lora.py`): bounds (a'), (b'), (c') of the fine-tune brief on
//! the widened tiny fixture, the targets, dropout, merge/unmerge.
//!
//! The fixture's base is the `qwen35_tiny` checkpoint; rank 4, alpha 8
//! (scaling 2). The same clauses at scaling 1 are `qwen35_lora_scale1.rs`.

mod qwen35_common;

use ojas_autograd::Tape;
use ojas_core::Tensor;
use ojas_data::CounterRng;
use ojas_model::lora::{LoraConfig, LoraError, LoraLinear, LoraSite};
use ojas_model::qwen35::{bind, forward_loss, lora_targets, LoraKind, Qwen35Lora, LORA_KINDS};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use qwen35_common::lora::*;
use qwen35_common::*;

#[test]
fn targets_are_read_from_the_checkpoint_index_and_cover_every_linear() {
    let t = targets();
    // One GDN layer (qkv, z, out) and one attention layer (q, k, v, o), two
    // MLPs of three.
    assert_eq!(t.len(), 3 + 4 + 2 * 3);
    let index: Vec<String> = base_file().names().map(str::to_string).collect();
    for x in &t {
        assert!(index.contains(&x.name), "{} is not in the index", x.name);
        assert!(x.name.ends_with(".weight"));
    }
    let suffixes: Vec<&str> = t
        .iter()
        .map(|x| x.name.split_once(".layers.").unwrap().1)
        .map(|rest| rest.split_once('.').unwrap().1)
        .collect();
    assert!(suffixes.contains(&"linear_attn.in_proj_qkv.weight"));
    assert!(suffixes.contains(&"self_attn.o_proj.weight"));
    assert!(!suffixes
        .iter()
        .any(|s| s.contains("in_proj_a") || s.contains("in_proj_b")));
    for kind in LORA_KINDS {
        assert!(t.iter().any(|x| x.kind == kind), "{kind:?} has no target");
    }
}

#[test]
fn a_missing_or_misshapen_target_is_refused_by_name() {
    let mut s = spec();
    s.layers.push(ojas_model::qwen35::Qwen35Mixer::Attention);
    let err = lora_targets(&s, &base_file(), PREFIX, &LORA_KINDS)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("model.layers.2.self_attn.q_proj.weight")
            && err.contains("not in the checkpoint"),
        "{err}"
    );
    let mut s = spec();
    s.intermediate += 1;
    let err = lora_targets(&s, &base_file(), PREFIX, &[LoraKind::MlpGate])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("mlp.gate_proj.weight") && err.contains("shape"),
        "{err}"
    );
    assert!(lora_targets(&spec(), &base_file(), PREFIX, &[]).is_err());
}

/// (a') loss and every adapter gradient against plain torch; no gradient
/// for any frozen leaf.
#[test]
fn a_prime_loss_and_adapter_gradients_match_torch() {
    check_a_prime(SCALING_2);
}

/// (c') at B = 0 the LoRA loss is the base loss bit for bit.
#[test]
fn c_prime_zero_b_is_the_identity_bit_for_bit() {
    check_c_prime(SCALING_2);
}

/// (b') the loss clauses, and the run is bit-repeatable.
#[test]
fn b_prime_twenty_step_losses_match_torch_and_repeat_bit_for_bit() {
    check_b_prime_losses(SCALING_2);
}

/// (b') the final adapters, within the oracle's own noise floor
/// ([`check_b_prime_finals`]).
#[test]
fn b_prime_final_adapters_are_within_the_oracles_own_noise_floor() {
    check_b_prime_finals(SCALING_2);
}

/// A plain LoRA linear: dX, dA, dB against torch, and the frozen weight has
/// no gradient.
#[test]
fn a_prime_a_lora_linear_matches_torch_on_dx_da_db() {
    check_lora_linear(SCALING_2);
}

#[test]
fn merge_matches_torch_and_unmerge_restores_the_weight_bit_for_bit() {
    let file = lora_file();
    let bud = budget();
    let (inn, out) = (40, 24);
    let t =
        |name: &str, shape: &[usize]| Tensor::from_f32(&read(&file, name), shape, &bud).unwrap();
    let w = t("lin.w", &[out, inn]);
    let before: Vec<u32> = to_vec(&w).iter().map(|x| x.to_bits()).collect();
    let mut lin = LoraLinear::new(
        config(),
        w,
        t("lin.a", &[RANK, inn]),
        t("lin.b", &[out, RANK]),
    )
    .unwrap();
    assert!(matches!(lin.unmerge(), Err(LoraError::NotMerged)));
    let backend = cpu();
    lin.merge(&backend).unwrap();
    assert!(lin.is_merged());
    assert!(matches!(lin.merge(&backend), Err(LoraError::AlreadyMerged)));
    let r = peak_ratio(&to_vec(lin.weight()), &read(&file, "lin.merged"), "merged");
    assert!(r <= 1e-6, "merged W {r:.3e} of peak");
    lin.unmerge().unwrap();
    let after: Vec<u32> = to_vec(lin.weight()).iter().map(|x| x.to_bits()).collect();
    assert_eq!(before, after);
    assert!(matches!(lin.unmerge(), Err(LoraError::NotMerged)));
    // Merging again after an unmerge works and gives the same bits.
    lin.merge(&backend).unwrap();
    let first: Vec<u32> = to_vec(lin.weight()).iter().map(|x| x.to_bits()).collect();
    lin.unmerge().unwrap();
    lin.merge(&backend).unwrap();
    let second: Vec<u32> = to_vec(lin.weight()).iter().map(|x| x.to_bits()).collect();
    assert_eq!(first, second);
}

#[test]
fn a_bf16_base_merges_in_f32_and_unmerges_to_the_stored_bits() {
    let bud = budget();
    let (inn, out) = (16, 8);
    let wf: Vec<f32> = (0..inn * out).map(|i| (i as f32 * 0.37).sin()).collect();
    let bits: Vec<u16> = wf.iter().map(|&x| ojas_core::f32_to_bf16(x)).collect();
    let w = Tensor::from_bf16_bits(&bits, &[out, inn], &bud).unwrap();
    let a = Tensor::from_f32(&vec![0.1; RANK * inn], &[RANK, inn], &bud).unwrap();
    let b = Tensor::from_f32(&vec![0.2; out * RANK], &[out, RANK], &bud).unwrap();
    let mut lin = LoraLinear::new(config(), w, a, b).unwrap();
    lin.merge(&cpu()).unwrap();
    assert_eq!(lin.weight().dtype(), ojas_core::DType::F32);
    lin.unmerge().unwrap();
    assert_eq!(lin.weight().bf16_slice().unwrap(), &bits[..]);
}

/// A merged model reproduces the unmerged adapter's loss: every part's
/// weight folded as `W_part + s B_part A` (the parts of a fused tensor
/// share its `A`).
#[test]
fn a_merged_model_reproduces_the_unmerged_loss() {
    use ojas_model::qwen35::{MixerLora, MixerParams};
    let file = lora_file();
    let (d, mut host) = (data(), host_params());
    let sites = fixture_sites(&file, true);
    let unmerged = lora_step(cpu(), &host, &sites, &d, &Opts::new()).loss;
    let backend = cpu();
    let fold = |w: &mut Tensor, a: &Tensor, b: &Tensor| {
        let mut lin = LoraLinear::new(config(), w.clone(), a.clone(), b.clone()).unwrap();
        lin.merge(&backend).unwrap();
        *w = lin.weight().clone();
    };
    for (l, ls) in host.layers.iter_mut().zip(&sites.layers) {
        match (&mut l.mixer, &ls.mixer) {
            (MixerParams::GatedDeltaNet(m), MixerLora::GatedDeltaNet { qkv, z, out }) => {
                let qkv = qkv.as_ref().unwrap();
                for (w, b) in [&mut m.wq, &mut m.wk, &mut m.wv].into_iter().zip(&qkv.b) {
                    fold(w, &qkv.a, b);
                }
                let z = z.as_ref().unwrap();
                fold(&mut m.wz, &z.a, &z.b[0]);
                let o = out.as_ref().unwrap();
                fold(&mut m.out, &o.a, &o.b[0]);
            }
            (MixerParams::Attention(m), MixerLora::Attention { q, k, v, o }) => {
                let q = q.as_ref().unwrap();
                fold(&mut m.wq, &q.a, &q.b[0]);
                fold(&mut m.wgate, &q.a, &q.b[1]);
                for (w, s) in [(&mut m.wk, k), (&mut m.wv, v), (&mut m.wo, o)] {
                    let s = s.as_ref().unwrap();
                    fold(w, &s.a, &s.b[0]);
                }
            }
            _ => panic!("layer kinds differ"),
        }
        for (w, s) in [
            (&mut l.gate, &ls.gate),
            (&mut l.up, &ls.up),
            (&mut l.down, &ls.down),
        ] {
            let s = s.as_ref().unwrap();
            fold(w, &s.a, &s.b[0]);
        }
    }
    let mut tape = Tape::new(cpu());
    let vars = bind(&mut tape, &host).unwrap();
    let loss = forward_loss(
        &mut tape,
        &spec(),
        &vars,
        &d.ids,
        &d.targets,
        &d.tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    let merged = to_vec(tape.value(loss).unwrap())[0];
    let r = rel(merged, unmerged);
    eprintln!("merged loss {merged} vs unmerged {unmerged} (rel {r:.2e})");
    assert!(r <= LORA_LOSS_BOUND, "rel {r:.3e}");
}

#[test]
fn dropout_is_deterministic_under_a_seed_and_replays_under_checkpointing() {
    let file = lora_file();
    let (d, host, sites) = (data(), host_params(), fixture_sites(&file, true));
    let with = |seed: u64, p: f32, act: ActivationCheckpoint| {
        let o = Opts {
            config: LoraConfig::new(RANK, ALPHA, p).unwrap(),
            dropout_seed: seed,
            activations: act,
            bf16: false,
        };
        lora_step(cpu(), &host, &sites, &d, &o)
    };
    let off = ActivationCheckpoint::Off;
    let a = with(11, 0.3, off);
    let b = with(11, 0.3, off);
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    let c = with(12, 0.3, off);
    assert_ne!(a.loss.to_bits(), c.loss.to_bits());
    // Dropout 0 is the no-mask path, whatever the seed.
    let z = with(99, 0.0, off);
    let plain = lora_step(cpu(), &host, &sites, &d, &Opts::new());
    assert_eq!(z.loss.to_bits(), plain.loss.to_bits());
    assert_ne!(a.loss.to_bits(), plain.loss.to_bits());
    // A checkpointed replay reads the masks the first forward drew.
    let ck = with(11, 0.3, ActivationCheckpoint::Blocks);
    assert_eq!(a.loss.to_bits(), ck.loss.to_bits());
    for (x, y) in a.grads.iter().zip(&ck.grads) {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            (bits(&x.1), bits(&x.2)),
            (bits(&y.1), bits(&y.2)),
            "{}",
            x.0
        );
    }
}

#[test]
fn checkpointing_does_not_change_the_adapter_gradients() {
    let file = lora_file();
    let (d, host, sites) = (data(), host_params(), fixture_sites(&file, true));
    let off = lora_step(cpu(), &host, &sites, &d, &Opts::new());
    let on = lora_step(
        cpu(),
        &host,
        &sites,
        &d,
        &Opts {
            activations: ActivationCheckpoint::Blocks,
            ..Opts::new()
        },
    );
    assert_eq!(off.loss.to_bits(), on.loss.to_bits());
    for (x, y) in off.grads.iter().zip(&on.grads) {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            (bits(&x.1), bits(&x.2)),
            (bits(&y.1), bits(&y.2)),
            "{}",
            x.0
        );
    }
}

#[test]
fn alpha_zero_trains_nothing_and_changes_nothing() {
    let file = lora_file();
    let (d, host, sites) = (data(), host_params(), fixture_sites(&file, true));
    let o = Opts {
        config: LoraConfig::new(RANK, 0.0, 0.0).unwrap(),
        ..Opts::new()
    };
    let run = lora_step(cpu(), &host, &sites, &d, &o);
    let base = read(&file, "c.base_loss")[0];
    assert!(rel(run.loss, base) <= LORA_LOSS_BOUND);
    for (module, a, b) in &run.grads {
        assert!(a.iter().chain(b).all(|&x| x == 0.0), "{module}");
    }
}

#[test]
fn rank_edges_are_typed_errors() {
    assert!(matches!(
        LoraConfig::new(0, 8.0, 0.0),
        Err(LoraError::ZeroRank)
    ));
    let bud = budget();
    let c = LoraConfig::new(9, 8.0, 0.0).unwrap();
    let mut rng = CounterRng::new(1);
    assert!(matches!(
        LoraSite::init(&c, 64, &[8], &mut rng, &bud),
        Err(LoraError::RankTooLarge { .. })
    ));
    let w = Tensor::from_f32(&vec![0.0; 8 * 64], &[8, 64], &bud).unwrap();
    let a = Tensor::from_f32(&vec![0.0; 9 * 64], &[9, 64], &bud).unwrap();
    let b = Tensor::from_f32(&vec![0.0; 8 * 9], &[8, 9], &bud).unwrap();
    assert!(matches!(
        LoraLinear::new(c, w, a, b),
        Err(LoraError::RankTooLarge { .. })
    ));
}

#[test]
fn a_fresh_init_has_zero_b_and_bounded_a_per_target() {
    let (s, t) = (spec(), targets());
    let lora = Qwen35Lora::init(&s, &t, &config(), 5, &budget()).unwrap();
    let again = Qwen35Lora::init(&s, &t, &config(), 5, &budget()).unwrap();
    let other = Qwen35Lora::init(&s, &t, &config(), 6, &budget()).unwrap();
    for ((x, y), z) in lora.sites().iter().zip(again.sites()).zip(other.sites()) {
        assert_eq!(to_vec(&x.a), to_vec(&y.a));
        assert_ne!(to_vec(&x.a), to_vec(&z.a));
        let bound = 1.0 / (x.a.shape()[1] as f32).sqrt();
        assert!(to_vec(&x.a).iter().all(|v| v.abs() <= bound));
        for b in &x.b {
            assert!(to_vec(b).iter().all(|v| v.to_bits() == 0));
        }
    }
    assert_eq!(lora.sites().len(), t.len());
}
