//! The provider's API shape and every host-side check, with no GPU: sequence
//! and external-gradient refusals, the clip coefficient, shard planning, the
//! layout transpose, and the state-directory format written and checked
//! through ojas-io.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_core::{Budget, OjasError, Tensor};
use ojas_qwen35::{
    clip_coefficient, plan_shards, tower_tensors, transpose_2d, validate_external_grad, write_kind, AdamWHyper,
    BankState, ExternalGrad, Kind, NamedTensor, Numerics, OptimizerPlan, Pending, Qwen35Error, Qwen35Step,
    Qwen35TextConfig, Sequence, Snapshot, StateIndex, Which, SHARD_BYTES,
};

const REAL: &str = include_str!("fixtures/qwen35_2b_base_config.json");

fn real() -> Qwen35TextConfig {
    Qwen35TextConfig::from_json(REAL).unwrap()
}

fn tiny() -> Qwen35TextConfig {
    Qwen35TextConfig::from_file(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl/tests/fixtures/qwen35_train/config.json"),
    )
    .unwrap()
}

/// The GPU methods exist with these signatures (checked by the compiler; no
/// runtime is opened). L-trainer's adapter is written against exactly these.
#[test]
fn the_provider_api_has_this_shape() {
    let _open: fn(&Snapshot, Numerics, Budget) -> ojas_qwen35::Result<Qwen35Step> = Qwen35Step::open;
    let _forward: fn(&Qwen35Step, &Sequence<'_>) -> ojas_qwen35::Result<Pending> = Qwen35Step::forward;
    let _backward: fn(&mut Qwen35Step, Pending, Option<ExternalGrad<'_>>) -> ojas_qwen35::Result<()> =
        Qwen35Step::backward;
    let _norm: fn(&Qwen35Step) -> ojas_qwen35::Result<f64> = Qwen35Step::grad_sq_norm;
    let _adamw: fn(&mut Qwen35Step, &AdamWHyper, &OptimizerPlan) -> ojas_qwen35::Result<u64> = Qwen35Step::adamw_step;
    let _save: fn(&Qwen35Step, &Path) -> ojas_qwen35::Result<()> = Qwen35Step::save_state;
    let _load: fn(&mut Qwen35Step, &Path) -> ojas_qwen35::Result<()> = Qwen35Step::load_state;
    let _read: fn(&Qwen35Step, Which) -> ojas_qwen35::Result<Vec<NamedTensor>> = Qwen35Step::read_table;
    let _state: fn(&Qwen35Step) -> &BankState = Qwen35Step::bank_state;
    let _hidden: fn(&Pending) -> &Tensor = Pending::hidden;
    let _sum: fn(&Pending) -> f64 = Pending::letter_ce_sum;
    // Numerics has exactly the two arms and no default.
    for n in [Numerics::ExactF32, Numerics::Bf16Operands] {
        match n {
            Numerics::ExactF32 | Numerics::Bf16Operands => {}
        }
    }
}

fn seq<'a>(ids: &'a [u32], rows: &'a [u32], targets: &'a [u32], scale: f32, spans: &'a [u32]) -> Sequence<'a> {
    Sequence {
        ids,
        letter_rows: rows,
        letter_targets: targets,
        letter_scale: scale,
        span_positions: spans,
    }
}

#[test]
fn a_sequence_is_refused_on_the_host_before_any_device_work() {
    let c = real();
    let ids: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    seq(&ids, &[5, 9], &[33, 34], 0.25, &[0, 3, 3, 31]).validate(&c).unwrap();
    // Span-only: no letter rows, any finite non-negative scale.
    seq(&ids, &[], &[], 0.0, &[2]).validate(&c).unwrap();
    let refused = |s: Sequence<'_>, needle: &str| {
        let e = s.validate(&c).err().unwrap_or_else(|| panic!("{needle}: accepted")).to_string();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    };
    refused(seq(&[], &[], &[], 1.0, &[]), "no tokens");
    refused(seq(&[1, 248_320], &[], &[], 1.0, &[]), "ids[1] = 248320 >= vocab 248320");
    refused(seq(&[1, 248_056, 2], &[], &[], 1.0, &[]), "image_token_id");
    refused(seq(&[1, 248_053], &[], &[], 1.0, &[]), "vision_start_token_id");
    refused(seq(&ids, &[5, 9], &[33], 0.25, &[]), "2 letter rows but 1 targets");
    refused(seq(&ids, &[5, 5], &[33, 34], 0.25, &[]), "letter row 5 is supervised twice");
    refused(seq(&ids, &[32], &[33], 0.25, &[]), "letter row 32 >= 32 tokens");
    refused(seq(&ids, &[5], &[248_320], 0.25, &[]), "letter target 248320 >= vocab");
    refused(seq(&ids, &[5], &[33], f32::NAN, &[]), "letter_scale NaN");
    refused(seq(&ids, &[5], &[33], 0.0, &[]), "> 0 when there are letter rows");
    refused(seq(&ids, &[5], &[33], -1.0, &[]), "letter_scale -1");
    refused(seq(&ids, &[], &[], 1.0, &[32]), "span position 32 >= 32 tokens");
    let long = vec![7u32; 262_145];
    refused(seq(&long, &[], &[], 1.0, &[]), "exceed max_position_embeddings 262144");
}

#[test]
fn an_external_gradient_is_refused_on_the_host() {
    let b = Budget::new(1 << 20);
    let h = 4;
    let dh = Tensor::from_f32(&[0.5; 8], &[2, h], &b).unwrap();
    let ok = validate_external_grad(&ExternalGrad { positions: &[3, 1], dh: &dh }, 8, h).unwrap();
    assert_eq!(ok, vec![0.5; 8]);
    let refused = |positions: &[u32], dh: &Tensor, needle: &str| {
        let e = validate_external_grad(&ExternalGrad { positions, dh }, 8, h)
            .err()
            .unwrap_or_else(|| panic!("{needle}: accepted"))
            .to_string();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    };
    refused(&[3, 3], &dh, "position 3 appears twice");
    refused(&[3, 8], &dh, "position 8 >= 8 tokens");
    refused(&[3], &dh, "dh must be f32 [1, 4]");
    let mut bad = vec![0.0f32; 8];
    bad[6] = f32::NAN;
    refused(&[3, 1], &Tensor::from_f32(&bad, &[2, h], &b).unwrap(), "dh[1, 2] = NaN is not finite");
    refused(&[3, 1], &Tensor::from_u32(&[0; 8], &[2, h], &b).unwrap(), "dh must be f32");
    // None at all is a valid (empty) gradient.
    let empty = Tensor::from_f32(&[], &[0, h], &b).unwrap();
    assert!(validate_external_grad(&ExternalGrad { positions: &[], dh: &empty }, 8, h).unwrap().is_empty());
}

#[test]
fn the_clip_coefficient_is_torchs_through_ojas_core() {
    // Below max_norm: 1.
    assert_eq!(clip_coefficient(1.0, 0.25, 0.0).unwrap(), 1.0);
    // Tower and head together: sqrt(9 + 16) = 5.
    let c = clip_coefficient(1.0, 9.0, 16.0).unwrap();
    assert_eq!(c, 1.0f32 / (5.0f32 + ojas_core::CLIP_GRAD_NORM_EPS));
    assert!(matches!(
        clip_coefficient(1.0, f64::NAN, 0.0),
        Err(Qwen35Error::Ojas(OjasError::NonFinite { .. }))
    ));
    assert!(matches!(
        clip_coefficient(1.0, 1.0, f64::INFINITY),
        Err(Qwen35Error::Ojas(OjasError::NonFinite { .. }))
    ));
    assert!(clip_coefficient(1.0, -1.0, 0.0).is_err());
    assert!(matches!(
        clip_coefficient(-1.0, 1.0, 0.0),
        Err(Qwen35Error::Ojas(OjasError::OutOfRange { .. }))
    ));
}

#[test]
fn shards_cover_the_table_in_order_and_the_embedding_stands_alone() {
    let t = tower_tensors(&real());
    let shards = plan_shards(&t, SHARD_BYTES);
    assert_eq!(shards[0], 0..1, "the 2 GB embedding is larger than a shard");
    let mut at = 0;
    for r in &shards {
        assert_eq!(r.start, at);
        assert!(r.end > r.start);
        let bytes: u64 = t[r.clone()].iter().map(|x| x.numel() as u64 * 4).sum();
        assert!(bytes <= SHARD_BYTES || r.len() == 1, "{r:?}: {bytes} bytes");
        at = r.end;
    }
    assert_eq!(at, t.len());
    assert!(plan_shards(&[], SHARD_BYTES).is_empty());
}

#[test]
fn transpose_is_its_own_inverse_and_checks_its_length() {
    let a: Vec<f32> = (0..6).map(|x| x as f32).collect();
    let t = transpose_2d(&a, 2, 3).unwrap();
    assert_eq!(t, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    assert_eq!(transpose_2d(&t, 3, 2).unwrap(), a);
    assert!(transpose_2d(&a, 4, 2).is_err());
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("ojas-qwen35-{name}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn value(kind: Kind, i: usize, j: usize) -> f32 {
    (kind as usize * 1000 + i) as f32 + j as f32 * 1e-3
}

fn write_all(dir: &Path, step: u64, summary: &str, table: &[ojas_qwen35::TensorSpec], cap: u64) {
    let shards = plan_shards(table, cap);
    for kind in Kind::ALL {
        write_kind(dir, kind, step, summary, table, &shards, |i| {
            Ok((0..table[i].numel()).map(|j| value(kind, i, j)).collect())
        })
        .unwrap();
    }
}

/// The state format round-trips through ojas-io on the host, and a
/// directory that is not exactly this tower at one step is refused.
#[test]
fn a_state_directory_round_trips_and_a_wrong_one_is_refused() {
    let c = tiny();
    let t = tower_tensors(&c);
    let summary = c.canonical_summary();
    // A small cap so the tiny tower spans several shards.
    let cap = 64 * 1024;
    let dir = scratch("state-ok");
    write_all(&dir, 7, &summary, &t, cap);
    assert!(plan_shards(&t, cap).len() > 1);
    let idx = StateIndex::read(&dir, &t, &summary).unwrap();
    assert_eq!(idx.step, 7);
    for kind in Kind::ALL {
        for i in [0, 3, t.len() - 1] {
            let got = idx.read_entry(kind, i).unwrap();
            let want: Vec<f32> = (0..t[i].numel()).map(|j| value(kind, i, j)).collect();
            assert_eq!(got, want, "{kind:?} {}", t[i].name);
        }
    }

    let refused = |dir: &Path, summary: &str, needle: &str| {
        let e = match StateIndex::read(dir, &t, summary) {
            Ok(_) => panic!("{needle}: accepted"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    };
    // Another config.
    refused(&dir, &tiny_other_summary(&c), "metadata config");
    // A kind missing.
    let d = scratch("state-missing-kind");
    write_all(&d, 7, &summary, &t, cap);
    for e in std::fs::read_dir(&d).unwrap() {
        let p = e.unwrap().path();
        if p.file_name().unwrap().to_string_lossy().starts_with("exp_avg_sq-") {
            std::fs::remove_file(p).unwrap();
        }
    }
    refused(&d, &summary, "no exp_avg_sq files");
    // Two steps in one directory.
    let d = scratch("state-mixed-step");
    let shards = plan_shards(&t, cap);
    for (kind, step) in [(Kind::Masters, 7), (Kind::ExpAvg, 7), (Kind::ExpAvgSq, 8)] {
        write_kind(&d, kind, step, &summary, &t, &shards, |i| Ok(vec![0.0; t[i].numel()])).unwrap();
    }
    refused(&d, &summary, "another state file says 7");
    // A tensor missing: written against a table without the final norm.
    let d = scratch("state-missing-tensor");
    let short = &t[..t.len() - 1];
    let shards = plan_shards(short, cap);
    for kind in Kind::ALL {
        write_kind(&d, kind, 1, &summary, short, &shards, |i| Ok(vec![0.0; short[i].numel()])).unwrap();
    }
    refused(&d, &summary, "metadata entries");
    // A file that is not state.
    let d = scratch("state-stray");
    write_all(&d, 7, &summary, &t, cap);
    std::fs::write(d.join("notes.txt"), b"x").unwrap();
    refused(&d, &summary, "not a state file");
    // A value of the wrong length is refused at write time.
    let d = scratch("state-short-value");
    let e = write_kind(&d, Kind::Masters, 1, &summary, &t, &plan_shards(&t, cap), |_| Ok(vec![0.0; 3]))
        .err()
        .expect("accepted a short value")
        .to_string();
    assert!(e.contains("3 values for shape"), "{e}");
}

/// The summary of the same tower with one more vocabulary row: what a state
/// directory from another checkpoint shape would carry.
fn tiny_other_summary(c: &Qwen35TextConfig) -> String {
    c.canonical_summary().replace("vocab=64", "vocab=65")
}
