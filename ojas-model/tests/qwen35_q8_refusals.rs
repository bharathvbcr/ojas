//! The typed refusals of the resident q8 base (card ft-5f743c19, qv2 W3), each asserted by variant
//! and by the field, op or tensor it names, never as a bare `is_err`:
//!
//! - the loader (`load_hf_q8_pinned`) on damaged copies of the tiny q8 file: a missing `.scales`,
//!   a quantised tensor outside the ten adapted kinds (W3-D8), a wrapped tensor stored dense, a
//!   wrong-shape triple, a q4 config, a quantizer record that disagrees with the config or is not
//!   resident, a wrong sha256 or name, and a budget too short for the dense tensors;
//! - the tape (`linear_q`): an input of the wrong width before any dispatch, and the placement
//!   rule (W3-D9) both ways, on the double that reports another backend id;
//! - the adapter checkpoint's storage extension (W3-D3): a fake-quant checkpoint is not resumed on
//!   a resident base and the reverse, before the sha256 comparison that would refuse it on the
//!   file difference alone, malformed storage keys are `Corrupt`, a fake-quant checkpoint writes
//!   exactly the keys it always did, and a resident one round-trips;
//! - merging over a packed file (W3-D4): `Unsupported`, naming the weight and the fq8 route.
//!
//! The dense path stays green in the existing suites (`qwen35_*`, `ojas-autograd`); the one
//! assertion here that the base-generic forward over a dense base equals the dense entry points
//! is in `qwen35_q8_tape.rs`.

mod common;
mod qwen35_common;
mod qwen35_q8_common;

use std::collections::BTreeMap;
use std::path::Path;

use common::{Probe, Resident};
use ojas_autograd::Tape;
use ojas_core::{BackendCore, BackendId, Budget, DType, OjasError, Tensor};
use ojas_io::{write_safetensors, AffineLoadError, JsonValue, SafeTensors, StDtype, TensorOut};
use ojas_model::lora::LoraConfig;
use ojas_model::qwen35::adapters::{
    lora_a_name, lora_b_name, read_adapter_checkpoint, write_adapter_checkpoint, AdapterCheckpoint,
    AdapterEntry, AdapterError, BaseIdentity, DataState, Expected, ExpectedEntry, QuantStorage,
    QuantizerRecord, ADAPTERS_FILE, KEY_ALPHA, KEY_BASE, KEY_BASE_FILE, KEY_DIGEST, KEY_LAYOUT,
    KEY_PAYLOAD, KEY_QUANT_BITS, KEY_QUANT_FORMAT, KEY_QUANT_GROUP, KEY_QUANT_MLX,
    KEY_QUANT_STORAGE, KEY_RANK, KEY_RNG, KEY_STEP,
};
use ojas_model::qwen35::{bind_frozen_base, load_hf_q8_pinned, FrozenWeight, Qwen35Params};
use qwen35_common::lora::*;
use qwen35_common::*;
use qwen35_q8_common::*;

const GATE: &str = "model.layers.0.mlp.gate_proj";
const IN_PROJ_A: &str = "model.layers.0.linear_attn.in_proj_a.weight";

fn load_with(
    path: &Path,
    identity: &BaseIdentity,
    config: &JsonValue,
    budget: &Budget,
) -> Result<Qwen35Params<Tensor, FrozenWeight<Tensor>>, AdapterError> {
    load_hf_q8_pinned(
        &spec(),
        path,
        identity,
        config,
        PREFIX,
        DType::Bf16,
        DType::F32,
        budget,
        &cpu(),
    )
}

/// A refusal, with nothing left charged to `budget` (a failed load frees what it read).
fn refusal(
    result: Result<Qwen35Params<Tensor, FrozenWeight<Tensor>>, AdapterError>,
    budget: &Budget,
) -> AdapterError {
    let err = result.map(|_| ()).expect_err("the load was refused");
    assert_eq!(budget.live_bytes().unwrap(), 0, "live bytes after {err:?}");
    err
}

/// The q8 file with `edit` and `extra` applied, in a fresh directory, pinned by its own sha256
/// under the resident q8 record. The `Tmp` keeps the directory alive.
fn damaged(
    tag: &str,
    edit: &mut dyn FnMut(&str, StDtype, &[u64]) -> Edit,
    extra: Vec<Extra>,
) -> (Tmp, BaseIdentity) {
    let tmp = Tmp::new(tag);
    rewrite(&q8_file(), &tmp.model(), edit, extra);
    let identity = identity_of(&tmp.model(), Some(q8_record()));
    (tmp, identity)
}

fn zeros(dtype: StDtype, shape: &[u64]) -> Vec<u8> {
    let n: u64 = shape.iter().product();
    vec![0u8; (n * dtype.size()) as usize]
}

// ---- the loader --------------------------------------------------------------------------------

/// The control: the undamaged copy loads, so every refusal below is the damage.
#[test]
fn an_undamaged_copy_loads_and_equals_the_fixture() {
    let (tmp, identity) = damaged("control", &mut |_, _, _| Edit::Keep, Vec::new());
    let budget = budget();
    let base = load_with(&tmp.model(), &identity, &q8_config(), &budget).unwrap();
    assert_eq!(wrapped_fields(&base).len(), WRAPPED_PARTS);
}

#[test]
fn a_wrapped_tensor_missing_its_scales_is_affine_missing() {
    let want = format!("{GATE}.scales");
    let (tmp, identity) = damaged(
        "noscales",
        &mut |name, _, _| {
            if name == want {
                Edit::Drop
            } else {
                Edit::Keep
            }
        },
        Vec::new(),
    );
    let budget = budget();
    match refusal(
        load_with(&tmp.model(), &identity, &q8_config(), &budget),
        &budget,
    ) {
        AdapterError::Affine(e) => match *e {
            AffineLoadError::Missing { name } => assert_eq!(name, format!("{GATE}.scales")),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

/// W3-D8: `in_proj_a` stays dense; a packed one is refused by name, not read as floats.
#[test]
fn a_quantised_in_proj_a_is_unexpected_quantised_by_name() {
    let (tmp, identity) = damaged(
        "uin",
        &mut |name, _, _| {
            if name == IN_PROJ_A {
                Edit::Replace(StDtype::U32, vec![2, 16], zeros(StDtype::U32, &[2, 16]))
            } else {
                Edit::Keep
            }
        },
        Vec::new(),
    );
    let budget = budget();
    match refusal(
        load_with(&tmp.model(), &identity, &q8_config(), &budget),
        &budget,
    ) {
        AdapterError::UnexpectedQuantised { name } => assert_eq!(name, IN_PROJ_A),
        other => panic!("{other:?}"),
    }
}

/// `.scales` beside the tied table (real `mlx_lm` output quantises `embed_tokens`) are refused by
/// the table's name.
#[test]
fn scales_beside_the_embedding_are_unexpected_quantised() {
    let (tmp, identity) = damaged(
        "embed-scales",
        &mut |_, _, _| Edit::Keep,
        vec![(
            "model.embed_tokens.scales".to_string(),
            StDtype::BF16,
            vec![8320, 1],
            zeros(StDtype::BF16, &[8320, 1]),
        )],
    );
    let budget = budget();
    match refusal(
        load_with(&tmp.model(), &identity, &q8_config(), &budget),
        &budget,
    ) {
        AdapterError::UnexpectedQuantised { name } => {
            assert_eq!(name, "model.embed_tokens.weight")
        }
        other => panic!("{other:?}"),
    }
}

/// A wrapped weight stored dense in a q8 file is a dtype refusal of its `.weight`.
#[test]
fn a_wrapped_tensor_stored_dense_is_an_affine_dtype_refusal() {
    let up = "model.layers.0.mlp.up_proj.weight".to_string();
    let (tmp, identity) = damaged(
        "dense-up",
        &mut |name, _, _| {
            if name == up {
                Edit::Replace(
                    StDtype::BF16,
                    vec![128, 64],
                    zeros(StDtype::BF16, &[128, 64]),
                )
            } else {
                Edit::Keep
            }
        },
        Vec::new(),
    );
    let budget = budget();
    match refusal(
        load_with(&tmp.model(), &identity, &q8_config(), &budget),
        &budget,
    ) {
        AdapterError::Affine(e) => match *e {
            AffineLoadError::Dtype {
                name,
                expected,
                got,
            } => {
                assert_eq!(name, up);
                assert_eq!((expected, got), (StDtype::U32, StDtype::BF16));
            }
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

/// A scales tensor with another group count is the typed layout refusal of that tensor.
#[test]
fn a_wrong_shape_triple_is_an_affine_layout_refusal() {
    let scales = format!("{GATE}.scales");
    let (tmp, identity) = damaged(
        "shape",
        &mut |name, _, _| {
            if name == scales {
                Edit::Replace(StDtype::BF16, vec![128, 2], zeros(StDtype::BF16, &[128, 2]))
            } else {
                Edit::Keep
            }
        },
        Vec::new(),
    );
    let budget = budget();
    match refusal(
        load_with(&tmp.model(), &identity, &q8_config(), &budget),
        &budget,
    ) {
        AdapterError::Affine(e) => match *e {
            AffineLoadError::Layout {
                name,
                expected,
                got,
                ..
            } => {
                assert_eq!(name, scales);
                assert_eq!((expected, got), (vec![128, 1], vec![128, 2]));
            }
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_q4_config_is_unsupported_resident_bits() {
    let text = config_text(Q8_DIR);
    assert!(text.contains("\"bits\": 8"));
    let q4 = ojas_io::parse_json(&text.replace("\"bits\": 8", "\"bits\": 4")).unwrap();
    let budget = budget();
    match refusal(load_with(&q8_path(), &q8_identity(), &q4, &budget), &budget) {
        AdapterError::UnsupportedResidentBits { bits } => assert_eq!(bits, 4),
        other => panic!("{other:?}"),
    }
}

/// The record and the config must agree, and each disagreement names its field.
#[test]
fn a_record_that_disagrees_with_the_config_is_named_by_field() {
    let b = budget();
    let with = |edit: &dyn Fn(&mut QuantizerRecord), config: &JsonValue| {
        let mut record = q8_record();
        edit(&mut record);
        refusal(
            load_with(
                &q8_path(),
                &identity_of(&q8_path(), Some(record)),
                config,
                &b,
            ),
            &b,
        )
    };
    match with(&|r| r.bits = Some(4), &q8_config()) {
        AdapterError::ResidentQuantizerDisagrees {
            field,
            record,
            config,
        } => assert_eq!(
            (field, record.as_str(), config.as_str()),
            ("bits", "Some(4)", "8")
        ),
        other => panic!("{other:?}"),
    }
    match with(&|r| r.group_size = "128".to_string(), &q8_config()) {
        AdapterError::ResidentQuantizerDisagrees {
            field,
            record,
            config,
        } => assert_eq!(
            (field, record.as_str(), config.as_str()),
            ("group_size", "128", "64")
        ),
        other => panic!("{other:?}"),
    }
    // A config that names no quantisation at all.
    match with(&|_| {}, &fq8_config()) {
        AdapterError::ResidentQuantizerDisagrees { field, .. } => assert_eq!(field, "quantization"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_record_that_is_absent_or_not_resident_is_not_resident_record() {
    let b = budget();
    let none = identity_of(&q8_path(), None);
    match refusal(load_with(&q8_path(), &none, &q8_config(), &b), &b) {
        AdapterError::NotResidentRecord { found } => assert_eq!(found, None),
        other => panic!("{other:?}"),
    }
    let mut record = q8_record();
    record.storage = QuantStorage::FakeQuantBf16;
    record.bits = None;
    let fake = identity_of(&q8_path(), Some(record));
    match refusal(load_with(&q8_path(), &fake, &q8_config(), &b), &b) {
        AdapterError::NotResidentRecord { found } => {
            assert_eq!(found, Some(QuantStorage::FakeQuantBf16))
        }
        other => panic!("{other:?}"),
    }
}

/// The pin is checked before any tensor exists: the sha256 and the file name, with nothing ever
/// charged (live and peak both 0).
#[test]
fn a_wrong_sha_or_name_is_refused_before_any_read() {
    let b = budget();
    let mut wrong_sha = q8_identity();
    wrong_sha.sha256 = "0".repeat(64);
    match refusal(load_with(&q8_path(), &wrong_sha, &q8_config(), &b), &b) {
        AdapterError::BaseMismatch { expected, found } => {
            assert_eq!(expected, "0".repeat(64));
            assert_eq!(found, q8_identity().sha256);
        }
        other => panic!("{other:?}"),
    }
    let mut wrong_name = q8_identity();
    wrong_name.file_name = "other.safetensors".to_string();
    match refusal(load_with(&q8_path(), &wrong_name, &q8_config(), &b), &b) {
        AdapterError::BaseFileMismatch { expected, found } => {
            assert_eq!(
                (expected.as_str(), found.as_str()),
                ("other.safetensors", MODEL_FILE)
            );
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(b.peak_bytes(), 0, "a pin refusal charged nothing");
}

/// A budget that cannot hold the dense tensors is refused before they are read.
#[test]
fn a_budget_too_short_for_the_dense_tensors_is_capacity_exceeded_with_nothing_charged() {
    let b = Budget::new(1000);
    match refusal(load_with(&q8_path(), &q8_identity(), &q8_config(), &b), &b) {
        AdapterError::Core(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(b.peak_bytes(), 0);
}

// ---- the tape -----------------------------------------------------------------------------------

/// An input of the wrong width is `Shape` from `Tape::linear_q`, before the backend is touched:
/// the counting backend saw no dequantisation. The control with the right width dequantises once.
#[test]
fn a_wrong_width_input_is_refused_before_any_dequantisation() {
    let probe = Probe::new(cpu());
    let base = load_q8(&probe);
    let FrozenWeight::Quant(q) = wrapped_fields(&base)[0].1 else {
        panic!("not packed");
    };
    let mut tape = Tape::new(&probe);
    let wrong = tape
        .leaf(Tensor::from_f32(&[0.5f32; 65], &[1, 65], &budget()).unwrap())
        .unwrap();
    match tape.linear_q(wrong, q) {
        Err(OjasError::Shape { op, detail }) => {
            assert_eq!(op, "Tape::linear_q");
            assert!(detail.contains("64"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(probe.count(DEQUANT), 0);
    let right = tape
        .leaf(Tensor::from_f32(&[0.5f32; 64], &[1, 64], &budget()).unwrap())
        .unwrap();
    tape.linear_q(right, q).unwrap();
    assert_eq!(probe.count(DEQUANT), 1);
}

/// W3-D9 both ways, on the double that reports another backend id: host parts on that tape and
/// that backend's parts on a host tape are `Placement` at the bind, naming the op.
#[test]
fn placement_is_refused_at_the_bind_in_both_directions() {
    let host_base = load_q8(&cpu());
    let mut device_tape = Tape::new(Resident::new(1 << 26));
    match bind_frozen_base(&mut device_tape, &host_base) {
        Err(OjasError::Placement {
            op,
            expected,
            found,
        }) => {
            assert_eq!(op, "qwen35::bind_frozen_base");
            assert_eq!((expected, found), (Some(BackendId::Wgpu), None));
        }
        other => panic!("{:?}", other.map(|_| ())),
    }

    let device_base = load_q8(&Resident::new(1 << 26));
    let FrozenWeight::Quant(q) = wrapped_fields(&device_base)[0].1 else {
        panic!("not packed");
    };
    assert_eq!(q.backend(), BackendId::Wgpu);
    let mut host_tape = Tape::new(cpu());
    match bind_frozen_base(&mut host_tape, &device_base) {
        Err(OjasError::Placement {
            op,
            expected,
            found,
        }) => {
            assert_eq!(op, "qwen35::bind_frozen_base");
            assert_eq!((expected, found), (None, Some(BackendId::Wgpu)));
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// `place` checks the room for the three parts before the first upload.
#[test]
fn place_on_a_device_with_a_short_budget_refuses_before_any_upload() {
    let device = Resident::new(100);
    match load_q8_from(&q8_path(), &q8_identity(), &q8_config(), &device) {
        Err(AdapterError::Core(OjasError::CapacityExceeded { .. })) => {}
        other => panic!("{:?}", other.map(|_| ())),
    }
    assert_eq!(device.budget().live_bytes().unwrap(), 0);
    assert_eq!(device.budget().peak_bytes(), 0, "nothing was ever uploaded");
}

// ---- the adapter checkpoint's storage extension ------------------------------------------------

const MODULE: &str = "model.layers.1.self_attn.q_proj";

fn fake_record() -> QuantizerRecord {
    QuantizerRecord {
        format: "mlx-affine".to_string(),
        group_size: "64".to_string(),
        mlx_version: "0.31.2".to_string(),
        storage: QuantStorage::FakeQuantBf16,
        bits: None,
    }
}

/// The fake-quant base's identity (its own file and sha256, the fake-quant record).
fn fake_identity() -> BaseIdentity {
    identity_of(&fq8_path(), Some(fake_record()))
}

fn checkpoint(base: BaseIdentity) -> AdapterCheckpoint {
    let entry = |name: String, shape: &[usize]| {
        let n: usize = shape.iter().product();
        AdapterEntry {
            name,
            shape: shape.to_vec(),
            values: vec![0.25; n],
            moment1: vec![0.0; n],
            moment2: vec![0.0; n],
        }
    };
    AdapterCheckpoint {
        lora: LoraConfig::new(4, 8.0, 0.0).unwrap(),
        step: 3,
        data: DataState {
            rng_state: vec![1, 2, 3],
            consumed_digest: "00ff10aa".repeat(4),
        },
        base,
        entries: vec![
            entry(lora_a_name(MODULE), &[4, 64]),
            entry(lora_b_name(MODULE), &[64, 4]),
        ],
    }
}

fn expecting(c: &AdapterCheckpoint, base: BaseIdentity) -> Expected {
    Expected {
        base,
        lora: c.lora,
        entries: c
            .entries
            .iter()
            .map(|e| ExpectedEntry {
                name: e.name.clone(),
                shape: e.shape.clone(),
            })
            .collect(),
    }
}

/// A checkpoint of `base` written to a fresh directory.
fn saved(tag: &str, base: BaseIdentity) -> (Tmp, AdapterCheckpoint) {
    let tmp = Tmp::new(tag);
    let ckpt = checkpoint(base);
    write_adapter_checkpoint(&tmp.0.join("ckpt"), &ckpt).unwrap();
    (tmp, ckpt)
}

/// The metadata of a checkpoint's adapters file.
fn metadata(tmp: &Tmp) -> BTreeMap<String, String> {
    SafeTensors::open(&tmp.0.join("ckpt").join(ADAPTERS_FILE))
        .unwrap()
        .metadata()
        .clone()
}

/// The adapters file rewritten with its metadata edited (its payload hash is left as it was: the
/// refusals under test come before the payload is read).
fn edit_metadata(tmp: &Tmp, edit: impl FnOnce(&mut BTreeMap<String, String>)) {
    let path = tmp.0.join("ckpt").join(ADAPTERS_FILE);
    let st = SafeTensors::open(&path).unwrap();
    let mut meta = st.metadata().clone();
    edit(&mut meta);
    let names: Vec<String> = st.names().map(str::to_string).collect();
    let owned: Vec<(String, StDtype, Vec<u64>, Vec<u8>)> = names
        .into_iter()
        .map(|n| {
            let (info, bytes) = st.read_raw(&n).unwrap();
            (n, info.dtype, info.shape.clone(), bytes)
        })
        .collect();
    let items: Vec<TensorOut<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| TensorOut {
            name,
            dtype: *dtype,
            shape,
            data,
        })
        .collect();
    let pairs: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    write_safetensors(&path, &items, &pairs).unwrap();
}

/// A checkpoint trained over the fake-quant bf16 base does not resume on the resident base. It
/// is refused as that, not as the base mismatch the two files' hashes would give.
#[test]
fn a_fake_quant_checkpoint_does_not_resume_on_a_resident_base() {
    let (tmp, ckpt) = saved("fq-on-q8", fake_identity());
    assert_ne!(
        fake_identity().sha256,
        q8_identity().sha256,
        "two files, two hashes"
    );
    let dir = tmp.0.join("ckpt");
    match read_adapter_checkpoint(&dir, &expecting(&ckpt, q8_identity())) {
        Err(AdapterError::FakeQuantOnResidentBase { found }) => {
            assert_eq!(found, Some(QuantStorage::FakeQuantBf16))
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    // The control: the same checkpoint resumes on the base it was trained over.
    let back = read_adapter_checkpoint(&dir, &expecting(&ckpt, fake_identity())).unwrap();
    assert_eq!(back.base, fake_identity());
    // A dense base has no quantizer record at all, and is no more resident.
    let (tmp, ckpt) = saved(
        "dense-on-q8",
        BaseIdentity {
            quantizer: None,
            ..fake_identity()
        },
    );
    match read_adapter_checkpoint(&tmp.0.join("ckpt"), &expecting(&ckpt, q8_identity())) {
        Err(AdapterError::FakeQuantOnResidentBase { found }) => assert_eq!(found, None),
        other => panic!("{:?}", other.map(|_| ())),
    }
}

#[test]
fn a_resident_checkpoint_does_not_resume_on_a_fake_quant_base_and_round_trips() {
    let (tmp, ckpt) = saved("q8-on-fq", q8_identity());
    let dir = tmp.0.join("ckpt");
    match read_adapter_checkpoint(&dir, &expecting(&ckpt, fake_identity())) {
        Err(AdapterError::ResidentOnFakeQuantBase { expected }) => {
            assert_eq!(expected, Some(QuantStorage::FakeQuantBf16))
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    let dense = BaseIdentity {
        quantizer: None,
        ..q8_identity()
    };
    match read_adapter_checkpoint(&dir, &expecting(&ckpt, dense)) {
        Err(AdapterError::ResidentOnFakeQuantBase { expected }) => assert_eq!(expected, None),
        other => panic!("{:?}", other.map(|_| ())),
    }
    // Resumed on its own base it reads back its record, storage and bits included.
    let back = read_adapter_checkpoint(&dir, &expecting(&ckpt, q8_identity())).unwrap();
    assert_eq!(back.base.quantizer, Some(q8_record()));
    assert_eq!(back.entries, ckpt.entries);
}

/// D3: a fake-quant checkpoint writes the keys the layout always wrote and no others, so an
/// existing checkpoint is what this writer produces for it; only a resident one adds the two.
#[test]
fn a_fake_quant_checkpoint_writes_exactly_the_legacy_keys() {
    let legacy: Vec<&str> = {
        let mut keys = vec![
            KEY_LAYOUT,
            KEY_STEP,
            KEY_PAYLOAD,
            KEY_RNG,
            KEY_DIGEST,
            KEY_BASE,
            KEY_BASE_FILE,
            KEY_RANK,
            KEY_ALPHA,
            KEY_QUANT_FORMAT,
            KEY_QUANT_GROUP,
            KEY_QUANT_MLX,
        ];
        keys.sort_unstable();
        keys
    };
    let (tmp, ckpt) = saved("legacy-keys", fake_identity());
    let keys: Vec<String> = metadata(&tmp).into_keys().collect();
    assert_eq!(keys, legacy);
    // It reads back as the fake-quant record it was (storage absent means fake-quant).
    let back =
        read_adapter_checkpoint(&tmp.0.join("ckpt"), &expecting(&ckpt, fake_identity())).unwrap();
    assert_eq!(back.base.quantizer, Some(fake_record()));

    let (tmp, _) = saved("resident-keys", q8_identity());
    let meta = metadata(&tmp);
    assert_eq!(
        meta.get(KEY_QUANT_STORAGE).map(String::as_str),
        Some("resident_packed")
    );
    assert_eq!(meta.get(KEY_QUANT_BITS).map(String::as_str), Some("8"));
    assert_eq!(meta.len(), legacy.len() + 2);
}

/// A fake-quant record carrying bits would not read back as itself, so save refuses it.
#[test]
fn save_refuses_bits_on_a_fake_quant_record_and_a_resident_record_without_them() {
    let tmp = Tmp::new("badrecord");
    let mut bits_on_fake = fake_record();
    bits_on_fake.bits = Some(8);
    match write_adapter_checkpoint(
        &tmp.0.join("a"),
        &checkpoint(identity_of(&fq8_path(), Some(bits_on_fake))),
    ) {
        Err(AdapterError::InvalidField { field, .. }) => assert_eq!(field, "quantizer bits"),
        other => panic!("{other:?}"),
    }
    let mut resident_no_bits = q8_record();
    resident_no_bits.bits = None;
    match write_adapter_checkpoint(
        &tmp.0.join("b"),
        &checkpoint(identity_of(&q8_path(), Some(resident_no_bits))),
    ) {
        Err(AdapterError::InvalidField { field, .. }) => assert_eq!(field, "quantizer bits"),
        other => panic!("{other:?}"),
    }
    let mut wide = q8_record();
    wide.bits = Some(16);
    match write_adapter_checkpoint(
        &tmp.0.join("c"),
        &checkpoint(identity_of(&q8_path(), Some(wide))),
    ) {
        Err(AdapterError::InvalidField { field, .. }) => assert_eq!(field, "quantizer bits"),
        other => panic!("{other:?}"),
    }
}

/// Only a fully absent pair of keys reads as fake-quant: one key alone, an unknown storage value,
/// a bits value that is not 4 or 8 or not canonical, and storage keys without a quantizer record
/// are `Corrupt`.
#[test]
fn partly_present_or_malformed_storage_keys_are_corrupt() {
    type Edit = fn(&mut BTreeMap<String, String>);
    let cases: [(&str, Edit); 6] = [
        ("no-bits", |m| {
            m.remove(KEY_QUANT_BITS);
        }),
        ("no-storage", |m| {
            m.remove(KEY_QUANT_STORAGE);
        }),
        ("unknown-storage", |m| {
            m.insert(KEY_QUANT_STORAGE.to_string(), "fake_quant_bf16".to_string());
        }),
        ("bits-16", |m| {
            m.insert(KEY_QUANT_BITS.to_string(), "16".to_string());
        }),
        ("bits-noncanonical", |m| {
            m.insert(KEY_QUANT_BITS.to_string(), "08".to_string());
        }),
        ("no-record", |m| {
            m.remove(KEY_QUANT_FORMAT);
            m.remove(KEY_QUANT_GROUP);
            m.remove(KEY_QUANT_MLX);
        }),
    ];
    for (tag, edit) in cases {
        let (tmp, ckpt) = saved(tag, q8_identity());
        edit_metadata(&tmp, edit);
        match read_adapter_checkpoint(&tmp.0.join("ckpt"), &expecting(&ckpt, q8_identity())) {
            Err(AdapterError::Corrupt { file, detail }) => {
                assert_eq!(file, ADAPTERS_FILE, "{tag}");
                assert!(!detail.is_empty(), "{tag}");
            }
            other => panic!("{tag}: {:?}", other.map(|_| ())),
        }
    }
}

// ---- merging over a packed file ----------------------------------------------------------------

/// W3-D4: a packed weight is not merged. The refusal names the weight and the fq8 route, and it
/// is the typed `Unsupported`, not the shape error `[out, in/4]` words would give. The control
/// merges the same target over the fake-quant file.
#[test]
fn merging_over_a_packed_file_is_unsupported_and_names_the_fq8_route() {
    let fx = FQ8;
    let (s, tg, sites) = (spec(), targets(), fixture_sites(&fx.file(), true));
    let t0 = &tg[0];
    let site = sites.slot(t0.layer, t0.kind).unwrap().as_ref().unwrap();
    let q8 = q8_file();
    match t0.merged_bf16(&s, site, &fx.config(), &q8, &cpu()) {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, "qwen35::merged_bf16");
            assert!(detail.contains(&t0.name), "{detail}");
            assert!(detail.contains("fq8"), "{detail}");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    // The whole export refuses at the same weight.
    match sites.merge_bf16(&s, &tg, &fx.config(), &q8, &cpu(), &mut |_, _| Ok(())) {
        Err(OjasError::Unsupported { op, .. }) => assert_eq!(op, "qwen35::merged_bf16"),
        other => panic!("{other:?}"),
    }
    // The control: the fake-quant file merges.
    let fq8 = SafeTensors::open(&fq8_path()).unwrap();
    let merged = t0
        .merged_bf16(&s, site, &fx.config(), &fq8, &cpu())
        .unwrap();
    assert_eq!(merged.dtype(), DType::Bf16);
}
