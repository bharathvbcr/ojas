//! `read_affine_quantized` and `affine_spec_from_config` against synthetic MLX affine files:
//! every refusal is a typed `AffineLoadError`, nothing stays charged, and the caller's expected
//! `[out, in]` is what separates widths and group sizes whose shapes would otherwise agree.

use ojas_core::{AffineSpec, Budget, OjasError};
use ojas_io::{
    affine_spec_from_config, encode_safetensors, parse_json, read_affine_quantized,
    AffineLoadError, IoError, SafeTensors, StDtype, TensorOut,
};

const P: &str = "m";

/// One tensor of a synthetic file.
struct T {
    name: &'static str,
    dtype: StDtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn u32_bytes(n: usize, fill: u32) -> Vec<u8> {
    (0..n).flat_map(|_| fill.to_le_bytes()).collect()
}

fn bf16_bytes(n: usize, bits: u16) -> Vec<u8> {
    (0..n).flat_map(|_| bits.to_le_bytes()).collect()
}

fn t(name: &'static str, dtype: StDtype, shape: &[u64], data: Vec<u8>) -> T {
    T {
        name,
        dtype,
        shape: shape.to_vec(),
        data,
    }
}

/// A valid linear `m` for `[rows, cols]` at `bits` and group size `group` (the file's own, not
/// necessarily ojas's): codes `0x01010101`-style words, scale 1.0, bias 0.
fn linear(rows: usize, cols: usize, bits: usize, group: usize) -> Vec<T> {
    let words = cols * bits / 32;
    let groups = cols / group;
    vec![
        t(
            "m.weight",
            StDtype::U32,
            &[rows as u64, words as u64],
            u32_bytes(rows * words, 0x0101_0101),
        ),
        t(
            "m.scales",
            StDtype::BF16,
            &[rows as u64, groups as u64],
            bf16_bytes(rows * groups, 0x3F80),
        ),
        t(
            "m.biases",
            StDtype::BF16,
            &[rows as u64, groups as u64],
            bf16_bytes(rows * groups, 0),
        ),
    ]
}

fn q8_linear() -> Vec<T> {
    linear(2, 128, 8, 64)
}

fn file(tensors: &[T]) -> SafeTensors<'static> {
    let items: Vec<TensorOut<'_>> = tensors
        .iter()
        .map(|x| TensorOut {
            name: x.name,
            dtype: x.dtype,
            shape: &x.shape,
            data: &x.data,
        })
        .collect();
    let bytes = encode_safetensors(&items, &[]).expect("encode");
    SafeTensors::from_bytes(bytes).expect("parse")
}

fn without(mut v: Vec<T>, name: &str) -> Vec<T> {
    v.retain(|x| x.name != name);
    v
}

fn replace(mut v: Vec<T>, with: T) -> Vec<T> {
    let at = v.iter().position(|x| x.name == with.name).unwrap();
    v[at] = with;
    v
}

fn read(
    tensors: &[T],
    spec: AffineSpec,
    rows: usize,
    cols: usize,
    budget: &Budget,
) -> Result<ojas_core::AffineQuantized, AffineLoadError> {
    read_affine_quantized(&file(tensors), P, spec, rows, cols, budget)
}

fn big() -> Budget {
    Budget::new(1 << 30)
}

#[track_caller]
fn expect_layout(r: Result<ojas_core::AffineQuantized, AffineLoadError>, name: &str, needle: &str) {
    match r {
        Err(AffineLoadError::Layout {
            name: got, detail, ..
        }) => {
            assert_eq!(got, name);
            assert!(detail.contains(needle), "{detail:?} lacks {needle:?}");
        }
        other => panic!("expected Layout on {name}, got {other:?}"),
    }
}

// --- the happy path ----------------------------------------------------------------------------

#[test]
fn a_valid_q8_linear_reads_exactly_and_charges_exactly_its_bytes() {
    let b = big();
    let q = read(&q8_linear(), AffineSpec::q8(), 2, 128, &b).unwrap();
    assert_eq!((q.rows(), q.cols(), q.spec()), (2, 128, AffineSpec::q8()));
    assert!(q
        .packed()
        .u32_slice()
        .unwrap()
        .iter()
        .all(|&w| w == 0x0101_0101));
    assert!(q
        .scales()
        .bf16_slice()
        .unwrap()
        .iter()
        .all(|&s| s == 0x3F80));
    assert_eq!(b.live_bytes().unwrap(), q.nbytes().unwrap() as u64);
    assert_eq!(q.nbytes().unwrap(), 2 * 32 * 4 + 2 * 2 * 2 * 2);
    drop(q);
    assert_eq!(b.live_bytes().unwrap(), 0);
    // q4 as well.
    let q = read(&linear(3, 128, 4, 64), AffineSpec::q4(), 3, 128, &b).unwrap();
    assert_eq!(q.packed().shape(), [3, 16]);
}

// --- missing, dtype, rank, rows ----------------------------------------------------------------

#[test]
fn a_missing_part_is_missing_and_names_it() {
    for name in ["m.weight", "m.scales", "m.biases"] {
        let b = big();
        let r = read(&without(q8_linear(), name), AffineSpec::q8(), 2, 128, &b);
        match r {
            Err(AffineLoadError::Missing { name: got }) => assert_eq!(got, name),
            other => panic!("{name}: {other:?}"),
        }
        assert_eq!(b.live_bytes().unwrap(), 0);
    }
}

#[test]
fn a_wrong_dtype_is_dtype_naming_both() {
    let b = big();
    for (bad, expected, got) in [
        (
            t("m.weight", StDtype::I32, &[2, 32], u32_bytes(64, 0)),
            StDtype::U32,
            StDtype::I32,
        ),
        (
            t("m.weight", StDtype::F32, &[2, 32], u32_bytes(64, 0)),
            StDtype::U32,
            StDtype::F32,
        ),
        (
            t("m.scales", StDtype::F16, &[2, 2], bf16_bytes(4, 0x3C00)),
            StDtype::BF16,
            StDtype::F16,
        ),
        (
            t("m.scales", StDtype::F32, &[2, 2], u32_bytes(4, 0x3F80_0000)),
            StDtype::BF16,
            StDtype::F32,
        ),
        (
            t("m.biases", StDtype::F16, &[2, 2], bf16_bytes(4, 0)),
            StDtype::BF16,
            StDtype::F16,
        ),
    ] {
        let name = bad.name;
        match read(&replace(q8_linear(), bad), AffineSpec::q8(), 2, 128, &b) {
            Err(AffineLoadError::Dtype {
                name: n,
                expected: e,
                got: g,
            }) => assert_eq!((n.as_str(), e, g), (name, expected, got)),
            other => panic!("{name}: {other:?}"),
        }
    }
    assert_eq!(b.live_bytes().unwrap(), 0);
}

#[test]
fn rank_one_or_three_and_disagreeing_rows_are_layout() {
    let b = big();
    expect_layout(
        read(
            &replace(
                q8_linear(),
                t("m.weight", StDtype::U32, &[64], u32_bytes(64, 0)),
            ),
            AffineSpec::q8(),
            2,
            128,
            &b,
        ),
        "m.weight",
        "imply",
    );
    expect_layout(
        read(
            &replace(
                q8_linear(),
                t("m.scales", StDtype::BF16, &[2, 2, 1], bf16_bytes(4, 0x3F80)),
            ),
            AffineSpec::q8(),
            2,
            128,
            &b,
        ),
        "m.scales",
        "group size",
    );
    // Rows that disagree between the parts: each is checked against the caller's `rows`.
    expect_layout(
        read(
            &replace(
                q8_linear(),
                t("m.scales", StDtype::BF16, &[3, 2], bf16_bytes(6, 0x3F80)),
            ),
            AffineSpec::q8(),
            2,
            128,
            &b,
        ),
        "m.scales",
        "group size",
    );
    expect_layout(
        read(
            &replace(
                q8_linear(),
                t("m.biases", StDtype::BF16, &[1, 2], bf16_bytes(2, 0)),
            ),
            AffineSpec::q8(),
            2,
            128,
            &b,
        ),
        "m.biases",
        "group size",
    );
    assert_eq!(b.live_bytes().unwrap(), 0);
}

// --- width and group size ----------------------------------------------------------------------

#[test]
fn a_q4_file_read_as_q8_is_refused_naming_four_bits() {
    let b = big();
    let q4 = linear(2, 128, 4, 64);
    expect_layout(
        read(&q4, AffineSpec::q8(), 2, 128, &b),
        "m.weight",
        "imply 4-bit codes",
    );
    // And read as what it is, it loads.
    read(&q4, AffineSpec::q4(), 2, 128, &b).unwrap();
    // A q8 file read as q4 is the same refusal the other way.
    expect_layout(
        read(&q8_linear(), AffineSpec::q4(), 2, 128, &b),
        "m.weight",
        "imply 8-bit codes",
    );
}

/// 4-bit codes at group size 128 have the same words-to-groups ratio (16) as 8-bit codes at group
/// size 64, so the triple alone reads as q8 with half the columns. With the true `[out, in]` the
/// stored shapes do not fit and it is refused; with a wrong `in` it is not, which is why the
/// caller must supply the real one.
#[test]
fn a_q4_group_128_file_read_as_q8_group_64_with_the_true_shape_is_refused() {
    let b = big();
    let (rows, true_cols) = (2usize, 256usize);
    let q4_g128 = linear(rows, true_cols, 4, 128);
    // Stored: weight [2, 32], scales [2, 2]. As q8/g64 over the true [2, 256]: weight [2, 64],
    // scales [2, 4].
    expect_layout(
        read(&q4_g128, AffineSpec::q8(), rows, true_cols, &b),
        "m.weight",
        "imply 4-bit codes",
    );
    // The wrong `in` (half of it) is the ambiguity: this read succeeds, as q8 over [2, 128].
    let misread = read(&q4_g128, AffineSpec::q8(), rows, true_cols / 2, &b);
    assert!(
        misread.is_ok(),
        "the shapes alone cannot tell q4/g128 from q8/g64: {misread:?}"
    );
    assert_eq!(
        b.live_bytes().unwrap(),
        misread.unwrap().nbytes().unwrap() as u64
    );
}

#[test]
fn a_group_32_file_read_as_group_64_is_refused_naming_the_group() {
    let b = big();
    // q8 at group 32: the same words per row as q8 at group 64, twice the groups.
    let g32 = linear(2, 128, 8, 32);
    expect_layout(
        read(&g32, AffineSpec::q8(), 2, 128, &b),
        "m.scales",
        "imply group size 32",
    );
}

#[test]
fn cols_not_a_multiple_of_64_and_empty_matrices_are_core_shape() {
    let b = big();
    for (rows, cols) in [(2usize, 100usize), (2, 63), (0, 128), (2, 0)] {
        match read(&q8_linear(), AffineSpec::q8(), rows, cols, &b) {
            Err(AffineLoadError::Core(OjasError::Shape { op, .. })) => {
                assert_eq!(op, "read_affine_quantized")
            }
            other => panic!("[{rows}, {cols}]: {other:?}"),
        }
    }
    assert_eq!(b.live_bytes().unwrap(), 0);
}

#[test]
fn size_overflow_is_out_of_range_without_allocating() {
    let b = big();
    for (rows, cols) in [
        (usize::MAX, 64usize),
        (usize::MAX / 2 + 1, 128),
        (2, usize::MAX - 63),
    ] {
        match read(&q8_linear(), AffineSpec::q8(), rows, cols, &b) {
            Err(AffineLoadError::Core(OjasError::OutOfRange { .. })) => {}
            other => panic!("[{rows}, {cols}]: {other:?}"),
        }
    }
    assert_eq!(b.live_bytes().unwrap(), 0);
    assert_eq!(b.peak_bytes(), 0, "a refusal allocated");
}

// --- non-finite parameters ---------------------------------------------------------------------

#[test]
fn nan_or_inf_scales_and_inf_biases_are_non_finite() {
    for (name, bad) in [
        ("m.scales", 0x7FC0u16),
        ("m.scales", 0x7F80),
        ("m.scales", 0xFF80),
        ("m.biases", 0x7F80),
        ("m.biases", 0xFF80),
        ("m.biases", 0x7FC0),
    ] {
        let b = big();
        let mut data = bf16_bytes(4, if name == "m.scales" { 0x3F80 } else { 0 });
        data[2..4].copy_from_slice(&bad.to_le_bytes());
        let tensors = replace(q8_linear(), t(name, StDtype::BF16, &[2, 2], data));
        match read(&tensors, AffineSpec::q8(), 2, 128, &b) {
            Err(AffineLoadError::Core(OjasError::NonFinite { .. })) => {}
            other => panic!("{name} {bad:#06x}: {other:?}"),
        }
        assert_eq!(
            b.live_bytes().unwrap(),
            0,
            "{name} {bad:#06x} left a charge"
        );
    }
}

// --- the budget --------------------------------------------------------------------------------

#[test]
fn a_short_budget_refuses_with_nothing_charged_and_the_exact_one_loads() {
    let need = (2 * 32 * 4 + 2 * 2 * 2 * 2) as u64;
    let short = Budget::new(need - 1);
    match read(&q8_linear(), AffineSpec::q8(), 2, 128, &short) {
        Err(AffineLoadError::Core(OjasError::CapacityExceeded { .. })) => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(short.live_bytes().unwrap(), 0);
    assert_eq!(
        short.peak_bytes(),
        0,
        "the room check came after an allocation"
    );
    let exact = Budget::new(need);
    let q = read(&q8_linear(), AffineSpec::q8(), 2, 128, &exact).unwrap();
    assert_eq!(exact.live_bytes().unwrap(), need);
    drop(q);
    assert_eq!(exact.live_bytes().unwrap(), 0);
}

// --- the config block --------------------------------------------------------------------------

fn spec_of(json: &str) -> Result<Option<AffineSpec>, AffineLoadError> {
    affine_spec_from_config(&parse_json(json).expect("test JSON"))
}

#[track_caller]
fn expect_config(json: &str, needle: &str) {
    match spec_of(json) {
        Err(AffineLoadError::Config { detail }) => {
            assert!(
                detail.contains(needle),
                "{detail:?} lacks {needle:?} for {json}"
            )
        }
        other => panic!("{json}: expected Config, got {other:?}"),
    }
}

#[test]
fn the_config_block_gives_the_spec_or_none() {
    assert_eq!(spec_of("{}").unwrap(), None);
    assert_eq!(spec_of(r#"{"hidden_size": 64}"#).unwrap(), None);
    assert_eq!(
        spec_of(r#"{"quantization": {"group_size": 64, "bits": 8}}"#).unwrap(),
        Some(AffineSpec::q8())
    );
    assert_eq!(
        spec_of(r#"{"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}"#).unwrap(),
        Some(AffineSpec::q4())
    );
    // quantization_config alone, and both when they agree.
    assert_eq!(
        spec_of(r#"{"quantization_config": {"group_size": 64, "bits": 8}}"#).unwrap(),
        Some(AffineSpec::q8())
    );
    // Key order does not matter: the blocks are compared by what they mean.
    assert_eq!(
        spec_of(
            r#"{"quantization": {"group_size": 64, "bits": 8},
                "quantization_config": {"bits": 8, "group_size": 64, "mode": "affine"}}"#
        )
        .unwrap(),
        Some(AffineSpec::q8())
    );
}

#[test]
fn unsupported_widths_and_group_sizes_are_out_of_range() {
    for (bits, group) in [
        (3, 64),
        (2, 64),
        (16, 64),
        (8, 32),
        (8, 128),
        (4, 128),
        (8, 0),
    ] {
        let json = format!(r#"{{"quantization": {{"group_size": {group}, "bits": {bits}}}}}"#);
        match spec_of(&json) {
            Err(AffineLoadError::Core(OjasError::OutOfRange { .. })) => {}
            other => panic!("{json}: {other:?}"),
        }
    }
}

#[test]
fn a_malformed_config_block_is_config() {
    expect_config(
        r#"{"quantization": {"group_size": 64, "bits": 8, "mode": "mxfp4"}}"#,
        "mxfp4",
    );
    expect_config(
        r#"{"quantization": {"group_size": 64, "bits": 8, "mode": 8}}"#,
        "mode",
    );
    expect_config(
        r#"{"quantization": {"group_size": 64, "bits": 8, "extra": 1}}"#,
        "unknown key",
    );
    // A per-layer override, as mlx_lm writes for a model whose layers differ.
    expect_config(
        r#"{"quantization": {"group_size": 64, "bits": 8,
            "model.layers.0.mlp.gate": {"group_size": 32, "bits": 4}}}"#,
        "model.layers.0.mlp.gate",
    );
    expect_config(
        r#"{"quantization": {"group_size": 64, "bits": 8},
            "quantization_config": {"group_size": 64, "bits": 4}}"#,
        "differ",
    );
    expect_config(r#"{"quantization": {"group_size": 64}}"#, "bits");
    expect_config(r#"{"quantization": {"bits": 8}}"#, "group_size");
    expect_config(
        r#"{"quantization": {"group_size": 64, "bits": 8.0}}"#,
        "bits",
    );
    expect_config(
        r#"{"quantization": {"group_size": "64", "bits": 8}}"#,
        "group_size",
    );
    expect_config(
        r#"{"quantization": {"group_size": 64, "bits": -8}}"#,
        "bits",
    );
    expect_config(r#"{"quantization": [8, 64]}"#, "object");
    expect_config(r#"[1, 2]"#, "object");
}

// --- the error type ----------------------------------------------------------------------------

#[test]
fn io_errors_keep_their_source_and_display() {
    let e = AffineLoadError::from(IoError::new("short read"));
    assert!(matches!(e, AffineLoadError::Io(_)));
    assert!(e.to_string().contains("short read"));
    assert!(std::error::Error::source(&e).is_some());
    let c = AffineLoadError::from(OjasError::NonFinite { op: "x" });
    assert!(std::error::Error::source(&c).is_some());
    assert!(std::error::Error::source(&AffineLoadError::Config { detail: "d".into() }).is_none());
}
