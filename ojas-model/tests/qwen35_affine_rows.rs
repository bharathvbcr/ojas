//! Row blocks of a quantised matrix: `split_rows_affine` (the fused `in_proj_qkv` to `wq`, `wk`,
//! `wv`) and `deinterleave_affine` (`q_proj` to its query and gate rows), on the tiny packed q8
//! base. Slicing the quantised parts and then dequantising equals dequantising and then slicing,
//! because a quantisation group runs along a row and a row slice never cuts one.
//!
//! The query and gate parts are compared one at a time against the rows they must be, so a swap
//! of the two fails; the CPU `load_hf` of the dequantised base (`qwen35_tiny_fq8`) is the
//! reference for what the loader would read.

mod qwen35_common;

use std::path::PathBuf;

use ojas_core::{
    dequantize_affine_bf16, quantize_affine, AffineQuantized, AffineSpec, Budget, DType, OjasError,
    Tensor,
};
use ojas_io::{read_affine_quantized, SafeTensors};
use ojas_model::qwen35::{
    deinterleave_affine, load_hf, split_rows_affine, MixerParams, Qwen35Params,
};
use qwen35_common::{budget, spec, PREFIX};

const QKV: &str = "model.layers.0.linear_attn.in_proj_qkv";
const Q_PROJ: &str = "model.layers.1.self_attn.q_proj";

fn open(dir: &str) -> SafeTensors<'static> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
        .join("model.safetensors");
    SafeTensors::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn q8_linear(prefix: &str, rows: usize, cols: usize) -> AffineQuantized {
    read_affine_quantized(
        &open("qwen35_tiny_q8"),
        prefix,
        AffineSpec::q8(),
        rows,
        cols,
        &Budget::new(1 << 30),
    )
    .unwrap_or_else(|e| panic!("{prefix}: {e}"))
}

/// The same weight at `bits`, quantised on the host from the fq8 base's bf16 values.
fn requantised(prefix: &str, rows: usize, cols: usize, spec: AffineSpec) -> AffineQuantized {
    let (shape, bits) = open("qwen35_tiny_fq8")
        .read_bf16_bits(&format!("{prefix}.weight"))
        .unwrap();
    assert_eq!(shape, [rows as u64, cols as u64]);
    let b = Budget::new(1 << 30);
    quantize_affine(
        &Tensor::from_bf16_bits(&bits, &[rows, cols], &b).unwrap(),
        spec,
        &b,
    )
    .unwrap()
}

fn dequantised(q: &AffineQuantized) -> Vec<u16> {
    dequantize_affine_bf16(q, &Budget::new(1 << 30))
        .unwrap()
        .bf16_slice()
        .unwrap()
        .to_vec()
}

/// Rows `first..first + n` of a row-major `[_, cols]` matrix.
fn rows_of(v: &[u16], cols: usize, first: usize, n: usize) -> Vec<u16> {
    v[first * cols..(first + n) * cols].to_vec()
}

fn qkv_counts() -> [usize; 3] {
    let chans = spec().gdn_qkv_channels();
    [chans[0], chans[1], chans[2]]
}

/// Slice-then-dequantise equals dequantise-then-slice, block by block.
fn check_split(q: &AffineQuantized, counts: &[usize], tag: &str) {
    let full = dequantised(q);
    let parts = split_rows_affine(q, counts, &Budget::new(1 << 30)).unwrap();
    assert_eq!(parts.len(), counts.len());
    let mut at = 0;
    for (i, (part, &n)) in parts.iter().zip(counts).enumerate() {
        assert_eq!(
            (part.rows(), part.cols(), part.spec()),
            (n, q.cols(), q.spec())
        );
        assert_eq!(
            dequantised(part),
            rows_of(&full, q.cols(), at, n),
            "{tag}: block {i}"
        );
        at += n;
    }
    assert_eq!(at, q.rows());
}

/// Per head, `d` query rows then `d` gate rows.
fn check_deinterleave(q: &AffineQuantized, heads: usize, d: usize, tag: &str) {
    let full = dequantised(q);
    let cols = q.cols();
    let (query, gate) = deinterleave_affine(q, heads, d, &Budget::new(1 << 30)).unwrap();
    let rows_for = |offset: usize| -> Vec<u16> {
        (0..heads)
            .flat_map(|h| rows_of(&full, cols, h * 2 * d + offset, d))
            .collect()
    };
    let (want_query, want_gate) = (rows_for(0), rows_for(d));
    assert_eq!((query.rows(), gate.rows()), (heads * d, heads * d), "{tag}");
    let (got_query, got_gate) = (dequantised(&query), dequantised(&gate));
    assert_eq!(got_query, want_query, "{tag}: query rows");
    assert_eq!(got_gate, want_gate, "{tag}: gate rows");
    // The comparison is per part: a swap of the two would pass a comparison of their union.
    assert_ne!(
        got_query, want_gate,
        "{tag}: query and gate are indistinguishable"
    );
    assert_ne!(
        got_gate, want_query,
        "{tag}: query and gate are indistinguishable"
    );
}

#[test]
fn gdn_qkv_split_then_dequant_equals_dequant_then_split() {
    let q = q8_linear(QKV, 768, 64);
    assert_eq!(qkv_counts().iter().sum::<usize>(), 768);
    check_split(&q, &qkv_counts(), "q8 in_proj_qkv");
    // An uneven split too: the rows of a block need not be a power of two.
    check_split(&q, &[1, 255, 300, 212], "q8 uneven");
    check_split(&q, &[768], "q8 whole");
}

#[test]
fn q_proj_deinterleave_then_dequant_equals_dequant_then_deinterleave() {
    let q = q8_linear(Q_PROJ, 1024, 64);
    check_deinterleave(&q, 2, 256, "q8 q_proj");
}

#[test]
fn the_same_equalities_hold_for_a_q4_requantisation_of_the_fq8_base() {
    let q4 = AffineSpec::q4();
    let qkv = requantised(QKV, 768, 64, q4);
    assert_eq!(qkv.spec(), q4);
    check_split(&qkv, &qkv_counts(), "q4 in_proj_qkv");
    let q_proj = requantised(Q_PROJ, 1024, 64, q4);
    check_deinterleave(&q_proj, 2, 256, "q4 q_proj");
    // And q8 requantised, whose codes differ from mlx's by nothing the equality depends on.
    check_split(
        &requantised(QKV, 768, 64, AffineSpec::q8()),
        &qkv_counts(),
        "q8 requantised",
    );
}

/// What `load_hf` reads from the dequantised base is what the quantised row blocks dequantise to:
/// the CPU reference for the loader's per-row-block requirement.
#[test]
fn row_blocks_dequantise_to_the_parts_load_hf_reads_from_fq8() {
    let fq8 = open("qwen35_tiny_fq8");
    let loaded: Qwen35Params<Tensor> =
        load_hf(&spec(), &fq8, PREFIX, DType::Bf16, DType::Bf16, &budget()).unwrap();
    let b = Budget::new(1 << 30);

    let MixerParams::GatedDeltaNet(gdn) = &loaded.layers[0].mixer else {
        panic!("layer 0 is a gated delta net")
    };
    let parts = split_rows_affine(&q8_linear(QKV, 768, 64), &qkv_counts(), &b).unwrap();
    for (name, part, want) in [
        ("wq", &parts[0], &gdn.wq),
        ("wk", &parts[1], &gdn.wk),
        ("wv", &parts[2], &gdn.wv),
    ] {
        assert_eq!(
            &dequantised(part)[..],
            want.bf16_slice().unwrap(),
            "gdn {name}"
        );
    }

    let MixerParams::Attention(attn) = &loaded.layers[1].mixer else {
        panic!("layer 1 is attention")
    };
    let (query, gate) = deinterleave_affine(&q8_linear(Q_PROJ, 1024, 64), 2, 256, &b).unwrap();
    assert_eq!(
        &dequantised(&query)[..],
        attn.wq.bf16_slice().unwrap(),
        "attention wq"
    );
    assert_eq!(
        &dequantised(&gate)[..],
        attn.wgate.bf16_slice().unwrap(),
        "attention wgate"
    );
}

// --- refusals ----------------------------------------------------------------------------------

fn shape_err<T: std::fmt::Debug>(r: Result<T, OjasError>, op: &str) {
    match r {
        Err(OjasError::Shape { op: got, .. }) => assert_eq!(got, op),
        other => panic!("expected Shape from {op}, got {other:?}"),
    }
}

fn range_err<T: std::fmt::Debug>(r: Result<T, OjasError>) {
    assert!(
        matches!(r, Err(OjasError::OutOfRange { .. })),
        "expected OutOfRange, got {r:?}"
    );
}

#[test]
fn split_refuses_counts_that_do_not_tile_the_rows() {
    let q = q8_linear(QKV, 768, 64);
    let b = Budget::new(1 << 30);
    for counts in [
        &[256usize, 256, 255][..],
        &[256, 256, 257],
        &[],
        &[768, 1],
        &[0, 768],
    ] {
        shape_err(split_rows_affine(&q, counts, &b), "split_rows_affine");
    }
    // A zero count among otherwise exact blocks.
    shape_err(
        split_rows_affine(&q, &[256, 0, 256, 256], &b),
        "split_rows_affine",
    );
    range_err(split_rows_affine(&q, &[usize::MAX, 2], &b));
    range_err(split_rows_affine(
        &q,
        &[usize::MAX / 2 + 1, usize::MAX / 2 + 1],
        &b,
    ));
    assert_eq!(b.live_bytes().unwrap(), 0);
    assert_eq!(b.peak_bytes(), 0, "a refusal allocated");
}

#[test]
fn deinterleave_refuses_a_head_geometry_that_is_not_the_rows() {
    let q = q8_linear(Q_PROJ, 1024, 64);
    let b = Budget::new(1 << 30);
    for (heads, d) in [
        (2usize, 255usize),
        (3, 256),
        (1, 256),
        (0, 256),
        (2, 0),
        (4, 128 + 1),
    ] {
        shape_err(deinterleave_affine(&q, heads, d, &b), "deinterleave_affine");
    }
    range_err(deinterleave_affine(&q, usize::MAX / 2 + 1, 2, &b));
    range_err(deinterleave_affine(&q, 2, usize::MAX / 2 + 1, &b));
    assert_eq!(b.live_bytes().unwrap(), 0);
    assert_eq!(b.peak_bytes(), 0, "a refusal allocated");
}

#[test]
fn a_budget_that_cannot_hold_the_outputs_refuses_with_nothing_charged() {
    let q = q8_linear(QKV, 768, 64);
    // 768 rows of 16 words (4 bytes) and 1 group (scale and bias, 2 bytes each).
    let need = (768 * (16 * 4 + 2 * 2)) as u64;
    let short = Budget::new(need - 1);
    assert!(matches!(
        split_rows_affine(&q, &qkv_counts(), &short),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert_eq!(short.live_bytes().unwrap(), 0);
    assert_eq!(
        short.peak_bytes(),
        0,
        "the room check came after an allocation"
    );
    let exact = Budget::new(need);
    let parts = split_rows_affine(&q, &qkv_counts(), &exact).unwrap();
    assert_eq!(exact.live_bytes().unwrap(), need);
    drop(parts);
    assert_eq!(exact.live_bytes().unwrap(), 0);

    let q_proj = q8_linear(Q_PROJ, 1024, 64);
    let need = (1024 * (16 * 4 + 2 * 2)) as u64;
    let short = Budget::new(need - 1);
    assert!(matches!(
        deinterleave_affine(&q_proj, 2, 256, &short),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert_eq!(short.live_bytes().unwrap(), 0);
    assert_eq!(short.peak_bytes(), 0);
    let exact = Budget::new(need);
    let pair = deinterleave_affine(&q_proj, 2, 256, &exact).unwrap();
    assert_eq!(exact.live_bytes().unwrap(), need);
    drop(pair);
    assert_eq!(exact.live_bytes().unwrap(), 0);
}

/// The input is untouched: the slices are copies, so dropping them leaves it readable.
#[test]
fn the_source_is_unchanged_by_slicing() {
    let q = q8_linear(QKV, 768, 64);
    let before = dequantised(&q);
    drop(split_rows_affine(&q, &qkv_counts(), &Budget::new(1 << 30)).unwrap());
    assert_eq!(dequantised(&q), before);
}
