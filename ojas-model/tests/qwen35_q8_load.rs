//! The MLX q8 loader on the tiny packed base (`tests/fixtures/qwen35_tiny_q8`, mlx 0.31.2
//! `mx.quantize(..., group_size=64, bits=8, stream=mx.gpu)` in mlx_lm's layout) and the
//! `qwen35_tiny_fq8` base that holds `mx.dequantize` of the same call.
//!
//! `ojas_io::read_affine_quantized` reads each of the 13 wrapped linears from the manifest's
//! `[out, in]`, `ojas_core::dequantize_affine_bf16` of the result equals the fq8 weight bit for
//! bit, the config block gives q8/g64, and the parse is charged exactly its bytes.

use std::path::PathBuf;

use ojas_core::{dequantize_affine_bf16, AffineQuantized, AffineSpec, Budget, OjasError};
use ojas_io::{
    affine_spec_from_config, parse_json, read_affine_quantized, AffineLoadError, JsonValue,
    SafeTensors,
};

fn fixture(dir: &str, file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
        .join(file)
}

fn open(dir: &str) -> SafeTensors<'static> {
    let path = fixture(dir, "model.safetensors");
    SafeTensors::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn json(dir: &str, file: &str) -> JsonValue {
    let path = fixture(dir, file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse_json(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// `(prefix, out, in)` of every wrapped linear, from the manifest.
fn manifest_linears() -> Vec<(String, usize, usize)> {
    let manifest = json("qwen35_tiny_q8", "manifest.json");
    manifest
        .get_array("quantised")
        .unwrap()
        .iter()
        .map(|entry| {
            let name = entry.get_str("name").unwrap();
            let shape = entry.get_array("shape").unwrap();
            let [out, inp] = shape else {
                panic!("{name}: shape is not [out, in]")
            };
            let (out, inp) = (
                out.as_u64().unwrap() as usize,
                inp.as_u64().unwrap() as usize,
            );
            assert_eq!(entry.get_u64("in_features").unwrap() as usize, inp);
            (
                name.strip_suffix(".weight")
                    .unwrap_or_else(|| panic!("{name}: not a .weight"))
                    .to_string(),
                out,
                inp,
            )
        })
        .collect()
}

fn read(
    q8: &SafeTensors<'_>,
    prefix: &str,
    out: usize,
    inp: usize,
    budget: &Budget,
) -> Result<AffineQuantized, AffineLoadError> {
    read_affine_quantized(q8, prefix, AffineSpec::q8(), out, inp, budget)
}

#[test]
fn every_wrapped_linear_parses_and_dequantises_to_fq8() {
    let (q8, fq8) = (open("qwen35_tiny_q8"), open("qwen35_tiny_fq8"));
    let budget = Budget::new(1 << 30);
    let linears = manifest_linears();
    assert_eq!(linears.len(), 13, "the tiny base has 13 wrapped linears");
    let mut widths = std::collections::BTreeSet::new();
    for (prefix, out, inp) in &linears {
        widths.insert(*inp);
        let q = read(&q8, prefix, *out, *inp, &budget).unwrap_or_else(|e| panic!("{prefix}: {e}"));
        assert_eq!(
            (q.rows(), q.cols(), q.spec()),
            (*out, *inp, AffineSpec::q8())
        );
        let (shape, want) = fq8
            .read_bf16_bits(&format!("{prefix}.weight"))
            .unwrap_or_else(|e| panic!("{prefix}: {e}"));
        assert_eq!(shape, [*out as u64, *inp as u64], "{prefix}");
        let got = dequantize_affine_bf16(&q, &budget).unwrap();
        assert_eq!(got.bf16_slice().unwrap(), &want[..], "{prefix} against fq8");
    }
    assert_eq!(widths.into_iter().collect::<Vec<_>>(), [64, 128, 256, 512]);
}

#[test]
fn config_block_gives_q8_g64() {
    let config = json("qwen35_tiny_q8", "config.json");
    assert_eq!(
        affine_spec_from_config(&config).unwrap(),
        Some(AffineSpec::q8())
    );
    // The dequantised base's config carries no quantization block.
    assert_eq!(
        affine_spec_from_config(&json("qwen35_tiny_fq8", "config.json")).unwrap(),
        None
    );
    let manifest = json("qwen35_tiny_q8", "manifest.json");
    assert_eq!(manifest.get_u64("bits").unwrap(), 8);
    assert_eq!(manifest.get_u64("group_size").unwrap(), 64);
}

#[test]
fn parse_charges_exactly_nbytes_and_a_short_budget_refuses_with_nothing_charged() {
    let q8 = open("qwen35_tiny_q8");
    let (prefix, out, inp) = manifest_linears()
        .into_iter()
        .find(|(p, _, _)| p.ends_with("self_attn.q_proj"))
        .expect("q_proj is wrapped");
    let probe = Budget::new(1 << 30);
    let nbytes = read(&q8, &prefix, out, inp, &probe)
        .unwrap()
        .nbytes()
        .unwrap() as u64;
    assert_eq!(
        nbytes,
        (out * (inp / 4) * 4 + 2 * out * (inp / 64) * 2) as u64
    );

    let short = Budget::new(nbytes - 1);
    match read(&q8, &prefix, out, inp, &short) {
        Err(AffineLoadError::Core(OjasError::CapacityExceeded { .. })) => {}
        other => panic!("{other:?}"),
    }
    assert_eq!(short.live_bytes().unwrap(), 0);
    assert_eq!(
        short.peak_bytes(),
        0,
        "the room check came after an allocation"
    );

    let exact = Budget::new(nbytes);
    let q = read(&q8, &prefix, out, inp, &exact).unwrap();
    assert_eq!(exact.live_bytes().unwrap(), nbytes);
    assert_eq!(q.nbytes().unwrap() as u64, nbytes);
    drop(q);
    assert_eq!(exact.live_bytes().unwrap(), 0);
}

/// The real file with the wrong expectation is refused by shape, naming the tensor, and nothing
/// is charged: the width the fixture was written at is not the one asked for.
#[test]
fn the_real_file_read_with_the_wrong_width_or_shape_is_refused() {
    let q8 = open("qwen35_tiny_q8");
    let budget = Budget::new(1 << 30);
    let (prefix, out, inp) = manifest_linears().remove(0);
    match read_affine_quantized(&q8, &prefix, AffineSpec::q4(), out, inp, &budget) {
        Err(AffineLoadError::Layout { name, detail, .. }) => {
            assert_eq!(name, format!("{prefix}.weight"));
            assert!(detail.contains("imply 8-bit codes"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
    match read(&q8, &prefix, out + 1, inp, &budget) {
        Err(AffineLoadError::Layout { .. }) => {}
        other => panic!("{other:?}"),
    }
    match read(&q8, "model.layers.0.no_such_linear", out, inp, &budget) {
        Err(AffineLoadError::Missing { name }) => {
            assert_eq!(name, "model.layers.0.no_such_linear.weight")
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(budget.live_bytes().unwrap(), 0);
}
