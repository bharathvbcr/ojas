//! The host affine quantiser against the tiny packed q8 base.
//!
//! `qwen35_tiny_q8` is `qwen35_tiny` with its wrapped linears replaced by mlx's own
//! `mx.quantize(w, group_size=64, bits=8, stream=mx.gpu)` output in mlx_lm's layout
//! (`<prefix>.weight` U32, `.scales`, `.biases` BF16; `qwen35_fakequant_base.py --packed-out`).
//! `qwen35_tiny_fq8` holds `mx.dequantize` of the same call. This pins, on the real tiny base
//! (in-features 64, 128, 256 and 512):
//!
//! - `ojas_core::quantize_affine` on the bf16 weights reproduces the packed codes, scales and
//!   biases byte for byte;
//! - `AffineQuantized::from_parts` accepts the file's three tensors, and
//!   `ojas_core::dequantize_affine_bf16` reproduces the fq8 weights bit for bit;
//! - every other tensor is a byte copy, and `config.json` carries the quantization block.

use ojas_core::{
    dequantize_affine_bf16, quantize_affine, AffineQuantized, AffineSpec, Budget, Tensor,
};
use ojas_io::SafeTensors;
use std::collections::BTreeSet;
use std::path::PathBuf;

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

fn bf16(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<u16>) {
    let (shape, bits) = st
        .read_bf16_bits(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    (shape.iter().map(|&d| d as usize).collect(), bits)
}

fn u32s(st: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<u32>) {
    let (shape, words) = st.read_u32(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    (shape.iter().map(|&d| d as usize).collect(), words)
}

/// The wrapped linears: every `<prefix>.weight` that has a `<prefix>.scales` in the q8 base.
fn wrapped(q8: &SafeTensors<'_>) -> Vec<String> {
    let names: BTreeSet<String> = q8.names().map(str::to_string).collect();
    names
        .iter()
        .filter_map(|n| n.strip_suffix(".scales"))
        .map(|prefix| format!("{prefix}.weight"))
        .inspect(|w| assert!(names.contains(w), "{w} is missing beside its scales"))
        .collect()
}

#[test]
fn host_quantiser_reproduces_the_packed_q8_base_and_dequantiser_the_fq8_base() {
    let (bf, q8, fq8) = (
        open("qwen35_tiny"),
        open("qwen35_tiny_q8"),
        open("qwen35_tiny_fq8"),
    );
    let budget = Budget::new(1 << 30);
    let linears = wrapped(&q8);
    assert_eq!(
        linears.len(),
        13,
        "the tiny base has 13 wrapped linears: {linears:?}"
    );
    let mut widths = BTreeSet::new();
    for name in &linears {
        let prefix = name.strip_suffix(".weight").unwrap();
        let (shape, w) = bf16(&bf, name);
        widths.insert(shape[1]);
        let weight = Tensor::from_bf16_bits(&w, &shape, &budget).unwrap();
        let quantized = quantize_affine(&weight, AffineSpec::q8(), &budget)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        let (packed_shape, packed) = u32s(&q8, name);
        let (sb_shape, scales) = bf16(&q8, &format!("{prefix}.scales"));
        let (_, biases) = bf16(&q8, &format!("{prefix}.biases"));
        assert_eq!(
            packed_shape,
            [shape[0], shape[1] * 8 / 32],
            "{name} packed shape"
        );
        assert_eq!(sb_shape, [shape[0], shape[1] / 64], "{name} scales shape");
        assert_eq!(
            quantized.packed().u32_slice().unwrap(),
            &packed[..],
            "{name} packed codes"
        );
        assert_eq!(
            quantized.scales().bf16_slice().unwrap(),
            &scales[..],
            "{name} scales"
        );
        assert_eq!(
            quantized.biases().bf16_slice().unwrap(),
            &biases[..],
            "{name} biases"
        );

        // The file's own parts load through the validating constructor and dequantise to fq8.
        let loaded = AffineQuantized::from_parts(
            AffineSpec::q8(),
            shape[0],
            shape[1],
            Tensor::from_u32(&packed, &packed_shape, &budget).unwrap(),
            Tensor::from_bf16_bits(&scales, &sb_shape, &budget).unwrap(),
            Tensor::from_bf16_bits(&biases, &sb_shape, &budget).unwrap(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let (fq_shape, fq) = bf16(&fq8, name);
        assert_eq!(fq_shape, shape);
        let from_file = dequantize_affine_bf16(&loaded, &budget).unwrap();
        assert_eq!(
            from_file.bf16_slice().unwrap(),
            &fq[..],
            "{name} dequantised (file parts)"
        );
        let from_host = dequantize_affine_bf16(&quantized, &budget).unwrap();
        assert_eq!(
            from_host.bf16_slice().unwrap(),
            &fq[..],
            "{name} dequantised (host parts)"
        );
        assert_ne!(
            fq, w,
            "{name}: quantisation changed nothing, the check is vacuous"
        );
    }
    assert_eq!(widths, BTreeSet::from([64, 128, 256, 512]));
}

#[test]
fn every_other_tensor_of_the_q8_base_is_a_byte_copy_and_the_config_says_q8() {
    let (bf, q8) = (open("qwen35_tiny"), open("qwen35_tiny_q8"));
    let linears: BTreeSet<String> = wrapped(&q8).into_iter().collect();
    let mut expected: BTreeSet<String> = bf.names().map(str::to_string).collect();
    for name in &linears {
        let prefix = name.strip_suffix(".weight").unwrap();
        assert!(expected.contains(name), "{name} is not in the bf16 base");
        expected.insert(format!("{prefix}.scales"));
        expected.insert(format!("{prefix}.biases"));
    }
    let got: BTreeSet<String> = q8.names().map(str::to_string).collect();
    assert_eq!(
        got, expected,
        "the q8 base is the bf16 base plus scales and biases"
    );
    for name in bf.names().filter(|n| !linears.contains(*n)) {
        assert_eq!(
            bf.read_bytes(name).unwrap(),
            q8.read_bytes(name).unwrap(),
            "{name} is not a byte copy"
        );
    }
    let config = std::fs::read_to_string(fixture("qwen35_tiny_q8", "config.json")).unwrap();
    assert!(
        config.contains("\"quantization\": {\n  \"group_size\": 64,\n  \"bits\": 8\n }"),
        "config.json lacks mlx_lm's quantization block:\n{config}"
    );
}
