use ojas_core::{Backend, Budget, OjasError, Tensor, RMS_NORM_EPS};
use ojas_cpu::CpuBackend;
use ojas_oracle::{parse_rms_norm, rms_norm_fixture};

#[test]
fn rms_norm_fixture_matches_cpu_reference() {
    let fixture = rms_norm_fixture().unwrap();
    assert_eq!(fixture.eps, 1e-6);
    assert_eq!(fixture.input_shape, vec![4]);
    assert_eq!(fixture.expected.len(), 4);

    let cpu = CpuBackend::new(Budget::new(1 << 20));
    let input: Vec<f32> = fixture.input.iter().copied().map(|v| v as f32).collect();
    let weight: Vec<f32> = fixture.weight.iter().copied().map(|v| v as f32).collect();
    let x = Tensor::from_f32(&input, &fixture.input_shape, cpu.budget()).unwrap();
    let w = Tensor::from_f32(&weight, &fixture.weight_shape, cpu.budget()).unwrap();
    let y = cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap();
    for (got, expect) in y.to_f32_vec().unwrap().iter().zip(&fixture.expected) {
        assert!(
            (f64::from(*got) - expect).abs() < 1e-5,
            "{got} vs {expect}"
        );
    }
}

#[test]
fn malformed_fixture_is_an_error() {
    match parse_rms_norm("{") {
        Err(_) => {}
        Ok(value) => panic!("truncated fixture parsed: {value:?}"),
    }
    match parse_rms_norm("") {
        Err(_) => {}
        Ok(_) => panic!("empty fixture parsed"),
    }
}

fn fixture_with(input: &str) -> String {
    format!(
        r#"{{"format": "ojas-oracle-fixture-v1", "op": "rms_norm", "eps": 1e-6,
        "input_shape": [2], "input": [{input}], "weight_shape": [2], "weight": [1.0, 1.0],
        "expected_shape": [2], "expected": [1.0, 1.0]}}"#
    )
}

#[test]
fn overflowing_fixture_numbers_are_errors_not_infinities() {
    assert!(parse_rms_norm(&fixture_with("0.5, 2.0")).is_ok());
    for bad in ["1e999, 2.0", "0.5, -1e400"] {
        match parse_rms_norm(&fixture_with(bad)) {
            Err(OjasError::OutOfRange { .. }) => {}
            other => panic!("{bad}: expected OutOfRange, got {other:?}"),
        }
    }
}

#[test]
fn duplicate_keys_and_deep_nesting_are_errors() {
    let dup = fixture_with("0.5, 2.0").replacen(r#""eps": 1e-6"#, r#""eps": 1e-6, "eps": 1e-5"#, 1);
    match parse_rms_norm(&dup) {
        Err(OjasError::OutOfRange { .. }) => {}
        other => panic!("duplicate eps: expected OutOfRange, got {other:?}"),
    }
    let deep = format!("{}{}", "[".repeat(1 << 20), "]".repeat(1 << 20));
    match parse_rms_norm(&deep) {
        Err(OjasError::OutOfRange { .. }) => {}
        other => panic!("deep nesting: expected OutOfRange, got {other:?}"),
    }
}
