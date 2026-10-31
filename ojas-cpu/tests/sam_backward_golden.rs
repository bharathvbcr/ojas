//! `group_norm_backward` and `interpolate_bilinear_backward` against torch
//! autograd, on the cases of `ojas-oracle/fixtures/sam_bwd/golden.txt` (made
//! by `ojas-oracle/python/sam_backward_fixture.py`): `F.group_norm` at 1, 2 and
//! 8 groups, and `F.interpolate(mode="bilinear")` with both `align_corners`,
//! up and down, a `1 x N` edge, the identity and an exact halving.
//!
//! Each gradient is compared with torch's float32 gradient and with torch's
//! float64 gradient from the same float32 inputs; the measured errors are
//! printed (`cargo test -- --nocapture`). The bounds were fixed before any
//! measurement and are on [`TORCH_F64_TOL`] and [`torch_f32_tol`].
//!
//! The fixture is not optional: a missing file fails this test, with the
//! command that makes it, rather than skipping it.

use std::collections::HashMap;

use ojas_core::{Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../ojas-oracle/fixtures/sam_bwd/golden.txt"
);

fn exact() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 30), 2)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// `max |got - want| / max |want|`, or the absolute error when `want` is 0.
fn normalized(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - w).abs()));
    if scale == 0.0 {
        err
    } else {
        err / scale
    }
}

struct Golden {
    op: String,
    name: String,
    eps: f32,
    param: usize,
    tensors: HashMap<String, (Vec<usize>, Vec<f64>)>,
}

impl Golden {
    fn shape(&self, key: &str) -> &[usize] {
        &self.tensors[key].0
    }

    /// The float32 tensor `key` as a tensor.
    fn tensor(&self, key: &str) -> Tensor {
        let (shape, data) = &self.tensors[key];
        let f: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        t(&f, shape)
    }

    fn values(&self, key: &str) -> &[f64] {
        &self.tensors[key].1
    }
}

fn parse_golden() -> Vec<Golden> {
    let text = std::fs::read_to_string(FIXTURE).unwrap_or_else(|e| {
        panic!(
            "the torch golden {FIXTURE} is missing ({e}); make it with: \
             /Users/bharath/.venvs/torch/bin/python -I \
             ojas-oracle/python/sam_backward_fixture.py \
             --out ojas-oracle/fixtures/sam_bwd/golden.txt"
        )
    });
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("ojas-sam-bwd-golden-v1"));
    let torch = lines.next().expect("torch version line");
    assert!(torch.starts_with("torch "), "{torch}");
    let mut cases = Vec::new();
    let mut open: Option<Golden> = None;
    for line in lines {
        let mut f = line.split_whitespace();
        match f.next() {
            Some("case") => {
                let (op, name, eps, param) = (
                    f.next().unwrap(),
                    f.next().unwrap(),
                    f.next().unwrap(),
                    f.next().unwrap(),
                );
                let eps = if eps == "-" {
                    0.0
                } else {
                    f32::from_bits(u32::from_str_radix(eps, 16).unwrap())
                };
                open = Some(Golden {
                    op: op.into(),
                    name: name.into(),
                    eps,
                    param: param.parse().unwrap(),
                    tensors: HashMap::new(),
                });
            }
            Some("t") => {
                let (key, dtype, dims) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
                let shape: Vec<usize> = if dims == "-" {
                    vec![]
                } else {
                    dims.split(',').map(|d| d.parse().unwrap()).collect()
                };
                let data: Vec<f64> = f
                    .map(|h| match dtype {
                        "f32" => f64::from(f32::from_bits(u32::from_str_radix(h, 16).unwrap())),
                        "f64" => f64::from_bits(u64::from_str_radix(h, 16).unwrap()),
                        other => panic!("dtype {other}"),
                    })
                    .collect();
                assert_eq!(
                    data.len(),
                    numel(&shape),
                    "{key} of {:?}",
                    open.as_ref().map(|c| &c.name)
                );
                open.as_mut()
                    .expect("tensor outside a case")
                    .tensors
                    .insert(key.into(), (shape, data));
            }
            Some("end") => cases.push(open.take().expect("end outside a case")),
            None => {}
            Some(other) => panic!("unknown line {other}"),
        }
    }
    assert!(open.is_none(), "unterminated case");
    cases
}

/// What one gradient measured against torch: the normalized error against
/// float32 torch and against float64 torch.
struct Measured {
    case: String,
    tensor: &'static str,
    vs_f32: f64,
    vs_f64: f64,
}

/// Held to float64 torch at what an `f32` result of an exact value can be:
/// the backward rounds an `f64` value once (an `f32` ulp is 1.2e-7 of a
/// value, so a normalized error of a few ulp is 1e-6 with room to spare).
const TORCH_F64_TOL: f64 = 1e-6;

/// Against float32 torch the bound is torch's own rounding: its float32
/// reductions and its float32 weights. GroupNorm sums up to a few hundred
/// float32 values (5e-6); bilinear sums at most four taps per output and a
/// few per input (1e-6). `gn_mean_offset` has a group mean of 100 against a
/// spread of 0.5, where torch's float32 statistics lose digits that float64
/// torch and this backend agree on; it is held to 1e-3.
fn torch_f32_tol(case: &Golden) -> f64 {
    match (case.op.as_str(), case.name.as_str()) {
        (_, "gn_mean_offset") => 1e-3,
        ("gn", _) => 5e-6,
        _ => 1e-6,
    }
}

fn compare(m: &mut Vec<Measured>, case: &Golden, tensor: &'static str, got: &Tensor, key: &str) {
    let g = v(got);
    assert_eq!(got.shape(), case.shape(key), "{} {tensor} shape", case.name);
    let vs_f32 = normalized(&g, case.values(key));
    let vs_f64 = normalized(&g, case.values(&format!("{key}_64")));
    m.push(Measured {
        case: case.name.clone(),
        tensor,
        vs_f32,
        vs_f64,
    });
}

#[test]
fn the_sam_backwards_match_torch_autograd_on_the_golden_cases() {
    let be = exact();
    let cases = parse_golden();
    let mut m: Vec<Measured> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for c in &cases {
        *seen.entry(c.op.clone()).or_default() += 1;
        match c.op.as_str() {
            "gn" => {
                let (x, w, b, gy) = (c.tensor("x"), c.tensor("w"), c.tensor("b"), c.tensor("gy"));
                // The parse is right if the forward reproduces torch's output.
                let fwd = be.group_norm_forward(&x, &w, &b, c.param, c.eps).unwrap();
                assert!(
                    normalized(&v(&fwd), c.values("y_64")) < 1e-4,
                    "{} forward",
                    c.name
                );
                let g = be
                    .group_norm_backward(&x, &w, &b, &gy, c.param, c.eps)
                    .unwrap();
                compare(&mut m, c, "dx", &g.input, "dx");
                compare(&mut m, c, "dw", &g.weight, "dw");
                compare(&mut m, c, "db", &g.bias, "db");
            }
            "bil" => {
                let (x, gy) = (c.tensor("x"), c.tensor("gy"));
                let align = match c.param {
                    0 => false,
                    1 => true,
                    other => panic!("{}: align_corners {other}", c.name),
                };
                let &[_, _, oh, ow] = c.shape("gy") else {
                    panic!("{}: gy is not rank 4", c.name)
                };
                let fwd = be
                    .interpolate_bilinear_forward(&x, [oh, ow], align)
                    .unwrap();
                assert!(
                    normalized(&v(&fwd), c.values("y_64")) < 1e-5,
                    "{} forward",
                    c.name
                );
                let dx = be.interpolate_bilinear_backward(&x, &gy, align).unwrap();
                compare(&mut m, c, "dx", &dx, "dx");
            }
            other => panic!("unknown op {other} in the fixture"),
        }
    }
    assert!(
        seen.get("gn").copied().unwrap_or(0) >= 8 && seen.get("bil").copied().unwrap_or(0) >= 12,
        "the fixture lost cases: {seen:?}"
    );
    let worst = |f: fn(&Measured) -> f64| m.iter().map(f).fold(0.0f64, f64::max);
    eprintln!(
        "torch goldens: {} cases, {} gradients; max normalized error {:.2e} vs torch f32, {:.2e} vs torch f64",
        cases.len(),
        m.len(),
        worst(|x| x.vs_f32),
        worst(|x| x.vs_f64)
    );
    for x in &m {
        eprintln!(
            "  {:<22} {:<3} vs f32 {:.2e}  vs f64 {:.2e}",
            x.case, x.tensor, x.vs_f32, x.vs_f64
        );
    }
    for x in &m {
        assert!(
            x.vs_f64 <= TORCH_F64_TOL,
            "{} {}: {:e} against torch float64 (> {TORCH_F64_TOL:e})",
            x.case,
            x.tensor,
            x.vs_f64
        );
        let case = cases.iter().find(|c| c.name == x.case).unwrap();
        assert!(
            x.vs_f32 <= torch_f32_tol(case),
            "{} {}: {:e} against torch float32 (> {:e})",
            x.case,
            x.tensor,
            x.vs_f32,
            torch_f32_tol(case)
        );
    }
}
