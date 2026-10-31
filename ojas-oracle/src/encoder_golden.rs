//! The encoder backwards against torch autograd, on any backend.
//!
//! `fixtures/encoder_bwd/golden.txt` (written by
//! `python/encoder_backward_fixture.py`) holds, for LayerNorm, channel
//! LayerNorm, the exact GELU, a biased linear, broadcast add and multiply, and
//! softmax, the operands, the incoming gradient and torch's gradients in
//! float32 and in float64. [`check`] runs a backend's forward and backward on
//! every case, through [`Backend::upload`], and measures each gradient
//! against both torch results. The CPU test
//! (`ojas-cpu/tests/encoder_backward.rs`) and the Metal one
//! (`ojas-metal/tests/encoder_backward.rs`) call it with their own
//! [`Tolerances`], so a gradient is held to the same fixture everywhere.

use std::collections::HashMap;

use ojas_core::{Backend, Budget, OjasError, Tensor};

const GOLDEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/encoder_bwd/golden.txt"
));

/// Cases the fixture must hold, so a truncated file fails.
pub const MIN_CASES: usize = 31;

/// One fixture case: its op, name, `eps` (0 where the op has none) and the
/// named tensors (`x`, `gy`, `dx`, `dx_64`, ...), each a shape and its values
/// widened to `f64`.
pub struct Golden {
    pub op: String,
    pub name: String,
    pub eps: f32,
    tensors: HashMap<String, (Vec<usize>, Vec<f64>)>,
}

impl Golden {
    pub fn shape(&self, key: &str) -> &[usize] {
        &self.tensors[key].0
    }

    pub fn values(&self, key: &str) -> &[f64] {
        &self.tensors[key].1
    }

    /// The `f32` tensor `key`, on the host.
    pub fn host(&self, key: &str) -> Tensor {
        let (shape, data) = &self.tensors[key];
        let f: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        Tensor::from_f32(&f, shape, &Budget::new(u64::MAX)).expect("fixture tensor")
    }
}

/// The fixture's cases, in file order.
pub fn parse() -> Vec<Golden> {
    let mut lines = GOLDEN.lines();
    assert_eq!(lines.next(), Some("ojas-encoder-bwd-golden-v1"));
    let torch = lines.next().expect("torch version line");
    assert!(torch.starts_with("torch "), "{torch}");
    let mut cases = Vec::new();
    let mut open: Option<Golden> = None;
    for line in lines {
        let mut f = line.split_whitespace();
        match f.next() {
            Some("case") => {
                let (op, name, eps) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap());
                let eps = if eps == "-" {
                    0.0
                } else {
                    f32::from_bits(u32::from_str_radix(eps, 16).unwrap())
                };
                open = Some(Golden {
                    op: op.into(),
                    name: name.into(),
                    eps,
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
                    shape.iter().product::<usize>(),
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

/// `max |got - want| / max |want|`, or the absolute error when `want` is 0.
pub fn normalized(got: &[f32], want: &[f64]) -> f64 {
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

/// What a backend is held to.
pub struct Tolerances {
    /// Normalized error of every gradient against float64 torch.
    pub vs_f64: f64,
    /// Normalized error against float32 torch, by case name.
    pub vs_f32: fn(&str) -> f64,
}

/// What one gradient measured: the normalized error against float32 torch and
/// against float64 torch.
pub struct Measured {
    pub case: String,
    pub tensor: &'static str,
    pub vs_f32: f64,
    pub vs_f64: f64,
}

/// A run's results.
pub struct Report {
    pub cases: usize,
    pub measured: Vec<Measured>,
}

impl Report {
    pub fn worst_f32(&self) -> f64 {
        self.measured.iter().fold(0.0, |m, x| f64::max(m, x.vs_f32))
    }

    pub fn worst_f64(&self) -> f64 {
        self.measured.iter().fold(0.0, |m, x| f64::max(m, x.vs_f64))
    }

    /// One line per gradient, for `--nocapture`.
    pub fn print(&self, backend: &str) {
        eprintln!(
            "{backend}: torch goldens: {} cases, {} gradients; max normalized error {:.2e} vs torch f32, {:.2e} vs torch f64",
            self.cases,
            self.measured.len(),
            self.worst_f32(),
            self.worst_f64()
        );
        for x in &self.measured {
            eprintln!(
                "  {:<22} {:<3} vs f32 {:.2e}  vs f64 {:.2e}",
                x.case, x.tensor, x.vs_f32, x.vs_f64
            );
        }
    }

    /// Panics naming the first gradient outside `tol`.
    pub fn assert_within(&self, tol: &Tolerances) {
        for x in &self.measured {
            assert!(
                x.vs_f64 <= tol.vs_f64,
                "{} {}: {:e} against torch float64 (> {:e})",
                x.case,
                x.tensor,
                x.vs_f64,
                tol.vs_f64
            );
            // A dimension of 1 has an input gradient of exactly 0 (the row is
            // its own mean); float64 torch gets 0 too (checked above), float32
            // torch leaves rounding noise, which normalized by itself is not a
            // bound.
            if x.case == "ln_dim_one" && x.tensor == "dx" {
                assert_eq!(x.vs_f64, 0.0, "ln_dim_one dx is exactly 0");
                continue;
            }
            let bound = (tol.vs_f32)(&x.case);
            assert!(
                x.vs_f32 <= bound,
                "{} {}: {:e} against torch float32 (> {:e})",
                x.case,
                x.tensor,
                x.vs_f32,
                bound
            );
        }
    }
}

fn host_values(t: &Tensor) -> Vec<f32> {
    t.to_host(&Budget::new(u64::MAX))
        .and_then(|h| h.to_f32_vec())
        .expect("a result reads back")
}

/// Runs every case on `be`. A forward that does not reproduce torch's output
/// (the parse is wrong, or the backend's forward is) panics; a backward error
/// is returned.
pub fn check<B: Backend>(be: &B) -> Result<Report, OjasError> {
    let cases = parse();
    assert!(
        cases.len() >= MIN_CASES,
        "the fixture lost cases: {}",
        cases.len()
    );
    let up = |c: &Golden, key: &str| be.upload(&c.host(key));
    let mut measured: Vec<Measured> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for c in &cases {
        *seen.entry(c.op.clone()).or_default() += 1;
        let mut compare = |tensor: &'static str, got: &Tensor, key: &str| {
            assert_eq!(
                got.shape(),
                c.shape(key),
                "{} {tensor} shape",
                c.name.as_str()
            );
            let g = host_values(got);
            measured.push(Measured {
                case: c.name.clone(),
                tensor,
                vs_f32: normalized(&g, c.values(key)),
                vs_f64: normalized(&g, c.values(&format!("{key}_64"))),
            });
        };
        match c.op.as_str() {
            "ln" | "cln" => {
                let (x, w, b, gy) = (up(c, "x")?, up(c, "w")?, up(c, "b")?, up(c, "gy")?);
                let (fwd, g) = if c.op == "ln" {
                    (
                        be.layer_norm_forward(&x, &w, &b, c.eps),
                        be.layer_norm_backward(&x, &w, &b, &gy, c.eps)?,
                    )
                } else {
                    (
                        be.channel_layer_norm_forward(&x, &w, &b, c.eps),
                        be.channel_layer_norm_backward(&x, &w, &b, &gy, c.eps)?,
                    )
                };
                // The parse is right if the forward reproduces torch's output.
                // A backend whose forward needs `eps > 0` (Metal's, on tessl's
                // LayerNorm) refuses `eps == 0` as `Unsupported`, and that
                // case's backward is still held to torch.
                match fwd {
                    Ok(fwd) => assert!(
                        normalized(&host_values(&fwd), c.values("y_64")) < 1e-4,
                        "{} forward",
                        c.name
                    ),
                    Err(OjasError::Unsupported { .. }) if c.eps <= 0.0 => {}
                    Err(e) => return Err(e),
                }
                compare("dx", &g.input, "dx");
                compare("dw", &g.weight, "dw");
                compare("db", &g.bias, "db");
            }
            "gelu" => {
                let (x, gy) = (up(c, "x")?, up(c, "gy")?);
                let fwd = be.gelu_erf_forward(&x)?;
                assert!(
                    normalized(&host_values(&fwd), c.values("y_64")) < 1e-6,
                    "{} forward",
                    c.name
                );
                compare("dx", &be.gelu_erf_backward(&x, &gy)?, "dx");
            }
            "lin" => {
                let (x, w, b, gy) = (up(c, "x")?, up(c, "w")?, up(c, "b")?, up(c, "gy")?);
                let fwd = be.linear_bias_forward(&x, &w, &b)?;
                assert!(
                    normalized(&host_values(&fwd), c.values("y_64")) < 1e-5,
                    "{} forward",
                    c.name
                );
                let g = be.linear_bias_backward(&x, &w, &b, &gy)?;
                compare("dx", &g.input, "dx");
                compare("dw", &g.weight, "dw");
                compare("db", &g.bias, "db");
            }
            "badd" | "bmul" => {
                let (a, b, gy) = (up(c, "a")?, up(c, "b")?, up(c, "gy")?);
                let (ga, gb) = if c.op == "badd" {
                    be.broadcast_add_backward(&a, &b, &gy)?
                } else {
                    be.broadcast_mul_backward(&a, &b, &gy)?
                };
                compare("da", &ga, "da");
                compare("db", &gb, "db");
            }
            "softmax" => {
                // Torch's own forward output is the backward's operand.
                let (y, gy) = (up(c, "y")?, up(c, "gy")?);
                compare("dx", &be.softmax_backward(&y, &gy)?, "dx");
            }
            other => panic!("unknown op {other} in the fixture"),
        }
    }
    for op in ["ln", "cln", "gelu", "lin", "badd", "bmul", "softmax"] {
        assert!(seen.contains_key(op), "no golden case for {op}");
    }
    Ok(Report {
        cases: cases.len(),
        measured,
    })
}
