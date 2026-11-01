//! Gemma Embedding ops golden parity verification against PyTorch.
//!
//! `fixtures/gemma_ops/golden.txt` (written by
//! `python/gemma_ops_fixture.py`) holds reference cases for:
//! 1. `gelu_tanh` forward & backward
//! 2. `seq_mean` masked sequence mean pooling
//! 3. `l2_normalize` Matryoshka prefix L2 normalization
//! 4. `rms_qkv_norm` with learned Q/K weights and weightless V
//! 5. `swa` non-causal symmetric sliding window attention (|i - j| <= W)
//!
//! [`check`] runs a backend through [`Backend::upload`] on every case and measures
//! against PyTorch references within pre-specified tolerance bounds.

use std::collections::HashMap;

use ojas_core::{AttentionOpts, Backend, Budget, OjasError, Tensor};

const GOLDEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fixtures/gemma_ops/golden.txt"
));

pub const MIN_CASES: usize = 15;

/// One fixture case.
pub struct GoldenCase {
    pub op: String,
    pub name: String,
    pub params: Vec<String>,
    f32_tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    f64_tensors: HashMap<String, (Vec<usize>, Vec<f64>)>,
    u32_tensors: HashMap<String, (Vec<usize>, Vec<u32>)>,
}

impl GoldenCase {
    pub fn shape(&self, key: &str) -> &[usize] {
        if let Some((shape, _)) = self.f32_tensors.get(key) {
            shape
        } else if let Some((shape, _)) = self.f64_tensors.get(key) {
            shape
        } else if let Some((shape, _)) = self.u32_tensors.get(key) {
            shape
        } else {
            panic!("tensor {key} not found in case {}", self.name);
        }
    }

    pub fn f32_values(&self, key: &str) -> &[f32] {
        &self.f32_tensors[key].1
    }

    pub fn f64_values(&self, key: &str) -> &[f64] {
        &self.f64_tensors[key].1
    }

    pub fn u32_values(&self, key: &str) -> &[u32] {
        &self.u32_tensors[key].1
    }

    pub fn host_f32(&self, key: &str, budget: &Budget) -> Result<Tensor, OjasError> {
        let (shape, data) = &self.f32_tensors[key];
        Tensor::from_f32(data, shape, budget)
    }

    pub fn host_u32(&self, key: &str, budget: &Budget) -> Result<Tensor, OjasError> {
        let (shape, data) = &self.u32_tensors[key];
        Tensor::from_u32(data, shape, budget)
    }
}

/// The fixture cases, in file order.
pub fn parse() -> Vec<GoldenCase> {
    let mut lines = GOLDEN.lines();
    assert_eq!(lines.next(), Some("ojas-gemma-ops-golden-v1"));
    let torch = lines.next().expect("torch version line");
    assert!(torch.starts_with("torch "), "{torch}");

    let mut cases = Vec::new();
    let mut open: Option<GoldenCase> = None;

    for line in lines {
        let mut f = line.split_whitespace();
        match f.next() {
            Some("case") => {
                let op = f.next().unwrap().to_string();
                let name = f.next().unwrap().to_string();
                let params = f.map(|s| s.to_string()).collect();
                open = Some(GoldenCase {
                    op,
                    name,
                    params,
                    f32_tensors: HashMap::new(),
                    f64_tensors: HashMap::new(),
                    u32_tensors: HashMap::new(),
                });
            }
            Some("t") => {
                let key = f.next().unwrap().to_string();
                let dtype = f.next().unwrap();
                let dims = f.next().unwrap();
                let shape: Vec<usize> = if dims == "-" {
                    vec![]
                } else {
                    dims.split(',').map(|d| d.parse().unwrap()).collect()
                };

                let case = open.as_mut().expect("tensor outside a case");
                match dtype {
                    "f32" => {
                        let data: Vec<f32> = f
                            .map(|h| f32::from_bits(u32::from_str_radix(h, 16).unwrap()))
                            .collect();
                        assert_eq!(
                            data.len(),
                            shape.iter().product::<usize>(),
                            "f32 size mismatch for {key} in {}",
                            case.name
                        );
                        case.f32_tensors.insert(key, (shape, data));
                    }
                    "f64" => {
                        let data: Vec<f64> = f
                            .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
                            .collect();
                        assert_eq!(
                            data.len(),
                            shape.iter().product::<usize>(),
                            "f64 size mismatch for {key} in {}",
                            case.name
                        );
                        case.f64_tensors.insert(key, (shape, data));
                    }
                    "u32" => {
                        let data: Vec<u32> =
                            f.map(|h| u32::from_str_radix(h, 16).unwrap()).collect();
                        assert_eq!(
                            data.len(),
                            shape.iter().product::<usize>(),
                            "u32 size mismatch for {key} in {}",
                            case.name
                        );
                        case.u32_tensors.insert(key, (shape, data));
                    }
                    other => panic!("unknown dtype {other}"),
                }
            }
            Some("end") => {
                cases.push(open.take().expect("end outside a case"));
            }
            None => {}
            Some(other) => panic!("unknown line {other}"),
        }
    }
    assert!(open.is_none(), "unterminated case");
    cases
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
        f64::max(m, (f64::from(x) - f64::from(y)).abs())
    })
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub fn l2_norm(a: &[f32]) -> f64 {
    a.iter()
        .fold(0.0f64, |acc, &v| acc + f64::from(v) * f64::from(v))
        .sqrt()
}

pub struct OpRecord {
    pub op: String,
    pub case: String,
    pub max_abs: f64,
    pub cosine: f64,
}

pub struct Report {
    pub cases_run: usize,
    pub records: Vec<OpRecord>,
}

impl Report {
    pub fn print(&self, backend_name: &str) {
        eprintln!(
            "{backend_name}: Gemma golden tests: {} ops evaluated across {} cases",
            self.records.len(),
            self.cases_run
        );
        for r in &self.records {
            eprintln!(
                "  {:<14} {:<18} max_abs: {:.2e}  cosine: {:.10}",
                r.op, r.case, r.max_abs, r.cosine
            );
        }
    }
}

fn host_f32(be: &impl Backend, t: &Tensor) -> Vec<f32> {
    let host = be.download(t).expect("download to host");
    host.to_f32_vec().expect("to f32 vec")
}

/// Runs every case in `fixtures/gemma_ops/golden.txt` on `be`.
pub fn check<B: Backend>(be: &B) -> Result<Report, OjasError> {
    let cases = parse();
    assert!(cases.len() >= MIN_CASES, "fewer cases than expected");

    let budget = be.budget();
    let mut records = Vec::new();

    for case in &cases {
        match case.op.as_str() {
            "gelu_tanh" => {
                let x_host = case.host_f32("x", budget)?;
                let gy_host = case.host_f32("gy", budget)?;

                let x = be.upload(&x_host)?;
                let gy = be.upload(&gy_host)?;

                // Forward
                let y = be.gelu_tanh_forward(&x)?;
                let y_got = host_f32(be, &y);
                let y_want = case.f32_values("y");

                let max_abs_y = max_abs_diff(&y_got, y_want);
                let cos_y = cosine_similarity(&y_got, y_want);

                assert!(
                    max_abs_y <= 2.0e-7 * (l2_norm(y_want) + 1.0) + 1.0e-6,
                    "gelu_tanh_fwd {} max_abs: {:e}",
                    case.name,
                    max_abs_y
                );
                assert!(
                    cos_y >= 1.0 - 1.0e-9,
                    "gelu_tanh_fwd {} cosine: {:e}",
                    case.name,
                    cos_y
                );

                records.push(OpRecord {
                    op: "gelu_tanh_fwd".into(),
                    case: case.name.clone(),
                    max_abs: max_abs_y,
                    cosine: cos_y,
                });

                // Backward
                let dx = be.gelu_tanh_backward(&x, &gy)?;
                let dx_got = host_f32(be, &dx);
                let dx_want = case.f32_values("dx");

                let max_abs_dx = max_abs_diff(&dx_got, dx_want);
                let cos_dx = cosine_similarity(&dx_got, dx_want);

                assert!(
                    max_abs_dx <= 3.0e-6 * (l2_norm(dx_want) + 1.0) + 1.0e-5,
                    "gelu_tanh_bwd {} max_abs: {:e}",
                    case.name,
                    max_abs_dx
                );

                records.push(OpRecord {
                    op: "gelu_tanh_bwd".into(),
                    case: case.name.clone(),
                    max_abs: max_abs_dx,
                    cosine: cos_dx,
                });
            }
            "seq_mean" => {
                let x_host = case.host_f32("x", budget)?;
                let key_lens_host = case.host_u32("key_lens", budget)?;

                let x = be.upload(&x_host)?;
                let key_lens = be.upload(&key_lens_host)?;

                let y = be.seq_mean_forward(&x, &key_lens)?;
                let y_got = host_f32(be, &y);
                let y_want = case.f32_values("y");

                let max_abs = max_abs_diff(&y_got, y_want);
                let cos = cosine_similarity(&y_got, y_want);

                assert!(
                    max_abs <= 2.0e-6,
                    "seq_mean {} max_abs: {:e}",
                    case.name,
                    max_abs
                );

                records.push(OpRecord {
                    op: "seq_mean".into(),
                    case: case.name.clone(),
                    max_abs,
                    cosine: cos,
                });
            }
            "l2_normalize" => {
                let x_host = case.host_f32("x", budget)?;
                let eps = f32::from_bits(u32::from_str_radix(&case.params[0], 16).unwrap());

                let x = be.upload(&x_host)?;
                let y = be.l2_normalize_forward(&x, eps)?;
                let y_got = host_f32(be, &y);
                let y_want = case.f32_values("y");

                let max_abs = max_abs_diff(&y_got, y_want);
                let cos = cosine_similarity(&y_got, y_want);

                assert!(
                    max_abs <= 1.0e-6,
                    "l2_normalize {} max_abs: {:e}",
                    case.name,
                    max_abs
                );

                // Unit norm invariant check on each row
                let rows = case.shape("x")[0];
                let dim = case.shape("x")[1];
                for r in 0..rows {
                    let row_got = &y_got[r * dim..(r + 1) * dim];
                    let row_in = &case.f32_values("x")[r * dim..(r + 1) * dim];
                    let in_norm = l2_norm(row_in);
                    let norm = l2_norm(row_got);
                    if in_norm > 1.0e-6 {
                        assert!(
                            (norm - 1.0).abs() <= 1.0e-6,
                            "l2_normalize row {r} not unit norm: got {norm}"
                        );
                        let row_cos = cosine_similarity(row_got, row_in);
                        assert!(
                            row_cos >= 1.0 - 1.0e-11,
                            "l2_normalize row {r} cosine degraded: got {row_cos}"
                        );
                    }
                }

                records.push(OpRecord {
                    op: "l2_normalize".into(),
                    case: case.name.clone(),
                    max_abs,
                    cosine: cos,
                });
            }
            "rms_qkv_norm" => {
                let q_host = case.host_f32("q", budget)?;
                let k_host = case.host_f32("k", budget)?;
                let v_host = case.host_f32("v", budget)?;
                let w_q_host = case.host_f32("w_q", budget)?;
                let w_k_host = case.host_f32("w_k", budget)?;
                let eps = f32::from_bits(u32::from_str_radix(&case.params[0], 16).unwrap());

                let q = be.upload(&q_host)?;
                let k = be.upload(&k_host)?;
                let v = be.upload(&v_host)?;
                let w_q = be.upload(&w_q_host)?;
                let w_k = be.upload(&w_k_host)?;

                let (qn, kn, vn) =
                    be.rms_qkv_norm_forward(&q, &k, &v, Some(&w_q), Some(&w_k), eps)?;

                let qn_got = host_f32(be, &qn);
                let kn_got = host_f32(be, &kn);
                let vn_got = host_f32(be, &vn);

                let qn_want = case.f32_values("q_norm");
                let kn_want = case.f32_values("k_norm");
                let vn_want = case.f32_values("v_norm");

                let max_abs_q = max_abs_diff(&qn_got, qn_want);
                let max_abs_k = max_abs_diff(&kn_got, kn_want);
                let max_abs_v = max_abs_diff(&vn_got, vn_want);

                assert!(
                    max_abs_q <= 1.0e-5,
                    "rms_qkv_norm q {} max_abs: {:e}",
                    case.name,
                    max_abs_q
                );
                assert!(
                    max_abs_k <= 1.0e-5,
                    "rms_qkv_norm k {} max_abs: {:e}",
                    case.name,
                    max_abs_k
                );
                assert!(
                    max_abs_v <= 1.0e-5,
                    "rms_qkv_norm v {} max_abs: {:e}",
                    case.name,
                    max_abs_v
                );

                records.push(OpRecord {
                    op: "rms_qkv_norm".into(),
                    case: case.name.clone(),
                    max_abs: max_abs_q.max(max_abs_k).max(max_abs_v),
                    cosine: cosine_similarity(&qn_got, qn_want),
                });
            }
            "swa" => {
                let q_host = case.host_f32("q", budget)?;
                let k_host = case.host_f32("k", budget)?;
                let v_host = case.host_f32("v", budget)?;
                let key_lens_host = case.host_u32("key_lens", budget)?;

                let radius = case.params[0].parse::<usize>().unwrap();
                let scale = f32::from_bits(u32::from_str_radix(&case.params[1], 16).unwrap());

                let q = be.upload(&q_host)?;
                let k = be.upload(&k_host)?;
                let v = be.upload(&v_host)?;
                let key_lens = be.upload(&key_lens_host)?;

                let opts = AttentionOpts {
                    causal: false,
                    radius: Some(radius),
                    scale: Some(scale),
                    key_lens: Some(&key_lens),
                    ..Default::default()
                };

                let y = be.attention_forward(&q, &k, &v, opts)?;
                let y_got = host_f32(be, &y);
                let y_want = case.f32_values("y");

                let max_abs = max_abs_diff(&y_got, y_want);
                let cos = cosine_similarity(&y_got, y_want);

                assert!(
                    max_abs <= 1.0e-4,
                    "swa {} max_abs: {:e}",
                    case.name,
                    max_abs
                );
                assert!(cos >= 1.0 - 1.0e-8, "swa {} cosine: {:e}", case.name, cos);

                records.push(OpRecord {
                    op: "swa".into(),
                    case: case.name.clone(),
                    max_abs,
                    cosine: cos,
                });
            }
            other => panic!("unknown op {other} in case {}", case.name),
        }
    }

    Ok(Report {
        cases_run: cases.len(),
        records,
    })
}
