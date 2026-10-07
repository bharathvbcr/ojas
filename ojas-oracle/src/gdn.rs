//! The gated delta rule (Gated DeltaNet) at transformers' training seam, in
//! `f64`, and the published goldens it and the backends are checked against.
//!
//! [`forward_f64`] is the forward of `Backend::chunked_gdn_forward`: per
//! `(b, h)`, with the state `S` `[Dk, Dv]` starting at `s0` or zeros,
//!
//! ```text
//! q^ = l2norm(q) / sqrt(Dk),  k^ = l2norm(k)       l2norm(x) = x / sqrt(|x|^2 + 1e-6)
//! S^ = exp(g_t) S;  u = v_t - S^^T k^;  S = S^ + k^ (beta_t u)^T;  o_t = S^T q^
//! ```
//!
//! It follows `ojas-cuda/tests/reference/gdn_published.rs` (itself a
//! port of tessl's `tests/common/gdn_train.rs`), forward only: the
//! backends' gradients are checked against central differences of it and
//! against the goldens, not against a second hand-written backward.
//!
//! The goldens are `ojas-cuda/tests/fixtures/goldens/
//! gdn_published_train_T{1,63,64,65,130}_*.npy`: transformers'
//! `torch_recurrent_gated_delta_rule` at float64 with torch autograd for the
//! gradients, `B 1, H 2, Dk 128, Dv 16`, each with an initial state and a
//! final-state gradient. They are read in place (pinned by sha256 in that
//! directory's `manifest.json`), not copied.

use std::path::{Path, PathBuf};

use ojas_core::OjasError;

/// transformers' `l2norm` epsilon.
pub const L2_EPS: f64 = 1e-6;

/// The sequence lengths of the published goldens: one token, either side of
/// the first 64-token checkpoint, and three chunks.
pub const GOLDEN_SEQ: [usize; 5] = [1, 63, 64, 65, 130];

/// `B`, `T`, `H`, `Dk`, `Dv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub b: usize,
    pub t: usize,
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
}

impl Shape {
    pub fn rows(&self) -> usize {
        self.b * self.t * self.h
    }

    pub fn state_len(&self) -> usize {
        self.b * self.h * self.dk * self.dv
    }
}

/// The forward's operands in the seam's layouts: `q`, `k` `[B, T, H, Dk]`,
/// `v` `[B, T, H, Dv]`, `g`, `beta` `[B, T, H]`, `s0` `[B, H, Dk, Dv]`.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub s: Shape,
    pub q: &'a [f64],
    pub k: &'a [f64],
    pub v: &'a [f64],
    pub g: &'a [f64],
    pub beta: &'a [f64],
    pub s0: Option<&'a [f64]>,
}

fn bad(detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op: "gdn_oracle",
        detail: detail.into(),
    }
}

impl Inputs<'_> {
    fn validate(&self) -> Result<(), OjasError> {
        let s = self.s;
        if s.b == 0 || s.t == 0 || s.h == 0 || s.dk == 0 || s.dv == 0 {
            return Err(bad(format!("every dimension must be non-zero, got {s:?}")));
        }
        let n = s.rows();
        for (name, got, want) in [
            ("q", self.q.len(), n * s.dk),
            ("k", self.k.len(), n * s.dk),
            ("v", self.v.len(), n * s.dv),
            ("g", self.g.len(), n),
            ("beta", self.beta.len(), n),
            (
                "s0",
                self.s0.map_or(s.state_len(), <[f64]>::len),
                s.state_len(),
            ),
        ] {
            if got != want {
                return Err(bad(format!("{name} has {got} values, want {want}")));
            }
        }
        let all = [
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            self.s0.unwrap_or(&[]),
        ];
        if all.iter().any(|xs| xs.iter().any(|x| !x.is_finite())) {
            return Err(OjasError::NonFinite { op: "gdn_oracle" });
        }
        Ok(())
    }
}

/// `o` `[B, T, H, Dv]` and the final state `[B, H, Dk, Dv]`.
#[derive(Clone, Debug)]
pub struct Forward {
    pub o: Vec<f64>,
    pub fin: Vec<f64>,
}

fn l2norm(x: &[f64]) -> Vec<f64> {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + L2_EPS).sqrt();
    x.iter().map(|v| v * r).collect()
}

/// The forward, published rule: the correction reads the decayed state.
// Indexed as the reference it follows, term for term.
#[allow(clippy::needless_range_loop)]
pub fn forward_f64(p: &Inputs<'_>) -> Result<Forward, OjasError> {
    p.validate()?;
    let Shape { b, t, h, dk, dv } = p.s;
    let per = dk * dv;
    let scale = 1.0 / f64::from(u32::try_from(dk).map_err(|_| bad("dk does not fit u32"))?).sqrt();
    let mut o = vec![0.0; p.s.rows() * dv];
    let mut fin = vec![0.0; p.s.state_len()];
    for bi in 0..b {
        for hi in 0..h {
            let bh = bi * h + hi;
            let mut st = match p.s0 {
                Some(s0) => s0[bh * per..(bh + 1) * per].to_vec(),
                None => vec![0.0; per],
            };
            for ti in 0..t {
                let row = (bi * t + ti) * h + hi;
                let kh = l2norm(&p.k[row * dk..(row + 1) * dk]);
                let qn = l2norm(&p.q[row * dk..(row + 1) * dk]);
                let a = p.g[row].exp();
                for x in st.iter_mut() {
                    *x *= a;
                }
                for j in 0..dv {
                    let kv: f64 = (0..dk).map(|i| st[i * dv + j] * kh[i]).sum();
                    let delta = p.beta[row] * (p.v[row * dv + j] - kv);
                    for i in 0..dk {
                        st[i * dv + j] += kh[i] * delta;
                    }
                }
                for j in 0..dv {
                    o[row * dv + j] = (0..dk).map(|i| st[i * dv + j] * qn[i] * scale).sum();
                }
            }
            fin[bh * per..(bh + 1) * per].copy_from_slice(&st);
        }
    }
    Ok(Forward { o, fin })
}

/// The directory holding the published goldens, beside this crate.
pub fn goldens_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ojas-cuda/tests/fixtures/goldens")
}

/// One `.npy` file of little-endian `<f8` values in C order, format 1.0 or
/// 2.0: its shape and values. Any other dtype, order or version is refused
/// by name rather than read, so a golden in an unexpected layout fails the
/// test that loads it.
pub fn read_npy_f64(path: &Path) -> Result<(Vec<usize>, Vec<f64>), OjasError> {
    let named = |detail: String| OjasError::OutOfRange {
        op: "gdn_golden",
        detail: format!("{}: {detail}", path.display()),
    };
    let bytes = std::fs::read(path).map_err(|e| named(e.to_string()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(named("not an .npy file".into()));
    }
    let (header_len, start) = match bytes[6] {
        1 => (
            usize::from(u16::from_le_bytes([bytes[8], bytes[9]])),
            10usize,
        ),
        2 if bytes.len() >= 12 => {
            let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (
                usize::try_from(n).map_err(|_| named("header length".into()))?,
                12,
            )
        }
        v => return Err(named(format!("unsupported .npy version {v}"))),
    };
    let end = start
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| named("header runs past the end".into()))?;
    let header =
        std::str::from_utf8(&bytes[start..end]).map_err(|_| named("header is not utf-8".into()))?;
    if !header.contains("'descr': '<f8'") {
        return Err(named(format!("dtype is not <f8: {header}")));
    }
    if !header.contains("'fortran_order': False") {
        return Err(named("not C order".into()));
    }
    let shape_src = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| named("header has no shape".into()))?;
    let shape = shape_src
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| named(format!("shape {s:?}: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| named("shape overflows".into()))?;
    let payload = &bytes[end..];
    if Some(payload.len()) != numel.checked_mul(8) {
        return Err(named(format!(
            "payload is {} bytes, shape {shape:?} needs {}",
            payload.len(),
            numel.saturating_mul(8)
        )));
    }
    // The length was checked against the shape, so the remainder is empty.
    let values = payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    Ok((shape, values))
}

/// One published golden case: every operand, output and gradient.
#[derive(Clone, Debug)]
pub struct Golden {
    pub s: Shape,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub g: Vec<f64>,
    pub beta: Vec<f64>,
    pub s0: Vec<f64>,
    pub d_o: Vec<f64>,
    pub dfin: Vec<f64>,
    pub o: Vec<f64>,
    pub fin: Vec<f64>,
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    pub dg: Vec<f64>,
    pub dbeta: Vec<f64>,
    pub ds0: Vec<f64>,
}

impl Golden {
    pub fn inputs(&self) -> Inputs<'_> {
        Inputs {
            s: self.s,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: Some(&self.s0),
        }
    }
}

/// `gdn_published_train_T{t}`, every file checked against the shape `q` and
/// `v` give.
pub fn golden(t: usize) -> Result<Golden, OjasError> {
    let dir = goldens_dir();
    let name = format!("gdn_published_train_T{t}");
    let load = |part: &str, want: Option<&[usize]>| -> Result<Vec<f64>, OjasError> {
        let (shape, values) = read_npy_f64(&dir.join(format!("{name}_{part}.npy")))?;
        if let Some(want) = want {
            if shape != want {
                return Err(bad(format!("{name}_{part} shape {shape:?} != {want:?}")));
            }
        }
        Ok(values)
    };
    let (qs, q) = read_npy_f64(&dir.join(format!("{name}_q.npy")))?;
    let (vs, v) = read_npy_f64(&dir.join(format!("{name}_v.npy")))?;
    let (&[b, tt, h, dk], &[_, _, _, dv]) = (qs.as_slice(), vs.as_slice()) else {
        return Err(bad(format!("{name}: q {qs:?} or v {vs:?} is not rank 4")));
    };
    if tt != t || vs[..3] != qs[..3] {
        return Err(bad(format!(
            "{name}: q {qs:?} and v {vs:?} disagree with T {t}"
        )));
    }
    let s = Shape { b, t, h, dk, dv };
    let tok: &[usize] = &[b, t, h];
    let key: &[usize] = &[b, t, h, dk];
    let val: &[usize] = &[b, t, h, dv];
    let state: &[usize] = &[b, h, dk, dv];
    Ok(Golden {
        s,
        q,
        v,
        k: load("k", Some(key))?,
        g: load("g", Some(tok))?,
        beta: load("beta", Some(tok))?,
        s0: load("s0", Some(state))?,
        d_o: load("d_o", Some(val))?,
        dfin: load("dfin", Some(state))?,
        o: load("o", Some(val))?,
        fin: load("fin", Some(state))?,
        dq: load("dq", Some(key))?,
        dk: load("dk", Some(key))?,
        dv: load("dv", Some(val))?,
        dg: load("dg", Some(tok))?,
        dbeta: load("dbeta", Some(tok))?,
        ds0: load("ds0", Some(state))?,
    })
}
