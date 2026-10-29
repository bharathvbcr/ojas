//! Parity of Ojas ops against the SpikeTrack SAM 3.1 activation fixtures.
//!
//! The fixtures (`fixtures/backbone/*.safetensors` in the SpikeTrack
//! checkout, written by its `tools/reference/dump_fixtures.py` from torch
//! 2.14 in fp32) hold the input and output activations of SAM 3.1's ViT
//! trunk and neck on one frame. They are far too large to ship (3.4 GB for a
//! 1008x1008 frame), so every function here reads only the tensors it needs,
//! by position, through [`ojas_io::SafeTensors::open`], and the tests that
//! use the real file are `#[ignore]`d unless `OJAS_SAM_FIXTURES` names it.
//!
//! Each function checks one op in isolation: the op is fed the tensor torch
//! recorded as its input and its output is compared with the tensor torch
//! recorded as its output, so errors never compound across ops. The
//! functions are generic over [`Backend`]: the CPU test and the Metal test
//! call the same ones.
//!
//! **What the fixtures do not hold: weights.** The files carry activations
//! only (no tensor of any of them is a weight or bias), so the ops that need
//! one (`layer_norm`, `linear_bias`, `conv2d`, `conv_transpose2d`) cannot be
//! compared with torch directly. [`skipped_ops`] records each one with the
//! reason, read from the file's own header. For LayerNorm only,
//! [`layer_norm_affine_recovered`] gives a weaker check: the per-channel
//! scale and shift are recovered from torch's own input and output by least
//! squares, and the residual of the op against that fit is measured. It
//! validates the row statistics (mean, variance, epsilon) but absorbs any
//! per-channel error, and its [`OpReport::kind`] says so.
//!
//! Every [`OpReport`] carries the measured error and the [`Tolerance`] it was
//! held to; `pass` is the conjunction of the three bounds. The tolerances are
//! the constants below, stated with their reasoning, not derived from the
//! measurement.

use std::fmt;
use std::path::Path;

use ojas_core::{axial_rope_2d_tables, AttentionOpts, Backend, OjasError, Tensor};
use ojas_io::SafeTensors;

use crate::{bad, io};

/// The environment variable that names the fixture file.
pub const ENV_FIXTURES: &str = "OJAS_SAM_FIXTURES";

/// The environment variable that names the GroupNorm fixture file
/// (`fixtures/groupnorm/pixel_decoder.safetensors`): the SpikeTrack pixel
/// decoder's two norms, a different file from the backbone's.
pub const ENV_GROUPNORM_FIXTURES: &str = "OJAS_SAM_GROUPNORM_FIXTURES";

/// A tensor above this many elements is refused before it is read: the
/// largest fixture tensor is 25 million, and a header that claims more is not
/// one of ours.
pub const MAX_TENSOR_ELEMENTS: usize = 1 << 28;

/// RoPE base frequency of SAM 3.1's ViT (`rope_theta`, README of the
/// reference tools).
pub const ROPE_THETA: f64 = 10_000.0;

/// What an op is held to. All three must hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Largest `|got - want|`.
    pub max_abs: f64,
    /// Largest `|got - want| / max(|want|, floor)` with `floor = 1e-3 *
    /// max|want|`, so a value near zero is judged against the tensor's scale.
    pub max_rel: f64,
    /// Smallest cosine similarity of the two tensors as vectors.
    pub min_cosine: f64,
}

/// RoPE tables: cos and sin of an f32 angle, torch's in f32, ours from f64
/// and rounded once. Both are within an ulp of the true value (6e-8 on
/// values of at most 1).
pub const TABLE_TOL: Tolerance = Tolerance {
    max_abs: 1e-6,
    max_rel: 1e-3,
    min_cosine: 1.0 - 1e-9,
};

/// A rotation applied to activations: two products and one add per element,
/// so a few ulp of the activation scale.
pub const ROPE_TOL: Tolerance = Tolerance {
    max_abs: 1e-5,
    max_rel: 1e-3,
    min_cosine: 1.0 - 1e-9,
};

/// Pointwise erf GELU: ours is correctly rounded, torch's is an f32 erf
/// within a few ulp.
pub const GELU_TOL: Tolerance = Tolerance {
    max_abs: 1e-5,
    max_rel: 1e-3,
    min_cosine: 1.0 - 1e-9,
};

/// LayerNorm outputs, f32 row sums against torch's: a few ulp of the output
/// scale (SAM's outlier channels reach tens).
pub const LAYER_NORM_TOL: Tolerance = Tolerance {
    max_abs: 1e-4,
    max_rel: 1e-3,
    min_cosine: 1.0 - 1e-9,
};

/// GroupNorm outputs: the same few-ulp-of-the-output-scale bound as
/// [`LAYER_NORM_TOL`], whose reasoning carries over. Torch's f32 statistics
/// over 0.66 to 2.6 million values per group and ours in `f64` differ by the
/// rounding of torch's sums, which is far below this bound; the bound is not
/// derived from the measurement.
pub const GROUP_NORM_TOL: Tolerance = Tolerance {
    max_abs: 1e-4,
    max_rel: 1e-3,
    min_cosine: 1.0 - 1e-9,
};

/// Attention over 576 or 5184 keys: f32 sums of that length differ in the
/// order torch's blocked kernel and ours add them, about `1e-7 * sqrt(keys)`
/// on outputs of order one.
pub const ATTENTION_TOL: Tolerance = Tolerance {
    max_abs: 1e-4,
    max_rel: 1e-2,
    min_cosine: 1.0 - 1e-8,
};

/// How a report was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The op's output against torch's recorded output.
    Direct,
    /// Against a fit recovered from torch's own recording
    /// ([`layer_norm_affine_recovered`]); a per-channel error is invisible.
    Derived,
}

/// One op's measured error against one fixture tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct OpReport {
    pub op: String,
    /// The tensors and parameters compared, exactly.
    pub detail: String,
    pub kind: Kind,
    pub elements: usize,
    /// `max |want|`, the scale the relative error is judged against.
    pub scale: f64,
    pub max_abs: f64,
    pub max_rel: f64,
    pub cosine: f64,
    pub tolerance: Tolerance,
    pub pass: bool,
}

impl OpReport {
    /// Header of [`OpReport::row`]: a Markdown table.
    pub fn header() -> &'static str {
        "| op | compared | kind | elements | max abs | max rel | cosine | tol abs | tol rel | tol cosine | result |\n\
         |---|---|---|---|---|---|---|---|---|---|---|"
    }

    /// One Markdown row.
    pub fn row(&self) -> String {
        format!(
            "| {} | {} | {} | {} | {:.3e} | {:.3e} | {:.12} | {:.0e} | {:.0e} | {:.12} | {} |",
            self.op,
            self.detail,
            match self.kind {
                Kind::Direct => "direct",
                Kind::Derived => "derived",
            },
            self.elements,
            self.max_abs,
            self.max_rel,
            self.cosine,
            self.tolerance.max_abs,
            self.tolerance.max_rel,
            self.tolerance.min_cosine,
            if self.pass { "pass" } else { "FAIL" },
        )
    }
}

impl fmt::Display for OpReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.row())
    }
}

/// An op the fixture cannot check, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    pub op: &'static str,
    /// The fixture tensors that would have been compared.
    pub tensors: String,
    pub reason: String,
}

/// Compare `got` with `want` and judge the result against `tol`.
///
/// A length mismatch, a non-finite `want` (a corrupt fixture) or a
/// non-finite `got` is an error, never a failing number: an op that produced
/// NaN has not been measured.
pub fn compare(
    op: &str,
    detail: &str,
    kind: Kind,
    got: &[f32],
    want: &[f32],
    tolerance: Tolerance,
) -> Result<OpReport, OjasError> {
    if got.len() != want.len() {
        return Err(bad(&format!(
            "{op}: {} values against a fixture of {}",
            got.len(),
            want.len()
        )));
    }
    if want.iter().any(|v| !v.is_finite()) {
        return Err(bad(&format!(
            "{op}: the fixture tensor holds a non-finite value"
        )));
    }
    if got.iter().any(|v| !v.is_finite()) {
        return Err(OjasError::NonFinite { op: "sam_fixtures" });
    }
    let scale = want.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let floor = (1e-3 * scale).max(f64::MIN_POSITIVE);
    let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
    let (mut dot, mut gg, mut ww) = (0.0f64, 0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (f64::from(g), f64::from(w));
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / w.abs().max(floor));
        dot += g * w;
        gg += g * g;
        ww += w * w;
    }
    let cosine = if gg == 0.0 && ww == 0.0 {
        1.0
    } else if gg == 0.0 || ww == 0.0 {
        0.0
    } else {
        dot / (gg.sqrt() * ww.sqrt())
    };
    let pass = max_abs <= tolerance.max_abs
        && max_rel <= tolerance.max_rel
        && cosine >= tolerance.min_cosine;
    Ok(OpReport {
        op: op.to_string(),
        detail: detail.to_string(),
        kind,
        elements: got.len(),
        scale,
        max_abs,
        max_rel,
        cosine,
        tolerance,
        pass,
    })
}

/// An open fixture file. Only the header is read at open; each
/// [`Fixture::tensor`] reads one tensor's bytes by position.
#[derive(Debug)]
pub struct Fixture<'a> {
    st: SafeTensors<'a>,
}

impl Fixture<'static> {
    /// Open the file at `path`.
    pub fn open(path: &Path) -> Result<Self, OjasError> {
        Ok(Self {
            st: SafeTensors::open(path).map_err(io)?,
        })
    }

    /// The file `OJAS_SAM_FIXTURES` names, or `None` when it is unset or empty.
    pub fn from_env() -> Result<Option<Self>, OjasError> {
        Self::from_env_var(ENV_FIXTURES)
    }

    /// The file the environment variable `var` names, or `None` when it is
    /// unset or empty.
    pub fn from_env_var(var: &str) -> Result<Option<Self>, OjasError> {
        match std::env::var_os(var) {
            Some(path) if !path.is_empty() => Self::open(Path::new(&path)).map(Some),
            _ => Ok(None),
        }
    }
}

impl<'a> Fixture<'a> {
    /// A whole fixture file in memory (the synthetic fixtures of the tests).
    pub fn parse(bytes: &'a [u8]) -> Result<Self, OjasError> {
        Ok(Self {
            st: SafeTensors::parse(bytes).map_err(io)?,
        })
    }

    /// A value of the file's `__metadata__` header.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.st.metadata().get(key).map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.st.info(name).is_ok()
    }

    /// Every tensor name, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.st.names().collect()
    }

    /// A tensor's shape, from the header alone: no tensor data is read.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>, OjasError> {
        let info = self.st.info(name).map_err(io)?;
        info.shape
            .iter()
            .map(|&d| usize::try_from(d).map_err(|_| bad(&format!("{name}: extent exceeds usize"))))
            .collect()
    }

    /// An `F32` tensor's shape and values.
    pub fn tensor(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), OjasError> {
        let info = self.st.info(name).map_err(io)?;
        let mut elements = 1usize;
        let mut shape = Vec::with_capacity(info.shape.len());
        for &d in &info.shape {
            let d =
                usize::try_from(d).map_err(|_| bad(&format!("{name}: extent exceeds usize")))?;
            elements = elements
                .checked_mul(d)
                .filter(|&n| n <= MAX_TENSOR_ELEMENTS)
                .ok_or_else(|| {
                    bad(&format!(
                        "{name}: more than {MAX_TENSOR_ELEMENTS} elements are refused"
                    ))
                })?;
            shape.push(d);
        }
        let (_, values) = self.st.read_f32(name).map_err(io)?;
        Ok((shape, values))
    }
}

fn put<B: Backend + ?Sized>(
    backend: &B,
    values: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let host = Tensor::from_f32(values, shape, backend.budget())?;
    backend.upload(&host)
}

fn get<B: Backend + ?Sized>(backend: &B, tensor: &Tensor) -> Result<Vec<f32>, OjasError> {
    backend.download(tensor)?.to_f32_vec()
}

fn shape_is(name: &str, got: &[usize], want: &[usize]) -> Result<(), OjasError> {
    if got == want {
        Ok(())
    } else {
        Err(bad(&format!("{name}: shape {got:?}, expected {want:?}")))
    }
}

/// `[b, h, t, d]` to `[b, t, h, d]`, on the host: the layout change between
/// torch's `[B, H, T, D]` and `rope_interleaved_forward`'s `[B, T, H, D]`.
fn bhtd_to_bthd(data: &[f32], [b, h, t, d]: [usize; 4]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                let src = ((bi * h + hi) * t + ti) * d;
                let dst = ((bi * t + ti) * h + hi) * d;
                out[dst..dst + d].copy_from_slice(&data[src..src + d]);
            }
        }
    }
    out
}

/// The inverse of [`bhtd_to_bthd`].
fn bthd_to_bhtd(data: &[f32], [b, h, t, d]: [usize; 4]) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                let src = ((bi * t + ti) * h + hi) * d;
                let dst = ((bi * h + hi) * t + ti) * d;
                out[dst..dst + d].copy_from_slice(&data[src..src + d]);
            }
        }
    }
    out
}

/// `freqs_cis` `[T, D/2, 2]` (real, imaginary) as `cos` and `sin`, each
/// `[T, D/2]`.
fn split_freqs(shape: &[usize], freqs: &[f32]) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    if shape.len() != 3 || shape[2] != 2 {
        return Err(bad(&format!(
            "freqs_cis shape {shape:?} is not [T, D/2, 2]"
        )));
    }
    let cos = freqs.iter().step_by(2).copied().collect();
    let sin = freqs.iter().skip(1).step_by(2).copied().collect();
    Ok((cos, sin))
}

/// [`axial_rope_2d_tables`] against the block's `attn.rope.freqs_cis`.
///
/// `grid` is the token grid of the block (`[24, 24]` for a window, `[72,
/// 72]` for a global layer) and `scale_pos` the position scale (1 for a
/// window, 24/72 for a global layer, which tiles the 24x24 pretraining grid
/// over 72x72). The head dimension is twice the table's width and `theta` is
/// [`ROPE_THETA`].
pub fn axial_tables<B: Backend + ?Sized>(
    backend: &B,
    fixture: &Fixture<'_>,
    block: &str,
    grid: [usize; 2],
    scale_pos: f32,
) -> Result<OpReport, OjasError> {
    let name = format!("{block}.attn.rope.freqs_cis");
    let (shape, freqs) = fixture.tensor(&name)?;
    let (cos, sin) = split_freqs(&shape, &freqs)?;
    let head_dim = 2 * shape[1];
    let (got_cos, got_sin) = axial_rope_2d_tables(
        head_dim,
        grid[0],
        grid[1],
        ROPE_THETA,
        scale_pos,
        backend.budget(),
    )?;
    let mut got = get(backend, &got_cos)?;
    got.extend(get(backend, &got_sin)?);
    let mut want = cos;
    want.extend(sin);
    compare(
        "axial_rope_2d_tables",
        &format!(
            "{name} (cos then sin); head_dim {head_dim}, grid {}x{}, theta {ROPE_THETA}, scale_pos {scale_pos}",
            grid[0], grid[1]
        ),
        Kind::Direct,
        &got,
        &want,
        TABLE_TOL,
    )
}

/// `rope_interleaved_forward` on `stream` (`"q"` or `"k"`): the block's
/// `attn.rope.{stream}_in` `[B, H, T, D]` rotated by the cos and sin of its
/// `attn.rope.freqs_cis`, against `attn.rope.{stream}_out`.
pub fn rope_interleaved<B: Backend + ?Sized>(
    backend: &B,
    fixture: &Fixture<'_>,
    block: &str,
    stream: &str,
) -> Result<OpReport, OjasError> {
    let input_name = format!("{block}.attn.rope.{stream}_in");
    let output_name = format!("{block}.attn.rope.{stream}_out");
    let freqs_name = format!("{block}.attn.rope.freqs_cis");
    let (shape, x) = fixture.tensor(&input_name)?;
    let (out_shape, want) = fixture.tensor(&output_name)?;
    shape_is(&output_name, &out_shape, &shape)?;
    let dims: [usize; 4] = shape.as_slice().try_into().map_err(|_| {
        bad(&format!(
            "{input_name}: shape {shape:?} is not [B, H, T, D]"
        ))
    })?;
    let [b, h, t, d] = dims;
    let (f_shape, freqs) = fixture.tensor(&freqs_name)?;
    shape_is(&freqs_name, &f_shape, &[t, d / 2, 2])?;
    let (cos, sin) = split_freqs(&f_shape, &freqs)?;
    let xt = put(backend, &bhtd_to_bthd(&x, dims), &[b, t, h, d])?;
    let ct = put(backend, &cos, &[t, d / 2])?;
    let st = put(backend, &sin, &[t, d / 2])?;
    let y = backend.rope_interleaved_forward(&xt, &ct, &st)?;
    let got = bthd_to_bhtd(&get(backend, &y)?, dims);
    compare(
        "rope_interleaved_forward",
        &format!("{input_name} x {freqs_name} -> {output_name} {shape:?}"),
        Kind::Direct,
        &got,
        &want,
        ROPE_TOL,
    )
}

/// The first `heads` heads of every batch row of `[B, H, T, D]`.
fn first_heads(data: &[f32], [b, h, t, d]: [usize; 4], heads: usize) -> Vec<f32> {
    let plane = t * d;
    let mut out = Vec::with_capacity(b * heads * plane);
    for bi in 0..b {
        let start = bi * h * plane;
        out.extend_from_slice(&data[start..start + heads * plane]);
    }
    out
}

/// `attention_forward` (plain: non-causal, no mask, no bias, scale `1/sqrt(D)`)
/// of the block's `attn.sdpa.{q,k,v}` `[B, H, T, D]` against `attn.sdpa.out`.
///
/// `heads` limits the check to the first `heads` heads of each batch row (the
/// global layer's 5184-token attention is slow on a CPU); `None` checks them
/// all, and the report's `detail` says which.
pub fn attention<B: Backend + ?Sized>(
    backend: &B,
    fixture: &Fixture<'_>,
    block: &str,
    heads: Option<usize>,
) -> Result<OpReport, OjasError> {
    let names = ["q", "k", "v", "out"].map(|n| format!("{block}.attn.sdpa.{n}"));
    let (shape, q) = fixture.tensor(&names[0])?;
    let (k_shape, k) = fixture.tensor(&names[1])?;
    let (v_shape, v) = fixture.tensor(&names[2])?;
    let (o_shape, out) = fixture.tensor(&names[3])?;
    shape_is(&names[1], &k_shape, &shape)?;
    shape_is(&names[2], &v_shape, &shape)?;
    shape_is(&names[3], &o_shape, &shape)?;
    let dims: [usize; 4] = shape.as_slice().try_into().map_err(|_| {
        bad(&format!(
            "{}: shape {shape:?} is not [B, H, T, D]",
            names[0]
        ))
    })?;
    let used = heads.unwrap_or(dims[1]).min(dims[1]);
    if used == 0 {
        return Err(bad("attention: zero heads requested"));
    }
    let sliced = [dims[0], used, dims[2], dims[3]];
    let slice = |data: &[f32]| first_heads(data, dims, used);
    let (qt, kt, vt) = (
        put(backend, &slice(&q), &sliced)?,
        put(backend, &slice(&k), &sliced)?,
        put(backend, &slice(&v), &sliced)?,
    );
    let y = backend.attention_forward(&qt, &kt, &vt, AttentionOpts::default())?;
    shape_is("attention output", y.shape(), &sliced)?;
    let got = get(backend, &y)?;
    compare(
        "attention_forward",
        &format!(
            "{block}.attn.sdpa.{{q,k,v}} -> {} {shape:?}, heads {used} of {}, non-causal, no mask, scale 1/{}",
            names[3],
            dims[1],
            (dims[3] as f64).sqrt()
        ),
        Kind::Direct,
        &got,
        &slice(&out),
        ATTENTION_TOL,
    )
}

/// `gelu_erf_forward` of fixture tensor `input` against fixture tensor
/// `output`, which must have the same shape.
pub fn gelu<B: Backend + ?Sized>(
    backend: &B,
    fixture: &Fixture<'_>,
    input: &str,
    output: &str,
) -> Result<OpReport, OjasError> {
    let (shape, x) = fixture.tensor(input)?;
    let (o_shape, want) = fixture.tensor(output)?;
    shape_is(output, &o_shape, &shape)?;
    let y = backend.gelu_erf_forward(&put(backend, &x, &shape)?)?;
    shape_is("gelu output", y.shape(), &shape)?;
    compare(
        "gelu_erf_forward",
        &format!("{input} -> {output} {shape:?}"),
        Kind::Direct,
        &get(backend, &y)?,
        &want,
        GELU_TOL,
    )
}

/// `groups` and `eps` the GroupNorm fixture recorded in its header. Both
/// must be present and parse: a fixture without them cannot say what it
/// checked.
pub fn group_norm_header(fixture: &Fixture<'_>) -> Result<(usize, f32), OjasError> {
    let field = |key: &str| {
        fixture
            .metadata(key)
            .ok_or_else(|| bad(&format!("group_norm fixture header has no `{key}`")))
    };
    let groups = field("groups")?
        .parse::<usize>()
        .map_err(|e| bad(&format!("group_norm fixture `groups`: {e}")))?;
    let eps = field("eps")?
        .parse::<f32>()
        .map_err(|e| bad(&format!("group_norm fixture `eps`: {e}")))?;
    Ok((groups, eps))
}

/// GroupNorm on `{prefix}.input`, `{prefix}.weight` and `{prefix}.bias`,
/// against torch's `{prefix}.output`, with the header's `groups` and `eps`
/// checked against the ones the caller expects.
pub fn group_norm<B: Backend + ?Sized>(
    backend: &B,
    fixture: &Fixture<'_>,
    prefix: &str,
    groups: usize,
    eps: f32,
) -> Result<OpReport, OjasError> {
    let header = group_norm_header(fixture)?;
    if header != (groups, eps) {
        return Err(bad(&format!(
            "{prefix}: the header records {header:?}, the call expects {:?}",
            (groups, eps)
        )));
    }
    let name = |part: &str| format!("{prefix}.{part}");
    let (shape, x) = fixture.tensor(&name("input"))?;
    let (w_shape, w) = fixture.tensor(&name("weight"))?;
    let (b_shape, b) = fixture.tensor(&name("bias"))?;
    let (o_shape, want) = fixture.tensor(&name("output"))?;
    shape_is(&name("output"), &o_shape, &shape)?;
    let Some(&channels) = shape.get(1) else {
        return Err(bad(&format!(
            "{prefix}.input: shape {shape:?} is not [N, C, H, W]"
        )));
    };
    shape_is(&name("weight"), &w_shape, &[channels])?;
    shape_is(&name("bias"), &b_shape, &[channels])?;
    let y = backend.group_norm_forward(
        &put(backend, &x, &shape)?,
        &put(backend, &w, &w_shape)?,
        &put(backend, &b, &b_shape)?,
        groups,
        eps,
    )?;
    shape_is("group_norm output", y.shape(), &shape)?;
    compare(
        "group_norm_forward",
        &format!("{prefix} {shape:?} groups {groups} eps {eps:e}"),
        Kind::Direct,
        &get(backend, &y)?,
        &want,
        GROUP_NORM_TOL,
    )
}

/// LayerNorm over the last axis of fixture tensor `input`, checked without
/// its weight and bias (the fixtures hold neither): `layer_norm_forward` with
/// weight 1 and bias 0 gives the normalized rows `n`; for each channel `c`
/// the scale `w_c` and shift `b_c` that best map `n[:, c]` onto torch's
/// `output[:, c]` (least squares, in f64) are recovered, and the op is
/// measured by the residual `output - (w n + b)`.
///
/// This holds the row statistics (mean, variance and epsilon, per row) to
/// torch's, and nothing about the affine part: an error that is the same on
/// every row of a channel is absorbed by the fit. The report is
/// [`Kind::Derived`].
pub fn layer_norm_affine_recovered<B: Backend + ?Sized>(
    backend: &B,
    fixture: &Fixture<'_>,
    input: &str,
    output: &str,
    eps: f32,
) -> Result<OpReport, OjasError> {
    let (shape, x) = fixture.tensor(input)?;
    let (o_shape, want) = fixture.tensor(output)?;
    shape_is(output, &o_shape, &shape)?;
    let dim = *shape.last().ok_or_else(|| bad("layer norm of a scalar"))?;
    if dim == 0 || x.len() % dim != 0 {
        return Err(bad(&format!(
            "{input}: shape {shape:?} has an empty last axis"
        )));
    }
    let rows = x.len() / dim;
    let ones = put(backend, &vec![1.0; dim], &[dim])?;
    let zeros = put(backend, &vec![0.0; dim], &[dim])?;
    let n = backend.layer_norm_forward(&put(backend, &x, &shape)?, &ones, &zeros, eps)?;
    shape_is("layer_norm output", n.shape(), &shape)?;
    let n = get(backend, &n)?;
    let (mut sx, mut sy) = (vec![0.0f64; dim], vec![0.0f64; dim]);
    let (mut sxx, mut sxy) = (vec![0.0f64; dim], vec![0.0f64; dim]);
    for (nrow, yrow) in n.chunks_exact(dim).zip(want.chunks_exact(dim)) {
        for c in 0..dim {
            let (a, b) = (f64::from(nrow[c]), f64::from(yrow[c]));
            sx[c] += a;
            sy[c] += b;
            sxx[c] += a * a;
            sxy[c] += a * b;
        }
    }
    let m = rows as f64;
    let fit: Vec<(f64, f64)> = (0..dim)
        .map(|c| {
            let var = m * sxx[c] - sx[c] * sx[c];
            let w = if var > 0.0 {
                (m * sxy[c] - sx[c] * sy[c]) / var
            } else {
                0.0
            };
            (w, (sy[c] - w * sx[c]) / m)
        })
        .collect();
    let got: Vec<f32> = n
        .chunks_exact(dim)
        .flat_map(|row| {
            row.iter()
                .zip(&fit)
                .map(|(&v, &(w, b))| (w * f64::from(v) + b) as f32)
        })
        .collect();
    compare(
        "layer_norm_forward",
        &format!(
            "{input} -> {output} {shape:?}, eps {eps:e}; weight and bias recovered per channel by least squares (the fixture has none)"
        ),
        Kind::Derived,
        &got,
        &want,
        LAYER_NORM_TOL,
    )
}

// ---- weight-bearing ops, against the real SAM 3.1 weights -----------------
//
// The activation fixture has no weights; `OJAS_SAM_WEIGHTS` names the
// converted checkpoint (`sam3.1_multiplex.safetensors`), a second [`Fixture`]
// kept apart from the activations. Each function below reads one op's input
// and torch's output from the activation fixture and its weight and bias from
// the weights file, so an op is checked against torch's own values.

// What these rows do not cover, by construction of the SAM 3.1 shapes:
//  - every transposed convolution here has kernel == stride (2x2, stride 2),
//    so its taps never overlap; overlapping taps are untested against torch;
//  - the window cut is 72 / 24 = 3 windows a side, so `window_partition`'s
//    padding is not exercised by these rows; it is held to torch by
//    `fixtures/window_pad/golden.txt` instead;
//  - the real `attn.proj` is the one square weight ([1024, 1024]), so a
//    transposed copy of it would not fail on shape; the synthetic tests
//    carry that negative, the real rows rely on the other weights' shapes.

/// The environment variable that names the SAM 3.1 weights file.
pub const ENV_WEIGHTS: &str = "OJAS_SAM_WEIGHTS";

/// Where the vision backbone's modules sit in the checkpoint.
pub const WEIGHTS_PREFIX: &str = "detector.backbone.vision_backbone.";

/// Tolerances for the weight-bearing ops, set before any of them was run
/// against the real file, and not changed after.
///
/// A linear layer or convolution sums `K` products in `f32` (`K` is 1024 for
/// qkv, proj and fc1, 4736 for fc2, 588 to 2304 for the convolutions, and
/// 512 or 1024 for the transposed convolutions, whose taps do not overlap).
/// Torch's blocked GEMM and ours add them in different orders, so the two
/// differ by about `1e-7 * sqrt(K) * sum|a_i b_i|`: a few `1e-6` times the
/// output scale for activations of order one to ten. `max_abs` 1e-3 is
/// two to three orders above that and below the size of any wrong weight,
/// bias, layout or transposition, which moves outputs by their own scale.
/// `max_rel` is per element, `|d| / max(|want_i|, 1e-3 * max|want|)`: an
/// element above the floor is judged by its ordinary relative error, and a
/// near-zero output from a cancelling sum by its whole absolute error divided
/// by the floor. 0.1 is deliberately weak, and the absolute bound and the
/// cosine are the binding ones.
pub const LINEAR_TOL: Tolerance = Tolerance {
    max_abs: 1e-3,
    max_rel: 1e-1,
    min_cosine: 1.0 - 1e-9,
};

/// Convolutions and transposed convolutions: [`LINEAR_TOL`]'s reasoning
/// (a sum of `K` f32 products in a different order), the same bounds.
pub const CONV_TOL: Tolerance = LINEAR_TOL;

/// The checkpoint name of a fixture module's weight or bias. `module` is the
/// fixture's own spelling: `trunk.…` names a module under the ViT trunk
/// (`trunk.blocks.0.attn.qkv`), `neck.convs.N.…` one of the neck's
/// (`neck.convs.0.conv_3x3`), and `part` is `weight` or `bias`. The neck's
/// modules are `convs.N.…` in the checkpoint, with no `neck.` segment. Any
/// other module is refused.
pub fn weight_name(module: &str, part: &str) -> Result<String, OjasError> {
    if part != "weight" && part != "bias" {
        return Err(bad(&format!("weight part `{part}` is not weight or bias")));
    }
    let local = if let Some(rest) = module.strip_prefix("neck.convs.") {
        format!("convs.{rest}")
    } else if module.starts_with("trunk.") {
        module.to_string()
    } else {
        return Err(bad(&format!(
            "module `{module}` is neither under trunk. nor neck.convs."
        )));
    };
    Ok(format!("{WEIGHTS_PREFIX}{local}.{part}"))
}

/// A module's weight and its bias, each with its shape; the bias is `None`
/// when `has_bias` is false.
type Params = (Vec<usize>, Vec<f32>, Option<(Vec<usize>, Vec<f32>)>);

fn params(weights: &Fixture<'_>, module: &str, has_bias: bool) -> Result<Params, OjasError> {
    let (ws, w) = weights.tensor(&weight_name(module, "weight")?)?;
    let bias = if has_bias {
        Some(weights.tensor(&weight_name(module, "bias")?)?)
    } else {
        None
    };
    Ok((ws, w, bias))
}

/// LayerNorm over the last axis with torch's own weight and bias: `module`
/// is the norm (`trunk.ln_pre`, `trunk.blocks.0.norm1`), `input` and `output`
/// the activation fixture's tensors. Compared directly, with
/// [`LAYER_NORM_TOL`].
pub fn layer_norm_weighted<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
    module: &str,
    input: &str,
    output: &str,
    eps: f32,
) -> Result<OpReport, OjasError> {
    let (shape, x) = acts.tensor(input)?;
    let (o_shape, want) = acts.tensor(output)?;
    shape_is(output, &o_shape, &shape)?;
    let (ws, w, bias) = params(weights, module, true)?;
    let (bs, b) = bias.ok_or_else(|| bad("layer norm bias missing"))?;
    let dim = *shape.last().ok_or_else(|| bad("layer norm of a scalar"))?;
    shape_is(&weight_name(module, "weight")?, &ws, &[dim])?;
    shape_is(&weight_name(module, "bias")?, &bs, &[dim])?;
    let y = backend.layer_norm_forward(
        &put(backend, &x, &shape)?,
        &put(backend, &w, &ws)?,
        &put(backend, &b, &bs)?,
        eps,
    )?;
    shape_is("layer_norm output", y.shape(), &shape)?;
    compare(
        "layer_norm_forward",
        &format!("{input} -> {output} {shape:?}, eps {eps:e}, weight and bias of {module}"),
        Kind::Direct,
        &get(backend, &y)?,
        &want,
        LAYER_NORM_TOL,
    )
}

#[allow(clippy::too_many_arguments)]
fn linear_rows<B: Backend + ?Sized>(
    backend: &B,
    weights: &Fixture<'_>,
    module: &str,
    shape: &[usize],
    x: &[f32],
    output: &str,
    want: &[f32],
    want_shape: &[usize],
    how: &str,
) -> Result<OpReport, OjasError> {
    let (ws, w, bias) = params(weights, module, true)?;
    let (bs, b) = bias.ok_or_else(|| bad("linear bias missing"))?;
    let y = backend.linear_bias_forward(
        &put(backend, x, shape)?,
        &put(backend, &w, &ws)?,
        &put(backend, &b, &bs)?,
    )?;
    shape_is("linear_bias output", y.shape(), want_shape)?;
    compare(
        "linear_bias_forward",
        &format!("{how} -> {output} {want_shape:?}, weight {ws:?} and bias of {module}"),
        Kind::Direct,
        &get(backend, &y)?,
        want,
        LINEAR_TOL,
    )
}

/// `linear_bias_forward` on the activation fixture's `input` with `module`'s
/// weight and bias, against `output`, with [`LINEAR_TOL`].
pub fn linear_bias<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
    module: &str,
    input: &str,
    output: &str,
) -> Result<OpReport, OjasError> {
    let (shape, x) = acts.tensor(input)?;
    let (o_shape, want) = acts.tensor(output)?;
    linear_rows(
        backend, weights, module, &shape, &x, output, &want, &o_shape, input,
    )
}

/// A block's qkv projection. The fixture records qkv's output but not its
/// input, which is the block's `norm1` output: used as it is for a global
/// block (`window` is `None`), and cut into `window x window` windows by
/// [`ojas_core::window_partition`] for a windowed one. The windowed row says
/// so in its detail, because a mismatch there could be the window order and
/// not the linear.
pub fn qkv_linear<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
    block: &str,
    window: Option<usize>,
) -> Result<OpReport, OjasError> {
    let norm = format!("{block}.norm1");
    let output = format!("{block}.attn.qkv");
    let (shape, x) = acts.tensor(&norm)?;
    let (o_shape, want) = acts.tensor(&output)?;
    let (shape, x, how) = match window {
        None => (shape, x, format!("{norm} (global, no window)")),
        Some(window) => {
            let cut = ojas_core::window_partition(backend, &put(backend, &x, &shape)?, window)?;
            (
                cut.shape().to_vec(),
                get(backend, &cut)?,
                format!("{norm} derived via window_partition({window})"),
            )
        }
    };
    linear_rows(
        backend, weights, &output, &shape, &x, &output, &want, &o_shape, &how,
    )
}

/// One convolution to compare.
#[derive(Clone, Copy, Debug)]
pub struct ConvCase<'a> {
    /// The module whose weight is used (`neck.convs.0.conv_3x3`,
    /// `trunk.patch_embed.proj`).
    pub module: &'a str,
    /// The activation fixture's input and output tensors.
    pub input: &'a str,
    pub output: &'a str,
    pub params: ojas_core::Conv2dParams,
    pub has_bias: bool,
    /// The recorded output is NHWC (the ViT patch embedding): the NCHW
    /// result is permuted to it before comparing.
    pub output_nhwc: bool,
}

/// `conv2d_forward` with the module's real weight (and bias), against torch's
/// output, with [`CONV_TOL`].
pub fn conv2d<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
    case: &ConvCase<'_>,
) -> Result<OpReport, OjasError> {
    let (shape, x) = acts.tensor(case.input)?;
    let (o_shape, want) = acts.tensor(case.output)?;
    let (ws, w, bias) = params(weights, case.module, case.has_bias)?;
    let bias_t = match &bias {
        Some((bs, b)) => Some(put(backend, b, bs)?),
        None => None,
    };
    let mut y = backend.conv2d_forward(
        &put(backend, &x, &shape)?,
        &put(backend, &w, &ws)?,
        bias_t.as_ref(),
        case.params,
    )?;
    if case.output_nhwc {
        y = backend.permute(&y, &[0, 2, 3, 1])?;
    }
    shape_is("conv2d output", y.shape(), &o_shape)?;
    compare(
        "conv2d_forward",
        &format!(
            "{} {shape:?} -> {} {o_shape:?}, weight {ws:?} of {}, stride {:?} padding {:?}{}{}",
            case.input,
            case.output,
            case.module,
            case.params.stride,
            case.params.padding,
            if case.has_bias { ", bias" } else { ", no bias" },
            if case.output_nhwc {
                ", output permuted to NHWC"
            } else {
                ""
            },
        ),
        Kind::Direct,
        &get(backend, &y)?,
        &want,
        CONV_TOL,
    )
}

/// `conv_transpose2d_forward` (kernel equal to stride) with the module's real
/// weight and bias, against torch's output, with [`CONV_TOL`].
pub fn conv_transpose2d<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
    module: &str,
    input: &str,
    output: &str,
    stride: usize,
) -> Result<OpReport, OjasError> {
    let (shape, x) = acts.tensor(input)?;
    let (o_shape, want) = acts.tensor(output)?;
    let (ws, w, bias) = params(weights, module, true)?;
    let (bs, b) = bias.ok_or_else(|| bad("conv transpose bias missing"))?;
    let y = backend.conv_transpose2d_forward(
        &put(backend, &x, &shape)?,
        &put(backend, &w, &ws)?,
        Some(&put(backend, &b, &bs)?),
        ojas_core::ConvTranspose2dParams {
            stride: [stride, stride],
            ..ojas_core::ConvTranspose2dParams::new()
        },
    )?;
    shape_is("conv_transpose2d output", y.shape(), &o_shape)?;
    compare(
        "conv_transpose2d_forward",
        &format!("{input} {shape:?} -> {output} {o_shape:?}, weight {ws:?} and bias of {module}, stride {stride}"),
        Kind::Direct,
        &get(backend, &y)?,
        &want,
        CONV_TOL,
    )
}

/// The epsilon of every LayerNorm in the ViT trunk, `ln_pre` and the blocks'
/// `norm1` and `norm2` alike: upstream builds the trunk with
/// `norm_layer="LayerNorm"`, which `vitdet.py` turns into
/// `partial(nn.LayerNorm, eps=1e-5)` (`sam3/model/vitdet.py:868-869`) and hands
/// to every `Block` (`:907`, used at `:703` and `:726`) and to `ln_pre`
/// (`:947`). With the real weights, eps 1e-5 fits the four block norms to
/// 1.4e-6 to 2.9e-6 absolute and 2.7e-5 to 1.0e-4 relative (CPU and Metal);
/// 1e-6 is off by 1.2e-5 to 2.7e-5 absolute and 5e-4 to 2e-3 relative
/// (`blocks.0.norm1`).
pub const VIT_LN_EPS: f32 = 1e-5;

/// Every LayerNorm with weights the fixture records: `ln_pre` and `norm1` and
/// `norm2` of blocks 0 and 7, all at [`VIT_LN_EPS`].
pub fn layer_norm_rows<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
) -> Result<Vec<OpReport>, OjasError> {
    let mut rows = vec![layer_norm_weighted(
        backend,
        acts,
        weights,
        "trunk.ln_pre",
        "trunk.ln_pre.out.in",
        "trunk.ln_pre.out",
        VIT_LN_EPS,
    )?];
    for block in ["trunk.blocks.0", "trunk.blocks.7"] {
        for norm in ["norm1", "norm2"] {
            rows.push(layer_norm_weighted(
                backend,
                acts,
                weights,
                &format!("{block}.{norm}"),
                &format!("{block}.{norm}.in"),
                &format!("{block}.{norm}"),
                VIT_LN_EPS,
            )?);
        }
    }
    Ok(rows)
}

/// qkv, proj, fc1 and fc2 of blocks 0 (windowed, 24x24 windows) and 7
/// (global).
pub fn linear_bias_rows<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
) -> Result<Vec<OpReport>, OjasError> {
    let mut rows = Vec::new();
    for (block, window) in [("trunk.blocks.0", Some(24)), ("trunk.blocks.7", None)] {
        rows.push(qkv_linear(backend, acts, weights, block, window)?);
        for (module, input, output) in [
            ("attn.proj", "attn.proj.in", "attn.proj"),
            ("mlp.fc1", "mlp.fc1.in", "mlp.fc1.out"),
            ("mlp.fc2", "mlp.fc2.in", "mlp.fc2"),
        ] {
            rows.push(linear_bias(
                backend,
                acts,
                weights,
                &format!("{block}.{module}"),
                &format!("{block}.{input}"),
                &format!("{block}.{output}"),
            )?);
        }
    }
    Ok(rows)
}

/// The ViT patch embedding (14x14, stride 14, no bias, NHWC output) and the
/// neck's 1x1 and 3x3 convolutions at its three scales.
pub fn conv2d_rows<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
) -> Result<Vec<OpReport>, OjasError> {
    use ojas_core::Conv2dParams;
    let mut rows = vec![conv2d(
        backend,
        acts,
        weights,
        &ConvCase {
            module: "trunk.patch_embed.proj",
            input: "trunk.patch_embed.out.in",
            output: "trunk.patch_embed.out",
            params: Conv2dParams {
                stride: [14, 14],
                ..Conv2dParams::new()
            },
            has_bias: false,
            output_nhwc: true,
        },
    )?];
    for n in 0..3 {
        let (m1, m3) = (
            format!("neck.convs.{n}.conv_1x1"),
            format!("neck.convs.{n}.conv_3x3"),
        );
        for (module, params) in [
            (&m1, Conv2dParams::new()),
            (
                &m3,
                Conv2dParams {
                    padding: [1, 1],
                    ..Conv2dParams::new()
                },
            ),
        ] {
            rows.push(conv2d(
                backend,
                acts,
                weights,
                &ConvCase {
                    module,
                    input: &format!("{module}.in"),
                    output: module,
                    params,
                    has_bias: true,
                    output_nhwc: false,
                },
            )?);
        }
    }
    Ok(rows)
}

/// The neck's three 2x2 stride-2 transposed convolutions.
pub fn conv_transpose2d_rows<B: Backend + ?Sized>(
    backend: &B,
    acts: &Fixture<'_>,
    weights: &Fixture<'_>,
) -> Result<Vec<OpReport>, OjasError> {
    [
        "neck.convs.0.dconv_2x2_0",
        "neck.convs.0.dconv_2x2_1",
        "neck.convs.1.dconv_2x2",
    ]
    .iter()
    .map(|module| {
        conv_transpose2d(
            backend,
            acts,
            weights,
            module,
            &format!("{module}.in"),
            module,
            2,
        )
    })
    .collect()
}

/// An op that [`skipped_ops`] lists and whose weights the file holds, with
/// the checkpoint tensors it needs and the test that compares it. Holding the
/// weights is not a comparison: [`weighted_ops`] reads headers only, so a
/// report built from it says the weights are present and names `compared_by`,
/// never that the op was checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checked {
    pub op: &'static str,
    pub tensors: String,
    /// The `#[ignore]`d test, in both the CPU and the Metal test file, that
    /// compares the op with torch.
    pub compared_by: &'static str,
}

/// Every weight and bias the four skipped ops need, looked up in `weights`
/// by name and shape-checked against the module they feed; an error names
/// the first one missing or misshapen.
pub fn weighted_ops(weights: &Fixture<'_>) -> Result<Vec<Checked>, OjasError> {
    let need = |module: &str, part: &str, shape: &[usize]| -> Result<(), OjasError> {
        let name = weight_name(module, part)?;
        shape_is(&name, &weights.shape(&name)?, shape)
    };
    for norm in [
        "trunk.ln_pre",
        "trunk.blocks.0.norm1",
        "trunk.blocks.0.norm2",
        "trunk.blocks.7.norm1",
        "trunk.blocks.7.norm2",
    ] {
        need(norm, "weight", &[1024])?;
        need(norm, "bias", &[1024])?;
    }
    for block in ["trunk.blocks.0", "trunk.blocks.7"] {
        for (module, out, inp) in [
            ("attn.qkv", 3072, 1024),
            ("attn.proj", 1024, 1024),
            ("mlp.fc1", 4736, 1024),
            ("mlp.fc2", 1024, 4736),
        ] {
            need(&format!("{block}.{module}"), "weight", &[out, inp])?;
            need(&format!("{block}.{module}"), "bias", &[out])?;
        }
    }
    need("trunk.patch_embed.proj", "weight", &[1024, 3, 14, 14])?;
    for (n, cin) in [(0, 256), (1, 512), (2, 1024)] {
        let m1 = format!("neck.convs.{n}.conv_1x1");
        need(&m1, "weight", &[256, cin, 1, 1])?;
        need(&m1, "bias", &[256])?;
        let m3 = format!("neck.convs.{n}.conv_3x3");
        need(&m3, "weight", &[256, 256, 3, 3])?;
        need(&m3, "bias", &[256])?;
    }
    for (module, cin, cout) in [
        ("neck.convs.0.dconv_2x2_0", 1024, 512),
        ("neck.convs.0.dconv_2x2_1", 512, 256),
        ("neck.convs.1.dconv_2x2", 1024, 512),
    ] {
        need(module, "weight", &[cin, cout, 2, 2])?;
        need(module, "bias", &[cout])?;
    }
    Ok(vec![
        Checked {
            op: "layer_norm_forward (direct)",
            tensors: "trunk.ln_pre and trunk.blocks.{0,7}.norm{1,2}: weight and bias".into(),
            compared_by: "layer_norm_with_the_true_affine_matches_torch",
        },
        Checked {
            op: "linear_bias_forward",
            tensors: "trunk.blocks.{0,7}.{attn.qkv, attn.proj, mlp.fc1, mlp.fc2}: weight and bias"
                .into(),
            compared_by: "linear_bias_matches_torch",
        },
        Checked {
            op: "conv2d_forward",
            tensors:
                "trunk.patch_embed.proj.weight, neck.convs.{0,1,2}.conv_{1x1,3x3}: weight and bias"
                    .into(),
            compared_by: "conv2d_matches_torch",
        },
        Checked {
            op: "conv_transpose2d_forward",
            tensors: "neck.convs.0.dconv_2x2_{0,1}, neck.convs.1.dconv_2x2: weight and bias".into(),
            compared_by: "conv_transpose2d_matches_torch",
        },
    ])
}

/// The ops this fixture cannot check directly because they need a weight, one
/// entry per op, with the reason read from `fixture`'s own tensor names. An
/// empty list means the file does hold weights and these ops should be added.
pub fn skipped_ops(fixture: &Fixture<'_>) -> Vec<Skipped> {
    let names = fixture.names();
    let weights: Vec<&str> = names
        .iter()
        .copied()
        .filter(|n| n.ends_with(".weight") || n.ends_with(".bias") || n.contains("weight"))
        .collect();
    if !weights.is_empty() {
        return Vec::new();
    }
    let reason = format!(
        "the fixture holds {} tensors, all activations or RoPE tables, and none is a weight or bias",
        names.len()
    );
    let skip = |op: &'static str, tensors: &str| Skipped {
        op,
        tensors: tensors.to_string(),
        reason: reason.clone(),
    };
    vec![
        skip(
            "layer_norm_forward (direct)",
            "trunk.ln_pre.out{.in,}, trunk.blocks.{0,7}.norm{1,2}{.in,}: needs norm weight and bias",
        ),
        skip(
            "linear_bias_forward",
            "trunk.blocks.{0,7}.attn.qkv, attn.proj{.in,}, mlp.fc1{.in,.out}, mlp.fc2{.in,}: needs fc and qkv weight and bias",
        ),
        skip(
            "conv2d_forward",
            "trunk.patch_embed.out{.in,} (14x14 stride 14, no bias), neck.convs.*.conv_{1x1,3x3}{.in,}: needs the conv weight",
        ),
        skip(
            "conv_transpose2d_forward",
            "neck.convs.{0,1}.dconv_2x2*{.in,}: needs the transposed-conv weight and bias",
        ),
    ]
}
