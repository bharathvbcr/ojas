//! K11's two goldens, embedded so `runga` carries them to the box, parsed
//! into the bank layout [`crate::k11_host`] steps.
//!
//! - [`DecaySensitive`]: L-oracle's decay-sensitive per-entry AdamW golden
//!   (Lappi `crates/qd-train/tests/fixtures/adamw-decay-sensitive/`, merged at
//!   Lappi 14601bc), copied byte-identically into
//!   `tests/fixtures/adamw_decay_sensitive/`. torch's f32 single-tensor AdamW
//!   through F's own builder (`MasterWeightAdamW`), five steps at lr 1e-2,
//!   per-entry lr scale 0.1/1.0, weight decay 0.01, and span-head entries with
//!   no gradient on steps 2 and 4 (D7). Its measure is L-oracle's
//!   pre-registered one: the largest `|w_t(impl) - w_t(golden)|` over steps,
//!   entries and elements, absolute, bound [`DecaySensitive::bound`] (1e-6).
//! - [`AdamwF`]: this crate's float64 torch golden over F's groups at F's lr
//!   1e-5 (`tests/fixtures/goldens/adamw_f_*`, built through F's builder by
//!   `gen_goldens.py`). At that lr a decay slip moves `p` by about
//!   `0.9 lr wd |p|` = 9e-8 per step, under any f32 bound, and
//!   `(1 - 1e-6 * 0.01) as f32 == 1.0`: the lower group's decay is invisible
//!   in f32. So a device run against it is a sanity check only; the
//!   decay-sensitive golden is what judges the semantics.
//!
//! The three tiers, for every K11 device run:
//! 1. device against [`crate::k11_host`]'s f32 emulation, bit for bit: the
//!    kernel is what the emulation says;
//! 2. the emulation against [`DecaySensitive`] within 1e-6: the semantics are
//!    torch's (host test, `tests/reference_k11_host.rs`);
//! 3. device against [`AdamwF`]'s float64 trajectory within
//!    [`ADAMW_F_PARAM_BOUND`]: a sanity check.
//!
//! The files are pinned: `tests/fixture_pins.rs` checks the copies against the
//! sha256s in their manifests, and `runga` reports the pins it was built with.

use ojas_io::{parse_json_with, JsonLimits, JsonValue, SafeTensors};

use crate::error::CudaError;
use crate::k11_host::{replay_f32, AdamwEntry, AdamwHyper, AdamwTable, Trajectory};
use crate::npy;

/// sha256 of L-oracle's `manifest.json` as merged at Lappi 14601bc: the pin
/// over the four pins it holds (`tests/fixture_pins.rs` checks the copy).
pub const DECAY_MANIFEST_SHA256: &str =
    "39882d3b5ad7ef1f3bcf013019ab544f0228068d2e16ae98b08ec185ff2cfe8e";

/// L-oracle's decay-sensitive golden, byte for byte.
pub mod decay_files {
    pub const GOLDEN: &[u8] =
        include_bytes!("../tests/fixtures/adamw_decay_sensitive/golden.safetensors");
    pub const INPUTS: &[u8] =
        include_bytes!("../tests/fixtures/adamw_decay_sensitive/inputs.safetensors");
    pub const TABLE: &[u8] = include_bytes!("../tests/fixtures/adamw_decay_sensitive/table.json");
    pub const MANIFEST: &[u8] =
        include_bytes!("../tests/fixtures/adamw_decay_sensitive/manifest.json");
}

/// The `adamw_f_*` goldens and the goldens manifest, byte for byte.
pub mod adamw_f_files {
    pub const NAMES: &[u8] = include_bytes!("../tests/fixtures/goldens/adamw_f_names.txt");
    pub const SIZES: &[u8] = include_bytes!("../tests/fixtures/goldens/adamw_f_sizes.npy");
    pub const HYPER: &[u8] = include_bytes!("../tests/fixtures/goldens/adamw_f_hyper.npy");
    pub const LR_SCALE: &[u8] = include_bytes!("../tests/fixtures/goldens/adamw_f_lr_scale.npy");
    pub const WEIGHT_DECAY: &[u8] =
        include_bytes!("../tests/fixtures/goldens/adamw_f_weight_decay.npy");
    pub const P0: &[u8] = include_bytes!("../tests/fixtures/goldens/adamw_f_p0.npy");
    pub const GRADS: &[u8] = include_bytes!("../tests/fixtures/goldens/adamw_f_grads.npy");
    pub const PARAMS: &[u8] = include_bytes!("../tests/fixtures/goldens/adamw_f_params.npy");
    pub const GRAD_SQ_NORM: &[u8] =
        include_bytes!("../tests/fixtures/goldens/adamw_f_grad_sq_norm.npy");
    pub const MANIFEST: &[u8] = include_bytes!("../tests/fixtures/goldens/manifest.json");
}

/// Limits for the small JSON files read here: each is under 30 KiB and three
/// or four levels deep.
const JSON_LIMITS: JsonLimits = JsonLimits {
    max_input_bytes: 64 << 10,
    max_depth: 8,
    max_nodes: 8192,
    ..JsonLimits::DEFAULT
};

/// The adamw_f bound on parameters, relative to the largest `|p|` of the step,
/// written before the first device run. Per step a parameter takes three f32
/// roundings of `|p| u` (the decay multiplier, the decayed value, the
/// update), and the update itself ~10 roundings of `lr u`; five steps plus the
/// f32 rounding of `p0` give about `16 u |p|` = 9.5e-7 `|p|`. 2e-6 is ~2x that.
pub const ADAMW_F_PARAM_BOUND: f64 = 2e-6;
/// The adamw_f bound on the squared gradient norm, relative. A partial sums 16
/// fused squares then 8 tree levels, and `g` itself is rounded to f32 (2u in
/// the square): about `26 u` = 1.55e-6.
pub const ADAMW_F_NORM_BOUND: f64 = 2e-6;

fn io(op: &str) -> impl Fn(ojas_io::IoError) -> CudaError + '_ {
    move |e| CudaError::invalid(op, e.detail().to_string())
}

fn json(op: &str, bytes: &[u8]) -> Result<JsonValue, CudaError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| CudaError::invalid(op, format!("not UTF-8: {e}")))?;
    parse_json_with(text, &JSON_LIMITS).map_err(io(op))
}

/// The largest absolute difference, and where.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxAbs {
    pub max_abs: f64,
    /// 1-based step; 0 when every difference is zero.
    pub at_step: usize,
    pub entry: String,
}

/// One row of L-oracle's table, as read off the built optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct DecayEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub group: String,
    pub lr_scale: f64,
    pub weight_decay: f64,
    /// Steps (1-based) this entry has a gradient on.
    pub grad_steps: Vec<usize>,
    /// Whether tessl's default decay mask would exclude it (mutation D1).
    pub tessl_default_excludes: bool,
    /// torch's per-entry step count after the run.
    pub adamw_steps_taken: u64,
}

/// L-oracle's decay-sensitive golden, laid out as one flat bank in table
/// order.
#[derive(Clone, Debug)]
pub struct DecaySensitive {
    pub entries: Vec<DecayEntry>,
    pub table: AdamwTable,
    /// lr, betas and eps; the same for every entry (checked).
    pub hyper: AdamwHyper,
    pub steps: usize,
    /// The pre-registered measure's bound.
    pub bound: f64,
    /// The manifest's pins: `(file, sha256)`.
    pub pins: Vec<(String, String)>,
    pub init: Vec<f32>,
    /// Per step: the gradient bank (zero where inactive) and the active flags.
    pub grads: Vec<(Vec<f32>, Vec<bool>)>,
    /// Per step: the golden fp32 masters, the whole bank.
    pub golden_w: Vec<Vec<f32>>,
}

impl DecaySensitive {
    /// From the embedded files.
    pub fn embedded() -> Result<Self, CudaError> {
        Self::parse(
            decay_files::TABLE,
            decay_files::MANIFEST,
            decay_files::INPUTS,
            decay_files::GOLDEN,
        )
    }

    pub fn parse(
        table_json: &[u8],
        manifest_json: &[u8],
        inputs: &[u8],
        golden: &[u8],
    ) -> Result<Self, CudaError> {
        const OP: &str = "DecaySensitive";
        let manifest = json(OP, manifest_json)?;
        let opt = manifest.get_object("optimizer").map_err(io(OP))?;
        let lr = opt.get_f64("lr").map_err(io(OP))?;
        let steps = usize::try_from(opt.get_u64("steps").map_err(io(OP))?)
            .map_err(|e| CudaError::invalid(OP, e.to_string()))?;
        if steps == 0 || steps > 64 {
            return Err(CudaError::invalid(OP, format!("{steps} steps")));
        }
        let bound = manifest
            .get_object("measure")
            .and_then(|m| m.get_f64("bound"))
            .map_err(io(OP))?;
        let mut pins = Vec::new();
        for (file, v) in manifest
            .get_object("files")
            .map_err(io(OP))?
            .as_object()
            .map_err(io(OP))?
        {
            pins.push((
                file.clone(),
                v.get_str("sha256").map_err(io(OP))?.to_string(),
            ));
        }

        let table_v = json(OP, table_json)?;
        let mut entries = Vec::new();
        let mut shared: Option<(f64, f64, f64)> = None;
        for (name, row) in table_v.as_object().map_err(io(OP))? {
            let at =
                |e: ojas_io::IoError| CudaError::invalid(OP, format!("{name}: {}", e.detail()));
            let shape = row
                .get_array("shape")
                .map_err(at)?
                .iter()
                .map(|d| d.as_u64().map_err(at).and_then(|d| usize_of(OP, d)))
                .collect::<Result<Vec<_>, _>>()?;
            let betas = row.get_array("betas").map_err(at)?;
            if betas.len() != 2 {
                return Err(CudaError::invalid(OP, format!("{name}: betas {betas:?}")));
            }
            let b = (
                betas[0].as_f64().map_err(at)?,
                betas[1].as_f64().map_err(at)?,
                row.get_f64("eps").map_err(at)?,
            );
            match shared {
                None => shared = Some(b),
                Some(s) if s.0.to_bits() == b.0.to_bits()
                    && s.1.to_bits() == b.1.to_bits()
                    && s.2.to_bits() == b.2.to_bits() => {}
                Some(s) => {
                    return Err(CudaError::invalid(
                        OP,
                        format!("{name}: betas/eps {b:?} differ from {s:?}; one AdamwHyper cannot hold both"),
                    ))
                }
            }
            let grad_steps = row
                .get_array("grad_steps")
                .map_err(at)?
                .iter()
                .map(|t| t.as_u64().map_err(at).and_then(|t| usize_of(OP, t)))
                .collect::<Result<Vec<_>, _>>()?;
            if grad_steps.iter().any(|&t| t == 0 || t > steps) {
                return Err(CudaError::invalid(
                    OP,
                    format!("{name}: grad_steps {grad_steps:?}"),
                ));
            }
            entries.push(DecayEntry {
                name: name.clone(),
                shape,
                group: row.get_str("group").map_err(at)?.to_string(),
                lr_scale: row.get_f64("lr_scale").map_err(at)?,
                weight_decay: row.get_f64("weight_decay").map_err(at)?,
                grad_steps,
                tessl_default_excludes: row.get_bool("tessl_default_excludes").map_err(at)?,
                adamw_steps_taken: row.get_u64("adamw_steps_taken").map_err(at)?,
            });
        }
        let (beta1, beta2, eps) =
            shared.ok_or_else(|| CudaError::invalid(OP, "the table has no entries"))?;
        let hyper = AdamwHyper {
            lr,
            beta1,
            beta2,
            eps,
            grad_scale: 1.0,
        };
        hyper.validate()?;

        let mut offset = 0usize;
        let mut rows = Vec::with_capacity(entries.len());
        for e in &entries {
            let len = numel(OP, &e.name, &e.shape)?;
            rows.push(AdamwEntry {
                name: e.name.clone(),
                offset,
                len,
                lr_scale: e.lr_scale,
                weight_decay: e.weight_decay,
            });
            offset += len;
        }
        let table = AdamwTable::new(rows, offset)?;

        let inp = SafeTensors::parse(inputs).map_err(io(OP))?;
        let gold = SafeTensors::parse(golden).map_err(io(OP))?;
        let read = |st: &SafeTensors<'_>, name: &str, e: &AdamwEntry, shape: &[usize]| {
            let (s, v) = st.read_f32(name).map_err(io(OP))?;
            let s: Vec<usize> = s
                .iter()
                .map(|&d| usize_of(OP, d))
                .collect::<Result<_, _>>()?;
            if s != shape || v.len() != e.len {
                return Err(CudaError::invalid(
                    OP,
                    format!("{name}: shape {s:?}, the table says {shape:?}"),
                ));
            }
            Ok(v)
        };
        let mut init = vec![0.0f32; table.bank_len()];
        for (e, d) in table.entries().iter().zip(&entries) {
            let v = read(&inp, &format!("init.{}", e.name), e, &d.shape)?;
            init[e.offset..e.offset + e.len].copy_from_slice(&v);
        }
        let mut grads = Vec::with_capacity(steps);
        let mut golden_w = Vec::with_capacity(steps);
        for t in 1..=steps {
            let mut g = vec![0.0f32; table.bank_len()];
            let mut active = vec![false; entries.len()];
            let mut w = vec![0.0f32; table.bank_len()];
            for (i, (e, d)) in table.entries().iter().zip(&entries).enumerate() {
                let name = format!("grad.step{t}.{}", e.name);
                let present = inp.info(&name).is_ok();
                if present != d.grad_steps.contains(&t) {
                    return Err(CudaError::invalid(
                        OP,
                        format!(
                            "{name}: present {present}, the table's grad_steps {:?}",
                            d.grad_steps
                        ),
                    ));
                }
                if present {
                    g[e.offset..e.offset + e.len].copy_from_slice(&read(&inp, &name, e, &d.shape)?);
                    active[i] = true;
                }
                let wv = read(&gold, &format!("w.step{t}.{}", e.name), e, &d.shape)?;
                w[e.offset..e.offset + e.len].copy_from_slice(&wv);
            }
            grads.push((g, active));
            golden_w.push(w);
        }
        Ok(DecaySensitive {
            entries,
            table,
            hyper,
            steps,
            bound,
            pins,
            init,
            grads,
            golden_w,
        })
    }

    /// The pre-registered measure over a trajectory `w[t-1]` = the bank
    /// after step `t`.
    pub fn measure(&self, w: &[Vec<f32>]) -> Result<MaxAbs, CudaError> {
        measure(&self.table, w, &self.golden_w)
    }

    /// The f32 emulation's trajectory, and its per-entry step counts.
    pub fn replay_f32(&self) -> Result<(Vec<Vec<f32>>, Vec<u64>), CudaError> {
        let t = self.emulate()?;
        Ok((t.w, t.steps))
    }

    /// The whole emulated run: what the device must equal bit for bit.
    pub fn emulate(&self) -> Result<Trajectory, CudaError> {
        replay_f32(&self.table, &self.hyper, &self.init, &self.grads)
    }
}

/// The largest `|got - want|` over steps, entries and elements.
pub fn measure(
    table: &AdamwTable,
    got: &[Vec<f32>],
    want: &[Vec<f32>],
) -> Result<MaxAbs, CudaError> {
    if got.len() != want.len() {
        return Err(CudaError::invalid(
            "measure",
            format!("{} steps against {}", got.len(), want.len()),
        ));
    }
    let mut worst = MaxAbs {
        max_abs: 0.0,
        at_step: 0,
        entry: String::new(),
    };
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        if g.len() != table.bank_len() || w.len() != table.bank_len() {
            return Err(CudaError::invalid("measure", "a bank of the wrong length"));
        }
        for e in table.entries() {
            for k in e.offset..e.offset + e.len {
                let d = (f64::from(g[k]) - f64::from(w[k])).abs();
                // A NaN anywhere is the worst possible result.
                if d.is_nan() || d > worst.max_abs {
                    worst = MaxAbs {
                        max_abs: if d.is_nan() { f64::INFINITY } else { d },
                        at_step: t + 1,
                        entry: e.name.clone(),
                    };
                }
            }
        }
    }
    Ok(worst)
}

/// The float64 golden over F's groups at F's lr.
#[derive(Clone, Debug)]
pub struct AdamwF {
    pub names: Vec<String>,
    pub table: AdamwTable,
    pub hyper: AdamwHyper,
    pub steps: usize,
    pub p0: Vec<f64>,
    /// Per step, the whole gradient bank.
    pub grads: Vec<Vec<f64>>,
    /// Per step, the parameters after it.
    pub params: Vec<Vec<f64>>,
    pub grad_sq_norm: Vec<f64>,
    /// `adamw_f_*` pins from the goldens manifest: `(file, sha256)`.
    pub pins: Vec<(String, String)>,
    /// What the manifest records of F's builder (checked by [`f_builder_of`]).
    pub f_builder: FBuilder,
}

fn npy_f64(op: &str, what: &str, bytes: &[u8]) -> Result<npy::Npy, CudaError> {
    let a = npy::parse(bytes).map_err(|e| CudaError::invalid(op, format!("{what}: {e}")))?;
    a.f64s()
        .map_err(|e| CudaError::invalid(op, format!("{what}: {e}")))?;
    Ok(a)
}

impl AdamwF {
    pub fn embedded() -> Result<Self, CudaError> {
        use adamw_f_files as f;
        const OP: &str = "AdamwF";
        let names: Vec<String> = std::str::from_utf8(f::NAMES)
            .map_err(|e| CudaError::invalid(OP, format!("names: {e}")))?
            .lines()
            .map(str::to_string)
            .collect();
        let sizes =
            npy::parse(f::SIZES).map_err(|e| CudaError::invalid(OP, format!("sizes: {e}")))?;
        let sizes: Vec<usize> = sizes
            .i64s()
            .map_err(|e| CudaError::invalid(OP, format!("sizes: {e}")))?
            .iter()
            .map(|&s| {
                usize::try_from(s).map_err(|e| CudaError::invalid(OP, format!("size {s}: {e}")))
            })
            .collect::<Result<_, _>>()?;
        let hyper = npy_f64(OP, "hyper", f::HYPER)?;
        let h = hyper.f64s().map_err(|e| CudaError::invalid(OP, e))?;
        if h.len() != 4 {
            return Err(CudaError::invalid(
                OP,
                format!("hyper has {} values, want [lr, beta1, beta2, eps]", h.len()),
            ));
        }
        let hyper = AdamwHyper {
            lr: h[0],
            beta1: h[1],
            beta2: h[2],
            eps: h[3],
            grad_scale: 1.0,
        };
        hyper.validate()?;
        let lr_scale = npy_f64(OP, "lr_scale", f::LR_SCALE)?;
        let wd = npy_f64(OP, "weight_decay", f::WEIGHT_DECAY)?;
        let (lr_scale, wd) = (
            lr_scale.f64s().map_err(|e| CudaError::invalid(OP, e))?,
            wd.f64s().map_err(|e| CudaError::invalid(OP, e))?,
        );
        if names.len() != sizes.len() || lr_scale.len() != sizes.len() || wd.len() != sizes.len() {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "{} names, {} sizes, {} lr scales, {} decays",
                    names.len(),
                    sizes.len(),
                    lr_scale.len(),
                    wd.len()
                ),
            ));
        }
        let mut offset = 0usize;
        let mut rows = Vec::with_capacity(sizes.len());
        for i in 0..sizes.len() {
            rows.push(AdamwEntry {
                name: names[i].clone(),
                offset,
                len: sizes[i],
                lr_scale: lr_scale[i],
                weight_decay: wd[i],
            });
            offset += sizes[i];
        }
        let table = AdamwTable::new(rows, offset)?;
        let n = table.bank_len();
        let p0 = npy_f64(OP, "p0", f::P0)?
            .f64s()
            .map_err(|e| CudaError::invalid(OP, e))?
            .to_vec();
        if p0.len() != n {
            return Err(CudaError::invalid(
                OP,
                format!("p0 holds {}, the bank {n}", p0.len()),
            ));
        }
        let per_step = |what: &str, bytes: &[u8]| -> Result<Vec<Vec<f64>>, CudaError> {
            let a = npy_f64(OP, what, bytes)?;
            if a.shape.len() != 2 || a.shape[1] != n {
                return Err(CudaError::invalid(
                    OP,
                    format!("{what}: shape {:?}, want [steps, {n}]", a.shape),
                ));
            }
            let v = a.f64s().map_err(|e| CudaError::invalid(OP, e))?;
            Ok(v.chunks(n).map(<[f64]>::to_vec).collect())
        };
        let grads = per_step("grads", f::GRADS)?;
        let params = per_step("params", f::PARAMS)?;
        let sq = npy_f64(OP, "grad_sq_norm", f::GRAD_SQ_NORM)?
            .f64s()
            .map_err(|e| CudaError::invalid(OP, e))?
            .to_vec();
        let steps = grads.len();
        if params.len() != steps || sq.len() != steps || steps == 0 {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "{steps} gradient steps, {} parameter steps, {} norms",
                    params.len(),
                    sq.len()
                ),
            ));
        }
        let manifest = json(OP, f::MANIFEST)?;
        let case = manifest
            .get_object("cases")
            .and_then(|c| c.get_object("adamw_f"))
            .map_err(io(OP))?;
        let f_builder = f_builder_of(case).map_err(|e| CudaError::invalid(OP, e))?;
        if hyper.lr.to_bits() != f_builder.base_lr.to_bits() {
            return Err(CudaError::invalid(
                OP,
                format!(
                    "hyper lr {} is not the builder's base lr {}",
                    hyper.lr, f_builder.base_lr
                ),
            ));
        }
        for e in table.entries() {
            let ok = f_builder.groups.iter().any(|g| {
                g.lr_scale.to_bits() == e.lr_scale.to_bits()
                    && g.weight_decay.to_bits() == e.weight_decay.to_bits()
            });
            if !ok {
                return Err(CudaError::invalid(
                    OP,
                    format!(
                        "{}: lr_scale {} / weight_decay {} is no group F's builder made",
                        e.name, e.lr_scale, e.weight_decay
                    ),
                ));
            }
        }
        let mut pins = Vec::new();
        for (file, v) in manifest
            .get_object("files")
            .map_err(io(OP))?
            .as_object()
            .map_err(io(OP))?
        {
            if file.starts_with("adamw_f_") {
                pins.push((file.clone(), v.as_str().map_err(io(OP))?.to_string()));
            }
        }
        if pins.is_empty() {
            return Err(CudaError::invalid(
                OP,
                "the goldens manifest pins no adamw_f_ file",
            ));
        }
        Ok(AdamwF {
            names,
            table,
            hyper,
            steps,
            p0,
            grads,
            params,
            grad_sq_norm: sq,
            pins,
            f_builder,
        })
    }

    /// The device's inputs: `p0` and each step's gradients rounded to f32.
    pub fn p0_f32(&self) -> Vec<f32> {
        self.p0.iter().map(|&x| x as f32).collect()
    }

    pub fn grads_f32(&self, step: usize) -> Vec<f32> {
        self.grads[step].iter().map(|&x| x as f32).collect()
    }

    /// Each step's f32 gradient bank, every entry active.
    pub fn grads_with_flags(&self) -> Vec<(Vec<f32>, Vec<bool>)> {
        (0..self.steps)
            .map(|t| (self.grads_f32(t), vec![true; self.table.entries().len()]))
            .collect()
    }

    /// The f32 emulation's trajectory and per-step squared norms (partials of
    /// every window, summed in f64 in order).
    pub fn replay_f32(&self) -> Result<(Vec<Vec<f32>>, Vec<f64>), CudaError> {
        let t = self.emulate()?;
        Ok((t.w, t.norms))
    }

    /// The whole emulated run from the f32-rounded inputs.
    pub fn emulate(&self) -> Result<Trajectory, CudaError> {
        replay_f32(
            &self.table,
            &self.hyper,
            &self.p0_f32(),
            &self.grads_with_flags(),
        )
    }

    /// The largest `|p_f32 - p_f64|` over the run, relative to the largest
    /// `|p_f64|` of its step.
    pub fn param_rel_err(&self, traj: &[Vec<f32>]) -> Result<f64, CudaError> {
        if traj.len() != self.steps {
            return Err(CudaError::invalid(
                "AdamwF::param_rel_err",
                "trajectory length",
            ));
        }
        let mut worst = 0.0f64;
        for (got, want) in traj.iter().zip(&self.params) {
            let scale = want.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            for (&g, &w) in got.iter().zip(want) {
                let d = (f64::from(g) - w).abs() / scale;
                if d.is_nan() {
                    return Ok(f64::INFINITY);
                }
                worst = worst.max(d);
            }
        }
        Ok(worst)
    }

    /// The largest relative error of per-step squared norms.
    pub fn norm_rel_err(&self, norms: &[f64]) -> Result<f64, CudaError> {
        if norms.len() != self.steps {
            return Err(CudaError::invalid("AdamwF::norm_rel_err", "norm count"));
        }
        let mut worst = 0.0f64;
        for (&g, &w) in norms.iter().zip(&self.grad_sq_norm) {
            let d = (g - w).abs() / w.abs();
            if d.is_nan() {
                return Ok(f64::INFINITY);
            }
            worst = worst.max(d);
        }
        Ok(worst)
    }
}

/// F's optimizer flags, as `gen_goldens.py` checks them against the text of
/// Lappi's `campaign/f-v4-preregistered.json`.
pub const F_FLAGS: [(&str, &str); 4] = [
    ("--lower-layers-lr-scale", "0.1"),
    ("--lower-layers-n", "8"),
    ("--lr", "1e-5"),
    ("--optimizer", "master"),
];

/// One parameter group as F's builder made it.
#[derive(Clone, Debug, PartialEq)]
pub struct FGroup {
    pub name: String,
    pub lr: f64,
    pub lr_scale: f64,
    pub weight_decay: f64,
}

/// What the goldens manifest records of F's builder for `adamw_f`.
#[derive(Clone, Debug, PartialEq)]
pub struct FBuilder {
    pub built_class: String,
    pub inner_class: String,
    pub base_lr: f64,
    pub groups: Vec<FGroup>,
}

/// Refuse an `adamw_f` manifest case that was not built through F's own
/// builder (GAP-OJAS-K11-GOLDEN-RETYPES-F-HYPER-2026-10-01): it must record
/// `f_builder` with `MasterWeightAdamW` over torch `AdamW`, exactly
/// [`F_FLAGS`], a base lr equal to the flags' `--lr`, and every group at
/// `lr = base_lr * lr_scale` (Lappi's `apply_lr`) with the builder's decay
/// 0.01, single-tensor (neither `foreach` nor `fused` set).
pub fn f_builder_of(case: &JsonValue) -> Result<FBuilder, String> {
    fn e(what: &'static str) -> impl Fn(ojas_io::IoError) -> String {
        move |err| format!("{what}: {}", err.detail())
    }
    let fb = case
        .field_opt("f_builder")
        .map_err(e("adamw_f"))?
        .ok_or("adamw_f records no f_builder: the golden was not built through F's builder")?;
    let built_class = fb
        .get_str("built_class")
        .map_err(e("f_builder"))?
        .to_string();
    let inner_class = fb
        .get_str("inner_class")
        .map_err(e("f_builder"))?
        .to_string();
    if built_class != "MasterWeightAdamW" || inner_class != "AdamW" {
        return Err(format!(
            "F's builder makes MasterWeightAdamW over AdamW; the golden records {built_class} over {inner_class}"
        ));
    }
    let flags = fb.get_object("f_flags").map_err(e("f_builder"))?;
    flags
        .deny_unknown_keys(&F_FLAGS.map(|(k, _)| k))
        .map_err(e("f_flags"))?;
    for (k, want) in F_FLAGS {
        let got = flags.get_str(k).map_err(e("f_flags"))?;
        if got != want {
            return Err(format!("f_flags {k} is {got:?}; F's is {want:?}"));
        }
    }
    let base_lr = case.get_f64("base_lr").map_err(e("adamw_f"))?;
    let flag_lr: f64 = F_FLAGS[2].1.parse().map_err(|err| format!("--lr: {err}"))?;
    if base_lr.to_bits() != flag_lr.to_bits() {
        return Err(format!("base_lr {base_lr} is not F's --lr {flag_lr}"));
    }
    let mut groups = Vec::new();
    for g in fb.get_array("groups").map_err(e("f_builder"))? {
        let name = g.get_str("name").map_err(e("group"))?.to_string();
        let at = |err: ojas_io::IoError| format!("group {name}: {}", err.detail());
        let lr = g.get_f64("lr").map_err(at)?;
        let lr_scale = g.get_f64("lr_scale").map_err(at)?;
        let weight_decay = g.get_f64("weight_decay").map_err(at)?;
        if lr.to_bits() != (base_lr * lr_scale).to_bits() {
            return Err(format!(
                "group {name}: lr {lr} is not base_lr * lr_scale = {}",
                base_lr * lr_scale
            ));
        }
        if weight_decay.to_bits() != 0.01f64.to_bits() {
            return Err(format!(
                "group {name}: weight_decay {weight_decay}, the builder's is 0.01"
            ));
        }
        for k in ["foreach", "fused"] {
            let v = g.field(k).map_err(at)?;
            if !(v.is_null() || matches!(v.as_bool(), Ok(false))) {
                return Err(format!(
                    "group {name}: {k} is set; the golden is single-tensor"
                ));
            }
        }
        groups.push(FGroup {
            name,
            lr,
            lr_scale,
            weight_decay,
        });
    }
    if groups.is_empty() {
        return Err("f_builder records no groups".to_string());
    }
    Ok(FBuilder {
        built_class,
        inner_class,
        base_lr,
        groups,
    })
}

fn usize_of(op: &str, x: u64) -> Result<usize, CudaError> {
    usize::try_from(x).map_err(|e| CudaError::invalid(op, format!("{x}: {e}")))
}

fn numel(op: &str, name: &str, shape: &[usize]) -> Result<usize, CudaError> {
    shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .filter(|&n| n > 0)
        .ok_or_else(|| CudaError::invalid(op, format!("{name}: shape {shape:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_case_text() -> String {
        let m = json("test", adamw_f_files::MANIFEST).unwrap();
        let case = m
            .get_object("cases")
            .unwrap()
            .get_object("adamw_f")
            .unwrap();
        to_ours(case).render()
    }

    /// Re-render through this crate's writer so a test can edit the text.
    fn to_ours(v: &JsonValue) -> crate::json::Json {
        use crate::json::{Json, JsonObj};
        match v {
            JsonValue::Null => Json::Null,
            JsonValue::Bool(b) => Json::from(*b),
            JsonValue::Number(ojas_io::JsonNumber::U64(n)) => Json::from(*n),
            JsonValue::Number(ojas_io::JsonNumber::I64(n)) => Json::from(*n),
            JsonValue::Number(ojas_io::JsonNumber::F64(x)) => Json::from(*x),
            JsonValue::String(s) => Json::from(s.as_str()),
            JsonValue::Array(a) => Json::Arr(a.iter().map(to_ours).collect()),
            JsonValue::Object(o) => {
                let mut obj = JsonObj::new();
                for (k, x) in o {
                    obj.push(k, to_ours(x));
                }
                obj.into()
            }
        }
    }

    fn check(text: &str) -> Result<FBuilder, String> {
        f_builder_of(&parse_json_with(text, &JSON_LIMITS).unwrap())
    }

    #[test]
    fn the_embedded_goldens_parse_and_name_fs_builder() {
        let ds = DecaySensitive::embedded().unwrap();
        assert_eq!((ds.steps, ds.entries.len(), ds.bound), (5, 19, 1e-6));
        assert_eq!(ds.hyper.lr, 1e-2);
        assert_eq!(ds.pins.len(), 4);
        let f = AdamwF::embedded().unwrap();
        assert_eq!(f.f_builder.built_class, "MasterWeightAdamW");
        assert_eq!(f.f_builder.groups.len(), 2);
        assert_eq!(f.hyper.lr, 1e-5);
        assert_eq!(f.pins.len(), 9, "every adamw_f_ file is pinned");
    }

    #[test]
    fn the_live_adamw_f_case_passes_and_its_retypings_do_not() {
        let live = live_case_text();
        check(&live).unwrap();
        for (from, to, why) in [
            ("\"MasterWeightAdamW\"", "\"AdamW\"", "built_class"),
            ("\"1e-5\"", "\"1e-4\"", "--lr flag"),
            (
                "\"--lower-layers-n\":\"8\"",
                "\"--lower-layers-n\":\"6\"",
                "lower layers",
            ),
            (
                "\"lr\":1e-5",
                "\"lr\":1.1e-5",
                "a group lr not from apply_lr",
            ),
            (
                "\"weight_decay\":0.01",
                "\"weight_decay\":0.1",
                "decay retyped",
            ),
            ("\"fused\":null", "\"fused\":true", "fused"),
        ] {
            assert!(
                live.contains(from),
                "{why}: sample {from} not in the live case"
            );
            let bad = live.replacen(from, to, 1);
            assert!(check(&bad).is_err(), "{why}: the retyped case passed");
        }
    }

    /// The pre-fix case, before `gen_goldens.py` built `adamw_f` through F's
    /// builder (keys as in `AUDIT/ojas-training-2026-10-01/
    /// l-cuda-m1-goldens-manifest-diff.txt`, abridged): no `f_builder`.
    #[test]
    fn the_pre_fix_adamw_f_case_is_refused() {
        let pre = r#"{"base_lr": 1e-05, "betas": [0.9, 0.999], "eps": 1e-08,
            "lappi_optim": "/Users/bharath/Code/research/Lappi-decision/python/qd_train/optim.py",
            "lower_layers_n": 8, "lower_lr_scale": 0.1, "seed": 7,
            "source": "torch.optim.AdamW(foreach=False, fused=False) single-tensor path, torch/optim/adam.py:416-545",
            "steps": 5, "weight_decay": 0.01}"#;
        let err = check(pre).unwrap_err();
        assert!(err.contains("no f_builder"), "{err}");
    }

    #[test]
    fn f32_quantizes_the_decay_multiplier_at_fs_lr() {
        // The lower group (lr 1e-6) does not decay at all in f32 ...
        assert_eq!((1.0 - 1e-6 * 0.01f64) as f32, 1.0f32);
        // ... and the base group's 1 - 1e-7 lands on 1 - 2^-23, not 1 - 1e-7.
        assert_eq!((1.0 - 1e-5 * 0.01f64) as f32, 1.0f32 - f32::EPSILON);
    }
}
