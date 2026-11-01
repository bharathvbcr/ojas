//! The paired A/B of the two Qwen3.5 training engines on Metal: ojas's
//! `TapeStepProvider` (`ojas-model/src/qwen35/step_provider.rs`, the Tape
//! tower) against the tessl whole-step provider (`ojas_qwen35::Qwen35Step`),
//! the gate of unify-plan section 4 (owner decision D5). `#[ignore]` and named
//! `gpu_*`/`ab_*`: run under the machine's heavy-job lock, one arm per process
//! (the device figures are process-wide).
//!
//! # What is compared
//!
//! A full-gradient f32 step, because that is the only step both engines
//! have: every Hugging Face tensor takes a gradient and an AdamW update
//! (`TapeStepProvider::open_full`, "to compare against the tessl provider
//! entry for entry"; `Qwen35Step` with `Precision::F32`,
//! `Numerics::ExactF32`). The LoRA step has no tessl counterpart, so the
//! `tape-lora` rows of `bench/results/2026-10-21-lora-stepcost-ab` are not
//! this comparison.
//!
//! Both arms see the same `tokens` ids (one deterministic xorshift stream, as
//! `gpu_memory_rows.rs`), forward over the first `tokens - 1` of them, score
//! every position against the next id at scale `1 / (tokens - 1)`, micro-batch
//! 1, and take one AdamW step with the same scalars (lr 1e-5, betas 0.9 and
//! 0.999, eps 1e-8, weight decay 0, grad scale 1, no clipping). The Tape arm
//! checkpoints every layer (`AB_CKPT=on`, the default); the tessl step keeps
//! its whole pending state and has no switch, so the two peaks are the two
//! engines' own configurations, not one policy.
//!
//! One warmup step (its wall includes shader compilation and first-use
//! allocation), then three timed steps. A step's wall is forward, backward
//! and AdamW through the device finishing: the Tape arm syncs the backend, the
//! tessl arm reads the final norm back (tessl's per-entry host reads wait for
//! prior GPU work). The gradient dump of the warmup step runs between the
//! backward and the AdamW step and is outside the wall (`dump_ms` is its
//! cost).
//!
//! # Gradient parity
//!
//! `AB_DUMP=1` makes the warmup step write the bank's gradients to
//! `$AB_OUT/<arm>.grads` and the loss to `$AB_OUT/<arm>.meta`: every 1-D
//! tensor, every tensor of layers 0, 3 and the last, the final norm, and the
//! tied embedding's rows for every id in the batch plus every 4099th row (the
//! 2B gate's selection in kind). `ab_compare_dumps` then reads both and
//! prints one row: the loss's relative difference and, per tensor, the worst
//! `|tape - tessl| / max|tessl|`, against the 2B gate's bounds (loss 1e-4,
//! gradient 1e-2 of its peak). The first step starts from identical weights,
//! so it is the parity step; later steps differ by one AdamW trajectory each
//! and are timing only (their losses are in the row).
//!
//! # Running
//!
//! `scripts/ab_provider.sh` runs the arms interleaved with the order rotated
//! each rep, each arm one process under `mac_heavy.sh`. One arm by hand:
//!
//! ```text
//! AB_ARM=tape AB_TOKENS=2048 AB_OUT=/tmp/ojas-ab AB_DUMP=1 \
//! cargo test -p ojas-qwen35 --release --test gpu_ab_provider \
//!     -- --ignored --test-threads=1 --nocapture gpu_ab_arm
//! ```
//!
//! The snapshot is `QWEN35_AB_SNAPSHOT`, or the one snapshot of
//! `Qwen/Qwen3.5-0.8B-Base` in the Hugging Face cache. A refusal at any phase
//! is a row too (`refused`). `gpu_ab_tiny_end_to_end` runs both arms and the
//! comparison on the two-layer fixture in one process, to check this harness
//! itself before it is spent on the real shape.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_core::{Backend, Budget, DType, OjasError};
use ojas_io::sha256::open_sha256_hex;
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use ojas_model::preflight::{Calibration, Point};
use ojas_model::qwen35::adapters::BaseIdentity;
use ojas_model::qwen35::{
    fuse_grads, load_hf, DeviceHooks, OpenConfig, Qwen35Params, TapeStepProvider,
};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_qwen35::{
    AdamWHyper, GroupSpec, LrRule, Numerics, OptimizerPlan, Precision, Qwen35Step, Select,
    Sequence, Snapshot, WdRule, Which,
};
use ojas_train::lappi::step as lstep;
use ojas_train::lappi::step::StepProvider as _;

use common::{footprint_or_refuse, refused, single_weights_file, Row};

const GIB: u64 = 1 << 30;
/// Below memguard's 32 GiB `phys_footprint` kill, so a misfit is a typed
/// refusal (the Tape arm's device budget), not a SIGTERM.
const DEVICE_BUDGET: u64 = 28 * GIB;
const TIMED_STEPS: usize = 3;
const LR: f64 = 1e-5;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
/// The 2B gate's bounds against transformers (`gpu_tape_2b.rs`), reused for
/// Tape against tessl.
const LOSS_BOUND: f64 = 1e-4;
const GRAD_BOUND: f64 = 1e-2;
/// The tied embedding's gradient is dumped for these rows (plus the batch's
/// ids).
const EMBED_ROW_STRIDE: usize = 4099;

type Fault = (&'static str, String);

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// The one snapshot of `Qwen/Qwen3.5-0.8B-Base` in the Hugging Face cache, or
/// `QWEN35_AB_SNAPSHOT`.
fn snapshot_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("QWEN35_AB_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let root = PathBuf::from(std::env::var_os("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots");
    let dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    assert_eq!(
        dirs.len(),
        1,
        "expected one snapshot under {}",
        root.display()
    );
    dirs[0].clone()
}

/// `n` deterministic ids below `min(150000, vocab)`, skipping the reserved
/// vision tokens: `gpu_memory_rows.rs`'s stream, so the rows compare.
fn ids(n: usize, vocab: u32, reserved: &[u32]) -> Vec<u32> {
    let bound = u64::from(vocab.min(150_000));
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let id = (s % bound) as u32;
        if !reserved.contains(&id) {
            out.push(id);
        }
    }
    out
}

/// The base's file name and sha256 for `TapeStepProvider`'s pinned load (a
/// Hugging Face cache blob is named by its sha256; any other file is hashed).
fn base_identity(path: &Path) -> BaseIdentity {
    let blob = std::fs::read_link(path).ok().and_then(|l| {
        l.file_name()
            .and_then(|n| n.to_str())
            .filter(|n| n.len() == 64 && n.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
            .map(str::to_string)
    });
    let sha256 = blob.unwrap_or_else(|| open_sha256_hex(path).unwrap().1);
    BaseIdentity {
        file_name: path.file_name().unwrap().to_string_lossy().into_owned(),
        sha256,
        quantizer: None,
    }
}

// ---- the gradient dump -------------------------------------------------------

/// `layers.{l}.` of a name below the tower prefix.
fn layer_of(short: &str) -> Option<usize> {
    short
        .strip_prefix("layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Whether a non-embedding tensor is in the dump: every 1-D tensor, the final
/// norm and every tensor of layers 0, 3 and the last.
fn kept(short: &str, shape: &[usize], layers: usize) -> bool {
    if short == "embed_tokens.weight" {
        return false;
    }
    if shape.len() == 1 {
        return true;
    }
    match layer_of(short) {
        Some(l) => l == 0 || l == 3 || l + 1 == layers,
        None => true,
    }
}

/// The embedding rows in the dump, ascending: the batch's ids and every
/// `EMBED_ROW_STRIDE`-th row.
fn embed_rows(ids: &[u32], vocab: usize) -> Vec<usize> {
    let mut rows: BTreeSet<usize> = ids
        .iter()
        .map(|&i| i as usize)
        .filter(|&i| i < vocab)
        .collect();
    rows.extend((0..vocab).step_by(EMBED_ROW_STRIDE));
    rows.into_iter().collect()
}

/// `values` of `rows` (each `width` wide) out of a `[vocab, width]` tensor.
fn gather_rows(values: &[f32], width: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|&r| values[r * width..][..width].iter().copied())
        .collect()
}

type Entries = Vec<(String, Vec<f32>)>;

/// Little-endian records: `u32` name length, the name, `u64` count, `count`
/// `f32`s.
fn write_dump(path: &Path, entries: &Entries) -> Result<(), String> {
    let mut out = Vec::new();
    for (name, values) in entries {
        out.extend((name.len() as u32).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((values.len() as u64).to_le_bytes());
        for v in values {
            out.extend(v.to_le_bytes());
        }
    }
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_dump(path: &Path) -> Entries {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut at = 0usize;
    let mut out = Vec::new();
    while at < bytes.len() {
        let name_len = u32::from_le_bytes(take(&bytes, &mut at, 4).try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&bytes, &mut at, name_len).to_vec()).unwrap();
        let count = u64::from_le_bytes(take(&bytes, &mut at, 8).try_into().unwrap()) as usize;
        let values = take(&bytes, &mut at, count * 4)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.push((name, values));
    }
    out
}

/// The next `n` bytes of `bytes` from `*at`; a short file is a panic naming
/// the shortfall, not a silent truncation.
fn take<'a>(bytes: &'a [u8], at: &mut usize, n: usize) -> &'a [u8] {
    let end = at
        .checked_add(n)
        .filter(|&e| e <= bytes.len())
        .unwrap_or_else(|| panic!("a dump ends {n} bytes short at offset {at}"));
    let s = &bytes[*at..end];
    *at = end;
    s
}

fn write_meta(out: &Path, arm: &str, loss: f64) -> Result<(), String> {
    let p = out.join(format!("{arm}.meta"));
    std::fs::write(&p, format!("loss {loss}\n")).map_err(|e| format!("{}: {e}", p.display()))
}

fn read_meta_loss(out: &Path, arm: &str) -> f64 {
    let p = out.join(format!("{arm}.meta"));
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    text.lines()
        .find_map(|l| l.strip_prefix("loss "))
        .unwrap_or_else(|| panic!("{}: no loss line", p.display()))
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

// ---- the arms ----------------------------------------------------------------

struct StepOut {
    loss: f64,
    /// Forward, backward and AdamW, through the device finishing; the dump is
    /// not in it.
    wall_ms: f64,
    dump_ms: f64,
}

trait Arm {
    /// One step; with `dump` the gradient dump runs between backward and
    /// AdamW.
    fn step(&mut self, dump: bool) -> Result<StepOut, Fault>;
    fn allocated(&self) -> u64;
    /// Device bytes at the most since the last [`Arm::reset_peak`].
    fn peak(&self) -> u64;
    fn reset_peak(&self);
}

/// The measured loop, identical for both arms: a warmup step, then
/// `TIMED_STEPS` timed ones.
fn drive(arm: &mut dyn Arm, mut r: Row, n: usize, dump: bool) -> Row {
    arm.reset_peak();
    let first = match arm.step(dump) {
        Ok(v) => v,
        Err((phase, e)) => refused(r, phase, e),
    };
    r.real("first_step_ms", first.wall_ms);
    r.num("first_step_peak_bytes", arm.peak());
    r.text("first_step_loss", &format!("{:.8}", first.loss));
    r.real("dump_ms", first.dump_ms);
    let fp = footprint_or_refuse(&r);
    r.num("first_step_footprint_max_bytes", fp.lifetime_max);
    r.phase("first step");
    arm.reset_peak();
    let mut walls = Vec::with_capacity(TIMED_STEPS);
    let mut losses = Vec::with_capacity(TIMED_STEPS);
    for _ in 0..TIMED_STEPS {
        match arm.step(false) {
            Ok(s) => {
                walls.push(s.wall_ms);
                losses.push(s.loss);
            }
            Err((phase, e)) => refused(r, phase, e),
        }
    }
    let mean_ms = walls.iter().sum::<f64>() / walls.len() as f64;
    for (k, w) in ["step1_ms", "step2_ms", "step3_ms"].into_iter().zip(&walls) {
        r.real(k, *w);
    }
    r.real("mean_step_ms", mean_ms);
    r.real(
        "min_step_ms",
        walls.iter().copied().fold(f64::INFINITY, f64::min),
    );
    r.real("tokens_per_s", n as f64 / (mean_ms / 1e3));
    r.num("step_peak_bytes", arm.peak());
    r.num("end_allocated_bytes", arm.allocated());
    r.text(
        "timed_losses",
        &losses
            .iter()
            .map(|l| format!("{l:.8}"))
            .collect::<Vec<_>>()
            .join(","),
    );
    let fp = footprint_or_refuse(&r);
    r.num("end_footprint_bytes", fp.now);
    r.num("footprint_max_bytes", fp.lifetime_max);
    r
}

fn hyper_ab() -> lstep::AdamWHyper {
    lstep::AdamWHyper {
        lr: LR,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        grad_scale: 1.0,
    }
}

struct TapeArm {
    p: TapeStepProvider<MetalBackend>,
    m: MetalBackend,
    ids: Vec<u32>,
    positions: Vec<u32>,
    entries: usize,
    /// The dump's inputs, present when the run dumps.
    dump: Option<TapeDump>,
}

struct TapeDump {
    out: PathBuf,
    spec: ojas_model::qwen35::Qwen35Spec,
    prefix: String,
    layers: usize,
    vocab: usize,
    /// The parameters' shape, to put the bank's gradients back in it.
    template: Qwen35Params<()>,
    host_budget: Budget,
}

impl TapeArm {
    fn dump_gradients(&self, d: &TapeDump, loss: f64) -> Result<(), String> {
        let mut i = 0usize;
        let grads = d.template.try_map(&mut |_| {
            let g = self
                .p
                .bank()
                .grad(i)
                .map_err(|e| format!("bank {i}: {e:?}"));
            i += 1;
            g
        })?;
        let fused = fuse_grads(&d.spec, &grads, &d.prefix, &d.host_budget)
            .map_err(|e| format!("fuse_grads: {e:?}"))?;
        let rows = embed_rows(&self.ids[..self.positions.len()], d.vocab);
        let mut out: Entries = Vec::new();
        for h in fused {
            let short = h
                .name
                .strip_prefix(&d.prefix)
                .unwrap_or(&h.name)
                .to_string();
            if short == "embed_tokens.weight" {
                out.push((short, gather_rows(&h.values, h.shape[1], &rows)));
            } else if kept(&short, &h.shape, d.layers) {
                out.push((short, h.values));
            }
        }
        write_dump(&d.out.join("tape.grads"), &out)?;
        write_meta(&d.out, "tape", loss)
    }
}

impl Arm for TapeArm {
    fn step(&mut self, dump: bool) -> Result<StepOut, Fault> {
        let n = self.positions.len();
        let job = lstep::SequenceJob {
            tokens: &self.ids[..n],
            rows: lstep::RowTargets {
                positions: &self.positions,
                targets: &self.ids[1..],
                scale: 1.0 / n as f32,
            },
            hidden_at: &[],
        };
        let started = Instant::now();
        let sum = self
            .p
            .accumulate(&job, lstep::BankMode::Overwrite, None)
            .map_err(|e| ("accumulate", format!("{e:?}")))?;
        let loss = sum / n as f64;
        let mut wall = started.elapsed();
        let mut dump_ms = 0.0;
        if dump {
            let t = Instant::now();
            match &self.dump {
                Some(d) => self
                    .dump_gradients(d, loss)
                    .map_err(|e| ("dump gradients", e))?,
                None => return Err(("dump gradients", "AB_DUMP set without a dump".into())),
            }
            dump_ms = ms(t.elapsed());
        }
        let t = Instant::now();
        let next = self.p.step_count() + 1;
        self.p
            .adamw_step(
                &hyper_ab(),
                next,
                &vec![1.0; self.entries],
                &vec![0.0; self.entries],
            )
            .map_err(|e| ("adamw_step", format!("{e:?}")))?;
        self.m
            .sync()
            .map_err(|e| ("sync adamw", format!("{e:?}")))?;
        wall += t.elapsed();
        Ok(StepOut {
            loss,
            wall_ms: ms(wall),
            dump_ms,
        })
    }

    fn allocated(&self) -> u64 {
        self.m.memory().map(|m| m.allocated).unwrap_or(0)
    }

    fn peak(&self) -> u64 {
        self.m.memory().map(|m| m.peak_allocated).unwrap_or(0)
    }

    fn reset_peak(&self) {
        // A failed read leaves the old peak, which only over-states it.
        let _ = self.m.memory_and_reset_peak();
    }
}

fn checkpointing(r: &mut Row) -> ActivationCheckpoint {
    let c = match std::env::var("AB_CKPT").as_deref() {
        Ok("on") | Err(std::env::VarError::NotPresent) => ActivationCheckpoint::Blocks,
        Ok("off") => ActivationCheckpoint::Off,
        other => panic!("AB_CKPT is on or off, got {other:?}"),
    };
    r.text(
        "checkpointing",
        if c == ActivationCheckpoint::Blocks {
            "on"
        } else {
            "off"
        },
    );
    c
}

fn run_tape(snap: &Snapshot, ids: &[u32], mut r: Row, out: Option<&Path>) -> Row {
    let cfg = snap.config.clone();
    let n = ids.len() - 1;
    let activations = checkpointing(&mut r);
    r.num("positions", n as u64);
    r.num("device_budget_bytes", DEVICE_BUDGET);
    let m = match MetalBackend::new(Budget::new(DEVICE_BUDGET)) {
        Ok(m) => m,
        Err(e) => refused(r, "MetalBackend::new", e),
    };
    let host_budget = Budget::new(24 * GIB);
    let weights = single_weights_file(&snap.weights_path);
    let dump = out.map(|out| {
        // The shape of the parameters, to put the bank's gradients back in it:
        // the host copy is dropped as soon as it is mapped to `()`.
        let file = SafeTensors::open(&weights).unwrap();
        let template = match load_hf(
            &cfg.tape_spec(),
            &file,
            &cfg.tower_prefix,
            DType::F32,
            DType::F32,
            &host_budget,
        )
        .and_then(|h| h.try_map(&mut |_| Ok::<(), OjasError>(())))
        {
            Ok(t) => t,
            Err(e) => refused(r.clone(), "dump template", e),
        };
        TapeDump {
            out: out.to_path_buf(),
            spec: cfg.tape_spec(),
            prefix: cfg.tower_prefix.clone(),
            layers: cfg.layers.len(),
            vocab: cfg.vocab as usize,
            template,
            host_budget: host_budget.clone(),
        }
    });
    let open_cfg = OpenConfig {
        spec: cfg.tape_spec(),
        prefix: cfg.tower_prefix.clone(),
        base_id: base_identity(&weights),
        base_path: weights,
        linears: DType::F32,
        table: DType::F32,
        widest: n,
        chunk: DEFAULT_CE_CHUNK,
        activations,
    };
    // A full fine-tune updates the base in place: no freeze hook.
    let hooks = DeviceHooks {
        freeze: None,
        rebind: Some(Box::new(|b: &MetalBackend, budget: Budget| {
            b.with_budget(budget)
        })),
    };
    let opened = Instant::now();
    let mut p = match TapeStepProvider::open_full(m.clone(), hooks, open_cfg) {
        Ok(p) => p,
        Err(e) => refused(r, "open_full", e),
    };
    if let Err(e) = m.sync() {
        refused(r, "sync open", e);
    }
    r.real("open_ms", ms(opened.elapsed()));
    let resident = m.memory_and_reset_peak().unwrap();
    r.num("resident_bytes", resident.allocated);
    r.num("working_set_bytes", resident.recommended_working_set);
    let fp = footprint_or_refuse(&r);
    r.num("resident_footprint_bytes", fp.now);
    r.num("resident_footprint_max_bytes", fp.lifetime_max);
    r.phase("resident");
    // The pre-flight gate a device backend needs before its first step. The
    // calibration is nominal (a full-gradient step adds about the bank and
    // some activations to the resident bytes); the gate is the real dry run
    // under the cap, not the estimate.
    let calibration = match Calibration::from_points(
        "ab-provider nominal",
        resident.allocated,
        &[
            Point {
                tokens: (n / 2).max(1) as u64,
                peak_bytes: resident.allocated + 4 * GIB,
            },
            Point {
                tokens: n as u64,
                peak_bytes: resident.allocated + 8 * GIB,
            },
        ],
    ) {
        Ok(c) => c,
        Err(e) => refused(r, "calibration", e),
    };
    let pre = Instant::now();
    if let Err(e) = p.preflight(&calibration, n, &[n as u64], DEVICE_BUDGET, 0) {
        refused(r, "preflight", e);
    }
    r.real("preflight_ms", ms(pre.elapsed()));
    r.num(
        "preflight_peak_bytes",
        m.memory_and_reset_peak().unwrap().peak_allocated,
    );
    r.phase("preflight");
    let entries = p.parameters().iter().filter(|s| s.trainable).count();
    r.num("trainable_entries", entries as u64);
    let mut arm = TapeArm {
        p,
        m,
        ids: ids.to_vec(),
        positions: (0..n as u32).collect(),
        entries,
        dump,
    };
    let dumping = arm.dump.is_some();
    drive(&mut arm, r, n, dumping)
}

struct TesslArm {
    s: Qwen35Step,
    plan: OptimizerPlan,
    ids: Vec<u32>,
    rows: Vec<u32>,
    /// Gradient-dump inputs.
    dump: Option<(PathBuf, usize, usize)>,
    /// A small entry read back after AdamW: the read waits for the device.
    barrier: String,
}

impl TesslArm {
    fn dump_gradients(
        &self,
        out: &Path,
        layers: usize,
        vocab: usize,
        loss: f64,
    ) -> Result<(), String> {
        let table = self.s.parameter_table();
        let names: Vec<&str> = table
            .iter()
            .filter(|t| kept(&t.name, &t.shape, layers))
            .map(|t| t.name.as_str())
            .collect();
        let mut entries: Entries = self
            .s
            .read_entries(Which::Gradients, &names)
            .map_err(|e| format!("read_entries: {e:?}"))?
            .into_iter()
            .map(|g| g.tensor.to_f32_vec().map(|v| (g.name, v)))
            .collect::<Result<Entries, OjasError>>()
            .map_err(|e| format!("to_f32_vec: {e:?}"))?;
        let embed = self
            .s
            .read_entries(Which::Gradients, &["embed_tokens.weight"])
            .map_err(|e| format!("read embed: {e:?}"))?;
        let g = embed
            .into_iter()
            .next()
            .ok_or_else(|| "no embedding gradient".to_string())?;
        let width = g.tensor.shape()[1];
        let values = g
            .tensor
            .to_f32_vec()
            .map_err(|e| format!("embed to_f32_vec: {e:?}"))?;
        let n = self.rows.len();
        let rows = embed_rows(&self.ids[..n], vocab);
        // The embedding record goes where `TapeArm` puts it: after the layers
        // in table order is not needed, the comparison matches by name.
        entries.push((
            "embed_tokens.weight".to_string(),
            gather_rows(&values, width, &rows),
        ));
        write_dump(&out.join("tessl.grads"), &entries)?;
        write_meta(out, "tessl", loss)
    }
}

impl Arm for TesslArm {
    fn step(&mut self, dump: bool) -> Result<StepOut, Fault> {
        let n = self.rows.len();
        let started = Instant::now();
        let pending = self
            .s
            .forward(&Sequence {
                ids: &self.ids[..n],
                letter_rows: &self.rows,
                letter_targets: &self.ids[1..],
                letter_scale: 1.0 / n as f32,
                span_positions: &[],
            })
            .map_err(|e| ("forward", format!("{e:?}")))?;
        let loss = pending.letter_ce_sum() / n as f64;
        self.s
            .backward(pending, None)
            .map_err(|e| ("backward", format!("{e:?}")))?;
        let mut wall = started.elapsed();
        let mut dump_ms = 0.0;
        if dump {
            let t = Instant::now();
            match &self.dump {
                Some((out, layers, vocab)) => self
                    .dump_gradients(out, *layers, *vocab, loss)
                    .map_err(|e| ("dump gradients", e))?,
                None => return Err(("dump gradients", "AB_DUMP set without a dump".into())),
            }
            dump_ms = ms(t.elapsed());
        }
        let t = Instant::now();
        let hyper = AdamWHyper {
            lr: LR,
            beta1: BETA1,
            beta2: BETA2,
            eps: EPS,
            grad_scale: 1.0,
        };
        self.s
            .adamw_step(&hyper, &self.plan)
            .map_err(|e| ("adamw_step", format!("{e:?}")))?;
        self.s
            .read_entries(Which::Parameters, &[self.barrier.as_str()])
            .map_err(|e| ("barrier read", format!("{e:?}")))?;
        wall += t.elapsed();
        Ok(StepOut {
            loss,
            wall_ms: ms(wall),
            dump_ms,
        })
    }

    fn allocated(&self) -> u64 {
        self.s.device_bytes().allocated
    }

    fn peak(&self) -> u64 {
        self.s.device_bytes().peak
    }

    fn reset_peak(&self) {
        self.s.reset_peak_device_bytes();
    }
}

fn run_tessl(snap: &Snapshot, ids: &[u32], mut r: Row, out: Option<&Path>) -> Row {
    let cfg = snap.config.clone();
    let n = ids.len() - 1;
    r.text("checkpointing", "n/a (whole pending state)");
    r.num("positions", n as u64);
    let opened = Instant::now();
    let s = match Qwen35Step::open(
        snap,
        Numerics::ExactF32,
        Precision::F32,
        Budget::new(16 * GIB),
    ) {
        Ok(s) => s,
        Err(e) => refused(r, "open", e),
    };
    r.real("open_ms", ms(opened.elapsed()));
    let open = s.device_bytes();
    r.num("resident_bytes", open.allocated);
    r.num("open_peak_bytes", open.peak);
    let fp = footprint_or_refuse(&r);
    r.num("resident_footprint_bytes", fp.now);
    r.num("resident_footprint_max_bytes", fp.lifetime_max);
    r.phase("resident");
    let plan = match OptimizerPlan::build(
        s.parameter_table(),
        &GroupSpec {
            lr: vec![LrRule {
                label: "all".into(),
                select: Select::All,
                lr_scale: 1.0,
            }],
            weight_decay: vec![WdRule {
                label: "all".into(),
                select: Select::All,
                weight_decay: 0.0,
            }],
        },
    ) {
        Ok(p) => p,
        Err(e) => refused(r, "optimizer plan", e),
    };
    let barrier = s
        .parameter_table()
        .last()
        .map(|t| t.name.clone())
        .expect("a parameter table");
    let mut arm = TesslArm {
        s,
        plan,
        ids: ids.to_vec(),
        rows: (0..n as u32).collect(),
        dump: out.map(|o| (o.to_path_buf(), cfg.layers.len(), cfg.vocab as usize)),
        barrier,
    };
    let dumping = arm.dump.is_some();
    drive(&mut arm, r, n, dumping)
}

// ---- entry points ------------------------------------------------------------

/// One arm, one process: see the module docs.
#[test]
#[ignore = "A/B arm (AB_ARM=tape|tessl); opens a Metal runtime, run alone under the heavy-job lock"]
fn gpu_ab_arm() {
    let arm = std::env::var("AB_ARM").expect("AB_ARM is tape or tessl");
    let tokens: usize = env_or("AB_TOKENS", "2048")
        .parse()
        .expect("AB_TOKENS is a number");
    let dir = snapshot_dir();
    let snap = Snapshot::from_dir(&dir).unwrap();
    let cfg = snap.config.clone();
    let reserved: Vec<u32> = cfg.reserved_token_ids.iter().map(|(_, id)| *id).collect();
    let ids = ids(tokens, cfg.vocab, &reserved);
    let out = (env_or("AB_DUMP", "0") == "1").then(|| {
        let d = PathBuf::from(env_or(
            "AB_OUT",
            &std::env::temp_dir()
                .join("ojas-ab-provider")
                .display()
                .to_string(),
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    });
    let mut row = Row::new();
    row.text("arm", &arm);
    row.text("snapshot", &dir.display().to_string());
    row.num("tokens", tokens as u64);
    row.text("rep", &env_or("AB_REP", "0"));
    row.text("order", &env_or("AB_ORDER", "n/a"));
    row.text("dump", if out.is_some() { "on" } else { "off" });
    let started = Instant::now();
    let mut row = match arm.as_str() {
        "tape" => run_tape(&snap, &ids, row, out.as_deref()),
        "tessl" => run_tessl(&snap, &ids, row, out.as_deref()),
        other => panic!("unknown AB_ARM {other}"),
    };
    row.num("wall_ms", started.elapsed().as_millis() as u64);
    row.text("status", "ok");
    row.print();
}

/// Worst `|got - want| / max|want|`; a NaN difference is infinite, not
/// skipped by `max`.
fn worst_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want
        .iter()
        .fold(0.0f64, |m, x| m.max(f64::from(*x).abs()))
        .max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let d = (f64::from(*g) - f64::from(*w)).abs() / peak;
            if d.is_nan() {
                f64::INFINITY
            } else {
                d
            }
        })
        .fold(0.0, f64::max)
}

/// The parity row of the two dumps in `out`, and whether it is within the
/// bounds: tessl is the reference.
fn compare(out: &Path) -> (Row, bool) {
    let tape = read_dump(&out.join("tape.grads"));
    let tessl = read_dump(&out.join("tessl.grads"));
    let (lt, ls) = (read_meta_loss(out, "tape"), read_meta_loss(out, "tessl"));
    let mut r = Row::new();
    r.text("arm", "tape-vs-tessl-parity");
    r.text("out", &out.display().to_string());
    r.text("tape_loss", &format!("{lt:.10}"));
    r.text("tessl_loss", &format!("{ls:.10}"));
    let loss_rel = (lt - ls).abs() / ls.abs();
    r.text("loss_rel", &format!("{loss_rel:.3e}"));
    let mut seen = 0u64;
    let mut worst = (String::new(), 0.0f64);
    let mut missing: Vec<String> = Vec::new();
    for (name, want) in &tessl {
        let Some((_, got)) = tape.iter().find(|(n, _)| n == name) else {
            missing.push(name.clone());
            continue;
        };
        if got.len() != want.len() {
            missing.push(format!("{name} (length {} vs {})", got.len(), want.len()));
            continue;
        }
        let x = worst_rel(got, want);
        eprintln!("{name}: {x:.3e} of its peak");
        if x > worst.1 {
            worst = (name.clone(), x);
        }
        seen += 1;
    }
    let extra: Vec<&String> = tape
        .iter()
        .map(|(n, _)| n)
        .filter(|n| !tessl.iter().any(|(m, _)| m == *n))
        .collect();
    r.num("gradients_compared", seen);
    r.num("tessl_entries", tessl.len() as u64);
    r.num("tape_entries", tape.len() as u64);
    r.text("worst_grad", &worst.0);
    r.text("worst_grad_rel", &format!("{:.3e}", worst.1));
    r.text("unmatched", &format!("{missing:?} {extra:?}"));
    let within = loss_rel.is_finite()
        && loss_rel <= LOSS_BOUND
        && worst.1 <= GRAD_BOUND
        && missing.is_empty()
        && extra.is_empty()
        && seen > 0;
    r.text("status", if within { "ok" } else { "out of bounds" });
    (r, within)
}

/// The comparison of an earlier pair of arm runs that both set `AB_DUMP=1`
/// with the same `AB_OUT`.
#[test]
#[ignore = "reads the dumps two gpu_ab_arm runs wrote (AB_OUT)"]
fn ab_compare_dumps() {
    let out = PathBuf::from(std::env::var("AB_OUT").expect("AB_OUT"));
    let (row, within) = compare(&out);
    // The row is printed before the bounds are asserted, as in the 2B gate.
    row.print();
    assert!(within, "the parity row is out of bounds");
}

/// Both arms and the comparison on the two-layer fixture, one process: the
/// harness's own check, cheap enough to run before the real shape.
#[test]
#[ignore = "opens two Metal runtimes on the tiny fixture; run under the heavy-job lock"]
fn gpu_ab_tiny_end_to_end() {
    // tessl's committed two-layer fixture: the one `gpu_parity.rs` already opens
    // a `Qwen35Step` on, and `cpu_tape_spec.rs` checks the Tape's spec against.
    let dir = common::tessl_root().join("tests/fixtures/qwen35_train");
    let snap = Snapshot::from_dir(&dir).unwrap();
    let cfg = snap.config.clone();
    let reserved: Vec<u32> = cfg.reserved_token_ids.iter().map(|(_, id)| *id).collect();
    let ids = ids(70, cfg.vocab, &reserved);
    let out = std::env::temp_dir().join(format!("ojas-ab-tiny-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let tape = run_tape(&snap, &ids, Row::new(), Some(&out));
    tape.print();
    let tessl = run_tessl(&snap, &ids, Row::new(), Some(&out));
    tessl.print();
    let (row, within) = compare(&out);
    row.print();
    std::fs::remove_dir_all(&out).ok();
    assert!(within, "the tiny fixture's parity row is out of bounds");
}
