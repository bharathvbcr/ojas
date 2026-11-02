//! One row of the nanolab training-step config A/B
//! (`bench/results/2026-11-02-nanolab-config-ab/`): the three levers a
//! 2026-11-02 research sweep ranked without measuring, each reached through
//! `TrainConfig` and nothing else.
//!
//! ```text
//! cargo run -p ojas-model --release --example nanolab_config_ab -- \
//!     step <batch> <f32|bf16> <ce_cols> <off|blocks> [steps 10] [seq 1024] [tag]
//! cargo run -p ojas-model --release --example nanolab_config_ab -- \
//!     phase <batch> [reps 8] [seq 1024]
//! ```
//!
//! `step` runs nanolab-124M training steps on Metal with Muon's
//! Newton-Schulz at `f32` (the default) or `bf16` (`TrainConfig::muon_ns5`),
//! the fused LM-head CE tiled at 1024 rows by `ce_cols` vocabulary columns
//! (`TrainConfig::chunk`; 8192 is the default, 50304 one tile), and
//! `ActivationCheckpoint` off or per block. Autocast stays off in every arm.
//! It prints one JSON line: every step's wall time and loss, and every
//! step's peak charged bytes (the trainer resets the budget's peak at the
//! start of each step, so the peak is read after each one). The caller
//! discards warm-up steps; the row keeps them.
//!
//! `phase` is the op-level profile the trainer does not have: on the same
//! device it times (a) one Muon sweep over every nanolab Muon-group matrix,
//! `f32` against `bf16` Newton-Schulz, and (b) one fused LM-head CE with
//! both gradients over `batch * seq` rows, the default tile against one
//! vocabulary tile. Arms alternate within the process; each call is synced.
//!
//! Both modes read `sysctl -n vm.loadavg` before and after the timed loop.

// Off macOS only the refusal in `run` is reachable.
#![cfg_attr(not(target_os = "macos"), allow(dead_code, unused_imports))]

use std::time::Instant;

use ojas_core::{
    Autocast, AutocastMode, Backend, Budget, CeChunk, CeGrads, MuonNs5Config, Ns5Precision,
    OjasError, Tensor,
};
use ojas_cpu::{CosineSchedule, LrSchedule, OptimGroup};
use ojas_data::TokenBin;
use ojas_model::{
    init_params, param_table, ActivationCheckpoint, ModelSpec, TrainConfig, Trainer,
    DEFAULT_CE_CHUNK,
};

type R<T> = Result<T, OjasError>;

const USAGE: &str = "usage: nanolab_config_ab step <batch> <f32|bf16> <ce_cols> <off|blocks> \
                     [steps] [seq] [tag] | phase <batch> [reps] [seq]";

fn fail(detail: impl Into<String>) -> OjasError {
    OjasError::Unsupported {
        op: "nanolab_config_ab",
        detail: detail.into(),
    }
}

fn count(a: &[String], i: usize, default: usize) -> R<usize> {
    match a.get(i) {
        None => Ok(default),
        Some(s) => s
            .parse()
            .map_err(|_| fail(format!("argument {} {s:?} is not a count; {USAGE}", i + 1))),
    }
}

struct StepArgs {
    batch: usize,
    ns5: Ns5Precision,
    ce_cols: usize,
    activations: ActivationCheckpoint,
    steps: usize,
    seq: usize,
    tag: String,
}

struct PhaseArgs {
    batch: usize,
    reps: usize,
    seq: usize,
}

enum Mode {
    Step(StepArgs),
    Phase(PhaseArgs),
}

fn args() -> R<Mode> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("step") if (5..=8).contains(&a.len()) => {
            let ns5 = match a[2].as_str() {
                "f32" => Ns5Precision::F32,
                "bf16" => Ns5Precision::Bf16,
                other => return Err(fail(format!("ns5 {other}; {USAGE}"))),
            };
            let activations = match a[4].as_str() {
                "off" => ActivationCheckpoint::Off,
                "blocks" => ActivationCheckpoint::Blocks,
                other => return Err(fail(format!("checkpoint {other}; {USAGE}"))),
            };
            Ok(Mode::Step(StepArgs {
                batch: count(&a, 1, 0)?,
                ns5,
                ce_cols: count(&a, 3, 0)?,
                activations,
                steps: count(&a, 5, 10)?,
                seq: count(&a, 6, 1024)?,
                tag: a.get(7).cloned().unwrap_or_default(),
            }))
        }
        Some("phase") if (2..=4).contains(&a.len()) => Ok(Mode::Phase(PhaseArgs {
            batch: count(&a, 1, 0)?,
            reps: count(&a, 2, 8)?,
            seq: count(&a, 3, 1024)?,
        })),
        _ => Err(fail(USAGE)),
    }
}

/// The 1, 5 and 15 minute load averages as `sysctl` prints them, without
/// the braces; "unavailable" when it cannot be read.
fn loadavg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().trim_matches(['{', '}']).trim().to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

/// A headerless u16 token bin of `len` tokens below `vocab`, in a fresh
/// directory under the system temp dir; removed with the returned guard.
struct TempBin(std::path::PathBuf);

impl Drop for TempBin {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn token_bin(len: usize, vocab: u32) -> R<(TempBin, TokenBin)> {
    let dir = std::env::temp_dir().join(format!("ojas-config-ab-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| fail(format!("temp dir: {e}")))?;
    let guard = TempBin(dir.clone());
    let mut bytes = Vec::with_capacity(len * 2);
    let mut x = 1u32;
    for i in 0..len {
        x = if i % 97 == 0 {
            (i as u32 / 97 * 31 + 7) % vocab
        } else {
            (5 * x + 3) % vocab
        };
        bytes.extend_from_slice(&(x as u16).to_le_bytes());
    }
    let path = dir.join("tokens.bin");
    std::fs::write(&path, &bytes).map_err(|e| fail(format!("token bin: {e}")))?;
    let bin = TokenBin::open_headerless(&path).map_err(|e| fail(format!("token bin: {e}")))?;
    Ok((guard, bin))
}

fn list<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn secs(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn step_row<B: Backend>(backend: B, a: &StepArgs) -> R<String> {
    let spec = ModelSpec::nanolab_124m();
    let schedule = LrSchedule::Cosine(CosineSchedule::new(1, 1000)?);
    let mut cfg = TrainConfig::nanolab(a.batch, a.seq, 1, 1337, schedule);
    cfg.autocast = AutocastMode::Off;
    cfg.muon_ns5 = a.ns5;
    cfg.chunk = CeChunk {
        rows: DEFAULT_CE_CHUNK.rows,
        cols: a.ce_cols,
    };
    cfg.activations = a.activations;
    let host = init_params(&spec, 1337, &Budget::new(4 << 30))?;
    let tokens = (a.batch * (a.seq + 1) * (a.steps + 2)).max(1 << 20);
    let (_guard, bin) = token_bin(tokens, spec.vocab as u32)?;
    let backend = Autocast::new(backend);
    let budget = backend.budget().clone();
    let limit = budget.cap_bytes();
    let mut t = Trainer::new(backend, spec, &host, bin, cfg)?;
    drop(host);
    let resident = budget.live_bytes()?;
    let load_before = loadavg();
    let mut times = Vec::with_capacity(a.steps);
    let mut losses = Vec::with_capacity(a.steps);
    let mut peaks = Vec::with_capacity(a.steps);
    for _ in 0..a.steps {
        let start = Instant::now();
        let r = t.step()?;
        t.backend().sync()?;
        times.push(start.elapsed().as_secs_f64());
        peaks.push(budget.peak_bytes());
        losses.push(format!("{:?}", r.loss));
    }
    let load_after = loadavg();
    let ns5 = match a.ns5 {
        Ns5Precision::F32 => "f32",
        Ns5Precision::Bf16 => "bf16",
    };
    Ok(format!(
        "{{\"mode\":\"step\",\"tag\":\"{}\",\"backend\":\"metal\",\"batch\":{},\"seq\":{},\
         \"steps\":{},\"ns5\":\"{ns5}\",\"ce_rows\":{},\"ce_cols\":{},\"checkpoint\":\"{:?}\",\
         \"autocast\":\"off\",\"budget_limit_bytes\":{limit},\
         \"resident_bytes_before_steps\":{resident},\"step_peak_charged_bytes\":[{}],\
         \"step_seconds\":[{}],\"losses\":[{}],\"load_before\":\"{load_before}\",\
         \"load_after\":\"{load_after}\"}}",
        a.tag,
        a.batch,
        a.seq,
        a.steps,
        DEFAULT_CE_CHUNK.rows,
        a.ce_cols,
        a.activations,
        list(&peaks),
        secs(&times),
        losses.join(","),
    ))
}

/// `n` values in `[-scale, scale)` from a fixed LCG, so every run times the
/// same operands.
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (s >> 40) as f32 / (1u64 << 24) as f32;
            (2.0 * unit - 1.0) * scale
        })
        .collect()
}

fn device<B: Backend>(b: &B, host: &Tensor) -> R<Tensor> {
    b.upload(host)
}

fn phase_row<B: Backend>(b: B, a: &PhaseArgs) -> R<String> {
    let spec = ModelSpec::nanolab_124m();
    let host = Budget::new(8 << 30);
    // (a) Muon: every trainable Muon-group matrix, as the trainer steps them.
    let mut shapes = Vec::new();
    let mut slots = Vec::new();
    for (i, p) in param_table(&spec)?.into_iter().enumerate() {
        if p.group != OptimGroup::MuonMatrix || !p.trains {
            continue;
        }
        let n = p.numel();
        let seed = i as u64;
        let param = device(
            &b,
            &Tensor::from_f32(&fill(n, seed, 0.02), &p.shape, &host)?,
        )?;
        let grad = device(
            &b,
            &Tensor::from_f32(&fill(n, seed + 1000, 1e-3), &p.shape, &host)?,
        )?;
        let momentum = device(&b, &Tensor::from_f32(&vec![0.0; n], &p.shape, &host)?)?;
        shapes.push(format!(
            "\"{}x{}\"",
            p.shape[0],
            p.shape[1..].iter().product::<usize>()
        ));
        slots.push((param, grad, momentum));
    }
    // (b) the fused LM-head CE over one micro-batch's rows.
    let (n, d, v) = (a.batch * a.seq, spec.n_embd, spec.vocab);
    let x = device(&b, &Tensor::from_f32(&fill(n * d, 7, 1.0), &[n, d], &host)?)?;
    let w = device(
        &b,
        &Tensor::from_f32(&fill(v * d, 8, 0.02), &[v, d], &host)?,
    )?;
    let ids: Vec<u32> = (0..n).map(|i| ((i * 7919) % v) as u32).collect();
    let t = device(&b, &Tensor::from_u32(&ids, &[n], &host)?)?;
    let grads = CeGrads {
        input: true,
        weight: true,
    };
    let one_tile = CeChunk {
        rows: DEFAULT_CE_CHUNK.rows,
        cols: v,
    };
    b.sync()?;
    let budget = b.budget().clone();
    let load_before = loadavg();
    let (mut muon_f32, mut muon_bf16, mut ce_default, mut ce_one) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut ce_default_peak, mut ce_one_peak) = (0u64, 0u64);
    for rep in 0..a.reps {
        // Alternate which arm goes first, so neither always runs warm.
        let order = if rep % 2 == 0 {
            [Ns5Precision::F32, Ns5Precision::Bf16]
        } else {
            [Ns5Precision::Bf16, Ns5Precision::F32]
        };
        for ns5 in order {
            let config = MuonNs5Config {
                ns5,
                ..MuonNs5Config::nanolab_default()
            };
            let start = Instant::now();
            for (param, grad, momentum) in &mut slots {
                b.muon_ns5_step(param, grad, momentum, config)?;
            }
            b.sync()?;
            let dt = start.elapsed().as_secs_f64();
            match ns5 {
                Ns5Precision::F32 => muon_f32.push(dt),
                Ns5Precision::Bf16 => muon_bf16.push(dt),
            }
        }
        let chunks = if rep % 2 == 0 {
            [DEFAULT_CE_CHUNK, one_tile]
        } else {
            [one_tile, DEFAULT_CE_CHUNK]
        };
        for chunk in chunks {
            budget.reset_peak();
            let start = Instant::now();
            let out = b.linear_cross_entropy_mean(&x, &w, &t, None, chunk, grads)?;
            b.sync()?;
            let dt = start.elapsed().as_secs_f64();
            let peak = budget.peak_bytes();
            drop(out);
            if chunk == one_tile {
                ce_one.push(dt);
                ce_one_peak = ce_one_peak.max(peak);
            } else {
                ce_default.push(dt);
                ce_default_peak = ce_default_peak.max(peak);
            }
        }
    }
    let load_after = loadavg();
    Ok(format!(
        "{{\"mode\":\"phase\",\"backend\":\"metal\",\"batch\":{},\"seq\":{},\"reps\":{},\
         \"muon_matrices\":{},\"muon_shapes\":[{}],\"muon_sweep_f32_seconds\":[{}],\
         \"muon_sweep_bf16_seconds\":[{}],\"ce_rows\":{n},\"ce_default_cols\":{},\
         \"ce_default_seconds\":[{}],\"ce_one_tile_seconds\":[{}],\
         \"ce_default_peak_bytes\":{ce_default_peak},\"ce_one_tile_peak_bytes\":{ce_one_peak},\
         \"load_before\":\"{load_before}\",\"load_after\":\"{load_after}\"}}",
        a.batch,
        a.seq,
        a.reps,
        slots.len(),
        shapes.join(","),
        secs(&muon_f32),
        secs(&muon_bf16),
        DEFAULT_CE_CHUNK.cols,
        secs(&ce_default),
        secs(&ce_one),
    ))
}

#[cfg(target_os = "macos")]
fn run(mode: &Mode) -> R<String> {
    let budget = Budget::new(48 << 30);
    match mode {
        Mode::Step(a) => {
            if a.batch == 0 || a.steps == 0 || a.ce_cols == 0 {
                return Err(fail("batch, steps and ce_cols must be at least 1"));
            }
            step_row(ojas_metal::MetalBackend::new(budget)?, a)
        }
        Mode::Phase(a) => {
            if a.batch == 0 || a.reps == 0 {
                return Err(fail("batch and reps must be at least 1"));
            }
            phase_row(ojas_metal::MetalBackend::new(budget)?, a)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn run(_mode: &Mode) -> R<String> {
    Err(fail("the config A/B runs on Metal, which is macOS-only"))
}

fn main() {
    match args().and_then(|m| run(&m)) {
        Ok(line) => println!("{line}"),
        Err(e) => {
            eprintln!("nanolab_config_ab: {e}");
            std::process::exit(1);
        }
    }
}
