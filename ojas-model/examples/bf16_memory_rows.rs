//! One memory row of the bf16 tier's micro-batch study (criterion 4 of
//! gp-bf16-compute-tier): a few nanolab-124M training steps on one backend
//! at one micro-batch size, with the trainer's autocast off (`f32`) or on
//! (`bf16`), and the budget's peak charged bytes.
//!
//! ```text
//! cargo run -p ojas-model --release --example bf16_memory_rows -- \
//!     <cpu|wgpu|metal> <f32|bf16> <batch> [seq 1024] [steps 3] [checkpoint off|on]
//! ```
//!
//! One row per process, so `/usr/bin/time -l` around it reports this run's
//! own page faults, swaps and peak footprint, which machine-wide counters
//! cannot separate from other processes'. `bench/bf16_memory_rows.sh` runs
//! the grid and keeps both. The row is one JSON line on stdout.
//!
//! Every backend runs through `Autocast`, so the two precisions differ only
//! in the region. A backend whose `bf16_operands()` is true gets `Bf16`
//! storage for the matmul-class operands; otherwise the operands are rounded
//! in `F32` storage. Either way the tape keeps `F32` activations and the
//! parameters are `F32` masters, so the bf16 row is not expected to be
//! smaller: it measures what the tier costs, not what a frozen bf16 base
//! (gp-ft-bf16-frozen-base) saves.

use std::time::Instant;

use ojas_core::{Autocast, AutocastMode, Backend, Budget, OjasError};
use ojas_cpu::{CosineSchedule, CpuBackend, LrSchedule};
use ojas_data::TokenBin;
use ojas_model::{init_params, ActivationCheckpoint, ModelSpec, TrainConfig, Trainer};

type R<T> = Result<T, OjasError>;

fn fail(detail: impl Into<String>) -> OjasError {
    OjasError::Unsupported {
        op: "bf16_memory_rows",
        detail: detail.into(),
    }
}

struct Args {
    backend: String,
    mode: AutocastMode,
    batch: usize,
    seq: usize,
    steps: usize,
    activations: ActivationCheckpoint,
}

fn args() -> R<Args> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage =
        "usage: bf16_memory_rows <cpu|wgpu|metal> <f32|bf16> <batch> [seq] [steps] [off|on]";
    if a.len() < 3 || a.len() > 6 {
        return Err(fail(usage));
    }
    let mode = match a[1].as_str() {
        "f32" => AutocastMode::Off,
        "bf16" => AutocastMode::Bf16,
        other => return Err(fail(format!("precision {other}; {usage}"))),
    };
    let num = |i: usize, default: usize| -> R<usize> {
        match a.get(i) {
            None => Ok(default),
            Some(s) => s
                .parse()
                .map_err(|_| fail(format!("argument {} {s:?} is not a count", i + 1))),
        }
    };
    let activations = match a.get(5).map(String::as_str) {
        None | Some("off") => ActivationCheckpoint::Off,
        Some("on") => ActivationCheckpoint::Blocks,
        Some(other) => return Err(fail(format!("checkpoint {other}; {usage}"))),
    };
    Ok(Args {
        backend: a[0].clone(),
        mode,
        batch: num(2, 0)?,
        seq: num(3, 1024)?,
        steps: num(4, 3)?,
        activations,
    })
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
    let dir = std::env::temp_dir().join(format!("ojas-bf16-rows-{}", std::process::id()));
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

fn row<B: Backend>(backend: B, a: &Args) -> R<String> {
    let spec = ModelSpec::nanolab_124m();
    let schedule = LrSchedule::Cosine(CosineSchedule::new(1, 1000)?);
    let mut cfg = TrainConfig::nanolab(a.batch, a.seq, 1, 1337, schedule);
    cfg.autocast = a.mode;
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
    budget.reset_peak();
    let mut times = Vec::with_capacity(a.steps);
    let mut losses = Vec::with_capacity(a.steps);
    for _ in 0..a.steps {
        let start = Instant::now();
        let r = t.step()?;
        t.backend().sync()?;
        times.push(start.elapsed().as_secs_f64());
        losses.push(f64::from(r.loss));
    }
    let peak = budget.peak_bytes();
    let precision = match a.mode {
        AutocastMode::Off => "f32",
        AutocastMode::Bf16 => "bf16",
    };
    let fmt = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    Ok(format!(
        "{{\"backend\":\"{}\",\"precision\":\"{precision}\",\"bf16_operands\":{},\
         \"batch\":{},\"seq\":{},\"steps\":{},\"checkpoint\":\"{:?}\",\
         \"budget_limit_bytes\":{limit},\"resident_bytes_before_steps\":{resident},\
         \"peak_charged_bytes\":{peak},\"step_seconds\":[{}],\"losses\":[{}]}}",
        a.backend,
        t.backend().bf16_operands(),
        a.batch,
        a.seq,
        a.steps,
        a.activations,
        fmt(&times),
        fmt(&losses),
    ))
}

fn run(a: &Args) -> R<String> {
    if a.batch == 0 || a.steps == 0 {
        return Err(fail("batch and steps must be at least 1"));
    }
    let budget = Budget::new(48 << 30);
    match a.backend.as_str() {
        "cpu" => row(CpuBackend::new(budget), a),
        "wgpu" => row(
            ojas_wgpu::WgpuBackend::open(budget).map_err(|e| fail(format!("wgpu: {e}")))?,
            a,
        ),
        #[cfg(target_os = "macos")]
        "metal" => row(ojas_metal::MetalBackend::new(budget)?, a),
        other => Err(fail(format!("backend {other} is not built here"))),
    }
}

fn main() {
    let out = args().and_then(|a| run(&a));
    match out {
        Ok(line) => println!("{line}"),
        Err(e) => {
            eprintln!("bf16_memory_rows: {e}");
            std::process::exit(1);
        }
    }
}
