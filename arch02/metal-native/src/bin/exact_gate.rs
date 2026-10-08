//! Exact-128M deterministic checkpoint/resume, memory, dispatch, and NaN gate.

use std::path::PathBuf;

use tessl_arch02::engine::{EngineCreateConfig, TrainingEngine};
use tessl_arch02::log::mem_current_physical_mb;
use tessl_arch02::OptimizerKind;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|v| v == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn samples(engine: &TrainingEngine) -> Vec<f32> {
    let values = engine.weights.qo_bank.buffer.contents_f32();
    (0..256)
        .map(|i| values[(i * 104_729) % values.len()])
        .collect()
}

#[cfg(target_os = "macos")]
fn swap_used_mb() -> Result<f64, String> {
    #[repr(C)]
    struct SwapUsage {
        total: u64,
        available: u64,
        used: u64,
        page_size: u32,
        encrypted: i32,
    }
    let name = std::ffi::CString::new("vm.swapusage").unwrap();
    let mut usage: SwapUsage = unsafe { std::mem::zeroed() };
    let mut length = std::mem::size_of::<SwapUsage>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut usage as *mut SwapUsage).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(format!(
            "sysctl vm.swapusage: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(usage.used as f64 / (1024.0 * 1024.0))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let optimizer: OptimizerKind = arg(&args, "--optimizer")
        .unwrap_or_else(|| "muon_ns5_adamw".into())
        .parse()?;
    if !optimizer.native_ready() {
        return Err(format!("{optimizer} is not native-parity-qualified"));
    }
    let checkpoint = arg(&args, "--checkpoint")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("arch02-exact-gate-{}", std::process::id()))
        });
    let keep = args.iter().any(|v| v == "--keep-checkpoint");
    let swap_before_mb = swap_used_mb()?;
    let mut config = EngineCreateConfig::default();
    config.optimizer = optimizer;
    config.total_steps = 2;
    config.warmdown_steps = 0;

    let tokens = config.batch.unwrap() * config.seq_len.unwrap();
    let input: Vec<i32> = (0..tokens).map(|i| ((i * 17 + 13) % 1024) as i32).collect();
    let target: Vec<i32> = (0..tokens).map(|i| ((i * 29 + 7) % 1024) as i32).collect();

    let mut uninterrupted = TrainingEngine::create(config.clone())?;
    uninterrupted.train_step(&input, &target)?;
    let expected = uninterrupted.train_step(&input, &target)?;
    let expected_samples = samples(&uninterrupted);
    let uninterrupted_footprint = mem_current_physical_mb();
    drop(uninterrupted);

    let mut split = TrainingEngine::create(config)?;
    split.train_step(&input, &target)?;
    split.save(&checkpoint)?;
    drop(split);
    let mut resumed = TrainingEngine::load(&checkpoint)?;
    let actual = resumed.train_step(&input, &target)?;
    let actual_samples = samples(&resumed);
    let resumed_footprint = mem_current_physical_mb();
    const LOSS_ATOL: f32 = 1e-5;
    let weight_compare = tessl_arch02::parity::compare_f32(
        "sampled_weights", &actual_samples, &expected_samples,
        tessl_arch02::optim::OPTIM_ATOL,
    );
    let loss_compare = tessl_arch02::parity::compare_f32(
        "loss", &[actual.loss], &[expected.loss], LOSS_ATOL,
    );
    let grad_compare = tessl_arch02::parity::compare_f32(
        "grad_norm", &[actual.grad_norm], &[expected.grad_norm],
        tessl_arch02::optim::OPTIM_ATOL,
    );
    let weight_delta = weight_compare.max_abs;
    let loss_delta = loss_compare.max_abs;
    let grad_delta = grad_compare.max_abs;
    let max_footprint = uninterrupted_footprint.max(resumed_footprint);
    let swap_after_mb = swap_used_mb()?;
    // Ambient residual swap on macOS is common after any prior pressure. The
    // gate measures *induced* pressure: the run must not grow used swap.
    let swap_delta_mb = swap_after_mb - swap_before_mb;
    const SWAP_DELTA_ATOL_MB: f64 = 1e-6;
    let swap_pressure = swap_delta_mb > SWAP_DELTA_ATOL_MB;
    // Fresh Metal command queues may choose a different parallel reduction
    // order. Exact resume means replay equivalence within the native parity
    // contract, not bit-identical reduction bits.
    let mut failures = Vec::new();
    if !loss_compare.passed {
        failures.push("loss_delta");
    }
    if !grad_compare.passed {
        failures.push("grad_norm_delta");
    }
    if !weight_compare.passed {
        failures.push("sampled_weight_max_delta");
    }
    if max_footprint >= 52.0 * 1024.0 {
        failures.push("current_physical_mb");
    }
    if actual.dispatches >= 10_000 {
        failures.push("dispatches");
    }
    if swap_pressure {
        failures.push("swap_pressure");
    }
    let passed = failures.is_empty();
    let report = serde_json::json!({
        "schema_version": 1,
        "optimizer": optimizer.as_str(),
        "parameter_count": resumed.weights.cfg.count_params(),
        "checkpoint_version": tessl_arch02::CHECKPOINT_VERSION,
        "loss_delta": loss_delta,
        "grad_norm_delta": grad_delta,
        "sampled_weight_max_delta": weight_delta,
        "resume_tolerances": {
            "loss_atol": LOSS_ATOL,
            "gradient_and_weight_atol": tessl_arch02::optim::OPTIM_ATOL,
        },
        "current_physical_mb": max_footprint,
        "swap_before_mb": swap_before_mb,
        "swap_after_mb": swap_after_mb,
        "swap_delta_mb": swap_delta_mb,
        "swap_pressure": swap_pressure,
        "dispatches": actual.dispatches,
        "failures": failures,
        "passed": passed,
    });
    println!("{report}");
    if let Some(path) = arg(&args, "--output").map(PathBuf::from) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create gate output {}: {e}", parent.display()))?;
        }
        let bytes = serde_json::to_vec_pretty(&report)
            .map_err(|e| format!("serialize gate report: {e}"))?;
        std::fs::write(&path, bytes)
            .map_err(|e| format!("write gate report {}: {e}", path.display()))?;
    }
    if !keep {
        let _ = std::fs::remove_dir_all(&checkpoint);
    }
    if !passed {
        return Err("exact-128M gate failed".into());
    }
    Ok(())
}
