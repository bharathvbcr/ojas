//! AOT-compile `kernels/*.metal` → `default.metallib`.
//!
//! Targets Metal 4 / macOS 26 for TensorOps (`matmul2d`) with bf16 enabled in
//! the language dialect. The portable `simdgroup_matrix` kernel is always
//! included for A/B.
//!
//! Requires the Xcode Metal Toolchain component:
//!   `xcodebuild -downloadComponent MetalToolchain`
//!
//! Important: do **not** invoke `xcrun -sdk macosx metal` — the `-sdk` switch
//! breaks cryptex Metal Toolchain resolution on Xcode 26+. Use `xcrun metal`
//! plus an explicit `-isysroot`.
//!
//! `METAL_NATIVE_SKIP_AOT` hazard: when set, this script skips compilation and
//! points `METAL_NATIVE_METALLIB` at the crate-root `default.metallib`. That
//! file may be stale or missing — training/tests will load an outdated or
//! absent library with no rebuild. Only use for intentional offline/CI skips
//! after a known-good metallib is already present at the crate root.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=METAL_NATIVE_SKIP_AOT");

    // CoreGraphics is required for MTLCreateSystemDefaultDevice.
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    // Seeded target-scale initialization uses LAPACK QR instead of the former
    // O(m*n^2) scalar modified Gram-Schmidt implementation.
    println!("cargo:rustc-link-lib=framework=Accelerate");

    if env::var_os("METAL_NATIVE_SKIP_AOT").is_some() {
        // Hazard: trusts crate-root default.metallib (may be stale/missing).
        println!("cargo:warning=METAL_NATIVE_SKIP_AOT set; skipping metallib AOT");
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let crate_lib = manifest_dir.join("default.metallib");
        println!(
            "cargo:rustc-env=METAL_NATIVE_METALLIB={}",
            crate_lib.display()
        );
        return;
    }

    ensure_developer_dir();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");
    track_kernel_sources(&kernels_dir);

    let sdk = xcrun_stdout(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = resolve_metal();
    let metallib = resolve_metallib();

    let mut air_files: Vec<PathBuf> = Vec::new();

    // TensorOps kernels — Metal 4 dialect (macOS 26+ / MPP). Hard-fail: NAX GEMM
    // is the hot path; a simdgroup-only metallib is not acceptable.
    // The shared GEMM kernels have exactly one source of truth: tessl's
    // `kernels/` directory, published through its `links = "tessl"` key. This
    // crate used to keep a byte-identical copy, kept in step only by a static
    // audit; compiling tessl's file directly removes the thing that could drift.
    let tessl_kernels = PathBuf::from(env::var("DEP_TESSL_KERNELS").expect(
        "DEP_TESSL_KERNELS not set — tessl must be a direct dependency and \
         declare `links = \"tessl\"`; without it this crate would silently \
         build a metallib with no GEMM kernels in it",
    ));
    track_kernel_sources(&tessl_kernels);
    let tensorops_sources: Vec<PathBuf> = vec![
        tessl_kernels.join("matmul_tensorops.metal"),
        kernels_dir.join("flash_attn_tensorops.metal"),
        kernels_dir.join("matmul_batched.metal"),
    ];
    for src in &tensorops_sources {
        let name = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>");
        if !src.exists() {
            panic!(
                "required TensorOps source missing: {}; Metal 4 / macOS 26 toolchain required",
                src.display()
            );
        }
        let air = out_dir.join(format!(
            "{}.air",
            src.file_stem().unwrap().to_string_lossy()
        ));
        let status = Command::new(&metal)
            .args([
                "-std=metal4.0",
                "-O2",
                "-isysroot",
                &sdk,
                "-mmacosx-version-min=26.0",
                "-c",
            ])
            .arg(src)
            .arg("-o")
            .arg(&air)
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn metal for {name}: {e}"));
        if !status.success() {
            panic!(
                "{name} failed to compile (need Metal 4 / macOS 26 SDK + MetalToolchain); \
                 refusing simdgroup-only metallib"
            );
        }
        air_files.push(air);
    }
    println!("cargo:rustc-cfg=metal_native_tensorops");

    // All other .metal sources (simdgroup GEMM + Phase 1 kernels).
    // Prefer -std=metal4.0; fall back to metal3.2 per-file only if metal4.0 fails
    // (shader dialect, not encode path). Fallback is documented via cargo:warning.
    let skip: &[&str] = &[
        "matmul_tensorops.metal",
        "flash_attn_tensorops.metal",
        "matmul_batched.metal",
    ];
    let mut others: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap_or_else(|e| panic!("read kernels/: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("metal")
                && !skip
                    .iter()
                    .any(|s| p.file_name().and_then(|n| n.to_str()) == Some(*s))
        })
        .collect();
    others.sort();
    // Compile only the Tessl sources used by its re-exported runtime/GEMM
    // surface. `DEP_TESSL_KERNELS` now also contains inference/NN kernels; a
    // directory-wide glob both bloated this training library and collided with
    // the training-specific `rms_norm_f32` ABI in stem_fwd.metal. Keep the
    // owner files explicit and fail immediately if Tessl moves one.
    for name in ["matmul_simdgroup.metal", "utils.metal"] {
        let source = tessl_kernels.join(name);
        if !source.is_file() {
            panic!("required Tessl shared kernel missing: {}", source.display());
        }
        others.push(source);
    }

    for src in &others {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let air = out_dir.join(format!("{stem}.air"));
        // Try metal4.0 first; fall back to metal3.2 if this kernel fails under 4.0.
        let ok_m4 = try_metal_compile(&metal, &sdk, src, &air, "metal4.0");
        if !ok_m4 {
            println!(
                "cargo:warning={} failed under -std=metal4.0; falling back to -std=metal3.2 \
                 (shader dialect only; encode remains Metal 4)",
                src.file_name().and_then(|n| n.to_str()).unwrap_or(&stem)
            );
            run(
                Command::new(&metal)
                    .args([
                        "-std=metal3.2",
                        "-O2",
                        "-isysroot",
                        &sdk,
                        "-mmacosx-version-min=26.0",
                        "-c",
                    ])
                    .arg(src)
                    .arg("-o")
                    .arg(&air),
                &format!("metal compile {} (metal3.2 fallback)", src.display()),
            );
        }
        air_files.push(air);
    }

    // Metal can retain file-backed library data after loading. Never relink a
    // pathname baked into a prior binary: each build owns an immutable artifact.
    let build_id = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos()
    );
    let metallib_out = out_dir.join(format!("default-{build_id}.metallib"));
    fs::File::create_new(&metallib_out).expect("reserve unique metallib output");
    let mut link = Command::new(&metallib);
    for air in &air_files {
        link.arg(air);
    }
    link.arg("-o").arg(&metallib_out);
    run(&mut link, "metallib link");

    // Publish the offline compatibility copy without truncating an inode that
    // an existing runtime may still use. Do not hide failed publication.
    let crate_copy = manifest_dir.join("default.metallib");
    let staged_copy = manifest_dir.join(format!(".default-{build_id}.metallib"));
    fs::File::create_new(&staged_copy).expect("reserve offline metallib staging file");
    fs::copy(&metallib_out, &staged_copy).expect("stage offline metallib");
    fs::rename(&staged_copy, &crate_copy).expect("publish offline metallib");

    println!(
        "cargo:rustc-env=METAL_NATIVE_METALLIB={}",
        metallib_out.display()
    );
}

/// Compile `src` → `air` with `-std=<std>` plus shared flags (`-O2`, macosx-min=26).
/// Returns true on success (stdout/stderr discarded so fallback stays quiet).
fn try_metal_compile(metal: &Path, sdk: &str, src: &Path, air: &Path, metal_std: &str) -> bool {
    let std_flag = format!("-std={metal_std}");
    Command::new(metal)
        .args([
            std_flag.as_str(),
            "-O2",
            "-isysroot",
            sdk,
            "-mmacosx-version-min=26.0",
            "-c",
        ])
        .arg(src)
        .arg("-o")
        .arg(air)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ensure_developer_dir() {
    if env::var_os("DEVELOPER_DIR").is_some() {
        return;
    }
    let xcode = Path::new("/Applications/Xcode.app/Contents/Developer");
    if xcode.is_dir() {
        unsafe { env::set_var("DEVELOPER_DIR", xcode) };
    }
}

fn resolve_metal() -> PathBuf {
    if let Ok(p) = xcrun_try(&["-f", "metal"]) {
        return PathBuf::from(p);
    }
    panic!(
        "metal compiler not found. Install Xcode and run:\n  \
         sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n  \
         xcodebuild -downloadComponent MetalToolchain"
    );
}

fn resolve_metallib() -> PathBuf {
    PathBuf::from(xcrun_stdout(&["-f", "metallib"]))
}

fn xcrun_stdout(args: &[&str]) -> String {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("xcrun {:?} failed to spawn: {e}", args));
    if !out.status.success() {
        panic!(
            "xcrun {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn xcrun_try(args: &[&str]) -> Result<String, ()> {
    let out = Command::new("xcrun").args(args).output().map_err(|_| ())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(())
    }
}

fn run(cmd: &mut Command, label: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("{label}: failed to spawn: {e}"));
    if !status.success() {
        panic!("{label}: exited with {status}");
    }
}

/// Emit `rerun-if-changed` for every `.metal` file, not just the directory.
///
/// A bare `cargo:rerun-if-changed=kernels/` tracks the *directory*, whose mtime
/// only moves when a file is created, deleted or renamed — editing a kernel in
/// place does not touch it. The result is a metallib that silently stays stale
/// while `cargo test` reports a pass, which is how a broken kernel can look
/// green. Listing the files individually is the only reliable form.
fn track_kernel_sources(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => panic!("read {} for change tracking: {e}", dir.display()),
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let p = entry.path();
        if p.is_dir() {
            track_kernel_sources(&p);
        } else if p.extension().and_then(|s| s.to_str()) == Some("metal") {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
}
