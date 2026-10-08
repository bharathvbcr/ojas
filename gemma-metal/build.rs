//! AOT-compile `kernels/*.metal` → an immutable gemma-metal metallib in `OUT_DIR`.
//!
//! `GEMMA_METAL_SKIP_AOT=1` is an explicit offline escape hatch. It requires
//! `GEMMA_METAL_PREBUILT_METALLIB` to name an existing absolute path; this
//! script never treats a mutable or ignored source-root artifact as canonical.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=GEMMA_METAL_SKIP_AOT");
    println!("cargo:rerun-if-env-changed=GEMMA_METAL_PREBUILT_METALLIB");
    println!("cargo:rerun-if-env-changed=DEP_TESSL_KERNELS");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("kernels");
    let configured_tessl_kernels = env::var_os("DEP_TESSL_KERNELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!(
            "DEP_TESSL_KERNELS not set — tessl must remain a direct dependency with links = \"tessl\""
        ));
    let tessl_kernels = configured_tessl_kernels.canonicalize().unwrap_or_else(|e| {
        panic!(
            "DEP_TESSL_KERNELS={} is not accessible: {e}",
            configured_tessl_kernels.display()
        )
    });
    if !tessl_kernels.is_dir() {
        panic!(
            "DEP_TESSL_KERNELS={} is not a directory",
            tessl_kernels.display()
        );
    }
    let shared_gelu = tessl_kernels.join("gelu.h");
    if !shared_gelu.is_file() {
        panic!(
            "required Tessl shared GELU header missing: {}",
            shared_gelu.display()
        );
    }
    // This is outside gemma-metal's own `kernels/` tree, so Cargo cannot infer
    // the transitive include dependency from the source file.
    println!("cargo:rerun-if-changed={}", shared_gelu.display());

    if env::var_os("GEMMA_METAL_SKIP_AOT").is_some() {
        println!("cargo:warning=GEMMA_METAL_SKIP_AOT set; skipping gemma metallib AOT");
        let configured = env::var_os("GEMMA_METAL_PREBUILT_METALLIB").unwrap_or_else(|| {
            panic!(
                "GEMMA_METAL_SKIP_AOT is set but GEMMA_METAL_PREBUILT_METALLIB is not. \
                 Name an existing absolute metallib path explicitly, or unset \
                 GEMMA_METAL_SKIP_AOT and build the shaders"
            )
        });
        let configured = PathBuf::from(configured);
        if !configured.is_absolute() {
            panic!(
                "GEMMA_METAL_PREBUILT_METALLIB must be an absolute path, got {}",
                configured.display()
            );
        }
        let prebuilt = configured.canonicalize().unwrap_or_else(|e| {
            panic!(
                "GEMMA_METAL_PREBUILT_METALLIB={} is not accessible: {e}",
                configured.display()
            )
        });
        if !prebuilt.is_file() {
            panic!(
                "GEMMA_METAL_PREBUILT_METALLIB={} is not a file",
                prebuilt.display()
            );
        }
        println!("cargo:rerun-if-changed={}", prebuilt.display());
        println!(
            "cargo:rustc-env=GEMMA_METAL_METALLIB={}",
            prebuilt.display()
        );
        return;
    }

    ensure_developer_dir();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if !kernels_dir.is_dir() {
        println!("cargo:warning=no kernels/ dir; skipping metallib");
        println!("cargo:rustc-env=GEMMA_METAL_METALLIB=");
        return;
    }

    let sdk = xcrun_stdout(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = resolve_metal();
    let metallib = PathBuf::from(xcrun_stdout(&["-f", "metallib"]));

    let mut sources: Vec<PathBuf> = fs::read_dir(&kernels_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("metal"))
        .collect();
    sources.sort();

    let mut air_files = Vec::new();
    for src in &sources {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let air = out_dir.join(format!("{stem}.air"));
        let ok = try_compile(&metal, &sdk, &tessl_kernels, src, &air, "metal4.0")
            || try_compile(&metal, &sdk, &tessl_kernels, src, &air, "metal3.2");
        if !ok {
            panic!("failed to compile {}", src.display());
        }
        air_files.push(air);
    }

    if air_files.is_empty() {
        println!("cargo:rustc-env=GEMMA_METAL_METALLIB=");
        return;
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
    let status = link.status().expect("metallib spawn");
    if !status.success() {
        panic!("metallib link failed");
    }
    println!(
        "cargo:rustc-env=GEMMA_METAL_METALLIB={}",
        metallib_out.display()
    );
}

fn try_compile(
    metal: &Path,
    sdk: &str,
    include_dir: &Path,
    src: &Path,
    air: &Path,
    metal_std: &str,
) -> bool {
    let std_flag = format!("-std={metal_std}");
    let mut command = Command::new(metal);
    command
        .args([std_flag.as_str(), "-O2", "-isysroot", sdk])
        .arg("-I")
        .arg(include_dir)
        .args(["-mmacosx-version-min=26.0", "-c"])
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
    PathBuf::from(xcrun_stdout(&["-f", "metal"]))
}

fn xcrun_stdout(args: &[&str]) -> String {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("xcrun {:?} spawn: {e}", args));
    if !out.status.success() {
        panic!(
            "xcrun {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
