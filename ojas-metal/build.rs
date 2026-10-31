//! Compile the crate's Metal kernels into one overlay metallib.
//!
//! Tessl owns GEMM, RMSNorm, AdamW and residual add. This script builds the
//! per-head gate and the kernels in `ojas_backend.metal`, which includes tessl's shared
//! arithmetic headers (AdamW's element math) from `DEP_TESSL_KERNELS`. The Metal
//! invocation matches tessl: `xcrun metal` with an explicit `-isysroot`, not
//! `xcrun -sdk macosx metal`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

const SOURCES: [&str; 6] = [
    "per_head_gate.metal",
    "causal_attn_bwd.metal",
    "ojas_backend.metal",
    "encoder_bwd.metal",
    "max_pool.metal",
    "spatial_bwd.metal",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for name in SOURCES {
        println!("cargo:rerun-if-changed=kernels/{name}");
    }
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "macos" || env::var_os("CARGO_FEATURE_METAL").is_none() {
        println!("cargo:rustc-env=OJAS_GATE_METALLIB=");
        return;
    }

    if env::var_os("DEVELOPER_DIR").is_none() {
        let xcode = std::path::Path::new("/Applications/Xcode.app/Contents/Developer");
        if xcode.is_dir() {
            // SAFETY: build script, single threaded, before any other thread exists.
            unsafe { env::set_var("DEVELOPER_DIR", xcode) };
        }
    }

    // tessl's canonical kernel sources: `tessl/build.rs` emits `cargo:kernels=`
    // and `links = "tessl"` hands it to this script. `ojas_backend.metal`
    // includes the shared arithmetic headers from there (`-I` below).
    let tessl_kernels = env::var("DEP_TESSL_KERNELS").unwrap_or_else(|_| {
        panic!("DEP_TESSL_KERNELS is not set: the `metal` feature needs the tessl dependency")
    });
    for header in SHARED_HEADERS {
        println!("cargo:rerun-if-changed={tessl_kernels}/{header}");
    }
    println!("cargo:rerun-if-env-changed=DEP_TESSL_KERNELS");

    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = xcrun(&["-f", "metal"]);
    let metallib = xcrun(&["-f", "metallib"]);
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out_dir.join("ojas_per_head_gate.metallib");
    let std_flag = "-std=metal4.0";
    let mut airs = Vec::new();
    for name in SOURCES {
        let src = manifest.join("kernels").join(name);
        let air = out_dir.join(name.replace(".metal", ".air"));
        let status = Command::new(&metal)
            .args([
                std_flag,
                "-O2",
                "-ffp-contract=off",
                "-fmetal-math-mode=safe",
                "-I",
                &tessl_kernels,
                "-isysroot",
                &sdk,
                "-mmacosx-version-min=26.0",
                "-c",
            ])
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .status()
            .unwrap_or_else(|e| panic!("metal: failed to spawn: {e}"));
        if !status.success() {
            panic!("{name} failed to compile ({std_flag})");
        }
        airs.push(air);
    }
    let mut link = Command::new(&metallib);
    link.args(&airs).arg("-o").arg(&lib);
    let status = link
        .status()
        .unwrap_or_else(|e| panic!("metallib: failed to spawn: {e}"));
    if !status.success() {
        panic!("ojas-metal metallib link failed");
    }
    println!("cargo:rustc-env=OJAS_GATE_METALLIB={}", lib.display());
}

/// Headers of tessl's kernel directory that `ojas_backend.metal` includes.
const SHARED_HEADERS: [&str; 1] = ["qwen35_adamw_math.h"];

fn xcrun(args: &[&str]) -> String {
    let out = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("xcrun {args:?}: {e}"));
    if !out.status.success() {
        panic!(
            "xcrun {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
