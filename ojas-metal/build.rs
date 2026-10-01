//! Compile `kernels/per_head_gate.metal` into an overlay metallib.
//!
//! Tessl owns GEMM. This script builds the per-head gate and the tiny-step
//! causal softmax backward. The Metal
//! invocation matches tessl: `xcrun metal` with an explicit `-isysroot`, not
//! `xcrun -sdk macosx metal`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/per_head_gate.metal");
    println!("cargo:rerun-if-changed=kernels/causal_attn_bwd.metal");
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "macos" {
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

    let sdk = xcrun(&["--sdk", "macosx", "--show-sdk-path"]);
    let metal = xcrun(&["-f", "metal"]);
    let metallib = xcrun(&["-f", "metallib"]);
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib = out_dir.join("ojas_per_head_gate.metallib");
    let std_flag = "-std=metal4.0";
    let mut airs = Vec::new();
    for name in ["per_head_gate.metal", "causal_attn_bwd.metal"] {
        let src = manifest.join("kernels").join(name);
        let air = out_dir.join(name.replace(".metal", ".air"));
        let status = Command::new(&metal)
            .args([
                std_flag,
                "-O2",
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
