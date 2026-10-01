import os, glob

files = [
    "ojas-capi/src/lib.rs", "ojas-capi/src/engine.rs", "ojas-capi/src/session.rs", 
    "ojas-capi/src/load.rs", "ojas-capi/src/step.rs", "ojas-capi/src/generate.rs", "ojas-capi/src/tests.rs",
    "ojas-gusset-engine/src/lib.rs",
    "ojas-metal/src/lib.rs", "ojas-metal/src/gpu.rs", "ojas-metal/build.rs",
    "ojas-wgpu/src/lib.rs",
    "ojas-cuda/src/lib.rs",
    "ojas-hip/src/lib.rs",
    "ojas-infer/src/lib.rs", "ojas-infer/src/gpt.rs", "ojas-infer/src/kernels.rs",
    "go/api.go", "go/ffi.go", "go/api_test.go", "go/go.mod", "go/README.md",
    "ojas-engine/src/lib.rs"
]

for f in files:
    if os.path.exists(f):
        print(f"--- {f} ---")
        lines = open(f).readlines()
        for i, line in enumerate(lines):
            if any(x in line for x in ["unsafe", "TODO", "FIXME", "HACK", "catch_unwind", "extern \"C\"", "ffi", "ptr", "alloc", "free", "panic"]):
                print(f"{i+1}: {line.strip()}")
