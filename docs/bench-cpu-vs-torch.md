# CPU reference versus torch CPU

Wall time of one forward, backward, and AdamW step. The graph is the regression test in `ojas-cpu/tests/torch_ref.rs`: RMSNorm (eps `1e-6`), bias-free linear `y = x @ W.T`, half-split RoPE, causal SDPA with scale `1/sqrt(d)` and one head, mean cross-entropy, backward, then AdamW on `wq` (weight decay 0.1) and the norm weight (weight decay 0). The learning rate is `6e-4` times the cosine multiplier at step 8 (warmup 4, total 20). `wk`, `wv`, and `wo` receive weight gradients and are not written by AdamW. Moments start at zero.

Both timers use CPU. Torch uses `SDPBackend.MATH`. The tensors in the torch run were on `cpu`. This torch build also has MPS compiled in; the timed step did not move tensors there.

Numbers below are **verified** from the commands in this file, on this machine, 2026-10-01. Two full runs were recorded. The within-run median is the median after dropping the warmup calls listed below. The second process did not repeat the first median closely on the tiny torch step, so both runs are listed and no single speedup is stated.

## Toolchain

| | Verified value |
| :--- | :--- |
| rustc | `rustc 1.98.0 (88d9e12ae 2026-08-18)` |
| cargo | `cargo 1.98.0 (797e8a9bc 2026-08-05)` |
| Python | 3.14.7, `/opt/homebrew/opt/python@3.14/bin/python3.14` |
| torch | 2.13.0, `/Users/bharath/Library/Python/3.14/lib/python/site-packages/torch/__init__.py` |
| torch CPU threads | `torch.get_num_threads() == 6`, interop threads 18 |
| Rust threads | the CPU reference has no thread pool |

Release profile from the workspace `Cargo.toml`: `panic = "unwind"`, `overflow-checks = true`.

## Commands

```text
cd /Users/bharath/Code/research/ojas
cargo test -p ojas-cpu --release --test torch_ref -- --nocapture --test-threads=1
python3 /tmp/ojas_cpu_bench.py
```

The second invocation repeated `one_step_wall_time` only, then the same python script. `ojas-cpu` already built; criterion was not added.

The Rust clock wraps `tiny_step` / `regression_graph`, which builds every tensor from host slices, runs the step, and copies the loss and the updated `wq` and norm back to host vectors. `std::hint::black_box` is applied to that loss and those two vectors.

The torch clock wraps the forward, backward, and `AdamW.step`. Gradients are cleared inside that region. Weights are copied back and the AdamW state is cleared before the clock starts, so each call is one step from the initial parameters.

## Correctness

The tiny Rust loss on the frozen `torch_ref` inputs stayed within `1e-4` of the frozen `LOSS`. The wall-time test printed `loss=3.4650407` and `loss_err=2.384e-7`. `one_step_matches_torch_2_13_float32` also passed, which checks that same loss and the other frozen tensors at `1e-4`.

The torch tiny loss, seed 0, matched the frozen bits before timing: `0x405dc33b`, absolute error `0`.

The larger shape has no frozen tensor file. The Rust run checked that its loss was finite (`4.852497` with the test's own generator). That value is not a torch match.

## Medians

Tiny: `B=1`, `T=4`, `d=16`, 1 head, `vocab=32`. 200 calls, drop the first 20, median of 180.

Larger: `B=2`, `T=32`, `d=64`, 1 head, `vocab=128`. 20 calls, drop the first 4, median of 16. This shape finished in well under a second per run, so the iteration count was left at 20.

| Run | Shape | Rust median | Torch CPU median |
| :--- | :--- | ---: | ---: |
| 1 | tiny | `4.572900000e-5` s | `1.755520498e-3` s |
| 1 | larger | `7.833208000e-3` s | `1.923291500e-3` s |
| 2 | tiny | `5.968750000e-5` s | `9.818130056e-4` s |
| 2 | larger | `8.672417000e-3` s | `1.707812495e-3` s |

Readable scale, same numbers: Rust tiny about 46 µs then 60 µs; torch tiny about 1.76 ms then 0.98 ms. Rust larger about 7.83 ms then 8.67 ms; torch larger about 1.92 ms then 1.71 ms.
