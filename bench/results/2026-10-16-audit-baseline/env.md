# Environment

Measured 2026-10-26 evening local time, worktree `/Users/bharath/Code/research/ojas-w19`, branch `gp/w19-audit-baseline`.

- git SHA of the worktree HEAD: `105e20343c6c8a30dfd01356aa689944f76a8dc1`
- `git diff 105e203 d5d33cc -- ojas-cpu ojas-capi ojas-cuda ojas-core/src/tensor.rs` was empty. Current main `d5d33ccbcab9c51c003cda175c666d03ee447ec4` matches the measured kernel sources in those trees. The 11 commits on main after `105e203` are Metal-only (rope, D=64 attention routing, conv, relu, bilinear, SAM). The worktree was not rebased.
- Harness dirty hash (sha256 of the tracked diff of `generate.rs`, `lib.rs`, and `bench_ops.rs`, then `ojas-capi/benches/generate_pressure.rs`, then `ojas-cuda/tests/nvrtc_cache_key_bench.rs`): `b92b3fb70b786679598bb484eb3e2fa10c79223058a9086f5e2bbc09d9340373`
- Tracked diff alone: `9028cc828eeaf9aa4d7a051ad920e4635baba378d5f2f413ced4b46f70df5057`
- `generate_pressure.rs`: `0acec5d3134dee9d5dd5afa8374a92f032af4c2ad78594cda6c85f0fba9385f4`
- `nvrtc_cache_key_bench.rs`: `5848ab71c46d8bdddc2585ff881806b6f0ceede45fabb771b72f7f0f9627e579`
- The results files are not in that hash. `Cargo.lock` is not in it either. Unlocked cargo had rewritten gusset `0.0.2` to `0.0.3` in the worktree lock. That lock was reverted with `git checkout -- Cargo.lock` before this file was written. `git diff --stat -- Cargo.lock` was empty afterward.
- rustc: `1.99.0 (b940084d7 2026-09-28)`
- hardware: Mac17,8, Apple M5 Pro, 18 cores, 68719476736 bytes RAM
- profile: release, `CARGO_PROFILE_RELEASE_DEBUG=0`. The timed processes were the already-built test binaries, not a fresh `cargo` compile.
- CPU thread list: `OJAS_BENCH_THREADS=1,18`. One case-group per process, so a later A/B can run one `OJAS_BENCH_OPS` name per process and interleave.
- numerics: the `numerics=` field on each `OJAS_OP` line. CPU warmup 2. capi warmup 20, n=200. CUDA warmup 2, n=200.
- `mac_heavy.sh` was not used.

## Load at the start of each group

Two samples under the gate, then this one was the start. `quiet=true` for every row in the group. `yes=0` for every start.

| Group | load1 | load5 | quiet |
|---|---:|---:|---|
| cuda | 10.35 | 16.69 | true |
| capi | 9.81 | 15.95 | true |
| cpu-g1 rms, gemv, NaN scans, cached 8k | 9.04 | 15.20 | true |
| cpu-g2 conv, gated RMS, muon 768, cached 32k | 11.21 | 14.45 | true |
| cpu-g3a fused CE V=50304 | 10.23 | 13.75 | true |
| cpu-g3b fused CE V=248320 | 8.76 | 12.92 | true |
| cpu-g3c muon 6144×2048 | 9.41 | 12.89 | true |
| cpu-g3d GDN T=2048 | 10.88 | 12.43 | true |

Each group log also has a `LOAD1=` line. Those logs are in the worktree `target/w19-rows/`, which is gitignored. `rows.txt` is the copy that belongs with this directory.
