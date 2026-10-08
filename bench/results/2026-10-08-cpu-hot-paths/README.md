# CPU and tape hot paths: interleaved A/B (2026-10-08)

Task `gp-cpu-autograd-hot-paths`. **Base** is `86a1096`, the tree before
`7763814` (in-place fan-in, seed scaling, pooled accumulate, hybrid-op
splits, Muon buffer reuse, 256/2048-wide fast paths). **After** is this
tree: `7763814` plus the X·Xᵀ `cblas_ssyrk` product (`ojas_cpu::gemm::gram_out`).
To run the same cases, both binaries were built from their own tree with
this tree's `ojas-cpu/tests/bench_ops.rs` and `ojas-autograd/tests/tape_bench.rs`
copied in. Torch 2.13.0 CPU covers the nanolab rows it has.

Apple M5 Pro, macOS 27.0.1, rustc 1.99.0, release, 6 threads on every lane.
The machine was shared with other builds and tests throughout: 1-minute load
was 17.5–82 during `ab/` and 20–44 during `ab-tape/`. Read every ratio
against that noise. Ratios between 0.9 and 1.1 are not changes.

## Reproduce

```bash
# build each tree's binaries (see run.sh's header), then:
bash run.sh BASE_BIN_DIR AFTER_BIN_DIR 5 OUT_DIR        # ab/
TAPE_ONLY=1 bash run.sh BASE_BIN_DIR AFTER_BIN_DIR 5 OUT_DIR   # ab-tape/
python3 summarize.py OUT_DIR
```

Odd rounds run base, after, torch, and even rounds torch, after, base. Each
cell is a min-of-N inside one process, after two warm-up calls. `summary.md`
gives the minimum over the 5 rounds and the median of the per-round ratios.

## Files

| file | what |
| :-- | :-- |
| `run.sh`, `summarize.py` | the runner and the table builder (`summarize.py` is analysis only) |
| `ab/` | 5 rounds of `bench_ops` (base, after, torch) and `tape_bench`, `load.txt`, and `parity.txt` (torch against the after tree's dump: 19 of 19 outputs ok) |
| `ab-tape/` | 5 rounds of `tape_bench` only. In `ab/`, libtest's `test tape_bench ... ` prefix swallowed the fan-in line; `run.sh` now matches markers anywhere on a line |
| `summary.md` | both tables |
| `bits.txt` | `bench_ops` dump of 15 changed cases at 6 threads, Fast, base against after: 81 files, 0 differ |
| `muon-gate.txt` | Muon no-transpose view gate: as shipped (`nanolab`), off, and open to every tall shape (`tall`, `tallbands`), 3 interleaved rounds through a temporary switch that is not in the tree |
| `probes/syrk.rs`, `probes/syrk.txt` | throwaway probe of X·Xᵀ: one `cblas_sgemm`, the two-band split ojas used, and `cblas_ssyrk` plus mirror, 3 runs of 11 interleaved rounds |
| `probes/spawn.rs`, `spawn.txt` | throwaway probe: one `std::thread::scope` with N−1 empty spawns, min/median of 2000 |

## Results (after / base, min over rounds; median of round ratios in brackets)

- `accumulate_grad`: `[50304,768]` 0.29 (0.28), 18.7 to 5.5 ms; `[768,768]` 0.67 (0.76).
- Loss seed ¼ on the fused LM-head CE backward (`tape_bench`, `ab-tape/`):
  0.17 (0.16), 9.0 to 1.5 ms. Seed 1 is a µs-level no-op on both sides.
- Fan-in (`fanin_8x[1024,768]`, `ab-tape/`): 1.15 (0.80). **Not a measured
  speedup**: min and median disagree under this load.
- conv1d backward `[1,1024,6144]` k4: 0.08 (0.09), 253 to 20 ms. Gated RMSNorm
  backward `[16384,128]`: 0.25 (0.26), 26.3 to 6.6 ms.
- Muon: `[768,768]` 0.72 (0.94), `[2048,768]` 0.75 (1.00), `[3072,768]` 0.77
  (1.60), `[2048,2048]` 0.92 (0.94), `[6144,2048]` 0.72 (0.90). By min each
  shape is faster, but the medians are noise-bound. The X·Xᵀ product alone
  (`probes/syrk.txt`): `ssyrk` plus mirror took 0.46–0.94 of the two-band
  split's time and 0.49–1.14 of one whole `sgemm`, with the same bits as
  that `sgemm` at every probed shape.
- Qwen3.5 shapes: embedding forward `[32768,2048]` 0.62 (0.77), permute
  `[1,1024,8,256]` 0.77 (1.06). Embedding backward 0.98 (1.07) is unchanged code.
- Controls (code unchanged between the trees): add forward 1.05, add backward
  0.98, SiLU backward 1.01, mul backward 1.09, embedding backward 1.11. They
  bound the noise at roughly ±10%.
- Muon gate (`muon-gate.txt`): opening the view to every tall matrix was
  1.2–2.8× slower at 3072×768 and 1.5–2.9× slower at 6144×2048 in all three
  rounds. At 2048×768, gate on and off were within noise. The gate stays.
- Spawn: 34–37 µs minimum, about 70 µs median for 5 spawned threads at load 26.
