# gate_bwd profile and the bias-sum fix (2026-10-02, M5 Pro)

The write-up is in `docs/bench-gpu-vs-torch.md`, "gate_bwd: where its time
went, and the bias sum".

| File | What it is |
|---|---|
| `gate-probe-1.md`, `gate-probe-2.md` | `metal_bench 40 gate` before the fix: GPU span of each dispatch of `per_head_sigmoid_gate_backward` at 4096 rows × 12 heads of 64, d_model 768. |
| `ab-dbias/old-*.md`, `new-*.md` | The same probe from the pre-fix binary (serial bias sum) and the post-fix one, interleaved with alternating order, 4 rounds per side (`ab.sh`). |
| `ab-fold/old-*.md`, `new-*.md` | The probe from the bias-sum binary (ten standalone finite checks) and from the one whose gate kernels check their own operands (four standalone checks), interleaved the same way (`slot-fold.sh`). |
| `ab-fold/summary.txt` | The ojas-metal suite, the five mutants of the folded checks (`mutate-fold.sh`), clippy, the build, and the per-round minimums. |
| `tn-sweep/tn-sweep-1.md` | `metal_bench 20 tn` with other jobs running (contended); superseded by `-2`. |
| `tn-sweep/tn-sweep-2.md` | `metal_bench 20 tn` in a quiet slot: per TN shape, the single dispatch, what tessl routes to, and the parallel split-K at widths 128–2048. The first attempt stopped at 96 × 2048 × 4096, whose width 128 needs more scratch than tessl's cap allows, and the bench now prints that row as refused. |
| `ab-tn/old-*.md`, `new-*.md`, `summary.txt` | The gate probe from the fold binary and from the one built on tessl's parallel TN split-K, interleaved with alternating order, 4 rounds per side. `summary.txt` holds the whole verification slot (`slot-tn-verify.sh`): the tessl and ojas-metal suites, clippy, the three `mutate-tn.sh` mutants, and the per-round minimums. The files are named `tn-old-*.md` and `tn-new-*.md`. |
| `ab-stage/old-*.md`, `new-*.md`, `summary.txt` | The gate probe from the parallel-TN binary (bias sum through `simd_shuffle`) and from the one whose bias sum stages rows in threadgroup memory, interleaved with alternating order, 4 rounds per side (`slot-stage.sh`). The files are named `stage-old-*.md` and `stage-new-*.md`. `summary.txt` also holds the ojas-metal suite and a clippy run that stopped in another lane's `ojas-cpu` edit. The mutants in `mutate-stage.sh` were not run (no fork-heavy steps on this Mac). |
| `paired-gate/` | `bench/run_paired.sh 5` on the gate rows only, under outside load (~30); flagged noisy. |
| `paired-gate-quiet/` | The same, re-run at load 8.3–9.1 with the outside agents idle (`slot-paired-gate.sh`). The GPU still read 43–53% busy between rows, and torch's spread (15%) flags `gate_bwd` noisy. |
| `ab-dbias/summary.txt` | The mutation results (`mutate.sh`: each mutant must fail the `gate_dbias_*` tests), clippy and build exit codes, and the per-round minimums. |

To repeat: build `ojas-metal`'s `metal_bench` example in release with
`--features metal`, then run `metal_bench 40 gate`. For the A/B, keep the
old binary and run `ab.sh` with `old` pointing at it.
