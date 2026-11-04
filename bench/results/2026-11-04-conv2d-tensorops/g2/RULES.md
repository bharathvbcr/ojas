# G2 for the Ojas conv route in SpikeTrack: held-out HOTA, first 32 frames (rules, 2026-11-04)

Written before the runs.

**Question:** can `OJAS_METAL_CONV2D_TENSOROPS=f16` (or `f32`) become SpikeTrack's default without
moving tracking accuracy?

**Runs** (`job.sh`): one binary, built from the SpikeTrack tree into `target-convtops` against the
current ojas and tessl trees. The shipped configuration (LoRA v3 + `lorav3_exact_it_3p00.json`,
prompt sperm, `--boxes detector`, cap 64, track's default placements), on the five held-out VISEM
videos (21, 22, 36, 47, 82), first 32 frames each, then `spiketrack eval-visem`. Configs:
- `base`: the env var unset (the im2col route);
- `conv32`: `OJAS_METAL_CONV2D_TENSOROPS=f32`;
- `conv16`: `OJAS_METAL_CONV2D_TENSOROPS=f16`.

**Rules** (SpikeTrack's G2 rules, `tools/analysis/g2_placements.py`):
- `base` must reproduce the documented 32-frame HOTA of 0.6025 to within 1e-4, or every verdict is
  void.
- `conv32` passes if |dHOTA| <= 0.002 and no video moves by more than 0.005.
- `conv16` passes if HOTA >= 0.6005 and no video drops by more than 0.01.
- Each config's run must show the route ran. The frame probe (`../spiketrack_speed.log`) already
  showed changed predictions on video 36, so a byte-identical track set would mean the route did not
  run, and the verdict is void.

**What a pass decides:** G2 only. SpikeTrack changed defaults after G2 and the 1,500-frame G3; a G2
pass makes `conv16` a candidate for G3, not a default.
