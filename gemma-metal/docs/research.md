# Research sources (concise)

Links + one-line takeaway for gemma-metal. Not a literature review.

## Fast Gemma Challenge

- [Bandwidth Is the Whole Game (Tom Campbell)](https://www.linkedin.com/pulse/bandwidth-whole-game-tom-campbell-pqbbe) — Honest SOTA ≈ INT4+MTP; frontier TPS burns PPL budget.
- [gemma-main-bucket artifacts](https://huggingface.co/buckets/gemma-challenge/gemma-main-bucket/tree/artifacts) — Winner stacks (~330 honest / ~510 frontier); Marlin/CUDA graphs do not transfer to Mac.

## Metal 4 / WWDC26

- [Machine learning passes (Apple)](https://developer.apple.com/documentation/metal/machine-learning-passes) — TensorOps matmul / ML encoder surface for Metal 4.
- [WWDC26-330](https://developer.apple.com/videos/play/wwdc2026/330/) — Quantized MTLTensor + cooperative tensors; NAX helps prefill, not M=1 decode.
- [`../tessl/docs/apple-silicon/metal4_mpp.md`](../../../tessl/docs/apple-silicon/metal4_mpp.md) — Local Metal 4 / MPP doctrine for this repo.

## MLX Gemma 4 / PLE

- mlx-lm Gemma 4 PRs (#1093 / #1095 / #1099 / #1103) — Reference order for PLE, p-RoPE, KV-share, `k_eq_v`.
- [`../tessl/docs/apple-silicon/mlx.md`](../../../tessl/docs/apple-silicon/mlx.md) — **PLE must split** for Metal 4GB single-buffer limit; 31B 4bit ~30 tok/s M5 Max → expect ~12–18 M5 Pro without MTP.

## MTP context (BaseRT / LiteRT / Ollama)

- BaseRT — Proprietary `libbaseRT.dylib`; publishes E2B / 26B-A4B (not E4B/31B); MTP + GEMV lessons transferable, kernels not.
- LiteRT-LM — Strong E4B ±MTP Mac baseline when installed; **missing on Phase-0 host**.
- Ollama `gemma4:*-mlx` — Local 31B floor (~11–12 tok/s); MTP knobs not exposed for these tags in Phase 0.

## Core AI (later)

- [Stateful models (coremltools)](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html) — KV as `StateType`; Gemma 4 needs ≥4 states — off v1 critical path.
- [TorchMetalKernel / Core AI](https://apple.github.io/coreai-torch/main/guides/custom-metal-kernels.html) — Package custom FA/GEMV into ML package later, not decode v1.
