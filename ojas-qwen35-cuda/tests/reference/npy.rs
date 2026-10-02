//! The `.npy` reader moved to the library, `ojas_qwen35_cuda::npy` (L-cuda-M1,
//! lead's ruling 2026-10-01): `runga` and the tiny-fixture loader read `.npy`
//! outside the test harness. Re-exported here so every `reference::npy` user
//! is unchanged.

pub use ojas_qwen35_cuda::npy::*;
