//! CPU phase markers correlated with Instruments.
//!
//! Training [`crate::log::Profiler`] records host-side phase wall times. For
//! GPU / NAX insight, record **Metal System Trace** in Instruments and enable
//! the **Neural Accelerators** utilization counter (do not use legacy
//! `MTLCounterSampleBuffer` — zeros on macOS 26). In-app GPU timestamps:
//! training Metal 4 command buffers fold `MTL4CounterHeap` t0/t1 into the same
//! CB as compute; [`crate::runtime::GpuRuntime::synchronize`] SharedEvent-waits
//! that queue and exposes stamps via `take_metal4_stamps` (DECISIONS M3).
//!
//! Phase names below match Points of Interest labels when you filter Time
//! Profiler samples around `Profiler::enter` call sites in `bin/train.rs`.

/// Instruments-oriented label for a training phase (matches [`crate::log::Phase::as_str`]).
#[inline]
pub fn instruments_label(phase: &'static str) -> &'static str {
    // Keep names stable for Instruments search / POI correlation.
    phase
}
