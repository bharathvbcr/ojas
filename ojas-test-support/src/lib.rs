//! Dev-only helpers shared by ojas test binaries.
//!
//! Use it only as a `[dev-dependencies]` path dependency. It is a leaf: it
//! names no ojas type, so any crate's tests can depend on it without a cycle.
//! A helper that needs an ojas type does not belong here.
//!
//! - [`alloc`] holds counting global allocators. A test binary still declares
//!   its own `#[global_allocator] static`, since the allocator is per binary.
//! - [`rng`] holds splitmix64.
//! - [`vals`] holds seeded `Vec` generators (one xorshift64 stream per call).

#![deny(unsafe_code)]

pub mod alloc;
pub mod rng;
pub mod vals;

pub use alloc::{thread_allocs, PeakAlloc, ThreadAllocCount};
pub use rng::{splitmix64_next, SplitMix64};
